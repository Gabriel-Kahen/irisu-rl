from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from irisu_pointer.action import PointerActionSpec
from irisu_pointer.geometry_learning import GeometryDataset, GeometryExample
from irisu_pointer.sequence_replay import (
    SEQUENCE_REPLAY_EVENT_FEATURES,
    SequenceReplayOutput,
    SequenceReplayTargets,
)
from irisu_pointer.sequence_replay_campaign import (
    LEGACY_GEOMETRY_COLLECTION_FORMAT,
    batch_legacy_geometry_episodes,
    batch_normalized_replay_episode,
    deterministic_tbptt_windows,
    load_legacy_geometry_collection,
    merge_offline_metrics,
    normalize_trusted_replay_examples,
    offline_sequence_metrics,
)
from irisu_pointer.steering import SteeringIntent
from irisu_pointer.steering_learning import SteeringExample
from irisu_rl.encoding import EncodedBatch
from irisu_rl.schema import TEACHER_V1


def _encoded(tick: int, *, score: float = 0.0, gauge: float = 0.5) -> EncodedBatch:
    globals_out = np.zeros(
        (1, len(TEACHER_V1.global_features)), dtype=np.float32
    )
    global_index = {
        name: index for index, name in enumerate(TEACHER_V1.global_features)
    }
    globals_out[0, global_index["tick_scaled"]] = tick / 100_000.0
    globals_out[0, global_index["score_signed_log1p"]] = score
    globals_out[0, global_index["gauge_fraction"]] = gauge
    globals_out[0, global_index["highest_chain_log1p"]] = score / 2.0
    globals_out[0, global_index["qualifying_clears_log1p"]] = score / 4.0
    bodies = np.zeros(
        (1, TEACHER_V1.capacity, len(TEACHER_V1.body_features)),
        dtype=np.float32,
    )
    mask = np.zeros((1, TEACHER_V1.capacity), dtype=np.bool_)
    mask[0, :2] = True
    body_index = {
        name: index for index, name in enumerate(TEACHER_V1.body_features)
    }
    bodies[0, :2, body_index["kind_piece"]] = 1.0
    bodies[0, :2, body_index["color_0"]] = 1.0
    bodies[0, :2, body_index["lifecycle_falling"]] = 1.0
    bodies[0, 0, body_index["id_scaled"]] = 0.1
    bodies[0, 1, body_index["id_scaled"]] = 0.2
    return EncodedBatch(
        globals_out,
        bodies,
        mask,
        np.array([tick], dtype=np.uint64),
        np.zeros(1, dtype=np.uint32),
        TEACHER_V1,
    )


def _steering(
    tick: int,
    ordinal: int,
    replay_sha256: str,
    collection_sha256: str,
    *,
    score: float = 0.0,
    gauge: float = 0.5,
) -> SteeringExample:
    intent = tuple(SteeringIntent).index(SteeringIntent.STEER_MATCH)
    return SteeringExample(
        f"replay:{replay_sha256}:{ordinal}",
        collection_sha256,
        _encoded(tick, score=score, gauge=gauge),
        0,
        1,
        1,
        0,
        intent,
        PointerActionSpec(),
    )


def _geometry(
    seed: int,
    tick: int,
    shot: int,
    candidate: int,
    improved: bool,
    candidate_sha256: str,
) -> GeometryExample:
    return GeometryExample(
        f"r3e-oracle-visited:{seed:08x}:{tick}:{shot}",
        hashlib.sha256(f"{seed}:{tick}:{shot}".encode()).hexdigest(),
        candidate_sha256,
        _encoded(tick),
        0,
        1,
        candidate,
        3,
        improved,
    )


def _geometry_payload(
    dataset: GeometryDataset,
    *,
    base_sha256: str,
    source_sha256: str,
    runtime_sha256: str,
    teacher_sha256: str,
    seeds: list[int],
) -> dict[str, object]:
    examples = []
    for example in dataset:
        observation = example.observation
        examples.append(
            {
                "episode_identity": example.episode_identity,
                "provenance_sha256": example.provenance_sha256,
                "candidate_set_sha256": example.candidate_set_sha256,
                "source_index": example.source_index,
                "destination_index": example.destination_index,
                "candidate_index": example.candidate_index,
                "candidate_count": example.candidate_count,
                "improved_over_incumbent": example.improved_over_incumbent,
                "global_features": torch.from_numpy(
                    observation.global_features.copy()
                ),
                "body_features": torch.from_numpy(
                    observation.body_features.copy()
                ),
                "body_mask": torch.from_numpy(observation.body_mask.copy()),
                "source_tick": torch.from_numpy(observation.source_tick.copy()),
                "health_flags": torch.from_numpy(
                    observation.health_flags.copy()
                ),
            }
        )
    return {
        "format": LEGACY_GEOMETRY_COLLECTION_FORMAT,
        "schema_sha256": TEACHER_V1.sha256,
        "candidate_set_sha256": dataset.candidate_set_sha256,
        "candidate_count": dataset.candidate_count,
        "dataset_sha256": dataset.sha256,
        "metadata": {
            "development_only": True,
            "canonical_r3_evidence": False,
            "sealed_test_material_used": False,
            "learner": "winner_classifier",
            "base_policy_sha256": base_sha256,
            "source_identity_sha256": source_sha256,
            "runtime_sha256": runtime_sha256,
            "teacher_sha256": teacher_sha256,
            "candidate_vocabulary_sha256": dataset.candidate_set_sha256,
            "seeds": seeds,
        },
        "examples": examples,
    }


class SequenceReplayCampaignTests(unittest.TestCase):
    def test_replay_normalization_is_chronological_and_causal(self) -> None:
        replay = "1" * 64
        collection = "2" * 64
        base = "3" * 64
        supplied = (
            _steering(30, 2, replay, collection, score=0.3, gauge=0.7),
            _steering(10, 0, replay, collection, score=0.1, gauge=0.5),
            _steering(20, 1, replay, collection, score=0.2, gauge=0.6),
        )
        normalized = normalize_trusted_replay_examples(
            supplied,
            expected_replay_sha256=replay,
            expected_collection_sha256=collection,
            base_checkpoint_sha256=base,
        )
        self.assertEqual(normalized.source_ticks, (10, 20, 30))
        self.assertEqual(
            {value.episode_identity for value in normalized.examples},
            {normalized.identity},
        )
        event_index = {
            name: index
            for index, name in enumerate(SEQUENCE_REPLAY_EVENT_FEATURES)
        }
        self.assertTrue(torch.equal(
            normalized.event_features[0],
            torch.zeros_like(normalized.event_features[0]),
        ))
        self.assertEqual(
            float(
                normalized.event_features[
                    1, event_index["previous_action_shot"]
                ]
            ),
            1.0,
        )
        self.assertAlmostEqual(
            float(normalized.event_features[1, event_index["delta_gauge"]]),
            0.1,
            places=6,
        )

        changed_future = (
            supplied[1],
            supplied[2],
            _steering(30, 2, replay, collection, score=9.0, gauge=0.1),
        )
        changed = normalize_trusted_replay_examples(
            changed_future,
            expected_replay_sha256=replay,
            expected_collection_sha256=collection,
            base_checkpoint_sha256=base,
        )
        self.assertTrue(torch.equal(
            normalized.event_features[:2], changed.event_features[:2]
        ))

        renamed = []
        id_index = TEACHER_V1.body_features.index("id_scaled")
        chain_index = TEACHER_V1.body_features.index("chain_id_scaled")
        for example in supplied:
            observation = example.observation.copy()
            observation.body_features[..., id_index] = 0.9
            observation.body_features[..., chain_index] = 0.8
            renamed.append(replace(example, observation=observation))
        renamed_events = normalize_trusted_replay_examples(
            renamed,
            expected_replay_sha256=replay,
            expected_collection_sha256=collection,
            base_checkpoint_sha256=base,
        ).event_features
        self.assertTrue(torch.equal(normalized.event_features, renamed_events))

        bound = batch_normalized_replay_episode(normalized)
        self.assertTrue(torch.allclose(
            bound.batch.targets.pair_weight,
            torch.full((3, 1), 0.25),
        ))
        self.assertEqual(bound.base_checkpoint_sha256, base)

    def test_legacy_geometry_load_group_and_batch_are_identity_bound(self) -> None:
        candidate = "4" * 64
        base = "3" * 64
        source = "5" * 64
        runtime = "6" * 64
        teacher = "7" * 64
        records = (
            _geometry(0x22, 40, 2, 2, True, candidate),
            _geometry(0x11, 30, 2, 0, False, candidate),
            _geometry(0x22, 20, 1, 1, False, candidate),
            _geometry(0x11, 10, 1, 1, True, candidate),
        )
        dataset = GeometryDataset(records)
        payload = _geometry_payload(
            dataset,
            base_sha256=base,
            source_sha256=source,
            runtime_sha256=runtime,
            teacher_sha256=teacher,
            seeds=[0x11, 0x22],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.pt"
            torch.save(payload, path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = load_legacy_geometry_collection(
                path,
                expected_sha256=digest,
                expected_base_checkpoint_sha256=base,
                expected_candidate_set_sha256=candidate,
                expected_source_identity_sha256=source,
                expected_runtime_sha256=runtime,
                expected_teacher_sha256=teacher,
            )
            with self.assertRaises(ValueError):
                load_legacy_geometry_collection(
                    path,
                    expected_sha256="0" * 64,
                    expected_base_checkpoint_sha256=base,
                    expected_candidate_set_sha256=candidate,
                )
            self.assertEqual(
                [value.seed for value in loaded.episodes], [0x11, 0x22]
            )
            self.assertEqual(loaded.episodes[0].source_ticks, (10, 30))
            with self.assertRaises(ValueError):
                load_legacy_geometry_collection(
                    path,
                    expected_sha256=digest,
                    expected_base_checkpoint_sha256="9" * 64,
                    expected_candidate_set_sha256=candidate,
                )

        logits = {
            episode.seed: torch.zeros(len(episode.records), 3)
            for episode in loaded.episodes
        }
        bound = batch_legacy_geometry_episodes(
            loaded,
            base_geometry_logits=logits,
            geometry_checkpoint_sha256="8" * 64,
        )
        self.assertEqual(bound.batch.global_features.shape[:2], (2, 2))
        self.assertTrue(torch.equal(
            bound.batch.targets.policy_weight,
            torch.zeros_like(bound.batch.targets.policy_weight),
        ))
        self.assertEqual(
            bound.batch.targets.geometry_index[:, 0].tolist(), [1, 0]
        )
        self.assertEqual(
            bound.batch.targets.geometry_apply_target[:, 0].tolist(),
            [1.0, 0.0],
        )
        self.assertTrue(bound.batch.geometry_candidate_mask.all())
        self.assertEqual(bound.geometry_checkpoint_sha256, "8" * 64)

    def test_tbptt_windows_are_deterministic_and_mask_burn_in(self) -> None:
        replay = "1" * 64
        collection = "2" * 64
        base = "3" * 64
        examples = tuple(
            _steering(
                10 * (index + 1),
                index,
                replay,
                collection,
                score=index / 10,
            )
            for index in range(5)
        )
        episode = normalize_trusted_replay_examples(
            examples,
            expected_replay_sha256=replay,
            expected_collection_sha256=collection,
            base_checkpoint_sha256=base,
        )
        bound = batch_normalized_replay_episode(episode)
        first = deterministic_tbptt_windows(
            bound, burn_in_steps=1, unroll_steps=2, seed=17, epoch=3
        )
        second = deterministic_tbptt_windows(
            bound, burn_in_steps=1, unroll_steps=2, seed=17, epoch=3
        )
        self.assertEqual(
            [value.sha256 for value in first],
            [value.sha256 for value in second],
        )
        self.assertEqual({value.train_start for value in first}, {0, 2, 4})
        for window in first:
            self.assertTrue(bool(window.batch.reset_before[0, 0]))
            if window.burn_in_steps:
                self.assertEqual(
                    float(window.batch.targets.policy_weight[0, 0]), 0.0
                )
            self.assertGreater(
                float(window.batch.targets.policy_weight[-1, 0]), 0.0
            )
        limited = deterministic_tbptt_windows(
            bound,
            burn_in_steps=1,
            unroll_steps=2,
            seed=17,
            epoch=3,
            maximum_windows=2,
        )
        self.assertEqual(len(limited), 2)

    def test_offline_metrics_are_mergeable_and_not_rollout_evidence(self) -> None:
        time, batch, bodies = 2, 1, 2
        act = torch.tensor([[[0.0, 3.0]], [[3.0, 0.0]]])
        wait = torch.zeros(time, batch, 5)
        wait[1, 0, 2] = 3.0
        pair = torch.zeros(time, batch, bodies, bodies)
        pair[0, 0, 0, 1] = 3.0
        kind = torch.zeros(time, batch, bodies, bodies, 2)
        template = torch.zeros(time, batch, bodies, bodies, 3)
        intent = torch.zeros(time, batch, bodies, bodies, 7)
        kind[0, 0, 0, 1, 1] = 3.0
        template[0, 0, 0, 1, 2] = 3.0
        intent[0, 0, 0, 1, 1] = 3.0
        geometry = torch.tensor([[[0.0, 3.0]], [[3.0, 0.0]]])
        output = SequenceReplayOutput(
            act,
            wait,
            pair,
            kind,
            template,
            intent,
            torch.ones(time, batch, bodies, bodies, dtype=torch.bool),
            geometry,
            torch.zeros(time, batch),
            torch.tensor([[True], [False]]),
            torch.tensor(
                [[[0.0, 1.0, 2.0]], [[1.0, 2.0, 3.0]]]
            ),
            torch.tensor([[[1.0, -1.0]], [[-1.0, 1.0]]]),
            torch.tensor([[[0.5, 1.0]], [[0.0, 2.0]]]),
            torch.zeros(batch, 4),
            torch.tensor(0.0),
        )
        valid = torch.ones(time, batch, dtype=torch.bool)
        targets = SequenceReplayTargets(
            valid,
            torch.tensor([[1], [0]]),
            torch.tensor([[0], [2]]),
            torch.tensor([[0], [0]]),
            torch.tensor([[1], [0]]),
            torch.tensor([[1], [0]]),
            torch.tensor([[2], [0]]),
            torch.tensor([[1], [0]]),
            pair_weight=torch.tensor([[1.0], [0.0]]),
            geometry_index=torch.tensor([[1], [0]]),
            geometry_weight=torch.ones(time, batch),
            geometry_apply_target=torch.tensor([[1.0], [0.0]]),
            return_target=torch.tensor([[1.0], [2.0]]),
            viability_target=torch.tensor(
                [[[1.0, 0.0]], [[0.0, 1.0]]]
            ),
            outcome_target=torch.tensor(
                [[[0.5, 1.0]], [[0.0, 2.0]]]
            ),
        )
        report = offline_sequence_metrics(output, targets)
        self.assertEqual(
            report["warning"], "offline diagnostics are not rollout evidence"
        )
        self.assertEqual(
            report["metrics"]["act_accuracy"]["value"], 1.0
        )
        self.assertEqual(
            report["metrics"]["outcome_absolute_error"]["value"], 0.0
        )
        merged = merge_offline_metrics((report, report))
        self.assertEqual(merged["reports"], 2)
        self.assertEqual(
            merged["metrics"]["act_accuracy"]["value"], 1.0
        )
        self.assertEqual(
            merged["metrics"]["act_accuracy"]["count"], 4
        )


if __name__ == "__main__":
    unittest.main()
