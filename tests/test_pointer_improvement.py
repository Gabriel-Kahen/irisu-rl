from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from irisu_env import IrisuEnv, NativeError, find_library
from irisu_pointer.action import PointerActionTensor
from irisu_pointer.checkpoint import (
    load_pointer_checkpoint,
    load_teacher_pointer_policy,
    save_pointer_checkpoint,
)
from irisu_pointer.experts import PointerExpertDecision
from irisu_pointer.improvement import (
    DaggerCollectionConfig,
    DaggerLoopConfig,
    DaggerPolicyImprover,
    DistributionalLeafEvaluator,
    advantage_weighted_behavior,
    collect_dagger_episode,
    file_sha256,
    stratified_replay_minibatches,
)
from irisu_pointer.model import EntityPointerActorCritic, PointerModelConfig
from irisu_pointer.sequence import (
    PointerSequenceConfig,
    PointerSequenceEpisode,
    PointerSequenceTrainer,
)
from irisu_rl.schema import TEACHER_V1


def _model() -> EntityPointerActorCritic:
    torch.manual_seed(31)
    return EntityPointerActorCritic(
        TEACHER_V1,
        config=PointerModelConfig(
            global_hidden=12,
            body_hidden=12,
            attention_hidden=24,
            attention_heads=4,
            attention_layers=1,
            feedforward_hidden=48,
            recurrent_hidden=20,
        ),
    )


def _wait_teacher(_observation, _spec) -> PointerExpertDecision:
    return PointerExpertDecision.wait(2)


def _sequence(identity: str, weight: float) -> PointerSequenceEpisode:
    return PointerSequenceEpisode(
        identity=identity,
        global_features=torch.zeros((1, len(TEACHER_V1.global_features))),
        body_features=torch.zeros((1, 1, len(TEACHER_V1.body_features))),
        body_mask=torch.ones((1, 1), dtype=torch.bool),
        actions=PointerActionTensor(
            kind=torch.zeros(1, dtype=torch.long),
            wait_index=torch.zeros(1, dtype=torch.long),
            target_index=torch.zeros(1, dtype=torch.long),
            template_index=torch.zeros(1, dtype=torch.long),
        ),
        returns=torch.zeros(1),
        schema=TEACHER_V1,
        policy_weight=torch.full((1,), weight),
    )


class ReplayMinibatchTests(unittest.TestCase):
    @staticmethod
    def _key(value: PointerSequenceEpisode) -> tuple[str, float]:
        assert value.policy_weight is not None
        return value.identity, float(value.policy_weight[0])

    def test_each_minibatch_interleaves_supervision_and_awr_behavior(self) -> None:
        supervision = tuple(_sequence(f"episode-{index}", 1.0) for index in range(8))
        behavior = tuple(_sequence(f"episode-{index}", 0.5) for index in range(8))
        first = stratified_replay_minibatches(
            supervision, behavior, updates=2, batch_size=4
        )
        second = stratified_replay_minibatches(
            supervision, behavior, updates=2, batch_size=4
        )
        self.assertEqual(
            tuple(tuple(map(self._key, batch)) for batch in first),
            tuple(tuple(map(self._key, batch)) for batch in second),
        )
        for batch in first:
            self.assertEqual(
                [self._key(value)[1] for value in batch],
                [1.0, 0.5, 1.0, 0.5],
            )

    def test_limited_updates_spread_both_pools_through_late_replay(self) -> None:
        supervision = tuple(
            _sequence(f"episode-{index:02d}", 1.0) for index in range(64)
        )
        behavior = tuple(
            _sequence(f"episode-{index:02d}", 0.5) for index in range(64)
        )
        flattened = tuple(
            value
            for batch in stratified_replay_minibatches(
                supervision, behavior, updates=8, batch_size=4
            )
            for value in batch
        )
        teacher_ids = {
            value.identity
            for value in flattened
            if self._key(value)[1] == 1.0
        }
        behavior_ids = {
            value.identity
            for value in flattened
            if self._key(value)[1] == 0.5
        }
        self.assertEqual(len(teacher_ids), 16)
        self.assertEqual(len(behavior_ids), 16)
        self.assertIn("episode-00", teacher_ids)
        self.assertIn("episode-63", teacher_ids)
        self.assertIn("episode-00", behavior_ids)
        self.assertIn("episode-63", behavior_ids)


class PointerCheckpointTests(unittest.TestCase):
    def test_round_trip_binds_file_and_model_identities(self) -> None:
        with tempfile.TemporaryDirectory(prefix="irisu-dev-checkpoint-") as directory:
            path = Path(directory) / "candidate.pt"
            model = _model()
            digest = save_pointer_checkpoint(
                path, model, metadata={"iteration": 3}
            )
            loaded = load_pointer_checkpoint(path, expected_sha256=digest)
            self.assertEqual(loaded.sha256, digest)
            self.assertEqual(loaded.metadata, {"iteration": 3})
            self.assertEqual(loaded.model.manifest(), model.manifest())
            policy = load_teacher_pointer_policy(
                path, expected_sha256=digest
            )
            self.assertEqual(policy.artifact_sha256, digest)
            self.assertEqual(policy.schema_sha256, TEACHER_V1.sha256)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_pointer_checkpoint(path, expected_sha256="0" * 64)


class DaggerCollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.runtime = Path(find_library()).resolve()
        except NativeError as exc:
            raise unittest.SkipTest("portable runtime is unavailable") from exc
        cls.runtime_sha256 = file_sha256(cls.runtime)

    def setUp(self) -> None:
        torch.set_num_threads(1)

    def _env(self) -> IrisuEnv:
        return IrisuEnv(
            library_path=self.runtime,
            physics_backend="portable",
            config={
                "max_episode_ticks": 8,
                "initial_falling_count": 2,
            },
        )

    def test_real_episode_has_delayed_returns_and_no_model_visible_ids(self) -> None:
        with self._env() as env:
            episode = collect_dagger_episode(
                env,
                seed=17,
                episode_identity="development-seed-17",
                source_revision="de701b3",
                runtime_sha256=self.runtime_sha256,
                collector_id="collector-v1",
                fallback_teacher=_wait_teacher,
                config=DaggerCollectionConfig(
                    max_decisions=10,
                    teacher_beta_ppm=1_000_000,
                    maximum_search_queries=0,
                ),
            )
        self.assertFalse(episode.collector_cut)
        self.assertEqual(len(episode.trajectory.steps), 4)
        self.assertEqual(episode.teacher_actions, 4)
        self.assertEqual(episode.search_queries, 0)
        self.assertTrue(torch.isfinite(episode.supervision.returns).all())
        id_index = TEACHER_V1.body_features.index("id_scaled")
        self.assertTrue(
            (episode.supervision.body_features[..., id_index] == 0).all()
        )
        self.assertEqual(
            episode.actor_supervision.schema.source, "actor_tracks"
        )
        self.assertNotIn(
            "id_scaled", episode.actor_supervision.schema.body_features
        )
        weighted = advantage_weighted_behavior(
            (episode,), temperature=100.0, maximum_weight=10.0, coefficient=0.5
        )
        self.assertEqual(len(weighted), 1)
        assert weighted[0].policy_weight is not None
        self.assertTrue((weighted[0].policy_weight > 0).all())
        self.assertLessEqual(float(weighted[0].policy_weight.max()), 5.0)

    def test_distributional_leaf_critic_is_explicitly_enabled(self) -> None:
        model = _model()
        evaluator = DistributionalLeafEvaluator(model)
        with self._env() as env:
            observation, _ = env.reset(seed=19)
        self.assertEqual(evaluator(observation), 0.0)
        evaluator.enable()
        self.assertTrue(torch.isfinite(torch.tensor(evaluator(observation))))
        state = model.initial_state(1)
        state.fill_(0.25)
        evaluator.set_recurrent_state(state)
        state.zero_()
        self.assertTrue(bool((evaluator._recurrent_state == 0.25).all()))
        self.assertTrue(torch.isfinite(torch.tensor(evaluator(observation))))
        with self.assertRaisesRegex(ValueError, "context"):
            evaluator.set_recurrent_state(torch.zeros(2))

    def test_two_wave_loop_shifts_behavior_to_the_recurrent_policy(self) -> None:
        model = _model()
        trainer = PointerSequenceTrainer(
            model,
            config=PointerSequenceConfig(
                learning_rate=1e-3,
                tbptt_steps=4,
                entropy_coefficient=0.0,
            ),
        )
        improver = DaggerPolicyImprover(
            model,
            trainer,
            source_revision="de701b3",
            runtime_sha256=self.runtime_sha256,
            fallback_teacher=_wait_teacher,
            collection_config=DaggerCollectionConfig(
                max_decisions=10,
                maximum_search_queries=0,
            ),
            loop_config=DaggerLoopConfig(
                teacher_beta_ppm=(1_000_000, 0),
                sequence_updates_per_iteration=1,
                replay_capacity_episodes=4,
            ),
        )
        metrics = improver.run(self._env, ((21,), (22,)))
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0].policy_actions, 0)
        self.assertGreater(metrics[1].policy_actions, 0)
        self.assertEqual(len(improver.replay), 2)
        self.assertTrue(all(value.sequence.examples > 0 for value in metrics))


if __name__ == "__main__":
    unittest.main()
