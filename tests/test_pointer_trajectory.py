from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import ActionKind, EventKind
from irisu_pointer.experts import PointerExpertDecision
from irisu_pointer.trajectory import (
    DelayedRewardSpec,
    PublicEvent,
    TrajectoryEpisode,
    TrajectoryProvenance,
    TrajectoryStep,
    deserialize_episodes,
    discounted_return_targets,
    episode_sequence_windows,
    serialize_episodes,
    split_episodes,
)


def _provenance(
    reward: DelayedRewardSpec,
    *,
    collector: str = "phase2-fixture",
) -> TrajectoryProvenance:
    return TrajectoryProvenance(
        source_revision="de701b3",
        runtime_sha256="a" * 64,
        config_hash=123,
        observation_schema_sha256="b" * 64,
        pointer_spec_sha256="c" * 64,
        reward_spec_sha256=reward.sha256,
        collector_id=collector,
    )


def _steps(
    episode_identity: str,
    provenance: TrajectoryProvenance,
    count: int,
    *,
    score_per_step: int = 1,
    terminal: bool = True,
) -> tuple[TrajectoryStep, ...]:
    return tuple(
        TrajectoryStep(
            episode_identity=episode_identity,
            provenance_sha256=provenance.sha256,
            decision_index=index,
            tick_start=index * 2,
            tick_end=(index + 1) * 2,
            action_kind=int(ActionKind.WAIT),
            wait_ticks=2,
            target_body_id=None,
            template_index=7,
            score_before=index * score_per_step,
            score_after=(index + 1) * score_per_step,
            gauge_before=100 - index,
            gauge_after=99 - index,
            highest_chain_before=0,
            highest_chain_after=0,
            terminated=terminal and index == count - 1,
        )
        for index in range(count)
    )


def _episode(
    identity: str,
    reward: DelayedRewardSpec,
    count: int,
    *,
    seed: int = 1,
    provenance: TrajectoryProvenance | None = None,
) -> TrajectoryEpisode:
    resolved = provenance or _provenance(reward)
    return TrajectoryEpisode(
        identity,
        seed,
        resolved,
        _steps(identity, resolved, count),
    )


class DelayedRewardTests(unittest.TestCase):
    def test_reward_decomposition_keeps_every_term_auditable(self) -> None:
        reward = DelayedRewardSpec(
            raw_score_delta=1,
            survival_tick=0.5,
            chain_continuation=10,
            chain_join=3,
            clear=7,
            gauge_delta=2,
            rot_penalty=-20,
            terminal_penalty=-50,
            invalid_penalty=-100,
        )
        provenance = _provenance(reward)
        step = TrajectoryStep(
            episode_identity="reward-episode",
            provenance_sha256=provenance.sha256,
            decision_index=0,
            tick_start=10,
            tick_end=20,
            action_kind=int(ActionKind.STRONG_SHOT),
            wait_ticks=1,
            target_body_id=8,
            template_index=7,
            score_before=100,
            score_after=200,
            gauge_before=50,
            gauge_after=40,
            highest_chain_before=1,
            highest_chain_after=3,
            events=(
                PublicEvent(12, int(EventKind.CHAIN_JOINED)),
                PublicEvent(13, int(EventKind.CHAIN_JOINED)),
                PublicEvent(14, int(EventKind.CLEARED)),
                PublicEvent(15, int(EventKind.ROTTEN)),
                PublicEvent(16, int(EventKind.INVALID_ACTION)),
            ),
            terminated=True,
        )
        value = reward.decompose(step)
        self.assertEqual(value.raw_score_delta, 100)
        self.assertEqual(value.survival_ticks, 10)
        self.assertEqual(value.chain_continuation, 2)
        self.assertEqual(value.chain_joins, 2)
        self.assertEqual(value.clears, 1)
        self.assertEqual(value.gauge_delta, -10)
        self.assertEqual(value.rot_events, 1)
        self.assertEqual(value.invalid_events, 1)
        self.assertEqual(value.total, -52.0)

    def test_duration_discount_and_51_quantile_targets(self) -> None:
        reward = DelayedRewardSpec(
            raw_score_delta=1,
            survival_tick=0,
            chain_continuation=0,
            chain_join=0,
            clear=0,
            gauge_delta=0,
            rot_penalty=0,
            terminal_penalty=0,
            invalid_penalty=0,
        )
        provenance = _provenance(reward)
        first, second = _steps(
            "returns",
            provenance,
            2,
            score_per_step=1,
        )
        first = replace(
            first,
            tick_end=2,
            score_before=0,
            score_after=2,
            gauge_after=99,
        )
        second = replace(
            second,
            tick_start=2,
            tick_end=5,
            score_before=2,
            score_after=5,
            gauge_before=99,
            gauge_after=98,
        )
        episode = TrajectoryEpisode(
            "returns", 7, provenance, (first, second)
        )
        targets = discounted_return_targets(
            episode,
            reward,
            gamma_tick=0.5,
        )
        self.assertEqual(targets.scalar_returns, (2.75, 3.0))
        self.assertEqual(len(targets.quantile_fractions), 51)
        self.assertEqual(len(targets.quantile_targets), 2)
        self.assertEqual(targets.quantile_targets[0], (2.75,) * 51)
        self.assertAlmostEqual(targets.quantile_fractions[0], 0.5 / 51)
        self.assertAlmostEqual(targets.quantile_fractions[-1], 50.5 / 51)

    def test_public_transition_constructor_binds_action_and_events(self) -> None:
        reward = DelayedRewardSpec()
        provenance = _provenance(reward)
        step = TrajectoryStep.from_public_transition(
            episode_identity="public",
            provenance_sha256=provenance.sha256,
            decision_index=0,
            decision=PointerExpertDecision.strong(9, template_index=12),
            before={
                "tick": 10,
                "score": 2,
                "gauge": 80,
                "highest_chain": 1,
            },
            after={
                "tick": 12,
                "score": 5,
                "gauge": 81,
                "highest_chain": 2,
            },
            info={
                "events": [
                    {
                        "tick": 11,
                        "kind": int(EventKind.PROJECTILE_HIT),
                        "a": 20,
                        "b": 9,
                        "value": 0,
                    }
                ]
            },
            terminated=False,
            truncated=False,
        )
        self.assertEqual(step.target_body_id, 9)
        self.assertEqual(step.template_index, 12)
        self.assertEqual(step.elapsed_ticks, 2)
        self.assertEqual(step.events[0].b, 9)


class SequenceWindowTests(unittest.TestCase):
    def test_windows_pad_burn_in_and_never_cross_episodes(self) -> None:
        reward = DelayedRewardSpec()
        episodes = (
            _episode("episode-a", reward, 5),
            _episode("episode-b", reward, 3, seed=2),
        )
        windows = episode_sequence_windows(
            episodes,
            burn_in=2,
            unroll=2,
            stride=2,
        )
        first = windows[0]
        self.assertEqual(first.episode_identity, "episode-a")
        self.assertEqual(first.burn_in_mask, (False, False, False, False))
        self.assertEqual(first.unroll_mask, (False, False, True, True))
        self.assertEqual(first.valid_mask, (False, False, True, True))
        middle = windows[1]
        self.assertEqual(middle.burn_in_mask, (True, True, False, False))
        self.assertEqual(middle.unroll_mask, (False, False, True, True))
        tail = windows[2]
        self.assertEqual(tail.valid_mask, (True, True, True, False))
        self.assertEqual(tail.unroll_mask, (False, False, True, False))
        for window in windows:
            self.assertTrue(
                all(
                    step is None
                    or step.episode_identity == window.episode_identity
                    for step in window.steps
                )
            )

    def test_split_is_deterministic_and_episode_disjoint(self) -> None:
        reward = DelayedRewardSpec()
        episodes = tuple(
            _episode(f"episode-{index}", reward, 2, seed=index)
            for index in range(6)
        )
        first = split_episodes(episodes, validation_fraction=0.33)
        second = split_episodes(episodes, validation_fraction=0.33)
        self.assertEqual(first, second)
        train_ids = {episode.episode_identity for episode in first[0]}
        validation_ids = {episode.episode_identity for episode in first[1]}
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(train_ids | validation_ids, {
            episode.episode_identity for episode in episodes
        })


class SerializationTests(unittest.TestCase):
    def test_canonical_round_trip_and_checksum_rejects_tampering(self) -> None:
        reward = DelayedRewardSpec()
        provenance = _provenance(reward)
        episodes = (
            _episode("episode-a", reward, 3, provenance=provenance),
            _episode("episode-b", reward, 2, seed=2, provenance=provenance),
        )
        encoded = serialize_episodes(episodes)
        self.assertEqual(serialize_episodes(episodes), encoded)
        self.assertEqual(deserialize_episodes(encoded), episodes)

        envelope = json.loads(encoded)
        envelope["payload"]["episodes"][0]["steps"][0]["score_after"] += 1
        tampered = json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        with self.assertRaisesRegex(ValueError, "checksum"):
            deserialize_episodes(tampered)

    def test_provenance_mixing_and_foreign_steps_fail_closed(self) -> None:
        reward = DelayedRewardSpec()
        first_provenance = _provenance(reward)
        second_provenance = _provenance(reward, collector="foreign")
        first = _episode(
            "first", reward, 2, provenance=first_provenance
        )
        second = _episode(
            "second", reward, 2, provenance=second_provenance
        )
        with self.assertRaisesRegex(ValueError, "share exact provenance"):
            serialize_episodes((first, second))
        with self.assertRaisesRegex(ValueError, "provenance"):
            TrajectoryEpisode(
                "first",
                1,
                first_provenance,
                tuple(
                    replace(
                        step,
                        provenance_sha256=second_provenance.sha256,
                    )
                    for step in first.steps
                ),
            )


if __name__ == "__main__":
    unittest.main()
