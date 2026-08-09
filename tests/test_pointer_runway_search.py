from __future__ import annotations

import copy
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import Action, ActionKind, EventKind
from irisu_pointer.geometry_search import enumerate_geometry_candidates
from irisu_pointer.runway_search import (
    RunwayGeometrySearch,
    RunwaySearchConfig,
)
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import ActionSpec, SemanticAction


def _body(identifier: int, *, x: float) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": "dynamic_fresh",
        "color": 1,
        "x": x,
        "y": 120.0,
        "vx": 0.0,
        "vy": 0.0,
        "vx_display_per_second": 0.0,
        "vy_display_per_second": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 40.0,
        "chain_id": 0,
        "projectile_hits": 0,
        "remaining_lifetime": 1000,
        "rot_timer": 0,
    }


def _observation(*, tick: int = 1, interval: int = 5) -> dict[str, Any]:
    return {
        "tick": tick,
        "score": 0,
        "gauge": 100,
        "gauge_max": 100,
        "level": 1,
        "highest_chain": 0,
        "qualifying_clear_count": 0,
        "terminated": False,
        "truncated": False,
        "difficulty": {"spawn_interval_ticks": interval},
        "field": {"x": 16.0, "y": 0.0, "width": 576.0, "height": 480.0},
        "bodies": [_body(1, x=200.0), _body(2, x=360.0)],
    }


def _incumbent() -> SteeringDecision:
    return SteeringDecision(
        SemanticAction.strong(180.0 / 640.0, 150.0 / 480.0),
        SteeringIntent.STEER_MATCH,
        source_body_id=1,
        destination_body_id=2,
        impact_x_sizes=-0.5,
        impact_y_sizes=0.75,
        reason="runway incumbent",
    )


def _action_key(candidate: object) -> tuple[int, int, int]:
    action = candidate.decision.primitive_actions(ActionSpec())[0]
    return (
        int(ActionKind.parse(action.kind)),
        int(action.cursor_x),
        int(action.cursor_y),
    )


class _RunwayEnv:
    physics_backend = "portable"

    def __init__(
        self,
        observation: Mapping[str, Any],
        *,
        runway_ticks: int,
        behaviors: Mapping[
            tuple[int, int, int], Mapping[str, object]
        ] | None = None,
    ) -> None:
        self.observation = copy.deepcopy(dict(observation))
        self.initial_tick = int(self.observation["tick"])
        self.runway_ticks = runway_ticks
        self.behaviors = dict(behaviors or {})
        self.branch_key: tuple[int, int, int] | None = None
        self.rng_state = 17
        self.branch_spawns: list[int] = []
        self.completed_spawn_sequences: list[tuple[int, ...]] = []
        self.restores = 0
        self.clones = 0
        self.max_tick = self.initial_tick

    def clone_state(
        self,
    ) -> tuple[dict[str, Any], object, int, list[int]]:
        self.clones += 1
        return (
            copy.deepcopy(self.observation),
            self.branch_key,
            self.rng_state,
            list(self.branch_spawns),
        )

    def restore_state(
        self,
        snapshot: tuple[dict[str, Any], object, int, list[int]],
    ) -> dict[str, Any]:
        self.restores += 1
        (
            self.observation,
            self.branch_key,
            self.rng_state,
            self.branch_spawns,
        ) = copy.deepcopy(snapshot)
        return copy.deepcopy(self.observation)

    def step(
        self, action: Action
    ) -> tuple[dict[str, Any], int, bool, bool, dict[str, Any]]:
        kind = ActionKind.parse(action.kind)
        duration = int(action.wait_ticks) if kind is ActionKind.WAIT else 1
        reward = 0
        events: list[dict[str, int]] = []
        terminated = False
        if kind is not ActionKind.WAIT:
            self.branch_key = (
                int(kind),
                int(action.cursor_x),
                int(action.cursor_y),
            )
            behavior = self.behaviors.get(self.branch_key, {})
            if bool(behavior.get("raise", False)):
                raise RuntimeError("synthetic runway branch failure")
            reward = int(behavior.get("score", 0))
            self.observation["score"] = int(self.observation["score"]) + reward
            self.observation["gauge"] = int(
                behavior.get("gauge", self.observation["gauge"])
            )
            events.append(
                {
                    "kind": int(EventKind.SHOT_FIRED),
                    "a": 900,
                    "b": 0,
                    "value": int(kind is ActionKind.STRONG_SHOT),
                }
            )
            if bool(behavior.get("hit", False)):
                events.append(
                    {
                        "kind": int(EventKind.PROJECTILE_HIT),
                        "a": 900,
                        "b": 1,
                        "value": 1,
                    }
                )
            if bool(behavior.get("join", False)):
                events.append(
                    {
                        "kind": int(EventKind.CHAIN_JOINED),
                        "a": 1,
                        "b": 2,
                        "value": 7,
                    }
                )
            terminated = bool(behavior.get("terminal", False))
            self.observation["terminated"] = terminated

        for _ in range(duration):
            self.observation["tick"] = int(self.observation["tick"]) + 1
            self.max_tick = max(self.max_tick, int(self.observation["tick"]))
            interval = int(
                self.observation["difficulty"]["spawn_interval_ticks"]
            )
            if int(self.observation["tick"]) % interval == 0:
                self.rng_state = (
                    1_103_515_245 * self.rng_state + 12_345
                ) & 0x7FFFFFFF
                self.branch_spawns.append(self.rng_state)
            if terminated:
                break
        if (
            int(self.observation["tick"]) - self.initial_tick
            == self.runway_ticks
        ):
            self.completed_spawn_sequences.append(tuple(self.branch_spawns))
        return (
            copy.deepcopy(self.observation),
            reward,
            terminated,
            False,
            {"events": events, "invalid_action": False},
        )


class RunwaySearchTests(unittest.TestCase):
    def test_default_identity_is_explicitly_development_only(self) -> None:
        teacher = RunwayGeometrySearch()
        self.assertEqual(teacher.config.runway_ticks, 256)
        self.assertEqual(teacher.config.candidate_config.slot_count, 32)
        manifest = teacher.identity_manifest()
        self.assertFalse(manifest["deployable"])
        self.assertFalse(manifest["canonical_evidence"])
        self.assertEqual(
            manifest["evidence_scope"], "development-teacher-only"
        )
        self.assertEqual(manifest["hidden_policy_inputs"], [])
        self.assertEqual(teacher.sha256, RunwayGeometrySearch().sha256)
        self.assertNotEqual(
            teacher.sha256,
            RunwayGeometrySearch(
                config=RunwaySearchConfig(runway_ticks=128)
            ).sha256,
        )

    def test_full_runway_crosses_spawns_with_identical_rng_for_every_branch(
        self,
    ) -> None:
        observation = _observation(tick=1, interval=5)
        incumbent = _incumbent()
        runway_ticks = 20
        config = RunwaySearchConfig(runway_ticks=runway_ticks)
        candidate_set = enumerate_geometry_candidates(
            observation, incumbent, config=config.candidate_config
        )
        preferred = next(
            value
            for value in candidate_set.candidates
            if value.name == "grid/0.75/0.6/weak"
        )
        env = _RunwayEnv(
            observation,
            runway_ticks=runway_ticks,
            behaviors={
                _action_key(preferred): {
                    "score": 50,
                    "gauge": 100,
                    "hit": True,
                    "join": True,
                }
            },
        )
        before = env.clone_state()
        result = RunwayGeometrySearch(config=config).search(
            env, observation, incumbent
        )
        self.assertEqual(result.runway_ticks, runway_ticks)
        self.assertEqual(result.winner_ordinal, preferred.ordinal)
        self.assertEqual(result.decision, preferred.decision)
        self.assertEqual(len(result.outcomes), len(candidate_set.candidates))
        self.assertEqual(
            {outcome.survival_ticks for outcome in result.outcomes},
            {runway_ticks},
        )
        self.assertEqual(env.max_tick, int(observation["tick"]) + runway_ticks)
        self.assertEqual(
            len(env.completed_spawn_sequences), len(result.outcomes)
        )
        self.assertEqual(
            len(set(env.completed_spawn_sequences)), 1
        )
        self.assertEqual(len(env.completed_spawn_sequences[0]), 4)
        self.assertEqual(env.clone_state(), before)

    def test_high_score_candidate_cannot_cross_below_reserve_band(self) -> None:
        observation = _observation()
        incumbent = _incumbent()
        config = RunwaySearchConfig(runway_ticks=12)
        candidates = enumerate_geometry_candidates(
            observation, incumbent, config=config.candidate_config
        )
        dominated = candidates.candidates[1]
        safe = candidates.candidates[2]
        env = _RunwayEnv(
            observation,
            runway_ticks=config.runway_ticks,
            behaviors={
                _action_key(dominated): {"score": 500, "gauge": 49},
                _action_key(safe): {"score": 30, "gauge": 100},
            },
        )
        result = RunwayGeometrySearch(config=config).search(
            env, observation, incumbent
        )
        incumbent_outcome = result.outcomes[0]
        dominated_outcome = next(
            value
            for value in result.outcomes
            if value.candidate.ordinal == dominated.ordinal
        )
        self.assertFalse(
            dominated_outcome.survival_nondominated_by(incumbent_outcome)
        )
        self.assertEqual(result.winner_ordinal, safe.ordinal)

    def test_reserve_band_prefers_more_gauge_below_half(self) -> None:
        observation = _observation()
        observation["gauge"] = 40
        incumbent = _incumbent()
        config = RunwaySearchConfig(runway_ticks=12)
        candidates = enumerate_geometry_candidates(
            observation, incumbent, config=config.candidate_config
        )
        high_score = candidates.candidates[1]
        renewable = candidates.candidates[2]
        env = _RunwayEnv(
            observation,
            runway_ticks=config.runway_ticks,
            behaviors={
                _action_key(high_score): {"score": 500, "gauge": 40},
                _action_key(renewable): {"score": 30, "gauge": 49},
            },
        )
        result = RunwayGeometrySearch(config=config).search(
            env, observation, incumbent
        )
        self.assertEqual(result.winner_ordinal, renewable.ordinal)

    def test_reserve_band_prefers_score_at_or_above_half(self) -> None:
        observation = _observation()
        observation["gauge"] = 80
        incumbent = _incumbent()
        config = RunwaySearchConfig(runway_ticks=12)
        candidates = enumerate_geometry_candidates(
            observation, incumbent, config=config.candidate_config
        )
        high_score = candidates.candidates[1]
        surplus = candidates.candidates[2]
        env = _RunwayEnv(
            observation,
            runway_ticks=config.runway_ticks,
            behaviors={
                _action_key(high_score): {"score": 500, "gauge": 50},
                _action_key(surplus): {"score": 30, "gauge": 90},
            },
        )
        result = RunwayGeometrySearch(config=config).search(
            env, observation, incumbent
        )
        self.assertEqual(result.winner_ordinal, high_score.ordinal)

    def test_branch_failure_restores_snapshot_and_propagates(self) -> None:
        observation = _observation()
        incumbent = _incumbent()
        config = RunwaySearchConfig(runway_ticks=12)
        candidate = enumerate_geometry_candidates(
            observation, incumbent, config=config.candidate_config
        ).candidates[1]
        env = _RunwayEnv(
            observation,
            runway_ticks=config.runway_ticks,
            behaviors={_action_key(candidate): {"raise": True}},
        )
        before = env.clone_state()
        with self.assertRaisesRegex(RuntimeError, "synthetic runway"):
            RunwayGeometrySearch(config=config).search(
                env, observation, incumbent
            )
        self.assertEqual(env.clone_state(), before)

    def test_nonportable_backend_fails_before_snapshotting(self) -> None:
        observation = _observation()
        env = _RunwayEnv(observation, runway_ticks=8)
        env.physics_backend = "exact"
        with self.assertRaisesRegex(ValueError, "portable"):
            RunwayGeometrySearch(
                config=RunwaySearchConfig(runway_ticks=8)
            ).search(env, observation, _incumbent())
        self.assertEqual(env.clones, 0)

    def test_result_identity_retains_all_branch_outcomes_and_winner(self) -> None:
        observation = _observation()
        config = RunwaySearchConfig(runway_ticks=8)
        first = RunwayGeometrySearch(config=config).search(
            _RunwayEnv(observation, runway_ticks=8),
            observation,
            _incumbent(),
        )
        second = RunwayGeometrySearch(config=config).search(
            _RunwayEnv(observation, runway_ticks=8),
            observation,
            _incumbent(),
        )
        manifest = first.manifest()
        self.assertEqual(len(manifest["outcomes"]), len(first.outcomes))
        self.assertEqual(manifest["winner_ordinal"], first.winner_ordinal)
        self.assertEqual(first.sha256, second.sha256)
        self.assertFalse(manifest["deployable"])
        self.assertFalse(manifest["canonical_evidence"])


if __name__ == "__main__":
    unittest.main()
