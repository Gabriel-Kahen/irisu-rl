from __future__ import annotations

import copy
import math
import sys
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import Action, ActionKind, EventKind
from irisu_pointer.geometry_search import (
    DirectedPairGeometrySearch,
    GeometrySearchConfig,
    causal_tick_budget,
    enumerate_geometry_candidates,
    geometry_candidate_slots,
)
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import ActionSpec, SemanticAction, SemanticActionKind


def _body(
    identifier: int,
    *,
    x: float,
    y: float = 120.0,
    shape: str = "circle",
    angle: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    angular_velocity: float = 0.0,
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": shape,
        "lifecycle": "dynamic_fresh",
        "color": 1,
        "x": x,
        "y": y,
        "vx": vx / 10.0,
        "vy": vy / 10.0,
        "vx_display_per_second": vx,
        "vy_display_per_second": vy,
        "angle": angle,
        "angular_velocity": angular_velocity,
        "size": 40.0,
        "chain_id": 0,
        "projectile_hits": 0,
        "remaining_lifetime": 1000,
        "rot_timer": 0,
    }


def _observation(
    *,
    tick: int = 1,
    interval: int = 100,
    source: Mapping[str, object] | None = None,
    destination: Mapping[str, object] | None = None,
) -> dict[str, Any]:
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
        "bodies": [
            dict(source or _body(1, x=200.0)),
            dict(destination or _body(2, x=360.0, y=150.0)),
        ],
    }


def _incumbent() -> SteeringDecision:
    return SteeringDecision(
        SemanticAction.strong(180.0 / 640.0, 150.0 / 480.0),
        SteeringIntent.STEER_MATCH,
        source_body_id=1,
        destination_body_id=2,
        impact_x_sizes=-0.5,
        impact_y_sizes=0.75,
        correction_index=1,
        reason="incumbent directed pair",
    )


class GeometryCandidateTests(unittest.TestCase):
    def test_fixed_slot_vocabulary_is_bounded_and_state_invariant(self) -> None:
        config = GeometrySearchConfig()
        slots = geometry_candidate_slots(config)
        self.assertEqual(len(slots), 32)
        self.assertEqual(slots[0]["slot"], 0)
        self.assertEqual(slots[0]["name"], "incumbent")
        self.assertEqual([value["slot"] for value in slots], list(range(32)))
        self.assertLessEqual(len(slots), config.max_candidates)
        self.assertEqual(config.slot_count, 32)
        self.assertEqual(slots[24]["name"], "grid/0.75/0.9/strong")
        self.assertEqual(slots[25]["name"], "interior/center/weak")
        self.assertEqual(slots[31]["name"], "interior/deep-center/strong")

        circle = enumerate_geometry_candidates(
            _observation(), _incumbent(), config=config
        )
        triangle = enumerate_geometry_candidates(
            _observation(
                source=_body(
                    1,
                    x=200.0,
                    shape="triangle",
                    angle=0.7,
                    vx=40.0,
                    vy=-20.0,
                    angular_velocity=0.5,
                ),
                destination=_body(
                    2,
                    x=360.0,
                    y=150.0,
                    shape="box",
                    angle=-0.3,
                    vx=-25.0,
                ),
            ),
            _incumbent(),
            config=config,
        )
        circle_slots = {
            candidate.name: candidate.ordinal for candidate in circle.candidates
        }
        triangle_slots = {
            candidate.name: candidate.ordinal for candidate in triangle.candidates
        }
        self.assertEqual(
            {
                name: slot
                for name, slot in circle_slots.items()
                if slot < 25
            },
            {
                name: slot
                for name, slot in triangle_slots.items()
                if slot < 25
            },
        )
        self.assertNotEqual(circle.sha256, triangle.sha256)
        self.assertNotEqual(
            [
                (value.cursor_x, value.cursor_y)
                for value in circle.candidates
                if value.family == "shape-support"
            ],
            [
                (value.cursor_x, value.cursor_y)
                for value in triangle.candidates
                if value.family == "shape-support"
            ],
        )

    def test_candidates_include_incumbent_support_and_weak_strong_grid(self) -> None:
        candidates = enumerate_geometry_candidates(
            _observation(), _incumbent()
        )
        self.assertEqual(candidates.candidates[0].name, "incumbent")
        self.assertEqual(candidates.candidates[0].ordinal, 0)
        families = {candidate.family for candidate in candidates.candidates}
        self.assertEqual(
            families,
            {
                "incumbent",
                "shape-support",
                "bounded-grid",
                "central-interior",
            },
        )
        for family in (
            "shape-support",
            "bounded-grid",
            "central-interior",
        ):
            kinds = {
                SemanticActionKind(candidate.decision.action.kind)
                for candidate in candidates.candidates
                if candidate.family == family
            }
            self.assertEqual(
                kinds,
                {
                    SemanticActionKind.FIRE_WEAK,
                    SemanticActionKind.FIRE_STRONG,
                },
            )
        repeated = enumerate_geometry_candidates(_observation(), _incumbent())
        self.assertEqual(candidates, repeated)
        self.assertEqual(candidates.sha256, repeated.sha256)

    def test_central_interior_slots_are_distinct_and_shape_local(self) -> None:
        circle = enumerate_geometry_candidates(
            _observation(), _incumbent()
        )
        interior = tuple(
            value
            for value in circle.candidates
            if value.family == "central-interior"
        )
        self.assertEqual(
            [value.ordinal for value in interior], list(range(25, 32))
        )
        self.assertEqual(
            len(
                {
                    (
                        int(value.decision.action.kind),
                        value.cursor_x,
                        value.cursor_y,
                    )
                    for value in interior
                }
            ),
            7,
        )

        rotated = enumerate_geometry_candidates(
            _observation(
                source=_body(
                    1, x=200.0, shape="box", angle=math.pi / 2.0
                )
            ),
            _incumbent(),
        )
        lower = rotated.candidate_at(29)
        assert lower is not None
        self.assertEqual((lower.cursor_x, lower.cursor_y), (190, 120))

    def test_legacy_slots_are_unchanged_by_interior_extension(self) -> None:
        observation = _observation()
        full = enumerate_geometry_candidates(observation, _incumbent())
        capped = enumerate_geometry_candidates(
            observation,
            _incumbent(),
            config=GeometrySearchConfig(max_candidates=25),
        )
        self.assertEqual(
            tuple(
                candidate
                for candidate in full.candidates
                if candidate.ordinal < 25
            ),
            capped.candidates,
        )

    def test_triangle_and_offscreen_interior_slots_are_masked(self) -> None:
        triangle = enumerate_geometry_candidates(
            _observation(
                source=_body(
                    1, x=200.0, shape="triangle", angle=0.8
                )
            ),
            _incumbent(),
        )
        self.assertFalse(triangle.availability_mask[27])
        self.assertFalse(triangle.availability_mask[28])
        self.assertTrue(
            all(
                triangle.availability_mask[slot]
                for slot in (25, 26, 29, 30, 31)
            )
        )

        top_edge = enumerate_geometry_candidates(
            _observation(source=_body(1, x=200.0, y=5.0)),
            _incumbent(),
        )
        self.assertFalse(top_edge.availability_mask[27])
        self.assertFalse(top_edge.availability_mask[28])
        self.assertTrue(top_edge.availability_mask[29])
        self.assertTrue(top_edge.availability_mask[30])

    def test_offscreen_slots_are_masked_without_renumbering(self) -> None:
        observation = _observation(
            source=_body(1, x=4.0),
            destination=_body(2, x=300.0),
        )
        candidates = enumerate_geometry_candidates(observation, _incumbent())
        by_name = {
            candidate.name: candidate.ordinal for candidate in candidates.candidates
        }
        self.assertEqual(by_name["incumbent"], 0)
        self.assertNotIn("support/0.85/weak", by_name)
        self.assertFalse(candidates.availability_mask[5])
        self.assertIsNone(candidates.candidate_at(5))
        for candidate in candidates.candidates:
            expected = next(
                slot["slot"]
                for slot in geometry_candidate_slots()
                if slot["name"] == candidate.name
            )
            self.assertEqual(candidate.ordinal, expected)

    def test_pair_validation_fails_closed(self) -> None:
        missing = _observation()
        missing["bodies"] = missing["bodies"][:1]
        with self.assertRaisesRegex(ValueError, "absent"):
            enumerate_geometry_candidates(missing, _incumbent())
        unknown = _observation(source=_body(1, x=200.0, shape="unknown"))
        with self.assertRaisesRegex(ValueError, "unknown shape"):
            enumerate_geometry_candidates(unknown, _incumbent())
        with self.assertRaisesRegex(ValueError, "directed-pair"):
            enumerate_geometry_candidates(
                _observation(),
                SteeringDecision(
                    SemanticAction.wait(1),
                    SteeringIntent.WAIT,
                    reason="not a pair",
                ),
            )

    def test_identity_manifest_binds_config_and_action_contract(self) -> None:
        first = DirectedPairGeometrySearch()
        second = DirectedPairGeometrySearch()
        changed = DirectedPairGeometrySearch(
            config=GeometrySearchConfig(horizon_ticks=32)
        )
        self.assertEqual(first.identity_manifest(), second.identity_manifest())
        self.assertEqual(first.sha256, second.sha256)
        self.assertNotEqual(first.sha256, changed.sha256)
        self.assertEqual(
            first.identity_manifest()["backend"], "portable-clone-only"
        )


class _PortableGeometryEnv:
    physics_backend = "portable"

    def __init__(
        self,
        observation: Mapping[str, Any],
        behaviors: Mapping[
            tuple[int, int, int], Mapping[str, object]
        ] | None = None,
    ) -> None:
        self.observation = copy.deepcopy(dict(observation))
        self.behaviors = dict(behaviors or {})
        self.branch_key: tuple[int, int, int] | None = None
        self.restores = 0
        self.clones = 0
        self.max_tick = int(self.observation["tick"])

    def clone_state(self) -> tuple[dict[str, Any], object]:
        self.clones += 1
        return copy.deepcopy(self.observation), self.branch_key

    def restore_state(
        self, snapshot: tuple[dict[str, Any], object]
    ) -> dict[str, Any]:
        self.restores += 1
        self.observation, self.branch_key = copy.deepcopy(snapshot)
        return copy.deepcopy(self.observation)

    def step(
        self, action: Action
    ) -> tuple[dict[str, Any], int, bool, bool, dict[str, Any]]:
        kind = ActionKind.parse(action.kind)
        duration = int(action.wait_ticks) if kind is ActionKind.WAIT else 1
        events: list[dict[str, int]] = []
        reward = 0
        terminated = False
        if kind is not ActionKind.WAIT:
            self.branch_key = (
                int(kind),
                int(action.cursor_x),
                int(action.cursor_y),
            )
            behavior = self.behaviors.get(self.branch_key, {})
            if bool(behavior.get("raise", False)):
                raise RuntimeError("synthetic geometry branch failure")
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
            if bool(behavior.get("invalid", False)):
                events.append(
                    {
                        "kind": int(EventKind.INVALID_ACTION),
                        "a": 0,
                        "b": 0,
                        "value": 0,
                    }
                )
            terminated = bool(behavior.get("terminal", False))
            self.observation["terminated"] = terminated
        for _ in range(duration):
            interval = int(
                self.observation["difficulty"]["spawn_interval_ticks"]
            )
            if int(self.observation["tick"]) % interval == 0:
                raise AssertionError("geometry search crossed a spawn boundary")
            self.observation["tick"] = int(self.observation["tick"]) + 1
            self.max_tick = max(self.max_tick, int(self.observation["tick"]))
            if terminated:
                break
        return (
            copy.deepcopy(self.observation),
            reward,
            terminated,
            False,
            {
                "events": events,
                "invalid_action": any(
                    event["kind"] == int(EventKind.INVALID_ACTION)
                    for event in events
                ),
            },
        )


def _action_key(candidate: object) -> tuple[int, int, int]:
    decision = candidate.decision
    action = decision.primitive_actions(ActionSpec())[0]
    return (
        int(ActionKind.parse(action.kind)),
        int(action.cursor_x),
        int(action.cursor_y),
    )


class GeometrySearchTests(unittest.TestCase):
    def test_search_selects_only_survival_nondominated_improvement(self) -> None:
        observation = _observation()
        incumbent = _incumbent()
        candidate_set = enumerate_geometry_candidates(observation, incumbent)
        dominated = next(
            value
            for value in candidate_set.candidates
            if value.name == "support/0.35/weak"
        )
        winner = next(
            value
            for value in candidate_set.candidates
            if value.name == "support/0.35/strong"
        )
        behaviors = {
            _action_key(dominated): {
                "score": 100,
                "gauge": 49,
                "hit": True,
                "join": True,
            },
            _action_key(winner): {
                "score": 30,
                "gauge": 100,
                "hit": True,
                "join": True,
            },
        }
        env = _PortableGeometryEnv(observation, behaviors)
        before = env.clone_state()
        result = DirectedPairGeometrySearch(
            config=GeometrySearchConfig(horizon_ticks=8)
        ).search(env, observation, incumbent)
        self.assertEqual(result.selected_candidate.name, winner.name)
        self.assertTrue(result.strictly_improved)
        dominated_outcome = next(
            value
            for value in result.outcomes
            if value.candidate.name == dominated.name
        )
        incumbent_outcome = result.outcomes[0]
        self.assertFalse(
            dominated_outcome.survival_nondominated_by(incumbent_outcome)
        )
        self.assertEqual(env.clone_state(), before)
        self.assertGreaterEqual(env.restores, len(result.outcomes) + 1)

    def test_search_is_deterministic_and_censored_before_next_spawn(self) -> None:
        observation = _observation(tick=7, interval=10)
        incumbent = _incumbent()
        first_env = _PortableGeometryEnv(observation)
        second_env = _PortableGeometryEnv(observation)
        search = DirectedPairGeometrySearch()
        first = search.search(first_env, observation, incumbent)
        second = search.search(second_env, observation, incumbent)
        self.assertEqual(first.causal_horizon_ticks, 3)
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(first.selected_candidate.name, "incumbent")
        self.assertEqual(first_env.max_tick, 10)
        self.assertEqual(second_env.max_tick, 10)

    def test_branch_error_restores_source_environment(self) -> None:
        observation = _observation()
        incumbent = _incumbent()
        candidate = enumerate_geometry_candidates(
            observation, incumbent
        ).candidates[1]
        env = _PortableGeometryEnv(
            observation, {_action_key(candidate): {"raise": True}}
        )
        before = env.clone_state()
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            DirectedPairGeometrySearch().search(env, observation, incumbent)
        self.assertEqual(env.clone_state(), before)

    def test_nonportable_backend_fails_before_cloning(self) -> None:
        env = _PortableGeometryEnv(_observation())
        env.physics_backend = "exact"
        with self.assertRaisesRegex(ValueError, "portable"):
            DirectedPairGeometrySearch().search(
                env, _observation(), _incumbent()
            )
        self.assertEqual(env.clones, 0)

    def test_too_short_causal_boundary_returns_incumbent_without_mutation(
        self,
    ) -> None:
        observation = _observation(tick=99, interval=100)
        env = _PortableGeometryEnv(observation)
        result = DirectedPairGeometrySearch().search(
            env, observation, _incumbent()
        )
        self.assertEqual(causal_tick_budget(observation, 64), 1)
        self.assertEqual(result.causal_horizon_ticks, 1)
        self.assertEqual(result.selected_candidate.name, "incumbent")
        self.assertEqual(result.outcomes, ())
        self.assertEqual(env.clones, 0)


if __name__ == "__main__":
    unittest.main()
