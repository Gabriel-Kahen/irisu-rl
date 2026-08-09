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
from irisu_pointer.experts import PointerExpertDecision, SearchCandidate
from irisu_pointer.macro_search import (
    SpawnCensoredMacroBeamTeacher,
    chain_score_potential,
    generate_macro_candidates,
    macro_utility,
)


def _body(
    identifier: int,
    *,
    x: float,
    color: int = 0,
    lifecycle: str = "dynamic_fresh",
    chain_id: int = 0,
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "box",
        "lifecycle": lifecycle,
        "color": color,
        "x": x,
        "y": 100.0,
        "vx": 0.0,
        "vy": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 40.0,
        "chain_id": chain_id,
        "projectile_hits": 0,
        "age_ticks": 10,
        "remaining_lifetime": 1000,
        "rot_timer": 0,
    }


def _observation(
    *,
    tick: int = 1,
    interval: int = 20,
    score: int = 0,
    gauge: int = 3000,
    bodies: list[dict[str, object]] | None = None,
) -> dict[str, Any]:
    return {
        "tick": tick,
        "score": score,
        "gauge": gauge,
        "gauge_max": 40000,
        "level": 1,
        "highest_chain": 0,
        "qualifying_clear_count": 0,
        "terminated": False,
        "truncated": False,
        "difficulty": {"spawn_interval_ticks": interval},
        "field": {"x": 130.0, "width": 320.0},
        "bodies": list(
            bodies
            or [
                _body(1, x=100.0),
                _body(2, x=300.0),
                _body(3, x=500.0, color=1),
            ]
        ),
    }


class _MacroEnv:
    physics_backend = "portable"

    def __init__(
        self,
        observation: Mapping[str, Any],
        *,
        fail_cursor_x: float | None = None,
    ) -> None:
        self.observation = copy.deepcopy(dict(observation))
        self.phase = 0
        self.restores = 0
        self.max_tick = int(self.observation["tick"])
        self.fail_cursor_x = fail_cursor_x

    def clone_state(self) -> tuple[dict[str, Any], int]:
        return copy.deepcopy(self.observation), self.phase

    def restore_state(
        self, snapshot: tuple[dict[str, Any], int]
    ) -> dict[str, Any]:
        self.restores += 1
        self.observation, self.phase = copy.deepcopy(snapshot)
        return copy.deepcopy(self.observation)

    def step(
        self, action: Action
    ) -> tuple[dict[str, Any], int, bool, bool, dict[str, Any]]:
        kind = ActionKind.parse(action.kind)
        ticks = int(action.wait_ticks) if kind is ActionKind.WAIT else 1
        interval = int(self.observation["difficulty"]["spawn_interval_ticks"])
        for _ in range(ticks):
            if int(self.observation["tick"]) % interval == 0:
                raise AssertionError("macro search crossed the next spawn")
            self.observation["tick"] = int(self.observation["tick"]) + 1
            self.max_tick = max(self.max_tick, int(self.observation["tick"]))
        events: list[dict[str, int]] = []
        reward = 0
        if kind is not ActionKind.WAIT:
            x = float(action.cursor_x)
            if self.fail_cursor_x is not None and x == self.fail_cursor_x:
                raise RuntimeError("synthetic branch failure")
            if kind is ActionKind.STRONG_SHOT and x == 100.0:
                self.phase = 1
            elif kind is ActionKind.WEAK_SHOT and x == 300.0 and self.phase == 1:
                reward = 100
                self.observation["score"] = int(self.observation["score"]) + reward
                self.observation["highest_chain"] = 2
                self.observation["qualifying_clear_count"] = 1
                for body in self.observation["bodies"][:2]:
                    body["chain_id"] = 7
                    body["lifecycle"] = "confirmed"
                events.extend(
                    [
                        {
                            "kind": int(EventKind.CHAIN_JOINED),
                            "a": 1,
                            "b": 2,
                            "value": 7,
                        },
                        {
                            "kind": int(EventKind.CHAIN_JOINED),
                            "a": 2,
                            "b": 1,
                            "value": 7,
                        },
                    ]
                )
            elif kind is ActionKind.STRONG_SHOT and x == 500.0:
                reward = 20
                self.observation["score"] = int(self.observation["score"]) + reward
        return (
            copy.deepcopy(self.observation),
            reward,
            False,
            False,
            {"events": events},
        )


class _ContextEvaluator:
    def __init__(self) -> None:
        self.state = None

    def set_recurrent_state(self, state: object | None) -> None:
        self.state = state

    def __call__(self, observation: Mapping[str, Any]) -> float:
        del observation
        return 0.0


def _three_candidates(
    _observation: Mapping[str, Any], _spec: object
) -> tuple[SearchCandidate, ...]:
    return (
        SearchCandidate("setup", PointerExpertDecision.strong(1)),
        SearchCandidate("finish", PointerExpertDecision.weak(2)),
        SearchCandidate("greedy", PointerExpertDecision.strong(3)),
    )


class MacroUtilityTests(unittest.TestCase):
    def test_public_chain_potential_matches_level_one_group_formula(self) -> None:
        pair = _observation(
            bodies=[
                _body(1, x=100.0, lifecycle="confirmed", chain_id=7),
                _body(2, x=300.0, lifecycle="confirmed", chain_id=7),
            ]
        )
        triple = {
            **pair,
            "bodies": pair["bodies"]
            + [_body(4, x=200.0, lifecycle="confirmed", chain_id=7)],
        }
        self.assertEqual(chain_score_potential(pair), 16.0)
        self.assertEqual(chain_score_potential(triple), 54.0)
        value = macro_utility(
            _observation(bodies=[]),
            triple,
            [
                {"kind": int(EventKind.CHAIN_JOINED), "a": value}
                for value in (1, 2, 4)
            ],
            terminated=False,
            truncated=False,
        )
        self.assertGreater(value.total, 54.0)

    def test_compact_candidates_do_not_directly_target_confirmed_groups(self) -> None:
        observation = _observation(
            bodies=[
                _body(1, x=100.0, lifecycle="confirmed", chain_id=4),
                _body(2, x=200.0),
                _body(3, x=260.0),
            ]
        )
        candidates = generate_macro_candidates(
            observation, SpawnCensoredMacroBeamTeacher().pointer_spec
        )
        custom_targets = {
            candidate.decision.target_body_id
            for candidate in candidates
            if candidate.name.startswith("macro/")
        }
        self.assertNotIn(1, custom_targets)
        self.assertTrue({2, 3}.issubset(custom_targets))


class MacroBeamTests(unittest.TestCase):
    def test_two_action_plan_beats_greedy_one_action_and_returns_first_step(self) -> None:
        observation = _observation(interval=50)
        env = _MacroEnv(observation)
        before = env.clone_state()
        result = SpawnCensoredMacroBeamTeacher(
            max_depth=2,
            beam_width=3,
            max_candidates=3,
            max_rollout_ticks=10,
            settle_ticks=(0,),
            max_branch_evaluations=16,
            candidate_generator=_three_candidates,
        ).search(env, observation)
        self.assertEqual(result.decision, PointerExpertDecision.strong(1))
        self.assertEqual(
            result.selected_name,
            "macro-plan/setup;settle=0 > finish;settle=0",
        )
        winner = max(result.evaluations, key=lambda value: value.utility.total)
        self.assertEqual(winner.path[:2], ("setup;settle=0", "finish;settle=0"))
        self.assertEqual(winner.utility.score_delta, 100)
        self.assertEqual(env.clone_state(), before)
        self.assertGreater(env.restores, 0)

    def test_leaf_coast_ends_at_but_never_steps_from_spawn_boundary(self) -> None:
        observation = _observation(tick=7, interval=10)
        env = _MacroEnv(observation)
        before = env.clone_state()
        result = SpawnCensoredMacroBeamTeacher(
            max_depth=3,
            beam_width=2,
            max_candidates=1,
            max_rollout_ticks=10,
            settle_ticks=(0,),
            max_branch_evaluations=8,
            candidate_generator=lambda _observation, _spec: (
                SearchCandidate("setup", PointerExpertDecision.strong(1)),
            ),
        ).search(env, observation)
        self.assertEqual(result.safe_tick_budget, 3)
        self.assertEqual(result.evaluated_tick_budget, 3)
        self.assertEqual(
            {evaluation.primitive_ticks for evaluation in result.evaluations},
            {3},
        )
        self.assertEqual(env.max_tick, 10)
        self.assertEqual(env.clone_state(), before)

    def test_branch_budget_is_hard_and_result_is_deterministic(self) -> None:
        observation = _observation(interval=50)
        kwargs = dict(
            max_depth=4,
            beam_width=4,
            max_candidates=3,
            max_rollout_ticks=12,
            settle_ticks=(0,),
            max_branch_evaluations=5,
            candidate_generator=_three_candidates,
        )
        first = SpawnCensoredMacroBeamTeacher(**kwargs).search(
            _MacroEnv(observation), observation
        )
        second = SpawnCensoredMacroBeamTeacher(**kwargs).search(
            _MacroEnv(observation), observation
        )
        self.assertEqual(first.branches_evaluated, 5)
        self.assertEqual(first.decision, second.decision)
        self.assertEqual(first.selected_name, second.selected_name)

    def test_continuation_evaluator_can_override_public_leaf_utility(self) -> None:
        observation = _observation(interval=50)
        result = SpawnCensoredMacroBeamTeacher(
            max_depth=1,
            beam_width=3,
            max_candidates=3,
            max_rollout_ticks=4,
            settle_ticks=(0,),
            max_branch_evaluations=4,
            candidate_generator=_three_candidates,
            continuation_evaluator=lambda final: (
                1000.0 if int(final["score"]) == 0 else 0.0
            ),
            continuation_scale=1.0,
        ).search(_MacroEnv(observation), observation)
        self.assertEqual(result.decision, PointerExpertDecision.strong(1))
        winner = max(result.evaluations, key=lambda value: value.search_score)
        self.assertEqual(winner.continuation_value, 1000.0)

    def test_nonpositive_plan_uses_configured_fallback(self) -> None:
        observation = _observation(interval=50)
        result = SpawnCensoredMacroBeamTeacher(
            max_depth=1,
            beam_width=1,
            max_candidates=1,
            max_rollout_ticks=4,
            settle_ticks=(0,),
            candidate_generator=lambda _observation, _spec: (
                SearchCandidate("wait", PointerExpertDecision.wait(1)),
            ),
            fallback_teacher=lambda _observation, _spec: (
                PointerExpertDecision.wait(8)
            ),
        ).search(_MacroEnv(observation), observation)
        self.assertEqual(result.decision, PointerExpertDecision.wait(8))
        self.assertEqual(
            result.selected_name, "macro-fallback/nonpositive-plan"
        )

    def test_recurrent_context_is_forwarded_to_continuation_evaluator(self) -> None:
        evaluator = _ContextEvaluator()
        teacher = SpawnCensoredMacroBeamTeacher(
            continuation_evaluator=evaluator
        )
        state = object()
        teacher.set_recurrent_context(state)
        self.assertIs(evaluator.state, state)

    def test_source_is_restored_when_a_branch_crashes(self) -> None:
        observation = _observation(interval=50)
        env = _MacroEnv(observation, fail_cursor_x=100.0)
        before = env.clone_state()
        teacher = SpawnCensoredMacroBeamTeacher(
            max_depth=1,
            beam_width=1,
            max_candidates=1,
            settle_ticks=(0,),
            candidate_generator=lambda _observation, _spec: (
                SearchCandidate("failure", PointerExpertDecision.strong(1)),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic branch failure"):
            teacher.search(env, observation)
        self.assertEqual(env.clone_state(), before)

    def test_nonportable_backend_fails_closed(self) -> None:
        observation = _observation()
        env = _MacroEnv(observation)
        env.physics_backend = "exact"
        with self.assertRaisesRegex(ValueError, "portable"):
            SpawnCensoredMacroBeamTeacher().search(env, observation)


if __name__ == "__main__":
    unittest.main()
