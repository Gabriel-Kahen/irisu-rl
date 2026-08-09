from __future__ import annotations

import sys
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import Action, ActionKind, EventKind, IrisuEnv, NativeError, find_library
from irisu_pointer.action import PointerActionSpec
from irisu_pointer.experts import (
    PointerExpertDecision,
    SearchCandidate,
    expert_anchors,
    generate_candidates,
    target_body,
)
from irisu_pointer.search import (
    SpawnCensoredSearchTeacher,
    lower_expert_decision,
    ticks_before_next_spawn,
    utility_breakdown,
)


def _body(
    identifier: int,
    *,
    color: int = 1,
    x: float = 200.0,
    y: float = 100.0,
    lifecycle: str = "dynamic_fresh",
    rot_timer: int = 0,
    projectile_hits: int = 0,
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "lifecycle": lifecycle,
        "color": color,
        "x": x,
        "y": y,
        "vx": 0.0,
        "vy": 0.0,
        "size": 40.0,
        "chain_id": 0,
        "projectile_hits": projectile_hits,
        "age_ticks": 10,
        "remaining_lifetime": 100,
        "rot_timer": rot_timer,
    }


def _observation(
    tick: int = 1,
    *,
    interval: int = 10,
    score: int = 0,
    gauge: int = 100,
    bodies: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, Any]:
    return {
        "tick": tick,
        "score": score,
        "gauge": gauge,
        "gauge_max": 100,
        "terminated": False,
        "truncated": False,
        "difficulty": {"spawn_interval_ticks": interval},
        "field": {"x": 16.0, "width": 576.0},
        "bodies": list(bodies or ()),
    }


class PointerExpertTests(unittest.TestCase):
    def test_ticks_before_next_spawn_respects_scene_entry_boundary(self) -> None:
        self.assertEqual(ticks_before_next_spawn(_observation(0, interval=90)), 0)
        self.assertEqual(ticks_before_next_spawn(_observation(1, interval=90)), 89)
        self.assertEqual(ticks_before_next_spawn(_observation(89, interval=90)), 1)
        self.assertEqual(ticks_before_next_spawn(_observation(90, interval=90)), 0)

    def test_all_anchors_are_id_targeted_and_always_present(self) -> None:
        observation = _observation(
            bodies=[
                _body(8, x=150.0, y=60.0),
                _body(3, x=160.0, y=180.0),
                _body(9, color=2, x=500.0, y=120.0, rot_timer=20),
            ]
        )
        anchors = expert_anchors(observation)
        self.assertEqual(
            [candidate.name for candidate in anchors],
            [
                "anchor/matcher",
                "anchor/direct",
                "anchor/eject",
                "anchor/hazard",
            ],
        )
        self.assertEqual(anchors[0].decision.target_body_id, 3)
        self.assertEqual(anchors[1].decision.target_body_id, 3)
        self.assertEqual(anchors[2].decision.target_body_id, 9)
        self.assertEqual(anchors[3].decision.target_body_id, 9)
        self.assertEqual(
            generate_candidates(observation)[:4],
            anchors,
        )

    def test_candidate_generation_is_deterministic(self) -> None:
        observation = _observation(
            bodies=[_body(5, rot_timer=10), _body(2, color=2, x=300.0)]
        )
        first = generate_candidates(observation, max_target_bodies=2)
        second = generate_candidates(observation, max_target_bodies=2)
        self.assertEqual(first, second)
        names = [candidate.name for candidate in first]
        self.assertEqual(len(names), len(set(names)))

    def test_matcher_strength_does_not_depend_on_absolute_board_y(self) -> None:
        high = _observation(
            bodies=[_body(1, y=40.0), _body(2, x=260.0, y=120.0)]
        )
        low = _observation(
            bodies=[_body(1, y=220.0), _body(2, x=260.0, y=300.0)]
        )
        for observation in (high, low):
            decision = expert_anchors(observation)[0].decision
            self.assertEqual(
                ActionKind.parse(decision.kind), ActionKind.STRONG_SHOT
            )

    def test_experts_are_invariant_to_body_id_renumbering(self) -> None:
        bodies = [
            _body(90, color=1, x=100.0, y=80.0, rot_timer=3),
            _body(2, color=1, x=200.0, y=80.0, rot_timer=7),
            _body(70, color=1, x=300.0, y=80.0, rot_timer=1),
            _body(4, color=2, x=500.0, y=80.0, rot_timer=20),
        ]
        renumbered = [
            {**body, "id": replacement}
            for body, replacement in zip(bodies, (1, 800, 3, 600), strict=True)
        ]
        first_observation = _observation(bodies=bodies)
        second_observation = _observation(bodies=renumbered)

        def signature(
            candidate: SearchCandidate, observation: Mapping[str, Any]
        ) -> tuple[object, ...]:
            decision = candidate.decision
            if decision.target_body_id is None:
                geometry = None
            else:
                body = target_body(observation, decision.target_body_id)
                assert body is not None
                geometry = (
                    body["lifecycle"],
                    body["color"],
                    body["x"],
                    body["y"],
                    body["vx"],
                    body["vy"],
                    body["size"],
                    body["rot_timer"],
                )
            return (
                decision.kind,
                decision.wait_ticks,
                decision.template_index,
                geometry,
            )

        self.assertEqual(
            [
                signature(candidate, first_observation)
                for candidate in expert_anchors(first_observation)
            ],
            [
                signature(candidate, second_observation)
                for candidate in expert_anchors(second_observation)
            ],
        )
        self.assertEqual(
            [
                signature(candidate, first_observation)
                for candidate in generate_candidates(
                    first_observation, max_target_bodies=4
                )
            ],
            [
                signature(candidate, second_observation)
                for candidate in generate_candidates(
                    second_observation, max_target_bodies=4
                )
            ],
        )

    def test_lowering_uses_body_id_radius_offset_and_release_tick(self) -> None:
        spec = PointerActionSpec()
        observation = _observation(bodies=[_body(7, x=200.0)])
        # First x offset (-one radius) and middle y-radius offset.
        actions = lower_expert_decision(
            PointerExpertDecision.weak(7, template_index=1),
            observation,
            spec,
        )
        self.assertEqual(len(actions), 2)
        self.assertEqual(ActionKind.parse(actions[0].kind), ActionKind.WEAK_SHOT)
        self.assertEqual(actions[0].cursor_x, 180.0)
        self.assertEqual(actions[0].cursor_y, 140.0)
        self.assertEqual(actions[1], Action.wait(1))


class UtilityTests(unittest.TestCase):
    def test_public_utility_rewards_progress_and_penalizes_destructive_hit(self) -> None:
        initial = _observation(
            score=100,
            gauge=50,
            bodies=[
                _body(
                    4,
                    lifecycle="confirmed",
                    projectile_hits=1,
                )
            ],
        )
        final = {**initial, "score": 250, "gauge": 45}
        events = [
            {"kind": int(EventKind.CHAIN_JOINED), "a": 4, "b": 0, "value": 2},
            {"kind": int(EventKind.CLEARED), "a": 4, "b": 0, "value": 2},
            {
                "kind": int(EventKind.PROJECTILE_HIT),
                "a": 20,
                "b": 4,
                "value": 1,
            },
            {"kind": int(EventKind.EJECTED), "a": 5, "b": 0, "value": 0},
            {"kind": int(EventKind.ROTTEN), "a": 6, "b": 0, "value": 0},
        ]
        value = utility_breakdown(
            initial,
            final,
            events,
            terminated=False,
            truncated=False,
        )
        self.assertEqual(value.score_delta, 150)
        self.assertEqual(value.gauge_delta, -5)
        self.assertEqual(value.confirmed_body_destructive_hit, 1)
        self.assertEqual(value.chain_joined, 1)
        self.assertEqual(value.cleared, 1)
        self.assertEqual(value.rotten, 1)
        without_destructive_contact = utility_breakdown(
            initial,
            final,
            events[:-3],
            terminated=False,
            truncated=False,
        )
        self.assertGreater(without_destructive_contact.total, value.total)

    def test_sustained_projectile_contact_counts_one_unique_hit(self) -> None:
        initial = _observation(
            gauge=80,
            bodies=[_body(4), _body(5, x=300.0)],
        )
        final = {**initial, "gauge": 70}
        events = [
            {
                "kind": int(EventKind.PROJECTILE_HIT),
                "a": 20,
                "b": 4,
                "value": 0,
            }
            for _ in range(50)
        ]
        events.append(
            {
                "kind": int(EventKind.PROJECTILE_HIT),
                "a": 20,
                "b": 5,
                "value": 0,
            }
        )
        value = utility_breakdown(
            initial,
            final,
            events,
            terminated=False,
            truncated=False,
        )
        self.assertEqual(value.projectile_hit, 2)
        self.assertAlmostEqual(value.gauge_fraction_delta, -0.1)


class _Branch:
    def __init__(
        self,
        observation: Mapping[str, Any],
        owner: _FakeExactEnv,
    ) -> None:
        self.observation = dict(observation)
        self.owner = owner
        self.executed_ticks = 0

    def __enter__(self) -> _Branch:
        return self

    def __exit__(self, *_: object) -> None:
        if hasattr(self.owner, "branch_tick_counts"):
            self.owner.branch_tick_counts.append(self.executed_ticks)
        return None

    def step(
        self, action: Action
    ) -> tuple[dict[str, Any], int, bool, bool, dict[str, Any]]:
        kind = ActionKind.parse(action.kind)
        ticks = int(action.wait_ticks) if kind is ActionKind.WAIT else 1
        interval = int(self.observation["difficulty"]["spawn_interval_ticks"])
        for _ in range(ticks):
            # This assertion makes a future-spawn peek fail the test immediately.
            if int(self.observation["tick"]) % interval == 0:
                raise AssertionError("search evaluated the cadence-spawn tick")
            self.observation["tick"] = int(self.observation["tick"]) + 1
        score_delta = 20 if kind is ActionKind.STRONG_SHOT else 0
        self.observation["score"] = int(self.observation["score"]) + score_delta
        events: list[dict[str, int]] = []
        if kind is ActionKind.STRONG_SHOT:
            events.append(
                {"kind": int(EventKind.EJECTED), "a": 1, "b": 0, "value": 0}
            )
        self.owner.branch_steps += ticks
        self.executed_ticks += ticks
        return dict(self.observation), score_delta, False, False, {"events": events}


class _Checkpoint:
    def __init__(self, owner: _FakeExactEnv) -> None:
        self.owner = owner

    def __enter__(self) -> _Checkpoint:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def branch(self) -> _Branch:
        self.owner.branches += 1
        return _Branch(self.owner.observation, self.owner)


class _FakeExactEnv:
    physics_backend = "exact"

    def __init__(self, observation: Mapping[str, Any]) -> None:
        self.observation = dict(observation)
        self.branches = 0
        self.branch_steps = 0
        self.branch_tick_counts: list[int] = []
        self.fast_checkpoints = 0

    def fast_checkpoint(self) -> _Checkpoint:
        self.fast_checkpoints += 1
        return _Checkpoint(self)

    def step(self, action: Action) -> object:
        del action
        raise AssertionError("exact search must never step its source environment")

    def state_hash(self) -> int:
        raise AssertionError("search must not inspect a state hash")


class _FakePortableEnv(_Branch):
    physics_backend = "portable"

    def __init__(self, observation: Mapping[str, Any]) -> None:
        self.source_observation = dict(observation)
        self.restores = 0
        self.branch_steps = 0
        self.branches = 0
        super().__init__(self.source_observation, self)

    def clone_state(self) -> tuple[tuple[str, object], ...]:
        return tuple(sorted(self.observation.items()))

    def restore_state(
        self, snapshot: tuple[tuple[str, object], ...]
    ) -> dict[str, Any]:
        self.restores += 1
        self.observation = dict(snapshot)
        return dict(self.observation)

    def state_hash(self) -> int:
        raise AssertionError("search must not inspect a state hash")


class SearchTeacherTests(unittest.TestCase):
    def test_exact_uses_fast_branches_and_censors_long_candidate(self) -> None:
        observation = _observation(
            tick=7,
            interval=10,
            bodies=[_body(1)],
        )
        candidates = (
            SearchCandidate("safe/strong", PointerExpertDecision.strong(1)),
            SearchCandidate("unsafe/wait", PointerExpertDecision.wait(4)),
        )
        teacher = SpawnCensoredSearchTeacher(
            candidate_generator=lambda _observation, _spec: candidates
        )
        env = _FakeExactEnv(observation)
        result = teacher.search(env, observation)
        self.assertEqual(result.safe_tick_budget, 3)
        self.assertEqual(result.evaluated_tick_budget, 3)
        self.assertEqual(result.selected_name, "safe/strong")
        self.assertEqual(env.fast_checkpoints, 1)
        self.assertEqual(env.branches, 1)
        self.assertEqual(env.branch_steps, 3)
        self.assertEqual(env.observation, observation)

    def test_long_cadence_caps_rollout_and_candidate_count_at_64(self) -> None:
        observation = _observation(
            tick=1,
            interval=1_000,
            bodies=[
                _body(
                    identifier,
                    color=identifier,
                    x=80.0 + identifier * 55.0,
                )
                for identifier in range(1, 9)
            ],
        )
        env = _FakeExactEnv(observation)
        result = SpawnCensoredSearchTeacher().search(env, observation)
        self.assertEqual(result.safe_tick_budget, 999)
        self.assertEqual(result.evaluated_tick_budget, 64)
        self.assertEqual(len(result.evaluations), 64)
        self.assertEqual(env.branches, 64)
        self.assertEqual(env.branch_tick_counts, [64] * 64)
        names = {evaluation.candidate.name for evaluation in result.evaluations}
        self.assertTrue(
            {
                "anchor/matcher",
                "anchor/direct",
                "anchor/eject",
                "anchor/hazard",
                "wait/1",
                "wait/2",
                "wait/4",
                "wait/8",
                "wait/16",
            }.issubset(names)
        )
        shots = [
            evaluation.candidate.decision
            for evaluation in result.evaluations
            if evaluation.candidate.name.startswith("shot/")
        ]
        self.assertEqual(
            {decision.target_body_id for decision in shots},
            set(range(1, 9)),
        )
        self.assertEqual(
            {decision.kind for decision in shots},
            {int(ActionKind.WEAK_SHOT), int(ActionKind.STRONG_SHOT)},
        )
        templates = PointerActionSpec().templates
        selected_templates = {
            templates[decision.template_index] for decision in shots
        }
        self.assertGreaterEqual(
            len({offset for offset, _ in selected_templates}), 3
        )
        self.assertEqual(
            len({height for _, height in selected_templates}), 3
        )

    def test_search_caps_require_positive_operational_values(self) -> None:
        with self.assertRaises(ValueError):
            SpawnCensoredSearchTeacher(max_rollout_ticks=0)
        with self.assertRaises(ValueError):
            SpawnCensoredSearchTeacher(max_candidates=8)

    def test_exact_ties_use_candidate_order(self) -> None:
        observation = _observation(tick=1, interval=10)
        candidates = (
            SearchCandidate("first", PointerExpertDecision.wait(1)),
            SearchCandidate("second", PointerExpertDecision.wait(1)),
        )
        result = SpawnCensoredSearchTeacher(
            candidate_generator=lambda _observation, _spec: candidates
        ).search(_FakeExactEnv(observation), observation)
        self.assertEqual(result.selected_name, "first")

    def test_continuation_critic_can_override_myopic_immediate_utility(self) -> None:
        observation = _observation(
            tick=1, interval=10, bodies=[_body(1)]
        )
        candidates = (
            SearchCandidate("myopic/strong", PointerExpertDecision.strong(1)),
            SearchCandidate("patient/wait", PointerExpertDecision.wait(1)),
        )
        result = SpawnCensoredSearchTeacher(
            candidate_generator=lambda _observation, _spec: candidates,
            continuation_evaluator=lambda final: (
                100.0 if int(final["score"]) == 0 else 0.0
            ),
        ).search(_FakeExactEnv(observation), observation)
        self.assertEqual(result.selected_name, "patient/wait")
        self.assertGreater(
            result.evaluations[1].search_score,
            result.evaluations[0].search_score,
        )

    def test_portable_search_restores_source_and_never_hashes(self) -> None:
        observation = _observation(tick=2, interval=10)
        env = _FakePortableEnv(observation)
        snapshot = env.clone_state()
        result = SpawnCensoredSearchTeacher(
            candidate_generator=lambda _observation, _spec: (
                SearchCandidate("wait", PointerExpertDecision.wait(2)),
            )
        ).search(env, observation)
        self.assertEqual(result.selected_name, "wait")
        self.assertEqual(env.clone_state(), snapshot)
        self.assertGreaterEqual(env.restores, 2)

    def test_boundary_returns_censored_fallback_without_branching(self) -> None:
        observation = _observation(tick=10, interval=10)
        env = _FakeExactEnv(observation)
        result = SpawnCensoredSearchTeacher().search(env, observation)
        self.assertEqual(result.selected_name, "censored/no-safe-candidate")
        self.assertEqual(result.evaluations, ())
        self.assertEqual(result.evaluated_tick_budget, 0)
        self.assertEqual(env.fast_checkpoints, 0)

    def test_real_portable_one_wait_smoke(self) -> None:
        try:
            library = find_library()
        except NativeError as exc:
            self.skipTest(str(exc))
        with IrisuEnv(library_path=library, physics_backend="portable") as env:
            observation, _ = env.reset(seed=7)
            observation, _, _, _, _ = env.step(Action.wait(1))
            before = env.clone_state()
            result = SpawnCensoredSearchTeacher(
                candidate_generator=lambda _observation, _spec: (
                    SearchCandidate("wait", PointerExpertDecision.wait(1)),
                )
            ).search(env, observation)
            self.assertEqual(result.selected_name, "wait")
            self.assertEqual(env.clone_state(), before)


if __name__ == "__main__":
    unittest.main()
