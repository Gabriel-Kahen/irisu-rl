from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import Action, ActionKind
from irisu_pointer.action import PointerActionSpec, decode_pointer_action
from irisu_pointer.archive import StrategicArchive
from irisu_pointer.archive_improvement import (
    ArchiveBranchOutcome,
    ArchiveImprovementBinding,
    ArchiveImprovementConfig,
    SteeringBranchCandidate,
    collect_archive_improvement,
)
from irisu_pointer.steering import (
    SteeringDecision,
    SteeringExpertConfig,
    SteeringIntent,
)
from irisu_rl.actions import ActionSpec, SemanticAction, SemanticActionKind


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _body(identifier: int, color: int, x: float, y: float) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": "dynamic_fresh",
        "color": color,
        "x": x,
        "y": y,
        "vx": 0.0,
        "vy": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 40.0,
        "chain_id": 0,
        "projectile_hits": 0,
        "age_ticks": 20,
        "remaining_lifetime": 1000,
        "rot_timer": 0,
    }


def _observation() -> dict[str, Any]:
    return {
        "tick": 25,
        "score": 0,
        "gauge": 1000,
        "gauge_max": 1000,
        "level": 1,
        "highest_chain": 0,
        "qualifying_clear_count": 0,
        "left_held": False,
        "right_held": False,
        "terminated": False,
        "truncated": False,
        "field": {"x": 0.0, "y": 0.0, "width": 640.0, "height": 480.0},
        "difficulty": {"active_colors": 2, "spawn_interval_ticks": 100},
        "bodies": [
            _body(1, 0, 160.0, 210.0),
            _body(2, 0, 250.0, 190.0),
            _body(3, 1, 400.0, 220.0),
            _body(4, 1, 500.0, 180.0),
        ],
    }


class _BranchEnv:
    physics_backend = "portable"
    runtime_sha256 = _sha("runtime")
    config = {"fixture": "archive-improvement"}

    def __init__(
        self,
        observation: dict[str, Any],
        *,
        reward_wait: bool = False,
        raise_on_step: bool = False,
        reward_cursor_x: int = 150,
        invalid_cursor_x: int | None = None,
    ) -> None:
        self.initial = copy.deepcopy(observation)
        self.observation = copy.deepcopy(observation)
        self.reward_wait = reward_wait
        self.raise_on_step = raise_on_step
        self.reward_cursor_x = reward_cursor_x
        self.invalid_cursor_x = invalid_cursor_x
        self.restore_count = 0
        self.closed = False

    def runner_identity_manifest(self) -> dict[str, object]:
        return {
            "environment": "fake-portable",
            "config": self.config,
        }

    def restore_state(self, snapshot: bytes) -> dict[str, Any]:
        if snapshot != b"elite-state":
            raise ValueError("unknown fake snapshot")
        self.restore_count += 1
        self.observation = copy.deepcopy(self.initial)
        return copy.deepcopy(self.observation)

    def step(
        self, action: Action
    ) -> tuple[dict[str, Any], int, bool, bool, dict[str, object]]:
        if self.raise_on_step:
            raise RuntimeError("synthetic branch failure")
        kind = ActionKind.parse(action.kind)
        start_tick = int(self.observation["tick"])
        duration = int(action.wait_ticks) if kind is ActionKind.WAIT else 1
        reward = 0
        if (
            start_tick == 25
            and kind is ActionKind.STRONG_SHOT
            and int(action.cursor_x) == self.reward_cursor_x
            and not self.reward_wait
        ):
            reward = 100
            self.observation["score"] = 100
            self.observation["highest_chain"] = 3
            self.observation["qualifying_clear_count"] = 1
        if (
            start_tick == 25
            and kind is ActionKind.WAIT
            and int(action.wait_ticks) == 8
            and self.reward_wait
        ):
            reward = 75
            self.observation["score"] = 75
        self.observation["tick"] = start_tick + duration
        self.observation["gauge"] = max(
            0, int(self.observation["gauge"]) - duration
        )
        invalid = (
            kind is ActionKind.STRONG_SHOT
            and int(action.cursor_x) == self.invalid_cursor_x
        )
        return (
            copy.deepcopy(self.observation),
            reward,
            False,
            False,
            {"invalid_action": invalid, "events": ()},
        )

    def close(self) -> None:
        self.closed = True


class _ExactEnv(_BranchEnv):
    physics_backend = "exact"


def _archive() -> StrategicArchive:
    archive = StrategicArchive(source_identity=_sha("source"))
    _elite, inserted = archive.capture(
        _observation(),
        b"elite-state",
        trajectory_identity="fixture:00000001:25:pre-shot",
    )
    assert inserted
    return archive


def _binding(archive: StrategicArchive) -> ArchiveImprovementBinding:
    return ArchiveImprovementBinding.create(
        archive,
        environment_config_sha256=_json_sha(_BranchEnv.config),
        runner_identity_sha256=_json_sha(
            _BranchEnv(_observation()).runner_identity_manifest()
        ),
        runtime_sha256=_sha("runtime"),
    )


def _shot_config() -> ArchiveImprovementConfig:
    baseline = SteeringExpertConfig()
    return ArchiveImprovementConfig(
        horizon_ticks=4,
        max_elites=1,
        wait_candidates=(4,),
        steering_branches=(
            SteeringBranchCandidate("baseline", baseline),
            SteeringBranchCandidate(
                "close-contact", replace(baseline, impact_side_sizes=0.25)
            ),
        ),
    )


class ArchiveImprovementTests(unittest.TestCase):
    def test_score_gain_cannot_trade_away_baseline_survival(self) -> None:
        decision = SteeringDecision(
            SemanticAction.wait(4),
            SteeringIntent.WAIT,
            reason="fixture",
        )
        baseline = ArchiveBranchOutcome(
            "steering/baseline", 0, decision, 10, True, 128, 900, 1, 1, 0
        )
        lower_gauge = ArchiveBranchOutcome(
            "steering/risky", 1, decision, 100, True, 128, 899, 2, 2, 0
        )
        safe_gain = ArchiveBranchOutcome(
            "steering/safe", 2, decision, 100, True, 128, 901, 2, 2, 0
        )
        self.assertFalse(lower_gauge.survival_nondominated_by(baseline))
        self.assertTrue(safe_gain.survival_nondominated_by(baseline))

    def test_selects_raw_score_branch_and_emits_bound_shot_label(self) -> None:
        archive = _archive()
        environments: list[_BranchEnv] = []

        def factory() -> _BranchEnv:
            value = _BranchEnv(_observation())
            environments.append(value)
            return value

        first = collect_archive_improvement(
            archive,
            factory,
            binding=_binding(archive),
            config=_shot_config(),
        )
        second = collect_archive_improvement(
            archive,
            factory,
            binding=_binding(archive),
            config=_shot_config(),
        )
        self.assertEqual(first.report.sha256, second.report.sha256)
        self.assertEqual(first.report.branch_count, 3)
        self.assertEqual(first.report.selectable_branch_count, 3)
        self.assertEqual(first.report.total_selected_score_gain, 100)
        self.assertEqual(first.report.maximum_selected_score_gain, 100)
        self.assertEqual(first.report.strict_improvement_count, 1)
        self.assertEqual(
            first.report.selections[0].selected_candidate,
            "steering/close-contact",
        )
        self.assertEqual(len(first.examples), 1)
        self.assertTrue(first.examples[0].is_shot)
        self.assertEqual(
            first.examples[0].provenance_sha256,
            first.report.selections[0].provenance_sha256,
        )
        for env in environments:
            self.assertTrue(env.closed)
            self.assertEqual(env.observation, env.initial)
            self.assertEqual(env.restore_count, 5)

    def test_wide_winner_roundtrips_to_the_exact_labeled_action(self) -> None:
        archive = _archive()
        baseline = SteeringExpertConfig()
        config = ArchiveImprovementConfig(
            horizon_ticks=4,
            max_elites=1,
            wait_candidates=(4,),
            steering_branches=(
                SteeringBranchCandidate("baseline", baseline),
                SteeringBranchCandidate(
                    "wide-contact",
                    replace(baseline, impact_side_sizes=0.75),
                ),
            ),
        )
        result = collect_archive_improvement(
            archive,
            lambda: _BranchEnv(_observation(), reward_cursor_x=130),
            binding=_binding(archive),
            config=config,
        )
        selection = result.report.selections[0]
        self.assertEqual(selection.selected_candidate, "steering/wide-contact")
        self.assertTrue(selection.strictly_improved)
        self.assertEqual(len(result.examples), 1)
        example = result.examples[0]
        decoded = decode_pointer_action(
            kind=int(SemanticActionKind.FIRE_STRONG),
            template_index=example.template_index,
            selected_body_row=example.observation.body_features[
                0, example.source_index
            ],
            schema=example.observation.schema,
            pointer_spec=PointerActionSpec(),
            action_spec=ActionSpec(),
        )
        winner = next(
            value
            for value in selection.outcomes
            if value.candidate_name == "steering/wide-contact"
        )
        self.assertAlmostEqual(decoded.x_norm, winner.first_decision.action.x_norm)
        self.assertAlmostEqual(decoded.y_norm, winner.first_decision.action.y_norm)

    def test_incumbent_tie_is_reported_but_not_emitted(self) -> None:
        archive = _archive()
        result = collect_archive_improvement(
            archive,
            lambda: _BranchEnv(_observation(), reward_cursor_x=-1),
            binding=_binding(archive),
            config=_shot_config(),
        )
        selection = result.report.selections[0]
        self.assertEqual(selection.selected_candidate, "steering/baseline")
        self.assertFalse(selection.strictly_improved)
        self.assertFalse(selection.emitted_example)
        self.assertEqual(selection.score_advantage_over_incumbent, 0)
        self.assertEqual(result.examples, ())
        self.assertEqual(result.report.strict_improvement_count, 0)

    def test_missing_selectable_incumbent_fails_closed(self) -> None:
        archive = _archive()
        result = collect_archive_improvement(
            archive,
            lambda: _BranchEnv(
                _observation(),
                reward_cursor_x=150,
                invalid_cursor_x=140,
            ),
            binding=_binding(archive),
            config=_shot_config(),
        )
        selection = result.report.selections[0]
        self.assertIsNone(selection.incumbent_candidate)
        self.assertEqual(selection.selected_candidate, "steering/close-contact")
        self.assertFalse(selection.strictly_improved)
        self.assertFalse(selection.emitted_example)
        self.assertEqual(result.examples, ())

    def test_deliberate_wait_can_win_and_become_restraint_supervision(self) -> None:
        archive = _archive()
        environments: list[_BranchEnv] = []
        baseline = SteeringExpertConfig()
        config = ArchiveImprovementConfig(
            horizon_ticks=8,
            max_elites=1,
            wait_candidates=(8,),
            steering_branches=(SteeringBranchCandidate("baseline", baseline),),
        )

        def factory() -> _BranchEnv:
            value = _BranchEnv(_observation(), reward_wait=True)
            environments.append(value)
            return value

        result = collect_archive_improvement(
            archive,
            factory,
            binding=_binding(archive),
            config=config,
        )
        self.assertEqual(result.report.selections[0].selected_candidate, "wait/8")
        self.assertEqual(result.report.total_selected_score_gain, 75)
        self.assertEqual(len(result.examples), 1)
        self.assertFalse(result.examples[0].is_shot)
        self.assertEqual(result.examples[0].wait_index, 3)
        self.assertTrue(environments[0].closed)
        self.assertEqual(environments[0].observation, environments[0].initial)

    def test_archive_binding_and_portable_backend_fail_closed(self) -> None:
        archive = _archive()
        bad = replace(_binding(archive), archive_sha256="f" * 64)
        called = False

        def unused_factory() -> _BranchEnv:
            nonlocal called
            called = True
            return _BranchEnv(_observation())

        with self.assertRaisesRegex(ValueError, "does not match"):
            collect_archive_improvement(
                archive, unused_factory, binding=bad, config=_shot_config()
            )
        self.assertFalse(called)

        environments: list[_ExactEnv] = []

        def exact_factory() -> _ExactEnv:
            value = _ExactEnv(_observation())
            environments.append(value)
            return value

        with self.assertRaisesRegex(ValueError, "portable"):
            collect_archive_improvement(
                archive,
                exact_factory,
                binding=_binding(archive),
                config=_shot_config(),
            )
        self.assertTrue(environments[0].closed)

        unsafe = StrategicArchive(source_identity=_sha("unsafe-source"))
        unsafe.capture(
            _observation(),
            b"elite-state",
            trajectory_identity="post-shot-state",
        )
        with self.assertRaisesRegex(ValueError, "pre-shot safe boundary"):
            collect_archive_improvement(
                unsafe,
                unused_factory,
                binding=_binding(unsafe),
                config=_shot_config(),
            )

    def test_branch_exception_restores_and_closes_environment(self) -> None:
        archive = _archive()
        environments: list[_BranchEnv] = []

        def factory() -> _BranchEnv:
            value = _BranchEnv(_observation(), raise_on_step=True)
            environments.append(value)
            return value

        with self.assertRaisesRegex(RuntimeError, "synthetic branch failure"):
            collect_archive_improvement(
                archive,
                factory,
                binding=_binding(archive),
                config=_shot_config(),
            )
        self.assertTrue(environments[0].closed)
        self.assertEqual(environments[0].observation, environments[0].initial)


if __name__ == "__main__":
    unittest.main()
