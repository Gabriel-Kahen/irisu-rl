from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import unittest

import torch

from irisu_rl.actions import ActionSpec, SemanticAction
from irisu_rl.curriculum import (
    CurriculumCoordinator,
    CurriculumSpec,
    SnapshotLibrary,
    SnapshotRecipe,
    StageSpec,
    ValidationEpisodeOutcome,
    ValidationReport,
    ValidationResult,
)
from irisu_rl.rewards import RewardKnot, RewardSchedule


def recipe(stage: str, split: str, family: str, suffix: str) -> SnapshotRecipe:
    action = ActionSpec()
    payload = b"snapshot-" + suffix.encode()
    identity = sum(suffix.encode())
    return SnapshotRecipe(
        f"{stage}-{split}-{suffix}",
        stage,
        split,
        family,
        "pool-a",
        "1" * 64,
        9,
        7 + identity,
        action.sha256,
        (action.serialize(SemanticAction.wait(1)).hex(),),
        1,
        0,
        11 + identity,
        hashlib.sha256(payload).hexdigest(),
        "2" * 64,
        "fixture-v1",
    )


def curriculum() -> CurriculumSpec:
    recipes = (
        recipe("wait", "train", "wait-train", "a"),
        recipe("wait", "validation", "wait-val", "b"),
        recipe("shot", "train", "shot-train", "c"),
        recipe("shot", "validation", "shot-val", "d"),
    )
    schedule0 = RewardSchedule(
        "wait-reward-v1", (RewardKnot(0, 500_000), RewardKnot(10, 0))
    )
    schedule1 = RewardSchedule("shot-reward-v1", (RewardKnot(0, 0),))
    stages = (
        StageSpec(
            "wait",
            0,
            "pool-a",
            (recipes[0].snapshot_id,),
            (recipes[1].snapshot_id,),
            (0,),
            (1, 2),
            8,
            10,
            7,
            10,
            2,
            20,
            schedule0,
        ),
        StageSpec(
            "shot",
            1,
            "pool-a",
            (recipes[2].snapshot_id,),
            (recipes[3].snapshot_id,),
            (0, 1, 2),
            (1, 2),
            8,
            10,
            7,
            10,
            1,
            20,
            schedule1,
        ),
    )
    return CurriculumSpec("fixture-v1", SnapshotLibrary(recipes), stages, 0xC011EC7)


def validation_report(
    coordinator: CurriculumCoordinator,
    policy_sha256: str,
    results: tuple[ValidationResult, ...],
) -> ValidationReport:
    request = coordinator.request_validation(
        policy_sha256=policy_sha256, evaluator_identity_sha256="e" * 64
    )
    requested = {stage.stage_id: stage for stage in request.stages}
    bound_results = tuple(
        bind_result(result, request, requested[result.stage_id]) for result in results
    )
    return ValidationReport(
        request.request_id,
        policy_sha256,
        request.evaluator_identity_sha256,
        bound_results,
    )


def bind_result(result: ValidationResult, validation_request, stage_request):
    outcomes = tuple(
        ValidationEpisodeOutcome(
            stage_request.snapshot_ids[index % len(stage_request.snapshot_ids)],
            index // len(stage_request.snapshot_ids),
            validation_request.episode_seed(
                result.stage_id,
                stage_request.snapshot_ids[index % len(stage_request.snapshot_ids)],
                index // len(stage_request.snapshot_ids),
            ),
            index < result.successes,
        )
        for index in range(stage_request.episodes)
    )
    return replace(
        result,
        snapshot_ids=stage_request.snapshot_ids,
        outcomes=outcomes,
    )


class R3ACurriculumTests(unittest.TestCase):
    def test_assignment_sampling_is_transactional_and_resume_exact(self) -> None:
        spec = curriculum()
        first = CurriculumCoordinator(spec, 4, learner_seed=19)
        with self.assertRaises(TypeError):
            first.reserve_assignments((True,))
        reservation = first.reserve_assignments((3, 1))
        with self.assertRaisesRegex(RuntimeError, "outstanding"):
            first.reserve_assignments((0,))
        with self.assertRaisesRegex(ValueError, "stale"):
            first.commit_assignments(replace(reservation, state_hash="0" * 64))
        first.commit_assignments(reservation)
        state = first.state_dict()

        restored = CurriculumCoordinator(spec, 4, learner_seed=19)
        restored.load_state_dict(state)
        self.assertEqual(
            first.reserve_assignments((0, 1, 2, 3)),
            restored.reserve_assignments((0, 1, 2, 3)),
        )

    def test_promotion_requires_consecutive_passes_and_regression_remediates(
        self,
    ) -> None:
        coordinator = CurriculumCoordinator(curriculum(), 2, learner_seed=3)
        passing = ValidationResult("wait", 8, 10)
        first = coordinator.record_validation(
            validation_report(coordinator, "a" * 64, (passing,))
        )
        self.assertFalse(first.promoted)
        promoted = coordinator.record_validation(
            validation_report(coordinator, "b" * 64, (passing,))
        )
        self.assertTrue(promoted.promoted)
        self.assertEqual(promoted.highest_unlocked_stage, "shot")
        coordinator.activate_focus_for_new_episodes(torch.ones(2, dtype=torch.bool))

        regression = coordinator.record_validation(
            validation_report(
                coordinator,
                "c" * 64,
                (ValidationResult("wait", 6, 10), ValidationResult("shot", 9, 10)),
            )
        )
        self.assertEqual(regression.phase, "remediation")
        self.assertEqual(regression.remediation_stage, "wait")
        self.assertEqual(regression.highest_unlocked_stage, "shot")
        recovered = coordinator.record_validation(
            validation_report(
                coordinator,
                "d" * 64,
                (ValidationResult("wait", 8, 10), ValidationResult("shot", 9, 10)),
            )
        )
        self.assertEqual(recovered.phase, "complete")

    def test_duplicate_reports_are_idempotent_but_conflicts_fail(self) -> None:
        coordinator = CurriculumCoordinator(curriculum(), 1, learner_seed=4)
        report = validation_report(
            coordinator, "a" * 64, (ValidationResult("wait", 1, 10),)
        )
        coordinator.record_validation(report)
        replay = coordinator.record_validation(report)
        self.assertEqual(replay.reason, "idempotent replay")
        with self.assertRaisesRegex(ValueError, "conflicting"):
            coordinator.record_validation(
                ValidationReport(
                    report.request_id,
                    "b" * 64,
                    "e" * 64,
                    (
                        ValidationResult(
                            "wait",
                            2,
                            10,
                            report.results[0].snapshot_ids,
                        ),
                    ),
                )
            )

    def test_one_policy_identity_cannot_be_reused_for_another_gate(self) -> None:
        coordinator = CurriculumCoordinator(curriculum(), 1, learner_seed=14)
        policy = "a" * 64
        coordinator.record_validation(
            validation_report(coordinator, policy, (ValidationResult("wait", 8, 10),))
        )
        with self.assertRaisesRegex(ValueError, "cannot satisfy multiple"):
            coordinator.request_validation(
                policy_sha256=policy, evaluator_identity_sha256="e" * 64
            )

    def test_action_masks_and_episode_stable_stage_assignment(self) -> None:
        coordinator = CurriculumCoordinator(curriculum(), 2, learner_seed=5)
        initial_weights = coordinator.shaping_weights_ppm()
        coordinator.advance_update()
        # Advancing the schedule cannot change a weight mid-episode.
        torch.testing.assert_close(coordinator.shaping_weights_ppm(), initial_weights)
        coordinator.activate_focus_for_new_episodes(torch.tensor([False, True]))
        self.assertLess(
            int(coordinator.shaping_weights_ppm()[1]),
            int(coordinator.shaping_weights_ppm()[0]),
        )
        kind, wait = coordinator.action_masks(ActionSpec())
        self.assertTrue(torch.all(kind[:, 0]))
        self.assertFalse(torch.any(kind[:, 1:]))
        self.assertEqual(wait.sum(dim=1).tolist(), [2, 2])
        coordinator.record_validation(
            validation_report(
                coordinator, "a" * 64, (ValidationResult("wait", 10, 10),)
            )
        )
        coordinator.record_validation(
            validation_report(
                coordinator, "b" * 64, (ValidationResult("wait", 10, 10),)
            )
        )
        # Promotion affects only newly reset lanes.
        coordinator.activate_focus_for_new_episodes(torch.tensor([False, True]))
        kind, _ = coordinator.action_masks(ActionSpec())
        self.assertEqual(kind[0].tolist(), [True, False, False])
        self.assertEqual(kind[1].tolist(), [True, True, True])

    def test_library_rejects_train_validation_family_leakage_and_bad_blob(self) -> None:
        train = recipe("wait", "train", "shared", "a")
        validation = recipe("wait", "validation", "shared", "b")
        with self.assertRaisesRegex(ValueError, "overlap"):
            SnapshotLibrary((train, validation))
        library = SnapshotLibrary((train, recipe("wait", "validation", "other", "c")))
        with self.assertRaisesRegex(ValueError, "blob hash"):
            library.verify_snapshot_blob(train.snapshot_id, b"tampered")

    def test_library_rejects_split_leakage_despite_distinct_labels(self) -> None:
        train = recipe("wait", "train", "train-family", "a")
        same_construction = replace(
            train,
            snapshot_id="wait-validation-construction",
            split="validation",
            scenario_family="validation-family",
            snapshot_sha256="f" * 64,
        )
        with self.assertRaisesRegex(ValueError, "construction provenance"):
            SnapshotLibrary((train, same_construction))
        calibration_copy = replace(
            train,
            snapshot_id="wait-calibration-construction",
            split="calibration",
            scenario_family="calibration-family",
            snapshot_sha256="e" * 64,
        )
        with self.assertRaisesRegex(ValueError, "construction provenance"):
            SnapshotLibrary((train, calibration_copy))

        different_construction = replace(
            recipe("wait", "validation", "validation-family", "b"),
            expected_tick=train.expected_tick,
            expected_score=train.expected_score,
            expected_state_hash=train.expected_state_hash,
        )
        with self.assertRaisesRegex(ValueError, "state identities"):
            SnapshotLibrary((train, different_construction))
        validation = recipe("wait", "validation", "validation-family", "c")
        test_copy = replace(
            validation,
            snapshot_id="wait-test-construction",
            split="test",
            scenario_family="test-family",
            snapshot_sha256="d" * 64,
        )
        with self.assertRaisesRegex(ValueError, "construction provenance"):
            SnapshotLibrary((validation, test_copy))

    def test_tampered_checkpoint_hash_is_rejected(self) -> None:
        coordinator = CurriculumCoordinator(curriculum(), 2, learner_seed=6)
        state = coordinator.state_dict()
        state["completed_updates"] = 3
        with self.assertRaisesRegex(ValueError, "state hash"):
            coordinator.load_state_dict(state)

    def test_validation_is_bound_to_issued_policy_evaluator_and_trials(self) -> None:
        coordinator = CurriculumCoordinator(curriculum(), 1, learner_seed=7)
        request = coordinator.request_validation(
            policy_sha256="a" * 64, evaluator_identity_sha256="e" * 64
        )
        with self.assertRaisesRegex(ValueError, "trial count"):
            coordinator.record_validation(
                ValidationReport(
                    request.request_id,
                    request.policy_sha256,
                    request.evaluator_identity_sha256,
                    (ValidationResult("wait", 8, 11, request.stages[0].snapshot_ids),),
                )
            )
        with self.assertRaisesRegex(RuntimeError, "pending"):
            coordinator.request_validation(
                policy_sha256="b" * 64, evaluator_identity_sha256="e" * 64
            )
        with self.assertRaisesRegex(RuntimeError, "validation is pending"):
            coordinator.advance_update()
        state = coordinator.state_dict()
        restored = CurriculumCoordinator(curriculum(), 1, learner_seed=7)
        restored.load_state_dict(state)
        accepted = restored.record_validation(
            ValidationReport(
                request.request_id,
                request.policy_sha256,
                request.evaluator_identity_sha256,
                (
                    bind_result(
                        ValidationResult("wait", 8, 10), request, request.stages[0]
                    ),
                ),
            )
        )
        self.assertFalse(accepted.promoted)

    def test_budget_exhaustion_is_terminal(self) -> None:
        spec = curriculum()
        coordinator = CurriculumCoordinator(spec, 1, learner_seed=8)
        for _ in range(spec.stages[0].max_updates):
            coordinator.advance_update()
        self.assertEqual(coordinator.phase, "budget_validation")
        with self.assertRaisesRegex(RuntimeError, "closed"):
            coordinator.advance_update()
        decision = coordinator.record_validation(
            validation_report(coordinator, "a" * 64, (ValidationResult("wait", 8, 10),))
        )
        self.assertEqual(decision.phase, "budget_exhausted")
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            coordinator.request_validation(
                policy_sha256="b" * 64, evaluator_identity_sha256="e" * 64
            )

    def test_final_budget_validation_survives_exact_state_restore(self) -> None:
        spec = curriculum()
        coordinator = CurriculumCoordinator(spec, 1, learner_seed=18)
        for _ in range(spec.stages[0].max_updates):
            coordinator.advance_update()
        request = coordinator.request_validation(
            policy_sha256="a" * 64, evaluator_identity_sha256="e" * 64
        )
        restored = CurriculumCoordinator(spec, 1, learner_seed=18)
        restored.load_state_dict(coordinator.state_dict())
        self.assertEqual(restored.phase, "budget_validation")
        report = ValidationReport(
            request.request_id,
            request.policy_sha256,
            request.evaluator_identity_sha256,
            (bind_result(ValidationResult("wait", 8, 10), request, request.stages[0]),),
        )
        self.assertEqual(restored.record_validation(report).phase, "budget_exhausted")

    def test_final_budget_pass_clears_stale_remediation_focus(self) -> None:
        spec = curriculum()
        coordinator = CurriculumCoordinator(spec, 1, learner_seed=38)
        for policy in ("a" * 64, "b" * 64):
            coordinator.record_validation(
                validation_report(
                    coordinator, policy, (ValidationResult("wait", 8, 10),)
                )
            )
        coordinator.activate_focus_for_new_episodes(torch.ones(1, dtype=torch.bool))
        coordinator.record_validation(
            validation_report(
                coordinator,
                "c" * 64,
                (ValidationResult("wait", 6, 10), ValidationResult("shot", 9, 10)),
            )
        )
        self.assertEqual(coordinator.current_stage.stage_id, "wait")
        for _ in range(spec.stages[1].max_updates):
            coordinator.advance_update()
        decision = coordinator.record_validation(
            validation_report(
                coordinator,
                "d" * 64,
                (ValidationResult("wait", 8, 10), ValidationResult("shot", 9, 10)),
            )
        )
        self.assertEqual(decision.phase, "complete")
        self.assertEqual(decision.focus_stage, "shot")
        restored = CurriculumCoordinator(spec, 1, learner_seed=38)
        restored.load_state_dict(coordinator.state_dict())
        self.assertEqual(restored.current_stage.stage_id, "shot")

    def test_checkpoint_rejects_content_hashed_invalid_phase_semantics(self) -> None:
        spec = curriculum()
        coordinator = CurriculumCoordinator(spec, 1, learner_seed=28)
        for _ in range(spec.stages[0].max_updates):
            coordinator.advance_update()
        state = coordinator.state_dict()
        state["phase"] = "normal"
        core = {key: value for key, value in state.items() if key != "state_sha256"}
        payload = json.dumps(
            core, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        state["state_sha256"] = hashlib.sha256(payload).hexdigest()
        with self.assertRaisesRegex(ValueError, "phase semantics"):
            CurriculumCoordinator(spec, 1, learner_seed=28).load_state_dict(state)

    def test_validation_requires_each_requested_recipe_repetition(self) -> None:
        coordinator = CurriculumCoordinator(curriculum(), 1, learner_seed=10)
        request = coordinator.request_validation(
            policy_sha256="a" * 64, evaluator_identity_sha256="e" * 64
        )
        result = bind_result(
            ValidationResult("wait", 8, 10), request, request.stages[0]
        )
        outcomes = list(result.outcomes)
        outcomes[-1] = replace(outcomes[-1], repetition=99)
        with self.assertRaisesRegex(ValueError, "recipe outcomes"):
            coordinator.record_validation(
                ValidationReport(
                    request.request_id,
                    request.policy_sha256,
                    request.evaluator_identity_sha256,
                    (replace(result, outcomes=tuple(outcomes)),),
                )
            )

    def test_validation_episode_seeds_are_fixed_across_policies_and_gates(self) -> None:
        coordinator = CurriculumCoordinator(curriculum(), 1, learner_seed=11)
        first = coordinator.request_validation(
            policy_sha256="a" * 64, evaluator_identity_sha256="e" * 64
        )
        second = replace(
            first,
            gate_ordinal=first.gate_ordinal + 1,
            completed_update=first.completed_update + 5,
            policy_sha256="b" * 64,
            evaluator_identity_sha256="f" * 64,
        )
        coordinates = ("wait", first.stages[0].snapshot_ids[0], 3)
        self.assertEqual(
            first.episode_seed(*coordinates), second.episode_seed(*coordinates)
        )

    def test_promoted_stage_clock_waits_for_every_lane_to_activate(self) -> None:
        spec = curriculum()
        coordinator = CurriculumCoordinator(spec, 2, learner_seed=12)
        for policy in ("a" * 64, "b" * 64):
            coordinator.record_validation(
                validation_report(
                    coordinator, policy, (ValidationResult("wait", 8, 10),)
                )
            )
        self.assertEqual(coordinator.phase, "activation")
        for _ in range(spec.stages[1].max_updates + 2):
            coordinator.advance_update()
        self.assertEqual(coordinator.phase, "activation")
        self.assertEqual(coordinator.unlock_updates[1], -1)
        coordinator.activate_focus_for_new_episodes(torch.tensor([True, False]))
        coordinator.advance_update()
        self.assertEqual(coordinator.phase, "activation")
        coordinator.activate_focus_for_new_episodes(torch.tensor([False, True]))
        self.assertEqual(coordinator.phase, "normal")
        activation_update = coordinator.unlock_updates[1]
        coordinator.advance_update()
        self.assertEqual(coordinator.phase, "normal")
        self.assertEqual(coordinator.completed_updates, activation_update + 1)

    def test_stage_and_mix_fields_reject_noncanonical_numeric_types(self) -> None:
        base = curriculum()
        with self.assertRaises(TypeError):
            replace(base.stages[0], enabled_action_kinds=(False,))
        with self.assertRaises(TypeError):
            replace(base.stages[0], enabled_wait_ticks=(1.0,))
        with self.assertRaises(TypeError):
            replace(base, prior_stage_mix_ppm=True)
        with self.assertRaises(ValueError):
            replace(base, evaluation_seed=True)
        with self.assertRaisesRegex(ValueError, "cover every"):
            replace(
                base.stages[0],
                validation_snapshot_ids=tuple(f"recipe-{index}" for index in range(11)),
            )

    def test_final_gate_cannot_complete_until_shaping_is_exactly_zero(self) -> None:
        base = curriculum()
        shaped_final = replace(base.stages[0], required_consecutive_passes=1)
        spec = CurriculumSpec(
            "shaped-final-v1",
            base.library,
            (shaped_final,),
            base.evaluation_seed,
            prior_stage_mix_ppm=0,
        )
        coordinator = CurriculumCoordinator(spec, 1, learner_seed=9)
        decision = coordinator.record_validation(
            validation_report(
                coordinator, "a" * 64, (ValidationResult("wait", 10, 10),)
            )
        )
        self.assertEqual(decision.phase, "normal")
        self.assertIn("shaping is not exactly zero", decision.reason)


if __name__ == "__main__":
    unittest.main()
