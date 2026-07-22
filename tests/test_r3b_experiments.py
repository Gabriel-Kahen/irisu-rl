from __future__ import annotations

import copy
import hashlib
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path

from irisu_rl.r3b_experiments import (
    ArmPhaseResult,
    BaselineEvidence,
    CandidateArm,
    CurvePoint,
    LearnerOutcome,
    R3BExperimentPlan,
    TrialSeedPlan,
    baseline_requirements_pass,
    build_sealed_confirmation_report,
    confirm_on_sealed_test,
    load_plan,
    select_calibrated_learning_rates,
    select_validation_candidate,
    tick_aligned_raw_score_auc,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/rl/experiments/r3b-completion-v1.toml"


def outcomes(
    seeds: tuple[int, ...],
    *,
    auc: float,
    final: float,
    p10: float = 20.0,
    engineering_pass: bool = True,
) -> tuple[LearnerOutcome, ...]:
    def identity(seed: int, domain: str) -> str:
        return hashlib.sha256(f"{seed}:{domain}".encode()).hexdigest()

    return tuple(
        LearnerOutcome(
            seed,
            raw_score_auc=auc,
            final_mean_raw_score=final,
            p10_raw_score=p10,
            initial_model_sha256=identity(seed, "model"),
            assignment_sha256=identity(seed, "assignment"),
            seed_plan_sha256=identity(seed, "seed-plan"),
            engineering_pass=engineering_pass,
        )
        for seed in seeds
    )


def phase_result(
    arm: CandidateArm,
    phase: str,
    seeds: tuple[int, ...],
    *,
    budget: int,
    auc: float,
    final: float,
    p10: float = 20.0,
    engineering_pass: bool = True,
) -> ArmPhaseResult:
    return ArmPhaseResult(
        arm.arm_id,
        phase,
        "complete",
        budget,
        outcomes(
            seeds,
            auc=auc,
            final=final,
            p10=p10,
            engineering_pass=engineering_pass,
        ),
    )


def valid_baselines(plan: R3BExperimentPlan) -> tuple[BaselineEvidence, ...]:
    return tuple(
        BaselineEvidence(
            baseline_id,
            "complete",
            plan.minimum_baseline_episodes,
            25.0,
            0,
            True,
            True,
            True,
            hashlib.sha256(f"baseline:{baseline_id}".encode()).hexdigest(),
        )
        for baseline_id in plan.required_baselines
    )


class R3BExperimentPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = load_plan(PLAN_PATH)

    def test_checked_plan_is_immutable_complete_and_hash_stable(self) -> None:
        with PLAN_PATH.open("rb") as handle:
            source = tomllib.load(handle)
        self.assertEqual(self.plan.manifest(), source)
        self.assertEqual(len(self.plan.arms), 12)
        self.assertEqual(len({arm.arm_id for arm in self.plan.arms}), 12)
        self.assertEqual(self.plan.zero_tail_updates, 400)
        self.assertFalse(self.plan.optimizer_updates_while_draining)
        phase_seeds = (
            self.plan.calibration_learner_seeds
            + self.plan.validation_learner_seeds
            + self.plan.test_learner_seeds
        )
        self.assertEqual(len(phase_seeds), len(set(phase_seeds)))
        self.assertEqual(self.plan.sha256, load_plan(PLAN_PATH).sha256)
        self.assertEqual(
            self.plan.tick_grid(100),
            (0, 1_638_400, 3_276_800),
        )

    def test_plan_parser_rejects_unknown_fields_and_weak_tail(self) -> None:
        with PLAN_PATH.open("rb") as handle:
            source = tomllib.load(handle)
        unknown = copy.deepcopy(source)
        unknown["statistics"]["unplanned"] = True
        with self.assertRaisesRegex(ValueError, "keys differ"):
            R3BExperimentPlan.from_mapping(unknown)
        with self.assertRaisesRegex(ValueError, "at least 400"):
            replace(
                self.plan,
                total_updates=999,
                shaped_updates=600,
                zero_tail_updates=399,
                validation_updates=999,
                test_updates=999,
            )

    def test_auc_requires_exact_tick_alignment(self) -> None:
        points = (CurvePoint(0, 0), CurvePoint(10, 10), CurvePoint(20, 30))
        self.assertEqual(tick_aligned_raw_score_auc(points, (0, 10, 20)), 12.5)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            tick_aligned_raw_score_auc(points, (0, 10, 30))

    def test_calibration_selects_one_lr_per_alpha_with_stable_ties(self) -> None:
        results: list[ArmPhaseResult] = []
        lowest_rate = self.plan.learning_rates[0]
        for arm in self.plan.arms:
            results.append(
                phase_result(
                    arm,
                    "calibration",
                    self.plan.calibration_learner_seeds,
                    budget=self.plan.calibration_budgets_updates[-1],
                    auc=100.0,
                    final=80.0,
                )
            )
        # Failed arms remain in the exact result set and cannot win.
        failed_arm = self.plan.arms[1]
        results[1] = ArmPhaseResult(
            failed_arm.arm_id,
            "calibration",
            "post_start_failure",
            self.plan.calibration_budgets_updates[0],
            failure_reason="synthetic worker failure",
        )
        selected = select_calibrated_learning_rates(self.plan, results)
        self.assertEqual(len(selected), 4)
        self.assertTrue(all(arm.learning_rate == lowest_rate for arm in selected))

    def test_calibration_missing_arm_rejects_phase_and_missing_seed_rejects_arm(
        self,
    ) -> None:
        results = [
            phase_result(
                arm,
                "calibration",
                self.plan.calibration_learner_seeds,
                budget=self.plan.calibration_budgets_updates[-1],
                auc=100,
                final=100,
            )
            for arm in self.plan.arms
        ]
        with self.assertRaisesRegex(ValueError, "exactly every expected arm"):
            select_calibrated_learning_rates(self.plan, results[:-1])
        results[0] = phase_result(
            self.plan.arms[0],
            "calibration",
            self.plan.calibration_learner_seeds[:-1],
            budget=self.plan.calibration_budgets_updates[-1],
            auc=100,
            final=100,
        )
        selected = select_calibrated_learning_rates(self.plan, results)
        self.assertEqual(selected[0], self.plan.arms[1])

    def test_validation_selection_uses_gain_retention_and_stable_ties(self) -> None:
        arms = tuple(
            CandidateArm(alpha, self.plan.learning_rates[1])
            for alpha in self.plan.alpha_weight_ppm
        )
        results = [
            phase_result(
                arms[0],
                "validation",
                self.plan.validation_learner_seeds,
                budget=self.plan.validation_updates,
                auc=100,
                final=100,
            ),
            phase_result(
                arms[1],
                "validation",
                self.plan.validation_learner_seeds,
                budget=self.plan.validation_updates,
                auc=110,
                final=100,
            ),
            phase_result(
                arms[2],
                "validation",
                self.plan.validation_learner_seeds,
                budget=self.plan.validation_updates,
                auc=110,
                final=100,
            ),
            phase_result(
                arms[3],
                "validation",
                self.plan.validation_learner_seeds,
                budget=self.plan.validation_updates,
                auc=130,
                final=90,
            ),
        ]
        self.assertEqual(
            select_validation_candidate(self.plan, arms, results),
            arms[1],
        )

    def test_baseline_requirements_are_exact_and_fail_closed(self) -> None:
        evidence = valid_baselines(self.plan)
        self.assertTrue(baseline_requirements_pass(self.plan, evidence))
        self.assertFalse(baseline_requirements_pass(self.plan, evidence[:-1]))
        bad = replace(evidence[0], invalid_actions=1)
        self.assertFalse(baseline_requirements_pass(self.plan, (bad, *evidence[1:])))
        with self.assertRaisesRegex(ValueError, "report SHA-256"):
            replace(evidence[0], report_sha256="0" * 64)

    def test_sealed_confirmation_accepts_clear_synthetic_effect(self) -> None:
        control = CandidateArm(0, self.plan.learning_rates[1])
        candidate = CandidateArm(100_000, self.plan.learning_rates[1])
        control_result = phase_result(
            control,
            "test",
            self.plan.test_learner_seeds,
            budget=self.plan.test_updates,
            auc=100,
            final=100,
            p10=20,
        )
        candidate_result = phase_result(
            candidate,
            "test",
            self.plan.test_learner_seeds,
            budget=self.plan.test_updates,
            auc=110,
            final=100,
            p10=20,
        )
        decision = confirm_on_sealed_test(
            self.plan,
            candidate,
            candidate_result,
            control,
            control_result,
            valid_baselines(self.plan),
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.p10_mode, "ratio")
        self.assertGreater(decision.relative_auc_gain_lower, 0.05)
        report = build_sealed_confirmation_report(
            self.plan,
            candidate,
            candidate_result,
            control,
            control_result,
            valid_baselines(self.plan),
        )
        self.assertTrue(report.accepted)
        self.assertNotEqual(report.sha256, "0" * 64)
        self.assertEqual(report.manifest()["decision"], decision.manifest())

    def test_sealed_confirmation_rejects_regression_or_bad_evidence(self) -> None:
        control = CandidateArm(0, self.plan.learning_rates[1])
        candidate = CandidateArm(100_000, self.plan.learning_rates[1])
        control_result = phase_result(
            control,
            "test",
            self.plan.test_learner_seeds,
            budget=self.plan.test_updates,
            auc=100,
            final=100,
            p10=0,
        )
        regressed = phase_result(
            candidate,
            "test",
            self.plan.test_learner_seeds,
            budget=self.plan.test_updates,
            auc=110,
            final=90,
            p10=0,
        )
        decision = confirm_on_sealed_test(
            self.plan,
            candidate,
            regressed,
            control,
            control_result,
            valid_baselines(self.plan),
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.p10_mode, "absolute_delta")
        self.assertFalse(dict(decision.gates)["final_mean_retention_lcb"])
        decision = confirm_on_sealed_test(
            self.plan,
            candidate,
            replace(
                regressed,
                outcomes=outcomes(
                    self.plan.test_learner_seeds,
                    auc=110,
                    final=100,
                    p10=0,
                ),
            ),
            control,
            control_result,
            (),
        )
        self.assertFalse(decision.accepted)
        self.assertFalse(dict(decision.gates)["required_baselines"])

        stronger_baselines = tuple(
            replace(item, mean_raw_score=101.0) for item in valid_baselines(self.plan)
        )
        decision = confirm_on_sealed_test(
            self.plan,
            candidate,
            replace(
                regressed,
                outcomes=outcomes(
                    self.plan.test_learner_seeds,
                    auc=110,
                    final=100,
                    p10=0,
                ),
            ),
            control,
            control_result,
            stronger_baselines,
        )
        self.assertFalse(decision.accepted)
        self.assertFalse(dict(decision.gates)["trivial_baseline_margin_lcb"])

    def test_noncomplete_test_result_is_rejection_not_silent_drop(self) -> None:
        control = CandidateArm(0, self.plan.learning_rates[1])
        candidate = CandidateArm(100_000, self.plan.learning_rates[1])
        failed = ArmPhaseResult(
            candidate.arm_id,
            "test",
            "pre_start_failure",
            self.plan.test_updates,
            failure_reason="synthetic startup failure",
        )
        control_result = phase_result(
            control,
            "test",
            self.plan.test_learner_seeds,
            budget=self.plan.test_updates,
            auc=100,
            final=100,
        )
        decision = confirm_on_sealed_test(
            self.plan,
            candidate,
            failed,
            control,
            control_result,
            valid_baselines(self.plan),
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.gates, (("complete_exact_test_results", False),))

    def test_seed_plans_are_domain_separated_and_arm_independent(self) -> None:
        first = TrialSeedPlan.derive(self.plan.sha256, 42)
        repeated = TrialSeedPlan.derive(self.plan.sha256, 42)
        other = TrialSeedPlan.derive(self.plan.sha256, 43)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first.sha256, other.sha256)
        streams = (
            first.model_initialization,
            first.policy_sampling,
            first.ppo_minibatching,
            first.assignment,
            first.session_numpy,
            first.evaluation,
        )
        self.assertEqual(len(streams), len(set(streams)))

    def test_phase_job_enumeration_is_exact_paired_and_sealed_late(self) -> None:
        calibration = self.plan.trial_jobs("calibration")
        self.assertEqual(len(calibration), 12 * 3)
        self.assertTrue(all(not job.sealed for job in calibration))
        self.assertEqual(
            {job.budget_updates for job in calibration},
            {self.plan.calibration_budgets_updates[-1]},
        )

        selected = tuple(
            CandidateArm(alpha, self.plan.learning_rates[0])
            for alpha in self.plan.alpha_weight_ppm
        )
        validation = self.plan.trial_jobs("validation", selected)
        self.assertEqual(len(validation), 4 * 8)
        self.assertTrue(all(not job.sealed for job in validation))
        candidate = selected[1]
        test = self.plan.trial_jobs("test", (selected[0], candidate))
        self.assertEqual(len(test), 2 * 12)
        self.assertTrue(all(job.sealed for job in test))
        self.assertEqual(
            {
                job.seed_plan_sha256
                for job in test
                if job.learner_seed == test[0].learner_seed
            },
            {test[0].seed_plan_sha256},
        )
        with self.assertRaisesRegex(ValueError, "control and one shaped"):
            self.plan.trial_jobs("test", selected)

    def test_selection_rejects_unpaired_model_or_assignment_identity(self) -> None:
        results = [
            phase_result(
                arm,
                "calibration",
                self.plan.calibration_learner_seeds,
                budget=self.plan.calibration_budgets_updates[-1],
                auc=100,
                final=100,
            )
            for arm in self.plan.arms
        ]
        changed = list(results[1].outcomes)
        changed[0] = replace(changed[0], assignment_sha256="f" * 64)
        results[1] = replace(results[1], outcomes=tuple(changed))
        with self.assertRaisesRegex(ValueError, "paired arms disagree"):
            select_calibrated_learning_rates(self.plan, results)


if __name__ == "__main__":
    unittest.main()
