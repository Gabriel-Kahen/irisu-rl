from __future__ import annotations

import copy
import hashlib
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path

from irisu_rl.r3b_evaluation import (
    BaselineArtifactBundle,
    EpisodeMetrics,
    EvaluationReport,
    EvaluationSuite,
    ScriptedBaselineSpec,
)
from irisu_rl.r3b_experiments import (
    ArmPhaseResult,
    BaselineEvidence,
    CalibrationSelectionAuthorization,
    CandidateArm,
    CurvePoint,
    EngineeringEvidence,
    LearnerOutcome,
    R3BExperimentPlan,
    SealedTestAuthorization,
    SealedTestRunAuthorization,
    TrialSeedPlan,
    authorize_sealed_test,
    authorize_validation,
    baseline_requirements_pass,
    bind_sealed_test_run,
    build_sealed_confirmation_report,
    confirm_on_sealed_test,
    load_plan,
    select_calibrated_learning_rates,
    select_validation_candidate,
    tick_aligned_raw_score_auc,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/rl/experiments/r3b-completion-v1.toml"
TEST_PLAN = load_plan(PLAN_PATH)
PLAN_SHA256 = TEST_PLAN.sha256
VALIDATION_EVALUATION_SUITE = EvaluationSuite(
    "validation-v1",
    "validation",
    ("validation-snapshot",),
    TEST_PLAN.validation_episodes_per_policy,
    61,
    1,
    1,
    "1" * 64,
    "2" * 64,
    "3" * 64,
    "4" * 64,
    "5" * 64,
)
TEST_EVALUATION_SUITE = EvaluationSuite(
    "sealed-test-v1",
    "test",
    ("test-snapshot",),
    512,
    71,
    1,
    1,
    "1" * 64,
    "2" * 64,
    "3" * 64,
    "4" * 64,
    "5" * 64,
)


def suite_identity(phase: str) -> str:
    if phase == "test":
        return TEST_EVALUATION_SUITE.sha256
    if phase == "validation":
        return VALIDATION_EVALUATION_SUITE.sha256
    return hashlib.sha256(f"suite:{phase}".encode()).hexdigest()


def outcomes(
    seeds: tuple[int, ...],
    *,
    auc: float,
    final: float,
    p10: float = 20.0,
    engineering_pass: bool = True,
    phase: str = "test",
    budget: int = 1000,
    arm_id: str = "synthetic-arm",
    authorization_sha256: str | None = None,
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
            engineering_evidence=(
                EngineeringEvidence(
                    phase=phase,
                    completed_updates=budget,
                    plan_sha256=PLAN_SHA256,
                    job_sha256=identity(seed, f"job:{arm_id}"),
                    arm_id=arm_id,
                    learner_seed=seed,
                    authorization_sha256=(
                        None if phase == "calibration" else authorization_sha256
                    ),
                    policy_sha256=identity(seed, f"policy:{arm_id}"),
                    trial_manifest_sha256=identity(seed, "trial-manifest"),
                    pairing_sha256=identity(seed, f"pairing:{phase}:{budget}"),
                    metrics_sha256=identity(seed, "metrics"),
                    evaluation_suite_sha256=suite_identity(phase),
                    evaluation_report_sha256=identity(seed, "evaluation-report"),
                    checkpoint_resume_sha256=identity(seed, "resume"),
                    exact_backend_parity_sha256=identity(seed, "parity"),
                    tail_state_sha256=(
                        None if phase == "calibration" else identity(seed, "tail")
                    ),
                    tail_phase=None if phase == "calibration" else "complete",
                    score_only_updates=0 if phase == "calibration" else 400,
                )
                if engineering_pass
                else None
            ),
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
    authorization_sha256: str | None = None,
) -> ArmPhaseResult:
    if phase == "test" and authorization_sha256 is None:
        plan = load_plan(PLAN_PATH)
        authorization_sha256 = authorization(
            plan,
            CandidateArm(0, arm.learning_rate),
            CandidateArm(100_000, arm.learning_rate),
        ).sha256
    elif phase == "validation" and authorization_sha256 is None:
        authorization_sha256 = validation_authorization(
            load_plan(PLAN_PATH), arm.learning_rate
        ).sha256
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
            phase=phase,
            budget=budget,
            arm_id=arm.arm_id,
            authorization_sha256=authorization_sha256,
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
            suite_identity("test"),
            hashlib.sha256(f"baseline:{baseline_id}".encode()).hexdigest(),
            hashlib.sha256(f"replay:{baseline_id}".encode()).hexdigest(),
            hashlib.sha256(f"exact:{baseline_id}".encode()).hexdigest(),
            "a" * 64,
            "b" * 64,
            hashlib.sha256(f"episodes:{baseline_id}".encode()).hexdigest(),
        )
        for baseline_id in plan.required_baselines
    )


def valid_baseline_artifacts(
    plan: R3BExperimentPlan, *, raw_score: int = 25
) -> tuple[BaselineArtifactBundle, ...]:
    episodes = tuple(
        EpisodeMetrics(
            "test-snapshot",
            repetition,
            TEST_EVALUATION_SUITE.episode_seed("test-snapshot", repetition),
            0,
            raw_score,
            raw_score,
            1,
            1,
            False,
            True,
            0,
            100,
            100,
        )
        for repetition in range(plan.minimum_baseline_episodes)
    )
    bundles = []
    for baseline_id in plan.required_baselines:
        baseline = ScriptedBaselineSpec(baseline_id)

        def report(backend: str, execution_domain: str) -> EvaluationReport:
            return EvaluationReport(
                TEST_EVALUATION_SUITE.sha256,
                baseline.sha256,
                "e" * 64,
                backend * 64,
                hashlib.sha256(
                    f"{baseline_id}:{execution_domain}".encode()
                ).hexdigest(),
                episodes,
            )

        bundles.append(
            BaselineArtifactBundle(
                baseline,
                TEST_EVALUATION_SUITE,
                report("a", "primary"),
                report("a", "replay"),
                report("b", "exact"),
            )
        )
    return tuple(bundles)


def authorization(
    plan: R3BExperimentPlan, control: CandidateArm, candidate: CandidateArm
) -> SealedTestAuthorization:
    return sealed_run(plan, control, candidate).authorization


def sealed_run(
    plan: R3BExperimentPlan, control: CandidateArm, candidate: CandidateArm
) -> SealedTestRunAuthorization:
    results = authorization_validation_results(plan, control, candidate)
    return bind_sealed_test_run(
        plan,
        validation_authorization(plan, control.learning_rate),
        results,
        VALIDATION_EVALUATION_SUITE,
        TEST_EVALUATION_SUITE,
    )


def validation_authorization(
    plan: R3BExperimentPlan, selected_learning_rate: float
) -> CalibrationSelectionAuthorization:
    results = tuple(
        phase_result(
            arm,
            "calibration",
            plan.calibration_learner_seeds,
            budget=plan.calibration_budgets_updates[-1],
            auc=110 if arm.learning_rate == selected_learning_rate else 100,
            final=100,
        )
        for arm in plan.arms
    )
    return authorize_validation(plan, results)


def authorization_validation_results(
    plan: R3BExperimentPlan, control: CandidateArm, candidate: CandidateArm
) -> tuple[ArmPhaseResult, ...]:
    arms = tuple(
        CandidateArm(alpha, control.learning_rate) for alpha in plan.alpha_weight_ppm
    )
    if candidate not in arms:
        raise ValueError("test helper candidate must share the calibrated LR")
    return tuple(
        phase_result(
            arm,
            "validation",
            plan.validation_learner_seeds,
            budget=plan.validation_updates,
            auc=110 if arm == candidate else 100,
            final=100,
        )
        for arm in arms
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
            select_validation_candidate(
                self.plan,
                validation_authorization(self.plan, self.plan.learning_rates[1]),
                results,
            ),
            arms[1],
        )
        sealed = authorize_sealed_test(
            self.plan,
            validation_authorization(self.plan, self.plan.learning_rates[1]),
            results,
            VALIDATION_EVALUATION_SUITE,
            TEST_EVALUATION_SUITE,
        )
        self.assertEqual(sealed.control_arm_id, arms[0].arm_id)
        self.assertEqual(sealed.candidate_arm_id, arms[1].arm_id)
        self.assertEqual(sealed.attempt, 1)

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
            authorization(self.plan, control, candidate),
            validation_authorization(self.plan, control.learning_rate),
            authorization_validation_results(self.plan, control, candidate),
            VALIDATION_EVALUATION_SUITE,
            TEST_EVALUATION_SUITE,
            candidate_result,
            control_result,
            valid_baseline_artifacts(self.plan),
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.p10_mode, "ratio")
        self.assertGreater(decision.relative_auc_gain_lower, 0.05)
        report = build_sealed_confirmation_report(
            self.plan,
            authorization(self.plan, control, candidate),
            validation_authorization(self.plan, control.learning_rate),
            authorization_validation_results(self.plan, control, candidate),
            VALIDATION_EVALUATION_SUITE,
            TEST_EVALUATION_SUITE,
            candidate_result,
            control_result,
            valid_baseline_artifacts(self.plan),
        )
        self.assertTrue(report.accepted)
        self.assertNotEqual(report.sha256, "0" * 64)
        self.assertEqual(report.manifest()["decision"], decision.manifest())

        unselected = CandidateArm(250_000, self.plan.learning_rates[1])
        forged_authorization = replace(
            authorization(self.plan, control, candidate),
            candidate_arm_id=unselected.arm_id,
        )
        forged_result = phase_result(
            unselected,
            "test",
            self.plan.test_learner_seeds,
            budget=self.plan.test_updates,
            auc=110,
            final=100,
        )
        rejected = confirm_on_sealed_test(
            self.plan,
            forged_authorization,
            validation_authorization(self.plan, control.learning_rate),
            authorization_validation_results(self.plan, control, candidate),
            VALIDATION_EVALUATION_SUITE,
            TEST_EVALUATION_SUITE,
            forged_result,
            control_result,
            valid_baseline_artifacts(self.plan),
        )
        self.assertFalse(rejected.accepted)

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
            authorization(self.plan, control, candidate),
            validation_authorization(self.plan, control.learning_rate),
            authorization_validation_results(self.plan, control, candidate),
            VALIDATION_EVALUATION_SUITE,
            TEST_EVALUATION_SUITE,
            regressed,
            control_result,
            valid_baseline_artifacts(self.plan),
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.p10_mode, "absolute_delta")
        self.assertFalse(dict(decision.gates)["final_mean_retention_lcb"])
        decision = confirm_on_sealed_test(
            self.plan,
            authorization(self.plan, control, candidate),
            validation_authorization(self.plan, control.learning_rate),
            authorization_validation_results(self.plan, control, candidate),
            VALIDATION_EVALUATION_SUITE,
            TEST_EVALUATION_SUITE,
            replace(
                regressed,
                outcomes=outcomes(
                    self.plan.test_learner_seeds,
                    auc=110,
                    final=100,
                    p10=0,
                    phase="test",
                    budget=self.plan.test_updates,
                    arm_id=candidate.arm_id,
                    authorization_sha256=authorization(
                        self.plan, control, candidate
                    ).sha256,
                ),
            ),
            control_result,
            (),
        )
        self.assertFalse(decision.accepted)
        self.assertFalse(dict(decision.gates)["required_baselines"])

        stronger_baselines = valid_baseline_artifacts(self.plan, raw_score=101)
        decision = confirm_on_sealed_test(
            self.plan,
            authorization(self.plan, control, candidate),
            validation_authorization(self.plan, control.learning_rate),
            authorization_validation_results(self.plan, control, candidate),
            VALIDATION_EVALUATION_SUITE,
            TEST_EVALUATION_SUITE,
            replace(
                regressed,
                outcomes=outcomes(
                    self.plan.test_learner_seeds,
                    auc=110,
                    final=100,
                    p10=0,
                    phase="test",
                    budget=self.plan.test_updates,
                    arm_id=candidate.arm_id,
                    authorization_sha256=authorization(
                        self.plan, control, candidate
                    ).sha256,
                ),
            ),
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
            authorization(self.plan, control, candidate),
            validation_authorization(self.plan, control.learning_rate),
            authorization_validation_results(self.plan, control, candidate),
            VALIDATION_EVALUATION_SUITE,
            TEST_EVALUATION_SUITE,
            failed,
            control_result,
            valid_baseline_artifacts(self.plan),
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
        self.assertEqual(len(calibration), 12 * 3 * 2)
        self.assertTrue(all(not job.sealed for job in calibration))
        self.assertEqual(
            {job.budget_updates for job in calibration},
            set(self.plan.calibration_budgets_updates),
        )

        selected = tuple(
            CandidateArm(alpha, self.plan.learning_rates[0])
            for alpha in self.plan.alpha_weight_ppm
        )
        validation = self.plan.trial_jobs(
            "validation",
            validation_authorization(self.plan, self.plan.learning_rates[0]),
        )
        self.assertEqual(len(validation), 4 * 8)
        self.assertTrue(all(not job.sealed for job in validation))
        candidate = selected[1]
        test = self.plan.trial_jobs(
            "test", authorization(self.plan, selected[0], candidate)
        )
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
        with self.assertRaisesRegex(ValueError, "validation-bound"):
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
    (authorize_sealed_test,)
