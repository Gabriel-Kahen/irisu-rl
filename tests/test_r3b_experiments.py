from __future__ import annotations

import copy
import functools
import hashlib
import itertools
import tempfile
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path

from irisu_rl.r3b_evaluation import (
    BaselineArtifactBundle,
    CrossBackendCellPair,
    CrossBackendEvaluationManifest,
    EpisodeMetrics,
    EvaluationReport,
    EvaluationSuite,
    LearnedPolicyBackendParityArtifact,
    ScriptedBaselineSpec,
)
from irisu_rl.actions import ActionSpec
from irisu_rl.curriculum import SnapshotLibrary, SnapshotRecipe
from irisu_rl.r3b_experiments import (
    ArmPhaseResult,
    BaselineEvidence,
    CandidateArm,
    CheckpointEvaluation,
    CurvePoint,
    EngineeringEvidence,
    ExactResumeArtifact,
    LearnerOutcome,
    RawScoreMetricsArtifact,
    R3BExperimentPlan,
    SealedTestLedger,
    SealedTestAuthorization,
    SealedTestRunAuthorization,
    ValidationRunAuthorization,
    TrialSeedPlan,
    TrainingCheckpointArtifact,
    baseline_requirements_pass,
    bind_validation_run,
    build_sealed_confirmation_report,
    confirm_on_sealed_test,
    load_plan,
    select_calibrated_learning_rates,
    select_validation_candidate,
    tick_aligned_raw_score_auc,
    _verified_exact_resume_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs/rl/experiments/r3b-completion-v1.toml"
TEST_PLAN = load_plan(PLAN_PATH)
PLAN_SHA256 = TEST_PLAN.sha256
_LEDGER_DIRECTORY = tempfile.TemporaryDirectory()
_LEDGER_IDS = itertools.count()
_METRICS_CACHE: dict[tuple[object, ...], RawScoreMetricsArtifact] = {}
unittest.addModuleCleanup(_LEDGER_DIRECTORY.cleanup)
_ACTION_SHA256 = ActionSpec().sha256


def _phase_pair(phase: str, seed: int):
    portable = SnapshotRecipe(
        f"{phase}-snapshot",
        f"{phase}-stage",
        phase,
        f"{phase}-family",
        f"portable-{phase}",
        "a" * 64,
        1,
        seed,
        _ACTION_SHA256,
        (),
        0,
        0,
        seed,
        hashlib.sha256(f"{phase}:portable".encode()).hexdigest(),
        "1" * 64,
        "test-generator",
    )
    exact = replace(
        portable,
        snapshot_id=f"exact-{phase}-snapshot",
        environment_pool=f"exact-{phase}",
        snapshot_sha256=hashlib.sha256(f"{phase}:exact".encode()).hexdigest(),
        runtime_identity_sha256="6" * 64,
    )
    portable_library = SnapshotLibrary((portable,))
    exact_library = SnapshotLibrary((exact,))
    logical = CrossBackendEvaluationManifest(
        (CrossBackendCellPair.from_recipes(portable, exact),)
    )
    return portable, exact, portable_library, exact_library, logical


(
    _PORTABLE_TEST_RECIPE,
    _EXACT_TEST_RECIPE,
    _PORTABLE_TEST_LIBRARY,
    _EXACT_TEST_LIBRARY,
    TEST_LOGICAL_MANIFEST,
) = _phase_pair("test", 71)
(
    _PORTABLE_VALIDATION_RECIPE,
    _EXACT_VALIDATION_RECIPE,
    _PORTABLE_VALIDATION_LIBRARY,
    _EXACT_VALIDATION_LIBRARY,
    VALIDATION_LOGICAL_MANIFEST,
) = _phase_pair("validation", 61)
(
    _PORTABLE_CALIBRATION_RECIPE,
    _EXACT_CALIBRATION_RECIPE,
    _PORTABLE_CALIBRATION_LIBRARY,
    _EXACT_CALIBRATION_LIBRARY,
    CALIBRATION_LOGICAL_MANIFEST,
) = _phase_pair("calibration", 51)
_PORTABLE_SEALED_LIBRARY = SnapshotLibrary(
    (_PORTABLE_VALIDATION_RECIPE, _PORTABLE_TEST_RECIPE)
)
_EXACT_SEALED_LIBRARY = SnapshotLibrary((_EXACT_VALIDATION_RECIPE, _EXACT_TEST_RECIPE))
_PORTABLE_VALIDATION_LIBRARY = _PORTABLE_TEST_LIBRARY = _PORTABLE_SEALED_LIBRARY
_EXACT_VALIDATION_LIBRARY = _EXACT_TEST_LIBRARY = _EXACT_SEALED_LIBRARY
TEST_LOGICAL_CELL_IDS = tuple(
    pair.logical_cell.sha256 for pair in TEST_LOGICAL_MANIFEST.pairs
)


def _phase_suite(
    phase: str,
    repetitions: int,
    policy_seed: int,
    recipe: SnapshotRecipe,
    library: SnapshotLibrary,
    logical: CrossBackendEvaluationManifest,
) -> EvaluationSuite:
    exact = recipe.runtime_identity_sha256 == "6" * 64
    return EvaluationSuite(
        f"{phase}-v1",
        phase,
        (recipe.snapshot_id,),
        repetitions,
        policy_seed,
        1,
        1,
        recipe.runtime_identity_sha256,
        "7" * 64 if exact else "2" * 64,
        library.sha256,
        "8" * 64 if exact else "4" * 64,
        _ACTION_SHA256,
        (recipe.sha256,),
        (logical.pairs[0].logical_cell.sha256,),
        "exact" if exact else "portable",
        logical.sha256,
    )


VALIDATION_EVALUATION_SUITE = _phase_suite(
    "validation",
    TEST_PLAN.validation_episodes_per_policy,
    61,
    _EXACT_VALIDATION_RECIPE,
    _EXACT_VALIDATION_LIBRARY,
    VALIDATION_LOGICAL_MANIFEST,
)
TEST_EVALUATION_SUITE = _phase_suite(
    "test",
    512,
    71,
    _EXACT_TEST_RECIPE,
    _EXACT_TEST_LIBRARY,
    TEST_LOGICAL_MANIFEST,
)
PORTABLE_TEST_EVALUATION_SUITE = replace(
    TEST_EVALUATION_SUITE,
    suite_id="sealed-test-portable-v1",
    snapshot_ids=(_PORTABLE_TEST_RECIPE.snapshot_id,),
    recipe_sha256s=(_PORTABLE_TEST_RECIPE.sha256,),
    runtime_identity_sha256="1" * 64,
    assignment_sha256="2" * 64,
    library_sha256=_PORTABLE_TEST_LIBRARY.sha256,
    snapshot_store_sha256="4" * 64,
    backend="portable",
)
EXACT_TEST_EVALUATION_SUITE = TEST_EVALUATION_SUITE
CALIBRATION_EVALUATION_SUITE = _phase_suite(
    "calibration",
    10,
    51,
    _EXACT_CALIBRATION_RECIPE,
    _EXACT_CALIBRATION_LIBRARY,
    CALIBRATION_LOGICAL_MANIFEST,
)


def suite_identity(phase: str) -> str:
    if phase == "test":
        return TEST_EVALUATION_SUITE.sha256
    if phase == "validation":
        return VALIDATION_EVALUATION_SUITE.sha256
    return CALIBRATION_EVALUATION_SUITE.sha256


def parity_artifact(
    exact_suite: EvaluationSuite,
    exact_report: EvaluationReport,
    phase: str,
) -> LearnedPolicyBackendParityArtifact:
    fixtures = {
        "calibration": (
            _PORTABLE_CALIBRATION_RECIPE,
            _PORTABLE_CALIBRATION_LIBRARY,
            _EXACT_CALIBRATION_LIBRARY,
            CALIBRATION_LOGICAL_MANIFEST,
        ),
        "validation": (
            _PORTABLE_VALIDATION_RECIPE,
            _PORTABLE_VALIDATION_LIBRARY,
            _EXACT_VALIDATION_LIBRARY,
            VALIDATION_LOGICAL_MANIFEST,
        ),
        "test": (
            _PORTABLE_TEST_RECIPE,
            _PORTABLE_TEST_LIBRARY,
            _EXACT_TEST_LIBRARY,
            TEST_LOGICAL_MANIFEST,
        ),
    }
    portable_recipe, portable_library, exact_library, logical_manifest = fixtures[phase]
    portable_suite = replace(
        exact_suite,
        suite_id=f"{phase}-portable-v1",
        snapshot_ids=(portable_recipe.snapshot_id,),
        recipe_sha256s=(portable_recipe.sha256,),
        runtime_identity_sha256="1" * 64,
        assignment_sha256="2" * 64,
        library_sha256=portable_library.sha256,
        snapshot_store_sha256="4" * 64,
        backend="portable",
    )
    portable_report = EvaluationReport(
        portable_suite.sha256,
        exact_report.policy_sha256,
        exact_report.evaluator_sha256,
        portable_suite.runtime_identity_sha256,
        hashlib.sha256(f"{exact_report.sha256}:portable".encode()).hexdigest(),
        tuple(
            replace(episode, snapshot_id=portable_recipe.snapshot_id)
            for episode in exact_report.episodes
        ),
    )
    return LearnedPolicyBackendParityArtifact(
        portable_suite,
        portable_report,
        exact_suite,
        exact_report,
        logical_manifest,
        portable_library,
        exact_library,
    )


@functools.lru_cache(maxsize=None)
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

    def metrics(seed: int) -> RawScoreMetricsArtifact:
        suite = {
            "calibration": CALIBRATION_EVALUATION_SUITE,
            "validation": VALIDATION_EVALUATION_SUITE,
            "test": TEST_EVALUATION_SUITE,
        }[phase]
        ticks = TEST_PLAN.tick_grid(budget)
        final_count = len(suite.snapshot_ids) * suite.repetitions
        lower_count = max(1, (final_count + 9) // 10)
        upper = round(
            (final * final_count - p10 * lower_count)
            / max(final_count - lower_count, 1)
        )
        final_scores = [round(p10)] * lower_count + [upper] * (
            final_count - lower_count
        )
        reports = []
        episode_cache: dict[tuple[int, ...], tuple[EpisodeMetrics, ...]] = {}
        for tick_index, tick in enumerate(ticks):
            scores = tuple(
                final_scores
                if tick_index == len(ticks) - 1
                else [round(auc)] * final_count
            )
            episodes = episode_cache.get(scores)
            if episodes is None:
                episodes = tuple(
                    EpisodeMetrics(
                        snapshot_id,
                        repetition,
                        suite.episode_seed(snapshot_id, repetition),
                        0,
                        scores[cell],
                        scores[cell],
                        1,
                        1,
                        False,
                        True,
                        0,
                        100,
                        100,
                    )
                    for cell, (snapshot_id, repetition) in enumerate(
                        (snapshot_id, repetition)
                        for snapshot_id in suite.snapshot_ids
                        for repetition in range(suite.repetitions)
                    )
                )
                episode_cache[scores] = episodes
            policy_sha256 = identity(seed, f"policy:{arm_id}:{tick_index}")
            checkpoint = TrainingCheckpointArtifact(
                seed,
                tick_index * TEST_PLAN.checkpoint_interval_updates,
                tick,
                tick,
                PLAN_SHA256,
                identity(seed, f"job:{arm_id}"),
                identity(seed, "trial-manifest"),
                hashlib.sha256(b"runner:canonical-r3b").hexdigest(),
                identity(seed, f"checkpoint-manifest:{arm_id}:{tick_index}"),
                identity(seed, f"checkpoint-model:{arm_id}:{tick_index}"),
                policy_sha256,
            )
            reports.append(
                CheckpointEvaluation(
                    checkpoint,
                    EvaluationReport(
                        suite.sha256,
                        policy_sha256,
                        "e" * 64,
                        suite.runtime_identity_sha256,
                        identity(seed, f"evaluation:{arm_id}:{tick_index}"),
                        episodes,
                    ),
                )
            )
        return RawScoreMetricsArtifact(
            seed, suite, suite, tuple(reports), reports[-1].report
        )

    values = []
    for seed in seeds:
        key = (seed, auc, final, p10, phase, budget, arm_id)
        artifact = _METRICS_CACHE.get(key)
        if artifact is None:
            artifact = metrics(seed)
            _METRICS_CACHE[key] = artifact
        evidence = (
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
                sealed_job_lease_sha256=(
                    identity(seed, f"lease:{arm_id}") if phase == "test" else None
                ),
                policy_sha256=artifact.final_report.policy_sha256,
                trial_manifest_sha256=identity(seed, "trial-manifest"),
                runner_spec_sha256=hashlib.sha256(b"runner:canonical-r3b").hexdigest(),
                pairing_sha256=identity(seed, f"pairing:{phase}:{budget}"),
                metrics_sha256=artifact.sha256,
                evaluation_suite_sha256=artifact.suite.sha256,
                evaluation_report_sha256=artifact.final_report.sha256,
                final_checkpoint_artifact=artifact.checkpoints[-1].checkpoint,
                resume_checkpoint_artifact=artifact.checkpoints[-2].checkpoint,
                checkpoint_resume_artifact=_verified_exact_resume_artifact(
                    identity(seed, "trial-manifest"),
                    artifact.checkpoints[-2].checkpoint.checkpoint_manifest_sha256,
                    artifact.checkpoints[-2].checkpoint.model_sha256,
                    identity(seed, f"resume-next:{arm_id}"),
                    identity(seed, f"resume-next:{arm_id}"),
                    identity(seed, f"resume-state:{arm_id}"),
                    identity(seed, f"resume-state:{arm_id}"),
                ),
                exact_backend_parity_artifact=parity_artifact(
                    artifact.suite, artifact.final_report, phase
                ),
                tail_state_sha256=(
                    None if phase == "calibration" else identity(seed, "tail")
                ),
                tail_phase=None if phase == "calibration" else "complete",
                score_only_updates=0 if phase == "calibration" else 400,
            )
            if engineering_pass
            else None
        )
        values.append(
            LearnerOutcome(
                seed,
                raw_score_auc=artifact.raw_score_auc,
                final_mean_raw_score=artifact.final_mean_raw_score,
                p10_raw_score=artifact.p10_raw_score,
                initial_model_sha256=identity(seed, "model"),
                assignment_sha256=identity(seed, "assignment"),
                seed_plan_sha256=identity(seed, "seed-plan"),
                metrics_artifact=artifact,
                engineering_evidence=evidence,
            )
        )
    return tuple(values)


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
            "exact-test-snapshot",
            repetition,
            TEST_EVALUATION_SUITE.episode_seed("exact-test-snapshot", repetition),
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

        def portable_report(execution_domain: str) -> EvaluationReport:
            return EvaluationReport(
                PORTABLE_TEST_EVALUATION_SUITE.sha256,
                baseline.sha256,
                "e" * 64,
                PORTABLE_TEST_EVALUATION_SUITE.runtime_identity_sha256,
                hashlib.sha256(
                    f"{baseline_id}:{execution_domain}".encode()
                ).hexdigest(),
                tuple(
                    replace(value, snapshot_id="test-snapshot") for value in episodes
                ),
            )

        def exact_report(execution_domain: str) -> EvaluationReport:
            return EvaluationReport(
                EXACT_TEST_EVALUATION_SUITE.sha256,
                baseline.sha256,
                "e" * 64,
                EXACT_TEST_EVALUATION_SUITE.runtime_identity_sha256,
                hashlib.sha256(
                    f"{baseline_id}:{execution_domain}".encode()
                ).hexdigest(),
                episodes,
            )

        bundles.append(
            BaselineArtifactBundle(
                baseline,
                EXACT_TEST_EVALUATION_SUITE,
                exact_report("exact-primary"),
                exact_report("exact-replay"),
                PORTABLE_TEST_EVALUATION_SUITE,
                portable_report("portable-diagnostic"),
                TEST_LOGICAL_MANIFEST,
                _PORTABLE_TEST_LIBRARY,
                _EXACT_TEST_LIBRARY,
            )
        )
    return tuple(bundles)


def authorization(
    plan: R3BExperimentPlan, control: CandidateArm, candidate: CandidateArm
) -> SealedTestAuthorization:
    return sealed_run(plan, control, candidate).authorization


@functools.lru_cache(maxsize=None)
def sealed_run(
    plan: R3BExperimentPlan, control: CandidateArm, candidate: CandidateArm
) -> SealedTestRunAuthorization:
    ledger, validation = validation_context(plan, control.learning_rate)
    results = authorization_validation_results(
        plan, control, candidate, validation_run=validation
    )
    return ledger.authorize_once(
        plan,
        validation,
        results,
        VALIDATION_EVALUATION_SUITE,
        TEST_EVALUATION_SUITE,
    )


def validation_context(
    plan: R3BExperimentPlan, selected_learning_rate: float
) -> tuple[SealedTestLedger, ValidationRunAuthorization]:
    path = Path(_LEDGER_DIRECTORY.name) / f"ledger-{next(_LEDGER_IDS)}.sqlite3"
    ledger = SealedTestLedger(path)
    commitment = ledger.precommit(plan, TEST_EVALUATION_SUITE)
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
    return ledger, bind_validation_run(plan, results, commitment)


@functools.lru_cache(maxsize=None)
def validation_authorization(
    plan: R3BExperimentPlan, selected_learning_rate: float
) -> ValidationRunAuthorization:
    return validation_context(plan, selected_learning_rate)[1]


def authorization_validation_results(
    plan: R3BExperimentPlan,
    control: CandidateArm,
    candidate: CandidateArm,
    *,
    validation_run: ValidationRunAuthorization | None = None,
) -> tuple[ArmPhaseResult, ...]:
    arms = tuple(
        CandidateArm(alpha, control.learning_rate) for alpha in plan.alpha_weight_ppm
    )
    if candidate not in arms:
        raise ValueError("test helper candidate must share the calibrated LR")
    run = validation_run or validation_authorization(plan, control.learning_rate)
    return tuple(
        phase_result(
            arm,
            "validation",
            plan.validation_learner_seeds,
            budget=plan.validation_updates,
            auc=110 if arm == candidate else 100,
            final=100,
            authorization_sha256=run.sha256,
        )
        for arm in arms
    )


class R3BExperimentPlanTests(unittest.TestCase):
    def test_engineering_evidence_rejects_unbound_parity_and_resume(self) -> None:
        outcome = outcomes(
            (self.plan.calibration_learner_seeds[0],),
            auc=100,
            final=100,
            phase="calibration",
            budget=self.plan.calibration_budgets_updates[0],
        )[0]
        evidence = outcome.engineering_evidence
        assert evidence is not None
        with self.assertRaisesRegex(ValueError, "disagree with the trial"):
            replace(evidence, evaluation_suite_sha256="f" * 64)
        forged_resume = _verified_exact_resume_artifact(
            evidence.trial_manifest_sha256,
            "a" * 64,
            "b" * 64,
            "c" * 64,
            "c" * 64,
            "d" * 64,
            "d" * 64,
        )
        with self.assertRaisesRegex(ValueError, "disagree with the trial"):
            replace(evidence, checkpoint_resume_artifact=forged_resume)
        with self.assertRaisesRegex(ValueError, "does not prove"):
            ExactResumeArtifact(
                evidence.trial_manifest_sha256,
                "a" * 64,
                "b" * 64,
                "c" * 64,
                "c" * 64,
                "d" * 64,
                "d" * 64,
            )

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
            (0, 102_400, 204_800),
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

    def test_auc_interpolates_observed_overshoot_to_target_grid(self) -> None:
        points = (CurvePoint(0, 0), CurvePoint(10, 10), CurvePoint(20, 30))
        self.assertEqual(tick_aligned_raw_score_auc(points, (0, 10, 20)), 12.5)
        overshot = (CurvePoint(0, 0), CurvePoint(12, 12), CurvePoint(24, 36))
        self.assertEqual(tick_aligned_raw_score_auc(overshot, (0, 10, 20)), 12.0)
        with self.assertRaisesRegex(ValueError, "bracketed"):
            tick_aligned_raw_score_auc(points, (0, 10, 30))

    def test_outcome_aggregates_are_derived_from_typed_reports(self) -> None:
        value = outcomes(
            (self.plan.test_learner_seeds[0],),
            auc=100,
            final=100,
            phase="test",
            budget=self.plan.test_updates,
            authorization_sha256="a" * 64,
        )[0]
        with self.assertRaisesRegex(ValueError, "disagree with checkpoint reports"):
            replace(value, raw_score_auc=value.raw_score_auc + 1)

        artifact = value.metrics_artifact
        reused = tuple(
            replace(
                checkpoint,
                report=artifact.final_report,
                checkpoint=replace(
                    checkpoint.checkpoint,
                    deployment_policy_sha256=artifact.final_report.policy_sha256,
                ),
            )
            for checkpoint in artifact.checkpoints
        )
        plateau = replace(artifact, checkpoints=reused)
        self.assertEqual(
            {checkpoint.report.sha256 for checkpoint in plateau.checkpoints},
            {artifact.final_report.sha256},
        )

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
        ledger, validation = validation_context(self.plan, self.plan.learning_rates[1])
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
                authorization_sha256=validation.sha256,
            ),
            phase_result(
                arms[1],
                "validation",
                self.plan.validation_learner_seeds,
                budget=self.plan.validation_updates,
                auc=110,
                final=100,
                authorization_sha256=validation.sha256,
            ),
            phase_result(
                arms[2],
                "validation",
                self.plan.validation_learner_seeds,
                budget=self.plan.validation_updates,
                auc=110,
                final=100,
                authorization_sha256=validation.sha256,
            ),
            phase_result(
                arms[3],
                "validation",
                self.plan.validation_learner_seeds,
                budget=self.plan.validation_updates,
                auc=130,
                final=90,
                authorization_sha256=validation.sha256,
            ),
        ]
        self.assertEqual(
            select_validation_candidate(
                self.plan,
                validation,
                results,
            ),
            arms[1],
        )
        wrong_runner = "f" * 64

        def with_wrong_runner(outcome: LearnerOutcome) -> LearnerOutcome:
            metrics = replace(
                outcome.metrics_artifact,
                checkpoints=tuple(
                    replace(
                        checkpoint,
                        checkpoint=replace(
                            checkpoint.checkpoint,
                            runner_spec_sha256=wrong_runner,
                        ),
                    )
                    for checkpoint in outcome.metrics_artifact.checkpoints
                ),
            )
            return replace(
                outcome,
                metrics_artifact=metrics,
                engineering_evidence=replace(
                    outcome.engineering_evidence,
                    runner_spec_sha256=wrong_runner,
                    metrics_sha256=metrics.sha256,
                    final_checkpoint_artifact=metrics.checkpoints[-1].checkpoint,
                    resume_checkpoint_artifact=metrics.checkpoints[-2].checkpoint,
                ),
            )

        mismatched = tuple(
            replace(
                result,
                outcomes=tuple(
                    with_wrong_runner(outcome) for outcome in result.outcomes
                ),
            )
            for result in results
        )
        with self.assertRaisesRegex(ValueError, "differs from calibrated runner"):
            select_validation_candidate(self.plan, validation, mismatched)
        sealed = ledger.authorize_once(
            self.plan,
            validation,
            results,
            VALIDATION_EVALUATION_SUITE,
            TEST_EVALUATION_SUITE,
        ).authorization
        self.assertEqual(sealed.control_arm_id, arms[0].arm_id)
        self.assertEqual(sealed.candidate_arm_id, arms[1].arm_id)
        self.assertEqual(sealed.attempt, 1)

        forged = replace(
            validation.authorization,
            arms=tuple(reversed(validation.authorization.arms)),
        )
        with self.assertRaisesRegex(ValueError, "calibration artifacts disagree"):
            replace(validation, authorization=forged)
        with self.assertRaisesRegex(ValueError, "calibrated-arm authorization"):
            self.plan.trial_jobs("validation", validation.authorization)

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
            sealed_run(self.plan, control, candidate),
            candidate_result,
            control_result,
            valid_baseline_artifacts(self.plan),
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.p10_mode, "ratio")
        self.assertGreater(decision.relative_auc_gain_lower, 0.05)
        report = build_sealed_confirmation_report(
            self.plan,
            sealed_run(self.plan, control, candidate),
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
        with self.assertRaisesRegex(ValueError, "artifacts disagree"):
            replace(
                sealed_run(self.plan, control, candidate),
                authorization=forged_authorization,
            )

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
            sealed_run(self.plan, control, candidate),
            regressed,
            control_result,
            valid_baseline_artifacts(self.plan),
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.p10_mode, "absolute_delta")
        self.assertFalse(dict(decision.gates)["final_mean_retention_lcb"])
        decision = confirm_on_sealed_test(
            self.plan,
            sealed_run(self.plan, control, candidate),
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
            sealed_run(self.plan, control, candidate),
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

        near_zero_control = phase_result(
            control,
            "test",
            self.plan.test_learner_seeds,
            budget=self.plan.test_updates,
            auc=100,
            final=0,
            p10=0,
        )
        near_zero_candidate = phase_result(
            candidate,
            "test",
            self.plan.test_learner_seeds,
            budget=self.plan.test_updates,
            auc=110,
            final=0,
            p10=0,
        )
        near_zero = confirm_on_sealed_test(
            self.plan,
            sealed_run(self.plan, control, candidate),
            near_zero_candidate,
            near_zero_control,
            valid_baseline_artifacts(self.plan),
        )
        self.assertEqual(near_zero.final_mean_mode, "absolute_delta")
        self.assertEqual(near_zero.final_mean_retention_lower, 0.0)

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
            sealed_run(self.plan, control, candidate),
            failed,
            control_result,
            valid_baseline_artifacts(self.plan),
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.gates, (("complete_exact_test_results", False),))

        malformed = confirm_on_sealed_test(
            self.plan,
            object(),  # type: ignore[arg-type]
            failed,
            control_result,
            valid_baseline_artifacts(self.plan),
        )
        self.assertFalse(malformed.accepted)
        self.assertEqual(malformed.gates, (("complete_exact_test_results", False),))

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
            "test", sealed_run(self.plan, selected[0], candidate)
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

    def test_sealed_test_ledger_is_precommitted_restart_safe_and_one_shot(
        self,
    ) -> None:
        control = CandidateArm(0, self.plan.learning_rates[1])
        candidate = CandidateArm(100_000, self.plan.learning_rates[1])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sealed.sqlite3"
            ledger = SealedTestLedger(path)
            commitment = ledger.precommit(self.plan, TEST_EVALUATION_SUITE)
            calibration = validation_authorization(
                self.plan, control.learning_rate
            ).calibration_results
            validation = bind_validation_run(self.plan, calibration, commitment)
            results = authorization_validation_results(
                self.plan,
                control,
                candidate,
                validation_run=validation,
            )
            first = ledger.authorize_once(
                self.plan,
                validation,
                results,
                VALIDATION_EVALUATION_SUITE,
                TEST_EVALUATION_SUITE,
            )
            forged_report = build_sealed_confirmation_report(
                self.plan,
                first,
                phase_result(
                    candidate,
                    "test",
                    self.plan.test_learner_seeds,
                    budget=self.plan.test_updates,
                    auc=110,
                    final=100,
                    authorization_sha256=first.sha256,
                ),
                phase_result(
                    control,
                    "test",
                    self.plan.test_learner_seeds,
                    budget=self.plan.test_updates,
                    auc=100,
                    final=100,
                    authorization_sha256=first.sha256,
                ),
                valid_baseline_artifacts(self.plan),
            )
            self.assertFalse(ledger.verify_finalized(forged_report))
            resumed = SealedTestLedger(path).authorize_once(
                self.plan,
                validation,
                results,
                VALIDATION_EVALUATION_SUITE,
                TEST_EVALUATION_SUITE,
            )
            self.assertEqual(first.authorization, resumed.authorization)
            with self.assertRaisesRegex(RuntimeError, "receipt is not active"):
                replace(first, ledger_receipt_token="f" * 64)
            alternate = replace(
                TEST_EVALUATION_SUITE,
                suite_id="post-hoc-test",
                snapshot_ids=("alternate-test-snapshot",),
            )
            with self.assertRaisesRegex(RuntimeError, "another suite"):
                ledger.precommit(self.plan, alternate)
            with self.assertRaisesRegex(ValueError, "sealed test lacks"):
                ledger.authorize_once(
                    self.plan,
                    validation,
                    results,
                    VALIDATION_EVALUATION_SUITE,
                    alternate,
                )

            baseline_lease = ledger.claim_baseline_batch(first)
            resumed_baseline_lease = SealedTestLedger(path).resume_baseline_batch(
                first, lease_token=baseline_lease.lease_token
            )
            self.assertEqual(resumed_baseline_lease, baseline_lease)
            with self.assertRaisesRegex(RuntimeError, "recovery token"):
                ledger.resume_baseline_batch(first, lease_token="f" * 64)
            ledger.begin_baseline_batch(baseline_lease)
            with self.assertRaisesRegex(RuntimeError, "recovery token"):
                ledger.resume_baseline_batch(
                    first, lease_token=baseline_lease.lease_token
                )
            baseline_artifacts = valid_baseline_artifacts(self.plan)
            evidence = ledger.complete_baseline_batch(
                baseline_lease, baseline_artifacts
            )
            self.assertEqual(
                tuple(item.baseline_id for item in evidence),
                self.plan.required_baselines,
            )
            with self.assertRaisesRegex(RuntimeError, "already claimed"):
                ledger.claim_baseline_batch(first)

            candidate_result = phase_result(
                candidate,
                "test",
                self.plan.test_learner_seeds,
                budget=self.plan.test_updates,
                auc=110,
                final=100,
                authorization_sha256=first.sha256,
            )
            control_result = phase_result(
                control,
                "test",
                self.plan.test_learner_seeds,
                budget=self.plan.test_updates,
                auc=100,
                final=100,
                authorization_sha256=first.sha256,
            )
            recorded_results = {
                candidate.arm_id: candidate_result,
                control.arm_id: control_result,
            }
            recorded_outcomes: dict[str, list[LearnerOutcome]] = {
                candidate.arm_id: [],
                control.arm_id: [],
            }
            for job in self.plan.trial_jobs("test", first):
                lease = ledger.claim_job(first, job)
                ledger.begin_job(lease)
                source = next(
                    outcome
                    for outcome in recorded_results[job.arm.arm_id].outcomes
                    if outcome.learner_seed == job.learner_seed
                )
                assert source.engineering_evidence is not None
                metrics = replace(
                    source.metrics_artifact,
                    checkpoints=tuple(
                        replace(
                            evaluated,
                            checkpoint=replace(
                                evaluated.checkpoint, job_sha256=job.sha256
                            ),
                        )
                        for evaluated in source.metrics_artifact.checkpoints
                    ),
                )
                bound = replace(
                    source,
                    seed_plan_sha256=job.seed_plan_sha256,
                    metrics_artifact=metrics,
                    engineering_evidence=replace(
                        source.engineering_evidence,
                        job_sha256=job.sha256,
                        sealed_job_lease_sha256=lease.sha256,
                        metrics_sha256=metrics.sha256,
                        final_checkpoint_artifact=metrics.checkpoints[-1].checkpoint,
                        resume_checkpoint_artifact=metrics.checkpoints[-2].checkpoint,
                    ),
                )
                ledger.complete_job(lease, bound)
                recorded_outcomes[job.arm.arm_id].append(bound)
            candidate_result = replace(
                candidate_result,
                outcomes=tuple(recorded_outcomes[candidate.arm_id]),
            )
            control_result = replace(
                control_result,
                outcomes=tuple(recorded_outcomes[control.arm_id]),
            )
            with self.assertRaisesRegex(ValueError, "differs"):
                ledger.finalize_once(
                    first,
                    candidate_result,
                    control_result,
                    valid_baseline_artifacts(self.plan, raw_score=26),
                )
            report = ledger.finalize_once(
                first,
                candidate_result,
                control_result,
                baseline_artifacts,
            )
            self.assertTrue(ledger.verify_finalized(report))
            with self.assertRaisesRegex(RuntimeError, "receipt is not active"):
                ledger.finalize_once(
                    first,
                    candidate_result,
                    control_result,
                    valid_baseline_artifacts(self.plan),
                )
            with self.assertRaisesRegex(RuntimeError, "consumed"):
                ledger.authorize_once(
                    self.plan,
                    validation,
                    results,
                    VALIDATION_EVALUATION_SUITE,
                    TEST_EVALUATION_SUITE,
                )

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
