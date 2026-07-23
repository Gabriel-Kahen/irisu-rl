from __future__ import annotations

import unittest

from irisu_rl.r3b_evaluation import (
    CrossBackendCellPair,
    CrossBackendEvaluationManifest,
    EpisodeMetrics,
    EvaluationReport,
    EvaluationSuite,
    LogicalEvaluationCell,
)
from irisu_rl.r3b_experiments import (
    BaselineEvidence,
    CalibrationSelectionAuthorization,
    CandidateArm,
    CheckpointEvaluation,
    ExactResumeArtifact,
    EngineeringEvidence,
    LearnerOutcome,
    ArmPhaseResult,
    RawScoreMetricsArtifact,
    SealedBaselineBatchCommitment,
    SealedTestAuthorization,
    TestSuiteCommitment,
    TrainingCheckpointArtifact,
    TrialJob,
)
from tests.test_r3b_experiments import TEST_PLAN, outcomes


def _hash(character: str) -> str:
    return character * 64


def _suite() -> EvaluationSuite:
    return EvaluationSuite(
        "validation-codec",
        "validation",
        ("snapshot",),
        1,
        17,
        2,
        10,
        _hash("1"),
        _hash("2"),
        _hash("3"),
        _hash("4"),
        _hash("5"),
        (_hash("6"),),
        (_hash("7"),),
        "exact",
        _hash("8"),
    )


def _report(
    suite: EvaluationSuite, policy_sha256: str, execution_sha256: str
) -> EvaluationReport:
    episode = EpisodeMetrics(
        "snapshot",
        0,
        suite.episode_seed("snapshot", 0),
        3,
        8,
        5,
        1,
        1,
        True,
        False,
        0,
        90,
        90,
    )
    return EvaluationReport(
        suite.sha256,
        policy_sha256,
        _hash("9"),
        suite.runtime_identity_sha256,
        execution_sha256,
        (episode,),
    )


def _checkpoint(
    *,
    learner_seed: int,
    completed_updates: int,
    simulated_ticks: int,
    policy_sha256: str,
    suffix: str,
) -> TrainingCheckpointArtifact:
    return TrainingCheckpointArtifact(
        learner_seed,
        completed_updates,
        simulated_ticks,
        simulated_ticks,
        _hash("a"),
        _hash("b"),
        _hash("c"),
        _hash("d"),
        _hash(suffix),
        _hash("e"),
        policy_sha256,
    )


class R3BCodecTests(unittest.TestCase):
    def test_evaluation_identity_codecs_round_trip(self) -> None:
        logical = LogicalEvaluationCell(
            "validation",
            "stage",
            "family",
            _hash("a"),
            7,
            11,
            _hash("b"),
            (),
            4,
            2,
        )
        pair = CrossBackendCellPair(
            logical,
            "portable-snapshot",
            "exact-snapshot",
            _hash("c"),
            _hash("d"),
        )
        cross_backend = CrossBackendEvaluationManifest((pair,))
        suite = _suite()
        report = _report(suite, _hash("e"), _hash("f"))

        self.assertEqual(
            LogicalEvaluationCell.from_manifest(logical.manifest()), logical
        )
        self.assertEqual(CrossBackendCellPair.from_manifest(pair.manifest()), pair)
        self.assertEqual(
            CrossBackendEvaluationManifest.from_manifest(cross_backend.manifest()),
            cross_backend,
        )
        self.assertEqual(EvaluationSuite.from_manifest(suite.manifest()), suite)
        self.assertEqual(
            EpisodeMetrics.from_manifest(report.manifest()["episodes"][0]),
            report.episodes[0],
        )
        self.assertEqual(
            EvaluationReport.from_manifest(report.manifest(), suite=suite), report
        )

    def test_experiment_artifact_codecs_round_trip_with_references(self) -> None:
        arm = CandidateArm(100_000, 0.0001)
        job = TrialJob(
            _hash("a"),
            "validation",
            arm,
            19,
            20,
            False,
            _hash("b"),
            _hash("c"),
        )
        selection = CalibrationSelectionAuthorization(
            _hash("a"),
            (arm,),
            _hash("d"),
            _hash("e"),
            _hash("f"),
        )
        commitment = TestSuiteCommitment(_hash("a"), _hash("b"), _hash("c"))
        suite = _suite()
        first_policy, second_policy = _hash("a"), _hash("b")
        first_report = _report(suite, first_policy, _hash("c"))
        second_report = _report(suite, second_policy, _hash("d"))
        first = CheckpointEvaluation(
            _checkpoint(
                learner_seed=19,
                completed_updates=0,
                simulated_ticks=0,
                policy_sha256=first_policy,
                suffix="1",
            ),
            first_report,
        )
        second = CheckpointEvaluation(
            _checkpoint(
                learner_seed=19,
                completed_updates=1,
                simulated_ticks=10,
                policy_sha256=second_policy,
                suffix="2",
            ),
            second_report,
        )
        metrics = RawScoreMetricsArtifact(
            19, suite, suite, (first, second), second_report
        )
        baseline = BaselineEvidence(
            "no_action_long_wait",
            "complete",
            1,
            5.0,
            0,
            suite.sha256,
            _hash("1"),
            _hash("2"),
            _hash("3"),
            _hash("4"),
            _hash("5"),
            _hash("6"),
        )
        baseline_commitment = SealedBaselineBatchCommitment(
            _hash("1"),
            _hash("2"),
            _hash("3"),
            (("no_action_long_wait", _hash("4")),),
            1,
        )
        sealed = SealedTestAuthorization(
            _hash("1"),
            _hash("2"),
            arm.arm_id,
            CandidateArm(250_000, 0.0001).arm_id,
            _hash("3"),
            _hash("4"),
            _hash("5"),
            _hash("6"),
            baseline_commitment.sha256,
        )

        self.assertEqual(CandidateArm.from_manifest(arm.manifest()), arm)
        self.assertEqual(TrialJob.from_manifest(job.manifest()), job)
        self.assertEqual(
            CalibrationSelectionAuthorization.from_manifest(selection.manifest()),
            selection,
        )
        self.assertEqual(
            TestSuiteCommitment.from_manifest(commitment.manifest()), commitment
        )
        self.assertEqual(
            TrainingCheckpointArtifact.from_manifest(first.checkpoint.manifest()),
            first.checkpoint,
        )
        self.assertEqual(
            CheckpointEvaluation.from_manifest(first.manifest(), report=first_report),
            first,
        )
        self.assertEqual(
            RawScoreMetricsArtifact.from_manifest(
                metrics.manifest(),
                curve_suite=suite,
                final_suite=suite,
                reports=(first_report, second_report),
                final_report=second_report,
            ),
            metrics,
        )
        self.assertEqual(BaselineEvidence.from_manifest(baseline.manifest()), baseline)
        self.assertEqual(
            SealedBaselineBatchCommitment.from_manifest(baseline_commitment.manifest()),
            baseline_commitment,
        )
        self.assertEqual(
            SealedTestAuthorization.from_manifest(sealed.manifest()), sealed
        )

    def test_result_codecs_require_their_typed_audit_dependencies(self) -> None:
        original = outcomes(
            (TEST_PLAN.calibration_learner_seeds[0],),
            auc=10,
            final=11,
            phase="calibration",
            budget=TEST_PLAN.calibration_budgets_updates[-1],
        )[0]
        evidence = original.engineering_evidence
        assert evidence is not None
        restored_evidence = EngineeringEvidence.from_manifest(
            evidence.manifest(),
            checkpoint_resume_artifact=evidence.checkpoint_resume_artifact,
            exact_backend_parity_artifact=evidence.exact_backend_parity_artifact,
        )
        restored_outcome = LearnerOutcome.from_manifest(
            original.manifest(),
            metrics_artifact=original.metrics_artifact,
            engineering_evidence=restored_evidence,
        )
        result = ArmPhaseResult(
            evidence.arm_id,
            "calibration",
            "complete",
            TEST_PLAN.calibration_budgets_updates[-1],
            (restored_outcome,),
        )
        self.assertEqual(
            ArmPhaseResult.from_manifest(
                result.manifest(), outcomes=(restored_outcome,)
            ),
            result,
        )
        with self.assertRaisesRegex(ValueError, "dependencies differ"):
            LearnerOutcome.from_manifest(
                original.manifest(),
                metrics_artifact=original.metrics_artifact,
                engineering_evidence=None,
            )

    def test_codecs_reject_schema_type_and_reference_tampering(self) -> None:
        arm = CandidateArm(100_000, 0.0001)
        unknown = {**arm.manifest(), "unknown": True}
        with self.assertRaisesRegex(ValueError, "keys differ"):
            CandidateArm.from_manifest(unknown)
        noncanonical = {**arm.manifest(), "learning_rate": 1}
        noncanonical["arm_id"] = "alpha-0100000-lr-1"
        with self.assertRaisesRegex(ValueError, "field types"):
            CandidateArm.from_manifest(noncanonical)

        suite = _suite()
        report = _report(suite, _hash("a"), _hash("b"))
        wrong_suite = EvaluationSuite.from_manifest(
            {**suite.manifest(), "suite_id": "other-suite"}
        )
        with self.assertRaisesRegex(ValueError, "suite reference mismatch"):
            EvaluationReport.from_manifest(report.manifest(), suite=wrong_suite)
        with self.assertRaisesRegex(ValueError, "report reference mismatch"):
            CheckpointEvaluation.from_manifest(
                {
                    "version": "r3b-checkpoint-evaluation-v2",
                    "checkpoint": _checkpoint(
                        learner_seed=1,
                        completed_updates=0,
                        simulated_ticks=0,
                        policy_sha256=report.policy_sha256,
                        suffix="1",
                    ).manifest(),
                    "report_sha256": _hash("c"),
                },
                report=report,
            )

        logical = LogicalEvaluationCell(
            "validation",
            "stage",
            "family",
            _hash("a"),
            7,
            11,
            _hash("b"),
            (),
            4,
            2,
        )
        pair = CrossBackendCellPair(
            logical, "portable", "exact", _hash("c"), _hash("d")
        )
        tampered_pair = {**pair.manifest(), "logical_cell_sha256": _hash("e")}
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            CrossBackendCellPair.from_manifest(tampered_pair)

        commitment = SealedBaselineBatchCommitment(
            _hash("1"),
            _hash("2"),
            _hash("3"),
            (("no_action_long_wait", _hash("4")),),
            1,
        )
        portable_primary = commitment.manifest()
        portable_primary["reports"] = {
            **portable_primary["reports"],
            "primary_backend": "portable",
        }
        with self.assertRaisesRegex(ValueError, "malformed"):
            SealedBaselineBatchCommitment.from_manifest(portable_primary)

    def test_exact_resume_proof_has_no_public_deserializer(self) -> None:
        self.assertFalse(hasattr(ExactResumeArtifact, "from_manifest"))


if __name__ == "__main__":
    unittest.main()
