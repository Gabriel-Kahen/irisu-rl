from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from irisu_rl.r3b_artifacts import ArtifactIntegrityError, ArtifactStore
from irisu_rl.r3b_experiments import (
    CandidateArm,
    LearnerOutcome,
    SealedBaselineEvidenceArtifact,
    SealedLearnerOutcomeReference,
    finalize_persisted_sealed_test,
)
from irisu_rl.r3b_operational import R3BOperationalConfig, R3BWorkflow
from tests.test_r3b_experiments import (
    TEST_EVALUATION_SUITE,
    TEST_PLAN,
    VALIDATION_EVALUATION_SUITE,
    authorization_validation_results,
    valid_baseline_artifacts,
    validation_context,
)


ROOT = Path(__file__).resolve().parents[1]


def _complete_phase(workflow: R3BWorkflow, phase: str) -> None:
    while (claim := workflow.claim_next(phase, owner=f"prepare-{phase}")) is not None:
        record = workflow.job_record(claim.job_sha256)
        workflow.begin(claim)
        workflow.record_checkpoint(
            claim,
            int(record["budget_updates"]),
            claim.job_sha256,
        )
        workflow.complete(claim, claim.job_sha256)


def _bind_ledger_outcome(
    source: LearnerOutcome,
    *,
    job_sha256: str,
    seed_plan_sha256: str,
    lease_sha256: str,
) -> LearnerOutcome:
    evidence = source.engineering_evidence
    assert evidence is not None
    metrics = replace(
        source.metrics_artifact,
        checkpoints=tuple(
            replace(
                evaluated,
                checkpoint=replace(
                    evaluated.checkpoint,
                    job_sha256=job_sha256,
                ),
            )
            for evaluated in source.metrics_artifact.checkpoints
        ),
    )
    return replace(
        source,
        seed_plan_sha256=seed_plan_sha256,
        metrics_artifact=metrics,
        engineering_evidence=replace(
            evidence,
            job_sha256=job_sha256,
            sealed_job_lease_sha256=lease_sha256,
            metrics_sha256=metrics.sha256,
            final_checkpoint_artifact=metrics.checkpoints[-1].checkpoint,
            resume_checkpoint_artifact=metrics.checkpoints[-2].checkpoint,
        ),
    )


class SealedPersistenceIntegrationTests(unittest.TestCase):
    def test_restart_finalization_reconciles_and_rejects_tampering(self) -> None:
        plan = TEST_PLAN
        control = CandidateArm(0, plan.learning_rates[1])
        candidate = CandidateArm(100_000, plan.learning_rates[1])
        ledger, validation_run = validation_context(plan, control.learning_rate)
        validation_results = authorization_validation_results(
            plan,
            control,
            candidate,
            validation_run=validation_run,
        )
        sealed_run = ledger.authorize_once(
            plan,
            validation_run,
            validation_results,
            VALIDATION_EVALUATION_SUITE,
            TEST_EVALUATION_SUITE,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ArtifactStore(root / "artifacts")
            config = R3BOperationalConfig.from_toml(
                ROOT / "configs/rl/experiments/r3b-operational-v1.toml"
            )
            workflow = R3BWorkflow.create(
                root / "workflow.sqlite3",
                run_id="sealed-persistence-test",
                run_class="smoke",
                plan=plan,
                config=config,
                snapshot_bundle_sha256="1" * 64,
                source_identity_sha256="2" * 64,
            )
            _complete_phase(workflow, "calibration")
            validation_jobs = plan.trial_jobs("validation", validation_run)
            workflow.append_jobs(validation_jobs, authorization=validation_run)
            _complete_phase(workflow, "validation")
            test_jobs = plan.trial_jobs("test", sealed_run)
            workflow.append_jobs(test_jobs, authorization=sealed_run)

            for _job in test_jobs:
                claim = workflow.claim_next("test", owner="sealed-worker")
                assert claim is not None
                job = next(job for job in test_jobs if job.sha256 == claim.job_sha256)
                workflow.begin(claim)
                workflow.record_checkpoint(
                    claim,
                    job.budget_updates,
                    job.sha256,
                )
                workflow.mark_trained(claim)

            authorization_id = sealed_run.publish(store)
            baseline_artifacts = valid_baseline_artifacts(plan)
            baseline_package = SealedBaselineEvidenceArtifact.from_artifacts(
                sealed_run, baseline_artifacts
            )
            baseline_id = baseline_package.publish(store)
            baseline_lease = ledger.claim_baseline_batch(sealed_run)
            ledger.begin_baseline_batch(baseline_lease)
            committed = ledger.complete_baseline_batch(
                baseline_lease, baseline_artifacts
            )
            self.assertEqual(
                tuple(item.sha256 for item in baseline_package.evidence),
                tuple(item.sha256 for item in committed),
            )

            # Rebuild test-phase source outcomes with the authorized test suite.
            from tests.test_r3b_experiments import phase_result

            source_results = {
                candidate.arm_id: phase_result(
                    candidate,
                    "test",
                    plan.test_learner_seeds,
                    budget=plan.test_updates,
                    auc=110,
                    final=100,
                    authorization_sha256=sealed_run.sha256,
                ),
                control.arm_id: phase_result(
                    control,
                    "test",
                    plan.test_learner_seeds,
                    budget=plan.test_updates,
                    auc=100,
                    final=100,
                    authorization_sha256=sealed_run.sha256,
                ),
            }
            output_to_outcome: dict[str, LearnerOutcome] = {}
            reference_ids: list[str] = []
            references: dict[str, SealedLearnerOutcomeReference] = {}
            for job in test_jobs:
                lease = ledger.claim_job(sealed_run, job)
                ledger.begin_job(lease)
                source = next(
                    outcome
                    for outcome in source_results[job.arm.arm_id].outcomes
                    if outcome.learner_seed == job.learner_seed
                )
                outcome = _bind_ledger_outcome(
                    source,
                    job_sha256=job.sha256,
                    seed_plan_sha256=job.seed_plan_sha256,
                    lease_sha256=lease.sha256,
                )
                output_id = store.publish(
                    kind="test.rich-sealed-outcome",
                    version="test-rich-sealed-outcome-v1",
                    payload={
                        "job_sha256": job.sha256,
                        "outcome_sha256": outcome.sha256,
                    },
                ).artifact_id
                reference = SealedLearnerOutcomeReference.capture(
                    sealed_run, job, outcome, output_id
                )
                reference_ids.append(reference.publish(store))
                references[job.sha256] = reference
                output_to_outcome[output_id] = outcome
                ledger.complete_job(lease, outcome)

            # Crash point: the authoritative ledger is complete while every
            # workflow row still has only a final checkpoint and no output.
            self.assertEqual(
                {workflow.job_record(job.sha256)["status"] for job in test_jobs},
                {"trained"},
            )

            first_job = test_jobs[0]
            first_reference = references[first_job.sha256]
            first_outcome = output_to_outcome[first_reference.output_artifact_sha256]
            workflow.reconcile_sealed_completion(
                ledger=ledger,
                sealed_run=sealed_run,
                job=first_job,
                outcome_sha256=first_outcome.sha256,
                output_sha256=first_reference.output_artifact_sha256,
            )
            with self.assertRaisesRegex(RuntimeError, "output differs"):
                workflow.reconcile_sealed_completion(
                    ledger=ledger,
                    sealed_run=sealed_run,
                    job=first_job,
                    outcome_sha256=first_outcome.sha256,
                    output_sha256="f" * 64,
                )

            finalized = finalize_persisted_sealed_test(
                store=store,
                workflow=workflow,
                ledger=ledger,
                sealed_run=sealed_run,
                authorization_artifact_sha256=authorization_id,
                baseline_artifact_sha256=baseline_id,
                outcome_reference_sha256s=reference_ids,
                outcome_loader=output_to_outcome.__getitem__,
            )
            self.assertTrue(ledger.verify_finalized(finalized.report))
            self.assertEqual(
                {workflow.job_record(job.sha256)["status"] for job in test_jobs},
                {"completed"},
            )

            repeated = finalize_persisted_sealed_test(
                store=store,
                workflow=workflow,
                ledger=ledger,
                sealed_run=sealed_run,
                authorization_artifact_sha256=authorization_id,
                baseline_artifact_sha256=baseline_id,
                outcome_reference_sha256s=reference_ids,
                outcome_loader=output_to_outcome.__getitem__,
            )
            self.assertEqual(repeated, finalized)

            tampered_id = reference_ids[0]
            path = store.path_for(tampered_id)
            data = path.read_bytes()
            path.write_bytes(data.replace(b'"job_sha256":"', b'"job_sha256":"0'))
            with self.assertRaises(ArtifactIntegrityError):
                SealedLearnerOutcomeReference.load(store, tampered_id)


if __name__ == "__main__":
    unittest.main()
