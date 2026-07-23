from __future__ import annotations

import hashlib
import sqlite3
import tempfile
import threading
import unittest
from contextlib import closing
from dataclasses import replace
from pathlib import Path

from irisu_rl.r3b_artifacts import ArtifactStore, ArtifactStoreError
from irisu_rl.r3b_experiments import _verified_exact_resume_artifact, load_plan
from irisu_rl.r3b_operational import (
    CANONICAL_EXACT_SNAPSHOT_BUNDLE_SHA256,
    CANONICAL_OPERATIONAL_CONFIG_SHA256,
    CANONICAL_PAIRING_SHA256S,
    CANONICAL_PORTABLE_SNAPSHOT_BUNDLE_SHA256,
    R3BOperationalConfig,
    R3BWorkflow,
)
from tests.test_r3b_experiments import validation_authorization


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/rl/experiments/r3b-completion-v1.toml"
CONFIG = ROOT / "configs/rl/experiments/r3b-operational-v1.toml"
HASH = hashlib.sha256(b"nonzero").hexdigest()


class R3BOperationalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = load_plan(PLAN)
        self.config = R3BOperationalConfig.from_toml(CONFIG)
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "workflow.sqlite3"
        self.workflow = R3BWorkflow.create(
            self.path,
            run_id="unit-run",
            run_class="smoke",
            plan=self.plan,
            config=self.config,
            snapshot_bundle_sha256=HASH,
            source_identity_sha256=HASH,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_config_binds_all_nonplan_choices_and_rejects_unknown_keys(self) -> None:
        self.assertEqual(self.config.primary_backend, "exact")
        self.assertFalse(self.config.transfer_eligible)
        manifest = self.config.manifest()
        self.assertEqual(
            R3BOperationalConfig.from_manifest(manifest).sha256,
            self.config.sha256,
        )
        manifest["unknown"] = 1
        with self.assertRaisesRegex(ValueError, "keys differ"):
            R3BOperationalConfig.from_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "invalid"):
            replace(self.config, ppo_entropy_coefficient=True)

    def test_progressive_calibration_uses_one_final_job_per_arm_seed(self) -> None:
        jobs = self.workflow.calibration_jobs(self.plan)
        self.assertEqual(
            len(jobs), len(self.plan.arms) * len(self.plan.calibration_learner_seeds)
        )
        self.assertEqual(
            {job.budget_updates for job in jobs},
            {self.plan.calibration_budgets_updates[-1]},
        )
        status = self.workflow.status()
        self.assertEqual(
            status["phases"]["calibration"]["pending"],  # type: ignore[index]
            len(jobs),
        )
        self.assertFalse(status["transfer_eligible"])

    def test_canonical_workflow_requires_every_preregistered_input(self) -> None:
        self.assertEqual(self.config.sha256, CANONICAL_OPERATIONAL_CONFIG_SHA256)
        canonical_path = Path(self.temporary.name) / "canonical.sqlite3"
        workflow = R3BWorkflow.create(
            canonical_path,
            run_id="canonical-run",
            run_class="canonical",
            plan=self.plan,
            config=self.config,
            snapshot_bundle_sha256=CANONICAL_EXACT_SNAPSHOT_BUNDLE_SHA256,
            portable_snapshot_bundle_sha256=(CANONICAL_PORTABLE_SNAPSHOT_BUNDLE_SHA256),
            pairing_sha256s=CANONICAL_PAIRING_SHA256S,
            source_identity_sha256=HASH,
            source_revision="a" * 40,
        )
        self.assertTrue(workflow.status()["acceptance_eligible"])

        with self.assertRaisesRegex(ValueError, "preregistered lock"):
            R3BWorkflow.create(
                Path(self.temporary.name) / "unlocked.sqlite3",
                run_id="unlocked",
                run_class="canonical",
                plan=self.plan,
                config=self.config,
                snapshot_bundle_sha256=HASH,
                portable_snapshot_bundle_sha256=(
                    CANONICAL_PORTABLE_SNAPSHOT_BUNDLE_SHA256
                ),
                pairing_sha256s=CANONICAL_PAIRING_SHA256S,
                source_identity_sha256=HASH,
                source_revision="a" * 40,
            )

    def test_claim_is_atomic_and_token_authorizes_mutations(self) -> None:
        claims = []

        def claim() -> None:
            claims.append(self.workflow.claim_next("calibration", owner="worker"))

        threads = [threading.Thread(target=claim) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        job_ids = [claim.job_sha256 for claim in claims if claim is not None]
        self.assertEqual(len(job_ids), 8)
        self.assertEqual(len(set(job_ids)), 8)
        selected = claims[0]
        assert selected is not None
        forged = type(selected)(
            selected.job_sha256,
            selected.phase,
            "f" * 64,
            selected.owner,
            0,
            None,
        )
        with self.assertRaisesRegex(ValueError, "unauthorized"):
            self.workflow.begin(forged)
        self.workflow.begin(selected)

    def test_precommitted_claim_token_recovers_post_commit_crash(self) -> None:
        token = "a" * 64
        claim = self.workflow.claim_next(
            "calibration", owner="durable-worker", token=token
        )
        assert claim is not None
        recovered = R3BWorkflow(self.path).resume_unstarted_claim(
            "calibration", owner="durable-worker", token=token
        )
        self.assertEqual(recovered, claim)
        self.workflow.begin(recovered)
        with self.assertRaisesRegex(RuntimeError, "already started"):
            self.workflow.resume_unstarted_claim(
                "calibration", owner="durable-worker", token=token
            )

    def test_resume_audit_is_anchored_by_active_trained_job(self) -> None:
        claim = self.workflow.claim_next("calibration", owner="auditor")
        assert claim is not None
        budget = int(self.workflow.job_record(claim.job_sha256)["budget_updates"])
        self.workflow.begin(claim)
        self.workflow.record_checkpoint(claim, budget, "a" * 64)
        self.workflow.mark_trained(claim)
        store = ArtifactStore(Path(self.temporary.name) / "artifacts")
        artifact = _verified_exact_resume_artifact(
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "4" * 64,
            "5" * 64,
            "5" * 64,
        )
        receipt = self.workflow.publish_resume_audit(
            claim,
            artifact,
            store=store,
            verifier_identity_sha256="6" * 64,
            build_identity_sha256="7" * 64,
        )
        repeated = self.workflow.publish_resume_audit(
            claim,
            artifact,
            store=store,
            verifier_identity_sha256="6" * 64,
            build_identity_sha256="7" * 64,
        )
        self.assertEqual(repeated, receipt)
        self.assertTrue(self.workflow.verify_resume_audit(claim.job_sha256, receipt))
        self.assertFalse(self.workflow.verify_resume_audit(claim.job_sha256, "c" * 64))
        with self.assertRaisesRegex(ArtifactStoreError, "verified"):
            self.workflow.publish_resume_audit(
                claim,
                object(),
                store=store,
                verifier_identity_sha256="6" * 64,
                build_identity_sha256="7" * 64,
            )

    def test_checkpoint_recovery_is_exact_and_sealed_recovery_is_forbidden(
        self,
    ) -> None:
        claim = self.workflow.claim_next("calibration", owner="first")
        assert claim is not None
        self.workflow.begin(claim)
        self.workflow.record_checkpoint(claim, 100, HASH)
        with self.assertRaisesRegex(ValueError, "eligible"):
            self.workflow.recover(
                claim.job_sha256,
                checkpoint_sha256=hashlib.sha256(b"wrong").hexdigest(),
                owner="second",
            )
        recovered = self.workflow.recover(
            claim.job_sha256, checkpoint_sha256=HASH, owner="second"
        )
        self.assertEqual(recovered.resume_from_update, 100)
        self.assertEqual(recovered.resume_checkpoint_sha256, HASH)
        with self.assertRaisesRegex(ValueError, "unauthorized"):
            self.workflow.record_checkpoint(claim, 200, HASH)
        self.workflow.begin(recovered)
        self.workflow.record_checkpoint(
            recovered, 200, hashlib.sha256(b"second").hexdigest()
        )

    def test_training_closes_authority_only_at_the_final_checkpoint(self) -> None:
        claim = self.workflow.claim_next("calibration", owner="worker")
        assert claim is not None
        self.workflow.begin(claim)
        self.workflow.record_checkpoint(claim, 0, HASH)
        with self.assertRaisesRegex(ValueError, "final checkpoint"):
            self.workflow.mark_trained(claim)
        budget = self.workflow.job_record(claim.job_sha256)["budget_updates"]
        self.workflow.record_checkpoint(
            claim, int(budget), hashlib.sha256(b"final").hexdigest()
        )
        self.workflow.mark_trained(claim)
        self.assertEqual(
            self.workflow.job_record(claim.job_sha256)["status"], "trained"
        )
        with self.assertRaisesRegex(ValueError, "unauthorized"):
            self.workflow.record_checkpoint(
                claim, int(budget), hashlib.sha256(b"late").hexdigest()
            )
        output = hashlib.sha256(b"evaluated-output").hexdigest()
        self.workflow.complete(claim, output)
        completed = self.workflow.job_record(claim.job_sha256)
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["output_sha256"], output)
        with self.assertRaisesRegex(ValueError, "unauthorized"):
            self.workflow.complete(claim, output)

    def test_append_requires_the_complete_typed_authorized_job_set(self) -> None:
        while claim := self.workflow.claim_next("calibration", owner="worker"):
            self.workflow.begin(claim)
            budget = int(self.workflow.job_record(claim.job_sha256)["budget_updates"])
            self.workflow.record_checkpoint(claim, budget, HASH)
            self.workflow.complete(claim, HASH)
        authorization = validation_authorization(self.plan, self.plan.learning_rates[0])
        jobs = self.plan.trial_jobs("validation", authorization)

        with self.assertRaisesRegex(ValueError, "complete authorized job set"):
            self.workflow.append_jobs(jobs[:-1], authorization=authorization)

        self.workflow.append_jobs(jobs, authorization=authorization)
        status = self.workflow.status()
        self.assertEqual(
            status["phases"]["validation"]["pending"],  # type: ignore[index]
            len(jobs),
        )

    def test_completion_requires_final_checkpoint_and_failures_are_terminal(
        self,
    ) -> None:
        claim = self.workflow.claim_next("calibration", owner="worker")
        assert claim is not None
        self.workflow.begin(claim)
        self.workflow.record_checkpoint(claim, 100, HASH)
        with self.assertRaisesRegex(ValueError, "final checkpoint"):
            self.workflow.complete(claim, HASH)
        self.workflow.fail(claim, "simulated crash")
        with self.assertRaisesRegex(ValueError, "unauthorized"):
            self.workflow.begin(claim)
        self.assertEqual(
            self.workflow.status()["phases"]["calibration"]["failed"],  # type: ignore[index]
            1,
        )

    def test_database_corruption_and_symlinks_fail_closed(self) -> None:
        self.workflow.verify()
        link = Path(self.temporary.name) / "workflow-link.sqlite3"
        link.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, "symlink"):
            R3BWorkflow(link).verify()

    def test_verify_rejects_state_not_derived_from_event_history(self) -> None:
        job = self.workflow.calibration_jobs(self.plan)[0]
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute(
                "UPDATE jobs SET status='completed',output_sha256=?,finished_ns=1 "
                "WHERE job_sha256=?",
                (HASH, job.sha256),
            )
            connection.commit()
        with self.assertRaisesRegex(ValueError, "event history"):
            self.workflow.verify()


if __name__ == "__main__":
    unittest.main()
