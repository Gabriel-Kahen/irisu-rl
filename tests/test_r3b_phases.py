from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from irisu_rl.r3b_experiments import CandidateArm
from irisu_rl.r3b_operational import R3BOperationalConfig, R3BWorkflow
from irisu_rl.r3b_phases import (
    PublishedSealedAuthorization,
    acquire_sealed_job,
)
from tests.test_r3b_experiments import (
    TEST_EVALUATION_SUITE,
    TEST_PLAN,
    VALIDATION_EVALUATION_SUITE,
    authorization_validation_results,
    validation_context,
)


ROOT = Path(__file__).resolve().parents[1]


def _complete_phase(workflow: R3BWorkflow, phase: str) -> None:
    while (claim := workflow.claim_next(phase, owner="phase-setup")) is not None:
        budget = int(workflow.job_record(claim.job_sha256)["budget_updates"])
        workflow.begin(claim)
        workflow.record_checkpoint(claim, budget, claim.job_sha256)
        workflow.complete(claim, claim.job_sha256)


class R3BPhaseTests(unittest.TestCase):
    def test_restarted_running_sealed_job_becomes_terminal_failure(self) -> None:
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
            workflow = R3BWorkflow.create(
                root / "workflow.sqlite3",
                run_id="sealed-orphan-test",
                run_class="smoke",
                plan=plan,
                config=R3BOperationalConfig.from_toml(
                    ROOT / "configs/rl/experiments/r3b-operational-v1.toml"
                ),
                snapshot_bundle_sha256="1" * 64,
                source_identity_sha256="2" * 64,
            )
            _complete_phase(workflow, "calibration")
            workflow.append_jobs(
                plan.trial_jobs("validation", validation_run),
                authorization=validation_run,
            )
            _complete_phase(workflow, "validation")
            workflow.append_jobs(
                plan.trial_jobs("test", sealed_run),
                authorization=sealed_run,
            )
            inputs = SimpleNamespace(root=root, workflow=workflow, plan=plan)
            sealed = PublishedSealedAuthorization(
                sealed_run,
                "a" * 64,
                "b" * 64,
            )
            acquired = acquire_sealed_job(
                inputs,  # type: ignore[arg-type]
                sealed,
                owner="canonical-runner",
            )
            workflow.begin(acquired.claim)
            acquired.ledger.begin_job(acquired.lease)

            with self.assertRaisesRegex(RuntimeError, "orphaned"):
                acquire_sealed_job(
                    inputs,  # type: ignore[arg-type]
                    sealed,
                    owner="canonical-runner",
                )
            self.assertEqual(
                workflow.job_record(acquired.claim.job_sha256)["status"],
                "failed",
            )
            self.assertEqual(
                acquired.ledger.job_state(sealed_run, acquired.lease.job)[0],
                "failure",
            )


if __name__ == "__main__":
    unittest.main()
