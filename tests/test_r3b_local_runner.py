from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from irisu_rl.r3b_artifacts import ArtifactStore
from irisu_rl.r3b_experiments import CandidateArm, TrialJob
from irisu_rl.r3b_local_runner import (
    _checkpoint_due_after_update,
    _load_claim,
    _load_claim_intent,
    _load_completed_training_result,
    _reconcile_sealed_training_failure,
    _run_update_and_checkpoint_due,
    _write_claim,
)
from irisu_rl.r3b_operational import JobClaim


class R3BLocalRunnerTests(unittest.TestCase):
    def test_checkpoint_waits_for_optimizer_progress_during_tail_drain(self) -> None:
        self.assertFalse(
            _checkpoint_due_after_update(
                600,
                600,
                optimizer_completed=False,
                target=1000,
                interval=50,
            )
        )
        self.assertTrue(
            _checkpoint_due_after_update(
                599,
                600,
                optimizer_completed=True,
                target=1000,
                interval=50,
            )
        )
        self.assertFalse(
            _checkpoint_due_after_update(
                600,
                601,
                optimizer_completed=True,
                target=1000,
                interval=50,
            )
        )
        self.assertTrue(
            _checkpoint_due_after_update(
                999,
                1000,
                optimizer_completed=True,
                target=1000,
                interval=50,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "advanced unexpectedly"):
            _checkpoint_due_after_update(
                600,
                602,
                optimizer_completed=True,
                target=1000,
                interval=50,
            )

    def test_training_attempt_loop_skips_unchanged_checkpoint_boundaries(self) -> None:
        class Session:
            def __init__(self) -> None:
                self.trainer = SimpleNamespace(
                    schedule=SimpleNamespace(completed_updates=599)
                )
                self.results = iter(
                    ((600, object()), (600, None), (600, None), (601, object()))
                )

            def run_update(self) -> object:
                completed_updates, optimizer = next(self.results)
                self.trainer.schedule.completed_updates = completed_updates
                return SimpleNamespace(optimizer=optimizer)

        session = Session()
        published = []
        for _ in range(4):
            completed_updates, checkpoint_due = _run_update_and_checkpoint_due(
                session,  # type: ignore[arg-type]
                target=1000,
                interval=50,
            )
            if checkpoint_due:
                published.append(completed_updates)
        self.assertEqual(published, [600])

    def test_training_attempt_rejects_result_and_clock_disagreement(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not advance"):
            _checkpoint_due_after_update(
                600,
                600,
                optimizer_completed=True,
                target=1000,
                interval=50,
            )
        with self.assertRaisesRegex(RuntimeError, "without an optimizer"):
            _checkpoint_due_after_update(
                600,
                601,
                optimizer_completed=False,
                target=1000,
                interval=50,
            )

    def test_claim_secret_round_trips_privately(self) -> None:
        claim = JobClaim("a" * 64, "calibration", "b" * 64, "worker", 0, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secrets" / f"{claim.job_sha256}.claim.json"
            _write_claim(path, claim)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
            self.assertEqual(_load_claim(path), claim)
            with self.assertRaises(FileExistsError):
                _write_claim(path, claim)

    def test_claim_secret_rejects_permissions_and_noncanonical_bytes(self) -> None:
        claim = JobClaim("a" * 64, "calibration", "b" * 64, "worker", 0, None)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "secrets" / f"{claim.job_sha256}.claim.json"
            _write_claim(path, claim)
            os.chmod(path, 0o644)
            with self.assertRaisesRegex(ValueError, "not private"):
                _load_claim(path)
            os.chmod(path, 0o600)
            value = json.loads(path.read_bytes())
            path.write_text(json.dumps(value, indent=2), encoding="utf-8")
            os.chmod(path, 0o600)
            with self.assertRaisesRegex(ValueError, "schema differs"):
                _load_claim(path)

    def test_claim_intent_loader_normalizes_malformed_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "intent.json"
            path.write_bytes(b"\xff")
            os.chmod(path, 0o600)

            with self.assertRaisesRegex(ValueError, "intent is malformed"):
                _load_claim_intent(path)

    def test_reconciles_failure_already_recorded_during_runner_build(self) -> None:
        sealed_run, job = object(), object()
        authorization = SimpleNamespace(
            sealed_run=sealed_run,
            job=job,
            assert_running=mock.Mock(side_effect=RuntimeError("not active")),
        )
        ledger = SimpleNamespace(
            job_state=mock.Mock(
                return_value=(
                    "failure",
                    None,
                    "runner construction failed: RuntimeError: worker",
                )
            ),
            fail_job=mock.Mock(),
        )
        workflow = SimpleNamespace(reconcile_sealed_failure=mock.Mock())

        _reconcile_sealed_training_failure(
            workflow=workflow,  # type: ignore[arg-type]
            authorization=authorization,  # type: ignore[arg-type]
            ledger=ledger,  # type: ignore[arg-type]
            reason="RuntimeError: worker",
        )

        ledger.fail_job.assert_not_called()
        workflow.reconcile_sealed_failure.assert_called_once_with(
            ledger=ledger,
            sealed_run=sealed_run,
            job=job,
            failure_reason="runner construction failed: RuntimeError: worker",
        )

    def test_trained_job_result_recovers_for_pending_evaluation(self) -> None:
        job = TrialJob(
            "1" * 64,
            "calibration",
            CandidateArm(0, 0.0001),
            7,
            300,
            False,
            "2" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = ArtifactStore(root / "artifacts").publish(
                kind="irisu.r3b.training-checkpoint",
                version="r3b-training-checkpoint-package-v2",
                payload={
                    "job_sha256": job.sha256,
                    "completed_updates": job.budget_updates,
                    "simulated_ticks": 614_400,
                },
            )
            result = _load_completed_training_result(
                root,
                job,
                {
                    "latest_checkpoint": {
                        "completed_updates": job.budget_updates,
                        "artifact_sha256": artifact.artifact_id,
                    }
                },
            )
        self.assertTrue(result.training_complete)
        self.assertEqual(result.completed_updates, job.budget_updates)
        self.assertEqual(result.simulated_ticks, 614_400)
        self.assertEqual(result.checkpoint_artifact_sha256, artifact.artifact_id)


if __name__ == "__main__":
    unittest.main()
