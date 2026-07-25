from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from irisu_rl.cpu_parallelism import resolve_training_cpu_plan
from irisu_rl.r3b_local_runner import _write_claim, _write_claim_intent
from irisu_rl.r3b_operational import JobClaim
from irisu_rl.r3b_training_batch import (
    _bind_cpu_wave_identity,
    _clear_cpu_wave,
    _load_cpu_wave,
    _prepare_claims,
    _publish_cpu_wave,
    _train_one,
)


class R3BTrainingBatchTests(unittest.TestCase):
    def test_training_child_applies_budget_before_running_job(self) -> None:
        result = object()
        with (
            mock.patch("os.sched_setaffinity") as affinity,
            mock.patch(
                "irisu_rl.r3b_training_batch.run_local_canonical_updates",
                return_value=result,
            ) as train,
        ):
            self.assertIs(
                _train_one(
                    "/run",
                    "/worker",
                    "calibration",
                    "batch-1",
                    "a" * 64,
                    (0, 1, 2),
                    None,
                ),
                result,
            )
        affinity.assert_called_once_with(0, (0, 1, 2))
        train.assert_called_once_with(
            "/run",
            worker_path="/worker",
            max_new_updates=2**31 - 1,
            owner="batch-1",
            phase="calibration",
            authorization=None,
            allow_parallel_claims=True,
            expected_job_sha256="a" * 64,
        )

    def test_cpu_plan_and_owner_are_immutable_within_one_wave(self) -> None:
        first = resolve_training_cpu_plan(
            workers_per_job=16,
            torch_threads_per_job=4,
            affinity_cpus=16,
        )
        changed = resolve_training_cpu_plan(
            workers_per_job=16,
            torch_threads_per_job=4,
            affinity_cpus=8,
        )
        claim = JobClaim("a" * 64, "calibration", "1" * 64, "batch-1", 0, None)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _bind_cpu_wave_identity(
                root,
                phase="calibration",
                owner="batch",
                plan=first,
            )
            _bind_cpu_wave_identity(
                root,
                phase="calibration",
                owner="batch",
                plan=first,
            )
            with self.assertRaisesRegex(ValueError, "changed across resume"):
                _bind_cpu_wave_identity(
                    root,
                    phase="calibration",
                    owner="batch",
                    plan=changed,
                )
            with self.assertRaisesRegex(ValueError, "changed across resume"):
                _bind_cpu_wave_identity(
                    root,
                    phase="calibration",
                    owner="different",
                    plan=first,
                )
            _write_claim(root / "secrets" / f"{claim.job_sha256}.claim.json", claim)
            _publish_cpu_wave(
                root,
                phase="calibration",
                owner="batch",
                plan=first,
                claims=(claim,),
            )
            self.assertEqual(
                _load_cpu_wave(
                    root,
                    phase="calibration",
                    owner="batch",
                    plan=first,
                ),
                (claim,),
            )
            with self.assertRaisesRegex(ValueError, "changed across resume"):
                _load_cpu_wave(
                    root,
                    phase="calibration",
                    owner="batch",
                    plan=changed,
                )
            with self.assertRaisesRegex(ValueError, "changed across resume"):
                _load_cpu_wave(
                    root,
                    phase="calibration",
                    owner="different",
                    plan=first,
                )
            _clear_cpu_wave(root, "calibration")
            self.assertIsNone(
                _load_cpu_wave(
                    root,
                    phase="calibration",
                    owner="different",
                    plan=changed,
                )
            )

    def test_claim_wave_is_durable_and_reused_before_spawn(self) -> None:
        claims = (
            JobClaim("a" * 64, "calibration", "1" * 64, "batch-1", 0, None),
            JobClaim("b" * 64, "calibration", "2" * 64, "batch-2", 0, None),
        )
        workflow = SimpleNamespace(
            claim_next=mock.Mock(side_effect=claims),
            resume_unstarted_claim=mock.Mock(),
            job_record=mock.Mock(return_value={"status": "claimed"}),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = _prepare_claims(
                root,
                workflow=workflow,
                phase="calibration",
                owner="batch",
                count=2,
            )
            self.assertEqual(prepared, claims)
            self.assertEqual(workflow.claim_next.call_count, 2)

            reused = _prepare_claims(
                root,
                workflow=workflow,
                phase="calibration",
                owner="batch",
                count=2,
            )
            self.assertEqual(reused, claims)
            self.assertEqual(workflow.claim_next.call_count, 2)

    def test_precommitted_intent_retries_atomic_claim(self) -> None:
        claim = JobClaim("a" * 64, "calibration", "1" * 64, "batch-1", 0, None)
        workflow = SimpleNamespace(
            claim_next=mock.Mock(return_value=claim),
            resume_unstarted_claim=mock.Mock(return_value=None),
            job_record=mock.Mock(return_value={"status": "claimed"}),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_claim_intent(
                root / "secrets" / "calibration.batch-1.intent.json",
                phase="calibration",
                token=claim.token,
                owner=claim.owner,
            )
            self.assertEqual(
                _prepare_claims(
                    root,
                    workflow=workflow,
                    phase="calibration",
                    owner="batch",
                    count=1,
                ),
                (claim,),
            )
        workflow.resume_unstarted_claim.assert_called_once()
        workflow.claim_next.assert_called_once_with(
            "calibration", owner="batch-1", token=claim.token
        )


if __name__ == "__main__":
    unittest.main()
