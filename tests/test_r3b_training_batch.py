from __future__ import annotations

import json
import os
import subprocess
import sys
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
    _preflight_batch_recovery,
    _prepare_claims,
    _publish_cpu_wave,
    _set_process_affinity,
    _train_one,
    run_canonical_training_batch,
)


class R3BTrainingBatchTests(unittest.TestCase):
    def test_training_child_applies_budget_before_running_job(self) -> None:
        result = object()
        with (
            mock.patch("irisu_rl.r3b_training_batch._set_process_affinity") as affinity,
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
        affinity.assert_called_once_with((0, 1, 2))
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

    def test_process_affinity_covers_every_existing_thread(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task_root = Path(directory)
            for task_id in ("101", "102"):
                (task_root / task_id).mkdir()
            with (
                mock.patch("os.sched_setaffinity") as set_affinity,
                mock.patch(
                    "os.sched_getaffinity",
                    return_value=frozenset({2, 3}),
                ) as get_affinity,
            ):
                _set_process_affinity((2, 3), task_root=task_root)
        self.assertEqual(
            set_affinity.call_args_list,
            [
                mock.call(0, frozenset({2, 3})),
                mock.call(101, frozenset({2, 3})),
                mock.call(102, frozenset({2, 3})),
            ],
        )
        self.assertEqual(
            get_affinity.call_args_list,
            [mock.call(101), mock.call(102)],
        )

    @unittest.skipUnless(
        sys.platform == "linux"
        and hasattr(os, "sched_getaffinity")
        and hasattr(os, "sched_setaffinity"),
        "process-wide affinity requires Linux",
    )
    def test_process_affinity_restricts_threads_and_descendants(self) -> None:
        target = tuple(sorted(os.sched_getaffinity(0))[:2])
        script = """
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from irisu_rl.r3b_training_batch import _set_process_affinity

target = tuple(json.loads(sys.argv[1]))
gate = threading.Event()
threads = [threading.Thread(target=gate.wait, daemon=True) for _ in range(3)]
for thread in threads:
    thread.start()
_set_process_affinity(target)
masks = [
    sorted(os.sched_getaffinity(int(entry.name)))
    for entry in Path("/proc/self/task").iterdir()
    if entry.name.isdecimal()
]
descendant = json.loads(
    subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))",
        ],
        text=True,
    )
)
print(json.dumps({"masks": masks, "descendant": descendant}))
"""
        completed = subprocess.run(
            [sys.executable, "-c", script, json.dumps(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        result = json.loads(completed.stdout)
        self.assertTrue(result["masks"])
        self.assertTrue(all(mask == list(target) for mask in result["masks"]))
        self.assertEqual(result["descendant"], list(target))

    def test_cpu_policy_and_owner_are_immutable_within_one_wave(self) -> None:
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
        changed_policy = resolve_training_cpu_plan(
            workers_per_job=16,
            torch_threads_per_job=4,
            affinity_cpus=8,
            target_percent=70,
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
                    owner="batch",
                    plan=changed_policy,
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
            self.assertEqual(
                _load_cpu_wave(
                    root,
                    phase="calibration",
                    owner="batch",
                    plan=changed,
                ),
                (claim,),
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
                count=1,
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

    def test_active_claim_reconciles_crash_window_intent(self) -> None:
        claim = JobClaim("a" * 64, "calibration", "1" * 64, "batch-1", 0, None)
        next_claim = JobClaim(
            "b" * 64,
            "calibration",
            "2" * 64,
            "batch-1",
            0,
            None,
        )
        workflow = SimpleNamespace(
            claim_next=mock.Mock(),
            resume_unstarted_claim=mock.Mock(),
            job_record=mock.Mock(return_value={"status": "claimed"}),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            secret_root = root / "secrets"
            intent = secret_root / "calibration.batch-1.intent.json"
            _write_claim_intent(
                intent,
                phase=claim.phase,
                token=claim.token,
                owner=claim.owner,
            )
            _write_claim(secret_root / f"{claim.job_sha256}.claim.json", claim)
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
            self.assertFalse(intent.exists())
            workflow.job_record.return_value = {"status": "completed"}
            workflow.claim_next.return_value = next_claim
            self.assertEqual(
                _prepare_claims(
                    root,
                    workflow=workflow,
                    phase="calibration",
                    owner="batch",
                    count=1,
                ),
                (next_claim,),
            )
        workflow.claim_next.assert_called_once()
        workflow.resume_unstarted_claim.assert_not_called()

    def test_foreign_active_claim_blocks_batch_claims(self) -> None:
        claim = JobClaim(
            "a" * 64,
            "calibration",
            "1" * 64,
            "canonical-runner",
            0,
            None,
        )
        workflow = SimpleNamespace(
            claim_next=mock.Mock(),
            resume_unstarted_claim=mock.Mock(),
            job_record=mock.Mock(return_value={"status": "running"}),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_claim(root / "secrets" / f"{claim.job_sha256}.claim.json", claim)
            with self.assertRaisesRegex(RuntimeError, "cannot adopt active claims"):
                _prepare_claims(
                    root,
                    workflow=workflow,
                    phase="calibration",
                    owner="batch",
                    count=2,
                )
        workflow.claim_next.assert_not_called()

    def test_batch_submits_parallel_training_before_sequential_evaluation(self) -> None:
        claims = tuple(
            JobClaim(
                character * 64,
                "calibration",
                str(index) * 64,
                f"batch-{index}",
                0,
                None,
            )
            for index, character in enumerate(("a", "b", "c"), start=1)
        )
        records: dict[str, int] = {}

        def job_record(job_sha256: str) -> dict[str, str]:
            records[job_sha256] = records.get(job_sha256, 0) + 1
            if job_sha256 == claims[2].job_sha256:
                return {"status": "claimed"}
            return {"status": "claimed" if records[job_sha256] == 1 else "completed"}

        workflow = SimpleNamespace(job_record=mock.Mock(side_effect=job_record))
        events: list[object] = []
        results = {
            claim.job_sha256: SimpleNamespace(
                job_sha256=claim.job_sha256,
                training_complete=True,
            )
            for claim in claims[:2]
        }

        class Future:
            def __init__(self, job_sha256: str) -> None:
                self.job_sha256 = job_sha256

            def result(self):
                events.append(("result", self.job_sha256))
                return results[self.job_sha256]

            def cancel(self) -> None:
                events.append(("cancel", self.job_sha256))

        class Executor:
            def __init__(self, *, max_workers: int, **_kwargs) -> None:
                self._max_workers = max_workers

            def submit(self, _function, *args):
                job_sha256 = args[4]
                events.append(("submit", job_sha256))
                return Future(job_sha256)

        def evaluate(*_args, job_sha256: str, **_kwargs):
            self.assertIn("stop", events)
            events.append(("evaluate", job_sha256))
            return SimpleNamespace(job_sha256=job_sha256)

        plan = resolve_training_cpu_plan(
            workers_per_job=16,
            torch_threads_per_job=4,
            affinity_cpus=16,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker"
            library = root / "library"
            worker.touch()
            library.touch()
            with (
                mock.patch(
                    "irisu_rl.r3b_training_batch.R3BWorkflow",
                    return_value=workflow,
                ),
                mock.patch("irisu_rl.r3b_training_batch._preflight_batch_recovery"),
                mock.patch("irisu_rl.r3b_training_batch._bind_cpu_wave_identity"),
                mock.patch(
                    "irisu_rl.r3b_training_batch._load_cpu_wave",
                    return_value=claims,
                ),
                mock.patch(
                    "irisu_rl.r3b_training_batch.multiprocessing.get_context",
                    return_value=SimpleNamespace(Event=lambda: object()),
                ),
                mock.patch(
                    "irisu_rl.r3b_training_batch.ProcessPoolExecutor",
                    Executor,
                ),
                mock.patch(
                    "irisu_rl.r3b_training_batch._capture_evaluation_process_groups",
                    side_effect=lambda _executor: events.append("capture"),
                ),
                mock.patch(
                    "irisu_rl.r3b_training_batch._stop_evaluation_executor",
                    side_effect=lambda _executor: events.append("stop"),
                ),
                mock.patch(
                    "irisu_rl.r3b_training_batch.evaluate_trained_canonical_job",
                    side_effect=evaluate,
                ),
                mock.patch("irisu_rl.r3b_training_batch._clear_cpu_wave") as clear_wave,
            ):
                result = run_canonical_training_batch(
                    root,
                    exact_worker_path=worker,
                    portable_library_path=library,
                    phase="calibration",
                    owner="batch",
                    cpu_plan=plan,
                )
        self.assertEqual(
            events[:3],
            [
                ("submit", claims[0].job_sha256),
                ("submit", claims[1].job_sha256),
                "capture",
            ],
        )
        self.assertEqual(
            [value.job_sha256 for value in result.training],
            [claims[0].job_sha256, claims[1].job_sha256],
        )
        self.assertEqual(
            [value.job_sha256 for value in result.evaluation],
            [claims[0].job_sha256, claims[1].job_sha256],
        )
        self.assertFalse(result.wave_complete)
        self.assertEqual(result.remaining_jobs, 1)
        clear_wave.assert_not_called()

    def test_preflight_rejects_foreign_workflow_claim_without_secret(self) -> None:
        workflow = SimpleNamespace(
            phase_job_records=mock.Mock(
                return_value=(
                    {
                        "job_sha256": "a" * 64,
                        "owner": "canonical-runner",
                        "status": "claimed",
                    },
                )
            )
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(RuntimeError, "cannot adopt active claims"),
        ):
            _preflight_batch_recovery(
                Path(directory),
                workflow=workflow,
                phase="calibration",
                owner="batch",
            )

    def test_smaller_resume_recovers_higher_slot_intent_without_secret(self) -> None:
        claim = JobClaim("c" * 64, "calibration", "3" * 64, "batch-3", 0, None)
        workflow = SimpleNamespace(
            phase_job_records=mock.Mock(
                return_value=(
                    {
                        "job_sha256": claim.job_sha256,
                        "owner": claim.owner,
                        "status": "claimed",
                    },
                )
            ),
            job_record=mock.Mock(),
            resume_unstarted_claim=mock.Mock(return_value=claim),
            claim_next=mock.Mock(return_value=None),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            intent = root / "secrets" / "calibration.batch-3.intent.json"
            _write_claim_intent(
                intent,
                phase=claim.phase,
                token=claim.token,
                owner=claim.owner,
            )
            _preflight_batch_recovery(
                root,
                workflow=workflow,
                phase="calibration",
                owner="batch",
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
            self.assertFalse(intent.exists())
            self.assertTrue(
                (root / "secrets" / f"{claim.job_sha256}.claim.json").is_file()
            )
        workflow.resume_unstarted_claim.assert_called_once_with(
            "calibration",
            owner=claim.owner,
            token=claim.token,
        )


if __name__ == "__main__":
    unittest.main()
