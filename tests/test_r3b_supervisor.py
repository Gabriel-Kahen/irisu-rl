from __future__ import annotations

import hashlib
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from irisu_env import ExactWorkerClient, ExactWorkerNotFoundError, find_exact_worker
from irisu_rl.r3b_artifacts import ArtifactStore
from irisu_rl.r3b_experiments import CandidateArm, TrainingCheckpointArtifact, TrialJob
from irisu_rl.r3b_local_runner import _write_claim
from irisu_rl.r3b_lock import R3BRunLock, evaluator_lease_path
from irisu_rl.r3b_operational import JobClaim
from irisu_rl.r3b_supervisor import (
    _active_claim,
    _capture_evaluation_process_groups,
    _checkpoint_package,
    _deployment,
    _evaluation_lease_holders,
    _evaluation_worker_initializer,
    _EvaluationProcessGroup,
    _fresh_restored_checkpoint,
    _signal_evaluation_group,
    _stop_evaluation_executor,
    evaluate_trained_canonical_job,
)

from tests.test_r3a_session_resume import PORTABLE, build_session


def _hash(character: str) -> str:
    return character * 64


def _sleep_for_test(seconds: float) -> None:
    time.sleep(seconds)


def _fail_for_test() -> None:
    raise RuntimeError("expected evaluator failure")


def _spawn_leased_descendant(path: str) -> None:
    from irisu_env.exact_ipc import _exact_worker_pass_fds

    child = subprocess.Popen(["sleep", "60"], pass_fds=_exact_worker_pass_fds())
    Path(path).write_text(f"{os.getpid()} {child.pid}\n", encoding="utf-8")
    child.wait()


def _spawn_stopped_exact_descendant(path: str, worker: str) -> None:
    client = ExactWorkerClient(worker)
    os.kill(client._transport_pid, signal.SIGSTOP)
    Path(path).write_text(
        f"{os.getpid()} {client._transport_pid}\n",
        encoding="utf-8",
    )
    time.sleep(60)


def _write_gate_result(path: str) -> None:
    Path(path).write_text("released\n", encoding="utf-8")


try:
    EXACT_WORKER = find_exact_worker()
except ExactWorkerNotFoundError:
    EXACT_WORKER = None


class R3BSupervisorTests(unittest.TestCase):
    def test_evaluation_selects_one_job_among_parallel_claims(self) -> None:
        first = JobClaim(_hash("a"), "calibration", _hash("1"), "batch-1", 0, None)
        second = JobClaim(_hash("b"), "calibration", _hash("2"), "batch-2", 0, None)
        workflow = SimpleNamespace(
            job_record=lambda _job: {"status": "trained"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for claim in (first, second):
                _write_claim(root / "secrets" / f"{claim.job_sha256}.claim.json", claim)
            with self.assertRaisesRegex(RuntimeError, "exactly one active"):
                _active_claim(root, workflow, "calibration")
            self.assertEqual(
                _active_claim(
                    root,
                    workflow,
                    "calibration",
                    job_sha256=second.job_sha256,
                )[1],
                second,
            )

    @unittest.skipUnless(
        sys.platform == "linux", "parent-death isolation is Linux-only"
    )
    def test_evaluator_and_descendant_die_with_parent(self) -> None:
        script = """
import os
import subprocess
import time
from irisu_rl.r3b_supervisor import _evaluation_worker_initializer

owner_pid = os.getpid()
evaluator_pid = os.fork()
if evaluator_pid == 0:
    from irisu_rl.r3b_lock import evaluator_lease_path
    _evaluation_worker_initializer(owner_pid, str(evaluator_lease_path()))
    descendant = subprocess.Popen(["sleep", "60"])
    print(f"{os.getpid()} {descendant.pid}", flush=True)
    time.sleep(60)
    raise SystemExit(0)
time.sleep(60)
"""
        owner = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.assertIsNotNone(owner.stdout)
        line = owner.stdout.readline()
        evaluator_pid, descendant_pid = (int(value) for value in line.split())
        try:
            os.kill(owner.pid, signal.SIGKILL)
            owner.wait(timeout=2)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                states = []
                for pid in (evaluator_pid, descendant_pid):
                    try:
                        states.append(Path(f"/proc/{pid}/stat").read_text().split()[2])
                    except FileNotFoundError:
                        states.append("gone")
                if all(state in {"gone", "Z"} for state in states):
                    break
                time.sleep(0.02)
            self.assertTrue(all(state in {"gone", "Z"} for state in states), states)
        finally:
            for pid in (evaluator_pid, descendant_pid):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if owner.poll() is None:
                owner.kill()
                owner.wait()
            owner.stdout.close()

    @unittest.skipUnless(sys.platform == "linux", "evaluator isolation is Linux-only")
    def test_evaluation_work_waits_for_pinned_group_capture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "released"
            context = multiprocessing.get_context("spawn")
            ready_gate = context.Event()
            executor = ProcessPoolExecutor(
                max_workers=2,
                mp_context=context,
                initializer=_evaluation_worker_initializer,
                initargs=(
                    os.getpid(),
                    str(evaluator_lease_path(directory)),
                    ready_gate,
                ),
            )
            setattr(executor, "_irisu_ready_gate", ready_gate)
            setattr(
                executor,
                "_irisu_evaluator_lease_path",
                str(evaluator_lease_path(directory)),
            )
            try:
                futures = tuple(
                    executor.submit(_write_gate_result, str(path)) for _ in range(2)
                )
                time.sleep(0.1)
                self.assertFalse(path.exists())
                groups = _capture_evaluation_process_groups(executor)
                self.assertEqual(len(groups), 2)
                for future in futures:
                    future.result(timeout=5)
            finally:
                started = time.monotonic()
                _stop_evaluation_executor(executor)
                self.assertLess(time.monotonic() - started, 7)
            with R3BRunLock(Path(directory), global_directory=directory):
                pass

    @unittest.skipUnless(sys.platform == "linux", "evaluator isolation is Linux-only")
    def test_nine_evaluators_are_captured_before_startup_deadline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("spawn")
            ready_gate = context.Event()
            lease_path = str(evaluator_lease_path(directory))
            executor = ProcessPoolExecutor(
                max_workers=9,
                mp_context=context,
                initializer=_evaluation_worker_initializer,
                initargs=(os.getpid(), lease_path, ready_gate),
            )
            setattr(executor, "_irisu_ready_gate", ready_gate)
            setattr(executor, "_irisu_evaluator_lease_path", lease_path)
            try:
                futures = tuple(
                    executor.submit(_sleep_for_test, 0.01) for _ in range(9)
                )
                groups = _capture_evaluation_process_groups(executor)
                self.assertEqual(len(groups), 9)
                for future in futures:
                    future.result(timeout=5)
            finally:
                _stop_evaluation_executor(executor)

    @unittest.skipUnless(sys.platform == "linux", "pidfd signaling is Linux-only")
    def test_group_cleanup_never_signals_a_reused_numeric_pgid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lease = Path(directory) / "lease"
            lease.write_bytes(b"")
            os.chmod(lease, 0o600)
            metadata = lease.stat()
            unrelated = subprocess.Popen(["sleep", "60"], start_new_session=True)
            former_leader = subprocess.Popen(["sleep", "60"])
            leader_pidfd = os.pidfd_open(former_leader.pid, 0)
            former_leader.kill()
            former_leader.wait()
            group = _EvaluationProcessGroup(
                SimpleNamespace(),
                unrelated.pid,
                leader_pidfd,
                metadata.st_dev,
                metadata.st_ino,
            )
            try:
                _signal_evaluation_group(group, signal.SIGTERM)
                self.assertIsNone(unrelated.poll())
            finally:
                os.close(leader_pidfd)
                unrelated.kill()
                unrelated.wait()

    @unittest.skipUnless(sys.platform == "linux", "pidfd signaling is Linux-only")
    def test_group_cleanup_ignores_foreign_group_holding_same_lease(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lease = Path(directory) / "lease"
            lease_descriptor = os.open(lease, os.O_RDWR | os.O_CREAT, 0o600)
            metadata = lease.stat()
            former_leader = subprocess.Popen(["sleep", "60"], start_new_session=True)
            leader_pidfd = os.pidfd_open(former_leader.pid, 0)
            former_pgid = former_leader.pid
            former_leader.kill()
            former_leader.wait()
            unrelated = subprocess.Popen(
                ["sleep", "60"],
                start_new_session=True,
                pass_fds=(lease_descriptor,),
            )
            group = _EvaluationProcessGroup(
                SimpleNamespace(),
                former_pgid,
                leader_pidfd,
                metadata.st_dev,
                metadata.st_ino,
            )
            try:
                _signal_evaluation_group(group, signal.SIGTERM)
                self.assertIsNone(unrelated.poll())
            finally:
                os.close(leader_pidfd)
                os.close(lease_descriptor)
                unrelated.kill()
                unrelated.wait()

    @unittest.skipUnless(sys.platform == "linux", "procfs identity is Linux-only")
    def test_lease_scan_rejects_pid_reuse_after_pidfd_open(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            entry = Path(directory) / "123"
            entry.mkdir()
            pidfd = os.open("/dev/null", os.O_RDONLY)
            group = _EvaluationProcessGroup(
                SimpleNamespace(),
                77,
                os.dup(pidfd),
                1,
                2,
            )
            original_iterdir = Path.iterdir

            def iterdir(path: Path):
                if path == Path("/proc"):
                    return iter((entry,))
                return original_iterdir(path)

            try:
                with (
                    patch.object(Path, "iterdir", iterdir),
                    patch(
                        "irisu_rl.r3b_supervisor.os.getpgid",
                        return_value=77,
                    ),
                    patch(
                        "irisu_rl.r3b_supervisor.os.pidfd_open",
                        return_value=pidfd,
                    ),
                    patch(
                        "irisu_rl.r3b_supervisor._evaluation_process_identity",
                        side_effect=((77, 100), (77, 101)),
                    ),
                ):
                    self.assertEqual(_evaluation_lease_holders(group), ())
                with self.assertRaises(OSError):
                    os.fstat(pidfd)
                pidfd = -1
            finally:
                os.close(group.pidfd)
                if pidfd >= 0:
                    os.close(pidfd)

    @unittest.skipUnless(sys.platform == "linux", "procfs identity is Linux-only")
    def test_lease_scan_closes_accumulated_pidfds_on_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lease = root / "lease"
            lease.write_bytes(b"")
            metadata = lease.stat()
            first = root / "111"
            second = root / "222"
            (first / "fd").mkdir(parents=True)
            second.mkdir()
            (first / "fd" / "5").symlink_to(lease)
            scanned_pidfd = os.open("/dev/null", os.O_RDONLY)
            group = _EvaluationProcessGroup(
                SimpleNamespace(),
                77,
                os.dup(scanned_pidfd),
                metadata.st_dev,
                metadata.st_ino,
            )
            original_iterdir = Path.iterdir

            def iterdir(path: Path):
                if path == Path("/proc"):
                    return iter((first, second))
                return original_iterdir(path)

            def identity(entry: Path) -> tuple[int, int]:
                if entry == second:
                    raise PermissionError("blocked")
                return 77, 100

            try:
                with (
                    patch.object(Path, "iterdir", iterdir),
                    patch(
                        "irisu_rl.r3b_supervisor.os.getpgid",
                        return_value=77,
                    ),
                    patch(
                        "irisu_rl.r3b_supervisor.os.pidfd_open",
                        return_value=scanned_pidfd,
                    ),
                    patch(
                        "irisu_rl.r3b_supervisor._evaluation_process_identity",
                        side_effect=identity,
                    ),
                    self.assertRaisesRegex(RuntimeError, "descendant state"),
                ):
                    _evaluation_lease_holders(group)
                with self.assertRaises(OSError):
                    os.fstat(scanned_pidfd)
                scanned_pidfd = -1
            finally:
                os.close(group.pidfd)
                if scanned_pidfd >= 0:
                    os.close(scanned_pidfd)

    @unittest.skipUnless(sys.platform == "linux", "evaluator isolation is Linux-only")
    def test_failed_evaluation_stops_blocked_workers_within_bound(self) -> None:
        executor = ProcessPoolExecutor(
            max_workers=2,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_evaluation_worker_initializer,
            initargs=(os.getpid(), str(evaluator_lease_path())),
        )
        setattr(
            executor,
            "_irisu_evaluator_lease_path",
            str(evaluator_lease_path()),
        )
        blocked = executor.submit(_sleep_for_test, 60.0)
        failed = executor.submit(_fail_for_test)
        _capture_evaluation_process_groups(executor)
        with self.assertRaisesRegex(RuntimeError, "expected evaluator failure"):
            failed.result(timeout=10)
        started = time.monotonic()
        _stop_evaluation_executor(executor, timeout_seconds=2.0)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertTrue(blocked.done())

    @unittest.skipUnless(sys.platform == "linux", "evaluator isolation is Linux-only")
    def test_cleanup_reaps_orphans_after_evaluator_sigkill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            path = Path(directory) / "pids"
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_evaluation_worker_initializer,
                initargs=(os.getpid(), str(evaluator_lease_path(directory))),
            )
            setattr(
                executor,
                "_irisu_evaluator_lease_path",
                str(evaluator_lease_path(directory)),
            )
            pids: tuple[int, ...] = ()
            cleanup_attempted = False
            try:
                executor.submit(_spawn_leased_descendant, str(path))
                groups = _capture_evaluation_process_groups(executor)
                deadline = time.monotonic() + 10
                while not path.is_file() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(path.is_file())
                evaluator_pid, descendant_pid = (
                    int(value) for value in path.read_text().split()
                )
                pids = (evaluator_pid, descendant_pid)
                self.assertEqual(groups[0].pgid, evaluator_pid)
                os.kill(evaluator_pid, signal.SIGKILL)
                groups[0].process.join(2)
                with self.assertRaisesRegex(RuntimeError, "another process"):
                    with R3BRunLock(root, global_directory=directory):
                        pass
                cleanup_attempted = True
                _stop_evaluation_executor(executor, timeout_seconds=2.0)
                deadline = time.monotonic() + 2
                proc = Path(f"/proc/{descendant_pid}/stat")
                while (
                    proc.exists()
                    and proc.read_text().split()[2] != "Z"
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.01)
                if proc.exists():
                    self.assertEqual(proc.read_text().split()[2], "Z")
                with R3BRunLock(root, global_directory=directory):
                    pass
            finally:
                if not cleanup_attempted:
                    _stop_evaluation_executor(executor, timeout_seconds=2.0)
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    @unittest.skipUnless(
        sys.platform == "linux" and EXACT_WORKER is not None,
        "requires the Linux exact worker",
    )
    def test_cleanup_reaps_stopped_exact_worker_after_evaluator_sigkill(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            root.mkdir()
            path = Path(directory) / "pids"
            lease_path = str(evaluator_lease_path(directory))
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_evaluation_worker_initializer,
                initargs=(os.getpid(), lease_path),
            )
            setattr(executor, "_irisu_evaluator_lease_path", lease_path)
            pids: tuple[int, ...] = ()
            cleanup_attempted = False
            try:
                executor.submit(
                    _spawn_stopped_exact_descendant,
                    str(path),
                    str(EXACT_WORKER),
                )
                groups = _capture_evaluation_process_groups(executor)
                deadline = time.monotonic() + 15
                while not path.is_file() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(path.is_file())
                evaluator_pid, worker_pid = (
                    int(value) for value in path.read_text().split()
                )
                pids = (evaluator_pid, worker_pid)
                os.kill(evaluator_pid, signal.SIGKILL)
                groups[0].process.join(2)
                with self.assertRaisesRegex(RuntimeError, "another process"):
                    with R3BRunLock(root, global_directory=directory):
                        pass
                cleanup_attempted = True
                _stop_evaluation_executor(executor, timeout_seconds=2.0)
                deadline = time.monotonic() + 2
                state = "live"
                while time.monotonic() < deadline:
                    try:
                        state = Path(f"/proc/{worker_pid}/stat").read_text().split()[2]
                    except FileNotFoundError:
                        state = "gone"
                    if state in {"gone", "Z"}:
                        break
                    time.sleep(0.01)
                self.assertIn(state, {"gone", "Z"})
                with R3BRunLock(root, global_directory=directory):
                    pass
            finally:
                if not cleanup_attempted:
                    _stop_evaluation_executor(executor, timeout_seconds=2.0)
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_rejects_phase_without_opening_a_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "phase"):
            evaluate_trained_canonical_job(
                "/missing",
                exact_worker_path="/missing",
                portable_library_path="/missing",
                phase="unknown",
            )
        with self.assertRaisesRegex(ValueError, "authorization"):
            evaluate_trained_canonical_job(
                "/missing",
                exact_worker_path="/missing",
                portable_library_path="/missing",
                phase="validation",
            )
        with self.assertRaisesRegex(ValueError, "sealed lease"):
            evaluate_trained_canonical_job(
                "/missing",
                exact_worker_path="/missing",
                portable_library_path="/missing",
                phase="test",
            )

    def test_checkpoint_package_binds_typed_receipt_and_manifest_bytes(self) -> None:
        job = TrialJob(
            _hash("1"),
            "calibration",
            CandidateArm(0, 0.0001),
            7,
            300,
            False,
            _hash("2"),
        )
        manifest_bytes = b'{"checkpoint":"fixture","files":{}}\n'
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        checkpoint = TrainingCheckpointArtifact(
            7,
            50,
            100,
            100,
            100,
            0,
            0,
            job.plan_sha256,
            job.sha256,
            _hash("3"),
            _hash("4"),
            manifest_sha,
            _hash("5"),
            _hash("6"),
        )
        built = SimpleNamespace(
            manifest=SimpleNamespace(sha256=_hash("3"), runner_spec_sha256=_hash("4"))
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = "update-0050"
            checkpoint_root = root / "jobs" / job.sha256 / "checkpoints" / generation
            checkpoint_root.mkdir(parents=True)
            (checkpoint_root / "manifest.json").write_bytes(manifest_bytes)
            store = ArtifactStore(root / "artifacts")
            envelope = store.publish(
                kind="irisu.r3b.training-checkpoint",
                version="r3b-training-checkpoint-package-v3",
                payload={
                    "job_sha256": job.sha256,
                    "trial_manifest_sha256": _hash("3"),
                    "runner_spec_sha256": _hash("4"),
                    "completed_updates": 50,
                    "optimizer_simulated_ticks": 100,
                    "total_simulated_ticks": 100,
                    "skipped_simulated_ticks": 0,
                    "drain_simulated_ticks": 0,
                    "model_sha256": _hash("5"),
                    "deployment_policy_sha256": _hash("6"),
                    "checkpoint_artifact": checkpoint.manifest(),
                    "generation": generation,
                    "checkpoint_manifest_sha256": manifest_sha,
                    "checkpoint_files": {},
                },
            )
            loaded, loaded_generation, _ = _checkpoint_package(
                root=root,
                store=store,
                artifact_sha256=envelope.artifact_id,
                built=built,
                job=job,
                target_update=50,
            )
            self.assertEqual(loaded, checkpoint)
            self.assertEqual(loaded_generation, generation)

            (checkpoint_root / "manifest.json").write_bytes(b"tampered\n")
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                _checkpoint_package(
                    root=root,
                    store=store,
                    artifact_sha256=envelope.artifact_id,
                    built=built,
                    job=job,
                    target_update=50,
                )

    @unittest.skipUnless(PORTABLE.exists(), "portable integration library not built")
    def test_multiple_checkpoints_restore_into_distinct_fresh_sessions(self) -> None:
        job = TrialJob(
            _hash("1"),
            "calibration",
            CandidateArm(0, 0.0001),
            7,
            2,
            False,
            _hash("2"),
        )
        identity = {"test": "supervisor-multi-checkpoint"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_root = root / "jobs" / job.sha256 / "checkpoints"
            source, source_vector = build_session(
                exact=False,
                construction_seed=101,
            )
            packages: dict[str, tuple[TrainingCheckpointArtifact, str, object]] = {}
            try:
                source.initialize()
                for update in range(2):
                    generation = f"update-{update:04d}"
                    saved = source.save(
                        checkpoint_root,
                        generation,
                        identity=identity,
                    )
                    manifest_sha256 = hashlib.sha256(
                        (saved / "manifest.json").read_bytes()
                    ).hexdigest()
                    deployment = _deployment(source.model)[3]
                    checkpoint = TrainingCheckpointArtifact(
                        job.learner_seed,
                        update,
                        source.optimizer_simulated_ticks,
                        source.optimizer_simulated_ticks,
                        source.collector.simulated_ticks,
                        source.skipped_simulated_ticks,
                        source.drain_simulated_ticks,
                        job.plan_sha256,
                        job.sha256,
                        _hash("3"),
                        _hash("4"),
                        manifest_sha256,
                        source.policy_sha256,
                        deployment.sha256,
                    )
                    packages[str(update)] = (checkpoint, generation, {})
                    source.run_update()
            finally:
                source_vector.close()

            vectors = []

            def build(_job: TrialJob, *, authorization: object) -> object:
                self.assertEqual((_job, authorization), (job, None))
                session, vector = build_session(
                    exact=False,
                    construction_seed=999,
                )
                vectors.append(vector)
                return SimpleNamespace(session=session, close=vector.close)

            builder = SimpleNamespace(build=build)
            store = ArtifactStore(root / "artifacts")
            restored = []
            try:
                with patch(
                    "irisu_rl.r3b_supervisor._checkpoint_package",
                    side_effect=lambda **kwargs: packages[kwargs["artifact_sha256"]],
                ):
                    for update in range(2):
                        built, checkpoint, generation = _fresh_restored_checkpoint(
                            builder=builder,
                            authorization=None,
                            root=root,
                            store=store,
                            artifact_sha256=str(update),
                            job=job,
                            target_update=update,
                            identity=identity,
                        )
                        restored.append(built)
                        self.assertEqual(
                            built.session.trainer.schedule.completed_updates,
                            checkpoint.completed_updates,
                        )
                        self.assertEqual(generation, f"update-{update:04d}")
                self.assertIsNot(restored[0].session, restored[1].session)
                with self.assertRaisesRegex(RuntimeError, "fresh training session"):
                    restored[0].session.restore(
                        checkpoint_root,
                        generation="update-0001",
                        identity=identity,
                    )
            finally:
                for built in restored:
                    built.close()


if __name__ == "__main__":
    unittest.main()
