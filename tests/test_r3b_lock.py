from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from irisu_env import ExactWorkerClient, ExactWorkerNotFoundError, find_exact_worker
import irisu_env.exact_ipc as exact_ipc_module
from irisu_rl.r3b_lock import (
    R3BRunLock,
    evaluator_lease_path,
    hold_evaluator_lease,
)

try:
    EXACT_WORKER = find_exact_worker()
except ExactWorkerNotFoundError:
    EXACT_WORKER = None


def _wait_until_lock_available(root: Path, locks: Path) -> None:
    deadline = time.monotonic() + 5
    while True:
        try:
            with R3BRunLock(root, global_directory=locks):
                return
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


class R3BRunLockTests(unittest.TestCase):
    def test_rejects_concurrent_operator_and_releases_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / "locks"
            locks.mkdir()
            with R3BRunLock(root, global_directory=locks):
                with self.assertRaisesRegex(RuntimeError, "another process"):
                    with R3BRunLock(root, global_directory=locks):
                        pass
            with R3BRunLock(root, global_directory=locks):
                self.assertTrue((root / "operator.lock").is_file())

    def test_rejects_concurrent_operators_for_different_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first"
            second = root / "second"
            locks = root / "locks"
            first.mkdir()
            second.mkdir()
            locks.mkdir()
            with R3BRunLock(first, global_directory=locks):
                with self.assertRaisesRegex(RuntimeError, "another process"):
                    with R3BRunLock(second, global_directory=locks):
                        pass

    def test_reentry_does_not_release_the_held_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / "locks"
            locks.mkdir()
            held = R3BRunLock(root, global_directory=locks)
            with held:
                with self.assertRaisesRegex(RuntimeError, "already held"):
                    held.__enter__()
                with self.assertRaisesRegex(RuntimeError, "another process"):
                    with R3BRunLock(root, global_directory=locks):
                        pass

    def test_rejects_operator_until_evaluator_lease_drains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / "locks"
            locks.mkdir()
            descriptor = hold_evaluator_lease(evaluator_lease_path(locks))
            try:
                with self.assertRaisesRegex(RuntimeError, "another process"):
                    with R3BRunLock(root, global_directory=locks):
                        pass
            finally:
                os.close(descriptor)
            with R3BRunLock(root, global_directory=locks):
                pass

    def test_registers_worker_lease_and_clears_it_on_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / "locks"
            locks.mkdir()
            self.assertEqual(exact_ipc_module._exact_worker_pass_fds(), ())
            with R3BRunLock(root, global_directory=locks) as held:
                descriptor = held._lease_descriptor
                self.assertIsNotNone(descriptor)
                self.assertEqual(
                    exact_ipc_module._exact_worker_pass_fds(), (descriptor,)
                )
            self.assertEqual(exact_ipc_module._exact_worker_pass_fds(), ())
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    @unittest.skipUnless(
        EXACT_WORKER is not None and hasattr(signal, "SIGSTOP"),
        "requires a POSIX exact worker",
    )
    def test_live_exact_worker_keeps_lease_after_clean_operator_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / "locks"
            locks.mkdir()
            client: ExactWorkerClient | None = None
            worker_pid: int | None = None
            try:
                with R3BRunLock(root, global_directory=locks):
                    client = ExactWorkerClient(EXACT_WORKER)
                    worker_pid = client._transport_pid
                    os.kill(worker_pid, signal.SIGSTOP)
                with self.assertRaisesRegex(RuntimeError, "another process"):
                    with R3BRunLock(root, global_directory=locks):
                        pass
                os.kill(worker_pid, signal.SIGKILL)
                _wait_until_lock_available(root, locks)
                worker_pid = None
            finally:
                if worker_pid is not None:
                    try:
                        os.kill(worker_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if client is not None:
                    client.close()

    @unittest.skipUnless(
        EXACT_WORKER is not None and hasattr(signal, "SIGSTOP"),
        "requires a POSIX exact worker",
    )
    def test_orphaned_exact_worker_keeps_lease_after_operator_sigkill(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / "locks"
            locks.mkdir()
            ready = root / "ready"
            script = """
import signal
from pathlib import Path
from irisu_env import ExactWorkerClient
from irisu_rl.r3b_lock import R3BRunLock

root, locks, worker, ready = map(Path, __import__("sys").argv[1:])
with R3BRunLock(root, global_directory=locks):
    client = ExactWorkerClient(worker)
    Path(ready).write_text(str(client._transport_pid), encoding="utf-8")
    signal.pause()
"""
            operator = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    script,
                    str(root),
                    str(locks),
                    str(EXACT_WORKER),
                    str(ready),
                ]
            )
            worker_pid: int | None = None
            try:
                deadline = time.monotonic() + 20
                while not ready.is_file() and time.monotonic() < deadline:
                    if operator.poll() is not None:
                        self.fail(f"operator exited with {operator.returncode}")
                    time.sleep(0.01)
                self.assertTrue(ready.is_file())
                worker_pid = int(ready.read_text(encoding="utf-8"))
                os.kill(worker_pid, signal.SIGSTOP)
                os.kill(operator.pid, signal.SIGKILL)
                operator.wait(timeout=5)
                with self.assertRaisesRegex(RuntimeError, "another process"):
                    with R3BRunLock(root, global_directory=locks):
                        pass
                os.kill(worker_pid, signal.SIGKILL)
                _wait_until_lock_available(root, locks)
                worker_pid = None
            finally:
                if operator.poll() is None:
                    operator.kill()
                    operator.wait()
                if worker_pid is not None:
                    try:
                        os.kill(worker_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_rejects_unsafe_host_lock_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = root / "locks"
            locks.mkdir()
            global_path = locks / f".irisu-r3b-operator-{os.geteuid()}.lock"
            global_path.write_text("foreign", encoding="utf-8")
            global_path.chmod(0o644)

            with self.assertRaisesRegex(ValueError, "metadata is unsafe"):
                with R3BRunLock(root, global_directory=locks):
                    pass


if __name__ == "__main__":
    unittest.main()
