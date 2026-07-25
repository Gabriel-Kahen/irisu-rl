from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from irisu_rl.r3b_lock import (
    R3BRunLock,
    evaluator_lease_path,
    hold_evaluator_lease,
)


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
