from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from irisu_rl.r3b_lock import R3BRunLock


class R3BRunLockTests(unittest.TestCase):
    def test_rejects_concurrent_operator_and_releases_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with R3BRunLock(root):
                with self.assertRaisesRegex(RuntimeError, "another process"):
                    with R3BRunLock(root):
                        pass
            with R3BRunLock(root):
                self.assertTrue((root / "operator.lock").is_file())


if __name__ == "__main__":
    unittest.main()
