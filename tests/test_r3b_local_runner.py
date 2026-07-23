from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from irisu_rl.r3b_local_runner import _load_claim, _write_claim
from irisu_rl.r3b_operational import JobClaim


class R3BLocalRunnerTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
