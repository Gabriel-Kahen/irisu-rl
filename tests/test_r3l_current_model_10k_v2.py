from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/rl_r3l_current_model_10k_v2.py"
SPEC = importlib.util.spec_from_file_location("r3l_exhibition_v2_test_target", SOURCE)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class ExhibitionV2Tests(unittest.TestCase):
    def test_query_schedule_is_fixed_and_extended(self) -> None:
        self.assertEqual(RUNNER.QUERY_THRESHOLDS[:4], (0, 2500, 5000, 7500))
        self.assertEqual(RUNNER.QUERY_THRESHOLDS[-1], 57500)
        self.assertEqual(len(RUNNER.QUERY_THRESHOLDS), 24)

    def test_failed_attempt_is_bound(self) -> None:
        identity = RUNNER._source_identity()
        RUNNER.BASE._verify_self_hash(identity, "source-v2")
        self.assertEqual(
            identity["failed_attempt_sha256"],
            RUNNER.EXPECTED_FAILED_EPISODE_ID,
        )


if __name__ == "__main__":
    unittest.main()
