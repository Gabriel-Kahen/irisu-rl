from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import padded as padded_module  # noqa: E402


class ExactDefaultConcurrencyTests(unittest.TestCase):
    def test_process_affinity_drives_adaptive_default(self) -> None:
        with mock.patch.object(
            padded_module.os, "sched_getaffinity", return_value={2, 4, 6}
        ), mock.patch.object(padded_module.os, "cpu_count", return_value=99):
            self.assertEqual(padded_module._available_logical_cpus(), 3)
            self.assertEqual(padded_module._default_exact_workers(100), 12)
            self.assertEqual(padded_module._default_exact_workers(7), 7)

    def test_cpu_count_is_used_when_affinity_is_unavailable(self) -> None:
        with mock.patch.object(
            padded_module.os,
            "sched_getaffinity",
            side_effect=OSError("affinity unavailable"),
        ), mock.patch.object(padded_module.os, "cpu_count", return_value=6):
            self.assertEqual(padded_module._available_logical_cpus(), 6)
            self.assertEqual(padded_module._default_exact_workers(100), 24)

    def test_missing_cpu_topology_falls_back_to_one_visible_cpu(self) -> None:
        with mock.patch.object(
            padded_module.os, "sched_getaffinity", side_effect=AttributeError
        ), mock.patch.object(padded_module.os, "cpu_count", return_value=None):
            self.assertEqual(padded_module._available_logical_cpus(), 1)
            self.assertEqual(padded_module._default_exact_workers(100), 4)


if __name__ == "__main__":
    unittest.main()
