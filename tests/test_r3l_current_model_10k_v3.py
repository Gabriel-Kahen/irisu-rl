from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/rl_r3l_current_model_10k_v3.py"
SPEC = importlib.util.spec_from_file_location("r3l_exhibition_v3_tested", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExhibitionV3Test(unittest.TestCase):
    def test_phase_is_exact_through_prefix(self) -> None:
        self.assertEqual([MODULE._effective_phase_index(i) for i in range(4)], [0, 1, 2, 3])

    def test_post_prefix_holds_last_phase(self) -> None:
        self.assertEqual([MODULE._effective_phase_index(i) for i in (4, 5, 19, 100)], [3, 3, 3, 3])

    def test_invalid_phase_rejected(self) -> None:
        for value in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                MODULE._effective_phase_index(value)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
