from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_exact_wrapper_trace",
    ROOT / "tools" / "compare-exact-wrapper-trace.py",
)
assert SPEC and SPEC.loader
comparator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparator
SPEC.loader.exec_module(comparator)


class ExactWrapperTraceComparatorTests(unittest.TestCase):
    def test_streams_match_with_bootstrap_and_original_teardown_removed(self) -> None:
        records = [
            {"seq": 0, "type": "proxy_loaded"},
            {"seq": 1, "type": "init", "args_f32": ["00"] * 6},
            {"seq": 2, "type": "create", "shape": "box", "ordinal": 1, "args_f32": ["01"] * 8},
            {"seq": 3, "type": "set_v", "ordinal": 1, "args_f32": ["00", "00"]},
            {"seq": 4, "type": "set_user_data", "ordinal": 1},
            {"seq": 5, "type": "get_scalar", "step": 0},
            {"seq": 6, "type": "step", "step": 1, "dt_f32": "3c", "iterations": 10},
            {"seq": 7, "type": "contact", "step": 1, "call": 1, "result": True, "a_ordinal": 1, "b_ordinal": 2},
            {"seq": 8, "type": "contact", "step": 1, "call": 2, "result": False},
            {"seq": 9, "type": "set_position", "step": 1, "ordinal": 1, "args_f32": ["02", "03", "04"]},
            {"seq": 10, "type": "destroy", "step": 1, "ordinal": 1},
            {"seq": 11, "type": "set_position", "step": 1, "ordinal": 2, "args_f32": ["05", "06", "07"]},
            {"seq": 12, "type": "destroy", "step": 1, "ordinal": 2},
            {"seq": 13, "type": "dispose", "step": 1},
        ]
        native = "\n".join(
            (
                "I boot0",
                "I boot1",
                "I 00 00 00 00 00 00",
                "B 1 01 01 01 01 01 01 01 01",
                "V 1 00 00",
                "U 1",
                "S 1 3c 10",
                "K 1 1 2",
                "K 2 0 0",
                "P 1 02 03 04",
                "D 1",
            )
        ) + "\n"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original_path = root / "original.jsonl"
            original_path.write_text(
                "".join(json.dumps(value) + "\n" for value in records),
                encoding="utf-8",
            )
            native_path = root / "native.trace"
            native_path.write_text(native, encoding="ascii")
            report = comparator.compare(original_path, native_path)

        self.assertEqual(report["status"], "exact_through_final_physics_step")
        comparison = report["comparison"]
        self.assertTrue(comparison["exact_through_final_physics_step"])
        self.assertEqual(comparison["streams"]["contact"]["matched_prefix"], 2)
        self.assertEqual(comparison["streams"]["set_position"]["original_only_suffix"], 1)
        self.assertEqual(
            comparison["original_only_post_step_epilogue"]["counts"],
            {"destroy": 1, "dispose": 1, "set_position": 1},
        )
        self.assertEqual(comparison["ignored_original_records"]["get_scalar"], 1)

    def test_reports_first_stream_mismatch(self) -> None:
        records = [
            {"seq": 0, "type": "init", "args_f32": ["00"] * 6},
            {"seq": 1, "type": "step", "step": 1, "dt_f32": "3c", "iterations": 10},
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original.jsonl"
            original.write_text(
                "".join(json.dumps(value) + "\n" for value in records),
                encoding="utf-8",
            )
            native = root / "native.trace"
            native.write_text(
                "I boot0\nI boot1\nI 00 00 00 00 00 00\nS 1 bad 10\n",
                encoding="ascii",
            )
            report = comparator.compare(original, native)

        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(report["comparison"]["first_mismatch"]["stream"], "step")


if __name__ == "__main__":
    unittest.main()
