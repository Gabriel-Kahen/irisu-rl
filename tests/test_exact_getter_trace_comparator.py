from __future__ import annotations

import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_exact_getter_trace",
    ROOT / "tools/compare-exact-getter-trace.py",
)
assert SPEC and SPEC.loader
comparator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = comparator
SPEC.loader.exec_module(comparator)


def f32(value: float) -> str:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    return f"{bits:08x}"


class ExactGetterTraceComparatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.directory = tempfile.TemporaryDirectory()
        root = Path(cls.directory.name)
        cls.library = root / "libfixture.so"
        cls.helper = root / "getter-trace-replay"
        try:
            subprocess.run(
                [
                    "cc",
                    "-m32",
                    "-shared",
                    "-fPIC",
                    str(ROOT / "tests/native/exact_getter_fixture.c"),
                    "-o",
                    str(cls.library),
                ],
                check=True,
                capture_output=True,
            )
            comparator.build_helper(comparator.HELPER_SOURCE, cls.helper, "cc")
        except (OSError, subprocess.CalledProcessError) as error:
            cls.directory.cleanup()
            raise unittest.SkipTest(f"32-bit C compiler unavailable: {error}")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.directory.cleanup()

    def records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = [
            {
                "type": "proxy_loaded",
                "schema": 1,
                "ok": True,
                "x87_cw": "027f",
            },
            {
                "type": "init",
                "world": 1,
                "args_f32": [f32(value) for value in (0, 0, 100, 100, 0, 100)],
                "result": 1,
                "x87_cw_before": "027f",
                "x87_cw_after": "027f",
            },
            {
                "type": "create",
                "world": 1,
                "step": 0,
                "shape": "box",
                "ordinal": 1,
                "args_f32": [
                    f32(value) for value in (2, 2, 10, 20, 0.5, 1, 0.8, 0)
                ],
            },
            {
                "type": "set_user_data",
                "world": 1,
                "step": 0,
                "ordinal": 1,
                "user": 1234,
            },
            {
                "type": "get_scalar",
                "world": 1,
                "step": 0,
                "field": "x",
                "ordinal": 1,
                "value_f32": f32(10),
            },
            {
                "type": "set_v",
                "world": 1,
                "step": 0,
                "ordinal": 1,
                "args_f32": [f32(100), f32(-200)],
            },
            {
                "type": "get_v",
                "world": 1,
                "step": 0,
                "ordinal": 1,
                "args_f32": [f32(1), f32(-2)],
            },
            {
                "type": "step",
                "world": 1,
                "step": 1,
                "dt_f32": f32(0.5),
                "iterations": 10,
            },
            {
                "type": "get_scalar",
                "world": 1,
                "step": 1,
                "field": "x",
                "ordinal": 1,
                "value_f32": f32(60),
            },
            {
                "type": "get_scalar",
                "world": 1,
                "step": 1,
                "field": "y",
                "ordinal": 1,
                "value_f32": f32(-80),
            },
            {
                "type": "contact",
                "world": 1,
                "step": 1,
                "call": 1,
                "result": False,
                "a_user": 0,
                "b_user": 0,
            },
            {"type": "destroy", "world": 1, "step": 1, "ordinal": 1},
            {"type": "dispose", "world": 1, "step": 1},
        ]
        for sequence, record in enumerate(records):
            record["seq"] = sequence
        return records

    def compare(self, records: list[dict[str, object]]) -> dict[str, object]:
        path = Path(self.directory.name) / "trace.jsonl"
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="ascii",
        )
        return comparator.compare(
            path, self.library, helper=self.helper, progress_every=0
        )

    def test_replays_getters_and_contacts_in_global_call_order(self) -> None:
        report = self.compare(self.records())
        self.assertEqual(report["status"], "exact_all_getters")
        comparison = report["comparison"]
        self.assertEqual(
            {
                key: comparison[key]
                for key in (
                    "all_getter_records_exact",
                    "all_contact_records_exact",
                    "getter_records",
                    "getter_values",
                    "contacts",
                    "first_mismatch",
                )
            },
            {
                "all_getter_records_exact": True,
                "all_contact_records_exact": True,
                "getter_records": 4,
                "getter_values": 5,
                "contacts": 1,
                "first_mismatch": None,
            },
        )

    def test_reports_first_raw_float_mismatch(self) -> None:
        records = self.records()
        records[8]["value_f32"] = f32(61)
        report = self.compare(records)
        self.assertEqual(report["status"], "mismatch")
        self.assertEqual(
            report["comparison"]["first_mismatch"],
            {
                "seq": 8,
                "operation": "get_x",
                "ordinal": 1,
                "component": "value",
                "expected_f32": f32(61),
                "actual_f32": f32(60),
            },
        )


if __name__ == "__main__":
    unittest.main()
