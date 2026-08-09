#!/usr/bin/env python3
import json
import pathlib
import sys


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_report.py REPORT.json")
    report = json.loads(pathlib.Path(sys.argv[1]).read_text())
    assert report["schema"] == 1
    assert report["control_word"] == "0x027f"
    assert report["final_control_word"] == "0x027f"
    assert report["input_count"] >= 10_000
    assert report["range_status"] == {
        "fsin_c2_mismatches": 0,
        "fcos_c2_mismatches": 0,
        "fsincos_c2_mismatches": 0,
        "fsincos_remaining_mismatches": 0,
    }
    assert report["portable_status_model"] == {
        "fsin_mismatches": 0,
        "fcos_mismatches": 0,
        "fsincos_mismatches": 0,
    }
    pair = report["native_pair_consistency"]
    assert pair["sine_disagrees_with_fsin"] == 0
    assert pair["cosine_disagrees_with_fcos"] == 0
    for key in ("fsin", "fcos", "fsincos_sine", "fsincos_cosine"):
        metric = report["metrics"][key]
        assert metric["compared"] > 0
        assert 0 <= metric["exact"] <= metric["compared"]
    print("x87 trig differential report invariants passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
