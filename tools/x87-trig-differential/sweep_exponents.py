#!/usr/bin/env python3
import argparse
import json
import pathlib
import subprocess
import sys


OPS = ("fsin", "fcos", "fsincos_sine", "fsincos_cosine")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=pathlib.Path, default=pathlib.Path(__file__).with_name("x87-trig-diff"))
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=145)
    parser.add_argument("--chunk", type=int, default=4)
    parser.add_argument(
        "--candidate",
        choices=("core-math", "host-f64", "v86-libm"),
        default="core-math",
    )
    args = parser.parse_args()
    if not (0 <= args.first <= args.last <= 254 and args.chunk > 0):
        parser.error("require 0 <= first <= last <= 254 and chunk > 0")

    summary = {
        "schema": 1,
        "first_exponent": args.first,
        "last_exponent": args.last,
        "candidate": args.candidate,
        "input_count": 0,
        "metrics": {op: {"compared": 0, "exact": 0, "raw_mismatches": 0} for op in OPS},
        "pair_disagreements": {"sine": 0, "cosine": 0},
        "range_status_mismatches": 0,
        "portable_status_model_mismatches": 0,
        "mismatch_chunks": [],
    }
    first = args.first
    while first <= args.last:
        last = min(args.last, first + args.chunk - 1)
        command = [
            str(args.binary), "--no-curated", "--samples", "4",
            "--exponent-min", str(first), "--exponent-max", str(last),
            "--candidate", args.candidate,
        ]
        report = json.loads(subprocess.check_output(command, text=True))
        summary["input_count"] += report["input_count"]
        chunk_mismatches = 0
        for op in OPS:
            metric = report["metrics"][op]
            raw_mismatches = metric["compared"] - metric["exact"]
            summary["metrics"][op]["compared"] += metric["compared"]
            summary["metrics"][op]["exact"] += metric["exact"]
            summary["metrics"][op]["raw_mismatches"] += raw_mismatches
            chunk_mismatches += raw_mismatches
        pair = report["native_pair_consistency"]
        summary["pair_disagreements"]["sine"] += pair["sine_disagrees_with_fsin"]
        summary["pair_disagreements"]["cosine"] += pair["cosine_disagrees_with_fcos"]
        status_mismatches = sum(report["range_status"].values())
        summary["range_status_mismatches"] += status_mismatches
        model_mismatches = sum(report["portable_status_model"].values())
        summary["portable_status_model_mismatches"] += model_mismatches
        if chunk_mismatches or status_mismatches or model_mismatches:
            summary["mismatch_chunks"].append(
                {"first": first, "last": last, "raw_mismatches": chunk_mismatches,
                 "range_status_mismatches": status_mismatches,
                 "portable_status_model_mismatches": model_mismatches}
            )
        print(f"completed exponents {first}-{last}: mismatches={chunk_mismatches}, status={model_mismatches}", file=sys.stderr, flush=True)
        first = last + 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
