#!/usr/bin/env python3
"""Compare batched v86 exact replays with the native runner and v2.03 oracles."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_CORPUS = ROOT / "reference" / "replays" / "raw" / "internet"
DEFAULT_RUNNER = ROOT / "build-physics-integration-exact-multiworld-2" / "irisu-exact-replay"
DEFAULT_ORACLES = ROOT / "reference" / "runs"
NATIVE_FIELDS = (
    "tick",
    "score",
    "gauge",
    "level",
    "highest_chain",
    "clears",
    "score_calls",
    "confirmed",
    "terminal_frame",
)
TIMELINES = ("score_timeline", "gauge_timeline")


def load_evaluator() -> Any:
    path = ROOT / "tools" / "evaluate-exact-replay-corpus.py"
    spec = importlib.util.spec_from_file_location("irisu_v86_corpus_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REFERENCE = load_evaluator()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compact_json_hash(value: Any) -> str:
    return sha256(json.dumps(value, separators=(",", ":")).encode())


def first_sequence_mismatch(expected: list[Any], actual: list[Any]) -> dict[str, Any] | None:
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return {"index": index, "expected": left, "actual": right}
    if len(expected) != len(actual):
        index = min(len(expected), len(actual))
        return {
            "index": index,
            "expected": expected[index] if index < len(expected) else None,
            "actual": actual[index] if index < len(actual) else None,
        }
    return None


def compare_results(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, Any]:
    fields = {
        name: {
            "expected": expected.get(name),
            "actual": actual.get(name),
            "matches": expected.get(name) == actual.get(name),
        }
        for name in NATIVE_FIELDS
    }
    timelines = {}
    first_mismatch = next(
        (
            {"kind": "field", "field": name, **comparison}
            for name, comparison in fields.items()
            if not comparison["matches"]
        ),
        None,
    )
    for name in TIMELINES:
        left = expected.get(name, [])
        right = actual.get(name, [])
        mismatch = first_sequence_mismatch(left, right)
        timelines[name] = {
            "expected_count": len(left),
            "actual_count": len(right),
            "expected_sha256": compact_json_hash(left),
            "actual_sha256": compact_json_hash(right),
            "matches": mismatch is None,
            "first_mismatch": mismatch,
        }
        if first_mismatch is None and mismatch is not None:
            first_mismatch = {"kind": "timeline", "timeline": name, **mismatch}
    return {
        "exact": first_mismatch is None,
        "fields": fields,
        "timelines": timelines,
        "first_mismatch": first_mismatch,
    }


def run_native(runner: Path, replay: Path, timeout: float) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["IRISU_EXACT_CW"] = "0x27f"
    started = time.perf_counter()
    completed = subprocess.run(
        [str(runner), str(replay)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(
            f"native runner failed for {replay.name}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    encoded = completed.stdout.encode()
    return {
        "elapsed_seconds": elapsed,
        "stdout_bytes": len(encoded),
        "stdout_sha256": sha256(encoded),
        "result": json.loads(completed.stdout),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--oracle-root", type=Path, default=DEFAULT_ORACLES)
    parser.add_argument("--node", default="node")
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = sorted(args.corpus.glob("*.rpy"))
    inventory = REFERENCE.inventory(paths)
    eligible = [entry for entry in inventory if entry["eligible"]]
    if not eligible:
        parser.error(f"no eligible padded normal-mode replay in {args.corpus}")
    runner = args.runner.resolve()
    if not runner.is_file() or not os.access(runner, os.X_OK):
        parser.error(f"native runner is not executable: {runner}")

    native = {}
    for entry in eligible:
        replay = Path(entry["path"])
        native[entry["sha256"]] = run_native(runner, replay, args.timeout)

    environment = os.environ.copy()
    environment["IRISU_V86_REPLAY_TIMEOUT"] = str(args.timeout)
    completed = subprocess.run(
        [args.node, str(HERE / "batch-replay.mjs"), *(entry["path"] for entry in eligible)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=args.timeout + 90.0,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        parser.error(f"v86 batch failed: {detail}")
    v86 = json.loads(completed.stdout)
    v86_by_sha = {entry["sha256"]: entry for entry in v86["entries"]}

    oracle_paths = REFERENCE.discover_oracle_metadata(args.oracle_root)
    oracles = REFERENCE.load_oracles(oracle_paths)
    reports = []
    for entry in eligible:
        replay_hash = entry["sha256"]
        native_entry = native[replay_hash]
        v86_entry = v86_by_sha.get(replay_hash)
        if v86_entry is None:
            raise RuntimeError(f"v86 omitted replay {replay_hash}")
        if v86_entry["exit_code"] != 0 or not isinstance(v86_entry["result"], dict):
            comparison = {"exact": False, "first_mismatch": {"kind": "runner_error"}}
            v86_result = None
        else:
            v86_result = v86_entry["result"]
            comparison = compare_results(native_entry["result"], v86_result)
        oracle = oracles.get(replay_hash)
        native_oracle = None if oracle is None else REFERENCE._compare_oracle(native_entry["result"], oracle)
        v86_oracle = (
            None
            if oracle is None or v86_result is None
            else REFERENCE._compare_oracle(v86_result, oracle)
        )
        frames = int(entry["frame_count"])
        v86_seconds = float(v86_entry["guest_elapsed_seconds"])
        native_seconds = float(native_entry["elapsed_seconds"])
        executed_ticks = int(native_entry["result"]["tick"])
        reports.append(
            {
                "name": Path(entry["path"]).name,
                "replay_sha256": replay_hash,
                "frame_count": frames,
                "executed_ticks": executed_ticks,
                "native": {
                    "elapsed_seconds": native_seconds,
                    "ticks_per_second": executed_ticks / native_seconds,
                    "stdout_bytes": native_entry["stdout_bytes"],
                    "stdout_sha256": native_entry["stdout_sha256"],
                },
                "v86": {
                    "exit_code": v86_entry["exit_code"],
                    "elapsed_seconds": v86_seconds,
                    "ticks_per_second": executed_ticks / v86_seconds if v86_seconds > 0 else None,
                    "wall_ms": v86_entry["wall_ms"],
                    "stdout_bytes": v86_entry["stdout_bytes"],
                    "stdout_sha256": v86_entry["stdout_sha256"],
                    "raw_stdout_matches_native": (
                        v86_entry["stdout_sha256"] == native_entry["stdout_sha256"]
                    ),
                    "stderr": v86_entry["stderr"],
                },
                "native_vs_v86": comparison,
                "native_vs_observed_v203": native_oracle,
                "v86_vs_observed_v203": v86_oracle,
            }
        )

    exact = sum(report["native_vs_v86"]["exact"] for report in reports)
    raw_exact = sum(report["v86"]["raw_stdout_matches_native"] for report in reports)
    oracle_exact = sum(
        report["v86_vs_observed_v203"] is not None
        and report["v86_vs_observed_v203"]["full_scoring_parity"]
        for report in reports
    )
    report = {
        "schema": 1,
        "status": (
            "pass" if exact == len(reports) and raw_exact == len(reports) else "mismatch"
        ),
        "scope": {
            "execution": "one v86 boot; each whole replay runs in guest with no per-tick host RPC",
            "transport": "replay inputs and runner outputs use virtio-9p; serial carries completion markers only",
            "control_word": "0x027f",
        },
        "artifacts": {
            "native_runner": {"path": str(runner), "sha256": sha256(runner.read_bytes())},
            "v86": v86["artifacts"],
        },
        "performance": {"boot_ms": v86["boot_ms"], "batch_ms": v86["batch_ms"]},
        "replays": reports,
        "summary": {
            "discovered": len(inventory),
            "eligible": len(eligible),
            "native_vs_v86_exact": exact,
            "native_vs_v86_raw_stdout_exact": raw_exact,
            "observed_v203_oracles": sum(report["v86_vs_observed_v203"] is not None for report in reports),
            "v86_full_scoring_parity": oracle_exact,
            "total_frames": sum(report["frame_count"] for report in reports),
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
