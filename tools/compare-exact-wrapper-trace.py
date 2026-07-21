#!/usr/bin/env python3
"""Stream-compare original proxy calls with an exact-forward wrapper trace."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


STREAMS = (
    "init",
    "create",
    "set_v",
    "set_user_data",
    "set_position",
    "destroy",
    "step",
    "contact",
)
EPILOGUE_TYPES = {"set_position", "destroy", "dispose"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _native_stream(line: str) -> str:
    kind = line.split(" ", 1)[0]
    return {
        "I": "init",
        "B": "create",
        "T": "create",
        "C": "create",
        "V": "set_v",
        "U": "set_user_data",
        "P": "set_position",
        "D": "destroy",
        "S": "step",
        "K": "contact",
    }.get(kind, "")


def _load_native(
    path: Path, skip_init_generations: int
) -> tuple[dict[str, list[str]], int, int]:
    streams: dict[str, list[str]] = {name: [] for name in STREAMS}
    init_count = 0
    skipped = 0
    included = 0
    with path.open(encoding="ascii") as source:
        for line_number, raw in enumerate(source, 1):
            line = raw.rstrip("\n")
            if line.startswith("I "):
                init_count += 1
            if init_count <= skip_init_generations:
                skipped += 1
                continue
            stream = _native_stream(line)
            if not stream:
                raise ValueError(f"unknown native record at {path}:{line_number}")
            streams[stream].append(line)
            included += 1
    if init_count <= skip_init_generations:
        raise ValueError("native trace does not contain the requested init generation")
    return streams, skipped, included


def _original_record(value: dict[str, Any]) -> tuple[str, str] | None:
    kind = value.get("type")
    if kind == "init":
        return kind, "I " + " ".join(value["args_f32"])
    if kind == "create":
        prefix = {"box": "B", "triangle": "T", "circle": "C"}[value["shape"]]
        return kind, f"{prefix} {value['ordinal']} " + " ".join(value["args_f32"])
    if kind == "set_v":
        return kind, f"V {value['ordinal']} " + " ".join(value["args_f32"])
    if kind == "set_user_data":
        return kind, f"U {value['ordinal']}"
    if kind == "set_position":
        return kind, f"P {value['ordinal']} " + " ".join(value["args_f32"])
    if kind == "destroy":
        return kind, f"D {value['ordinal']}"
    if kind == "step":
        return kind, f"S {value['step']} {value['dt_f32']} {value['iterations']}"
    if kind == "contact":
        a = value.get("a_ordinal", 0) if value["result"] else 0
        b = value.get("b_ordinal", 0) if value["result"] else 0
        return kind, f"K {value['call']} {a} {b}"
    return None


def compare(
    original: Path, native: Path, *, skip_native_init_generations: int = 2
) -> dict[str, Any]:
    original = original.resolve()
    native = native.resolve()
    native_streams, bootstrap_lines, native_lines = _load_native(
        native, skip_native_init_generations
    )
    indices = Counter()
    original_counts = Counter()
    ignored_counts = Counter()
    extras = Counter()
    extra_steps: dict[str, set[int]] = defaultdict(set)
    first_mismatch: dict[str, Any] | None = None
    record_count = 0
    final_step = 0
    digest = hashlib.sha256()

    with original.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            digest.update(raw)
            record_count += 1
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {original}:{line_number}: {error}") from error
            kind = str(value.get("type", ""))
            original_counts[kind] += 1
            if kind == "step":
                final_step = max(final_step, int(value["step"]))
            normalized = _original_record(value)
            if normalized is None:
                if kind == "dispose":
                    extras[kind] += 1
                    extra_steps[kind].add(int(value.get("step", -1)))
                else:
                    ignored_counts[kind] += 1
                continue
            stream, text = normalized
            index = indices[stream]
            expected = native_streams[stream]
            if index >= len(expected):
                extras[stream] += 1
                extra_steps[stream].add(int(value.get("step", -1)))
                continue
            if first_mismatch is None and text != expected[index]:
                first_mismatch = {
                    "stream": stream,
                    "index": index,
                    "original_line": line_number,
                    "original_seq": value.get("seq"),
                    "original": text,
                    "native": expected[index],
                }
            indices[stream] += 1

    stream_report: dict[str, Any] = {}
    native_fully_consumed = True
    for stream in STREAMS:
        consumed = indices[stream]
        native_count = len(native_streams[stream])
        native_fully_consumed &= consumed == native_count
        stream_report[stream] = {
            "native": native_count,
            "original": original_counts[stream],
            "matched_prefix": consumed,
            "original_only_suffix": extras[stream],
            "native_fully_consumed": consumed == native_count,
        }

    epilogue_counts = dict(extras)
    epilogue_valid = all(
        kind in EPILOGUE_TYPES and steps == {final_step}
        for kind, steps in extra_steps.items()
    )
    exact = first_mismatch is None and native_fully_consumed and epilogue_valid
    return {
        "schema": 1,
        "status": "exact_through_final_physics_step" if exact else "mismatch",
        "original": {
            "path": str(original),
            "sha256": digest.hexdigest(),
            "records": record_count,
            "type_counts": dict(sorted(original_counts.items())),
        },
        "native": {
            "path": str(native),
            "sha256": _sha256(native),
            "skipped_init_generations": skip_native_init_generations,
            "bootstrap_lines_skipped": bootstrap_lines,
            "compared_lines": native_lines,
        },
        "comparison": {
            "method": "independent per-operation streams; original JSONL read once",
            "final_physics_step": final_step,
            "streams": stream_report,
            "first_mismatch": first_mismatch,
            "native_streams_exact": first_mismatch is None and native_fully_consumed,
            "original_only_post_step_epilogue": {
                "valid": epilogue_valid,
                "counts": dict(sorted(epilogue_counts.items())),
                "steps": {
                    kind: sorted(steps) for kind, steps in sorted(extra_steps.items())
                },
            },
            "ignored_original_records": dict(sorted(ignored_counts.items())),
            "exact_through_final_physics_step": exact,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, help="original proxy JSONL")
    parser.add_argument("native", type=Path, help="IRISU_EXACT_TRACE output")
    parser.add_argument("--skip-native-init-generations", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.skip_native_init_generations < 0:
        parser.error("skip count must be non-negative")
    try:
        report = compare(
            args.original,
            args.native,
            skip_native_init_generations=args.skip_native_init_generations,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if not report["comparison"]["exact_through_final_physics_step"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
