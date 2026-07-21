#!/usr/bin/env python3
"""Structural validator and summary for Box2D trace-proxy JSONL."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


F32 = re.compile(r"^[0-9a-f]{8}$")
X87_CW = re.compile(r"^[0-9a-f]{4}$")
ARG_COUNTS = {"init": 6, "set_position": 3, "set_v": 2, "get_v": 2}
KNOWN_TYPES = {
    "proxy_loaded",
    "mapping_overflow",
    "init",
    "dispose",
    "create",
    "destroy",
    "contact",
    "set_position",
    "set_user_data",
    "set_v",
    "step",
    "get_scalar",
    "get_v",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"line {line_number}: invalid JSON: {error}") from error
            require(isinstance(value, dict), f"line {line_number}: expected object")
            records.append(value)
    require(records, "trace is empty")
    return records


def validate(path: Path) -> Counter[str]:
    records = read_records(path)
    first = records[0]
    require(first.get("type") == "proxy_loaded", "first record is not proxy_loaded")
    require(first.get("schema") == 1, "unsupported trace schema")
    require(first.get("ok") is True, "proxy did not resolve the complete ABI")
    require(first.get("export_mask") == "0000ffff", "wrong resolved-export mask")
    require(
        isinstance(first.get("x87_cw"), str) and X87_CW.fullmatch(first["x87_cw"]),
        "proxy_loaded has malformed x87 control word",
    )

    counts: Counter[str] = Counter()
    create_next: dict[int, int] = {}
    latest_step: dict[int, int] = {}
    contact_next: dict[tuple[int, int], int] = {}
    contact_closed: set[tuple[int, int]] = set()

    for expected_sequence, record in enumerate(records):
        require(
            record.get("seq") == expected_sequence,
            f"record {expected_sequence}: non-contiguous seq {record.get('seq')!r}",
        )
        event = record.get("type")
        require(event in KNOWN_TYPES, f"record {expected_sequence}: unknown type {event!r}")
        counts[event] += 1
        require(event != "mapping_overflow", "body/user mapping capacity overflowed")

        if event in ARG_COUNTS:
            arguments = record.get("args_f32")
            require(
                isinstance(arguments, list) and len(arguments) == ARG_COUNTS[event],
                f"record {expected_sequence}: wrong {event} argument vector",
            )
            require(
                all(isinstance(value, str) and F32.fullmatch(value) for value in arguments),
                f"record {expected_sequence}: malformed f32 bits",
            )
        if event == "init":
            for field in ("x87_cw_before", "x87_cw_after"):
                require(
                    isinstance(record.get(field), str)
                    and X87_CW.fullmatch(record[field]),
                    f"record {expected_sequence}: malformed {field}",
                )

        if event == "create":
            world = record.get("world")
            ordinal = record.get("ordinal")
            shape = record.get("shape")
            require(isinstance(world, int) and world > 0, "create has invalid world")
            require(shape in {"box", "circle", "triangle"}, "create has invalid shape")
            expected_count = 6 if shape == "circle" else 8
            arguments = record.get("args_f32")
            require(
                isinstance(arguments, list)
                and len(arguments) == expected_count
                and all(isinstance(value, str) and F32.fullmatch(value) for value in arguments),
                f"record {expected_sequence}: malformed create arguments",
            )
            expected_ordinal = create_next.setdefault(world, 1)
            require(ordinal == expected_ordinal, f"world {world}: create ordinal gap")
            create_next[world] += 1

        if "world" in record:
            world = record["world"]
            require(isinstance(world, int) and world > 0, "invalid world number")
            if "step" in record:
                step = record["step"]
                require(isinstance(step, int) and step >= 0, "invalid step number")
                require(step >= latest_step.get(world, 0), f"world {world}: step regressed")
                latest_step[world] = step

        if event == "step":
            require(
                isinstance(record.get("dt_f32"), str)
                and F32.fullmatch(record["dt_f32"]),
                f"record {expected_sequence}: malformed step dt",
            )
            require(isinstance(record.get("iterations"), int), "invalid iterations")

        if event == "get_scalar":
            require(record.get("field") in {"x", "y", "r"}, "invalid scalar field")
            require(
                isinstance(record.get("value_f32"), str)
                and F32.fullmatch(record["value_f32"]),
                f"record {expected_sequence}: malformed scalar result",
            )

        if event == "contact":
            key = (record["world"], record["step"])
            require(key not in contact_closed, f"step {key}: contact after cursor end")
            expected_call = contact_next.setdefault(key, 1)
            require(record.get("call") == expected_call, f"step {key}: contact call gap")
            contact_next[key] += 1
            require(isinstance(record.get("result"), bool), "invalid contact result")
            if record["result"] is False:
                contact_closed.add(key)

    require(counts["init"] > 0, "trace contains no init")
    require(counts["proxy_loaded"] == 1, "trace must contain exactly one proxy_loaded")
    require(
        contact_closed == set(contact_next),
        "trace ends with an unterminated contact cursor batch",
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    if not args.trace.is_file():
        parser.error(f"missing trace: {args.trace}")
    try:
        counts = validate(args.trace)
    except ValueError as error:
        parser.error(str(error))
    detail = " ".join(f"{name}={counts[name]}" for name in sorted(counts))
    print(f"validated trace: {sum(counts.values())} records ({detail})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
