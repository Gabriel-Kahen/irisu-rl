#!/usr/bin/env python3
"""Extract compact, decoded body states from a getter-enabled proxy trace."""

from __future__ import annotations

import argparse
import json
import struct
from collections import defaultdict
from pathlib import Path
from typing import Any


def f32(bits: str) -> float:
    return struct.unpack(">f", bytes.fromhex(bits))[0]


def decode_arguments(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    if "args_f32" in result:
        result["args"] = [f32(value) for value in result["args_f32"]]
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--ordinals", default="14,107")
    parser.add_argument("--start-step", type=int, default=4000)
    parser.add_argument("--end-step", type=int, default=4035)
    args = parser.parse_args()

    ordinals = {int(value) for value in args.ordinals.split(",")}
    states: dict[tuple[int, int], dict[str, Any]] = defaultdict(dict)
    context: list[dict[str, Any]] = []

    with args.trace.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            event = record["type"]
            step = record.get("step", -1)
            ordinal = record.get("ordinal")

            if event in {"create", "set_v", "set_user_data", "destroy"} and ordinal in ordinals:
                context.append(decode_arguments(record))
            elif (
                event == "contact"
                and args.start_step <= step <= args.end_step
                and (record.get("a_ordinal") in ordinals or record.get("b_ordinal") in ordinals)
            ):
                context.append(record)
            elif event == "get_scalar" and ordinal in ordinals and args.start_step <= step <= args.end_step:
                state = states[(step, ordinal)]
                state.update(
                    {
                        "type": "body_state",
                        "seq": min(record["seq"], state.get("seq", record["seq"])),
                        "seq_end": max(record["seq"], state.get("seq_end", record["seq"])),
                        "step": step,
                        "ordinal": ordinal,
                        "body": record["body"],
                        "user": record["user"],
                        f"{record['field']}_f32": record["value_f32"],
                        record["field"]: f32(record["value_f32"]),
                    }
                )
            elif event == "get_v" and ordinal in ordinals and args.start_step <= step <= args.end_step:
                state = states[(step, ordinal)]
                state.update(
                    {
                        "type": "body_state",
                        "seq": min(record["seq"], state.get("seq", record["seq"])),
                        "seq_end": max(record["seq"], state.get("seq_end", record["seq"])),
                        "step": step,
                        "ordinal": ordinal,
                        "body": record["body"],
                        "user": record["user"],
                        "v_f32": record["args_f32"],
                        "v": [f32(value) for value in record["args_f32"]],
                    }
                )

    rows = context + [states[key] for key in sorted(states)]
    rows.sort(key=lambda row: (row.get("step", -1), row.get("seq", 1 << 62), row.get("ordinal", -1)))
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            json.dump(row, stream, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
