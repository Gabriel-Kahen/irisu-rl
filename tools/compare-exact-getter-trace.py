#!/usr/bin/env python3
"""Replay an original getter trace against the exact 32-bit Box2D host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, BinaryIO


ROOT = Path(__file__).resolve().parents[1]
HELPER_SOURCE = ROOT / "tools/exact-physics-prototype/getter_trace_replay.c"
MAGIC = b"IRGTRC1\0"
OP = {
    "init": 1,
    "box": 2,
    "triangle": 3,
    "circle": 4,
    "destroy": 5,
    "contact": 6,
    "x": 7,
    "y": 8,
    "r": 9,
    "get_v": 10,
    "set_position": 11,
    "set_user_data": 12,
    "set_v": 13,
    "step": 14,
    "dispose": 15,
    "end": 255,
}


class ComparisonError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _compiler_identity(cc: str) -> str:
    result = subprocess.run(
        [cc, "--version"], check=True, capture_output=True, text=True
    )
    return result.stdout.splitlines()[0]


def build_helper(source: Path, output: Path, cc: str) -> None:
    subprocess.run(
        [
            cc,
            "-m32",
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(source),
            "-ldl",
            "-o",
            str(output),
        ],
        check=True,
    )


def _bits(value: Any, *, context: str) -> int:
    if not isinstance(value, str) or len(value) != 8:
        raise ComparisonError(f"{context}: expected eight hexadecimal f32 digits")
    try:
        result = int(value, 16)
    except ValueError as error:
        raise ComparisonError(f"{context}: malformed f32 bits {value!r}") from error
    if value.lower() != f"{result:08x}":
        raise ComparisonError(f"{context}: malformed f32 bits {value!r}")
    return result


def _integer(value: Any, *, context: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ComparisonError(f"{context}: expected integer >= {minimum}")
    if value > 0xFFFFFFFF:
        raise ComparisonError(f"{context}: integer exceeds uint32")
    return value


class Encoder:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.buffer = bytearray(MAGIC)

    def emit(self, opcode: int, sequence: int, payload: bytes = b"") -> None:
        self.buffer.extend(struct.pack("<BI", opcode, sequence))
        self.buffer.extend(payload)
        if len(self.buffer) >= 1024 * 1024:
            self.flush()

    def finish(self) -> None:
        self.buffer.append(OP["end"])
        self.flush()

    def flush(self) -> None:
        if self.buffer:
            self.stream.write(self.buffer)
            self.buffer.clear()


def _args(record: dict[str, Any], count: int, sequence: int) -> tuple[int, ...]:
    values = record.get("args_f32")
    if not isinstance(values, list) or len(values) != count:
        raise ComparisonError(f"seq {sequence}: wrong f32 argument count")
    return tuple(_bits(value, context=f"seq {sequence}") for value in values)


def _encode_record(
    encoder: Encoder,
    record: dict[str, Any],
    sequence: int,
    active_world: int | None,
) -> int | None:
    kind = record.get("type")
    world = record.get("world")
    if world is not None:
        world = _integer(world, context=f"seq {sequence} world", minimum=1)
        if active_world is not None and world != active_world:
            raise ComparisonError(f"seq {sequence}: unexpected world {world}")

    if kind == "init":
        if (
            record.get("x87_cw_before") != "027f"
            or record.get("x87_cw_after") != "027f"
        ):
            raise ComparisonError(f"seq {sequence}: original init did not use x87 0x027f")
        result = record.get("result")
        if result not in (0, 1) or isinstance(result, bool):
            raise ComparisonError(f"seq {sequence}: invalid init result")
        encoder.emit(
            OP[kind],
            sequence,
            struct.pack("<6IB", *_args(record, 6, sequence), result),
        )
        return world

    if active_world is None:
        raise ComparisonError(f"seq {sequence}: {kind!r} precedes init")
    ordinal = (
        _integer(
            record.get("ordinal"), context=f"seq {sequence} ordinal", minimum=1
        )
        if kind
        in {
            "create",
            "destroy",
            "get_scalar",
            "get_v",
            "set_position",
            "set_user_data",
            "set_v",
        }
        else 0
    )

    if kind == "create":
        shape = record.get("shape")
        if shape not in {"box", "triangle", "circle"}:
            raise ComparisonError(f"seq {sequence}: unknown shape {shape!r}")
        count = 6 if shape == "circle" else 8
        encoder.emit(
            OP[shape],
            sequence,
            struct.pack(f"<{count + 1}I", ordinal, *_args(record, count, sequence)),
        )
    elif kind == "destroy":
        encoder.emit(OP[kind], sequence, struct.pack("<I", ordinal))
    elif kind == "contact":
        result = record.get("result")
        if not isinstance(result, bool):
            raise ComparisonError(f"seq {sequence}: contact result is not boolean")
        a = _integer(record.get("a_user", 0), context=f"seq {sequence} a_user")
        b = _integer(record.get("b_user", 0), context=f"seq {sequence} b_user")
        encoder.emit(OP[kind], sequence, struct.pack("<BII", result, a, b))
    elif kind == "get_scalar":
        field = record.get("field")
        if field not in {"x", "y", "r"}:
            raise ComparisonError(f"seq {sequence}: unknown scalar field {field!r}")
        expected = _bits(record.get("value_f32"), context=f"seq {sequence}")
        encoder.emit(OP[field], sequence, struct.pack("<II", ordinal, expected))
    elif kind == "get_v":
        encoder.emit(
            OP[kind],
            sequence,
            struct.pack("<III", ordinal, *_args(record, 2, sequence)),
        )
    elif kind == "set_position":
        encoder.emit(
            OP[kind],
            sequence,
            struct.pack("<4I", ordinal, *_args(record, 3, sequence)),
        )
    elif kind == "set_user_data":
        user = _integer(record.get("user"), context=f"seq {sequence} user")
        encoder.emit(OP[kind], sequence, struct.pack("<II", ordinal, user))
    elif kind == "set_v":
        encoder.emit(
            OP[kind],
            sequence,
            struct.pack("<3I", ordinal, *_args(record, 2, sequence)),
        )
    elif kind == "step":
        step = _integer(record.get("step"), context=f"seq {sequence} step", minimum=1)
        iterations = _integer(
            record.get("iterations"), context=f"seq {sequence} iterations"
        )
        dt = _bits(record.get("dt_f32"), context=f"seq {sequence}")
        encoder.emit(OP[kind], sequence, struct.pack("<III", step, dt, iterations))
    elif kind == "dispose":
        encoder.emit(OP[kind], sequence)
        return None
    else:
        raise ComparisonError(f"seq {sequence}: unknown record type {kind!r}")
    return active_world


def _feed_trace(
    original: Path,
    stream: BinaryIO,
    *,
    progress_every: int,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    counts: Counter[str] = Counter()
    encoder = Encoder(stream)
    active_world: int | None = None
    record_count = 0
    final_step = 0
    with original.open("rb") as source:
        for line_number, raw in enumerate(source, 1):
            digest.update(raw)
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ComparisonError(f"line {line_number}: invalid JSON: {error}") from error
            if not isinstance(record, dict):
                raise ComparisonError(f"line {line_number}: expected JSON object")
            sequence = record.get("seq")
            if sequence != record_count:
                raise ComparisonError(
                    f"line {line_number}: expected seq {record_count}, got {sequence!r}"
                )
            kind = record.get("type")
            if not isinstance(kind, str):
                raise ComparisonError(f"seq {sequence}: missing record type")
            counts[kind] += 1
            if sequence == 0:
                if (
                    kind != "proxy_loaded"
                    or record.get("schema") != 1
                    or record.get("ok") is not True
                    or record.get("x87_cw") != "027f"
                ):
                    raise ComparisonError("trace lacks a valid canonical proxy header")
            else:
                active_world = _encode_record(
                    encoder, record, sequence, active_world
                )
            if kind == "step":
                final_step = max(final_step, int(record["step"]))
            record_count += 1
            if progress_every and record_count % progress_every == 0:
                print(f"streamed {record_count:,} records", file=sys.stderr)
    if record_count == 0:
        raise ComparisonError("trace is empty")
    encoder.finish()
    return {
        "path": str(original),
        "sha256": digest.hexdigest(),
        "records": record_count,
        "type_counts": dict(sorted(counts.items())),
        "final_physics_step": final_step,
    }


def _clean_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LD_") and key != "GLIBC_TUNABLES"
    }


def compare(
    original: Path,
    library: Path,
    *,
    helper: Path | None = None,
    cc: str = "cc",
    progress_every: int = 1_000_000,
) -> dict[str, Any]:
    original = original.resolve(strict=True)
    library = library.resolve(strict=True)
    with library.open("rb") as stream:
        elf_header = stream.read(5)
    if elf_header != b"\x7fELF\x01":
        raise ComparisonError("exact library is not an ELF32 object")

    with tempfile.TemporaryDirectory(prefix="irisu-getter-trace-") as directory:
        helper_supplied = helper is not None
        if helper is None:
            helper_path = Path(directory) / "getter-trace-replay"
            build_helper(HELPER_SOURCE, helper_path, cc)
        else:
            helper_path = helper.resolve(strict=True)
        helper_sha256 = _sha256(helper_path)
        process = subprocess.Popen(
            [str(helper_path), str(library)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/",
            env=_clean_environment(),
        )
        assert process.stdin is not None
        assert process.stdout is not None
        assert process.stderr is not None
        try:
            original_report = _feed_trace(
                original, process.stdin, progress_every=progress_every
            )
            process.stdin.close()
            stdout = process.stdout.read()
            stderr = process.stderr.read()
            process.stdout.close()
            process.stderr.close()
            status = process.wait()
        except BaseException:
            process.kill()
            process.wait()
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()
            raise

    if status not in (0, 1):
        detail = stderr.decode("utf-8", errors="replace").strip()
        raise ComparisonError(
            f"getter replay helper failed with status {status}: {detail}"
        )
    try:
        helper_report = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise ComparisonError("getter replay helper emitted invalid JSON") from error
    if helper_report.get("status") not in {"exact", "mismatch"}:
        raise ComparisonError("getter replay helper emitted an invalid status")

    counts = original_report["type_counts"]
    expected = {
        "commands": original_report["records"] - counts.get("proxy_loaded", 0),
        "final_step": original_report["final_physics_step"],
        "scalar_getters": counts.get("get_scalar", 0),
        "velocity_getters": counts.get("get_v", 0),
        "getter_records": counts.get("get_scalar", 0) + counts.get("get_v", 0),
        "getter_values": counts.get("get_scalar", 0) + 2 * counts.get("get_v", 0),
        "contacts": counts.get("contact", 0),
    }
    for field, value in expected.items():
        if helper_report.get(field) != value:
            raise ComparisonError(
                f"getter replay coverage mismatch for {field}: "
                f"expected {value}, got {helper_report.get(field)!r}"
            )

    canonical = helper_report.get("canonical_x87") is True
    getters_exact = canonical and helper_report["getter_record_mismatches"] == 0
    contacts_exact = canonical and helper_report["contact_mismatches"] == 0
    exact = (
        status == 0
        and helper_report["status"] == "exact"
        and getters_exact
        and contacts_exact
    )
    library_stat = os.stat(library)
    return {
        "schema": 1,
        "status": "exact_all_getters" if exact else "mismatch",
        "original": original_report,
        "exact_backend": {
            "library_path": str(library),
            "library_sha256": _sha256(library),
            "library_device": f"{library_stat.st_dev:x}",
            "library_inode": library_stat.st_ino,
            "helper_source": str(HELPER_SOURCE),
            "helper_source_sha256": _sha256(HELPER_SOURCE),
            "helper_sha256": helper_sha256,
            "helper_supplied": helper_supplied,
            "compiler": None if helper_supplied else _compiler_identity(cc),
        },
        "comparison": {
            "method": "single-pass global-call-order replay; raw binary32 equality",
            "all_recorded_commands_replayed": True,
            "all_getter_records_exact": getters_exact,
            "all_contact_records_exact": contacts_exact,
            **helper_report,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path, help="getter-enabled proxy JSONL")
    parser.add_argument("library", type=Path, help="exact multiworld ELF32 library")
    parser.add_argument("--helper", type=Path, help="prebuilt 32-bit replay helper")
    parser.add_argument("--cc", default="cc", help="32-bit-capable C compiler")
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    try:
        report = compare(
            args.original,
            args.library,
            helper=args.helper,
            cc=args.cc,
            progress_every=args.progress_every,
        )
    except (ComparisonError, OSError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["status"] == "exact_all_getters" else 1


if __name__ == "__main__":
    raise SystemExit(main())
