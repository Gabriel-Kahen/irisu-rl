#!/usr/bin/env python3
"""Validate and score original-game-derived golden scenarios.

Exit codes are 0 for a passed gate, 1 for an evaluated fidelity failure, and 2
when the manifest or its evidence is not evaluable.  This tool is intentionally
separate from ``evaluate-rpy.py``: replay headers alone are not golden evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping, Sequence, TextIO


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from irisu_env import IrisuEnv  # noqa: E402
from irisu_env.exact_ipc import (  # noqa: E402
    _mount_device,
    normalize_exact_library_sha256,
)
from irisu_env.native import NativeError  # noqa: E402


CATEGORIES = ("match", "rot", "chain", "ejection", "orb")
THRESHOLD_PERCENT = 95
MIN_TRAJECTORY_HORIZON_FRAMES = 25
MAX_TRAJECTORY_HORIZON_FRAMES = 50
MAX_TRAJECTORY_TOLERANCE = 15.0
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAPPED_DEVICE = re.compile(r"[0-9a-f]+:[0-9a-f]+")
_U64 = re.compile(r"0x[0-9a-f]{16}")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_MAX_PNG_DECOMPRESSED_BYTES = 256 * 1024 * 1024
_EVENT_KINDS = {
    "invalid_action",
    "spawned",
    "shot_fired",
    "activated",
    "contact",
    "confirmed",
    "chain_joined",
    "cleared",
    "rotten",
    "ejected",
    "destroyed",
    "gauge_changed",
    "score_changed",
    "level_changed",
    "game_over",
    "projectile_hit",
    "projectile_contact",
    "held_input_ignored",
    "level_completed",
}
_RELEVANT_EVENTS = {
    "match": {"confirmed", "cleared"},
    "rot": {"rotten"},
    "chain": {"chain_joined", "confirmed"},
    "ejection": {"ejected"},
    "orb": {"cleared", "destroyed", "gauge_changed"},
}
_FILE_ROLES = {
    "metadata",
    "actions",
    "measurements",
    "replay",
    "frame",
    "video",
    "notes",
}
_HASH_KEYS = (
    "executable_sha256",
    "dat_dxa_sha256",
    "config_sha256",
    "box2d_sha256",
)
CANONICAL_TARGET: Mapping[str, str] = MappingProxyType(
    {
        "profile": "v2.03-normal",
        "game_version": "v2.03 with English-patched data",
        "executable_sha256": (
            "0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255"
        ),
        "dat_dxa_sha256": (
            "b36ef6864bf2d0e626d5087edb5b571ef548ebd5dde9fbc9b87f7b4ac3e89d4a"
        ),
        "config_sha256": (
            "1e29431fe8209c25784d4741f7972737561281169bbb5a56f62e3e0f0b63de35"
        ),
        "box2d_sha256": (
            "34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd"
        ),
        "clone_config_u64": "0xec0e8463feaf2670",
    }
)


class GoldenError(ValueError):
    """The manifest or evidence cannot support a fidelity verdict."""


@dataclass(frozen=True, slots=True)
class CapturedInput:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int, int]


@dataclass(frozen=True, slots=True)
class ValidatedScenario:
    raw: dict[str, Any]
    replay: Any
    reported_repeat_count: int
    source_probe_sha256: str
    inputs: tuple[CapturedInput, ...]


@dataclass(frozen=True, slots=True)
class ValidatedManifest:
    path: Path
    raw: dict[str, Any]
    scenarios: tuple[ValidatedScenario, ...]
    manifest_input: CapturedInput
    inputs: tuple[CapturedInput, ...]


def _load_module(name: str, path: Path) -> Any:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVALUATE_RPY = _load_module("irisu_evaluate_rpy", ROOT / "tools" / "evaluate-rpy.py")
INSPECT_RPY = EVALUATE_RPY.INSPECT_RPY


def _reject_constant(value: str) -> None:
    raise GoldenError(f"non-finite JSON number {value!r} is forbidden")


def _read_json_bytes(data: bytes, path: Path) -> Any:
    try:
        return json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GoldenError(f"cannot read JSON {path}: {exc}") from exc


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GoldenError(f"{where} must be an object")
    return value


def _array(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise GoldenError(f"{where} must be an array")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    where: str,
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = sorted(required_set - value.keys())
    extra = sorted(value.keys() - allowed)
    if missing:
        raise GoldenError(f"{where} is missing fields: {', '.join(missing)}")
    if extra:
        raise GoldenError(f"{where} has unknown fields: {', '.join(extra)}")


def _integer(value: Any, where: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise GoldenError(f"{where} must be an integer")
    if minimum is not None and value < minimum:
        raise GoldenError(f"{where} must be at least {minimum}")
    return value


def _number(value: Any, where: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GoldenError(f"{where} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise GoldenError(f"{where} must be finite")
    if minimum is not None and result < minimum:
        raise GoldenError(f"{where} must be at least {minimum}")
    return result


def _string(value: Any, where: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise GoldenError(f"{where} must be a nonempty string")
    return value


def _sha(value: Any, where: str) -> str:
    text = _string(value, where)
    if _SHA256.fullmatch(text) is None:
        raise GoldenError(f"{where} must be a lowercase SHA-256")
    return text


def _u64(value: Any, where: str) -> str:
    text = _string(value, where)
    if _U64.fullmatch(text) is None:
        raise GoldenError(f"{where} must be 0x followed by 16 lowercase hex digits")
    return text


def _identifier(value: Any, where: str) -> str:
    text = _string(value, where)
    if _IDENTIFIER.fullmatch(text) is None:
        raise GoldenError(f"{where} is not a safe identifier")
    return text


def _resolved_child(base: Path, relative: Any, where: str) -> Path:
    text = _string(relative, where)
    child = Path(text)
    if child.is_absolute():
        raise GoldenError(f"{where} must be relative")
    resolved_base = base.resolve()
    resolved = (resolved_base / child).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as exc:
        raise GoldenError(f"{where} escapes its evidence bundle") from exc
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _resolve_bundle(manifest_dir: Path, relative: Any, where: str) -> Path:
    text = _string(relative, where)
    child = Path(text)
    if child.is_absolute():
        raise GoldenError(f"{where} must be relative to the manifest")

    unresolved = manifest_dir / child
    lexical = Path(os.path.abspath(unresolved))
    allowed_roots = (manifest_dir, manifest_dir.parent / "captures")
    if not any(_is_within(lexical, root) for root in allowed_roots):
        raise GoldenError(
            f"{where} must stay under the manifest directory or its sibling captures directory"
        )
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise GoldenError(f"{where} is missing: {lexical}") from exc
    if resolved != lexical:
        raise GoldenError(f"{where} must not traverse a symlink")
    if not resolved.is_dir():
        raise GoldenError(f"{where} is not a directory: {resolved}")
    return resolved


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _capture_input(
    path: Path, *, keep_bytes: bool
) -> tuple[CapturedInput, bytes | None]:
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if keep_bytes else None
    byte_count = 0
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while block := stream.read(1024 * 1024):
                byte_count += len(block)
                digest.update(block)
                if chunks is not None:
                    chunks.append(block)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as exc:
        raise GoldenError(f"cannot read validated input {path}: {exc}") from exc
    identity = _file_identity(after)
    if (
        _file_identity(before) != identity
        or identity != _file_identity(current)
        or byte_count != after.st_size
    ):
        raise GoldenError(f"validated input changed while it was captured: {path}")
    captured = CapturedInput(path, digest.hexdigest(), identity)
    return captured, None if chunks is None else b"".join(chunks)


def _verify_inputs_unchanged(inputs: Sequence[CapturedInput]) -> None:
    for expected in inputs:
        current, _ = _capture_input(expected.path, keep_bytes=False)
        if current != expected:
            raise GoldenError(
                f"validated input changed during scenario scoring: {expected.path}"
            )


def _validate_actions(data: bytes, path: Path) -> None:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise GoldenError(f"cannot read action evidence {path}: {exc}") from exc
    if not lines:
        raise GoldenError(f"action evidence is empty: {path}")
    sequence_field: str | None = None
    previous_sequence = 0
    for index, line in enumerate(lines, 1):
        if not line:
            raise GoldenError(f"action evidence {path}:{index} is blank")
        try:
            record = json.loads(line, parse_constant=_reject_constant)
        except (json.JSONDecodeError, GoldenError) as exc:
            raise GoldenError(f"invalid action evidence {path}:{index}: {exc}") from exc
        if not isinstance(record, dict):
            raise GoldenError(f"action evidence {path}:{index} must be an object")
        present_sequence_fields = [
            name for name in ("sequence", "monotonic_sequence") if name in record
        ]
        if len(present_sequence_fields) != 1:
            raise GoldenError(
                f"action evidence {path}:{index} must contain exactly one of "
                "sequence or monotonic_sequence"
            )
        current_field = present_sequence_fields[0]
        if sequence_field is None:
            sequence_field = current_field
        elif current_field != sequence_field:
            raise GoldenError(
                f"action evidence {path}:{index} changes sequence field from "
                f"{sequence_field} to {current_field}"
            )
        sequence = _integer(
            record[current_field],
            f"action evidence {path}:{index}.{current_field}",
            minimum=1,
        )
        if sequence <= previous_sequence:
            raise GoldenError(
                f"action evidence {path}:{index}.{current_field} must be strictly increasing"
            )
        previous_sequence = sequence
        action = _string(
            record.get("action"), f"action evidence {path}:{index}.action"
        )
        if not action.strip():
            raise GoldenError(f"action evidence {path}:{index}.action must not be blank")
        result = _string(
            record.get("result"), f"action evidence {path}:{index}.result"
        )
        if not result.strip():
            raise GoldenError(f"action evidence {path}:{index}.result must not be blank")


def _capture_clone_library(env: Any) -> dict[str, Any]:
    required = isinstance(env, IrisuEnv)
    try:
        raw_path = env.library_path
    except (AttributeError, NotImplementedError):
        if required:
            raise GoldenError("IrisuEnv does not expose its loaded clone library path")
        return {
            "status": "unavailable",
            "reason": "custom environment does not expose library_path",
        }

    try:
        path = Path(raw_path).expanduser().resolve(strict=True)
    except (OSError, TypeError, ValueError) as exc:
        if required:
            raise GoldenError(
                "loaded clone library is not a readable filesystem path; "
                "pass --library with an explicit path"
            ) from exc
        return {
            "status": "unavailable",
            "reason": "custom environment library_path is not a readable file",
        }
    if not path.is_file():
        if required:
            raise GoldenError(f"loaded clone library is not a regular file: {path}")
        return {
            "status": "unavailable",
            "reason": "custom environment library_path is not a regular file",
        }

    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while block := stream.read(1024 * 1024):
                byte_count += len(block)
                digest.update(block)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except OSError as exc:
        raise GoldenError(f"cannot read loaded clone library {path}: {exc}") from exc

    if (
        _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(current)
        or byte_count != after.st_size
    ):
        raise GoldenError(f"loaded clone library changed while it was hashed: {path}")
    return {
        "status": "captured",
        "path": str(path),
        "bytes": byte_count,
        "sha256": digest.hexdigest(),
        "file_identity": {
            "device": after.st_dev,
            "inode": after.st_ino,
            "mtime_ns": after.st_mtime_ns,
            "ctime_ns": after.st_ctime_ns,
        },
    }


def _verify_clone_library_stability(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    if before != after:
        raise GoldenError("loaded clone library changed during scenario scoring")
    if before["status"] == "unavailable":
        return {**before, "stability_check": "not_applicable"}
    return {**before, "status": "verified", "stability_check": "passed"}


def _capture_exact_library(env: Any) -> dict[str, Any]:
    try:
        raw = _object(
            env.exact_library_provenance(), "exact library provenance"
        )
    except (AttributeError, NotImplementedError) as exc:
        raise GoldenError(
            "exact environment does not expose mapped-library provenance"
        ) from exc
    _exact_keys(
        raw,
        "exact library provenance",
        required=(
            "status",
            "path",
            "bytes",
            "sha256",
            "file_identity",
            "mapped_identity",
        ),
    )
    if raw["status"] != "captured":
        raise GoldenError("exact library provenance was not captured")
    try:
        path = Path(_string(raw["path"], "exact library provenance.path")).resolve(
            strict=True
        )
    except OSError as exc:
        raise GoldenError("mapped exact library path is not readable") from exc
    if str(path) != raw["path"] or not path.is_file():
        raise GoldenError("mapped exact library path is not a resolved regular file")
    byte_count = _integer(
        raw["bytes"], "exact library provenance.bytes", minimum=1
    )
    sha256 = normalize_exact_library_sha256(raw["sha256"])
    if sha256 is None:
        raise GoldenError("mapped exact library has an invalid SHA-256")

    file_identity = _object(
        raw["file_identity"], "exact library provenance.file_identity"
    )
    _exact_keys(
        file_identity,
        "exact library provenance.file_identity",
        required=("device", "inode", "mtime_ns", "ctime_ns"),
    )
    normalized_file_identity = {
        key: _integer(
            file_identity[key],
            f"exact library provenance.file_identity.{key}",
            minimum=0,
        )
        for key in ("device", "inode", "mtime_ns", "ctime_ns")
    }
    mapped_identity = _object(
        raw["mapped_identity"], "exact library provenance.mapped_identity"
    )
    _exact_keys(
        mapped_identity,
        "exact library provenance.mapped_identity",
        required=("device", "inode"),
    )
    mapped_device = _string(
        mapped_identity["device"], "exact library provenance.mapped_identity.device"
    )
    if _MAPPED_DEVICE.fullmatch(mapped_device) is None:
        raise GoldenError("mapped exact library has an invalid mapped device")
    try:
        parent_device = _mount_device(os.getpid(), str(path))
    except NativeError as exc:
        raise GoldenError(
            "cannot verify the mapped exact library's client mount"
        ) from exc
    if parent_device != mapped_device:
        raise GoldenError(
            "mapped exact library device disagrees with the client mount"
        )
    mapped_inode = _integer(
        mapped_identity["inode"],
        "exact library provenance.mapped_identity.inode",
        minimum=1,
    )

    captured, _ = _capture_input(path, keep_bytes=False)
    device, inode, size, mtime_ns, ctime_ns = captured.identity
    if (
        captured.sha256 != sha256
        or size != byte_count
        or normalized_file_identity
        != {
            "device": device,
            "inode": inode,
            "mtime_ns": mtime_ns,
            "ctime_ns": ctime_ns,
        }
        or mapped_inode != inode
    ):
        raise GoldenError(
            "mapped exact library provenance disagrees with its current bytes"
        )
    return {
        "status": "captured",
        "path": str(path),
        "bytes": byte_count,
        "sha256": sha256,
        "file_identity": normalized_file_identity,
        "mapped_identity": {"device": mapped_device, "inode": mapped_inode},
    }


def _verify_exact_library_stability(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    if before != after:
        raise GoldenError("mapped exact library changed during scenario scoring")
    return {**before, "status": "verified", "stability_check": "passed"}


def _png_rows(
    width: int, height: int, bits_per_pixel: int, interlace: int
) -> list[int]:
    passes = (
        ((0, 0, 1, 1),)
        if interlace == 0
        else (
            (0, 0, 8, 8),
            (4, 0, 8, 8),
            (0, 4, 4, 8),
            (2, 0, 4, 4),
            (0, 2, 2, 4),
            (1, 0, 2, 2),
            (0, 1, 1, 2),
        )
    )
    rows: list[int] = []
    for start_x, start_y, step_x, step_y in passes:
        pass_width = max(0, (width - start_x + step_x - 1) // step_x)
        pass_height = max(0, (height - start_y + step_y - 1) // step_y)
        if pass_width and pass_height:
            row_bytes = (pass_width * bits_per_pixel + 7) // 8
            rows.extend([row_bytes] * pass_height)
    return rows


def _validate_png(data: bytes, path: Path) -> None:
    if path.suffix.lower() != ".png" or not data.startswith(_PNG_SIGNATURE):
        raise GoldenError(f"frame evidence must be a PNG capture: {path}")

    offset = len(_PNG_SIGNATURE)
    chunk_index = 0
    header: tuple[int, int, int, int, int] | None = None
    palette_entries: int | None = None
    compressed = bytearray()
    saw_idat = False
    idat_closed = False
    saw_iend = False
    while offset < len(data):
        if len(data) - offset < 12:
            raise GoldenError(f"PNG chunk is truncated: {path}")
        length = struct.unpack_from(">I", data, offset)[0]
        if length > 0x7FFFFFFF:
            raise GoldenError(f"PNG chunk exceeds the format limit: {path}")
        chunk_type = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            raise GoldenError(f"PNG chunk payload is truncated: {path}")
        payload = data[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack_from(">I", data, offset + 8 + length)[0]
        actual_crc = zlib.crc32(payload, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise GoldenError(f"PNG chunk CRC mismatch for {path}")
        if len(chunk_type) != 4 or not all(
            ord("A") <= byte <= ord("Z") or ord("a") <= byte <= ord("z")
            for byte in chunk_type
        ):
            raise GoldenError(f"PNG chunk type is invalid: {path}")
        if chunk_type[2] & 0x20:
            raise GoldenError(f"PNG chunk uses the reserved type bit: {path}")
        if chunk_index == 0 and chunk_type != b"IHDR":
            raise GoldenError(f"PNG IHDR must be the first chunk: {path}")

        if chunk_type == b"IHDR":
            if header is not None or length != 13:
                raise GoldenError(f"PNG must contain one 13-byte IHDR: {path}")
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", payload)
            )
            valid_depths = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if (
                width == 0
                or height == 0
                or width > 0x7FFFFFFF
                or height > 0x7FFFFFFF
                or color_type not in valid_depths
                or bit_depth not in valid_depths[color_type]
                or compression != 0
                or filtering != 0
                or interlace not in (0, 1)
            ):
                raise GoldenError(f"PNG IHDR fields are invalid: {path}")
            channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
            header = (width, height, bit_depth * channels, color_type, interlace)
        elif chunk_type == b"PLTE":
            if header is None or saw_idat or palette_entries is not None:
                raise GoldenError(f"PNG PLTE ordering is invalid: {path}")
            if length == 0 or length % 3 or length > 768:
                raise GoldenError(f"PNG PLTE length is invalid: {path}")
            palette_entries = length // 3
        elif chunk_type == b"IDAT":
            if header is None or idat_closed:
                raise GoldenError(f"PNG IDAT ordering is invalid: {path}")
            saw_idat = True
            compressed.extend(payload)
        elif chunk_type == b"IEND":
            if length != 0 or not saw_idat:
                raise GoldenError(f"PNG IEND is invalid or precedes IDAT: {path}")
            saw_iend = True
            offset = end
            if offset != len(data):
                raise GoldenError(f"PNG has bytes after IEND: {path}")
            break
        else:
            if not (chunk_type[0] & 0x20):
                raise GoldenError(f"PNG contains an unknown critical chunk: {path}")
            if saw_idat:
                idat_closed = True
        chunk_index += 1
        offset = end

    if header is None or not saw_idat or not saw_iend:
        raise GoldenError(f"PNG is missing IHDR, IDAT, or IEND: {path}")
    width, height, bits_per_pixel, color_type, interlace = header
    if color_type == 3:
        if palette_entries is None or palette_entries > 1 << bits_per_pixel:
            raise GoldenError(f"indexed PNG has an invalid or missing PLTE: {path}")
    elif color_type in (0, 4) and palette_entries is not None:
        raise GoldenError(f"grayscale PNG must not contain PLTE: {path}")

    rows = _png_rows(width, height, bits_per_pixel, interlace)
    expected_size = sum(row_bytes + 1 for row_bytes in rows)
    if expected_size > _MAX_PNG_DECOMPRESSED_BYTES:
        raise GoldenError(f"PNG decompressed image is too large: {path}")
    try:
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(bytes(compressed), expected_size + 1)
        if decoder.unconsumed_tail or len(decoded) > expected_size:
            raise GoldenError(f"PNG scanline payload exceeds IHDR dimensions: {path}")
        decoded += decoder.flush(expected_size + 1 - len(decoded))
    except zlib.error as exc:
        raise GoldenError(f"PNG IDAT zlib stream is invalid: {path}") from exc
    if (
        not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
        or len(decoded) != expected_size
    ):
        raise GoldenError(f"PNG scanline payload disagrees with IHDR dimensions: {path}")
    position = 0
    for row_bytes in rows:
        if decoded[position] > 4:
            raise GoldenError(f"PNG scanline uses an invalid filter: {path}")
        position += row_bytes + 1


def _source_probe_sha256(replay: Any, last_frame: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"irisu-source-probe-v1\0")
    digest.update(struct.pack("<I", int(replay.header.seed) & 0xFFFFFFFF))
    previous_left = False
    previous_right = False
    for frame in replay.frames[: last_frame + 1]:
        left = bool(frame.left)
        right = bool(frame.right)
        digest.update(bytes((int(left) | (int(right) << 1),)))
        if (left and not previous_left) or (right and not previous_right):
            digest.update(struct.pack("<HH", int(frame.x), int(frame.y)))
        previous_left = left
        previous_right = right
    return digest.hexdigest()


def _validate_target(value: Any) -> dict[str, Any]:
    target = _object(value, "target")
    _exact_keys(
        target,
        "target",
        required=("profile", "game_version", *_HASH_KEYS, "clone_config_u64"),
    )
    _string(target["profile"], "target.profile")
    _string(target["game_version"], "target.game_version")
    for key in _HASH_KEYS:
        _sha(target[key], f"target.{key}")
    _u64(target["clone_config_u64"], "target.clone_config_u64")
    mismatches = [
        key for key, expected in CANONICAL_TARGET.items() if target[key] != expected
    ]
    if mismatches:
        raise GoldenError(
            "target does not match the canonical v2.03 target: "
            + ", ".join(mismatches)
        )
    return target


def _validate_event(
    value: Any, where: str, *, first_frame: int, last_frame: int
) -> dict[str, Any]:
    event = _object(value, where)
    _exact_keys(
        event,
        where,
        required=("kind", "min_count", "max_count"),
        optional=("first_frame", "last_frame", "a", "b", "value", "detail"),
    )
    kind = _string(event["kind"], f"{where}.kind")
    if kind not in _EVENT_KINDS:
        raise GoldenError(f"{where}.kind is unknown: {kind!r}")
    minimum = _integer(event["min_count"], f"{where}.min_count", minimum=0)
    maximum = _integer(event["max_count"], f"{where}.max_count", minimum=0)
    if minimum > maximum:
        raise GoldenError(f"{where}.min_count exceeds max_count")
    window_first = _integer(event.get("first_frame", first_frame), f"{where}.first_frame")
    window_last = _integer(event.get("last_frame", last_frame), f"{where}.last_frame")
    if not first_frame <= window_first <= window_last <= last_frame:
        raise GoldenError(f"{where} event window lies outside the replay window")
    for key in ("a", "b", "value"):
        if key in event:
            _integer(event[key], f"{where}.{key}")
    if "detail" in event:
        _string(event["detail"], f"{where}.detail", nonempty=False)
    return event


def _validate_scalar_transition(
    value: Any, where: str, *, first_frame: int, last_frame: int
) -> dict[str, Any]:
    transition = _object(value, where)
    _exact_keys(
        transition,
        where,
        required=("from_frame", "to_frame", "score", "gauge", "level"),
    )
    before_frame = _integer(transition["from_frame"], f"{where}.from_frame")
    after_frame = _integer(transition["to_frame"], f"{where}.to_frame")
    if before_frame != first_frame - 1 or after_frame != last_frame:
        raise GoldenError(
            f"{where} must span first_frame-1 through last_frame exactly"
        )
    for name in ("score", "gauge", "level"):
        scalar = _object(transition[name], f"{where}.{name}")
        _exact_keys(
            scalar,
            f"{where}.{name}",
            required=("before", "after", "delta"),
        )
        before = _integer(scalar["before"], f"{where}.{name}.before")
        after = _integer(scalar["after"], f"{where}.{name}.after")
        delta = _integer(scalar["delta"], f"{where}.{name}.delta")
        if after - before != delta:
            raise GoldenError(f"{where}.{name}.delta is inconsistent")
        if name == "level" and (before < 1 or after < 1):
            raise GoldenError(f"{where}.level values must be positive")
    return transition


def _validate_trajectory(
    value: Any, where: str, *, first_frame: int, last_frame: int
) -> dict[str, Any]:
    point = _object(value, where)
    _exact_keys(
        point,
        where,
        required=("frame", "body_id", "x", "y", "tolerance"),
    )
    frame = _integer(point["frame"], f"{where}.frame")
    if not first_frame <= frame <= last_frame:
        raise GoldenError(f"{where}.frame lies outside the replay window")
    horizon = frame - first_frame + 1
    if not MIN_TRAJECTORY_HORIZON_FRAMES <= horizon <= MAX_TRAJECTORY_HORIZON_FRAMES:
        raise GoldenError(
            f"{where}.frame must measure a {MIN_TRAJECTORY_HORIZON_FRAMES}.."
            f"{MAX_TRAJECTORY_HORIZON_FRAMES}-update horizon from first_frame"
        )
    _integer(point["body_id"], f"{where}.body_id", minimum=1)
    _number(point["x"], f"{where}.x")
    _number(point["y"], f"{where}.y")
    tolerance = _number(point["tolerance"], f"{where}.tolerance", minimum=0.0)
    if tolerance > MAX_TRAJECTORY_TOLERANCE:
        raise GoldenError(
            f"{where}.tolerance must not exceed {MAX_TRAJECTORY_TOLERANCE:g} pixels"
        )
    return point


def _validate_expected(
    value: Any,
    where: str,
    *,
    category: str,
    first_frame: int,
    last_frame: int,
) -> dict[str, Any]:
    expected = _object(value, where)
    _exact_keys(
        expected,
        where,
        required=("events", "scalar_transition"),
        optional=("trajectories",),
    )
    events = _array(expected["events"], f"{where}.events")
    if not events:
        raise GoldenError(f"{where}.events must contain a discrete assertion")
    validated_events = [
        _validate_event(
            event,
            f"{where}.events[{index}]",
            first_frame=first_frame,
            last_frame=last_frame,
        )
        for index, event in enumerate(events)
    ]
    relevant = [
        event
        for event in validated_events
        if event["kind"] in _RELEVANT_EVENTS[category]
    ]
    if category == "orb":
        relevant = [
            event
            for event in relevant
            if "special" in str(event.get("detail", "")).lower()
            or "orb" in str(event.get("detail", "")).lower()
        ]
    positive_relevant = [event for event in relevant if int(event["min_count"]) >= 1]
    if not positive_relevant:
        raise GoldenError(
            f"{where} has no positive category-relevant event assertion"
        )
    _validate_scalar_transition(
        expected["scalar_transition"],
        f"{where}.scalar_transition",
        first_frame=first_frame,
        last_frame=last_frame,
    )
    trajectories = _array(expected.get("trajectories", []), f"{where}.trajectories")
    if "trajectories" in expected and not trajectories:
        raise GoldenError(f"{where}.trajectories must not be empty when present")
    for index, point in enumerate(trajectories):
        _validate_trajectory(
            point,
            f"{where}.trajectories[{index}]",
            first_frame=first_frame,
            last_frame=last_frame,
        )
    return expected


def _measurement_entry(
    measurements: Mapping[str, Any], measurement_id: str, where: str
) -> dict[str, Any]:
    entries = _array(
        measurements.get("valid_mechanics_measurements"),
        f"{where}.valid_mechanics_measurements",
    )
    matches = [
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("id") == measurement_id
    ]
    if len(matches) != 1:
        raise GoldenError(
            f"{where} must contain exactly one valid measurement {measurement_id!r}"
        )
    return matches[0]


def _validate_scenario(
    value: Any,
    where: str,
    *,
    manifest_dir: Path,
    target: Mapping[str, Any],
) -> ValidatedScenario:
    scenario = _object(value, where)
    _exact_keys(
        scenario,
        where,
        required=("id", "category", "evidence", "replay", "expected"),
    )
    _identifier(scenario["id"], f"{where}.id")
    category = _string(scenario["category"], f"{where}.category")
    if category not in CATEGORIES:
        raise GoldenError(f"{where}.category must be one of {', '.join(CATEGORIES)}")

    evidence = _object(scenario["evidence"], f"{where}.evidence")
    _exact_keys(
        evidence,
        f"{where}.evidence",
        required=("status", "experiment_id", "measurement_id", "bundle", "files"),
    )
    if evidence["status"] != "observed":
        raise GoldenError(f"{where}.evidence.status must be 'observed'")
    experiment_id = _identifier(evidence["experiment_id"], f"{where}.evidence.experiment_id")
    measurement_id = _identifier(evidence["measurement_id"], f"{where}.evidence.measurement_id")
    bundle = _resolve_bundle(
        manifest_dir, evidence["bundle"], f"{where}.evidence.bundle"
    )

    files = _array(evidence["files"], f"{where}.evidence.files")
    if not files:
        raise GoldenError(f"{where}.evidence.files must not be empty")
    role_paths: dict[str, list[Path]] = {role: [] for role in _FILE_ROLES}
    expected_hashes: dict[Path, str] = {}
    captured_inputs: dict[Path, CapturedInput] = {}
    captured_bytes: dict[Path, bytes] = {}
    for index, raw_file in enumerate(files):
        file_where = f"{where}.evidence.files[{index}]"
        file_entry = _object(raw_file, file_where)
        _exact_keys(file_entry, file_where, required=("role", "path", "sha256"))
        role = _string(file_entry["role"], f"{file_where}.role")
        if role not in _FILE_ROLES:
            raise GoldenError(f"{file_where}.role is unknown: {role!r}")
        path = _resolved_child(bundle, file_entry["path"], f"{file_where}.path")
        expected_sha = _sha(file_entry["sha256"], f"{file_where}.sha256")
        if path in expected_hashes:
            raise GoldenError(f"{file_where}.path duplicates another evidence file")
        if not path.is_file():
            raise GoldenError(f"evidence file is missing: {path}")
        captured, data = _capture_input(path, keep_bytes=role != "video")
        if captured.sha256 != expected_sha:
            raise GoldenError(
                f"evidence hash mismatch for {path}: expected {expected_sha}, "
                f"got {captured.sha256}"
            )
        expected_hashes[path] = expected_sha
        captured_inputs[path] = captured
        if data is not None:
            captured_bytes[path] = data
        role_paths[role].append(path)

    for role in ("metadata", "actions", "measurements", "replay", "notes"):
        if len(role_paths[role]) != 1:
            raise GoldenError(f"{where} requires exactly one {role} evidence file")
    if not role_paths["frame"]:
        raise GoldenError(f"{where} requires at least one PNG frame")
    _validate_actions(
        captured_bytes[role_paths["actions"][0]], role_paths["actions"][0]
    )
    for frame_path in role_paths["frame"]:
        _validate_png(captured_bytes[frame_path], frame_path)
    try:
        notes_path = role_paths["notes"][0]
        if not captured_bytes[notes_path].decode("utf-8").strip():
            raise GoldenError(f"{where} notes evidence is empty")
    except UnicodeError as exc:
        raise GoldenError(f"cannot read notes evidence: {exc}") from exc

    metadata_path = role_paths["metadata"][0]
    metadata = _object(
        _read_json_bytes(captured_bytes[metadata_path], metadata_path),
        "capture metadata",
    )
    if metadata.get("status") != "valid_for_mechanics_calibration":
        raise GoldenError(
            f"{where} capture status is not valid_for_mechanics_calibration"
        )
    if metadata.get("experiment_id") != experiment_id:
        raise GoldenError(f"{where} experiment ID disagrees with capture metadata")
    game = _object(metadata.get("game"), "capture metadata.game")
    if game.get("version") != target["game_version"]:
        raise GoldenError(f"{where} game version disagrees with the manifest target")
    for key in _HASH_KEYS:
        if game.get(key) != target[key]:
            raise GoldenError(f"{where} target hash {key} disagrees with capture metadata")

    replay_spec = _object(scenario["replay"], f"{where}.replay")
    _exact_keys(
        replay_spec,
        f"{where}.replay",
        required=("path", "sha256", "layout", "first_frame", "last_frame"),
    )
    replay_path = _resolved_child(bundle, replay_spec["path"], f"{where}.replay.path")
    if replay_path != role_paths["replay"][0]:
        raise GoldenError(f"{where}.replay.path must name the replay evidence file")
    replay_sha = _sha(replay_spec["sha256"], f"{where}.replay.sha256")
    if expected_hashes[replay_path] != replay_sha:
        raise GoldenError(f"{where}.replay.sha256 disagrees with evidence.files")
    layout = _string(replay_spec["layout"], f"{where}.replay.layout")
    if layout not in ("legacy", "padded"):
        raise GoldenError(f"{where}.replay.layout must be explicit legacy or padded")
    first_frame = _integer(replay_spec["first_frame"], f"{where}.replay.first_frame", minimum=0)
    last_frame = _integer(replay_spec["last_frame"], f"{where}.replay.last_frame", minimum=0)
    if last_frame < first_frame:
        raise GoldenError(f"{where}.replay.last_frame precedes first_frame")
    try:
        parsed_replay = INSPECT_RPY.parse_replay(captured_bytes[replay_path], layout)
    except ValueError as exc:
        raise GoldenError(f"{where} replay is invalid: {exc}") from exc
    if parsed_replay.header.mode != 0:
        raise GoldenError(f"{where} replay is not normal mode")
    if last_frame >= len(parsed_replay.frames):
        raise GoldenError(f"{where}.replay.last_frame exceeds the replay")
    run = _object(metadata.get("run"), "capture metadata.run")
    if run.get("final_replay_sha256") != replay_sha:
        raise GoldenError(f"{where} replay hash disagrees with capture metadata.run")

    expected = _validate_expected(
        scenario["expected"],
        f"{where}.expected",
        category=category,
        first_frame=first_frame,
        last_frame=last_frame,
    )
    measurements_path = role_paths["measurements"][0]
    measurements = _object(
        _read_json_bytes(captured_bytes[measurements_path], measurements_path),
        "capture measurements",
    )
    measured = _measurement_entry(measurements, measurement_id, "capture measurements")
    _exact_keys(
        measured,
        f"measurement {measurement_id!r}",
        required=("id", "status", "category", "repeat_count", "replay_window", "oracle"),
        optional=(
            "annotator",
            "evidence_frames",
            "method",
            "notes",
            "uncertainty",
            "units",
        ),
    )
    if measured["status"] != "observed":
        raise GoldenError(f"{where} measurement status must be 'observed'")
    if measured["category"] != category:
        raise GoldenError(f"{where} measurement category disagrees with scenario")
    repeat_count = _integer(
        measured["repeat_count"], f"measurement {measurement_id!r}.repeat_count", minimum=1
    )
    if measured["replay_window"] != {
        "first_frame": first_frame,
        "last_frame": last_frame,
    }:
        raise GoldenError(f"{where} measurement replay window disagrees with scenario")
    if measured["oracle"] != expected:
        raise GoldenError(f"{where} oracle is not an exact copy of the observed measurement")
    return ValidatedScenario(
        scenario,
        parsed_replay,
        repeat_count,
        _source_probe_sha256(parsed_replay, last_frame),
        tuple(captured_inputs.values()),
    )


def validate_manifest(path: Path) -> ValidatedManifest:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise GoldenError(f"manifest is missing: {path}") from exc
    manifest_input, manifest_bytes = _capture_input(resolved, keep_bytes=True)
    assert manifest_bytes is not None
    manifest = _object(_read_json_bytes(manifest_bytes, resolved), "manifest")
    _exact_keys(
        manifest,
        "manifest",
        required=("schema_version", "threshold_percent", "target", "scenarios"),
    )
    if manifest["schema_version"] != 1:
        raise GoldenError("manifest.schema_version must be 1")
    if manifest["threshold_percent"] != THRESHOLD_PERCENT:
        raise GoldenError(f"manifest.threshold_percent must be {THRESHOLD_PERCENT}")
    target = _validate_target(manifest["target"])
    raw_scenarios = _array(manifest["scenarios"], "manifest.scenarios")
    if not raw_scenarios:
        raise GoldenError("manifest has no observed golden scenarios")
    scenarios = tuple(
        _validate_scenario(
            scenario,
            f"manifest.scenarios[{index}]",
            manifest_dir=resolved.parent,
            target=target,
        )
        for index, scenario in enumerate(raw_scenarios)
    )
    ids = [scenario.raw["id"] for scenario in scenarios]
    if len(ids) != len(set(ids)):
        raise GoldenError("manifest scenario IDs must be unique")
    measurement_identities = [
        (
            scenario.raw["evidence"]["experiment_id"],
            scenario.raw["evidence"]["measurement_id"],
        )
        for scenario in scenarios
    ]
    if len(measurement_identities) != len(set(measurement_identities)):
        raise GoldenError(
            "manifest evidence measurement identities must be unique"
        )
    prefix_hashes: dict[tuple[int, int], str] = {}

    def prefix_hash(index: int, scenario: ValidatedScenario, last_frame: int) -> str:
        key = (index, last_frame)
        if key not in prefix_hashes:
            prefix_hashes[key] = _source_probe_sha256(scenario.replay, last_frame)
        return prefix_hashes[key]

    for index, scenario in enumerate(scenarios):
        replay = scenario.raw["replay"]
        first_frame = replay["first_frame"]
        last_frame = replay["last_frame"]
        for existing_index, existing in enumerate(scenarios[:index]):
            existing_replay = existing.raw["replay"]
            existing_first = existing_replay["first_frame"]
            existing_last = existing_replay["last_frame"]
            if max(first_frame, existing_first) > min(last_frame, existing_last):
                continue
            common_last = min(last_frame, existing_last)
            current_prefix = prefix_hash(index, scenario, common_last)
            existing_prefix = prefix_hash(existing_index, existing, common_last)
            if current_prefix == existing_prefix:
                raise GoldenError(
                    "manifest source probe replay windows must not overlap for canonical "
                    f"behavior: {existing.raw['id']!r} and {scenario.raw['id']!r}"
                )

    inputs_by_path = {manifest_input.path: manifest_input}
    for scenario in scenarios:
        for captured in scenario.inputs:
            previous = inputs_by_path.setdefault(captured.path, captured)
            if previous != captured:
                raise GoldenError(
                    f"validated input changed during manifest validation: {captured.path}"
                )
    return ValidatedManifest(
        resolved,
        manifest,
        scenarios,
        manifest_input,
        tuple(inputs_by_path.values()),
    )


def _plain_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "tick": int(observation["tick"]),
        "score": int(observation["score"]),
        "gauge": int(observation["gauge"]),
        "level": int(observation["level"]),
        "terminated": bool(observation["terminated"]),
        "truncated": bool(observation["truncated"]),
        "bodies": [dict(body) for body in observation["bodies"]],
    }


def _event_matches(event: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if event.get("kind_name") != expected["kind"]:
        return False
    for key in ("a", "b", "value", "detail"):
        if key in expected and event.get(key) != expected[key]:
            return False
    return True


def _score_events(
    expected_events: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    assertions: list[dict[str, Any]] = []
    for expected in expected_events:
        first_frame = int(expected.get("first_frame", -1))
        last_frame = int(expected.get("last_frame", 2**63 - 1))
        matched = [
            event
            for event in events
            if first_frame <= int(event["frame"]) <= last_frame
            and _event_matches(event, expected)
        ]
        count = len(matched)
        passed = int(expected["min_count"]) <= count <= int(expected["max_count"])
        assertions.append(
            {
                "expected": dict(expected),
                "actual_count": count,
                "matched_frames": [int(event["frame"]) for event in matched],
                "passed": passed,
            }
        )
    return {"passed": all(item["passed"] for item in assertions), "assertions": assertions}


def _score_scalars(
    expected: Mapping[str, Any], observations: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    before_frame = int(expected["from_frame"])
    after_frame = int(expected["to_frame"])
    before = observations[before_frame]
    after = observations[after_frame]
    fields: dict[str, dict[str, Any]] = {}
    for name in ("score", "gauge", "level"):
        values = {
            "before": int(before[name]),
            "after": int(after[name]),
            "delta": int(after[name]) - int(before[name]),
        }
        fields[name] = {
            "expected": dict(expected[name]),
            "actual": values,
            "passed": values == expected[name],
        }
    return {
        "passed": all(item["passed"] for item in fields.values()),
        "from_frame": before_frame,
        "to_frame": after_frame,
        "fields": fields,
    }


def _score_trajectories(
    expected: Sequence[Mapping[str, Any]], observations: Mapping[int, Mapping[str, Any]]
) -> dict[str, Any]:
    assertions: list[dict[str, Any]] = []
    for point in expected:
        frame = int(point["frame"])
        body_id = int(point["body_id"])
        bodies = [body for body in observations[frame]["bodies"] if int(body["id"]) == body_id]
        if len(bodies) != 1:
            assertions.append(
                {
                    "expected": dict(point),
                    "actual": None,
                    "distance": None,
                    "passed": False,
                    "reason": "body is not uniquely visible",
                }
            )
            continue
        body = bodies[0]
        actual_x = float(body["x"])
        actual_y = float(body["y"])
        distance = math.hypot(actual_x - float(point["x"]), actual_y - float(point["y"]))
        passed = distance <= float(point["tolerance"])
        assertions.append(
            {
                "expected": dict(point),
                "actual": {"x": actual_x, "y": actual_y},
                "distance": distance,
                "passed": passed,
            }
        )
    return {
        "status": "evaluated" if assertions else "not_requested",
        "passed": all(item["passed"] for item in assertions),
        "assertions": assertions,
    }


def _score_scenario(
    scenario: ValidatedScenario,
    *,
    library_path: str | None,
    worker_path: str | None,
    env_factory: Callable[..., Any],
) -> tuple[dict[str, Any], dict[str, Any], int, dict[str, Any]]:
    raw = scenario.raw
    replay_spec = raw["replay"]
    first_frame = int(replay_spec["first_frame"])
    last_frame = int(replay_spec["last_frame"])
    mapped = EVALUATE_RPY.map_frames(
        scenario.replay.frames[: last_frame + 1], support_both_shots=True
    )
    expected = raw["expected"]
    needed_frames = {
        int(expected["scalar_transition"]["from_frame"]),
        int(expected["scalar_transition"]["to_frame"]),
        *(int(point["frame"]) for point in expected.get("trajectories", [])),
    }
    observations: dict[int, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    invalid_action_frames: list[int] = []
    if library_path is not None and worker_path is not None:
        raise GoldenError("library_path and worker_path are mutually exclusive")
    if worker_path is not None:
        kwargs = {"physics_backend": "exact", "worker_path": worker_path}
    elif library_path is not None:
        kwargs = {"library_path": library_path}
    else:
        kwargs = {}
    with env_factory(**kwargs) as env:
        if isinstance(env, IrisuEnv):
            expected_backend = "exact" if worker_path is not None else "portable"
            if env.physics_backend != expected_backend:
                raise GoldenError(
                    "IrisuEnv physics backend disagrees with the scorer request"
                )
        library_before = _capture_clone_library(env)
        exact_library_before = (
            _capture_exact_library(env) if worker_path is not None else None
        )
        seed = int(scenario.replay.header.seed) & 0xFFFFFFFF
        initial, _ = env.reset(seed=seed)
        if -1 in needed_frames:
            observations[-1] = _plain_observation(initial)
        config_hash = int(env.config_hash())
        build_info = dict(env.build_info)
        last_observation = initial
        for frame in mapped:
            observation, _, _, _, info = env.step(EVALUATE_RPY._env_action(frame))
            last_observation = observation
            if frame.index in needed_frames:
                observations[frame.index] = _plain_observation(observation)
            if first_frame <= frame.index <= last_frame:
                for event in info.get("events", ()):
                    record = dict(event)
                    record["frame"] = frame.index
                    events.append(record)
            if info.get("invalid_action"):
                invalid_action_frames.append(frame.index)
        final_tick = int(last_observation["tick"])
        library_after = _capture_clone_library(env)
        clone_artifact = _verify_clone_library_stability(
            library_before, library_after
        )
        exact_library_after = (
            _capture_exact_library(env) if worker_path is not None else None
        )

    runtime: dict[str, Any] = {}
    if worker_path is not None:
        worker_sha = build_info.get("worker_executable_sha256")
        exact_library_sha = normalize_exact_library_sha256(
            build_info.get("exact_library_sha256")
        )
        worker_pid = build_info.get("worker_pid")
        if (
            build_info.get("physics_backend") != "exact-msvc9-r58-worker"
            or build_info.get("worker_backend")
            != "exact-msvc9-r58-multiworld-forward"
            or not isinstance(worker_sha, str)
            or _SHA256.fullmatch(worker_sha) is None
            or exact_library_sha is None
            or type(worker_pid) is not int
            or worker_pid <= 0
        ):
            raise GoldenError("exact worker reported incomplete backend provenance")
        if clone_artifact.get("sha256") != worker_sha:
            raise GoldenError(
                "loaded exact worker bytes disagree with worker-reported provenance"
            )
        assert exact_library_before is not None and exact_library_after is not None
        exact_library_artifact = _verify_exact_library_stability(
            exact_library_before, exact_library_after
        )
        if exact_library_artifact["sha256"] != exact_library_sha:
            raise GoldenError(
                "mapped exact library bytes disagree with worker-reported provenance"
            )
        runtime["worker_pid"] = worker_pid
        build_info.pop("worker_pid", None)
        build_info["exact_library_sha256"] = exact_library_sha
        clone_artifact = {
            **clone_artifact,
            "artifact_role": "exact_worker_executable",
            "linked_exact_library_sha256": exact_library_sha,
            "linked_exact_library": exact_library_artifact,
        }

    missing_frames = sorted(needed_frames - observations.keys())
    if missing_frames:
        raise GoldenError(
            f"scenario {raw['id']!r} did not produce required frames {missing_frames}"
        )
    execution = {
        "passed": final_tick == last_frame + 1 and not invalid_action_frames,
        "expected_final_tick": last_frame + 1,
        "actual_final_tick": final_tick,
        "invalid_action_frames": invalid_action_frames,
    }
    discrete = _score_events(expected["events"], events)
    scalars = _score_scalars(expected["scalar_transition"], observations)
    trajectories = _score_trajectories(expected.get("trajectories", []), observations)
    passed = (
        execution["passed"]
        and discrete["passed"]
        and scalars["passed"]
        and trajectories["passed"]
    )
    report = {
        "id": raw["id"],
        "category": raw["category"],
        "evidence": {
            "experiment_id": raw["evidence"]["experiment_id"],
            "measurement_id": raw["evidence"]["measurement_id"],
            "source_probe_sha256": scenario.source_probe_sha256,
            "reported_repeat_count": scenario.reported_repeat_count,
            "repeat_count_status": "advisory_not_independently_verified",
            "repeat_count_used_for_gate": False,
        },
        "execution": execution,
        "discrete": discrete,
        "scalars": scalars,
        "trajectories": trajectories,
        "passed": passed,
    }
    if runtime:
        report["runtime"] = runtime
    return report, build_info, config_hash, clone_artifact


def _threshold_met(passed: int, total: int) -> bool:
    return total > 0 and passed * 100 >= total * THRESHOLD_PERCENT


def score_manifest(
    path: Path,
    *,
    library_path: str | None = None,
    worker_path: str | None = None,
    env_factory: Callable[..., Any] = IrisuEnv,
) -> dict[str, Any]:
    if library_path is not None and worker_path is not None:
        raise GoldenError("library_path and worker_path are mutually exclusive")
    manifest = validate_manifest(path)
    scenario_reports: list[dict[str, Any]] = []
    build_info: dict[str, Any] | None = None
    actual_config: int | None = None
    clone_artifact: dict[str, Any] | None = None
    for scenario in manifest.scenarios:
        report, scenario_build, scenario_config, scenario_artifact = _score_scenario(
            scenario,
            library_path=library_path,
            worker_path=worker_path,
            env_factory=env_factory,
        )
        if build_info is None:
            build_info = scenario_build
            actual_config = scenario_config
            clone_artifact = scenario_artifact
        elif scenario_build != build_info or scenario_config != actual_config:
            raise GoldenError("clone build or configuration changed during scoring")
        elif scenario_artifact != clone_artifact:
            raise GoldenError("loaded clone artifact changed between scenarios")
        scenario_reports.append(report)

    _verify_inputs_unchanged(manifest.inputs)

    assert (
        build_info is not None
        and actual_config is not None
        and clone_artifact is not None
    )
    expected_config = int(manifest.raw["target"]["clone_config_u64"], 16)
    if actual_config != expected_config:
        raise GoldenError(
            "clone configuration hash mismatch: "
            f"expected 0x{expected_config:016x}, got 0x{actual_config:016x}"
        )

    categories: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        selected = [report for report in scenario_reports if report["category"] == category]
        passed = sum(bool(report["passed"]) for report in selected)
        total = len(selected)
        categories[category] = {
            "passed": passed,
            "total": total,
            "rate_percent": None if total == 0 else (passed * 100.0 / total),
            "threshold_met": _threshold_met(passed, total),
        }
    passed_total = sum(bool(report["passed"]) for report in scenario_reports)
    total = len(scenario_reports)
    coverage_complete = all(categories[name]["total"] > 0 for name in CATEGORIES)
    category_thresholds_met = all(
        categories[name]["threshold_met"] for name in CATEGORIES
    )
    overall_threshold_met = _threshold_met(passed_total, total)
    exact_scalars = all(report["scalars"]["passed"] for report in scenario_reports)
    trajectory_coverage = {
        category: sum(
            len(report["trajectories"]["assertions"])
            for report in scenario_reports
            if report["category"] == category
        )
        for category in CATEGORIES
    }
    trajectory_coverage_complete = all(
        trajectory_coverage[category] > 0 for category in CATEGORIES
    )
    trajectories = all(report["trajectories"]["passed"] for report in scenario_reports)
    gate_passed = (
        coverage_complete
        and category_thresholds_met
        and overall_threshold_met
        and exact_scalars
        and trajectory_coverage_complete
        and trajectories
    )
    report = {
        "schema_version": 1,
        "status": "pass" if gate_passed else "fail",
        "purpose": "strict original-game controlled-scenario fidelity subgate",
        "manifest": {
            "path": str(manifest.path),
            "sha256": manifest.manifest_input.sha256,
            "target": manifest.raw["target"],
        },
        "clone_build": build_info,
        "clone_config_u64": f"0x{actual_config:016x}",
        "scenarios": scenario_reports,
        "gate": {
            "threshold_percent": THRESHOLD_PERCENT,
            "calculation": "passed*100 >= total*95 (integer, no rounding)",
            "one_vote_per_scenario": True,
            "one_vote_per_unique_source_measurement": True,
            "one_vote_per_nonoverlapping_source_probe": True,
            "coverage_complete": coverage_complete,
            "category_thresholds_met": category_thresholds_met,
            "overall": {
                "passed": passed_total,
                "total": total,
                "rate_percent": passed_total * 100.0 / total,
                "threshold_met": overall_threshold_met,
            },
            "categories": categories,
            "all_scalar_transitions_exact": exact_scalars,
            "trajectory_assertions_by_category": trajectory_coverage,
            "trajectory_coverage_complete": trajectory_coverage_complete,
            "all_trajectories_within_tolerance": trajectories,
            "passed": gate_passed,
        },
        "scope": {
            "controlled_scenario_subgate_met": gate_passed,
            "full_clone_md_fidelity_gate_met": False,
            "spawn_difficulty_distribution_status": "not_evaluated",
            "policy_transfer_status": "not_evaluated",
        },
    }
    report["clone_worker" if worker_path is not None else "clone_library"] = (
        clone_artifact
    )
    return report


def report_exit_code(report: Mapping[str, Any]) -> int:
    return {"pass": 0, "fail": 1, "not_evaluable": 2}.get(str(report.get("status")), 2)


def main(argv: Sequence[str] | None = None, *, stdout: TextIO = sys.stdout) -> int:
    parser = argparse.ArgumentParser(
        description="score strict original-game-derived golden scenarios"
    )
    parser.add_argument("manifest", type=Path)
    backend = parser.add_mutually_exclusive_group()
    backend.add_argument(
        "--library", help="explicit portable libirisu_clone shared-library path"
    )
    backend.add_argument(
        "--worker", help="explicit exact-MSVC9 irisu-exact-worker path"
    )
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args(argv)
    try:
        report = score_manifest(
            args.manifest,
            library_path=args.library,
            worker_path=args.worker,
        )
    except (GoldenError, NativeError, OSError, RuntimeError, ValueError) as exc:
        report = {
            "schema_version": 1,
            "status": "not_evaluable",
            "purpose": "strict original-game controlled-scenario fidelity subgate",
            "scope": {
                "full_clone_md_fidelity_gate_met": False,
                "spawn_difficulty_distribution_status": "not_evaluated",
                "policy_transfer_status": "not_evaluated",
            },
            "error": str(exc),
        }
    print(
        json.dumps(report, indent=None if args.compact else 2, sort_keys=True),
        file=stdout,
    )
    return report_exit_code(report)


if __name__ == "__main__":
    raise SystemExit(main())
