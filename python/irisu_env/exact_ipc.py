"""Exact 32-bit MSVC9 Box2D backend hosted in a worker process."""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import socket
import struct
import subprocess
import threading
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, replace
from numbers import Integral, Real
from pathlib import Path
from typing import Any, BinaryIO

from .native import (
    EVENT_DETAIL_CAPACITY,
    LibraryNotFoundError,
    NativeError,
    _flatten_config,
)


_MAGIC = 0x43505249
_PROTOCOL_VERSION = 1
_BODY_CAPACITY = 196
_EXACT_BACKEND = "exact-msvc9-r58-multiworld-forward"
_EXACT_LIBRARY_SONAME = "libirisu_box2d_msvc_exact_multiworld.so"

_HELLO = 1
_RESET = 2
_STEP = 3
_OBSERVE = 4
_CLOSE = 5
_CONFIGURE = 6
_CONFIG_JSON = 7
_STEP_PADDED = 8
_FETCH_EVENTS = 9
_FAST_CHECKPOINT = 10
_FAST_RELEASE = 11
_FAST_BRANCH = 12
_EXACT_ATTESTATION = 13

_HEADER = struct.Struct("<IHHII")
_STATUS = struct.Struct("<i")
_HELLO_FIXED = struct.Struct("<IIIIQII")
_RESET_REQUEST = struct.Struct("<Q")
_STEP_REQUEST = struct.Struct("<IddII")
_OBSERVATION_HEADER = struct.Struct("<QqqqQddddddIIIII4B")
_BODY = struct.Struct("<QqQdddddddIiII4B")
_TRANSITION = struct.Struct("<qQQQqQqQIIIIBBBB")
_EVENT = struct.Struct("<QQqIIHBB")
_CONFIG_COUNT = struct.Struct("<I")
_CONFIG_KEY_SIZE = struct.Struct("<H")
_CONFIG_VALUE = struct.Struct("<d")
_CONFIG_HASH = struct.Struct("<Q")
_EVENT_GENERATION = struct.Struct("<Q")
_EVENT_COUNT = struct.Struct("<Q")
_FAST_CHECKPOINT_RESPONSE = struct.Struct("<16sI")
_FAST_BRANCH_RESPONSE = struct.Struct("<IH16s")
_EXACT_ATTESTATION_FIXED = struct.Struct("<IIQ")
_FAST_TOKEN_BYTES = 16
_MAXIMUM_BRANCH_ADDRESS_BYTES = 108
_EXACT_ATTESTATION_SCHEMA = 1
_EXACT_ENTRYPOINT_COUNT = 15

_BODY_KIND_NAMES = ("piece", "projectile", "bonus")
_SHAPE_NAMES = ("circle", "box", "triangle")
_LIFECYCLE_NAMES = (
    "scripted_falling",
    "dynamic_fresh",
    "confirmed",
    "rotten",
    "deleted",
)

_SNAPSHOT_MAGIC = 0x49524953
_SNAPSHOT_SCHEMA = 0x45580001
_SNAPSHOT_PREFIX = struct.Struct("<IIQIIQQ32s")
_SNAPSHOT_CHECKSUM_SIZE = 32
_STATE_PERSON = b"irisu-xstate-v1"


class ExactProtocolError(NativeError):
    """The exact worker emitted a malformed or mismatched response."""


class ExactWorkerError(NativeError):
    """The exact worker rejected an operation or exited unexpectedly."""


class ExactWorkerNotFoundError(LibraryNotFoundError):
    """No usable exact worker executable could be located."""


def normalize_exact_library_sha256(value: object) -> str | None:
    """Return one canonical non-placeholder exact-library digest."""

    if not isinstance(value, str) or len(value) != 64:
        return None
    normalized = value.lower()
    if normalized == "0" * 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        return None
    return normalized


def _valid_exact_library_sha256(value: object) -> bool:
    return normalize_exact_library_sha256(value) is not None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _capture_executed_worker(
    pid: int,
) -> tuple[str, tuple[int, int, int, int, int]]:
    path = Path(f"/proc/{pid}/exe")
    try:
        before = path.stat()
        digest = _sha256_file(path)
        after = path.stat()
    except OSError as exc:
        raise ExactProtocolError(
            f"cannot capture executed exact worker {path}: {exc}"
        ) from exc
    identity = _file_identity(after)
    if _file_identity(before) != identity:
        raise ExactProtocolError("executed exact worker changed while it was captured")
    return digest, identity


def _proc_path(value: str) -> str:
    for encoded, decoded in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return value


def _mount_device(pid: int, path_text: str) -> str:
    mountinfo_path = Path(f"/proc/{pid}/mountinfo")
    try:
        lines = mountinfo_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExactProtocolError(
            f"cannot inspect exact worker mount identity: {exc}"
        ) from exc
    matches: list[tuple[int, str]] = []
    for line in lines:
        fields = line.split()
        if len(fields) < 6:
            continue
        device = fields[2]
        mountpoint = _proc_path(fields[4])
        if path_text == mountpoint or path_text.startswith(
            mountpoint.rstrip("/") + "/"
        ):
            try:
                major_text, minor_text = device.split(":", 1)
                maps_device = f"{int(major_text):02x}:{int(minor_text):02x}"
            except ValueError as exc:
                raise ExactProtocolError(
                    "exact worker mountinfo contains an invalid device"
                ) from exc
            matches.append((len(mountpoint), maps_device))
    if not matches:
        raise ExactProtocolError("mapped exact library has no worker mount identity")
    return max(matches)[1]


def _mapped_exact_library(pid: int) -> tuple[str, str, int, tuple[tuple[str, str], ...]]:
    maps_path = Path(f"/proc/{pid}/maps")
    try:
        lines = maps_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExactProtocolError(
            f"cannot inspect exact worker mapped libraries: {exc}"
        ) from exc

    candidates: dict[tuple[str, str, int], list[tuple[str, str]]] = {}
    for line in lines:
        fields = line.split(None, 5)
        if len(fields) != 6:
            continue
        address, permissions, offset, device, inode_text, raw_path = fields
        deleted = raw_path.endswith(" (deleted)")
        path_text = _proc_path(raw_path[:-10] if deleted else raw_path)
        if Path(path_text).name != _EXACT_LIBRARY_SONAME:
            continue
        try:
            inode = int(inode_text)
        except ValueError as exc:
            raise ExactProtocolError(
                "exact worker maps contain an invalid library inode"
            ) from exc
        if deleted:
            raise ExactProtocolError("mapped exact library was deleted")
        key = (path_text, device, inode)
        candidates.setdefault(key, []).append((permissions, offset))

    if len(candidates) != 1:
        raise ExactProtocolError(
            "exact worker must map exactly one identifiable exact physics library"
        )
    (path_text, device, inode), segments = next(iter(candidates.items()))
    if not Path(path_text).is_absolute() or inode <= 0:
        raise ExactProtocolError("mapped exact library has an invalid identity")
    if device.lower() != _mount_device(pid, path_text):
        raise ExactProtocolError(
            "mapped exact library device disagrees with the worker mount"
        )
    if not any(offset == "00000000" for _, offset in segments) or not any(
        "x" in permissions for permissions, _ in segments
    ):
        raise ExactProtocolError("mapped exact library has incomplete ELF mappings")
    return path_text, device, inode, tuple(sorted(segments))


def _capture_mapped_exact_library(
    pid: int, reported_sha256: object
) -> dict[str, Any]:
    expected_sha256 = normalize_exact_library_sha256(reported_sha256)
    if expected_sha256 is None:
        raise ExactProtocolError(
            "exact worker did not report a valid non-placeholder exact-library SHA-256"
        )

    mapped_before = _mapped_exact_library(pid)
    path_text, mapped_device, mapped_inode, _ = mapped_before
    path = Path(path_text)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if before.st_ino != mapped_inode:
                raise ExactProtocolError(
                    "mapped exact library path no longer names the loaded inode"
                )
            while block := stream.read(1 << 20):
                byte_count += len(block)
                digest.update(block)
            after = os.fstat(stream.fileno())
        current = path.stat()
        resolved = path.resolve(strict=True)
    except ExactProtocolError:
        raise
    except OSError as exc:
        raise ExactProtocolError(
            f"cannot capture mapped exact library {path}: {exc}"
        ) from exc

    mapped_after = _mapped_exact_library(pid)
    parent_device = _mount_device(os.getpid(), str(resolved))
    if parent_device != mapped_device.lower():
        raise ExactProtocolError(
            "mapped exact library device disagrees with the client mount"
        )
    identity = _file_identity(after)
    if (
        mapped_after != mapped_before
        or _file_identity(before) != identity
        or identity != _file_identity(current)
        or byte_count != after.st_size
    ):
        raise ExactProtocolError("mapped exact library changed while it was captured")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise ExactProtocolError(
            "mapped exact library bytes disagree with worker-reported provenance"
        )
    return {
        "status": "captured",
        "path": str(resolved),
        "bytes": byte_count,
        "sha256": actual_sha256,
        "file_identity": {
            "device": after.st_dev,
            "inode": after.st_ino,
            "mtime_ns": after.st_mtime_ns,
            "ctime_ns": after.st_ctime_ns,
        },
        "mapped_identity": {
            "device": mapped_device,
            "inode": mapped_inode,
        },
    }


_PEER_CREDENTIALS = struct.Struct("3i")


def _captured_library_file(
    provenance: Mapping[str, Any],
) -> tuple[Path, tuple[int, int, int, int, int]]:
    try:
        file_identity = provenance["file_identity"]
        if not isinstance(file_identity, Mapping):
            raise TypeError
        path = Path(provenance["path"])
        identity = (
            file_identity["device"],
            file_identity["inode"],
            provenance["bytes"],
            file_identity["mtime_ns"],
            file_identity["ctime_ns"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ExactProtocolError(
            "inherited exact-library provenance is incomplete"
        ) from exc
    if (
        provenance.get("status") != "captured"
        or not path.is_absolute()
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in identity
        )
        or identity[1] <= 0
        or identity[2] < 0
    ):
        raise ExactProtocolError("inherited exact-library provenance is invalid")
    return path, identity


def _process_parent_pid(pid: int) -> int:
    path = Path(f"/proc/{pid}/stat")
    try:
        contents = path.read_text(encoding="utf-8")
        suffix = contents[contents.rindex(")") + 1 :].split()
        parent = int(suffix[1])
    except (OSError, ValueError, IndexError) as exc:
        raise ExactProtocolError(
            f"cannot inspect exact fast-branch ancestry {path}: {exc}"
        ) from exc
    if parent <= 0:
        raise ExactProtocolError("exact fast branch has an invalid parent PID")
    return parent


def _verify_keeper_peer(connection: socket.socket, expected_pid: int) -> None:
    try:
        credentials = connection.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, _PEER_CREDENTIALS.size
        )
    except (AttributeError, OSError) as exc:
        raise ExactProtocolError(
            f"cannot authenticate exact checkpoint keeper: {exc}"
        ) from exc
    if len(credentials) != _PEER_CREDENTIALS.size:
        raise ExactProtocolError("exact checkpoint keeper credentials are truncated")
    peer = _PEER_CREDENTIALS.unpack(credentials)
    expected = (expected_pid, os.geteuid(), os.getegid())
    if peer != expected:
        raise ExactProtocolError("exact checkpoint keeper identity changed")


def _verify_inherited_branch_files(
    pid: int,
    keeper_pid: int,
    executable_identity: tuple[int, int, int, int, int],
    library_provenance: Mapping[str, Any],
) -> None:
    if _process_parent_pid(pid) != keeper_pid:
        raise ExactProtocolError("exact fast branch is not a direct keeper child")
    try:
        current_executable = _file_identity(Path(f"/proc/{pid}/exe").stat())
    except OSError as exc:
        raise ExactProtocolError(
            f"cannot inspect exact fast-branch executable: {exc}"
        ) from exc
    if current_executable != executable_identity:
        raise ExactProtocolError("exact fast-branch executable identity changed")
    library_path, library_identity = _captured_library_file(library_provenance)
    try:
        current_library = _file_identity(library_path.stat())
    except OSError as exc:
        raise ExactProtocolError(
            f"cannot inspect inherited exact physics library: {exc}"
        ) from exc
    if current_library != library_identity:
        raise ExactProtocolError("inherited exact physics library identity changed")


@dataclass(frozen=True, slots=True)
class ExactWorkerInfo:
    protocol_version: int
    pointer_bits: int
    body_capacity: int
    pid: int
    config_hash: int
    x87_control_word: int
    process_model: int
    backend: str
    compiler: str
    exact_library_sha256: str


@dataclass(frozen=True, slots=True)
class ExactBodyState:
    age_ticks: int
    remaining_lifetime: int
    rot_timer: int
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    size: float
    id: int
    color: int
    chain_id: int
    projectile_hits: int
    kind: int
    shape: int
    lifecycle: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": _BODY_KIND_NAMES[self.kind],
            "shape": _SHAPE_NAMES[self.shape],
            "lifecycle": _LIFECYCLE_NAMES[self.lifecycle],
            "color": self.color,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "angle": self.angle,
            "angular_velocity": self.angular_velocity,
            "size": self.size,
            "chain_id": self.chain_id,
            "projectile_hits": self.projectile_hits,
            "age_ticks": self.age_ticks,
            "remaining_lifetime": self.remaining_lifetime,
            "rot_timer": self.rot_timer,
        }


@dataclass(frozen=True, slots=True)
class ExactObservation:
    tick: int
    score: int
    gauge: int
    gauge_max: int
    qualifying_clear_count: int
    field_x: float
    field_y: float
    field_width: float
    field_height: float
    side_wall_top: float
    side_wall_bottom: float
    level: int
    active_colors: int
    spawn_interval_ticks: int
    highest_chain: int
    terminated: bool
    truncated: bool
    left_held: bool
    right_held: bool
    bodies: tuple[ExactBodyState, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "score": self.score,
            "gauge": self.gauge,
            "level": self.level,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "left_held": self.left_held,
            "right_held": self.right_held,
            "highest_chain": self.highest_chain,
            "qualifying_clear_count": self.qualifying_clear_count,
            "field": {
                "x": self.field_x,
                "y": self.field_y,
                "width": self.field_width,
                "height": self.field_height,
                "side_wall_top": self.side_wall_top,
                "side_wall_bottom": self.side_wall_bottom,
            },
            "gauge_max": self.gauge_max,
            "difficulty": {
                "active_colors": self.active_colors,
                "spawn_interval_ticks": self.spawn_interval_ticks,
            },
            "bodies": [body.to_dict() for body in self.bodies],
        }


@dataclass(frozen=True, slots=True)
class ExactEventState:
    tick: int
    sequence: int
    value: int
    a: int
    b: int
    kind: int
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "sequence": self.sequence,
            "kind": self.kind,
            "a": self.a,
            "b": self.b,
            "value": self.value,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class ExactTransition:
    observation: ExactObservation
    reward: int
    event_count: int
    config_hash: int
    finish_call_count: int
    recorded_final_score: int
    recorded_final_clears: int
    latest_final_score: int
    latest_final_clears: int
    recorded_final_highest_chain: int
    recorded_final_level: int
    latest_final_highest_chain: int
    latest_final_level: int
    terminated: bool
    truncated: bool
    terminal_metadata_recorded: bool
    invalid_action: bool
    events: tuple[ExactEventState, ...]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "config_hash": self.config_hash,
            "finish_call_count": self.finish_call_count,
            "terminal_metadata_recorded": self.terminal_metadata_recorded,
            "recorded_final_score": self.recorded_final_score,
            "recorded_final_highest_chain": self.recorded_final_highest_chain,
            "recorded_final_level": self.recorded_final_level,
            "recorded_final_clears": self.recorded_final_clears,
            "latest_final_score": self.latest_final_score,
            "latest_final_highest_chain": self.latest_final_highest_chain,
            "latest_final_level": self.latest_final_level,
            "latest_final_clears": self.latest_final_clears,
        }

    def to_step_dict(self) -> dict[str, Any]:
        return {
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "events": [event.to_dict() for event in self.events],
            "diagnostics": self.diagnostics(),
        }


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"worker response ended with {remaining} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(stream: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    while view:
        count = stream.write(view)
        if count is None or count <= 0:
            raise BrokenPipeError("worker request stream accepted no bytes")
        view = view[count:]
    stream.flush()


def _decode_exact_attestation(payload: bytes) -> tuple[int, str, int]:
    if len(payload) < _EXACT_ATTESTATION_FIXED.size + 2:
        raise ExactProtocolError("exact call-target attestation is truncated")
    schema, entrypoint_count, inode = _EXACT_ATTESTATION_FIXED.unpack_from(payload)
    offset = _EXACT_ATTESTATION_FIXED.size
    device_size = struct.unpack_from("<H", payload, offset)[0]
    offset += 2
    if len(payload) - offset != device_size:
        raise ExactProtocolError("exact call-target attestation has an invalid size")
    try:
        device = payload[offset:].decode("ascii").lower()
    except UnicodeDecodeError as exc:
        raise ExactProtocolError(
            "exact call-target attestation device is not ASCII"
        ) from exc
    device_parts = device.split(":")
    if (
        schema != _EXACT_ATTESTATION_SCHEMA
        or entrypoint_count != _EXACT_ENTRYPOINT_COUNT
        or inode <= 0
        or len(device_parts) != 2
        or any(
            not part or any(character not in "0123456789abcdef" for character in part)
            for part in device_parts
        )
    ):
        raise ExactProtocolError("exact call-target attestation is invalid")
    return entrypoint_count, device, inode


def _decode_observation(payload: bytes, offset: int = 0) -> tuple[ExactObservation, int]:
    if len(payload) - offset < _OBSERVATION_HEADER.size:
        raise ExactProtocolError("truncated exact observation header")
    values = _OBSERVATION_HEADER.unpack_from(payload, offset)
    offset += _OBSERVATION_HEADER.size
    body_count = values[15]
    if body_count > _BODY_CAPACITY:
        raise ExactProtocolError(
            f"exact observation body count {body_count} exceeds capacity"
        )
    if len(payload) - offset < body_count * _BODY.size:
        raise ExactProtocolError("truncated exact observation bodies")
    bodies: list[ExactBodyState] = []
    for _ in range(body_count):
        body = _BODY.unpack_from(payload, offset)
        offset += _BODY.size
        if body[14] > 2 or body[15] > 2 or body[16] > 4 or body[17] != 0:
            raise ExactProtocolError("exact observation contains invalid body metadata")
        bodies.append(ExactBodyState(*body[:17]))
    return (
        ExactObservation(
            tick=values[0],
            score=values[1],
            gauge=values[2],
            gauge_max=values[3],
            qualifying_clear_count=values[4],
            field_x=values[5],
            field_y=values[6],
            field_width=values[7],
            field_height=values[8],
            side_wall_top=values[9],
            side_wall_bottom=values[10],
            level=values[11],
            active_colors=values[12],
            spawn_interval_ticks=values[13],
            highest_chain=values[14],
            terminated=bool(values[16]),
            truncated=bool(values[17]),
            left_held=bool(values[18]),
            right_held=bool(values[19]),
            bodies=tuple(bodies),
        ),
        offset,
    )


def _decode_transition(payload: bytes) -> ExactTransition:
    observation, offset = _decode_observation(payload)
    if len(payload) - offset < _TRANSITION.size:
        raise ExactProtocolError("exact transition has a truncated diagnostics suffix")
    values = _TRANSITION.unpack_from(payload, offset)
    offset += _TRANSITION.size
    if bool(values[12]) != observation.terminated:
        raise ExactProtocolError("exact transition termination flags disagree")
    if bool(values[13]) != observation.truncated:
        raise ExactProtocolError("exact transition truncation flags disagree")
    events: list[ExactEventState] = []
    for _ in range(values[1]):
        if len(payload) - offset < _EVENT.size:
            raise ExactProtocolError("truncated exact event header")
        event = _EVENT.unpack_from(payload, offset)
        offset += _EVENT.size
        detail_size, kind, reserved = event[5:]
        if kind > 18 or reserved:
            raise ExactProtocolError("exact event contains invalid metadata")
        if len(payload) - offset < detail_size:
            raise ExactProtocolError("truncated exact event detail")
        try:
            detail = payload[offset : offset + detail_size].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExactProtocolError("exact event detail is not UTF-8") from exc
        offset += detail_size
        events.append(
            ExactEventState(
                tick=event[0],
                sequence=event[1],
                value=event[2],
                a=event[3],
                b=event[4],
                kind=kind,
                detail=detail,
            )
        )
    if offset != len(payload):
        raise ExactProtocolError("exact transition response has trailing bytes")
    return ExactTransition(
        observation=observation,
        reward=values[0],
        event_count=values[1],
        config_hash=values[2],
        finish_call_count=values[3],
        recorded_final_score=values[4],
        recorded_final_clears=values[5],
        latest_final_score=values[6],
        latest_final_clears=values[7],
        recorded_final_highest_chain=values[8],
        recorded_final_level=values[9],
        latest_final_highest_chain=values[10],
        latest_final_level=values[11],
        terminated=bool(values[12]),
        truncated=bool(values[13]),
        terminal_metadata_recorded=bool(values[14]),
        invalid_action=bool(values[15]),
        events=tuple(events),
    )


def _validate_raw_observation(payload: bytes) -> tuple[tuple[Any, ...], int]:
    if len(payload) < _OBSERVATION_HEADER.size:
        raise ExactProtocolError("truncated exact observation header")
    header = _OBSERVATION_HEADER.unpack_from(payload)
    body_count = header[15]
    if body_count > _BODY_CAPACITY:
        raise ExactProtocolError(
            f"exact observation body count {body_count} exceeds capacity"
        )
    body_begin = _OBSERVATION_HEADER.size
    body_end = body_begin + body_count * _BODY.size
    if len(payload) - body_begin < body_count * _BODY.size:
        raise ExactProtocolError("truncated exact observation bodies")
    for offset in range(body_begin, body_end, _BODY.size):
        kind, shape, lifecycle, reserved = payload[offset + 96 : offset + 100]
        if kind > 2 or shape > 2 or lifecycle > 4 or reserved:
            raise ExactProtocolError("exact observation contains invalid body metadata")
    return header, body_end


def _validate_raw_transition(payload: bytes) -> tuple[int, int]:
    """Validate a transition without allocating body/event dataclasses.

    Returns the exclusive observation offset and transition configuration hash.
    The exact padded vector uses this path to retain the worker wire layout in a
    reusable packed ctypes buffer.
    """

    header, body_end = _validate_raw_observation(payload)
    if len(payload) - body_end < _TRANSITION.size:
        raise ExactProtocolError("exact transition has a truncated diagnostics suffix")
    transition = _TRANSITION.unpack_from(payload, body_end)
    if bool(transition[12]) != bool(header[16]):
        raise ExactProtocolError("exact transition termination flags disagree")
    if bool(transition[13]) != bool(header[17]):
        raise ExactProtocolError("exact transition truncation flags disagree")

    offset = body_end + _TRANSITION.size
    for _ in range(transition[1]):
        if len(payload) - offset < _EVENT.size:
            raise ExactProtocolError("truncated exact event header")
        event = _EVENT.unpack_from(payload, offset)
        offset += _EVENT.size
        detail_size, kind, reserved = event[5:]
        if kind > 18 or reserved:
            raise ExactProtocolError("exact event contains invalid metadata")
        if detail_size >= EVENT_DETAIL_CAPACITY:
            raise ExactProtocolError("exact event detail exceeds padded capacity")
        if len(payload) - offset < detail_size:
            raise ExactProtocolError("truncated exact event detail")
        try:
            payload[offset : offset + detail_size].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExactProtocolError("exact event detail is not UTF-8") from exc
        offset += detail_size
    if offset != len(payload):
        raise ExactProtocolError("exact transition response has trailing bytes")
    return body_end, int(transition[2])


def _validate_raw_padded_transition(payload: bytes) -> tuple[int, int, int]:
    """Validate the fixed padded response and return body end, hash, generation."""

    header, body_end = _validate_raw_observation(payload)
    expected = body_end + _TRANSITION.size + _EVENT_GENERATION.size
    if len(payload) != expected:
        raise ExactProtocolError("exact padded transition has an invalid size")
    transition = _TRANSITION.unpack_from(payload, body_end)
    if bool(transition[12]) != bool(header[16]):
        raise ExactProtocolError("exact transition termination flags disagree")
    if bool(transition[13]) != bool(header[17]):
        raise ExactProtocolError("exact transition truncation flags disagree")
    generation = _EVENT_GENERATION.unpack_from(
        payload, body_end + _TRANSITION.size
    )[0]
    if generation == 0:
        raise ExactProtocolError("exact padded event generation must be nonzero")
    return body_end, int(transition[2]), int(generation)


def _validate_raw_events(payload: bytes, expected_count: int) -> None:
    if len(payload) < _EVENT_COUNT.size:
        raise ExactProtocolError("exact padded events response is truncated")
    count = _EVENT_COUNT.unpack_from(payload)[0]
    if count != expected_count:
        raise ExactProtocolError("exact padded event count changed")
    offset = _EVENT_COUNT.size
    for _ in range(count):
        if len(payload) - offset < _EVENT.size:
            raise ExactProtocolError("truncated exact event header")
        event = _EVENT.unpack_from(payload, offset)
        offset += _EVENT.size
        detail_size, kind, reserved = event[5:]
        if kind > 18 or reserved:
            raise ExactProtocolError("exact event contains invalid metadata")
        if detail_size >= EVENT_DETAIL_CAPACITY:
            raise ExactProtocolError("exact event detail exceeds padded capacity")
        if len(payload) - offset < detail_size:
            raise ExactProtocolError("truncated exact event detail")
        try:
            payload[offset : offset + detail_size].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExactProtocolError("exact event detail is not UTF-8") from exc
        offset += detail_size
    if offset != len(payload):
        raise ExactProtocolError("exact padded events response has trailing bytes")


def find_exact_worker(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve an explicit worker, environment override, or workspace build."""

    requested = explicit or os.environ.get("IRISU_EXACT_WORKER")
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise ExactWorkerNotFoundError(f"exact worker does not exist: {path}")
        if not os.access(path, os.X_OK):
            raise ExactWorkerNotFoundError(f"exact worker is not executable: {path}")
        return path

    root = Path(__file__).resolve().parents[2]
    candidates = (
        root / "build" / "irisu-exact-worker",
        root / "build" / "Release" / "irisu-exact-worker",
        root / "build-physics-integration-exact-multiworld-2" / "irisu-exact-worker",
        root / "build-physics-integration-exact" / "irisu-exact-worker",
        root / "build-exact-ipc" / "irisu-exact-worker",
        root / "build-exact-ipc-cmake" / "irisu-exact-worker",
        Path(__file__).resolve().parent / "irisu-exact-worker",
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise ExactWorkerNotFoundError(
        "could not find the exact physics worker; build the exact-msvc CMake "
        "backend, set IRISU_EXACT_WORKER, or pass worker_path "
        f"(searched: {searched})"
    )


class ExactWorkerClient:
    """Own one persistent 32-bit exact worker and its single physics world."""

    def __init__(
        self,
        worker: str | os.PathLike[str],
        *,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = find_exact_worker(worker)
        self._process: subprocess.Popen[bytes] | None = None
        self._connection: socket.socket | None = None
        try:
            self._process = subprocess.Popen(
                [str(self.path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise ExactWorkerError(f"failed to launch exact worker {self.path}: {exc}") from exc
        if self._process.stdin is None or self._process.stdout is None:
            self._process.kill()
            self._process.wait()
            raise ExactWorkerError("exact worker pipes were not created")
        self._reader: BinaryIO = self._process.stdout
        self._writer: BinaryIO = self._process.stdin
        self._transport_pid = self._process.pid
        self._initialize_protocol(config=config, expected_pid=self._process.pid)

    def _connect_fast_branch(
        self,
        address: bytes,
        secret: bytes,
        expected_pid: int,
        expected_keeper_pid: int,
    ) -> ExactWorkerClient:
        branch = type(self).__new__(type(self))
        branch.path = self.path
        branch._process = None
        branch._connection = None
        branch._transport_pid = expected_pid
        branch._closed = False
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        branch._connection = connection
        try:
            connection.settimeout(3.0)
            connection.connect(address)
            _verify_keeper_peer(connection, expected_keeper_pid)
            _verify_inherited_branch_files(
                expected_pid,
                expected_keeper_pid,
                self._executed_file_identity,
                self.initial_exact_library_provenance,
            )
            connection.sendall(secret)
            connection.settimeout(None)
            branch._reader = connection.makefile("rb", buffering=0)
            branch._writer = connection.makefile("wb", buffering=0)
            branch._initialize_protocol(
                config=None,
                expected_pid=expected_pid,
                inherited_from=self,
            )
            return branch
        except BaseException:
            try:
                branch._abort()
            except Exception:
                connection.close()
            raise

    def _initialize_protocol(
        self,
        *,
        config: Mapping[str, Any] | None,
        expected_pid: int,
        inherited_from: ExactWorkerClient | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._request_id = 0
        self._pending: tuple[int, int] | None = None
        self._closed = False
        self.last_response_bytes = 0
        try:
            self.info = self._hello()
            if self.info.pid != expected_pid:
                raise ExactProtocolError(
                    "exact worker hello PID disagrees with its transport"
                )
            if self.info.protocol_version != _PROTOCOL_VERSION:
                raise ExactProtocolError("exact worker protocol version is unsupported")
            if (
                self.info.pointer_bits != 32
                or self.info.body_capacity != _BODY_CAPACITY
                or self.info.process_model != 1
            ):
                raise ExactProtocolError("exact worker exposes an incompatible ABI")
            if self.info.backend != _EXACT_BACKEND:
                raise ExactProtocolError(
                    "exact worker is not the required exact multiworld backend "
                    f"(expected {_EXACT_BACKEND!r}, got {self.info.backend!r})"
                )
            normalized_library_sha256 = normalize_exact_library_sha256(
                self.info.exact_library_sha256
            )
            if normalized_library_sha256 is None:
                raise ExactProtocolError(
                    "exact worker did not report a valid non-placeholder "
                    "exact-library SHA-256"
                )
            self.info = replace(
                self.info, exact_library_sha256=normalized_library_sha256
            )
            try:
                (
                    self.exact_entrypoint_count,
                    attested_device,
                    attested_inode,
                ) = _decode_exact_attestation(
                    self._request(_EXACT_ATTESTATION)
                )
                self.exact_call_target_identity = {
                    "device": attested_device,
                    "inode": attested_inode,
                }
            except ExactWorkerError as exc:
                raise ExactProtocolError(
                    "exact worker does not expose required call-target attestation"
                ) from exc
            if inherited_from is None:
                # Hello proves exec completed; inspecting /proc earlier can race
                # posix_spawn and capture the parent Python executable instead.
                (
                    self.executable_sha256,
                    self._executed_file_identity,
                ) = _capture_executed_worker(expected_pid)
                self.initial_exact_library_provenance = (
                    _capture_mapped_exact_library(
                        expected_pid, normalized_library_sha256
                    )
                )
            else:
                inherited_library = inherited_from.initial_exact_library_provenance
                if inherited_library.get("sha256") != normalized_library_sha256:
                    raise ExactProtocolError(
                        "exact fast-branch library provenance changed"
                    )
                self.executable_sha256 = inherited_from.executable_sha256
                self._executed_file_identity = (
                    inherited_from._executed_file_identity
                )
                self.initial_exact_library_provenance = copy.deepcopy(
                    inherited_library
                )
            mapped_identity = self.initial_exact_library_provenance.get(
                "mapped_identity"
            )
            if (
                not isinstance(mapped_identity, Mapping)
                or mapped_identity.get("device") != attested_device
                or mapped_identity.get("inode") != attested_inode
            ):
                raise ExactProtocolError(
                    "exact call targets disagree with the captured library mapping"
                )
            self.current_config_hash = self.info.config_hash
            if config is not None:
                self.configure(config)
        except BaseException:
            self._abort()
            raise

    @property
    def closed(self) -> bool:
        return self._closed

    def _transport_status(self) -> object:
        if self._process is not None:
            return self._process.poll()
        return f"forked-pid-{self._transport_pid}"

    def _begin_request(self, opcode: int, payload: bytes = b"") -> None:
        with self._lock:
            if self._closed:
                raise ExactWorkerError("exact worker client is closed")
            if self._pending is not None:
                raise RuntimeError("exact worker already has an outstanding request")
            self._request_id = (self._request_id + 1) & 0xFFFFFFFF or 1
            request_id = self._request_id
            frame = _HEADER.pack(
                _MAGIC, _PROTOCOL_VERSION, opcode, request_id, len(payload)
            ) + payload
            try:
                _write_all(self._writer, frame)
            except (BrokenPipeError, OSError) as exc:
                status = self._transport_status()
                try:
                    self._abort()
                except Exception:
                    pass
                raise ExactWorkerError(
                    "exact worker exited unexpectedly "
                    f"(status={status})"
                ) from exc
            self._pending = (opcode, request_id)

    def _finish_response(self, opcode: int) -> bytes:
        with self._lock:
            if self._closed:
                raise ExactWorkerError("exact worker client is closed")
            if self._pending is None or self._pending[0] != opcode:
                raise RuntimeError("exact worker has no matching outstanding request")
            _, request_id = self._pending
            try:
                response_header = _read_exact(self._reader, _HEADER.size)
                magic, version, response_opcode, response_id, size = _HEADER.unpack(
                    response_header
                )
                if magic != _MAGIC or version != _PROTOCOL_VERSION:
                    raise ExactProtocolError("exact worker returned an invalid frame header")
                if response_opcode != opcode or response_id != request_id:
                    raise ExactProtocolError("exact worker response does not match request")
                if size < _STATUS.size or size > (4 << 20):
                    raise ExactProtocolError("exact worker returned an invalid payload size")
                response = _read_exact(self._reader, size)
            except ExactProtocolError:
                # A malformed frame header does not tell us where the next
                # response begins. Close the transport instead of presenting a
                # desynchronized stream as reusable.
                try:
                    self._abort()
                except Exception:
                    pass
                raise
            except (EOFError, OSError) as exc:
                status = self._transport_status()
                try:
                    self._abort()
                except Exception:
                    pass
                raise ExactWorkerError(
                    "exact worker exited unexpectedly "
                    f"(status={status})"
                ) from exc
            finally:
                self._pending = None
            self.last_response_bytes = _HEADER.size + size
            status = _STATUS.unpack_from(response)[0]
            content = response[_STATUS.size :]
            if status:
                detail = content.decode("utf-8", errors="replace")
                if status != 1:
                    # Internal errors are fail-closed in the worker. Unknown
                    # status values are likewise not a reusable protocol state.
                    try:
                        self._abort()
                    except Exception:
                        pass
                if status not in (1, 2):
                    raise ExactProtocolError(
                        f"exact worker returned unknown status {status}: {detail}"
                    )
                raise ExactWorkerError(f"exact worker status {status}: {detail}")
            return content

    def _request(self, opcode: int, payload: bytes = b"") -> bytes:
        self._begin_request(opcode, payload)
        return self._finish_response(opcode)

    def _hello(self) -> ExactWorkerInfo:
        payload = self._request(_HELLO)
        if len(payload) < _HELLO_FIXED.size:
            raise ExactProtocolError("exact worker hello response is truncated")
        values = _HELLO_FIXED.unpack_from(payload)
        offset = _HELLO_FIXED.size
        strings: list[str] = []
        for _ in range(3):
            if len(payload) - offset < 2:
                raise ExactProtocolError("exact worker hello string is truncated")
            size = struct.unpack_from("<H", payload, offset)[0]
            offset += 2
            if len(payload) - offset < size:
                raise ExactProtocolError("exact worker hello string is truncated")
            try:
                strings.append(payload[offset : offset + size].decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ExactProtocolError("exact worker hello string is not UTF-8") from exc
            offset += size
        if offset != len(payload):
            raise ExactProtocolError("exact worker hello response has trailing bytes")
        return ExactWorkerInfo(*values, *strings)

    def reset(self, seed: int = 0) -> ExactObservation:
        if not isinstance(seed, Integral) or isinstance(seed, bool):
            raise TypeError("seed must be an integer")
        seed = int(seed)
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("normal-mode seed must fit in uint32")
        payload = self._request(_RESET, _RESET_REQUEST.pack(seed))
        observation, offset = _decode_observation(payload)
        if offset != len(payload):
            raise ExactProtocolError("exact reset response has trailing bytes")
        return observation

    def observe(self) -> ExactObservation:
        payload = self._request(_OBSERVE)
        observation, offset = _decode_observation(payload)
        if offset != len(payload):
            raise ExactProtocolError("exact observation response has trailing bytes")
        return observation

    def configure(self, config: Mapping[str, Any]) -> int:
        flattened = _flatten_config(config)
        if len(flattened) > 1024:
            raise ValueError("configuration override count exceeds worker limit")
        payload = bytearray(_CONFIG_COUNT.pack(len(flattened)))
        for key, value in flattened:
            encoded = key.encode("utf-8")
            if len(encoded) > 0xFFFF:
                raise ValueError("encoded configuration key exceeds uint16 length")
            payload.extend(_CONFIG_KEY_SIZE.pack(len(encoded)))
            payload.extend(encoded)
            payload.extend(_CONFIG_VALUE.pack(value))
        response = self._request(_CONFIGURE, bytes(payload))
        if len(response) != _CONFIG_HASH.size:
            raise ExactProtocolError("exact configure response has the wrong size")
        self.current_config_hash = _CONFIG_HASH.unpack(response)[0]
        return self.current_config_hash

    def config_json(self) -> dict[str, Any]:
        payload = self._request(_CONFIG_JSON)
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExactProtocolError("exact worker returned invalid config JSON") from exc
        if not isinstance(value, dict):
            raise ExactProtocolError("exact worker config JSON is not an object")
        if value.get("config_hash") != self.current_config_hash:
            raise ExactProtocolError("exact worker config JSON hash disagrees")
        return value

    def step(
        self,
        kind: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        wait_ticks: int = 1,
        *,
        suppress_fresh_edges: bool = False,
    ) -> ExactTransition:
        self.send_step(
            kind, x, y, wait_ticks, suppress_fresh_edges=suppress_fresh_edges
        )
        return self.receive_step()

    def send_step(
        self,
        kind: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        wait_ticks: int = 1,
        *,
        suppress_fresh_edges: bool = False,
    ) -> None:
        self._send_step_request(
            _STEP,
            kind,
            x,
            y,
            wait_ticks,
            suppress_fresh_edges=suppress_fresh_edges,
        )

    def send_step_padded(
        self,
        kind: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        wait_ticks: int = 1,
        *,
        suppress_fresh_edges: bool = False,
    ) -> None:
        self._send_step_request(
            _STEP_PADDED,
            kind,
            x,
            y,
            wait_ticks,
            suppress_fresh_edges=suppress_fresh_edges,
        )

    def _send_step_request(
        self,
        opcode: int,
        kind: int,
        x: float,
        y: float,
        wait_ticks: int,
        *,
        suppress_fresh_edges: bool,
    ) -> None:
        if not isinstance(kind, Integral) or isinstance(kind, bool):
            raise TypeError("action kind must be an integer")
        if not isinstance(wait_ticks, Integral) or isinstance(wait_ticks, bool):
            raise TypeError("wait_ticks must be an integer")
        if not isinstance(x, Real) or isinstance(x, bool):
            raise TypeError("x must be a real number")
        if not isinstance(y, Real) or isinstance(y, bool):
            raise TypeError("y must be a real number")
        kind, wait_ticks = int(kind), int(wait_ticks)
        if not 0 <= kind <= 3:
            raise ValueError("action kind must be in [0, 3]")
        if not 0 <= wait_ticks <= 0xFFFFFFFF:
            raise ValueError("wait_ticks must fit in uint32")
        payload = _STEP_REQUEST.pack(
            kind, float(x), float(y), wait_ticks, int(bool(suppress_fresh_edges))
        )
        self._begin_request(opcode, payload)

    def receive_step(self) -> ExactTransition:
        return _decode_transition(self.receive_step_payload())

    def receive_step_payload(self) -> bytes:
        """Return a validated-frame step payload for allocation-light decoders."""

        return self._finish_response(_STEP)

    def receive_step_padded_payload(self) -> bytes:
        return self._finish_response(_STEP_PADDED)

    def fetch_events_payload(self, generation: int) -> bytes:
        return self._request(_FETCH_EVENTS, _EVENT_GENERATION.pack(generation))

    @staticmethod
    def _fast_token(value: bytes | bytearray | memoryview) -> bytes:
        token = bytes(value)
        if len(token) != _FAST_TOKEN_BYTES:
            raise ValueError("fast checkpoint token must contain 16 bytes")
        return token

    def fast_checkpoint(self) -> tuple[bytes, int]:
        """Create a Linux-local fork/COW keeper at this request boundary."""

        response = self._request(_FAST_CHECKPOINT)
        if len(response) != _FAST_CHECKPOINT_RESPONSE.size:
            raise ExactProtocolError("exact fast-checkpoint response has the wrong size")
        token, keeper_pid = _FAST_CHECKPOINT_RESPONSE.unpack(response)
        if token == bytes(_FAST_TOKEN_BYTES) or keeper_pid == 0:
            raise ExactProtocolError("exact fast-checkpoint metadata is invalid")
        return token, keeper_pid

    def release_fast_checkpoint(
        self, token: bytes | bytearray | memoryview
    ) -> None:
        """Release a keeper; the worker refuses while a branch remains alive."""

        response = self._request(_FAST_RELEASE, self._fast_token(token))
        if response:
            raise ExactProtocolError("exact fast-release response must be empty")

    def branch_fast_checkpoint(
        self,
        token: bytes | bytearray | memoryview,
        keeper_pid: int,
    ) -> ExactWorkerClient:
        """Connect to one rollout process forked from a frozen keeper."""

        response = self._request(_FAST_BRANCH, self._fast_token(token))
        if len(response) < _FAST_BRANCH_RESPONSE.size:
            raise ExactProtocolError("exact fast-branch response is truncated")
        process, address_size, secret = _FAST_BRANCH_RESPONSE.unpack_from(response)
        address = response[_FAST_BRANCH_RESPONSE.size :]
        if (
            process == 0
            or address_size == 0
            or address_size > _MAXIMUM_BRANCH_ADDRESS_BYTES
            or len(address) != address_size
            or not address.startswith(b"\0")
        ):
            raise ExactProtocolError("exact fast-branch metadata is invalid")
        branch = self._connect_fast_branch(
            address,
            secret,
            expected_pid=process,
            expected_keeper_pid=keeper_pid,
        )
        if (
            branch.current_config_hash != self.current_config_hash
            or branch.info.exact_library_sha256 != self.info.exact_library_sha256
            or branch.executable_sha256 != self.executable_sha256
            or branch.info.x87_control_word != self.info.x87_control_word
            or branch.info.compiler != self.info.compiler
        ):
            branch.close()
            raise ExactProtocolError("exact fast branch identity changed")
        return branch

    def _abort(self) -> None:
        self._closed = True
        connection = getattr(self, "_connection", None)
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        process = getattr(self, "_process", None)
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
        for name in ("_writer", "_reader"):
            stream = getattr(self, name, None)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        if connection is not None:
            connection.close()
        if process is not None and process.stderr is not None:
            process.stderr.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self._pending is not None:
                    self._finish_response(self._pending[0])
                if self._process is None or self._process.poll() is None:
                    self._request(_CLOSE)
            except (NativeError, OSError, RuntimeError):
                pass
            self._closed = True
            if self._connection is not None:
                try:
                    self._connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            for stream in (self._writer, self._reader):
                try:
                    stream.close()
                except OSError:
                    pass
            if self._connection is not None:
                self._connection.close()
            if self._process is not None:
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()
                if self._process.stderr is not None:
                    self._process.stderr.close()

    def __enter__(self) -> ExactWorkerClient:
        if self.closed:
            raise ExactWorkerError("exact worker client is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _worker_identity(
    info: ExactWorkerInfo, config_hash: int, executable_sha256: str
) -> bytes:
    description = "\0".join(
        (
            str(info.protocol_version),
            str(info.pointer_bits),
            str(info.body_capacity),
            str(config_hash),
            str(info.x87_control_word),
            info.backend,
            info.compiler,
            info.exact_library_sha256,
            executable_sha256,
        )
    )
    return hashlib.sha256(description.encode("utf-8")).digest()


def _new_state_hasher(identity: bytes, config_hash: int, seed: int):
    hasher = hashlib.blake2b(digest_size=8, person=_STATE_PERSON)
    hasher.update(identity)
    hasher.update(struct.pack("<QQ", config_hash, seed))
    return hasher


class ExactFastCheckpoint:
    """One reusable Linux-local fork/COW checkpoint capability.

    The source worker owns the dormant keeper and must outlive this object and
    every branch. Explicit release refuses while any handed-out branch remains
    open. The ordinary byte snapshot on each branch remains independently
    durable because the Python action history is copied with the native fork.
    """

    def __init__(
        self,
        source: ExactSimulator,
        token: bytes,
        keeper_pid: int,
    ) -> None:
        self._lock = threading.RLock()
        self._client = source._require_open()
        self._token = token
        self._keeper_pid = keeper_pid
        self._closed = False
        self._branches: list[weakref.ReferenceType[ExactSimulator]] = []
        self._path = source._path
        self._overrides = dict(source._overrides)
        self._config_hash = source._config_hash
        self._config = {
            key: list(value) if isinstance(value, list) else value
            for key, value in source._config.items()
        }
        self._identity = source._identity
        assert source._seed is not None and source._state_hasher is not None
        self._seed = source._seed
        self._actions = tuple(source._actions)
        self._observation = source._observation
        self._raw_observation = source._raw_observation
        self._state_hasher = source._state_hasher.copy()

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._closed or self._client.closed

    @property
    def keeper_pid(self) -> int:
        return self._keeper_pid

    def _active_branches(self) -> list[ExactSimulator]:
        retained: list[weakref.ReferenceType[ExactSimulator]] = []
        active: list[ExactSimulator] = []
        for reference in self._branches:
            branch = reference()
            if branch is not None:
                retained.append(reference)
                if not branch.closed and branch._fast_checkpoint_owner is self:
                    active.append(branch)
        self._branches = retained
        return active

    def branch(self) -> ExactSimulator:
        """Fork and connect one independent rollout at the checkpoint state."""

        with self._lock:
            if self._closed:
                raise NativeError("exact fast checkpoint is closed")
            if self._client.closed:
                self._closed = True
                raise NativeError("exact fast checkpoint source worker is closed")
            client = self._client.branch_fast_checkpoint(
                self._token, self._keeper_pid
            )
            try:
                simulator = ExactSimulator._from_fast_checkpoint(self, client)
            except BaseException:
                client.close()
                raise
            self._branches.append(weakref.ref(simulator))
            return simulator

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self._client.closed:
                self._closed = True
                return
            if self._active_branches():
                raise NativeError(
                    "cannot release exact fast checkpoint with active branches"
                )
            self._client.release_fast_checkpoint(self._token)
            self._closed = True

    release = close

    def __enter__(self) -> ExactFastCheckpoint:
        if self.closed:
            raise NativeError("exact fast checkpoint is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __copy__(self) -> ExactFastCheckpoint:
        raise TypeError("ExactFastCheckpoint is a unique process capability")

    def __deepcopy__(self, memo: dict[int, object]) -> ExactFastCheckpoint:
        del memo
        raise TypeError("ExactFastCheckpoint is a unique process capability")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ExactSimulator:
    """NativeSimulator-shaped owner for the exact out-of-process backend.

    Snapshots contain the reset seed and accepted action log. Restoring starts a
    fresh worker and deterministically replays that log, preserving hidden
    solver state without pretending it can be serialized directly.
    """

    def __init__(
        self,
        worker_path: str | os.PathLike[str] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        flattened = _flatten_config(config or {})
        self._overrides = dict(flattened)
        self._lock = threading.RLock()
        self._path = find_exact_worker(worker_path)
        self._client: ExactWorkerClient | None = ExactWorkerClient(
            self._path,
            config=self._overrides if self._overrides else None,
        )
        self._config_hash = self._client.current_config_hash
        self._config = self._client.config_json()
        self._identity = _worker_identity(
            self._client.info,
            self._config_hash,
            self._client.executable_sha256,
        )
        self._seed: int | None = None
        self._actions: list[bytes] = []
        self._observation: ExactObservation | None = None
        self._raw_observation: tuple[bytes, int] | None = None
        self._state_hasher = None
        self._pending_action: bytes | None = None
        self._fast_checkpoint_owner: ExactFastCheckpoint | None = None
        self._fast_checkpoints: list[
            weakref.ReferenceType[ExactFastCheckpoint]
        ] = []

    @classmethod
    def _from_fast_checkpoint(
        cls,
        checkpoint: ExactFastCheckpoint,
        client: ExactWorkerClient,
    ) -> ExactSimulator:
        if (
            client.current_config_hash != checkpoint._config_hash
            or _worker_identity(
                client.info,
                checkpoint._config_hash,
                client.executable_sha256,
            )
            != checkpoint._identity
        ):
            raise ExactProtocolError("exact fast branch simulator identity changed")
        self = cls.__new__(cls)
        self._overrides = dict(checkpoint._overrides)
        self._lock = threading.RLock()
        self._path = checkpoint._path
        self._client = client
        self._config_hash = checkpoint._config_hash
        self._config = {
            key: list(value) if isinstance(value, list) else value
            for key, value in checkpoint._config.items()
        }
        self._identity = checkpoint._identity
        self._seed = checkpoint._seed
        self._actions = list(checkpoint._actions)
        self._observation = checkpoint._observation
        self._raw_observation = checkpoint._raw_observation
        self._state_hasher = checkpoint._state_hasher.copy()
        self._pending_action = None
        # Keep the capability alive until this branch closes. The checkpoint
        # itself tracks branches weakly so it does not create an ownership cycle.
        self._fast_checkpoint_owner = checkpoint
        self._fast_checkpoints = []
        return self

    @property
    def library_path(self) -> str:
        """Compatibility alias: the exact backend's executable path."""

        return str(self._path)

    @property
    def worker_path(self) -> str:
        return str(self._path)

    @property
    def closed(self) -> bool:
        with self._lock:
            return self._client is None or self._client.closed

    def _require_open(self) -> ExactWorkerClient:
        if self._client is None or self._client.closed:
            raise NativeError("exact simulator is closed")
        return self._client

    def _require_no_fast_checkpoints(self, operation: str) -> None:
        retained: list[weakref.ReferenceType[ExactFastCheckpoint]] = []
        for reference in self._fast_checkpoints:
            checkpoint = reference()
            if checkpoint is not None and not checkpoint.closed:
                retained.append(reference)
        self._fast_checkpoints = retained
        if retained:
            raise NativeError(
                f"cannot {operation} with an active exact fast checkpoint"
            )

    def _spawn_matching_client(self) -> ExactWorkerClient:
        candidate = ExactWorkerClient(
            self._path,
            config=self._overrides if self._overrides else None,
        )
        if (
            candidate.current_config_hash != self._config_hash
            or _worker_identity(
                candidate.info,
                self._config_hash,
                candidate.executable_sha256,
            )
            != self._identity
        ):
            candidate.close()
            raise NativeError("fresh exact worker identity changed")
        return candidate

    def _require_replacement_safe(self, operation: str) -> None:
        """Fail before an owner swaps a client that backs local checkpoints."""

        with self._lock:
            self._require_open()
            if self._pending_action is not None:
                raise NativeError(f"cannot {operation} with a pending exact step")
            self._require_no_fast_checkpoints(operation)

    def reset_typed(self, seed: int) -> ExactObservation:
        """Reset and return the decoded observation without materializing a dict."""

        with self._lock:
            current = self._require_open()
            self._require_no_fast_checkpoints("reset")
            candidate: ExactWorkerClient | None = None
            target = current
            try:
                if self._seed is not None:
                    candidate = self._spawn_matching_client()
                    target = candidate
                observation = target.reset(seed)
            except BaseException:
                if candidate is not None:
                    candidate.close()
                raise

            seed = int(seed)
            if candidate is not None:
                self._client = candidate
            self._seed = seed
            self._actions.clear()
            self._observation = observation
            self._raw_observation = None
            self._pending_action = None
            self._state_hasher = _new_state_hasher(
                self._identity, self._config_hash, seed
            )
            if candidate is not None:
                self._fast_checkpoint_owner = None
                current.close()
            return observation

    def reset(self, seed: int) -> dict[str, Any]:
        return self.reset_typed(seed).to_dict()

    def step(
        self, action_kind: int, x: float, y: float, wait_ticks: int
    ) -> dict[str, Any]:
        return self.step_typed(action_kind, x, y, wait_ticks).to_step_dict()

    def step_typed(
        self, action_kind: int, x: float, y: float, wait_ticks: int
    ) -> ExactTransition:
        """Step and retain the worker's typed transition representation."""

        self.send_step(action_kind, x, y, wait_ticks)
        return self.receive_step_typed()

    def send_step(
        self, action_kind: int, x: float, y: float, wait_ticks: int
    ) -> None:
        """Start one step so vector owners can run worker processes in parallel."""

        self._send_step_request(False, action_kind, x, y, wait_ticks)

    def send_step_padded(
        self, action_kind: int, x: float, y: float, wait_ticks: int
    ) -> None:
        """Start an exact step whose response defers event serialization."""

        self._send_step_request(True, action_kind, x, y, wait_ticks)

    def _pending_response_fd(self) -> int:
        """Return the readable descriptor for this simulator's pending step."""

        with self._lock:
            if self._pending_action is None:
                raise NativeError("exact simulator has no pending step")
            return self._require_open()._reader.fileno()

    def _send_step_request(
        self,
        padded: bool,
        action_kind: int,
        x: float,
        y: float,
        wait_ticks: int,
    ) -> None:

        with self._lock:
            if self._seed is None or self._state_hasher is None:
                raise NativeError("exact simulator must be reset before step")
            if self._pending_action is not None:
                raise NativeError("exact simulator already has a pending step")
            client = self._require_open()
            sender = client.send_step_padded if padded else client.send_step
            sender(action_kind, x, y, wait_ticks)
            self._pending_action = _STEP_REQUEST.pack(
                int(action_kind), float(x), float(y), int(wait_ticks), 0
            )

    def receive_step_typed(self) -> ExactTransition:
        """Finish a step previously started by :meth:`send_step`."""

        with self._lock:
            action = self._pending_action
            if action is None:
                raise NativeError("exact simulator has no pending step")
            client = self._require_open()
            try:
                transition = client.receive_step()
                if transition.config_hash != self._config_hash:
                    raise ExactProtocolError("exact transition config hash changed")
            except BaseException:
                # Once a request was sent, a failed response leaves the remote
                # world's advancement unknowable. Kill it rather than allow a
                # stale action log to drive subsequent steps or snapshots.
                self._pending_action = None
                self._client = None
                try:
                    client._abort()
                except Exception:
                    pass
                raise
            self._pending_action = None
            self._actions.append(action)
            self._state_hasher.update(action)
            self._observation = transition.observation
            self._raw_observation = None
            return transition

    def receive_step(self) -> dict[str, Any]:
        return self.receive_step_typed().to_step_dict()

    def receive_step_raw(self) -> bytes:
        """Finish a step and retain its validated allocation-light wire payload."""

        with self._lock:
            action = self._pending_action
            if action is None:
                raise NativeError("exact simulator has no pending step")
            client = self._require_open()
            try:
                payload = client.receive_step_payload()
                observation_end, config_hash = _validate_raw_transition(payload)
                if config_hash != self._config_hash:
                    raise ExactProtocolError("exact transition config hash changed")
            except BaseException:
                # A malformed or missing response makes remote advancement
                # unknowable, so this lane cannot safely continue.
                self._pending_action = None
                self._client = None
                try:
                    client._abort()
                except Exception:
                    pass
                raise
            self._pending_action = None
            self._actions.append(action)
            self._state_hasher.update(action)
            self._observation = None
            self._raw_observation = (payload, observation_end)
            return payload

    def receive_step_padded_raw(self) -> tuple[bytes, int]:
        """Finish a padded step and return its fixed payload and event generation."""

        with self._lock:
            action = self._pending_action
            if action is None:
                raise NativeError("exact simulator has no pending step")
            client = self._require_open()
            try:
                payload = client.receive_step_padded_payload()
                observation_end, config_hash, generation = (
                    _validate_raw_padded_transition(payload)
                )
                if config_hash != self._config_hash:
                    raise ExactProtocolError("exact transition config hash changed")
            except BaseException:
                self._pending_action = None
                self._client = None
                try:
                    client._abort()
                except Exception:
                    pass
                raise
            self._pending_action = None
            self._actions.append(action)
            self._state_hasher.update(action)
            self._observation = None
            self._raw_observation = (payload, observation_end)
            return payload, generation

    def fetch_padded_events_raw(
        self, generation: int, expected_count: int
    ) -> bytes:
        """Fetch one unexpired padded event batch without advancing the world."""

        if not isinstance(generation, Integral) or isinstance(generation, bool):
            raise TypeError("event generation must be an integer")
        if not isinstance(expected_count, Integral) or isinstance(expected_count, bool):
            raise TypeError("expected event count must be an integer")
        generation, expected_count = int(generation), int(expected_count)
        if not 0 < generation <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("event generation must fit in nonzero uint64")
        if not 0 <= expected_count <= 0xFFFFFFFFFFFFFFFF:
            raise ValueError("expected event count must fit in uint64")
        with self._lock:
            payload = self._require_open().fetch_events_payload(generation)
            _validate_raw_events(payload, expected_count)
            return payload

    def observation_typed(self) -> ExactObservation:
        """Return the immutable decoded observation without allocating a dict."""

        with self._lock:
            self._require_open()
            if self._observation is None:
                if self._raw_observation is None:
                    raise NativeError("exact simulator must be reset before observation")
                payload, observation_end = self._raw_observation
                observation, offset = _decode_observation(payload)
                if offset != observation_end:
                    raise ExactProtocolError("cached exact observation size changed")
                self._observation = observation
            return self._observation

    def observation(self) -> dict[str, Any]:
        return self.observation_typed().to_dict()

    def state_hash(self) -> int:
        with self._lock:
            self._require_open()
            if self._state_hasher is None:
                raise NativeError("exact simulator must be reset before state_hash")
            return int.from_bytes(self._state_hasher.copy().digest(), "little")

    def config_hash(self) -> int:
        with self._lock:
            self._require_open()
            return self._config_hash

    def config(self) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            return {
                key: list(value) if isinstance(value, list) else value
                for key, value in self._config.items()
            }

    def build_info(self) -> dict[str, Any]:
        with self._lock:
            client = self._require_open()
            info = client.info
            return {
                "physics_backend": "exact-msvc9-r58-worker",
                "physics": "Box2D SVN r58 compiled by MSVC9",
                "protocol_version": info.protocol_version,
                "pointer_bits": info.pointer_bits,
                "body_capacity": info.body_capacity,
                "worker_pid": info.pid,
                "process_model": "one-world-per-worker",
                "config_hash": self._config_hash,
                "x87_control_word": info.x87_control_word,
                "worker_backend": info.backend,
                "worker_compiler": info.compiler,
                "exact_library_sha256": info.exact_library_sha256,
                "exact_library_runtime_verified": True,
                "exact_call_targets_runtime_verified": True,
                "exact_entrypoint_count": client.exact_entrypoint_count,
                "worker_executable_sha256": client.executable_sha256,
                "snapshot_schema": _SNAPSHOT_SCHEMA,
                "snapshot_model": "seed-and-action-log-replay",
                "fast_snapshot_model": "linux-fork-cow-keeper",
                "fast_snapshot_reusable": True,
                "config_overrides": True,
                "seed_bits": 32,
            }

    def exact_library_provenance(self) -> dict[str, Any]:
        """Capture the exact physics library mapped by the live worker."""

        with self._lock:
            client = self._require_open()
            captured = _capture_mapped_exact_library(
                client.info.pid, client.info.exact_library_sha256
            )
            if captured != client.initial_exact_library_provenance:
                raise ExactProtocolError(
                    "mapped exact library changed since worker launch"
                )
            return captured

    def fast_checkpoint(self) -> ExactFastCheckpoint:
        """Create a reusable constant-time local checkpoint.

        This is an opt-in Linux process capability. Use :meth:`clone_state`
        for persistent or cross-process state bytes.
        """

        with self._lock:
            client = self._require_open()
            if self._seed is None or self._state_hasher is None:
                raise NativeError(
                    "exact simulator must be reset before fast_checkpoint"
                )
            if self._pending_action is not None:
                raise NativeError("cannot checkpoint with a pending exact step")
            token, keeper_pid = client.fast_checkpoint()
            try:
                checkpoint = ExactFastCheckpoint(self, token, keeper_pid)
                self._fast_checkpoints.append(weakref.ref(checkpoint))
                return checkpoint
            except BaseException:
                try:
                    client.release_fast_checkpoint(token)
                except Exception:
                    pass
                raise

    def clone_state(self) -> bytes:
        with self._lock:
            client = self._require_open()
            if self._seed is None:
                raise NativeError("exact simulator must be reset before clone_state")
            if self._pending_action is not None:
                raise NativeError("cannot clone state with a pending exact step")
            prefix = _SNAPSHOT_PREFIX.pack(
                _SNAPSHOT_MAGIC,
                _SNAPSHOT_SCHEMA,
                self._config_hash,
                _PROTOCOL_VERSION,
                0,
                self._seed,
                len(self._actions),
                self._identity,
            )
            payload = prefix + b"".join(self._actions)
            return payload + hashlib.sha256(payload).digest()

    def _decode_snapshot(self, snapshot: bytes) -> tuple[int, list[bytes]]:
        minimum = _SNAPSHOT_PREFIX.size + _SNAPSHOT_CHECKSUM_SIZE
        if len(snapshot) < minimum:
            raise NativeError("exact snapshot is shorter than its versioned header")
        payload = snapshot[:-_SNAPSHOT_CHECKSUM_SIZE]
        checksum = snapshot[-_SNAPSHOT_CHECKSUM_SIZE:]
        if not hmac.compare_digest(hashlib.sha256(payload).digest(), checksum):
            raise NativeError("exact snapshot checksum mismatch")
        (
            magic,
            schema,
            config_hash,
            protocol,
            flags,
            seed,
            count,
            identity,
        ) = _SNAPSHOT_PREFIX.unpack_from(payload)
        if magic != _SNAPSHOT_MAGIC:
            raise NativeError("exact snapshot magic mismatch")
        if schema != _SNAPSHOT_SCHEMA:
            raise NativeError(f"unsupported exact snapshot schema {schema}")
        if protocol != _PROTOCOL_VERSION or flags:
            raise NativeError("exact snapshot protocol metadata is incompatible")
        client = self._require_open()
        if config_hash != self._config_hash or identity != self._identity:
            raise NativeError("exact snapshot backend/config identity mismatch")
        if seed > 0xFFFFFFFF:
            raise NativeError("exact snapshot seed exceeds uint32")
        expected = _SNAPSHOT_PREFIX.size + count * _STEP_REQUEST.size
        if expected != len(payload):
            raise NativeError("exact snapshot action log length mismatch")
        actions: list[bytes] = []
        offset = _SNAPSHOT_PREFIX.size
        for _ in range(count):
            action = payload[offset : offset + _STEP_REQUEST.size]
            kind, _, _, wait_ticks, action_flags = _STEP_REQUEST.unpack(action)
            if kind > 3 or wait_ticks > 0xFFFFFFFF or action_flags:
                raise NativeError("exact snapshot contains an invalid action")
            actions.append(action)
            offset += _STEP_REQUEST.size
        return int(seed), actions

    def restore_state_typed(
        self, snapshot: bytes | bytearray | memoryview
    ) -> ExactObservation:
        """Restore transactionally and return the decoded typed observation."""

        data = bytes(snapshot)
        with self._lock:
            seed, actions = self._decode_snapshot(data)
            self._require_no_fast_checkpoints("restore state")
            candidate: ExactWorkerClient | None = None
            try:
                candidate = self._spawn_matching_client()
                observation = candidate.reset(seed)
                for action in actions:
                    kind, x, y, wait_ticks, flags = _STEP_REQUEST.unpack(action)
                    transition = candidate.step(
                        kind,
                        x,
                        y,
                        wait_ticks,
                        suppress_fresh_edges=bool(flags),
                    )
                    observation = transition.observation
            except BaseException:
                if candidate is not None:
                    candidate.close()
                raise

            previous = self._require_open()
            self._client = candidate
            self._seed = seed
            self._actions = actions
            self._observation = observation
            self._raw_observation = None
            self._pending_action = None
            self._state_hasher = _new_state_hasher(
                self._identity, self._config_hash, seed
            )
            for action in actions:
                self._state_hasher.update(action)
            previous.close()
            self._fast_checkpoint_owner = None
            return observation

    def restore_state(
        self, snapshot: bytes | bytearray | memoryview
    ) -> dict[str, Any]:
        return self.restore_state_typed(snapshot).to_dict()

    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            client = self._client
            self._client = None
            if client is not None:
                client.close()
            self._observation = None
            self._raw_observation = None
            self._state_hasher = None
            self._pending_action = None
            self._fast_checkpoint_owner = None
            self._fast_checkpoints = []

    def __enter__(self) -> ExactSimulator:
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __copy__(self) -> ExactSimulator:
        raise TypeError("ExactSimulator owns a unique worker and cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> ExactSimulator:
        del memo
        raise TypeError("ExactSimulator owns a unique worker and cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
