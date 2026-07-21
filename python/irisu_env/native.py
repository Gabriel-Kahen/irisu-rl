"""Dependency-free ctypes binding for the headless clone C ABI."""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import struct
import sys
import threading
from contextlib import ExitStack
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any


class NativeError(RuntimeError):
    """The clone library rejected an operation or returned invalid data."""


class LibraryNotFoundError(NativeError):
    """No usable clone shared library could be located."""


class _ConfigOverride(ctypes.Structure):
    _fields_ = [("key", ctypes.c_char_p), ("value", ctypes.c_double)]


PADDED_BODY_CAPACITY = 196
EVENT_DETAIL_CAPACITY = 96
_BODY_KIND_NAMES = ("piece", "projectile", "bonus")
_SHAPE_NAMES = ("circle", "box", "triangle")
_LIFECYCLE_NAMES = (
    "scripted_falling",
    "dynamic_fresh",
    "confirmed",
    "rotten",
    "deleted",
)
_EVENT_KIND_NAMES = (
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
)


class PaddedBody(ctypes.Structure):
    """One public policy body in the version-1 typed observation ABI."""

    _fields_ = [
        ("age_ticks", ctypes.c_uint64),
        ("remaining_lifetime", ctypes.c_int64),
        ("rot_timer", ctypes.c_uint64),
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("vx", ctypes.c_double),
        ("vy", ctypes.c_double),
        ("angle", ctypes.c_double),
        ("angular_velocity", ctypes.c_double),
        ("size", ctypes.c_double),
        ("id", ctypes.c_uint32),
        ("color", ctypes.c_int32),
        ("chain_id", ctypes.c_uint32),
        ("projectile_hits", ctypes.c_uint32),
        ("kind", ctypes.c_uint8),
        ("shape", ctypes.c_uint8),
        ("lifecycle", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
    ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": int(self.id),
            "kind": _BODY_KIND_NAMES[self.kind],
            "shape": _SHAPE_NAMES[self.shape],
            "lifecycle": _LIFECYCLE_NAMES[self.lifecycle],
            "color": int(self.color),
            "x": float(self.x),
            "y": float(self.y),
            "vx": float(self.vx),
            "vy": float(self.vy),
            "angle": float(self.angle),
            "angular_velocity": float(self.angular_velocity),
            "size": float(self.size),
            "chain_id": int(self.chain_id),
            "projectile_hits": int(self.projectile_hits),
            "age_ticks": int(self.age_ticks),
            "remaining_lifetime": int(self.remaining_lifetime),
            "rot_timer": int(self.rot_timer),
        }


class PaddedObservation(ctypes.Structure):
    """Fixed-capacity policy observation; only slots below body_count are live."""

    _fields_ = [
        ("tick", ctypes.c_uint64),
        ("score", ctypes.c_int64),
        ("gauge", ctypes.c_int64),
        ("gauge_max", ctypes.c_int64),
        ("qualifying_clear_count", ctypes.c_uint64),
        ("field_x", ctypes.c_double),
        ("field_y", ctypes.c_double),
        ("field_width", ctypes.c_double),
        ("field_height", ctypes.c_double),
        ("side_wall_top", ctypes.c_double),
        ("side_wall_bottom", ctypes.c_double),
        ("level", ctypes.c_uint32),
        ("active_colors", ctypes.c_uint32),
        ("spawn_interval_ticks", ctypes.c_uint32),
        ("highest_chain", ctypes.c_uint32),
        ("body_count", ctypes.c_uint32),
        ("terminated", ctypes.c_uint8),
        ("truncated", ctypes.c_uint8),
        ("left_held", ctypes.c_uint8),
        ("right_held", ctypes.c_uint8),
        ("bodies", PaddedBody * PADDED_BODY_CAPACITY),
    ]

    def body_mask(self, index: int) -> bool:
        """Return the explicit padded-slot mask without allocating a mask array."""

        if not 0 <= index < PADDED_BODY_CAPACITY:
            raise IndexError(index)
        return index < self.body_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": int(self.tick),
            "score": int(self.score),
            "gauge": int(self.gauge),
            "level": int(self.level),
            "terminated": bool(self.terminated),
            "truncated": bool(self.truncated),
            "left_held": bool(self.left_held),
            "right_held": bool(self.right_held),
            "highest_chain": int(self.highest_chain),
            "qualifying_clear_count": int(self.qualifying_clear_count),
            "field": {
                "x": float(self.field_x),
                "y": float(self.field_y),
                "width": float(self.field_width),
                "height": float(self.field_height),
                "side_wall_top": float(self.side_wall_top),
                "side_wall_bottom": float(self.side_wall_bottom),
            },
            "gauge_max": int(self.gauge_max),
            "difficulty": {
                "active_colors": int(self.active_colors),
                "spawn_interval_ticks": int(self.spawn_interval_ticks),
            },
            "bodies": [self.bodies[index].to_dict() for index in range(self.body_count)],
        }


class PaddedTransition(ctypes.Structure):
    """Typed observation, reward, termination, and hidden diagnostics."""

    _fields_ = [
        ("observation", PaddedObservation),
        ("reward", ctypes.c_int64),
        ("event_count", ctypes.c_uint64),
        ("config_hash", ctypes.c_uint64),
        ("finish_call_count", ctypes.c_uint64),
        ("recorded_final_score", ctypes.c_int64),
        ("recorded_final_clears", ctypes.c_uint64),
        ("latest_final_score", ctypes.c_int64),
        ("latest_final_clears", ctypes.c_uint64),
        ("recorded_final_highest_chain", ctypes.c_uint32),
        ("recorded_final_level", ctypes.c_uint32),
        ("latest_final_highest_chain", ctypes.c_uint32),
        ("latest_final_level", ctypes.c_uint32),
        ("terminated", ctypes.c_uint8),
        ("truncated", ctypes.c_uint8),
        ("terminal_metadata_recorded", ctypes.c_uint8),
        ("invalid_action", ctypes.c_uint8),
    ]

    def diagnostics(self) -> dict[str, Any]:
        return {
            "config_hash": int(self.config_hash),
            "finish_call_count": int(self.finish_call_count),
            "terminal_metadata_recorded": bool(self.terminal_metadata_recorded),
            "recorded_final_score": int(self.recorded_final_score),
            "recorded_final_highest_chain": int(self.recorded_final_highest_chain),
            "recorded_final_level": int(self.recorded_final_level),
            "recorded_final_clears": int(self.recorded_final_clears),
            "latest_final_score": int(self.latest_final_score),
            "latest_final_highest_chain": int(self.latest_final_highest_chain),
            "latest_final_level": int(self.latest_final_level),
            "latest_final_clears": int(self.latest_final_clears),
        }


class _PaddedAction(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_double),
        ("y", ctypes.c_double),
        ("wait_ticks", ctypes.c_uint32),
        ("kind", ctypes.c_int32),
    ]


class PaddedEvent(ctypes.Structure):
    """One exact structured event from the version-1 typed ABI."""

    _fields_ = [
        ("tick", ctypes.c_uint64),
        ("sequence", ctypes.c_uint64),
        ("value", ctypes.c_int64),
        ("a", ctypes.c_uint32),
        ("b", ctypes.c_uint32),
        ("detail_size", ctypes.c_uint16),
        ("kind", ctypes.c_uint8),
        ("reserved", ctypes.c_uint8),
        ("detail", ctypes.c_char * EVENT_DETAIL_CAPACITY),
    ]

    @property
    def detail_text(self) -> str:
        return bytes(self.detail[: self.detail_size]).decode("utf-8")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": int(self.tick),
            "sequence": int(self.sequence),
            "kind": int(self.kind),
            "kind_name": _EVENT_KIND_NAMES[self.kind],
            "a": int(self.a),
            "b": int(self.b),
            "value": int(self.value),
            "detail": self.detail_text,
        }


class PaddedEvents(Sequence[PaddedEvent]):
    """Lazy exact event view, valid until its simulator advances again."""

    def __init__(self, simulator: NativeSimulator, count: int, generation: int) -> None:
        self._simulator = simulator
        self._count = count
        self._generation = generation
        self._cache: ctypes.Array[PaddedEvent] | None = None

    def _values(self) -> ctypes.Array[PaddedEvent]:
        if self._cache is None:
            self._cache = self._simulator._read_padded_events(
                self._count, self._generation
            )
        return self._cache

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int | slice) -> PaddedEvent | list[PaddedEvent]:
        values = self._values()
        if isinstance(index, slice):
            return [values[position] for position in range(*index.indices(self._count))]
        return values[index]

    def __iter__(self):
        values = self._values()
        return (values[index] for index in range(self._count))

    def materialize(self) -> tuple[PaddedEvent, ...]:
        return tuple(self)


def _flatten_config(config: Mapping[str, Any]) -> list[tuple[str, float]]:
    def encoded_number(value: object, label: str) -> float:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"configuration value {label} must be numeric")
        encoded = float(value)
        if isinstance(value, Integral) and int(encoded) != int(value):
            raise ValueError(
                f"configuration integer {label} is not exactly representable "
                "by the C ABI's double override value"
            )
        return encoded

    flattened: list[tuple[str, float]] = []
    for key, value in config.items():
        if not isinstance(key, str) or not key:
            raise TypeError("configuration keys must be non-empty strings")
        if "\0" in key:
            raise ValueError("configuration keys must not contain NUL characters")
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                label = f"{key}[{index}]"
                flattened.append((label, encoded_number(item, label)))
        else:
            flattened.append((key, encoded_number(value, key)))
    flattened.sort(key=lambda item: item[0])
    return flattened


_SNAPSHOT_HEADER = struct.Struct("<IIQ")
_SNAPSHOT_MAGIC = 0x49524953
_LIBRARY_CACHE: dict[str, ctypes.CDLL] = {}
_LIBRARY_CACHE_LOCK = threading.Lock()
_PADDED_BUFFER_TOKEN = object()
_RLOCK_TYPE = type(threading.RLock())


def _bounded_integer(value: object, label: str, maximum: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if not 0 <= result <= maximum:
        raise ValueError(f"{label} must be in [0, {maximum}]")
    return result


def _native_action(
    action_kind: object, x: object, y: object, wait_ticks: object
) -> tuple[int, float, float, int]:
    kind = _bounded_integer(action_kind, "action_kind", 3)
    if not isinstance(x, Real) or isinstance(x, bool):
        raise TypeError("x must be a real number")
    if not isinstance(y, Real) or isinstance(y, bool):
        raise TypeError("y must be a real number")
    wait = _bounded_integer(wait_ticks, "wait_ticks", 0xFFFFFFFF)
    return kind, float(x), float(y), wait


def _is_ctypes_array(value: object, element: type[Any], count: int) -> bool:
    return (
        isinstance(value, ctypes.Array)
        and getattr(value, "_type_", None) is element
        and getattr(value, "_length_", None) == count
    )


def snapshot_config_hash(snapshot: bytes | bytearray | memoryview) -> int:
    """Read the native mechanics hash from a versioned opaque snapshot header."""

    data = bytes(snapshot)
    if len(data) < _SNAPSHOT_HEADER.size:
        raise NativeError("snapshot is shorter than its versioned header")
    magic, schema_version, config_hash = _SNAPSHOT_HEADER.unpack_from(data)
    if magic != _SNAPSHOT_MAGIC:
        raise NativeError("snapshot magic mismatch")
    # 0x45580001 is the exact worker's durable seed/action-log snapshot.
    if schema_version not in (1, 2, 3, 4, 5, 6, 7, 0x45580001):
        raise NativeError(f"unsupported snapshot schema {schema_version}")
    return config_hash


def _library_name() -> str:
    if sys.platform == "win32":
        return "irisu_clone.dll"
    if sys.platform == "darwin":
        return "libirisu_clone.dylib"
    return "libirisu_clone.so"


def find_library(explicit: str | os.PathLike[str] | None = None) -> Path | str:
    """Resolve an explicit path, environment override, or workspace build."""

    requested = explicit or os.environ.get("IRISU_CLONE_LIBRARY")
    if requested:
        path = Path(requested).expanduser().resolve()
        if not path.is_file():
            raise LibraryNotFoundError(f"clone library does not exist: {path}")
        return path

    name = _library_name()
    root = Path(__file__).resolve().parents[2]
    candidates = (
        root / "build" / name,
        root / "build" / "Debug" / name,
        root / "build" / "Release" / name,
        Path(__file__).resolve().parent / name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    system = ctypes.util.find_library("irisu_clone")
    if system:
        return system
    searched = ", ".join(str(path) for path in candidates)
    raise LibraryNotFoundError(
        "could not find the headless clone shared library; run "
        "`cmake -S . -B build && cmake --build build`, set "
        f"IRISU_CLONE_LIBRARY, or pass library_path (searched: {searched})"
    )


class NativeSimulator:
    """Own one independent native simulator handle."""

    def __init__(
        self,
        library_path: str | os.PathLike[str] | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._handle: int | None = None
        self._path = find_library(library_path)
        cache_key = str(self._path)
        try:
            with _LIBRARY_CACHE_LOCK:
                library = _LIBRARY_CACHE.get(cache_key)
                if library is None:
                    library = ctypes.CDLL(cache_key)
                    _LIBRARY_CACHE[cache_key] = library
                self._library = library
                self._configure_abi()
        except OSError as exc:
            raise LibraryNotFoundError(f"failed to load clone library {self._path}: {exc}") from exc
        except AttributeError as exc:
            raise NativeError(f"clone library has an incompatible C ABI: {exc}") from exc
        self._handle = self._library.irisu_create()
        if not self._handle:
            raise NativeError("native simulator allocation failed")
        self._padded_generation = 0
        if config:
            try:
                self.configure(config)
            except Exception:
                self.close()
                raise

    @property
    def library_path(self) -> str:
        return str(self._path)

    @property
    def closed(self) -> bool:
        with self._lock:
            return not bool(getattr(self, "_handle", None))

    def _configure_abi(self) -> None:
        library = self._library
        library.irisu_create.argtypes = []
        library.irisu_create.restype = ctypes.c_void_p
        library.irisu_abi_version.argtypes = []
        library.irisu_abi_version.restype = ctypes.c_uint32
        library.irisu_destroy.argtypes = [ctypes.c_void_p]
        library.irisu_destroy.restype = None
        library.irisu_configure.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ConfigOverride),
            ctypes.c_size_t,
        ]
        library.irisu_configure.restype = ctypes.c_int
        library.irisu_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
        library.irisu_reset.restype = ctypes.c_int
        library.irisu_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_uint32,
        ]
        library.irisu_step.restype = ctypes.c_int
        library.irisu_padded_abi_version.argtypes = []
        library.irisu_padded_abi_version.restype = ctypes.c_uint32
        library.irisu_padded_body_capacity.argtypes = []
        library.irisu_padded_body_capacity.restype = ctypes.c_size_t
        library.irisu_padded_observation_size.argtypes = []
        library.irisu_padded_observation_size.restype = ctypes.c_size_t
        library.irisu_padded_transition_size.argtypes = []
        library.irisu_padded_transition_size.restype = ctypes.c_size_t
        library.irisu_padded_action_size.argtypes = []
        library.irisu_padded_action_size.restype = ctypes.c_size_t
        library.irisu_padded_event_size.argtypes = []
        library.irisu_padded_event_size.restype = ctypes.c_size_t
        library.irisu_padded_observation.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PaddedObservation),
        ]
        library.irisu_padded_observation.restype = ctypes.c_int
        library.irisu_padded_reset.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint64,
            ctypes.POINTER(PaddedObservation),
        ]
        library.irisu_padded_reset.restype = ctypes.c_int
        library.irisu_padded_step.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_uint32,
            ctypes.POINTER(PaddedTransition),
        ]
        library.irisu_padded_step.restype = ctypes.c_int
        library.irisu_padded_step_batch.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_PaddedAction),
            ctypes.POINTER(PaddedTransition),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        library.irisu_padded_step_batch.restype = ctypes.c_int
        library.irisu_padded_events.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(PaddedEvent),
            ctypes.c_size_t,
        ]
        library.irisu_padded_events.restype = ctypes.c_int
        library.irisu_observation_json.argtypes = [ctypes.c_void_p]
        library.irisu_observation_json.restype = ctypes.c_char_p
        library.irisu_step_json.argtypes = [ctypes.c_void_p]
        library.irisu_step_json.restype = ctypes.c_char_p
        library.irisu_state_hash.argtypes = [ctypes.c_void_p]
        library.irisu_state_hash.restype = ctypes.c_uint64
        library.irisu_config_hash.argtypes = [ctypes.c_void_p]
        library.irisu_config_hash.restype = ctypes.c_uint64
        library.irisu_config_json.argtypes = [ctypes.c_void_p]
        library.irisu_config_json.restype = ctypes.c_char_p
        library.irisu_build_info_json.argtypes = []
        library.irisu_build_info_json.restype = ctypes.c_char_p
        library.irisu_snapshot_size.argtypes = [ctypes.c_void_p]
        library.irisu_snapshot_size.restype = ctypes.c_size_t
        library.irisu_snapshot_write.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.irisu_snapshot_write.restype = ctypes.c_int
        library.irisu_snapshot_restore.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        library.irisu_snapshot_restore.restype = ctypes.c_int
        library.irisu_last_error.argtypes = [ctypes.c_void_p]
        library.irisu_last_error.restype = ctypes.c_char_p
        if int(library.irisu_abi_version()) != 1:
            raise NativeError("clone library C ABI version is not supported")
        padded_layout = (
            int(library.irisu_padded_abi_version()),
            int(library.irisu_padded_body_capacity()),
            int(library.irisu_padded_observation_size()),
            int(library.irisu_padded_transition_size()),
            int(library.irisu_padded_action_size()),
            int(library.irisu_padded_event_size()),
        )
        expected_layout = (
            1,
            PADDED_BODY_CAPACITY,
            ctypes.sizeof(PaddedObservation),
            ctypes.sizeof(PaddedTransition),
            ctypes.sizeof(_PaddedAction),
            ctypes.sizeof(PaddedEvent),
        )
        if padded_layout != expected_layout:
            raise NativeError(
                f"clone library padded ABI layout {padded_layout} does not match "
                f"Python layout {expected_layout}"
            )

    def _require_open(self) -> ctypes.c_void_p:
        handle = getattr(self, "_handle", None)
        if not handle:
            raise NativeError("native simulator is closed")
        return handle

    def _error(self, operation: str) -> NativeError:
        handle = getattr(self, "_handle", None)
        raw = self._library.irisu_last_error(handle) if handle else None
        detail = raw.decode("utf-8", errors="replace") if raw else "unknown native error"
        return NativeError(f"{operation} failed: {detail}")

    def _check(self, result: int, operation: str) -> None:
        if not result:
            raise self._error(operation)

    def _json(self, pointer: bytes | None, operation: str) -> dict[str, Any]:
        if pointer is None:
            raise self._error(operation)
        try:
            value = json.loads(pointer.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise NativeError(f"{operation} returned invalid JSON: {exc}") from exc
        if not isinstance(value, dict):
            raise NativeError(f"{operation} returned a non-object JSON value")
        return value

    def reset(self, seed: int) -> dict[str, Any]:
        resolved_seed = _bounded_integer(seed, "seed", 0xFFFFFFFF)
        with self._lock:
            handle = self._require_open()
            self._check(self._library.irisu_reset(handle, resolved_seed), "reset")
            self._padded_generation += 1
            return self.observation()

    def reset_padded(self, seed: int) -> PaddedObservation:
        resolved_seed = _bounded_integer(seed, "seed", 0xFFFFFFFF)
        with self._lock:
            destination = PaddedObservation()
            self._check(
                self._library.irisu_padded_reset(
                    self._require_open(), resolved_seed, ctypes.byref(destination)
                ),
                "padded reset",
            )
            self._padded_generation += 1
            return destination

    def configure(self, config: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(config, Mapping):
            raise TypeError("config must be a mapping")
        flattened = _flatten_config(config)
        encoded = [key.encode("utf-8") for key, _ in flattened]
        overrides = (_ConfigOverride * len(flattened))(
            *(_ConfigOverride(key, value) for key, (_, value) in zip(encoded, flattened))
        )
        pointer = overrides if flattened else None
        with self._lock:
            handle = self._require_open()
            self._check(
                self._library.irisu_configure(handle, pointer, len(flattened)),
                "configure",
            )
            self._padded_generation += 1
            return self.config()

    def step(self, action_kind: int, x: float, y: float, wait_ticks: int) -> dict[str, Any]:
        kind, cursor_x, cursor_y, wait = _native_action(
            action_kind, x, y, wait_ticks
        )
        with self._lock:
            handle = self._require_open()
            self._check(
                self._library.irisu_step(handle, kind, cursor_x, cursor_y, wait),
                "step",
            )
            self._padded_generation += 1
            return self._json(self._library.irisu_step_json(handle), "step result")

    def step_padded(
        self, action_kind: int, x: float, y: float, wait_ticks: int
    ) -> tuple[PaddedTransition, PaddedEvents]:
        kind, cursor_x, cursor_y, wait = _native_action(
            action_kind, x, y, wait_ticks
        )
        with self._lock:
            transition = PaddedTransition()
            self._check(
                self._library.irisu_padded_step(
                    self._require_open(),
                    kind,
                    cursor_x,
                    cursor_y,
                    wait,
                    ctypes.byref(transition),
                ),
                "padded step",
            )
            self._padded_generation += 1
            return transition, PaddedEvents(
                self, int(transition.event_count), self._padded_generation
            )

    @staticmethod
    def step_padded_batch(
        simulators: Sequence[NativeSimulator],
        actions: Sequence[tuple[int, float, float, int]],
        buffers: dict[str, Any] | None = None,
        workers: int = 8,
    ) -> tuple[list[tuple[PaddedTransition, PaddedEvents]], dict[str, Any]]:
        count = len(simulators)
        if len(actions) != count:
            raise ValueError("batch actions must match simulator count")
        if (
            not isinstance(workers, Integral)
            or isinstance(workers, bool)
            or workers <= 0
            or workers > ctypes.c_size_t(-1).value
        ):
            raise ValueError("workers must be a positive integer")
        if count == 0:
            return [], {}
        if any(not isinstance(simulator, NativeSimulator) for simulator in simulators):
            raise TypeError("batch simulators must be NativeSimulator instances")
        if len({id(simulator) for simulator in simulators}) != count:
            raise ValueError("batch simulators must be distinct")
        encoded_actions = [_native_action(*action) for action in actions]
        library, lock_order = NativeSimulator._padded_batch_topology(simulators)
        return NativeSimulator._step_padded_batch_prevalidated(
            simulators,
            encoded_actions,
            buffers,
            workers,
            library,
            lock_order,
        )

    @staticmethod
    def _padded_batch_topology(
        simulators: Sequence[NativeSimulator],
    ) -> tuple[Any, tuple[NativeSimulator, ...]]:
        library = simulators[0]._library
        if any(simulator._library is not library for simulator in simulators):
            raise ValueError("batch simulators must use one native library")
        lock_order = tuple(sorted(simulators, key=lambda value: id(value._lock)))
        return library, lock_order

    @staticmethod
    def _step_padded_batch_prevalidated(
        simulators: Sequence[NativeSimulator],
        encoded_actions: Sequence[tuple[int, float, float, int]],
        buffers: dict[str, Any] | None,
        workers: int,
        library: Any,
        lock_order: Sequence[NativeSimulator],
    ) -> tuple[list[tuple[PaddedTransition, PaddedEvents]], dict[str, Any]]:
        """Run a batch whose fixed topology and actions were already validated."""

        count = len(simulators)
        with ExitStack() as simulator_locks:
            for simulator in lock_order:
                simulator_locks.enter_context(simulator._lock)
            handles = tuple(simulator._require_open() for simulator in simulators)

            valid_buffers = (
                isinstance(buffers, dict)
                and buffers.get("_token") is _PADDED_BUFFER_TOKEN
                and buffers.get("count") == count
                and _is_ctypes_array(buffers.get("handles"), ctypes.c_void_p, count)
                and _is_ctypes_array(buffers.get("actions"), _PaddedAction, count)
                and isinstance(buffers.get("transitions"), list)
                and len(buffers["transitions"]) == 2
                and all(
                    _is_ctypes_array(value, PaddedTransition, count)
                    for value in buffers["transitions"]
                )
                and _is_ctypes_array(buffers.get("statuses"), ctypes.c_uint8, count)
                and type(buffers.get("slot")) is int
                and buffers["slot"] in (0, 1)
                and isinstance(buffers.get("lock"), _RLOCK_TYPE)
            )
            if not valid_buffers:
                buffers = {
                    "_token": _PADDED_BUFFER_TOKEN,
                    "count": count,
                    "handle_values": handles,
                    "handles": (ctypes.c_void_p * count)(*handles),
                    "actions": (_PaddedAction * count)(),
                    "transitions": [
                        (PaddedTransition * count)(),
                        (PaddedTransition * count)(),
                    ],
                    "statuses": (ctypes.c_uint8 * count)(),
                    "slot": 0,
                    "lock": threading.RLock(),
                }

            with buffers["lock"]:
                for index, handle in enumerate(handles):
                    buffers["handles"][index] = handle
                buffers["handle_values"] = handles
                native_actions = buffers["actions"]
                for output, (action_kind, x, y, wait_ticks) in zip(
                    native_actions, encoded_actions
                ):
                    output.x = x
                    output.y = y
                    output.wait_ticks = wait_ticks
                    output.kind = action_kind
                buffers["slot"] ^= 1
                transitions = buffers["transitions"][buffers["slot"]]
                statuses = buffers["statuses"]
                if not library.irisu_padded_step_batch(
                    buffers["handles"],
                    native_actions,
                    transitions,
                    statuses,
                    count,
                    workers,
                ):
                    raise NativeError("native padded batch invocation failed")
                failure: NativeError | None = None
                generations: list[int] = []
                for index, simulator in enumerate(simulators):
                    if statuses[index]:
                        simulator._padded_generation += 1
                    elif failure is None:
                        failure = simulator._error(f"padded batch lane {index}")
                    generations.append(simulator._padded_generation)
                if failure is not None:
                    raise failure
                return (
                    [
                        (
                            transitions[index],
                            PaddedEvents(
                                simulator,
                                int(transitions[index].event_count),
                                generations[index],
                            ),
                        )
                        for index, simulator in enumerate(simulators)
                    ],
                    buffers,
                )

    def _read_padded_events(
        self, event_count: int, generation: int
    ) -> ctypes.Array[PaddedEvent]:
        with self._lock:
            if generation != self._padded_generation:
                raise NativeError(
                    "lazy padded events expired; materialize them before the next step or reset"
                )
            event_type = PaddedEvent * event_count
            events = event_type()
            self._check(
                self._library.irisu_padded_events(
                    self._require_open(), events if event_count else None, event_count
                ),
                "padded events",
            )
            return events

    def observation(self) -> dict[str, Any]:
        with self._lock:
            handle = self._require_open()
            return self._json(self._library.irisu_observation_json(handle), "observation")

    def observation_padded(self) -> PaddedObservation:
        with self._lock:
            destination = PaddedObservation()
            self._check(
                self._library.irisu_padded_observation(
                    self._require_open(), ctypes.byref(destination)
                ),
                "padded observation",
            )
            return destination

    def state_hash(self) -> int:
        with self._lock:
            return int(self._library.irisu_state_hash(self._require_open()))

    def config_hash(self) -> int:
        with self._lock:
            return int(self._library.irisu_config_hash(self._require_open()))

    def config(self) -> dict[str, Any]:
        with self._lock:
            handle = self._require_open()
            return self._json(self._library.irisu_config_json(handle), "config")

    def build_info(self) -> dict[str, Any]:
        return self._json(self._library.irisu_build_info_json(), "build info")

    def clone_state(self) -> bytes:
        with self._lock:
            handle = self._require_open()
            size = int(self._library.irisu_snapshot_size(handle))
            if size <= 0:
                raise self._error("snapshot size")
            buffer = (ctypes.c_ubyte * size)()
            self._check(
                self._library.irisu_snapshot_write(handle, buffer, size),
                "snapshot write",
            )
            return bytes(buffer)

    def restore_state(self, snapshot: bytes | bytearray | memoryview) -> dict[str, Any]:
        data = bytes(snapshot)
        if not data:
            raise NativeError("snapshot must not be empty")
        buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        with self._lock:
            handle = self._require_open()
            self._check(
                self._library.irisu_snapshot_restore(handle, buffer, len(data)),
                "snapshot restore",
            )
            self._padded_generation += 1
            return self.observation()

    def restore_state_padded(
        self, snapshot: bytes | bytearray | memoryview
    ) -> PaddedObservation:
        data = bytes(snapshot)
        if not data:
            raise NativeError("snapshot must not be empty")
        buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
        with self._lock:
            handle = self._require_open()
            self._check(
                self._library.irisu_snapshot_restore(handle, buffer, len(data)),
                "snapshot restore",
            )
            self._padded_generation += 1
            return self.observation_padded()

    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            handle = getattr(self, "_handle", None)
            if handle:
                self._library.irisu_destroy(handle)
                self._handle = None
                self._padded_generation += 1

    def __enter__(self) -> NativeSimulator:
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __copy__(self) -> NativeSimulator:
        raise TypeError("NativeSimulator owns a unique native handle and cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> NativeSimulator:
        del memo
        raise TypeError("NativeSimulator owns a unique native handle and cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
