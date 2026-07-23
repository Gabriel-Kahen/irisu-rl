"""Fail-closed simulator binary and build-information attestation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from irisu_env.exact_ipc import ExactSimulator


_ZERO_SHA256 = "0" * 64
_VOLATILE_BUILD_INFO_KEYS = frozenset({"config_hash", "worker_pid"})


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value != _ZERO_SHA256
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True, slots=True)
class _CapturedArtifact:
    path: str
    sha256: str
    bytes: int


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _capture_artifact(path_value: object, label: str) -> _CapturedArtifact:
    if not isinstance(path_value, (str, os.PathLike)):
        raise TypeError(f"{label} path must be path-like")
    supplied = Path(path_value).expanduser()
    if not supplied.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    try:
        path = supplied.resolve(strict=True)
        if not path.is_file():
            raise ValueError(f"{label} path must name a regular file")
        digest = hashlib.sha256()
        byte_count = 0
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            while block := stream.read(1 << 20):
                byte_count += len(block)
                digest.update(block)
            after = os.fstat(stream.fileno())
        current = path.stat()
    except (OSError, RuntimeError) as exc:
        raise RuntimeError(f"cannot capture {label} artifact: {exc}") from exc
    if (
        _stat_identity(before) != _stat_identity(after)
        or _stat_identity(after) != _stat_identity(current)
        or byte_count != after.st_size
    ):
        raise RuntimeError(f"{label} artifact changed while it was captured")
    return _CapturedArtifact(str(path), digest.hexdigest(), byte_count)


def _read_member(owner: object, name: str) -> object:
    try:
        value = getattr(owner, name)
    except AttributeError as exc:
        raise TypeError(f"simulator does not expose {name}") from exc
    return value() if callable(value) else value


def _normalized_build_info(simulator: object) -> tuple[dict[str, Any], str, int]:
    supplied = _read_member(simulator, "build_info")
    if not isinstance(supplied, Mapping) or any(
        not isinstance(key, str) for key in supplied
    ):
        raise TypeError("simulator build_info must be a string-keyed mapping")
    normalized = {
        key: value
        for key, value in supplied.items()
        if key not in _VOLATILE_BUILD_INFO_KEYS
    }
    try:
        encoded = _canonical_json(normalized)
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError("simulator build_info must be finite canonical JSON") from exc
    backend_value = normalized.get("physics_backend")
    snapshot_schema = normalized.get("snapshot_schema")
    if not isinstance(backend_value, str) or not backend_value:
        raise ValueError("simulator build_info lacks a backend identity")
    if (
        isinstance(snapshot_schema, bool)
        or not isinstance(snapshot_schema, int)
        or snapshot_schema <= 0
    ):
        raise ValueError("simulator build_info lacks a snapshot schema")
    if backend_value.startswith("portable-"):
        backend = "portable"
    elif backend_value.startswith("exact-"):
        backend = "exact"
    else:
        raise ValueError("simulator build_info names an unsupported backend")
    return normalized, backend, snapshot_schema


def _provenance_artifact(simulator: object, expected_sha256: str) -> _CapturedArtifact:
    supplied = _read_member(simulator, "exact_library_provenance")
    if not isinstance(supplied, Mapping):
        raise TypeError("exact-library provenance must be a mapping")
    if supplied.get("status") != "captured":
        raise RuntimeError("exact-library provenance was not captured")
    captured = _capture_artifact(supplied.get("path"), "exact library")
    byte_count = supplied.get("bytes")
    supplied_sha256 = supplied.get("sha256")
    if (
        isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count != captured.bytes
        or supplied_sha256 != captured.sha256
        or captured.sha256 != expected_sha256
    ):
        raise RuntimeError("mapped exact-library provenance disagrees with its bytes")
    file_identity = supplied.get("file_identity")
    mapped_identity = supplied.get("mapped_identity")
    if not isinstance(file_identity, Mapping) or not isinstance(
        mapped_identity, Mapping
    ):
        raise RuntimeError("mapped exact-library provenance lacks file identity")
    stat = Path(captured.path).stat()
    expected_file = {
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }
    if any(file_identity.get(key) != value for key, value in expected_file.items()):
        raise RuntimeError("mapped exact-library file identity changed")
    mapped_inode = mapped_identity.get("inode")
    if (
        isinstance(mapped_inode, bool)
        or not isinstance(mapped_inode, int)
        or mapped_inode != stat.st_ino
    ):
        raise RuntimeError("mapped exact-library inode disagrees with its path")
    return captured


def _portable_artifact(simulator: object) -> _CapturedArtifact:
    provenance_owner = getattr(simulator, "_native", simulator)
    if not hasattr(provenance_owner, "portable_library_provenance"):
        return _capture_artifact(
            _read_member(simulator, "library_path"), "portable library"
        )
    supplied = _read_member(provenance_owner, "portable_library_provenance")
    if not isinstance(supplied, Mapping) or supplied.get("status") != "captured":
        raise RuntimeError("portable-library provenance was not captured")
    path = supplied.get("path")
    byte_count = supplied.get("bytes")
    sha256 = supplied.get("sha256")
    file_identity = supplied.get("file_identity")
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or isinstance(byte_count, bool)
        or not isinstance(byte_count, int)
        or byte_count <= 0
        or not _is_sha256(sha256)
        or not isinstance(file_identity, Mapping)
        or any(
            isinstance(file_identity.get(key), bool)
            or not isinstance(file_identity.get(key), int)
            for key in ("device", "inode", "mtime_ns", "ctime_ns")
        )
        or file_identity["inode"] <= 0
    ):
        raise RuntimeError("portable-library provenance is malformed")
    return _CapturedArtifact(path, sha256, byte_count)


@dataclass(frozen=True, slots=True)
class SimulatorRuntimeAttestation:
    """Canonical runtime identity plus the paths measured for every lane.

    Paths and lane count are audit evidence, not identity inputs: relocating
    byte-identical artifacts does not change the runtime SHA-256.
    """

    backend: str
    snapshot_schema: int
    build_info_json: str
    runtime_artifact_kind: str
    runtime_artifact_sha256: str
    runtime_artifact_bytes: int
    exact_library_sha256: str | None
    exact_library_bytes: int | None
    runtime_artifact_paths: tuple[str, ...]
    exact_library_paths: tuple[str, ...]
    version: str = "simulator-runtime-attestation-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "simulator-runtime-attestation-v1"
            or self.backend not in {"portable", "exact"}
            or isinstance(self.snapshot_schema, bool)
            or not isinstance(self.snapshot_schema, int)
            or self.snapshot_schema <= 0
            or self.runtime_artifact_kind
            != ("shared-library" if self.backend == "portable" else "worker-executable")
            or not _is_sha256(self.runtime_artifact_sha256)
            or isinstance(self.runtime_artifact_bytes, bool)
            or not isinstance(self.runtime_artifact_bytes, int)
            or self.runtime_artifact_bytes <= 0
            or not isinstance(self.runtime_artifact_paths, tuple)
            or not self.runtime_artifact_paths
            or any(
                not isinstance(path, str) or not Path(path).is_absolute()
                for path in self.runtime_artifact_paths
            )
            or not isinstance(self.exact_library_paths, tuple)
        ):
            raise ValueError("simulator runtime attestation is malformed")
        try:
            build_info = json.loads(self.build_info_json)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError("runtime build-info identity is malformed") from exc
        if (
            not isinstance(build_info, dict)
            or _canonical_json(build_info) != self.build_info_json
            or any(key in build_info for key in _VOLATILE_BUILD_INFO_KEYS)
            or build_info.get("snapshot_schema") != self.snapshot_schema
            or not isinstance(build_info.get("physics_backend"), str)
            or not build_info["physics_backend"].startswith(f"{self.backend}-")
        ):
            raise ValueError("runtime build-info identity is not canonical")
        if self.backend == "portable":
            if (
                self.exact_library_sha256 is not None
                or self.exact_library_bytes is not None
                or self.exact_library_paths
            ):
                raise ValueError("portable runtime cannot carry exact-library evidence")
        elif (
            not _is_sha256(self.exact_library_sha256)
            or isinstance(self.exact_library_bytes, bool)
            or not isinstance(self.exact_library_bytes, int)
            or self.exact_library_bytes <= 0
            or len(self.exact_library_paths) != len(self.runtime_artifact_paths)
            or any(
                not isinstance(path, str) or not Path(path).is_absolute()
                for path in self.exact_library_paths
            )
            or build_info.get("worker_executable_sha256")
            != self.runtime_artifact_sha256
            or build_info.get("exact_library_sha256") != self.exact_library_sha256
            or build_info.get("exact_library_runtime_verified") is not True
            or build_info.get("exact_call_targets_runtime_verified") is not True
        ):
            raise ValueError("exact runtime lacks mapped-library evidence")

    @property
    def verified_lanes(self) -> int:
        return len(self.runtime_artifact_paths)

    @property
    def build_info(self) -> dict[str, Any]:
        return json.loads(self.build_info_json)

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "backend": self.backend,
            "snapshot_schema": self.snapshot_schema,
            "build_info": self.build_info,
            "runtime_artifact": {
                "kind": self.runtime_artifact_kind,
                "sha256": self.runtime_artifact_sha256,
                "bytes": self.runtime_artifact_bytes,
            },
            "exact_library": (
                None
                if self.backend == "portable"
                else {
                    "sha256": self.exact_library_sha256,
                    "bytes": self.exact_library_bytes,
                }
            ),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.manifest()).encode()).hexdigest()

    def evidence_manifest(self) -> dict[str, object]:
        return {
            **self.manifest(),
            "verified_lanes": self.verified_lanes,
            "runtime_artifact_paths": list(self.runtime_artifact_paths),
            "exact_library_paths": list(self.exact_library_paths),
        }


def _attest_lane(simulator: object) -> SimulatorRuntimeAttestation:
    build_info, backend, snapshot_schema = _normalized_build_info(simulator)
    runtime = (
        _portable_artifact(simulator)
        if backend == "portable"
        else _capture_artifact(_read_member(simulator, "worker_path"), "exact worker")
    )
    exact_library: _CapturedArtifact | None = None
    if backend == "exact":
        worker_sha256 = build_info.get("worker_executable_sha256")
        exact_library_sha256 = build_info.get("exact_library_sha256")
        if not _is_sha256(worker_sha256) or worker_sha256 != runtime.sha256:
            raise RuntimeError("exact worker bytes disagree with build_info")
        if not _is_sha256(exact_library_sha256):
            raise RuntimeError("exact build_info lacks a library SHA-256")
        if (
            build_info.get("exact_library_runtime_verified") is not True
            or build_info.get("exact_call_targets_runtime_verified") is not True
        ):
            raise RuntimeError("exact runtime did not verify its mapped call targets")
        exact_library = _provenance_artifact(simulator, exact_library_sha256)
    return SimulatorRuntimeAttestation(
        backend=backend,
        snapshot_schema=snapshot_schema,
        build_info_json=_canonical_json(build_info),
        runtime_artifact_kind=(
            "shared-library" if backend == "portable" else "worker-executable"
        ),
        runtime_artifact_sha256=runtime.sha256,
        runtime_artifact_bytes=runtime.bytes,
        exact_library_sha256=(None if exact_library is None else exact_library.sha256),
        exact_library_bytes=None if exact_library is None else exact_library.bytes,
        runtime_artifact_paths=(runtime.path,),
        exact_library_paths=(() if exact_library is None else (exact_library.path,)),
    )


def attest_simulator_runtime(simulator: object) -> SimulatorRuntimeAttestation:
    """Measure a simulator or homogeneous vector and return its runtime identity."""

    lanes_value = getattr(simulator, "envs", None)
    if lanes_value is None:
        lanes = (simulator,)
    else:
        lanes_value = lanes_value() if callable(lanes_value) else lanes_value
        if (
            not isinstance(lanes_value, Sequence)
            or isinstance(lanes_value, (str, bytes, bytearray))
            or not lanes_value
        ):
            raise ValueError("simulator vector must expose a nonempty lane sequence")
        lanes = tuple(lanes_value)
        declared = getattr(simulator, "num_envs", len(lanes))
        declared = declared() if callable(declared) else declared
        if (
            isinstance(declared, bool)
            or not isinstance(declared, int)
            or declared != len(lanes)
        ):
            raise ValueError("simulator vector lane count is inconsistent")
    attestations = tuple(_attest_lane(lane) for lane in lanes)
    first = attestations[0]
    if any(value.sha256 != first.sha256 for value in attestations[1:]):
        raise RuntimeError("simulator vector lanes have heterogeneous runtimes")
    return SimulatorRuntimeAttestation(
        backend=first.backend,
        snapshot_schema=first.snapshot_schema,
        build_info_json=first.build_info_json,
        runtime_artifact_kind=first.runtime_artifact_kind,
        runtime_artifact_sha256=first.runtime_artifact_sha256,
        runtime_artifact_bytes=first.runtime_artifact_bytes,
        exact_library_sha256=first.exact_library_sha256,
        exact_library_bytes=first.exact_library_bytes,
        runtime_artifact_paths=tuple(
            value.runtime_artifact_paths[0] for value in attestations
        ),
        exact_library_paths=tuple(
            value.exact_library_paths[0]
            for value in attestations
            if value.exact_library_paths
        ),
    )


@dataclass(frozen=True, slots=True)
class ExactRuntimeIdentity:
    worker_sha256: str
    exact_library_sha256: str
    protocol_version: int = 1
    body_capacity: int = 196
    pointer_bits: int = 32
    backend: str = "exact-msvc9-r58-multiworld-forward"

    def attest(self, worker_path: str | Path) -> dict[str, object]:
        supplied = Path(worker_path).expanduser()
        if not supplied.is_absolute():
            raise ValueError("exact worker path must be absolute")
        path = supplied.resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.worker_sha256:
            raise RuntimeError("exact worker executable hash mismatch")
        with ExactSimulator(worker_path=path) as simulator:
            info = simulator.build_info()
            provenance = simulator.exact_library_provenance()
        expected = {
            "worker_executable_sha256": self.worker_sha256,
            "exact_library_sha256": self.exact_library_sha256,
            "protocol_version": self.protocol_version,
            "body_capacity": self.body_capacity,
            "pointer_bits": self.pointer_bits,
            "worker_backend": self.backend,
        }
        mismatches = {
            key: (info.get(key), value)
            for key, value in expected.items()
            if info.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"exact runtime identity mismatch: {mismatches}")
        if provenance.get("sha256") != self.exact_library_sha256:
            raise RuntimeError("mapped exact library provenance mismatch")
        return {"worker_path": str(path), "build_info": info, "provenance": provenance}


ACCEPTED_EXACT_RUNTIME_2026_07_21 = ExactRuntimeIdentity(
    worker_sha256="4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261",
    exact_library_sha256="ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5",
)
