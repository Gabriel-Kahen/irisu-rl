"""Immutable, checksummed training checkpoints with atomic publication."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .actions import ActionSpec
from .encoding import EncodedBatch
from .schema import TensorSchema
from .vector_adapter import AdapterCheckpoint


CHECKPOINT_VERSION = "irisu-r2-checkpoint-v1"
_TYPE_MARKER = "__irisu_checkpoint_type__"


def _weights_only_safe(value: object, *, path: str = "state") -> object:
    """Normalize a state tree to the subset accepted by weights-only loading."""

    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError(f"{path} contains an object NumPy array")
        return {
            _TYPE_MARKER: "numpy.ndarray",
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "data": torch.from_numpy(np.ascontiguousarray(value)),
        }
    if isinstance(value, np.generic):
        return _weights_only_safe(value.item(), path=path)
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, Mapping):
        if _TYPE_MARKER in value:
            raise ValueError(f"{path} uses a reserved checkpoint key")
        normalized: dict[object, object] = {}
        for key, item in value.items():
            if not isinstance(key, (bool, int, float, str)):
                raise TypeError(f"{path} contains an unsupported mapping key")
            normalized[key] = _weights_only_safe(item, path=f"{path}[{key!r}]")
        return normalized
    if isinstance(value, tuple):
        return tuple(
            _weights_only_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, list):
        return [
            _weights_only_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} contains unsupported type {type(value).__name__}")


def _restore_normalized(value: object) -> object:
    if isinstance(value, dict):
        marker = value.get(_TYPE_MARKER)
        if marker is not None:
            if marker != "numpy.ndarray" or set(value) != {
                _TYPE_MARKER,
                "dtype",
                "shape",
                "data",
            }:
                raise ValueError("checkpoint contains an invalid normalized value")
            data = value["data"]
            shape = value["shape"]
            dtype = value["dtype"]
            if not isinstance(data, torch.Tensor) or not isinstance(shape, list):
                raise ValueError("checkpoint NumPy array payload is malformed")
            array = data.cpu().numpy().astype(np.dtype(dtype), copy=True)
            if tuple(array.shape) != tuple(shape):
                raise ValueError("checkpoint NumPy array shape mismatch")
            return array
        return {key: _restore_normalized(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_restore_normalized(item) for item in value)
    if isinstance(value, list):
        return [_restore_normalized(item) for item in value]
    return value


def _validate_generation(generation: object) -> str:
    if (
        not isinstance(generation, str)
        or not generation
        or generation in {".", ".."}
        or Path(generation).name != generation
        or any(separator in generation for separator in ("/", "\\"))
    ):
        raise ValueError("checkpoint generation must be one safe path component")
    return generation


def capture_rng_state(numpy_generator: Any) -> dict[str, object]:
    """Capture independent Python, NumPy-generator, and Torch RNG streams."""

    return {
        "python": random.getstate(),
        "numpy_bit_generator": numpy_generator.bit_generator.__class__.__name__,
        "numpy": numpy_generator.bit_generator.state,
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def _nested_tuple(value: object) -> object:
    return (
        tuple(_nested_tuple(item) for item in value)
        if isinstance(value, list)
        else value
    )


def restore_rng_state(state: Mapping[str, object], numpy_generator: Any) -> None:
    expected = {"python", "numpy_bit_generator", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != expected:
        raise ValueError("RNG state keys do not match the checkpoint version")
    if state["numpy_bit_generator"] != numpy_generator.bit_generator.__class__.__name__:
        raise ValueError("NumPy bit-generator identity mismatch")
    if not isinstance(state["torch_cpu"], torch.Tensor):
        raise ValueError("Torch CPU RNG state is malformed")
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint requires CUDA RNG state on a CPU-only runtime")
    previous_python = random.getstate()
    previous_numpy = copy.deepcopy(numpy_generator.bit_generator.state)
    previous_torch = torch.get_rng_state()
    previous_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        random.setstate(_nested_tuple(state["python"]))
        numpy_generator.bit_generator.state = state["numpy"]
        torch.set_rng_state(state["torch_cpu"])
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
    except BaseException:
        random.setstate(previous_python)
        numpy_generator.bit_generator.state = previous_numpy
        torch.set_rng_state(previous_torch)
        if previous_cuda:
            torch.cuda.set_rng_state_all(previous_cuda)
        raise


def pack_adapter_checkpoint(
    checkpoint: AdapterCheckpoint,
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Convert an adapter checkpoint into weights-only-safe state and blobs."""

    current = checkpoint.current
    state: dict[str, object] = {
        "version": checkpoint.version,
        "schema_sha256": checkpoint.schema_sha256,
        "action_sha256": checkpoint.action_sha256,
        "num_envs": checkpoint.num_envs,
        "capture_events": checkpoint.capture_events,
        "raw_ticks": list(checkpoint.raw_ticks),
        "raw_scores": list(checkpoint.raw_scores),
        "seeds": list(checkpoint.seeds),
        "episode_ids": list(checkpoint.episode_ids),
        "seed_allocator": checkpoint.seed_allocator,
        "state_hashes": list(checkpoint.state_hashes),
        "snapshot_names": [
            f"lane-{lane:04d}.snapshot" for lane in range(checkpoint.num_envs)
        ],
        "current": {
            "global_features": torch.from_numpy(current.global_features.copy()),
            "body_features": torch.from_numpy(current.body_features.copy()),
            "body_mask": torch.from_numpy(current.body_mask.copy()),
            "source_tick": torch.from_numpy(current.source_tick.copy()),
            "health_flags": torch.from_numpy(current.health_flags.copy()),
        },
    }
    blobs = {
        name: snapshot
        for name, snapshot in zip(state["snapshot_names"], checkpoint.snapshots)
    }
    return state, blobs


def unpack_adapter_checkpoint(
    state: Mapping[str, object],
    blobs: Mapping[str, bytes],
    *,
    schema: TensorSchema,
    action_spec: ActionSpec,
) -> AdapterCheckpoint:
    expected = {
        "version",
        "schema_sha256",
        "action_sha256",
        "num_envs",
        "capture_events",
        "raw_ticks",
        "raw_scores",
        "seeds",
        "episode_ids",
        "seed_allocator",
        "state_hashes",
        "snapshot_names",
        "current",
    }
    if set(state) != expected:
        raise ValueError("adapter checkpoint state keys do not match the version")
    if state["version"] != "macro-vector-adapter-checkpoint-v2":
        raise ValueError("adapter checkpoint version mismatch")
    if (
        state["schema_sha256"] != schema.sha256
        or state["action_sha256"] != action_spec.sha256
    ):
        raise ValueError("adapter checkpoint schema/action identity mismatch")
    names = state["snapshot_names"]
    lane_count = state["num_envs"]
    if (
        isinstance(lane_count, bool)
        or not isinstance(lane_count, int)
        or lane_count <= 0
    ):
        raise ValueError("adapter checkpoint lane count is invalid")
    capture_events = state["capture_events"]
    if not isinstance(capture_events, bool):
        raise ValueError("adapter checkpoint event-capture mode is invalid")
    canonical_names = [f"lane-{lane:04d}.snapshot" for lane in range(lane_count)]
    if names != canonical_names or set(names) != set(blobs):
        raise ValueError("adapter checkpoint snapshot set mismatch")
    current = state["current"]
    if not isinstance(current, dict):
        raise ValueError("adapter current observation payload is malformed")
    required_current = {
        "global_features",
        "body_features",
        "body_mask",
        "source_tick",
        "health_flags",
    }
    if set(current) != required_current or not all(
        isinstance(current[name], torch.Tensor) for name in required_current
    ):
        raise ValueError("adapter observation tensor set is malformed")
    encoded = EncodedBatch(
        current["global_features"].numpy().copy(),
        current["body_features"].numpy().copy(),
        current["body_mask"].numpy().copy(),
        current["source_tick"].numpy().copy(),
        current["health_flags"].numpy().copy(),
        schema,
    )
    return AdapterCheckpoint(
        str(state["version"]),
        str(state["schema_sha256"]),
        str(state["action_sha256"]),
        lane_count,
        capture_events,
        encoded,
        tuple(int(value) for value in state["raw_ticks"]),
        tuple(int(value) for value in state["raw_scores"]),
        tuple(int(value) for value in state["seeds"]),
        tuple(int(value) for value in state["episode_ids"]),
        dict(state["seed_allocator"]),
        tuple(blobs[name] for name in names),
        tuple(int(value) for value in state["state_hashes"]),
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_bytes(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_checkpoint(
    root: str | os.PathLike[str],
    generation: str,
    *,
    identity: Mapping[str, object],
    state: Mapping[str, Any],
    blobs: Mapping[str, bytes] | None = None,
) -> Path:
    """Publish one immutable generation and atomically move ``latest.json``.

    Callers may checkpoint only at a complete semantic decision boundary and
    after a complete optimizer update. Existing generations are never replaced.
    """

    generation = _validate_generation(generation)
    root_path = Path(root).resolve()
    root_path.mkdir(parents=True, exist_ok=True)
    destination = root_path / generation
    if destination.exists():
        raise FileExistsError(f"checkpoint generation already exists: {generation}")
    temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=root_path))
    pointer_temp: Path | None = None
    try:
        state_path = temporary / "state.pt"
        normalized_state = _weights_only_safe(dict(state))
        with state_path.open("xb") as stream:
            torch.save(normalized_state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        verified = torch.load(state_path, map_location="cpu", weights_only=True)
        if not isinstance(verified, dict):
            raise ValueError("checkpoint state payload must be a mapping")
        files: dict[str, str] = {"state.pt": _sha256(state_path)}
        for name, payload in sorted((blobs or {}).items()):
            if Path(name).name != name or not name:
                raise ValueError("checkpoint blob names must be safe path components")
            blob_path = temporary / name
            _write_bytes(blob_path, bytes(payload))
            files[name] = _sha256(blob_path)
        manifest = {
            "version": CHECKPOINT_VERSION,
            "generation": generation,
            "identity": dict(identity),
            "files": files,
        }
        _write_bytes(temporary / "manifest.json", _canonical_json(manifest) + b"\n")
        _fsync_directory(temporary)
        os.rename(temporary, destination)
        _fsync_directory(root_path)
        pointer = {
            "version": CHECKPOINT_VERSION,
            "generation": generation,
            "manifest_sha256": _sha256(destination / "manifest.json"),
        }
        descriptor, pointer_name = tempfile.mkstemp(prefix=".latest-", dir=root_path)
        pointer_temp = Path(pointer_name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(_canonical_json(pointer) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(pointer_temp, root_path / "latest.json")
        pointer_temp = None
        _fsync_directory(root_path)
        return destination
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        if pointer_temp is not None:
            pointer_temp.unlink(missing_ok=True)
        raise


def load_checkpoint(
    root: str | os.PathLike[str],
    *,
    generation: str | None = None,
    expected_identity: Mapping[str, object],
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, object]]:
    """Validate all identities and hashes before deserializing trainer state."""

    root_path = Path(root).resolve()
    pointer_manifest_hash: str | None = None
    if generation is None:
        pointer_path = root_path / "latest.json"
        pointer_bytes = pointer_path.read_bytes()
        pointer = json.loads(pointer_bytes)
        if pointer.get("version") != CHECKPOINT_VERSION:
            raise ValueError("checkpoint pointer version mismatch")
        generation = pointer.get("generation")
        pointer_manifest_hash = pointer.get("manifest_sha256")
        generation = _validate_generation(generation)
        if not isinstance(pointer_manifest_hash, str):
            raise ValueError("checkpoint pointer manifest hash is invalid")
    generation = _validate_generation(generation)
    directory = root_path / generation
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("checkpoint generation must be a real directory")
    manifest_path = directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    if (
        pointer_manifest_hash is not None
        and hashlib.sha256(manifest_bytes).hexdigest() != pointer_manifest_hash
    ):
        raise ValueError("checkpoint pointer does not match the generation manifest")
    manifest = json.loads(manifest_bytes)
    if (
        manifest.get("version") != CHECKPOINT_VERSION
        or manifest.get("generation") != generation
    ):
        raise ValueError("checkpoint manifest identity mismatch")
    if manifest.get("identity") != dict(expected_identity):
        raise ValueError("checkpoint runtime/configuration identity mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or "state.pt" not in files:
        raise ValueError("checkpoint file manifest is incomplete")
    for name, expected_hash in files.items():
        if Path(name).name != name or not isinstance(expected_hash, str):
            raise ValueError("checkpoint file manifest is invalid")
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"checkpoint file is missing or unsafe: {name}")
        if _sha256(path) != expected_hash:
            raise ValueError(f"checkpoint file hash mismatch: {name}")
    state = torch.load(directory / "state.pt", map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("checkpoint state payload must be a mapping")
    state = _restore_normalized(state)
    blobs = {
        name: (directory / name).read_bytes() for name in files if name != "state.pt"
    }
    return state, blobs, manifest
