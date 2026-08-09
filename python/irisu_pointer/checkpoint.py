"""Identity-bound development checkpoints for pointer models."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from irisu_rl.schema import ACTOR_VISION_V1, TEACHER_V1

from .action import PointerActionSpec
from .model import EntityPointerActorCritic, PointerModelConfig
from .policy import RecurrentActorPointerPolicy, RecurrentPointerPolicy


_FORMAT = "irisu-pointer-checkpoint-v2"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    output = {} if value is None else dict(value)
    try:
        encoded = json.dumps(
            output, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint metadata must be canonical JSON data") from exc
    return json.loads(encoded)


@dataclass(frozen=True, slots=True)
class PointerCheckpoint:
    path: Path
    sha256: str
    model: EntityPointerActorCritic
    metadata: Mapping[str, Any]


def save_pointer_checkpoint(
    path: str | Path,
    model: EntityPointerActorCritic,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> str:
    """Atomically save a CPU state dict plus reconstructible identity manifests."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"pointer checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": _FORMAT,
        "model_manifest": model.manifest(),
        "pointer_spec": model.pointer_spec.manifest(),
        "metadata": _json_mapping(metadata),
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
    }
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        torch.save(payload, temporary)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _file_sha256(target)


def load_pointer_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    device: torch.device | str = "cpu",
) -> PointerCheckpoint:
    """Load only the declared format and fail closed on every model identity."""

    source = Path(path).resolve(strict=True)
    digest = _file_sha256(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("pointer checkpoint SHA-256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != {
        "format",
        "model_manifest",
        "pointer_spec",
        "metadata",
        "state_dict",
    }:
        raise ValueError("pointer checkpoint has unknown or missing fields")
    if payload["format"] != _FORMAT:
        raise ValueError("pointer checkpoint format is unsupported")
    manifest = payload["model_manifest"]
    pointer_value = payload["pointer_spec"]
    if not isinstance(manifest, dict) or not isinstance(pointer_value, dict):
        raise ValueError("pointer checkpoint manifests are malformed")
    schemas = {
        TEACHER_V1.sha256: TEACHER_V1,
        ACTOR_VISION_V1.sha256: ACTOR_VISION_V1,
    }
    schema = schemas.get(manifest.get("schema_sha256"))
    if schema is None:
        raise ValueError("pointer checkpoint observation schema is unknown")
    try:
        pointer_spec = PointerActionSpec(
            wait_choices=tuple(pointer_value["wait_choices"]),
            x_radius_offsets=tuple(pointer_value["x_radius_offsets"]),
            y_radius_offsets=tuple(pointer_value["y_radius_offsets"]),
        )
        config = PointerModelConfig(**manifest["config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pointer checkpoint model configuration is invalid") from exc
    if pointer_spec.manifest() != pointer_value:
        raise ValueError("reconstructed pointer action identity differs")
    model = EntityPointerActorCritic(
        schema,
        pointer_spec=pointer_spec,
        config=config,
    )
    if model.manifest() != manifest:
        raise ValueError("reconstructed pointer model identity differs")
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise ValueError("pointer checkpoint state dict is malformed")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    metadata = _json_mapping(payload["metadata"])
    return PointerCheckpoint(source, digest, model, metadata)


def load_teacher_pointer_policy(
    path: str | Path,
    *,
    expected_sha256: str,
    device: torch.device | str = "cpu",
) -> RecurrentPointerPolicy:
    checkpoint = load_pointer_checkpoint(
        path, expected_sha256=expected_sha256, device=device
    )
    if checkpoint.model.schema.sha256 != TEACHER_V1.sha256:
        raise ValueError("teacher policy loader cannot load an actor-schema model")
    checkpoint.model.eval()
    return RecurrentPointerPolicy(
        checkpoint.model,
        artifact_sha256=checkpoint.sha256,
    )


def load_actor_pointer_policy(
    path: str | Path,
    *,
    expected_sha256: str,
    device: torch.device | str = "cpu",
) -> RecurrentActorPointerPolicy:
    """Load an identity-bound actor-track policy ready for primitive lowering."""

    checkpoint = load_pointer_checkpoint(
        path, expected_sha256=expected_sha256, device=device
    )
    if checkpoint.model.schema.sha256 != ACTOR_VISION_V1.sha256:
        raise ValueError("actor policy loader cannot load a teacher-schema model")
    checkpoint.model.eval()
    return RecurrentActorPointerPolicy(
        checkpoint.model,
        artifact_sha256=checkpoint.sha256,
    )


__all__ = [
    "PointerCheckpoint",
    "load_actor_pointer_policy",
    "load_pointer_checkpoint",
    "load_teacher_pointer_policy",
    "save_pointer_checkpoint",
]
