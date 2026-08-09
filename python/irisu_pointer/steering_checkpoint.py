"""Identity-bound checkpoints for goal-conditioned steering models."""

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
from .steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
    SteeringModelConfig,
)


_FORMAT = "irisu-goal-conditioned-steering-checkpoint-v1"


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
        raise ValueError("steering checkpoint metadata must be canonical JSON") from exc
    return json.loads(encoded)


@dataclass(frozen=True, slots=True)
class SteeringCheckpoint:
    path: Path
    sha256: str
    model: GoalConditionedSteeringModel
    metadata: Mapping[str, Any]


def save_steering_checkpoint(
    path: str | Path,
    model: GoalConditionedSteeringModel,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> str:
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"steering checkpoint already exists: {target}")
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


def load_steering_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    device: torch.device | str = "cpu",
) -> SteeringCheckpoint:
    source = Path(path).resolve(strict=True)
    digest = _file_sha256(source)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("steering checkpoint SHA-256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    expected_fields = {
        "format",
        "model_manifest",
        "pointer_spec",
        "metadata",
        "state_dict",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("steering checkpoint has unknown or missing fields")
    if payload["format"] != _FORMAT:
        raise ValueError("steering checkpoint format is unsupported")
    manifest = payload["model_manifest"]
    pointer_value = payload["pointer_spec"]
    if not isinstance(manifest, dict) or not isinstance(pointer_value, dict):
        raise ValueError("steering checkpoint manifests are malformed")
    schema = {
        TEACHER_V1.sha256: TEACHER_V1,
        ACTOR_VISION_V1.sha256: ACTOR_VISION_V1,
    }.get(manifest.get("schema_sha256"))
    if schema is None:
        raise ValueError("steering checkpoint observation schema is unknown")
    try:
        pointer_spec = PointerActionSpec(
            wait_choices=tuple(pointer_value["wait_choices"]),
            x_radius_offsets=tuple(pointer_value["x_radius_offsets"]),
            y_radius_offsets=tuple(pointer_value["y_radius_offsets"]),
        )
        config = SteeringModelConfig(**manifest["config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("steering checkpoint configuration is invalid") from exc
    if pointer_spec.manifest() != pointer_value:
        raise ValueError("reconstructed steering action identity differs")
    model = GoalConditionedSteeringModel(
        schema, pointer_spec=pointer_spec, config=config
    )
    if model.manifest() != manifest:
        raise ValueError("reconstructed steering model identity differs")
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise ValueError("steering checkpoint state dict is malformed")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    return SteeringCheckpoint(
        source, digest, model, _json_mapping(payload["metadata"])
    )


def load_goal_conditioned_steering_policy(
    path: str | Path,
    *,
    expected_sha256: str,
    device: torch.device | str = "cpu",
    cooldown_ticks: int = 16,
    minimum_pair_closure_sizes: float = 0.05,
    impact_side_sizes: float = 0.5,
    impact_below_sizes: float = 0.75,
    source_velocity_lead_ticks: float = 1.0,
    ticks_per_second: float = 50.0,
    act_logit_bias: float = 0.0,
) -> GoalConditionedSteeringPolicy:
    checkpoint = load_steering_checkpoint(
        path, expected_sha256=expected_sha256, device=device
    )
    if checkpoint.model.schema.sha256 != TEACHER_V1.sha256:
        raise ValueError("teacher steering policy requires the teacher schema")
    checkpoint.model.eval()
    return GoalConditionedSteeringPolicy(
        checkpoint.model,
        cooldown_ticks=cooldown_ticks,
        minimum_pair_closure_sizes=minimum_pair_closure_sizes,
        impact_side_sizes=impact_side_sizes,
        impact_below_sizes=impact_below_sizes,
        source_velocity_lead_ticks=source_velocity_lead_ticks,
        ticks_per_second=ticks_per_second,
        act_logit_bias=act_logit_bias,
        artifact_sha256=checkpoint.sha256,
    )


__all__ = [
    "SteeringCheckpoint",
    "load_goal_conditioned_steering_policy",
    "load_steering_checkpoint",
    "save_steering_checkpoint",
]
