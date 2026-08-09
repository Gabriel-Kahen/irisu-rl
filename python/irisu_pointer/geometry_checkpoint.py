"""Fail-closed checkpoints for safeguarded geometry deployment."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from irisu_rl.schema import ACTOR_VISION_V1, TEACHER_V1

from .geometry_learning import GeometryModelConfig, GeometrySelectorModel
from .geometry_policy import (
    GeometryPolicyConfig,
    GeometrySelectorEnsemble,
    SafeguardedGeometryPolicy,
    geometry_candidate_vocabulary_manifest,
    geometry_candidate_vocabulary_sha256,
)
from .geometry_search import GeometrySearchConfig
from .steering_learning import GoalConditionedSteeringPolicy


_FORMAT = "irisu-safeguarded-geometry-checkpoint-v1"
_BINDING_FIELDS = {
    "schema_sha256",
    "geometry_config_sha256",
    "candidate_vocabulary_sha256",
    "architecture_sha256",
    "base_policy_checkpoint_sha256",
    "source_identity",
}


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_mapping(
    value: Mapping[str, Any] | None, name: str
) -> dict[str, Any]:
    output = {} if value is None else dict(value)
    try:
        encoded = json.dumps(
            output, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc
    return json.loads(encoded)


def _geometry_config(value: object) -> GeometrySearchConfig:
    if not isinstance(value, dict):
        raise ValueError("geometry search config manifest is malformed")
    try:
        config = GeometrySearchConfig(
            horizon_ticks=value["horizon_ticks"],
            velocity_lead_ticks=value["velocity_lead_ticks"],
            ticks_per_second=value["ticks_per_second"],
            support_fractions=tuple(value["support_fractions"]),
            support_clearance_sizes=value["support_clearance_sizes"],
            grid_x_fractions=tuple(value["grid_x_fractions"]),
            grid_y_sizes=tuple(value["grid_y_sizes"]),
            max_candidates=value["max_candidates"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("geometry search config is invalid") from exc
    if config.manifest() != value:
        raise ValueError("reconstructed geometry search identity differs")
    return config


@dataclass(frozen=True, slots=True)
class GeometryCheckpoint:
    path: Path
    sha256: str
    model: GeometrySelectorModel
    geometry_config: GeometrySearchConfig
    policy_config: GeometryPolicyConfig
    base_policy_checkpoint_sha256: str
    source_identity: str
    metadata: Mapping[str, Any]


def save_geometry_checkpoint(
    path: str | Path,
    model: GeometrySelectorModel,
    *,
    geometry_config: GeometrySearchConfig,
    policy_config: GeometryPolicyConfig,
    base_policy_checkpoint_sha256: str,
    source_identity: str,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> str:
    if not isinstance(model, GeometrySelectorModel):
        raise TypeError("geometry model must be a GeometrySelectorModel")
    if not isinstance(geometry_config, GeometrySearchConfig):
        raise TypeError("geometry config must be a GeometrySearchConfig")
    if not isinstance(policy_config, GeometryPolicyConfig):
        raise TypeError("policy config must be a GeometryPolicyConfig")
    base_sha256 = _sha256(
        base_policy_checkpoint_sha256, "base-policy checkpoint"
    )
    source_sha256 = _sha256(source_identity, "source identity")
    vocabulary_sha256 = geometry_candidate_vocabulary_sha256(geometry_config)
    if model.candidate_count != geometry_config.max_candidates:
        raise ValueError("model width and geometry candidate vocabulary differ")
    if model.candidate_set_sha256 != vocabulary_sha256:
        raise ValueError("model and geometry candidate identities differ")

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"geometry checkpoint already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    bindings = {
        "schema_sha256": model.schema.sha256,
        "geometry_config_sha256": geometry_config.sha256,
        "candidate_vocabulary_sha256": vocabulary_sha256,
        "architecture_sha256": model.architecture_sha256,
        "base_policy_checkpoint_sha256": base_sha256,
        "source_identity": source_sha256,
    }
    payload = {
        "format": _FORMAT,
        "bindings": bindings,
        "model_manifest": model.manifest(),
        "geometry_config": geometry_config.manifest(),
        "candidate_vocabulary": geometry_candidate_vocabulary_manifest(
            geometry_config
        ),
        "policy_config": policy_config.manifest(),
        "metadata": _json_mapping(metadata, "geometry checkpoint metadata"),
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


def load_geometry_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_base_policy_checkpoint_sha256: str,
    expected_source_identity: str,
    device: torch.device | str = "cpu",
) -> GeometryCheckpoint:
    expected_file = _sha256(expected_sha256, "geometry checkpoint")
    expected_base = _sha256(
        expected_base_policy_checkpoint_sha256, "base-policy checkpoint"
    )
    expected_source = _sha256(expected_source_identity, "source identity")
    source = Path(path).resolve(strict=True)
    digest = _file_sha256(source)
    if digest != expected_file:
        raise ValueError("geometry checkpoint SHA-256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    expected_fields = {
        "format",
        "bindings",
        "model_manifest",
        "geometry_config",
        "candidate_vocabulary",
        "policy_config",
        "metadata",
        "state_dict",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("geometry checkpoint has unknown or missing fields")
    if payload["format"] != _FORMAT:
        raise ValueError("geometry checkpoint format is unsupported")
    bindings = payload["bindings"]
    if not isinstance(bindings, dict) or set(bindings) != _BINDING_FIELDS:
        raise ValueError("geometry checkpoint bindings are malformed")
    for name in _BINDING_FIELDS:
        _sha256(bindings[name], f"geometry binding {name}")
    if bindings["base_policy_checkpoint_sha256"] != expected_base:
        raise ValueError("geometry checkpoint base-policy identity mismatch")
    if bindings["source_identity"] != expected_source:
        raise ValueError("geometry checkpoint source identity mismatch")

    geometry_config = _geometry_config(payload["geometry_config"])
    vocabulary = geometry_candidate_vocabulary_manifest(geometry_config)
    vocabulary_sha256 = geometry_candidate_vocabulary_sha256(geometry_config)
    if payload["candidate_vocabulary"] != vocabulary:
        raise ValueError("geometry checkpoint vocabulary manifest differs")
    if (
        bindings["geometry_config_sha256"] != geometry_config.sha256
        or bindings["candidate_vocabulary_sha256"] != vocabulary_sha256
    ):
        raise ValueError("geometry checkpoint candidate bindings differ")
    policy_value = payload["policy_config"]
    if not isinstance(policy_value, dict):
        raise ValueError("geometry policy config is malformed")
    try:
        policy_config = GeometryPolicyConfig(**policy_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("geometry policy config is invalid") from exc
    if policy_config.manifest() != policy_value:
        raise ValueError("reconstructed geometry policy identity differs")

    manifest = payload["model_manifest"]
    if not isinstance(manifest, dict):
        raise ValueError("geometry model manifest is malformed")
    schema = {
        TEACHER_V1.sha256: TEACHER_V1,
        ACTOR_VISION_V1.sha256: ACTOR_VISION_V1,
    }.get(manifest.get("schema_sha256"))
    if schema is None or bindings["schema_sha256"] != schema.sha256:
        raise ValueError("geometry checkpoint observation schema is unknown")
    try:
        model_config = GeometryModelConfig(**manifest["config"])
        candidate_count = int(manifest["candidate_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("geometry model configuration is invalid") from exc
    if candidate_count != geometry_config.max_candidates:
        raise ValueError("geometry model width and vocabulary differ")
    model = GeometrySelectorModel(
        schema,
        candidate_count=candidate_count,
        candidate_set_sha256=vocabulary_sha256,
        config=model_config,
    )
    if model.manifest() != manifest:
        raise ValueError("reconstructed geometry model identity differs")
    if bindings["architecture_sha256"] != model.architecture_sha256:
        raise ValueError("geometry checkpoint architecture binding differs")
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise ValueError("geometry checkpoint state dict is malformed")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError("geometry checkpoint parameter identity differs") from exc
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return GeometryCheckpoint(
        source,
        digest,
        model,
        geometry_config,
        policy_config,
        expected_base,
        expected_source,
        _json_mapping(payload["metadata"], "geometry checkpoint metadata"),
    )


def load_safeguarded_geometry_policy(
    path: str | Path,
    *,
    base_policy: GoalConditionedSteeringPolicy,
    expected_sha256: str,
    expected_base_policy_checkpoint_sha256: str,
    expected_source_identity: str,
    device: torch.device | str = "cpu",
) -> SafeguardedGeometryPolicy:
    if not isinstance(base_policy, GoalConditionedSteeringPolicy):
        raise TypeError("base policy must be goal-conditioned steering")
    expected_base = _sha256(
        expected_base_policy_checkpoint_sha256, "base-policy checkpoint"
    )
    if base_policy.artifact_sha256 != expected_base:
        raise ValueError("provided base policy checkpoint identity differs")
    checkpoint = load_geometry_checkpoint(
        path,
        expected_sha256=expected_sha256,
        expected_base_policy_checkpoint_sha256=expected_base,
        expected_source_identity=expected_source_identity,
        device=device,
    )
    return SafeguardedGeometryPolicy(
        base_policy,
        checkpoint.model,
        geometry_config=checkpoint.geometry_config,
        policy_config=checkpoint.policy_config,
        selector_artifact_sha256=checkpoint.sha256,
        source_identity=checkpoint.source_identity,
    )


def load_safeguarded_geometry_ensemble_policy(
    paths: Sequence[str | Path],
    *,
    base_policy: GoalConditionedSteeringPolicy,
    expected_sha256s: Sequence[str],
    expected_base_policy_checkpoint_sha256: str,
    expected_source_identity: str,
    device: torch.device | str = "cpu",
) -> SafeguardedGeometryPolicy:
    """Load a contract-identical selector ensemble without unsafe rewiring."""

    sources = tuple(paths)
    identities = tuple(expected_sha256s)
    if len(sources) < 2:
        raise ValueError("geometry ensemble requires at least two checkpoints")
    if len(sources) != len(identities):
        raise ValueError("geometry ensemble paths and identities differ")
    if not isinstance(base_policy, GoalConditionedSteeringPolicy):
        raise TypeError("base policy must be goal-conditioned steering")
    expected_base = _sha256(
        expected_base_policy_checkpoint_sha256, "base-policy checkpoint"
    )
    if base_policy.artifact_sha256 != expected_base:
        raise ValueError("provided base policy checkpoint identity differs")
    checkpoints = tuple(
        load_geometry_checkpoint(
            path,
            expected_sha256=identity,
            expected_base_policy_checkpoint_sha256=expected_base,
            expected_source_identity=expected_source_identity,
            device=device,
        )
        for path, identity in zip(sources, identities, strict=True)
    )
    first = checkpoints[0]
    if any(
        value.geometry_config != first.geometry_config
        or value.policy_config != first.policy_config
        or value.base_policy_checkpoint_sha256
        != first.base_policy_checkpoint_sha256
        or value.source_identity != first.source_identity
        for value in checkpoints[1:]
    ):
        raise ValueError("geometry ensemble checkpoint contracts differ")
    ensemble = GeometrySelectorEnsemble(
        tuple(value.model for value in checkpoints),
        artifact_sha256s=tuple(value.sha256 for value in checkpoints),
    )
    return SafeguardedGeometryPolicy(
        base_policy,
        ensemble,
        geometry_config=first.geometry_config,
        policy_config=first.policy_config,
        selector_artifact_sha256=ensemble.sha256,
        source_identity=first.source_identity,
    )


__all__ = [
    "GeometryCheckpoint",
    "load_geometry_checkpoint",
    "load_safeguarded_geometry_ensemble_policy",
    "load_safeguarded_geometry_policy",
    "save_geometry_checkpoint",
]
