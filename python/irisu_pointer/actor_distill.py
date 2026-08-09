"""Fail-closed teacher-label alignment onto causal actor-track sequences."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch

from irisu_env import ActionKind
from irisu_rl.encoding import ActorTrackEncoder, EncodedBatch
from irisu_rl.schema import ACTOR_VISION_V1

from .action import PointerActionSpec, PointerActionTensor
from .experts import PointerExpertDecision
from .sequence import (
    PointerSequenceBatch,
    PointerSequenceEpisode,
    pad_pointer_episodes,
)


_PRIVILEGED_KEYS = frozenset(
    {
        "id",
        "chain_id",
        "projectile_hits",
        "age_ticks",
        "remaining_lifetime",
        "rot_timer",
        "score",
        "highest_chain",
        "qualifying_clear_count",
        "active_colors",
        "spawn_interval_ticks",
        "rng_state",
        "future_spawns",
        "snapshot",
    }
)
_TRACK_KEYS = frozenset(
    {
        "kind",
        "shape",
        "color",
        "lifecycle",
        "kind_probabilities",
        "shape_probabilities",
        "color_probabilities",
        "lifecycle_probabilities",
        "effect_x",
        "effect_y",
        "vx_display_per_second",
        "vy_display_per_second",
        "angle",
        "orientation_valid",
        "angular_velocity",
        "size",
        "width",
        "height",
        "confidence",
        "track_age_seconds",
        "missing_age_seconds",
        "occluded_probability",
        "merged_probability",
        "position_uncertainty_x",
        "position_uncertainty_y",
    }
)


class ActorAlignmentError(ValueError):
    """Teacher target cannot be uniquely reproduced from causal actor fields."""


@dataclass(frozen=True, slots=True)
class ActorTeacherStep:
    actor_record: Mapping[str, Any]
    teacher_observation: Mapping[str, Any]
    decision: PointerExpertDecision
    trajectory_return: float

    def __post_init__(self) -> None:
        if not isinstance(self.actor_record, Mapping):
            raise TypeError("actor record must be a causal mapping")
        if not isinstance(self.teacher_observation, Mapping):
            raise TypeError("teacher observation must be a public mapping")
        if not isinstance(self.decision, PointerExpertDecision):
            raise TypeError("teacher label must be a pointer expert decision")
        if (
            isinstance(self.trajectory_return, bool)
            or not isinstance(self.trajectory_return, Real)
            or not math.isfinite(float(self.trajectory_return))
        ):
            raise ValueError("trajectory return must be finite")


@dataclass(frozen=True, slots=True)
class ActorDistillConfig:
    minimum_alignment_confidence: float = 0.60
    minimum_uniqueness_margin: float = 0.05
    maximum_geometry_error: float = 0.20
    geometry_confidence_scale: float = 0.05
    position_noise_pixels: float = 0.0
    velocity_noise_pixels_per_second: float = 0.0
    confidence_noise: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        thresholds = (
            self.minimum_alignment_confidence,
            self.minimum_uniqueness_margin,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 < float(value) <= 1.0
            for value in thresholds
        ) or (
            isinstance(self.confidence_noise, bool)
            or not isinstance(self.confidence_noise, (int, float))
            or not math.isfinite(float(self.confidence_noise))
            or not 0.0 <= float(self.confidence_noise) <= 1.0
        ):
            raise ValueError("actor alignment probability bounds are invalid")
        positive = (
            self.maximum_geometry_error,
            self.geometry_confidence_scale,
        )
        noise = (
            self.position_noise_pixels,
            self.velocity_noise_pixels_per_second,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in positive
        ) or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in noise
        ):
            raise ValueError("actor alignment geometry or noise bounds are invalid")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("actor augmentation seed is invalid")


def perfect_actor_record(
    observation: Mapping[str, Any],
) -> dict[str, object]:
    """Project public simulator truth into the causal track contract.

    This is a development-only perfect-detector bridge for paired distillation;
    the returned record contains no simulator identity, timers, score, chain
    state, or future information.
    """

    bodies = observation.get("bodies", ())
    if not isinstance(bodies, Sequence) or isinstance(bodies, (str, bytes)):
        raise TypeError("public observation bodies must be a sequence")
    global_record = {
        "gauge": float(observation.get("gauge", 0.0)),
        "gauge_max": float(observation.get("gauge_max", 1.0)),
        "gauge_confidence": 1.0,
        "level": float(observation.get("level", 0.0)),
        "level_confidence": 1.0,
        "elapsed_seconds": max(float(observation.get("tick", 0.0)), 0.0) / 50.0,
    }
    tracks: list[dict[str, object]] = []
    for body in bodies:
        if not isinstance(body, Mapping):
            raise TypeError("public observation body must be a mapping")
        lifecycle = body.get("lifecycle", "unknown")
        velocity_scale = 50.0 if lifecycle == "scripted_falling" else 10.0
        size = float(body.get("size", 0.0))
        tracks.append(
            {
                "kind": body.get("kind", "unknown"),
                "shape": body.get("shape", "unknown"),
                "color": body.get("color", -1),
                "lifecycle": {
                    "scripted_falling": "falling",
                    "dynamic_fresh": "fresh",
                }.get(lifecycle, lifecycle),
                "effect_x": float(body.get("x", 0.0)),
                "effect_y": float(body.get("y", 0.0)),
                "vx_display_per_second": float(body.get("vx", 0.0))
                * velocity_scale,
                "vy_display_per_second": float(body.get("vy", 0.0))
                * velocity_scale,
                "angle": float(body.get("angle", 0.0)),
                "orientation_valid": body.get("shape") not in {"circle", "unknown"},
                "angular_velocity": float(body.get("angular_velocity", 0.0)),
                "size": size,
                "width": size,
                "height": size,
                "confidence": 1.0,
            }
        )
    return {"global": global_record, "tracks": tracks}


def _rng(config: ActorDistillConfig, identity: str, index: int) -> np.random.Generator:
    digest = hashlib.sha256(
        f"{config.seed}\0{identity}\0{index}".encode()
    ).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def sanitize_actor_record(
    record: Mapping[str, Any],
    *,
    config: ActorDistillConfig | None = None,
    episode_identity: str = "",
    step_index: int = 0,
) -> dict[str, object]:
    """Copy only causal fields and apply deterministic bounded sensor noise."""

    resolved = config or ActorDistillConfig()
    random = _rng(resolved, episode_identity, step_index)
    source_global = record.get("global", record)
    if not isinstance(source_global, Mapping):
        raise TypeError("actor global record must be a mapping")
    global_values = {
        str(key): value
        for key, value in source_global.items()
        if key not in _PRIVILEGED_KEYS and key != "tracks"
    }
    raw_tracks = record.get("tracks", ())
    if not isinstance(raw_tracks, Sequence) or isinstance(raw_tracks, (str, bytes)):
        raise TypeError("actor tracks must be a sequence")
    tracks: list[dict[str, object]] = []
    for raw in raw_tracks:
        if not isinstance(raw, Mapping):
            raise TypeError("actor track must be a mapping")
        track = {key: raw[key] for key in _TRACK_KEYS if key in raw}
        for name in ("effect_x", "effect_y"):
            if name in track and resolved.position_noise_pixels:
                track[name] = float(track[name]) + random.uniform(
                    -resolved.position_noise_pixels,
                    resolved.position_noise_pixels,
                )
        for name in (
            "vx_display_per_second",
            "vy_display_per_second",
        ):
            if name in track and resolved.velocity_noise_pixels_per_second:
                track[name] = float(track[name]) + random.uniform(
                    -resolved.velocity_noise_pixels_per_second,
                    resolved.velocity_noise_pixels_per_second,
                )
        if "confidence" in track and resolved.confidence_noise:
            track["confidence"] = min(
                1.0,
                max(
                    0.0,
                    float(track["confidence"])
                    + random.uniform(
                        -resolved.confidence_noise,
                        resolved.confidence_noise,
                    ),
                ),
            )
        tracks.append(track)
    return {"global": global_values, "tracks": tracks}


def _teacher_target(
    observation: Mapping[str, Any], target_body_id: int
) -> Mapping[str, Any]:
    bodies = observation.get("bodies", ())
    if not isinstance(bodies, Sequence) or isinstance(bodies, (str, bytes)):
        raise ActorAlignmentError("teacher bodies must be a public sequence")
    matches = [
        body
        for body in bodies
        if isinstance(body, Mapping)
        and isinstance(body.get("id"), Integral)
        and not isinstance(body.get("id"), bool)
        and int(body["id"]) == target_body_id
    ]
    if len(matches) != 1:
        raise ActorAlignmentError("teacher target ID is absent or nonunique")
    return matches[0]


def _category_index(value: object, names: tuple[object, ...], unknown: int) -> int:
    try:
        return names.index(value)
    except ValueError:
        return unknown


def _target_descriptor(body: Mapping[str, Any]) -> tuple[int, int, int, int, float, ...]:
    lifecycle = {
        "scripted_falling": "falling",
        "dynamic_fresh": "fresh",
        0: "falling",
        1: "fresh",
        2: "confirmed",
        3: "rotten",
    }.get(body.get("lifecycle"), body.get("lifecycle"))
    kind = _category_index(
        body.get("kind"), ("piece", "projectile", "bonus", "unknown"), 3
    )
    shape = _category_index(
        body.get("shape"), ("circle", "box", "triangle", "unknown"), 3
    )
    color_raw = body.get("color")
    color = (
        int(color_raw)
        if isinstance(color_raw, Integral)
        and not isinstance(color_raw, bool)
        and 0 <= int(color_raw) < 6
        else (6 if color_raw == -2 else 7)
    )
    lifecycle_index = _category_index(
        lifecycle,
        ("falling", "fresh", "confirmed", "rotten", "ambiguous", "unknown"),
        5,
    )
    size = float(body.get("size", 0.0))
    return (
        kind,
        shape,
        color,
        lifecycle_index,
        float(body.get("effect_x", body.get("x", 0.0))) / 640.0,
        float(body.get("effect_y", body.get("y", 0.0))) / 480.0,
        float(body.get("width", size)) / 640.0,
        float(body.get("height", size)) / 480.0,
    )


def align_teacher_target(
    encoded: EncodedBatch,
    row: int,
    teacher_body: Mapping[str, Any],
    *,
    config: ActorDistillConfig | None = None,
) -> int:
    """Return one encoded actor row using no identity or privileged mechanics."""

    resolved = config or ActorDistillConfig()
    if encoded.schema.sha256 != ACTOR_VISION_V1.sha256:
        raise ValueError("target alignment requires actor-vision-v1")
    if not 0 <= row < encoded.global_features.shape[0]:
        raise IndexError("encoded alignment row is out of range")
    descriptor = _target_descriptor(teacher_body)
    names = encoded.schema.body_features
    kind, shape, color, lifecycle, x, y, width, height = descriptor
    geometry_indices = tuple(
        names.index(name)
        for name in (
            "effect_x_norm",
            "effect_y_norm",
            "width_norm",
            "height_norm",
        )
    )
    confidence_index = names.index("detection_confidence")
    candidates: list[tuple[float, float, int]] = []
    for index in np.flatnonzero(encoded.body_mask[row]):
        body = encoded.body_features[row, index]
        category = float(
            (
                body[kind]
                + body[4 + shape]
                + body[8 + color]
                + body[16 + lifecycle]
            )
            / 4.0
        )
        geometry = sum(
            abs(float(body[column]) - target)
            for column, target in zip(
                geometry_indices, (x, y, width, height)
            )
        )
        confidence = (
            float(body[confidence_index])
            * category
            * math.exp(-geometry / resolved.geometry_confidence_scale)
        )
        candidates.append((confidence, geometry, int(index)))
    if not candidates:
        raise ActorAlignmentError("actor record has no visible alignment candidate")
    candidates.sort(key=lambda value: (-value[0], value[1], value[2]))
    best = candidates[0]
    if (
        best[0] < resolved.minimum_alignment_confidence
        or best[1] > resolved.maximum_geometry_error
    ):
        raise ActorAlignmentError("actor target alignment confidence is too low")
    if (
        len(candidates) > 1
        and best[0] - candidates[1][0] < resolved.minimum_uniqueness_margin
    ):
        raise ActorAlignmentError("actor target alignment is not unique")
    return best[2]


def build_actor_sequence_episode(
    identity: str,
    steps: Sequence[ActorTeacherStep],
    *,
    config: ActorDistillConfig | None = None,
    pointer_spec: PointerActionSpec | None = None,
) -> PointerSequenceEpisode:
    resolved = config or ActorDistillConfig()
    spec = pointer_spec or PointerActionSpec()
    supplied = tuple(steps)
    if not supplied:
        raise ValueError("actor sequence needs at least one paired step")
    records = [
        sanitize_actor_record(
            value.actor_record,
            config=resolved,
            episode_identity=identity,
            step_index=index,
        )
        for index, value in enumerate(supplied)
    ]
    encoded = ActorTrackEncoder().encode(records)
    kinds: list[int] = []
    waits: list[int] = []
    targets: list[int] = []
    templates: list[int] = []
    for row, value in enumerate(supplied):
        kind = int(ActionKind.parse(value.decision.kind))
        kinds.append(kind)
        if kind == int(ActionKind.WAIT):
            try:
                wait_index = spec.wait_choices.index(value.decision.wait_ticks)
            except ValueError as exc:
                raise ActorAlignmentError(
                    "teacher wait is outside the pointer vocabulary"
                ) from exc
            waits.append(wait_index)
            targets.append(0)
            templates.append(0)
            continue
        if not 0 <= value.decision.template_index < spec.template_count:
            raise ActorAlignmentError("teacher template is outside the vocabulary")
        assert value.decision.target_body_id is not None
        target = _teacher_target(
            value.teacher_observation, value.decision.target_body_id
        )
        targets.append(
            align_teacher_target(encoded, row, target, config=resolved)
        )
        waits.append(0)
        templates.append(value.decision.template_index)
    occupied = np.flatnonzero(encoded.body_mask.any(axis=0))
    width = int(occupied[-1]) + 1 if occupied.size else 1
    return PointerSequenceEpisode(
        identity,
        torch.from_numpy(encoded.global_features.copy()),
        torch.from_numpy(encoded.body_features[:, :width].copy()),
        torch.from_numpy(encoded.body_mask[:, :width].copy()),
        PointerActionTensor(
            torch.tensor(kinds, dtype=torch.long),
            torch.tensor(waits, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long),
            torch.tensor(templates, dtype=torch.long),
        ),
        torch.tensor(
            [float(value.trajectory_return) for value in supplied],
            dtype=torch.float32,
        ),
        ACTOR_VISION_V1,
        spec,
    )


def build_actor_sequence_batch(
    episodes: Mapping[str, Sequence[ActorTeacherStep]],
    *,
    config: ActorDistillConfig | None = None,
    pointer_spec: PointerActionSpec | None = None,
    device: torch.device | str | None = None,
) -> PointerSequenceBatch:
    if not isinstance(episodes, Mapping) or not episodes:
        raise ValueError("paired actor episodes must be a nonempty mapping")
    return pad_pointer_episodes(
        tuple(
            build_actor_sequence_episode(
                identity,
                episodes[identity],
                config=config,
                pointer_spec=pointer_spec,
            )
            for identity in sorted(episodes)
        ),
        device=device,
    )


__all__ = [
    "ActorAlignmentError",
    "ActorDistillConfig",
    "ActorTeacherStep",
    "align_teacher_target",
    "build_actor_sequence_batch",
    "build_actor_sequence_episode",
    "perfect_actor_record",
    "sanitize_actor_record",
]
