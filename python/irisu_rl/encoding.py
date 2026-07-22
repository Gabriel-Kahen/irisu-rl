"""Owned NumPy encoders for privileged state and causal perception tracks."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .schema import ACTOR_VISION_V1, TEACHER_V1, TensorSchema

CLIENT_WIDTH = 640.0
CLIENT_HEIGHT = 480.0
VELOCITY_SCALE = 1000.0
ANGULAR_VELOCITY_SCALE = 10.0


@dataclass(slots=True)
class EncodedBatch:
    """Contiguous, owned tensors plus non-model metadata."""

    global_features: np.ndarray
    body_features: np.ndarray
    body_mask: np.ndarray
    source_tick: np.ndarray
    health_flags: np.ndarray
    schema: TensorSchema

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        values = (
            self.global_features,
            self.body_features,
            self.body_mask,
            self.source_tick,
            self.health_flags,
        )
        if any(not isinstance(value, np.ndarray) for value in values):
            raise TypeError("encoded tensors must be NumPy arrays")
        if self.global_features.ndim != 2:
            raise ValueError("global features must have rank 2")
        batch = self.global_features.shape[0]
        expected = (batch, self.schema.capacity, len(self.schema.body_features))
        if self.global_features.shape != (batch, len(self.schema.global_features)):
            raise ValueError("invalid global feature shape")
        if self.body_features.shape != expected:
            raise ValueError("invalid body feature shape")
        if self.body_mask.shape != expected[:2]:
            raise ValueError("invalid body mask shape")
        if self.source_tick.shape != (batch,):
            raise ValueError("invalid source tick shape")
        if self.health_flags.shape != (batch,):
            raise ValueError("invalid health flag shape")
        if self.global_features.dtype != np.float32:
            raise ValueError("global features must be float32")
        if self.body_features.dtype != np.float32:
            raise ValueError("body features must be float32")
        if self.body_mask.dtype != np.bool_:
            raise ValueError("body mask must be bool")
        if self.source_tick.dtype != np.uint64:
            raise ValueError("source ticks must be uint64")
        if self.health_flags.dtype != np.uint32:
            raise ValueError("health flags must be uint32")
        for value in values:
            if not value.flags.c_contiguous or not value.flags.owndata:
                raise ValueError("encoded tensors must be contiguous and owned")

    def row(self, index: int) -> EncodedBatch:
        return EncodedBatch(
            np.array(self.global_features[index : index + 1], copy=True),
            np.array(self.body_features[index : index + 1], copy=True),
            np.array(self.body_mask[index : index + 1], copy=True),
            np.array(self.source_tick[index : index + 1], copy=True),
            np.array(self.health_flags[index : index + 1], copy=True),
            self.schema,
        )

    def copy(self) -> EncodedBatch:
        return EncodedBatch(
            np.array(self.global_features, copy=True),
            np.array(self.body_features, copy=True),
            np.array(self.body_mask, copy=True),
            np.array(self.source_tick, copy=True),
            np.array(self.health_flags, copy=True),
            self.schema,
        )


def _empty(schema: TensorSchema, batch: int) -> EncodedBatch:
    return EncodedBatch(
        np.zeros((batch, len(schema.global_features)), dtype=np.float32),
        np.zeros((batch, schema.capacity, len(schema.body_features)), dtype=np.float32),
        np.zeros((batch, schema.capacity), dtype=np.bool_),
        np.zeros(batch, dtype=np.uint64),
        np.zeros(batch, dtype=np.uint32),
        schema,
    )


def _value(source: object, key: str, default: Any = 0) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def _nested(source: object, parent: str, key: str, default: Any = 0) -> Any:
    value = _value(source, parent, None)
    return _value(value, key, default) if value is not None else default


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(value)), value)


def _probabilities(value: Any, names: tuple[str, ...], hard: Any) -> np.ndarray:
    output = np.zeros(len(names), dtype=np.float32)
    if isinstance(value, Mapping):
        for index, name in enumerate(names):
            output[index] = float(value.get(name, 0.0))
    elif value is not None and not isinstance(value, (str, bytes)):
        supplied = np.asarray(value, dtype=np.float32)
        if supplied.shape != output.shape:
            raise ValueError(f"expected {len(names)} category probabilities")
        output[:] = supplied
    else:
        aliases = {name: index for index, name in enumerate(names)}
        if hard in aliases:
            output[aliases[hard]] = 1.0
        elif isinstance(hard, (int, np.integer)) and 0 <= int(hard) < len(names) - 1:
            output[int(hard)] = 1.0
        else:
            output[-1] = 1.0
    if not np.all(np.isfinite(output)) or np.any(output < 0):
        raise ValueError("category probabilities must be finite and nonnegative")
    total = float(output.sum())
    if total <= 0:
        output[-1] = 1.0
    else:
        output /= total
    return output


def _body_core(body: object, *, teacher: bool) -> tuple[np.ndarray, tuple[float, ...]]:
    kind_names = ("piece", "projectile", "bonus", "unknown")
    shape_names = ("circle", "box", "triangle", "unknown")
    color_names = ("0", "1", "2", "3", "4", "5", "bonus", "unknown")
    lifecycle_names = (
        "falling",
        "fresh",
        "confirmed",
        "rotten",
        "ambiguous",
        "unknown",
    )
    kind_raw = _value(body, "kind", "unknown")
    shape_raw = _value(body, "shape", "unknown")
    color_raw = _value(body, "color", "unknown")
    lifecycle_raw = _value(body, "lifecycle", "unknown")
    lifecycle_alias = {
        "scripted_falling": "falling",
        "dynamic_fresh": "fresh",
        "deleted": "unknown",
        0: "falling",
        1: "fresh",
        2: "confirmed",
        3: "rotten",
    }
    color_hard = "bonus" if color_raw == -2 else str(color_raw)
    lifecycle_hard = lifecycle_alias.get(lifecycle_raw, lifecycle_raw)
    categories = np.concatenate(
        (
            _probabilities(
                _value(body, "kind_probabilities", None), kind_names, kind_raw
            ),
            _probabilities(
                _value(body, "shape_probabilities", None), shape_names, shape_raw
            ),
            _probabilities(
                _value(body, "color_probabilities", None), color_names, color_hard
            ),
            _probabilities(
                _value(body, "lifecycle_probabilities", None),
                lifecycle_names,
                lifecycle_hard,
            ),
        )
    )
    if teacher:
        x = float(_value(body, "effect_x", _value(body, "x", 0.0)))
        y = float(_value(body, "effect_y", _value(body, "y", 0.0)))
        explicit_velocity = (
            _value(body, "vx_display_per_second", None) is not None
            and _value(body, "vy_display_per_second", None) is not None
        )
        vx = float(_value(body, "vx_display_per_second", _value(body, "vx", 0.0)))
        vy = float(_value(body, "vy_display_per_second", _value(body, "vy", 0.0)))
    else:
        if not isinstance(body, Mapping):
            raise TypeError("actor body must be a causal track mapping")
        required = (
            "effect_x",
            "effect_y",
            "vx_display_per_second",
            "vy_display_per_second",
        )
        missing = [key for key in required if key not in body]
        if missing:
            raise ValueError(
                f"actor track missing explicit effect-time fields: {missing}"
            )
        x = float(body["effect_x"])
        y = float(body["effect_y"])
        vx = float(body["vx_display_per_second"])
        vy = float(body["vy_display_per_second"])
        explicit_velocity = True
    if teacher and not explicit_velocity:
        factor = 50.0 if lifecycle_hard == "falling" else 10.0
        vx *= factor
        vy *= factor
    angle = float(_value(body, "angle", 0.0))
    shape_probabilities = _value(body, "shape_probabilities", None)
    explicit_orientation_valid = _value(body, "orientation_valid", None)
    if shape_raw in ("circle", 0):
        orient_sin, orient_cos, orient_valid = 0.0, 0.0, 0.0
    elif shape_raw in ("box", 1):
        orient_sin, orient_cos, orient_valid = (
            math.sin(4 * angle),
            math.cos(4 * angle),
            1.0,
        )
    elif shape_raw in ("triangle", 2):
        orient_sin, orient_cos, orient_valid = math.sin(angle), math.cos(angle), 1.0
    else:
        orient_sin, orient_cos, orient_valid = 0.0, 0.0, 0.0
    if (
        not teacher
        and shape_probabilities is not None
        and explicit_orientation_valid is None
    ):
        orient_sin, orient_cos, orient_valid = 0.0, 0.0, 0.0
    if explicit_orientation_valid is not None and not bool(explicit_orientation_valid):
        orient_sin, orient_cos, orient_valid = 0.0, 0.0, 0.0
    size = float(_value(body, "size", 0.0))
    width = float(_value(body, "width", size))
    height = float(_value(body, "height", size))
    numeric = (
        x / CLIENT_WIDTH,
        y / CLIENT_HEIGHT,
        vx / VELOCITY_SCALE,
        vy / VELOCITY_SCALE,
        orient_sin,
        orient_cos,
        orient_valid,
        float(_value(body, "angular_velocity", 0.0)) / ANGULAR_VELOCITY_SCALE,
        width / CLIENT_WIDTH,
        height / CLIENT_HEIGHT,
        float(_value(body, "confidence", 1.0 if teacher else 0.0)),
        math.log1p(max(float(_value(body, "track_age_seconds", 0.0)), 0.0)) / 4.0,
        math.log1p(max(float(_value(body, "missing_age_seconds", 0.0)), 0.0)) / 4.0,
        float(_value(body, "occluded_probability", 0.0)),
        float(_value(body, "merged_probability", 0.0)),
        float(_value(body, "position_uncertainty_x", 0.0)) / CLIENT_WIDTH,
        float(_value(body, "position_uncertainty_y", 0.0)) / CLIENT_HEIGHT,
    )
    combined = np.concatenate((categories, np.asarray(numeric, dtype=np.float32)))
    if not np.all(np.isfinite(combined)):
        raise ValueError("body features must be finite")
    return combined, (
        x,
        y,
        width,
        height,
        float(numeric[10]),
        float(_value(body, "missing_age_seconds", 0.0)),
        float(_value(body, "occluded_probability", 0.0)),
    )


def _visible(meta: tuple[float, ...]) -> bool:
    x, y, width, height, _, missing_age, occluded = meta
    return (
        x + width / 2 >= 0
        and x - width / 2 <= CLIENT_WIDTH
        and y + height / 2 >= 0
        and y - height / 2 <= CLIENT_HEIGHT
        and missing_age <= 1.0
        and occluded < 0.999
    )


def _typed_teacher_bodies(
    observation: object, count: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized aligned-or-packed ctypes extraction; returns owned rows."""

    bodies = np.ctypeslib.as_array(observation.bodies)[:count]
    output = np.zeros((count, width), dtype=np.float32)
    if count == 0:
        return output, np.empty(0, dtype=np.int64)
    kind = bodies["kind"].astype(np.int64)
    valid_kind = (kind >= 0) & (kind < 3)
    output[np.arange(count), np.where(valid_kind, kind, 3)] = 1.0
    shape = bodies["shape"].astype(np.int64)
    valid_shape = (shape >= 0) & (shape < 3)
    output[np.arange(count), 4 + np.where(valid_shape, shape, 3)] = 1.0
    color = bodies["color"].astype(np.int64)
    color_index = np.where(
        (color >= 0) & (color < 6), color, np.where(color == -2, 6, 7)
    )
    output[np.arange(count), 8 + color_index] = 1.0
    lifecycle = bodies["lifecycle"].astype(np.int64)
    lifecycle_index = np.where((lifecycle >= 0) & (lifecycle < 4), lifecycle, 5)
    output[np.arange(count), 16 + lifecycle_index] = 1.0
    x = bodies["x"].astype(np.float64)
    y = bodies["y"].astype(np.float64)
    output[:, 22] = x / CLIENT_WIDTH
    output[:, 23] = y / CLIENT_HEIGHT
    velocity_factor = np.where(lifecycle == 0, 50.0, 10.0)
    output[:, 24] = bodies["vx"] * velocity_factor / VELOCITY_SCALE
    output[:, 25] = bodies["vy"] * velocity_factor / VELOCITY_SCALE
    angle = bodies["angle"].astype(np.float64)
    box = shape == 1
    triangle = shape == 2
    output[:, 26] = np.where(
        box, np.sin(4 * angle), np.where(triangle, np.sin(angle), 0.0)
    )
    output[:, 27] = np.where(
        box, np.cos(4 * angle), np.where(triangle, np.cos(angle), 0.0)
    )
    output[:, 28] = (box | triangle).astype(np.float32)
    output[:, 29] = bodies["angular_velocity"] / ANGULAR_VELOCITY_SCALE
    output[:, 30] = bodies["size"] / CLIENT_WIDTH
    output[:, 31] = bodies["size"] / CLIENT_HEIGHT
    output[:, 32] = 1.0
    output[:, 39] = bodies["id"].astype(np.float64) / 2**32
    output[:, 40] = bodies["chain_id"].astype(np.float64) / 2**32
    output[:, 41] = np.log1p(bodies["projectile_hits"].astype(np.float64)) / 8.0
    output[:, 42] = np.log1p(bodies["age_ticks"].astype(np.float64)) / 16.0
    lifetime = bodies["remaining_lifetime"].astype(np.float64)
    output[:, 43] = np.sign(lifetime) * np.log1p(np.abs(lifetime)) / 16.0
    output[:, 44] = np.log1p(bodies["rot_timer"].astype(np.float64)) / 16.0
    if not np.all(np.isfinite(output)):
        raise ValueError("body features must be finite")
    tie_keys = tuple(output[:, column] for column in reversed(range(width)))
    order = np.lexsort(tie_keys + (x, y))
    return np.ascontiguousarray(output[order]), order


class ActorTrackEncoder:
    """Encode only deployment-reproducible HUD, bridge, and track records."""

    schema = ACTOR_VISION_V1

    def encode(self, observations: Sequence[Mapping[str, Any]]) -> EncodedBatch:
        if any(not isinstance(value, Mapping) for value in observations):
            raise TypeError("actor-vision-v1 requires causal mapping records")
        output = _empty(self.schema, len(observations))
        names = {name: index for index, name in enumerate(self.schema.global_features)}
        for lane, observation in enumerate(observations):
            global_values = observation.get("global", observation)
            row = output.global_features[lane]
            gauge_max = max(float(_value(global_values, "gauge_max", 1000.0)), 1.0)
            row[names["gauge_fraction"]] = (
                float(_value(global_values, "gauge", 0.0)) / gauge_max
            )
            row[names["gauge_confidence"]] = float(
                _value(global_values, "gauge_confidence", 0.0)
            )
            row[names["level_log1p"]] = (
                math.log1p(max(float(_value(global_values, "level", 0.0)), 0.0)) / 8.0
            )
            direct = set(self.schema.global_features) - {
                "gauge_fraction",
                "gauge_confidence",
                "level_log1p",
            }
            scales = {
                "elapsed_seconds_scaled": ("elapsed_seconds", "log8"),
                "cursor_age_seconds_scaled": ("cursor_age_seconds", "log4"),
                "previous_requested_duration_seconds_scaled": (
                    "previous_requested_duration_seconds",
                    "log4",
                ),
                "previous_executed_duration_seconds_scaled": (
                    "previous_executed_duration_seconds",
                    "log4",
                ),
                "previous_down_age_seconds_scaled": (
                    "previous_down_age_seconds",
                    "log4",
                ),
                "previous_up_age_seconds_scaled": ("previous_up_age_seconds", "log4"),
                "previous_injection_age_seconds_scaled": (
                    "previous_injection_age_seconds",
                    "log4",
                ),
                "previous_projectile_confirmation_age_seconds_scaled": (
                    "previous_projectile_confirmation_age_seconds",
                    "log4",
                ),
                "pending_age_seconds_scaled": ("pending_age_seconds", "log4"),
                "frame_age_seconds_scaled": ("frame_age_seconds", "tenth"),
                "effect_horizon_seconds_scaled": ("effect_horizon_seconds", "tenth"),
                "effect_horizon_uncertainty_scaled": (
                    "effect_horizon_uncertainty",
                    "tenth",
                ),
                "recent_births_scaled": ("recent_births", "log4"),
                "recent_deaths_scaled": ("recent_deaths", "log4"),
                "recent_merges_scaled": ("recent_merges", "log4"),
                "recent_clears_scaled": ("recent_clears", "log4"),
            }
            for feature in direct:
                key, transform = scales.get(feature, (feature, "identity"))
                value = float(_value(global_values, key, 0.0))
                if transform == "log8":
                    value = math.log1p(max(value, 0.0)) / 8.0
                elif transform == "log4":
                    value = math.log1p(max(value, 0.0)) / 4.0
                elif transform == "tenth":
                    value /= 0.1
                row[names[feature]] = value
            effect_features = (
                "previous_effect_pending",
                "previous_effect_confirmed",
                "previous_effect_missed",
                "previous_effect_ambiguous",
            )
            effect = np.asarray(
                [row[names[feature]] for feature in effect_features],
                dtype=np.float32,
            )
            if np.any(effect < 0) or not np.all(np.isfinite(effect)):
                raise ValueError(
                    "previous effect posterior must be finite and nonnegative"
                )
            effect_total = float(effect.sum())
            if effect_total == 0:
                row[names["previous_effect_ambiguous"]] = 1.0
            else:
                for feature, probability in zip(effect_features, effect / effect_total):
                    row[names[feature]] = probability
            bodies = observation.get("tracks", ())
            candidates: list[tuple[tuple[float, ...], np.ndarray]] = []
            for body in bodies:
                encoded, meta = _body_core(body, teacher=False)
                if _visible(meta):
                    key = (
                        -meta[4],
                        meta[5],
                        meta[1],
                        meta[0],
                        *encoded.tolist(),
                    )
                    candidates.append((key, encoded))
            candidates.sort(key=lambda item: item[0])
            if len(candidates) > self.schema.capacity:
                output.health_flags[lane] |= np.uint32(1)
                row[names["detection_overflow"]] = 1.0
            selected = candidates[: self.schema.capacity]
            for index, (_, encoded) in enumerate(selected):
                output.body_features[lane, index] = encoded
                output.body_mask[lane, index] = True
            row[names["detection_count_fraction"]] = (
                len(selected) / self.schema.capacity
            )
            if selected:
                conf_index = self.schema.body_features.index("detection_confidence")
                row[names["mean_detection_confidence"]] = float(
                    output.body_features[lane, : len(selected), conf_index].mean()
                )
            if not np.all(np.isfinite(row)):
                raise ValueError("global features must be finite")
            if np.any(np.abs(row) > 32.0) or (
                selected
                and np.any(np.abs(output.body_features[lane, : len(selected)]) > 32.0)
            ):
                output.health_flags[lane] |= np.uint32(2)
        return output


class TeacherStateEncoder:
    """Encode public simulator truth for a teacher or privileged critic."""

    schema = TEACHER_V1

    def encode(self, observations: Sequence[object]) -> EncodedBatch:
        output = _empty(self.schema, len(observations))
        for lane, observation in enumerate(observations):
            tick = int(_value(observation, "tick", 0))
            score = float(_value(observation, "score", 0))
            gauge = float(_value(observation, "gauge", 0))
            gauge_max = max(float(_value(observation, "gauge_max", 1000)), 1.0)
            level = max(float(_value(observation, "level", 0)), 0.0)
            highest = max(float(_value(observation, "highest_chain", 0)), 0.0)
            clears = max(float(_value(observation, "qualifying_clear_count", 0)), 0.0)
            row = (
                tick / 100_000.0,
                _signed_log1p(score) / 20.0,
                gauge / gauge_max,
                math.log1p(level) / 8.0,
                math.log1p(highest) / 8.0,
                math.log1p(clears) / 8.0,
                float(
                    _nested(
                        observation,
                        "difficulty",
                        "active_colors",
                        _value(observation, "active_colors", 0),
                    )
                )
                / 6.0,
                float(
                    _nested(
                        observation,
                        "difficulty",
                        "spawn_interval_ticks",
                        _value(observation, "spawn_interval_ticks", 0),
                    )
                )
                / 100.0,
                float(bool(_value(observation, "left_held", False))),
                float(bool(_value(observation, "right_held", False))),
                float(bool(_value(observation, "terminated", False))),
                float(bool(_value(observation, "truncated", False))),
            )
            output.global_features[lane] = row
            if not np.all(np.isfinite(output.global_features[lane])):
                raise ValueError("global features must be finite")
            output.source_tick[lane] = tick
            if isinstance(observation, Mapping):
                bodies = observation.get("bodies", ())
            else:
                count = int(_value(observation, "body_count", 0))
                if not 0 <= count <= self.schema.capacity:
                    raise ValueError("teacher observation exceeds body capacity")
                encoded, _ = _typed_teacher_bodies(
                    observation, count, len(self.schema.body_features)
                )
                output.body_features[lane, :count] = encoded
                output.body_mask[lane, :count] = True
                continue
            if len(bodies) > self.schema.capacity:
                raise ValueError("teacher observation exceeds body capacity")
            encoded_bodies: list[tuple[tuple[float, ...], np.ndarray]] = []
            for body in bodies:
                core, meta = _body_core(body, teacher=True)
                privileged = np.asarray(
                    (
                        int(_value(body, "id", 0)) / 2**32,
                        int(_value(body, "chain_id", 0)) / 2**32,
                        math.log1p(max(int(_value(body, "projectile_hits", 0)), 0))
                        / 8.0,
                        math.log1p(max(int(_value(body, "age_ticks", 0)), 0)) / 16.0,
                        _signed_log1p(float(_value(body, "remaining_lifetime", 0)))
                        / 16.0,
                        math.log1p(max(int(_value(body, "rot_timer", 0)), 0)) / 16.0,
                    ),
                    dtype=np.float32,
                )
                combined = np.concatenate((core, privileged))
                key = (meta[1], meta[0], *combined.tolist())
                encoded_bodies.append((key, combined))
            encoded_bodies.sort(key=lambda item: item[0])
            for index, (_, encoded) in enumerate(encoded_bodies):
                output.body_features[lane, index] = encoded
                output.body_mask[lane, index] = True
        return output
