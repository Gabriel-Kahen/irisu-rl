"""Immutable feature allowlists and their canonical identities."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

BODY_CAPACITY = 196


@dataclass(frozen=True, slots=True)
class TensorSchema:
    version: str
    source: str
    global_features: tuple[str, ...]
    body_features: tuple[str, ...]
    preprocessing: tuple[str, ...]
    capacity: int = BODY_CAPACITY

    def __post_init__(self) -> None:
        if self.source not in {"actor_tracks", "teacher_state"}:
            raise ValueError("schema source must be actor_tracks or teacher_state")
        if self.capacity != BODY_CAPACITY:
            raise ValueError(f"schema capacity must be {BODY_CAPACITY}")
        if len(set(self.global_features)) != len(self.global_features):
            raise ValueError("duplicate global feature")
        if len(set(self.body_features)) != len(self.body_features):
            raise ValueError("duplicate body feature")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source": self.source,
            "capacity": self.capacity,
            "global_features": list(self.global_features),
            "body_features": list(self.body_features),
            "preprocessing": list(self.preprocessing),
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


ACTOR_GLOBAL_FEATURES = (
    "gauge_fraction",
    "gauge_confidence",
    "level_log1p",
    "level_confidence",
    "elapsed_seconds_scaled",
    "timing_confidence",
    "commanded_left",
    "commanded_right",
    "cursor_x_norm",
    "cursor_y_norm",
    "cursor_age_seconds_scaled",
    "transform_confidence",
    "previous_kind_wait",
    "previous_kind_weak",
    "previous_kind_strong",
    "previous_x_norm",
    "previous_y_norm",
    "previous_duration_seconds_scaled",
    "previous_acknowledged",
    "pending_kind_wait",
    "pending_kind_weak",
    "pending_kind_strong",
    "pending_age_seconds_scaled",
    "pending_acknowledged",
    "frame_age_seconds_scaled",
    "effect_horizon_seconds_scaled",
    "effect_horizon_uncertainty_scaled",
    "recent_births_scaled",
    "recent_deaths_scaled",
    "recent_merges_scaled",
    "recent_clears_scaled",
    "detection_count_fraction",
    "mean_detection_confidence",
    "detection_overflow",
)

ACTOR_BODY_FEATURES = (
    "kind_piece",
    "kind_projectile",
    "kind_bonus",
    "kind_unknown",
    "shape_circle",
    "shape_box",
    "shape_triangle",
    "shape_unknown",
    "color_0",
    "color_1",
    "color_2",
    "color_3",
    "color_4",
    "color_5",
    "color_bonus",
    "color_unknown",
    "lifecycle_falling",
    "lifecycle_fresh",
    "lifecycle_confirmed",
    "lifecycle_rotten",
    "lifecycle_ambiguous",
    "lifecycle_unknown",
    "effect_x_norm",
    "effect_y_norm",
    "velocity_x_display_per_second_scaled",
    "velocity_y_display_per_second_scaled",
    "orientation_sin",
    "orientation_cos",
    "orientation_valid",
    "angular_velocity_scaled",
    "width_norm",
    "height_norm",
    "detection_confidence",
    "track_age_seconds_scaled",
    "missing_age_seconds_scaled",
    "occluded_probability",
    "merged_probability",
    "position_uncertainty_x_norm",
    "position_uncertainty_y_norm",
)

TEACHER_GLOBAL_FEATURES = (
    "tick_scaled",
    "score_signed_log1p",
    "gauge_fraction",
    "level_log1p",
    "highest_chain_log1p",
    "qualifying_clears_log1p",
    "active_colors_scaled",
    "spawn_interval_scaled",
    "left_held",
    "right_held",
    "terminated",
    "truncated",
)

TEACHER_BODY_FEATURES = ACTOR_BODY_FEATURES + (
    "id_scaled",
    "chain_id_scaled",
    "projectile_hits_log1p",
    "age_ticks_log1p",
    "remaining_lifetime_signed_log1p",
    "rot_timer_log1p",
)

ACTOR_PREPROCESSING = (
    "coordinates: effect-time pixels divided by 640x480",
    "velocity: explicit display pixels/second divided by 1000",
    "angular_velocity: radians/second divided by 10",
    "elapsed_seconds: log1p(value)/8",
    "age/duration seconds: log1p(value)/4",
    "frame/effect seconds: value/0.1",
    "recent event counts: log1p(value)/4",
    "level: log1p(value)/8",
    "categories: normalized probabilities with explicit unknown bucket",
    "orientation: circle/unknown invalid; box sin(4a),cos(4a); triangle sin(a),cos(a)",
    "fully offscreen, fully occluded, and missing-age>1s tracks excluded",
    "health flag bit 1: any normalized feature magnitude exceeds 32",
)
TEACHER_PREPROCESSING = (
    "coordinates: public pixels divided by 640x480",
    "scripted velocity: public displacement/tick * 50 / 1000",
    "physics velocity: public world-units/second * 10 / 1000",
    "explicit display velocity fields: pixels/second / 1000 without conversion",
    "angular_velocity: public value divided by 10",
    "orientation: circle/unknown invalid; box sin(4a),cos(4a); triangle sin(a),cos(a)",
    "wide signed values: signed log1p with declared divisors",
)

ACTOR_VISION_V1 = TensorSchema(
    "actor-vision-v1", "actor_tracks", ACTOR_GLOBAL_FEATURES, ACTOR_BODY_FEATURES,
    ACTOR_PREPROCESSING,
)
TEACHER_V1 = TensorSchema(
    "teacher-v1", "teacher_state", TEACHER_GLOBAL_FEATURES, TEACHER_BODY_FEATURES,
    TEACHER_PREPROCESSING,
)

PROHIBITED_ACTOR_FIELDS = frozenset(
    {
        "tick",
        "score",
        "id",
        "chain_id",
        "projectile_hits",
        "age_ticks",
        "remaining_lifetime",
        "rot_timer",
        "rng_state",
        "state_hash",
        "snapshot",
        "future_spawns",
        "terminated",
        "truncated",
    }
)
