"""Development-only geometry search for one public directed steering pair.

The pair choice is held fixed.  Search varies only shot strength and cursor
geometry, rolls every branch to the same next-spawn-censored horizon in the
portable clone, and restores the source environment transactionally.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

from irisu_env import Action, ActionKind, EventKind
from irisu_rl.actions import (
    ActionSpec,
    SemanticAction,
    SemanticActionKind,
)

from .steering import SteeringDecision


GEOMETRY_SEARCH_VERSION = "r3d-directed-pair-geometry-search-v4"
_KNOWN_SHAPES = frozenset({"circle", "box", "triangle"})
_INTERIOR_SLOT_SPECS = (
    (25, "interior/center/weak", 0.0, SemanticActionKind.FIRE_WEAK),
    (26, "interior/center/strong", 0.0, SemanticActionKind.FIRE_STRONG),
    (27, "interior/upper/weak", -0.25, SemanticActionKind.FIRE_WEAK),
    (28, "interior/upper/strong", -0.25, SemanticActionKind.FIRE_STRONG),
    (29, "interior/lower/weak", 0.25, SemanticActionKind.FIRE_WEAK),
    (30, "interior/lower/strong", 0.25, SemanticActionKind.FIRE_STRONG),
    (
        31,
        "interior/deep-center/strong",
        0.125,
        SemanticActionKind.FIRE_STRONG,
    ),
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _plain_int(value: Any, name: str) -> int:
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    item = getattr(value, "item", None)
    if callable(item):
        return _plain_int(item(), name)
    raise TypeError(f"{name} must be an integer")


def _plain_float(value: Any, name: str) -> float:
    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        item = getattr(value, "item", None)
        if not callable(item):
            raise TypeError(f"{name} must be numeric")
        result = float(item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _event_kind(event: Mapping[str, Any]) -> int | None:
    raw = event.get("kind")
    if isinstance(raw, Integral) and not isinstance(raw, bool):
        return int(raw)
    name = event.get("kind_name")
    if isinstance(name, str):
        try:
            return int(EventKind[name.upper()])
        except KeyError:
            return None
    return None


@dataclass(frozen=True, slots=True)
class GeometrySearchConfig:
    """Finite candidate vocabulary and causal branch horizon."""

    horizon_ticks: int = 64
    velocity_lead_ticks: float = 1.0
    ticks_per_second: float = 50.0
    support_fractions: tuple[float, ...] = (0.35, 0.65, 0.85)
    support_clearance_sizes: float = 0.25
    grid_x_fractions: tuple[float, ...] = (0.25, 0.50, 0.75)
    grid_y_sizes: tuple[float, ...] = (0.60, 0.75, 0.90)
    max_candidates: int = 32

    def __post_init__(self) -> None:
        if (
            type(self.horizon_ticks) is not int
            or not 2 <= self.horizon_ticks <= 100_000
        ):
            raise ValueError("geometry horizon must be in [2, 100000]")
        if (
            type(self.max_candidates) is not int
            or not 1 <= self.max_candidates <= 64
        ):
            raise ValueError("geometry candidate cap must be in [1, 64]")
        for name in (
            "velocity_lead_ticks",
            "ticks_per_second",
            "support_clearance_sizes",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.ticks_per_second <= 0.0 or self.support_clearance_sizes <= 0.0:
            raise ValueError(
                "ticks per second and support clearance must be positive"
            )
        for name in ("support_fractions", "grid_x_fractions"):
            values = getattr(self, name)
            if (
                len(values) != 3
                or tuple(sorted(set(values))) != values
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    or not 0.0 < float(value) < 1.0
                    for value in values
                )
            ):
                raise ValueError(
                    f"{name} must contain three unique increasing fractions "
                    "in (0, 1)"
                )
        if (
            len(self.grid_y_sizes) != 3
            or tuple(sorted(set(self.grid_y_sizes))) != self.grid_y_sizes
            or any(
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or float(value) <= 0.0
                for value in self.grid_y_sizes
            )
        ):
            raise ValueError(
                "grid_y_sizes must contain three unique increasing positive values"
            )
        required = 1 + 2 * len(self.support_fractions)
        if self.max_candidates < required:
            raise ValueError(
                "candidate cap must retain the incumbent and all support shots"
            )

    def manifest(self) -> dict[str, object]:
        return {
            "version": GEOMETRY_SEARCH_VERSION,
            "horizon_ticks": self.horizon_ticks,
            "velocity_lead_ticks": float(self.velocity_lead_ticks),
            "ticks_per_second": float(self.ticks_per_second),
            "support_fractions": list(self.support_fractions),
            "support_clearance_sizes": float(self.support_clearance_sizes),
            "grid_x_fractions": list(self.grid_x_fractions),
            "grid_y_sizes": list(self.grid_y_sizes),
            "max_candidates": self.max_candidates,
            "candidate_slots": list(geometry_candidate_slots(self)),
            "causal_horizon": "min(configured ticks, ticks before next spawn)",
            "selection_rule": (
                "valid branches survival-nondominated by incumbent; "
                "reserve-band lexicographic public objective: viability below "
                "half gauge, score first at or above half gauge, then causal "
                "control; "
                "incumbent wins exact ties"
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    @property
    def slot_count(self) -> int:
        """Number of fixed selector outputs after applying the hard cap."""

        return len(geometry_candidate_slots(self))


def geometry_candidate_slots(
    config: GeometrySearchConfig | None = None,
) -> tuple[dict[str, object], ...]:
    """Return the state-invariant selector vocabulary and fixed slot indices."""

    resolved = GeometrySearchConfig() if config is None else config
    if not isinstance(resolved, GeometrySearchConfig):
        raise TypeError("geometry config must be a GeometrySearchConfig")
    slots: list[dict[str, object]] = [
        {"slot": 0, "family": "incumbent", "name": "incumbent"}
    ]
    for fraction_index, fraction in enumerate(resolved.support_fractions):
        for strength_index, strength in enumerate(("weak", "strong")):
            slot = 1 + 2 * fraction_index + strength_index
            if slot >= resolved.max_candidates:
                return tuple(slots)
            slots.append(
                {
                    "slot": slot,
                    "family": "shape-support",
                    "name": f"support/{fraction:g}/{strength}",
                    "support_fraction": float(fraction),
                    "strength": strength,
                }
            )
    grid_start = 1 + 2 * len(resolved.support_fractions)
    for x_index, x_fraction in enumerate(resolved.grid_x_fractions):
        for y_index, y_sizes in enumerate(resolved.grid_y_sizes):
            for strength_index, strength in enumerate(("weak", "strong")):
                slot = (
                    grid_start
                    + 2
                    * (x_index * len(resolved.grid_y_sizes) + y_index)
                    + strength_index
                )
                if slot >= resolved.max_candidates:
                    return tuple(slots)
                slots.append(
                    {
                        "slot": slot,
                        "family": "bounded-grid",
                        "name": (
                            f"grid/{x_fraction:g}/{y_sizes:g}/{strength}"
                        ),
                        "x_fraction": float(x_fraction),
                        "y_sizes": float(y_sizes),
                        "strength": strength,
                    }
                )
    for slot, name, local_y_sizes, kind in _INTERIOR_SLOT_SPECS:
        if slot >= resolved.max_candidates:
            return tuple(slots)
        slots.append(
            {
                "slot": slot,
                "family": "central-interior",
                "name": name,
                "local_x_sizes": 0.0,
                "local_y_sizes": local_y_sizes,
                "strength": (
                    "weak"
                    if kind is SemanticActionKind.FIRE_WEAK
                    else "strong"
                ),
            }
        )
    return tuple(slots)


@dataclass(frozen=True, slots=True)
class _PublicBody:
    identifier: int
    kind: str
    shape: str
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    size: float

    def predicted(self, seconds: float) -> _PredictedBody:
        return _PredictedBody(
            self.identifier,
            self.shape,
            self.x + self.vx * seconds,
            self.y + self.vy * seconds,
            self.angle + self.angular_velocity * seconds,
            self.size,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "id": self.identifier,
            "kind": self.kind,
            "shape": self.shape,
            "x": self.x,
            "y": self.y,
            "vx_display_per_second": self.vx,
            "vy_display_per_second": self.vy,
            "angle": self.angle,
            "angular_velocity": self.angular_velocity,
            "size": self.size,
        }


@dataclass(frozen=True, slots=True)
class _PredictedBody:
    identifier: int
    shape: str
    x: float
    y: float
    angle: float
    size: float


def _body(value: Mapping[str, Any]) -> _PublicBody:
    identifier = _plain_int(value.get("id"), "public body id")
    if identifier < 0:
        raise ValueError("public body id must be nonnegative")
    shape = str(value.get("shape", "unknown"))
    size = _plain_float(value.get("size"), "public body size")
    if shape not in _KNOWN_SHAPES:
        raise ValueError(f"directed-pair body {identifier} has unknown shape")
    if size <= 0.0:
        raise ValueError("public body size must be positive")
    lifecycle = str(value.get("lifecycle", ""))
    velocity_factor = (
        50.0 if lifecycle in {"scripted_falling", "falling"} else 10.0
    )
    vx = (
        _plain_float(
            value.get("vx_display_per_second"), "public body display vx"
        )
        if value.get("vx_display_per_second") is not None
        else _plain_float(value.get("vx", 0.0), "public body vx")
        * velocity_factor
    )
    vy = (
        _plain_float(
            value.get("vy_display_per_second"), "public body display vy"
        )
        if value.get("vy_display_per_second") is not None
        else _plain_float(value.get("vy", 0.0), "public body vy")
        * velocity_factor
    )
    return _PublicBody(
        identifier,
        str(value.get("kind", "")),
        shape,
        _plain_float(value.get("effect_x", value.get("x")), "public body x"),
        _plain_float(value.get("effect_y", value.get("y")), "public body y"),
        vx,
        vy,
        _plain_float(value.get("angle", 0.0), "public body angle"),
        _plain_float(
            value.get("angular_velocity", 0.0),
            "public body angular velocity",
        ),
        size,
    )


def _directed_pair(
    observation: Mapping[str, Any], decision: SteeringDecision
) -> tuple[_PublicBody, _PublicBody]:
    if not isinstance(observation, Mapping):
        raise TypeError("geometry search observation must be a public mapping")
    if not isinstance(decision, SteeringDecision):
        raise TypeError("geometry search requires a SteeringDecision")
    if (
        not decision.is_shot
        or decision.source_body_id is None
        or decision.destination_body_id is None
        or decision.source_body_id == decision.destination_body_id
    ):
        raise ValueError("geometry search requires a directed-pair shot")
    values = observation.get("bodies", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("public observation bodies must be a sequence")
    selected: dict[int, _PublicBody] = {}
    target_ids = {decision.source_body_id, decision.destination_body_id}
    seen: set[int] = set()
    for value in values:
        if not isinstance(value, Mapping):
            continue
        identifier = _plain_int(value.get("id"), "public body id")
        if identifier in seen:
            raise ValueError("public observation body ids must be unique")
        seen.add(identifier)
        if identifier in target_ids:
            selected[identifier] = _body(value)
    if set(selected) != target_ids:
        raise ValueError("directed-pair body is absent from the public observation")
    source = selected[decision.source_body_id]
    destination = selected[decision.destination_body_id]
    if source.kind != "piece" or destination.kind != "piece":
        raise ValueError("directed-pair geometry search supports pieces only")
    return source, destination


def _rotated_vertices(body: _PredictedBody) -> tuple[tuple[float, float], ...]:
    half = body.size / 2.0
    if body.shape == "box":
        local = ((-half, -half), (-half, half), (half, half), (half, -half))
    elif body.shape == "triangle":
        # Measured clone fixture: lower-left, upper-left, upper-right in
        # screen coordinates before rotation.
        local = ((-half, -half), (-half, half), (half, half))
    else:
        raise ValueError("circles do not have polygon vertices")
    cosine, sine = math.cos(body.angle), math.sin(body.angle)
    return tuple(
        (x * cosine - y * sine, x * sine + y * cosine) for x, y in local
    )


def _horizontal_extents(body: _PredictedBody) -> tuple[float, float]:
    if body.shape == "circle":
        radius = body.size / 2.0
        return -radius, radius
    values = tuple(x for x, _ in _rotated_vertices(body))
    return min(values), max(values)


def _lower_support_y(body: _PredictedBody, local_x: float) -> float:
    if body.shape == "circle":
        radius = body.size / 2.0
        inside = max(0.0, radius * radius - local_x * local_x)
        return math.sqrt(inside)
    vertices = _rotated_vertices(body)
    intersections: list[float] = []
    for first, second in zip(vertices, vertices[1:] + vertices[:1], strict=True):
        x1, y1 = first
        x2, y2 = second
        if local_x < min(x1, x2) - 1e-9 or local_x > max(x1, x2) + 1e-9:
            continue
        if math.isclose(x1, x2, abs_tol=1e-12):
            if math.isclose(local_x, x1, abs_tol=1e-9):
                intersections.extend((y1, y2))
            continue
        ratio = (local_x - x1) / (x2 - x1)
        if -1e-9 <= ratio <= 1.0 + 1e-9:
            intersections.append(y1 + ratio * (y2 - y1))
    if not intersections:
        raise RuntimeError("shape-support ray missed the public source polygon")
    return max(intersections)


def _contains_local_point(
    body: _PredictedBody, local_x: float, local_y: float
) -> bool:
    """Whether a source-local cursor point lies on or inside its public shape."""

    half = body.size / 2.0
    epsilon = max(body.size, 1.0) * 1e-9
    if body.shape == "circle":
        return (
            local_x * local_x + local_y * local_y
            <= half * half + epsilon
        )
    if body.shape == "box":
        return (
            abs(local_x) <= half + epsilon
            and abs(local_y) <= half + epsilon
        )
    if body.shape == "triangle":
        # Measured local screen-space fixture:
        # (-half,-half), (-half,+half), (+half,+half).
        return (
            local_x >= -half - epsilon
            and local_y <= half + epsilon
            and local_y >= local_x - epsilon
        )
    return False


def _world_point(
    body: _PredictedBody, local_x: float, local_y: float
) -> tuple[float, float]:
    cosine, sine = math.cos(body.angle), math.sin(body.angle)
    return (
        body.x + local_x * cosine - local_y * sine,
        body.y + local_x * sine + local_y * cosine,
    )


def _decision_manifest(decision: SteeringDecision) -> dict[str, object]:
    return {
        "kind": int(decision.action.kind),
        "wait_ticks": int(decision.action.wait_ticks),
        "x_norm": float(decision.action.x_norm),
        "y_norm": float(decision.action.y_norm),
        "intent": decision.intent.value,
        "source_body_id": decision.source_body_id,
        "destination_body_id": decision.destination_body_id,
        "destination_chain_id": decision.destination_chain_id,
        "impact_x_sizes": float(decision.impact_x_sizes),
        "impact_y_sizes": float(decision.impact_y_sizes),
        "correction_index": decision.correction_index,
        "reason": decision.reason,
    }


@dataclass(frozen=True, slots=True)
class GeometryCandidate:
    """One legal cursor/strength alternative with a fixed directed pair."""

    name: str
    family: str
    ordinal: int
    decision: SteeringDecision
    cursor_x: int
    cursor_y: int

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "family": self.family,
            "ordinal": self.ordinal,
            "cursor_x": self.cursor_x,
            "cursor_y": self.cursor_y,
            "decision": _decision_manifest(self.decision),
        }


@dataclass(frozen=True, slots=True)
class GeometryCandidateSet:
    """Identity-bound deterministic candidates for one public pair state."""

    config: GeometrySearchConfig
    action_spec_sha256: str
    source_manifest: Mapping[str, object]
    destination_manifest: Mapping[str, object]
    predicted_surface_gap_sizes: float
    candidates: tuple[GeometryCandidate, ...]

    def __post_init__(self) -> None:
        if (
            not self.candidates
            or self.candidates[0].family != "incumbent"
            or self.candidates[0].ordinal != 0
        ):
            raise ValueError("geometry candidates must begin with the incumbent")
        if len(self.candidates) > self.config.max_candidates:
            raise ValueError("geometry candidate cap was exceeded")
        ordinals = tuple(candidate.ordinal for candidate in self.candidates)
        if (
            ordinals != tuple(sorted(set(ordinals)))
            or ordinals[-1] >= self.config.slot_count
        ):
            raise ValueError("geometry candidates require unique fixed slots")

    def manifest(self) -> dict[str, object]:
        return {
            "version": GEOMETRY_SEARCH_VERSION,
            "config": self.config.manifest(),
            "action_spec_sha256": self.action_spec_sha256,
            "source": dict(self.source_manifest),
            "destination": dict(self.destination_manifest),
            "predicted_surface_gap_sizes": self.predicted_surface_gap_sizes,
            "slot_count": self.config.slot_count,
            "availability_mask": list(self.availability_mask),
            "candidates": [candidate.manifest() for candidate in self.candidates],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    @property
    def availability_mask(self) -> tuple[bool, ...]:
        available = {candidate.ordinal for candidate in self.candidates}
        return tuple(slot in available for slot in range(self.config.slot_count))

    def candidate_at(self, slot: int) -> GeometryCandidate | None:
        if isinstance(slot, bool) or not isinstance(slot, int):
            raise TypeError("geometry candidate slot must be an integer")
        if not 0 <= slot < self.config.slot_count:
            raise IndexError(slot)
        return next(
            (
                candidate
                for candidate in self.candidates
                if candidate.ordinal == slot
            ),
            None,
        )


def _shot_decision(
    incumbent: SteeringDecision,
    spec: ActionSpec,
    source: _PublicBody,
    kind: SemanticActionKind,
    x: float,
    y: float,
    *,
    reason: str,
) -> SteeringDecision | None:
    if (
        not 0.0 <= x < spec.client_width
        or not 0.0 <= y < spec.client_height
    ):
        return None
    constructor = (
        SemanticAction.weak
        if kind is SemanticActionKind.FIRE_WEAK
        else SemanticAction.strong
    )
    action = spec.validate(
        constructor(x / spec.client_width, y / spec.client_height)
    )
    return SteeringDecision(
        action,
        incumbent.intent,
        source_body_id=incumbent.source_body_id,
        destination_body_id=incumbent.destination_body_id,
        destination_chain_id=incumbent.destination_chain_id,
        impact_x_sizes=(x - source.x) / source.size,
        impact_y_sizes=(y - source.y) / source.size,
        correction_index=incumbent.correction_index,
        reason=reason,
    )


def enumerate_geometry_candidates(
    observation: Mapping[str, Any],
    incumbent: SteeringDecision,
    *,
    config: GeometrySearchConfig | None = None,
    action_spec: ActionSpec | None = None,
) -> GeometryCandidateSet:
    """Enumerate public, shape-aware alternatives for an already chosen pair."""

    resolved = GeometrySearchConfig() if config is None else config
    spec = ActionSpec() if action_spec is None else action_spec
    if not isinstance(resolved, GeometrySearchConfig):
        raise TypeError("geometry config must be a GeometrySearchConfig")
    if not isinstance(spec, ActionSpec):
        raise TypeError("geometry action spec must be an ActionSpec")
    source, destination = _directed_pair(observation, incumbent)
    incumbent_action = spec.validate(incumbent.action)
    if SemanticActionKind(incumbent_action.kind) is SemanticActionKind.WAIT:
        raise ValueError("geometry incumbent must be a shot")

    seconds = resolved.velocity_lead_ticks / resolved.ticks_per_second
    predicted_source = source.predicted(seconds)
    predicted_destination = destination.predicted(seconds)
    source_left, source_right = _horizontal_extents(predicted_source)
    destination_left, destination_right = _horizontal_extents(
        predicted_destination
    )
    delta = predicted_destination.x - predicted_source.x
    if math.isclose(delta, 0.0, abs_tol=1e-9):
        delta = (destination.vx - source.vx) or (
            spec.client_width / 2.0 - predicted_source.x
        )
    direction = 1.0 if delta >= 0.0 else -1.0
    if direction > 0.0:
        opposing_support = source_left
        surface_gap = (
            predicted_destination.x
            + destination_left
            - predicted_source.x
            - source_right
        )
    else:
        opposing_support = source_right
        surface_gap = (
            predicted_source.x
            + source_left
            - predicted_destination.x
            - destination_right
        )
    mean_size = max((source.size + destination.size) / 2.0, 1e-9)

    candidates: list[GeometryCandidate] = []
    seen_actions: set[tuple[int, int, int]] = set()

    def append(
        slot: int, name: str, family: str, decision: SteeringDecision
    ) -> None:
        if slot >= resolved.max_candidates:
            return
        action = spec.validate(decision.action)
        cursor_x, cursor_y = spec.client_point(action)
        key = (int(action.kind), cursor_x, cursor_y)
        if key in seen_actions:
            return
        seen_actions.add(key)
        candidates.append(
            GeometryCandidate(name, family, slot, decision, cursor_x, cursor_y)
        )

    append(0, "incumbent", "incumbent", incumbent)
    strengths = (
        ("weak", SemanticActionKind.FIRE_WEAK),
        ("strong", SemanticActionKind.FIRE_STRONG),
    )
    for fraction_index, fraction in enumerate(resolved.support_fractions):
        local_x = opposing_support * fraction
        local_y = _lower_support_y(predicted_source, local_x)
        x = predicted_source.x + local_x
        y = (
            predicted_source.y
            + local_y
            + resolved.support_clearance_sizes * source.size
        )
        for strength_index, (strength_name, kind) in enumerate(strengths):
            decision = _shot_decision(
                incumbent,
                spec,
                source,
                kind,
                x,
                y,
                reason=(
                    "shape-support horizontal-impulse geometry "
                    f"(fraction={fraction:g}, strength={strength_name})"
                ),
            )
            if decision is not None:
                append(
                    1 + 2 * fraction_index + strength_index,
                    f"support/{fraction:g}/{strength_name}",
                    "shape-support",
                    decision,
                )

    grid_start = 1 + 2 * len(resolved.support_fractions)
    for x_index, x_fraction in enumerate(resolved.grid_x_fractions):
        local_x = opposing_support * x_fraction
        x = predicted_source.x + local_x
        for y_index, y_sizes in enumerate(resolved.grid_y_sizes):
            y = predicted_source.y + y_sizes * source.size
            for strength_index, (strength_name, kind) in enumerate(strengths):
                decision = _shot_decision(
                    incumbent,
                    spec,
                    source,
                    kind,
                    x,
                    y,
                    reason=(
                        "bounded directed-pair geometry grid "
                        f"(x_fraction={x_fraction:g}, "
                        f"y_sizes={y_sizes:g}, strength={strength_name})"
                    ),
                )
                if decision is not None:
                    append(
                        grid_start
                        + 2 * (
                            x_index * len(resolved.grid_y_sizes) + y_index
                        )
                        + strength_index,
                        f"grid/{x_fraction:g}/{y_sizes:g}/{strength_name}",
                        "bounded-grid",
                        decision,
                    )

    for slot, name, local_y_sizes, kind in _INTERIOR_SLOT_SPECS:
        local_x = 0.0
        local_y = local_y_sizes * source.size
        if not _contains_local_point(predicted_source, local_x, local_y):
            continue
        x, y = _world_point(predicted_source, local_x, local_y)
        strength_name = (
            "weak"
            if kind is SemanticActionKind.FIRE_WEAK
            else "strong"
        )
        decision = _shot_decision(
            incumbent,
            spec,
            source,
            kind,
            x,
            y,
            reason=(
                "shape-local central/interior downward-drive geometry "
                f"(local_y_sizes={local_y_sizes:g}, "
                f"strength={strength_name})"
            ),
        )
        if decision is not None:
            append(slot, name, "central-interior", decision)

    return GeometryCandidateSet(
        resolved,
        spec.sha256,
        source.manifest(),
        destination.manifest(),
        max(0.0, surface_gap) / mean_size,
        tuple(candidates),
    )


def causal_tick_budget(
    observation: Mapping[str, Any], maximum_ticks: int
) -> int:
    """Return ticks available before the next public cadence spawn."""

    if type(maximum_ticks) is not int or maximum_ticks < 1:
        raise ValueError("maximum causal ticks must be a positive integer")
    tick = _plain_int(observation.get("tick"), "public observation tick")
    difficulty = observation.get("difficulty")
    if not isinstance(difficulty, Mapping):
        raise TypeError("public difficulty must be a mapping")
    interval = _plain_int(
        difficulty.get("spawn_interval_ticks"), "public spawn interval"
    )
    if tick < 0 or interval < 1:
        raise ValueError("public tick and spawn interval are invalid")
    remainder = tick % interval
    safe = 0 if remainder == 0 else interval - remainder
    return min(maximum_ticks, safe)


def _public_state_signature(observation: Mapping[str, Any]) -> tuple[object, ...]:
    bodies = observation.get("bodies", ())
    if not isinstance(bodies, Sequence) or isinstance(bodies, (str, bytes)):
        raise TypeError("public observation bodies must be a sequence")
    body_values: list[tuple[object, ...]] = []
    for value in bodies:
        if not isinstance(value, Mapping):
            continue
        body_values.append(
            (
                _plain_int(value.get("id"), "public body id"),
                str(value.get("kind", "")),
                str(value.get("shape", "unknown")),
                str(value.get("lifecycle", "")),
                _plain_int(value.get("color", -1), "public body color"),
                _plain_float(value.get("x"), "public body x"),
                _plain_float(value.get("y"), "public body y"),
                _plain_float(value.get("vx", 0.0), "public body vx"),
                _plain_float(value.get("vy", 0.0), "public body vy"),
                _plain_float(value.get("angle", 0.0), "public body angle"),
                _plain_float(
                    value.get("angular_velocity", 0.0),
                    "public body angular velocity",
                ),
                _plain_float(value.get("size"), "public body size"),
                _plain_int(value.get("chain_id", 0), "public body chain id"),
            )
        )
    body_values.sort(key=lambda value: int(value[0]))
    return (
        _plain_int(observation.get("tick"), "public observation tick"),
        _plain_int(observation.get("score", 0), "public score"),
        _plain_int(observation.get("gauge", 0), "public gauge"),
        _plain_int(observation.get("gauge_max", 0), "public gauge max"),
        _plain_int(observation.get("level", 0), "public level"),
        _plain_int(
            observation.get("highest_chain", 0), "public highest chain"
        ),
        _plain_int(
            observation.get("qualifying_clear_count", 0),
            "public qualifying clear count",
        ),
        bool(observation.get("terminated", False)),
        bool(observation.get("truncated", False)),
        tuple(body_values),
    )


def _pair_distance_sizes(
    observation: Mapping[str, Any], source_id: int, destination_id: int
) -> float | None:
    values = observation.get("bodies", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("public observation bodies must be a sequence")
    selected: dict[int, tuple[float, float, float]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            continue
        identifier = _plain_int(value.get("id"), "public body id")
        if identifier in {source_id, destination_id}:
            selected[identifier] = (
                _plain_float(value.get("x"), "public body x"),
                _plain_float(value.get("y"), "public body y"),
                _plain_float(value.get("size"), "public body size"),
            )
    if set(selected) != {source_id, destination_id}:
        return None
    source = selected[source_id]
    destination = selected[destination_id]
    scale = max((source[2] + destination[2]) / 2.0, 1e-9)
    return math.hypot(
        destination[0] - source[0], destination[1] - source[1]
    ) / scale


@dataclass(frozen=True, slots=True)
class GeometryBranchOutcome:
    """Public causal measurements for one geometry branch."""

    candidate: GeometryCandidate
    score_gain: int
    alive: bool
    survival_ticks: int
    final_gauge: int
    gauge_max: int
    qualifying_clear_gain: int
    highest_chain_gain: int
    intended_source_hits: int
    intended_pair_joined: bool
    pair_closure_sizes: float
    invalid_actions: int

    def __post_init__(self) -> None:
        if (
            type(self.gauge_max) is not int
            or self.gauge_max < 1
            or type(self.final_gauge) is not int
            or self.final_gauge > self.gauge_max
        ):
            raise ValueError("geometry branch gauge evidence is invalid")

    @property
    def selectable(self) -> bool:
        return self.invalid_actions == 0

    @property
    def reserve_target(self) -> int:
        return self.gauge_max // 2

    @property
    def protected_reserve(self) -> int:
        return min(self.final_gauge, self.reserve_target)

    def survival_nondominated_by(
        self, incumbent: GeometryBranchOutcome
    ) -> bool:
        if self.gauge_max != incumbent.gauge_max:
            raise ValueError("geometry branches use different gauge maxima")
        return (
            int(self.alive) >= int(incumbent.alive)
            and self.survival_ticks >= incumbent.survival_ticks
            and self.protected_reserve >= incumbent.protected_reserve
        )

    @property
    def objective(self) -> tuple[int | float, ...]:
        """Protect half gauge, then spend surplus capacity on score."""

        return (
            int(self.alive),
            self.survival_ticks,
            self.protected_reserve,
            self.score_gain,
            self.qualifying_clear_gain,
            self.final_gauge,
            int(self.intended_pair_joined),
            self.intended_source_hits,
            self.pair_closure_sizes,
            self.highest_chain_gain,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "candidate_name": self.candidate.name,
            "candidate_ordinal": self.candidate.ordinal,
            "score_gain": self.score_gain,
            "alive": self.alive,
            "survival_ticks": self.survival_ticks,
            "final_gauge": self.final_gauge,
            "gauge_max": self.gauge_max,
            "reserve_target": self.reserve_target,
            "protected_reserve": self.protected_reserve,
            "qualifying_clear_gain": self.qualifying_clear_gain,
            "highest_chain_gain": self.highest_chain_gain,
            "intended_source_hits": self.intended_source_hits,
            "intended_pair_joined": self.intended_pair_joined,
            "pair_closure_sizes": self.pair_closure_sizes,
            "invalid_actions": self.invalid_actions,
            "objective": list(self.objective),
        }


@dataclass(frozen=True, slots=True)
class GeometrySearchResult:
    """Selected first shot and all public branch evidence."""

    candidate_set: GeometryCandidateSet
    configured_horizon_ticks: int
    causal_horizon_ticks: int
    selected_candidate: GeometryCandidate
    strictly_improved: bool
    outcomes: tuple[GeometryBranchOutcome, ...]

    @property
    def decision(self) -> SteeringDecision:
        return self.selected_candidate.decision

    def manifest(self) -> dict[str, object]:
        return {
            "version": GEOMETRY_SEARCH_VERSION,
            "candidate_set_sha256": self.candidate_set.sha256,
            "configured_horizon_ticks": self.configured_horizon_ticks,
            "causal_horizon_ticks": self.causal_horizon_ticks,
            "selected_candidate": self.selected_candidate.name,
            "strictly_improved": self.strictly_improved,
            "outcomes": [outcome.manifest() for outcome in self.outcomes],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def _step_duration(action: Action) -> int:
    kind = ActionKind.parse(action.kind)
    return int(action.wait_ticks) if kind is ActionKind.WAIT else 1


def evaluate_geometry_candidate(
    env: Any,
    initial: Mapping[str, Any],
    candidate: GeometryCandidate,
    *,
    horizon_ticks: int,
    action_spec: ActionSpec,
) -> GeometryBranchOutcome:
    current = initial
    start_tick = _plain_int(initial.get("tick"), "public observation tick")
    events: list[Mapping[str, Any]] = []
    invalid_actions = 0
    terminated = bool(initial.get("terminated", False))
    truncated = bool(initial.get("truncated", False))
    primitives = candidate.decision.primitive_actions(action_spec)
    for action in primitives:
        if terminated or truncated:
            break
        elapsed = _plain_int(
            current.get("tick"), "public observation tick"
        ) - start_tick
        remaining = horizon_ticks - elapsed
        duration = _step_duration(action)
        if duration > remaining:
            raise RuntimeError("geometry branch primitive exceeds causal horizon")
        previous_tick = _plain_int(
            current.get("tick"), "public observation tick"
        )
        current, _reward, terminated, truncated, info = env.step(action)
        if not isinstance(current, Mapping) or not isinstance(info, Mapping):
            raise TypeError("portable branch transition must expose public mappings")
        advanced = _plain_int(
            current.get("tick"), "public observation tick"
        ) - previous_tick
        if advanced < 1 or advanced > duration:
            raise RuntimeError("geometry branch action violated public time")
        if not (terminated or truncated) and advanced != duration:
            raise RuntimeError("nonterminal geometry action ended before its duration")
        if (
            bool(current.get("terminated", False)) != bool(terminated)
            or bool(current.get("truncated", False)) != bool(truncated)
        ):
            raise RuntimeError(
                "geometry transition flags disagree with public observation"
            )
        transition_events = tuple(
            event
            for event in info.get("events", ())
            if isinstance(event, Mapping)
        )
        events.extend(transition_events)
        invalid_events = sum(
            _event_kind(event) == int(EventKind.INVALID_ACTION)
            for event in transition_events
        )
        invalid_actions += max(
            int(bool(info.get("invalid_action", False))), invalid_events
        )

    elapsed = _plain_int(current.get("tick"), "public observation tick") - start_tick
    if not terminated and not truncated and elapsed < horizon_ticks:
        wait = Action.wait(horizon_ticks - elapsed)
        previous_tick = _plain_int(
            current.get("tick"), "public observation tick"
        )
        current, _reward, terminated, truncated, info = env.step(wait)
        if not isinstance(current, Mapping) or not isinstance(info, Mapping):
            raise TypeError("portable branch transition must expose public mappings")
        advanced = _plain_int(
            current.get("tick"), "public observation tick"
        ) - previous_tick
        if advanced < 1 or advanced > horizon_ticks - elapsed:
            raise RuntimeError("geometry coast violated its causal horizon")
        if not (terminated or truncated) and advanced != horizon_ticks - elapsed:
            raise RuntimeError("nonterminal geometry coast ended early")
        if (
            bool(current.get("terminated", False)) != bool(terminated)
            or bool(current.get("truncated", False)) != bool(truncated)
        ):
            raise RuntimeError(
                "geometry transition flags disagree with public observation"
            )
        transition_events = tuple(
            event
            for event in info.get("events", ())
            if isinstance(event, Mapping)
        )
        events.extend(transition_events)
        invalid_events = sum(
            _event_kind(event) == int(EventKind.INVALID_ACTION)
            for event in transition_events
        )
        invalid_actions += max(
            int(bool(info.get("invalid_action", False))), invalid_events
        )

    final_tick = _plain_int(current.get("tick"), "public observation tick")
    survival_ticks = final_tick - start_tick
    if not 0 <= survival_ticks <= horizon_ticks:
        raise RuntimeError("geometry branch exceeded its causal horizon")
    source_id = candidate.decision.source_body_id
    destination_id = candidate.decision.destination_body_id
    assert source_id is not None and destination_id is not None
    projectile_ids = {
        _plain_int(event.get("a", -1), "shot projectile id")
        for event in events
        if _event_kind(event) == int(EventKind.SHOT_FIRED)
    }
    source_hit_pairs = {
        (
            _plain_int(event.get("a", -1), "projectile hit source"),
            _plain_int(event.get("b", -1), "projectile hit target"),
        )
        for event in events
        if _event_kind(event) == int(EventKind.PROJECTILE_HIT)
        and _plain_int(event.get("a", -1), "projectile hit source")
        in projectile_ids
        and _plain_int(event.get("b", -1), "projectile hit target") == source_id
    }
    joined = any(
        _event_kind(event) == int(EventKind.CHAIN_JOINED)
        and {
            _plain_int(event.get("a", -1), "chain join body a"),
            _plain_int(event.get("b", -1), "chain join body b"),
        }
        == {source_id, destination_id}
        for event in events
    )
    initial_distance = _pair_distance_sizes(initial, source_id, destination_id)
    final_distance = _pair_distance_sizes(current, source_id, destination_id)
    closure = (
        0.0
        if initial_distance is None or final_distance is None
        else initial_distance - final_distance
    )
    alive = not (
        bool(terminated)
        or bool(truncated)
        or bool(current.get("terminated", False))
        or bool(current.get("truncated", False))
    )
    initial_gauge_max = _plain_int(
        initial.get("gauge_max"), "public maximum gauge"
    )
    final_gauge_max = _plain_int(
        current.get("gauge_max"), "public maximum gauge"
    )
    if initial_gauge_max < 1 or final_gauge_max != initial_gauge_max:
        raise RuntimeError("geometry branch maximum gauge changed")
    return GeometryBranchOutcome(
        candidate,
        _plain_int(current.get("score", 0), "public score")
        - _plain_int(initial.get("score", 0), "public score"),
        alive,
        survival_ticks,
        _plain_int(current.get("gauge", 0), "public gauge"),
        final_gauge_max,
        _plain_int(
            current.get("qualifying_clear_count", 0),
            "public qualifying clear count",
        )
        - _plain_int(
            initial.get("qualifying_clear_count", 0),
            "public qualifying clear count",
        ),
        _plain_int(
            current.get("highest_chain", 0), "public highest chain"
        )
        - _plain_int(
            initial.get("highest_chain", 0), "public highest chain"
        ),
        len(source_hit_pairs),
        joined,
        closure,
        invalid_actions,
    )


class DirectedPairGeometrySearch:
    """Transactional portable search over one directed pair's shot geometry."""

    def __init__(
        self,
        *,
        config: GeometrySearchConfig | None = None,
        action_spec: ActionSpec | None = None,
    ) -> None:
        self.config = GeometrySearchConfig() if config is None else config
        self.action_spec = ActionSpec() if action_spec is None else action_spec
        if not isinstance(self.config, GeometrySearchConfig):
            raise TypeError("geometry config must be a GeometrySearchConfig")
        if not isinstance(self.action_spec, ActionSpec):
            raise TypeError("geometry action spec must be an ActionSpec")

    def identity_manifest(self) -> dict[str, object]:
        return {
            "version": GEOMETRY_SEARCH_VERSION,
            "config": self.config.manifest(),
            "action_spec": self.action_spec.manifest(),
            "public_inputs": [
                "source/destination id,shape,size,position,velocity,angle",
                "tick and spawn interval",
                "public transition observations and events",
            ],
            "backend": "portable-clone-only",
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())

    def search(
        self,
        env: Any,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
    ) -> GeometrySearchResult:
        if getattr(env, "physics_backend", None) != "portable":
            raise ValueError("directed-pair geometry search requires portable backend")
        if bool(observation.get("terminated", False)) or bool(
            observation.get("truncated", False)
        ):
            raise ValueError("cannot geometry-search a terminal observation")
        candidates = enumerate_geometry_candidates(
            observation,
            incumbent,
            config=self.config,
            action_spec=self.action_spec,
        )
        horizon = causal_tick_budget(observation, self.config.horizon_ticks)
        if horizon < 2:
            return GeometrySearchResult(
                candidates,
                self.config.horizon_ticks,
                horizon,
                candidates.candidates[0],
                False,
                (),
            )
        clone = getattr(env, "clone_state", None)
        restore = getattr(env, "restore_state", None)
        if not callable(clone) or not callable(restore):
            raise TypeError("portable geometry environment lacks clone/restore")
        expected = _public_state_signature(observation)
        snapshot = clone()
        outcomes: list[GeometryBranchOutcome] = []
        try:
            for candidate in candidates.candidates:
                restored = restore(snapshot)
                if not isinstance(restored, Mapping):
                    raise TypeError("portable restore must return a public mapping")
                if _public_state_signature(restored) != expected:
                    raise RuntimeError(
                        "portable restore disagrees with the supplied public state"
                    )
                outcomes.append(
                    evaluate_geometry_candidate(
                        env,
                        restored,
                        candidate,
                        horizon_ticks=horizon,
                        action_spec=self.action_spec,
                    )
                )
        finally:
            restore(snapshot)

        incumbent_outcome = outcomes[0]
        if not incumbent_outcome.selectable:
            raise RuntimeError("incumbent geometry branch emitted an invalid action")
        eligible = tuple(
            outcome
            for outcome in outcomes
            if outcome.selectable
            and outcome.survival_nondominated_by(incumbent_outcome)
        )
        winner = max(
            eligible,
            key=lambda outcome: (outcome.objective, -outcome.candidate.ordinal),
        )
        strictly_improved = winner.objective > incumbent_outcome.objective
        if not strictly_improved:
            winner = incumbent_outcome
        return GeometrySearchResult(
            candidates,
            self.config.horizon_ticks,
            horizon,
            winner.candidate,
            strictly_improved,
            tuple(outcomes),
        )

    def act(
        self,
        env: Any,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
    ) -> SteeringDecision:
        return self.search(env, observation, incumbent).decision

    choose = act


__all__ = [
    "DirectedPairGeometrySearch",
    "GEOMETRY_SEARCH_VERSION",
    "GeometryBranchOutcome",
    "GeometryCandidate",
    "GeometryCandidateSet",
    "GeometrySearchConfig",
    "GeometrySearchResult",
    "causal_tick_budget",
    "enumerate_geometry_candidates",
    "evaluate_geometry_candidate",
    "geometry_candidate_slots",
]
