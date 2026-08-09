"""Deterministic pointer-aware expert anchors and search candidates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from irisu_env import ActionKind

from .action import PointerActionSpec


_TARGETABLE_LIFECYCLES = frozenset(
    {"scripted_falling", "dynamic_fresh", "confirmed", "rotten"}
)


@dataclass(frozen=True, slots=True)
class PointerExpertDecision:
    """A semantic decision whose pointer is a stable public body ID."""

    kind: int
    wait_ticks: int = 1
    target_body_id: int | None = None
    template_index: int = 7

    def __post_init__(self) -> None:
        kind = ActionKind.parse(self.kind)
        if kind not in (
            ActionKind.WAIT,
            ActionKind.WEAK_SHOT,
            ActionKind.STRONG_SHOT,
        ):
            raise ValueError("pointer expert kind must be wait, weak, or strong")
        if (
            isinstance(self.wait_ticks, bool)
            or not isinstance(self.wait_ticks, int)
            or self.wait_ticks < 1
        ):
            raise ValueError("pointer expert wait_ticks must be positive")
        if kind is not ActionKind.WAIT and (
            isinstance(self.target_body_id, bool)
            or not isinstance(self.target_body_id, int)
            or self.target_body_id < 0
        ):
            raise ValueError("pointer expert shot requires a nonnegative body ID")
        if (
            isinstance(self.template_index, bool)
            or not isinstance(self.template_index, int)
            or self.template_index < 0
        ):
            raise ValueError("pointer expert template index must be nonnegative")

    @classmethod
    def wait(cls, ticks: int = 1) -> PointerExpertDecision:
        return cls(int(ActionKind.WAIT), wait_ticks=int(ticks))

    @classmethod
    def weak(
        cls, target_body_id: int, *, template_index: int = 7
    ) -> PointerExpertDecision:
        return cls(
            int(ActionKind.WEAK_SHOT),
            target_body_id=int(target_body_id),
            template_index=int(template_index),
        )

    @classmethod
    def strong(
        cls, target_body_id: int, *, template_index: int = 7
    ) -> PointerExpertDecision:
        return cls(
            int(ActionKind.STRONG_SHOT),
            target_body_id=int(target_body_id),
            template_index=int(template_index),
        )

    @property
    def primitive_ticks(self) -> int:
        """Ticks consumed by the deployment-v1 press/release macro."""

        return self.wait_ticks if self.kind == int(ActionKind.WAIT) else 2


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """A named decision; the name keeps anchor inclusion auditable."""

    name: str
    decision: PointerExpertDecision


def _pieces(observation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        body
        for body in observation.get("bodies", ())
        if body.get("kind") == "piece"
        and body.get("lifecycle") in _TARGETABLE_LIFECYCLES
    ]


def _geometry_key(body: Mapping[str, Any]) -> tuple[object, ...]:
    """Public semantic ordering; ID is deliberately excluded."""

    return (
        str(body.get("lifecycle", "")),
        int(body.get("color", -1)),
        float(body.get("x", 0.0)),
        float(body.get("y", 0.0)),
        float(body.get("vx", 0.0)),
        float(body.get("vy", 0.0)),
        float(body.get("size", 0.0)),
        int(body.get("projectile_hits", 0)),
        int(body.get("age_ticks", 0)),
        int(body.get("remaining_lifetime", 0)),
        int(body.get("rot_timer", 0)),
    )


def diverse_template_indices(spec: PointerActionSpec) -> tuple[int, ...]:
    """Return a deterministic farthest-first traversal of template geometry."""

    templates = spec.templates
    x_values = tuple(float(value) for value in spec.x_radius_offsets)
    y_values = tuple(float(value) for value in spec.y_radius_offsets)
    x_mid = (min(x_values) + max(x_values)) / 2.0
    y_mid = (min(y_values) + max(y_values)) / 2.0
    x_range = max(x_values) - min(x_values)
    y_range = max(y_values) - min(y_values)
    x_scale = x_range if x_range > 0.0 else 1.0
    y_scale = y_range if y_range > 0.0 else 1.0

    def point(index: int) -> tuple[float, float]:
        x, y = templates[index]
        return ((float(x) - x_mid) / x_scale, (float(y) - y_mid) / y_scale)

    remaining = set(range(spec.template_count))
    first = min(
        remaining,
        key=lambda index: (
            abs(point(index)[0]) + abs(point(index)[1]),
            index,
        ),
    )
    order = [first]
    remaining.remove(first)
    while remaining:
        selected_points = tuple(point(index) for index in order)

        def priority(index: int) -> tuple[float, int]:
            x, y = point(index)
            minimum_distance = min(
                (x - other_x) ** 2 + (y - other_y) ** 2
                for other_x, other_y in selected_points
            )
            return minimum_distance, -index

        next_index = max(remaining, key=priority)
        order.append(next_index)
        remaining.remove(next_index)
    return tuple(order)


def _central_template(spec: PointerActionSpec) -> int:
    """Return the zero-x, middle-y-radius template deterministically."""

    offsets = tuple(float(value) for value in spec.x_radius_offsets)
    y_offsets = tuple(float(value) for value in spec.y_radius_offsets)
    x_index = min(range(len(offsets)), key=lambda index: (abs(offsets[index]), index))
    y_index = min(
        range(len(y_offsets)),
        key=lambda index: (abs(y_offsets[index] - 2.0), index),
    )
    # PointerActionSpec templates are x-major, then y-radius-offset.
    return x_index * len(y_offsets) + y_index


def _template_for_offset(
    spec: PointerActionSpec, offset: float, *, y_radius_offset: float = 2.0
) -> int:
    offsets = tuple(float(value) for value in spec.x_radius_offsets)
    y_offsets = tuple(float(value) for value in spec.y_radius_offsets)
    x_index = min(
        range(len(offsets)),
        key=lambda index: (abs(offsets[index] - offset), index),
    )
    y_index = min(
        range(len(y_offsets)),
        key=lambda index: (abs(y_offsets[index] - y_radius_offset), index),
    )
    return x_index * len(y_offsets) + y_index


def matcher_anchor(
    observation: Mapping[str, Any],
    spec: PointerActionSpec | None = None,
) -> PointerExpertDecision:
    """Target the lower member of the closest visible same-color pair."""

    resolved = PointerActionSpec() if spec is None else spec
    pieces = [
        body
        for body in _pieces(observation)
        if body.get("lifecycle") in {"scripted_falling", "dynamic_fresh"}
    ]
    pairs: list[
        tuple[
            float,
            tuple[object, ...],
            tuple[object, ...],
            int,
            int,
            Mapping[str, Any],
            Mapping[str, Any],
        ]
    ] = []
    for index, first in enumerate(pieces):
        for second in pieces[index + 1 :]:
            if first.get("color") != second.get("color"):
                continue
            distance = abs(float(first["x"]) - float(second["x"])) + 0.25 * abs(
                float(first["y"]) - float(second["y"])
            )
            first_geometry, second_geometry = sorted(
                (_geometry_key(first), _geometry_key(second))
            )
            pairs.append(
                (
                    distance,
                    first_geometry,
                    second_geometry,
                    min(int(first["id"]), int(second["id"])),
                    max(int(first["id"]), int(second["id"])),
                    first,
                    second,
                )
            )
    if not pairs:
        return PointerExpertDecision.wait(1)
    *_, first, second = min(pairs, key=lambda value: value[:5])
    target = max(
        (first, second),
        key=lambda body: (
            float(body["y"]),
            _geometry_key(body),
            -int(body["id"]),
        ),
    )
    # Relative aiming keeps the launch point beside the selected body, so the
    # old fixed-row travel-distance heuristic no longer applies.
    return PointerExpertDecision.strong(
        int(target["id"]), template_index=_central_template(resolved)
    )


def direct_anchor(
    observation: Mapping[str, Any],
    spec: PointerActionSpec | None = None,
) -> PointerExpertDecision:
    """Weak-shot the lowest member of any visible same-color pair."""

    resolved = PointerActionSpec() if spec is None else spec
    by_color: dict[object, list[Mapping[str, Any]]] = {}
    for body in _pieces(observation):
        by_color.setdefault(body.get("color"), []).append(body)
    candidates = [
        body for bodies in by_color.values() if len(bodies) >= 2 for body in bodies
    ]
    if not candidates:
        return PointerExpertDecision.wait(1)
    target = max(
        candidates,
        key=lambda body: (
            float(body["y"]),
            _geometry_key(body),
            -int(body["id"]),
        ),
    )
    return PointerExpertDecision.weak(
        int(target["id"]), template_index=_central_template(resolved)
    )


def eject_anchor(
    observation: Mapping[str, Any],
    spec: PointerActionSpec | None = None,
) -> PointerExpertDecision:
    """Strong-shot an outer piece from its inner side to push it outward."""

    resolved = PointerActionSpec() if spec is None else spec
    pieces = _pieces(observation)
    if not pieces:
        return PointerExpertDecision.wait(1)
    field = observation.get("field", {})
    center_x = float(field.get("x", 0.0)) + float(field.get("width", 608.0)) / 2.0
    target = max(
        pieces,
        key=lambda body: (
            abs(float(body["x"]) - center_x),
            float(body["y"]),
            _geometry_key(body),
            -int(body["id"]),
        ),
    )
    # A projectile launched on the inner side transfers momentum outward.
    offset = 1.0 if float(target["x"]) < center_x else -1.0
    return PointerExpertDecision.strong(
        int(target["id"]),
        template_index=_template_for_offset(resolved, offset),
    )


def hazard_anchor(
    observation: Mapping[str, Any],
    spec: PointerActionSpec | None = None,
) -> PointerExpertDecision:
    """Prioritize the most visibly rot-imminent piece."""

    resolved = PointerActionSpec() if spec is None else spec
    pieces = _pieces(observation)
    if not pieces:
        return PointerExpertDecision.wait(1)
    target = max(
        pieces,
        key=lambda body: (
            body.get("lifecycle") == "rotten",
            int(body.get("rot_timer", 0)),
            -int(body.get("remaining_lifetime", 0)),
            float(body["y"]),
            _geometry_key(body),
            -int(body["id"]),
        ),
    )
    return PointerExpertDecision.strong(
        int(target["id"]), template_index=_central_template(resolved)
    )


def expert_anchors(
    observation: Mapping[str, Any],
    spec: PointerActionSpec | None = None,
) -> tuple[SearchCandidate, ...]:
    """Return all four anchors in a fixed, documented order."""

    resolved = PointerActionSpec() if spec is None else spec
    return (
        SearchCandidate("anchor/matcher", matcher_anchor(observation, resolved)),
        SearchCandidate("anchor/direct", direct_anchor(observation, resolved)),
        SearchCandidate("anchor/eject", eject_anchor(observation, resolved)),
        SearchCandidate("anchor/hazard", hazard_anchor(observation, resolved)),
    )


def generate_candidates(
    observation: Mapping[str, Any],
    spec: PointerActionSpec | None = None,
    *,
    max_target_bodies: int = 8,
) -> tuple[SearchCandidate, ...]:
    """Generate anchors, waits, and body/template shots deterministically."""

    if type(max_target_bodies) is not int or max_target_bodies < 1:
        raise ValueError("max_target_bodies must be a positive integer")
    resolved = PointerActionSpec() if spec is None else spec
    candidates = list(expert_anchors(observation, resolved))
    for ticks in resolved.wait_choices:
        candidates.append(
            SearchCandidate(
                f"wait/{int(ticks)}", PointerExpertDecision.wait(int(ticks))
            )
        )

    # Urgency only uses public, current-state fields. ID is the final tie-break.
    pieces = sorted(
        _pieces(observation),
        key=lambda body: (
            body.get("lifecycle") != "rotten",
            -int(body.get("rot_timer", 0)),
            int(body.get("remaining_lifetime", 0)),
            -float(body["y"]),
            _geometry_key(body),
            int(body["id"]),
        ),
    )[:max_target_bodies]
    for template_index in diverse_template_indices(resolved):
        for body in pieces:
            body_id = int(body["id"])
            candidates.append(
                SearchCandidate(
                    f"shot/weak/{body_id}/{template_index}",
                    PointerExpertDecision.weak(
                        body_id, template_index=template_index
                    ),
                )
            )
            candidates.append(
                SearchCandidate(
                    f"shot/strong/{body_id}/{template_index}",
                    PointerExpertDecision.strong(
                        body_id, template_index=template_index
                    ),
                )
            )
    return tuple(candidates)


def target_body(
    observation: Mapping[str, Any], target_body_id: int
) -> Mapping[str, Any] | None:
    """Resolve a public ID without relying on padded-row position."""

    return next(
        (
            body
            for body in observation.get("bodies", ())
            if int(body.get("id", -1)) == int(target_body_id)
        ),
        None,
    )


__all__ = [
    "PointerExpertDecision",
    "SearchCandidate",
    "direct_anchor",
    "diverse_template_indices",
    "eject_anchor",
    "expert_anchors",
    "generate_candidates",
    "hazard_anchor",
    "matcher_anchor",
    "target_body",
]
