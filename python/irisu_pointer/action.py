"""Discrete entity-pointer actions and deterministic semantic lowering."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch
from torch import Tensor

from irisu_rl.actions import (
    ActionSpec,
    SemanticAction,
    SemanticActionKind,
)
from irisu_rl.schema import TensorSchema


@dataclass(frozen=True, slots=True)
class PointerActionSpec:
    """Target-relative action vocabulary used by the pointer policy."""

    wait_choices: tuple[int, ...] = (1, 2, 4, 8, 16)
    x_radius_offsets: tuple[float, ...] = (
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        # Append-only extension: the first 15 template indices retain the
        # original 5x3 vocabulary used by R3c artifacts and expert defaults.
        -1.5,
        1.5,
    )
    y_radius_offsets: tuple[float, ...] = (1.5, 2.0, 2.5)

    def __post_init__(self) -> None:
        if (
            not self.wait_choices
            or tuple(sorted(set(self.wait_choices))) != self.wait_choices
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in self.wait_choices
            )
        ):
            raise ValueError("pointer wait choices must be positive and increasing")
        if not self.x_radius_offsets or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in self.x_radius_offsets
        ):
            raise ValueError("pointer x-radius offsets must be finite")
        if not self.y_radius_offsets or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in self.y_radius_offsets
        ):
            raise ValueError("pointer y-radius offsets must be finite")

    @property
    def templates(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (float(x_offset), float(y_offset))
            for x_offset in self.x_radius_offsets
            for y_offset in self.y_radius_offsets
        )

    @property
    def template_count(self) -> int:
        return len(self.x_radius_offsets) * len(self.y_radius_offsets)

    def manifest(self) -> dict[str, object]:
        return {
            "wait_choices": list(self.wait_choices),
            "x_radius_offsets": list(self.x_radius_offsets),
            "y_radius_offsets": list(self.y_radius_offsets),
            "coordinate_frame": "selected_effect_center_and_half_extents-v1",
            "template_order": "x_radius_offset_major_then_y_radius_offset",
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PointerActionTensor:
    """Tensor form of a pointer action with one target per shot decision."""

    kind: Tensor
    wait_index: Tensor
    target_index: Tensor
    template_index: Tensor

    def validate(
        self,
        leading_shape: torch.Size | tuple[int, ...],
        body_count: int,
        spec: PointerActionSpec | None = None,
    ) -> None:
        resolved = spec or PointerActionSpec()
        expected = torch.Size(leading_shape)
        if isinstance(body_count, bool) or not isinstance(body_count, int):
            raise TypeError("body count must be an integer")
        if body_count < 0:
            raise ValueError("body count must be nonnegative")
        fields = (
            ("kind", self.kind),
            ("wait_index", self.wait_index),
            ("target_index", self.target_index),
            ("template_index", self.template_index),
        )
        for name, value in fields:
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a torch tensor")
            if value.shape != expected:
                raise ValueError(f"{name} shape must be {tuple(expected)}")
            if value.dtype != torch.long:
                raise ValueError(f"{name} must be int64")
        if self.kind.numel() == 0:
            return
        if bool(((self.kind < 0) | (self.kind > 2)).any()):
            raise ValueError("pointer action kind is outside [0, 2]")
        if bool(
            ((self.wait_index < 0) | (self.wait_index >= len(resolved.wait_choices))).any()
        ):
            raise ValueError("pointer wait index is out of range")
        if bool(
            (
                (self.template_index < 0)
                | (self.template_index >= resolved.template_count)
            ).any()
        ):
            raise ValueError("pointer template index is out of range")
        shot = self.kind != int(SemanticActionKind.WAIT)
        if bool(shot.any()):
            if body_count == 0:
                raise ValueError("a shot action requires at least one body")
            shot_targets = self.target_index[shot]
            if bool(((shot_targets < 0) | (shot_targets >= body_count)).any()):
                raise ValueError("active pointer target is out of range")


def _body_value(row: Tensor | Sequence[float], index: int, name: str) -> float:
    try:
        raw = row[index]
    except (IndexError, TypeError) as exc:
        raise ValueError(f"selected body row does not contain {name}") from exc
    if isinstance(raw, Tensor):
        if raw.numel() != 1:
            raise ValueError(f"selected body field {name} must be scalar")
        raw = raw.detach().item()
    if isinstance(raw, bool) or not isinstance(raw, Real):
        raise TypeError(f"selected body field {name} must be numeric")
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"selected body field {name} must be finite")
    return value


def decode_pointer_action(
    *,
    kind: int,
    wait_index: int = 0,
    template_index: int = 0,
    selected_body_row: Tensor | Sequence[float] | None,
    schema: TensorSchema,
    pointer_spec: PointerActionSpec | None = None,
    action_spec: ActionSpec | None = None,
) -> SemanticAction:
    """Lower one pointer decision without consulting mutable game state."""

    resolved_pointer = pointer_spec or PointerActionSpec()
    resolved_action = action_spec or ActionSpec()
    try:
        parsed_kind = SemanticActionKind(kind)
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown pointer action kind") from exc
    if parsed_kind is SemanticActionKind.WAIT:
        if (
            isinstance(wait_index, bool)
            or not isinstance(wait_index, int)
            or not 0 <= wait_index < len(resolved_pointer.wait_choices)
        ):
            raise ValueError("pointer wait index is out of range")
        return resolved_action.validate(
            SemanticAction.wait(resolved_pointer.wait_choices[wait_index])
        )
    if selected_body_row is None:
        raise ValueError("shot pointer action requires a selected body row")
    if (
        isinstance(template_index, bool)
        or not isinstance(template_index, int)
        or not 0 <= template_index < resolved_pointer.template_count
    ):
        raise ValueError("pointer template index is out of range")
    try:
        x_index = schema.body_features.index("effect_x_norm")
        y_index = schema.body_features.index("effect_y_norm")
        width_index = schema.body_features.index("width_norm")
        height_index = schema.body_features.index("height_norm")
    except ValueError as exc:
        raise ValueError("pointer schema lacks required target geometry") from exc
    expected_width = len(schema.body_features)
    try:
        actual_width = int(selected_body_row.shape[-1])  # type: ignore[union-attr]
    except AttributeError:
        actual_width = len(selected_body_row)
    if actual_width != expected_width:
        raise ValueError("selected body row width differs from the schema")
    effect_x = _body_value(selected_body_row, x_index, "effect_x_norm")
    effect_y = _body_value(selected_body_row, y_index, "effect_y_norm")
    width = _body_value(selected_body_row, width_index, "width_norm")
    height = _body_value(selected_body_row, height_index, "height_norm")
    if width < 0.0 or height < 0.0:
        raise ValueError("selected body extents must be nonnegative")
    x_radius_offset, y_radius_offset = resolved_pointer.templates[template_index]
    x_norm = min(
        1.0, max(0.0, effect_x + x_radius_offset * width / 2.0)
    )
    y_norm = min(
        1.0, max(0.0, effect_y + y_radius_offset * height / 2.0)
    )
    constructor = (
        SemanticAction.weak
        if parsed_kind is SemanticActionKind.FIRE_WEAK
        else SemanticAction.strong
    )
    return resolved_action.validate(constructor(x_norm, y_norm))
