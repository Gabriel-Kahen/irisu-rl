"""Goal-conditioned source-to-destination steering imitation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irisu_env import Action
from irisu_rl.actions import ActionSpec, SemanticAction, SemanticActionKind
from irisu_rl.encoding import EncodedBatch, TeacherStateEncoder
from irisu_rl.schema import TensorSchema

from .action import PointerActionSpec
from .policy import encoded_body_ids
from .replay_supervision import ReplaySteeringCollection
from .steering import SteeringDecision, SteeringIntent
from .steering_progress import DirectedPairProgressTracker


_INTENTS = tuple(SteeringIntent)
_INTENT_INDEX = {value: index for index, value in enumerate(_INTENTS)}


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("steering model has no parameters") from exc


def _array_identity(value: np.ndarray) -> dict[str, object]:
    owned = np.ascontiguousarray(value)
    return {
        "dtype": owned.dtype.str,
        "shape": list(owned.shape),
        "sha256": hashlib.sha256(owned.tobytes(order="C")).hexdigest(),
    }


def _single_encoded(encoded: EncodedBatch) -> EncodedBatch:
    encoded.validate()
    if encoded.global_features.shape[0] != 1:
        raise ValueError("one steering example requires one encoded observation")
    return encoded.copy()


def _body_index(
    encoded: EncodedBatch, observation: Mapping[str, Any], body_id: int
) -> int:
    matches = [
        index
        for index, identifier in enumerate(encoded_body_ids(encoded, observation))
        if identifier == body_id
    ]
    if len(matches) != 1:
        raise RuntimeError("steering body ID did not bind to exactly one encoded row")
    return matches[0]


def _body_mapping(
    observation: Mapping[str, Any], identifier: int
) -> Mapping[str, Any]:
    matches = [
        value
        for value in observation.get("bodies", ())
        if isinstance(value, Mapping) and int(value.get("id", -1)) == identifier
    ]
    if len(matches) != 1:
        raise RuntimeError("steering body ID is absent or duplicated")
    return matches[0]


def _template_for_action(
    observation: Mapping[str, Any],
    source_body_id: int,
    action: SemanticAction,
    spec: PointerActionSpec,
    action_spec: ActionSpec,
) -> tuple[int, float]:
    source = _body_mapping(observation, source_body_id)
    x = float(source.get("effect_x", source.get("x", 0.0)))
    y = float(source.get("effect_y", source.get("y", 0.0)))
    size = float(source.get("size", 0.0))
    width = float(source.get("width", size))
    height = float(source.get("height", size))
    if width <= 0.0 or height <= 0.0:
        raise ValueError("steering source extents must be positive")
    cursor_x = float(action.x_norm) * float(action_spec.client_width)
    cursor_y = float(action.y_norm) * float(action_spec.client_height)
    x_offset = (cursor_x - x) / (width / 2.0)
    y_offset = (cursor_y - y) / (height / 2.0)
    index = min(
        range(spec.template_count),
        key=lambda index: (
            (spec.templates[index][0] - x_offset) ** 2
            + (spec.templates[index][1] - y_offset) ** 2,
            index,
        ),
    )
    template_x, template_y = spec.templates[index]
    return index, math.hypot(template_x - x_offset, template_y - y_offset)


@dataclass(frozen=True, slots=True)
class SteeringExample:
    """One supervised steering decision, including deliberate restraint."""

    episode_identity: str
    provenance_sha256: str
    observation: EncodedBatch
    source_index: int
    destination_index: int
    kind_index: int
    template_index: int
    intent_index: int
    pointer_spec: PointerActionSpec
    act_index: int = 1
    wait_index: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_identity, str)
            or not self.episode_identity
            or "\x00" in self.episode_identity
        ):
            raise ValueError("steering episode identity must be nonempty and NUL-free")
        if (
            not isinstance(self.provenance_sha256, str)
            or len(self.provenance_sha256) != 64
            or any(value not in "0123456789abcdef" for value in self.provenance_sha256)
        ):
            raise ValueError("steering provenance must be a lowercase SHA-256")
        owned = _single_encoded(self.observation)
        values = (
            self.source_index,
            self.destination_index,
            self.kind_index,
            self.template_index,
            self.intent_index,
            self.act_index,
            self.wait_index,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("steering labels must be integers")
        if self.act_index not in (0, 1):
            raise ValueError("steering act index must encode wait or shot")
        if not 0 <= self.wait_index < len(self.pointer_spec.wait_choices):
            raise ValueError("steering wait index is out of range")
        capacity = owned.schema.capacity
        if self.act_index == 1 and (
            not 0 <= self.source_index < capacity
            or not 0 <= self.destination_index < capacity
            or self.source_index == self.destination_index
            or not bool(owned.body_mask[0, self.source_index])
            or not bool(owned.body_mask[0, self.destination_index])
        ):
            raise ValueError("steering shot source/destination rows are invalid")
        if self.kind_index not in (0, 1):
            raise ValueError("steering kind index must encode weak or strong")
        if not 0 <= self.template_index < self.pointer_spec.template_count:
            raise ValueError("steering template index is out of range")
        if not 0 <= self.intent_index < len(_INTENTS):
            raise ValueError("steering intent index is out of range")
        object.__setattr__(self, "observation", owned)

    @property
    def is_shot(self) -> bool:
        return self.act_index == 1

    @property
    def schema_sha256(self) -> str:
        return self.observation.schema.sha256

    @property
    def pointer_spec_sha256(self) -> str:
        return self.pointer_spec.sha256

    def manifest(self) -> dict[str, object]:
        return {
            "episode_identity": self.episode_identity,
            "provenance_sha256": self.provenance_sha256,
            "schema_sha256": self.schema_sha256,
            "pointer_spec_sha256": self.pointer_spec_sha256,
            "observation": {
                "global_features": _array_identity(
                    self.observation.global_features
                ),
                "body_features": _array_identity(self.observation.body_features),
                "body_mask": _array_identity(self.observation.body_mask),
            },
            "labels": {
                "source_index": self.source_index,
                "destination_index": self.destination_index,
                "kind_index": self.kind_index,
                "template_index": self.template_index,
                "intent_index": self.intent_index,
                "act_index": self.act_index,
                "wait_index": self.wait_index,
            },
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.manifest())).hexdigest()


def steering_example_from_decision(
    observation: Mapping[str, Any],
    decision: SteeringDecision,
    *,
    episode_identity: str,
    provenance_sha256: str,
    encoder: TeacherStateEncoder | None = None,
    pointer_spec: PointerActionSpec | None = None,
    action_spec: ActionSpec | None = None,
    max_template_error: float = 0.15,
    require_representable_template: bool = True,
) -> SteeringExample | None:
    """Convert an expert decision to bounded-quantization pair supervision."""

    resolved_encoder = TeacherStateEncoder() if encoder is None else encoder
    resolved_pointer = PointerActionSpec() if pointer_spec is None else pointer_spec
    resolved_action = ActionSpec() if action_spec is None else action_spec
    if (
        isinstance(max_template_error, bool)
        or not isinstance(max_template_error, (int, float))
        or not math.isfinite(float(max_template_error))
        or float(max_template_error) < 0.0
    ):
        raise ValueError("maximum steering template error must be finite and nonnegative")
    if type(require_representable_template) is not bool:
        raise TypeError("template representability switch must be boolean")
    encoded = resolved_encoder.encode([observation])
    if not decision.is_shot:
        wait_index = min(
            range(len(resolved_pointer.wait_choices)),
            key=lambda index: (
                abs(
                    resolved_pointer.wait_choices[index]
                    - int(decision.action.wait_ticks)
                ),
                index,
            ),
        )
        return SteeringExample(
            episode_identity,
            provenance_sha256,
            encoded,
            0,
            0,
            0,
            0,
            _INTENT_INDEX[SteeringIntent.WAIT],
            resolved_pointer,
            act_index=0,
            wait_index=wait_index,
        )
    if decision.source_body_id is None or decision.destination_body_id is None:
        # Boundary ejection is a strategic primitive, not a body-pair label.
        return None
    source = _body_mapping(observation, decision.source_body_id)
    destination = _body_mapping(observation, decision.destination_body_id)
    source_safe = (
        source.get("kind") in {"piece", "bonus"}
        and int(source.get("chain_id", 0)) == 0
        and str(source.get("lifecycle", ""))
        in {
            "scripted_falling",
            "dynamic_fresh",
            "falling",
            "fresh",
        }
    )
    destination_safe = (
        destination.get("kind") == "piece"
        and str(destination.get("lifecycle", ""))
        in {
            "scripted_falling",
            "dynamic_fresh",
            "falling",
            "fresh",
            "confirmed",
            "rotten",
        }
    )
    color_safe = (
        source.get("kind") == "bonus"
        or source.get("color") == destination.get("color")
    )
    if not (source_safe and destination_safe and color_safe):
        # Keep unsafe strategic candidates in branch evidence. A rotten body
        # may be a destination, but is never a legal direct-shot source.
        return None
    kind = SemanticActionKind(decision.action.kind)
    template_index, template_error = _template_for_action(
        observation,
        decision.source_body_id,
        decision.action,
        resolved_pointer,
        resolved_action,
    )
    if (
        require_representable_template
        and template_error > float(max_template_error)
    ):
        # Small velocity-lead offsets are deliberately projected onto the
        # finite deployment vocabulary. Larger geometry changes remain
        # unrepresentable and are excluded rather than silently relabeled.
        return None
    return SteeringExample(
        episode_identity,
        provenance_sha256,
        encoded,
        _body_index(encoded, observation, decision.source_body_id),
        _body_index(encoded, observation, decision.destination_body_id),
        0 if kind is SemanticActionKind.FIRE_WEAK else 1,
        template_index,
        _INTENT_INDEX[decision.intent],
        resolved_pointer,
    )


def steering_examples_from_replay(
    collection: ReplaySteeringCollection,
    *,
    encoder: TeacherStateEncoder | None = None,
    pointer_spec: PointerActionSpec | None = None,
    max_first_hit_delay_ticks: int = 64,
    max_destination_distance_pixels: float = 200.0,
    directional_offset_threshold: float = 0.25,
) -> tuple[SteeringExample, ...]:
    """Convert prompt public hits into safe, learnable pair labels."""

    resolved_encoder = TeacherStateEncoder() if encoder is None else encoder
    resolved_pointer = PointerActionSpec() if pointer_spec is None else pointer_spec
    if (
        isinstance(max_first_hit_delay_ticks, bool)
        or not isinstance(max_first_hit_delay_ticks, int)
        or max_first_hit_delay_ticks < 0
    ):
        raise ValueError("maximum replay hit delay must be nonnegative")
    for name, value in (
        ("maximum destination distance", max_destination_distance_pixels),
        ("directional offset threshold", directional_offset_threshold),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name} must be finite and nonnegative")
    if collection.identity.observation_schema_sha256 != resolved_encoder.schema.sha256:
        raise ValueError("replay collection and steering encoder identities differ")
    if collection.identity.pointer_spec_sha256 != resolved_pointer.sha256:
        raise ValueError("replay collection and steering action identities differ")
    output: list[SteeringExample] = []
    for ordinal, (shot, observation) in enumerate(
        zip(collection.shots, collection.shot_observations, strict=True)
    ):
        if (
            shot.destination_body_id is None
            or shot.target_grouped
            or shot.target_lifecycle in {"confirmed", "rotten", "deleted"}
            or shot.first_hit_delay_ticks > max_first_hit_delay_ticks
        ):
            # A direct second hit on a confirmed/grouped source can destroy it.
            # Late contacts no longer bind the shot cursor cleanly to target
            # geometry. Keep both in evidence, but never imitate either.
            continue
        source = _body_mapping(observation, shot.target_body_id)
        destination = _body_mapping(observation, shot.destination_body_id)
        if (
            source.get("kind") not in {"piece", "bonus"}
            or int(source.get("chain_id", 0)) != 0
            or str(source.get("lifecycle", ""))
            not in {
                "scripted_falling",
                "dynamic_fresh",
                "falling",
                "fresh",
            }
            or destination.get("kind") != "piece"
            or str(destination.get("lifecycle", ""))
            not in {
                "scripted_falling",
                "dynamic_fresh",
                "falling",
                "fresh",
                "confirmed",
                "rotten",
            }
            or (
                source.get("kind") != "bonus"
                and source.get("color") != destination.get("color")
            )
        ):
            continue
        source_x = float(source.get("effect_x", source.get("x", 0.0)))
        source_y = float(source.get("effect_y", source.get("y", 0.0)))
        destination_x = float(
            destination.get("effect_x", destination.get("x", 0.0))
        )
        destination_y = float(
            destination.get("effect_y", destination.get("y", 0.0))
        )
        distance = math.hypot(
            destination_x - source_x, destination_y - source_y
        )
        if distance > float(max_destination_distance_pixels):
            continue
        horizontal_goal = destination_x - source_x
        if (
            abs(shot.cursor_x_radius_offset)
            >= float(directional_offset_threshold)
            and abs(horizontal_goal) >= 1.0
            and shot.cursor_x_radius_offset * horizontal_goal > 0.0
        ):
            # A side impact on the destination-facing side would push away
            # from the inferred peer, so that peer is not a credible label.
            continue
        encoded = resolved_encoder.encode([observation])
        lifecycle = str(destination.get("lifecycle", ""))
        intent = (
            SteeringIntent.MATCH_ROTTEN
            if lifecycle == "rotten"
            else SteeringIntent.STEER_MATCH
        )
        output.append(
            SteeringExample(
                f"replay:{collection.identity.replay_sha256}:{ordinal}",
                collection.sha256,
                encoded,
                _body_index(encoded, observation, shot.target_body_id),
                _body_index(encoded, observation, shot.destination_body_id),
                0 if shot.action_kind == 1 else 1,
                shot.template_index,
                _INTENT_INDEX[intent],
                resolved_pointer,
            )
        )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class SteeringTensorBatch:
    global_features: Tensor
    body_features: Tensor
    body_mask: Tensor
    source_index: Tensor
    destination_index: Tensor
    kind_index: Tensor
    template_index: Tensor
    intent_index: Tensor
    act_index: Tensor
    wait_index: Tensor
    schema: TensorSchema
    pointer_spec: PointerActionSpec

    @property
    def size(self) -> int:
        return int(self.source_index.numel())

    def to(self, device: torch.device | str) -> SteeringTensorBatch:
        target = torch.device(device)
        return SteeringTensorBatch(
            *(value.to(target) for value in (
                self.global_features,
                self.body_features,
                self.body_mask,
                self.source_index,
                self.destination_index,
                self.kind_index,
                self.template_index,
                self.intent_index,
                self.act_index,
                self.wait_index,
            )),
            self.schema,
            self.pointer_spec,
        )


class SteeringDataset(Sequence[SteeringExample]):
    def __init__(self, examples: Sequence[SteeringExample]) -> None:
        values = tuple(examples)
        if not values:
            raise ValueError("steering dataset must not be empty")
        if any(not isinstance(value, SteeringExample) for value in values):
            raise TypeError("steering dataset contains a non-steering example")
        if len({value.schema_sha256 for value in values}) != 1:
            raise ValueError("steering dataset mixes observation schemas")
        if len({value.pointer_spec_sha256 for value in values}) != 1:
            raise ValueError("steering dataset mixes pointer action schemas")
        self._examples = values
        self.schema = values[0].observation.schema
        self.pointer_spec = values[0].pointer_spec

    def manifest(self) -> dict[str, object]:
        return {
            "format": "irisu-directed-steering-dataset-v1",
            "schema_sha256": self.schema.sha256,
            "pointer_spec_sha256": self.pointer_spec.sha256,
            "examples": [value.sha256 for value in self._examples],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.manifest())).hexdigest()

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int | slice) -> SteeringExample | tuple[SteeringExample, ...]:
        return self._examples[index]

    def as_tensors(
        self, indices: Sequence[int] | None = None
    ) -> SteeringTensorBatch:
        selected = (
            self._examples
            if indices is None
            else tuple(self._examples[index] for index in indices)
        )
        if not selected:
            raise ValueError("steering tensor batch must not be empty")
        active_width = max(
            (
                int(active[-1]) + 1 if active.size else 1
                for value in selected
                for active in [np.flatnonzero(value.observation.body_mask[0])]
            ),
            default=1,
        )
        globals_out = torch.from_numpy(
            np.concatenate(
                [value.observation.global_features for value in selected], axis=0
            )
        )
        bodies_out = torch.from_numpy(
            np.concatenate(
                [
                    value.observation.body_features[:, :active_width]
                    for value in selected
                ],
                axis=0,
            )
        )
        mask_out = torch.from_numpy(
            np.concatenate(
                [value.observation.body_mask[:, :active_width] for value in selected],
                axis=0,
            )
        )
        labels = (
            "source_index",
            "destination_index",
            "kind_index",
            "template_index",
            "intent_index",
            "act_index",
            "wait_index",
        )
        tensors = [
            torch.tensor([getattr(value, name) for value in selected], dtype=torch.long)
            for name in labels
        ]
        return SteeringTensorBatch(
            globals_out,
            bodies_out,
            mask_out,
            *tensors,
            self.schema,
            self.pointer_spec,
        )


@dataclass(frozen=True, slots=True)
class SteeringModelConfig:
    body_hidden: int = 96
    global_hidden: int = 64
    pair_hidden: int = 128
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (self.body_hidden, self.global_hidden, self.pair_hidden)
        ):
            raise ValueError("steering model widths must be positive")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise ValueError("steering dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class GoalConditionedSteeringOutput:
    act_logits: Tensor
    wait_logits: Tensor
    pair_logits: Tensor
    kind_logits: Tensor
    template_logits: Tensor
    intent_logits: Tensor
    legal_pair_mask: Tensor


class GoalConditionedSteeringModel(nn.Module):
    """Permutation-equivariant policy over directed source/destination pairs."""

    def __init__(
        self,
        schema: TensorSchema,
        *,
        pointer_spec: PointerActionSpec | None = None,
        config: SteeringModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.pointer_spec = PointerActionSpec() if pointer_spec is None else pointer_spec
        self.config = SteeringModelConfig() if config is None else config
        try:
            self._kind_piece = schema.body_features.index("kind_piece")
            self._kind_bonus = schema.body_features.index("kind_bonus")
            self._kind_projectile = schema.body_features.index("kind_projectile")
            self._color_indices = tuple(
                schema.body_features.index(f"color_{index}") for index in range(6)
            )
            self._x_index = schema.body_features.index("effect_x_norm")
            self._y_index = schema.body_features.index("effect_y_norm")
            self._rot_timer_index = schema.body_features.index("rot_timer_log1p")
            self._rotten_lifecycle_index = schema.body_features.index(
                "lifecycle_rotten"
            )
            self._source_lifecycle_indices = tuple(
                schema.body_features.index(name)
                for name in ("lifecycle_falling", "lifecycle_fresh")
            )
            self._destination_lifecycle_indices = tuple(
                schema.body_features.index(name)
                for name in (
                    "lifecycle_falling",
                    "lifecycle_fresh",
                    "lifecycle_confirmed",
                    "lifecycle_rotten",
                )
            )
        except ValueError as exc:
            raise ValueError("steering schema lacks public kind/color/geometry") from exc
        self._id_indices = tuple(
            schema.body_features.index(name)
            for name in ("id_scaled", "chain_id_scaled")
            if name in schema.body_features
        )
        self._chain_id_index = (
            schema.body_features.index("chain_id_scaled")
            if "chain_id_scaled" in schema.body_features
            else None
        )
        body_hidden = self.config.body_hidden
        pair_hidden = self.config.pair_hidden
        self.body_encoder = nn.Sequential(
            nn.Linear(len(schema.body_features), body_hidden),
            nn.LayerNorm(body_hidden),
            nn.GELU(),
            nn.Linear(body_hidden, body_hidden),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(len(schema.global_features), self.config.global_hidden),
            nn.GELU(),
        )
        pair_input = 2 * body_hidden + self.config.global_hidden + 9
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_input, pair_hidden),
            nn.LayerNorm(pair_hidden),
            nn.GELU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(pair_hidden, pair_hidden),
            nn.GELU(),
        )
        self.pair_head = nn.Linear(pair_hidden, 1)
        self.kind_head = nn.Linear(pair_hidden, 2)
        self.template_head = nn.Linear(pair_hidden, self.pointer_spec.template_count)
        self.intent_head = nn.Linear(pair_hidden, len(_INTENTS))
        decision_width = body_hidden + self.config.global_hidden + pair_hidden
        self.act_head = nn.Linear(decision_width, 2)
        self.wait_head = nn.Linear(
            decision_width, len(self.pointer_spec.wait_choices)
        )

    def manifest(self) -> dict[str, object]:
        return {
            "architecture": "goal-conditioned-directed-pair-steering-v4",
            "schema": self.schema.version,
            "schema_sha256": self.schema.sha256,
            "pointer_action_sha256": self.pointer_spec.sha256,
            "input_ablations": [
                self.schema.body_features[index] for index in self._id_indices
            ],
            "public_pair_relations": [
                "destination_x_minus_source_x",
                "destination_y_minus_source_y",
                "center_distance",
                "same_color",
                "destination_below_source",
                "destination_grouped",
                "destination_group_size_log_scaled",
                "destination_rotten",
                "source_rot_active",
            ],
            "config": asdict(self.config),
            "intent_order": [value.value for value in _INTENTS],
        }

    @property
    def architecture_sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.manifest())).hexdigest()

    def _ablate_ids(self, body_features: Tensor) -> Tensor:
        if not self._id_indices:
            return body_features
        output = body_features.clone()
        output[..., self._id_indices] = 0.0
        return output

    def legal_pair_mask(self, body_features: Tensor, body_mask: Tensor) -> Tensor:
        piece = body_features[..., self._kind_piece] > body_features[
            ..., self._kind_projectile
        ]
        bonus = body_features[..., self._kind_bonus] > body_features[
            ..., self._kind_projectile
        ]
        colors = body_features[..., self._color_indices]
        same_color = torch.einsum("bic,bjc->bij", colors, colors) > 0.5
        source_lifecycle = body_features[
            ..., self._source_lifecycle_indices
        ].sum(dim=-1) > 0.5
        destination_lifecycle = body_features[
            ..., self._destination_lifecycle_indices
        ].sum(dim=-1) > 0.5
        ungrouped = (
            torch.ones_like(piece)
            if self._chain_id_index is None
            else body_features[..., self._chain_id_index] <= 0.0
        )
        source_allowed = (piece | bonus) & source_lifecycle & ungrouped
        destination_allowed = piece & destination_lifecycle
        count = body_features.shape[1]
        distinct = ~torch.eye(count, dtype=torch.bool, device=body_features.device)
        compatible = same_color | bonus.unsqueeze(-1)
        return (
            body_mask.unsqueeze(-1)
            & body_mask.unsqueeze(-2)
            & source_allowed.unsqueeze(-1)
            & destination_allowed.unsqueeze(-2)
            & compatible
            & distinct
        )

    def _public_pair_relations(
        self, body_features: Tensor, body_mask: Tensor
    ) -> Tensor:
        """Permutation-safe strategic geometry derived from public tensors."""

        dtype = body_features.dtype
        count = body_features.shape[1]
        x = body_features[..., self._x_index]
        y = body_features[..., self._y_index]
        dx = x.unsqueeze(1) - x.unsqueeze(2)
        dy = y.unsqueeze(1) - y.unsqueeze(2)
        distance = torch.sqrt(dx.square() + dy.square() + 1e-12)
        colors = body_features[..., self._color_indices]
        same_color = torch.einsum("bic,bjc->bij", colors, colors)
        destination_below = (y.unsqueeze(1) > y.unsqueeze(2)).to(dtype)
        piece = body_features[..., self._kind_piece] > body_features[
            ..., self._kind_projectile
        ]
        if self._chain_id_index is None:
            grouped = torch.zeros_like(x, dtype=torch.bool)
            group_size = torch.zeros_like(x)
        else:
            chain = body_features[..., self._chain_id_index]
            grouped = chain > 0.0
            member = body_mask & piece & grouped
            same_chain = (
                (chain.unsqueeze(-1) == chain.unsqueeze(-2))
                & member.unsqueeze(-1)
                & member.unsqueeze(-2)
            )
            group_size = torch.log1p(
                same_chain.sum(dim=-1).to(dtype)
            ) / math.log1p(self.schema.capacity)
        destination_grouped = grouped.unsqueeze(1).expand(-1, count, -1)
        destination_group_size = group_size.unsqueeze(1).expand(-1, count, -1)
        destination_rotten = (
            body_features[..., self._rotten_lifecycle_index] > 0.5
        ).unsqueeze(1).expand(-1, count, -1)
        source_rot_active = (
            body_features[..., self._rot_timer_index] > 0.0
        ).unsqueeze(-1).expand(-1, -1, count)
        return torch.stack(
            (
                dx,
                dy,
                distance,
                same_color,
                destination_below,
                destination_grouped.to(dtype),
                destination_group_size,
                destination_rotten.to(dtype),
                source_rot_active.to(dtype),
            ),
            dim=-1,
        )

    def forward(
        self,
        global_features: Tensor,
        body_features: Tensor,
        body_mask: Tensor,
    ) -> GoalConditionedSteeringOutput:
        if (
            global_features.ndim != 2
            or body_features.ndim != 3
            or body_mask.shape != body_features.shape[:2]
            or body_mask.dtype != torch.bool
            or global_features.shape[0] != body_features.shape[0]
            or global_features.shape[1] != len(self.schema.global_features)
            or body_features.shape[2] != len(self.schema.body_features)
        ):
            raise ValueError("steering model inputs differ from the schema")
        clean = self._ablate_ids(body_features)
        body = self.body_encoder(clean)
        global_context = self.global_encoder(global_features)
        mask_float = body_mask.unsqueeze(-1).to(body.dtype)
        pooled_body = (body * mask_float).sum(dim=1) / mask_float.sum(
            dim=1
        ).clamp_min(1.0)
        count = body.shape[1]
        source = body.unsqueeze(2).expand(-1, -1, count, -1)
        destination = body.unsqueeze(1).expand(-1, count, -1, -1)
        global_pair = global_context[:, None, None, :].expand(-1, count, count, -1)
        relation = self._public_pair_relations(body_features, body_mask)
        pair = self.pair_encoder(
            torch.cat((source, destination, global_pair, relation), dim=-1)
        )
        legal = self.legal_pair_mask(body_features, body_mask)
        floor = torch.finfo(pair.dtype).min
        pair_summary = pair.masked_fill(~legal.unsqueeze(-1), floor).flatten(
            1, 2
        ).max(dim=1).values
        has_legal_pair = legal.flatten(1).any(dim=-1)
        pair_summary = torch.where(
            has_legal_pair.unsqueeze(-1),
            pair_summary,
            torch.zeros_like(pair_summary),
        )
        decision_context = torch.cat(
            (global_context, pooled_body, pair_summary), dim=-1
        )
        pair_logits = self.pair_head(pair).squeeze(-1).masked_fill(~legal, floor)
        return GoalConditionedSteeringOutput(
            self.act_head(decision_context),
            self.wait_head(decision_context),
            pair_logits,
            self.kind_head(pair),
            self.template_head(pair),
            self.intent_head(pair),
            legal,
        )


@dataclass(frozen=True, slots=True)
class SteeringLoss:
    total: Tensor
    act: Tensor
    wait: Tensor
    pair: Tensor
    kind: Tensor
    template: Tensor
    intent: Tensor


def steering_imitation_loss(
    output: GoalConditionedSteeringOutput, batch: SteeringTensorBatch
) -> SteeringLoss:
    act = F.cross_entropy(output.act_logits, batch.act_index)
    zero = output.act_logits.sum() * 0.0
    wait_rows = batch.act_index == 0
    wait = (
        F.cross_entropy(output.wait_logits[wait_rows], batch.wait_index[wait_rows])
        if bool(wait_rows.any())
        else zero
    )
    shot_rows = (batch.act_index == 1).nonzero(as_tuple=False).reshape(-1)
    if shot_rows.numel():
        source = batch.source_index[shot_rows]
        destination = batch.destination_index[shot_rows]
        if not bool(
            output.legal_pair_mask[shot_rows, source, destination].all()
        ):
            raise ValueError("steering label selects a pair masked by public semantics")
        pair_target = source * output.pair_logits.shape[2] + destination
        pair = F.cross_entropy(
            output.pair_logits[shot_rows].flatten(1), pair_target
        )
        selected = (shot_rows, source, destination)
        kind = F.cross_entropy(
            output.kind_logits[selected], batch.kind_index[shot_rows]
        )
        template = F.cross_entropy(
            output.template_logits[selected], batch.template_index[shot_rows]
        )
        intent = F.cross_entropy(
            output.intent_logits[selected], batch.intent_index[shot_rows]
        )
    else:
        pair = kind = template = intent = zero
    # Deployment learns when and which directed pair to choose; impact
    # geometry is lowered analytically from that pair. Keep the remaining
    # heads as light audit auxiliaries instead of letting them dominate.
    total = (
        act
        + 0.25 * wait
        + pair
        + 0.1 * kind
        + 0.1 * template
        + 0.25 * intent
    )
    return SteeringLoss(total, act, wait, pair, kind, template, intent)


@dataclass(frozen=True, slots=True)
class SteeringTrainingReport:
    steps: int
    shot_examples: int
    wait_examples: int
    initial_loss: float
    final_loss: float
    act_accuracy: float
    shot_recall: float
    restraint_recall: float
    act_balanced_accuracy: float
    wait_accuracy: float
    pair_accuracy: float
    kind_accuracy: float
    template_accuracy: float
    intent_accuracy: float


def train_goal_conditioned_steering(
    model: GoalConditionedSteeringModel,
    dataset: SteeringDataset,
    *,
    steps: int = 200,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    seed: int = 0,
) -> SteeringTrainingReport:
    """Deterministic supervised fitting for expert/replay pair labels."""

    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 1
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0.0
    ):
        raise ValueError("steering training parameters are invalid")
    if model.schema.sha256 != dataset.schema.sha256:
        raise ValueError("steering model and dataset schema identities differ")
    if model.pointer_spec.sha256 != dataset.pointer_spec.sha256:
        raise ValueError("steering model and dataset action identities differ")
    device = _model_device(model)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    shot_pool = torch.tensor(
        [index for index, value in enumerate(dataset) if value.is_shot],
        dtype=torch.long,
    )
    wait_pool = torch.tensor(
        [index for index, value in enumerate(dataset) if not value.is_shot],
        dtype=torch.long,
    )
    full = dataset.as_tensors().to(device)
    model.eval()
    with torch.no_grad():
        initial_output = model(
            full.global_features, full.body_features, full.body_mask
        )
        initial_loss = float(
            steering_imitation_loss(initial_output, full).total
        )
    del initial_output
    model.train()
    for step in range(steps):
        count = min(batch_size, len(dataset))
        if shot_pool.numel() and wait_pool.numel() and count >= 2:
            shot_count = count // 2
            wait_count = count - shot_count
            selected = torch.cat(
                (
                    shot_pool[
                        torch.randint(
                            shot_pool.numel(),
                            (shot_count,),
                            generator=generator,
                        )
                    ],
                    wait_pool[
                        torch.randint(
                            wait_pool.numel(),
                            (wait_count,),
                            generator=generator,
                        )
                    ],
                )
            )
            indices = selected[
                torch.randperm(count, generator=generator)
            ].tolist()
        elif shot_pool.numel() and wait_pool.numel():
            pool = shot_pool if step % 2 == 0 else wait_pool
            indices = pool[
                torch.randint(pool.numel(), (1,), generator=generator)
            ].tolist()
        else:
            pool = shot_pool if shot_pool.numel() else wait_pool
            indices = pool[
                torch.randint(pool.numel(), (count,), generator=generator)
            ].tolist()
        batch = dataset.as_tensors(indices).to(device)
        output = model(batch.global_features, batch.body_features, batch.body_mask)
        loss = steering_imitation_loss(output, batch)
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        output = model(full.global_features, full.body_features, full.body_mask)
        final = steering_imitation_loss(output, full)
        shot_rows = (full.act_index == 1).nonzero(as_tuple=False).reshape(-1)
        wait_rows = full.act_index == 0
        predicted_act = output.act_logits.argmax(-1)
        flat_pair = output.pair_logits[shot_rows].flatten(1).argmax(dim=-1)
        predicted_source = flat_pair // output.pair_logits.shape[2]
        predicted_destination = flat_pair % output.pair_logits.shape[2]
        selected = (
            shot_rows,
            full.source_index[shot_rows],
            full.destination_index[shot_rows],
        )
        pair_accuracy = (
            (
                (predicted_source == full.source_index[shot_rows])
                & (predicted_destination == full.destination_index[shot_rows])
            )
            .float()
            .mean()
            if shot_rows.numel()
            else torch.tensor(0.0, device=device)
        )
        wait_accuracy = (
            (
                output.wait_logits[wait_rows].argmax(-1)
                == full.wait_index[wait_rows]
            )
            .float()
            .mean()
            if bool(wait_rows.any())
            else torch.tensor(0.0, device=device)
        )
        if shot_rows.numel():
            kind_accuracy = (
                output.kind_logits[selected].argmax(-1)
                == full.kind_index[shot_rows]
            ).float().mean()
            template_accuracy = (
                output.template_logits[selected].argmax(-1)
                == full.template_index[shot_rows]
            ).float().mean()
            intent_accuracy = (
                output.intent_logits[selected].argmax(-1)
                == full.intent_index[shot_rows]
            ).float().mean()
        else:
            kind_accuracy = template_accuracy = intent_accuracy = torch.tensor(
                0.0, device=device
            )
        accuracies = (
            (predicted_act == full.act_index).float().mean(),
            (
                (predicted_act[shot_rows] == 1).float().mean()
                if shot_rows.numel()
                else torch.tensor(0.0, device=device)
            ),
            (
                (predicted_act[wait_rows] == 0).float().mean()
                if bool(wait_rows.any())
                else torch.tensor(0.0, device=device)
            ),
            wait_accuracy,
            pair_accuracy,
            kind_accuracy,
            template_accuracy,
            intent_accuracy,
        )
    act_accuracy, shot_recall, restraint_recall, *conditional = (
        float(value) for value in accuracies
    )
    present_recalls = [
        value
        for value, present in (
            (shot_recall, bool(shot_pool.numel())),
            (restraint_recall, bool(wait_pool.numel())),
        )
        if present
    ]
    return SteeringTrainingReport(
        steps=steps,
        shot_examples=int(shot_pool.numel()),
        wait_examples=int(wait_pool.numel()),
        initial_loss=initial_loss,
        final_loss=float(final.total),
        act_accuracy=act_accuracy,
        shot_recall=shot_recall,
        restraint_recall=restraint_recall,
        act_balanced_accuracy=sum(present_recalls) / len(present_recalls),
        wait_accuracy=conditional[0],
        pair_accuracy=conditional[1],
        kind_accuracy=conditional[2],
        template_accuracy=conditional[3],
        intent_accuracy=conditional[4],
    )


class GoalConditionedSteeringPolicy:
    """Greedy learned pair policy with conservative closed-loop cooldown."""

    def __init__(
        self,
        model: GoalConditionedSteeringModel,
        *,
        encoder: TeacherStateEncoder | None = None,
        action_spec: ActionSpec | None = None,
        pointer_spec: PointerActionSpec | None = None,
        cooldown_ticks: int = 16,
        minimum_pair_closure_sizes: float = 0.05,
        impact_side_sizes: float = 0.5,
        impact_below_sizes: float = 0.75,
        source_velocity_lead_ticks: float = 1.0,
        ticks_per_second: float = 50.0,
        act_logit_bias: float = 0.0,
        artifact_sha256: str | None = None,
    ) -> None:
        self.model = model
        self.encoder = TeacherStateEncoder() if encoder is None else encoder
        self.action_spec = ActionSpec() if action_spec is None else action_spec
        self.pointer_spec = model.pointer_spec if pointer_spec is None else pointer_spec
        if self.encoder.schema.sha256 != model.schema.sha256:
            raise ValueError("steering model and inference schema identities differ")
        if self.pointer_spec.sha256 != model.pointer_spec.sha256:
            raise ValueError("steering model and inference action identities differ")
        if (
            isinstance(cooldown_ticks, bool)
            or not isinstance(cooldown_ticks, int)
            or cooldown_ticks < 1
        ):
            raise ValueError("steering cooldown must be positive")
        if artifact_sha256 is not None and (
            len(artifact_sha256) != 64
            or any(value not in "0123456789abcdef" for value in artifact_sha256)
        ):
            raise ValueError("steering artifact identity must be a SHA-256")
        self.cooldown_ticks = cooldown_ticks
        self.minimum_pair_closure_sizes = float(minimum_pair_closure_sizes)
        self._progress = DirectedPairProgressTracker(
            minimum_closure_sizes=minimum_pair_closure_sizes
        )
        for name, value in (
            ("impact side", impact_side_sizes),
            ("impact below", impact_below_sizes),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise ValueError(f"steering {name} scale must be finite and positive")
        self.impact_side_sizes = float(impact_side_sizes)
        self.impact_below_sizes = float(impact_below_sizes)
        if (
            isinstance(source_velocity_lead_ticks, bool)
            or not isinstance(source_velocity_lead_ticks, (int, float))
            or not math.isfinite(float(source_velocity_lead_ticks))
            or float(source_velocity_lead_ticks) < 0.0
            or isinstance(ticks_per_second, bool)
            or not isinstance(ticks_per_second, (int, float))
            or not math.isfinite(float(ticks_per_second))
            or float(ticks_per_second) <= 0.0
        ):
            raise ValueError("steering velocity lead configuration is invalid")
        self.source_velocity_lead_ticks = float(source_velocity_lead_ticks)
        self.ticks_per_second = float(ticks_per_second)
        if (
            isinstance(act_logit_bias, bool)
            or not isinstance(act_logit_bias, (int, float))
            or not math.isfinite(float(act_logit_bias))
        ):
            raise ValueError("steering act-logit bias must be finite")
        self.act_logit_bias = float(act_logit_bias)
        self.artifact_sha256 = artifact_sha256
        self.schema_sha256 = self.encoder.schema.sha256
        self.pointer_action_sha256 = self.pointer_spec.sha256
        self._cooldown_until = 0
        self._last_tick: int | None = None
        self._last_decision: SteeringDecision | None = None

    def reset(self, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("steering seed must fit uint32")
        self._cooldown_until = 0
        self._last_tick = None
        self._last_decision = None
        self._progress.reset()

    def _wait(self, ticks: int, reason: str) -> SteeringDecision:
        legal = [value for value in self.action_spec.wait_choices if value <= ticks]
        chosen = max(legal) if legal else min(self.action_spec.wait_choices)
        return SteeringDecision(
            self.action_spec.validate(SemanticAction.wait(chosen)),
            SteeringIntent.WAIT,
            reason=reason,
        )

    def _analytic_action(
        self,
        source: Mapping[str, Any],
        destination: Mapping[str, Any],
    ) -> tuple[SemanticAction, float, float] | None:
        source_x = float(source.get("effect_x", source.get("x", 0.0)))
        source_y = float(source.get("effect_y", source.get("y", 0.0)))
        destination_x = float(
            destination.get("effect_x", destination.get("x", 0.0))
        )
        direction = 1.0 if destination_x > source_x else -1.0
        if math.isclose(destination_x, source_x, abs_tol=1e-9):
            direction = (
                1.0
                if source_x <= self.action_spec.client_width / 2.0
                else -1.0
            )
        lifecycle = str(source.get("lifecycle", ""))
        velocity = source.get("vx_display_per_second")
        if velocity is None:
            velocity = float(source.get("vx", 0.0)) * (
                50.0
                if lifecycle in {"scripted_falling", "falling"}
                else 10.0
            )
        lead_x = (
            float(velocity)
            * self.source_velocity_lead_ticks
            / self.ticks_per_second
        )
        size = float(source.get("size", 0.0))
        if not math.isfinite(size) or size <= 0.0:
            raise ValueError("learned steering source size must be positive")
        impact_x = -direction * self.impact_side_sizes
        raw_x = source_x + lead_x + impact_x * size
        raw_y = source_y + self.impact_below_sizes * size
        if not (
            0.0 <= raw_x <= self.action_spec.client_width
            and 0.0 <= raw_y <= self.action_spec.client_height
        ):
            return None
        return (
            self.action_spec.validate(
                SemanticAction.strong(
                    raw_x / self.action_spec.client_width,
                    raw_y / self.action_spec.client_height,
                )
            ),
            impact_x,
            self.impact_below_sizes,
        )

    @torch.no_grad()
    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        tick = int(observation.get("tick", 0))
        if self._last_tick is not None:
            if tick < self._last_tick:
                raise RuntimeError("steering observation tick moved backwards")
            if tick == self._last_tick:
                assert self._last_decision is not None
                return self._last_decision
        if tick < self._cooldown_until:
            decision = self._wait(
                self._cooldown_until - tick,
                "wait for the previous learned correction to resolve",
            )
        else:
            self._progress.prune(observation)
            self._progress.assess(observation)
            encoded = self.encoder.encode([observation])
            active = np.flatnonzero(encoded.body_mask[0])
            width = int(active[-1]) + 1 if active.size else 1
            device = _model_device(self.model)
            global_features = torch.from_numpy(encoded.global_features).to(device)
            body_features = torch.from_numpy(
                encoded.body_features[:, :width]
            ).to(device)
            body_mask = torch.from_numpy(encoded.body_mask[:, :width]).to(device)
            output = self.model(global_features, body_features, body_mask)
            act_logits = output.act_logits[0].clone()
            act_logits[1] += self.act_logit_bias
            act_index = int(act_logits.argmax())
            if act_index == 0:
                wait_index = int(output.wait_logits[0].argmax())
                decision = self._wait(
                    self.pointer_spec.wait_choices[wait_index],
                    "learned restraint from expert cadence supervision",
                )
            elif not bool(output.legal_pair_mask.any()):
                decision = self._wait(8, "no legal learned source/destination pair")
            else:
                identifiers = encoded_body_ids(encoded, observation)
                legal_flat = output.legal_pair_mask[0].flatten().nonzero(
                    as_tuple=False
                ).reshape(-1)
                ranked = legal_flat[
                    output.pair_logits[0].flatten()[legal_flat].argsort(
                        descending=True
                    )
                ]
                selected_pair: tuple[
                    int,
                    int,
                    int,
                    int,
                    SemanticAction,
                    float,
                    float,
                    Mapping[str, Any],
                    Mapping[str, Any],
                ] | None = None
                for candidate in ranked.tolist():
                    source_index, destination_index = divmod(
                        int(candidate), width
                    )
                    source_id = identifiers[source_index]
                    destination_id = identifiers[destination_index]
                    if source_id is None or destination_id is None:
                        continue
                    source = _body_mapping(observation, source_id)
                    destination = _body_mapping(observation, destination_id)
                    source_lifecycle = str(source.get("lifecycle", ""))
                    source_safe = (
                        source.get("kind") in {"piece", "bonus"}
                        and int(source.get("chain_id", 0)) == 0
                        and (
                            source.get("kind") == "bonus"
                            or source_lifecycle
                            in {
                                "scripted_falling",
                                "dynamic_fresh",
                                "falling",
                                "fresh",
                            }
                        )
                    )
                    destination_safe = (
                        destination.get("kind") == "piece"
                        and str(destination.get("lifecycle", ""))
                        != "deleted"
                    )
                    same_color = (
                        source.get("kind") == "bonus"
                        or source.get("color") == destination.get("color")
                    )
                    analytic = self._analytic_action(source, destination)
                    stalled = self._progress.is_stalled(
                        observation, source_id, destination_id
                    )
                    if (
                        source_safe
                        and destination_safe
                        and same_color
                        and analytic is not None
                        and not stalled
                    ):
                        semantic, impact_x, impact_y = analytic
                        selected_pair = (
                            source_index,
                            destination_index,
                            source_id,
                            destination_id,
                            semantic,
                            impact_x,
                            impact_y,
                            source,
                            destination,
                        )
                        break
                if selected_pair is None:
                    decision = self._wait(
                        8, "all learned pairs failed public safety checks"
                    )
                else:
                    (
                        source_index,
                        destination_index,
                        source_id,
                        destination_id,
                        semantic,
                        impact_x,
                        impact_y,
                        source,
                        destination,
                    ) = selected_pair
                    intent_index = int(
                        output.intent_logits[
                            0, source_index, destination_index
                        ].argmax()
                    )
                    decision = SteeringDecision(
                        semantic,
                        _INTENTS[intent_index],
                        source_body_id=source_id,
                        destination_body_id=destination_id,
                        destination_chain_id=int(destination.get("chain_id", 0)),
                        impact_x_sizes=impact_x,
                        impact_y_sizes=impact_y,
                        reason="learned explicit source-to-destination correction",
                    )
                    self._cooldown_until = tick + self.cooldown_ticks
                    self._progress.begin(
                        observation, source_id, destination_id
                    )
        self._last_tick = tick
        self._last_decision = decision
        return decision

    def act(self, observation: Mapping[str, Any]) -> tuple[Action, ...]:
        return self.predict(observation).primitive_actions(self.action_spec)


__all__ = [
    "GoalConditionedSteeringModel",
    "GoalConditionedSteeringOutput",
    "GoalConditionedSteeringPolicy",
    "SteeringDataset",
    "SteeringExample",
    "SteeringLoss",
    "SteeringModelConfig",
    "SteeringTensorBatch",
    "SteeringTrainingReport",
    "steering_example_from_decision",
    "steering_examples_from_replay",
    "steering_imitation_loss",
    "train_goal_conditioned_steering",
]
