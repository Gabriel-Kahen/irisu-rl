"""Causal recurrent residual learning over the frozen v5 steering policy.

The recurrent path never consumes numeric entity identifiers.  Event features
describe only the previous completed transition; delayed outcomes remain
training targets rather than same-timestep policy inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irisu_rl.ppo import quantile_huber_loss

from .steering import SteeringIntent
from .steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringOutput,
    SteeringExample,
)


_CHECKPOINT_FORMAT = "irisu-sequence-replay-checkpoint-v1"


SEQUENCE_REPLAY_EVENT_FEATURES = (
    "previous_action_wait",
    "previous_action_shot",
    *(f"previous_intent_{intent.value}" for intent in SteeringIntent),
    "elapsed_ticks_log_scaled",
    "time_since_shot_log_scaled",
    "delta_score_scaled",
    "delta_gauge",
    "delta_clears_scaled",
    "delta_highest_chain_scaled",
    "shot_fired_count_log_scaled",
    "projectile_hit_count_log_scaled",
    "intended_source_hit_count_log_scaled",
    "exact_pair_join_count_log_scaled",
    "chain_confirmed_count_log_scaled",
    "chain_cleared_count_log_scaled",
    "rotten_count_log_scaled",
    "ejected_count_log_scaled",
    "invalid_count_log_scaled",
    "game_over",
    "source_present",
    "destination_present",
    "pair_joined",
    "closure_observed",
    "progress_failed",
    "spawn_boundary",
)
SEQUENCE_REPLAY_EVENT_WIDTH = len(SEQUENCE_REPLAY_EVENT_FEATURES)

VIABILITY_TARGETS = ("alive_at_horizon", "gauge_failure")
OUTCOME_TARGETS = ("final_gauge_delta", "qualifying_clear_delta")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(_canonical_sha256(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _optional_sha256(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def sequence_replay_event_manifest() -> dict[str, object]:
    return {
        "format": "irisu-sequence-replay-event-v1",
        "causal_boundary": "previous-completed-transition-only",
        "features": list(SEQUENCE_REPLAY_EVENT_FEATURES),
    }


@dataclass(frozen=True, slots=True)
class SequenceReplayConfig:
    body_hidden: int = 64
    global_hidden: int = 48
    event_hidden: int = 32
    recurrent_hidden: int = 96
    pair_hidden: int = 96
    value_quantiles: int = 51
    residual_scale: float = 4.0
    geometry_residual_scale: float = 2.0
    geometry_gate_threshold: float = 0.90
    dropout: float = 0.0

    def __post_init__(self) -> None:
        widths = (
            self.body_hidden,
            self.global_hidden,
            self.event_hidden,
            self.recurrent_hidden,
            self.pair_hidden,
        )
        if any(type(value) is not int or value < 1 for value in widths):
            raise ValueError("sequence replay widths must be positive integers")
        if (
            type(self.value_quantiles) is not int
            or not 3 <= self.value_quantiles <= 101
            or self.value_quantiles % 2 != 1
        ):
            raise ValueError("value quantiles must be odd and within [3, 101]")
        positive = (self.residual_scale, self.geometry_residual_scale)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in positive
        ):
            raise ValueError("residual scales must be finite and positive")
        if (
            isinstance(self.geometry_gate_threshold, bool)
            or not isinstance(self.geometry_gate_threshold, (int, float))
            or not 0.5 < float(self.geometry_gate_threshold) < 1.0
            or isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise ValueError("sequence replay probability settings are invalid")


@dataclass(frozen=True, slots=True)
class SequenceReplayOutput:
    act_logits: Tensor
    wait_logits: Tensor
    pair_logits: Tensor
    kind_logits: Tensor
    template_logits: Tensor
    intent_logits: Tensor
    legal_pair_mask: Tensor
    geometry_logits: Tensor | None
    geometry_gate_logit: Tensor | None
    geometry_apply_mask: Tensor | None
    return_quantiles: Tensor
    viability_logits: Tensor
    outcome_values: Tensor
    recurrent_state: Tensor
    residual_energy: Tensor


@dataclass(frozen=True, slots=True)
class SequenceReplayTargets:
    """Labels for a padded ``[T, B]`` sequence.

    ``pair_weight`` is the confidence seam for exact causal joins versus
    inferred nearest-peer labels.  A zero weight leaves a field unlabeled.
    """

    valid_mask: Tensor
    act_index: Tensor
    wait_index: Tensor
    source_index: Tensor
    destination_index: Tensor
    kind_index: Tensor
    template_index: Tensor
    intent_index: Tensor
    policy_weight: Tensor | None = None
    pair_weight: Tensor | None = None
    geometry_index: Tensor | None = None
    geometry_weight: Tensor | None = None
    geometry_apply_target: Tensor | None = None
    return_target: Tensor | None = None
    value_mask: Tensor | None = None
    viability_target: Tensor | None = None
    viability_mask: Tensor | None = None
    outcome_target: Tensor | None = None
    outcome_mask: Tensor | None = None

    def time_slice(self, start: int, stop: int) -> SequenceReplayTargets:
        if not 0 <= start < stop <= self.valid_mask.shape[0]:
            raise IndexError("sequence target time slice is invalid")
        values = {
            field.name: (
                value[start:stop]
                if isinstance(value := getattr(self, field.name), Tensor)
                else value
            )
            for field in fields(self)
        }
        return replace(self, **values)

    def to(self, device: torch.device | str) -> SequenceReplayTargets:
        target = torch.device(device)
        values = {
            field.name: (
                value.to(target)
                if isinstance(value := getattr(self, field.name), Tensor)
                else value
            )
            for field in fields(self)
        }
        return replace(self, **values)


@dataclass(frozen=True, slots=True)
class SequenceReplayBatch:
    global_features: Tensor
    body_features: Tensor
    body_mask: Tensor
    event_features: Tensor
    reset_before: Tensor
    targets: SequenceReplayTargets
    base_geometry_logits: Tensor | None = None
    geometry_source_index: Tensor | None = None
    geometry_destination_index: Tensor | None = None
    geometry_pair_mask: Tensor | None = None
    geometry_candidate_mask: Tensor | None = None

    @property
    def time(self) -> int:
        return int(self.global_features.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.global_features.shape[1])

    def time_slice(self, start: int, stop: int) -> SequenceReplayBatch:
        if not 0 <= start < stop <= self.time:
            raise IndexError("sequence replay time slice is invalid")
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, SequenceReplayTargets):
                values[field.name] = value.time_slice(start, stop)
            elif isinstance(value, Tensor):
                values[field.name] = value[start:stop]
            else:
                values[field.name] = value
        return replace(self, **values)

    def to(self, device: torch.device | str) -> SequenceReplayBatch:
        target = torch.device(device)
        values: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = (
                value.to(target)
                if isinstance(value, (Tensor, SequenceReplayTargets))
                else value
            )
        return replace(self, **values)


def batch_steering_example_sequences(
    episodes: Sequence[Sequence[SteeringExample]],
    *,
    event_sequences: Sequence[Tensor] | None = None,
    pair_weight_sequences: Sequence[Tensor] | None = None,
    device: torch.device | str = "cpu",
) -> SequenceReplayBatch:
    """Pad chronological steering examples without inventing event history."""

    grouped = tuple(tuple(episode) for episode in episodes)
    if not grouped or any(not episode for episode in grouped):
        raise ValueError("steering sequence groups must be nonempty")
    examples = tuple(value for episode in grouped for value in episode)
    if any(not isinstance(value, SteeringExample) for value in examples):
        raise TypeError("steering sequence contains a non-steering example")
    schema = examples[0].observation.schema
    pointer_spec = examples[0].pointer_spec
    if (
        any(value.schema_sha256 != schema.sha256 for value in examples)
        or any(value.pointer_spec_sha256 != pointer_spec.sha256 for value in examples)
        or any(
            len({value.episode_identity for value in episode}) != 1
            for episode in grouped
        )
    ):
        raise ValueError("steering sequence identities or schemas disagree")
    supplied_events = (
        None if event_sequences is None else tuple(event_sequences)
    )
    supplied_pair_weights = (
        None if pair_weight_sequences is None else tuple(pair_weight_sequences)
    )
    if (
        supplied_events is not None
        and len(supplied_events) != len(grouped)
    ) or (
        supplied_pair_weights is not None
        and len(supplied_pair_weights) != len(grouped)
    ):
        raise ValueError("sequence side inputs must have one tensor per episode")

    target = torch.device(device)
    time = max(len(episode) for episode in grouped)
    batch = len(grouped)
    active_width = max(
        (
            int(active[-1]) + 1 if active.size else 1
            for value in examples
            for active in [value.observation.body_mask[0].nonzero()[0]]
        ),
        default=1,
    )
    sample = torch.from_numpy(examples[0].observation.global_features)
    dtype = sample.dtype
    globals_out = torch.zeros(
        time,
        batch,
        len(schema.global_features),
        dtype=dtype,
        device=target,
    )
    bodies_out = torch.zeros(
        time,
        batch,
        active_width,
        len(schema.body_features),
        dtype=dtype,
        device=target,
    )
    body_mask = torch.zeros(
        time, batch, active_width, dtype=torch.bool, device=target
    )
    events_out = torch.zeros(
        time,
        batch,
        SEQUENCE_REPLAY_EVENT_WIDTH,
        dtype=dtype,
        device=target,
    )
    valid = torch.zeros(time, batch, dtype=torch.bool, device=target)
    reset = torch.zeros_like(valid)
    labels = {
        name: torch.zeros(time, batch, dtype=torch.long, device=target)
        for name in (
            "act_index",
            "wait_index",
            "source_index",
            "destination_index",
            "kind_index",
            "template_index",
            "intent_index",
        )
    }
    policy_weight = torch.zeros(time, batch, dtype=dtype, device=target)
    pair_weight = torch.zeros_like(policy_weight)
    for lane, episode in enumerate(grouped):
        length = len(episode)
        valid[:length, lane] = True
        reset[0, lane] = True
        policy_weight[:length, lane] = 1.0
        if supplied_events is not None:
            event = supplied_events[lane]
            if event.shape != (length, SEQUENCE_REPLAY_EVENT_WIDTH):
                raise ValueError("event sequence has an invalid shape")
            events_out[:length, lane].copy_(event.to(target, dtype=dtype))
        if supplied_pair_weights is None:
            pair_weight[:length, lane] = 1.0
        else:
            weight = supplied_pair_weights[lane]
            if (
                weight.shape != (length,)
                or not weight.is_floating_point()
                or not bool(torch.isfinite(weight).all())
                or bool((weight < 0).any())
            ):
                raise ValueError("pair confidence must be finite nonnegative [L]")
            pair_weight[:length, lane].copy_(weight.to(target, dtype=dtype))
        for step, example in enumerate(episode):
            observation = example.observation
            globals_out[step, lane].copy_(
                torch.from_numpy(observation.global_features[0]).to(target)
            )
            bodies_out[step, lane].copy_(
                torch.from_numpy(
                    observation.body_features[0, :active_width]
                ).to(target)
            )
            body_mask[step, lane].copy_(
                torch.from_numpy(observation.body_mask[0, :active_width]).to(
                    target
                )
            )
            for name, output in labels.items():
                output[step, lane] = getattr(example, name)
    targets = SequenceReplayTargets(
        valid,
        labels["act_index"],
        labels["wait_index"],
        labels["source_index"],
        labels["destination_index"],
        labels["kind_index"],
        labels["template_index"],
        labels["intent_index"],
        policy_weight,
        pair_weight,
    )
    return SequenceReplayBatch(
        globals_out, bodies_out, body_mask, events_out, reset, targets
    )


@dataclass(frozen=True, slots=True)
class SequenceReplayLossWeights:
    act: float = 1.0
    wait: float = 0.25
    pair: float = 1.0
    kind: float = 0.1
    template: float = 0.1
    intent: float = 0.25
    geometry: float = 1.0
    geometry_gate: float = 0.25
    value: float = 0.1
    viability: float = 0.25
    outcome: float = 0.1
    residual: float = 1e-4
    quantile_huber_kappa: float = 1.0

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"loss weight {name} must be finite and nonnegative")
        if self.quantile_huber_kappa <= 0.0:
            raise ValueError("quantile Huber kappa must be positive")


@dataclass(frozen=True, slots=True)
class SequenceReplayLoss:
    total: Tensor
    act: Tensor
    wait: Tensor
    pair: Tensor
    kind: Tensor
    template: Tensor
    intent: Tensor
    geometry: Tensor
    geometry_gate: Tensor
    value: Tensor
    viability: Tensor
    outcome: Tensor
    residual: Tensor

    def scalars(self) -> dict[str, float]:
        return {
            field.name: float(getattr(self, field.name).detach())
            for field in fields(self)
        }


class SequenceReplayModel(nn.Module):
    """Zero-initialized recurrent residual around a frozen steering model."""

    def __init__(
        self,
        base_model: GoalConditionedSteeringModel,
        *,
        config: SequenceReplayConfig | None = None,
        base_checkpoint_sha256: str | None = None,
        geometry_candidate_count: int = 0,
        geometry_candidate_set_sha256: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(base_model, GoalConditionedSteeringModel):
            raise TypeError("base model must be a GoalConditionedSteeringModel")
        if (
            type(geometry_candidate_count) is not int
            or geometry_candidate_count < 0
            or geometry_candidate_count == 1
        ):
            raise ValueError("geometry candidate count must be zero or at least two")
        self.base_model = base_model
        self.base_model.requires_grad_(False)
        self.base_model.eval()
        self.config = config or SequenceReplayConfig()
        self.schema = base_model.schema
        self.pointer_spec = base_model.pointer_spec
        self.base_checkpoint_sha256 = _optional_sha256(
            base_checkpoint_sha256, "base checkpoint identity"
        )
        self.geometry_candidate_count = geometry_candidate_count
        self.geometry_candidate_set_sha256 = _optional_sha256(
            geometry_candidate_set_sha256, "geometry candidate-set identity"
        )
        if bool(geometry_candidate_count) != bool(geometry_candidate_set_sha256):
            raise ValueError(
                "geometry candidate count and candidate-set identity must coexist"
            )
        self._id_indices = tuple(
            self.schema.body_features.index(name)
            for name in ("id_scaled", "chain_id_scaled")
            if name in self.schema.body_features
        )
        self._global_ablation_indices = tuple(
            self.schema.global_features.index(name)
            for name in ("tick_scaled",)
            if name in self.schema.global_features
        )

        cfg = self.config
        self.body_encoder = nn.Sequential(
            nn.Linear(len(self.schema.body_features), cfg.body_hidden),
            nn.LayerNorm(cfg.body_hidden),
            nn.GELU(),
            nn.Linear(cfg.body_hidden, cfg.body_hidden),
            nn.GELU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(len(self.schema.global_features), cfg.global_hidden),
            nn.GELU(),
        )
        self.event_encoder = nn.Sequential(
            nn.Linear(SEQUENCE_REPLAY_EVENT_WIDTH, cfg.event_hidden),
            nn.GELU(),
        )
        recurrent_input = 2 * cfg.body_hidden + cfg.global_hidden + cfg.event_hidden
        self.recurrent = nn.GRUCell(recurrent_input, cfg.recurrent_hidden)
        self.act_residual = nn.Linear(cfg.recurrent_hidden, 2)
        self.wait_residual = nn.Linear(
            cfg.recurrent_hidden, len(self.pointer_spec.wait_choices)
        )

        pair_input = 2 * cfg.body_hidden + cfg.recurrent_hidden + 9
        self.pair_encoder = nn.Sequential(
            nn.Linear(pair_input, cfg.pair_hidden),
            nn.LayerNorm(cfg.pair_hidden),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(cfg.pair_hidden, cfg.pair_hidden),
            nn.GELU(),
        )
        self.pair_residual = nn.Linear(cfg.pair_hidden, 1)
        self.kind_residual = nn.Linear(cfg.pair_hidden, 2)
        self.template_residual = nn.Linear(
            cfg.pair_hidden, self.pointer_spec.template_count
        )
        self.intent_residual = nn.Linear(
            cfg.pair_hidden, len(tuple(SteeringIntent))
        )

        if geometry_candidate_count:
            geometry_input = cfg.pair_hidden + cfg.recurrent_hidden
            self.geometry_residual: nn.Linear | None = nn.Linear(
                geometry_input, geometry_candidate_count
            )
            self.geometry_gate: nn.Linear | None = nn.Linear(geometry_input, 1)
        else:
            self.geometry_residual = None
            self.geometry_gate = None
        self.value_head = nn.Linear(cfg.recurrent_hidden, cfg.value_quantiles)
        self.viability_head = nn.Linear(
            cfg.recurrent_hidden, len(VIABILITY_TARGETS)
        )
        self.outcome_head = nn.Linear(
            cfg.recurrent_hidden, len(OUTCOME_TARGETS)
        )
        for layer in self._residual_heads():
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def _residual_heads(self) -> tuple[nn.Linear, ...]:
        heads = [
            self.act_residual,
            self.wait_residual,
            self.pair_residual,
            self.kind_residual,
            self.template_residual,
            self.intent_residual,
            self.value_head,
            self.viability_head,
            self.outcome_head,
        ]
        if self.geometry_residual is not None:
            heads.extend((self.geometry_residual, self.geometry_gate))
        return tuple(head for head in heads if head is not None)

    def train(self, mode: bool = True) -> SequenceReplayModel:
        super().train(mode)
        self.base_model.eval()
        return self

    def manifest(self) -> dict[str, object]:
        return {
            "architecture": "sequence-replay-recurrent-residual-v1",
            "schema_sha256": self.schema.sha256,
            "pointer_action_sha256": self.pointer_spec.sha256,
            "base": {
                "architecture_sha256": self.base_model.architecture_sha256,
                "checkpoint_sha256": self.base_checkpoint_sha256,
                "state_sha256": _module_state_sha256(self.base_model),
                "frozen": True,
            },
            "input_ablations": {
                "body": [
                    self.schema.body_features[index] for index in self._id_indices
                ],
                "global": [
                    self.schema.global_features[index]
                    for index in self._global_ablation_indices
                ],
            },
            "event": sequence_replay_event_manifest(),
            "pair_relations": self.base_model.manifest()["public_pair_relations"],
            "geometry": {
                "candidate_count": self.geometry_candidate_count,
                "candidate_set_sha256": self.geometry_candidate_set_sha256,
                "gate_threshold": self.config.geometry_gate_threshold,
                "initial_action": "frozen-baseline",
            },
            "values": {
                "quantiles": self.config.value_quantiles,
                "viability": list(VIABILITY_TARGETS),
                "outcomes": list(OUTCOME_TARGETS),
            },
            "config": asdict(self.config),
        }

    @property
    def architecture_sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch size must be positive")
        parameter = next(self.body_encoder.parameters())
        return torch.zeros(
            batch_size,
            self.config.recurrent_hidden,
            device=parameter.device if device is None else device,
            dtype=parameter.dtype if dtype is None else dtype,
        )

    def _clean_bodies(self, body_features: Tensor) -> Tensor:
        clean = body_features.clone()
        if self._id_indices:
            clean[..., self._id_indices] = 0.0
        return clean

    def _clean_globals(self, global_features: Tensor) -> Tensor:
        clean = global_features.clone()
        if self._global_ablation_indices:
            clean[..., self._global_ablation_indices] = 0.0
        return clean

    def _validate_inputs(
        self,
        global_features: Tensor,
        body_features: Tensor,
        body_mask: Tensor,
        event_features: Tensor,
        reset_before: Tensor,
        valid_mask: Tensor,
    ) -> tuple[int, int, int]:
        if global_features.ndim != 3 or body_features.ndim != 4:
            raise ValueError("sequence observations need [T,B,G] and [T,B,N,F]")
        time, batch, global_count = global_features.shape
        bodies = body_features.shape[2]
        if (
            time < 1
            or batch < 1
            or not 0 < bodies <= self.schema.capacity
            or global_count != len(self.schema.global_features)
            or body_features.shape
            != (time, batch, bodies, len(self.schema.body_features))
            or body_mask.shape != (time, batch, bodies)
            or event_features.shape
            != (time, batch, SEQUENCE_REPLAY_EVENT_WIDTH)
            or reset_before.shape != (time, batch)
            or valid_mask.shape != (time, batch)
            or body_mask.dtype != torch.bool
            or reset_before.dtype != torch.bool
            or valid_mask.dtype != torch.bool
        ):
            raise ValueError("sequence replay inputs differ from the bound schemas")
        floating = (global_features, body_features, event_features)
        if (
            not all(value.is_floating_point() for value in floating)
            or len({value.dtype for value in floating}) != 1
            or len({value.device for value in floating}) != 1
            or body_mask.device != global_features.device
            or reset_before.device != global_features.device
            or valid_mask.device != global_features.device
            or not all(bool(torch.isfinite(value).all()) for value in floating)
        ):
            raise ValueError("sequence replay tensors are malformed")
        return time, batch, bodies

    def forward(
        self,
        global_features: Tensor,
        body_features: Tensor,
        body_mask: Tensor,
        event_features: Tensor,
        reset_before: Tensor,
        *,
        valid_mask: Tensor | None = None,
        initial_state: Tensor | None = None,
        base_geometry_logits: Tensor | None = None,
        geometry_source_index: Tensor | None = None,
        geometry_destination_index: Tensor | None = None,
        geometry_pair_mask: Tensor | None = None,
        geometry_candidate_mask: Tensor | None = None,
    ) -> SequenceReplayOutput:
        if valid_mask is None:
            valid_mask = torch.ones_like(reset_before)
        time, batch, bodies = self._validate_inputs(
            global_features,
            body_features,
            body_mask,
            event_features,
            reset_before,
            valid_mask,
        )
        clean_bodies = self._clean_bodies(body_features)
        encoded_body = self.body_encoder(clean_bodies)
        mask_float = body_mask.unsqueeze(-1).to(encoded_body.dtype)
        pooled_mean = (encoded_body * mask_float).sum(dim=2) / mask_float.sum(
            dim=2
        ).clamp_min(1.0)
        floor = torch.finfo(encoded_body.dtype).min
        pooled_max = encoded_body.masked_fill(~body_mask.unsqueeze(-1), floor).max(
            dim=2
        ).values
        pooled_max = torch.where(
            body_mask.any(dim=2, keepdim=True),
            pooled_max,
            torch.zeros_like(pooled_max),
        )
        recurrent_input = torch.cat(
            (
                self.global_encoder(self._clean_globals(global_features)),
                pooled_mean,
                pooled_max,
                self.event_encoder(event_features),
            ),
            dim=-1,
        )
        state = (
            self.initial_state(
                batch,
                device=global_features.device,
                dtype=global_features.dtype,
            )
            if initial_state is None
            else initial_state
        )
        if state.shape != (batch, self.config.recurrent_hidden):
            raise ValueError("initial recurrent state has the wrong shape")
        states = []
        for step in range(time):
            state = torch.where(
                reset_before[step, :, None], torch.zeros_like(state), state
            )
            candidate = self.recurrent(recurrent_input[step], state)
            state = torch.where(valid_mask[step, :, None], candidate, state)
            states.append(
                torch.where(
                    valid_mask[step, :, None], state, torch.zeros_like(state)
                )
            )
        recurrent = torch.stack(states)

        flat_valid = valid_mask.reshape(-1)
        valid_rows = flat_valid.nonzero(as_tuple=False).reshape(-1)
        if not valid_rows.numel():
            raise ValueError("a sequence replay batch needs a valid timestep")
        flat_globals = global_features.flatten(0, 1)
        flat_bodies = body_features.flatten(0, 1)
        flat_mask = body_mask.flatten(0, 1)
        selected_globals = flat_globals[valid_rows]
        selected_bodies = flat_bodies[valid_rows]
        selected_mask = flat_mask[valid_rows]
        with torch.no_grad():
            base = self.base_model(
                selected_globals, selected_bodies, selected_mask
            )
        selected_encoded = encoded_body.flatten(0, 1)[valid_rows]
        selected_state = recurrent.flatten(0, 1)[valid_rows]
        count = selected_encoded.shape[1]
        source = selected_encoded.unsqueeze(2).expand(-1, -1, count, -1)
        destination = selected_encoded.unsqueeze(1).expand(-1, count, -1, -1)
        state_pair = selected_state[:, None, None, :].expand(
            -1, count, count, -1
        )
        relations = self.base_model._public_pair_relations(  # noqa: SLF001
            selected_bodies, selected_mask
        )
        pair = self.pair_encoder(
            torch.cat((source, destination, state_pair, relations), dim=-1)
        )
        scale = float(self.config.residual_scale)
        act_residual = scale * torch.tanh(self.act_residual(selected_state))
        wait_residual = scale * torch.tanh(self.wait_residual(selected_state))
        pair_residual = scale * torch.tanh(
            self.pair_residual(pair).squeeze(-1)
        )
        kind_residual = scale * torch.tanh(self.kind_residual(pair))
        template_residual = scale * torch.tanh(self.template_residual(pair))
        intent_residual = scale * torch.tanh(self.intent_residual(pair))
        legal = base.legal_pair_mask
        selected_output = GoalConditionedSteeringOutput(
            base.act_logits + act_residual,
            base.wait_logits + wait_residual,
            (base.pair_logits + pair_residual).masked_fill(~legal, floor),
            base.kind_logits + kind_residual,
            base.template_logits + template_residual,
            base.intent_logits + intent_residual,
            legal,
        )

        total_rows = time * batch

        def scatter(value: Tensor, *, fill: float = 0.0) -> Tensor:
            output = value.new_full((total_rows, *value.shape[1:]), fill)
            return output.index_copy(0, valid_rows, value).reshape(
                time, batch, *value.shape[1:]
            )

        act_logits = scatter(selected_output.act_logits)
        wait_logits = scatter(selected_output.wait_logits)
        pair_logits = scatter(selected_output.pair_logits, fill=floor)
        kind_logits = scatter(selected_output.kind_logits)
        template_logits = scatter(selected_output.template_logits)
        intent_logits = scatter(selected_output.intent_logits)
        legal_pair_mask = scatter(
            selected_output.legal_pair_mask, fill=False
        ).bool()

        geometry_logits: Tensor | None = None
        geometry_gate_logit: Tensor | None = None
        geometry_apply_mask: Tensor | None = None
        geometry_residual_energy = selected_state.sum() * 0.0
        if base_geometry_logits is not None:
            if self.geometry_residual is None or self.geometry_gate is None:
                raise ValueError("this sequence model has no geometry correction")
            expected = (time, batch, self.geometry_candidate_count)
            pair_shape = (time, batch)
            if (
                base_geometry_logits.shape != expected
                or geometry_source_index is None
                or geometry_destination_index is None
                or geometry_source_index.shape != pair_shape
                or geometry_destination_index.shape != pair_shape
            ):
                raise ValueError("geometry correction tensors have invalid shapes")
            pair_mask = (
                valid_mask
                if geometry_pair_mask is None
                else valid_mask & geometry_pair_mask
            )
            if pair_mask.shape != pair_shape or pair_mask.dtype != torch.bool:
                raise ValueError("geometry pair mask must be boolean [T,B]")
            selected_source = geometry_source_index.reshape(-1)[valid_rows]
            selected_destination = geometry_destination_index.reshape(-1)[valid_rows]
            selected_pair_mask = pair_mask.reshape(-1)[valid_rows]
            invalid_index = (
                (selected_source < 0)
                | (selected_source >= bodies)
                | (selected_destination < 0)
                | (selected_destination >= bodies)
            )
            if bool((selected_pair_mask & invalid_index).any()):
                raise ValueError("geometry correction pair index is out of range")
            safe_source = selected_source.clamp(0, bodies - 1)
            safe_destination = selected_destination.clamp(0, bodies - 1)
            rows = torch.arange(valid_rows.numel(), device=valid_rows.device)
            active_rows = rows[selected_pair_mask]
            if bool(selected_pair_mask.any()) and not bool(
                legal[
                    active_rows,
                    selected_source[selected_pair_mask],
                    selected_destination[selected_pair_mask],
                ].all()
            ):
                raise ValueError("geometry correction selects an illegal pair")
            selected_pair = pair[rows, safe_source, safe_destination]
            geometry_context = torch.cat((selected_pair, selected_state), dim=-1)
            raw_geometry = float(self.config.geometry_residual_scale) * torch.tanh(
                self.geometry_residual(geometry_context)
            )
            selected_gate_logit = self.geometry_gate(
                geometry_context
            ).squeeze(-1)
            gate = torch.sigmoid(selected_gate_logit)
            selected_base_geometry = base_geometry_logits.flatten(0, 1)[valid_rows]
            correction_gate = gate * selected_pair_mask.to(gate.dtype)
            selected_geometry = (
                selected_base_geometry + correction_gate[:, None] * raw_geometry
            )
            selected_candidate_mask = None
            if geometry_candidate_mask is not None:
                if (
                    geometry_candidate_mask.shape != expected
                    or geometry_candidate_mask.dtype != torch.bool
                ):
                    raise ValueError(
                        "geometry candidate mask must be boolean [T,B,K]"
                    )
                selected_candidate_mask = geometry_candidate_mask.flatten(
                    0, 1
                )[valid_rows]
                selected_geometry = selected_geometry.masked_fill(
                    ~selected_candidate_mask, floor
                )
            selected_apply = selected_pair_mask & (
                gate >= float(self.config.geometry_gate_threshold)
            )
            if selected_candidate_mask is not None:
                selected_apply &= selected_candidate_mask.sum(dim=-1) >= 2
            geometry_logits = scatter(selected_geometry, fill=floor)
            geometry_gate_logit = scatter(selected_gate_logit)
            geometry_apply_mask = scatter(selected_apply, fill=False).bool()
            geometry_residual_energy = _weighted_mean(
                raw_geometry.square().mean(dim=-1),
                selected_pair_mask.to(raw_geometry.dtype),
                selected_state.sum() * 0.0,
            )
        elif any(
            value is not None
            for value in (
                geometry_source_index,
                geometry_destination_index,
                geometry_pair_mask,
                geometry_candidate_mask,
            )
        ):
            raise ValueError("geometry metadata requires base geometry logits")

        legal_float = legal.to(pair_residual.dtype)
        legal_count = legal_float.sum().clamp_min(1.0)
        pair_energy = (
            pair_residual.square() * legal_float
        ).sum() / legal_count
        residual_energy = (
            act_residual.square().mean()
            + wait_residual.square().mean()
            + pair_energy
            + (kind_residual.square().mean(dim=-1) * legal_float).sum()
            / legal_count
            + (template_residual.square().mean(dim=-1) * legal_float).sum()
            / legal_count
            + (intent_residual.square().mean(dim=-1) * legal_float).sum()
            / legal_count
            + geometry_residual_energy
        )
        return SequenceReplayOutput(
            act_logits,
            wait_logits,
            pair_logits,
            kind_logits,
            template_logits,
            intent_logits,
            legal_pair_mask,
            geometry_logits,
            geometry_gate_logit,
            geometry_apply_mask,
            self.value_head(recurrent),
            self.viability_head(recurrent),
            self.outcome_head(recurrent),
            state,
            residual_energy,
        )


class SequenceReplayStream:
    """Stateful one-decision inference adapter; outputs retain a length-1 axis."""

    def __init__(self, model: SequenceReplayModel, *, batch_size: int = 1) -> None:
        if not isinstance(model, SequenceReplayModel):
            raise TypeError("stream model must be a SequenceReplayModel")
        self.model = model
        self.state = model.initial_state(batch_size)

    def reset(self) -> None:
        self.state.zero_()

    @torch.no_grad()
    def step(
        self,
        global_features: Tensor,
        body_features: Tensor,
        body_mask: Tensor,
        event_features: Tensor,
        *,
        reset_before: Tensor | None = None,
        base_geometry_logits: Tensor | None = None,
        geometry_source_index: Tensor | None = None,
        geometry_destination_index: Tensor | None = None,
        geometry_pair_mask: Tensor | None = None,
        geometry_candidate_mask: Tensor | None = None,
    ) -> SequenceReplayOutput:
        batch = self.state.shape[0]
        if global_features.shape[0] != batch:
            raise ValueError("stream observation batch size changed")
        reset = (
            torch.zeros(batch, dtype=torch.bool, device=global_features.device)
            if reset_before is None
            else reset_before
        )

        def time_axis(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.unsqueeze(0)

        self.model.eval()
        output = self.model(
            global_features.unsqueeze(0),
            body_features.unsqueeze(0),
            body_mask.unsqueeze(0),
            event_features.unsqueeze(0),
            reset.unsqueeze(0),
            initial_state=self.state,
            base_geometry_logits=time_axis(base_geometry_logits),
            geometry_source_index=time_axis(geometry_source_index),
            geometry_destination_index=time_axis(geometry_destination_index),
            geometry_pair_mask=time_axis(geometry_pair_mask),
            geometry_candidate_mask=time_axis(geometry_candidate_mask),
        )
        self.state = output.recurrent_state.detach()
        return output


def _weights(
    value: Tensor | None, mask: Tensor, reference: Tensor
) -> Tensor:
    output = mask.to(reference.dtype) if value is None else value
    if (
        output.shape != mask.shape
        or not output.is_floating_point()
        or output.device != mask.device
        or not bool(torch.isfinite(output).all())
        or bool((output < 0).any())
    ):
        raise ValueError("sequence replay weights must be finite nonnegative [T,B]")
    return output * mask.to(output.dtype)


def _weighted_mean(value: Tensor, weight: Tensor, zero: Tensor) -> Tensor:
    total = weight.sum()
    return (value * weight).sum() / total if bool(total > 0) else zero


def _weighted_cross_entropy(
    logits: Tensor, target: Tensor, weight: Tensor, zero: Tensor
) -> Tensor:
    active = weight > 0
    if not bool(active.any()):
        return zero
    per_row = F.cross_entropy(logits[active], target[active], reduction="none")
    return _weighted_mean(per_row, weight[active], zero)


def sequence_replay_objective(
    output: SequenceReplayOutput,
    targets: SequenceReplayTargets,
    *,
    weights: SequenceReplayLossWeights | None = None,
) -> SequenceReplayLoss:
    resolved = weights or SequenceReplayLossWeights()
    shape = output.act_logits.shape[:2]
    labels = (
        targets.act_index,
        targets.wait_index,
        targets.source_index,
        targets.destination_index,
        targets.kind_index,
        targets.template_index,
        targets.intent_index,
    )
    if (
        targets.valid_mask.shape != shape
        or targets.valid_mask.dtype != torch.bool
        or any(value.shape != shape or value.dtype != torch.long for value in labels)
    ):
        raise ValueError("sequence replay policy labels must have shape [T,B]")
    zero = output.act_logits.sum() * 0.0
    policy_weight = _weights(
        targets.policy_weight, targets.valid_mask, output.act_logits
    )
    act = _weighted_cross_entropy(
        output.act_logits, targets.act_index, policy_weight, zero
    )
    wait_weight = policy_weight * (targets.act_index == 0)
    wait = _weighted_cross_entropy(
        output.wait_logits, targets.wait_index, wait_weight, zero
    )
    shot_mask = targets.valid_mask & (targets.act_index == 1)
    pair_weight = _weights(targets.pair_weight, shot_mask, output.act_logits)
    pair_rows = pair_weight > 0
    if bool(pair_rows.any()):
        source = targets.source_index[pair_rows]
        destination = targets.destination_index[pair_rows]
        rows = pair_rows.nonzero(as_tuple=True)
        count = output.pair_logits.shape[-1]
        if bool(
            (
                (source < 0)
                | (source >= count)
                | (destination < 0)
                | (destination >= count)
            ).any()
        ):
            raise ValueError("sequence pair label is out of range")
        if not bool(output.legal_pair_mask[(*rows, source, destination)].all()):
            raise ValueError("sequence pair label is masked by public semantics")
        pair_target = source * count + destination
        pair = _weighted_cross_entropy(
            output.pair_logits[pair_rows].flatten(1),
            pair_target,
            pair_weight[pair_rows],
            zero,
        )
        selected = (*rows, source, destination)
        kind = _weighted_cross_entropy(
            output.kind_logits[selected],
            targets.kind_index[pair_rows],
            pair_weight[pair_rows],
            zero,
        )
        template = _weighted_cross_entropy(
            output.template_logits[selected],
            targets.template_index[pair_rows],
            pair_weight[pair_rows],
            zero,
        )
        intent = _weighted_cross_entropy(
            output.intent_logits[selected],
            targets.intent_index[pair_rows],
            pair_weight[pair_rows],
            zero,
        )
    else:
        pair = kind = template = intent = zero

    geometry = geometry_gate = zero
    if targets.geometry_index is not None or targets.geometry_weight is not None:
        if (
            output.geometry_logits is None
            or output.geometry_gate_logit is None
            or targets.geometry_index is None
            or targets.geometry_index.shape != shape
            or targets.geometry_index.dtype != torch.long
        ):
            raise ValueError("geometry targets and model outputs disagree")
        geometry_weight = _weights(
            targets.geometry_weight, targets.valid_mask, output.act_logits
        )
        geometry = _weighted_cross_entropy(
            output.geometry_logits,
            targets.geometry_index,
            geometry_weight,
            zero,
        )
        if targets.geometry_apply_target is not None:
            if targets.geometry_apply_target.shape != shape:
                raise ValueError("geometry gate target must have shape [T,B]")
            gate_target = targets.geometry_apply_target.to(
                output.geometry_gate_logit.dtype
            )
            gate_rows = geometry_weight > 0
            if bool(gate_rows.any()):
                gate_loss = F.binary_cross_entropy_with_logits(
                    output.geometry_gate_logit[gate_rows],
                    gate_target[gate_rows],
                    reduction="none",
                )
                geometry_gate = _weighted_mean(
                    gate_loss, geometry_weight[gate_rows], zero
                )

    value = zero
    if targets.return_target is not None:
        value_mask = (
            targets.valid_mask
            if targets.value_mask is None
            else targets.valid_mask & targets.value_mask
        )
        if (
            targets.return_target.shape != shape
            or value_mask.shape != shape
            or value_mask.dtype != torch.bool
        ):
            raise ValueError("return targets must have shape [T,B]")
        if bool(value_mask.any()):
            value = quantile_huber_loss(
                output.return_quantiles,
                targets.return_target,
                value_mask,
                resolved.quantile_huber_kappa,
            )

    viability = zero
    if targets.viability_target is not None:
        viability_mask = (
            targets.valid_mask
            if targets.viability_mask is None
            else targets.valid_mask & targets.viability_mask
        )
        if (
            targets.viability_target.shape != output.viability_logits.shape
            or viability_mask.shape != shape
            or viability_mask.dtype != torch.bool
        ):
            raise ValueError("viability targets have invalid shapes")
        if bool(viability_mask.any()):
            viability = F.binary_cross_entropy_with_logits(
                output.viability_logits[viability_mask],
                targets.viability_target[viability_mask],
            )

    outcome = zero
    if targets.outcome_target is not None:
        outcome_mask = (
            targets.valid_mask
            if targets.outcome_mask is None
            else targets.valid_mask & targets.outcome_mask
        )
        if (
            targets.outcome_target.shape != output.outcome_values.shape
            or outcome_mask.shape != shape
            or outcome_mask.dtype != torch.bool
        ):
            raise ValueError("outcome targets have invalid shapes")
        if bool(outcome_mask.any()):
            outcome = F.smooth_l1_loss(
                output.outcome_values[outcome_mask],
                targets.outcome_target[outcome_mask],
            )

    residual = output.residual_energy
    total = (
        resolved.act * act
        + resolved.wait * wait
        + resolved.pair * pair
        + resolved.kind * kind
        + resolved.template * template
        + resolved.intent * intent
        + resolved.geometry * geometry
        + resolved.geometry_gate * geometry_gate
        + resolved.value * value
        + resolved.viability * viability
        + resolved.outcome * outcome
        + resolved.residual * residual
    )
    return SequenceReplayLoss(
        total,
        act,
        wait,
        pair,
        kind,
        template,
        intent,
        geometry,
        geometry_gate,
        value,
        viability,
        outcome,
        residual,
    )


def forward_sequence_replay(
    model: SequenceReplayModel,
    batch: SequenceReplayBatch,
    *,
    initial_state: Tensor | None = None,
) -> SequenceReplayOutput:
    return model(
        batch.global_features,
        batch.body_features,
        batch.body_mask,
        batch.event_features,
        batch.reset_before,
        valid_mask=batch.targets.valid_mask,
        initial_state=initial_state,
        base_geometry_logits=batch.base_geometry_logits,
        geometry_source_index=batch.geometry_source_index,
        geometry_destination_index=batch.geometry_destination_index,
        geometry_pair_mask=batch.geometry_pair_mask,
        geometry_candidate_mask=batch.geometry_candidate_mask,
    )


def sequence_replay_loss(
    model: SequenceReplayModel,
    batch: SequenceReplayBatch,
    *,
    weights: SequenceReplayLossWeights | None = None,
    initial_state: Tensor | None = None,
) -> tuple[SequenceReplayLoss, SequenceReplayOutput]:
    output = forward_sequence_replay(model, batch, initial_state=initial_state)
    return sequence_replay_objective(output, batch.targets, weights=weights), output


def train_sequence_replay_step(
    model: SequenceReplayModel,
    batch: SequenceReplayBatch,
    optimizer: torch.optim.Optimizer,
    *,
    weights: SequenceReplayLossWeights | None = None,
    initial_state: Tensor | None = None,
    max_gradient_norm: float = 1.0,
) -> tuple[SequenceReplayLoss, Tensor, float]:
    if (
        isinstance(max_gradient_norm, bool)
        or not isinstance(max_gradient_norm, (int, float))
        or not math.isfinite(float(max_gradient_norm))
        or float(max_gradient_norm) <= 0.0
    ):
        raise ValueError("maximum gradient norm must be finite and positive")
    model.train()
    loss, output = sequence_replay_loss(
        model, batch, weights=weights, initial_state=initial_state
    )
    optimizer.zero_grad(set_to_none=True)
    loss.total.backward()
    trainable = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    gradient = nn.utils.clip_grad_norm_(trainable, float(max_gradient_norm))
    if not math.isfinite(float(gradient)):
        raise FloatingPointError("sequence replay gradient norm is nonfinite")
    optimizer.step()
    detached_loss = replace(
        loss,
        **{
            field.name: getattr(loss, field.name).detach()
            for field in fields(loss)
        },
    )
    return detached_loss, output.recurrent_state.detach(), float(gradient)


@dataclass(frozen=True, slots=True)
class SequenceReplayCheckpoint:
    path: Path
    sha256: str
    model: SequenceReplayModel
    source_identity: str
    metadata: Mapping[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            {} if value is None else dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint metadata must be canonical JSON") from exc
    return json.loads(encoded)


def save_sequence_replay_checkpoint(
    path: str | Path,
    model: SequenceReplayModel,
    *,
    source_identity: str,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> str:
    """Atomically save trainable residuals, binding but not copying frozen v5."""

    if not isinstance(model, SequenceReplayModel):
        raise TypeError("checkpoint model must be a SequenceReplayModel")
    if model.base_checkpoint_sha256 is None:
        raise ValueError("an identity-bound base checkpoint is required")
    source = _optional_sha256(source_identity, "source identity")
    assert source is not None
    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"sequence replay checkpoint exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    residual_state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if not name.startswith("base_model.")
    }
    payload = {
        "format": _CHECKPOINT_FORMAT,
        "source_identity": source,
        "model_manifest": model.manifest(),
        "metadata": _json_mapping(metadata),
        "state_dict": residual_state,
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


def load_sequence_replay_checkpoint(
    path: str | Path,
    base_model: GoalConditionedSteeringModel,
    *,
    expected_sha256: str,
    expected_base_checkpoint_sha256: str,
    expected_source_identity: str,
    device: torch.device | str = "cpu",
) -> SequenceReplayCheckpoint:
    checkpoint_sha = _optional_sha256(expected_sha256, "checkpoint identity")
    base_sha = _optional_sha256(
        expected_base_checkpoint_sha256, "base checkpoint identity"
    )
    source_sha = _optional_sha256(expected_source_identity, "source identity")
    assert checkpoint_sha is not None and base_sha is not None
    assert source_sha is not None
    source = Path(path).resolve(strict=True)
    if _file_sha256(source) != checkpoint_sha:
        raise ValueError("sequence replay checkpoint SHA-256 mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    expected_fields = {
        "format",
        "source_identity",
        "model_manifest",
        "metadata",
        "state_dict",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ValueError("sequence replay checkpoint fields are malformed")
    if payload["format"] != _CHECKPOINT_FORMAT:
        raise ValueError("sequence replay checkpoint format is unsupported")
    if payload["source_identity"] != source_sha:
        raise ValueError("sequence replay source identity mismatch")
    manifest = payload["model_manifest"]
    if not isinstance(manifest, dict) or manifest.get("architecture") != (
        "sequence-replay-recurrent-residual-v1"
    ):
        raise ValueError("sequence replay model manifest is malformed")
    base_manifest = manifest.get("base")
    geometry_manifest = manifest.get("geometry")
    config_manifest = manifest.get("config")
    if (
        not isinstance(base_manifest, dict)
        or base_manifest.get("checkpoint_sha256") != base_sha
        or base_manifest.get("architecture_sha256")
        != base_model.architecture_sha256
        or base_manifest.get("state_sha256") != _module_state_sha256(base_model)
        or not isinstance(geometry_manifest, dict)
        or not isinstance(config_manifest, dict)
    ):
        raise ValueError("sequence replay checkpoint bindings disagree")
    try:
        config = SequenceReplayConfig(**config_manifest)
        model = SequenceReplayModel(
            base_model,
            config=config,
            base_checkpoint_sha256=base_sha,
            geometry_candidate_count=int(
                geometry_manifest["candidate_count"]
            ),
            geometry_candidate_set_sha256=geometry_manifest[
                "candidate_set_sha256"
            ],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("sequence replay checkpoint cannot be reconstructed") from exc
    if model.manifest() != manifest:
        raise ValueError("reconstructed sequence replay identity differs")
    supplied_state = payload["state_dict"]
    current_state = model.state_dict()
    residual_keys = {
        name for name in current_state if not name.startswith("base_model.")
    }
    if not isinstance(supplied_state, dict) or set(supplied_state) != residual_keys:
        raise ValueError("sequence replay residual state is malformed")
    current_state.update(supplied_state)
    model.load_state_dict(current_state, strict=True)
    model.to(device).eval()
    return SequenceReplayCheckpoint(
        source,
        checkpoint_sha,
        model,
        source_sha,
        _json_mapping(payload["metadata"]),
    )


__all__ = [
    "OUTCOME_TARGETS",
    "SEQUENCE_REPLAY_EVENT_FEATURES",
    "SEQUENCE_REPLAY_EVENT_WIDTH",
    "VIABILITY_TARGETS",
    "SequenceReplayBatch",
    "SequenceReplayCheckpoint",
    "SequenceReplayConfig",
    "SequenceReplayLoss",
    "SequenceReplayLossWeights",
    "SequenceReplayModel",
    "SequenceReplayOutput",
    "SequenceReplayStream",
    "SequenceReplayTargets",
    "batch_steering_example_sequences",
    "forward_sequence_replay",
    "load_sequence_replay_checkpoint",
    "save_sequence_replay_checkpoint",
    "sequence_replay_event_manifest",
    "sequence_replay_loss",
    "sequence_replay_objective",
    "train_sequence_replay_step",
]
