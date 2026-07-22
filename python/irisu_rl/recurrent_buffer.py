"""Owned time-major storage for recurrent R2 rollouts."""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from .actions import ActionSpec
from .encoding import EncodedBatch
from .ppo import RecurrentTrainingBatch
from .returns import AdvantageResult, smdp_gae
from .schema import TensorSchema
from .torch_distribution import ActionTensor, LogProbabilityComponents
from .vector_adapter import MacroTransition


class RecurrentRolloutBuffer:
    """Fixed-horizon synchronous lane storage with explicit censoring masks.

    The incoming recurrent state is the state before the first stored
    observation. Interrupted non-bootstrap truncations remain in audit tensors
    but are excluded from all policy and value losses.
    """

    def __init__(
        self,
        horizon: int,
        lanes: int,
        schema: TensorSchema,
        initial_state: Tensor,
        *,
        action_spec: ActionSpec | None = None,
        reward_scale: float = 1.0,
    ) -> None:
        for name, value in (("horizon", horizon), ("lanes", lanes)):
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not torch.isfinite(torch.tensor(reward_scale)) or reward_scale <= 0:
            raise ValueError("reward scale must be finite and positive")
        if initial_state.ndim != 3 or initial_state.shape[1] != lanes:
            raise ValueError("initial recurrent state must have shape [L, B, H]")
        if (
            not initial_state.is_floating_point()
            or not torch.isfinite(initial_state).all()
        ):
            raise ValueError("initial recurrent state must be finite and floating")
        self.horizon = int(horizon)
        self.lanes = int(lanes)
        self.schema = schema
        self.action_spec = action_spec or ActionSpec()
        self.reward_scale = float(reward_scale)
        self.initial_state = initial_state.detach().cpu().clone()
        g = len(schema.global_features)
        f = len(schema.body_features)
        n = schema.capacity
        shape = (self.horizon, self.lanes)
        self.global_features = torch.zeros((*shape, g), dtype=torch.float32)
        self.body_features = torch.zeros((*shape, n, f), dtype=torch.float32)
        self.body_mask = torch.zeros((*shape, n), dtype=torch.bool)
        self.reset_before = torch.zeros(shape, dtype=torch.bool)
        self.action_kind = torch.zeros(shape, dtype=torch.long)
        self.action_wait_index = torch.zeros(shape, dtype=torch.long)
        self.action_xy = torch.zeros((*shape, 2), dtype=torch.float32)
        self.kind_mask = torch.ones((*shape, 3), dtype=torch.bool)
        self.wait_mask = torch.ones(
            (*shape, len(self.action_spec.wait_choices)), dtype=torch.bool
        )
        self.old_log_prob = torch.zeros(shape, dtype=torch.float32)
        self.old_kind_log_prob = torch.zeros(shape, dtype=torch.float32)
        self.old_wait_log_prob = torch.zeros(shape, dtype=torch.float32)
        self.old_coordinate_log_prob = torch.zeros(shape, dtype=torch.float32)
        self.old_values = torch.zeros(shape, dtype=torch.float32)
        self.raw_reward = torch.zeros(shape, dtype=torch.int64)
        self.optimizer_reward = torch.zeros(shape, dtype=torch.float32)
        self.elapsed_ticks = torch.zeros(shape, dtype=torch.int64)
        self.terminated = torch.zeros(shape, dtype=torch.bool)
        self.truncated = torch.zeros(shape, dtype=torch.bool)
        self.macro_interrupted = torch.zeros(shape, dtype=torch.bool)
        self.bootstrap_mask = torch.zeros(shape, dtype=torch.bool)
        self.trace_mask = torch.zeros(shape, dtype=torch.bool)
        self.train_mask = torch.zeros(shape, dtype=torch.bool)
        self.episode_id = torch.zeros(shape, dtype=torch.int64)
        self.seed = torch.zeros(shape, dtype=torch.int64)
        self.config_hash = torch.zeros(shape, dtype=torch.uint64)
        self.size = 0
        self._sealed = False
        self.advantage_result: AdvantageResult | None = None

    def append(
        self,
        observations: EncodedBatch,
        transitions: Sequence[MacroTransition],
        old_log_prob: Tensor,
        old_values: Tensor,
        *,
        old_log_prob_components: LogProbabilityComponents,
        reset_before: Tensor,
        kind_mask: Tensor | None = None,
        wait_mask: Tensor | None = None,
    ) -> None:
        if self._sealed:
            raise RuntimeError("cannot append to a sealed rollout")
        if self.size >= self.horizon:
            raise BufferError("recurrent rollout is full")
        if (
            observations.schema != self.schema
            or observations.global_features.shape[0] != self.lanes
        ):
            raise ValueError(
                "observation batch does not match rollout schema or lane count"
            )
        if len(transitions) != self.lanes:
            raise ValueError("transition lane count mismatch")
        for name, value, dtype in (
            ("old log probability", old_log_prob, torch.float32),
            ("old value", old_values, torch.float32),
            ("reset-before", reset_before, torch.bool),
        ):
            if value.shape != (self.lanes,) or value.dtype != dtype:
                raise ValueError(f"{name} must have the canonical lane shape and dtype")
        component_values = (
            old_log_prob_components.kind,
            old_log_prob_components.wait,
            old_log_prob_components.coordinates,
        )
        if any(
            value.shape != (self.lanes,) or value.dtype != torch.float32
            for value in component_values
        ):
            raise ValueError("old log-probability components must be float32 [B]")
        if any(value.device != old_log_prob.device for value in component_values):
            raise ValueError("old likelihood components and total must share a device")
        if old_log_prob.requires_grad or old_values.requires_grad:
            raise ValueError("old policy outputs must be detached")
        if any(value.requires_grad for value in component_values):
            raise ValueError("old likelihood components must be detached")
        if (
            not torch.isfinite(old_log_prob).all()
            or not torch.isfinite(old_values).all()
            or not all(torch.isfinite(value).all() for value in component_values)
        ):
            raise ValueError("old policy outputs must be finite")
        if not torch.allclose(
            old_log_prob_components.total, old_log_prob, rtol=0, atol=1e-6
        ):
            raise ValueError("old likelihood components do not sum to total")
        if any(
            transition.lane_id != lane for lane, transition in enumerate(transitions)
        ):
            raise ValueError("transitions are not in canonical lane order")
        prepared_kind_mask = torch.ones((self.lanes, 3), dtype=torch.bool)
        if kind_mask is not None:
            if kind_mask.shape != (self.lanes, 3) or kind_mask.dtype != torch.bool:
                raise ValueError("kind mask must have shape [B, 3]")
            prepared_kind_mask.copy_(kind_mask.detach().cpu())
        prepared_wait_mask = torch.ones(
            (self.lanes, len(self.action_spec.wait_choices)), dtype=torch.bool
        )
        if wait_mask is not None:
            expected = (self.lanes, len(self.action_spec.wait_choices))
            if wait_mask.shape != expected or wait_mask.dtype != torch.bool:
                raise ValueError("wait mask does not match wait support")
            prepared_wait_mask.copy_(wait_mask.detach().cpu())
        if not torch.all(prepared_kind_mask.any(dim=-1)):
            raise ValueError("every lane must allow at least one action kind")
        missing_wait = ~prepared_wait_mask.any(dim=-1)
        if torch.any(missing_wait & prepared_kind_mask[:, 0]):
            raise ValueError("every WAIT-enabled lane needs an allowed duration")

        index = self.size
        prepared: list[tuple[int, int, float, float, int, int, int, int]] = []
        for lane, transition in enumerate(transitions):
            expected_observation = transition.observation
            if expected_observation.schema != self.schema or any(
                not np.array_equal(left[lane : lane + 1], right)
                for left, right in (
                    (
                        observations.global_features,
                        expected_observation.global_features,
                    ),
                    (observations.body_features, expected_observation.body_features),
                    (observations.body_mask, expected_observation.body_mask),
                    (observations.source_tick, expected_observation.source_tick),
                    (observations.health_flags, expected_observation.health_flags),
                )
            ):
                raise ValueError("rollout observation does not match transition input")
            kind, wait_index, x, y = self.action_spec.encode(transition.action)
            if not prepared_kind_mask[lane, kind]:
                raise ValueError("collected action kind is disabled by its mask")
            if kind == 0 and not prepared_wait_mask[lane, wait_index]:
                raise ValueError("collected wait duration is disabled by its mask")
            integral_fields = {
                "raw reward": transition.raw_reward,
                "elapsed ticks": transition.elapsed_ticks,
                "episode id": transition.episode_id,
                "seed": transition.seed,
                "config hash": transition.diagnostics.config_hash,
            }
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in integral_fields.values()
            ):
                raise TypeError("transition integer fields must be canonical integers")
            if transition.elapsed_ticks <= 0:
                raise ValueError("semantic transitions must advance at least one tick")
            if not 0 <= transition.diagnostics.config_hash < 2**64:
                raise ValueError("config hash does not fit uint64 checkpoint storage")
            if not -(2**63) <= transition.raw_reward < 2**63:
                raise ValueError("raw reward does not fit int64 storage")
            if not all(
                -(2**63) <= value < 2**63
                for value in (
                    transition.elapsed_ticks,
                    transition.episode_id,
                    transition.seed,
                )
            ):
                raise ValueError("transition metadata does not fit int64 storage")
            if index > 0:
                expected_reset = bool(
                    self.terminated[index - 1, lane] or self.truncated[index - 1, lane]
                )
                if bool(reset_before[lane]) != expected_reset:
                    raise ValueError("reset-before does not match episode boundary")
                expected_episode = int(self.episode_id[index - 1, lane]) + int(
                    expected_reset
                )
                if transition.episode_id != expected_episode:
                    raise ValueError("episode id does not match transition continuity")
            prepared.append(
                (
                    kind,
                    wait_index,
                    x,
                    y,
                    transition.raw_reward,
                    transition.diagnostics.config_hash,
                    transition.episode_id,
                    transition.seed,
                )
            )

        # No validation that can fail occurs after this transaction boundary.
        self.global_features[index].copy_(
            torch.from_numpy(observations.global_features)
        )
        self.body_features[index].copy_(torch.from_numpy(observations.body_features))
        self.body_mask[index].copy_(torch.from_numpy(observations.body_mask))
        self.reset_before[index].copy_(reset_before.detach().cpu())
        self.old_log_prob[index].copy_(old_log_prob.detach().cpu())
        self.old_kind_log_prob[index].copy_(
            old_log_prob_components.kind.detach().cpu()
        )
        self.old_wait_log_prob[index].copy_(
            old_log_prob_components.wait.detach().cpu()
        )
        self.old_coordinate_log_prob[index].copy_(
            old_log_prob_components.coordinates.detach().cpu()
        )
        self.old_values[index].copy_(old_values.detach().cpu())
        self.kind_mask[index].copy_(prepared_kind_mask)
        self.wait_mask[index].copy_(prepared_wait_mask)
        for lane, transition in enumerate(transitions):
            kind, wait_index, x, y, raw_reward, config_hash, episode_id, seed = (
                prepared[lane]
            )
            self.action_kind[index, lane] = kind
            self.action_wait_index[index, lane] = wait_index
            self.action_xy[index, lane] = torch.tensor((x, y))
            self.raw_reward[index, lane] = raw_reward
            self.optimizer_reward[index, lane] = raw_reward / self.reward_scale
            self.elapsed_ticks[index, lane] = transition.elapsed_ticks
            self.terminated[index, lane] = transition.terminated
            self.truncated[index, lane] = transition.truncated
            self.macro_interrupted[index, lane] = transition.macro_interrupted
            self.bootstrap_mask[index, lane] = transition.bootstrap_mask
            self.trace_mask[index, lane] = transition.trace_mask
            self.train_mask[index, lane] = not (
                transition.truncated
                and transition.macro_interrupted
                and not transition.bootstrap_mask
            )
            self.episode_id[index, lane] = episode_id
            self.seed[index, lane] = seed
            self.config_hash[index, lane] = config_hash
        self.size += 1

    def finalize(
        self,
        bootstrap_values: Tensor,
        *,
        gamma_tick: float = 1.0,
        lambda_tick: float,
    ) -> RecurrentTrainingBatch:
        if self._sealed:
            raise RuntimeError("rollout is already sealed")
        if self.size <= 0:
            raise RuntimeError("cannot seal an empty rollout")
        if (
            bootstrap_values.shape != (self.size, self.lanes)
            or bootstrap_values.dtype != torch.float32
        ):
            raise ValueError("bootstrap values must be float32 [T, B]")
        valid = torch.ones((self.size, self.lanes), dtype=torch.bool)
        gae_valid = self.train_mask[: self.size]
        self.advantage_result = smdp_gae(
            self.optimizer_reward[: self.size],
            self.old_values[: self.size],
            bootstrap_values,
            self.elapsed_ticks[: self.size],
            self.bootstrap_mask[: self.size],
            self.trace_mask[: self.size],
            gae_valid,
            gamma_tick=gamma_tick,
            lambda_tick=lambda_tick,
        )
        self._sealed = True
        return RecurrentTrainingBatch(
            self.global_features[: self.size],
            self.body_features[: self.size],
            self.body_mask[: self.size],
            self.reset_before[: self.size],
            self.initial_state,
            ActionTensor(
                self.action_kind[: self.size],
                self.action_wait_index[: self.size],
                self.action_xy[: self.size],
            ),
            self.old_log_prob[: self.size],
            self.old_kind_log_prob[: self.size],
            self.old_wait_log_prob[: self.size],
            self.old_coordinate_log_prob[: self.size],
            self.old_values[: self.size],
            self.advantage_result.advantages,
            self.advantage_result.returns,
            valid,
            self.train_mask[: self.size],
            self.kind_mask[: self.size],
            self.wait_mask[: self.size],
        )
