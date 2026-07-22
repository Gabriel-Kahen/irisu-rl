"""Duration-aware semi-Markov return and advantage calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class AdvantageResult:
    advantages: Tensor
    returns: Tensor
    deltas: Tensor


def lambda_tick_from_half_life(
    half_life_seconds: float, *, tick_seconds: float = 0.02
) -> float:
    """Convert a trace half-life into one multiplicative factor per game tick."""

    if not math.isfinite(half_life_seconds) or half_life_seconds <= 0:
        raise ValueError("trace half-life must be finite and positive")
    if not math.isfinite(tick_seconds) or tick_seconds <= 0:
        raise ValueError("tick duration must be finite and positive")
    return 2.0 ** (-tick_seconds / half_life_seconds)


def smdp_gae(
    rewards: Tensor,
    values: Tensor,
    bootstrap_values: Tensor,
    elapsed_ticks: Tensor,
    bootstrap_mask: Tensor,
    trace_mask: Tensor,
    valid_mask: Tensor,
    *,
    gamma_tick: float = 1.0,
    lambda_tick: float,
    reward_is_event_discounted: bool = False,
) -> AdvantageResult:
    """Compute SMDP GAE for time-major tensors shaped ``[T, B]``.

    ``bootstrap_values`` must already select the correct observation: retained
    final observation for neutral truncation, ordinary next observation for a
    live transition, and any finite placeholder when ``bootstrap_mask`` is
    false. ``trace_mask`` must stop at both termination and truncation.
    """

    tensors = (
        rewards,
        values,
        bootstrap_values,
        elapsed_ticks,
        bootstrap_mask,
        trace_mask,
        valid_mask,
    )
    shape = rewards.shape
    if (
        rewards.ndim != 2
        or not all(shape)
        or any(value.shape != shape for value in tensors)
    ):
        raise ValueError("SMDP inputs must share a time-major [T, B] shape")
    if not all(
        value.is_floating_point() for value in (rewards, values, bootstrap_values)
    ):
        raise TypeError("reward and value tensors must be floating point")
    if elapsed_ticks.dtype not in (torch.int32, torch.int64):
        raise TypeError("elapsed ticks must be int32 or int64")
    if any(
        value.dtype != torch.bool for value in (bootstrap_mask, trace_mask, valid_mask)
    ):
        raise TypeError("bootstrap, trace, and valid masks must be boolean")
    if len({value.device for value in tensors}) != 1:
        raise ValueError("SMDP inputs must be on one device")
    if rewards.dtype != values.dtype or rewards.dtype != bootstrap_values.dtype:
        raise TypeError("reward and value tensors must share one dtype")
    if torch.any(elapsed_ticks[valid_mask] <= 0):
        raise ValueError("every semantic action must advance at least one tick")
    if torch.any(trace_mask & ~bootstrap_mask):
        raise ValueError("a continuing trace must also permit value bootstrap")
    if torch.any((bootstrap_mask | trace_mask) & ~valid_mask):
        raise ValueError("padded decisions cannot bootstrap or continue a trace")
    if not all(
        torch.isfinite(value).all() for value in (rewards, values, bootstrap_values)
    ):
        raise ValueError("reward and value tensors must be finite")
    if not math.isfinite(gamma_tick) or not 0 < gamma_tick <= 1:
        raise ValueError("gamma_tick must be in (0, 1]")
    if gamma_tick != 1.0 and not reward_is_event_discounted:
        raise ValueError(
            "gamma_tick below 1 requires per-event discounted rewards, which R2 does not store"
        )
    if not math.isfinite(lambda_tick) or not 0 < lambda_tick <= 1:
        raise ValueError("lambda_tick must be in (0, 1]")

    duration = elapsed_ticks.to(dtype=rewards.dtype)
    discount = torch.pow(
        torch.as_tensor(gamma_tick, dtype=rewards.dtype, device=rewards.device),
        duration,
    )
    trace_discount = torch.pow(
        torch.as_tensor(
            gamma_tick * lambda_tick, dtype=rewards.dtype, device=rewards.device
        ),
        duration,
    )
    deltas = torch.where(
        valid_mask,
        rewards + bootstrap_mask * discount * bootstrap_values - values,
        0.0,
    )
    advantages = torch.zeros_like(rewards)
    accumulator = torch.zeros(shape[1], dtype=rewards.dtype, device=rewards.device)
    for index in range(shape[0] - 1, -1, -1):
        accumulator = torch.where(
            valid_mask[index],
            deltas[index] + trace_mask[index] * trace_discount[index] * accumulator,
            0.0,
        )
        advantages[index] = accumulator
    returns = torch.where(valid_mask, advantages + values, 0.0)
    return AdvantageResult(advantages, returns, deltas)
