"""Differentiable conditional action distribution for recurrent PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.distributions import Beta, Categorical

from .actions import ActionSpec


@dataclass(frozen=True, slots=True)
class ActionTensor:
    """Canonical tensor action batch with arbitrary leading dimensions."""

    kind: Tensor
    wait_index: Tensor
    xy: Tensor

    def validate(self, leading_shape: torch.Size) -> None:
        if self.kind.shape != leading_shape or self.kind.dtype != torch.long:
            raise ValueError("action kind must be int64 with the distribution shape")
        if (
            self.wait_index.shape != leading_shape
            or self.wait_index.dtype != torch.long
        ):
            raise ValueError("wait index must be int64 with the distribution shape")
        if self.xy.shape != (*leading_shape, 2) or not self.xy.is_floating_point():
            raise ValueError(
                "action coordinates must be floating with a final size-2 axis"
            )
        if torch.any((self.kind < 0) | (self.kind > 2)):
            raise ValueError("action kind is outside [0, 2]")


@dataclass(frozen=True, slots=True)
class LogProbabilityComponents:
    kind: Tensor
    wait: Tensor
    coordinates: Tensor

    @property
    def total(self) -> Tensor:
        return self.kind + self.wait + self.coordinates


@dataclass(frozen=True, slots=True)
class EntropyComponents:
    kind: Tensor
    wait: Tensor
    coordinates: Tensor


class TorchConditionalActionDistribution:
    """Masked WAIT/WEAK/STRONG distribution with conditional likelihoods.

    Coordinate concentration tensors end in ``[2 shot kinds, 2 coordinates]``.
    Inactive branches make no contribution to selected-action likelihood.
    Analytic entropy is the kind-probability-weighted expectation over branches.
    """

    def __init__(
        self,
        kind_logits: Tensor,
        wait_logits: Tensor,
        coordinate_alpha: Tensor,
        coordinate_beta: Tensor,
        *,
        spec: ActionSpec | None = None,
        kind_mask: Tensor | None = None,
        wait_mask: Tensor | None = None,
    ) -> None:
        self.spec = spec or ActionSpec()
        if kind_logits.ndim < 2 or kind_logits.shape[-1] != 3:
            raise ValueError("kind logits must end in size 3")
        self.leading_shape = kind_logits.shape[:-1]
        if wait_logits.shape != (*self.leading_shape, len(self.spec.wait_choices)):
            raise ValueError("wait logits do not match the action specification")
        expected_coordinates = (*self.leading_shape, 2, 2)
        if (
            coordinate_alpha.shape != expected_coordinates
            or coordinate_beta.shape != expected_coordinates
        ):
            raise ValueError("coordinate concentrations must end in [2, 2]")
        tensors = (kind_logits, wait_logits, coordinate_alpha, coordinate_beta)
        if not all(value.is_floating_point() for value in tensors):
            raise TypeError("distribution parameters must be floating tensors")
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("distribution parameters must be finite")
        if torch.any(coordinate_alpha <= 0) or torch.any(coordinate_beta <= 0):
            raise ValueError("Beta concentrations must be positive")
        if (
            len({value.device for value in tensors}) != 1
            or len({value.dtype for value in tensors}) != 1
        ):
            raise ValueError("distribution parameters must share one device and dtype")

        self.kind_logits = kind_logits
        self.wait_logits = wait_logits
        self.alpha = coordinate_alpha
        self.beta = coordinate_beta
        self.kind_mask = self._mask(kind_mask, kind_logits, "kind")
        self.wait_mask = self._mask(wait_mask, wait_logits, "wait")
        if torch.any(~self.kind_mask.any(dim=-1)):
            raise ValueError("all-masked action kind")
        missing_wait = ~self.wait_mask.any(dim=-1)
        if torch.any(missing_wait & self.kind_mask[..., 0]):
            raise ValueError("all-masked active wait branch")
        effective_wait_mask = self.wait_mask.clone()
        effective_wait_mask[..., 0] |= missing_wait
        floor = torch.finfo(kind_logits.dtype).min
        self._kind = Categorical(logits=kind_logits.masked_fill(~self.kind_mask, floor))
        self._wait = Categorical(
            logits=wait_logits.masked_fill(~effective_wait_mask, floor)
        )
        self._coordinates = Beta(self.alpha, self.beta)

    @staticmethod
    def _mask(mask: Tensor | None, reference: Tensor, name: str) -> Tensor:
        if mask is None:
            return torch.ones_like(reference, dtype=torch.bool)
        if mask.shape != reference.shape or mask.dtype != torch.bool:
            raise ValueError(f"{name} mask shape or dtype mismatch")
        return mask.clone()

    def log_prob_components(self, actions: ActionTensor) -> LogProbabilityComponents:
        actions.validate(self.leading_shape)
        wait_count = len(self.spec.wait_choices)
        if torch.any((actions.wait_index < 0) | (actions.wait_index >= wait_count)):
            raise ValueError("wait index is outside the declared support")
        if not torch.isfinite(actions.xy).all() or torch.any(
            (actions.xy < 0) | (actions.xy > 1)
        ):
            raise ValueError("coordinates must be finite and within [0, 1]")
        if torch.any(
            ~self.kind_mask.gather(-1, actions.kind.unsqueeze(-1)).squeeze(-1)
        ):
            raise ValueError("action selects a masked kind")

        kind_log_prob = self._kind.log_prob(actions.kind)
        is_wait = actions.kind == 0
        selected_wait_allowed = self.wait_mask.gather(
            -1, actions.wait_index.unsqueeze(-1)
        ).squeeze(-1)
        if torch.any(is_wait & ~selected_wait_allowed):
            raise ValueError("action selects a masked wait duration")
        wait_log_prob = torch.where(
            is_wait, self._wait.log_prob(actions.wait_index), 0.0
        )

        branch = (actions.kind - 1).clamp(0, 1)
        gather_index = branch[..., None, None].expand(*self.leading_shape, 1, 2)
        alpha = self.alpha.gather(-2, gather_index).squeeze(-2)
        beta = self.beta.gather(-2, gather_index).squeeze(-2)
        epsilon = self.spec.coordinate_log_prob_epsilon
        xy = actions.xy.clamp(epsilon, 1.0 - epsilon)
        coordinate_log_prob = Beta(alpha, beta).log_prob(xy).sum(dim=-1)
        coordinate_log_prob = torch.where(is_wait, 0.0, coordinate_log_prob)
        return LogProbabilityComponents(
            kind_log_prob, wait_log_prob, coordinate_log_prob
        )

    def log_prob(self, actions: ActionTensor) -> Tensor:
        return self.log_prob_components(actions).total

    def entropy(self) -> Tensor:
        components = self.entropy_components()
        kind_probability = self._kind.probs
        return (
            components.kind
            + kind_probability[..., 0] * components.wait
            + kind_probability[..., 1] * components.coordinates[..., 0]
            + kind_probability[..., 2] * components.coordinates[..., 1]
        )

    def entropy_components(self) -> EntropyComponents:
        return EntropyComponents(
            self._kind.entropy(),
            self._wait.entropy(),
            self._coordinates.entropy().sum(dim=-1),
        )

    def sample(self) -> ActionTensor:
        kind = self._kind.sample()
        flat_kind = kind.reshape(-1)
        flat_wait = torch.zeros_like(flat_kind)
        flat_xy = torch.zeros(
            (flat_kind.numel(), 2), dtype=self.alpha.dtype, device=self.alpha.device
        )
        wait_rows = flat_kind == 0
        if torch.any(wait_rows):
            probabilities = self._wait.probs.reshape(-1, self._wait.probs.shape[-1])
            flat_wait[wait_rows] = torch.multinomial(
                probabilities[wait_rows], 1
            ).squeeze(-1)
        flat_alpha = self.alpha.reshape(-1, 2, 2)
        flat_beta = self.beta.reshape(-1, 2, 2)
        for action_kind in (1, 2):
            rows = flat_kind == action_kind
            if torch.any(rows):
                branch = action_kind - 1
                flat_xy[rows] = Beta(
                    flat_alpha[rows, branch], flat_beta[rows, branch]
                ).sample()
        wait_index = flat_wait.reshape(self.leading_shape)
        xy = flat_xy.reshape(*self.leading_shape, 2)
        return ActionTensor(kind, wait_index, xy)

    def deterministic(self) -> ActionTensor:
        kind = self._kind.logits.argmax(dim=-1)
        wait_index = self._wait.logits.argmax(dim=-1)
        means = self.alpha / (self.alpha + self.beta)
        branch = (kind - 1).clamp(0, 1)
        index = branch[..., None, None].expand(*self.leading_shape, 1, 2)
        xy = means.gather(-2, index).squeeze(-2)
        is_wait = kind == 0
        wait_index = torch.where(is_wait, wait_index, 0)
        xy = torch.where(is_wait.unsqueeze(-1), 0.0, xy)
        return ActionTensor(kind, wait_index, xy)
