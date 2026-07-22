"""Project-owned recurrent PPO update with explicit conditional action math."""

from __future__ import annotations

import math
import copy
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .models import RecurrentActorCritic
from .torch_distribution import ActionTensor, TorchConditionalActionDistribution


@dataclass(frozen=True, slots=True)
class PPOConfig:
    learning_rate: float = 3e-4
    final_learning_rate_fraction: float = 0.1
    epochs: int = 4
    lane_minibatch_size: int = 8
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    max_gradient_norm: float = 0.5
    target_kl: float = 0.03

    def __post_init__(self) -> None:
        positive = (
            self.learning_rate,
            self.final_learning_rate_fraction,
            self.clip_ratio,
            self.value_clip,
            self.value_coefficient,
            self.max_gradient_norm,
            self.target_kl,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("positive PPO hyperparameters must be finite")
        if not 0 < self.final_learning_rate_fraction <= 1:
            raise ValueError("final learning-rate fraction must be in (0, 1]")
        if not 0 < self.clip_ratio < 1:
            raise ValueError("PPO clip ratio must be in (0, 1)")
        if not math.isfinite(self.entropy_coefficient) or self.entropy_coefficient < 0:
            raise ValueError("entropy coefficient must be finite and nonnegative")
        if (
            isinstance(self.epochs, bool)
            or not isinstance(self.epochs, int)
            or self.epochs <= 0
        ):
            raise ValueError("PPO epochs must be a positive integer")
        if (
            isinstance(self.lane_minibatch_size, bool)
            or not isinstance(self.lane_minibatch_size, int)
            or self.lane_minibatch_size <= 0
        ):
            raise ValueError("lane minibatch size must be a positive integer")

    def manifest(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RecurrentTrainingBatch:
    global_features: Tensor
    body_features: Tensor
    body_mask: Tensor
    reset_before: Tensor
    initial_state: Tensor
    actions: ActionTensor
    old_log_prob: Tensor
    old_values: Tensor
    advantages: Tensor
    returns: Tensor
    valid: Tensor
    train_mask: Tensor
    kind_mask: Tensor
    wait_mask: Tensor

    def validate(self, model: RecurrentActorCritic) -> tuple[int, int]:
        if self.global_features.ndim != 3:
            raise ValueError("training observations must be time-major [T, B, ...]")
        time, batch, _ = self.global_features.shape
        scalar_shape = (time, batch)
        for value in (
            self.old_log_prob,
            self.old_values,
            self.advantages,
            self.returns,
        ):
            if value.shape != scalar_shape or not value.is_floating_point():
                raise ValueError("training scalars must be floating [T, B] tensors")
        if any(
            value.shape != scalar_shape or value.dtype != torch.bool
            for value in (self.valid, self.train_mask, self.reset_before)
        ):
            raise ValueError("valid, train, and reset masks must be boolean [T, B]")
        if torch.any(self.train_mask & ~self.valid):
            raise ValueError("loss-bearing decisions must also be valid")
        if not torch.any(self.train_mask):
            raise ValueError("training batch contains no valid decisions")
        self.actions.validate(torch.Size(scalar_shape))
        if (
            self.kind_mask.shape != (*scalar_shape, 3)
            or self.kind_mask.dtype != torch.bool
        ):
            raise ValueError("kind mask must be boolean [T, B, 3]")
        waits = len(model.action_spec.wait_choices)
        if (
            self.wait_mask.shape != (*scalar_shape, waits)
            or self.wait_mask.dtype != torch.bool
        ):
            raise ValueError("wait mask does not match the action specification")
        tensors = (
            self.global_features,
            self.body_features,
            self.body_mask,
            self.reset_before,
            self.initial_state,
            self.actions.kind,
            self.actions.wait_index,
            self.actions.xy,
            self.old_log_prob,
            self.old_values,
            self.advantages,
            self.returns,
            self.valid,
            self.train_mask,
            self.kind_mask,
            self.wait_mask,
        )
        if len({value.device for value in tensors}) != 1:
            raise ValueError("all recurrent training tensors must share one device")
        if any(value.requires_grad for value in tensors):
            raise ValueError(
                "stored rollout tensors must be detached from collection graphs"
            )
        parameter = next(model.parameters())
        for value in (
            self.global_features,
            self.body_features,
            self.initial_state,
            self.actions.xy,
            self.old_log_prob,
            self.old_values,
            self.advantages,
            self.returns,
        ):
            if value.dtype != parameter.dtype:
                raise TypeError("floating training tensors must match the model dtype")
        if not all(
            torch.isfinite(value).all()
            for value in (
                self.global_features,
                self.body_features,
                self.old_log_prob,
                self.old_values,
                self.advantages,
                self.returns,
                self.initial_state,
            )
        ):
            raise ValueError("training batch contains nonfinite values")
        model(
            self.global_features,
            self.body_features,
            self.body_mask,
            self.initial_state,
            reset_before=self.reset_before,
        )
        return time, batch


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    learning_rate: float
    optimizer_steps: int
    early_stopped: bool


class LinearLearningRate:
    version = "linear-learning-rate-v1"

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        initial: float,
        final_fraction: float,
        total_updates: int,
    ) -> None:
        if (
            isinstance(total_updates, bool)
            or not isinstance(total_updates, int)
            or total_updates <= 0
        ):
            raise ValueError("total updates must be a positive integer")
        self.optimizer = optimizer
        self.initial = float(initial)
        self.final_fraction = float(final_fraction)
        self.total_updates = total_updates
        self.completed_updates = 0
        self._apply()

    @property
    def learning_rate(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _apply(self) -> None:
        progress = min(self.completed_updates / max(self.total_updates - 1, 1), 1.0)
        fraction = 1.0 - progress * (1.0 - self.final_fraction)
        value = self.initial * fraction
        for group in self.optimizer.param_groups:
            group["lr"] = value

    def step(self) -> None:
        if self.completed_updates >= self.total_updates:
            raise RuntimeError("learning-rate schedule exceeded its declared budget")
        self.completed_updates += 1
        self._apply()

    def state_dict(self) -> dict[str, int | float | str]:
        return {
            "version": self.version,
            "initial": self.initial,
            "final_fraction": self.final_fraction,
            "total_updates": self.total_updates,
            "completed_updates": self.completed_updates,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = {
            "version",
            "initial",
            "final_fraction",
            "total_updates",
            "completed_updates",
        }
        if set(state) != expected or state["version"] != self.version:
            raise ValueError("learning-rate schedule identity mismatch")
        if (
            float(state["initial"]) != self.initial
            or float(state["final_fraction"]) != self.final_fraction
            or int(state["total_updates"]) != self.total_updates
        ):
            raise ValueError("learning-rate schedule configuration mismatch")
        completed = state["completed_updates"]
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or not 0 <= completed <= self.total_updates
        ):
            raise ValueError("invalid completed update count")
        self.completed_updates = completed
        self._apply()


class PPOTrainer:
    """Recurrent PPO optimizer that minibatches complete lane sequences."""

    def __init__(
        self,
        model: RecurrentActorCritic,
        *,
        config: PPOConfig | None = None,
        total_updates: int,
        sampler_seed: int,
    ) -> None:
        self.model = model
        self.config = config or PPOConfig()
        if (
            isinstance(sampler_seed, bool)
            or not isinstance(sampler_seed, int)
            or sampler_seed < 0
        ):
            raise ValueError("sampler seed must be a nonnegative integer")
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
            foreach=False,
        )
        self.schedule = LinearLearningRate(
            self.optimizer,
            initial=self.config.learning_rate,
            final_fraction=self.config.final_learning_rate_fraction,
            total_updates=total_updates,
        )
        self.sampler = torch.Generator(device="cpu")
        self.sampler.manual_seed(sampler_seed)

    def update(self, batch: RecurrentTrainingBatch) -> PPOUpdateStats:
        _, lane_count = batch.validate(self.model)
        self.verify_batch_policy(batch)
        valid_advantages = batch.advantages[batch.train_mask]
        advantage_mean = valid_advantages.mean()
        advantage_std = valid_advantages.std(unbiased=False).clamp_min(1e-8)
        normalized_advantages = (batch.advantages - advantage_mean) / advantage_std
        records: list[tuple[tuple[float, float, float, float, float, float], int]] = []
        early_stopped = False
        learning_rate_used = self.schedule.learning_rate
        for _ in range(self.config.epochs):
            order = torch.randperm(lane_count, generator=self.sampler)
            for begin in range(0, lane_count, self.config.lane_minibatch_size):
                lanes = order[begin : begin + self.config.lane_minibatch_size].to(
                    batch.global_features.device
                )
                train_mask = batch.train_mask[:, lanes]
                if not torch.any(train_mask):
                    continue
                output = self.model(
                    batch.global_features[:, lanes],
                    batch.body_features[:, lanes],
                    batch.body_mask[:, lanes],
                    batch.initial_state[:, lanes],
                    reset_before=batch.reset_before[:, lanes],
                )
                distribution = TorchConditionalActionDistribution(
                    output.kind_logits,
                    output.wait_logits,
                    output.coordinate_alpha,
                    output.coordinate_beta,
                    spec=self.model.action_spec,
                    kind_mask=batch.kind_mask[:, lanes],
                    wait_mask=batch.wait_mask[:, lanes],
                )
                actions = ActionTensor(
                    batch.actions.kind[:, lanes],
                    batch.actions.wait_index[:, lanes],
                    batch.actions.xy[:, lanes],
                )
                new_log_prob = distribution.log_prob(actions)
                entropy_values = distribution.entropy()
                old_log_prob = batch.old_log_prob[:, lanes]
                log_ratio = torch.where(train_mask, new_log_prob - old_log_prob, 0.0)
                ratio = torch.exp(log_ratio)
                selected = (
                    new_log_prob[train_mask],
                    entropy_values[train_mask],
                    output.values[train_mask],
                    ratio[train_mask],
                )
                if not all(torch.isfinite(value).all() for value in selected):
                    raise FloatingPointError("nonfinite PPO model output")
                entropy = entropy_values[train_mask].mean()
                advantages = normalized_advantages[:, lanes]
                unclipped = ratio * advantages
                clipped = (
                    ratio.clamp(
                        1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio
                    )
                    * advantages
                )
                policy_loss = -torch.minimum(unclipped, clipped)[train_mask].mean()

                old_values = batch.old_values[:, lanes]
                returns = batch.returns[:, lanes]
                value_error = (output.values - returns).square()
                clipped_values = old_values + (output.values - old_values).clamp(
                    -self.config.value_clip, self.config.value_clip
                )
                clipped_error = (clipped_values - returns).square()
                value_loss = (
                    0.5 * torch.maximum(value_error, clipped_error)[train_mask].mean()
                )
                loss = (
                    policy_loss
                    + self.config.value_coefficient * value_loss
                    - self.config.entropy_coefficient * entropy
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_gradient_norm
                )
                if not torch.isfinite(gradient_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    raise FloatingPointError("nonfinite PPO gradient norm")
                self.optimizer.step()

                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio)[train_mask].mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_ratio)[train_mask]
                        .float()
                        .mean()
                    )
                records.append(
                    (
                        (
                            float(policy_loss.detach()),
                            float(value_loss.detach()),
                            float(entropy.detach()),
                            float(approximate_kl.detach()),
                            float(clip_fraction.detach()),
                            float(gradient_norm.detach()),
                        ),
                        int(train_mask.sum()),
                    )
                )
                if approximate_kl > self.config.target_kl:
                    early_stopped = True
                    break
            if early_stopped:
                break
        if not records:
            raise RuntimeError("PPO update produced no optimizer steps")
        self.schedule.step()
        total_weight = sum(weight for _, weight in records)
        means = [
            sum(row[index] * weight for row, weight in records) / total_weight
            for index in range(6)
        ]
        return PPOUpdateStats(
            *means,
            learning_rate_used,
            len(records),
            early_stopped,
        )

    @torch.no_grad()
    def verify_batch_policy(
        self, batch: RecurrentTrainingBatch, *, tolerance: float = 2e-5
    ) -> None:
        output = self.model(
            batch.global_features,
            batch.body_features,
            batch.body_mask,
            batch.initial_state,
            reset_before=batch.reset_before,
        )
        distribution = TorchConditionalActionDistribution(
            output.kind_logits,
            output.wait_logits,
            output.coordinate_alpha,
            output.coordinate_beta,
            spec=self.model.action_spec,
            kind_mask=batch.kind_mask,
            wait_mask=batch.wait_mask,
        )
        log_prob = distribution.log_prob(batch.actions)
        if not torch.allclose(
            log_prob[batch.train_mask],
            batch.old_log_prob[batch.train_mask],
            rtol=0,
            atol=tolerance,
        ):
            raise ValueError(
                "stored action likelihoods do not match the collection policy"
            )
        if not torch.allclose(
            output.values[batch.train_mask],
            batch.old_values[batch.train_mask],
            rtol=0,
            atol=tolerance,
        ):
            raise ValueError("stored values do not match the collection policy")

    def state_dict(self) -> dict[str, object]:
        return {
            "version": "ppo-trainer-v1",
            "model_manifest": self.model.manifest(),
            "config": self.config.manifest(),
            "optimizer": self.optimizer.state_dict(),
            "schedule": self.schedule.state_dict(),
            "sampler_state": self.sampler.get_state(),
        }

    def _validate_optimizer_state(self) -> None:
        for parameter, values in self.optimizer.state.items():
            for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                value = values.get(name)
                if value is None:
                    continue
                if (
                    not isinstance(value, Tensor)
                    or value.shape != parameter.shape
                    or value.dtype != parameter.dtype
                    or value.device != parameter.device
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError(
                        f"optimizer {name} is incompatible with its parameter"
                    )
            step = values.get("step")
            if step is not None and (
                not isinstance(step, Tensor)
                or step.numel() != 1
                or not torch.isfinite(step).all()
            ):
                raise ValueError("optimizer step state is malformed")

    def load_state_dict(self, state: dict[str, object]) -> None:
        expected = {
            "version",
            "model_manifest",
            "config",
            "optimizer",
            "schedule",
            "sampler_state",
        }
        if set(state) != expected or state["version"] != "ppo-trainer-v1":
            raise ValueError("PPO trainer state identity mismatch")
        if (
            state["model_manifest"] != self.model.manifest()
            or state["config"] != self.config.manifest()
        ):
            raise ValueError("PPO trainer configuration mismatch")
        previous_optimizer = copy.deepcopy(self.optimizer.state_dict())
        previous_schedule = self.schedule.state_dict()
        previous_sampler = self.sampler.get_state()
        try:
            self.optimizer.load_state_dict(state["optimizer"])
            self._validate_optimizer_state()
            self.schedule.load_state_dict(state["schedule"])
            self.sampler.set_state(state["sampler_state"])
        except BaseException:
            self.optimizer.load_state_dict(previous_optimizer)
            self.schedule.load_state_dict(previous_schedule)
            self.sampler.set_state(previous_sampler)
            raise
