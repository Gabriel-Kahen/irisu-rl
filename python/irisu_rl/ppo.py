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
    old_kind_log_prob: Tensor
    old_wait_log_prob: Tensor
    old_coordinate_log_prob: Tensor
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
        time, batch, global_count = self.global_features.shape
        if time <= 0 or batch <= 0:
            raise ValueError("training sequence dimensions must be nonzero")
        expected_body = (
            time,
            batch,
            model.schema.capacity,
            len(model.schema.body_features),
        )
        expected_state = (
            model.config.recurrent_layers,
            batch,
            model.config.recurrent_hidden,
        )
        if (
            global_count != len(model.schema.global_features)
            or self.body_features.shape != expected_body
            or self.body_mask.shape != expected_body[:-1]
            or self.body_mask.dtype != torch.bool
        ):
            raise ValueError("training observations do not match the model schema")
        if self.initial_state.shape != expected_state:
            raise ValueError("initial recurrent state does not match the model")
        scalar_shape = (time, batch)
        for value in (
            self.old_log_prob,
            self.old_kind_log_prob,
            self.old_wait_log_prob,
            self.old_coordinate_log_prob,
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
            self.old_kind_log_prob,
            self.old_wait_log_prob,
            self.old_coordinate_log_prob,
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
        if self.global_features.device != parameter.device:
            raise ValueError("training tensors and model must share one device")
        for value in (
            self.global_features,
            self.body_features,
            self.initial_state,
            self.actions.xy,
            self.old_log_prob,
            self.old_kind_log_prob,
            self.old_wait_log_prob,
            self.old_coordinate_log_prob,
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
                self.old_kind_log_prob,
                self.old_wait_log_prob,
                self.old_coordinate_log_prob,
                self.old_values,
                self.advantages,
                self.returns,
                self.initial_state,
            )
        ):
            raise ValueError("training batch contains nonfinite values")
        if not torch.allclose(
            self.old_kind_log_prob
            + self.old_wait_log_prob
            + self.old_coordinate_log_prob,
            self.old_log_prob,
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError("old likelihood components do not sum to total")
        return time, batch


@dataclass(frozen=True, slots=True)
class PPOUpdateStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approximate_kl: float
    kind_approximate_kl: float
    wait_approximate_kl: float
    coordinate_approximate_kl: float
    clip_fraction: float
    gradient_norm: float
    kind_entropy: float
    wait_entropy: float
    coordinate_entropy: float
    learning_rate: float
    optimizer_steps: int
    early_stopped: bool


def clipped_surrogate_loss(
    ratio: Tensor, advantages: Tensor, train_mask: Tensor, clip_ratio: float
) -> Tensor:
    """Return the masked PPO clipped-surrogate loss for audited fixtures."""

    if (
        ratio.shape != advantages.shape
        or train_mask.shape != ratio.shape
        or train_mask.dtype != torch.bool
        or not torch.any(train_mask)
    ):
        raise ValueError("surrogate inputs need one nonempty shared mask")
    if (
        not math.isfinite(clip_ratio)
        or not 0 < clip_ratio < 1
        or not torch.isfinite(ratio[train_mask]).all()
        or not torch.isfinite(advantages[train_mask]).all()
        or torch.any(ratio[train_mask] <= 0)
    ):
        raise ValueError("surrogate ratio, advantages, or clip ratio is invalid")
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
    return -torch.minimum(unclipped, clipped)[train_mask].mean()


def _masked_metric(value: Tensor, mask: Tensor) -> tuple[float, int]:
    count = int(mask.sum())
    return (float(value[mask].mean().detach()), count) if count else (0.0, 0)


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
        if self.schedule.completed_updates >= self.schedule.total_updates:
            raise RuntimeError("PPO update budget is exhausted")
        _, lane_count = batch.validate(self.model)
        self.verify_batch_policy(batch)
        valid_advantages = batch.advantages[batch.train_mask]
        advantage_mean = valid_advantages.mean()
        advantage_std = valid_advantages.std(unbiased=False).clamp_min(1e-8)
        normalized_advantages = (batch.advantages - advantage_mean) / advantage_std
        records: list[dict[str, tuple[float, int]]] = []
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
                new_components = distribution.log_prob_components(actions)
                new_log_prob = new_components.total
                entropy_values = distribution.entropy()
                entropy_components = distribution.entropy_components()
                old_log_prob = batch.old_log_prob[:, lanes]
                log_ratio = torch.where(train_mask, new_log_prob - old_log_prob, 0.0)
                ratio = torch.exp(log_ratio)
                is_wait = actions.kind == 0
                wait_train_mask = train_mask & is_wait
                coordinate_train_mask = train_mask & ~is_wait
                branch = (actions.kind - 1).clamp(0, 1)
                coordinate_entropy = entropy_components.coordinates.gather(
                    -1, branch.unsqueeze(-1)
                ).squeeze(-1)
                selected = (
                    new_log_prob[train_mask],
                    new_components.kind[train_mask],
                    new_components.wait[wait_train_mask],
                    new_components.coordinates[coordinate_train_mask],
                    entropy_values[train_mask],
                    entropy_components.kind[train_mask],
                    entropy_components.wait[wait_train_mask],
                    coordinate_entropy[coordinate_train_mask],
                    output.values[train_mask],
                    ratio[train_mask],
                )
                if not all(torch.isfinite(value).all() for value in selected):
                    raise FloatingPointError("nonfinite PPO model output")
                entropy = entropy_values[train_mask].mean()
                advantages = normalized_advantages[:, lanes]
                policy_loss = clipped_surrogate_loss(
                    ratio, advantages, train_mask, self.config.clip_ratio
                )

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
                    kind_log_ratio = torch.where(
                        train_mask,
                        new_components.kind - batch.old_kind_log_prob[:, lanes],
                        0.0,
                    )
                    wait_log_ratio = torch.where(
                        wait_train_mask,
                        new_components.wait - batch.old_wait_log_prob[:, lanes],
                        0.0,
                    )
                    coordinate_log_ratio = torch.where(
                        coordinate_train_mask,
                        new_components.coordinates
                        - batch.old_coordinate_log_prob[:, lanes],
                        0.0,
                    )
                    kind_kl = torch.exp(kind_log_ratio) - 1.0 - kind_log_ratio
                    wait_kl = torch.exp(wait_log_ratio) - 1.0 - wait_log_ratio
                    coordinate_kl = (
                        torch.exp(coordinate_log_ratio)
                        - 1.0
                        - coordinate_log_ratio
                    )
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.config.clip_ratio)[train_mask]
                        .float()
                        .mean()
                    )
                train_count = int(train_mask.sum())
                records.append(
                    {
                        "policy_loss": (float(policy_loss.detach()), train_count),
                        "value_loss": (float(value_loss.detach()), train_count),
                        "entropy": (float(entropy.detach()), train_count),
                        "approximate_kl": (
                            float(approximate_kl.detach()),
                            train_count,
                        ),
                        "kind_approximate_kl": _masked_metric(kind_kl, train_mask),
                        "wait_approximate_kl": _masked_metric(
                            wait_kl, wait_train_mask
                        ),
                        "coordinate_approximate_kl": _masked_metric(
                            coordinate_kl, coordinate_train_mask
                        ),
                        "clip_fraction": (
                            float(clip_fraction.detach()),
                            train_count,
                        ),
                        "gradient_norm": (float(gradient_norm.detach()), train_count),
                        "kind_entropy": _masked_metric(
                            entropy_components.kind, train_mask
                        ),
                        "wait_entropy": _masked_metric(
                            entropy_components.wait, wait_train_mask
                        ),
                        "coordinate_entropy": _masked_metric(
                            coordinate_entropy, coordinate_train_mask
                        ),
                    }
                )
                if approximate_kl > self.config.target_kl:
                    early_stopped = True
                    break
            if early_stopped:
                break
        if not records:
            raise RuntimeError("PPO update produced no optimizer steps")
        self.schedule.step()
        def weighted_mean(name: str) -> float:
            total_weight = sum(record[name][1] for record in records)
            return (
                sum(value * weight for value, weight in (r[name] for r in records))
                / total_weight
                if total_weight
                else 0.0
            )

        return PPOUpdateStats(
            policy_loss=weighted_mean("policy_loss"),
            value_loss=weighted_mean("value_loss"),
            entropy=weighted_mean("entropy"),
            approximate_kl=weighted_mean("approximate_kl"),
            kind_approximate_kl=weighted_mean("kind_approximate_kl"),
            wait_approximate_kl=weighted_mean("wait_approximate_kl"),
            coordinate_approximate_kl=weighted_mean(
                "coordinate_approximate_kl"
            ),
            clip_fraction=weighted_mean("clip_fraction"),
            gradient_norm=weighted_mean("gradient_norm"),
            kind_entropy=weighted_mean("kind_entropy"),
            wait_entropy=weighted_mean("wait_entropy"),
            coordinate_entropy=weighted_mean("coordinate_entropy"),
            learning_rate=learning_rate_used,
            optimizer_steps=len(records),
            early_stopped=early_stopped,
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
        components = distribution.log_prob_components(batch.actions)
        log_prob = components.total
        if not torch.allclose(
            log_prob[batch.train_mask],
            batch.old_log_prob[batch.train_mask],
            rtol=0,
            atol=tolerance,
        ):
            raise ValueError(
                "stored action likelihoods do not match the collection policy"
            )
        for name, actual, expected in (
            ("kind", components.kind, batch.old_kind_log_prob),
            ("wait", components.wait, batch.old_wait_log_prob),
            (
                "coordinate",
                components.coordinates,
                batch.old_coordinate_log_prob,
            ),
        ):
            if not torch.allclose(
                actual[batch.train_mask],
                expected[batch.train_mask],
                rtol=0,
                atol=tolerance,
            ):
                raise ValueError(
                    f"stored {name} likelihoods do not match the collection policy"
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
