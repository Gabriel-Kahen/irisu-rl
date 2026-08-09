"""Deterministic padded-sequence distillation with recurrent TBPTT."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Categorical

from irisu_rl.ppo import quantile_huber_loss
from irisu_rl.schema import TensorSchema

from .action import PointerActionSpec, PointerActionTensor
from .model import PointerPolicyOutput


@dataclass(frozen=True, slots=True)
class PointerSequenceEpisode:
    identity: str
    global_features: Tensor
    body_features: Tensor
    body_mask: Tensor
    actions: PointerActionTensor
    returns: Tensor
    schema: TensorSchema
    pointer_spec: PointerActionSpec = PointerActionSpec()
    policy_weight: Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, str) or not self.identity:
            raise ValueError("sequence episode identity must be nonempty")
        if self.global_features.ndim != 2:
            raise ValueError("episode globals must have shape [L, G]")
        length, global_count = self.global_features.shape
        body_count = self.body_features.shape[-2] if self.body_features.ndim == 3 else 0
        if (
            length <= 0
            or global_count != len(self.schema.global_features)
            or not 0 < body_count <= self.schema.capacity
            or self.body_features.shape
            != (length, body_count, len(self.schema.body_features))
        ):
            raise ValueError("episode observations differ from the schema")
        if (
            self.body_mask.shape != (length, body_count)
            or self.body_mask.dtype != torch.bool
        ):
            raise ValueError("episode body mask must be boolean [L, N]")
        tensors = (self.global_features, self.body_features, self.returns)
        if (
            not all(value.is_floating_point() for value in tensors)
            or len({value.dtype for value in tensors}) != 1
            or len({value.device for value in tensors}) != 1
            or self.returns.shape != (length,)
            or not all(bool(torch.isfinite(value).all()) for value in tensors)
        ):
            raise ValueError("episode floating tensors are malformed")
        action_fields = (
            self.actions.kind,
            self.actions.wait_index,
            self.actions.target_index,
            self.actions.template_index,
        )
        if any(value.device != self.global_features.device for value in action_fields):
            raise ValueError("episode actions and observations must share a device")
        if self.policy_weight is None:
            object.__setattr__(self, "policy_weight", torch.ones_like(self.returns))
        elif (
            self.policy_weight.shape != (length,)
            or self.policy_weight.dtype != self.returns.dtype
            or self.policy_weight.device != self.returns.device
            or not bool(torch.isfinite(self.policy_weight).all())
            or bool((self.policy_weight < 0).any())
            or not bool((self.policy_weight > 0).any())
        ):
            raise ValueError("episode policy weights must be finite nonnegative [L]")
        self.actions.validate(
            torch.Size((length,)), body_count, self.pointer_spec
        )
        shots = self.actions.kind != 0
        if bool(shots.any()):
            rows = shots.nonzero(as_tuple=False).reshape(-1)
            targets = self.actions.target_index[shots]
            if not bool(self.body_mask[rows, targets].all()):
                raise ValueError("sequence action selects a masked target")

    @property
    def length(self) -> int:
        return int(self.global_features.shape[0])

    @property
    def body_count(self) -> int:
        return int(self.body_features.shape[1])


@dataclass(frozen=True, slots=True)
class PointerSequenceBatch:
    global_features: Tensor
    body_features: Tensor
    body_mask: Tensor
    kind: Tensor
    wait_index: Tensor
    target_index: Tensor
    template_index: Tensor
    returns: Tensor
    policy_weight: Tensor
    valid_mask: Tensor
    reset_before: Tensor
    schema: TensorSchema
    pointer_spec: PointerActionSpec
    identities: tuple[str, ...]

    @property
    def time(self) -> int:
        return int(self.global_features.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.global_features.shape[1])

    def training_mask(self, burn_in_steps: int = 0) -> Tensor:
        if (
            isinstance(burn_in_steps, bool)
            or not isinstance(burn_in_steps, int)
            or burn_in_steps < 0
        ):
            raise ValueError("burn-in steps must be a nonnegative integer")
        time = torch.arange(self.time, device=self.valid_mask.device).unsqueeze(1)
        return self.valid_mask & (time >= burn_in_steps)

    def time_slice(self, start: int, stop: int) -> PointerSequenceBatch:
        if not 0 <= start < stop <= self.time:
            raise IndexError("sequence time slice is invalid")
        values = {
            name: getattr(self, name)[start:stop]
            for name in (
                "global_features",
                "body_features",
                "body_mask",
                "kind",
                "wait_index",
                "target_index",
                "template_index",
                "returns",
                "policy_weight",
                "valid_mask",
                "reset_before",
            )
        }
        return PointerSequenceBatch(
            **values,
            schema=self.schema,
            pointer_spec=self.pointer_spec,
            identities=self.identities,
        )

    def to(self, device: torch.device | str) -> PointerSequenceBatch:
        target = torch.device(device)
        values = {
            name: getattr(self, name).to(target)
            for name in (
                "global_features",
                "body_features",
                "body_mask",
                "kind",
                "wait_index",
                "target_index",
                "template_index",
                "returns",
                "policy_weight",
                "valid_mask",
                "reset_before",
            )
        }
        return PointerSequenceBatch(
            **values,
            schema=self.schema,
            pointer_spec=self.pointer_spec,
            identities=self.identities,
        )


def pad_pointer_episodes(
    episodes: Sequence[PointerSequenceEpisode],
    *,
    device: torch.device | str | None = None,
) -> PointerSequenceBatch:
    supplied = tuple(episodes)
    if not supplied:
        raise ValueError("at least one sequence episode is required")
    if any(not isinstance(value, PointerSequenceEpisode) for value in supplied):
        raise TypeError("sequence batch contains a non-episode value")
    schema = supplied[0].schema
    spec = supplied[0].pointer_spec
    if any(value.schema.sha256 != schema.sha256 for value in supplied):
        raise ValueError("sequence batch mixes observation schemas")
    if any(value.pointer_spec.sha256 != spec.sha256 for value in supplied):
        raise ValueError("sequence batch mixes pointer action schemas")
    target = (
        supplied[0].global_features.device
        if device is None
        else torch.device(device)
    )
    dtype = supplied[0].global_features.dtype
    if any(value.global_features.dtype != dtype for value in supplied):
        raise ValueError("sequence batch mixes floating dtypes")
    time = max(value.length for value in supplied)
    batch = len(supplied)
    bodies = max(value.body_count for value in supplied)
    globals_out = torch.zeros(
        (time, batch, len(schema.global_features)), dtype=dtype, device=target
    )
    bodies_out = torch.zeros(
        (time, batch, bodies, len(schema.body_features)),
        dtype=dtype,
        device=target,
    )
    body_mask = torch.zeros((time, batch, bodies), dtype=torch.bool, device=target)
    labels = {
        name: torch.zeros((time, batch), dtype=torch.long, device=target)
        for name in ("kind", "wait_index", "target_index", "template_index")
    }
    returns = torch.zeros((time, batch), dtype=dtype, device=target)
    policy_weight = torch.zeros((time, batch), dtype=dtype, device=target)
    valid = torch.zeros((time, batch), dtype=torch.bool, device=target)
    reset = torch.zeros((time, batch), dtype=torch.bool, device=target)
    for lane, episode in enumerate(supplied):
        length, width = episode.length, episode.body_count
        globals_out[:length, lane].copy_(episode.global_features.to(target))
        bodies_out[:length, lane, :width].copy_(episode.body_features.to(target))
        body_mask[:length, lane, :width].copy_(episode.body_mask.to(target))
        for name in labels:
            labels[name][:length, lane].copy_(
                getattr(episode.actions, name).to(target)
            )
        returns[:length, lane].copy_(episode.returns.to(target))
        assert episode.policy_weight is not None
        policy_weight[:length, lane].copy_(episode.policy_weight.to(target))
        valid[:length, lane] = True
        reset[0, lane] = True
    return PointerSequenceBatch(
        globals_out,
        bodies_out,
        body_mask,
        labels["kind"],
        labels["wait_index"],
        labels["target_index"],
        labels["template_index"],
        returns,
        policy_weight,
        valid,
        reset,
        schema,
        spec,
        tuple(value.identity for value in supplied),
    )


@dataclass(frozen=True, slots=True)
class PointerSequenceConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 1e-5
    kind_coefficient: float = 1.0
    wait_coefficient: float = 1.0
    target_coefficient: float = 1.0
    template_coefficient: float = 1.0
    value_coefficient: float = 0.1
    entropy_coefficient: float = 0.001
    kind_balance_power: float = 1.0
    maximum_kind_weight: float = 32.0
    quantile_huber_kappa: float = 1.0
    max_gradient_norm: float = 1.0
    burn_in_steps: int = 0
    tbptt_steps: int = 16
    seed: int = 0

    def __post_init__(self) -> None:
        positive = (
            self.learning_rate,
            self.kind_coefficient,
            self.wait_coefficient,
            self.target_coefficient,
            self.template_coefficient,
            self.value_coefficient,
            self.quantile_huber_kappa,
            self.max_gradient_norm,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in positive
        ):
            raise ValueError("sequence training coefficients must be positive")
        nonnegative = (self.weight_decay, self.entropy_coefficient)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in nonnegative
        ):
            raise ValueError("sequence regularization coefficients are invalid")
        if (
            isinstance(self.kind_balance_power, bool)
            or not isinstance(self.kind_balance_power, (int, float))
            or not math.isfinite(float(self.kind_balance_power))
            or not 0.0 <= float(self.kind_balance_power) <= 1.0
            or isinstance(self.maximum_kind_weight, bool)
            or not isinstance(self.maximum_kind_weight, (int, float))
            or not math.isfinite(float(self.maximum_kind_weight))
            or float(self.maximum_kind_weight) < 1.0
        ):
            raise ValueError("kind balancing configuration is invalid")
        if (
            isinstance(self.burn_in_steps, bool)
            or not isinstance(self.burn_in_steps, int)
            or self.burn_in_steps < 0
            or isinstance(self.tbptt_steps, bool)
            or not isinstance(self.tbptt_steps, int)
            or self.tbptt_steps <= 0
            or isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed < 2**63
        ):
            raise ValueError("sequence burn-in, TBPTT, or seed is invalid")


@dataclass(frozen=True, slots=True)
class PointerSequenceMetrics:
    total_loss: float
    kind_loss: float
    wait_loss: float
    target_loss: float
    template_loss: float
    value_loss: float
    entropy: float
    kind_accuracy: float
    wait_recall: float
    actionable_recall: float
    predicted_actionable_rate: float
    wait_accuracy: float
    target_accuracy: float
    template_accuracy: float
    examples: int
    wait_examples: int
    actionable_examples: int
    gradient_norm: float = 0.0
    clipped_gradient_norm: float = 0.0
    optimizer_steps: int = 0


def _zero(reference: Tensor) -> Tensor:
    if reference.numel() == 0:
        raise ValueError("cannot construct a differentiable zero from an empty tensor")
    return reference.reshape(-1)[0] * 0.0


def _mean_accuracy(prediction: Tensor, target: Tensor) -> float:
    return float((prediction == target).float().mean()) if target.numel() else 0.0


def _kind_weights(
    kinds: Tensor, policy_weights: Tensor, config: PointerSequenceConfig
) -> Tensor:
    if kinds.shape != policy_weights.shape:
        raise ValueError("kind labels and policy weights must align")
    mass = torch.zeros(3, dtype=policy_weights.dtype, device=kinds.device)
    mass.scatter_add_(0, kinds, policy_weights)
    present = mass > 0
    weights = torch.zeros_like(mass)
    weights[present] = (
        policy_weights.sum() / (int(present.sum()) * mass[present])
    ).pow(float(config.kind_balance_power))
    return weights.clamp_max(float(config.maximum_kind_weight))


def _weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    if (
        values.shape != weights.shape
        or not bool(torch.isfinite(weights).all())
        or bool((weights < 0).any())
        or not bool((weights > 0).any())
    ):
        raise ValueError("policy loss weights are malformed")
    return (values * weights).sum() / weights.sum()


def _validate_batch_model(model: nn.Module, batch: PointerSequenceBatch) -> None:
    if getattr(getattr(model, "schema", None), "sha256", None) != batch.schema.sha256:
        raise ValueError("sequence model and batch schemas differ")
    if (
        getattr(getattr(model, "pointer_spec", None), "sha256", None)
        != batch.pointer_spec.sha256
    ):
        raise ValueError("sequence model and action schemas differ")


def forward_pointer_sequence(
    model: nn.Module,
    batch: PointerSequenceBatch,
    *,
    initial_state: Tensor | None = None,
    chunk_steps: int | None = None,
    detach_between_chunks: bool = False,
) -> PointerPolicyOutput:
    _validate_batch_model(model, batch)
    if chunk_steps is None:
        chunk_steps = batch.time
    if (
        isinstance(chunk_steps, bool)
        or not isinstance(chunk_steps, int)
        or chunk_steps <= 0
    ):
        raise ValueError("sequence chunk size must be positive")
    state = (
        model.initial_state(batch.batch_size, device=batch.global_features.device)
        if initial_state is None
        else initial_state
    )
    outputs: list[PointerPolicyOutput] = []
    for start in range(0, batch.time, chunk_steps):
        chunk = batch.time_slice(start, min(start + chunk_steps, batch.time))
        output = model(
            chunk.global_features,
            chunk.body_features,
            chunk.body_mask,
            state,
            reset_before=chunk.reset_before,
        )
        outputs.append(output)
        state = output.recurrent_state
        if detach_between_chunks:
            state = state.detach()
    return PointerPolicyOutput(
        kind_logits=torch.cat([value.kind_logits for value in outputs], dim=0),
        wait_logits=torch.cat([value.wait_logits for value in outputs], dim=0),
        target_logits=torch.cat([value.target_logits for value in outputs], dim=0),
        template_logits=torch.cat(
            [value.template_logits for value in outputs], dim=0
        ),
        values=torch.cat([value.values for value in outputs], dim=0),
        value_quantiles=torch.cat(
            [value.value_quantiles for value in outputs], dim=0
        ),
        recurrent_state=state,
    )


def pointer_sequence_objective(
    output: PointerPolicyOutput,
    batch: PointerSequenceBatch,
    train_mask: Tensor,
    *,
    config: PointerSequenceConfig | None = None,
) -> tuple[Tensor, PointerSequenceMetrics]:
    resolved = config or PointerSequenceConfig()
    if (
        train_mask.shape != batch.valid_mask.shape
        or train_mask.dtype != torch.bool
        or train_mask.device != batch.valid_mask.device
        or not bool(train_mask.any())
        or bool((train_mask & ~batch.valid_mask).any())
    ):
        raise ValueError("sequence training mask must select valid timesteps")
    expected = (batch.time, batch.batch_size)
    if (
        batch.policy_weight.shape != expected
        or batch.policy_weight.dtype != batch.returns.dtype
        or batch.policy_weight.device != batch.returns.device
        or output.kind_logits.shape != (*expected, 3)
        or output.wait_logits.shape
        != (*expected, len(batch.pointer_spec.wait_choices))
        or output.target_logits.shape[:3]
        != (*expected, 2)
        or output.target_logits.shape[-1] != batch.body_features.shape[-2]
        or output.template_logits.shape[:4]
        != (*expected, 2, batch.body_features.shape[-2])
        or output.template_logits.shape[-1] != batch.pointer_spec.template_count
        or output.value_quantiles.shape != (*expected, 51)
    ):
        raise ValueError("sequence model output shapes are malformed")

    kind_logits = output.kind_logits[train_mask]
    kinds = batch.kind[train_mask]
    policy_weights = batch.policy_weight[train_mask]
    kind_loss = _weighted_mean(
        F.cross_entropy(
            kind_logits,
            kinds,
            weight=_kind_weights(kinds, policy_weights, resolved),
            reduction="none",
        ),
        policy_weights,
    )
    kind_entropy = Categorical(logits=kind_logits).entropy()

    wait_mask = train_mask & (batch.kind == 0)
    if bool(wait_mask.any()):
        wait_logits = output.wait_logits[wait_mask]
        waits = batch.wait_index[wait_mask]
        wait_loss = _weighted_mean(
            F.cross_entropy(wait_logits, waits, reduction="none"),
            batch.policy_weight[wait_mask],
        )
        wait_entropy = Categorical(logits=wait_logits).entropy()
    else:
        wait_logits = output.wait_logits.new_empty(
            (0, output.wait_logits.shape[-1])
        )
        waits = batch.wait_index.new_empty((0,))
        wait_loss = _zero(output.wait_logits)
        wait_entropy = output.wait_logits.new_empty((0,))

    shot_mask = train_mask & (batch.kind != 0)
    if bool(shot_mask.any()):
        time_index, batch_index = shot_mask.nonzero(as_tuple=True)
        branches = batch.kind[shot_mask] - 1
        targets = batch.target_index[shot_mask]
        if not bool(batch.body_mask[time_index, batch_index, targets].all()):
            raise ValueError("sequence label selects a masked target")
        target_logits = output.target_logits[time_index, batch_index, branches]
        selected = target_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        if bool((selected == torch.finfo(selected.dtype).min).any()):
            raise ValueError("sequence label selects an ineligible target")
        shot_weights = batch.policy_weight[shot_mask]
        target_loss = _weighted_mean(
            F.cross_entropy(target_logits, targets, reduction="none"),
            shot_weights,
        )
        target_entropy = Categorical(logits=target_logits).entropy()
        templates = batch.template_index[shot_mask]
        template_logits = output.template_logits[
            time_index, batch_index, branches, targets
        ]
        template_loss = _weighted_mean(
            F.cross_entropy(template_logits, templates, reduction="none"),
            shot_weights,
        )
        template_entropy = Categorical(logits=template_logits).entropy()
    else:
        targets = batch.target_index.new_empty((0,))
        target_logits = output.target_logits.new_empty(
            (0, output.target_logits.shape[-1])
        )
        templates = batch.template_index.new_empty((0,))
        template_logits = output.template_logits.new_empty(
            (0, output.template_logits.shape[-1])
        )
        target_loss = _zero(output.target_logits)
        template_loss = _zero(output.template_logits)
        target_entropy = output.target_logits.new_empty((0,))
        template_entropy = output.template_logits.new_empty((0,))

    value_loss = quantile_huber_loss(
        output.value_quantiles,
        batch.returns,
        train_mask,
        resolved.quantile_huber_kappa,
    )
    branch_entropy = kind_entropy.sum()
    branch_entropy = branch_entropy + wait_entropy.sum()
    branch_entropy = branch_entropy + target_entropy.sum() + template_entropy.sum()
    entropy = branch_entropy / int(train_mask.sum())
    total = (
        resolved.kind_coefficient * kind_loss
        + resolved.wait_coefficient * wait_loss
        + resolved.target_coefficient * target_loss
        + resolved.template_coefficient * template_loss
        + resolved.value_coefficient * value_loss
        - resolved.entropy_coefficient * entropy
    )
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("sequence objective is nonfinite")
    with torch.no_grad():
        kind_prediction = kind_logits.argmax(-1)
        true_wait = kinds == 0
        true_actionable = kinds != 0
        metrics = PointerSequenceMetrics(
            total_loss=float(total.detach()),
            kind_loss=float(kind_loss.detach()),
            wait_loss=float(wait_loss.detach()),
            target_loss=float(target_loss.detach()),
            template_loss=float(template_loss.detach()),
            value_loss=float(value_loss.detach()),
            entropy=float(entropy.detach()),
            kind_accuracy=_mean_accuracy(kind_prediction, kinds),
            wait_recall=_mean_accuracy(
                kind_prediction[true_wait], kinds[true_wait]
            ),
            actionable_recall=(
                float((kind_prediction[true_actionable] != 0).float().mean())
                if bool(true_actionable.any())
                else 0.0
            ),
            predicted_actionable_rate=float((kind_prediction != 0).float().mean()),
            wait_accuracy=_mean_accuracy(wait_logits.argmax(-1), waits),
            target_accuracy=_mean_accuracy(target_logits.argmax(-1), targets),
            template_accuracy=_mean_accuracy(
                template_logits.argmax(-1), templates
            ),
            examples=int(train_mask.sum()),
            wait_examples=int(wait_mask.sum()),
            actionable_examples=int(shot_mask.sum()),
        )
    return total, metrics


def pointer_sequence_loss(
    model: nn.Module,
    batch: PointerSequenceBatch,
    *,
    config: PointerSequenceConfig | None = None,
    initial_state: Tensor | None = None,
    chunk_steps: int | None = None,
    detach_between_chunks: bool = False,
) -> tuple[Tensor, PointerSequenceMetrics, Tensor]:
    resolved = config or PointerSequenceConfig()
    output = forward_pointer_sequence(
        model,
        batch,
        initial_state=initial_state,
        chunk_steps=chunk_steps,
        detach_between_chunks=detach_between_chunks,
    )
    loss, metrics = pointer_sequence_objective(
        output,
        batch,
        batch.training_mask(resolved.burn_in_steps),
        config=resolved,
    )
    return loss, metrics, output.recurrent_state


class PointerSequenceTrainer:
    """Deterministic full-batch development trainer with detached TBPTT carry."""

    def __init__(
        self, model: nn.Module, *, config: PointerSequenceConfig | None = None
    ) -> None:
        self.model = model
        self.config = config or PointerSequenceConfig()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            eps=1e-5,
            foreach=False,
        )
        self._step_index = 0

    def _device_batch(self, batch: PointerSequenceBatch) -> PointerSequenceBatch:
        device = next(self.model.parameters()).device
        return batch if batch.global_features.device == device else batch.to(device)

    def step(self, batch: PointerSequenceBatch) -> PointerSequenceMetrics:
        torch.manual_seed(self.config.seed + self._step_index)
        resolved_batch = self._device_batch(batch)
        _validate_batch_model(self.model, resolved_batch)
        if not bool(
            resolved_batch.training_mask(self.config.burn_in_steps).any()
        ):
            raise ValueError("burn-in consumes every valid sequence timestep")
        self.model.train()
        state = self.model.initial_state(
            resolved_batch.batch_size, device=resolved_batch.global_features.device
        )
        start = min(self.config.burn_in_steps, resolved_batch.time)
        if start:
            burn = resolved_batch.time_slice(0, start)
            with torch.no_grad():
                state = self.model(
                    burn.global_features,
                    burn.body_features,
                    burn.body_mask,
                    state,
                    reset_before=burn.reset_before,
                ).recurrent_state.detach()
        maximum_gradient = 0.0
        maximum_clipped = 0.0
        optimizer_steps = 0
        for offset in range(start, resolved_batch.time, self.config.tbptt_steps):
            chunk = resolved_batch.time_slice(
                offset,
                min(offset + self.config.tbptt_steps, resolved_batch.time),
            )
            output = self.model(
                chunk.global_features,
                chunk.body_features,
                chunk.body_mask,
                state,
                reset_before=chunk.reset_before,
            )
            state = output.recurrent_state.detach()
            train_mask = chunk.valid_mask
            if not bool(train_mask.any()):
                continue
            loss, _ = pointer_sequence_objective(
                output, chunk, train_mask, config=self.config
            )
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.max_gradient_norm
            )
            raw = float(gradient)
            if not math.isfinite(raw):
                raise FloatingPointError("sequence gradient norm is nonfinite")
            maximum_gradient = max(maximum_gradient, raw)
            maximum_clipped = max(
                maximum_clipped, min(raw, self.config.max_gradient_norm)
            )
            self.optimizer.step()
            optimizer_steps += 1
        metrics = self.evaluate(resolved_batch)
        self._step_index += 1
        return replace(
            metrics,
            gradient_norm=maximum_gradient,
            clipped_gradient_norm=maximum_clipped,
            optimizer_steps=optimizer_steps,
        )

    @torch.no_grad()
    def evaluate(self, batch: PointerSequenceBatch) -> PointerSequenceMetrics:
        resolved_batch = self._device_batch(batch)
        prior = self.model.training
        self.model.eval()
        try:
            _, metrics, _ = pointer_sequence_loss(
                self.model,
                resolved_batch,
                config=self.config,
                chunk_steps=self.config.tbptt_steps,
                detach_between_chunks=True,
            )
            return metrics
        finally:
            self.model.train(prior)

    def fit(
        self, batch: PointerSequenceBatch, steps: int
    ) -> PointerSequenceMetrics:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("sequence training steps must be positive")
        metrics: PointerSequenceMetrics | None = None
        for _ in range(steps):
            metrics = self.step(batch)
        assert metrics is not None
        return metrics


__all__ = [
    "PointerSequenceBatch",
    "PointerSequenceConfig",
    "PointerSequenceEpisode",
    "PointerSequenceMetrics",
    "PointerSequenceTrainer",
    "forward_pointer_sequence",
    "pad_pointer_episodes",
    "pointer_sequence_loss",
    "pointer_sequence_objective",
]
