"""Supervised distillation for the entity-pointer policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irisu_rl.ppo import quantile_huber_loss

from .dataset import PointerDataset, PointerTensorBatch


@dataclass(frozen=True, slots=True)
class PointerBCConfig:
    learning_rate: float = 3e-4
    kind_coefficient: float = 1.0
    wait_coefficient: float = 1.0
    target_coefficient: float = 1.0
    template_coefficient: float = 1.0
    value_coefficient: float = 0.1
    quantile_huber_kappa: float = 1.0
    max_gradient_norm: float = 1.0

    def __post_init__(self) -> None:
        values = (
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
            for value in values
        ):
            raise ValueError("pointer BC configuration values must be finite and positive")


@dataclass(frozen=True, slots=True)
class PointerBCMetrics:
    total_loss: float
    kind_loss: float
    wait_loss: float
    target_loss: float
    template_loss: float
    value_loss: float
    kind_accuracy: float
    wait_accuracy: float
    target_accuracy: float
    template_accuracy: float
    weak_target_accuracy: float
    strong_target_accuracy: float
    weak_template_accuracy: float
    strong_template_accuracy: float
    weak_recall: float
    strong_recall: float
    actionable_recall: float
    wait_only_rate: float
    examples: int
    wait_examples: int
    actionable_examples: int
    weak_examples: int
    strong_examples: int
    gradient_norm: float = 0.0


def _zero_loss(reference: Tensor) -> Tensor:
    if reference.numel() == 0:
        raise ValueError("cannot construct a differentiable zero from an empty tensor")
    return reference.reshape(-1)[0] * 0.0


def _mean_or_zero(values: Tensor, mask: Tensor) -> float:
    return float(values[mask].float().mean()) if bool(mask.any()) else 0.0


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("pointer model has no trainable parameters") from exc


def _validate_model_dataset(model: nn.Module, dataset: PointerDataset) -> None:
    schema = getattr(model, "schema", None)
    pointer_spec = getattr(model, "pointer_spec", None)
    if schema is None or getattr(schema, "sha256", None) != dataset.schema_sha256:
        raise ValueError("pointer model and dataset schema identities differ")
    model_pointer_sha256 = getattr(pointer_spec, "sha256", None)
    if (
        isinstance(model_pointer_sha256, str)
        and model_pointer_sha256 != dataset.pointer_spec_sha256
    ):
        raise ValueError("pointer model and dataset action identities differ")


def _forward(model: nn.Module, batch: PointerTensorBatch):
    size = batch.size
    state = model.initial_state(size, device=batch.global_features.device)
    return model(
        batch.global_features.unsqueeze(0),
        batch.body_features.unsqueeze(0),
        batch.body_mask.unsqueeze(0),
        state,
        reset_before=torch.ones(
            (1, size), dtype=torch.bool, device=batch.global_features.device
        ),
    )


def _trim_trailing_masked_bodies(batch: PointerTensorBatch) -> PointerTensorBatch:
    active = batch.body_mask.any(dim=0).nonzero(as_tuple=False).reshape(-1)
    if not active.numel():
        return batch
    width = int(active[-1]) + 1
    return PointerTensorBatch(
        batch.global_features,
        batch.body_features[:, :width],
        batch.body_mask[:, :width],
        batch.kind,
        batch.wait_index,
        batch.target_index,
        batch.template_index,
        batch.value_target,
    )


def pointer_bc_loss(
    model: nn.Module,
    dataset: PointerDataset,
    *,
    config: PointerBCConfig | None = None,
) -> tuple[Tensor, PointerBCMetrics]:
    """Compute branch-conditioned BC loss without mutating the model."""

    resolved = config or PointerBCConfig()
    _validate_model_dataset(model, dataset)
    batch = _trim_trailing_masked_bodies(
        dataset.as_tensors(device=_model_device(model))
    )
    output = _forward(model, batch)
    kind_logits = output.kind_logits[0]
    wait_logits = output.wait_logits[0]
    target_logits = output.target_logits[0]
    template_logits = output.template_logits[0]
    quantiles = output.value_quantiles
    if quantiles is None or quantiles.shape != (1, batch.size, 51):
        raise ValueError("pointer model must emit exactly 51 value quantiles")
    if (
        kind_logits.shape != (batch.size, 3)
        or wait_logits.shape[0] != batch.size
        or target_logits.shape[:3] != (batch.size, 2, batch.body_mask.shape[1])
        or template_logits.ndim != 4
        or template_logits.shape[:3]
        != (batch.size, 2, batch.body_mask.shape[1])
    ):
        raise ValueError("pointer model output shapes differ from the dataset")

    wait_mask = batch.kind == 0
    actionable_mask = batch.kind != 0
    weak_mask = batch.kind == 1
    strong_mask = batch.kind == 2
    kind_loss = F.cross_entropy(kind_logits, batch.kind)
    wait_loss = (
        F.cross_entropy(wait_logits[wait_mask], batch.wait_index[wait_mask])
        if bool(wait_mask.any())
        else _zero_loss(wait_logits)
    )
    rows = torch.arange(batch.size, device=batch.kind.device)
    branches = (batch.kind - 1).clamp_min(0)
    selected_target_logits = target_logits[rows, branches]
    if bool(actionable_mask.any()):
        active_rows = rows[actionable_mask]
        active_branches = branches[actionable_mask]
        active_targets = batch.target_index[actionable_mask]
        if not bool(batch.body_mask[active_rows, active_targets].all()):
            raise ValueError("active pointer label selects a masked or missing target")
        target_loss = F.cross_entropy(
            selected_target_logits[actionable_mask],
            active_targets,
        )
        selected_template_logits = template_logits[
            active_rows, active_branches, active_targets
        ]
        template_loss = F.cross_entropy(
            selected_template_logits,
            batch.template_index[actionable_mask],
        )
    else:
        target_loss = _zero_loss(target_logits)
        template_loss = _zero_loss(template_logits)
        selected_template_logits = template_logits.new_empty(
            (0, template_logits.shape[-1])
        )
    value_loss = quantile_huber_loss(
        quantiles,
        batch.value_target.unsqueeze(0),
        torch.ones((1, batch.size), dtype=torch.bool, device=batch.kind.device),
        resolved.quantile_huber_kappa,
    )
    total = (
        resolved.kind_coefficient * kind_loss
        + resolved.wait_coefficient * wait_loss
        + resolved.target_coefficient * target_loss
        + resolved.template_coefficient * template_loss
        + resolved.value_coefficient * value_loss
    )
    if not torch.isfinite(total):
        raise FloatingPointError("pointer BC loss is nonfinite")

    with torch.no_grad():
        kind_prediction = kind_logits.argmax(-1)
        wait_prediction = wait_logits.argmax(-1)
        target_prediction = selected_target_logits.argmax(-1)
        target_correct = target_prediction == batch.target_index
        template_prediction = torch.full_like(batch.template_index, -1)
        if bool(actionable_mask.any()):
            template_prediction[actionable_mask] = selected_template_logits.argmax(-1)
        template_correct = template_prediction == batch.template_index
        metrics = PointerBCMetrics(
            total_loss=float(total),
            kind_loss=float(kind_loss),
            wait_loss=float(wait_loss),
            target_loss=float(target_loss),
            template_loss=float(template_loss),
            value_loss=float(value_loss),
            kind_accuracy=float((kind_prediction == batch.kind).float().mean()),
            wait_accuracy=_mean_or_zero(wait_prediction == batch.wait_index, wait_mask),
            target_accuracy=_mean_or_zero(target_correct, actionable_mask),
            template_accuracy=_mean_or_zero(template_correct, actionable_mask),
            weak_target_accuracy=_mean_or_zero(target_correct, weak_mask),
            strong_target_accuracy=_mean_or_zero(target_correct, strong_mask),
            weak_template_accuracy=_mean_or_zero(template_correct, weak_mask),
            strong_template_accuracy=_mean_or_zero(template_correct, strong_mask),
            weak_recall=_mean_or_zero(kind_prediction == 1, weak_mask),
            strong_recall=_mean_or_zero(kind_prediction == 2, strong_mask),
            actionable_recall=_mean_or_zero(kind_prediction != 0, actionable_mask),
            wait_only_rate=float((kind_prediction == 0).float().mean()),
            examples=batch.size,
            wait_examples=int(wait_mask.sum()),
            actionable_examples=int(actionable_mask.sum()),
            weak_examples=int(weak_mask.sum()),
            strong_examples=int(strong_mask.sum()),
        )
    return total, metrics


class PointerBCTrainer:
    """Small deterministic full-batch trainer used by behavioral gates."""

    def __init__(
        self,
        model: nn.Module,
        *,
        config: PointerBCConfig | None = None,
    ) -> None:
        self.model = model
        self.config = config or PointerBCConfig()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config.learning_rate,
            eps=1e-5,
            foreach=False,
        )

    def step(self, dataset: PointerDataset) -> PointerBCMetrics:
        self.model.train()
        loss, metrics = pointer_bc_loss(self.model, dataset, config=self.config)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_gradient_norm
        )
        if not torch.isfinite(torch.as_tensor(gradient)):
            raise FloatingPointError("pointer BC gradient norm is nonfinite")
        self.optimizer.step()
        return PointerBCMetrics(
            **{
                name: getattr(metrics, name)
                for name in PointerBCMetrics.__dataclass_fields__
                if name != "gradient_norm"
            },
            gradient_norm=float(gradient),
        )

    @torch.no_grad()
    def evaluate(self, dataset: PointerDataset) -> PointerBCMetrics:
        prior = self.model.training
        self.model.eval()
        try:
            _loss, metrics = pointer_bc_loss(
                self.model, dataset, config=self.config
            )
            return metrics
        finally:
            self.model.train(prior)

    def fit(self, dataset: PointerDataset, steps: int) -> PointerBCMetrics:
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("pointer BC steps must be positive")
        metrics: PointerBCMetrics | None = None
        for _ in range(steps):
            metrics = self.step(dataset)
        assert metrics is not None
        return metrics


PointerDistillationTrainer = PointerBCTrainer
PointerDistillationConfig = PointerBCConfig
DistillationTrainer = PointerBCTrainer
DistillationConfig = PointerBCConfig
