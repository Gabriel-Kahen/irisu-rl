"""Masked set encoder and recurrent actor-critic used by R2 PPO."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import Tensor, nn

from .actions import ActionSpec
from .schema import TensorSchema


@dataclass(frozen=True, slots=True)
class RecurrentModelConfig:
    global_hidden: int = 96
    body_hidden: int = 96
    fused_hidden: int = 192
    recurrent_hidden: int = 192
    recurrent_layers: int = 1
    minimum_concentration: float = 1.001
    coordinate_concentration_log_bias: float = 2.0

    def __post_init__(self) -> None:
        widths = (
            self.global_hidden,
            self.body_hidden,
            self.fused_hidden,
            self.recurrent_hidden,
            self.recurrent_layers,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in widths
        ):
            raise ValueError("model widths and layer count must be positive integers")
        if not 1.0 <= self.minimum_concentration <= 10.0:
            raise ValueError("minimum concentration must be within [1, 10]")
        if (
            not math.isfinite(self.coordinate_concentration_log_bias)
            or not -5 <= self.coordinate_concentration_log_bias <= 5
        ):
            raise ValueError("coordinate concentration log bias must be within [-5, 5]")

    def manifest(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PolicyValueOutput:
    kind_logits: Tensor
    wait_logits: Tensor
    coordinate_alpha: Tensor
    coordinate_beta: Tensor
    values: Tensor
    recurrent_state: Tensor


class MaskedBodySetEncoder(nn.Module):
    """Order-invariant Deep Sets encoder with mean and maximum aggregation."""

    def __init__(self, feature_count: int, hidden: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(feature_count, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden), nn.GELU()
        )

    def forward(self, bodies: Tensor, mask: Tensor) -> Tensor:
        if (
            bodies.ndim < 3
            or mask.shape != bodies.shape[:-1]
            or mask.dtype != torch.bool
        ):
            raise ValueError("body tensor and mask shapes do not match")
        expanded = mask.unsqueeze(-1)
        encoded = self.body(torch.where(expanded, bodies, 0.0))
        encoded = torch.where(expanded, encoded, 0.0)
        count = expanded.sum(dim=-2).clamp_min(1)
        mean = (encoded * expanded).sum(dim=-2) / count
        floor = torch.finfo(encoded.dtype).min
        maximum = encoded.masked_fill(~expanded, floor).amax(dim=-2)
        maximum = torch.where(mask.any(dim=-1, keepdim=True), maximum, 0.0)
        return self.output(torch.cat((mean, maximum), dim=-1))


class RecurrentActorCritic(nn.Module):
    """Shared masked observation encoder, GRU core, and conditional heads."""

    def __init__(
        self,
        schema: TensorSchema,
        *,
        action_spec: ActionSpec | None = None,
        config: RecurrentModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.action_spec = action_spec or ActionSpec()
        self.config = config or RecurrentModelConfig()
        self.global_encoder = nn.Sequential(
            nn.Linear(len(schema.global_features), self.config.global_hidden),
            nn.LayerNorm(self.config.global_hidden),
            nn.GELU(),
            nn.Linear(self.config.global_hidden, self.config.global_hidden),
            nn.GELU(),
        )
        self.body_encoder = MaskedBodySetEncoder(
            len(schema.body_features), self.config.body_hidden
        )
        self.fusion = nn.Sequential(
            nn.Linear(
                self.config.global_hidden + self.config.body_hidden,
                self.config.fused_hidden,
            ),
            nn.LayerNorm(self.config.fused_hidden),
            nn.GELU(),
        )
        self.recurrent = nn.GRU(
            self.config.fused_hidden,
            self.config.recurrent_hidden,
            self.config.recurrent_layers,
        )
        hidden = self.config.recurrent_hidden
        self.kind_head = nn.Linear(hidden, 3)
        self.wait_head = nn.Linear(hidden, len(self.action_spec.wait_choices))
        self.coordinate_head = nn.Linear(hidden, 2 * 2 * 2)
        self.value_head = nn.Linear(hidden, 1)
        self.apply(self._initialize)
        nn.init.orthogonal_(self.kind_head.weight, gain=0.01)
        nn.init.orthogonal_(self.wait_head.weight, gain=0.01)
        nn.init.orthogonal_(self.coordinate_head.weight, gain=0.01)
        coordinate_bias = self.coordinate_head.bias.reshape(2, 2, 2)
        with torch.no_grad():
            coordinate_bias[..., 0].zero_()
            coordinate_bias[..., 1].fill_(self.config.coordinate_concentration_log_bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=2**0.5)
            nn.init.zeros_(module.bias)

    def initial_state(
        self, batch_size: int, *, device: torch.device | str | None = None
    ) -> Tensor:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch size must be a positive integer")
        parameter = next(self.parameters())
        return torch.zeros(
            self.config.recurrent_layers,
            batch_size,
            self.config.recurrent_hidden,
            dtype=parameter.dtype,
            device=parameter.device if device is None else device,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "architecture": "recurrent-actor-critic-v2",
            "actor_schema": self.schema.version,
            "actor_schema_sha256": self.schema.sha256,
            "critic_schema": self.schema.version,
            "critic_schema_sha256": self.schema.sha256,
            "deployable": False,
            "actor_input_is_deployment_compatible": self.schema.source
            == "actor_tracks",
            "transfer_gate": "R4 tracker/input calibration pending",
            "action_schema": self.action_spec.version,
            "action_schema_sha256": self.action_spec.sha256,
            "config": self.config.manifest(),
        }

    def forward(
        self,
        global_features: Tensor,
        body_features: Tensor,
        body_mask: Tensor,
        recurrent_state: Tensor,
        *,
        reset_before: Tensor | None = None,
    ) -> PolicyValueOutput:
        """Run a time-major sequence shaped ``[T, B, ...]``.

        ``reset_before[t, b]`` clears lane ``b`` before consuming timestep
        ``t``. The explicit loop is intentional: hidden-state resets within a
        packed rollout cannot be represented by one monolithic GRU call.
        """

        if global_features.ndim != 3:
            raise ValueError("global features must have shape [T, B, G]")
        time, batch, global_count = global_features.shape
        if time <= 0 or batch <= 0:
            raise ValueError("observation sequence dimensions must be nonzero")
        expected_body = (
            time,
            batch,
            self.schema.capacity,
            len(self.schema.body_features),
        )
        if (
            global_count != len(self.schema.global_features)
            or body_features.shape != expected_body
        ):
            raise ValueError("observation tensor does not match the model schema")
        if body_mask.shape != expected_body[:-1] or body_mask.dtype != torch.bool:
            raise ValueError("body mask shape or dtype mismatch")
        expected_state = (
            self.config.recurrent_layers,
            batch,
            self.config.recurrent_hidden,
        )
        if recurrent_state.shape != expected_state:
            raise ValueError("recurrent state shape mismatch")
        if reset_before is None:
            reset_before = torch.zeros(
                (time, batch), dtype=torch.bool, device=global_features.device
            )
        if reset_before.shape != (time, batch) or reset_before.dtype != torch.bool:
            raise ValueError("reset-before mask must be boolean [T, B]")

        global_embedding = self.global_encoder(global_features)
        body_embedding = self.body_encoder(body_features, body_mask)
        fused = self.fusion(torch.cat((global_embedding, body_embedding), dim=-1))
        hidden = recurrent_state
        sequence: list[Tensor] = []
        for index in range(time):
            hidden = hidden * (~reset_before[index])[None, :, None]
            value, hidden = self.recurrent(fused[index : index + 1], hidden)
            sequence.append(value)
        encoded = torch.cat(sequence, dim=0)
        raw_coordinates = self.coordinate_head(encoded).reshape(time, batch, 2, 2, 2)
        coordinate_mean = torch.sigmoid(raw_coordinates[..., 0])
        concentration_mass = torch.exp(raw_coordinates[..., 1].clamp(-10.0, 10.0))
        alpha = self.config.minimum_concentration + coordinate_mean * concentration_mass
        beta = (
            self.config.minimum_concentration
            + (1.0 - coordinate_mean) * concentration_mass
        )
        return PolicyValueOutput(
            self.kind_head(encoded),
            self.wait_head(encoded),
            alpha,
            beta,
            self.value_head(encoded).squeeze(-1),
            hidden,
        )
