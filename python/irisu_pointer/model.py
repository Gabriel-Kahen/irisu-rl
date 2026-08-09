"""Permutation-equivariant recurrent entity-pointer actor-critic."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irisu_rl.actions import ActionSpec
from irisu_rl.schema import TensorSchema

from .action import PointerActionSpec


@dataclass(frozen=True, slots=True)
class PointerModelConfig:
    global_hidden: int = 96
    body_hidden: int = 96
    attention_hidden: int = 192
    attention_heads: int = 4
    attention_layers: int = 2
    feedforward_hidden: int = 384
    relation_hidden: int = 64
    relation_color_scale: float = 4.0
    relation_distance_scale: float = 2.0
    matcher_prior_scale: float = 12.0
    matcher_gate_bias: float = 2.0
    target_residual_scale: float = 4.0
    matcher_kind_prior_scale: float = 12.0
    kind_residual_scale: float = 4.0
    recurrent_hidden: int = 192
    recurrent_layers: int = 1
    dropout: float = 0.0
    value_quantiles: int = 51

    def __post_init__(self) -> None:
        widths = (
            self.global_hidden,
            self.body_hidden,
            self.attention_hidden,
            self.attention_heads,
            self.attention_layers,
            self.feedforward_hidden,
            self.relation_hidden,
            self.recurrent_hidden,
            self.recurrent_layers,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in widths
        ):
            raise ValueError("pointer model widths and layer counts must be positive")
        if self.attention_hidden % self.attention_heads:
            raise ValueError("attention width must be divisible by its head count")
        relation_scales = (
            self.relation_color_scale,
            self.relation_distance_scale,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in relation_scales
        ):
            raise ValueError("relation preference scales must be finite and positive")
        matcher_values = (
            self.matcher_prior_scale,
            self.target_residual_scale,
            self.matcher_kind_prior_scale,
            self.kind_residual_scale,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value <= 0
            for value in matcher_values
        ):
            raise ValueError("matcher prior and residual scales must be positive")
        if (
            isinstance(self.matcher_gate_bias, bool)
            or not isinstance(self.matcher_gate_bias, (int, float))
            or not math.isfinite(float(self.matcher_gate_bias))
            or self.matcher_gate_bias <= 0
        ):
            raise ValueError("matcher gate bias must be finite and positive")
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise ValueError("dropout must be finite and in [0, 1)")
        if self.value_quantiles != 51:
            raise ValueError("the pointer critic must emit exactly 51 quantiles")

    def manifest(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PointerPolicyOutput:
    kind_logits: Tensor
    wait_logits: Tensor
    target_logits: Tensor
    template_logits: Tensor
    values: Tensor
    value_quantiles: Tensor
    recurrent_state: Tensor


class EntityPointerActorCritic(nn.Module):
    """Attend over individual bodies and retain one score for every entity row."""

    def __init__(
        self,
        schema: TensorSchema,
        *,
        action_spec: ActionSpec | None = None,
        pointer_spec: PointerActionSpec | None = None,
        config: PointerModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.schema = schema
        self.action_spec = action_spec or ActionSpec()
        self.pointer_spec = pointer_spec or PointerActionSpec()
        self.config = config or PointerModelConfig()
        try:
            self._kind_indices = tuple(
                schema.body_features.index(name)
                for name in (
                    "kind_piece",
                    "kind_projectile",
                    "kind_bonus",
                    "kind_unknown",
                )
            )
            self._color_indices = tuple(
                schema.body_features.index(f"color_{index}") for index in range(6)
            )
            self._geometry_indices = tuple(
                schema.body_features.index(name)
                for name in (
                    "effect_x_norm",
                    "effect_y_norm",
                    "width_norm",
                    "height_norm",
                )
            )
            self._lifecycle_indices = tuple(
                schema.body_features.index(name)
                for name in ("lifecycle_falling", "lifecycle_fresh")
            )
        except ValueError as exc:
            raise ValueError(
                "pointer schema lacks public kind, color, or geometry features"
            ) from exc
        self._id_index = (
            schema.body_features.index("id_scaled")
            if "id_scaled" in schema.body_features
            else None
        )
        for wait in self.pointer_spec.wait_choices:
            if wait not in self.action_spec.wait_choices:
                raise ValueError("pointer wait choice is illegal under the action spec")

        attention = self.config.attention_hidden
        self.global_encoder = nn.Sequential(
            nn.Linear(len(schema.global_features), self.config.global_hidden),
            nn.LayerNorm(self.config.global_hidden),
            nn.GELU(),
            nn.Linear(self.config.global_hidden, attention),
        )
        self.body_encoder = nn.Sequential(
            nn.Linear(len(schema.body_features), self.config.body_hidden),
            nn.LayerNorm(self.config.body_hidden),
            nn.GELU(),
            nn.Linear(self.config.body_hidden, attention),
        )
        self.relation_encoder = nn.Sequential(
            nn.Linear(8, self.config.relation_hidden),
            nn.LayerNorm(self.config.relation_hidden),
            nn.GELU(),
            nn.Linear(self.config.relation_hidden, attention),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=attention,
            nhead=self.config.attention_heads,
            dim_feedforward=self.config.feedforward_hidden,
            dropout=float(self.config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.entity_attention = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.config.attention_layers,
            norm=nn.LayerNorm(attention),
            enable_nested_tensor=False,
        )
        self.recurrent = nn.GRU(
            attention,
            self.config.recurrent_hidden,
            self.config.recurrent_layers,
        )
        recurrent = self.config.recurrent_hidden
        self.kind_head = nn.Linear(recurrent, 3)
        self.matcher_kind_gate_head = nn.Linear(recurrent, 1)
        self.wait_head = nn.Linear(recurrent, len(self.pointer_spec.wait_choices))
        self.value_quantile_head = nn.Linear(recurrent, self.config.value_quantiles)

        self.target_query = nn.Linear(recurrent, 2 * attention)
        self.target_key = nn.Linear(attention, attention, bias=False)
        self.matcher_gate_head = nn.Linear(recurrent, 2)
        self.template_query = nn.Linear(recurrent, 2 * attention)
        self.template_body = nn.Linear(attention, attention, bias=False)
        self.template_head = nn.Linear(attention, self.pointer_spec.template_count)

        self.apply(self._initialize)
        for head in (
            self.kind_head,
            self.matcher_kind_gate_head,
            self.wait_head,
            self.target_query,
            self.matcher_gate_head,
        ):
            nn.init.orthogonal_(head.weight, gain=0.01)
        nn.init.constant_(
            self.matcher_gate_head.bias, self.config.matcher_gate_bias
        )
        nn.init.constant_(
            self.matcher_kind_gate_head.bias, self.config.matcher_gate_bias
        )
        nn.init.orthogonal_(self.value_quantile_head.weight, gain=1.0)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=2**0.5)
            if module.bias is not None:
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
            "architecture": "entity-pointer-actor-critic-v2",
            "schema": self.schema.version,
            "schema_sha256": self.schema.sha256,
            "action_schema": self.action_spec.version,
            "action_schema_sha256": self.action_spec.sha256,
            "pointer_action_sha256": self.pointer_spec.sha256,
            "input_ablations": (
                ["id_scaled"] if self._id_index is not None else []
            ),
            "deployable": False,
            "config": self.config.manifest(),
        }

    def _semantic_piece_mask(
        self, body_features: Tensor, body_mask: Tensor
    ) -> Tensor:
        kinds = body_features[..., self._kind_indices]
        return body_mask & (kinds[..., 0] > kinds[..., 1:].amax(dim=-1))

    def _ablate_body_inputs(self, body_features: Tensor) -> Tensor:
        if self._id_index is None:
            return body_features
        output = body_features.clone()
        output[..., self._id_index] = 0.0
        return output

    def _matcher_prior(
        self, body_features: Tensor, piece_mask: Tensor
    ) -> Tensor:
        """Return the bounded analytic closest-pair target prior."""

        x, y, _, _ = (
            body_features[..., index] for index in self._geometry_indices
        )
        colors = body_features[..., self._color_indices]
        lifecycle = body_features[..., self._lifecycle_indices].sum(dim=-1)
        piece_confidence = body_features[..., self._kind_indices[0]]
        eligibility = (piece_confidence * lifecycle).clamp(0.0, 1.0)
        color_similarity = torch.einsum(
            "...ic,...jc->...ij", colors, colors
        ).clamp(0.0, 1.0)
        pair_confidence = (
            eligibility.unsqueeze(-1)
            * eligibility.unsqueeze(-2)
            * color_similarity
        )
        body_count = body_features.shape[-2]
        distinct = ~torch.eye(
            body_count, dtype=torch.bool, device=body_features.device
        )
        pair_mask = (
            piece_mask.unsqueeze(-1)
            & piece_mask.unsqueeze(-2)
            & distinct
            & (pair_confidence > 0)
        )
        x_i, x_j = x.unsqueeze(-1), x.unsqueeze(-2)
        y_i, y_j = y.unsqueeze(-1), y.unsqueeze(-2)
        cost = (x_i - x_j).abs() + 0.1875 * (y_i - y_j).abs()
        pair_quality = -cost - (1.0 - pair_confidence)
        preferred_member = (y_i > y_j) | ((y_i == y_j) & (x_i >= x_j))
        directed_mask = pair_mask & preferred_member
        floor = torch.finfo(pair_quality.dtype).min
        per_target = pair_quality.masked_fill(
            ~directed_mask, floor
        ).amax(dim=-1)
        has_partner = directed_mask.any(dim=-1)
        global_best = per_target.amax(dim=-1, keepdim=True)
        winner = has_partner & (per_target == global_best)
        return winner.to(body_features.dtype)

    def _gated_target_logits(
        self, raw_residual: Tensor, matcher_prior: Tensor, matcher_gate: Tensor
    ) -> Tensor:
        residual = self.config.target_residual_scale * torch.tanh(raw_residual)
        has_prior = matcher_prior.any(dim=-1, keepdim=True)
        effective_gate = matcher_gate * has_prior.to(matcher_gate.dtype)
        return (
            (1.0 - effective_gate.unsqueeze(-1)) * residual
            + effective_gate.unsqueeze(-1)
            * self.config.matcher_prior_scale
            * matcher_prior.unsqueeze(-2)
        )

    def _gated_kind_logits(
        self,
        raw_residual: Tensor,
        matcher_prior: Tensor,
        body_features: Tensor,
        matcher_gate: Tensor,
    ) -> Tensor:
        """Bootstrap match timing/strength while leaving non-match intent learnable."""

        residual = self.config.kind_residual_scale * torch.tanh(raw_residual)
        has_prior = matcher_prior.any(dim=-1, keepdim=True)
        y_index = self._geometry_indices[1]
        target_y = (
            matcher_prior * body_features[..., y_index]
        ).sum(dim=-1)
        # MatcherShotPolicy uses a strong shot when 390 - target_y_px > 150.
        strong = target_y < 0.5
        prior_kind = F.one_hot(
            torch.where(strong, 2, 1), num_classes=3
        ).to(dtype=residual.dtype)
        effective_gate = matcher_gate * has_prior.to(matcher_gate.dtype)
        return (
            (1.0 - effective_gate) * residual
            + effective_gate
            * self.config.matcher_kind_prior_scale
            * prior_kind
        )

    def _relation_context(
        self, body_features: Tensor, piece_mask: Tensor
    ) -> Tensor:
        """Project an equivariant nearest-same-color pair summary per piece."""

        x, y, width, height = (
            body_features[..., index] for index in self._geometry_indices
        )
        colors = body_features[..., self._color_indices]
        dx = x.unsqueeze(-2) - x.unsqueeze(-1)
        dy = y.unsqueeze(-2) - y.unsqueeze(-1)
        half_width_sum = (
            width.unsqueeze(-2) + width.unsqueeze(-1)
        ) / 2.0
        half_height_sum = (
            height.unsqueeze(-2) + height.unsqueeze(-1)
        ) / 2.0
        x_gap = dx.abs() - half_width_sum
        y_gap = dy.abs() - half_height_sum
        x_overlap = (-x_gap).clamp_min(0.0)
        y_overlap = (-y_gap).clamp_min(0.0)
        distance = torch.sqrt(dx.square() + dy.square() + 1e-12)
        color_similarity = torch.einsum(
            "...ic,...jc->...ij", colors, colors
        )
        pair_features = torch.stack(
            (
                color_similarity,
                dx,
                dy,
                x_gap,
                y_gap,
                x_overlap,
                y_overlap,
                distance,
            ),
            dim=-1,
        )
        body_count = body_features.shape[-2]
        distinct = ~torch.eye(
            body_count, dtype=torch.bool, device=body_features.device
        )
        pair_mask = (
            piece_mask.unsqueeze(-1)
            & piece_mask.unsqueeze(-2)
            & distinct
        )
        preference = (
            self.config.relation_color_scale * color_similarity
            - self.config.relation_distance_scale * distance
        )
        floor = torch.finfo(preference.dtype).min
        weights = torch.softmax(
            preference.masked_fill(~pair_mask, floor), dim=-1
        )
        weights = torch.where(pair_mask, weights, torch.zeros_like(weights))
        summary = (weights.unsqueeze(-1) * pair_features).sum(dim=-2)
        context = self.relation_encoder(summary)
        has_partner = pair_mask.any(dim=-1)
        return torch.where(
            has_partner.unsqueeze(-1), context, torch.zeros_like(context)
        )

    def _validate_inputs(
        self,
        global_features: Tensor,
        body_features: Tensor,
        body_mask: Tensor,
        recurrent_state: Tensor,
        reset_before: Tensor | None,
    ) -> tuple[int, int, int, Tensor]:
        if global_features.ndim != 3:
            raise ValueError("global features must have shape [T, B, G]")
        time, batch, global_count = global_features.shape
        if time <= 0 or batch <= 0:
            raise ValueError("observation sequence dimensions must be nonzero")
        body_count = body_features.shape[-2] if body_features.ndim == 4 else 0
        expected_body = (
            time,
            batch,
            body_count,
            len(self.schema.body_features),
        )
        if (
            global_count != len(self.schema.global_features)
            or not 0 < body_count <= self.schema.capacity
            or body_features.shape != expected_body
        ):
            raise ValueError("observation tensor does not match the model schema")
        if (
            body_mask.shape != expected_body[:-1]
            or body_mask.dtype != torch.bool
            or body_mask.device != global_features.device
        ):
            raise ValueError("body mask shape or dtype mismatch")
        if (
            not global_features.is_floating_point()
            or body_features.dtype != global_features.dtype
            or body_features.device != global_features.device
        ):
            raise ValueError("observation tensors must share a floating dtype and device")
        expected_state = (
            self.config.recurrent_layers,
            batch,
            self.config.recurrent_hidden,
        )
        if (
            recurrent_state.shape != expected_state
            or recurrent_state.dtype != global_features.dtype
            or recurrent_state.device != global_features.device
        ):
            raise ValueError("recurrent state shape, dtype, or device mismatch")
        if reset_before is None:
            reset_before = torch.zeros(
                (time, batch), dtype=torch.bool, device=global_features.device
            )
        if (
            reset_before.shape != (time, batch)
            or reset_before.dtype != torch.bool
            or reset_before.device != global_features.device
        ):
            raise ValueError("reset-before mask must be boolean [T, B]")
        return time, batch, body_count, reset_before

    def forward(
        self,
        global_features: Tensor,
        body_features: Tensor,
        body_mask: Tensor,
        recurrent_state: Tensor,
        *,
        reset_before: Tensor | None = None,
        target_mask: Tensor | None = None,
    ) -> PointerPolicyOutput:
        """Run a time-major sequence shaped ``[T, B, ...]``."""

        time, batch, body_count, resets = self._validate_inputs(
            global_features,
            body_features,
            body_mask,
            recurrent_state,
            reset_before,
        )
        body_features = self._ablate_body_inputs(body_features)
        semantic_piece_mask = self._semantic_piece_mask(
            body_features, body_mask
        )
        if target_mask is None:
            target_mask = semantic_piece_mask
        elif (
            target_mask.shape != body_mask.shape
            or target_mask.dtype != torch.bool
            or target_mask.device != body_mask.device
        ):
            raise ValueError("target mask must be boolean [T, B, N]")
        else:
            target_mask = target_mask & body_mask
        global_tokens = self.global_encoder(global_features).reshape(
            time * batch, 1, self.config.attention_hidden
        )
        visible_bodies = torch.where(
            body_mask.unsqueeze(-1), body_features, torch.zeros_like(body_features)
        )
        body_tokens = (
            self.body_encoder(visible_bodies)
            + self._relation_context(visible_bodies, semantic_piece_mask)
        ).reshape(
            time * batch, body_count, self.config.attention_hidden
        )
        tokens = torch.cat((global_tokens, body_tokens), dim=1)
        padding_mask = torch.cat(
            (
                torch.zeros(
                    (time * batch, 1),
                    dtype=torch.bool,
                    device=body_mask.device,
                ),
                ~body_mask.reshape(time * batch, body_count),
            ),
            dim=1,
        )
        attended = self.entity_attention(tokens, src_key_padding_mask=padding_mask)
        global_context = attended[:, 0].reshape(
            time, batch, self.config.attention_hidden
        )
        body_context = attended[:, 1:].reshape(
            time, batch, body_count, self.config.attention_hidden
        )

        hidden = recurrent_state
        sequence: list[Tensor] = []
        for index in range(time):
            hidden = hidden * (~resets[index])[None, :, None]
            output, hidden = self.recurrent(
                global_context[index : index + 1], hidden
            )
            sequence.append(output)
        encoded = torch.cat(sequence, dim=0)

        target_query = self.target_query(encoded).reshape(
            time, batch, 2, self.config.attention_hidden
        )
        target_key = self.target_key(body_context)
        target_residual = torch.einsum(
            "tbsh,tbnh->tbsn", target_query, target_key
        ) / math.sqrt(self.config.attention_hidden)
        matcher_prior = self._matcher_prior(
            visible_bodies, semantic_piece_mask & target_mask
        )
        matcher_gate = torch.sigmoid(self.matcher_gate_head(encoded))
        target_logits = self._gated_target_logits(
            target_residual, matcher_prior, matcher_gate
        )

        template_query = self.template_query(encoded).reshape(
            time, batch, 2, self.config.attention_hidden
        )
        template_features = F.gelu(
            template_query.unsqueeze(-2)
            + self.template_body(body_context).unsqueeze(-3)
        )
        template_logits = self.template_head(template_features)
        floor = torch.finfo(target_logits.dtype).min
        target_logits = target_logits.masked_fill(
            ~target_mask.unsqueeze(-2), floor
        )
        template_logits = template_logits.masked_fill(
            ~target_mask.unsqueeze(-2).unsqueeze(-1), floor
        )
        kind_logits = self._gated_kind_logits(
            self.kind_head(encoded),
            matcher_prior,
            visible_bodies,
            torch.sigmoid(self.matcher_kind_gate_head(encoded)),
        )
        has_target = target_mask.any(dim=-1)
        kind_logits = kind_logits.masked_fill(
            torch.stack(
                (
                    torch.zeros_like(has_target),
                    ~has_target,
                    ~has_target,
                ),
                dim=-1,
            ),
            floor,
        )
        value_quantiles = self.value_quantile_head(encoded)
        return PointerPolicyOutput(
            kind_logits=kind_logits,
            wait_logits=self.wait_head(encoded),
            target_logits=target_logits,
            template_logits=template_logits,
            values=value_quantiles.mean(dim=-1),
            value_quantiles=value_quantiles,
            recurrent_state=hidden,
        )
