"""Identity-bound distillation for searched steering geometry."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irisu_rl.encoding import EncodedBatch, TeacherStateEncoder
from irisu_rl.schema import TensorSchema

from .policy import encoded_body_ids


_BOARD_CONTEXT_RELATIONS = (
    "source_dx",
    "source_dy",
    "source_distance",
    "destination_dx",
    "destination_dy",
    "destination_distance",
    "is_source",
    "is_destination",
    "same_color_as_source",
    "same_color_as_destination",
    "grouped",
    "same_chain_as_destination",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _single(batch: EncodedBatch) -> EncodedBatch:
    batch.validate()
    if batch.global_features.shape[0] != 1:
        raise ValueError("geometry supervision requires one observation")
    return batch.copy()


@dataclass(frozen=True, slots=True)
class GeometryExample:
    """One searched geometry label for an already selected directed pair."""

    episode_identity: str
    provenance_sha256: str
    candidate_set_sha256: str
    observation: EncodedBatch
    source_index: int
    destination_index: int
    candidate_index: int
    candidate_count: int
    improved_over_incumbent: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_identity, str)
            or not self.episode_identity
            or "\x00" in self.episode_identity
        ):
            raise ValueError("geometry episode identity must be nonempty")
        _sha256(self.provenance_sha256, "geometry provenance")
        _sha256(self.candidate_set_sha256, "geometry candidate set")
        owned = _single(self.observation)
        for value in (
            self.source_index,
            self.destination_index,
            self.candidate_index,
            self.candidate_count,
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("geometry labels must be integers")
        if self.candidate_count < 2:
            raise ValueError("geometry candidate count must be at least two")
        if not 0 <= self.candidate_index < self.candidate_count:
            raise ValueError("geometry candidate index is out of range")
        if (
            self.source_index == self.destination_index
            or not 0 <= self.source_index < owned.schema.capacity
            or not 0 <= self.destination_index < owned.schema.capacity
            or not bool(owned.body_mask[0, self.source_index])
            or not bool(owned.body_mask[0, self.destination_index])
        ):
            raise ValueError("geometry source/destination rows are invalid")
        if type(self.improved_over_incumbent) is not bool:
            raise TypeError("geometry improvement label must be boolean")
        object.__setattr__(self, "observation", owned)

    @property
    def schema_sha256(self) -> str:
        return self.observation.schema.sha256

    def manifest(self) -> dict[str, object]:
        return {
            "episode_identity": self.episode_identity,
            "provenance_sha256": self.provenance_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "schema_sha256": self.schema_sha256,
            "source_index": self.source_index,
            "destination_index": self.destination_index,
            "candidate_index": self.candidate_index,
            "candidate_count": self.candidate_count,
            "improved_over_incumbent": self.improved_over_incumbent,
            "global_sha256": hashlib.sha256(
                self.observation.global_features.tobytes()
            ).hexdigest(),
            "bodies_sha256": hashlib.sha256(
                self.observation.body_features.tobytes()
            ).hexdigest(),
            "mask_sha256": hashlib.sha256(
                self.observation.body_mask.tobytes()
            ).hexdigest(),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def geometry_example(
    observation: Mapping[str, Any],
    *,
    source_body_id: int,
    destination_body_id: int,
    candidate_index: int,
    candidate_count: int,
    improved_over_incumbent: bool,
    episode_identity: str,
    provenance_sha256: str,
    candidate_set_sha256: str,
    encoder: TeacherStateEncoder | None = None,
) -> GeometryExample:
    resolved = TeacherStateEncoder() if encoder is None else encoder
    encoded = resolved.encode([observation])
    identifiers = encoded_body_ids(encoded, observation)

    def index(identifier: int) -> int:
        matches = [
            position
            for position, value in enumerate(identifiers)
            if value == identifier
        ]
        if len(matches) != 1:
            raise ValueError("geometry body ID is absent or duplicated")
        return matches[0]

    return GeometryExample(
        episode_identity,
        provenance_sha256,
        candidate_set_sha256,
        encoded,
        index(source_body_id),
        index(destination_body_id),
        candidate_index,
        candidate_count,
        improved_over_incumbent,
    )


class GeometryDataset(Sequence[GeometryExample]):
    def __init__(self, examples: Sequence[GeometryExample]) -> None:
        self._examples = tuple(examples)
        if not self._examples:
            raise ValueError("geometry dataset must not be empty")
        if len({value.schema_sha256 for value in self._examples}) != 1:
            raise ValueError("geometry dataset mixes observation schemas")
        if len({value.candidate_set_sha256 for value in self._examples}) != 1:
            raise ValueError("geometry dataset mixes candidate sets")
        if len({value.candidate_count for value in self._examples}) != 1:
            raise ValueError("geometry dataset mixes candidate counts")
        self.schema = self._examples[0].observation.schema
        self.candidate_set_sha256 = self._examples[0].candidate_set_sha256
        self.candidate_count = self._examples[0].candidate_count

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(
        self, index: int | slice
    ) -> GeometryExample | tuple[GeometryExample, ...]:
        return self._examples[index]

    def manifest(self) -> dict[str, object]:
        return {
            "format": "irisu-directed-geometry-dataset-v1",
            "schema_sha256": self.schema.sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "candidate_count": self.candidate_count,
            "examples": [value.sha256 for value in self._examples],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    def tensors(
        self, indices: Sequence[int] | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        selected = (
            self._examples
            if indices is None
            else tuple(self._examples[index] for index in indices)
        )
        if not selected:
            raise ValueError("geometry tensor batch must not be empty")
        width = max(
            int(active[-1]) + 1 if active.size else 1
            for example in selected
            for active in [np.flatnonzero(example.observation.body_mask[0])]
        )
        return (
            torch.from_numpy(
                np.concatenate(
                    [value.observation.global_features for value in selected],
                    axis=0,
                )
            ),
            torch.from_numpy(
                np.concatenate(
                    [
                        value.observation.body_features[:, :width]
                        for value in selected
                    ],
                    axis=0,
                )
            ),
            torch.from_numpy(
                np.concatenate(
                    [value.observation.body_mask[:, :width] for value in selected],
                    axis=0,
                )
            ),
            torch.tensor(
                [value.source_index for value in selected], dtype=torch.long
            ),
            torch.tensor(
                [value.destination_index for value in selected], dtype=torch.long
            ),
            torch.tensor(
                [value.candidate_index for value in selected], dtype=torch.long
            ),
        )


@dataclass(frozen=True, slots=True)
class GeometryModelConfig:
    body_hidden: int = 96
    pair_hidden: int = 128
    dropout: float = 0.0
    board_context_hidden: int = 0

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in (self.body_hidden, self.pair_hidden)
        ):
            raise ValueError("geometry model widths must be positive")
        if (
            isinstance(self.board_context_hidden, bool)
            or not isinstance(self.board_context_hidden, int)
            or self.board_context_hidden < 0
        ):
            raise ValueError(
                "geometry board context width must be a nonnegative integer"
            )
        if (
            isinstance(self.dropout, bool)
            or not isinstance(self.dropout, (int, float))
            or not math.isfinite(float(self.dropout))
            or not 0.0 <= float(self.dropout) < 1.0
        ):
            raise ValueError("geometry dropout must be in [0, 1)")


class GeometrySelectorModel(nn.Module):
    """Permutation-safe geometry classifier for a bound directed pair."""

    def __init__(
        self,
        schema: TensorSchema,
        *,
        candidate_count: int,
        candidate_set_sha256: str,
        config: GeometryModelConfig | None = None,
    ) -> None:
        super().__init__()
        if (
            isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count < 2
        ):
            raise ValueError("geometry candidate count must be at least two")
        self.schema = schema
        self.candidate_count = candidate_count
        self.candidate_set_sha256 = _sha256(
            candidate_set_sha256, "geometry candidate set"
        )
        self.config = GeometryModelConfig() if config is None else config
        self._id_indices = tuple(
            schema.body_features.index(name)
            for name in ("id_scaled", "chain_id_scaled")
            if name in schema.body_features
        )
        self._x_index = schema.body_features.index("effect_x_norm")
        self._y_index = schema.body_features.index("effect_y_norm")
        self._color_indices = tuple(
            schema.body_features.index(f"color_{index}") for index in range(6)
        )
        self._chain_id_index = (
            schema.body_features.index("chain_id_scaled")
            if "chain_id_scaled" in schema.body_features
            else None
        )
        body_width = len(schema.body_features)
        hidden = self.config.body_hidden
        self.body_encoder = nn.Sequential(
            nn.Linear(body_width, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
        context_hidden = self.config.board_context_hidden
        if context_hidden:
            self.board_context_encoder: nn.Module | None = nn.Sequential(
                nn.Linear(
                    body_width + len(_BOARD_CONTEXT_RELATIONS),
                    context_hidden,
                ),
                nn.LayerNorm(context_hidden),
                nn.GELU(),
                nn.Linear(context_hidden, context_hidden),
                nn.GELU(),
            )
            self.board_context_attention: nn.Module | None = nn.Sequential(
                nn.Linear(context_hidden, context_hidden),
                nn.Tanh(),
                nn.Linear(context_hidden, 1, bias=False),
            )
        else:
            self.board_context_encoder = None
            self.board_context_attention = None
        self.selector = nn.Sequential(
            nn.Linear(
                2 * hidden
                + len(schema.global_features)
                + 3
                + context_hidden,
                self.config.pair_hidden,
            ),
            nn.LayerNorm(self.config.pair_hidden),
            nn.GELU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(self.config.pair_hidden, candidate_count),
        )

    def manifest(self) -> dict[str, object]:
        config = {
            "body_hidden": self.config.body_hidden,
            "pair_hidden": self.config.pair_hidden,
            "dropout": self.config.dropout,
        }
        manifest: dict[str, object] = {
            "architecture": (
                "directed-pair-geometry-selector-v3-strategic-board-context"
                if self.config.board_context_hidden
                else "directed-pair-geometry-selector-v1"
            ),
            "schema_sha256": self.schema.sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "candidate_count": self.candidate_count,
            "id_ablations": [
                self.schema.body_features[index] for index in self._id_indices
            ],
            "relations": ["dx", "dy", "distance"],
            "config": config,
        }
        if self.config.board_context_hidden:
            config["board_context_hidden"] = self.config.board_context_hidden
            manifest["board_context"] = {
                "bodies": "all active rows including the bound pair",
                "id_ablations": [
                    self.schema.body_features[index]
                    for index in self._id_indices
                ],
                "pair_relative_features": list(_BOARD_CONTEXT_RELATIONS),
                "aggregation": "masked permutation-invariant softmax attention",
            }
        return manifest

    @property
    def architecture_sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    def _board_context_relations(
        self,
        body_features: Tensor,
        source_index: Tensor,
        destination_index: Tensor,
    ) -> Tensor:
        batch, count = body_features.shape[:2]
        rows = torch.arange(batch, device=body_features.device)
        x = body_features[..., self._x_index]
        y = body_features[..., self._y_index]
        source_x = x[rows, source_index]
        source_y = y[rows, source_index]
        destination_x = x[rows, destination_index]
        destination_y = y[rows, destination_index]
        source_dx = x - source_x[:, None]
        source_dy = y - source_y[:, None]
        destination_dx = x - destination_x[:, None]
        destination_dy = y - destination_y[:, None]
        positions = torch.arange(count, device=body_features.device)[None, :]
        colors = body_features[..., self._color_indices]
        source_colors = colors[rows, source_index]
        destination_colors = colors[rows, destination_index]
        same_color_as_source = (
            (colors * source_colors[:, None]).sum(dim=-1) > 0.5
        )
        same_color_as_destination = (
            (colors * destination_colors[:, None]).sum(dim=-1) > 0.5
        )
        if self._chain_id_index is None:
            grouped = torch.zeros_like(x, dtype=torch.bool)
            same_chain_as_destination = grouped
        else:
            chains = body_features[..., self._chain_id_index]
            destination_chains = chains[rows, destination_index, None]
            grouped = chains > 0.0
            same_chain_as_destination = (
                grouped
                & (destination_chains > 0.0)
                & (chains == destination_chains)
            )
        return torch.stack(
            (
                source_dx,
                source_dy,
                torch.sqrt(source_dx.square() + source_dy.square() + 1e-12),
                destination_dx,
                destination_dy,
                torch.sqrt(
                    destination_dx.square()
                    + destination_dy.square()
                    + 1e-12
                ),
                positions == source_index[:, None],
                positions == destination_index[:, None],
                same_color_as_source,
                same_color_as_destination,
                grouped,
                same_chain_as_destination,
            ),
            dim=-1,
        ).to(body_features.dtype)

    def forward(
        self,
        global_features: Tensor,
        body_features: Tensor,
        body_mask: Tensor,
        source_index: Tensor,
        destination_index: Tensor,
    ) -> Tensor:
        batch = body_features.shape[0]
        if (
            global_features.shape[0] != batch
            or body_mask.shape != body_features.shape[:2]
            or source_index.shape != (batch,)
            or destination_index.shape != (batch,)
        ):
            raise ValueError("geometry selector batch shapes disagree")
        rows = torch.arange(batch, device=body_features.device)
        if not bool(
            (
                body_mask[rows, source_index]
                & body_mask[rows, destination_index]
                & (source_index != destination_index)
            ).all()
        ):
            raise ValueError("geometry selector received an inactive pair")
        context_relations = (
            self._board_context_relations(
                body_features, source_index, destination_index
            )
            if self.board_context_encoder is not None
            else None
        )
        ablated = body_features.clone()
        if self._id_indices:
            ablated[..., self._id_indices] = 0.0
        encoded = self.body_encoder(ablated)
        source = encoded[rows, source_index]
        destination = encoded[rows, destination_index]
        source_x = body_features[rows, source_index, self._x_index]
        source_y = body_features[rows, source_index, self._y_index]
        destination_x = body_features[rows, destination_index, self._x_index]
        destination_y = body_features[rows, destination_index, self._y_index]
        dx = destination_x - source_x
        dy = destination_y - source_y
        relation = torch.stack(
            (dx, dy, torch.sqrt(dx.square() + dy.square() + 1e-12)), dim=-1
        )
        selector_inputs = [source, destination, global_features, relation]
        if self.board_context_encoder is not None:
            assert self.board_context_attention is not None
            assert context_relations is not None
            context_input = torch.cat(
                (ablated, context_relations), dim=-1
            )
            context_input = context_input.masked_fill(
                ~body_mask[..., None], 0.0
            )
            context_rows = self.board_context_encoder(context_input)
            attention = self.board_context_attention(
                context_rows
            ).squeeze(-1)
            attention = attention.masked_fill(~body_mask, -torch.inf)
            weights = torch.softmax(attention, dim=-1)
            context = (weights[..., None] * context_rows).sum(dim=1)
            selector_inputs.append(context)
        return self.selector(
            torch.cat(selector_inputs, dim=-1)
        )


@dataclass(frozen=True, slots=True)
class GeometryTrainingReport:
    steps: int
    examples: int
    improved_examples: int
    initial_loss: float
    final_loss: float
    accuracy: float
    improved_accuracy: float
    incumbent_accuracy: float


def train_geometry_selector(
    model: GeometrySelectorModel,
    dataset: GeometryDataset,
    *,
    steps: int,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    seed: int = 0,
) -> GeometryTrainingReport:
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 1
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0.0
    ):
        raise ValueError("geometry training parameters are invalid")
    if (
        model.schema.sha256 != dataset.schema.sha256
        or model.candidate_set_sha256 != dataset.candidate_set_sha256
        or model.candidate_count != dataset.candidate_count
    ):
        raise ValueError("geometry model and dataset identities differ")
    device = next(model.parameters()).device
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    improved = torch.tensor(
        [
            index
            for index, value in enumerate(dataset)
            if value.improved_over_incumbent
        ],
        dtype=torch.long,
    )
    incumbent = torch.tensor(
        [
            index
            for index, value in enumerate(dataset)
            if not value.improved_over_incumbent
        ],
        dtype=torch.long,
    )

    def batch(indices: Sequence[int] | None = None) -> tuple[Tensor, ...]:
        return tuple(value.to(device) for value in dataset.tensors(indices))

    full = batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    model.eval()
    with torch.no_grad():
        initial_loss = float(F.cross_entropy(model(*full[:-1]), full[-1]))
    model.train()
    for _ in range(steps):
        count = min(batch_size, len(dataset))
        if improved.numel() and incumbent.numel() and count >= 2:
            improved_count = count // 2
            incumbent_count = count - improved_count
            selected = torch.cat(
                (
                    improved[
                        torch.randint(
                            improved.numel(),
                            (improved_count,),
                            generator=generator,
                        )
                    ],
                    incumbent[
                        torch.randint(
                            incumbent.numel(),
                            (incumbent_count,),
                            generator=generator,
                        )
                    ],
                )
            )
        else:
            pool = improved if improved.numel() else incumbent
            selected = pool[
                torch.randint(pool.numel(), (count,), generator=generator)
            ]
        values = batch(selected.tolist())
        loss = F.cross_entropy(model(*values[:-1]), values[-1])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(*full[:-1])
        final_loss = float(F.cross_entropy(logits, full[-1]))
        predicted = logits.argmax(dim=-1)
        correct = predicted == full[-1]
        improved_mask = torch.tensor(
            [value.improved_over_incumbent for value in dataset],
            dtype=torch.bool,
            device=device,
        )

        def accuracy(mask: Tensor) -> float:
            return float(correct[mask].float().mean()) if bool(mask.any()) else 0.0

    return GeometryTrainingReport(
        steps,
        len(dataset),
        int(improved.numel()),
        initial_loss,
        final_loss,
        float(correct.float().mean()),
        accuracy(improved_mask),
        accuracy(~improved_mask),
    )


__all__ = [
    "GeometryDataset",
    "GeometryExample",
    "GeometryModelConfig",
    "GeometrySelectorModel",
    "GeometryTrainingReport",
    "geometry_example",
    "train_geometry_selector",
]
