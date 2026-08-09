"""All-branch ranking supervision for directed-pair geometry search."""

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

from .geometry_learning import GeometrySelectorModel
from .geometry_policy import geometry_candidate_vocabulary_sha256
from .geometry_search import GeometryBranchOutcome, GeometrySearchResult
from .policy import encoded_body_ids
from .runway_search import RunwaySearchResult


GEOMETRY_RANKING_VERSION = "r3d-all-branch-geometry-ranking-v2"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _single(value: EncodedBatch) -> EncodedBatch:
    value.validate()
    if value.global_features.shape[0] != 1:
        raise ValueError("geometry ranking requires one public observation")
    return value.copy()


@dataclass(frozen=True, slots=True)
class GeometryOutcomeOrdering:
    """The search rule expressed as a winner and comparable ordered pairs."""

    incumbent_ordinal: int
    winner_ordinal: int
    strictly_improved: bool
    eligible_ordinals: tuple[int, ...]
    preferences: tuple[tuple[int, int], ...]

    def manifest(self) -> dict[str, object]:
        return {
            "incumbent_ordinal": self.incumbent_ordinal,
            "winner_ordinal": self.winner_ordinal,
            "strictly_improved": self.strictly_improved,
            "eligible_ordinals": list(self.eligible_ordinals),
            "preferences": [list(value) for value in self.preferences],
        }


def geometry_outcome_ordering(
    outcomes: Sequence[GeometryBranchOutcome],
) -> GeometryOutcomeOrdering:
    """Reproduce search admissibility, objective order, and ordinal ties.

    Search compares every branch to the incumbent survival boundary, then
    orders eligible branches by the reserve-band public objective and lower
    fixed-slot ordinal. All eligible branches outrank ineligible branches; two
    ineligible branches are intentionally left unordered because search never
    compares them.
    """

    values = tuple(outcomes)
    if not values:
        raise ValueError("geometry ranking requires branch outcomes")
    if any(not isinstance(value, GeometryBranchOutcome) for value in values):
        raise TypeError("geometry ranking outcomes have an invalid type")
    ordinals = tuple(value.candidate.ordinal for value in values)
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("geometry ranking outcomes repeat a candidate slot")
    if values[0].candidate.family != "incumbent" or ordinals[0] != 0:
        raise ValueError("geometry ranking outcomes must begin with incumbent")
    incumbent = values[0]
    if not incumbent.selectable:
        raise ValueError("geometry ranking incumbent is not selectable")
    eligible = tuple(
        value
        for value in values
        if value.selectable and value.survival_nondominated_by(incumbent)
    )
    winner = max(
        eligible,
        key=lambda value: (value.objective, -value.candidate.ordinal),
    )
    strictly_improved = winner.objective > incumbent.objective
    if not strictly_improved:
        winner = incumbent
    eligible_ids = {id(value) for value in eligible}

    def order(
        left: GeometryBranchOutcome, right: GeometryBranchOutcome
    ) -> tuple[int, int] | None:
        left_eligible = id(left) in eligible_ids
        right_eligible = id(right) in eligible_ids
        if left_eligible != right_eligible:
            preferred, rejected = (
                (left, right) if left_eligible else (right, left)
            )
        elif not left_eligible:
            return None
        else:
            left_key = (left.objective, -left.candidate.ordinal)
            right_key = (right.objective, -right.candidate.ordinal)
            preferred, rejected = (
                (left, right) if left_key > right_key else (right, left)
            )
        return preferred.candidate.ordinal, rejected.candidate.ordinal

    preferences = tuple(
        result
        for left_index, left in enumerate(values)
        for right in values[left_index + 1 :]
        if (result := order(left, right)) is not None
    )
    return GeometryOutcomeOrdering(
        incumbent_ordinal=incumbent.candidate.ordinal,
        winner_ordinal=winner.candidate.ordinal,
        strictly_improved=strictly_improved,
        eligible_ordinals=tuple(value.candidate.ordinal for value in eligible),
        preferences=preferences,
    )


@dataclass(frozen=True, slots=True)
class GeometryRankingExample:
    episode_identity: str
    provenance_sha256: str
    search_result_sha256: str
    candidate_set_sha256: str
    candidate_vocabulary_sha256: str
    observation: EncodedBatch
    source_index: int
    destination_index: int
    candidate_count: int
    available_mask: np.ndarray
    winner_index: int
    improved_over_incumbent: bool
    preferences: tuple[tuple[int, int], ...]
    outcome_sha256s: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_identity, str)
            or not self.episode_identity
            or "\x00" in self.episode_identity
        ):
            raise ValueError("geometry ranking episode identity must be nonempty")
        for value, name in (
            (self.provenance_sha256, "geometry ranking provenance"),
            (self.search_result_sha256, "geometry search result"),
            (self.candidate_set_sha256, "geometry candidate set"),
            (self.candidate_vocabulary_sha256, "geometry candidate vocabulary"),
        ):
            _sha256(value, name)
        observation = _single(self.observation)
        if (
            isinstance(self.candidate_count, bool)
            or not isinstance(self.candidate_count, int)
            or self.candidate_count < 2
        ):
            raise ValueError("geometry ranking candidate count must be at least two")
        available = np.asarray(self.available_mask)
        if available.dtype != np.bool_ or available.shape != (
            self.candidate_count,
        ):
            raise ValueError("geometry ranking availability mask is invalid")
        available = np.array(available, dtype=np.bool_, copy=True)
        if not bool(available[0]) or not bool(available.any()):
            raise ValueError("geometry ranking incumbent must be available")
        if (
            isinstance(self.winner_index, bool)
            or not isinstance(self.winner_index, int)
            or not 0 <= self.winner_index < self.candidate_count
            or not bool(available[self.winner_index])
        ):
            raise ValueError("geometry ranking winner is unavailable")
        if type(self.improved_over_incumbent) is not bool:
            raise TypeError("geometry ranking improvement flag must be boolean")
        if self.improved_over_incumbent != (self.winner_index != 0):
            raise ValueError("geometry ranking improvement and winner disagree")
        if (
            isinstance(self.source_index, bool)
            or isinstance(self.destination_index, bool)
            or not isinstance(self.source_index, int)
            or not isinstance(self.destination_index, int)
            or self.source_index == self.destination_index
            or not 0 <= self.source_index < observation.schema.capacity
            or not 0 <= self.destination_index < observation.schema.capacity
            or not bool(observation.body_mask[0, self.source_index])
            or not bool(observation.body_mask[0, self.destination_index])
        ):
            raise ValueError("geometry ranking directed-pair rows are invalid")
        preferences = tuple(tuple(value) for value in self.preferences)
        if len(set(preferences)) != len(preferences):
            raise ValueError("geometry ranking preferences repeat")
        for preferred, rejected in preferences:
            if (
                isinstance(preferred, bool)
                or isinstance(rejected, bool)
                or not isinstance(preferred, int)
                or not isinstance(rejected, int)
                or preferred == rejected
                or not 0 <= preferred < self.candidate_count
                or not 0 <= rejected < self.candidate_count
                or not bool(available[preferred])
                or not bool(available[rejected])
            ):
                raise ValueError("geometry ranking preference is invalid")
        for value in self.outcome_sha256s:
            _sha256(value, "geometry branch outcome")
        if len(self.outcome_sha256s) != int(available.sum()):
            raise ValueError("geometry ranking outcomes and availability differ")
        available.setflags(write=False)
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "available_mask", available)
        object.__setattr__(self, "preferences", preferences)

    @property
    def schema_sha256(self) -> str:
        return self.observation.schema.sha256

    def manifest(self) -> dict[str, object]:
        return {
            "version": GEOMETRY_RANKING_VERSION,
            "episode_identity": self.episode_identity,
            "provenance_sha256": self.provenance_sha256,
            "search_result_sha256": self.search_result_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "candidate_vocabulary_sha256": self.candidate_vocabulary_sha256,
            "schema_sha256": self.schema_sha256,
            "source_index": self.source_index,
            "destination_index": self.destination_index,
            "candidate_count": self.candidate_count,
            "available_mask": self.available_mask.astype(int).tolist(),
            "winner_index": self.winner_index,
            "improved_over_incumbent": self.improved_over_incumbent,
            "preferences": [list(value) for value in self.preferences],
            "outcome_sha256s": list(self.outcome_sha256s),
            "global_sha256": hashlib.sha256(
                self.observation.global_features.tobytes()
            ).hexdigest(),
            "bodies_sha256": hashlib.sha256(
                self.observation.body_features.tobytes()
            ).hexdigest(),
            "body_mask_sha256": hashlib.sha256(
                self.observation.body_mask.tobytes()
            ).hexdigest(),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def geometry_ranking_example(
    observation: Mapping[str, Any],
    result: GeometrySearchResult | RunwaySearchResult,
    *,
    episode_identity: str,
    provenance_sha256: str,
    encoder: TeacherStateEncoder | None = None,
) -> GeometryRankingExample:
    """Convert one complete runway search result into all-branch supervision."""

    if not isinstance(result, (GeometrySearchResult, RunwaySearchResult)):
        raise TypeError(
            "geometry ranking requires a geometry or runway search result"
        )
    candidates = result.candidate_set.candidates
    outcomes = result.outcomes
    if len(outcomes) != len(candidates) or any(
        outcome.candidate != candidate
        for outcome, candidate in zip(outcomes, candidates, strict=True)
    ):
        raise ValueError("geometry search result lacks one outcome per candidate")
    ordering = geometry_outcome_ordering(outcomes)
    expected_winner = next(
        candidate
        for candidate in candidates
        if candidate.ordinal == ordering.winner_ordinal
    )
    if (
        result.selected_candidate != expected_winner
        or result.strictly_improved != ordering.strictly_improved
    ):
        raise ValueError("geometry search result disagrees with search ordering")
    config = result.candidate_set.config
    candidate_count = config.max_candidates
    available = np.zeros(candidate_count, dtype=np.bool_)
    for outcome in outcomes:
        ordinal = outcome.candidate.ordinal
        if not 0 <= ordinal < candidate_count or available[ordinal]:
            raise ValueError("geometry outcome has an invalid fixed slot")
        available[ordinal] = True

    resolved = TeacherStateEncoder() if encoder is None else encoder
    encoded = resolved.encode([observation])
    identifiers = encoded_body_ids(encoded, observation)
    source_id = candidates[0].decision.source_body_id
    destination_id = candidates[0].decision.destination_body_id

    def index(identifier: int | None, name: str) -> int:
        matches = [
            position
            for position, value in enumerate(identifiers)
            if value == identifier
        ]
        if len(matches) != 1:
            raise ValueError(f"geometry ranking {name} did not bind exactly once")
        return matches[0]

    return GeometryRankingExample(
        episode_identity=episode_identity,
        provenance_sha256=_sha256(
            provenance_sha256, "geometry ranking provenance"
        ),
        search_result_sha256=result.sha256,
        candidate_set_sha256=result.candidate_set.sha256,
        candidate_vocabulary_sha256=geometry_candidate_vocabulary_sha256(
            config
        ),
        observation=encoded,
        source_index=index(source_id, "source"),
        destination_index=index(destination_id, "destination"),
        candidate_count=candidate_count,
        available_mask=available,
        winner_index=ordering.winner_ordinal,
        improved_over_incumbent=ordering.strictly_improved,
        preferences=ordering.preferences,
        outcome_sha256s=tuple(
            _canonical_sha256(outcome.manifest()) for outcome in outcomes
        ),
    )


@dataclass(frozen=True, slots=True)
class GeometryRankingTensorBatch:
    global_features: Tensor
    body_features: Tensor
    body_mask: Tensor
    source_index: Tensor
    destination_index: Tensor
    available_mask: Tensor
    winner_index: Tensor
    improved_mask: Tensor
    pair_example_index: Tensor
    preferred_index: Tensor
    rejected_index: Tensor

    def to(self, device: torch.device | str) -> GeometryRankingTensorBatch:
        return GeometryRankingTensorBatch(
            self.global_features.to(device),
            self.body_features.to(device),
            self.body_mask.to(device),
            self.source_index.to(device),
            self.destination_index.to(device),
            self.available_mask.to(device),
            self.winner_index.to(device),
            self.improved_mask.to(device),
            self.pair_example_index.to(device),
            self.preferred_index.to(device),
            self.rejected_index.to(device),
        )


class GeometryRankingDataset(Sequence[GeometryRankingExample]):
    def __init__(self, examples: Sequence[GeometryRankingExample]) -> None:
        self._examples = tuple(examples)
        if not self._examples:
            raise ValueError("geometry ranking dataset must not be empty")
        if len({value.schema_sha256 for value in self._examples}) != 1:
            raise ValueError("geometry ranking dataset mixes schemas")
        if (
            len(
                {
                    value.candidate_vocabulary_sha256
                    for value in self._examples
                }
            )
            != 1
            or len({value.candidate_count for value in self._examples}) != 1
        ):
            raise ValueError("geometry ranking dataset mixes candidate vocabularies")
        self.schema = self._examples[0].observation.schema
        self.candidate_vocabulary_sha256 = self._examples[
            0
        ].candidate_vocabulary_sha256
        self.candidate_count = self._examples[0].candidate_count

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(
        self, index: int | slice
    ) -> GeometryRankingExample | tuple[GeometryRankingExample, ...]:
        return self._examples[index]

    def manifest(self) -> dict[str, object]:
        return {
            "format": GEOMETRY_RANKING_VERSION,
            "schema_sha256": self.schema.sha256,
            "candidate_vocabulary_sha256": self.candidate_vocabulary_sha256,
            "candidate_count": self.candidate_count,
            "examples": [value.sha256 for value in self._examples],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    def batch(
        self, indices: Sequence[int] | None = None
    ) -> GeometryRankingTensorBatch:
        selected = (
            self._examples
            if indices is None
            else tuple(self._examples[index] for index in indices)
        )
        if not selected:
            raise ValueError("geometry ranking tensor batch must not be empty")
        width = max(
            int(active[-1]) + 1 if active.size else 1
            for example in selected
            for active in [np.flatnonzero(example.observation.body_mask[0])]
        )
        pair_rows: list[int] = []
        preferred: list[int] = []
        rejected: list[int] = []
        for row, example in enumerate(selected):
            for winner, loser in example.preferences:
                pair_rows.append(row)
                preferred.append(winner)
                rejected.append(loser)
        return GeometryRankingTensorBatch(
            global_features=torch.from_numpy(
                np.concatenate(
                    [value.observation.global_features for value in selected],
                    axis=0,
                )
            ),
            body_features=torch.from_numpy(
                np.concatenate(
                    [
                        value.observation.body_features[:, :width]
                        for value in selected
                    ],
                    axis=0,
                )
            ),
            body_mask=torch.from_numpy(
                np.concatenate(
                    [
                        value.observation.body_mask[:, :width]
                        for value in selected
                    ],
                    axis=0,
                )
            ),
            source_index=torch.tensor(
                [value.source_index for value in selected], dtype=torch.long
            ),
            destination_index=torch.tensor(
                [value.destination_index for value in selected],
                dtype=torch.long,
            ),
            available_mask=torch.from_numpy(
                np.stack([value.available_mask for value in selected])
            ),
            winner_index=torch.tensor(
                [value.winner_index for value in selected], dtype=torch.long
            ),
            improved_mask=torch.tensor(
                [value.improved_over_incumbent for value in selected],
                dtype=torch.bool,
            ),
            pair_example_index=torch.tensor(pair_rows, dtype=torch.long),
            preferred_index=torch.tensor(preferred, dtype=torch.long),
            rejected_index=torch.tensor(rejected, dtype=torch.long),
        )


@dataclass(frozen=True, slots=True)
class GeometryRankingLoss:
    total: Tensor
    listwise: Tensor
    pairwise: Tensor
    pairwise_accuracy: Tensor


def geometry_ranking_loss(
    model: GeometrySelectorModel,
    batch: GeometryRankingTensorBatch,
    *,
    listwise_weight: float = 1.0,
    pairwise_weight: float = 1.0,
) -> GeometryRankingLoss:
    for value, name in (
        (listwise_weight, "listwise weight"),
        (pairwise_weight, "pairwise weight"),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"geometry ranking {name} must be nonnegative")
    if listwise_weight == 0.0 and pairwise_weight == 0.0:
        raise ValueError("geometry ranking requires a positive loss weight")
    logits = model(
        batch.global_features,
        batch.body_features,
        batch.body_mask,
        batch.source_index,
        batch.destination_index,
    )
    if logits.shape != batch.available_mask.shape:
        raise ValueError("geometry ranking logits and availability differ")
    rows = torch.arange(logits.shape[0], device=logits.device)
    if not bool(batch.available_mask[rows, batch.winner_index].all()):
        raise ValueError("geometry ranking batch has an unavailable winner")
    masked = logits.masked_fill(~batch.available_mask, -torch.inf)
    listwise = F.cross_entropy(masked, batch.winner_index)
    if batch.pair_example_index.numel():
        difference = (
            logits[batch.pair_example_index, batch.preferred_index]
            - logits[batch.pair_example_index, batch.rejected_index]
        )
        pairwise = F.softplus(-difference).mean()
        pairwise_accuracy = (difference > 0.0).float().mean()
    else:
        pairwise = logits.sum() * 0.0
        pairwise_accuracy = logits.new_zeros(())
    return GeometryRankingLoss(
        float(listwise_weight) * listwise + float(pairwise_weight) * pairwise,
        listwise,
        pairwise,
        pairwise_accuracy,
    )


@dataclass(frozen=True, slots=True)
class GeometryRankingTrainingReport:
    steps: int
    examples: int
    preferences: int
    initial_loss: float
    final_loss: float
    final_listwise_loss: float
    final_pairwise_loss: float
    top1_accuracy: float
    incumbent_accuracy: float
    improved_accuracy: float
    pairwise_accuracy: float


def train_geometry_ranker(
    model: GeometrySelectorModel,
    dataset: GeometryRankingDataset,
    *,
    steps: int,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    listwise_weight: float = 1.0,
    pairwise_weight: float = 1.0,
    seed: int = 0,
) -> GeometryRankingTrainingReport:
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps < 1
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0.0
    ):
        raise ValueError("geometry ranking training parameters are invalid")
    if (
        model.schema.sha256 != dataset.schema.sha256
        or model.candidate_set_sha256
        != dataset.candidate_vocabulary_sha256
        or model.candidate_count != dataset.candidate_count
    ):
        raise ValueError("geometry ranking model and dataset identities differ")
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

    def values(indices: Sequence[int] | None = None) -> GeometryRankingTensorBatch:
        return dataset.batch(indices).to(device)

    full = values()
    model.eval()
    with torch.no_grad():
        initial = geometry_ranking_loss(
            model,
            full,
            listwise_weight=listwise_weight,
            pairwise_weight=pairwise_weight,
        )
        initial_loss = float(initial.total)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
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
        batch = values(selected.tolist())
        loss = geometry_ranking_loss(
            model,
            batch,
            listwise_weight=listwise_weight,
            pairwise_weight=pairwise_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        final = geometry_ranking_loss(
            model,
            full,
            listwise_weight=listwise_weight,
            pairwise_weight=pairwise_weight,
        )
        logits = model(
            full.global_features,
            full.body_features,
            full.body_mask,
            full.source_index,
            full.destination_index,
        ).masked_fill(~full.available_mask, -torch.inf)
        correct = logits.argmax(dim=-1) == full.winner_index

        def accuracy(mask: Tensor) -> float:
            return float(correct[mask].float().mean()) if bool(mask.any()) else 0.0

    return GeometryRankingTrainingReport(
        steps=steps,
        examples=len(dataset),
        preferences=sum(len(value.preferences) for value in dataset),
        initial_loss=initial_loss,
        final_loss=float(final.total),
        final_listwise_loss=float(final.listwise),
        final_pairwise_loss=float(final.pairwise),
        top1_accuracy=float(correct.float().mean()),
        incumbent_accuracy=accuracy(~full.improved_mask),
        improved_accuracy=accuracy(full.improved_mask),
        pairwise_accuracy=float(final.pairwise_accuracy),
    )


__all__ = [
    "GEOMETRY_RANKING_VERSION",
    "GeometryOutcomeOrdering",
    "GeometryRankingDataset",
    "GeometryRankingExample",
    "GeometryRankingLoss",
    "GeometryRankingTensorBatch",
    "GeometryRankingTrainingReport",
    "geometry_outcome_ordering",
    "geometry_ranking_example",
    "geometry_ranking_loss",
    "train_geometry_ranker",
]
