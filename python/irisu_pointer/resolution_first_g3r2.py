"""Generation-03 revision-02 wide relational resolution learner.

This module is intentionally independent of the frozen G2 and G3-v1
learners.  It promotes the successful opened-board prototype into a
deterministic, serializable implementation:

* a 712-column identity-free relational-board representation;
* train-only correlation screening after retaining the complete 126-column
  global/candidate/incumbent block;
* depth-three histogram Newton boosting;
* whole-seed cross-fitting;
* an independent candidate deployability head (the primary score); and
* an invariant source/destination pair-fraction head used only as an
  auxiliary score.

Query and pair keys are used to partition rows for aggregation.  They are
never model inputs.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from irisu_rl.encoding import TeacherStateEncoder
from .resolution_first import FEATURE_NAMES
from .resolution_first_g2 import (
    BoardBranchRecord,
    BoardResolutionDataset,
    board_branch_records,
)


_RELATIONAL_NAMES = (
    "role_candidate_source",
    "role_candidate_destination",
    "role_incumbent_source",
    "role_incumbent_destination",
    "chain_grouped",
    "chain_group_size_log",
    "same_chain_candidate_source",
    "same_chain_candidate_destination",
    "same_chain_incumbent_source",
    "same_chain_incumbent_destination",
    "candidate_source_dx",
    "candidate_source_dy",
    "candidate_source_distance",
    "candidate_destination_dx",
    "candidate_destination_dy",
    "candidate_destination_distance",
    "same_color_candidate_source",
    "same_color_candidate_destination",
    "same_color_incumbent_source",
    "same_color_incumbent_destination",
)
_GLOBAL_NAMES = (
    "tick_scaled_zero",
    *TeacherStateEncoder.schema.global_features[1:],
    "query_phase",
    "shot_phase",
    "spawn_phase",
)
_BODY_NAMES = tuple(
    f"{name}_zero" if name in {"id_scaled", "chain_id_scaled"} else name
    for name in TeacherStateEncoder.schema.body_features
) + _RELATIONAL_NAMES
_PAIRED_NAMES = (
    *(f"candidate_{name}" for name in FEATURE_NAMES),
    *(f"incumbent_{name}" for name in FEATURE_NAMES),
    *(f"candidate_minus_incumbent_{name}" for name in FEATURE_NAMES),
)
WIDE_FEATURE_NAMES = (
    *_GLOBAL_NAMES,
    *_PAIRED_NAMES,
    "body_count_over_30",
    *(f"body_mean_{name}" for name in _BODY_NAMES),
    *(f"body_std_{name}" for name in _BODY_NAMES),
    *(f"body_min_{name}" for name in _BODY_NAMES),
    *(f"body_max_{name}" for name in _BODY_NAMES),
    *(f"candidate_source_body_{name}" for name in _BODY_NAMES),
    *(f"candidate_destination_body_{name}" for name in _BODY_NAMES),
    *(f"incumbent_source_body_{name}" for name in _BODY_NAMES),
    *(f"incumbent_destination_body_{name}" for name in _BODY_NAMES),
    *(f"candidate_source_minus_destination_body_{name}" for name in _BODY_NAMES),
)
WIDE_FEATURE_WIDTH = 712
PRESERVED_BLOCK_WIDTH = 126
_ID_BODY_COLUMNS = tuple(
    TeacherStateEncoder.schema.body_features.index(name)
    for name in ("id_scaled", "chain_id_scaled")
)
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

if (
    len(_GLOBAL_NAMES) != 15
    or len(_BODY_NAMES) != 65
    or len(_PAIRED_NAMES) != 111
    or len(WIDE_FEATURE_NAMES) != WIDE_FEATURE_WIDTH
    or len(set(WIDE_FEATURE_NAMES)) != WIDE_FEATURE_WIDTH
):
    raise RuntimeError("G3R2 wide feature schema is inconsistent")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{name} is malformed")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(f"{name} is malformed")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"{name} is malformed")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{name} is malformed")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{name} is malformed")
    return result


def _readonly(value: np.ndarray, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class WideBoard:
    """Candidate rows and the non-input keys needed for evaluation/grouping."""

    features: np.ndarray
    labels: np.ndarray
    seeds: np.ndarray
    query_ids: np.ndarray
    source_indices: np.ndarray
    destination_indices: np.ndarray
    ordinals: np.ndarray

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        count = len(features)
        arrays = (
            self.labels,
            self.seeds,
            self.query_ids,
            self.source_indices,
            self.destination_indices,
            self.ordinals,
        )
        if (
            features.ndim != 2
            or features.shape[1] != WIDE_FEATURE_WIDTH
            or any(np.asarray(value).shape != (count,) for value in arrays)
            or count == 0
            or not np.isfinite(features).all()
        ):
            raise ValueError("G3R2 wide board is malformed")
        labels = np.asarray(self.labels)
        if not np.isin(labels, (False, True, 0, 1)).all():
            raise ValueError("G3R2 labels are malformed")
        seeds = np.asarray(self.seeds)
        indices = (
            np.asarray(self.source_indices),
            np.asarray(self.destination_indices),
            np.asarray(self.ordinals),
        )
        if (
            not np.issubdtype(seeds.dtype, np.integer)
            or any(not np.issubdtype(value.dtype, np.integer) for value in indices)
            or np.any(indices[0] < 0)
            or np.any(indices[1] < 0)
            or np.any(indices[0] == indices[1])
            or np.any(indices[2] < 0)
            or any(not isinstance(value, str) or not value for value in self.query_ids)
        ):
            raise ValueError("G3R2 grouping keys are malformed")
        object.__setattr__(self, "features", _readonly(features, np.float64))
        object.__setattr__(self, "labels", _readonly(labels, np.bool_))
        object.__setattr__(self, "seeds", _readonly(seeds, np.int64))
        object.__setattr__(
            self, "query_ids", _readonly(np.asarray(self.query_ids), object)
        )
        object.__setattr__(
            self, "source_indices", _readonly(indices[0], np.int64)
        )
        object.__setattr__(
            self, "destination_indices", _readonly(indices[1], np.int64)
        )
        object.__setattr__(self, "ordinals", _readonly(indices[2], np.int64))

    @property
    def unique_seeds(self) -> tuple[int, ...]:
        return tuple(sorted(int(value) for value in np.unique(self.seeds)))

    @property
    def sha256(self) -> str:
        rows = [
            {
                "features": [float(value) for value in self.features[index]],
                "label": bool(self.labels[index]),
                "seed": int(self.seeds[index]),
                "query_id": str(self.query_ids[index]),
                "source_index": int(self.source_indices[index]),
                "destination_index": int(self.destination_indices[index]),
                "ordinal": int(self.ordinals[index]),
            }
            for index in range(len(self.features))
        ]
        return _sha256(
            {
                "schema": "irisu-r3i-wide-relational-board-g3r2-v1",
                "feature_names": list(WIDE_FEATURE_NAMES),
                "rows": rows,
            }
        )

    def take(self, rows: np.ndarray | Sequence[int] | Sequence[bool]) -> WideBoard:
        selected = np.asarray(rows)
        return WideBoard(
            self.features[selected],
            self.labels[selected],
            self.seeds[selected],
            self.query_ids[selected],
            self.source_indices[selected],
            self.destination_indices[selected],
            self.ordinals[selected],
        )


def wide_board_from_records(
    records: Iterable[BoardBranchRecord], *, include_incumbents: bool = False
) -> WideBoard:
    """Build the exact 712-column prototype representation."""

    rows = tuple(records)
    dataset = BoardResolutionDataset(rows)
    (
        global_features,
        bodies,
        mask,
        candidate,
        incumbent,
        source,
        destination,
        incumbent_pair,
    ) = dataset.tensors()
    active = bodies.masked_fill(~mask[:, :, None], 0.0)
    for column in _ID_BODY_COLUMNS:
        if bool((active[:, :, column] != 0).any()):
            raise RuntimeError("absolute body/chain identity leaked into G3R2")
    if bool((global_features[:, 0] != 0).any()):
        raise RuntimeError("absolute tick leaked into G3R2")
    count = mask.sum(1, keepdim=True).float()
    mean = active.sum(1) / count
    variance = (active.square().sum(1) / count - mean.square()).clamp_min(0.0)
    maximum = bodies.masked_fill(~mask[:, :, None], -math.inf).amax(1)
    minimum = bodies.masked_fill(~mask[:, :, None], math.inf).amin(1)
    row_index = np.arange(len(rows))
    torch_rows = __import__("torch").arange(len(rows))
    endpoints = __import__("torch").cat(
        (
            bodies[torch_rows, source],
            bodies[torch_rows, destination],
            bodies[torch_rows, incumbent_pair[:, 0]],
            bodies[torch_rows, incumbent_pair[:, 1]],
            bodies[torch_rows, source] - bodies[torch_rows, destination],
        ),
        dim=1,
    )
    paired = __import__("torch").cat(
        (candidate, incumbent, candidate - incumbent), dim=1
    )
    features = __import__("torch").cat(
        (
            global_features,
            paired,
            count / 30.0,
            mean,
            variance.sqrt(),
            minimum,
            maximum,
            endpoints,
        ),
        dim=1,
    ).numpy()
    if features.shape[1] != WIDE_FEATURE_WIDTH or not np.isfinite(features).all():
        raise RuntimeError("G3R2 wide feature extraction failed")
    keep = (
        np.ones(len(rows), dtype=bool)
        if include_incumbents
        else np.asarray([row.ordinal != 0 for row in rows])
    )
    if not keep.any():
        raise ValueError("G3R2 board has no nonincumbent candidates")
    return WideBoard(
        features[keep],
        np.asarray([row.candidate_resolved for row in rows])[keep],
        np.asarray([row.seed for row in rows])[keep],
        np.asarray([row.query_id for row in rows], dtype=object)[keep],
        np.asarray([row.source_index for row in rows])[keep],
        np.asarray([row.destination_index for row in rows])[keep],
        np.asarray([row.ordinal for row in rows])[keep],
    )


def wide_board_from_entries(
    entries: Iterable[Mapping[str, Any]], *, include_incumbents: bool = False
) -> WideBoard:
    return wide_board_from_records(
        board_branch_records(entries), include_incumbents=include_incumbents
    )


@dataclass(frozen=True, slots=True)
class PairGroups:
    """Invariant pair aggregation; keys remain outside the feature matrix."""

    features: np.ndarray
    targets: np.ndarray
    seeds: np.ndarray
    keys: tuple[tuple[int, str, int, int], ...]
    row_groups: tuple[tuple[int, ...], ...]

    @classmethod
    def from_board(cls, board: WideBoard) -> PairGroups:
        grouped: dict[tuple[int, str, int, int], list[int]] = defaultdict(list)
        for index, values in enumerate(
            zip(
                board.seeds,
                board.query_ids,
                board.source_indices,
                board.destination_indices,
                strict=True,
            )
        ):
            key = (int(values[0]), str(values[1]), int(values[2]), int(values[3]))
            grouped[key].append(index)
        keys = tuple(sorted(grouped))
        row_groups = tuple(tuple(grouped[key]) for key in keys)
        return cls(
            _readonly(
                np.asarray(
                    [board.features[list(indices)].mean(0) for indices in row_groups]
                ),
                np.float64,
            ),
            _readonly(
                np.asarray(
                    [board.labels[list(indices)].mean() for indices in row_groups]
                ),
                np.float64,
            ),
            _readonly(np.asarray([key[0] for key in keys]), np.int64),
            keys,
            row_groups,
        )

    def expand(self, values: np.ndarray, count: int) -> np.ndarray:
        supplied = np.asarray(values, dtype=np.float64)
        if supplied.shape != (len(self.keys),) or not np.isfinite(supplied).all():
            raise ValueError("G3R2 pair scores are malformed")
        output = np.empty(count, dtype=np.float64)
        covered = np.zeros(count, dtype=bool)
        for score, indices in zip(supplied, self.row_groups, strict=True):
            output[list(indices)] = score
            covered[list(indices)] = True
        if not covered.all():
            raise RuntimeError("G3R2 pair grouping is incomplete")
        return output


@dataclass(frozen=True, slots=True)
class BoostConfig:
    rounds: int
    depth: int = 3
    learning_rate: float = 0.05
    l2: float = 8.0
    minimum_leaf: int = 18
    maximum_features: int = 180
    preserved_features: int = PRESERVED_BLOCK_WIDTH
    bins: int = 16
    balance_classes: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.balance_classes, np.bool_)
            or not isinstance(self.balance_classes, bool)
            or min(
                self.rounds,
                self.depth,
                self.minimum_leaf,
                self.maximum_features,
                self.preserved_features,
                self.bins,
            )
            < 1
            or self.preserved_features > self.maximum_features
            or self.maximum_features > WIDE_FEATURE_WIDTH
            or not 0 < self.learning_rate <= 1
            or self.l2 <= 0
        ):
            raise ValueError("G3R2 boost configuration is invalid")


@dataclass(frozen=True, slots=True)
class FeatureBinner:
    raw_feature_count: int
    selected_columns: tuple[int, ...]
    thresholds: tuple[tuple[float, ...], ...]

    @classmethod
    def fit(
        cls, values: np.ndarray, targets: np.ndarray, config: BoostConfig
    ) -> FeatureBinner:
        x = np.asarray(values, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if (
            x.ndim != 2
            or y.shape != (len(x),)
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or x.shape[1] != WIDE_FEATURE_WIDTH
        ):
            raise ValueError("G3R2 screening supervision is malformed")
        centered_target = y - y.mean()
        correlations: list[float] = []
        for column in range(x.shape[1]):
            candidate = x[:, column]
            centered = candidate - candidate.mean()
            denominator = float(
                np.sqrt(np.square(centered).sum() * np.square(centered_target).sum())
            )
            correlations.append(
                0.0
                if denominator <= 0
                else float(abs(np.dot(centered, centered_target) / denominator))
            )
        selected = list(range(config.preserved_features))
        selected.extend(
            int(column)
            # Preserve the authoritative prototype convention exactly,
            # including NumPy's deterministic reverse-argsort tie order.
            for column in np.argsort(correlations)[::-1]
            if column not in selected
        )
        selected = selected[: config.maximum_features]
        quantiles = np.linspace(0.0, 1.0, config.bins + 1)[1:-1]
        thresholds = tuple(
            tuple(
                float(value)
                for value in np.unique(np.quantile(x[:, column], quantiles))
            )
            for column in selected
        )
        return cls(x.shape[1], tuple(selected), thresholds)

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if (
            x.ndim != 2
            or x.shape[1] != self.raw_feature_count
            or not np.isfinite(x).all()
        ):
            raise ValueError("G3R2 binner input is malformed")
        output = np.empty((len(x), len(self.selected_columns)), dtype=np.uint8)
        for target, (source, thresholds) in enumerate(
            zip(self.selected_columns, self.thresholds, strict=True)
        ):
            output[:, target] = np.searchsorted(
                np.asarray(thresholds), x[:, source]
            )
        return output

    @property
    def widths(self) -> np.ndarray:
        return np.asarray([len(values) for values in self.thresholds])

    def manifest(self) -> dict[str, object]:
        return {
            "raw_feature_count": self.raw_feature_count,
            "selected_columns": list(self.selected_columns),
            "thresholds": [list(values) for values in self.thresholds],
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> FeatureBinner:
        try:
            raw = _integer(value["raw_feature_count"], "raw feature count")
            selected = tuple(
                _integer(item, "selected feature")
                for item in _sequence(value["selected_columns"], "selected columns")
            )
            thresholds = tuple(
                tuple(
                    _number(item, "bin threshold")
                    for item in _sequence(row, "bin thresholds")
                )
                for row in _sequence(value["thresholds"], "threshold inventory")
            )
            result = cls(raw, selected, thresholds)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("G3R2 binner manifest is malformed") from exc
        if (
            raw != WIDE_FEATURE_WIDTH
            or not selected
            or len(selected) != len(thresholds)
            or len(selected) != len(set(selected))
            or any(not 0 <= item < raw for item in selected)
            or any(
                tuple(sorted(set(row))) != row or len(row) > 15
                for row in thresholds
            )
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("G3R2 binner manifest is malformed")
        return result


@dataclass(frozen=True, slots=True)
class TreeNode:
    feature: int
    threshold: int
    value: float
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature == -1

    @property
    def depth(self) -> int:
        if self.is_leaf:
            return 0
        assert self.left is not None and self.right is not None
        return 1 + max(self.left.depth, self.right.depth)

    def apply(self, bins: np.ndarray) -> np.ndarray:
        output = np.empty(len(bins), dtype=np.float64)
        stack: list[tuple[TreeNode, np.ndarray]] = [
            (self, np.arange(len(bins), dtype=np.int64))
        ]
        while stack:
            node, rows = stack.pop()
            if node.is_leaf:
                output[rows] = node.value
                continue
            assert node.left is not None and node.right is not None
            branch = bins[rows, node.feature] <= node.threshold
            stack.append((node.left, rows[branch]))
            stack.append((node.right, rows[~branch]))
        return output

    def manifest(self) -> dict[str, object]:
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "value": self.value,
            "left": None if self.left is None else self.left.manifest(),
            "right": None if self.right is None else self.right.manifest(),
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> TreeNode:
        try:
            feature = _integer(value["feature"], "tree feature")
            threshold = _integer(value["threshold"], "tree threshold")
            node_value = _number(value["value"], "tree value")
            raw_left, raw_right = value["left"], value["right"]
            if feature == -1:
                if threshold != -1 or raw_left is not None or raw_right is not None:
                    raise RuntimeError("G3R2 tree leaf is malformed")
                result = cls(feature, threshold, node_value)
            else:
                result = cls(
                    feature,
                    threshold,
                    node_value,
                    cls.from_manifest(_mapping(raw_left, "left tree")),
                    cls.from_manifest(_mapping(raw_right, "right tree")),
                )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("G3R2 tree node is malformed") from exc
        if (
            feature < -1
            or threshold < -1
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("G3R2 tree node is malformed")
        return result


def _fit_tree(
    bins: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    widths: np.ndarray,
    rows: np.ndarray,
    *,
    depth: int,
    l2: float,
    minimum_leaf: int,
) -> TreeNode:
    total_gradient = float(gradient[rows].sum())
    total_hessian = float(hessian[rows].sum())
    leaf = -total_gradient / (total_hessian + l2)
    if depth == 0 or len(rows) < 2 * minimum_leaf:
        return TreeNode(-1, -1, leaf)
    parent = total_gradient * total_gradient / (total_hessian + l2)
    best_gain = -math.inf
    best_split: tuple[int, int] | None = None
    for feature, width in enumerate(widths):
        if width < 1:
            continue
        count = np.bincount(bins[rows, feature], minlength=int(width) + 1)
        grad = np.bincount(
            bins[rows, feature],
            weights=gradient[rows],
            minlength=int(width) + 1,
        )
        hess = np.bincount(
            bins[rows, feature],
            weights=hessian[rows],
            minlength=int(width) + 1,
        )
        left_count = np.cumsum(count)[:-1]
        left_gradient = np.cumsum(grad)[:-1]
        left_hessian = np.cumsum(hess)[:-1]
        gain = np.where(
            (left_count >= minimum_leaf)
            & (len(rows) - left_count >= minimum_leaf),
            left_gradient**2 / (left_hessian + l2)
            + (total_gradient - left_gradient) ** 2
            / (total_hessian - left_hessian + l2)
            - parent,
            -np.inf,
        )
        threshold = int(np.argmax(gain))
        candidate_gain = float(gain[threshold])
        if candidate_gain > best_gain:
            best_gain = candidate_gain
            best_split = (feature, threshold)
    if best_split is None or best_gain <= 1e-10:
        return TreeNode(-1, -1, leaf)
    feature, threshold = best_split
    branch = bins[rows, feature] <= threshold
    return TreeNode(
        feature,
        threshold,
        leaf,
        _fit_tree(
            bins,
            gradient,
            hessian,
            widths,
            rows[branch],
            depth=depth - 1,
            l2=l2,
            minimum_leaf=minimum_leaf,
        ),
        _fit_tree(
            bins,
            gradient,
            hessian,
            widths,
            rows[~branch],
            depth=depth - 1,
            l2=l2,
            minimum_leaf=minimum_leaf,
        ),
    )


def _seed_weights(
    seeds: np.ndarray, targets: np.ndarray, balance_classes: bool
) -> np.ndarray:
    counts = Counter(int(seed) for seed in seeds)
    weights = np.asarray([1.0 / counts[int(seed)] for seed in seeds])
    if balance_classes:
        if set(np.unique(targets)) != {0.0, 1.0}:
            raise ValueError("class-balanced G3R2 head requires two classes")
        prevalence = float(targets.mean())
        weights *= np.where(
            targets == 1.0, 0.5 / prevalence, 0.5 / (1.0 - prevalence)
        )
    return weights / weights.mean()


@dataclass(frozen=True, slots=True)
class HistogramNewtonBoost:
    config: BoostConfig
    binner: FeatureBinner
    base_logit: float
    trees: tuple[TreeNode, ...]

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        targets: np.ndarray,
        seeds: np.ndarray,
        config: BoostConfig,
    ) -> HistogramNewtonBoost:
        x = np.asarray(values, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        seed = np.asarray(seeds, dtype=np.int64)
        if (
            x.shape != (len(y), WIDE_FEATURE_WIDTH)
            or seed.shape != y.shape
            or len(y) == 0
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.any((y < 0) | (y > 1))
        ):
            raise ValueError("G3R2 boost supervision is malformed")
        binner = FeatureBinner.fit(x, y, config)
        bins = binner.transform(x)
        weights = _seed_weights(seed, y, config.balance_classes)
        prior = float(np.clip(np.average(y, weights=weights), 0.01, 0.99))
        base = math.log(prior / (1.0 - prior))
        score = np.full(len(y), base, dtype=np.float64)
        rows = np.arange(len(y), dtype=np.int64)
        trees: list[TreeNode] = []
        for _round in range(config.rounds):
            probability = 1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0)))
            gradient = (probability - y) * weights
            hessian = probability * (1.0 - probability) * weights
            tree = _fit_tree(
                bins,
                gradient,
                hessian,
                binner.widths,
                rows,
                depth=config.depth,
                l2=config.l2,
                minimum_leaf=config.minimum_leaf,
            )
            trees.append(tree)
            score += config.learning_rate * tree.apply(bins)
        return cls(config, binner, base, tuple(trees))

    def logits(self, values: np.ndarray) -> np.ndarray:
        bins = self.binner.transform(values)
        score = np.full(len(bins), self.base_logit, dtype=np.float64)
        for tree in self.trees:
            score += self.config.learning_rate * tree.apply(bins)
        return score

    def probabilities(self, values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(self.logits(values), -30.0, 30.0)))

    def manifest(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "binner": self.binner.manifest(),
            "base_logit": self.base_logit,
            "trees": [tree.manifest() for tree in self.trees],
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> HistogramNewtonBoost:
        try:
            config_raw = dict(_mapping(value["config"], "boost config"))
            if set(config_raw) != set(asdict(BoostConfig(rounds=1))):
                raise RuntimeError("G3R2 boost configuration is malformed")
            config = BoostConfig(**config_raw)
            binner = FeatureBinner.from_manifest(
                _mapping(value["binner"], "feature binner")
            )
            base = _number(value["base_logit"], "base logit")
            trees = tuple(
                TreeNode.from_manifest(_mapping(row, "tree"))
                for row in _sequence(value["trees"], "trees")
            )
            result = cls(config, binner, base, trees)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("G3R2 boost manifest is malformed") from exc
        if (
            len(trees) != config.rounds
            or tuple(binner.selected_columns[: config.preserved_features])
            != tuple(range(config.preserved_features))
            or len(binner.selected_columns) != config.maximum_features
            or any(
                tree.depth > config.depth
                or any(
                    node.feature >= len(binner.selected_columns)
                    or (
                        node.feature >= 0
                        and node.threshold >= len(binner.thresholds[node.feature])
                    )
                    for node in _walk_tree(tree)
                )
                for tree in trees
            )
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("G3R2 boost manifest is malformed")
        return result


def _walk_tree(root: TreeNode) -> tuple[TreeNode, ...]:
    output: list[TreeNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        output.append(node)
        if not node.is_leaf:
            assert node.left is not None and node.right is not None
            stack.extend((node.left, node.right))
    return tuple(output)


@dataclass(frozen=True, slots=True)
class G3R2Config:
    folds: int = 5
    blend_pair_weight: float = 0.0
    partition_id: str = "development-internal-sorted-v1"
    seed_partition: tuple[tuple[int, ...], ...] = ()
    candidate: BoostConfig = BoostConfig(rounds=200)
    pair: BoostConfig = BoostConfig(
        rounds=400,
        depth=3,
        learning_rate=0.04,
        l2=8.0,
        minimum_leaf=8,
        maximum_features=180,
        preserved_features=PRESERVED_BLOCK_WIDTH,
        bins=16,
        balance_classes=False,
    )

    def __post_init__(self) -> None:
        flattened = tuple(seed for fold in self.seed_partition for seed in fold)
        if (
            self.folds not in {5, 8}
            or not 0 <= self.blend_pair_weight <= 1
            or not self.partition_id
            or (self.folds == 8 and not self.seed_partition)
            or (
                self.seed_partition
                and (
                    len(self.seed_partition) != self.folds
                    or any(not fold or tuple(sorted(set(fold))) != fold for fold in self.seed_partition)
                    or len(set(flattened)) != len(flattened)
                )
            )
            or (
                self.seed_partition
                and self.partition_id == "development-internal-sorted-v1"
            )
        ):
            raise ValueError("G3R2 cross-fit configuration is invalid")


def whole_seed_folds(
    seeds: Sequence[int], count: int
) -> tuple[tuple[int, ...], ...]:
    unique = tuple(sorted(set(int(seed) for seed in seeds)))
    if count not in {5, 8} or len(unique) <= count:
        raise ValueError("G3R2 whole-seed cross-fit is underspecified")
    return tuple(tuple(unique[index::count]) for index in range(count))


def _resolved_partition(
    seeds: Sequence[int], config: G3R2Config
) -> tuple[tuple[int, ...], ...]:
    inventory = tuple(sorted(set(int(seed) for seed in seeds)))
    if config.seed_partition:
        supplied = tuple(config.seed_partition)
        if tuple(sorted(seed for fold in supplied for seed in fold)) != inventory:
            raise ValueError("G3R2 explicit seed partition does not cover training seeds")
        return supplied
    if config.folds != 5:
        raise ValueError("G3R2 production cross-fit requires an explicit partition")
    return whole_seed_folds(inventory, config.folds)


def _partition_sha256(
    partition_id: str, partition: Sequence[Sequence[int]]
) -> str:
    return _sha256(
        {
            "schema": "irisu-r3i-whole-seed-partition-g3r2-v1",
            "partition_id": partition_id,
            "folds": [list(fold) for fold in partition],
        }
    )


@dataclass(frozen=True, slots=True)
class G3R2Fold:
    heldout_seeds: tuple[int, ...]
    candidate: HistogramNewtonBoost
    pair: HistogramNewtonBoost

    def manifest(self) -> dict[str, object]:
        return {
            "heldout_seeds": list(self.heldout_seeds),
            "candidate": self.candidate.manifest(),
            "pair": self.pair.manifest(),
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> G3R2Fold:
        try:
            heldout = tuple(
                _integer(item, "heldout seed")
                for item in _sequence(value["heldout_seeds"], "heldout seeds")
            )
            result = cls(
                heldout,
                HistogramNewtonBoost.from_manifest(
                    _mapping(value["candidate"], "candidate head")
                ),
                HistogramNewtonBoost.from_manifest(
                    _mapping(value["pair"], "pair head")
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("G3R2 fold manifest is malformed") from exc
        if (
            not heldout
            or tuple(sorted(set(heldout))) != heldout
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("G3R2 fold manifest is malformed")
        return result


@dataclass(frozen=True, slots=True)
class G3R2Prediction:
    seed: int
    query_id: str
    ordinal: int
    candidate_mean: float
    candidate_std: float
    pair_fraction_mean: float
    pair_fraction_std: float
    primary_score: float


@dataclass(frozen=True, slots=True)
class ResolutionFirstG3R2:
    config: G3R2Config
    folds: tuple[G3R2Fold, ...]
    fold_by_seed: tuple[tuple[int, int], ...]
    training_seeds: tuple[int, ...]
    partition_sha256: str
    training_dataset_sha256: str

    @property
    def seed_folds(self) -> dict[int, int]:
        return dict(self.fold_by_seed)

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3i-resolution-first-g3r2-v1",
            "feature_names": list(WIDE_FEATURE_NAMES),
            "feature_width": WIDE_FEATURE_WIDTH,
            "preserved_block_width": PRESERVED_BLOCK_WIDTH,
            "training_dataset_sha256": self.training_dataset_sha256,
            "training_seeds": list(self.training_seeds),
            "partition_sha256": self.partition_sha256,
            "primary_target": "frozen-g2-candidate_resolved",
            "primary_score": "candidate-direct-depth3-newton",
            "pair_auxiliary": "invariant-pair-resolved-fraction",
            "group_keys_are_inputs": False,
            "absolute_identity_inputs": False,
            "cross_fit_unit": "whole-seed",
            "fold_by_seed": [list(value) for value in self.fold_by_seed],
            "folds": [fold.manifest() for fold in self.folds],
            "config": {
                "folds": self.config.folds,
                "blend_pair_weight": self.config.blend_pair_weight,
                "partition_id": self.config.partition_id,
                "seed_partition": [
                    list(fold) for fold in self.config.seed_partition
                ],
                "candidate": asdict(self.config.candidate),
                "pair": asdict(self.config.pair),
            },
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> ResolutionFirstG3R2:
        if (
            value.get("schema") != "irisu-r3i-resolution-first-g3r2-v1"
            or tuple(value.get("feature_names", ())) != WIDE_FEATURE_NAMES
            or value.get("feature_width") != WIDE_FEATURE_WIDTH
            or value.get("preserved_block_width") != PRESERVED_BLOCK_WIDTH
            or value.get("primary_target") != "frozen-g2-candidate_resolved"
            or value.get("primary_score") != "candidate-direct-depth3-newton"
            or value.get("pair_auxiliary")
            != "invariant-pair-resolved-fraction"
            or value.get("group_keys_are_inputs") is not False
            or value.get("absolute_identity_inputs") is not False
            or value.get("cross_fit_unit") != "whole-seed"
        ):
            raise RuntimeError("G3R2 checkpoint feature identity mismatch")
        try:
            raw_config = dict(_mapping(value["config"], "G3R2 config"))
            if set(raw_config) != {
                "folds",
                "blend_pair_weight",
                "partition_id",
                "seed_partition",
                "candidate",
                "pair",
            }:
                raise RuntimeError("G3R2 configuration is malformed")
            config = G3R2Config(
                folds=_integer(raw_config["folds"], "fold count"),
                blend_pair_weight=_number(
                    raw_config["blend_pair_weight"], "pair blend"
                ),
                partition_id=str(raw_config["partition_id"]),
                seed_partition=tuple(
                    tuple(
                        _integer(seed, "partition seed")
                        for seed in _sequence(fold, "partition fold")
                    )
                    for fold in _sequence(
                        raw_config["seed_partition"], "seed partition"
                    )
                ),
                candidate=BoostConfig(
                    **dict(_mapping(raw_config["candidate"], "candidate config"))
                ),
                pair=BoostConfig(
                    **dict(_mapping(raw_config["pair"], "pair config"))
                ),
            )
            folds = tuple(
                G3R2Fold.from_manifest(_mapping(row, "G3R2 fold"))
                for row in _sequence(value["folds"], "G3R2 folds")
            )
            assignments = tuple(
                (
                    _integer(row[0], "fold seed"),
                    _integer(row[1], "fold index"),
                )
                for row in _sequence(value["fold_by_seed"], "fold mapping")
            )
            training_seeds = tuple(
                _integer(seed, "training seed")
                for seed in _sequence(value["training_seeds"], "training seeds")
            )
            partition_sha = str(value["partition_sha256"])
            dataset_sha = str(value["training_dataset_sha256"])
            result = cls(
                config,
                folds,
                assignments,
                training_seeds,
                partition_sha,
                dataset_sha,
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise RuntimeError("G3R2 checkpoint manifest is malformed") from exc
        expected = tuple(
            sorted(
                (seed, index)
                for index, fold in enumerate(folds)
                for seed in fold.heldout_seeds
            )
        )
        assigned_seeds = tuple(seed for seed, _index in assignments)
        try:
            exact_partition = _resolved_partition(training_seeds, config)
        except ValueError as exc:
            raise RuntimeError("G3R2 checkpoint partition is malformed") from exc
        if (
            len(folds) != config.folds
            or assignments != expected
            or len(set(assigned_seeds)) != len(assigned_seeds)
            or training_seeds != tuple(sorted(set(training_seeds)))
            or assigned_seeds != training_seeds
            or tuple(fold.heldout_seeds for fold in folds) != exact_partition
            or partition_sha
            != _partition_sha256(config.partition_id, exact_partition)
            or not _SHA256_RE.fullmatch(partition_sha)
            or not _SHA256_RE.fullmatch(dataset_sha)
            or any(
                fold.candidate.config != config.candidate
                or fold.pair.config != config.pair
                for fold in folds
            )
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("G3R2 checkpoint manifest is malformed")
        return result

    def checkpoint(
        self, metadata: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        body = {
            "schema": "irisu-r3i-resolution-first-checkpoint-g3r2-v1",
            "model_sha256": self.sha256,
            "model": self.manifest(),
            "metadata": dict(metadata or {}),
        }
        return {**body, "checkpoint_sha256": _sha256(body)}

    @classmethod
    def from_checkpoint(
        cls, value: Mapping[str, Any]
    ) -> tuple[ResolutionFirstG3R2, dict[str, object]]:
        if set(value) != {
            "schema",
            "model_sha256",
            "model",
            "metadata",
            "checkpoint_sha256",
        }:
            raise RuntimeError("G3R2 checkpoint envelope is malformed")
        body = {key: value[key] for key in value if key != "checkpoint_sha256"}
        if (
            value.get("schema")
            != "irisu-r3i-resolution-first-checkpoint-g3r2-v1"
            or value.get("checkpoint_sha256") != _sha256(body)
        ):
            raise RuntimeError("G3R2 checkpoint identity mismatch")
        model = cls.from_manifest(_mapping(value["model"], "G3R2 model"))
        if value.get("model_sha256") != model.sha256:
            raise RuntimeError("G3R2 checkpoint model identity mismatch")
        metadata = value["metadata"]
        if not isinstance(metadata, Mapping):
            raise RuntimeError("G3R2 checkpoint metadata is malformed")
        return model, dict(metadata)

    def predict(
        self, board: WideBoard, *, oof: bool = False
    ) -> tuple[G3R2Prediction, ...]:
        pair_groups = PairGroups.from_board(board)
        known = self.seed_folds
        candidate_members: list[list[float]] = [[] for _ in board.labels]
        pair_members: list[list[float]] = [[] for _ in board.labels]
        for fold_index, fold in enumerate(self.folds):
            candidate_probability = fold.candidate.probabilities(board.features)
            pair_probability = pair_groups.expand(
                fold.pair.probabilities(pair_groups.features), len(board.labels)
            )
            for index, seed in enumerate(board.seeds):
                if oof:
                    if int(seed) not in known:
                        raise ValueError(
                            "OOF prediction requested for an unknown seed"
                        )
                    if known[int(seed)] != fold_index:
                        continue
                candidate_members[index].append(float(candidate_probability[index]))
                pair_members[index].append(float(pair_probability[index]))
        predictions: list[G3R2Prediction] = []
        pair_weight = self.config.blend_pair_weight
        for index in range(len(board.labels)):
            if not candidate_members[index] or not pair_members[index]:
                raise RuntimeError("G3R2 prediction has no eligible fold")
            candidate_mean = float(np.mean(candidate_members[index]))
            pair_mean = float(np.mean(pair_members[index]))
            predictions.append(
                G3R2Prediction(
                    int(board.seeds[index]),
                    str(board.query_ids[index]),
                    int(board.ordinals[index]),
                    candidate_mean,
                    float(np.std(candidate_members[index])),
                    pair_mean,
                    float(np.std(pair_members[index])),
                    (1.0 - pair_weight) * candidate_mean
                    + pair_weight * pair_mean,
                )
            )
        return tuple(predictions)


def train_resolution_first_g3r2(
    board: WideBoard, *, config: G3R2Config | None = None
) -> ResolutionFirstG3R2:
    resolved = G3R2Config() if config is None else config
    partitions = _resolved_partition(board.unique_seeds, resolved)
    pair_groups = PairGroups.from_board(board)
    all_seeds = set(board.unique_seeds)
    folds: list[G3R2Fold] = []
    for heldout in partitions:
        training_seeds = all_seeds - set(heldout)
        candidate_mask = np.asarray(
            [int(seed) in training_seeds for seed in board.seeds]
        )
        pair_mask = np.asarray(
            [int(seed) in training_seeds for seed in pair_groups.seeds]
        )
        folds.append(
            G3R2Fold(
                heldout,
                HistogramNewtonBoost.fit(
                    board.features[candidate_mask],
                    board.labels[candidate_mask].astype(np.float64),
                    board.seeds[candidate_mask],
                    resolved.candidate,
                ),
                HistogramNewtonBoost.fit(
                    pair_groups.features[pair_mask],
                    pair_groups.targets[pair_mask],
                    pair_groups.seeds[pair_mask],
                    resolved.pair,
                ),
            )
        )
    assignments = tuple(
        sorted(
            (seed, fold_index)
            for fold_index, heldout in enumerate(partitions)
            for seed in heldout
        )
    )
    return ResolutionFirstG3R2(
        resolved,
        tuple(folds),
        assignments,
        board.unique_seeds,
        _partition_sha256(resolved.partition_id, partitions),
        board.sha256,
    )


def save_checkpoint_g3r2(
    path: str | os.PathLike[str],
    model: ResolutionFirstG3R2,
    *,
    metadata: Mapping[str, object] | None = None,
) -> str:
    destination = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(destination)
    if (
        os.path.lexists(destination)
        or not os.path.isdir(parent)
        or os.path.realpath(parent) != parent
    ):
        raise RuntimeError("G3R2 checkpoint destination is occupied or indirect")
    encoded = _canonical_bytes(model.checkpoint(metadata))
    temporary = os.path.join(
        parent, f".{os.path.basename(destination)}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    linked = False
    try:
        with open(temporary, "xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        linked = True
        descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    if not linked:
        raise RuntimeError("G3R2 checkpoint creation failed")
    return hashlib.sha256(encoded).hexdigest()


def load_checkpoint_g3r2(
    path: str | os.PathLike[str],
    *,
    expected_metadata: Mapping[str, object] | None = None,
    expected_partition_sha256: str | None = None,
    expected_model_sha256: str | None = None,
) -> tuple[ResolutionFirstG3R2, dict[str, object]]:
    source = os.path.abspath(os.fspath(path))
    before = os.lstat(source)
    if (
        not os.path.isfile(source)
        or os.path.islink(source)
        or os.path.realpath(source) != source
        or before.st_nlink != 1
    ):
        raise RuntimeError("G3R2 checkpoint source is indirect")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError("G3R2 checkpoint descriptor identity changed")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        after_descriptor = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after_path = os.lstat(source)
    identity_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(before, name)
        != getattr(opened, name)
        or getattr(opened, name) != getattr(after_descriptor, name)
        or getattr(after_descriptor, name) != getattr(after_path, name)
        for name in identity_fields
    ):
        raise RuntimeError("G3R2 checkpoint changed while being read")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("G3R2 checkpoint JSON is malformed") from exc
    if not isinstance(payload, Mapping) or raw != _canonical_bytes(payload):
        raise RuntimeError("G3R2 checkpoint encoding is noncanonical")
    model, metadata = ResolutionFirstG3R2.from_checkpoint(payload)
    if expected_metadata is not None and metadata != dict(expected_metadata):
        raise RuntimeError("G3R2 checkpoint metadata expectation mismatch")
    if (
        expected_partition_sha256 is not None
        and model.partition_sha256 != expected_partition_sha256
    ):
        raise RuntimeError("G3R2 checkpoint partition expectation mismatch")
    if expected_model_sha256 is not None and model.sha256 != expected_model_sha256:
        raise RuntimeError("G3R2 checkpoint model expectation mismatch")
    return model, metadata
