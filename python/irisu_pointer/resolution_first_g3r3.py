"""Hardened generation-03 revision-03 relational resolution learner.

G3R3 preserves the G3R2 712-column, identity-free scientific representation
while replacing its learner, manifests, and checkpoint I/O.  Every seed has
equal optimization and screening mass, including under within-seed class
balancing.  Checkpoint loading and publication finish on descriptor-relative
identity checks after all lexical path checks.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .resolution_first_g3r2 import (
    PRESERVED_BLOCK_WIDTH,
    WIDE_FEATURE_NAMES,
    WIDE_FEATURE_WIDTH,
    PairGroups,
    WideBoard,
    wide_board_from_entries,
    wide_board_from_records,
)


MODEL_SCHEMA = "irisu-r3i-resolution-first-g3r3-v1"
CHECKPOINT_SCHEMA = "irisu-r3i-resolution-first-checkpoint-g3r3-v1"
PARTITION_SCHEMA = "irisu-r3i-whole-seed-partition-g3r3-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DEPENDENCY_RE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_SAFE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}\Z")
_FILE_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
_DIRECTORY_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)


def _validate_json(value: Any, field: str = "$") -> None:
    kind = type(value)
    if value is None or kind in (bool, int, str):
        if kind is str:
            try:
                value.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise RuntimeError(f"{field} is not UTF-8") from exc
        return
    if kind is float:
        if not math.isfinite(value):
            raise RuntimeError(f"{field} is non-finite")
        return
    if kind is list:
        for index, item in enumerate(value):
            _validate_json(item, f"{field}[{index}]")
        return
    if kind is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise RuntimeError(f"{field} has a non-string key")
            _validate_json(item, f"{field}.{key}")
        return
    raise RuntimeError(f"{field} has unsupported type {kind.__name__}")


def _canonical_bytes(value: Any) -> bytes:
    _validate_json(value)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _exact_dict(value: Any, field: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise RuntimeError(f"{field} is malformed")
    return value


def _exact_list(value: Any, field: str) -> list[Any]:
    if type(value) is not list:
        raise RuntimeError(f"{field} is malformed")
    return value


def _exact_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise RuntimeError(f"{field} must be an exact non-bool integer")
    return value


def _exact_finite_float(value: Any, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise RuntimeError(f"{field} is malformed")
    return value


def _require_sha256(value: Any, field: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeError(f"{field} must be a lowercase SHA-256")
    return value


def _canonical_dict(value: Any, field: str) -> dict[str, Any]:
    supplied = _exact_dict(value, field)
    return json.loads(_canonical_bytes(supplied).decode("utf-8"))


def _readonly(value: np.ndarray, dtype: Any) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def seed_equal_weights(
    seeds: np.ndarray,
    targets: np.ndarray,
    balance_classes: bool,
) -> np.ndarray:
    """Return mean-one row weights with exactly equal total mass per seed."""

    if type(balance_classes) is not bool:
        raise ValueError("balance_classes must be exact bool")
    seed = np.asarray(seeds)
    target = np.asarray(targets, dtype=np.float64)
    if (
        seed.ndim != 1
        or target.shape != seed.shape
        or len(seed) == 0
        or not np.issubdtype(seed.dtype, np.integer)
        or np.issubdtype(seed.dtype, np.bool_)
        or np.any(seed < 0)
        or not np.isfinite(target).all()
    ):
        raise ValueError("G3R3 seed weighting supervision is malformed")
    if balance_classes and set(np.unique(target)) != {0.0, 1.0}:
        raise ValueError("class-balanced G3R3 head requires two classes")

    unique = np.unique(seed)
    weights = np.zeros(len(seed), dtype=np.float64)
    for seed_value in unique:
        rows = np.flatnonzero(seed == seed_value)
        if balance_classes:
            positives = rows[target[rows] == 1.0]
            negatives = rows[target[rows] == 0.0]
            if len(positives) and len(negatives):
                weights[positives] = 0.5 / len(positives)
                weights[negatives] = 0.5 / len(negatives)
            else:
                weights[rows] = 1.0 / len(rows)
        else:
            weights[rows] = 1.0 / len(rows)
    weights *= len(seed) / len(unique)
    expected_mass = len(seed) / len(unique)
    masses = np.asarray([weights[seed == value].sum() for value in unique])
    if not np.allclose(masses, expected_mass, rtol=0.0, atol=1e-12):
        raise RuntimeError("G3R3 failed to equalize seed loss mass")
    return weights


@dataclass(frozen=True, slots=True)
class BoostConfigR3:
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
        integers = (
            self.rounds,
            self.depth,
            self.minimum_leaf,
            self.maximum_features,
            self.preserved_features,
            self.bins,
        )
        if (
            any(type(value) is not int for value in integers)
            or type(self.balance_classes) is not bool
            or type(self.learning_rate) is not float
            or type(self.l2) is not float
            or min(integers) < 1
            or self.preserved_features > self.maximum_features
            or self.maximum_features > WIDE_FEATURE_WIDTH
            or not 0.0 < self.learning_rate <= 1.0
            or not math.isfinite(self.learning_rate)
            or not math.isfinite(self.l2)
            or self.l2 <= 0.0
        ):
            raise ValueError("G3R3 boost configuration is invalid")

    def manifest(self) -> dict[str, Any]:
        return {
            "rounds": self.rounds,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "l2": self.l2,
            "minimum_leaf": self.minimum_leaf,
            "maximum_features": self.maximum_features,
            "preserved_features": self.preserved_features,
            "bins": self.bins,
            "balance_classes": self.balance_classes,
        }

    @classmethod
    def from_manifest(cls, value: Any) -> BoostConfigR3:
        raw = _exact_dict(value, "boost config")
        required = {
            "rounds",
            "depth",
            "learning_rate",
            "l2",
            "minimum_leaf",
            "maximum_features",
            "preserved_features",
            "bins",
            "balance_classes",
        }
        if set(raw) != required:
            raise RuntimeError("G3R3 boost configuration is malformed")
        try:
            result = cls(
                rounds=_exact_int(raw["rounds"], "rounds"),
                depth=_exact_int(raw["depth"], "depth"),
                learning_rate=_exact_finite_float(
                    raw["learning_rate"], "learning_rate"
                ),
                l2=_exact_finite_float(raw["l2"], "l2"),
                minimum_leaf=_exact_int(raw["minimum_leaf"], "minimum_leaf"),
                maximum_features=_exact_int(
                    raw["maximum_features"], "maximum_features"
                ),
                preserved_features=_exact_int(
                    raw["preserved_features"], "preserved_features"
                ),
                bins=_exact_int(raw["bins"], "bins"),
                balance_classes=raw["balance_classes"],
            )
        except ValueError as exc:
            raise RuntimeError("G3R3 boost configuration is malformed") from exc
        if (
            type(raw["balance_classes"]) is not bool
            or _canonical_bytes(result.manifest()) != _canonical_bytes(raw)
        ):
            raise RuntimeError("G3R3 boost configuration is malformed")
        return result


def _weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: np.ndarray
) -> tuple[float, ...]:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    if not len(cumulative) or cumulative[-1] <= 0.0:
        raise ValueError("G3R3 weighted quantiles have no mass")
    indices = np.searchsorted(
        cumulative, quantiles * cumulative[-1], side="left"
    )
    indices = np.clip(indices, 0, len(ordered_values) - 1)
    return tuple(float(value) for value in np.unique(ordered_values[indices]))


@dataclass(frozen=True, slots=True)
class FeatureBinnerR3:
    raw_feature_count: int
    selected_columns: tuple[int, ...]
    thresholds: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if (
            type(self.raw_feature_count) is not int
            or type(self.selected_columns) is not tuple
            or type(self.thresholds) is not tuple
            or self.raw_feature_count != WIDE_FEATURE_WIDTH
            or not self.selected_columns
            or len(self.selected_columns) != len(self.thresholds)
            or len(set(self.selected_columns)) != len(self.selected_columns)
            or any(
                type(column) is not int
                or not 0 <= column < self.raw_feature_count
                for column in self.selected_columns
            )
            or any(
                type(row) is not tuple
                or any(type(value) is not float or not math.isfinite(value) for value in row)
                or tuple(sorted(set(row))) != row
                for row in self.thresholds
            )
        ):
            raise ValueError("G3R3 binner is malformed")

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        targets: np.ndarray,
        seeds: np.ndarray,
        config: BoostConfigR3,
    ) -> FeatureBinnerR3:
        x = np.asarray(values, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        seed = np.asarray(seeds)
        if (
            x.ndim != 2
            or x.shape[1] != WIDE_FEATURE_WIDTH
            or y.shape != (len(x),)
            or seed.shape != y.shape
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
        ):
            raise ValueError("G3R3 screening supervision is malformed")
        weights = seed_equal_weights(seed, y, config.balance_classes)
        total = float(weights.sum())
        target_mean = float(np.dot(weights, y) / total)
        centered_target = y - target_mean
        target_variance = float(np.dot(weights, centered_target**2))
        correlations: list[float] = []
        for column in range(x.shape[1]):
            candidate = x[:, column]
            mean = float(np.dot(weights, candidate) / total)
            centered = candidate - mean
            denominator = math.sqrt(
                float(np.dot(weights, centered**2)) * target_variance
            )
            correlations.append(
                0.0
                if denominator <= 0.0
                else abs(float(np.dot(weights, centered * centered_target)))
                / denominator
            )
        selected = list(range(config.preserved_features))
        selected.extend(
            column
            for column in sorted(
                range(x.shape[1]), key=lambda item: (-correlations[item], item)
            )
            if column not in selected
        )
        selected = selected[: config.maximum_features]
        quantiles = np.linspace(0.0, 1.0, config.bins + 1)[1:-1]
        thresholds = tuple(
            _weighted_quantiles(x[:, column], weights, quantiles)
            for column in selected
        )
        return cls(WIDE_FEATURE_WIDTH, tuple(selected), thresholds)

    def transform(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if (
            x.ndim != 2
            or x.shape[1] != self.raw_feature_count
            or not np.isfinite(x).all()
        ):
            raise ValueError("G3R3 binner input is malformed")
        output = np.empty((len(x), len(self.selected_columns)), dtype=np.uint16)
        for target, (source, thresholds) in enumerate(
            zip(self.selected_columns, self.thresholds, strict=True)
        ):
            output[:, target] = np.searchsorted(
                np.asarray(thresholds), x[:, source]
            )
        return output

    @property
    def widths(self) -> np.ndarray:
        return np.asarray([len(row) for row in self.thresholds], dtype=np.int64)

    def manifest(self) -> dict[str, Any]:
        return {
            "raw_feature_count": self.raw_feature_count,
            "selected_columns": list(self.selected_columns),
            "thresholds": [list(row) for row in self.thresholds],
        }

    @classmethod
    def from_manifest(cls, value: Any) -> FeatureBinnerR3:
        raw = _exact_dict(value, "feature binner")
        if set(raw) != {"raw_feature_count", "selected_columns", "thresholds"}:
            raise RuntimeError("G3R3 binner manifest is malformed")
        try:
            result = cls(
                _exact_int(raw["raw_feature_count"], "raw_feature_count"),
                tuple(
                    _exact_int(item, "selected column")
                    for item in _exact_list(
                        raw["selected_columns"], "selected_columns"
                    )
                ),
                tuple(
                    tuple(
                        _exact_finite_float(item, "threshold")
                        for item in _exact_list(row, "threshold row")
                    )
                    for row in _exact_list(raw["thresholds"], "thresholds")
                ),
            )
        except ValueError as exc:
            raise RuntimeError("G3R3 binner manifest is malformed") from exc
        if _canonical_bytes(result.manifest()) != _canonical_bytes(raw):
            raise RuntimeError("G3R3 binner manifest is malformed")
        return result


@dataclass(frozen=True, slots=True)
class TreeNodeR3:
    feature: int
    threshold: int
    value: float
    left: TreeNodeR3 | None = None
    right: TreeNodeR3 | None = None

    def __post_init__(self) -> None:
        leaf = self.feature == -1
        if (
            type(self.feature) is not int
            or type(self.threshold) is not int
            or type(self.value) is not float
            or not math.isfinite(self.value)
            or (
                leaf
                and (
                    self.threshold != -1
                    or self.left is not None
                    or self.right is not None
                )
            )
            or (
                not leaf
                and (
                    self.feature < 0
                    or self.threshold < 0
                    or type(self.left) is not TreeNodeR3
                    or type(self.right) is not TreeNodeR3
                )
            )
        ):
            raise ValueError("G3R3 tree node is malformed")

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
        stack: list[tuple[TreeNodeR3, np.ndarray]] = [
            (self, np.arange(len(bins), dtype=np.int64))
        ]
        while stack:
            node, rows = stack.pop()
            if node.is_leaf:
                output[rows] = node.value
            else:
                assert node.left is not None and node.right is not None
                branch = bins[rows, node.feature] <= node.threshold
                stack.append((node.left, rows[branch]))
                stack.append((node.right, rows[~branch]))
        return output

    def manifest(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "threshold": self.threshold,
            "value": self.value,
            "left": None if self.left is None else self.left.manifest(),
            "right": None if self.right is None else self.right.manifest(),
        }

    @classmethod
    def from_manifest(cls, value: Any) -> TreeNodeR3:
        raw = _exact_dict(value, "tree node")
        if set(raw) != {"feature", "threshold", "value", "left", "right"}:
            raise RuntimeError("G3R3 tree manifest is malformed")
        try:
            feature = _exact_int(raw["feature"], "tree feature")
            threshold = _exact_int(raw["threshold"], "tree threshold")
            node_value = _exact_finite_float(raw["value"], "tree value")
            result = (
                cls(feature, threshold, node_value)
                if feature == -1
                else cls(
                    feature,
                    threshold,
                    node_value,
                    cls.from_manifest(raw["left"]),
                    cls.from_manifest(raw["right"]),
                )
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("G3R3 tree manifest is malformed") from exc
        if _canonical_bytes(result.manifest()) != _canonical_bytes(raw):
            raise RuntimeError("G3R3 tree manifest is malformed")
        return result


def _walk_tree(root: TreeNodeR3) -> tuple[TreeNodeR3, ...]:
    output: list[TreeNodeR3] = []
    stack = [root]
    while stack:
        node = stack.pop()
        output.append(node)
        if not node.is_leaf:
            assert node.left is not None and node.right is not None
            stack.extend((node.left, node.right))
    return tuple(output)


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
) -> TreeNodeR3:
    total_gradient = float(gradient[rows].sum())
    total_hessian = float(hessian[rows].sum())
    leaf = float(-total_gradient / (total_hessian + l2))
    if depth == 0 or len(rows) < 2 * minimum_leaf:
        return TreeNodeR3(-1, -1, leaf)
    parent = total_gradient * total_gradient / (total_hessian + l2)
    best_gain = -math.inf
    best_split: tuple[int, int] | None = None
    for feature, width_value in enumerate(widths):
        width = int(width_value)
        if width < 1:
            continue
        count = np.bincount(bins[rows, feature], minlength=width + 1)
        grad = np.bincount(
            bins[rows, feature], weights=gradient[rows], minlength=width + 1
        )
        hess = np.bincount(
            bins[rows, feature], weights=hessian[rows], minlength=width + 1
        )
        left_count = np.cumsum(count)[:-1]
        left_gradient = np.cumsum(grad)[:-1]
        left_hessian = np.cumsum(hess)[:-1]
        valid = (left_count >= minimum_leaf) & (
            len(rows) - left_count >= minimum_leaf
        )
        gain = np.where(
            valid,
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
        return TreeNodeR3(-1, -1, leaf)
    feature, threshold = best_split
    branch = bins[rows, feature] <= threshold
    return TreeNodeR3(
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


@dataclass(frozen=True, slots=True)
class HistogramNewtonBoostR3:
    config: BoostConfigR3
    binner: FeatureBinnerR3
    base_logit: float
    trees: tuple[TreeNodeR3, ...]

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        targets: np.ndarray,
        seeds: np.ndarray,
        config: BoostConfigR3,
    ) -> HistogramNewtonBoostR3:
        x = np.asarray(values, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        seed = np.asarray(seeds)
        if (
            x.shape != (len(y), WIDE_FEATURE_WIDTH)
            or seed.shape != y.shape
            or len(y) == 0
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or np.any((y < 0.0) | (y > 1.0))
        ):
            raise ValueError("G3R3 boost supervision is malformed")
        weights = seed_equal_weights(seed, y, config.balance_classes)
        binner = FeatureBinnerR3.fit(x, y, seed, config)
        bins = binner.transform(x)
        prior = float(np.clip(np.average(y, weights=weights), 0.01, 0.99))
        base = float(math.log(prior / (1.0 - prior)))
        score = np.full(len(y), base, dtype=np.float64)
        rows = np.arange(len(y), dtype=np.int64)
        trees: list[TreeNodeR3] = []
        for _ in range(config.rounds):
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

    def manifest(self) -> dict[str, Any]:
        return {
            "config": self.config.manifest(),
            "binner": self.binner.manifest(),
            "base_logit": self.base_logit,
            "trees": [tree.manifest() for tree in self.trees],
        }

    @classmethod
    def from_manifest(cls, value: Any) -> HistogramNewtonBoostR3:
        raw = _exact_dict(value, "boost model")
        if set(raw) != {"config", "binner", "base_logit", "trees"}:
            raise RuntimeError("G3R3 boost manifest is malformed")
        try:
            config = BoostConfigR3.from_manifest(raw["config"])
            binner = FeatureBinnerR3.from_manifest(raw["binner"])
            base = _exact_finite_float(raw["base_logit"], "base_logit")
            trees = tuple(
                TreeNodeR3.from_manifest(row)
                for row in _exact_list(raw["trees"], "trees")
            )
            result = cls(config, binner, base, trees)
        except ValueError as exc:
            raise RuntimeError("G3R3 boost manifest is malformed") from exc
        widths = binner.widths
        if (
            len(trees) != config.rounds
            or len(binner.selected_columns) != config.maximum_features
            or binner.selected_columns[: config.preserved_features]
            != tuple(range(config.preserved_features))
            or any(len(row) > config.bins - 1 for row in binner.thresholds)
            or any(
                tree.depth > config.depth
                or any(
                    (
                        not node.is_leaf
                        and (
                            node.feature < 0
                            or node.feature >= len(binner.selected_columns)
                            or node.threshold < 0
                            or node.threshold >= int(widths[node.feature])
                        )
                    )
                    for node in _walk_tree(tree)
                )
                for tree in trees
            )
            or _canonical_bytes(result.manifest()) != _canonical_bytes(raw)
        ):
            raise RuntimeError("G3R3 boost manifest is malformed")
        return result


@dataclass(frozen=True, slots=True)
class G3R3Config:
    folds: int = 5
    blend_pair_weight: float = 0.0
    partition_id: str = "development-internal-sorted-g3r3-v1"
    seed_partition: tuple[tuple[int, ...], ...] = ()
    candidate: BoostConfigR3 = BoostConfigR3(rounds=200)
    pair: BoostConfigR3 = BoostConfigR3(
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
        if (
            type(self.folds) is not int
            or type(self.blend_pair_weight) is not float
            or type(self.partition_id) is not str
            or type(self.seed_partition) is not tuple
            or type(self.candidate) is not BoostConfigR3
            or type(self.pair) is not BoostConfigR3
        ):
            raise ValueError("G3R3 cross-fit configuration has inexact types")
        if any(
            type(fold) is not tuple
            or not fold
            or any(type(seed) is not int or seed < 0 for seed in fold)
            for fold in self.seed_partition
        ):
            raise ValueError("G3R3 seed partition has inexact seeds")
        flattened = tuple(seed for fold in self.seed_partition for seed in fold)
        if (
            self.folds not in {5, 8}
            or not math.isfinite(self.blend_pair_weight)
            or not 0.0 <= self.blend_pair_weight <= 1.0
            or not self.partition_id
            or (self.folds == 8 and not self.seed_partition)
            or (
                self.seed_partition
                and (
                    len(self.seed_partition) != self.folds
                    or any(tuple(sorted(set(fold))) != fold for fold in self.seed_partition)
                    or len(set(flattened)) != len(flattened)
                    or self.partition_id
                    == "development-internal-sorted-g3r3-v1"
                )
            )
        ):
            raise ValueError("G3R3 cross-fit configuration is invalid")

    def manifest(self) -> dict[str, Any]:
        return {
            "folds": self.folds,
            "blend_pair_weight": self.blend_pair_weight,
            "partition_id": self.partition_id,
            "seed_partition": [list(fold) for fold in self.seed_partition],
            "candidate": self.candidate.manifest(),
            "pair": self.pair.manifest(),
        }

    @classmethod
    def from_manifest(cls, value: Any) -> G3R3Config:
        raw = _exact_dict(value, "G3R3 config")
        if set(raw) != {
            "folds",
            "blend_pair_weight",
            "partition_id",
            "seed_partition",
            "candidate",
            "pair",
        }:
            raise RuntimeError("G3R3 configuration is malformed")
        if type(raw["partition_id"]) is not str:
            raise RuntimeError("G3R3 configuration is malformed")
        try:
            result = cls(
                folds=_exact_int(raw["folds"], "folds"),
                blend_pair_weight=_exact_finite_float(
                    raw["blend_pair_weight"], "blend_pair_weight"
                ),
                partition_id=raw["partition_id"],
                seed_partition=tuple(
                    tuple(
                        _exact_int(seed, "partition seed")
                        for seed in _exact_list(fold, "partition fold")
                    )
                    for fold in _exact_list(
                        raw["seed_partition"], "seed_partition"
                    )
                ),
                candidate=BoostConfigR3.from_manifest(raw["candidate"]),
                pair=BoostConfigR3.from_manifest(raw["pair"]),
            )
        except ValueError as exc:
            raise RuntimeError("G3R3 configuration is malformed") from exc
        if _canonical_bytes(result.manifest()) != _canonical_bytes(raw):
            raise RuntimeError("G3R3 configuration is malformed")
        return result


def _exact_seeds(seeds: Any, field: str) -> tuple[int, ...]:
    if type(seeds) not in (list, tuple):
        raise ValueError(f"{field} must be an exact list or tuple")
    result = tuple(seeds)
    if (
        not result
        or any(type(seed) is not int or seed < 0 for seed in result)
        or len(result) != len(set(result))
    ):
        raise ValueError(f"{field} must contain unique exact non-bool integers")
    return result


def whole_seed_folds_g3r3(
    seeds: Sequence[int], count: int
) -> tuple[tuple[int, ...], ...]:
    if type(count) is not int or count not in {5, 8}:
        raise ValueError("G3R3 fold count must be an exact supported integer")
    unique = tuple(sorted(_exact_seeds(seeds, "seeds")))
    if len(unique) <= count:
        raise ValueError("G3R3 whole-seed cross-fit is underspecified")
    return tuple(tuple(unique[index::count]) for index in range(count))


def _resolved_partition(
    seeds: Sequence[int], config: G3R3Config
) -> tuple[tuple[int, ...], ...]:
    inventory = tuple(sorted(_exact_seeds(seeds, "training seeds")))
    if config.seed_partition:
        supplied = config.seed_partition
        if tuple(sorted(seed for fold in supplied for seed in fold)) != inventory:
            raise ValueError("G3R3 explicit partition does not cover training seeds")
        return supplied
    if config.folds != 5:
        raise ValueError("G3R3 production cross-fit requires explicit partition")
    return whole_seed_folds_g3r3(inventory, config.folds)


def _partition_sha256(
    partition_id: str, partition: Sequence[Sequence[int]]
) -> str:
    return _sha256(
        {
            "schema": PARTITION_SCHEMA,
            "partition_id": partition_id,
            "folds": [list(fold) for fold in partition],
        }
    )


@dataclass(frozen=True, slots=True)
class G3R3Fold:
    heldout_seeds: tuple[int, ...]
    candidate: HistogramNewtonBoostR3
    pair: HistogramNewtonBoostR3

    def __post_init__(self) -> None:
        if (
            type(self.heldout_seeds) is not tuple
            or not self.heldout_seeds
            or any(type(seed) is not int for seed in self.heldout_seeds)
            or tuple(sorted(set(self.heldout_seeds))) != self.heldout_seeds
            or type(self.candidate) is not HistogramNewtonBoostR3
            or type(self.pair) is not HistogramNewtonBoostR3
        ):
            raise ValueError("G3R3 fold is malformed")

    def manifest(self) -> dict[str, Any]:
        return {
            "heldout_seeds": list(self.heldout_seeds),
            "candidate": self.candidate.manifest(),
            "pair": self.pair.manifest(),
        }

    @classmethod
    def from_manifest(cls, value: Any) -> G3R3Fold:
        raw = _exact_dict(value, "G3R3 fold")
        if set(raw) != {"heldout_seeds", "candidate", "pair"}:
            raise RuntimeError("G3R3 fold manifest is malformed")
        try:
            result = cls(
                tuple(
                    _exact_int(seed, "heldout seed")
                    for seed in _exact_list(
                        raw["heldout_seeds"], "heldout_seeds"
                    )
                ),
                HistogramNewtonBoostR3.from_manifest(raw["candidate"]),
                HistogramNewtonBoostR3.from_manifest(raw["pair"]),
            )
        except ValueError as exc:
            raise RuntimeError("G3R3 fold manifest is malformed") from exc
        if _canonical_bytes(result.manifest()) != _canonical_bytes(raw):
            raise RuntimeError("G3R3 fold manifest is malformed")
        return result


@dataclass(frozen=True, slots=True)
class G3R3Prediction:
    seed: int
    query_id: str
    ordinal: int
    candidate_mean: float
    candidate_std: float
    pair_fraction_mean: float
    pair_fraction_std: float
    primary_score: float


@dataclass(frozen=True, slots=True)
class ResolutionFirstG3R3:
    config: G3R3Config
    folds: tuple[G3R3Fold, ...]
    fold_by_seed: tuple[tuple[int, int], ...]
    training_seeds: tuple[int, ...]
    partition_sha256: str
    training_dataset_sha256: str

    @property
    def seed_folds(self) -> dict[int, int]:
        return dict(self.fold_by_seed)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": MODEL_SCHEMA,
            "feature_names": list(WIDE_FEATURE_NAMES),
            "feature_width": WIDE_FEATURE_WIDTH,
            "preserved_block_width": PRESERVED_BLOCK_WIDTH,
            "training_dataset_sha256": self.training_dataset_sha256,
            "training_seeds": list(self.training_seeds),
            "partition_sha256": self.partition_sha256,
            "primary_target": "frozen-g2-candidate_resolved",
            "primary_score": "candidate-direct-depth3-seed-equal-newton",
            "pair_auxiliary": "invariant-pair-resolved-fraction",
            "screening": "train-only-seed-equal-weighted-correlation",
            "quantiles": "train-only-seed-equal-weighted-empirical",
            "group_keys_are_inputs": False,
            "absolute_identity_inputs": False,
            "cross_fit_unit": "whole-seed",
            "fold_by_seed": [list(row) for row in self.fold_by_seed],
            "folds": [fold.manifest() for fold in self.folds],
            "config": self.config.manifest(),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())

    @classmethod
    def from_manifest(cls, value: Any) -> ResolutionFirstG3R3:
        raw = _exact_dict(value, "G3R3 model")
        required = {
            "schema",
            "feature_names",
            "feature_width",
            "preserved_block_width",
            "training_dataset_sha256",
            "training_seeds",
            "partition_sha256",
            "primary_target",
            "primary_score",
            "pair_auxiliary",
            "screening",
            "quantiles",
            "group_keys_are_inputs",
            "absolute_identity_inputs",
            "cross_fit_unit",
            "fold_by_seed",
            "folds",
            "config",
        }
        if (
            set(raw) != required
            or raw.get("schema") != MODEL_SCHEMA
            or tuple(raw.get("feature_names", ())) != WIDE_FEATURE_NAMES
            or _exact_int(raw.get("feature_width"), "feature_width")
            != WIDE_FEATURE_WIDTH
            or _exact_int(
                raw.get("preserved_block_width"), "preserved_block_width"
            )
            != PRESERVED_BLOCK_WIDTH
            or raw.get("primary_target") != "frozen-g2-candidate_resolved"
            or raw.get("primary_score")
            != "candidate-direct-depth3-seed-equal-newton"
            or raw.get("pair_auxiliary")
            != "invariant-pair-resolved-fraction"
            or raw.get("screening")
            != "train-only-seed-equal-weighted-correlation"
            or raw.get("quantiles")
            != "train-only-seed-equal-weighted-empirical"
            or raw.get("group_keys_are_inputs") is not False
            or raw.get("absolute_identity_inputs") is not False
            or raw.get("cross_fit_unit") != "whole-seed"
        ):
            raise RuntimeError("G3R3 checkpoint feature identity mismatch")
        try:
            config = G3R3Config.from_manifest(raw["config"])
            folds = tuple(
                G3R3Fold.from_manifest(row)
                for row in _exact_list(raw["folds"], "folds")
            )
            assignments = tuple(
                (
                    _exact_int(
                        _exact_list(row, "fold assignment")[0], "fold seed"
                    ),
                    _exact_int(row[1], "fold index"),
                )
                for row in _exact_list(raw["fold_by_seed"], "fold_by_seed")
                if len(row) == 2
            )
            training_seeds = tuple(
                _exact_int(seed, "training seed")
                for seed in _exact_list(
                    raw["training_seeds"], "training_seeds"
                )
            )
            partition_sha = _require_sha256(
                raw["partition_sha256"], "partition_sha256"
            )
            dataset_sha = _require_sha256(
                raw["training_dataset_sha256"], "training_dataset_sha256"
            )
            result = cls(
                config,
                folds,
                assignments,
                training_seeds,
                partition_sha,
                dataset_sha,
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("G3R3 checkpoint manifest is malformed") from exc
        expected_assignments = tuple(
            sorted(
                (seed, index)
                for index, fold in enumerate(folds)
                for seed in fold.heldout_seeds
            )
        )
        try:
            exact_partition = _resolved_partition(training_seeds, config)
        except ValueError as exc:
            raise RuntimeError("G3R3 checkpoint partition is malformed") from exc
        if (
            len(assignments) != len(_exact_list(raw["fold_by_seed"], "fold_by_seed"))
            or len(folds) != config.folds
            or assignments != expected_assignments
            or training_seeds != tuple(sorted(set(training_seeds)))
            or tuple(seed for seed, _ in assignments) != training_seeds
            or tuple(fold.heldout_seeds for fold in folds) != exact_partition
            or partition_sha
            != _partition_sha256(config.partition_id, exact_partition)
            or any(
                fold.candidate.config != config.candidate
                or fold.pair.config != config.pair
                for fold in folds
            )
            or _canonical_bytes(result.manifest()) != _canonical_bytes(raw)
        ):
            raise RuntimeError("G3R3 checkpoint manifest is malformed")
        return result

    def predict(
        self, board: WideBoard, *, oof: bool = False
    ) -> tuple[G3R3Prediction, ...]:
        pair_groups = PairGroups.from_board(board)
        known = self.seed_folds
        candidate_members: list[list[float]] = [[] for _ in board.labels]
        pair_members: list[list[float]] = [[] for _ in board.labels]
        for fold_index, fold in enumerate(self.folds):
            candidate = fold.candidate.probabilities(board.features)
            pair = pair_groups.expand(
                fold.pair.probabilities(pair_groups.features), len(board.labels)
            )
            for index, seed_value in enumerate(board.seeds):
                seed = int(seed_value)
                if oof:
                    if seed not in known:
                        raise ValueError("OOF prediction requested for unknown seed")
                    if known[seed] != fold_index:
                        continue
                candidate_members[index].append(float(candidate[index]))
                pair_members[index].append(float(pair[index]))
        predictions: list[G3R3Prediction] = []
        weight = self.config.blend_pair_weight
        for index in range(len(board.labels)):
            if not candidate_members[index] or not pair_members[index]:
                raise RuntimeError("G3R3 prediction has no eligible fold")
            candidate_mean = float(np.mean(candidate_members[index]))
            pair_mean = float(np.mean(pair_members[index]))
            predictions.append(
                G3R3Prediction(
                    int(board.seeds[index]),
                    str(board.query_ids[index]),
                    int(board.ordinals[index]),
                    candidate_mean,
                    float(np.std(candidate_members[index])),
                    pair_mean,
                    float(np.std(pair_members[index])),
                    (1.0 - weight) * candidate_mean + weight * pair_mean,
                )
            )
        return tuple(predictions)


def train_resolution_first_g3r3(
    board: WideBoard, *, config: G3R3Config | None = None
) -> ResolutionFirstG3R3:
    if type(board) is not WideBoard:
        raise ValueError("G3R3 training requires an exact WideBoard")
    resolved = G3R3Config() if config is None else config
    if type(resolved) is not G3R3Config:
        raise ValueError("G3R3 training configuration has the wrong type")
    partition = _resolved_partition(board.unique_seeds, resolved)
    pair_groups = PairGroups.from_board(board)
    all_seeds = set(board.unique_seeds)
    folds: list[G3R3Fold] = []
    for heldout in partition:
        training = all_seeds - set(heldout)
        candidate_mask = np.asarray(
            [int(seed) in training for seed in board.seeds], dtype=bool
        )
        pair_mask = np.asarray(
            [int(seed) in training for seed in pair_groups.seeds], dtype=bool
        )
        folds.append(
            G3R3Fold(
                heldout,
                HistogramNewtonBoostR3.fit(
                    board.features[candidate_mask],
                    board.labels[candidate_mask].astype(np.float64),
                    board.seeds[candidate_mask],
                    resolved.candidate,
                ),
                HistogramNewtonBoostR3.fit(
                    pair_groups.features[pair_mask],
                    pair_groups.targets[pair_mask],
                    pair_groups.seeds[pair_mask],
                    resolved.pair,
                ),
            )
        )
    assignments = tuple(
        sorted(
            (seed, index)
            for index, heldout in enumerate(partition)
            for seed in heldout
        )
    )
    return ResolutionFirstG3R3(
        resolved,
        tuple(folds),
        assignments,
        board.unique_seeds,
        _partition_sha256(resolved.partition_id, partition),
        board.sha256,
    )


def _dependencies(value: Any) -> dict[str, str]:
    raw = _exact_dict(value, "dependencies")
    if (
        not raw
        or any(
            _DEPENDENCY_RE.fullmatch(name) is None
            or _SHA256_RE.fullmatch(sha) is None
            for name, sha in raw.items()
            if type(name) is str and type(sha) is str
        )
        or any(type(name) is not str or type(sha) is not str for name, sha in raw.items())
    ):
        raise RuntimeError("G3R3 dependency identity is malformed")
    return {name: raw[name] for name in sorted(raw)}


def _checkpoint(
    model: ResolutionFirstG3R3,
    metadata: Mapping[str, Any],
    dependencies: Mapping[str, str],
) -> dict[str, Any]:
    metadata_value = _canonical_dict(metadata, "metadata")
    dependency_value = _dependencies(dependencies)
    body = {
        "schema": CHECKPOINT_SCHEMA,
        "model_sha256": model.sha256,
        "partition_sha256": model.partition_sha256,
        "dataset_sha256": model.training_dataset_sha256,
        "metadata": metadata_value,
        "metadata_sha256": _sha256(metadata_value),
        "dependencies": dependency_value,
        "dependencies_sha256": _sha256(dependency_value),
        "model": model.manifest(),
    }
    return {**body, "checkpoint_sha256": _sha256(body)}


def _validate_checkpoint(
    value: Any,
    *,
    expected_metadata: Mapping[str, Any],
    expected_model_sha256: str,
    expected_partition_sha256: str,
    expected_dataset_sha256: str,
    expected_dependencies: Mapping[str, str],
) -> tuple[ResolutionFirstG3R3, dict[str, Any]]:
    raw = _exact_dict(value, "checkpoint")
    required = {
        "schema",
        "model_sha256",
        "partition_sha256",
        "dataset_sha256",
        "metadata",
        "metadata_sha256",
        "dependencies",
        "dependencies_sha256",
        "model",
        "checkpoint_sha256",
    }
    if set(raw) != required or raw.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("G3R3 checkpoint envelope is malformed")
    expected_model = _require_sha256(expected_model_sha256, "expected model")
    expected_partition = _require_sha256(
        expected_partition_sha256, "expected partition"
    )
    expected_dataset = _require_sha256(
        expected_dataset_sha256, "expected dataset"
    )
    metadata_expected = _canonical_dict(expected_metadata, "expected metadata")
    dependencies_expected = _dependencies(expected_dependencies)
    body = {key: raw[key] for key in raw if key != "checkpoint_sha256"}
    if (
        _require_sha256(raw["checkpoint_sha256"], "checkpoint_sha256")
        != _sha256(body)
        or _require_sha256(raw["model_sha256"], "model_sha256")
        != expected_model
        or _require_sha256(raw["partition_sha256"], "partition_sha256")
        != expected_partition
        or _require_sha256(raw["dataset_sha256"], "dataset_sha256")
        != expected_dataset
        or _require_sha256(raw["metadata_sha256"], "metadata_sha256")
        != _sha256(raw["metadata"])
        or _canonical_bytes(raw["metadata"]) != _canonical_bytes(metadata_expected)
        or _require_sha256(raw["dependencies_sha256"], "dependencies_sha256")
        != _sha256(raw["dependencies"])
        or _dependencies(raw["dependencies"]) != dependencies_expected
    ):
        raise RuntimeError("G3R3 checkpoint expectation or identity mismatch")
    model = ResolutionFirstG3R3.from_manifest(raw["model"])
    if (
        model.sha256 != expected_model
        or model.partition_sha256 != expected_partition
        or model.training_dataset_sha256 != expected_dataset
    ):
        raise RuntimeError("G3R3 checkpoint model binding mismatch")
    return model, metadata_expected


def _decode_checkpoint(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            if key in output:
                raise RuntimeError("G3R3 checkpoint has duplicate keys")
            output[key] = value
        return output

    def reject_constant(value: str) -> None:
        raise RuntimeError(f"G3R3 checkpoint has non-finite {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("G3R3 checkpoint JSON is malformed") from exc
    if type(value) is not dict or raw != _canonical_bytes(value):
        raise RuntimeError("G3R3 checkpoint encoding is noncanonical")
    return value


def _absolute_direct(
    path: str | os.PathLike[str], root: str | os.PathLike[str]
) -> tuple[Path, Path, str]:
    target = Path(os.path.abspath(os.fspath(path)))
    directory = Path(os.path.abspath(os.fspath(root)))
    if (
        target.parent != directory
        or _SAFE_NAME_RE.fullmatch(target.name) is None
        or os.path.realpath(directory) != str(directory)
    ):
        raise RuntimeError("G3R3 checkpoint path is indirect")
    return target, directory, target.name


def _lstat_path(path: Path) -> os.stat_result:
    return path.lstat()


def _open_directory(path: Path) -> tuple[int, os.stat_result]:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise RuntimeError("G3R3 checkpoint root cannot be opened") from exc
    opened = os.fstat(descriptor)
    live = _lstat_path(path)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or stat.S_ISLNK(live.st_mode)
        or not stat.S_ISDIR(live.st_mode)
        or (opened.st_dev, opened.st_ino) != (live.st_dev, live.st_ino)
    ):
        os.close(descriptor)
        raise RuntimeError("G3R3 checkpoint root identity changed")
    return descriptor, opened


def _same_stat(left: os.stat_result, right: os.stat_result, fields: tuple[str, ...]) -> bool:
    return all(getattr(left, field) == getattr(right, field) for field in fields)


def _read_descriptor(descriptor: int) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError("G3R3 checkpoint must be a single-link regular file")
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1 << 20)
        if not chunk:
            break
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if not _same_stat(before, after, _FILE_FIELDS):
        raise RuntimeError("G3R3 checkpoint changed while being read")
    raw = b"".join(chunks)
    if len(raw) != after.st_size:
        raise RuntimeError("G3R3 checkpoint read was incomplete")
    return raw, after


def load_checkpoint_g3r3(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str],
    expected_metadata: Mapping[str, Any],
    expected_model_sha256: str,
    expected_partition_sha256: str,
    expected_dataset_sha256: str,
    expected_dependencies: Mapping[str, str],
) -> tuple[ResolutionFirstG3R3, dict[str, Any]]:
    target, directory, name = _absolute_direct(path, root)
    directory_fd, directory_before = _open_directory(directory)
    descriptor = -1
    try:
        flags = (
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(name, flags, dir_fd=directory_fd)
        raw, opened = _read_descriptor(descriptor)
        model, metadata = _validate_checkpoint(
            _decode_checkpoint(raw),
            expected_metadata=expected_metadata,
            expected_model_sha256=expected_model_sha256,
            expected_partition_sha256=expected_partition_sha256,
            expected_dataset_sha256=expected_dataset_sha256,
            expected_dependencies=expected_dependencies,
        )

        # All lexical checks precede the final descriptor-relative boundary.
        live_root = _lstat_path(directory)
        if (
            stat.S_ISLNK(live_root.st_mode)
            or (live_root.st_dev, live_root.st_ino)
            != (directory_before.st_dev, directory_before.st_ino)
        ):
            raise RuntimeError("G3R3 checkpoint root path changed")
        live_target = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        final_descriptor = os.fstat(descriptor)
        directory_after = os.fstat(directory_fd)
        if (
            not _same_stat(opened, live_target, _FILE_FIELDS)
            or not _same_stat(opened, final_descriptor, _FILE_FIELDS)
            or not _same_stat(
                directory_before, directory_after, _DIRECTORY_FIELDS
            )
        ):
            raise RuntimeError("G3R3 checkpoint path changed or ABA occurred")
    except OSError as exc:
        raise RuntimeError("G3R3 checkpoint cannot be read safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)
    return model, metadata


def save_checkpoint_g3r3(
    path: str | os.PathLike[str],
    model: ResolutionFirstG3R3,
    *,
    root: str | os.PathLike[str],
    metadata: Mapping[str, Any],
    expected_model_sha256: str,
    expected_partition_sha256: str,
    expected_dataset_sha256: str,
    dependencies: Mapping[str, str],
) -> str:
    if type(model) is not ResolutionFirstG3R3:
        raise RuntimeError("G3R3 checkpoint model has the wrong type")
    try:
        validated_model = ResolutionFirstG3R3.from_manifest(model.manifest())
    except RuntimeError as exc:
        raise RuntimeError("G3R3 checkpoint model is not canonical") from exc
    if validated_model.manifest() != model.manifest():
        raise RuntimeError("G3R3 checkpoint model is not canonical")
    expected_model = _require_sha256(expected_model_sha256, "expected model")
    expected_partition = _require_sha256(
        expected_partition_sha256, "expected partition"
    )
    expected_dataset = _require_sha256(
        expected_dataset_sha256, "expected dataset"
    )
    if (
        model.sha256 != expected_model
        or model.partition_sha256 != expected_partition
        or model.training_dataset_sha256 != expected_dataset
    ):
        raise RuntimeError("G3R3 save expectations do not bind the model")
    checkpoint = _checkpoint(model, metadata, dependencies)
    encoded = _canonical_bytes(checkpoint)
    target, directory, name = _absolute_direct(path, root)
    directory_fd, _directory_opened = _open_directory(directory)
    temporary = f".{name}.tmp-{os.getpid()}-{secrets.token_hex(12)}"
    temporary_fd = -1
    target_fd = -1
    temporary_live = False
    published = False
    try:
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("G3R3 checkpoint destination already exists")
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        temporary_fd = os.open(
            temporary, flags, 0o600, dir_fd=directory_fd
        )
        temporary_live = True
        view = memoryview(encoded)
        while view:
            written = os.write(temporary_fd, view)
            if written <= 0:
                raise RuntimeError("G3R3 checkpoint write was incomplete")
            view = view[written:]
        os.fchmod(temporary_fd, 0o444)
        os.fsync(temporary_fd)
        os.link(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary, dir_fd=directory_fd)
        temporary_live = False
        os.fsync(directory_fd)
        directory_baseline = os.fstat(directory_fd)

        target_fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        raw, opened = _read_descriptor(target_fd)
        if raw != encoded:
            raise RuntimeError("G3R3 published checkpoint bytes changed")

        live_root = _lstat_path(directory)
        if (
            stat.S_ISLNK(live_root.st_mode)
            or (live_root.st_dev, live_root.st_ino)
            != (directory_baseline.st_dev, directory_baseline.st_ino)
        ):
            raise RuntimeError("G3R3 checkpoint root path changed")
        live_target = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        final_target = os.fstat(target_fd)
        final_temporary = os.fstat(temporary_fd)
        directory_after = os.fstat(directory_fd)
        if (
            not _same_stat(opened, live_target, _FILE_FIELDS)
            or not _same_stat(opened, final_target, _FILE_FIELDS)
            or not _same_stat(opened, final_temporary, _FILE_FIELDS)
            or not _same_stat(
                directory_baseline, directory_after, _DIRECTORY_FIELDS
            )
        ):
            raise RuntimeError("G3R3 checkpoint publication changed or ABA occurred")
    except OSError as exc:
        raise RuntimeError("G3R3 checkpoint cannot be published safely") from exc
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_live:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
    if not published:
        raise RuntimeError("G3R3 checkpoint publication failed")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BoostConfigR3",
    "CHECKPOINT_SCHEMA",
    "FeatureBinnerR3",
    "G3R3Config",
    "G3R3Fold",
    "G3R3Prediction",
    "HistogramNewtonBoostR3",
    "MODEL_SCHEMA",
    "PRESERVED_BLOCK_WIDTH",
    "ResolutionFirstG3R3",
    "TreeNodeR3",
    "WIDE_FEATURE_NAMES",
    "WIDE_FEATURE_WIDTH",
    "WideBoard",
    "load_checkpoint_g3r3",
    "save_checkpoint_g3r3",
    "seed_equal_weights",
    "train_resolution_first_g3r3",
    "whole_seed_folds_g3r3",
    "wide_board_from_entries",
    "wide_board_from_records",
]
