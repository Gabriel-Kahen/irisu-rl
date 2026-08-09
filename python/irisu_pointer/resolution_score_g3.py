"""Survival-first score ranking behind an exact two-renewal shield.

The ranker consumes the frozen G3R2 712-column identity-free representation,
but it is trained only on branches certified by the exact training oracle as
resolved and not unsafe.  Certification is a row-selection label, never a
model input.  Deployment is deliberately lexicographic: an external exact
shield supplies the eligible candidates, then this model ranks only those
candidates and may fall back to the incumbent.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from . import resolution_first_g2 as _g2
from . import resolution_first_g3r2 as _g3r2
from .resolution_first_g2 import BoardBranchRecord
from .resolution_first_g3r2 import (
    WIDE_FEATURE_NAMES,
    WIDE_FEATURE_WIDTH,
    BoostConfig,
    HistogramNewtonBoost,
    WideBoard,
    wide_board_from_records,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
SCORE_TARGET_SCALE = 256.0


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _dependency_identities() -> tuple[tuple[str, str], ...]:
    paths = (
        ("resolution_first_g2", Path(_g2.__file__).resolve()),
        ("resolution_first_g3r2", Path(_g3r2.__file__).resolve()),
        ("resolution_score_g3", Path(__file__).resolve()),
    )
    return tuple((name, _sha256_file(path)) for name, path in paths)


def soft_score_target(score_advantage: np.ndarray | Sequence[float]) -> np.ndarray:
    values = np.asarray(score_advantage, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("score advantages must be finite")
    return 0.5 + 0.5 * np.tanh(values / SCORE_TARGET_SCALE)


def implied_score_advantage(soft_score: float) -> float:
    value = float(soft_score)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("soft score must be a finite probability")
    clipped = float(np.clip(value, 1e-12, 1.0 - 1e-12))
    return SCORE_TARGET_SCALE * math.atanh(2.0 * clipped - 1.0)


@dataclass(frozen=True, slots=True)
class ResolutionScoreBoard:
    """Aligned G3R2 features, score target, and non-input shield labels."""

    wide: WideBoard
    score_advantages: np.ndarray
    shield_certified: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.wide.labels)
        scores = np.asarray(self.score_advantages)
        shield = np.asarray(self.shield_certified)
        if (
            scores.shape != (count,)
            or shield.shape != (count,)
            or not np.isfinite(scores).all()
            or not np.isin(shield, (False, True, 0, 1)).all()
        ):
            raise ValueError("resolution-score board is malformed")
        copied_scores = np.array(scores, dtype=np.float64, copy=True)
        copied_shield = np.array(shield, dtype=np.bool_, copy=True)
        copied_scores.setflags(write=False)
        copied_shield.setflags(write=False)
        object.__setattr__(self, "score_advantages", copied_scores)
        object.__setattr__(self, "shield_certified", copied_shield)

    @property
    def features(self) -> np.ndarray:
        return self.wide.features

    @property
    def seeds(self) -> np.ndarray:
        return self.wide.seeds

    @property
    def query_ids(self) -> np.ndarray:
        return self.wide.query_ids

    @property
    def ordinals(self) -> np.ndarray:
        return self.wide.ordinals

    @property
    def unique_seeds(self) -> tuple[int, ...]:
        return self.wide.unique_seeds

    def take(self, rows: np.ndarray | Sequence[int] | Sequence[bool]) -> ResolutionScoreBoard:
        selected = np.asarray(rows)
        return ResolutionScoreBoard(
            self.wide.take(selected),
            self.score_advantages[selected],
            self.shield_certified[selected],
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3i-resolution-score-board-g3-v1",
            "wide_board_sha256": self.wide.sha256,
            "feature_names": list(WIDE_FEATURE_NAMES),
            "shield_certified_definition":
                "candidate_resolved-and-not-exact_unsafe",
            "shield_certified_is_model_input": False,
            "score_target_is_model_input": False,
            "rows": [
                {
                    "seed": int(self.seeds[index]),
                    "query_id": str(self.query_ids[index]),
                    "ordinal": int(self.ordinals[index]),
                    "score_advantage": float(self.score_advantages[index]),
                    "shield_certified": bool(self.shield_certified[index]),
                }
                for index in range(len(self.features))
            ],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())


def resolution_score_board_from_records(
    records: Iterable[BoardBranchRecord],
) -> ResolutionScoreBoard:
    supplied = tuple(records)
    alternatives = tuple(row for row in supplied if row.ordinal != 0)
    if not alternatives:
        raise ValueError("resolution-score board has no alternatives")
    wide = wide_board_from_records(supplied, include_incumbents=False)
    expected = tuple(
        (row.seed, row.query_id, row.ordinal) for row in alternatives
    )
    actual = tuple(
        (int(seed), str(query), int(ordinal))
        for seed, query, ordinal in zip(
            wide.seeds, wide.query_ids, wide.ordinals, strict=True
        )
    )
    if actual != expected:
        raise RuntimeError("G3R2 feature and score rows are misaligned")
    return ResolutionScoreBoard(
        wide,
        np.asarray([row.score_advantage for row in alternatives]),
        np.asarray(
            [
                row.candidate_resolved and not row.exact_unsafe
                for row in alternatives
            ]
        ),
    )


@dataclass(frozen=True, slots=True)
class ResolutionScoreConfig:
    folds: int = 5
    partition_id: str = "development-internal-sorted-v1"
    seed_partition: tuple[tuple[int, ...], ...] = ()
    boost: BoostConfig = BoostConfig(
        rounds=300,
        depth=3,
        learning_rate=0.05,
        l2=8.0,
        minimum_leaf=12,
        maximum_features=180,
        preserved_features=126,
        bins=16,
        balance_classes=False,
    )

    def __post_init__(self) -> None:
        flattened = tuple(seed for fold in self.seed_partition for seed in fold)
        if (
            self.folds not in {5, 8}
            or not self.partition_id
            or self.boost.balance_classes
            or (self.folds == 8 and not self.seed_partition)
            or (
                self.seed_partition
                and (
                    len(self.seed_partition) != self.folds
                    or any(
                        not fold or tuple(sorted(set(fold))) != fold
                        for fold in self.seed_partition
                    )
                    or len(set(flattened)) != len(flattened)
                    or self.partition_id == "development-internal-sorted-v1"
                )
            )
        ):
            raise ValueError("resolution-score configuration is invalid")


def _whole_seed_folds(
    seeds: Sequence[int], count: int
) -> tuple[tuple[int, ...], ...]:
    unique = tuple(sorted(set(int(seed) for seed in seeds)))
    if count not in {5, 8} or len(unique) <= count:
        raise ValueError("resolution-score whole-seed cross-fit is underspecified")
    return tuple(tuple(unique[index::count]) for index in range(count))


def _resolved_partition(
    seeds: Sequence[int], config: ResolutionScoreConfig
) -> tuple[tuple[int, ...], ...]:
    inventory = tuple(sorted(set(int(seed) for seed in seeds)))
    if config.seed_partition:
        if tuple(
            sorted(seed for fold in config.seed_partition for seed in fold)
        ) != inventory:
            raise ValueError("explicit score partition does not cover training seeds")
        return config.seed_partition
    if config.folds != 5:
        raise ValueError("production score cross-fit requires an explicit partition")
    return _whole_seed_folds(inventory, config.folds)


def _partition_sha256(
    partition_id: str, partition: Sequence[Sequence[int]]
) -> str:
    return _sha256(
        {
            "schema": "irisu-r3i-resolution-score-partition-g3-v1",
            "partition_id": partition_id,
            "folds": [list(fold) for fold in partition],
        }
    )


@dataclass(frozen=True, slots=True)
class ResolutionScoreFold:
    heldout_seeds: tuple[int, ...]
    head: HistogramNewtonBoost
    training_shielded_rows: int
    training_shielded_seeds: tuple[int, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "heldout_seeds": list(self.heldout_seeds),
            "head": self.head.manifest(),
            "training_shielded_rows": self.training_shielded_rows,
            "training_shielded_seeds": list(self.training_shielded_seeds),
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> ResolutionScoreFold:
        try:
            result = cls(
                tuple(
                    _integer(seed, "heldout seed")
                    for seed in _sequence(value["heldout_seeds"], "heldout seeds")
                ),
                HistogramNewtonBoost.from_manifest(
                    _mapping(value["head"], "score head")
                ),
                _integer(
                    value["training_shielded_rows"], "training shielded rows"
                ),
                tuple(
                    _integer(seed, "training shielded seed")
                    for seed in _sequence(
                        value["training_shielded_seeds"],
                        "training shielded seeds",
                    )
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("resolution-score fold is malformed") from exc
        if (
            not result.heldout_seeds
            or result.heldout_seeds != tuple(sorted(set(result.heldout_seeds)))
            or result.training_shielded_rows < 1
            or result.training_shielded_seeds
            != tuple(sorted(set(result.training_shielded_seeds)))
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("resolution-score fold is malformed")
        return result


@dataclass(frozen=True, slots=True)
class ResolutionScorePrediction:
    seed: int
    query_id: str
    ordinal: int
    soft_score_mean: float
    soft_score_std: float
    implied_score_advantage: float


@dataclass(frozen=True, slots=True)
class ResolutionScoreG3:
    config: ResolutionScoreConfig
    folds: tuple[ResolutionScoreFold, ...]
    fold_by_seed: tuple[tuple[int, int], ...]
    training_seeds: tuple[int, ...]
    partition_sha256: str
    training_dataset_sha256: str
    dependency_identities: tuple[tuple[str, str], ...]

    @property
    def seed_folds(self) -> dict[int, int]:
        return dict(self.fold_by_seed)

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3i-resolution-score-g3-v1",
            "feature_names": list(WIDE_FEATURE_NAMES),
            "feature_width": WIDE_FEATURE_WIDTH,
            "training_dataset_sha256": self.training_dataset_sha256,
            "training_seeds": list(self.training_seeds),
            "partition_sha256": self.partition_sha256,
            "dependency_identities": [list(row) for row in self.dependency_identities],
            "training_subset": "shield_certified-only",
            "shield_certified_is_model_input": False,
            "soft_target":
                "0.5+0.5*tanh(score_advantage/256.0)",
            "primary_prediction": "soft-score-ensemble-mean",
            "cross_fit_unit": "whole-seed",
            "fold_by_seed": [list(row) for row in self.fold_by_seed],
            "folds": [fold.manifest() for fold in self.folds],
            "config": {
                "folds": self.config.folds,
                "partition_id": self.config.partition_id,
                "seed_partition": [
                    list(fold) for fold in self.config.seed_partition
                ],
                "boost": asdict(self.config.boost),
            },
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> ResolutionScoreG3:
        if (
            value.get("schema") != "irisu-r3i-resolution-score-g3-v1"
            or tuple(value.get("feature_names", ())) != WIDE_FEATURE_NAMES
            or value.get("feature_width") != WIDE_FEATURE_WIDTH
            or value.get("training_subset") != "shield_certified-only"
            or value.get("shield_certified_is_model_input") is not False
            or value.get("soft_target")
            != "0.5+0.5*tanh(score_advantage/256.0)"
            or value.get("primary_prediction") != "soft-score-ensemble-mean"
            or value.get("cross_fit_unit") != "whole-seed"
        ):
            raise RuntimeError("resolution-score feature identity mismatch")
        try:
            raw_config = dict(_mapping(value["config"], "score config"))
            if set(raw_config) != {
                "folds",
                "partition_id",
                "seed_partition",
                "boost",
            }:
                raise RuntimeError("resolution-score configuration is malformed")
            config = ResolutionScoreConfig(
                folds=_integer(raw_config["folds"], "fold count"),
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
                boost=BoostConfig(
                    **dict(_mapping(raw_config["boost"], "boost config"))
                ),
            )
            folds = tuple(
                ResolutionScoreFold.from_manifest(_mapping(row, "score fold"))
                for row in _sequence(value["folds"], "score folds")
            )
            assignments = tuple(
                (
                    _integer(row[0], "fold seed"),
                    _integer(row[1], "fold index"),
                )
                for row in _sequence(value["fold_by_seed"], "fold mapping")
            )
            seeds = tuple(
                _integer(seed, "training seed")
                for seed in _sequence(value["training_seeds"], "training seeds")
            )
            dependencies = tuple(
                (str(row[0]), str(row[1]))
                for row in _sequence(
                    value["dependency_identities"], "dependency identities"
                )
                if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
            )
            result = cls(
                config,
                folds,
                assignments,
                seeds,
                str(value["partition_sha256"]),
                str(value["training_dataset_sha256"]),
                dependencies,
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise RuntimeError("resolution-score manifest is malformed") from exc
        try:
            expected_partition = _resolved_partition(seeds, config)
        except ValueError as exc:
            raise RuntimeError("resolution-score partition is malformed") from exc
        expected_assignments = tuple(
            sorted(
                (seed, index)
                for index, fold in enumerate(folds)
                for seed in fold.heldout_seeds
            )
        )
        if (
            len(folds) != config.folds
            or tuple(fold.heldout_seeds for fold in folds) != expected_partition
            or assignments != expected_assignments
            or tuple(seed for seed, _index in assignments) != seeds
            or seeds != tuple(sorted(set(seeds)))
            or result.partition_sha256
            != _partition_sha256(config.partition_id, expected_partition)
            or not _SHA256_RE.fullmatch(result.partition_sha256)
            or not _SHA256_RE.fullmatch(result.training_dataset_sha256)
            or dependencies != _dependency_identities()
            or any(
                fold.head.config != config.boost
                or set(fold.training_shielded_seeds)
                & set(fold.heldout_seeds)
                or not set(fold.training_shielded_seeds)
                <= (set(seeds) - set(fold.heldout_seeds))
                for fold in folds
            )
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("resolution-score manifest is malformed")
        return result

    def checkpoint(
        self, metadata: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        body = {
            "schema": "irisu-r3i-resolution-score-checkpoint-g3-v1",
            "model_sha256": self.sha256,
            "model": self.manifest(),
            "metadata": dict(metadata or {}),
        }
        return {**body, "checkpoint_sha256": _sha256(body)}

    @classmethod
    def from_checkpoint(
        cls, value: Mapping[str, Any]
    ) -> tuple[ResolutionScoreG3, dict[str, object]]:
        if set(value) != {
            "schema",
            "model_sha256",
            "model",
            "metadata",
            "checkpoint_sha256",
        }:
            raise RuntimeError("resolution-score checkpoint envelope is malformed")
        body = {key: value[key] for key in value if key != "checkpoint_sha256"}
        if (
            value.get("schema")
            != "irisu-r3i-resolution-score-checkpoint-g3-v1"
            or value.get("checkpoint_sha256") != _sha256(body)
        ):
            raise RuntimeError("resolution-score checkpoint identity mismatch")
        model = cls.from_manifest(_mapping(value["model"], "score model"))
        if value.get("model_sha256") != model.sha256:
            raise RuntimeError("resolution-score checkpoint model identity mismatch")
        metadata = value["metadata"]
        if not isinstance(metadata, Mapping):
            raise RuntimeError("resolution-score checkpoint metadata is malformed")
        return model, dict(metadata)

    def predict(
        self, board: ResolutionScoreBoard, *, oof: bool = False
    ) -> tuple[ResolutionScorePrediction, ...]:
        known = self.seed_folds
        members: list[list[float]] = [[] for _ in board.features]
        for fold_index, fold in enumerate(self.folds):
            values = fold.head.probabilities(board.features)
            for index, seed in enumerate(board.seeds):
                if oof:
                    if int(seed) not in known:
                        raise ValueError(
                            "OOF score prediction requested for an unknown seed"
                        )
                    if known[int(seed)] != fold_index:
                        continue
                members[index].append(float(values[index]))
        output: list[ResolutionScorePrediction] = []
        for index, values in enumerate(members):
            if not values:
                raise RuntimeError("resolution-score prediction has no eligible fold")
            mean = float(np.mean(values))
            output.append(
                ResolutionScorePrediction(
                    int(board.seeds[index]),
                    str(board.query_ids[index]),
                    int(board.ordinals[index]),
                    mean,
                    float(np.std(values)),
                    implied_score_advantage(mean),
                )
            )
        return tuple(output)


def train_resolution_score_g3(
    board: ResolutionScoreBoard,
    *,
    config: ResolutionScoreConfig | None = None,
) -> ResolutionScoreG3:
    resolved = ResolutionScoreConfig() if config is None else config
    partition = _resolved_partition(board.unique_seeds, resolved)
    all_seeds = set(board.unique_seeds)
    targets = soft_score_target(board.score_advantages)
    folds: list[ResolutionScoreFold] = []
    for heldout in partition:
        training_seeds = all_seeds - set(heldout)
        selected = board.shield_certified & np.asarray(
            [int(seed) in training_seeds for seed in board.seeds]
        )
        shielded_seeds = tuple(
            sorted(set(int(seed) for seed in board.seeds[selected]))
        )
        if not selected.any() or len(shielded_seeds) < 2:
            raise ValueError("score fold lacks shield-certified training support")
        folds.append(
            ResolutionScoreFold(
                heldout,
                HistogramNewtonBoost.fit(
                    board.features[selected],
                    targets[selected],
                    board.seeds[selected],
                    resolved.boost,
                ),
                int(selected.sum()),
                shielded_seeds,
            )
        )
    assignments = tuple(
        sorted(
            (seed, index)
            for index, heldout in enumerate(partition)
            for seed in heldout
        )
    )
    return ResolutionScoreG3(
        resolved,
        tuple(folds),
        assignments,
        board.unique_seeds,
        _partition_sha256(resolved.partition_id, partition),
        board.sha256,
        _dependency_identities(),
    )


def select_shielded_score_candidate(
    predictions: Sequence[ResolutionScorePrediction],
    shield_certified_ordinals: Iterable[int],
    *,
    incumbent_ordinal: int = 0,
) -> int:
    """Choose the best positive predicted advantage behind an external shield."""

    if isinstance(incumbent_ordinal, bool) or not isinstance(incumbent_ordinal, int):
        raise ValueError("incumbent ordinal is malformed")
    certified: set[int] = set()
    for ordinal in shield_certified_ordinals:
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
            raise ValueError("shield-certified ordinal is malformed")
        certified.add(ordinal)
    by_ordinal: dict[int, ResolutionScorePrediction] = {}
    for prediction in predictions:
        if prediction.ordinal in by_ordinal:
            raise ValueError("score predictions contain duplicate ordinals")
        by_ordinal[prediction.ordinal] = prediction
    eligible = [
        prediction
        for ordinal, prediction in by_ordinal.items()
        if ordinal in certified and prediction.soft_score_mean > 0.5
    ]
    if not eligible:
        return incumbent_ordinal
    return min(
        eligible,
        key=lambda prediction: (-prediction.soft_score_mean, prediction.ordinal),
    ).ordinal


def save_resolution_score_checkpoint(
    path: str | os.PathLike[str],
    model: ResolutionScoreG3,
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
        raise RuntimeError("resolution-score destination is occupied or indirect")
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
        raise RuntimeError("resolution-score checkpoint creation failed")
    return hashlib.sha256(encoded).hexdigest()


def load_resolution_score_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected_metadata: Mapping[str, object] | None = None,
    expected_partition_sha256: str | None = None,
    expected_model_sha256: str | None = None,
) -> tuple[ResolutionScoreG3, dict[str, object]]:
    source = os.path.abspath(os.fspath(path))
    before = os.lstat(source)
    if (
        not os.path.isfile(source)
        or os.path.islink(source)
        or os.path.realpath(source) != source
        or before.st_nlink != 1
    ):
        raise RuntimeError("resolution-score checkpoint source is indirect")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError("resolution-score descriptor identity changed")
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
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(before, name) != getattr(opened, name)
        or getattr(opened, name) != getattr(after_descriptor, name)
        or getattr(after_descriptor, name) != getattr(after_path, name)
        for name in fields
    ):
        raise RuntimeError("resolution-score checkpoint changed while being read")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("resolution-score checkpoint JSON is malformed") from exc
    if not isinstance(payload, Mapping) or raw != _canonical_bytes(payload):
        raise RuntimeError("resolution-score checkpoint encoding is noncanonical")
    model, metadata = ResolutionScoreG3.from_checkpoint(payload)
    if expected_metadata is not None and metadata != dict(expected_metadata):
        raise RuntimeError("resolution-score metadata expectation mismatch")
    if (
        expected_partition_sha256 is not None
        and model.partition_sha256 != expected_partition_sha256
    ):
        raise RuntimeError("resolution-score partition expectation mismatch")
    if expected_model_sha256 is not None and model.sha256 != expected_model_sha256:
        raise RuntimeError("resolution-score model expectation mismatch")
    return model, metadata
