"""Strict-solvency score ranker with identity-bound exact certificates.

This version supersedes the development-only v1 shield without modifying it.
Training and selection admit only rows satisfying all of:

``candidate_resolved && finite_pair && !exact_unsafe && !severe_unsafe
   && finite(b2) && b2 >= 0 && finite(delta_b2) && delta_b2 > 0``.

The strict certificate and its identity fields are never model inputs.
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
from . import resolution_score_g3 as _score_v1
from .resolution_first_g2 import BoardBranchRecord
from .resolution_first_g3r2 import (
    WIDE_FEATURE_NAMES,
    WIDE_FEATURE_WIDTH,
    HistogramNewtonBoost,
    WideBoard,
    wide_board_from_records,
)
from .resolution_score_g3 import (
    ResolutionScoreConfig,
    ResolutionScoreFold,
    implied_score_advantage,
    soft_score_target,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


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


def _dependencies() -> tuple[tuple[str, str], ...]:
    modules = (
        ("resolution_first_g2", _g2),
        ("resolution_first_g3r2", _g3r2),
        ("resolution_score_g3", _score_v1),
        ("resolution_score_g3v2", None),
    )
    return tuple(
        (
            name,
            _sha256_file(
                Path(__file__).resolve()
                if module is None
                else Path(module.__file__).resolve()
            ),
        )
        for name, module in modules
    )


@dataclass(frozen=True, slots=True, order=True)
class ScoreCandidateIdentity:
    seed: int
    query_id: str
    ordinal: int
    candidate_id: str
    action_id: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not isinstance(self.query_id, str)
            or not self.query_id
            or isinstance(self.ordinal, bool)
            or not isinstance(self.ordinal, int)
            or self.ordinal < 0
            or not isinstance(self.candidate_id, str)
            or not self.candidate_id
            or not isinstance(self.action_id, str)
            or not self.action_id
        ):
            raise ValueError("score candidate identity is malformed")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_manifest(
        cls, value: Mapping[str, Any]
    ) -> ScoreCandidateIdentity:
        try:
            result = cls(
                _integer(value["seed"], "identity seed"),
                str(value["query_id"]),
                _integer(value["ordinal"], "identity ordinal"),
                str(value["candidate_id"]),
                str(value["action_id"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("score candidate identity is malformed") from exc
        if result.manifest() != dict(value):
            raise RuntimeError("score candidate identity is malformed")
        return result


@dataclass(frozen=True, slots=True)
class StrictShieldCertificate:
    identity: ScoreCandidateIdentity
    candidate_resolved: bool
    finite_pair: bool
    exact_unsafe: bool
    severe_unsafe: bool
    b2: float
    delta_b2: float

    def __post_init__(self) -> None:
        if (
            self.identity.ordinal < 1
            or self.candidate_resolved is not True
            or self.finite_pair is not True
            or self.exact_unsafe is not False
            or self.severe_unsafe is not False
            or not math.isfinite(self.b2)
            or not math.isfinite(self.delta_b2)
            or self.b2 < 0
            or self.delta_b2 <= 0
        ):
            raise ValueError("strict shield certificate is invalid")


def strict_shield_certified(record: BoardBranchRecord) -> bool:
    return bool(
        record.ordinal != 0
        and record.candidate_resolved
        and record.finite_pair
        and not record.exact_unsafe
        and not record.severe_unsafe
        and record.b2 is not None
        and math.isfinite(record.b2)
        and record.b2 >= 0
        and record.delta_b2 is not None
        and math.isfinite(record.delta_b2)
        and record.delta_b2 > 0
    )


def strict_certificate(record: BoardBranchRecord) -> StrictShieldCertificate:
    if not strict_shield_certified(record):
        raise ValueError("record does not satisfy the strict shield")
    assert record.b2 is not None and record.delta_b2 is not None
    return StrictShieldCertificate(
        ScoreCandidateIdentity(
            record.seed,
            record.query_id,
            record.ordinal,
            record.candidate_id,
            record.action_id,
        ),
        record.candidate_resolved,
        record.finite_pair,
        record.exact_unsafe,
        record.severe_unsafe,
        float(record.b2),
        float(record.delta_b2),
    )


@dataclass(frozen=True, slots=True)
class ResolutionScoreBoardV2:
    wide: WideBoard
    identities: tuple[ScoreCandidateIdentity, ...]
    score_advantages: np.ndarray
    strict_certified: np.ndarray

    def __post_init__(self) -> None:
        count = len(self.wide.labels)
        scores = np.asarray(self.score_advantages)
        certified = np.asarray(self.strict_certified)
        if (
            len(self.identities) != count
            or len(set(self.identities)) != count
            or scores.shape != (count,)
            or certified.shape != (count,)
            or not np.isfinite(scores).all()
            or not np.isin(certified, (False, True, 0, 1)).all()
        ):
            raise ValueError("strict resolution-score board is malformed")
        expected = tuple(
            (int(seed), str(query), int(ordinal))
            for seed, query, ordinal in zip(
                self.wide.seeds,
                self.wide.query_ids,
                self.wide.ordinals,
                strict=True,
            )
        )
        actual = tuple(
            (row.seed, row.query_id, row.ordinal) for row in self.identities
        )
        if actual != expected or any(row.ordinal < 1 for row in self.identities):
            raise ValueError("strict score identities are misaligned")
        copied_scores = np.array(scores, dtype=np.float64, copy=True)
        copied_certified = np.array(certified, dtype=np.bool_, copy=True)
        copied_scores.setflags(write=False)
        copied_certified.setflags(write=False)
        object.__setattr__(self, "score_advantages", copied_scores)
        object.__setattr__(self, "strict_certified", copied_certified)

    @property
    def features(self) -> np.ndarray:
        return self.wide.features

    @property
    def seeds(self) -> np.ndarray:
        return self.wide.seeds

    @property
    def unique_seeds(self) -> tuple[int, ...]:
        return self.wide.unique_seeds

    def take(
        self, rows: np.ndarray | Sequence[int] | Sequence[bool]
    ) -> ResolutionScoreBoardV2:
        selected = np.asarray(rows)
        if selected.dtype == bool:
            indices = np.flatnonzero(selected)
        else:
            indices = selected.astype(np.int64)
        return ResolutionScoreBoardV2(
            self.wide.take(selected),
            tuple(self.identities[int(index)] for index in indices),
            self.score_advantages[selected],
            self.strict_certified[selected],
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3i-resolution-score-board-g3v2-v1",
            "wide_board_sha256": self.wide.sha256,
            "feature_names": list(WIDE_FEATURE_NAMES),
            "strict_shield_definition":
                "resolved-and-finite-pair-and-not-unsafe-and-not-severe"
                "-and-b2-ge-zero-and-delta-b2-gt-zero",
            "certificate_or_identity_is_model_input": False,
            "score_target_is_model_input": False,
            "rows": [
                {
                    "identity": identity.manifest(),
                    "score_advantage": float(self.score_advantages[index]),
                    "strict_certified": bool(self.strict_certified[index]),
                }
                for index, identity in enumerate(self.identities)
            ],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())


def resolution_score_board_v2_from_records(
    records: Iterable[BoardBranchRecord],
) -> ResolutionScoreBoardV2:
    supplied = tuple(records)
    alternatives = tuple(row for row in supplied if row.ordinal != 0)
    wide = wide_board_from_records(supplied)
    identities = tuple(
        ScoreCandidateIdentity(
            row.seed,
            row.query_id,
            row.ordinal,
            row.candidate_id,
            row.action_id,
        )
        for row in alternatives
    )
    return ResolutionScoreBoardV2(
        wide,
        identities,
        np.asarray([row.score_advantage for row in alternatives]),
        np.asarray([strict_shield_certified(row) for row in alternatives]),
    )


def _partition(
    seeds: Sequence[int], config: ResolutionScoreConfig
) -> tuple[tuple[int, ...], ...]:
    inventory = tuple(sorted(set(int(seed) for seed in seeds)))
    if config.seed_partition:
        supplied = config.seed_partition
        if tuple(sorted(seed for fold in supplied for seed in fold)) != inventory:
            raise ValueError("strict score partition does not cover training seeds")
        return supplied
    if config.folds != 5:
        raise ValueError("production strict score partition must be explicit")
    return tuple(tuple(inventory[index::5]) for index in range(5))


def _partition_sha(
    identifier: str, folds: Sequence[Sequence[int]]
) -> str:
    return _sha256(
        {
            "schema": "irisu-r3i-resolution-score-partition-g3v2-v1",
            "partition_id": identifier,
            "folds": [list(fold) for fold in folds],
        }
    )


@dataclass(frozen=True, slots=True)
class ResolutionScorePredictionV2:
    identity: ScoreCandidateIdentity
    soft_score_mean: float
    soft_score_std: float
    implied_score_advantage: float

    def validate(self) -> None:
        if (
            self.identity.ordinal < 1
            or not math.isfinite(self.soft_score_mean)
            or not 0 <= self.soft_score_mean <= 1
            or not math.isfinite(self.soft_score_std)
            or self.soft_score_std < 0
            or not math.isfinite(self.implied_score_advantage)
            or not math.isclose(
                self.implied_score_advantage,
                implied_score_advantage(self.soft_score_mean),
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("strict score prediction is inconsistent")


@dataclass(frozen=True, slots=True)
class ResolutionScoreG3V2:
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
            "schema": "irisu-r3i-resolution-score-g3v2-v1",
            "feature_names": list(WIDE_FEATURE_NAMES),
            "feature_width": WIDE_FEATURE_WIDTH,
            "training_dataset_sha256": self.training_dataset_sha256,
            "training_seeds": list(self.training_seeds),
            "partition_sha256": self.partition_sha256,
            "dependency_identities": [list(row) for row in self.dependency_identities],
            "training_subset": "strict-shield-certified-only",
            "certificate_or_identity_is_model_input": False,
            "soft_target": "0.5+0.5*tanh(score_advantage/256.0)",
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
    def from_manifest(
        cls, value: Mapping[str, Any]
    ) -> ResolutionScoreG3V2:
        if (
            value.get("schema") != "irisu-r3i-resolution-score-g3v2-v1"
            or tuple(value.get("feature_names", ())) != WIDE_FEATURE_NAMES
            or value.get("feature_width") != WIDE_FEATURE_WIDTH
            or value.get("training_subset") != "strict-shield-certified-only"
            or value.get("certificate_or_identity_is_model_input") is not False
            or value.get("soft_target")
            != "0.5+0.5*tanh(score_advantage/256.0)"
            or value.get("cross_fit_unit") != "whole-seed"
        ):
            raise RuntimeError("strict score feature identity mismatch")
        try:
            raw_config = dict(_mapping(value["config"], "strict score config"))
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
                boost=_g3r2.BoostConfig(
                    **dict(_mapping(raw_config["boost"], "boost config"))
                ),
            )
            folds = tuple(
                ResolutionScoreFold.from_manifest(_mapping(row, "strict score fold"))
                for row in _sequence(value["folds"], "strict score folds")
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
                    value["dependency_identities"], "dependencies"
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
            raise RuntimeError("strict score manifest is malformed") from exc
        try:
            partition = _partition(seeds, config)
        except ValueError as exc:
            raise RuntimeError("strict score partition is malformed") from exc
        expected = tuple(
            sorted(
                (seed, index)
                for index, fold in enumerate(folds)
                for seed in fold.heldout_seeds
            )
        )
        if (
            set(raw_config) != {"folds", "partition_id", "seed_partition", "boost"}
            or len(folds) != config.folds
            or tuple(fold.heldout_seeds for fold in folds) != partition
            or assignments != expected
            or tuple(seed for seed, _index in assignments) != seeds
            or seeds != tuple(sorted(set(seeds)))
            or result.partition_sha256
            != _partition_sha(config.partition_id, partition)
            or not _SHA256_RE.fullmatch(result.partition_sha256)
            or not _SHA256_RE.fullmatch(result.training_dataset_sha256)
            or dependencies != _dependencies()
            or any(
                fold.head.config != config.boost
                or set(fold.training_shielded_seeds) & set(fold.heldout_seeds)
                or not set(fold.training_shielded_seeds)
                <= (set(seeds) - set(fold.heldout_seeds))
                for fold in folds
            )
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("strict score manifest is malformed")
        return result

    def checkpoint(
        self, metadata: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        body = {
            "schema": "irisu-r3i-resolution-score-checkpoint-g3v2-v1",
            "model_sha256": self.sha256,
            "model": self.manifest(),
            "metadata": dict(metadata or {}),
        }
        return {**body, "checkpoint_sha256": _sha256(body)}

    @classmethod
    def from_checkpoint(
        cls, value: Mapping[str, Any]
    ) -> tuple[ResolutionScoreG3V2, dict[str, object]]:
        if set(value) != {
            "schema",
            "model_sha256",
            "model",
            "metadata",
            "checkpoint_sha256",
        }:
            raise RuntimeError("strict score checkpoint envelope is malformed")
        body = {key: value[key] for key in value if key != "checkpoint_sha256"}
        if (
            value.get("schema")
            != "irisu-r3i-resolution-score-checkpoint-g3v2-v1"
            or value.get("checkpoint_sha256") != _sha256(body)
        ):
            raise RuntimeError("strict score checkpoint identity mismatch")
        model = cls.from_manifest(_mapping(value["model"], "strict score model"))
        if value.get("model_sha256") != model.sha256:
            raise RuntimeError("strict score checkpoint model identity mismatch")
        metadata = value["metadata"]
        if not isinstance(metadata, Mapping):
            raise RuntimeError("strict score checkpoint metadata is malformed")
        return model, dict(metadata)

    def predict(
        self, board: ResolutionScoreBoardV2, *, oof: bool = False
    ) -> tuple[ResolutionScorePredictionV2, ...]:
        known = self.seed_folds
        members: list[list[float]] = [[] for _ in board.features]
        for fold_index, fold in enumerate(self.folds):
            scores = fold.head.probabilities(board.features)
            for index, seed in enumerate(board.seeds):
                if oof:
                    if int(seed) not in known:
                        raise ValueError(
                            "OOF strict score prediction has an unknown seed"
                        )
                    if known[int(seed)] != fold_index:
                        continue
                members[index].append(float(scores[index]))
        output: list[ResolutionScorePredictionV2] = []
        for identity, values in zip(board.identities, members, strict=True):
            if not values:
                raise RuntimeError("strict score prediction has no eligible fold")
            mean = float(np.mean(values))
            prediction = ResolutionScorePredictionV2(
                identity,
                mean,
                float(np.std(values)),
                implied_score_advantage(mean),
            )
            prediction.validate()
            output.append(prediction)
        return tuple(output)


def train_resolution_score_g3v2(
    board: ResolutionScoreBoardV2,
    *,
    config: ResolutionScoreConfig | None = None,
) -> ResolutionScoreG3V2:
    resolved = ResolutionScoreConfig() if config is None else config
    partition = _partition(board.unique_seeds, resolved)
    all_seeds = set(board.unique_seeds)
    targets = soft_score_target(board.score_advantages)
    folds: list[ResolutionScoreFold] = []
    for heldout in partition:
        selected = board.strict_certified & np.asarray(
            [int(seed) in all_seeds - set(heldout) for seed in board.seeds]
        )
        supported = tuple(
            sorted(set(int(seed) for seed in board.seeds[selected]))
        )
        if not selected.any() or len(supported) < 2:
            raise ValueError("strict score fold lacks certified training support")
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
                supported,
            )
        )
    assignments = tuple(
        sorted(
            (seed, index)
            for index, heldout in enumerate(partition)
            for seed in heldout
        )
    )
    return ResolutionScoreG3V2(
        resolved,
        tuple(folds),
        assignments,
        board.unique_seeds,
        _partition_sha(resolved.partition_id, partition),
        board.sha256,
        _dependencies(),
    )


def select_strict_score_candidate(
    predictions: Sequence[ResolutionScorePredictionV2],
    certificates: Sequence[StrictShieldCertificate],
    *,
    incumbent: ScoreCandidateIdentity,
) -> ScoreCandidateIdentity:
    """Identity-bound lexicographic selection behind the strict shield."""

    if incumbent.ordinal != 0:
        raise ValueError("incumbent must have ordinal zero")
    predictions_by_identity: dict[
        ScoreCandidateIdentity, ResolutionScorePredictionV2
    ] = {}
    ordinal_identity: dict[int, ScoreCandidateIdentity] = {}
    for prediction in predictions:
        prediction.validate()
        identity = prediction.identity
        if (identity.seed, identity.query_id) != (incumbent.seed, incumbent.query_id):
            raise ValueError("score predictions mix seed or query identity")
        if identity in predictions_by_identity or identity.ordinal in ordinal_identity:
            raise ValueError("score predictions contain duplicate identity")
        predictions_by_identity[identity] = prediction
        ordinal_identity[identity.ordinal] = identity
    certificate_by_identity: dict[
        ScoreCandidateIdentity, StrictShieldCertificate
    ] = {}
    certificate_ordinals: set[int] = set()
    for certificate in certificates:
        identity = certificate.identity
        if (identity.seed, identity.query_id) != (incumbent.seed, incumbent.query_id):
            raise ValueError("score certificates mix seed or query identity")
        if (
            identity in certificate_by_identity
            or identity.ordinal in certificate_ordinals
        ):
            raise ValueError("score certificates contain duplicate identity")
        if identity not in predictions_by_identity:
            raise ValueError("score certificate lacks an exact prediction identity")
        certificate_by_identity[identity] = certificate
        certificate_ordinals.add(identity.ordinal)
    eligible = [
        predictions_by_identity[identity]
        for identity in certificate_by_identity
        if predictions_by_identity[identity].soft_score_mean > 0.5
    ]
    if not eligible:
        return incumbent
    return min(
        eligible,
        key=lambda row: (-row.soft_score_mean, row.identity.ordinal),
    ).identity


def save_resolution_score_checkpoint_v2(
    path: str | os.PathLike[str],
    model: ResolutionScoreG3V2,
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
        raise RuntimeError("strict score destination is occupied or indirect")
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
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    if not linked:
        raise RuntimeError("strict score checkpoint creation failed")
    return hashlib.sha256(encoded).hexdigest()


def load_resolution_score_checkpoint_v2(
    path: str | os.PathLike[str],
    *,
    expected_metadata: Mapping[str, object] | None = None,
    expected_partition_sha256: str | None = None,
    expected_model_sha256: str | None = None,
) -> tuple[ResolutionScoreG3V2, dict[str, object]]:
    source = os.path.abspath(os.fspath(path))
    before = os.lstat(source)
    if (
        not os.path.isfile(source)
        or os.path.islink(source)
        or os.path.realpath(source) != source
        or before.st_nlink != 1
    ):
        raise RuntimeError("strict score checkpoint source is indirect")
    descriptor = os.open(
        source,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise RuntimeError("strict score descriptor identity changed")
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
        raise RuntimeError("strict score checkpoint changed while being read")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("strict score checkpoint JSON is malformed") from exc
    if not isinstance(payload, Mapping) or raw != _canonical_bytes(payload):
        raise RuntimeError("strict score checkpoint encoding is noncanonical")
    model, metadata = ResolutionScoreG3V2.from_checkpoint(payload)
    if expected_metadata is not None and metadata != dict(expected_metadata):
        raise RuntimeError("strict score metadata expectation mismatch")
    if (
        expected_partition_sha256 is not None
        and model.partition_sha256 != expected_partition_sha256
    ):
        raise RuntimeError("strict score partition expectation mismatch")
    if expected_model_sha256 is not None and model.sha256 != expected_model_sha256:
        raise RuntimeError("strict score model expectation mismatch")
    return model, metadata
