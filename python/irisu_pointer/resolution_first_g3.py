"""Generation-03 causal, pair-factorized resolution learner.

The frozen generation-02 extractor remains the exact board/candidate binder.
Generation 03 adds only identity-free causal features and explicit observation
labels, then fits independent low-capacity histogram-Newton heads.  No learned
representation or gradient is shared between deployability and the diagnostic
bind/renewal/censor heads.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .resolution_first import FEATURE_NAMES
from .resolution_first_g2 import BoardBranchRecord, board_branch_records


_INTENTS = ("extend_anchor", "match_rotten", "steer_match")
_CATEGORIES = ("viable-anchor", "rotten-hazard", "fresh-match")
_INTENT_CATEGORY = dict(zip(_INTENTS, _CATEGORIES, strict=True))
_GEOMETRIES = (
    "analytic-strong",
    "analytic-weak",
    "close-strong",
    "deep-strong",
    "wide-strong",
)
_LIFECYCLES = ("confirmed", "dynamic_fresh", "rotten", "scripted_falling")
_QUERY_G2_FEATURES = (
    "gauge_fraction",
    "half_band",
    "level_scaled",
    "score_log",
    "qualifying_clears_log",
    "body_count_scaled",
    "visible_rot_debt_scaled",
    "spawn_interval_scaled",
)
_ENDPOINT_G2_FEATURES = (
    "x",
    "y",
    "vx",
    "vy",
    "size",
    "rot_timer",
)


def _endpoint_names(role: str) -> tuple[str, ...]:
    return (
        *(f"{role}_{name}" for name in _ENDPOINT_G2_FEATURES),
        f"{role}_lifetime_horizon_fraction",
        f"{role}_lifetime_log_overrun",
        f"{role}_expires_within_horizon",
        f"{role}_lifetime_sentinel",
        f"{role}_grouped",
        *(f"{role}_lifecycle_{name}" for name in _LIFECYCLES),
        f"{role}_shape_triangle",
        f"{role}_is_projectile",
    )


PAIR_FEATURE_NAMES = (
    *_QUERY_G2_FEATURES,
    "query_phase",
    "shot_phase",
    "spawn_phase",
    "board_rotten_fraction",
    "board_fresh_fraction",
    "board_falling_fraction",
    "board_confirmed_fraction",
    "board_projectile_fraction",
    "board_mean_size_log",
    *_endpoint_names("source"),
    *_endpoint_names("destination"),
    "pair_dx",
    "pair_dy",
    "pair_relative_vx",
    "pair_relative_vy",
    "pair_distance_sizes",
    "pair_log_size_ratio",
    "pair_same_color",
    "pair_same_chain",
    "pair_both_grouped",
    "source_nearest_clearance",
    "destination_nearest_clearance",
    "segment_nearest_clearance",
    *(f"intent_{name}" for name in _INTENTS),
    *(f"category_{name}" for name in _CATEGORIES),
)
GEOMETRY_FEATURE_NAMES = (
    "action_x",
    "action_y",
    "action_weak",
    "action_strong",
    "aim_source_dx",
    "aim_source_dy",
    "aim_destination_dx",
    "aim_destination_dy",
    "geometry_side_sizes",
    "geometry_below_sizes",
    "geometry_strong",
    "geometry_weak",
    "impact_x_sizes",
    "impact_y_sizes",
    *(f"geometry_{name}" for name in _GEOMETRIES),
    *(
        f"intent_geometry_{intent}_{geometry}"
        for intent in _INTENTS
        for geometry in _GEOMETRIES
    ),
)


def _sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} is malformed")
    return value


def _sequence(value: object, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} is malformed")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _number(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _feature(record: BoardBranchRecord, name: str) -> float:
    return float(record.features[FEATURE_NAMES.index(name)])


def _one_hot(value: str, choices: Sequence[str]) -> tuple[float, ...]:
    if value not in choices:
        raise ValueError(f"unsupported causal category {value!r}")
    return tuple(float(value == choice) for choice in choices)


def _lifetime_features(body: Mapping[str, Any], horizon: int) -> tuple[float, ...]:
    remaining = _number(body.get("remaining_lifetime"), "remaining_lifetime")
    if remaining < 0:
        raise ValueError("remaining_lifetime must be nonnegative")
    capped = min(remaining, float(horizon))
    overrun = max(remaining - horizon, 0.0)
    return (
        capped / horizon,
        min(math.log1p(overrun) / math.log1p(horizon), 4.0),
        float(remaining <= horizon),
        float(remaining == 99999.0),
    )


def _clearance(
    point: np.ndarray, others: Sequence[Mapping[str, Any]], scale: float
) -> float:
    if not others:
        return 8.0
    return min(
        math.hypot(
            _number(body.get("x"), "body x") - point[0],
            _number(body.get("y"), "body y") - point[1],
        )
        / scale
        for body in others
    )


def _segment_clearance(
    source: np.ndarray,
    destination: np.ndarray,
    others: Sequence[Mapping[str, Any]],
    scale: float,
) -> float:
    if not others:
        return 8.0
    delta = destination - source
    norm = float(np.dot(delta, delta))
    values: list[float] = []
    for body in others:
        point = np.asarray(
            (_number(body.get("x"), "body x"), _number(body.get("y"), "body y"))
        )
        fraction = (
            0.0
            if norm == 0
            else float(np.clip(np.dot(point - source, delta) / norm, 0.0, 1.0))
        )
        values.append(float(np.linalg.norm(point - (source + fraction * delta))) / scale)
    return min(values)


def _endpoint_features(
    role: str,
    record: BoardBranchRecord,
    body: Mapping[str, Any],
    horizon: int,
) -> tuple[float, ...]:
    prefix = f"{role}_"
    chain = body.get("chain_id")
    lifecycle = str(body.get("lifecycle"))
    shape = str(body.get("shape"))
    kind = str(body.get("kind"))
    return (
        *(_feature(record, prefix + name) for name in _ENDPOINT_G2_FEATURES),
        *_lifetime_features(body, horizon),
        float(chain not in (0, None)),
        *_one_hot(lifecycle, _LIFECYCLES),
        float(shape == "triangle"),
        float(kind == "projectile"),
    )


@dataclass(frozen=True, slots=True)
class G3Outcome:
    """Validated exact outcome and identity-free model inputs."""

    base: BoardBranchRecord
    pair_partition: int
    intent: str
    category: str
    geometry: str
    horizon_ticks: int
    survival_ticks: int
    renewal_count: int
    first_renewal_fraction: float | None
    second_renewal_fraction: float | None
    bind_ok: bool
    censored: bool
    pair_features: tuple[float, ...]
    geometry_features: tuple[float, ...]
    source_sha256: str

    def __post_init__(self) -> None:
        if not 0 <= self.renewal_count <= 2:
            raise ValueError("renewal_count is outside 0/1/2")
        if self.pair_partition < 0:
            raise ValueError("pair partition is negative")
        if len(self.pair_features) != len(PAIR_FEATURE_NAMES):
            raise ValueError("pair feature width mismatch")
        if len(self.geometry_features) != len(GEOMETRY_FEATURE_NAMES):
            raise ValueError("geometry feature width mismatch")
        if not all(
            math.isfinite(value)
            for value in (*self.pair_features, *self.geometry_features)
        ):
            raise ValueError("generation-03 feature is nonfinite")

    @property
    def seed(self) -> int:
        return self.base.seed

    @property
    def query_id(self) -> str:
        return self.base.query_id

    @property
    def ordinal(self) -> int:
        return self.base.ordinal

    @property
    def query_key(self) -> tuple[int, str]:
        return self.base.query_key

    @property
    def pair_key(self) -> tuple[int, str, int]:
        return self.seed, self.query_id, self.pair_partition

    @property
    def deployable(self) -> bool:
        return self.base.candidate_resolved

    def manifest(self) -> dict[str, object]:
        return {
            "base_sha256": self.base.sha256,
            "pair_partition": self.pair_partition,
            "intent": self.intent,
            "category": self.category,
            "geometry": self.geometry,
            "horizon_ticks": self.horizon_ticks,
            "survival_ticks": self.survival_ticks,
            "renewal_count": self.renewal_count,
            "first_renewal_fraction": self.first_renewal_fraction,
            "second_renewal_fraction": self.second_renewal_fraction,
            "bind_ok": self.bind_ok,
            "censored": self.censored,
            "pair_features": list(self.pair_features),
            "geometry_features": list(self.geometry_features),
            "source_sha256": self.source_sha256,
        }


def _outcome_features(
    record: BoardBranchRecord,
    candidate: Mapping[str, Any],
    observation: Mapping[str, Any],
    horizon: int,
) -> tuple[tuple[float, ...], tuple[float, ...], str, str, str]:
    pair = _mapping(candidate.get("pair"), "candidate pair")
    geometry_row = _mapping(candidate.get("geometry"), "candidate geometry")
    action = _mapping(candidate.get("action"), "candidate action")
    intent = str(pair.get("intent"))
    category = str(pair.get("category"))
    geometry = str(geometry_row.get("name"))
    if _INTENT_CATEGORY.get(intent) != category:
        raise ValueError("candidate intent/category mismatch")
    _one_hot(geometry, _GEOMETRIES)
    strength = str(geometry_row.get("strength"))
    if strength not in {"weak", "strong"}:
        raise ValueError("candidate geometry strength is invalid")
    expected_kind = 1 if strength == "weak" else 2
    if _integer(action.get("kind"), "action kind") != expected_kind:
        raise ValueError("action kind and geometry strength disagree")

    bodies = _sequence(observation.get("bodies"), "public bodies")
    by_id: dict[int, Mapping[str, Any]] = {}
    for supplied in bodies:
        body = _mapping(supplied, "public body")
        identifier = _integer(body.get("id"), "body id")
        if identifier in by_id:
            raise ValueError("public body identities are not unique")
        by_id[identifier] = body
    source_id = _integer(pair.get("source_body_id"), "source_body_id")
    destination_id = _integer(
        pair.get("destination_body_id"), "destination_body_id"
    )
    try:
        source_body, destination_body = by_id[source_id], by_id[destination_id]
    except KeyError as exc:
        raise ValueError("candidate endpoint is absent from observation") from exc
    supplied_chain = pair.get("destination_chain_id")
    if supplied_chain is not None and supplied_chain != destination_body.get("chain_id"):
        raise ValueError("candidate destination chain binding mismatch")

    count = len(bodies)
    lifecycle_counts = {
        name: sum(str(body.get("lifecycle")) == name for body in by_id.values())
        for name in _LIFECYCLES
    }
    if sum(lifecycle_counts.values()) != count:
        raise ValueError("unsupported public lifecycle")
    source_xy = np.asarray(
        (_number(source_body.get("x"), "source x"), _number(source_body.get("y"), "source y"))
    )
    destination_xy = np.asarray(
        (
            _number(destination_body.get("x"), "destination x"),
            _number(destination_body.get("y"), "destination y"),
        )
    )
    others = [
        body
        for identifier, body in by_id.items()
        if identifier not in {source_id, destination_id}
    ]
    source_size = _number(source_body.get("size"), "source size")
    destination_size = _number(destination_body.get("size"), "destination size")
    if source_size <= 0 or destination_size <= 0:
        raise ValueError("endpoint size must be positive")
    scale = max((source_size + destination_size) / 2.0, 1e-6)
    source_chain = source_body.get("chain_id")
    destination_chain = destination_body.get("chain_id")
    source_grouped = source_chain not in (0, None)
    destination_grouped = destination_chain not in (0, None)
    query_features = tuple(_feature(record, name) for name in _QUERY_G2_FEATURES)
    pair_features = (
        *query_features,
        *record.phase_features,
        *(lifecycle_counts[name] / count for name in _LIFECYCLES),
        sum(str(body.get("kind")) == "projectile" for body in by_id.values()) / count,
        math.log1p(
            sum(_number(body.get("size"), "body size") for body in by_id.values())
            / count
        ),
        *_endpoint_features("source", record, source_body, horizon),
        *_endpoint_features("destination", record, destination_body, horizon),
        _feature(record, "destination_x") - _feature(record, "source_x"),
        _feature(record, "destination_y") - _feature(record, "source_y"),
        _feature(record, "destination_vx") - _feature(record, "source_vx"),
        _feature(record, "destination_vy") - _feature(record, "source_vy"),
        _number(pair.get("distance_sizes"), "pair distance_sizes"),
        math.log(source_size / destination_size),
        float(source_body.get("color") == destination_body.get("color")),
        float(
            source_grouped
            and destination_grouped
            and source_chain == destination_chain
        ),
        float(source_grouped and destination_grouped),
        min(_clearance(source_xy, others, scale), 8.0),
        min(_clearance(destination_xy, others, scale), 8.0),
        min(_segment_clearance(source_xy, destination_xy, others, scale), 8.0),
        *_one_hot(intent, _INTENTS),
        *_one_hot(category, _CATEGORIES),
    )
    action_x = _number(action.get("x_norm"), "action x")
    action_y = _number(action.get("y_norm"), "action y")
    geometry_hot = _one_hot(geometry, _GEOMETRIES)
    intent_hot = _one_hot(intent, _INTENTS)
    geometry_features = (
        action_x,
        action_y,
        float(expected_kind == 1),
        float(expected_kind == 2),
        action_x - _feature(record, "source_x"),
        action_y - _feature(record, "source_y"),
        action_x - _feature(record, "destination_x"),
        action_y - _feature(record, "destination_y"),
        _number(geometry_row.get("side_sizes"), "geometry side_sizes"),
        _number(geometry_row.get("below_sizes"), "geometry below_sizes"),
        float(strength == "strong"),
        float(strength == "weak"),
        _feature(record, "impact_x_sizes"),
        _feature(record, "impact_y_sizes"),
        *geometry_hot,
        *(left * right for left in intent_hot for right in geometry_hot),
    )
    return pair_features, geometry_features, intent, category, geometry


def g3_outcomes(entries: Iterable[Mapping[str, Any]]) -> tuple[G3Outcome, ...]:
    """Validate exact outcomes and wrap the frozen G2 extraction.

    Raw body and chain identities are used only to bind endpoint roles and to
    create a query-local pair partition.  Neither identity nor partition index
    appears in either model feature vector.
    """

    supplied = tuple(entries)
    bases = board_branch_records(supplied)
    output: list[G3Outcome] = []
    offset = 0
    for entry in supplied:
        query = _mapping(entry.get("exact_query"), "exact_query")
        outcomes = _sequence(query.get("outcomes"), "exact outcomes")
        observation = _mapping(
            entry.get("pre_query_public_observation"), "public observation"
        )
        pair_labels: dict[tuple[int, int], int] = {}
        declared: dict[tuple[int, int], int] = {}
        reverse_declared: dict[int, tuple[int, int]] = {}
        pair_metadata: dict[tuple[int, int], tuple[str, str, float]] = {}
        pair_geometries: dict[tuple[int, int], set[str]] = defaultdict(set)
        for local, raw in enumerate(outcomes):
            outcome_row = _mapping(raw, "exact outcome")
            candidate = _mapping(outcome_row.get("candidate"), "candidate")
            pair = _mapping(candidate.get("pair"), "candidate pair")
            endpoint_pair = (
                _integer(pair.get("source_body_id"), "source_body_id"),
                _integer(pair.get("destination_body_id"), "destination_body_id"),
            )
            partition = pair_labels.setdefault(endpoint_pair, len(pair_labels))
            raw_partition = _integer(
                candidate.get("pair_ordinal"), "candidate pair_ordinal"
            )
            if endpoint_pair in declared and declared[endpoint_pair] != raw_partition:
                raise ValueError("one endpoint pair has multiple declared partitions")
            if (
                raw_partition in reverse_declared
                and reverse_declared[raw_partition] != endpoint_pair
            ):
                raise ValueError("declared pair partition aliases endpoint pairs")
            declared[endpoint_pair] = raw_partition
            reverse_declared[raw_partition] = endpoint_pair

            result = _mapping(outcome_row.get("outcome"), "outcome")
            ledger = _mapping(outcome_row.get("ledger"), "ledger")
            horizon = _integer(result.get("horizon_ticks"), "horizon_ticks")
            survival = _integer(result.get("survival_ticks"), "survival_ticks")
            if horizon <= 0 or not 0 <= survival <= horizon:
                raise ValueError("outcome horizon/survival is invalid")
            ticks = tuple(
                _integer(value, "renewal tick")
                for value in _sequence(result.get("renewal_ticks"), "renewal ticks")
            )
            if any(
                tick <= 0 or tick > survival
                for tick in ticks
            ) or any(left >= right for left, right in zip(ticks, ticks[1:])):
                raise ValueError("renewal ticks are not strictly ordered in survival")
            renewal_count = min(len(ticks), 2)
            resolved = bool(result.get("renewals_resolved"))
            if resolved != (renewal_count == 2):
                raise ValueError("renewals_resolved disagrees with renewal_count")
            unresolved = tuple(
                str(value)
                for value in _sequence(ledger.get("unresolved"), "unresolved ledger")
            )
            rebind_failed = bool(ledger.get("continuation_rebind_failed"))
            if ("continuation-rebind-failed" in unresolved) != rebind_failed:
                raise ValueError("continuation rebind label is inconsistent")
            bind_ok = not rebind_failed and not unresolved
            censored = bool(
                bind_ok
                and not resolved
                and survival == horizon
                and not bool(result.get("terminal"))
                and not bool(result.get("gauge_failure"))
                and not bool(result.get("new_terminal"))
            )
            base = bases[offset + local]
            if base.ordinal != local or base.candidate_resolved != (
                resolved
                and bind_ok
                and not unresolved
                and base.b2 is not None
            ):
                raise ValueError("frozen G2 and G3 exact labels disagree")
            pair_features, geometry_features, intent, category, geometry = (
                _outcome_features(base, candidate, observation, horizon)
            )
            metadata = (
                intent,
                category,
                _number(pair.get("distance_sizes"), "pair distance_sizes"),
            )
            if (
                endpoint_pair in pair_metadata
                and pair_metadata[endpoint_pair] != metadata
            ):
                raise ValueError("one endpoint pair has inconsistent causal metadata")
            if geometry in pair_geometries[endpoint_pair]:
                raise ValueError("one endpoint pair repeats a geometry")
            pair_metadata[endpoint_pair] = metadata
            pair_geometries[endpoint_pair].add(geometry)
            output.append(
                G3Outcome(
                    base=base,
                    pair_partition=partition,
                    intent=intent,
                    category=category,
                    geometry=geometry,
                    horizon_ticks=horizon,
                    survival_ticks=survival,
                    renewal_count=renewal_count,
                    first_renewal_fraction=(
                        ticks[0] / horizon if renewal_count >= 1 else None
                    ),
                    second_renewal_fraction=(
                        ticks[1] / horizon if renewal_count >= 2 else None
                    ),
                    bind_ok=bind_ok,
                    censored=censored,
                    pair_features=pair_features,
                    geometry_features=geometry_features,
                    source_sha256=_sha256(outcome_row),
                )
            )
        offset += len(outcomes)
    if offset != len(bases) or not output:
        raise RuntimeError("frozen G2/G3 outcome cardinality mismatch")
    return tuple(output)


@dataclass(frozen=True, slots=True)
class G3Dataset:
    outcomes: tuple[G3Outcome, ...]

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError("generation-03 dataset is empty")
        grouped: dict[tuple[int, str], list[G3Outcome]] = defaultdict(list)
        for row in self.outcomes:
            grouped[row.query_key].append(row)
        for key, rows in grouped.items():
            if [row.ordinal for row in rows] != list(range(len(rows))):
                raise ValueError(f"query {key!r} is not incumbent-first/contiguous")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted({row.seed for row in self.outcomes}))

    @property
    def pair_matrix(self) -> np.ndarray:
        return np.asarray([row.pair_features for row in self.outcomes], dtype=np.float64)

    @property
    def geometry_matrix(self) -> np.ndarray:
        return np.asarray(
            [row.geometry_features for row in self.outcomes], dtype=np.float64
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3i-causal-pair-dataset-g3-v1",
            "frozen_extractor": "resolution_first_g2.board_branch_records",
            "pair_feature_names": list(PAIR_FEATURE_NAMES),
            "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
            "absolute_id_input": False,
            "pair_partition_input": False,
            "absolute_chain_id_input": False,
            "absolute_tick_input": False,
            "records": [_sha256(row.manifest()) for row in self.outcomes],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class HistogramNewtonConfig:
    bins: int = 16
    pair_steps: int = 48
    geometry_steps: int = 48
    learning_rate: float = 0.06
    l2: float = 4.0
    min_leaf: int = 8
    max_leaf_value: float = 2.0

    def __post_init__(self) -> None:
        if min(self.bins, self.pair_steps, self.geometry_steps, self.min_leaf) < 1:
            raise ValueError("histogram learner sizes must be positive")
        if (
            not 0 < self.learning_rate <= 1
            or self.l2 <= 0
            or self.max_leaf_value <= 0
        ):
            raise ValueError("histogram learner numeric configuration is invalid")


@dataclass(frozen=True, slots=True)
class HistogramStump:
    feature: int
    threshold: float
    left: float
    right: float

    def predict(self, values: np.ndarray) -> np.ndarray:
        return np.where(values[:, self.feature] <= self.threshold, self.left, self.right)

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> HistogramStump:
        try:
            result = cls(
                int(value["feature"]),
                float(value["threshold"]),
                float(value["left"]),
                float(value["right"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("G3 histogram stump is malformed") from exc
        if (
            result.feature < 0
            or not all(
                math.isfinite(item)
                for item in (result.threshold, result.left, result.right)
            )
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("G3 histogram stump is malformed")
        return result


@dataclass(frozen=True, slots=True)
class HistogramNewtonBinary:
    base_logit: float
    stumps: tuple[HistogramStump, ...]
    feature_count: int

    def logits(self, values: np.ndarray, offset: np.ndarray | None = None) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.feature_count:
            raise ValueError("histogram prediction feature width mismatch")
        result = np.full(matrix.shape[0], self.base_logit, dtype=np.float64)
        if offset is not None:
            supplied = np.asarray(offset, dtype=np.float64)
            if supplied.shape != result.shape or not np.isfinite(supplied).all():
                raise ValueError("histogram prediction offset is malformed")
            result += supplied
        for stump in self.stumps:
            result += stump.predict(matrix)
        return result

    def probabilities(
        self, values: np.ndarray, offset: np.ndarray | None = None
    ) -> np.ndarray:
        logits = np.clip(self.logits(values, offset), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def manifest(self) -> dict[str, object]:
        return {
            "base_logit": self.base_logit,
            "feature_count": self.feature_count,
            "stumps": [stump.manifest() for stump in self.stumps],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())

    @classmethod
    def from_manifest(
        cls, value: Mapping[str, object]
    ) -> HistogramNewtonBinary:
        raw_stumps = value.get("stumps")
        if not isinstance(raw_stumps, Sequence) or isinstance(
            raw_stumps, (str, bytes)
        ):
            raise RuntimeError("G3 histogram ensemble is malformed")
        try:
            result = cls(
                float(value["base_logit"]),
                tuple(
                    HistogramStump.from_manifest(_mapping(row, "histogram stump"))
                    for row in raw_stumps
                ),
                int(value["feature_count"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("G3 histogram ensemble is malformed") from exc
        if (
            result.feature_count < 1
            or not math.isfinite(result.base_logit)
            or any(stump.feature >= result.feature_count for stump in result.stumps)
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("G3 histogram ensemble is malformed")
        return result

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        *,
        config: HistogramNewtonConfig,
        steps: int,
        offset: np.ndarray | None = None,
        zero_base: bool = False,
    ) -> HistogramNewtonBinary:
        x = np.asarray(values, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if (
            x.ndim != 2
            or y.shape != (len(x),)
            or w.shape != y.shape
            or not np.isfinite(x).all()
            or not np.isfinite(y).all()
            or not np.isfinite(w).all()
            or np.any(w <= 0)
            or set(np.unique(y)) != {0.0, 1.0}
        ):
            raise ValueError("histogram binary supervision is malformed")
        external = np.zeros(len(x)) if offset is None else np.asarray(offset, dtype=np.float64)
        if external.shape != y.shape or not np.isfinite(external).all():
            raise ValueError("histogram training offset is malformed")
        prevalence = float(np.average(y, weights=w))
        base = 0.0 if zero_base else math.log(
            np.clip(prevalence, 1e-6, 1 - 1e-6)
            / np.clip(1 - prevalence, 1e-6, 1 - 1e-6)
        )
        prediction = external + base
        thresholds: list[np.ndarray] = []
        quantiles = np.linspace(0.0, 1.0, config.bins + 1)[1:-1]
        for column in range(x.shape[1]):
            unique = np.unique(x[:, column])
            if len(unique) < 2:
                thresholds.append(np.empty(0))
            else:
                candidates = np.unique(
                    np.quantile(unique, quantiles, method="midpoint")
                )
                thresholds.append(
                    candidates[
                        (candidates > unique[0]) & (candidates < unique[-1])
                    ]
                )
        stumps: list[HistogramStump] = []
        for _ in range(steps):
            probability = 1.0 / (1.0 + np.exp(-np.clip(prediction, -40.0, 40.0)))
            gradient = w * (y - probability)
            hessian = w * probability * (1.0 - probability)
            total_g, total_h = float(gradient.sum()), float(hessian.sum())
            parent = total_g * total_g / (total_h + config.l2)
            best: tuple[float, int, float, float, float] | None = None
            for feature, candidates in enumerate(thresholds):
                for threshold in candidates:
                    left_mask = x[:, feature] <= threshold
                    left_count = int(left_mask.sum())
                    if (
                        left_count < config.min_leaf
                        or len(x) - left_count < config.min_leaf
                    ):
                        continue
                    left_g = float(gradient[left_mask].sum())
                    left_h = float(hessian[left_mask].sum())
                    right_g, right_h = total_g - left_g, total_h - left_h
                    gain = (
                        left_g * left_g / (left_h + config.l2)
                        + right_g * right_g / (right_h + config.l2)
                        - parent
                    )
                    candidate = (gain, -feature, -float(threshold), left_g, left_h)
                    if best is None or candidate[:3] > best[:3]:
                        best = candidate
            if best is None or best[0] <= 1e-12:
                break
            _, negative_feature, negative_threshold, left_g, left_h = best
            feature, threshold = -negative_feature, -negative_threshold
            left = config.learning_rate * float(
                np.clip(
                    left_g / (left_h + config.l2),
                    -config.max_leaf_value,
                    config.max_leaf_value,
                )
            )
            right_g, right_h = total_g - left_g, total_h - left_h
            right = config.learning_rate * float(
                np.clip(
                    right_g / (right_h + config.l2),
                    -config.max_leaf_value,
                    config.max_leaf_value,
                )
            )
            stump = HistogramStump(feature, threshold, left, right)
            stumps.append(stump)
            prediction += stump.predict(x)
        return cls(base, tuple(stumps), x.shape[1])


@dataclass(frozen=True, slots=True)
class PairFactorizedHead:
    pair: HistogramNewtonBinary
    geometry: HistogramNewtonBinary

    def logits(self, pair_values: np.ndarray, geometry_values: np.ndarray) -> np.ndarray:
        pair_logits = self.pair.logits(pair_values)
        return self.geometry.logits(geometry_values, pair_logits)

    def probabilities(
        self, pair_values: np.ndarray, geometry_values: np.ndarray
    ) -> np.ndarray:
        logits = np.clip(self.logits(pair_values, geometry_values), -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))

    def manifest(self) -> dict[str, object]:
        return {
            "pair": self.pair.manifest(),
            "geometry": self.geometry.manifest(),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> PairFactorizedHead:
        try:
            result = cls(
                HistogramNewtonBinary.from_manifest(
                    _mapping(value["pair"], "pair head")
                ),
                HistogramNewtonBinary.from_manifest(
                    _mapping(value["geometry"], "geometry head")
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("G3 pair-factorized head is malformed") from exc
        if result.manifest() != dict(value):
            raise RuntimeError("G3 pair-factorized head is malformed")
        return result

    @classmethod
    def fit(
        cls,
        pair_values: np.ndarray,
        geometry_values: np.ndarray,
        targets: np.ndarray,
        weights: np.ndarray,
        config: HistogramNewtonConfig,
    ) -> PairFactorizedHead:
        pair = HistogramNewtonBinary.fit(
            pair_values,
            targets,
            weights,
            config=config,
            steps=config.pair_steps,
        )
        pair_logits = pair.logits(pair_values)
        geometry = HistogramNewtonBinary.fit(
            geometry_values,
            targets,
            weights,
            config=config,
            steps=config.geometry_steps,
            offset=pair_logits,
            zero_base=True,
        )
        return cls(pair, geometry)


@dataclass(frozen=True, slots=True)
class G3Bag:
    deployability: PairFactorizedHead
    bind: PairFactorizedHead
    first_renewal: PairFactorizedHead
    second_renewal: PairFactorizedHead
    censor: PairFactorizedHead
    head_counts: tuple[tuple[str, int, int], ...]

    def manifest(self) -> dict[str, object]:
        return {
            "heads": {
                name: getattr(self, name).manifest()
                for name in (
                    "deployability",
                    "bind",
                    "first_renewal",
                    "second_renewal",
                    "censor",
                )
            },
            "head_counts": [list(value) for value in self.head_counts],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> G3Bag:
        heads = _mapping(value.get("heads"), "G3 bag heads")
        names = (
            "deployability",
            "bind",
            "first_renewal",
            "second_renewal",
            "censor",
        )
        if set(heads) != set(names):
            raise RuntimeError("G3 bag head inventory is malformed")
        raw_counts = _sequence(value.get("head_counts"), "G3 head counts")
        try:
            counts = tuple(
                (str(row[0]), int(row[1]), int(row[2]))
                for row in raw_counts
                if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
            )
            result = cls(
                *(PairFactorizedHead.from_manifest(heads[name]) for name in names),
                counts,
            )
        except (TypeError, ValueError, IndexError) as exc:
            raise RuntimeError("G3 bag is malformed") from exc
        if (
            tuple(name for name, _, _ in counts) != names
            or any(min(negative, positive) < 1 for _, negative, positive in counts)
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("G3 bag is malformed")
        return result


@dataclass(frozen=True, slots=True)
class G3Config:
    folds: int = 5
    bags_per_fold: int = 3
    bootstrap_attempts: int = 128
    random_seed: int = 2026073003
    histogram: HistogramNewtonConfig = HistogramNewtonConfig()

    def __post_init__(self) -> None:
        if min(self.folds, self.bags_per_fold, self.bootstrap_attempts) < 1:
            raise ValueError("generation-03 cross-fit sizes must be positive")


def _folds(seeds: Sequence[int], count: int) -> tuple[tuple[int, ...], ...]:
    if count < 2 or len(seeds) < count + 1:
        raise ValueError("whole-seed cross-fit needs more seeds than folds")
    ordered = sorted(
        seeds,
        key=lambda seed: hashlib.sha256(f"g3-fold|{seed}".encode()).digest(),
    )
    return tuple(tuple(ordered[index::count]) for index in range(count))


def _balanced_weights(
    rows: Sequence[G3Outcome],
    selected: np.ndarray,
    targets: np.ndarray,
    multiplicity: Mapping[int, int],
) -> np.ndarray:
    by_seed_pair: dict[int, dict[tuple[int, str, int], int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for index in selected:
        row = rows[int(index)]
        by_seed_pair[row.seed][row.pair_key] += 1
    base = np.asarray(
        [
            multiplicity[rows[int(index)].seed]
            / len(by_seed_pair[rows[int(index)].seed])
            / by_seed_pair[rows[int(index)].seed][rows[int(index)].pair_key]
            for index in selected
        ],
        dtype=np.float64,
    )
    for label in (0.0, 1.0):
        mask = targets == label
        total = float(base[mask].sum())
        if total <= 0:
            raise ValueError("head supervision lacks one class")
        base[mask] *= 0.5 / total
    return base


def _fit_head(
    rows: Sequence[G3Outcome],
    pair: np.ndarray,
    geometry: np.ndarray,
    mask: np.ndarray,
    target: np.ndarray,
    multiplicity: Mapping[int, int],
    config: HistogramNewtonConfig,
) -> tuple[PairFactorizedHead, tuple[int, int]]:
    selected = np.flatnonzero(
        mask & np.asarray([row.seed in multiplicity for row in rows])
    )
    labels = target[selected].astype(np.float64)
    weights = _balanced_weights(rows, selected, labels, multiplicity)
    return (
        PairFactorizedHead.fit(
            pair[selected], geometry[selected], labels, weights, config
        ),
        (int(np.sum(labels == 0)), int(np.sum(labels == 1))),
    )


def _fit_bag(
    dataset: G3Dataset,
    multiplicity: Mapping[int, int],
    config: HistogramNewtonConfig,
) -> G3Bag:
    rows = dataset.outcomes
    pair, geometry = dataset.pair_matrix, dataset.geometry_matrix
    bind = np.asarray([row.bind_ok for row in rows])
    first = np.asarray([row.renewal_count >= 1 for row in rows])
    second = np.asarray([row.renewal_count >= 2 for row in rows])
    deployable = np.asarray([row.deployable for row in rows])
    censor = np.asarray([row.censored for row in rows])
    all_rows = np.ones(len(rows), dtype=bool)
    specifications = (
        ("deployability", all_rows, deployable),
        ("bind", all_rows, bind),
        ("first_renewal", bind, first),
        ("second_renewal", bind & first, second),
        ("censor", bind, censor),
    )
    heads: dict[str, PairFactorizedHead] = {}
    counts: list[tuple[str, int, int]] = []
    for name, mask, target in specifications:
        head, (negative, positive) = _fit_head(
            rows, pair, geometry, mask, target, multiplicity, config
        )
        heads[name] = head
        counts.append((name, negative, positive))
    return G3Bag(
        heads["deployability"],
        heads["bind"],
        heads["first_renewal"],
        heads["second_renewal"],
        heads["censor"],
        tuple(counts),
    )


@dataclass(frozen=True, slots=True)
class G3Fold:
    heldout_seeds: tuple[int, ...]
    bags: tuple[G3Bag, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "heldout_seeds": list(self.heldout_seeds),
            "bags": [bag.manifest() for bag in self.bags],
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> G3Fold:
        try:
            heldout = tuple(
                _integer(seed, "heldout seed")
                for seed in _sequence(value["heldout_seeds"], "heldout seeds")
            )
            bags = tuple(
                G3Bag.from_manifest(_mapping(row, "G3 bag"))
                for row in _sequence(value["bags"], "G3 bags")
            )
            result = cls(heldout, bags)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("G3 fold is malformed") from exc
        if (
            not heldout
            or not bags
            or tuple(sorted(set(heldout))) != heldout
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("G3 fold is malformed")
        return result


@dataclass(frozen=True, slots=True)
class G3Prediction:
    seed: int
    query_id: str
    ordinal: int
    deployability_mean: float
    deployability_std: float
    bind_mean: float
    first_renewal_mean: float
    second_renewal_mean: float
    censor_mean: float
    causal_product_mean: float

    @property
    def primary_score(self) -> float:
        return self.deployability_mean


@dataclass(frozen=True, slots=True)
class ResolutionFirstG3:
    config: G3Config
    folds: tuple[G3Fold, ...]
    fold_by_seed: tuple[tuple[int, int], ...]
    training_dataset_sha256: str

    @property
    def seed_folds(self) -> dict[int, int]:
        return dict(self.fold_by_seed)

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3i-resolution-first-g3-v1",
            "training_dataset_sha256": self.training_dataset_sha256,
            "pair_feature_names": list(PAIR_FEATURE_NAMES),
            "geometry_feature_names": list(GEOMETRY_FEATURE_NAMES),
            "primary_target": "frozen-g2-candidate_resolved",
            "primary_score": "direct-pair-plus-geometry-residual",
            "diagnostic_heads": [
                "bind",
                "first-renewal-given-bind",
                "second-renewal-given-bind-and-first",
                "censor-given-bind",
            ],
            "downstream_multitask_gradients": False,
            "pair_partition_input": False,
            "cross_fit_unit": "whole-seed",
            "fold_by_seed": [list(value) for value in self.fold_by_seed],
            "folds": [fold.manifest() for fold in self.folds],
            "config": asdict(self.config),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> ResolutionFirstG3:
        if (
            value.get("schema") != "irisu-r3i-resolution-first-g3-v1"
            or tuple(value.get("pair_feature_names", ())) != PAIR_FEATURE_NAMES
            or tuple(value.get("geometry_feature_names", ()))
            != GEOMETRY_FEATURE_NAMES
            or value.get("primary_target") != "frozen-g2-candidate_resolved"
            or value.get("primary_score")
            != "direct-pair-plus-geometry-residual"
            or value.get("downstream_multitask_gradients") is not False
            or value.get("pair_partition_input") is not False
            or value.get("cross_fit_unit") != "whole-seed"
        ):
            raise RuntimeError("G3 checkpoint feature identity mismatch")
        try:
            config_raw = dict(_mapping(value["config"], "G3 config"))
            histogram_raw = dict(
                _mapping(config_raw.pop("histogram"), "G3 histogram config")
            )
            config = G3Config(
                **config_raw, histogram=HistogramNewtonConfig(**histogram_raw)
            )
            folds = tuple(
                G3Fold.from_manifest(_mapping(row, "G3 fold"))
                for row in _sequence(value["folds"], "G3 folds")
            )
            fold_by_seed = tuple(
                (
                    _integer(row[0], "fold seed"),
                    _integer(row[1], "fold index"),
                )
                for row in _sequence(value["fold_by_seed"], "G3 fold mapping")
                if isinstance(row, Sequence) and not isinstance(row, (str, bytes))
            )
            result = cls(
                config,
                folds,
                fold_by_seed,
                str(value["training_dataset_sha256"]),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise RuntimeError("G3 checkpoint manifest is malformed") from exc
        expected_assignments = tuple(
            sorted(
                (seed, index)
                for index, fold in enumerate(folds)
                for seed in fold.heldout_seeds
            )
        )
        heads = (
            head
            for fold in folds
            for bag in fold.bags
            for head in (
                bag.deployability,
                bag.bind,
                bag.first_renewal,
                bag.second_renewal,
                bag.censor,
            )
        )
        if (
            len(folds) != config.folds
            or any(len(fold.bags) != config.bags_per_fold for fold in folds)
            or fold_by_seed != expected_assignments
            or len(result.training_dataset_sha256) != 64
            or any(
                head.pair.feature_count != len(PAIR_FEATURE_NAMES)
                or head.geometry.feature_count != len(GEOMETRY_FEATURE_NAMES)
                for head in heads
            )
            or result.manifest() != dict(value)
        ):
            raise RuntimeError("G3 checkpoint manifest is malformed")
        return result

    def checkpoint(
        self, *, metadata: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        return {
            "schema": "irisu-r3i-resolution-first-checkpoint-g3-v1",
            "model_sha256": self.sha256,
            "model": self.manifest(),
            "metadata": dict(metadata or {}),
        }

    @classmethod
    def from_checkpoint(
        cls, value: Mapping[str, object]
    ) -> tuple[ResolutionFirstG3, dict[str, object]]:
        if value.get("schema") != "irisu-r3i-resolution-first-checkpoint-g3-v1":
            raise RuntimeError("G3 checkpoint schema mismatch")
        model = cls.from_manifest(_mapping(value.get("model"), "G3 model"))
        if value.get("model_sha256") != model.sha256:
            raise RuntimeError("G3 checkpoint model identity mismatch")
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise RuntimeError("G3 checkpoint metadata is malformed")
        if set(value) != {"schema", "model_sha256", "model", "metadata"}:
            raise RuntimeError("G3 checkpoint envelope is malformed")
        return model, dict(metadata)

    def predict(
        self, dataset: G3Dataset, *, oof: bool = False
    ) -> tuple[G3Prediction, ...]:
        pair, geometry = dataset.pair_matrix, dataset.geometry_matrix
        known = self.seed_folds
        output: list[G3Prediction] = []
        for index, row in enumerate(dataset.outcomes):
            if oof:
                if row.seed not in known:
                    raise ValueError("OOF prediction requested for an unknown seed")
                selected_folds = (self.folds[known[row.seed]],)
            else:
                selected_folds = self.folds
            selected_bags = [
                bag for fold in selected_folds for bag in fold.bags
            ]
            pair_row, geometry_row = pair[index : index + 1], geometry[index : index + 1]
            values: dict[str, list[float]] = defaultdict(list)
            for bag in selected_bags:
                for name in (
                    "deployability",
                    "bind",
                    "first_renewal",
                    "second_renewal",
                    "censor",
                ):
                    values[name].append(
                        float(
                            getattr(bag, name).probabilities(
                                pair_row, geometry_row
                            )[0]
                        )
                    )
            causal = [
                left * middle * right
                for left, middle, right in zip(
                    values["bind"],
                    values["first_renewal"],
                    values["second_renewal"],
                    strict=True,
                )
            ]
            output.append(
                G3Prediction(
                    row.seed,
                    row.query_id,
                    row.ordinal,
                    float(np.mean(values["deployability"])),
                    float(np.std(values["deployability"])),
                    float(np.mean(values["bind"])),
                    float(np.mean(values["first_renewal"])),
                    float(np.mean(values["second_renewal"])),
                    float(np.mean(values["censor"])),
                    float(np.mean(causal)),
                )
            )
        return tuple(output)


def train_resolution_first_g3(
    dataset: G3Dataset, *, config: G3Config | None = None
) -> ResolutionFirstG3:
    """Fit deterministic whole-seed cross-fit bags.

    Every head owns its pair and geometry learners.  Technical bind failures
    are excluded from renewal/censor supervision, while the independent direct
    deployability head retains the frozen official target on every row.
    """

    resolved = G3Config() if config is None else config
    seed_folds = _folds(dataset.seeds, resolved.folds)
    fold_by_seed = tuple(
        sorted(
            (seed, fold_index)
            for fold_index, seeds in enumerate(seed_folds)
            for seed in seeds
        )
    )
    generator = np.random.default_rng(resolved.random_seed)
    folds: list[G3Fold] = []
    all_seeds = set(dataset.seeds)
    for heldout in seed_folds:
        available = tuple(sorted(all_seeds - set(heldout)))
        bags: list[G3Bag] = []
        for _bag in range(resolved.bags_per_fold):
            for _attempt in range(resolved.bootstrap_attempts):
                sampled = generator.choice(
                    np.asarray(available, dtype=np.int64),
                    size=len(available),
                    replace=True,
                )
                multiplicity = {
                    int(seed): int(np.sum(sampled == seed))
                    for seed in np.unique(sampled)
                }
                try:
                    bag = _fit_bag(dataset, multiplicity, resolved.histogram)
                except ValueError:
                    continue
                bags.append(bag)
                break
            else:
                raise RuntimeError(
                    "whole-seed bootstrap could not supervise every G3 head"
                )
        folds.append(G3Fold(tuple(sorted(heldout)), tuple(bags)))
    return ResolutionFirstG3(
        resolved, tuple(folds), fold_by_seed, dataset.sha256
    )


def save_checkpoint_g3(
    path: str | os.PathLike[str],
    model: ResolutionFirstG3,
    *,
    metadata: Mapping[str, object] | None = None,
) -> str:
    """Atomically create a canonical JSON checkpoint without overwriting."""

    destination = os.path.abspath(os.fspath(path))
    parent = os.path.dirname(destination)
    if (
        os.path.lexists(destination)
        or not os.path.isdir(parent)
        or os.path.realpath(parent) != parent
    ):
        raise RuntimeError("G3 checkpoint destination is occupied or indirect")
    payload = model.checkpoint(metadata=metadata)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
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
        raise RuntimeError("G3 checkpoint creation failed")
    with open(destination, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def load_checkpoint_g3(
    path: str | os.PathLike[str],
) -> tuple[ResolutionFirstG3, dict[str, object]]:
    source = os.path.abspath(os.fspath(path))
    stat = os.lstat(source)
    if (
        not os.path.isfile(source)
        or os.path.islink(source)
        or os.path.realpath(source) != source
        or stat.st_nlink != 1
    ):
        raise RuntimeError("G3 checkpoint source is indirect")
    with open(source, "rb") as stream:
        try:
            payload = json.loads(stream.read())
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("G3 checkpoint JSON is malformed") from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError("G3 checkpoint envelope is malformed")
    return ResolutionFirstG3.from_checkpoint(payload)
