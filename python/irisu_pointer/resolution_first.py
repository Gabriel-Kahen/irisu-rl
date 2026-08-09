"""Resolution-first learning and selective certification.

The exact branch evaluator remains the label oracle.  This module contains
only the teacher-free student, its identity-bound records, and the fail-closed
calibration logic used by the development-only R3H campaign.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


FEATURE_NAMES = (
    "gauge_fraction",
    "half_band",
    "level_scaled",
    "score_log",
    "qualifying_clears_log",
    "body_count_scaled",
    "visible_rot_debt_scaled",
    "spawn_interval_scaled",
    "source_x",
    "source_y",
    "source_vx",
    "source_vy",
    "source_size",
    "source_rot_timer",
    "source_remaining_lifetime",
    "source_chain",
    "destination_x",
    "destination_y",
    "destination_vx",
    "destination_vy",
    "destination_size",
    "destination_rot_timer",
    "destination_remaining_lifetime",
    "destination_chain",
    "pair_distance_sizes",
    "impact_x_sizes",
    "impact_y_sizes",
    "shot_weak",
    "category_rotten_hazard",
    "category_viable_anchor",
    "category_fresh_match",
    "geometry_analytic_strong",
    "geometry_close_strong",
    "geometry_wide_strong",
    "geometry_deep_strong",
    "geometry_analytic_weak",
    "incumbent_pair",
)
_INPUT_WIDTH = 3 * len(FEATURE_NAMES)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("exact numeric target is nonfinite")
    return result


def _candidate_identity(candidate: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in candidate.items()
        if key not in {"ordinal", "pair_ordinal", "geometry_ordinal"}
    }
    return _canonical_sha256(stable)


def _action_identity(candidate: Mapping[str, Any]) -> str:
    action = candidate.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("candidate action manifest is malformed")
    return _canonical_sha256(action)


def _candidate_signature(candidate: Mapping[str, Any]) -> str:
    pair = candidate.get("pair", {})
    geometry = candidate.get("geometry", {})
    if not isinstance(pair, Mapping) or not isinstance(geometry, Mapping):
        raise ValueError("candidate pair/geometry manifest is malformed")
    category, name = pair.get("category"), geometry.get("name")
    if not isinstance(category, str) or not isinstance(name, str):
        raise ValueError("candidate signature is absent")
    return f"{category}|{name}"


@dataclass(frozen=True, slots=True)
class BranchRecord:
    """One exact candidate label, retaining censored rows explicitly."""

    seed: int
    query_id: str
    ordinal: int
    candidate_id: str
    action_id: str
    signature: str
    features: tuple[float, ...]
    candidate_resolved: bool
    finite_pair: bool
    exact_unsafe: bool
    severe_unsafe: bool
    b2: float | None
    delta_b2: float | None
    score_advantage: float
    source_sha256: str

    @property
    def query_key(self) -> tuple[int, str]:
        return self.seed, self.query_id

    @property
    def safe_score_row(self) -> bool:
        return self.finite_pair and not self.exact_unsafe

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["features"] = list(self.features)
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def branch_records(
    queries: Iterable[Mapping[str, Any]],
) -> tuple[BranchRecord, ...]:
    """Validate exact query manifests and produce paired-support records."""

    staged: list[dict[str, Any]] = []
    for supplied in queries:
        query = supplied.get("exact_query", supplied)
        if not isinstance(query, Mapping):
            raise TypeError("exact query is malformed")
        outcomes = query.get("outcomes")
        if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
            raise TypeError("exact query outcomes are malformed")
        if not outcomes:
            raise ValueError("exact query has no candidates")
        query_seed = query.get("seed")
        query_id = query.get("query_id")
        if isinstance(query_seed, bool) or not isinstance(query_seed, int):
            raise ValueError("exact query seed is absent")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("exact query identity is absent")
        query_rows: list[dict[str, Any]] = []
        for outcome in outcomes:
            if not isinstance(outcome, Mapping):
                raise TypeError("exact branch outcome is malformed")
            candidate = outcome.get("candidate")
            targets = outcome.get("targets")
            result = outcome.get("outcome")
            ledger = outcome.get("ledger")
            if not all(
                isinstance(value, Mapping)
                for value in (candidate, targets, result, ledger)
            ):
                raise TypeError("exact branch sections are malformed")
            names = tuple(outcome.get("feature_names", ()))
            features = tuple(float(value) for value in outcome["feature_vector"])
            if names != FEATURE_NAMES or len(features) != len(FEATURE_NAMES):
                raise ValueError("exact feature schema mismatch")
            if not all(math.isfinite(value) for value in features):
                raise ValueError("exact feature vector is nonfinite")
            seed = outcome.get("seed", query_seed)
            identity = outcome.get("query_id", query_id)
            if seed != query_seed or identity != query_id:
                raise ValueError("branch/query identity mismatch")
            ordinal = int(candidate.get("ordinal", -1))
            b2 = _finite(targets.get("b2_margin"))
            delta = _finite(targets.get("delta_b2"))
            unresolved = tuple(ledger.get("unresolved", ()))
            rebind_failed = bool(ledger.get("continuation_rebind_failed", False))
            resolved = bool(result.get("renewals_resolved", False))
            candidate_resolved = bool(
                resolved and b2 is not None and not unresolved and not rebind_failed
            )
            raw = json.loads(
                json.dumps(
                    outcome, sort_keys=True, separators=(",", ":"), allow_nan=False
                )
            )
            query_rows.append(
                {
                    "seed": seed,
                    "query_id": query_id,
                    "ordinal": ordinal,
                    "candidate_id": _candidate_identity(candidate),
                    "action_id": _action_identity(candidate),
                    "signature": _candidate_signature(candidate),
                    "features": features,
                    "candidate_resolved": candidate_resolved,
                    "exact_unsafe": bool(
                        targets.get("exact_unsafe", result.get("exact_unsafe", False))
                    ),
                    "severe_unsafe": bool(
                        targets.get(
                            "severe_unsafe", result.get("severe_unsafe", False)
                        )
                    ),
                    "b2": b2,
                    "delta_b2": delta,
                    "score_advantage": float(targets.get("score_advantage", 0.0)),
                    "source_sha256": _canonical_sha256(raw),
                }
            )
        ordinals = [int(row["ordinal"]) for row in query_rows]
        if ordinals != list(range(len(query_rows))) or ordinals[0] != 0:
            raise ValueError("candidate ordinals must be incumbent-first and contiguous")
        if len({row["candidate_id"] for row in query_rows}) != len(query_rows):
            raise ValueError("exact query contains duplicate stable candidates")
        incumbent_resolved = bool(query_rows[0]["candidate_resolved"])
        for row in query_rows:
            finite_pair = bool(
                incumbent_resolved
                and row["candidate_resolved"]
                and row["delta_b2"] is not None
            )
            if row["ordinal"] == 0 and finite_pair and row["delta_b2"] != 0.0:
                raise ValueError("incumbent exact delta must be zero")
            row["finite_pair"] = finite_pair
            staged.append(row)
    records = tuple(BranchRecord(**row) for row in staged)
    if not records:
        raise ValueError("no exact branch records")
    return records


@dataclass(frozen=True, slots=True)
class ResolutionDataset:
    records: tuple[BranchRecord, ...]
    incumbent_features: tuple[tuple[float, ...], ...] = field(init=False)

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("resolution dataset must not be empty")
        groups: dict[tuple[int, str], list[BranchRecord]] = defaultdict(list)
        for record in self.records:
            groups[record.query_key].append(record)
        incumbents: dict[tuple[int, str], tuple[float, ...]] = {}
        for key, rows in groups.items():
            anchors = [row for row in rows if row.ordinal == 0]
            if len(anchors) != 1:
                raise ValueError(f"query {key!r} lacks one incumbent")
            incumbents[key] = anchors[0].features
        object.__setattr__(
            self,
            "incumbent_features",
            tuple(incumbents[row.query_key] for row in self.records),
        )

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted({record.seed for record in self.records}))

    def paired_features(self) -> Tensor:
        values = []
        for record, incumbent in zip(
            self.records, self.incumbent_features, strict=True
        ):
            values.append(
                (*record.features, *incumbent, *np.subtract(record.features, incumbent))
            )
        return torch.tensor(values, dtype=torch.float32)

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3h-resolution-dataset-v1",
            "feature_names": list(FEATURE_NAMES),
            "records": [record.sha256 for record in self.records],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class SupportEnvelope:
    feature_median: tuple[float, ...]
    feature_scale: tuple[float, ...]
    radii: tuple[tuple[str, float], ...]
    seed_counts: tuple[tuple[str, int], ...]
    quantile: float = 0.975
    minimum_signature_seeds: int = 8

    def __post_init__(self) -> None:
        if (
            len(self.feature_median) != len(FEATURE_NAMES)
            or len(self.feature_scale) != len(FEATURE_NAMES)
            or any(value <= 0 or not math.isfinite(value) for value in self.feature_scale)
        ):
            raise ValueError("support envelope feature schema is invalid")
        if not 0.5 < self.quantile < 1.0:
            raise ValueError("support envelope quantile is invalid")

    @property
    def radius_by_signature(self) -> dict[str, float]:
        return dict(self.radii)

    def distance(self, record: BranchRecord) -> float:
        delta = (
            np.asarray(record.features) - np.asarray(self.feature_median)
        ) / np.asarray(self.feature_scale)
        return float(np.sqrt(np.mean(np.square(delta))))

    def contains(self, record: BranchRecord) -> bool:
        radius = self.radius_by_signature.get(record.signature)
        return radius is not None and self.distance(record) <= radius

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        for key in ("feature_median", "feature_scale", "radii", "seed_counts"):
            value[key] = [list(item) if isinstance(item, tuple) else item for item in value[key]]
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def fit_support_envelope(
    records: Sequence[BranchRecord],
    *,
    quantile: float = 0.975,
    minimum_signature_seeds: int = 8,
) -> SupportEnvelope:
    if not records:
        raise ValueError("support envelope requires training records")
    x = np.asarray([record.features for record in records], dtype=np.float64)
    median = np.median(x, axis=0)
    mad = 1.4826 * np.median(np.abs(x - median), axis=0)
    fallback = np.std(x, axis=0)
    scale = np.where(mad > 1e-4, mad, np.maximum(fallback, 1e-3))
    distance = np.sqrt(np.mean(np.square((x - median) / scale), axis=1))
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record.signature].append(index)
    radii, seed_counts = [], []
    for signature in sorted(grouped):
        indices = grouped[signature]
        seed_count = len({records[index].seed for index in indices})
        seed_counts.append((signature, seed_count))
        if seed_count >= minimum_signature_seeds:
            radius = float(
                np.quantile(distance[indices], quantile, method="higher")
            )
            radii.append((signature, max(radius, 1e-6)))
    if not radii:
        raise ValueError("no candidate signature has enough training seed support")
    return SupportEnvelope(
        tuple(float(value) for value in median),
        tuple(float(value) for value in scale),
        tuple(radii),
        tuple(seed_counts),
        quantile,
        minimum_signature_seeds,
    )


@dataclass(frozen=True, slots=True)
class ResolutionFirstConfig:
    members: int = 5
    hidden_width: int = 96
    training_steps: int = 600
    batch_size: int = 128
    learning_rate: float = 7.5e-4
    weight_decay: float = 1e-4
    confidence_z: float = 1.6448536269514722
    gradient_clip: float = 5.0
    score_transform_scale: float = 8.0
    conformal_alpha: float = 0.05
    support_thresholds: tuple[float, ...] = (
        -0.75,
        -0.50,
        -0.25,
        0.00,
        0.25,
        0.50,
        0.75,
    )
    minimum_support_coverage: float = 0.05
    training_seeds: tuple[int, ...] = (
        2026073001,
        2026073002,
        2026073003,
        2026073004,
        2026073005,
    )

    def __post_init__(self) -> None:
        if min(
            self.members,
            self.hidden_width,
            self.training_steps,
            self.batch_size,
        ) < 1:
            raise ValueError("learner sizes must be positive")
        if len(self.training_seeds) != self.members:
            raise ValueError("one fixed seed is required per ensemble member")
        if not 0 < self.conformal_alpha < 1:
            raise ValueError("conformal alpha is invalid")

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["support_thresholds"] = list(self.support_thresholds)
        value["training_seeds"] = list(self.training_seeds)
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


class _BinaryHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(_INPUT_WIDTH, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value).squeeze(-1)


class _QuantileHead(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(_INPUT_WIDTH, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, 2),
        )

    def forward(self, value: Tensor) -> tuple[Tensor, Tensor]:
        raw = self.network(value)
        q10 = raw[:, 0]
        return q10, q10 + F.softplus(raw[:, 1])


class _Member(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.resolution = _BinaryHead(width)
        self.unsafe = _BinaryHead(width)
        self.delta = _QuantileHead(width)
        self.score = _QuantileHead(width)


class ResolutionFirstEnsemble(nn.Module):
    def __init__(
        self,
        config: ResolutionFirstConfig | None = None,
        *,
        support_envelope: SupportEnvelope,
    ) -> None:
        super().__init__()
        self.config = ResolutionFirstConfig() if config is None else config
        self.support_envelope = support_envelope
        self.members = nn.ModuleList(
            _Member(self.config.hidden_width) for _ in range(self.config.members)
        )
        self.register_buffer("feature_mean", torch.zeros(_INPUT_WIDTH))
        self.register_buffer("feature_scale", torch.ones(_INPUT_WIDTH))
        self.register_buffer("delta_mean", torch.tensor(0.0))
        self.register_buffer("delta_scale", torch.tensor(1.0))

    def normalized(self, value: Tensor) -> Tensor:
        return (value - self.feature_mean) / self.feature_scale

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3h-resolution-first-ensemble-v1",
            "feature_names": list(FEATURE_NAMES),
            "input": "candidate-incumbent-difference",
            "config": self.config.manifest(),
            "support_envelope_sha256": self.support_envelope.sha256,
        }


def save_checkpoint(
    path: str | os.PathLike[str],
    model: ResolutionFirstEnsemble,
    *,
    metadata: Mapping[str, object] | None = None,
) -> str:
    destination = os.path.abspath(os.fspath(path))
    if os.path.exists(destination):
        raise FileExistsError(f"refusing to replace checkpoint {destination}")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    payload = {
        "schema": "irisu-r3h-resolution-first-checkpoint-v1",
        "manifest": model.manifest(),
        "support_envelope": model.support_envelope.manifest(),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    temporary = f"{destination}.tmp-{os.getpid()}-{time.time_ns()}"
    try:
        torch.save(payload, temporary)
        os.link(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    digest = hashlib.sha256()
    with open(destination, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(
    path: str | os.PathLike[str],
) -> tuple[ResolutionFirstEnsemble, dict[str, object]]:
    payload = torch.load(
        os.fspath(path), map_location="cpu", weights_only=True
    )
    if payload.get("schema") != "irisu-r3h-resolution-first-checkpoint-v1":
        raise RuntimeError("resolution-first checkpoint schema mismatch")
    manifest = payload.get("manifest")
    envelope_raw = payload.get("support_envelope")
    if not isinstance(manifest, Mapping) or not isinstance(envelope_raw, Mapping):
        raise RuntimeError("resolution-first checkpoint manifest is malformed")
    if (
        manifest.get("schema") != "irisu-r3h-resolution-first-ensemble-v1"
        or tuple(manifest.get("feature_names", ())) != FEATURE_NAMES
        or manifest.get("input") != "candidate-incumbent-difference"
    ):
        raise RuntimeError("resolution-first checkpoint feature identity mismatch")
    config_raw = dict(manifest["config"])
    config_raw["support_thresholds"] = tuple(config_raw["support_thresholds"])
    config_raw["training_seeds"] = tuple(config_raw["training_seeds"])
    envelope_value = dict(envelope_raw)
    for key in ("feature_median", "feature_scale"):
        envelope_value[key] = tuple(envelope_value[key])
    for key in ("radii", "seed_counts"):
        envelope_value[key] = tuple(
            (str(row[0]), row[1]) for row in envelope_value[key]
        )
    envelope = SupportEnvelope(**envelope_value)
    model = ResolutionFirstEnsemble(
        ResolutionFirstConfig(**config_raw),
        support_envelope=envelope,
    )
    if manifest.get("support_envelope_sha256") != envelope.sha256:
        raise RuntimeError("resolution-first support identity mismatch")
    model.load_state_dict(payload["state_dict"], strict=True)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise RuntimeError("resolution-first checkpoint metadata is malformed")
    return model.eval(), dict(metadata)


def _pinball(prediction: Tensor, target: Tensor, quantile: float) -> Tensor:
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def _score_target(value: Tensor, scale: float) -> Tensor:
    return torch.sign(value) * torch.log1p(value.abs()) / scale


def _balanced_indices(
    records: Sequence[BranchRecord],
    pool: Sequence[int],
    *,
    count: int,
    generator: torch.Generator,
) -> Tensor:
    strata: tuple[list[int], ...] = (
        [i for i in pool if not records[i].finite_pair],
        [i for i in pool if records[i].finite_pair and records[i].exact_unsafe],
        [
            i
            for i in pool
            if records[i].finite_pair
            and not records[i].exact_unsafe
            and float(records[i].delta_b2 or 0.0) < 0
        ],
        [
            i
            for i in pool
            if records[i].finite_pair
            and not records[i].exact_unsafe
            and float(records[i].delta_b2 or 0.0) >= 0
        ],
    )
    active = [values for values in strata if values]
    if not active:
        raise ValueError("balanced sampler has no rows")
    selected: list[int] = []
    base, remainder = divmod(count, len(active))
    for position, values in enumerate(active):
        n = base + int(position < remainder)
        draws = torch.randint(len(values), (n,), generator=generator).tolist()
        selected.extend(values[index] for index in draws)
    order = torch.randperm(len(selected), generator=generator).tolist()
    return torch.tensor([selected[index] for index in order], dtype=torch.long)


def _pairwise_rank_loss(
    prediction: Tensor,
    target: Tensor,
    indices: Tensor,
    records: Sequence[BranchRecord],
) -> Tensor:
    by_query: dict[tuple[int, str], list[int]] = defaultdict(list)
    for local, global_index in enumerate(indices.tolist()):
        if records[global_index].safe_score_row:
            by_query[records[global_index].query_key].append(local)
    losses: list[Tensor] = []
    for local_indices in by_query.values():
        for left in range(len(local_indices)):
            for right in range(left + 1, len(local_indices)):
                i, j = local_indices[left], local_indices[right]
                sign = torch.sign(target[i] - target[j])
                if bool(sign):
                    losses.append(F.softplus(-sign * (prediction[i] - prediction[j])))
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


def train_resolution_first(
    dataset: ResolutionDataset,
    *,
    config: ResolutionFirstConfig | None = None,
    support_envelope: SupportEnvelope | None = None,
) -> ResolutionFirstEnsemble:
    resolved = ResolutionFirstConfig() if config is None else config
    records = dataset.records
    if len(dataset.seeds) < 2:
        raise ValueError("whole-seed bootstrap requires at least two seeds")
    if not any(record.finite_pair for record in records):
        raise ValueError("delta training requires finite paired rows")
    envelope = (
        fit_support_envelope(records)
        if support_envelope is None
        else support_envelope
    )
    model = ResolutionFirstEnsemble(resolved, support_envelope=envelope)
    x = dataset.paired_features()
    finite = torch.tensor([record.finite_pair for record in records])
    unsafe = torch.tensor([float(record.exact_unsafe) for record in records])
    delta = torch.tensor(
        [0.0 if record.delta_b2 is None else record.delta_b2 for record in records],
        dtype=torch.float32,
    )
    score = torch.tensor(
        [record.score_advantage for record in records], dtype=torch.float32
    )
    safe_score = torch.tensor([record.safe_score_row for record in records])
    with torch.no_grad():
        model.feature_mean.copy_(x.mean(0))
        model.feature_scale.copy_(x.std(0, unbiased=False).clamp_min(1e-5))
        model.delta_mean.copy_(delta[finite].mean())
        model.delta_scale.copy_(delta[finite].std(unbiased=False).clamp_min(1.0))
    z = model.normalized(x)
    delta_target = (delta - model.delta_mean) / model.delta_scale
    transformed_score = _score_target(score, resolved.score_transform_scale)
    groups: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record.seed].append(index)
    seed_groups = list(groups.values())
    for member, seed in zip(
        model.members, resolved.training_seeds, strict=True
    ):
        torch.manual_seed(seed)
        for module in member.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        generator = torch.Generator().manual_seed(seed)
        bootstrap = torch.randint(
            len(seed_groups), (len(seed_groups),), generator=generator
        ).tolist()
        pool = [
            index for group_index in bootstrap for index in seed_groups[group_index]
        ]
        optimizer = torch.optim.AdamW(
            member.parameters(),
            lr=resolved.learning_rate,
            weight_decay=resolved.weight_decay,
        )
        for _step in range(resolved.training_steps):
            chosen = _balanced_indices(
                records,
                pool,
                count=resolved.batch_size,
                generator=generator,
            )
            value = z[chosen]
            resolution_logit = member.resolution(value)
            unsafe_logit = member.unsafe(value)
            delta_q10, delta_q50 = member.delta(value)
            score_q10, score_q50 = member.score(value)
            resolution_loss = F.binary_cross_entropy_with_logits(
                resolution_logit, finite[chosen].float()
            )
            unsafe_loss = F.binary_cross_entropy_with_logits(
                unsafe_logit, unsafe[chosen]
            )
            finite_mask = finite[chosen]
            delta_loss = (
                _pinball(
                    delta_q10[finite_mask],
                    delta_target[chosen][finite_mask],
                    0.10,
                )
                + _pinball(
                    delta_q50[finite_mask],
                    delta_target[chosen][finite_mask],
                    0.50,
                )
                if bool(finite_mask.any())
                else delta_q10.sum() * 0.0
            )
            score_mask = safe_score[chosen]
            if bool(score_mask.any()):
                score_loss = _pinball(
                    score_q10[score_mask],
                    transformed_score[chosen][score_mask],
                    0.10,
                ) + F.smooth_l1_loss(
                    score_q50[score_mask],
                    transformed_score[chosen][score_mask],
                )
                score_loss = score_loss + 0.5 * _pairwise_rank_loss(
                    score_q50,
                    transformed_score[chosen],
                    chosen,
                    records,
                )
            else:
                score_loss = score_q10.sum() * 0.0
            loss = resolution_loss + unsafe_loss + delta_loss + score_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                member.parameters(), resolved.gradient_clip
            )
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def predict_records(
    model: ResolutionFirstEnsemble,
    dataset: ResolutionDataset,
) -> tuple[dict[str, object], ...]:
    model.eval()
    z = model.normalized(dataset.paired_features())
    resolution, unsafe, delta, score = [], [], [], []
    for member in model.members:
        resolution.append(member.resolution(z).sigmoid())
        unsafe.append(member.unsafe(z).sigmoid())
        delta_q10, _delta_q50 = member.delta(z)
        score_q10, _score_q50 = member.score(z)
        delta.append(delta_q10 * model.delta_scale + model.delta_mean)
        score.append(score_q10)
    r = torch.stack(resolution)
    u = torch.stack(unsafe)
    d = torch.stack(delta)
    s = torch.stack(score)
    output = []
    for index, record in enumerate(dataset.records):
        resolution_mean = float(r[:, index].mean())
        resolution_std = float(r[:, index].std(unbiased=False))
        unsafe_mean = float(u[:, index].mean())
        unsafe_std = float(u[:, index].std(unbiased=False))
        delta_mean = float(d[:, index].mean())
        delta_std = float(d[:, index].std(unbiased=False))
        score_mean = float(s[:, index].mean())
        score_std = float(s[:, index].std(unbiased=False))
        resolution_lcb = resolution_mean - model.config.confidence_z * resolution_std
        unsafe_ucb = unsafe_mean + model.config.confidence_z * unsafe_std
        output.append(
            {
                "seed": record.seed,
                "query_id": record.query_id,
                "ordinal": record.ordinal,
                "candidate_id": record.candidate_id,
                "action_id": record.action_id,
                "signature": record.signature,
                "envelope_supported": model.support_envelope.contains(record),
                "support_distance": model.support_envelope.distance(record),
                "resolution_mean": resolution_mean,
                "resolution_std": resolution_std,
                "resolution_lcb": resolution_lcb,
                "unsafe_mean": unsafe_mean,
                "unsafe_std": unsafe_std,
                "unsafe_ucb": unsafe_ucb,
                "support_score": resolution_lcb - unsafe_ucb,
                "delta_q10_mean": delta_mean,
                "delta_q10_std": delta_std,
                "delta_lcb": delta_mean
                - model.config.confidence_z * delta_std,
                "score_q10_mean": score_mean,
                "score_q10_std": score_std,
                "score_lcb": score_mean
                - model.config.confidence_z * score_std,
                "exact_finite_pair": record.finite_pair,
                "exact_unsafe": record.exact_unsafe,
                "exact_delta_b2": record.delta_b2,
                "exact_score_advantage": record.score_advantage,
            }
        )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class SupportCalibration:
    threshold: float
    selected_candidates: int
    candidate_count: int
    coverage: float
    selected_bad_candidates: int
    selected_bad_seeds: int
    grid: tuple[Mapping[str, object], ...]

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["grid"] = [dict(row) for row in self.grid]
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def fit_support_calibration(
    predictions: Sequence[Mapping[str, object]],
    *,
    thresholds: Sequence[float] = ResolutionFirstConfig().support_thresholds,
    minimum_coverage: float = 0.05,
) -> SupportCalibration:
    candidates = [row for row in predictions if int(row["ordinal"]) != 0]
    if not candidates:
        raise ValueError("support calibration has no non-incumbent candidates")
    grid: list[dict[str, object]] = []
    winner: dict[str, object] | None = None
    for threshold in thresholds:
        selected = [
            row
            for row in candidates
            if bool(row["envelope_supported"])
            and float(row["support_score"]) > float(threshold)
        ]
        bad = [
            row
            for row in selected
            if not bool(row["exact_finite_pair"]) or bool(row["exact_unsafe"])
        ]
        coverage = len(selected) / len(candidates)
        point: dict[str, object] = {
            "threshold": float(threshold),
            "selected_candidates": len(selected),
            "candidate_count": len(candidates),
            "coverage": coverage,
            "selected_bad_candidates": len(bad),
            "selected_bad_seeds": len({int(row["seed"]) for row in bad}),
            "passed": not bad and coverage >= minimum_coverage,
        }
        grid.append(point)
        if winner is None and point["passed"] is True:
            winner = point
    if winner is None:
        raise RuntimeError("no preregistered support threshold passed")
    return SupportCalibration(
        float(winner["threshold"]),
        int(winner["selected_candidates"]),
        int(winner["candidate_count"]),
        float(winner["coverage"]),
        int(winner["selected_bad_candidates"]),
        int(winner["selected_bad_seeds"]),
        tuple(grid),
    )


def _support_selected(
    row: Mapping[str, object], calibration: SupportCalibration
) -> bool:
    return bool(row["envelope_supported"]) and float(
        row["support_score"]
    ) > calibration.threshold


@dataclass(frozen=True, slots=True)
class SelectiveCalibration:
    alpha: float
    q: float
    episode_count: int
    rank: int
    selected_candidates: int
    selected_bad_candidates: int
    nonempty_episodes: int
    episode_residuals: tuple[tuple[int, float], ...]
    support_calibration_sha256: str

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        if not math.isfinite(self.q):
            value["q"] = "Infinity" if self.q > 0 else "-Infinity"
        value["episode_residuals"] = [
            [
                seed,
                (
                    residual
                    if math.isfinite(residual)
                    else ("Infinity" if residual > 0 else "-Infinity")
                ),
            ]
            for seed, residual in self.episode_residuals
        ]
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def fit_selective_calibration(
    predictions: Sequence[Mapping[str, object]],
    support: SupportCalibration,
    *,
    alpha: float = 0.05,
    required_episodes: int | None = None,
) -> SelectiveCalibration:
    if not 0 < alpha < 1:
        raise ValueError("selective conformal alpha is invalid")
    seeds = {int(row["seed"]) for row in predictions}
    if required_episodes is not None and len(seeds) != required_episodes:
        raise ValueError("selective conformal episode count mismatch")
    residuals: dict[int, list[float]] = {seed: [] for seed in seeds}
    selected = bad = 0
    for row in predictions:
        if int(row["ordinal"]) == 0 or not _support_selected(row, support):
            continue
        selected += 1
        seed = int(row["seed"])
        exact = row.get("exact_delta_b2")
        if (
            not bool(row["exact_finite_pair"])
            or bool(row["exact_unsafe"])
            or exact is None
        ):
            bad += 1
            residuals[seed].append(math.inf)
        else:
            residuals[seed].append(
                float(row["delta_lcb"]) - float(exact)
            )
    maxima = tuple(
        sorted(
            (
                seed,
                max(values) if values else 0.0,
            )
            for seed, values in residuals.items()
        )
    )
    ordered = sorted(value for _seed, value in maxima)
    rank = math.ceil((len(ordered) + 1) * (1.0 - alpha))
    q = math.inf if rank > len(ordered) else ordered[rank - 1]
    return SelectiveCalibration(
        alpha,
        float(q),
        len(seeds),
        rank,
        selected,
        bad,
        sum(bool(values) for values in residuals.values()),
        maxima,
        support.sha256,
    )


@dataclass(frozen=True, slots=True)
class SelectedCandidate:
    seed: int
    query_id: str
    ordinal: int
    candidate_id: str
    override: bool
    exact_finite_pair: bool
    exact_unsafe: bool
    exact_delta_b2: float | None
    exact_score_advantage: float
    reasons: tuple[str, ...]


def select_candidates(
    predictions: Sequence[Mapping[str, object]],
    support: SupportCalibration,
    calibration: SelectiveCalibration,
) -> tuple[SelectedCandidate, ...]:
    if calibration.support_calibration_sha256 != support.sha256:
        raise ValueError("support/selective calibration identity mismatch")
    grouped: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in predictions:
        grouped[(int(row["seed"]), str(row["query_id"]))].append(row)
    output: list[SelectedCandidate] = []
    for (seed, query_id), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: int(row["ordinal"]))
        if int(rows[0]["ordinal"]) != 0:
            raise ValueError("selection query lacks incumbent")
        incumbent_action = str(rows[0]["action_id"])
        eligible = [
            row
            for row in rows[1:]
            if _support_selected(row, support)
            and str(row["action_id"]) != incumbent_action
            and math.isfinite(calibration.q)
            and float(row["delta_lcb"]) - calibration.q >= 0
            and float(row["score_lcb"]) > 0
        ]
        reasons: tuple[str, ...]
        if eligible:
            best_score = max(float(row["score_lcb"]) for row in eligible)
            highest = [
                row
                for row in eligible
                if float(row["score_lcb"]) == best_score
            ]
            if len(highest) == 1:
                selected, reasons = highest[0], ()
            else:
                selected, reasons = rows[0], ("conservative-tie",)
        else:
            selected, reasons = rows[0], ("no-certified-alternative",)
        output.append(
            SelectedCandidate(
                seed,
                query_id,
                int(selected["ordinal"]),
                str(selected["candidate_id"]),
                int(selected["ordinal"]) != 0,
                bool(selected["exact_finite_pair"]),
                bool(selected["exact_unsafe"]),
                (
                    None
                    if selected.get("exact_delta_b2") is None
                    else float(selected["exact_delta_b2"])
                ),
                float(selected["exact_score_advantage"]),
                reasons,
            )
        )
    return tuple(output)


def _clopper_pearson_upper(
    failures: int, episodes: int, *, alpha: float = 0.05
) -> float:
    if not 0 <= failures <= episodes or episodes < 1:
        raise ValueError("invalid binomial counts")
    if failures == episodes:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(80):
        probability = (low + high) / 2
        cdf = sum(
            math.comb(episodes, k)
            * probability**k
            * (1 - probability) ** (episodes - k)
            for k in range(failures + 1)
        )
        if cdf > alpha:
            low = probability
        else:
            high = probability
    return high


def _clustered_coverage_lcb(
    selections: Sequence[SelectedCandidate],
    *,
    replicates: int = 10_000,
    alpha: float = 0.05,
) -> float:
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for selection in selections:
        counts[selection.seed][0] += int(selection.override)
        counts[selection.seed][1] += 1
    if not counts:
        return 0.0
    values = np.asarray(list(counts.values()), dtype=np.int64)
    digest = hashlib.sha256(
        b"r3h-resolution-first-20260730|coverage-bootstrap"
    ).digest()
    rng = np.random.default_rng(int.from_bytes(digest, "big"))
    indices = rng.integers(0, len(values), size=(replicates, len(values)))
    totals = values[indices].sum(axis=1)
    ratios = np.divide(
        totals[:, 0],
        totals[:, 1],
        out=np.zeros(replicates),
        where=totals[:, 1] > 0,
    )
    return float(np.quantile(ratios, alpha, method="linear"))


def viability_report(
    selections: Sequence[SelectedCandidate],
    calibration: SelectiveCalibration,
    *,
    minimum_coverage: float = 0.05,
) -> dict[str, object]:
    if not selections:
        raise ValueError("viability screen has no decisions")
    overrides = [selection for selection in selections if selection.override]
    bad_seeds = {
        selection.seed
        for selection in overrides
        if not selection.exact_finite_pair or selection.exact_unsafe
    }
    score = [selection.exact_score_advantage for selection in overrides]
    delta = [
        float(selection.exact_delta_b2)
        for selection in overrides
        if selection.exact_delta_b2 is not None
    ]
    coverage = len(overrides) / len(selections)
    coverage_lcb = _clustered_coverage_lcb(selections)
    episode_count = len({selection.seed for selection in selections})
    gates = {
        "finite_conformal_q": math.isfinite(calibration.q),
        "raw_coverage": coverage >= minimum_coverage,
        "positive_clustered_coverage_lcb": coverage_lcb > 0,
        "zero_selected_bad_candidates": not any(
            not selection.exact_finite_pair or selection.exact_unsafe
            for selection in overrides
        ),
        "zero_false_safe_episodes": not bad_seeds,
        "positive_median_score_advantage": bool(score)
        and float(np.median(score)) > 0,
        "nonnegative_median_delta_b2": bool(delta)
        and float(np.median(delta)) >= 0,
    }
    return {
        "schema": "irisu-r3h-offline-viability-v1",
        "decisions": len(selections),
        "episodes": episode_count,
        "overrides": len(overrides),
        "coverage": coverage,
        "coverage_seed_clustered_lcb_95": coverage_lcb,
        "false_safe_episodes": len(bad_seeds),
        "false_safe_cp_upper_95": _clopper_pearson_upper(
            len(bad_seeds), episode_count
        ),
        "median_exact_score_advantage": (
            None if not score else float(np.median(score))
        ),
        "median_exact_delta_b2": (
            None if not delta else float(np.median(delta))
        ),
        "gates": gates,
        "passed": all(gates.values()),
    }


def pilot_report(records: Sequence[BranchRecord]) -> dict[str, object]:
    grouped: dict[int, list[BranchRecord]] = defaultdict(list)
    for record in records:
        grouped[record.seed].append(record)
    resolved_seeds = {
        seed
        for seed, rows in grouped.items()
        if any(row.ordinal and row.finite_pair for row in rows)
    }
    viable_seeds = {
        seed
        for seed, rows in grouped.items()
        if any(
            row.ordinal
            and row.finite_pair
            and not row.exact_unsafe
            and float(row.delta_b2 or 0) >= 0
            and row.score_advantage > 0
            for row in rows
        )
    }
    resolved_rows = sum(
        record.ordinal != 0 and record.finite_pair for record in records
    )
    gates = {
        "resolved_alternative_seeds": len(resolved_seeds) >= 18,
        "viable_alternative_seeds": len(viable_seeds) >= 12,
        "resolved_nonincumbent_rows": resolved_rows >= 96,
    }
    return {
        "schema": "irisu-r3h-structural-pilot-v1",
        "seeds": len(grouped),
        "resolved_alternative_seeds": len(resolved_seeds),
        "viable_alternative_seeds": len(viable_seeds),
        "resolved_nonincumbent_rows": resolved_rows,
        "gates": gates,
        "passed": all(gates.values()),
    }
