"""Generation-02 relational-board resolution, solvency, and score learner.

Exact branch outcomes remain the label oracle.  Numeric body and chain
identities are used only to bind invariant relational roles and never enter the
model.  Selection is deliberately staged:

``resolution/unsafe support -> absolute B2 certification -> score ranking``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from irisu_rl.encoding import EncodedBatch, TeacherStateEncoder
from .resolution_first import FEATURE_NAMES, BranchRecord, branch_records


_TEACHER_GLOBAL_WIDTH = 12
_TEACHER_BODY_WIDTH = 45
_MODEL_GLOBAL_WIDTH = _TEACHER_GLOBAL_WIDTH + 3
_ID_COLUMN = TeacherStateEncoder.schema.body_features.index("id_scaled")
_CHAIN_COLUMN = TeacherStateEncoder.schema.body_features.index("chain_id_scaled")
_X_COLUMN = TeacherStateEncoder.schema.body_features.index("effect_x_norm")
_Y_COLUMN = TeacherStateEncoder.schema.body_features.index("effect_y_norm")
_COLOR_SLICE = slice(
    TeacherStateEncoder.schema.body_features.index("color_0"),
    TeacherStateEncoder.schema.body_features.index("color_unknown") + 1,
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
_MODEL_BODY_WIDTH = _TEACHER_BODY_WIDTH + len(_RELATIONAL_NAMES)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _canonical_token(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _partition(tokens: Sequence[str]) -> tuple[int, ...]:
    labels: dict[str, int] = {}
    return tuple(labels.setdefault(token, len(labels)) for token in tokens)


def _encoded_body_ids(
    encoded: EncodedBatch,
    observation: Mapping[str, Any],
    *,
    row: int = 0,
) -> tuple[int | None, ...]:
    """Bind active teacher rows to public IDs before both ID channels are removed."""

    encoded.validate()
    if not 0 <= row < encoded.global_features.shape[0]:
        raise IndexError("encoded row is outside the batch")
    public_ids = {
        int(body["id"])
        for body in observation.get("bodies", ())
        if isinstance(body, Mapping) and "id" in body
    }
    output: list[int | None] = []
    bound: set[int] = set()
    for index in range(encoded.schema.capacity):
        if not bool(encoded.body_mask[row, index]):
            output.append(None)
            continue
        identifier = round(
            float(encoded.body_features[row, index, _ID_COLUMN]) * 2**32
        )
        if identifier not in public_ids:
            raise RuntimeError(
                "encoded body row does not bind to a current public body"
            )
        if identifier in bound:
            raise RuntimeError("encoded body rows collide on one public body ID")
        bound.add(identifier)
        output.append(identifier)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class BoardBranchRecord(BranchRecord):
    """One exact branch label bound to a complete public board."""

    global_features: tuple[float, ...]
    phase_features: tuple[float, float, float]
    body_features: tuple[tuple[float, ...], ...]
    body_chain_groups: tuple[int, ...]
    body_grouped_flags: tuple[bool, ...]
    body_color_groups: tuple[int, ...]
    source_index: int
    destination_index: int
    incumbent_source_index: int
    incumbent_destination_index: int
    observation_sha256: str

    @property
    def model_global_features(self) -> tuple[float, ...]:
        values = list(self.global_features)
        values[0] = 0.0
        return (*values, *self.phase_features)

    @property
    def safe_score_row_g2(self) -> bool:
        return self.candidate_resolved and not self.exact_unsafe

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        for name in (
            "features",
            "global_features",
            "phase_features",
            "body_chain_groups",
            "body_grouped_flags",
            "body_color_groups",
        ):
            value[name] = list(value[name])
        value["body_features"] = [list(row) for row in self.body_features]
        return value


def _public_observation(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    observation = entry.get("pre_query_public_observation")
    if not isinstance(observation, Mapping):
        raise TypeError("pre_query_public_observation is required")
    alias = entry.get("public_observation")
    if alias is not None and (
        not isinstance(alias, Mapping)
        or _canonical_sha256(alias) != _canonical_sha256(observation)
    ):
        raise ValueError("public observation aliases differ")
    digest = _canonical_sha256(observation)
    supplied = entry.get("pre_query_public_observation_sha256")
    if supplied is not None and supplied != digest:
        raise ValueError("public observation hash mismatch")
    return observation


def _pair_indices(
    candidate: Mapping[str, Any], body_index: Mapping[int, int]
) -> tuple[int, int]:
    pair = candidate.get("pair")
    if not isinstance(pair, Mapping):
        raise TypeError("candidate pair is malformed")
    source = _plain_int(pair.get("source_body_id"), "source_body_id")
    destination = _plain_int(pair.get("destination_body_id"), "destination_body_id")
    if source == destination:
        raise ValueError("candidate source and destination must differ")
    try:
        return body_index[source], body_index[destination]
    except KeyError as exc:
        raise ValueError("candidate body is absent from public observation") from exc


def _raw_body_metadata(
    observation: Mapping[str, Any], bound_ids: Sequence[int]
) -> tuple[tuple[int, ...], tuple[bool, ...], tuple[int, ...]]:
    raw = observation.get("bodies")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("public observation bodies are malformed")
    by_id: dict[int, Mapping[str, Any]] = {}
    for body in raw:
        if not isinstance(body, Mapping):
            raise TypeError("public body is malformed")
        identifier = _plain_int(body.get("id"), "body id")
        if identifier in by_id:
            raise ValueError("public body identities are not unique")
        by_id[identifier] = body
    try:
        grouped = tuple(
            by_id[identifier].get("chain_id") not in (0, None)
            for identifier in bound_ids
        )
        chains = [
            (
                _canonical_token(by_id[identifier].get("chain_id"))
                if is_grouped
                else f"ungrouped-body:{identifier}"
            )
            for identifier, is_grouped in zip(bound_ids, grouped, strict=True)
        ]
        colors = [_canonical_token(by_id[identifier].get("color")) for identifier in bound_ids]
    except KeyError as exc:
        raise ValueError("encoded body is absent from public observation") from exc
    return _partition(chains), grouped, _partition(colors)


def board_branch_records(
    entries: Iterable[Mapping[str, Any]],
    *,
    encoder: TeacherStateEncoder | None = None,
) -> tuple[BoardBranchRecord, ...]:
    """Bind exact records to identity-free, relationally recoverable boards."""

    resolved_encoder = TeacherStateEncoder() if encoder is None else encoder
    if resolved_encoder.schema.sha256 != TeacherStateEncoder.schema.sha256:
        raise ValueError("generation-02 requires the teacher-v1 schema")
    output: list[BoardBranchRecord] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TypeError("generation-02 query entry is malformed")
        query = entry.get("exact_query")
        if not isinstance(query, Mapping):
            raise TypeError("exact_query is required")
        observation = _public_observation(entry)
        query_index = _plain_int(entry.get("query_index"), "query_index")
        shot_index = _plain_int(entry.get("shot_index"), "shot_index")
        tick = _plain_int(entry.get("tick"), "tick")
        observation_tick = _plain_int(observation.get("tick"), "observation tick")
        if not 0 <= query_index <= 3 or not 1 <= shot_index <= 19:
            raise ValueError("phase indices are outside preregistered bounds")
        if tick != observation_tick:
            raise ValueError("query tick differs from public observation")
        difficulty = observation.get("difficulty", {})
        interval = float(
            difficulty.get("spawn_interval_ticks", 0)
            if isinstance(difficulty, Mapping)
            else 0
        )
        if not math.isfinite(interval) or interval <= 0:
            raise ValueError("spawn interval must be finite and positive")

        exact_records = branch_records([query])
        outcomes = query.get("outcomes")
        assert isinstance(outcomes, Sequence)
        encoded = resolved_encoder.encode([observation])
        active = np.flatnonzero(encoded.body_mask[0])
        identifiers = _encoded_body_ids(encoded, observation)
        bound_ids = [identifiers[index] for index in active]
        if any(identifier is None for identifier in bound_ids):
            raise RuntimeError("active teacher body failed identity binding")
        concrete_ids = tuple(int(identifier) for identifier in bound_ids if identifier is not None)
        body_index = {identifier: position for position, identifier in enumerate(concrete_ids)}
        if len(body_index) != len(active):
            raise RuntimeError("public body identities are not unique")
        bodies = np.array(encoded.body_features[0, active], copy=True)
        bodies[:, _ID_COLUMN] = 0.0
        bodies[:, _CHAIN_COLUMN] = 0.0
        body_rows = tuple(tuple(float(value) for value in row) for row in bodies)
        chain_groups, grouped_flags, color_groups = _raw_body_metadata(
            observation, concrete_ids
        )
        global_row = tuple(float(value) for value in encoded.global_features[0])
        incumbent_raw = outcomes[0]
        if not isinstance(incumbent_raw, Mapping) or not isinstance(
            incumbent_raw.get("candidate"), Mapping
        ):
            raise TypeError("incumbent candidate is malformed")
        incumbent_pair = _pair_indices(incumbent_raw["candidate"], body_index)
        observation_sha256 = _canonical_sha256(observation)
        phase = (
            query_index / 3.0,
            shot_index / 19.0,
            (tick % interval) / interval,
        )
        for base, raw in zip(exact_records, outcomes, strict=True):
            if not isinstance(raw, Mapping) or not isinstance(
                raw.get("candidate"), Mapping
            ):
                raise TypeError("exact candidate is malformed")
            source, destination = _pair_indices(raw["candidate"], body_index)
            values = asdict(base)
            output.append(
                BoardBranchRecord(
                    **values,
                    global_features=global_row,
                    phase_features=phase,
                    body_features=body_rows,
                    body_chain_groups=chain_groups,
                    body_grouped_flags=grouped_flags,
                    body_color_groups=color_groups,
                    source_index=source,
                    destination_index=destination,
                    incumbent_source_index=incumbent_pair[0],
                    incumbent_destination_index=incumbent_pair[1],
                    observation_sha256=observation_sha256,
                )
            )
    if not output:
        raise ValueError("no generation-02 records")
    return tuple(output)


def _relational_body_rows(record: BoardBranchRecord) -> tuple[tuple[float, ...], ...]:
    bodies = np.asarray(record.body_features, dtype=np.float64)
    chains = record.body_chain_groups
    grouped = record.body_grouped_flags
    colors = record.body_color_groups
    if (
        len(chains) != len(bodies)
        or len(grouped) != len(bodies)
        or len(colors) != len(bodies)
    ):
        raise ValueError("body relationship metadata width mismatch")
    chain_counts = Counter(chains)
    source, destination = record.source_index, record.destination_index
    incumbent_source = record.incumbent_source_index
    incumbent_destination = record.incumbent_destination_index
    endpoints = (source, destination, incumbent_source, incumbent_destination)
    source_xy = bodies[source, [_X_COLUMN, _Y_COLUMN]]
    destination_xy = bodies[destination, [_X_COLUMN, _Y_COLUMN]]
    denominator = max(math.log1p(len(bodies)), 1.0)
    output: list[tuple[float, ...]] = []
    for index, body in enumerate(bodies):
        size = chain_counts[chains[index]] if grouped[index] else 0
        delta_source = body[[_X_COLUMN, _Y_COLUMN]] - source_xy
        delta_destination = body[[_X_COLUMN, _Y_COLUMN]] - destination_xy
        relational = (
            *(float(index == endpoint) for endpoint in endpoints),
            float(grouped[index]),
            math.log1p(size) / denominator,
            *(
                float(
                    grouped[index]
                    and grouped[endpoint]
                    and chains[index] == chains[endpoint]
                )
                for endpoint in endpoints
            ),
            float(delta_source[0]),
            float(delta_source[1]),
            float(np.linalg.norm(delta_source)),
            float(delta_destination[0]),
            float(delta_destination[1]),
            float(np.linalg.norm(delta_destination)),
            *(float(colors[index] == colors[endpoint]) for endpoint in endpoints),
        )
        output.append((*tuple(float(value) for value in body), *relational))
    return tuple(output)


@dataclass(frozen=True, slots=True)
class BoardResolutionDataset:
    records: tuple[BoardBranchRecord, ...]
    incumbent_features: tuple[tuple[float, ...], ...] = field(init=False)

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("board dataset must not be empty")
        grouped: dict[tuple[int, str], list[BoardBranchRecord]] = defaultdict(list)
        for record in self.records:
            grouped[record.query_key].append(record)
        incumbents: dict[tuple[int, str], tuple[float, ...]] = {}
        for key, rows in grouped.items():
            anchors = [row for row in rows if row.ordinal == 0]
            if len(anchors) != 1:
                raise ValueError(f"query {key!r} lacks one incumbent")
            if len({row.observation_sha256 for row in rows}) != 1:
                raise ValueError("one query is bound to multiple observations")
            incumbents[key] = anchors[0].features
        object.__setattr__(
            self,
            "incumbent_features",
            tuple(incumbents[row.query_key] for row in self.records),
        )

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(sorted({record.seed for record in self.records}))

    def tensors(
        self,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        count = len(self.records)
        width = max(len(record.body_features) for record in self.records)
        bodies = torch.zeros((count, width, _MODEL_BODY_WIDTH))
        mask = torch.zeros((count, width), dtype=torch.bool)
        for index, record in enumerate(self.records):
            rows = _relational_body_rows(record)
            bodies[index, : len(rows)] = torch.tensor(rows, dtype=torch.float32)
            mask[index, : len(rows)] = True
        return (
            torch.tensor(
                [record.model_global_features for record in self.records],
                dtype=torch.float32,
            ),
            bodies,
            mask,
            torch.tensor(
                [record.features for record in self.records], dtype=torch.float32
            ),
            torch.tensor(self.incumbent_features, dtype=torch.float32),
            torch.tensor([record.source_index for record in self.records]),
            torch.tensor([record.destination_index for record in self.records]),
            torch.tensor(
                [
                    (
                        record.incumbent_source_index,
                        record.incumbent_destination_index,
                    )
                    for record in self.records
                ]
            ),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3h-resolution-board-dataset-g2-v2",
            "teacher_schema_sha256": TeacherStateEncoder.schema.sha256,
            "candidate_feature_names": list(FEATURE_NAMES),
            "relational_feature_names": list(_RELATIONAL_NAMES),
            "records": [record.sha256 for record in self.records],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class RobustSignatureProfile:
    signature: str
    median: tuple[float, ...]
    scale: tuple[float, ...]
    rms_radius: float
    topk_radius: float
    max_radius: float
    seed_count: int

    def __post_init__(self) -> None:
        if not self.signature or not self.median or len(self.median) != len(self.scale):
            raise ValueError("robust signature profile width is invalid")
        if self.seed_count < 8:
            raise ValueError("robust signature profile requires at least eight seeds")
        values = (*self.scale, self.rms_radius, self.topk_radius, self.max_radius)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("robust signature profile scale/radius is invalid")

    def manifest(self) -> dict[str, object]:
        return {
            "signature": self.signature,
            "median": list(self.median),
            "scale": list(self.scale),
            "rms_radius": self.rms_radius,
            "topk_radius": self.topk_radius,
            "max_radius": self.max_radius,
            "seed_count": self.seed_count,
        }


@dataclass(frozen=True, slots=True)
class RepresentationEnvelope:
    profiles: tuple[RobustSignatureProfile, ...]
    quantile: float = 0.975
    minimum_signature_seeds: int = 8
    top_k: int = 8

    def __post_init__(self) -> None:
        if not 0.5 < self.quantile < 1:
            raise ValueError("learned support quantile is invalid")
        if self.minimum_signature_seeds < 8 or self.top_k < 1:
            raise ValueError("learned support envelope constraints are invalid")
        signatures = [profile.signature for profile in self.profiles]
        if not signatures or len(signatures) != len(set(signatures)):
            raise ValueError("learned support profiles are empty or duplicated")
        if any(
            profile.seed_count < self.minimum_signature_seeds
            for profile in self.profiles
        ):
            raise ValueError("learned support profile seed count is insufficient")

    @property
    def profile_by_signature(self) -> dict[str, RobustSignatureProfile]:
        return {profile.signature: profile for profile in self.profiles}

    def distances(
        self, signature: str, representation: Sequence[float]
    ) -> tuple[float, float, float] | None:
        profile = self.profile_by_signature.get(signature)
        if profile is None:
            return None
        if len(representation) != len(profile.median):
            raise ValueError("learned support representation width mismatch")
        standardized = np.abs(
            (np.asarray(representation) - np.asarray(profile.median))
            / np.asarray(profile.scale)
        )
        rms = float(np.sqrt(np.mean(np.square(standardized))))
        count = min(self.top_k, len(standardized))
        top = np.partition(standardized, len(standardized) - count)[-count:]
        topk = float(np.sqrt(np.mean(np.square(top))))
        return rms, topk, float(np.max(standardized))

    def contains(self, signature: str, representation: Sequence[float]) -> bool:
        profile = self.profile_by_signature.get(signature)
        distances = self.distances(signature, representation)
        return bool(
            profile is not None
            and distances is not None
            and distances[0] <= profile.rms_radius
            and distances[1] <= profile.topk_radius
            and distances[2] <= profile.max_radius
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3h-robust-representation-envelope-g2-v2",
            "profiles": [profile.manifest() for profile in self.profiles],
            "quantile": self.quantile,
            "minimum_signature_seeds": self.minimum_signature_seeds,
            "top_k": self.top_k,
            "seed_weighting": "equal-per-seed-max-distance",
            "gate": "rms-and-topk-and-max",
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> RepresentationEnvelope:
        if value.get("schema") != "irisu-r3h-robust-representation-envelope-g2-v2":
            raise RuntimeError("generation-02 support envelope schema mismatch")
        raw_profiles = value.get("profiles")
        if not isinstance(raw_profiles, Sequence):
            raise RuntimeError("generation-02 support profiles are malformed")
        profiles = tuple(
            RobustSignatureProfile(
                str(row["signature"]),
                tuple(float(item) for item in row["median"]),
                tuple(float(item) for item in row["scale"]),
                float(row["rms_radius"]),
                float(row["topk_radius"]),
                float(row["max_radius"]),
                int(row["seed_count"]),
            )
            for row in raw_profiles
            if isinstance(row, Mapping)
        )
        if len(profiles) != len(raw_profiles):
            raise RuntimeError("generation-02 support profiles are malformed")
        return cls(
            profiles,
            float(value["quantile"]),
            int(value["minimum_signature_seeds"]),
            int(value["top_k"]),
        )


@dataclass(frozen=True, slots=True)
class ResolutionFirstG2Config:
    members: int = 5
    body_width: int = 64
    hidden_width: int = 96
    training_steps: int = 800
    batch_size: int = 128
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    confidence_z: float = 1.6448536269514722
    gradient_clip: float = 5.0
    score_transform_scale: float = 8.0
    conformal_alpha: float = 0.05
    boundary_delta: float = 0.25
    envelope_quantile: float = 0.975
    envelope_top_k: int = 8
    minimum_signature_seeds: int = 8
    bootstrap_max_attempts: int = 4096
    support_thresholds: tuple[float, ...] = (
        0.00,
        0.25,
        0.50,
        0.65,
        0.75,
        0.85,
        0.90,
        0.95,
    )
    training_seeds: tuple[int, ...] = (
        2026073101,
        2026073102,
        2026073103,
        2026073104,
        2026073105,
    )

    def __post_init__(self) -> None:
        if min(
            self.members,
            self.body_width,
            self.hidden_width,
            self.training_steps,
            self.batch_size,
            self.envelope_top_k,
            self.bootstrap_max_attempts,
        ) < 1:
            raise ValueError("generation-02 learner sizes must be positive")
        if len(self.training_seeds) != self.members:
            raise ValueError("one fixed seed is required per ensemble member")
        if len(set(self.training_seeds)) != self.members:
            raise ValueError("ensemble training seeds must be distinct")
        if self.minimum_signature_seeds < 8:
            raise ValueError("generation-02 signatures require at least eight seeds")
        if not 0 < self.conformal_alpha < 1:
            raise ValueError("conformal alpha is invalid")
        if (
            not math.isfinite(self.boundary_delta)
            or self.boundary_delta < 0
            or not 0.5 < self.envelope_quantile < 1
            or not math.isfinite(self.score_transform_scale)
            or self.score_transform_scale <= 0
        ):
            raise ValueError("generation-02 numeric configuration is invalid")
        if (
            not self.support_thresholds
            or any(not math.isfinite(value) for value in self.support_thresholds)
        ):
            raise ValueError("generation-02 support threshold grid is invalid")

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["support_thresholds"] = list(self.support_thresholds)
        value["training_seeds"] = list(self.training_seeds)
        return value


class _Head(nn.Module):
    def __init__(self, input_width: int, hidden_width: int, outputs: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.SiLU(),
            nn.Linear(hidden_width, outputs),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.network(value)


class _G2Member(nn.Module):
    def __init__(self, config: ResolutionFirstG2Config) -> None:
        super().__init__()
        width = config.body_width
        self.body_encoder = nn.Sequential(
            nn.Linear(_MODEL_BODY_WIDTH, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        query_width = _MODEL_GLOBAL_WIDTH + 3 * len(FEATURE_NAMES) + 4 * width
        self.attention_query = nn.Sequential(
            nn.Linear(query_width, width), nn.SiLU(), nn.Linear(width, width)
        )
        self.representation_width = query_width + 3 * width
        self.resolution = _Head(self.representation_width, config.hidden_width, 1)
        self.unsafe = _Head(self.representation_width, config.hidden_width, 1)
        self.delta = _Head(self.representation_width, config.hidden_width, 2)
        self.b2 = _Head(self.representation_width, config.hidden_width, 2)
        self.score = _Head(self.representation_width, config.hidden_width, 2)

    def encode(
        self,
        global_features: Tensor,
        body_features: Tensor,
        body_mask: Tensor,
        candidate: Tensor,
        incumbent: Tensor,
        source: Tensor,
        destination: Tensor,
        incumbent_pair: Tensor,
    ) -> Tensor:
        encoded = self.body_encoder(body_features)
        batch = torch.arange(encoded.shape[0], device=encoded.device)
        endpoints = (
            encoded[batch, source],
            encoded[batch, destination],
            encoded[batch, incumbent_pair[:, 0]],
            encoded[batch, incumbent_pair[:, 1]],
        )
        candidate_values = torch.cat(
            (candidate, incumbent, candidate - incumbent), dim=-1
        )
        role_context = torch.cat(
            (global_features, candidate_values, *endpoints), dim=-1
        )
        query = self.attention_query(role_context)
        logits = torch.einsum("bnd,bd->bn", encoded, query)
        logits = (logits / math.sqrt(encoded.shape[-1])).masked_fill(
            ~body_mask, -torch.inf
        )
        attention = torch.softmax(logits, dim=1)
        attention_pool = torch.einsum("bn,bnd->bd", attention, encoded)
        count = body_mask.sum(1, keepdim=True).clamp_min(1)
        mean_pool = (encoded * body_mask.unsqueeze(-1)).sum(1) / count
        max_pool = encoded.masked_fill(
            ~body_mask.unsqueeze(-1), -torch.inf
        ).amax(1)
        return torch.cat(
            (role_context, mean_pool, max_pool, attention_pool), dim=-1
        )

    @staticmethod
    def quantiles(raw: Tensor) -> tuple[Tensor, Tensor]:
        q10 = raw[:, 0]
        return q10, q10 + F.softplus(raw[:, 1])


class ResolutionFirstG2Ensemble(nn.Module):
    def __init__(self, config: ResolutionFirstG2Config | None = None) -> None:
        super().__init__()
        self.config = ResolutionFirstG2Config() if config is None else config
        self.members = nn.ModuleList(
            _G2Member(self.config) for _ in range(self.config.members)
        )
        self.register_buffer("global_mean", torch.zeros(_MODEL_GLOBAL_WIDTH))
        self.register_buffer("global_scale", torch.ones(_MODEL_GLOBAL_WIDTH))
        self.register_buffer("body_mean", torch.zeros(_MODEL_BODY_WIDTH))
        self.register_buffer("body_scale", torch.ones(_MODEL_BODY_WIDTH))
        self.register_buffer("candidate_mean", torch.zeros(len(FEATURE_NAMES)))
        self.register_buffer("candidate_scale", torch.ones(len(FEATURE_NAMES)))
        self.register_buffer("delta_mean", torch.tensor(0.0))
        self.register_buffer("delta_scale", torch.tensor(1.0))
        self.register_buffer("b2_mean", torch.tensor(0.0))
        self.register_buffer("b2_scale", torch.tensor(1.0))
        self.support_envelope: RepresentationEnvelope | None = None

    def normalize(
        self,
        values: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        global_features, bodies, mask, candidate, incumbent, source, destination, pair = values
        return (
            (global_features - self.global_mean) / self.global_scale,
            (bodies - self.body_mean) / self.body_scale,
            mask,
            (candidate - self.candidate_mean) / self.candidate_scale,
            (incumbent - self.candidate_mean) / self.candidate_scale,
            source,
            destination,
            pair,
        )

    def manifest(self) -> dict[str, object]:
        if self.support_envelope is None:
            raise RuntimeError("generation-02 model lacks a support envelope")
        return {
            "schema": "irisu-r3h-resolution-first-ensemble-g2-v2",
            "teacher_schema_sha256": TeacherStateEncoder.schema.sha256,
            "candidate_feature_names": list(FEATURE_NAMES),
            "relational_feature_names": list(_RELATIONAL_NAMES),
            "absolute_tick_input": False,
            "absolute_id_input": False,
            "absolute_chain_id_input": False,
            "resolution_target": "candidate_resolved",
            "solvency_target": "absolute_b2",
            "score_target": "candidate_resolved_and_exact_safe",
            "config": self.config.manifest(),
            "support_envelope_sha256": self.support_envelope.sha256,
        }


def _record_stratum(record: BoardBranchRecord, boundary: float) -> str:
    if record.exact_unsafe:
        return "unsafe"
    if not record.candidate_resolved:
        return "unresolved"
    if not record.finite_pair:
        return "resolved-unpaired-rescue"
    assert record.delta_b2 is not None
    if record.delta_b2 < -boundary:
        return "resolved-safe-negative-delta"
    if record.delta_b2 <= boundary:
        return "resolved-safe-boundary-delta"
    return "resolved-safe-positive-delta"


def _pool_violations(
    records: Sequence[BoardBranchRecord],
    pool: Sequence[int],
    *,
    boundary: float,
    active_strata: frozenset[str],
    supported_signatures: frozenset[str],
) -> tuple[str, ...]:
    rows = [records[index] for index in pool]
    violations: list[str] = []
    if {row.candidate_resolved for row in rows} != {False, True}:
        violations.append("resolution-classes")
    if any(row.exact_unsafe for row in records) and not any(
        row.exact_unsafe for row in rows
    ):
        violations.append("unsafe-examples")
    missing_strata = active_strata - {
        _record_stratum(row, boundary) for row in rows
    }
    if missing_strata:
        violations.append(f"active-strata:{','.join(sorted(missing_strata))}")
    missing_signatures = supported_signatures - {row.signature for row in rows}
    if missing_signatures:
        violations.append(
            f"supported-signatures:{','.join(sorted(missing_signatures))}"
        )
    if any(row.finite_pair for row in records) and not any(row.finite_pair for row in rows):
        violations.append("finite-pair-diagnostic")
    return tuple(violations)


def _bootstrap_pool(
    records: Sequence[BoardBranchRecord],
    seed_groups: Sequence[Sequence[int]],
    *,
    config: ResolutionFirstG2Config,
    generator: torch.Generator,
) -> list[int]:
    active_strata = frozenset(
        _record_stratum(record, config.boundary_delta) for record in records
    )
    signature_seeds: dict[str, set[int]] = defaultdict(set)
    for record in records:
        signature_seeds[record.signature].add(record.seed)
    supported = frozenset(
        signature
        for signature, seeds in signature_seeds.items()
        if len(seeds) >= config.minimum_signature_seeds
    )
    if not supported:
        raise ValueError("no signature has enough seed support")
    for _attempt in range(config.bootstrap_max_attempts):
        bootstrap = torch.randint(
            len(seed_groups), (len(seed_groups),), generator=generator
        ).tolist()
        pool = [
            index for group_index in bootstrap for index in seed_groups[group_index]
        ]
        violations = _pool_violations(
            records,
            pool,
            boundary=config.boundary_delta,
            active_strata=active_strata,
            supported_signatures=supported,
        )
        if not violations:
            return pool
    raise RuntimeError(
        "whole-seed bootstrap could not satisfy fail-closed supervision constraints"
    )


def _balanced_indices(
    records: Sequence[BoardBranchRecord],
    pool: Sequence[int],
    *,
    count: int,
    boundary: float,
    generator: torch.Generator,
) -> Tensor:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in pool:
        grouped[_record_stratum(records[index], boundary)].append(index)
    active = [grouped[name] for name in sorted(grouped)]
    if not active:
        raise ValueError("balanced sampler has no rows")
    chosen: list[int] = []
    base, extra = divmod(count, len(active))
    for position, stratum in enumerate(active):
        draws = torch.randint(
            len(stratum),
            (base + int(position < extra),),
            generator=generator,
        )
        chosen.extend(stratum[index] for index in draws.tolist())
    order = torch.randperm(len(chosen), generator=generator)
    return torch.tensor(chosen, dtype=torch.long)[order]


def _pinball(prediction: Tensor, target: Tensor, quantile: float) -> Tensor:
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def _score_target(value: Tensor, scale: float) -> Tensor:
    return torch.sign(value) * torch.log1p(value.abs()) / scale


def _rank_loss(
    prediction: Tensor,
    target: Tensor,
    chosen: Tensor,
    records: Sequence[BoardBranchRecord],
) -> Tensor:
    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for local, global_index in enumerate(chosen.tolist()):
        if records[global_index].safe_score_row_g2:
            grouped[records[global_index].query_key].append(local)
    losses: list[Tensor] = []
    for local_indices in grouped.values():
        for left in range(len(local_indices)):
            for right in range(left + 1, len(local_indices)):
                i, j = local_indices[left], local_indices[right]
                sign = torch.sign(target[i] - target[j])
                if bool(sign):
                    losses.append(F.softplus(-sign * (prediction[i] - prediction[j])))
    return torch.stack(losses).mean() if losses else prediction.sum() * 0.0


@torch.inference_mode()
def _learned_representations(
    model: ResolutionFirstG2Ensemble,
    dataset: BoardResolutionDataset,
) -> Tensor:
    values = model.normalize(dataset.tensors())
    return torch.cat([member.encode(*values) for member in model.members], dim=-1)


def _fit_envelope(
    representations: Tensor,
    records: Sequence[BoardBranchRecord],
    config: ResolutionFirstG2Config,
) -> RepresentationEnvelope:
    x = representations.detach().cpu().double().numpy()
    grouped: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        grouped[record.signature][record.seed].append(index)
    profiles: list[RobustSignatureProfile] = []
    for signature in sorted(grouped):
        per_seed = grouped[signature]
        if len(per_seed) < config.minimum_signature_seeds:
            continue
        seed_centers = np.asarray(
            [np.median(x[per_seed[seed]], axis=0) for seed in sorted(per_seed)]
        )
        median = np.median(seed_centers, axis=0)
        mad = 1.4826 * np.median(np.abs(seed_centers - median), axis=0)
        fallback = np.std(seed_centers, axis=0)
        scale = np.where(mad > 1e-4, mad, np.maximum(fallback, 1e-3))
        per_seed_distances: list[tuple[float, float, float]] = []
        top_count = min(config.envelope_top_k, x.shape[1])
        for seed in sorted(per_seed):
            standardized = np.abs((x[per_seed[seed]] - median) / scale)
            rms = np.sqrt(np.mean(np.square(standardized), axis=1))
            sorted_values = np.sort(standardized, axis=1)
            topk = np.sqrt(
                np.mean(np.square(sorted_values[:, -top_count:]), axis=1)
            )
            maximum = np.max(standardized, axis=1)
            per_seed_distances.append(
                (float(np.max(rms)), float(np.max(topk)), float(np.max(maximum)))
            )
        distances = np.asarray(per_seed_distances)
        radii = [
            max(
                float(
                    np.quantile(
                        distances[:, column],
                        config.envelope_quantile,
                        method="higher",
                    )
                ),
                1e-6,
            )
            for column in range(3)
        ]
        profiles.append(
            RobustSignatureProfile(
                signature,
                tuple(float(value) for value in median),
                tuple(float(value) for value in scale),
                *radii,
                len(per_seed),
            )
        )
    if not profiles:
        raise ValueError("no signature has enough learned-representation support")
    return RepresentationEnvelope(
        tuple(profiles),
        config.envelope_quantile,
        config.minimum_signature_seeds,
        config.envelope_top_k,
    )


def train_resolution_first_g2(
    dataset: BoardResolutionDataset,
    *,
    config: ResolutionFirstG2Config | None = None,
) -> ResolutionFirstG2Ensemble:
    """Train a fail-closed whole-seed ensemble and robust support envelope."""

    resolved = ResolutionFirstG2Config() if config is None else config
    records = dataset.records
    if len(dataset.seeds) < resolved.minimum_signature_seeds:
        raise ValueError("whole-seed training has fewer than eight seeds")
    if {record.candidate_resolved for record in records} != {False, True}:
        raise ValueError("resolution training requires both candidate-resolved classes")
    if not any(record.finite_pair for record in records):
        raise ValueError("delta diagnostic requires finite paired rows")
    resolved_b2 = [record for record in records if record.candidate_resolved]
    if any(record.b2 is None for record in resolved_b2):
        raise ValueError("candidate-resolved row lacks absolute B2")
    model = ResolutionFirstG2Ensemble(resolved)
    raw = dataset.tensors()
    global_features, bodies, mask, candidate, _incumbent, *_ = raw
    resolution = torch.tensor([record.candidate_resolved for record in records])
    unsafe = torch.tensor([float(record.exact_unsafe) for record in records])
    finite = torch.tensor([record.finite_pair for record in records])
    b2_mask = resolution
    delta = torch.tensor(
        [0.0 if record.delta_b2 is None else record.delta_b2 for record in records],
        dtype=torch.float32,
    )
    b2 = torch.tensor(
        [0.0 if record.b2 is None else record.b2 for record in records],
        dtype=torch.float32,
    )
    score = torch.tensor(
        [record.score_advantage for record in records], dtype=torch.float32
    )
    safe_score = torch.tensor([record.safe_score_row_g2 for record in records])
    with torch.no_grad():
        model.global_mean.copy_(global_features.mean(0))
        model.global_scale.copy_(
            global_features.std(0, unbiased=False).clamp_min(1e-5)
        )
        active_bodies = bodies[mask]
        model.body_mean.copy_(active_bodies.mean(0))
        model.body_scale.copy_(
            active_bodies.std(0, unbiased=False).clamp_min(1e-5)
        )
        model.candidate_mean.copy_(candidate.mean(0))
        model.candidate_scale.copy_(
            candidate.std(0, unbiased=False).clamp_min(1e-5)
        )
        model.delta_mean.copy_(delta[finite].mean())
        model.delta_scale.copy_(delta[finite].std(unbiased=False).clamp_min(1.0))
        model.b2_mean.copy_(b2[b2_mask].mean())
        model.b2_scale.copy_(b2[b2_mask].std(unbiased=False).clamp_min(1.0))
    values = model.normalize(raw)
    delta_target = (delta - model.delta_mean) / model.delta_scale
    b2_target = (b2 - model.b2_mean) / model.b2_scale
    transformed_score = _score_target(score, resolved.score_transform_scale)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record.seed].append(index)
    seed_groups = [grouped[seed] for seed in sorted(grouped)]
    for member, seed in zip(model.members, resolved.training_seeds, strict=True):
        torch.manual_seed(seed)
        for module in member.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
        generator = torch.Generator().manual_seed(seed)
        pool = _bootstrap_pool(
            records, seed_groups, config=resolved, generator=generator
        )
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
                boundary=resolved.boundary_delta,
                generator=generator,
            )
            representation = member.encode(*(value[chosen] for value in values))
            resolution_logit = member.resolution(representation).squeeze(-1)
            unsafe_logit = member.unsafe(representation).squeeze(-1)
            delta_q10, delta_q50 = member.quantiles(member.delta(representation))
            b2_q10, b2_q50 = member.quantiles(member.b2(representation))
            score_q10, score_q50 = member.quantiles(member.score(representation))
            resolution_loss = F.binary_cross_entropy_with_logits(
                resolution_logit, resolution[chosen].float()
            )
            unsafe_loss = F.binary_cross_entropy_with_logits(
                unsafe_logit, unsafe[chosen]
            )
            finite_rows = finite[chosen]
            delta_loss = (
                _pinball(
                    delta_q10[finite_rows],
                    delta_target[chosen][finite_rows],
                    0.10,
                )
                + _pinball(
                    delta_q50[finite_rows],
                    delta_target[chosen][finite_rows],
                    0.50,
                )
                if bool(finite_rows.any())
                else delta_q10.sum() * 0.0
            )
            b2_rows = b2_mask[chosen]
            b2_loss = (
                _pinball(b2_q10[b2_rows], b2_target[chosen][b2_rows], 0.10)
                + _pinball(b2_q50[b2_rows], b2_target[chosen][b2_rows], 0.50)
                if bool(b2_rows.any())
                else b2_q10.sum() * 0.0
            )
            score_rows = safe_score[chosen]
            if bool(score_rows.any()):
                score_loss = _pinball(
                    score_q10[score_rows],
                    transformed_score[chosen][score_rows],
                    0.10,
                ) + _pinball(
                    score_q50[score_rows],
                    transformed_score[chosen][score_rows],
                    0.50,
                )
                score_loss = score_loss + 0.5 * _rank_loss(
                    score_q50,
                    transformed_score[chosen],
                    chosen,
                    records,
                )
            else:
                score_loss = score_q10.sum() * 0.0
            loss = (
                resolution_loss
                + unsafe_loss
                + delta_loss
                + b2_loss
                + score_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                member.parameters(), resolved.gradient_clip
            )
            optimizer.step()
    model.eval()
    model.support_envelope = _fit_envelope(
        _learned_representations(model, dataset), records, resolved
    )
    return model


@torch.inference_mode()
def predict_records_g2(
    model: ResolutionFirstG2Ensemble,
    dataset: BoardResolutionDataset,
) -> tuple[dict[str, object], ...]:
    if model.support_envelope is None:
        raise RuntimeError("generation-02 model lacks a support envelope")
    model.eval()
    values = model.normalize(dataset.tensors())
    representations: list[Tensor] = []
    resolution: list[Tensor] = []
    unsafe: list[Tensor] = []
    delta: list[Tensor] = []
    b2: list[Tensor] = []
    score: list[Tensor] = []
    score_q50s: list[Tensor] = []
    for member in model.members:
        representation = member.encode(*values)
        representations.append(representation)
        resolution.append(member.resolution(representation).squeeze(-1).sigmoid())
        unsafe.append(member.unsafe(representation).squeeze(-1).sigmoid())
        delta_q10, _ = member.quantiles(member.delta(representation))
        b2_q10, _ = member.quantiles(member.b2(representation))
        score_q10, score_q50 = member.quantiles(member.score(representation))
        delta.append(delta_q10 * model.delta_scale + model.delta_mean)
        b2.append(b2_q10 * model.b2_scale + model.b2_mean)
        score.append(score_q10)
        score_q50s.append(score_q50)
    learned = torch.cat(representations, dim=-1).cpu().numpy()
    stacks = [
        torch.stack(items)
        for items in (resolution, unsafe, delta, b2, score, score_q50s)
    ]
    output: list[dict[str, object]] = []
    for index, record in enumerate(dataset.records):
        means = [float(value[:, index].mean()) for value in stacks]
        deviations = [
            float(value[:, index].std(unbiased=False)) for value in stacks
        ]
        (
            resolution_mean,
            unsafe_mean,
            delta_mean,
            b2_mean,
            score_mean,
            score_q50_mean,
        ) = means
        (
            resolution_std,
            unsafe_std,
            delta_std,
            b2_std,
            score_std,
            score_q50_std,
        ) = deviations
        resolution_lcb = (
            resolution_mean - model.config.confidence_z * resolution_std
        )
        unsafe_ucb = unsafe_mean + model.config.confidence_z * unsafe_std
        representation = learned[index].tolist()
        distances = model.support_envelope.distances(
            record.signature, representation
        )
        output.append(
            {
                "seed": record.seed,
                "query_id": record.query_id,
                "ordinal": record.ordinal,
                "candidate_id": record.candidate_id,
                "action_id": record.action_id,
                "signature": record.signature,
                "envelope_supported": model.support_envelope.contains(
                    record.signature, representation
                ),
                "support_distance": None if distances is None else distances[0],
                "support_distance_rms": None if distances is None else distances[0],
                "support_distance_topk": None if distances is None else distances[1],
                "support_distance_max": None if distances is None else distances[2],
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
                "b2_q10_mean": b2_mean,
                "b2_q10_std": b2_std,
                "b2_lcb": b2_mean - model.config.confidence_z * b2_std,
                "score_q10_mean": score_mean,
                "score_q10_std": score_std,
                "score_q50_mean": score_q50_mean,
                "score_q50_std": score_q50_std,
                "score_lcb": score_mean
                - model.config.confidence_z * score_std,
                "exact_candidate_resolved": record.candidate_resolved,
                "exact_finite_pair": record.finite_pair,
                "exact_unsafe": record.exact_unsafe,
                "exact_b2": record.b2,
                "exact_delta_b2": record.delta_b2,
                "exact_score_advantage": record.score_advantage,
            }
        )
    return tuple(output)


def _query_groups(
    predictions: Sequence[Mapping[str, object]],
) -> dict[tuple[int, str], list[Mapping[str, object]]]:
    grouped: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in predictions:
        grouped[(int(row["seed"]), str(row["query_id"]))].append(row)
    if not grouped:
        raise ValueError("generation-02 predictions are empty")
    for key, rows in grouped.items():
        ordinals = sorted(int(row["ordinal"]) for row in rows)
        if ordinals != list(range(len(rows))):
            raise ValueError(f"query {key!r} has invalid candidate ordinals")
    return grouped


def resolution_auroc_g2(
    predictions: Sequence[Mapping[str, object]],
    *,
    minimum: float = 0.75,
) -> dict[str, object]:
    alternatives = [row for row in predictions if int(row["ordinal"]) != 0]
    positive = [
        row for row in alternatives if bool(row["exact_candidate_resolved"])
    ]
    negative = [
        row for row in alternatives if not bool(row["exact_candidate_resolved"])
    ]
    if not positive or not negative:
        return {
            "schema": "irisu-r3h-resolution-auroc-g2-v2",
            "target": "candidate_resolved",
            "candidates": len(alternatives),
            "positive": len(positive),
            "negative": len(negative),
            "auroc": None,
            "minimum": minimum,
            "passed": False,
            "reason": "undefined-single-class",
        }
    wins = 0.0
    for left in positive:
        for right in negative:
            a = float(left["resolution_mean"])
            b = float(right["resolution_mean"])
            wins += float(a > b) + 0.5 * float(a == b)
    auroc = wins / (len(positive) * len(negative))
    return {
        "schema": "irisu-r3h-resolution-auroc-g2-v2",
        "target": "candidate_resolved",
        "candidates": len(alternatives),
        "positive": len(positive),
        "negative": len(negative),
        "auroc": auroc,
        "minimum": minimum,
        "passed": auroc >= minimum,
        "reason": None,
    }


@dataclass(frozen=True, slots=True)
class G2SupportCalibration:
    threshold: float
    selected_candidates: int
    candidate_count: int
    selected_queries: int
    query_count: int
    coverage: float
    selected_bad_candidates: int
    selected_bad_seeds: int
    resolution_auroc: float
    grid: tuple[Mapping[str, object], ...]

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["schema"] = "irisu-r3h-support-calibration-g2-v2"
        value["grid"] = [dict(row) for row in self.grid]
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def _prefix_candidates(
    predictions: Sequence[Mapping[str, object]],
    threshold: float,
) -> tuple[Mapping[str, object], ...]:
    selected: list[Mapping[str, object]] = []
    for rows in _query_groups(predictions).values():
        ordered = sorted(rows, key=lambda row: int(row["ordinal"]))
        incumbent_action = str(ordered[0]["action_id"])
        for row in ordered[1:]:
            support_score = float(row["support_score"])
            score_lcb = float(row["score_lcb"])
            if (
                bool(row["envelope_supported"])
                and str(row["action_id"]) != incumbent_action
                and math.isfinite(support_score)
                and support_score > threshold
                and math.isfinite(score_lcb)
                and score_lcb > 0
            ):
                selected.append(row)
    return tuple(selected)


def fit_support_calibration_g2(
    predictions: Sequence[Mapping[str, object]],
    *,
    thresholds: Sequence[float] = ResolutionFirstG2Config().support_thresholds,
    minimum_coverage: float = 0.05,
    minimum_auroc: float = 0.75,
) -> G2SupportCalibration:
    auroc = resolution_auroc_g2(predictions, minimum=minimum_auroc)
    if not auroc["passed"]:
        raise RuntimeError("held-out candidate-resolution AUROC gate failed")
    groups = _query_groups(predictions)
    alternatives = sum(
        sum(int(row["ordinal"]) != 0 for row in rows) for rows in groups.values()
    )
    grid: list[dict[str, object]] = []
    winner: dict[str, object] | None = None
    for threshold in thresholds:
        selected = _prefix_candidates(predictions, float(threshold))
        bad = [
            row
            for row in selected
            if not bool(row["exact_candidate_resolved"])
            or bool(row["exact_unsafe"])
        ]
        selected_queries = len(
            {(int(row["seed"]), str(row["query_id"])) for row in selected}
        )
        coverage = selected_queries / len(groups)
        point: dict[str, object] = {
            "threshold": float(threshold),
            "selected_candidates": len(selected),
            "candidate_count": alternatives,
            "selected_queries": selected_queries,
            "query_count": len(groups),
            "coverage": coverage,
            "selected_bad_candidates": len(bad),
            "selected_bad_seeds": len({int(row["seed"]) for row in bad}),
            "passed": not bad and coverage >= minimum_coverage,
        }
        grid.append(point)
        if winner is None and point["passed"] is True:
            winner = point
    if winner is None:
        raise RuntimeError("no preregistered generation-02 support threshold passed")
    return G2SupportCalibration(
        float(winner["threshold"]),
        int(winner["selected_candidates"]),
        int(winner["candidate_count"]),
        int(winner["selected_queries"]),
        int(winner["query_count"]),
        float(winner["coverage"]),
        int(winner["selected_bad_candidates"]),
        int(winner["selected_bad_seeds"]),
        float(auroc["auroc"]),
        tuple(grid),
    )


@dataclass(frozen=True, slots=True)
class G2B2Calibration:
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
        value["schema"] = "irisu-r3h-absolute-b2-calibration-g2-v2"
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


def fit_selective_calibration_g2(
    predictions: Sequence[Mapping[str, object]],
    support: G2SupportCalibration,
    *,
    alpha: float = 0.05,
    required_episodes: int | None = None,
) -> G2B2Calibration:
    if not 0 < alpha < 1:
        raise ValueError("absolute-B2 conformal alpha is invalid")
    seeds = {int(row["seed"]) for row in predictions}
    if required_episodes is not None and len(seeds) != required_episodes:
        raise ValueError("absolute-B2 conformal episode count mismatch")
    residuals: dict[int, list[float]] = {seed: [] for seed in seeds}
    selected = bad = 0
    for row in _prefix_candidates(predictions, support.threshold):
        selected += 1
        seed = int(row["seed"])
        exact = row.get("exact_b2")
        predicted = float(row["b2_lcb"])
        if (
            not bool(row["exact_candidate_resolved"])
            or bool(row["exact_unsafe"])
            or exact is None
            or not math.isfinite(float(exact))
            or not math.isfinite(predicted)
        ):
            bad += 1
            residuals[seed].append(math.inf)
        else:
            residuals[seed].append(predicted - float(exact))
    maxima = tuple(
        sorted(
            (seed, max(values) if values else 0.0)
            for seed, values in residuals.items()
        )
    )
    ordered = sorted(value for _seed, value in maxima)
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    q = math.inf if rank > len(ordered) else ordered[rank - 1]
    return G2B2Calibration(
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
class G2SelectedCandidate:
    seed: int
    query_id: str
    ordinal: int
    candidate_id: str
    override: bool
    exact_candidate_resolved: bool
    exact_finite_pair: bool
    exact_unsafe: bool
    exact_b2: float | None
    exact_delta_b2: float | None
    exact_score_advantage: float
    reasons: tuple[str, ...]


def select_candidates_g2(
    predictions: Sequence[Mapping[str, object]],
    support: G2SupportCalibration,
    calibration: G2B2Calibration,
) -> tuple[G2SelectedCandidate, ...]:
    if calibration.support_calibration_sha256 != support.sha256:
        raise ValueError("support/absolute-B2 calibration identity mismatch")
    output: list[G2SelectedCandidate] = []
    for key, rows in sorted(_query_groups(predictions).items()):
        ordered = sorted(rows, key=lambda row: int(row["ordinal"]))
        incumbent = ordered[0]
        prefix = [
            row
            for row in ordered[1:]
            if bool(row["envelope_supported"])
            and str(row["action_id"]) != str(incumbent["action_id"])
            and math.isfinite(float(row["support_score"]))
            and float(row["support_score"]) > support.threshold
            and math.isfinite(float(row["score_lcb"]))
            and float(row["score_lcb"]) > 0
        ]
        certified = [
            row
            for row in prefix
            if math.isfinite(calibration.q)
            and math.isfinite(float(row["b2_lcb"]))
            and float(row["b2_lcb"]) - calibration.q >= 0
        ]
        if certified:
            best = max(float(row["score_lcb"]) for row in certified)
            winners = [
                row for row in certified if float(row["score_lcb"]) == best
            ]
        else:
            winners = []
        if len(winners) == 1:
            selected, reasons = winners[0], ()
        elif len(winners) > 1:
            selected, reasons = incumbent, ("conservative-score-tie",)
        elif not math.isfinite(calibration.q):
            selected, reasons = incumbent, ("nonfinite-b2-calibration",)
        else:
            selected, reasons = incumbent, ("no-certified-alternative",)
        output.append(
            G2SelectedCandidate(
                int(key[0]),
                key[1],
                int(selected["ordinal"]),
                str(selected["candidate_id"]),
                int(selected["ordinal"]) != 0,
                bool(selected["exact_candidate_resolved"]),
                bool(selected["exact_finite_pair"]),
                bool(selected["exact_unsafe"]),
                (
                    None
                    if selected.get("exact_b2") is None
                    else float(selected["exact_b2"])
                ),
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
    selections: Sequence[G2SelectedCandidate],
    *,
    replicates: int = 10_000,
    alpha: float = 0.05,
) -> float:
    counts: dict[int, list[int]] = defaultdict(lambda: [0, 0])
    for selection in selections:
        counts[selection.seed][0] += int(selection.override)
        counts[selection.seed][1] += 1
    values = np.asarray(list(counts.values()), dtype=np.int64)
    digest = hashlib.sha256(
        b"r3h-resolution-first-g2-v2|coverage-bootstrap"
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


def viability_report_g2(
    selections: Sequence[G2SelectedCandidate],
    calibration: G2B2Calibration,
    *,
    minimum_coverage: float = 0.05,
) -> dict[str, object]:
    if not selections:
        raise ValueError("viability screen has no decisions")
    overrides = [selection for selection in selections if selection.override]
    bad_seeds = {
        selection.seed
        for selection in overrides
        if not selection.exact_candidate_resolved or selection.exact_unsafe
    }
    score = [selection.exact_score_advantage for selection in overrides]
    b2 = [
        float(selection.exact_b2)
        for selection in overrides
        if selection.exact_b2 is not None
        and math.isfinite(float(selection.exact_b2))
    ]
    complete_b2 = len(b2) == len(overrides)
    coverage = len(overrides) / len(selections)
    coverage_lcb = _clustered_coverage_lcb(selections)
    episode_count = len({selection.seed for selection in selections})
    gates = {
        "finite_absolute_b2_conformal_q": math.isfinite(calibration.q),
        "raw_coverage": coverage >= minimum_coverage,
        "positive_clustered_coverage_lcb": coverage_lcb > 0,
        "zero_selected_bad_candidates": not any(
            not selection.exact_candidate_resolved or selection.exact_unsafe
            for selection in overrides
        ),
        "zero_false_safe_episodes": not bad_seeds,
        "positive_median_score_advantage": bool(score)
        and float(np.median(score)) > 0,
        "complete_finite_absolute_b2": bool(overrides) and complete_b2,
        "all_nonnegative_absolute_b2": bool(overrides)
        and complete_b2
        and all(value >= 0 for value in b2),
        "nonnegative_median_absolute_b2": bool(overrides)
        and complete_b2
        and float(np.median(b2)) >= 0,
    }
    return {
        "schema": "irisu-r3h-offline-viability-g2-v2",
        "selection_policy": "all-prefix-absolute-b2-then-unique-score",
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
        "median_exact_absolute_b2": None if not b2 else float(np.median(b2)),
        "missing_or_nonfinite_absolute_b2": len(overrides) - len(b2),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _fsync_directory_g2(path: str) -> None:
    if (
        os.path.islink(path)
        or not os.path.isdir(path)
        or os.path.realpath(path) != path
    ):
        raise RuntimeError(f"generation-02 checkpoint directory is indirect: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable_g2(path: str) -> None:
    path = os.path.abspath(path)
    missing: list[str] = []
    cursor = path
    while not os.path.lexists(cursor):
        missing.append(cursor)
        parent = os.path.dirname(cursor)
        if parent == cursor:
            raise RuntimeError(f"cannot create checkpoint directory: {path}")
        cursor = parent
    _fsync_directory_g2(cursor)
    for directory in reversed(missing):
        try:
            os.mkdir(directory)
        except FileExistsError:
            pass
        _fsync_directory_g2(directory)
        _fsync_directory_g2(os.path.dirname(directory))
    _fsync_directory_g2(path)


def save_checkpoint_g2(
    path: str | os.PathLike[str],
    model: ResolutionFirstG2Ensemble,
    *,
    metadata: Mapping[str, object] | None = None,
) -> str:
    destination = os.path.abspath(os.fspath(path))
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to replace checkpoint {destination}")
    if model.support_envelope is None:
        raise RuntimeError("generation-02 model lacks a support envelope")
    parent = os.path.dirname(destination)
    _mkdir_durable_g2(parent)
    name = os.path.basename(destination)
    prefix = f".{name}.tmp-"
    if any(entry.startswith(prefix) for entry in os.listdir(parent)):
        raise RuntimeError("generation-02 checkpoint has unrecovered atomic remnants")
    payload = {
        "schema": "irisu-r3h-resolution-first-checkpoint-g2-v2",
        "manifest": model.manifest(),
        "support_envelope": model.support_envelope.manifest(),
        "state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    temporary = os.path.join(
        parent, f".{name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    linked = False
    try:
        torch.save(payload, temporary)
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, destination)
        linked = True
        _fsync_directory_g2(parent)
    finally:
        if linked and os.path.exists(temporary):
            os.unlink(temporary)
        _fsync_directory_g2(parent)
    if (
        os.path.islink(destination)
        or not os.path.isfile(destination)
        or os.path.realpath(destination) != destination
        or os.stat(destination).st_nlink != 1
    ):
        raise RuntimeError("generation-02 checkpoint link is indirect")
    with open(destination, "rb") as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def load_checkpoint_g2(
    path: str | os.PathLike[str],
) -> tuple[ResolutionFirstG2Ensemble, dict[str, object]]:
    payload = torch.load(os.fspath(path), map_location="cpu", weights_only=True)
    if payload.get("schema") != "irisu-r3h-resolution-first-checkpoint-g2-v2":
        raise RuntimeError("generation-02 checkpoint schema mismatch")
    manifest = payload.get("manifest")
    envelope_raw = payload.get("support_envelope")
    if not isinstance(manifest, Mapping) or not isinstance(envelope_raw, Mapping):
        raise RuntimeError("generation-02 checkpoint manifest is malformed")
    if (
        manifest.get("schema") != "irisu-r3h-resolution-first-ensemble-g2-v2"
        or manifest.get("teacher_schema_sha256")
        != TeacherStateEncoder.schema.sha256
        or tuple(manifest.get("candidate_feature_names", ())) != FEATURE_NAMES
        or tuple(manifest.get("relational_feature_names", ()))
        != _RELATIONAL_NAMES
        or manifest.get("absolute_tick_input") is not False
        or manifest.get("absolute_id_input") is not False
        or manifest.get("absolute_chain_id_input") is not False
        or manifest.get("resolution_target") != "candidate_resolved"
        or manifest.get("solvency_target") != "absolute_b2"
        or manifest.get("score_target")
        != "candidate_resolved_and_exact_safe"
    ):
        raise RuntimeError("generation-02 checkpoint feature identity mismatch")
    config_raw = dict(manifest["config"])
    config_raw["support_thresholds"] = tuple(config_raw["support_thresholds"])
    config_raw["training_seeds"] = tuple(config_raw["training_seeds"])
    model = ResolutionFirstG2Ensemble(ResolutionFirstG2Config(**config_raw))
    envelope = RepresentationEnvelope.from_manifest(envelope_raw)
    if manifest.get("support_envelope_sha256") != envelope.sha256:
        raise RuntimeError("generation-02 learned support identity mismatch")
    state = payload.get("state_dict")
    if not isinstance(state, Mapping):
        raise RuntimeError("generation-02 checkpoint state is malformed")
    if any(
        not isinstance(tensor, Tensor)
        or (
            (tensor.is_floating_point() or tensor.is_complex())
            and not bool(torch.isfinite(tensor).all())
        )
        for tensor in state.values()
    ):
        raise RuntimeError("generation-02 checkpoint state is nonfinite")
    model.support_envelope = envelope
    model.load_state_dict(state, strict=True)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise RuntimeError("generation-02 checkpoint metadata is malformed")
    return model.eval(), dict(metadata)
