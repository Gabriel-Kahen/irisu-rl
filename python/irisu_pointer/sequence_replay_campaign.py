"""Identity-bound development data utilities for sequence-replay campaigns."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from irisu_rl.encoding import EncodedBatch
from irisu_rl.schema import TEACHER_V1

from .action import PointerActionSpec
from .geometry_learning import GeometryDataset, GeometryExample
from .sequence_replay import (
    SEQUENCE_REPLAY_EVENT_FEATURES,
    SEQUENCE_REPLAY_EVENT_WIDTH,
    SequenceReplayBatch,
    SequenceReplayOutput,
    SequenceReplayTargets,
    batch_steering_example_sequences,
)
from .steering import SteeringIntent
from .steering_learning import SteeringExample


LEGACY_GEOMETRY_COLLECTION_FORMAT = "irisu-r3e-geometry-collection-v1"
CAMPAIGN_DATA_FORMAT = "irisu-sequence-replay-campaign-data-v1"
_REPLAY_IDENTITY = re.compile(r"^replay:([0-9a-f]{64}):([0-9]+)$")
_GEOMETRY_IDENTITY = re.compile(
    r"^r3e-(oracle|learner)-visited:([0-9a-f]{8}):([0-9]+):([0-9]+)$"
)
_INTENTS = tuple(SteeringIntent)
_EVENT_INDEX = {
    name: index for index, name in enumerate(SEQUENCE_REPLAY_EVENT_FEATURES)
}
_GLOBAL_DELTAS = {
    "delta_score_scaled": "score_signed_log1p",
    "delta_gauge": "gauge_fraction",
    "delta_clears_scaled": "qualifying_clears_log1p",
    "delta_highest_chain_scaled": "highest_chain_log1p",
}


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _optional_sha256(value: str | None, name: str) -> str | None:
    return None if value is None else _sha256(value, name)


def _file_bytes(path: str | Path) -> tuple[Path, bytes, str]:
    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise ValueError("legacy geometry collection must not be a symbolic link")
    resolved = supplied.resolve(strict=True)
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or not resolved.is_file():
        raise RuntimeError("legacy geometry collection changed while loading")
    return resolved, payload, hashlib.sha256(payload).hexdigest()


def _tensor_identity(value: Tensor) -> dict[str, object]:
    tensor = value.detach().cpu().contiguous()
    return {
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": hashlib.sha256(
            tensor.numpy().tobytes(order="C")
        ).hexdigest(),
    }


def _source_tick(example: SteeringExample) -> int:
    return int(example.observation.source_tick[0])


def causal_replay_event_vectors(
    examples: Sequence[SteeringExample],
) -> Tensor:
    """Build events at ``t`` from labels/observations available by ``t`` only."""

    ordered = tuple(examples)
    if not ordered:
        raise ValueError("causal replay events require at least one example")
    schema = ordered[0].observation.schema
    if any(value.schema_sha256 != schema.sha256 for value in ordered):
        raise ValueError("causal replay episode mixes observation schemas")
    ticks = tuple(_source_tick(value) for value in ordered)
    if any(right <= left for left, right in zip(ticks, ticks[1:])):
        raise ValueError("causal replay examples must be strictly chronological")
    dtype = torch.from_numpy(ordered[0].observation.global_features).dtype
    output = torch.zeros(
        len(ordered), SEQUENCE_REPLAY_EVENT_WIDTH, dtype=dtype
    )
    global_indices = {
        event_name: schema.global_features.index(feature_name)
        for event_name, feature_name in _GLOBAL_DELTAS.items()
    }
    last_shot_tick: int | None = None
    for step in range(1, len(ordered)):
        previous = ordered[step - 1]
        current = ordered[step]
        output[
            step,
            _EVENT_INDEX[
                "previous_action_shot"
                if previous.is_shot
                else "previous_action_wait"
            ],
        ] = 1.0
        intent = _INTENTS[previous.intent_index]
        output[step, _EVENT_INDEX[f"previous_intent_{intent.value}"]] = 1.0
        elapsed = ticks[step] - ticks[step - 1]
        output[step, _EVENT_INDEX["elapsed_ticks_log_scaled"]] = (
            math.log1p(elapsed) / 8.0
        )
        if previous.is_shot:
            last_shot_tick = ticks[step - 1]
        if last_shot_tick is not None:
            output[step, _EVENT_INDEX["time_since_shot_log_scaled"]] = (
                math.log1p(ticks[step] - last_shot_tick) / 8.0
            )
        previous_global = torch.from_numpy(
            previous.observation.global_features[0]
        )
        current_global = torch.from_numpy(
            current.observation.global_features[0]
        )
        for event_name, global_index in global_indices.items():
            output[step, _EVENT_INDEX[event_name]] = (
                current_global[global_index] - previous_global[global_index]
            )
    if not bool(torch.isfinite(output).all()):
        raise ValueError("causal replay event construction produced nonfinite values")
    return output


@dataclass(frozen=True, slots=True)
class NormalizedReplayEpisode:
    identity: str
    replay_sha256: str
    collection_sha256: str
    base_checkpoint_sha256: str
    examples: tuple[SteeringExample, ...]
    source_ticks: tuple[int, ...]
    event_features: Tensor
    original_example_sha256s: tuple[str, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "format": CAMPAIGN_DATA_FORMAT,
            "kind": "trusted-replay-episode",
            "identity": self.identity,
            "replay_sha256": self.replay_sha256,
            "collection_sha256": self.collection_sha256,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "source_ticks": list(self.source_ticks),
            "original_examples": list(self.original_example_sha256s),
            "normalized_examples": [value.sha256 for value in self.examples],
            "events": _tensor_identity(self.event_features),
            "event_semantics": (
                "prior label and current-minus-prior public observation only; "
                "unobserved hit/join/clear fields remain zero"
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def normalize_trusted_replay_examples(
    examples: Sequence[SteeringExample],
    *,
    expected_replay_sha256: str,
    expected_collection_sha256: str,
    base_checkpoint_sha256: str,
) -> NormalizedReplayEpisode:
    """Sort strict replay labels and replace per-shot IDs with one episode ID."""

    replay_sha = _sha256(expected_replay_sha256, "trusted replay identity")
    collection_sha = _sha256(
        expected_collection_sha256, "trusted replay collection identity"
    )
    base_sha = _sha256(base_checkpoint_sha256, "frozen base checkpoint")
    supplied = tuple(examples)
    if not supplied:
        raise ValueError("trusted replay examples must not be empty")
    parsed: list[tuple[int, int, SteeringExample]] = []
    for example in supplied:
        if not isinstance(example, SteeringExample):
            raise TypeError("trusted replay contains a non-steering example")
        match = _REPLAY_IDENTITY.fullmatch(example.episode_identity)
        if (
            match is None
            or match.group(1) != replay_sha
            or example.provenance_sha256 != collection_sha
        ):
            raise ValueError("trusted replay example identity is not bound")
        parsed.append((_source_tick(example), int(match.group(2)), example))
    if len({(tick, ordinal) for tick, ordinal, _ in parsed}) != len(parsed):
        raise ValueError("trusted replay repeats a chronological label")
    parsed.sort(key=lambda value: (value[0], value[1]))
    ticks = tuple(value[0] for value in parsed)
    if any(right <= left for left, right in zip(ticks, ticks[1:])):
        raise ValueError("trusted replay labels do not have unique increasing ticks")
    if len({value.schema_sha256 for value in supplied}) != 1 or len(
        {value.pointer_spec_sha256 for value in supplied}
    ) != 1:
        raise ValueError("trusted replay examples mix schemas")
    identity = f"sequence-replay:trusted:{replay_sha}"
    normalized = tuple(
        replace(value[2], episode_identity=identity) for value in parsed
    )
    events = causal_replay_event_vectors(normalized)
    return NormalizedReplayEpisode(
        identity,
        replay_sha,
        collection_sha,
        base_sha,
        normalized,
        ticks,
        events,
        tuple(value[2].sha256 for value in parsed),
    )


@dataclass(frozen=True, slots=True)
class LegacyGeometryEpisode:
    identity: str
    seed: int
    rollout_mode: str
    records: tuple[GeometryExample, ...]
    source_ticks: tuple[int, ...]
    shot_indices: tuple[int, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "identity": self.identity,
            "seed": self.seed,
            "rollout_mode": self.rollout_mode,
            "source_ticks": list(self.source_ticks),
            "shot_indices": list(self.shot_indices),
            "records": [value.sha256 for value in self.records],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class LegacyGeometryCollection:
    path: Path
    sha256: str
    dataset_sha256: str
    candidate_set_sha256: str
    candidate_count: int
    base_checkpoint_sha256: str
    metadata: Mapping[str, Any]
    episodes: tuple[LegacyGeometryEpisode, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "format": CAMPAIGN_DATA_FORMAT,
            "kind": "legacy-r3e-geometry-collection",
            "path": str(self.path),
            "sha256": self.sha256,
            "dataset_sha256": self.dataset_sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "candidate_count": self.candidate_count,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "metadata": dict(self.metadata),
            "episodes": [value.sha256 for value in self.episodes],
        }

    @property
    def manifest_sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def _tensor_array(
    value: object, dtype: np.dtype[Any], name: str
) -> np.ndarray:
    if not isinstance(value, Tensor):
        raise ValueError(f"legacy geometry {name} tensor is missing")
    result = value.detach().cpu().numpy()
    if result.dtype != dtype:
        raise ValueError(f"legacy geometry {name} tensor dtype is invalid")
    return np.array(result, dtype=dtype, order="C", copy=True)


def _plain_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"legacy geometry {name} must be an integer")
    return value


def _geometry_episode_groups(
    dataset: GeometryDataset,
    *,
    collection_sha256: str,
) -> tuple[LegacyGeometryEpisode, ...]:
    grouped: dict[int, list[tuple[int, int, str, GeometryExample]]] = {}
    for example in dataset:
        match = _GEOMETRY_IDENTITY.fullmatch(example.episode_identity)
        if match is None:
            raise ValueError("legacy geometry episode identity is unsupported")
        mode = match.group(1)
        seed = int(match.group(2), 16)
        tick = int(match.group(3))
        shot = int(match.group(4))
        if shot < 1 or int(example.observation.source_tick[0]) != tick:
            raise ValueError("legacy geometry chronology is malformed")
        grouped.setdefault(seed, []).append((tick, shot, mode, example))
    episodes: list[LegacyGeometryEpisode] = []
    for seed, records in sorted(grouped.items()):
        records.sort(key=lambda value: (value[0], value[1]))
        if (
            len({(value[0], value[1]) for value in records}) != len(records)
            or len({value[2] for value in records}) != 1
            or any(
                right[0] <= left[0]
                for left, right in zip(records, records[1:])
            )
        ):
            raise ValueError("legacy geometry seed has ambiguous chronology")
        mode = records[0][2]
        identity = (
            f"sequence-replay:legacy-r3e:{collection_sha256}:{seed:08x}"
        )
        episodes.append(
            LegacyGeometryEpisode(
                identity,
                seed,
                mode,
                tuple(value[3] for value in records),
                tuple(value[0] for value in records),
                tuple(value[1] for value in records),
            )
        )
    return tuple(episodes)


def load_legacy_geometry_collection(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_base_checkpoint_sha256: str,
    expected_candidate_set_sha256: str,
    expected_source_identity_sha256: str | None = None,
    expected_runtime_sha256: str | None = None,
    expected_teacher_sha256: str | None = None,
) -> LegacyGeometryCollection:
    """Load the frozen R3e winner collection without importing its benchmark."""

    expected = _sha256(expected_sha256, "legacy geometry collection")
    base_sha = _sha256(
        expected_base_checkpoint_sha256, "legacy geometry base checkpoint"
    )
    candidate_sha = _sha256(
        expected_candidate_set_sha256, "legacy geometry candidate set"
    )
    optional_bindings = {
        "source_identity_sha256": _optional_sha256(
            expected_source_identity_sha256, "legacy geometry source identity"
        ),
        "runtime_sha256": _optional_sha256(
            expected_runtime_sha256, "legacy geometry runtime"
        ),
        "teacher_sha256": _optional_sha256(
            expected_teacher_sha256, "legacy geometry teacher"
        ),
    }
    resolved, raw_bytes, observed = _file_bytes(path)
    if observed != expected:
        raise ValueError("legacy geometry collection SHA-256 mismatch")
    payload = torch.load(
        io.BytesIO(raw_bytes), map_location="cpu", weights_only=True
    )
    required = {
        "format",
        "schema_sha256",
        "candidate_set_sha256",
        "candidate_count",
        "dataset_sha256",
        "metadata",
        "examples",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("legacy geometry collection fields are malformed")
    if (
        payload["format"] != LEGACY_GEOMETRY_COLLECTION_FORMAT
        or payload["schema_sha256"] != TEACHER_V1.sha256
        or payload["candidate_set_sha256"] != candidate_sha
        or not isinstance(payload["examples"], list)
    ):
        raise ValueError("legacy geometry collection schema is unsupported")
    candidate_count = _plain_int(
        payload["candidate_count"], "candidate count"
    )
    if candidate_count < 2:
        raise ValueError("legacy geometry candidate count is invalid")
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise ValueError("legacy geometry metadata is malformed")
    if (
        metadata.get("development_only") is not True
        or metadata.get("canonical_r3_evidence") is not False
        or metadata.get("sealed_test_material_used") is not False
        or metadata.get("learner") != "winner_classifier"
        or metadata.get("base_policy_sha256") != base_sha
        or metadata.get("candidate_vocabulary_sha256") != candidate_sha
    ):
        raise ValueError("legacy geometry collection is not safely bound")
    for name, value in optional_bindings.items():
        if value is not None and metadata.get(name) != value:
            raise ValueError(f"legacy geometry {name} binding differs")

    required_example = {
        "episode_identity",
        "provenance_sha256",
        "candidate_set_sha256",
        "source_index",
        "destination_index",
        "candidate_index",
        "candidate_count",
        "improved_over_incumbent",
        "global_features",
        "body_features",
        "body_mask",
        "source_tick",
        "health_flags",
    }
    examples: list[GeometryExample] = []
    for raw in payload["examples"]:
        if not isinstance(raw, dict) or set(raw) != required_example:
            raise ValueError("legacy geometry example is malformed")
        if type(raw["improved_over_incumbent"]) is not bool:
            raise ValueError("legacy geometry improvement label is malformed")
        encoded = EncodedBatch(
            _tensor_array(
                raw["global_features"], np.dtype(np.float32), "global"
            ),
            _tensor_array(
                raw["body_features"], np.dtype(np.float32), "body"
            ),
            _tensor_array(raw["body_mask"], np.dtype(np.bool_), "mask"),
            _tensor_array(raw["source_tick"], np.dtype(np.uint64), "tick"),
            _tensor_array(
                raw["health_flags"], np.dtype(np.uint32), "health"
            ),
            TEACHER_V1,
        )
        examples.append(
            GeometryExample(
                str(raw["episode_identity"]),
                str(raw["provenance_sha256"]),
                str(raw["candidate_set_sha256"]),
                encoded,
                _plain_int(raw["source_index"], "source index"),
                _plain_int(raw["destination_index"], "destination index"),
                _plain_int(raw["candidate_index"], "candidate index"),
                _plain_int(raw["candidate_count"], "example candidate count"),
                raw["improved_over_incumbent"],
            )
        )
    dataset = GeometryDataset(examples)
    dataset_sha = _sha256(
        str(payload["dataset_sha256"]), "legacy geometry dataset"
    )
    if (
        dataset.sha256 != dataset_sha
        or dataset.candidate_set_sha256 != candidate_sha
        or dataset.candidate_count != candidate_count
    ):
        raise ValueError("legacy geometry reconstructed identity differs")
    episodes = _geometry_episode_groups(dataset, collection_sha256=observed)
    raw_seeds = metadata.get("seeds")
    if (
        not isinstance(raw_seeds, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in raw_seeds
        )
        or len(set(raw_seeds)) != len(raw_seeds)
        or set(raw_seeds) != {value.seed for value in episodes}
    ):
        raise ValueError("legacy geometry metadata seed binding differs")
    canonical_metadata = json.loads(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    )
    return LegacyGeometryCollection(
        resolved,
        observed,
        dataset_sha,
        candidate_sha,
        candidate_count,
        base_sha,
        canonical_metadata,
        episodes,
    )


@dataclass(frozen=True, slots=True)
class IdentityBoundSequenceBatch:
    batch: SequenceReplayBatch
    episode_identities: tuple[str, ...]
    source_kind: str
    source_sha256: str
    base_checkpoint_sha256: str
    geometry_checkpoint_sha256: str | None = None
    extra_manifest: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source_kind, str)
            or not self.source_kind
            or not self.episode_identities
            or len(self.episode_identities) != self.batch.batch_size
            or len(set(self.episode_identities)) != len(self.episode_identities)
        ):
            raise ValueError("identity-bound batch episode identities are invalid")
        _sha256(self.source_sha256, "campaign batch source")
        _sha256(self.base_checkpoint_sha256, "campaign batch base checkpoint")
        _optional_sha256(
            self.geometry_checkpoint_sha256,
            "campaign batch geometry checkpoint",
        )
        try:
            json.dumps(
                {} if self.extra_manifest is None else dict(self.extra_manifest),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("campaign batch extra manifest is not JSON") from exc

    def manifest(self) -> dict[str, object]:
        tensors = {
            "global_features": _tensor_identity(self.batch.global_features),
            "body_features": _tensor_identity(self.batch.body_features),
            "body_mask": _tensor_identity(self.batch.body_mask),
            "event_features": _tensor_identity(self.batch.event_features),
            "valid_mask": _tensor_identity(self.batch.targets.valid_mask),
            "act_index": _tensor_identity(self.batch.targets.act_index),
            "pair_weight": _tensor_identity(
                self.batch.targets.pair_weight
                if self.batch.targets.pair_weight is not None
                else torch.zeros_like(
                    self.batch.targets.valid_mask, dtype=torch.float32
                )
            ),
        }
        if self.batch.base_geometry_logits is not None:
            tensors["base_geometry_logits"] = _tensor_identity(
                self.batch.base_geometry_logits
            )
        if self.batch.targets.geometry_index is not None:
            tensors["geometry_index"] = _tensor_identity(
                self.batch.targets.geometry_index
            )
        return {
            "format": CAMPAIGN_DATA_FORMAT,
            "kind": "identity-bound-sequence-batch",
            "source_kind": self.source_kind,
            "source_sha256": self.source_sha256,
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "geometry_checkpoint_sha256": self.geometry_checkpoint_sha256,
            "episode_identities": list(self.episode_identities),
            "tensors": tensors,
            "extra": {} if self.extra_manifest is None else dict(self.extra_manifest),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def batch_normalized_replay_episode(
    episode: NormalizedReplayEpisode,
    *,
    inferred_pair_weight: float = 0.25,
    device: torch.device | str = "cpu",
) -> IdentityBoundSequenceBatch:
    if not isinstance(episode, NormalizedReplayEpisode):
        raise TypeError("replay batch requires a normalized replay episode")
    if (
        isinstance(inferred_pair_weight, bool)
        or not isinstance(inferred_pair_weight, (int, float))
        or not math.isfinite(float(inferred_pair_weight))
        or not 0.0 <= float(inferred_pair_weight) <= 1.0
    ):
        raise ValueError("inferred pair weight must be in [0, 1]")
    pair_weight = torch.full(
        (len(episode.examples),),
        float(inferred_pair_weight),
        dtype=episode.event_features.dtype,
    )
    batch = batch_steering_example_sequences(
        (episode.examples,),
        event_sequences=(episode.event_features,),
        pair_weight_sequences=(pair_weight,),
        device=device,
    )
    return IdentityBoundSequenceBatch(
        batch,
        (episode.identity,),
        "trusted-replay",
        episode.sha256,
        episode.base_checkpoint_sha256,
        extra_manifest={
            "replay_sha256": episode.replay_sha256,
            "collection_sha256": episode.collection_sha256,
            "destination_semantics": (
                "nearest-visible-same-color-peer-inference-v1"
            ),
            "inferred_pair_weight": float(inferred_pair_weight),
        },
    )


def _geometry_steering_examples(
    episode: LegacyGeometryEpisode,
    pointer_spec: PointerActionSpec,
) -> tuple[SteeringExample, ...]:
    intent = _INTENTS.index(SteeringIntent.STEER_MATCH)
    return tuple(
        SteeringExample(
            episode.identity,
            record.provenance_sha256,
            record.observation,
            record.source_index,
            record.destination_index,
            1,
            0,
            intent,
            pointer_spec,
        )
        for record in episode.records
    )


def batch_legacy_geometry_episodes(
    collection: LegacyGeometryCollection,
    *,
    base_geometry_logits: Mapping[int, Tensor],
    geometry_checkpoint_sha256: str,
    candidate_masks: Mapping[int, Tensor] | None = None,
    pointer_spec: PointerActionSpec | None = None,
    device: torch.device | str = "cpu",
) -> IdentityBoundSequenceBatch:
    """Create geometry-only lanes; policy labels are present but weight zero."""

    if not isinstance(collection, LegacyGeometryCollection):
        raise TypeError("legacy geometry batch requires a verified collection")
    geometry_sha = _sha256(
        geometry_checkpoint_sha256, "frozen geometry checkpoint"
    )
    logits_by_seed = dict(base_geometry_logits)
    masks_by_seed = None if candidate_masks is None else dict(candidate_masks)
    expected_seeds = {value.seed for value in collection.episodes}
    if set(logits_by_seed) != expected_seeds or (
        masks_by_seed is not None and set(masks_by_seed) != expected_seeds
    ):
        raise ValueError("legacy geometry side inputs must bind every seed exactly")
    resolved_pointer = pointer_spec or PointerActionSpec()
    steering_groups = tuple(
        _geometry_steering_examples(episode, resolved_pointer)
        for episode in collection.episodes
    )
    event_groups = tuple(
        causal_replay_event_vectors(group) for group in steering_groups
    )
    zero_pair = tuple(
        torch.zeros(len(group), dtype=event_groups[index].dtype)
        for index, group in enumerate(steering_groups)
    )
    batch = batch_steering_example_sequences(
        steering_groups,
        event_sequences=event_groups,
        pair_weight_sequences=zero_pair,
        device=device,
    )
    target = batch.global_features.device
    dtype = batch.global_features.dtype
    time, lanes = batch.targets.valid_mask.shape
    count = collection.candidate_count
    base_logits = torch.zeros(time, lanes, count, dtype=dtype, device=target)
    candidate_mask = torch.zeros(
        time, lanes, count, dtype=torch.bool, device=target
    )
    source = torch.zeros(time, lanes, dtype=torch.long, device=target)
    destination = torch.zeros_like(source)
    geometry_index = torch.zeros_like(source)
    geometry_weight = torch.zeros(time, lanes, dtype=dtype, device=target)
    apply_target = torch.zeros_like(geometry_weight)
    pair_mask = torch.zeros_like(batch.targets.valid_mask)
    availability_semantics = (
        "explicit-per-state-mask"
        if masks_by_seed is not None
        else "legacy-winner-collection-all-slots-assumed"
    )
    for lane, episode in enumerate(collection.episodes):
        length = len(episode.records)
        logits = logits_by_seed[episode.seed]
        if (
            logits.shape != (length, count)
            or not logits.is_floating_point()
            or not bool(torch.isfinite(logits).all())
        ):
            raise ValueError("legacy geometry base logits are malformed")
        base_logits[:length, lane].copy_(logits.to(target, dtype=dtype))
        if masks_by_seed is None:
            candidate_mask[:length, lane] = True
        else:
            mask = masks_by_seed[episode.seed]
            if mask.shape != (length, count) or mask.dtype != torch.bool:
                raise ValueError("legacy geometry candidate mask is malformed")
            candidate_mask[:length, lane].copy_(mask.to(target))
        for step, record in enumerate(episode.records):
            source[step, lane] = record.source_index
            destination[step, lane] = record.destination_index
            geometry_index[step, lane] = record.candidate_index
            geometry_weight[step, lane] = 1.0
            apply_target[step, lane] = float(record.improved_over_incumbent)
            pair_mask[step, lane] = True
            if not bool(candidate_mask[step, lane, record.candidate_index]):
                raise ValueError("legacy geometry winner is unavailable")
    targets = replace(
        batch.targets,
        policy_weight=torch.zeros_like(geometry_weight),
        pair_weight=torch.zeros_like(geometry_weight),
        geometry_index=geometry_index,
        geometry_weight=geometry_weight,
        geometry_apply_target=apply_target,
    )
    prepared = replace(
        batch,
        targets=targets,
        base_geometry_logits=base_logits,
        geometry_source_index=source,
        geometry_destination_index=destination,
        geometry_pair_mask=pair_mask,
        geometry_candidate_mask=candidate_mask,
    )
    return IdentityBoundSequenceBatch(
        prepared,
        tuple(value.identity for value in collection.episodes),
        "legacy-r3e-geometry",
        collection.manifest_sha256,
        collection.base_checkpoint_sha256,
        geometry_sha,
        {
            "collection_sha256": collection.sha256,
            "dataset_sha256": collection.dataset_sha256,
            "candidate_set_sha256": collection.candidate_set_sha256,
            "candidate_count": count,
            "availability_semantics": availability_semantics,
            "policy_supervision_weight": 0.0,
        },
    )


@dataclass(frozen=True, slots=True)
class TBPTTWindow:
    batch: SequenceReplayBatch
    parent_sha256: str
    episode_identity: str
    lane: int
    source_start: int
    train_start: int
    source_stop: int
    burn_in_steps: int
    training_steps: int

    def manifest(self) -> dict[str, object]:
        return {
            "format": CAMPAIGN_DATA_FORMAT,
            "kind": "deterministic-tbptt-window",
            "parent_sha256": self.parent_sha256,
            "episode_identity": self.episode_identity,
            "lane": self.lane,
            "source_start": self.source_start,
            "train_start": self.train_start,
            "source_stop": self.source_stop,
            "burn_in_steps": self.burn_in_steps,
            "training_steps": self.training_steps,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def _lane_time_slice(
    batch: SequenceReplayBatch, lane: int, start: int, stop: int
) -> SequenceReplayBatch:
    def sliced(value: Tensor | None) -> Tensor | None:
        return None if value is None else value[start:stop, lane : lane + 1]

    target_values = {
        field.name: (
            sliced(value)
            if isinstance(value := getattr(batch.targets, field.name), Tensor)
            else value
        )
        for field in fields(batch.targets)
    }
    targets = replace(batch.targets, **target_values)
    return SequenceReplayBatch(
        sliced(batch.global_features),
        sliced(batch.body_features),
        sliced(batch.body_mask),
        sliced(batch.event_features),
        sliced(batch.reset_before),
        targets,
        sliced(batch.base_geometry_logits),
        sliced(batch.geometry_source_index),
        sliced(batch.geometry_destination_index),
        sliced(batch.geometry_pair_mask),
        sliced(batch.geometry_candidate_mask),
    )


def _mask_burn_in(
    batch: SequenceReplayBatch, burn_in_steps: int
) -> SequenceReplayBatch:
    train_mask = batch.targets.valid_mask.clone()
    train_mask[:burn_in_steps] = False
    dtype = batch.global_features.dtype

    def weighted(value: Tensor | None, default: Tensor) -> Tensor:
        source = default if value is None else value
        return source * train_mask.to(source.dtype)

    targets = batch.targets
    policy_default = targets.valid_mask.to(dtype)
    pair_default = (
        targets.valid_mask & (targets.act_index == 1)
    ).to(dtype)
    geometry_default = targets.valid_mask.to(dtype)
    values: dict[str, Any] = {
        "policy_weight": weighted(targets.policy_weight, policy_default),
        "pair_weight": weighted(targets.pair_weight, pair_default),
    }
    if targets.geometry_index is not None:
        values["geometry_weight"] = weighted(
            targets.geometry_weight, geometry_default
        )
    for target_name, mask_name in (
        ("return_target", "value_mask"),
        ("viability_target", "viability_mask"),
        ("outcome_target", "outcome_mask"),
    ):
        if getattr(targets, target_name) is not None:
            existing = getattr(targets, mask_name)
            values[mask_name] = (
                targets.valid_mask if existing is None else existing
            ) & train_mask
    reset = batch.reset_before.clone()
    reset[0] = True
    return replace(
        batch, reset_before=reset, targets=replace(targets, **values)
    )


def deterministic_tbptt_windows(
    bound: IdentityBoundSequenceBatch,
    *,
    burn_in_steps: int,
    unroll_steps: int,
    seed: int,
    epoch: int = 0,
    maximum_windows: int | None = None,
) -> tuple[TBPTTWindow, ...]:
    """Cover nonoverlapping train segments in a reproducible hashed order."""

    if not isinstance(bound, IdentityBoundSequenceBatch):
        raise TypeError("TBPTT windows require an identity-bound batch")
    integers = (burn_in_steps, unroll_steps, seed, epoch)
    if (
        type(burn_in_steps) is not int
        or burn_in_steps < 0
        or type(unroll_steps) is not int
        or unroll_steps < 1
        or type(seed) is not int
        or type(epoch) is not int
        or epoch < 0
        or maximum_windows is not None
        and (type(maximum_windows) is not int or maximum_windows < 1)
    ):
        raise ValueError(f"invalid TBPTT settings: {integers}")
    valid = bound.batch.targets.valid_mask
    if valid.ndim != 2 or valid.shape[1] != len(bound.episode_identities):
        raise ValueError("TBPTT batch lanes and episode identities disagree")
    parent_sha256 = bound.sha256
    candidates: list[tuple[str, int, int, int, int]] = []
    for lane, identity in enumerate(bound.episode_identities):
        length = int(valid[:, lane].sum())
        if length < 1 or not bool(valid[:length, lane].all()) or bool(
            valid[length:, lane].any()
        ):
            raise ValueError("TBPTT valid timesteps must form one prefix per lane")
        for train_start in range(0, length, unroll_steps):
            start = max(0, train_start - burn_in_steps)
            stop = min(length, train_start + unroll_steps)
            ordering = _canonical_sha256(
                {
                    "seed": seed,
                    "epoch": epoch,
                    "parent": parent_sha256,
                    "episode": identity,
                    "lane": lane,
                    "train_start": train_start,
                }
            )
            candidates.append((ordering, lane, start, train_start, stop))
    candidates.sort()
    if maximum_windows is not None:
        candidates = candidates[:maximum_windows]
    output: list[TBPTTWindow] = []
    for _, lane, start, train_start, stop in candidates:
        burn = train_start - start
        sliced = _mask_burn_in(
            _lane_time_slice(bound.batch, lane, start, stop), burn
        )
        output.append(
            TBPTTWindow(
                sliced,
                parent_sha256,
                bound.episode_identities[lane],
                lane,
                start,
                train_start,
                stop,
                burn,
                stop - train_start,
            )
        )
    return tuple(output)


def _metric(
    numerator: Tensor | float,
    weight: Tensor | float,
    count: int,
) -> dict[str, float | int]:
    total = float(numerator)
    denominator = float(weight)
    return {
        "numerator": total,
        "weight": denominator,
        "count": int(count),
        "value": total / denominator if denominator > 0.0 else 0.0,
    }


def _weighted_metric(
    value: Tensor, weight: Tensor
) -> dict[str, float | int]:
    active = weight > 0
    return _metric(
        (value * weight).sum().detach(),
        weight.sum().detach(),
        int(active.sum()),
    )


@torch.no_grad()
def offline_sequence_metrics(
    output: SequenceReplayOutput,
    targets: SequenceReplayTargets,
) -> dict[str, object]:
    """Return mergeable weighted offline diagnostics, never a success verdict."""

    shape = targets.valid_mask.shape
    if output.act_logits.shape[:2] != shape:
        raise ValueError("offline output and target sequence shapes disagree")
    dtype = output.act_logits.dtype
    policy_weight = (
        targets.valid_mask.to(dtype)
        if targets.policy_weight is None
        else targets.policy_weight
    )
    policy_weight = policy_weight * targets.valid_mask.to(dtype)
    act_prediction = output.act_logits.argmax(dim=-1)
    metrics: dict[str, dict[str, float | int]] = {
        "act_accuracy": _weighted_metric(
            (act_prediction == targets.act_index).to(dtype), policy_weight
        )
    }
    wait_weight = policy_weight * (targets.act_index == 0)
    shot_weight = policy_weight * (targets.act_index == 1)
    metrics["wait_recall"] = _weighted_metric(
        (act_prediction == 0).to(dtype), wait_weight
    )
    metrics["shot_recall"] = _weighted_metric(
        (act_prediction == 1).to(dtype), shot_weight
    )
    metrics["wait_duration_accuracy"] = _weighted_metric(
        (output.wait_logits.argmax(dim=-1) == targets.wait_index).to(dtype),
        wait_weight,
    )

    pair_weight = (
        (targets.valid_mask & (targets.act_index == 1)).to(dtype)
        if targets.pair_weight is None
        else targets.pair_weight
    )
    pair_weight = pair_weight * (
        targets.valid_mask & (targets.act_index == 1)
    ).to(dtype)
    pair_prediction = output.pair_logits.flatten(2).argmax(dim=-1)
    body_count = output.pair_logits.shape[-1]
    pair_target = targets.source_index * body_count + targets.destination_index
    metrics["pair_accuracy"] = _weighted_metric(
        (pair_prediction == pair_target).to(dtype), pair_weight
    )
    rows = pair_weight > 0
    if bool(rows.any()):
        positions = rows.nonzero(as_tuple=True)
        selected = (
            *positions,
            targets.source_index[rows],
            targets.destination_index[rows],
        )
        selected_weight = pair_weight[rows]
        for name, logits, labels in (
            ("kind_accuracy", output.kind_logits, targets.kind_index),
            (
                "template_accuracy",
                output.template_logits,
                targets.template_index,
            ),
            ("intent_accuracy", output.intent_logits, targets.intent_index),
        ):
            metrics[name] = _weighted_metric(
                (logits[selected].argmax(dim=-1) == labels[rows]).to(dtype),
                selected_weight,
            )
    else:
        for name in ("kind_accuracy", "template_accuracy", "intent_accuracy"):
            metrics[name] = _metric(0.0, 0.0, 0)

    if (
        output.geometry_logits is not None
        and targets.geometry_index is not None
    ):
        geometry_weight = (
            targets.valid_mask.to(dtype)
            if targets.geometry_weight is None
            else targets.geometry_weight
        )
        geometry_weight = geometry_weight * targets.valid_mask.to(dtype)
        metrics["geometry_accuracy"] = _weighted_metric(
            (
                output.geometry_logits.argmax(dim=-1)
                == targets.geometry_index
            ).to(dtype),
            geometry_weight,
        )
        if (
            output.geometry_apply_mask is not None
            and targets.geometry_apply_target is not None
        ):
            gate_target = targets.geometry_apply_target >= 0.5
            metrics["geometry_gate_accuracy"] = _weighted_metric(
                (output.geometry_apply_mask == gate_target).to(dtype),
                geometry_weight,
            )
            metrics["geometry_gate_recall"] = _weighted_metric(
                output.geometry_apply_mask.to(dtype),
                geometry_weight * gate_target.to(dtype),
            )
    if targets.return_target is not None:
        value_mask = (
            targets.valid_mask
            if targets.value_mask is None
            else targets.valid_mask & targets.value_mask
        )
        median = output.return_quantiles[
            ..., output.return_quantiles.shape[-1] // 2
        ]
        metrics["value_median_absolute_error"] = _weighted_metric(
            (median - targets.return_target).abs(),
            value_mask.to(dtype),
        )
    if targets.viability_target is not None:
        viability_mask = (
            targets.valid_mask
            if targets.viability_mask is None
            else targets.valid_mask & targets.viability_mask
        )
        expanded = viability_mask[..., None].expand_as(
            output.viability_logits
        )
        metrics["viability_accuracy"] = _weighted_metric(
            (
                (output.viability_logits >= 0)
                == (targets.viability_target >= 0.5)
            ).to(dtype),
            expanded.to(dtype),
        )
    if targets.outcome_target is not None:
        outcome_mask = (
            targets.valid_mask
            if targets.outcome_mask is None
            else targets.valid_mask & targets.outcome_mask
        )
        expanded = outcome_mask[..., None].expand_as(output.outcome_values)
        metrics["outcome_absolute_error"] = _weighted_metric(
            (output.outcome_values - targets.outcome_target).abs(),
            expanded.to(dtype),
        )
    return {
        "format": "irisu-sequence-replay-offline-metrics-v1",
        "warning": "offline diagnostics are not rollout evidence",
        "metrics": metrics,
    }


def merge_offline_metrics(
    reports: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    supplied = tuple(reports)
    if not supplied:
        raise ValueError("at least one offline metric report is required")
    totals: dict[str, list[float]] = {}
    counts: dict[str, int] = {}
    for report in supplied:
        if report.get("format") != "irisu-sequence-replay-offline-metrics-v1":
            raise ValueError("offline metric report format is unsupported")
        metrics = report.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("offline metric report is malformed")
        for name, raw in metrics.items():
            if not isinstance(name, str) or not isinstance(raw, Mapping):
                raise ValueError("offline metric entry is malformed")
            numerator = float(raw["numerator"])
            weight = float(raw["weight"])
            count = int(raw["count"])
            if (
                not math.isfinite(numerator)
                or not math.isfinite(weight)
                or weight < 0.0
                or count < 0
            ):
                raise ValueError("offline metric values are invalid")
            aggregate = totals.setdefault(name, [0.0, 0.0])
            aggregate[0] += numerator
            aggregate[1] += weight
            counts[name] = counts.get(name, 0) + count
    return {
        "format": "irisu-sequence-replay-offline-metrics-v1",
        "warning": "offline diagnostics are not rollout evidence",
        "reports": len(supplied),
        "metrics": {
            name: _metric(values[0], values[1], counts[name])
            for name, values in sorted(totals.items())
        },
    }


__all__ = [
    "CAMPAIGN_DATA_FORMAT",
    "LEGACY_GEOMETRY_COLLECTION_FORMAT",
    "IdentityBoundSequenceBatch",
    "LegacyGeometryCollection",
    "LegacyGeometryEpisode",
    "NormalizedReplayEpisode",
    "TBPTTWindow",
    "batch_legacy_geometry_episodes",
    "batch_normalized_replay_episode",
    "causal_replay_event_vectors",
    "deterministic_tbptt_windows",
    "load_legacy_geometry_collection",
    "merge_offline_metrics",
    "normalize_trusted_replay_examples",
    "offline_sequence_metrics",
]
