#!/usr/bin/env python3
"""Development-only R3e geometry-search distillation and plateau benchmark.

The frozen R3d v5 policy continues to choose whether to act and which directed
pair to pursue.  At sampled shots an injectable full-runway teacher
transactionally searches geometry and drives the collection rollout with its
winner, making later labels teacher-policy-visited rather than offline v5
states.  R3e distills those fixed slots and compares the shielded selector with
unchanged v5 on disjoint development seeds.  This file never reads sealed
inputs or writes canonical R3 run storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from irisu_env import IrisuEnv
from irisu_pointer.geometry_learning import (
    GeometryDataset,
    GeometryExample,
    GeometryModelConfig,
    GeometrySelectorModel,
    GeometryTrainingReport,
    geometry_example,
    train_geometry_selector,
)
from irisu_pointer.geometry_policy import (
    GeometryPolicyConfig,
    SafeguardedGeometryPolicy,
    geometry_candidate_vocabulary_sha256,
)
from irisu_pointer.geometry_ranking import (
    GeometryRankingDataset,
    GeometryRankingExample,
    GeometryRankingTrainingReport,
    geometry_ranking_example,
    train_geometry_ranker,
)
from irisu_pointer.geometry_search import (
    GeometrySearchConfig,
)
from irisu_pointer.runway_search import (
    RunwayGeometrySearch,
    RunwaySearchConfig,
)
from irisu_pointer.steering import SteeringDecision
from irisu_pointer.steering_checkpoint import (
    load_goal_conditioned_steering_policy,
)
from irisu_rl.encoding import EncodedBatch
from irisu_rl.schema import TEACHER_V1


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = Path(__file__).resolve().parent
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))
import rl_r3d_steering as r3d  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/rl/experiments/r3e-sustainable-v1.toml"
TRUSTED_PORTABLE = (
    ROOT
    / "artifacts/r3/runtime/main-0c48dba-20260723/"
    "portable-build/libirisu_clone.so"
)
_COLLECTION_FORMAT = "irisu-r3e-geometry-collection-v1"
_RANKING_COLLECTION_FORMAT = "irisu-r3e-ranking-collection-v1"
_CHECKPOINT_FORMAT = "irisu-r3e-geometry-selector-checkpoint-v1"
_REPORT_FORMAT = "irisu-r3e-sustainable-development-run-v1"


def _derive_seeds(label: str, count: int = 32) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(
            hashlib.sha256(f"{label}:{index}".encode()).digest()[:4],
            "big",
        )
        for index in range(count)
    )


COLLECTION_SEEDS = _derive_seeds("irisu-r3e-policy-visited-collection-v1")
DEVELOPMENT_SEEDS = _derive_seeds("irisu-r3e-fixed-disjoint-development-v1")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _canonical_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = {} if value is None else dict(value)
    try:
        return json.loads(_canonical_bytes(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError("R3e metadata must be canonical JSON") from exc


def _development_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    supplied = _canonical_mapping(value)
    required = {
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
    }
    for name, expected in required.items():
        if name in supplied and supplied[name] is not expected:
            raise ValueError(f"R3e metadata cannot change {name}")
    return {**supplied, **required}


def _verify_development_metadata(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("R3e artifact metadata is malformed")
    metadata = _canonical_mapping(value)
    if (
        metadata.get("development_only") is not True
        or metadata.get("canonical_r3_evidence") is not False
        or metadata.get("sealed_test_material_used") is not False
    ):
        raise ValueError("R3e artifact is not development-only")
    return metadata


def _require_bound_metadata(
    metadata: Mapping[str, Any],
    expected: Mapping[str, object],
    name: str,
) -> None:
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError(f"R3e {name} metadata identity mismatch")


def _atomic_torch_save(
    path: Path, payload: Mapping[str, object], *, overwrite: bool
) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace R3e artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return r3d._file_sha256(path)


def _atomic_write_json(path: Path, value: Mapping[str, object], *, overwrite: bool) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace R3e report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return r3d._file_sha256(path)


def _source_identity(config_path: Path) -> dict[str, object]:
    files = (
        Path(__file__).resolve(),
        Path(r3d.__file__).resolve(),
        config_path,
        *sorted((ROOT / "python/irisu_pointer").glob("*.py")),
        ROOT / "python/irisu_rl/actions.py",
        ROOT / "python/irisu_rl/encoding.py",
        ROOT / "python/irisu_rl/schema.py",
        ROOT / "pyproject.toml",
    )
    manifest = {
        "schema": "irisu-r3e-sustainable-source-v1",
        "git_revision": r3d._source_revision(),
        "files": {
            str(path.relative_to(ROOT)): r3d._file_sha256(path)
            for path in files
        },
    }
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _require_source_identity(
    expected: Mapping[str, object], config_path: Path
) -> None:
    if _source_identity(config_path) != dict(expected):
        raise RuntimeError("R3e source identity changed during the benchmark")


def _encoded_payload(example: GeometryExample) -> dict[str, object]:
    encoded = example.observation
    return {
        "episode_identity": example.episode_identity,
        "provenance_sha256": example.provenance_sha256,
        "candidate_set_sha256": example.candidate_set_sha256,
        "source_index": example.source_index,
        "destination_index": example.destination_index,
        "candidate_index": example.candidate_index,
        "candidate_count": example.candidate_count,
        "improved_over_incumbent": example.improved_over_incumbent,
        "global_features": torch.from_numpy(encoded.global_features.copy()),
        "body_features": torch.from_numpy(encoded.body_features.copy()),
        "body_mask": torch.from_numpy(encoded.body_mask.copy()),
        "source_tick": torch.from_numpy(encoded.source_tick.copy()),
        "health_flags": torch.from_numpy(encoded.health_flags.copy()),
    }


def _tensor_array(
    value: object, dtype: np.dtype[Any], name: str
) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"R3e {name} tensor is missing")
    result = value.detach().cpu().numpy()
    if result.dtype != dtype:
        raise ValueError(f"R3e {name} tensor dtype is invalid")
    return np.array(result, dtype=dtype, order="C", copy=True)


@dataclass(frozen=True, slots=True)
class GeometryCollectionArtifact:
    path: Path
    sha256: str
    dataset: GeometryDataset
    metadata: Mapping[str, Any]


def save_geometry_collection(
    path: str | Path,
    dataset: GeometryDataset,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> str:
    target = r3d._output_path(Path(path), "R3e collection output", ".pt")
    payload = {
        "format": _COLLECTION_FORMAT,
        "schema_sha256": dataset.schema.sha256,
        "candidate_set_sha256": dataset.candidate_set_sha256,
        "candidate_count": dataset.candidate_count,
        "dataset_sha256": dataset.sha256,
        "metadata": _development_metadata(metadata),
        "examples": [_encoded_payload(example) for example in dataset],
    }
    return _atomic_torch_save(target, payload, overwrite=overwrite)


def load_geometry_collection(
    path: str | Path,
    *,
    expected_sha256: str,
) -> GeometryCollectionArtifact:
    expected = _sha256(expected_sha256, "R3e collection identity")
    snapshot = r3d._snapshot_file(Path(path), "R3e collection input")
    if snapshot.sha256 != expected:
        raise ValueError("R3e collection SHA-256 mismatch")
    payload = torch.load(snapshot.path, map_location="cpu", weights_only=True)
    fields = {
        "format",
        "schema_sha256",
        "candidate_set_sha256",
        "candidate_count",
        "dataset_sha256",
        "metadata",
        "examples",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("R3e collection has unknown or missing fields")
    if (
        payload["format"] != _COLLECTION_FORMAT
        or payload["schema_sha256"] != TEACHER_V1.sha256
        or not isinstance(payload["examples"], list)
    ):
        raise ValueError("R3e collection format or schema is unsupported")
    candidate_identity = _sha256(
        payload["candidate_set_sha256"], "R3e candidate-set identity"
    )
    count = payload["candidate_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("R3e collection candidate count is invalid")
    examples: list[GeometryExample] = []
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
    for raw in payload["examples"]:
        if not isinstance(raw, dict) or set(raw) != required_example:
            raise ValueError("R3e collection example is malformed")
        encoded = EncodedBatch(
            _tensor_array(raw["global_features"], np.dtype(np.float32), "global"),
            _tensor_array(raw["body_features"], np.dtype(np.float32), "body"),
            _tensor_array(raw["body_mask"], np.dtype(np.bool_), "mask"),
            _tensor_array(raw["source_tick"], np.dtype(np.uint64), "tick"),
            _tensor_array(raw["health_flags"], np.dtype(np.uint32), "health"),
            TEACHER_V1,
        )
        examples.append(
            GeometryExample(
                str(raw["episode_identity"]),
                str(raw["provenance_sha256"]),
                str(raw["candidate_set_sha256"]),
                encoded,
                int(raw["source_index"]),
                int(raw["destination_index"]),
                int(raw["candidate_index"]),
                int(raw["candidate_count"]),
                bool(raw["improved_over_incumbent"]),
            )
        )
    dataset = GeometryDataset(examples)
    if (
        dataset.candidate_set_sha256 != candidate_identity
        or dataset.candidate_count != count
        or dataset.sha256 != payload["dataset_sha256"]
    ):
        raise ValueError("R3e reconstructed collection identity differs")
    return GeometryCollectionArtifact(
        snapshot.path,
        snapshot.sha256,
        dataset,
        _verify_development_metadata(payload["metadata"]),
    )


def _ranking_payload(example: GeometryRankingExample) -> dict[str, object]:
    encoded = example.observation
    return {
        "episode_identity": example.episode_identity,
        "provenance_sha256": example.provenance_sha256,
        "search_result_sha256": example.search_result_sha256,
        "candidate_set_sha256": example.candidate_set_sha256,
        "candidate_vocabulary_sha256": (
            example.candidate_vocabulary_sha256
        ),
        "source_index": example.source_index,
        "destination_index": example.destination_index,
        "candidate_count": example.candidate_count,
        "available_mask": torch.from_numpy(example.available_mask.copy()),
        "winner_index": example.winner_index,
        "improved_over_incumbent": example.improved_over_incumbent,
        "preferences": torch.tensor(
            example.preferences, dtype=torch.int64
        ).reshape(-1, 2),
        "outcome_sha256s": list(example.outcome_sha256s),
        "global_features": torch.from_numpy(encoded.global_features.copy()),
        "body_features": torch.from_numpy(encoded.body_features.copy()),
        "body_mask": torch.from_numpy(encoded.body_mask.copy()),
        "source_tick": torch.from_numpy(encoded.source_tick.copy()),
        "health_flags": torch.from_numpy(encoded.health_flags.copy()),
    }


@dataclass(frozen=True, slots=True)
class GeometryRankingCollectionArtifact:
    path: Path
    sha256: str
    dataset: GeometryRankingDataset
    metadata: Mapping[str, Any]


def save_geometry_ranking_collection(
    path: str | Path,
    dataset: GeometryRankingDataset,
    *,
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> str:
    target = r3d._output_path(
        Path(path), "R3e ranking collection output", ".pt"
    )
    payload = {
        "format": _RANKING_COLLECTION_FORMAT,
        "schema_sha256": dataset.schema.sha256,
        "candidate_vocabulary_sha256": (
            dataset.candidate_vocabulary_sha256
        ),
        "candidate_count": dataset.candidate_count,
        "dataset_sha256": dataset.sha256,
        "metadata": _development_metadata(metadata),
        "examples": [_ranking_payload(example) for example in dataset],
    }
    return _atomic_torch_save(target, payload, overwrite=overwrite)


def load_geometry_ranking_collection(
    path: str | Path,
    *,
    expected_sha256: str,
) -> GeometryRankingCollectionArtifact:
    expected = _sha256(expected_sha256, "R3e ranking collection identity")
    snapshot = r3d._snapshot_file(Path(path), "R3e ranking collection input")
    if snapshot.sha256 != expected:
        raise ValueError("R3e ranking collection SHA-256 mismatch")
    payload = torch.load(snapshot.path, map_location="cpu", weights_only=True)
    fields = {
        "format",
        "schema_sha256",
        "candidate_vocabulary_sha256",
        "candidate_count",
        "dataset_sha256",
        "metadata",
        "examples",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("R3e ranking collection has unknown or missing fields")
    if (
        payload["format"] != _RANKING_COLLECTION_FORMAT
        or payload["schema_sha256"] != TEACHER_V1.sha256
        or not isinstance(payload["examples"], list)
    ):
        raise ValueError("R3e ranking collection format or schema is unsupported")
    vocabulary_sha256 = _sha256(
        payload["candidate_vocabulary_sha256"],
        "R3e ranking candidate vocabulary",
    )
    count = payload["candidate_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise ValueError("R3e ranking candidate count is invalid")
    required = {
        "episode_identity",
        "provenance_sha256",
        "search_result_sha256",
        "candidate_set_sha256",
        "candidate_vocabulary_sha256",
        "source_index",
        "destination_index",
        "candidate_count",
        "available_mask",
        "winner_index",
        "improved_over_incumbent",
        "preferences",
        "outcome_sha256s",
        "global_features",
        "body_features",
        "body_mask",
        "source_tick",
        "health_flags",
    }
    examples: list[GeometryRankingExample] = []
    for raw in payload["examples"]:
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("R3e ranking collection example is malformed")
        encoded = EncodedBatch(
            _tensor_array(raw["global_features"], np.dtype(np.float32), "global"),
            _tensor_array(raw["body_features"], np.dtype(np.float32), "body"),
            _tensor_array(raw["body_mask"], np.dtype(np.bool_), "mask"),
            _tensor_array(raw["source_tick"], np.dtype(np.uint64), "tick"),
            _tensor_array(raw["health_flags"], np.dtype(np.uint32), "health"),
            TEACHER_V1,
        )
        available = _tensor_array(
            raw["available_mask"], np.dtype(np.bool_), "ranking availability"
        )
        preferences = _tensor_array(
            raw["preferences"], np.dtype(np.int64), "ranking preferences"
        )
        if preferences.ndim != 2 or preferences.shape[1:] != (2,):
            raise ValueError("R3e ranking preference tensor is invalid")
        outcome_sha256s = raw["outcome_sha256s"]
        if not isinstance(outcome_sha256s, list):
            raise ValueError("R3e ranking outcome identities are malformed")
        examples.append(
            GeometryRankingExample(
                episode_identity=str(raw["episode_identity"]),
                provenance_sha256=str(raw["provenance_sha256"]),
                search_result_sha256=str(raw["search_result_sha256"]),
                candidate_set_sha256=str(raw["candidate_set_sha256"]),
                candidate_vocabulary_sha256=str(
                    raw["candidate_vocabulary_sha256"]
                ),
                observation=encoded,
                source_index=int(raw["source_index"]),
                destination_index=int(raw["destination_index"]),
                candidate_count=int(raw["candidate_count"]),
                available_mask=available,
                winner_index=int(raw["winner_index"]),
                improved_over_incumbent=bool(
                    raw["improved_over_incumbent"]
                ),
                preferences=tuple(
                    (int(preferred), int(rejected))
                    for preferred, rejected in preferences.tolist()
                ),
                outcome_sha256s=tuple(str(value) for value in outcome_sha256s),
            )
        )
    dataset = GeometryRankingDataset(examples)
    if (
        dataset.candidate_vocabulary_sha256 != vocabulary_sha256
        or dataset.candidate_count != count
        or dataset.sha256 != payload["dataset_sha256"]
    ):
        raise ValueError("R3e reconstructed ranking identity differs")
    return GeometryRankingCollectionArtifact(
        snapshot.path,
        snapshot.sha256,
        dataset,
        _verify_development_metadata(payload["metadata"]),
    )


def merge_geometry_ranking_datasets(
    datasets: Sequence[GeometryRankingDataset],
) -> GeometryRankingDataset:
    """Merge identity-compatible oracle/learner-visited ranking shards."""

    values = tuple(datasets)
    if not values:
        raise ValueError("R3e ranking merge requires at least one dataset")
    first = values[0]
    if any(
        dataset.schema.sha256 != first.schema.sha256
        or dataset.candidate_vocabulary_sha256
        != first.candidate_vocabulary_sha256
        or dataset.candidate_count != first.candidate_count
        for dataset in values[1:]
    ):
        raise ValueError("R3e ranking merge mixes incompatible identities")
    by_episode: dict[str, str] = {}
    seen: set[str] = set()
    examples: list[GeometryRankingExample] = []
    for dataset in values:
        for example in dataset:
            prior = by_episode.setdefault(
                example.episode_identity, example.sha256
            )
            if prior != example.sha256:
                raise ValueError(
                    "R3e ranking merge found a conflicting episode identity"
                )
            if example.sha256 not in seen:
                seen.add(example.sha256)
                examples.append(example)
    return GeometryRankingDataset(examples)


@dataclass(frozen=True, slots=True)
class GeometryCheckpointArtifact:
    path: Path
    sha256: str
    model: GeometrySelectorModel
    search_identity: Mapping[str, Any]
    metadata: Mapping[str, Any]


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(_canonical_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_geometry_checkpoint(
    path: str | Path,
    model: GeometrySelectorModel,
    *,
    search_identity: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> str:
    target = r3d._output_path(Path(path), "R3e geometry checkpoint output", ".pt")
    payload = {
        "format": _CHECKPOINT_FORMAT,
        "model_manifest": model.manifest(),
        "state_dict_sha256": _state_dict_sha256(model),
        "search_identity": _canonical_mapping(search_identity),
        "metadata": _development_metadata(metadata),
        "state_dict": {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        },
    }
    return _atomic_torch_save(target, payload, overwrite=overwrite)


def load_geometry_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    device: torch.device | str = "cpu",
) -> GeometryCheckpointArtifact:
    expected = _sha256(expected_sha256, "R3e checkpoint identity")
    snapshot = r3d._snapshot_file(Path(path), "R3e geometry checkpoint input")
    if snapshot.sha256 != expected:
        raise ValueError("R3e geometry checkpoint SHA-256 mismatch")
    payload = torch.load(snapshot.path, map_location="cpu", weights_only=True)
    fields = {
        "format",
        "model_manifest",
        "state_dict_sha256",
        "search_identity",
        "metadata",
        "state_dict",
    }
    if not isinstance(payload, dict) or set(payload) != fields:
        raise ValueError("R3e checkpoint has unknown or missing fields")
    manifest = payload["model_manifest"]
    if payload["format"] != _CHECKPOINT_FORMAT or not isinstance(manifest, dict):
        raise ValueError("R3e checkpoint format is unsupported")
    try:
        model = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=int(manifest["candidate_count"]),
            candidate_set_sha256=str(manifest["candidate_set_sha256"]),
            config=GeometryModelConfig(**manifest["config"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("R3e checkpoint model configuration is invalid") from exc
    if model.manifest() != manifest:
        raise ValueError("R3e reconstructed model identity differs")
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise ValueError("R3e checkpoint state dictionary is malformed")
    model.load_state_dict(state_dict, strict=True)
    if _state_dict_sha256(model) != payload["state_dict_sha256"]:
        raise ValueError("R3e checkpoint state identity differs")
    model.to(device)
    model.eval()
    return GeometryCheckpointArtifact(
        snapshot.path,
        snapshot.sha256,
        model,
        _canonical_mapping(payload["search_identity"]),
        _verify_development_metadata(payload["metadata"]),
    )


@dataclass(frozen=True, slots=True)
class CampaignSettings:
    collection_seeds: int
    collection_ticks: int
    query_stride_shots: int
    maximum_search_queries_per_episode: int
    evaluation_seeds: int
    evaluation_horizons: tuple[int, ...]
    training_steps: tuple[int, ...]
    batch_size: int

    def __post_init__(self) -> None:
        counts = (
            self.collection_seeds,
            self.collection_ticks,
            self.query_stride_shots,
            self.maximum_search_queries_per_episode,
            self.evaluation_seeds,
            self.batch_size,
        )
        if any(type(value) is not int or value < 1 for value in counts):
            raise ValueError("R3e campaign counts must be positive integers")
        for name, values in (
            ("evaluation horizons", self.evaluation_horizons),
            ("training steps", self.training_steps),
        ):
            if (
                not values
                or tuple(sorted(set(values))) != values
                or any(type(value) is not int or value < 1 for value in values)
            ):
                raise ValueError(f"R3e {name} must increase uniquely")
        if self.collection_seeds > len(COLLECTION_SEEDS):
            raise ValueError("R3e collection seed count exceeds the fixed suite")
        if self.evaluation_seeds > len(DEVELOPMENT_SEEDS):
            raise ValueError("R3e evaluation seed count exceeds the fixed suite")


def _load_config(snapshot: Any) -> dict[str, Any]:
    value = tomllib.loads(snapshot.path.read_text(encoding="utf-8"))
    required = {
        "version": "r3e-sustainable-v1",
        "status": "development_only_not_canonical_evidence",
        "deployable": False,
        "canonical_r3_evidence": False,
        "sealed_evaluation_allowed": False,
    }
    if any(value.get(name) != expected for name, expected in required.items()):
        raise ValueError("R3e config does not preserve development-only status")
    runtime = (ROOT / value["trusted_runtime"]).resolve(strict=True)
    if runtime != TRUSTED_PORTABLE.resolve(strict=True):
        raise ValueError("R3e config does not bind the trusted portable runtime")
    return value


def _settings(profile: Mapping[str, Any]) -> CampaignSettings:
    return CampaignSettings(
        int(profile["collection_seeds"]),
        int(profile["collection_ticks"]),
        int(profile["query_stride_shots"]),
        int(profile["maximum_search_queries_per_episode"]),
        int(profile["evaluation_seeds"]),
        tuple(int(value) for value in profile["evaluation_horizons"]),
        tuple(int(value) for value in profile["training_steps"]),
        int(profile["batch_size"]),
    )


def _base_policy(
    path: Path, sha256: str, options: Mapping[str, Any]
) -> object:
    return load_goal_conditioned_steering_policy(
        path,
        expected_sha256=sha256,
        cooldown_ticks=int(options["cooldown_ticks"]),
        minimum_pair_closure_sizes=float(
            options["minimum_pair_closure_sizes"]
        ),
        impact_side_sizes=float(options["impact_side_sizes"]),
        impact_below_sizes=float(options["impact_below_sizes"]),
        source_velocity_lead_ticks=float(
            options["source_velocity_lead_ticks"]
        ),
        ticks_per_second=float(options["ticks_per_second"]),
        act_logit_bias=float(options["act_logit_bias"]),
    )


def _piece_pair(
    observation: Mapping[str, Any], decision: SteeringDecision
) -> bool:
    if (
        not decision.is_shot
        or decision.source_body_id is None
        or decision.destination_body_id is None
    ):
        return False
    selected = {
        int(body.get("id", -1)): body
        for body in observation.get("bodies", ())
        if isinstance(body, Mapping)
        and int(body.get("id", -1))
        in {decision.source_body_id, decision.destination_body_id}
    }
    return (
        set(selected)
        == {decision.source_body_id, decision.destination_body_id}
        and all(body.get("kind") == "piece" for body in selected.values())
        and all(
            body.get("shape") in {"circle", "box", "triangle"}
            for body in selected.values()
        )
    )


@dataclass(frozen=True, slots=True)
class CollectionRun:
    dataset: GeometryDataset
    ranking_dataset: GeometryRankingDataset
    episodes: tuple[Any, ...]
    runner: Mapping[str, Any]
    query_report: Mapping[str, Any]


class OracleVisitedCollectionPolicy:
    """Run v5 pair selection, but execute sampled full-runway winners."""

    def __init__(
        self,
        *,
        env: IrisuEnv,
        base_policy: object,
        teacher: Any,
        seed: int,
        episode_ticks: int,
        query_stride_shots: int,
        maximum_search_queries: int,
        source_identity_sha256: str,
        runtime_sha256: str,
        base_policy_sha256: str,
        rollout_mode: str = "oracle_visited",
    ) -> None:
        self.env = env
        self.base_policy = base_policy
        self.teacher = teacher
        self.seed = int(seed)
        self.episode_ticks = int(episode_ticks)
        self.query_stride_shots = int(query_stride_shots)
        self.maximum_search_queries = int(maximum_search_queries)
        if rollout_mode not in {"oracle_visited", "learner_visited"}:
            raise ValueError("R3e collection rollout mode is unsupported")
        self.rollout_mode = rollout_mode
        self.source_identity_sha256 = _sha256(
            source_identity_sha256, "R3e source identity"
        )
        self.runtime_sha256 = _sha256(runtime_sha256, "R3e runtime identity")
        self.base_policy_sha256 = _sha256(
            base_policy_sha256, "R3e base policy identity"
        )
        if any(
            type(value) is not int or value < 1
            for value in (
                self.episode_ticks,
                self.query_stride_shots,
                self.maximum_search_queries,
            )
        ):
            raise ValueError("R3e collection policy counts must be positive")
        config = getattr(teacher, "config", None)
        candidate_config = getattr(config, "candidate_config", None)
        if not isinstance(candidate_config, GeometrySearchConfig):
            raise TypeError("R3e teacher must expose a geometry candidate config")
        self.candidate_config = candidate_config
        self.candidate_vocabulary_sha256 = (
            geometry_candidate_vocabulary_sha256(candidate_config)
        )
        self.runway_ticks = int(getattr(config, "runway_ticks", 0))
        self.teacher_sha256 = _sha256(
            getattr(teacher, "sha256", ""), "R3e teacher identity"
        )
        if not callable(getattr(teacher, "search", None)):
            raise TypeError("R3e teacher must expose transactional search")
        self.examples: list[GeometryExample] = []
        self.ranking_examples: list[GeometryRankingExample] = []
        self.selected: Counter[str] = Counter()
        self.seen_shots = 0
        self.searches = 0
        self.boundary_skips = 0
        self.unsupported_pairs = 0
        self.strict_improvements = 0
        self.branch_outcomes = 0
        self.transactional_checks = 0

    def reset(self, seed: int = 0) -> None:
        if int(seed) != self.seed:
            raise RuntimeError("R3e collection policy seed binding changed")
        getattr(self.base_policy, "reset")(seed)
        self.examples.clear()
        self.ranking_examples.clear()
        self.selected.clear()
        self.seen_shots = 0
        self.searches = 0
        self.boundary_skips = 0
        self.unsupported_pairs = 0
        self.strict_improvements = 0
        self.branch_outcomes = 0
        self.transactional_checks = 0

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = getattr(self.base_policy, "predict")(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("R3e base policy did not expose a SteeringDecision")
        if not incumbent.is_shot:
            return incumbent
        self.seen_shots += 1
        if not _piece_pair(observation, incumbent):
            self.unsupported_pairs += 1
            return incumbent
        tick = int(observation.get("tick", 0))
        if (
            (self.seen_shots - 1) % self.query_stride_shots
            or self.searches >= self.maximum_search_queries
        ):
            return incumbent
        if tick + self.runway_ticks > self.episode_ticks:
            self.boundary_skips += 1
            return incumbent

        self.searches += 1
        before = self.env.clone_state()
        result = self.teacher.search(self.env, observation, incumbent)
        after = self.env.clone_state()
        self.transactional_checks += 1
        if before != after:
            raise RuntimeError("R3e runway teacher did not restore its source state")
        if (
            not result.outcomes
            or result.runway_ticks != self.runway_ticks
            or result.candidate_set.config != self.candidate_config
            or result.selected_candidate.ordinal != result.winner_ordinal
            or result.candidate_set.candidate_at(result.winner_ordinal) is None
        ):
            raise RuntimeError("R3e runway teacher returned inconsistent evidence")

        self.selected[result.selected_candidate.name] += 1
        self.strict_improvements += int(result.strictly_improved)
        self.branch_outcomes += len(result.outcomes)
        provenance = _canonical_sha256(
            {
                "source_identity_sha256": self.source_identity_sha256,
                "runtime_sha256": self.runtime_sha256,
                "base_policy_sha256": self.base_policy_sha256,
                "teacher_sha256": self.teacher_sha256,
                "candidate_vocabulary_sha256": (
                    self.candidate_vocabulary_sha256
                ),
                "rollout_mode": self.rollout_mode,
                "seed": self.seed,
                "tick": tick,
                "shot_index": self.seen_shots,
                "search_result_sha256": result.sha256,
            }
        )
        episode_identity = (
            f"r3e-{self.rollout_mode.replace('_', '-')}:{self.seed:08x}:{tick}:"
            f"{self.seen_shots}"
        )
        ranking = geometry_ranking_example(
            observation,
            result,
            episode_identity=episode_identity,
            provenance_sha256=provenance,
        )
        if (
            ranking.winner_index != result.winner_ordinal
            or ranking.improved_over_incumbent != result.strictly_improved
            or ranking.candidate_vocabulary_sha256
            != self.candidate_vocabulary_sha256
        ):
            raise RuntimeError("R3e ranking conversion changed the oracle label")
        self.ranking_examples.append(ranking)
        self.examples.append(
            geometry_example(
                observation,
                source_body_id=int(incumbent.source_body_id),
                destination_body_id=int(incumbent.destination_body_id),
                candidate_index=ranking.winner_index,
                candidate_count=ranking.candidate_count,
                improved_over_incumbent=ranking.improved_over_incumbent,
                episode_identity=episode_identity,
                provenance_sha256=provenance,
                candidate_set_sha256=self.candidate_vocabulary_sha256,
            )
        )
        return (
            result.decision
            if self.rollout_mode == "oracle_visited"
            else incumbent
        )

    def statistics(self) -> dict[str, object]:
        return {
            "seen_shots": self.seen_shots,
            "search_queries": self.searches,
            "episode_boundary_skips": self.boundary_skips,
            "unsupported_pairs": self.unsupported_pairs,
            "transactional_restore_checks": self.transactional_checks,
            "strict_improvements": self.strict_improvements,
            "branch_outcomes": self.branch_outcomes,
            "selected_candidate_counts": dict(sorted(self.selected.items())),
        }


def collect_geometry_labels(
    *,
    library_path: Path,
    base_policy_path: Path,
    base_policy_sha256: str,
    base_options: Mapping[str, Any],
    teacher: Any,
    seeds: Sequence[int],
    episode_ticks: int,
    query_stride_shots: int,
    maximum_search_queries_per_episode: int,
    source_identity_sha256: str,
    rollout_mode: str = "oracle_visited",
) -> CollectionRun:
    examples: list[GeometryExample] = []
    ranking_examples: list[GeometryRankingExample] = []
    episodes: list[Any] = []
    totals: Counter[str] = Counter()
    selected: Counter[str] = Counter()
    runtime_sha256 = r3d._file_sha256(library_path)
    with IrisuEnv(
        library_path=library_path,
        physics_backend="portable",
        config={"max_episode_ticks": episode_ticks},
    ) as env:
        if Path(env.library_path).resolve() != library_path:
            raise RuntimeError("R3e collection loaded a foreign runtime")
        runner = env.runner_identity_manifest()
        config_hash = int(runner["config_hash"])
        for seed in seeds:
            policy = OracleVisitedCollectionPolicy(
                env=env,
                base_policy=_base_policy(
                    base_policy_path, base_policy_sha256, base_options
                ),
                teacher=teacher,
                seed=int(seed),
                episode_ticks=episode_ticks,
                query_stride_shots=query_stride_shots,
                maximum_search_queries=maximum_search_queries_per_episode,
                source_identity_sha256=source_identity_sha256,
                runtime_sha256=runtime_sha256,
                base_policy_sha256=base_policy_sha256,
                rollout_mode=rollout_mode,
            )
            episodes.append(
                r3d._run_episode(
                    env,
                    policy,
                    label=f"r3e_collection_{rollout_mode}",
                    seed=int(seed),
                    config_hash=config_hash,
                )
            )
            examples.extend(policy.examples)
            ranking_examples.extend(policy.ranking_examples)
            statistics = policy.statistics()
            selected.update(statistics.pop("selected_candidate_counts"))
            totals.update(statistics)
    if not examples or len(examples) != len(ranking_examples):
        raise RuntimeError("R3e collection produced no runway geometry labels")
    return CollectionRun(
        GeometryDataset(examples),
        GeometryRankingDataset(ranking_examples),
        tuple(episodes),
        runner,
        {
            **dict(sorted(totals.items())),
            "selected_candidate_counts": dict(sorted(selected.items())),
            "rollout_policy": (
                "queried runway winner executed"
                if rollout_mode == "oracle_visited"
                else "queried oracle labels; rollout policy action executed"
            ),
            "rollout_mode": rollout_mode,
        },
    )


def evaluate_policy(
    *,
    label: str,
    library_path: Path,
    seeds: Sequence[int],
    horizon_ticks: int,
    factory: Any,
) -> dict[str, object]:
    episodes: list[Any] = []
    policy_counts: Counter[str] = Counter()
    with IrisuEnv(
        library_path=library_path,
        physics_backend="portable",
        config={"max_episode_ticks": horizon_ticks},
    ) as env:
        if Path(env.library_path).resolve() != library_path:
            raise RuntimeError("R3e evaluation loaded a foreign runtime")
        runner = env.runner_identity_manifest()
        config_hash = int(runner["config_hash"])
        for seed in seeds:
            policy = factory()
            episodes.append(
                r3d._run_episode(
                    env,
                    policy,
                    label=label,
                    seed=int(seed),
                    config_hash=config_hash,
                )
            )
            statistics = getattr(policy, "statistics", None)
            if callable(statistics):
                policy_counts.update(statistics())
    return {
        "runner": runner,
        "episodes": [episode.manifest() for episode in episodes],
        "aggregate": r3d._aggregate(episodes),
        "policy_counts": dict(sorted(policy_counts.items())),
    }


def _paired_safety_comparison(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    horizon_ticks: int,
    catastrophic_survival_ratio: float,
    catastrophic_survival_loss_ticks: int,
) -> dict[str, object]:
    if not isinstance(candidate, Mapping) or not isinstance(
        baseline, Mapping
    ):
        raise ValueError("paired evaluations must be mappings")
    if type(horizon_ticks) is not int or horizon_ticks < 1:
        raise ValueError("paired evaluation horizon must be positive")
    if (
        isinstance(catastrophic_survival_ratio, bool)
        or not isinstance(catastrophic_survival_ratio, (int, float))
        or not math.isfinite(catastrophic_survival_ratio)
        or not 0.0 < catastrophic_survival_ratio <= 1.0
    ):
        raise ValueError("catastrophic survival ratio must be in (0, 1]")
    if (
        type(catastrophic_survival_loss_ticks) is not int
        or catastrophic_survival_loss_ticks < 1
    ):
        raise ValueError("catastrophic survival loss must be positive")

    candidate_runner = candidate.get("runner")
    baseline_runner = baseline.get("runner")
    if (
        not isinstance(candidate_runner, Mapping)
        or not isinstance(baseline_runner, Mapping)
        or dict(candidate_runner) != dict(baseline_runner)
    ):
        raise ValueError("R3e paired evaluations use different runner identities")

    def by_seed(
        evaluation: Mapping[str, Any],
        label: str,
    ) -> dict[int, dict[str, int | bool]]:
        episodes = evaluation.get("episodes")
        if not isinstance(episodes, Sequence) or isinstance(
            episodes, (str, bytes)
        ):
            raise ValueError("R3e paired evaluation episodes are malformed")
        output: dict[int, dict[str, int | bool]] = {}
        for episode in episodes:
            if not isinstance(episode, Mapping):
                raise ValueError("R3e paired evaluation episode is malformed")
            seed = episode.get("seed")
            if type(seed) is not int or seed in output:
                raise ValueError("R3e paired evaluation seeds are invalid")
            conversion = episode.get("conversion")
            if not isinstance(conversion, Mapping):
                raise ValueError("R3e paired conversion evidence is malformed")
            survival = conversion.get("survival_ticks")
            score = conversion.get("final_score")
            terminated = conversion.get("terminated")
            truncated = conversion.get("truncated")
            gauge_failure = episode.get("gauge_failure")
            final_gauge = episode.get("final_gauge")
            gauge_max = episode.get("gauge_max")
            if (
                type(survival) is not int
                or not 0 <= survival <= horizon_ticks
                or type(score) is not int
                or score < 0
                or type(terminated) is not bool
                or type(truncated) is not bool
                or terminated == truncated
                or type(gauge_failure) is not bool
                or type(final_gauge) is not int
                or type(gauge_max) is not int
                or gauge_max < 1
                or final_gauge > gauge_max
            ):
                raise ValueError("R3e paired outcome values are invalid")
            expected_failure = terminated and final_gauge <= 1
            if gauge_failure != expected_failure:
                raise ValueError(
                    "R3e paired gauge-failure evidence is inconsistent"
                )
            output[seed] = {
                "survival_ticks": survival,
                "final_score": score,
                "terminated": terminated,
                "truncated": truncated,
                "gauge_failure": gauge_failure,
                "final_gauge": final_gauge,
                "gauge_max": gauge_max,
            }
        if not output:
            raise ValueError("R3e paired evaluation cannot be empty")
        aggregate = evaluation.get("aggregate")
        if not isinstance(aggregate, Mapping):
            raise ValueError("R3e paired aggregate evidence is malformed")
        aggregate_episodes = aggregate.get("episodes")
        aggregate_failures = aggregate.get("gauge_failures")
        if (
            type(aggregate_episodes) is not int
            or aggregate_episodes != len(output)
            or type(aggregate_failures) is not int
            or aggregate_failures
            != sum(bool(row["gauge_failure"]) for row in output.values())
        ):
            raise ValueError(
                f"R3e {label} aggregate evidence disagrees with episodes"
            )
        return output

    candidate_by_seed = by_seed(candidate, "candidate")
    baseline_by_seed = by_seed(baseline, "baseline")
    if candidate_by_seed.keys() != baseline_by_seed.keys():
        raise ValueError("R3e paired evaluations use different seeds")

    survival_deltas: list[int] = []
    survival_ratios: list[float] = []
    score_deltas: list[int] = []
    new_terminal_failure_seeds: list[int] = []
    rescued_terminal_failure_seeds: list[int] = []
    new_gauge_failure_seeds: list[int] = []
    rescued_gauge_failure_seeds: list[int] = []
    catastrophic_seeds: list[int] = []
    material_regression_seeds: list[int] = []
    survival_regressions = 0
    survival_improvements = 0
    score_improvements = 0
    pairs: list[dict[str, object]] = []
    for seed in sorted(candidate_by_seed):
        candidate_episode = candidate_by_seed[seed]
        baseline_episode = baseline_by_seed[seed]
        candidate_survival = int(candidate_episode["survival_ticks"])
        baseline_survival = int(baseline_episode["survival_ticks"])
        candidate_score = int(candidate_episode["final_score"])
        baseline_score = int(baseline_episode["final_score"])
        candidate_terminated = bool(candidate_episode["terminated"])
        baseline_terminated = bool(baseline_episode["terminated"])
        candidate_failure = bool(candidate_episode["gauge_failure"])
        baseline_failure = bool(baseline_episode["gauge_failure"])

        survival_delta = candidate_survival - baseline_survival
        survival_ratio = (
            candidate_survival / baseline_survival
            if baseline_survival > 0
            else 1.0
        )
        score_delta = candidate_score - baseline_score
        survival_deltas.append(survival_delta)
        survival_ratios.append(survival_ratio)
        score_deltas.append(score_delta)
        survival_regressions += int(survival_delta < 0)
        survival_improvements += int(survival_delta > 0)
        score_improvements += int(score_delta > 0)
        new_terminal_failure = candidate_terminated and not baseline_terminated
        rescued_terminal_failure = (
            baseline_terminated and not candidate_terminated
        )
        if new_terminal_failure:
            new_terminal_failure_seeds.append(seed)
        if rescued_terminal_failure:
            rescued_terminal_failure_seeds.append(seed)
        if candidate_failure and not baseline_failure:
            new_gauge_failure_seeds.append(seed)
        if baseline_failure and not candidate_failure:
            rescued_gauge_failure_seeds.append(seed)
        material_regression = (
            survival_delta <= -catastrophic_survival_loss_ticks
        )
        catastrophic = (
            survival_delta <= -catastrophic_survival_loss_ticks
            and survival_ratio <= catastrophic_survival_ratio
        )
        if material_regression:
            material_regression_seeds.append(seed)
        if catastrophic:
            catastrophic_seeds.append(seed)
        pairs.append(
            {
                "seed": seed,
                "baseline_survival_ticks": baseline_survival,
                "candidate_survival_ticks": candidate_survival,
                "survival_delta_ticks": survival_delta,
                "candidate_to_base_survival_ratio": survival_ratio,
                "baseline_final_score": baseline_score,
                "candidate_final_score": candidate_score,
                "score_delta": score_delta,
                "baseline_terminated": baseline_terminated,
                "candidate_terminated": candidate_terminated,
                "baseline_gauge_failure": baseline_failure,
                "candidate_gauge_failure": candidate_failure,
                "new_terminal_failure": new_terminal_failure,
                "rescued_terminal_failure": rescued_terminal_failure,
                "new_gauge_failure": (
                    candidate_failure and not baseline_failure
                ),
                "rescued_gauge_failure": (
                    baseline_failure and not candidate_failure
                ),
                "material_survival_regression": material_regression,
                "catastrophic_survival_regression": catastrophic,
            }
        )

    return {
        "schema": "irisu-r3e-paired-safety-v1",
        "horizon_ticks": horizon_ticks,
        "runner": dict(candidate_runner),
        "episodes": len(candidate_by_seed),
        "seeds": sorted(candidate_by_seed),
        "catastrophic_definition": {
            "maximum_candidate_to_base_survival_ratio": (
                catastrophic_survival_ratio
            ),
            "minimum_survival_loss_ticks": catastrophic_survival_loss_ticks,
            "conjunction": True,
        },
        "zero_baseline_survival_ratio_convention": (
            "1.0 because candidate survival cannot regress below zero"
        ),
        "candidate_gauge_failures": sum(
            bool(row["gauge_failure"]) for row in candidate_by_seed.values()
        ),
        "baseline_gauge_failures": sum(
            bool(row["gauge_failure"]) for row in baseline_by_seed.values()
        ),
        "new_terminal_failures": len(new_terminal_failure_seeds),
        "new_terminal_failure_seeds": new_terminal_failure_seeds,
        "rescued_terminal_failures": len(rescued_terminal_failure_seeds),
        "rescued_terminal_failure_seeds": rescued_terminal_failure_seeds,
        "new_gauge_failures": len(new_gauge_failure_seeds),
        "new_gauge_failure_seeds": new_gauge_failure_seeds,
        "rescued_gauge_failures": len(rescued_gauge_failure_seeds),
        "rescued_gauge_failure_seeds": rescued_gauge_failure_seeds,
        "material_survival_regressions": len(material_regression_seeds),
        "material_survival_regression_seeds": material_regression_seeds,
        "catastrophic_survival_regressions": len(catastrophic_seeds),
        "catastrophic_survival_regression_seeds": catastrophic_seeds,
        "survival_regressions": survival_regressions,
        "survival_improvements": survival_improvements,
        "score_improvements": score_improvements,
        "minimum_survival_delta": min(survival_deltas),
        "median_survival_delta": float(np.median(survival_deltas)),
        "minimum_survival_ratio": min(survival_ratios),
        "median_score_delta": float(np.median(score_deltas)),
        "pairs": pairs,
    }


@dataclass(frozen=True, slots=True)
class CurveModel:
    steps: int
    model: GeometrySelectorModel
    report: GeometryTrainingReport
    state_dict_sha256: str


def train_plateau_models(
    dataset: GeometryDataset,
    *,
    budgets: Sequence[int],
    model_config: GeometryModelConfig,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[CurveModel, ...]:
    output: list[CurveModel] = []
    for steps in budgets:
        torch.manual_seed(seed)
        model = GeometrySelectorModel(
            dataset.schema,
            candidate_count=dataset.candidate_count,
            candidate_set_sha256=dataset.candidate_set_sha256,
            config=model_config,
        )
        report = train_geometry_selector(
            model,
            dataset,
            steps=int(steps),
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
        )
        model.eval()
        output.append(
            CurveModel(int(steps), model, report, _state_dict_sha256(model))
        )
    return tuple(output)


@dataclass(frozen=True, slots=True)
class RankingCurveModel:
    steps: int
    model: GeometrySelectorModel
    report: GeometryRankingTrainingReport
    state_dict_sha256: str


def train_ranking_plateau_models(
    dataset: GeometryRankingDataset,
    *,
    budgets: Sequence[int],
    model_config: GeometryModelConfig,
    batch_size: int,
    learning_rate: float,
    listwise_weight: float,
    pairwise_weight: float,
    seed: int,
) -> tuple[RankingCurveModel, ...]:
    output: list[RankingCurveModel] = []
    for steps in budgets:
        torch.manual_seed(seed)
        model = GeometrySelectorModel(
            dataset.schema,
            candidate_count=dataset.candidate_count,
            candidate_set_sha256=dataset.candidate_vocabulary_sha256,
            config=model_config,
        )
        report = train_geometry_ranker(
            model,
            dataset,
            steps=int(steps),
            batch_size=batch_size,
            learning_rate=learning_rate,
            listwise_weight=listwise_weight,
            pairwise_weight=pairwise_weight,
            seed=seed,
        )
        model.eval()
        output.append(
            RankingCurveModel(
                int(steps), model, report, _state_dict_sha256(model)
            )
        )
    return tuple(output)


def _require_aligned_collections(
    winner: GeometryDataset,
    ranking: GeometryRankingDataset,
) -> None:
    if (
        len(winner) != len(ranking)
        or winner.candidate_set_sha256
        != ranking.candidate_vocabulary_sha256
        or winner.candidate_count != ranking.candidate_count
        or any(
            left.episode_identity != right.episode_identity
            or left.provenance_sha256 != right.provenance_sha256
            or left.candidate_index != right.winner_index
            or left.improved_over_incumbent
            != right.improved_over_incumbent
            for left, right in zip(winner, ranking, strict=True)
        )
    ):
        raise ValueError("R3e winner and ranking collections are not aligned")


def _curve_payloads(
    curves: Sequence[CurveModel | RankingCurveModel],
    *,
    learner: str,
    campaign: bool,
    library_path: Path,
    seeds: Sequence[int],
    horizons: Sequence[int],
    factory: Any,
    base_evaluation: Mapping[str, Any] | None = None,
    catastrophic_survival_ratio: float = 0.5,
    catastrophic_survival_loss_ticks: int = 1_000,
) -> list[dict[str, object]]:
    normalized_horizons = _selection_horizons(horizons)
    if campaign:
        if not isinstance(base_evaluation, Mapping):
            raise ValueError(
                "R3e campaign requires paired base-v5 evaluation evidence"
            )
        missing = [
            horizon
            for horizon in normalized_horizons
            if str(horizon) not in base_evaluation
        ]
        if missing:
            raise ValueError(
                "R3e campaign base-v5 evaluation omits required horizons"
            )
    payloads: list[dict[str, object]] = []
    for point in curves:
        payload: dict[str, object] = {
            "training_steps": point.steps,
            "state_dict_sha256": point.state_dict_sha256,
            "training": asdict(point.report),
            "evaluation": {},
        }
        if campaign:
            evaluations: dict[str, object] = {}
            assert base_evaluation is not None
            for horizon in normalized_horizons:
                evaluation = evaluate_policy(
                    label=f"r3e_{learner}_steps_{point.steps}",
                    library_path=library_path,
                    seeds=seeds,
                    horizon_ticks=int(horizon),
                    factory=lambda model=point.model: factory(model),
                )
                evaluation["paired_safety"] = _paired_safety_comparison(
                    evaluation,
                    base_evaluation[str(horizon)],
                    horizon_ticks=horizon,
                    catastrophic_survival_ratio=(
                        catastrophic_survival_ratio
                    ),
                    catastrophic_survival_loss_ticks=(
                        catastrophic_survival_loss_ticks
                    ),
                )
                evaluations[str(horizon)] = evaluation
            payload["evaluation"] = evaluations
        payloads.append(payload)
    return payloads


def _selection_horizons(horizons: Sequence[int]) -> tuple[int, ...]:
    if isinstance(horizons, (str, bytes)):
        raise ValueError("R3e selection horizons are malformed")
    values = tuple(horizons)
    if (
        not values
        or any(type(value) is not int or value < 1 for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError("R3e selection horizons must be unique and positive")
    return tuple(sorted(values))


def _required_count(value: Mapping[str, Any], name: str) -> int:
    if name not in value:
        raise ValueError(f"R3e paired safety field {name} is missing")
    result = value[name]
    if type(result) is not int or result < 0:
        raise ValueError(f"R3e paired safety field {name} is invalid")
    return result


def _required_finite(value: Mapping[str, Any], name: str) -> float:
    if name not in value:
        raise ValueError(f"R3e selection field {name} is missing")
    result = value[name]
    if (
        isinstance(result, bool)
        or not isinstance(result, (int, float))
        or not math.isfinite(float(result))
    ):
        raise ValueError(f"R3e selection field {name} is invalid")
    return float(result)


def _required_integer(value: Mapping[str, Any], name: str) -> int:
    if name not in value or type(value[name]) is not int:
        raise ValueError(f"R3e selection field {name} is invalid")
    return int(value[name])


def _point_safety_assessment(
    point: Mapping[str, Any],
    horizons: Sequence[int],
) -> dict[str, object]:
    normalized = _selection_horizons(horizons)
    evaluation = point.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise ValueError("R3e plateau evaluation evidence is malformed")
    totals = {
        "new_terminal_failures": 0,
        "new_gauge_failures": 0,
        "catastrophic_survival_regressions": 0,
    }
    rejected_by: list[dict[str, object]] = []
    for horizon in normalized:
        horizon_evaluation = evaluation.get(str(horizon))
        if not isinstance(horizon_evaluation, Mapping):
            raise ValueError("R3e plateau horizon evidence is malformed")
        paired = horizon_evaluation.get("paired_safety")
        if (
            not isinstance(paired, Mapping)
            or paired.get("schema") != "irisu-r3e-paired-safety-v1"
            or paired.get("horizon_ticks") != horizon
        ):
            raise ValueError("R3e plateau paired safety evidence is malformed")
        for criterion in totals:
            count = _required_count(paired, criterion)
            totals[criterion] += count
            seed_field = f"{criterion[:-1]}_seeds"
            seeds = paired.get(seed_field)
            if (
                not isinstance(seeds, Sequence)
                or isinstance(seeds, (str, bytes))
                or len(seeds) != count
                or any(type(seed) is not int for seed in seeds)
                or len(set(seeds)) != len(seeds)
            ):
                raise ValueError(
                    f"R3e paired safety seeds for {criterion} are invalid"
                )
            if count:
                rejected_by.append(
                    {
                        "horizon_ticks": horizon,
                        "criterion": criterion,
                        "count": count,
                        "seeds": list(seeds),
                    }
                )
    return {
        "eligible": not rejected_by,
        "eligibility_horizon_ticks": list(normalized),
        **totals,
        "rejected_by": rejected_by,
    }


def _aggregate_selection_metrics(
    evaluation: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    aggregate = evaluation.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise ValueError("R3e plateau aggregate evidence is malformed")
    gauge_failures = _required_count(aggregate, "gauge_failures")
    survival = aggregate.get("survival_ticks")
    score = aggregate.get("raw_score")
    if not isinstance(survival, Mapping) or not isinstance(score, Mapping):
        raise ValueError("R3e plateau distributions are malformed")
    return (
        float(gauge_failures),
        _required_finite(survival, "p10"),
        _required_finite(survival, "median"),
        _required_finite(score, "median"),
    )


def _selection_key(
    point: Mapping[str, Any], horizons: Sequence[int]
) -> tuple[float, ...]:
    normalized = _selection_horizons(horizons)
    assessment = _point_safety_assessment(point, normalized)
    evaluation = point.get("evaluation")
    assert isinstance(evaluation, Mapping)
    horizon = evaluation.get(str(normalized[-1]))
    if not isinstance(horizon, Mapping):
        raise ValueError("R3e plateau horizon evidence is malformed")
    paired = horizon.get("paired_safety")
    if not isinstance(paired, Mapping):
        raise ValueError("R3e plateau paired safety evidence is malformed")
    gauge_failures, p10, median_survival, median_score = (
        _aggregate_selection_metrics(horizon)
    )
    paired_gauge_failures = _required_count(
        paired, "candidate_gauge_failures"
    )
    if paired_gauge_failures != int(gauge_failures):
        raise ValueError(
            "R3e paired and aggregate gauge-failure evidence disagree"
        )
    minimum_survival_ratio = _required_finite(
        paired, "minimum_survival_ratio"
    )
    if minimum_survival_ratio < 0.0:
        raise ValueError("R3e minimum survival ratio cannot be negative")
    steps = point.get("training_steps")
    if type(steps) is not int or steps < 1:
        raise ValueError("R3e plateau training steps are invalid")
    return (
        float(bool(assessment["eligible"])),
        -float(assessment["new_terminal_failures"]),
        -float(assessment["new_gauge_failures"]),
        -float(assessment["catastrophic_survival_regressions"]),
        -gauge_failures,
        float(_required_count(paired, "rescued_gauge_failures")),
        minimum_survival_ratio,
        float(_required_integer(paired, "minimum_survival_delta")),
        -float(_required_count(paired, "survival_regressions")),
        p10,
        median_survival,
        median_score,
        -float(steps),
    )


def _baseline_selection_key(
    base_evaluation: Mapping[str, Any],
    horizons: Sequence[int],
) -> tuple[float, ...]:
    normalized = _selection_horizons(horizons)
    horizon = base_evaluation.get(str(normalized[-1]))
    if not isinstance(horizon, Mapping):
        raise ValueError("R3e base-v5 horizon evidence is malformed")
    gauge_failures, p10, median_survival, median_score = (
        _aggregate_selection_metrics(horizon)
    )
    return (
        1.0,
        0.0,
        0.0,
        0.0,
        -gauge_failures,
        0.0,
        1.0,
        0.0,
        0.0,
        p10,
        median_survival,
        median_score,
        0.0,
    )


def _select_plateau_candidate(
    points: Sequence[Mapping[str, Any]],
    base_evaluation: Mapping[str, Any],
    horizons: Sequence[int],
) -> dict[str, object]:
    normalized = _selection_horizons(horizons)
    if not points:
        raise ValueError("R3e plateau selection requires candidate points")
    base_key = _baseline_selection_key(base_evaluation, normalized)
    records: list[dict[str, object]] = []
    for index, point in enumerate(points):
        assessment = _point_safety_assessment(point, normalized)
        key = _selection_key(point, normalized)
        records.append(
            {
                "curve_index": index,
                "training_steps": point["training_steps"],
                "state_dict_sha256": point.get("state_dict_sha256"),
                "eligible": assessment["eligible"],
                "selection_key": list(key),
                "safety": assessment,
            }
        )
    eligible = [
        index
        for index, record in enumerate(records)
        if bool(record["eligible"])
    ]
    best_eligible = (
        max(eligible, key=lambda index: tuple(records[index]["selection_key"]))
        if eligible
        else None
    )
    accepted = (
        best_eligible is not None
        and tuple(records[best_eligible]["selection_key"]) > base_key
    )
    selected_index = best_eligible if accepted else None
    rejected: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if index == selected_index:
            continue
        reasons = list(record["safety"]["rejected_by"])
        if bool(record["eligible"]):
            reasons.append(
                {
                    "criterion": (
                        "lower_ranked_eligible_plateau"
                        if accepted
                        else "did_not_outperform_frozen_v5"
                    )
                }
            )
        rejected.append({**record, "rejection_reasons": reasons})
    best_learned = max(
        range(len(records)),
        key=lambda index: tuple(records[index]["selection_key"]),
    )
    return {
        "schema": "irisu-r3e-fail-closed-plateau-selection-v1",
        "eligibility_horizon_ticks": list(normalized),
        "ranking_horizon_ticks": normalized[-1],
        "baseline": "frozen_v5",
        "baseline_selection_key": list(base_key),
        "best_learned_curve_index": best_learned,
        "best_eligible_curve_index": best_eligible,
        "selected_curve_index": selected_index,
        "accepted_learned_candidate": accepted,
        "retained_policy": (
            "learned_candidate" if accepted else "frozen_v5"
        ),
        "rejected_candidates": rejected,
    }


def _search_config(value: Mapping[str, Any]) -> GeometrySearchConfig:
    return GeometrySearchConfig(
        horizon_ticks=int(value["horizon_ticks"]),
        velocity_lead_ticks=float(value["velocity_lead_ticks"]),
        ticks_per_second=float(value["ticks_per_second"]),
        support_fractions=tuple(float(item) for item in value["support_fractions"]),
        support_clearance_sizes=float(value["support_clearance_sizes"]),
        grid_x_fractions=tuple(float(item) for item in value["grid_x_fractions"]),
        grid_y_sizes=tuple(float(item) for item in value["grid_y_sizes"]),
        max_candidates=int(value["max_candidates"]),
    )


def _collection_teacher(
    value: Mapping[str, Any],
    candidate_config: GeometrySearchConfig,
) -> RunwayGeometrySearch:
    expected = {
        "kind": "runway_geometry",
        "cross_spawn_boundaries": True,
        "drives_collection_rollout_on_queried_shots": True,
    }
    if any(value.get(name) != expected_value for name, expected_value in expected.items()):
        raise ValueError("R3e collection teacher must be the oracle-runway protocol")
    return RunwayGeometrySearch(
        config=RunwaySearchConfig(
            runway_ticks=int(value["runway_ticks"]),
            candidate_config=candidate_config,
        )
    )


def _final_report(
    payload: Mapping[str, object],
    *,
    source_identity: Mapping[str, object],
    config_snapshot: Any,
    runtime_snapshot: Any,
    base_snapshot: Any,
    started: float,
) -> dict[str, object]:
    content = {
        "schema": _REPORT_FORMAT,
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
        "source_identity": dict(source_identity),
        "config": {
            "path": str(config_snapshot.path),
            "sha256": config_snapshot.sha256,
        },
        "runtime": {
            "path": str(runtime_snapshot.path),
            "sha256": runtime_snapshot.sha256,
            "backend": "portable",
        },
        "base_policy": {
            "path": str(base_snapshot.path),
            "sha256": base_snapshot.sha256,
        },
        **dict(payload),
        "execution": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
        },
    }
    return {**content, "payload_sha256": _canonical_sha256(content)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("collect", "train", "evaluate", "campaign"),
        default="campaign",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--profile",
        choices=("smoke", "fast", "medium", "deep", "long"),
        default="fast",
    )
    parser.add_argument(
        "--learner",
        choices=("winner", "ranking", "both"),
        default="both",
        help="train/evaluate the winner baseline, all-branch ranker, or both",
    )
    parser.add_argument("--library", type=Path, default=TRUSTED_PORTABLE)
    parser.add_argument("--base-policy", type=Path)
    parser.add_argument("--base-policy-sha256")
    parser.add_argument(
        "--collection-path",
        type=Path,
        default=Path("/tmp/r3e-sustainable-collection.pt"),
    )
    parser.add_argument("--collection-sha256")
    parser.add_argument(
        "--ranking-collection-path",
        type=Path,
        default=Path("/tmp/r3e-sustainable-ranking-collection.pt"),
    )
    parser.add_argument("--ranking-collection-sha256")
    parser.add_argument(
        "--geometry-path",
        type=Path,
        default=Path("/tmp/r3e-sustainable-geometry.pt"),
    )
    parser.add_argument("--geometry-sha256")
    parser.add_argument(
        "--ranking-geometry-path",
        type=Path,
        default=Path("/tmp/r3e-sustainable-ranking-geometry.pt"),
    )
    parser.add_argument("--ranking-geometry-sha256")
    parser.add_argument(
        "--result-out",
        type=Path,
        default=Path("/tmp/r3e-sustainable-report.json"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    config_snapshot = r3d._snapshot_file(args.config, "R3e config")
    config = _load_config(config_snapshot)
    settings = _settings(config["profiles"][args.profile])
    runtime_snapshot = r3d._snapshot_file(args.library, "R3e portable runtime")
    if runtime_snapshot.path != TRUSTED_PORTABLE.resolve(strict=True):
        parser.error("--library must be the trusted portable runtime")
    configured_base = ROOT / config["base_policy"]["checkpoint"]
    base_path = configured_base if args.base_policy is None else args.base_policy
    base_snapshot = r3d._snapshot_file(base_path, "R3e base policy")
    base_expected = (
        str(config["base_policy"]["sha256"])
        if args.base_policy_sha256 is None
        else args.base_policy_sha256
    )
    if base_snapshot.sha256 != _sha256(base_expected, "R3e base policy identity"):
        parser.error("base policy SHA-256 does not match")
    source_identity = _source_identity(config_snapshot.path)
    torch_threads = int(config["training"]["torch_threads"])
    if torch_threads < 1:
        raise ValueError("R3e torch thread count must be positive")
    torch.set_num_threads(torch_threads)
    training_seed = int(config["training"]["seed"])
    learning_rate = float(config["training"]["learning_rate"])
    candidate_config = _search_config(config["search"])
    teacher = _collection_teacher(
        config["collection_teacher"],
        candidate_config,
    )
    rollout_mode = str(config["collection_teacher"]["rollout_mode"])
    if rollout_mode != "oracle_visited":
        raise ValueError(
            "R3e CLI currently requires oracle-visited collection; "
            "learner-visited mode is an injectable policy hook"
        )
    candidate_vocabulary_sha256 = geometry_candidate_vocabulary_sha256(
        candidate_config
    )
    model_config = GeometryModelConfig(**config["geometry_model"])
    base_options = {
        name: value
        for name, value in config["base_policy"].items()
        if name not in {"checkpoint", "sha256"}
    }
    deployment = config["deployment"]
    selection = config["selection"]
    required_selection_protocol = {
        "eligibility_horizons": "all_profile_evaluation_horizons",
        "ranking_horizon": "longest_profile_evaluation_horizon",
        "baseline_tie_policy": "retain_frozen_v5",
    }
    if any(
        selection.get(name) != expected
        for name, expected in required_selection_protocol.items()
    ):
        raise ValueError("R3e fail-closed selection protocol is not enabled")
    catastrophic_survival_ratio = float(
        selection["catastrophic_survival_ratio"]
    )
    catastrophic_survival_loss_ticks = int(
        selection["catastrophic_survival_loss_ticks"]
    )
    if (
        not math.isfinite(catastrophic_survival_ratio)
        or not 0.0 < catastrophic_survival_ratio <= 1.0
    ):
        raise ValueError("R3e catastrophic survival ratio must be in (0, 1]")
    if catastrophic_survival_loss_ticks < 1:
        raise ValueError("R3e catastrophic survival loss must be positive")
    collection_path = r3d._output_path(
        args.collection_path, "R3e collection path", ".pt"
    )
    ranking_collection_path = r3d._output_path(
        args.ranking_collection_path, "R3e ranking collection path", ".pt"
    )
    geometry_path = r3d._output_path(
        args.geometry_path, "R3e geometry path", ".pt"
    )
    ranking_geometry_path = r3d._output_path(
        args.ranking_geometry_path, "R3e ranking geometry path", ".pt"
    )
    result_path = r3d._output_path(
        args.result_out, "R3e result output", ".json"
    )
    payload: dict[str, object] = {
        "mode": args.mode,
        "profile": args.profile,
        "learner": args.learner,
        "settings": asdict(settings),
        "seed_suites": {
            "collection": list(COLLECTION_SEEDS[: settings.collection_seeds]),
            "development": list(
                DEVELOPMENT_SEEDS[: settings.evaluation_seeds]
            ),
            "disjoint": not bool(
                set(COLLECTION_SEEDS) & set(DEVELOPMENT_SEEDS)
            ),
        },
        "collection_teacher": {
            "identity": teacher.identity_manifest(),
            "sha256": teacher.sha256,
            "candidate_vocabulary_sha256": candidate_vocabulary_sha256,
            "rollout_mode": rollout_mode,
        },
        "paired_safety_protocol": {
            "baseline": "frozen_v5_same_seed_same_horizon",
            **required_selection_protocol,
            "new_terminal_failures_must_be_zero": True,
            "new_gauge_failures_must_be_zero": True,
            "catastrophic_survival_regressions_must_be_zero": True,
            "catastrophic_survival_ratio": catastrophic_survival_ratio,
            "catastrophic_survival_loss_ticks": (
                catastrophic_survival_loss_ticks
            ),
        },
    }
    artifact_bindings = {
        "source_identity_sha256": source_identity["sha256"],
        "runtime_sha256": runtime_snapshot.sha256,
        "base_policy_sha256": base_snapshot.sha256,
        "teacher_sha256": teacher.sha256,
        "candidate_vocabulary_sha256": candidate_vocabulary_sha256,
    }

    collection: GeometryCollectionArtifact | None = None
    ranking_collection: GeometryRankingCollectionArtifact | None = None
    if args.mode in {"collect", "campaign"}:
        run = collect_geometry_labels(
            library_path=runtime_snapshot.path,
            base_policy_path=base_snapshot.path,
            base_policy_sha256=base_snapshot.sha256,
            base_options=base_options,
            teacher=teacher,
            seeds=COLLECTION_SEEDS[: settings.collection_seeds],
            episode_ticks=settings.collection_ticks,
            query_stride_shots=settings.query_stride_shots,
            maximum_search_queries_per_episode=(
                settings.maximum_search_queries_per_episode
            ),
            source_identity_sha256=str(source_identity["sha256"]),
            rollout_mode=rollout_mode,
        )
        collection_metadata = {
            **artifact_bindings,
            "profile": args.profile,
            "seeds": list(COLLECTION_SEEDS[: settings.collection_seeds]),
        }
        collection_sha = save_geometry_collection(
            collection_path,
            run.dataset,
            metadata={**collection_metadata, "learner": "winner_classifier"},
            overwrite=args.overwrite,
        )
        ranking_collection_sha = save_geometry_ranking_collection(
            ranking_collection_path,
            run.ranking_dataset,
            metadata={
                **collection_metadata,
                "learner": "all_candidate_ranker",
                "winner_dataset_sha256": run.dataset.sha256,
            },
            overwrite=args.overwrite,
        )
        collection = GeometryCollectionArtifact(
            collection_path,
            collection_sha,
            run.dataset,
            _development_metadata(collection_metadata),
        )
        ranking_collection = GeometryRankingCollectionArtifact(
            ranking_collection_path,
            ranking_collection_sha,
            run.ranking_dataset,
            _development_metadata(collection_metadata),
        )
        _require_aligned_collections(
            collection.dataset, ranking_collection.dataset
        )
        payload["collection"] = {
            "artifact": {
                "path": str(collection_path),
                "sha256": collection_sha,
            },
            "dataset_sha256": run.dataset.sha256,
            "examples": len(run.dataset),
            "improved_examples": sum(
                example.improved_over_incumbent for example in run.dataset
            ),
            "candidate_count": run.dataset.candidate_count,
            "candidate_set_sha256": run.dataset.candidate_set_sha256,
            "query_report": dict(run.query_report),
            "episodes": [episode.manifest() for episode in run.episodes],
            "aggregate": r3d._aggregate(run.episodes),
            "runner": dict(run.runner),
        }
        payload["ranking_collection"] = {
            "artifact": {
                "path": str(ranking_collection_path),
                "sha256": ranking_collection_sha,
            },
            "dataset_sha256": run.ranking_dataset.sha256,
            "examples": len(run.ranking_dataset),
            "preferences": sum(
                len(example.preferences) for example in run.ranking_dataset
            ),
            "branch_outcomes": sum(
                len(example.outcome_sha256s)
                for example in run.ranking_dataset
            ),
            "improved_examples": sum(
                example.improved_over_incumbent
                for example in run.ranking_dataset
            ),
            "candidate_count": run.ranking_dataset.candidate_count,
            "candidate_vocabulary_sha256": (
                run.ranking_dataset.candidate_vocabulary_sha256
            ),
            "unique_state_candidate_sets": len(
                {
                    example.candidate_set_sha256
                    for example in run.ranking_dataset
                }
            ),
        }
    elif args.mode == "train":
        if args.learner in {"winner", "both"}:
            if args.collection_sha256 is None:
                parser.error("--collection-sha256 is required for winner training")
            collection = load_geometry_collection(
                collection_path, expected_sha256=args.collection_sha256
            )
            _require_bound_metadata(
                collection.metadata,
                artifact_bindings,
                "winner collection",
            )
            if (
                collection.dataset.candidate_set_sha256
                != candidate_vocabulary_sha256
                or collection.dataset.candidate_count
                != teacher.config.candidate_config.slot_count
            ):
                raise ValueError("R3e collection and configured teacher differ")
            payload["collection"] = {
                "artifact": {
                    "path": str(collection.path),
                    "sha256": collection.sha256,
                },
                "dataset_sha256": collection.dataset.sha256,
                "examples": len(collection.dataset),
            }
        if args.learner in {"ranking", "both"}:
            if args.ranking_collection_sha256 is None:
                parser.error(
                    "--ranking-collection-sha256 is required for ranking training"
                )
            ranking_collection = load_geometry_ranking_collection(
                ranking_collection_path,
                expected_sha256=args.ranking_collection_sha256,
            )
            _require_bound_metadata(
                ranking_collection.metadata,
                artifact_bindings,
                "ranking collection",
            )
            if (
                ranking_collection.dataset.candidate_vocabulary_sha256
                != candidate_vocabulary_sha256
                or ranking_collection.dataset.candidate_count
                != teacher.config.candidate_config.slot_count
            ):
                raise ValueError(
                    "R3e ranking collection and configured teacher differ"
                )
            payload["ranking_collection"] = {
                "artifact": {
                    "path": str(ranking_collection.path),
                    "sha256": ranking_collection.sha256,
                },
                "dataset_sha256": ranking_collection.dataset.sha256,
                "examples": len(ranking_collection.dataset),
                "preferences": sum(
                    len(example.preferences)
                    for example in ranking_collection.dataset
                ),
            }
        if collection is not None and ranking_collection is not None:
            _require_aligned_collections(
                collection.dataset, ranking_collection.dataset
            )

    deployment_policy_config = GeometryPolicyConfig(
        minimum_confidence=float(
            deployment["minimum_candidate_confidence"]
        ),
        minimum_logit_margin=float(
            deployment["minimum_logit_margin_over_incumbent"]
        ),
        minimum_gauge_fraction=float(
            deployment["minimum_gauge_fraction"]
        ),
        maximum_unverified_corrections=int(
            deployment["maximum_unverified_corrections"]
        ),
    )
    payload["deployment_policy"] = deployment_policy_config.manifest()

    def deployed_policy(
        model: GeometrySelectorModel,
    ) -> SafeguardedGeometryPolicy:
        return SafeguardedGeometryPolicy(
            _base_policy(
                base_snapshot.path, base_snapshot.sha256, base_options
            ),
            model,
            geometry_config=teacher.config.candidate_config,
            policy_config=deployment_policy_config,
            selector_artifact_sha256=_state_dict_sha256(model),
            source_identity=source_identity["sha256"],
        )

    base_evaluation: dict[str, object] = {}
    development_seeds = DEVELOPMENT_SEEDS[: settings.evaluation_seeds]
    if args.mode == "campaign":
        for horizon in settings.evaluation_horizons:
            base_evaluation[str(horizon)] = evaluate_policy(
                label="r3e_base_v5",
                library_path=runtime_snapshot.path,
                seeds=development_seeds,
                horizon_ticks=horizon,
                factory=lambda: _base_policy(
                    base_snapshot.path, base_snapshot.sha256, base_options
                ),
            )

    selection_horizons = settings.evaluation_horizons
    selection_horizon = max(selection_horizons)
    selected_payloads: dict[str, Mapping[str, object]] = {}

    if (
        args.mode in {"train", "campaign"}
        and args.learner in {"winner", "both"}
    ):
        assert collection is not None
        curves = train_plateau_models(
            collection.dataset,
            budgets=settings.training_steps,
            model_config=model_config,
            batch_size=settings.batch_size,
            learning_rate=learning_rate,
            seed=training_seed,
        )
        curve_payload = _curve_payloads(
            curves,
            learner="winner",
            campaign=args.mode == "campaign",
            library_path=runtime_snapshot.path,
            seeds=development_seeds,
            horizons=settings.evaluation_horizons,
            factory=deployed_policy,
            base_evaluation=base_evaluation,
            catastrophic_survival_ratio=catastrophic_survival_ratio,
            catastrophic_survival_loss_ticks=(
                catastrophic_survival_loss_ticks
            ),
        )
        if args.mode == "campaign":
            winner_selection = _select_plateau_candidate(
                curve_payload,
                base_evaluation,
                selection_horizons,
            )
            selected_index = winner_selection["selected_curve_index"]
        else:
            selected_index = len(curves) - 1
            winner_selection = {
                "schema": "irisu-r3e-unevaluated-training-selection-v1",
                "selected_curve_index": selected_index,
                "accepted_learned_candidate": None,
                "retained_policy": "unevaluated_learned_candidate",
            }
        selected_model = (
            curves[selected_index] if selected_index is not None else None
        )
        checkpoint: dict[str, object] | None = None
        if selected_model is not None:
            geometry_sha = save_geometry_checkpoint(
                geometry_path,
                selected_model.model,
                search_identity=teacher.identity_manifest(),
                metadata={
                    "learner": "winner_classifier",
                    "source_identity_sha256": source_identity["sha256"],
                    "runtime_sha256": runtime_snapshot.sha256,
                    "base_policy_sha256": base_snapshot.sha256,
                    "collection_sha256": collection.sha256,
                    "dataset_sha256": collection.dataset.sha256,
                    "teacher_sha256": teacher.sha256,
                    "candidate_vocabulary_sha256": (
                        candidate_vocabulary_sha256
                    ),
                    "selected_training_steps": selected_model.steps,
                    "selection_rule": config["selection"]["ranking"],
                    "selection_horizon_ticks": selection_horizon,
                    "selection_status": (
                        "accepted_against_frozen_v5"
                        if args.mode == "campaign"
                        else "unevaluated_training_output"
                    ),
                },
                overwrite=args.overwrite,
            )
            checkpoint = {
                "path": str(geometry_path),
                "sha256": geometry_sha,
                "state_dict_sha256": selected_model.state_dict_sha256,
            }
            if args.mode == "campaign":
                selected_payloads["winner_classifier"] = curve_payload[
                    selected_index
                ]
        payload["training"] = {
            "learner": "winner_classifier",
            "protocol": config["training"]["curve_protocol"],
            "plateau_curve": curve_payload,
            "selected_curve_index": selected_index,
            "selected_training_steps": (
                selected_model.steps if selected_model is not None else None
            ),
            "selection_horizon_ticks": selection_horizon,
            "selection": winner_selection,
            "base_evaluation": base_evaluation,
            "checkpoint": checkpoint,
        }

    if (
        args.mode in {"train", "campaign"}
        and args.learner in {"ranking", "both"}
    ):
        assert ranking_collection is not None
        ranking_options = config["training"]["ranking"]
        if (
            ranking_options.get("all_available_candidate_outcomes") is not True
            or ranking_options.get("winner_classifier_retained_as_baseline")
            is not True
        ):
            raise ValueError("R3e ranking protocol flags are not enabled")
        ranking_curves = train_ranking_plateau_models(
            ranking_collection.dataset,
            budgets=settings.training_steps,
            model_config=model_config,
            batch_size=settings.batch_size,
            learning_rate=learning_rate,
            listwise_weight=float(ranking_options["listwise_weight"]),
            pairwise_weight=float(ranking_options["pairwise_weight"]),
            seed=training_seed,
        )
        ranking_curve_payload = _curve_payloads(
            ranking_curves,
            learner="ranking",
            campaign=args.mode == "campaign",
            library_path=runtime_snapshot.path,
            seeds=development_seeds,
            horizons=settings.evaluation_horizons,
            factory=deployed_policy,
            base_evaluation=base_evaluation,
            catastrophic_survival_ratio=catastrophic_survival_ratio,
            catastrophic_survival_loss_ticks=(
                catastrophic_survival_loss_ticks
            ),
        )
        if args.mode == "campaign":
            ranking_selection = _select_plateau_candidate(
                ranking_curve_payload,
                base_evaluation,
                selection_horizons,
            )
            ranking_selected_index = ranking_selection[
                "selected_curve_index"
            ]
        else:
            ranking_selected_index = len(ranking_curves) - 1
            ranking_selection = {
                "schema": "irisu-r3e-unevaluated-training-selection-v1",
                "selected_curve_index": ranking_selected_index,
                "accepted_learned_candidate": None,
                "retained_policy": "unevaluated_learned_candidate",
            }
        ranking_selected = (
            ranking_curves[ranking_selected_index]
            if ranking_selected_index is not None
            else None
        )
        ranking_checkpoint: dict[str, object] | None = None
        if ranking_selected is not None:
            ranking_geometry_sha = save_geometry_checkpoint(
                ranking_geometry_path,
                ranking_selected.model,
                search_identity=teacher.identity_manifest(),
                metadata={
                    "learner": "all_candidate_ranker",
                    "source_identity_sha256": source_identity["sha256"],
                    "runtime_sha256": runtime_snapshot.sha256,
                    "base_policy_sha256": base_snapshot.sha256,
                    "ranking_collection_sha256": ranking_collection.sha256,
                    "ranking_dataset_sha256": (
                        ranking_collection.dataset.sha256
                    ),
                    "teacher_sha256": teacher.sha256,
                    "candidate_vocabulary_sha256": (
                        candidate_vocabulary_sha256
                    ),
                    "selected_training_steps": ranking_selected.steps,
                    "ranking_loss": {
                        "listwise_weight": float(
                            ranking_options["listwise_weight"]
                        ),
                        "pairwise_weight": float(
                            ranking_options["pairwise_weight"]
                        ),
                    },
                    "selection_rule": config["selection"]["ranking"],
                    "selection_horizon_ticks": selection_horizon,
                    "selection_status": (
                        "accepted_against_frozen_v5"
                        if args.mode == "campaign"
                        else "unevaluated_training_output"
                    ),
                },
                overwrite=args.overwrite,
            )
            ranking_checkpoint = {
                "path": str(ranking_geometry_path),
                "sha256": ranking_geometry_sha,
                "state_dict_sha256": ranking_selected.state_dict_sha256,
            }
            if args.mode == "campaign":
                selected_payloads[
                    "all_candidate_ranker"
                ] = ranking_curve_payload[ranking_selected_index]
        payload["ranking_training"] = {
            "learner": "all_candidate_ranker",
            "protocol": config["training"]["curve_protocol"],
            "loss": {
                "listwise_weight": float(ranking_options["listwise_weight"]),
                "pairwise_weight": float(ranking_options["pairwise_weight"]),
            },
            "plateau_curve": ranking_curve_payload,
            "selected_curve_index": ranking_selected_index,
            "selected_training_steps": (
                ranking_selected.steps if ranking_selected is not None else None
            ),
            "selection_horizon_ticks": selection_horizon,
            "selection": ranking_selection,
            "base_evaluation": base_evaluation,
            "checkpoint": ranking_checkpoint,
        }

    if args.mode == "campaign" and selected_payloads:
        payload["selected_comparison"] = {
            str(horizon): {
                "base_v5": base_evaluation[str(horizon)]["aggregate"],
                **{
                    learner: point["evaluation"][str(horizon)]["aggregate"]
                    for learner, point in selected_payloads.items()
                },
            }
            for horizon in settings.evaluation_horizons
        }
        payload["selected_paired_safety"] = {
            str(horizon): {
                learner: point["evaluation"][str(horizon)]["paired_safety"]
                for learner, point in selected_payloads.items()
            }
            for horizon in settings.evaluation_horizons
        }

    if args.mode == "evaluate":
        checkpoints: dict[str, GeometryCheckpointArtifact] = {}
        for learner, path, supplied_sha in (
            ("winner_classifier", geometry_path, args.geometry_sha256),
            (
                "all_candidate_ranker",
                ranking_geometry_path,
                args.ranking_geometry_sha256,
            ),
        ):
            requested = (
                args.learner == "both"
                or args.learner == "winner"
                and learner == "winner_classifier"
                or args.learner == "ranking"
                and learner == "all_candidate_ranker"
            )
            if not requested:
                continue
            if supplied_sha is None:
                parser.error(f"{learner} checkpoint SHA-256 is required")
            checkpoint = load_geometry_checkpoint(
                path, expected_sha256=supplied_sha
            )
            _require_bound_metadata(
                checkpoint.metadata,
                {**artifact_bindings, "learner": learner},
                f"{learner} checkpoint",
            )
            if (
                checkpoint.model.candidate_set_sha256
                != candidate_vocabulary_sha256
                or checkpoint.model.candidate_count
                != teacher.config.candidate_config.slot_count
                or checkpoint.search_identity != teacher.identity_manifest()
            ):
                raise ValueError(
                    f"R3e {learner} checkpoint and configured teacher differ"
                )
            checkpoints[learner] = checkpoint
        evaluations: dict[str, object] = {}
        for horizon in settings.evaluation_horizons:
            values: dict[str, object] = {
                "base_v5": evaluate_policy(
                    label="r3e_base_v5",
                    library_path=runtime_snapshot.path,
                    seeds=development_seeds,
                    horizon_ticks=horizon,
                    factory=lambda: _base_policy(
                        base_snapshot.path,
                        base_snapshot.sha256,
                        base_options,
                    ),
                )
            }
            for learner, checkpoint in checkpoints.items():
                values[learner] = evaluate_policy(
                    label=f"r3e_{learner}",
                    library_path=runtime_snapshot.path,
                    seeds=development_seeds,
                    horizon_ticks=horizon,
                    factory=lambda model=checkpoint.model: deployed_policy(
                        model
                    ),
                )
            evaluations[str(horizon)] = values
        payload["evaluation"] = {
            "checkpoints": {
                learner: {
                    "path": str(checkpoint.path),
                    "sha256": checkpoint.sha256,
                }
                for learner, checkpoint in checkpoints.items()
            },
            "horizons": evaluations,
        }

    _require_source_identity(source_identity, config_snapshot.path)
    r3d._require_unchanged(config_snapshot, "R3e config")
    r3d._require_unchanged(runtime_snapshot, "R3e portable runtime")
    r3d._require_unchanged(base_snapshot, "R3e base policy")
    report = _final_report(
        payload,
        source_identity=source_identity,
        config_snapshot=config_snapshot,
        runtime_snapshot=runtime_snapshot,
        base_snapshot=base_snapshot,
        started=started,
    )
    _atomic_write_json(result_path, report, overwrite=args.overwrite)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
