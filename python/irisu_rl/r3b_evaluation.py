"""Raw-score-only fixed-cell evaluation for R3b policies and baselines."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from functools import lru_cache
from dataclasses import asdict, dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import irisu_env
import numpy as np
import torch
from torch import Tensor

from irisu_env import Action, ActionKind
from irisu_env.policies import (
    DirectMatcherPolicy,
    ImminentRotHazardPolicy,
    LongWaitPolicy,
    MatcherShotPolicy,
    RandomPolicy,
    SideEjectorPolicy,
)

from .actions import ActionSpec, SemanticAction, SemanticActionKind
from .collector import model_state_sha256
from .curriculum import SnapshotBlobStore, SnapshotLibrary, SnapshotRecipe
from .encoding import EncodedBatch
from .models import RecurrentActorCritic
from .runtime_identity import attest_simulator_runtime


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_object(value: object, label: str) -> object:
    try:
        return json.loads(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite canonical JSON") from exc


def _manifest_mapping(
    value: object, expected: set[str], *, location: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{location} must be a string-keyed object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{location} keys differ: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return value


def _manifest_list(value: object, *, location: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{location} must be an array")
    return value


def _require_manifest_round_trip(
    original: Mapping[str, object],
    reconstructed: Mapping[str, object],
    *,
    location: str,
) -> None:
    try:
        left = json.dumps(
            original, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        right = json.dumps(
            reconstructed, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be finite JSON") from exc
    if left != right:
        raise ValueError(f"{location} is not a canonical serialization")


def _stable_file_manifest(path: Path, logical_path: str) -> dict[str, object]:
    try:
        resolved = path.resolve(strict=True)
        if path.is_symlink() or not resolved.is_file():
            raise ValueError(f"build input {logical_path} must be a regular owned file")
        before = resolved.stat()
        payload = resolved.read_bytes()
        after = resolved.stat()
    except OSError as exc:
        raise RuntimeError(f"cannot capture build input {logical_path}: {exc}") from exc

    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if identity(before) != identity(after) or len(payload) != after.st_size:
        raise RuntimeError(f"build input {logical_path} changed while hashing")
    return {
        "path": logical_path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _default_source_roots() -> dict[str, Path]:
    rl_root = Path(__file__).resolve().parent
    env_file = getattr(irisu_env, "__file__", None)
    if not isinstance(env_file, str):
        raise RuntimeError("irisu_env source location is unavailable")
    return {"irisu_env": Path(env_file).resolve().parent, "irisu_rl": rl_root}


def _default_dependency_inputs() -> dict[str, Path]:
    for parent in Path(__file__).resolve().parents:
        lock = parent / "uv.lock"
        project = parent / "pyproject.toml"
        if lock.is_file() and project.is_file():
            return {"pyproject.toml": project, "uv.lock": lock}
    raise RuntimeError("versioned dependency inputs are unavailable")


def _capture_build_files(
    roots: Mapping[str, str | os.PathLike[str]],
    inputs: Mapping[str, str | os.PathLike[str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    source_files: list[dict[str, object]] = []
    for name, raw_root in sorted(roots.items()):
        root = Path(raw_root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError(f"source root {name} is not a directory")
        files = sorted(root.rglob("*.py"))
        if not files:
            raise ValueError(f"source root {name} contains no Python modules")
        for path in files:
            source_files.append(
                _stable_file_manifest(
                    path, f"{name}/{path.relative_to(root).as_posix()}"
                )
            )
    dependencies = [
        _stable_file_manifest(Path(path), name) for name, path in sorted(inputs.items())
    ]
    return source_files, dependencies


@lru_cache(maxsize=1)
def _default_build_files_json() -> str:
    source_files, dependencies = _capture_build_files(
        _default_source_roots(), _default_dependency_inputs()
    )
    return json.dumps(
        {"source_files": source_files, "dependency_inputs": dependencies},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def encoder_instance_manifest(encoder: object) -> dict[str, object]:
    """Return the complete declared configuration of one evaluation encoder."""

    schema = getattr(encoder, "schema", None)
    schema_manifest = getattr(schema, "manifest", None)
    if schema is None or not callable(schema_manifest):
        raise ValueError("encoder must expose a manifested schema")
    declared = getattr(encoder, "manifest", None)
    if callable(declared):
        configuration = declared()
    else:
        try:
            configuration = vars(encoder)
        except TypeError:
            configuration = {}
    manifest = {
        "version": "r3b-encoder-instance-v1",
        "class": f"{type(encoder).__module__}.{type(encoder).__qualname__}",
        "schema": schema_manifest(),
        "configuration": configuration,
    }
    canonical = _canonical_json_object(manifest, "encoder instance manifest")
    if not isinstance(canonical, dict):
        raise TypeError("encoder instance manifest must be an object")
    return canonical


def behavior_build_identity_manifest(
    configuration: Mapping[str, object],
    *,
    source_roots: Mapping[str, str | os.PathLike[str]] | None = None,
    dependency_inputs: Mapping[str, str | os.PathLike[str]] | None = None,
) -> dict[str, object]:
    """Capture source, dependencies, runtime, and deterministic execution settings."""

    if not isinstance(configuration, Mapping) or any(
        not isinstance(key, str) for key in configuration
    ):
        raise TypeError("build configuration must be a string-keyed mapping")
    use_defaults = source_roots is None and dependency_inputs is None
    roots = _default_source_roots() if source_roots is None else dict(source_roots)
    inputs = (
        _default_dependency_inputs()
        if dependency_inputs is None
        else dict(dependency_inputs)
    )
    if (
        not roots
        or not inputs
        or any(not isinstance(name, str) or not name for name in (*roots, *inputs))
    ):
        raise ValueError("build identity requires named source and dependency inputs")

    if use_defaults:
        captured = json.loads(_default_build_files_json())
        source_files = captured["source_files"]
        dependencies = captured["dependency_inputs"]
    else:
        source_files, dependencies = _capture_build_files(roots, inputs)
    torch_build = torch.__config__.show()
    runtime = {
        "python": sys.version,
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_build_config_sha256": hashlib.sha256(torch_build.encode()).hexdigest(),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }
    deterministic = {
        "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "algorithms_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "default_dtype": str(torch.get_default_dtype()),
        "num_threads": torch.get_num_threads(),
        "num_interop_threads": torch.get_num_interop_threads(),
    }
    manifest = {
        "version": "r3b-behavior-build-identity-v1",
        "source_files": source_files,
        "dependency_inputs": dependencies,
        "runtime": runtime,
        "deterministic_settings": deterministic,
        "configuration": dict(configuration),
    }
    canonical = _canonical_json_object(manifest, "behavior build identity")
    if not isinstance(canonical, dict):
        raise TypeError("behavior build identity must be an object")
    return canonical


def behavior_build_identity_sha256(
    configuration: Mapping[str, object],
    *,
    source_roots: Mapping[str, str | os.PathLike[str]] | None = None,
    dependency_inputs: Mapping[str, str | os.PathLike[str]] | None = None,
) -> str:
    return _canonical_sha256(
        behavior_build_identity_manifest(
            configuration,
            source_roots=source_roots,
            dependency_inputs=dependency_inputs,
        )
    )


@lru_cache(maxsize=128)
def _episode_content_sha256(episodes: tuple[EpisodeMetrics, ...]) -> str:
    return _canonical_sha256([asdict(value) for value in episodes])


@dataclass(frozen=True, slots=True)
class LogicalEvaluationCell:
    """Backend-neutral construction provenance for one parity cell."""

    split: str
    stage_id: str
    scenario_family: str
    config_sha256: str
    config_hash: int
    reset_seed: int
    action_spec_sha256: str
    semantic_actions_hex: tuple[str, ...]
    expected_tick: int
    expected_score: int
    version: str = "r3b-logical-evaluation-cell-v2"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-logical-evaluation-cell-v2"
            or self.split not in {"calibration", "validation", "test"}
            or not isinstance(self.stage_id, str)
            or not self.stage_id
            or not isinstance(self.scenario_family, str)
            or not self.scenario_family
            or not _is_sha256(self.config_sha256)
            or self.config_sha256 == "0" * 64
            or isinstance(self.config_hash, bool)
            or not isinstance(self.config_hash, Integral)
            or not 0 <= self.config_hash < 2**64
            or not _is_sha256(self.action_spec_sha256)
            or self.action_spec_sha256 == "0" * 64
            or isinstance(self.reset_seed, bool)
            or not isinstance(self.reset_seed, Integral)
            or not 0 <= self.reset_seed < 2**32
            or not isinstance(self.semantic_actions_hex, tuple)
            or any(not isinstance(value, str) for value in self.semantic_actions_hex)
            or isinstance(self.expected_tick, bool)
            or not isinstance(self.expected_tick, Integral)
            or self.expected_tick < 0
            or isinstance(self.expected_score, bool)
            or not isinstance(self.expected_score, Integral)
        ):
            raise ValueError("logical evaluation cell provenance is malformed")

    @classmethod
    def from_recipe(cls, recipe: SnapshotRecipe) -> LogicalEvaluationCell:
        split = "calibration" if recipe.split == "train" else recipe.split
        return cls(
            split,
            recipe.stage_id,
            recipe.scenario_family,
            recipe.config_sha256,
            int(recipe.config_hash),
            int(recipe.reset_seed),
            recipe.action_spec_sha256,
            recipe.semantic_actions_hex,
            int(recipe.expected_tick),
            int(recipe.expected_score),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "split": self.split,
            "stage_id": self.stage_id,
            "scenario_family": self.scenario_family,
            "config_sha256": self.config_sha256,
            "config_hash": int(self.config_hash),
            "reset_seed": int(self.reset_seed),
            "action_spec_sha256": self.action_spec_sha256,
            "semantic_actions_hex": list(self.semantic_actions_hex),
            "expected_tick": int(self.expected_tick),
            "expected_score": int(self.expected_score),
        }

    @classmethod
    def from_manifest(cls, value: object) -> LogicalEvaluationCell:
        manifest = _manifest_mapping(
            value,
            {
                "version",
                "split",
                "stage_id",
                "scenario_family",
                "config_sha256",
                "config_hash",
                "reset_seed",
                "action_spec_sha256",
                "semantic_actions_hex",
                "expected_tick",
                "expected_score",
            },
            location="logical evaluation cell",
        )
        trace = _manifest_list(
            manifest["semantic_actions_hex"],
            location="logical evaluation cell semantic_actions_hex",
        )
        if (
            any(type(item) is not str for item in trace)
            or any(
                type(manifest[name]) is not str
                for name in (
                    "version",
                    "split",
                    "stage_id",
                    "scenario_family",
                    "config_sha256",
                    "action_spec_sha256",
                )
            )
            or type(manifest["config_hash"]) is not int
            or type(manifest["reset_seed"]) is not int
            or type(manifest["expected_tick"]) is not int
            or type(manifest["expected_score"]) is not int
        ):
            raise ValueError("logical evaluation cell field types are malformed")
        try:
            result = cls(
                split=manifest["split"],  # type: ignore[arg-type]
                stage_id=manifest["stage_id"],  # type: ignore[arg-type]
                scenario_family=manifest["scenario_family"],  # type: ignore[arg-type]
                config_sha256=manifest["config_sha256"],  # type: ignore[arg-type]
                config_hash=manifest["config_hash"],  # type: ignore[arg-type]
                reset_seed=manifest["reset_seed"],  # type: ignore[arg-type]
                action_spec_sha256=manifest["action_spec_sha256"],  # type: ignore[arg-type]
                semantic_actions_hex=tuple(trace),  # type: ignore[arg-type]
                expected_tick=manifest["expected_tick"],  # type: ignore[arg-type]
                expected_score=manifest["expected_score"],  # type: ignore[arg-type]
                version=manifest["version"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("logical evaluation cell is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="logical evaluation cell"
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class CrossBackendCellPair:
    logical_cell: LogicalEvaluationCell
    portable_snapshot_id: str
    exact_snapshot_id: str
    portable_recipe_sha256: str
    exact_recipe_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.logical_cell, LogicalEvaluationCell)
            or not self.portable_snapshot_id
            or not self.exact_snapshot_id
            or self.portable_snapshot_id == self.exact_snapshot_id
            or not _is_sha256(self.portable_recipe_sha256)
            or self.portable_recipe_sha256 == "0" * 64
            or not _is_sha256(self.exact_recipe_sha256)
            or self.exact_recipe_sha256 == "0" * 64
        ):
            raise ValueError("cross-backend cell pair is malformed")

    @classmethod
    def from_recipes(
        cls, portable: SnapshotRecipe, exact: SnapshotRecipe
    ) -> CrossBackendCellPair:
        portable_logical = LogicalEvaluationCell.from_recipe(portable)
        exact_logical = LogicalEvaluationCell.from_recipe(exact)
        if portable_logical != exact_logical:
            raise ValueError("portable/exact recipes do not share canonical provenance")
        if (
            portable.runtime_identity_sha256 == exact.runtime_identity_sha256
            or portable.snapshot_sha256 == exact.snapshot_sha256
        ):
            raise ValueError("cross-backend recipes are not physically independent")
        return cls(
            portable_logical,
            portable.snapshot_id,
            exact.snapshot_id,
            portable.sha256,
            exact.sha256,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "logical_cell": self.logical_cell.manifest(),
            "logical_cell_sha256": self.logical_cell.sha256,
            "portable_snapshot_id": self.portable_snapshot_id,
            "exact_snapshot_id": self.exact_snapshot_id,
            "portable_recipe_sha256": self.portable_recipe_sha256,
            "exact_recipe_sha256": self.exact_recipe_sha256,
        }

    @classmethod
    def from_manifest(cls, value: object) -> CrossBackendCellPair:
        manifest = _manifest_mapping(
            value,
            {
                "logical_cell",
                "logical_cell_sha256",
                "portable_snapshot_id",
                "exact_snapshot_id",
                "portable_recipe_sha256",
                "exact_recipe_sha256",
            },
            location="cross-backend cell pair",
        )
        logical_cell = LogicalEvaluationCell.from_manifest(manifest["logical_cell"])
        if (
            any(
                type(manifest[name]) is not str
                for name in (
                    "logical_cell_sha256",
                    "portable_snapshot_id",
                    "exact_snapshot_id",
                    "portable_recipe_sha256",
                    "exact_recipe_sha256",
                )
            )
            or manifest["logical_cell_sha256"] != logical_cell.sha256
        ):
            raise ValueError("cross-backend cell pair logical-cell hash mismatch")
        try:
            result = cls(
                logical_cell,
                manifest["portable_snapshot_id"],  # type: ignore[arg-type]
                manifest["exact_snapshot_id"],  # type: ignore[arg-type]
                manifest["portable_recipe_sha256"],  # type: ignore[arg-type]
                manifest["exact_recipe_sha256"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("cross-backend cell pair is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="cross-backend cell pair"
        )
        return result


@dataclass(frozen=True, slots=True)
class CrossBackendEvaluationManifest:
    pairs: tuple[CrossBackendCellPair, ...]
    version: str = "r3b-cross-backend-evaluation-manifest-v2"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-cross-backend-evaluation-manifest-v2"
            or not isinstance(self.pairs, tuple)
            or not self.pairs
            or any(not isinstance(value, CrossBackendCellPair) for value in self.pairs)
            or len({value.logical_cell.sha256 for value in self.pairs})
            != len(self.pairs)
            or len({value.portable_snapshot_id for value in self.pairs})
            != len(self.pairs)
            or len({value.exact_snapshot_id for value in self.pairs}) != len(self.pairs)
            or len({value.logical_cell.split for value in self.pairs}) != 1
            or len({value.logical_cell.action_spec_sha256 for value in self.pairs}) != 1
        ):
            raise ValueError("cross-backend evaluation manifest is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "pairs": [value.manifest() for value in self.pairs],
        }

    @classmethod
    def from_manifest(cls, value: object) -> CrossBackendEvaluationManifest:
        manifest = _manifest_mapping(
            value,
            {"version", "pairs"},
            location="cross-backend evaluation manifest",
        )
        pairs = _manifest_list(
            manifest["pairs"], location="cross-backend evaluation manifest pairs"
        )
        try:
            result = cls(
                tuple(CrossBackendCellPair.from_manifest(pair) for pair in pairs),
                manifest["version"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("cross-backend evaluation manifest is malformed") from exc
        _require_manifest_round_trip(
            manifest,
            result.manifest(),
            location="cross-backend evaluation manifest",
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    suite_id: str
    split: str
    snapshot_ids: tuple[str, ...]
    repetitions: int
    policy_seed: int
    max_decisions: int
    max_simulated_ticks: int
    runtime_identity_sha256: str
    assignment_sha256: str
    library_sha256: str
    snapshot_store_sha256: str
    action_spec_sha256: str
    recipe_sha256s: tuple[str, ...]
    logical_cell_ids: tuple[str, ...] = ()
    backend: str = "portable"
    logical_manifest_sha256: str | None = None
    version: str = "r3b-evaluation-suite-v4"
    _sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-evaluation-suite-v4"
            or not self.suite_id
            or not self.suite_id.isascii()
            or self.split not in {"calibration", "validation", "test"}
            or not isinstance(self.snapshot_ids, tuple)
            or not self.snapshot_ids
            or len(set(self.snapshot_ids)) != len(self.snapshot_ids)
            or any(not value for value in self.snapshot_ids)
            or not isinstance(self.recipe_sha256s, tuple)
            or len(self.recipe_sha256s) != len(self.snapshot_ids)
            or any(
                not _is_sha256(value) or value == "0" * 64
                for value in self.recipe_sha256s
            )
            or self.backend not in {"portable", "exact"}
            or (
                self.logical_manifest_sha256 is not None
                and (
                    not _is_sha256(self.logical_manifest_sha256)
                    or self.logical_manifest_sha256 == "0" * 64
                )
            )
        ):
            raise ValueError("evaluation suite identity is invalid")
        if not self.logical_cell_ids:
            object.__setattr__(self, "logical_cell_ids", self.snapshot_ids)
        if (
            len(self.logical_cell_ids) != len(self.snapshot_ids)
            or len(set(self.logical_cell_ids)) != len(self.logical_cell_ids)
            or any(not value for value in self.logical_cell_ids)
        ):
            raise ValueError("evaluation suite logical cells are invalid")
        for name in ("repetitions", "max_decisions", "max_simulated_ticks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.policy_seed, bool)
            or not isinstance(self.policy_seed, Integral)
            or not 0 <= self.policy_seed < 2**64
            or not _is_sha256(self.runtime_identity_sha256)
            or any(
                not _is_sha256(value) or value == "0" * 64
                for value in (
                    self.runtime_identity_sha256,
                    self.assignment_sha256,
                    self.library_sha256,
                    self.snapshot_store_sha256,
                    self.action_spec_sha256,
                )
            )
        ):
            raise ValueError("evaluation suite seed or identity is invalid")
        object.__setattr__(self, "_sha256", _canonical_sha256(self.manifest()))

    def manifest(self) -> dict[str, object]:
        return {
            "suite_id": self.suite_id,
            "split": self.split,
            "snapshot_ids": list(self.snapshot_ids),
            "repetitions": self.repetitions,
            "policy_seed": self.policy_seed,
            "max_decisions": self.max_decisions,
            "max_simulated_ticks": self.max_simulated_ticks,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "assignment_sha256": self.assignment_sha256,
            "library_sha256": self.library_sha256,
            "snapshot_store_sha256": self.snapshot_store_sha256,
            "action_spec_sha256": self.action_spec_sha256,
            "recipe_sha256s": list(self.recipe_sha256s),
            "logical_cell_ids": list(self.logical_cell_ids),
            "backend": self.backend,
            "logical_manifest_sha256": self.logical_manifest_sha256,
            "version": self.version,
        }

    @classmethod
    def from_manifest(cls, value: object) -> EvaluationSuite:
        manifest = _manifest_mapping(
            value,
            {
                "suite_id",
                "split",
                "snapshot_ids",
                "repetitions",
                "policy_seed",
                "max_decisions",
                "max_simulated_ticks",
                "runtime_identity_sha256",
                "assignment_sha256",
                "library_sha256",
                "snapshot_store_sha256",
                "action_spec_sha256",
                "recipe_sha256s",
                "logical_cell_ids",
                "backend",
                "logical_manifest_sha256",
                "version",
            },
            location="evaluation suite",
        )
        snapshots = _manifest_list(
            manifest["snapshot_ids"], location="evaluation suite snapshot_ids"
        )
        recipes = _manifest_list(
            manifest["recipe_sha256s"], location="evaluation suite recipe_sha256s"
        )
        logical_cells = _manifest_list(
            manifest["logical_cell_ids"],
            location="evaluation suite logical_cell_ids",
        )
        if (
            any(
                type(item) is not str for item in (*snapshots, *recipes, *logical_cells)
            )
            or any(
                type(manifest[name]) is not str
                for name in (
                    "suite_id",
                    "split",
                    "runtime_identity_sha256",
                    "assignment_sha256",
                    "library_sha256",
                    "snapshot_store_sha256",
                    "action_spec_sha256",
                    "backend",
                    "version",
                )
            )
            or any(
                type(manifest[name]) is not int
                for name in (
                    "repetitions",
                    "policy_seed",
                    "max_decisions",
                    "max_simulated_ticks",
                )
            )
            or (
                manifest["logical_manifest_sha256"] is not None
                and type(manifest["logical_manifest_sha256"]) is not str
            )
        ):
            raise ValueError("evaluation suite field types are malformed")
        try:
            result = cls(
                suite_id=manifest["suite_id"],  # type: ignore[arg-type]
                split=manifest["split"],  # type: ignore[arg-type]
                snapshot_ids=tuple(snapshots),  # type: ignore[arg-type]
                repetitions=manifest["repetitions"],  # type: ignore[arg-type]
                policy_seed=manifest["policy_seed"],  # type: ignore[arg-type]
                max_decisions=manifest["max_decisions"],  # type: ignore[arg-type]
                max_simulated_ticks=manifest["max_simulated_ticks"],  # type: ignore[arg-type]
                runtime_identity_sha256=manifest["runtime_identity_sha256"],  # type: ignore[arg-type]
                assignment_sha256=manifest["assignment_sha256"],  # type: ignore[arg-type]
                library_sha256=manifest["library_sha256"],  # type: ignore[arg-type]
                snapshot_store_sha256=manifest["snapshot_store_sha256"],  # type: ignore[arg-type]
                action_spec_sha256=manifest["action_spec_sha256"],  # type: ignore[arg-type]
                recipe_sha256s=tuple(recipes),  # type: ignore[arg-type]
                logical_cell_ids=tuple(logical_cells),  # type: ignore[arg-type]
                backend=manifest["backend"],  # type: ignore[arg-type]
                logical_manifest_sha256=manifest["logical_manifest_sha256"],  # type: ignore[arg-type]
                version=manifest["version"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation suite is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="evaluation suite"
        )
        return result

    @property
    def sha256(self) -> str:
        return self._sha256

    def episode_seed(self, snapshot_id: str, repetition: int) -> int:
        if (
            snapshot_id not in self.snapshot_ids
            or not 0 <= repetition < self.repetitions
        ):
            raise ValueError("evaluation cell is outside the suite")
        logical_id = self.logical_cell_ids[self.snapshot_ids.index(snapshot_id)]
        payload = f"r3b-cell-v1:{self.policy_seed}:{logical_id}:{repetition}".encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    snapshot_id: str
    repetition: int
    policy_seed: int
    initial_score: int
    final_score: int
    raw_score: int
    elapsed_ticks: int
    decisions: int
    terminated: bool
    truncated: bool
    invalid_actions: int
    minimum_gauge: int
    final_gauge: int

    def __post_init__(self) -> None:
        integer_fields = (
            self.repetition,
            self.policy_seed,
            self.initial_score,
            self.final_score,
            self.raw_score,
            self.elapsed_ticks,
            self.decisions,
            self.invalid_actions,
            self.minimum_gauge,
            self.final_gauge,
        )
        if (
            not self.snapshot_id
            or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in integer_fields
            )
            or self.repetition < 0
            or not 0 <= self.policy_seed < 2**64
            or self.elapsed_ticks < 0
            or self.decisions < 0
            or self.invalid_actions < 0
            # The exact runtime stores gauge as a signed int64. Rot damage
            # happens after the scene gauge floor, so a live observation can
            # remain negative until the next tick latches game over. That
            # terminal tick can also end negative because actor updates
            # continue after the latch.
            or not -(2**63) <= self.minimum_gauge < 2**63
            or self.minimum_gauge > self.final_gauge
            or not -(2**63) <= self.final_gauge < 2**63
            or self.raw_score != self.final_score - self.initial_score
            or not isinstance(self.terminated, bool)
            or not isinstance(self.truncated, bool)
        ):
            raise ValueError("evaluation episode metrics are malformed")

    @classmethod
    def from_manifest(cls, value: object) -> EpisodeMetrics:
        expected = {
            "snapshot_id",
            "repetition",
            "policy_seed",
            "initial_score",
            "final_score",
            "raw_score",
            "elapsed_ticks",
            "decisions",
            "terminated",
            "truncated",
            "invalid_actions",
            "minimum_gauge",
            "final_gauge",
        }
        manifest = _manifest_mapping(value, expected, location="episode metrics")
        integer_fields = expected - {"snapshot_id", "terminated", "truncated"}
        if (
            type(manifest["snapshot_id"]) is not str
            or any(type(manifest[name]) is not int for name in integer_fields)
            or type(manifest["terminated"]) is not bool
            or type(manifest["truncated"]) is not bool
        ):
            raise ValueError("episode metrics field types are malformed")
        try:
            result = cls(**manifest)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("episode metrics are malformed") from exc
        _require_manifest_round_trip(
            manifest, asdict(result), location="episode metrics"
        )
        return result


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    suite_sha256: str
    policy_sha256: str
    evaluator_sha256: str
    backend_identity_sha256: str
    execution_identity_sha256: str
    episodes: tuple[EpisodeMetrics, ...]
    version: str = "r3b-evaluation-report-v3"
    _sha256: str = field(init=False, repr=False, compare=False)
    _episode_content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-evaluation-report-v3"
            or not isinstance(self.episodes, tuple)
            or not all(
                _is_sha256(value)
                for value in (
                    self.suite_sha256,
                    self.policy_sha256,
                    self.evaluator_sha256,
                    self.backend_identity_sha256,
                    self.execution_identity_sha256,
                )
            )
            or any(
                value == "0" * 64
                for value in (
                    self.suite_sha256,
                    self.policy_sha256,
                    self.evaluator_sha256,
                    self.backend_identity_sha256,
                    self.execution_identity_sha256,
                )
            )
            or not self.episodes
            or len({(value.snapshot_id, value.repetition) for value in self.episodes})
            != len(self.episodes)
        ):
            raise ValueError("evaluation report identity or cells are malformed")
        object.__setattr__(
            self,
            "_episode_content_sha256",
            _episode_content_sha256(self.episodes),
        )
        object.__setattr__(
            self,
            "_sha256",
            _canonical_sha256(self.manifest()),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "suite_sha256": self.suite_sha256,
            "policy_sha256": self.policy_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "backend_identity_sha256": self.backend_identity_sha256,
            "execution_identity_sha256": self.execution_identity_sha256,
            "episodes": [asdict(value) for value in self.episodes],
        }

    @classmethod
    def from_manifest(
        cls,
        value: object,
        *,
        suite: EvaluationSuite | None = None,
    ) -> EvaluationReport:
        manifest = _manifest_mapping(
            value,
            {
                "version",
                "suite_sha256",
                "policy_sha256",
                "evaluator_sha256",
                "backend_identity_sha256",
                "execution_identity_sha256",
                "episodes",
            },
            location="evaluation report",
        )
        episodes = _manifest_list(
            manifest["episodes"], location="evaluation report episodes"
        )
        if any(
            type(manifest[name]) is not str
            for name in (
                "version",
                "suite_sha256",
                "policy_sha256",
                "evaluator_sha256",
                "backend_identity_sha256",
                "execution_identity_sha256",
            )
        ):
            raise ValueError("evaluation report field types are malformed")
        if suite is not None and (
            not isinstance(suite, EvaluationSuite)
            or manifest["suite_sha256"] != suite.sha256
            or manifest["backend_identity_sha256"] != suite.runtime_identity_sha256
        ):
            raise ValueError("evaluation report suite reference mismatch")
        try:
            result = cls(
                suite_sha256=manifest["suite_sha256"],  # type: ignore[arg-type]
                policy_sha256=manifest["policy_sha256"],  # type: ignore[arg-type]
                evaluator_sha256=manifest["evaluator_sha256"],  # type: ignore[arg-type]
                backend_identity_sha256=manifest["backend_identity_sha256"],  # type: ignore[arg-type]
                execution_identity_sha256=manifest["execution_identity_sha256"],  # type: ignore[arg-type]
                episodes=tuple(EpisodeMetrics.from_manifest(item) for item in episodes),
                version=manifest["version"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation report is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="evaluation report"
        )
        return result

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def episode_content_sha256(self) -> str:
        return self._episode_content_sha256


def _cross_backend_episode_diagnostics(
    portable_report: EvaluationReport,
    portable_suite: EvaluationSuite,
    exact_report: EvaluationReport,
    exact_suite: EvaluationSuite,
) -> tuple[dict[str, object], ...]:
    """Return exact-minus-portable deltas without claiming byte-identical physics."""

    def indexed(
        report: EvaluationReport, suite: EvaluationSuite
    ) -> dict[tuple[str, int], EpisodeMetrics]:
        logical = dict(zip(suite.snapshot_ids, suite.logical_cell_ids))
        return {
            (logical[value.snapshot_id], value.repetition): value
            for value in report.episodes
        }

    portable = indexed(portable_report, portable_suite)
    exact = indexed(exact_report, exact_suite)
    if set(portable) != set(exact):
        raise ValueError("cross-backend reports do not cover the same logical cells")
    diagnostics: list[dict[str, object]] = []
    integer_fields = (
        "initial_score",
        "final_score",
        "raw_score",
        "elapsed_ticks",
        "decisions",
        "invalid_actions",
        "minimum_gauge",
        "final_gauge",
    )
    for logical_id in portable_suite.logical_cell_ids:
        for repetition in range(portable_suite.repetitions):
            key = (logical_id, repetition)
            left, right = portable[key], exact[key]
            if left.policy_seed != right.policy_seed:
                raise ValueError("cross-backend policy seeds differ")
            if left.initial_score != right.initial_score:
                raise ValueError("cross-backend cells start at different raw scores")
            if left.invalid_actions or right.invalid_actions:
                raise ValueError("cross-backend evaluation produced an invalid action")
            left_content = {**asdict(left), "snapshot_id": logical_id}
            right_content = {**asdict(right), "snapshot_id": logical_id}
            diagnostics.append(
                {
                    "logical_cell_sha256": logical_id,
                    "repetition": repetition,
                    "policy_seed": int(left.policy_seed),
                    "exact_minus_portable": {
                        name: int(getattr(right, name)) - int(getattr(left, name))
                        for name in integer_fields
                    },
                    "portable_outcome": {
                        "terminated": left.terminated,
                        "truncated": left.truncated,
                    },
                    "exact_outcome": {
                        "terminated": right.terminated,
                        "truncated": right.truncated,
                    },
                    "portable_episode_sha256": _canonical_sha256(left_content),
                    "exact_episode_sha256": _canonical_sha256(right_content),
                }
            )
    return tuple(diagnostics)


@dataclass(frozen=True, slots=True)
class LearnedPolicyBackendParityArtifact:
    """A learned policy evaluated identically on anchored portable/exact cells."""

    portable_suite: EvaluationSuite
    portable_report: EvaluationReport
    exact_suite: EvaluationSuite
    exact_report: EvaluationReport
    logical_manifest: CrossBackendEvaluationManifest
    portable_library: SnapshotLibrary
    exact_library: SnapshotLibrary
    version: str = "r3b-learned-policy-backend-parity-v2"
    _diagnostics: tuple[dict[str, object], ...] = field(
        init=False, repr=False, compare=False
    )
    _diagnostics_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-learned-policy-backend-parity-v2"
            or not isinstance(self.portable_suite, EvaluationSuite)
            or not isinstance(self.portable_report, EvaluationReport)
            or not isinstance(self.exact_suite, EvaluationSuite)
            or not isinstance(self.exact_report, EvaluationReport)
            or not isinstance(self.logical_manifest, CrossBackendEvaluationManifest)
            or not isinstance(self.portable_library, SnapshotLibrary)
            or not isinstance(self.exact_library, SnapshotLibrary)
        ):
            raise ValueError("learned-policy parity artifact is malformed")
        if (
            self.portable_suite.backend != "portable"
            or self.exact_suite.backend != "exact"
            or self.portable_report.suite_sha256 != self.portable_suite.sha256
            or self.exact_report.suite_sha256 != self.exact_suite.sha256
            or self.portable_report.policy_sha256 != self.exact_report.policy_sha256
            or self.portable_report.evaluator_sha256
            != self.exact_report.evaluator_sha256
            or self.portable_report.backend_identity_sha256
            != self.portable_suite.runtime_identity_sha256
            or self.exact_report.backend_identity_sha256
            != self.exact_suite.runtime_identity_sha256
            or self.portable_suite.runtime_identity_sha256
            == self.exact_suite.runtime_identity_sha256
            or self.portable_suite.library_sha256 != self.portable_library.sha256
            or self.exact_suite.library_sha256 != self.exact_library.sha256
            or self.portable_suite.logical_manifest_sha256
            != self.logical_manifest.sha256
            or self.exact_suite.logical_manifest_sha256 != self.logical_manifest.sha256
        ):
            raise ValueError("learned-policy reports lack independent backend identity")
        pairs = self.logical_manifest.pairs
        logical_ids = tuple(pair.logical_cell.sha256 for pair in pairs)
        if (
            self.portable_suite.snapshot_ids
            != tuple(pair.portable_snapshot_id for pair in pairs)
            or self.exact_suite.snapshot_ids
            != tuple(pair.exact_snapshot_id for pair in pairs)
            or self.portable_suite.recipe_sha256s
            != tuple(pair.portable_recipe_sha256 for pair in pairs)
            or self.exact_suite.recipe_sha256s
            != tuple(pair.exact_recipe_sha256 for pair in pairs)
            or self.portable_suite.recipe_sha256s
            != tuple(
                self.portable_library[snapshot_id].sha256
                for snapshot_id in self.portable_suite.snapshot_ids
            )
            or self.exact_suite.recipe_sha256s
            != tuple(
                self.exact_library[snapshot_id].sha256
                for snapshot_id in self.exact_suite.snapshot_ids
            )
            or self.portable_suite.logical_cell_ids != logical_ids
            or self.exact_suite.logical_cell_ids != logical_ids
            or (
                self.portable_suite.split,
                self.portable_suite.repetitions,
                self.portable_suite.policy_seed,
                self.portable_suite.max_decisions,
                self.portable_suite.max_simulated_ticks,
                self.portable_suite.action_spec_sha256,
            )
            != (
                self.exact_suite.split,
                self.exact_suite.repetitions,
                self.exact_suite.policy_seed,
                self.exact_suite.max_decisions,
                self.exact_suite.max_simulated_ticks,
                self.exact_suite.action_spec_sha256,
            )
        ):
            raise ValueError("learned-policy parity cells lack shared provenance")

        def require_cells(report: EvaluationReport, suite: EvaluationSuite) -> None:
            expected = {
                (snapshot_id, repetition)
                for snapshot_id in suite.snapshot_ids
                for repetition in range(suite.repetitions)
            }
            if {
                (episode.snapshot_id, episode.repetition) for episode in report.episodes
            } != expected:
                raise ValueError("learned-policy parity report cells are incomplete")

        require_cells(self.portable_report, self.portable_suite)
        require_cells(self.exact_report, self.exact_suite)
        diagnostics = _cross_backend_episode_diagnostics(
            self.portable_report,
            self.portable_suite,
            self.exact_report,
            self.exact_suite,
        )
        object.__setattr__(self, "_diagnostics", diagnostics)
        object.__setattr__(self, "_diagnostics_sha256", _canonical_sha256(diagnostics))

    @property
    def policy_sha256(self) -> str:
        return self.portable_report.policy_sha256

    @property
    def cross_backend_diagnostics(self) -> tuple[dict[str, object], ...]:
        return self._diagnostics

    @property
    def cross_backend_diagnostics_sha256(self) -> str:
        return self._diagnostics_sha256

    @property
    def normalized_content_sha256(self) -> str:
        """Compatibility alias; this now identifies explicit delta diagnostics."""

        return self.cross_backend_diagnostics_sha256

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "policy_sha256": self.policy_sha256,
            "portable_suite_sha256": self.portable_suite.sha256,
            "portable_report_sha256": self.portable_report.sha256,
            "exact_suite_sha256": self.exact_suite.sha256,
            "exact_report_sha256": self.exact_report.sha256,
            "logical_manifest_sha256": self.logical_manifest.sha256,
            "portable_library_sha256": self.portable_library.sha256,
            "exact_library_sha256": self.exact_library.sha256,
            "cross_backend_diagnostics": list(self.cross_backend_diagnostics),
            "cross_backend_diagnostics_sha256": (self.cross_backend_diagnostics_sha256),
        }

    @classmethod
    def from_manifest(
        cls,
        value: object,
        *,
        portable_suite: EvaluationSuite,
        portable_report: EvaluationReport,
        exact_suite: EvaluationSuite,
        exact_report: EvaluationReport,
        logical_manifest: CrossBackendEvaluationManifest,
        portable_library: SnapshotLibrary,
        exact_library: SnapshotLibrary,
    ) -> LearnedPolicyBackendParityArtifact:
        manifest = _manifest_mapping(
            value,
            {
                "version",
                "policy_sha256",
                "portable_suite_sha256",
                "portable_report_sha256",
                "exact_suite_sha256",
                "exact_report_sha256",
                "logical_manifest_sha256",
                "portable_library_sha256",
                "exact_library_sha256",
                "cross_backend_diagnostics",
                "cross_backend_diagnostics_sha256",
            },
            location="learned-policy backend parity",
        )
        result = cls(
            portable_suite,
            portable_report,
            exact_suite,
            exact_report,
            logical_manifest,
            portable_library,
            exact_library,
            manifest["version"],  # type: ignore[arg-type]
        )
        if result.manifest() != manifest:
            raise ValueError("learned-policy backend parity references differ")
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class ScriptedBaselineSpec:
    baseline_id: str
    parameters: tuple[tuple[str, int | float], ...] = ()
    version: str = "r3b-scripted-baseline-v1"

    def __post_init__(self) -> None:
        supported = {
            "no_action_long_wait",
            "seeded_legal_random",
            "matcher_shot_policy",
            "scripted_direct_matcher",
            "scripted_side_ejector",
            "scripted_imminent_rot_hazard",
        }
        if (
            self.version != "r3b-scripted-baseline-v1"
            or self.baseline_id not in supported
            or not isinstance(self.parameters, tuple)
            or any(
                not isinstance(item, tuple) or len(item) != 2
                for item in self.parameters
            )
        ):
            raise ValueError("unknown scripted baseline")
        names = tuple(name for name, _ in self.parameters)
        if len(names) != len(set(names)) or any(not name for name in names):
            raise ValueError("baseline parameter names must be unique")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for _, value in self.parameters
        ):
            raise ValueError("baseline parameters must be finite numbers")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "baseline_id": self.baseline_id,
            "parameters": {name: value for name, value in self.parameters},
        }

    @property
    def sha256(self) -> str:
        return behavior_build_identity_sha256(
            {
                "purpose": "r3b-scripted-baseline-inference-v1",
                "baseline": self.manifest(),
            }
        )

    def build(self, seed: int) -> Any:
        parameters = dict(self.parameters)
        if self.baseline_id == "no_action_long_wait":
            policy = LongWaitPolicy(**parameters)
        elif self.baseline_id == "seeded_legal_random":
            policy = RandomPolicy(seed=seed, **parameters)
        elif self.baseline_id == "matcher_shot_policy":
            policy = MatcherShotPolicy(**parameters)
        elif self.baseline_id == "scripted_direct_matcher":
            policy = DirectMatcherPolicy(**parameters)
        elif self.baseline_id == "scripted_side_ejector":
            policy = SideEjectorPolicy(**parameters)
        else:
            policy = ImminentRotHazardPolicy(**parameters)
        policy.reset(seed)
        return policy


@dataclass(frozen=True, slots=True)
class DeploymentPolicyIdentity:
    """Everything that changes learned-policy actions at inference time."""

    model_sha256: str
    model_manifest_sha256: str
    schema_sha256: str
    encoder_identity_sha256: str
    inference_build_sha256: str
    action_spec_sha256: str
    kind_mask: tuple[bool, ...]
    wait_mask: tuple[bool, ...]
    version: str = "r3b-deployment-policy-v2"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-deployment-policy-v2"
            or any(
                not _is_sha256(value) or value == "0" * 64
                for value in (
                    self.model_sha256,
                    self.model_manifest_sha256,
                    self.schema_sha256,
                    self.encoder_identity_sha256,
                    self.inference_build_sha256,
                    self.action_spec_sha256,
                )
            )
            or len(self.kind_mask) != 3
            or not self.wait_mask
            or any(not isinstance(value, bool) for value in self.kind_mask)
            or any(not isinstance(value, bool) for value in self.wait_mask)
            or not any(self.kind_mask)
            or (
                self.kind_mask[int(SemanticActionKind.WAIT)] and not any(self.wait_mask)
            )
        ):
            raise ValueError("deployment policy identity is malformed")

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["kind_mask"] = list(self.kind_mask)
        value["wait_mask"] = list(self.wait_mask)
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    @classmethod
    def from_components(
        cls,
        model: RecurrentActorCritic,
        encoder: object,
        kind_mask: Tensor,
        wait_mask: Tensor,
    ) -> DeploymentPolicyIdentity:
        schema = getattr(encoder, "schema", None)
        if schema != model.schema:
            raise ValueError("evaluation encoder schema disagrees with the model")
        encoder_manifest = encoder_instance_manifest(encoder)
        model_manifest = model.manifest()
        model_manifest_sha256 = _canonical_sha256(model_manifest)
        encoder_identity_sha256 = _canonical_sha256(encoder_manifest)
        inference_build_sha256 = behavior_build_identity_sha256(
            {
                "purpose": "r3b-recurrent-policy-inference-v1",
                "model": model_manifest,
                "encoder": encoder_manifest,
                "action_spec_sha256": model.action_spec.sha256,
            }
        )
        return cls(
            model_state_sha256(model),
            model_manifest_sha256,
            model.schema.sha256,
            encoder_identity_sha256,
            inference_build_sha256,
            model.action_spec.sha256,
            tuple(bool(value) for value in kind_mask.detach().cpu().reshape(-1)),
            tuple(bool(value) for value in wait_mask.detach().cpu().reshape(-1)),
        )


def _mapping(observation: object) -> Mapping[str, Any]:
    if isinstance(observation, Mapping):
        return observation
    converter = getattr(observation, "to_dict", None)
    if converter is None:
        raise TypeError("scripted evaluation requires a mapping-capable observation")
    value = converter()
    if not isinstance(value, Mapping):
        raise TypeError("observation to_dict() did not return a mapping")
    return value


def semantic_from_native(action: Action, spec: ActionSpec) -> SemanticAction:
    kind = ActionKind.parse(action.kind)
    if kind is ActionKind.WAIT:
        return spec.validate(SemanticAction.wait(int(action.wait_ticks)))
    if kind not in {ActionKind.WEAK_SHOT, ActionKind.STRONG_SHOT}:
        raise ValueError("scripted policy emitted an unsupported simultaneous shot")
    constructor = (
        SemanticAction.weak if kind is ActionKind.WEAK_SHOT else SemanticAction.strong
    )
    return spec.validate(
        constructor(
            float(action.cursor_x) / spec.client_width,
            float(action.cursor_y) / spec.client_height,
        )
    )


class RecurrentSemanticPolicy:
    """Deterministic deployment-style recurrent inference at semantic boundaries."""

    def __init__(self, model: RecurrentActorCritic) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self._state: Tensor | None = None
        self._reset_before: Tensor | None = None

    def reset(self, lanes: int) -> None:
        if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes <= 0:
            raise ValueError("policy lane count must be positive")
        self._state = self.model.initial_state(lanes).detach()
        self._reset_before = torch.ones(lanes, dtype=torch.bool, device=self.device)

    def act(
        self,
        observation: EncodedBatch,
        kind_mask: Tensor,
        wait_mask: Tensor,
    ) -> tuple[SemanticAction, ...]:
        observation.validate()
        if observation.schema != self.model.schema:
            raise ValueError("encoded evaluation batch uses the wrong tensor schema")
        lanes = observation.global_features.shape[0]
        if self._state is None or self._reset_before is None:
            raise RuntimeError("recurrent evaluation policy must be reset")
        if kind_mask.shape != (lanes, 3) or kind_mask.dtype != torch.bool:
            raise ValueError("evaluation kind mask must be boolean [B, 3]")
        expected_wait = (lanes, len(self.model.action_spec.wait_choices))
        if wait_mask.shape != expected_wait or wait_mask.dtype != torch.bool:
            raise ValueError("evaluation wait mask does not match the action schema")
        if not bool(torch.all(kind_mask.any(dim=1))):
            raise ValueError("evaluation kind mask contains an all-masked lane")
        wait_lanes = kind_mask[:, int(SemanticActionKind.WAIT)]
        if bool(torch.any(wait_lanes & ~wait_mask.any(dim=1))):
            raise ValueError("WAIT is enabled without a legal wait duration")
        global_features = (
            torch.from_numpy(observation.global_features).to(self.device).unsqueeze(0)
        )
        body_features = (
            torch.from_numpy(observation.body_features).to(self.device).unsqueeze(0)
        )
        body_mask = torch.from_numpy(observation.body_mask).to(self.device).unsqueeze(0)
        arguments: dict[str, Tensor] = {}
        if self.model.config.critic_condition_features:
            arguments["critic_condition"] = torch.zeros(
                (1, lanes, self.model.config.critic_condition_features),
                dtype=torch.float32,
                device=self.device,
            )
        prior_mode = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(
                    global_features,
                    body_features,
                    body_mask,
                    self._state,
                    reset_before=self._reset_before.unsqueeze(0),
                    **arguments,
                )
        finally:
            self.model.train(prior_mode)
        kind = (
            output.kind_logits[0]
            .masked_fill(~kind_mask.to(self.device), -torch.inf)
            .argmax(-1)
        )
        wait = (
            output.wait_logits[0]
            .masked_fill(~wait_mask.to(self.device), -torch.inf)
            .argmax(-1)
        )
        coordinate_mean = output.coordinate_alpha[0] / (
            output.coordinate_alpha[0] + output.coordinate_beta[0]
        )
        actions = []
        for lane in range(lanes):
            kind_value = int(kind[lane])
            xy = (
                coordinate_mean[lane, kind_value - 1]
                if kind_value > 0
                else torch.zeros(2, device=self.device)
            )
            actions.append(
                self.model.action_spec.decode(
                    kind_value,
                    int(wait[lane]),
                    float(xy[0]),
                    float(xy[1]),
                )
            )
        self._state = output.recurrent_state.detach()
        self._reset_before = torch.zeros(lanes, dtype=torch.bool, device=self.device)
        return tuple(actions)


def evaluate_scripted_baseline(
    simulator: Any,
    store: SnapshotBlobStore,
    suite: EvaluationSuite,
    baseline: ScriptedBaselineSpec,
    *,
    evaluator_sha256: str,
    expected_assignment_sha256: str,
    execution_identity_sha256: str,
    cells: tuple[tuple[str, int], ...] | None = None,
) -> EvaluationReport:
    """Evaluate fixed snapshot/repetition cells using deployment macro semantics."""

    action_spec = ActionSpec()

    def factory(seed: int):
        policy = baseline.build(seed)
        return lambda observation: semantic_from_native(
            policy.act(_mapping(observation)), action_spec
        )

    return _evaluate_semantic_policy(
        simulator,
        store,
        suite,
        policy_sha256=baseline.sha256,
        evaluator_sha256=evaluator_sha256,
        expected_assignment_sha256=expected_assignment_sha256,
        execution_identity_sha256=execution_identity_sha256,
        cells=cells,
        action_spec=action_spec,
        policy_factory=factory,
    )


def evaluate_recurrent_policy(
    simulator: Any,
    store: SnapshotBlobStore,
    suite: EvaluationSuite,
    model: RecurrentActorCritic,
    encoder: Any,
    kind_mask: Tensor,
    wait_mask: Tensor,
    *,
    evaluator_sha256: str,
    expected_assignment_sha256: str,
    execution_identity_sha256: str,
    cells: tuple[tuple[str, int], ...] | None = None,
) -> EvaluationReport:
    """Evaluate a learned policy on the same fixed cells and macro semantics."""

    if kind_mask.shape != (1, 3) or kind_mask.dtype != torch.bool:
        raise ValueError("recurrent evaluation kind mask must be boolean [1, 3]")
    expected_wait = (1, len(model.action_spec.wait_choices))
    if wait_mask.shape != expected_wait or wait_mask.dtype != torch.bool:
        raise ValueError(
            "recurrent evaluation wait mask disagrees with the action schema"
        )
    if not bool(torch.all(kind_mask.any(dim=1))):
        raise ValueError("recurrent evaluation kind mask is all-masked")
    if bool(kind_mask[0, int(SemanticActionKind.WAIT)]) and not bool(wait_mask.any()):
        raise ValueError("WAIT is enabled without a legal wait duration")
    kind_mask = kind_mask.detach().cpu().clone()
    wait_mask = wait_mask.detach().cpu().clone()
    deployment_identity = DeploymentPolicyIdentity.from_components(
        model, encoder, kind_mask, wait_mask
    )

    def factory(seed: int):
        del seed
        policy = RecurrentSemanticPolicy(model)
        policy.reset(1)

        def act(observation: object) -> SemanticAction:
            encoded = encoder.encode((observation,))
            return policy.act(encoded, kind_mask, wait_mask)[0]

        return act

    return _evaluate_semantic_policy(
        simulator,
        store,
        suite,
        policy_sha256=deployment_identity.sha256,
        evaluator_sha256=evaluator_sha256,
        expected_assignment_sha256=expected_assignment_sha256,
        execution_identity_sha256=execution_identity_sha256,
        cells=cells,
        action_spec=model.action_spec,
        policy_factory=factory,
    )


def evaluate_recurrent_policy_vectorized(
    simulator: Any,
    store: SnapshotBlobStore,
    suite: EvaluationSuite,
    model: RecurrentActorCritic,
    encoder: Any,
    kind_mask: Tensor,
    wait_mask: Tensor,
    *,
    evaluator_sha256: str,
    expected_assignment_sha256: str,
    execution_identity_sha256: str,
    cells: tuple[tuple[str, int], ...] | None = None,
) -> EvaluationReport:
    """Evaluate fixed cells concurrently on a subset-capable vector simulator.

    Each lane owns one complete episode at a time. Policy decisions are batched,
    while press/release primitives use ``step_many`` subsets so a terminating
    lane never receives a synthetic action and macro boundaries remain identical
    to :func:`evaluate_recurrent_policy`.
    """

    lane_count = getattr(simulator, "num_envs", None)
    lane_count = lane_count() if callable(lane_count) else lane_count
    envs = getattr(simulator, "envs", None)
    envs = envs() if callable(envs) else envs
    required_methods = (
        "reset_many",
        "restore_many",
        "step_many",
        "state_hash_many",
        "config_hash_many",
    )
    if (
        isinstance(lane_count, bool)
        or not isinstance(lane_count, Integral)
        or lane_count <= 0
        or not isinstance(envs, (tuple, list))
        or len(envs) != lane_count
        or any(
            not callable(getattr(simulator, name, None)) for name in required_methods
        )
    ):
        raise ValueError(
            "vector evaluation requires a nonempty subset-capable simulator"
        )
    if kind_mask.shape != (1, 3) or kind_mask.dtype != torch.bool:
        raise ValueError("recurrent evaluation kind mask must be boolean [1, 3]")
    expected_wait = (1, len(model.action_spec.wait_choices))
    if wait_mask.shape != expected_wait or wait_mask.dtype != torch.bool:
        raise ValueError(
            "recurrent evaluation wait mask disagrees with the action schema"
        )
    if not bool(torch.all(kind_mask.any(dim=1))):
        raise ValueError("recurrent evaluation kind mask is all-masked")
    if bool(kind_mask[0, int(SemanticActionKind.WAIT)]) and not bool(wait_mask.any()):
        raise ValueError("WAIT is enabled without a legal wait duration")
    kind_mask = kind_mask.detach().cpu().clone()
    wait_mask = wait_mask.detach().cpu().clone()
    deployment_identity = DeploymentPolicyIdentity.from_components(
        model, encoder, kind_mask, wait_mask
    )
    identities = (
        evaluator_sha256,
        deployment_identity.sha256,
        expected_assignment_sha256,
        execution_identity_sha256,
    )
    if any(not _is_sha256(value) or value == "0" * 64 for value in identities):
        raise ValueError("policy and evaluator identities must be lowercase SHA-256")
    runtime = attest_simulator_runtime(simulator)
    if (
        runtime.sha256 != suite.runtime_identity_sha256
        or runtime.backend != suite.backend
    ):
        raise ValueError("evaluated simulator runtime identity mismatch")
    if suite.assignment_sha256 != expected_assignment_sha256:
        raise ValueError("evaluation suite assignment identity mismatch")
    if suite.library_sha256 != store.library.sha256:
        raise ValueError("evaluation suite snapshot-library identity mismatch")
    if suite.snapshot_store_sha256 != store.sha256:
        raise ValueError("evaluation suite snapshot-store identity mismatch")
    if suite.action_spec_sha256 != model.action_spec.sha256:
        raise ValueError("evaluation suite action identity mismatch")

    all_cells = tuple(
        (snapshot_id, repetition)
        for snapshot_id in suite.snapshot_ids
        for repetition in range(suite.repetitions)
    )
    if cells is None:
        selected_cells = frozenset(all_cells)
    else:
        if (
            not isinstance(cells, tuple)
            or not cells
            or any(
                not isinstance(cell, tuple)
                or len(cell) != 2
                or not isinstance(cell[0], str)
                or isinstance(cell[1], bool)
                or not isinstance(cell[1], Integral)
                for cell in cells
            )
            or len(set(cells)) != len(cells)
            or not set(cells).issubset(all_cells)
        ):
            raise ValueError("selected evaluation cells are malformed or foreign")
        selected_cells = frozenset(cells)
    ordered_cells = tuple(cell for cell in all_cells if cell in selected_cells)

    recipes = {}
    for snapshot_id, recipe_sha256 in zip(suite.snapshot_ids, suite.recipe_sha256s):
        recipe = store.library[snapshot_id]
        if recipe.sha256 != recipe_sha256:
            raise ValueError("evaluation suite recipe identity mismatch")
        if recipe.split != suite.split:
            raise ValueError("evaluation snapshot belongs to the wrong split")
        if recipe.runtime_identity_sha256 != suite.runtime_identity_sha256:
            raise ValueError("evaluation snapshot runtime identity mismatch")
        recipes[snapshot_id] = recipe

    def materialize(observation: object) -> dict[str, Any]:
        return dict(_mapping(observation))

    # Exact PaddedVectorEnv must own an initialized state before restore_many
    # can capture transactional rollback backups. These states are immediately
    # replaced by declared snapshots and never enter evaluation evidence.
    all_lanes = tuple(range(int(lane_count)))
    initialized = simulator.reset_many(
        all_lanes,
        seeds=tuple(range(int(lane_count))),
    )
    if len(initialized) != int(lane_count):
        raise RuntimeError("vector reset returned the wrong lane count")

    episodes: list[EpisodeMetrics] = []
    for start in range(0, len(ordered_cells), int(lane_count)):
        batch = ordered_cells[start : start + int(lane_count)]
        lanes = tuple(range(len(batch)))
        snapshots = tuple(store[snapshot_id] for snapshot_id, _ in batch)
        observations = [
            materialize(value) for value in simulator.restore_many(lanes, snapshots)
        ]
        if len(observations) != len(batch):
            raise RuntimeError("vector restore returned the wrong lane count")
        config_hashes = tuple(simulator.config_hash_many(lanes))
        state_hashes = tuple(simulator.state_hash_many(lanes))
        if len(config_hashes) != len(batch) or len(state_hashes) != len(batch):
            raise RuntimeError("vector identity query returned the wrong lane count")

        initial_ticks: list[int] = []
        initial_scores: list[int] = []
        minimum_gauges: list[int] = []
        seeds: list[int] = []
        for lane, ((snapshot_id, repetition), observation) in enumerate(
            zip(batch, observations)
        ):
            recipe = recipes[snapshot_id]
            config = getattr(envs[lane], "config", None)
            config = config() if callable(config) else config
            gauge = observation.get("gauge")
            gauge_max = observation.get("gauge_max")
            if int(config_hashes[lane]) != recipe.config_hash:
                raise ValueError("evaluated simulator config hash mismatch")
            if _canonical_sha256(config) != recipe.config_sha256:
                raise ValueError("evaluated simulator canonical config mismatch")
            if int(state_hashes[lane]) != recipe.expected_state_hash:
                raise ValueError("evaluation snapshot state hash mismatch")
            if (
                int(observation.get("tick", -1)) != recipe.expected_tick
                or int(observation.get("score", -1)) != recipe.expected_score
                or bool(observation.get("terminated", False))
                or bool(observation.get("truncated", False))
                or isinstance(gauge, bool)
                or not isinstance(gauge, Integral)
                or isinstance(gauge_max, bool)
                or not isinstance(gauge_max, Integral)
                or gauge_max <= 0
                or not 0 <= gauge <= gauge_max
            ):
                raise ValueError(
                    "evaluation snapshot is not the declared live boundary"
                )
            initial_ticks.append(int(observation["tick"]))
            initial_scores.append(int(observation["score"]))
            minimum_gauges.append(int(observation["gauge"]))
            seeds.append(suite.episode_seed(snapshot_id, repetition))

        policy = RecurrentSemanticPolicy(model)
        policy.reset(len(batch))
        batch_kind_mask = kind_mask.expand(len(batch), -1).clone()
        batch_wait_mask = wait_mask.expand(len(batch), -1).clone()
        decisions = [0] * len(batch)
        invalid_actions = [0] * len(batch)
        accumulated_rewards = [0] * len(batch)
        terminated = [False] * len(batch)
        truncated = [False] * len(batch)

        while True:
            active = tuple(
                lane
                for lane, observation in enumerate(observations)
                if not terminated[lane]
                and not truncated[lane]
                and decisions[lane] < suite.max_decisions
                and int(observation["tick"]) - initial_ticks[lane]
                < suite.max_simulated_ticks
            )
            if not active:
                break
            encoded = encoder.encode(tuple(observations))
            semantic_actions = policy.act(encoded, batch_kind_mask, batch_wait_mask)
            first_actions: list[Action] = []
            for lane in active:
                semantic = model.action_spec.validate(semantic_actions[lane])
                elapsed = int(observations[lane]["tick"]) - initial_ticks[lane]
                remaining = suite.max_simulated_ticks - elapsed
                first_actions.append(
                    Action.wait(min(semantic.wait_ticks, remaining))
                    if semantic.kind is SemanticActionKind.WAIT
                    else model.action_spec.press(semantic)
                )
            transitions = simulator.step_many(active, tuple(first_actions))
            if any(len(values) != len(active) for values in transitions):
                raise RuntimeError("vector step returned the wrong lane count")
            next_observations, rewards, ended, cut, infos = transitions
            for offset, lane in enumerate(active):
                observations[lane] = materialize(next_observations[offset])
                accumulated_rewards[lane] += int(rewards[offset])
                invalid_actions[lane] += int(
                    bool(infos[offset].get("invalid_action", False))
                )
                minimum_gauges[lane] = min(
                    minimum_gauges[lane], int(observations[lane]["gauge"])
                )
                terminated[lane] = bool(ended[offset])
                truncated[lane] = bool(cut[offset])

            release_lanes = tuple(
                lane
                for lane in active
                if semantic_actions[lane].kind is not SemanticActionKind.WAIT
                and not terminated[lane]
                and not truncated[lane]
                and int(observations[lane]["tick"]) - initial_ticks[lane]
                < suite.max_simulated_ticks
            )
            if release_lanes:
                releases = tuple(model.action_spec.release() for _ in release_lanes)
                release_results = simulator.step_many(release_lanes, releases)
                if any(len(values) != len(release_lanes) for values in release_results):
                    raise RuntimeError("vector release returned the wrong lane count")
                next_observations, rewards, ended, cut, infos = release_results
                for offset, lane in enumerate(release_lanes):
                    observations[lane] = materialize(next_observations[offset])
                    accumulated_rewards[lane] += int(rewards[offset])
                    invalid_actions[lane] += int(
                        bool(infos[offset].get("invalid_action", False))
                    )
                    minimum_gauges[lane] = min(
                        minimum_gauges[lane],
                        int(observations[lane]["gauge"]),
                    )
                    terminated[lane] = bool(ended[offset])
                    truncated[lane] = bool(cut[offset])
            for lane in active:
                decisions[lane] += 1

        for lane, (snapshot_id, repetition) in enumerate(batch):
            final = observations[lane]
            final_score = int(final["score"])
            elapsed = int(final["tick"]) - initial_ticks[lane]
            if elapsed > suite.max_simulated_ticks:
                raise ValueError("evaluation exceeded its simulated-tick horizon")
            if accumulated_rewards[lane] != final_score - initial_scores[lane]:
                raise ValueError("evaluation raw reward does not equal score delta")
            budget_cut = not terminated[lane] and not truncated[lane]
            episodes.append(
                EpisodeMetrics(
                    snapshot_id,
                    repetition,
                    seeds[lane],
                    initial_scores[lane],
                    final_score,
                    final_score - initial_scores[lane],
                    elapsed,
                    decisions[lane],
                    terminated[lane],
                    truncated[lane] or budget_cut,
                    invalid_actions[lane],
                    minimum_gauges[lane],
                    int(final["gauge"]),
                )
            )
    return EvaluationReport(
        suite.sha256,
        deployment_identity.sha256,
        evaluator_sha256,
        runtime.sha256,
        execution_identity_sha256,
        tuple(episodes),
    )


def _evaluate_semantic_policy(
    simulator: Any,
    store: SnapshotBlobStore,
    suite: EvaluationSuite,
    *,
    policy_sha256: str,
    evaluator_sha256: str,
    expected_assignment_sha256: str,
    execution_identity_sha256: str,
    cells: tuple[tuple[str, int], ...] | None,
    action_spec: ActionSpec,
    policy_factory: Any,
) -> EvaluationReport:
    """Shared fixed-cell evaluator after a policy has entered semantic space."""

    if (
        not _is_sha256(evaluator_sha256)
        or not _is_sha256(policy_sha256)
        or not _is_sha256(expected_assignment_sha256)
        or not _is_sha256(execution_identity_sha256)
        or "0" * 64
        in (
            evaluator_sha256,
            policy_sha256,
            expected_assignment_sha256,
            execution_identity_sha256,
        )
    ):
        raise ValueError("policy and evaluator identities must be lowercase SHA-256")
    runtime = attest_simulator_runtime(simulator)
    if (
        runtime.sha256 != suite.runtime_identity_sha256
        or runtime.backend != suite.backend
    ):
        raise ValueError("evaluated simulator runtime identity mismatch")
    if suite.assignment_sha256 != expected_assignment_sha256:
        raise ValueError("evaluation suite assignment identity mismatch")
    if suite.library_sha256 != store.library.sha256:
        raise ValueError("evaluation suite snapshot-library identity mismatch")
    if suite.snapshot_store_sha256 != store.sha256:
        raise ValueError("evaluation suite snapshot-store identity mismatch")
    if suite.action_spec_sha256 != action_spec.sha256:
        raise ValueError("evaluation suite action identity mismatch")
    all_cells = tuple(
        (snapshot_id, repetition)
        for snapshot_id in suite.snapshot_ids
        for repetition in range(suite.repetitions)
    )
    if cells is None:
        selected_cells = frozenset(all_cells)
    else:
        if (
            not isinstance(cells, tuple)
            or not cells
            or any(
                not isinstance(cell, tuple)
                or len(cell) != 2
                or not isinstance(cell[0], str)
                or isinstance(cell[1], bool)
                or not isinstance(cell[1], Integral)
                for cell in cells
            )
            or len(set(cells)) != len(cells)
            or not set(cells).issubset(all_cells)
        ):
            raise ValueError("selected evaluation cells are malformed or foreign")
        selected_cells = frozenset(cells)
    episodes: list[EpisodeMetrics] = []
    for snapshot_id, recipe_sha256 in zip(suite.snapshot_ids, suite.recipe_sha256s):
        recipe = store.library[snapshot_id]
        if recipe.sha256 != recipe_sha256:
            raise ValueError("evaluation suite recipe identity mismatch")
        if recipe.split != suite.split:
            raise ValueError("evaluation snapshot belongs to the wrong split")
        if recipe.runtime_identity_sha256 != suite.runtime_identity_sha256:
            raise ValueError("evaluation snapshot runtime identity mismatch")
        for repetition in range(suite.repetitions):
            if (snapshot_id, repetition) not in selected_cells:
                continue
            observation = simulator.restore_state(store[snapshot_id])
            if int(simulator.config_hash()) != recipe.config_hash:
                raise ValueError("evaluated simulator config hash mismatch")
            config = simulator.config
            config = config() if callable(config) else config
            if _canonical_sha256(config) != recipe.config_sha256:
                raise ValueError("evaluated simulator canonical config mismatch")
            if int(simulator.state_hash()) != recipe.expected_state_hash:
                raise ValueError("evaluation snapshot state hash mismatch")
            restored = _mapping(observation)
            gauge = restored.get("gauge")
            gauge_max = restored.get("gauge_max")
            if (
                int(restored.get("tick", -1)) != recipe.expected_tick
                or int(restored.get("score", -1)) != recipe.expected_score
                or bool(restored.get("terminated", False))
                or bool(restored.get("truncated", False))
                or isinstance(gauge, bool)
                or not isinstance(gauge, Integral)
                or isinstance(gauge_max, bool)
                or not isinstance(gauge_max, Integral)
                or gauge_max <= 0
                or not 0 <= gauge <= gauge_max
            ):
                raise ValueError(
                    "evaluation snapshot is not the declared live boundary"
                )
            seed = suite.episode_seed(snapshot_id, repetition)
            act = policy_factory(seed)
            if not callable(act):
                raise TypeError("evaluation policy factory must return a callable")
            initial_tick = int(_mapping(observation)["tick"])
            initial_score = int(_mapping(observation)["score"])
            minimum_gauge = int(_mapping(observation)["gauge"])
            decisions = 0
            invalid_actions = 0
            accumulated_reward = 0
            terminated = False
            truncated = False
            while (
                not terminated
                and not truncated
                and decisions < suite.max_decisions
                and int(_mapping(observation)["tick"]) - initial_tick
                < suite.max_simulated_ticks
            ):
                semantic = action_spec.validate(act(observation))
                elapsed = int(_mapping(observation)["tick"]) - initial_tick
                remaining = suite.max_simulated_ticks - elapsed
                if remaining <= 0:
                    break
                first = (
                    Action.wait(min(semantic.wait_ticks, remaining))
                    if semantic.kind is SemanticActionKind.WAIT
                    else action_spec.press(semantic)
                )
                primitives = [first]
                if semantic.kind is not SemanticActionKind.WAIT:
                    primitives.append(action_spec.release())
                for primitive in primitives:
                    elapsed = int(_mapping(observation)["tick"]) - initial_tick
                    remaining = suite.max_simulated_ticks - elapsed
                    if remaining <= 0:
                        break
                    if (
                        ActionKind.parse(primitive.kind) is ActionKind.WAIT
                        and primitive.wait_ticks > remaining
                    ):
                        primitive = Action.wait(remaining)
                    observation, reward, terminated, truncated, info = simulator.step(
                        primitive
                    )
                    accumulated_reward += int(reward)
                    invalid_actions += int(bool(info.get("invalid_action", False)))
                    minimum_gauge = min(
                        minimum_gauge, int(_mapping(observation)["gauge"])
                    )
                    if terminated or truncated:
                        break
                decisions += 1
            budget_cut = not terminated and not truncated
            final = _mapping(observation)
            final_score = int(final["score"])
            if int(final["tick"]) - initial_tick > suite.max_simulated_ticks:
                raise ValueError("evaluation exceeded its simulated-tick horizon")
            if accumulated_reward != final_score - initial_score:
                raise ValueError("evaluation raw reward does not equal score delta")
            episodes.append(
                EpisodeMetrics(
                    snapshot_id,
                    repetition,
                    seed,
                    initial_score,
                    final_score,
                    final_score - initial_score,
                    int(final["tick"]) - initial_tick,
                    decisions,
                    terminated,
                    truncated or budget_cut,
                    invalid_actions,
                    minimum_gauge,
                    int(final["gauge"]),
                )
            )
    return EvaluationReport(
        suite.sha256,
        policy_sha256,
        evaluator_sha256,
        runtime.sha256,
        execution_identity_sha256,
        tuple(episodes),
    )


def build_baseline_evidence(
    baseline: ScriptedBaselineSpec,
    primary_suite: EvaluationSuite,
    primary_report: EvaluationReport,
    primary_replay_report: EvaluationReport,
    diagnostic_suite: EvaluationSuite,
    diagnostic_report: EvaluationReport,
    logical_manifest: CrossBackendEvaluationManifest,
    portable_library: SnapshotLibrary,
    exact_library: SnapshotLibrary,
):
    """Build exact-primary evidence plus a portable transfer diagnostic."""

    from .r3b_experiments import BaselineEvidence

    reports = (primary_report, primary_replay_report, diagnostic_report)
    if (
        primary_report.suite_sha256 != primary_suite.sha256
        or primary_replay_report.suite_sha256 != primary_suite.sha256
        or diagnostic_report.suite_sha256 != diagnostic_suite.sha256
    ):
        raise ValueError("baseline reports disagree with their evaluation suites")
    if any(value.policy_sha256 != baseline.sha256 for value in reports):
        raise ValueError("baseline reports disagree with the scripted policy")
    if (
        primary_suite.backend != "exact"
        or diagnostic_suite.backend != "portable"
        or primary_suite.runtime_identity_sha256
        == diagnostic_suite.runtime_identity_sha256
        or primary_suite.library_sha256 == diagnostic_suite.library_sha256
        or primary_suite.snapshot_store_sha256 == diagnostic_suite.snapshot_store_sha256
        or primary_report.backend_identity_sha256
        != primary_suite.runtime_identity_sha256
        or primary_replay_report.backend_identity_sha256
        != primary_suite.runtime_identity_sha256
        or diagnostic_report.backend_identity_sha256
        != diagnostic_suite.runtime_identity_sha256
        or len({value.execution_identity_sha256 for value in reports}) != 3
        or len({value.sha256 for value in reports}) != 3
    ):
        raise ValueError("baseline evidence lacks independent backend executions")
    if not isinstance(logical_manifest, CrossBackendEvaluationManifest):
        raise TypeError("cross-backend parity requires typed recipe provenance")
    logical_ids = tuple(value.logical_cell.sha256 for value in logical_manifest.pairs)
    if (
        primary_suite.logical_manifest_sha256 != logical_manifest.sha256
        or diagnostic_suite.logical_manifest_sha256 != logical_manifest.sha256
        or diagnostic_suite.snapshot_ids
        != tuple(value.portable_snapshot_id for value in logical_manifest.pairs)
        or primary_suite.snapshot_ids
        != tuple(value.exact_snapshot_id for value in logical_manifest.pairs)
        or primary_suite.logical_cell_ids != logical_ids
        or diagnostic_suite.logical_cell_ids != logical_ids
        or primary_suite.split != diagnostic_suite.split
        or primary_suite.split != logical_manifest.pairs[0].logical_cell.split
        or primary_suite.repetitions != diagnostic_suite.repetitions
        or primary_suite.policy_seed != diagnostic_suite.policy_seed
        or primary_suite.max_decisions != diagnostic_suite.max_decisions
        or primary_suite.max_simulated_ticks != diagnostic_suite.max_simulated_ticks
        or primary_suite.action_spec_sha256 != diagnostic_suite.action_spec_sha256
        or primary_suite.action_spec_sha256
        != logical_manifest.pairs[0].logical_cell.action_spec_sha256
        or diagnostic_suite.recipe_sha256s
        != tuple(value.portable_recipe_sha256 for value in logical_manifest.pairs)
        or primary_suite.recipe_sha256s
        != tuple(value.exact_recipe_sha256 for value in logical_manifest.pairs)
        or not isinstance(portable_library, SnapshotLibrary)
        or not isinstance(exact_library, SnapshotLibrary)
        or diagnostic_suite.library_sha256 != portable_library.sha256
        or primary_suite.library_sha256 != exact_library.sha256
        or diagnostic_suite.recipe_sha256s
        != tuple(
            portable_library[snapshot_id].sha256
            for snapshot_id in diagnostic_suite.snapshot_ids
        )
        or primary_suite.recipe_sha256s
        != tuple(
            exact_library[snapshot_id].sha256
            for snapshot_id in primary_suite.snapshot_ids
        )
    ):
        raise ValueError("portable/exact suites lack shared recipe provenance")

    def validate_cells(value: EvaluationReport, value_suite: EvaluationSuite) -> None:
        expected_cells = {
            (snapshot_id, repetition)
            for snapshot_id in value_suite.snapshot_ids
            for repetition in range(value_suite.repetitions)
        }
        if {
            (episode.snapshot_id, episode.repetition) for episode in value.episodes
        } != expected_cells or any(
            episode.policy_seed
            != value_suite.episode_seed(episode.snapshot_id, episode.repetition)
            or episode.decisions > value_suite.max_decisions
            or episode.elapsed_ticks > value_suite.max_simulated_ticks
            or not (episode.terminated or episode.truncated)
            for episode in value.episodes
        ):
            raise ValueError("baseline report cells do not exactly match the suite")

    validate_cells(primary_report, primary_suite)
    validate_cells(primary_replay_report, primary_suite)
    validate_cells(diagnostic_report, diagnostic_suite)
    if (
        primary_report.episode_content_sha256
        != primary_replay_report.episode_content_sha256
    ):
        raise ValueError("baseline replay is not deterministic")

    diagnostics = _cross_backend_episode_diagnostics(
        diagnostic_report,
        diagnostic_suite,
        primary_report,
        primary_suite,
    )
    normalized_content_sha256 = _canonical_sha256(
        {
            "version": "r3b-cross-backend-episode-diagnostics-v1",
            "exact_minus_portable": diagnostics,
        }
    )
    episodes = len(primary_report.episodes)
    return BaselineEvidence(
        baseline.baseline_id,
        "complete",
        episodes,
        sum(value.raw_score for value in primary_report.episodes) / episodes,
        sum(value.invalid_actions for value in primary_report.episodes),
        primary_suite.sha256,
        primary_report.sha256,
        primary_replay_report.sha256,
        diagnostic_report.sha256,
        diagnostic_report.backend_identity_sha256,
        primary_report.backend_identity_sha256,
        normalized_content_sha256,
    )


@dataclass(frozen=True, slots=True)
class BaselineArtifactBundle:
    """Typed exact-primary evidence and portable transfer diagnostics."""

    baseline: ScriptedBaselineSpec
    primary_suite: EvaluationSuite
    primary_report: EvaluationReport
    primary_replay_report: EvaluationReport
    diagnostic_suite: EvaluationSuite
    diagnostic_report: EvaluationReport
    logical_manifest: CrossBackendEvaluationManifest
    portable_library: SnapshotLibrary
    exact_library: SnapshotLibrary

    def evidence(self):
        return build_baseline_evidence(
            self.baseline,
            self.primary_suite,
            self.primary_report,
            self.primary_replay_report,
            self.diagnostic_suite,
            self.diagnostic_report,
            self.logical_manifest,
            self.portable_library,
            self.exact_library,
        )
