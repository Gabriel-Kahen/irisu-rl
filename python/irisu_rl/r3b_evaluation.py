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
    expected_state_hash: int
    version: str = "r3b-logical-evaluation-cell-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-logical-evaluation-cell-v1"
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
            or isinstance(self.expected_state_hash, bool)
            or not isinstance(self.expected_state_hash, Integral)
            or not 0 <= self.expected_state_hash < 2**64
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
            int(recipe.expected_state_hash),
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
            "expected_state_hash": int(self.expected_state_hash),
        }

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


@dataclass(frozen=True, slots=True)
class CrossBackendEvaluationManifest:
    pairs: tuple[CrossBackendCellPair, ...]
    version: str = "r3b-cross-backend-evaluation-manifest-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-cross-backend-evaluation-manifest-v1"
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
            or self.minimum_gauge < 0
            or self.final_gauge < 0
            or self.raw_score != self.final_score - self.initial_score
            or not isinstance(self.terminated, bool)
            or not isinstance(self.truncated, bool)
        ):
            raise ValueError("evaluation episode metrics are malformed")


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

    @property
    def sha256(self) -> str:
        return self._sha256

    @property
    def episode_content_sha256(self) -> str:
        return self._episode_content_sha256


def _logical_episode_content(
    report: EvaluationReport, suite: EvaluationSuite
) -> tuple[dict[str, object], ...]:
    logical = dict(zip(suite.snapshot_ids, suite.logical_cell_ids))
    return tuple(
        {**asdict(episode), "snapshot_id": logical[episode.snapshot_id]}
        for episode in report.episodes
    )


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
    version: str = "r3b-learned-policy-backend-parity-v1"
    _normalized_content_sha256: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-learned-policy-backend-parity-v1"
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
        portable_content = _canonical_sha256(
            _logical_episode_content(self.portable_report, self.portable_suite)
        )
        exact_content = _canonical_sha256(
            _logical_episode_content(self.exact_report, self.exact_suite)
        )
        if portable_content != exact_content:
            raise ValueError("learned policy differs across simulator backends")
        object.__setattr__(self, "_normalized_content_sha256", portable_content)

    @property
    def policy_sha256(self) -> str:
        return self.portable_report.policy_sha256

    @property
    def normalized_content_sha256(self) -> str:
        return self._normalized_content_sha256

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
            "normalized_content_sha256": self.normalized_content_sha256,
        }

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
        action_spec=model.action_spec,
        policy_factory=factory,
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
            observation = simulator.restore_state(store[snapshot_id])
            if int(simulator.config_hash()) != recipe.config_hash:
                raise ValueError("evaluated simulator config hash mismatch")
            if _canonical_sha256(simulator.config()) != recipe.config_sha256:
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
    suite: EvaluationSuite,
    report: EvaluationReport,
    replay_report: EvaluationReport,
    exact_suite: EvaluationSuite,
    exact_backend_report: EvaluationReport,
    logical_manifest: CrossBackendEvaluationManifest,
    portable_library: SnapshotLibrary,
    exact_library: SnapshotLibrary,
):
    """Build acceptance evidence only from matching evaluated report artifacts."""

    from .r3b_experiments import BaselineEvidence

    reports = (report, replay_report, exact_backend_report)
    if (
        report.suite_sha256 != suite.sha256
        or replay_report.suite_sha256 != suite.sha256
        or exact_backend_report.suite_sha256 != exact_suite.sha256
    ):
        raise ValueError("baseline reports disagree with their evaluation suites")
    if any(value.policy_sha256 != baseline.sha256 for value in reports):
        raise ValueError("baseline reports disagree with the scripted policy")
    if (
        {suite.backend, exact_suite.backend} != {"portable", "exact"}
        or suite.runtime_identity_sha256 == exact_suite.runtime_identity_sha256
        or suite.library_sha256 == exact_suite.library_sha256
        or suite.snapshot_store_sha256 == exact_suite.snapshot_store_sha256
        or report.backend_identity_sha256 != suite.runtime_identity_sha256
        or replay_report.backend_identity_sha256 != suite.runtime_identity_sha256
        or exact_backend_report.backend_identity_sha256
        != exact_suite.runtime_identity_sha256
        or len({value.execution_identity_sha256 for value in reports}) != 3
        or len({value.sha256 for value in reports}) != 3
    ):
        raise ValueError("baseline evidence lacks independent backend executions")
    if not isinstance(logical_manifest, CrossBackendEvaluationManifest):
        raise TypeError("cross-backend parity requires typed recipe provenance")
    portable_suite = suite if suite.backend == "portable" else exact_suite
    physical_exact_suite = exact_suite if exact_suite.backend == "exact" else suite
    logical_ids = tuple(value.logical_cell.sha256 for value in logical_manifest.pairs)
    if (
        suite.logical_manifest_sha256 != logical_manifest.sha256
        or exact_suite.logical_manifest_sha256 != logical_manifest.sha256
        or portable_suite.snapshot_ids
        != tuple(value.portable_snapshot_id for value in logical_manifest.pairs)
        or physical_exact_suite.snapshot_ids
        != tuple(value.exact_snapshot_id for value in logical_manifest.pairs)
        or suite.logical_cell_ids != logical_ids
        or exact_suite.logical_cell_ids != logical_ids
        or suite.split != exact_suite.split
        or suite.split != logical_manifest.pairs[0].logical_cell.split
        or suite.repetitions != exact_suite.repetitions
        or suite.policy_seed != exact_suite.policy_seed
        or suite.max_decisions != exact_suite.max_decisions
        or suite.max_simulated_ticks != exact_suite.max_simulated_ticks
        or suite.action_spec_sha256 != exact_suite.action_spec_sha256
        or suite.action_spec_sha256
        != logical_manifest.pairs[0].logical_cell.action_spec_sha256
        or portable_suite.recipe_sha256s
        != tuple(value.portable_recipe_sha256 for value in logical_manifest.pairs)
        or physical_exact_suite.recipe_sha256s
        != tuple(value.exact_recipe_sha256 for value in logical_manifest.pairs)
        or not isinstance(portable_library, SnapshotLibrary)
        or not isinstance(exact_library, SnapshotLibrary)
        or portable_suite.library_sha256 != portable_library.sha256
        or physical_exact_suite.library_sha256 != exact_library.sha256
        or portable_suite.recipe_sha256s
        != tuple(
            portable_library[snapshot_id].sha256
            for snapshot_id in portable_suite.snapshot_ids
        )
        or physical_exact_suite.recipe_sha256s
        != tuple(
            exact_library[snapshot_id].sha256
            for snapshot_id in physical_exact_suite.snapshot_ids
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

    validate_cells(report, suite)
    validate_cells(replay_report, suite)
    validate_cells(exact_backend_report, exact_suite)
    if report.episode_content_sha256 != replay_report.episode_content_sha256:
        raise ValueError("baseline replay is not deterministic")

    def logical_content(
        value: EvaluationReport, value_suite: EvaluationSuite
    ) -> tuple[dict[str, object], ...]:
        logical = dict(zip(value_suite.snapshot_ids, value_suite.logical_cell_ids))
        return tuple(
            {
                **asdict(episode),
                "snapshot_id": logical[episode.snapshot_id],
            }
            for episode in value.episodes
        )

    normalized_content_sha256 = _canonical_sha256(logical_content(report, suite))
    if normalized_content_sha256 != _canonical_sha256(
        logical_content(exact_backend_report, exact_suite)
    ):
        raise ValueError("portable and exact baseline episode metrics differ")
    episodes = len(report.episodes)
    return BaselineEvidence(
        baseline.baseline_id,
        "complete",
        episodes,
        sum(value.raw_score for value in report.episodes) / episodes,
        sum(value.invalid_actions for value in report.episodes),
        suite.sha256,
        report.sha256,
        replay_report.sha256,
        exact_backend_report.sha256,
        report.backend_identity_sha256,
        exact_backend_report.backend_identity_sha256,
        normalized_content_sha256,
    )


@dataclass(frozen=True, slots=True)
class BaselineArtifactBundle:
    """Typed primary/replay/exact artifacts consumed by sealed confirmation."""

    baseline: ScriptedBaselineSpec
    suite: EvaluationSuite
    report: EvaluationReport
    replay_report: EvaluationReport
    exact_suite: EvaluationSuite
    exact_backend_report: EvaluationReport
    logical_manifest: CrossBackendEvaluationManifest
    portable_library: SnapshotLibrary
    exact_library: SnapshotLibrary

    def evidence(self):
        return build_baseline_evidence(
            self.baseline,
            self.suite,
            self.report,
            self.replay_report,
            self.exact_suite,
            self.exact_backend_report,
            self.logical_manifest,
            self.portable_library,
            self.exact_library,
        )
