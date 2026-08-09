#!/usr/bin/env python3
"""Development-only recurrent replay/option learning and portable evaluation."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from irisu_env import Action, ActionKind, EventKind, IrisuEnv  # noqa: E402
from irisu_pointer.action import PointerActionSpec  # noqa: E402
from irisu_pointer.sequence_replay import (  # noqa: E402
    SEQUENCE_REPLAY_EVENT_FEATURES,
    SequenceReplayBatch,
    SequenceReplayConfig,
    SequenceReplayLossWeights,
    SequenceReplayModel,
    SequenceReplayTargets,
    batch_steering_example_sequences,
    load_sequence_replay_checkpoint,
    save_sequence_replay_checkpoint,
    sequence_replay_loss,
    train_sequence_replay_step,
)
from irisu_pointer.sequence_replay_legacy import (  # noqa: E402
    LegacyR3eGeometryCheckpoint,
    LegacyR3eGeometryPolicy,
    LegacyR3eGeometryPolicyFactory,
    load_legacy_r3e_geometry_checkpoint,
)
from irisu_pointer.sequence_replay_policy import (  # noqa: E402
    CausalEventHistory,
    SequenceReplayPolicy,
)
from irisu_pointer.replay_supervision import (  # noqa: E402
    ReplayEvidenceIdentity,
    ReplayInputFrame,
    ReplaySteeringCollection,
    collect_replay_steering_supervision,
)
from irisu_pointer.steering import (  # noqa: E402
    ClosedLoopSteeringExpert,
    SteeringDecision,
    SteeringIntent,
)
from irisu_pointer.steering_checkpoint import (  # noqa: E402
    load_goal_conditioned_steering_policy,
    load_steering_checkpoint,
)
from irisu_pointer.steering_learning import (  # noqa: E402
    GoalConditionedSteeringPolicy,
    SteeringExample,
    steering_example_from_decision,
    steering_examples_from_replay,
)
from irisu_rl.actions import SemanticAction  # noqa: E402
from irisu_rl.encoding import EncodedBatch, TeacherStateEncoder  # noqa: E402
from irisu_rl.schema import TEACHER_V1  # noqa: E402


PRIMARY_ROOT = Path(
    os.environ.get("IRISU_PRIMARY_ROOT", "/home/gabe/Documents/irisu")
)
DEFAULT_CONFIG = ROOT / "configs/rl/experiments/sequence-replay-v1.toml"
DEFAULT_REPLAY = (
    PRIMARY_ROOT
    / "reference/replays/raw/internet/irisu_00041449_20100725_182435_7.rpy"
)
DEFAULT_EXACT_WORKER = (
    PRIMARY_ROOT
    / "artifacts/r3/runtime/main-0c48dba-20260723/"
    "exact-runtime-backup/irisu-exact-worker"
)
DEFAULT_PORTABLE = (
    PRIMARY_ROOT
    / "artifacts/r3/runtime/main-0c48dba-20260723/"
    "portable-build/libirisu_clone.so"
)
DEFAULT_V5 = (
    PRIMARY_ROOT
    / "artifacts/r3/development/r3d-survival-v5-20260729/"
    "long-development.pt"
)
DEFAULT_FAST_GEOMETRY = (
    PRIMARY_ROOT
    / "artifacts/r3/development/r3e-sustainable-v1-20260729/"
    "fast-geometry.pt"
)
DEFAULT_EXTENDED_GEOMETRY = (
    PRIMARY_ROOT
    / "artifacts/r3/development/r3e-sustainable-v1-20260729/"
    "extended-geometry.pt"
)
DEFAULT_GEOMETRY_COLLECTION = (
    PRIMARY_ROOT
    / "artifacts/r3/development/r3e-sustainable-v1-20260729/"
    "fast-collection.pt"
)
DEFAULT_OUTPUT = (
    ROOT / "artifacts/r3/development/sequence-replay-20260729-r1"
)
R3D_PATH = ROOT / "benchmarks/rl_r3d_steering.py"
R3E_PATH = ROOT / "benchmarks/rl_r3e_sustainable.py"
REPLAY_BENCHMARK_PATH = ROOT / "benchmarks/r3d_replay_supervision.py"
_FORBIDDEN_PATH = re.compile(
    r"(?:^|[/_.-])(?:sealed|test|canonical)(?:$|[/_.-])", re.IGNORECASE
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _safe_path(path: Path, name: str, *, output: bool = False) -> Path:
    expanded = path.expanduser()
    text = str(expanded.absolute()).replace("\\", "/")
    if _FORBIDDEN_PATH.search(text) or "/artifacts/r3/runs/" in text:
        raise ValueError(f"{name} must remain development-only")
    if expanded.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link")
    if output:
        resolved = expanded.resolve()
        resolved_text = str(resolved).replace("\\", "/")
        if (
            _FORBIDDEN_PATH.search(resolved_text)
            or "/artifacts/r3/runs/" in resolved_text
        ):
            raise ValueError(f"{name} must remain development-only")
        protected = (
            ROOT / ".git",
            ROOT / "benchmarks",
            ROOT / "configs",
            ROOT / "docs",
            ROOT / "python",
            ROOT / "tests",
        )
        if any(value == resolved or value in resolved.parents for value in protected):
            raise ValueError(f"{name} must not overwrite source")
        return resolved
    resolved = expanded.resolve(strict=True)
    resolved_text = str(resolved).replace("\\", "/")
    if (
        _FORBIDDEN_PATH.search(resolved_text)
        or "/artifacts/r3/runs/" in resolved_text
    ):
        raise ValueError(f"{name} must remain development-only")
    return resolved


def _verified_file(path: Path, expected: str, name: str) -> Path:
    resolved = _safe_path(path, name)
    if _file_sha256(resolved) != _sha256(expected, f"{name} identity"):
        raise ValueError(f"{name} SHA-256 mismatch")
    return resolved


def _load_script(name: str, path: Path) -> ModuleType:
    source = path.resolve(strict=True)
    module_name = f"{name}_{_file_sha256(source)[:16]}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _derive_seeds(label: str, count: int) -> tuple[int, ...]:
    if not label or count < 1:
        raise ValueError("seed derivation requires a label and positive count")
    return tuple(
        int.from_bytes(
            hashlib.sha256(f"{label}:{index}".encode()).digest()[:4], "big"
        )
        for index in range(count)
    )


def _distribution(values: Sequence[int | float]) -> dict[str, float]:
    if not values:
        raise ValueError("distribution must not be empty")
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "maximum": float(array.max()),
    }


def _episode_map(result: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    episodes = result.get("episodes")
    if not isinstance(episodes, Sequence):
        raise ValueError("evaluation lacks episodes")
    mapped = {int(value["seed"]): value for value in episodes}
    if len(mapped) != len(episodes):
        raise ValueError("evaluation repeats a seed")
    return mapped


def paired_comparison(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    horizon: int,
) -> dict[str, object]:
    """Report all paired losses, rescues, and preregistered catastrophes."""

    base = _episode_map(baseline)
    other = _episode_map(candidate)
    if set(base) != set(other):
        raise ValueError("paired policies must use identical seeds")
    records: list[dict[str, object]] = []
    score_deltas: list[int] = []
    survival_deltas: list[int] = []
    for seed in sorted(base):
        left, right = base[seed], other[seed]
        left_conversion = left["conversion"]
        right_conversion = right["conversion"]
        base_score = int(left_conversion["final_score"])
        score = int(right_conversion["final_score"])
        base_survival = int(left_conversion["survival_ticks"])
        survival = int(right_conversion["survival_ticks"])
        base_failure = bool(left["gauge_failure"])
        failure = bool(right["gauge_failure"])
        terminal_flip = (
            base_survival == horizon and not base_failure and failure
        )
        severe_joint = (
            base_survival - survival >= 2_000
            and 2 * survival <= base_survival
            and base_score > 0
            and 2 * score <= base_score
        )
        reasons = tuple(
            name
            for name, present in (
                ("terminal_flip", terminal_flip),
                ("severe_joint_collapse", severe_joint),
            )
            if present
        )
        score_deltas.append(score - base_score)
        survival_deltas.append(survival - base_survival)
        records.append(
            {
                "seed": seed,
                "base_score": base_score,
                "candidate_score": score,
                "score_delta": score - base_score,
                "base_survival": base_survival,
                "candidate_survival": survival,
                "survival_delta": survival - base_survival,
                "base_gauge_failure": base_failure,
                "candidate_gauge_failure": failure,
                "gauge_regression": not base_failure and failure,
                "gauge_rescue": base_failure and not failure,
                "catastrophic": bool(reasons),
                "catastrophic_reasons": list(reasons),
            }
        )
    return {
        "definition": {
            "terminal_flip": (
                "baseline survives the horizon without gauge failure and "
                "candidate gauge-fails"
            ),
            "severe_joint_collapse": (
                "candidate loses >=2000 ticks, retains <=50% survival, and "
                "retains <=50% score from a positive-score baseline"
            ),
        },
        "seeds": sorted(base),
        "score_regressions": sum(value["score_delta"] < 0 for value in records),
        "survival_regressions": sum(
            value["survival_delta"] < 0 for value in records
        ),
        "joint_score_survival_regressions": sum(
            value["score_delta"] < 0 and value["survival_delta"] < 0
            for value in records
        ),
        "gauge_regressions": sum(value["gauge_regression"] for value in records),
        "gauge_rescues": sum(value["gauge_rescue"] for value in records),
        "catastrophic_regressions": sum(
            value["catastrophic"] for value in records
        ),
        "catastrophic_seeds": [
            value for value in records if value["catastrophic"]
        ],
        "score_delta": _distribution(score_deltas),
        "survival_delta": _distribution(survival_deltas),
        "pairs": records,
    }


@dataclass(frozen=True, slots=True)
class InputIdentity:
    replay: str
    exact_worker: str
    portable_runtime: str
    frozen_v5: str
    fast_geometry: str
    extended_geometry: str
    geometry_collection: str

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> InputIdentity:
        values = config["inputs"]
        return cls(
            str(values["trusted_replay_sha256"]),
            str(values["exact_worker_sha256"]),
            str(values["portable_runtime_sha256"]),
            str(values["frozen_v5_sha256"]),
            str(values["fast_geometry_sha256"]),
            str(values["extended_geometry_sha256"]),
            str(values["geometry_collection_sha256"]),
        )


@dataclass(frozen=True, slots=True)
class OptionBranchOutcome:
    name: str
    decision: SteeringDecision
    survival_ticks: int
    alive: bool
    final_gauge: int
    score_gain: int
    qualifying_clear_gain: int
    rotten_events: int
    invalid_actions: int

    @property
    def objective(self) -> tuple[int, ...]:
        return (
            int(self.alive),
            self.survival_ticks,
            self.final_gauge,
            self.qualifying_clear_gain,
            self.score_gain,
            -self.rotten_events,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "survival_ticks": self.survival_ticks,
            "alive": self.alive,
            "final_gauge": self.final_gauge,
            "score_gain": self.score_gain,
            "qualifying_clear_gain": self.qualifying_clear_gain,
            "rotten_events": self.rotten_events,
            "invalid_actions": self.invalid_actions,
            "objective": list(self.objective),
        }


def _event_kind(event: Mapping[str, Any]) -> int | None:
    raw = event.get("kind")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    name = event.get("kind_name")
    if isinstance(name, str):
        try:
            return int(EventKind[name.upper()])
        except KeyError:
            return None
    return None


def _step_decision(
    env: IrisuEnv,
    observation: Mapping[str, Any],
    decision: SteeringDecision,
    *,
    stop_tick: int,
) -> tuple[Mapping[str, Any], bool, bool, list[Mapping[str, Any]]]:
    current = observation
    terminated = bool(current.get("terminated", False))
    truncated = bool(current.get("truncated", False))
    events: list[Mapping[str, Any]] = []
    for action in decision.primitive_actions():
        if terminated or truncated:
            break
        remaining = stop_tick - int(current.get("tick", 0))
        if remaining <= 0:
            break
        duration = (
            int(action.wait_ticks)
            if ActionKind.parse(action.kind) is ActionKind.WAIT
            else 1
        )
        chosen = action if duration <= remaining else Action.wait(remaining)
        before = int(current.get("tick", 0))
        current, _reward, terminated, truncated, info = env.step(chosen)
        if not before < int(current.get("tick", 0)) <= stop_tick:
            raise RuntimeError("option branch violated its public tick budget")
        events.extend(
            value
            for value in info.get("events", ())
            if isinstance(value, Mapping)
        )
    return current, terminated, truncated, events


def _evaluate_option_branch(
    env: IrisuEnv,
    snapshot: object,
    initial: Mapping[str, Any],
    *,
    name: str,
    decision: SteeringDecision,
    continuation_factory: Callable[[], object],
    seed: int,
    horizon_ticks: int,
) -> OptionBranchOutcome:
    current = env.restore_state(snapshot)
    if not isinstance(current, Mapping):
        raise TypeError("portable restore must return a public observation")
    start_tick = int(initial.get("tick", 0))
    stop_tick = start_tick + horizon_ticks
    policy = continuation_factory()
    getattr(policy, "reset")(seed)
    current, terminated, truncated, events = _step_decision(
        env, current, decision, stop_tick=stop_tick
    )
    decisions = 0
    while (
        not terminated
        and not truncated
        and int(current.get("tick", 0)) < stop_tick
    ):
        if decisions >= horizon_ticks * 2:
            raise RuntimeError("option continuation exceeded its decision budget")
        continuation = getattr(policy, "predict")(current)
        if not isinstance(continuation, SteeringDecision):
            raise TypeError("option continuation must expose SteeringDecision")
        current, terminated, truncated, transition_events = _step_decision(
            env, current, continuation, stop_tick=stop_tick
        )
        events.extend(transition_events)
        decisions += 1
    elapsed = int(current.get("tick", 0)) - start_tick
    invalid = sum(
        _event_kind(value) == int(EventKind.INVALID_ACTION) for value in events
    )
    return OptionBranchOutcome(
        name,
        decision,
        elapsed,
        not terminated and elapsed == horizon_ticks,
        int(current.get("gauge", 0)),
        int(current.get("score", 0)) - int(initial.get("score", 0)),
        int(current.get("qualifying_clear_count", 0))
        - int(initial.get("qualifying_clear_count", 0)),
        sum(_event_kind(value) == int(EventKind.ROTTEN) for value in events),
        invalid,
    )


def _select_option_branch(
    env: IrisuEnv,
    observation: Mapping[str, Any],
    candidates: Sequence[tuple[str, SteeringDecision]],
    *,
    continuation_factory: Callable[[], object],
    seed: int,
    horizon_ticks: int,
) -> tuple[OptionBranchOutcome, tuple[OptionBranchOutcome, ...]]:
    if getattr(env, "physics_backend", None) != "portable":
        raise ValueError("option branching requires the portable environment")
    if not candidates or candidates[0][0] != "incumbent":
        raise ValueError("option branching requires incumbent slot zero")
    snapshot = env.clone_state()
    outcomes: list[OptionBranchOutcome] = []
    try:
        for name, decision in candidates:
            outcomes.append(
                _evaluate_option_branch(
                    env,
                    snapshot,
                    observation,
                    name=name,
                    decision=decision,
                    continuation_factory=continuation_factory,
                    seed=seed,
                    horizon_ticks=horizon_ticks,
                )
            )
    finally:
        restored = env.restore_state(snapshot)
        if int(restored.get("tick", -1)) != int(observation.get("tick", -2)):
            raise RuntimeError("option branching failed transactional restore")
    incumbent = outcomes[0]
    if incumbent.invalid_actions:
        raise RuntimeError("incumbent option emitted an invalid action")
    eligible = [
        value
        for value in outcomes
        if not value.invalid_actions
        and int(value.alive) >= int(incumbent.alive)
        and value.survival_ticks >= incumbent.survival_ticks
        and value.final_gauge >= incumbent.final_gauge
    ]
    winner = max(
        eligible,
        key=lambda value: (
            value.objective,
            -next(
                index
                for index, (name, _decision) in enumerate(candidates)
                if name == value.name
            ),
        ),
    )
    return winner, tuple(outcomes)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        config = tomllib.load(stream)
    required = {
        "version",
        "status",
        "deployable",
        "canonical_r3_evidence",
        "sealed_evaluation_allowed",
        "objective",
        "inputs",
        "frozen_v5",
        "model",
        "training",
        "selection",
        "deployment",
        "evaluation",
        "catastrophic_regression",
    }
    if (
        set(config) != required
        or config["version"] != "sequence-replay-v1"
        or config["status"] != "development_only_not_canonical_evidence"
        or config["deployable"] is not False
        or config["canonical_r3_evidence"] is not False
        or config["sealed_evaluation_allowed"] is not False
        or config["training"].get("development_seed_label")
        == config["evaluation"].get("suite_label")
        or config["evaluation"].get("training_seed_overlap_allowed") is not False
        or config["evaluation"].get("sealed_test_allowed") is not False
    ):
        raise ValueError("sequence-replay config is not development-only")
    return config


def _source_identity(config_path: Path) -> dict[str, object]:
    names = (
        "benchmarks/rl_sequence_replay.py",
        "benchmarks/r3d_replay_supervision.py",
        "benchmarks/rl_r3d_steering.py",
        "benchmarks/rl_r3e_sustainable.py",
        "configs/rl/experiments/sequence-replay-v1.toml",
        "python/irisu_pointer/sequence_replay.py",
        "python/irisu_pointer/sequence_replay_campaign.py",
        "python/irisu_pointer/sequence_replay_legacy.py",
        "python/irisu_pointer/sequence_replay_policy.py",
        "python/irisu_pointer/replay_supervision.py",
        "python/irisu_pointer/steering.py",
        "python/irisu_pointer/steering_learning.py",
        "python/irisu_pointer/steering_checkpoint.py",
        "python/irisu_pointer/geometry_learning.py",
        "python/irisu_pointer/geometry_search.py",
        "python/irisu_rl/encoding.py",
        "python/irisu_rl/schema.py",
        "pyproject.toml",
        "uv.lock",
    )
    files: dict[str, str] = {}
    for name in names:
        path = ROOT / name
        if path.exists():
            files[name] = _file_sha256(path)
    if str(config_path.relative_to(ROOT)) not in files:
        files[str(config_path.relative_to(ROOT))] = _file_sha256(config_path)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "schema": "irisu-sequence-replay-source-v1",
        "git_revision": revision,
        "files": dict(sorted(files.items())),
    }
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _collect_exact_replay(
    replay_benchmark: ModuleType,
    *,
    replay_path: Path,
    worker_path: Path,
) -> tuple[
    tuple[SteeringExample, ...],
    ReplaySteeringCollection,
    dict[str, object],
]:
    """Reproduce the trusted score and return chronological strict shot labels."""

    replay_bytes = replay_path.read_bytes()
    parser = replay_benchmark._load_replay_parser()
    parsed = parser.parse_replay(replay_bytes, "padded")
    if (
        int(parsed.header.mode) != 0
        or int(parsed.header.final_score) != 41_449
        or len(parsed.frames) != 47_019
    ):
        raise RuntimeError("trusted replay header or frame count changed")
    seed = int(parsed.header.seed) & 0xFFFFFFFF
    frames = tuple(
        ReplayInputFrame.from_object(value, index)
        for index, value in enumerate(parsed.frames)
    )
    worker_sha256 = _file_sha256(worker_path)
    encoder = TeacherStateEncoder()
    pointer_spec = PointerActionSpec()
    config_hash, exact_identity = replay_benchmark._probe_exact_identity(
        IrisuEnv, worker_path, worker_sha256, seed
    )
    evidence = ReplayEvidenceIdentity(
        source_revision=_canonical_sha256(
            {
                "collector": _file_sha256(
                    ROOT / "python/irisu_pointer/replay_supervision.py"
                ),
                "benchmark": _file_sha256(REPLAY_BENCHMARK_PATH),
                "sequence_benchmark": _file_sha256(Path(__file__)),
            }
        ),
        replay_sha256=_file_sha256(replay_path),
        runtime_sha256=worker_sha256,
        config_hash=config_hash,
        observation_schema_sha256=encoder.schema.sha256,
        pointer_spec_sha256=pointer_spec.sha256,
        mapped_runtime_sha256=str(exact_identity["mapped_library"]["sha256"]),
    )
    with IrisuEnv(
        physics_backend="exact",
        worker_path=str(worker_path),
        diagnostic_hashes=False,
    ) as env:
        observed = replay_benchmark._ObservedEnvironment(env)
        collection = collect_replay_steering_supervision(
            observed,
            frames,
            seed=seed,
            identity=evidence,
            pointer_spec=pointer_spec,
        )
        final = observed.observation
        if final is None:
            raise RuntimeError("trusted replay produced no terminal observation")
    if (
        collection.metrics.frames != len(frames)
        or collection.metrics.final_score != 41_449
        or int(final["level"]) != int(parsed.header.highest_level)
        or int(final["highest_chain"]) != int(parsed.header.highest_chain)
        or collection.metrics.invalid_actions
    ):
        raise RuntimeError("trusted replay reproduction failed")
    examples = steering_examples_from_replay(
        collection, encoder=encoder, pointer_spec=pointer_spec
    )
    if not examples:
        raise RuntimeError("trusted replay produced no strict labels")
    return examples, collection, {
        "replay_sha256": _file_sha256(replay_path),
        "exact_worker_sha256": worker_sha256,
        "mapped_runtime_sha256": exact_identity["mapped_library"]["sha256"],
        "evidence_sha256": evidence.sha256,
        "collection_sha256": collection.sha256,
        "dataset_examples": len(examples),
        "first_hit_labels": len(collection.shots),
        "frame_count": len(frames),
        "seed": seed,
        "metrics": collection.metrics.manifest(),
        "reproduced": {
            "score": int(final["score"]),
            "level": int(final["level"]),
            "highest_chain": int(final["highest_chain"]),
        },
        "destination_semantics": (
            "nearest-visible-same-color-peer-inference-v1; behavioral "
            "inference, not recovered human intent"
        ),
    }


@dataclass(frozen=True, slots=True)
class _PolicySequence:
    name: str
    examples: tuple[SteeringExample, ...]
    events: torch.Tensor
    pair_weights: torch.Tensor
    value_target: torch.Tensor
    value_mask: torch.Tensor
    viability_target: torch.Tensor
    outcome_target: torch.Tensor

    def __post_init__(self) -> None:
        length = len(self.examples)
        if (
            not self.name
            or length < 1
            or self.events.shape
            != (length, len(SEQUENCE_REPLAY_EVENT_FEATURES))
            or self.pair_weights.shape != (length,)
            or self.value_target.shape != (length,)
            or self.value_mask.shape != (length,)
            or self.viability_target.shape != (length, 2)
            or self.outcome_target.shape != (length, 2)
        ):
            raise ValueError("sequence training episode has inconsistent tensors")


def _empty_targets(length: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(length),
        torch.zeros(length, dtype=torch.bool),
        torch.zeros(length, 2),
        torch.zeros(length, 2),
    )


def _replay_sequence(
    examples: Sequence[SteeringExample],
    collection: ReplaySteeringCollection,
) -> _PolicySequence:
    """Turn chronological exact shot labels into one causal replay episode."""

    history = CausalEventHistory()
    intents = tuple(SteeringIntent)
    normalized: list[SteeringExample] = []
    events: list[torch.Tensor] = []
    for example in examples:
        try:
            ordinal = int(example.episode_identity.rsplit(":", 1)[1])
        except (IndexError, ValueError) as exc:
            raise ValueError("replay label lost its collection ordinal") from exc
        if not 0 <= ordinal < len(collection.shots):
            raise ValueError("replay label ordinal is out of bounds")
        shot = collection.shots[ordinal]
        observation = collection.shot_observations[ordinal]
        event = history.features(observation)
        action = (
            SemanticAction.strong(0.5, 0.5)
            if ActionKind.parse(shot.action_kind) is ActionKind.STRONG_SHOT
            else SemanticAction.weak(0.5, 0.5)
        )
        decision = SteeringDecision(
            action,
            intents[example.intent_index],
            source_body_id=shot.target_body_id,
            destination_body_id=shot.destination_body_id,
            reason="trusted replay first-hit supervision",
        )
        normalized.append(
            replace(
                example,
                episode_identity=(
                    f"sequence-replay:trusted-human:{collection.identity.replay_sha256}"
                ),
            )
        )
        events.append(event)
        history.commit(observation, decision)
    value, mask, viability, outcome = _empty_targets(len(normalized))
    return _PolicySequence(
        "trusted-exact-human-replay",
        tuple(normalized),
        torch.stack(events),
        torch.full((len(normalized),), 0.5),
        value,
        mask,
        viability,
        outcome,
    )


def _base_policy_for_model(
    model: object,
    *,
    checkpoint_sha256: str,
    options: Mapping[str, Any],
) -> GoalConditionedSteeringPolicy:
    return GoalConditionedSteeringPolicy(
        model,  # type: ignore[arg-type]
        cooldown_ticks=int(options["cooldown_ticks"]),
        minimum_pair_closure_sizes=float(options["minimum_pair_closure_sizes"]),
        impact_side_sizes=float(options["impact_side_sizes"]),
        impact_below_sizes=float(options["impact_below_sizes"]),
        source_velocity_lead_ticks=float(options["source_velocity_lead_ticks"]),
        ticks_per_second=float(options["ticks_per_second"]),
        act_logit_bias=float(options["act_logit_bias"]),
        artifact_sha256=checkpoint_sha256,
    )


def _decision_key(decision: SteeringDecision) -> tuple[object, ...]:
    action = decision.action
    return (
        str(action.kind),
        int(action.wait_ticks),
        round(float(action.x_norm), 9),
        round(float(action.y_norm), 9),
        decision.intent.value,
        decision.source_body_id,
        decision.destination_body_id,
    )


def _branch_candidates(
    incumbent: SteeringDecision,
    *,
    raw_v5: SteeringDecision,
    expert: SteeringDecision,
) -> tuple[tuple[str, SteeringDecision], ...]:
    values: list[tuple[str, SteeringDecision]] = [("incumbent", incumbent)]
    proposed = (
        ("raw_v5", raw_v5),
        ("closed_loop_expert", expert),
        *(
            (
                f"wait_{ticks}",
                SteeringDecision(
                    SemanticAction.wait(ticks),
                    SteeringIntent.WAIT,
                    reason=f"causal restraint option {ticks}",
                ),
            )
            for ticks in (4, 8, 16)
        ),
    )
    seen = {_decision_key(incumbent)}
    for name, decision in proposed:
        key = _decision_key(decision)
        if key not in seen:
            seen.add(key)
            values.append((name, decision))
    return tuple(values)


def _collect_dagger_sequences(
    *,
    library: Path,
    seeds: Sequence[int],
    ticks: int,
    source_identity: str,
    behavior_factory: Callable[[], LegacyR3eGeometryPolicy],
    continuation_factory: Callable[[], GoalConditionedSteeringPolicy],
    training: Mapping[str, Any],
) -> tuple[tuple[_PolicySequence, ...], dict[str, object]]:
    """Collect behavior-visited expert labels and sparse causal option labels."""

    started_wall = time.monotonic()
    started_cpu = time.process_time()
    query_limit = int(training["branch_queries_per_episode"])
    query_stride = int(training["branch_query_stride_decisions"])
    branch_horizon = int(training["branch_horizon_ticks"])
    if min(ticks, query_limit, query_stride, branch_horizon) < 1:
        raise ValueError("development collection budgets must be positive")
    sequences: list[_PolicySequence] = []
    branch_records: list[dict[str, object]] = []
    totals: Counter[str] = Counter()
    with IrisuEnv(
        library_path=library,
        physics_backend="portable",
        config={"max_episode_ticks": ticks + branch_horizon + 32},
    ) as env:
        if Path(env.library_path).resolve() != library:
            raise RuntimeError("DAgger loaded a foreign portable runtime")
        runner = env.runner_identity_manifest()
        for seed in seeds:
            behavior = behavior_factory()
            expert = ClosedLoopSteeringExpert()
            behavior.reset(int(seed))
            expert.reset(int(seed))
            history = CausalEventHistory()
            observation, reset_info = env.reset(seed=int(seed))
            if int(reset_info.get("config_hash", -1)) != int(
                runner["config_hash"]
            ):
                raise RuntimeError("DAgger reset configuration changed")
            episode_identity = f"sequence-replay:dagger:{int(seed):08x}"
            provenance = _canonical_sha256(
                {
                    "source_identity": source_identity,
                    "runtime_sha256": _file_sha256(library),
                    "seed": int(seed),
                    "ticks": ticks,
                    "behavior_checkpoint": behavior.checkpoint.sha256,
                    "branch_horizon": branch_horizon,
                }
            )
            examples: list[SteeringExample] = []
            event_values: list[torch.Tensor] = []
            pair_weights: list[float] = []
            value_targets: list[float] = []
            value_masks: list[bool] = []
            viability_targets: list[tuple[float, float]] = []
            outcome_targets: list[tuple[float, float]] = []
            queries = 0
            decisions = 0
            terminated = bool(observation.get("terminated", False))
            truncated = bool(observation.get("truncated", False))
            episode_events: Counter[int] = Counter()
            while (
                not terminated
                and not truncated
                and int(observation.get("tick", 0)) < ticks
            ):
                if decisions >= ticks * 2:
                    raise RuntimeError("DAgger exceeded its decision budget")
                event = history.features(observation)
                behavior_decision = behavior.predict(observation)
                raw_v5 = behavior.base_policy._last_decision  # noqa: SLF001
                if not isinstance(raw_v5, SteeringDecision):
                    raise RuntimeError("legacy behavior lost its v5 incumbent")
                expert_decision = expert.predict(observation)
                target_decision = expert_decision
                target_outcome: OptionBranchOutcome | None = None
                if queries < query_limit and decisions % query_stride == 0:
                    candidates = _branch_candidates(
                        behavior_decision,
                        raw_v5=raw_v5,
                        expert=expert_decision,
                    )
                    winner, outcomes = _select_option_branch(
                        env,
                        observation,
                        candidates,
                        continuation_factory=continuation_factory,
                        seed=int(seed),
                        horizon_ticks=branch_horizon,
                    )
                    queries += 1
                    totals["queries"] += 1
                    totals["branch_outcomes"] += len(outcomes)
                    target_decision = winner.decision
                    target_outcome = winner
                    corrected = _decision_key(target_decision) != _decision_key(
                        behavior_decision
                    )
                    totals["causal_corrections"] += int(corrected)
                    branch_records.append(
                        {
                            "seed": int(seed),
                            "tick": int(observation.get("tick", 0)),
                            "winner": winner.name,
                            "corrected_incumbent": corrected,
                            "options": [value.manifest() for value in outcomes],
                        }
                    )
                example = steering_example_from_decision(
                    observation,
                    target_decision,
                    episode_identity=episode_identity,
                    provenance_sha256=provenance,
                    require_representable_template=False,
                )
                if example is None:
                    target_outcome = None
                    example = steering_example_from_decision(
                        observation,
                        behavior_decision,
                        episode_identity=episode_identity,
                        provenance_sha256=provenance,
                        require_representable_template=False,
                    )
                    totals["unrepresentable_target_fallbacks"] += 1
                if example is not None:
                    examples.append(example)
                    event_values.append(event)
                    pair_weights.append(float(example.is_shot))
                    if target_outcome is None:
                        value_targets.append(0.0)
                        value_masks.append(False)
                        viability_targets.append((0.0, 0.0))
                        outcome_targets.append((0.0, 0.0))
                    else:
                        gauge_max = max(
                            int(observation.get("gauge_max", 0)), 1
                        )
                        value_targets.append(
                            math.log1p(max(0, target_outcome.score_gain)) / 8.0
                            + target_outcome.final_gauge / gauge_max
                            + math.log1p(
                                max(0, target_outcome.qualifying_clear_gain)
                            )
                        )
                        value_masks.append(True)
                        gauge_failure = (
                            not target_outcome.alive
                            and target_outcome.final_gauge <= 1
                        )
                        viability_targets.append(
                            (float(target_outcome.alive), float(gauge_failure))
                        )
                        outcome_targets.append(
                            (
                                (
                                    target_outcome.final_gauge
                                    - int(observation.get("gauge", 0))
                                )
                                / gauge_max,
                                math.log1p(
                                    max(
                                        0,
                                        target_outcome.qualifying_clear_gain,
                                    )
                                ),
                            )
                        )
                history.commit(observation, behavior_decision)
                observation, terminated, truncated, transition_events = (
                    _step_decision(
                        env,
                        observation,
                        behavior_decision,
                        stop_tick=ticks,
                    )
                )
                for value in transition_events:
                    kind = _event_kind(value)
                    if kind is not None:
                        episode_events[kind] += 1
                decisions += 1
            if not examples:
                raise RuntimeError("DAgger episode produced no representable labels")
            totals["decisions"] += decisions
            totals["labels"] += len(examples)
            totals["shots"] += sum(value.is_shot for value in examples)
            totals["waits"] += sum(not value.is_shot for value in examples)
            totals["qualifying_clears"] += int(
                observation.get("qualifying_clear_count", 0)
            )
            totals["rotten"] += episode_events[int(EventKind.ROTTEN)]
            sequences.append(
                _PolicySequence(
                    episode_identity,
                    tuple(examples),
                    torch.stack(event_values),
                    torch.tensor(pair_weights),
                    torch.tensor(value_targets),
                    torch.tensor(value_masks, dtype=torch.bool),
                    torch.tensor(viability_targets),
                    torch.tensor(outcome_targets),
                )
            )
    evidence = {
        "schema": "irisu-sequence-replay-dagger-v1",
        "development_only": True,
        "sealed_test_material_used": False,
        "seed_label": str(training["development_seed_label"]),
        "seeds": [int(value) for value in seeds],
        "ticks_per_seed": ticks,
        "branch_horizon_ticks": branch_horizon,
        "branch_query_limit_per_seed": query_limit,
        "runner": runner,
        "counts": dict(sorted(totals.items())),
        "episodes": [
            {
                "name": value.name,
                "examples": len(value.examples),
                "event_sha256": hashlib.sha256(
                    value.events.numpy().tobytes()
                ).hexdigest(),
            }
            for value in sequences
        ],
        "branches": branch_records,
        "cost": {
            "wall_seconds": time.monotonic() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
        },
    }
    return tuple(sequences), {
        **evidence,
        "sha256": _canonical_sha256(evidence),
    }


def _base_factory(
    path: Path, sha256: str, options: Mapping[str, Any]
) -> Callable[[], object]:
    def factory() -> object:
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

    return factory


def _sample_policy_batch(
    sequences: Sequence[_PolicySequence],
    *,
    rng: random.Random,
    batch_episodes: int,
    window: int,
    burn_in: int,
    device: torch.device | str = "cpu",
) -> SequenceReplayBatch:
    if not sequences or min(batch_episodes, window) < 1:
        raise ValueError("policy batch sampling requires data and a budget")
    chosen: list[tuple[_PolicySequence, int, int]] = []
    for _lane in range(batch_episodes):
        episode = sequences[rng.randrange(len(sequences))]
        length = min(window, len(episode.examples))
        start = rng.randrange(len(episode.examples) - length + 1)
        chosen.append((episode, start, start + length))
    batch = batch_steering_example_sequences(
        [
            value.examples[start:stop]
            for value, start, stop in chosen
        ],
        event_sequences=[
            value.events[start:stop] for value, start, stop in chosen
        ],
        pair_weight_sequences=[
            value.pair_weights[start:stop]
            for value, start, stop in chosen
        ],
        device=device,
    )
    policy_weight = batch.targets.valid_mask.to(batch.global_features.dtype)
    pair_weight = batch.targets.pair_weight.clone()
    assert pair_weight is not None
    value_target = torch.zeros_like(policy_weight)
    value_mask = torch.zeros_like(batch.targets.valid_mask)
    viability = torch.zeros(
        *policy_weight.shape,
        2,
        dtype=policy_weight.dtype,
        device=policy_weight.device,
    )
    outcomes = torch.zeros_like(viability)
    for lane, (episode, start, stop) in enumerate(chosen):
        length = stop - start
        scale = 0.5 if episode.name == "trusted-exact-human-replay" else 1.0
        policy_weight[:length, lane] *= scale
        pair_weight[:length, lane] *= scale
        value_target[:length, lane].copy_(
            episode.value_target[start:stop].to(value_target.device)
        )
        value_mask[:length, lane].copy_(
            episode.value_mask[start:stop].to(value_mask.device)
        )
        viability[:length, lane].copy_(
            episode.viability_target[start:stop].to(viability.device)
        )
        outcomes[:length, lane].copy_(
            episode.outcome_target[start:stop].to(outcomes.device)
        )
        # Mid-episode windows use a short causal burn-in before receiving loss.
        burn = min(burn_in if start else 0, max(0, length - 1))
        if burn:
            policy_weight[:burn, lane] = 0
            pair_weight[:burn, lane] = 0
            value_mask[:burn, lane] = False
    targets = replace(
        batch.targets,
        policy_weight=policy_weight,
        pair_weight=pair_weight,
        return_target=value_target,
        value_mask=value_mask,
        viability_target=viability,
        viability_mask=value_mask,
        outcome_target=outcomes,
        outcome_mask=value_mask,
    )
    return replace(batch, targets=targets)


def _load_geometry_batches(
    path: Path,
    *,
    checkpoint: LegacyR3eGeometryCheckpoint,
    expected_collection_sha256: str,
    expected_vocabulary_sha256: str,
) -> tuple[tuple[SequenceReplayBatch, ...], dict[str, object]]:
    """Convert the immutable R3e collection without inventing slot IDs."""

    if _file_sha256(path) != expected_collection_sha256:
        raise ValueError("geometry collection changed after input verification")
    payload = torch.load(path, map_location="cpu", weights_only=True)
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
        raise ValueError("legacy geometry collection fields differ")
    metadata = payload["metadata"]
    if (
        payload["format"] != "irisu-r3e-geometry-collection-v1"
        or payload["schema_sha256"] != TEACHER_V1.sha256
        or payload["candidate_set_sha256"] != expected_vocabulary_sha256
        or int(payload["candidate_count"]) != checkpoint.model.candidate_count
        or not isinstance(metadata, Mapping)
        or metadata.get("development_only") is not True
        or metadata.get("canonical_r3_evidence") is not False
        or metadata.get("sealed_test_material_used") is not False
        or metadata.get("candidate_vocabulary_sha256")
        != expected_vocabulary_sha256
    ):
        raise ValueError("legacy geometry collection identity is incompatible")
    raw_examples = payload["examples"]
    if not isinstance(raw_examples, list) or not raw_examples:
        raise ValueError("legacy geometry collection has no examples")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for raw in raw_examples:
        if not isinstance(raw, Mapping):
            raise ValueError("legacy geometry example is malformed")
        parts = str(raw["episode_identity"]).split(":")
        if len(parts) < 2:
            raise ValueError("legacy geometry episode identity is malformed")
        grouped.setdefault(parts[1], []).append(raw)
    batches: list[SequenceReplayBatch] = []
    event_index = {
        name: index for index, name in enumerate(SEQUENCE_REPLAY_EVENT_FEATURES)
    }
    for episode, values in sorted(grouped.items()):
        values.sort(key=lambda value: int(value["source_tick"][0]))
        length = len(values)
        width = max(
            int(torch.as_tensor(value["body_mask"])[0].nonzero()[-1]) + 1
            for value in values
        )
        globals_tensor = torch.stack(
            [torch.as_tensor(value["global_features"])[0] for value in values]
        )
        bodies = torch.stack(
            [
                torch.as_tensor(value["body_features"])[0, :width]
                for value in values
            ]
        )
        body_mask = torch.stack(
            [
                torch.as_tensor(value["body_mask"])[0, :width]
                for value in values
            ]
        ).bool()
        source = torch.tensor(
            [int(value["source_index"]) for value in values], dtype=torch.long
        )
        destination = torch.tensor(
            [int(value["destination_index"]) for value in values],
            dtype=torch.long,
        )
        geometry_index = torch.tensor(
            [int(value["candidate_index"]) for value in values],
            dtype=torch.long,
        )
        improved = torch.tensor(
            [
                bool(value["improved_over_incumbent"])
                and int(value["candidate_index"]) != 0
                for value in values
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            base_logits = checkpoint.model(
                globals_tensor,
                bodies,
                body_mask,
                source,
                destination,
            )
        events = torch.zeros(length, len(SEQUENCE_REPLAY_EVENT_FEATURES))
        ticks = [int(value["source_tick"][0]) for value in values]
        for index in range(1, length):
            elapsed = max(0, ticks[index] - ticks[index - 1])
            events[index, event_index["previous_action_shot"]] = 1.0
            events[index, event_index["elapsed_ticks_log_scaled"]] = (
                math.log1p(elapsed) / 4.0
            )
            events[index, event_index["time_since_shot_log_scaled"]] = (
                math.log1p(elapsed) / 8.0
            )
        valid = torch.ones(length, 1, dtype=torch.bool)
        reset = torch.zeros_like(valid)
        reset[0] = True
        long_zeros = torch.zeros(length, 1, dtype=torch.long)
        targets = SequenceReplayTargets(
            valid,
            torch.ones_like(long_zeros),
            long_zeros.clone(),
            source[:, None],
            destination[:, None],
            long_zeros.clone(),
            long_zeros.clone(),
            long_zeros.clone(),
            torch.zeros(length, 1),
            torch.zeros(length, 1),
            geometry_index=geometry_index[:, None],
            geometry_weight=torch.ones(length, 1),
            geometry_apply_target=improved[:, None],
        )
        batches.append(
            SequenceReplayBatch(
                globals_tensor[:, None],
                bodies[:, None],
                body_mask[:, None],
                events[:, None],
                reset,
                targets,
                base_geometry_logits=base_logits[:, None],
                geometry_source_index=source[:, None],
                geometry_destination_index=destination[:, None],
                geometry_pair_mask=valid.clone(),
                # Availability was not retained by legacy v1; this explicit
                # all-slot approximation is reported as a limitation.
                geometry_candidate_mask=torch.ones(
                    length,
                    1,
                    checkpoint.model.candidate_count,
                    dtype=torch.bool,
                ),
            )
        )
    evidence = {
        "schema": "irisu-sequence-replay-legacy-geometry-conversion-v1",
        "collection_sha256": expected_collection_sha256,
        "dataset_sha256": str(payload["dataset_sha256"]),
        "candidate_vocabulary_sha256": expected_vocabulary_sha256,
        "base_geometry_checkpoint_sha256": checkpoint.sha256,
        "seeds": list(metadata.get("seeds", ())),
        "examples": len(raw_examples),
        "improved_examples": sum(
            bool(value["improved_over_incumbent"]) for value in raw_examples
        ),
        "episodes": {
            name: len(values) for name, values in sorted(grouped.items())
        },
        "availability": (
            "legacy v1 collection omitted per-state availability; all 32 "
            "semantically fixed slots are exposed during auxiliary training"
        ),
    }
    return tuple(batches), {**evidence, "sha256": _canonical_sha256(evidence)}


def _sample_geometry_batch(
    batches: Sequence[SequenceReplayBatch],
    *,
    rng: random.Random,
    window: int,
    burn_in: int,
) -> SequenceReplayBatch:
    original = batches[rng.randrange(len(batches))]
    length = min(window, original.time)
    start = rng.randrange(original.time - length + 1)
    batch = original.time_slice(start, start + length)
    reset = batch.reset_before.clone()
    reset[0] = True
    targets = batch.targets
    geometry_weight = targets.geometry_weight.clone()
    assert geometry_weight is not None
    burn = min(burn_in if start else 0, max(0, length - 1))
    if burn:
        geometry_weight[:burn] = 0
    return replace(
        batch,
        reset_before=reset,
        targets=replace(targets, geometry_weight=geometry_weight),
    )


@torch.no_grad()
def _offline_geometry_metrics(
    model: SequenceReplayModel,
    batches: Sequence[SequenceReplayBatch],
) -> dict[str, object]:
    losses: list[dict[str, float]] = []
    correct = total = gate_correct = 0
    model.eval()
    for batch in batches:
        loss, output = sequence_replay_loss(model, batch)
        losses.append(loss.scalars())
        assert output.geometry_logits is not None
        assert output.geometry_gate_logit is not None
        target = batch.targets
        assert target.geometry_index is not None
        assert target.geometry_weight is not None
        assert target.geometry_apply_target is not None
        active = target.geometry_weight > 0
        prediction = output.geometry_logits.argmax(dim=-1)
        correct += int(((prediction == target.geometry_index) & active).sum())
        gate_prediction = output.geometry_gate_logit >= 0
        gate_target = target.geometry_apply_target.bool()
        gate_correct += int(((gate_prediction == gate_target) & active).sum())
        total += int(active.sum())
    return {
        "loss": {
            name: float(np.mean([value[name] for value in losses]))
            for name in losses[0]
        },
        "candidate_accuracy": correct / total if total else 0.0,
        "gate_accuracy": gate_correct / total if total else 0.0,
        "examples": total,
    }


@torch.no_grad()
def _offline_policy_metrics(
    model: SequenceReplayModel,
    sequences: Sequence[_PolicySequence],
    *,
    seed: int,
    training: Mapping[str, Any],
    samples: int = 8,
) -> dict[str, object]:
    rng = random.Random(seed)
    losses: list[dict[str, float]] = []
    counts: Counter[str] = Counter()
    model.eval()
    for _index in range(samples):
        batch = _sample_policy_batch(
            sequences,
            rng=rng,
            batch_episodes=int(training["batch_episodes"]),
            window=int(training["burn_in"]) + int(training["tbptt_steps"]),
            burn_in=int(training["burn_in"]),
        )
        loss, output = sequence_replay_loss(model, batch)
        losses.append(loss.scalars())
        target = batch.targets
        active = target.valid_mask & (
            target.policy_weight is not None
        )
        assert target.policy_weight is not None
        active &= target.policy_weight > 0
        act_prediction = output.act_logits.argmax(dim=-1)
        counts["act_total"] += int(active.sum())
        counts["act_correct"] += int(
            ((act_prediction == target.act_index) & active).sum()
        )
        waits = active & (target.act_index == 0)
        shots = active & (target.act_index == 1)
        counts["restraint_total"] += int(waits.sum())
        counts["restraint_correct"] += int(
            ((act_prediction == 0) & waits).sum()
        )
        counts["shot_total"] += int(shots.sum())
        counts["shot_correct"] += int(
            ((act_prediction == 1) & shots).sum()
        )
        pair_rows = shots
        if target.pair_weight is not None:
            pair_rows &= target.pair_weight > 0
        if bool(pair_rows.any()):
            width = output.pair_logits.shape[-1]
            pair_prediction = output.pair_logits.flatten(-2).argmax(dim=-1)
            pair_target = target.source_index * width + target.destination_index
            counts["pair_total"] += int(pair_rows.sum())
            counts["pair_correct"] += int(
                ((pair_prediction == pair_target) & pair_rows).sum()
            )
    mean_loss = {
        name: float(np.mean([value[name] for value in losses]))
        for name in losses[0]
    }
    rates = {
        name.removesuffix("_correct") + "_accuracy": (
            counts[name] / counts[name.replace("_correct", "_total")]
            if counts[name.replace("_correct", "_total")]
            else 0.0
        )
        for name in (
            "act_correct",
            "restraint_correct",
            "shot_correct",
            "pair_correct",
        )
    }
    return {
        "loss": mean_loss,
        "counts": dict(sorted(counts.items())),
        "rates": rates,
    }


def _sequence_factory(
    checkpoint_model: SequenceReplayModel,
    *,
    checkpoint_sha256: str,
    source_identity: str,
    options: Mapping[str, Any],
    deployment: Mapping[str, Any],
    geometry: LegacyR3eGeometryCheckpoint,
) -> Callable[[], SequenceReplayPolicy]:
    def factory() -> SequenceReplayPolicy:
        model = copy.deepcopy(checkpoint_model).eval()
        base_policy = _base_policy_for_model(
            model.base_model,
            checkpoint_sha256=model.base_checkpoint_sha256 or "",
            options=options,
        )
        return SequenceReplayPolicy(
            model,
            base_policy,
            geometry,
            checkpoint_sha256=checkpoint_sha256,
            source_identity=source_identity,
            minimum_restraint_probability=float(
                deployment["minimum_restraint_probability"]
            ),
            minimum_pair_probability=float(
                deployment["minimum_pair_probability"]
            ),
            minimum_geometry_probability=float(
                deployment["minimum_geometry_probability"]
            ),
            minimum_geometry_margin=float(
                deployment["minimum_geometry_margin"]
            ),
        )

    return factory


def _selection_key(
    evaluations: Mapping[int, Mapping[str, Any]],
) -> tuple[object, ...]:
    catastrophes = 0
    gauge_regressions = 0
    p10_survival: list[float] = []
    median_survival: list[float] = []
    median_score: list[float] = []
    for horizon, packet in sorted(evaluations.items()):
        paired = packet["paired_vs_v5"]
        aggregate = packet["sequence"]["aggregate"]
        catastrophes += int(paired["catastrophic_regressions"])
        gauge_regressions += int(paired["gauge_regressions"])
        p10_survival.append(float(aggregate["survival_ticks"]["p10"]))
        median_survival.append(float(aggregate["survival_ticks"]["median"]))
        median_score.append(float(aggregate["raw_score"]["median"]))
    return (
        catastrophes,
        gauge_regressions,
        -min(p10_survival),
        -min(median_survival),
        -min(median_score),
    )


def _train_plateau_models(
    model: SequenceReplayModel,
    *,
    replay: _PolicySequence,
    development: Sequence[_PolicySequence],
    geometry_batches: Sequence[SequenceReplayBatch],
    training: Mapping[str, Any],
    output_dir: Path,
    source_identity: str,
    data_identities: Mapping[str, Any],
    overwrite: bool,
) -> tuple[dict[int, Path], dict[str, object]]:
    torch.set_num_threads(int(training["torch_threads"]))
    torch.manual_seed(int(training["seed"]))
    rng = random.Random(int(training["seed"]))
    optimizer = torch.optim.AdamW(
        (value for value in model.parameters() if value.requires_grad),
        lr=float(training["learning_rate"]),
        weight_decay=0.0,
    )
    loss_weights = SequenceReplayLossWeights(
        value=float(training["value_weight"]),
        viability=float(training["value_weight"]),
        outcome=float(training["value_weight"]),
        residual=float(training["baseline_kl_weight"]),
    )
    checkpoints: dict[int, Path] = {}
    points: list[dict[str, object]] = []
    recent: list[dict[str, float]] = []
    gradients: list[float] = []
    requested = tuple(int(value) for value in training["steps"])
    if not requested or requested[0] != 0 or sorted(set(requested)) != list(
        requested
    ):
        raise ValueError("training plateaus must be unique and start at zero")
    started_wall = time.monotonic()
    started_cpu = time.process_time()

    def capture(step: int) -> None:
        checkpoint = output_dir / f"sequence-step-{step:04d}.pt"
        checkpoint_sha = save_sequence_replay_checkpoint(
            checkpoint,
            model,
            source_identity=source_identity,
            metadata={
                "development_only": True,
                "canonical_r3_evidence": False,
                "sealed_test_material_used": False,
                "training_steps": step,
                "data_identities": dict(data_identities),
                "optimizer": "AdamW",
                "learning_rate": float(training["learning_rate"]),
            },
            overwrite=overwrite,
        )
        checkpoints[step] = checkpoint
        point = {
            "training_steps": step,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
            },
            "offline": {
                "replay": _offline_policy_metrics(
                    model,
                    (replay,),
                    seed=int(training["seed"]) + step,
                    training=training,
                    samples=4,
                ),
                "development": _offline_policy_metrics(
                    model,
                    development,
                    seed=int(training["seed"]) + 10_000 + step,
                    training=training,
                    samples=8,
                ),
                "geometry": _offline_geometry_metrics(
                    model, geometry_batches
                ),
            },
            "since_previous": {
                "mean_loss": (
                    {
                        name: float(np.mean([value[name] for value in recent]))
                        for name in recent[0]
                    }
                    if recent
                    else {}
                ),
                "gradient_norm": (
                    _distribution(gradients) if gradients else None
                ),
            },
            "cost_checkpoint": {
                "wall_seconds": time.monotonic() - started_wall,
                "cpu_seconds": time.process_time() - started_cpu,
            },
        }
        points.append(point)
        recent.clear()
        gradients.clear()

    capture(0)
    window = int(training["burn_in"]) + int(training["tbptt_steps"])
    for step in range(1, requested[-1] + 1):
        schedule = (step - 1) % 5
        if schedule == 0:
            batch = _sample_policy_batch(
                (replay,),
                rng=rng,
                batch_episodes=int(training["batch_episodes"]),
                window=window,
                burn_in=int(training["burn_in"]),
            )
        elif schedule in (1, 2):
            batch = _sample_policy_batch(
                development,
                rng=rng,
                batch_episodes=int(training["batch_episodes"]),
                window=window,
                burn_in=int(training["burn_in"]),
            )
        else:
            batch = _sample_geometry_batch(
                geometry_batches,
                rng=rng,
                window=window,
                burn_in=int(training["burn_in"]),
            )
        loss, _state, gradient = train_sequence_replay_step(
            model,
            batch,
            optimizer,
            weights=loss_weights,
            max_gradient_norm=float(training["gradient_clip"]),
        )
        recent.append(loss.scalars())
        gradients.append(gradient)
        if step in requested:
            capture(step)
    report = {
        "schema": "irisu-sequence-replay-training-curve-v1",
        "seed": int(training["seed"]),
        "torch_threads": int(training["torch_threads"]),
        "schedule": {
            "five_step_cycle": [
                "trusted replay",
                "development DAgger",
                "development DAgger",
                "legacy geometry",
                "legacy geometry",
            ],
            "replay_policy_weight": 0.5,
            "tbptt_window": window,
            "burn_in": int(training["burn_in"]),
            "batch_episodes": int(training["batch_episodes"]),
            "loss_weights": asdict(loss_weights),
            "baseline_kl_note": (
                "configured baseline_kl_weight is implemented as an L2 "
                "penalty on bounded residual logits, not a literal KL"
            ),
        },
        "points": points,
        "cost": {
            "wall_seconds": time.monotonic() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
        },
    }
    return checkpoints, {**report, "sha256": _canonical_sha256(report)}


def _evaluate(
    r3e: ModuleType,
    *,
    label: str,
    library: Path,
    seeds: Sequence[int],
    horizon: int,
    factory: Callable[[], object],
) -> dict[str, object]:
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    result = r3e.evaluate_policy(
        label=label,
        library_path=library,
        seeds=seeds,
        horizon_ticks=horizon,
        factory=factory,
    )
    return {
        **result,
        "cost": {
            "wall_seconds": time.monotonic() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
        },
    }


def _rubric_summary(result: Mapping[str, Any]) -> dict[str, object]:
    episodes = result["episodes"]
    aggregate = result["aggregate"]
    return {
        "score": aggregate["raw_score"],
        "survival_ticks": aggregate["survival_ticks"],
        "gauge_failures": int(aggregate["gauge_failures"]),
        "final_gauge": _distribution(
            [int(value["final_gauge"]) for value in episodes]
        ),
        "final_level": _distribution(
            [int(value["final_level"]) for value in episodes]
        ),
        "qualifying_clears": int(aggregate["qualifying_clears"]),
        "rotten_events": int(aggregate["conversion"]["rotten"]),
        "decisions": sum(int(value["decisions"]) for value in episodes),
        "primitive_actions": sum(
            int(value["primitive_actions"]) for value in episodes
        ),
        "policy_counts": dict(result.get("policy_counts", {})),
    }


def _load_sequence_model(
    path: Path,
    *,
    checkpoint_sha256: str,
    v5_path: Path,
    v5_sha256: str,
    source_identity: str,
) -> SequenceReplayModel:
    base = load_steering_checkpoint(
        v5_path,
        expected_sha256=v5_sha256,
    )
    loaded = load_sequence_replay_checkpoint(
        path,
        base.model,
        expected_sha256=checkpoint_sha256,
        expected_base_checkpoint_sha256=v5_sha256,
        expected_source_identity=source_identity,
    )
    return loaded.model


def _paired_packet(
    *,
    horizon: int,
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "candidate_summary": _rubric_summary(candidate),
        "paired_vs_v5": paired_comparison(
            base,
            candidate,
            horizon=horizon,
        ),
    }


def _load_comparator_artifact(
    path: Path,
    *,
    seeds: Sequence[int],
    horizons: Sequence[int],
    inputs: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, object]] | None:
    if not path.exists():
        return None
    resolved = _safe_path(path, "sequence-replay comparator artifact")
    value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("comparator artifact is not a mapping")
    payload_sha = value.get("payload_sha256")
    unsigned = {name: item for name, item in value.items() if name != "payload_sha256"}
    declared = value.get("identities", {}).get("declared_inputs", {})
    suite = value.get("protocol", {}).get("seed_suite", {})
    evaluations = value.get("evaluations", {})
    if (
        value.get("format")
        != "irisu-sequence-replay-comparator-evaluation-v1"
        or value.get("development_only") is not True
        or value.get("canonical_r3_evidence") is not False
        or value.get("sealed_test_material_used") is not False
        or value.get("test_phase_run") is not False
        or payload_sha != _canonical_sha256(unsigned)
        or list(suite.get("seeds", ())) != [int(seed) for seed in seeds]
        or any(str(horizon) not in evaluations for horizon in horizons)
        or any(declared.get(name) != expected for name, expected in inputs.items())
    ):
        raise ValueError("comparator artifact does not match this campaign")
    identity = {
        "path": str(resolved),
        "file_sha256": _file_sha256(resolved),
        "payload_sha256": payload_sha,
        "suite_sha256": suite.get("sha256"),
        "successful_cost": value.get("cost", {}).get("successful_attempt"),
        "source_changed_during_evaluation": value.get("identities", {}).get(
            "source_changed_during_evaluation"
        ),
    }
    return value, identity


def _atomic_json(path: Path, value: Mapping[str, Any], *, overwrite: bool) -> str:
    target = _safe_path(path, "sequence-replay report", output=True)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to replace {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return _file_sha256(target)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--replay", type=Path, default=DEFAULT_REPLAY)
    parser.add_argument("--exact-worker", type=Path, default=DEFAULT_EXACT_WORKER)
    parser.add_argument("--library", type=Path, default=DEFAULT_PORTABLE)
    parser.add_argument("--v5", type=Path, default=DEFAULT_V5)
    parser.add_argument("--fast-geometry", type=Path, default=DEFAULT_FAST_GEOMETRY)
    parser.add_argument(
        "--extended-geometry", type=Path, default=DEFAULT_EXTENDED_GEOMETRY
    )
    parser.add_argument(
        "--geometry-collection", type=Path, default=DEFAULT_GEOMETRY_COLLECTION
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--mode",
        choices=("campaign", "train", "evaluate"),
        default="campaign",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--evaluation-seeds", type=int)
    parser.add_argument("--horizons", type=int, nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    campaign_wall = time.monotonic()
    campaign_cpu = time.process_time()
    args = _parser().parse_args(argv)
    config_path = _safe_path(args.config, "sequence-replay config")
    config = _load_config(config_path)
    identities = InputIdentity.from_config(config)
    replay = _verified_file(args.replay, identities.replay, "trusted replay")
    exact_worker = _verified_file(
        args.exact_worker, identities.exact_worker, "exact worker"
    )
    library = _verified_file(
        args.library, identities.portable_runtime, "portable runtime"
    )
    v5 = _verified_file(args.v5, identities.frozen_v5, "frozen v5")
    fast_geometry = _verified_file(
        args.fast_geometry, identities.fast_geometry, "fast geometry"
    )
    extended_geometry = _verified_file(
        args.extended_geometry,
        identities.extended_geometry,
        "extended geometry",
    )
    geometry_collection = _verified_file(
        args.geometry_collection,
        identities.geometry_collection,
        "geometry collection",
    )
    output_dir = _safe_path(args.output_dir, "sequence-replay output", output=True)
    # Training and recurrent factories are attached below once their input
    # identities have all been verified.  Keeping this early packet makes a
    # failed campaign auditable without treating it as learning evidence.
    inputs = {
        **asdict(identities),
        "paths": {
            "replay": str(replay),
            "exact_worker": str(exact_worker),
            "portable_runtime": str(library),
            "frozen_v5": str(v5),
            "fast_geometry": str(fast_geometry),
            "extended_geometry": str(extended_geometry),
            "geometry_collection": str(geometry_collection),
        },
        "config": {
            "path": str(config_path),
            "sha256": _file_sha256(config_path),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    source = _source_identity(config_path)
    r3e = _load_script("sequence_r3e", R3E_PATH)
    replay_benchmark = _load_script(
        "sequence_replay_supervision", REPLAY_BENCHMARK_PATH
    )
    training = config["training"]
    torch.set_num_threads(int(training["torch_threads"]))
    selection_config = config["selection"]
    evaluation_config = config["evaluation"]
    training_seeds = _derive_seeds(
        str(training["development_seed_label"]),
        int(training["development_seeds"]),
    )
    selection_seeds = _derive_seeds(
        str(selection_config["seed_label"]),
        int(selection_config["seeds"]),
    )
    evaluation_count = (
        int(evaluation_config["seed_count"])
        if args.evaluation_seeds is None
        else int(args.evaluation_seeds)
    )
    if not 1 <= evaluation_count <= 32:
        raise ValueError("development evaluation seed count must lie in [1,32]")
    evaluation_seeds = _derive_seeds(
        str(evaluation_config["suite_label"]), evaluation_count
    )
    horizons = tuple(
        int(value)
        for value in (
            evaluation_config["horizons"]
            if args.horizons is None
            else args.horizons
        )
    )
    if not horizons or any(value < 1 for value in horizons):
        raise ValueError("evaluation horizons must be positive")
    replay_seed = 1_681_750_29
    geometry_training_seeds = {
        3_643_205_411,
        1_211_936_443,
        4_184_595_850,
        2_573_156_672,
    }
    if (
        set(training_seeds) & set(selection_seeds)
        or set(training_seeds) & set(evaluation_seeds)
        or set(selection_seeds) & set(evaluation_seeds)
        or replay_seed in set(selection_seeds) | set(evaluation_seeds)
        or geometry_training_seeds
        & (set(selection_seeds) | set(evaluation_seeds))
    ):
        raise RuntimeError("sequence replay development seed suites overlap")
    suites = {
        "training": {
            "label": str(training["development_seed_label"]),
            "seeds": list(training_seeds),
        },
        "selection": {
            "label": str(selection_config["seed_label"]),
            "seeds": list(selection_seeds),
        },
        "evaluation": {
            "label": str(evaluation_config["suite_label"]),
            "seeds": list(evaluation_seeds),
            "horizons": list(horizons),
        },
    }
    suites["sha256"] = _canonical_sha256(suites)

    base_checkpoint = load_steering_checkpoint(
        v5,
        expected_sha256=identities.frozen_v5,
    )
    base_checkpoint.model.eval()
    fast_checkpoint = load_legacy_r3e_geometry_checkpoint(
        fast_geometry,
        expected_sha256=identities.fast_geometry,
        expected_base_policy_sha256=identities.frozen_v5,
        expected_runtime_sha256=identities.portable_runtime,
    )
    extended_checkpoint = load_legacy_r3e_geometry_checkpoint(
        extended_geometry,
        expected_sha256=identities.extended_geometry,
        expected_base_policy_sha256=identities.frozen_v5,
        expected_runtime_sha256=identities.portable_runtime,
    )

    if args.mode == "evaluate":
        if args.checkpoint is None or args.checkpoint_sha256 is None:
            raise ValueError("evaluate mode requires checkpoint and SHA-256")
        checkpoint_path = _safe_path(
            args.checkpoint, "sequence-replay checkpoint"
        )
        checkpoint_sha = _sha256(
            args.checkpoint_sha256, "sequence-replay checkpoint"
        )
        model = _load_sequence_model(
            checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            v5_path=v5,
            v5_sha256=identities.frozen_v5,
            source_identity=str(source["sha256"]),
        )
        sequence_factory = _sequence_factory(
            model,
            checkpoint_sha256=checkpoint_sha,
            source_identity=str(source["sha256"]),
            options=config["frozen_v5"],
            deployment=config["deployment"],
            geometry=extended_checkpoint,
        )
        evaluations = {
            str(horizon): _evaluate(
                r3e,
                label=f"sequence-replay-evaluate-{horizon}",
                library=library,
                seeds=evaluation_seeds,
                horizon=horizon,
                factory=sequence_factory,
            )
            for horizon in horizons
        }
        report = {
            "schema": "irisu-sequence-replay-evaluation-v1",
            "development_only": True,
            "canonical_r3_evidence": False,
            "sealed_test_material_used": False,
            "test_phase_run": False,
            "inputs": inputs,
            "source": source,
            "suites": suites,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha,
            },
            "evaluations": evaluations,
            "summaries": {
                horizon: _rubric_summary(value)
                for horizon, value in evaluations.items()
            },
            "cost": {
                "wall_seconds": time.monotonic() - campaign_wall,
                "cpu_seconds": time.process_time() - campaign_cpu,
            },
        }
        digest = _atomic_json(
            output_dir / "evaluation-report.json",
            report,
            overwrite=args.overwrite,
        )
        print(json.dumps({"report_sha256": digest, "report": report["checkpoint"]}))
        return 0

    exact_examples, exact_collection, replay_evidence = _collect_exact_replay(
        replay_benchmark,
        replay_path=replay,
        worker_path=exact_worker,
    )
    replay_sequence = _replay_sequence(exact_examples, exact_collection)

    def shared_base_factory() -> GoalConditionedSteeringPolicy:
        return _base_policy_for_model(
            base_checkpoint.model,
            checkpoint_sha256=identities.frozen_v5,
            options=config["frozen_v5"],
        )

    behavior_factory = LegacyR3eGeometryPolicyFactory(
        extended_checkpoint,
        shared_base_factory,
    )
    dagger_sequences, dagger_evidence = _collect_dagger_sequences(
        library=library,
        seeds=training_seeds,
        ticks=int(training["development_ticks"]),
        source_identity=str(source["sha256"]),
        behavior_factory=behavior_factory,
        continuation_factory=shared_base_factory,
        training=training,
    )
    geometry_batches, geometry_evidence = _load_geometry_batches(
        geometry_collection,
        checkpoint=extended_checkpoint,
        expected_collection_sha256=identities.geometry_collection,
        expected_vocabulary_sha256=str(
            config["inputs"]["legacy_geometry_vocabulary_sha256"]
        ),
    )
    model = SequenceReplayModel(
        base_checkpoint.model,
        config=SequenceReplayConfig(**config["model"]),
        base_checkpoint_sha256=identities.frozen_v5,
        geometry_candidate_count=extended_checkpoint.model.candidate_count,
        geometry_candidate_set_sha256=(
            extended_checkpoint.model.candidate_set_sha256
        ),
    )
    data_evidence = {
        "schema": "irisu-sequence-replay-training-data-v1",
        "replay": replay_evidence,
        "replay_event_sha256": hashlib.sha256(
            replay_sequence.events.numpy().tobytes()
        ).hexdigest(),
        "dagger": dagger_evidence,
        "geometry": geometry_evidence,
        "training_example_counts": {
            "trusted_replay": len(replay_sequence.examples),
            "development_dagger": sum(
                len(value.examples) for value in dagger_sequences
            ),
            "legacy_geometry": sum(
                int(value.targets.valid_mask.sum())
                for value in geometry_batches
            ),
        },
        "seed_suites": suites,
    }
    data_evidence["sha256"] = _canonical_sha256(data_evidence)
    data_sha = _atomic_json(
        output_dir / "training-data-evidence.json",
        data_evidence,
        overwrite=args.overwrite,
    )
    checkpoints, training_curve = _train_plateau_models(
        model,
        replay=replay_sequence,
        development=dagger_sequences,
        geometry_batches=geometry_batches,
        training=training,
        output_dir=output_dir,
        source_identity=str(source["sha256"]),
        data_identities={
            "training_data_payload_sha256": data_evidence["sha256"],
            "training_data_file_sha256": data_sha,
            "trusted_replay_collection_sha256": replay_evidence[
                "collection_sha256"
            ],
            "dagger_sha256": dagger_evidence["sha256"],
            "legacy_geometry_sha256": geometry_evidence["sha256"],
        },
        overwrite=args.overwrite,
    )
    curve_sha = _atomic_json(
        output_dir / "training-curve.json",
        training_curve,
        overwrite=args.overwrite,
    )
    if args.mode == "train":
        report = {
            "schema": "irisu-sequence-replay-training-v1",
            "development_only": True,
            "canonical_r3_evidence": False,
            "sealed_test_material_used": False,
            "test_phase_run": False,
            "inputs": inputs,
            "source": source,
            "suites": suites,
            "training_data": {
                "path": str(output_dir / "training-data-evidence.json"),
                "sha256": data_sha,
            },
            "training_curve": {
                "path": str(output_dir / "training-curve.json"),
                "sha256": curve_sha,
            },
            "checkpoints": {
                str(step): {
                    "path": str(path),
                    "sha256": _file_sha256(path),
                }
                for step, path in checkpoints.items()
            },
            "cost": {
                "wall_seconds": time.monotonic() - campaign_wall,
                "cpu_seconds": time.process_time() - campaign_cpu,
            },
        }
        digest = _atomic_json(
            output_dir / "training-report.json",
            report,
            overwrite=args.overwrite,
        )
        print(json.dumps({"report_sha256": digest}))
        return 0

    selection_started_wall = time.monotonic()
    selection_started_cpu = time.process_time()
    selection_horizons = tuple(
        int(value) for value in selection_config["horizons"]
    )
    base_selection = {
        horizon: _evaluate(
            r3e,
            label=f"frozen-v5-selection-{horizon}",
            library=library,
            seeds=selection_seeds,
            horizon=horizon,
            factory=shared_base_factory,
        )
        for horizon in selection_horizons
    }
    selection_candidates: dict[int, dict[int, dict[str, Any]]] = {}
    for step, checkpoint_path in sorted(checkpoints.items()):
        checkpoint_sha = _file_sha256(checkpoint_path)
        candidate_model = _load_sequence_model(
            checkpoint_path,
            checkpoint_sha256=checkpoint_sha,
            v5_path=v5,
            v5_sha256=identities.frozen_v5,
            source_identity=str(source["sha256"]),
        )
        candidate_factory = _sequence_factory(
            candidate_model,
            checkpoint_sha256=checkpoint_sha,
            source_identity=str(source["sha256"]),
            options=config["frozen_v5"],
            deployment=config["deployment"],
            geometry=extended_checkpoint,
        )
        candidate_points: dict[int, dict[str, Any]] = {}
        for horizon in selection_horizons:
            evaluation = _evaluate(
                r3e,
                label=f"sequence-step-{step}-selection-{horizon}",
                library=library,
                seeds=selection_seeds,
                horizon=horizon,
                factory=candidate_factory,
            )
            candidate_points[horizon] = {
                "v5": base_selection[horizon],
                "sequence": evaluation,
                "paired_vs_v5": paired_comparison(
                    base_selection[horizon],
                    evaluation,
                    horizon=horizon,
                ),
                "summary": _rubric_summary(evaluation),
            }
        selection_candidates[step] = candidate_points
    selected_step = min(
        selection_candidates,
        key=lambda step: (
            _selection_key(selection_candidates[step]),
            step,
        ),
    )
    selection_packet = {
        "schema": "irisu-sequence-replay-selection-v1",
        "ranking": list(selection_config["ranking"]),
        "seeds": list(selection_seeds),
        "horizons": list(selection_horizons),
        "selected_training_steps": selected_step,
        "selected_key": list(
            _selection_key(selection_candidates[selected_step])
        ),
        "candidates": {
            str(step): {
                str(horizon): value
                for horizon, value in points.items()
            }
            for step, points in selection_candidates.items()
        },
        "cost": {
            "wall_seconds": time.monotonic() - selection_started_wall,
            "cpu_seconds": time.process_time() - selection_started_cpu,
        },
    }
    selection_packet["sha256"] = _canonical_sha256(selection_packet)
    selection_sha = _atomic_json(
        output_dir / "selection.json",
        selection_packet,
        overwrite=args.overwrite,
    )

    selected_path = checkpoints[selected_step]
    selected_sha = _file_sha256(selected_path)
    selected_model = _load_sequence_model(
        selected_path,
        checkpoint_sha256=selected_sha,
        v5_path=v5,
        v5_sha256=identities.frozen_v5,
        source_identity=str(source["sha256"]),
    )
    selected_factory = _sequence_factory(
        selected_model,
        checkpoint_sha256=selected_sha,
        source_identity=str(source["sha256"]),
        options=config["frozen_v5"],
        deployment=config["deployment"],
        geometry=extended_checkpoint,
    )
    comparator_path = output_dir / "comparator-evaluation.json"
    comparator = _load_comparator_artifact(
        comparator_path,
        seeds=evaluation_seeds,
        horizons=horizons,
        inputs={
            "trusted_replay_sha256": identities.replay,
            "exact_worker_sha256": identities.exact_worker,
            "portable_runtime_sha256": identities.portable_runtime,
            "frozen_v5_sha256": identities.frozen_v5,
            "fast_geometry_sha256": identities.fast_geometry,
            "extended_geometry_sha256": identities.extended_geometry,
            "geometry_collection_sha256": identities.geometry_collection,
            "legacy_geometry_vocabulary_sha256": str(
                config["inputs"]["legacy_geometry_vocabulary_sha256"]
            ),
        },
    )
    comparator_packet: dict[str, Any] | None
    comparator_identity: dict[str, object] | None
    if comparator is None:
        comparator_packet = None
        comparator_identity = None
    else:
        comparator_packet, comparator_identity = comparator
    final_started_wall = time.monotonic()
    final_started_cpu = time.process_time()
    final_evaluations: dict[str, Any] = {}
    fast_factory = LegacyR3eGeometryPolicyFactory(
        fast_checkpoint, shared_base_factory
    )
    extended_factory = LegacyR3eGeometryPolicyFactory(
        extended_checkpoint, shared_base_factory
    )
    for horizon in horizons:
        if comparator_packet is None:
            comparators = {
                "frozen_v5": _evaluate(
                    r3e,
                    label=f"frozen-v5-final-{horizon}",
                    library=library,
                    seeds=evaluation_seeds,
                    horizon=horizon,
                    factory=shared_base_factory,
                ),
                "legacy_fast_r3e": _evaluate(
                    r3e,
                    label=f"legacy-fast-final-{horizon}",
                    library=library,
                    seeds=evaluation_seeds,
                    horizon=horizon,
                    factory=fast_factory,
                ),
                "legacy_extended_r3e": _evaluate(
                    r3e,
                    label=f"legacy-extended-final-{horizon}",
                    library=library,
                    seeds=evaluation_seeds,
                    horizon=horizon,
                    factory=extended_factory,
                ),
            }
        else:
            comparators = comparator_packet["evaluations"][str(horizon)]
        sequence_evaluation = _evaluate(
            r3e,
            label=f"sequence-replay-final-{horizon}",
            library=library,
            seeds=evaluation_seeds,
            horizon=horizon,
            factory=selected_factory,
        )
        base = comparators["frozen_v5"]
        final_evaluations[str(horizon)] = {
            "policies": {**comparators, "sequence_replay": sequence_evaluation},
            "summaries": {
                name: _rubric_summary(value)
                for name, value in {
                    **comparators,
                    "sequence_replay": sequence_evaluation,
                }.items()
            },
            "paired_vs_v5": {
                name: paired_comparison(
                    base,
                    value,
                    horizon=horizon,
                )
                for name, value in {
                    "legacy_fast_r3e": comparators["legacy_fast_r3e"],
                    "legacy_extended_r3e": comparators[
                        "legacy_extended_r3e"
                    ],
                    "sequence_replay": sequence_evaluation,
                }.items()
            },
        }
    optional_50k: dict[str, Any] | None = None
    if args.horizons is None:
        optional_horizon = int(evaluation_config["optional_horizon"])
        optional_count = min(
            evaluation_count,
            int(evaluation_config["optional_50k_seed_count"]),
        )
        optional_seeds = evaluation_seeds[:optional_count]
        optional_base = _evaluate(
            r3e,
            label="frozen-v5-final-50000",
            library=library,
            seeds=optional_seeds,
            horizon=optional_horizon,
            factory=shared_base_factory,
        )
        optional_sequence = _evaluate(
            r3e,
            label="sequence-replay-final-50000",
            library=library,
            seeds=optional_seeds,
            horizon=optional_horizon,
            factory=selected_factory,
        )
        optional_50k = {
            "horizon": optional_horizon,
            "seeds": list(optional_seeds),
            "policies": {
                "frozen_v5": optional_base,
                "sequence_replay": optional_sequence,
            },
            "summaries": {
                "frozen_v5": _rubric_summary(optional_base),
                "sequence_replay": _rubric_summary(optional_sequence),
            },
            "paired_vs_v5": paired_comparison(
                optional_base,
                optional_sequence,
                horizon=optional_horizon,
            ),
            "omission": (
                "50k legacy geometry comparators omitted because their exact "
                "2k/10k/20k paired regressions are in the comparator artifact"
            ),
        }
    campaign_report = {
        "schema": "irisu-sequence-replay-campaign-v1",
        "status": "development_only_not_canonical_evidence",
        "development_only": True,
        "deployable": False,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
        "test_phase_run": False,
        "objective": str(config["objective"]),
        "inputs": inputs,
        "source": source,
        "environment": {
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
        },
        "seed_suites": suites,
        "diagnosis": {
            "frozen_v5": (
                "feed-forward directed-pair logits with only hand-coded "
                "cooldown/progress state; its auxiliary intent prediction is "
                "not an option memory or deployed action controller"
            ),
            "legacy_r3e": (
                "freezes v5 act/pair/cadence and selects shot geometry from "
                "one snapshot; it cannot learn delayed restraint, option "
                "duration, join/clear credit, or miss/rot recovery"
            ),
            "sequence_replay": (
                "ID-ablated recurrent residual over invariant body pooling "
                "and previous-completed-transition events; conservative "
                "act/pair/geometry gates fall back to exact frozen v5"
            ),
        },
        "model": selected_model.manifest(),
        "selected_checkpoint": {
            "training_steps": selected_step,
            "path": str(selected_path),
            "sha256": selected_sha,
        },
        "training_data": {
            "path": str(output_dir / "training-data-evidence.json"),
            "file_sha256": data_sha,
            "payload_sha256": data_evidence["sha256"],
        },
        "training_curve": {
            "path": str(output_dir / "training-curve.json"),
            "file_sha256": curve_sha,
            "payload_sha256": training_curve["sha256"],
            "points": [
                {
                    "training_steps": point["training_steps"],
                    "checkpoint": point["checkpoint"],
                    "offline": point["offline"],
                    "cost_checkpoint": point["cost_checkpoint"],
                }
                for point in training_curve["points"]
            ],
        },
        "selection": {
            "path": str(output_dir / "selection.json"),
            "file_sha256": selection_sha,
            "payload_sha256": selection_packet["sha256"],
            "selected_training_steps": selected_step,
            "selected_key": selection_packet["selected_key"],
        },
        "comparator_artifact": comparator_identity,
        "final_evaluations": final_evaluations,
        "optional_50k": optional_50k,
        "limitations": [
            (
                "the replay destination is nearest-visible-same-color "
                "behavioral inference, not recovered human intent"
            ),
            (
                "legacy geometry collection omitted per-state slot "
                "availability; auxiliary training uses all 32 fixed slots"
            ),
            (
                "branch values use a 128-tick frozen-v5 continuation and are "
                "causal local option labels, not full-episode counterfactuals"
            ),
            (
                "recurrent pair residuals remain quadratic in active bodies; "
                "training therefore uses 8-step burn-in plus 32-step TBPTT"
            ),
            (
                "all rollout evidence is disjoint development evidence; no "
                "sealed, test, or canonical evaluation was run"
            ),
        ],
        "cost": {
            "prior_aborted_integration_attempt": {
                "wall_seconds_observed": 250,
                "cpu_seconds_observed": 953,
                "artifacts_written": False,
                "reason": (
                    "Torch thread cap was initially applied after DAgger; "
                    "attempt stopped and code corrected before training"
                ),
            },
            "data_collection": dagger_evidence["cost"],
            "training": training_curve["cost"],
            "selection": selection_packet["cost"],
            "final_sequence_evaluation": {
                "wall_seconds": time.monotonic() - final_started_wall,
                "cpu_seconds": time.process_time() - final_started_cpu,
            },
            "campaign_wall_seconds": time.monotonic() - campaign_wall,
            "campaign_cpu_seconds": time.process_time() - campaign_cpu,
            "comparator_external": (
                None
                if comparator_identity is None
                else comparator_identity["successful_cost"]
            ),
        },
    }
    report_sha = _atomic_json(
        output_dir / "campaign-report.json",
        campaign_report,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "report": str(output_dir / "campaign-report.json"),
                "report_sha256": report_sha,
                "selected_checkpoint": str(selected_path),
                "selected_checkpoint_sha256": selected_sha,
                "selected_training_steps": selected_step,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
