#!/usr/bin/env python3
"""Development-only R3G Strategy C event-world-model campaign.

The command materializes the locked shared splits, verifies two frozen-v5
sentinels, screens an exact two-renewal teacher, fits a whole-seed conformal
event model from every exact candidate branch, and (only after freezing) runs
the heldout/stress barrier and teacher-free student resource screen.  It never
opens sealed, canonical, test, or parent-only winner material.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r3g_artifact_report import finalize_artifacts

from irisu_env import EventKind, IrisuEnv
from irisu_pointer.event_mpc import (
    EVENT_MPC_VERSION,
    FEATURE_NAMES,
    TARGET_NAMES,
    CandidateCertificate,
    EventMPCConfig,
    ExactEventPlanner,
    ExactSearchResult,
    KNNEventWorldModel,
    ModelBarrierPolicy,
    ModelExample,
    candidate_features,
    search_examples,
)
from irisu_pointer.steering import SteeringDecision
from irisu_pointer.steering_checkpoint import load_steering_checkpoint
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3g-solvency-barrier-tournament-20260729/shared-protocol.md"
)
PROTOCOL_SHA256 = (
    "6dfb2ffa3a76cc00447e3dcf889f6209a17ff5e2f4c3382fe0959bbabbd52991"
)
RUNTIME = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/runtime/"
    "main-0c48dba-20260723/portable-build/libirisu_clone.so"
)
RUNTIME_SHA256 = (
    "4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79"
)
CHECKPOINT = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3d-survival-v5-20260729/long-development.pt"
)
CHECKPOINT_SHA256 = (
    "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d"
)
REVISION = "de701b36355d5ec582df30f4223aabde7bc537df"
JOINT_SOURCE = Path(
    "/home/gabe/.codex/worktrees/e2f8/irisu/python/"
    "irisu_pointer/joint_planner.py"
)
JOINT_SOURCE_SHA256 = (
    "dc7009fc18a322eca5dace55b9baf982b6ced26c18517af752aab0f6365d362e"
)
JOINT_BENCHMARK = Path(
    "/home/gabe/.codex/worktrees/e2f8/irisu/benchmarks/rl_joint_planner.py"
)
JOINT_BENCHMARK_SHA256 = (
    "aa1c0ccd34ffc37ee794ad340e00c69b89f580f0c9bb77be5b7d3c082becfc5f"
)
JOINT_CHECKS = Path(
    "/home/gabe/.codex/worktrees/e2f8/irisu/benchmarks/"
    "joint-planner-dev-checks.py"
)
JOINT_CHECKS_SHA256 = (
    "c4cbb4feda98a30a2d7eb655b52c019b4479dcb1a5977e7ac7a47b92bbbf8be7"
)
DEVELOPMENT_CHECKS = ROOT / "benchmarks/rl_r3g_event_mpc_checks.py"
REPORTING_HELPER = ROOT / "benchmarks/r3g_artifact_report.py"
DEFAULT_CONFIG = (
    ROOT / "configs/rl/experiments/r3g-event-world-model-mpc-v1.toml"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/r3/development/r3g-event-world-model-mpc-20260729"
)
SEED_PREFIX = "r3g-solvency-barrier-tournament-20260729"
LOGICAL_CPU = 9


def _progress(stage: str, **values: object) -> None:
    print(
        json.dumps(
            {"stage": stage, **values},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_source(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load trusted source {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _trusted_modules() -> tuple[Any, Any]:
    if _file_sha256(JOINT_SOURCE) != JOINT_SOURCE_SHA256:
        raise RuntimeError("trusted corrected joint-v2 source identity changed")
    if _file_sha256(JOINT_BENCHMARK) != JOINT_BENCHMARK_SHA256:
        raise RuntimeError("trusted corrected joint-v2 benchmark identity changed")
    if _file_sha256(JOINT_CHECKS) != JOINT_CHECKS_SHA256:
        raise RuntimeError("trusted corrected joint-v2 checks identity changed")
    joint = _load_source("irisu_pointer.joint_planner", JOINT_SOURCE)
    benchmark = _load_source("_r3g_trusted_joint_benchmark", JOINT_BENCHMARK)
    return joint, benchmark


JOINT, TRUSTED_BENCHMARK = _trusted_modules()


def _derive_seed(split: str, index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{SEED_PREFIX}|{split}|{index}".encode()).digest()[:4],
        "big",
    )


def _safe_output(path: Path) -> Path:
    resolved = path.resolve()
    root = (
        ROOT
        / "artifacts/r3/development/r3g-event-world-model-mpc-20260729"
    ).resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("R3G output escaped the Strategy C namespace")
    lowered = {part.lower() for part in resolved.parts}
    if lowered & {"sealed", "canonical", "test"}:
        raise ValueError("R3G development output entered a forbidden namespace")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _atomic_json(path: Path, value: Mapping[str, object]) -> str:
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
    return _file_sha256(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            for row in rows:
                stream.write(
                    json.dumps(
                        row, sort_keys=True, separators=(",", ":"), allow_nan=False
                    )
                    + "\n"
                )
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _file_sha256(path)


def _preformal_manifest(output: Path) -> dict[str, object]:
    allowed = output / "preformal"
    unexpected = [
        value
        for value in output.iterdir()
        if value != allowed
    ]
    if unexpected:
        raise FileExistsError(
            "refusing to rewrite a prior Strategy C artifact namespace"
        )
    if not allowed.exists():
        return {"files": [], "sha256": _canonical_sha256([])}
    if not allowed.is_dir() or allowed.is_symlink():
        raise RuntimeError("preformal evidence path is not a real directory")
    files = []
    for path in sorted(allowed.rglob("*")):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise RuntimeError("preformal evidence contains an unsafe leaf")
        if path.is_file():
            files.append(
                {
                    "path": path.relative_to(output).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    return {"files": files, "sha256": _canonical_sha256(files)}


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    experiment = value.get("experiment", {})
    if (
        experiment.get("development_only") is not True
        or experiment.get("sealed_test_allowed") is not False
        or experiment.get("canonical_run_allowed") is not False
        or experiment.get("protocol_sha256") != PROTOCOL_SHA256
        or experiment.get("repository_revision") != REVISION
        or experiment.get("runtime_sha256") != RUNTIME_SHA256
        or experiment.get("checkpoint_sha256") != CHECKPOINT_SHA256
        or experiment.get("trusted_joint_source_sha256")
        != JOINT_SOURCE_SHA256
        or int(experiment.get("logical_cpu", -1)) != LOGICAL_CPU
    ):
        raise ValueError("Strategy C config identity or boundary changed")
    expected_splits = {
        "teacher_screen_count": 8,
        "teacher_screen_horizon": 2_000,
        "student_screen_count": 8,
        "student_screen_horizon": 10_000,
        "dagger_train_count": 16,
        "barrier_calibration_count": 64,
        "barrier_heldout_count": 64,
        "barrier_stress_count": 64,
    }
    splits = value.get("splits", {})
    if not isinstance(splits, Mapping) or any(
        type(splits.get(name)) is not int or splits[name] != expected
        for name, expected in expected_splits.items()
    ):
        raise ValueError("Strategy C common seed schedule changed")
    budgets = value.get("budgets", {})
    if (
        not isinstance(budgets, Mapping)
        or type(budgets.get("dagger_horizon_ticks")) is not int
        or not 1 <= budgets["dagger_horizon_ticks"] <= 10_000
        or any(
            type(budgets.get(name)) is not int or budgets[name] < 1
            for name in (
                "calibration_decisions_per_seed",
                "dagger_decisions_per_seed",
                "heldout_independent_queries_per_seed",
                "heldout_execution_queries_per_seed",
                "heldout_exact_queries_per_seed",
                "stress_exact_queries_per_seed",
                "student_execution_queries_per_seed",
            )
        )
        or budgets["heldout_exact_queries_per_seed"]
        != budgets["heldout_independent_queries_per_seed"]
        + budgets["heldout_execution_queries_per_seed"]
    ):
        raise ValueError("Strategy C exact-query schedule changed")
    policy = value.get("policy", {})
    if (
        not isinstance(policy, Mapping)
        or policy.get("maximum_overrides_per_episode") != 1
    ):
        raise ValueError("Strategy C compositional override policy changed")
    barrier = value.get("barrier", {})
    required_barrier = {
        "minimum_heldout_unsafe_outcomes": 32,
        "minimum_heldout_unsafe_states": 8,
        "minimum_heldout_unsafe_seeds": 4,
        "minimum_heldout_severe_outcomes": 8,
        "minimum_valid_episodes": 59,
        "coverage_bootstrap_replicates": 10_000,
    }
    if not isinstance(barrier, Mapping) or any(
        barrier.get(name) != expected
        for name, expected in required_barrier.items()
    ):
        raise ValueError("Strategy C nonvacuity barrier changed")
    maximum_false_safe = barrier.get("maximum_false_safe_cp_upper")
    minimum_coverage = barrier.get("minimum_coverage_cluster_lower")
    if (
        isinstance(maximum_false_safe, bool)
        or not isinstance(maximum_false_safe, (int, float))
        or not math.isfinite(float(maximum_false_safe))
        or not 0.0 < float(maximum_false_safe) <= 0.05
        or isinstance(minimum_coverage, bool)
        or not isinstance(minimum_coverage, (int, float))
        or not math.isfinite(float(minimum_coverage))
        or not 0.05 <= float(minimum_coverage) < 1.0
    ):
        raise ValueError("Strategy C locked barrier thresholds changed")
    return value


def _source_identity(config_path: Path) -> dict[str, object]:
    files = (
        Path(__file__).resolve(),
        DEVELOPMENT_CHECKS,
        REPORTING_HELPER,
        ROOT / "python/irisu_pointer/event_mpc.py",
        config_path.resolve(),
        ROOT / "python/irisu_pointer/steering.py",
        ROOT / "python/irisu_pointer/steering_learning.py",
        ROOT / "python/irisu_pointer/steering_checkpoint.py",
        ROOT / "python/irisu_pointer/steering_progress.py",
        ROOT / "python/irisu_env/env.py",
        ROOT / "python/irisu_env/native.py",
        ROOT / "python/irisu_rl/actions.py",
        ROOT / "python/irisu_rl/encoding.py",
        ROOT / "pyproject.toml",
        JOINT_SOURCE,
        JOINT_BENCHMARK,
        JOINT_CHECKS,
        PROTOCOL,
    )
    manifest = {
        "schema": "irisu-r3g-event-mpc-source-v1",
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        ).stdout.strip(),
        "files": {str(path): _file_sha256(path) for path in files},
    }
    if manifest["git_revision"] != REVISION:
        raise RuntimeError("Strategy C source revision changed")
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _require_identity(
    source: Mapping[str, object], config_path: Path
) -> None:
    if _file_sha256(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("locked R3G protocol changed")
    if _file_sha256(RUNTIME) != RUNTIME_SHA256:
        raise RuntimeError("trusted runtime changed")
    if _file_sha256(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("frozen-v5 checkpoint changed")
    if _source_identity(config_path) != dict(source):
        raise RuntimeError("Strategy C source changed after freeze")


def _require_artifact_hashes(
    output: Path, expected: Mapping[str, str]
) -> None:
    for relative, sha256 in expected.items():
        path = output / relative
        if not path.is_file() or _file_sha256(path) != sha256:
            raise RuntimeError(f"frozen artifact changed: {relative}")


def _seed_manifest(config: Mapping[str, Any]) -> dict[str, object]:
    budgets = config["budgets"]
    split_config = config["splits"]
    specs = (
        (
            "teacher-screen",
            int(split_config["teacher_screen_count"]),
            int(split_config["teacher_screen_horizon"]),
            "exact-teacher allocation screen only",
        ),
        (
            "student-screen",
            int(split_config["student_screen_count"]),
            int(split_config["student_screen_horizon"]),
            "teacher-free allocation screen only",
        ),
        (
            "dagger-train",
            int(split_config["dagger_train_count"]),
            int(budgets["dagger_horizon_ticks"]),
            "on-policy exact querying and training only",
        ),
        (
            "barrier-calibration",
            int(split_config["barrier_calibration_count"]),
            int(budgets["calibration_horizon_ticks"]),
            "whole-seed fit and split-conformal calibration only",
        ),
        (
            "barrier-heldout",
            int(split_config["barrier_heldout_count"]),
            int(budgets["heldout_horizon_ticks"]),
            "final-student on-policy false-safe and coverage only",
        ),
        (
            "barrier-stress",
            int(split_config["barrier_stress_count"]),
            int(budgets["stress_horizon_ticks"]),
            "targeted exact unsafe-proposal false-safe stress only",
        ),
    )
    rows = []
    seen: set[int] = set()
    for split, count, horizon, purpose in specs:
        for index in range(count):
            seed = _derive_seed(split, index)
            if seed in seen:
                raise RuntimeError("R3G seed collision across materialized splits")
            seen.add(seed)
            row = {
                "split": split,
                "index": index,
                "seed": seed,
                "seed_hex": f"0x{seed:08X}",
                "horizon": horizon,
                "purpose": purpose,
            }
            rows.append({**row, "sha256": _canonical_sha256(row)})
    manifest = {
        "schema": "irisu-r3g-seed-manifest-v1",
        "derivation": (
            'unsigned big-endian digest[0:4] of SHA256("'
            f'{SEED_PREFIX}|S|i")'
        ),
        "rows": rows,
    }
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _seeds(manifest: Mapping[str, Any], split: str) -> tuple[int, ...]:
    return tuple(
        int(value["seed"])
        for value in manifest["rows"]
        if value["split"] == split
    )


def _base_identity(config: Mapping[str, Any]) -> str:
    policy = {
        "type": "frozen-r3d-v5-goal-conditioned-steering-policy",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "cooldown_ticks": 16,
        "minimum_pair_closure_sizes": 0.05,
        "impact_side_sizes": 0.5,
        "impact_below_sizes": 0.75,
        "source_velocity_lead_ticks": 1.0,
        "ticks_per_second": 50.0,
        "act_logit_bias": 1.0,
        "candidate_config": dict(config["candidate"]),
    }
    return _canonical_sha256(policy)


def _base_policy(
    model: GoalConditionedSteeringModel, identity: str
) -> GoalConditionedSteeringPolicy:
    model.eval()
    return GoalConditionedSteeringPolicy(
        model,
        cooldown_ticks=16,
        minimum_pair_closure_sizes=0.05,
        impact_side_sizes=0.5,
        impact_below_sizes=0.75,
        source_velocity_lead_ticks=1.0,
        ticks_per_second=50.0,
        act_logit_bias=1.0,
        artifact_sha256=identity,
    )


def _planner_components(
    model: GoalConditionedSteeringModel,
    identity: str,
    config: Mapping[str, Any],
) -> tuple[Any, EventMPCConfig, Callable[..., Any]]:
    candidate = config["candidate"]
    world = config["world_model"]
    joint_config = JOINT.JointPlannerConfig(
        pair_cap=int(candidate["pair_cap"]),
        geometry_cap=int(candidate["geometry_cap"]),
        horizons=(48, 160),
        cooldown_ticks=int(candidate["cooldown_ticks"]),
        velocity_lead_ticks=1.0,
        ticks_per_second=50.0,
        require_pristine_source=bool(candidate["require_pristine_source"]),
    )
    if joint_config.branch_cap != int(candidate["candidate_cap"]):
        raise RuntimeError("pre-registered candidate cap is inconsistent")
    searcher = JOINT.JointPairGeometrySearch(
        lambda: _base_policy(model, identity),
        config=joint_config,
        continuation_identity_sha256=identity,
    )
    event_config = EventMPCConfig(
        renewal_events=int(world["renewal_events"]),
        maximum_event_ticks=int(world["maximum_event_ticks"]),
        cooldown_ticks=int(candidate["cooldown_ticks"]),
        rot_delay_ticks=int(world["rot_delay_ticks"]),
        neighbor_count=int(world["neighbor_count"]),
        calibration_fraction=float(world["calibration_fraction"]),
        conformal_alpha=float(world["conformal_alpha"]),
        risk_upper_limit=float(world["risk_upper_limit"]),
        minimum_score_advantage=float(world["minimum_score_advantage"]),
        continuation_checkpoint_ticks=tuple(
            int(value) for value in world["continuation_checkpoint_ticks"]
        ),
    )
    return searcher, event_config, searcher._candidates


def _planner(
    model: GoalConditionedSteeringModel,
    identity: str,
    config: Mapping[str, Any],
) -> ExactEventPlanner:
    _searcher, event_config, provider = _planner_components(
        model, identity, config
    )
    return ExactEventPlanner(
        lambda: _base_policy(model, identity),
        provider,
        config=event_config,
        continuation_identity_sha256=identity,
        continuation_rebind=JOINT._commit_base_decision,
    )


class ExactTeacherPolicy:
    def __init__(
        self,
        env: IrisuEnv,
        base: object,
        planner: ExactEventPlanner,
        *,
        seed: int,
        stride: int,
        maximum_queries: int,
        sink: list[dict[str, object]],
    ) -> None:
        self.env = env
        self.base = base
        self.planner = planner
        self.seed = seed
        self.stride = stride
        self.maximum_queries = maximum_queries
        self.sink = sink
        self.counts: Counter[str] = Counter()
        self.query_ticks: list[int] = []
        self.override_ticks: list[int] = []
        self.barrier_until_tick = 0

    def reset(self, seed: int = 0) -> None:
        getattr(self.base, "reset")(seed)
        self.counts.clear()
        self.query_ticks.clear()
        self.override_ticks.clear()
        self.barrier_until_tick = 0

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = getattr(self.base, "predict")(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("exact teacher base returned a non-decision")
        if not incumbent.is_shot:
            return incumbent
        self.counts["seen_shots"] += 1
        tick = int(observation.get("tick", 0))
        if tick < self.barrier_until_tick:
            self.counts["two_renewal_barrier_abstentions"] += 1
            return incumbent
        if (
            (self.counts["seen_shots"] - 1) % self.stride
            or self.counts["queries"] >= self.maximum_queries
        ):
            self.counts["budget_abstentions"] += 1
            return incumbent
        query_id = (
            f"teacher-screen:{self.seed:08x}:"
            f"{int(observation.get('tick', 0))}:{self.counts['queries']}"
        )
        try:
            result = self.planner.search(
                self.env,
                observation,
                incumbent,
                continuation_policy=self.base,
                query_id=query_id,
            )
        except ValueError:
            self.counts["unsupported_abstentions"] += 1
            return incumbent
        self.counts["queries"] += 1
        tick = int(observation.get("tick", 0))
        self.query_ticks.append(tick)
        self.counts[f"query_tick_bucket/{tick // 1_000}"] += 1
        self.counts["candidate_outcomes"] += len(result.outcomes)
        self.counts["restore_checks"] += result.restore_checks
        self.counts["search_wall_seconds"] += result.wall_seconds
        self.counts["search_cpu_seconds"] += result.cpu_seconds
        self.counts["simulated_branch_ticks"] += sum(
            value.survival_ticks for value in result.outcomes
        )
        self.counts["unresolved_candidates"] += sum(
            not value.two_renewal_complete for value in result.outcomes
        )
        selected = result.selected_ordinal
        self.sink.append(result.manifest())
        if selected == 0:
            self.counts["barrier_abstentions"] += 1
            return incumbent
        candidate = result.candidate_for_ordinal(selected)
        if not JOINT._commit_base_decision(
            self.base, observation, incumbent, candidate.decision
        ):
            self.counts["progress_rebind_abstentions"] += 1
            return incumbent
        self.counts["overrides"] += 1
        self.counts["pair_corrections"] += int(candidate.pair_ordinal != 0)
        self.counts["geometry_corrections"] += int(
            candidate.geometry_ordinal != 0
        )
        self.override_ticks.append(tick)
        self.counts[f"override_tick_bucket/{tick // 1_000}"] += 1
        self.barrier_until_tick = result.outcome_for_ordinal(
            selected
        ).renewable_ticks[1]
        return candidate.decision

    def statistics(self) -> dict[str, int | float]:
        return {
            **dict(sorted(self.counts.items())),
            "first_query_tick": min(self.query_ticks) if self.query_ticks else -1,
            "last_query_tick": max(self.query_ticks) if self.query_ticks else -1,
            "first_override_tick": (
                min(self.override_ticks) if self.override_ticks else -1
            ),
            "last_override_tick": (
                max(self.override_ticks) if self.override_ticks else -1
            ),
        }


class BranchCollectorPolicy:
    def __init__(
        self,
        env: IrisuEnv,
        base: object,
        planner: ExactEventPlanner,
        *,
        split: str,
        seed: int,
        stride: int,
        maximum_queries: int,
        examples: list[ModelExample],
        branches: list[dict[str, object]],
    ) -> None:
        self.env = env
        self.base = base
        self.planner = planner
        self.split = split
        self.seed = seed
        self.stride = stride
        self.maximum_queries = maximum_queries
        self.examples = examples
        self.branches = branches
        self.counts: Counter[str] = Counter()
        self.query_ticks: list[int] = []

    def reset(self, seed: int = 0) -> None:
        getattr(self.base, "reset")(seed)
        self.counts.clear()
        self.query_ticks.clear()

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = getattr(self.base, "predict")(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("collector base returned a non-decision")
        if not incumbent.is_shot:
            return incumbent
        self.counts["seen_shots"] += 1
        if (
            (self.counts["seen_shots"] - 1) % self.stride
            or self.counts["queries"] >= self.maximum_queries
        ):
            return incumbent
        query_id = (
            f"{self.split}:{self.seed:08x}:"
            f"{int(observation.get('tick', 0))}:{self.counts['queries']}"
        )
        try:
            result = self.planner.search(
                self.env,
                observation,
                incumbent,
                continuation_policy=self.base,
                query_id=query_id,
            )
        except ValueError:
            self.counts["unsupported_abstentions"] += 1
            return incumbent
        self.counts["queries"] += 1
        tick = int(observation.get("tick", 0))
        self.query_ticks.append(tick)
        self.counts[f"query_tick_bucket/{tick // 1_000}"] += 1
        values = search_examples(
            result, observation, split=self.split, seed=self.seed
        )
        self.examples.extend(values)
        self.branches.append(
            {
                "split": self.split,
                "seed": self.seed,
                "decision_tick": int(observation.get("tick", 0)),
                "search": result.manifest(),
                "examples": [value.manifest() for value in values],
            }
        )
        self.counts["candidate_outcomes"] += len(result.outcomes)
        self.counts["restore_checks"] += result.restore_checks
        self.counts["search_wall_seconds"] += result.wall_seconds
        self.counts["search_cpu_seconds"] += result.cpu_seconds
        self.counts["simulated_branch_ticks"] += sum(
            value.survival_ticks for value in result.outcomes
        )
        self.counts["unresolved_candidates"] += sum(
            not value.two_renewal_complete for value in result.outcomes
        )
        return incumbent

    def statistics(self) -> dict[str, int | float]:
        return {
            **dict(sorted(self.counts.items())),
            "first_query_tick": min(self.query_ticks) if self.query_ticks else -1,
            "last_query_tick": max(self.query_ticks) if self.query_ticks else -1,
        }


@dataclass(slots=True)
class AuditSink:
    env: IrisuEnv
    planner: ExactEventPlanner
    model: KNNEventWorldModel
    split: str
    seed: int
    maximum_independent_queries: int
    maximum_execution_queries: int
    rows: list[dict[str, object]]
    examples: list[ModelExample]
    queries: int = 0
    independent_queries: int = 0
    execution_queries: int = 0
    proposal_false_safe: bool = False
    committed_false_safe: bool = False
    unsafe_executions: int = 0
    cache: dict[int, tuple[ExactSearchResult, dict[str, object]]] = field(
        default_factory=dict
    )

    @property
    def false_safe(self) -> bool:
        return self.proposal_false_safe or self.committed_false_safe

    def _search(
        self,
        observation: Mapping[str, Any],
        base_policy: object,
        incumbent: SteeringDecision,
        *,
        audit_kind: str,
    ) -> tuple[ExactSearchResult, dict[str, object]]:
        maximum = (
            self.maximum_independent_queries
            + self.maximum_execution_queries
        )
        if self.queries >= maximum:
            raise RuntimeError("exact heldout query budget exhausted")
        query_id = (
            f"{self.split}:{self.seed:08x}:"
            f"{int(observation.get('tick', 0))}:{self.queries}"
        )
        result = self.planner.search(
            self.env,
            observation,
            incumbent,
            continuation_policy=base_policy,
            query_id=query_id,
        )
        self.queries += 1
        examples = search_examples(
            result, observation, split=self.split, seed=self.seed
        )
        self.examples.extend(examples)
        predictions = []
        for candidate, outcome in zip(
            result.candidates, result.outcomes, strict=True
        ):
            prediction, local_certificate = self.model.certify(
                candidate_features(observation, candidate)
            )
            policy_eligible = (
                candidate.ordinal != 0
                and not result.action_equivalent_to_incumbent(
                    candidate.ordinal
                )
            )
            exact_safe = not policy_eligible or result.safe(outcome)
            if (
                policy_eligible
                and local_certificate.certified
                and not exact_safe
            ):
                self.proposal_false_safe = True
            predictions.append(
                {
                    "candidate_ordinal": candidate.ordinal,
                    "exact_target": list(
                        outcome.target_vector(result.incumbent)
                    ),
                    "prediction": asdict(prediction),
                    "certificate": local_certificate.manifest(),
                    "policy_eligible": policy_eligible,
                    "exact_safe": exact_safe,
                    "severe_unsafe": (
                        policy_eligible
                        and not exact_safe
                        and (
                            not outcome.alive
                            or outcome.full_action_gauge_failure
                            or outcome.minimum_surplus < 0
                        )
                    ),
                }
            )
        row: dict[str, object] = {
            "split": self.split,
            "seed": self.seed,
            "decision_tick": int(observation.get("tick", 0)),
            "audit_kind": audit_kind,
            "selected_ordinal": None,
            "selected_certificate": None,
            "selected_exact_safe": None,
            "committed": False,
            "search": result.manifest(),
            "predictions": predictions,
        }
        self.rows.append(row)
        return result, row

    @staticmethod
    def _candidate_signature(candidates: Sequence[Any]) -> tuple[Any, ...]:
        return tuple(
            (
                int(candidate.ordinal),
                int(candidate.pair_ordinal),
                int(candidate.geometry_ordinal),
                candidate.decision.primitive_actions(),
            )
            for candidate in candidates
        )

    def proposal_hook(
        self,
        observation: Mapping[str, Any],
        base_policy: object,
        incumbent: SteeringDecision,
        candidates: Sequence[Any],
    ) -> None:
        incumbent_actions = incumbent.primitive_actions()
        has_non_tied_alternative = any(
            candidate.ordinal != 0
            and candidate.decision.primitive_actions() != incumbent_actions
            for candidate in candidates
        )
        if (
            self.independent_queries >= self.maximum_independent_queries
            or len(candidates) <= 1
            or not has_non_tied_alternative
        ):
            return
        result, row = self._search(
            observation,
            base_policy,
            incumbent,
            audit_kind="independent-first-eligible-state",
        )
        if self._candidate_signature(candidates) != self._candidate_signature(
            result.candidates
        ):
            raise RuntimeError("heldout audit candidate generator changed")
        self.independent_queries += 1
        self.cache[int(observation.get("tick", 0))] = (result, row)

    def selected_hook(
        self,
        observation: Mapping[str, Any],
        base_policy: object,
        incumbent: SteeringDecision,
        selected: Any,
        certificate: CandidateCertificate,
    ) -> object:
        tick = int(observation.get("tick", 0))
        cached = self.cache.get(tick)
        if cached is None:
            if self.execution_queries >= self.maximum_execution_queries:
                raise RuntimeError("exact execution audit budget exhausted")
            cached = self._search(
                observation,
                base_policy,
                incumbent,
                audit_kind="selected-execution-proposal",
            )
            self.execution_queries += 1
        result, row = cached
        exact = result.outcome_for_ordinal(selected.ordinal)
        safe = result.safe(exact)
        row.update(
            {
                "selected_ordinal": selected.ordinal,
                "selected_certificate": certificate.manifest(),
                "selected_exact_safe": safe,
            }
        )
        return {"row": row, "safe": safe}

    def commit_hook(self, token: object) -> None:
        if not isinstance(token, Mapping):
            raise TypeError("exact execution audit token is malformed")
        row = token.get("row")
        if not isinstance(row, dict):
            raise TypeError("exact execution audit row is malformed")
        safe = bool(token.get("safe", False))
        row["committed"] = True
        self.committed_false_safe |= not safe
        self.unsafe_executions += int(not safe)


class StressPolicy:
    def __init__(
        self,
        env: IrisuEnv,
        base: object,
        planner: ExactEventPlanner,
        model: KNNEventWorldModel,
        *,
        seed: int,
        maximum_queries: int,
        rows: list[dict[str, object]],
    ) -> None:
        self.env = env
        self.base = base
        self.planner = planner
        self.model = model
        self.seed = seed
        self.maximum_queries = maximum_queries
        self.rows = rows
        self.counts: Counter[str] = Counter()
        self.has_unsafe = False
        self.false_safe = False
        self.query_ticks: list[int] = []

    def reset(self, seed: int = 0) -> None:
        getattr(self.base, "reset")(seed)
        self.counts.clear()
        self.has_unsafe = False
        self.false_safe = False
        self.query_ticks.clear()

    @staticmethod
    def _target(observation: Mapping[str, Any]) -> bool:
        gauge = int(observation.get("gauge", 0))
        gauge_max = max(1, int(observation.get("gauge_max", 1)))
        level = int(observation.get("level", 1))
        liabilities = sum(
            isinstance(value, Mapping)
            and value.get("kind") == "piece"
            and value.get("lifecycle") != "rotten"
            and int(value.get("rot_timer", 0)) > 0
            for value in observation.get("bodies", ())
        )
        return gauge <= 0.35 * gauge_max or level >= 3 or liabilities >= 2

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = getattr(self.base, "predict")(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("stress base returned a non-decision")
        if (
            not incumbent.is_shot
            or self.has_unsafe
            or self.counts["queries"] >= self.maximum_queries
            or not self._target(observation)
        ):
            return incumbent
        query_id = (
            f"barrier-stress:{self.seed:08x}:"
            f"{int(observation.get('tick', 0))}:{self.counts['queries']}"
        )
        try:
            result = self.planner.search(
                self.env,
                observation,
                incumbent,
                continuation_policy=self.base,
                query_id=query_id,
            )
        except ValueError:
            self.counts["unsupported"] += 1
            return incumbent
        self.counts["queries"] += 1
        tick = int(observation.get("tick", 0))
        self.query_ticks.append(tick)
        self.counts[f"query_tick_bucket/{tick // 1_000}"] += 1
        self.counts["search_wall_seconds"] += result.wall_seconds
        self.counts["search_cpu_seconds"] += result.cpu_seconds
        self.counts["candidate_outcomes"] += len(result.outcomes)
        self.counts["restore_checks"] += result.restore_checks
        self.counts["simulated_branch_ticks"] += sum(
            value.survival_ticks for value in result.outcomes
        )
        predictions = []
        unsafe = []
        for candidate, outcome in zip(
            result.candidates, result.outcomes, strict=True
        ):
            prediction, certificate = self.model.certify(
                candidate_features(observation, candidate)
            )
            exact_unsafe = (
                candidate.ordinal != 0
                and not result.action_equivalent_to_incumbent(
                    candidate.ordinal
                )
                and not result.safe(outcome)
            )
            if exact_unsafe:
                unsafe.append(candidate.ordinal)
                self.false_safe |= certificate.certified
            predictions.append(
                {
                    "candidate_ordinal": candidate.ordinal,
                    "exact_unsafe": exact_unsafe,
                    "exact_target": list(
                        outcome.target_vector(result.incumbent)
                    ),
                    "prediction": asdict(prediction),
                    "certificate": certificate.manifest(),
                }
            )
        self.has_unsafe |= bool(unsafe)
        self.rows.append(
            {
                "split": "barrier-stress",
                "seed": self.seed,
                "decision_tick": int(observation.get("tick", 0)),
                "target_state": {
                    "gauge": int(observation.get("gauge", 0)),
                    "gauge_max": int(observation.get("gauge_max", 0)),
                    "level": int(observation.get("level", 0)),
                    "visible_rot_liabilities": sum(
                        isinstance(value, Mapping)
                        and value.get("kind") == "piece"
                        and value.get("lifecycle") != "rotten"
                        and int(value.get("rot_timer", 0)) > 0
                        for value in observation.get("bodies", ())
                    ),
                },
                "unsafe_ordinals": unsafe,
                "search": result.manifest(),
                "predictions": predictions,
            }
        )
        return incumbent

    def statistics(self) -> dict[str, int | float]:
        return {
            **dict(sorted(self.counts.items())),
            "has_exact_unsafe": int(self.has_unsafe),
            "false_safe": int(self.false_safe),
            "first_query_tick": min(self.query_ticks) if self.query_ticks else -1,
            "last_query_tick": max(self.query_ticks) if self.query_ticks else -1,
        }


def _run_suite(
    seeds: Sequence[int],
    horizon: int,
    factory: Callable[[IrisuEnv, int], object],
    label: str,
) -> tuple[list[Any], Mapping[str, object], Mapping[str, object]]:
    episodes = []
    runners = []
    attestations = []
    for seed in seeds:
        episode, runner, attestation = TRUSTED_BENCHMARK._run_episode(
            library=RUNTIME,
            label=label,
            seed=int(seed),
            horizon=horizon,
            factory=lambda env, value=int(seed): factory(env, value),
        )
        episodes.append(episode)
        runners.append(runner)
        attestations.append(attestation)
    if len({_canonical_sha256(value) for value in runners}) != 1:
        raise RuntimeError("Strategy C suite mixed runner identities")
    if len({_canonical_sha256(value) for value in attestations}) != 1:
        raise RuntimeError("Strategy C suite mixed runtime attestations")
    return episodes, runners[0], attestations[0]


def _distribution(values: Sequence[int | float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.1, method="linear")),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


def _paired(base: Sequence[Any], candidate: Sequence[Any]) -> dict[str, object]:
    first = {value.seed: value for value in base}
    second = {value.seed: value for value in candidate}
    if set(first) != set(second):
        raise ValueError("paired suites use different seeds")
    rows = []
    for seed in sorted(first):
        left, right = first[seed], second[seed]
        regression = left.survival_ticks - right.survival_ticks
        material = regression >= 1_000
        catastrophic = (
            material
            and left.survival_ticks > 0
            and right.survival_ticks / left.survival_ticks <= 0.5
        )
        rows.append(
            {
                "seed": seed,
                "seed_hex": f"0x{seed:08X}",
                "base_score": left.final_score,
                "candidate_score": right.final_score,
                "score_delta": right.final_score - left.final_score,
                "base_survival": left.survival_ticks,
                "candidate_survival": right.survival_ticks,
                "survival_delta": right.survival_ticks - left.survival_ticks,
                "new_terminal": right.terminated and not left.terminated,
                "new_gauge_failure": right.gauge_failure and not left.gauge_failure,
                "material_survival_regression": material,
                "catastrophic_paired_regression": catastrophic,
                "base_gauge": left.final_gauge,
                "candidate_gauge": right.final_gauge,
                "base_level": left.final_level,
                "candidate_level": right.final_level,
                "base_qualifying_clears": left.qualifying_clears,
                "candidate_qualifying_clears": right.qualifying_clears,
                "base_rotten_events": left.rotten_events,
                "candidate_rotten_events": right.rotten_events,
            }
        )
    return {
        "rows": rows,
        "new_terminals": sum(value["new_terminal"] for value in rows),
        "new_gauge_failures": sum(
            value["new_gauge_failure"] for value in rows
        ),
        "material_regressions": sum(
            value["material_survival_regression"] for value in rows
        ),
        "catastrophic_regressions": sum(
            value["catastrophic_paired_regression"] for value in rows
        ),
    }


def _teacher_gate(
    baseline: Sequence[Any], teacher: Sequence[Any]
) -> dict[str, object]:
    paired = _paired(baseline, teacher)
    base_score = _distribution([value.final_score for value in baseline])
    other_score = _distribution([value.final_score for value in teacher])
    base_survival = _distribution(
        [value.survival_ticks for value in baseline]
    )
    other_survival = _distribution(
        [value.survival_ticks for value in teacher]
    )
    horizon = int(baseline[0].horizon_ticks)
    at_ceiling = math.isclose(base_survival["p10"], horizon)
    survival_pass = (
        other_survival["p10"] > base_survival["p10"]
        if not at_ceiling
        else (
            math.isclose(other_survival["p10"], horizon)
            and not any(
                row["survival_delta"] < 0 for row in paired["rows"]
            )
        )
    )
    checks = {
        "zero_new_terminal": paired["new_terminals"] == 0,
        "zero_new_gauge_failure": paired["new_gauge_failures"] == 0,
        "zero_catastrophic_regression": paired["catastrophic_regressions"] == 0,
        "score_median_improved": (
            other_score["median"] > base_score["median"]
        ),
        "survival_p10_gate": survival_pass,
    }
    return {
        "passed": all(checks.values()),
        "allocation_screen_only": True,
        "checks": checks,
        "baseline_score": base_score,
        "teacher_score": other_score,
        "baseline_survival": base_survival,
        "teacher_survival": other_survival,
        "paired": paired,
    }


def _clopper_pearson_upper(failures: int, episodes: int) -> float:
    if episodes < 1 or not 0 <= failures <= episodes:
        raise ValueError("invalid binomial cohort")
    if failures == episodes:
        return 1.0
    alpha = 0.05

    def cdf(probability: float) -> float:
        return sum(
            math.comb(episodes, index)
            * probability**index
            * (1.0 - probability) ** (episodes - index)
            for index in range(failures + 1)
        )

    low, high = 0.0, 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if cdf(middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def _coverage_lower(
    episodes: Sequence[tuple[int, int]]
) -> tuple[float, float]:
    numerator = sum(value[0] for value in episodes)
    denominator = sum(value[1] for value in episodes)
    estimate = numerator / denominator if denominator else 0.0
    seed = int.from_bytes(
        hashlib.sha256(f"{SEED_PREFIX}|bootstrap|coverage".encode()).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(10_000):
        indices = rng.integers(0, len(episodes), size=len(episodes))
        top = sum(episodes[index][0] for index in indices)
        bottom = sum(episodes[index][1] for index in indices)
        values.append(top / bottom if bottom else 0.0)
    return estimate, float(np.quantile(values, 0.05, method="linear"))


def _model_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    exact = []
    predicted = []
    certified = []
    unsafe = []
    catastrophic = []
    risk = []
    for row in rows:
        for value in row.get("predictions", ()):
            exact.append(value["exact_target"])
            predicted.append(value["prediction"]["mean"])
            certificate = bool(value["certificate"]["certified"])
            exact_safe = bool(
                value.get("exact_safe", not value.get("exact_unsafe", False))
            )
            certified.append(certificate)
            unsafe.append(not exact_safe)
            catastrophic.append(
                bool(
                    value["exact_target"][
                        TARGET_NAMES.index("catastrophic")
                    ]
                )
            )
            risk.append(float(value["prediction"]["catastrophic_probability"]))
    if not exact:
        return {"rows": 0, "horizon_breakdown": "no audited proposals"}
    actual = np.asarray(exact, dtype=np.float64)
    estimate = np.asarray(predicted, dtype=np.float64)
    errors = estimate - actual
    metrics = {
        name: {
            "mae": float(np.mean(np.abs(errors[:, index]))),
            "rmse": float(np.sqrt(np.mean(errors[:, index] ** 2))),
            "bias": float(np.mean(errors[:, index])),
        }
        for index, name in enumerate(TARGET_NAMES)
    }
    order = np.argsort(np.asarray(risk), kind="stable")
    bins = []
    for decile, indices in enumerate(np.array_split(order, 10)):
        if not len(indices):
            continue
        bins.append(
            {
                "decile": decile,
                "rows": len(indices),
                "mean_predicted_risk": float(
                    np.mean(np.asarray(risk)[indices])
                ),
                "observed_catastrophic_rate": float(
                    np.mean(
                        np.asarray(catastrophic, dtype=np.float64)[indices]
                    )
                ),
                "observed_barrier_unsafe_rate": float(
                    np.mean(np.asarray(unsafe, dtype=np.float64)[indices])
                ),
                "certified": int(
                    np.asarray(certified, dtype=np.int64)[indices].sum()
                ),
            }
        )
    second = TARGET_NAMES.index("second_renewal_reached")
    return {
        "rows": len(exact),
        "multi_step_error": metrics,
        "risk_deciles": bins,
        "exact_second_renewal_unresolved": int(
            np.sum(actual[:, second] < 1.0)
        ),
        "predicted_second_renewal_unresolved": int(
            np.sum(estimate[:, second] < 0.5)
        ),
    }


def _episode_manifests(values: Sequence[Any]) -> list[dict[str, object]]:
    return [value.manifest() for value in values]


def _aggregate(values: Sequence[Any]) -> dict[str, object]:
    return TRUSTED_BENCHMARK._aggregate(values)


def _sentinels(
    model: GoalConditionedSteeringModel,
    identity: str,
) -> dict[str, object]:
    expected = {
        1_339_819_096: {
            "survival_ticks": 1_845,
            "final_score": 36,
            "final_gauge": 1,
            "final_level": 1,
            "qualifying_clears": 2,
            "terminated": True,
            "truncated": False,
        },
        1_979_477_456: {
            "survival_ticks": 2_000,
            "final_score": 80,
            "final_gauge": 4_780,
            "final_level": 1,
            "qualifying_clears": 6,
            "terminated": False,
            "truncated": True,
        },
    }
    episodes, runner, attestation = _run_suite(
        tuple(expected),
        2_000,
        lambda _env, _seed: _base_policy(model, identity),
        "frozen_v5_sentinel",
    )
    checks = []
    for episode in episodes:
        actual = {
            key: getattr(episode, key) for key in expected[episode.seed]
        }
        checks.append(
            {
                "seed": episode.seed,
                "expected": expected[episode.seed],
                "actual": actual,
                "expected_sha256": _canonical_sha256(
                    expected[episode.seed]
                ),
                "actual_sha256": _canonical_sha256(actual),
                "matched": actual == expected[episode.seed],
            }
        )
    return {
        "passed": all(value["matched"] for value in checks),
        "source": (
            "trusted corrected-joint-v2 development report frozen-v5 "
            "teacher-ceiling baseline"
        ),
        "checks": checks,
        "runner": runner,
        "runtime_attestation": attestation,
    }


def _correctness(output: Path) -> dict[str, object]:
    del output
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONPATH": str(ROOT / "python"),
    }
    commands = (
        [
            "taskset",
            "-c",
            str(LOGICAL_CPU),
            sys.executable,
            str(DEVELOPMENT_CHECKS),
        ],
        [
            "taskset",
            "-c",
            str(LOGICAL_CPU),
            sys.executable,
            str(JOINT_CHECKS),
        ],
    )
    rows = []
    for index, command in enumerate(commands):
        command_environment = dict(environment)
        if index == 1:
            command_environment["PYTHONPATH"] = str(
                Path("/home/gabe/.codex/worktrees/e2f8/irisu/python")
            )
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        wall_started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=command_environment,
            text=True,
            capture_output=True,
        )
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
        child_cpu = (
            usage_after.ru_utime
            + usage_after.ru_stime
            - usage_before.ru_utime
            - usage_before.ru_stime
        )
        rows.append(
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "wall_seconds": time.perf_counter() - wall_started,
                "child_cpu_seconds": child_cpu,
            }
        )
    return {
        "passed": all(value["returncode"] == 0 for value in rows),
        "commands": rows,
        "logical_cpu": LOGICAL_CPU,
    }


def _cost(
    started_wall: float,
    started_cpu: float,
    correctness: Mapping[str, Any],
) -> dict[str, object]:
    return {
        "wall_seconds": time.perf_counter() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
        "development_check_child_cpu_seconds": sum(
            float(value.get("child_cpu_seconds", 0.0))
            for value in correctness.get("commands", ())
        ),
        "max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "affinity": sorted(os.sched_getaffinity(0)),
        "platform": platform.platform(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def campaign(args: argparse.Namespace) -> None:
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    _progress("preflight")
    output = _safe_output(args.output)
    preformal_manifest = _preformal_manifest(output)
    config = _load_config(args.config)
    source = _source_identity(args.config)
    _require_identity(source, args.config)
    seed_manifest = _seed_manifest(config)
    seed_manifest_sha = _atomic_json(
        output / "seed-manifest.json", seed_manifest
    )
    config_sha = _file_sha256(args.config)
    preregistration = {
        "schema": "irisu-r3g-event-mpc-preregistration-v1",
        "created_before_screen_results": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": source["sha256"],
        "config_sha256": config_sha,
        "seed_manifest_sha256": seed_manifest_sha,
        "preformal_manifest": preformal_manifest,
        "candidate": dict(config["candidate"]),
        "world_model": dict(config["world_model"]),
        "budgets": dict(config["budgets"]),
        "policy": dict(config["policy"]),
        "barrier": dict(config["barrier"]),
        "splits": dict(config["splits"]),
        "training_data_budget": {
            "barrier_calibration_whole_seeds": 64,
            "exact_decisions_per_calibration_seed": int(
                config["budgets"]["calibration_decisions_per_seed"]
            ),
            "dagger_whole_seeds": 16,
            "exact_decisions_per_dagger_seed": int(
                config["budgets"]["dagger_decisions_per_seed"]
            ),
            "maximum_candidates_per_decision": int(
                config["candidate"]["candidate_cap"]
            ),
        },
        "screens_are_allocation_only": True,
        "parent_only_winner_suites_materialized": False,
    }
    preregistration["sha256"] = _canonical_sha256(preregistration)
    preregistration_sha = _atomic_json(
        output / "pre-registration.json", preregistration
    )
    _progress("development-checks")
    artifact = load_steering_checkpoint(
        CHECKPOINT, expected_sha256=CHECKPOINT_SHA256
    )
    base_model = artifact.model
    base_model.eval()
    identity = _base_identity(config)
    correctness = _correctness(output)
    _atomic_json(output / "correctness.json", correctness)
    if not correctness["passed"]:
        raise RuntimeError("Strategy C correctness gate failed")
    _progress("sentinels")
    sentinel = _sentinels(base_model, identity)
    _atomic_json(output / "sentinel.json", sentinel)
    if not sentinel["passed"]:
        raise RuntimeError("frozen-v5 sentinel reproduction failed")
    _progress("teacher-baseline", seeds=8, horizon=2_000)
    teacher_branches: list[dict[str, object]] = []
    teacher_seeds = _seeds(seed_manifest, "teacher-screen")
    teacher_horizon = int(config["splits"]["teacher_screen_horizon"])
    baseline, runner, attestation = _run_suite(
        teacher_seeds,
        teacher_horizon,
        lambda _env, _seed: _base_policy(base_model, identity),
        "frozen_v5",
    )
    _progress("teacher-exact", seeds=8, horizon=2_000)
    teacher, _, _ = _run_suite(
        teacher_seeds,
        teacher_horizon,
        lambda env, seed: ExactTeacherPolicy(
            env,
            _base_policy(base_model, identity),
            _planner(base_model, identity, config),
            seed=seed,
            stride=int(config["budgets"]["teacher_query_stride_shots"]),
            maximum_queries=int(
                config["budgets"]["teacher_queries_per_seed"]
            ),
            sink=teacher_branches,
        ),
        "event_exact_teacher",
    )
    teacher_gate = _teacher_gate(baseline, teacher)
    teacher_report = {
        "gate": teacher_gate,
        "baseline": {
            "aggregate": _aggregate(baseline),
            "episodes": _episode_manifests(baseline),
        },
        "teacher": {
            "aggregate": _aggregate(teacher),
            "episodes": _episode_manifests(teacher),
        },
        "runner": runner,
        "runtime_attestation": attestation,
        "branches": teacher_branches,
    }
    _atomic_json(output / "teacher-screen.json", teacher_report)
    gates: dict[str, object] = {
        "correctness": True,
        "sentinel": True,
        "teacher_screen": bool(teacher_gate["passed"]),
    }
    report: dict[str, Any] = {
        "schema": "irisu-r3g-event-world-model-campaign-v1",
        "development_only": True,
        "sealed_test_material_used": False,
        "canonical_r3_evidence": False,
        "authorization_material_used": False,
        "parent_winner_materialized": False,
        "screens_are_allocation_only": True,
        "source_identity": source,
        "config_sha256": config_sha,
        "seed_manifest_sha256": seed_manifest_sha,
        "preregistration_sha256": preregistration_sha,
        "preregistration": preregistration,
        "preformal_manifest": preformal_manifest,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_path": str(PROTOCOL),
        "runtime_sha256": RUNTIME_SHA256,
        "runtime_path": str(RUNTIME),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_path": str(CHECKPOINT),
        "base_policy_identity_sha256": identity,
        "joint_source_sha256": JOINT_SOURCE_SHA256,
        "joint_source_path": str(JOINT_SOURCE),
        "joint_benchmark_sha256": JOINT_BENCHMARK_SHA256,
        "development_checks": correctness,
        "sentinel_reproduction": sentinel,
        "gates": gates,
        "teacher_screen": teacher_report,
        "limitations": [
            "Development-only; no sealed, test-phase, canonical, authorization, "
            "or parent-only winner material was opened.",
            "The teacher and student suites are allocation screens only.",
            "The student can make at most one override per episode because exact "
            "renewal-event provenance is not present in public observations.",
            "An unresolved second renewal is a horizon breakdown and always "
            "abstains; the claimed horizon is never shortened.",
        ],
    }
    if not teacher_gate["passed"]:
        report["verdict"] = (
            "Rejected at the exact-teacher allocation screen; later splits "
            "were not opened."
        )
        report["cost"] = _cost(started_wall, started_cpu, correctness)
        report["artifact_identities"] = {
            "seed_manifest": seed_manifest_sha,
            "preregistration": preregistration_sha,
        }
        _require_identity(source, args.config)
        _require_artifact_hashes(
            output,
            {
                "seed-manifest.json": seed_manifest_sha,
                "pre-registration.json": preregistration_sha,
            },
        )
        finalize_artifacts(output, report)
        _progress("rejected-teacher")
        return

    _progress("barrier-calibration", seeds=64)
    calibration_examples: list[ModelExample] = []
    calibration_branches: list[dict[str, object]] = []
    calibration_seeds = _seeds(seed_manifest, "barrier-calibration")
    calibration, _, _ = _run_suite(
        calibration_seeds,
        int(config["budgets"]["calibration_horizon_ticks"]),
        lambda env, seed: BranchCollectorPolicy(
            env,
            _base_policy(base_model, identity),
            _planner(base_model, identity, config),
            split="barrier-calibration",
            seed=seed,
            stride=int(config["budgets"]["calibration_query_stride_shots"]),
            maximum_queries=int(
                config["budgets"]["calibration_decisions_per_seed"]
            ),
            examples=calibration_examples,
            branches=calibration_branches,
        ),
        "barrier_calibration_collector",
    )
    represented = {value.seed for value in calibration_examples}
    if represented != set(calibration_seeds):
        raise RuntimeError("calibration did not retain every whole-seed cluster")
    branch_sha = _atomic_jsonl(
        output / "barrier-calibration-branches.jsonl",
        calibration_branches,
    )
    _atomic_json(
        output / "barrier-calibration-summary.json",
        {
            "seeds": list(calibration_seeds),
            "whole_seed_clusters": len(represented),
            "examples": len(calibration_examples),
            "queries": len(calibration_branches),
            "branches_sha256": branch_sha,
            "aggregate": _aggregate(calibration),
            "episodes": _episode_manifests(calibration),
        },
    )
    _searcher, event_config, provider = _planner_components(
        base_model, identity, config
    )
    world_model = KNNEventWorldModel(event_config)
    world_model.fit_provisional(calibration_examples)

    _progress("dagger", seeds=16)
    dagger_rows: list[dict[str, object]] = []
    dagger_examples: list[ModelExample] = []
    dagger_seeds = _seeds(seed_manifest, "dagger-train")
    dagger_audits: dict[int, AuditSink] = {}

    def dagger_factory(env: IrisuEnv, seed: int) -> ModelBarrierPolicy:
        sink = AuditSink(
            env,
            _planner(base_model, identity, config),
            world_model,
            "dagger-train",
            seed,
            0,
            int(config["budgets"]["dagger_decisions_per_seed"]),
            dagger_rows,
            dagger_examples,
        )
        dagger_audits[seed] = sink
        return ModelBarrierPolicy(
            _base_policy(base_model, identity),
            provider,
            JOINT._commit_base_decision,
            world_model,
            query_stride_shots=int(
                config["budgets"]["student_query_stride_shots"]
            ),
            maximum_overrides=int(
                config["policy"]["maximum_overrides_per_episode"]
            ),
            audit_hook=sink.selected_hook,
            audit_commit_hook=sink.commit_hook,
        )

    try:
        dagger, _, _ = _run_suite(
            dagger_seeds,
            int(config["budgets"]["dagger_horizon_ticks"]),
            dagger_factory,
            "event_model_dagger_train",
        )
    except RuntimeError as exc:
        if "query budget exhausted" not in str(exc):
            raise
        raise RuntimeError(
            "DAgger exact query cap exhausted instead of abstaining"
        ) from exc
    world_model.fit(
        calibration_examples, extra_training=dagger_examples
    )
    dagger_sha = _atomic_jsonl(
        output / "dagger-train-branches.jsonl", dagger_rows
    )
    _atomic_json(
        output / "dagger-train-summary.json",
        {
            "seeds": list(dagger_seeds),
            "examples": len(dagger_examples),
            "branches_sha256": dagger_sha,
            "aggregate": _aggregate(dagger),
            "episodes": _episode_manifests(dagger),
        },
    )
    training_manifest = {
        "calibration_branch_sha256": branch_sha,
        "dagger_branch_sha256": dagger_sha,
        "calibration_examples": len(calibration_examples),
        "dagger_examples": len(dagger_examples),
        "calibration_seeds": list(calibration_seeds),
        "dagger_seeds": list(dagger_seeds),
    }
    training_manifest["sha256"] = _canonical_sha256(training_manifest)
    if not math.isfinite(world_model.conformal_q):
        gates["barrier_calibration"] = False
        report["barrier_calibration"] = {
            "passed": False,
            "reason": "nonfinite whole-seed conformal threshold",
            "conformal_q": "inf",
            "conformal_alpha": event_config.conformal_alpha,
            "fit_seeds": list(world_model.fit_seeds),
            "calibration_seeds": list(world_model.calibration_seeds),
            "training_manifest": training_manifest,
        }
        report["verdict"] = (
            "Rejected before heldout: the whole-seed conformal threshold "
            "was nonfinite, so no two-renewal safety claim was made."
        )
        report["cost"] = _cost(started_wall, started_cpu, correctness)
        report["artifact_identities"] = {
            "seed_manifest": seed_manifest_sha,
            "preregistration": preregistration_sha,
            "calibration_branches": branch_sha,
            "dagger_branches": dagger_sha,
        }
        _require_identity(source, args.config)
        _require_artifact_hashes(
            output,
            {
                "seed-manifest.json": seed_manifest_sha,
                "pre-registration.json": preregistration_sha,
                "barrier-calibration-branches.jsonl": branch_sha,
                "dagger-train-branches.jsonl": dagger_sha,
            },
        )
        finalize_artifacts(output, report)
        _progress("rejected-calibration")
        return
    gates["barrier_calibration"] = True
    model_manifest = world_model.manifest()
    model_sha = _atomic_json(output / "world-model.json", model_manifest)
    freeze = {
        "schema": "irisu-r3g-event-model-heldout-freeze-v1",
        "source_identity_sha256": source["sha256"],
        "config_sha256": config_sha,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "runtime_sha256": RUNTIME_SHA256,
        "protocol_sha256": PROTOCOL_SHA256,
        "seed_manifest_sha256": seed_manifest_sha,
        "candidate_generator": {
            "source_sha256": JOINT_SOURCE_SHA256,
            "config_sha256": _searcher.config.sha256,
            "candidate_cap": int(config["candidate"]["candidate_cap"]),
        },
        "model_sha256": model_sha,
        "conformal_q": world_model.conformal_q,
        "conformal_alpha": event_config.conformal_alpha,
        "training_manifest": training_manifest,
        "exact_simulator_budget": {
            "maximum_event_ticks": event_config.maximum_event_ticks,
            "renewal_events": 2,
            "calibration_decisions_per_seed": int(
                config["budgets"]["calibration_decisions_per_seed"]
            ),
            "heldout_queries_per_seed": int(
                config["budgets"]["heldout_exact_queries_per_seed"]
            ),
            "heldout_independent_queries_per_seed": int(
                config["budgets"]["heldout_independent_queries_per_seed"]
            ),
            "heldout_execution_queries_per_seed": int(
                config["budgets"]["heldout_execution_queries_per_seed"]
            ),
            "stress_queries_per_seed": int(
                config["budgets"]["stress_exact_queries_per_seed"]
            ),
        },
        "final_policy": {
            "base_policy_identity_sha256": identity,
            "candidate_query_stride_shots": int(
                config["budgets"]["student_query_stride_shots"]
            ),
            "maximum_overrides_per_episode": int(
                config["policy"]["maximum_overrides_per_episode"]
            ),
            "override_epoch": config["policy"]["override_epoch"],
            "heldout_sampling": config["policy"]["heldout_sampling"],
            "selected_audit": (
                "exact before commit; unsafe execution counted only after "
                "successful controller rebind"
            ),
            "action_ties": "primitive-equivalent candidates abstain",
            "top_ties": "complete certificate-objective ties abstain",
        },
        "coverage_estimator": {
            "unit": "eligible on-policy decision states / actual overrides",
            "clusters": "whole seed episode",
            "replicates": 10_000,
            "lower_quantile": 0.05,
            "rng": f"{SEED_PREFIX}|bootstrap|coverage",
        },
    }
    freeze["sha256"] = _canonical_sha256(freeze)
    freeze_sha = _atomic_json(output / "heldout-freeze.json", freeze)
    _require_identity(source, args.config)

    _progress("barrier-heldout", seeds=64)
    heldout_rows: list[dict[str, object]] = []
    heldout_examples: list[ModelExample] = []
    heldout_audits: dict[int, AuditSink] = {}
    heldout_seeds = _seeds(seed_manifest, "barrier-heldout")

    def heldout_factory(env: IrisuEnv, seed: int) -> ModelBarrierPolicy:
        sink = AuditSink(
            env,
            _planner(base_model, identity, config),
            world_model,
            "barrier-heldout",
            seed,
            int(
                config["budgets"]["heldout_independent_queries_per_seed"]
            ),
            int(config["budgets"]["heldout_execution_queries_per_seed"]),
            heldout_rows,
            heldout_examples,
        )
        heldout_audits[seed] = sink
        return ModelBarrierPolicy(
            _base_policy(base_model, identity),
            provider,
            JOINT._commit_base_decision,
            world_model,
            query_stride_shots=int(
                config["budgets"]["student_query_stride_shots"]
            ),
            maximum_overrides=int(
                config["policy"]["maximum_overrides_per_episode"]
            ),
            proposal_audit_hook=sink.proposal_hook,
            audit_hook=sink.selected_hook,
            audit_commit_hook=sink.commit_hook,
        )

    heldout, _, _ = _run_suite(
        heldout_seeds,
        int(config["budgets"]["heldout_horizon_ticks"]),
        heldout_factory,
        "event_model_barrier_heldout",
    )
    heldout_false = sum(
        heldout_audits[seed].false_safe for seed in heldout_seeds
    )
    independent_rows = [
        row
        for row in heldout_rows
        if row.get("audit_kind") == "independent-first-eligible-state"
    ]
    unsafe_by_row = [
        [
            value
            for value in row.get("predictions", ())
            if int(value.get("candidate_ordinal", 0)) != 0
            and not bool(value.get("exact_safe", False))
        ]
        for row in independent_rows
    ]
    heldout_unsafe_outcomes = sum(len(values) for values in unsafe_by_row)
    heldout_unsafe_states = sum(bool(values) for values in unsafe_by_row)
    heldout_unsafe_seeds = len(
        {
            int(row["seed"])
            for row, values in zip(
                independent_rows, unsafe_by_row, strict=True
            )
            if values
        }
    )
    heldout_severe_outcomes = sum(
        bool(value.get("severe_unsafe", False))
        for values in unsafe_by_row
        for value in values
    )
    heldout_audited_episodes = len(
        {int(row["seed"]) for row in independent_rows}
    )
    coverage_clusters = [
        (
            int(episode.policy_counts.get("overrides", 0)),
            int(episode.policy_counts.get("eligible_decision_states", 0)),
        )
        for episode in heldout
    ]
    coverage, coverage_lower = _coverage_lower(coverage_clusters)
    heldout_cp = _clopper_pearson_upper(
        heldout_false, len(heldout_seeds)
    )
    heldout_valid = len(heldout_seeds) >= 59
    heldout_report = {
        "seeds": list(heldout_seeds),
        "episodes": _episode_manifests(heldout),
        "aggregate": _aggregate(heldout),
        "false_safe_episodes": heldout_false,
        "episode_false_safe": {
            f"0x{seed:08X}": heldout_audits[seed].false_safe
            for seed in heldout_seeds
        },
        "clopper_pearson_upper_95": heldout_cp,
        "cohort_valid": heldout_valid,
        "independently_audited_episodes": heldout_audited_episodes,
        "exact_unsafe_outcomes": heldout_unsafe_outcomes,
        "exact_unsafe_states": heldout_unsafe_states,
        "exact_unsafe_seeds": heldout_unsafe_seeds,
        "severe_unsafe_outcomes": heldout_severe_outcomes,
        "coverage": coverage,
        "coverage_clustered_lower_95": coverage_lower,
        "coverage_clusters": coverage_clusters,
        "diagnostics": _model_diagnostics(heldout_rows),
        "audits": heldout_rows,
    }
    _atomic_json(output / "barrier-heldout.json", heldout_report)
    _require_identity(source, args.config)
    if _file_sha256(output / "world-model.json") != model_sha:
        raise RuntimeError("frozen world model changed after heldout")
    if _file_sha256(output / "heldout-freeze.json") != freeze_sha:
        raise RuntimeError("heldout freeze changed during evaluation")

    _progress("barrier-stress", seeds=64)
    stress_rows: list[dict[str, object]] = []
    stress_policies: dict[int, StressPolicy] = {}
    stress_seeds = _seeds(seed_manifest, "barrier-stress")

    def stress_factory(env: IrisuEnv, seed: int) -> StressPolicy:
        policy = StressPolicy(
            env,
            _base_policy(base_model, identity),
            _planner(base_model, identity, config),
            world_model,
            seed=seed,
            maximum_queries=int(
                config["budgets"]["stress_exact_queries_per_seed"]
            ),
            rows=stress_rows,
        )
        stress_policies[seed] = policy
        return policy

    stress, _, _ = _run_suite(
        stress_seeds,
        int(config["budgets"]["stress_horizon_ticks"]),
        stress_factory,
        "event_model_barrier_stress",
    )
    stress_valid_episodes = sum(
        stress_policies[seed].has_unsafe for seed in stress_seeds
    )
    stress_false = sum(
        stress_policies[seed].false_safe for seed in stress_seeds
    )
    stress_cp = _clopper_pearson_upper(stress_false, len(stress_seeds))
    stress_report = {
        "seeds": list(stress_seeds),
        "episodes": _episode_manifests(stress),
        "aggregate": _aggregate(stress),
        "episodes_with_exact_unsafe": stress_valid_episodes,
        "cohort_valid": stress_valid_episodes == len(stress_seeds),
        "false_safe_episodes": stress_false,
        "episode_false_safe": {
            f"0x{seed:08X}": stress_policies[seed].false_safe
            for seed in stress_seeds
        },
        "clopper_pearson_upper_95": stress_cp,
        "diagnostics": _model_diagnostics(stress_rows),
        "audits": stress_rows,
    }
    _atomic_json(output / "barrier-stress.json", stress_report)
    barrier_checks = {
        "heldout_at_least_59": heldout_valid,
        "heldout_independent_audits_at_least_59": (
            heldout_audited_episodes
            >= int(config["barrier"]["minimum_valid_episodes"])
        ),
        "heldout_unsafe_outcomes_at_least_32": (
            heldout_unsafe_outcomes
            >= int(config["barrier"]["minimum_heldout_unsafe_outcomes"])
        ),
        "heldout_unsafe_states_at_least_8": (
            heldout_unsafe_states
            >= int(config["barrier"]["minimum_heldout_unsafe_states"])
        ),
        "heldout_unsafe_seeds_at_least_4": (
            heldout_unsafe_seeds
            >= int(config["barrier"]["minimum_heldout_unsafe_seeds"])
        ),
        "heldout_severe_unsafe_outcomes_at_least_8": (
            heldout_severe_outcomes
            >= int(config["barrier"]["minimum_heldout_severe_outcomes"])
        ),
        "stress_every_episode_has_unsafe": (
            stress_valid_episodes == len(stress_seeds)
            and stress_valid_episodes >= 59
        ),
        "heldout_zero_false_safe": heldout_false == 0,
        "stress_zero_false_safe": stress_false == 0,
        "heldout_cp_upper_at_most_5pct": heldout_cp
        <= float(config["barrier"]["maximum_false_safe_cp_upper"]),
        "stress_cp_upper_at_most_5pct": stress_cp
        <= float(config["barrier"]["maximum_false_safe_cp_upper"]),
        "coverage_cluster_lower_at_least_5pct": coverage_lower
        >= float(config["barrier"]["minimum_coverage_cluster_lower"]),
        "finite_conformal_threshold": math.isfinite(world_model.conformal_q),
    }
    barrier_passed = all(barrier_checks.values())
    gates["barrier"] = barrier_passed
    report.update(
        {
            "heldout_freeze": freeze,
            "barrier_calibration": {
                "whole_seed_clusters": len(represented),
                "examples": len(calibration_examples),
                "queries": len(calibration_branches),
                "conformal_q": world_model.conformal_q,
                "fit_seeds": list(world_model.fit_seeds),
                "calibration_seeds": list(world_model.calibration_seeds),
            },
            "barrier_heldout": heldout_report,
            "barrier_stress": stress_report,
            "barrier_checks": barrier_checks,
        }
    )
    if not barrier_passed:
        report["verdict"] = (
            "Rejected at the frozen whole-seed barrier gate; the student "
            "screen and parent-only winner suites were not opened."
        )
    else:
        _progress("student-baseline", seeds=8, horizon=10_000)
        student_audits: dict[int, AuditSink] = {}
        student_rows: list[dict[str, object]] = []
        student_examples: list[ModelExample] = []
        student_seeds = _seeds(seed_manifest, "student-screen")
        horizon = int(config["splits"]["student_screen_horizon"])
        student_base, _, _ = _run_suite(
            student_seeds,
            horizon,
            lambda _env, _seed: _base_policy(base_model, identity),
            "frozen_v5",
        )
        _progress("student", seeds=8, horizon=10_000)

        def student_factory(env: IrisuEnv, seed: int) -> ModelBarrierPolicy:
            sink = AuditSink(
                env,
                _planner(base_model, identity, config),
                world_model,
                "student-screen-audit",
                seed,
                0,
                int(config["budgets"]["student_execution_queries_per_seed"]),
                student_rows,
                student_examples,
            )
            student_audits[seed] = sink
            return ModelBarrierPolicy(
                _base_policy(base_model, identity),
                provider,
                JOINT._commit_base_decision,
                world_model,
                query_stride_shots=int(
                    config["budgets"]["student_query_stride_shots"]
                ),
                maximum_overrides=int(
                    config["policy"]["maximum_overrides_per_episode"]
                ),
                audit_hook=sink.selected_hook,
                audit_commit_hook=sink.commit_hook,
            )

        student, _, _ = _run_suite(
            student_seeds,
            horizon,
            student_factory,
            "event_model_student",
        )
        paired = _paired(student_base, student)
        unsafe_actions = sum(
            student_audits[seed].unsafe_executions for seed in student_seeds
        )
        hard_failures = (
            int(paired["new_terminals"])
            + int(paired["new_gauge_failures"])
            + int(paired["catastrophic_regressions"])
            + unsafe_actions
        )
        student_passed = hard_failures == 0
        gates["student_screen"] = student_passed
        student_report = {
            "allocation_screen_only": True,
            "passed": student_passed,
            "hard_failures": hard_failures,
            "exact_unsafe_executions": unsafe_actions,
            "baseline": {
                "aggregate": _aggregate(student_base),
                "episodes": _episode_manifests(student_base),
            },
            "student": {
                "aggregate": _aggregate(student),
                "episodes": _episode_manifests(student),
            },
            "paired": paired,
            "diagnostics": _model_diagnostics(student_rows),
            "audits": student_rows,
        }
        _atomic_json(output / "student-screen.json", student_report)
        report["student_screen"] = student_report
        report["verdict"] = (
            "Passed branch-local allocation screens and frozen barrier; "
            "eligible only for parent ranking, not a positive scientific "
            "conclusion. Parent-only winner suites remain untouched."
            if student_passed
            else (
                "Rejected at the teacher-free student allocation safety "
                "screen; parent-only winner suites remain untouched."
            )
        )
    report["cost"] = _cost(started_wall, started_cpu, correctness)
    report["artifact_identities"] = {
        "seed_manifest": seed_manifest_sha,
        "preregistration": preregistration_sha,
        "model": model_sha,
        "heldout_freeze": freeze_sha,
        "calibration_branches": branch_sha,
        "dagger_branches": dagger_sha,
    }
    _require_identity(source, args.config)
    _require_artifact_hashes(
        output,
        {
            "seed-manifest.json": seed_manifest_sha,
            "pre-registration.json": preregistration_sha,
            "world-model.json": model_sha,
            "heldout-freeze.json": freeze_sha,
            "barrier-calibration-branches.jsonl": branch_sha,
            "dagger-train-branches.jsonl": dagger_sha,
        },
    )
    finalize_artifacts(output, report)
    _progress("complete", verdict=report["verdict"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if sorted(os.sched_getaffinity(0)) != [LOGICAL_CPU]:
        parser.error(
            f"heavy Strategy C campaign must be pinned to logical CPU {LOGICAL_CPU}"
        )
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            parser.error(f"{name}=1 is required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    campaign(args)


if __name__ == "__main__":
    main()
