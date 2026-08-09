#!/usr/bin/env python3
"""Locked development driver for R3G Strategy A.

Stages are immutable and successive: boundary-audit, preregister, correctness,
sentinel, teacher-screen, calibrate, dagger, freeze-final, barrier, and
student-screen. Winner-only confirmation manifests are intentionally absent;
the locked protocol reserves them for the parent tournament.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from irisu_env import Action, ActionKind, EventKind, IrisuEnv
from irisu_pointer.joint_planner import JointPlannerConfig
from irisu_pointer.solvency_shield import (
    AnalyticSolvencyTeacherPolicy,
    CandidateLocalSolvencySearch,
    FrozenPolicyState,
    LearnedScoreResidualPolicy,
    ResidualSelection,
    SCORE_RESIDUAL_FEATURES,
    ScoreResidualModel,
    SolvencyBarrierConfig,
    SolvencySearchResult,
    score_residual_features,
    visible_liability_ids,
)
from irisu_pointer.steering import SteeringDecision
from irisu_pointer.steering_checkpoint import load_steering_checkpoint
from irisu_pointer.steering_learning import GoalConditionedSteeringPolicy
from irisu_rl.runtime_identity import attest_simulator_runtime


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "artifacts/r3/development/r3g-analytic-solvency-shield-20260729"
)
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
JOINT_SOURCE_SHA256 = (
    "dc7009fc18a322eca5dace55b9baf982b6ced26c18517af752aab0f6365d362e"
)
BASE_REVISION = "de701b36355d5ec582df30f4223aabde7bc537df"
SEED_ROOT = "r3g-solvency-barrier-tournament-20260729"
MAX_DECISIONS = 2_000_000

JOINT_CONFIG = JointPlannerConfig(
    pair_cap=3,
    geometry_cap=5,
    horizons=(48, 160),
    cooldown_ticks=16,
    velocity_lead_ticks=1.0,
    ticks_per_second=50.0,
    require_pristine_source=True,
)
BARRIER_CONFIG = SolvencyBarrierConfig(
    required_renewal_epochs=2,
    branch_tick_cap=1_024,
    query_interval_ticks=256,
    terminal_floor=1,
)
RIDGE = 1e-3
CONFORMAL_ALPHA = 0.05
STRESS_LOW_GAUGE_FRACTION = 0.25
STRESS_MIN_LEVEL = 10
STRESS_MIN_LIABILITIES = 2

SPLITS: dict[str, tuple[int, int, str]] = {
    "teacher-screen": (8, 2_000, "resource-only analytic teacher screen"),
    "barrier-calibration": (
        64,
        2_000,
        "whole-seed branch fitting and conformal calibration",
    ),
    "dagger-train": (16, 2_000, "on-policy residual querying only"),
    "barrier-heldout": (
        64,
        2_000,
        "whole-seed final-student false-safe and coverage evaluation",
    ),
    "barrier-stress": (
        64,
        20_000,
        "targeted low-gauge/high-level/high-rot-debt unsafe proposals",
    ),
    "student-screen": (8, 10_000, "resource-only teacher-free screen"),
}

SENTINELS = (
    {
        "seed": 0x4FDC0458,
        "horizon": 2_000,
        "expected": {
            "survival_ticks": 1_845,
            "final_score": 36,
            "final_gauge": 1,
            "final_level": 1,
            "qualifying_clears": 2,
            "cleared_events": 3,
            "rotten_events": 2,
            "positive_gauge_renewal": 2_228,
            "terminated": True,
            "truncated": False,
        },
    },
    {
        "seed": 0x75FC6DD0,
        "horizon": 2_000,
        "expected": {
            "survival_ticks": 2_000,
            "final_score": 80,
            "final_gauge": 4_780,
            "final_level": 1,
            "qualifying_clears": 6,
            "cleared_events": 8,
            "rotten_events": 1,
            "positive_gauge_renewal": 5_600,
            "terminated": False,
            "truncated": True,
        },
    },
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=lambda item: item.item()
        if callable(getattr(item, "item", None))
        else str(item),
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> str:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                value,
                stream,
                sort_keys=True,
                indent=2,
                allow_nan=False,
                default=lambda item: item.item()
                if callable(getattr(item, "item", None))
                else str(item),
            )
            stream.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _file_sha256(path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _derive_seed(split: str, index: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{SEED_ROOT}|{split}|{index}".encode()).digest()[:4],
        "big",
    )


def _seed_manifests() -> dict[str, object]:
    manifests: dict[str, object] = {}
    all_seeds: list[int] = []
    for split, (count, horizon, purpose) in SPLITS.items():
        rows = [
            {
                "split": split,
                "index": index,
                "seed": _derive_seed(split, index),
                "seed_hex": f"{_derive_seed(split, index):08X}",
                "horizon": horizon,
                "purpose": purpose,
            }
            for index in range(count)
        ]
        manifest = {
            "schema": "r3g-materialized-seed-manifest-v1",
            "derivation": (
                'uint32_be(SHA256("r3g-solvency-barrier-tournament-'
                '20260729|S|i")[0:4])'
            ),
            "rows": rows,
        }
        manifests[split] = {
            **manifest,
            "manifest_sha256": _canonical_sha256(manifest),
        }
        all_seeds.extend(row["seed"] for row in rows)
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("R3G split seeds overlap")
    return manifests


SOURCE_FILES = tuple(
    ROOT / path
    for path in (
        "benchmarks/rl_r3g_analytic_solvency.py",
        "python/irisu_env/__init__.py",
        "python/irisu_env/env.py",
        "python/irisu_env/exact_ipc.py",
        "python/irisu_env/mechanics.py",
        "python/irisu_env/native.py",
        "python/irisu_env/padded.py",
        "python/irisu_env/policies.py",
        "python/irisu_env/randomization.py",
        "python/irisu_env/render.py",
        "python/irisu_env/transfer.py",
        "python/irisu_env/vector.py",
        "python/irisu_pointer/__init__.py",
        "python/irisu_pointer/action.py",
        "python/irisu_pointer/experts.py",
        "python/irisu_pointer/joint_planner.py",
        "python/irisu_pointer/policy.py",
        "python/irisu_pointer/replay_supervision.py",
        "python/irisu_pointer/solvency_shield.py",
        "python/irisu_pointer/steering.py",
        "python/irisu_pointer/steering_checkpoint.py",
        "python/irisu_pointer/steering_learning.py",
        "python/irisu_pointer/steering_progress.py",
        "python/irisu_rl/__init__.py",
        "python/irisu_rl/actions.py",
        "python/irisu_rl/encoding.py",
        "python/irisu_rl/runtime_identity.py",
        "python/irisu_rl/schema.py",
        "tests/test_r3g_solvency_shield.py",
        "pyproject.toml",
    )
)


def _source_identity() -> dict[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    if revision != BASE_REVISION:
        raise RuntimeError("repository baseline identity changed")
    manifest = {
        "schema": "r3g-analytic-solvency-source-v1",
        "git_revision": revision,
        "files": {
            str(path.relative_to(ROOT)): _file_sha256(path)
            for path in SOURCE_FILES
        },
    }
    if (
        manifest["files"]["python/irisu_pointer/joint_planner.py"]
        != JOINT_SOURCE_SHA256
    ):
        raise RuntimeError("corrected joint-v2 reconstruction changed")
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _assert_trusted_inputs() -> None:
    expected = (
        (PROTOCOL, PROTOCOL_SHA256),
        (RUNTIME, RUNTIME_SHA256),
        (CHECKPOINT, CHECKPOINT_SHA256),
    )
    for path, digest in expected:
        if _file_sha256(path) != digest:
            raise RuntimeError(f"trusted input identity changed: {path}")


def _preregistered_config() -> dict[str, object]:
    source = _source_identity()
    seeds = _seed_manifests()
    per_episode_queries = {
        split: (
            1
            if split == "barrier-stress"
            else math.ceil(horizon / BARRIER_CONFIG.query_interval_ticks)
        )
        for split, (_count, horizon, _purpose) in SPLITS.items()
    }
    per_split_queries = {
        split: SPLITS[split][0] * count
        for split, count in per_episode_queries.items()
    }
    per_split_exact_ticks = {
        split: count
        * JOINT_CONFIG.branch_cap
        * BARRIER_CONFIG.branch_tick_cap
        for split, count in per_split_queries.items()
    }
    maximum_training_rows = (
        64 * math.ceil(2_000 / BARRIER_CONFIG.query_interval_ticks) * 15
        + 16 * math.ceil(2_000 / BARRIER_CONFIG.query_interval_ticks) * 15
    )
    value = {
        "schema": "r3g-analytic-solvency-preregistration-v1",
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
        "protocol": {"path": str(PROTOCOL), "sha256": PROTOCOL_SHA256},
        "source_identity": source,
        "runtime": {"path": str(RUNTIME), "sha256": RUNTIME_SHA256},
        "checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": CHECKPOINT_SHA256,
        },
        "candidate_generator": {
            "source_sha256": JOINT_SOURCE_SHA256,
            "config": JOINT_CONFIG.manifest(),
            "config_sha256": JOINT_CONFIG.sha256,
            "candidate_cap": JOINT_CONFIG.branch_cap,
        },
        "barrier": {
            "config": BARRIER_CONFIG.manifest(),
            "config_sha256": BARRIER_CONFIG.sha256,
            "predicted_delta_b2": "exact analytic delta_B2",
            "conformal_alpha": CONFORMAL_ALPHA,
            "calibration_estimator": (
                "whole-seed maximum overprediction residual; "
                "ceil((n+1)*(1-alpha)) order statistic"
            ),
        },
        "query_budget": {
            "schedule": {
                "default": "first v5 shot at/after each 256-tick epoch",
                "barrier-stress": (
                    "first target-qualified v5 shot only"
                ),
            },
            "maximum_queries_per_episode": per_episode_queries,
            "maximum_queries_per_split": per_split_queries,
            "maximum_candidates_per_query": JOINT_CONFIG.branch_cap,
            "maximum_branch_ticks_per_candidate": (
                BARRIER_CONFIG.branch_tick_cap
            ),
            "maximum_exact_simulator_ticks_per_split": per_split_exact_ticks,
            "maximum_exact_simulator_ticks_all_splits": sum(
                per_split_exact_ticks.values()
            ),
            "no_wall_clock_decision_rule": True,
        },
        "training": {
            "model": "deterministic standardized ridge score residual",
            "ridge": RIDGE,
            "feature_names": list(SCORE_RESIDUAL_FEATURES),
            "maximum_candidate_rows": maximum_training_rows,
            "fitting_splits": ["barrier-calibration", "dagger-train"],
            "unsafe_rows_persisted_but_not_fit": True,
        },
        "paired_resampling": {
            "replicates": 10_000,
            "rng_seed": (
                'uint64_be(SHA256("r3g-solvency-barrier-tournament-'
                '20260729|bootstrap|<metric>")[0:8])'
            ),
            "unit": "paired whole seed",
            "one_sided_lower_quantile": 0.05,
            "statistics": ["median score delta", "RMST delta"],
            "confirmation_execution_reserved_for_parent": True,
        },
        "barrier_gates": {
            "minimum_cohort_episodes": 59,
            "maximum_false_safe_cp95": 0.05,
            "minimum_clustered_coverage_lcb95": 0.05,
            "legacy_nonvacuity": {
                "minimum_unsafe_outcomes": 32,
                "minimum_unsafe_states": 8,
                "minimum_unsafe_seeds": 4,
                "minimum_hard_unsafe_outcomes": 8,
            },
            "stress_target": {
                "gauge_fraction_at_most": STRESS_LOW_GAUGE_FRACTION,
                "level_at_least": STRESS_MIN_LEVEL,
                "visible_rot_liabilities_at_least": STRESS_MIN_LIABILITIES,
                "require_exact_unsafe_unshielded_proposal_each_episode": True,
            },
        },
        "seed_manifests": seeds,
        "threading": {
            "logical_cores": 1,
            "torch_threads": 1,
            "nested_blas_openmp_threads": 1,
        },
        "stages": [
            "boundary-audit",
            "preregister",
            "correctness",
            "sentinel",
            "teacher-screen",
            "calibrate",
            "dagger",
            "freeze-final",
            "barrier",
            "student-screen",
        ],
        "winner_confirmation_materialized": False,
    }
    return {**value, "config_sha256": _canonical_sha256(value)}


def _require_frozen() -> dict[str, Any]:
    _assert_trusted_inputs()
    recorded = _load_json(OUTPUT / "preregistered-config.json")
    current = _preregistered_config()
    if recorded != current:
        raise RuntimeError("R3G preregistered source/config identity changed")
    return recorded


def _boundary_audit_payload() -> dict[str, object]:
    relative = [str(path.relative_to(ROOT)) for path in SOURCE_FILES]
    forbidden_components = {"sealed", "authorization", "runs"}
    if (
        len(relative) != len(set(relative))
        or any(path.startswith("artifacts/") for path in relative)
        or any(
            forbidden_components.intersection(Path(path).parts)
            for path in relative
        )
    ):
        raise RuntimeError("source allowlist crosses a forbidden boundary")
    source = _source_identity()
    value = {
        "schema": "r3g-development-boundary-audit-v1",
        "development_only": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "passed": True,
        "review_basis": (
            "current task tool-call record and subagent read reports"
        ),
        "explicit_transitive_source_allowlist": relative,
        "source_identity_sha256": source["sha256"],
        "forbidden_evidence_paths_read": [],
        "verified_not_read": [
            "artifacts/r3/runs",
            "sealed evidence",
            "test evidence",
            "authorization artifacts",
        ],
        "allowed_unit_test_source": "tests/test_r3g_solvency_shield.py",
        "disclosed_incident": (
            "Before preregistration, an over-broad package source hash read "
            "canonical-named .py source modules in this isolated development "
            "worktree. No canonical-run result, sealed/test evidence, or "
            "authorization artifact was accessed or surfaced."
        ),
        "parent_ruling": (
            "hash-only canonical-named development source access is "
            "non-disqualifying; explicit transitive allowlist required"
        ),
        "split_materialized": False,
    }
    return {**value, "audit_sha256": _canonical_sha256(value)}


def _require_boundary_audit() -> dict[str, Any]:
    recorded = _load_json(OUTPUT / "boundary-audit.json")
    current = _boundary_audit_payload()
    if recorded != current or recorded.get("passed") is not True:
        raise RuntimeError("development boundary audit changed or failed")
    return recorded


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


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    array = np.asarray([int(value) for value in values], dtype=np.float64)
    return {
        "minimum": int(array.min()),
        "p10": float(np.quantile(array, 0.1, method="linear")),
        "median": float(np.median(array)),
        "maximum": int(array.max()),
    }


def _threading_manifest() -> dict[str, object]:
    return {
        "cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
    }


def _base_policy(model: Any) -> GoalConditionedSteeringPolicy:
    return GoalConditionedSteeringPolicy(
        model,
        cooldown_ticks=16,
        minimum_pair_closure_sizes=0.05,
        impact_side_sizes=0.5,
        impact_below_sizes=0.75,
        source_velocity_lead_ticks=1.0,
        ticks_per_second=50.0,
        act_logit_bias=1.0,
        artifact_sha256=CHECKPOINT_SHA256,
    )


def _searcher(
    model: Any,
    *,
    conformal_q: float = 0.0,
) -> CandidateLocalSolvencySearch:
    return CandidateLocalSolvencySearch(
        lambda: _base_policy(model),
        joint_config=JOINT_CONFIG,
        barrier_config=BARRIER_CONFIG,
        continuation_identity_sha256=CHECKPOINT_SHA256,
        conformal_q=conformal_q,
    )


def _policy_results(policy: object) -> tuple[object, ...]:
    values = getattr(policy, "results", ())
    return tuple(values)


def _policy_executed_ordinals(policy: object) -> tuple[int, ...]:
    values = tuple(int(value) for value in getattr(policy, "executed_ordinals", ()))
    results = _policy_results(policy)
    if results and len(values) != len(results):
        raise RuntimeError("policy result/execution accounting is misaligned")
    return values


def _result_manifest(value: object) -> Mapping[str, object]:
    manifest = getattr(value, "manifest", None)
    if not callable(manifest):
        raise TypeError("policy result lacks a manifest")
    return manifest()


def _search_from_result(value: object) -> SolvencySearchResult:
    return value.search if isinstance(value, ResidualSelection) else value


def _exact_unsafe(
    search: SolvencySearchResult,
    ordinal: int,
) -> bool:
    outcome = search.outcome_for(ordinal)
    certificate = search.certificate_for(ordinal)
    exact_terminal = (
        outcome.game_over or outcome.terminated or outcome.final_gauge <= 0
    )
    exact_negative_solvency = (
        outcome.resolved_two_renewals
        and outcome.error is None
        and (
            outcome.minimum_margin < 0
            or certificate.exact_delta_b2 < 0
        )
    )
    return exact_terminal or exact_negative_solvency


def _executed_override_unsafe(
    search: SolvencySearchResult,
    ordinal: int,
) -> bool:
    return (
        not search.certificate_for(ordinal).eligible
        or not search.outcome_for(ordinal).hard_valid
        or _exact_unsafe(search, ordinal)
    )


def _stress_target(search: SolvencySearchResult) -> bool:
    incumbent = search.incumbent
    return (
        incumbent.initial_gauge
        / max(incumbent.gauge_max, 1)
        <= STRESS_LOW_GAUGE_FRACTION
        and incumbent.initial_level >= STRESS_MIN_LEVEL
        and incumbent.initial_liabilities >= STRESS_MIN_LIABILITIES
    )


def _stress_observation(observation: Mapping[str, Any]) -> bool:
    return (
        int(observation.get("gauge", 0))
        / max(int(observation.get("gauge_max", 1)), 1)
        <= STRESS_LOW_GAUGE_FRACTION
        and int(observation.get("level", 0)) >= STRESS_MIN_LEVEL
        and len(visible_liability_ids(observation))
        >= STRESS_MIN_LIABILITIES
    )


def _run_episode(
    *,
    seed: int,
    horizon: int,
    label: str,
    factory: Callable[[IrisuEnv], object],
    maximum_query_budget: int | None = None,
) -> tuple[dict[str, object], Mapping[str, object], Mapping[str, object], tuple[object, ...]]:
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    checkpoints = tuple(range(500, horizon + 1, 500))
    captures: dict[str, Mapping[str, object]] = {}
    counts: Counter[int] = Counter()
    positive_gauge = decisions = primitives = 0
    game_over = False
    with IrisuEnv(
        library_path=RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": horizon},
    ) as env:
        runner = env.runner_identity_manifest()
        attestation = attest_simulator_runtime(env).manifest()
        observation, reset_info = env.reset(seed=seed)
        config_hash = int(runner["config_hash"])
        if int(reset_info.get("config_hash", -1)) != config_hash:
            raise RuntimeError("R3G reset config identity mismatch")
        start_tick = int(observation.get("tick", 0))
        start_clears = int(observation.get("qualifying_clear_count", 0))
        policy = factory(env)
        getattr(policy, "reset")(seed)
        terminated = bool(observation.get("terminated", False))
        truncated = bool(observation.get("truncated", False))

        def step(action: Action) -> None:
            nonlocal observation, terminated, truncated, primitives
            nonlocal positive_gauge, game_over
            observation, _reward, terminated, truncated, info = env.step(action)
            primitives += 1
            if int(info.get("config_hash", -1)) != config_hash:
                raise RuntimeError("R3G step config identity mismatch")
            for event in info.get("events", ()):
                if not isinstance(event, Mapping):
                    continue
                kind = _event_kind(event)
                if kind is None:
                    continue
                counts[kind] += 1
                if kind == int(EventKind.GAUGE_CHANGED):
                    positive_gauge += max(0, int(event.get("value", 0)))
                if kind == int(EventKind.GAME_OVER):
                    game_over = True
            tick = int(observation.get("tick", 0))
            if tick in checkpoints:
                captures[str(tick)] = {
                    "tick": tick,
                    "reached": True,
                    "score": int(observation.get("score", 0)),
                    "gauge": int(observation.get("gauge", 0)),
                    "level": int(observation.get("level", 0)),
                    "qualifying_clears": int(
                        observation.get("qualifying_clear_count", 0)
                    )
                    - start_clears,
                    "rotten_events": counts[int(EventKind.ROTTEN)],
                    "positive_gauge_renewal": positive_gauge,
                }

        while not terminated and not truncated:
            if decisions >= MAX_DECISIONS:
                raise RuntimeError("R3G episode exceeded decision budget")
            decision = getattr(policy, "predict")(observation)
            if not isinstance(decision, SteeringDecision):
                raise TypeError("R3G policy returned a non-decision")
            decisions += 1
            for action in decision.primitive_actions():
                if terminated or truncated:
                    break
                if ActionKind.parse(action.kind) is not ActionKind.WAIT:
                    step(action)
                    continue
                remaining = int(action.wait_ticks)
                while remaining and not terminated and not truncated:
                    tick = int(observation.get("tick", 0))
                    upcoming = [value for value in checkpoints if value > tick]
                    duration = min(
                        remaining,
                        upcoming[0] - tick if upcoming else remaining,
                    )
                    step(Action.wait(duration))
                    remaining -= duration

        survival = int(observation.get("tick", 0)) - start_tick
        for checkpoint in checkpoints:
            if str(checkpoint) not in captures:
                captures[str(checkpoint)] = {
                    "tick": survival,
                    "reached": False,
                    "terminal_carried": True,
                    "score": int(observation.get("score", 0)),
                    "gauge": int(observation.get("gauge", 0)),
                    "level": int(observation.get("level", 0)),
                    "qualifying_clears": int(
                        observation.get("qualifying_clear_count", 0)
                    )
                    - start_clears,
                    "rotten_events": counts[int(EventKind.ROTTEN)],
                    "positive_gauge_renewal": positive_gauge,
                }
        results = _policy_results(policy)
        executed_ordinals = _policy_executed_ordinals(policy)
        statistics = getattr(policy, "statistics", None)
        policy_counts = statistics() if callable(statistics) else {}
        maximum_queries = (
            math.ceil(horizon / BARRIER_CONFIG.query_interval_ticks)
            if maximum_query_budget is None
            else maximum_query_budget
        )
        if (
            int(policy_counts.get("queries", 0)) > maximum_queries
            or int(policy_counts.get("candidates", 0))
            > maximum_queries * JOINT_CONFIG.branch_cap
            or int(policy_counts.get("simulated_ticks", 0))
            > maximum_queries
            * JOINT_CONFIG.branch_cap
            * BARRIER_CONFIG.branch_tick_cap
        ):
            raise RuntimeError("R3G episode exceeded its preregistered budget")
        unsafe_execution = False
        for result, ordinal in zip(
            results, executed_ordinals, strict=True
        ):
            if ordinal:
                unsafe_execution |= _executed_override_unsafe(
                    _search_from_result(result), ordinal
                )
        episode = {
            "policy": label,
            "seed": seed,
            "seed_hex": f"{seed:08X}",
            "horizon_ticks": horizon,
            "survival_ticks": survival,
            "final_score": int(observation.get("score", 0)),
            "final_gauge": int(observation.get("gauge", 0)),
            "gauge_max": int(observation.get("gauge_max", 0)),
            "final_level": int(observation.get("level", 0)),
            "highest_chain": int(observation.get("highest_chain", 0)),
            "qualifying_clears": int(
                observation.get("qualifying_clear_count", 0)
            )
            - start_clears,
            "cleared_events": counts[int(EventKind.CLEARED)],
            "rotten_events": counts[int(EventKind.ROTTEN)],
            "positive_gauge_renewal": positive_gauge,
            "invalid_actions": counts[int(EventKind.INVALID_ACTION)],
            "decisions": decisions,
            "primitive_actions": primitives,
            "terminated": terminated,
            "truncated": truncated,
            "game_over": game_over,
            "gauge_failure": game_over,
            "unsafe_executed_override": unsafe_execution,
            "checkpoints": captures,
            "policy_counts": policy_counts,
            "query_result_sha256s": [
                getattr(value, "sha256") for value in results
            ],
            "query_executed_ordinals": list(executed_ordinals),
            "query_results": [_result_manifest(value) for value in results],
            "wall_seconds": time.perf_counter() - wall_started,
            "cpu_seconds": time.process_time() - cpu_started,
        }
    return episode, runner, attestation, results


def _evaluate(
    *,
    seeds: Sequence[int],
    horizon: int,
    label: str,
    factory: Callable[[IrisuEnv], object],
    split: str,
) -> tuple[list[dict[str, object]], Mapping[str, object], Mapping[str, object], list[tuple[object, ...]]]:
    episodes = []
    result_sets = []
    maximum_queries = int(
        _preregistered_config()["query_budget"][
            "maximum_queries_per_episode"
        ][split]
    )
    runner: Mapping[str, object] | None = None
    attestation: Mapping[str, object] | None = None
    for index, seed in enumerate(seeds):
        episode, current_runner, current_attestation, results = _run_episode(
            seed=seed,
            horizon=horizon,
            label=label,
            factory=factory,
            maximum_query_budget=maximum_queries,
        )
        if runner is None:
            runner, attestation = current_runner, current_attestation
        elif runner != current_runner or attestation != current_attestation:
            raise RuntimeError("R3G runtime identity changed across episodes")
        episodes.append(episode)
        result_sets.append(results)
        _atomic_json(
            OUTPUT
            / "partial"
            / f"{split}-{label}-{index:03d}-{seed:08X}.json",
            {
                "schema": "r3g-completed-episode-evidence-v1",
                "protocol_sha256": PROTOCOL_SHA256,
                "split": split,
                "index": index,
                "episode": episode,
            },
        )
    assert runner is not None and attestation is not None
    maximum = _preregistered_config()["query_budget"]
    if (
        sum(
            int(value["policy_counts"].get("queries", 0))
            for value in episodes
        )
        > int(maximum["maximum_queries_per_split"][split])
        or sum(
            int(value["policy_counts"].get("simulated_ticks", 0))
            for value in episodes
        )
        > int(maximum["maximum_exact_simulator_ticks_per_split"][split])
    ):
        raise RuntimeError(f"{split} exceeded its preregistered total budget")
    return episodes, runner, attestation, result_sets


def _aggregate(episodes: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    return {
        "episodes": len(episodes),
        "score": _distribution([value["final_score"] for value in episodes]),
        "survival_ticks": _distribution(
            [value["survival_ticks"] for value in episodes]
        ),
        "final_gauge": _distribution(
            [value["final_gauge"] for value in episodes]
        ),
        "final_level": _distribution(
            [value["final_level"] for value in episodes]
        ),
        "terminals": sum(bool(value["terminated"]) for value in episodes),
        "gauge_failures": sum(bool(value["gauge_failure"]) for value in episodes),
        "qualifying_clears": sum(
            int(value["qualifying_clears"]) for value in episodes
        ),
        "cleared_events": sum(int(value["cleared_events"]) for value in episodes),
        "rotten_events": sum(int(value["rotten_events"]) for value in episodes),
        "positive_gauge_renewal": sum(
            int(value["positive_gauge_renewal"]) for value in episodes
        ),
        "invalid_actions": sum(
            int(value["invalid_actions"]) for value in episodes
        ),
        "decisions": sum(int(value["decisions"]) for value in episodes),
        "queries": sum(
            int(value["policy_counts"].get("queries", 0)) for value in episodes
        ),
        "candidates": sum(
            int(value["policy_counts"].get("candidates", 0))
            for value in episodes
        ),
        "certified_candidates": sum(
            int(value["policy_counts"].get("certified_candidates", 0))
            for value in episodes
        ),
        "overrides": sum(
            int(value["policy_counts"].get("overrides", 0))
            for value in episodes
        ),
        "corrections": sum(
            int(value["policy_counts"].get("overrides", 0))
            for value in episodes
        ),
        "score_proposals": sum(
            int(value["policy_counts"].get("score_proposals", 0))
            for value in episodes
        ),
        "uncertified_score_proposals": sum(
            int(
                value["policy_counts"].get(
                    "uncertified_score_proposals", 0
                )
            )
            for value in episodes
        ),
        "eligible_decision_states": sum(
            int(
                value["policy_counts"].get(
                    "states_with_certified_override", 0
                )
            )
            for value in episodes
        ),
        "abstentions": sum(
            sum(
                int(count)
                for name, count in value["policy_counts"].items()
                if "abstention" in name
            )
            for value in episodes
        ),
        "wall_seconds": sum(float(value["wall_seconds"]) for value in episodes),
        "cpu_seconds": sum(float(value["cpu_seconds"]) for value in episodes),
    }


def _paired(
    baseline: Sequence[Mapping[str, Any]],
    candidate: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    other = {int(value["seed"]): value for value in candidate}
    rows = []
    for base in baseline:
        value = other[int(base["seed"])]
        regression = int(base["survival_ticks"]) - int(value["survival_ticks"])
        material = regression >= 1_000
        catastrophe = (
            material
            and int(value["survival_ticks"]) / max(int(base["survival_ticks"]), 1)
            <= 0.5
        )
        new_terminal = bool(value["terminated"]) and not bool(base["terminated"])
        new_gauge = bool(value["gauge_failure"]) and not bool(
            base["gauge_failure"]
        )
        unsafe = bool(value["unsafe_executed_override"])
        hard = new_terminal or new_gauge or catastrophe or unsafe
        rows.append(
            {
                "seed": int(base["seed"]),
                "seed_hex": base["seed_hex"],
                "baseline_score": int(base["final_score"]),
                "candidate_score": int(value["final_score"]),
                "score_delta": int(value["final_score"])
                - int(base["final_score"]),
                "baseline_survival": int(base["survival_ticks"]),
                "candidate_survival": int(value["survival_ticks"]),
                "survival_delta": int(value["survival_ticks"])
                - int(base["survival_ticks"]),
                "material_survival_regression": material,
                "catastrophic_regression": catastrophe,
                "new_terminal": new_terminal,
                "new_gauge_failure": new_gauge,
                "unsafe_executed_override": unsafe,
                "hard_failure": hard,
            }
        )
    return {
        "definition": {
            "material": "base-candidate >= 1000 ticks",
            "catastrophic": "material and candidate/base <= 0.5",
            "hard": (
                "new terminal OR new gauge OR catastrophic OR exact-unsafe "
                "executed override"
            ),
        },
        "rows": rows,
        "hard_failures": [value for value in rows if value["hard_failure"]],
        "material_regressions": [
            value for value in rows if value["material_survival_regression"]
        ],
        "score_wins": sum(value["score_delta"] > 0 for value in rows),
        "survival_wins": sum(value["survival_delta"] > 0 for value in rows),
        "survival_regressions": sum(value["survival_delta"] < 0 for value in rows),
    }


def _split_rows(split: str) -> tuple[list[int], int, Mapping[str, Any]]:
    preregistered = _require_frozen()
    manifest = preregistered["seed_manifests"][split]
    expected = _seed_manifests()[split]
    if manifest != expected:
        raise RuntimeError(f"{split} seed manifest changed")
    rows = manifest["rows"]
    return [int(value["seed"]) for value in rows], int(rows[0]["horizon"]), manifest


def _training_rows(
    result_sets: Sequence[Sequence[object]],
) -> tuple[list[list[float]], list[float], list[dict[str, object]]]:
    rows: list[list[float]] = []
    labels: list[float] = []
    identities: list[dict[str, object]] = []
    for results in result_sets:
        for result in results:
            search = _search_from_result(result)
            incumbent_score = search.incumbent.score_gain
            for outcome, certificate in search.ordered_pairs():
                if outcome.candidate.ordinal == 0 or not certificate.eligible:
                    continue
                features = score_residual_features(outcome, certificate)
                label = float(outcome.score_gain - incumbent_score)
                rows.append([float(value) for value in features])
                labels.append(label)
                identities.append(
                    {
                        "search_sha256": search.sha256,
                        "outcome_sha256": outcome.sha256,
                        "ordinal": outcome.candidate.ordinal,
                        "label": label,
                    }
                )
    return rows, labels, identities


def _model_payload(
    model: ScoreResidualModel,
    *,
    source_identity_sha256: str,
    training_manifest: Mapping[str, object],
) -> dict[str, object]:
    value = {
        "schema": "r3g-score-residual-artifact-v1",
        "development_only": True,
        "source_identity_sha256": source_identity_sha256,
        "training_manifest": training_manifest,
        "model": model.manifest(),
        "model_sha256": model.sha256,
    }
    return {**value, "payload_sha256": _canonical_sha256(value)}


def _validated_training_manifest(
    value: Mapping[str, Any],
    expected_training_splits: Sequence[str],
) -> dict[str, Any]:
    manifest = dict(value)
    recorded_sha = manifest.pop("sha256", None)
    expected_seeds = _seed_manifests()
    splits = list(expected_training_splits)
    if (
        recorded_sha != _canonical_sha256(manifest)
        or value.get("splits") != splits
        or int(value.get("rows", -1)) != len(value.get("row_identities", ()))
    ):
        raise RuntimeError("score training manifest identity is invalid")
    expected_hashes = [
        expected_seeds[split]["manifest_sha256"] for split in splits
    ]
    recorded_hashes = (
        [value.get("seed_manifest_sha256")]
        if len(splits) == 1
        else list(value.get("seed_manifest_sha256s", ()))
    )
    if recorded_hashes != expected_hashes:
        raise RuntimeError("score training seed identity is invalid")
    return dict(value)


def _load_model(path: Path, expected_training_splits: Sequence[str]) -> ScoreResidualModel:
    preregistered = _require_frozen()
    payload = _load_json(path)
    value = dict(payload)
    recorded_sha = value.pop("payload_sha256", None)
    training_manifest = _validated_training_manifest(
        payload.get("training_manifest", {}),
        expected_training_splits,
    )
    if (
        recorded_sha != _canonical_sha256(value)
        or payload.get("source_identity_sha256")
        != preregistered["source_identity"]["sha256"]
    ):
        raise RuntimeError("score-residual artifact chain is invalid")
    model = ScoreResidualModel.from_manifest(payload["model"])
    if (
        model.sha256 != payload.get("model_sha256")
        or model.training_manifest_sha256
        != training_manifest["sha256"]
        or model.training_rows != int(training_manifest["rows"])
    ):
        raise RuntimeError("score-residual model hash changed")
    return model


def _validated_stage_report(
    path: Path,
    *,
    schema: str,
    split: str | None = None,
) -> dict[str, Any]:
    preregistered = _require_frozen()
    report = _load_json(path)
    if (
        report.get("schema") != schema
        or report.get("protocol_sha256") != PROTOCOL_SHA256
        or report.get("source_identity_sha256")
        != preregistered["source_identity"]["sha256"]
    ):
        raise RuntimeError(f"invalid predecessor report: {path}")
    if split is not None:
        expected = _seed_manifests()[split]
        if report.get("seed_manifest") != expected:
            raise RuntimeError(f"invalid predecessor seed identity: {path}")
    return report


def _verify_reported_model(
    report: Mapping[str, Any],
    *,
    path: Path,
    model: ScoreResidualModel,
) -> None:
    recorded = report.get("model", {})
    if (
        recorded.get("path") != str(path)
        or recorded.get("file_sha256") != _file_sha256(path)
        or recorded.get("model_sha256") != model.sha256
    ):
        raise RuntimeError(f"reported score model identity is invalid: {path}")


def _final_frozen_identity() -> dict[str, object]:
    preregistered = _require_frozen()
    calibration_path = OUTPUT / "barrier-calibration.json"
    dagger_path = OUTPUT / "dagger-train.json"
    initial_path = OUTPUT / "score-residual-initial.json"
    final_path = OUTPUT / "score-residual-final.json"
    calibration = _validated_stage_report(
        calibration_path,
        schema="r3g-barrier-calibration-report-v1",
        split="barrier-calibration",
    )
    dagger_report = _validated_stage_report(
        dagger_path,
        schema="r3g-score-residual-dagger-report-v1",
        split="dagger-train",
    )
    initial_model = _load_model(initial_path, ["barrier-calibration"])
    final_model = _load_model(
        final_path,
        ["barrier-calibration", "dagger-train"],
    )
    _verify_reported_model(
        calibration, path=initial_path, model=initial_model
    )
    _verify_reported_model(
        dagger_report, path=final_path, model=final_model
    )
    final_payload = _load_json(final_path)
    training_manifest = _validated_training_manifest(
        final_payload["training_manifest"],
        ["barrier-calibration", "dagger-train"],
    )
    q = float(calibration["conformal"]["q"])
    if not math.isfinite(q) or q < 0.0:
        raise RuntimeError("calibration threshold is invalid")
    value = {
        "schema": "r3g-final-student-frozen-identity-v1",
        "development_only": True,
        "frozen_before_heldout": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": preregistered["source_identity"]["sha256"],
        "preregistered_config_sha256": preregistered["config_sha256"],
        "candidate_generator": preregistered["candidate_generator"],
        "barrier_config": preregistered["barrier"],
        "query_and_exact_tick_budget": preregistered["query_budget"],
        "all_seed_manifest_sha256s": {
            split: manifest["manifest_sha256"]
            for split, manifest in preregistered["seed_manifests"].items()
        },
        "calibration": {
            "q": q,
            "report_path": str(calibration_path),
            "report_sha256": _file_sha256(calibration_path),
        },
        "training_manifest": training_manifest,
        "training_manifest_sha256": training_manifest["sha256"],
        "dagger_report": {
            "path": str(dagger_path),
            "sha256": _file_sha256(dagger_path),
        },
        "final_model": {
            "path": str(final_path),
            "file_sha256": _file_sha256(final_path),
            "model_sha256": final_model.sha256,
        },
        "runtime_sha256": RUNTIME_SHA256,
        "checkpoint_sha256": CHECKPOINT_SHA256,
    }
    return {**value, "identity_sha256": _canonical_sha256(value)}


def _require_final_frozen() -> dict[str, Any]:
    recorded = _load_json(OUTPUT / "frozen-final-student.json")
    current = _final_frozen_identity()
    if recorded != current:
        raise RuntimeError("final student identity changed after freeze")
    return recorded


def _conformal_q(
    episode_results: Sequence[Sequence[object]],
    *,
    alpha: float,
) -> tuple[float, list[float], list[int], int]:
    residuals = []
    exact_label_counts = []
    for results in episode_results:
        maximum = -math.inf
        exact_labels = 0
        for result in results:
            search = _search_from_result(result)
            for outcome, certificate in search.ordered_pairs():
                if (
                    not outcome.resolved_two_renewals
                    or outcome.error is not None
                ):
                    continue
                # Strategy A's analytic prediction is the exact replayed delta.
                predicted = float(certificate.exact_delta_b2)
                maximum = max(
                    maximum, predicted - float(certificate.exact_delta_b2)
                )
                exact_labels += 1
        if maximum == -math.inf:
            raise RuntimeError(
                "conformal whole-seed cluster has no exact two-renewal label"
            )
        residuals.append(maximum)
        exact_label_counts.append(exact_labels)
    rank = math.ceil((len(residuals) + 1) * (1.0 - alpha))
    rank = min(max(rank, 1), len(residuals))
    return (
        sorted(residuals)[rank - 1],
        residuals,
        exact_label_counts,
        rank,
    )


def _clopper_pearson_upper(
    failures: int,
    episodes: int,
    alpha: float = 0.05,
) -> float:
    if episodes < 1 or failures < 0 or failures > episodes:
        return 1.0
    if failures == episodes:
        return 1.0
    if failures == 0:
        return 1.0 - alpha ** (1.0 / episodes)

    def cdf(probability: float) -> float:
        return sum(
            math.comb(episodes, index)
            * probability**index
            * (1.0 - probability) ** (episodes - index)
            for index in range(failures + 1)
        )

    lower, upper = 0.0, 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if cdf(midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


def _coverage_lcb(
    episodes: Sequence[Mapping[str, Any]],
    *,
    replicates: int = 10_000,
) -> float:
    eligible_states = np.asarray(
        [
            int(
                value["policy_counts"].get(
                    "states_with_certified_override", 0
                )
            )
            for value in episodes
        ],
        dtype=np.int64,
    )
    overrides = np.asarray(
        [int(value["policy_counts"].get("overrides", 0)) for value in episodes],
        dtype=np.int64,
    )
    seed = int.from_bytes(
        hashlib.sha256(f"{SEED_ROOT}|bootstrap|barrier-coverage".encode()).digest()[
            :8
        ],
        "big",
    )
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = generator.integers(0, len(episodes), len(episodes))
        denominator = int(eligible_states[selected].sum())
        estimates[index] = (
            float(overrides[selected].sum()) / denominator
            if denominator
            else 0.0
        )
    return float(np.quantile(estimates, 0.05, method="linear"))


def _risk_deciles(
    result_sets: Sequence[Sequence[object]],
) -> list[dict[str, object]]:
    rows = []
    for results in result_sets:
        for result in results:
            search = _search_from_result(result)
            for ordinal in range(1, len(search.outcomes)):
                outcome = search.outcome_for(ordinal)
                certificate = search.certificate_for(ordinal)
                if (
                    not outcome.resolved_two_renewals
                    or outcome.error is not None
                ):
                    continue
                rows.append(
                    {
                        "risk": -certificate.lower_bound_delta_b2,
                        "exact_delta_b2": certificate.exact_delta_b2,
                        "minimum_margin": outcome.minimum_margin,
                        "certified": certificate.eligible,
                        "unsafe": _exact_unsafe(search, ordinal),
                    }
                )
    rows.sort(key=lambda value: value["risk"])
    if not rows:
        return []
    buckets = np.array_split(
        np.arange(len(rows)),
        min(10, len(rows)),
    )
    output = []
    for ordinal, indices in enumerate(buckets):
        selected = [rows[int(index)] for index in indices]
        output.append(
            {
                "decile": ordinal + 1,
                "count": len(selected),
                "risk_min": min(value["risk"] for value in selected),
                "risk_max": max(value["risk"] for value in selected),
                "mean_exact_delta_b2": float(
                    np.mean([value["exact_delta_b2"] for value in selected])
                ),
                "unsafe_rate": float(
                    np.mean([value["unsafe"] for value in selected])
                ),
                "certified": sum(value["certified"] for value in selected),
            }
        )
    return output


def _barrier_cohort(
    episodes: Sequence[Mapping[str, Any]],
    result_sets: Sequence[Sequence[object]],
    *,
    targeted_stress: bool,
) -> dict[str, object]:
    false_safe_episodes = 0
    unsafe_outcomes = hard_unsafe = unresolved_outcomes = 0
    unsafe_state_ids: set[tuple[int, str, int]] = set()
    target_state_ids: set[tuple[int, str, int]] = set()
    target_seed_ids: set[int] = set()
    unsafe_seed_ids: set[int] = set()
    unsafe_proposal_episodes = 0
    for episode, results in zip(episodes, result_sets, strict=True):
        executed_ordinals = tuple(
            int(value) for value in episode["query_executed_ordinals"]
        )
        if len(executed_ordinals) != len(results):
            raise RuntimeError("barrier execution evidence is misaligned")
        episode_false_safe = False
        episode_unsafe_proposal = False
        for result, executed_ordinal in zip(
            results, executed_ordinals, strict=True
        ):
            if not isinstance(result, ResidualSelection):
                raise TypeError("barrier cohort requires final-student selections")
            search = result.search
            episode_false_safe |= any(
                search.certificate_for(ordinal).eligible
                and _exact_unsafe(search, ordinal)
                for ordinal in range(1, len(search.outcomes))
            )
            if executed_ordinal:
                episode_false_safe |= _executed_override_unsafe(
                    search, executed_ordinal
                )
            proposal_is_targeted = (
                _stress_target(search) if targeted_stress else True
            )
            if targeted_stress and proposal_is_targeted:
                target_state_ids.add(
                    (
                        int(episode["seed"]),
                        search.snapshot_sha256,
                        search.incumbent.start_tick,
                    )
                )
                target_seed_ids.add(int(episode["seed"]))
            if result.proposed_ordinal and proposal_is_targeted:
                episode_unsafe_proposal |= _exact_unsafe(
                    search, result.proposed_ordinal
                )
            state_unsafe = False
            for ordinal in range(1, len(search.outcomes)):
                outcome = search.outcome_for(ordinal)
                if (
                    not outcome.resolved_two_renewals
                    and not (
                        outcome.game_over
                        or outcome.terminated
                        or outcome.final_gauge <= 0
                    )
                ):
                    unresolved_outcomes += 1
                unsafe = _exact_unsafe(search, ordinal)
                if not unsafe:
                    continue
                unsafe_outcomes += 1
                state_unsafe = True
                unsafe_seed_ids.add(int(episode["seed"]))
                hard_unsafe += int(
                    outcome.game_over
                    or outcome.terminated
                    or outcome.final_gauge <= 0
                    or (
                        outcome.resolved_two_renewals
                        and outcome.minimum_margin < 0
                    )
                )
            if state_unsafe:
                unsafe_state_ids.add(
                    (
                        int(episode["seed"]),
                        search.snapshot_sha256,
                        search.incumbent.start_tick,
                    )
                )
        false_safe_episodes += int(episode_false_safe)
        unsafe_proposal_episodes += int(episode_unsafe_proposal)

    eligible_states = sum(
        int(
            value["policy_counts"].get(
                "states_with_certified_override", 0
            )
        )
        for value in episodes
    )
    overrides = sum(
        int(value["policy_counts"].get("overrides", 0)) for value in episodes
    )
    return {
        "episodes": len(episodes),
        "false_safe_episodes": false_safe_episodes,
        "false_safe_rate": false_safe_episodes / len(episodes),
        "false_safe_cp95_upper": _clopper_pearson_upper(
            false_safe_episodes,
            len(episodes),
        ),
        "eligible_on_policy_decision_states": eligible_states,
        "overrides": overrides,
        "certified_override_coverage": (
            overrides / eligible_states if eligible_states else 0.0
        ),
        "seed_clustered_coverage_lcb95": _coverage_lcb(episodes),
        "unsafe_outcomes": unsafe_outcomes,
        "hard_unsafe_outcomes": hard_unsafe,
        "unresolved_outcomes_excluded_from_unsafe": unresolved_outcomes,
        "unsafe_decision_states": len(unsafe_state_ids),
        "unsafe_seeds": len(unsafe_seed_ids),
        "unsafe_proposal_episodes": unsafe_proposal_episodes,
        "target_decision_states": len(target_state_ids),
        "target_episodes": len(target_seed_ids),
        "stress_target": (
            {
                "gauge_fraction_at_most": STRESS_LOW_GAUGE_FRACTION,
                "level_at_least": STRESS_MIN_LEVEL,
                "visible_rot_liabilities_at_least": STRESS_MIN_LIABILITIES,
            }
            if targeted_stress
            else None
        ),
        "risk_deciles": _risk_deciles(result_sets),
    }


def boundary_audit() -> None:
    _assert_trusted_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    value = _boundary_audit_payload()
    path = OUTPUT / "boundary-audit.json"
    digest = _atomic_json(path, value)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": digest,
                "audit_sha256": value["audit_sha256"],
                "passed": value["passed"],
            },
            sort_keys=True,
        )
    )


def preregister() -> None:
    _assert_trusted_inputs()
    _require_boundary_audit()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    value = _preregistered_config()
    path = OUTPUT / "preregistered-config.json"
    digest = _atomic_json(path, value)
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))


def sentinel(model: Any) -> None:
    preregistered = _require_frozen()
    correctness_report = _validated_stage_report(
        OUTPUT / "correctness.json",
        schema="r3g-solvency-correctness-report-v1",
    )
    if correctness_report.get("passed") is not True:
        raise RuntimeError("correctness gate did not pass")
    episodes = []
    mismatches = []
    runner = attestation = None
    for value in SENTINELS:
        episode, current_runner, current_attestation, _ = _run_episode(
            seed=value["seed"],
            horizon=value["horizon"],
            label="frozen_v5_sentinel",
            factory=lambda _env: _base_policy(model),
        )
        observed = {
            key: episode[key] for key in value["expected"]
        }
        if observed != value["expected"]:
            mismatches.append(
                {
                    "seed": value["seed"],
                    "seed_hex": f"{value['seed']:08X}",
                    "expected": value["expected"],
                    "observed": observed,
                }
            )
        if runner is None:
            runner, attestation = current_runner, current_attestation
        elif runner != current_runner or attestation != current_attestation:
            raise RuntimeError("sentinel runtime identity changed")
        episodes.append(episode)
    report = {
        "schema": "r3g-frozen-v5-sentinel-report-v1",
        "development_only": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": preregistered["source_identity"]["sha256"],
        "runtime": {"runner": runner, "attestation": attestation},
        "sentinels": episodes,
        "mismatches": mismatches,
        "execution": {
            "threading": _threading_manifest(),
            "wall_seconds": sum(
                float(value["wall_seconds"]) for value in episodes
            ),
            "cpu_seconds": sum(
                float(value["cpu_seconds"]) for value in episodes
            ),
        },
        "passed": not mismatches and len(episodes) == len(SENTINELS),
    }
    path = OUTPUT / "sentinel.json"
    digest = _atomic_json(path, report)
    print(json.dumps({"path": str(path), "sha256": digest}, sort_keys=True))


def correctness(model: Any) -> None:
    preregistered = _require_frozen()
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    command = [
        sys.executable,
        str(ROOT / "tests/test_r3g_solvency_shield.py"),
        "-v",
    ]
    unit = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT / "python"),
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        },
    )
    unit_evidence = {
        "schema": "r3g-correctness-unit-evidence-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "command": command,
        "returncode": unit.returncode,
        "stdout": unit.stdout,
        "stderr": unit.stderr,
        "source_sha256": _file_sha256(
            ROOT / "tests/test_r3g_solvency_shield.py"
        ),
    }
    unit_path = OUTPUT / "partial/correctness-unit.json"
    _atomic_json(unit_path, unit_evidence)
    with IrisuEnv(
        library_path=RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": 2_000},
    ) as env:
        observation, _ = env.reset(seed=0x442BF8A4)
        base = _base_policy(model)
        base.reset(0x442BF8A4)
        incumbent = base.predict(observation)
        if not incumbent.is_shot:
            raise RuntimeError("correctness sentinel expected an initial v5 shot")
        policy_state = FrozenPolicyState.capture(base)
        before = env.clone_state()
        before_hash = env.state_hash()
        first = _searcher(model).search(
            env, observation, incumbent, policy_state
        )
        middle = env.clone_state()
        second = _searcher(model).search(
            env, observation, incumbent, policy_state
        )
        after = env.clone_state()
        checks = {
            "unit_tests_passed": unit.returncode == 0,
            "exact_snapshot_restore": before == middle == after,
            "exact_native_state_hash_restore": (
                before_hash == env.state_hash()
            ),
            "duplicate_search_identity": first.sha256 == second.sha256,
            "candidate_zero_exact_incumbent": (
                first.incumbent.candidate.decision is incumbent
            ),
            "candidate_count_bounded": (
                len(first.outcomes) <= JOINT_CONFIG.branch_cap
            ),
            "branch_tick_cap_exact": all(
                value.simulated_ticks <= BARRIER_CONFIG.branch_tick_cap
                for value in first.outcomes
            ),
            "restore_accounting": (
                first.restore_checks == len(first.outcomes) + 1
            ),
            "policy_state_unchanged": (
                policy_state.sha256 == FrozenPolicyState.capture(base).sha256
            ),
            "all_outcomes_persisted": (
                len(first.outcomes) == len(first.certificates)
            ),
            "candidate_zero_delta_tie": (
                first.certificate_for(0).exact_delta_b2 == 0
                and not first.certificate_for(0).eligible
            ),
        }
        runner = env.runner_identity_manifest()
        attestation = attest_simulator_runtime(env).manifest()
    checks["passed"] = all(checks.values())
    report = {
        "schema": "r3g-solvency-correctness-report-v1",
        "development_only": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": preregistered["source_identity"]["sha256"],
        "unit_tests": {
            **unit_evidence,
            "partial_path": str(unit_path),
        },
        "integration": {
            "checks": checks,
            "first_search_sha256": first.sha256,
            "second_search_sha256": second.sha256,
            "outcomes": len(first.outcomes),
            "unsafe_outcomes": sum(
                not value.hard_valid or value.minimum_margin < 0
                for value in first.outcomes
                if value.candidate.ordinal != 0
            ),
            "runtime": {"runner": runner, "attestation": attestation},
        },
        "passed": checks["passed"],
        "execution": {
            "threading": _threading_manifest(),
            "wall_seconds": time.perf_counter() - wall_started,
            "cpu_seconds": time.process_time() - cpu_started,
        },
    }
    path = OUTPUT / "correctness.json"
    digest = _atomic_json(path, report)
    print(
        json.dumps(
            {"path": str(path), "sha256": digest, "checks": checks},
            sort_keys=True,
        )
    )


def teacher_screen(model: Any) -> None:
    preregistered = _require_frozen()
    sentinel_report = _validated_stage_report(
        OUTPUT / "sentinel.json",
        schema="r3g-frozen-v5-sentinel-report-v1",
    )
    if sentinel_report.get("passed") is not True:
        raise RuntimeError("sentinel gate did not pass")
    seeds, horizon, seed_manifest = _split_rows("teacher-screen")
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    baseline, runner, attestation, _ = _evaluate(
        seeds=seeds,
        horizon=horizon,
        label="frozen_v5",
        factory=lambda _env: _base_policy(model),
        split="teacher-screen",
    )
    candidate, second_runner, second_attestation, result_sets = _evaluate(
        seeds=seeds,
        horizon=horizon,
        label="analytic_solvency_teacher",
        factory=lambda env: AnalyticSolvencyTeacherPolicy(
            env, _base_policy(model), _searcher(model)
        ),
        split="teacher-screen",
    )
    if runner != second_runner or attestation != second_attestation:
        raise RuntimeError("teacher screen paired runtime identity changed")
    paired = _paired(baseline, candidate)
    base_aggregate = _aggregate(baseline)
    candidate_aggregate = _aggregate(candidate)
    base_p10 = float(base_aggregate["survival_ticks"]["p10"])
    candidate_p10 = float(candidate_aggregate["survival_ticks"]["p10"])
    survival_gate = (
        candidate_p10 > base_p10
        if base_p10 < horizon
        else candidate_p10 == horizon
        and not paired["survival_regressions"]
    )
    gate = {
        "zero_hard_failures": not paired["hard_failures"],
        "score_median_improved": (
            float(candidate_aggregate["score"]["median"])
            > float(base_aggregate["score"]["median"])
        ),
        "survival_p10_rule": survival_gate,
    }
    gate["passed"] = all(gate.values())
    report = {
        "schema": "r3g-analytic-teacher-screen-v1",
        "development_only": True,
        "allocation_screen_only": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": preregistered["source_identity"]["sha256"],
        "seed_manifest": seed_manifest,
        "runtime": {
            "path": str(RUNTIME),
            "sha256": RUNTIME_SHA256,
            "runner": runner,
            "attestation": attestation,
        },
        "checkpoint": {
            "path": str(CHECKPOINT),
            "sha256": CHECKPOINT_SHA256,
        },
        "baseline": {"aggregate": base_aggregate, "episodes": baseline},
        "candidate": {
            "aggregate": candidate_aggregate,
            "episodes": candidate,
            "all_candidate_outcomes_retained": all(
                bool(results) for results in result_sets
            ),
        },
        "paired": paired,
        "gate": gate,
        "execution": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threading": _threading_manifest(),
            "wall_seconds": time.perf_counter() - wall_started,
            "cpu_seconds": time.process_time() - cpu_started,
        },
    }
    path = OUTPUT / "teacher-screen.json"
    digest = _atomic_json(path, report)
    print(
        json.dumps(
            {"path": str(path), "sha256": digest, "gate": gate},
            sort_keys=True,
        )
    )


def calibrate(model: Any) -> None:
    preregistered = _require_frozen()
    teacher_report = _validated_stage_report(
        OUTPUT / "teacher-screen.json",
        schema="r3g-analytic-teacher-screen-v1",
        split="teacher-screen",
    )
    if teacher_report.get("gate", {}).get("passed") is not True:
        raise RuntimeError("teacher screen did not pass")
    seeds, horizon, seed_manifest = _split_rows("barrier-calibration")
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    episodes, runner, attestation, result_sets = _evaluate(
        seeds=seeds,
        horizon=horizon,
        label="analytic_solvency_calibration_teacher",
        factory=lambda env: AnalyticSolvencyTeacherPolicy(
            env, _base_policy(model), _searcher(model)
        ),
        split="barrier-calibration",
    )
    q, residuals, exact_label_counts, rank = _conformal_q(
        result_sets, alpha=CONFORMAL_ALPHA
    )
    rows, labels, identities = _training_rows(result_sets)
    training_manifest = {
        "schema": "r3g-score-training-manifest-v1",
        "splits": ["barrier-calibration"],
        "seed_manifest_sha256": seed_manifest["manifest_sha256"],
        "row_identities": identities,
        "rows": len(rows),
    }
    training_manifest["sha256"] = _canonical_sha256(training_manifest)
    model_artifact = ScoreResidualModel.fit_rows(
        rows,
        labels,
        ridge=RIDGE,
        training_manifest_sha256=training_manifest["sha256"],
    )
    payload = _model_payload(
        model_artifact,
        source_identity_sha256=preregistered["source_identity"]["sha256"],
        training_manifest=training_manifest,
    )
    model_path = OUTPUT / "score-residual-initial.json"
    model_digest = _atomic_json(model_path, payload)
    report = {
        "schema": "r3g-barrier-calibration-report-v1",
        "development_only": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": preregistered["source_identity"]["sha256"],
        "seed_manifest": seed_manifest,
        "runtime": {"runner": runner, "attestation": attestation},
        "episodes": episodes,
        "aggregate": _aggregate(episodes),
        "conformal": {
            "alpha": CONFORMAL_ALPHA,
            "episode_max_overprediction_residuals": residuals,
            "episode_exact_two_renewal_label_counts": exact_label_counts,
            "evaluable_whole_seed_clusters": len(exact_label_counts),
            "rank": rank,
            "q": q,
            "prediction": "exact analytic delta_B2",
        },
        "calibration_risk_deciles": _risk_deciles(result_sets),
        "training_rows": [
            {
                "features": row,
                "label": label,
                "identity": identity,
            }
            for row, label, identity in zip(
                rows, labels, identities, strict=True
            )
        ],
        "training_manifest": training_manifest,
        "model": {
            "path": str(model_path),
            "file_sha256": model_digest,
            "model_sha256": model_artifact.sha256,
        },
        "execution": {
            "threading": _threading_manifest(),
            "wall_seconds": time.perf_counter() - wall_started,
            "cpu_seconds": time.process_time() - cpu_started,
        },
    }
    path = OUTPUT / "barrier-calibration.json"
    digest = _atomic_json(path, report)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": digest,
                "model": str(model_path),
                "model_sha256": model_artifact.sha256,
                "rows": len(rows),
                "conformal_q": q,
            },
            sort_keys=True,
        )
    )


def dagger(model: Any) -> None:
    preregistered = _require_frozen()
    calibration = _validated_stage_report(
        OUTPUT / "barrier-calibration.json",
        schema="r3g-barrier-calibration-report-v1",
        split="barrier-calibration",
    )
    initial = _load_model(
        OUTPUT / "score-residual-initial.json", ["barrier-calibration"]
    )
    _verify_reported_model(
        calibration,
        path=OUTPUT / "score-residual-initial.json",
        model=initial,
    )
    seeds, horizon, seed_manifest = _split_rows("dagger-train")
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    episodes, runner, attestation, result_sets = _evaluate(
        seeds=seeds,
        horizon=horizon,
        label="score_residual_dagger",
        factory=lambda env: LearnedScoreResidualPolicy(
            env,
            _base_policy(model),
            _searcher(
                model,
                conformal_q=float(calibration["conformal"]["q"]),
            ),
            initial,
        ),
        split="dagger-train",
    )
    new_rows, new_labels, new_identities = _training_rows(result_sets)
    old_records = calibration["training_rows"]
    rows = [value["features"] for value in old_records] + new_rows
    labels = [float(value["label"]) for value in old_records] + new_labels
    identities = [value["identity"] for value in old_records] + new_identities
    training_manifest = {
        "schema": "r3g-score-training-manifest-v1",
        "splits": ["barrier-calibration", "dagger-train"],
        "seed_manifest_sha256s": [
            calibration["seed_manifest"]["manifest_sha256"],
            seed_manifest["manifest_sha256"],
        ],
        "row_identities": identities,
        "rows": len(rows),
    }
    training_manifest["sha256"] = _canonical_sha256(training_manifest)
    final_model = ScoreResidualModel.fit_rows(
        rows,
        labels,
        ridge=RIDGE,
        training_manifest_sha256=training_manifest["sha256"],
    )
    payload = _model_payload(
        final_model,
        source_identity_sha256=preregistered["source_identity"]["sha256"],
        training_manifest=training_manifest,
    )
    model_path = OUTPUT / "score-residual-final.json"
    model_digest = _atomic_json(model_path, payload)
    report = {
        "schema": "r3g-score-residual-dagger-report-v1",
        "development_only": True,
        "on_policy_querying_only": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": preregistered["source_identity"]["sha256"],
        "seed_manifest": seed_manifest,
        "runtime": {"runner": runner, "attestation": attestation},
        "episodes": episodes,
        "aggregate": _aggregate(episodes),
        "dagger_training_rows": [
            {
                "features": row,
                "label": label,
                "identity": identity,
            }
            for row, label, identity in zip(
                new_rows, new_labels, new_identities, strict=True
            )
        ],
        "training_manifest": training_manifest,
        "model": {
            "path": str(model_path),
            "file_sha256": model_digest,
            "model_sha256": final_model.sha256,
            "teacher_free_selection": True,
        },
        "execution": {
            "threading": _threading_manifest(),
            "wall_seconds": time.perf_counter() - wall_started,
            "cpu_seconds": time.process_time() - cpu_started,
        },
    }
    path = OUTPUT / "dagger-train.json"
    digest = _atomic_json(path, report)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": digest,
                "model_sha256": final_model.sha256,
                "new_rows": len(new_rows),
                "total_rows": len(rows),
            },
            sort_keys=True,
        )
    )


def freeze_final(_model: Any) -> None:
    _require_frozen()
    forbidden = [
        OUTPUT / "barrier-heldout-stress.json",
        OUTPUT / "student-screen.json",
        *(OUTPUT / "partial").glob("barrier-heldout-*"),
        *(OUTPUT / "partial").glob("barrier-stress-*"),
    ]
    if any(path.exists() for path in forbidden):
        raise RuntimeError("heldout material was viewed before final freeze")
    value = _final_frozen_identity()
    path = OUTPUT / "frozen-final-student.json"
    digest = _atomic_json(path, value)
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": digest,
                "identity_sha256": value["identity_sha256"],
            },
            sort_keys=True,
        )
    )


def barrier(model: Any) -> None:
    preregistered = _require_frozen()
    frozen = _require_final_frozen()
    final_model = _load_model(
        OUTPUT / "score-residual-final.json",
        ["barrier-calibration", "dagger-train"],
    )
    calibration = _validated_stage_report(
        OUTPUT / "barrier-calibration.json",
        schema="r3g-barrier-calibration-report-v1",
        split="barrier-calibration",
    )
    if float(calibration["conformal"]["q"]) != 0.0:
        raise RuntimeError("analytic exact-delta calibration unexpectedly nonzero")
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    cohorts: dict[str, object] = {}
    all_wall = all_cpu = 0.0
    for split in ("barrier-heldout", "barrier-stress"):
        seeds, horizon, seed_manifest = _split_rows(split)
        episodes, runner, attestation, result_sets = _evaluate(
            seeds=seeds,
            horizon=horizon,
            label=f"final_score_residual_{split}",
            factory=lambda env: LearnedScoreResidualPolicy(
                env,
                _base_policy(model),
                _searcher(
                    model,
                    conformal_q=float(calibration["conformal"]["q"]),
                ),
                final_model,
                query_predicate=(
                    _stress_observation
                    if split == "barrier-stress"
                    else None
                ),
                maximum_queries=1 if split == "barrier-stress" else None,
            ),
            split=split,
        )
        metrics = _barrier_cohort(
            episodes,
            result_sets,
            targeted_stress=split == "barrier-stress",
        )
        cohorts[split] = {
            "seed_manifest": seed_manifest,
            "runtime": {"runner": runner, "attestation": attestation},
            "episodes": episodes,
            "aggregate": _aggregate(episodes),
            "barrier": metrics,
        }
        all_wall += sum(float(value["wall_seconds"]) for value in episodes)
        all_cpu += sum(float(value["cpu_seconds"]) for value in episodes)

    heldout = cohorts["barrier-heldout"]["barrier"]
    stress = cohorts["barrier-stress"]["barrier"]
    common_checks = {
        "heldout_at_least_59_episodes": heldout["episodes"] >= 59,
        "stress_at_least_59_episodes": stress["episodes"] >= 59,
        "heldout_zero_false_safe": heldout["false_safe_episodes"] == 0,
        "stress_zero_false_safe": stress["false_safe_episodes"] == 0,
        "heldout_cp95_at_most_5pct": heldout["false_safe_cp95_upper"] <= 0.05,
        "stress_cp95_at_most_5pct": stress["false_safe_cp95_upper"] <= 0.05,
        "heldout_coverage_lcb_at_least_5pct": (
            heldout["seed_clustered_coverage_lcb95"] >= 0.05
        ),
        "stress_each_episode_has_unsafe_proposal": (
            stress["unsafe_proposal_episodes"] == stress["episodes"]
        ),
        "stress_each_episode_reaches_target": (
            stress["target_episodes"] == stress["episodes"]
        ),
        "heldout_minimum_unsafe_outcomes": heldout["unsafe_outcomes"] >= 32,
        "heldout_minimum_unsafe_states": heldout["unsafe_decision_states"] >= 8,
        "heldout_minimum_unsafe_seeds": heldout["unsafe_seeds"] >= 4,
        "heldout_minimum_hard_unsafe": heldout["hard_unsafe_outcomes"] >= 8,
    }
    common_checks["passed"] = all(common_checks.values())
    report = {
        "schema": "r3g-final-student-barrier-report-v1",
        "development_only": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": preregistered["source_identity"]["sha256"],
        "frozen_final_identity_sha256": frozen["identity_sha256"],
        "model": {
            "path": str(OUTPUT / "score-residual-final.json"),
            "sha256": final_model.sha256,
        },
        "conformal_q": calibration["conformal"]["q"],
        "cohorts": cohorts,
        "gate": common_checks,
        "execution": {
            "threading": _threading_manifest(),
            "wall_seconds_observed": time.perf_counter() - wall_started,
            "cpu_seconds_observed": time.process_time() - cpu_started,
            "episode_wall_seconds_sum": all_wall,
            "episode_cpu_seconds_sum": all_cpu,
        },
    }
    path = OUTPUT / "barrier-heldout-stress.json"
    digest = _atomic_json(path, report)
    print(
        json.dumps(
            {"path": str(path), "sha256": digest, "gate": common_checks},
            sort_keys=True,
        )
    )


def student_screen(model: Any) -> None:
    preregistered = _require_frozen()
    frozen = _require_final_frozen()
    barrier_report = _validated_stage_report(
        OUTPUT / "barrier-heldout-stress.json",
        schema="r3g-final-student-barrier-report-v1",
    )
    if (
        barrier_report.get("frozen_final_identity_sha256")
        != frozen["identity_sha256"]
        or barrier_report.get("gate", {}).get("passed") is not True
    ):
        raise RuntimeError("final-student barrier gate did not pass")
    final_model = _load_model(
        OUTPUT / "score-residual-final.json",
        ["barrier-calibration", "dagger-train"],
    )
    seeds, horizon, seed_manifest = _split_rows("student-screen")
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    baseline, runner, attestation, _ = _evaluate(
        seeds=seeds,
        horizon=horizon,
        label="frozen_v5",
        factory=lambda _env: _base_policy(model),
        split="student-screen",
    )
    candidate, second_runner, second_attestation, _ = _evaluate(
        seeds=seeds,
        horizon=horizon,
        label="teacher_free_score_residual",
        factory=lambda env: LearnedScoreResidualPolicy(
            env,
            _base_policy(model),
            _searcher(
                model,
                conformal_q=float(frozen["calibration"]["q"]),
            ),
            final_model,
        ),
        split="student-screen",
    )
    if runner != second_runner or attestation != second_attestation:
        raise RuntimeError("student screen paired runtime identity changed")
    paired = _paired(baseline, candidate)
    gate = {
        "teacher_free_inference": True,
        "zero_hard_paired_failures": not paired["hard_failures"],
    }
    gate["passed"] = all(gate.values())
    report = {
        "schema": "r3g-teacher-free-student-screen-v1",
        "development_only": True,
        "allocation_screen_only": True,
        "cannot_authorize_confirmation": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": preregistered["source_identity"]["sha256"],
        "frozen_final_identity_sha256": frozen["identity_sha256"],
        "seed_manifest": seed_manifest,
        "runtime": {"runner": runner, "attestation": attestation},
        "model": {
            "path": str(OUTPUT / "score-residual-final.json"),
            "sha256": final_model.sha256,
        },
        "baseline": {"aggregate": _aggregate(baseline), "episodes": baseline},
        "candidate": {
            "aggregate": _aggregate(candidate),
            "episodes": candidate,
        },
        "paired": paired,
        "gate": gate,
        "execution": {
            "threading": _threading_manifest(),
            "wall_seconds": time.perf_counter() - wall_started,
            "cpu_seconds": time.process_time() - cpu_started,
        },
    }
    path = OUTPUT / "student-screen.json"
    digest = _atomic_json(path, report)
    print(
        json.dumps(
            {"path": str(path), "sha256": digest, "gate": gate},
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "boundary-audit",
            "preregister",
            "correctness",
            "sentinel",
            "teacher-screen",
            "calibrate",
            "dagger",
            "freeze-final",
            "barrier",
            "student-screen",
        ),
    )
    args = parser.parse_args()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            parser.error(f"{name}=1 is required")
    if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) != 1:
        parser.error("process affinity must contain exactly one logical CPU")
    _assert_trusted_inputs()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (
        torch.get_num_threads() != 1
        or torch.get_num_interop_threads() != 1
    ):
        parser.error("Torch thread counts are not one")
    try:
        if args.stage == "boundary-audit":
            boundary_audit()
            return
        if args.stage == "preregister":
            preregister()
            return
        artifact = load_steering_checkpoint(
            CHECKPOINT, expected_sha256=CHECKPOINT_SHA256
        )
        {
            "correctness": correctness,
            "sentinel": sentinel,
            "teacher-screen": teacher_screen,
            "calibrate": calibrate,
            "dagger": dagger,
            "freeze-final": freeze_final,
            "barrier": barrier,
            "student-screen": student_screen,
        }[args.stage](artifact.model)
    except Exception as exc:
        failure = OUTPUT / f"stage-failure-{args.stage}.json"
        if not failure.exists():
            _atomic_json(
                failure,
                {
                    "schema": "r3g-stage-failure-evidence-v1",
                    "development_only": True,
                    "protocol_sha256": PROTOCOL_SHA256,
                    "stage": args.stage,
                    "exception_type": type(exc).__name__,
                    "exception": str(exc),
                    "traceback": traceback.format_exc(),
                    "completed_partial_episode_files": sorted(
                        str(path)
                        for path in (OUTPUT / "partial").glob("*.json")
                    ),
                },
            )
        raise


if __name__ == "__main__":
    main()
