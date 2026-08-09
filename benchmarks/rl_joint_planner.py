#!/usr/bin/env python3
"""Development-only joint pair/geometry teacher and plateau benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from irisu_env import Action, ActionKind, EventKind, IrisuEnv
from irisu_pointer.action import PointerActionSpec
from irisu_pointer.joint_planner import (
    JOINT_PLANNER_VERSION,
    JointPairGeometrySearch,
    JointPlannerConfig,
    JointPlannerStudentPolicy,
    JointTeacherPolicy,
)
from irisu_pointer.steering import SteeringDecision
from irisu_pointer.steering_checkpoint import (
    load_steering_checkpoint,
    save_steering_checkpoint,
)
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
    SteeringDataset,
    SteeringExample,
    steering_example_from_decision,
    train_goal_conditioned_steering,
)
from irisu_rl.runtime_identity import attest_simulator_runtime
from irisu_rl.schema import TEACHER_V1


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/runtime/"
    "main-0c48dba-20260723/portable-build/libirisu_clone.so"
)
RUNTIME_SHA256 = (
    "4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79"
)
BASE = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3d-survival-v5-20260729/long-development.pt"
)
BASE_SHA256 = (
    "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/r3/development/"
    "r3f-joint-planner-v2-20260729"
)
MAX_DECISIONS = 2_000_000
TRAINING_SEED = 2026072907


def _derive(label: str, count: int) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(
            hashlib.sha256(f"{label}:{index}".encode()).digest()[:4], "big"
        )
        for index in range(count)
    )


COLLECTION_LABEL = "irisu-joint-planner-collection-development-v1"
CURVE_LABEL = "irisu-joint-planner-curve-development-v1"
CEILING_LABEL = "irisu-joint-planner-teacher-ceiling-development-v1"
PLATEAU_10K_LABEL = "irisu-joint-planner-plateau-10k-development-v1"
PLATEAU_20K_LABEL = "irisu-joint-planner-plateau-20k-development-v1"
FEASIBILITY_50K_LABEL = "irisu-joint-planner-feasibility-50k-development-v1"
COLLECTION_SEEDS = _derive(COLLECTION_LABEL, 16)
CURVE_SEEDS = _derive(CURVE_LABEL, 12)
CEILING_SEEDS = _derive(CEILING_LABEL, 8)
PLATEAU_10K_SEEDS = _derive(PLATEAU_10K_LABEL, 16)
PLATEAU_20K_SEEDS = _derive(PLATEAU_20K_LABEL, 16)
FEASIBILITY_50K_SEEDS = _derive(FEASIBILITY_50K_LABEL, 4)
EXPECTED_SEEDS = {
    COLLECTION_LABEL: (
        0x63D96529,
        0x2672B386,
        0x2A4A4342,
        0xFF9DF4C0,
        0x7452D4A8,
        0x1120DD04,
        0x66D7072C,
        0x278A52CA,
        0x1C36B07A,
        0x11F4D927,
        0x6A5A5087,
        0x4DA68F1E,
        0x0CB95922,
        0x6063B863,
        0x33B297FD,
        0xD2C9BE8C,
    ),
    CURVE_LABEL: (
        0xE9BAD9ED,
        0xC12ACC09,
        0x911317AF,
        0x802B0153,
        0xF2133742,
        0xAFFF48B1,
        0xE25D0E9C,
        0xDD1A0751,
        0x630149FA,
        0x528560D5,
        0x726D35FD,
        0x0637DB8E,
    ),
    CEILING_LABEL: CEILING_SEEDS,
    PLATEAU_10K_LABEL: PLATEAU_10K_SEEDS,
    PLATEAU_20K_LABEL: PLATEAU_20K_SEEDS,
    FEASIBILITY_50K_LABEL: FEASIBILITY_50K_SEEDS,
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _validate_suites() -> None:
    derived = {
        COLLECTION_LABEL: COLLECTION_SEEDS,
        CURVE_LABEL: CURVE_SEEDS,
        CEILING_LABEL: CEILING_SEEDS,
        PLATEAU_10K_LABEL: PLATEAU_10K_SEEDS,
        PLATEAU_20K_LABEL: PLATEAU_20K_SEEDS,
        FEASIBILITY_50K_LABEL: FEASIBILITY_50K_SEEDS,
    }
    if derived != EXPECTED_SEEDS:
        raise RuntimeError("joint suite derivation changed")
    flat = [seed for values in derived.values() for seed in values]
    if len(flat) != len(set(flat)):
        raise RuntimeError("joint development suites overlap")


def _suite_manifest(
    label: str, seeds: Sequence[int], horizon: int
) -> dict[str, object]:
    value = {
        "version": "irisu-joint-development-suite-v1",
        "label": label,
        "seeds": list(seeds),
        "config": {
            "max_episode_ticks": horizon,
            "physics_backend": "portable",
        },
        "max_decisions_per_episode": MAX_DECISIONS,
        "disjoint_from_training": label != COLLECTION_LABEL,
    }
    return {**value, "sha256": _canonical_sha256(value)}


def _safe_output_dir(path: Path) -> Path:
    supplied = path.expanduser()
    if supplied.is_symlink():
        raise ValueError("joint output directory must not be a symlink")
    resolved = supplied.resolve()
    development = (ROOT / "artifacts/r3/development").resolve()
    if development not in resolved.parents or "joint-planner" not in resolved.name:
        raise ValueError(
            "joint outputs require a unique development namespace containing "
            "'joint-planner'"
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _atomic_json(path: Path, value: Mapping[str, object], overwrite: bool) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
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
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _file_sha256(path)


def _atomic_torch(path: Path, value: Mapping[str, object], overwrite: bool) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(path)
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
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _file_sha256(path)


def _source_identity() -> dict[str, object]:
    files = (
        Path(__file__).resolve(),
        ROOT / "benchmarks/joint-planner-dev-checks.py",
        ROOT / "python/irisu_pointer/joint_planner.py",
        ROOT / "python/irisu_pointer/action.py",
        ROOT / "python/irisu_pointer/policy.py",
        ROOT / "python/irisu_pointer/steering.py",
        ROOT / "python/irisu_pointer/steering_checkpoint.py",
        ROOT / "python/irisu_pointer/steering_learning.py",
        ROOT / "python/irisu_pointer/steering_progress.py",
        ROOT / "python/irisu_rl/actions.py",
        ROOT / "python/irisu_rl/encoding.py",
        ROOT / "python/irisu_rl/runtime_identity.py",
        ROOT / "python/irisu_rl/schema.py",
        ROOT / "python/irisu_env/env.py",
        ROOT / "python/irisu_env/native.py",
        ROOT / "pyproject.toml",
    )
    manifest = {
        "schema": "irisu-joint-planner-source-v1",
        "git_revision": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "files": {
            str(path.relative_to(ROOT)): _file_sha256(path) for path in files
        },
    }
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _assert_inputs_unchanged(source: Mapping[str, object]) -> None:
    if _source_identity() != source:
        raise RuntimeError("joint planner source identity changed during execution")
    if _file_sha256(RUNTIME) != RUNTIME_SHA256:
        raise RuntimeError("trusted portable runtime changed during execution")
    if _file_sha256(BASE) != BASE_SHA256:
        raise RuntimeError("frozen v5 comparator changed during execution")


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    ordered = sorted(int(value) for value in values)

    def percentile(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower, upper = math.floor(position), math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "minimum": ordered[0],
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "maximum": ordered[-1],
    }


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    policy: str
    seed: int
    horizon_ticks: int
    survival_ticks: int
    final_score: int
    final_gauge: int
    gauge_max: int
    final_level: int
    highest_chain: int
    qualifying_clears: int
    cleared_events: int
    rotten_events: int
    positive_gauge_renewal: int
    shots_fired: int
    shots_hit: int
    chain_joins: int
    invalid_actions: int
    decisions: int
    primitive_actions: int
    terminated: bool
    truncated: bool
    checkpoints: Mapping[str, Mapping[str, object]]
    policy_counts: Mapping[str, int | float]
    search_result_sha256s: tuple[str, ...]
    wall_seconds: float
    cpu_seconds: float
    max_rss_kib: int

    @property
    def gauge_failure(self) -> bool:
        return self.terminated and self.final_gauge <= 1

    def manifest(self) -> dict[str, object]:
        return {
            **asdict(self),
            "gauge_failure": self.gauge_failure,
        }


def _policy_factory(
    model: GoalConditionedSteeringModel,
    *,
    policy_identity: str,
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
        artifact_sha256=policy_identity,
    )


def _run_episode(
    *,
    library: Path,
    label: str,
    seed: int,
    horizon: int,
    factory: Callable[[IrisuEnv], object],
    decision_hook: Callable[
        [Mapping[str, Any], SteeringDecision, object], None
    ]
    | None = None,
) -> tuple[EpisodeMetrics, Mapping[str, object], Mapping[str, object]]:
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    usage_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    checkpoints = tuple(
        value for value in (2_000, 10_000, 20_000, 50_000) if value <= horizon
    )
    captured: dict[str, Mapping[str, object]] = {}
    counts: Counter[int] = Counter()
    hit_projectiles: set[int] = set()
    positive_gauge = 0
    decisions = primitive_actions = 0
    highest_chain = 0
    with IrisuEnv(
        library_path=library,
        physics_backend="portable",
        config={"max_episode_ticks": horizon},
    ) as env:
        runner = env.runner_identity_manifest()
        attestation = attest_simulator_runtime(env).manifest()
        observation, reset_info = env.reset(seed=seed)
        config_hash = int(runner["config_hash"])
        if int(reset_info.get("config_hash", -1)) != config_hash:
            raise RuntimeError("joint reset config identity mismatch")
        initial_tick = int(observation.get("tick", 0))
        initial_clears = int(observation.get("qualifying_clear_count", 0))
        policy = factory(env)
        getattr(policy, "reset")(seed)
        terminated = bool(observation.get("terminated", False))
        truncated = bool(observation.get("truncated", False))
        while not terminated and not truncated:
            if decisions >= MAX_DECISIONS:
                raise RuntimeError("joint episode exceeded its decision budget")
            decision = getattr(policy, "predict")(observation)
            if not isinstance(decision, SteeringDecision):
                raise TypeError("joint evaluated policy returned a non-decision")
            if decision_hook is not None:
                decision_hook(observation, decision, policy)
            actions = decision.primitive_actions()
            decisions += 1
            for action in actions:
                if terminated or truncated:
                    break
                current_tick = int(observation.get("tick", 0))
                pending = [value for value in checkpoints if value > current_tick]
                if (
                    ActionKind.parse(action.kind) is ActionKind.WAIT
                    and pending
                    and current_tick + int(action.wait_ticks) > pending[0]
                ):
                    action = Action.wait(pending[0] - current_tick)
                observation, _, terminated, truncated, info = env.step(action)
                primitive_actions += 1
                if int(info.get("config_hash", -1)) != config_hash:
                    raise RuntimeError("joint step config identity mismatch")
                for event in info.get("events", ()):
                    if not isinstance(event, Mapping):
                        continue
                    kind = _event_kind(event)
                    if kind is None:
                        continue
                    counts[kind] += 1
                    if kind == int(EventKind.PROJECTILE_HIT):
                        hit_projectiles.add(int(event.get("a", -1)))
                    if kind == int(EventKind.GAUGE_CHANGED):
                        positive_gauge += max(0, int(event.get("value", 0)))
                highest_chain = max(
                    highest_chain, int(observation.get("highest_chain", 0))
                )
                tick = int(observation.get("tick", 0))
                if tick in checkpoints:
                    captured[str(tick)] = {
                        "tick": tick,
                        "score": int(observation.get("score", 0)),
                        "gauge": int(observation.get("gauge", 0)),
                        "level": int(observation.get("level", 0)),
                        "qualifying_clears": int(
                            observation.get("qualifying_clear_count", 0)
                        )
                        - initial_clears,
                        "reached": True,
                    }
        survival = int(observation.get("tick", 0)) - initial_tick
        final_score = int(observation.get("score", 0))
        for checkpoint in checkpoints:
            if str(checkpoint) not in captured:
                captured[str(checkpoint)] = {
                    "tick": survival,
                    "score": final_score,
                    "gauge": int(observation.get("gauge", 0)),
                    "level": int(observation.get("level", 0)),
                    "qualifying_clears": int(
                        observation.get("qualifying_clear_count", 0)
                    )
                    - initial_clears,
                    "reached": False,
                    "carried_terminal_score": True,
                }
        statistics = getattr(policy, "statistics", None)
        policy_counts = statistics() if callable(statistics) else {}
        results = getattr(policy, "results", ())
        search_ids = tuple(value.sha256 for value in results)
        result = EpisodeMetrics(
            label,
            seed,
            horizon,
            survival,
            final_score,
            int(observation.get("gauge", 0)),
            int(observation.get("gauge_max", 0)),
            int(observation.get("level", 0)),
            highest_chain,
            int(observation.get("qualifying_clear_count", 0))
            - initial_clears,
            counts[int(EventKind.CLEARED)],
            counts[int(EventKind.ROTTEN)],
            positive_gauge,
            counts[int(EventKind.SHOT_FIRED)],
            len(hit_projectiles - {-1}),
            counts[int(EventKind.CHAIN_JOINED)],
            counts[int(EventKind.INVALID_ACTION)],
            decisions,
            primitive_actions,
            bool(terminated),
            bool(truncated),
            captured,
            policy_counts,
            search_ids,
            time.perf_counter() - wall_started,
            time.process_time() - cpu_started,
            max(
                0,
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                - int(usage_before),
            ),
        )
    return result, runner, attestation


def _aggregate(episodes: Sequence[EpisodeMetrics]) -> dict[str, object]:
    policy_counts: Counter[str] = Counter()
    for episode in episodes:
        for key, value in episode.policy_counts.items():
            policy_counts[key] += value
    checkpoint_keys = [
        str(checkpoint)
        for checkpoint in (2_000, 10_000, 20_000, 50_000)
        if all(str(checkpoint) in value.checkpoints for value in episodes)
    ]
    reached_scores = {
        key: [
            int(value.checkpoints[key]["score"])
            for value in episodes
            if bool(value.checkpoints[key]["reached"])
        ]
        for key in checkpoint_keys
    }
    return {
        "episodes": len(episodes),
        "score": _distribution([value.final_score for value in episodes]),
        "survival_ticks": _distribution(
            [value.survival_ticks for value in episodes]
        ),
        "final_gauge": _distribution(
            [value.final_gauge for value in episodes]
        ),
        "final_level": _distribution(
            [value.final_level for value in episodes]
        ),
        "highest_chain": _distribution(
            [value.highest_chain for value in episodes]
        ),
        "gauge_failures": sum(value.gauge_failure for value in episodes),
        "qualifying_clears": sum(value.qualifying_clears for value in episodes),
        "cleared_events": sum(value.cleared_events for value in episodes),
        "rotten_events": sum(value.rotten_events for value in episodes),
        "positive_gauge_renewal": sum(
            value.positive_gauge_renewal for value in episodes
        ),
        "shots_fired": sum(value.shots_fired for value in episodes),
        "shots_hit": sum(value.shots_hit for value in episodes),
        "chain_joins": sum(value.chain_joins for value in episodes),
        "invalid_actions": sum(value.invalid_actions for value in episodes),
        "decisions": sum(value.decisions for value in episodes),
        "wall_seconds": sum(value.wall_seconds for value in episodes),
        "cpu_seconds": sum(value.cpu_seconds for value in episodes),
        "policy_counts": dict(sorted(policy_counts.items())),
        "checkpoint_reach": {
            key: {
                "reached": len(reached_scores[key]),
                "episodes": len(episodes),
            }
            for key in checkpoint_keys
        },
        "score_by_checkpoint": {
            key: _distribution(reached_scores[key])
            for key in checkpoint_keys
            if reached_scores[key]
        },
        "score_by_checkpoint_carried_terminal": {
            key: _distribution(
                [int(value.checkpoints[key]["score"]) for value in episodes]
            )
            for key in checkpoint_keys
        },
    }


def _paired(
    baseline: Sequence[EpisodeMetrics],
    candidate: Sequence[EpisodeMetrics],
) -> dict[str, object]:
    base = {value.seed: value for value in baseline}
    other = {value.seed: value for value in candidate}
    if set(base) != set(other):
        raise ValueError("joint paired evaluations use different seeds")
    rows = []
    for seed in sorted(base):
        first, second = base[seed], other[seed]
        survival_delta = second.survival_ticks - first.survival_ticks
        catastrophe_floor = -max(2_000, int(0.25 * first.survival_ticks))
        original_catastrophe = (
            first.survival_ticks >= 2_000
            and survival_delta <= catastrophe_floor
        )
        below_half_baseline = (
            first.survival_ticks >= 2_000
            and second.survival_ticks < 0.5 * first.survival_ticks
        )
        lost_full_survivor = (
            first.survival_ticks >= first.horizon_ticks
            and second.survival_ticks < second.horizon_ticks
        )
        new_gauge_death = (
            second.gauge_failure and not first.gauge_failure
        )
        rows.append(
            {
                "seed": seed,
                "seed_hex": f"0x{seed:08X}",
                "baseline_score": first.final_score,
                "candidate_score": second.final_score,
                "score_delta": second.final_score - first.final_score,
                "baseline_survival": first.survival_ticks,
                "candidate_survival": second.survival_ticks,
                "survival_delta": survival_delta,
                "baseline_gauge_failure": first.gauge_failure,
                "candidate_gauge_failure": second.gauge_failure,
                "baseline_final_gauge": first.final_gauge,
                "candidate_final_gauge": second.final_gauge,
                "baseline_final_level": first.final_level,
                "candidate_final_level": second.final_level,
                "baseline_qualifying_clears": first.qualifying_clears,
                "candidate_qualifying_clears": second.qualifying_clears,
                "baseline_positive_gauge_renewal": first.positive_gauge_renewal,
                "candidate_positive_gauge_renewal": second.positive_gauge_renewal,
                "positive_gauge_renewal_delta": (
                    second.positive_gauge_renewal
                    - first.positive_gauge_renewal
                ),
                "baseline_rotten_events": first.rotten_events,
                "candidate_rotten_events": second.rotten_events,
                "survival_regression": survival_delta < 0,
                "catastrophic_survival_regression": original_catastrophe,
                "below_half_baseline_survival": below_half_baseline,
                "lost_full_survivor": lost_full_survivor,
                "new_gauge_death": new_gauge_death,
                "promotion_catastrophic_tail_regression": (
                    original_catastrophe
                    or below_half_baseline
                    or lost_full_survivor
                    or new_gauge_death
                ),
            }
        )
    return {
        "definition": {
            "catastrophe": (
                "baseline survival >= 2000 and candidate delta <= "
                "-max(2000, 25% baseline)"
            ),
            "new_gauge_death": (
                "candidate gauge failure while paired baseline is not a "
                "gauge failure"
            ),
            "promotion_catastrophic_tail": (
                "union of catastrophe, candidate below 50% of baseline "
                "survival, loss of a full-horizon survivor, or new gauge death"
            ),
        },
        "score_wins": sum(value["score_delta"] > 0 for value in rows),
        "survival_wins": sum(value["survival_delta"] > 0 for value in rows),
        "survival_regressions": [
            value for value in rows if value["survival_regression"]
        ],
        "catastrophic_survival_regressions": [
            value
            for value in rows
            if value["catastrophic_survival_regression"]
        ],
        "new_gauge_deaths": [
            value for value in rows if value["new_gauge_death"]
        ],
        "promotion_catastrophic_tail_regressions": [
            value
            for value in rows
            if value["promotion_catastrophic_tail_regression"]
        ],
        "pairs": rows,
    }


def _curve_selection_key(curve: Mapping[str, object]) -> tuple[float, ...]:
    evaluation = curve["evaluation"]
    assert isinstance(evaluation, Mapping)
    paired = evaluation["paired"]
    baseline = evaluation["baseline"]
    student = evaluation["student"]
    assert (
        isinstance(paired, Mapping)
        and isinstance(baseline, Mapping)
        and isinstance(student, Mapping)
    )
    survival = student["survival_ticks"]
    score = student["score"]
    assert isinstance(survival, Mapping) and isinstance(score, Mapping)
    catastrophes = paired["promotion_catastrophic_tail_regressions"]
    assert isinstance(catastrophes, Sequence)
    return (
        -float(len(catastrophes)),
        -float(student["invalid_actions"]),
        -float(student["gauge_failures"]),
        float(survival["p10"]),
        float(survival["median"]),
        float(student["positive_gauge_renewal"])
        - float(baseline["positive_gauge_renewal"]),
        float(student["qualifying_clears"]),
        float(score["p10"]),
        float(score["median"]),
        -float(curve["steps"]),
    )


def _promotion_gate(
    paired: Mapping[str, object],
    baseline: Sequence[EpisodeMetrics],
    candidate: Sequence[EpisodeMetrics],
    *,
    selection_suite: bool,
) -> dict[str, object]:
    catastrophes = paired["promotion_catastrophic_tail_regressions"]
    assert isinstance(catastrophes, Sequence)
    invalid = sum(value.invalid_actions for value in candidate)
    baseline_aggregate = _aggregate(baseline)
    candidate_aggregate = _aggregate(candidate)
    aggregate_checks = {
        "gauge_failures_noninferior": (
            int(candidate_aggregate["gauge_failures"])
            <= int(baseline_aggregate["gauge_failures"])
        ),
        "survival_p10_noninferior": (
            float(candidate_aggregate["survival_ticks"]["p10"])
            >= float(baseline_aggregate["survival_ticks"]["p10"])
        ),
        "survival_median_noninferior": (
            float(candidate_aggregate["survival_ticks"]["median"])
            >= float(baseline_aggregate["survival_ticks"]["median"])
        ),
        "score_p10_noninferior": (
            float(candidate_aggregate["score"]["p10"])
            >= float(baseline_aggregate["score"]["p10"])
        ),
        "score_median_noninferior": (
            float(candidate_aggregate["score"]["median"])
            >= float(baseline_aggregate["score"]["median"])
        ),
        "qualifying_clears_noninferior": (
            int(candidate_aggregate["qualifying_clears"])
            >= int(baseline_aggregate["qualifying_clears"])
        ),
        "positive_gauge_renewal_noninferior": (
            int(candidate_aggregate["positive_gauge_renewal"])
            >= int(baseline_aggregate["positive_gauge_renewal"])
        ),
    }
    safety_passed = not catastrophes and invalid == 0
    aggregate_passed = all(aggregate_checks.values())
    passed = selection_suite and safety_passed and aggregate_passed
    return {
        "passed": passed,
        "selection_suite": selection_suite,
        "safety_passed": safety_passed,
        "aggregate_noninferiority_passed": aggregate_passed,
        "aggregate_checks": aggregate_checks,
        "verdict": (
            "suite-local development promotion gate passed"
            if passed
            else (
                "feasibility-only suite; not promotion eligible"
                if not selection_suite
                else "suite-local development promotion gate failed"
            )
        ),
        "promotion_catastrophic_tail_regressions": len(catastrophes),
        "invalid_actions": invalid,
    }


def _evaluate_suite(
    *,
    library: Path,
    label: str,
    seeds: Sequence[int],
    horizon: int,
    factory: Callable[[IrisuEnv], object],
    policy_label: str,
) -> tuple[list[EpisodeMetrics], Mapping[str, object], Mapping[str, object]]:
    episodes = []
    runners = []
    attestations = []
    for seed in seeds:
        episode, runner, attestation = _run_episode(
            library=library,
            label=policy_label,
            seed=int(seed),
            horizon=horizon,
            factory=factory,
        )
        episodes.append(episode)
        runners.append(runner)
        attestations.append(attestation)
    if len({_canonical_sha256(value) for value in runners}) != 1:
        raise RuntimeError("joint suite mixed runner identities")
    if len({_canonical_sha256(value) for value in attestations}) != 1:
        raise RuntimeError("joint suite mixed runtime attestations")
    return episodes, runners[0], attestations[0]


def _validate_teacher_accounting(
    episodes: Sequence[EpisodeMetrics],
) -> None:
    for episode in episodes:
        counts = episode.policy_counts
        attempts = int(counts.get("search_attempts", 0))
        successes = int(counts.get("search_queries", 0))
        rebinds = int(counts.get("progress_rebind_fallbacks", 0))
        unsupported = int(counts.get("unsupported_fallbacks", 0))
        completed = successes + rebinds
        if (
            attempts != completed + unsupported
            or int(counts.get("restore_checks", 0))
            != int(counts.get("branch_outcomes", 0)) + completed
            or attempts > 0
            and attempts > int(counts.get("seen_shots", 0))
        ):
            raise RuntimeError("joint teacher attempt accounting is inconsistent")


def _planner_config(args: argparse.Namespace) -> JointPlannerConfig:
    return JointPlannerConfig(
        pair_cap=args.pair_cap,
        geometry_cap=args.geometry_cap,
        horizons=tuple(args.horizons),
        cooldown_ticks=args.planning_cooldown,
        require_pristine_source=True,
    )


def _thresholds(args: argparse.Namespace) -> dict[str, float]:
    return {
        "pair_confidence": float(args.pair_confidence),
        "pair_margin": float(args.pair_margin),
        "geometry_confidence": float(args.geometry_confidence),
        "geometry_margin": float(args.geometry_margin),
    }


def _base_identity(source: Mapping[str, object]) -> str:
    return _canonical_sha256(
        {
            "type": "frozen-r3d-v5-goal-conditioned-steering-policy",
            "checkpoint_sha256": BASE_SHA256,
            "source_revision": source["git_revision"],
            "source_identity_sha256": source["sha256"],
            "implementation_sha256": source["files"][
                "python/irisu_pointer/steering_learning.py"
            ],
            "progress_tracker_sha256": source["files"][
                "python/irisu_pointer/steering_progress.py"
            ],
            "cooldown_ticks": 16,
            "minimum_pair_closure_sizes": 0.05,
            "impact_side_sizes": 0.5,
            "impact_below_sizes": 0.75,
            "source_velocity_lead_ticks": 1.0,
            "ticks_per_second": 50.0,
            "act_logit_bias": 1.0,
        }
    )


def _searcher(
    model: GoalConditionedSteeringModel,
    config: JointPlannerConfig,
    policy_identity: str,
) -> JointPairGeometrySearch:
    return JointPairGeometrySearch(
        lambda: _policy_factory(model, policy_identity=policy_identity),
        config=config,
        continuation_identity_sha256=policy_identity,
    )


def develop(args: argparse.Namespace) -> None:
    phase_wall_started = time.perf_counter()
    phase_cpu_started = time.process_time()
    output = _safe_output_dir(args.output_dir)
    source = _source_identity()
    base_artifact = load_steering_checkpoint(BASE, expected_sha256=BASE_SHA256)
    base_model = base_artifact.model
    base_model.eval()
    config = _planner_config(args)
    base_policy_identity = _base_identity(source)
    searcher_identity = _searcher(
        base_model, config, base_policy_identity
    ).identity_manifest()
    collection_seeds = COLLECTION_SEEDS[: args.collection_seeds]
    curve_seeds = CURVE_SEEDS[: args.curve_seeds]
    ceiling_seeds = CEILING_SEEDS[: args.ceiling_seeds]
    examples: list[SteeringExample] = []
    collection_episodes: list[EpisodeMetrics] = []
    collection_search_ids: list[str] = []
    collection_search_manifests: list[Mapping[str, object]] = []
    collection_runner: Mapping[str, object] | None = None
    attestation: Mapping[str, object] | None = None
    collection_started = time.perf_counter()
    for seed in collection_seeds:
        seen = 0

        def factory(env: IrisuEnv) -> JointTeacherPolicy:
            return JointTeacherPolicy(
                env,
                _policy_factory(
                    base_model, policy_identity=base_policy_identity
                ),
                _searcher(base_model, config, base_policy_identity),
                query_stride_shots=args.query_stride,
                maximum_queries=args.maximum_queries,
            )

        def collect(
            observation: Mapping[str, Any],
            decision: SteeringDecision,
            policy: object,
        ) -> None:
            nonlocal seen
            results = getattr(policy, "results")
            if len(results) == seen:
                return
            if len(results) != seen + 1:
                raise RuntimeError("joint teacher query accounting jumped")
            result = results[-1]
            seen += 1
            collection_search_manifests.append(result.identity_manifest())
            example = steering_example_from_decision(
                observation,
                decision,
                episode_identity=(
                    f"joint-collection:{seed:08x}:"
                    f"{int(observation.get('tick', 0))}:{seen}"
                ),
                provenance_sha256=result.sha256,
                pointer_spec=PointerActionSpec(),
                require_representable_template=False,
            )
            if example is not None:
                examples.append(example)

        episode, runner, runtime_attestation = _run_episode(
            library=RUNTIME,
            label="joint_teacher_collection",
            seed=seed,
            horizon=args.collection_ticks,
            factory=factory,
            decision_hook=collect,
        )
        collection_episodes.append(episode)
        collection_search_ids.extend(episode.search_result_sha256s)
        if collection_runner is not None and collection_runner != runner:
            raise RuntimeError("joint collection mixed runner identities")
        if attestation is not None and attestation != runtime_attestation:
            raise RuntimeError("joint collection mixed runtime attestations")
        collection_runner = runner
        attestation = runtime_attestation
    collection_wall_seconds = time.perf_counter() - collection_started
    if not examples or collection_runner is None or attestation is None:
        raise RuntimeError("joint collection produced no distillation labels")
    dataset = SteeringDataset(examples)
    if [
        _canonical_sha256(value) for value in collection_search_manifests
    ] != collection_search_ids:
        raise RuntimeError("joint search evidence identities are incomplete")
    _validate_teacher_accounting(collection_episodes)
    curves: list[dict[str, object]] = []
    trained_models: list[GoalConditionedSteeringModel] = []
    curve_baseline, curve_runner, _ = _evaluate_suite(
        library=RUNTIME,
        label=CURVE_LABEL,
        seeds=curve_seeds,
        horizon=args.curve_ticks,
        factory=lambda _env: _policy_factory(
            base_model, policy_identity=base_policy_identity
        ),
        policy_label="frozen_v5",
    )
    for steps in args.training_budgets:
        torch.manual_seed(TRAINING_SEED)
        model = GoalConditionedSteeringModel(
            TEACHER_V1,
            pointer_spec=dataset.pointer_spec,
            config=base_model.config,
        )
        training_started = time.perf_counter()
        training_cpu = time.process_time()
        training = train_goal_conditioned_steering(
            model,
            dataset,
            steps=steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=TRAINING_SEED,
        )
        training_wall_seconds = time.perf_counter() - training_started
        training_cpu_seconds = time.process_time() - training_cpu
        model.eval()
        student, _, _ = _evaluate_suite(
            library=RUNTIME,
            label=CURVE_LABEL,
            seeds=curve_seeds,
            horizon=args.curve_ticks,
            factory=lambda _env, selected=model: JointPlannerStudentPolicy(
                _policy_factory(
                    base_model, policy_identity=base_policy_identity
                ),
                selected,
                config=config,
                pair_confidence=args.pair_confidence,
                pair_margin=args.pair_margin,
                geometry_confidence=args.geometry_confidence,
                geometry_margin=args.geometry_margin,
            ),
            policy_label=f"joint_student_{steps}",
        )
        curves.append(
            {
                "steps": steps,
                "training_report": asdict(training),
                "training_wall_seconds": training_wall_seconds,
                "training_cpu_seconds": training_cpu_seconds,
                "evaluation": {
                    "baseline": _aggregate(curve_baseline),
                    "student": _aggregate(student),
                    "paired": _paired(curve_baseline, student),
                    "episodes": [value.manifest() for value in student],
                },
            }
        )
        trained_models.append(model)
    selected_curve_index = max(
        range(len(curves)), key=lambda index: _curve_selection_key(curves[index])
    )
    final_model = trained_models[selected_curve_index]
    selected_training_steps = int(curves[selected_curve_index]["steps"])
    teacher_evidence_path = output / "teacher-evidence.pt"
    _assert_inputs_unchanged(source)
    teacher_evidence_sha256 = _atomic_torch(
        teacher_evidence_path,
        {
            "format": "irisu-joint-planner-teacher-evidence-v2",
            "source_identity": source,
            "runtime_sha256": RUNTIME_SHA256,
            "base_checkpoint_sha256": BASE_SHA256,
            "base_policy_identity": base_policy_identity,
            "planner_identity": searcher_identity,
            "dataset_manifest": dataset.manifest(),
            "example_manifests": [value.manifest() for value in examples],
            "search_results": collection_search_manifests,
            "tensors": {
                "global_features": torch.from_numpy(
                    np.concatenate(
                        [
                            value.observation.global_features
                            for value in examples
                        ],
                        axis=0,
                    )
                ),
                "body_features": torch.from_numpy(
                    np.concatenate(
                        [
                            value.observation.body_features
                            for value in examples
                        ],
                        axis=0,
                    )
                ),
                "body_mask": torch.from_numpy(
                    np.concatenate(
                        [value.observation.body_mask for value in examples],
                        axis=0,
                    )
                ),
                **{
                    name: torch.tensor(
                        [getattr(value, name) for value in examples],
                        dtype=torch.long,
                    )
                    for name in (
                        "source_index",
                        "destination_index",
                        "kind_index",
                        "template_index",
                        "intent_index",
                        "act_index",
                        "wait_index",
                    )
                },
            },
        },
        args.overwrite,
    )
    checkpoint_path = output / "joint-student.pt"
    _assert_inputs_unchanged(source)
    checkpoint_sha256 = save_steering_checkpoint(
        checkpoint_path,
        final_model,
        metadata={
            "development_only": True,
            "canonical_r3_evidence": False,
            "sealed_test_material_used": False,
            "source_identity": source,
            "runtime_sha256": RUNTIME_SHA256,
            "base_checkpoint_sha256": BASE_SHA256,
            "base_policy_identity": base_policy_identity,
            "planner_identity": searcher_identity,
            "dataset_sha256": dataset.sha256,
            "teacher_evidence_sha256": teacher_evidence_sha256,
            "collection_suite": _suite_manifest(
                COLLECTION_LABEL, collection_seeds, args.collection_ticks
            ),
            "selected_curve_index": selected_curve_index,
            "selected_training_steps": selected_training_steps,
            "selection_rule": (
                "fewest promotion-catastrophic tails, invalid actions, and "
                "gauge failures; then survival p10/median, positive gauge "
                "renewal gain, clears, score p10/median, and fewer steps"
            ),
            "training_thresholds": _thresholds(args),
            "training_seed": TRAINING_SEED,
        },
        overwrite=args.overwrite,
    )

    ceiling_baseline, ceiling_runner, _ = _evaluate_suite(
        library=RUNTIME,
        label=CEILING_LABEL,
        seeds=ceiling_seeds,
        horizon=args.ceiling_ticks,
        factory=lambda _env: _policy_factory(
            base_model, policy_identity=base_policy_identity
        ),
        policy_label="frozen_v5",
    )

    def teacher_factory(env: IrisuEnv) -> JointTeacherPolicy:
        return JointTeacherPolicy(
            env,
            _policy_factory(
                base_model, policy_identity=base_policy_identity
            ),
            _searcher(base_model, config, base_policy_identity),
            query_stride_shots=args.query_stride,
            maximum_queries=args.maximum_queries,
        )

    ceiling_teacher, _, _ = _evaluate_suite(
        library=RUNTIME,
        label=CEILING_LABEL,
        seeds=ceiling_seeds,
        horizon=args.ceiling_ticks,
        factory=teacher_factory,
        policy_label="joint_teacher",
    )
    _validate_teacher_accounting(ceiling_teacher)
    report = {
        "schema": "irisu-joint-planner-development-report-v2",
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
        "source_identity": source,
        "execution": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "wall_seconds": time.perf_counter() - phase_wall_started,
            "cpu_seconds": time.process_time() - phase_cpu_started,
        },
        "runtime": {
            "path": str(RUNTIME),
            "sha256": RUNTIME_SHA256,
            "attestation": attestation,
        },
        "comparator": {
            "path": str(BASE),
            "checkpoint_sha256": BASE_SHA256,
            "policy_identity": base_policy_identity,
            "architecture_sha256": base_model.architecture_sha256,
            "schema_sha256": base_model.schema.sha256,
            "pointer_sha256": base_model.pointer_spec.sha256,
        },
        "planner": {
            "version": JOINT_PLANNER_VERSION,
            "config": config.manifest(),
            "config_sha256": config.sha256,
            "search_identity": searcher_identity,
            "query_stride_shots": args.query_stride,
            "maximum_queries_per_episode": args.maximum_queries,
        },
        "collection": {
            "suite": _suite_manifest(
                COLLECTION_LABEL, collection_seeds, args.collection_ticks
            ),
            "runner": collection_runner,
            "episodes": [
                value.manifest() for value in collection_episodes
            ],
            "aggregate": _aggregate(collection_episodes),
            "search_result_sha256s": collection_search_ids,
            "examples": len(dataset),
            "dataset_sha256": dataset.sha256,
            "teacher_evidence": {
                "path": str(teacher_evidence_path),
                "sha256": teacher_evidence_sha256,
                "search_results": len(collection_search_manifests),
                "examples": len(examples),
            },
            "shot_examples": sum(value.is_shot for value in dataset),
            "wall_seconds": collection_wall_seconds,
        },
        "student": {
            "architecture": final_model.manifest(),
            "architecture_sha256": final_model.architecture_sha256,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": checkpoint_sha256,
            },
            "selected_curve_index": selected_curve_index,
            "selected_training_steps": selected_training_steps,
            "selection_rule": (
                "fewest promotion-catastrophic tails, invalid actions, and "
                "gauge failures; then survival p10/median, positive gauge "
                "renewal gain, clears, score p10/median, and fewer steps"
            ),
            "curve_suite": _suite_manifest(
                CURVE_LABEL, curve_seeds, args.curve_ticks
            ),
            "curve_runner": curve_runner,
            "thresholds": _thresholds(args),
            "curves": curves,
        },
        "teacher_ceiling": {
            "suite": _suite_manifest(
                CEILING_LABEL, ceiling_seeds, args.ceiling_ticks
            ),
            "runner": ceiling_runner,
            "baseline": {
                "aggregate": _aggregate(ceiling_baseline),
                "episodes": [
                    value.manifest() for value in ceiling_baseline
                ],
            },
            "teacher": {
                "aggregate": _aggregate(ceiling_teacher),
                "episodes": [
                    value.manifest() for value in ceiling_teacher
                ],
            },
            "paired": _paired(ceiling_baseline, ceiling_teacher),
        },
    }
    result_path = output / "development-report.json"
    _assert_inputs_unchanged(source)
    if (
        _file_sha256(checkpoint_path) != checkpoint_sha256
        or _file_sha256(teacher_evidence_path) != teacher_evidence_sha256
    ):
        raise RuntimeError("joint development output identity changed")
    result_sha256 = _atomic_json(result_path, report, args.overwrite)
    print(
        json.dumps(
            {
                "report": str(result_path),
                "report_sha256": result_sha256,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_sha256,
                "dataset_sha256": dataset.sha256,
                "teacher_evidence_sha256": teacher_evidence_sha256,
                "selected_training_steps": selected_training_steps,
                "examples": len(dataset),
            },
            sort_keys=True,
        )
    )


def plateau(args: argparse.Namespace) -> None:
    phase_wall_started = time.perf_counter()
    phase_cpu_started = time.process_time()
    output = _safe_output_dir(args.output_dir)
    source = _source_identity()
    base_artifact = load_steering_checkpoint(BASE, expected_sha256=BASE_SHA256)
    base_policy_identity = _base_identity(source)
    student_path = args.student or output / "joint-student.pt"
    student_sha256 = args.student_sha256 or _file_sha256(student_path)
    student_artifact = load_steering_checkpoint(
        student_path, expected_sha256=student_sha256
    )
    config = _planner_config(args)
    expected_planner_identity = _searcher(
        base_artifact.model, config, base_policy_identity
    ).identity_manifest()
    metadata = student_artifact.metadata
    thresholds = _thresholds(args)
    if (
        metadata.get("source_identity") != source
        or metadata.get("runtime_sha256") != RUNTIME_SHA256
        or metadata.get("base_checkpoint_sha256") != BASE_SHA256
        or metadata.get("base_policy_identity") != base_policy_identity
        or metadata.get("planner_identity") != expected_planner_identity
        or metadata.get("training_thresholds") != thresholds
    ):
        raise ValueError(
            "joint student metadata does not match this source/runtime/planner"
        )
    development_path = student_path.resolve().parent / "development-report.json"
    with development_path.open("r", encoding="utf-8") as stream:
        development_report = json.load(stream)
    development_sha256 = _file_sha256(development_path)
    recorded_checkpoint = development_report.get("student", {}).get(
        "checkpoint", {}
    )
    recorded_evidence = development_report.get("collection", {}).get(
        "teacher_evidence", {}
    )
    evidence_path = Path(str(recorded_evidence.get("path", "")))
    if (
        development_report.get("source_identity") != source
        or recorded_checkpoint.get("sha256") != student_sha256
        or recorded_evidence.get("sha256")
        != metadata.get("teacher_evidence_sha256")
        or not evidence_path.is_file()
        or _file_sha256(evidence_path) != recorded_evidence.get("sha256")
        or development_report.get("student", {}).get("thresholds")
        != thresholds
    ):
        raise ValueError(
            "joint student is not chained to its completed development report"
        )
    suite = {
        "10k": (PLATEAU_10K_LABEL, PLATEAU_10K_SEEDS, 10_000),
        "20k": (PLATEAU_20K_LABEL, PLATEAU_20K_SEEDS, 20_000),
        "50k": (FEASIBILITY_50K_LABEL, FEASIBILITY_50K_SEEDS, 50_000),
    }[args.suite]
    label, seeds, horizon = suite
    baseline, runner, attestation = _evaluate_suite(
        library=RUNTIME,
        label=label,
        seeds=seeds,
        horizon=horizon,
        factory=lambda _env: _policy_factory(
            base_artifact.model, policy_identity=base_policy_identity
        ),
        policy_label="frozen_v5",
    )
    student, student_runner, student_attestation = _evaluate_suite(
        library=RUNTIME,
        label=label,
        seeds=seeds,
        horizon=horizon,
        factory=lambda _env: JointPlannerStudentPolicy(
            _policy_factory(
                base_artifact.model, policy_identity=base_policy_identity
            ),
            student_artifact.model,
            config=config,
            pair_confidence=args.pair_confidence,
            pair_margin=args.pair_margin,
            geometry_confidence=args.geometry_confidence,
            geometry_margin=args.geometry_margin,
        ),
        policy_label="joint_student",
    )
    if runner != student_runner or attestation != student_attestation:
        raise RuntimeError("paired plateau runtime identity changed")
    paired = _paired(baseline, student)
    gate = _promotion_gate(
        paired,
        baseline,
        student,
        selection_suite=args.suite in {"10k", "20k"},
    )
    report = {
        "schema": "irisu-joint-planner-plateau-report-v2",
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
        "source_identity": source,
        "execution": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "wall_seconds": time.perf_counter() - phase_wall_started,
            "cpu_seconds": time.process_time() - phase_cpu_started,
        },
        "suite": _suite_manifest(label, seeds, horizon),
        "runtime": {
            "path": str(RUNTIME),
            "sha256": RUNTIME_SHA256,
            "attestation": attestation,
            "runner": runner,
        },
        "comparator": {
            "path": str(BASE),
            "checkpoint_sha256": BASE_SHA256,
            "policy_identity": base_policy_identity,
        },
        "student": {
            "path": str(student_path.resolve()),
            "checkpoint_sha256": student_sha256,
            "development_report": {
                "path": str(development_path),
                "sha256": development_sha256,
            },
            "teacher_evidence": {
                "path": str(evidence_path),
                "sha256": recorded_evidence["sha256"],
            },
            "architecture_sha256": student_artifact.model.architecture_sha256,
            "planner_config": config.manifest(),
            "planner_config_sha256": config.sha256,
            "planner_identity": expected_planner_identity,
            "checkpoint_metadata": metadata,
            "thresholds": thresholds,
        },
        "baseline": {
            "aggregate": _aggregate(baseline),
            "episodes": [value.manifest() for value in baseline],
        },
        "candidate": {
            "aggregate": _aggregate(student),
            "episodes": [value.manifest() for value in student],
        },
        "paired": paired,
        "promotion_gate": gate,
    }
    path = output / f"plateau-{args.suite}.json"
    _assert_inputs_unchanged(source)
    if (
        _file_sha256(student_path) != student_sha256
        or _file_sha256(evidence_path) != recorded_evidence["sha256"]
        or _file_sha256(development_path) != development_sha256
    ):
        raise RuntimeError("joint plateau input artifact identity changed")
    digest = _atomic_json(path, report, args.overwrite)
    print(json.dumps({"report": str(path), "sha256": digest}, sort_keys=True))


def _positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _budgets(value: str) -> tuple[int, ...]:
    result = tuple(int(item) for item in value.split(","))
    if not result or tuple(sorted(set(result))) != result or result[0] < 1:
        raise argparse.ArgumentTypeError(
            "training budgets must be unique positive increasing integers"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("develop", "plateau"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pair-cap", type=_positive, default=3)
    parser.add_argument("--geometry-cap", type=_positive, default=4)
    parser.add_argument("--horizons", type=int, nargs="+", default=(48, 160))
    parser.add_argument("--planning-cooldown", type=_positive, default=16)
    parser.add_argument("--query-stride", type=_positive, default=2)
    parser.add_argument("--maximum-queries", type=_positive, default=48)
    parser.add_argument("--pair-confidence", type=float, default=0.0)
    parser.add_argument("--pair-margin", type=float, default=0.05)
    parser.add_argument("--geometry-confidence", type=float, default=0.0)
    parser.add_argument("--geometry-margin", type=float, default=0.05)
    parser.add_argument("--collection-seeds", type=_positive, default=8)
    parser.add_argument("--collection-ticks", type=_positive, default=2_000)
    parser.add_argument("--curve-seeds", type=_positive, default=6)
    parser.add_argument("--curve-ticks", type=_positive, default=2_000)
    parser.add_argument("--ceiling-seeds", type=_positive, default=8)
    parser.add_argument("--ceiling-ticks", type=_positive, default=2_000)
    parser.add_argument(
        "--training-budgets", type=_budgets, default=(50, 150, 400)
    )
    parser.add_argument("--batch-size", type=_positive, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--suite", choices=("10k", "20k", "50k"), default="10k")
    parser.add_argument("--student", type=Path)
    parser.add_argument("--student-sha256")
    args = parser.parse_args()
    _validate_suites()
    if _file_sha256(RUNTIME) != RUNTIME_SHA256:
        parser.error("trusted portable runtime identity changed")
    if _file_sha256(BASE) != BASE_SHA256:
        parser.error("frozen v5 comparator identity changed")
    if args.collection_seeds > len(COLLECTION_SEEDS):
        parser.error("collection seed count exceeds fixed suite")
    if args.curve_seeds > len(CURVE_SEEDS):
        parser.error("curve seed count exceeds fixed suite")
    if args.ceiling_seeds > len(CEILING_SEEDS):
        parser.error("ceiling seed count exceeds fixed suite")
    torch.set_num_threads(1)
    torch.manual_seed(TRAINING_SEED)
    develop(args) if args.phase == "develop" else plateau(args)


if __name__ == "__main__":
    main()
