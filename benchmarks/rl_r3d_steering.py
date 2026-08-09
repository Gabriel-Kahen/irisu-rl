#!/usr/bin/env python3
"""Development-only R3d relative-steering imitation benchmark.

This harness never reads sealed inputs and never writes canonical R3 evidence.
It collects closed-loop public-state demonstrations, fits an explicit
source-to-destination pair policy, then compares it with its teacher and the
legacy absolute-row matcher on fixed, disjoint development seeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import subprocess
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from irisu_env import Action, ActionKind, EventKind, IrisuEnv, MatcherShotPolicy
from irisu_pointer.action import PointerActionSpec
from irisu_pointer.archive import StrategicArchive
from irisu_pointer.archive_improvement import (
    ArchiveImprovementBinding,
    ArchiveImprovementConfig,
    SteeringBranchCandidate,
    collect_archive_improvement,
)
from irisu_pointer.evaluate import DevelopmentSuite
from irisu_pointer.replay_supervision import SteeringConversionMetrics
from irisu_pointer.steering import (
    ClosedLoopSteeringExpert,
    SteeringDecision,
    SteeringExpertConfig,
)
from irisu_pointer.steering_checkpoint import (
    load_goal_conditioned_steering_policy,
    save_steering_checkpoint,
)
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
    SteeringDataset,
    SteeringExample,
    SteeringModelConfig,
    steering_example_from_decision,
    train_goal_conditioned_steering,
)
from irisu_pointer.strategic import (
    CurriculumMetrics,
    available_intents,
    evaluate_curriculum,
    extract_strategic_features,
    strategic_potential,
)
from irisu_rl.schema import TEACHER_V1


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs/rl/experiments/r3d-relative-steering-v1.toml"
)
TRUSTED_PORTABLE = (
    ROOT
    / "artifacts/r3/runtime/main-0c48dba-20260723/portable-build/libirisu_clone.so"
)
DEMONSTRATION_SEEDS = (
    0x0D3A0001,
    0x0D3A0002,
    0x0D3A0003,
    0x0D3A0004,
    0x0D3A0005,
    0x0D3A0006,
    0x0D3A0007,
    0x0D3A0008,
    0x435F4DBA,
    0xF8B2D88D,
    0xB12B167F,
    0x89B55527,
    0x23715B20,
    0x98E56B27,
    0x50B1818E,
    0xABA6938C,
    0xE042D419,
    0xAAB35627,
    0x564FE2D7,
    0xC5D7026F,
    0x93F27BF3,
    0x4239EE58,
    0x8FCC8590,
    0xB97E0F41,
)
UNSEEN_DEVELOPMENT_SEEDS = (
    0x13579BDF,
    0x2468ACE0,
    0x31415926,
    0x5A17C0DE,
    0x6C8E9CF1,
    0x7B1D3F59,
    0x8D2E4A60,
    0xA5C31E79,
    0xB4D72E13,
    0xC61A9047,
    0xD83F5B29,
    0xE9274C61,
    0xF15B8D03,
    0x19C4E7A5,
    0x2BD6F819,
    0x47E10AC3,
)
SURVIVAL_HOLDOUT_SEEDS = (
    0x390A6C20,
    0xA8B6C2AB,
    0xBAD0911C,
    0x22783142,
    0x53F04ED4,
    0xAC3C94AA,
    0x44168882,
    0x0E029A7B,
    0xB2A6B128,
    0x6AF242CE,
    0x4BC721BB,
    0x49EFD0E3,
    0xBE70B5C9,
    0x2177CD53,
    0xB9EFAB3F,
    0x255604E3,
)
EVALUATION_SEED_SUITES = {
    "development": (
        "r3d-fixed-unseen-development-v1",
        UNSEEN_DEVELOPMENT_SEEDS,
    ),
    "survival-holdout": (
        "r3d-survival-holdout-development-v1",
        SURVIVAL_HOLDOUT_SEEDS,
    ),
}
MAX_DECISIONS_PER_EPISODE = 2_000_000
_FORBIDDEN_PATH = re.compile(
    r"(?:^|[/_.-])(?:sealed|test)(?:$|[/_.-])", re.IGNORECASE
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
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str


def _reject_path(path: Path, name: str) -> None:
    text = str(path.expanduser().absolute()).replace("\\", "/")
    if _FORBIDDEN_PATH.search(text):
        raise ValueError(f"{name} must not reference sealed or test material")
    if "/artifacts/r3/runs/" in text:
        raise ValueError(f"{name} must not reference canonical R3 run storage")


def _output_path(path: Path, name: str, suffix: str) -> Path:
    _reject_path(path, name)
    expanded = path.expanduser()
    if expanded.suffix != suffix:
        raise ValueError(f"{name} must use the {suffix} suffix")
    if expanded.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link")
    resolved = expanded.resolve()
    _reject_path(resolved, name)
    protected = (
        ROOT / ".git",
        ROOT / "benchmarks",
        ROOT / "configs",
        ROOT / "docs",
        ROOT / "python",
        ROOT / "tests",
    )
    if any(root == resolved or root in resolved.parents for root in protected):
        raise ValueError(f"{name} must not overwrite repository source")
    if resolved == TRUSTED_PORTABLE.resolve():
        raise ValueError(f"{name} must not overwrite the portable runtime")
    return resolved


def _snapshot_file(path: Path, name: str) -> _FileSnapshot:
    _reject_path(path, name)
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link")
    resolved = expanded.resolve(strict=True)
    _reject_path(resolved, name)
    before = resolved.stat()
    digest = _file_sha256(resolved)
    after = resolved.stat()
    first = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    second = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if first != second or not resolved.is_file():
        raise RuntimeError(f"{name} changed while it was identified")
    return _FileSnapshot(resolved, *map(int, second), digest)


def _require_unchanged(snapshot: _FileSnapshot, name: str) -> None:
    if _snapshot_file(snapshot.path, name) != snapshot:
        raise RuntimeError(f"{name} changed during the benchmark")


def _source_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity(config_path: Path) -> dict[str, object]:
    pointer_files = sorted((ROOT / "python/irisu_pointer").glob("*.py"))
    environment_files = sorted((ROOT / "python/irisu_env").glob("*.py"))
    support_files = (
        ROOT / "python/irisu_rl/__init__.py",
        ROOT / "python/irisu_rl/actions.py",
        ROOT / "python/irisu_rl/encoding.py",
        ROOT / "python/irisu_rl/schema.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    )
    files = (
        Path(__file__).resolve(),
        config_path,
        *pointer_files,
        *environment_files,
        *support_files,
    )
    manifest = {
        "schema": "irisu-r3d-relative-steering-source-v2",
        "git_revision": _source_revision(),
        "files": {
            str(path.relative_to(ROOT)): _file_sha256(path) for path in files
        },
    }
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _require_source_identity(
    expected: Mapping[str, object], config_path: Path
) -> None:
    if _source_identity(config_path) != dict(expected):
        raise RuntimeError("R3d source identity changed during the benchmark")


def _load_config(snapshot: _FileSnapshot) -> dict[str, Any]:
    value = tomllib.loads(snapshot.path.read_text(encoding="utf-8"))
    required = {
        "version": "r3d-relative-steering-v1",
        "status": "development_only_not_canonical_evidence",
        "deployable": False,
        "canonical_r3_evidence": False,
        "sealed_evaluation_allowed": False,
        "selection_backend": "portable",
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("R3d config weakens development-only evidence boundaries")
    trusted = (ROOT / str(value.get("trusted_runtime", ""))).resolve()
    if trusted != TRUSTED_PORTABLE.resolve():
        raise ValueError("R3d config does not bind the trusted portable runtime")
    return value


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


def _strategic_gauge_valid(observation: Mapping[str, Any]) -> bool:
    gauge = int(observation.get("gauge", 0))
    gauge_max = int(observation.get("gauge_max", 0))
    return gauge_max > 0 and 0 <= gauge <= gauge_max


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    policy: str
    seed: int
    decisions: int
    primitive_actions: int
    highest_chain: int
    qualifying_clears: int
    final_level: int
    final_gauge: int
    gauge_max: int
    archive_insertions: int
    archive_rejections: int
    conversion: SteeringConversionMetrics

    @property
    def gauge_failure(self) -> bool:
        # The public terminal observation clamps a depleted gauge to one.
        return self.conversion.terminated and self.final_gauge <= 1

    def manifest(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "seed": self.seed,
            "decisions": self.decisions,
            "primitive_actions": self.primitive_actions,
            "highest_chain": self.highest_chain,
            "qualifying_clears": self.qualifying_clears,
            "final_level": self.final_level,
            "final_gauge": self.final_gauge,
            "gauge_max": self.gauge_max,
            "gauge_failure": self.gauge_failure,
            "archive_insertions": self.archive_insertions,
            "archive_rejections": self.archive_rejections,
            "conversion": self.conversion.manifest(),
        }


def _run_episode(
    env: IrisuEnv,
    policy: object,
    *,
    label: str,
    seed: int,
    config_hash: int,
    decision_hook: Callable[[Mapping[str, Any], object], None] | None = None,
    archive: StrategicArchive | None = None,
    archive_stride_ticks: int = 128,
    max_decisions: int = MAX_DECISIONS_PER_EPISODE,
) -> EpisodeResult:
    if (
        isinstance(max_decisions, bool)
        or not isinstance(max_decisions, int)
        or max_decisions < 1
    ):
        raise ValueError("R3d maximum decision count must be positive")
    getattr(policy, "reset")(seed)
    observation, reset_info = env.reset(seed=seed)
    if int(reset_info.get("config_hash", -1)) != config_hash:
        raise RuntimeError("R3d reset config identity mismatch")
    initial_tick = int(observation.get("tick", 0))
    terminated = bool(observation.get("terminated", False))
    truncated = bool(observation.get("truncated", False))
    events: Counter[int] = Counter()
    hit_projectiles: set[int] = set()
    decisions = 0
    primitive_actions = 0
    requested_shots = 0
    highest_chain = int(observation.get("highest_chain", 0))
    initial_qualifying_clears = int(
        observation.get("qualifying_clear_count", 0)
    )
    archive_insertions = 0
    archive_rejections = 0
    next_archive_tick = initial_tick

    while not terminated and not truncated:
        if decisions >= max_decisions:
            raise RuntimeError("R3d episode exceeded its fail-closed decision budget")
        predict = getattr(policy, "predict", None)
        decision = (
            predict(observation)
            if callable(predict)
            else getattr(policy, "act")(observation)
        )
        if decision_hook is not None:
            decision_hook(observation, decision)
        if isinstance(decision, SteeringDecision):
            actions = decision.primitive_actions()
        elif isinstance(decision, Action):
            actions = (decision,)
        elif (
            isinstance(decision, tuple)
            and decision
            and all(isinstance(value, Action) for value in decision)
        ):
            actions = decision
        else:
            raise TypeError("R3d policy returned an unsupported decision")
        if (
            archive is not None
            and isinstance(decision, SteeringDecision)
            and decision.is_shot
            and int(observation.get("tick", 0)) >= next_archive_tick
        ):
            tick = int(observation.get("tick", 0))
            if _strategic_gauge_valid(observation):
                _, inserted = archive.capture(
                    observation,
                    env.clone_state(),
                    trajectory_identity=(
                        f"{label}:{seed:08x}:{tick}:pre-shot"
                    ),
                )
                archive_insertions += int(inserted)
            else:
                archive_rejections += 1
            while next_archive_tick <= tick:
                next_archive_tick += archive_stride_ticks
        decisions += 1
        for action in actions:
            if terminated or truncated:
                break
            before_tick = int(observation.get("tick", 0))
            requested_shots += int(
                ActionKind.parse(action.kind) is not ActionKind.WAIT
            )
            observation, _, terminated, truncated, info = env.step(action)
            primitive_actions += 1
            if int(info.get("config_hash", -1)) != config_hash:
                raise RuntimeError("R3d step config identity mismatch")
            if int(observation.get("tick", 0)) <= before_tick:
                raise RuntimeError("R3d action failed to advance simulator time")
            for event in info.get("events", ()):
                if not isinstance(event, Mapping):
                    continue
                kind = _event_kind(event)
                if kind is None:
                    continue
                events[kind] += 1
                if kind == int(EventKind.PROJECTILE_HIT):
                    hit_projectiles.add(int(event.get("a", -1)))
            highest_chain = max(
                highest_chain, int(observation.get("highest_chain", 0))
            )
    final_tick = int(observation.get("tick", 0))
    qualifying_clears = (
        int(observation.get("qualifying_clear_count", 0))
        - initial_qualifying_clears
    )
    if qualifying_clears < 0:
        raise RuntimeError("R3d qualifying clear counter moved backwards")
    shots_fired = events[int(EventKind.SHOT_FIRED)]
    hit_projectiles.discard(-1)
    conversion = SteeringConversionMetrics(
        frames=primitive_actions,
        survival_ticks=final_tick - initial_tick,
        requested_shots=requested_shots,
        shots_fired=shots_fired,
        shots_hit=len(hit_projectiles),
        projectile_hit_events=events[int(EventKind.PROJECTILE_HIT)],
        chain_joins=events[int(EventKind.CHAIN_JOINED)],
        clears=events[int(EventKind.CLEARED)],
        rotten=events[int(EventKind.ROTTEN)],
        ejected=events[int(EventKind.EJECTED)],
        invalid_actions=events[int(EventKind.INVALID_ACTION)],
        final_score=int(observation.get("score", 0)),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )
    return EpisodeResult(
        label,
        seed,
        decisions,
        primitive_actions,
        highest_chain,
        qualifying_clears,
        int(observation.get("level", 0)),
        int(observation.get("gauge", 0)),
        int(observation.get("gauge_max", 0)),
        archive_insertions,
        archive_rejections,
        conversion,
    )


def _percentile(values: Sequence[int], fraction: float) -> float:
    ordered = sorted(int(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    return {
        "minimum": min(values),
        "p10": _percentile(values, 0.10),
        "median": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "maximum": max(values),
    }


def _aggregate(episodes: Sequence[EpisodeResult]) -> dict[str, object]:
    shots = sum(value.conversion.shots_fired for value in episodes)
    hits = sum(value.conversion.shots_hit for value in episodes)
    joins = sum(value.conversion.chain_joins for value in episodes)
    clears = sum(value.conversion.clears for value in episodes)
    rate = lambda numerator: numerator / shots if shots else 0.0
    return {
        "episodes": len(episodes),
        "raw_score": _distribution(
            [value.conversion.final_score for value in episodes]
        ),
        "survival_ticks": _distribution(
            [value.conversion.survival_ticks for value in episodes]
        ),
        "highest_chain": _distribution(
            [value.highest_chain for value in episodes]
        ),
        "qualifying_clears": sum(
            value.qualifying_clears for value in episodes
        ),
        "conversion": {
            "shots_fired": shots,
            "shots_hit": hits,
            "projectile_hit_events": sum(
                value.conversion.projectile_hit_events for value in episodes
            ),
            "chain_joins": joins,
            "clears": clears,
            "rotten": sum(value.conversion.rotten for value in episodes),
            "ejected": sum(value.conversion.ejected for value in episodes),
            "invalid_actions": sum(
                value.conversion.invalid_actions for value in episodes
            ),
            "shot_hit_rate": rate(hits),
            "chain_joins_per_shot": rate(joins),
            "cleared_events_per_shot": rate(clears),
            # Compatibility alias. CLEARED events are not qualifying clears.
            "clears_per_shot": rate(clears),
        },
        "gauge_failures": sum(value.gauge_failure for value in episodes),
        "completion_rate": sum(
            value.conversion.terminated or value.conversion.truncated
            for value in episodes
        )
        / len(episodes),
    }


def _curriculum(
    episodes: Sequence[EpisodeResult],
    baseline: Mapping[str, object],
) -> dict[str, object]:
    score_distribution = baseline["raw_score"]
    survival_distribution = baseline["survival_ticks"]
    assert isinstance(score_distribution, Mapping)
    assert isinstance(survival_distribution, Mapping)
    metrics = CurriculumMetrics(
        episodes=len(episodes),
        shots=sum(value.conversion.shots_fired for value in episodes),
        projectile_hits=sum(value.conversion.shots_hit for value in episodes),
        chain_joins=sum(value.conversion.chain_joins for value in episodes),
        qualifying_clears=sum(value.qualifying_clears for value in episodes),
        raw_scores=tuple(value.conversion.final_score for value in episodes),
        survival_ticks=tuple(
            value.conversion.survival_ticks for value in episodes
        ),
        highest_chains=tuple(value.highest_chain for value in episodes),
        gauge_failures=sum(value.gauge_failure for value in episodes),
        baseline_median_score=float(score_distribution["median"]),
        baseline_median_survival=float(survival_distribution["median"]),
    )
    gates = evaluate_curriculum(metrics)
    return {
        "metrics": {
            **asdict(metrics),
            "hit_rate": metrics.hit_rate,
            "joins_per_shot": metrics.joins_per_shot,
            "qualifying_clears_per_shot": metrics.clears_per_shot,
            # Compatibility alias used by the existing curriculum gate schema.
            "clears_per_shot": metrics.clears_per_shot,
            "median_score": metrics.median_score,
            "median_survival": metrics.median_survival,
            "gauge_failure_rate": metrics.gauge_failure_rate,
        },
        "gates": [
            {
                "stage": result.stage.value,
                "passed": result.passed,
                "failed_metrics": list(result.failed_metrics),
            }
            for result in gates
        ],
    }


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _resolved_count(
    argument: int | None, profile: Mapping[str, Any], name: str
) -> int:
    return _positive_int(profile[name] if argument is None else argument, name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--library", type=Path, default=TRUSTED_PORTABLE)
    parser.add_argument("--profile", choices=("fast", "long"), default="fast")
    parser.add_argument("--demonstration-seeds", type=int)
    parser.add_argument("--demonstration-ticks", type=int)
    parser.add_argument("--evaluation-seeds", type=int)
    parser.add_argument("--evaluation-ticks", type=int)
    parser.add_argument(
        "--evaluation-suite",
        choices=tuple(EVALUATION_SEED_SUITES),
        default="development",
    )
    parser.add_argument("--training-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--policy-out",
        type=Path,
        default=Path("/tmp/r3d-relative-steering-dev.pt"),
    )
    parser.add_argument("--result-out", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    config_snapshot = _snapshot_file(args.config, "R3d config")
    config = _load_config(config_snapshot)
    profile = config["profiles"][args.profile]
    demo_seed_count = _resolved_count(
        args.demonstration_seeds, profile, "demonstration_seeds"
    )
    demo_ticks = _resolved_count(
        args.demonstration_ticks, profile, "demonstration_ticks"
    )
    eval_seed_count = _resolved_count(
        args.evaluation_seeds, profile, "evaluation_seeds"
    )
    eval_ticks = _resolved_count(
        args.evaluation_ticks, profile, "evaluation_ticks"
    )
    training_steps = _resolved_count(
        args.training_steps, profile, "training_steps"
    )
    batch_size = _resolved_count(args.batch_size, profile, "batch_size")
    if demo_seed_count > len(DEMONSTRATION_SEEDS):
        parser.error("demonstration-seeds exceeds the fixed training suite")
    suite_label, suite_seeds = EVALUATION_SEED_SUITES[args.evaluation_suite]
    if eval_seed_count > len(suite_seeds):
        parser.error("evaluation-seeds exceeds the fixed development suite")
    demo_seeds = DEMONSTRATION_SEEDS[:demo_seed_count]
    evaluation_seeds = suite_seeds[:eval_seed_count]
    if set(demo_seeds) & set(evaluation_seeds):
        raise RuntimeError("demonstration and unseen development seeds overlap")
    evaluation_suite = DevelopmentSuite(
        label=suite_label,
        seeds=evaluation_seeds,
        config=(
            ("max_episode_ticks", eval_ticks),
            ("physics_backend", "portable"),
        ),
        max_decisions_per_episode=MAX_DECISIONS_PER_EPISODE,
    )

    runtime_snapshot = _snapshot_file(args.library, "portable runtime")
    if runtime_snapshot.path != TRUSTED_PORTABLE.resolve():
        raise ValueError("R3d benchmark requires the trusted portable runtime")
    policy_out = _output_path(
        args.policy_out, "steering checkpoint output", ".pt"
    )
    result_out = (
        None
        if args.result_out is None
        else _output_path(args.result_out, "R3d result output", ".json")
    )
    source_identity = _source_identity(config_snapshot.path)
    config_sha256 = config_snapshot.sha256
    torch_threads = _positive_int(
        int(config["training"]["torch_threads"]), "torch_threads"
    )
    torch.set_num_threads(torch_threads)
    training_seed = int(config["training"]["seed"])
    torch.manual_seed(training_seed)

    controller_config = SteeringExpertConfig(**config["controller"])
    act_logit_bias = float(config["inference"]["act_logit_bias"])
    if not math.isfinite(act_logit_bias):
        raise ValueError("R3d inference act-logit bias must be finite")
    pointer_spec = PointerActionSpec()
    model_config = SteeringModelConfig(**config["model"])
    demonstration_config = {"max_episode_ticks": demo_ticks}
    archive: StrategicArchive | None = None
    archive_evidence: dict[str, object] | None = None
    demonstration_environment_config: Mapping[str, Any] | None = None
    archive_stride = _positive_int(
        int(config["archive"]["capture_stride_ticks"]),
        "archive capture stride",
    )
    examples: list[SteeringExample] = []
    strategic_intents: Counter[str] = Counter()
    strategic_potentials: list[float] = []
    pressure_decisions = 0
    rejected_strategic_gauges: list[int] = []
    model_managed_cooldown_waits = 0
    demonstration_episodes: list[EpisodeResult] = []

    with IrisuEnv(
        library_path=runtime_snapshot.path,
        physics_backend="portable",
        config=demonstration_config,
    ) as env:
        if Path(env.library_path).resolve() != runtime_snapshot.path:
            raise RuntimeError("R3d demonstration loaded a foreign runtime")
        demonstration_runner = env.runner_identity_manifest()
        demonstration_environment_config = env.config
        config_hash = int(demonstration_runner["config_hash"])
        archive_evidence = {
            "source_identity": source_identity["sha256"],
            "runtime_sha256": runtime_snapshot.sha256,
            "config_sha256": config_sha256,
            "runner": demonstration_runner,
        }
        archive = StrategicArchive(
            source_identity=_canonical_sha256(archive_evidence)
        )
        for seed in demo_seeds:
            provenance = _canonical_sha256(
                {
                    "source_identity": source_identity["sha256"],
                    "runtime_sha256": runtime_snapshot.sha256,
                    "config_sha256": config_sha256,
                    "runner": demonstration_runner,
                    "seed": seed,
                }
            )

            def collect(
                observation: Mapping[str, Any],
                decision: object,
                *,
                episode_seed: int = seed,
                episode_provenance: str = provenance,
            ) -> None:
                nonlocal model_managed_cooldown_waits
                nonlocal pressure_decisions
                if not isinstance(decision, SteeringDecision):
                    raise TypeError("demonstration policy did not expose its decision")
                if (
                    not decision.is_shot
                    and decision.reason
                    == "observe the previous correction before acting again"
                ):
                    # Inference enforces this cooldown before consulting the
                    # model. Training it as strategic restraint caused the act
                    # head to repeat waits after the cooldown had already ended.
                    model_managed_cooldown_waits += 1
                else:
                    example = steering_example_from_decision(
                        observation,
                        decision,
                        episode_identity=f"r3d-demo:{episode_seed:08x}",
                        provenance_sha256=episode_provenance,
                        pointer_spec=pointer_spec,
                        require_representable_template=False,
                    )
                    if example is not None:
                        examples.append(example)
                if not _strategic_gauge_valid(observation):
                    rejected_strategic_gauges.append(
                        int(observation.get("gauge", 0))
                    )
                    return
                features = extract_strategic_features(observation)
                pressure_decisions += int(features.under_pressure)
                strategic_potentials.append(strategic_potential(features).total)
                for intent in available_intents(features):
                    strategic_intents[intent.value] += 1

            demonstration_episodes.append(
                _run_episode(
                    env,
                    ClosedLoopSteeringExpert(config=controller_config),
                    label="closed_loop_demonstrator",
                    seed=seed,
                    config_hash=config_hash,
                    decision_hook=collect,
                    archive=archive,
                    archive_stride_ticks=archive_stride,
                )
            )

    if (
        not examples
        or archive is None
        or archive_evidence is None
        or demonstration_environment_config is None
    ):
        raise RuntimeError("R3d demonstrator produced incomplete steering evidence")
    demonstration_example_count = len(examples)
    improvement_binding = ArchiveImprovementBinding.create(
        archive,
        environment_config_sha256=_canonical_sha256(
            demonstration_environment_config
        ),
        runner_identity_sha256=_canonical_sha256(demonstration_runner),
        runtime_sha256=runtime_snapshot.sha256,
    )
    steering_branches = (
        SteeringBranchCandidate("baseline", controller_config),
        SteeringBranchCandidate(
            "close-contact",
            replace(controller_config, impact_side_sizes=0.25),
        ),
        SteeringBranchCandidate(
            "wide-contact",
            replace(controller_config, impact_side_sizes=0.75),
        ),
        SteeringBranchCandidate(
            "lower-contact",
            replace(controller_config, impact_below_sizes=1.0),
        ),
        SteeringBranchCandidate(
            "rotten-destination-disabled",
            replace(controller_config, enable_rotten_matching=False),
        ),
        SteeringBranchCandidate(
            "bonus-activation",
            replace(controller_config, enable_bonus=True),
        ),
    )
    improvement_config = ArchiveImprovementConfig(
        horizon_ticks=_positive_int(
            int(config["archive"]["improvement_horizon_ticks"]),
            "archive improvement horizon",
        ),
        max_elites=_positive_int(
            int(config["archive"]["improvement_max_elites"]),
            "archive improvement elite limit",
        ),
        wait_candidates=tuple(
            int(value) for value in config["archive"]["wait_candidates"]
        ),
        steering_branches=steering_branches,
        continuation_config=controller_config,
    )
    improvement = collect_archive_improvement(
        archive,
        lambda: IrisuEnv(
            library_path=runtime_snapshot.path,
            physics_backend="portable",
            config=demonstration_config,
        ),
        binding=improvement_binding,
        config=improvement_config,
        pointer_spec=pointer_spec,
    )
    examples.extend(improvement.examples)
    dataset = SteeringDataset(examples)
    model = GoalConditionedSteeringModel(
        TEACHER_V1, pointer_spec=pointer_spec, config=model_config
    )
    training_report = train_goal_conditioned_steering(
        model,
        dataset,
        steps=training_steps,
        batch_size=batch_size,
        learning_rate=float(config["training"]["learning_rate"]),
        seed=training_seed,
    )
    _require_source_identity(source_identity, config_snapshot.path)
    checkpoint_sha256 = save_steering_checkpoint(
        policy_out,
        model,
        metadata={
            "development_only": True,
            "canonical_r3_evidence": False,
            "sealed_test_material_used": False,
            "source_identity": source_identity,
            "runtime_sha256": runtime_snapshot.sha256,
            "config_sha256": config_sha256,
            "demonstration_seeds": list(demo_seeds),
            "demonstration_runner": demonstration_runner,
            "demonstration_examples": demonstration_example_count,
            "archive_improvement_examples": len(improvement.examples),
            "total_training_examples": len(examples),
            "training_dataset_sha256": dataset.sha256,
            "strategic_archive_sha256": archive.sha256,
            "strategic_archive_evidence": archive_evidence,
            "archive_improvement_report_sha256": improvement.report.sha256,
            "execution_environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "torch_threads": torch.get_num_threads(),
            },
            "training_report": asdict(training_report),
        },
        overwrite=args.overwrite,
    )

    evaluation_config = {"max_episode_ticks": eval_ticks}
    policies: tuple[tuple[str, Callable[[], object]], ...] = (
        (
            "closed_loop_controller",
            lambda: ClosedLoopSteeringExpert(config=controller_config),
        ),
        (
            "learned_pair_imitator",
            lambda: load_goal_conditioned_steering_policy(
                policy_out,
                expected_sha256=checkpoint_sha256,
                cooldown_ticks=controller_config.observe_ticks,
                minimum_pair_closure_sizes=(
                    controller_config.minimum_pair_closure_sizes
                ),
                impact_side_sizes=controller_config.impact_side_sizes,
                impact_below_sizes=controller_config.impact_below_sizes,
                source_velocity_lead_ticks=(
                    controller_config.source_velocity_lead_ticks
                ),
                ticks_per_second=controller_config.ticks_per_second,
                act_logit_bias=act_logit_bias,
            ),
        ),
        (
            "legacy_matcher",
            lambda: MatcherShotPolicy(**config["legacy"]),
        ),
    )
    evaluations: dict[str, list[EpisodeResult]] = {
        label: [] for label, _ in policies
    }
    with IrisuEnv(
        library_path=runtime_snapshot.path,
        physics_backend="portable",
        config=evaluation_config,
    ) as env:
        if Path(env.library_path).resolve() != runtime_snapshot.path:
            raise RuntimeError("R3d evaluation loaded a foreign runtime")
        evaluation_runner = env.runner_identity_manifest()
        config_hash = int(evaluation_runner["config_hash"])
        for label, factory in policies:
            for seed in evaluation_seeds:
                policy = factory()
                if isinstance(policy, GoalConditionedSteeringPolicy):
                    if (
                        policy.artifact_sha256 != checkpoint_sha256
                        or policy.schema_sha256 != TEACHER_V1.sha256
                        or policy.pointer_action_sha256 != pointer_spec.sha256
                    ):
                        raise RuntimeError("loaded R3d policy identity mismatch")
                evaluations[label].append(
                    _run_episode(
                        env,
                        policy,
                        label=label,
                        seed=seed,
                        config_hash=config_hash,
                        max_decisions=(
                            evaluation_suite.max_decisions_per_episode
                        ),
                    )
                )

    aggregates = {
        label: _aggregate(episodes)
        for label, episodes in evaluations.items()
    }
    baseline = aggregates["legacy_matcher"]
    curriculum = {
        label: _curriculum(episodes, baseline)
        for label, episodes in evaluations.items()
    }
    controller_score = aggregates["closed_loop_controller"]["raw_score"]
    learned_score = aggregates["learned_pair_imitator"]["raw_score"]
    legacy_score = baseline["raw_score"]
    assert isinstance(controller_score, Mapping)
    assert isinstance(learned_score, Mapping)
    assert isinstance(legacy_score, Mapping)
    _require_source_identity(source_identity, config_snapshot.path)
    _require_unchanged(runtime_snapshot, "portable runtime")
    _require_unchanged(config_snapshot, "R3d config")
    learned_policy_identity = _canonical_sha256(
        {
            "type": "goal-conditioned-steering-policy-v1",
            "checkpoint_sha256": checkpoint_sha256,
            "cooldown_ticks": controller_config.observe_ticks,
            "minimum_pair_closure_sizes": (
                controller_config.minimum_pair_closure_sizes
            ),
            "impact_side_sizes": controller_config.impact_side_sizes,
            "impact_below_sizes": controller_config.impact_below_sizes,
            "source_velocity_lead_ticks": (
                controller_config.source_velocity_lead_ticks
            ),
            "ticks_per_second": controller_config.ticks_per_second,
            "act_logit_bias": act_logit_bias,
            "impact_rule": "analytic-continuous-opposite-side-and-below-v2",
            "schema_sha256": TEACHER_V1.sha256,
            "pointer_action_sha256": pointer_spec.sha256,
        }
    )

    output = {
        "schema": "irisu-r3d-relative-steering-development-run-v2",
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
        "source_identity": source_identity,
        "config": {
            "path": str(config_snapshot.path),
            "sha256": config_sha256,
            "version": config["version"],
            "profile": args.profile,
        },
        "runtime": {
            "path": str(runtime_snapshot.path),
            "sha256": runtime_snapshot.sha256,
            "backend": "portable",
            "demonstration_runner": demonstration_runner,
            "evaluation_runner": evaluation_runner,
        },
        "policy_identities": {
            "closed_loop_controller": _canonical_sha256(
                {
                    "type": "closed-loop-steering-expert-v1",
                    "source_identity": source_identity["sha256"],
                    "controller_config": asdict(controller_config),
                    "pointer_action_sha256": pointer_spec.sha256,
                }
            ),
            "learned_pair_imitator": learned_policy_identity,
            "legacy_matcher": _canonical_sha256(
                {
                    "type": "irisu-env-matcher-shot-policy",
                    "source_identity": source_identity["sha256"],
                    "config": config["legacy"],
                }
            ),
        },
        "run_parameters": {
            "demonstration_seeds": list(demo_seeds),
            "demonstration_ticks": demo_ticks,
            "evaluation_seeds": list(evaluation_seeds),
            "evaluation_ticks": eval_ticks,
            "evaluation_suite": args.evaluation_suite,
            "training_steps": training_steps,
            "batch_size": batch_size,
            "training_seed": training_seed,
        },
        "evaluation_suite": {
            "manifest": evaluation_suite.manifest(),
            "sha256": evaluation_suite.sha256,
        },
        "demonstration": {
            "examples": demonstration_example_count,
            "episodes": [value.manifest() for value in demonstration_episodes],
            "aggregate": _aggregate(demonstration_episodes),
            "strategic": {
                "available_intent_counts": dict(sorted(strategic_intents.items())),
                "under_pressure_decisions": pressure_decisions,
                "model_managed_cooldown_waits_excluded": (
                    model_managed_cooldown_waits
                ),
                "rejected_out_of_range_gauge_decisions": len(
                    rejected_strategic_gauges
                ),
                "minimum_rejected_gauge": min(
                    rejected_strategic_gauges, default=0
                ),
                "potential_minimum": min(strategic_potentials),
                "potential_maximum": max(strategic_potentials),
            },
            "archive": {
                "cells": len(archive),
                "sha256": archive.sha256,
                "evidence": archive_evidence,
                "evidence_sha256": _canonical_sha256(archive_evidence),
                "manifest": archive.manifest(),
            },
            "archive_improvement": {
                "examples": len(improvement.examples),
                "report_sha256": improvement.report.sha256,
                "report": improvement.report.manifest(),
            },
        },
        "training": {
            "examples": len(dataset),
            "model": model.manifest(),
            "dataset_schema_sha256": dataset.schema.sha256,
            "dataset_sha256": dataset.sha256,
            "pointer_action_sha256": dataset.pointer_spec.sha256,
            "report": asdict(training_report),
            "checkpoint": {
                "path": str(policy_out),
                "sha256": checkpoint_sha256,
            },
        },
        "evaluation": {
            label: {
                "episodes": [value.manifest() for value in episodes],
                "aggregate": aggregates[label],
                "curriculum": curriculum[label],
            }
            for label, episodes in evaluations.items()
        },
        "comparison": {
            "controller_median_score_delta_vs_legacy": float(
                controller_score["median"]
            )
            - float(legacy_score["median"]),
            "learned_median_score_delta_vs_legacy": float(
                learned_score["median"]
            )
            - float(legacy_score["median"]),
            "learned_median_score_retention_vs_controller": (
                float(learned_score["median"]) / float(controller_score["median"])
                if float(controller_score["median"])
                else 0.0
            ),
        },
        "execution": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
        },
    }
    encoded = json.dumps(output, sort_keys=True, indent=2, allow_nan=False)
    if result_out is not None:
        if result_out.exists() and not args.overwrite:
            raise FileExistsError(
                f"refusing to replace R3d result: {result_out}"
            )
        result_out.parent.mkdir(parents=True, exist_ok=True)
        result_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
