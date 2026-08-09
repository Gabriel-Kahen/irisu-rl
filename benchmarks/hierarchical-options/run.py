#!/usr/bin/env python3
"""Development-only offline option-value fitting and paired endurance A/Bs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHON = ROOT / "python"
HERE = Path(__file__).resolve().parent
for path in (PYTHON, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from irisu_env import Action, ActionKind, EventKind, IrisuEnv  # noqa: E402
from irisu_pointer.steering import (  # noqa: E402
    ClosedLoopSteeringExpert,
    SteeringDecision,
    SteeringExpertConfig,
)
from irisu_pointer.steering_checkpoint import (  # noqa: E402
    load_goal_conditioned_steering_policy,
)
from irisu_rl.seeds import SeedAllocator  # noqa: E402

from core import (  # noqa: E402
    FEATURE_NAMES,
    OPTION_ORDER,
    HierarchicalOptionPolicy,
    Option,
    OptionValueModel,
    UtilityWeights,
    applicable_options,
    branch_utility,
    event_signature,
    feature_vector,
)


DEFAULT_CONFIG = (
    ROOT / "configs/rl/experiments/hierarchical-options-v1.toml"
)
_FORMAT = "irisu-hierarchical-options-development-v1"
_FORBIDDEN_PATH = re.compile(
    r"(?:^|[/_.-])(?:sealed|test|canonical)(?:$|[/_.-])",
    re.IGNORECASE,
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_path(path: Path, name: str) -> None:
    text = str(path.expanduser().absolute()).replace("\\", "/")
    if _FORBIDDEN_PATH.search(text) or "/artifacts/r3/runs/" in text:
        raise ValueError(f"{name} references forbidden evaluation material")


def _input(path: Path, name: str) -> Path:
    _reject_path(path, name)
    if path.expanduser().is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    resolved = path.expanduser().resolve(strict=True)
    _reject_path(resolved, name)
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file")
    return resolved


def _artifact_dir(path: Path) -> Path:
    _reject_path(path, "artifact namespace")
    resolved = path.expanduser().resolve()
    allowed = (ROOT / "artifacts/r3/development").resolve()
    if allowed not in resolved.parents:
        raise ValueError("artifacts must remain below artifacts/r3/development")
    if not any("hierarchical-options" in part for part in resolved.parts):
        raise ValueError("artifact namespace must contain hierarchical-options")
    resolved.mkdir(parents=True, exist_ok=True, mode=0o700)
    return resolved


def _write_json(path: Path, value: Mapping[str, object]) -> str:
    _reject_path(path, "output")
    if path.exists():
        raise FileExistsError(f"refusing to replace development artifact: {path}")
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
    return _file_sha(path)


def _load_config(path: Path) -> tuple[Path, dict[str, Any]]:
    source = _input(path, "configuration")
    value = tomllib.loads(source.read_text(encoding="utf-8"))
    required = {
        "version": "hierarchical-options-v1",
        "status": "development_only_not_canonical_evidence",
        "deployable": False,
        "canonical_r3_evidence": False,
        "sealed_evaluation_allowed": False,
        "physics_backend": "portable",
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("configuration weakens the development-only boundary")
    evidence = value.get("evidence", {})
    if (
        not isinstance(evidence, Mapping)
        or evidence.get("policy_snapshot_access") is not False
        or evidence.get("sealed_material_used") is not False
        or evidence.get("canonical_run_storage_used") is not False
    ):
        raise ValueError("configuration evidence boundary is incomplete")
    if value.get("seeds", {}).get("split") != "train":
        raise ValueError("this branch only allocates the train seed split")
    hierarchy = value.get("hierarchy", {})
    if (
        not isinstance(hierarchy, Mapping)
        or hierarchy.get("residual_variants")
        != ["agreement_only", "high_advantage_override"]
        or hierarchy.get("base_wait_only") is not True
    ):
        raise ValueError("residual abstention boundary is incomplete")
    return source, value


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity(config_path: Path) -> dict[str, object]:
    files = (
        Path(__file__).resolve(),
        HERE / "core.py",
        config_path,
        ROOT / "python/irisu_pointer/steering.py",
        ROOT / "python/irisu_pointer/steering_checkpoint.py",
        ROOT / "python/irisu_pointer/steering_learning.py",
        ROOT / "python/irisu_pointer/steering_progress.py",
        ROOT / "python/irisu_pointer/strategic.py",
        ROOT / "python/irisu_env/env.py",
        ROOT / "python/irisu_rl/actions.py",
        ROOT / "python/irisu_rl/seeds.py",
        ROOT / "clone/core/normal_rules.cpp",
        ROOT / "clone/core/simulator.cpp",
    )
    manifest = {
        "schema": "hierarchical-options-source-v1",
        "git_revision": _git_revision(),
        "files": {
            str(path.relative_to(ROOT)): _file_sha(path) for path in files
        },
    }
    return {**manifest, "sha256": _sha(manifest)}


def _seed_suite(config: Mapping[str, Any], name: str) -> dict[str, object]:
    values = config["seeds"]
    key = int(values["allocator_key"])
    cursor = int(values[f"{name}_cursor"])
    count = int(values[f"{name}_count"])
    allocator = SeedAllocator("train", key=key, cursor=cursor)
    seeds = allocator.take(count)
    return {
        "label": f"hierarchical-options-{name}-development-v1",
        "split": "train",
        "allocator_key": key,
        "cursor_start": cursor,
        "cursor_end": allocator.cursor,
        "manifest_sha256": allocator.manifest_sha256,
        "seeds": list(seeds),
        "sha256": _sha(
            {
                "label": f"hierarchical-options-{name}-development-v1",
                "split": "train",
                "allocator_key": key,
                "cursor_start": cursor,
                "cursor_end": allocator.cursor,
                "seeds": list(seeds),
            }
        ),
    }


def _all_suites(config: Mapping[str, Any]) -> dict[str, dict[str, object]]:
    suites = {
        name: _seed_suite(config, name)
        for name in ("fit", "screen", "paired", "plateau")
    }
    seen: set[int] = set()
    for suite in suites.values():
        seeds = set(int(value) for value in suite["seeds"])
        if seen & seeds:
            raise RuntimeError("development seed suites overlap")
        seen |= seeds
    return suites


def _controller_config(config: Mapping[str, Any]) -> SteeringExpertConfig:
    return SteeringExpertConfig(**dict(config["controller"]))


def _controller_identity(
    config: SteeringExpertConfig, source: Mapping[str, object]
) -> dict[str, object]:
    manifest = {
        "type": "closed-loop-steering-expert-v1",
        "checkpoint_free": True,
        "artifact": None,
        "prior_development_policy_identity": (
            "d112fa1b51e71bbb37150591a45ec1af"
            "d924c04916900e250a9949a0b6155431"
        ),
        "steering_source_sha256": source["files"][
            "python/irisu_pointer/steering.py"
        ],
        "controller_config": asdict(config),
        "micro_rule": "analytic-continuous-opposite-side-and-below-v2",
        "press_release_contract": "deployment-v1",
    }
    return {**manifest, "sha256": _sha(manifest)}


def _v5_identity(
    path: Path, config: Mapping[str, Any], source: Mapping[str, object]
) -> dict[str, object]:
    options = {
        key: value
        for key, value in config["frozen_v5"].items()
        if key not in {"checkpoint", "sha256"}
    }
    manifest = {
        "type": "goal-conditioned-steering-policy-v1",
        "path": str(path),
        "checkpoint_sha256": _file_sha(path),
        "configured_sha256": str(config["frozen_v5"]["sha256"]),
        "prior_development_policy_identity": (
            "609fbacebd39bba95eb534725e7cfd80b"
            "a12f17d0a092fb901957e6ae8767c83"
        ),
        "checkpoint_source_sha256": source["files"][
            "python/irisu_pointer/steering_checkpoint.py"
        ],
        "policy_source_sha256": source["files"][
            "python/irisu_pointer/steering_learning.py"
        ],
        "options": options,
    }
    return {**manifest, "sha256": _sha(manifest)}


def _utility_weights(config: Mapping[str, Any]) -> UtilityWeights:
    return UtilityWeights(**dict(config["utility"]))


def _event_name(event: Mapping[str, Any]) -> str | None:
    raw = event.get("kind_name")
    if isinstance(raw, str):
        return raw
    kind = event.get("kind")
    if isinstance(kind, int) and not isinstance(kind, bool):
        try:
            return EventKind(kind).name.lower()
        except ValueError:
            return None
    return None


def _actions(decision: object) -> tuple[Action, ...]:
    if isinstance(decision, SteeringDecision):
        return decision.primitive_actions()
    if isinstance(decision, Action):
        return (decision,)
    if (
        isinstance(decision, tuple)
        and decision
        and all(isinstance(value, Action) for value in decision)
    ):
        return decision
    raise TypeError("policy returned an unsupported decision")


def _limit_action(action: Action, remaining: int) -> Action:
    if remaining < 1:
        raise ValueError("remaining ticks must be positive")
    if (
        ActionKind.parse(action.kind) is ActionKind.WAIT
        and int(action.wait_ticks) > remaining
    ):
        return Action.wait(remaining)
    return action


@dataclass(frozen=True, slots=True)
class RolloutResult:
    observation: Mapping[str, Any]
    events: Mapping[str, int]
    decisions: int
    primitives: int
    simulated_ticks: int


def _continue_rollout(
    env: IrisuEnv,
    observation: Mapping[str, Any],
    policy: object,
    *,
    target_tick: int,
) -> RolloutResult:
    start = int(observation["tick"])
    events: Counter[str] = Counter()
    decisions = 0
    primitives = 0
    terminated = bool(observation.get("terminated", False))
    truncated = bool(observation.get("truncated", False))
    current = observation
    while int(current["tick"]) < target_tick and not terminated and not truncated:
        decision = getattr(policy, "predict")(current)
        decisions += 1
        for raw_action in _actions(decision):
            remaining = target_tick - int(current["tick"])
            if remaining <= 0 or terminated or truncated:
                break
            action = _limit_action(raw_action, remaining)
            before = int(current["tick"])
            current, _, terminated, truncated, info = env.step(action)
            primitives += 1
            if int(current["tick"]) <= before:
                raise RuntimeError("rollout action did not advance simulator time")
            for event in info.get("events", ()):
                if isinstance(event, Mapping):
                    name = _event_name(event)
                    if name is not None:
                        events[name] += 1
    return RolloutResult(
        current,
        dict(sorted(events.items())),
        decisions,
        primitives,
        int(current["tick"]) - start,
    )


class _ForcedThenExpert:
    def __init__(
        self,
        option: Option,
        *,
        model: OptionValueModel,
        controller_config: SteeringExpertConfig,
        commit_ticks: int,
        hierarchy: Mapping[str, Any],
    ) -> None:
        self.option = option
        self.commit_ticks = int(commit_ticks)
        self.forced = HierarchicalOptionPolicy(
            model,
            expert_config=controller_config,
            forced_option=option,
            option_commit_ticks=int(hierarchy["option_commit_ticks"]),
            minimum_option_dwell_ticks=int(
                hierarchy["minimum_option_dwell_ticks"]
            ),
            maximum_option_dwell_ticks=int(
                hierarchy["maximum_option_dwell_ticks"]
            ),
            low_gauge_fraction=float(hierarchy["low_gauge_fraction"]),
            conservative_advantage_margin=float(
                hierarchy["conservative_advantage_margin"]
            ),
            minimum_option_samples=int(
                hierarchy["minimum_option_samples"]
            ),
        )
        self.expert = ClosedLoopSteeringExpert(config=controller_config)
        self.start_tick: int | None = None

    def reset(self, seed: int = 0) -> None:
        self.forced.reset(seed)
        self.expert.reset(seed)
        self.start_tick = None

    def predict(self, observation: Mapping[str, Any]) -> object:
        if self.start_tick is None:
            self.start_tick = int(observation["tick"])
        if int(observation["tick"]) < self.start_tick + self.commit_ticks:
            return self.forced.predict(observation)
        return self.expert.predict(observation)


def _sample_payload(
    *,
    seed: int,
    observation: Mapping[str, Any],
    option: Option,
    outcome: RolloutResult,
    value: float,
) -> dict[str, object]:
    final = outcome.observation
    return {
        "seed": seed,
        "tick": int(observation["tick"]),
        "option": option.value,
        "features": feature_vector(observation).tolist(),
        "utility": value,
        "elapsed_ticks": outcome.simulated_ticks,
        "events": dict(outcome.events),
        "source": {
            "score": int(observation["score"]),
            "gauge": int(observation["gauge"]),
            "level": int(observation["level"]),
            "qualifying_clears": int(
                observation["qualifying_clear_count"]
            ),
        },
        "final": {
            "tick": int(final["tick"]),
            "score": int(final["score"]),
            "gauge": int(final["gauge"]),
            "level": int(final["level"]),
            "qualifying_clears": int(final["qualifying_clear_count"]),
            "terminated": bool(final["terminated"]),
            "truncated": bool(final["truncated"]),
        },
    }


def _collect_samples(
    *,
    runtime: Path,
    seeds: Sequence[int],
    config: Mapping[str, Any],
    model: OptionValueModel,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    settings = config["collection"]
    horizon = int(settings["episode_ticks"])
    stride = int(settings["event_stride_ticks"])
    minimum_gap = int(settings["minimum_event_gap_ticks"])
    state_cap = int(settings["maximum_states_per_seed"])
    branch_horizon = int(settings["branch_horizon_ticks"])
    commit_ticks = int(settings["option_commit_ticks"])
    controller_config = _controller_config(config)
    hierarchy = config["hierarchy"]
    weights = _utility_weights(config)
    samples: list[dict[str, object]] = []
    option_counts: Counter[str] = Counter()
    episodes: list[dict[str, object]] = []
    branch_ticks = 0
    branch_decisions = 0
    restore_checks = 0
    started_wall = time.monotonic()
    started_cpu = time.process_time()

    with IrisuEnv(
        library_path=runtime,
        physics_backend="portable",
        config={"max_episode_ticks": horizon + branch_horizon + 1},
    ) as env:
        runner = env.runner_identity_manifest()
        for seed in seeds:
            expert = ClosedLoopSteeringExpert(config=controller_config)
            expert.reset(int(seed))
            observation, info = env.reset(seed=int(seed))
            if int(info["config_hash"]) != int(runner["config_hash"]):
                raise RuntimeError("collection environment identity changed")
            last_sample_tick = -minimum_gap
            next_stride = 0
            last_signature: tuple[object, ...] | None = None
            state_count = 0
            main_decisions = 0
            while (
                not bool(observation["terminated"])
                and not bool(observation["truncated"])
                and int(observation["tick"]) < horizon
            ):
                tick = int(observation["tick"])
                signature = event_signature(observation)
                changed = last_signature is not None and signature != last_signature
                due = tick >= next_stride or (
                    changed and tick - last_sample_tick >= minimum_gap
                )
                if due and state_count < state_cap:
                    snapshot = env.clone_state()
                    source = observation
                    options = applicable_options(source)
                    for option in options:
                        restored = env.restore_state(snapshot)
                        if event_signature(restored) != signature:
                            raise RuntimeError(
                                "portable option branch restore changed public state"
                            )
                        branch_policy = _ForcedThenExpert(
                            option,
                            model=model,
                            controller_config=controller_config,
                            commit_ticks=commit_ticks,
                            hierarchy=hierarchy,
                        )
                        branch_policy.reset(int(seed))
                        outcome = _continue_rollout(
                            env,
                            restored,
                            branch_policy,
                            target_tick=tick + branch_horizon,
                        )
                        value = branch_utility(
                            source,
                            outcome.observation,
                            elapsed_ticks=outcome.simulated_ticks,
                            horizon_ticks=branch_horizon,
                            event_counts=outcome.events,
                            weights=weights,
                        )
                        samples.append(
                            _sample_payload(
                                seed=int(seed),
                                observation=source,
                                option=option,
                                outcome=outcome,
                                value=value,
                            )
                        )
                        option_counts[option.value] += 1
                        branch_ticks += outcome.simulated_ticks
                        branch_decisions += outcome.decisions
                    observation = env.restore_state(snapshot)
                    if env.clone_state() != snapshot:
                        raise RuntimeError(
                            "option branching failed transactional restoration"
                        )
                    restore_checks += 1
                    state_count += 1
                    last_sample_tick = tick
                    while next_stride <= tick:
                        next_stride += stride
                last_signature = signature
                decision = expert.predict(observation)
                main_decisions += 1
                for raw_action in _actions(decision):
                    remaining = horizon - int(observation["tick"])
                    if remaining <= 0:
                        break
                    observation, _, terminated, truncated, _ = env.step(
                        _limit_action(raw_action, remaining)
                    )
                    if terminated or truncated:
                        break
            episodes.append(
                {
                    "seed": int(seed),
                    "sampled_states": state_count,
                    "main_decisions": main_decisions,
                    "final_tick": int(observation["tick"]),
                    "final_score": int(observation["score"]),
                    "final_gauge": int(observation["gauge"]),
                    "final_level": int(observation["level"]),
                    "terminated": bool(observation["terminated"]),
                    "truncated": bool(observation["truncated"]),
                }
            )

    return samples, {
        "runner": runner,
        "episodes": episodes,
        "sampled_states": restore_checks,
        "branch_queries": len(samples),
        "branch_ticks": branch_ticks,
        "branch_decisions": branch_decisions,
        "option_counts": dict(sorted(option_counts.items())),
        "transactional_restore_checks": restore_checks,
        "wall_seconds": time.monotonic() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
    }


def _model_sample(sample: Mapping[str, object]) -> tuple[np.ndarray, Option, float]:
    features = np.asarray(sample["features"], dtype=np.float64)
    if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
        raise ValueError("saved option sample has invalid features")
    return features, Option(str(sample["option"])), float(sample["utility"])


def _offline_metrics(
    model: OptionValueModel, samples: Sequence[Mapping[str, object]]
) -> dict[str, float | int]:
    errors: list[float] = []
    grouped: dict[tuple[int, int], list[Mapping[str, object]]] = defaultdict(list)
    for sample in samples:
        features, option, target = _model_sample(sample)
        prediction = model.predict_features(features, option)
        errors.append(prediction - target)
        grouped[(int(sample["seed"]), int(sample["tick"]))].append(sample)
    regrets: list[float] = []
    correct = 0
    for values in grouped.values():
        ranked = max(
            values,
            key=lambda item: model.predict_features(
                np.asarray(item["features"], dtype=np.float64),
                Option(str(item["option"])),
            ),
        )
        oracle = max(values, key=lambda item: float(item["utility"]))
        correct += int(ranked["option"] == oracle["option"])
        regrets.append(float(oracle["utility"]) - float(ranked["utility"]))
    squared = [value * value for value in errors]
    return {
        "samples": len(samples),
        "states": len(grouped),
        "rmse": math.sqrt(sum(squared) / len(squared)) if squared else 0.0,
        "mae": sum(abs(value) for value in errors) / len(errors) if errors else 0.0,
        "mean_selection_regret": (
            sum(regrets) / len(regrets) if regrets else 0.0
        ),
        "maximum_selection_regret": max(regrets, default=0.0),
        "oracle_option_accuracy": correct / len(grouped) if grouped else 0.0,
    }


def _fit_models(
    samples: Sequence[Mapping[str, object]],
    *,
    config: Mapping[str, Any],
) -> tuple[OptionValueModel, list[dict[str, object]], float]:
    settings = config["value_fit"]
    fit_seeds = sorted({int(value["seed"]) for value in samples})
    if len(fit_seeds) < 2:
        raise RuntimeError("option fitting requires at least two collection seeds")
    held_seeds = set(fit_seeds[-max(1, len(fit_seeds) // 4) :])
    training = [value for value in samples if int(value["seed"]) not in held_seeds]
    held = [value for value in samples if int(value["seed"]) in held_seeds]
    candidates: list[tuple[tuple[float, float, float], float, OptionValueModel, dict[str, object]]] = []
    for ridge in (float(value) for value in settings["ridge_candidates"]):
        model = OptionValueModel.fit(
            [_model_sample(value) for value in training],
            ridge=ridge,
            minimum_samples_per_option=int(
                settings["minimum_samples_per_option"]
            ),
        )
        metrics = _offline_metrics(model, held)
        report = {
            "ridge": ridge,
            "training": _offline_metrics(model, training),
            "held_seed_list": sorted(held_seeds),
            "held": metrics,
            "model_manifest": model.manifest(),
        }
        key = (
            float(metrics["mean_selection_regret"]),
            float(metrics["rmse"]),
            ridge,
        )
        candidates.append((key, ridge, model, report))
    _, selected_ridge, _, _ = min(candidates, key=lambda value: value[0])
    final = OptionValueModel.fit(
        [_model_sample(value) for value in samples],
        ridge=selected_ridge,
        minimum_samples_per_option=int(settings["minimum_samples_per_option"]),
    )
    curve: list[dict[str, object]] = []
    for fraction in (float(value) for value in settings["curve_fractions"]):
        count = max(1, min(len(training), math.ceil(len(training) * fraction)))
        partial = OptionValueModel.fit(
            [_model_sample(value) for value in training[:count]],
            ridge=selected_ridge,
            minimum_samples_per_option=1,
        )
        curve.append(
            {
                "fraction": fraction,
                "training_samples": count,
                "held": _offline_metrics(partial, held),
                "model_sha256": partial.sha256,
            }
        )
    return final, [value[3] for value in candidates] + [
        {
            "selected_ridge": selected_ridge,
            "full_fit": _offline_metrics(final, samples),
            "sample_plateau_curve": curve,
        }
    ], selected_ridge


def _percentile(values: Sequence[int], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    if not values:
        raise ValueError("distribution requires values")
    return {
        "minimum": min(values),
        "p10": _percentile(values, 0.10),
        "median": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "maximum": max(values),
    }


def _policy_stats(policy: object) -> dict[str, int | float | str]:
    method = getattr(policy, "statistics", None)
    if not callable(method):
        return {}
    raw = method()
    if not isinstance(raw, Mapping):
        raise TypeError("policy statistics must be a mapping")
    return {str(key): value for key, value in raw.items()}


def _checkpoint_metrics(
    observation: Mapping[str, Any],
    *,
    initial_clears: int,
    events: Counter[str],
    decisions: int,
    primitives: int,
    stats: Mapping[str, object],
) -> dict[str, object]:
    terminated = bool(observation["terminated"])
    gauge = int(observation["gauge"])
    return {
        "tick": int(observation["tick"]),
        "score": int(observation["score"]),
        "survival_ticks": int(observation["tick"]),
        "alive": not terminated,
        "gauge_failure": terminated and gauge <= 1,
        "final_gauge": gauge,
        "final_level": int(observation["level"]),
        "highest_chain": int(observation["highest_chain"]),
        "qualifying_clears": (
            int(observation["qualifying_clear_count"]) - initial_clears
        ),
        "rotten": events["rotten"],
        "ejected": events["ejected"],
        "cleared_events": events["cleared"],
        "chain_join_events": events["chain_joined"],
        "shots_fired": events["shot_fired"],
        "projectile_hit_events": events["projectile_hit"],
        "invalid_actions": events["invalid_action"],
        "decisions": decisions,
        "primitive_actions": primitives,
        "policy_stats": dict(stats),
    }


def _run_trace(
    *,
    env: IrisuEnv,
    policy: object,
    label: str,
    seed: int,
    checkpoints: Sequence[int],
    config_hash: int,
) -> dict[str, object]:
    getattr(policy, "reset")(seed)
    observation, info = env.reset(seed=seed)
    if int(info["config_hash"]) != config_hash:
        raise RuntimeError("evaluation reset identity changed")
    initial_clears = int(observation["qualifying_clear_count"])
    events: Counter[str] = Counter()
    decisions = 0
    primitives = 0
    curve: dict[str, object] = {}
    terminal_metrics: dict[str, object] | None = None
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    next_index = 0
    maximum = max(checkpoints)

    while (
        int(observation["tick"]) < maximum
        and not bool(observation["terminated"])
        and not bool(observation["truncated"])
    ):
        decision = getattr(policy, "predict")(observation)
        decisions += 1
        for raw_action in _actions(decision):
            if bool(observation["terminated"]) or bool(observation["truncated"]):
                break
            next_checkpoint = (
                checkpoints[next_index]
                if next_index < len(checkpoints)
                else maximum
            )
            remaining = next_checkpoint - int(observation["tick"])
            if remaining <= 0:
                break
            before = int(observation["tick"])
            observation, _, _, _, step_info = env.step(
                _limit_action(raw_action, remaining)
            )
            primitives += 1
            if int(step_info["config_hash"]) != config_hash:
                raise RuntimeError("evaluation step identity changed")
            if int(observation["tick"]) <= before:
                raise RuntimeError("evaluation action did not advance time")
            for event in step_info.get("events", ()):
                if isinstance(event, Mapping):
                    name = _event_name(event)
                    if name is not None:
                        events[name] += 1
            while (
                next_index < len(checkpoints)
                and int(observation["tick"]) >= checkpoints[next_index]
            ):
                curve[str(checkpoints[next_index])] = _checkpoint_metrics(
                    observation,
                    initial_clears=initial_clears,
                    events=events,
                    decisions=decisions,
                    primitives=primitives,
                    stats=_policy_stats(policy),
                )
                next_index += 1
        if bool(observation["terminated"]) or bool(observation["truncated"]):
            break

    if bool(observation["terminated"]):
        terminal_metrics = _checkpoint_metrics(
            observation,
            initial_clears=initial_clears,
            events=events,
            decisions=decisions,
            primitives=primitives,
            stats=_policy_stats(policy),
        )
    while next_index < len(checkpoints):
        if terminal_metrics is None:
            curve[str(checkpoints[next_index])] = _checkpoint_metrics(
                observation,
                initial_clears=initial_clears,
                events=events,
                decisions=decisions,
                primitives=primitives,
                stats=_policy_stats(policy),
            )
        else:
            curve[str(checkpoints[next_index])] = dict(terminal_metrics)
        next_index += 1

    return {
        "policy": label,
        "seed": seed,
        "curve": curve,
        "terminal": bool(observation["terminated"]),
        "truncated": bool(observation["truncated"]),
        "simulated_ticks": int(observation["tick"]),
        "wall_seconds": time.monotonic() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
    }


def _aggregate_traces(
    traces: Sequence[Mapping[str, object]], checkpoint: int
) -> dict[str, object]:
    rows = [
        trace["curve"][str(checkpoint)]  # type: ignore[index]
        for trace in traces
    ]
    stats_keys = {
        "abstention_count",
        "agreement_actuator_substitution_count",
        "option_queries",
        "event_replans",
        "micro_corrections",
        "option_fallbacks",
        "option_override_count",
        "selection_abstention_count",
    }
    return {
        "episodes": len(rows),
        "raw_score": _distribution([int(row["score"]) for row in rows]),
        "survival_ticks": _distribution(
            [int(row["survival_ticks"]) for row in rows]
        ),
        "final_gauge": _distribution(
            [int(row["final_gauge"]) for row in rows]
        ),
        "final_level": _distribution(
            [int(row["final_level"]) for row in rows]
        ),
        "gauge_failures": sum(bool(row["gauge_failure"]) for row in rows),
        "qualifying_clears": sum(int(row["qualifying_clears"]) for row in rows),
        "rotten": sum(int(row["rotten"]) for row in rows),
        "ejected": sum(int(row["ejected"]) for row in rows),
        "decisions": sum(int(row["decisions"]) for row in rows),
        "primitive_actions": sum(int(row["primitive_actions"]) for row in rows),
        "policy_counts": {
            key: sum(
                int(row.get("policy_stats", {}).get(key, 0))  # type: ignore[union-attr]
                for row in rows
            )
            for key in sorted(stats_keys)
        },
        "invalid_actions": sum(int(row["invalid_actions"]) for row in rows),
        "wall_seconds": sum(float(trace["wall_seconds"]) for trace in traces),
        "cpu_seconds": sum(float(trace["cpu_seconds"]) for trace in traces),
        "simulated_ticks": sum(
            min(int(trace["simulated_ticks"]), checkpoint) for trace in traces
        ),
    }


def _paired_report(
    baseline: Sequence[Mapping[str, object]],
    candidate: Sequence[Mapping[str, object]],
    checkpoints: Sequence[int],
    hierarchy: Mapping[str, Any],
) -> dict[str, object]:
    if [value["seed"] for value in baseline] != [
        value["seed"] for value in candidate
    ]:
        raise RuntimeError("paired policy traces are not seed-aligned")
    absolute = int(hierarchy["catastrophic_absolute_drop_ticks"])
    relative = float(hierarchy["catastrophic_relative_survival"])
    curves: dict[str, object] = {}
    all_catastrophes: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        pairs: list[dict[str, object]] = []
        catastrophes: list[dict[str, object]] = []
        new_failures: list[int] = []
        for left, right in zip(baseline, candidate, strict=True):
            base = left["curve"][str(checkpoint)]  # type: ignore[index]
            cand = right["curve"][str(checkpoint)]  # type: ignore[index]
            survival_delta = int(cand["survival_ticks"]) - int(
                base["survival_ticks"]
            )
            new_failure = bool(cand["gauge_failure"]) and not bool(
                base["gauge_failure"]
            )
            catastrophic = (
                new_failure
                or survival_delta <= -absolute
                and int(cand["survival_ticks"])
                <= relative * int(base["survival_ticks"])
            )
            row = {
                "seed": int(left["seed"]),
                "baseline": base,
                "candidate": cand,
                "delta": {
                    "score": int(cand["score"]) - int(base["score"]),
                    "survival_ticks": survival_delta,
                    "final_gauge": int(cand["final_gauge"])
                    - int(base["final_gauge"]),
                    "final_level": int(cand["final_level"])
                    - int(base["final_level"]),
                    "qualifying_clears": int(cand["qualifying_clears"])
                    - int(base["qualifying_clears"]),
                    "rotten": int(cand["rotten"]) - int(base["rotten"]),
                    "ejected": int(cand["ejected"]) - int(base["ejected"]),
                },
                "new_gauge_failure": new_failure,
                "catastrophic_survival_regression": catastrophic,
            }
            pairs.append(row)
            if new_failure:
                new_failures.append(int(left["seed"]))
            if catastrophic:
                catastrophes.append(row)
                all_catastrophes.append(
                    {"checkpoint": checkpoint, **row}
                )
        score_deltas = [int(row["delta"]["score"]) for row in pairs]  # type: ignore[index]
        survival_deltas = [
            int(row["delta"]["survival_ticks"]) for row in pairs  # type: ignore[index]
        ]
        curves[str(checkpoint)] = {
            "baseline": _aggregate_traces(baseline, checkpoint),
            "candidate": _aggregate_traces(candidate, checkpoint),
            "paired": pairs,
            "score_delta": _distribution(score_deltas),
            "survival_delta": _distribution(survival_deltas),
            "score_wins_ties_losses": {
                "wins": sum(value > 0 for value in score_deltas),
                "ties": sum(value == 0 for value in score_deltas),
                "losses": sum(value < 0 for value in score_deltas),
            },
            "survival_wins_ties_losses": {
                "wins": sum(value > 0 for value in survival_deltas),
                "ties": sum(value == 0 for value in survival_deltas),
                "losses": sum(value < 0 for value in survival_deltas),
            },
            "new_gauge_failure_seeds": new_failures,
            "catastrophic_survival_regressions": catastrophes,
        }
    return {
        "curve": curves,
        "all_catastrophic_survival_regressions": all_catastrophes,
    }


def _evaluate(
    *,
    runtime: Path,
    frozen_v5: Path,
    model: OptionValueModel,
    seeds: Sequence[int],
    checkpoints: Sequence[int],
    config: Mapping[str, Any],
) -> dict[str, object]:
    controller_config = _controller_config(config)
    hierarchy = config["hierarchy"]
    v5: list[dict[str, object]] = []
    analytic: list[dict[str, object]] = []
    agreement: list[dict[str, object]] = []
    high_advantage: list[dict[str, object]] = []
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    with IrisuEnv(
        library_path=runtime,
        physics_backend="portable",
        config={"max_episode_ticks": max(checkpoints)},
    ) as env:
        runner = env.runner_identity_manifest()
        config_hash = int(runner["config_hash"])
        v5_options = {
            key: value
            for key, value in config["frozen_v5"].items()
            if key not in {"checkpoint", "sha256"}
        }

        def load_v5() -> object:
            return load_goal_conditioned_steering_policy(
                frozen_v5,
                expected_sha256=str(config["frozen_v5"]["sha256"]),
                **v5_options,
            )

        v5_policy = load_v5()
        for seed in seeds:
            v5.append(
                _run_trace(
                    env=env,
                    policy=v5_policy,
                    label="frozen_v5_common_comparator",
                    seed=int(seed),
                    checkpoints=checkpoints,
                    config_hash=config_hash,
                )
            )
        for seed in seeds:
            analytic.append(
                _run_trace(
                    env=env,
                    policy=ClosedLoopSteeringExpert(config=controller_config),
                    label="analytic_controller_ablation",
                    seed=int(seed),
                    checkpoints=checkpoints,
                    config_hash=config_hash,
                )
            )
        residual_policies = {
            "agreement_only_residual": HierarchicalOptionPolicy(
                model,
                expert_config=controller_config,
                option_commit_ticks=int(hierarchy["option_commit_ticks"]),
                minimum_option_dwell_ticks=int(
                    hierarchy["minimum_option_dwell_ticks"]
                ),
                maximum_option_dwell_ticks=int(
                    hierarchy["maximum_option_dwell_ticks"]
                ),
                low_gauge_fraction=float(hierarchy["low_gauge_fraction"]),
                conservative_advantage_margin=float(
                    hierarchy["conservative_advantage_margin"]
                ),
                minimum_option_samples=int(
                    hierarchy["minimum_option_samples"]
                ),
                residual_mode="agreement_only",
                default_controller=load_v5(),
                base_wait_only=bool(hierarchy["base_wait_only"]),
            ),
            "high_advantage_override": HierarchicalOptionPolicy(
                model,
                expert_config=controller_config,
                option_commit_ticks=int(hierarchy["option_commit_ticks"]),
                minimum_option_dwell_ticks=int(
                    hierarchy["minimum_option_dwell_ticks"]
                ),
                maximum_option_dwell_ticks=int(
                    hierarchy["maximum_option_dwell_ticks"]
                ),
                low_gauge_fraction=float(hierarchy["low_gauge_fraction"]),
                conservative_advantage_margin=float(
                    hierarchy["conservative_advantage_margin"]
                ),
                minimum_option_samples=int(
                    hierarchy["minimum_option_samples"]
                ),
                residual_mode="high_advantage_override",
                default_controller=load_v5(),
                base_wait_only=bool(hierarchy["base_wait_only"]),
            ),
        }
        target_lists = {
            "agreement_only_residual": agreement,
            "high_advantage_override": high_advantage,
        }
        for label, policy in residual_policies.items():
            for seed in seeds:
                target_lists[label].append(
                    _run_trace(
                        env=env,
                        policy=policy,
                        label=label,
                        seed=int(seed),
                        checkpoints=checkpoints,
                        config_hash=config_hash,
                    )
                )
    return {
        "runner": runner,
        "checkpoints": list(checkpoints),
        "traces": {
            "frozen_v5_common_comparator": v5,
            "analytic_controller_ablation": analytic,
            "agreement_only_residual": agreement,
            "high_advantage_override": high_advantage,
        },
        "causal_decomposition": {
            "analytic_actuator_ablation_vs_frozen_v5": _paired_report(
                v5, analytic, checkpoints, hierarchy
            ),
            "agreement_only_analytic_residual_vs_frozen_v5": _paired_report(
                v5, agreement, checkpoints, hierarchy
            ),
            "high_advantage_hierarchy_vs_frozen_v5": _paired_report(
                v5, high_advantage, checkpoints, hierarchy
            ),
            "agreement_only_vs_analytic_ablation": _paired_report(
                analytic, agreement, checkpoints, hierarchy
            ),
            "high_advantage_vs_analytic_ablation": _paired_report(
                analytic, high_advantage, checkpoints, hierarchy
            ),
            "high_advantage_vs_agreement_only": _paired_report(
                agreement, high_advantage, checkpoints, hierarchy
            ),
        },
        "execution": {
            "wall_seconds": time.monotonic() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "requested_policy_seed_runs": 4 * len(seeds),
        },
    }


def _base_report(
    *,
    mode: str,
    config_path: Path,
    config: Mapping[str, Any],
    runtime: Path,
    frozen_v5: Path,
    source: Mapping[str, object],
    suites: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    runtime_sha = _file_sha(runtime)
    controller_config = _controller_config(config)
    return {
        "schema": _FORMAT,
        "mode": mode,
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
        "source_identity": dict(source),
        "config": {
            "path": str(config_path),
            "sha256": _file_sha(config_path),
            "version": config["version"],
        },
        "runtime": {
            "path": str(runtime),
            "sha256": runtime_sha,
            "backend": "portable",
        },
        "comparator_identities": {
            "frozen_v5_common_tournament_comparator": _v5_identity(
                frozen_v5, config, source
            ),
            "analytic_controller_only_ablation": _controller_identity(
                controller_config, source
            ),
        },
        "seed_suites": {key: dict(value) for key, value in suites.items()},
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "logical_cpus": os.cpu_count(),
            "platform": platform.platform(),
        },
        "prior_negative_evidence": {
            "candidate_b": {
                "verdict": "reject",
                "screen_artifact": str(
                    ROOT
                    / "artifacts/r3/development/"
                    "hierarchical-options-20260729/candidate-b/"
                    "screen-report.json"
                ),
                "screen_artifact_sha256": (
                    "ea94542ea8cd71ed076da20edec6446d2"
                    "f0ac324750dec48d7e8622304e0cc32"
                ),
                "v5_relative_catastrophic_seeds": [
                    851851769,
                    12510751,
                    129711154,
                    364111960,
                    481312363,
                ],
                "failure_mode": (
                    "rotten-triage attractor with repeated micro overrides"
                ),
            },
            "candidate_c": {
                "verdict": "reject",
                "screen_artifact_sha256": (
                    "caa93b52c8433d417e83c75597a69ea8"
                    "006c80f71b40a7a6a6afadcc4418ad46"
                ),
                "same_five_v5_catastrophes_resolved": 0,
            },
        },
    }


def _save_model(path: Path, model: OptionValueModel) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to replace option model: {path}")
    model.save(path)
    return _file_sha(path)


def _load_model(path: Path) -> tuple[Path, OptionValueModel]:
    source = _input(path, "option value model")
    return source, OptionValueModel.load(source)


def _zero_model() -> OptionValueModel:
    samples = [
        (np.zeros(len(FEATURE_NAMES), dtype=np.float64), option, 0.0)
        for option in OPTION_ORDER
    ]
    return OptionValueModel.fit(samples, ridge=1.0, minimum_samples_per_option=1)


def _self_check(
    runtime: Path,
    frozen_v5: Path,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, object]:
    model = _zero_model()
    controller = _controller_config(config)
    hierarchy = config["hierarchy"]
    with IrisuEnv(
        library_path=runtime,
        physics_backend="portable",
        config={"max_episode_ticks": 512},
    ) as env:
        observation, _ = env.reset(seed=seed)
        features = feature_vector(observation)
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise RuntimeError("economic feature self-check failed")
        snapshot = env.clone_state()
        option = applicable_options(observation)[0]
        policy = HierarchicalOptionPolicy(
            model,
            expert_config=controller,
            forced_option=option,
            option_commit_ticks=int(hierarchy["option_commit_ticks"]),
            minimum_option_dwell_ticks=int(
                hierarchy["minimum_option_dwell_ticks"]
            ),
            maximum_option_dwell_ticks=int(
                hierarchy["maximum_option_dwell_ticks"]
            ),
            minimum_option_samples=int(
                hierarchy["minimum_option_samples"]
            ),
        )
        policy.reset(seed)
        outcome = _continue_rollout(
            env, observation, policy, target_tick=256
        )
        restored = env.restore_state(snapshot)
        if env.clone_state() != snapshot or int(restored["tick"]) != 0:
            raise RuntimeError("portable snapshot transaction self-check failed")
        v5_options = {
            key: value
            for key, value in config["frozen_v5"].items()
            if key not in {"checkpoint", "sha256"}
        }
        residual_checks: dict[str, object] = {}
        for mode in ("agreement_only", "high_advantage_override"):
            restored = env.restore_state(snapshot)
            default = load_goal_conditioned_steering_policy(
                frozen_v5,
                expected_sha256=str(config["frozen_v5"]["sha256"]),
                **v5_options,
            )
            residual = HierarchicalOptionPolicy(
                model,
                expert_config=controller,
                option_commit_ticks=int(hierarchy["option_commit_ticks"]),
                minimum_option_dwell_ticks=int(
                    hierarchy["minimum_option_dwell_ticks"]
                ),
                maximum_option_dwell_ticks=int(
                    hierarchy["maximum_option_dwell_ticks"]
                ),
                low_gauge_fraction=float(hierarchy["low_gauge_fraction"]),
                conservative_advantage_margin=float(
                    hierarchy["conservative_advantage_margin"]
                ),
                minimum_option_samples=int(
                    hierarchy["minimum_option_samples"]
                ),
                residual_mode=mode,
                default_controller=default,
                base_wait_only=bool(hierarchy["base_wait_only"]),
            )
            residual.reset(seed)
            result = _continue_rollout(
                env, restored, residual, target_tick=128
            )
            if int(result.events.get("invalid_action", 0)):
                raise RuntimeError(f"{mode} residual emitted an invalid action")
            residual_checks[mode] = {
                "branch_ticks": result.simulated_ticks,
                "events": dict(result.events),
                "policy_statistics": residual.statistics(),
            }
        restored = env.restore_state(snapshot)
        if env.clone_state() != snapshot or int(restored["tick"]) != 0:
            raise RuntimeError("residual self-check changed snapshot identity")
        return {
            "seed": seed,
            "selected_option": option.value,
            "applicable_options": [
                value.value for value in applicable_options(observation)
            ],
            "feature_count": len(features),
            "branch_ticks": outcome.simulated_ticks,
            "events": dict(outcome.events),
            "invalid_actions": int(outcome.events.get("invalid_action", 0)),
            "policy_statistics": policy.statistics(),
            "residual_checks": residual_checks,
            "snapshot_restore_equal": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("self-check", "fit", "screen", "paired", "plateau"),
        required=True,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()

    config_path, config = _load_config(args.config)
    runtime = _input(Path(config["trusted_runtime"]), "portable runtime")
    frozen_v5 = _input(
        Path(config["frozen_v5"]["checkpoint"]),
        "frozen v5 development comparator",
    )
    expected_runtime = (
        "4f6928f18c83159b0db1cb895891007ac"
        "805d2542954b41d767619eedf3f7c79"
    )
    if _file_sha(runtime) != expected_runtime:
        raise RuntimeError("portable runtime identity is not the trusted build")
    if _file_sha(frozen_v5) != str(config["frozen_v5"]["sha256"]):
        raise RuntimeError("frozen v5 comparator identity changed")
    configured_artifacts = ROOT / str(config["artifact_namespace"])
    artifacts = _artifact_dir(
        configured_artifacts if args.artifact_dir is None else args.artifact_dir
    )
    source = _source_identity(config_path)
    suites = _all_suites(config)
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    payload = _base_report(
        mode=args.mode,
        config_path=config_path,
        config=config,
        runtime=runtime,
        frozen_v5=frozen_v5,
        source=source,
        suites=suites,
    )

    if args.mode == "self-check":
        payload["self_check"] = _self_check(
            runtime, frozen_v5, config, int(suites["fit"]["seeds"][0])
        )
        output = artifacts / "self-check.json"
    elif args.mode == "fit":
        bootstrap = _zero_model()
        samples, collection = _collect_samples(
            runtime=runtime,
            seeds=[int(value) for value in suites["fit"]["seeds"]],
            config=config,
            model=bootstrap,
        )
        sample_payload = {
            "schema": "hierarchical-options-branch-samples-v1",
            "development_only": True,
            "source_identity_sha256": source["sha256"],
            "runtime_sha256": _file_sha(runtime),
            "seed_suite": suites["fit"],
            "feature_names": list(FEATURE_NAMES),
            "collection": collection,
            "samples": samples,
        }
        samples_path = artifacts / "branch-samples.json"
        samples_sha = _write_json(samples_path, sample_payload)
        model, fit_reports, ridge = _fit_models(samples, config=config)
        model_path = artifacts / "value-model.json"
        model_sha = _save_model(model_path, model)
        payload["offline_collection"] = {
            **collection,
            "artifact": {
                "path": str(samples_path),
                "sha256": samples_sha,
            },
            "samples_sha256": _sha(samples),
        }
        payload["value_fit"] = {
            "method": "per-option standardized ridge regression",
            "selected_ridge": ridge,
            "candidates_and_plateau": fit_reports,
            "model": {
                "path": str(model_path),
                "sha256": model_sha,
                "identity": model.sha256,
                "manifest": model.manifest(),
            },
        }
        output = artifacts / "fit-report.json"
    else:
        model_path = (
            artifacts / "value-model.json" if args.model is None else args.model
        )
        resolved_model, model = _load_model(model_path)
        payload["candidate_identity"] = {
            "type": "hierarchical-event-driven-abstaining-residual-v1",
            "model_path": str(resolved_model),
            "model_file_sha256": _file_sha(resolved_model),
            "model_identity": model.sha256,
            "model_manifest": model.manifest(),
            "core_source_sha256": source["files"][
                "benchmarks/hierarchical-options/core.py"
            ],
            "default_micro_controller_and_common_comparator": payload[
                "comparator_identities"
            ]["frozen_v5_common_tournament_comparator"],
            "projected_option_controller_and_full_ablation": payload[
                "comparator_identities"
            ][
                "analytic_controller_only_ablation"
            ],
            "variants": {
                "agreement_only_residual": {
                    "gate": (
                        "option agrees with v5 intent and projected analytic "
                        "action preserves v5 shot/wait and body binding; "
                        "otherwise explicitly abstain to v5"
                    ),
                    "one_override_per_event_replan": True,
                },
                "high_advantage_override": {
                    "gate": (
                        "agreement-only behavior plus one learned dissent "
                        "when v5 waits and option advantage meets the "
                        "configured held-error margin"
                    ),
                    "one_override_per_event_replan": True,
                },
            },
            "hierarchy": dict(config["hierarchy"]),
        }
        if args.mode == "screen":
            section = config["screen"]
            suite = suites["screen"]
            checkpoints = (2000, int(section["horizon_ticks"]))
            output = artifacts / "screen-report.json"
        elif args.mode == "paired":
            section = config["paired_evaluation"]
            suite = suites["paired"]
            checkpoints = tuple(int(value) for value in section["checkpoints"])
            output = artifacts / "paired-ab-report.json"
        else:
            section = config["plateau_probe"]
            suite = suites["plateau"]
            checkpoints = tuple(int(value) for value in section["checkpoints"])
            output = artifacts / "plateau-report.json"
        payload["evaluation_suite"] = suite
        payload["evaluation"] = _evaluate(
            runtime=runtime,
            frozen_v5=frozen_v5,
            model=model,
            seeds=[int(value) for value in suite["seeds"]],
            checkpoints=checkpoints,
            config=config,
        )

    payload["execution"] = {
        "wall_seconds": time.monotonic() - started_wall,
        "cpu_seconds": time.process_time() - started_cpu,
    }
    if _source_identity(config_path) != source:
        raise RuntimeError("source identity changed during development run")
    payload["payload_sha256"] = _sha(payload)
    digest = _write_json(output, payload)
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": digest,
                "payload_sha256": payload["payload_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
