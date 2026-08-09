#!/usr/bin/env python3
"""Development-only genuine-prefix long probe for the reserve-band oracle.

Each mode/seed is one 50,000-tick trajectory.  The collector observes exact
2k/10k/20k/50k prefixes without asking the policy to act at a checkpoint.
Only a WAIT primitive may be split, and its unconsumed ticks resume before the
next policy decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from irisu_env import Action, ActionKind, EventKind, IrisuEnv
from irisu_pointer.geometry_policy import geometry_candidate_vocabulary_sha256
from irisu_pointer.steering import SteeringDecision


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = Path(__file__).resolve().parent
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))
import rl_r3d_steering as r3d  # noqa: E402
import rl_r3e_reserve_band as reserve  # noqa: E402
import rl_r3e_sustainable as r3e  # noqa: E402


PREFIX_VERSION = "r3e-reserve-band-genuine-prefix-long-v1"
REPORT_SCHEMA = "irisu-r3e-reserve-band-prefix-long-development-v1"
DEFAULT_CONFIG = (
    ROOT / "configs/rl/experiments/r3e-reserve-band-debt-mpc-v2.toml"
)
EXPECTED_CONFIG_SHA256 = (
    "4c0d83c43f6cb7f57f4478c95511a7ea353326c17be4463ff731d70ac74b09ba"
)
EXPECTED_RUNTIME = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/runtime/"
    "main-0c48dba-20260723/portable-build/libirisu_clone.so"
)
EXPECTED_RUNTIME_SHA256 = (
    "4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79"
)
EXPECTED_FROZEN_V5 = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3d-survival-v5-20260729/long-development.pt"
)
EXPECTED_FROZEN_V5_SHA256 = (
    "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d"
)
CHECKPOINTS = (2_000, 10_000, 20_000, 50_000)
EPISODE_TICKS = CHECKPOINTS[-1]
DEFAULT_MODES = ("reserve_band",)
SAFE_MODES = reserve.MODES
PARITY_WAIT_TICKS = 7
PARITY_FIRST_TICKS = 3


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode()


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"prefix manifest contains {type(value).__name__}")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _chain_sha256(previous: str, value: object) -> str:
    return hashlib.sha256(bytes.fromhex(previous) + _canonical_bytes(value)).hexdigest()


def _source_identity(config_path: Path) -> dict[str, object]:
    harness = reserve._source_identity(config_path)
    collector = Path(__file__).resolve()
    environment_sources = sorted((ROOT / "python/irisu_env").glob("*.py"))
    manifest = {
        "schema": "irisu-r3e-reserve-band-prefix-source-v1",
        "version": PREFIX_VERSION,
        "harness_source_identity": harness,
        "collector": {
            "path": str(collector.relative_to(ROOT)),
            "sha256": r3d._file_sha256(collector),
        },
        "environment_sources": {
            str(path.relative_to(ROOT)): r3d._file_sha256(path)
            for path in environment_sources
        },
    }
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _require_source_identity(
    expected: Mapping[str, object], config_path: Path
) -> None:
    if _source_identity(config_path) != dict(expected):
        raise RuntimeError("prefix-probe source identity changed during execution")


def _action_manifest(action: Action) -> dict[str, int | float]:
    return {
        "kind": int(ActionKind.parse(action.kind)),
        "cursor_x": float(action.cursor_x),
        "cursor_y": float(action.cursor_y),
        "wait_ticks": int(action.wait_ticks),
    }


def _decision_manifest(decision: SteeringDecision) -> dict[str, object]:
    semantic = decision.action
    return {
        "semantic_action": {
            "kind": int(semantic.kind),
            "wait_ticks": int(semantic.wait_ticks),
            "x_norm": float(semantic.x_norm),
            "y_norm": float(semantic.y_norm),
        },
        "intent": decision.intent.value,
        "source_body_id": decision.source_body_id,
        "destination_body_id": decision.destination_body_id,
        "destination_chain_id": decision.destination_chain_id,
        "impact_x_sizes": float(decision.impact_x_sizes),
        "impact_y_sizes": float(decision.impact_y_sizes),
        "correction_index": decision.correction_index,
        "reason": decision.reason,
    }


def _policy_counts(policy: object) -> dict[str, int]:
    statistics = getattr(policy, "statistics", None)
    if not callable(statistics):
        return {}
    raw = statistics()
    if not isinstance(raw, Mapping):
        raise TypeError("prefix policy statistics must be a mapping")
    result: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or type(value) is not int:
            raise TypeError("prefix policy statistics must contain integer counts")
        result[key] = value
    return dict(sorted(result.items()))


class _TrackedPolicy:
    """Observe oracle query timing without changing its decision cadence."""

    def __init__(self, policy: object, query_cap: int | None) -> None:
        self.policy = policy
        self.query_cap = query_cap
        self.last_query_tick: int | None = None
        self.query_cap_reached_tick: int | None = None

    def reset(self, seed: int) -> None:
        getattr(self.policy, "reset")(seed)
        self.last_query_tick = None
        self.query_cap_reached_tick = None

    def predict(self, observation: Mapping[str, Any]) -> object:
        before = _policy_counts(self.policy).get("search_queries", 0)
        predict = getattr(self.policy, "predict", None)
        decision = (
            predict(observation)
            if callable(predict)
            else getattr(self.policy, "act")(observation)
        )
        after = _policy_counts(self.policy).get("search_queries", 0)
        if after < before or after > before + 1:
            raise RuntimeError("prefix policy query counter changed unexpectedly")
        if after > before:
            tick = int(observation.get("tick", -1))
            self.last_query_tick = tick
            if self.query_cap is not None and after == self.query_cap:
                self.query_cap_reached_tick = tick
        return decision

    def statistics(self) -> dict[str, int]:
        return _policy_counts(self.policy)


def _wait_split_parity(
    env: IrisuEnv, *, seed: int, config_hash: int
) -> dict[str, object]:
    observation, info = env.reset(seed=seed)
    if int(info.get("config_hash", -1)) != config_hash:
        raise RuntimeError("wait parity reset config identity mismatch")
    source = env.clone_state()
    try:
        unsplit = env.step(Action.wait(PARITY_WAIT_TICKS))
        unsplit_state = env.clone_state()
        env.restore_state(source)
        first = env.step(Action.wait(PARITY_FIRST_TICKS))
        second = env.step(
            Action.wait(PARITY_WAIT_TICKS - PARITY_FIRST_TICKS)
        )
        split_state = env.clone_state()
        unsplit_events = list(unsplit[4].get("events", ()))
        split_events = [
            *first[4].get("events", ()),
            *second[4].get("events", ()),
        ]
        checks = {
            "snapshot_exact": unsplit_state == split_state,
            "public_observation_exact": (
                _canonical_sha256(unsplit[0]) == _canonical_sha256(second[0])
            ),
            "reward_exact": int(unsplit[1]) == int(first[1]) + int(second[1]),
            "terminal_flags_exact": unsplit[2:4] == second[2:4],
            "events_exact": unsplit_events == split_events,
            "config_identity_exact": all(
                int(value[4].get("config_hash", -1)) == config_hash
                for value in (unsplit, first, second)
            ),
        }
    finally:
        env.restore_state(source)
    checks["source_restored"] = env.clone_state() == source
    if not all(checks.values()):
        raise RuntimeError("split-vs-unsplit WAIT parity failed")
    return {
        "status": "PASS",
        "seed": seed,
        "total_wait_ticks": PARITY_WAIT_TICKS,
        "split_wait_ticks": [
            PARITY_FIRST_TICKS,
            PARITY_WAIT_TICKS - PARITY_FIRST_TICKS,
        ],
        "checks": checks,
        "initial_tick": int(observation.get("tick", -1)),
        "source_snapshot_sha256": hashlib.sha256(source).hexdigest(),
        "final_snapshot_sha256": hashlib.sha256(unsplit_state).hexdigest(),
        "event_count": len(unsplit_events),
    }


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
    raise TypeError("prefix policy returned an unsupported decision")


def _checkpoint_policy(
    policy: _TrackedPolicy,
) -> dict[str, object]:
    counts = policy.statistics()
    queries = counts.get("search_queries", 0)
    corrections = counts.get("executed_corrections", 0)
    regimes = {
        key.removeprefix("regime_"): value
        for key, value in counts.items()
        if key.startswith("regime_")
    }
    return {
        "search_queries": queries,
        "executed_corrections": corrections,
        "strict_improvements": counts.get("strict_improvements", 0),
        "seen_shots": counts.get("seen_shots", 0),
        "unsupported_pairs": counts.get("unsupported_pairs", 0),
        "episode_boundary_skips": counts.get("episode_boundary_skips", 0),
        "regime_counts": regimes,
        "query_cap": policy.query_cap,
        "query_cap_reached": (
            policy.query_cap is not None and queries >= policy.query_cap
        ),
        "query_cap_reached_tick": policy.query_cap_reached_tick,
        "last_query_tick": policy.last_query_tick,
    }


def _run_prefix_episode(
    env: IrisuEnv,
    policy: _TrackedPolicy,
    *,
    mode: str,
    seed: int,
    config_hash: int,
) -> dict[str, object]:
    wall_started = time.monotonic()
    cpu_started = time.process_time()
    policy.reset(seed)
    observation, reset_info = env.reset(seed=seed)
    if int(reset_info.get("config_hash", -1)) != config_hash:
        raise RuntimeError("prefix reset config identity mismatch")
    initial_tick = int(observation.get("tick", -1))
    initial_clears = int(observation.get("qualifying_clear_count", 0))
    if initial_tick != 0:
        raise RuntimeError("prefix trajectory did not begin at tick zero")

    event_counts: Counter[int] = Counter()
    decisions = 0
    primitive_actions = 0
    environment_steps = 0
    wait_boundary_splits = 0
    terminated = bool(observation.get("terminated", False))
    truncated = bool(observation.get("truncated", False))
    checkpoint_index = 0
    checkpoints: list[dict[str, object]] = []
    transcript = _canonical_sha256(
        {
            "schema": "irisu-r3e-prefix-transcript-v1",
            "mode": mode,
            "seed": seed,
            "initial_observation_sha256": _canonical_sha256(observation),
        }
    )

    def capture(checkpoint_tick: int, *, carried: bool) -> None:
        observed_tick = int(observation.get("tick", -1))
        if not carried and observed_tick != checkpoint_tick:
            raise RuntimeError("prefix checkpoint was not observed at its exact tick")
        qualifying = (
            int(observation.get("qualifying_clear_count", 0)) - initial_clears
        )
        if qualifying < 0:
            raise RuntimeError("prefix qualifying-clear counter moved backwards")
        checkpoints.append(
            {
                "checkpoint_tick": checkpoint_tick,
                "observed_tick": observed_tick,
                "survival_ticks": observed_tick - initial_tick,
                "terminal_carried_forward": carried,
                "score": int(observation.get("score", 0)),
                "gauge": int(observation.get("gauge", 0)),
                "gauge_max": int(observation.get("gauge_max", 0)),
                "level": int(observation.get("level", 0)),
                "qualifying_clears": qualifying,
                "cleared_events": event_counts[int(EventKind.CLEARED)],
                "rot_events": event_counts[int(EventKind.ROTTEN)],
                "game_over_events": event_counts[int(EventKind.GAME_OVER)],
                "level_completed_events": event_counts[
                    int(EventKind.LEVEL_COMPLETED)
                ],
                "gauge_failure": event_counts[int(EventKind.GAME_OVER)] > 0,
                "level_completed": (
                    event_counts[int(EventKind.LEVEL_COMPLETED)] > 0
                ),
                "terminated": terminated,
                "truncated": truncated,
                "decisions": decisions,
                "primitive_actions": primitive_actions,
                "environment_steps": environment_steps,
                "wait_boundary_splits": wait_boundary_splits,
                "policy": _checkpoint_policy(policy),
                "transcript_sha256": transcript,
                "public_observation_sha256": _canonical_sha256(observation),
            }
        )

    def capture_due() -> None:
        nonlocal checkpoint_index
        tick = int(observation.get("tick", -1))
        while (
            checkpoint_index < len(CHECKPOINTS)
            and tick >= CHECKPOINTS[checkpoint_index]
        ):
            capture(CHECKPOINTS[checkpoint_index], carried=False)
            checkpoint_index += 1

    capture_due()
    while not terminated and not truncated and checkpoint_index < len(CHECKPOINTS):
        capture_due()
        tick = int(observation.get("tick", -1))
        decision = policy.predict(observation)
        actions = _actions(decision)
        decisions += 1
        transcript = _chain_sha256(
            transcript,
            {
                "type": "decision",
                "tick": tick,
                "decision": (
                    _decision_manifest(decision)
                    if isinstance(decision, SteeringDecision)
                    else None
                ),
                "actions": [_action_manifest(action) for action in actions],
                "policy": _checkpoint_policy(policy),
            },
        )
        for action in actions:
            if terminated or truncated:
                break
            primitive_actions += 1
            original = _action_manifest(action)
            kind = ActionKind.parse(action.kind)
            remaining = int(action.wait_ticks) if kind is ActionKind.WAIT else 1
            fragment_index = 0
            while remaining > 0 and not terminated and not truncated:
                capture_due()
                before_tick = int(observation.get("tick", -1))
                duration = remaining
                if kind is ActionKind.WAIT and checkpoint_index < len(CHECKPOINTS):
                    boundary = CHECKPOINTS[checkpoint_index] - before_tick
                    if boundary <= 0:
                        raise RuntimeError("prefix checkpoint capture fell behind")
                    duration = min(duration, boundary)
                fragment = Action.wait(duration) if kind is ActionKind.WAIT else action
                if kind is ActionKind.WAIT and duration < remaining:
                    wait_boundary_splits += 1
                result = env.step(fragment)
                current, reward, terminated, truncated, info = result
                if not isinstance(current, Mapping) or not isinstance(info, Mapping):
                    raise TypeError("prefix transition must expose public mappings")
                if int(info.get("config_hash", -1)) != config_hash:
                    raise RuntimeError("prefix step config identity mismatch")
                after_tick = int(current.get("tick", -1))
                advanced = after_tick - before_tick
                if advanced < 1 or advanced > duration:
                    raise RuntimeError("prefix primitive violated public time")
                if not (terminated or truncated) and advanced != duration:
                    raise RuntimeError("prefix nonterminal primitive ended early")
                if (
                    bool(current.get("terminated", False)) != bool(terminated)
                    or bool(current.get("truncated", False)) != bool(truncated)
                ):
                    raise RuntimeError(
                        "prefix transition flags disagree with observation"
                    )
                raw_events = info.get("events", ())
                if (
                    not isinstance(raw_events, Sequence)
                    or isinstance(raw_events, (str, bytes))
                    or any(
                        not isinstance(event, Mapping)
                        for event in raw_events
                    )
                ):
                    raise TypeError("prefix transition events are malformed")
                transition_events = [dict(event) for event in raw_events]
                for event in transition_events:
                    event_kind = r3d._event_kind(event)
                    if event_kind is not None:
                        event_counts[event_kind] += 1
                environment_steps += 1
                transcript = _chain_sha256(
                    transcript,
                    {
                        "type": "transition",
                        "original_primitive": original,
                        "fragment": _action_manifest(fragment),
                        "fragment_index": fragment_index,
                        "before_tick": before_tick,
                        "after_tick": after_tick,
                        "reward": int(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "events": transition_events,
                        "observation_sha256": _canonical_sha256(current),
                    },
                )
                observation = current
                remaining -= advanced
                fragment_index += 1
                capture_due()
                if kind is not ActionKind.WAIT:
                    remaining = 0

    final_tick = int(observation.get("tick", -1))
    if not (terminated or truncated) and final_tick != EPISODE_TICKS:
        raise RuntimeError("prefix trajectory ended before its terminal condition")
    while checkpoint_index < len(CHECKPOINTS):
        if not terminated:
            raise RuntimeError("only a terminal episode may carry checkpoints")
        capture(CHECKPOINTS[checkpoint_index], carried=True)
        checkpoint_index += 1
    if tuple(value["checkpoint_tick"] for value in checkpoints) != CHECKPOINTS:
        raise RuntimeError("prefix trajectory did not retain every checkpoint")
    game_over_events = event_counts[int(EventKind.GAME_OVER)]
    level_completed_events = event_counts[int(EventKind.LEVEL_COMPLETED)]
    terminal_cause = (
        "GAME_OVER+LEVEL_COMPLETED"
        if game_over_events and level_completed_events
        else "GAME_OVER"
        if game_over_events
        else "LEVEL_COMPLETED"
        if level_completed_events
        else "HORIZON"
        if truncated
        else "NONE"
    )
    return {
        "mode": mode,
        "seed": seed,
        "episode_ticks": EPISODE_TICKS,
        "checkpoints": checkpoints,
        "final_observed_tick": final_tick,
        "final_score": int(observation.get("score", 0)),
        "final_gauge": int(observation.get("gauge", 0)),
        "final_gauge_max": int(observation.get("gauge_max", 0)),
        "final_level": int(observation.get("level", 0)),
        "final_qualifying_clears": (
            int(observation.get("qualifying_clear_count", 0))
            - initial_clears
        ),
        "cleared_events": event_counts[int(EventKind.CLEARED)],
        "rot_events": event_counts[int(EventKind.ROTTEN)],
        "game_over_events": game_over_events,
        "level_completed_events": level_completed_events,
        "terminal_cause": terminal_cause,
        "terminated": terminated,
        "truncated": truncated,
        "gauge_failure": game_over_events > 0,
        "level_completed": level_completed_events > 0,
        "decisions": decisions,
        "primitive_actions": primitive_actions,
        "environment_steps": environment_steps,
        "wait_boundary_splits": wait_boundary_splits,
        "final_policy_counts": policy.statistics(),
        "query_cap": policy.query_cap,
        "query_cap_reached_tick": policy.query_cap_reached_tick,
        "last_query_tick": policy.last_query_tick,
        "transcript_sha256": transcript,
        "cost": {
            "wall_seconds": time.monotonic() - wall_started,
            "cpu_seconds": time.process_time() - cpu_started,
        },
    }


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    return r3d._distribution(tuple(int(value) for value in values))


def _checkpoint_aggregates(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for index, checkpoint in enumerate(CHECKPOINTS):
        rows = [episode["checkpoints"][index] for episode in episodes]
        regime_counts: Counter[str] = Counter()
        for row in rows:
            regime_counts.update(row["policy"]["regime_counts"])
        result[str(checkpoint)] = {
            "episodes": len(rows),
            "score": _distribution([row["score"] for row in rows]),
            "survival_ticks": _distribution(
                [row["survival_ticks"] for row in rows]
            ),
            "gauge": _distribution([row["gauge"] for row in rows]),
            "level": _distribution([row["level"] for row in rows]),
            "qualifying_clears": _distribution(
                [row["qualifying_clears"] for row in rows]
            ),
            "cleared_events": _distribution(
                [row["cleared_events"] for row in rows]
            ),
            "rot_events": _distribution([row["rot_events"] for row in rows]),
            "decisions": _distribution([row["decisions"] for row in rows]),
            "primitive_actions": _distribution(
                [row["primitive_actions"] for row in rows]
            ),
            "search_queries": _distribution(
                [row["policy"]["search_queries"] for row in rows]
            ),
            "executed_corrections": _distribution(
                [row["policy"]["executed_corrections"] for row in rows]
            ),
            "gauge_failures": sum(row["gauge_failure"] for row in rows),
            "level_completions": sum(row["level_completed"] for row in rows),
            "terminal_carried_forward": sum(
                row["terminal_carried_forward"] for row in rows
            ),
            "regime_counts": dict(sorted(regime_counts.items())),
        }
    return result


def _evaluate_mode(
    *,
    mode: str,
    config: Mapping[str, Any],
    runtime_path: Path,
    base_path: Path,
    base_sha256: str,
    seeds: Sequence[int],
    teacher: object | None,
) -> dict[str, object]:
    wall_started = time.monotonic()
    cpu_started = time.process_time()
    episodes: list[dict[str, object]] = []
    with IrisuEnv(
        library_path=runtime_path,
        physics_backend="portable",
        config={"max_episode_ticks": EPISODE_TICKS},
    ) as env:
        if Path(env.library_path).resolve() != runtime_path:
            raise RuntimeError("prefix evaluation loaded a foreign runtime")
        runner = env.runner_identity_manifest()
        config_hash = int(runner["config_hash"])
        for seed in seeds:
            base = r3e._base_policy(
                base_path,
                base_sha256,
                reserve._base_options(config),
            )
            if mode == "base_v5":
                inner = base
                query_cap = None
            else:
                if teacher is None:
                    raise RuntimeError("prefix oracle mode lacks its teacher")
                inner = reserve.SampledOraclePolicy(
                    env=env,
                    base_policy=base,
                    teacher=teacher,
                    seed=int(seed),
                    episode_ticks=EPISODE_TICKS,
                    query_stride_shots=int(
                        config["oracle"]["query_stride_shots"]
                    ),
                    maximum_search_queries=int(
                        config["oracle"]["maximum_search_queries_per_episode"]
                    ),
                )
                query_cap = int(
                    config["oracle"]["maximum_search_queries_per_episode"]
                )
            episodes.append(
                _run_prefix_episode(
                    env,
                    _TrackedPolicy(inner, query_cap),
                    mode=mode,
                    seed=int(seed),
                    config_hash=config_hash,
                )
            )
    return {
        "runner": runner,
        "episodes": episodes,
        "checkpoint_aggregates": _checkpoint_aggregates(episodes),
        "cost": {
            "wall_seconds": time.monotonic() - wall_started,
            "cpu_seconds": time.process_time() - cpu_started,
            "episodes": len(episodes),
        },
    }


def _safe_output(config: Mapping[str, Any], path: Path) -> Path:
    namespace = reserve._path(str(config["artifact_namespace"])).resolve()
    output = r3d._output_path(path, "reserve-band prefix report", ".json")
    if namespace != output.parent and namespace not in output.parents:
        raise ValueError("prefix report must stay in reserve-band namespace")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--suite", choices=("long",), default="long")
    parser.add_argument("--modes", nargs="+", choices=SAFE_MODES)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--result-out", type=Path)
    args = parser.parse_args()

    started = time.monotonic()
    cpu_started = time.process_time()
    config_snapshot = r3d._snapshot_file(args.config, "prefix config")
    if (
        config_snapshot.path != DEFAULT_CONFIG.resolve(strict=True)
        or config_snapshot.sha256 != EXPECTED_CONFIG_SHA256
    ):
        parser.error("prefix probe requires the exact debt-aware config")
    config = reserve._load_config(config_snapshot)
    if (
        tuple(int(value) for value in config["ab"]["long_probe_horizons"])
        != (EPISODE_TICKS,)
    ):
        parser.error("prefix probe requires the bound 50k long endpoint")
    source_identity = _source_identity(config_snapshot.path)
    development_check = reserve._development_check_result()

    runtime_snapshot = r3d._snapshot_file(
        reserve._path(str(config["trusted_runtime"])),
        "prefix portable runtime",
    )
    if (
        runtime_snapshot.path != EXPECTED_RUNTIME.resolve(strict=True)
        or runtime_snapshot.path
        != reserve.TRUSTED_RUNTIME.resolve(strict=True)
        or runtime_snapshot.sha256 != EXPECTED_RUNTIME_SHA256
        or runtime_snapshot.sha256 != str(config["trusted_runtime_sha256"])
    ):
        parser.error("prefix probe portable runtime identity differs")
    base_snapshot = r3d._snapshot_file(
        reserve._path(str(config["base_policy"]["checkpoint"])),
        "prefix frozen v5",
    )
    if (
        base_snapshot.path != EXPECTED_FROZEN_V5.resolve(strict=True)
        or base_snapshot.path != reserve.FROZEN_V5.resolve(strict=True)
        or base_snapshot.sha256 != EXPECTED_FROZEN_V5_SHA256
        or base_snapshot.sha256 != str(config["base_policy"]["sha256"])
    ):
        parser.error("prefix probe frozen-v5 identity differs")

    registry = reserve._seed_registry(config)
    suite = registry["suites"]["long"]
    default_seed_count = int(config["ab"]["long_probe_seed_count"])
    seed_count = (
        default_seed_count if args.seed_count is None else args.seed_count
    )
    if (
        type(seed_count) is not int
        or not 1 <= seed_count <= len(suite["seeds"])
    ):
        parser.error("prefix seed count is outside the fixed long suite")
    seeds = tuple(int(value) for value in suite["seeds"][:seed_count])
    modes = tuple(DEFAULT_MODES if args.modes is None else args.modes)
    if len(set(modes)) != len(modes):
        parser.error("prefix modes must not repeat")

    torch.set_num_threads(int(config["distillation"]["torch_threads"]))
    with IrisuEnv(
        library_path=runtime_snapshot.path,
        physics_backend="portable",
        config={"max_episode_ticks": EPISODE_TICKS},
    ) as parity_env:
        parity_runner = parity_env.runner_identity_manifest()
        parity = _wait_split_parity(
            parity_env,
            seed=seeds[0],
            config_hash=int(parity_runner["config_hash"]),
        )

    teachers = {
        mode: reserve._teacher(config, mode)
        for mode in modes
        if mode != "base_v5"
    }
    trajectories = {
        mode: _evaluate_mode(
            mode=mode,
            config=config,
            runtime_path=runtime_snapshot.path,
            base_path=base_snapshot.path,
            base_sha256=base_snapshot.sha256,
            seeds=seeds,
            teacher=teachers.get(mode),
        )
        for mode in modes
    }

    _require_source_identity(source_identity, config_snapshot.path)
    r3d._require_unchanged(config_snapshot, "prefix config")
    r3d._require_unchanged(runtime_snapshot, "prefix portable runtime")
    r3d._require_unchanged(base_snapshot, "prefix frozen v5")
    content = {
        "schema": REPORT_SCHEMA,
        "version": PREFIX_VERSION,
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
        "source_identity": source_identity,
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
            "identity": "frozen-r3d-v5",
        },
        "development_checks": development_check,
        "seed_registry": registry,
        "active_suite": {
            "name": "long",
            "label": suite["label"],
            "suite_sha256": suite["sha256"],
            "seeds": list(seeds),
            "seed_count": seed_count,
        },
        "protocol": {
            "modes": list(modes),
            "default_mode_subset": args.modes is None,
            "exploratory_mode_subset": set(modes) != set(SAFE_MODES),
            "episode_ticks": EPISODE_TICKS,
            "checkpoints": list(CHECKPOINTS),
            "single_trajectory_per_mode_seed": True,
            "checkpoint_policy_queries": False,
            "checkpoint_snapshot_order": (
                "capture before any natural next decision at the same tick"
            ),
            "wait_split_rule": (
                "split only a WAIT primitive at an exact checkpoint; resume "
                "its original remaining ticks before the next policy decision"
            ),
            "endpoint_semantics": (
                "genuine prefixes from one 50k trajectory, not independently "
                "horizon-censored evaluations"
            ),
            "prefix_protocol_extension": True,
            "preregistered_long_endpoint_ticks": [EPISODE_TICKS],
            "query_stride_shots": int(
                config["oracle"]["query_stride_shots"]
            ),
            "maximum_search_queries_per_episode": int(
                config["oracle"]["maximum_search_queries_per_episode"]
            ),
            "terminal_carry_rule": (
                "after GAME_OVER or LEVEL_COMPLETED, retain the terminal public "
                "metrics at later requested checkpoints and mark them carried"
            ),
            "gauge_failure_signal": "actual public GAME_OVER events",
            "level_completion_signal": "actual public LEVEL_COMPLETED events",
        },
        "wait_split_parity": {
            "runner": parity_runner,
            **parity,
        },
        "teachers": {
            mode: {
                "identity": teacher.identity_manifest(),
                "sha256": teacher.sha256,
            }
            for mode, teacher in teachers.items()
        },
        "candidate_vocabulary_sha256": (
            geometry_candidate_vocabulary_sha256(
                reserve._teacher(
                    config, "reserve_band"
                ).config.candidate_config
            )
        ),
        "trajectories": trajectories,
        "execution": {
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
        },
    }
    report = {**content, "payload_sha256": _canonical_sha256(content)}
    default_name = f"long-prefix-{'-'.join(modes)}.json"
    output = _safe_output(
        config,
        (
            reserve._path(str(config["artifact_namespace"])) / default_name
            if args.result_out is None
            else args.result_out
        ),
    )
    r3e._atomic_write_json(output, report, overwrite=False)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
