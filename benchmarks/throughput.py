#!/usr/bin/env python3
"""Reproducible Python/native throughput benchmark for the provisional clone."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import platform
import re
import struct
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import (  # noqa: E402
    Action,
    ActionKind,
    IrisuEnv,
    PaddedVectorEnv,
    RandomPolicy,
    SyncVectorEnv,
    ThreadVectorEnv,
    __version__,
    find_library,
    load_profile,
)
import irisu_env.env as irisu_env_module  # noqa: E402


LABEL = "current provisional v2.03-normal profile; not fidelity-certified"
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1
ACTION_STRUCT = struct.Struct("<BddI")
ACTION_ENCODING = "base64 of repeated little-endian <uint8 kind,float64 x,float64 y,uint32 wait>"
PUBLIC_TRANSITION_DIGEST_METHOD = (
    "sha256-v1 over the irisu-cross-api-public-transition-v1 domain and "
    "uint64-le-length-prefixed canonical JSON reset/step records; keys sorted, "
    "typed views materialized to canonical dictionaries, records ordered by "
    "vector step then ascending environment index"
)
SOURCE_DIRS = ("clone", "python/irisu_env", "third_party/box2d_legacy")
SOURCE_FILES = (
    "CMakeLists.txt",
    "pyproject.toml",
    "benchmarks/native_physics.cpp",
    "benchmarks/throughput.py",
)
OPTIONAL_SOURCE_FILES = ("uv.lock",)


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hex64(value: int) -> str:
    return f"0x{value:016x}"


def rate(count: int, elapsed_ns: int) -> float:
    return round(count * 1_000_000_000.0 / max(1, elapsed_ns), 3)


def elapsed_seconds(elapsed_ns: int) -> float:
    return round(elapsed_ns / 1_000_000_000.0, 9)


def action_name(action: Action) -> str:
    return ActionKind.parse(action.kind).name.lower()


def append_action(trace: bytearray, action: Action) -> None:
    trace.extend(
        ACTION_STRUCT.pack(
            int(ActionKind.parse(action.kind)),
            float(action.cursor_x),
            float(action.cursor_y),
            int(action.wait_ticks),
        )
    )


def encoded_trace(
    actions: bytearray,
    resets: list[dict[str, int]],
    *,
    ordering: str,
) -> dict[str, Any]:
    reset_bytes = json.dumps(resets, sort_keys=True, separators=(",", ":")).encode("utf-8")
    combined = hashlib.sha256(
        b"irisu-action-reset-trace-v1\0" + actions + b"\0" + reset_bytes
    ).hexdigest()
    return {
        "action_count": len(actions) // ACTION_STRUCT.size,
        "action_encoding": ACTION_ENCODING,
        "actions_base64": base64.b64encode(actions).decode("ascii"),
        "actions_sha256": hashlib.sha256(actions).hexdigest(),
        "ordering": ordering,
        "reset_markers": resets,
        "reset_markers_sha256": hashlib.sha256(reset_bytes).hexdigest(),
        "trace_sha256": combined,
        "trace_schema_version": 1,
    }


def canonical_public_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return canonical_public_value(to_dict())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("public transition mappings must use string keys")
        return {key: canonical_public_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return [canonical_public_value(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        resolved = item()
        if resolved is not value:
            return canonical_public_value(resolved)
    raise TypeError(f"unsupported public transition value {type(value).__name__}")


def canonical_observation(value: Any) -> dict[str, Any]:
    result = canonical_public_value(value)
    if not isinstance(result, dict):
        raise TypeError("public observation must normalize to a mapping")
    for key in ("terminated", "truncated", "left_held", "right_held"):
        result[key] = bool(result[key])
    for key in (
        "x",
        "y",
        "width",
        "height",
        "side_wall_top",
        "side_wall_bottom",
    ):
        result["field"][key] = float(result["field"][key])
    for body in result["bodies"]:
        for key in ("x", "y", "vx", "vy", "angle", "angular_velocity", "size"):
            body[key] = float(body[key])
    return result


def canonical_info(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    if "events" in result:
        result["events"] = [canonical_public_value(event) for event in result["events"]]
    diagnostics = result.get("diagnostics")
    to_diagnostics = getattr(diagnostics, "diagnostics", None)
    if callable(to_diagnostics):
        result["diagnostics"] = to_diagnostics()
    normalized = canonical_public_value(result)
    if not isinstance(normalized, dict):
        raise TypeError("public info must normalize to a mapping")
    if "invalid_action" in normalized:
        normalized["invalid_action"] = bool(normalized["invalid_action"])
    return normalized


def update_public_transition_digest(
    digest: Any, record: Mapping[str, Any]
) -> None:
    payload = json.dumps(
        canonical_public_value(record),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def _cross_api_public_transition_workload(
    vector: SyncVectorEnv | PaddedVectorEnv, calls: int, seed: int
) -> dict[str, Any]:
    seeds = [(seed + index) & MASK32 for index in range(vector.num_envs)]
    policy_seeds = [
        (seed ^ (0x564543544F52 + index)) & MASK64
        for index in range(vector.num_envs)
    ]
    policies = [RandomPolicy(value) for value in policy_seeds]
    observations, reset_infos = vector.reset(seed=seeds)
    action_trace = bytearray()
    reset_markers: list[dict[str, int]] = []
    episode_counts = [0] * vector.num_envs
    digest = hashlib.sha256(b"irisu-cross-api-public-transition-v1\0")
    record_count = 0

    initial_hashes = vector.state_hash()
    for index in range(vector.num_envs):
        marker = {
            "before_vector_step": 0,
            "env_index": index,
            "env_seed": seeds[index],
            "policy_seed": policy_seeds[index],
        }
        reset_markers.append(marker)
        update_public_transition_digest(
            digest,
            {
                "record": "reset",
                **marker,
                "info": canonical_info(reset_infos[index]),
                "observation": canonical_observation(observations[index]),
                "state_hash": hex64(initial_hashes[index]),
            },
        )
        record_count += 1

    for call in range(calls):
        actions = [
            policy.act(observation)
            for policy, observation in zip(policies, observations)
        ]
        for action in actions:
            append_action(action_trace, action)
        observations, rewards, terminated, truncated, infos = vector.step(actions)
        normalized_observations = [canonical_observation(value) for value in observations]
        normalized_infos = [canonical_info(value) for value in infos]
        state_hashes = vector.state_hash()
        for index in range(vector.num_envs):
            action = actions[index]
            update_public_transition_digest(
                digest,
                {
                    "action": {
                        "kind": int(ActionKind.parse(action.kind)),
                        "cursor_x": float(action.cursor_x),
                        "cursor_y": float(action.cursor_y),
                        "wait_ticks": int(action.wait_ticks),
                    },
                    "env_index": index,
                    "info": normalized_infos[index],
                    "observation": normalized_observations[index],
                    "record": "step",
                    "reward": canonical_public_value(rewards[index]),
                    "state_hash": hex64(state_hashes[index]),
                    "terminated": bool(terminated[index]),
                    "truncated": bool(truncated[index]),
                    "vector_step": call,
                },
            )
            record_count += 1

        if call + 1 >= calls:
            continue
        for index, done in enumerate(
            ended or limited for ended, limited in zip(terminated, truncated)
        ):
            if not done:
                continue
            episode_counts[index] += 1
            episode = episode_counts[index]
            next_env_seed = (seed + episode * vector.num_envs + index) & MASK32
            next_policy_seed = (
                policy_seeds[index] + episode * 0x9E3779B97F4A7C15
            ) & MASK64
            if isinstance(vector, PaddedVectorEnv):
                observation = vector.reset_at(index, seed=next_env_seed)
                reset_info = {
                    "seed": next_env_seed,
                    "config_hash": vector.envs[index].config_hash(),
                }
            else:
                observation, reset_info = vector.envs[index].reset(
                    seed=next_env_seed
                )
            observations[index] = observation
            policies[index].reset(next_policy_seed)
            marker = {
                "before_vector_step": call + 1,
                "env_index": index,
                "env_seed": next_env_seed,
                "policy_seed": next_policy_seed,
            }
            reset_markers.append(marker)
            update_public_transition_digest(
                digest,
                {
                    "record": "reset",
                    **marker,
                    "info": canonical_info(reset_info),
                    "observation": canonical_observation(observation),
                    "state_hash": hex64(vector.envs[index].state_hash()),
                },
            )
            record_count += 1

    trace = encoded_trace(
        action_trace,
        reset_markers,
        ordering="vector-step-major, then ascending env_index",
    )
    return {
        "action_reset_trace_sha256": trace["trace_sha256"],
        "final_state_hashes": [hex64(value) for value in vector.state_hash()],
        "record_count": record_count,
        "sha256": digest.hexdigest(),
    }


def verify_cross_api_public_transitions(
    library: str | os.PathLike[str],
    num_envs: int,
    calls: int,
    seed: int,
    workers: int | None,
) -> dict[str, Any]:
    results: dict[str, dict[str, Any]] = {}
    with SyncVectorEnv(num_envs, library_path=library) as vector:
        results["sync_vector"] = _cross_api_public_transition_workload(
            vector, calls, seed
        )
    with ThreadVectorEnv(
        num_envs, library_path=library, workers=workers
    ) as vector:
        results["thread_vector"] = _cross_api_public_transition_workload(
            vector, calls, seed
        )
    with PaddedVectorEnv(
        num_envs, library_path=library, workers=workers
    ) as vector:
        results["padded_vector"] = _cross_api_public_transition_workload(
            vector, calls, seed
        )

    signatures = {
        (
            result["sha256"],
            result["record_count"],
            result["action_reset_trace_sha256"],
            tuple(result["final_state_hashes"]),
        )
        for result in results.values()
    }
    if len(signatures) != 1:
        summaries = {
            name: {
                "action_reset_trace_sha256": result["action_reset_trace_sha256"],
                "record_count": result["record_count"],
                "sha256": result["sha256"],
            }
            for name, result in results.items()
        }
        raise RuntimeError(
            "vector APIs diverged on the untimed public transition replay: "
            + json.dumps(summaries, sort_keys=True)
        )
    shared = results["sync_vector"]
    return {
        "action_reset_trace_sha256": shared["action_reset_trace_sha256"],
        "method": PUBLIC_TRANSITION_DIGEST_METHOD,
        "record_count": shared["record_count"],
        "sha256": shared["sha256"],
        "status": "passed",
    }


def _single_workload(
    env: IrisuEnv, decisions: int, env_seed: int, policy_seed: int
) -> dict[str, Any]:
    policy = RandomPolicy(policy_seed)
    observation, _ = env.reset(seed=env_seed)
    actions: Counter[str] = Counter()
    action_trace = bytearray()
    reset_markers = [
        {
            "before_decision": 0,
            "env_seed": env_seed,
            "policy_seed": policy_seed,
        }
    ]
    simulation_ticks = 0
    total_reward = 0
    resets = 0
    for decision in range(decisions):
        action = policy.act(observation)
        actions[action_name(action)] += 1
        append_action(action_trace, action)
        previous_tick = int(observation["tick"])
        observation, reward, terminated, truncated, _ = env.step(action)
        simulation_ticks += int(observation["tick"]) - previous_tick
        total_reward += reward
        if (terminated or truncated) and decision + 1 < decisions:
            resets += 1
            next_env_seed = (env_seed + resets) & MASK32
            next_policy_seed = (policy_seed + resets) & MASK64
            observation, _ = env.reset(seed=next_env_seed)
            policy.reset(next_policy_seed)
            reset_markers.append(
                {
                    "before_decision": decision + 1,
                    "env_seed": next_env_seed,
                    "policy_seed": next_policy_seed,
                }
            )
    return {
        "action_counts": dict(sorted(actions.items())),
        "action_reset_trace": encoded_trace(
            action_trace, reset_markers, ordering="decision-major"
        ),
        "decision_steps": decisions,
        "episode_resets": resets,
        "final_state_hash": hex64(env.state_hash()),
        "simulation_ticks": simulation_ticks,
        "total_reward": total_reward,
    }


def benchmark_single(
    library: str | os.PathLike[str], decisions: int, warmup: int, seed: int
) -> dict[str, Any]:
    with IrisuEnv(library_path=library) as env:
        _single_workload(env, warmup, seed, seed ^ 0x53494E474C45)
        start = time.perf_counter_ns()
        result = _single_workload(env, decisions, seed, seed ^ 0x53494E474C45)
        elapsed = time.perf_counter_ns() - start
    result.update(
        {
            "elapsed_seconds": elapsed_seconds(elapsed),
            "decision_steps_per_second": rate(decisions, elapsed),
            "simulation_ticks_per_second": rate(result["simulation_ticks"], elapsed),
        }
    )
    return result


def _vector_workload(
    vector: SyncVectorEnv, calls: int, seed: int
) -> dict[str, Any]:
    seeds = [(seed + index) & MASK32 for index in range(vector.num_envs)]
    policy_seeds = [
        (seed ^ (0x564543544F52 + index)) & MASK64
        for index in range(vector.num_envs)
    ]
    policies = [RandomPolicy(value) for value in policy_seeds]
    observations, _ = vector.reset(seed=seeds)
    actions_seen: Counter[str] = Counter()
    action_trace = bytearray()
    reset_markers = [
        {
            "before_vector_step": 0,
            "env_index": index,
            "env_seed": seeds[index],
            "policy_seed": policy_seeds[index],
        }
        for index in range(vector.num_envs)
    ]
    episode_counts = [0] * vector.num_envs
    body_counts = [len(observation["bodies"]) for observation in observations]
    simulation_ticks = 0
    total_reward = 0
    for call in range(calls):
        actions = [policy.act(observation) for policy, observation in zip(policies, observations)]
        actions_seen.update(action_name(action) for action in actions)
        for action in actions:
            append_action(action_trace, action)
        previous_ticks = [int(observation["tick"]) for observation in observations]
        observations, rewards, terminated, truncated, _ = vector.step(actions)
        body_counts.extend(len(observation["bodies"]) for observation in observations)
        simulation_ticks += sum(
            int(observation["tick"]) - previous
            for observation, previous in zip(observations, previous_ticks)
        )
        total_reward += sum(rewards)
        if call + 1 < calls:
            for index, done in enumerate(
                ended or limited for ended, limited in zip(terminated, truncated)
            ):
                if not done:
                    continue
                episode_counts[index] += 1
                episode = episode_counts[index]
                next_env_seed = (seed + episode * vector.num_envs + index) & MASK32
                next_policy_seed = (
                    policy_seeds[index] + episode * 0x9E3779B97F4A7C15
                ) & MASK64
                observations[index], _ = vector.envs[index].reset(seed=next_env_seed)
                policies[index].reset(next_policy_seed)
                reset_markers.append(
                    {
                        "before_vector_step": call + 1,
                        "env_index": index,
                        "env_seed": next_env_seed,
                        "policy_seed": next_policy_seed,
                    }
                )
    return {
        "action_counts": dict(sorted(actions_seen.items())),
        "action_reset_trace": encoded_trace(
            action_trace,
            reset_markers,
            ordering="vector-step-major, then ascending env_index",
        ),
        "aggregate_decision_steps": calls * vector.num_envs,
        "episode_resets": sum(episode_counts),
        "episode_resets_by_env": episode_counts,
        "observed_body_count_max": max(body_counts),
        "observed_body_count_mean": round(sum(body_counts) / len(body_counts), 3),
        "observed_body_count_min": min(body_counts),
        "final_state_hashes": [hex64(value) for value in vector.state_hash()],
        "num_envs": vector.num_envs,
        "simulation_ticks": simulation_ticks,
        "total_reward": total_reward,
        "vector_step_calls": calls,
    }


def benchmark_vector(
    library: str | os.PathLike[str],
    num_envs: int,
    calls: int,
    warmup: int,
    seed: int,
    *,
    threaded: bool = False,
    workers: int | None = None,
) -> dict[str, Any]:
    vector_type = ThreadVectorEnv if threaded else SyncVectorEnv
    options = {"workers": workers} if threaded else {}
    with vector_type(num_envs, library_path=library, **options) as vector:
        _vector_workload(vector, warmup, seed)
        start = time.perf_counter_ns()
        result = _vector_workload(vector, calls, seed)
        elapsed = time.perf_counter_ns() - start
    result.update(
        {
            "execution": "thread_pool" if threaded else "sequential",
            "elapsed_seconds": elapsed_seconds(elapsed),
            "aggregate_decision_steps_per_second": rate(
                result["aggregate_decision_steps"], elapsed
            ),
            "simulation_ticks_per_second": rate(result["simulation_ticks"], elapsed),
            "vector_step_calls_per_second": rate(calls, elapsed),
        }
    )
    if threaded:
        result["workers"] = min(num_envs, workers or num_envs)
    return result


def _padded_vector_workload(
    vector: PaddedVectorEnv, calls: int, seed: int
) -> dict[str, Any]:
    seeds = [(seed + index) & MASK32 for index in range(vector.num_envs)]
    policy_seeds = [
        (seed ^ (0x564543544F52 + index)) & MASK64
        for index in range(vector.num_envs)
    ]
    policies = [RandomPolicy(value) for value in policy_seeds]
    observations, _ = vector.reset(seed=seeds)
    actions_seen: Counter[str] = Counter()
    action_trace = bytearray()
    reset_markers = [
        {
            "before_vector_step": 0,
            "env_index": index,
            "env_seed": seeds[index],
            "policy_seed": policy_seeds[index],
        }
        for index in range(vector.num_envs)
    ]
    episode_counts = [0] * vector.num_envs
    body_counts = [int(observation.body_count) for observation in observations]
    simulation_ticks = 0
    total_reward = 0
    total_events = 0
    for call in range(calls):
        actions = [policy.act(observation) for policy, observation in zip(policies, observations)]
        actions_seen.update(action_name(action) for action in actions)
        for action in actions:
            append_action(action_trace, action)
        previous_ticks = [int(observation.tick) for observation in observations]
        observations, rewards, terminated, truncated, infos = vector.step(actions)
        simulation_ticks += sum(
            int(observation.tick) - previous
            for observation, previous in zip(observations, previous_ticks)
        )
        total_reward += sum(rewards)
        total_events += sum(len(info["events"]) for info in infos)
        body_counts.extend(int(observation.body_count) for observation in observations)
        if call + 1 < calls:
            for index, done in enumerate(
                ended or limited for ended, limited in zip(terminated, truncated)
            ):
                if not done:
                    continue
                episode_counts[index] += 1
                episode = episode_counts[index]
                next_env_seed = (seed + episode * vector.num_envs + index) & MASK32
                next_policy_seed = (
                    policy_seeds[index] + episode * 0x9E3779B97F4A7C15
                ) & MASK64
                observations[index] = vector.reset_at(index, seed=next_env_seed)
                policies[index].reset(next_policy_seed)
                reset_markers.append(
                    {
                        "before_vector_step": call + 1,
                        "env_index": index,
                        "env_seed": next_env_seed,
                        "policy_seed": next_policy_seed,
                    }
                )
    return {
        "action_counts": dict(sorted(actions_seen.items())),
        "action_reset_trace": encoded_trace(
            action_trace,
            reset_markers,
            ordering="vector-step-major, then ascending env_index",
        ),
        "aggregate_decision_steps": calls * vector.num_envs,
        "body_capacity": vector.body_capacity,
        "episode_resets": sum(episode_counts),
        "episode_resets_by_env": episode_counts,
        "event_count": total_events,
        "final_state_hashes": [hex64(value) for value in vector.state_hash()],
        "num_envs": vector.num_envs,
        "observation_encoding": "typed_padded_v1_with_explicit_body_count_mask",
        "observed_body_count_max": max(body_counts),
        "observed_body_count_mean": round(sum(body_counts) / len(body_counts), 3),
        "observed_body_count_min": min(body_counts),
        "simulation_ticks": simulation_ticks,
        "total_reward": total_reward,
        "vector_step_calls": calls,
    }


def benchmark_padded_vector(
    library: str | os.PathLike[str],
    num_envs: int,
    calls: int,
    warmup: int,
    seed: int,
    workers: int | None = None,
) -> dict[str, Any]:
    with PaddedVectorEnv(
        num_envs, library_path=library, workers=workers
    ) as vector:
        _padded_vector_workload(vector, warmup, seed)
        start = time.perf_counter_ns()
        result = _padded_vector_workload(vector, calls, seed)
        elapsed = time.perf_counter_ns() - start
    result.update(
        {
            "execution": "native_persistent_batch_pool",
            "elapsed_seconds": elapsed_seconds(elapsed),
            "aggregate_decision_steps_per_second": rate(
                result["aggregate_decision_steps"], elapsed
            ),
            "simulation_ticks_per_second": rate(result["simulation_ticks"], elapsed),
            "vector_step_calls_per_second": rate(calls, elapsed),
            "workers": vector.workers,
        }
    )
    return result


def benchmark_snapshots(
    library: str | os.PathLike[str], iterations: int, warmup: int, seed: int
) -> dict[str, Any]:
    with IrisuEnv(library_path=library) as env:
        env.reset(seed=seed)
        preparation = Action.wait(50)
        env.step(preparation)
        action_trace = bytearray()
        append_action(action_trace, preparation)
        baseline = env.clone_state()
        for _ in range(warmup):
            env.clone_state()
            env.restore_state(baseline)

        start = time.perf_counter_ns()
        clones = [env.clone_state() for _ in range(iterations)]
        clone_elapsed = time.perf_counter_ns() - start
        if any(snapshot != baseline for snapshot in clones):
            raise RuntimeError("snapshot cloning changed an otherwise stationary state")

        start = time.perf_counter_ns()
        for _ in range(iterations):
            env.restore_state(baseline)
        restore_elapsed = time.perf_counter_ns() - start
        result = {
            "action_reset_trace": encoded_trace(
                action_trace,
                [{"before_decision": 0, "env_seed": seed}],
                ordering="single preparation action before snapshot timing",
            ),
            "clone_elapsed_seconds": elapsed_seconds(clone_elapsed),
            "clone_operations_per_second": rate(iterations, clone_elapsed),
            "final_state_hash": hex64(env.state_hash()),
            "iterations": iterations,
            "restore_elapsed_seconds": elapsed_seconds(restore_elapsed),
            "restore_operations_per_second": rate(iterations, restore_elapsed),
            "snapshot_bytes": len(baseline),
            "snapshot_sha256": hashlib.sha256(baseline).hexdigest(),
        }
    return result


def physics_benchmark_path(library: Path) -> Path:
    candidates = (
        library.parent / "irisu-physics-benchmark",
        library.parent / "irisu-physics-benchmark.exe",
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        "native physics benchmark is missing; build target irisu-physics-benchmark"
    )


def benchmark_physics(
    library: Path,
    *,
    ticks: int,
    warmup: int,
    bodies: int,
    seed: int,
) -> dict[str, Any]:
    executable = physics_benchmark_path(library)
    executable_hash = sha256_file(executable)
    completed = subprocess.run(
        [
            str(executable),
            "--ticks",
            str(ticks),
            "--warmup",
            str(warmup),
            "--bodies",
            str(bodies),
            "--seed",
            str(seed),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    if executable_hash != sha256_file(executable):
        raise RuntimeError("native physics benchmark executable changed while running")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("native physics benchmark returned invalid JSON") from exc
    expected = {
        "body_count": bodies,
        "physics_ticks": ticks,
        "schema_version": 2,
        "seed": seed,
        "warmup_ticks": warmup,
        "workload": "legacy_physics_typical_board_v2",
    }
    if not isinstance(result, dict) or any(result.get(key) != value for key, value in expected.items()):
        raise RuntimeError("native physics benchmark result disagrees with its invocation")
    result.update(
        {
            "executable_bytes": executable.stat().st_size,
            "executable_path": str(executable),
            "executable_sha256": executable_hash,
            "stderr": completed.stderr,
        }
    )
    return result


def cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def git_revision() -> str | None:
    try:
        head = (ROOT / ".git" / "HEAD").read_text(encoding="ascii").strip()
        if head.startswith("ref: "):
            target = ROOT / ".git" / head[5:]
            return target.read_text(encoding="ascii").strip() if target.is_file() else None
        return head or None
    except OSError:
        return None


def git_metadata() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        status = result.stdout.encode("utf-8")
        dirty: bool | None = bool(status)
        status_sha256: str | None = hashlib.sha256(status).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        dirty = None
        status_sha256 = None
    return {
        "git_dirty": dirty,
        "git_revision": git_revision(),
        "git_status_sha256": status_sha256,
    }


def optional_file_metadata(path: Path) -> dict[str, Any]:
    expected_path = path.absolute()
    if not path.exists():
        if path.is_symlink():
            raise RuntimeError(f"dependency identity path is a broken symlink: {path}")
        return {
            "status": "absent",
            "path": str(expected_path),
            "bytes": None,
            "sha256": None,
        }
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise RuntimeError(f"dependency identity path is not a file: {resolved}")
    before = resolved.stat()
    digest = sha256_file(resolved)
    after = resolved.stat()

    def fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )

    if fingerprint(before) != fingerprint(after):
        raise RuntimeError(f"dependency identity file changed while hashing: {resolved}")
    return {
        "status": "present",
        "path": str(resolved),
        "bytes": after.st_size,
        "sha256": digest,
    }


def package_runtime_metadata(
    distribution: str, module: Any, *, enabled: bool
) -> dict[str, Any]:
    try:
        distribution_version = importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        distribution_version = None
    loaded_version = (
        getattr(module, "__version__", None) if module is not None else None
    )
    return {
        "distribution_version": distribution_version,
        "enabled_for_observation_conversion": enabled,
        "installed": distribution_version is not None,
        "loaded_module_version": (
            str(loaded_version) if loaded_version is not None else None
        ),
    }


def runtime_dependency_metadata() -> dict[str, Any]:
    gymnasium_module = irisu_env_module._gym
    gymnasium_enabled = gymnasium_module is not None
    numpy_enabled = gymnasium_enabled
    numpy_module = sys.modules.get("numpy")
    if numpy_enabled and numpy_module is None:
        raise RuntimeError(
            "Gymnasium observation conversion is enabled but NumPy is not loaded"
        )
    return {
        "observation_conversion": {
            "mode": (
                "gymnasium_numpy_arrays"
                if gymnasium_enabled
                else "plain_python_values"
            ),
            "gymnasium_enabled": gymnasium_enabled,
            "numpy_enabled": numpy_enabled,
            "padded_vector_mode": "ctypes_typed_padded_v1",
        },
        "packages": {
            "gymnasium": package_runtime_metadata(
                "gymnasium", gymnasium_module, enabled=gymnasium_enabled
            ),
            "numpy": package_runtime_metadata(
                "numpy", numpy_module, enabled=numpy_enabled
            ),
        },
        "uv_lock": optional_file_metadata(ROOT / "uv.lock"),
    }


def source_metadata() -> dict[str, Any]:
    paths = [ROOT / name for name in SOURCE_FILES]
    paths.extend(
        ROOT / name for name in OPTIONAL_SOURCE_FILES if (ROOT / name).is_file()
    )
    for directory in SOURCE_DIRS:
        paths.extend(path for path in (ROOT / directory).rglob("*") if path.is_file())
    files = {
        path.relative_to(ROOT).as_posix(): sha256_file(path)
        for path in sorted(set(paths))
        if "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pyo")
    }
    serialized = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "source_file_count": len(files),
        "source_files_sha256": files,
        "source_manifest_sha256": hashlib.sha256(serialized).hexdigest(),
    }


def cmake_metadata(library: Path) -> dict[str, Any]:
    cache_path = library.parent / "CMakeCache.txt"
    wanted = {
        "CMAKE_BUILD_TYPE",
        "CMAKE_CXX_COMPILER",
        "CMAKE_CXX_FLAGS",
        "CMAKE_GENERATOR",
        "CMAKE_SHARED_LINKER_FLAGS",
    }
    values: dict[str, str] = {}
    try:
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if "=" not in line or line.startswith(("//", "#")):
                continue
            typed_key, value = line.split("=", 1)
            key = typed_key.split(":", 1)[0]
            if key in wanted or key.startswith(("CMAKE_CXX_FLAGS_", "CMAKE_SHARED_LINKER_FLAGS_")):
                values[key] = value
    except OSError:
        pass
    compiler_details: dict[str, str] = {}
    for path in sorted((library.parent / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake")):
        try:
            contents = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for key in ("CMAKE_CXX_COMPILER_ID", "CMAKE_CXX_COMPILER_VERSION"):
            match = re.search(rf'^set\({key} "([^"]*)"\)', contents, re.MULTILINE)
            if match:
                compiler_details[key] = match.group(1)
        break
    build_type = values.get("CMAKE_BUILD_TYPE", "")
    compiler_path = Path(values["CMAKE_CXX_COMPILER"]).resolve() if values.get("CMAKE_CXX_COMPILER") else None
    result: dict[str, Any] = {
        "cmake_build_type": values.get("CMAKE_BUILD_TYPE") or None,
        "cmake_cache_sha256": sha256_file(cache_path) if cache_path.is_file() else None,
        "cmake_generator": values.get("CMAKE_GENERATOR") or None,
        "cxx_compiler": str(compiler_path) if compiler_path else None,
        "cxx_compiler_id": compiler_details.get("CMAKE_CXX_COMPILER_ID") or None,
        "cxx_compiler_sha256": sha256_file(compiler_path) if compiler_path and compiler_path.is_file() else None,
        "cxx_compiler_version": compiler_details.get("CMAKE_CXX_COMPILER_VERSION") or None,
        "cxx_flags": values.get("CMAKE_CXX_FLAGS", ""),
        "cxx_flags_for_build_type": values.get(f"CMAKE_CXX_FLAGS_{build_type.upper()}", ""),
        "shared_linker_flags": values.get("CMAKE_SHARED_LINKER_FLAGS", ""),
        "shared_linker_flags_for_build_type": values.get(
            f"CMAKE_SHARED_LINKER_FLAGS_{build_type.upper()}", ""
        ),
    }
    return result


def metadata(library: Path, seed: int) -> dict[str, Any]:
    profile_path = ROOT / "configs" / "v2.03-normal.toml"
    profile = load_profile(profile_path)
    implementation_json = json.dumps(
        profile.implementation_mapping(), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    with IrisuEnv(library_path=library) as env:
        env.reset(seed=seed)
        native_config_hash = env.config_hash()
        native_config = env.config
        native_build_info = env.build_info
    if int(native_config["config_hash"]) != native_config_hash:
        raise RuntimeError("native config JSON/hash disagreement")
    try:
        logical_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        logical_cpus = os.cpu_count()
    build = {
        "benchmark_script_sha256": sha256_file(Path(__file__)),
        "library_bytes": library.stat().st_size,
        "library_path": str(library),
        "library_sha256": sha256_file(library),
        "native_build_info": native_build_info,
        "python_package_version": __version__,
    }
    try:
        native_benchmark = physics_benchmark_path(library)
    except FileNotFoundError:
        native_benchmark = None
    build.update(
        {
            "native_physics_benchmark_bytes": (
                native_benchmark.stat().st_size if native_benchmark else None
            ),
            "native_physics_benchmark_path": (
                str(native_benchmark) if native_benchmark else None
            ),
            "native_physics_benchmark_sha256": (
                sha256_file(native_benchmark) if native_benchmark else None
            ),
        }
    )
    build.update(git_metadata())
    build.update(source_metadata())
    build.update(cmake_metadata(library))
    return {
        "build": build,
        "configuration": {
            "fidelity_certified": False,
            "implementation_parameter_count": len(profile.implementation_parameters),
            "implementation_parameters_sha256": hashlib.sha256(implementation_json).hexdigest(),
            "native_config_hash": hex64(native_config_hash),
            "native_config": native_config,
            "profile_file": str(profile_path),
            "profile_file_sha256": sha256_file(profile_path),
            "profile_id": profile.profile_id,
            "profile_status": profile.status,
            "provisional": True,
        },
        "host": {
            "cpu_model": cpu_model(),
            "hostname": platform.node(),
            "logical_cpus_available": logical_cpus,
            "machine": platform.machine(),
            "operating_system": platform.platform(),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "runtime_dependencies": runtime_dependency_metadata(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--single-steps", type=positive, default=5_000)
    parser.add_argument("--physics-ticks", type=positive, default=5_000)
    parser.add_argument("--physics-warmup", type=positive, default=200)
    parser.add_argument("--physics-bodies", type=positive, default=48)
    parser.add_argument("--vector-steps", type=positive, default=1_000)
    parser.add_argument("--num-envs", type=positive, default=8)
    parser.add_argument(
        "--thread-workers",
        type=positive,
        help="worker count for ThreadVectorEnv (default: one per environment)",
    )
    parser.add_argument("--snapshot-iterations", type=positive, default=2_000)
    parser.add_argument("--warmup", type=positive, default=50)
    parser.add_argument("--seed", type=int, default=20_260_717)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 0 <= args.seed <= MASK32:
        parser.error("--seed must fit in uint32")
    if args.physics_bodies > 96:
        parser.error("--physics-bodies must be at most 96")
    return args


def main() -> None:
    args = parse_args()
    library = Path(find_library(args.library)).resolve()
    before = metadata(library, args.seed)
    report: dict[str, Any] = {
        "schema_version": 3,
        "label": LABEL,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "parameters": {
            "num_envs": args.num_envs,
            "physics_bodies": args.physics_bodies,
            "physics_ticks": args.physics_ticks,
            "physics_warmup": args.physics_warmup,
            "policy": {
                "max_wait_ticks": 5,
                "name": "RandomPolicy",
                "prng": "SplitMix64",
                "shot_probability": 0.25,
                "strong_probability": 0.5,
            },
            "seed": args.seed,
            "single_steps": args.single_steps,
            "snapshot_iterations": args.snapshot_iterations,
            "thread_workers": min(args.num_envs, args.thread_workers or args.num_envs),
            "vector_steps": args.vector_steps,
            "warmup": args.warmup,
        },
    }
    report["results"] = {
        "physics_only": benchmark_physics(
            library,
            ticks=args.physics_ticks,
            warmup=args.physics_warmup,
            bodies=args.physics_bodies,
            seed=args.seed,
        ),
        "single_env": benchmark_single(library, args.single_steps, args.warmup, args.seed),
        "snapshot": benchmark_snapshots(
            library, args.snapshot_iterations, args.warmup, args.seed
        ),
        "padded_vector": benchmark_padded_vector(
            library,
            args.num_envs,
            args.vector_steps,
            args.warmup,
            args.seed,
            workers=args.thread_workers,
        ),
        "sync_vector": benchmark_vector(
            library, args.num_envs, args.vector_steps, args.warmup, args.seed
        ),
        "thread_vector": benchmark_vector(
            library,
            args.num_envs,
            args.vector_steps,
            args.warmup,
            args.seed,
            threaded=True,
            workers=args.thread_workers,
        ),
    }
    public_transition_equivalence = verify_cross_api_public_transitions(
        library,
        args.num_envs,
        args.vector_steps,
        args.seed,
        args.thread_workers,
    )
    after = metadata(library, args.seed)
    if before != after:
        raise RuntimeError(
            "benchmark inputs or build metadata changed during the run; discard and rerun"
        )
    report.update(before)
    report["build"]["pre_post_stability_check"] = "passed"
    vector_results = [
        report["results"][name]
        for name in ("sync_vector", "thread_vector", "padded_vector")
    ]
    trace_hashes = {
        result["action_reset_trace"]["trace_sha256"] for result in vector_results
    }
    state_hashes = {
        tuple(result["final_state_hashes"]) for result in vector_results
    }
    if len(trace_hashes) != 1 or len(state_hashes) != 1:
        raise RuntimeError("vector APIs diverged on the benchmark action trace")
    report["build"]["cross_api_trace_equivalence"] = "passed"
    timed_trace_hash = next(iter(trace_hashes))
    if public_transition_equivalence["action_reset_trace_sha256"] != timed_trace_hash:
        raise RuntimeError(
            "untimed public transition replay used a different action/reset trace"
        )
    report["build"][
        "cross_api_public_transition_equivalence"
    ] = public_transition_equivalence
    vector_rates = {
        name: report["results"][name]["aggregate_decision_steps_per_second"]
        for name in ("sync_vector", "thread_vector", "padded_vector")
    }
    target = 20_000.0
    best_path = max(vector_rates, key=lambda name: vector_rates[name])
    best = vector_rates[best_path]
    report["training_readiness"] = {
        "accepted_for_bulk_training": False,
        "best_aggregate_decision_steps_per_second": best,
        "fidelity_gate_met": False,
        "performance_gate_met": best >= target,
        "performance_target_decision_steps_per_second": target,
        "performance_path": best_path,
        "physics_only_measurement_present": True,
        "status": "provisional-not-ready",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)


if __name__ == "__main__":
    main()
