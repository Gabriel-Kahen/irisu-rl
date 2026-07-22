#!/usr/bin/env python3
"""Controlled synchronous-barrier versus exact actor-rollout comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import ExactSimulator, PaddedVectorEnv, RandomPolicy  # noqa: E402
from irisu_env.rollout import ExactActorRolloutPool  # noqa: E402


CONFIG = {
    "gauge_initial": 1_000_000_000_000,
    "gauge_max": 1_000_000_000_000,
    "passive_gauge_decay_per_tick": 0,
    "qualifying_clears_per_level": 0xFFFFFFFF,
    "rotten_penalty": 0,
    "max_episode_ticks": 0xFFFFFFFF,
}

SOURCE_FILES = (
    "benchmarks/exact_actor_rollout.py",
    "python/irisu_env/exact_ipc.py",
    "python/irisu_env/padded.py",
    "python/irisu_env/rollout.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def policies(lanes: int) -> list[RandomPolicy]:
    return [
        RandomPolicy(0x52574157 + lane, max_wait_ticks=1)
        for lane in range(lanes)
    ]


def synchronous(
    worker: Path, lanes: int, steps: int, warmup: int
) -> dict[str, Any]:
    digests = [hashlib.sha256() for _ in range(lanes)]
    event_counts = [0] * lanes
    with PaddedVectorEnv(
        lanes,
        physics_backend="exact",
        worker_path=worker,
        workers=lanes,
        config=CONFIG,
    ) as vector:
        observations, _ = vector.reset(seed=41)
        lane_policies = policies(lanes)
        for _ in range(warmup):
            actions = [
                policy.act(observation)
                for policy, observation in zip(lane_policies, observations)
            ]
            observations, _, terminated, truncated, _ = vector.step(actions)
            if any(terminated) or any(truncated):
                raise RuntimeError("nonterminating synchronous warmup ended")
        started = time.perf_counter_ns()
        for _ in range(steps):
            actions = [
                policy.act(observation)
                for policy, observation in zip(lane_policies, observations)
            ]
            observations, _, terminated, truncated, infos = vector.step(actions)
            if any(terminated) or any(truncated):
                raise RuntimeError("nonterminating synchronous workload ended")
            for lane, (env, info) in enumerate(zip(vector.envs, infos)):
                raw = env._raw_observation
                assert raw is not None
                digests[lane].update(raw[0])
                event_counts[lane] += len(info["events"])
        elapsed_ns = time.perf_counter_ns() - started
        hashes = vector.state_hash()
    return {
        "mode": "synchronous_step_barrier",
        "elapsed_seconds": elapsed_ns / 1e9,
        "operations_per_second": lanes * steps * 1e9 / elapsed_ns,
        "digests": [value.hexdigest() for value in digests],
        "state_hashes": list(hashes),
        "event_counts": event_counts,
    }


def actor(
    worker: Path, lanes: int, steps: int, rollout_horizon: int, warmup: int
) -> dict[str, Any]:
    digests = [hashlib.sha256() for _ in range(lanes)]
    event_counts = [0] * lanes
    with ExactActorRolloutPool(
        lanes,
        worker_path=worker,
        workers=lanes,
        config=CONFIG,
    ) as pool:
        pool.reset(seed=41)
        lane_policies = policies(lanes)
        warmed = pool.collect(lane_policies, warmup)
        if any(len(rollout.steps) != warmup for rollout in warmed):
            raise RuntimeError("nonterminating actor warmup ended")
        started = time.perf_counter_ns()
        remaining = steps
        while remaining:
            count = min(remaining, rollout_horizon)
            rollouts = pool.collect(lane_policies, count)
            for lane, rollout in enumerate(rollouts):
                if len(rollout.steps) != count:
                    raise RuntimeError("nonterminating actor workload ended")
                for step in rollout.steps:
                    digests[lane].update(step.payload)
                    event_counts[lane] += len(step.events)
            remaining -= count
        elapsed_ns = time.perf_counter_ns() - started
        hashes = pool.state_hash()
    return {
        "mode": "barrier_free_actor_rollouts",
        "rollout_horizon": rollout_horizon,
        "elapsed_seconds": elapsed_ns / 1e9,
        "operations_per_second": lanes * steps * 1e9 / elapsed_ns,
        "digests": [value.hexdigest() for value in digests],
        "state_hashes": list(hashes),
        "event_counts": event_counts,
    }


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": ordered[len(ordered) // 2],
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, type=Path)
    parser.add_argument("--lanes", default="8,16,32")
    parser.add_argument("--steps", type=int, default=3_000)
    parser.add_argument("--rollout-horizon", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.rollout_horizon <= 0
        or args.warmup <= 0
        or args.repeats <= 0
    ):
        parser.error("steps, rollout-horizon, warmup, and repeats must be positive")
    lane_counts = [int(value) for value in args.lanes.split(",")]
    if not lane_counts or any(value <= 0 for value in lane_counts):
        parser.error("lanes must be a comma-separated list of positive integers")

    simulator = ExactSimulator(args.worker, config=CONFIG)
    try:
        runtime = simulator.build_info()
    finally:
        simulator.close()

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "workload": "exact_nonterminating_random_policy_one_tick_v1",
        "config": CONFIG,
        "worker": str(args.worker.resolve()),
        "worker_sha256": sha256(args.worker),
        "benchmark_sha256": sha256(Path(__file__)),
        "source_sha256": {
            name: sha256(ROOT / name) for name in SOURCE_FILES
        },
        "runtime": runtime,
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "host": {
            "logical_cpus": os.cpu_count(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "steps_per_lane": args.steps,
        "rollout_horizon": args.rollout_horizon,
        "warmup_steps_per_lane": args.warmup,
        "repeats": args.repeats,
        "results": {},
    }
    for lanes in lane_counts:
        repeats: list[dict[str, Any]] = []
        for repeat in range(args.repeats):
            if repeat % 2:
                actor_result = actor(
                    args.worker,
                    lanes,
                    args.steps,
                    args.rollout_horizon,
                    args.warmup,
                )
                sync_result = synchronous(
                    args.worker, lanes, args.steps, args.warmup
                )
                order = "actor,sync"
            else:
                sync_result = synchronous(
                    args.worker, lanes, args.steps, args.warmup
                )
                actor_result = actor(
                    args.worker,
                    lanes,
                    args.steps,
                    args.rollout_horizon,
                    args.warmup,
                )
                order = "sync,actor"
            parity = all(
                actor_result[key] == sync_result[key]
                for key in ("digests", "state_hashes", "event_counts")
            )
            if not parity:
                raise RuntimeError(f"trajectory mismatch at {lanes} lanes")
            repeats.append(
                {
                    "order": order,
                    "sync": sync_result,
                    "actor": actor_result,
                    "speedup": (
                        actor_result["operations_per_second"]
                        / sync_result["operations_per_second"]
                    ),
                    "trajectory_parity": True,
                }
            )
        report["results"][str(lanes)] = {
            "repeats": repeats,
            "sync_operations_per_second": distribution(
                [value["sync"]["operations_per_second"] for value in repeats]
            ),
            "actor_operations_per_second": distribution(
                [value["actor"]["operations_per_second"] for value in repeats]
            ),
            "speedup": distribution([value["speedup"] for value in repeats]),
            "trajectory_parity": True,
        }

    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
        print(args.output)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
