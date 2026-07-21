#!/usr/bin/env python3
"""Profile the exact worker pipeline without changing its wire semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import struct
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import (  # noqa: E402
    Action,
    ExactSimulator,
    IrisuEnv,
    PaddedVectorEnv,
    RandomPolicy,
    ThreadVectorEnv,
)
from irisu_env.exact_ipc import (  # noqa: E402
    ExactTransition,
    ExactWorkerClient,
    _BODY,
    _OBSERVATION_HEADER,
    _STEP,
    _TRANSITION,
    _decode_transition,
    _worker_identity,
)


SOURCE_FILES = (
    "CMakeLists.txt",
    "benchmarks/exact_pipeline.py",
    "benchmarks/native_physics.cpp",
    "clone/core/config.cpp",
    "clone/core/config_io.cpp",
    "clone/core/dx_random.cpp",
    "clone/core/normal_rules.cpp",
    "clone/core/physics.cpp",
    "clone/core/simulator.cpp",
    "clone/include/irisu/config.hpp",
    "clone/include/irisu/config_io.hpp",
    "clone/include/irisu/dx_random.hpp",
    "clone/include/irisu/floating_point.hpp",
    "clone/include/irisu/math.hpp",
    "clone/include/irisu/normal_rules.hpp",
    "clone/include/irisu/physics.hpp",
    "clone/include/irisu/simulator.hpp",
    "clone/include/irisu/types.hpp",
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
    "reference/native-box2d/msvc-runtime.S",
    "reference/native-box2d/multiworld/box2d-wrapper-msvc.cpp",
    "reference/native-box2d/multiworld/msvc-bridge.c",
    "reference/native-box2d/multiworld/msvc-runtime.S",
    "tools/exact-physics-prototype/ipc_worker.cpp",
    "tools/exact-physics-prototype/physics_wrapper_forward.cpp",
    "tools/host-msvc9-box2d-multiworld.py",
    "tools/host-msvc9-box2d.py",
)


def positive(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    if SOURCE_FILES != tuple(sorted(SOURCE_FILES)):
        raise RuntimeError("benchmark source manifest paths are not sorted")
    missing = [name for name in SOURCE_FILES if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "benchmark source manifest is incomplete on disk: "
            + ", ".join(missing)
        )
    return {name: sha256_file(ROOT / name) for name in SOURCE_FILES}


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)] / 1_000.0


def summary(latencies: list[int], operations_per_sample: int = 1) -> dict[str, Any]:
    elapsed = sum(latencies)
    operations = len(latencies) * operations_per_sample
    return {
        "operations": operations,
        "elapsed_seconds": elapsed / 1_000_000_000.0,
        "operations_per_second": operations * 1_000_000_000.0 / elapsed,
        "sample_mean_us": statistics.fmean(latencies) / 1_000.0,
        "sample_p50_us": percentile(latencies, 0.50),
        "sample_p95_us": percentile(latencies, 0.95),
        "sample_p99_us": percentile(latencies, 0.99),
    }


def timed(function: Callable[[], Any], count: int) -> tuple[list[int], Any]:
    latencies: list[int] = []
    result: Any = None
    for _ in range(count):
        started = time.perf_counter_ns()
        result = function()
        latencies.append(time.perf_counter_ns() - started)
    return latencies, result


def raw_transition_metadata(payload: bytes) -> tuple[int, int, bool]:
    header = _OBSERVATION_HEADER.unpack_from(payload)
    body_count = int(header[15])
    offset = _OBSERVATION_HEADER.size + body_count * _BODY.size
    transition = _TRANSITION.unpack_from(payload, offset)
    return body_count, int(transition[1]), bool(transition[12] or transition[13])


def replace_raw_client(
    worker: Path, client: ExactWorkerClient, seed: int
) -> ExactWorkerClient:
    """Reset an episode in a fresh, identity-matched worker process."""

    expected_identity = _worker_identity(
        client.info, client.current_config_hash, client.executable_sha256
    )
    candidate = ExactWorkerClient(worker)
    try:
        candidate_identity = _worker_identity(
            candidate.info,
            candidate.current_config_hash,
            candidate.executable_sha256,
        )
        if candidate_identity != expected_identity:
            raise RuntimeError("replacement exact worker identity changed")
        candidate.reset(seed)
    except BaseException:
        candidate.close()
        raise
    client.close()
    return candidate


def prepare(
    worker: Path, client: ExactWorkerClient, seed: int, steps: int
) -> tuple[ExactWorkerClient, ExactTransition]:
    client.reset(seed)
    transition: ExactTransition | None = None
    for _ in range(steps):
        transition = client.step(wait_ticks=1)
        if transition.terminated or transition.truncated:
            client = replace_raw_client(worker, client, seed)
    assert transition is not None
    return client, transition


def worker_pipeline(
    worker: Path, iterations: int, decode_iterations: int, prepare_steps: int
) -> dict[str, Any]:
    client = ExactWorkerClient(worker)
    try:
        client, _ = prepare(worker, client, 41, prepare_steps)

        observe_raw, observation_payload = timed(
            lambda: client._request(4), iterations
        )

        def raw_step() -> bytes:
            client.send_step(wait_ticks=1)
            return client._finish_response(_STEP)

        step_raw: list[int] = []
        body_counts: list[int] = []
        event_counts: list[int] = []
        transition_payload = b""
        for _ in range(iterations):
            started = time.perf_counter_ns()
            transition_payload = raw_step()
            step_raw.append(time.perf_counter_ns() - started)
            body_count, event_count, done = raw_transition_metadata(transition_payload)
            body_counts.append(body_count)
            event_counts.append(event_count)
            if done:
                client = replace_raw_client(worker, client, 41)

        decode_times, transition = timed(
            lambda: _decode_transition(transition_payload), decode_iterations
        )
        observation_dict_times, _ = timed(
            transition.observation.to_dict, decode_iterations
        )
        info_dict_times, _ = timed(transition.to_step_dict, decode_iterations)
    finally:
        client.close()

    client = ExactWorkerClient(worker)
    try:
        client, _ = prepare(worker, client, 41, prepare_steps)
        decoded_step: list[int] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            transition = client.step(wait_ticks=1)
            decoded_step.append(time.perf_counter_ns() - started)
            if transition.terminated or transition.truncated:
                client = replace_raw_client(worker, client, 41)
    finally:
        client.close()

    return {
        "raw_observe_round_trip": summary(observe_raw),
        "raw_step_round_trip": summary(step_raw),
        "decoded_client_step": summary(decoded_step),
        "decode_captured_transition": summary(decode_times),
        "observation_dict_materialization": summary(observation_dict_times),
        "info_dict_materialization": summary(info_dict_times),
        "captured_observation_payload_bytes": len(observation_payload),
        "captured_transition_payload_bytes": len(transition_payload),
        "observed_body_count": {
            "min": min(body_counts),
            "mean": statistics.fmean(body_counts),
            "max": max(body_counts),
        },
        "observed_event_count": {
            "min": min(event_counts),
            "mean": statistics.fmean(event_counts),
            "max": max(event_counts),
        },
    }


def raw_random_vector_pipeline(
    worker: Path,
    iterations: int,
    lane_counts: list[int],
    *,
    padded: bool = False,
) -> dict[str, Any]:
    """Measure worker/pipe traffic without Python body decoding."""

    results: dict[str, Any] = {}
    for lanes in lane_counts:
        clients = [ExactWorkerClient(worker) for _ in range(lanes)]
        policies = [RandomPolicy(0x52574157 + index, max_wait_ticks=1)
                    for index in range(lanes)]
        episodes = [0] * lanes
        action_counts: Counter[str] = Counter()
        body_counts: list[int] = []
        event_counts: list[int] = []
        payload_sizes: list[int] = []
        latencies: list[int] = []
        try:
            for index, client in enumerate(clients):
                client.reset(41 + index)
            for _ in range(iterations):
                actions = [policy.act({}) for policy in policies]
                action_counts.update(str(int(action.kind)) for action in actions)
                started = time.perf_counter_ns()
                for client, action in zip(clients, actions):
                    sender = client.send_step_padded if padded else client.send_step
                    sender(
                        int(action.kind), action.cursor_x, action.cursor_y,
                        action.wait_ticks
                    )
                payloads = [
                    (
                        client.receive_step_padded_payload()
                        if padded
                        else client._finish_response(_STEP)
                    )
                    for client in clients
                ]
                latencies.append(time.perf_counter_ns() - started)
                for index, payload in enumerate(payloads):
                    body_count, event_count, done = raw_transition_metadata(payload)
                    body_counts.append(body_count)
                    event_counts.append(event_count)
                    payload_sizes.append(len(payload))
                    if done:
                        episodes[index] += 1
                        clients[index] = replace_raw_client(
                            worker,
                            clients[index],
                            41 + index + episodes[index] * lanes,
                        )
                        policies[index].reset(
                            0x52574157 + index + episodes[index] * lanes
                        )
        finally:
            for client in clients:
                client.close()
        result = summary(latencies, lanes)
        result.update(
            {
                "lanes": lanes,
                "wire_mode": "live bodies without events" if padded else "live bodies plus events",
                "action_counts": dict(sorted(action_counts.items())),
                "episode_resets": sum(episodes),
                "observed_body_count": distribution(body_counts),
                "observed_event_count": distribution(event_counts),
                "response_content_bytes": distribution(payload_sizes),
            }
        )
        results[str(lanes)] = result
    return results


def distribution(values: list[int]) -> dict[str, float | int]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def distributions_equivalent(
    left: dict[str, float | int], right: dict[str, float | int]
) -> bool:
    """Compare exact extrema and tolerate only JSON mean rounding."""

    return (
        left["min"] == right["min"]
        and left["max"] == right["max"]
        and math.isclose(
            float(left["mean"]),
            float(right["mean"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )


def env_pipeline(worker: Path, iterations: int, prepare_steps: int) -> dict[str, Any]:
    action = Action.wait(1)
    with IrisuEnv(physics_backend="exact", worker_path=worker) as env:
        observation, _ = env.reset(seed=41)
        for _ in range(prepare_steps):
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                observation, _ = env.reset(seed=41)
        body_counts: list[int] = []
        latencies: list[int] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            observation, _, terminated, truncated, _ = env.step(action)
            latencies.append(time.perf_counter_ns() - started)
            body_counts.append(len(observation["bodies"]))
            if terminated or truncated:
                observation, _ = env.reset(seed=41)
    result = summary(latencies)
    result["observed_body_count"] = {
        "min": min(body_counts),
        "mean": statistics.fmean(body_counts),
        "max": max(body_counts),
    }
    return result


def snapshot_pipeline(
    worker: Path,
    history_steps: int,
    branch_iterations: int,
    restore_iterations: int,
) -> dict[str, Any]:
    """Compare local fork/COW branching with durable action-log restore."""

    with ExactSimulator(worker) as simulator:
        simulator.reset(41)
        for _ in range(history_steps):
            simulator.step(0, 0.0, 0.0, 1)
        durable = simulator.clone_state()
        expected_hash = simulator.state_hash()

        started = time.perf_counter_ns()
        checkpoint = simulator.fast_checkpoint()
        checkpoint_latency = time.perf_counter_ns() - started
        branch_latencies: list[int] = []
        try:
            for _ in range(branch_iterations):
                started = time.perf_counter_ns()
                branch = checkpoint.branch()
                branch_latencies.append(time.perf_counter_ns() - started)
                try:
                    if branch.state_hash() != expected_hash:
                        raise RuntimeError("fast branch state hash changed")
                    if branch.clone_state() != durable:
                        raise RuntimeError("fast branch durable snapshot changed")
                finally:
                    branch.close()
        finally:
            checkpoint.close()

        restore_latencies: list[int] = []
        for _ in range(restore_iterations):
            started = time.perf_counter_ns()
            simulator.restore_state(durable)
            restore_latencies.append(time.perf_counter_ns() - started)
            if simulator.state_hash() != expected_hash:
                raise RuntimeError("durable restore state hash changed")

    branch_result = summary(branch_latencies)
    restore_result = summary(restore_latencies)
    return {
        "history_actions": history_steps,
        "durable_snapshot_bytes": len(durable),
        "fast_checkpoint_us": checkpoint_latency / 1_000.0,
        "fast_branch": branch_result,
        "durable_restore": restore_result,
        "median_restore_over_branch": (
            restore_result["sample_p50_us"] / branch_result["sample_p50_us"]
        ),
        "validation": "state hash and durable snapshot equality per branch",
    }


def vector_pipeline(
    worker: Path,
    iterations: int,
    lane_counts: list[int],
    *,
    random_policy: bool = False,
) -> dict[str, Any]:
    action = Action.wait(1)
    results: dict[str, Any] = {}
    for lanes in lane_counts:
        with ThreadVectorEnv(
            lanes,
            physics_backend="exact",
            worker_path=worker,
            workers=lanes,
        ) as vector:
            observations, _ = vector.reset(seed=41)
            policies = [
                RandomPolicy(0x52574157 + index, max_wait_ticks=1)
                for index in range(lanes)
            ]
            episodes = [0] * lanes
            actions = [action] * lanes
            warmup_rounds = 0 if random_policy else min(100, iterations)
            for _ in range(warmup_rounds):
                if random_policy:
                    actions = [
                        policy.act(observation)
                        for policy, observation in zip(policies, observations)
                    ]
                observations, _, terminated, truncated, _ = vector.step(actions)
                for index, done in enumerate(
                    a or b for a, b in zip(terminated, truncated)
                ):
                    if done:
                        observations[index], _ = vector.envs[index].reset(seed=41 + index)
            latencies: list[int] = []
            body_counts: list[int] = []
            event_counts: list[int] = []
            action_counts: Counter[str] = Counter()
            for _ in range(iterations):
                started = time.perf_counter_ns()
                if random_policy:
                    actions = [
                        policy.act(observation)
                        for policy, observation in zip(policies, observations)
                    ]
                action_counts.update(str(int(value.kind)) for value in actions)
                observations, _, terminated, truncated, infos = vector.step(actions)
                latencies.append(time.perf_counter_ns() - started)
                body_counts.extend(len(value["bodies"]) for value in observations)
                event_counts.extend(len(info["events"]) for info in infos)
                for index, done in enumerate(
                    a or b for a, b in zip(terminated, truncated)
                ):
                    if done:
                        episodes[index] += 1
                        observations[index], _ = vector.envs[index].reset(
                            seed=41 + index + episodes[index] * lanes
                        )
                        policies[index].reset(
                            0x52574157 + index + episodes[index] * lanes
                        )
        result = summary(latencies, lanes)
        result["lanes"] = lanes
        result["vector_round_p50_us"] = result.pop("sample_p50_us")
        result["vector_round_p95_us"] = result.pop("sample_p95_us")
        result["vector_round_p99_us"] = result.pop("sample_p99_us")
        result["vector_round_mean_us"] = result.pop("sample_mean_us")
        result["observed_body_count"] = {
            "min": min(body_counts),
            "mean": statistics.fmean(body_counts),
            "max": max(body_counts),
        }
        result["observed_event_count"] = distribution(event_counts)
        result["action_counts"] = dict(sorted(action_counts.items()))
        result["episode_resets"] = sum(episodes)
        result["policy"] = "RandomPolicy(max_wait_ticks=1)" if random_policy else "wait(1)"
        results[str(lanes)] = result
    return results


def padded_lazy_pipeline(
    worker: Path, iterations: int, lane_counts: list[int]
) -> dict[str, Any]:
    """Measure fixed padded responses while exposing only lazy event counts."""

    results: dict[str, Any] = {}
    for lanes in lane_counts:
        with PaddedVectorEnv(
            lanes,
            physics_backend="exact",
            worker_path=worker,
            workers=lanes,
        ) as vector:
            observations, _ = vector.reset(seed=41)
            policies = [
                RandomPolicy(0x52574157 + index, max_wait_ticks=1)
                for index in range(lanes)
            ]
            episodes = [0] * lanes
            body_counts: list[int] = []
            event_counts: list[int] = []
            action_counts: Counter[str] = Counter()
            latencies: list[int] = []
            response_sizes: list[int] = []
            response_size_mismatches = 0
            for _ in range(iterations):
                started = time.perf_counter_ns()
                actions = [
                    policy.act(observation)
                    for policy, observation in zip(policies, observations)
                ]
                action_counts.update(str(int(value.kind)) for value in actions)
                observations, _, terminated, truncated, infos = vector.step(actions)
                # __len__ reads the fixed transition header; it does not issue
                # FetchEvents. Event detail remains available until lane advance.
                event_counts.extend(len(info["events"]) for info in infos)
                latencies.append(time.perf_counter_ns() - started)
                current_body_counts = [int(value.body_count) for value in observations]
                body_counts.extend(current_body_counts)
                for env, body_count in zip(vector.envs, current_body_counts):
                    client = env._client
                    assert client is not None
                    response_size = client.last_response_bytes
                    response_sizes.append(response_size)
                    # Frame header + status + observation header + live bodies
                    # + transition header + event generation.
                    if response_size != 224 + 100 * body_count:
                        response_size_mismatches += 1
                for index, done in enumerate(
                    a or b for a, b in zip(terminated, truncated)
                ):
                    if done:
                        episodes[index] += 1
                        observations[index] = vector.reset_at(
                            index,
                            seed=41 + index + episodes[index] * lanes,
                        )
                        policies[index].reset(
                            0x52574157 + index + episodes[index] * lanes
                        )
        result = summary(latencies, lanes)
        result.update(
            {
                "lanes": lanes,
                "policy": "RandomPolicy(max_wait_ticks=1)",
                "event_access": "lazy count only; no FetchEvents request",
                "action_counts": dict(sorted(action_counts.items())),
                "episode_resets": sum(episodes),
                "observed_body_count": distribution(body_counts),
                "observed_event_count": distribution(event_counts),
                "worker_response_frame_bytes": distribution(response_sizes),
                "worker_response_size_formula": "224 + 100 * body_count",
                "worker_response_size_mismatches": response_size_mismatches,
            }
        )
        results[str(lanes)] = result
    return results


def physics_result(executable: Path | None, ticks: int) -> dict[str, Any] | None:
    if executable is None:
        return None
    completed = subprocess.run(
        [
            str(executable.resolve()),
            "--ticks",
            str(ticks),
            "--warmup",
            "500",
            "--bodies",
            "48",
            "--seed",
            "20260720",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def simulator_result(
    executable: Path | None, decisions: int
) -> dict[str, Any] | None:
    if executable is None:
        return None
    completed = subprocess.run(
        [
            str(executable.resolve()),
            "--simulator-decisions",
            str(decisions),
            "--seed",
            "41",
            "--policy-seed",
            str(0x52574157),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--iterations", type=positive, default=2_000)
    parser.add_argument("--decode-iterations", type=positive, default=10_000)
    parser.add_argument("--prepare-steps", type=positive, default=500)
    parser.add_argument("--vector-lanes", default="1,4,8")
    parser.add_argument("--physics-benchmark", type=Path)
    parser.add_argument("--physics-ticks", type=positive, default=10_000)
    parser.add_argument("--snapshot-history-steps", type=positive, default=1_000)
    parser.add_argument("--snapshot-branches", type=positive, default=10)
    parser.add_argument("--snapshot-restores", type=positive, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        args.vector_lanes = [positive(value) for value in args.vector_lanes.split(",")]
    except (TypeError, ValueError, argparse.ArgumentTypeError) as exc:
        parser.error(f"invalid --vector-lanes: {exc}")
    if max(args.vector_lanes) > 64:
        parser.error("--vector-lanes values must be at most 64")
    if args.snapshot_history_steps > 10_000:
        parser.error("--snapshot-history-steps must be at most 10000")
    if args.snapshot_branches > 100:
        parser.error("--snapshot-branches must be at most 100")
    if args.snapshot_restores > 20:
        parser.error("--snapshot-restores must be at most 20")
    return args


def main() -> None:
    args = parse_args()
    worker = args.worker.resolve()
    if not worker.is_file():
        raise FileNotFoundError(worker)
    started = time.perf_counter_ns()
    with ExactWorkerClient(worker) as client:
        worker_info = {
            name: getattr(client.info, name)
            for name in client.info.__dataclass_fields__
        }
    raw_random = raw_random_vector_pipeline(
        worker, args.iterations, args.vector_lanes
    )
    raw_padded = raw_random_vector_pipeline(
        worker, args.iterations, args.vector_lanes, padded=True
    )
    decoded_random = vector_pipeline(
        worker,
        args.iterations,
        args.vector_lanes,
        random_policy=True,
    )
    padded_random = padded_lazy_pipeline(
        worker, args.iterations, args.vector_lanes
    )
    native_simulator = simulator_result(args.physics_benchmark, args.iterations)
    representative_equivalence = {
        lane: {
            field: raw_random[lane][field] == decoded_random[lane][field]
            for field in (
                "action_counts",
                "episode_resets",
                "observed_body_count",
                "observed_event_count",
            )
        }
        for lane in raw_random
    }
    padded_equivalence = {
        lane: {
            field: raw_random[lane][field] == padded_random[lane][field]
            for field in (
                "action_counts",
                "episode_resets",
                "observed_body_count",
                "observed_event_count",
            )
        }
        for lane in raw_random
    }
    raw_padded_equivalence = {
        lane: {
            field: raw_random[lane][field] == raw_padded[lane][field]
            for field in (
                "action_counts",
                "episode_resets",
                "observed_body_count",
                "observed_event_count",
            )
        }
        for lane in raw_random
    }
    native_equivalence: dict[str, bool] | None = None
    if native_simulator is not None and "1" in raw_random:
        lane = raw_random["1"]
        native_equivalence = {
            "action_counts": native_simulator["action_counts"]
            == [int(lane["action_counts"].get(str(index), 0)) for index in range(4)],
            "episode_resets": native_simulator["episode_resets"]
            == lane["episode_resets"],
            "observed_body_count": distributions_equivalent(
                {
                    "min": native_simulator["observed_body_count_min"],
                    "mean": native_simulator["observed_body_count_mean"],
                    "max": native_simulator["observed_body_count_max"],
                },
                lane["observed_body_count"],
            ),
            "observed_event_count": distributions_equivalent(
                {
                    "min": native_simulator["event_count_min"],
                    "mean": native_simulator["event_count_mean"],
                    "max": native_simulator["event_count_max"],
                },
                lane["observed_event_count"],
            ),
        }
    maximum_lane_count = max(args.vector_lanes)
    dense_padded_rate = padded_random[str(maximum_lane_count)][
        "operations_per_second"
    ]
    dense_gate_target = 20_000.0
    report = {
        "schema_version": 2,
        "workload": "exact_worker_pipeline_one_tick_v1",
        "semantics": (
            "exact eager transitions plus packed transitions with lazy event detail"
        ),
        "raw_episode_reset": {
            "strategy": "fresh identity-matched worker process",
            "included_in_step_latency": False,
        },
        "parameters": {
            "iterations": args.iterations,
            "decode_iterations": args.decode_iterations,
            "prepare_steps": args.prepare_steps,
            "physics_ticks": args.physics_ticks,
            "vector_lanes": args.vector_lanes,
            "snapshot_history_steps": args.snapshot_history_steps,
            "snapshot_branches": args.snapshot_branches,
            "snapshot_restores": args.snapshot_restores,
        },
        "host": {
            "logical_cpus": os.cpu_count(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "artifacts": {
            "worker": str(worker),
            "worker_sha256": sha256_file(worker),
            "worker_info": worker_info,
            "sources": source_manifest(),
        },
        "results": {
            "native_physics_48_body": physics_result(
                args.physics_benchmark, args.physics_ticks
            ),
            "native_simulator_random_policy": native_simulator,
            "worker_pipeline": worker_pipeline(
                worker,
                args.iterations,
                args.decode_iterations,
                args.prepare_steps,
            ),
            "snapshot_branching": snapshot_pipeline(
                worker,
                args.snapshot_history_steps,
                args.snapshot_branches,
                args.snapshot_restores,
            ),
            "raw_random_policy_vector": raw_random,
            "raw_padded_random_policy_vector": raw_padded,
            "irisu_env_step": env_pipeline(
                worker, args.iterations, args.prepare_steps
            ),
            "thread_vector_step": vector_pipeline(
                worker, args.iterations, args.vector_lanes
            ),
            "thread_vector_random_policy": decoded_random,
            "padded_lazy_random_policy": padded_random,
            "representative_workload_equivalence": representative_equivalence,
            "padded_workload_equivalence": padded_equivalence,
            "raw_padded_workload_equivalence": raw_padded_equivalence,
            "native_simulator_workload_equivalence": native_equivalence,
        },
        "gates": {
            "dense_padded_lazy_max_lanes": {
                "metric": (
                    "results.padded_lazy_random_policy."
                    f"{maximum_lane_count}.operations_per_second"
                ),
                "lanes": maximum_lane_count,
                "unit": "decisions_per_second",
                "actual": dense_padded_rate,
                "target": dense_gate_target,
                "passed": dense_padded_rate >= dense_gate_target,
            }
        },
    }
    report["elapsed_seconds"] = (
        time.perf_counter_ns() - started
    ) / 1_000_000_000.0
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(output, end="")
    else:
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
