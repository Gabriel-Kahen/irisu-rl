#!/usr/bin/env python3
"""Measure real exact-backend recurrent evaluation throughput."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import resource
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter, process_time

import torch
from irisu_env import IrisuEnv, PaddedVectorEnv
from irisu_rl.collector import model_state_sha256
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.r3b_evaluation import (
    EvaluationSuite,
    evaluate_recurrent_policy_vectorized,
)
from irisu_rl.r3b_snapshots import load_snapshot_bundle
from irisu_rl.schema import TEACHER_V1


def _sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--lanes", type=int, required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--max-ticks", type=int, default=8192)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument(
        "--model-state",
        type=Path,
        help="trusted canonical state.pt whose learned model should be replayed",
    )
    args = parser.parse_args()
    if (
        args.lanes <= 0
        or args.workers <= 0
        or args.workers > args.lanes
        or args.cells <= 0
        or args.max_ticks <= 0
        or args.torch_threads <= 0
        or args.replicas <= 0
    ):
        parser.error("numeric arguments are outside the supported range")

    worker = args.worker.resolve(strict=True)
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(20260724)
    with IrisuEnv(physics_backend="exact", worker_path=worker) as loader:
        bundle = load_snapshot_bundle(args.bundle, loader)
    recipes = tuple(
        recipe for recipe in bundle.library.recipes if recipe.split == "calibration"
    )[: args.cells]
    if len(recipes) != args.cells:
        parser.error("bundle cannot supply the requested calibration cells")

    assignment_sha256 = _sha256({"purpose": "r3b-evaluation-throughput-v1"})
    suite = EvaluationSuite(
        suite_id="r3b-evaluation-throughput-v1",
        split="calibration",
        snapshot_ids=tuple(recipe.snapshot_id for recipe in recipes),
        repetitions=1,
        policy_seed=51,
        max_decisions=args.max_ticks,
        max_simulated_ticks=args.max_ticks,
        runtime_identity_sha256=bundle.runtime_identity_sha256,
        assignment_sha256=assignment_sha256,
        library_sha256=bundle.library.sha256,
        snapshot_store_sha256=bundle.store.sha256,
        action_spec_sha256=bundle.source.action_spec_sha256,
        recipe_sha256s=tuple(recipe.sha256 for recipe in recipes),
        backend="exact",
    )
    model = RecurrentActorCritic(
        TEACHER_V1,
        config=RecurrentModelConfig(
            96,
            96,
            192,
            192,
            1,
            critic_condition_features=1,
        ),
    )
    if args.model_state is None:
        policy_source = "fixed-wait"
        kind_mask = torch.zeros((1, 3), dtype=torch.bool)
        kind_mask[:, 0] = True
        wait_mask = torch.zeros(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
        wait_mask[:, -1] = True
    else:
        model_state_path = args.model_state.resolve(strict=True)
        checkpoint = torch.load(
            model_state_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("model"), dict
        ):
            parser.error("model state is not a canonical checkpoint mapping")
        model.load_state_dict(checkpoint["model"], strict=True)
        policy_source = str(model_state_path)
        kind_mask = torch.ones((1, 3), dtype=torch.bool)
        wait_mask = torch.ones(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
    evaluator_sha256 = _sha256({"evaluator": "throughput-v1"})

    def evaluate_replica(replica: int):
        local_model = copy.deepcopy(model)
        execution_sha256 = _sha256(
            {
                "execution": "throughput-v1",
                "replica": replica,
                "lanes": args.lanes,
                "workers": args.workers,
            }
        )
        with PaddedVectorEnv(
            args.lanes,
            workers=args.workers,
            physics_backend="exact",
            worker_path=worker,
        ) as simulator:
            started = perf_counter()
            report = evaluate_recurrent_policy_vectorized(
                simulator,
                bundle.store,
                suite,
                local_model,
                TeacherStateEncoder(),
                kind_mask,
                wait_mask,
                evaluator_sha256=evaluator_sha256,
                expected_assignment_sha256=assignment_sha256,
                execution_identity_sha256=execution_sha256,
            )
            return perf_counter() - started, report

    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu_before = child_usage.ru_utime + child_usage.ru_stime
    process_cpu_before = process_time()
    started = perf_counter()
    if args.replicas == 1:
        results = (evaluate_replica(0),)
    else:
        with ThreadPoolExecutor(
            max_workers=args.replicas,
            thread_name_prefix="r3b-benchmark",
        ) as executor:
            results = tuple(executor.map(evaluate_replica, range(args.replicas)))
    elapsed = perf_counter() - started
    process_cpu = process_time() - process_cpu_before
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    child_cpu = child_usage.ru_utime + child_usage.ru_stime - child_cpu_before

    simulated_ticks = sum(
        episode.elapsed_ticks for _, report in results for episode in report.episodes
    )
    content_hashes = {report.episode_content_sha256 for _, report in results}
    if len(content_hashes) != 1:
        raise RuntimeError("replica episode content differs")
    print(
        json.dumps(
            {
                "version": "r3b-evaluation-throughput-result-v1",
                "evaluation_module": str(
                    Path(
                        inspect.getfile(evaluate_recurrent_policy_vectorized)
                    ).resolve()
                ),
                "cells_per_replica": len(results[0][1].episodes),
                "replicas": args.replicas,
                "lanes": args.lanes,
                "workers": args.workers,
                "max_ticks": args.max_ticks,
                "policy_source": policy_source,
                "model_state_sha256": model_state_sha256(model),
                "wall_seconds": elapsed,
                "replica_wall_seconds": [value for value, _ in results],
                "process_cpu_seconds": process_cpu,
                "child_cpu_seconds": child_cpu,
                "average_cpu_cores": (process_cpu + child_cpu) / elapsed,
                "simulated_ticks": simulated_ticks,
                "simulated_ticks_per_second": simulated_ticks / elapsed,
                "episode_content_sha256": content_hashes.pop(),
            },
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
