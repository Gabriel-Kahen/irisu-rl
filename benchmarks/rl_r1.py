#!/usr/bin/env python3
"""Measure the complete R1 collection path, including encoding and storage."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import date
from pathlib import Path

import numpy as np

from irisu_env import PaddedVectorEnv
from irisu_rl import (
    ACCEPTED_EXACT_RUNTIME_2026_07_21,
    ACTOR_VISION_V1,
    MacroVectorAdapter,
    RolloutBuffer,
    SeedAllocator,
    SemanticAction,
    TeacherStateEncoder,
    TEACHER_V1,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lanes", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--library")
    parser.add_argument("--backend", choices=("portable", "exact"), default="portable")
    parser.add_argument("--worker")
    args = parser.parse_args()
    identity = None
    if args.backend == "exact":
        if not args.worker:
            parser.error("--backend exact requires an absolute --worker path")
        identity = ACCEPTED_EXACT_RUNTIME_2026_07_21.attest(args.worker)
    rng = np.random.default_rng(20260722)
    with PaddedVectorEnv(
        args.lanes,
        library_path=args.library if args.backend == "portable" else None,
        physics_backend=args.backend,
        worker_path=args.worker if args.backend == "exact" else None,
        config={"max_episode_ticks": 100_000},
    ) as vector:
        adapter = MacroVectorAdapter(
            vector,
            encoder=TeacherStateEncoder(),
            seed_allocator=SeedAllocator(key=20260722),
        )
        initial = adapter.reset()
        buffer = RolloutBuffer(args.lanes * args.iterations, initial.schema)
        invalid = 0
        config_hashes: set[int] = set()
        started = time.perf_counter()
        for _ in range(args.iterations):
            actions = []
            for draw in rng.integers(0, 10, size=args.lanes):
                if draw < 7:
                    actions.append(SemanticAction.wait((1, 2, 4, 8, 16, 32)[draw % 6]))
                elif draw < 9:
                    actions.append(SemanticAction.weak(float(rng.random()), float(rng.random())))
                else:
                    actions.append(SemanticAction.strong(float(rng.random()), float(rng.random())))
            transitions = adapter.step(actions)
            for transition in transitions:
                buffer.append(transition)
                invalid += int(transition.diagnostics.invalid_action)
                config_hashes.add(transition.diagnostics.config_hash)
        buffer.seal(adapter.current_observation)
        elapsed = time.perf_counter() - started
    decisions = args.lanes * args.iterations
    result = {
        "schema": "rl-r1-end-to-end-benchmark-v1",
        "recorded_at": date.today().isoformat(),
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "parameters": {"backend": args.backend, "lanes": args.lanes, "iterations": args.iterations, "seed": 20260722},
        "results": {
            "semantic_decisions": decisions,
            "elapsed_seconds": elapsed,
            "semantic_decisions_per_second": decisions / elapsed,
            "invalid_actions": invalid,
            "stored_transitions": buffer.size,
            "unique_initial_and_autoreset_seeds": adapter.seed_allocator.cursor,
        },
        "identity": {
            "worker_sha256": (
                ACCEPTED_EXACT_RUNTIME_2026_07_21.worker_sha256
                if identity is not None else None
            ),
            "exact_library_sha256": (
                ACCEPTED_EXACT_RUNTIME_2026_07_21.exact_library_sha256
                if identity is not None else None
            ),
            "actor_schema_sha256": ACTOR_VISION_V1.sha256,
            "teacher_schema_sha256": TEACHER_V1.sha256,
            "config_hashes": sorted(config_hashes),
            "uv_lock_sha256": hashlib.sha256(
                (Path(__file__).resolve().parents[1] / "uv.lock").read_bytes()
            ).hexdigest(),
        },
        "scope": f"{args.backend} padded vector + semantic macros + vectorized teacher encoding + owned rollout writes",
        "interpretation": "R1 engineering throughput only; not evidence of policy quality or sim-to-game fidelity",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
