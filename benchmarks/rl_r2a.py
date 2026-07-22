#!/usr/bin/env python3
"""Measure deterministic CPU inference for the R2a recurrent model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import date
from pathlib import Path

import torch

from irisu_rl.models import RecurrentActorCritic
from irisu_rl.schema import TEACHER_V1


ROOT = Path(__file__).resolve().parents[1]


def git_output(*args: str) -> str:
    result = subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def timed(callable_, iterations: int) -> float:
    started = time.perf_counter()
    for _ in range(iterations):
        callable_()
    return time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--step-batch", type=int, default=16)
    parser.add_argument("--step-iterations", type=int, default=500)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--sequence-batch", type=int, default=8)
    parser.add_argument("--sequence-iterations", type=int, default=50)
    args = parser.parse_args()
    if min(vars(args).values()) <= 0:
        parser.error("all benchmark dimensions must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260722)
    model = RecurrentActorCritic(TEACHER_V1).eval()

    def observations(time_steps: int, batch: int):
        global_features = torch.randn(
            time_steps, batch, len(TEACHER_V1.global_features)
        )
        bodies = torch.randn(
            time_steps,
            batch,
            TEACHER_V1.capacity,
            len(TEACHER_V1.body_features),
        )
        mask = torch.zeros(time_steps, batch, TEACHER_V1.capacity, dtype=torch.bool)
        mask[..., :64] = True
        return global_features, bodies, mask, model.initial_state(batch)

    step = observations(1, args.step_batch)
    sequence = observations(args.sequence_length, args.sequence_batch)
    with torch.inference_mode():
        for _ in range(20):
            model(*step)
        step_elapsed = timed(lambda: model(*step), args.step_iterations)
        for _ in range(5):
            model(*sequence)
        sequence_elapsed = timed(lambda: model(*sequence), args.sequence_iterations)
    step_decisions = args.step_batch * args.step_iterations
    sequence_decisions = (
        args.sequence_length * args.sequence_batch * args.sequence_iterations
    )
    result = {
        "schema": "rl-r2a-cpu-inference-v1",
        "recorded_at": date.today().isoformat(),
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": args.threads,
            "interop_threads": torch.get_num_interop_threads(),
            "cpu_model": cpu_model(),
            "affinity": sorted(os.sched_getaffinity(0)),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "source": {
            "commit": git_output("rev-parse", "HEAD"),
            "dirty": bool(git_output("status", "--porcelain")),
            "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
        },
        "identity": model.manifest(),
        "parameters": {**vars(args), "active_bodies": 64},
        "results": {
            "model_parameters": sum(value.numel() for value in model.parameters()),
            "step_elapsed_seconds": step_elapsed,
            "step_decisions_per_second": step_decisions / step_elapsed,
            "sequence_elapsed_seconds": sequence_elapsed,
            "sequence_decisions_per_second": sequence_decisions / sequence_elapsed,
        },
        "interpretation": "CPU model engineering throughput only; excludes environment collection and is not learning or transfer evidence",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
