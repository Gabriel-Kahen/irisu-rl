#!/usr/bin/env python3
"""Bounded R3a production-collector throughput and numerical-health smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tomllib
from pathlib import Path

import torch

from irisu_env import PaddedVectorEnv
from irisu_rl.collector import (
    CollectorConfig,
    R3ATrainingSession,
    RecurrentCollector,
    ScoreTaskContract,
)
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.ppo import PPOConfig, PPOTrainer
from irisu_rl.vector_adapter import MacroVectorAdapter


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/rl/experiments/r3a-multistep-v1.toml"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--backend", choices=("portable", "exact"), required=True)
    result.add_argument("--runtime", type=Path, required=True)
    result.add_argument("--lanes", type=int)
    result.add_argument("--updates", type=int)
    result.add_argument("--decisions", type=int)
    result.add_argument("--target-ticks", type=int)
    result.add_argument("--no-tick-target", action="store_true")
    result.add_argument("--torch-threads", type=int)
    result.add_argument("--seed", type=int)
    result.add_argument("--output", type=Path)
    return result


def main() -> int:
    args = parser().parse_args()
    config_path = args.config.resolve(strict=True)
    config_bytes = config_path.read_bytes()
    checked = tomllib.loads(config_bytes.decode())
    smoke = checked["smoke_run"]
    collector_config = checked["collector"]
    ppo_config = checked["ppo"]
    lanes = smoke["lanes"] if args.lanes is None else args.lanes
    updates = smoke["updates"] if args.updates is None else args.updates
    decisions = (
        collector_config["max_decisions"] if args.decisions is None else args.decisions
    )
    target_ticks = (
        None
        if args.no_tick_target
        else collector_config["target_simulated_ticks"]
        if args.target_ticks is None
        else args.target_ticks
    )
    torch_threads = (
        smoke["torch_threads"] if args.torch_threads is None else args.torch_threads
    )
    seed = smoke["seed"] if args.seed is None else args.seed
    if lanes <= 0 or updates <= 0 or decisions <= 0 or torch_threads <= 0:
        raise SystemExit("lanes, updates, and decisions must be positive")
    if updates > ppo_config["total_updates"]:
        raise SystemExit("smoke updates exceed the checked PPO update budget")
    runtime = args.runtime.resolve(strict=True)
    torch.set_num_threads(torch_threads)
    torch.manual_seed(seed)
    model = RecurrentActorCritic(
        TeacherStateEncoder().schema,
        config=RecurrentModelConfig(32, 32, 64, 64, 1),
    )
    vector_kwargs = (
        {"physics_backend": "exact", "worker_path": runtime}
        if args.backend == "exact"
        else {"physics_backend": "portable", "library_path": runtime}
    )
    started = time.perf_counter()
    reports = []
    with PaddedVectorEnv(lanes, **vector_kwargs) as vector:
        adapter = MacroVectorAdapter(vector, encoder=TeacherStateEncoder())
        task = ScoreTaskContract(lanes, reward_scale=checked["reward"]["scale"])
        collector = RecurrentCollector(
            model,
            adapter,
            task,
            config=CollectorConfig(
                max_decisions=decisions,
                target_simulated_ticks=target_ticks,
                gamma_tick=collector_config["gamma_tick"],
                lambda_tick=collector_config["lambda_tick"],
            ),
            policy_sampler_seed=seed ^ 0xA5A5,
        )
        trainer = PPOTrainer(
            model,
            config=PPOConfig(
                learning_rate=ppo_config["learning_rate"],
                final_learning_rate_fraction=ppo_config["final_learning_rate_fraction"],
                epochs=ppo_config["epochs"],
                lane_minibatch_size=min(ppo_config["lane_minibatch_size"], lanes),
                clip_ratio=ppo_config["clip_ratio"],
                value_clip=ppo_config["value_clip"],
                value_coefficient=ppo_config["value_coefficient"],
                entropy_coefficient=ppo_config["entropy_coefficient"],
                max_gradient_norm=ppo_config["max_gradient_norm"],
                target_kl=ppo_config["target_kl"],
            ),
            total_updates=ppo_config["total_updates"],
            sampler_seed=seed ^ 0x5A5A,
        )
        session = R3ATrainingSession(
            collector,
            trainer,
            numpy_seed=seed ^ 17,
            max_consecutive_skips=smoke["max_consecutive_skips"],
        )
        session.initialize()
        target_completed_updates = trainer.schedule.completed_updates + updates
        max_attempts = updates * (session.max_consecutive_skips + 1)
        while trainer.schedule.completed_updates < target_completed_updates:
            if len(reports) >= max_attempts:
                raise RuntimeError("smoke exceeded its bounded rollout-attempt budget")
            before = time.perf_counter()
            result = session.run_update()
            report = {
                "attempt": len(reports) + 1,
                "completed_update": trainer.schedule.completed_updates,
                "seconds": time.perf_counter() - before,
                "decision_rows": result.collection.decision_rows,
                "transitions": result.collection.transitions,
                "simulated_ticks": result.collection.simulated_ticks,
                "tick_target_overshoot": result.collection.tick_target_overshoot,
                "raw_reward": result.collection.raw_reward,
                "invalid_actions": result.collection.invalid_actions,
                "skipped_reason": result.skipped_reason,
            }
            if result.optimizer is not None:
                report.update(
                    approximate_kl=result.optimizer.approximate_kl,
                    gradient_norm=result.optimizer.gradient_norm,
                    learning_rate=result.optimizer.learning_rate,
                )
            reports.append(report)
    elapsed = time.perf_counter() - started
    transitions = sum(report["transitions"] for report in reports)
    ticks = sum(report["simulated_ticks"] for report in reports)
    output = {
        "version": "rl-r3a-throughput-v1",
        "deployable": False,
        "observation_provenance": "privileged_simulator",
        "backend": args.backend,
        "runtime": str(runtime),
        "runtime_sha256": hashlib.sha256(runtime.read_bytes()).hexdigest(),
        "config": str(config_path),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "lanes": lanes,
        "requested_optimizer_updates": updates,
        "completed_optimizer_updates": trainer.schedule.completed_updates,
        "attempted_rollouts": len(reports),
        "skipped_rollouts": session.skipped_rollouts,
        "max_decisions": decisions,
        "target_simulated_ticks": target_ticks,
        "torch_threads": torch_threads,
        "seed": seed,
        "wall_seconds": elapsed,
        "transitions_per_second": transitions / elapsed,
        "simulated_ticks_per_second": ticks / elapsed,
        "zero_invalid_actions": all(
            report["invalid_actions"] == 0 for report in reports
        ),
        "zero_skipped_updates": all(
            report["skipped_reason"] is None for report in reports
        ),
        "finite_optimizer_metrics": all(
            report["skipped_reason"] is None
            and all(
                torch.isfinite(torch.tensor(report[name]))
                for name in ("approximate_kl", "gradient_norm", "learning_rate")
            )
            for report in reports
        ),
        "reports": reports,
    }
    payload = json.dumps(output, sort_keys=True, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
