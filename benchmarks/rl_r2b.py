#!/usr/bin/env python3
"""Run the preregistered R2b behavioral-cloning and one-body PPO proof."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import tomllib
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.checkpoints import save_checkpoint
from irisu_rl.one_body import (
    OneBodySpec,
    OneBodyTask,
    concatenate_encoded,
    expert_actions,
    one_body_training_batch,
    policy_distribution,
)
from irisu_rl.ppo import PPOConfig, PPOTrainer
from irisu_rl.schema import TEACHER_V1
from irisu_rl.runtime_identity import ACCEPTED_EXACT_RUNTIME_2026_07_21
from irisu_rl.seeds import SeedAllocator
from irisu_rl.torch_distribution import ActionTensor


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = ROOT / "configs/rl/experiments/r2b-one-body-v1.toml"
EXPERIMENT_BYTES = EXPERIMENT_PATH.read_bytes()
EXPERIMENT = tomllib.loads(EXPERIMENT_BYTES.decode())
MODEL_CONFIG = RecurrentModelConfig(**EXPERIMENT["model"])
MODEL_SEEDS = tuple(EXPERIMENT["ppo"]["model_seeds"])
LEARNING_RATES = tuple(EXPERIMENT["ppo"]["candidate_learning_rates"])
VALIDATION_ALLOCATOR_KEY = EXPERIMENT["seeds"]["validation_allocator_key"]
FINAL_TEST_ALLOCATOR_KEY = EXPERIMENT["seeds"]["final_test_allocator_key"]
CALIBRATION_ALLOCATOR_KEY = EXPERIMENT["seeds"]["calibration_allocator_key"]
RANDOM_VALIDATION_SEED = EXPERIMENT["seeds"]["random_validation"]
RANDOM_TEST_SEED = EXPERIMENT["seeds"]["random_test"]
PPO_SAMPLER_XOR = EXPERIMENT["seeds"]["ppo_sampler_xor"]
BC_SEED = EXPERIMENT["behavioral_cloning"]["seed"]
CANONICAL_BUDGETS = {
    "updates": EXPERIMENT["ppo"]["updates"],
    "lanes": EXPERIMENT["ppo"]["lanes_per_height"],
    "evaluation_rounds": EXPERIMENT["evaluation"]["rounds"],
    "bc_steps": EXPERIMENT["behavioral_cloning"]["steps"],
    "model_seeds": list(MODEL_SEEDS),
}


def experiment_spec() -> OneBodySpec:
    mechanics = EXPERIMENT["mechanics"]
    reward = EXPERIMENT["reward"]
    return OneBodySpec(
        version=EXPERIMENT["task_version"],
        initial_rotten_count=mechanics["initial_rotten_count"],
        initial_falling_count=mechanics["initial_falling_count"],
        spawn_y=mechanics["spawn_y"],
        max_episode_ticks=mechanics["max_episode_ticks"],
        train_heights=tuple(mechanics["train_heights"]),
        calibration_heights=tuple(mechanics["calibration_heights"]),
        validation_heights=tuple(mechanics["validation_heights"]),
        test_heights=tuple(mechanics["test_heights"]),
        hit_weight=reward["projectile_hit_weight"],
        aim_weight=reward["gaussian_aim_weight"],
        aim_sigma=reward["gaussian_aim_sigma_normalized"],
    )


def source_identity() -> dict[str, object]:
    def git(*args: str) -> str:
        return subprocess.run(
            ("git", *args),
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
    }


def acceptance_predicates(result: dict[str, object]) -> dict[str, bool]:
    test = result["held_out_test"]
    policy_runs = test["policy_runs"]
    random = test["random"]
    evaluations = [
        result["behavioral_cloning"]["calibration"],
        result["behavioral_cloning"]["validation"],
        result["random_validation"],
        random,
        *policy_runs,
    ]
    training_summaries = []
    for runs in result["ppo_candidates"].values():
        for run in runs:
            evaluations.extend((run["pre_ppo_validation"], run["validation"]))
            training_summaries.append(run["training"])

    raw_score_audit_passes = all(
        evaluation["raw_score_delta_count"] == evaluation["episodes"]
        and evaluation["raw_score_delta_sum"] == 0
        and evaluation["raw_score_delta_min"] == 0
        and evaluation["raw_score_delta_max"] == 0
        and evaluation["invalid_action_count"] == 0
        for evaluation in evaluations
    )
    optimization_statistics_are_finite = all(
        math.isfinite(value)
        for training in training_summaries
        for key, value in training.items()
        if isinstance(value, float)
    )
    selected_runs = result["ppo_candidates"][f"{result['selected_learning_rate']:.1e}"]
    minimum_delta = EXPERIMENT["evaluation"][
        "minimum_selected_seed_ppo_validation_delta"
    ]
    accepted_runtime = result["exact_runtime_attestation"]["accepted_identity"]
    exact_build = result["exact_runtime_build_info"]
    exact_runtime_matches = (
        exact_build["worker_executable_sha256"] == accepted_runtime["worker_sha256"]
        and exact_build["exact_library_sha256"]
        == accepted_runtime["exact_library_sha256"]
        and exact_build["protocol_version"] == accepted_runtime["protocol_version"]
        and exact_build["body_capacity"] == accepted_runtime["body_capacity"]
        and exact_build["pointer_bits"] == accepted_runtime["pointer_bits"]
        and exact_build["worker_backend"] == accepted_runtime["backend"]
        and result["exact_runtime_attestation"]["provenance"]["sha256"]
        == accepted_runtime["exact_library_sha256"]
    )
    return {
        "clean_source_tree": result["source"]["dirty"] is False,
        "checked_experiment_is_canonical": result["experiment"]["resolved"]
        == EXPERIMENT
        and result["budgets"] == CANONICAL_BUDGETS,
        "accepted_exact_runtime_attested": exact_runtime_matches,
        "exact_worker_hash_matches_runtime": result["exact_runtime_build_info"][
            "worker_executable_sha256"
        ]
        == result["runtime_files"]["exact_worker_sha256"],
        "bc_validation_at_least_90_percent": result["behavioral_cloning"]["validation"][
            "hit_rate"
        ]
        >= 0.90,
        "exact_test_backend": test["backend"] == "exact",
        "every_policy_seed_at_least_90_percent": min(
            run["hit_rate"] for run in policy_runs
        )
        >= 0.90,
        "median_margin_over_random_at_least_70_points": statistics.median(
            run["hit_rate"] for run in policy_runs
        )
        >= random["hit_rate"] + 0.70,
        "all_raw_score_audits_zero_and_no_invalid_actions": raw_score_audit_passes,
        "all_optimization_statistics_finite": optimization_statistics_are_finite,
        "every_selected_seed_ppo_preserves_warm_start": all(
            run["validation"]["hit_rate"]
            >= run["pre_ppo_validation"]["hit_rate"] + minimum_delta
            for run in selected_runs
        ),
        "selected_policy_checkpoints_persisted": len(
            result["selected_policy_checkpoints"]
        )
        == len(MODEL_SEEDS)
        and all(
            len(checkpoint["manifest_sha256"]) == 64
            and len(checkpoint["state_sha256"]) == 64
            for checkpoint in result["selected_policy_checkpoints"]
        ),
    }


def summarize_result(result: dict[str, object]) -> dict[str, object]:
    return {
        "schema": result["schema"],
        "source": result["source"],
        "runtime": result["runtime"],
        "runtime_files": result["runtime_files"],
        "experiment": result["experiment"],
        "reproduction": result["reproduction"],
        "exact_runtime_build_info": result["exact_runtime_build_info"],
        "exact_runtime_attestation": result["exact_runtime_attestation"],
        "seed_manifest_sha256": result["seed_manifest_sha256"],
        "task": result["task"],
        "model": result["model"],
        "budgets": result["budgets"],
        "allocator_keys": result["allocator_keys"],
        "learning_rate_candidates": result["learning_rate_candidates"],
        "behavioral_cloning": result["behavioral_cloning"],
        "random_validation": result["random_validation"],
        "ppo_candidates": result["ppo_candidates"],
        "selected_learning_rate": result["selected_learning_rate"],
        "selected_policy_checkpoints": result["selected_policy_checkpoints"],
        "held_out_test": result["held_out_test"],
        "acceptance": result["acceptance"],
        "deployable": False,
        "observation_provenance": "privileged_simulator",
        "transfer_gate": result["transfer_gate"],
    }


class TaskFamily:
    def __init__(
        self,
        heights: tuple[float, ...],
        lanes: int,
        *,
        library: Path | None,
        worker: Path | None,
        backend: str,
        spec: OneBodySpec,
    ) -> None:
        stack = ExitStack()
        self._stack = stack
        self.tasks = tuple(
            stack.enter_context(
                OneBodyTask(
                    lanes,
                    height,
                    library_path=library,
                    worker_path=worker,
                    physics_backend=backend,
                    spec=spec,
                )
            )
            for height in heights
        )
        self.lanes = lanes

    def config_hashes(self) -> dict[str, int]:
        return {str(task.height): task.config_hash for task in self.tasks}

    def close(self) -> None:
        self._stack.close()

    def reset(self, allocator: SeedAllocator):
        observations = []
        targets = []
        for task in self.tasks:
            observations.append(task.reset(allocator.take(self.lanes)))
            targets.append(task.target_xy)
        return concatenate_encoded(observations), torch.cat(targets)

    def step(self, actions: ActionTensor):
        outcomes = []
        for index, task in enumerate(self.tasks):
            begin = index * self.lanes
            end = begin + self.lanes
            outcomes.append(
                task.step(
                    ActionTensor(
                        actions.kind[:, begin:end],
                        actions.wait_index[:, begin:end],
                        actions.xy[:, begin:end],
                    )
                )
            )
        return outcomes


def summarize(outcomes) -> dict[str, float | int]:
    hit = torch.cat([value.hit for value in outcomes]).float()
    aim = torch.cat([value.aim_score for value in outcomes])
    reward = torch.cat([value.optimizer_reward for value in outcomes])
    raw = torch.cat([value.raw_reward for value in outcomes])
    target = torch.cat([value.target_xy for value in outcomes])
    action = torch.cat([value.action_xy for value in outcomes])
    return {
        "episodes": int(hit.numel()),
        "hit_rate": float(hit.mean()),
        "aim_score": float(aim.mean()),
        "optimizer_return": float(reward.mean()),
        "raw_score_delta": float(raw.double().mean()),
        "raw_score_delta_count": int(raw.numel()),
        "raw_score_delta_sum": int(raw.sum()),
        "raw_score_delta_min": int(raw.min()),
        "raw_score_delta_max": int(raw.max()),
        # OneBodyTask aborts and poisons itself before returning an outcome if
        # either primitive reports an invalid action.
        "invalid_action_count": 0,
        "coordinate_rmse": float((action - target).square().mean().sqrt()),
    }


@torch.no_grad()
def evaluate(
    model: RecurrentActorCritic | None,
    family: TaskFamily,
    *,
    split: str,
    rounds: int,
    key: int,
    random_seed: int = 0,
) -> dict[str, float]:
    allocator = SeedAllocator(split, key=key)
    random = torch.Generator().manual_seed(random_seed)
    all_outcomes = []
    coordinate_standard_deviations = []
    for _ in range(rounds):
        observations, _ = family.reset(allocator)
        lanes = len(family.tasks) * family.lanes
        if model is None:
            actions = ActionTensor(
                torch.ones((1, lanes), dtype=torch.long),
                torch.zeros((1, lanes), dtype=torch.long),
                torch.rand((1, lanes, 2), generator=random),
            )
        else:
            distribution, _ = policy_distribution(model, observations)
            alpha = distribution.alpha[..., 0, :]
            beta = distribution.beta[..., 0, :]
            variance = alpha * beta / ((alpha + beta).square() * (alpha + beta + 1))
            coordinate_standard_deviations.append(float(variance.sqrt().mean()))
            actions = distribution.deterministic()
        all_outcomes.extend(family.step(actions))
    result = summarize(all_outcomes)
    result["policy_coordinate_std"] = (
        statistics.mean(coordinate_standard_deviations)
        if coordinate_standard_deviations
        else 1.0 / (12.0**0.5)
    )
    return result


def fit_expert(
    model: RecurrentActorCritic,
    family: TaskFamily,
    *,
    steps: int,
    seed: int,
    dataset_rounds: int,
) -> dict[str, float]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=EXPERIMENT["behavioral_cloning"]["learning_rate"],
        eps=1e-5,
        foreach=False,
    )
    allocator = SeedAllocator("train", key=seed)
    observation_batches = []
    target_batches = []
    for _ in range(dataset_rounds):
        observations, targets = family.reset(allocator)
        observation_batches.append(observations)
        target_batches.append(targets)
    observations = concatenate_encoded(observation_batches)
    targets = torch.cat(target_batches)
    expert = expert_actions(targets)
    initial_loss = 0.0
    final_loss = 0.0
    regression_steps = max(1, (3 * steps) // 4)
    for index in range(steps):
        distribution, _ = policy_distribution(model, observations)
        mean = distribution.alpha[..., 0, :] / (
            distribution.alpha[..., 0, :] + distribution.beta[..., 0, :]
        )
        variance = (
            distribution.alpha[..., 0, :]
            * distribution.beta[..., 0, :]
            / (
                (distribution.alpha[..., 0, :] + distribution.beta[..., 0, :]).square()
                * (distribution.alpha[..., 0, :] + distribution.beta[..., 0, :] + 1.0)
            )
        ).mean()
        coordinate_mse = (mean - expert.xy).square().mean()
        nll = -distribution.log_prob(expert).mean()
        loss = (
            100.0 * coordinate_mse
            if index < regression_steps
            else 1000.0 * coordinate_mse + nll + 100.0 * variance
        )
        if index == 0:
            initial_loss = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
    return {
        "steps": steps,
        "dataset_rounds": dataset_rounds,
        "examples": int(targets.shape[0]),
        "objective": "75%: 100*mse; 25%: 1000*mse + nll + 100*beta_variance",
        "initial_objective": initial_loss,
        "final_objective": final_loss,
    }


def train_bc(
    family: TaskFamily, *, steps: int, seed: int
) -> tuple[RecurrentActorCritic, dict[str, float]]:
    torch.manual_seed(seed)
    model = RecurrentActorCritic(TEACHER_V1, config=MODEL_CONFIG)
    return model, fit_expert(
        model,
        family,
        steps=steps,
        seed=seed,
        dataset_rounds=EXPERIMENT["behavioral_cloning"]["dataset_rounds"],
    )


def train_ppo(
    family: TaskFamily,
    *,
    learning_rate: float,
    model_seed: int,
    updates: int,
) -> tuple[RecurrentActorCritic, dict[str, float], dict[str, torch.Tensor]]:
    torch.manual_seed(model_seed)
    model = RecurrentActorCritic(TEACHER_V1, config=MODEL_CONFIG)
    ppo = EXPERIMENT["ppo"]
    expert_steps = ppo["expert_warm_start_steps"]
    fit_expert(
        model,
        family,
        steps=expert_steps,
        seed=model_seed,
        dataset_rounds=ppo["expert_warm_start_dataset_rounds"],
    )
    pre_ppo_state = copy.deepcopy(model.state_dict())
    config = PPOConfig(
        learning_rate=learning_rate,
        final_learning_rate_fraction=ppo["final_learning_rate_fraction"],
        epochs=ppo["epochs"],
        lane_minibatch_size=ppo["lane_minibatch_size"],
        clip_ratio=ppo["clip_ratio"],
        value_clip=ppo["value_clip"],
        value_coefficient=ppo["value_coefficient"],
        entropy_coefficient=ppo["entropy_coefficient"],
        max_gradient_norm=ppo["max_gradient_norm"],
        target_kl=ppo["target_kl"],
    )
    trainer = PPOTrainer(
        model,
        config=config,
        total_updates=updates,
        sampler_seed=model_seed ^ PPO_SAMPLER_XOR,
    )
    allocator = SeedAllocator("train", key=model_seed)
    hit_history = []
    stats_history = []
    for _ in range(updates):
        observations, _ = family.reset(allocator)
        with torch.no_grad():
            distribution, values = policy_distribution(model, observations)
            actions = distribution.sample()
            old_log_prob_components = distribution.log_prob_components(actions)
            old_log_prob = old_log_prob_components.total
        outcomes = family.step(actions)
        rewards = torch.cat([value.optimizer_reward for value in outcomes])
        hit_history.append(
            float(torch.cat([value.hit for value in outcomes]).float().mean())
        )
        batch = one_body_training_batch(
            model,
            observations,
            actions,
            old_log_prob,
            old_log_prob_components,
            values,
            rewards,
        )
        stats_history.append(trainer.update(batch))
    return (
        model,
        {
            "updates": updates,
            "expert_warm_start_steps": expert_steps,
            "training_episodes": updates * len(family.tasks) * family.lanes,
            "initial_10_update_hit_rate": statistics.mean(hit_history[:10]),
            "final_10_update_hit_rate": statistics.mean(hit_history[-10:]),
            "final_10_entropy": statistics.mean(
                value.entropy for value in stats_history[-10:]
            ),
            "final_10_approximate_kl": statistics.mean(
                value.approximate_kl for value in stats_history[-10:]
            ),
            "final_10_gradient_norm": statistics.mean(
                value.gradient_norm for value in stats_history[-10:]
            ),
            "early_stopped_updates": sum(
                value.early_stopped for value in stats_history
            ),
            "final_learning_rate": trainer.schedule.learning_rate,
        },
        pre_ppo_state,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--library",
        type=Path,
        default=ROOT / "build-physics-integration-portable" / "libirisu_clone.so",
    )
    parser.add_argument(
        "--exact-worker",
        type=Path,
        default=ROOT
        / "build-physics-integration-exact-multiworld-2"
        / "irisu-exact-worker",
    )
    parser.add_argument("--updates", type=int, default=CANONICAL_BUDGETS["updates"])
    parser.add_argument("--lanes", type=int, default=CANONICAL_BUDGETS["lanes"])
    parser.add_argument(
        "--evaluation-rounds",
        type=int,
        default=CANONICAL_BUDGETS["evaluation_rounds"],
    )
    parser.add_argument("--bc-steps", type=int, default=CANONICAL_BUDGETS["bc_steps"])
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=ROOT / "benchmarks/results/rl-r2b-one-body-models",
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if min(args.updates, args.lanes, args.evaluation_rounds, args.bc_steps) <= 0:
        parser.error("all budgets must be positive")
    if not args.library.is_file():
        parser.error("portable training library does not exist")
    if not args.exact_worker.is_file():
        parser.error("the attested exact worker is required for R2b evidence")
    if args.checkpoint_root.exists():
        parser.error("checkpoint root must not already exist")
    attestation = ACCEPTED_EXACT_RUNTIME_2026_07_21.attest(args.exact_worker.resolve())
    attestation["build_info"].pop("worker_pid", None)
    attestation["build_info"].pop("config_hash", None)
    attestation = {
        "accepted_identity": asdict(ACCEPTED_EXACT_RUNTIME_2026_07_21),
        "build_info": attestation["build_info"],
        "provenance": {
            key: attestation["provenance"][key] for key in ("status", "bytes", "sha256")
        },
    }
    runtime = EXPERIMENT["runtime"]
    torch.set_num_threads(runtime["torch_threads"])
    torch.set_num_interop_threads(runtime["torch_interop_threads"])
    torch.use_deterministic_algorithms(runtime["deterministic_algorithms"])
    spec = experiment_spec()
    result: dict[str, object] = {
        "schema": "irisu-r2b-learning-proof-v1",
        "source": source_identity(),
        "runtime": {
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "threads": runtime["torch_threads"],
            "interop_threads": runtime["torch_interop_threads"],
            "deterministic_algorithms": runtime["deterministic_algorithms"],
        },
        "experiment": {
            "path": str(EXPERIMENT_PATH.relative_to(ROOT)),
            "sha256": hashlib.sha256(EXPERIMENT_BYTES).hexdigest(),
            "resolved": EXPERIMENT,
        },
        "exact_runtime_attestation": attestation,
        "task": {**spec.manifest(), "sha256": spec.sha256},
        "model": RecurrentActorCritic(TEACHER_V1, config=MODEL_CONFIG).manifest(),
        "budgets": {
            "updates": args.updates,
            "lanes": args.lanes,
            "evaluation_rounds": args.evaluation_rounds,
            "bc_steps": args.bc_steps,
            "model_seeds": list(MODEL_SEEDS),
        },
        "learning_rate_candidates": LEARNING_RATES,
        "seed_manifest_sha256": SeedAllocator().manifest_sha256,
        "runtime_files": {
            "portable_library_sha256": hashlib.sha256(
                args.library.read_bytes()
            ).hexdigest(),
            "exact_worker_sha256": (
                hashlib.sha256(args.exact_worker.read_bytes()).hexdigest()
                if args.exact_worker.is_file()
                else None
            ),
        },
        "reproduction": {
            "command": (
                "PYTHONPATH=python uv run --extra training python "
                "benchmarks/rl_r2b.py --summary"
            ),
            "output": "benchmarks/results/rl-r2b-one-body-2026-07-22.json",
        },
        "allocator_keys": {
            "validation": VALIDATION_ALLOCATOR_KEY,
            "final_test": FINAL_TEST_ALLOCATOR_KEY,
        },
        "deployable": False,
        "observation_provenance": "privileged_simulator",
        "transfer_gate": "R4 causal tracker and real input calibration pending",
    }
    with ExitStack() as stack:
        train = TaskFamily(
            spec.train_heights,
            args.lanes,
            library=args.library,
            worker=None,
            backend="portable",
            spec=spec,
        )
        stack.callback(train.close)
        calibration = TaskFamily(
            spec.calibration_heights,
            args.lanes,
            library=args.library,
            worker=None,
            backend="portable",
            spec=spec,
        )
        stack.callback(calibration.close)
        validation = TaskFamily(
            spec.validation_heights,
            args.lanes,
            library=args.library,
            worker=None,
            backend="portable",
            spec=spec,
        )
        stack.callback(validation.close)
        result["task"]["mechanics_config_hashes"] = {
            "train": train.config_hashes(),
            "calibration": calibration.config_hashes(),
            "validation": validation.config_hashes(),
        }

        bc_model, bc_training = train_bc(train, steps=args.bc_steps, seed=BC_SEED)
        result["behavioral_cloning"] = {
            "training": bc_training,
            "calibration": evaluate(
                bc_model,
                calibration,
                split="calibration",
                rounds=args.evaluation_rounds,
                key=CALIBRATION_ALLOCATOR_KEY,
            ),
            "validation": evaluate(
                bc_model,
                validation,
                split="validation",
                rounds=args.evaluation_rounds,
                key=VALIDATION_ALLOCATOR_KEY,
            ),
        }
        result["random_validation"] = evaluate(
            None,
            validation,
            split="validation",
            rounds=args.evaluation_rounds,
            key=VALIDATION_ALLOCATOR_KEY,
            random_seed=RANDOM_VALIDATION_SEED,
        )

        candidates: dict[str, list[dict[str, object]]] = {}
        states: dict[float, list[dict[str, torch.Tensor]]] = {}
        for learning_rate in LEARNING_RATES:
            runs = []
            states[learning_rate] = []
            for model_seed in MODEL_SEEDS:
                model, training, pre_ppo_state = train_ppo(
                    train,
                    learning_rate=learning_rate,
                    model_seed=model_seed,
                    updates=args.updates,
                )
                validation_metrics = evaluate(
                    model,
                    validation,
                    split="validation",
                    rounds=args.evaluation_rounds,
                    key=VALIDATION_ALLOCATOR_KEY,
                )
                warm_model = RecurrentActorCritic(TEACHER_V1, config=MODEL_CONFIG)
                warm_model.load_state_dict(pre_ppo_state, strict=True)
                pre_ppo_validation = evaluate(
                    warm_model,
                    validation,
                    split="validation",
                    rounds=args.evaluation_rounds,
                    key=VALIDATION_ALLOCATOR_KEY,
                )
                runs.append(
                    {
                        "model_seed": model_seed,
                        "training": training,
                        "pre_ppo_validation": pre_ppo_validation,
                        "validation": validation_metrics,
                    }
                )
                states[learning_rate].append(copy.deepcopy(model.state_dict()))
            candidates[f"{learning_rate:.1e}"] = runs
        result["ppo_candidates"] = candidates
        selected = max(
            LEARNING_RATES,
            key=lambda rate: (
                statistics.median(
                    run["validation"]["hit_rate"] for run in candidates[f"{rate:.1e}"]
                ),
                statistics.median(
                    run["validation"]["aim_score"] for run in candidates[f"{rate:.1e}"]
                ),
                -rate,
            ),
        )
        result["selected_learning_rate"] = selected

        test_backend = "exact"
        test = TaskFamily(
            spec.test_heights,
            args.lanes,
            library=None,
            worker=args.exact_worker,
            backend=test_backend,
            spec=spec,
        )
        stack.callback(test.close)
        exact_build_info = test.tasks[0].env.envs[0].build_info()
        exact_build_info.pop("worker_pid", None)
        result["exact_runtime_build_info"] = exact_build_info
        result["task"]["mechanics_config_hashes"]["test"] = test.config_hashes()
        test_runs = []
        for model_seed, state in zip(MODEL_SEEDS, states[selected]):
            model = RecurrentActorCritic(TEACHER_V1, config=MODEL_CONFIG)
            model.load_state_dict(state, strict=True)
            test_runs.append(
                {
                    "model_seed": model_seed,
                    **evaluate(
                        model,
                        test,
                        split="test",
                        rounds=args.evaluation_rounds,
                        key=FINAL_TEST_ALLOCATOR_KEY,
                    ),
                }
            )
        random_test = evaluate(
            None,
            test,
            split="test",
            rounds=args.evaluation_rounds,
            key=FINAL_TEST_ALLOCATOR_KEY,
            random_seed=RANDOM_TEST_SEED,
        )
        result["held_out_test"] = {
            "backend": test_backend,
            "policy_runs": test_runs,
            "random": random_test,
            "median_policy_hit_rate": statistics.median(
                run["hit_rate"] for run in test_runs
            ),
        }
        selected_checkpoints = []
        for model_seed, state, test_metrics in zip(
            MODEL_SEEDS, states[selected], test_runs
        ):
            identity = {
                "schema": "irisu-r2b-selected-policy-v1",
                "source": result["source"],
                "experiment_sha256": result["experiment"]["sha256"],
                "task_sha256": result["task"]["sha256"],
                "model": result["model"],
                "model_seed": model_seed,
                "selected_learning_rate": selected,
                "held_out_test": test_metrics,
                "exact_runtime": result["exact_runtime_attestation"][
                    "accepted_identity"
                ],
                "seed_manifest_sha256": result["seed_manifest_sha256"],
                "observation_provenance": "privileged_simulator",
                "deployable": False,
            }
            generation = f"seed-{model_seed}"
            directory = save_checkpoint(
                args.checkpoint_root,
                generation,
                identity=identity,
                state={"model": state},
            )
            manifest_path = directory / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            selected_checkpoints.append(
                {
                    "generation": generation,
                    "path": str(
                        directory.relative_to(ROOT)
                        if directory.is_relative_to(ROOT)
                        else directory
                    ),
                    "identity": identity,
                    "manifest_sha256": hashlib.sha256(
                        manifest_path.read_bytes()
                    ).hexdigest(),
                    "state_sha256": manifest["files"]["state.pt"],
                }
            )
        result["selected_policy_checkpoints"] = selected_checkpoints
        predicates = acceptance_predicates(result)
        result["acceptance"] = {
            "predicates": predicates,
            "pass": all(predicates.values()),
        }
    payload = summarize_result(result) if args.summary else result
    print(json.dumps(payload, indent=2, sort_keys=True, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
