#!/usr/bin/env python3
"""Run the preregistered R2b behavioral-cloning and one-body PPO proof."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import statistics
import subprocess
from contextlib import ExitStack
from pathlib import Path

import torch
from torch import nn

from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
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
from irisu_rl.seeds import SeedAllocator
from irisu_rl.torch_distribution import ActionTensor


ROOT = Path(__file__).resolve().parents[1]
MODEL_CONFIG = RecurrentModelConfig(32, 32, 64, 64, 1)
MODEL_SEEDS = (17, 29, 43)
LEARNING_RATES = (1e-4, 3e-4, 6e-4)


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


def summarize(outcomes) -> dict[str, float]:
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
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-5, foreach=False)
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
    return model, fit_expert(model, family, steps=steps, seed=seed, dataset_rounds=8)


def train_ppo(
    family: TaskFamily,
    *,
    learning_rate: float,
    model_seed: int,
    updates: int,
) -> tuple[RecurrentActorCritic, dict[str, float], dict[str, torch.Tensor]]:
    torch.manual_seed(model_seed)
    model = RecurrentActorCritic(TEACHER_V1, config=MODEL_CONFIG)
    expert_steps = 200
    fit_expert(
        model,
        family,
        steps=expert_steps,
        seed=model_seed,
        dataset_rounds=4,
    )
    pre_ppo_state = copy.deepcopy(model.state_dict())
    config = PPOConfig(
        learning_rate=learning_rate,
        final_learning_rate_fraction=0.2,
        epochs=3,
        lane_minibatch_size=20,
        entropy_coefficient=0.002,
        target_kl=0.08,
    )
    trainer = PPOTrainer(
        model,
        config=config,
        total_updates=updates,
        sampler_seed=model_seed ^ 0xA5A5,
    )
    allocator = SeedAllocator("train", key=model_seed)
    hit_history = []
    stats_history = []
    for _ in range(updates):
        observations, _ = family.reset(allocator)
        with torch.no_grad():
            distribution, values = policy_distribution(model, observations)
            actions = distribution.sample()
            old_log_prob = distribution.log_prob(actions)
        outcomes = family.step(actions)
        rewards = torch.cat([value.optimizer_reward for value in outcomes])
        hit_history.append(
            float(torch.cat([value.hit for value in outcomes]).float().mean())
        )
        batch = one_body_training_batch(
            model, observations, actions, old_log_prob, values, rewards
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
    parser.add_argument("--updates", type=int, default=120)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--evaluation-rounds", type=int, default=8)
    parser.add_argument("--bc-steps", type=int, default=500)
    args = parser.parse_args()
    if min(args.updates, args.lanes, args.evaluation_rounds, args.bc_steps) <= 0:
        parser.error("all budgets must be positive")
    if not args.library.is_file():
        parser.error("portable training library does not exist")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    spec = OneBodySpec()
    result: dict[str, object] = {
        "schema": "irisu-r2b-learning-proof-v1",
        "source": source_identity(),
        "task": {**spec.manifest(), "sha256": spec.sha256},
        "model": MODEL_CONFIG.manifest(),
        "budgets": vars(args) | {"model_seeds": MODEL_SEEDS},
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

        bc_model, bc_training = train_bc(train, steps=args.bc_steps, seed=7)
        result["behavioral_cloning"] = {
            "training": bc_training,
            "calibration": evaluate(
                bc_model,
                calibration,
                split="calibration",
                rounds=args.evaluation_rounds,
                key=701,
            ),
            "validation": evaluate(
                bc_model,
                validation,
                split="validation",
                rounds=args.evaluation_rounds,
                key=702,
            ),
        }
        result["random_validation"] = evaluate(
            None,
            validation,
            split="validation",
            rounds=args.evaluation_rounds,
            key=702,
            random_seed=703,
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
                    key=702,
                )
                warm_model = RecurrentActorCritic(TEACHER_V1, config=MODEL_CONFIG)
                warm_model.load_state_dict(pre_ppo_state, strict=True)
                pre_ppo_validation = evaluate(
                    warm_model,
                    validation,
                    split="validation",
                    rounds=args.evaluation_rounds,
                    key=702,
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

        test_backend = "exact" if args.exact_worker.is_file() else "portable"
        test = TaskFamily(
            spec.test_heights,
            args.lanes,
            library=args.library if test_backend == "portable" else None,
            worker=args.exact_worker if test_backend == "exact" else None,
            backend=test_backend,
            spec=spec,
        )
        stack.callback(test.close)
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
                        key=704,
                    ),
                }
            )
        random_test = evaluate(
            None,
            test,
            split="test",
            rounds=args.evaluation_rounds,
            key=704,
            random_seed=705,
        )
        result["held_out_test"] = {
            "backend": test_backend,
            "policy_runs": test_runs,
            "random": random_test,
            "median_policy_hit_rate": statistics.median(
                run["hit_rate"] for run in test_runs
            ),
            "pass": (
                min(run["hit_rate"] for run in test_runs) >= 0.90
                and statistics.median(run["hit_rate"] for run in test_runs)
                >= random_test["hit_rate"] + 0.70
            ),
        }
    print(json.dumps(result, indent=2, sort_keys=True, default=str, allow_nan=False))


if __name__ == "__main__":
    main()
