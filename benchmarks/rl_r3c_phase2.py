#!/usr/bin/env python3
"""Development-only R3c recurrent DAgger and complete-game score experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from irisu_env import IrisuEnv
from irisu_pointer.checkpoint import (
    load_actor_pointer_policy,
    load_teacher_pointer_policy,
    save_pointer_checkpoint,
)
from irisu_pointer.evaluate import (
    ArtifactBinding,
    DevelopmentSuite,
    PromotionCriteria,
    evaluate_full_games,
    identity_sha256,
)
from irisu_pointer.experts import (
    PointerExpertDecision,
    matcher_anchor,
)
from irisu_pointer.actor_distill import (
    ActorDistillConfig,
    perfect_actor_record,
)
from irisu_pointer.improvement import (
    DaggerCollectionConfig,
    DaggerLoopConfig,
    DaggerPolicyImprover,
    DistributionalLeafEvaluator,
    file_sha256,
    spread_sequence_minibatches,
)
from irisu_pointer.model import EntityPointerActorCritic, PointerModelConfig
from irisu_pointer.macro_search import SpawnCensoredMacroBeamTeacher
from irisu_pointer.search import SpawnCensoredSearchTeacher
from irisu_pointer.sequence import (
    PointerSequenceConfig,
    PointerSequenceTrainer,
    pad_pointer_episodes,
)
from irisu_pointer.trajectory import DelayedRewardSpec
from irisu_rl.schema import ACTOR_VISION_V1, TEACHER_V1


ROOT = Path(__file__).resolve().parents[1]
TRUSTED_PORTABLE = (
    ROOT
    / "artifacts/r3/runtime/main-0c48dba-20260723/portable-build/libirisu_clone.so"
)
DEVELOPMENT_SEEDS = (
    0x13579BDF,
    0x2468ACE0,
    0x31415926,
    0x5A17C0DE,
    0x6C8E9CF1,
    0x7B1D3F59,
    0x8D2E4A60,
    0xA5C31E79,
)


def _source_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_identity() -> dict[str, object]:
    revision = _source_revision()
    files = (
        Path(__file__).resolve(),
        *sorted((ROOT / "python/irisu_pointer").glob("*.py")),
        ROOT / "configs/rl/experiments/r3c-policy-iteration-v1.toml",
    )
    manifest = {
        "schema": "irisu-r3c-phase2-source-identity-v1",
        "git_revision": revision,
        "files": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in files
        },
    }
    encoded = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {**manifest, "sha256": hashlib.sha256(encoded).hexdigest()}


def _require_source_identity(expected: dict[str, object]) -> None:
    if _source_identity() != expected:
        raise RuntimeError("Phase-2 source identity changed during the run")


def _paced_matcher(
    observation: dict[str, Any], spec
) -> PointerExpertDecision:
    decision = matcher_anchor(observation, spec)
    return PointerExpertDecision.wait(4) if decision.kind == 0 else decision


class _BoundMatcher:
    schema_sha256 = TEACHER_V1.sha256

    def __init__(self, artifact_sha256: str, pointer_sha256: str) -> None:
        self.artifact_sha256 = artifact_sha256
        self.pointer_action_sha256 = pointer_sha256

    def reset(self, seed: int = 0) -> None:
        del seed

    def act(self, observation: dict[str, Any]) -> PointerExpertDecision:
        from irisu_pointer.action import PointerActionSpec

        return _paced_matcher(observation, PointerActionSpec())


class _PerfectTrackActor:
    """Development adapter standing in for the not-yet-connected vision tracker."""

    def __init__(self, policy) -> None:
        self.policy = policy
        self.artifact_sha256 = policy.artifact_sha256
        self.schema_sha256 = policy.schema_sha256
        self.pointer_action_sha256 = policy.pointer_action_sha256

    def reset(self, seed: int = 0) -> None:
        self.policy.reset(seed)

    def act(self, observation: dict[str, Any]):
        return self.policy.act(perfect_actor_record(observation))


def _marker(path: Path, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    if path.exists() and path.read_bytes() != encoded:
        raise FileExistsError(f"refusing to replace a different baseline marker: {path}")
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


def _beta_schedule(iterations: int) -> tuple[int, ...]:
    if iterations == 1:
        return (1_000_000,)
    return tuple(
        round(1_000_000 * (1.0 - index / (iterations - 1)) ** 2)
        for index in range(iterations)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, default=TRUSTED_PORTABLE)
    parser.add_argument(
        "--policy-out",
        type=Path,
        default=Path("/tmp/r3c-phase2-dev-policy-20260728.pt"),
    )
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--episodes-per-iteration", type=int, default=3)
    parser.add_argument("--training-episode-ticks", type=int, default=800)
    parser.add_argument("--evaluation-episode-ticks", type=int, default=20_000)
    parser.add_argument("--evaluation-seeds", type=int, default=8)
    parser.add_argument("--sequence-updates", type=int, default=6)
    parser.add_argument("--maximum-search-queries", type=int, default=8)
    parser.add_argument(
        "--search-mode",
        choices=("macro", "tactical"),
        default="macro",
    )
    parser.add_argument("--torch-threads", type=int, default=1)
    args = parser.parse_args()
    positive = (
        args.iterations,
        args.episodes_per_iteration,
        args.training_episode_ticks,
        args.evaluation_episode_ticks,
        args.evaluation_seeds,
        args.sequence_updates,
        args.torch_threads,
    )
    if any(value < 1 for value in positive):
        parser.error("count and tick arguments must be positive")
    if not 1 <= args.evaluation_seeds <= len(DEVELOPMENT_SEEDS):
        parser.error("evaluation-seeds exceeds the fixed development suite")
    if args.maximum_search_queries < 0:
        parser.error("maximum-search-queries must be nonnegative")

    started = time.monotonic()
    torch.set_num_threads(args.torch_threads)
    torch.manual_seed(2026072802)
    library = args.library.resolve(strict=True)
    runtime_sha256 = file_sha256(library)
    source_identity = _source_identity()
    source_revision = str(source_identity["git_revision"])
    training_config = {
        "max_episode_ticks": args.training_episode_ticks,
    }

    model_config = PointerModelConfig(
        global_hidden=64,
        body_hidden=64,
        attention_hidden=96,
        attention_heads=4,
        attention_layers=2,
        feedforward_hidden=192,
        relation_hidden=48,
        matcher_prior_scale=16.0,
        recurrent_hidden=96,
    )
    model = EntityPointerActorCritic(TEACHER_V1, config=model_config)
    sequence_config = PointerSequenceConfig(
        learning_rate=3e-4,
        kind_coefficient=1.0,
        wait_coefficient=0.5,
        target_coefficient=2.0,
        template_coefficient=1.0,
        value_coefficient=0.01,
        entropy_coefficient=0.001,
        burn_in_steps=4,
        tbptt_steps=32,
        seed=2026072803,
    )
    trainer = PointerSequenceTrainer(model, config=sequence_config)
    leaf_evaluator = DistributionalLeafEvaluator(
        model,
        lower_quantile_fraction=0.25,
    )
    if args.search_mode == "macro":
        search = SpawnCensoredMacroBeamTeacher(
            max_depth=3,
            beam_width=5,
            max_candidates=25,
            max_rollout_ticks=64,
            max_branch_evaluations=512,
            continuation_evaluator=leaf_evaluator,
            continuation_scale=0.25,
            fallback_teacher=_paced_matcher,
            minimum_search_score=0.0,
        )
    else:
        search = SpawnCensoredSearchTeacher(
            max_rollout_ticks=64,
            max_candidates=64,
            continuation_evaluator=leaf_evaluator,
            continuation_scale=0.25,
        )
    beta = _beta_schedule(args.iterations)
    improver = DaggerPolicyImprover(
        model,
        trainer,
        source_revision=str(source_identity["sha256"]),
        runtime_sha256=runtime_sha256,
        expensive_teacher=search,
        fallback_teacher=_paced_matcher,
        reward_spec=DelayedRewardSpec(),
        actor_distill_config=ActorDistillConfig(
            position_noise_pixels=2.0,
            velocity_noise_pixels_per_second=10.0,
            confidence_noise=0.05,
            seed=2026072804,
        ),
        collection_config=DaggerCollectionConfig(
            max_decisions=args.training_episode_ticks,
            uncertainty_threshold=0.55,
            search_query_stride=32,
            maximum_search_queries=args.maximum_search_queries,
            gamma_tick=0.9995,
        ),
        loop_config=DaggerLoopConfig(
            teacher_beta_ppm=beta,
            sequence_updates_per_iteration=args.sequence_updates,
            sequence_minibatch_episodes=4,
            replay_capacity_episodes=max(
                8, args.iterations * args.episodes_per_iteration
            ),
            elite_fraction=0.25,
            failure_fraction=0.25,
        ),
        leaf_evaluator=leaf_evaluator,
    )

    def training_env() -> IrisuEnv:
        return IrisuEnv(
            library_path=library,
            physics_backend="portable",
            config=training_config,
        )

    training_seed_base = 20260801
    seed_waves = tuple(
        tuple(
            training_seed_base
            + iteration * args.episodes_per_iteration
            + episode
            for episode in range(args.episodes_per_iteration)
        )
        for iteration in range(args.iterations)
    )
    iteration_metrics = improver.run(training_env, seed_waves)

    _require_source_identity(source_identity)
    checkpoint_sha256 = save_pointer_checkpoint(
        args.policy_out,
        model,
        metadata={
            "source_revision": source_revision,
            "source_identity": source_identity,
            "runtime_sha256": runtime_sha256,
            "training_seed_waves": seed_waves,
            "iteration_metrics": [
                value.manifest() for value in iteration_metrics
            ],
            "development_only": True,
        },
        overwrite=True,
    )
    actor_model = EntityPointerActorCritic(
        ACTOR_VISION_V1, config=model_config
    )
    actor_sequence_config = PointerSequenceConfig(
        **{
            **asdict(sequence_config),
            "seed": 2026072805,
        }
    )
    actor_trainer = PointerSequenceTrainer(
        actor_model, config=actor_sequence_config
    )
    actor_sequences = [value.actor_supervision for value in improver.replay]
    actor_metrics = None
    for sequences in spread_sequence_minibatches(
        actor_sequences,
        updates=args.sequence_updates,
        batch_size=min(4, len(actor_sequences)),
    ):
        actor_metrics = actor_trainer.step(pad_pointer_episodes(sequences))
    assert actor_metrics is not None
    actor_policy_out = args.policy_out.with_name(
        f"{args.policy_out.stem}-actor{args.policy_out.suffix}"
    )
    actor_checkpoint_sha256 = save_pointer_checkpoint(
        actor_policy_out,
        actor_model,
        metadata={
            "source_revision": source_revision,
            "source_identity": source_identity,
            "teacher_checkpoint_sha256": checkpoint_sha256,
            "training_episode_identities": [
                value.identity for value in actor_sequences
            ],
            "sequence_metrics": asdict(actor_metrics),
            "development_only": True,
            "perfect_track_training_bridge": True,
        },
        overwrite=True,
    )
    evaluation_config = {
        "max_episode_ticks": args.evaluation_episode_ticks,
    }
    criteria = PromotionCriteria()
    full_objective_eligible = (
        args.evaluation_episode_ticks
        >= criteria.minimum_median_survival_ticks
    )
    suite = DevelopmentSuite(
        label=(
            "r3c-phase2-unseen-development-v1"
            if full_objective_eligible
            else "r3c-phase2-short-unseen-development-v1"
        ),
        seeds=DEVELOPMENT_SEEDS[: args.evaluation_seeds],
        config=tuple(evaluation_config.items()),
        max_decisions_per_episode=args.evaluation_episode_ticks + 1,
    )
    with IrisuEnv(
        library_path=library,
        physics_backend="portable",
        config=evaluation_config,
    ) as env:
        runner_sha256 = identity_sha256(env.runner_identity_manifest())
    binding = ArtifactBinding(
        label="r3c-phase2-development-policy-v1",
        policy_path=args.policy_out,
        policy_sha256=checkpoint_sha256,
        runtime_path=library,
        runtime_sha256=runtime_sha256,
        runner_identity_sha256=runner_sha256,
    )
    learned = evaluate_full_games(
        lambda: load_teacher_pointer_policy(
            args.policy_out, expected_sha256=checkpoint_sha256
        ),
        binding,
        suite=suite,
        criteria=criteria,
    )
    actor_binding = ArtifactBinding(
        label="r3c-phase2-development-actor-policy-v1",
        policy_path=actor_policy_out,
        policy_sha256=actor_checkpoint_sha256,
        runtime_path=library,
        runtime_sha256=runtime_sha256,
        runner_identity_sha256=runner_sha256,
        schema_sha256=ACTOR_VISION_V1.sha256,
    )
    actor = evaluate_full_games(
        lambda: _PerfectTrackActor(
            load_actor_pointer_policy(
                actor_policy_out,
                expected_sha256=actor_checkpoint_sha256,
            )
        ),
        actor_binding,
        suite=suite,
        criteria=criteria,
    )

    marker_path = args.policy_out.with_name("r3c-phase2-dev-matcher-v1.json")
    marker_sha256 = _marker(
        marker_path,
        {"policy": "paced-matcher-v1", "wait_ticks": 4},
    )
    matcher_binding = ArtifactBinding(
        label="r3c-phase2-development-matcher-v1",
        policy_path=marker_path,
        policy_sha256=marker_sha256,
        runtime_path=library,
        runtime_sha256=runtime_sha256,
        runner_identity_sha256=runner_sha256,
    )
    matcher = evaluate_full_games(
        lambda: _BoundMatcher(
            marker_sha256, matcher_binding.pointer_action_sha256
        ),
        matcher_binding,
        suite=suite,
        criteria=criteria,
    )
    _require_source_identity(source_identity)
    learned_score = learned.aggregate["raw_score"]["median"]
    actor_score = actor.aggregate["raw_score"]["median"]
    matcher_score = matcher.aggregate["raw_score"]["median"]
    output = {
        "schema": "irisu-r3c-phase2-development-run-v1",
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
        "source_revision": source_revision,
        "source_identity": source_identity,
        "runtime": {
            "path": str(library),
            "sha256": runtime_sha256,
        },
        "training": {
            "config": training_config,
            "search_mode": args.search_mode,
            "seed_waves": seed_waves,
            "teacher_beta_ppm": beta,
            "sequence_config": asdict(sequence_config),
            "model_config": model_config.manifest(),
            "iterations": [value.manifest() for value in iteration_metrics],
            "replay_episodes": len(improver.replay),
        },
        "checkpoint": {
            "path": str(args.policy_out.resolve()),
            "sha256": checkpoint_sha256,
        },
        "actor_distillation": {
            "perfect_track_training_bridge": True,
            "noise": asdict(improver.actor_distill_config),
            "sequence_metrics": asdict(actor_metrics),
            "checkpoint": {
                "path": str(actor_policy_out.resolve()),
                "sha256": actor_checkpoint_sha256,
            },
        },
        "evaluation": {
            "learned": learned.manifest(),
            "actor": actor.manifest(),
            "paced_matcher": matcher.manifest(),
            "median_raw_score_delta": learned_score - matcher_score,
            "actor_median_raw_score_delta": actor_score - matcher_score,
            "full_objective_horizon_eligible": full_objective_eligible,
            "six_figure_gate_passed": (
                full_objective_eligible and actor.promoted
            ),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    print(json.dumps(output, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
