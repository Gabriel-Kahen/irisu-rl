#!/usr/bin/env python3
"""Development-only contextual ranking and safety-DAgger campaign.

This driver binds the trusted portable runtime and frozen R3d-v5 policy,
collects exact all-branch runway labels on base- and learner-visited states,
re-trains a three-member whole-board contextual ranker from scratch after each
merge, and rejects any aggregate improvement with a paired survival disaster.
It never reads sealed material or writes canonical R3 storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from irisu_env import IrisuEnv
from irisu_pointer.geometry_checkpoint import (
    load_geometry_checkpoint,
    load_safeguarded_geometry_ensemble_policy,
    save_geometry_checkpoint,
)
from irisu_pointer.geometry_dagger import (
    GeometryDaggerConfig,
    LearnerVisitedGeometryDaggerPolicy,
)
from irisu_pointer.geometry_learning import (
    GeometryModelConfig,
    GeometrySelectorModel,
)
from irisu_pointer.geometry_policy import (
    GeometryPolicyConfig,
    GeometrySelectorEnsemble,
    geometry_candidate_vocabulary_sha256,
)
from irisu_pointer.geometry_ranking import (
    GeometryRankingDataset,
    geometry_ranking_loss,
    train_geometry_ranker,
)
from irisu_pointer.geometry_search import GeometrySearchConfig
from irisu_pointer.runway_search import (
    RunwayGeometrySearch,
    RunwaySearchConfig,
)
from irisu_rl.schema import TEACHER_V1


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = Path(__file__).resolve().parent
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))
import rl_r3d_steering as r3d  # noqa: E402
import rl_r3e_sustainable as r3e  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/rl/experiments/r3e-dagger-ensemble-v1.toml"
DEFAULT_RUNTIME = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/runtime/"
    "main-0c48dba-20260723/portable-build/libirisu_clone.so"
)
DEFAULT_BASE = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3d-survival-v5-20260729/long-development.pt"
)
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts/r3/development/"
    "dagger-ensemble-contextual-safety-b691-20260729-e"
)
REPORT_FORMAT = "irisu-r3e-dagger-ensemble-development-run-v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _log(message: str) -> None:
    print(f"[dagger-ensemble] {message}", flush=True)


@dataclass(frozen=True, slots=True)
class Cost:
    wall_seconds: float
    cpu_seconds: float

    def manifest(self) -> dict[str, float]:
        return asdict(self)


def _cost(started: tuple[float, float]) -> Cost:
    return Cost(time.perf_counter() - started[0], time.process_time() - started[1])


def _start_cost() -> tuple[float, float]:
    return time.perf_counter(), time.process_time()


def _development_root(path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"refusing to reuse campaign namespace: {path}")
    target = path.expanduser().resolve()
    allowed = (ROOT / "artifacts/r3/development").resolve()
    if allowed not in target.parents or not any(
        "dagger-ensemble" in part for part in target.parts
    ):
        raise ValueError(
            "output root must be a fresh artifacts/r3/development namespace "
            "containing dagger-ensemble"
        )
    r3d._reject_path(target, "DAgger ensemble output root")
    target.mkdir(parents=True)
    return target


def _load_config(snapshot: Any) -> dict[str, Any]:
    value = tomllib.loads(snapshot.path.read_text(encoding="utf-8"))
    required = {
        "version": "r3e-dagger-ensemble-v1",
        "status": "development_only_not_canonical_evidence",
        "deployable": False,
        "canonical_r3_evidence": False,
        "sealed_evaluation_allowed": False,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("campaign config weakens development-only boundaries")
    seeds = tuple(int(item) for item in value["training"]["ensemble_member_seeds"])
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ValueError("campaign requires at least three distinct ensemble seeds")
    return value


def _source_identity(config_path: Path) -> dict[str, object]:
    files = (
        Path(__file__).resolve(),
        Path(r3d.__file__).resolve(),
        Path(r3e.__file__).resolve(),
        config_path,
        *sorted((ROOT / "python/irisu_pointer").glob("*.py")),
        *sorted((ROOT / "python/irisu_env").glob("*.py")),
        ROOT / "python/irisu_rl/actions.py",
        ROOT / "python/irisu_rl/encoding.py",
        ROOT / "python/irisu_rl/schema.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    )
    manifest = {
        "schema": "irisu-r3e-dagger-ensemble-source-v1",
        "git_revision": r3d._source_revision(),
        "files": {
            str(path.relative_to(ROOT)): r3d._file_sha256(path) for path in files
        },
    }
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _search_config(value: Mapping[str, Any]) -> GeometrySearchConfig:
    return GeometrySearchConfig(
        horizon_ticks=int(value["horizon_ticks"]),
        velocity_lead_ticks=float(value["velocity_lead_ticks"]),
        ticks_per_second=float(value["ticks_per_second"]),
        support_fractions=tuple(float(item) for item in value["support_fractions"]),
        support_clearance_sizes=float(value["support_clearance_sizes"]),
        grid_x_fractions=tuple(float(item) for item in value["grid_x_fractions"]),
        grid_y_sizes=tuple(float(item) for item in value["grid_y_sizes"]),
        max_candidates=int(value["max_candidates"]),
    )


def _base_options(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config["base_policy"])


def _policy_config(value: Mapping[str, Any]) -> GeometryPolicyConfig:
    return GeometryPolicyConfig(
        minimum_confidence=float(value["minimum_confidence"]),
        minimum_logit_margin=float(value["minimum_logit_margin"]),
        minimum_ensemble_members=int(value["minimum_ensemble_members"]),
        minimum_ensemble_agreement=float(value["minimum_ensemble_agreement"]),
        minimum_member_incumbent_logit_margin=float(
            value["minimum_member_incumbent_logit_margin"]
        ),
        minimum_gauge_fraction=float(value["minimum_gauge_fraction"]),
        maximum_unverified_corrections=int(
            value["maximum_unverified_corrections"]
        ),
    )


def _teacher(
    config: Mapping[str, Any], geometry: GeometrySearchConfig
) -> RunwayGeometrySearch:
    if (
        config["teacher"].get("cross_spawn_boundaries") is not True
        or config["teacher"].get("all_available_candidate_outcomes") is not True
    ):
        raise ValueError("campaign requires complete spawn-crossing runway labels")
    return RunwayGeometrySearch(
        config=RunwaySearchConfig(
            runway_ticks=int(config["teacher"]["runway_ticks"]),
            candidate_config=geometry,
        )
    )


def _suite(label: str, seeds: Sequence[int]) -> dict[str, object]:
    manifest = {"label": label, "seeds": [int(seed) for seed in seeds]}
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _dataset_coverage(dataset: GeometryRankingDataset) -> dict[str, object]:
    available = [int(example.available_mask.sum()) for example in dataset]
    ticks = [int(example.observation.source_tick[0]) for example in dataset]
    winners = Counter(str(example.winner_index) for example in dataset)
    return {
        "dataset_sha256": dataset.sha256,
        "examples": len(dataset),
        "preferences": sum(len(example.preferences) for example in dataset),
        "exact_branch_outcomes": sum(
            len(example.outcome_sha256s) for example in dataset
        ),
        "strict_improvements": sum(
            example.improved_over_incumbent for example in dataset
        ),
        "incumbent_labels": sum(
            not example.improved_over_incumbent for example in dataset
        ),
        "candidate_count": dataset.candidate_count,
        "candidate_vocabulary_sha256": dataset.candidate_vocabulary_sha256,
        "unique_pair_candidate_sets": len(
            {example.candidate_set_sha256 for example in dataset}
        ),
        "available_candidates": r3d._distribution(available),
        "winner_slot_counts": dict(sorted(winners.items(), key=lambda item: int(item[0]))),
        "source_tick": r3d._distribution(ticks),
    }


def _state_label_identity(example: Any) -> str:
    encoded = example.observation
    digest = hashlib.sha256()
    for value in (
        encoded.global_features,
        encoded.body_features,
        encoded.body_mask,
    ):
        digest.update(str(value.dtype).encode())
        digest.update(_canonical_bytes(list(value.shape)))
        digest.update(value.tobytes(order="C"))
    digest.update(
        _canonical_bytes(
            {
                "source_index": example.source_index,
                "destination_index": example.destination_index,
                "candidate_set_sha256": example.candidate_set_sha256,
                "available_mask": example.available_mask.astype(int).tolist(),
                "winner_index": example.winner_index,
                "preferences": [list(value) for value in example.preferences],
            }
        )
    )
    return digest.hexdigest()


def _permutation_audit(
    selectors: Sequence[GeometrySelectorModel],
    ensemble: GeometrySelectorEnsemble,
    dataset: GeometryRankingDataset,
    *,
    seed: int,
    tolerance: float = 1e-5,
    trials: int = 3,
) -> dict[str, object]:
    count = min(64, len(dataset))
    batch = dataset.batch(tuple(range(count))).to("cpu")
    width = batch.body_features.shape[1]
    original_args = (
        batch.global_features,
        batch.body_features,
        batch.body_mask,
        batch.source_index,
        batch.destination_index,
    )
    selectors_to_audit = (
        *(
            (f"member_{index}", member)
            for index, member in enumerate(selectors)
        ),
        ("ensemble", ensemble),
    )
    originals: dict[str, Any] = {}
    for label, selector in selectors_to_audit:
        selector.eval()
        with torch.no_grad():
            originals[label] = selector(*original_args)
    maximums = Counter[str]()
    mismatch_counts = Counter[str]()
    permutations: list[list[int]] = []
    for trial in range(trials):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + trial)
        permutation = torch.randperm(width, generator=generator)
        permutations.append(permutation.tolist())
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(width)
        permuted_args = (
            batch.global_features,
            batch.body_features[:, permutation],
            batch.body_mask[:, permutation],
            inverse[batch.source_index],
            inverse[batch.destination_index],
        )
        for label, selector in selectors_to_audit:
            with torch.no_grad():
                after = selector(*permuted_args)
            before = originals[label]
            maximums[label] = max(
                float(maximums[label]),
                float((before - after).abs().max()),
            )
            before_slots = before.masked_fill(~batch.available_mask, -torch.inf)
            after_slots = after.masked_fill(~batch.available_mask, -torch.inf)
            mismatch_counts[label] += int(
                (before_slots.argmax(dim=-1) != after_slots.argmax(dim=-1)).sum()
            )
    reports = [
        (
            {
                "selector": label,
                "examples": count,
                "permutation_trials": trials,
                "maximum_absolute_logit_difference": float(maximums[label]),
                "available_argmax_mismatches": int(mismatch_counts[label]),
                "passed": (
                    float(maximums[label]) <= tolerance
                    and int(mismatch_counts[label]) == 0
                ),
            }
        )
        for label, _selector in selectors_to_audit
    ]
    return {
        "body_row_permutations": permutations,
        "source_destination_indices_remapped": True,
        "fixed_candidate_slots_permuted": False,
        "tolerance": tolerance,
        "reports": reports,
        "passed": all(bool(report["passed"]) for report in reports),
    }


@dataclass(frozen=True, slots=True)
class EnsembleArtifacts:
    iteration: int
    budget: int
    paths: tuple[Path, ...]
    sha256s: tuple[str, ...]
    ensemble_sha256: str
    dataset_sha256: str

    def manifest(self) -> dict[str, object]:
        return {
            "iteration": self.iteration,
            "training_steps": self.budget,
            "member_paths": [str(path) for path in self.paths],
            "member_sha256s": list(self.sha256s),
            "ensemble_sha256": self.ensemble_sha256,
            "dataset_sha256": self.dataset_sha256,
        }


def _train_plateau(
    dataset: GeometryRankingDataset,
    *,
    iteration: int,
    output_root: Path,
    config: Mapping[str, Any],
    geometry: GeometrySearchConfig,
    policy: GeometryPolicyConfig,
    base_sha256: str,
    source_sha256: str,
) -> tuple[list[dict[str, object]], EnsembleArtifacts]:
    training = config["training"]
    budgets = tuple(int(value) for value in training["plateau_steps"])
    seeds = tuple(int(value) for value in training["ensemble_member_seeds"])
    model_config = GeometryModelConfig(**config["model"])
    curve: list[dict[str, object]] = []
    selected: EnsembleArtifacts | None = None
    for budget in budgets:
        started = _start_cost()
        models: list[GeometrySelectorModel] = []
        paths: list[Path] = []
        identities: list[str] = []
        members: list[dict[str, object]] = []
        for ordinal, seed in enumerate(seeds):
            torch.manual_seed(seed)
            model = GeometrySelectorModel(
                dataset.schema,
                candidate_count=dataset.candidate_count,
                candidate_set_sha256=dataset.candidate_vocabulary_sha256,
                config=model_config,
            )
            report = train_geometry_ranker(
                model,
                dataset,
                steps=budget,
                batch_size=int(training["batch_size"]),
                learning_rate=float(training["learning_rate"]),
                listwise_weight=float(training["listwise_weight"]),
                pairwise_weight=float(training["pairwise_weight"]),
                seed=seed,
            )
            model.eval()
            path = r3d._output_path(
                output_root
                / f"iteration-{iteration}"
                / f"steps-{budget}"
                / f"member-{ordinal}.pt",
                "DAgger ensemble member checkpoint",
                ".pt",
            )
            identity = save_geometry_checkpoint(
                path,
                model,
                geometry_config=geometry,
                policy_config=policy,
                base_policy_checkpoint_sha256=base_sha256,
                source_identity=source_sha256,
                metadata={
                    "development_only": True,
                    "canonical_r3_evidence": False,
                    "sealed_test_material_used": False,
                    "learner": "contextual_all_candidate_ranker",
                    "iteration": iteration,
                    "training_steps": budget,
                    "member_ordinal": ordinal,
                    "training_seed": seed,
                    "ranking_dataset_sha256": dataset.sha256,
                },
            )
            verified = load_geometry_checkpoint(
                path,
                expected_sha256=identity,
                expected_base_policy_checkpoint_sha256=base_sha256,
                expected_source_identity=source_sha256,
            )
            models.append(verified.model)
            paths.append(path)
            identities.append(identity)
            members.append(
                {
                    "ordinal": ordinal,
                    "seed": seed,
                    "checkpoint": {"path": str(path), "sha256": identity},
                    "architecture_sha256": model.architecture_sha256,
                    "training": asdict(report),
                }
            )
        ensemble = GeometrySelectorEnsemble(models, artifact_sha256s=identities)
        full = dataset.batch().to("cpu")
        with torch.no_grad():
            loss = geometry_ranking_loss(
                ensemble,  # type: ignore[arg-type]
                full,
                listwise_weight=float(training["listwise_weight"]),
                pairwise_weight=float(training["pairwise_weight"]),
            )
            logits = ensemble(
                full.global_features,
                full.body_features,
                full.body_mask,
                full.source_index,
                full.destination_index,
            ).masked_fill(~full.available_mask, -torch.inf)
            top1 = float((logits.argmax(dim=-1) == full.winner_index).float().mean())
        permutation = _permutation_audit(
            models, ensemble, dataset, seed=seeds[0] ^ budget ^ iteration
        )
        if not permutation["passed"]:
            raise RuntimeError("whole-board permutation audit failed")
        point = {
            "iteration": iteration,
            "training_steps": budget,
            "dataset_sha256": dataset.sha256,
            "members": members,
            "ensemble": {
                "sha256": ensemble.sha256,
                "manifest": ensemble.manifest(),
                "top1_accuracy": top1,
                "loss": {
                    "total": float(loss.total),
                    "listwise": float(loss.listwise),
                    "pairwise": float(loss.pairwise),
                    "pairwise_accuracy": float(loss.pairwise_accuracy),
                },
            },
            "permutation_audit": permutation,
            "cost": _cost(started).manifest(),
        }
        curve.append(point)
        selected = EnsembleArtifacts(
            iteration,
            budget,
            tuple(paths),
            tuple(identities),
            ensemble.sha256,
            dataset.sha256,
        )
        _log(
            f"trained iteration={iteration} steps={budget} "
            f"dataset={dataset.sha256} ensemble={ensemble.sha256}"
        )
    assert selected is not None
    return curve, selected


def _save_ranking_dataset(
    path: Path,
    dataset: GeometryRankingDataset,
    *,
    metadata: Mapping[str, object],
) -> dict[str, object]:
    target = r3d._output_path(path, "DAgger ranking dataset", ".pt")
    identity = r3e.save_geometry_ranking_collection(
        target, dataset, metadata=metadata
    )
    return {
        "path": str(target),
        "sha256": identity,
        **_dataset_coverage(dataset),
    }


def _bootstrap_collection(
    *,
    config: Mapping[str, Any],
    runtime_path: Path,
    runtime_sha256: str,
    base_path: Path,
    base_sha256: str,
    teacher: RunwayGeometrySearch,
    seeds: Sequence[int],
    source_sha256: str,
) -> tuple[GeometryRankingDataset, dict[str, object]]:
    values = config["collection"]
    started = _start_cost()
    run = r3e.collect_geometry_labels(
        library_path=runtime_path,
        base_policy_path=base_path,
        base_policy_sha256=base_sha256,
        base_options=_base_options(config),
        teacher=teacher,
        seeds=seeds,
        episode_ticks=int(values["bootstrap_ticks"]),
        query_stride_shots=int(values["bootstrap_query_stride_shots"]),
        maximum_search_queries_per_episode=int(
            values["bootstrap_maximum_queries_per_episode"]
        ),
        source_identity_sha256=source_sha256,
        rollout_mode="learner_visited",
    )
    return run.ranking_dataset, {
        "kind": "frozen_v5_visited_exact_all_branch",
        "seeds": [int(seed) for seed in seeds],
        "episode_ticks": int(values["bootstrap_ticks"]),
        "episodes": [episode.manifest() for episode in run.episodes],
        "aggregate": r3d._aggregate(run.episodes),
        "query_report": dict(run.query_report),
        "runner": dict(run.runner),
        "runtime_sha256": runtime_sha256,
        "coverage": _dataset_coverage(run.ranking_dataset),
        "cost": _cost(started).manifest(),
    }


def _ensemble_policy(
    artifacts: EnsembleArtifacts,
    *,
    base_path: Path,
    base_sha256: str,
    base_options: Mapping[str, Any],
    source_sha256: str,
) -> Any:
    base = r3e._base_policy(base_path, base_sha256, base_options)
    return load_safeguarded_geometry_ensemble_policy(
        artifacts.paths,
        base_policy=base,
        expected_sha256s=artifacts.sha256s,
        expected_base_policy_checkpoint_sha256=base_sha256,
        expected_source_identity=source_sha256,
    )


def _dagger_collection(
    *,
    iteration: int,
    config: Mapping[str, Any],
    runtime_path: Path,
    runtime_sha256: str,
    base_path: Path,
    base_sha256: str,
    teacher: RunwayGeometrySearch,
    student_artifacts: EnsembleArtifacts,
    seeds: Sequence[int],
    source_sha256: str,
) -> tuple[GeometryRankingDataset, dict[str, object]]:
    values = config["collection"]
    dagger_values = config["dagger"]
    started = _start_cost()
    examples: list[Any] = []
    episodes: list[Any] = []
    per_seed: list[dict[str, object]] = []
    queries: list[dict[str, object]] = []
    totals = Counter[str]()
    nested = Counter[str]()
    reasons = Counter[str]()
    selected = Counter[str]()
    with IrisuEnv(
        library_path=runtime_path,
        physics_backend="portable",
        config={"max_episode_ticks": int(values["dagger_ticks"])},
    ) as env:
        if Path(env.library_path).resolve() != runtime_path:
            raise RuntimeError("DAgger collection loaded a foreign runtime")
        runner = env.runner_identity_manifest()
        config_hash = int(runner["config_hash"])
        for seed in seeds:
            student = _ensemble_policy(
                student_artifacts,
                base_path=base_path,
                base_sha256=base_sha256,
                base_options=_base_options(config),
                source_sha256=source_sha256,
            )
            dagger = LearnerVisitedGeometryDaggerPolicy(
                env=env,
                base_policy=student.base_policy,
                student_policy=student,
                teacher=teacher,
                source_identity=source_sha256,
                runtime_sha256=runtime_sha256,
                config=GeometryDaggerConfig(
                    execution_mode=str(dagger_values["execution_mode"]),
                    cadence_shots=int(dagger_values["cadence_shots"]),
                    low_gauge_fraction=float(
                        dagger_values["low_gauge_fraction"]
                    ),
                    query_on_rejection=bool(
                        dagger_values["query_on_rejection"]
                    ),
                    query_on_disagreement=bool(
                        dagger_values["query_on_disagreement"]
                    ),
                    disagreement_below=float(
                        dagger_values["disagreement_below"]
                    ),
                    maximum_queries=int(
                        values["dagger_maximum_queries_per_episode"]
                    ),
                    episode_ticks=int(values["dagger_ticks"]),
                ),
            )
            episode = r3d._run_episode(
                env,
                dagger,
                label=f"r3e_dagger_iteration_{iteration}",
                seed=int(seed),
                config_hash=config_hash,
            )
            statistics = dagger.statistics()
            student_statistics = student.statistics()
            examples.extend(dagger.ranking_examples)
            episodes.append(episode)
            queries.extend(record.manifest() for record in dagger.query_records)
            for key, value in statistics.items():
                if isinstance(value, int):
                    totals[key] += value
            reasons.update(statistics["query_reason_counts"])
            selected.update(statistics["selected_candidate_counts"])
            nested.update(student_statistics)
            per_seed.append(
                {
                    "seed": int(seed),
                    "episode": episode.manifest(),
                    "dagger_counts": statistics,
                    "student_gate_counts": student_statistics,
                }
            )
    if not examples:
        raise RuntimeError("learner-visited DAgger produced no exact labels")
    dataset = GeometryRankingDataset(examples)
    query_summary = {
        "teacher_nonincumbent_winners": sum(
            int(record["winner_slot"]) != 0 for record in queries
        ),
        "student_learned_geometry_at_queries": sum(
            bool(record["student_used_learned_geometry"]) for record in queries
        ),
        "teacher_winner_differed_from_student_deployment": sum(
            record["winner_slot"] != record["student_deployed_slot"]
            for record in queries
        ),
        "teacher_strict_improvements": sum(
            bool(record["strictly_improved"]) for record in queries
        ),
    }
    return dataset, {
        "kind": "learner_visited_safety_dagger_exact_all_branch",
        "iteration": iteration,
        "student_ensemble": student_artifacts.manifest(),
        "seeds": [int(seed) for seed in seeds],
        "episode_ticks": int(values["dagger_ticks"]),
        "episodes": [episode.manifest() for episode in episodes],
        "per_seed": per_seed,
        "aggregate": r3d._aggregate(episodes),
        "dagger_counts": {
            **dict(sorted(totals.items())),
            "query_reason_counts": dict(sorted(reasons.items())),
            "selected_candidate_counts": dict(sorted(selected.items())),
        },
        "student_gate_counts": dict(sorted(nested.items())),
        "query_records": queries,
        "teacher_student_query_summary": query_summary,
        "runner": dict(runner),
        "coverage": _dataset_coverage(dataset),
        "cost": _cost(started).manifest(),
    }


def _extended_aggregate(episodes: Sequence[Any]) -> dict[str, object]:
    aggregate = r3d._aggregate(episodes)
    return {
        **aggregate,
        "final_gauge": r3d._distribution([value.final_gauge for value in episodes]),
        "final_level": r3d._distribution([value.final_level for value in episodes]),
        "decisions": sum(value.decisions for value in episodes),
        "primitive_actions": sum(value.primitive_actions for value in episodes),
        "full_horizon_survivors": sum(
            value.conversion.truncated and not value.conversion.terminated
            for value in episodes
        ),
    }


def _evaluate(
    *,
    label: str,
    runtime_path: Path,
    seeds: Sequence[int],
    horizon: int,
    factory: Callable[[], Any],
) -> dict[str, object]:
    started = _start_cost()
    episodes: list[Any] = []
    per_seed: list[dict[str, object]] = []
    totals = Counter[str]()
    with IrisuEnv(
        library_path=runtime_path,
        physics_backend="portable",
        config={"max_episode_ticks": horizon},
    ) as env:
        if Path(env.library_path).resolve() != runtime_path:
            raise RuntimeError("paired evaluation loaded a foreign runtime")
        runner = env.runner_identity_manifest()
        config_hash = int(runner["config_hash"])
        for seed in seeds:
            policy = factory()
            episode = r3d._run_episode(
                env,
                policy,
                label=label,
                seed=int(seed),
                config_hash=config_hash,
            )
            statistics = (
                dict(policy.statistics())
                if callable(getattr(policy, "statistics", None))
                else {}
            )
            totals.update(
                {
                    key: int(value)
                    for key, value in statistics.items()
                    if isinstance(value, int)
                }
            )
            episodes.append(episode)
            per_seed.append(
                {
                    "seed": int(seed),
                    "episode": episode.manifest(),
                    "policy_counts": statistics,
                }
            )
    return {
        "label": label,
        "horizon_ticks": horizon,
        "seeds": [int(seed) for seed in seeds],
        "episodes": [episode.manifest() for episode in episodes],
        "per_seed": per_seed,
        "aggregate": _extended_aggregate(episodes),
        "policy_counts": dict(sorted(totals.items())),
        "runner": dict(runner),
        "cost": _cost(started).manifest(),
    }


def _paired_audit(
    baseline: Mapping[str, Any],
    learner: Mapping[str, Any],
    *,
    horizon: int,
    safety: Mapping[str, Any],
) -> dict[str, object]:
    base = {int(item["seed"]): item for item in baseline["per_seed"]}
    candidate = {int(item["seed"]): item for item in learner["per_seed"]}
    if base.keys() != candidate.keys():
        raise ValueError("paired evaluation seed sets differ")
    pairs: list[dict[str, object]] = []
    score_deltas: list[int] = []
    catastrophic: list[int] = []
    new_gauge_failures: list[int] = []
    for seed in base:
        left = base[seed]["episode"]
        right = candidate[seed]["episode"]
        left_conversion = left["conversion"]
        right_conversion = right["conversion"]
        base_survival = int(left_conversion["survival_ticks"])
        learner_survival = int(right_conversion["survival_ticks"])
        survival_loss = base_survival - learner_survival
        reasons: list[str] = []
        if (
            base_survival >= horizon
            and learner_survival
            < float(safety["full_horizon_minimum_retention"]) * horizon
        ):
            reasons.append("v5_full_horizon_learner_below_retention")
        if (
            survival_loss
            >= int(safety["catastrophic_absolute_loss_ticks"])
            and learner_survival
            < float(safety["catastrophic_ratio"]) * base_survival
        ):
            reasons.append("large_absolute_and_fractional_survival_loss")
        new_gauge = bool(right["gauge_failure"]) and not bool(
            left["gauge_failure"]
        )
        if reasons:
            catastrophic.append(seed)
        if new_gauge:
            new_gauge_failures.append(seed)
        score_delta = int(right_conversion["final_score"]) - int(
            left_conversion["final_score"]
        )
        score_deltas.append(score_delta)
        pairs.append(
            {
                "seed": seed,
                "base_v5": left,
                "learner": right,
                "learner_policy_counts": candidate[seed]["policy_counts"],
                "score_delta": score_delta,
                "survival_delta": learner_survival - base_survival,
                "survival_retention": (
                    learner_survival / base_survival if base_survival else 1.0
                ),
                "new_gauge_failure": new_gauge,
                "catastrophic_survival_regression": bool(reasons),
                "catastrophic_reasons": reasons,
            }
        )
    base_aggregate = baseline["aggregate"]
    learner_aggregate = learner["aggregate"]
    base_survival = base_aggregate["survival_ticks"]
    learner_survival = learner_aggregate["survival_ticks"]
    base_full = int(base_aggregate["full_horizon_survivors"])
    learner_full = int(learner_aggregate["full_horizon_survivors"])
    invalid = int(learner_aggregate["conversion"]["invalid_actions"])
    retention = float(safety["minimum_survival_retention"])
    gates = {
        "no_catastrophic_pairs": not catastrophic,
        "no_new_gauge_failures": (
            not new_gauge_failures
            if safety["reject_new_gauge_failures"]
            else True
        ),
        "p10_survival_retention": float(learner_survival["p10"])
        >= retention * float(base_survival["p10"]),
        "median_survival_retention": float(learner_survival["median"])
        >= retention * float(base_survival["median"]),
        "non_decreasing_full_survivors": (
            learner_full >= base_full
            if safety["require_non_decreasing_full_survivors"]
            else True
        ),
        "zero_invalid_actions": (
            invalid == 0 if safety["require_zero_invalid_actions"] else True
        ),
    }
    wins = sum(delta > 0 for delta in score_deltas)
    losses = sum(delta < 0 for delta in score_deltas)
    deployments = sum(
        int(item["learner_policy_counts"].get("learned_geometry_deployments", 0))
        > 0
        for item in pairs
    )
    median_delta = r3d._percentile(score_deltas, 0.5)
    return {
        "horizon_ticks": horizon,
        "pairs": pairs,
        "catastrophic_seed_list": catastrophic,
        "new_gauge_failure_seed_list": new_gauge_failures,
        "aggregate_gates": gates,
        "safety_passed": all(gates.values()),
        "score": {
            "paired_deltas": score_deltas,
            "median_paired_delta": median_delta,
            "wins": wins,
            "ties": len(score_deltas) - wins - losses,
            "losses": losses,
            "difference_of_medians": (
                float(learner_aggregate["raw_score"]["median"])
                - float(base_aggregate["raw_score"]["median"])
            ),
        },
        "survival": {
            "base_full_survivors": base_full,
            "learner_full_survivors": learner_full,
            "base_p10": float(base_survival["p10"]),
            "learner_p10": float(learner_survival["p10"]),
            "base_median": float(base_survival["median"]),
            "learner_median": float(learner_survival["median"]),
        },
        "seeds_with_learned_deployments": deployments,
        "strict_score_win": median_delta > 0 and wins > losses and deployments >= 2,
    }


def _candidate_key(
    evaluation: Mapping[str, Any], audit: Mapping[str, Any], iteration: int
) -> tuple[float, ...]:
    aggregate = evaluation["aggregate"]
    return (
        float(audit["score"]["median_paired_delta"]),
        float(aggregate["raw_score"]["median"]),
        float(aggregate["survival_ticks"]["p10"]),
        float(aggregate["survival_ticks"]["median"]),
        float(iteration),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--library", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--base-policy", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify-inputs-only",
        action="store_true",
        help="validate development boundaries and trusted identities, then exit",
    )
    args = parser.parse_args()

    campaign_started = _start_cost()
    config_snapshot = r3d._snapshot_file(args.config, "DAgger ensemble config")
    config = _load_config(config_snapshot)
    runtime_snapshot = r3d._snapshot_file(
        args.library, "trusted R3e portable runtime"
    )
    base_snapshot = r3d._snapshot_file(args.base_policy, "frozen R3e v5 policy")
    trusted = config["trusted_inputs"]
    if runtime_snapshot.sha256 != trusted["portable_sha256"]:
        parser.error("portable runtime SHA-256 is not the trusted R3e identity")
    if base_snapshot.sha256 != trusted["frozen_v5_sha256"]:
        parser.error("base checkpoint SHA-256 is not frozen R3d v5")
    source_identity = _source_identity(config_snapshot.path)
    geometry = _search_config(config["search"])
    teacher = _teacher(config, geometry)
    policy_config = _policy_config(config["deployment"])
    if policy_config.minimum_ensemble_members < 3:
        raise ValueError("campaign deployment must require at least three members")
    torch_threads = int(config["training"]["torch_threads"])
    torch.set_num_threads(torch_threads)
    _log(
        "trusted identities "
        f"runtime={runtime_snapshot.sha256} base={base_snapshot.sha256} "
        f"source={source_identity['sha256']} teacher={teacher.sha256}"
    )
    if args.verify_inputs_only:
        _log("input verification complete; no campaign artifacts written")
        return

    output_root = _development_root(args.output_root)
    collection_values = config["collection"]
    bootstrap_start = int(collection_values["bootstrap_seed_start"])
    bootstrap_count = int(collection_values["bootstrap_seeds"])
    bootstrap_seeds = r3e.COLLECTION_SEEDS[
        bootstrap_start : bootstrap_start + bootstrap_count
    ]
    dagger_starts = tuple(
        int(value) for value in collection_values["dagger_seed_starts"]
    )
    dagger_count = int(collection_values["dagger_seeds_per_iteration"])
    dagger_seed_waves = tuple(
        r3e.COLLECTION_SEEDS[start : start + dagger_count]
        for start in dagger_starts
    )
    selection_values = config["selection_evaluation"]
    selection_start = int(selection_values["seed_start"])
    selection_count = int(selection_values["seeds"])
    selection_seeds = r3e.DEVELOPMENT_SEEDS[
        selection_start : selection_start + selection_count
    ]
    final_values = config["final_evaluation"]
    final_start = int(final_values["seed_start"])
    final_count = int(final_values["seeds"])
    final_seeds = r3e.DEVELOPMENT_SEEDS[final_start : final_start + final_count]
    all_training_seeds = set(bootstrap_seeds).union(*map(set, dagger_seed_waves))
    all_evaluation_seeds = set(selection_seeds) | set(final_seeds)
    if (
        len(bootstrap_seeds) != bootstrap_count
        or any(len(wave) != dagger_count for wave in dagger_seed_waves)
        or len(selection_seeds) != selection_count
        or len(final_seeds) != final_count
        or all_training_seeds & all_evaluation_seeds
        or set(selection_seeds) & set(final_seeds)
        or any(
            set(left) & set(right)
            for index, left in enumerate((bootstrap_seeds, *dagger_seed_waves))
            for right in (bootstrap_seeds, *dagger_seed_waves)[index + 1 :]
        )
    ):
        raise ValueError("campaign seed partitions are not complete and disjoint")

    bootstrap, bootstrap_report = _bootstrap_collection(
        config=config,
        runtime_path=runtime_snapshot.path,
        runtime_sha256=runtime_snapshot.sha256,
        base_path=base_snapshot.path,
        base_sha256=base_snapshot.sha256,
        teacher=teacher,
        seeds=bootstrap_seeds,
        source_sha256=str(source_identity["sha256"]),
    )
    bootstrap_artifact = _save_ranking_dataset(
        output_root / "iteration-0" / "ranking-dataset.pt",
        bootstrap,
        metadata={
            "development_only": True,
            "canonical_r3_evidence": False,
            "sealed_test_material_used": False,
            "iteration": 0,
            "source_identity_sha256": source_identity["sha256"],
            "runtime_sha256": runtime_snapshot.sha256,
            "base_policy_sha256": base_snapshot.sha256,
            "teacher_sha256": teacher.sha256,
            "collection_seeds": list(bootstrap_seeds),
        },
    )
    _log(
        f"bootstrap examples={len(bootstrap)} dataset={bootstrap.sha256} "
        f"artifact={bootstrap_artifact['sha256']}"
    )

    datasets: list[GeometryRankingDataset] = [bootstrap]
    collection_reports: list[dict[str, object]] = [
        {**bootstrap_report, "artifact": bootstrap_artifact}
    ]
    iteration_reports: list[dict[str, object]] = []
    curves, ensemble = _train_plateau(
        bootstrap,
        iteration=0,
        output_root=output_root,
        config=config,
        geometry=geometry,
        policy=policy_config,
        base_sha256=base_snapshot.sha256,
        source_sha256=str(source_identity["sha256"]),
    )
    iteration_reports.append(
        {
            "iteration": 0,
            "input_datasets": [bootstrap.sha256],
            "merged_dataset": _dataset_coverage(bootstrap),
            "training_plateau": curves,
            "selected_ensemble": ensemble.manifest(),
        }
    )
    ensembles = [ensemble]
    merged = bootstrap
    continuation_decisions: list[dict[str, object]] = []
    executed_dagger_waves: list[tuple[int, ...]] = []

    for iteration, seeds in enumerate(dagger_seed_waves, start=1):
        shard, shard_report = _dagger_collection(
            iteration=iteration,
            config=config,
            runtime_path=runtime_snapshot.path,
            runtime_sha256=runtime_snapshot.sha256,
            base_path=base_snapshot.path,
            base_sha256=base_snapshot.sha256,
            teacher=teacher,
            student_artifacts=ensemble,
            seeds=seeds,
            source_sha256=str(source_identity["sha256"]),
        )
        datasets.append(shard)
        executed_dagger_waves.append(tuple(int(seed) for seed in seeds))
        prior = merged
        merged = r3e.merge_geometry_ranking_datasets(datasets)
        duplicates = len(prior) + len(shard) - len(merged)
        new_examples = len(merged) - len(prior)
        prior_state_labels = {_state_label_identity(value) for value in prior}
        shard_state_labels = {_state_label_identity(value) for value in shard}
        new_state_labels = len(shard_state_labels - prior_state_labels)
        shard_artifact = _save_ranking_dataset(
            output_root / f"iteration-{iteration}" / "dagger-shard.pt",
            shard,
            metadata={
                "development_only": True,
                "canonical_r3_evidence": False,
                "sealed_test_material_used": False,
                "iteration": iteration,
                "kind": "learner_visited_safety_dagger",
                "source_identity_sha256": source_identity["sha256"],
                "runtime_sha256": runtime_snapshot.sha256,
                "base_policy_sha256": base_snapshot.sha256,
                "teacher_sha256": teacher.sha256,
                "student_ensemble_sha256": ensemble.ensemble_sha256,
                "collection_seeds": list(seeds),
            },
        )
        merged_artifact = _save_ranking_dataset(
            output_root / f"iteration-{iteration}" / "merged-ranking-dataset.pt",
            merged,
            metadata={
                "development_only": True,
                "canonical_r3_evidence": False,
                "sealed_test_material_used": False,
                "iteration": iteration,
                "kind": "merged_base_and_learner_visited",
                "source_identity_sha256": source_identity["sha256"],
                "runtime_sha256": runtime_snapshot.sha256,
                "base_policy_sha256": base_snapshot.sha256,
                "teacher_sha256": teacher.sha256,
                "component_dataset_sha256s": [
                    dataset.sha256 for dataset in datasets
                ],
            },
        )
        shard_report["artifact"] = shard_artifact
        shard_report["merge"] = {
            "prior_dataset_sha256": prior.sha256,
            "shard_dataset_sha256": shard.sha256,
            "merged_dataset_sha256": merged.sha256,
            "input_examples": len(prior) + len(shard),
            "new_artifact_examples": new_examples,
            "new_unique_state_labels": new_state_labels,
            "duplicates_removed": duplicates,
            "conflicts": 0,
            "artifact": merged_artifact,
        }
        collection_reports.append(shard_report)
        _log(
            f"DAgger iteration={iteration} shard={shard.sha256} "
            f"examples={len(shard)} merged={merged.sha256} "
            f"new_state_labels={new_state_labels}"
        )
        curves, ensemble = _train_plateau(
            merged,
            iteration=iteration,
            output_root=output_root,
            config=config,
            geometry=geometry,
            policy=policy_config,
            base_sha256=base_snapshot.sha256,
            source_sha256=str(source_identity["sha256"]),
        )
        ensembles.append(ensemble)
        iteration_reports.append(
            {
                "iteration": iteration,
                "input_datasets": [dataset.sha256 for dataset in datasets],
                "merged_dataset": _dataset_coverage(merged),
                "merge": dict(shard_report["merge"]),
                "training_plateau": curves,
                "selected_ensemble": ensemble.manifest(),
            }
        )
        if iteration < len(dagger_seed_waves):
            minimum = max(
                int(collection_values["minimum_new_examples_for_another_iteration"]),
                math.ceil(
                    float(
                        collection_values[
                            "minimum_new_fraction_for_another_iteration"
                        ]
                    )
                    * len(prior)
                ),
            )
            counts = shard_report["dagger_counts"]
            safety_queries = sum(
                int(counts.get(name, 0))
                for name in (
                    "low_gauge_triggers",
                    "safeguard_rejection_triggers",
                    "ensemble_disagreement_triggers",
                    "strict_improvements",
                )
            )
            warranted = new_state_labels >= minimum and (
                safety_queries > 0 or new_state_labels >= 2 * minimum
            )
            continuation_decisions.append(
                {
                    "after_iteration": iteration,
                    "new_artifact_examples": new_examples,
                    "new_unique_state_labels": new_state_labels,
                    "minimum_required": minimum,
                    "safety_or_improvement_triggers": safety_queries,
                    "another_iteration_warranted": warranted,
                }
            )
            if not warranted:
                _log("second DAgger iteration not warranted by configured evidence")
                break

    selection_horizons = tuple(
        int(value) for value in selection_values["horizons"]
    )
    primary_horizon = int(selection_values["primary_horizon"])
    if primary_horizon not in selection_horizons or max(selection_horizons) < 20_000:
        raise ValueError("selection must include a primary horizon of at least 20k")
    baseline_evaluations: dict[str, object] = {}
    for horizon in selection_horizons:
        baseline_evaluations[str(horizon)] = _evaluate(
            label="frozen_r3d_v5",
            runtime_path=runtime_snapshot.path,
            seeds=selection_seeds,
            horizon=horizon,
            factory=lambda: r3e._base_policy(
                base_snapshot.path,
                base_snapshot.sha256,
                _base_options(config),
            ),
        )
        _log(f"paired baseline evaluation complete horizon={horizon}")

    candidate_evaluations: list[dict[str, object]] = []
    for artifacts in ensembles:
        horizons: dict[str, object] = {}
        audits: dict[str, object] = {}
        for horizon in selection_horizons:
            evaluation = _evaluate(
                label=f"dagger_ensemble_iteration_{artifacts.iteration}",
                runtime_path=runtime_snapshot.path,
                seeds=selection_seeds,
                horizon=horizon,
                factory=lambda value=artifacts: _ensemble_policy(
                    value,
                    base_path=base_snapshot.path,
                    base_sha256=base_snapshot.sha256,
                    base_options=_base_options(config),
                    source_sha256=str(source_identity["sha256"]),
                ),
            )
            audit = _paired_audit(
                baseline_evaluations[str(horizon)],
                evaluation,
                horizon=horizon,
                safety=config["safety"],
            )
            horizons[str(horizon)] = evaluation
            audits[str(horizon)] = audit
            _log(
                f"paired iteration={artifacts.iteration} horizon={horizon} "
                f"safety={audit['safety_passed']} "
                f"catastrophic={audit['catastrophic_seed_list']}"
            )
        candidate_evaluations.append(
            {
                "ensemble": artifacts.manifest(),
                "horizons": horizons,
                "paired_audits": audits,
            }
        )

    safe = [
        item
        for item in candidate_evaluations
        if all(
            item["paired_audits"][str(horizon)]["safety_passed"]
            for horizon in selection_horizons
        )
    ]
    pool = safe if safe else candidate_evaluations
    selected = max(
        pool,
        key=lambda item: _candidate_key(
            item["horizons"][str(primary_horizon)],
            item["paired_audits"][str(primary_horizon)],
            int(item["ensemble"]["iteration"]),
        ),
    )
    selected_iteration = int(selected["ensemble"]["iteration"])
    finalist = next(
        value for value in ensembles if value.iteration == selected_iteration
    )
    selection_eligible = bool(safe)
    _log(
        f"locked finalist iteration={selected_iteration} "
        f"ensemble={finalist.ensemble_sha256} eligible={selection_eligible}"
    )

    final_horizons = tuple(int(value) for value in final_values["horizons"])
    final_primary_horizon = int(final_values["primary_horizon"])
    if (
        final_primary_horizon not in final_horizons
        or max(final_horizons) < 50_000
    ):
        raise ValueError("locked finalist evaluation must reach 50k ticks")
    final_curve: dict[str, object] = {}
    for horizon in final_horizons:
        final_baseline = _evaluate(
            label="frozen_r3d_v5_final",
            runtime_path=runtime_snapshot.path,
            seeds=final_seeds,
            horizon=horizon,
            factory=lambda: r3e._base_policy(
                base_snapshot.path,
                base_snapshot.sha256,
                _base_options(config),
            ),
        )
        final_learner = _evaluate(
            label=f"dagger_ensemble_iteration_{selected_iteration}_final",
            runtime_path=runtime_snapshot.path,
            seeds=final_seeds,
            horizon=horizon,
            factory=lambda: _ensemble_policy(
                finalist,
                base_path=base_snapshot.path,
                base_sha256=base_snapshot.sha256,
                base_options=_base_options(config),
                source_sha256=str(source_identity["sha256"]),
            ),
        )
        final_curve[str(horizon)] = {
            "baseline": final_baseline,
            "learner": final_learner,
            "paired_audit": _paired_audit(
                final_baseline,
                final_learner,
                horizon=horizon,
                safety=config["safety"],
            ),
        }
        _log(
            f"locked finalist paired curve complete horizon={horizon} "
            f"safety={final_curve[str(horizon)]['paired_audit']['safety_passed']}"
        )
    final_audit = final_curve[str(final_primary_horizon)]["paired_audit"]
    final_all_horizons_safe = all(
        value["paired_audit"]["safety_passed"]
        for value in final_curve.values()
    )
    final_catastrophic_by_horizon = {
        horizon: value["paired_audit"]["catastrophic_seed_list"]
        for horizon, value in final_curve.items()
        if value["paired_audit"]["catastrophic_seed_list"]
    }
    final_new_gauge_by_horizon = {
        horizon: value["paired_audit"]["new_gauge_failure_seed_list"]
        for horizon, value in final_curve.items()
        if value["paired_audit"]["new_gauge_failure_seed_list"]
    }
    should_win = bool(
        selection_eligible
        and final_all_horizons_safe
        and final_audit["strict_score_win"]
    )
    _log(
        f"50k finalist complete safety={final_all_horizons_safe} "
        f"catastrophic={final_audit['catastrophic_seed_list']} "
        f"strict_score_win={final_audit['strict_score_win']}"
    )

    current_source = _source_identity(config_snapshot.path)
    if current_source != source_identity:
        raise RuntimeError("campaign source identity changed during execution")
    r3d._require_unchanged(config_snapshot, "DAgger ensemble config")
    r3d._require_unchanged(runtime_snapshot, "trusted R3e portable runtime")
    r3d._require_unchanged(base_snapshot, "frozen R3e v5 policy")
    limitations = [
        (
            "The runway teacher crosses future spawn boundaries on one exact "
            "continuation; hidden-future variation is not estimated."
        ),
        (
            "Ensemble disagreement measures initialization uncertainty, not "
            "uncertainty over future simulator continuations."
        ),
        (
            "Progress-credit replenishment uses correlated public score or "
            "qualifying-clear progress, not per-shot causal attribution."
        ),
        (
            "One unverified correction may suppress useful multi-shot learned "
            "sequences."
        ),
        (
            "Fixed candidate slots are semantic; permutation safety applies to "
            "whole-board body rows with source/destination indices remapped."
        ),
        (
            "Training-budget curves use in-sample ranking metrics; only the "
            "maximum-step ensemble from each iteration receives policy evaluation."
        ),
        (
            "Ranking artifacts retain exact winner/preferences and branch-outcome "
            "hashes, but not the numeric branch-outcome manifests."
        ),
        "All results are development-only and are neither sealed nor canonical.",
    ]
    if should_win:
        recommendation = "win"
    elif not selection_eligible:
        recommendation = "reject_no_selection_candidate_passed_all_safety_gates"
    elif not final_all_horizons_safe:
        recommendation = "reject_for_safety"
    else:
        recommendation = "safe_but_no_strict_paired_score_win"
    report_content: dict[str, object] = {
        "schema": REPORT_FORMAT,
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
        "source_identity": source_identity,
        "config": {
            "path": str(config_snapshot.path),
            "sha256": config_snapshot.sha256,
            "value": config,
        },
        "trusted_inputs": {
            "runtime": {
                "path": str(runtime_snapshot.path),
                "sha256": runtime_snapshot.sha256,
                "backend": "portable",
            },
            "frozen_r3d_v5": {
                "path": str(base_snapshot.path),
                "sha256": base_snapshot.sha256,
            },
        },
        "contract_identities": {
            "schema_sha256": TEACHER_V1.sha256,
            "teacher_sha256": teacher.sha256,
            "teacher_manifest": teacher.identity_manifest(),
            "geometry_config_sha256": geometry.sha256,
            "candidate_vocabulary_sha256": (
                geometry_candidate_vocabulary_sha256(geometry)
            ),
            "action_spec_sha256": teacher.action_spec.sha256,
            "deployment_policy_config_sha256": policy_config.sha256,
        },
        "seed_suites": {
            "bootstrap": _suite(
                str(collection_values["suite"]) + ":bootstrap", bootstrap_seeds
            ),
            "dagger_iterations": [
                _suite(
                    str(collection_values["suite"]) + f":dagger-{index}",
                    wave,
                )
                for index, wave in enumerate(executed_dagger_waves, start=1)
            ],
            "planned_dagger_iterations": [
                _suite(
                    str(collection_values["suite"]) + f":planned-dagger-{index}",
                    wave,
                )
                for index, wave in enumerate(dagger_seed_waves, start=1)
            ],
            "selection": _suite(
                str(selection_values["suite"]) + ":selection", selection_seeds
            ),
            "final": _suite(
                str(final_values["suite"]) + ":final", final_seeds
            ),
            "training_evaluation_disjoint": not bool(
                all_training_seeds & all_evaluation_seeds
            ),
            "selection_final_disjoint": not bool(
                set(selection_seeds) & set(final_seeds)
            ),
        },
        "collection_and_merge": collection_reports,
        "iteration_training": iteration_reports,
        "continuation_decisions": continuation_decisions,
        "tests_and_validation": {
            "unit_test_or_test_asset_access": False,
            "reason": (
                "The delegated protocol forbids entering test material; validation "
                "is development-only and source-local."
            ),
            "python_compile_and_import_preflight": True,
            "trusted_input_identity_preflight": True,
            "permutation_audits": sum(
                len(value["training_plateau"]) for value in iteration_reports
            ),
            "all_permutation_audits_passed": all(
                point["permutation_audit"]["passed"]
                for value in iteration_reports
                for point in value["training_plateau"]
            ),
            "checkpoint_save_reload_verifications": sum(
                len(point["members"])
                for value in iteration_reports
                for point in value["training_plateau"]
            ),
            "transactional_teacher_restore_checks": sum(
                int(
                    value.get("query_report", value.get("dagger_counts", {})).get(
                        "transactional_restore_checks", 0
                    )
                )
                for value in collection_reports
            ),
        },
        "selection_evaluation": {
            "primary_horizon_ticks": primary_horizon,
            "baseline": baseline_evaluations,
            "candidates": candidate_evaluations,
            "safe_candidate_iterations": [
                int(item["ensemble"]["iteration"]) for item in safe
            ],
            "selected_iteration": selected_iteration,
            "selected_ensemble": finalist.manifest(),
            "selection_eligible": selection_eligible,
            "rule": (
                "reject every candidate with any configured-horizon safety failure, "
                "then maximize paired median score delta, score median, survival "
                "p10/median, and iteration"
            ),
        },
        "locked_finalist_evaluation": {
            "primary_horizon_ticks": final_primary_horizon,
            "horizons": final_curve,
            "all_horizons_safety_passed": final_all_horizons_safe,
        },
        "verdict": {
            "should_win": should_win,
            "recommendation": recommendation,
            "selection_eligible": selection_eligible,
            "final_safety_passed": final_all_horizons_safe,
            "strict_paired_score_win": final_audit["strict_score_win"],
            "catastrophic_seeds_by_horizon": final_catastrophic_by_horizon,
            "new_gauge_failures_by_horizon": final_new_gauge_by_horizon,
        },
        "limitations": limitations,
        "execution": {
            "campaign_cost": _cost(campaign_started).manifest(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "pid": os.getpid(),
        },
    }
    report = {
        **report_content,
        "payload_sha256": _canonical_sha256(report_content),
    }
    report_path = r3d._output_path(
        output_root / "evidence-packet.json",
        "DAgger ensemble evidence packet",
        ".json",
    )
    report_sha256 = r3e._atomic_write_json(report_path, report, overwrite=False)
    summary = {
        "report": {"path": str(report_path), "sha256": report_sha256},
        "payload_sha256": report["payload_sha256"],
        "runtime_sha256": runtime_snapshot.sha256,
        "base_policy_sha256": base_snapshot.sha256,
        "source_identity_sha256": source_identity["sha256"],
        "finalist_ensemble_sha256": finalist.ensemble_sha256,
        "finalist_iteration": selected_iteration,
        "should_win": should_win,
        "recommendation": report["verdict"]["recommendation"],
        "catastrophic_seeds_by_horizon": final_catastrophic_by_horizon,
    }
    r3e._atomic_write_json(
        r3d._output_path(
            output_root / "summary.json",
            "DAgger ensemble summary",
            ".json",
        ),
        summary,
        overwrite=False,
    )
    _log(
        f"evidence={report_path} sha256={report_sha256} "
        f"should_win={should_win}"
    )


if __name__ == "__main__":
    main()
