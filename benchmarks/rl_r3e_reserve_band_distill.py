#!/usr/bin/env python3
"""Development-only reserve-band oracle collection and ranker distillation.

Only the exact trusted portable runtime and frozen R3d v5 checkpoint are
accepted.  Collection and evaluation use disjoint SHA-derived development
suites.  No sealed input or canonical R3 run storage is read or written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = Path(__file__).resolve().parent
for import_root in (ROOT / "python", BENCHMARKS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from irisu_env import IrisuEnv  # noqa: E402
from irisu_pointer.geometry_policy import (  # noqa: E402
    geometry_candidate_vocabulary_manifest,
    geometry_candidate_vocabulary_sha256,
)

import rl_r3d_steering as r3d  # noqa: E402
import rl_r3e_reserve_band as oracle  # noqa: E402
import rl_r3e_sustainable as r3e  # noqa: E402


DEFAULT_CONFIG = oracle.DEFAULT_CONFIG
TRUSTED_RUNTIME = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/runtime/"
    "main-0c48dba-20260723/portable-build/libirisu_clone.so"
)
TRUSTED_RUNTIME_SHA256 = (
    "4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79"
)
FROZEN_V5 = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3d-survival-v5-20260729/long-development.pt"
)
FROZEN_V5_SHA256 = (
    "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d"
)
REPORT_SCHEMA = "irisu-r3e-reserve-band-debt-distillation-development-v2"
_RUN_NAME = re.compile(r"reserve-band[a-z0-9._-]{0,96}\Z")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _source_identity(config_path: Path) -> dict[str, object]:
    files = {
        Path(__file__).resolve(),
        Path(oracle.__file__).resolve(),
        Path(r3d.__file__).resolve(),
        Path(r3e.__file__).resolve(),
        config_path,
        *sorted((ROOT / "python/irisu_env").glob("*.py")),
        *sorted((ROOT / "python/irisu_pointer").glob("*.py")),
        *sorted((ROOT / "python/irisu_rl").glob("*.py")),
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    }
    manifest = {
        "schema": "irisu-r3e-reserve-band-distill-source-v1",
        "git_revision": r3d._source_revision(),
        "files": {
            str(path.relative_to(ROOT)): r3d._file_sha256(path)
            for path in sorted(files)
        },
    }
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _require_source_identity(
    expected: Mapping[str, object], config_path: Path
) -> None:
    if _source_identity(config_path) != dict(expected):
        raise RuntimeError("reserve-band distillation source changed during execution")


def _require_exact_inputs(
    config: Mapping[str, Any], parser: argparse.ArgumentParser
) -> tuple[Any, Any]:
    runtime = oracle._path(str(config["trusted_runtime"])).resolve(strict=True)
    base = oracle._path(
        str(config["base_policy"]["checkpoint"])
    ).resolve(strict=True)
    if (
        runtime != TRUSTED_RUNTIME.resolve(strict=True)
        or str(config["trusted_runtime_sha256"]) != TRUSTED_RUNTIME_SHA256
    ):
        parser.error("config is not bound to the exact trusted portable runtime")
    if (
        base != FROZEN_V5.resolve(strict=True)
        or str(config["base_policy"]["sha256"]) != FROZEN_V5_SHA256
    ):
        parser.error("config is not bound to the exact frozen R3d v5 checkpoint")
    runtime_snapshot = r3d._snapshot_file(runtime, "reserve-band portable runtime")
    base_snapshot = r3d._snapshot_file(base, "reserve-band frozen v5")
    if runtime_snapshot.sha256 != TRUSTED_RUNTIME_SHA256:
        parser.error("trusted portable runtime SHA-256 differs")
    if base_snapshot.sha256 != FROZEN_V5_SHA256:
        parser.error("frozen R3d v5 SHA-256 differs")
    return runtime_snapshot, base_snapshot


def _base_identity(
    base_path: Path,
    base_sha256: str,
    options: Mapping[str, Any],
) -> dict[str, object]:
    policy = r3e._base_policy(base_path, base_sha256, options)
    manifest = {
        "type": "frozen-r3d-v5-analytic-geometry",
        "checkpoint_sha256": base_sha256,
        "options": dict(options),
        "schema_sha256": str(policy.schema_sha256),
        "pointer_action_sha256": str(policy.pointer_action_sha256),
        "model": policy.model.manifest(),
    }
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _safe_run_directory(
    config: Mapping[str, Any],
    run_name: str | None,
    *,
    smoke: bool,
) -> Path:
    if run_name is None:
        stamp = time.strftime("%Y%m%dt%H%M%Sz", time.gmtime())
        suffix = f"{time.time_ns() % 1_000_000_000:09d}"
        run_name = (
            f"reserve-band-distill-{'smoke-' if smoke else ''}{stamp}-{suffix}"
        )
    if _RUN_NAME.fullmatch(run_name) is None:
        raise ValueError("run name must be a safe reserve-band slug")
    namespace = oracle._path(str(config["artifact_namespace"])).resolve()
    development = (ROOT / "artifacts/r3/development").resolve()
    if (
        development not in namespace.parents
        or "reserve-band" not in namespace.name.lower()
    ):
        raise ValueError("configured output namespace is not reserve-band development")
    target = (namespace / run_name).resolve()
    if namespace not in target.parents:
        raise ValueError("distillation run escaped its reserve-band namespace")
    r3d._reject_path(target, "reserve-band distillation output")
    target.mkdir(parents=True, exist_ok=False)
    return target


def _record(
    episode: Any,
    statistics: Mapping[str, object],
    *,
    wall_seconds: float,
    cpu_seconds: float,
) -> dict[str, object]:
    content = {
        **episode.manifest(),
        "policy_statistics": dict(sorted(statistics.items())),
        "cost": {
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
        },
    }
    return {**content, "payload_sha256": _canonical_sha256(content)}


def _aggregate(
    episodes: Sequence[Any],
    records: Sequence[Mapping[str, object]],
    *,
    wall_seconds: float,
    cpu_seconds: float,
) -> dict[str, object]:
    counts: Counter[str] = Counter()
    for record in records:
        for name, value in record["policy_statistics"].items():
            if type(value) is int:
                counts[name] += value
            elif isinstance(value, Mapping) and all(
                isinstance(key, str) and type(count) is int
                for key, count in value.items()
            ):
                for key, count in value.items():
                    counts[f"{name}:{key}"] += count
            else:
                raise TypeError("episode policy statistics are not integer counts")
    return {
        "episodes": list(records),
        "aggregate": oracle._augment_aggregate(
            episodes, r3d._aggregate(episodes)
        ),
        "policy_counts": dict(sorted(counts.items())),
        "cost": {
            "wall_seconds": wall_seconds,
            "cpu_seconds": cpu_seconds,
            "episode_wall_seconds": sum(
                float(record["cost"]["wall_seconds"]) for record in records
            ),
            "episode_cpu_seconds": sum(
                float(record["cost"]["cpu_seconds"]) for record in records
            ),
        },
    }


def _collect(
    *,
    library_path: Path,
    base_path: Path,
    base_sha256: str,
    base_options: Mapping[str, Any],
    teacher: Any,
    seeds: Sequence[int],
    episode_ticks: int,
    query_stride_shots: int,
    maximum_search_queries: int,
    source_identity_sha256: str,
) -> tuple[Any, Any, dict[str, object]]:
    wall_started = time.monotonic()
    cpu_started = time.process_time()
    examples: list[Any] = []
    ranking_examples: list[Any] = []
    episodes: list[Any] = []
    records: list[dict[str, object]] = []
    runtime_sha256 = r3d._file_sha256(library_path)
    with IrisuEnv(
        library_path=library_path,
        physics_backend="portable",
        config={"max_episode_ticks": episode_ticks},
    ) as env:
        if Path(env.library_path).resolve() != library_path:
            raise RuntimeError("distillation collection loaded a foreign runtime")
        runner = env.runner_identity_manifest()
        config_hash = int(runner["config_hash"])
        for seed in seeds:
            policy = r3e.OracleVisitedCollectionPolicy(
                env=env,
                base_policy=r3e._base_policy(
                    base_path, base_sha256, base_options
                ),
                teacher=teacher,
                seed=int(seed),
                episode_ticks=episode_ticks,
                query_stride_shots=query_stride_shots,
                maximum_search_queries=maximum_search_queries,
                source_identity_sha256=source_identity_sha256,
                runtime_sha256=runtime_sha256,
                base_policy_sha256=base_sha256,
                rollout_mode="oracle_visited",
            )
            episode_wall = time.monotonic()
            episode_cpu = time.process_time()
            episode = r3d._run_episode(
                env,
                policy,
                label="reserve_band_oracle_visited_collection",
                seed=int(seed),
                config_hash=config_hash,
            )
            statistics = policy.statistics()
            episodes.append(episode)
            records.append(
                _record(
                    episode,
                    statistics,
                    wall_seconds=time.monotonic() - episode_wall,
                    cpu_seconds=time.process_time() - episode_cpu,
                )
            )
            examples.extend(policy.examples)
            ranking_examples.extend(policy.ranking_examples)
    winner = r3e.GeometryDataset(examples)
    ranking = r3e.GeometryRankingDataset(ranking_examples)
    r3e._require_aligned_collections(winner, ranking)
    report = {
        "runner": runner,
        **_aggregate(
            episodes,
            records,
            wall_seconds=time.monotonic() - wall_started,
            cpu_seconds=time.process_time() - cpu_started,
        ),
        "winner_dataset": {
            "sha256": winner.sha256,
            "examples": len(winner),
            "strict_improvements": sum(
                example.improved_over_incumbent for example in winner
            ),
        },
        "all_branch_dataset": {
            "sha256": ranking.sha256,
            "examples": len(ranking),
            "preferences": sum(len(example.preferences) for example in ranking),
        },
    }
    return winner, ranking, report


def _evaluate(
    *,
    label: str,
    library_path: Path,
    seeds: Sequence[int],
    horizon_ticks: int,
    factory: Callable[[IrisuEnv, int], object],
) -> dict[str, object]:
    wall_started = time.monotonic()
    cpu_started = time.process_time()
    episodes: list[Any] = []
    records: list[dict[str, object]] = []
    with IrisuEnv(
        library_path=library_path,
        physics_backend="portable",
        config={"max_episode_ticks": horizon_ticks},
    ) as env:
        if Path(env.library_path).resolve() != library_path:
            raise RuntimeError("distillation evaluation loaded a foreign runtime")
        runner = env.runner_identity_manifest()
        config_hash = int(runner["config_hash"])
        for seed in seeds:
            policy = factory(env, int(seed))
            episode_wall = time.monotonic()
            episode_cpu = time.process_time()
            episode = r3d._run_episode(
                env,
                policy,
                label=label,
                seed=int(seed),
                config_hash=config_hash,
            )
            statistics = getattr(policy, "statistics", None)
            values = statistics() if callable(statistics) else {}
            if isinstance(policy, oracle.SampledOraclePolicy):
                values = {
                    **values,
                    "geometry_corrections": int(
                        values.get("strict_improvements", 0)
                    ),
                    "model_queries": 0,
                }
            elif isinstance(policy, r3e.GeometrySelectorPolicy):
                values = {
                    **values,
                    "model_queries": sum(
                        int(values.get(name, 0))
                        for name in (
                            "geometry_corrections",
                            "incumbent_selections",
                            "confidence_fallbacks",
                        )
                    ),
                    "teacher_search_queries": 0,
                }
            else:
                values = {
                    **values,
                    "geometry_corrections": 0,
                    "model_queries": 0,
                    "teacher_search_queries": 0,
                }
            if any(type(value) is not int for value in values.values()):
                raise TypeError("evaluation policy statistics must be integers")
            episodes.append(episode)
            records.append(
                _record(
                    episode,
                    values,
                    wall_seconds=time.monotonic() - episode_wall,
                    cpu_seconds=time.process_time() - episode_cpu,
                )
            )
    return {
        "runner": runner,
        **_aggregate(
            episodes,
            records,
            wall_seconds=time.monotonic() - wall_started,
            cpu_seconds=time.process_time() - cpu_started,
        ),
    }


@dataclass(frozen=True, slots=True)
class TrainedPoint:
    steps: int
    model: Any
    initial_state_sha256: str
    final_state_sha256: str
    report: Any
    wall_seconds: float
    cpu_seconds: float


def _train_curve(
    dataset: Any,
    *,
    config: Mapping[str, Any],
    budgets: Sequence[int],
) -> tuple[TrainedPoint, ...]:
    distillation = config["distillation"]
    seed = int(distillation["training_seed"])
    model_config = r3e.GeometryModelConfig(**config["geometry_model"])
    points: list[TrainedPoint] = []
    for steps in budgets:
        torch.manual_seed(seed)
        model = r3e.GeometrySelectorModel(
            dataset.schema,
            candidate_count=dataset.candidate_count,
            candidate_set_sha256=dataset.candidate_vocabulary_sha256,
            config=model_config,
        )
        initial = r3e._state_dict_sha256(model)
        wall_started = time.monotonic()
        cpu_started = time.process_time()
        report = r3e.train_geometry_ranker(
            model,
            dataset,
            steps=int(steps),
            batch_size=int(distillation["batch_size"]),
            learning_rate=float(distillation["learning_rate"]),
            seed=seed,
        )
        model.eval()
        points.append(
            TrainedPoint(
                int(steps),
                model,
                initial,
                r3e._state_dict_sha256(model),
                report,
                time.monotonic() - wall_started,
                time.process_time() - cpu_started,
            )
        )
    if len({point.initial_state_sha256 for point in points}) != 1:
        raise RuntimeError("ranker plateau did not use fresh identical initialization")
    return tuple(points)


def _artifact_metadata(
    *,
    source_identity: Mapping[str, object],
    config_sha256: str,
    runtime_sha256: str,
    base_sha256: str,
    teacher_sha256: str,
    vocabulary_sha256: str,
    suite: Mapping[str, Any],
    seeds: Sequence[int],
    protocol: Mapping[str, object],
) -> dict[str, object]:
    return {
        "source_identity_sha256": source_identity["sha256"],
        "config_sha256": config_sha256,
        "runtime_sha256": runtime_sha256,
        "base_policy_sha256": base_sha256,
        "teacher_sha256": teacher_sha256,
        "candidate_vocabulary_sha256": vocabulary_sha256,
        "suite_label": suite["label"],
        "suite_sha256": suite["sha256"],
        "seeds": list(seeds),
        "protocol": dict(protocol),
    }


def _paired(
    candidate_name: str,
    candidate: Mapping[str, Any],
    comparator_name: str,
    comparator: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, object]:
    return oracle._paired(
        candidate_name,
        candidate,
        comparator_name,
        comparator,
        minimum_tick_loss=int(config["ab"]["catastrophic_minimum_tick_loss"]),
        survival_ratio=float(config["ab"]["catastrophic_survival_ratio"]),
    )


def _select_student(
    evaluations: Mapping[str, Mapping[str, Any]],
    regressions: Mapping[str, Mapping[str, Any]],
    *,
    primary_horizon: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    horizon = str(primary_horizon)
    for steps, curves in evaluations.items():
        aggregate = curves[horizon]["aggregate"]
        catastrophe = int(
            regressions[steps][horizon]["student_vs_base"][
                "catastrophic_regressions"
            ]
        )
        rank = (
            -catastrophe,
            -int(aggregate["gauge_failures"]),
            float(aggregate["survival_ticks"]["p10"]),
            float(aggregate["survival_ticks"]["median"]),
            float(aggregate["raw_score"]["median"]),
            -int(steps),
        )
        rows.append(
            {
                "steps": int(steps),
                "rank": list(rank),
                "catastrophic_regressions_vs_base": catastrophe,
                "gauge_failures": int(aggregate["gauge_failures"]),
                "survival_p10": float(
                    aggregate["survival_ticks"]["p10"]
                ),
                "survival_median": float(
                    aggregate["survival_ticks"]["median"]
                ),
                "score_median": float(aggregate["raw_score"]["median"]),
            }
        )
    selected = max(rows, key=lambda row: tuple(row["rank"]))
    return {
        "primary_horizon": primary_horizon,
        "rule": [
            "minimum paired catastrophic regressions versus frozen v5",
            "minimum gauge failures",
            "maximum survival p10",
            "maximum median survival",
            "maximum median score",
            "minimum training steps",
        ],
        "rows": rows,
        "selected_steps": int(selected["steps"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    wall_started = time.monotonic()
    cpu_started = time.process_time()
    config_path = args.config.resolve(strict=True)
    if config_path != DEFAULT_CONFIG.resolve(strict=True):
        parser.error("--config must be the current reserve-band R3e config")
    config_snapshot = r3d._snapshot_file(
        config_path, "reserve-band distillation config"
    )
    config = oracle._load_config(config_snapshot)
    runtime_snapshot, base_snapshot = _require_exact_inputs(config, parser)
    source_identity = _source_identity(config_snapshot.path)
    registry = oracle._seed_registry(config)
    collection_suite = registry["suites"]["collection"]
    student_suite = registry["suites"]["student"]
    if set(collection_suite["seeds"]) & set(student_suite["seeds"]):
        raise RuntimeError("collection and student development suites overlap")

    distillation = config["distillation"]
    if args.smoke:
        collection_seeds = tuple(collection_suite["seeds"][:1])
        student_seeds = tuple(student_suite["seeds"][:1])
        collection_ticks = 400
        horizons = (400,)
        budgets = (1, 2)
        maximum_queries = 2
        runway_ticks = 64
        batch_size = 4
    else:
        collection_seeds = tuple(
            collection_suite["seeds"][: int(distillation["collection_seeds"])]
        )
        student_seeds = tuple(
            student_suite["seeds"][: int(distillation["evaluation_seeds"])]
        )
        collection_ticks = int(distillation["collection_ticks"])
        horizons = tuple(
            int(value) for value in distillation["evaluation_horizons"]
        )
        budgets = tuple(int(value) for value in distillation["training_steps"])
        maximum_queries = int(
            distillation["maximum_search_queries_per_episode"]
        )
        runway_ticks = int(config["oracle"]["runway_ticks"])
        batch_size = int(distillation["batch_size"])
    if (
        not horizons
        or tuple(sorted(set(horizons))) != horizons
        or not budgets
        or tuple(sorted(set(budgets))) != budgets
    ):
        raise ValueError("distillation horizons and budgets must increase uniquely")
    if set(collection_seeds) & set(student_seeds):
        raise RuntimeError("active collection and student seeds overlap")

    torch.set_num_threads(int(distillation["torch_threads"]))
    run_dir = _safe_run_directory(
        config, args.run_name, smoke=bool(args.smoke)
    )
    base_options = oracle._base_options(config)
    base_identity = _base_identity(
        base_snapshot.path, base_snapshot.sha256, base_options
    )
    teacher = oracle._teacher(
        config, "reserve_band", runway_ticks=runway_ticks
    )
    vocabulary = geometry_candidate_vocabulary_manifest(
        teacher.config.candidate_config
    )
    vocabulary_sha256 = geometry_candidate_vocabulary_sha256(
        teacher.config.candidate_config
    )
    if vocabulary_sha256 != _canonical_sha256(vocabulary):
        raise RuntimeError("candidate vocabulary identity is inconsistent")

    collection_protocol = {
        "rollout_mode": "oracle_visited",
        "episode_ticks": collection_ticks,
        "query_stride_shots": int(distillation["query_stride_shots"]),
        "maximum_search_queries_per_episode": maximum_queries,
        "runway_ticks": runway_ticks,
        "smoke_override": bool(args.smoke),
    }
    winner, ranking, collection_report = _collect(
        library_path=runtime_snapshot.path,
        base_path=base_snapshot.path,
        base_sha256=base_snapshot.sha256,
        base_options=base_options,
        teacher=teacher,
        seeds=collection_seeds,
        episode_ticks=collection_ticks,
        query_stride_shots=int(distillation["query_stride_shots"]),
        maximum_search_queries=maximum_queries,
        source_identity_sha256=str(source_identity["sha256"]),
    )
    common_metadata = _artifact_metadata(
        source_identity=source_identity,
        config_sha256=config_snapshot.sha256,
        runtime_sha256=runtime_snapshot.sha256,
        base_sha256=base_snapshot.sha256,
        teacher_sha256=teacher.sha256,
        vocabulary_sha256=vocabulary_sha256,
        suite=collection_suite,
        seeds=collection_seeds,
        protocol=collection_protocol,
    )
    winner_path = run_dir / "reserve-band-winner-collection.pt"
    ranking_path = run_dir / "reserve-band-all-branch-collection.pt"
    winner_sha256 = r3e.save_geometry_collection(
        winner_path,
        winner,
        metadata={
            **common_metadata,
            "artifact_role": "reserve_band_winner_labels",
        },
        overwrite=False,
    )
    ranking_sha256 = r3e.save_geometry_ranking_collection(
        ranking_path,
        ranking,
        metadata={
            **common_metadata,
            "artifact_role": "reserve_band_all_branch_preferences",
            "winner_dataset_sha256": winner.sha256,
            "winner_artifact_sha256": winner_sha256,
        },
        overwrite=False,
    )
    winner_loaded = r3e.load_geometry_collection(
        winner_path, expected_sha256=winner_sha256
    )
    ranking_loaded = r3e.load_geometry_ranking_collection(
        ranking_path, expected_sha256=ranking_sha256
    )
    r3e._require_bound_metadata(
        winner_loaded.metadata,
        {
            **common_metadata,
            "artifact_role": "reserve_band_winner_labels",
        },
        "winner collection",
    )
    r3e._require_bound_metadata(
        ranking_loaded.metadata,
        {
            **common_metadata,
            "artifact_role": "reserve_band_all_branch_preferences",
            "winner_dataset_sha256": winner.sha256,
            "winner_artifact_sha256": winner_sha256,
        },
        "all-branch collection",
    )
    r3e._require_aligned_collections(
        winner_loaded.dataset, ranking_loaded.dataset
    )

    training_config = {
        **dict(distillation),
        "batch_size": batch_size,
    }
    config_for_training = {
        **dict(config),
        "distillation": training_config,
    }
    points = _train_curve(
        ranking_loaded.dataset,
        config=config_for_training,
        budgets=budgets,
    )
    checkpoint_bindings = {
        **common_metadata,
        "ranking_artifact_sha256": ranking_sha256,
        "ranking_dataset_sha256": ranking_loaded.dataset.sha256,
        "fresh_identical_initialization": True,
        "training_seed": int(distillation["training_seed"]),
        "learning_rate": float(distillation["learning_rate"]),
        "batch_size": batch_size,
        "loss": {"listwise_weight": 1.0, "pairwise_weight": 1.0},
    }
    checkpoints: dict[int, Any] = {}
    checkpoint_reports: dict[str, object] = {}
    for point in points:
        checkpoint_path = (
            run_dir / f"reserve-band-ranker-steps-{point.steps}.pt"
        )
        metadata = {
            **checkpoint_bindings,
            "training_steps": point.steps,
            "initial_state_sha256": point.initial_state_sha256,
            "final_state_sha256": point.final_state_sha256,
            "training_report": asdict(point.report),
        }
        checkpoint_sha256 = r3e.save_geometry_checkpoint(
            checkpoint_path,
            point.model,
            search_identity=teacher.identity_manifest(),
            metadata=metadata,
            overwrite=False,
        )
        loaded = r3e.load_geometry_checkpoint(
            checkpoint_path, expected_sha256=checkpoint_sha256
        )
        if (
            dict(loaded.search_identity) != teacher.identity_manifest()
            or r3e._state_dict_sha256(loaded.model)
            != point.final_state_sha256
        ):
            raise RuntimeError("reloaded ranker checkpoint identity differs")
        r3e._require_bound_metadata(
            loaded.metadata, metadata, f"ranker-{point.steps}"
        )
        checkpoints[point.steps] = loaded
        checkpoint_reports[str(point.steps)] = {
            "path": str(loaded.path),
            "sha256": loaded.sha256,
            "model_manifest": loaded.model.manifest(),
            "initial_state_sha256": point.initial_state_sha256,
            "final_state_sha256": point.final_state_sha256,
            "training_report": asdict(point.report),
            "cost": {
                "wall_seconds": point.wall_seconds,
                "cpu_seconds": point.cpu_seconds,
            },
            "reload_verified": True,
        }

    query_stride = int(config["oracle"]["query_stride_shots"])
    teacher_max_queries = int(
        config["oracle"]["maximum_search_queries_per_episode"]
    )
    if args.smoke:
        teacher_max_queries = 2
    base_curves: dict[str, object] = {}
    teacher_curves: dict[str, object] = {}
    student_curves: dict[str, dict[str, object]] = {
        str(point.steps): {} for point in points
    }
    for horizon in horizons:
        base_curves[str(horizon)] = _evaluate(
            label=f"reserve_band_student_base_v5_{horizon}",
            library_path=runtime_snapshot.path,
            seeds=student_seeds,
            horizon_ticks=horizon,
            factory=lambda _env, _seed: r3e._base_policy(
                base_snapshot.path, base_snapshot.sha256, base_options
            ),
        )
        teacher_curves[str(horizon)] = _evaluate(
            label=f"reserve_band_student_teacher_{horizon}",
            library_path=runtime_snapshot.path,
            seeds=student_seeds,
            horizon_ticks=horizon,
            factory=lambda env, seed, horizon=horizon: oracle.SampledOraclePolicy(
                env=env,
                base_policy=r3e._base_policy(
                    base_snapshot.path, base_snapshot.sha256, base_options
                ),
                teacher=teacher,
                seed=seed,
                episode_ticks=horizon,
                query_stride_shots=query_stride,
                maximum_search_queries=teacher_max_queries,
            ),
        )
        for point in points:
            loaded = checkpoints[point.steps]
            student_curves[str(point.steps)][str(horizon)] = _evaluate(
                label=f"reserve_band_student_{point.steps}_{horizon}",
                library_path=runtime_snapshot.path,
                seeds=student_seeds,
                horizon_ticks=horizon,
                factory=lambda _env, _seed, loaded=loaded: (
                    r3e.GeometrySelectorPolicy(
                        r3e._base_policy(
                            base_snapshot.path,
                            base_snapshot.sha256,
                            base_options,
                        ),
                        loaded.model,
                        teacher,
                        minimum_confidence=float(
                            config["deployment"][
                                "minimum_candidate_confidence"
                            ]
                        ),
                        minimum_margin=float(
                            config["deployment"][
                                "minimum_probability_margin_over_incumbent"
                            ]
                        ),
                    )
                ),
            )

    regressions: dict[str, dict[str, object]] = {}
    teacher_regressions: dict[str, object] = {}
    for horizon in horizons:
        key = str(horizon)
        teacher_regressions[key] = _paired(
            "reserve_band_teacher",
            teacher_curves[key],
            "base_v5",
            base_curves[key],
            config,
        )
    for point in points:
        step_key = str(point.steps)
        regressions[step_key] = {}
        for horizon in horizons:
            key = str(horizon)
            student = student_curves[step_key][key]
            regressions[step_key][key] = {
                "student_vs_base": _paired(
                    f"student_{point.steps}",
                    student,
                    "base_v5",
                    base_curves[key],
                    config,
                ),
                "student_vs_teacher": _paired(
                    f"student_{point.steps}",
                    student,
                    "reserve_band_teacher",
                    teacher_curves[key],
                    config,
                ),
            }
    selection = _select_student(
        student_curves,
        regressions,
        primary_horizon=max(horizons),
    )

    _require_source_identity(source_identity, config_snapshot.path)
    for snapshot, name in (
        (config_snapshot, "reserve-band distillation config"),
        (runtime_snapshot, "reserve-band portable runtime"),
        (base_snapshot, "reserve-band frozen v5"),
    ):
        r3d._require_unchanged(snapshot, name)
    if (
        r3d._file_sha256(winner_loaded.path) != winner_loaded.sha256
        or r3d._file_sha256(ranking_loaded.path) != ranking_loaded.sha256
        or any(
            r3d._file_sha256(checkpoint.path) != checkpoint.sha256
            for checkpoint in checkpoints.values()
        )
    ):
        raise RuntimeError("a generated distillation artifact changed after reload")

    content = {
        "schema": REPORT_SCHEMA,
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
        "source_identity": source_identity,
        "config": {
            "path": str(config_snapshot.path),
            "sha256": config_snapshot.sha256,
        },
        "runtime": {
            "path": str(runtime_snapshot.path),
            "sha256": runtime_snapshot.sha256,
            "backend": "portable",
        },
        "base_policy": {
            "path": str(base_snapshot.path),
            "sha256": base_snapshot.sha256,
            "identity": base_identity,
        },
        "teacher": {
            "identity": teacher.identity_manifest(),
            "sha256": teacher.sha256,
        },
        "candidate_vocabulary": {
            "manifest": vocabulary,
            "sha256": vocabulary_sha256,
        },
        "seed_registry": registry,
        "active_suites": {
            "collection": {
                "label": collection_suite["label"],
                "sha256": collection_suite["sha256"],
                "seeds": list(collection_seeds),
            },
            "student": {
                "label": student_suite["label"],
                "sha256": student_suite["sha256"],
                "seeds": list(student_seeds),
            },
            "disjoint": not bool(set(collection_seeds) & set(student_seeds)),
        },
        "protocol": {
            "collection": collection_protocol,
            "evaluation_horizons": list(horizons),
            "training_steps": list(budgets),
            "training_seed": int(distillation["training_seed"]),
            "fresh_identical_initialization": True,
            "all_branch_ranker": True,
            "student_policy": (
                "confidence-and-margin shielded ranker over frozen-v5 pair "
                "selection; evaluated on every supported v5 shot"
            ),
            "selection_disclosure": (
                "development student seeds select the plateau point and report "
                "its development metrics; there is no independent test claim"
            ),
            "teacher_policy": (
                "sampled transactional reserve-band MPC over frozen-v5 pair "
                "selection and cadence"
            ),
            "smoke_override": bool(args.smoke),
        },
        "collection": collection_report,
        "artifacts": {
            "winner_collection": {
                "path": str(winner_loaded.path),
                "sha256": winner_loaded.sha256,
                "dataset_sha256": winner_loaded.dataset.sha256,
                "reload_verified": True,
            },
            "all_branch_collection": {
                "path": str(ranking_loaded.path),
                "sha256": ranking_loaded.sha256,
                "dataset_sha256": ranking_loaded.dataset.sha256,
                "reload_verified": True,
            },
            "ranker_checkpoints": checkpoint_reports,
        },
        "evaluation": {
            "base_v5": base_curves,
            "reserve_band_teacher": teacher_curves,
            "students": student_curves,
        },
        "paired_regressions": {
            "teacher_vs_base": teacher_regressions,
            "students": regressions,
        },
        "selection": selection,
        "immutable_verification": {
            "source": True,
            "config": True,
            "runtime": True,
            "base_policy": True,
            "generated_artifacts": True,
        },
        "execution": {
            "wall_seconds": time.monotonic() - wall_started,
            "cpu_seconds": time.process_time() - cpu_started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
        },
    }
    report = {**content, "payload_sha256": _canonical_sha256(content)}
    report_path = run_dir / "reserve-band-distillation-report.json"
    report_sha256 = r3e._atomic_write_json(
        report_path, report, overwrite=False
    )
    print(
        json.dumps(
            {
                "report": str(report_path),
                "file_sha256": report_sha256,
                "payload_sha256": report["payload_sha256"],
                "selected_steps": selection["selected_steps"],
                "collection_examples": len(ranking_loaded.dataset),
                "smoke": bool(args.smoke),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
