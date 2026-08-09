#!/usr/bin/env python3
"""Development-only reserve-band oracle A/B benchmark.

This driver uses only the trusted portable runtime and frozen R3d v5 policy.
It never reads sealed inputs or writes canonical R3 run storage.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import platform
import sys
import time
import tomllib
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from irisu_env import IrisuEnv
from irisu_pointer.geometry_policy import (
    geometry_candidate_vocabulary_sha256,
)
from irisu_pointer.reserve_band_search import (
    ReserveBandBranchOutcome,
    ReserveBandGeometrySearch,
    ReserveBandSearchConfig,
)
from irisu_pointer.steering import SteeringDecision


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = Path(__file__).resolve().parent
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))
import rl_r3d_steering as r3d  # noqa: E402
import rl_r3e_reserve_band_checks as reserve_checks  # noqa: E402
import rl_r3e_sustainable as r3e  # noqa: E402


DEFAULT_CONFIG = (
    ROOT / "configs/rl/experiments/r3e-reserve-band-debt-mpc-v2.toml"
)
TRUSTED_RUNTIME = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/runtime/"
    "main-0c48dba-20260723/portable-build/libirisu_clone.so"
)
FROZEN_V5 = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3d-survival-v5-20260729/long-development.pt"
)
REPORT_SCHEMA = "irisu-r3e-reserve-band-debt-oracle-ab-development-v2"
MODES = ("base_v5", "score_first", "pure_gauge", "reserve_band")
SUITE_KEYS = {
    "tuning": "tuning_suite",
    "confirmation": "confirmation_suite",
    "long": "long_probe_suite",
    "collection": "collection_suite",
    "student": "evaluation_suite",
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _derive_seeds(label: str, count: int = 32) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(
            hashlib.sha256(f"{label}:{index}".encode()).digest()[:4],
            "big",
        )
        for index in range(count)
    )


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _load_config(snapshot: Any) -> dict[str, Any]:
    value = tomllib.loads(snapshot.path.read_text(encoding="utf-8"))
    required = {
        "version": "r3e-reserve-band-debt-mpc-v2",
        "status": "development_only_not_canonical_evidence",
        "deployable": False,
        "canonical_r3_evidence": False,
        "sealed_evaluation_allowed": False,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("reserve-band config weakens development-only boundaries")
    oracle = value["oracle"]
    if (
        oracle["modes"] != ["score_first", "pure_gauge", "reserve_band"]
        or int(oracle["passive_drain_above_half_multiplier"]) != 3
        or int(oracle["passive_drain_at_or_below_half_multiplier"]) != 1
        or str(oracle["regime_hysteresis"]) != "none"
        or int(oracle["rot_delay_ticks"]) != 40
        or int(oracle["minimum_contingency_rot_events"]) != 1
    ):
        raise ValueError("reserve-band config changed the preregistered mechanic")
    return value


def _source_identity(config_path: Path) -> dict[str, object]:
    files = (
        Path(__file__).resolve(),
        Path(r3d.__file__).resolve(),
        Path(r3e.__file__).resolve(),
        ROOT / "benchmarks/rl_r3e_reserve_band_checks.py",
        config_path,
        *sorted((ROOT / "python/irisu_pointer").glob("*.py")),
        ROOT / "python/irisu_rl/actions.py",
        ROOT / "python/irisu_rl/encoding.py",
        ROOT / "python/irisu_rl/schema.py",
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
    )
    manifest = {
        "schema": "irisu-r3e-reserve-band-source-v1",
        "git_revision": r3d._source_revision(),
        "files": {
            str(path.relative_to(ROOT)): r3d._file_sha256(path)
            for path in files
        },
    }
    return {**manifest, "sha256": _canonical_sha256(manifest)}


def _require_source_identity(
    expected: Mapping[str, object], config_path: Path
) -> None:
    if _source_identity(config_path) != dict(expected):
        raise RuntimeError("reserve-band source identity changed during execution")


def _development_check_result() -> dict[str, object]:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        reserve_checks.main()
    value = json.loads(stream.getvalue())
    if value.get("status") != "PASS":
        raise RuntimeError("reserve-band development checks did not pass")
    return value


def _seed_registry(config: Mapping[str, Any]) -> dict[str, object]:
    labels = {
        "tuning": str(config["ab"]["tuning_suite"]),
        "confirmation": str(config["ab"]["confirmation_suite"]),
        "long": str(config["ab"]["long_probe_suite"]),
        "collection": str(config["distillation"]["collection_suite"]),
        "student": str(config["distillation"]["evaluation_suite"]),
    }
    suites = {
        name: {
            "label": label,
            "seeds": list(_derive_seeds(label)),
            "sha256": _canonical_sha256(
                {"label": label, "seeds": list(_derive_seeds(label))}
            ),
        }
        for name, label in labels.items()
    }
    sets = {
        name: set(value["seeds"])
        for name, value in suites.items()
    }
    overlaps = {
        f"{left}:{right}": sorted(sets[left] & sets[right])
        for left_index, left in enumerate(sets)
        for right in tuple(sets)[left_index + 1 :]
    }
    if any(overlaps.values()):
        raise RuntimeError("reserve-band development seed suites overlap")
    return {
        "suites": suites,
        "pairwise_overlaps": overlaps,
        "pairwise_disjoint": True,
    }


def _teacher(
    config: Mapping[str, Any],
    mode: str,
    *,
    runway_ticks: int | None = None,
) -> ReserveBandGeometrySearch:
    oracle = config["oracle"]
    candidate_config = r3e._search_config(config["search"])
    return ReserveBandGeometrySearch(
        config=ReserveBandSearchConfig(
            mode=mode,
            runway_ticks=(
                int(oracle["runway_ticks"])
                if runway_ticks is None
                else int(runway_ticks)
            ),
            candidate_config=candidate_config,
            efficiency_ceiling_numerator=int(
                oracle["efficiency_ceiling_numerator"]
            ),
            efficiency_ceiling_denominator=int(
                oracle["efficiency_ceiling_denominator"]
            ),
            rot_delay_ticks=int(oracle["rot_delay_ticks"]),
            minimum_contingency_rot_events=int(
                oracle["minimum_contingency_rot_events"]
            ),
        )
    )


def _base_options(config: Mapping[str, Any]) -> dict[str, object]:
    return {
        key: value
        for key, value in config["base_policy"].items()
        if key not in {"checkpoint", "sha256"}
    }


class SampledOraclePolicy:
    """Replace sampled v5 shot geometry with one transactional oracle winner."""

    def __init__(
        self,
        *,
        env: IrisuEnv,
        base_policy: object,
        teacher: ReserveBandGeometrySearch,
        seed: int,
        episode_ticks: int,
        query_stride_shots: int,
        maximum_search_queries: int,
    ) -> None:
        self.env = env
        self.base_policy = base_policy
        self.teacher = teacher
        self.seed = int(seed)
        self.episode_ticks = int(episode_ticks)
        self.query_stride_shots = int(query_stride_shots)
        self.maximum_search_queries = int(maximum_search_queries)
        if any(
            type(value) is not int or value < 1
            for value in (
                self.episode_ticks,
                self.query_stride_shots,
                self.maximum_search_queries,
            )
        ):
            raise ValueError("sampled oracle counts must be positive")
        self.counts: Counter[str] = Counter()
        self.selected: Counter[str] = Counter()

    def reset(self, seed: int = 0) -> None:
        if int(seed) != self.seed:
            raise RuntimeError("sampled oracle seed binding changed")
        getattr(self.base_policy, "reset")(seed)
        self.counts.clear()
        self.selected.clear()

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = getattr(self.base_policy, "predict")(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("frozen v5 did not return a SteeringDecision")
        if not incumbent.is_shot:
            return incumbent
        self.counts["seen_shots"] += 1
        if not r3e._piece_pair(observation, incumbent):
            self.counts["unsupported_pairs"] += 1
            return incumbent
        if (
            (self.counts["seen_shots"] - 1) % self.query_stride_shots
            or self.counts["search_queries"] >= self.maximum_search_queries
        ):
            return incumbent
        tick = int(observation.get("tick", 0))
        if tick + self.teacher.config.runway_ticks >= self.episode_ticks:
            self.counts["episode_boundary_skips"] += 1
            return incumbent
        before = self.env.clone_state()
        result = self.teacher.search(self.env, observation, incumbent)
        after = self.env.clone_state()
        self.counts["transactional_restore_checks"] += 1
        if before != after:
            raise RuntimeError("sampled reserve-band search changed source state")
        self.counts["search_queries"] += 1
        self.counts["branch_outcomes"] += len(result.outcomes)
        self.counts["strict_improvements"] += int(result.strictly_improved)
        self.counts["executed_corrections"] += int(result.winner_ordinal != 0)
        selected = next(
            outcome
            for outcome in result.outcomes
            if outcome.candidate.ordinal == result.winner_ordinal
        )
        if not isinstance(selected, ReserveBandBranchOutcome):
            raise TypeError("reserve-band result lost traced branch evidence")
        self.counts[f"regime_{selected.selection_regime}"] += 1
        self.counts["selected_minimum_gauge_sum"] += selected.minimum_gauge
        self.counts["selected_gauge_tick_sum"] += selected.gauge_tick_sum
        self.counts["selected_gauge_tick_count"] += selected.gauge_tick_count
        self.counts["selected_gross_recovery"] += selected.gross_gauge_recovery
        self.counts["selected_branch_rot"] += selected.rot_count
        self.counts["selected_branch_cleared_events"] += selected.cleared_events
        self.selected[selected.candidate.name] += 1
        return result.decision

    def statistics(self) -> dict[str, int]:
        return {
            **dict(sorted(self.counts.items())),
            **{
                f"selected_candidate:{name}": count
                for name, count in sorted(self.selected.items())
            },
        }


def _distribution(values: Sequence[int]) -> dict[str, int | float]:
    return r3d._distribution(tuple(int(value) for value in values))


def _augment_aggregate(
    episodes: Sequence[Any], aggregate: Mapping[str, object]
) -> dict[str, object]:
    return {
        **dict(aggregate),
        "final_gauge": _distribution(
            [episode.final_gauge for episode in episodes]
        ),
        "final_level": _distribution(
            [episode.final_level for episode in episodes]
        ),
        "qualifying_clears_distribution": _distribution(
            [episode.qualifying_clears for episode in episodes]
        ),
        "decisions": {
            "total": sum(episode.decisions for episode in episodes),
            "distribution": _distribution(
                [episode.decisions for episode in episodes]
            ),
        },
        "primitive_actions": {
            "total": sum(episode.primitive_actions for episode in episodes),
            "distribution": _distribution(
                [episode.primitive_actions for episode in episodes]
            ),
        },
    }


def _evaluate(
    *,
    label: str,
    mode: str,
    library_path: Path,
    base_path: Path,
    base_sha256: str,
    base_options: Mapping[str, Any],
    teacher: ReserveBandGeometrySearch | None,
    seeds: Sequence[int],
    horizon_ticks: int,
    query_stride_shots: int,
    maximum_search_queries: int,
) -> dict[str, object]:
    wall_started = time.monotonic()
    cpu_started = time.process_time()
    episodes: list[Any] = []
    episode_records: list[dict[str, object]] = []
    policy_counts: Counter[str] = Counter()
    with IrisuEnv(
        library_path=library_path,
        physics_backend="portable",
        config={"max_episode_ticks": horizon_ticks},
    ) as env:
        if Path(env.library_path).resolve() != library_path:
            raise RuntimeError("oracle evaluation loaded a foreign runtime")
        runner = env.runner_identity_manifest()
        config_hash = int(runner["config_hash"])
        for seed in seeds:
            base = r3e._base_policy(base_path, base_sha256, base_options)
            policy: object
            if mode == "base_v5":
                policy = base
            else:
                assert teacher is not None
                policy = SampledOraclePolicy(
                    env=env,
                    base_policy=base,
                    teacher=teacher,
                    seed=int(seed),
                    episode_ticks=horizon_ticks,
                    query_stride_shots=query_stride_shots,
                    maximum_search_queries=maximum_search_queries,
                )
            episode = r3d._run_episode(
                env,
                policy,
                label=label,
                seed=int(seed),
                config_hash=config_hash,
            )
            episodes.append(episode)
            statistics = getattr(policy, "statistics", None)
            episode_counts = statistics() if callable(statistics) else {}
            if callable(statistics):
                policy_counts.update(episode_counts)
            manifest = episode.manifest()
            legacy_failure = bool(manifest["gauge_failure"])
            exact_failure = bool(manifest["conversion"]["terminated"])
            manifest["legacy_final_gauge_failure"] = legacy_failure
            manifest["gauge_failure"] = exact_failure
            manifest["gauge_failure_signal"] = (
                "terminal GAME_OVER/terminated state, independent of "
                "same-tick final-gauge recovery"
            )
            manifest["policy_counts"] = dict(sorted(episode_counts.items()))
            episode_records.append(manifest)
    aggregate = _augment_aggregate(episodes, r3d._aggregate(episodes))
    aggregate["gauge_failures"] = sum(
        bool(record["gauge_failure"]) for record in episode_records
    )
    wall = time.monotonic() - wall_started
    cpu = time.process_time() - cpu_started
    return {
        "runner": runner,
        "episodes": episode_records,
        "aggregate": aggregate,
        "policy_counts": dict(sorted(policy_counts.items())),
        "cost": {
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "episodes": len(episodes),
            "wall_seconds_per_episode": wall / len(episodes),
        },
    }


def _episode_metrics(value: Mapping[str, Any]) -> dict[str, int | bool]:
    conversion = value["conversion"]
    return {
        "score": int(conversion["final_score"]),
        "survival_ticks": int(conversion["survival_ticks"]),
        "final_gauge": int(value["final_gauge"]),
        "final_level": int(value["final_level"]),
        "gauge_failure": bool(value["gauge_failure"]),
        "qualifying_clears": int(value["qualifying_clears"]),
        "cleared_events": int(conversion["clears"]),
        "rot": int(conversion["rotten"]),
        "decisions": int(value["decisions"]),
    }


def _paired(
    candidate_name: str,
    candidate: Mapping[str, Any],
    comparator_name: str,
    comparator: Mapping[str, Any],
    *,
    minimum_tick_loss: int,
    survival_ratio: float,
) -> dict[str, object]:
    candidate_by_seed = {
        int(value["seed"]): value for value in candidate["episodes"]
    }
    comparator_by_seed = {
        int(value["seed"]): value for value in comparator["episodes"]
    }
    if set(candidate_by_seed) != set(comparator_by_seed):
        raise RuntimeError("paired oracle seed sets differ")
    rows: list[dict[str, object]] = []
    for seed in sorted(candidate_by_seed):
        left = _episode_metrics(candidate_by_seed[seed])
        right = _episode_metrics(comparator_by_seed[seed])
        deltas = {
            key: int(left[key]) - int(right[key])
            for key in (
                "score",
                "survival_ticks",
                "final_gauge",
                "final_level",
                "qualifying_clears",
                "cleared_events",
                "rot",
                "decisions",
            )
        }
        reasons: list[str] = []
        if bool(left["gauge_failure"]) and not bool(right["gauge_failure"]):
            reasons.append("new_gauge_failure")
        loss = int(right["survival_ticks"]) - int(left["survival_ticks"])
        if (
            loss >= minimum_tick_loss
            and int(left["survival_ticks"])
            <= float(right["survival_ticks"]) * survival_ratio
        ):
            reasons.append("large_paired_survival_collapse")
        rows.append(
            {
                "seed": seed,
                "candidate": left,
                "comparator": right,
                "delta": deltas,
                "catastrophic_survival_regression": bool(reasons),
                "catastrophic_reasons": reasons,
            }
        )

    def record(key: str) -> dict[str, int]:
        values = [int(row["delta"][key]) for row in rows]
        return {
            "wins": sum(value > 0 for value in values),
            "ties": sum(value == 0 for value in values),
            "losses": sum(value < 0 for value in values),
        }

    return {
        "candidate": candidate_name,
        "comparator": comparator_name,
        "score": record("score"),
        "survival": record("survival_ticks"),
        "catastrophic_regressions": sum(
            bool(row["catastrophic_survival_regression"]) for row in rows
        ),
        "rows": rows,
    }


def _selection_gate(
    values: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, object] | None:
    if not all(name in values for name in ("score_first", "pure_gauge", "reserve_band")):
        return None
    reserve = values["reserve_band"]["aggregate"]
    gauge = values["pure_gauge"]["aggregate"]
    score = values["score_first"]["aggregate"]
    selection = config["selection"]
    catastrophes = {
        comparator: _paired(
            "reserve_band",
            values["reserve_band"],
            comparator,
            values[comparator],
            minimum_tick_loss=int(
                config["ab"]["catastrophic_minimum_tick_loss"]
            ),
            survival_ratio=float(
                config["ab"]["catastrophic_survival_ratio"]
            ),
        )["catastrophic_regressions"]
        for comparator in ("score_first", "pure_gauge")
    }
    checks = {
        "strictly_higher_median_score_than_pure_gauge": (
            float(reserve["raw_score"]["median"])
            > float(gauge["raw_score"]["median"])
        ),
        "pure_gauge_failure_budget": (
            int(reserve["gauge_failures"])
            <= int(gauge["gauge_failures"])
            + int(selection["maximum_extra_gauge_failures"])
        ),
        "pure_gauge_survival_p10_ratio": (
            float(reserve["survival_ticks"]["p10"])
            >= float(gauge["survival_ticks"]["p10"])
            * float(selection["minimum_survival_p10_ratio"])
        ),
        "no_more_failures_than_score_first": (
            int(reserve["gauge_failures"]) <= int(score["gauge_failures"])
        ),
        "zero_new_catastrophic_regressions": not any(
            catastrophes.values()
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "catastrophic_regressions": catastrophes,
        "preregistered_rule": dict(selection),
    }


def _safe_output(config: Mapping[str, Any], path: Path) -> Path:
    namespace = _path(str(config["artifact_namespace"])).resolve()
    output = r3d._output_path(path, "reserve-band oracle report", ".json")
    if namespace != output.parent and namespace not in output.parents:
        raise ValueError("oracle report must stay in reserve-band namespace")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--suite", choices=("tuning", "confirmation", "long"), default="tuning"
    )
    parser.add_argument("--modes", nargs="+", choices=MODES)
    parser.add_argument("--horizons", nargs="+", type=int)
    parser.add_argument("--seed-count", type=int)
    parser.add_argument("--result-out", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    started = time.monotonic()
    cpu_started = time.process_time()
    config_snapshot = r3d._snapshot_file(args.config, "reserve-band config")
    if config_snapshot.path != DEFAULT_CONFIG.resolve(strict=True):
        parser.error("--config must be the identity-bound debt-aware config")
    config = _load_config(config_snapshot)
    development_check = _development_check_result()
    source_identity = _source_identity(config_snapshot.path)
    runtime_snapshot = r3d._snapshot_file(
        _path(str(config["trusted_runtime"])), "reserve-band portable runtime"
    )
    if (
        runtime_snapshot.path != TRUSTED_RUNTIME.resolve(strict=True)
        or runtime_snapshot.sha256 != str(config["trusted_runtime_sha256"])
    ):
        parser.error("trusted portable runtime SHA-256 differs")
    base_snapshot = r3d._snapshot_file(
        _path(str(config["base_policy"]["checkpoint"])),
        "reserve-band frozen v5",
    )
    if (
        base_snapshot.path != FROZEN_V5.resolve(strict=True)
        or base_snapshot.sha256 != str(config["base_policy"]["sha256"])
    ):
        parser.error("frozen v5 SHA-256 differs")

    registry = _seed_registry(config)
    suite = registry["suites"][args.suite]
    default_count = (
        int(config["ab"]["long_probe_seed_count"])
        if args.suite == "long"
        else int(config["ab"]["seed_count_per_suite"])
    )
    seed_count = 1 if args.smoke else (
        default_count if args.seed_count is None else args.seed_count
    )
    if type(seed_count) is not int or not 1 <= seed_count <= len(suite["seeds"]):
        parser.error("seed count is outside the fixed suite")
    seeds = tuple(int(value) for value in suite["seeds"][:seed_count])
    default_horizons = (
        config["ab"]["long_probe_horizons"]
        if args.suite == "long"
        else config["ab"]["horizons"]
    )
    horizons = (
        (400,)
        if args.smoke
        else tuple(
            int(value)
            for value in (
                default_horizons if args.horizons is None else args.horizons
            )
        )
    )
    if (
        not horizons
        or tuple(sorted(set(horizons))) != horizons
        or any(value < 1 for value in horizons)
    ):
        parser.error("horizons must be unique increasing positive integers")
    modes = tuple(MODES if args.modes is None else args.modes)
    if len(set(modes)) != len(modes):
        parser.error("oracle modes must not repeat")

    oracle = config["oracle"]
    runway_ticks = 64 if args.smoke else int(oracle["runway_ticks"])
    query_stride = int(oracle["query_stride_shots"])
    maximum_queries = (
        2 if args.smoke else int(oracle["maximum_search_queries_per_episode"])
    )
    protocol_compliant = (
        not args.smoke
        and args.modes is None
        and args.horizons is None
        and args.seed_count is None
    )
    teachers = {
        mode: _teacher(config, mode, runway_ticks=runway_ticks)
        for mode in modes
        if mode != "base_v5"
    }
    torch.set_num_threads(int(config["distillation"]["torch_threads"]))
    curves: dict[str, dict[str, object]] = {}
    paired: dict[str, dict[str, object]] = {}
    gates: dict[str, object] = {}
    for horizon in horizons:
        values: dict[str, object] = {}
        for mode in modes:
            values[mode] = _evaluate(
                label=f"reserve_band_{args.suite}_{mode}_{horizon}",
                mode=mode,
                library_path=runtime_snapshot.path,
                base_path=base_snapshot.path,
                base_sha256=base_snapshot.sha256,
                base_options=_base_options(config),
                teacher=teachers.get(mode),
                seeds=seeds,
                horizon_ticks=horizon,
                query_stride_shots=query_stride,
                maximum_search_queries=maximum_queries,
            )
        curves[str(horizon)] = values
        pairs: dict[str, object] = {}
        if "reserve_band" in values:
            for comparator in ("base_v5", "score_first", "pure_gauge"):
                if comparator in values:
                    pairs[f"reserve_band_vs_{comparator}"] = _paired(
                        "reserve_band",
                        values["reserve_band"],
                        comparator,
                        values[comparator],
                        minimum_tick_loss=int(
                            config["ab"]["catastrophic_minimum_tick_loss"]
                        ),
                        survival_ratio=float(
                            config["ab"]["catastrophic_survival_ratio"]
                        ),
                    )
        paired[str(horizon)] = pairs
        gate = _selection_gate(values, config)
        if gate is not None:
            gates[str(horizon)] = gate

    _require_source_identity(source_identity, config_snapshot.path)
    r3d._require_unchanged(config_snapshot, "reserve-band config")
    r3d._require_unchanged(runtime_snapshot, "reserve-band portable runtime")
    r3d._require_unchanged(base_snapshot, "reserve-band frozen v5")
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
            "identity": "frozen-r3d-v5",
        },
        "seed_registry": registry,
        "active_suite": {
            "name": args.suite,
            "label": suite["label"],
            "suite_sha256": suite["sha256"],
            "seeds": list(seeds),
            "seed_count": seed_count,
        },
        "protocol": {
            "modes": list(modes),
            "horizons": list(horizons),
            "runway_ticks": runway_ticks,
            "query_stride_shots": query_stride,
            "maximum_search_queries_per_episode": maximum_queries,
            "sampling_scope": (
                "sampled receding-horizon geometry MPC over frozen-v5 pair "
                "and cadence; not every-shot MPC"
            ),
            "endpoint_semantics": (
                "independent horizon-censored evaluations; these are not "
                "prefix checkpoints from one trajectory"
            ),
            "preregistered_full_suite_protocol": protocol_compliant,
            "exploratory_slice": not protocol_compliant,
            "catastrophic_regression_rule": {
                "new_gauge_failure": True,
                "minimum_tick_loss": int(
                    config["ab"]["catastrophic_minimum_tick_loss"]
                ),
                "maximum_survival_ratio": float(
                    config["ab"]["catastrophic_survival_ratio"]
                ),
            },
            "smoke_override": bool(args.smoke),
        },
        "teachers": {
            mode: {
                "identity": teacher.identity_manifest(),
                "sha256": teacher.sha256,
            }
            for mode, teacher in teachers.items()
        },
        "candidate_vocabulary_sha256": (
            geometry_candidate_vocabulary_sha256(
                _teacher(config, "reserve_band").config.candidate_config
            )
        ),
        "development_checks": development_check,
        "curves": curves,
        "paired": paired,
        "reserve_band_selection_gates": gates,
        "execution": {
            "wall_seconds": time.monotonic() - started,
            "cpu_seconds": time.process_time() - cpu_started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
        },
    }
    report = {**content, "payload_sha256": _canonical_sha256(content)}
    default_name = (
        f"{args.suite}-oracle-smoke.json"
        if args.smoke
        else f"{args.suite}-oracle-ab.json"
    )
    result_path = _safe_output(
        config,
        (
            _path(str(config["artifact_namespace"])) / default_name
            if args.result_out is None
            else args.result_out
        ),
    )
    r3e._atomic_write_json(result_path, report, overwrite=False)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
