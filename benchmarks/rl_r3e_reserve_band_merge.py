#!/usr/bin/env python3
"""Merge four identity-bound reserve-band oracle slices.

The inputs must partition base_v5, score_first, pure_gauge, and reserve_band
exactly while sharing the complete development identity, seed, horizon, and
runner context.  This tool performs no simulation and makes no canonical-R3
claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = Path(__file__).resolve().parent
for import_root in (ROOT / "python", BENCHMARKS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import rl_r3d_steering as r3d  # noqa: E402
import rl_r3e_reserve_band as rb  # noqa: E402
import rl_r3e_sustainable as r3e  # noqa: E402


MERGER_VERSION = "r3e-reserve-band-four-slice-merger-v1"
REPORT_SCHEMA = "irisu-r3e-reserve-band-partitioned-composition-development-v1"
MODES = ("base_v5", "score_first", "pure_gauge", "reserve_band")
_SHA_CHARS = frozenset("0123456789abcdef")
_REPORT_FIELDS = {
    "schema",
    "development_only",
    "canonical_r3_evidence",
    "sealed_test_material_used",
    "source_identity",
    "config",
    "runtime",
    "base_policy",
    "seed_registry",
    "active_suite",
    "protocol",
    "teachers",
    "candidate_vocabulary_sha256",
    "development_checks",
    "curves",
    "paired",
    "reserve_band_selection_gates",
    "execution",
    "payload_sha256",
}
_CURVE_FIELDS = {"runner", "episodes", "aggregate", "policy_counts", "cost"}
_EPISODE_EXTRA_FIELDS = {
    "legacy_final_gauge_failure",
    "gauge_failure_signal",
    "policy_counts",
}
_GAUGE_FAILURE_SIGNAL = (
    "terminal GAME_OVER/terminated state, independent of "
    "same-tick final-gauge recovery"
)


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA_CHARS for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _integer(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _unique_object(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _invalid_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _read_json_bytes(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is not strict unique-key JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} root must be a mapping")
    return value


def _validate_embedded_identity(
    value: object, name: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    claimed = _sha256(value.get("sha256"), f"{name} identity")
    content = {key: item for key, item in value.items() if key != "sha256"}
    if _canonical_sha256(content) != claimed:
        raise ValueError(f"{name} embedded SHA-256 differs")
    return value


@dataclass(frozen=True, slots=True)
class Slice:
    mode: str
    snapshot: Any
    report: Mapping[str, Any]


def _load_slice(
    mode: str,
    path: Path,
    expected_sha256: str,
) -> Slice:
    expected = _sha256(expected_sha256, f"{mode} file")
    snapshot = r3d._snapshot_file(path, f"{mode} reserve-band slice")
    if snapshot.sha256 != expected:
        raise ValueError(f"{mode} file SHA-256 mismatch")
    payload = snapshot.path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != snapshot.sha256:
        raise RuntimeError(f"{mode} slice changed between snapshot and read")
    report = _read_json_bytes(payload, f"{mode} slice")
    if set(report) != _REPORT_FIELDS:
        raise ValueError(f"{mode} slice has unknown or missing report fields")
    claimed = _sha256(report["payload_sha256"], f"{mode} payload")
    content = {
        key: value for key, value in report.items() if key != "payload_sha256"
    }
    if _canonical_sha256(content) != claimed:
        raise ValueError(f"{mode} payload SHA-256 mismatch")
    if (
        report["schema"] != rb.REPORT_SCHEMA
        or report["development_only"] is not True
        or report["canonical_r3_evidence"] is not False
        or report["sealed_test_material_used"] is not False
    ):
        raise ValueError(f"{mode} slice weakens development-only boundaries")
    _validate_embedded_identity(report["source_identity"], f"{mode} source")
    return Slice(mode, snapshot, report)


def _common(slices: Sequence[Slice], field: str) -> Any:
    expected = slices[0].report[field]
    if any(value.report[field] != expected for value in slices[1:]):
        raise ValueError(f"slice {field} identities differ")
    return expected


def _episode(record: Mapping[str, Any]) -> Any:
    conversion_value = record.get("conversion")
    if not isinstance(conversion_value, Mapping):
        raise ValueError("episode conversion manifest is missing")
    metric_fields = (
        "frames",
        "survival_ticks",
        "requested_shots",
        "shots_fired",
        "shots_hit",
        "projectile_hit_events",
        "chain_joins",
        "clears",
        "rotten",
        "ejected",
        "invalid_actions",
    )
    try:
        conversion = r3d.SteeringConversionMetrics(
            **{
                name: _integer(
                    conversion_value[name],
                    f"conversion {name}",
                    minimum=0,
                )
                for name in metric_fields
            },
            final_score=_integer(
                conversion_value["final_score"], "conversion final_score"
            ),
            terminated=conversion_value["terminated"],
            truncated=conversion_value["truncated"],
        )
        if (
            type(conversion.terminated) is not bool
            or type(conversion.truncated) is not bool
        ):
            raise TypeError("conversion terminal flags must be booleans")
        episode = r3d.EpisodeResult(
            str(record["policy"]),
            _integer(record["seed"], "episode seed", minimum=0),
            _integer(record["decisions"], "episode decisions", minimum=0),
            _integer(
                record["primitive_actions"],
                "episode primitive_actions",
                minimum=0,
            ),
            _integer(
                record["highest_chain"], "episode highest_chain", minimum=0
            ),
            _integer(
                record["qualifying_clears"],
                "episode qualifying_clears",
                minimum=0,
            ),
            _integer(record["final_level"], "episode final_level", minimum=0),
            _integer(record["final_gauge"], "episode final_gauge", minimum=0),
            _integer(record["gauge_max"], "episode gauge_max", minimum=1),
            _integer(
                record["archive_insertions"],
                "episode archive_insertions",
                minimum=0,
            ),
            _integer(
                record["archive_rejections"],
                "episode archive_rejections",
                minimum=0,
            ),
            conversion,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("episode row cannot reconstruct typed metrics") from error
    if conversion.manifest() != dict(conversion_value):
        raise ValueError("episode conversion derived rates are inconsistent")
    base = episode.manifest()
    expected_fields = set(base) | _EPISODE_EXTRA_FIELDS
    if set(record) != expected_fields:
        raise ValueError("episode row has unknown or missing fields")
    for name, value in base.items():
        if name != "gauge_failure" and record[name] != value:
            raise ValueError(f"episode field {name} is inconsistent")
    if (
        record["legacy_final_gauge_failure"] is not base["gauge_failure"]
        or record["gauge_failure"] is not conversion.terminated
        or record["gauge_failure_signal"] != _GAUGE_FAILURE_SIGNAL
    ):
        raise ValueError("episode gauge-failure semantics are inconsistent")
    counts = record["policy_counts"]
    if not isinstance(counts, Mapping) or any(
        not isinstance(name, str)
        or type(count) is not int
        or count < 0
        for name, count in counts.items()
    ):
        raise ValueError("episode policy counts are invalid")
    return episode


def _recompute_curve(
    curve: Mapping[str, Any],
    *,
    seeds: Sequence[int],
    horizon: int,
) -> None:
    if set(curve) != _CURVE_FIELDS:
        raise ValueError("curve has unknown or missing fields")
    records = curve["episodes"]
    if not isinstance(records, list) or not records:
        raise ValueError("curve episode rows are missing")
    if [record.get("seed") for record in records] != list(seeds):
        raise ValueError("curve episode rows do not match the exact seed order")
    episodes = tuple(_episode(record) for record in records)
    if any(
        episode.conversion.survival_ticks > horizon
        or (
            episode.conversion.truncated
            and episode.conversion.survival_ticks != horizon
        )
        for episode in episodes
    ):
        raise ValueError("curve episode violates horizon censoring")
    aggregate = rb._augment_aggregate(episodes, r3d._aggregate(episodes))
    aggregate["gauge_failures"] = sum(
        bool(record["gauge_failure"]) for record in records
    )
    if aggregate != curve["aggregate"]:
        raise ValueError("curve aggregate does not recompute from episode rows")
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(record["policy_counts"])
    if dict(sorted(counts.items())) != curve["policy_counts"]:
        raise ValueError("curve policy counts do not sum from episode rows")
    cost = curve["cost"]
    if not isinstance(cost, Mapping) or set(cost) != {
        "wall_seconds",
        "cpu_seconds",
        "episodes",
        "wall_seconds_per_episode",
    }:
        raise ValueError("curve cost manifest is malformed")
    wall = cost["wall_seconds"]
    cpu = cost["cpu_seconds"]
    if (
        isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or float(wall) < 0.0
        or isinstance(cpu, bool)
        or not isinstance(cpu, (int, float))
        or not math.isfinite(float(cpu))
        or float(cpu) < 0.0
        or type(cost["episodes"]) is not int
        or cost["episodes"] != len(episodes)
        or not math.isclose(
            float(cost["wall_seconds_per_episode"]),
            float(wall) / len(episodes),
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("curve cost values are inconsistent")


def _validate_seed_registry(
    registry: Mapping[str, Any],
    active_suite: Mapping[str, Any],
) -> None:
    overlaps = registry.get("pairwise_overlaps")
    if (
        set(registry) != {
            "suites",
            "pairwise_overlaps",
            "pairwise_disjoint",
        }
        or registry["pairwise_disjoint"] is not True
        or not isinstance(overlaps, Mapping)
        or any(overlaps.values())
    ):
        raise ValueError("seed registry is not pairwise disjoint")
    suites = registry["suites"]
    if not isinstance(suites, Mapping):
        raise ValueError("seed suites are missing")
    for name, suite in suites.items():
        if (
            not isinstance(name, str)
            or not isinstance(suite, Mapping)
            or set(suite) != {"label", "seeds", "sha256"}
            or suite["sha256"]
            != _canonical_sha256(
                {"label": suite["label"], "seeds": suite["seeds"]}
            )
        ):
            raise ValueError("seed suite identity is inconsistent")
    if set(active_suite) != {
        "name",
        "label",
        "suite_sha256",
        "seeds",
        "seed_count",
    }:
        raise ValueError("active suite has unknown or missing fields")
    name = active_suite.get("name")
    if name not in suites:
        raise ValueError("active suite is absent from the seed registry")
    suite = suites[name]
    count = active_suite.get("seed_count")
    if (
        type(count) is not int
        or count < 1
        or active_suite.get("label") != suite["label"]
        or active_suite.get("suite_sha256") != suite["sha256"]
        or active_suite.get("seeds") != suite["seeds"][:count]
        or len(set(active_suite["seeds"])) != count
    ):
        raise ValueError("active suite does not bind the registry prefix")


def _protocol_common(protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "modes",
        "horizons",
        "runway_ticks",
        "query_stride_shots",
        "maximum_search_queries_per_episode",
        "sampling_scope",
        "endpoint_semantics",
        "preregistered_full_suite_protocol",
        "exploratory_slice",
        "catastrophic_regression_rule",
        "smoke_override",
    }
    if set(protocol) != expected:
        raise ValueError("slice protocol has unknown or missing fields")
    return {
        key: value
        for key, value in protocol.items()
        if key
        not in {
            "modes",
            "preregistered_full_suite_protocol",
            "exploratory_slice",
        }
    }


def _composed_gate(
    values: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, object]:
    gate = rb._selection_gate(values, config)
    if gate is None:
        raise RuntimeError("four-mode composition did not produce a selection gate")
    base = rb._paired(
        "reserve_band",
        values["reserve_band"],
        "base_v5",
        values["base_v5"],
        minimum_tick_loss=int(config["ab"]["catastrophic_minimum_tick_loss"]),
        survival_ratio=float(config["ab"]["catastrophic_survival_ratio"]),
    )
    checks = {
        **dict(gate["checks"]),
        "zero_new_catastrophic_regressions_vs_base_v5": (
            int(base["catastrophic_regressions"]) == 0
        ),
    }
    return {
        **dict(gate),
        "passed_without_base_v5_safety": bool(gate["passed"]),
        "passed": all(checks.values()),
        "checks": checks,
        "base_v5_catastrophic_regressions": int(
            base["catastrophic_regressions"]
        ),
    }


def _execution_environment(value: Mapping[str, Any]) -> dict[str, object]:
    expected = {
        "wall_seconds",
        "cpu_seconds",
        "python",
        "torch",
        "numpy",
        "torch_threads",
    }
    if set(value) != expected:
        raise ValueError("slice execution manifest is malformed")
    for name in ("wall_seconds", "cpu_seconds"):
        number = value[name]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) < 0.0
        ):
            raise ValueError("slice execution cost is invalid")
    return {
        key: item
        for key, item in value.items()
        if key not in {"wall_seconds", "cpu_seconds"}
    }


def _compose(
    slices: Sequence[Slice],
    *,
    config: Mapping[str, Any],
    config_snapshot: Any,
    merger_source_sha256: str,
    started_wall: float,
    started_cpu: float,
) -> dict[str, object]:
    if tuple(value.mode for value in slices) != MODES:
        raise ValueError("slice order does not match the exact mode partition")
    common_fields = (
        "schema",
        "development_only",
        "canonical_r3_evidence",
        "sealed_test_material_used",
        "source_identity",
        "config",
        "runtime",
        "base_policy",
        "seed_registry",
        "active_suite",
        "candidate_vocabulary_sha256",
        "development_checks",
    )
    common = {field: _common(slices, field) for field in common_fields}
    expected_runtime = {
        "path": str(
            rb._path(str(config["trusted_runtime"])).resolve(strict=True)
        ),
        "sha256": str(config["trusted_runtime_sha256"]),
        "backend": "portable",
    }
    expected_base = {
        "path": str(
            rb._path(
                str(config["base_policy"]["checkpoint"])
            ).resolve(strict=True)
        ),
        "sha256": str(config["base_policy"]["sha256"]),
        "identity": "frozen-r3d-v5",
    }
    if (
        common["config"]
        != {
            "path": str(config_snapshot.path),
            "sha256": config_snapshot.sha256,
        }
        or common["source_identity"] != rb._source_identity(config_snapshot.path)
        or common["runtime"] != expected_runtime
        or common["base_policy"] != expected_base
        or common["seed_registry"] != rb._seed_registry(config)
        or common["development_checks"] != rb._development_check_result()
    ):
        raise ValueError("slice common identity differs from current config/source")
    _validate_seed_registry(common["seed_registry"], common["active_suite"])
    seeds = tuple(int(value) for value in common["active_suite"]["seeds"])

    protocols: list[dict[str, Any]] = []
    for value in slices:
        protocol = value.report["protocol"]
        if (
            not isinstance(protocol, Mapping)
            or protocol.get("modes") != [value.mode]
            or protocol.get("preregistered_full_suite_protocol") is not False
            or protocol.get("exploratory_slice") is not True
        ):
            raise ValueError(f"{value.mode} is not an honest one-mode slice")
        protocols.append(_protocol_common(protocol))
    if any(protocol != protocols[0] for protocol in protocols[1:]):
        raise ValueError("slice protocol identities differ beyond mode partition")
    horizons = tuple(int(value) for value in protocols[0]["horizons"])
    if (
        not horizons
        or tuple(sorted(set(horizons))) != horizons
        or any(value < 1 for value in horizons)
    ):
        raise ValueError("composed horizons are invalid")
    suite_name = str(common["active_suite"]["name"])
    expected_count = (
        int(config["ab"]["long_probe_seed_count"])
        if suite_name == "long"
        else int(config["ab"]["seed_count_per_suite"])
    )
    expected_horizons = tuple(
        int(value)
        for value in (
            config["ab"]["long_probe_horizons"]
            if suite_name == "long"
            else config["ab"]["horizons"]
        )
    )
    coverage_matches_configured_grid = (
        len(seeds) == expected_count and horizons == expected_horizons
    )

    environment = _execution_environment(slices[0].report["execution"])
    if any(
        _execution_environment(value.report["execution"]) != environment
        for value in slices[1:]
    ):
        raise ValueError("slice execution environments differ")
    teachers: dict[str, object] = {}
    curves: dict[str, dict[str, object]] = {
        str(horizon): {} for horizon in horizons
    }
    runners: dict[str, object] = {}
    for value in slices:
        expected_teachers: dict[str, object] = {}
        if value.mode != "base_v5":
            teacher = rb._teacher(
                config,
                value.mode,
                runway_ticks=int(protocols[0]["runway_ticks"]),
            )
            expected_teachers[value.mode] = {
                "identity": teacher.identity_manifest(),
                "sha256": teacher.sha256,
            }
            teachers[value.mode] = expected_teachers[value.mode]
        if value.report["teachers"] != expected_teachers:
            raise ValueError(f"{value.mode} teacher identity differs")
        if (
            set(value.report["curves"]) != set(curves)
            or set(value.report["paired"]) != set(curves)
            or any(value.report["paired"][key] for key in curves)
            or value.report["reserve_band_selection_gates"] != {}
        ):
            raise ValueError(f"{value.mode} slice contains non-partition evidence")
        for horizon in horizons:
            key = str(horizon)
            partition = value.report["curves"][key]
            if not isinstance(partition, Mapping) or set(partition) != {value.mode}:
                raise ValueError(f"{value.mode} curve is not exactly partitioned")
            curve = partition[value.mode]
            _recompute_curve(curve, seeds=seeds, horizon=horizon)
            prior = runners.setdefault(key, curve["runner"])
            if curve["runner"] != prior:
                raise ValueError(f"runner identity differs at horizon {horizon}")
            curves[key][value.mode] = curve
    expected_vocabulary = rb.geometry_candidate_vocabulary_sha256(
        rb._teacher(
            config,
            "reserve_band",
            runway_ticks=int(protocols[0]["runway_ticks"]),
        ).config.candidate_config
    )
    if common["candidate_vocabulary_sha256"] != expected_vocabulary:
        raise ValueError("candidate vocabulary differs from current config")

    paired: dict[str, object] = {}
    gates: dict[str, object] = {}
    horizon_costs: dict[str, object] = {}
    threshold = int(config["ab"]["catastrophic_minimum_tick_loss"])
    ratio = float(config["ab"]["catastrophic_survival_ratio"])
    for horizon in horizons:
        key = str(horizon)
        values = curves[key]
        paired[key] = {
            f"reserve_band_vs_{comparator}": rb._paired(
                "reserve_band",
                values["reserve_band"],
                comparator,
                values[comparator],
                minimum_tick_loss=threshold,
                survival_ratio=ratio,
            )
            for comparator in ("base_v5", "score_first", "pure_gauge")
        }
        gates[key] = _composed_gate(values, config)
        horizon_costs[key] = {
            "summed_policy_wall_seconds": sum(
                float(values[mode]["cost"]["wall_seconds"]) for mode in MODES
            ),
            "summed_policy_cpu_seconds": sum(
                float(values[mode]["cost"]["cpu_seconds"]) for mode in MODES
            ),
            "semantics": (
                "arithmetic sum of independently recorded policy-slice costs; "
                "not observed elapsed wall time"
            ),
        }

    slice_costs = {
        value.mode: {
            "wall_seconds": float(value.report["execution"]["wall_seconds"]),
            "cpu_seconds": float(value.report["execution"]["cpu_seconds"]),
        }
        for value in slices
    }
    common_protocol = {
        **protocols[0],
        "modes": list(MODES),
        "preregistered_full_suite_protocol": False,
        "exploratory_slice": True,
        "partitioned_composition": True,
        "composition_covers_all_modes": True,
        "coverage_matches_configured_suite_grid": (
            coverage_matches_configured_grid
        ),
        "partition_identity_rule": (
            "four independently executed one-mode slices with exact common "
            "source, inputs, suite, seeds, horizons, protocol, and runners"
        ),
    }
    content = {
        "schema": REPORT_SCHEMA,
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
        "composition": {
            "version": MERGER_VERSION,
            "partitioned": True,
            "single_joint_execution": False,
            "claim": (
                "identity-bound composition of four independent development "
                "slices; not a preregistered single-process full-suite run"
            ),
            "merger_source": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": merger_source_sha256,
            },
            "inputs": {
                value.mode: {
                    "path": str(value.snapshot.path),
                    "file_sha256": value.snapshot.sha256,
                    "payload_sha256": value.report["payload_sha256"],
                }
                for value in slices
            },
        },
        "source_identity": common["source_identity"],
        "config": common["config"],
        "runtime": common["runtime"],
        "base_policy": common["base_policy"],
        "seed_registry": common["seed_registry"],
        "active_suite": common["active_suite"],
        "protocol": common_protocol,
        "teachers": teachers,
        "candidate_vocabulary_sha256": common[
            "candidate_vocabulary_sha256"
        ],
        "development_checks": common["development_checks"],
        "runners": runners,
        "curves": curves,
        "paired": paired,
        "reserve_band_selection_gates": gates,
        "cost": {
            "slice_execution": slice_costs,
            "summed_slice_wall_seconds": sum(
                value["wall_seconds"] for value in slice_costs.values()
            ),
            "summed_slice_cpu_seconds": sum(
                value["cpu_seconds"] for value in slice_costs.values()
            ),
            "per_horizon": horizon_costs,
            "summed_wall_semantics": (
                "arithmetic sum of independent slice wall durations; no "
                "elapsed wall duration is inferred"
            ),
        },
        "merge_execution": {
            "wall_seconds": time.monotonic() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
            "python": platform.python_version(),
        },
    }
    return {**content, "payload_sha256": _canonical_sha256(content)}


def _synthetic_record(
    mode: str,
    seed: int,
    *,
    score: int,
    survival: int,
    final_gauge: int,
    terminated: bool,
) -> dict[str, object]:
    conversion = r3d.SteeringConversionMetrics(
        frames=10,
        survival_ticks=survival,
        requested_shots=2,
        shots_fired=2,
        shots_hit=1,
        projectile_hit_events=1,
        chain_joins=1,
        clears=1,
        rotten=int(terminated),
        ejected=0,
        invalid_actions=0,
        final_score=score,
        terminated=terminated,
        truncated=not terminated,
    )
    episode = r3d.EpisodeResult(
        mode,
        seed,
        8,
        10,
        2,
        1,
        1,
        final_gauge,
        40_000,
        0,
        0,
        conversion,
    )
    value = episode.manifest()
    value["legacy_final_gauge_failure"] = value["gauge_failure"]
    value["gauge_failure"] = terminated
    value["gauge_failure_signal"] = _GAUGE_FAILURE_SIGNAL
    value["policy_counts"] = (
        {} if mode == "base_v5" else {"search_queries": 1}
    )
    return value


def _synthetic_curve(
    mode: str,
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    episodes = tuple(_episode(value) for value in records)
    aggregate = rb._augment_aggregate(episodes, r3d._aggregate(episodes))
    aggregate["gauge_failures"] = sum(
        bool(value["gauge_failure"]) for value in records
    )
    counts: Counter[str] = Counter()
    for value in records:
        counts.update(value["policy_counts"])
    return {
        "runner": {"synthetic": True},
        "episodes": list(records),
        "aggregate": aggregate,
        "policy_counts": dict(sorted(counts.items())),
        "cost": {
            "wall_seconds": 1.0,
            "cpu_seconds": 0.5,
            "episodes": len(records),
            "wall_seconds_per_episode": 1.0 / len(records),
        },
    }


def _self_check() -> dict[str, object]:
    seeds = (11, 22)
    values: dict[str, object] = {}
    for mode in MODES:
        rows = [
            _synthetic_record(
                mode,
                seeds[0],
                score=20 if mode == "reserve_band" else 10,
                survival=2_000,
                final_gauge=5_000,
                terminated=False,
            ),
            _synthetic_record(
                mode,
                seeds[1],
                score=20 if mode == "reserve_band" else 10,
                survival=500 if mode == "reserve_band" else 2_000,
                final_gauge=5_000,
                terminated=mode == "reserve_band",
            ),
        ]
        values[mode] = _synthetic_curve(mode, rows)
        _recompute_curve(values[mode], seeds=seeds, horizon=2_000)
    config = {
        "ab": {
            "catastrophic_minimum_tick_loss": 1_000,
            "catastrophic_survival_ratio": 0.5,
        },
        "selection": {
            "candidate": "reserve_band",
            "primary_comparator": "pure_gauge",
            "requires_strictly_higher_median_score": True,
            "maximum_extra_gauge_failures": 0,
            "minimum_survival_p10_ratio": 0.9,
            "secondary_comparator": "score_first",
            "secondary_requires_no_more_gauge_failures": True,
            "requires_zero_new_catastrophic_regressions": True,
        },
    }
    base = rb._paired(
        "reserve_band",
        values["reserve_band"],
        "base_v5",
        values["base_v5"],
        minimum_tick_loss=1_000,
        survival_ratio=0.5,
    )
    gate = _composed_gate(values, config)
    if (
        base["catastrophic_regressions"] != 1
        or gate["base_v5_catastrophic_regressions"] != 1
        or gate["passed"] is not False
    ):
        raise RuntimeError("synthetic base catastrophe safety check failed")
    return {
        "status": "PASS",
        "version": MERGER_VERSION,
        "checks": [
            "typed episode and aggregate reconstruction",
            "policy-count summation",
            "reserve-band pairing through oracle helper",
            "base-v5 catastrophic regression safety gate",
        ],
        "synthetic_catastrophic_regressions": 1,
        "merger_source_sha256": r3d._file_sha256(Path(__file__).resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--config", type=Path, default=rb.DEFAULT_CONFIG)
    for mode in MODES:
        option = mode.replace("_", "-")
        parser.add_argument(f"--{option}", type=Path)
        parser.add_argument(f"--{option}-sha256")
    parser.add_argument("--result-out", type=Path)
    args = parser.parse_args()

    if args.self_check:
        print(json.dumps(_self_check(), sort_keys=True, indent=2))
        return
    missing = [
        name
        for mode in MODES
        for name in (
            mode,
            f"{mode}_sha256",
        )
        if getattr(args, name) is None
    ]
    if args.result_out is None:
        missing.append("result_out")
    if missing:
        parser.error("four slice paths, file SHA-256s, and --result-out are required")

    started_wall = time.monotonic()
    started_cpu = time.process_time()
    config_snapshot = r3d._snapshot_file(args.config, "reserve-band merge config")
    if config_snapshot.path != rb.DEFAULT_CONFIG.resolve(strict=True):
        parser.error("--config must be the identity-bound debt-aware config")
    config = rb._load_config(config_snapshot)
    merger_source = r3d._snapshot_file(
        Path(__file__).resolve(), "reserve-band merger source"
    )
    slices = tuple(
        _load_slice(
            mode,
            getattr(args, mode),
            getattr(args, f"{mode}_sha256"),
        )
        for mode in MODES
    )
    output = rb._safe_output(config, args.result_out)
    if any(output == value.snapshot.path for value in slices):
        parser.error("--result-out must not replace an input slice")
    report = _compose(
        slices,
        config=config,
        config_snapshot=config_snapshot,
        merger_source_sha256=merger_source.sha256,
        started_wall=started_wall,
        started_cpu=started_cpu,
    )
    rb._require_source_identity(report["source_identity"], config_snapshot.path)
    r3d._require_unchanged(config_snapshot, "reserve-band merge config")
    r3d._require_unchanged(merger_source, "reserve-band merger source")
    for value in slices:
        r3d._require_unchanged(value.snapshot, f"{value.mode} reserve-band slice")
    file_sha256 = r3e._atomic_write_json(output, report, overwrite=False)
    print(
        json.dumps(
            {
                "result": str(output),
                "file_sha256": file_sha256,
                "payload_sha256": report["payload_sha256"],
                "merger_source_sha256": merger_source.sha256,
                "partitioned_composition": True,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
