"""Deterministic final reporting for the development-only R3G campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPORT_NAME = "campaign-report.json"
EVIDENCE_NAME = "evidence-packet.md"
INDEX_NAME = "artifact-index.json"
INDEX_SHA_NAME = "artifact-index.sha256"
_FINAL_NAMES = frozenset(
    {REPORT_NAME, EVIDENCE_NAME, INDEX_NAME, INDEX_SHA_NAME}
)
_SPLIT_ORDER = (
    "teacher-screen",
    "student-screen",
    "dagger-train",
    "barrier-calibration",
    "barrier-heldout",
    "barrier-stress",
)


def _json_bytes(value: object, *, pretty: bool) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2 if pretty else None,
            separators=None if pretty else (",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _items(value: object) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return list(value)
    return []


def _number(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:,.6g}"
    return str(value)


def _cell(value: object) -> str:
    if isinstance(value, (Mapping, list, tuple)):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return _number(value).replace("|", "\\|").replace("\n", " ")


def _distribution(value: object) -> str:
    data = _mapping(value)
    fields = []
    for name in ("minimum", "p10", "median", "maximum"):
        if name in data:
            fields.append(f"{name}={_number(data[name])}")
    return ", ".join(fields) if fields else "not recorded"


def _seed_from_episode(value: object) -> int | None:
    episode = _mapping(value)
    seed = episode.get("seed")
    return seed if type(seed) is int else None


def _fallback_seed_schedule(
    report: Mapping[str, Any],
) -> dict[str, list[dict[str, object]]]:
    output: dict[str, set[int]] = {}

    def add(split: str, values: object) -> None:
        for value in _items(values):
            seed = _seed_from_episode(value)
            if seed is not None:
                output.setdefault(split, set()).add(seed)

    teacher = _mapping(report.get("teacher_screen"))
    add(
        "teacher-screen",
        _mapping(teacher.get("baseline")).get("episodes"),
    )
    calibration = _mapping(report.get("barrier_calibration"))
    for key in ("fit_seeds", "calibration_seeds", "seeds"):
        for seed in _items(calibration.get(key)):
            if type(seed) is int:
                output.setdefault("barrier-calibration", set()).add(seed)
    freeze = _mapping(report.get("heldout_freeze"))
    training = _mapping(freeze.get("training_manifest"))
    for key, split in (
        ("calibration_seeds", "barrier-calibration"),
        ("dagger_seeds", "dagger-train"),
    ):
        for seed in _items(training.get(key)):
            if type(seed) is int:
                output.setdefault(split, set()).add(seed)
    heldout = _mapping(report.get("barrier_heldout"))
    for seed in _items(heldout.get("seeds")):
        if type(seed) is int:
            output.setdefault("barrier-heldout", set()).add(seed)
    stress = _mapping(report.get("barrier_stress"))
    for seed in _items(stress.get("seeds")):
        if type(seed) is int:
            output.setdefault("barrier-stress", set()).add(seed)
    student = _mapping(report.get("student_screen"))
    add(
        "student-screen",
        _mapping(student.get("baseline")).get("episodes"),
    )
    return {
        split: [
            {"index": index, "seed": seed, "seed_hex": f"0x{seed:08X}"}
            for index, seed in enumerate(sorted(seeds))
        ]
        for split, seeds in output.items()
    }


def _append_scope(lines: list[str], report: Mapping[str, Any]) -> None:
    lines.extend(["## Scope and identities", ""])
    for label, key in (
        ("Development only", "development_only"),
        ("Sealed material used", "sealed_test_material_used"),
        ("Canonical evidence", "canonical_r3_evidence"),
        ("Authorization material used", "authorization_material_used"),
        ("Parent winner materialized", "parent_winner_materialized"),
        ("Screens are allocation only", "screens_are_allocation_only"),
    ):
        if key in report:
            lines.append(f"- {label}: `{_cell(report[key])}`")
    source = _mapping(report.get("source_identity"))
    identities = (
        ("Source", source.get("sha256")),
        ("Protocol", report.get("protocol_sha256")),
        ("Runtime", report.get("runtime_sha256")),
        ("Frozen-v5 checkpoint", report.get("checkpoint_sha256")),
        ("Frozen-v5 policy", report.get("base_policy_identity_sha256")),
        ("Corrected joint source", report.get("joint_source_sha256")),
        ("Corrected joint benchmark", report.get("joint_benchmark_sha256")),
        ("Configuration", report.get("config_sha256")),
        ("Seed manifest", report.get("seed_manifest_sha256")),
        ("Preregistration", report.get("preregistration_sha256")),
    )
    for label, value in identities:
        if value not in (None, ""):
            lines.append(f"- {label}: `{_cell(value)}`")
    lines.append("")


def _append_seeds(lines: list[str], report: Mapping[str, Any]) -> None:
    schedule = _mapping(report.get("exact_seed_schedule"))
    if not schedule:
        schedule = _fallback_seed_schedule(report)
    lines.extend(["## Exact seed schedule", ""])
    derivation = report.get("seed_derivation")
    if derivation not in (None, ""):
        lines.append(f"- Derivation: `{_cell(derivation)}`")
    lines.append(
        "- Pairing: every reached comparison uses frozen-v5 on the same "
        "seed and horizon."
    )
    if not schedule:
        lines.extend(
            ["- No seed schedule was retained in the reached report.", ""]
        )
        return
    ordered = list(_SPLIT_ORDER) + sorted(
        set(schedule) - set(_SPLIT_ORDER)
    )
    for split in ordered:
        rows = _items(schedule.get(split))
        if not rows:
            continue
        rows.sort(
            key=lambda value: (
                int(_mapping(value).get("index", 0)),
                int(_mapping(value).get("seed", 0)),
            )
        )
        horizons = sorted(
            {
                int(row["horizon"])
                for value in rows
                if (row := _mapping(value))
                and type(row.get("horizon")) is int
            }
        )
        seeds = ", ".join(
            str(
                _mapping(value).get(
                    "seed_hex",
                    f"0x{int(_mapping(value).get('seed', 0)):08X}",
                )
            )
            for value in rows
        )
        horizon = (
            f"; horizon={','.join(str(value) for value in horizons)}"
            if horizons
            else ""
        )
        lines.append(
            f"- **{split}** ({len(rows)} seeds{horizon}): {seeds}"
        )
    lines.append("")


def _append_gates(lines: list[str], report: Mapping[str, Any]) -> None:
    lines.extend(["## Gates", ""])
    gates = _mapping(report.get("gates"))
    if not gates:
        lines.append("No gate result was reached.")
    else:
        for name in sorted(gates):
            lines.append(f"- {name}: `{_cell(gates[name])}`")
    lines.append("")


def _command_text(value: object) -> str:
    command = _items(value)
    if command:
        return shlex.join(str(part) for part in command)
    return str(value)


def _check_result(row: Mapping[str, Any]) -> str:
    if "returncode" in row:
        result = f"returncode={_number(row['returncode'])}"
    elif "passed" in row:
        result = f"passed={_number(row['passed'])}"
    elif "result" in row:
        result = str(row["result"])
    else:
        result = "not recorded"
    detail = []
    for key in ("stdout", "stderr", "detail", "reason"):
        value = str(row.get(key, "")).strip()
        if value:
            detail.append(f"{key}={value}")
    if detail:
        result += "; " + "; ".join(detail)
    return result


def _append_development_checks(
    lines: list[str],
    checks: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    lines.extend(["### Development checks", ""])
    for label, value in (
        ("Passed", checks.get("passed")),
        ("Logical CPU", checks.get("logical_cpu")),
        ("CPU seconds", checks.get("cpu_seconds")),
        ("Source SHA-256", checks.get("source_sha256")),
        (
            "Validated source SHA-256",
            _mapping(report.get("source_identity")).get("sha256"),
        ),
    ):
        if value is not None:
            lines.append(f"- {label}: `{_cell(value)}`")
    for key in sorted(checks):
        if key.endswith("sha256") and key != "source_sha256":
            lines.append(f"- {key}: `{_cell(checks[key])}`")
    commands = [_mapping(value) for value in _items(checks.get("commands"))]
    if commands:
        lines.extend(
            [
                "",
                "| Command | Result | Child CPU seconds | Wall seconds |",
                "|---|---|---:|---:|",
            ]
        )
        for row in commands:
            lines.append(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        _command_text(row.get("command", row.get("name", ""))),
                        _check_result(row),
                        row.get("child_cpu_seconds", row.get("cpu_seconds", "")),
                        row.get("wall_seconds", ""),
                    )
                )
                + " |"
            )
    named_checks = [
        _mapping(value) for value in _items(checks.get("checks"))
    ]
    if named_checks:
        lines.extend(
            [
                "",
                "| Check | Result | Expected | Actual |",
                "|---|---|---|---|",
            ]
        )
        for row in named_checks:
            lines.append(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        row.get("name", row.get("check", "")),
                        _check_result(row),
                        row.get("expected", ""),
                        row.get("actual", ""),
                    )
                )
                + " |"
            )
    lines.append("")


def _append_sentinels(
    lines: list[str], sentinel: Mapping[str, Any]
) -> None:
    lines.extend(["### Frozen-v5 sentinel reproduction", ""])
    for label, key in (("Passed", "passed"), ("Source", "source")):
        if key in sentinel:
            lines.append(f"- {label}: `{_cell(sentinel[key])}`")
    checks = [_mapping(value) for value in _items(sentinel.get("checks"))]
    if checks:
        lines.extend(
            [
                "",
                "| Seed | Metric | Expected | Actual | Matched |",
                "|---:|---|---:|---:|:---:|",
            ]
        )
        for check in sorted(
            checks,
            key=lambda value: int(value.get("seed", 0)),
        ):
            expected = _mapping(check.get("expected"))
            actual = _mapping(check.get("actual"))
            for metric in sorted(set(expected) | set(actual)):
                lines.append(
                    "| "
                    + " | ".join(
                        _cell(value)
                        for value in (
                            check.get("seed", ""),
                            metric,
                            expected.get(metric, ""),
                            actual.get(metric, ""),
                            expected.get(metric) == actual.get(metric),
                        )
                    )
                    + " |"
                )
    lines.append("")


def _append_validation(
    lines: list[str], report: Mapping[str, Any]
) -> None:
    checks = _mapping(report.get("development_checks"))
    sentinel = _mapping(report.get("sentinel_reproduction"))
    if not checks and not sentinel:
        return
    lines.extend(["## Development validation", ""])
    if checks:
        _append_development_checks(lines, checks, report)
    if sentinel:
        _append_sentinels(lines, sentinel)


def _suite_entries(
    report: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    output: list[tuple[str, Mapping[str, Any]]] = []
    teacher = _mapping(report.get("teacher_screen"))
    if teacher:
        output.extend(
            (
                ("teacher-screen/frozen-v5", _mapping(teacher.get("baseline"))),
                ("teacher-screen/exact-teacher", _mapping(teacher.get("teacher"))),
            )
        )
    heldout = _mapping(report.get("barrier_heldout"))
    if heldout:
        output.append(("barrier-heldout/student", heldout))
    stress = _mapping(report.get("barrier_stress"))
    if stress:
        output.append(("barrier-stress/model-barrier", stress))
    student = _mapping(report.get("student_screen"))
    if student:
        output.extend(
            (
                ("student-screen/frozen-v5", _mapping(student.get("baseline"))),
                ("student-screen/student", _mapping(student.get("student"))),
            )
        )
    return [(name, value) for name, value in output if value]


def _append_suite_summary(
    lines: list[str], label: str, suite: Mapping[str, Any]
) -> None:
    aggregate = _mapping(suite.get("aggregate"))
    episodes = _items(suite.get("episodes"))
    count = aggregate.get("episodes", len(episodes))
    lines.append(f"### {label}")
    lines.append("")
    lines.append(f"- Episodes: {_number(count)}")
    horizons = sorted(
        {
            int(episode["horizon_ticks"])
            for value in episodes
            if (episode := _mapping(value))
            and type(episode.get("horizon_ticks")) is int
        }
    )
    if horizons:
        lines.append(
            "- Horizon ticks: "
            + ", ".join(_number(value) for value in horizons)
        )
    if episodes:
        lines.append(
            "- Terminal episodes: "
            + _number(
                sum(
                    bool(_mapping(value).get("terminated", False))
                    for value in episodes
                )
            )
        )
    if aggregate:
        lines.append(f"- Score: {_distribution(aggregate.get('score'))}")
        lines.append(
            "- Survival ticks: "
            f"{_distribution(aggregate.get('survival_ticks'))}"
        )
        for label_text, key in (
            ("Final gauge", "final_gauge"),
            ("Final level", "final_level"),
        ):
            if key in aggregate:
                lines.append(
                    f"- {label_text}: {_distribution(aggregate.get(key))}"
                )
        for label_text, key in (
            ("Gauge failures", "gauge_failures"),
            ("Qualifying clears", "qualifying_clears"),
            ("Rotten events", "rotten_events"),
            ("Invalid actions", "invalid_actions"),
        ):
            if key in aggregate:
                lines.append(f"- {label_text}: {_number(aggregate[key])}")
    lines.append("")


def _delta_summary(rows: list[Any], key: str) -> str:
    values = [
        float(row[key])
        for value in rows
        if (row := _mapping(value))
        and isinstance(row.get(key), (int, float))
        and not isinstance(row.get(key), bool)
    ]
    if not values:
        return "not recorded"
    return (
        f"minimum={_number(min(values))}, "
        f"median={_number(float(statistics.median(values)))}, "
        f"maximum={_number(max(values))}"
    )


def _append_paired(
    lines: list[str], label: str, paired_value: object
) -> None:
    paired = _mapping(paired_value)
    if not paired:
        return
    rows = _items(paired.get("rows"))
    lines.append(f"### {label} paired outcomes")
    lines.append("")
    for label_text, key in (
        ("New terminals", "new_terminals"),
        ("New gauge failures", "new_gauge_failures"),
        ("Material survival regressions", "material_regressions"),
        ("Catastrophic regressions", "catastrophic_regressions"),
    ):
        if key in paired:
            lines.append(f"- {label_text}: {_number(paired[key])}")
    lines.append(f"- Score delta: {_delta_summary(rows, 'score_delta')}")
    lines.append(
        f"- Survival delta: {_delta_summary(rows, 'survival_delta')}"
    )
    failures = [
        _mapping(value)
        for value in rows
        if any(
            bool(_mapping(value).get(key, False))
            for key in (
                "new_terminal",
                "new_gauge_failure",
                "material_survival_regression",
                "catastrophic_paired_regression",
            )
        )
    ]
    if failures:
        lines.extend(
            [
                "",
                "| Seed | Terminal | Gauge | Material | Catastrophic | "
                "Score Δ | Survival Δ |",
                "|---:|:---:|:---:|:---:|:---:|---:|---:|",
            ]
        )
        for row in sorted(failures, key=lambda value: int(value["seed"])):
            lines.append(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        row.get("seed_hex", row.get("seed")),
                        row.get("new_terminal", False),
                        row.get("new_gauge_failure", False),
                        row.get("material_survival_regression", False),
                        row.get("catastrophic_paired_regression", False),
                        row.get("score_delta", ""),
                        row.get("survival_delta", ""),
                    )
                )
                + " |"
            )
    else:
        lines.append("- Paired failure seeds: none observed.")
    lines.append("")


def _append_performance(lines: list[str], report: Mapping[str, Any]) -> None:
    entries = _suite_entries(report)
    if not entries:
        return
    lines.extend(["## Score, survival, and paired failures", ""])
    for label, suite in entries:
        _append_suite_summary(lines, label, suite)
    teacher = _mapping(report.get("teacher_screen"))
    _append_paired(
        lines,
        "teacher-screen",
        _mapping(teacher.get("gate")).get("paired"),
    )
    student = _mapping(report.get("student_screen"))
    _append_paired(lines, "student-screen", student.get("paired"))


def _numeric_summary(values: Sequence[int | float]) -> str:
    if not values:
        return "not recorded"
    return (
        f"minimum={_number(min(values))}, "
        f"median={_number(float(statistics.median(values)))}, "
        f"maximum={_number(max(values))}"
    )


def _checkpoint_rows(
    suite: Mapping[str, Any],
) -> list[tuple[str, int, int, str, str]]:
    aggregate = _mapping(suite.get("aggregate"))
    reach = _mapping(aggregate.get("checkpoint_reach"))
    score = _mapping(aggregate.get("score_by_checkpoint"))
    carried = _mapping(
        aggregate.get("score_by_checkpoint_carried_terminal")
    )
    rows = []
    ticks = set(reach) | set(score) | set(carried)
    for tick in sorted(ticks, key=lambda value: int(value)):
        reached = _mapping(reach.get(tick))
        rows.append(
            (
                str(tick),
                int(reached.get("reached", 0)),
                int(reached.get("episodes", 0)),
                _distribution(score.get(tick)),
                _distribution(carried.get(tick)),
            )
        )
    if rows:
        return rows
    episodes = _items(suite.get("episodes"))
    ticks = sorted(
        {
            tick
            for value in episodes
            for tick in _mapping(_mapping(value).get("checkpoints"))
        },
        key=lambda value: int(value),
    )
    for tick in ticks:
        values = [
            _mapping(_mapping(value).get("checkpoints")).get(tick)
            for value in episodes
        ]
        reached_values = [
            _mapping(value)
            for value in values
            if bool(_mapping(value).get("reached", False))
        ]
        scores = [
            float(value["score"])
            for value in reached_values
            if isinstance(value.get("score"), (int, float))
        ]
        carried_scores = [
            float(value["score"])
            for value in map(_mapping, values)
            if isinstance(value.get("score"), (int, float))
        ]
        rows.append(
            (
                tick,
                len(reached_values),
                len(values),
                _numeric_summary(scores),
                _numeric_summary(carried_scores),
            )
        )
    return rows


def _append_policy_counts(
    lines: list[str], policy: Mapping[str, Any]
) -> None:
    timing_prefixes = (
        ("query_tick_bucket/", "Query timing"),
        ("override_tick_bucket/", "Override timing"),
    )
    timing_keys = {
        key
        for key in policy
        if any(key.startswith(prefix) for prefix, _ in timing_prefixes)
    }
    counters = [
        f"{key}={_number(policy[key])}"
        for key in sorted(set(policy) - timing_keys)
    ]
    if counters:
        lines.append("- Policy counters: " + ", ".join(counters))
    for prefix, label in timing_prefixes:
        buckets = sorted(
            (
                (int(key.removeprefix(prefix)), policy[key])
                for key in timing_keys
                if key.startswith(prefix)
            ),
            key=lambda value: value[0],
        )
        if buckets:
            rendered = ", ".join(
                f"{bucket}k={_number(value)}" for bucket, value in buckets
            )
            lines.append(f"- {label} (1k-tick buckets): {rendered}")


def _append_telemetry(lines: list[str], report: Mapping[str, Any]) -> None:
    entries = _suite_entries(report)
    if not entries:
        return
    lines.extend(["## Telemetry and checkpoints", ""])
    for label, suite in entries:
        aggregate = _mapping(suite.get("aggregate"))
        policy = _mapping(aggregate.get("policy_counts"))
        lines.append(f"### {label}")
        lines.append("")
        _append_policy_counts(lines, policy)
        for key in ("wall_seconds", "cpu_seconds"):
            if key in aggregate:
                lines.append(f"- {key}: {_number(aggregate[key])}")
        checkpoints = _checkpoint_rows(suite)
        if checkpoints:
            lines.extend(
                [
                    "",
                    "| Tick | Survived | Episodes | Reached-score | "
                    "Terminal-carried score |",
                    "|---:|---:|---:|---|---|",
                ]
            )
            for tick, reached, total, reached_score, carried in checkpoints:
                lines.append(
                    f"| {_cell(tick)} | {reached} | {total} | "
                    f"{_cell(reached_score)} | {_cell(carried)} |"
                )
        lines.append("")


def _append_conformal(lines: list[str], report: Mapping[str, Any]) -> None:
    calibration = _mapping(report.get("barrier_calibration"))
    freeze = _mapping(report.get("heldout_freeze"))
    if not calibration and not freeze:
        return
    lines.extend(["## Whole-seed conformal proof", ""])
    if calibration.get("passed") is False:
        lines.append(
            "- Calibration rejected: "
            f"{_cell(calibration.get('reason', 'unspecified'))}."
        )
    q = calibration.get("conformal_q", freeze.get("conformal_q"))
    alpha = calibration.get("conformal_alpha", freeze.get("conformal_alpha"))
    fit = [int(value) for value in _items(calibration.get("fit_seeds"))]
    held = [
        int(value) for value in _items(calibration.get("calibration_seeds"))
    ]
    if q is not None:
        lines.append(f"- Frozen q: `{_cell(q)}`")
    if alpha is not None:
        lines.append(f"- Alpha: `{_cell(alpha)}`")
    if fit or held:
        lines.append(
            f"- Whole-seed split: fit={len(fit)}, calibration={len(held)}, "
            f"overlap={len(set(fit) & set(held))}"
        )
    if held and isinstance(alpha, (int, float)):
        rank = math.ceil((len(held) + 1) * (1.0 - float(alpha)))
        lines.append(
            "- Order statistic: "
            f"`ceil(({len(held)}+1)*(1-{_number(alpha)}))={rank}`"
        )
    lines.append(
        "- Episode residual: maximum candidate overprediction of delta-B2 "
        "within each calibration seed; certification uses predicted "
        "delta-B2 minus q."
    )
    if freeze:
        for label, value in (
            ("Frozen model", freeze.get("model_sha256")),
            (
                "Training manifest",
                _mapping(freeze.get("training_manifest")).get("sha256"),
            ),
            (
                "Candidate generator",
                _mapping(freeze.get("candidate_generator")).get(
                    "config_sha256"
                ),
            ),
        ):
            if value is not None:
                lines.append(f"- {label}: `{_cell(value)}`")
        budget = _mapping(freeze.get("exact_simulator_budget"))
        if budget:
            lines.append(
                "- Exact budget: "
                + ", ".join(
                    f"{key}={_number(budget[key])}"
                    for key in sorted(budget)
                )
            )
    lines.append("")


def _append_diagnostics(
    lines: list[str], label: str, diagnostics_value: object
) -> None:
    diagnostics = _mapping(diagnostics_value)
    if not diagnostics:
        return
    lines.append(f"### {label} model diagnostics")
    lines.append("")
    lines.append(f"- Exact rows: {_number(diagnostics.get('rows', 0))}")
    if "horizon_breakdown" in diagnostics:
        lines.append(
            f"- Horizon breakdown: {_cell(diagnostics['horizon_breakdown'])}"
        )
    for key in (
        "exact_second_renewal_unresolved",
        "predicted_second_renewal_unresolved",
    ):
        if key in diagnostics:
            lines.append(f"- {key}: {_number(diagnostics[key])}")
    errors = _mapping(diagnostics.get("multi_step_error"))
    if errors:
        lines.extend(
            [
                "",
                "#### Multi-step model error",
                "",
                "| Target | MAE | RMSE | Bias |",
                "|---|---:|---:|---:|",
            ]
        )
        for target in sorted(errors):
            value = _mapping(errors[target])
            lines.append(
                f"| {_cell(target)} | {_cell(value.get('mae', ''))} | "
                f"{_cell(value.get('rmse', ''))} | "
                f"{_cell(value.get('bias', ''))} |"
            )
    deciles = sorted(
        (_mapping(value) for value in _items(diagnostics.get("risk_deciles"))),
        key=lambda value: int(value.get("decile", 0)),
    )
    if deciles:
        lines.extend(
            [
                "",
                "#### Risk deciles",
                "",
                "| Decile | Rows | Predicted catastrophic risk | "
                "Observed catastrophic | Barrier unsafe | Certified |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for value in deciles:
            lines.append(
                "| "
                + " | ".join(
                    _cell(item)
                    for item in (
                        value.get("decile", ""),
                        value.get("rows", ""),
                        value.get("mean_predicted_risk", ""),
                        value.get(
                            "observed_catastrophic_rate",
                            value.get("observed_unsafe_rate", ""),
                        ),
                        value.get("observed_barrier_unsafe_rate", ""),
                        value.get("certified", ""),
                    )
                )
                + " |"
            )
    lines.append("")


def _append_barrier(lines: list[str], report: Mapping[str, Any]) -> None:
    heldout = _mapping(report.get("barrier_heldout"))
    stress = _mapping(report.get("barrier_stress"))
    if not heldout and not stress:
        return
    lines.extend(["## Heldout and stress barrier", ""])
    if heldout:
        lines.extend(["### Heldout", ""])
        heldout_count = len(_items(heldout.get("seeds")))
        for label, key in (
            ("Episodes", "seeds"),
            ("False-safe episodes", "false_safe_episodes"),
            ("Clopper-Pearson upper 95%", "clopper_pearson_upper_95"),
            ("Certified coverage", "coverage"),
            ("Seed-clustered coverage lower 95%", "coverage_clustered_lower_95"),
            ("Cohort valid", "cohort_valid"),
        ):
            if key == "seeds" and key in heldout:
                lines.append(f"- {label}: {len(_items(heldout[key]))}")
            elif key in heldout:
                lines.append(f"- {label}: `{_cell(heldout[key])}`")
        if "false_safe_episodes" in heldout and heldout_count:
            rate = float(heldout["false_safe_episodes"]) / heldout_count
            lines.append(
                f"- False-safe rate: `{_number(rate)}` "
                f"({heldout['false_safe_episodes']}/{heldout_count})"
            )
        nonvacuity = _mapping(heldout.get("nonvacuity"))
        for key, value in heldout.items():
            lowered = key.lower()
            if (
                not isinstance(value, Mapping)
                and (
                    (
                        "unsafe" in lowered
                        and any(
                            name in lowered
                            for name in ("outcome", "state", "seed")
                        )
                    )
                    or "severe" in lowered
                )
            ):
                nonvacuity = {**nonvacuity, key: value}
        if nonvacuity:
            lines.extend(["", "#### Heldout nonvacuity", ""])
            for key in sorted(nonvacuity):
                value = nonvacuity[key]
                if isinstance(value, (list, tuple)):
                    value = len(value)
                elif isinstance(value, Mapping):
                    value = json.dumps(
                        value, sort_keys=True, separators=(",", ":")
                    )
                lines.append(
                    f"- {key.replace('_', ' ')}: `{_cell(value)}`"
                )
        lines.append("")
        _append_diagnostics(lines, "Heldout", heldout.get("diagnostics"))
    if stress:
        lines.extend(["### Stress", ""])
        stress_count = len(_items(stress.get("seeds")))
        for label, key in (
            ("Episodes", "seeds"),
            ("Episodes with exact unsafe", "episodes_with_exact_unsafe"),
            ("False-safe episodes", "false_safe_episodes"),
            ("Clopper-Pearson upper 95%", "clopper_pearson_upper_95"),
            ("Cohort valid", "cohort_valid"),
        ):
            if key == "seeds" and key in stress:
                lines.append(f"- {label}: {len(_items(stress[key]))}")
            elif key in stress:
                lines.append(f"- {label}: `{_cell(stress[key])}`")
        if "false_safe_episodes" in stress and stress_count:
            rate = float(stress["false_safe_episodes"]) / stress_count
            lines.append(
                f"- False-safe rate: `{_number(rate)}` "
                f"({stress['false_safe_episodes']}/{stress_count})"
            )
        lines.append("")
        _append_diagnostics(lines, "Stress", stress.get("diagnostics"))
    checks = _mapping(report.get("barrier_checks"))
    if checks:
        lines.extend(["### Barrier checks", ""])
        for name in sorted(checks):
            lines.append(f"- {name}: `{_cell(checks[name])}`")
        lines.append("")


def _append_student(lines: list[str], report: Mapping[str, Any]) -> None:
    student = _mapping(report.get("student_screen"))
    if not student:
        return
    lines.extend(["## Teacher-free student screen", ""])
    for label, key in (
        ("Passed", "passed"),
        ("Hard failures", "hard_failures"),
        ("Exact unsafe executions", "exact_unsafe_executions"),
        ("Allocation screen only", "allocation_screen_only"),
    ):
        if key in student:
            lines.append(f"- {label}: `{_cell(student[key])}`")
    lines.append("")


def _append_cost(lines: list[str], report: Mapping[str, Any]) -> None:
    cost = _mapping(report.get("cost"))
    if not cost:
        return
    lines.extend(["## Cost", ""])
    for key in sorted(cost):
        value = cost[key]
        if isinstance(value, (Mapping, list, tuple)):
            value = json.dumps(value, sort_keys=True, separators=(",", ":"))
        lines.append(f"- {key}: `{_cell(value)}`")
    lines.append("")


def _append_integrity(lines: list[str], report: Mapping[str, Any]) -> None:
    registry = _mapping(report.get("prior_leaf_registry"))
    identities = _mapping(report.get("artifact_identities"))
    if not registry and not identities:
        return
    lines.extend(["## Artifact integrity", ""])
    if registry:
        lines.append(
            f"- Prior leaves: {_number(registry.get('leaf_count', 0))}"
        )
        if registry.get("root_sha256"):
            lines.append(
                f"- Prior-leaf root: `{_cell(registry['root_sha256'])}`"
            )
    for name in sorted(identities):
        lines.append(f"- {name}: `{_cell(identities[name])}`")
    lines.append("")


def _append_limitations(lines: list[str], report: Mapping[str, Any]) -> None:
    lines.extend(["## Limitations", ""])
    values = _items(report.get("limitations"))
    if not values:
        values = [
            "This is development-only evidence; it does not authorize sealed, "
            "canonical, or deployment claims.",
            "Teacher and student screens are allocation screens only.",
        ]
        if "barrier_heldout" not in report:
            values.append("Heldout barrier evidence was not reached.")
        if "barrier_stress" not in report:
            values.append("Unsafe-stress evidence was not reached.")
        if "student_screen" not in report:
            values.append("The teacher-free student screen was not reached.")
    for value in values:
        lines.append(f"- {_cell(value)}")
    lines.append("")


def _render_evidence(report: Mapping[str, Any]) -> str:
    title = "# R3G Strategy C: event-world-model renewable MPC"
    verdict = report.get("verdict", "No verdict was recorded.")
    lines = [title, "", "## Verdict", "", str(verdict), ""]
    _append_scope(lines, report)
    _append_seeds(lines, report)
    _append_gates(lines, report)
    _append_validation(lines, report)
    _append_performance(lines, report)
    _append_telemetry(lines, report)
    _append_conformal(lines, report)
    _append_barrier(lines, report)
    _append_student(lines, report)
    _append_cost(lines, report)
    _append_integrity(lines, report)
    _append_limitations(lines, report)
    return "\n".join(lines).rstrip() + "\n"


def write_evidence(path: Path | str, report: Mapping[str, Any]) -> str:
    """Write reached-phase evidence and return its SHA-256."""

    destination = Path(path)
    content = _render_evidence(report).encode("utf-8")
    _atomic_write(destination, content)
    return _sha256_bytes(content)


def _leaf_entry(root: Path, path: Path) -> dict[str, object]:
    relative = path.relative_to(root).as_posix()
    return {
        "path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _scan_prior_leaves(root: Path) -> list[dict[str, object]]:
    output = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if path.is_symlink():
            raise RuntimeError(f"artifact leaf is a symlink: {path}")
        if not path.is_file():
            continue
        if path.parent == root and path.name in _FINAL_NAMES:
            raise FileExistsError(f"final artifact already exists: {path}")
        output.append(_leaf_entry(root, path))
    return output


def _seed_manifest(
    root: Path,
) -> tuple[dict[str, list[dict[str, object]]], str | None]:
    path = root / "seed-manifest.json"
    if not path.is_file():
        return {}, None
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = _items(_mapping(value).get("rows"))
    derivation = _mapping(value).get("derivation")
    if derivation is not None and not isinstance(derivation, str):
        raise ValueError("seed manifest derivation is malformed")
    output: dict[str, list[dict[str, object]]] = {}
    for raw in rows:
        row = _mapping(raw)
        split = row.get("split")
        if not isinstance(split, str) or type(row.get("seed")) is not int:
            raise ValueError("seed manifest row is malformed")
        retained = {
            key: row[key]
            for key in (
                "index",
                "seed",
                "seed_hex",
                "horizon",
                "purpose",
                "sha256",
            )
            if key in row
        }
        output.setdefault(split, []).append(retained)
    for split in output:
        output[split].sort(
            key=lambda row: (int(row.get("index", 0)), int(row["seed"]))
        )
    return (
        {split: output[split] for split in sorted(output)},
        derivation,
    )


def finalize_artifacts(
    output: Path | str, report: Mapping[str, Any]
) -> dict[str, object]:
    """Finalize an acyclic report/evidence/index hash graph.

    The input mapping is normalized but never mutated. The report registers
    every file present before finalization. The detached index then registers
    those leaves plus the report and evidence; its sidecar hashes only the
    index JSON, so neither file contains its own digest.
    """

    root = Path(output).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory does not exist: {root}")
    prior_files = _scan_prior_leaves(root)
    registry_body = {
        "schema": "irisu-r3g-prior-leaf-registry-v1",
        "files": prior_files,
    }
    registry = {
        **registry_body,
        "leaf_count": len(prior_files),
        "root_sha256": _sha256_bytes(_json_bytes(registry_body, pretty=False)),
    }
    normalized = json.loads(
        json.dumps(
            report,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
    )
    if not isinstance(normalized, dict):
        raise TypeError("campaign report must be a mapping")
    if "prior_leaf_registry" in normalized:
        raise ValueError("campaign report already contains a leaf registry")
    normalized["prior_leaf_registry"] = registry
    schedule, derivation = _seed_manifest(root)
    if schedule:
        existing = normalized.get("exact_seed_schedule")
        if existing is not None and existing != schedule:
            raise ValueError("campaign report seed schedule disagrees with manifest")
        normalized["exact_seed_schedule"] = schedule
    if derivation:
        existing = normalized.get("seed_derivation")
        if existing is not None and existing != derivation:
            raise ValueError(
                "campaign report seed derivation disagrees with manifest"
            )
        normalized["seed_derivation"] = derivation

    report_path = root / REPORT_NAME
    evidence_path = root / EVIDENCE_NAME
    report_bytes = _json_bytes(normalized, pretty=True)
    evidence_bytes = _render_evidence(normalized).encode("utf-8")
    _atomic_write(report_path, report_bytes)
    _atomic_write(evidence_path, evidence_bytes)
    report_entry = _leaf_entry(root, report_path)
    evidence_entry = _leaf_entry(root, evidence_path)
    indexed_files = sorted(
        [*prior_files, report_entry, evidence_entry],
        key=lambda value: str(value["path"]),
    )
    artifact_root_body = {
        "schema": "irisu-r3g-artifact-root-v1",
        "files": indexed_files,
    }
    artifact_root_sha256 = _sha256_bytes(
        _json_bytes(artifact_root_body, pretty=False)
    )
    index = {
        "schema": "irisu-r3g-detached-artifact-index-v1",
        "artifact_root_sha256": artifact_root_sha256,
        "file_count": len(indexed_files),
        "files": indexed_files,
        "prior_leaf_root_sha256": registry["root_sha256"],
        "campaign_report_sha256": report_entry["sha256"],
        "evidence_sha256": evidence_entry["sha256"],
    }
    index_path = root / INDEX_NAME
    sidecar_path = root / INDEX_SHA_NAME
    index_bytes = _json_bytes(index, pretty=True)
    index_sha256 = _sha256_bytes(index_bytes)
    _atomic_write(index_path, index_bytes)
    _atomic_write(
        sidecar_path,
        f"{index_sha256}  {INDEX_NAME}\n".encode("ascii"),
    )
    return {
        "artifact_root": str(root),
        "artifact_root_sha256": artifact_root_sha256,
        "prior_leaf_root_sha256": registry["root_sha256"],
        "prior_leaf_count": len(prior_files),
        "indexed_file_count": len(indexed_files),
        "campaign_report_path": str(report_path),
        "campaign_report_sha256": report_entry["sha256"],
        "evidence_path": str(evidence_path),
        "evidence_sha256": evidence_entry["sha256"],
        "artifact_index_path": str(index_path),
        "artifact_index_sha256": index_sha256,
        "artifact_index_sidecar_path": str(sidecar_path),
    }


__all__ = ["finalize_artifacts", "write_evidence"]
