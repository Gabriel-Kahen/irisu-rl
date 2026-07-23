"""Tamper-evident, asset-free evidence reports for original-game soak runs."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean, stdev
from typing import Any


EVENT_SCHEMA = "r4a-safe-event-v1"
THRESHOLD_SCHEMA = "r4a-soak-thresholds-v1"
REPORT_SCHEMA = "r4a-soak-report-v1"
ZERO_SHA256 = "0" * 64
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
PROVENANCE_KEY = re.compile(r"[a-z][a-z0-9_]{0,62}_sha256")
FORBIDDEN_PROVENANCE_TERMS = ("path", "pixel", "token")
MAX_JSONL_LINE_BYTES = 1 << 20

METRIC_UNITS = {
    "capture_fps": "frames/second",
    "capture_jitter_seconds": "seconds",
    "ring_age_seconds": "seconds",
    "request_ack_seconds": "seconds",
    "poll_effect_interval_seconds": "seconds",
    "effect_visible_seconds": "seconds",
    "total_latency_seconds": "seconds",
    "duplicate_frames": "count",
    "dropped_frames": "count",
    "stale_frames": "count",
    "out_of_order_frames": "count",
    "deadline_misses": "count",
    "action_confirmations": "count",
    "missed_effects": "count",
    "ambiguous_effects": "count",
    "effect_confirmation_failures": "count",
    "button_release_failures": "count",
    "cross_window_misroutes": "count",
    "crop_drift_pixels": "pixels",
    "resource_growth_bytes": "bytes",
}
METRICS = tuple(METRIC_UNITS)
COUNT_METRICS = frozenset(
    {
        "duplicate_frames",
        "dropped_frames",
        "stale_frames",
        "out_of_order_frames",
        "deadline_misses",
        "action_confirmations",
        "missed_effects",
        "ambiguous_effects",
        "effect_confirmation_failures",
        "button_release_failures",
        "cross_window_misroutes",
    }
)
INTRINSIC_ZERO_METRICS = frozenset(
    {"button_release_failures", "cross_window_misroutes"}
)
_EVENT_KEYS = frozenset(
    {
        "schema",
        "sequence",
        "monotonic_ns",
        "experiment_id",
        "measurements",
        "provenance",
        "threshold_sha256",
        "previous_sha256",
        "sha256",
    }
)
_THRESHOLD_KEYS = frozenset(
    {
        "schema",
        "experiment_ids",
        "required_provenance",
        "minimum_duration_seconds",
        "metrics",
    }
)
_METRIC_SPEC_KEYS = frozenset(
    {
        "direction",
        "minimum_samples",
        "p50",
        "p95",
        "p99",
        "worst",
        "total",
        "uncertainty_95",
    }
)
_BOUND_KEYS = frozenset({"min", "max"})


class EvidenceError(ValueError):
    """Evidence is malformed, unsafe, inconsistent, or tampered with."""


class PublicationError(FileExistsError):
    """The report could not be published without replacing an existing path."""


def canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise EvidenceError("value is not finite canonical JSON") from exc


def _is_sha256(value: object, *, allow_zero: bool = False) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and (allow_zero or value != ZERO_SHA256)
        and all(character in "0123456789abcdef" for character in value)
    )


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EvidenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise EvidenceError(f"non-finite JSON number: {value}")


def _decode_json(data: bytes, label: str) -> Any:
    try:
        text = data.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except EvidenceError:
        raise
    except (UnicodeDecodeError, ValueError) as exc:
        raise EvidenceError(f"{label} is not valid UTF-8 JSON") from exc


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    actual = frozenset(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EvidenceError(f"{label} keys disagree: missing={missing}, extra={extra}")


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be a number")
    try:
        result = float(value)
    except OverflowError as exc:
        raise EvidenceError(f"{label} must be finite and nonnegative") from exc
    if not math.isfinite(result) or result < 0:
        raise EvidenceError(f"{label} must be finite and nonnegative")
    return result


def _validate_provenance(value: object, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise EvidenceError(f"{label} must be an object")
    result: dict[str, str] = {}
    for key, digest in value.items():
        if (
            PROVENANCE_KEY.fullmatch(key) is None
            or any(term in key for term in FORBIDDEN_PROVENANCE_TERMS)
        ):
            raise EvidenceError(f"{label} contains unsafe provenance key {key!r}")
        if not _is_sha256(digest):
            raise EvidenceError(f"{label}.{key} must be a nonzero lowercase SHA-256")
        result[key] = digest
    return result


def _validate_event(
    value: object,
    *,
    expected_sequence: int | None = None,
    expected_previous: str | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise EvidenceError("event must be an object")
    _exact_keys(value, _EVENT_KEYS, "event")
    if value["schema"] != EVENT_SCHEMA:
        raise EvidenceError("event schema mismatch")
    sequence = value["sequence"]
    timestamp = value["monotonic_ns"]
    experiment_id = value["experiment_id"]
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        raise EvidenceError("event sequence must be a positive integer")
    if expected_sequence is not None and sequence != expected_sequence:
        raise EvidenceError(
            f"event sequence mismatch: expected {expected_sequence}, got {sequence}"
        )
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise EvidenceError("event monotonic_ns must be a nonnegative integer")
    if not isinstance(experiment_id, str) or IDENTIFIER.fullmatch(experiment_id) is None:
        raise EvidenceError("event experiment_id is unsafe")
    previous = value["previous_sha256"]
    digest = value["sha256"]
    if not _is_sha256(previous, allow_zero=True):
        raise EvidenceError("event previous_sha256 is malformed")
    if expected_previous is not None and previous != expected_previous:
        raise EvidenceError("event SHA-256 chain link mismatch")
    if not _is_sha256(digest):
        raise EvidenceError("event sha256 is malformed")
    measurements = value["measurements"]
    if not isinstance(measurements, Mapping) or any(
        not isinstance(key, str) for key in measurements
    ):
        raise EvidenceError("event measurements must be an object")
    unknown = sorted(set(measurements) - set(METRICS))
    if unknown:
        raise EvidenceError(f"event contains unknown measurements: {unknown}")
    for key, item in measurements.items():
        measured = _finite_nonnegative(item, f"event measurements.{key}")
        if key in COUNT_METRICS and not measured.is_integer():
            raise EvidenceError(f"event measurements.{key} must be an integer count")
    provenance = _validate_provenance(value["provenance"], "event provenance")
    if not _is_sha256(value["threshold_sha256"]):
        raise EvidenceError("event threshold_sha256 is malformed")
    unsigned = dict(value)
    del unsigned["sha256"]
    expected_digest = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if digest != expected_digest:
        raise EvidenceError("event SHA-256 does not match its canonical contents")
    return {
        **dict(value),
        "measurements": dict(measurements),
        "provenance": provenance,
    }


def seal_event(value: Mapping[str, Any], previous_sha256: str = ZERO_SHA256) -> dict[str, Any]:
    """Add a chain link and digest to one otherwise complete safe event."""

    if "previous_sha256" in value or "sha256" in value:
        raise EvidenceError("unsealed event must not contain chain fields")
    if not _is_sha256(previous_sha256, allow_zero=True):
        raise EvidenceError("previous_sha256 is malformed")
    sealed = {**dict(value), "previous_sha256": previous_sha256}
    sealed["sha256"] = hashlib.sha256(canonical_json_bytes(sealed)).hexdigest()
    return _validate_event(sealed)


def encode_event(event: Mapping[str, Any]) -> bytes:
    """Encode one already sealed event as a canonical JSONL record."""

    validated = _validate_event(event)
    return canonical_json_bytes(validated) + b"\n"


def load_event_chain(path: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    previous_timestamp = -1
    input_digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as stream:
            line_number = 0
            while line := stream.readline(MAX_JSONL_LINE_BYTES + 1):
                line_number += 1
                input_digest.update(line)
                if len(line) > MAX_JSONL_LINE_BYTES:
                    raise EvidenceError(f"event line {line_number} exceeds the size limit")
                if not line.endswith(b"\n"):
                    raise EvidenceError(f"event line {line_number} lacks a newline")
                if not line.strip():
                    raise EvidenceError(f"event line {line_number} is blank")
                value = _decode_json(line, f"event line {line_number}")
                event = _validate_event(
                    value,
                    expected_sequence=line_number,
                    expected_previous=previous,
                )
                if event["monotonic_ns"] < previous_timestamp:
                    raise EvidenceError("event monotonic timestamps moved backwards")
                records.append(event)
                previous = event["sha256"]
                previous_timestamp = event["monotonic_ns"]
    except OSError as exc:
        raise EvidenceError(f"cannot read event stream: {exc}") from exc
    return records, input_digest.hexdigest()


def _bound(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise EvidenceError(f"{label} must be an object")
    if not value or not set(value) <= _BOUND_KEYS:
        raise EvidenceError(f"{label} must contain min and/or max")
    result = {
        key: _finite_nonnegative(item, f"{label}.{key}")
        for key, item in value.items()
    }
    if "min" in result and "max" in result and result["min"] > result["max"]:
        raise EvidenceError(f"{label} min exceeds max")
    return result


def load_thresholds(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise EvidenceError(f"cannot read threshold config: {exc}") from exc
    if len(raw) > MAX_JSONL_LINE_BYTES:
        raise EvidenceError("threshold config exceeds the size limit")
    value = _decode_json(raw, "threshold config")
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise EvidenceError("threshold config must be an object")
    _exact_keys(value, _THRESHOLD_KEYS, "threshold config")
    if value["schema"] != THRESHOLD_SCHEMA:
        raise EvidenceError("threshold config schema mismatch")
    experiment_ids = value["experiment_ids"]
    if (
        not isinstance(experiment_ids, Sequence)
        or isinstance(experiment_ids, (str, bytes))
        or not experiment_ids
        or any(
            not isinstance(item, str) or IDENTIFIER.fullmatch(item) is None
            for item in experiment_ids
        )
        or len(set(experiment_ids)) != len(experiment_ids)
    ):
        raise EvidenceError("threshold experiment_ids must be unique safe identifiers")
    provenance = _validate_provenance(
        value["required_provenance"], "required provenance"
    )
    if not provenance:
        raise EvidenceError("threshold config requires at least one provenance hash")
    minimum_duration = _finite_nonnegative(
        value["minimum_duration_seconds"], "minimum_duration_seconds"
    )
    if minimum_duration <= 0:
        raise EvidenceError("minimum_duration_seconds must be positive")
    metrics = value["metrics"]
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRICS):
        raise EvidenceError("threshold config must predeclare every R4a metric")
    normalized_metrics: dict[str, dict[str, Any]] = {}
    for name in METRICS:
        spec = metrics[name]
        if not isinstance(spec, Mapping) or any(
            not isinstance(key, str) for key in spec
        ):
            raise EvidenceError(f"threshold metrics.{name} must be an object")
        if not set(spec) <= _METRIC_SPEC_KEYS:
            raise EvidenceError(f"threshold metrics.{name} contains unknown keys")
        direction = spec.get("direction")
        minimum_samples = spec.get("minimum_samples")
        if direction not in {"min", "max"}:
            raise EvidenceError(f"threshold metrics.{name}.direction is invalid")
        if (
            isinstance(minimum_samples, bool)
            or not isinstance(minimum_samples, int)
            or minimum_samples <= 0
        ):
            raise EvidenceError(
                f"threshold metrics.{name}.minimum_samples must be positive"
            )
        bounds = {
            key: _bound(item, f"threshold metrics.{name}.{key}")
            for key, item in spec.items()
            if key not in {"direction", "minimum_samples"}
        }
        if not bounds:
            raise EvidenceError(f"threshold metrics.{name} lacks an acceptance bound")
        normalized_metrics[name] = {
            "direction": direction,
            "minimum_samples": minimum_samples,
            **bounds,
        }
    normalized = {
        "schema": THRESHOLD_SCHEMA,
        "experiment_ids": list(experiment_ids),
        "required_provenance": provenance,
        "minimum_duration_seconds": minimum_duration,
        "metrics": normalized_metrics,
    }
    return normalized, hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _summary(values: list[float], direction: str) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "p50": None,
            "p95": None,
            "p99": None,
            "worst": None,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "total": 0.0,
            "uncertainty_95": None,
            "uncertainty_method": "1.96 * sample_standard_deviation / sqrt(count)",
        }
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "worst": min(values) if direction == "min" else max(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": fmean(values),
        "total": math.fsum(values),
        "uncertainty_95": (
            1.96 * stdev(values) / math.sqrt(len(values))
            if len(values) >= 2
            else None
        ),
        "uncertainty_method": "1.96 * sample_standard_deviation / sqrt(count)",
    }


def _evaluate_metric(
    summary: Mapping[str, Any], spec: Mapping[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    unavailable: list[str] = []
    count = summary["count"]
    minimum_samples = spec["minimum_samples"]
    if count < minimum_samples:
        unavailable.append(f"requires {minimum_samples} samples; observed {count}")
    else:
        for statistic, bounds in spec.items():
            if statistic in {"direction", "minimum_samples"}:
                continue
            measured = summary[statistic]
            if measured is None:
                unavailable.append(f"{statistic} is unavailable")
                continue
            if "min" in bounds and measured < bounds["min"]:
                failures.append(f"{statistic} {measured} < minimum {bounds['min']}")
            if "max" in bounds and measured > bounds["max"]:
                failures.append(f"{statistic} {measured} > maximum {bounds['max']}")
    status = "fail" if failures else "not_evaluable" if unavailable else "pass"
    return {
        "status": status,
        "failures": failures,
        "unavailable": unavailable,
        "minimum_samples": minimum_samples,
    }


def build_report(
    event_path: str | os.PathLike[str],
    threshold_path: str | os.PathLike[str],
) -> dict[str, Any]:
    events, input_sha256 = load_event_chain(event_path)
    thresholds, threshold_sha256 = load_thresholds(threshold_path)
    expected_ids = thresholds["experiment_ids"]
    expected_id_set = set(expected_ids)
    observed_ids: list[str] = []
    observed_id_set: set[str] = set()
    provenance: dict[str, str] = {}
    samples = {name: [] for name in METRICS}
    measurement_events: list[dict[str, Any]] = []
    missing_event_provenance = 0
    for event in events:
        if event["threshold_sha256"] != threshold_sha256:
            raise EvidenceError("event stream is not bound to this threshold config")
        experiment_id = event["experiment_id"]
        if experiment_id not in expected_id_set:
            raise EvidenceError(
                f"event names undeclared experiment ID {experiment_id!r}"
            )
        if experiment_id not in observed_id_set:
            observed_id_set.add(experiment_id)
            observed_ids.append(experiment_id)
        if event["measurements"]:
            measurement_events.append(event)
        event_provenance_complete = True
        for key, expected in thresholds["required_provenance"].items():
            if event["provenance"].get(key) != expected:
                event_provenance_complete = False
        if event["measurements"] and not event_provenance_complete:
            missing_event_provenance += 1
        for key, digest in event["provenance"].items():
            prior = provenance.setdefault(key, digest)
            if prior != digest:
                raise EvidenceError(f"provenance hash changed during stream: {key}")
        for name, measured in event["measurements"].items():
            samples[name].append(measured)

    metric_reports: dict[str, Any] = {}
    metric_statuses: list[str] = []
    for name in METRICS:
        spec = thresholds["metrics"][name]
        summary = _summary(samples[name], spec["direction"])
        evaluation = _evaluate_metric(summary, spec)
        if name in INTRINSIC_ZERO_METRICS and summary["total"] != 0.0:
            evaluation = {
                **evaluation,
                "status": "fail",
                "failures": [
                    *evaluation["failures"],
                    f"R4a requires total 0.0; observed {summary['total']}",
                ],
            }
        metric_statuses.append(evaluation["status"])
        metric_reports[name] = {
            "unit": METRIC_UNITS[name],
            **summary,
            "threshold": spec,
            "evaluation": evaluation,
        }

    missing_ids = [item for item in expected_ids if item not in observed_id_set]
    expected_provenance = thresholds["required_provenance"]
    missing_provenance = [
        key for key in expected_provenance if key not in provenance
    ]
    mismatched_provenance = [
        key
        for key, expected in expected_provenance.items()
        if key in provenance and provenance[key] != expected
    ]
    provenance_status = (
        "fail"
        if mismatched_provenance
        else "not_evaluable"
        if missing_provenance or missing_event_provenance
        else "pass"
    )
    experiment_status = "not_evaluable" if missing_ids else "pass"
    duration_seconds = (
        (
            measurement_events[-1]["monotonic_ns"]
            - measurement_events[0]["monotonic_ns"]
        )
        / 1e9
        if len(measurement_events) >= 2
        else 0.0
    )
    minimum_duration = thresholds["minimum_duration_seconds"]
    duration_status = (
        "pass" if duration_seconds >= minimum_duration else "not_evaluable"
    )
    statuses = [
        *metric_statuses,
        provenance_status,
        experiment_status,
        duration_status,
    ]
    status = (
        "fail"
        if "fail" in statuses
        else "not_evaluable"
        if "not_evaluable" in statuses
        else "pass"
    )
    return {
        "schema": REPORT_SCHEMA,
        "status": status,
        "event_stream": {
            "count": len(events),
            "sha256": input_sha256,
            "chain_head_sha256": events[-1]["sha256"] if events else ZERO_SHA256,
        },
        "threshold_config_sha256": threshold_sha256,
        "duration": {
            "seconds": duration_seconds,
            "minimum_seconds": minimum_duration,
            "status": duration_status,
        },
        "experiments": {
            "required": expected_ids,
            "observed": observed_ids,
            "missing": missing_ids,
            "status": experiment_status,
        },
        "provenance": {
            "required": expected_provenance,
            "observed": provenance,
            "missing": missing_provenance,
            "mismatched": mismatched_provenance,
            "measurement_events_missing_required": missing_event_provenance,
            "status": provenance_status,
        },
        "metrics": metric_reports,
    }


def validate_report(report: object) -> str:
    """Validate a complete aggregate report and return its canonical SHA-256."""

    if not isinstance(report, Mapping):
        raise EvidenceError("soak report must be an object")
    expected_keys = {
        "schema",
        "status",
        "event_stream",
        "threshold_config_sha256",
        "duration",
        "experiments",
        "provenance",
        "metrics",
    }
    if set(report) != expected_keys:
        raise EvidenceError("soak report fields disagree with the report schema")
    if report.get("schema") != REPORT_SCHEMA or report.get("status") != "pass":
        raise EvidenceError("soak report is not a passing R4a report")
    stream = report.get("event_stream")
    if not isinstance(stream, Mapping) or set(stream) != {
        "count",
        "sha256",
        "chain_head_sha256",
    }:
        raise EvidenceError("soak event-stream summary is malformed")
    count = stream.get("count")
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise EvidenceError("soak report event count must be positive")
    for key in ("sha256", "chain_head_sha256"):
        if not _is_sha256(stream.get(key)):
            raise EvidenceError(f"soak event_stream.{key} is invalid")
    if not _is_sha256(report.get("threshold_config_sha256")):
        raise EvidenceError("soak threshold-config SHA-256 is invalid")

    duration = report.get("duration")
    if not isinstance(duration, Mapping) or set(duration) != {
        "seconds",
        "minimum_seconds",
        "status",
    }:
        raise EvidenceError("soak duration summary is malformed")
    seconds = _finite_nonnegative(duration.get("seconds"), "soak duration")
    minimum = _finite_nonnegative(
        duration.get("minimum_seconds"), "minimum soak duration"
    )
    if minimum <= 0 or seconds < minimum or duration.get("status") != "pass":
        raise EvidenceError("soak duration gate did not pass")

    experiments = report.get("experiments")
    if not isinstance(experiments, Mapping):
        raise EvidenceError("soak experiment summary is malformed")
    if (
        experiments.get("status") != "pass"
        or experiments.get("missing") != []
        or not experiments.get("required")
        or set(experiments.get("required", ()))
        != set(experiments.get("observed", ()))
    ):
        raise EvidenceError("soak experiment coverage did not pass")

    provenance = report.get("provenance")
    if not isinstance(provenance, Mapping):
        raise EvidenceError("soak provenance summary is malformed")
    required = provenance.get("required")
    observed = provenance.get("observed")
    if not isinstance(required, Mapping) or not isinstance(observed, Mapping):
        raise EvidenceError("soak provenance mappings are malformed")
    if (
        provenance.get("status") != "pass"
        or provenance.get("missing") != []
        or provenance.get("mismatched") != []
        or provenance.get("measurement_events_missing_required") != 0
        or not required
        or any(observed.get(key) != digest for key, digest in required.items())
    ):
        raise EvidenceError("soak provenance coverage did not pass")

    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != set(METRICS):
        raise EvidenceError("soak report metric set is incomplete")
    for name in METRICS:
        metric = metrics[name]
        if not isinstance(metric, Mapping) or metric.get("unit") != METRIC_UNITS[name]:
            raise EvidenceError(f"soak metric {name} is malformed")
        metric_count = metric.get("count")
        if (
            isinstance(metric_count, bool)
            or not isinstance(metric_count, int)
            or metric_count <= 0
        ):
            raise EvidenceError(f"soak metric {name} has no samples")
        evaluation = metric.get("evaluation")
        if not isinstance(evaluation, Mapping) or evaluation.get("status") != "pass":
            raise EvidenceError(f"soak metric {name} did not pass")
        if name in INTRINSIC_ZERO_METRICS and metric.get("total") != 0.0:
            raise EvidenceError(f"soak metric {name} violates its intrinsic zero gate")
    return hashlib.sha256(canonical_json_bytes(report)).hexdigest()


def write_report_noreplace(
    report: Mapping[str, Any], destination: str | os.PathLike[str]
) -> None:
    """Atomically publish a canonical report without replacing any entry."""

    target = Path(destination)
    parent = target.parent
    if not parent.is_dir():
        raise PublicationError(f"report parent directory does not exist: {parent}")
    data = json.dumps(
        report,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ).encode("ascii") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target, follow_symlinks=False)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise PublicationError(f"report destination already exists: {target}") from exc
            raise
        directory_descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def generate_report(
    event_path: str | os.PathLike[str],
    threshold_path: str | os.PathLike[str],
    destination: str | os.PathLike[str],
) -> dict[str, Any]:
    report = build_report(event_path, threshold_path)
    write_report_noreplace(report, destination)
    return report
