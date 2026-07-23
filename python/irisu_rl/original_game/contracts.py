"""Validation and evidence-backed finalization of the live deployment contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
import tomllib
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from .evidence import REPORT_SCHEMA, EvidenceError, verify_report

CONTRACT_SCHEMA = "original-game-deployment-contract-v2"
EVIDENCE_SCHEMA = "r4b-deployment-measurements-v2"
SEMANTIC_SHA256 = "dd764fd625b2a6604128fe1605b988144cd0def9d044c43b8fa3fb260e16e677"
PROVISIONAL_BASE_SHA256 = (
    "aa20ada39669d283cd2a2081ecdc5b769f5d7b2270e739e225790aeb8ef9258c"
)
EMPIRICAL_SECTIONS = (
    "wait_duration",
    "click_macro",
    "cursor",
    "capture",
    "effect_timing",
    "coordinate_calibration",
)
MEASUREMENT_FIELDS: Mapping[str, tuple[str, ...]] = {
    "wait_duration": (
        "gameplay_period_seconds",
        "scheduler_error_seconds",
    ),
    "click_macro": (
        "press_duration_seconds",
        "release_duration_seconds",
        "maximum_clicks_per_second",
    ),
    "capture": (
        "frame_rate_hz",
        "request_to_completion_seconds",
        "stale_after_seconds",
    ),
    "effect_timing": (
        "injection_to_poll_seconds",
        "effect_to_visible_seconds",
        "request_to_visible_seconds",
    ),
    "coordinate_calibration": ("residual_client_pixels",),
}
MEASUREMENT_UNITS: Mapping[str, str] = {
    "gameplay_period_seconds": "seconds",
    "scheduler_error_seconds": "seconds",
    "press_duration_seconds": "seconds",
    "release_duration_seconds": "seconds",
    "maximum_clicks_per_second": "clicks/second",
    "frame_rate_hz": "frames/second",
    "request_to_completion_seconds": "seconds",
    "stale_after_seconds": "seconds",
    "injection_to_poll_seconds": "seconds",
    "effect_to_visible_seconds": "seconds",
    "request_to_visible_seconds": "seconds",
    "residual_client_pixels": "client_pixels",
    "fixed_action_rate_hz": "actions/second",
    "maximum_speed_client_pixels_per_second": "client_pixels/second",
    "maximum_acceleration_client_pixels_per_second2": "client_pixels/second2",
}
MINIMUM_IS_WORST = frozenset(
    {
        "frame_rate_hz",
        "maximum_clicks_per_second",
        "fixed_action_rate_hz",
    }
)
QUANTILES = ("p50", "p95", "p99", "worst")
SHA256_LENGTH = 64
TOML_KEY = re.compile(r"[A-Za-z0-9_-]+")
TOP_LEVEL_FIELDS = {
    "contract_schema",
    "version",
    "sha256",
    "measurement_status",
    "live_deployment_enabled",
    "live_deployment_blockers",
    "semantic_kinds",
    "wait_min_ticks",
    "wait_max_ticks",
    "wait_support",
    "conditional_distribution",
    "deterministic_evaluation",
    "serialization",
    "coordinate_log_prob_epsilon",
    "coordinate_quantization",
    "client_width",
    "client_height",
    "allow_both",
    "wait_duration",
    "click_macro",
    "cursor",
    "capture",
    "effect_timing",
    "coordinate_calibration",
    "fairness",
    "safety",
    "evidence",
}
BASE_SECTION_FIELDS = {
    "wait_duration": {
        "status",
        "simulator_default_min_ticks",
        "simulator_default_max_ticks",
        "owner",
        "evidence",
        "decision_deadline",
    },
    "click_macro": {
        "press_ticks",
        "release_ticks",
        "status",
        "owner",
        "evidence",
        "decision_deadline",
    },
    "cursor": {
        "status",
        "simulator_default_travel_model",
        "owner",
        "evidence",
        "decision_deadline",
    },
    "capture": {"status", "owner", "evidence", "decision_deadline"},
    "effect_timing": {"status", "owner", "evidence", "decision_deadline"},
    "coordinate_calibration": {
        "status",
        "owner",
        "evidence",
        "decision_deadline",
    },
}
MEASURED_COMMON_FIELDS = {
    "sample_count",
    "uncertainty_method",
    "provenance_category",
    "experiment_ids",
    "artifact_sha256",
    "measurements",
}
MEASURED_EXTRA_FIELDS = {
    "wait_duration": set(),
    "click_macro": {
        "input_provider_capability",
        "weak_button",
        "strong_button",
    },
    "cursor": {
        "travel_model",
        "quantization",
        "cursor_retention",
        "path_logging",
    },
    "capture": set(),
    "effect_timing": set(),
    "coordinate_calibration": {
        "click_sweep_dimensions",
        "continuous_drift_check",
    },
}


class ContractError(ValueError):
    """A deployment contract or its measurement evidence is unsafe."""


def _mapping(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where} must be an object")
    return value


def _string(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{where} must be a nonempty string")
    return value


def _sha256(value: object, where: str) -> str:
    text = _string(value, where)
    if len(text) != SHA256_LENGTH or any(c not in "0123456789abcdef" for c in text):
        raise ContractError(f"{where} must be a lowercase SHA-256")
    if text == "0" * SHA256_LENGTH:
        raise ContractError(f"{where} must not be the zero SHA-256 sentinel")
    return text


def _positive_number(value: object, where: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ContractError(f"{where} must be finite and {qualifier}")
    return result


def _positive_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{where} must be a positive integer")
    return value


def _string_list(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{where} must be a nonempty array")
    result = tuple(_string(item, f"{where}[]") for item in value)
    if len(set(result)) != len(result):
        raise ContractError(f"{where} must not contain duplicates")
    return result


def _exact_keys(value: Mapping[str, Any], expected: set[str], where: str) -> None:
    if set(value) != expected:
        raise ContractError(
            f"{where} fields disagree: "
            f"missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}"
        )


def _safe_label(value: object, where: str) -> str:
    text = _string(value, where)
    if (
        len(text) > 256
        or any(character in text for character in ("/", "\\", "\n", "\r", "\0"))
        or any(term in text.lower() for term in ("token", "secret"))
    ):
        raise ContractError(f"{where} contains unsafe or private text")
    return text


def _validate_metric(metric: object, field: str, where: str) -> None:
    value = _mapping(metric, where)
    _exact_keys(
        value,
        {
            "unit",
            "direction",
            "sample_count",
            "p50",
            "p95",
            "p99",
            "worst",
            "uncertainty",
        },
        where,
    )
    if value.get("unit") != MEASUREMENT_UNITS[field]:
        raise ContractError(f"{where}.unit must be {MEASUREMENT_UNITS[field]}")
    expected_direction = "min" if field in MINIMUM_IS_WORST else "max"
    if value.get("direction") != expected_direction:
        raise ContractError(f"{where}.direction must be {expected_direction}")
    _positive_int(value.get("sample_count"), f"{where}.sample_count")
    p50, p95, p99 = (
        _positive_number(value.get(key), f"{where}.{key}", allow_zero=True)
        for key in ("p50", "p95", "p99")
    )
    worst = _positive_number(value.get("worst"), f"{where}.worst", allow_zero=True)
    if not p50 <= p95 <= p99:
        raise ContractError(f"{where} percentiles are not monotone")
    if expected_direction == "max" and worst < p99:
        raise ContractError(f"{where}.worst must be at least p99")
    if expected_direction == "min" and worst > p50:
        raise ContractError(f"{where}.worst must be at most p50")
    _positive_number(value.get("uncertainty"), f"{where}.uncertainty")


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_deployment_contract(path: Path) -> dict[str, Any]:
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ContractError(f"cannot read deployment contract {path}: {exc}") from exc


def _validate_common(contract: Mapping[str, Any]) -> None:
    measured = contract.get("measurement_status") in {
        "measured",
        "measured_pending_review",
    }
    expected_top = set(TOP_LEVEL_FIELDS)
    if measured:
        expected_top.update(
            {
                "measurement_bundle_sha256",
                "soak_report_sha256",
                "runtime_provenance",
            }
        )
    _exact_keys(contract, expected_top, "deployment contract")
    if contract.get("contract_schema") != CONTRACT_SCHEMA:
        raise ContractError(f"contract_schema must be {CONTRACT_SCHEMA}")
    if contract.get("version") != "deployment-v1":
        raise ContractError("only deployment-v1 is supported")
    if contract.get("sha256") != SEMANTIC_SHA256:
        raise ContractError("deployment-v1 semantic SHA-256 changed")
    if contract.get("semantic_kinds") != ["WAIT", "FIRE_WEAK", "FIRE_STRONG"]:
        raise ContractError("semantic action kinds changed")
    if contract.get("allow_both") is not False:
        raise ContractError("simultaneous gameplay buttons must remain disabled")
    expected_action_fields = {
        "wait_min_ticks": 1,
        "wait_max_ticks": 100,
        "wait_support": "every integer in the inclusive range",
        "conditional_distribution": (
            "kind categorical; wait categorical; per-shot independent x/y Beta"
        ),
        "deterministic_evaluation": (
            "masked kind argmax; masked wait argmax; coordinate Beta mean"
        ),
        "serialization": "little-endian <BIdd after canonical validation",
        "coordinate_log_prob_epsilon": 0.000001,
        "coordinate_quantization": (
            "floor(normalized * extent), clamped to [0, extent - 1]"
        ),
    }
    for key, expected in expected_action_fields.items():
        if contract.get(key) != expected:
            raise ContractError(f"deployment-v1 {key} changed")
    if contract.get("client_width") != 640.0 or contract.get("client_height") != 480.0:
        raise ContractError("deployment-v1 client geometry changed")
    safety = _mapping(contract.get("safety"), "safety")
    _exact_keys(
        safety,
        {
            "maximum_pending_actions",
            "claim_required",
            "exact_window_identity_required",
            "explicit_button_release_required",
            "atomic_click_provider_supported",
            "broker_release_deadline_required",
            "neutralize_on_claim_end_or_expiry_required",
            "fail_closed_on_crop_or_claim_drift",
        },
        "safety",
    )
    if safety.get("maximum_pending_actions") != 1:
        raise ContractError("maximum_pending_actions must be one")
    if safety.get("claim_required") is not True:
        raise ContractError("an exact window claim must be required")
    if safety.get("explicit_button_release_required") is not True:
        raise ContractError("explicit button release must be required")
    if safety.get("exact_window_identity_required") is not True:
        raise ContractError("exact window identity must be required")
    if safety.get("atomic_click_provider_supported") is not False:
        raise ContractError("atomic-click-only providers must remain unsupported")
    if safety.get("broker_release_deadline_required") is not True:
        raise ContractError("broker-enforced release deadlines must be required")
    if safety.get("neutralize_on_claim_end_or_expiry_required") is not True:
        raise ContractError("claim end or expiry must neutralize held buttons")
    if safety.get("fail_closed_on_crop_or_claim_drift") is not True:
        raise ContractError("crop and claim drift must fail closed")
    click = _mapping(contract.get("click_macro"), "click_macro")
    if click.get("press_ticks") != 1 or click.get("release_ticks") != 1:
        raise ContractError("deployment-v1 simulator click edges changed")
    fairness = _mapping(contract.get("fairness"), "fairness")
    _exact_keys(
        fairness,
        {"fast_forward", "simultaneous_buttons", "status"},
        "fairness",
    )
    if fairness.get("fast_forward") is not False:
        raise ContractError("fast-forward must remain forbidden")
    if fairness.get("simultaneous_buttons") is not False:
        raise ContractError("simultaneous buttons must remain forbidden")
    evidence = _mapping(contract.get("evidence"), "evidence")
    _exact_keys(
        evidence,
        {
            "measurement_schema",
            "soak_report_schema",
            "raw_frames_allowed_in_git",
            "measured_values_require_positive_sample_count",
            "measured_values_require_uncertainty",
            "measured_values_require_artifact_hashes",
        },
        "evidence",
    )
    if evidence.get("raw_frames_allowed_in_git") is not False:
        raise ContractError("raw original-game frames must remain outside git")
    expected_evidence = {
        "measurement_schema": EVIDENCE_SCHEMA,
        "soak_report_schema": REPORT_SCHEMA,
        "raw_frames_allowed_in_git": False,
        "measured_values_require_positive_sample_count": True,
        "measured_values_require_uncertainty": True,
        "measured_values_require_artifact_hashes": True,
    }
    for key, expected in expected_evidence.items():
        if evidence.get(key) != expected:
            raise ContractError(f"evidence.{key} changed")


def _validate_provisional(contract: Mapping[str, Any]) -> None:
    if contract.get("live_deployment_enabled") is not False:
        raise ContractError("provisional contract must disable live deployment")
    blockers = _string_list(contract.get("live_deployment_blockers"), "blockers")
    if "explicit_targeted_button_down_up_unavailable" not in blockers:
        raise ContractError(
            "the current explicit down/up capability blocker is missing"
        )
    forbidden = {
        "frame_rate_hz",
        "latency_seconds",
        "game_poll_latency_seconds",
        "press_duration_seconds",
        "release_duration_seconds",
    }
    for name in EMPIRICAL_SECTIONS:
        section = _mapping(contract.get(name), name)
        _exact_keys(section, BASE_SECTION_FIELDS[name], name)
        if section.get("status") != "unmeasured":
            raise ContractError(f"{name}.status must remain unmeasured")
        leaked = forbidden.intersection(section)
        if leaked:
            raise ContractError(
                f"{name} contains measurement-looking placeholders: {sorted(leaked)}"
            )


def _validate_measured(contract: Mapping[str, Any]) -> None:
    status = contract.get("measurement_status")
    if status == "measured_pending_review":
        if contract.get("live_deployment_enabled") is not False:
            raise ContractError("unreviewed measurements must not enable deployment")
        blockers = contract.get("live_deployment_blockers")
        if blockers != ["measurement_bundle_requires_human_review"]:
            raise ContractError("pending measurements require the review blocker")
    elif contract.get("live_deployment_enabled") is not True:
        raise ContractError("reviewed measured contract must enable live deployment")
    elif contract.get("live_deployment_blockers") != []:
        raise ContractError("reviewed measured contract must have no blockers")
    _sha256(contract.get("measurement_bundle_sha256"), "measurement_bundle_sha256")
    _sha256(contract.get("soak_report_sha256"), "soak_report_sha256")
    provenance = _mapping(contract.get("runtime_provenance"), "runtime_provenance")
    _exact_keys(
        provenance,
        {
            "game_executable_sha256",
            "box2d_sha256",
            "dxlib_sha256",
            "game_config_sha256",
            "measurement_tool_sha256",
            "wine_prefix_sha256",
            "runtime",
            "hardware_id",
        },
        "runtime_provenance",
    )
    for key in (
        "game_executable_sha256",
        "box2d_sha256",
        "dxlib_sha256",
        "game_config_sha256",
        "measurement_tool_sha256",
        "wine_prefix_sha256",
    ):
        _sha256(provenance.get(key), f"runtime_provenance.{key}")
    _safe_label(provenance.get("runtime"), "runtime_provenance.runtime")
    _safe_label(provenance.get("hardware_id"), "runtime_provenance.hardware_id")
    for name in EMPIRICAL_SECTIONS:
        section = _mapping(contract.get(name), name)
        allowed = (
            BASE_SECTION_FIELDS[name]
            | MEASURED_COMMON_FIELDS
            | MEASURED_EXTRA_FIELDS[name]
        )
        if (
            name == "cursor"
            and section.get("travel_model") == "preregistered_immaterial"
        ):
            allowed = allowed | {"preregistration_sha256"}
        _exact_keys(section, allowed, name)
        if section.get("status") != "measured":
            raise ContractError(f"{name}.status must be measured")
        _positive_int(section.get("sample_count"), f"{name}.sample_count")
        _safe_label(section.get("uncertainty_method"), f"{name}.uncertainty_method")
        if section.get("provenance_category") not in {
            "official",
            "observed",
            "community",
            "inferred",
        }:
            raise ContractError(f"{name}.provenance_category is invalid")
        for index, experiment_id in enumerate(
            _string_list(section.get("experiment_ids"), f"{name}.experiment_ids")
        ):
            _safe_label(experiment_id, f"{name}.experiment_ids[{index}]")
        hashes = _string_list(section.get("artifact_sha256"), f"{name}.artifact_sha256")
        for index, digest in enumerate(hashes):
            _sha256(digest, f"{name}.artifact_sha256[{index}]")

    for section_name, fields in MEASUREMENT_FIELDS.items():
        section = _mapping(contract[section_name], section_name)
        measurements = _mapping(
            section.get("measurements"), f"{section_name}.measurements"
        )
        _exact_keys(
            measurements,
            set(fields),
            f"{section_name}.measurements",
        )
        for field in fields:
            _validate_metric(
                measurements.get(field),
                field,
                f"{section_name}.measurements.{field}",
            )

    cursor = _mapping(contract["cursor"], "cursor")
    travel_model = cursor.get("travel_model")
    if travel_model not in {
        "measured_speed_acceleration",
        "abstract_coordinate_fixed_rate",
        "preregistered_immaterial",
    }:
        raise ContractError(
            "cursor.travel_model does not satisfy the fairness contract"
        )
    _safe_label(cursor.get("quantization"), "cursor.quantization")
    if cursor.get("path_logging") is not True:
        raise ContractError("cursor proposed/executed path logging must be enabled")
    if cursor.get("cursor_retention") not in {"retained", "measured_episode_reset"}:
        raise ContractError("cursor retention behavior is not frozen")
    cursor_measurements = _mapping(cursor.get("measurements"), "cursor.measurements")
    cursor_fields = {
        "measured_speed_acceleration": (
            "maximum_speed_client_pixels_per_second",
            "maximum_acceleration_client_pixels_per_second2",
        ),
        "abstract_coordinate_fixed_rate": ("fixed_action_rate_hz",),
        "preregistered_immaterial": (),
    }[travel_model]
    _exact_keys(cursor_measurements, set(cursor_fields), "cursor.measurements")
    for field in cursor_fields:
        _validate_metric(
            cursor_measurements.get(field),
            field,
            f"cursor.measurements.{field}",
        )
    if travel_model == "preregistered_immaterial":
        _sha256(cursor.get("preregistration_sha256"), "cursor.preregistration_sha256")
    calibration = _mapping(contract["coordinate_calibration"], "coordinate_calibration")
    if calibration.get("click_sweep_dimensions", 0) < 2:
        raise ContractError("coordinate calibration requires a 2-D click sweep")
    if calibration.get("continuous_drift_check") is not True:
        raise ContractError("continuous coordinate drift checks must be enabled")
    click = _mapping(contract["click_macro"], "click_macro")
    if (
        click.get("input_provider_capability")
        != "targeted_edges_broker_deadline_claim_neutralization"
    ):
        raise ContractError("measured click macro lacks broker-enforced release safety")
    if click.get("weak_button") != "left" or click.get("strong_button") != "right":
        raise ContractError("measured weak/strong button mapping is invalid")


def validate_deployment_contract(contract: Mapping[str, Any]) -> str:
    _validate_common(contract)
    status = contract.get("measurement_status")
    if status == "provisional_unmeasured":
        _validate_provisional(contract)
    elif status in {"measured", "measured_pending_review"}:
        _validate_measured(contract)
    else:
        raise ContractError(
            "measurement_status must be provisional_unmeasured, "
            "measured_pending_review, or measured"
        )
    return str(status)


def _validate_evidence(
    evidence: Mapping[str, Any],
    soak_report: Mapping[str, Any],
    event_path: Path,
    threshold_path: Path,
) -> str:
    _exact_keys(
        evidence,
        {
            "schema_version",
            "contract_version",
            "status",
            "soak_experiment_ids",
            "soak_event_stream_sha256",
            "soak_threshold_config_sha256",
            "soak_report_sha256",
            "provenance",
            "sections",
        },
        "evidence",
    )
    if evidence.get("schema_version") != EVIDENCE_SCHEMA:
        raise ContractError(f"evidence schema_version must be {EVIDENCE_SCHEMA}")
    if evidence.get("contract_version") != "deployment-v1":
        raise ContractError("evidence targets the wrong deployment contract")
    if evidence.get("status") != "measured":
        raise ContractError("evidence status must be measured")
    try:
        soak_sha256 = verify_report(soak_report, event_path, threshold_path)
    except EvidenceError as exc:
        raise ContractError(f"invalid soak report: {exc}") from exc
    stream = _mapping(soak_report.get("event_stream"), "soak event_stream")
    if evidence.get("soak_event_stream_sha256") != stream.get("sha256"):
        raise ContractError("evidence does not bind the soak event stream")
    if evidence.get("soak_threshold_config_sha256") != soak_report.get(
        "threshold_config_sha256"
    ):
        raise ContractError("evidence does not bind the soak threshold config")
    if evidence.get("soak_report_sha256") != soak_sha256:
        raise ContractError("evidence does not bind the verified soak report")
    soak_experiment_ids = _string_list(
        evidence.get("soak_experiment_ids"), "soak_experiment_ids"
    )
    if list(soak_experiment_ids) != soak_report["experiments"]["observed"]:
        raise ContractError("evidence soak experiments do not match the report")
    soak_experiment_set = set(soak_experiment_ids)
    provenance = _mapping(evidence.get("provenance"), "provenance")
    _exact_keys(
        provenance,
        {
            "game_executable_sha256",
            "box2d_sha256",
            "dxlib_sha256",
            "game_config_sha256",
            "measurement_tool_sha256",
            "wine_prefix_sha256",
            "runtime",
            "hardware_id",
        },
        "provenance",
    )
    for key in (
        "game_executable_sha256",
        "box2d_sha256",
        "dxlib_sha256",
        "game_config_sha256",
        "measurement_tool_sha256",
        "wine_prefix_sha256",
    ):
        _sha256(provenance.get(key), f"provenance.{key}")
    _safe_label(provenance.get("runtime"), "provenance.runtime")
    _safe_label(provenance.get("hardware_id"), "provenance.hardware_id")
    report_provenance = soak_report["provenance"]["observed"]
    for key in (
        "game_executable_sha256",
        "measurement_tool_sha256",
        "wine_prefix_sha256",
    ):
        if report_provenance.get(key) != provenance[key]:
            raise ContractError(f"soak report provenance does not bind {key}")

    sections = _mapping(evidence.get("sections"), "sections")
    _exact_keys(sections, set(EMPIRICAL_SECTIONS), "sections")
    for name in EMPIRICAL_SECTIONS:
        section = _mapping(sections.get(name), f"sections.{name}")
        allowed = MEASURED_COMMON_FIELDS | MEASURED_EXTRA_FIELDS[name]
        if (
            name == "cursor"
            and section.get("travel_model") == "preregistered_immaterial"
        ):
            allowed = allowed | {"preregistration_sha256"}
        _exact_keys(section, allowed, f"sections.{name}")
        _positive_int(section.get("sample_count"), f"sections.{name}.sample_count")
        _safe_label(
            section.get("uncertainty_method"),
            f"sections.{name}.uncertainty_method",
        )
        if section.get("provenance_category") not in {
            "official",
            "observed",
            "community",
            "inferred",
        }:
            raise ContractError(f"sections.{name}.provenance_category is invalid")
        for index, experiment_id in enumerate(
            _string_list(
                section.get("experiment_ids"), f"sections.{name}.experiment_ids"
            )
        ):
            _safe_label(experiment_id, f"sections.{name}.experiment_ids[{index}]")
            if experiment_id not in soak_experiment_set:
                raise ContractError(
                    f"sections.{name} cites an experiment absent from the soak"
                )
        hashes = _string_list(
            section.get("artifact_sha256"), f"sections.{name}.artifact_sha256"
        )
        for index, digest in enumerate(hashes):
            _sha256(digest, f"sections.{name}.artifact_sha256[{index}]")
        required_artifacts = {
            str(stream["sha256"]),
            str(soak_report["threshold_config_sha256"]),
            soak_sha256,
        }
        if not required_artifacts.issubset(hashes):
            raise ContractError(
                f"sections.{name}.artifact_sha256 is not bound to "
                "verified soak artifacts"
            )
        if name in MEASUREMENT_FIELDS:
            measurements = _mapping(
                section.get("measurements"), f"sections.{name}.measurements"
            )
            _exact_keys(
                measurements,
                set(MEASUREMENT_FIELDS[name]),
                f"sections.{name}.measurements",
            )
            for field in MEASUREMENT_FIELDS[name]:
                _validate_metric(
                    measurements.get(field),
                    field,
                    f"sections.{name}.measurements.{field}",
                )
                if measurements[field]["sample_count"] != section["sample_count"]:
                    raise ContractError(
                        f"sections.{name}.{field} sample count disagrees"
                    )

    cursor = sections["cursor"]
    travel_model = cursor.get("travel_model")
    if travel_model not in {
        "measured_speed_acceleration",
        "abstract_coordinate_fixed_rate",
        "preregistered_immaterial",
    }:
        raise ContractError(
            "evidence cursor travel_model is not a legal fairness choice"
        )
    _safe_label(cursor.get("quantization"), "sections.cursor.quantization")
    if cursor.get("path_logging") is not True:
        raise ContractError(
            "evidence must enable proposed/executed cursor path logging"
        )
    if cursor.get("cursor_retention") not in {"retained", "measured_episode_reset"}:
        raise ContractError("evidence must freeze cursor retention")
    cursor_measurements = _mapping(
        cursor.get("measurements"), "sections.cursor.measurements"
    )
    cursor_fields = {
        "measured_speed_acceleration": (
            "maximum_speed_client_pixels_per_second",
            "maximum_acceleration_client_pixels_per_second2",
        ),
        "abstract_coordinate_fixed_rate": ("fixed_action_rate_hz",),
        "preregistered_immaterial": (),
    }[travel_model]
    _exact_keys(cursor_measurements, set(cursor_fields), "sections.cursor.measurements")
    for field in cursor_fields:
        _validate_metric(
            cursor_measurements.get(field),
            field,
            f"sections.cursor.measurements.{field}",
        )
        if cursor_measurements[field]["sample_count"] != cursor["sample_count"]:
            raise ContractError(f"sections.cursor.{field} sample count disagrees")
    if travel_model == "preregistered_immaterial":
        _sha256(
            cursor.get("preregistration_sha256"),
            "sections.cursor.preregistration_sha256",
        )
    click = sections["click_macro"]
    if (
        click.get("input_provider_capability")
        != "targeted_edges_broker_deadline_claim_neutralization"
    ):
        raise ContractError("evidence lacks broker-enforced release safety")
    if click.get("weak_button") != "left" or click.get("strong_button") != "right":
        raise ContractError("evidence weak/strong button mapping is invalid")
    coordinate = sections["coordinate_calibration"]
    if coordinate.get("click_sweep_dimensions") != 2:
        raise ContractError("evidence must contain a 2-D click sweep")
    if coordinate.get("continuous_drift_check") is not True:
        raise ContractError("evidence must enable continuous drift checks")
    return soak_sha256


def finalize_deployment_contract(
    base: Mapping[str, Any],
    evidence: Mapping[str, Any],
    soak_report: Mapping[str, Any],
    event_path: Path,
    threshold_path: Path,
) -> dict[str, Any]:
    """Return a measured contract only after complete evidence validation."""

    validate_deployment_contract(base)
    if base.get("measurement_status") != "provisional_unmeasured":
        raise ContractError("only the provisional checked-in contract may be finalized")
    if canonical_json_sha256(base) != PROVISIONAL_BASE_SHA256:
        raise ContractError("base contract is not the checked provisional contract")
    soak_sha256 = _validate_evidence(evidence, soak_report, event_path, threshold_path)
    result = deepcopy(dict(base))
    result["measurement_status"] = "measured_pending_review"
    result["live_deployment_enabled"] = False
    result["live_deployment_blockers"] = ["measurement_bundle_requires_human_review"]
    result["measurement_bundle_sha256"] = canonical_json_sha256(evidence)
    result["soak_report_sha256"] = soak_sha256
    result["runtime_provenance"] = deepcopy(dict(evidence["provenance"]))
    for name in EMPIRICAL_SECTIONS:
        result[name] = {
            **deepcopy(dict(base[name])),
            **deepcopy(dict(evidence["sections"][name])),
        }
        result[name]["status"] = "measured"
    validate_deployment_contract(result)
    return result


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=True)


def _toml_value(value: object) -> str:
    if isinstance(value, str):
        return _toml_string(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float) and math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise ContractError(f"cannot serialize TOML value of type {type(value).__name__}")


def render_toml(document: Mapping[str, Any]) -> str:
    """Render the contract's primitive/nested mapping subset deterministically."""

    lines: list[str] = []

    def emit(table: Mapping[str, Any], path: tuple[str, ...]) -> None:
        for key in table:
            if not isinstance(key, str) or TOML_KEY.fullmatch(key) is None:
                raise ContractError(f"unsafe TOML key: {key!r}")
        scalars = [
            (key, value)
            for key, value in table.items()
            if not isinstance(value, Mapping)
        ]
        children = [
            (key, value) for key, value in table.items() if isinstance(value, Mapping)
        ]
        if path:
            if lines and lines[-1]:
                lines.append("")
            lines.append("[" + ".".join(path) + "]")
        for key, value in scalars:
            lines.append(f"{key} = {_toml_value(value)}")
        for key, value in children:
            emit(value, (*path, key))

    emit(document, ())
    return "\n".join(lines) + "\n"
