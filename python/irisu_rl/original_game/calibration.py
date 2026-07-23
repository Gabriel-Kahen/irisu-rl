"""Strict R4b calibration journals and derived deployment evidence.

The journal stores only typed scalar measurements.  It deliberately has no
field capable of carrying pixels, claim tokens, window titles, or filesystem
paths.  Every deployment statistic is rebuilt from the verified journal; a
caller cannot supply precomputed quantiles or sample counts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Self

from .evidence import (
    EvidenceError,
    canonical_json_bytes,
    load_json_document,
    load_thresholds,
    verify_report,
)

PLAN_SCHEMA = "r4b-calibration-plan-v1"
RECORD_SCHEMA = "r4b-calibration-record-v1"
DEPLOYMENT_EVIDENCE_SCHEMA = "r4a-deployment-measurements-v1"
SAFE_PROVIDER_CAPABILITY = "targeted_edges_broker_deadline_claim_neutralization"
CLIENT_PIXEL_QUANTIZATION = "floor(normalized * extent), clamped to [0, extent - 1]"
ZERO_SHA256 = "0" * 64
MAX_LINE_BYTES = 1 << 20
MAX_ACTIONS = 100_000
MAX_RUNTIME_SECONDS = 86_400.0
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SAFE_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

METRIC_SPECS: Mapping[str, tuple[str, str, str]] = {
    "gameplay_period_seconds": ("wait_duration", "seconds", "max"),
    "scheduler_error_seconds": ("wait_duration", "seconds", "max"),
    "press_duration_seconds": ("click_macro", "seconds", "max"),
    "release_duration_seconds": ("click_macro", "seconds", "max"),
    "maximum_clicks_per_second": (
        "click_macro",
        "clicks/second",
        "min",
    ),
    "frame_rate_hz": ("capture", "frames/second", "min"),
    "request_to_completion_seconds": ("capture", "seconds", "max"),
    "stale_after_seconds": ("capture", "seconds", "max"),
    "injection_to_poll_seconds": ("effect_timing", "seconds", "max"),
    "effect_to_visible_seconds": ("effect_timing", "seconds", "max"),
    "request_to_visible_seconds": ("effect_timing", "seconds", "max"),
    "residual_client_pixels": (
        "coordinate_calibration",
        "client_pixels",
        "max",
    ),
    "fixed_action_rate_hz": ("cursor", "actions/second", "min"),
}
METRIC_FIELDS = tuple(METRIC_SPECS)
SECTION_FIELDS: Mapping[str, tuple[str, ...]] = {
    section: tuple(
        field for field, (owner, _, _) in METRIC_SPECS.items() if owner == section
    )
    for section in (
        "wait_duration",
        "click_macro",
        "cursor",
        "capture",
        "effect_timing",
        "coordinate_calibration",
    )
}
PROVENANCE_FIELDS = frozenset(
    {
        "game_executable_sha256",
        "box2d_sha256",
        "dxlib_sha256",
        "game_config_sha256",
        "measurement_tool_sha256",
        "runtime",
        "hardware_id",
    }
)


class CalibrationError(ValueError):
    """A calibration plan, journal, or derived artifact is unsafe."""


class JournalPublicationError(FileExistsError):
    """A private journal could not be created without replacing an entry."""


def _exact_keys(
    value: Mapping[str, Any], expected: set[str] | frozenset[str], label: str
) -> None:
    actual = set(value)
    if actual != set(expected):
        raise CalibrationError(
            f"{label} fields disagree: "
            f"missing={sorted(set(expected) - actual)}, "
            f"extra={sorted(actual - set(expected))}"
        )


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise CalibrationError(f"{label} must be an object with string keys")
    return value


def _finite(
    value: object,
    label: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CalibrationError(f"{label} must be finite")
    if positive and result <= 0:
        raise CalibrationError(f"{label} must be positive")
    if nonnegative and result < 0:
        raise CalibrationError(f"{label} must be nonnegative")
    return result


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CalibrationError(f"{label} must be a positive integer")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value == ZERO_SHA256
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CalibrationError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or IDENTIFIER.fullmatch(value) is None:
        raise CalibrationError(f"{label} must be a safe identifier")
    return value


def _safe_label(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise CalibrationError(f"{label} must be nonempty bounded text")
    lowered = value.lower()
    if any(character in value for character in ("/", "\\", "\n", "\r", "\0")) or any(
        term in lowered for term in ("token", "secret", "private_path")
    ):
        raise CalibrationError(f"{label} contains private or unsafe text")
    return value


def _unique_numbers(
    value: object, label: str, *, minimum_count: int
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) < minimum_count:
        raise CalibrationError(
            f"{label} must contain at least {minimum_count} coordinates"
        )
    result = tuple(_finite(item, f"{label}[]") for item in value)
    if len(set(result)) != len(result):
        raise CalibrationError(f"{label} must not contain duplicates")
    return result


@dataclass(frozen=True, slots=True)
class SweepCell:
    experiment_id: str
    client_x: float
    client_y: float
    button: str
    repetition: int

    def manifest(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "client_x": self.client_x,
            "client_y": self.client_y,
            "button": self.button,
            "repetition": self.repetition,
        }


@lru_cache(maxsize=32)
def _ordered_cells(
    experiment_ids: tuple[str, ...],
    x_coordinates: tuple[float, ...],
    y_coordinates: tuple[float, ...],
    repetitions: int,
    order_seed_sha256: str,
) -> tuple[SweepCell, ...]:
    result: list[SweepCell] = []
    seed = bytes.fromhex(order_seed_sha256)
    for experiment_id in experiment_ids:
        process_cells = [
            SweepCell(experiment_id, x, y, button, repetition)
            for x in x_coordinates
            for y in y_coordinates
            for button in ("weak", "strong")
            for repetition in range(1, repetitions + 1)
        ]
        process_cells.sort(
            key=lambda cell: (
                hashlib.sha256(seed + canonical_json_bytes(cell.manifest())).digest(),
                canonical_json_bytes(cell.manifest()),
            )
        )
        result.extend(process_cells)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class CalibrationPlan:
    experiment_ids: tuple[str, ...]
    provider_build_sha256: str
    client_width: float
    client_height: float
    x_coordinates: tuple[float, ...]
    y_coordinates: tuple[float, ...]
    repetitions: int
    order_seed_sha256: str
    cursor_quantization: str
    cursor_retention: str
    maximum_actions: int
    maximum_runtime_seconds: float
    episode_envelope_ticks: int
    nominal_gameplay_hz: float
    minimum_soak_duration_seconds: float
    maximum_measurement_gap_seconds: float
    soak_threshold_config_sha256: str
    instrument_resolution: Mapping[str, float]
    bootstrap_replicates: int
    block_length: int
    minimum_confirmed_actions: int
    minimum_registration_rate: float
    provenance: Mapping[str, str]

    @classmethod
    def from_mapping(cls, value: object) -> CalibrationPlan:
        root = _mapping(value, "calibration plan")
        _exact_keys(
            root,
            {
                "schema",
                "contract_version",
                "experiment_ids",
                "provider",
                "sweep",
                "cursor_protocol",
                "limits",
                "soak",
                "instrument_resolution",
                "uncertainty",
                "acceptance",
                "provenance",
            },
            "calibration plan",
        )
        if root.get("schema") != PLAN_SCHEMA:
            raise CalibrationError(f"calibration plan schema must be {PLAN_SCHEMA}")
        if root.get("contract_version") != "deployment-v1":
            raise CalibrationError("calibration plan targets the wrong contract")

        identifiers = root.get("experiment_ids")
        if not isinstance(identifiers, list) or len(identifiers) < 3:
            raise CalibrationError(
                "calibration plan requires at least three fresh-process experiment IDs"
            )
        experiment_ids = tuple(
            _identifier(item, "experiment_ids[]") for item in identifiers
        )
        if len(set(experiment_ids)) != len(experiment_ids):
            raise CalibrationError("experiment IDs must be unique")

        provider = _mapping(root.get("provider"), "provider")
        _exact_keys(
            provider,
            {"required_capability", "provider_build_sha256"},
            "provider",
        )
        if provider.get("required_capability") != SAFE_PROVIDER_CAPABILITY:
            raise CalibrationError("plan does not require the safe provider capability")
        provider_build_sha256 = _sha256(
            provider.get("provider_build_sha256"), "provider.provider_build_sha256"
        )

        sweep = _mapping(root.get("sweep"), "sweep")
        _exact_keys(
            sweep,
            {
                "client_width",
                "client_height",
                "x_coordinates",
                "y_coordinates",
                "buttons",
                "repetitions",
                "order_algorithm",
                "order_seed_sha256",
            },
            "sweep",
        )
        client_width = _finite(
            sweep.get("client_width"), "sweep.client_width", positive=True
        )
        client_height = _finite(
            sweep.get("client_height"), "sweep.client_height", positive=True
        )
        if client_width != 640.0 or client_height != 480.0:
            raise CalibrationError("deployment-v1 sweep must use the 640x480 client")
        xs = _unique_numbers(
            sweep.get("x_coordinates"), "sweep.x_coordinates", minimum_count=9
        )
        ys = _unique_numbers(
            sweep.get("y_coordinates"), "sweep.y_coordinates", minimum_count=7
        )
        if any(
            not coordinate.is_integer() or not 0 <= coordinate < client_width
            for coordinate in xs
        ):
            raise CalibrationError(
                "sweep x coordinates must be integer pixels inside the client"
            )
        if any(
            not coordinate.is_integer() or not 0 <= coordinate < client_height
            for coordinate in ys
        ):
            raise CalibrationError(
                "sweep y coordinates must be integer pixels inside the client"
            )
        if sweep.get("buttons") != ["weak", "strong"]:
            raise CalibrationError("sweep must include weak and strong buttons")
        repetitions = _positive_int(sweep.get("repetitions"), "sweep.repetitions")
        if repetitions * len(experiment_ids) < 3:
            raise CalibrationError(
                "each coordinate/button needs at least three cross-process repeats"
            )
        if sweep.get("order_algorithm") != "sha256-v1":
            raise CalibrationError("sweep order algorithm must be sha256-v1")
        order_seed = _sha256(sweep.get("order_seed_sha256"), "sweep.order_seed_sha256")

        cursor = _mapping(root.get("cursor_protocol"), "cursor_protocol")
        _exact_keys(
            cursor,
            {
                "travel_model",
                "quantization",
                "cursor_retention",
                "path_logging",
            },
            "cursor_protocol",
        )
        if cursor.get("travel_model") != "abstract_coordinate_fixed_rate":
            raise CalibrationError(
                "R4b evidence builder requires abstract_coordinate_fixed_rate"
            )
        quantization = _safe_label(
            cursor.get("quantization"), "cursor_protocol.quantization"
        )
        if quantization != CLIENT_PIXEL_QUANTIZATION:
            raise CalibrationError(
                "cursor quantization must match deployment-v1 execution lowering"
            )
        retention = cursor.get("cursor_retention")
        if retention not in {"retained", "measured_episode_reset"}:
            raise CalibrationError("cursor retention protocol is invalid")
        if cursor.get("path_logging") is not True:
            raise CalibrationError("proposed/executed cursor path logging is required")

        limits = _mapping(root.get("limits"), "limits")
        _exact_keys(limits, {"maximum_actions", "maximum_runtime_seconds"}, "limits")
        maximum_actions = _positive_int(
            limits.get("maximum_actions"), "limits.maximum_actions"
        )
        expected_actions = len(experiment_ids) * len(xs) * len(ys) * 2 * repetitions
        if maximum_actions != expected_actions or maximum_actions > MAX_ACTIONS:
            raise CalibrationError(
                "maximum_actions must equal the complete preregistered sweep"
            )
        maximum_runtime = _finite(
            limits.get("maximum_runtime_seconds"),
            "limits.maximum_runtime_seconds",
            positive=True,
        )
        if maximum_runtime > MAX_RUNTIME_SECONDS:
            raise CalibrationError("maximum runtime exceeds the safety bound")

        soak = _mapping(root.get("soak"), "soak")
        _exact_keys(
            soak,
            {
                "episode_envelope_ticks",
                "nominal_gameplay_hz",
                "minimum_duration_seconds",
                "maximum_measurement_gap_seconds",
                "threshold_config_sha256",
            },
            "soak",
        )
        envelope = _positive_int(
            soak.get("episode_envelope_ticks"), "soak.episode_envelope_ticks"
        )
        if envelope != 8192:
            raise CalibrationError("R4b episode envelope must remain 8192 ticks")
        gameplay_hz = _finite(
            soak.get("nominal_gameplay_hz"),
            "soak.nominal_gameplay_hz",
            positive=True,
        )
        if gameplay_hz != 50.0:
            raise CalibrationError("R4b nominal gameplay cadence must remain 50 Hz")
        minimum_soak = _finite(
            soak.get("minimum_duration_seconds"),
            "soak.minimum_duration_seconds",
            positive=True,
        )
        if minimum_soak <= envelope / gameplay_hz:
            raise CalibrationError(
                "soak must exceed the complete 8192-tick episode envelope"
            )
        if minimum_soak > maximum_runtime:
            raise CalibrationError("soak minimum exceeds the run-time safety cap")
        maximum_gap = _finite(
            soak.get("maximum_measurement_gap_seconds"),
            "soak.maximum_measurement_gap_seconds",
            positive=True,
        )
        if maximum_gap > minimum_soak:
            raise CalibrationError("soak measurement gap exceeds its duration")

        resolutions = _mapping(
            root.get("instrument_resolution"), "instrument_resolution"
        )
        _exact_keys(resolutions, set(METRIC_FIELDS), "instrument_resolution")
        normalized_resolution = {
            field: _finite(
                resolutions[field],
                f"instrument_resolution.{field}",
                positive=True,
            )
            for field in METRIC_FIELDS
        }

        uncertainty = _mapping(root.get("uncertainty"), "uncertainty")
        _exact_keys(
            uncertainty,
            {
                "method",
                "confidence_level",
                "bootstrap_replicates",
                "block_length",
            },
            "uncertainty",
        )
        if uncertainty.get("method") != "moving_block_bootstrap_v1":
            raise CalibrationError(
                "uncertainty method must be moving_block_bootstrap_v1"
            )
        if uncertainty.get("confidence_level") != 0.95:
            raise CalibrationError("R4b confidence level must be 0.95")
        replicates = _positive_int(
            uncertainty.get("bootstrap_replicates"),
            "uncertainty.bootstrap_replicates",
        )
        if replicates < 200 or replicates > 100_000:
            raise CalibrationError(
                "bootstrap replicates must be between 200 and 100000"
            )
        block_length = _positive_int(
            uncertainty.get("block_length"), "uncertainty.block_length"
        )
        samples_per_process = len(xs) * len(ys) * 2 * repetitions
        if block_length < 2 or block_length > samples_per_process:
            raise CalibrationError("bootstrap block length is outside one process run")

        acceptance = _mapping(root.get("acceptance"), "acceptance")
        _exact_keys(
            acceptance,
            {
                "minimum_confirmed_actions",
                "minimum_registration_rate",
                "registration_interval_method",
            },
            "acceptance",
        )
        confirmed = _positive_int(
            acceptance.get("minimum_confirmed_actions"),
            "acceptance.minimum_confirmed_actions",
        )
        if confirmed < 512 or confirmed > maximum_actions:
            raise CalibrationError(
                "minimum confirmed actions must be in [512, maximum_actions]"
            )
        registration_rate = _finite(
            acceptance.get("minimum_registration_rate"),
            "acceptance.minimum_registration_rate",
            positive=True,
        )
        if registration_rate > 1:
            raise CalibrationError("minimum registration rate must not exceed one")
        if acceptance.get("registration_interval_method") != "wilson-score-v1":
            raise CalibrationError(
                "registration interval method must be wilson-score-v1"
            )

        provenance = _mapping(root.get("provenance"), "provenance")
        _exact_keys(provenance, PROVENANCE_FIELDS, "provenance")
        normalized_provenance: dict[str, str] = {}
        for key in PROVENANCE_FIELDS:
            normalized_provenance[key] = (
                _safe_label(provenance[key], f"provenance.{key}")
                if key in {"runtime", "hardware_id"}
                else _sha256(provenance[key], f"provenance.{key}")
            )

        return cls(
            experiment_ids,
            provider_build_sha256,
            client_width,
            client_height,
            xs,
            ys,
            repetitions,
            order_seed,
            quantization,
            str(retention),
            maximum_actions,
            maximum_runtime,
            envelope,
            gameplay_hz,
            minimum_soak,
            maximum_gap,
            _sha256(
                soak.get("threshold_config_sha256"),
                "soak.threshold_config_sha256",
            ),
            MappingProxyType(normalized_resolution),
            replicates,
            block_length,
            confirmed,
            registration_rate,
            MappingProxyType(normalized_provenance),
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema": PLAN_SCHEMA,
            "contract_version": "deployment-v1",
            "experiment_ids": list(self.experiment_ids),
            "provider": {
                "required_capability": SAFE_PROVIDER_CAPABILITY,
                "provider_build_sha256": self.provider_build_sha256,
            },
            "sweep": {
                "client_width": self.client_width,
                "client_height": self.client_height,
                "x_coordinates": list(self.x_coordinates),
                "y_coordinates": list(self.y_coordinates),
                "buttons": ["weak", "strong"],
                "repetitions": self.repetitions,
                "order_algorithm": "sha256-v1",
                "order_seed_sha256": self.order_seed_sha256,
            },
            "cursor_protocol": {
                "travel_model": "abstract_coordinate_fixed_rate",
                "quantization": self.cursor_quantization,
                "cursor_retention": self.cursor_retention,
                "path_logging": True,
            },
            "limits": {
                "maximum_actions": self.maximum_actions,
                "maximum_runtime_seconds": self.maximum_runtime_seconds,
            },
            "soak": {
                "episode_envelope_ticks": self.episode_envelope_ticks,
                "nominal_gameplay_hz": self.nominal_gameplay_hz,
                "minimum_duration_seconds": self.minimum_soak_duration_seconds,
                "maximum_measurement_gap_seconds": (
                    self.maximum_measurement_gap_seconds
                ),
                "threshold_config_sha256": self.soak_threshold_config_sha256,
            },
            "instrument_resolution": dict(self.instrument_resolution),
            "uncertainty": {
                "method": "moving_block_bootstrap_v1",
                "confidence_level": 0.95,
                "bootstrap_replicates": self.bootstrap_replicates,
                "block_length": self.block_length,
            },
            "acceptance": {
                "minimum_confirmed_actions": self.minimum_confirmed_actions,
                "minimum_registration_rate": self.minimum_registration_rate,
                "registration_interval_method": "wilson-score-v1",
            },
            "provenance": dict(self.provenance),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.manifest())).hexdigest()

    @property
    def expected_cells(self) -> tuple[SweepCell, ...]:
        return _ordered_cells(
            self.experiment_ids,
            self.x_coordinates,
            self.y_coordinates,
            self.repetitions,
            self.order_seed_sha256,
        )


def load_calibration_plan(path: str | os.PathLike[str]) -> CalibrationPlan:
    return CalibrationPlan.from_mapping(
        load_json_document(path, "R4b calibration plan")
    )


@dataclass(frozen=True, slots=True)
class CalibrationSample:
    experiment_id: str
    client_x: float
    client_y: float
    button: str
    repetition: int
    provider_capability: str
    provider_build_sha256: str
    registered: bool
    measurements: Mapping[str, float] | None

    @classmethod
    def from_mapping(cls, value: object) -> CalibrationSample:
        sample = _mapping(value, "calibration sample")
        _exact_keys(
            sample,
            {
                "experiment_id",
                "client_x",
                "client_y",
                "button",
                "repetition",
                "provider_capability",
                "provider_build_sha256",
                "registered",
                "measurements",
            },
            "calibration sample",
        )
        registered = sample.get("registered")
        if type(registered) is not bool:
            raise CalibrationError("calibration sample registered must be boolean")
        raw_measurements = sample.get("measurements")
        measurements: dict[str, float] | None
        if registered:
            values = _mapping(raw_measurements, "calibration sample measurements")
            _exact_keys(values, set(METRIC_FIELDS), "calibration sample measurements")
            measurements = {
                field: _finite(
                    values[field],
                    f"calibration sample measurements.{field}",
                    positive=field
                    in {
                        "gameplay_period_seconds",
                        "press_duration_seconds",
                        "release_duration_seconds",
                        "maximum_clicks_per_second",
                        "frame_rate_hz",
                        "stale_after_seconds",
                        "fixed_action_rate_hz",
                    },
                    nonnegative=field
                    not in {
                        "gameplay_period_seconds",
                        "press_duration_seconds",
                        "release_duration_seconds",
                        "maximum_clicks_per_second",
                        "frame_rate_hz",
                        "stale_after_seconds",
                        "fixed_action_rate_hz",
                    },
                )
                for field in METRIC_FIELDS
            }
            if (
                measurements["request_to_completion_seconds"]
                > measurements["stale_after_seconds"]
            ):
                raise CalibrationError(
                    "capture completion exceeds the declared stale threshold"
                )
            if measurements["request_to_visible_seconds"] < (
                measurements["injection_to_poll_seconds"]
                + measurements["effect_to_visible_seconds"]
            ):
                raise CalibrationError(
                    "request-to-visible latency is shorter than its causal components"
                )
        elif raw_measurements is not None:
            raise CalibrationError(
                "unregistered attempts must not fabricate measurement values"
            )
        else:
            measurements = None
        return cls(
            _identifier(sample.get("experiment_id"), "sample.experiment_id"),
            _finite(sample.get("client_x"), "sample.client_x"),
            _finite(sample.get("client_y"), "sample.client_y"),
            str(sample.get("button")),
            _positive_int(sample.get("repetition"), "sample.repetition"),
            str(sample.get("provider_capability")),
            _sha256(
                sample.get("provider_build_sha256"),
                "sample.provider_build_sha256",
            ),
            registered,
            MappingProxyType(measurements) if measurements is not None else None,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "client_x": self.client_x,
            "client_y": self.client_y,
            "button": self.button,
            "repetition": self.repetition,
            "provider_capability": self.provider_capability,
            "provider_build_sha256": self.provider_build_sha256,
            "registered": self.registered,
            "measurements": (
                dict(self.measurements) if self.measurements is not None else None
            ),
        }

    @property
    def cell(self) -> SweepCell:
        return SweepCell(
            self.experiment_id,
            self.client_x,
            self.client_y,
            self.button,
            self.repetition,
        )


def seal_calibration_record(
    plan: CalibrationPlan,
    sample: CalibrationSample | Mapping[str, Any],
    *,
    sequence: int,
    monotonic_ns: int,
    previous_sha256: str = ZERO_SHA256,
) -> dict[str, object]:
    if not isinstance(plan, CalibrationPlan):
        raise TypeError("plan must be a CalibrationPlan")
    parsed = (
        sample
        if isinstance(sample, CalibrationSample)
        else CalibrationSample.from_mapping(sample)
    )
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
        raise CalibrationError("journal sequence must be a positive integer")
    if (
        isinstance(monotonic_ns, bool)
        or not isinstance(monotonic_ns, int)
        or monotonic_ns < 0
    ):
        raise CalibrationError("journal monotonic_ns must be nonnegative")
    previous = (
        ZERO_SHA256
        if previous_sha256 == ZERO_SHA256
        else _sha256(previous_sha256, "previous_sha256")
    )
    unsigned: dict[str, object] = {
        "schema": RECORD_SCHEMA,
        "sequence": sequence,
        "monotonic_ns": monotonic_ns,
        "plan_sha256": plan.sha256,
        "sample": parsed.manifest(),
        "previous_sha256": previous,
    }
    unsigned["sha256"] = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    return unsigned


def encode_calibration_record(record: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(record) + b"\n"


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CalibrationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _decode_record(data: bytes, label: str) -> object:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                CalibrationError(f"non-finite JSON number: {value}")
            ),
        )
    except CalibrationError:
        raise
    except (UnicodeDecodeError, ValueError) as exc:
        raise CalibrationError(f"{label} is not valid UTF-8 JSON") from exc


def _validate_record(
    value: object,
    plan: CalibrationPlan,
    *,
    expected_sequence: int,
    expected_previous: str,
) -> tuple[dict[str, Any], CalibrationSample]:
    record = _mapping(value, "calibration record")
    _exact_keys(
        record,
        {
            "schema",
            "sequence",
            "monotonic_ns",
            "plan_sha256",
            "sample",
            "previous_sha256",
            "sha256",
        },
        "calibration record",
    )
    if record.get("schema") != RECORD_SCHEMA:
        raise CalibrationError("calibration record schema mismatch")
    if record.get("sequence") != expected_sequence:
        raise CalibrationError(
            f"journal sequence mismatch: expected {expected_sequence}"
        )
    timestamp = record.get("monotonic_ns")
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp < 0:
        raise CalibrationError("journal monotonic_ns must be nonnegative")
    if record.get("plan_sha256") != plan.sha256:
        raise CalibrationError("journal record is bound to a different plan")
    if record.get("previous_sha256") != expected_previous:
        raise CalibrationError("journal SHA-256 chain link mismatch")
    digest = _sha256(record.get("sha256"), "record.sha256")
    unsigned = dict(record)
    del unsigned["sha256"]
    expected_digest = hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest()
    if digest != expected_digest:
        raise CalibrationError("journal record SHA-256 does not match its contents")
    sample = CalibrationSample.from_mapping(record.get("sample"))
    if expected_sequence > len(plan.expected_cells):
        raise CalibrationError("journal contains more actions than the plan")
    expected_cell = plan.expected_cells[expected_sequence - 1]
    if sample.cell != expected_cell:
        raise CalibrationError("journal action disagrees with randomized sweep order")
    if (
        sample.provider_capability != SAFE_PROVIDER_CAPABILITY
        or sample.provider_build_sha256 != plan.provider_build_sha256
    ):
        raise CalibrationError("journal sample lacks the required safe provider")
    return dict(record), sample


@dataclass(frozen=True, slots=True)
class VerifiedCalibrationJournal:
    samples: tuple[CalibrationSample, ...]
    monotonic_ns: tuple[int, ...]
    input_sha256: str
    chain_head_sha256: str
    registration_lower_bound: float


def _wilson_lower(successes: int, attempts: int) -> float:
    if attempts <= 0 or not 0 <= successes <= attempts:
        raise CalibrationError("registration counts are invalid")
    z = NormalDist().inv_cdf(0.975)
    rate = successes / attempts
    denominator = 1 + z * z / attempts
    center = rate + z * z / (2 * attempts)
    spread = z * math.sqrt(
        rate * (1 - rate) / attempts + z * z / (4 * attempts * attempts)
    )
    return max(0.0, (center - spread) / denominator)


def verify_calibration_journal(
    path: str | os.PathLike[str],
    plan: CalibrationPlan,
    *,
    require_complete: bool = True,
) -> VerifiedCalibrationJournal:
    if not isinstance(plan, CalibrationPlan):
        raise TypeError("plan must be a CalibrationPlan")
    samples: list[CalibrationSample] = []
    timestamps: list[int] = []
    input_digest = hashlib.sha256()
    previous = ZERO_SHA256
    journal_path = Path(path)
    if (
        not journal_path.is_absolute()
        or SAFE_FILENAME.fullmatch(journal_path.name) is None
    ):
        raise CalibrationError(
            "calibration journal must use an absolute safe private path"
        )
    try:
        directory_descriptor = _open_private_directory(journal_path.parent)
    except JournalPublicationError as exc:
        raise CalibrationError(str(exc)) from exc
    try:
        descriptor = os.open(
            journal_path.name,
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
        ):
            os.close(descriptor)
            raise CalibrationError(
                "calibration journal must be an owned regular file with mode 0600"
            )
        with os.fdopen(descriptor, "rb") as stream:
            line_number = 0
            while line := stream.readline(MAX_LINE_BYTES + 1):
                line_number += 1
                input_digest.update(line)
                if len(line) > MAX_LINE_BYTES:
                    raise CalibrationError(
                        f"journal line {line_number} exceeds the size limit"
                    )
                if not line.endswith(b"\n") or not line.strip():
                    raise CalibrationError(
                        f"journal line {line_number} is incomplete or blank"
                    )
                record, sample = _validate_record(
                    _decode_record(line, f"journal line {line_number}"),
                    plan,
                    expected_sequence=line_number,
                    expected_previous=previous,
                )
                timestamp = int(record["monotonic_ns"])
                if timestamps and timestamp <= timestamps[-1]:
                    raise CalibrationError(
                        "journal monotonic timestamps must increase strictly"
                    )
                samples.append(sample)
                timestamps.append(timestamp)
                previous = str(record["sha256"])
    except OSError as exc:
        raise CalibrationError(f"cannot read calibration journal: {exc}") from exc
    finally:
        os.close(directory_descriptor)

    if require_complete and len(samples) != plan.maximum_actions:
        raise CalibrationError(
            f"journal is incomplete: expected {plan.maximum_actions}, "
            f"observed {len(samples)}"
        )
    if len(samples) > plan.maximum_actions:
        raise CalibrationError("journal exceeds the maximum action count")
    if (
        timestamps
        and (timestamps[-1] - timestamps[0]) / 1e9 > plan.maximum_runtime_seconds
    ):
        raise CalibrationError("journal exceeds the maximum run time")

    confirmed = sum(sample.registered for sample in samples)
    registration_lower = _wilson_lower(confirmed, len(samples)) if samples else 0.0
    if require_complete:
        if confirmed < plan.minimum_confirmed_actions:
            raise CalibrationError("journal has too few confirmed actions")
        if registration_lower < plan.minimum_registration_rate:
            raise CalibrationError(
                "Wilson registration-rate lower bound misses the frozen gate"
            )
        confirmed_cells = {
            (sample.client_x, sample.client_y, sample.button)
            for sample in samples
            if sample.registered
        }
        required_cells = {
            (x, y, button)
            for x in plan.x_coordinates
            for y in plan.y_coordinates
            for button in ("weak", "strong")
        }
        if confirmed_cells != required_cells:
            raise CalibrationError(
                "confirmed actions do not cover the complete 2-D/button sweep"
            )

    return VerifiedCalibrationJournal(
        tuple(samples),
        tuple(timestamps),
        input_digest.hexdigest(),
        previous,
        registration_lower,
    )


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("journal write made no progress")
        view = view[written:]


def _open_private_directory(path: Path) -> int:
    if not path.is_absolute():
        raise JournalPublicationError("private journal directory must be absolute")
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise JournalPublicationError(
            f"cannot inspect private journal directory: {exc}"
        ) from exc
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        raise JournalPublicationError(
            "journal directory must be an owned non-symlink directory with mode 0700"
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise JournalPublicationError(
            f"cannot open private journal directory: {exc}"
        ) from exc
    opened = os.fstat(descriptor)
    if (
        (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
        or not stat.S_ISDIR(opened.st_mode)
        or opened.st_uid != os.getuid()
        or stat.S_IMODE(opened.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise JournalPublicationError("journal directory changed during validation")
    return descriptor


class CalibrationJournalWriter:
    """Create and append one private no-replace journal."""

    def __init__(
        self,
        directory: str | os.PathLike[str],
        filename: str,
        plan: CalibrationPlan,
    ) -> None:
        if not isinstance(plan, CalibrationPlan):
            raise TypeError("plan must be a CalibrationPlan")
        if not isinstance(filename, str) or SAFE_FILENAME.fullmatch(filename) is None:
            raise JournalPublicationError("journal filename is unsafe")
        self.plan = plan
        self.path = Path(directory) / filename
        self._directory_descriptor = _open_private_directory(Path(directory))
        flags = (
            os.O_WRONLY
            | os.O_APPEND
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | os.O_CLOEXEC
        )
        descriptor: int | None = None
        try:
            descriptor = os.open(
                filename,
                flags,
                0o600,
                dir_fd=self._directory_descriptor,
            )
            os.fchmod(descriptor, 0o600)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_uid != os.getuid()
                or stat.S_IMODE(opened.st_mode) != 0o600
            ):
                raise JournalPublicationError(
                    "journal file ownership or mode is unsafe"
                )
            os.fsync(self._directory_descriptor)
        except FileExistsError as exc:
            os.close(self._directory_descriptor)
            raise JournalPublicationError(
                f"journal destination already exists: {self.path}"
            ) from exc
        except Exception:
            if descriptor is not None:
                os.close(descriptor)
            os.close(self._directory_descriptor)
            raise
        assert descriptor is not None
        self._descriptor = descriptor
        self._sequence = 0
        self._previous = ZERO_SHA256
        self._first_timestamp: int | None = None
        self._last_timestamp: int | None = None
        self._closed = False

    def __enter__(self) -> Self:
        return self

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def next_cell(self) -> SweepCell | None:
        if self._sequence >= len(self.plan.expected_cells):
            return None
        return self.plan.expected_cells[self._sequence]

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        self.close()
        return False

    def append(
        self,
        sample: CalibrationSample | Mapping[str, Any],
        *,
        monotonic_ns: int,
    ) -> dict[str, object]:
        if self._closed:
            raise CalibrationError("journal writer is closed")
        if self._sequence >= self.plan.maximum_actions:
            raise CalibrationError("journal reached its maximum action count")
        if (
            isinstance(monotonic_ns, bool)
            or not isinstance(monotonic_ns, int)
            or monotonic_ns < 0
        ):
            raise CalibrationError("journal monotonic_ns must be nonnegative")
        if self._last_timestamp is not None and monotonic_ns <= self._last_timestamp:
            raise CalibrationError("journal timestamps must increase strictly")
        first = monotonic_ns if self._first_timestamp is None else self._first_timestamp
        if (monotonic_ns - first) / 1e9 > self.plan.maximum_runtime_seconds:
            raise CalibrationError("journal exceeded its maximum run time")
        record = seal_calibration_record(
            self.plan,
            sample,
            sequence=self._sequence + 1,
            monotonic_ns=monotonic_ns,
            previous_sha256=self._previous,
        )
        _validate_record(
            record,
            self.plan,
            expected_sequence=self._sequence + 1,
            expected_previous=self._previous,
        )
        _write_all(self._descriptor, encode_calibration_record(record))
        os.fsync(self._descriptor)
        self._sequence += 1
        self._previous = str(record["sha256"])
        self._first_timestamp = first
        self._last_timestamp = monotonic_ns
        return record

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        error: OSError | None = None
        try:
            os.fsync(self._descriptor)
        except OSError as exc:
            error = exc
        finally:
            os.close(self._descriptor)
        try:
            os.fsync(self._directory_descriptor)
        except OSError as exc:
            if error is None:
                error = exc
        finally:
            os.close(self._directory_descriptor)
        if error is not None:
            raise error

    def finalize(self) -> VerifiedCalibrationJournal:
        if self._sequence != self.plan.maximum_actions:
            raise CalibrationError(
                f"cannot finalize incomplete journal: {self._sequence}/"
                f"{self.plan.maximum_actions}"
            )
        self.close()
        return verify_calibration_journal(self.path, self.plan)


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _statistics(values: Sequence[float], direction: str) -> dict[str, float]:
    if not values:
        raise CalibrationError("cannot summarize an empty calibration metric")
    return {
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "worst": min(values) if direction == "min" else max(values),
    }


def _circular_block_sample(
    values: Sequence[float], block_length: int, rng: random.Random
) -> list[float]:
    result: list[float] = []
    while len(result) < len(values):
        start = rng.randrange(len(values))
        result.extend(
            values[(start + offset) % len(values)] for offset in range(block_length)
        )
    return result[: len(values)]


def _bootstrap_uncertainty(
    field: str,
    grouped_values: Sequence[Sequence[float]],
    point: Mapping[str, float],
    plan: CalibrationPlan,
) -> float:
    seed = hashlib.sha256(
        bytes.fromhex(plan.order_seed_sha256) + field.encode("ascii")
    ).digest()
    rng = random.Random(int.from_bytes(seed, "big"))
    replicated = {name: [] for name in ("p50", "p95", "p99", "worst")}
    direction = METRIC_SPECS[field][2]
    run_count = len(grouped_values)
    for _ in range(plan.bootstrap_replicates):
        sample: list[float] = []
        for _ in range(run_count):
            run = grouped_values[rng.randrange(run_count)]
            sample.extend(_circular_block_sample(run, plan.block_length, rng))
        summary = _statistics(sample, direction)
        for name, values in replicated.items():
            values.append(summary[name])
    uncertainty = 0.0
    for name, values in replicated.items():
        lower = _percentile(values, 0.025)
        upper = _percentile(values, 0.975)
        uncertainty = max(
            uncertainty,
            abs(point[name] - lower),
            abs(upper - point[name]),
        )
    # Treat instrument quantization as a systematic floor.  Adjacent frames do
    # not make the underlying clock/pixel resolution disappear.
    resolution_floor = plan.instrument_resolution[field] / 2
    return max(uncertainty, resolution_floor)


def _metric_artifact(
    field: str,
    samples: Sequence[CalibrationSample],
    plan: CalibrationPlan,
) -> dict[str, object]:
    confirmed = [sample for sample in samples if sample.registered]
    grouped: list[list[float]] = []
    for experiment_id in plan.experiment_ids:
        values = [
            float(sample.measurements[field])
            for sample in confirmed
            if sample.experiment_id == experiment_id and sample.measurements is not None
        ]
        if not values:
            raise CalibrationError(
                f"experiment {experiment_id!r} has no confirmed {field} samples"
            )
        grouped.append(values)
    values = [value for group in grouped for value in group]
    _, unit, direction = METRIC_SPECS[field]
    summary = _statistics(values, direction)
    return {
        "unit": unit,
        "direction": direction,
        "sample_count": len(values),
        **summary,
        "uncertainty": _bootstrap_uncertainty(field, grouped, summary, plan),
    }


def _verify_soak_binding(
    plan: CalibrationPlan,
    soak_report: Mapping[str, Any],
    event_path: Path,
    threshold_path: Path,
) -> tuple[str, str]:
    try:
        report_sha256 = verify_report(soak_report, event_path, threshold_path)
        thresholds, threshold_sha256 = load_thresholds(threshold_path)
    except EvidenceError as exc:
        raise CalibrationError(f"invalid bound soak artifacts: {exc}") from exc
    if threshold_sha256 != plan.soak_threshold_config_sha256:
        raise CalibrationError("soak threshold hash disagrees with the plan")
    stream = _mapping(soak_report.get("event_stream"), "soak event_stream")
    event_stream_sha256 = _sha256(stream.get("sha256"), "soak event-stream SHA-256")
    if thresholds["experiment_ids"] != list(plan.experiment_ids):
        raise CalibrationError("soak experiment IDs disagree with the plan")
    if thresholds["minimum_duration_seconds"] < plan.minimum_soak_duration_seconds:
        raise CalibrationError("soak threshold duration is below the plan")
    if (
        thresholds["maximum_measurement_gap_seconds"]
        > plan.maximum_measurement_gap_seconds
    ):
        raise CalibrationError("soak threshold measurement gap exceeds the plan")
    duration = _mapping(soak_report.get("duration"), "soak duration")
    if (
        soak_report.get("status") != "pass"
        or _finite(duration.get("seconds"), "soak duration", nonnegative=True)
        < plan.minimum_soak_duration_seconds
    ):
        raise CalibrationError("soak report did not pass the frozen duration")
    observed = _mapping(soak_report.get("experiments"), "soak experiments").get(
        "observed"
    )
    if observed != list(plan.experiment_ids):
        raise CalibrationError("soak report experiment order disagrees with the plan")
    report_provenance = _mapping(
        _mapping(soak_report.get("provenance"), "soak provenance").get("observed"),
        "soak observed provenance",
    )
    for key in ("game_executable_sha256", "measurement_tool_sha256"):
        if report_provenance.get(key) != plan.provenance[key]:
            raise CalibrationError(f"soak report does not bind {key}")
    return report_sha256, event_stream_sha256


def build_deployment_evidence(
    plan: CalibrationPlan,
    journal_path: str | os.PathLike[str],
    soak_report: Mapping[str, Any],
    soak_event_path: str | os.PathLike[str],
    soak_threshold_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Rebuild a contract-compatible measurement bundle from raw artifacts."""

    if not isinstance(plan, CalibrationPlan):
        raise TypeError("plan must be a CalibrationPlan")
    report_sha256, event_stream_sha256 = _verify_soak_binding(
        plan,
        soak_report,
        Path(soak_event_path),
        Path(soak_threshold_path),
    )
    journal = verify_calibration_journal(journal_path, plan)
    confirmed_count = sum(sample.registered for sample in journal.samples)
    metrics = {
        field: _metric_artifact(field, journal.samples, plan) for field in METRIC_FIELDS
    }
    artifacts = list(
        dict.fromkeys(
            (
                event_stream_sha256,
                plan.soak_threshold_config_sha256,
                report_sha256,
                plan.sha256,
                journal.input_sha256,
                journal.chain_head_sha256,
            )
        )
    )
    common: dict[str, object] = {
        "sample_count": confirmed_count,
        "uncertainty_method": (
            "95 percent run-cluster circular moving-block bootstrap"
        ),
        "provenance_category": "observed",
        "experiment_ids": list(plan.experiment_ids),
        "artifact_sha256": artifacts,
    }

    sections: dict[str, dict[str, object]] = {}
    for section, fields in SECTION_FIELDS.items():
        sections[section] = {
            **common,
            "measurements": {field: metrics[field] for field in fields},
        }
    sections["click_macro"].update(
        {
            "input_provider_capability": SAFE_PROVIDER_CAPABILITY,
            "weak_button": "left",
            "strong_button": "right",
        }
    )
    sections["cursor"].update(
        {
            "travel_model": "abstract_coordinate_fixed_rate",
            "quantization": plan.cursor_quantization,
            "cursor_retention": plan.cursor_retention,
            "path_logging": True,
        }
    )
    sections["coordinate_calibration"].update(
        {
            "click_sweep_dimensions": 2,
            "continuous_drift_check": True,
        }
    )
    return {
        "schema_version": DEPLOYMENT_EVIDENCE_SCHEMA,
        "contract_version": "deployment-v1",
        "status": "measured",
        "soak_experiment_ids": list(plan.experiment_ids),
        "soak_event_stream_sha256": event_stream_sha256,
        "soak_threshold_config_sha256": plan.soak_threshold_config_sha256,
        "soak_report_sha256": report_sha256,
        "provenance": dict(plan.provenance),
        "sections": sections,
    }


__all__ = [
    "SAFE_PROVIDER_CAPABILITY",
    "CalibrationError",
    "CalibrationJournalWriter",
    "CalibrationPlan",
    "CalibrationSample",
    "JournalPublicationError",
    "VerifiedCalibrationJournal",
    "build_deployment_evidence",
    "encode_calibration_record",
    "load_calibration_plan",
    "seal_calibration_record",
    "verify_calibration_journal",
]
