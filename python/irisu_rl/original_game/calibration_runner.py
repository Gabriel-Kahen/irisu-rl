"""Explicitly armed executor for the frozen R4b calibration lattice."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .calibration import (
    SAFE_PROVIDER_CAPABILITY,
    CalibrationError,
    CalibrationJournalWriter,
    CalibrationPlan,
    CalibrationSample,
    SweepCell,
)
from .evidence import load_json_document
from .harness import (
    EffectStatus,
    ExecutionStatus,
    OriginalGameHarness,
    ShotKind,
)
from .runtime import CANONICAL_WINE_SHA256

ARM_SCHEMA = "r4b-calibration-arm-v1"
MAX_ARM_SECONDS = 300


def current_boot_id_sha256() -> str:
    try:
        boot_id = (
            Path("/proc/sys/kernel/random/boot_id").read_text(encoding="ascii").strip()
        )
    except OSError as exc:
        raise CalibrationError("cannot bind calibration arm to this boot") from exc
    return hashlib.sha256(boot_id.encode("ascii")).hexdigest()


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value == "0" * 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CalibrationError(f"{label} must be a nonzero lowercase SHA-256")
    return value


def _positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CalibrationError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class CalibrationArm:
    plan_sha256: str
    experiment_id: str
    provider_build_sha256: str
    runtime_identity_sha256: str
    launch_nonce_sha256: str
    boot_id_sha256: str
    issued_monotonic_ns: int
    expires_monotonic_ns: int
    maximum_actions: int

    @classmethod
    def from_mapping(
        cls,
        value: object,
        plan: CalibrationPlan,
        *,
        now_ns: int,
    ) -> CalibrationArm:
        if not isinstance(value, Mapping) or any(
            not isinstance(key, str) for key in value
        ):
            raise CalibrationError("calibration arm must be an object")
        expected = {
            "schema",
            "plan_sha256",
            "experiment_id",
            "provider_build_sha256",
            "runtime_identity_sha256",
            "launch_nonce_sha256",
            "boot_id_sha256",
            "issued_monotonic_ns",
            "expires_monotonic_ns",
            "maximum_actions",
        }
        if set(value) != expected:
            raise CalibrationError("calibration arm fields disagree")
        experiment_id = value["experiment_id"]
        if experiment_id not in plan.experiment_ids:
            raise CalibrationError("calibration arm names an undeclared experiment")
        issued = _positive_int(value["issued_monotonic_ns"], "arm issued time")
        expires = _positive_int(value["expires_monotonic_ns"], "arm expiry")
        if (
            value["schema"] != ARM_SCHEMA
            or value["plan_sha256"] != plan.sha256
            or value["provider_build_sha256"] != plan.provider_build_sha256
            or value["boot_id_sha256"] != current_boot_id_sha256()
        ):
            raise CalibrationError("calibration arm identity binding mismatch")
        if not issued <= now_ns < expires or expires - issued > MAX_ARM_SECONDS * 10**9:
            raise CalibrationError("calibration arm is stale or exceeds five minutes")
        per_experiment = plan.maximum_actions // len(plan.experiment_ids)
        maximum_actions = _positive_int(value["maximum_actions"], "arm maximum actions")
        if maximum_actions != per_experiment:
            raise CalibrationError("calibration arm action cap disagrees with the plan")
        return cls(
            plan.sha256,
            str(experiment_id),
            plan.provider_build_sha256,
            _sha256(value["runtime_identity_sha256"], "arm runtime identity"),
            _sha256(value["launch_nonce_sha256"], "arm launch nonce"),
            str(value["boot_id_sha256"]),
            issued,
            expires,
            maximum_actions,
        )


def load_calibration_arm(
    path: Path,
    plan: CalibrationPlan,
    *,
    now_ns: int | None = None,
) -> CalibrationArm:
    return CalibrationArm.from_mapping(
        load_json_document(path, "R4b calibration arm"),
        plan,
        now_ns=time.monotonic_ns() if now_ns is None else now_ns,
    )


@dataclass(frozen=True, slots=True)
class CalibrationOutcome:
    status: EffectStatus
    frame_sequence: int
    measurements: Mapping[str, float] | None

    def __post_init__(self) -> None:
        if self.status not in {
            EffectStatus.CONFIRMED,
            EffectStatus.MISSED,
            EffectStatus.AMBIGUOUS,
        }:
            raise CalibrationError("calibration outcome must be terminal")
        if type(self.frame_sequence) is not int or self.frame_sequence <= 0:
            raise CalibrationError("outcome frame sequence must be positive")
        if self.status is EffectStatus.CONFIRMED:
            if not isinstance(self.measurements, Mapping):
                raise CalibrationError("confirmed outcome requires measurements")
        elif self.measurements is not None:
            raise CalibrationError("failed outcome must not fabricate measurements")


class CalibrationObserver(Protocol):
    def observe(
        self,
        harness: OriginalGameHarness,
        cell: SweepCell,
        action_sequence: int,
    ) -> CalibrationOutcome: ...


class R4BCalibrationRunner:
    """Execute plan cells only; arbitrary policy actions have no entry point."""

    def __init__(
        self,
        plan: CalibrationPlan,
        arm: CalibrationArm,
        writer: CalibrationJournalWriter,
        harness_factory: Callable[[], OriginalGameHarness],
        observer: CalibrationObserver,
        *,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if (
            arm.plan_sha256 != plan.sha256
            or arm.provider_build_sha256 != plan.provider_build_sha256
            or writer.plan != plan
        ):
            raise CalibrationError("runner components disagree on the frozen plan")
        self.plan = plan
        self.arm = arm
        self.writer = writer
        self.harness_factory = harness_factory
        self.observer = observer
        self.clock_ns = clock_ns

    def run(self) -> int:
        now = self.clock_ns()
        if not self.arm.issued_monotonic_ns <= now < self.arm.expires_monotonic_ns:
            raise CalibrationError("calibration arm expired before input")
        expected_start = self.writer.next_cell
        if (
            expected_start is None
            or expected_start.experiment_id != self.arm.experiment_id
        ):
            raise CalibrationError("journal is not positioned at the armed experiment")
        completed = 0
        with self.harness_factory() as harness:
            target = harness.target_descriptor
            if (
                target is None
                or target.runtime_sha256 != self.arm.runtime_identity_sha256
                or target.launch_nonce_sha256 != self.arm.launch_nonce_sha256
                or target.executable_sha256
                != self.plan.provenance["game_executable_sha256"]
                or target.wine_executable_sha256 != CANONICAL_WINE_SHA256
                or getattr(harness.provider, "broker_implementation_sha256", None)
                != self.plan.provider_build_sha256
            ):
                raise CalibrationError("armed runtime differs from the claimed process")
            while (
                self.writer.next_cell is not None
                and self.writer.next_cell.experiment_id == self.arm.experiment_id
            ):
                if completed >= self.arm.maximum_actions:
                    raise CalibrationError(
                        "armed action cap reached before experiment end"
                    )
                if self.clock_ns() >= self.arm.expires_monotonic_ns:
                    raise CalibrationError("calibration arm expired during the sweep")
                cell = self.writer.next_cell
                assert cell is not None
                before = harness.capture()
                if not before.usable:
                    raise CalibrationError("pre-action gameplay frame is unusable")
                kind = ShotKind.WEAK if cell.button == "weak" else ShotKind.STRONG
                action = harness.fire(kind, cell.client_x, cell.client_y)
                if action.status is not ExecutionStatus.EXECUTED:
                    raise CalibrationError(
                        "planned calibration action was not executed"
                    )
                outcome = self.observer.observe(harness, cell, action.proposed.sequence)
                if self.clock_ns() >= self.arm.expires_monotonic_ns:
                    raise CalibrationError(
                        "calibration arm expired before outcome journaling"
                    )
                harness.record_action_effect(
                    action.proposed.sequence,
                    outcome.status,
                    frame_sequence=outcome.frame_sequence,
                )
                sample = CalibrationSample.from_mapping(
                    {
                        **cell.manifest(),
                        "provider_capability": SAFE_PROVIDER_CAPABILITY,
                        "provider_build_sha256": self.plan.provider_build_sha256,
                        "registered": outcome.status is EffectStatus.CONFIRMED,
                        "measurements": (
                            dict(outcome.measurements)
                            if outcome.measurements is not None
                            else None
                        ),
                    }
                )
                self.writer.append(sample, monotonic_ns=self.clock_ns())
                completed += 1
        if completed != self.arm.maximum_actions:
            raise CalibrationError("armed experiment ended before every planned cell")
        return completed
