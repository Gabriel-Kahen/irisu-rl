"""Explicitly armed executor for the frozen R4b calibration lattice."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn, Protocol

from .calibration import (
    SAFE_PROVIDER_CAPABILITY,
    CalibrationError,
    CalibrationJournalWriter,
    CalibrationPlan,
    CalibrationRunAttestation,
    CalibrationSample,
    SweepCell,
    measurement_tool_bundle_sha256,
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
ARTIFACT_ROLE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._/-]{0,255}")


def _artifact_sha256(path: Path, label: str) -> str:
    if not isinstance(path, Path) or not path.is_absolute() or path.is_symlink():
        raise CalibrationError(f"{label} must be an absolute non-symlink file")
    try:
        if path.parent.resolve(strict=True) != path.parent:
            raise CalibrationError(f"{label} ancestry contains a symlink")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        raise CalibrationError(f"cannot open {label}") from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid not in {0, os.getuid()}
            or before.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
            or before.st_nlink != 1
        ):
            raise CalibrationError(f"{label} ownership or mode is unsafe")
        digest = hashlib.sha256()
        offset = 0
        while chunk := os.pread(descriptor, 1 << 20, offset):
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise CalibrationError(f"{label} changed while it was hashed")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def measurement_artifact_bundle_sha256(
    artifacts: Mapping[str, Path],
    *,
    schema: str,
) -> str:
    """Hash observed, role-bound files without persisting their local paths."""

    if (
        not isinstance(artifacts, Mapping)
        or not artifacts
        or not isinstance(schema, str)
        or ARTIFACT_ROLE.fullmatch(schema) is None
    ):
        raise CalibrationError("measurement artifact bundle is invalid")
    manifest: dict[str, str] = {}
    for role, path in sorted(artifacts.items()):
        if (
            not isinstance(role, str)
            or ARTIFACT_ROLE.fullmatch(role) is None
            or role in manifest
        ):
            raise CalibrationError("measurement artifact role is invalid")
        manifest[role] = _artifact_sha256(path, f"measurement artifact {role}")
    payload = {
        "schema": schema,
        "artifact_sha256": manifest,
    }
    return hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def measurement_runner_build_sha256() -> str:
    package = Path(__file__).resolve().parent
    artifacts = {
        path.name: path.resolve()
        for path in package.glob("*.py")
        if path.is_file()
    }
    return measurement_artifact_bundle_sha256(
        artifacts,
        schema="r4b-measurement-runner-build-v1",
    )


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
    implementation_artifacts: Mapping[str, Path]

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
        self.measurement_runner_sha256 = measurement_runner_build_sha256()
        self.observer_sha256 = measurement_artifact_bundle_sha256(
            getattr(observer, "implementation_artifacts", {}),
            schema="r4b-measurement-observer-build-v1",
        )
        bundle = measurement_tool_bundle_sha256(
            self.measurement_runner_sha256, self.observer_sha256
        )
        if (
            self.measurement_runner_sha256 != plan.measurement_runner_sha256
            or self.observer_sha256 != plan.observer_sha256
            or bundle != plan.provenance["measurement_tool_sha256"]
        ):
            raise CalibrationError(
                "observed runner/observer tool bundle differs from the frozen plan"
            )
        self.measurement_tool_sha256 = bundle
        self.clock_ns = clock_ns

    def _record_failed_attempt(self, status: str, cause: Exception) -> NoReturn:
        try:
            self.writer.complete_attempt(
                None,
                monotonic_ns=self.clock_ns(),
                terminal_status=status,
            )
        except Exception as journal_exc:
            raise CalibrationError(
                "calibration attempt failed and its terminal outcome could not "
                "be journaled; the durable intent taints this run"
            ) from journal_exc
        raise cause

    @staticmethod
    def _verify_runtime(harness: OriginalGameHarness) -> None:
        verifier = getattr(harness, "verify_runtime_unchanged", None)
        if not callable(verifier):
            raise CalibrationError(
                "calibration harness lacks mandatory runtime re-attestation"
            )
        try:
            verifier()
        except Exception as exc:
            raise CalibrationError(
                "calibration runtime re-attestation failed"
            ) from exc

    @staticmethod
    def _verify_input_environment(harness: OriginalGameHarness) -> None:
        verifier = getattr(harness, "verify_input_environment_unchanged", None)
        if not callable(verifier):
            raise CalibrationError(
                "calibration harness lacks mandatory Wine-prefix re-attestation"
            )
        try:
            verifier()
        except Exception as exc:
            raise CalibrationError(
                "calibration Wine-prefix re-attestation failed"
            ) from exc

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
            self._verify_runtime(harness)
            self._verify_input_environment(harness)
            target = harness.target_descriptor
            if (
                target is None
                or target.runtime_sha256 != self.arm.runtime_identity_sha256
                or target.launch_nonce_sha256 != self.arm.launch_nonce_sha256
                or target.executable_sha256
                != self.plan.provenance["game_executable_sha256"]
                or target.wine_executable_sha256 != CANONICAL_WINE_SHA256
                or target.wine_prefix_sha256
                != self.plan.provenance["wine_prefix_sha256"]
                or getattr(harness.provider, "broker_implementation_sha256", None)
                != self.plan.provider_build_sha256
            ):
                raise CalibrationError("armed runtime differs from the claimed process")
            run_attestation = CalibrationRunAttestation(
                self.arm.experiment_id,
                target.process_id,
                target.process_start_ticks,
                target.launch_nonce_sha256,
                target.runtime_sha256,
                target.wine_prefix_sha256,
                self.measurement_runner_sha256,
                self.observer_sha256,
                self.measurement_tool_sha256,
            )
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
                self.writer.begin_attempt(run_attestation, monotonic_ns=self.clock_ns())
                try:
                    before = harness.capture()
                    if not before.usable:
                        raise CalibrationError("pre-action gameplay frame is unusable")
                except Exception as exc:  # noqa: BLE001 - persist every failure
                    self._record_failed_attempt("capture_failed", exc)
                kind = ShotKind.WEAK if cell.button == "weak" else ShotKind.STRONG
                try:
                    action = harness.fire(kind, cell.client_x, cell.client_y)
                    if action.status is not ExecutionStatus.EXECUTED:
                        raise CalibrationError(
                            "planned calibration action was not executed"
                        )
                except Exception as exc:  # noqa: BLE001 - persist every failure
                    self._record_failed_attempt("fire_failed", exc)
                try:
                    outcome = self.observer.observe(
                        harness, cell, action.proposed.sequence
                    )
                    if self.clock_ns() >= self.arm.expires_monotonic_ns:
                        raise CalibrationError(
                            "calibration arm expired before outcome journaling"
                        )
                except Exception as exc:  # noqa: BLE001 - persist every failure
                    self._record_failed_attempt("observer_failed", exc)
                try:
                    harness.record_action_effect(
                        action.proposed.sequence,
                        outcome.status,
                        frame_sequence=outcome.frame_sequence,
                    )
                except Exception as exc:  # noqa: BLE001 - persist every failure
                    self._record_failed_attempt("effect_record_failed", exc)
                try:
                    sample = CalibrationSample.from_mapping(
                        {
                            **cell.manifest(),
                            "provider_capability": SAFE_PROVIDER_CAPABILITY,
                            "provider_build_sha256": (self.plan.provider_build_sha256),
                            "registered": (outcome.status is EffectStatus.CONFIRMED),
                            "measurements": (
                                dict(outcome.measurements)
                                if outcome.measurements is not None
                                else None
                            ),
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - persist every failure
                    self._record_failed_attempt("sample_validation_failed", exc)
                try:
                    self._verify_runtime(harness)
                    self._verify_input_environment(harness)
                except Exception as exc:  # noqa: BLE001 - persist every failure
                    self._record_failed_attempt("runtime_revalidation_failed", exc)
                self.writer.complete_attempt(
                    sample,
                    monotonic_ns=self.clock_ns(),
                    terminal_status="completed",
                )
                completed += 1
        if completed != self.arm.maximum_actions:
            raise CalibrationError("armed experiment ended before every planned cell")
        return completed
