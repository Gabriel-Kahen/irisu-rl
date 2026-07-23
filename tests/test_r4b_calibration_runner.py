from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_rl.original_game.calibration import (
    CalibrationError,
    CalibrationJournalWriter,
    CalibrationPlan,
    measurement_tool_bundle_sha256,
    verify_calibration_journal,
)
from irisu_rl.original_game.calibration_runner import (
    ARM_SCHEMA,
    CalibrationArm,
    CalibrationOutcome,
    R4BCalibrationRunner,
    current_boot_id_sha256,
    measurement_artifact_bundle_sha256,
    measurement_runner_build_sha256,
)
from irisu_rl.original_game.harness import (
    EffectStatus,
    ExecutionStatus,
    TargetRuntimeDescriptor,
    WindowIdentity,
)
from irisu_rl.original_game.runtime import CANONICAL_WINE_SHA256

from tests.test_r4b_calibration import (
    _measurements,
    _plan_mapping,
    _soak_artifacts,
)

OBSERVER_ARTIFACTS = {"observer-fixture": Path(__file__).resolve()}
OBSERVER_SHA256 = measurement_artifact_bundle_sha256(
    OBSERVER_ARTIFACTS,
    schema="r4b-measurement-observer-build-v1",
)


class Clock:
    def __init__(self, value: int = 1_000_000_000) -> None:
        self.value = value

    def __call__(self) -> int:
        self.value += 1_000_000
        return self.value


class Harness:
    def __init__(
        self,
        plan: CalibrationPlan,
        runtime: str,
        nonce: str,
        *,
        process_id: int = 123,
        process_start_ticks: int = 456,
        failure: str | None = None,
    ) -> None:
        self.provider = SimpleNamespace(
            broker_implementation_sha256=plan.provider_build_sha256
        )
        self.target_descriptor = TargetRuntimeDescriptor(
            WindowIdentity("0xabc", "capture-1"),
            process_id,
            process_start_ticks,
            nonce,
            plan.provenance["game_executable_sha256"],
            runtime,
            CANONICAL_WINE_SHA256,
            plan.provenance["wine_prefix_sha256"],
        )
        self.actions = 0
        self.failure = failure
        self.runtime_verifications = 0
        self.input_environment_verifications = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def capture(self):
        if self.failure == "capture":
            raise RuntimeError("capture failed")
        return SimpleNamespace(usable=True)

    def verify_runtime_unchanged(self):
        if self.failure == "runtime":
            raise RuntimeError("runtime changed")
        if self.failure == "late_runtime" and self.actions:
            raise RuntimeError("runtime changed after action")
        self.runtime_verifications += 1

    def verify_input_environment_unchanged(self):
        if self.failure == "prefix":
            raise RuntimeError("Wine prefix changed")
        self.input_environment_verifications += 1

    def fire(self, kind, x, y):
        self.actions += 1
        if self.failure == "fire":
            raise RuntimeError("fire failed")
        return SimpleNamespace(
            status=ExecutionStatus.EXECUTED,
            proposed=SimpleNamespace(sequence=self.actions),
        )

    def record_action_effect(self, *args, **kwargs):
        return None


class Observer:
    implementation_artifacts = OBSERVER_ARTIFACTS

    def __init__(self) -> None:
        self.sequence = 1

    def observe(self, harness, cell, action_sequence):
        if harness.failure == "observer":
            raise RuntimeError("observer failed")
        self.sequence += 1
        return CalibrationOutcome(
            EffectStatus.CONFIRMED,
            self.sequence,
            _measurements(action_sequence),
        )


class R4BCalibrationRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.private = self.root / "private"
        self.private.mkdir(mode=0o700)
        report, _, _ = _soak_artifacts(self.root)
        mapping = _plan_mapping(report)
        runner_sha256 = measurement_runner_build_sha256()
        mapping["measurement_tools"] = {
            "runner_sha256": runner_sha256,
            "observer_sha256": OBSERVER_SHA256,
        }
        mapping["provenance"]["measurement_tool_sha256"] = (
            measurement_tool_bundle_sha256(runner_sha256, OBSERVER_SHA256)
        )
        self.plan = CalibrationPlan.from_mapping(mapping)
        self.clock = Clock()
        self.runtime = "e" * 64
        self.nonce = "f" * 64

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def arm(
        self,
        *,
        expires_offset: int = 200_000_000_000,
        experiment_index: int = 0,
        nonce: str | None = None,
    ) -> CalibrationArm:
        now = self.clock()
        per_experiment = self.plan.maximum_actions // len(self.plan.experiment_ids)
        return CalibrationArm.from_mapping(
            {
                "schema": ARM_SCHEMA,
                "plan_sha256": self.plan.sha256,
                "experiment_id": self.plan.experiment_ids[experiment_index],
                "provider_build_sha256": self.plan.provider_build_sha256,
                "runtime_identity_sha256": self.runtime,
                "launch_nonce_sha256": self.nonce if nonce is None else nonce,
                "boot_id_sha256": current_boot_id_sha256(),
                "issued_monotonic_ns": now,
                "expires_monotonic_ns": now + expires_offset,
                "maximum_actions": per_experiment,
            },
            self.plan,
            now_ns=now,
        )

    def test_runs_only_the_armed_experiment_in_frozen_order(self) -> None:
        harness = Harness(self.plan, self.runtime, self.nonce)
        with CalibrationJournalWriter(
            self.private, "calibration.jsonl", self.plan
        ) as writer:
            runner = R4BCalibrationRunner(
                self.plan,
                self.arm(),
                writer,
                lambda: harness,
                Observer(),
                clock_ns=self.clock,
            )
            completed = runner.run()
            self.assertEqual(completed, 252)
            self.assertEqual(writer.sequence, 252)
            self.assertEqual(
                writer.next_cell.experiment_id, self.plan.experiment_ids[1]
            )
        self.assertEqual(harness.actions, 252)
        self.assertGreaterEqual(harness.runtime_verifications, 1)
        self.assertGreaterEqual(harness.input_environment_verifications, 1)

    def test_expired_or_wrong_runtime_never_fires(self) -> None:
        arm = self.arm(expires_offset=1)
        harness = Harness(self.plan, self.runtime, self.nonce)
        with (
            CalibrationJournalWriter(
                self.private, "expired.jsonl", self.plan
            ) as writer,
            self.assertRaisesRegex(CalibrationError, "expired"),
        ):
            R4BCalibrationRunner(
                self.plan,
                arm,
                writer,
                lambda: harness,
                Observer(),
                clock_ns=self.clock,
            ).run()
        self.assertEqual(harness.actions, 0)

        harness = Harness(self.plan, "9" * 64, self.nonce)
        with (
            CalibrationJournalWriter(self.private, "wrong.jsonl", self.plan) as writer,
            self.assertRaisesRegex(CalibrationError, "runtime"),
        ):
            R4BCalibrationRunner(
                self.plan,
                self.arm(),
                writer,
                lambda: harness,
                Observer(),
                clock_ns=self.clock,
            ).run()
        self.assertEqual(harness.actions, 0)

    def test_missing_or_failed_runtime_guard_never_fires(self) -> None:
        for failure in ("missing", "runtime", "prefix"):
            with self.subTest(failure=failure):
                harness = Harness(
                    self.plan,
                    self.runtime,
                    self.nonce,
                    failure=None if failure == "missing" else failure,
                )
                if failure == "missing":
                    harness.verify_runtime_unchanged = None
                with (
                    CalibrationJournalWriter(
                        self.private,
                        f"runtime-guard-{failure}.jsonl",
                        self.plan,
                    ) as writer,
                    self.assertRaisesRegex(
                        CalibrationError,
                        "re-attestation",
                    ),
                ):
                    R4BCalibrationRunner(
                        self.plan,
                        self.arm(),
                        writer,
                        lambda harness=harness: harness,
                        Observer(),
                        clock_ns=self.clock,
                    ).run()
                self.assertEqual(harness.actions, 0)

    def test_fire_and_observer_failures_leave_terminal_taint_records(self) -> None:
        for failure, expected_status, error in (
            ("fire", "fire_failed", "fire failed"),
            ("observer", "observer_failed", "observer failed"),
            (
                "late_runtime",
                "runtime_revalidation_failed",
                "runtime re-attestation",
            ),
        ):
            with self.subTest(failure=failure):
                filename = f"{failure}.jsonl"
                harness = Harness(self.plan, self.runtime, self.nonce, failure=failure)
                with (
                    CalibrationJournalWriter(
                        self.private, filename, self.plan
                    ) as writer,
                    self.assertRaisesRegex(Exception, error),
                ):
                    R4BCalibrationRunner(
                        self.plan,
                        self.arm(),
                        writer,
                        lambda harness=harness: harness,
                        Observer(),
                        clock_ns=self.clock,
                    ).run()
                verified = verify_calibration_journal(
                    self.private / filename,
                    self.plan,
                    require_complete=False,
                )
                self.assertTrue(verified.tainted)
                self.assertEqual(verified.terminal_failures, (expected_status,))

    def test_unapproved_runner_or_observer_build_never_fires(self) -> None:
        class WrongObserver(Observer):
            implementation_artifacts = {
                "wrong-observer": ROOT / "tests/test_r4b_calibration.py"
            }

        harness = Harness(self.plan, self.runtime, self.nonce)
        with (
            CalibrationJournalWriter(
                self.private, "wrong-tool.jsonl", self.plan
            ) as writer,
            self.assertRaisesRegex(CalibrationError, "tool bundle"),
        ):
            R4BCalibrationRunner(
                self.plan,
                self.arm(),
                writer,
                lambda: harness,
                WrongObserver(),
                clock_ns=self.clock,
            )
        self.assertEqual(harness.actions, 0)


if __name__ == "__main__":
    unittest.main()
