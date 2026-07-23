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
)
from irisu_rl.original_game.calibration_runner import (
    ARM_SCHEMA,
    CalibrationArm,
    CalibrationOutcome,
    R4BCalibrationRunner,
    current_boot_id_sha256,
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


class Clock:
    def __init__(self, value: int = 1_000_000_000) -> None:
        self.value = value

    def __call__(self) -> int:
        self.value += 1_000_000
        return self.value


class Harness:
    def __init__(self, plan: CalibrationPlan, runtime: str, nonce: str) -> None:
        self.provider = SimpleNamespace(
            broker_implementation_sha256=plan.provider_build_sha256
        )
        self.target_descriptor = TargetRuntimeDescriptor(
            WindowIdentity("0xabc", "capture-1"),
            123,
            456,
            nonce,
            plan.provenance["game_executable_sha256"],
            runtime,
            CANONICAL_WINE_SHA256,
        )
        self.actions = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def capture(self):
        return SimpleNamespace(usable=True)

    def fire(self, kind, x, y):
        self.actions += 1
        return SimpleNamespace(
            status=ExecutionStatus.EXECUTED,
            proposed=SimpleNamespace(sequence=self.actions),
        )

    def record_action_effect(self, *args, **kwargs):
        return None


class Observer:
    def __init__(self) -> None:
        self.sequence = 1

    def observe(self, harness, cell, action_sequence):
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
        self.plan = CalibrationPlan.from_mapping(_plan_mapping(report))
        self.clock = Clock()
        self.runtime = "e" * 64
        self.nonce = "f" * 64

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def arm(self, *, expires_offset: int = 200_000_000_000) -> CalibrationArm:
        now = self.clock()
        per_experiment = self.plan.maximum_actions // len(self.plan.experiment_ids)
        return CalibrationArm.from_mapping(
            {
                "schema": ARM_SCHEMA,
                "plan_sha256": self.plan.sha256,
                "experiment_id": self.plan.experiment_ids[0],
                "provider_build_sha256": self.plan.provider_build_sha256,
                "runtime_identity_sha256": self.runtime,
                "launch_nonce_sha256": self.nonce,
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

    def test_expired_or_wrong_runtime_never_fires(self) -> None:
        arm = self.arm(expires_offset=1)
        harness = Harness(self.plan, self.runtime, self.nonce)
        with CalibrationJournalWriter(
            self.private, "expired.jsonl", self.plan
        ) as writer:
            with self.assertRaisesRegex(CalibrationError, "expired"):
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
        with CalibrationJournalWriter(self.private, "wrong.jsonl", self.plan) as writer:
            with self.assertRaisesRegex(CalibrationError, "runtime"):
                R4BCalibrationRunner(
                    self.plan,
                    self.arm(),
                    writer,
                    lambda: harness,
                    Observer(),
                    clock_ns=self.clock,
                ).run()
        self.assertEqual(harness.actions, 0)


if __name__ == "__main__":
    unittest.main()
