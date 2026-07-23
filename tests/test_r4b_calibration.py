from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

from irisu_rl.original_game.calibration import (
    CLIENT_PIXEL_QUANTIZATION,
    METRIC_FIELDS,
    SAFE_PROVIDER_CAPABILITY,
    CalibrationError,
    CalibrationJournalWriter,
    CalibrationPlan,
    CalibrationSample,
    JournalPublicationError,
    build_deployment_evidence,
    encode_calibration_record,
    seal_calibration_record,
    verify_calibration_journal,
)
from irisu_rl.original_game.contracts import (
    finalize_deployment_contract,
    load_deployment_contract,
)
from irisu_rl.original_game.evidence import (
    EVENT_SCHEMA,
    METRICS,
    THRESHOLD_SCHEMA,
    build_report,
    canonical_json_bytes,
    encode_event,
    seal_event,
)

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_IDS = ("r4b-process-001", "r4b-process-002", "r4b-process-003")
PROVENANCE = {
    "game_executable_sha256": hashlib.sha256(b"game").hexdigest(),
    "box2d_sha256": hashlib.sha256(b"box2d").hexdigest(),
    "dxlib_sha256": hashlib.sha256(b"dxlib").hexdigest(),
    "game_config_sha256": hashlib.sha256(b"config").hexdigest(),
    "measurement_tool_sha256": hashlib.sha256(b"measurement-tool").hexdigest(),
    "runtime": "Wine controlled runtime",
    "hardware_id": "opaque-test-machine",
}


def _thresholds() -> dict[str, object]:
    specs: dict[str, object] = {}
    for name in METRICS:
        direction = "min" if name in {"capture_fps", "action_confirmations"} else "max"
        if name == "capture_fps":
            bound = {"worst": {"min": 1.0}}
        elif name == "action_confirmations":
            bound = {"total": {"min": 1.0}}
        elif name in {"button_release_failures", "cross_window_misroutes"}:
            bound = {"total": {"max": 0.0}}
        else:
            bound = {"worst": {"max": 1_000_000.0}}
        specs[name] = {
            "direction": direction,
            "minimum_samples": 3,
            **bound,
        }
    return {
        "schema": THRESHOLD_SCHEMA,
        "experiment_ids": list(EXPERIMENT_IDS),
        "required_provenance": {
            "game_executable_sha256": PROVENANCE["game_executable_sha256"],
            "measurement_tool_sha256": PROVENANCE["measurement_tool_sha256"],
        },
        "minimum_duration_seconds": 200.0,
        "maximum_measurement_gap_seconds": 100.0,
        "metrics": specs,
    }


def _soak_artifacts(root: Path) -> tuple[dict[str, object], Path, Path]:
    threshold = _thresholds()
    threshold_path = root / "thresholds.json"
    threshold_path.write_text(json.dumps(threshold, sort_keys=True), encoding="utf-8")
    threshold_sha256 = hashlib.sha256(canonical_json_bytes(threshold)).hexdigest()
    previous = "0" * 64
    records = []
    sequence = 0
    for process_index, experiment_id in enumerate(EXPERIMENT_IDS):
        base_seconds = process_index * 201
        for offset_seconds in (0, 100, 200):
            sequence += 1
            values = {name: 0.0 for name in METRICS}
            values.update(
                {
                    "capture_fps": 60.0,
                    "capture_jitter_seconds": 0.001,
                    "ring_age_seconds": 0.002,
                    "request_ack_seconds": 0.003,
                    "poll_effect_interval_seconds": 0.02,
                    "effect_visible_seconds": 0.01,
                    "total_latency_seconds": 0.04,
                    "action_confirmations": 1.0,
                    "crop_drift_pixels": 0.1,
                    "resource_growth_bytes": 1024.0,
                }
            )
            record = seal_event(
                {
                    "schema": EVENT_SCHEMA,
                    "sequence": sequence,
                    "monotonic_ns": (base_seconds + offset_seconds) * 1_000_000_000,
                    "experiment_id": experiment_id,
                    "measurements": values,
                    "provenance": threshold["required_provenance"],
                    "threshold_sha256": threshold_sha256,
                },
                previous,
            )
            records.append(record)
            previous = record["sha256"]
    event_path = root / "events.jsonl"
    event_path.write_bytes(b"".join(encode_event(record) for record in records))
    return build_report(event_path, threshold_path), event_path, threshold_path


def _plan_mapping(report: dict[str, object]) -> dict[str, object]:
    x_coordinates = [32.0 + 64.0 * index for index in range(9)]
    y_coordinates = [30.0 + 60.0 * index for index in range(7)]
    return {
        "schema": "r4b-calibration-plan-v1",
        "contract_version": "deployment-v1",
        "experiment_ids": list(EXPERIMENT_IDS),
        "provider": {
            "required_capability": SAFE_PROVIDER_CAPABILITY,
            "provider_build_sha256": hashlib.sha256(b"safe-broker").hexdigest(),
        },
        "sweep": {
            "client_width": 640.0,
            "client_height": 480.0,
            "x_coordinates": x_coordinates,
            "y_coordinates": y_coordinates,
            "buttons": ["weak", "strong"],
            "repetitions": 2,
            "order_algorithm": "sha256-v1",
            "order_seed_sha256": hashlib.sha256(b"sweep-order").hexdigest(),
        },
        "cursor_protocol": {
            "travel_model": "abstract_coordinate_fixed_rate",
            "quantization": CLIENT_PIXEL_QUANTIZATION,
            "cursor_retention": "retained",
            "path_logging": True,
        },
        "limits": {
            "maximum_actions": len(EXPERIMENT_IDS)
            * len(x_coordinates)
            * len(y_coordinates)
            * 2
            * 2,
            "maximum_runtime_seconds": 1000.0,
        },
        "soak": {
            "episode_envelope_ticks": 8192,
            "nominal_gameplay_hz": 50.0,
            "minimum_duration_seconds": 200.0,
            "maximum_measurement_gap_seconds": 100.0,
            "threshold_config_sha256": report["threshold_config_sha256"],
        },
        "instrument_resolution": {field: 0.000001 for field in METRIC_FIELDS},
        "uncertainty": {
            "method": "moving_block_bootstrap_v1",
            "confidence_level": 0.95,
            "bootstrap_replicates": 200,
            "block_length": 4,
        },
        "acceptance": {
            "minimum_confirmed_actions": 512,
            "minimum_registration_rate": 0.99,
            "registration_interval_method": "wilson-score-v1",
        },
        "provenance": PROVENANCE,
    }


def _measurements(index: int) -> dict[str, float]:
    injection = 0.010 + (index % 5) * 0.000001
    visible = 0.020 + (index % 7) * 0.000001
    return {
        "gameplay_period_seconds": 0.020 + (index % 5) * 0.000001,
        "scheduler_error_seconds": (index % 7) * 0.000001,
        "press_duration_seconds": 0.010 + (index % 3) * 0.000001,
        "release_duration_seconds": 0.005 + (index % 4) * 0.000001,
        "maximum_clicks_per_second": 5.0 + (index % 5) * 0.001,
        "frame_rate_hz": 59.0 + (index % 7) * 0.001,
        "request_to_completion_seconds": 0.003 + (index % 3) * 0.000001,
        "stale_after_seconds": 0.100 + (index % 3) * 0.000001,
        "injection_to_poll_seconds": injection,
        "effect_to_visible_seconds": visible,
        "request_to_visible_seconds": injection + visible + 0.010,
        "residual_client_pixels": 0.100 + (index % 11) * 0.001,
        "fixed_action_rate_hz": 5.0 + (index % 3) * 0.001,
    }


def _sample(plan: CalibrationPlan, index: int) -> CalibrationSample:
    cell = plan.expected_cells[index]
    return CalibrationSample.from_mapping(
        {
            **cell.manifest(),
            "provider_capability": SAFE_PROVIDER_CAPABILITY,
            "provider_build_sha256": plan.provider_build_sha256,
            "registered": True,
            "measurements": _measurements(index),
        }
    )


class R4BCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.private = self.root / "private"
        self.private.mkdir(mode=0o700)
        self.report, self.events, self.thresholds = _soak_artifacts(self.root)
        self.plan_mapping = _plan_mapping(self.report)
        self.plan = CalibrationPlan.from_mapping(self.plan_mapping)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_complete_journal(self, filename: str = "calibration.jsonl") -> Path:
        writer = CalibrationJournalWriter(self.private, filename, self.plan)
        for index in range(self.plan.maximum_actions):
            writer.append(
                _sample(self.plan, index),
                monotonic_ns=1_000_000_000 + index * 1_000_000,
            )
        writer.finalize()
        return self.private / filename

    def test_plan_freezes_safe_provider_randomized_full_sweep_and_limits(self) -> None:
        self.assertEqual(len(self.plan.experiment_ids), 3)
        self.assertEqual(len(self.plan.x_coordinates), 9)
        self.assertEqual(len(self.plan.y_coordinates), 7)
        self.assertEqual(len(self.plan.expected_cells), 756)
        self.assertNotEqual(
            self.plan.expected_cells[0],
            self.plan.expected_cells[1],
        )
        self.assertEqual(
            {cell.button for cell in self.plan.expected_cells},
            {"weak", "strong"},
        )
        self.assertGreater(
            self.plan.minimum_soak_duration_seconds,
            self.plan.episode_envelope_ticks / self.plan.nominal_gameplay_hz,
        )

    def test_plan_rejects_missing_2d_process_button_capability_and_resolution(
        self,
    ) -> None:
        cases = []
        value = deepcopy(self.plan_mapping)
        value["experiment_ids"] = value["experiment_ids"][:2]
        cases.append(value)
        value = deepcopy(self.plan_mapping)
        value["sweep"]["x_coordinates"] = value["sweep"]["x_coordinates"][:8]
        cases.append(value)
        value = deepcopy(self.plan_mapping)
        value["sweep"]["buttons"] = ["weak"]
        cases.append(value)
        value = deepcopy(self.plan_mapping)
        value["provider"]["required_capability"] = "atomic_click"
        cases.append(value)
        value = deepcopy(self.plan_mapping)
        del value["instrument_resolution"]["frame_rate_hz"]
        cases.append(value)
        value = deepcopy(self.plan_mapping)
        value["instrument_resolution"]["frame_rate_hz"] = 0.0
        cases.append(value)
        value = deepcopy(self.plan_mapping)
        value["soak"]["minimum_duration_seconds"] = 163.84
        cases.append(value)
        for candidate in cases:
            with self.subTest(candidate=candidate), self.assertRaises(CalibrationError):
                CalibrationPlan.from_mapping(candidate)

    def test_private_writer_is_no_replace_durable_and_handles_short_writes(
        self,
    ) -> None:
        real_write = os.write

        def partial_write(descriptor: int, payload: object) -> int:
            return real_write(descriptor, bytes(payload)[:7])

        writer = CalibrationJournalWriter(self.private, "partial.jsonl", self.plan)
        with mock.patch(
            "irisu_rl.original_game.calibration.os.write",
            side_effect=partial_write,
        ):
            writer.append(_sample(self.plan, 0), monotonic_ns=1)
        writer.close()
        verified = verify_calibration_journal(
            self.private / "partial.jsonl",
            self.plan,
            require_complete=False,
        )
        self.assertEqual(len(verified.samples), 1)
        mode = stat.S_IMODE((self.private / "partial.jsonl").stat().st_mode)
        self.assertEqual(mode, 0o600)
        with self.assertRaises(JournalPublicationError):
            CalibrationJournalWriter(self.private, "partial.jsonl", self.plan)

    def test_private_writer_rejects_unsafe_directory_and_symlink(self) -> None:
        unsafe = self.root / "unsafe"
        unsafe.mkdir(mode=0o755)
        with self.assertRaises(JournalPublicationError):
            CalibrationJournalWriter(unsafe, "journal.jsonl", self.plan)
        link = self.root / "private-link"
        link.symlink_to(self.private, target_is_directory=True)
        with self.assertRaises(JournalPublicationError):
            CalibrationJournalWriter(link, "journal.jsonl", self.plan)

    def test_journal_rejects_wrong_order_chain_tampering_and_unsafe_payload(
        self,
    ) -> None:
        wrong = _sample(self.plan, 1)
        with (
            self.assertRaisesRegex(CalibrationError, "sweep order"),
            CalibrationJournalWriter(self.private, "wrong.jsonl", self.plan) as writer,
        ):
            writer.append(wrong, monotonic_ns=1)

        first = seal_calibration_record(
            self.plan,
            _sample(self.plan, 0),
            sequence=1,
            monotonic_ns=1,
        )
        first["sample"]["measurements"]["frame_rate_hz"] = 999.0
        tampered = self.private / "tampered.jsonl"
        tampered.write_bytes(encode_calibration_record(first))
        tampered.chmod(0o600)
        with self.assertRaisesRegex(CalibrationError, "SHA-256"):
            verify_calibration_journal(tampered, self.plan, require_complete=False)

        unsafe = _sample(self.plan, 0).manifest()
        unsafe["claim_token"] = "do-not-log"
        with self.assertRaisesRegex(CalibrationError, "extra"):
            CalibrationSample.from_mapping(unsafe)

    def test_unregistered_attempt_cannot_fabricate_measurements(self) -> None:
        value = _sample(self.plan, 0).manifest()
        value["registered"] = False
        with self.assertRaisesRegex(CalibrationError, "must not fabricate"):
            CalibrationSample.from_mapping(value)
        value["measurements"] = None
        parsed = CalibrationSample.from_mapping(value)
        self.assertFalse(parsed.registered)

    def test_complete_journal_builds_only_raw_derived_contract_evidence(self) -> None:
        journal = self.write_complete_journal()
        evidence = build_deployment_evidence(
            self.plan,
            journal,
            self.report,
            self.events,
            self.thresholds,
        )
        self.assertEqual(evidence["status"], "measured")
        sections = evidence["sections"]
        for section in sections.values():
            self.assertEqual(section["sample_count"], 756)
            for metric in section["measurements"].values():
                self.assertEqual(metric["sample_count"], 756)
                self.assertGreater(metric["uncertainty"], 0.0)
        fps = sections["capture"]["measurements"]["frame_rate_hz"]
        expected_fps = [
            _measurements(index)["frame_rate_hz"]
            for index in range(self.plan.maximum_actions)
        ]
        self.assertEqual(fps["worst"], min(expected_fps))
        self.assertEqual(
            sections["click_macro"]["input_provider_capability"],
            SAFE_PROVIDER_CAPABILITY,
        )
        serialized = json.dumps(evidence, sort_keys=True)
        self.assertNotIn("do-not-log", serialized)
        self.assertNotIn(str(self.private), serialized)

        base = load_deployment_contract(ROOT / "configs/rl/actions/deployment-v1.toml")
        measured = finalize_deployment_contract(
            base,
            evidence,
            self.report,
            self.events,
            self.thresholds,
        )
        self.assertEqual(measured["measurement_status"], "measured_pending_review")
        self.assertFalse(measured["live_deployment_enabled"])

    def test_r4b_finalizer_cli_rebuilds_raw_evidence_and_stays_blocked(self) -> None:
        journal = self.write_complete_journal("cli.jsonl")
        plan_path = self.root / "plan.json"
        report_path = self.root / "report.json"
        plan_path.write_text(json.dumps(self.plan.manifest()), encoding="utf-8")
        report_path.write_text(json.dumps(self.report), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(ROOT / "tools/finalize-r4b-contract.py"),
                str(ROOT / "configs/rl/actions/deployment-v1.toml"),
                str(plan_path),
                str(journal),
                str(report_path),
                str(self.events),
                str(self.thresholds),
                str(self.private),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "measured_pending_review")
        contract = (self.private / "deployment-v1.measured.toml").read_text(
            encoding="utf-8"
        )
        self.assertIn('measurement_status = "measured_pending_review"', contract)
        self.assertIn("live_deployment_enabled = false", contract)

    def test_builder_rejects_soak_and_plan_tampering(self) -> None:
        journal = self.write_complete_journal("bound.jsonl")
        changed = deepcopy(self.report)
        changed["metrics"]["capture_fps"]["p50"] = 999.0
        with self.assertRaisesRegex(CalibrationError, "invalid bound soak"):
            build_deployment_evidence(
                self.plan,
                journal,
                changed,
                self.events,
                self.thresholds,
            )
        changed_plan = deepcopy(self.plan_mapping)
        changed_plan["soak"]["threshold_config_sha256"] = "f" * 64
        plan = CalibrationPlan.from_mapping(changed_plan)
        with self.assertRaisesRegex(CalibrationError, "threshold hash"):
            build_deployment_evidence(
                plan,
                journal,
                self.report,
                self.events,
                self.thresholds,
            )

    def test_incomplete_low_registration_and_runtime_overrun_do_not_promote(
        self,
    ) -> None:
        writer = CalibrationJournalWriter(self.private, "incomplete.jsonl", self.plan)
        writer.append(_sample(self.plan, 0), monotonic_ns=1)
        with self.assertRaisesRegex(CalibrationError, "incomplete"):
            writer.finalize()
        writer.close()
        with self.assertRaisesRegex(CalibrationError, "incomplete"):
            verify_calibration_journal(self.private / "incomplete.jsonl", self.plan)

        writer = CalibrationJournalWriter(self.private, "runtime.jsonl", self.plan)
        writer.append(_sample(self.plan, 0), monotonic_ns=1)
        with self.assertRaisesRegex(CalibrationError, "maximum run time"):
            writer.append(
                _sample(self.plan, 1),
                monotonic_ns=int((self.plan.maximum_runtime_seconds + 1) * 1e9),
            )
        writer.close()


if __name__ == "__main__":
    unittest.main()
