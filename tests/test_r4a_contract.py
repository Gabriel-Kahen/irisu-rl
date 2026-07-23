from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from irisu_rl.original_game.contracts import (
    ContractError,
    canonical_json_sha256,
    finalize_deployment_contract,
    load_deployment_contract,
    render_toml,
    validate_deployment_contract,
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
DIGEST = "a" * 64


def soak_artifacts(root: Path) -> tuple[dict[str, object], Path, Path]:
    metrics = {}
    for name in METRICS:
        direction = "min" if name in {"capture_fps", "action_confirmations"} else "max"
        metrics[name] = {
            "direction": direction,
            "minimum_samples": 3,
            "total": {"min": 3.0}
            if name == "action_confirmations"
            else {"max": 0.0}
            if name in {"button_release_failures", "cross_window_misroutes"}
            else {"min": 50.0}
            if name == "capture_fps"
            else {"max": 1_000_000.0},
        }
    config = {
        "schema": THRESHOLD_SCHEMA,
        "experiment_ids": ["controlled-001"],
        "required_provenance": {
            "game_executable_sha256": DIGEST,
            "measurement_tool_sha256": DIGEST,
        },
        "minimum_duration_seconds": 0.002,
        "maximum_measurement_gap_seconds": 0.002,
        "metrics": metrics,
    }
    threshold_path = root / "thresholds.json"
    threshold_path.write_text(json.dumps(config), encoding="utf-8")
    threshold_sha256 = hashlib.sha256(canonical_json_bytes(config)).hexdigest()
    previous = "0" * 64
    records = []
    for sequence in range(1, 4):
        values = {name: 0.0 for name in METRICS}
        values["capture_fps"] = 60.0
        values["action_confirmations"] = 1.0
        record = seal_event(
            {
                "schema": EVENT_SCHEMA,
                "sequence": sequence,
                "monotonic_ns": sequence * 1_000_000,
                "experiment_id": "controlled-001",
                "measurements": values,
                "provenance": config["required_provenance"],
                "threshold_sha256": threshold_sha256,
            },
            previous,
        )
        records.append(record)
        previous = record["sha256"]
    event_path = root / "events.jsonl"
    event_path.write_bytes(b"".join(encode_event(record) for record in records))
    return build_report(event_path, threshold_path), event_path, threshold_path


def evidence(report: dict[str, object]) -> dict[str, object]:
    artifact_hashes = [
        report["event_stream"]["sha256"],
        report["threshold_config_sha256"],
        canonical_json_sha256(report),
    ]

    def metric(field: str) -> dict[str, object]:
        direction = (
            "min"
            if field
            in {
                "frame_rate_hz",
                "maximum_clicks_per_second",
                "fixed_action_rate_hz",
            }
            else "max"
        )
        return {
            "unit": units[field],
            "direction": direction,
            "sample_count": 100,
            "p50": 0.01,
            "p95": 0.02,
            "p99": 0.03,
            "worst": 0.005 if direction == "min" else 0.04,
            "uncertainty": 0.001,
        }

    units = {
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
    }

    def section(*fields: str) -> dict[str, object]:
        return {
            "sample_count": 100,
            "uncertainty_method": "bootstrap-95-percent",
            "provenance_category": "observed",
            "experiment_ids": ["controlled-001"],
            "artifact_sha256": artifact_hashes,
            "measurements": {field: metric(field) for field in fields},
        }

    sections = {
        "wait_duration": section("gameplay_period_seconds", "scheduler_error_seconds"),
        "click_macro": section(
            "press_duration_seconds",
            "release_duration_seconds",
            "maximum_clicks_per_second",
        ),
        "cursor": section(),
        "capture": section(
            "frame_rate_hz", "request_to_completion_seconds", "stale_after_seconds"
        ),
        "effect_timing": section(
            "injection_to_poll_seconds",
            "effect_to_visible_seconds",
            "request_to_visible_seconds",
        ),
        "coordinate_calibration": section("residual_client_pixels"),
    }
    sections["cursor"].update(
        {
            "travel_model": "abstract_coordinate_fixed_rate",
            "quantization": "none observed",
            "cursor_retention": "retained",
            "path_logging": True,
            "measurements": {
                "fixed_action_rate_hz": metric("fixed_action_rate_hz"),
            },
        }
    )
    sections["click_macro"].update(
        {
            "input_provider_capability": (
                "targeted_edges_broker_deadline_claim_neutralization"
            ),
            "weak_button": "left",
            "strong_button": "right",
        }
    )
    sections["coordinate_calibration"].update(
        {"click_sweep_dimensions": 2, "continuous_drift_check": True}
    )
    return {
        "schema_version": "r4a-deployment-measurements-v1",
        "contract_version": "deployment-v1",
        "status": "measured",
        "soak_experiment_ids": ["controlled-001"],
        "soak_event_stream_sha256": report["event_stream"]["sha256"],
        "soak_threshold_config_sha256": report["threshold_config_sha256"],
        "soak_report_sha256": canonical_json_sha256(report),
        "provenance": {
            "game_executable_sha256": DIGEST,
            "box2d_sha256": DIGEST,
            "dxlib_sha256": DIGEST,
            "game_config_sha256": DIGEST,
            "measurement_tool_sha256": DIGEST,
            "runtime": "Wine test runtime",
            "hardware_id": "opaque-machine-profile",
        },
        "sections": sections,
    }


class R4AContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = load_deployment_contract(
            ROOT / "configs/rl/actions/deployment-v1.toml"
        )
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.report, self.events, self.thresholds = soak_artifacts(self.root)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def finalize(self, candidate, report=None):
        return finalize_deployment_contract(
            self.base,
            candidate,
            self.report if report is None else report,
            self.events,
            self.thresholds,
        )

    def test_checked_in_contract_is_explicitly_unmeasured_and_fail_closed(self) -> None:
        self.assertEqual(validate_deployment_contract(self.base), "provisional_unmeasured")
        self.assertFalse(self.base["live_deployment_enabled"])
        self.assertNotIn("frame_rate", self.base["capture"])
        self.assertNotIn("game_poll_latency_seconds", self.base["effect_timing"])

    def test_complete_evidence_finalizes_a_round_trip_measured_contract(self) -> None:
        measured = self.finalize(evidence(self.report))
        self.assertFalse(measured["live_deployment_enabled"])
        self.assertEqual(measured["measurement_status"], "measured_pending_review")
        self.assertEqual(measured["click_macro"]["press_ticks"], 1)
        self.assertEqual(
            measured["measurement_bundle_sha256"],
            canonical_json_sha256(evidence(self.report)),
        )
        rendered = render_toml(measured)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "measured.toml"
            path.write_text(rendered, encoding="utf-8")
            loaded = load_deployment_contract(path)
        self.assertEqual(
            validate_deployment_contract(loaded), "measured_pending_review"
        )

    def test_missing_or_fabricated_evidence_never_enables_deployment(self) -> None:
        for mutate in (
            lambda value: value["provenance"].pop("box2d_sha256"),
            lambda value: value["sections"]["capture"].update(sample_count=0),
            lambda value: value["sections"]["click_macro"]["measurements"][
                "press_duration_seconds"
            ].update(uncertainty=0.0),
            lambda value: value["sections"]["coordinate_calibration"].update(
                click_sweep_dimensions=1
            ),
        ):
            candidate = evidence(self.report)
            mutate(candidate)
            with self.subTest(candidate=candidate), self.assertRaises(ContractError):
                self.finalize(candidate)
            self.assertFalse(self.base["live_deployment_enabled"])
        failed_report = deepcopy(self.report)
        failed_report["status"] = "fail"
        candidate = evidence(failed_report)
        with self.assertRaises(ContractError):
            self.finalize(candidate, failed_report)
        forged_report = deepcopy(self.report)
        forged_report["metrics"]["capture_fps"]["p50"] = 999.0
        with self.assertRaisesRegex(ContractError, "does not match"):
            self.finalize(evidence(forged_report), forged_report)

    def test_provisional_safety_invariants_cannot_be_weakened(self) -> None:
        for key, value in (
            ("exact_window_identity_required", False),
            ("atomic_click_provider_supported", True),
            ("fail_closed_on_crop_or_claim_drift", False),
        ):
            candidate = deepcopy(self.base)
            candidate["safety"][key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_deployment_contract(candidate)
        for key, value in (
            ("wait_min_ticks", 999),
            ("wait_max_ticks", 1),
            ("serialization", "arbitrary"),
            ("coordinate_log_prob_epsilon", 0.49),
            ("sha256", "f" * 64),
        ):
            candidate = deepcopy(self.base)
            candidate[key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_deployment_contract(candidate)
        candidate = deepcopy(self.base)
        candidate["fairness"]["fast_forward"] = True
        with self.assertRaises(ContractError):
            validate_deployment_contract(candidate)
        for key, value in (
            ("measurement_schema", "anything"),
            ("soak_report_schema", "anything"),
            ("measured_values_require_positive_sample_count", False),
            ("measured_values_require_uncertainty", False),
            ("measured_values_require_artifact_hashes", False),
        ):
            candidate = deepcopy(self.base)
            candidate["evidence"][key] = value
            with self.subTest(key=key), self.assertRaises(ContractError):
                validate_deployment_contract(candidate)

    def test_soak_hash_and_exact_private_schema_are_enforced(self) -> None:
        candidate = evidence(self.report)
        candidate["soak_experiment_ids"] = ["other-experiment"]
        with self.assertRaisesRegex(ContractError, "experiments do not match"):
            self.finalize(candidate)

        for section, key, value in (
            ("capture", "private_path", "/home/user/private/capture"),
            ("click_macro", "claim_token", "SECRET"),
        ):
            candidate = evidence(self.report)
            candidate["sections"][section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                ContractError, "extra"
            ):
                self.finalize(candidate)
        candidate = evidence(self.report)
        candidate["sections"]["capture"]["measurements"]["private_path"] = (
            "/home/user/private/capture"
        )
        with self.assertRaisesRegex(ContractError, "extra"):
            self.finalize(candidate)
        candidate = evidence(self.report)
        candidate["sections"]["capture"]["experiment_ids"] = ["absent"]
        with self.assertRaisesRegex(ContractError, "absent from the soak"):
            self.finalize(candidate)
        candidate = evidence(self.report)
        candidate["sections"]["capture"]["artifact_sha256"] = [DIGEST]
        with self.assertRaisesRegex(ContractError, "not bound"):
            self.finalize(candidate)
        candidate = evidence(self.report)
        candidate["sections"]["cursor"]["quantization"] = (
            "/home/user/private/frame.png"
        )
        with self.assertRaisesRegex(ContractError, "private"):
            self.finalize(candidate)

    def test_tampered_quantiles_and_cursor_protocol_are_rejected(self) -> None:
        candidate = evidence(self.report)
        candidate["sections"]["effect_timing"]["measurements"][
            "request_to_visible_seconds"
        ]["p99"] = 0.001
        with self.assertRaisesRegex(ContractError, "not monotone"):
            self.finalize(candidate)
        candidate = evidence(self.report)
        candidate["sections"]["cursor"]["travel_model"] = "teleport"
        with self.assertRaisesRegex(ContractError, "fairness"):
            self.finalize(candidate)

    def test_measured_runtime_provenance_is_exact_schema(self) -> None:
        measured = self.finalize(evidence(self.report))
        measured["runtime_provenance"]["claim_token"] = "SECRET"
        with self.assertRaisesRegex(ContractError, "extra"):
            validate_deployment_contract(measured)

    def test_finalizer_publishes_no_replace(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "finalize_r4a_contract", ROOT / "tools/finalize-r4a-contract.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "contract.toml"
            module.publish_noreplace(output, "first\n")
            with self.assertRaisesRegex(ContractError, "already exists"):
                module.publish_noreplace(output, "second\n")
            self.assertEqual(output.read_text(encoding="utf-8"), "first\n")

    def test_r4a_tools_help_works_with_stdlib_only_imports(self) -> None:
        for name in ("report-r4a-soak.py", "finalize-r4a-contract.py"):
            result = subprocess.run(
                [sys.executable, "-S", str(ROOT / "tools" / name), "--help"],
                text=True,
                capture_output=True,
                check=False,
            )
            with self.subTest(name=name):
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_finalizer_cli_rebuilds_sources_and_stays_review_blocked(self) -> None:
        evidence_path = self.root / "evidence.json"
        report_path = self.root / "report.json"
        output = self.root / "measured.toml"
        evidence_path.write_text(
            json.dumps(evidence(self.report)), encoding="utf-8"
        )
        report_path.write_text(json.dumps(self.report), encoding="utf-8")
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(ROOT / "tools/finalize-r4a-contract.py"),
                str(ROOT / "configs/rl/actions/deployment-v1.toml"),
                str(evidence_path),
                str(report_path),
                str(self.events),
                str(self.thresholds),
                str(output),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        finalized = load_deployment_contract(output)
        self.assertEqual(
            finalized["measurement_status"], "measured_pending_review"
        )
        self.assertFalse(finalized["live_deployment_enabled"])


if __name__ == "__main__":
    unittest.main()
