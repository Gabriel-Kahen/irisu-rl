from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
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
    METRICS,
    METRIC_UNITS,
    REPORT_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
DIGEST = "a" * 64


def soak_report() -> dict[str, object]:
    metric_reports = {}
    for name in METRICS:
        metric_reports[name] = {
            "unit": METRIC_UNITS[name],
            "count": 100,
            "total": 0.0 if name in {
                "button_release_failures",
                "cross_window_misroutes",
            } else 100.0,
            "evaluation": {"status": "pass"},
        }
    return {
        "schema": REPORT_SCHEMA,
        "status": "pass",
        "event_stream": {
            "count": 10_000,
            "sha256": DIGEST,
            "chain_head_sha256": DIGEST,
        },
        "threshold_config_sha256": DIGEST,
        "duration": {
            "seconds": 600.0,
            "minimum_seconds": 300.0,
            "status": "pass",
        },
        "experiments": {
            "required": ["controlled-001"],
            "observed": ["controlled-001"],
            "missing": [],
            "status": "pass",
        },
        "provenance": {
            "required": {
                "game_executable_sha256": DIGEST,
                "measurement_tool_sha256": DIGEST,
            },
            "observed": {
                "game_executable_sha256": DIGEST,
                "measurement_tool_sha256": DIGEST,
            },
            "missing": [],
            "mismatched": [],
            "measurement_events_missing_required": 0,
            "status": "pass",
        },
        "metrics": metric_reports,
    }


def evidence(report: dict[str, object] | None = None) -> dict[str, object]:
    report = soak_report() if report is None else report

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
            "artifact_sha256": [DIGEST],
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
            "input_provider_capability": "targeted_explicit_down_up_release_all",
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

    def test_checked_in_contract_is_explicitly_unmeasured_and_fail_closed(self) -> None:
        self.assertEqual(validate_deployment_contract(self.base), "provisional_unmeasured")
        self.assertFalse(self.base["live_deployment_enabled"])
        self.assertNotIn("frame_rate", self.base["capture"])
        self.assertNotIn("game_poll_latency_seconds", self.base["effect_timing"])

    def test_complete_evidence_finalizes_a_round_trip_measured_contract(self) -> None:
        report = soak_report()
        measured = finalize_deployment_contract(self.base, evidence(report), report)
        self.assertTrue(measured["live_deployment_enabled"])
        self.assertEqual(measured["click_macro"]["press_ticks"], 1)
        self.assertEqual(
            measured["measurement_bundle_sha256"],
            canonical_json_sha256(evidence(report)),
        )
        rendered = render_toml(measured)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "measured.toml"
            path.write_text(rendered, encoding="utf-8")
            loaded = load_deployment_contract(path)
        self.assertEqual(validate_deployment_contract(loaded), "measured")

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
            candidate = evidence()
            mutate(candidate)
            with self.subTest(candidate=candidate), self.assertRaises(ContractError):
                finalize_deployment_contract(self.base, candidate, soak_report())
            self.assertFalse(self.base["live_deployment_enabled"])
        failed_report = soak_report()
        failed_report["status"] = "fail"
        candidate = evidence(failed_report)
        with self.assertRaises(ContractError):
            finalize_deployment_contract(self.base, candidate, failed_report)

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

    def test_soak_hash_and_exact_private_schema_are_enforced(self) -> None:
        report = soak_report()
        candidate = evidence(report)
        candidate["soak_experiment_ids"] = ["other-experiment"]
        with self.assertRaisesRegex(ContractError, "experiments do not match"):
            finalize_deployment_contract(self.base, candidate, report)

        for section, key, value in (
            ("capture", "private_path", "/home/user/private/capture"),
            ("click_macro", "claim_token", "SECRET"),
        ):
            candidate = evidence(report)
            candidate["sections"][section][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                ContractError, "extra"
            ):
                finalize_deployment_contract(self.base, candidate, report)

    def test_tampered_quantiles_and_cursor_protocol_are_rejected(self) -> None:
        candidate = evidence()
        candidate["sections"]["effect_timing"]["measurements"][
            "request_to_visible_seconds"
        ]["p99"] = 0.001
        with self.assertRaisesRegex(ContractError, "not monotone"):
            finalize_deployment_contract(self.base, candidate, soak_report())
        candidate = evidence()
        candidate["sections"]["cursor"]["travel_model"] = "teleport"
        with self.assertRaisesRegex(ContractError, "fairness"):
            finalize_deployment_contract(self.base, candidate, soak_report())

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


if __name__ == "__main__":
    unittest.main()
