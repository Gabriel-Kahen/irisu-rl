from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_rl.original_game.evidence import (
    EVENT_SCHEMA,
    METRICS,
    REPORT_SCHEMA,
    THRESHOLD_SCHEMA,
    EvidenceError,
    PublicationError,
    build_report,
    canonical_json_bytes,
    encode_event,
    generate_report,
    seal_event,
    verify_report,
    write_report_noreplace,
)

PROVENANCE = {
    "game_executable_sha256": hashlib.sha256(b"authorized-game").hexdigest(),
    "deployment_contract_sha256": hashlib.sha256(b"deployment-v1").hexdigest(),
}


def process_binding(experiment_id: str) -> dict[str, int | str]:
    identity = hashlib.sha256(experiment_id.encode("ascii")).hexdigest()
    return {
        "process_id": 1_000 + int(identity[:6], 16),
        "process_start_ticks": 10_000 + int(identity[6:12], 16),
        "launch_nonce_sha256": hashlib.sha256(
            f"{experiment_id}:launch".encode("ascii")
        ).hexdigest(),
        "runtime_identity_sha256": hashlib.sha256(
            f"{experiment_id}:runtime".encode("ascii")
        ).hexdigest(),
        "wine_prefix_sha256": "f" * 64,
    }


def thresholds(*, minimum_samples: int = 5) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name in METRICS:
        if name == "capture_fps":
            metrics[name] = {
                "direction": "min",
                "minimum_samples": minimum_samples,
                "worst": {"min": 50.0},
                "uncertainty_95": {"max": 10.0},
            }
        elif name == "action_confirmations":
            metrics[name] = {
                "direction": "min",
                "minimum_samples": minimum_samples,
                "total": {"min": float(minimum_samples)},
            }
        elif name in {
            "duplicate_frames",
            "dropped_frames",
            "stale_frames",
            "out_of_order_frames",
            "deadline_misses",
            "button_release_failures",
            "cross_window_misroutes",
        }:
            metrics[name] = {
                "direction": "max",
                "minimum_samples": minimum_samples,
                "total": {"max": 0.0},
            }
        else:
            metrics[name] = {
                "direction": "max",
                "minimum_samples": minimum_samples,
                "worst": {"max": 1_000_000.0},
            }
    return {
        "schema": THRESHOLD_SCHEMA,
        "experiment_ids": ["synthetic-soak-001"],
        "required_provenance": PROVENANCE,
        "minimum_duration_seconds": 0.004,
        "maximum_measurement_gap_seconds": 0.002,
        "metrics": metrics,
    }


def measurements(index: int) -> dict[str, float]:
    result = {name: 0.0 for name in METRICS}
    result.update(
        {
            "capture_fps": 59.0 + ((index - 1) % 5),
            "capture_jitter_seconds": 0.001 * index,
            "ring_age_seconds": 0.002 * index,
            "request_ack_seconds": 0.003 * index,
            "poll_effect_interval_seconds": 0.020,
            "effect_visible_seconds": 0.010 + 0.001 * index,
            "total_latency_seconds": 0.030 + 0.001 * index,
            "action_confirmations": 1.0,
            "crop_drift_pixels": 0.1 * index,
            "resource_growth_bytes": 1024.0 * index,
        }
    )
    return result


def threshold_digest(value: dict[str, object] | None = None) -> str:
    return hashlib.sha256(
        canonical_json_bytes(thresholds() if value is None else value)
    ).hexdigest()


def event(
    sequence: int,
    previous: str,
    *,
    values=None,
    provenance=None,
    threshold_value=None,
    monotonic_ns=None,
):
    return seal_event(
        {
            "schema": EVENT_SCHEMA,
            "sequence": sequence,
            "monotonic_ns": (
                sequence * 1_000_000 if monotonic_ns is None else monotonic_ns
            ),
            "experiment_id": "synthetic-soak-001",
            "process_binding": process_binding("synthetic-soak-001"),
            "measurements": measurements(sequence) if values is None else values,
            "provenance": PROVENANCE if provenance is None else provenance,
            "threshold_sha256": threshold_digest(threshold_value),
        },
        previous,
    )


class R4AEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.events = self.root / "events.jsonl"
        self.config = self.root / "thresholds.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_config(self, value=None) -> None:
        self.config.write_text(
            json.dumps(thresholds() if value is None else value),
            encoding="utf-8",
        )

    def write_events(
        self, count: int = 5, mutate=None, threshold_value=None
    ) -> list[dict]:
        records = []
        previous = "0" * 64
        for sequence in range(1, count + 1):
            record = event(sequence, previous, threshold_value=threshold_value)
            if mutate is not None:
                record = mutate(sequence, record)
            records.append(record)
            previous = record["sha256"]
        self.events.write_bytes(b"".join(encode_event(item) for item in records))
        return records

    def test_deterministic_passing_report_has_all_aggregate_statistics(self) -> None:
        records = self.write_events()
        self.write_config()
        first = build_report(self.events, self.config)
        second = build_report(self.events, self.config)
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], REPORT_SCHEMA)
        self.assertEqual(first["status"], "pass")
        self.assertEqual(first["event_stream"]["count"], 5)
        self.assertEqual(
            first["event_stream"]["chain_head_sha256"], records[-1]["sha256"]
        )
        self.assertEqual(
            first["process_bindings"]["synthetic-soak-001"],
            process_binding("synthetic-soak-001"),
        )
        self.assertEqual(set(first["metrics"]), set(METRICS))
        fps = first["metrics"]["capture_fps"]
        self.assertEqual(fps["count"], 5)
        self.assertEqual(fps["p50"], 61.0)
        self.assertEqual(fps["p95"], 62.8)
        self.assertEqual(fps["p99"], 62.96)
        self.assertEqual(fps["worst"], 59.0)
        self.assertGreater(fps["uncertainty_95"], 0)
        self.assertEqual(first["metrics"]["action_confirmations"]["total"], 5.0)
        self.assertIn("effect_confirmation_failures", first["metrics"])
        self.assertNotIn(str(self.events), json.dumps(first))

    def test_sustained_synthetic_soak_is_reproducible(self) -> None:
        config = thresholds(minimum_samples=512)
        self.write_events(512, threshold_value=config)
        self.write_config(config)
        left = build_report(self.events, self.config)
        right = build_report(self.events, self.config)
        self.assertEqual(left, right)
        self.assertEqual(left["status"], "pass")
        self.assertEqual(left["event_stream"]["count"], 512)
        self.assertEqual(left["duration"]["status"], "pass")
        self.assertEqual(left["metrics"]["total_latency_seconds"]["count"], 512)
        self.assertEqual(left["metrics"]["action_confirmations"]["total"], 512.0)

    def test_each_experiment_must_pass_instead_of_hiding_in_aggregate(self) -> None:
        config = thresholds(minimum_samples=2)
        config["experiment_ids"] = ["healthy", "degraded"]
        config["minimum_duration_seconds"] = 0.001
        config["maximum_measurement_gap_seconds"] = 0.001
        config["metrics"]["resource_growth_bytes"] = {
            "direction": "max",
            "minimum_samples": 2,
            "p50": {"max": 10.0},
        }
        previous = "0" * 64
        records = []
        for sequence in range(1, 103):
            experiment_id = "healthy" if sequence <= 100 else "degraded"
            values = measurements(sequence)
            values["resource_growth_bytes"] = (
                0.0 if experiment_id == "healthy" else 100.0
            )
            record = seal_event(
                {
                    "schema": EVENT_SCHEMA,
                    "sequence": sequence,
                    "monotonic_ns": sequence * 1_000_000,
                    "experiment_id": experiment_id,
                    "process_binding": process_binding(experiment_id),
                    "measurements": values,
                    "provenance": PROVENANCE,
                    "threshold_sha256": threshold_digest(config),
                },
                previous,
            )
            records.append(record)
            previous = record["sha256"]
        self.events.write_bytes(b"".join(encode_event(item) for item in records))
        self.write_config(config)
        report = build_report(self.events, self.config)
        metric = report["metrics"]["resource_growth_bytes"]
        self.assertEqual(metric["p50"], 0.0)
        self.assertEqual(metric["per_experiment"]["degraded"]["p50"], 100.0)
        self.assertEqual(metric["evaluation"]["status"], "fail")
        self.assertNotEqual(
            report["process_bindings"]["healthy"],
            report["process_bindings"]["degraded"],
        )

    def test_process_binding_cannot_change_within_an_experiment(self) -> None:
        records = self.write_events()
        changed = dict(records[2])
        unsigned = {
            key: value
            for key, value in changed.items()
            if key not in {"previous_sha256", "sha256"}
        }
        unsigned["process_binding"] = {
            **unsigned["process_binding"],
            "process_start_ticks": (
                unsigned["process_binding"]["process_start_ticks"] + 1
            ),
        }
        records[2] = seal_event(unsigned, records[1]["sha256"])
        previous = records[2]["sha256"]
        for index in range(3, len(records)):
            unsigned = {
                key: value
                for key, value in records[index].items()
                if key not in {"previous_sha256", "sha256"}
            }
            records[index] = seal_event(unsigned, previous)
            previous = records[index]["sha256"]
        self.events.write_bytes(b"".join(encode_event(item) for item in records))
        self.write_config()
        with self.assertRaisesRegex(EvidenceError, "process binding changed"):
            build_report(self.events, self.config)

    def test_distinct_experiment_labels_cannot_reuse_one_process(self) -> None:
        config = thresholds(minimum_samples=2)
        config["experiment_ids"] = ["run-a", "run-b"]
        config["minimum_duration_seconds"] = 0.001
        config["maximum_measurement_gap_seconds"] = 0.001
        shared = process_binding("shared-process")
        previous = "0" * 64
        records = []
        for sequence in range(1, 5):
            experiment_id = "run-a" if sequence <= 2 else "run-b"
            record = seal_event(
                {
                    "schema": EVENT_SCHEMA,
                    "sequence": sequence,
                    "monotonic_ns": sequence * 1_000_000,
                    "experiment_id": experiment_id,
                    "process_binding": shared,
                    "measurements": measurements(sequence),
                    "provenance": PROVENANCE,
                    "threshold_sha256": threshold_digest(config),
                },
                previous,
            )
            records.append(record)
            previous = record["sha256"]
        self.events.write_bytes(b"".join(encode_event(item) for item in records))
        self.write_config(config)
        with self.assertRaisesRegex(EvidenceError, "reuse the same process generation"):
            build_report(self.events, self.config)

    def test_threshold_breach_fails_with_specific_incident(self) -> None:
        def add_failure(sequence, record):
            if sequence == 3:
                unsigned = {
                    key: value
                    for key, value in record.items()
                    if key not in {"previous_sha256", "sha256"}
                }
                unsigned["measurements"] = dict(unsigned["measurements"])
                unsigned["measurements"]["cross_window_misroutes"] = 1.0
                return seal_event(unsigned, record["previous_sha256"])
            return record

        self.write_events(mutate=add_failure)
        # Rebuild the chain after altering the third record.
        source = [json.loads(line) for line in self.events.read_text().splitlines()]
        previous = "0" * 64
        rebuilt = []
        for item in source:
            unsigned = {
                key: value
                for key, value in item.items()
                if key not in {"previous_sha256", "sha256"}
            }
            sealed = seal_event(unsigned, previous)
            rebuilt.append(sealed)
            previous = sealed["sha256"]
        self.events.write_bytes(b"".join(encode_event(item) for item in rebuilt))
        self.write_config()
        report = build_report(self.events, self.config)
        self.assertEqual(report["status"], "fail")
        evaluation = report["metrics"]["cross_window_misroutes"]["evaluation"]
        self.assertEqual(evaluation["status"], "fail")
        self.assertIn("total 1.0 > maximum 0.0", evaluation["failures"])

    def test_misroute_and_release_failures_are_intrinsic_zero_gates(self) -> None:
        permissive = thresholds()
        for name in ("cross_window_misroutes", "button_release_failures"):
            permissive["metrics"][name]["total"]["max"] = 10.0
        self.write_config(permissive)
        for failed_metric in ("cross_window_misroutes", "button_release_failures"):
            previous = "0" * 64
            records = []
            for sequence in range(1, 6):
                values = measurements(sequence)
                if sequence == 3:
                    values[failed_metric] = 1.0
                sealed = event(
                    sequence,
                    previous,
                    values=values,
                    threshold_value=permissive,
                )
                records.append(sealed)
                previous = sealed["sha256"]
            self.events.write_bytes(b"".join(encode_event(item) for item in records))
            report = build_report(self.events, self.config)
            self.assertEqual(report["status"], "fail")
            failures = report["metrics"][failed_metric]["evaluation"]["failures"]
            self.assertIn("R4a requires total 0.0; observed 1.0", failures)

    def test_insufficient_samples_and_missing_provenance_are_not_evaluable(
        self,
    ) -> None:
        self.write_events(2)
        self.write_config()
        report = build_report(self.events, self.config)
        self.assertEqual(report["status"], "not_evaluable")
        self.assertEqual(
            report["metrics"]["total_latency_seconds"]["evaluation"]["status"],
            "not_evaluable",
        )

        previous = "0" * 64
        records = []
        for sequence in range(1, 6):
            sealed = event(sequence, previous, provenance={})
            records.append(sealed)
            previous = sealed["sha256"]
        self.events.write_bytes(b"".join(encode_event(item) for item in records))
        report = build_report(self.events, self.config)
        self.assertEqual(report["status"], "not_evaluable")
        self.assertEqual(set(report["provenance"]["missing"]), set(PROVENANCE))

    def test_short_soak_is_not_evaluable_even_with_enough_samples(self) -> None:
        self.write_events()
        config = thresholds()
        config["minimum_duration_seconds"] = 60.0
        self.write_events(threshold_value=config)
        self.write_config(config)
        report = build_report(self.events, self.config)
        self.assertEqual(report["status"], "not_evaluable")
        self.assertEqual(report["duration"]["status"], "not_evaluable")

    def test_empty_tail_cannot_supply_provenance_or_extend_soak_duration(self) -> None:
        previous = "0" * 64
        records = []
        for sequence in range(1, 5):
            sealed = event(sequence, previous, provenance={})
            records.append(sealed)
            previous = sealed["sha256"]
        tail = event(
            5,
            previous,
            values={},
            provenance=PROVENANCE,
            monotonic_ns=600_000_000_000,
        )
        records.append(tail)
        self.events.write_bytes(b"".join(encode_event(item) for item in records))
        self.write_config()
        report = build_report(self.events, self.config)
        self.assertEqual(report["status"], "not_evaluable")
        self.assertEqual(report["provenance"]["measurement_events_missing_required"], 4)
        self.assertAlmostEqual(report["duration"]["seconds"], 0.003)

    def test_event_chain_commits_to_thresholds_before_collection(self) -> None:
        self.write_events()
        changed = thresholds()
        changed["metrics"]["capture_fps"]["worst"]["min"] = 1.0
        self.write_config(changed)
        with self.assertRaisesRegex(EvidenceError, "not bound"):
            build_report(self.events, self.config)

    def test_metric_direction_is_fixed_by_the_r4a_schema(self) -> None:
        inverted = thresholds()
        inverted["metrics"]["capture_fps"]["direction"] = "max"
        self.write_events(threshold_value=inverted)
        self.write_config(inverted)
        with self.assertRaisesRegex(EvidenceError, "direction must be min"):
            build_report(self.events, self.config)

    def test_measurement_gap_coverage_is_required(self) -> None:
        config = thresholds()
        previous = "0" * 64
        records = []
        for sequence, timestamp in enumerate(
            (1_000_000, 2_000_000, 20_000_000, 21_000_000, 22_000_000),
            start=1,
        ):
            record = event(
                sequence,
                previous,
                threshold_value=config,
                monotonic_ns=timestamp,
            )
            records.append(record)
            previous = record["sha256"]
        self.events.write_bytes(b"".join(encode_event(item) for item in records))
        self.write_config(config)
        report = build_report(self.events, self.config)
        self.assertEqual(report["status"], "not_evaluable")
        self.assertEqual(report["coverage"]["status"], "not_evaluable")

    def test_each_metric_must_cover_each_experiment(self) -> None:
        config = thresholds()
        previous = "0" * 64
        records = []
        for sequence in range(1, 11):
            values = measurements(sequence) if sequence <= 5 else {"capture_fps": 60.0}
            record = event(
                sequence,
                previous,
                values=values,
                threshold_value=config,
            )
            records.append(record)
            previous = record["sha256"]
        self.events.write_bytes(b"".join(encode_event(item) for item in records))
        self.write_config(config)
        report = build_report(self.events, self.config)
        self.assertEqual(report["status"], "not_evaluable")
        coverage = report["coverage"]["experiments"]["synthetic-soak-001"]
        self.assertEqual(
            coverage["metrics"]["cross_window_misroutes"]["status"],
            "not_evaluable",
        )
        self.assertEqual(coverage["metrics"]["capture_fps"]["status"], "pass")

    def test_verified_report_must_match_its_source_artifacts(self) -> None:
        self.write_events()
        self.write_config()
        report = build_report(self.events, self.config)
        verify_report(report, self.events, self.config)
        report["metrics"]["capture_fps"]["p50"] = 999.0
        with self.assertRaisesRegex(EvidenceError, "does not match"):
            verify_report(report, self.events, self.config)

    def test_tampering_broken_links_and_unsafe_payloads_are_rejected(self) -> None:
        records = self.write_events()
        self.write_config()
        records[2]["measurements"]["total_latency_seconds"] = 999.0
        self.events.write_bytes(
            b"".join(canonical_json_bytes(item) + b"\n" for item in records)
        )
        with self.assertRaisesRegex(EvidenceError, "SHA-256"):
            build_report(self.events, self.config)

        with self.assertRaisesRegex(EvidenceError, "unsafe provenance"):
            event(1, "0" * 64, provenance={"claim_token_sha256": "a" * 64})
        fractional = measurements(1)
        fractional["dropped_frames"] = 0.5
        with self.assertRaisesRegex(EvidenceError, "integer count"):
            event(1, "0" * 64, values=fractional)
        unsafe = {
            "schema": EVENT_SCHEMA,
            "sequence": 1,
            "monotonic_ns": 1,
            "experiment_id": "synthetic-soak-001",
            "process_binding": process_binding("synthetic-soak-001"),
            "measurements": {},
            "provenance": PROVENANCE,
            "threshold_sha256": threshold_digest(),
            "pixels": "secret",
        }
        with self.assertRaisesRegex(EvidenceError, "keys disagree"):
            seal_event(unsafe)

    def test_provenance_mismatch_fails_and_midstream_change_is_rejected(self) -> None:
        wrong = dict(PROVENANCE)
        wrong["game_executable_sha256"] = "f" * 64
        previous = "0" * 64
        records = []
        for sequence in range(1, 6):
            sealed = event(sequence, previous, provenance=wrong)
            records.append(sealed)
            previous = sealed["sha256"]
        self.events.write_bytes(b"".join(encode_event(item) for item in records))
        self.write_config()
        report = build_report(self.events, self.config)
        self.assertEqual(report["status"], "fail")
        self.assertEqual(report["provenance"]["mismatched"], ["game_executable_sha256"])

        records[3] = event(4, records[2]["sha256"], provenance=PROVENANCE)
        records[4] = event(5, records[3]["sha256"], provenance=PROVENANCE)
        self.events.write_bytes(b"".join(encode_event(item) for item in records))
        with self.assertRaisesRegex(EvidenceError, "changed during stream"):
            build_report(self.events, self.config)

    def test_atomic_no_replace_preserves_existing_output_and_cleans_temp(self) -> None:
        self.write_events()
        self.write_config()
        destination = self.root / "report.json"
        destination.write_text("owned", encoding="utf-8")
        with self.assertRaises(PublicationError):
            generate_report(self.events, self.config, destination)
        self.assertEqual(destination.read_text(), "owned")
        self.assertEqual(
            sorted(path.name for path in self.root.glob(".report.json.*.tmp")), []
        )

        fresh = self.root / "fresh.json"
        report = build_report(self.events, self.config)
        write_report_noreplace(report, fresh)
        self.assertEqual(json.loads(fresh.read_text()), report)
        with self.assertRaises(PublicationError):
            write_report_noreplace(report, fresh)

    def test_cli_publishes_pass_and_returns_nonzero_for_a_failed_gate(self) -> None:
        self.write_events()
        self.write_config()
        output = self.root / "report.json"
        command = [
            sys.executable,
            str(ROOT / "tools/report-r4a-soak.py"),
            str(self.events),
            str(self.config),
            str(output),
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(output.read_text())["status"], "pass")
        self.assertEqual(json.loads(result.stdout)["status"], "pass")

        failed_config = thresholds()
        failed_config["metrics"]["capture_fps"]["worst"]["min"] = 100.0
        self.write_config(failed_config)
        self.write_events(threshold_value=failed_config)
        failed_output = self.root / "failed.json"
        result = subprocess.run(
            [*command[:-1], str(failed_output)],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(failed_output.read_text())["status"], "fail")


if __name__ == "__main__":
    unittest.main()
