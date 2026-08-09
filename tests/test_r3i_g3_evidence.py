from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_pointer.r3i_g3_evidence import (
    AuthorizationError,
    EvidenceError,
    assert_split_disjointness,
    audit_threshold_calibration,
    build_affinity_report,
    build_authorization,
    build_seed_manifest,
    build_stage_receipt,
    build_threshold_rule,
    build_two_slot_plan,
    build_whole_seed_partition,
    calibrate_threshold,
    canonical_json_bytes,
    seal_record,
    sha256_bytes,
    sha256_json,
    validate_affinity_reports,
    validate_authorization,
    validate_collection_closure,
    validate_file_binding,
    validate_oof_closure,
    validate_regular_file,
    validate_seed_manifest,
    validate_stage_receipt,
    validate_whole_seed_partition,
    verify_sealed_record,
    write_once_atomic_json,
)


CAMPAIGN_SHA256 = hashlib.sha256(b"g3-campaign").hexdigest()


class _IntSubclass(int):
    pass


class R3IG3CanonicalEvidenceTests(unittest.TestCase):
    def test_canonical_json_and_hash_are_stable_and_strict(self) -> None:
        left = {"z": [1, 2.5, True, None], "a": "π"}
        right = {"a": "π", "z": [1, 2.5, True, None]}
        self.assertEqual(canonical_json_bytes(left), canonical_json_bytes(right))
        self.assertEqual(sha256_json(left), sha256_json(right))
        self.assertEqual(
            sha256_bytes(canonical_json_bytes(left)),
            hashlib.sha256(canonical_json_bytes(left)).hexdigest(),
        )
        for invalid in (
            {"x": math.nan},
            {"x": math.inf},
            {"x": (1, 2)},
            {"x": _IntSubclass(1)},
            {1: "non-string-key"},
        ):
            with self.assertRaises(EvidenceError):
                canonical_json_bytes(invalid)

    def test_seal_rejects_tampering_and_digest_field_in_payload(self) -> None:
        sealed = seal_record({"a": 1}, "digest_sha256")
        self.assertEqual(verify_sealed_record(sealed, "digest_sha256"), sealed)
        tampered = dict(sealed)
        tampered["a"] = 2
        with self.assertRaises(EvidenceError):
            verify_sealed_record(tampered, "digest_sha256")
        with self.assertRaises(EvidenceError):
            seal_record(sealed, "digest_sha256")


class R3IG3FileEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_regular_binding_is_live_single_link_and_root_relative(self) -> None:
        path = self.root / "evidence.json"
        path.write_bytes(b"evidence")
        binding = validate_regular_file(path, root=self.root)
        self.assertEqual(binding["path"], "evidence.json")
        self.assertEqual(binding["sha256"], hashlib.sha256(b"evidence").hexdigest())
        self.assertEqual(validate_file_binding(binding, root=self.root), binding)

        path.write_bytes(b"changed!")
        with self.assertRaises(EvidenceError):
            validate_file_binding(binding, root=self.root)

    def test_symlink_hardlink_and_parent_escape_fail_closed(self) -> None:
        source = self.root / "source"
        source.write_bytes(b"x")
        symlink = self.root / "symlink"
        symlink.symlink_to(source)
        with self.assertRaises(EvidenceError):
            validate_regular_file(symlink, root=self.root)

        hardlink = self.root / "hardlink"
        os.link(source, hardlink)
        with self.assertRaises(EvidenceError):
            validate_regular_file(source, root=self.root)
        with self.assertRaises(EvidenceError):
            validate_regular_file(hardlink, root=self.root)

        outside = self.root.parent / f"{self.root.name}-outside"
        outside.write_bytes(b"outside")
        try:
            with self.assertRaises(EvidenceError):
                validate_regular_file(outside, root=self.root)
        finally:
            outside.unlink()

    def test_atomic_json_is_canonical_fsynced_single_link_and_write_once(self) -> None:
        target = self.root / "receipt.json"
        value = {"z": 2, "a": [1, True]}
        binding = write_once_atomic_json(target, value)
        self.assertEqual(target.read_bytes(), canonical_json_bytes(value) + b"\n")
        self.assertEqual(target.stat().st_nlink, 1)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o444)
        self.assertEqual(binding["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
        self.assertFalse(any(".tmp-" in path.name for path in self.root.iterdir()))
        with self.assertRaises(EvidenceError):
            write_once_atomic_json(target, {"different": True})
        self.assertEqual(target.read_bytes(), canonical_json_bytes(value) + b"\n")


class R3IG3SeedAndCollectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_seed_manifest_is_deterministic_exact_and_disjoint(self) -> None:
        sizes = {"train": 16, "calibration": 8, "audit": 8}
        first = build_seed_manifest("g3-dev", 726_2026, sizes)
        second = build_seed_manifest("g3-dev", 726_2026, dict(reversed(list(sizes.items()))))
        self.assertEqual(first, second)
        splits = validate_seed_manifest(
            first, required_splits=["train", "calibration", "audit"]
        )
        self.assertEqual({name: len(values) for name, values in splits.items()}, sizes)
        assert_split_disjointness({key: list(value) for key, value in splits.items()})
        self.assertEqual(sum(map(len, splits.values())), len(set().union(*map(set, splits.values()))))

        tampered = copy.deepcopy(first)
        tampered["splits"][0]["seeds"][0] += 1
        tampered["manifest_sha256"] = sha256_json(
            {key: value for key, value in tampered.items() if key != "manifest_sha256"}
        )
        with self.assertRaises(EvidenceError):
            validate_seed_manifest(tampered)
        with self.assertRaises(EvidenceError):
            assert_split_disjointness({"left": [1], "right": [1]})

    def test_collection_closure_binds_exact_files_and_rejects_any_extra(self) -> None:
        for seed in (3, 7):
            (self.root / f"{seed}.episode.json").write_text(
                json.dumps({"seed": seed}), encoding="utf-8"
            )
            (self.root / f"{seed}.queries.jsonl").write_text(
                json.dumps({"seed": seed}) + "\n", encoding="utf-8"
            )
        closure = validate_collection_closure(self.root, [7, 3])
        self.assertEqual(closure["seeds"], [3, 7])
        self.assertEqual(len(closure["files"]), 4)
        self.assertEqual(
            verify_sealed_record(closure, "closure_sha256")["closure_sha256"],
            closure["closure_sha256"],
        )

        extra = self.root / "unexpected.tmp"
        extra.write_bytes(b"x")
        with self.assertRaises(EvidenceError):
            validate_collection_closure(self.root, [3, 7])
        extra.unlink()
        (self.root / "7.queries.jsonl").unlink()
        with self.assertRaises(EvidenceError):
            validate_collection_closure(self.root, [3, 7])

    def test_collection_closure_rejects_symlink_even_with_expected_name(self) -> None:
        outside = self.root.parent / f"{self.root.name}-outside"
        outside.write_bytes(b"outside")
        try:
            (self.root / "1.episode.json").symlink_to(outside)
            (self.root / "1.queries.jsonl").write_bytes(b"{}\n")
            with self.assertRaises(EvidenceError):
                validate_collection_closure(self.root, [1])
        finally:
            outside.unlink()


class R3IG3ReceiptAndAuthorizationTests(unittest.TestCase):
    def receipt(
        self,
        stage: str,
        *,
        prerequisite: tuple[str, ...] = (),
        passed: bool = True,
        terminal: bool = False,
    ) -> dict:
        return build_stage_receipt(
            campaign_sha256=CAMPAIGN_SHA256,
            stage=stage,
            inputs={"source_sha256": "1" * 64},
            outputs={"artifact_sha256": hashlib.sha256(stage.encode()).hexdigest()},
            prerequisite_receipt_sha256s=prerequisite,
            passed=passed,
            terminal=terminal,
            metadata={"cpu_seconds": 1.25},
        )

    def test_stage_receipt_binds_exact_inputs_outputs_and_prerequisites(self) -> None:
        first = self.receipt("collect")
        second = self.receipt("train", prerequisite=(first["receipt_sha256"],))
        self.assertEqual(
            validate_stage_receipt(
                second,
                campaign_sha256=CAMPAIGN_SHA256,
                stage="train",
                inputs={"source_sha256": "1" * 64},
                outputs={"artifact_sha256": hashlib.sha256(b"train").hexdigest()},
                prerequisite_receipt_sha256s=[first["receipt_sha256"]],
            ),
            second,
        )
        with self.assertRaises(EvidenceError):
            validate_stage_receipt(second, stage="evaluate")
        tampered = copy.deepcopy(second)
        tampered["outputs"]["artifact_sha256"] = "2" * 64
        with self.assertRaises(EvidenceError):
            validate_stage_receipt(tampered)
        with self.assertRaises(EvidenceError):
            self.receipt("bad", passed=False, terminal=False)

    def test_authorization_is_exact_and_revalidates_receipt_set(self) -> None:
        collect = self.receipt("collect")
        train = self.receipt("train", prerequisite=(collect["receipt_sha256"],))
        authorization = build_authorization(
            campaign_sha256=CAMPAIGN_SHA256,
            required_stages=["collect", "train"],
            receipts=[train, collect],
        )
        self.assertTrue(authorization["passed"])
        self.assertEqual(
            validate_authorization(authorization, receipts=[collect, train]),
            authorization,
        )
        replacement = self.receipt("train")
        with self.assertRaises(AuthorizationError):
            validate_authorization(authorization, receipts=[collect, replacement])
        with self.assertRaises(AuthorizationError):
            build_authorization(
                campaign_sha256=CAMPAIGN_SHA256,
                required_stages=["collect", "train"],
                receipts=[collect],
            )

    def test_terminal_passed_false_gate_and_receipt_block_authorization(self) -> None:
        failed_receipt = self.receipt("calibrate", passed=False, terminal=True)
        with self.assertRaises(AuthorizationError):
            build_authorization(
                campaign_sha256=CAMPAIGN_SHA256,
                required_stages=["calibrate"],
                receipts=[failed_receipt],
            )

        passed_receipt = self.receipt("calibrate")
        failed_gate = {
            "schema": "example-gate.v1",
            "passed": False,
            "terminal": True,
            "reason": "no-safe-threshold",
        }
        with self.assertRaises(AuthorizationError):
            build_authorization(
                campaign_sha256=CAMPAIGN_SHA256,
                required_stages=["calibrate"],
                receipts=[passed_receipt],
                gate_reports=[failed_gate],
            )


class R3IG3FoldAndCalibrationTests(unittest.TestCase):
    def test_eight_fold_partition_is_deterministic_whole_seed_and_exact(self) -> None:
        seeds = list(range(101, 117))
        first = build_whole_seed_partition(seeds, namespace="g3-oof")
        second = build_whole_seed_partition(list(reversed(seeds)), namespace="g3-oof")
        self.assertEqual(first, second)
        seed_to_fold = validate_whole_seed_partition(first, expected_seeds=seeds)
        self.assertEqual(set(seed_to_fold), set(seeds))
        self.assertEqual(set(seed_to_fold.values()), set(range(8)))
        self.assertTrue(all(len(fold["heldout_seeds"]) == 2 for fold in first["folds"]))
        with self.assertRaises(EvidenceError):
            build_whole_seed_partition(seeds, namespace="g3-oof", fold_count=4)

    def test_oof_closure_requires_exact_item_and_heldout_fold(self) -> None:
        partition = build_whole_seed_partition(list(range(8)), namespace="g3-oof")
        seed_to_fold = validate_whole_seed_partition(partition)
        expected = {seed: ["a", "b"] for seed in range(8)}
        predictions = [
            {
                "seed": seed,
                "fold": seed_to_fold[seed],
                "item_id": item,
                "score": float(seed) / 8.0,
            }
            for seed in range(8)
            for item in ("a", "b")
        ]
        closure = validate_oof_closure(
            partition, predictions, expected_items=expected
        )
        self.assertEqual(closure["prediction_count"], 16)

        wrong_fold = copy.deepcopy(predictions)
        wrong_fold[0]["fold"] = (wrong_fold[0]["fold"] + 1) % 8
        with self.assertRaises(EvidenceError):
            validate_oof_closure(partition, wrong_fold, expected_items=expected)
        with self.assertRaises(EvidenceError):
            validate_oof_closure(partition, predictions[:-1], expected_items=expected)
        with self.assertRaises(EvidenceError):
            validate_oof_closure(
                partition, predictions + [copy.deepcopy(predictions[0])], expected_items=expected
            )

    def test_continuous_calibration_checks_every_float_and_never_splits_ties(self) -> None:
        rule = build_threshold_rule(max_false_positives=1, minimum_selected=4)
        observations = [
            {"item": "a", "score": 0.9317, "acceptable": True},
            {"item": "b", "score": 0.8132, "acceptable": True},
            {"item": "c", "score": 0.8132, "acceptable": False},
            {"item": "d", "score": 0.8001, "acceptable": True},
            {"item": "e", "score": 0.7999, "acceptable": False},
        ]
        report = calibrate_threshold(observations, rule=rule)
        self.assertTrue(report["passed"])
        self.assertEqual(report["comparator"], ">=")
        self.assertEqual(report["threshold"], 0.8001)
        self.assertEqual(report["selected_count"], 4)
        self.assertEqual(report["false_positive_count"], 1)
        self.assertEqual(report["threshold_tie_block_size"], 1)
        self.assertEqual(
            audit_threshold_calibration(observations, report, fixed_rule=rule),
            report,
        )

        strict = build_threshold_rule(max_false_positives=0, minimum_selected=2)
        strict_report = calibrate_threshold(observations, rule=strict)
        self.assertFalse(strict_report["passed"])
        self.assertTrue(strict_report["terminal"])
        self.assertEqual(strict_report["threshold"], 0.9317)
        self.assertEqual(strict_report["selected_count"], 1)

    def test_fixed_rule_audit_rejects_report_rule_and_input_tampering(self) -> None:
        rule = build_threshold_rule(max_false_positives=0, minimum_selected=1)
        observations = [
            {"score": 0.9, "acceptable": True},
            {"score": 0.2, "acceptable": False},
        ]
        report = calibrate_threshold(observations, rule=rule)
        modified = copy.deepcopy(report)
        modified["threshold"] = 0.2
        payload = {key: value for key, value in modified.items() if key != "calibration_sha256"}
        modified["calibration_sha256"] = sha256_json(payload)
        with self.assertRaises(EvidenceError):
            audit_threshold_calibration(observations, modified, fixed_rule=rule)
        with self.assertRaises(EvidenceError):
            audit_threshold_calibration(
                observations + [{"score": 0.1, "acceptable": True}],
                report,
                fixed_rule=rule,
            )
        alternate_rule = build_threshold_rule(max_false_positives=1, minimum_selected=1)
        with self.assertRaises(EvidenceError):
            audit_threshold_calibration(
                observations, report, fixed_rule=alternate_rule
            )
        with self.assertRaises(EvidenceError):
            calibrate_threshold(
                [{"score": 1, "acceptable": True}], rule=rule
            )


class R3IG3ResourcePlanTests(unittest.TestCase):
    def test_two_slot_plan_and_affinity_reports_bind_single_cpu_and_threads(self) -> None:
        plan = build_two_slot_plan([2, 5])
        limits = plan["thread_limits"]
        reports = [
            build_affinity_report(
                plan,
                slot=slot,
                observed_cpus=[cpu],
                pid=1000 + slot,
                observed_thread_limits=limits,
            )
            for slot, cpu in enumerate((2, 5))
        ]
        verified = validate_affinity_reports(plan, list(reversed(reports)))
        self.assertEqual(set(verified), {0, 1})
        with self.assertRaises(EvidenceError):
            build_two_slot_plan([2, 2])
        with self.assertRaises(EvidenceError):
            build_affinity_report(
                plan,
                slot=0,
                observed_cpus=[2, 5],
                pid=1000,
                observed_thread_limits=limits,
            )
        bad_limits = dict(limits)
        bad_limits["OMP_NUM_THREADS"] = "2"
        with self.assertRaises(EvidenceError):
            build_affinity_report(
                plan,
                slot=0,
                observed_cpus=[2],
                pid=1000,
                observed_thread_limits=bad_limits,
            )


if __name__ == "__main__":
    unittest.main()
