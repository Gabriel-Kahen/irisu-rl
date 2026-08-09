from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_pointer import r3i_g3_evidence as v1
from irisu_pointer import r3i_g3_evidence_v2 as evidence


CAMPAIGN_SHA256 = hashlib.sha256(b"r3i-g3-v2-campaign").hexdigest()
PARTITION_SHA256 = hashlib.sha256(b"whole-seed-partition").hexdigest()
DATASET_SHA256 = hashlib.sha256(b"dataset-closure").hexdigest()


class R3IG3AuthorizationV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.stages = ["collect", "train", "calibrate"]
        self.gates = ["resolution-auroc", "safe-query-coverage"]
        self.topology = {
            "collect": [],
            "train": ["collect"],
            "calibrate": ["collect", "train"],
        }
        self.preregistration = evidence.build_authorization_preregistration(
            campaign_sha256=CAMPAIGN_SHA256,
            receipt_stage_inventory=self.stages,
            gate_name_inventory=self.gates,
            prerequisite_topology=self.topology,
        )
        collect = self.receipt("collect")
        train = self.receipt("train", [collect["receipt_sha256"]])
        calibrate = self.receipt(
            "calibrate",
            sorted([collect["receipt_sha256"], train["receipt_sha256"]]),
        )
        self.receipts = {
            "collect": collect,
            "train": train,
            "calibrate": calibrate,
        }
        self.reports = {
            "resolution-auroc": self.gate_report("resolution-auroc", value=0.81),
            "safe-query-coverage": self.gate_report(
                "safe-query-coverage", value=0.07
            ),
        }

    def receipt(
        self, stage: str, prerequisite_sha256s: list[str] | None = None
    ) -> dict:
        return v1.build_stage_receipt(
            campaign_sha256=CAMPAIGN_SHA256,
            stage=stage,
            inputs={"source_sha256": "1" * 64},
            outputs={"artifact_sha256": hashlib.sha256(stage.encode()).hexdigest()},
            prerequisite_receipt_sha256s=prerequisite_sha256s or [],
            passed=True,
            terminal=False,
        )

    def gate_report(
        self,
        name: str,
        *,
        passed: bool = True,
        terminal: bool = False,
        value: float = 0.0,
        campaign_sha256: str = CAMPAIGN_SHA256,
        preregistration_sha256: str | None = None,
    ) -> dict:
        return evidence.build_gate_report(
            gate_name=name,
            campaign_sha256=campaign_sha256,
            preregistration_sha256=(
                self.preregistration["preregistration_sha256"]
                if preregistration_sha256 is None
                else preregistration_sha256
            ),
            passed=passed,
            terminal=terminal,
            evidence={"value": value},
        )

    def authorize(
        self,
        *,
        receipts: dict | None = None,
        reports: dict | None = None,
        expected_preregistration_sha256: str | None = None,
        expected_stages: list[str] | None = None,
        expected_gates: list[str] | None = None,
    ) -> dict:
        return evidence.build_authorization(
            preregistration=self.preregistration,
            expected_preregistration_sha256=(
                self.preregistration["preregistration_sha256"]
                if expected_preregistration_sha256 is None
                else expected_preregistration_sha256
            ),
            expected_receipt_stage_inventory=(
                self.stages if expected_stages is None else expected_stages
            ),
            expected_gate_name_inventory=(
                self.gates if expected_gates is None else expected_gates
            ),
            receipts_by_stage=self.receipts if receipts is None else receipts,
            gate_reports=self.reports if reports is None else reports,
        )

    def test_dependency_and_ordered_preregistration_are_exact(self) -> None:
        manifest = evidence.dependency_manifest()
        self.assertEqual(
            manifest["source_sha256"], evidence.V1_EVIDENCE_SOURCE_SHA256
        )
        self.assertEqual(
            manifest["manifest_sha256"],
            evidence.V1_DEPENDENCY_MANIFEST_SHA256,
        )
        verified = evidence.validate_authorization_preregistration(
            self.preregistration
        )
        self.assertEqual(verified["receipt_stage_inventory"], self.stages)
        self.assertEqual(verified["gate_name_inventory"], self.gates)
        with self.assertRaises(evidence.EvidenceError):
            evidence.build_authorization_preregistration(
                campaign_sha256=CAMPAIGN_SHA256,
                receipt_stage_inventory=["train", "collect"],
                gate_name_inventory=["gate"],
                prerequisite_topology={
                    "train": ["collect"],
                    "collect": [],
                },
            )

    def test_authorization_binds_exact_named_gate_and_receipt_closure(self) -> None:
        authorization = self.authorize(
            receipts=dict(reversed(list(self.receipts.items()))),
            reports=dict(reversed(list(self.reports.items()))),
        )
        self.assertEqual(
            [item["stage"] for item in authorization["receipts"]], self.stages
        )
        self.assertEqual(
            [item["name"] for item in authorization["gates"]], self.gates
        )
        self.assertEqual(
            evidence.validate_authorization(
                authorization,
                preregistration=self.preregistration,
                expected_preregistration_sha256=self.preregistration[
                    "preregistration_sha256"
                ],
                expected_receipt_stage_inventory=self.stages,
                expected_gate_name_inventory=self.gates,
                receipts_by_stage=self.receipts,
                gate_reports=self.reports,
            ),
            authorization,
        )

        missing_receipt = dict(self.receipts)
        del missing_receipt["collect"]
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(receipts=missing_receipt)
        extra_receipt = dict(self.receipts)
        extra_receipt["stale"] = self.receipt("stale")
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(receipts=extra_receipt)

    def test_omitted_or_present_failed_gate_cannot_authorize(self) -> None:
        failed = dict(self.reports)
        failed["safe-query-coverage"] = self.gate_report(
            "safe-query-coverage", passed=False, terminal=True
        )
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(reports=failed)

        omitted = dict(failed)
        del omitted["safe-query-coverage"]
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(reports=omitted)

    def test_missing_extra_and_stale_prerequisite_hashes_fail(self) -> None:
        missing = dict(self.receipts)
        missing["train"] = self.receipt("train")
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(receipts=missing)

        stale = dict(self.receipts)
        stale["train"] = self.receipt("train", ["f" * 64])
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(receipts=stale)

        extra = dict(self.receipts)
        extra["collect"] = self.receipt(
            "collect", [self.receipts["train"]["receipt_sha256"]]
        )
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(receipts=extra)

    def test_gate_identity_substitution_relabel_and_foreign_binding_fail(self) -> None:
        substituted = {
            "resolution-auroc": self.reports["safe-query-coverage"],
            "safe-query-coverage": self.reports["resolution-auroc"],
        }
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(reports=substituted)

        relabeled = dict(self.reports)
        relabeled["resolution-auroc"] = self.gate_report("foreign-gate")
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(reports=relabeled)

        foreign_campaign = dict(self.reports)
        foreign_campaign["resolution-auroc"] = self.gate_report(
            "resolution-auroc", campaign_sha256="a" * 64
        )
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(reports=foreign_campaign)

        foreign_preregistration = dict(self.reports)
        foreign_preregistration["resolution-auroc"] = self.gate_report(
            "resolution-auroc", preregistration_sha256="b" * 64
        )
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(reports=foreign_preregistration)

    def test_external_preregistration_and_ordered_inventories_are_mandatory(self) -> None:
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(expected_preregistration_sha256="c" * 64)
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(expected_stages=list(reversed(self.stages)))
        with self.assertRaises(evidence.AuthorizationError):
            self.authorize(expected_gates=list(reversed(self.gates)))
        with self.assertRaises(evidence.EvidenceError):
            evidence.build_authorization_preregistration(
                campaign_sha256=CAMPAIGN_SHA256,
                receipt_stage_inventory=self.stages,
                gate_name_inventory=["gate", "gate"],
                prerequisite_topology=self.topology,
            )

    def test_duplicate_gate_report_identity_is_rejected_defensively(self) -> None:
        original = evidence._validate_gate_report
        first_identity = self.reports["resolution-auroc"]["gate_report_sha256"]

        def duplicate_identity(*args, **kwargs):
            report = original(*args, **kwargs)
            if kwargs["mapping_name"] == "safe-query-coverage":
                report["gate_report_sha256"] = first_identity
            return report

        with mock.patch.object(
            evidence, "_validate_gate_report", side_effect=duplicate_identity
        ):
            with self.assertRaises(evidence.AuthorizationError):
                self.authorize()


class R3IG3CollectionClosureV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "collection"
        self.root.mkdir()
        for seed in (3, 7):
            (self.root / f"{seed}.episode.json").write_text(
                json.dumps({"seed": seed}), encoding="utf-8"
            )
            (self.root / f"{seed}.queries.jsonl").write_text(
                json.dumps({"seed": seed}) + "\n", encoding="utf-8"
            )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_closure_binds_directory_and_final_file_rescan(self) -> None:
        closure = evidence.build_collection_closure(self.root, [7, 3])
        self.assertEqual(closure["seeds"], [3, 7])
        self.assertEqual(len(closure["files"]), 4)
        self.assertEqual(
            closure["directory"]["before"], closure["directory"]["after"]
        )
        self.assertEqual(
            evidence.validate_collection_closure(
                closure, root=self.root, expected_seeds=[3, 7]
            ),
            closure,
        )

    def test_concurrent_extra_after_hashing_is_rejected(self) -> None:
        original = evidence._hash_direct_file
        calls = 0

        def add_extra(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == 4:
                (self.root / "concurrent.tmp").write_bytes(b"x")
            return result

        with mock.patch.object(evidence, "_hash_direct_file", side_effect=add_extra):
            with self.assertRaises(evidence.EvidenceError):
                evidence.build_collection_closure(self.root, [3, 7])

    def test_late_extra_in_prior_path_stat_window_is_rejected(self) -> None:
        original = Path.lstat
        root_stats = 0

        def inject_after_stat(path, *args, **kwargs):
            nonlocal root_stats
            info = original(path, *args, **kwargs)
            if path == self.root:
                root_stats += 1
                if root_stats == 2:
                    (self.root / "late.tmp").write_bytes(b"late")
            return info

        with mock.patch.object(
            Path, "lstat", autospec=True, side_effect=inject_after_stat
        ):
            with self.assertRaises(evidence.EvidenceError):
                evidence.build_collection_closure(self.root, [3, 7])

    def test_replacement_after_descriptor_hash_is_rejected(self) -> None:
        original = evidence._hash_direct_file
        replacement = self.root.parent / "replacement"
        replacement.write_bytes(b"replacement")
        swapped = False

        def replace_hashed(*args, **kwargs):
            nonlocal swapped
            result = original(*args, **kwargs)
            if not swapped:
                os.replace(replacement, self.root / args[1])
                swapped = True
            return result

        with mock.patch.object(
            evidence, "_hash_direct_file", side_effect=replace_hashed
        ):
            with self.assertRaises(evidence.EvidenceError):
                evidence.build_collection_closure(self.root, [3, 7])

    def test_symlink_expected_name_and_root_path_swap_are_rejected(self) -> None:
        episode = self.root / "3.episode.json"
        outside = self.root.parent / "outside"
        episode.rename(outside)
        episode.symlink_to(outside)
        with self.assertRaises(evidence.EvidenceError):
            evidence.build_collection_closure(self.root, [3, 7])
        episode.unlink()
        outside.rename(episode)

        original = evidence._hash_direct_file
        calls = 0
        old_root = self.root.parent / "old-collection"

        def swap_root(*args, **kwargs):
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            if calls == 4:
                self.root.rename(old_root)
                self.root.mkdir()
            return result

        with mock.patch.object(evidence, "_hash_direct_file", side_effect=swap_root):
            with self.assertRaises(evidence.EvidenceError):
                evidence.build_collection_closure(self.root, [3, 7])


class R3IG3CheckpointEnvelopeV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.models = self.root / "models"
        self.evidence_root = self.root / "evidence"
        self.models.mkdir()
        self.evidence_root.mkdir()
        self.model = self.models / "model.bin"
        self.model.write_bytes(b"model-weights-v1")
        self.metadata = {
            "architecture": "dedicated-resolution-trunk",
            "member_count": 5,
            "features": ["candidate", "pooled-board"],
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def publish(
        self,
        name: str,
        *,
        metadata: dict | None = None,
        partition_sha256: str = PARTITION_SHA256,
        dataset_sha256: str = DATASET_SHA256,
    ) -> dict:
        return evidence.write_checkpoint_envelope(
            self.evidence_root / name,
            evidence_root=self.evidence_root,
            model_path=self.model,
            model_root=self.models,
            metadata=self.metadata if metadata is None else metadata,
            partition_sha256=partition_sha256,
            dataset_sha256=dataset_sha256,
        )

    def load(
        self,
        path: Path,
        envelope: dict,
        *,
        metadata: dict | None = None,
        partition_sha256: str = PARTITION_SHA256,
        dataset_sha256: str = DATASET_SHA256,
    ) -> dict:
        return evidence.load_checkpoint_envelope(
            path,
            evidence_root=self.evidence_root,
            model_root=self.models,
            expected_envelope_sha256=envelope["envelope_sha256"],
            expected_model_sha256=envelope["model_sha256"],
            expected_metadata=self.metadata if metadata is None else metadata,
            expected_partition_sha256=partition_sha256,
            expected_dataset_sha256=dataset_sha256,
        )

    def test_write_once_envelope_and_exact_load_bind_every_identity(self) -> None:
        result = self.publish("checkpoint.json")
        envelope = result["envelope"]
        self.assertEqual(envelope["model_sha256"], hashlib.sha256(self.model.read_bytes()).hexdigest())
        self.assertEqual(envelope["metadata_sha256"], v1.sha256_json(self.metadata))
        self.assertEqual(
            self.load(self.evidence_root / "checkpoint.json", envelope), envelope
        )
        with self.assertRaises(evidence.EvidenceError):
            self.publish("checkpoint.json")
        with self.assertRaises(evidence.EvidenceError):
            evidence.build_checkpoint_envelope(
                model_path=self.model,
                model_root=self.models,
                metadata=self.metadata,
                partition_sha256=PARTITION_SHA256.upper(),
                dataset_sha256=DATASET_SHA256,
            )

    def test_exact_expected_metadata_partition_and_dataset_are_required(self) -> None:
        changed_metadata = dict(self.metadata)
        changed_metadata["member_count"] = 4
        changed = self.publish("changed-metadata.json", metadata=changed_metadata)[
            "envelope"
        ]
        with self.assertRaises(evidence.EvidenceError):
            self.load(self.evidence_root / "changed-metadata.json", changed)

        other_partition = hashlib.sha256(b"other-partition").hexdigest()
        changed_partition = self.publish(
            "changed-partition.json", partition_sha256=other_partition
        )["envelope"]
        with self.assertRaises(evidence.EvidenceError):
            self.load(
                self.evidence_root / "changed-partition.json", changed_partition
            )

        other_dataset = hashlib.sha256(b"other-dataset").hexdigest()
        changed_dataset = self.publish(
            "changed-dataset.json", dataset_sha256=other_dataset
        )["envelope"]
        with self.assertRaises(evidence.EvidenceError):
            self.load(self.evidence_root / "changed-dataset.json", changed_dataset)

    def test_model_replacement_and_symlink_envelope_fail_closed(self) -> None:
        result = self.publish("checkpoint.json")
        envelope = result["envelope"]

        replacement = self.models / "replacement"
        replacement.write_bytes(self.model.read_bytes())
        os.replace(replacement, self.model)
        with self.assertRaises(evidence.EvidenceError):
            self.load(self.evidence_root / "checkpoint.json", envelope)

        link = self.evidence_root / "checkpoint-link.json"
        link.symlink_to(self.evidence_root / "checkpoint.json")
        with self.assertRaises(evidence.EvidenceError):
            self.load(link, envelope)

    def test_model_symlink_and_envelope_path_swap_are_rejected(self) -> None:
        result = self.publish("checkpoint.json")
        envelope = result["envelope"]
        original = self.models / "original-model"
        self.model.rename(original)
        self.model.symlink_to(original)
        with self.assertRaises(evidence.EvidenceError):
            self.load(self.evidence_root / "checkpoint.json", envelope)
        self.model.unlink()
        original.rename(self.model)

        replacement_record = dict(envelope)
        replacement_record["partition_sha256"] = hashlib.sha256(
            b"tampered"
        ).hexdigest()
        replacement_record = v1.seal_record(
            {
                key: value
                for key, value in replacement_record.items()
                if key != "envelope_sha256"
            },
            "envelope_sha256",
        )
        replacement_path = self.evidence_root / "replacement.json"
        v1.write_once_atomic_json(replacement_path, replacement_record)
        os.replace(replacement_path, self.evidence_root / "checkpoint.json")
        with self.assertRaises(evidence.EvidenceError):
            self.load(self.evidence_root / "checkpoint.json", envelope)

    def test_model_swap_during_final_root_lstat_is_rejected(self) -> None:
        result = self.publish("checkpoint.json")
        envelope = result["envelope"]
        replacement = self.root / "replacement-model"
        replacement.write_bytes(self.model.read_bytes())
        original = Path.lstat
        model_root_stats = 0

        def swap_after_stat(path, *args, **kwargs):
            nonlocal model_root_stats
            info = original(path, *args, **kwargs)
            if path == self.models:
                model_root_stats += 1
                if model_root_stats == 2:
                    os.replace(replacement, self.model)
            return info

        with mock.patch.object(
            Path, "lstat", autospec=True, side_effect=swap_after_stat
        ):
            with self.assertRaises(evidence.EvidenceError):
                self.load(self.evidence_root / "checkpoint.json", envelope)

    def test_load_rebinds_envelope_after_semantic_validation(self) -> None:
        result = self.publish("checkpoint.json")
        envelope = result["envelope"]
        target = self.evidence_root / "checkpoint.json"
        replacement = self.root / "replacement-envelope"
        replacement.write_bytes(target.read_bytes())
        original = evidence._direct_file_binding
        envelope_reads = 0

        def swap_after_initial_binding(path, *args, **kwargs):
            nonlocal envelope_reads
            bound = original(path, *args, **kwargs)
            if Path(path) == target:
                envelope_reads += 1
                if envelope_reads == 1:
                    os.replace(replacement, target)
            return bound

        with mock.patch.object(
            evidence, "_direct_file_binding", side_effect=swap_after_initial_binding
        ):
            with self.assertRaises(evidence.EvidenceError):
                self.load(target, envelope)

    def test_pair_boundary_catches_envelope_swap_during_model_path_check(self) -> None:
        result = self.publish("checkpoint.json")
        envelope = result["envelope"]
        target = self.evidence_root / "checkpoint.json"
        replacement = self.root / "replacement-envelope"
        replacement.write_bytes(target.read_bytes())
        original = Path.lstat
        model_root_stats = 0

        def swap_envelope_after_model_stat(path, *args, **kwargs):
            nonlocal model_root_stats
            info = original(path, *args, **kwargs)
            if path == self.models:
                model_root_stats += 1
                if model_root_stats == 3:
                    os.replace(replacement, target)
            return info

        with mock.patch.object(
            Path,
            "lstat",
            autospec=True,
            side_effect=swap_envelope_after_model_stat,
        ):
            with self.assertRaises(evidence.EvidenceError):
                self.load(target, envelope)

    def test_write_rebinds_model_and_exact_published_envelope(self) -> None:
        original_writer = evidence.write_once_atomic_json
        replacement_model = self.root / "replacement-model"
        replacement_model.write_bytes(self.model.read_bytes())

        def replace_model_after_publish(*args, **kwargs):
            publication = original_writer(*args, **kwargs)
            os.replace(replacement_model, self.model)
            return publication

        with mock.patch.object(
            evidence,
            "write_once_atomic_json",
            side_effect=replace_model_after_publish,
        ):
            with self.assertRaises(evidence.EvidenceError):
                self.publish("model-race.json")

        self.model.write_bytes(b"model-weights-v1")
        replacement_envelope = self.root / "replacement-envelope"

        def replace_envelope_after_publish(*args, **kwargs):
            publication = original_writer(*args, **kwargs)
            replacement_envelope.write_bytes(Path(args[0]).read_bytes())
            os.replace(replacement_envelope, args[0])
            return publication

        with mock.patch.object(
            evidence,
            "write_once_atomic_json",
            side_effect=replace_envelope_after_publish,
        ):
            with self.assertRaises(evidence.EvidenceError):
                self.publish("envelope-race.json")


if __name__ == "__main__":
    unittest.main()
