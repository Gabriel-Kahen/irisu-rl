from __future__ import annotations

import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from irisu_rl.r3b_artifacts import (
    ArtifactIntegrityError,
    ArtifactLookupIndex,
    ArtifactStore,
    ArtifactStoreError,
    ArtifactTypeError,
    ExactResumeVerificationReceipt,
    UnsafeArtifactIdError,
    canonical_json_bytes,
)
from irisu_rl.r3b_experiments import (
    ExactResumeArtifact,
    _verified_exact_resume_artifact,
)


class ArtifactStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "artifacts"
        self.store = ArtifactStore(self.root)

    def test_round_trip_is_canonical_private_and_content_addressed(self) -> None:
        envelope = self.store.publish(
            kind="test.metrics",
            version="metrics-v1",
            payload={"z": [True, None, 1.25], "a": "snowman \N{SNOWMAN}"},
        )

        path = self.store.path_for(envelope.artifact_id)
        self.assertEqual(path.read_bytes(), canonical_json_bytes(envelope.manifest()))
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(self.root.stat().st_mode), 0o700)
        self.assertEqual(
            self.store.load(
                envelope.artifact_id,
                expected_kind="test.metrics",
                expected_version="metrics-v1",
            ),
            envelope,
        )
        self.assertEqual(self.store.list(), (envelope.artifact_id,))
        self.assertEqual(self.store.verify_all(), (envelope,))

    def test_publication_is_idempotent_and_preserves_existing_bytes(self) -> None:
        first = self.store.publish(kind="test", version="v1", payload={"a": 1})
        path = self.store.path_for(first.artifact_id)
        before = path.read_bytes()

        repeated = self.store.publish(kind="test", version="v1", payload={"a": 1})

        self.assertEqual(repeated, first)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.root.glob(".artifact-*.tmp")), [])

    def test_failed_fsync_publishes_nothing_and_cleans_temp(self) -> None:
        with mock.patch("irisu_rl.r3b_artifacts.os.fsync", side_effect=OSError("disk")):
            with self.assertRaisesRegex(OSError, "disk"):
                self.store.publish(kind="test", version="v1", payload={})

        self.assertEqual(list(self.root.iterdir()), [])

    def test_rejects_unsafe_ids_without_touching_outside_files(self) -> None:
        outside = Path(self.temporary.name) / "outside.json"
        outside.write_text("safe")
        for artifact_id in (
            "",
            "../outside",
            "a" * 63,
            "A" * 64,
            "g" * 64,
            f"{'a' * 64}/suffix",
        ):
            with self.subTest(artifact_id=artifact_id):
                with self.assertRaises(UnsafeArtifactIdError):
                    self.store.load(artifact_id)
        self.assertEqual(outside.read_text(), "safe")

    def test_rejects_non_finite_and_non_json_payloads(self) -> None:
        for payload in (
            {"value": float("nan")},
            {"value": float("inf")},
            {"value": (1, 2)},
            {1: "non-string"},
        ):
            with self.subTest(payload=payload):
                with self.assertRaises(ArtifactStoreError):
                    self.store.publish(kind="test", version="v1", payload=payload)

    def test_rejects_kind_and_version_mismatch(self) -> None:
        artifact = self.store.publish(kind="metrics", version="v1", payload={})
        with self.assertRaises(ArtifactTypeError):
            self.store.load(artifact.artifact_id, expected_kind="checkpoint")
        with self.assertRaises(ArtifactTypeError):
            self.store.load(artifact.artifact_id, expected_version="v2")

    def test_rejects_corruption_noncanonical_bytes_and_permission_changes(self) -> None:
        artifact = self.store.publish(kind="test", version="v1", payload={"a": 1})
        path = self.store.path_for(artifact.artifact_id)
        original = path.read_bytes()

        path.write_bytes(original.replace(b'"a":1', b'"a":2'))
        os.chmod(path, 0o600)
        with self.assertRaisesRegex(ArtifactIntegrityError, "SHA-256"):
            self.store.load(artifact.artifact_id)

        path.write_bytes(
            json.dumps(json.loads(original), indent=2, sort_keys=True).encode()
        )
        os.chmod(path, 0o600)
        with self.assertRaisesRegex(ArtifactIntegrityError, "canonical"):
            self.store.load(artifact.artifact_id)

        path.write_bytes(original)
        os.chmod(path, 0o644)
        with self.assertRaisesRegex(ArtifactIntegrityError, "metadata"):
            self.store.load(artifact.artifact_id)

    def test_rejects_symlink_root_and_artifact_file(self) -> None:
        actual = Path(self.temporary.name) / "actual"
        actual.mkdir(mode=0o700)
        linked = Path(self.temporary.name) / "linked"
        linked.symlink_to(actual, target_is_directory=True)
        with self.assertRaisesRegex(ArtifactIntegrityError, "symlink"):
            ArtifactStore(linked)

        artifact_id = "a" * 64
        target = Path(self.temporary.name) / "target"
        target.write_text("{}")
        self.store.path_for(artifact_id).symlink_to(target)
        with self.assertRaises(ArtifactIntegrityError):
            self.store.load(artifact_id)
        with self.assertRaises(ArtifactIntegrityError):
            self.store.list()

    def test_rejects_noncanonical_envelope_schema_and_duplicate_keys(self) -> None:
        artifact = self.store.publish(kind="test", version="v1", payload={})
        path = self.store.path_for(artifact.artifact_id)
        value = json.loads(path.read_bytes())
        value["extra"] = True
        path.write_bytes(canonical_json_bytes(value))
        os.chmod(path, 0o600)
        with self.assertRaisesRegex(ArtifactIntegrityError, "schema"):
            self.store.load(artifact.artifact_id)

        duplicate = (
            b'{"content_sha256":"'
            + artifact.artifact_id.encode()
            + b'","kind":"test","kind":"test","payload":{},"version":"v1"}'
        )
        path.write_bytes(duplicate)
        os.chmod(path, 0o600)
        with self.assertRaisesRegex(ArtifactIntegrityError, "duplicate"):
            self.store.load(artifact.artifact_id)

    def test_rejects_hardlinked_artifacts_and_unexpected_entries(self) -> None:
        artifact = self.store.publish(kind="test", version="v1", payload={})
        hardlink = Path(self.temporary.name) / "copy"
        os.link(self.store.path_for(artifact.artifact_id), hardlink)
        with self.assertRaisesRegex(ArtifactIntegrityError, "metadata"):
            self.store.load(artifact.artifact_id)
        hardlink.unlink()

        unexpected = self.root / "notes.txt"
        unexpected.write_text("not evidence")
        os.chmod(unexpected, 0o600)
        with self.assertRaisesRegex(ArtifactIntegrityError, "unexpected"):
            self.store.list()

    def test_lookup_index_is_durable_idempotent_and_not_authority(self) -> None:
        index = ArtifactLookupIndex(Path(self.temporary.name) / "index.sqlite3")
        lookup_key = "9" * 64
        envelope = self.store.publish(
            kind="test.lookup", version="lookup-v1", payload={"value": 1}
        )
        self.assertIsNone(
            index.lookup(
                lookup_key,
                self.store,
                expected_kind="test.lookup",
                expected_version="lookup-v1",
            )
        )
        index.record(lookup_key, envelope)
        index.record(lookup_key, envelope)
        self.assertEqual(
            ArtifactLookupIndex(index.path).lookup(
                lookup_key,
                self.store,
                expected_kind="test.lookup",
                expected_version="lookup-v1",
            ),
            envelope,
        )
        different = self.store.publish(
            kind="test.lookup", version="lookup-v1", payload={"value": 2}
        )
        with self.assertRaisesRegex(ArtifactIntegrityError, "different content"):
            index.record(lookup_key, different)

        self.store.path_for(envelope.artifact_id).write_bytes(b"{}")
        os.chmod(self.store.path_for(envelope.artifact_id), 0o600)
        with self.assertRaises(ArtifactIntegrityError):
            index.lookup(
                lookup_key,
                self.store,
                expected_kind="test.lookup",
                expected_version="lookup-v1",
            )

    def test_lookup_index_closes_every_sqlite_connection(self) -> None:
        index = ArtifactLookupIndex(Path(self.temporary.name) / "index.sqlite3")
        envelope = self.store.publish(kind="test", version="v1", payload={})
        index.record("8" * 64, envelope)
        descriptors = Path("/proc/self/fd")
        before = len(tuple(descriptors.iterdir()))
        for _ in range(500):
            self.assertEqual(
                index.lookup(
                    "8" * 64,
                    self.store,
                    expected_kind="test",
                    expected_version="v1",
                ),
                envelope,
            )
        after = len(tuple(descriptors.iterdir()))
        self.assertLessEqual(after, before + 1)


class ExactResumeVerificationReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.store = ArtifactStore(Path(self.temporary.name) / "artifacts")
        self.artifact = _verified_exact_resume_artifact(
            "1" * 64,
            "2" * 64,
            "3" * 64,
            "4" * 64,
            "4" * 64,
            "5" * 64,
            "5" * 64,
        )

    def test_verified_artifact_round_trips_only_as_a_receipt(self) -> None:
        receipt = ExactResumeVerificationReceipt.capture(
            self.artifact,
            verifier_identity_sha256="6" * 64,
            build_identity_sha256="7" * 64,
        )
        envelope = receipt.publish(self.store)
        loaded = ExactResumeVerificationReceipt.load(self.store, envelope.artifact_id)

        self.assertEqual(loaded, receipt)
        self.assertNotIsInstance(loaded, ExactResumeArtifact)
        self.assertFalse(hasattr(loaded, "_verification_token"))

    def test_capture_rejects_lookalikes_and_unequal_states(self) -> None:
        with self.assertRaisesRegex(ArtifactStoreError, "verified"):
            ExactResumeVerificationReceipt.capture(
                object(),  # type: ignore[arg-type]
                verifier_identity_sha256="6" * 64,
                build_identity_sha256="7" * 64,
            )
        manifest = ExactResumeVerificationReceipt.capture(
            self.artifact,
            verifier_identity_sha256="6" * 64,
            build_identity_sha256="7" * 64,
        ).manifest()
        manifest["restored_after_state_sha256"] = "8" * 64
        with self.assertRaisesRegex(ArtifactStoreError, "equal"):
            ExactResumeVerificationReceipt.from_manifest(manifest)

    def test_receipt_loader_rejects_wrong_envelope_type(self) -> None:
        envelope = self.store.publish(
            kind="not-a-receipt",
            version="r3b-exact-resume-verification-receipt-v1",
            payload={},
        )
        with self.assertRaises(ArtifactTypeError):
            ExactResumeVerificationReceipt.load(self.store, envelope.artifact_id)


if __name__ == "__main__":
    unittest.main()
