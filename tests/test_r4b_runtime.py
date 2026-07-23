from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_rl.original_game.private_io import (
    PrivateArtifactError,
    PrivateJournal,
    publish_private_bundle_noreplace,
    publish_private_noreplace,
    snapshot_private_files,
)
from irisu_rl.original_game.runtime import (
    REQUIRED_RUNTIME_FILES,
    RuntimeAttestationError,
    attest_disposable_run,
    attest_wine_prefix,
    attest_wine_runtime_descriptor,
    verify_attestation_unchanged,
    verify_wine_prefix_unchanged,
)


class R4BRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "reference" / "runs" / "experiment-001"
        self.run.mkdir(parents=True)
        hashes = {}
        for index, (relative, key) in enumerate(REQUIRED_RUNTIME_FILES.items()):
            path = self.run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"fixture-{index}".encode()
            path.write_bytes(payload)
            hashes[key] = hashlib.sha256(payload).hexdigest()
        self.hashes = hashes
        (self.run / ".irisu-reference-run").write_text(
            "created_by=tools/create-reference-run.sh\n"
            "created_utc=2026-07-23T00:00:00Z\n"
            f"irisu_exe_sha256={hashes['game_executable_sha256']}\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_attests_and_detects_runtime_change(self) -> None:
        with self.assertRaisesRegex(RuntimeAttestationError, "canonical"):
            attest_disposable_run(
                self.root, self.run, expected_experiment_id="experiment-001"
            )
        attestation = attest_disposable_run(
            self.root,
            self.run,
            expected_experiment_id="experiment-001",
            canonical_runtime_sha256=self.hashes,
        )
        self.assertEqual(attestation.experiment_id, "experiment-001")
        verify_attestation_unchanged(attestation, self.root, self.run)
        (self.run / "data/doc/irisu.ini").write_bytes(b"changed")
        with self.assertRaisesRegex(RuntimeAttestationError, "canonical|changed"):
            verify_attestation_unchanged(attestation, self.root, self.run)

    def test_immutable_baseline_and_windows_module_allowlist_fail_closed(self) -> None:
        attestation = attest_disposable_run(
            self.root,
            self.run,
            expected_experiment_id="experiment-001",
            canonical_runtime_sha256=self.hashes,
        )
        (self.run / "readme.txt").write_text("added later", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeAttestationError, "changed"):
            verify_attestation_unchanged(attestation, self.root, self.run)
        (self.run / "readme.txt").unlink()
        (self.run / "proxy.dll").write_bytes(b"not approved")
        with self.assertRaisesRegex(
            RuntimeAttestationError, "unapproved Windows module"
        ):
            attest_disposable_run(
                self.root,
                self.run,
                expected_experiment_id="experiment-001",
                canonical_runtime_sha256=self.hashes,
            )

    def test_rejects_preserved_wrong_named_and_linked_runs(self) -> None:
        with self.assertRaises(RuntimeAttestationError):
            attest_disposable_run(
                self.root,
                self.run,
                expected_experiment_id="other",
                canonical_runtime_sha256=self.hashes,
            )
        link = self.run / "linked"
        link.symlink_to(self.run / "irisu.exe")
        with self.assertRaisesRegex(RuntimeAttestationError, "symlink"):
            attest_disposable_run(
                self.root,
                self.run,
                expected_experiment_id="experiment-001",
                canonical_runtime_sha256=self.hashes,
            )

    def test_rejects_hardlinked_required_file(self) -> None:
        os.link(self.run / "irisu.exe", self.run / "irisu-copy.exe")
        with self.assertRaisesRegex(
            RuntimeAttestationError, "private regular|unapproved Windows module"
        ):
            attest_disposable_run(
                self.root,
                self.run,
                expected_experiment_id="experiment-001",
                canonical_runtime_sha256=self.hashes,
            )

    def test_wine_attestation_binds_an_open_safe_executable(self) -> None:
        wine = self.root / "wine"
        wine.write_bytes(b"wine-fixture")
        wine.chmod(0o700)
        descriptor = os.open(wine, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            with patch(
                "irisu_rl.original_game.runtime.CANONICAL_WINE_SHA256",
                hashlib.sha256(b"wine-fixture").hexdigest(),
            ):
                self.assertEqual(
                    attest_wine_runtime_descriptor(descriptor),
                    hashlib.sha256(b"wine-fixture").hexdigest(),
                )
                wine.chmod(0o720)
                with self.assertRaisesRegex(RuntimeAttestationError, "permissions"):
                    attest_wine_runtime_descriptor(descriptor)
        finally:
            os.close(descriptor)

    def test_wine_prefix_is_content_bound_and_rejects_unsafe_symlinks(self) -> None:
        prefix = self.root / "prefix"
        (prefix / "drive_c/windows").mkdir(parents=True)
        (prefix / "system.reg").write_text("registry-v1", encoding="utf-8")
        (prefix / "drive_c/windows/kernel.dll").write_bytes(b"builtin")
        attestation = attest_wine_prefix(prefix)
        self.assertNotIn("registry-v1", repr(attestation))
        self.assertNotIn(str(prefix), repr(attestation))
        verify_wine_prefix_unchanged(attestation, prefix)
        (prefix / "system.reg").write_text("registry-v2", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeAttestationError, "changed"):
            verify_wine_prefix_unchanged(attestation, prefix)
        (prefix / "system.reg").write_text("registry-v1", encoding="utf-8")
        (prefix / "drive_c/windows/injected.dll").symlink_to(
            self.root / "outside.dll"
        )
        with self.assertRaisesRegex(RuntimeAttestationError, "unsafe symlink"):
            attest_wine_prefix(prefix)

    def test_private_publication_requires_0700_and_never_replaces(self) -> None:
        output = self.root / "private"
        output.mkdir(mode=0o700)
        publish_private_noreplace(output, "report.json", b"{}\n")
        self.assertEqual((output / "report.json").read_bytes(), b"{}\n")
        self.assertEqual((output / "report.json").stat().st_mode & 0o777, 0o600)
        self.assertEqual((output / "report.json").stat().st_nlink, 1)
        self.assertEqual(tuple(output.iterdir()), (output / "report.json",))
        with self.assertRaises(PrivateArtifactError):
            publish_private_noreplace(output, "report.json", b"changed\n")
        output.chmod(0o755)
        with self.assertRaisesRegex(PrivateArtifactError, "0700"):
            publish_private_noreplace(output, "other.json", b"{}\n")

    def test_private_journal_is_complete_line_only_and_single_create(self) -> None:
        output = self.root / "journal"
        output.mkdir(mode=0o700)
        with PrivateJournal(output, "events.jsonl") as journal:
            with self.assertRaises(PrivateArtifactError):
                journal.append(b"partial")
            journal.append(b'{"sequence":1}\n')
        with self.assertRaises(PrivateArtifactError):
            PrivateJournal(output, "events.jsonl")

    def test_private_bundle_publishes_all_artifacts_at_once(self) -> None:
        output = self.root / "bundle-output"
        output.mkdir(mode=0o700)
        bundle = publish_private_bundle_noreplace(
            output,
            "measured",
            {"evidence.json": b"{}\n", "contract.toml": b"status = 'measured'\n"},
        )
        self.assertEqual(bundle.stat().st_mode & 0o777, 0o700)
        self.assertEqual(
            sorted(path.name for path in bundle.iterdir()),
            ["contract.toml", "evidence.json"],
        )
        self.assertTrue(
            all((path.stat().st_mode & 0o777) == 0o600 for path in bundle.iterdir())
        )
        with self.assertRaisesRegex(PrivateArtifactError, "already exists"):
            publish_private_bundle_noreplace(
                output,
                "measured",
                {"replacement.json": b"{}\n"},
            )
        self.assertEqual(
            sorted(path.name for path in output.iterdir()),
            ["measured"],
        )

    def test_private_inputs_are_single_open_snapshots(self) -> None:
        source_directory = self.root / "source"
        source_directory.mkdir(mode=0o700)
        source = source_directory / "events.jsonl"
        source.write_bytes(b'{"sequence":1}\n')
        source.chmod(0o600)
        with snapshot_private_files({"events.jsonl": (source, 1024)}) as snapshots:
            self.assertEqual(
                snapshots["events.jsonl"].read_bytes(), b'{"sequence":1}\n'
            )
            self.assertEqual(snapshots["events.jsonl"].stat().st_mode & 0o777, 0o600)
        source.chmod(0o644)
        with self.assertRaisesRegex(PrivateArtifactError, "0600"):
            with snapshot_private_files({"events.jsonl": (source, 1024)}):
                pass


if __name__ == "__main__":
    unittest.main()
