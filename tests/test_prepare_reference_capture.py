from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    name = "irisu_prepare_reference_capture"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools" / "prepare-reference-capture.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PrepareReferenceCaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        launcher = self.root / "tools/launch-reference-game.sh"
        launcher.parent.mkdir(parents=True)
        launcher.write_text("#!/bin/sh\nexit 0\n")
        launcher.chmod(0o755)
        self.source = self.root / "reference/game/irisu-v2.03-en"
        files = {
            "irisu.exe": b"game",
            "data/dll/Box2D.dll": b"box2d",
            "data/dll/DxLib.dll": b"dxlib",
            "data/doc/irisu.ini": b"ini",
            "data/doc/irisu.dat": b"config-data",
            "data/dat.dxa": b"data",
            "data/img.dxa": b"images",
            "data/snd.dxa": b"sound",
            "save.dat": b"save",
            "replay/new.rpy": struct.pack("<5i", 9, 0, 0, 0, 0) + bytes(32),
            "launch-irisu.sh": b"#!/bin/sh\nexec /home/gabe/Games/irisu.exe\n",
            "readme.txt": b"copied whole tree",
        }
        for relative, data in files.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        self.replay = self.root / "seed-41.rpy"
        self.replay.write_bytes(
            struct.pack("<5i", 41, 1, 16, 2, 0) + bytes(32) + struct.pack("<II", 0, 7)
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepare(
        self, experiment_id: str = "seed41-controlled-001", layout: str = "padded"
    ):
        return TOOL.prepare_capture(
            experiment_id,
            self.replay,
            repo_root=self.root,
            source_dir=self.source,
            now=datetime(2026, 7, 19, 12, 34, 56, tzinfo=timezone.utc),
            layout=layout,
        )

    def test_prepares_verified_run_and_non_golden_bundle(self) -> None:
        report = self.prepare()
        run = Path(report["run_dir"])
        capture = Path(report["capture_dir"])
        copied = run / "replay/seed41-controlled-001.rpy"
        self.assertEqual(copied.read_bytes(), self.replay.read_bytes())
        self.assertEqual((capture / "input.rpy").read_bytes(), self.replay.read_bytes())
        self.assertEqual((run / "readme.txt").read_bytes(), b"copied whole tree")
        self.assertFalse((run / "launch-irisu.sh").exists())
        self.assertTrue((run / ".irisu-reference-run").is_file())
        self.assertEqual(
            (self.source / "launch-irisu.sh").read_bytes(),
            b"#!/bin/sh\nexec /home/gabe/Games/irisu.exe\n",
        )
        self.assertTrue((capture / "frames").is_dir())
        self.assertFalse((capture / "result.rpy").exists())

        metadata = json.loads((capture / "metadata.json").read_text())
        self.assertEqual(metadata["status"], "prepared_non_golden")
        self.assertFalse(metadata["capture"]["golden_eligible"])
        self.assertEqual(metadata["input_replay"]["seed"], 41)
        self.assertEqual(metadata["input_replay"]["frame_count"], 2)
        self.assertEqual(metadata["input_replay"]["sha256"], digest(self.replay.read_bytes()))
        self.assertIn("one in-memory read", metadata["input_replay"]["snapshot_provenance"])
        self.assertEqual(metadata["game"]["box2d_sha256"], digest(b"box2d"))
        self.assertEqual(metadata["game"]["dxlib_sha256"], digest(b"dxlib"))
        self.assertEqual(metadata["game"]["config_sha256"], digest(b"ini"))
        self.assertEqual(metadata["game"]["dat_dxa_sha256"], digest(b"data"))
        self.assertEqual(metadata["run"]["initial_save_sha256"], digest(b"save"))
        self.assertEqual(
            metadata["run"]["initial_new_replay_sha256"],
            digest((self.source / "replay/new.rpy").read_bytes()),
        )
        self.assertIn("replay/seed41-controlled-001.rpy", metadata["run"]["initial_tree_sha256"])
        self.assertNotIn("launch-irisu.sh", metadata["run"]["initial_tree_sha256"])
        self.assertEqual(
            metadata["run"]["source_tree_adjustments"][0]["operation"],
            "removed_from_disposable_copy",
        )
        self.assertEqual(
            metadata["run"]["launch_command"],
            f"IRISU_GAME_DIR={run} {self.root}/tools/launch-reference-game.sh",
        )
        self.assertNotIn("launch-irisu.sh", metadata["run"]["launch_command"])

        actions = [
            json.loads(line)
            for line in (capture / "actions.jsonl").read_text().splitlines()
        ]
        self.assertEqual([entry["monotonic_sequence"] for entry in actions], [1, 2, 3])
        self.assertEqual(actions[1]["action"], "remove_copied_historical_launcher")
        self.assertEqual(
            actions[2]["result"],
            "both copies are byte-identical to the single source snapshot",
        )

    def test_refuses_existing_destination_without_mutating_it(self) -> None:
        run = self.root / "reference/runs/collision"
        run.mkdir(parents=True)
        marker = run / "owned-by-user"
        marker.write_text("keep")
        with self.assertRaisesRegex(TOOL.PreparationError, "already exists"):
            self.prepare("collision")
        self.assertEqual(marker.read_text(), "keep")
        self.assertFalse((self.root / "reference/captures/collision").exists())

    def test_rejects_path_traversal_and_malformed_replay_before_writes(self) -> None:
        with self.assertRaisesRegex(TOOL.PreparationError, "experiment ID"):
            self.prepare("../escape")
        self.replay.write_bytes(b"not a replay")
        with self.assertRaisesRegex(TOOL.PreparationError, "shorter"):
            self.prepare("malformed")
        self.assertFalse((self.root / "reference/runs").exists())
        self.assertFalse((self.root / "reference/captures").exists())

    def test_requires_complete_source_before_creating_destinations(self) -> None:
        (self.source / "data/snd.dxa").unlink()
        with self.assertRaisesRegex(TOOL.PreparationError, "data/snd.dxa"):
            self.prepare("missing-runtime")
        self.assertFalse((self.root / "reference/runs").exists())
        self.assertFalse((self.root / "reference/captures").exists())

    def test_requires_workspace_launcher_before_creating_destinations(self) -> None:
        (self.root / "tools/launch-reference-game.sh").unlink()
        with self.assertRaisesRegex(TOOL.PreparationError, "reference launcher"):
            self.prepare("missing-launcher")
        self.assertFalse((self.root / "reference/runs").exists())
        self.assertFalse((self.root / "reference/captures").exists())

    def test_auto_layout_refuses_zero_prefixed_legacy_ambiguity(self) -> None:
        with self.assertRaisesRegex(TOOL.PreparationError, "layout is ambiguous"):
            self.prepare("ambiguous", layout="auto")
        self.assertFalse((self.root / "reference/runs").exists())
        self.assertFalse((self.root / "reference/captures").exists())

    def test_explicit_legacy_layout_keeps_first_eight_zero_records(self) -> None:
        self.replay.write_bytes(
            struct.pack("<5i", 41, 0, 0, 0, 0) + bytes(32) + struct.pack("<I", 7)
        )
        report = self.prepare("legacy-zero-prefix", layout="legacy")
        metadata = json.loads((Path(report["capture_dir"]) / "metadata.json").read_text())
        self.assertEqual(metadata["input_replay"]["layout"], "legacy")
        self.assertEqual(metadata["input_replay"]["frame_count"], 9)
        self.assertEqual(metadata["input_replay"]["layout_selection"], "explicit")

    def test_replay_provenance_uses_one_immutable_byte_snapshot(self) -> None:
        original_bytes = self.replay.read_bytes()
        replacement = struct.pack("<5i", 99, 0, 0, 0, 0) + bytes(32)
        original_read_bytes = Path.read_bytes
        changed = False

        def mutate_after_snapshot(path: Path) -> bytes:
            nonlocal changed
            data = original_read_bytes(path)
            if path == self.replay and not changed:
                changed = True
                self.replay.write_bytes(replacement)
            return data

        with mock.patch.object(Path, "read_bytes", mutate_after_snapshot):
            report = self.prepare("single-snapshot")

        run = Path(report["run_dir"])
        capture = Path(report["capture_dir"])
        metadata = json.loads((capture / "metadata.json").read_text())
        self.assertEqual(self.replay.read_bytes(), replacement)
        self.assertEqual((capture / "input.rpy").read_bytes(), original_bytes)
        self.assertEqual((run / "replay/single-snapshot.rpy").read_bytes(), original_bytes)
        self.assertEqual(metadata["input_replay"]["seed"], 41)
        self.assertEqual(metadata["input_replay"]["size_bytes"], len(original_bytes))
        self.assertEqual(metadata["input_replay"]["sha256"], digest(original_bytes))

    def test_rejects_source_directory_symlink_without_touching_target(self) -> None:
        outside = self.root / "outside-replay"
        outside.mkdir()
        inherited = self.source / "replay/new.rpy"
        original = inherited.read_bytes()
        shutil.rmtree(self.source / "replay")
        (outside / "new.rpy").write_bytes(original)
        (self.source / "replay").symlink_to(outside, target_is_directory=True)

        with self.assertRaisesRegex(TOOL.PreparationError, "contains a symlink"):
            self.prepare("linked-source")
        self.assertEqual((outside / "new.rpy").read_bytes(), original)
        self.assertEqual(list(outside.iterdir()), [outside / "new.rpy"])
        self.assertFalse((self.root / "reference/runs").exists())
        self.assertFalse((self.root / "reference/captures").exists())

    def test_rejects_symlinked_source_root_before_creating_destinations(self) -> None:
        linked_source = self.root / "linked-source-root"
        linked_source.symlink_to(self.source, target_is_directory=True)
        with self.assertRaisesRegex(TOOL.PreparationError, "source tree must not"):
            TOOL.prepare_capture(
                "linked-root",
                self.replay,
                repo_root=self.root,
                source_dir=linked_source,
                layout="padded",
            )
        self.assertFalse((self.root / "reference/runs").exists())
        self.assertFalse((self.root / "reference/captures").exists())

    def test_atomic_publish_never_replaces_even_an_empty_directory(self) -> None:
        staged = self.root / "staged"
        target = self.root / "target"
        staged.mkdir()
        target.mkdir()
        (staged / "ours").write_text("ours")
        with self.assertRaisesRegex(TOOL.PreparationError, "destination appeared"):
            TOOL.publish_noreplace(staged, target)
        self.assertTrue((staged / "ours").is_file())
        self.assertEqual(list(target.iterdir()), [])

    def test_capture_publish_failure_never_deletes_replaced_run_path(self) -> None:
        real_publish = TOOL.publish_noreplace
        calls = 0
        displaced = self.root / "reference/runs/displaced-owned-run"

        def race_publish(source: Path, destination: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                run = self.root / "reference/runs/publish-race"
                os.rename(run, displaced)
                run.mkdir()
                (run / "concurrent-owner").write_text("keep")
                destination.mkdir()
                (destination / "concurrent-owner").write_text("keep")
            real_publish(source, destination)

        with mock.patch.object(TOOL, "publish_noreplace", race_publish):
            with self.assertRaisesRegex(TOOL.PreparationError, "destination appeared"):
                self.prepare("publish-race")

        run = self.root / "reference/runs/publish-race"
        capture = self.root / "reference/captures/publish-race"
        self.assertEqual((run / "concurrent-owner").read_text(), "keep")
        self.assertEqual((capture / "concurrent-owner").read_text(), "keep")
        self.assertTrue((displaced / ".irisu-reference-run").is_file())


if __name__ == "__main__":
    unittest.main()
