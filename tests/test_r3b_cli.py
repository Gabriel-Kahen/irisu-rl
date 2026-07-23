from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from irisu_rl.r3b_cli import main


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/rl/experiments/r3b-completion-v1.toml"
CONFIG = ROOT / "configs/rl/experiments/r3b-operational-v1.toml"


class R3BCLITests(unittest.TestCase):
    def invoke(self, *arguments: str) -> tuple[int, str, str]:
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = main(arguments)
        return result, stdout.getvalue(), stderr.getvalue()

    def test_config_verify_reports_frozen_identities(self) -> None:
        code, output, error = self.invoke(
            "config",
            "verify",
            "--config",
            str(CONFIG),
            "--plan",
            str(PLAN),
        )
        self.assertEqual((code, error), (0, ""))
        result = json.loads(output)
        self.assertEqual(result["primary_backend"], "exact")
        self.assertFalse(result["transfer_eligible"])

    def test_experiment_init_status_and_verify_round_trip(self) -> None:
        manifest = {
            "version": "r3b-snapshot-bundle-v1",
            "source_sha256": "1" * 64,
            "library_sha256": "2" * 64,
            "store_sha256": "3" * 64,
            "runtime_backend": "exact-test",
            "runtime_identity_sha256": "4" * 64,
            "action_spec_sha256": "5" * 64,
            "generator_version": "r3b-full-game-generator-v1",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            (snapshots / "bundle.json").write_text(
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            run = root / "run"
            with mock.patch(
                "irisu_rl.r3b_cli._snapshot_bundle_sha256",
                return_value="a" * 64,
            ):
                code, output, error = self.invoke(
                    "experiment",
                    "init",
                    "--run-id",
                    "cli-smoke",
                    "--run-class",
                    "smoke",
                    "--snapshots",
                    str(snapshots),
                    "--output",
                    str(run),
                    "--config",
                    str(CONFIG),
                    "--plan",
                    str(PLAN),
                )
            self.assertEqual((code, error), (0, ""))
            self.assertFalse(json.loads(output)["acceptance_eligible"])
            for command in ("status", "verify"):
                code, output, error = self.invoke(
                    "experiment", command, "--run", str(run)
                )
                self.assertEqual((code, error), (0, ""))
                self.assertEqual(json.loads(output)["run_id"], "cli-smoke")

    def test_init_rejects_noncanonical_bundle_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshots = root / "snapshots"
            snapshots.mkdir()
            (snapshots / "bundle.json").write_text("{}\n", encoding="utf-8")
            code, _output, error = self.invoke(
                "experiment",
                "init",
                "--run-id",
                "bad",
                "--run-class",
                "canonical",
                "--snapshots",
                str(snapshots),
                "--portable-snapshots",
                str(snapshots),
                "--output",
                str(root / "run"),
                "--config",
                str(CONFIG),
                "--plan",
                str(PLAN),
            )
            self.assertEqual(code, 2)
            self.assertIn("schema differs", error)

    def test_one_process_and_sealed_commands_are_locked_and_dispatched(self) -> None:
        cases = {
            "canonical-run-job": (
                "irisu_rl.r3b_cli._command_experiment_canonical_run_job",
                ("--phase", "test", "--authorization", "a" * 64),
            ),
            "run-baselines": (
                "irisu_rl.r3b_cli._command_experiment_run_baselines",
                ("--authorization", "a" * 64),
            ),
            "finalize-test": (
                "irisu_rl.r3b_cli._command_experiment_finalize_test",
                ("--authorization", "a" * 64, "--baseline-artifact", "b" * 64),
            ),
        }
        for command, (handler, extra) in cases.items():
            with (
                self.subTest(command=command),
                mock.patch(handler, return_value={"command": command}) as execute,
                mock.patch("irisu_rl.r3b_cli.R3BRunLock") as lock,
            ):
                code, output, error = self.invoke(
                    "experiment",
                    command,
                    "--run",
                    "/tmp/r3-cli-test",
                    "--worker",
                    "/tmp/r3-worker",
                    "--library",
                    "/tmp/r3-library",
                    *extra,
                )
                self.assertEqual((code, error), (0, ""))
                self.assertEqual(json.loads(output), {"command": command})
                execute.assert_called_once()
                lock.assert_called_once_with("/tmp/r3-cli-test")

    def test_split_process_commands_do_not_advertise_test_phase(self) -> None:
        for command in ("canonical-update", "canonical-evaluate"):
            with self.subTest(command=command):
                error = io.StringIO()
                with (
                    contextlib.redirect_stderr(error),
                    self.assertRaisesRegex(SystemExit, "2"),
                ):
                    main(
                        (
                            "experiment",
                            command,
                            "--run",
                            "/tmp/r3-cli-test",
                            "--worker",
                            "/tmp/r3-worker",
                            "--library",
                            "/tmp/r3-library",
                            "--phase",
                            "test",
                        )
                    )
                self.assertIn("invalid choice", error.getvalue())


if __name__ == "__main__":
    unittest.main()
