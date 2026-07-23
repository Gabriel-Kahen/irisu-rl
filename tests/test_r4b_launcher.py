from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_rl.original_game.launcher import (
    LaunchError,
    MeasurementProcess,
)
from irisu_rl.original_game.runtime import DisposableRunAttestation


class R4BLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "reference/runs/experiment-001"
        self.run.mkdir(parents=True)
        self.prefix = self.root / "prefix"
        self.prefix.mkdir()
        self.wine = self.root / "wine"
        self.wine.write_text(
            f"#!{sys.executable}\nimport time\ntime.sleep(60)\n",
            encoding="utf-8",
        )
        self.wine.chmod(0o700)
        self.attestation = DisposableRunAttestation(
            "experiment-001", "a" * 64, {"game_executable_sha256": "b" * 64}
        )
        self.patch_attest = patch(
            "irisu_rl.original_game.launcher.attest_disposable_run",
            return_value=self.attestation,
        )
        self.patch_wine = patch(
            "irisu_rl.original_game.launcher.attest_wine_runtime_descriptor",
            return_value="c" * 64,
        )
        self.patch_verify = patch(
            "irisu_rl.original_game.launcher.verify_attestation_unchanged"
        )
        self.patch_attest.start()
        self.patch_wine.start()
        self.verify = self.patch_verify.start()

    def tearDown(self) -> None:
        self.patch_verify.stop()
        self.patch_wine.stop()
        self.patch_attest.stop()
        self.temporary.cleanup()

    def process(self) -> MeasurementProcess:
        return MeasurementProcess(
            repo_root=self.root,
            run_dir=self.run,
            experiment_id="experiment-001",
            wine_executable=self.wine,
            wine_prefix=self.prefix,
            session_environment={"LANG": "C", "SECRET": "not-forwarded"},
        )

    def test_launch_is_exclusive_attested_and_stops_only_its_process(self) -> None:
        first = self.process()
        with self.assertRaisesRegex(LaunchError, "owns the lock"):
            self.process()
        with first:
            attestation = first.attestation
            self.assertGreater(attestation.launcher_process_id, 0)
            self.assertGreater(attestation.launcher_process_start_ticks, 0)
            self.assertEqual(len(attestation.launch_nonce_sha256), 64)
            self.assertEqual(len(attestation.wine_prefix_sha256), 64)
        self.verify.assert_called_once()
        second = self.process()
        second.close()

    def test_nonce_is_not_exposed_in_safe_attestation(self) -> None:
        process = self.process()
        raw_nonce = process.launch_nonce
        with process:
            self.assertNotIn(raw_nonce, repr(process.attestation))

    def test_discovered_target_is_bound_to_nonce_generation_and_session(self) -> None:
        process = self.process()
        with process:
            attestation = process.attestation
            binding = process.bind_target(
                attestation.launcher_process_id,
                attestation.launcher_process_start_ticks,
            )
            self.assertEqual(binding.process_id, attestation.launcher_process_id)
            self.assertEqual(binding.session_id, attestation.launcher_process_id)
            process.verify_target_binding(
                attestation.launcher_process_id,
                attestation.launcher_process_start_ticks,
            )
            with self.assertRaisesRegex(LaunchError, "bound process generation"):
                process.verify_target_binding(
                    attestation.launcher_process_id + 1,
                    attestation.launcher_process_start_ticks,
                )
            with self.assertRaisesRegex(LaunchError, "already bound"):
                process.bind_target(
                    attestation.launcher_process_id,
                    attestation.launcher_process_start_ticks,
                )

    def test_cleanup_releases_resources_even_when_final_attestation_fails(self) -> None:
        process = self.process()
        self.verify.side_effect = RuntimeError("mutated")
        with self.assertRaisesRegex(LaunchError, "runtime changed"):
            process.close()
        self.verify.side_effect = None
        replacement = self.process()
        replacement.close()

    def test_wine_prefix_change_invalidates_the_measurement(self) -> None:
        process = self.process()
        process.start()
        (self.prefix / "user.reg").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(LaunchError, "runtime changed"):
            process.close()
        replacement = self.process()
        replacement.close()


if __name__ == "__main__":
    unittest.main()
