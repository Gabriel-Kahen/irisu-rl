from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from irisu_rl.r3b_artifacts import ArtifactStore
from irisu_rl.r3b_experiments import CandidateArm, TrainingCheckpointArtifact, TrialJob
from irisu_rl.r3b_supervisor import (
    _checkpoint_package,
    evaluate_trained_canonical_job,
)


def _hash(character: str) -> str:
    return character * 64


class R3BSupervisorTests(unittest.TestCase):
    def test_rejects_phase_without_opening_a_run(self) -> None:
        with self.assertRaisesRegex(ValueError, "phase"):
            evaluate_trained_canonical_job(
                "/missing",
                exact_worker_path="/missing",
                portable_library_path="/missing",
                phase="unknown",
            )
        with self.assertRaisesRegex(ValueError, "authorization"):
            evaluate_trained_canonical_job(
                "/missing",
                exact_worker_path="/missing",
                portable_library_path="/missing",
                phase="validation",
            )
        with self.assertRaisesRegex(ValueError, "sealed lease"):
            evaluate_trained_canonical_job(
                "/missing",
                exact_worker_path="/missing",
                portable_library_path="/missing",
                phase="test",
            )

    def test_checkpoint_package_binds_typed_receipt_and_manifest_bytes(self) -> None:
        job = TrialJob(
            _hash("1"),
            "calibration",
            CandidateArm(0, 0.0001),
            7,
            300,
            False,
            _hash("2"),
        )
        manifest_bytes = b'{"checkpoint":"fixture","files":{}}\n'
        manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
        checkpoint = TrainingCheckpointArtifact(
            7,
            50,
            100,
            100,
            job.plan_sha256,
            job.sha256,
            _hash("3"),
            _hash("4"),
            manifest_sha,
            _hash("5"),
            _hash("6"),
        )
        built = SimpleNamespace(
            manifest=SimpleNamespace(sha256=_hash("3"), runner_spec_sha256=_hash("4"))
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generation = "update-0050"
            checkpoint_root = root / "jobs" / job.sha256 / "checkpoints" / generation
            checkpoint_root.mkdir(parents=True)
            (checkpoint_root / "manifest.json").write_bytes(manifest_bytes)
            store = ArtifactStore(root / "artifacts")
            envelope = store.publish(
                kind="irisu.r3b.training-checkpoint",
                version="r3b-training-checkpoint-package-v2",
                payload={
                    "job_sha256": job.sha256,
                    "trial_manifest_sha256": _hash("3"),
                    "runner_spec_sha256": _hash("4"),
                    "completed_updates": 50,
                    "simulated_ticks": 100,
                    "model_sha256": _hash("5"),
                    "deployment_policy_sha256": _hash("6"),
                    "checkpoint_artifact": checkpoint.manifest(),
                    "generation": generation,
                    "checkpoint_manifest_sha256": manifest_sha,
                    "checkpoint_files": {},
                },
            )
            loaded, loaded_generation, _ = _checkpoint_package(
                root=root,
                store=store,
                artifact_sha256=envelope.artifact_id,
                built=built,
                job=job,
                target_update=50,
            )
            self.assertEqual(loaded, checkpoint)
            self.assertEqual(loaded_generation, generation)

            (checkpoint_root / "manifest.json").write_bytes(b"tampered\n")
            with self.assertRaisesRegex(ValueError, "missing or unsafe"):
                _checkpoint_package(
                    root=root,
                    store=store,
                    artifact_sha256=envelope.artifact_id,
                    built=built,
                    job=job,
                    target_update=50,
                )


if __name__ == "__main__":
    unittest.main()
