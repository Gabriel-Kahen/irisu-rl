from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from irisu_env import PaddedVectorEnv
from irisu_rl import MacroVectorAdapter, SemanticAction, TeacherStateEncoder
from irisu_rl.actions import ActionSpec
from irisu_rl.checkpoints import (
    capture_rng_state,
    load_checkpoint,
    pack_adapter_checkpoint,
    restore_rng_state,
    save_checkpoint,
    unpack_adapter_checkpoint,
)
from irisu_rl.schema import TEACHER_V1

ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT / "build-physics-integration-portable" / "libirisu_clone.so"
EXACT = ROOT / "build-physics-integration-exact-multiworld-2" / "irisu-exact-worker"


class CheckpointFormatTests(unittest.TestCase):
    def test_immutable_generation_hash_validation_and_rng_round_trip(self) -> None:
        identity = {"schema": TEACHER_V1.sha256, "deployable": False}
        random.seed(13)
        torch.manual_seed(13)
        generator = np.random.default_rng(13)
        rng_state = capture_rng_state(generator)
        with tempfile.TemporaryDirectory() as directory:
            target = save_checkpoint(
                directory,
                "update-0001",
                identity=identity,
                state={"tensor": torch.arange(4), "counter": 7, "rng": rng_state},
                blobs={"lane-0000.snapshot": b"snapshot"},
            )
            state, blobs, manifest = load_checkpoint(
                directory, expected_identity=identity
            )
            torch.testing.assert_close(state["tensor"], torch.arange(4))
            self.assertEqual(blobs, {"lane-0000.snapshot": b"snapshot"})
            self.assertEqual(manifest["generation"], "update-0001")
            with self.assertRaises(FileExistsError):
                save_checkpoint(
                    directory,
                    "update-0001",
                    identity=identity,
                    state={},
                )
            state_path = target / "state.pt"
            payload = bytearray(state_path.read_bytes())
            payload[-1] ^= 1
            state_path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_checkpoint(directory, expected_identity=identity)

        expected = (random.random(), float(generator.random()), float(torch.rand(())))
        random.random()
        generator.random()
        torch.rand(())
        restore_rng_state(state["rng"], generator)
        actual = (random.random(), float(generator.random()), float(torch.rand(())))
        self.assertEqual(actual, expected)

    def test_unsupported_state_never_publishes(self) -> None:
        class Unsupported:
            pass

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(TypeError, "unsupported type"):
                save_checkpoint(
                    directory,
                    "bad",
                    identity={"test": True},
                    state={"bad": Unsupported()},
                )
            self.assertFalse((Path(directory) / "bad").exists())
            self.assertFalse((Path(directory) / "latest.json").exists())

    def test_mt19937_rng_arrays_are_normalized_and_restored(self) -> None:
        generator = np.random.Generator(np.random.MT19937(41))
        captured = capture_rng_state(generator)
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(
                directory,
                "mt",
                identity={"test": "mt19937"},
                state={"rng": captured},
            )
            loaded, _, _ = load_checkpoint(
                directory, expected_identity={"test": "mt19937"}
            )
        expected = generator.integers(0, 2**31, size=8)
        restore_rng_state(loaded["rng"], generator)
        np.testing.assert_array_equal(generator.integers(0, 2**31, size=8), expected)


@unittest.skipUnless(PORTABLE.exists(), "portable integration library not built")
class AdapterResumeTests(unittest.TestCase):
    def assert_resume(self, *, exact: bool) -> None:
        vector_args = (
            {"physics_backend": "exact", "worker_path": EXACT}
            if exact
            else {"library_path": PORTABLE}
        )
        first_actions = (SemanticAction.wait(7), SemanticAction.weak(0.3, 0.7))
        resumed_actions = (SemanticAction.strong(0.6, 0.5), SemanticAction.wait(11))
        with PaddedVectorEnv(2, **vector_args) as vector:
            adapter = MacroVectorAdapter(vector, encoder=TeacherStateEncoder())
            adapter.reset()
            adapter.step(first_actions)
            checkpoint = adapter.checkpoint()
            expected = adapter.step(resumed_actions)
            expected_hashes = vector.state_hash()

        packed, blobs = pack_adapter_checkpoint(checkpoint)
        identity = {
            "schema": TEACHER_V1.sha256,
            "action": ActionSpec().sha256,
            "backend": "exact" if exact else "portable",
        }
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(
                directory,
                "resume",
                identity=identity,
                state={"adapter": packed, "recurrent_state": torch.zeros(1, 2, 4)},
                blobs=blobs,
            )
            state, loaded_blobs, _ = load_checkpoint(
                directory, expected_identity=identity
            )
        restored_checkpoint = unpack_adapter_checkpoint(
            state["adapter"],
            loaded_blobs,
            schema=TEACHER_V1,
            action_spec=ActionSpec(),
        )
        with PaddedVectorEnv(2, **vector_args) as vector:
            adapter = MacroVectorAdapter(vector, encoder=TeacherStateEncoder())
            adapter.reset()
            adapter.restore_checkpoint(restored_checkpoint)
            actual = adapter.step(resumed_actions)
            actual_hashes = vector.state_hash()
        self.assertEqual(actual_hashes, expected_hashes)
        self.assertEqual(
            [
                (
                    value.raw_reward,
                    value.elapsed_ticks,
                    value.terminated,
                    value.truncated,
                )
                for value in actual
            ],
            [
                (
                    value.raw_reward,
                    value.elapsed_ticks,
                    value.terminated,
                    value.truncated,
                )
                for value in expected
            ],
        )
        for left, right in zip(actual, expected):
            np.testing.assert_array_equal(
                left.transition_next_observation.body_features,
                right.transition_next_observation.body_features,
            )

    def test_portable_trajectory_resume_is_exact(self) -> None:
        self.assert_resume(exact=False)

    @unittest.skipUnless(EXACT.exists(), "exact integration worker not built")
    def test_exact_trajectory_resume_is_exact(self) -> None:
        self.assert_resume(exact=True)


if __name__ == "__main__":
    unittest.main()
