from __future__ import annotations

import unittest
from pathlib import Path

from irisu_env import Action, NativeError, PaddedVectorEnv


ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT / "build-physics-integration-portable" / "libirisu_clone.so"


@unittest.skipUnless(PORTABLE.is_file(), "build the portable native library")
class PaddedSubsetSnapshotTests(unittest.TestCase):
    def test_subset_validation_is_atomic_and_preserves_supplied_order(self) -> None:
        with PaddedVectorEnv(3, library_path=PORTABLE) as vector:
            vector.reset(seed=[41, 42, 43])
            snapshots = vector.clone_state()
            hashes = vector.state_hash()
            configs = vector.config_hash_many((2, 0))

            self.assertEqual(
                vector.clone_state_many((2, 0)), (snapshots[2], snapshots[0])
            )
            self.assertEqual(vector.state_hash_many((2, 0)), (hashes[2], hashes[0]))
            self.assertEqual(configs, (configs[0], configs[0]))
            self.assertEqual(vector.clone_state_many(()), ())
            self.assertEqual(vector.state_hash_many(()), ())
            self.assertEqual(vector.restore_many((), ()), [])

            invalid_calls = (
                lambda: vector.clone_state_many((True,)),
                lambda: vector.state_hash_many((0, 0)),
                lambda: vector.config_hash_many((3,)),
                lambda: vector.restore_many((0,), ()),
                lambda: vector.restore_many((0,), (bytearray(snapshots[0]),)),
            )
            for call in invalid_calls:
                with (
                    self.subTest(call=call),
                    self.assertRaises((TypeError, ValueError, IndexError)),
                ):
                    call()
                self.assertEqual(vector.clone_state(), snapshots)
                self.assertEqual(vector.state_hash(), hashes)

    def test_restore_many_changes_only_selected_lanes(self) -> None:
        with PaddedVectorEnv(3, library_path=PORTABLE) as vector:
            vector.reset(seed=[51, 52, 53])
            target = vector.clone_state()
            target_hashes = vector.state_hash()
            vector.step([Action.wait(3), Action.wait(5), Action.wait(7)])
            advanced = vector.clone_state()
            advanced_hashes = vector.state_hash()

            observations = vector.restore_many((2, 0), (target[2], target[0]))
            self.assertEqual([int(value.tick) for value in observations], [0, 0])
            self.assertEqual(
                vector.state_hash(),
                (target_hashes[0], advanced_hashes[1], target_hashes[2]),
            )
            current = vector.clone_state()
            self.assertEqual(current[0], target[0])
            self.assertEqual(current[1], advanced[1])
            self.assertEqual(current[2], target[2])

    def test_partial_subset_failure_rolls_back_every_target_lane(self) -> None:
        with PaddedVectorEnv(3, library_path=PORTABLE) as vector:
            vector.reset(seed=[61, 62, 63])
            target = vector.clone_state()
            vector.step([Action.wait(2), Action.wait(4), Action.wait(6)])
            before = vector.clone_state()
            before_hashes = vector.state_hash()
            corrupted = target[2][:-1]

            with self.assertRaises(NativeError):
                vector.restore_many((0, 2), (target[0], corrupted))
            self.assertEqual(vector.clone_state(), before)
            self.assertEqual(vector.state_hash(), before_hashes)

            # A successful rollback leaves the vector usable and deterministic.
            observations = vector.restore_many((0, 2), (target[0], target[2]))
            self.assertEqual([int(value.tick) for value in observations], [0, 0])


if __name__ == "__main__":
    unittest.main()
