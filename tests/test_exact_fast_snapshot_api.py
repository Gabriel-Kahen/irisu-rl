from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import (  # noqa: E402
    Action,
    ExactFastCheckpoint,
    ExactSimulator,
    ExactWorkerNotFoundError,
    IrisuEnv,
    IrisuFastCheckpoint,
    NativeError,
    find_exact_worker,
)
import irisu_env.exact_ipc as exact_ipc_module  # noqa: E402


try:
    EXACT_WORKER = find_exact_worker()
except ExactWorkerNotFoundError:
    EXACT_WORKER = None


@unittest.skipUnless(
    EXACT_WORKER and os.name == "posix" and os.uname().sysname == "Linux",
    "requires the Linux exact worker",
)
class ExactFastSnapshotApiTests(unittest.TestCase):
    def make_simulator(self, **kwargs: object) -> ExactSimulator:
        return ExactSimulator(EXACT_WORKER, **kwargs)

    def make_env(self, **kwargs: object) -> IrisuEnv:
        return IrisuEnv(
            physics_backend="exact", worker_path=EXACT_WORKER, **kwargs
        )

    def test_irisu_env_rejects_unsupported_or_uninitialized_checkpoint(self) -> None:
        with IrisuEnv(physics_backend="portable") as portable:
            with self.assertRaisesRegex(NativeError, "physics_backend='exact'"):
                portable.fast_checkpoint()
        with self.make_env() as exact:
            with self.assertRaisesRegex(RuntimeError, "reset must be called"):
                exact.fast_checkpoint()

    def test_source_close_invalidates_checkpoint_and_live_branch(self) -> None:
        source = self.make_simulator()
        source.reset(51)
        checkpoint = source.fast_checkpoint()
        branch = checkpoint.branch()
        source.close()
        try:
            self.assertTrue(checkpoint.closed)
            checkpoint.close()
            with self.assertRaisesRegex(NativeError, "checkpoint is closed"):
                checkpoint.branch()
            # Cached observation bytes remain inspectable, but the next native
            # request observes the recursive worker shutdown and poisons branch.
            with self.assertRaises(NativeError):
                branch.step(0, 0.0, 0.0, 1)
            self.assertTrue(branch.closed)
        finally:
            branch.close()

    def test_irisu_env_branch_is_fully_usable_and_independently_owned(self) -> None:
        with self.make_env(render_mode="svg", diagnostic_hashes=True) as source:
            source.reset(seed=61)
            source.step(Action.wait(17))
            durable = source.clone_state()
            source_hash = source.state_hash()
            source_svg = source.render()
            checkpoint = source.fast_checkpoint()
            self.assertIsInstance(checkpoint, IrisuFastCheckpoint)
            branch = checkpoint.branch()
            try:
                self.assertIsInstance(branch, IrisuEnv)
                self.assertEqual(branch.physics_backend, "exact")
                self.assertEqual(branch.render_mode, "svg")
                self.assertTrue(branch.diagnostic_hashes)
                self.assertEqual(branch.config, source.config)
                self.assertEqual(branch.clone_state(), durable)
                self.assertEqual(branch.state_hash(), source_hash)
                self.assertEqual(branch.render(), source_svg)
                self.assertNotEqual(
                    branch.build_info["worker_pid"], source.build_info["worker_pid"]
                )
                action = Action.strong(300, 360)
                self.assertEqual(branch.step(action), source.step(action))
                source_state = source.clone_state()
                with self.assertRaisesRegex(
                    NativeError, "active exact fast checkpoint"
                ):
                    source.reset(
                        seed=62, options={"config": {"gravity_y": 101.0}}
                    )
                self.assertEqual(source.clone_state(), source_state)

                # A normal reset detaches this branch into its own fresh worker.
                observation, info = branch.reset(seed=62)
                self.assertEqual(observation["tick"], 0)
                self.assertEqual(info["seed"], 62)
                checkpoint.close()
                self.assertTrue(checkpoint.closed)
                self.assertEqual(branch.step(Action.wait())[0]["tick"], 1)
            finally:
                branch.close()

    def test_reusable_branches_preserve_hidden_and_durable_state(self) -> None:
        with self.make_simulator(
            config={"gravity_y": 117.0, "linear_damping": 0.02}
        ) as source:
            source.reset(73)
            source.send_step_padded(0, 0.0, 0.0, 19)
            source.receive_step_padded_raw()
            durable = source.clone_state()
            state_hash = source.state_hash()
            observation = source.observation()

            checkpoint = source.fast_checkpoint()
            self.assertIsInstance(checkpoint, ExactFastCheckpoint)
            self.assertEqual(source.clone_state(), durable)
            self.assertEqual(
                source.build_info()["fast_snapshot_model"],
                "linux-fork-cow-keeper",
            )

            first = checkpoint.branch()
            divergent_action = (2, 300.0, 360.0, 1)
            divergent_transition = source.step_typed(*divergent_action)
            self.assertNotEqual(source.state_hash(), state_hash)
            # A later branch must come from the frozen keeper, not the source's
            # now-advanced live world.
            second = checkpoint.branch()
            try:
                pids = {
                    source.build_info()["worker_pid"],
                    first.build_info()["worker_pid"],
                    second.build_info()["worker_pid"],
                }
                self.assertEqual(len(pids), 3)
                for branch in (first, second):
                    self.assertEqual(branch.observation(), observation)
                    self.assertEqual(branch.state_hash(), state_hash)
                    self.assertEqual(branch.clone_state(), durable)
                    self.assertEqual(branch.config(), source.config())

                with self.assertRaisesRegex(NativeError, "active branches"):
                    checkpoint.close()
                advanced_source = source.clone_state()
                with self.assertRaisesRegex(NativeError, "active exact fast checkpoint"):
                    source.reset(74)
                self.assertEqual(source.clone_state(), advanced_source)
                self.assertEqual(first.clone_state(), durable)
                self.assertEqual(second.clone_state(), durable)

                self.assertEqual(
                    first.step_typed(*divergent_action), divergent_transition
                )
                self.assertEqual(
                    second.step_typed(*divergent_action), divergent_transition
                )
                self.assertEqual(first.clone_state(), source.clone_state())
                self.assertEqual(second.clone_state(), source.clone_state())

                actions = (
                    (0, 0.0, 0.0, 7),
                    (2, 300.0, 360.0, 1),
                    (0, 0.0, 0.0, 13),
                    (1, 240.0, 350.0, 1),
                )
                for action in actions:
                    expected = source.step_typed(*action)
                    self.assertEqual(first.step_typed(*action), expected)
                    self.assertEqual(second.step_typed(*action), expected)
                self.assertEqual(first.clone_state(), source.clone_state())
                self.assertEqual(second.clone_state(), source.clone_state())

                nested = first.fast_checkpoint()
                nested_branch = nested.branch()
                try:
                    self.assertEqual(nested_branch.clone_state(), first.clone_state())
                    action = (0, 0.0, 0.0, 5)
                    self.assertEqual(
                        nested_branch.step_typed(*action), first.step_typed(*action)
                    )
                finally:
                    nested_branch.close()
                    nested.close()
            finally:
                first.close()
                second.close()
            checkpoint.close()
            self.assertTrue(checkpoint.closed)

    def test_durable_restore_detaches_branch_from_keeper(self) -> None:
        with self.make_simulator() as source:
            source.reset(91)
            source.step(0, 0.0, 0.0, 23)
            checkpoint = source.fast_checkpoint()
            branch = checkpoint.branch()
            try:
                durable = branch.clone_state()
                branch.step(2, 300.0, 360.0, 1)
                branch.restore_state(durable)

                # Restore moved the simulator to a fresh replay-built worker;
                # the original fork child has exited, so its keeper is releasable.
                checkpoint.close()
                self.assertTrue(checkpoint.closed)
                action = (0, 0.0, 0.0, 11)
                self.assertEqual(branch.step_typed(*action), source.step_typed(*action))
            finally:
                branch.close()

    def test_nested_branches_inherit_launch_verified_provenance(self) -> None:
        with self.make_simulator() as source:
            source.reset(101)
            source.step(0, 0.0, 0.0, 37)
            expected_snapshot = source.clone_state()
            expected_hash = source.state_hash()
            checkpoint = source.fast_checkpoint()
            capture = exact_ipc_module._capture_mapped_exact_library
            with mock.patch.object(
                exact_ipc_module,
                "_capture_mapped_exact_library",
                wraps=capture,
            ) as capture_mock:
                branch = checkpoint.branch()
                nested = None
                nested_branch = None
                try:
                    self.assertEqual(capture_mock.call_count, 0)
                    self.assertEqual(branch.clone_state(), expected_snapshot)
                    self.assertEqual(branch.state_hash(), expected_hash)
                    branch.exact_library_provenance()
                    self.assertEqual(capture_mock.call_count, 1)

                    nested = branch.fast_checkpoint()
                    nested_branch = nested.branch()
                    self.assertEqual(capture_mock.call_count, 1)
                    self.assertEqual(nested_branch.clone_state(), expected_snapshot)
                    self.assertEqual(nested_branch.state_hash(), expected_hash)
                finally:
                    if nested_branch is not None:
                        nested_branch.close()
                    if nested is not None:
                        nested.close()
                    branch.close()
            checkpoint.close()


if __name__ == "__main__":
    unittest.main()
