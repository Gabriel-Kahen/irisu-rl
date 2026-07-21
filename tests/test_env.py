from __future__ import annotations

import os
import struct
import subprocess
import sys
import textwrap
import threading
import time
import unittest
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import (  # noqa: E402
    Action,
    IrisuEnv,
    NativeError,
    NativeSimulator,
    PADDED_BODY_CAPACITY,
    PaddedVectorEnv,
    Provenance,
    SyncVectorEnv,
    ThreadVectorEnv,
    find_library,
    load_profile,
)


try:
    LIBRARY = find_library()
except NativeError:
    LIBRARY = None


@unittest.skipIf(LIBRARY is None, "build the native shared library before environment tests")
class EnvironmentTests(unittest.TestCase):
    def make_env(self) -> IrisuEnv:
        return IrisuEnv(library_path=LIBRARY)

    def test_same_seed_action_trace_is_identical(self) -> None:
        actions = [
            Action.wait(25),
            Action.weak(240, 350),
            Action.wait(50),
            Action.strong(360, 320),
            Action.wait(100),
        ]
        with self.make_env() as first, self.make_env() as second:
            first_reset = first.reset(seed=42)
            second_reset = second.reset(seed=42)
            self.assertEqual(first_reset, second_reset)
            for action in actions:
                self.assertEqual(first.step(action), second.step(action))
                self.assertEqual(first.state_hash(), second.state_hash())

    def test_snapshot_restore_reproduces_future(self) -> None:
        with self.make_env() as env:
            env.reset(seed=99)
            env.step(Action.wait(80))
            snapshot = env.clone_state()
            before = env.state_hash()
            actions = (Action.strong(300, 350), Action.wait(50))
            first_future = [env.step(action) for action in actions]
            future_hash = env.state_hash()

            restored = env.restore_state(snapshot)
            self.assertNotIn("state_hash", restored)
            self.assertEqual(env.state_hash(), before)
            self.assertEqual([env.step(action) for action in actions], first_future)
            self.assertEqual(env.state_hash(), future_hash)

            stable = env.state_hash()
            with self.assertRaises(NativeError):
                env.restore_state(snapshot[:-1])
            self.assertEqual(env.state_hash(), stable)

    def test_invalid_actions_are_explicit_and_do_not_wrap(self) -> None:
        with self.make_env() as env:
            env.reset(seed=5)
            before = env.state_hash()
            with self.assertRaises(ValueError):
                env.step({"kind": "teleport"})
            self.assertEqual(env.state_hash(), before)

            observation, reward, terminated, truncated, info = env.step(Action.weak(-1, 250))
            self.assertEqual(reward, 0)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["invalid_action"])
            self.assertIn("invalid_action", {event["kind_name"] for event in info["events"]})
            self.assertNotIn("projectile", {body["kind"] for body in observation["bodies"]})

            zero_wait = env.step(Action.wait(0))
            self.assertTrue(zero_wait[-1]["invalid_action"])

    def test_vector_environments_are_independent(self) -> None:
        with SyncVectorEnv(2, library_path=LIBRARY) as vector:
            observations, _ = vector.reset(seed=[7, 7])
            self.assertEqual(observations[0], observations[1])
            initial = vector.clone_state()
            self.assertEqual(initial[0], initial[1])

            observations, _, _, _, _ = vector.step([Action.weak(200, 300), Action.wait()])
            self.assertNotEqual(vector.state_hash()[0], vector.state_hash()[1])
            self.assertIn("projectile", {body["kind"] for body in observations[0]["bodies"]})
            self.assertNotIn("projectile", {body["kind"] for body in observations[1]["bodies"]})

            vector.restore_state(initial)
            self.assertEqual(vector.state_hash()[0], vector.state_hash()[1])
            vector.step([Action.wait(10), Action.wait(10)])
            self.assertEqual(vector.state_hash()[0], vector.state_hash()[1])

    def test_thread_vector_matches_sequential_vector(self) -> None:
        actions = [Action.wait(3), Action.weak(240, 350), Action.wait(2)]
        with SyncVectorEnv(3, library_path=LIBRARY) as sequential, ThreadVectorEnv(
            3, library_path=LIBRARY
        ) as threaded:
            self.assertEqual(sequential.reset(seed=50), threaded.reset(seed=50))
            for offset in range(5):
                rotated = actions[offset % 3 :] + actions[: offset % 3]
                self.assertEqual(sequential.step(rotated), threaded.step(rotated))
                self.assertEqual(sequential.state_hash(), threaded.state_hash())
            snapshots = sequential.clone_state()
            self.assertEqual(threaded.restore_state(snapshots), sequential.restore_state(snapshots))

    def test_vector_snapshot_restore_is_transactional(self) -> None:
        for vector_type in (SyncVectorEnv, ThreadVectorEnv, PaddedVectorEnv):
            with self.subTest(vector_type=vector_type.__name__), vector_type(
                2, library_path=LIBRARY
            ) as vector:
                vector.reset(seed=[41, 42])
                target = vector.clone_state()
                vector.step([Action.strong(300, 380), Action.wait(5)])
                before = vector.clone_state()
                before_hashes = vector.state_hash()
                with self.assertRaises(NativeError):
                    vector.restore_state((target[0], target[1][:-1]))
                self.assertEqual(vector.clone_state(), before)
                self.assertEqual(vector.state_hash(), before_hashes)

    def test_vector_reset_seed_validation_is_atomic(self) -> None:
        for vector_type in (SyncVectorEnv, ThreadVectorEnv, PaddedVectorEnv):
            with self.subTest(vector_type=vector_type.__name__), vector_type(
                2, library_path=LIBRARY
            ) as vector:
                vector.reset(seed=[41, 42])
                before = vector.clone_state()
                before_hashes = vector.state_hash()
                with self.assertRaises(ValueError):
                    vector.reset(seed=0xFFFFFFFFFFFFFFFF)
                self.assertEqual(vector.clone_state(), before)
                self.assertEqual(vector.state_hash(), before_hashes)
                with self.assertRaises(TypeError):
                    vector.reset(seed=[7, "invalid"])
                self.assertEqual(vector.clone_state(), before)
                self.assertEqual(vector.state_hash(), before_hashes)

    def test_vector_action_validation_is_atomic(self) -> None:
        for vector_type in (SyncVectorEnv, ThreadVectorEnv, PaddedVectorEnv):
            with self.subTest(vector_type=vector_type.__name__), vector_type(
                2, library_path=LIBRARY
            ) as vector:
                vector.reset(seed=[41, 42])
                before = vector.clone_state()
                before_hashes = vector.state_hash()
                with self.assertRaises(ValueError):
                    vector.step([Action.wait(), Action.wait(-1)])
                self.assertEqual(vector.clone_state(), before)
                self.assertEqual(vector.state_hash(), before_hashes)

    def test_padded_abi_matches_canonical_json_exactly(self) -> None:
        actions = [
            Action.wait(30),
            Action.weak(240, 350),
            Action.wait(4),
            Action.strong(360, 320),
        ]
        with NativeSimulator(LIBRARY) as canonical, NativeSimulator(LIBRARY) as padded:
            expected = canonical.reset(71)
            observation = padded.reset_padded(71)
            self.assertEqual(observation.to_dict(), expected)
            self.assertEqual(len(observation.bodies), PADDED_BODY_CAPACITY)
            self.assertFalse(observation.body_mask(PADDED_BODY_CAPACITY - 1))
            for index, action in enumerate(actions):
                result = canonical.step(
                    int(action.kind), action.cursor_x, action.cursor_y, action.wait_ticks
                )
                transition, events = padded.step_padded(
                    int(action.kind), action.cursor_x, action.cursor_y, action.wait_ticks
                )
                self.assertEqual(transition.observation.to_dict(), canonical.observation())
                self.assertEqual([event.to_dict() for event in events], result["events"])
                self.assertEqual(transition.diagnostics(), result["diagnostics"])
                self.assertEqual(int(transition.reward), result["reward"])
                self.assertEqual(bool(transition.terminated), result["terminated"])
                self.assertEqual(bool(transition.truncated), result["truncated"])
                self.assertEqual(padded.state_hash(), canonical.state_hash())
                if index == 0:
                    self.assertTrue(transition.observation.body_mask(0))

            _, lazy_events = padded.step_padded(1, -1.0, 250.0, 1)
            self.assertGreater(len(lazy_events), 0)
            padded.step_padded(0, 0.0, 0.0, 1)
            with self.assertRaisesRegex(NativeError, "events expired"):
                lazy_events[0]

    def test_padded_vector_is_independent_and_snapshot_complete(self) -> None:
        with PaddedVectorEnv(3, library_path=LIBRARY) as vector:
            observations, infos = vector.reset(seed=[17, 17, 17])
            self.assertEqual([value.to_dict() for value in observations[1:]],
                             [observations[0].to_dict()] * 2)
            self.assertEqual(len({info["config_hash"] for info in infos}), 1)
            snapshots = vector.clone_state()
            result = vector.step(
                [Action.weak(220, 350), Action.wait(3), Action.wait(3)]
            )
            invalid = vector.step(
                [Action.weak(-1, 250), Action.wait(), Action.wait()]
            )
            self.assertTrue(invalid[-1][0]["invalid_action"])
            self.assertEqual(invalid[-1][0]["events"][0].kind, 0)
            self.assertNotEqual(vector.state_hash()[0], vector.state_hash()[1])
            self.assertEqual(vector.state_hash()[1], vector.state_hash()[2])
            self.assertEqual(result[0][1].to_dict(), result[0][2].to_dict())
            restored = vector.restore_state(snapshots)
            self.assertEqual(len(set(vector.state_hash())), 1)
            self.assertEqual(restored[0].to_dict(), observations[0].to_dict())

    def test_padded_batch_buffer_reuse_tracks_current_handles(self) -> None:
        with NativeSimulator(LIBRARY) as first, NativeSimulator(LIBRARY) as second:
            first.reset_padded(10)
            second.reset_padded(11)
            _, buffers = NativeSimulator.step_padded_batch(
                [first, second], [(0, 0.0, 0.0, 1)] * 2
            )
            results, buffers = NativeSimulator.step_padded_batch(
                [first], [(0, 0.0, 0.0, 1)], buffers
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(buffers["count"], 1)
            with self.assertRaisesRegex(ValueError, "workers"):
                NativeSimulator.step_padded_batch(
                    [first], [(0, 0.0, 0.0, 1)], buffers, workers=0
                )
            first.close()
            with self.assertRaisesRegex(NativeError, "closed"):
                NativeSimulator.step_padded_batch(
                    [first], [(0, 0.0, 0.0, 1)], buffers
                )

    def test_native_typed_arguments_reject_ctypes_wrapping_atomically(self) -> None:
        with NativeSimulator(LIBRARY) as first, NativeSimulator(LIBRARY) as second:
            first.reset_padded(10)
            second.reset_padded(11)
            for operation in (
                lambda: first.step(2**32, 0.0, 0.0, 1),
                lambda: first.step_padded(0, 0.0, 0.0, -1),
                lambda: first.reset(-1),
                lambda: first.reset_padded(2**64),
            ):
                before = first.state_hash()
                with self.assertRaises((TypeError, ValueError)):
                    operation()
                self.assertEqual(first.state_hash(), before)

            for action in (
                (-1, 0.0, 0.0, 1),
                (2**32, 0.0, 0.0, 1),
                (0, 0.0, 0.0, -1),
                (0, 0.0, 0.0, 2**32),
            ):
                before = (first.state_hash(), second.state_hash())
                with self.assertRaises((TypeError, ValueError)):
                    NativeSimulator.step_padded_batch(
                        [first, second], [action, (0, 0.0, 0.0, 1)]
                    )
                self.assertEqual((first.state_hash(), second.state_hash()), before)

    def test_padded_batch_replaces_tampered_undersized_buffers(self) -> None:
        with NativeSimulator(LIBRARY) as first, NativeSimulator(LIBRARY) as second:
            first.reset_padded(20)
            second.reset_padded(21)
            _, buffers = NativeSimulator.step_padded_batch(
                [first, second], [(0, 0.0, 0.0, 1)] * 2
            )
            buffers["handles"][0] = None
            buffers["handles"][1] = None
            results, buffers = NativeSimulator.step_padded_batch(
                [first, second], [(0, 0.0, 0.0, 1)] * 2, buffers
            )
            self.assertEqual(len(results), 2)
            undersized_type = buffers["actions"]._type_ * 1
            buffers["actions"] = undersized_type()
            results, repaired = NativeSimulator.step_padded_batch(
                [first, second], [(0, 0.0, 0.0, 1)] * 2, buffers
            )
            self.assertEqual(len(results), 2)
            self.assertIsNot(repaired, buffers)
            self.assertEqual(len(repaired["actions"]), 2)

    def test_padded_vector_fast_path_stays_atomic_and_repairs_buffers(self) -> None:
        actions = [Action.wait(2), Action.weak(320.0, 240.0)]
        with PaddedVectorEnv(2, library_path=LIBRARY) as vector:
            vector.reset(seed=[20, 21])
            before_invalid = vector.state_hash()
            with self.assertRaisesRegex(ValueError, "wait_ticks"):
                vector.step([Action.wait(-1), Action.wait()])
            self.assertEqual(vector.state_hash(), before_invalid)

            vector.step(actions)
            snapshot = vector.clone_state()
            expected = vector.step(actions)
            expected_observations = [value.to_dict() for value in expected[0]]
            expected_hashes = vector.state_hash()
            vector.restore_state(snapshot)

            buffers = vector._batch_buffers
            self.assertIsNotNone(buffers)
            assert buffers is not None
            undersized_type = buffers["actions"]._type_ * 1
            buffers["actions"] = undersized_type()
            actual = vector.step(actions)
            self.assertIsNot(vector._batch_buffers, buffers)
            self.assertEqual(len(vector._batch_buffers["actions"]), 2)
            self.assertEqual([value.to_dict() for value in actual[0]], expected_observations)
            self.assertEqual(actual[1:4], expected[1:4])
            self.assertEqual(vector.state_hash(), expected_hashes)

    def test_padded_vector_fast_path_rejects_inconsistent_sequences_and_topology(self) -> None:
        class ShortSequence(Sequence[object]):
            def __init__(self, value: object) -> None:
                self.value = value

            def __len__(self) -> int:
                return 2

            def __getitem__(self, index: int) -> object:
                if index == 0:
                    return self.value
                raise IndexError(index)

        with PaddedVectorEnv(2, library_path=LIBRARY) as vector:
            vector.reset(seed=[20, 21])
            before = vector.state_hash()
            with self.assertRaisesRegex(ValueError, "exactly 2"):
                vector.step(ShortSequence(Action.wait()))
            with self.assertRaisesRegex(ValueError, "exactly 2"):
                vector.reset(seed=ShortSequence(20))
            snapshots = vector.clone_state()
            with self.assertRaisesRegex(ValueError, "exactly 2"):
                vector.restore_state(ShortSequence(snapshots[0]))
            self.assertEqual(vector.state_hash(), before)
            with self.assertRaises(AttributeError):
                vector.envs = vector.envs[::-1]
            with self.assertRaises(AttributeError):
                vector.num_envs = 3
            with self.assertRaises(AttributeError):
                vector.workers = 0
            self.assertEqual(vector.state_hash(), before)

    def test_native_concurrent_step_and_close_are_serialized(self) -> None:
        for seed in range(20):
            simulator = NativeSimulator(
                LIBRARY, config={"max_episode_ticks": 100_000}
            )
            simulator.reset_padded(seed)
            with ThreadPoolExecutor(max_workers=2) as executor:
                step = executor.submit(simulator.step_padded, 0, 0.0, 0.0, 100_000)
                close = executor.submit(simulator.close)
                try:
                    step.result()
                except NativeError as error:
                    self.assertIn("closed", str(error))
                close.result()
            self.assertTrue(simulator.closed)
            simulator.close()

    def test_padded_vector_closes_partial_construction(self) -> None:
        import irisu_env.padded as padded_module

        created: list[NativeSimulator] = []

        def create(*args: object, **kwargs: object) -> NativeSimulator:
            if created:
                raise RuntimeError("injected constructor failure")
            simulator = NativeSimulator(*args, **kwargs)
            created.append(simulator)
            return simulator

        with mock.patch.object(padded_module, "NativeSimulator", side_effect=create):
            with self.assertRaisesRegex(RuntimeError, "injected constructor failure"):
                PaddedVectorEnv(2, library_path=LIBRARY)
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].closed)

    def test_thread_vector_drains_sibling_futures_before_raising(self) -> None:
        with ThreadVectorEnv(2, library_path=LIBRARY) as vector:
            vector.reset(seed=30)
            completed = threading.Event()
            second_step = vector.envs[1].step

            def fail(_: object) -> None:
                raise RuntimeError("lane zero")

            def slow(action: object) -> object:
                time.sleep(0.03)
                result = second_step(action)
                completed.set()
                return result

            vector.envs[0].step = fail  # type: ignore[method-assign]
            vector.envs[1].step = slow  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "lane zero"):
                vector.step([Action.wait(), Action.wait()])
            self.assertTrue(completed.is_set())

            del vector.envs[0].step
            del vector.envs[1].step
            vector.step([Action.wait(), Action.wait()])

    def test_svg_render_is_deterministic_and_asset_free(self) -> None:
        with self.make_env() as env:
            env.reset(seed=12)
            env.step(Action.wait())
            first = env.render("svg")
            self.assertEqual(first, env.render())
            root = ET.fromstring(first)
            self.assertEqual(root.tag, "{http://www.w3.org/2000/svg}svg")
            self.assertNotIn("<image", first)
            self.assertNotIn("href=", first)
            self.assertIn("data-id=", first)
            env.step(Action.wait())
            self.assertNotEqual(first, env.render())

    def test_public_method_shapes(self) -> None:
        with self.make_env() as env:
            reset_result = env.reset(seed=1, options={})
            self.assertEqual(len(reset_result), 2)
            step_result = env.step({"kind": 0, "wait_ticks": 1})
            self.assertEqual(len(step_result), 5)
            body = step_result[0]["bodies"][0]
            self.assertIn("remaining_lifetime", body)
            self.assertIn("rot_timer", body)
            self.assertNotIn("state_ticks", body)
            self.assertIsInstance(env.clone_state(), bytes)
            self.assertIsInstance(env.state_hash(), int)

        with IrisuEnv(library_path=LIBRARY, diagnostic_hashes=True) as diagnostic:
            observation, reset_info = diagnostic.reset(seed=1)
            self.assertNotIn("state_hash", observation)
            self.assertEqual(reset_info["state_hash"], diagnostic.state_hash())
            step_info = diagnostic.step(Action.wait())[-1]
            self.assertEqual(step_info["state_hash"], diagnostic.state_hash())

    def test_gymnasium_spaces_and_checker_when_available(self) -> None:
        try:
            from gymnasium.utils.env_checker import check_env
        except ImportError:
            self.skipTest("Gymnasium optional dependency is not installed")

        with self.make_env() as env:
            self.assertEqual(env.action_space["kind"].n, 4)
            self.assertTrue(env.action_space["kind"].contains(3))
            self.assertFalse(env.action_space["kind"].contains(4))
            self.assertEqual(float(env.action_space["cursor_x"].high), 640.0)
            self.assertEqual(float(env.action_space["cursor_y"].high), 480.0)
            observation, _ = env.reset(seed=123)
            self.assertTrue(env.observation_space.contains(observation))
            observation = env.step(Action.wait())[0]
            self.assertTrue(observation["bodies"])
            self.assertTrue(env.observation_space.contains(observation))
            sampled = env.action_space.sample()
            observation = env.step(sampled)[0]
            self.assertTrue(env.observation_space.contains(observation))
            check_env(env, skip_render_check=True)

    def test_config_overrides_metadata_and_snapshot_identity(self) -> None:
        with self.make_env() as nominal:
            _, nominal_info = nominal.reset(seed=1)
            nominal_snapshot = nominal.clone_state()

        overrides = {
            "gravity_y": 100.0,
            "linear_damping": 0.0,
            "piece_sizes": (28.0, 44.0, 60.0),
            "max_episode_ticks": 25,
        }
        with IrisuEnv(library_path=LIBRARY, config=overrides) as env:
            observation, info = env.reset(seed=1)
            self.assertNotEqual(info["config_hash"], nominal_info["config_hash"])
            self.assertNotIn("config_hash", observation)
            for hidden in (
                "finish_call_count",
                "terminal_metadata_recorded",
                "recorded_final_score",
                "latest_final_score",
            ):
                self.assertNotIn(hidden, observation)
            self.assertEqual(env.config["gravity_y"], 100.0)
            self.assertEqual(
                env.config["piece_sizes"],
                [28.0, 44.0, 60.0, 60.0, 72.0, 90.0, 140.0, 5.0, 5.0, 5.0],
            )
            self.assertEqual(env.build_info["abi_version"], 1)
            self.assertEqual(env.build_info["snapshot_schema"], 7)
            self.assertEqual(env.build_info["padded_abi_version"], 1)
            self.assertTrue(env.build_info["cxx_compiler_id"])
            self.assertTrue(env.build_info["cxx_compiler_version"])
            self.assertTrue(env.build_info["cmake_build_type"])
            self.assertIn(
                env.build_info["legacy_fp_mode"], ("x87", "compiler-default")
            )
            self.assertIn(
                env.build_info["fp_environment"],
                ("nearest,x87-pc53", "nearest"),
            )
            self.assertTrue(env.build_info["system_processor"])
            self.assertEqual(
                env.build_info["pointer_bits"], struct.calcsize("P") * 8
            )
            self.assertEqual(env.build_info["seed_bits"], 32)
            self.assertIn("SVN r58", env.build_info["physics"])
            step_info = env.step(Action.wait())[-1]
            self.assertEqual(step_info["config_hash"], info["config_hash"])
            self.assertEqual(
                step_info["diagnostics"]["config_hash"], info["config_hash"]
            )
            stable = env.state_hash()
            with self.assertRaises(NativeError):
                env.restore_state(nominal_snapshot)
            self.assertEqual(env.state_hash(), stable)

            _, reset_info = env.reset(seed=2, options={"config": {"gravity_y": 250.0}})
            self.assertEqual(env.config["gravity_y"], 250.0)
            self.assertNotEqual(reset_info["config_hash"], info["config_hash"])

        with self.assertRaises(NativeError):
            IrisuEnv(library_path=LIBRARY, config={"not_a_parameter": 1})

    def test_reset_config_and_seed_validation_is_atomic(self) -> None:
        with self.make_env() as env:
            env.reset(seed=5)
            stable_hash = env.state_hash()
            stable_config = env.config

            with self.assertRaises(ValueError):
                env.reset(seed=-1, options={"config": {"gravity_y": 123.0}})
            self.assertEqual(env.state_hash(), stable_hash)
            self.assertEqual(env.config, stable_config)

            with self.assertRaises(ValueError):
                env.reset(seed=1 << 32)
            self.assertEqual(env.state_hash(), stable_hash)
            self.assertEqual(env.config, stable_config)

            with self.assertRaises(NativeError):
                env.reset(seed=6, options={"config": {"world_min_x": 700.0}})
            self.assertEqual(env.state_hash(), stable_hash)
            self.assertEqual(env.config, stable_config)

    def test_python_rejects_inexact_integer_overrides(self) -> None:
        with self.assertRaises(ValueError):
            IrisuEnv(
                library_path=LIBRARY,
                config={"gravity_y\0ignored": 123.0},
            )
        with self.assertRaises(ValueError):
            IrisuEnv(
                library_path=LIBRARY,
                config={"max_episode_ticks": 2**53 + 1},
            )
        with IrisuEnv(
            library_path=LIBRARY,
            config={"max_episode_ticks": 2**53},
        ) as env:
            self.assertEqual(env.config["max_episode_ticks"], 2**53)

    def test_invalid_config_and_copy_contracts_do_not_abort_subprocess(self) -> None:
        script = textwrap.dedent(
            """
            import copy
            import ctypes
            import os

            from irisu_env import (
                IrisuEnv,
                NativeError,
                NativeSimulator,
                SyncVectorEnv,
                ThreadVectorEnv,
            )

            path = os.environ["IRISU_CLONE_LIBRARY"]
            library = ctypes.CDLL(path)

            class Override(ctypes.Structure):
                _fields_ = [("key", ctypes.c_char_p), ("value", ctypes.c_double)]

            library.irisu_create.argtypes = []
            library.irisu_create.restype = ctypes.c_void_p
            library.irisu_destroy.argtypes = [ctypes.c_void_p]
            library.irisu_configure.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(Override), ctypes.c_size_t
            ]
            library.irisu_configure.restype = ctypes.c_int
            library.irisu_config_hash.argtypes = [ctypes.c_void_p]
            library.irisu_config_hash.restype = ctypes.c_uint64
            library.irisu_reset.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
            library.irisu_reset.restype = ctypes.c_int
            library.irisu_state_hash.argtypes = [ctypes.c_void_p]
            library.irisu_state_hash.restype = ctypes.c_uint64

            handle = library.irisu_create()
            assert handle
            assert library.irisu_reset(handle, 7) == 1
            seed_state = library.irisu_state_hash(handle)
            assert library.irisu_reset(handle, 1 << 32) == 0
            assert library.irisu_state_hash(handle) == seed_state
            baseline = library.irisu_config_hash(handle)
            for key, value in (
                (b"world_min_x", 700.0),
                (b"field_top_width", -1.0),
                (b"field_bottom_height", 0.0),
                (b"piece_life_ticks", 0.0),
                (b"maximum_level", 101.0),
                (b"projectile_restitution", 1.01),
                (b"piece_sizes[0]", float.fromhex("0x1.fffffep+127")),
            ):
                override = Override(key, value)
                assert library.irisu_configure(handle, ctypes.byref(override), 1) == 0
                assert library.irisu_config_hash(handle) == baseline
            library.irisu_destroy(handle)

            for config in (
                {"world_min_x": 700.0},
                {"field_top_width": -1.0},
                {"piece_life_ticks": 0},
                {"maximum_level": 101},
            ):
                try:
                    IrisuEnv(library_path=path, config=config)
                except (NativeError, ValueError):
                    pass
                else:
                    raise AssertionError(f"invalid config was accepted: {config}")

            owned = [
                NativeSimulator(path),
                IrisuEnv(library_path=path),
                SyncVectorEnv(1, library_path=path),
                ThreadVectorEnv(1, library_path=path),
            ]
            try:
                for value in owned:
                    for operation in (copy.copy, copy.deepcopy):
                        try:
                            operation(value)
                        except TypeError:
                            pass
                        else:
                            raise AssertionError(f"copy unexpectedly succeeded: {value!r}")
            finally:
                for value in reversed(owned):
                    value.close()
            print("safe")
            """
        )
        environment = os.environ.copy()
        environment["IRISU_CLONE_LIBRARY"] = str(LIBRARY)
        environment["PYTHONPATH"] = str(ROOT / "python")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("safe", result.stdout)


class MechanicsProfileTests(unittest.TestCase):
    def test_provisional_runtime_mapping_is_quarantined(self) -> None:
        profile = load_profile(ROOT / "configs" / "v2.03-normal.toml")
        values = profile.implementation_mapping()

        self.assertEqual(profile.mechanic("normal.initial_board_counts").value, (10, 10))
        self.assertEqual(profile.mechanic("normal.initial_board_y").value, (200, 60))
        self.assertEqual(profile.mechanic("normal.initial_actor_passes").value, 1)
        self.assertEqual(values["gravity_y"], 160.0)
        self.assertEqual(values["world_magnification"], 10.0)
        self.assertEqual(values["spawn_interval_ticks"], 100)
        self.assertEqual(values["initial_rotten_count"], 10)
        self.assertEqual(values["initial_falling_count"], 10)
        self.assertEqual(values["initial_rotten_y"], 200.0)
        self.assertEqual(values["initial_falling_y"], 60.0)
        self.assertEqual(values["size_score_values"], (20, 28, 40))
        self.assertNotIn("a", values)
        required = {
            "gravity_y",
            "world_magnification",
            "spawn_interval_ticks",
            "initial_rotten_count",
            "initial_falling_count",
            "initial_rotten_y",
            "initial_falling_y",
            "size_score_values",
            "piece_life_ticks",
            "projectile_life_ticks",
            "weak_projectile_vy",
            "strong_projectile_vy",
            "cleanup_margin_x",
        }
        self.assertTrue(required.issubset(values))
        for entry in profile.implementation_parameters:
            if entry.provenance in (Provenance.PLACEHOLDER, Provenance.INFERRED):
                self.assertFalse(entry.validating_experiments, entry.key)
            elif entry.provenance is Provenance.BINARY_DERIVED:
                self.assertTrue(entry.validating_experiments, entry.key)

    @unittest.skipIf(LIBRARY is None, "build the native shared library before mapping test")
    def test_profile_implementation_values_match_native_defaults(self) -> None:
        profile = load_profile(ROOT / "configs" / "v2.03-normal.toml")
        with IrisuEnv(library_path=LIBRARY) as env:
            native = env.config
        for key, expected in profile.implementation_mapping().items():
            actual = native[key]
            if isinstance(expected, tuple):
                actual = tuple(actual)
            self.assertEqual(actual, expected, key)


if __name__ == "__main__":
    unittest.main()
