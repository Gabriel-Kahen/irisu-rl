from __future__ import annotations

import ctypes
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import (  # noqa: E402
    Action,
    ExactPaddedObservation,
    ExactProtocolError,
    ExactWorkerClient,
    ExactWorkerError,
    ExactWorkerInfo,
    ExactWorkerNotFoundError,
    IrisuEnv,
    NativeError,
    PaddedVectorEnv,
    SyncVectorEnv,
    ThreadVectorEnv,
    find_exact_worker,
    snapshot_config_hash,
)
import irisu_env.padded as padded_module  # noqa: E402
import irisu_env.exact_ipc as exact_ipc_module  # noqa: E402


try:
    EXACT_WORKER = find_exact_worker()
except ExactWorkerNotFoundError:
    EXACT_WORKER = None


class FakeExactSimulator:
    instances: list[FakeExactSimulator] = []
    active = 0
    peak = 0
    sent: list[int] = []
    received: list[int] = []
    send_failures: set[int] = set()
    receive_failures: set[int] = set()

    @classmethod
    def reset_tracking(cls) -> None:
        cls.instances = []
        cls.active = 0
        cls.peak = 0
        cls.sent = []
        cls.received = []
        cls.send_failures = set()
        cls.receive_failures = set()

    def __init__(self, *_: object, **__: object) -> None:
        self.index = len(self.instances)
        self.instances.append(self)
        self._client = self
        self._lock = None
        self._pending = False

    def config_hash(self) -> int:
        return self.index

    def send_step_padded(self, *_: object) -> None:
        type(self).sent.append(self.index)
        if self.index in type(self).send_failures:
            raise RuntimeError(f"send lane {self.index}")
        self._pending = True
        type(self).active += 1
        type(self).peak = max(type(self).peak, type(self).active)

    def receive_step_padded_raw(self) -> tuple[bytes, int]:
        if not self._pending:
            raise AssertionError("receive without send")
        self._pending = False
        type(self).active -= 1
        type(self).received.append(self.index)
        if self.index in type(self).receive_failures:
            raise RuntimeError(f"receive lane {self.index}")
        return b"", 1

    def _pending_response_fd(self) -> int:
        if not self._pending:
            raise AssertionError("response descriptor without send")
        return 1_000 + self.index

    def close(self) -> None:
        if self._pending:
            self._pending = False
            type(self).active -= 1


class OrderedFakePoll:
    def __init__(self, lane_order: list[int]) -> None:
        self._descriptors = [1_000 + index for index in lane_order]
        self._registered: set[int] = set()
        self._poll_calls = 0

    def register(self, descriptor: int, _: int) -> None:
        self._registered.add(descriptor)

    def unregister(self, descriptor: int) -> None:
        self._registered.remove(descriptor)

    def poll(self) -> list[tuple[int, int]]:
        self._poll_calls += 1
        if self._poll_calls > 2 * len(self._descriptors):
            raise AssertionError("ready descriptor was not unregistered")
        for descriptor in self._descriptors:
            if descriptor in self._registered:
                return [(descriptor, 1)]
        if self._registered:
            descriptor = min(self._registered)
            return [(descriptor, 1)]
        return []


class RegistrationOrderFakePoll:
    def __init__(self, lane_order: list[int]) -> None:
        self._lane_order = lane_order
        self._descriptors: list[int] = []
        self._registered: set[int] = set()
        self.delivered: list[int] = []

    def register(self, descriptor: int, _: int) -> None:
        self._descriptors.append(descriptor)
        self._registered.add(descriptor)

    def unregister(self, descriptor: int) -> None:
        self._registered.remove(descriptor)

    def poll(self) -> list[tuple[int, int]]:
        for lane in self._lane_order:
            descriptor = self._descriptors[lane]
            if descriptor in self._registered:
                self.delivered.append(lane)
                return [(descriptor, 1)]
        return []


class FakeNativeSimulator:
    def __init__(self, *_: object, **__: object) -> None:
        pass

    @staticmethod
    def _padded_batch_topology(
        simulators: tuple[FakeNativeSimulator, ...],
    ) -> tuple[object, tuple[FakeNativeSimulator, ...]]:
        return object(), simulators

    def config_hash(self) -> int:
        return 0

    def close(self) -> None:
        pass


def decode_fake_exact_transition(
    _: bytes, destination: padded_module.ExactPaddedTransition
) -> tuple[padded_module.ExactPaddedTransition, int]:
    destination.event_count = 0
    return destination, 1


class ExactBackendArgumentTests(unittest.TestCase):
    @staticmethod
    def worker_info(**changes: object) -> ExactWorkerInfo:
        values = {
            "protocol_version": 1,
            "pointer_bits": 32,
            "body_capacity": 196,
            "pid": 1,
            "config_hash": 1,
            "x87_control_word": 0x027F,
            "process_model": 1,
            "backend": "exact-msvc9-r58-multiworld-forward",
            "compiler": "test compiler",
            "exact_library_sha256": "1" * 64,
        }
        values.update(changes)
        return ExactWorkerInfo(**values)

    def test_backend_arguments_are_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "worker_path"):
            IrisuEnv(physics_backend="portable", worker_path="worker")
        with self.assertRaisesRegex(ValueError, "library_path"):
            IrisuEnv(
                physics_backend="exact",
                library_path="library",
                worker_path="worker",
            )
        with self.assertRaisesRegex(ValueError, "physics_backend"):
            IrisuEnv(physics_backend="approximate")
        with self.assertRaisesRegex(ValueError, "worker_path"):
            PaddedVectorEnv(1, worker_path="worker")
        with self.assertRaisesRegex(ValueError, "library_path"):
            PaddedVectorEnv(
                1,
                physics_backend="exact",
                library_path="library",
                worker_path="worker",
            )

    def test_exact_explicit_workers_exceed_default_portable_cap(self) -> None:
        cases = (
            ("exact", None, 8),
            ("exact", 5, 5),
            ("exact", 12, 12),
            ("exact", 20, 12),
            ("portable", None, 8),
            ("portable", 5, 5),
            ("portable", 12, 8),
        )
        with mock.patch.object(
            padded_module, "ExactSimulator", FakeExactSimulator
        ), mock.patch.object(
            padded_module, "NativeSimulator", FakeNativeSimulator
        ):
            for backend, workers, expected in cases:
                with self.subTest(backend=backend, workers=workers):
                    kwargs: dict[str, object] = {
                        "physics_backend": backend,
                        "workers": workers,
                    }
                    if backend == "exact":
                        kwargs["worker_path"] = "unused"
                    with PaddedVectorEnv(12, **kwargs) as vector:
                        self.assertEqual(vector.workers, expected)

    def test_exact_explicit_workers_send_more_than_eight_lanes(self) -> None:
        FakeExactSimulator.reset_tracking()
        with mock.patch.object(
            padded_module, "ExactSimulator", FakeExactSimulator
        ), mock.patch.object(
            padded_module, "_decode_exact_transition", decode_fake_exact_transition
        ), PaddedVectorEnv(
            12,
            workers=12,
            physics_backend="exact",
            worker_path="unused",
        ) as vector:
            transitions, events = vector._step_exact([(0, 0.0, 0.0, 1)] * 12)

        self.assertEqual(vector.workers, 12)
        self.assertEqual(FakeExactSimulator.peak, 12)
        self.assertEqual(FakeExactSimulator.active, 0)
        self.assertEqual(FakeExactSimulator.sent, list(range(12)))
        self.assertEqual(FakeExactSimulator.received, list(range(12)))
        self.assertEqual((len(transitions), len(events)), (12, 12))

    def test_exact_large_wave_drains_and_raises_lowest_lane_failure(self) -> None:
        FakeExactSimulator.reset_tracking()
        FakeExactSimulator.send_failures = {10}
        FakeExactSimulator.receive_failures = {3}
        with mock.patch.object(
            padded_module, "ExactSimulator", FakeExactSimulator
        ), mock.patch.object(
            padded_module, "_decode_exact_transition", decode_fake_exact_transition
        ), PaddedVectorEnv(
            12,
            workers=12,
            physics_backend="exact",
            worker_path="unused",
        ) as vector:
            with self.assertRaisesRegex(RuntimeError, "receive lane 3"):
                vector._step_exact([(0, 0.0, 0.0, 1)] * 12)

        self.assertEqual(FakeExactSimulator.active, 0)
        self.assertEqual(FakeExactSimulator.sent, list(range(12)))
        self.assertEqual(
            FakeExactSimulator.received,
            [index for index in range(12) if index != 10],
        )

    def test_exact_wave_drains_in_readiness_order_and_raises_lowest_failure(
        self,
    ) -> None:
        FakeExactSimulator.reset_tracking()
        FakeExactSimulator.receive_failures = {2, 7}
        ready_order = [7, 0, 5, 3, 1, 6, 4, 2]
        with mock.patch.object(
            padded_module, "ExactSimulator", FakeExactSimulator
        ), mock.patch.object(
            padded_module, "_decode_exact_transition", decode_fake_exact_transition
        ), mock.patch.object(
            padded_module.select, "poll", return_value=OrderedFakePoll(ready_order)
        ), PaddedVectorEnv(
            8,
            workers=8,
            physics_backend="exact",
            worker_path="unused",
        ) as vector:
            with self.assertRaisesRegex(RuntimeError, "receive lane 2"):
                vector._step_exact([(0, 0.0, 0.0, 1)] * 8)

        self.assertEqual(FakeExactSimulator.active, 0)
        self.assertEqual(FakeExactSimulator.received, ready_order)

    def test_exact_poll_construction_failure_sends_no_requests(self) -> None:
        FakeExactSimulator.reset_tracking()
        with mock.patch.object(
            padded_module, "ExactSimulator", FakeExactSimulator
        ), mock.patch.object(
            padded_module.select,
            "poll",
            side_effect=RuntimeError("poll unavailable"),
        ), PaddedVectorEnv(
            8,
            workers=8,
            physics_backend="exact",
            worker_path="unused",
        ) as vector:
            with self.assertRaisesRegex(RuntimeError, "poll unavailable"):
                vector._step_exact([(0, 0.0, 0.0, 1)] * 8)

        self.assertEqual(FakeExactSimulator.active, 0)
        self.assertEqual(FakeExactSimulator.sent, [])
        self.assertEqual(FakeExactSimulator.received, [])

    def test_worker_handshake_rejects_non_exact_backend(self) -> None:
        def hello(client: ExactWorkerClient) -> ExactWorkerInfo:
            return self.worker_info(
                pid=client._transport_pid, backend="portable-gnu-r58"
            )

        with mock.patch.object(ExactWorkerClient, "_hello", hello):
            with self.assertRaisesRegex(
                ExactProtocolError, "required exact multiworld backend"
            ):
                ExactWorkerClient(sys.executable)

    def test_worker_handshake_rejects_missing_or_placeholder_sha(self) -> None:
        invalid = ("unknown", "g" * 64, "0" * 64, "1" * 63)
        for digest in invalid:
            with self.subTest(digest=digest):
                def hello(client: ExactWorkerClient) -> ExactWorkerInfo:
                    return self.worker_info(
                        pid=client._transport_pid,
                        exact_library_sha256=digest,
                    )

                with mock.patch.object(ExactWorkerClient, "_hello", hello):
                    with self.assertRaisesRegex(
                        ExactProtocolError, "valid non-placeholder"
                    ):
                        ExactWorkerClient(sys.executable)

    def test_mapped_library_client_mount_device_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = (
                Path(directory) / "libirisu_box2d_msvc_exact_multiworld.so"
            )
            library.write_bytes(b"synthetic exact library")
            inode = library.stat().st_ino
            mapped = (
                str(library.resolve()),
                "00:21",
                inode,
                (("r--p", "00000000"), ("r-xp", "00001000")),
            )
            digest = exact_ipc_module._sha256_file(library)

            with mock.patch.object(
                exact_ipc_module, "_mapped_exact_library", return_value=mapped
            ), mock.patch.object(
                exact_ipc_module, "_mount_device", return_value="ff:ff"
            ), self.assertRaisesRegex(
                ExactProtocolError, "device disagrees with the client mount"
            ):
                exact_ipc_module._capture_mapped_exact_library(os.getpid(), digest)

    def test_worker_executable_hash_follows_handshake_and_uses_proc(self) -> None:
        calls: list[str] = []
        transport_pid: int | None = None

        def hello(client: ExactWorkerClient) -> ExactWorkerInfo:
            nonlocal transport_pid
            calls.append("hello")
            transport_pid = client._transport_pid
            return self.worker_info(pid=transport_pid)

        def executable_sha(path: Path) -> str:
            calls.append("executable")
            self.assertIsNotNone(transport_pid)
            self.assertEqual(path, Path(f"/proc/{transport_pid}/exe"))
            return "2" * 64

        def request_attestation(
            _: ExactWorkerClient, opcode: int, payload: bytes = b""
        ) -> bytes:
            calls.append("attestation")
            self.assertEqual(opcode, exact_ipc_module._EXACT_ATTESTATION)
            self.assertEqual(payload, b"")
            device = b"00:21"
            return (
                exact_ipc_module._EXACT_ATTESTATION_FIXED.pack(1, 15, 1)
                + len(device).to_bytes(2, "little")
                + device
            )

        def mapped_library(pid: int, digest: object) -> dict[str, object]:
            calls.append("library")
            self.assertEqual(pid, transport_pid)
            self.assertEqual(digest, "1" * 64)
            return {"mapped_identity": {"device": "00:21", "inode": 1}}

        with mock.patch.object(
            ExactWorkerClient, "_hello", hello
        ), mock.patch.object(
            ExactWorkerClient, "_request", request_attestation
        ), mock.patch.object(
            exact_ipc_module, "_sha256_file", executable_sha
        ), mock.patch.object(
            exact_ipc_module,
            "_capture_mapped_exact_library",
            mapped_library,
        ):
            client = ExactWorkerClient(sys.executable)
            try:
                self.assertEqual(client.executable_sha256, "2" * 64)
                self.assertEqual(
                    calls, ["hello", "attestation", "executable", "library"]
                )
            finally:
                client._abort()

    def test_fast_branch_keeper_credentials_fail_closed(self) -> None:
        keeper_pid = 1234

        class PeerSocket:
            def __init__(self, credentials: bytes) -> None:
                self.credentials = credentials

            def getsockopt(self, level: int, option: int, size: int) -> bytes:
                self.assertions = (level, option, size)
                return self.credentials

        expected = (keeper_pid, os.geteuid(), os.getegid())
        exact_ipc_module._verify_keeper_peer(
            PeerSocket(exact_ipc_module._PEER_CREDENTIALS.pack(*expected)),
            keeper_pid,
        )
        for peer in (
            (keeper_pid + 1, expected[1], expected[2]),
            (keeper_pid, expected[1] + 1, expected[2]),
            (keeper_pid, expected[1], expected[2] + 1),
        ):
            with self.subTest(peer=peer), self.assertRaisesRegex(
                ExactProtocolError, "keeper identity changed"
            ):
                exact_ipc_module._verify_keeper_peer(
                    PeerSocket(exact_ipc_module._PEER_CREDENTIALS.pack(*peer)),
                    keeper_pid,
                )
        with self.assertRaisesRegex(ExactProtocolError, "truncated"):
            exact_ipc_module._verify_keeper_peer(PeerSocket(b""), keeper_pid)

    def test_fast_branch_inherited_files_fail_closed(self) -> None:
        pid = os.getpid()
        keeper_pid = os.getppid()
        executable_identity = exact_ipc_module._file_identity(
            Path(f"/proc/{pid}/exe").stat()
        )
        library_path = Path(__file__).resolve()
        library_status = library_path.stat()
        library_identity = exact_ipc_module._file_identity(library_status)
        provenance = {
            "status": "captured",
            "path": str(library_path),
            "bytes": library_identity[2],
            "sha256": "1" * 64,
            "file_identity": {
                "device": library_identity[0],
                "inode": library_identity[1],
                "mtime_ns": library_identity[3],
                "ctime_ns": library_identity[4],
            },
        }
        exact_ipc_module._verify_inherited_branch_files(
            pid, keeper_pid, executable_identity, provenance
        )

        with self.assertRaisesRegex(ExactProtocolError, "direct keeper child"):
            exact_ipc_module._verify_inherited_branch_files(
                pid, keeper_pid + 1, executable_identity, provenance
            )
        changed_executable = (
            executable_identity[0],
            executable_identity[1] + 1,
            *executable_identity[2:],
        )
        with self.assertRaisesRegex(ExactProtocolError, "executable identity"):
            exact_ipc_module._verify_inherited_branch_files(
                pid, keeper_pid, changed_executable, provenance
            )
        changed_library = {**provenance, "bytes": library_identity[2] + 1}
        with self.assertRaisesRegex(ExactProtocolError, "library identity"):
            exact_ipc_module._verify_inherited_branch_files(
                pid, keeper_pid, executable_identity, changed_library
            )

    def test_fast_branch_hello_identity_changes_fail_closed(self) -> None:
        source = ExactWorkerClient.__new__(ExactWorkerClient)
        source.info = self.worker_info(pid=10)
        source.current_config_hash = 1
        source.executable_sha256 = "2" * 64
        process = 11
        keeper = 12
        address = b"\0test"
        response = exact_ipc_module._FAST_BRANCH_RESPONSE.pack(
            process, len(address), b"s" * 16
        ) + address
        source._request = mock.Mock(return_value=response)

        cases = {
            "config": {"current_config_hash": 2},
            "library": {
                "info": self.worker_info(
                    pid=process, exact_library_sha256="3" * 64
                )
            },
            "executable": {"executable_sha256": "3" * 64},
            "x87": {
                "info": self.worker_info(pid=process, x87_control_word=0x037F)
            },
            "compiler": {
                "info": self.worker_info(pid=process, compiler="other compiler")
            },
        }
        for name, changes in cases.items():
            with self.subTest(name=name):
                branch = mock.Mock()
                branch.current_config_hash = 1
                branch.info = self.worker_info(pid=process)
                branch.executable_sha256 = "2" * 64
                for field, value in changes.items():
                    setattr(branch, field, value)
                source._connect_fast_branch = mock.Mock(return_value=branch)
                with self.assertRaisesRegex(
                    ExactProtocolError, "fast branch identity changed"
                ):
                    source.branch_fast_checkpoint(b"t" * 16, keeper)
                branch.close.assert_called_once_with()
                source._connect_fast_branch.assert_called_once_with(
                    address,
                    b"s" * 16,
                    expected_pid=process,
                    expected_keeper_pid=keeper,
                )


@unittest.skipIf(EXACT_WORKER is None, "build the exact physics worker")
class ExactEnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs: object) -> IrisuEnv:
        return IrisuEnv(
            physics_backend="exact", worker_path=EXACT_WORKER, **kwargs
        )

    def test_transition_maps_observation_events_and_diagnostics(self) -> None:
        with self.make_env(diagnostic_hashes=True) as env, ExactWorkerClient(
            EXACT_WORKER
        ) as oracle:
            observation, info = env.reset(seed=41)
            self.assertEqual(observation, oracle.reset(41).to_dict())
            self.assertEqual(env.physics_backend, "exact")
            self.assertEqual(Path(env.worker_path), EXACT_WORKER)
            self.assertEqual(observation["tick"], 0)
            self.assertTrue(observation["bodies"])
            self.assertIn(observation["bodies"][0]["kind"], ("piece", "bonus"))
            self.assertEqual(info["config_hash"], env.config_hash())
            self.assertEqual(info["state_hash"], env.state_hash())

            observation, reward, terminated, truncated, info = env.step(
                Action.wait(3)
            )
            exact = oracle.step(wait_ticks=3)
            self.assertEqual(observation, exact.observation.to_dict())
            self.assertEqual(reward, exact.reward)
            self.assertEqual(info["events"], [
                {**event.to_dict(), "kind_name": info["events"][index]["kind_name"]}
                for index, event in enumerate(exact.events)
            ])
            self.assertEqual(info["diagnostics"], exact.diagnostics())
            self.assertEqual(observation["tick"], 3)
            self.assertIsInstance(reward, int)
            self.assertFalse(terminated)
            self.assertFalse(truncated)
            self.assertTrue(info["events"])
            self.assertTrue(
                {"tick", "sequence", "kind", "kind_name", "a", "b", "value", "detail"}
                .issubset(info["events"][0])
            )
            self.assertEqual(info["diagnostics"]["config_hash"], env.config_hash())
            self.assertEqual(info["state_hash"], env.state_hash())
            self.assertEqual(env.build_info["pointer_bits"], 32)
            self.assertEqual(env.build_info["snapshot_model"], "seed-and-action-log-replay")
            self.assertRegex(
                env.build_info["worker_executable_sha256"], r"^[0-9a-f]{64}$"
            )

    def test_action_log_snapshot_restore_reproduces_hidden_future(self) -> None:
        actions = (Action.wait(9), Action.strong(300, 360), Action.wait(7))
        future = (Action.weak(240, 350), Action.wait(11))
        with self.make_env() as env:
            env.reset(seed=73)
            for action in actions:
                env.step(action)
            snapshot = env.clone_state()
            snapshot_hash = env.state_hash()
            self.assertEqual(snapshot_config_hash(snapshot), env.config_hash())
            before_pid = env.build_info["worker_pid"]

            expected = [env.step(action) for action in future]
            expected_hash = env.state_hash()
            restored = env.restore_state(snapshot)
            self.assertEqual(restored["tick"], sum(action.wait_ticks for action in actions))
            self.assertNotEqual(env.build_info["worker_pid"], before_pid)
            self.assertEqual(env.state_hash(), snapshot_hash)
            self.assertEqual([env.step(action) for action in future], expected)
            self.assertEqual(env.state_hash(), expected_hash)

            stable_hash = env.state_hash()
            stable_observation = env.render()
            corrupted = bytearray(snapshot)
            corrupted[-1] ^= 1
            with self.assertRaisesRegex(NativeError, "checksum"):
                env.restore_state(corrupted)
            self.assertEqual(env.state_hash(), stable_hash)
            self.assertEqual(env.render(), stable_observation)

    def test_durable_snapshot_rejects_an_in_flight_step(self) -> None:
        with self.make_env() as env:
            env.reset(seed=17)
            before = env.clone_state()
            native = env._native
            native.send_step(0, 0.0, 0.0, 5)

            with self.assertRaisesRegex(NativeError, "pending exact step"):
                env.clone_state()

            transition = native.receive_step_typed()
            self.assertEqual(transition.observation.tick, 5)
            self.assertNotEqual(env.clone_state(), before)

    def test_independent_workers_have_stable_history_hashes(self) -> None:
        trace = (Action.wait(5), Action.weak(250, 350), Action.wait(2))
        with self.make_env() as first, self.make_env() as second:
            self.assertNotEqual(
                first.build_info["worker_pid"], second.build_info["worker_pid"]
            )
            first.reset(seed=91)
            second.reset(seed=91)
            self.assertEqual(first.state_hash(), second.state_hash())
            for action in trace:
                self.assertEqual(first.step(action), second.step(action))
                self.assertEqual(first.state_hash(), second.state_hash())
            self.assertEqual(first.clone_state(), second.clone_state())

    def test_config_overrides_hash_readback_and_snapshot_restore(self) -> None:
        overrides = {
            "gravity_y": 100.0,
            "linear_damping": 0.05,
            "piece_sizes": (28.0, 44.0, 60.0),
            "max_episode_ticks": 500,
        }
        with self.make_env() as nominal, self.make_env(config=overrides) as custom:
            nominal.reset(seed=31)
            observation, info = custom.reset(seed=31)
            self.assertNotEqual(info["config_hash"], nominal.config_hash())
            self.assertEqual(custom.config["gravity_y"], 100.0)
            self.assertEqual(custom.config["linear_damping"], 0.05)
            self.assertEqual(custom.config["piece_sizes"][:3], [28.0, 44.0, 60.0])
            self.assertEqual(custom.config["max_episode_ticks"], 500)
            self.assertEqual(custom.build_info["config_hash"], info["config_hash"])

            custom.step(Action.wait(8))
            snapshot = custom.clone_state()
            expected = custom.step(Action.strong(300, 360))
            expected_hash = custom.state_hash()
            self.assertEqual(custom.restore_state(snapshot)["tick"], 8)
            self.assertEqual(custom.step(Action.strong(300, 360)), expected)
            self.assertEqual(custom.state_hash(), expected_hash)

            stable_hash = custom.state_hash()
            stable_observation = custom.render()
            with self.assertRaises(NativeError):
                custom.reset(seed=32, options={"config": {"not_a_parameter": 1}})
            self.assertEqual(custom.state_hash(), stable_hash)
            self.assertEqual(custom.render(), stable_observation)

    def test_sync_and_thread_vectors_propagate_exact_backend(self) -> None:
        parameters = {
            "physics_backend": "exact",
            "worker_path": EXACT_WORKER,
        }
        with SyncVectorEnv(2, **parameters) as sequential, ThreadVectorEnv(
            2, **parameters
        ) as threaded:
            self.assertTrue(
                all(env.physics_backend == "exact" for env in sequential.envs)
            )
            self.assertEqual(sequential.reset(seed=101), threaded.reset(seed=101))
            actions = (Action.weak(220, 350), Action.wait(4))
            self.assertEqual(sequential.step(actions), threaded.step(actions))
            self.assertEqual(sequential.state_hash(), threaded.state_hash())
            snapshots = sequential.clone_state()
            self.assertEqual(
                sequential.restore_state(snapshots), threaded.restore_state(snapshots)
            )

    def test_receive_failure_poisons_advanced_worker(self) -> None:
        with self.make_env() as env:
            env.reset(seed=17)
            native = env._native
            client = native._client
            self.assertIsNotNone(client)
            native.send_step(0, 0.0, 0.0, 1)
            with mock.patch.object(
                client,
                "receive_step",
                side_effect=ExactProtocolError("synthetic decode failure"),
            ):
                with self.assertRaisesRegex(ExactProtocolError, "synthetic"):
                    native.receive_step()
            self.assertTrue(native.closed)
            self.assertTrue(client.closed)
            for operation in (
                lambda: env.step(Action.wait()),
                env.state_hash,
                env.clone_state,
                env.render,
            ):
                with self.assertRaises(NativeError):
                    operation()

    def test_malformed_frame_header_aborts_low_level_client(self) -> None:
        with ExactWorkerClient(EXACT_WORKER) as client:
            client.reset(17)
            client._begin_request(4)
            with mock.patch(
                "irisu_env.exact_ipc._read_exact", return_value=bytes(16)
            ):
                with self.assertRaisesRegex(ExactProtocolError, "frame header"):
                    client._finish_response(4)
            self.assertTrue(client.closed)

    def test_low_level_worker_allows_only_one_episode_reset(self) -> None:
        with ExactWorkerClient(EXACT_WORKER) as client:
            initial = client.reset(19)
            with self.assertRaisesRegex(
                ExactWorkerError, "one successful reset per process"
            ):
                client.reset(20)
            with self.assertRaisesRegex(
                ExactWorkerError, "configuration is immutable after reset"
            ):
                client.configure({"gravity_y": 117.0})
            self.assertEqual(client.observe(), initial)
            self.assertEqual(client.step(wait_ticks=1).observation.tick, 1)

    def test_exact_thread_vector_honors_worker_cap(self) -> None:
        with ThreadVectorEnv(
            5,
            workers=2,
            physics_backend="exact",
            worker_path=EXACT_WORKER,
        ) as vector:
            vector.reset(seed=200)
            active = 0
            peak = 0

            with ExitStack() as patches:
                for env in vector.envs:
                    original_send = env._send_exact_step
                    original_receive = env._receive_exact_step

                    def send(action: Action, original=original_send) -> None:
                        nonlocal active, peak
                        original(action)
                        active += 1
                        peak = max(peak, active)

                    def receive(original=original_receive):
                        nonlocal active
                        try:
                            return original()
                        finally:
                            active -= 1

                    patches.enter_context(mock.patch.object(env, "_send_exact_step", send))
                    patches.enter_context(
                        mock.patch.object(env, "_receive_exact_step", receive)
                    )
                observations = vector.step([Action.wait()] * 5)[0]

            self.assertEqual(peak, 2)
            self.assertEqual(active, 0)
            self.assertEqual([value["tick"] for value in observations], [1] * 5)

    def test_exact_thread_vector_drains_other_lanes_after_receive_failure(self) -> None:
        with ThreadVectorEnv(
            4,
            workers=2,
            physics_backend="exact",
            worker_path=EXACT_WORKER,
        ) as vector:
            vector.reset(seed=300)
            failed = vector.envs[0]._native
            client = failed._client
            self.assertIsNotNone(client)
            with mock.patch.object(
                client,
                "receive_step",
                side_effect=ExactProtocolError("synthetic vector decode failure"),
            ):
                with self.assertRaisesRegex(ExactProtocolError, "synthetic vector"):
                    vector.step([Action.wait()] * 4)

            self.assertTrue(failed.closed)
            self.assertTrue(client.closed)
            self.assertEqual(
                [env._native.observation()["tick"] for env in vector.envs[1:]],
                [1, 1, 1],
            )
            self.assertEqual(vector.envs[1].step(Action.wait())[0]["tick"], 2)

    def test_exact_padded_vector_matches_full_worker_contract(self) -> None:
        with PaddedVectorEnv(
            1, physics_backend="exact", worker_path=EXACT_WORKER
        ) as vector, ExactWorkerClient(EXACT_WORKER) as oracle:
            observations, infos = vector.reset(seed=41)
            expected = oracle.reset(41)
            self.assertEqual(vector.physics_backend, "exact")
            self.assertIsInstance(observations[0], ExactPaddedObservation)
            self.assertEqual(observations[0].to_dict(), expected.to_dict())
            self.assertEqual(infos[0]["config_hash"], oracle.current_config_hash)

            actions = (
                Action.wait(3),
                Action.weak(240, 350),
                Action.wait(2),
                Action.strong(360, 320),
            )
            for action in actions:
                actual = vector.step([action])
                exact = oracle.step(
                    int(action.kind),
                    action.cursor_x,
                    action.cursor_y,
                    action.wait_ticks,
                )
                self.assertEqual(actual[0][0].to_dict(), exact.observation.to_dict())
                self.assertEqual(actual[1], [exact.reward])
                self.assertEqual(actual[2], [exact.terminated])
                self.assertEqual(actual[3], [exact.truncated])
                events = [event.to_dict() for event in actual[4][0]["events"]]
                self.assertEqual(
                    [{key: value for key, value in event.items() if key != "kind_name"}
                     for event in events],
                    [event.to_dict() for event in exact.events],
                )
                self.assertEqual(
                    actual[4][0]["diagnostics"].diagnostics(), exact.diagnostics()
                )

    def test_exact_padded_ready_order_preserves_independent_lane_results(self) -> None:
        seeds = [41, 71, 91, 101, 137, 211, 307, 401]
        actions = [
            Action.wait(1),
            Action.wait(2),
            Action.weak(220, 350),
            Action.strong(360, 320),
            Action.wait(7),
            Action.weak(280, 330),
            Action.strong(320, 370),
            Action.wait(11),
        ]
        ready_order = [7, 0, 5, 3, 1, 6, 4, 2]

        with ExitStack() as stack:
            vector = stack.enter_context(
                PaddedVectorEnv(
                    8,
                    workers=8,
                    physics_backend="exact",
                    worker_path=EXACT_WORKER,
                )
            )
            oracles = [
                stack.enter_context(ExactWorkerClient(EXACT_WORKER))
                for _ in seeds
            ]
            observations, _ = vector.reset(seed=seeds)
            expected_resets = [
                oracle.reset(seed) for oracle, seed in zip(oracles, seeds)
            ]
            self.assertEqual(
                [observation.to_dict() for observation in observations],
                [observation.to_dict() for observation in expected_resets],
            )

            poller = RegistrationOrderFakePoll(ready_order)
            with mock.patch.object(
                padded_module.select, "poll", return_value=poller
            ):
                actual = vector.step(actions)
            self.assertEqual(poller.delivered, ready_order)
            expected = [
                oracle.step(
                    int(action.kind),
                    action.cursor_x,
                    action.cursor_y,
                    action.wait_ticks,
                )
                for oracle, action in zip(oracles, actions)
            ]

            self.assertEqual(
                [observation.to_dict() for observation in actual[0]],
                [transition.observation.to_dict() for transition in expected],
            )
            self.assertEqual(actual[1], [transition.reward for transition in expected])
            self.assertEqual(
                actual[2], [transition.terminated for transition in expected]
            )
            self.assertEqual(actual[3], [transition.truncated for transition in expected])
            for info, transition in zip(actual[4], expected):
                events = [
                    {
                        key: value
                        for key, value in event.to_dict().items()
                        if key != "kind_name"
                    }
                    for event in info["events"]
                ]
                self.assertEqual(events, [event.to_dict() for event in transition.events])
                self.assertEqual(
                    info["diagnostics"].diagnostics(), transition.diagnostics()
                )
            self.assertEqual(len(set(vector.state_hash())), len(seeds))

    def test_exact_padded_events_are_lazy_cached_and_expire(self) -> None:
        with PaddedVectorEnv(
            2, physics_backend="exact", worker_path=EXACT_WORKER
        ) as vector:
            vector.reset(seed=[71, 71])
            fetches = 0
            client = vector.envs[0]._client
            self.assertIsNotNone(client)
            original = client.fetch_events_payload

            def fetch(generation: int) -> bytes:
                nonlocal fetches
                fetches += 1
                return original(generation)

            with mock.patch.object(client, "fetch_events_payload", fetch):
                result = vector.step([Action.wait(), Action.wait()])
                first, expiring = result[4][0]["events"], result[4][1]["events"]
                self.assertGreater(len(first), 0)
                self.assertEqual(fetches, 0)
                cached = first.materialize()
                cached_dicts = [event.to_dict() for event in cached]
                self.assertEqual(fetches, 1)
                self.assertEqual(
                    [event.to_dict() for event in first.materialize()], cached_dicts
                )
                self.assertEqual(fetches, 1)
                vector.step([Action.wait(), Action.wait()])
                self.assertEqual(
                    [event.to_dict() for event in first.materialize()], cached_dicts
                )
                with self.assertRaisesRegex(NativeError, "events expired"):
                    expiring.materialize()

    def test_oversized_lazy_events_fail_bounded_without_desynchronizing(self) -> None:
        config = {
            "gauge_initial": 40_000,
            "passive_gauge_decay_per_tick": 0,
            "rotten_penalty": 0,
            "spawn_interval_ticks": 1,
            "piece_life_ticks": 200,
            "max_episode_ticks": 100_000,
        }
        with PaddedVectorEnv(
            1,
            physics_backend="exact",
            worker_path=EXACT_WORKER,
            config=config,
        ) as vector:
            vector.reset(seed=[41])
            result = vector.step([Action.wait(20_000)])
            events = result[4][0]["events"]
            self.assertGreater(len(events), 80_000)
            with self.assertRaisesRegex(
                ExactWorkerError, "exceeds 4 MiB response limit"
            ):
                events.materialize()
            client = vector.envs[0]._client
            self.assertIsNotNone(client)
            self.assertFalse(client.closed)
            self.assertLess(client.last_response_bytes, 1_024)
            observation = vector.step([Action.wait()])[0][0]
            self.assertEqual(observation.tick, 20_001)

    def test_exact_padded_snapshot_isolation_and_transactional_restore(self) -> None:
        with PaddedVectorEnv(
            3,
            workers=2,
            physics_backend="exact",
            worker_path=EXACT_WORKER,
        ) as vector:
            observations, _ = vector.reset(seed=[91, 91, 91])
            self.assertEqual(
                [observation.to_dict() for observation in observations[1:]],
                [observations[0].to_dict()] * 2,
            )
            initial = vector.clone_state()
            vector.step([Action.weak(220, 350), Action.wait(3), Action.wait(3)])
            self.assertNotEqual(vector.state_hash()[0], vector.state_hash()[1])
            self.assertEqual(vector.state_hash()[1], vector.state_hash()[2])

            vector.restore_state(initial)
            self.assertEqual(len(set(vector.state_hash())), 1)
            snapshot = vector.clone_state()
            expected = vector.step([Action.wait(4)] * 3)
            expected_observations = [value.to_dict() for value in expected[0]]
            expected_hashes = vector.state_hash()
            vector.restore_state(snapshot)
            actual = vector.step([Action.wait(4)] * 3)
            self.assertEqual(
                [value.to_dict() for value in actual[0]], expected_observations
            )
            self.assertEqual(actual[1:4], expected[1:4])
            self.assertEqual(vector.state_hash(), expected_hashes)

            before = vector.clone_state()
            before_hashes = vector.state_hash()
            corrupted = bytearray(snapshot[1])
            corrupted[-1] ^= 1
            with self.assertRaisesRegex(NativeError, "checksum"):
                vector.restore_state((snapshot[0], bytes(corrupted), snapshot[2]))
            self.assertEqual(vector.clone_state(), before)
            self.assertEqual(vector.state_hash(), before_hashes)

    def test_exact_padded_honors_cap_and_drains_after_receive_failure(self) -> None:
        with PaddedVectorEnv(
            4,
            workers=2,
            physics_backend="exact",
            worker_path=EXACT_WORKER,
        ) as vector:
            vector.reset(seed=300)
            active = 0
            peak = 0
            with ExitStack() as patches:
                for env in vector.envs:
                    original_send = env.send_step_padded
                    original_receive = env.receive_step_padded_raw

                    def send(*args: object, original=original_send) -> None:
                        nonlocal active, peak
                        original(*args)
                        active += 1
                        peak = max(peak, active)

                    def receive(original=original_receive):
                        nonlocal active
                        try:
                            return original()
                        finally:
                            active -= 1

                    patches.enter_context(mock.patch.object(env, "send_step_padded", send))
                    patches.enter_context(
                        mock.patch.object(env, "receive_step_padded_raw", receive)
                    )
                vector.step([Action.wait()] * 4)
            self.assertEqual((active, peak), (0, 2))

            failed = vector.envs[0]
            client = failed._client
            self.assertIsNotNone(client)
            with mock.patch.object(
                client,
                "receive_step_padded_payload",
                side_effect=ExactProtocolError("synthetic padded decode failure"),
            ):
                with self.assertRaisesRegex(ExactProtocolError, "synthetic padded"):
                    vector.step([Action.wait()] * 4)
            self.assertTrue(failed.closed)
            self.assertTrue(client.closed)
            self.assertEqual(
                [env.observation()["tick"] for env in vector.envs[1:]],
                [2, 2, 2],
            )

    def test_exact_padded_buffers_reuse_and_invalid_input_is_atomic(self) -> None:
        with PaddedVectorEnv(
            2, physics_backend="exact", worker_path=EXACT_WORKER
        ) as vector:
            vector.reset(seed=[11, 12])
            before = vector.state_hash()
            with self.assertRaisesRegex(ValueError, "finite"):
                vector.step([Action.wait(), Action.weak(float("nan"), 300)])
            self.assertEqual(vector.state_hash(), before)

            first = vector.step([Action.wait()] * 2)[0]
            first_addresses = [ctypes.addressof(value) for value in first]
            second = vector.step([Action.wait()] * 2)[0]
            second_addresses = [ctypes.addressof(value) for value in second]
            third = vector.step([Action.wait()] * 2)[0]
            third_addresses = [ctypes.addressof(value) for value in third]
            self.assertNotEqual(first_addresses, second_addresses)
            self.assertEqual(first_addresses, third_addresses)
            self.assertEqual([value.tick for value in first], [3, 3])
            self.assertEqual([value.tick for value in second], [2, 2])


if __name__ == "__main__":
    unittest.main()
