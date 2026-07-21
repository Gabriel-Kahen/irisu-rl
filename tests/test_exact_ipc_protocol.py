from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from irisu_env.exact_ipc import ExactWorkerClient as AttestedExactWorkerClient

MODULE_PATH = ROOT / "tools/exact-physics-prototype/ipc_client.py"
SPEC = importlib.util.spec_from_file_location("irisu_exact_ipc_client", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
IPC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IPC
SPEC.loader.exec_module(IPC)


def observation_payload() -> bytes:
    return IPC.OBSERVATION_HEADER.pack(
        7, 11, 2993, 40_000, 2,
        130.0, 120.0, 320.0, 250.0, 120.0, 370.0,
        3, 4, 96, 5, 0, 0, 0, 1, 0,
    )


def request(process: subprocess.Popen[bytes], opcode: int, request_id: int,
            payload: bytes = b"") -> bytes:
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(
        IPC.HEADER.pack(IPC.MAGIC, IPC.VERSION, opcode, request_id, len(payload))
        + payload
    )
    process.stdin.flush()
    header = IPC._read_exact(process.stdout, IPC.HEADER.size)
    magic, version, response_opcode, response_id, size = IPC.HEADER.unpack(header)
    if (magic != IPC.MAGIC or version != IPC.VERSION):
        raise AssertionError("worker returned an invalid frame header")
    if response_opcode != opcode or response_id != request_id:
        raise AssertionError("worker response does not match request")
    response = IPC._read_exact(process.stdout, size)
    status = IPC.STATUS.unpack_from(response)[0]
    if status:
        raise AssertionError(response[IPC.STATUS.size:].decode(errors="replace"))
    return response[IPC.STATUS.size:]


class ExactIpcProtocolTests(unittest.TestCase):
    def test_exact_library_sha_must_be_real_hex(self) -> None:
        self.assertTrue(IPC.valid_exact_library_sha256("a" * 64))
        self.assertTrue(IPC.valid_exact_library_sha256("A" * 64))
        self.assertFalse(IPC.valid_exact_library_sha256("unknown"))
        self.assertFalse(IPC.valid_exact_library_sha256("g" * 64))
        self.assertFalse(IPC.valid_exact_library_sha256("0" * 64))

    def test_config_flattening_matches_c_abi_keys(self) -> None:
        self.assertEqual(
            IPC.flatten_config(
                {"piece_sizes": [31, 42], "gravity_y": 123.5}
            ),
            [("gravity_y", 123.5), ("piece_sizes[0]", 31.0),
             ("piece_sizes[1]", 42.0)],
        )
        with self.assertRaisesRegex(TypeError, "non-empty strings"):
            IPC.flatten_config({"": 1})
        with self.assertRaisesRegex(ValueError, "NUL"):
            IPC.flatten_config({"bad\0key": 1})

    def test_transition_decodes_structured_events(self) -> None:
        detail = b"normal rot penalty"
        suffix = IPC.TRANSITION.pack(
            20, 1, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        event = IPC.EVENT.pack(7, 9, -1_860, 17, 0, len(detail), 11, 0) + detail
        transition = IPC.decode_transition(observation_payload() + suffix + event)
        self.assertEqual(transition.reward, 20)
        self.assertEqual(transition.event_count, 1)
        self.assertEqual(transition.events[0].detail, "normal rot penalty")
        self.assertEqual(transition.events[0].value, -1_860)

    def test_transition_rejects_event_count_mismatch(self) -> None:
        suffix = IPC.TRANSITION.pack(
            0, 1, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        )
        with self.assertRaisesRegex(IPC.ProtocolError, "truncated event"):
            IPC.decode_transition(observation_payload() + suffix)

    @unittest.skipUnless(os.environ.get("IRISU_EXACT_WORKER"),
                         "IRISU_EXACT_WORKER is not set")
    def test_optional_worker_smoke(self) -> None:
        result = IPC.smoke(Path(os.environ["IRISU_EXACT_WORKER"]), 41)
        self.assertEqual(result["worker"]["pointer_bits"], 32)
        self.assertEqual(result["worker"]["backend"], IPC.EXACT_BACKEND)
        self.assertTrue(
            IPC.valid_exact_library_sha256(
                result["worker"]["exact_library_sha256"]
            )
        )
        self.assertEqual(result["step"]["tick"], 1)

    @unittest.skipUnless(os.environ.get("IRISU_EXACT_WORKER"),
                         "IRISU_EXACT_WORKER is not set")
    def test_optional_production_client_requires_call_target_attestation(self) -> None:
        with AttestedExactWorkerClient(
            os.environ["IRISU_EXACT_WORKER"]
        ) as client:
            self.assertEqual(client.exact_entrypoint_count, 15)
            self.assertEqual(
                client.exact_call_target_identity,
                client.initial_exact_library_provenance["mapped_identity"],
            )

    @unittest.skipUnless(
        os.environ.get("IRISU_EXACT_WORKER")
        and os.environ.get("IRISU_EXACT_B2D_INTERPOSER"),
        "exact worker/interposer is not set",
    )
    def test_worker_rejects_b2d_preload_while_genuine_library_is_mapped(self) -> None:
        interposer = Path(os.environ["IRISU_EXACT_B2D_INTERPOSER"])
        self.assertEqual(interposer.read_bytes()[4], 1, "interposer must be ELF32")
        env = os.environ.copy()
        env["LD_PRELOAD"] = str(interposer)
        result = subprocess.run(
            [os.environ["IRISU_EXACT_WORKER"]],
            input=b"",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=5,
            check=False,
        )
        stderr = result.stderr.decode("utf-8", errors="replace")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("IRISU_TEST_GENUINE_EXACT_SONAME_MAPPED", stderr)
        self.assertIn("b2d_world_step", stderr)
        self.assertIn("does not match the attested call target", stderr)

    @unittest.skipUnless(
        os.environ.get("IRISU_EXACT_WORKER")
        and os.environ.get("IRISU_EXACT_MSVC_INTERPOSER"),
        "exact worker/interposer is not set",
    )
    def test_internal_msvc_preload_cannot_redirect_exact_calls(self) -> None:
        interposer = Path(os.environ["IRISU_EXACT_MSVC_INTERPOSER"])
        self.assertEqual(interposer.read_bytes()[4], 1, "interposer must be ELF32")
        env = os.environ.copy()
        env["LD_PRELOAD"] = str(interposer)
        process = subprocess.Popen(
            [os.environ["IRISU_EXACT_WORKER"]],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        stderr = ""
        try:
            request(process, IPC.HELLO, 1)
            request(process, IPC.RESET, 2, IPC.RESET_REQUEST.pack(41))
            request(
                process,
                IPC.STEP,
                3,
                IPC.STEP_REQUEST.pack(0, 0.0, 0.0, 1, 0),
            )
            request(process, IPC.CLOSE, 4)
            assert process.stdin is not None
            process.stdin.close()
            self.assertEqual(process.wait(timeout=5), 0)
            assert process.stderr is not None
            stderr = process.stderr.read().decode("utf-8", errors="replace")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()
        self.assertIn("IRISU_TEST_MSVC_INTERPOSER_LOADED", stderr)
        self.assertNotIn("IRISU_TEST_MSVC_INTERPOSER_CALLED", stderr)

    @unittest.skipUnless(os.environ.get("IRISU_EXACT_WORKER"),
                         "IRISU_EXACT_WORKER is not set")
    def test_optional_worker_configure_is_atomic(self) -> None:
        with IPC.ExactWorkerClient(os.environ["IRISU_EXACT_WORKER"]) as client:
            default_hash = client.current_config_hash
            configured_hash = client.configure(
                {"gravity_y": 123.5, "gauge_initial": 1_234}
            )
            self.assertNotEqual(configured_hash, default_hash)
            config = client.config_json()
            self.assertEqual(config["config_hash"], configured_hash)
            self.assertEqual(config["gravity_y"], 123.5)
            self.assertEqual(client.reset(7).gauge, 1_234)

            with self.assertRaisesRegex(IPC.WorkerError, "unknown configuration"):
                client.configure({"not_a_key": 1})
            self.assertEqual(client.current_config_hash, configured_hash)
            self.assertEqual(client.config_json()["config_hash"], configured_hash)

            with self.assertRaisesRegex(IPC.WorkerError, "gauge configuration"):
                client.configure({"gauge_initial": 50_000})
            self.assertEqual(client.current_config_hash, configured_hash)
            with self.assertRaisesRegex(IPC.WorkerError, "one successful reset"):
                client.reset(7)
            with self.assertRaisesRegex(IPC.WorkerError, "immutable after reset"):
                client.configure({})
            self.assertNotEqual(configured_hash, default_hash)


if __name__ == "__main__":
    unittest.main()
