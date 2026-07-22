from __future__ import annotations

import functools
import importlib.util
import os
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from irisu_env.exact_ipc import ExactWorkerClient as AttestedExactWorkerClient

MODULE_PATH = ROOT / "tools/exact-physics-prototype/ipc_client.py"
SPEC = importlib.util.spec_from_file_location("irisu_exact_ipc_client", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
IPC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IPC
SPEC.loader.exec_module(IPC)


@functools.cache
def linked_libraries(binary: str) -> dict[str, Path]:
    ldd = shutil.which("ldd")
    if ldd is None:
        raise AssertionError("ldd is required to resolve exact worker dependencies")
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("LD_") and key != "GLIBC_TUNABLES"
    }
    environment["LC_ALL"] = "C"
    result = subprocess.run(
        [ldd, binary],
        capture_output=True,
        text=True,
        env=environment,
        timeout=5,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"cannot inspect exact worker dependencies: {result.stderr}")
    libraries = {}
    for line in result.stdout.splitlines():
        name, separator, target = line.partition("=>")
        if separator and target.strip() != "not found":
            libraries[name.strip()] = Path(
                target.split("(", 1)[0].strip()
            ).resolve(strict=True)
    return libraries


def linked_address_sanitizer_runtime(worker: str) -> Path | None:
    if "libasan.so" not in elf_dynamic_metadata(Path(worker)):
        return None
    for name, path in linked_libraries(worker).items():
        if name.startswith("libasan.so"):
            with path.open("rb") as stream:
                elf_header = stream.read(5)
            if elf_header != b"\x7fELF\x01":
                raise AssertionError("worker address-sanitizer runtime is not ELF32")
            return path
    raise AssertionError("worker requires libasan but ldd did not resolve it")


def elf_dynamic_metadata(binary: Path) -> str:
    readelf = shutil.which("readelf")
    if readelf is None:
        raise AssertionError("readelf is required to inspect sanitizer coverage")
    result = subprocess.run(
        [readelf, "--wide", "--dynamic", "--symbols", str(binary)],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(f"cannot inspect {binary}: {result.stderr}")
    return result.stdout


def preload_environment(interposer: Path) -> dict[str, str]:
    environment = os.environ.copy()
    runtime = linked_address_sanitizer_runtime(os.environ["IRISU_EXACT_WORKER"])
    libraries = ([str(runtime)] if runtime is not None else []) + [str(interposer)]
    environment["LD_PRELOAD"] = os.pathsep.join(libraries)
    return environment


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
    def test_worker_forces_and_reports_canonical_x87_control_word(self) -> None:
        with mock.patch.dict(os.environ, {"IRISU_EXACT_CW": "0x037f"}):
            with IPC.ExactWorkerClient(
                os.environ["IRISU_EXACT_WORKER"]
            ) as client:
                self.assertEqual(client.info.x87_control_word, 0x027F)

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
        result = subprocess.run(
            [os.environ["IRISU_EXACT_WORKER"]],
            input=b"",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=preload_environment(interposer),
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
        and os.environ.get("IRISU_EXACT_B2D_INTERPOSER")
        and os.environ.get("IRISU_EXACT_DLSYM_INTERPOSER")
        and os.environ.get("IRISU_EXACT_MSVC_INTERPOSER"),
        "exact worker/interposers are not set",
    )
    def test_sanitizer_instrumentation_boundary_is_explicit(self) -> None:
        worker = os.environ["IRISU_EXACT_WORKER"]
        if linked_address_sanitizer_runtime(worker) is None:
            self.skipTest("worker is not linked with address sanitizer")

        instrumented = [
            Path(worker),
            Path(os.environ["IRISU_EXACT_B2D_INTERPOSER"]),
            Path(os.environ["IRISU_EXACT_DLSYM_INTERPOSER"]),
            Path(os.environ["IRISU_EXACT_MSVC_INTERPOSER"]),
        ]
        for binary in instrumented:
            metadata = elf_dynamic_metadata(binary)
            with self.subTest(binary=binary.name):
                self.assertIn("libasan.so", metadata)
                self.assertIn("libubsan.so", metadata)
                self.assertIn("__asan_init", metadata)

        exact_hosts = [
            path
            for name, path in linked_libraries(worker).items()
            if name.startswith("libirisu_box2d_msvc_exact")
        ]
        self.assertEqual(len(exact_hosts), 1)
        host_metadata = elf_dynamic_metadata(exact_hosts[0])
        self.assertNotIn("libasan.so", host_metadata)
        self.assertNotIn("libubsan.so", host_metadata)
        self.assertNotIn("__asan_", host_metadata)
        self.assertNotIn("__ubsan_", host_metadata)

    @unittest.skipUnless(
        os.environ.get("IRISU_EXACT_WORKER")
        and os.environ.get("IRISU_EXACT_DLSYM_INTERPOSER"),
        "exact worker/dlsym interposer is not set",
    )
    def test_worker_rejects_same_object_entrypoint_permutation(self) -> None:
        interposer = Path(os.environ["IRISU_EXACT_DLSYM_INTERPOSER"])
        self.assertEqual(interposer.read_bytes()[4], 1, "interposer must be ELF32")
        result = subprocess.run(
            [os.environ["IRISU_EXACT_WORKER"]],
            input=b"",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=preload_environment(interposer),
            timeout=5,
            check=False,
        )
        stderr = result.stderr.decode("utf-8", errors="replace")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("IRISU_TEST_DLSYM_INTERPOSER_LOADED", stderr)
        self.assertIn("b2d_world_get_x", stderr)
        self.assertIn("expected symbol identity", stderr)

    @unittest.skipUnless(
        os.environ.get("IRISU_EXACT_WORKER")
        and os.environ.get("IRISU_EXACT_DLSYM_INTERPOSER"),
        "exact worker/dlsym interposer is not set",
    )
    def test_production_client_scrubs_loader_injection_environment(self) -> None:
        interposer = os.environ["IRISU_EXACT_DLSYM_INTERPOSER"]
        hostile = {
            "LD_PRELOAD": interposer,
            "LD_AUDIT": interposer,
            "LD_LIBRARY_PATH": str(ROOT),
            "LD_DEBUG": "all",
            "GLIBC_TUNABLES": "glibc.rtld.nns=16",
            "IRISU_EXACT_CW": "0x037f",
        }
        with mock.patch.dict(os.environ, hostile):
            with AttestedExactWorkerClient(
                os.environ["IRISU_EXACT_WORKER"]
            ) as client:
                self.assertEqual(client.info.x87_control_word, 0x027F)
                self.assertEqual(client.reset(41).tick, 0)

    @unittest.skipUnless(
        os.environ.get("IRISU_EXACT_WORKER")
        and os.environ.get("IRISU_EXACT_MSVC_INTERPOSER"),
        "exact worker/interposer is not set",
    )
    def test_internal_msvc_preload_cannot_redirect_exact_calls(self) -> None:
        interposer = Path(os.environ["IRISU_EXACT_MSVC_INTERPOSER"])
        self.assertEqual(interposer.read_bytes()[4], 1, "interposer must be ELF32")
        process = subprocess.Popen(
            [os.environ["IRISU_EXACT_WORKER"]],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=preload_environment(interposer),
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
