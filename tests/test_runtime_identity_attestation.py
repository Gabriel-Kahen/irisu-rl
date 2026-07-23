from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from irisu_env.native import NativeSimulator, find_library
from irisu_rl.runtime_identity import attest_simulator_runtime


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _PortableLane:
    def __init__(self, library: Path, *, clone_version: str = "1") -> None:
        self.library_path = str(library)
        self._clone_version = clone_version

    @property
    def build_info(self):
        return {
            "physics_backend": "portable-test-r58",
            "snapshot_schema": 7,
            "clone_version": self._clone_version,
            "pointer_bits": 64,
        }


class _ExactLane:
    def __init__(
        self,
        worker: Path,
        library: Path,
        *,
        worker_pid: int,
        config_hash: int,
    ) -> None:
        self.worker_path = str(worker)
        self._library = library
        self._worker_pid = worker_pid
        self._config_hash = config_hash

    def build_info(self):
        return {
            "physics_backend": "exact-test-r58-worker",
            "snapshot_schema": 0x45580001,
            "protocol_version": 1,
            "worker_pid": self._worker_pid,
            "config_hash": self._config_hash,
            "worker_executable_sha256": _sha256(Path(self.worker_path)),
            "exact_library_sha256": _sha256(self._library),
            "exact_library_runtime_verified": True,
            "exact_call_targets_runtime_verified": True,
        }

    def exact_library_provenance(self):
        stat = self._library.stat()
        return {
            "status": "captured",
            "path": str(self._library.resolve()),
            "bytes": stat.st_size,
            "sha256": _sha256(self._library),
            "file_identity": {
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "mtime_ns": stat.st_mtime_ns,
                "ctime_ns": stat.st_ctime_ns,
            },
            "mapped_identity": {"device": "00:00", "inode": stat.st_ino},
        }


class _Vector:
    def __init__(self, *lanes: object) -> None:
        self.envs = tuple(lanes)
        self.num_envs = len(lanes)


class RuntimeIdentityAttestationTests(unittest.TestCase):
    def test_portable_attestation_tracks_loaded_inode_after_path_replacement(
        self,
    ) -> None:
        source = Path(find_library()).resolve(strict=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / source.name
            replacement = root / "replacement.so"
            shutil.copyfile(source, library)
            original_sha256 = _sha256(library)
            with NativeSimulator(library) as simulator:
                before = attest_simulator_runtime(simulator)
                replacement.write_bytes(b"replacement-that-is-not-the-loaded-library")
                os.replace(replacement, library)
                after = attest_simulator_runtime(simulator)
                provenance = simulator.portable_library_provenance()

                self.assertNotEqual(_sha256(library), original_sha256)
                self.assertNotEqual(
                    provenance["file_identity"]["inode"], library.stat().st_ino
                )

        self.assertEqual(before.sha256, after.sha256)
        self.assertEqual(before.runtime_artifact_sha256, original_sha256)
        self.assertEqual(after.runtime_artifact_sha256, original_sha256)

    def test_portable_identity_hashes_actual_library_and_is_relocatable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "first.so"
            second = root / "second.so"
            first.write_bytes(b"portable-runtime")
            second.write_bytes(first.read_bytes())

            single = attest_simulator_runtime(_PortableLane(first))
            relocated = attest_simulator_runtime(_PortableLane(second))
            vector = attest_simulator_runtime(
                _Vector(_PortableLane(first), _PortableLane(second))
            )

        self.assertEqual(single.backend, "portable")
        self.assertEqual(single.snapshot_schema, 7)
        self.assertEqual(
            single.runtime_artifact_sha256,
            hashlib.sha256(b"portable-runtime").hexdigest(),
        )
        self.assertEqual(single.sha256, relocated.sha256)
        self.assertEqual(single.sha256, vector.sha256)
        self.assertEqual(vector.verified_lanes, 2)
        self.assertIsNone(vector.manifest()["exact_library"])

    def test_exact_vector_attests_worker_and_mapped_library_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker"
            library = root / "legacy.dll"
            worker.write_bytes(b"exact-worker")
            library.write_bytes(b"mapped-legacy-library")
            first = _ExactLane(worker, library, worker_pid=101, config_hash=7)
            second = _ExactLane(worker, library, worker_pid=202, config_hash=9)

            single = attest_simulator_runtime(first)
            vector = attest_simulator_runtime(_Vector(first, second))

        self.assertEqual(vector.backend, "exact")
        self.assertEqual(vector.snapshot_schema, 0x45580001)
        self.assertEqual(vector.verified_lanes, 2)
        self.assertEqual(vector.sha256, single.sha256)
        self.assertEqual(
            vector.runtime_artifact_sha256, hashlib.sha256(b"exact-worker").hexdigest()
        )
        self.assertEqual(
            vector.exact_library_sha256,
            hashlib.sha256(b"mapped-legacy-library").hexdigest(),
        )
        self.assertNotIn("worker_pid", vector.build_info)
        self.assertNotIn("config_hash", vector.build_info)

    def test_exact_worker_report_must_match_measured_worker_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker"
            library = root / "legacy.dll"
            worker.write_bytes(b"worker")
            library.write_bytes(b"library")
            lane = _ExactLane(worker, library, worker_pid=1, config_hash=2)
            original = lane.build_info

            def wrong_build_info():
                value = original()
                value["worker_executable_sha256"] = "f" * 64
                return value

            lane.build_info = wrong_build_info  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "worker bytes"):
                attest_simulator_runtime(lane)

    def test_exact_provenance_must_match_current_file_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker"
            library = root / "legacy.dll"
            worker.write_bytes(b"worker")
            library.write_bytes(b"library")
            lane = _ExactLane(worker, library, worker_pid=1, config_hash=2)
            original = lane.exact_library_provenance

            def stale_provenance():
                value = original()
                value["file_identity"] = dict(value["file_identity"])
                value["file_identity"]["inode"] += 1
                return value

            lane.exact_library_provenance = stale_provenance  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "file identity"):
                attest_simulator_runtime(lane)

    def test_vector_rejects_heterogeneous_builds_and_bad_lane_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            library = Path(directory) / "runtime.so"
            library.write_bytes(b"portable-runtime")
            with self.assertRaisesRegex(RuntimeError, "heterogeneous"):
                attest_simulator_runtime(
                    _Vector(
                        _PortableLane(library, clone_version="one"),
                        _PortableLane(library, clone_version="two"),
                    )
                )
            vector = _Vector(_PortableLane(library))
            vector.num_envs = 2
            with self.assertRaisesRegex(ValueError, "lane count"):
                attest_simulator_runtime(vector)


if __name__ == "__main__":
    unittest.main()
