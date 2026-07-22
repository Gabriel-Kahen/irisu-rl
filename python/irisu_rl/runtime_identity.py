"""Fail-closed attestation for the exact worker used by training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from irisu_env.exact_ipc import ExactSimulator


@dataclass(frozen=True, slots=True)
class ExactRuntimeIdentity:
    worker_sha256: str
    exact_library_sha256: str
    protocol_version: int = 1
    body_capacity: int = 196
    pointer_bits: int = 32
    backend: str = "exact-msvc9-r58-multiworld-forward"

    def attest(self, worker_path: str | Path) -> dict[str, object]:
        supplied = Path(worker_path).expanduser()
        if not supplied.is_absolute():
            raise ValueError("exact worker path must be absolute")
        path = supplied.resolve(strict=True)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.worker_sha256:
            raise RuntimeError("exact worker executable hash mismatch")
        with ExactSimulator(worker_path=path) as simulator:
            info = simulator.build_info()
            provenance = simulator.exact_library_provenance()
        expected = {
            "worker_executable_sha256": self.worker_sha256,
            "exact_library_sha256": self.exact_library_sha256,
            "protocol_version": self.protocol_version,
            "body_capacity": self.body_capacity,
            "pointer_bits": self.pointer_bits,
            "worker_backend": self.backend,
        }
        mismatches = {
            key: (info.get(key), value)
            for key, value in expected.items()
            if info.get(key) != value
        }
        if mismatches:
            raise RuntimeError(f"exact runtime identity mismatch: {mismatches}")
        if provenance.get("sha256") != self.exact_library_sha256:
            raise RuntimeError("mapped exact library provenance mismatch")
        return {"worker_path": str(path), "build_info": info, "provenance": provenance}


ACCEPTED_EXACT_RUNTIME_2026_07_21 = ExactRuntimeIdentity(
    worker_sha256="4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261",
    exact_library_sha256="ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5",
)
