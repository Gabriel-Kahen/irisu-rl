"""Canonical run identities for reproducible R2 experiments."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .models import RecurrentActorCritic
from .ppo import PPOConfig


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class SimulatorIdentity:
    """Runtime/mechanics identity required by every resumable training run."""

    backend: str
    worker_executable_sha256: str | None
    physics_library_sha256: str
    mechanics_config_sha256: str
    config_hashes: tuple[int, ...]
    protocol_version: int
    seed_manifest_sha256: str

    def __post_init__(self) -> None:
        if self.backend not in {"portable", "exact"}:
            raise ValueError("simulator backend must be portable or exact")
        if self.backend == "exact" and not _is_sha256(
            self.worker_executable_sha256
        ):
            raise ValueError("exact simulator identity requires a worker hash")
        if self.backend == "portable" and self.worker_executable_sha256 is not None:
            raise ValueError("portable simulator identity cannot name an exact worker")
        if not _is_sha256(self.physics_library_sha256) or not _is_sha256(
            self.mechanics_config_sha256
        ):
            raise ValueError("simulator library and mechanics hashes must be SHA-256")
        if not _is_sha256(self.seed_manifest_sha256):
            raise ValueError("seed manifest hash must be SHA-256")
        if (
            not self.config_hashes
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value < 2**64
                for value in self.config_hashes
            )
        ):
            raise ValueError("simulator config hashes must be nonempty uint64 values")
        if (
            isinstance(self.protocol_version, bool)
            or not isinstance(self.protocol_version, int)
            or self.protocol_version <= 0
        ):
            raise ValueError("simulator protocol version must be a positive integer")

    def manifest(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "worker_executable_sha256": self.worker_executable_sha256,
            "physics_library_sha256": self.physics_library_sha256,
            "mechanics_config_sha256": self.mechanics_config_sha256,
            "config_hashes": list(self.config_hashes),
            "protocol_version": self.protocol_version,
            "seed_manifest_sha256": self.seed_manifest_sha256,
        }


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def runtime_manifest(
    repository_root: str | Path,
    *,
    model: RecurrentActorCritic,
    ppo: PPOConfig,
    reward_scale: float,
    gamma_tick: float,
    lambda_tick: float,
    code_revision: str,
    observation_provenance: str,
    simulator: SimulatorIdentity,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build the immutable identity embedded in every checkpoint and report."""

    root = Path(repository_root).resolve()
    lock = root / "uv.lock"
    if not lock.is_file():
        raise FileNotFoundError("runtime manifest requires the repository uv.lock")
    if not code_revision:
        raise ValueError("code revision must be explicit")
    if (
        not math.isfinite(reward_scale)
        or reward_scale <= 0
        or gamma_tick != 1.0
        or not 0 < lambda_tick <= 1
    ):
        raise ValueError("invalid R2 reward/discount configuration")
    allowed_provenance = {"privileged_simulator", "causal_actor_tracks"}
    if observation_provenance not in allowed_provenance:
        raise ValueError("observation provenance must identify the actual producer")
    if (
        observation_provenance == "causal_actor_tracks"
        and model.schema.source != "actor_tracks"
    ):
        raise ValueError("causal actor-track provenance requires an actor schema")
    torch_config = torch.__config__.show()
    manifest: dict[str, object] = {
        "version": "irisu-r2-run-manifest-v1",
        "code_revision": code_revision,
        "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_build_config_sha256": hashlib.sha256(torch_config.encode()).hexdigest(),
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "model": model.manifest(),
        "simulator": simulator.manifest(),
        "ppo": ppo.manifest(),
        "reward": {
            "raw": "score_after - score_before",
            "optimizer_scale": reward_scale,
            "clip": False,
        },
        "smdp": {"gamma_tick": gamma_tick, "lambda_tick": lambda_tick},
        "observation_provenance": observation_provenance,
        "deployable": False,
        "transfer_gate": "R4 tracker/input calibration pending",
    }
    if extra:
        overlap = set(manifest) & set(extra)
        if overlap:
            raise ValueError(
                f"extra manifest fields collide with canonical fields: {sorted(overlap)}"
            )
        manifest.update(extra)
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest
