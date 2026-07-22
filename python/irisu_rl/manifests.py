"""Canonical run identities for reproducible R2 experiments."""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .models import RecurrentActorCritic
from .ppo import PPOConfig


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
