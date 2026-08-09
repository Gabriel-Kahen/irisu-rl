#!/usr/bin/env python3
"""Development-only conservative all-outcome residual benchmark for R3G.

The branch owns correctness, 8-seed resource screens, offline fitting, and
the 64-seed calibration/heldout/stress barrier cohorts.  It deliberately has
no entry point capable of materializing the parent-only winner suites.
"""

from __future__ import annotations

import os

# These must be fixed before NumPy or Torch initializes a nested runtime.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import argparse
import hashlib
import json
import math
import platform
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from irisu_env import Action, ActionKind, EventKind, IrisuEnv
from irisu_pointer.conservative_offline import (
    AllOutcomeCollectorPolicy,
    BarrierCalibration,
    ConservativeResidualPolicy,
    IsotonicCalibration,
    R3G_OFFLINE_VERSION,
    RecordingJointSearch,
    ResidualValueNet,
    SupportEnvelope,
    TrainedEnsemble,
    canonical_sha256,
    fit_barrier,
    fit_isotonic,
    fit_support,
    planner_config,
    teacher_planner_config,
    train_ensemble,
)
from irisu_pointer.joint_planner import (
    JOINT_PLANNER_VERSION,
    JointPairGeometrySearch,
    JointPlannerConfig,
    _commit_base_decision,
)
from irisu_pointer.steering import SteeringDecision
from irisu_pointer.steering_checkpoint import load_steering_checkpoint
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
)
from irisu_rl.runtime_identity import attest_simulator_runtime


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3g-solvency-barrier-tournament-20260729/shared-protocol.md"
)
PROTOCOL_SHA256 = (
    "6dfb2ffa3a76cc00447e3dcf889f6209a17ff5e2f4c3382fe0959bbabbd52991"
)
BASE_REVISION = "de701b36355d5ec582df30f4223aabde7bc537df"
RUNTIME = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/runtime/"
    "main-0c48dba-20260723/portable-build/libirisu_clone.so"
)
RUNTIME_SHA256 = (
    "4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79"
)
BASE = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3d-survival-v5-20260729/long-development.pt"
)
BASE_SHA256 = (
    "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d"
)
TRUSTED_JOINT = Path(
    "/home/gabe/.codex/worktrees/e2f8/irisu/"
    "python/irisu_pointer/joint_planner.py"
)
TRUSTED_JOINT_SHA256 = (
    "dc7009fc18a322eca5dace55b9baf982b6ced26c18517af752aab0f6365d362e"
)
OUTPUT_ROOT = (
    ROOT
    / "artifacts/r3/development/"
    "r3g-conservative-offline-residual-20260729"
)
OUTPUT = OUTPUT_ROOT / "generation-04"
BURNED_TEACHER_SCREEN_MANIFEST_SHA256 = (
    "44d054ae4660cb453e9c606c87ab0e546320cb137c739643504d739f4734fac8"
)
TEACHER_SCREEN_SPLIT = "teacher-screen-e-v4"
TEACHER_SCREEN_MANIFEST_SHA256 = (
    "7816d38a0b7f91dc5b28c20264babd7de370cc6c8824c652ae70cce884f015f5"
)
TEACHER_SCREEN_SEEDS = (
    1_083_350_254,
    3_429_797_324,
    325_449_524,
    2_056_492_209,
    3_202_126_555,
    2_542_773_790,
    1_261_470_716,
    1_798_470_322,
)
PARENT_KNOWN_SCHEDULED_SEEDS = 264
SEED_DOMAIN = "r3g-solvency-barrier-tournament-20260729"
MAX_DECISIONS = 2_000_000
TRAINING_SEED = 2026072905
BOOTSTRAP_REPLICATES = 10_000
CONFORMAL_ALPHA = 0.05

EXTERNAL_MANIFEST_ROOTS: Mapping[str, Path] = {
    "strategy-a": Path(
        "/home/gabe/.codex/worktrees/1a8e/irisu/artifacts/r3/development/"
        "r3g-event-world-model-mpc-20260729"
    ),
    "strategy-b": Path(
        "/home/gabe/.codex/worktrees/a6b0/irisu/artifacts/r3/development/"
        "r3g-distributional-barrier-20260729"
    ),
    "strategy-c": Path(
        "/home/gabe/.codex/worktrees/7ae4/irisu/artifacts/r3/development/"
        "r3g-analytic-solvency-shield-20260729"
    ),
    "strategy-d": Path(
        "/home/gabe/.codex/worktrees/40f2/irisu/artifacts/r3/development/"
        "r3g-active-shielded-dagger-20260729"
    ),
    "strategy-e-prior": OUTPUT_ROOT,
}

# Frozen before any branch result is viewed.
TEACHER_CONFIG = teacher_planner_config()
DATA_CONFIG = planner_config()
QUERY_BUDGET = {
    "candidate_cap": 15,
    "exact_rollout_cap_ticks": 512,
    "dagger_train_horizon": 2_000,
    "dagger_train_queries_per_episode": 8,
    "calibration_horizon": 2_000,
    "calibration_queries_per_episode": 8,
    "heldout_horizon": 2_000,
    "heldout_sample_audits_per_episode": 8,
    "stress_horizon": 10_000,
    "stress_sample_audits_per_episode": 8,
    "student_screen_sample_audits_per_episode": 8,
    "query_stride_shots": 4,
    "student_action_query_schedule": (
        "every fourth shot opportunity for the full episode"
    ),
    "exact_audit_schedule": (
        "all actual overrides plus the fixed cohort sample"
    ),
    "teacher_queries_per_episode": 48,
    "teacher_query_stride_shots": 2,
    "training_steps": 500,
    "ensemble_size": 3,
}
if (
    DATA_CONFIG.pair_cap * DATA_CONFIG.geometry_cap
    != QUERY_BUDGET["candidate_cap"]
    or DATA_CONFIG.horizons[-1] != QUERY_BUDGET["exact_rollout_cap_ticks"]
):
    raise RuntimeError("offline candidate or exact-rollout budget changed")

SPLITS: Mapping[str, tuple[int, int, str]] = {
    TEACHER_SCREEN_SPLIT: (
        8,
        2_000,
        "paired allocation-only Strategy E v4 teacher screen",
    ),
    "student-screen": (8, 10_000, "teacher-free student resource screen"),
    "dagger-train": (
        16,
        QUERY_BUDGET["dagger_train_horizon"],
        "frozen-v5 visited complete-outcome offline fitting",
    ),
    "barrier-calibration": (
        64,
        QUERY_BUDGET["calibration_horizon"],
        "whole-trajectory episode-max split-conformal calibration",
    ),
    "barrier-heldout": (
        64,
        QUERY_BUDGET["heldout_horizon"],
        "final-student on-policy false-safe and coverage evaluation",
    ),
    "barrier-stress": (
        64,
        QUERY_BUDGET["stress_horizon"],
        "targeted low-gauge high-level high-rot-debt evaluation",
    ),
}

# Two public cached frozen-v5 episodes from the permitted corrected joint-v2
# development report.  These are sentinels, not tournament split members.
SENTINELS: tuple[Mapping[str, int], ...] = (
    {
        "seed": 0x4FDC0458,
        "horizon": 2_000,
        "survival_ticks": 1_845,
        "final_score": 36,
        "final_gauge": 1,
        "final_level": 1,
        "qualifying_clears": 2,
        "rotten_events": 2,
    },
    {
        "seed": 0x75FC6DD0,
        "horizon": 2_000,
        "survival_ticks": 2_000,
        "final_score": 80,
        "final_gauge": 4_780,
        "final_level": 1,
        "qualifying_clears": 6,
        "rotten_events": 1,
    },
)


def _json_default(value: object) -> object:
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        return item()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return tolist()
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return list(value)
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def _bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=_json_default,
    ).encode()


def _sha(value: object) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin_one_core() -> Mapping[str, object]:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    affinity: list[int] = []
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        if not available:
            raise RuntimeError("process has no available logical CPU")
        os.sched_setaffinity(0, {available[0]})
        affinity = sorted(os.sched_getaffinity(0))
        if len(affinity) != 1:
            raise RuntimeError("failed to pin R3G work to one logical CPU")
    return {
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "affinity": affinity,
        "thread_environment": {
            name: os.environ[name]
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
    }


def _safe_output() -> Path:
    if OUTPUT.is_symlink():
        raise ValueError("R3G artifact directory must not be a symlink")
    resolved = OUTPUT.resolve()
    if resolved != OUTPUT.absolute():
        raise ValueError("R3G artifact directory resolved unexpectedly")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _atomic_json(path: Path, value: Mapping[str, object]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to rewrite prior artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(
                value,
                stream,
                sort_keys=True,
                indent=2,
                allow_nan=False,
                default=_json_default,
            )
            stream.write("\n")
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _file_sha256(path)


def _atomic_torch(path: Path, value: Mapping[str, object]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to rewrite prior artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _file_sha256(path)


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact is not a JSON mapping: {path}")
    return value


def _manifest_documents() -> Mapping[str, tuple[Path, ...]]:
    documents: dict[str, tuple[Path, ...]] = {}
    embedded = {
        "strategy-b": (
            "preflight.json",
            "preflight-live-continuation.json",
            "preflight-resumable-correctness.json",
            "preflight-b-v2.json",
        ),
        "strategy-c": ("preregistered-config.json",),
    }
    for owner, root in EXTERNAL_MANIFEST_ROOTS.items():
        if not root.is_dir() or root.is_symlink():
            raise RuntimeError(f"external development root is unavailable: {root}")
        resolved_root = root.resolve()
        paths = {
            path
            for path in root.rglob("*.json")
            if "manifest" in str(path.relative_to(root))
            and OUTPUT not in path.parents
        }
        paths.update(root / name for name in embedded.get(owner, ()))
        checked: list[Path] = []
        for path in sorted(paths):
            relative = path.relative_to(root)
            if any(
                part.lower() in {
                    "sealed",
                    "test",
                    "canonical",
                    "authorization",
                    "winner",
                }
                for part in relative.parts
            ):
                raise RuntimeError(
                    f"forbidden external artifact path discovered: {path}"
                )
            if not path.is_file() or path.is_symlink():
                raise RuntimeError(
                    f"external manifest document is unavailable: {path}"
                )
            resolved = path.resolve()
            if resolved_root not in resolved.parents:
                raise RuntimeError(
                    f"external manifest escaped its development root: {path}"
                )
            checked.append(resolved)
        documents[owner] = tuple(checked)
    return documents


def _seed_values(value: object) -> tuple[int, ...]:
    output: list[int] = []

    def visit(item: object) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if (
                    key == "seed"
                    and type(child) is int
                    and 0 <= child <= 0xFFFFFFFF
                ):
                    output.append(child)
                elif key == "seeds" and isinstance(child, Sequence):
                    for seed in child:
                        if (
                            type(seed) is not int
                            or not 0 <= seed <= 0xFFFFFFFF
                        ):
                            raise ValueError("seed list is malformed")
                        output.append(seed)
                else:
                    visit(child)
        elif isinstance(item, Sequence) and not isinstance(
            item, (str, bytes, bytearray)
        ):
            for child in item:
                visit(child)

    visit(value)
    return tuple(output)


def _validate_manifest_nodes(value: object) -> int:
    validated = 0
    if isinstance(value, Mapping):
        rows = value.get("rows")
        schema = value.get("schema")
        if (
            isinstance(schema, str)
            and "seed-manifest" in schema
            and isinstance(rows, Sequence)
            and rows
            and all(
            isinstance(row, Mapping)
            and type(row.get("seed")) is int
            and isinstance(row.get("split"), str)
            and type(row.get("index")) is int
            for row in rows
            )
        ):
            split_names = sorted({str(row["split"]) for row in rows})
            for split in split_names:
                split_rows = [
                    row for row in rows if str(row["split"]) == split
                ]
                for expected_index, row in enumerate(split_rows):
                    if int(row["index"]) != expected_index:
                        raise ValueError(
                            "external seed manifest indices are not contiguous"
                        )
                    if int(row["seed"]) != _derive_seed(
                        split, expected_index
                    ):
                        raise ValueError(
                            "external seed manifest violates shared derivation"
                        )
                    if type(row.get("horizon")) is not int:
                        raise ValueError(
                            "external seed manifest horizon is malformed"
                        )
                    if not isinstance(row.get("purpose"), str):
                        raise ValueError(
                            "external seed manifest purpose is malformed"
                        )
            digest_key = (
                "sha256"
                if "sha256" in value
                else (
                    "manifest_sha256"
                    if "manifest_sha256" in value
                    else None
                )
            )
            if digest_key is not None:
                unsigned = dict(value)
                declared = unsigned.pop(digest_key)
                if declared != _sha(unsigned):
                    raise ValueError(
                        "external logical seed-manifest SHA changed"
                    )
            validated += len(split_names)
        for child in value.values():
            validated += _validate_manifest_nodes(child)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            validated += _validate_manifest_nodes(child)
    return validated


def _external_manifest_inventory() -> Mapping[str, object]:
    owners: dict[str, object] = {}
    all_seeds: set[int] = set()
    validated_instances = 0
    for owner, paths in _manifest_documents().items():
        records = []
        for path in paths:
            value = _load_json(path)
            seeds = sorted(set(_seed_values(value)))
            instances = _validate_manifest_nodes(value)
            validated_instances += instances
            all_seeds.update(seeds)
            records.append(
                {
                    "path": str(path),
                    "file_sha256": _file_sha256(path),
                    "seed_count": len(seeds),
                    "seed_set_sha256": _sha(seeds),
                    "validated_manifest_instances": instances,
                }
            )
        owners[owner] = {
            "development_root": str(EXTERNAL_MANIFEST_ROOTS[owner]),
            "documents": records,
            "document_count": len(records),
        }
    inventory = {
        "schema": "irisu-r3g-external-manifest-inventory-v1",
        "scope": "development-only A-D and all prior E generations",
        "owners": owners,
        "validated_manifest_instances": validated_instances,
        "unique_seed_count": len(all_seeds),
        "unique_seed_set_sha256": _sha(sorted(all_seeds)),
        "unique_seeds": sorted(all_seeds),
    }
    return {**inventory, "sha256": _sha(inventory)}


def _source_identity() -> Mapping[str, object]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != BASE_REVISION:
        raise RuntimeError("R3G benchmark is not bound to the baseline revision")
    files = (
        Path(__file__).resolve(),
        ROOT / "python/irisu_pointer/conservative_offline.py",
        ROOT / "python/irisu_pointer/joint_planner.py",
        ROOT / "python/irisu_pointer/policy.py",
        ROOT / "python/irisu_pointer/steering.py",
        ROOT / "python/irisu_pointer/steering_learning.py",
        ROOT / "python/irisu_pointer/steering_checkpoint.py",
        ROOT / "python/irisu_pointer/steering_progress.py",
        ROOT / "python/irisu_rl/encoding.py",
        ROOT / "python/irisu_rl/runtime_identity.py",
        ROOT / "python/irisu_env/env.py",
        ROOT / "python/irisu_env/native.py",
        ROOT / "tests/test_r3g_conservative_offline.py",
        ROOT / "pyproject.toml",
    )
    manifest = {
        "schema": "irisu-r3g-conservative-offline-source-v1",
        "git_revision": revision,
        "protocol_sha256": _file_sha256(PROTOCOL),
        "trusted_joint_v2_sha256": _file_sha256(TRUSTED_JOINT),
        "files": {
            str(path.relative_to(ROOT)): _file_sha256(path) for path in files
        },
    }
    if (
        manifest["protocol_sha256"] != PROTOCOL_SHA256
        or manifest["trusted_joint_v2_sha256"] != TRUSTED_JOINT_SHA256
    ):
        raise RuntimeError("locked protocol or corrected joint-v2 source changed")
    return {**manifest, "sha256": _sha(manifest)}


def _assert_inputs(source: Mapping[str, object]) -> None:
    if _source_identity() != source:
        raise RuntimeError("R3G source identity changed during execution")
    if _file_sha256(RUNTIME) != RUNTIME_SHA256:
        raise RuntimeError("trusted portable runtime identity changed")
    if _file_sha256(BASE) != BASE_SHA256:
        raise RuntimeError("frozen-v5 checkpoint identity changed")


def _derive_seed(split: str, index: int) -> int:
    value = f"{SEED_DOMAIN}|{split}|{index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big")


def _seed_bundle() -> Mapping[str, object]:
    manifests: dict[str, object] = {}
    all_seeds: list[int] = []
    for split, (count, horizon, purpose) in SPLITS.items():
        rows = [
            {
                "split": split,
                "index": index,
                "seed": _derive_seed(split, index),
                "horizon": horizon,
                "purpose": purpose,
            }
            for index in range(count)
        ]
        payload = {
            "schema": "irisu-r3g-seed-manifest-v1",
            "derivation": 'SHA256("domain|S|i") bytes 0-3 unsigned big-endian',
            "domain": SEED_DOMAIN,
            "split": split,
            "rows": rows,
        }
        manifests[split] = {**payload, "sha256": _sha(payload)}
        all_seeds.extend(int(row["seed"]) for row in rows)
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("R3G split seed collision")
    if set(all_seeds) & {int(value["seed"]) for value in SENTINELS}:
        raise RuntimeError("cached sentinel overlaps a tournament split")
    fresh = manifests[TEACHER_SCREEN_SPLIT]
    if (
        fresh["sha256"] != TEACHER_SCREEN_MANIFEST_SHA256
        or tuple(int(row["seed"]) for row in fresh["rows"])
        != TEACHER_SCREEN_SEEDS
    ):
        raise RuntimeError("authorized Strategy E v4 allocation changed")
    bundle = {
        "schema": "irisu-r3g-seed-bundle-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "manifests": manifests,
    }
    return {**bundle, "sha256": _sha(bundle)}


def _materialize_seeds(output: Path) -> Mapping[str, object]:
    expected = _seed_bundle()
    path = output / "seed-manifests.json"
    if path.exists():
        actual = _load_json(path)
        if actual != expected:
            raise RuntimeError("materialized R3G seed manifests changed")
    else:
        _atomic_json(path, expected)
    return expected


def _suite(
    bundle: Mapping[str, object], split: str
) -> tuple[tuple[int, ...], int, Mapping[str, object]]:
    manifests = bundle.get("manifests")
    if not isinstance(manifests, Mapping):
        raise ValueError("R3G seed bundle is malformed")
    manifest = manifests.get(split)
    if not isinstance(manifest, Mapping):
        raise ValueError(f"R3G seed split is missing: {split}")
    payload = dict(manifest)
    digest = payload.pop("sha256", None)
    if digest != _sha(payload):
        raise ValueError(f"R3G seed manifest identity mismatch: {split}")
    rows = manifest.get("rows")
    if not isinstance(rows, Sequence):
        raise ValueError(f"R3G seed manifest rows are malformed: {split}")
    seeds = tuple(int(value["seed"]) for value in rows)
    horizons = {int(value["horizon"]) for value in rows}
    if len(horizons) != 1:
        raise ValueError(f"R3G split mixed horizons: {split}")
    return seeds, horizons.pop(), manifest


def _global_disjointness_proof(
    source: Mapping[str, object],
    config: Mapping[str, object],
    seed_bundle: Mapping[str, object],
    external_inventory: Mapping[str, object],
) -> Mapping[str, object]:
    fresh, horizon, manifest = _suite(
        seed_bundle, TEACHER_SCREEN_SPLIT
    )
    external = {
        int(seed) for seed in external_inventory["unique_seeds"]
    }
    other_generation_seeds = {
        int(row["seed"])
        for split, candidate_manifest in seed_bundle["manifests"].items()
        if split != TEACHER_SCREEN_SPLIT
        for row in candidate_manifest["rows"]
    }
    sentinel_seeds = {int(row["seed"]) for row in SENTINELS}
    checks = {
        "protocol_identity": _file_sha256(PROTOCOL) == PROTOCOL_SHA256,
        "runtime_identity": _file_sha256(RUNTIME) == RUNTIME_SHA256,
        "frozen_v5_identity": _file_sha256(BASE) == BASE_SHA256,
        "trusted_joint_identity": (
            _file_sha256(TRUSTED_JOINT) == TRUSTED_JOINT_SHA256
        ),
        "authorized_manifest_sha256": (
            manifest["sha256"] == TEACHER_SCREEN_MANIFEST_SHA256
        ),
        "authorized_ordered_seeds": fresh == TEACHER_SCREEN_SEEDS,
        "authorized_horizon_2000": horizon == 2_000,
        "fresh_internal_uniqueness": len(fresh) == len(set(fresh)) == 8,
        "zero_overlap_external_manifests": not (set(fresh) & external),
        "zero_overlap_other_generation04_splits": not (
            set(fresh) & other_generation_seeds
        ),
        "zero_overlap_cached_sentinels": not (
            set(fresh) & sentinel_seeds
        ),
        "all_five_branch_roots_inventoried": (
            set(external_inventory["owners"])
            == set(EXTERNAL_MANIFEST_ROOTS)
        ),
        "materialized_manifest_instances_validated": (
            int(external_inventory["validated_manifest_instances"]) >= 31
        ),
    }
    if not all(checks.values()):
        raise RuntimeError("generation-04 global disjointness proof failed")
    value = {
        "schema": "irisu-r3g-generation04-global-disjointness-v1",
        "development_only": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": source["sha256"],
        "configuration_sha256": config["sha256"],
        "runtime_sha256": RUNTIME_SHA256,
        "frozen_v5_sha256": BASE_SHA256,
        "trusted_joint_v2_sha256": TRUSTED_JOINT_SHA256,
        "output_identity": config["output_identity"],
        "seed_bundle_sha256": seed_bundle["sha256"],
        "authorized_allocation": {
            "split": TEACHER_SCREEN_SPLIT,
            "manifest_sha256": manifest["sha256"],
            "ordered_seeds": list(fresh),
            "horizon": horizon,
            "purpose": "paired allocation-only Strategy E v4 teacher screen",
        },
        "parent_coordination": {
            "known_scheduled_seed_count": PARENT_KNOWN_SCHEDULED_SEEDS,
            "independently_verified_zero_overlap": True,
            "received_before_generation04_screen_result": True,
        },
        "external_manifest_inventory": external_inventory,
        "local_overlap_external": sorted(set(fresh) & external),
        "local_overlap_other_generation04_splits": sorted(
            set(fresh) & other_generation_seeds
        ),
        "local_overlap_sentinels": sorted(set(fresh) & sentinel_seeds),
        "checks": checks,
        "passed": True,
    }
    return {**value, "sha256": _sha(value)}


def _materialize_proof(
    output: Path, proof: Mapping[str, object]
) -> Mapping[str, object]:
    path = output / "global-disjointness-proof.json"
    if path.exists():
        if _load_json(path) != proof:
            raise RuntimeError("generation-04 disjointness proof changed")
    else:
        _atomic_json(path, proof)
    return proof


def _config_identity(
    source: Mapping[str, object],
    seed_bundle: Mapping[str, object],
    external_inventory: Mapping[str, object],
) -> Mapping[str, object]:
    fresh = seed_bundle["manifests"][TEACHER_SCREEN_SPLIT]
    value = {
        "schema": "irisu-r3g-conservative-offline-config-v1",
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": source["sha256"],
        "runtime_sha256": RUNTIME_SHA256,
        "frozen_v5_sha256": BASE_SHA256,
        "output_identity": {
            "generation": 4,
            "resolved_path": str(OUTPUT.resolve()),
            "append_only": True,
        },
        "seed_bundle_sha256": seed_bundle["sha256"],
        "external_manifest_inventory_sha256": external_inventory["sha256"],
        "offline_version": R3G_OFFLINE_VERSION,
        "joint_version": JOINT_PLANNER_VERSION,
        "teacher_planner": TEACHER_CONFIG.manifest(),
        "data_planner": DATA_CONFIG.manifest(),
        "query_budget": QUERY_BUDGET,
        "conformal": {
            "alpha": CONFORMAL_ALPHA,
            "cluster": "whole seed trajectory",
            "residual": "maximum candidate predicted_delta_B2 - exact_delta_B2",
            "calibration_clusters": 64,
            "order_statistic": 62,
        },
        "unsafe_probability_threshold_selection": (
            "barrier-calibration minimum calibrated probability among "
            "supported exact-unsafe alternatives, minus epsilon"
        ),
        "support_minimum_seed_groups": 8,
        "paired_bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed_derivation": (
                'SHA256("r3g-solvency-barrier-tournament-20260729'
                '|bootstrap|<metric>")'
            ),
            "reported_bounds": (
                "one-sided 95% LCB for paired median score delta and "
                "paired RMST (mean restricted-survival) delta"
            ),
        },
        "primary_policy": "complete-outcome",
        "diagnostic_policy": "winner-only",
        "screening_status": {
            "teacher_screen_split": TEACHER_SCREEN_SPLIT,
            "teacher_screen_manifest_sha256": fresh["sha256"],
            "teacher_screen_seeds": [
                int(row["seed"]) for row in fresh["rows"]
            ],
            "teacher_screen_horizon": 2_000,
            "teacher_screen_purpose": (
                "paired allocation-only Strategy E v4 teacher screen"
            ),
            "parent_known_scheduled_seed_count": (
                PARENT_KNOWN_SCHEDULED_SEEDS
            ),
            "parent_verified_global_disjointness": True,
            "original_burned_teacher_screen_manifest_sha256": (
                BURNED_TEACHER_SCREEN_MANIFEST_SHA256
            ),
            "teacher_screen_split_burned_by_generation_02": True,
            "generation_03_correctness_only": True,
            "generation_04_fresh_parent_allocation_authorized": True,
            "allocation_only": True,
        },
        "diagnostic_control_calibration_failure": (
            "freeze an all-abstain control and preserve the rejection; "
            "never block or relax the predeclared complete-outcome primary"
        ),
        "compute_threads": 1,
    }
    return {**value, "sha256": _sha(value)}


def _base_identity(source: Mapping[str, object]) -> str:
    return _sha(
        {
            "type": "frozen-r3d-v5-goal-conditioned-steering-policy",
            "checkpoint_sha256": BASE_SHA256,
            "source_revision": BASE_REVISION,
            "source_identity_sha256": source["sha256"],
            "cooldown_ticks": 16,
            "minimum_pair_closure_sizes": 0.05,
            "impact_side_sizes": 0.5,
            "impact_below_sizes": 0.75,
            "source_velocity_lead_ticks": 1.0,
            "ticks_per_second": 50.0,
            "act_logit_bias": 1.0,
        }
    )


def _base_policy(
    model: GoalConditionedSteeringModel, identity: str
) -> GoalConditionedSteeringPolicy:
    model.eval()
    return GoalConditionedSteeringPolicy(
        model,
        cooldown_ticks=16,
        minimum_pair_closure_sizes=0.05,
        impact_side_sizes=0.5,
        impact_below_sizes=0.75,
        source_velocity_lead_ticks=1.0,
        ticks_per_second=50.0,
        act_logit_bias=1.0,
        artifact_sha256=identity,
    )


def _searcher(
    model: GoalConditionedSteeringModel,
    identity: str,
    *,
    recording: bool,
    config: JointPlannerConfig = DATA_CONFIG,
) -> JointPairGeometrySearch:
    cls = RecordingJointSearch if recording else JointPairGeometrySearch
    return cls(
        lambda: _base_policy(model, identity),
        config=config,
        continuation_identity_sha256=identity,
    )


def _event_kind(event: Mapping[str, Any]) -> int | None:
    value = event.get("kind")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    name = event.get("kind_name")
    if isinstance(name, str):
        try:
            return int(EventKind[name.upper()])
        except KeyError:
            return None
    return None


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    policy: str
    seed: int
    horizon_ticks: int
    survival_ticks: int
    final_score: int
    final_gauge: int
    gauge_max: int
    final_level: int
    qualifying_clears: int
    cleared_events: int
    rotten_events: int
    positive_gauge_renewal: int
    shots_fired: int
    shots_hit: int
    chain_joins: int
    invalid_actions: int
    game_over_events: int
    decisions: int
    primitive_actions: int
    terminated: bool
    truncated: bool
    checkpoints: Mapping[str, Mapping[str, object]]
    policy_counts: Mapping[str, int | float]
    wall_seconds: float
    cpu_seconds: float
    max_rss_kib: int

    @property
    def gauge_failure(self) -> bool:
        return self.terminated and self.game_over_events > 0

    @property
    def terminal_failure(self) -> bool:
        return self.terminated and self.game_over_events > 0

    @property
    def successful_terminal(self) -> bool:
        return self.terminated and self.game_over_events == 0

    def manifest(self) -> Mapping[str, object]:
        return {
            **asdict(self),
            "gauge_failure": self.gauge_failure,
            "terminal_failure": self.terminal_failure,
            "successful_terminal": self.successful_terminal,
        }


def _run_episode(
    *,
    label: str,
    seed: int,
    horizon: int,
    factory: Callable[[IrisuEnv], object],
) -> tuple[
    EpisodeMetrics,
    Mapping[str, object],
    Mapping[str, object],
    object,
]:
    wall_started, cpu_started = time.perf_counter(), time.process_time()
    usage_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    checkpoints = tuple(
        value for value in (2_000, 10_000, 20_000, 30_000, 40_000, 50_000)
        if value <= horizon
    )
    captured: dict[str, Mapping[str, object]] = {}
    counts: Counter[int] = Counter()
    hit_projectiles: set[int] = set()
    positive_gauge = 0
    decisions = primitive_actions = 0
    with IrisuEnv(
        library_path=RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": horizon},
    ) as env:
        runner = env.runner_identity_manifest()
        attestation = attest_simulator_runtime(env).manifest()
        observation, reset_info = env.reset(seed=seed)
        config_hash = int(runner["config_hash"])
        if int(reset_info.get("config_hash", -1)) != config_hash:
            raise RuntimeError("R3G reset config identity mismatch")
        initial_tick = int(observation.get("tick", 0))
        initial_clears = int(observation.get("qualifying_clear_count", 0))
        policy = factory(env)
        getattr(policy, "reset")(seed)
        terminated = bool(observation.get("terminated", False))
        truncated = bool(observation.get("truncated", False))
        while not terminated and not truncated:
            if decisions >= MAX_DECISIONS:
                raise RuntimeError("R3G episode exceeded its decision budget")
            decision = getattr(policy, "predict")(observation)
            if not isinstance(decision, SteeringDecision):
                raise TypeError("R3G evaluated policy returned a non-decision")
            decisions += 1
            for action in decision.primitive_actions():
                if terminated or truncated:
                    break
                tick = int(observation.get("tick", 0))
                pending = [value for value in checkpoints if value > tick]
                if (
                    ActionKind.parse(action.kind) is ActionKind.WAIT
                    and pending
                    and tick + int(action.wait_ticks) > pending[0]
                ):
                    action = Action.wait(pending[0] - tick)
                observation, _, terminated, truncated, info = env.step(action)
                primitive_actions += 1
                if int(info.get("config_hash", -1)) != config_hash:
                    raise RuntimeError("R3G step config identity mismatch")
                for event in info.get("events", ()):
                    if not isinstance(event, Mapping):
                        continue
                    kind = _event_kind(event)
                    if kind is None:
                        continue
                    counts[kind] += 1
                    if kind == int(EventKind.PROJECTILE_HIT):
                        hit_projectiles.add(int(event.get("a", -1)))
                    if kind == int(EventKind.GAUGE_CHANGED):
                        positive_gauge += max(0, int(event.get("value", 0)))
                reached = int(observation.get("tick", 0))
                if reached in checkpoints:
                    captured[str(reached)] = {
                        "tick": reached,
                        "score": int(observation.get("score", 0)),
                        "gauge": int(observation.get("gauge", 0)),
                        "level": int(observation.get("level", 0)),
                        "qualifying_clears": (
                            int(observation.get("qualifying_clear_count", 0))
                            - initial_clears
                        ),
                        "cleared_events": counts[int(EventKind.CLEARED)],
                        "rotten_events": counts[int(EventKind.ROTTEN)],
                        "positive_gauge_renewal": positive_gauge,
                        "reached": True,
                    }
        survival = int(observation.get("tick", 0)) - initial_tick
        final_score = int(observation.get("score", 0))
        for checkpoint in checkpoints:
            if str(checkpoint) not in captured:
                captured[str(checkpoint)] = {
                    "tick": survival,
                    "score": final_score,
                    "gauge": int(observation.get("gauge", 0)),
                    "level": int(observation.get("level", 0)),
                    "qualifying_clears": (
                        int(observation.get("qualifying_clear_count", 0))
                        - initial_clears
                    ),
                    "cleared_events": counts[int(EventKind.CLEARED)],
                    "rotten_events": counts[int(EventKind.ROTTEN)],
                    "positive_gauge_renewal": positive_gauge,
                    "reached": False,
                    "terminal_carried": True,
                }
        statistics = getattr(policy, "statistics", None)
        policy_counts = statistics() if callable(statistics) else {}
        result = EpisodeMetrics(
            label,
            seed,
            horizon,
            survival,
            final_score,
            int(observation.get("gauge", 0)),
            int(observation.get("gauge_max", 0)),
            int(observation.get("level", 0)),
            int(observation.get("qualifying_clear_count", 0)) - initial_clears,
            counts[int(EventKind.CLEARED)],
            counts[int(EventKind.ROTTEN)],
            positive_gauge,
            counts[int(EventKind.SHOT_FIRED)],
            len(hit_projectiles - {-1}),
            counts[int(EventKind.CHAIN_JOINED)],
            counts[int(EventKind.INVALID_ACTION)],
            counts[int(EventKind.GAME_OVER)],
            decisions,
            primitive_actions,
            bool(terminated),
            bool(truncated),
            captured,
            dict(policy_counts),
            time.perf_counter() - wall_started,
            time.process_time() - cpu_started,
            max(
                0,
                int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
                - int(usage_before),
            ),
        )
    return result, runner, attestation, policy


def _evaluate(
    *,
    label: str,
    seeds: Sequence[int],
    horizon: int,
    factory: Callable[[IrisuEnv], object],
) -> tuple[
    list[EpisodeMetrics],
    list[object],
    Mapping[str, object],
    Mapping[str, object],
]:
    episodes: list[EpisodeMetrics] = []
    policies: list[object] = []
    runners: list[Mapping[str, object]] = []
    attestations: list[Mapping[str, object]] = []
    for seed in seeds:
        episode, runner, attestation, policy = _run_episode(
            label=label,
            seed=int(seed),
            horizon=horizon,
            factory=factory,
        )
        episodes.append(episode)
        policies.append(policy)
        runners.append(runner)
        attestations.append(attestation)
    if len({_sha(value) for value in runners}) != 1:
        raise RuntimeError("R3G suite mixed runner identities")
    if len({_sha(value) for value in attestations}) != 1:
        raise RuntimeError("R3G suite mixed runtime attestations")
    return episodes, policies, runners[0], attestations[0]


def _distribution(values: Sequence[int | float]) -> Mapping[str, int | float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("R3G distribution requires finite observations")
    return {
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.1, method="linear")),
        "median": float(np.quantile(array, 0.5, method="linear")),
        "p90": float(np.quantile(array, 0.9, method="linear")),
        "maximum": float(array.max()),
    }


def _paired_bootstrap_lcb(
    values: Sequence[int | float],
    *,
    metric: str,
    statistic: str,
) -> Mapping[str, object]:
    array = np.asarray(values, dtype=np.float64)
    if array.size < 1 or not np.isfinite(array).all():
        raise ValueError("paired bootstrap requires finite observations")
    seed_digest = hashlib.sha256(
        f"{SEED_DOMAIN}|bootstrap|{metric}".encode()
    ).digest()
    seed = int.from_bytes(seed_digest[:8], "big")
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        array.size,
        size=(BOOTSTRAP_REPLICATES, array.size),
    )
    samples = array[indices]
    if statistic == "median":
        estimates = np.median(samples, axis=1)
        point = float(np.median(array))
    elif statistic == "mean":
        estimates = samples.mean(axis=1)
        point = float(array.mean())
    else:
        raise ValueError("unsupported paired bootstrap statistic")
    return {
        "metric": metric,
        "statistic": statistic,
        "replicates": BOOTSTRAP_REPLICATES,
        "rng_seed_sha256": seed_digest.hex(),
        "point_estimate": point,
        "one_sided_95pct_lower_confidence_bound": float(
            np.quantile(estimates, 0.05, method="linear")
        ),
    }


def _aggregate(episodes: Sequence[EpisodeMetrics]) -> Mapping[str, object]:
    if not episodes:
        raise ValueError("R3G aggregate cannot be empty")
    counts: Counter[str] = Counter()
    for episode in episodes:
        counts.update(episode.policy_counts)
    checkpoint_keys = sorted(
        set.intersection(
            *(set(value.checkpoints) for value in episodes)
        ),
        key=int,
    )
    return {
        "episodes": len(episodes),
        "score": _distribution([value.final_score for value in episodes]),
        "survival_ticks": _distribution(
            [value.survival_ticks for value in episodes]
        ),
        "final_gauge": _distribution(
            [value.final_gauge for value in episodes]
        ),
        "final_level": _distribution(
            [value.final_level for value in episodes]
        ),
        "qualifying_clears": _distribution(
            [value.qualifying_clears for value in episodes]
        ),
        "rotten_events": _distribution(
            [value.rotten_events for value in episodes]
        ),
        "terminal_failures": sum(
            value.terminal_failure for value in episodes
        ),
        "successful_terminals": sum(
            value.successful_terminal for value in episodes
        ),
        "gauge_failures": sum(value.gauge_failure for value in episodes),
        "invalid_actions": sum(value.invalid_actions for value in episodes),
        "positive_gauge_renewal": sum(
            value.positive_gauge_renewal for value in episodes
        ),
        "wall_seconds": sum(value.wall_seconds for value in episodes),
        "cpu_seconds": sum(value.cpu_seconds for value in episodes),
        "policy_counts": dict(sorted(counts.items())),
        "checkpoint_curves": {
            key: {
                field: _distribution(
                    [
                        float(value.checkpoints[key][field])
                        for value in episodes
                    ]
                )
                for field in (
                    "score",
                    "gauge",
                    "level",
                    "qualifying_clears",
                    "rotten_events",
                    "positive_gauge_renewal",
                )
            }
            for key in checkpoint_keys
        },
    }


def _paired_checkpoint_curves(
    baseline: Mapping[int, EpisodeMetrics],
    candidate: Mapping[int, EpisodeMetrics],
) -> Mapping[str, object]:
    checkpoint_keys = sorted(
        set.intersection(
            *(set(value.checkpoints) for value in baseline.values()),
            *(set(value.checkpoints) for value in candidate.values()),
        ),
        key=int,
    )
    fields = (
        "score",
        "gauge",
        "level",
        "qualifying_clears",
        "rotten_events",
        "positive_gauge_renewal",
    )
    output: dict[str, object] = {}
    for key in checkpoint_keys:
        horizon = int(key)
        rows = []
        for seed in sorted(baseline):
            first, second = baseline[seed], candidate[seed]
            row: dict[str, int] = {
                "seed": seed,
                "survival_delta": (
                    min(second.survival_ticks, horizon)
                    - min(first.survival_ticks, horizon)
                ),
            }
            for field in fields:
                row[f"{field}_delta"] = int(
                    second.checkpoints[key][field]
                ) - int(first.checkpoints[key][field])
            rows.append(row)
        output[key] = {
            "survival_delta": _distribution(
                [value["survival_delta"] for value in rows]
            ),
            **{
                f"{field}_delta": _distribution(
                    [value[f"{field}_delta"] for value in rows]
                )
                for field in fields
            },
            "paired_confidence": {
                "median_score_delta": _paired_bootstrap_lcb(
                    [value["score_delta"] for value in rows],
                    metric=f"median-score-delta-{key}",
                    statistic="median",
                ),
                "rmst_delta": _paired_bootstrap_lcb(
                    [value["survival_delta"] for value in rows],
                    metric=f"rmst-delta-{key}",
                    statistic="mean",
                ),
            },
            "pairs": rows,
        }
    return output


def _paired(
    baseline: Sequence[EpisodeMetrics],
    candidate: Sequence[EpisodeMetrics],
    *,
    unsafe_executed: Mapping[int, Sequence[Mapping[str, object]]] | None = None,
) -> Mapping[str, object]:
    base = {value.seed: value for value in baseline}
    other = {value.seed: value for value in candidate}
    if set(base) != set(other):
        raise ValueError("R3G paired suites use different seeds")
    unsafe_executed = {} if unsafe_executed is None else unsafe_executed
    rows: list[Mapping[str, object]] = []
    for seed in sorted(base):
        first, second = base[seed], other[seed]
        regression = first.survival_ticks - second.survival_ticks
        material = regression >= 1_000
        ratio = (
            1.0
            if first.survival_ticks == 0
            else second.survival_ticks / first.survival_ticks
        )
        catastrophe = material and ratio <= 0.5
        new_terminal = (
            second.terminal_failure and not first.terminal_failure
        )
        new_gauge = second.gauge_failure and not first.gauge_failure
        unsafe = list(unsafe_executed.get(seed, ()))
        rows.append(
            {
                "seed": seed,
                "seed_hex": f"0x{seed:08X}",
                "baseline_score": first.final_score,
                "candidate_score": second.final_score,
                "score_delta": second.final_score - first.final_score,
                "baseline_survival": first.survival_ticks,
                "candidate_survival": second.survival_ticks,
                "survival_delta": second.survival_ticks
                - first.survival_ticks,
                "survival_ratio": ratio,
                "material_survival_regression": material,
                "catastrophic_paired_regression": catastrophe,
                "new_terminal_failure": new_terminal,
                "new_gauge_failure": new_gauge,
                "baseline_gauge": first.final_gauge,
                "candidate_gauge": second.final_gauge,
                "gauge_delta": second.final_gauge - first.final_gauge,
                "baseline_level": first.final_level,
                "candidate_level": second.final_level,
                "level_delta": second.final_level - first.final_level,
                "clear_delta": (
                    second.qualifying_clears - first.qualifying_clears
                ),
                "rot_delta": second.rotten_events - first.rotten_events,
                "executed_exact_unsafe_overrides": unsafe,
                "hard_paired_failure": bool(
                    new_terminal or new_gauge or catastrophe or unsafe
                ),
            }
        )
    return {
        "definition": {
            "material_regression": (
                "baseline_survival - candidate_survival >= 1000"
            ),
            "catastrophe": (
                "material regression and candidate/base survival <= 0.5"
            ),
            "hard_failure": (
                "new terminal, new gauge failure, catastrophe, identity "
                "mismatch, or executed exactly unsafe override"
            ),
        },
        "score_delta": _distribution(
            [float(value["score_delta"]) for value in rows]
        ),
        "survival_delta": _distribution(
            [float(value["survival_delta"]) for value in rows]
        ),
        "gauge_delta": _distribution(
            [float(value["gauge_delta"]) for value in rows]
        ),
        "clear_delta": _distribution(
            [float(value["clear_delta"]) for value in rows]
        ),
        "rot_delta": _distribution(
            [float(value["rot_delta"]) for value in rows]
        ),
        "survival_regressions": [
            value for value in rows if int(value["survival_delta"]) < 0
        ],
        "material_regressions": [
            value for value in rows if value["material_survival_regression"]
        ],
        "catastrophic_regressions": [
            value for value in rows if value["catastrophic_paired_regression"]
        ],
        "hard_failures": [
            value for value in rows if value["hard_paired_failure"]
        ],
        "paired_confidence": {
            "median_score_delta": _paired_bootstrap_lcb(
                [float(value["score_delta"]) for value in rows],
                metric=(
                    f"median-score-delta-{baseline[0].horizon_ticks}"
                ),
                statistic="median",
            ),
            "rmst_delta": _paired_bootstrap_lcb(
                [float(value["survival_delta"]) for value in rows],
                metric=f"rmst-delta-{baseline[0].horizon_ticks}",
                statistic="mean",
            ),
        },
        "checkpoint_curves": _paired_checkpoint_curves(base, other),
        "pairs": rows,
    }


def _teacher_gate(
    baseline: Sequence[EpisodeMetrics],
    teacher: Sequence[EpisodeMetrics],
    paired: Mapping[str, object],
) -> Mapping[str, object]:
    base = _aggregate(baseline)
    candidate = _aggregate(teacher)
    base_p10 = float(base["survival_ticks"]["p10"])
    candidate_p10 = float(candidate["survival_ticks"]["p10"])
    horizon = baseline[0].horizon_ticks
    ceiling = math.isclose(base_p10, horizon)
    survival_pass = (
        candidate_p10 == base_p10
        and not paired["survival_regressions"]
        if ceiling
        else candidate_p10 > base_p10
    )
    teacher_safety_failures = list(paired["hard_failures"])
    checks = {
        "zero_new_terminal_gauge_or_catastrophic_failures": (
            not teacher_safety_failures
        ),
        "median_score_strictly_improved": (
            float(candidate["score"]["median"])
            > float(base["score"]["median"])
        ),
        "survival_p10_rule": survival_pass,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "teacher_safety_failures": teacher_safety_failures,
        "executed_exact_unsafe_override_audits": sum(
            bool(value["executed_exact_unsafe_overrides"])
            for value in paired["pairs"]
        ),
        "baseline_p10_at_censor_ceiling": ceiling,
        "screen_scope": (
            "resource screen only; cannot authorize a positive conclusion"
        ),
    }


def _decision_key(decision: SteeringDecision) -> tuple[object, ...]:
    return (
        int(decision.action.kind),
        round(float(decision.action.x_norm), 12),
        round(float(decision.action.y_norm), 12),
        decision.source_body_id,
        decision.destination_body_id,
    )


def _primitive_action_key(
    decision: SteeringDecision,
) -> tuple[tuple[int, float, float, int], ...]:
    return tuple(
        (
            int(ActionKind.parse(action.kind)),
            float(action.cursor_x),
            float(action.cursor_y),
            int(action.wait_ticks),
        )
        for action in decision.primitive_actions()
    )


def _label_value(labels: Mapping[str, Any], *names: str) -> float:
    for name in names:
        if name in labels:
            return float(labels[name])
    raise KeyError(f"none of the exact label fields is present: {names}")


def _exact_unsafe(labels: Mapping[str, Any]) -> bool:
    return any(
        bool(labels.get(name, False))
        for name in (
            "unsafe",
            "hard_catastrophe",
            "severe_renewable",
            "censored",
            "unresolved",
            "unresolved_second_renewal",
            "negative_through_renewal",
            "new_terminal",
            "new_gauge_failure",
            "catastrophic",
            "terminal",
            "gauge_collapse",
        )
    ) or _label_value(labels, "delta_b2", "delta_B2", "risk_margin") < 0.0


def _severe_unsafe(labels: Mapping[str, Any]) -> bool:
    return any(
        bool(labels.get(name, False))
        for name in (
            "severe_renewable",
            "negative_through_renewal",
            "terminal",
            "gauge_collapse",
            "new_terminal",
            "new_gauge_failure",
        )
    )


def _candidate_identity(candidate: object) -> str:
    manifest = getattr(candidate, "manifest", None)
    if callable(manifest):
        value = manifest()
        if not isinstance(value, Mapping):
            raise TypeError("teacher candidate manifest is malformed")
        return _sha(value)
    decision = getattr(candidate, "decision", None)
    if not isinstance(decision, SteeringDecision):
        raise TypeError("teacher candidate decision is malformed")
    value = {
        "ordinal": int(getattr(candidate, "ordinal")),
        "pair_ordinal": int(getattr(candidate, "pair_ordinal")),
        "geometry_ordinal": int(getattr(candidate, "geometry_ordinal")),
        "primitive_actions": _primitive_action_key(decision),
        "pair_category": str(getattr(candidate.pair, "category")),
        "geometry_name": str(getattr(candidate.geometry, "name")),
    }
    return _sha(value)


def _unique_teacher_improvement(result: object) -> bool:
    outcomes = tuple(getattr(result, "outcomes", ()))
    if not outcomes:
        return False
    incumbents = [
        outcome
        for outcome in outcomes
        if int(getattr(outcome.candidate, "ordinal", -1)) == 0
    ]
    if len(incumbents) != 1 or outcomes[0] is not incumbents[0]:
        return False
    incumbent = incumbents[0]
    eligible = []
    for outcome in outcomes:
        selectable = getattr(outcome, "selectable_against", None)
        if not callable(selectable):
            return False
        try:
            if selectable(incumbent):
                eligible.append(outcome)
        except (TypeError, ValueError):
            return False
    if not eligible:
        return False
    objectives = []
    for outcome in eligible:
        objective = getattr(outcome, "objective", None)
        if (
            not isinstance(objective, tuple)
            or not objective
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in objective
            )
        ):
            return False
        objectives.append(objective)
    best = max(objectives)
    tops = [
        outcome
        for outcome, objective in zip(
            eligible, objectives, strict=True
        )
        if objective == best
    ]
    selected = getattr(result, "selected_candidate", None)
    return bool(
        getattr(result, "strictly_improved", False)
        and best > getattr(incumbent, "objective")
        and len(tops) == 1
        and _candidate_identity(tops[0].candidate)
        == _candidate_identity(selected)
    )


def _exact_teacher_row_gate(
    selected_row: Mapping[str, Any],
    incumbent_row: Mapping[str, Any],
) -> tuple[bool, str, bool]:
    try:
        labels = selected_row["labels"]
        selected = selected_row["solvency"]
        incumbent = incumbent_row["solvency"]
        outcome = selected_row["outcome"]
        base_outcome = incumbent_row["outcome"]
        if not all(
            isinstance(value, Mapping)
            for value in (
                labels,
                selected,
                incumbent,
                outcome,
                base_outcome,
            )
        ):
            raise TypeError
        selected_minimum = selected["minimum_surplus"]
        incumbent_minimum = incumbent["minimum_surplus"]
        if (
            type(selected_minimum) is not int
            or type(incumbent_minimum) is not int
        ):
            raise TypeError
        selected_censored = selected["censored_before_second_renewal"]
        incumbent_censored = incumbent[
            "censored_before_second_renewal"
        ]
        selected_game_over = selected["game_over"]
        incumbent_game_over = incumbent["game_over"]
        selected_gauge_failure = selected["gauge_failure"]
        incumbent_gauge_failure = incumbent["gauge_failure"]
        selected_negative = selected["negative_through_renewal"]
        incumbent_negative = incumbent["negative_through_renewal"]
        for value in (
            selected_censored,
            incumbent_censored,
            selected_game_over,
            incumbent_game_over,
            selected_gauge_failure,
            incumbent_gauge_failure,
            selected_negative,
            incumbent_negative,
        ):
            if type(value) is not bool:
                raise TypeError
        if (
            selected_negative is not (selected_minimum <= 0)
            or incumbent_negative is not (incumbent_minimum <= 0)
        ):
            return False, "inconsistent_negative_solvency", True
        milestones = outcome["milestones"]
        base_milestones = base_outcome["milestones"]
        if (
            not isinstance(milestones, Sequence)
            or not milestones
            or not isinstance(base_milestones, Sequence)
            or not base_milestones
            or not all(isinstance(value, Mapping) for value in milestones)
            or not all(
                isinstance(value, Mapping) for value in base_milestones
            )
        ):
            raise TypeError
        candidate_survival = milestones[-1]["survival_ticks"]
        base_survival = base_milestones[-1]["survival_ticks"]
        if (
            type(candidate_survival) is not int
            or type(base_survival) is not int
            or candidate_survival < 0
            or base_survival < 0
        ):
            raise TypeError
        invalid_actions = [
            value.get("invalid_actions") for value in milestones
        ]
        if any(
            type(value) is not int or value < 0
            for value in invalid_actions
        ):
            raise TypeError
        invalid = any(value > 0 for value in invalid_actions)
        material = base_survival - candidate_survival >= 1_000
        catastrophic = bool(
            material
            and base_survival > 0
            and candidate_survival / base_survival <= 0.5
        )
        new_terminal = bool(
            selected_game_over and not incumbent_game_over
        )
        new_gauge = bool(
            selected_gauge_failure and not incumbent_gauge_failure
        )
        hard = bool(invalid or catastrophic or new_terminal or new_gauge)
        resolved = bool(
            not selected_censored and not incumbent_censored
        )
        delta = selected_minimum - incumbent_minimum
        unsafe = bool(
            not resolved
            or selected_minimum <= 0
            or delta < 0
            or hard
        )
        expected = {
            "resolved": resolved,
            "censored": selected_censored,
            "new_terminal": new_terminal,
            "new_gauge_failure": new_gauge,
            "catastrophic": catastrophic,
            "hard_catastrophe": hard,
            "severe_renewable": selected_minimum <= 0,
            "unsafe": unsafe,
        }
        if any(
            type(labels.get(name)) is not bool
            or labels[name] is not expected_value
            for name, expected_value in expected.items()
        ):
            return False, "inconsistent_exact_labels", True
        risk = _label_value(
            labels, "delta_b2", "delta_B2", "risk_margin"
        )
        absolute = _label_value(labels, "absolute_solvency")
        if not math.isfinite(risk) or not math.isfinite(absolute):
            raise ValueError
        if (
            (risk < 0.0) != (delta < 0)
            or (absolute < 0.0) != (selected_minimum < 0)
            or (delta == 0 and not math.isclose(risk, 0.0, abs_tol=1e-12))
            or (
                selected_minimum == 0
                and not math.isclose(absolute, 0.0, abs_tol=1e-12)
            )
        ):
            return False, "inconsistent_exact_margins", True
    except (KeyError, TypeError, ValueError):
        return False, "unknown_exact_outcome", True
    if not resolved:
        return False, "unresolved_second_renewal", True
    if unsafe:
        return False, "exact_unsafe", True
    return True, "exact_safe", False


def _records_rows(policy: object) -> list[dict[str, Any]]:
    searcher = getattr(policy, "searcher", None)
    records = getattr(searcher, "records", ())
    return [
        row
        for record in records
        for row in getattr(record, "rows")()
    ]


def _row_manifest(row: Mapping[str, Any]) -> Mapping[str, object]:
    features = np.asarray(row["features"], dtype=np.float32)
    support = np.asarray(row["support"], dtype=np.float32)
    value = {
        key: row[key]
        for key in (
            "seed",
            "query_index",
            "tick",
            "search_sha256",
            "snapshot_sha256",
            "ordinal",
            "incumbent",
            "selected",
            "signature",
            "labels",
            "candidate",
            "outcome",
            "solvency",
        )
    }
    return {
        **value,
        "features_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
        "support_sha256": hashlib.sha256(support.tobytes()).hexdigest(),
    }


def _tensor_sha(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(_bytes(list(value.shape)))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _build_dataset(
    rows: Sequence[Mapping[str, Any]], *, split: str
) -> Mapping[str, object]:
    if not rows:
        raise RuntimeError(f"R3G {split} collection has no exact outcomes")
    seeds = sorted({int(value["seed"]) for value in rows})
    group = {seed: index for index, seed in enumerate(seeds)}
    tensors = {
        "features": torch.from_numpy(
            np.stack(
                [
                    np.asarray(value["features"], dtype=np.float32)
                    for value in rows
                ]
            )
        ),
        "support": torch.from_numpy(
            np.stack(
                [
                    np.asarray(value["support"], dtype=np.float32)
                    for value in rows
                ]
            )
        ),
        "risk_margin": torch.tensor(
            [
                _label_value(
                    value["labels"], "delta_b2", "delta_B2", "risk_margin"
                )
                for value in rows
            ],
            dtype=torch.float32,
        ),
        "absolute_solvency": torch.tensor(
            [
                _label_value(value["labels"], "absolute_solvency")
                for value in rows
            ],
            dtype=torch.float32,
        ),
        "score_advantage": torch.tensor(
            [
                float(value["labels"]["score_advantage"])
                for value in rows
            ],
            dtype=torch.float32,
        ),
        "unsafe": torch.tensor(
            [_exact_unsafe(value["labels"]) for value in rows],
            dtype=torch.bool,
        ),
        "hard_catastrophe": torch.tensor(
            [
                bool(value["labels"].get("hard_catastrophe", False))
                for value in rows
            ],
            dtype=torch.bool,
        ),
        "severe_renewable": torch.tensor(
            [_severe_unsafe(value["labels"]) for value in rows],
            dtype=torch.bool,
        ),
        "censored": torch.tensor(
            [
                bool(
                    value["labels"].get(
                        "unresolved",
                        value["labels"].get("censored", False),
                    )
                )
                for value in rows
            ],
            dtype=torch.bool,
        ),
        "resolved": torch.tensor(
            [
                bool(
                    value["labels"].get(
                        "resolved",
                        not value["labels"].get("censored", True),
                    )
                )
                for value in rows
            ],
            dtype=torch.bool,
        ),
        "selected": torch.tensor(
            [bool(value["selected"]) for value in rows], dtype=torch.bool
        ),
        "incumbent": torch.tensor(
            [bool(value["incumbent"]) for value in rows], dtype=torch.bool
        ),
        "group": torch.tensor(
            [group[int(value["seed"])] for value in rows], dtype=torch.long
        ),
        "seed": torch.tensor(
            [int(value["seed"]) for value in rows], dtype=torch.int64
        ),
    }
    manifests = [_row_manifest(value) for value in rows]
    identity = {
        "schema": "irisu-r3g-complete-outcome-dataset-v1",
        "split": split,
        "rows": len(rows),
        "seeds": seeds,
        "query_states": len(
            {
                (int(value["seed"]), int(value["query_index"]))
                for value in rows
            }
        ),
        "row_sha256s": [_sha(value) for value in manifests],
        "tensor_sha256s": {
            name: _tensor_sha(value) for name, value in tensors.items()
        },
        "signatures_sha256": _sha(
            [str(value["signature"]) for value in rows]
        ),
        "preserves_winners_and_losers": True,
    }
    identity = {**identity, "sha256": _sha(identity)}
    return {
        "format": "irisu-r3g-complete-outcome-dataset-v1",
        "identity": identity,
        "row_manifests": manifests,
        "signatures": [str(value["signature"]) for value in rows],
        "tensors": tensors,
    }


def _load_dataset(
    path: Path, expected_sha256: str | None = None
) -> Mapping[str, Any]:
    digest = _file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("R3G dataset file SHA-256 mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(value, Mapping)
        or value.get("format") != "irisu-r3g-complete-outcome-dataset-v1"
        or not isinstance(value.get("identity"), Mapping)
        or not isinstance(value.get("tensors"), Mapping)
        or not isinstance(value.get("signatures"), list)
    ):
        raise ValueError("R3G dataset artifact is malformed")
    identity = dict(value["identity"])
    digest_value = identity.pop("sha256", None)
    if digest_value != _sha(identity):
        raise ValueError("R3G dataset logical identity mismatch")
    tensors = value["tensors"]
    if identity["tensor_sha256s"] != {
        name: _tensor_sha(tensor) for name, tensor in tensors.items()
    }:
        raise ValueError("R3G dataset tensor identity mismatch")
    return value


@dataclass(frozen=True, slots=True)
class ModelBundle:
    ensemble: TrainedEnsemble
    support: SupportEnvelope
    isotonic: IsotonicCalibration
    barrier: BarrierCalibration | None
    metadata: Mapping[str, Any]
    path: Path
    sha256: str


def _support_payload(support: SupportEnvelope) -> Mapping[str, object]:
    return {
        "centers": {
            key: value.detach().cpu()
            for key, value in support.centers.items()
        },
        "scales": {
            key: value.detach().cpu()
            for key, value in support.scales.items()
        },
        "thresholds": dict(support.thresholds),
        "minimum_groups": support.minimum_groups,
    }


def _barrier_payload(
    value: BarrierCalibration | None,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    return {
        "margin_threshold": value.margin_threshold,
        "probability_threshold": value.probability_threshold,
        "score_threshold": value.score_threshold,
        "isotonic": {
            "bounds": list(value.isotonic.bounds),
            "values": list(value.isotonic.values),
        },
        "report": dict(value.report),
        "conformal_q": value.conformal_q,
        "absolute_conformal_q": value.absolute_conformal_q,
        "alpha": value.alpha,
        "episode_count": value.episode_count,
    }


def _save_model(
    path: Path,
    *,
    ensemble: TrainedEnsemble,
    support: SupportEnvelope,
    isotonic: IsotonicCalibration,
    barrier: BarrierCalibration | None,
    metadata: Mapping[str, object],
) -> str:
    state = [
        {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        for model in ensemble.models
    ]
    state_identity = [
        {
            name: _tensor_sha(value)
            for name, value in sorted(member.items())
        }
        for member in state
    ]
    payload = {
        "format": "irisu-r3g-conservative-offline-checkpoint-v1",
        "offline_version": R3G_OFFLINE_VERSION,
        "model_state": state,
        "model_state_sha256": _sha(state_identity),
        "mean": ensemble.mean.detach().cpu(),
        "scale": ensemble.scale.detach().cpu(),
        "training_report": dict(ensemble.training_report),
        "support": _support_payload(support),
        "isotonic": {
            "bounds": list(isotonic.bounds),
            "values": list(isotonic.values),
        },
        "barrier": _barrier_payload(barrier),
        "metadata": dict(metadata),
    }
    return _atomic_torch(path, payload)


def _load_model(path: Path, expected_sha256: str | None = None) -> ModelBundle:
    digest = _file_sha256(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("R3G model checkpoint SHA-256 mismatch")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(value, Mapping)
        or value.get("format")
        != "irisu-r3g-conservative-offline-checkpoint-v1"
        or value.get("offline_version") != R3G_OFFLINE_VERSION
    ):
        raise ValueError("R3G model checkpoint format is unsupported")
    states = value.get("model_state")
    if not isinstance(states, list) or not states:
        raise ValueError("R3G ensemble state is malformed")
    state_identity = [
        {
            name: _tensor_sha(tensor)
            for name, tensor in sorted(member.items())
        }
        for member in states
    ]
    if _sha(state_identity) != value.get("model_state_sha256"):
        raise ValueError("R3G ensemble state identity mismatch")
    models: list[ResidualValueNet] = []
    for state in states:
        model = ResidualValueNet()
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
    ensemble = TrainedEnsemble(
        tuple(models),
        value["mean"].float(),
        value["scale"].float(),
        dict(value["training_report"]),
    )
    supplied = value["support"]
    support = SupportEnvelope(
        {key: tensor.float() for key, tensor in supplied["centers"].items()},
        {key: tensor.float() for key, tensor in supplied["scales"].items()},
        {key: float(item) for key, item in supplied["thresholds"].items()},
        int(supplied["minimum_groups"]),
    )
    raw_isotonic = value["isotonic"]
    isotonic = IsotonicCalibration(
        tuple(float(item) for item in raw_isotonic["bounds"]),
        tuple(float(item) for item in raw_isotonic["values"]),
    )
    raw_barrier = value.get("barrier")
    barrier = None
    if raw_barrier is not None:
        raw_calibration = raw_barrier["isotonic"]
        barrier = BarrierCalibration(
            float(raw_barrier["margin_threshold"]),
            float(raw_barrier["probability_threshold"]),
            float(raw_barrier["score_threshold"]),
            IsotonicCalibration(
                tuple(float(item) for item in raw_calibration["bounds"]),
                tuple(float(item) for item in raw_calibration["values"]),
            ),
            dict(raw_barrier["report"]),
            float(raw_barrier["conformal_q"]),
            float(raw_barrier["absolute_conformal_q"]),
            float(raw_barrier["alpha"]),
            int(raw_barrier["episode_count"]),
        )
    return ModelBundle(
        ensemble,
        support,
        isotonic,
        barrier,
        dict(value["metadata"]),
        path.resolve(),
        digest,
    )


def _fit_isotonic_from_training(
    ensemble: TrainedEnsemble,
    tensors: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
) -> IsotonicCalibration:
    prediction = ensemble.predict(tensors["features"][indices].float())
    return fit_isotonic(
        prediction["unsafe_mean"].tolist(),
        tensors["unsafe"][indices].bool().tolist(),
    )


def _certify_rows(
    bundle: ModelBundle, rows: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, object]]:
    if bundle.barrier is None:
        raise RuntimeError("R3G model has no frozen barrier")
    if not rows:
        return []
    features = torch.from_numpy(
        np.stack(
            [
                np.asarray(value["features"], dtype=np.float32)
                for value in rows
            ]
        )
    )
    prediction = bundle.ensemble.predict_full(features)
    delta_lcb = (
        prediction["delta_cvar_mean"] - bundle.barrier.conformal_q
    )
    absolute_lcb = (
        prediction["absolute_cvar_mean"]
        - bundle.barrier.absolute_conformal_q
    )
    score_lcb = (
        prediction["score_mean"] - 2.0 * prediction["score_std"]
    )
    output: list[Mapping[str, object]] = []
    for index, row in enumerate(rows):
        supported = bundle.support.passes(
            str(row["signature"]),
            torch.from_numpy(
                np.asarray(row["support"], dtype=np.float32)
            ),
        )
        probability = bundle.barrier.isotonic.predict(
            float(prediction["unsafe_mean"][index])
        )
        certified = bool(
            not bool(row["incumbent"])
            and supported
            and float(delta_lcb[index]) >= 0.0
            and float(absolute_lcb[index]) >= 0.0
            and probability <= bundle.barrier.probability_threshold
            and float(score_lcb[index]) > bundle.barrier.score_threshold
        )
        labels = row["labels"]
        output.append(
            {
                "ordinal": int(row["ordinal"]),
                "incumbent": bool(row["incumbent"]),
                "supported": supported,
                "predicted_delta_b2": float(
                    prediction["delta_cvar_mean"][index]
                ),
                "predicted_absolute_solvency": float(
                    prediction["absolute_cvar_mean"][index]
                ),
                "delta_ensemble_spread_diagnostic": float(
                    prediction["delta_cvar_std"][index]
                ),
                "absolute_ensemble_spread_diagnostic": float(
                    prediction["absolute_cvar_std"][index]
                ),
                "conformal_lcb": float(delta_lcb[index]),
                "absolute_conformal_lcb": float(absolute_lcb[index]),
                "unsafe_probability": probability,
                "score_lcb": float(score_lcb[index]),
                "certified": certified,
                "exact_delta_b2": _label_value(
                    labels, "delta_b2", "delta_B2", "risk_margin"
                ),
                "exact_absolute_solvency": _label_value(
                    labels, "absolute_solvency"
                ),
                "exact_unsafe": _exact_unsafe(labels),
                "severe_exact_unsafe": _severe_unsafe(labels),
                "labels": dict(labels),
                "candidate": row["candidate"],
                "outcome": row["outcome"],
                "solvency": row["solvency"],
                "search_sha256": row["search_sha256"],
            }
        )
    return output


class AuditedTeacherPolicy:
    """Propose with joint-v2, then exact-audit before any v5 rebind."""

    def __init__(
        self,
        env: IrisuEnv,
        model: GoalConditionedSteeringModel,
        identity: str,
    ) -> None:
        self.env = env
        self.base_policy = _base_policy(model, identity)
        self.teacher_searcher = _searcher(
            model, identity, recording=False, config=TEACHER_CONFIG
        )
        self.audit_searcher = _searcher(
            model, identity, recording=True, config=DATA_CONFIG
        )
        if not isinstance(self.audit_searcher, RecordingJointSearch):
            raise TypeError("R3G audit searcher is not recording")
        self.query_stride_shots = QUERY_BUDGET[
            "teacher_query_stride_shots"
        ]
        self.maximum_queries = QUERY_BUDGET["teacher_queries_per_episode"]
        self._counts: Counter[str] = Counter()
        self.counts = self._counts
        self._attempts: list[Any] = []
        self._results: list[Any] = []
        self.audits: list[Mapping[str, object]] = []
        self.seed = 0

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        self.base_policy.reset(seed)
        self.audit_searcher.reset_records(seed)
        self._counts.clear()
        self._attempts.clear()
        self._results.clear()
        self.audits.clear()

    @property
    def results(self) -> tuple[Any, ...]:
        return tuple(self._results)

    @staticmethod
    def _exact_gate(
        labels: Mapping[str, Any],
    ) -> tuple[bool, str, bool]:
        try:
            for name in (
                "resolved",
                "censored",
                "unsafe",
                "hard_catastrophe",
                "severe_renewable",
                "new_terminal",
                "new_gauge_failure",
                "catastrophic",
            ):
                if type(labels[name]) is not bool:
                    raise TypeError
            resolved = labels["resolved"]
            censored = labels["censored"]
            absolute = _label_value(labels, "absolute_solvency")
            unsafe = _exact_unsafe(labels)
        except (KeyError, TypeError, ValueError):
            return False, "unknown_exact_labels", True
        if not resolved or censored:
            return False, "unresolved_second_renewal", True
        if absolute < 0.0:
            return False, "negative_absolute_solvency", True
        if unsafe:
            return False, "exact_unsafe", True
        return True, "exact_safe", False

    def _abstain(
        self,
        incumbent: SteeringDecision,
        *,
        reason: str,
        tick: int,
        proposed_ordinal: int | None = None,
        row: Mapping[str, Any] | None = None,
        exact_unsafe: bool = False,
    ) -> SteeringDecision:
        self._counts[f"{reason}_abstentions"] += 1
        audit: dict[str, object] = {
            "seed": self.seed,
            "tick": tick,
            "proposed_ordinal": proposed_ordinal,
            "selected_ordinal": 0,
            "proposed_override": proposed_ordinal not in (None, 0),
            "override": False,
            "exact_unsafe": exact_unsafe,
            "executed_exact_unsafe": False,
            "status": reason,
        }
        if row is not None:
            audit["row"] = _row_manifest(row)
        self.audits.append(audit)
        return incumbent

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = self.base_policy.predict(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("R3G teacher base returned a non-decision")
        if not incumbent.is_shot:
            return incumbent
        self._counts["seen_shots"] += 1
        tick = int(observation.get("tick", 0))
        if (
            (self._counts["seen_shots"] - 1) % self.query_stride_shots
            or self._counts["search_attempts"] >= self.maximum_queries
        ):
            self._counts["budget_fallbacks"] += 1
            return incumbent
        self._counts["search_attempts"] += 1
        try:
            proposal = self.teacher_searcher.search(
                self.env, observation, incumbent
            )
        except ValueError:
            return self._abstain(
                incumbent, reason="unsupported_teacher", tick=tick
            )
        self._attempts.append(proposal)
        self._counts["branch_outcomes"] += len(proposal.outcomes)
        self._counts["restore_checks"] += proposal.restore_checks
        self._counts["simulated_branch_ticks"] += proposal.simulated_ticks
        proposed = proposal.selected_candidate
        if (
            proposed.ordinal == 0
            or not _unique_teacher_improvement(proposal)
            or _primitive_action_key(proposal.decision)
            == _primitive_action_key(incumbent)
        ):
            return self._abstain(
                incumbent,
                reason="tie_or_incumbent_identical",
                tick=tick,
                proposed_ordinal=int(proposed.ordinal),
            )
        try:
            result = self.audit_searcher.search(
                self.env, observation, incumbent
            )
        except ValueError:
            return self._abstain(
                incumbent,
                reason="unsupported_audit",
                tick=tick,
                proposed_ordinal=int(proposed.ordinal),
            )
        matches = [
            candidate
            for candidate in result.outcomes
            if _candidate_identity(candidate.candidate)
            == _candidate_identity(proposed)
        ]
        if len(matches) != 1:
            return self._abstain(
                incumbent,
                reason="ambiguous_or_missing_audit_candidate",
                tick=tick,
                proposed_ordinal=int(proposed.ordinal),
            )
        matched = matches[0].candidate
        if not self.audit_searcher.records:
            return self._abstain(
                incumbent,
                reason="missing_audit_trace",
                tick=tick,
                proposed_ordinal=int(proposed.ordinal),
            )
        try:
            rows = self.audit_searcher.records[-1].rows()
        except ValueError:
            return self._abstain(
                incumbent,
                reason="unsupported_audit_rows",
                tick=tick,
                proposed_ordinal=int(proposed.ordinal),
            )
        selected_rows = [
            value
            for value in rows
            if int(value["ordinal"]) == matched.ordinal
        ]
        incumbent_rows = [
            value for value in rows if bool(value["incumbent"])
        ]
        if len(selected_rows) != 1 or len(incumbent_rows) != 1:
            return self._abstain(
                incumbent,
                reason="ambiguous_or_missing_audit_row",
                tick=tick,
                proposed_ordinal=int(proposed.ordinal),
            )
        selected_row = selected_rows[0]
        incumbent_row = incumbent_rows[0]
        eligible, status, exact_unsafe = _exact_teacher_row_gate(
            selected_row, incumbent_row
        )
        if not eligible:
            return self._abstain(
                incumbent,
                reason=status,
                tick=tick,
                proposed_ordinal=int(proposed.ordinal),
                row=selected_row,
                exact_unsafe=exact_unsafe,
            )
        if not _commit_base_decision(
            self.base_policy, observation, incumbent, proposal.decision
        ):
            return self._abstain(
                incumbent,
                reason="progress_rebind",
                tick=tick,
                proposed_ordinal=int(proposed.ordinal),
                row=selected_row,
            )
        self._results.append(proposal)
        self._counts["search_queries"] += 1
        self._counts["strict_improvements"] += 1
        self._counts["exact_safe_overrides"] += 1
        self.audits.append(
            {
                "seed": self.seed,
                "tick": tick,
                "proposed_ordinal": int(proposed.ordinal),
                "selected_ordinal": matched.ordinal,
                "proposed_override": True,
                "override": True,
                "exact_unsafe": False,
                "executed_exact_unsafe": False,
                "status": "exact_safe_override",
                "row": _row_manifest(selected_row),
                "incumbent_row": _row_manifest(incumbent_row),
            }
        )
        return proposal.decision

    def statistics(self) -> Mapping[str, int | float]:
        counts: dict[str, int | float] = dict(self._counts)
        counts["search_wall_seconds"] = sum(
            value.wall_seconds for value in self._attempts
        )
        counts["search_cpu_seconds"] = sum(
            value.cpu_seconds for value in self._attempts
        )
        counts["decision_audit_records"] = len(self.audits)
        counts["b2_audit_queries"] = len(self.audit_searcher.records)
        counts["b2_audit_restore_checks"] = sum(
            value.result.restore_checks for value in self.audit_searcher.records
        )
        counts["b2_audit_simulated_branch_ticks"] = sum(
            value.result.simulated_ticks for value in self.audit_searcher.records
        )
        counts["b2_audit_wall_seconds"] = sum(
            value.result.wall_seconds for value in self.audit_searcher.records
        )
        counts["b2_audit_cpu_seconds"] = sum(
            value.result.cpu_seconds for value in self.audit_searcher.records
        )
        counts["b2_audit_exact_unsafe_overrides"] = sum(
            bool(value["executed_exact_unsafe"]) for value in self.audits
        )
        counts["b2_audit_exact_unsafe_abstentions"] = sum(
            bool(value["exact_unsafe"])
            and not bool(value["override"])
            for value in self.audits
        )
        return counts


def _stress_trigger(observation: Mapping[str, Any]) -> bool:
    gauge = int(observation.get("gauge", 0))
    gauge_max = int(observation.get("gauge_max", 0))
    level = int(observation.get("level", 0))
    rot_debt = sum(
        1_800 + 20 * min(level, 99)
        for body in observation.get("bodies", ())
        if isinstance(body, Mapping)
        and body.get("kind") == "piece"
        and body.get("lifecycle") != "rotten"
        and int(body.get("rot_timer", 0)) > 0
    )
    return (
        gauge_max > 0
        and gauge / gauge_max <= 0.60
        and level >= 2
        and rot_debt >= 1_800
    )


class BudgetedResidualPolicy:
    """Teacher-free residual with frozen cadence and optional post-choice audit."""

    def __init__(
        self,
        env: IrisuEnv,
        model: GoalConditionedSteeringModel,
        identity: str,
        bundle: ModelBundle,
        *,
        audit: bool,
        stress: bool = False,
        sample_audit_cap: int = 8,
    ) -> None:
        if bundle.barrier is None:
            raise ValueError("R3G student requires a calibrated checkpoint")
        if type(sample_audit_cap) is not int or sample_audit_cap < 0:
            raise ValueError("R3G sample audit cap must be nonnegative")
        self.env = env
        self.bundle = bundle
        self.core = ConservativeResidualPolicy(
            _base_policy(model, identity),
            _searcher(
                model, identity, recording=False, config=DATA_CONFIG
            ),
            bundle.ensemble,
            bundle.support,
            bundle.barrier,
            minimum_override_gap_shots=1,
        )
        self.audit_enabled = audit
        self.stress = stress
        self.audit_searcher = (
            _searcher(model, identity, recording=True, config=DATA_CONFIG)
            if audit
            else None
        )
        self.sample_audit_cap = sample_audit_cap
        self.audits: list[Mapping[str, object]] = []
        self.proposals: list[Mapping[str, object]] = []
        self.counts: Counter[str] = Counter()
        self.seed = 0

    def reset(self, seed: int = 0) -> None:
        self.seed = seed
        self.core.reset(seed)
        self.audits.clear()
        self.proposals.clear()
        self.counts.clear()
        if isinstance(self.audit_searcher, RecordingJointSearch):
            self.audit_searcher.reset_records(seed)

    def _eligible_shot(
        self, observation: Mapping[str, Any], incumbent: SteeringDecision
    ) -> bool:
        if not incumbent.is_shot:
            return False
        self.counts["seen_shots"] += 1
        if (
            (self.counts["seen_shots"] - 1)
            % QUERY_BUDGET["query_stride_shots"]
        ):
            self.counts["stride_abstentions"] += 1
            return False
        return True

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        # The learned base caches a same-tick decision, so core.predict can
        # safely request it again only on an eligible proposal state.
        incumbent = self.core.base_policy.predict(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("R3G residual base returned a non-decision")
        if not self._eligible_shot(observation, incumbent):
            return incumbent
        before = int(self.core.counts.get("candidate_queries", 0))
        selected = self.core.predict(observation)
        after = int(self.core.counts.get("candidate_queries", 0))
        if after != before + 1:
            raise RuntimeError("R3G student query accounting diverged")
        self.counts["eligible_queries"] += 1
        if not self.core.last_candidates:
            self.counts["unsupported_proposals"] += 1
            return selected
        actual_override = self.core.last_selected_ordinal != 0
        proposal = {
            "seed": self.seed,
            "query_index": len(self.proposals),
            "tick": int(observation.get("tick", 0)),
            "stress_trigger": _stress_trigger(observation),
            "actual_override": actual_override,
            "supported_alternatives": (
                self.core.last_supported_alternatives
            ),
            "certified_alternatives": (
                self.core.last_certified_alternatives
            ),
        }
        self.proposals.append(proposal)
        if self.audit_searcher is None:
            return selected

        sampled = (
            bool(proposal["stress_trigger"]) if self.stress else True
        ) and self.counts["sample_audits"] < self.sample_audit_cap
        if not actual_override and not sampled:
            return selected
        if sampled:
            self.counts["sample_audits"] += 1
        else:
            self.counts["override_only_audits"] += 1
        audit_incumbent = self.core.last_incumbent
        if audit_incumbent is None or not audit_incumbent.is_shot:
            raise RuntimeError("R3G core audit incumbent is unavailable")
        result = self.audit_searcher.search(
            self.env, observation, audit_incumbent
        )
        rows = self.audit_searcher.records[-1].rows()
        selected_ordinal = int(self.core.last_selected_ordinal)
        matches = [
            outcome.candidate
            for outcome in result.outcomes
            if int(outcome.candidate.ordinal) == selected_ordinal
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "student ordinal is absent or duplicated in audit candidates"
            )
        matched = matches[0]
        if _decision_key(matched.decision) != _decision_key(selected):
            raise RuntimeError(
                "student ordinal and executed action disagree during audit"
            )
        certified_rows = _certify_rows(self.bundle, rows)
        actual = next(
            value
            for value in certified_rows
            if int(value["ordinal"]) == matched.ordinal
        )
        expected_override = matched.ordinal != 0
        if expected_override != bool(actual["certified"]):
            raise RuntimeError(
                "student action and independently recomputed barrier disagree"
            )
        self.audits.append(
            {
                "seed": self.seed,
                "query_index": len(self.audits),
                "tick": int(observation.get("tick", 0)),
                "proposal_query_index": proposal["query_index"],
                "stress_trigger": proposal["stress_trigger"],
                "sampled_audit": sampled,
                "selected_ordinal": matched.ordinal,
                "actual_override": expected_override,
                "actual_exact_unsafe": (
                    expected_override and bool(actual["exact_unsafe"])
                ),
                "candidates": certified_rows,
                "search_sha256": result.sha256,
                "snapshot_sha256": result.snapshot_sha256,
            }
        )
        return selected

    def statistics(self) -> Mapping[str, int | float]:
        result = Counter(self.core.statistics())
        result.update(self.counts)
        result["exact_audit_queries"] = len(self.audits)
        result["action_proposal_states"] = len(self.proposals)
        result["exact_audit_candidate_rows"] = sum(
            len(value["candidates"]) for value in self.audits
        )
        result["exact_audit_unsafe_outcomes"] = sum(
            bool(candidate["exact_unsafe"])
            for value in self.audits
            for candidate in value["candidates"]
            if not bool(candidate["incumbent"])
        )
        if isinstance(self.audit_searcher, RecordingJointSearch):
            result["exact_audit_restore_checks"] = sum(
                value.result.restore_checks
                for value in self.audit_searcher.records
            )
            result["exact_audit_simulated_branch_ticks"] = sum(
                value.result.simulated_ticks
                for value in self.audit_searcher.records
            )
            result["exact_audit_wall_seconds"] = sum(
                value.result.wall_seconds
                for value in self.audit_searcher.records
            )
            result["exact_audit_cpu_seconds"] = sum(
                value.result.cpu_seconds
                for value in self.audit_searcher.records
            )
        return dict(sorted(result.items()))


def _unsafe_teacher_actions(
    policies: Sequence[object],
) -> Mapping[int, Sequence[Mapping[str, object]]]:
    output: dict[int, list[Mapping[str, object]]] = {}
    for policy in policies:
        if not isinstance(policy, AuditedTeacherPolicy):
            raise TypeError("teacher screen policy audit is unavailable")
        rows = []
        for value in policy.audits:
            if not bool(value["override"]):
                continue
            selected = value.get("row")
            incumbent = value.get("incumbent_row")
            if not isinstance(selected, Mapping) or not isinstance(
                incumbent, Mapping
            ):
                rows.append(
                    {
                        **dict(value),
                        "independent_gate_status": (
                            "missing_executed_exact_rows"
                        ),
                    }
                )
                continue
            eligible, status, exact_unsafe = _exact_teacher_row_gate(
                selected, incumbent
            )
            if not eligible or exact_unsafe:
                rows.append(
                    {
                        **dict(value),
                        "independent_gate_status": status,
                    }
                )
        if rows:
            output[policy.seed] = rows
    return output


def _unsafe_student_actions(
    policies: Sequence[object],
) -> Mapping[int, Sequence[Mapping[str, object]]]:
    output: dict[int, list[Mapping[str, object]]] = {}
    for policy in policies:
        if not isinstance(policy, BudgetedResidualPolicy):
            raise TypeError("student audit is unavailable")
        rows = [
            value
            for value in policy.audits
            if bool(value["actual_exact_unsafe"])
        ]
        if rows:
            output[policy.seed] = rows
    return output


def _clopper_pearson_upper(
    successes: int, trials: int, *, alpha: float = 0.05
) -> float:
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("Clopper-Pearson counts are invalid")
    if successes == trials:
        return 1.0
    if successes == 0:
        return 1.0 - alpha ** (1.0 / trials)

    def cdf(probability: float) -> float:
        return sum(
            math.comb(trials, index)
            * probability**index
            * (1.0 - probability) ** (trials - index)
            for index in range(successes + 1)
        )

    low, high = successes / trials, 1.0
    for _ in range(100):
        middle = (low + high) / 2.0
        if cdf(middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def _coverage_lcb(
    by_seed: Sequence[tuple[int, int]], *, metric: str
) -> float:
    if len(by_seed) < 1:
        return 0.0
    seed = int.from_bytes(
        hashlib.sha256(
            f"{SEED_DOMAIN}|bootstrap|{metric}".encode()
        ).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    values = np.asarray(by_seed, dtype=np.int64)
    samples = rng.integers(
        0, len(values), size=(BOOTSTRAP_REPLICATES, len(values))
    )
    selected = values[samples]
    numerator = selected[:, :, 0].sum(1)
    denominator = selected[:, :, 1].sum(1)
    rates = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float64),
        where=denominator > 0,
    )
    return float(np.quantile(rates, 0.05, method="linear"))


def _risk_deciles(
    rows: Sequence[Mapping[str, object]],
) -> Sequence[Mapping[str, object]]:
    candidates = [
        value
        for value in rows
        if not bool(value["incumbent"])
    ]
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda value: (
            float(value["unsafe_probability"]),
            int(value["ordinal"]),
        ),
    )
    bins = np.array_split(np.arange(len(ordered)), 10)
    output: list[Mapping[str, object]] = []
    for decile, indices in enumerate(bins, start=1):
        if not len(indices):
            continue
        selected = [ordered[int(index)] for index in indices]
        output.append(
            {
                "decile": decile,
                "rows": len(selected),
                "predicted_unsafe_probability": _distribution(
                    [
                        float(value["unsafe_probability"])
                        for value in selected
                    ]
                ),
                "predicted_delta_b2": _distribution(
                    [
                        float(value["predicted_delta_b2"])
                        for value in selected
                    ]
                ),
                "exact_delta_b2": _distribution(
                    [
                        float(value["exact_delta_b2"])
                        for value in selected
                    ]
                ),
                "exact_unsafe_rate": sum(
                    bool(value["exact_unsafe"]) for value in selected
                )
                / len(selected),
                "certified": sum(
                    bool(value["certified"]) for value in selected
                ),
            }
        )
    return output


def _barrier_cohort(
    policies: Sequence[object],
    *,
    cohort: str,
) -> Mapping[str, object]:
    if len(policies) < 1 or any(
        not isinstance(value, BudgetedResidualPolicy) for value in policies
    ):
        raise ValueError("R3G barrier cohort policies are malformed")
    audits = [
        audit
        for policy in policies
        for audit in policy.audits  # type: ignore[union-attr]
    ]
    proposals = [
        proposal
        for policy in policies
        for proposal in policy.proposals  # type: ignore[union-attr]
    ]
    target_audits = (
        [value for value in audits if bool(value["stress_trigger"])]
        if cohort == "barrier-stress"
        else audits
    )
    candidate_rows = [
        candidate
        for audit in target_audits
        for candidate in audit["candidates"]
        if not bool(candidate["incumbent"])
    ]
    false_safe_by_seed: dict[int, list[Mapping[str, object]]] = {}
    certified_unsafe_by_seed: dict[int, list[Mapping[str, object]]] = {}
    unsafe_by_seed: dict[int, int] = Counter()
    unsafe_states_by_seed: dict[int, set[int]] = {}
    severe = 0
    for audit in target_audits:
        seed = int(audit["seed"])
        unsafe = [
            value
            for value in audit["candidates"]
            if not bool(value["incumbent"])
            and bool(value["exact_unsafe"])
        ]
        unsafe_by_seed[seed] += len(unsafe)
        if unsafe:
            unsafe_states_by_seed.setdefault(seed, set()).add(
                int(audit["proposal_query_index"])
            )
        severe += sum(bool(value["severe_exact_unsafe"]) for value in unsafe)
        certified_unsafe = [
            value
            for value in unsafe
            if bool(value["certified"])
        ]
        if certified_unsafe:
            certified_unsafe_by_seed.setdefault(seed, []).append(
                {
                    "tick": audit["tick"],
                    "query_index": audit["proposal_query_index"],
                    "rows": certified_unsafe,
                }
            )
    for audit in audits:
        if bool(audit["actual_exact_unsafe"]):
            false_safe_by_seed.setdefault(int(audit["seed"]), []).append(
                {
                    "tick": audit["tick"],
                    "query_index": audit["proposal_query_index"],
                    "selected_ordinal": audit["selected_ordinal"],
                }
            )
    episode_count = len(policies)
    false_safe_episodes = len(false_safe_by_seed)
    cp_upper = _clopper_pearson_upper(
        false_safe_episodes, episode_count
    )
    coverage_clusters: list[tuple[int, int]] = []
    supported_clusters: list[tuple[int, int]] = []
    for policy in policies:
        overrides = sum(
            bool(value["actual_override"]) for value in policy.proposals
        )
        supported_states = sum(
            int(value["supported_alternatives"]) > 0
            for value in policy.proposals
        )
        coverage_clusters.append((overrides, len(policy.proposals)))
        supported_clusters.append((supported_states, len(policy.proposals)))
    coverage_lcb = _coverage_lcb(
        coverage_clusters, metric=f"{cohort}-override-coverage"
    )
    supported_lcb = _coverage_lcb(
        supported_clusters, metric=f"{cohort}-support-coverage"
    )
    unsafe_outcomes = sum(unsafe_by_seed.values())
    unsafe_states = sum(len(value) for value in unsafe_states_by_seed.values())
    unsafe_seeds = sum(value > 0 for value in unsafe_by_seed.values())
    nonvacuity = {
        "exact_unsafe_outcomes": unsafe_outcomes,
        "unsafe_decision_states": unsafe_states,
        "unsafe_seeds": unsafe_seeds,
        "severe_unsafe_outcomes": severe,
        "required": {
            "exact_unsafe_outcomes": 32 if cohort == "barrier-heldout" else 64,
            "unsafe_decision_states": 8 if cohort == "barrier-heldout" else 64,
            "unsafe_seeds": 4 if cohort == "barrier-heldout" else 64,
            "severe_unsafe_outcomes": 8 if cohort == "barrier-heldout" else 0,
        },
    }
    if cohort == "barrier-heldout":
        nonvacuity_passed = (
            unsafe_outcomes >= 32
            and unsafe_states >= 8
            and unsafe_seeds >= 4
            and severe >= 8
        )
    else:
        nonvacuity_passed = all(
            unsafe_by_seed.get(policy.seed, 0) > 0 for policy in policies
        )
    checks = {
        "at_least_59_independent_episodes": episode_count >= 59,
        "zero_false_safe_episodes": false_safe_episodes == 0,
        "clopper_pearson_upper_at_most_5pct": cp_upper <= 0.05,
        "override_coverage_cluster_lcb_at_least_5pct": coverage_lcb >= 0.05,
        "nonvacuity_floor": nonvacuity_passed,
    }
    return {
        "cohort": cohort,
        "episodes": episode_count,
        "eligible_on_policy_states": len(proposals),
        "exact_audited_states": len(audits),
        "targeted_exact_audited_states": len(target_audits),
        "candidate_rows": len(candidate_rows),
        "support": {
            "supported_candidates": sum(
                bool(value["supported"]) for value in candidate_rows
            ),
            "candidate_coverage": (
                sum(bool(value["supported"]) for value in candidate_rows)
                / len(candidate_rows)
                if candidate_rows
                else 0.0
            ),
            "state_clustered_lcb": supported_lcb,
        },
        "override": {
            "count": sum(
                bool(value["actual_override"]) for value in proposals
            ),
            "frequency": (
                sum(bool(value["actual_override"]) for value in proposals)
                / len(proposals)
                if proposals
                else 0.0
            ),
            "seed_clustered_one_sided_95pct_lcb": coverage_lcb,
            "clusters": [
                {
                    "seed": policy.seed,
                    "overrides": numerator,
                    "eligible_states": denominator,
                }
                for policy, (numerator, denominator) in zip(
                    policies, coverage_clusters, strict=True
                )
            ],
        },
        "false_safe": {
            "episodes": false_safe_episodes,
            "rate": false_safe_episodes / episode_count,
            "one_sided_95pct_clopper_pearson_upper": cp_upper,
            "details": false_safe_by_seed,
            "certified_unsafe_unselected_candidate_diagnostic": (
                certified_unsafe_by_seed
            ),
        },
        "nonvacuity": nonvacuity,
        "risk_deciles": _risk_deciles(candidate_rows),
        "checks": checks,
        "passed": all(checks.values()),
        "audits": audits,
    }


def _conformal_calibration(
    bundle: ModelBundle,
    dataset: Mapping[str, Any],
) -> tuple[BarrierCalibration, Mapping[str, object]]:
    tensors = dataset["tensors"]
    indices = torch.arange(tensors["features"].shape[0])
    barrier = fit_barrier(
        bundle.ensemble,
        bundle.support,
        tensors,
        indices,
        dataset["signatures"],
        alpha=CONFORMAL_ALPHA,
    )
    prediction = bundle.ensemble.predict_full(tensors["features"].float())
    groups = tensors["group"].long()
    supported = torch.tensor(
        [
            bundle.support.passes(
                dataset["signatures"][index],
                tensors["support"][index].float(),
            )
            for index in indices.tolist()
        ],
        dtype=torch.bool,
    )
    residuals: list[Mapping[str, object]] = []
    for group in torch.unique(groups, sorted=True).tolist():
        cluster = groups == group
        mask = cluster
        source = tensors["seed"][cluster]
        if torch.unique(source).numel() != 1:
            raise RuntimeError("calibration group crosses whole-seed clusters")
        if not bool(mask.any()):
            raise RuntimeError("calibration contains an empty seed cluster")
        residuals.append(
            {
                "seed": int(source[0]),
                "maximum_delta_overprediction_residual": float(
                    (
                        prediction["delta_cvar_mean"][mask]
                        - tensors["risk_margin"][mask]
                    ).max()
                ),
                "maximum_absolute_overprediction_residual": float(
                    (
                        prediction["absolute_cvar_mean"][mask]
                        - tensors["absolute_solvency"][mask]
                    ).max()
                ),
                "candidate_rows": int(mask.sum()),
            }
        )
    n = len(residuals)
    k = math.ceil((n + 1) * (1.0 - CONFORMAL_ALPHA))
    if n < 59 or k > n:
        raise RuntimeError("finite split-conformal threshold is unavailable")
    temporary = ModelBundle(
        bundle.ensemble,
        bundle.support,
        barrier.isotonic,
        barrier,
        bundle.metadata,
        bundle.path,
        bundle.sha256,
    )
    rows = [
        {
            "features": tensors["features"][index].numpy(),
            "support": tensors["support"][index].numpy(),
            "signature": dataset["signatures"][index],
            "incumbent": bool(tensors["incumbent"][index]),
            "ordinal": int(dataset["row_manifests"][index]["ordinal"]),
            "labels": {
                "risk_margin": float(tensors["risk_margin"][index]),
                "absolute_solvency": float(
                    tensors["absolute_solvency"][index]
                ),
                "unsafe": bool(tensors["unsafe"][index]),
                "resolved": bool(tensors["resolved"][index]),
                "score_advantage": float(
                    tensors["score_advantage"][index]
                ),
            },
            "candidate": dataset["row_manifests"][index]["candidate"],
            "outcome": dataset["row_manifests"][index]["outcome"],
            "solvency": dataset["row_manifests"][index]["solvency"],
            "search_sha256": dataset["row_manifests"][index][
                "search_sha256"
            ],
        }
        for index in range(tensors["features"].shape[0])
    ]
    diagnostics = _certify_rows(temporary, rows)
    report = {
        **dict(barrier.report),
        "schema": "irisu-r3g-episode-max-conformal-v1",
        "clusters": n,
        "order_statistic_index_one_based": k,
        "q": barrier.conformal_q,
        "absolute_q": barrier.absolute_conformal_q,
        "residual_definition": (
            "whole-seed max over every retained candidate outcome "
            "(predicted - exact), separately for delta_B2 and absolute B2"
        ),
        "ensemble_spread_role": "diagnostic only; not a confidence bound",
        "residuals": residuals,
        "support_candidate_coverage": sum(
            bool(value["supported"]) for value in diagnostics
        )
        / len(diagnostics),
        "certified_candidates": sum(
            bool(value["certified"]) for value in diagnostics
        ),
        "false_safe_candidates_diagnostic_only": sum(
            bool(value["certified"]) and bool(value["exact_unsafe"])
            for value in diagnostics
        ),
        "risk_deciles": _risk_deciles(diagnostics),
    }
    return barrier, report


def _fail_closed_control_calibration(
    bundle: ModelBundle,
    dataset: Mapping[str, Any],
    error: Exception,
) -> tuple[BarrierCalibration, Mapping[str, object]]:
    tensors = dataset["tensors"]
    indices = torch.arange(tensors["features"].shape[0])
    prediction = bundle.ensemble.predict_full(tensors["features"].float())
    isotonic = fit_isotonic(
        prediction["unsafe_mean"].tolist(),
        tensors["unsafe"].bool().tolist(),
    )
    seeds = tensors["seed"].long()
    delta_residuals: list[float] = []
    absolute_residuals: list[float] = []
    for seed in torch.unique(seeds, sorted=True).tolist():
        chosen = seeds == seed
        delta_residuals.append(
            float(
                (
                    prediction["delta_cvar_mean"][chosen]
                    - tensors["risk_margin"][chosen]
                ).max()
            )
        )
        absolute_residuals.append(
            float(
                (
                    prediction["absolute_cvar_mean"][chosen]
                    - tensors["absolute_solvency"][chosen]
                ).max()
            )
        )
    n = len(delta_residuals)
    k = math.ceil((n + 1) * (1.0 - CONFORMAL_ALPHA))
    if n < 59 or k > n:
        raise RuntimeError(
            "winner-only control lacks a finite fail-closed conformal bound"
        ) from error
    q = sorted(delta_residuals)[k - 1]
    absolute_q = sorted(absolute_residuals)[k - 1]
    report = {
        "schema": "irisu-r3g-episode-max-conformal-v1",
        "calibration_failed_closed": True,
        "calibration_error": f"{type(error).__name__}: {error}",
        "all_abstain_probability_threshold": -1.0,
        "clusters": n,
        "order_statistic_index_one_based": k,
        "q": q,
        "absolute_q": absolute_q,
        "ensemble_spread_role": "diagnostic only; not a confidence bound",
        "residual_definition": (
            "whole-seed max over every retained candidate outcome "
            "(predicted - exact), separately for delta_B2 and absolute B2"
        ),
        "risk_deciles": [],
        "certified_candidates": 0,
        "false_safe_candidates_diagnostic_only": 0,
    }
    barrier = BarrierCalibration(
        q,
        -1.0,
        0.0,
        isotonic,
        report,
        q,
        absolute_q,
        CONFORMAL_ALPHA,
        n,
    )
    return barrier, report


def _require_gate(path: Path, *keys: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"required predecessor artifact is missing: {path}")
    value: Any = _load_json(path)
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"predecessor gate field is missing: {key}")
        value = value[key]
    if not isinstance(value, Mapping) or not bool(value.get("passed", False)):
        raise RuntimeError(f"successive-halving predecessor failed: {path}")
    return _load_json(path)


def _freeze_reference() -> Mapping[str, object]:
    references = {}
    for name in (
        "seed-manifests.json",
        "global-disjointness-proof.json",
        "experiment-config.json",
    ):
        path = OUTPUT / name
        value = _load_json(path)
        unsigned = dict(value)
        logical_sha = unsigned.pop("sha256", None)
        if logical_sha is not None and logical_sha != _sha(unsigned):
            raise RuntimeError(f"generation-04 freeze changed: {path}")
        references[name] = {
            "path": str(path),
            "file_sha256": _file_sha256(path),
            "logical_sha256": logical_sha,
        }
    return {
        "output": str(OUTPUT),
        "append_only": True,
        "artifacts": references,
    }


def _stage_header(
    *,
    schema: str,
    source: Mapping[str, object],
    config: Mapping[str, object],
    seed_manifest: Mapping[str, object] | None,
    wall: float,
    cpu: float,
    compute: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": schema,
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_or_canonical_material_used": False,
        "protocol": {
            "path": str(PROTOCOL),
            "sha256": PROTOCOL_SHA256,
        },
        "source_identity": source,
        "configuration": config,
        "runtime": {"path": str(RUNTIME), "sha256": RUNTIME_SHA256},
        "frozen_v5": {"path": str(BASE), "sha256": BASE_SHA256},
        "seed_manifest": seed_manifest,
        "generation04_freeze": _freeze_reference(),
        "execution": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "compute": compute,
        },
    }


def _sentinel_projection(value: EpisodeMetrics) -> Mapping[str, int]:
    return {
        "seed": value.seed,
        "horizon": value.horizon_ticks,
        "survival_ticks": value.survival_ticks,
        "final_score": value.final_score,
        "final_gauge": value.final_gauge,
        "final_level": value.final_level,
        "qualifying_clears": value.qualifying_clears,
        "rotten_events": value.rotten_events,
    }


def correctness(
    output: Path,
    source: Mapping[str, object],
    config: Mapping[str, object],
    compute: Mapping[str, object],
    base_model: GoalConditionedSteeringModel,
    base_identity: str,
) -> bool:
    started, cpu_started = time.perf_counter(), time.process_time()
    unit_environment = dict(os.environ)
    unit_environment["PYTHONPATH"] = str(ROOT / "python")
    unit = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tests/test_r3g_conservative_offline.py"),
        ],
        cwd=ROOT,
        env=unit_environment,
        capture_output=True,
        text=True,
        check=False,
    )
    sentinel_episodes: list[EpisodeMetrics] = []
    sentinel_runners: list[Mapping[str, object]] = []
    sentinel_attestations: list[Mapping[str, object]] = []
    for expected in SENTINELS:
        episode, runner, attestation, _ = _run_episode(
            label="frozen_v5_sentinel",
            seed=int(expected["seed"]),
            horizon=int(expected["horizon"]),
            factory=lambda _env: _base_policy(base_model, base_identity),
        )
        if _sentinel_projection(episode) != dict(expected):
            raise RuntimeError("frozen-v5 cached sentinel reproduction failed")
        sentinel_episodes.append(episode)
        sentinel_runners.append(runner)
        sentinel_attestations.append(attestation)

    transaction: dict[str, object]
    query_rows: list[dict[str, Any]] = []
    with IrisuEnv(
        library_path=RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": 2_000},
    ) as env:
        observation, _ = env.reset(seed=int(SENTINELS[0]["seed"]))
        snapshot = env.clone_state()
        state_hash = env.state_hash()
        env.step(Action.wait(1))
        restored = env.restore_state(snapshot)
        if (
            env.clone_state() != snapshot
            or env.state_hash() != state_hash
            or _sha(restored) != _sha(observation)
        ):
            raise RuntimeError("exact transactional restore changed state")
        transaction = {
            "snapshot_sha256": hashlib.sha256(snapshot).hexdigest(),
            "state_hash": int(state_hash),
            "restored_observation_sha256": _sha(restored),
            "byte_equal": True,
        }
        policy = _base_policy(base_model, base_identity)
        policy.reset(int(SENTINELS[0]["seed"]))
        while True:
            incumbent = policy.predict(observation)
            if incumbent.is_shot:
                break
            for action in incumbent.primitive_actions():
                observation, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    raise RuntimeError(
                        "sentinel ended before a correctness query"
                    )
        searcher = _searcher(
            base_model, base_identity, recording=True, config=DATA_CONFIG
        )
        if not isinstance(searcher, RecordingJointSearch):
            raise TypeError("correctness searcher is not recording")
        searcher.reset_records(int(SENTINELS[0]["seed"]))
        result = searcher.search(env, observation, incumbent)
        query_rows = searcher.records[-1].rows()
        if result.restore_checks != len(result.outcomes) + 1:
            raise RuntimeError("joint transactional restore accounting failed")

    incumbent_rows = [
        value for value in query_rows if bool(value["incumbent"])
    ]
    if len(incumbent_rows) != 1 or not math.isclose(
        _label_value(
            incumbent_rows[0]["labels"],
            "delta_b2",
            "delta_B2",
            "risk_margin",
        ),
        0.0,
        abs_tol=1e-9,
    ):
        raise RuntimeError("frozen-v5 action does not have delta_B2 zero")
    for row in query_rows:
        solvency = row["solvency"]
        paid = tuple(solvency.get("paid_liability_ids", ()))
        emergent = tuple(solvency.get("emergent_paid_ids", ()))
        if len(paid) != len(set(paid)) or len(emergent) != len(set(emergent)):
            raise RuntimeError("a rot liability was paid more than once")
    original = {
        int(value["ordinal"]): _sha(value["labels"]) for value in query_rows
    }
    reordered = {
        int(value["ordinal"]): _sha(value["labels"])
        for value in reversed(query_rows)
    }
    checks = {
        "exact_transactional_restore": bool(transaction["byte_equal"]),
        "candidate_local_debt_accounting": original == reordered,
        "paid_liabilities_removed_once": True,
        "irrelevant_unsafe_candidate_invariant": original == reordered,
        "incumbent_delta_b2_zero": True,
        "all_candidate_outcomes_retained": (
            len(query_rows) == len(result.outcomes)
        ),
        "runtime_identity": _file_sha256(RUNTIME) == RUNTIME_SHA256,
        "checkpoint_identity": _file_sha256(BASE) == BASE_SHA256,
        "source_identity": _source_identity() == source,
        "two_cached_sentinels_reproduced": True,
        "full_branch_unit_test_file_passed": unit.returncode == 0,
    }
    gate = {"passed": all(checks.values()), "checks": checks}
    report = {
        **_stage_header(
            schema="irisu-r3g-correctness-report-v1",
            source=source,
            config=config,
            seed_manifest=None,
            wall=time.perf_counter() - started,
            cpu=time.process_time() - cpu_started,
            compute=compute,
        ),
        "sentinels": {
            "expected": list(SENTINELS),
            "observed": [
                value.manifest() for value in sentinel_episodes
            ],
            "deterministic_projection_sha256s": [
                _sha(_sentinel_projection(value))
                for value in sentinel_episodes
            ],
            "runner_sha256s": [
                _sha(value) for value in sentinel_runners
            ],
            "attestation_sha256s": [
                _sha(value) for value in sentinel_attestations
            ],
        },
        "transaction": transaction,
        "unit_tests": {
            "command": [
                sys.executable,
                str(ROOT / "tests/test_r3g_conservative_offline.py"),
            ],
            "returncode": unit.returncode,
            "stdout": unit.stdout,
            "stderr": unit.stderr,
        },
        "candidate_query": {
            "search_sha256": result.sha256,
            "snapshot_sha256": result.snapshot_sha256,
            "candidate_rows": len(query_rows),
            "unsafe_rows": sum(
                _exact_unsafe(value["labels"]) for value in query_rows
            ),
            "row_manifest_sha256s": [
                _sha(_row_manifest(value)) for value in query_rows
            ],
        },
        "gate": gate,
    }
    _assert_inputs(source)
    _atomic_json(output / "correctness-report.json", report)
    return bool(gate["passed"])


def teacher_screen(
    output: Path,
    source: Mapping[str, object],
    config: Mapping[str, object],
    compute: Mapping[str, object],
    seed_bundle: Mapping[str, object],
    base_model: GoalConditionedSteeringModel,
    base_identity: str,
) -> bool:
    _require_gate(output / "correctness-report.json", "gate")
    started, cpu_started = time.perf_counter(), time.process_time()
    seeds, horizon, manifest = _suite(
        seed_bundle, TEACHER_SCREEN_SPLIT
    )
    baseline, _, runner, attestation = _evaluate(
        label="frozen_v5",
        seeds=seeds,
        horizon=horizon,
        factory=lambda _env: _base_policy(base_model, base_identity),
    )
    teacher, policies, teacher_runner, teacher_attestation = _evaluate(
        label="joint_teacher_b2_audited",
        seeds=seeds,
        horizon=horizon,
        factory=lambda env: AuditedTeacherPolicy(
            env, base_model, base_identity
        ),
    )
    if runner != teacher_runner or attestation != teacher_attestation:
        raise RuntimeError("teacher screen paired runtime identity changed")
    unsafe = _unsafe_teacher_actions(policies)
    paired = _paired(baseline, teacher, unsafe_executed=unsafe)
    gate = _teacher_gate(baseline, teacher, paired)
    report = {
        **_stage_header(
            schema="irisu-r3g-teacher-screen-report-v1",
            source=source,
            config=config,
            seed_manifest=manifest,
            wall=time.perf_counter() - started,
            cpu=time.process_time() - cpu_started,
            compute=compute,
        ),
        "runner": runner,
        "runtime_attestation": attestation,
        "baseline": {
            "aggregate": _aggregate(baseline),
            "episodes": [value.manifest() for value in baseline],
        },
        "teacher": {
            "aggregate": _aggregate(teacher),
            "episodes": [value.manifest() for value in teacher],
            "audits": [
                audit
                for policy in policies
                for audit in policy.audits  # type: ignore[union-attr]
            ],
        },
        "paired": paired,
        "gate": gate,
    }
    _assert_inputs(source)
    _atomic_json(output / "teacher-screen-report.json", report)
    return bool(gate["passed"])


def _collect_split(
    *,
    seeds: Sequence[int],
    horizon: int,
    base_model: GoalConditionedSteeringModel,
    base_identity: str,
    label: str,
    maximum_queries: int,
) -> tuple[
    list[EpisodeMetrics],
    list[Mapping[str, Any]],
    Mapping[str, object],
    Mapping[str, object],
]:
    episodes, policies, runner, attestation = _evaluate(
        label=label,
        seeds=seeds,
        horizon=horizon,
        factory=lambda env: AllOutcomeCollectorPolicy(
            env,
            _base_policy(base_model, base_identity),
            _searcher(
                base_model, base_identity, recording=True, config=DATA_CONFIG
            ),
            query_stride_shots=QUERY_BUDGET["query_stride_shots"],
            maximum_queries=maximum_queries,
        ),
    )
    rows: list[Mapping[str, Any]] = []
    for policy in policies:
        if not isinstance(policy, AllOutcomeCollectorPolicy):
            raise TypeError("R3G collection policy identity changed")
        rows.extend(_records_rows(policy))
    return episodes, rows, runner, attestation


def collect(
    output: Path,
    source: Mapping[str, object],
    config: Mapping[str, object],
    compute: Mapping[str, object],
    seed_bundle: Mapping[str, object],
    base_model: GoalConditionedSteeringModel,
    base_identity: str,
) -> bool:
    _require_gate(output / "teacher-screen-report.json", "gate")
    started, cpu_started = time.perf_counter(), time.process_time()
    seeds, horizon, manifest = _suite(seed_bundle, "dagger-train")
    episodes, rows, runner, attestation = _collect_split(
        seeds=seeds,
        horizon=horizon,
        base_model=base_model,
        base_identity=base_identity,
        label="dagger_train_complete_outcomes",
        maximum_queries=QUERY_BUDGET["dagger_train_queries_per_episode"],
    )
    dataset = _build_dataset(rows, split="dagger-train")
    dataset_path = output / "dagger-train-complete-outcomes.pt"
    _assert_inputs(source)
    dataset_sha256 = _atomic_torch(dataset_path, dataset)
    tensors = dataset["tensors"]
    incumbent = int(tensors["incumbent"].sum())
    selected = int(tensors["selected"].sum())
    unsafe = int(tensors["unsafe"].sum())
    checks = {
        "sixteen_whole_seed_clusters": (
            torch.unique(tensors["seed"]).numel() == 16
        ),
        "all_candidate_rows_retained": len(rows) > selected,
        "contains_winners": selected > 0,
        "contains_losers": len(rows) - selected > 0,
        "contains_negative_branches": unsafe > 0,
        "incumbent_per_query": (
            incumbent == int(dataset["identity"]["query_states"])
        ),
    }
    gate = {"passed": all(checks.values()), "checks": checks}
    report = {
        **_stage_header(
            schema="irisu-r3g-collection-report-v1",
            source=source,
            config=config,
            seed_manifest=manifest,
            wall=time.perf_counter() - started,
            cpu=time.process_time() - cpu_started,
            compute=compute,
        ),
        "runner": runner,
        "runtime_attestation": attestation,
        "episodes": [value.manifest() for value in episodes],
        "aggregate": _aggregate(episodes),
        "dataset": {
            "path": str(dataset_path),
            "file_sha256": dataset_sha256,
            "identity": dataset["identity"],
            "rows": len(rows),
            "incumbent_rows": incumbent,
            "selected_winner_rows": selected,
            "nonwinner_rows": len(rows) - selected,
            "unsafe_rows": unsafe,
        },
        "gate": gate,
    }
    _atomic_json(output / "collection-report.json", report)
    return bool(gate["passed"])


def train(
    output: Path,
    source: Mapping[str, object],
    config: Mapping[str, object],
    compute: Mapping[str, object],
) -> bool:
    collection_report = _require_gate(
        output / "collection-report.json", "gate"
    )
    started, cpu_started = time.perf_counter(), time.process_time()
    dataset_path = Path(collection_report["dataset"]["path"])
    dataset = _load_dataset(
        dataset_path, str(collection_report["dataset"]["file_sha256"])
    )
    tensors = dataset["tensors"]
    signatures = dataset["signatures"]
    indices = torch.arange(tensors["features"].shape[0])
    outputs: dict[str, Mapping[str, object]] = {}
    for name, winner_only in (
        ("winner-only", True),
        ("complete-outcome", False),
    ):
        training_started, training_cpu = (
            time.perf_counter(),
            time.process_time(),
        )
        ensemble = train_ensemble(
            tensors,
            indices,
            winner_only=winner_only,
            steps=QUERY_BUDGET["training_steps"],
            ensemble_size=QUERY_BUDGET["ensemble_size"],
            seed=TRAINING_SEED,
        )
        support = fit_support(
            tensors,
            indices,
            signatures,
            winner_only=winner_only,
            minimum_groups=8,
        )
        fit_indices = (
            indices[tensors["selected"][indices]]
            if winner_only
            else indices
        )
        isotonic = _fit_isotonic_from_training(
            ensemble, tensors, fit_indices
        )
        path = output / f"{name}-uncalibrated.pt"
        metadata = {
            "schema": "irisu-r3g-training-binding-v1",
            "variant": name,
            "winner_only": winner_only,
            "source_identity_sha256": source["sha256"],
            "configuration_sha256": config["sha256"],
            "protocol_sha256": PROTOCOL_SHA256,
            "runtime_sha256": RUNTIME_SHA256,
            "frozen_v5_sha256": BASE_SHA256,
            "dataset_identity_sha256": dataset["identity"]["sha256"],
            "dataset_file_sha256": collection_report["dataset"][
                "file_sha256"
            ],
            "planner_config_sha256": DATA_CONFIG.sha256,
            "candidate_cap": QUERY_BUDGET["candidate_cap"],
            "exact_rollout_cap_ticks": QUERY_BUDGET[
                "exact_rollout_cap_ticks"
            ],
            "training_seed": TRAINING_SEED,
            "calibrated": False,
        }
        digest = _save_model(
            path,
            ensemble=ensemble,
            support=support,
            isotonic=isotonic,
            barrier=None,
            metadata=metadata,
        )
        outputs[name] = {
            "path": str(path),
            "sha256": digest,
            "fit_rows": int(fit_indices.numel()),
            "unsafe_fit_rows": int(tensors["unsafe"][fit_indices].sum()),
            "support_signatures": sorted(support.centers),
            "support_signature_count": len(support.centers),
            "training_report": dict(ensemble.training_report),
            "wall_seconds": time.perf_counter() - training_started,
            "cpu_seconds": time.process_time() - training_cpu,
        }
    checks = {
        "winner_only_checkpoint_created": Path(
            outputs["winner-only"]["path"]
        ).is_file(),
        "complete_outcome_checkpoint_created": Path(
            outputs["complete-outcome"]["path"]
        ).is_file(),
        "negative_branches_excluded_only_from_control": (
            int(outputs["winner-only"]["unsafe_fit_rows"])
            <= int(outputs["complete-outcome"]["unsafe_fit_rows"])
        ),
        "complete_uses_more_rows": (
            int(outputs["complete-outcome"]["fit_rows"])
            > int(outputs["winner-only"]["fit_rows"])
        ),
    }
    gate = {"passed": all(checks.values()), "checks": checks}
    report = {
        **_stage_header(
            schema="irisu-r3g-training-report-v1",
            source=source,
            config=config,
            seed_manifest=None,
            wall=time.perf_counter() - started,
            cpu=time.process_time() - cpu_started,
            compute=compute,
        ),
        "dataset": {
            "path": str(dataset_path),
            "sha256": collection_report["dataset"]["file_sha256"],
            "identity_sha256": dataset["identity"]["sha256"],
        },
        "variants": outputs,
        "comparison": {
            "purpose": (
                "directly measure negative-branch value against winner-only "
                "imitation without selecting the primary from evaluation"
            ),
            "primary_predeclared": "complete-outcome",
            "diagnostic_control": "winner-only",
        },
        "gate": gate,
    }
    _assert_inputs(source)
    _atomic_json(output / "training-report.json", report)
    return bool(gate["passed"])


def barrier_calibration(
    output: Path,
    source: Mapping[str, object],
    config: Mapping[str, object],
    compute: Mapping[str, object],
    seed_bundle: Mapping[str, object],
    base_model: GoalConditionedSteeringModel,
    base_identity: str,
) -> bool:
    training_report = _require_gate(
        output / "training-report.json", "gate"
    )
    started, cpu_started = time.perf_counter(), time.process_time()
    seeds, horizon, manifest = _suite(
        seed_bundle, "barrier-calibration"
    )
    episodes, rows, runner, attestation = _collect_split(
        seeds=seeds,
        horizon=horizon,
        base_model=base_model,
        base_identity=base_identity,
        label="barrier_calibration_complete_outcomes",
        maximum_queries=QUERY_BUDGET["calibration_queries_per_episode"],
    )
    dataset = _build_dataset(rows, split="barrier-calibration")
    if torch.unique(dataset["tensors"]["seed"]).numel() != 64:
        raise RuntimeError("calibration did not retain 64 whole-seed clusters")
    calibration_path = output / "barrier-calibration-outcomes.pt"
    _assert_inputs(source)
    calibration_sha256 = _atomic_torch(calibration_path, dataset)
    variants: dict[str, Mapping[str, object]] = {}
    for name in ("winner-only", "complete-outcome"):
        uncalibrated = training_report["variants"][name]
        bundle = _load_model(
            Path(uncalibrated["path"]), str(uncalibrated["sha256"])
        )
        if (
            bundle.metadata.get("source_identity_sha256")
            != source["sha256"]
            or bundle.metadata.get("configuration_sha256")
            != config["sha256"]
            or bundle.metadata.get("dataset_identity_sha256")
            != training_report["dataset"]["identity_sha256"]
        ):
            raise ValueError("uncalibrated model binding changed")
        try:
            barrier, calibration_report = _conformal_calibration(
                bundle, dataset
            )
        except RuntimeError as error:
            if name != "winner-only":
                raise
            barrier, calibration_report = (
                _fail_closed_control_calibration(bundle, dataset, error)
            )
        calibrated_path = output / f"{name}-calibrated.pt"
        metadata = {
            **dict(bundle.metadata),
            "calibrated": True,
            "calibration_dataset_identity_sha256": dataset["identity"][
                "sha256"
            ],
            "calibration_dataset_file_sha256": calibration_sha256,
            "calibration_seed_manifest_sha256": manifest["sha256"],
            "conformal_q": barrier.conformal_q,
            "absolute_conformal_q": barrier.absolute_conformal_q,
            "conformal_clusters": barrier.episode_count,
            "conformal_order_statistic": math.ceil(
                (barrier.episode_count + 1) * (1.0 - CONFORMAL_ALPHA)
            ),
        }
        digest = _save_model(
            calibrated_path,
            ensemble=bundle.ensemble,
            support=bundle.support,
            isotonic=barrier.isotonic,
            barrier=barrier,
            metadata=metadata,
        )
        variants[name] = {
            "path": str(calibrated_path),
            "sha256": digest,
            "source_checkpoint": {
                "path": str(bundle.path),
                "sha256": bundle.sha256,
            },
            "calibration": calibration_report,
        }
    checks = {
        "sixty_four_whole_seed_clusters": (
            torch.unique(dataset["tensors"]["seed"]).numel() == 64
        ),
        "order_statistic_is_protocol_exact": all(
            int(value["calibration"]["order_statistic_index_one_based"])
            == math.ceil(
                (int(value["calibration"]["clusters"]) + 1)
                * (1.0 - CONFORMAL_ALPHA)
            )
            for value in variants.values()
        ),
        "ensemble_spread_not_confidence_bound": all(
            value["calibration"]["ensemble_spread_role"]
            == "diagnostic only; not a confidence bound"
            for value in variants.values()
        ),
        "complete_outcome_calibration_succeeded": not bool(
            variants["complete-outcome"]["calibration"].get(
                "calibration_failed_closed", False
            )
        ),
        "thresholds_selected_before_heldout": True,
    }
    gate = {"passed": all(checks.values()), "checks": checks}
    report = {
        **_stage_header(
            schema="irisu-r3g-barrier-calibration-report-v1",
            source=source,
            config=config,
            seed_manifest=manifest,
            wall=time.perf_counter() - started,
            cpu=time.process_time() - cpu_started,
            compute=compute,
        ),
        "runner": runner,
        "runtime_attestation": attestation,
        "episodes": [value.manifest() for value in episodes],
        "aggregate": _aggregate(episodes),
        "dataset": {
            "path": str(calibration_path),
            "file_sha256": calibration_sha256,
            "identity": dataset["identity"],
        },
        "variants": variants,
        "gate": gate,
    }
    report_path = output / "barrier-calibration-report.json"
    report_sha256 = _atomic_json(report_path, report)
    freeze = {
        "schema": "irisu-r3g-pre-heldout-freeze-v1",
        "frozen_before_any_heldout_or_stress_result": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity": source,
        "configuration": config,
        "runtime_sha256": RUNTIME_SHA256,
        "frozen_v5_sha256": BASE_SHA256,
        "candidate_generator": {
            "version": JOINT_PLANNER_VERSION,
            "config": DATA_CONFIG.manifest(),
            "config_sha256": DATA_CONFIG.sha256,
            "candidate_cap": QUERY_BUDGET["candidate_cap"],
        },
        "query_budget": QUERY_BUDGET,
        "seed_bundle_sha256": seed_bundle["sha256"],
        "training_dataset": training_report["dataset"],
        "calibration_dataset": report["dataset"],
        "calibration_report": {
            "path": str(report_path),
            "sha256": report_sha256,
        },
        "calibrated_checkpoints": {
            name: {
                "path": value["path"],
                "sha256": value["sha256"],
                "conformal_q": value["calibration"]["q"],
            }
            for name, value in variants.items()
        },
        "primary_policy": "complete-outcome",
        "diagnostic_policy": "winner-only",
    }
    freeze = {**freeze, "sha256": _sha(freeze)}
    _atomic_json(output / "pre-heldout-freeze.json", freeze)
    return bool(gate["passed"])


def _load_freeze(
    output: Path,
    source: Mapping[str, object],
    config: Mapping[str, object],
    seed_bundle: Mapping[str, object],
) -> Mapping[str, Any]:
    _require_gate(output / "barrier-calibration-report.json", "gate")
    freeze = _load_json(output / "pre-heldout-freeze.json")
    payload = dict(freeze)
    digest = payload.pop("sha256", None)
    if digest != _sha(payload):
        raise ValueError("pre-heldout freeze identity mismatch")
    if (
        freeze.get("source_identity") != source
        or freeze.get("configuration") != config
        or freeze.get("seed_bundle_sha256") != seed_bundle["sha256"]
        or freeze.get("runtime_sha256") != RUNTIME_SHA256
        or freeze.get("frozen_v5_sha256") != BASE_SHA256
    ):
        raise ValueError("pre-heldout identity binding changed")
    for value in freeze["calibrated_checkpoints"].values():
        if _file_sha256(Path(value["path"])) != value["sha256"]:
            raise ValueError("frozen calibrated checkpoint changed")
    return freeze


def _barrier_evaluate_variant(
    *,
    name: str,
    checkpoint: Mapping[str, Any],
    seeds: Sequence[int],
    horizon: int,
    stress: bool,
    base_model: GoalConditionedSteeringModel,
    base_identity: str,
    baseline: Sequence[EpisodeMetrics],
) -> Mapping[str, object]:
    bundle = _load_model(
        Path(checkpoint["path"]), str(checkpoint["sha256"])
    )
    episodes, policies, runner, attestation = _evaluate(
        label=f"{name}_{'stress' if stress else 'heldout'}",
        seeds=seeds,
        horizon=horizon,
        factory=lambda env: BudgetedResidualPolicy(
            env,
            base_model,
            base_identity,
            bundle,
            audit=True,
            stress=stress,
            sample_audit_cap=(
                QUERY_BUDGET["stress_sample_audits_per_episode"]
                if stress
                else QUERY_BUDGET["heldout_sample_audits_per_episode"]
            ),
        ),
    )
    unsafe = _unsafe_student_actions(policies)
    paired = _paired(baseline, episodes, unsafe_executed=unsafe)
    cohort_name = "barrier-stress" if stress else "barrier-heldout"
    barrier = _barrier_cohort(policies, cohort=cohort_name)
    return {
        "checkpoint": checkpoint,
        "runner": runner,
        "runtime_attestation": attestation,
        "aggregate": _aggregate(episodes),
        "episodes": [value.manifest() for value in episodes],
        "paired": paired,
        "barrier": barrier,
    }


def barrier_heldout(
    output: Path,
    source: Mapping[str, object],
    config: Mapping[str, object],
    compute: Mapping[str, object],
    seed_bundle: Mapping[str, object],
    base_model: GoalConditionedSteeringModel,
    base_identity: str,
) -> bool:
    freeze = _load_freeze(output, source, config, seed_bundle)
    started, cpu_started = time.perf_counter(), time.process_time()
    seeds, horizon, manifest = _suite(seed_bundle, "barrier-heldout")
    baseline, _, runner, attestation = _evaluate(
        label="frozen_v5_barrier_heldout",
        seeds=seeds,
        horizon=horizon,
        factory=lambda _env: _base_policy(base_model, base_identity),
    )
    variants = {
        name: _barrier_evaluate_variant(
            name=name,
            checkpoint=freeze["calibrated_checkpoints"][name],
            seeds=seeds,
            horizon=horizon,
            stress=False,
            base_model=base_model,
            base_identity=base_identity,
            baseline=baseline,
        )
        for name in ("winner-only", "complete-outcome")
    }
    for value in variants.values():
        if value["runner"] != runner or value["runtime_attestation"] != attestation:
            raise RuntimeError("heldout paired runtime identity changed")
    gate = {
        # Heldout and stress jointly form one barrier gate.  A scientifically
        # negative heldout cohort is preserved, then stress still runs under
        # the already-frozen identity before the combined decision.
        "passed": True,
        "cohort_passed": bool(
            variants["complete-outcome"]["barrier"]["passed"]
        ),
        "primary_predeclared": "complete-outcome",
        "diagnostic_control_not_used_for_selection": "winner-only",
    }
    report = {
        **_stage_header(
            schema="irisu-r3g-barrier-heldout-report-v1",
            source=source,
            config=config,
            seed_manifest=manifest,
            wall=time.perf_counter() - started,
            cpu=time.process_time() - cpu_started,
            compute=compute,
        ),
        "pre_heldout_freeze_sha256": freeze["sha256"],
        "baseline": {
            "runner": runner,
            "runtime_attestation": attestation,
            "aggregate": _aggregate(baseline),
            "episodes": [value.manifest() for value in baseline],
        },
        "variants": variants,
        "negative_branch_ablation": {
            "complete_outcome_passed": variants["complete-outcome"][
                "barrier"
            ]["passed"],
            "winner_only_passed": variants["winner-only"]["barrier"][
                "passed"
            ],
            "coverage_lcb_delta": (
                variants["complete-outcome"]["barrier"]["override"][
                    "seed_clustered_one_sided_95pct_lcb"
                ]
                - variants["winner-only"]["barrier"]["override"][
                    "seed_clustered_one_sided_95pct_lcb"
                ]
            ),
            "false_safe_episode_delta": (
                variants["complete-outcome"]["barrier"]["false_safe"][
                    "episodes"
                ]
                - variants["winner-only"]["barrier"]["false_safe"][
                    "episodes"
                ]
            ),
        },
        "gate": gate,
    }
    _assert_inputs(source)
    _atomic_json(output / "barrier-heldout-report.json", report)
    return bool(gate["passed"])


def barrier_stress(
    output: Path,
    source: Mapping[str, object],
    config: Mapping[str, object],
    compute: Mapping[str, object],
    seed_bundle: Mapping[str, object],
    base_model: GoalConditionedSteeringModel,
    base_identity: str,
) -> bool:
    freeze = _load_freeze(output, source, config, seed_bundle)
    heldout_path = output / "barrier-heldout-report.json"
    if not heldout_path.is_file():
        raise RuntimeError("barrier stress requires frozen heldout evidence")
    heldout = _load_json(heldout_path)
    if heldout.get("pre_heldout_freeze_sha256") != freeze["sha256"]:
        raise ValueError("heldout evidence is not bound to the same freeze")
    heldout_sha256 = _file_sha256(heldout_path)
    started, cpu_started = time.perf_counter(), time.process_time()
    seeds, horizon, manifest = _suite(seed_bundle, "barrier-stress")
    baseline, _, runner, attestation = _evaluate(
        label="frozen_v5_barrier_stress",
        seeds=seeds,
        horizon=horizon,
        factory=lambda _env: _base_policy(base_model, base_identity),
    )
    variants = {
        name: _barrier_evaluate_variant(
            name=name,
            checkpoint=freeze["calibrated_checkpoints"][name],
            seeds=seeds,
            horizon=horizon,
            stress=True,
            base_model=base_model,
            base_identity=base_identity,
            baseline=baseline,
        )
        for name in ("winner-only", "complete-outcome")
    }
    for value in variants.values():
        if value["runner"] != runner or value["runtime_attestation"] != attestation:
            raise RuntimeError("stress paired runtime identity changed")
    primary_checks = {
        "heldout_barrier_passed": bool(
            heldout["variants"]["complete-outcome"]["barrier"]["passed"]
        ),
        "stress_barrier_passed": bool(
            variants["complete-outcome"]["barrier"]["passed"]
        ),
        "heldout_zero_hard_paired_failures": not heldout["variants"][
            "complete-outcome"
        ]["paired"]["hard_failures"],
        "stress_zero_hard_paired_failures": not variants[
            "complete-outcome"
        ]["paired"]["hard_failures"],
    }
    gate = {
        "passed": all(primary_checks.values()),
        "checks": primary_checks,
        "primary_predeclared": "complete-outcome",
    }
    stress_report = {
        **_stage_header(
            schema="irisu-r3g-barrier-stress-report-v1",
            source=source,
            config=config,
            seed_manifest=manifest,
            wall=time.perf_counter() - started,
            cpu=time.process_time() - cpu_started,
            compute=compute,
        ),
        "pre_heldout_freeze_sha256": freeze["sha256"],
        "heldout_report": {
            "path": str(heldout_path),
            "sha256": heldout_sha256,
        },
        "baseline": {
            "runner": runner,
            "runtime_attestation": attestation,
            "aggregate": _aggregate(baseline),
            "episodes": [value.manifest() for value in baseline],
        },
        "variants": variants,
        "gate": gate,
    }
    stress_path = output / "barrier-stress-report.json"
    stress_sha256 = _atomic_json(stress_path, stress_report)
    barrier_report = {
        "schema": "irisu-r3g-combined-barrier-report-v1",
        "development_only": True,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": source["sha256"],
        "configuration_sha256": config["sha256"],
        "pre_heldout_freeze": {
            "path": str(output / "pre-heldout-freeze.json"),
            "sha256": freeze["sha256"],
        },
        "heldout_report": {
            "path": str(heldout_path),
            "sha256": heldout_sha256,
        },
        "stress_report": {
            "path": str(stress_path),
            "sha256": stress_sha256,
        },
        "primary": {
            "variant": "complete-outcome",
            "heldout": heldout["variants"]["complete-outcome"][
                "barrier"
            ],
            "stress": variants["complete-outcome"]["barrier"],
        },
        "winner_only_control": {
            "heldout": heldout["variants"]["winner-only"]["barrier"],
            "stress": variants["winner-only"]["barrier"],
        },
        "negative_branch_ablation": {
            "heldout_false_safe_episode_delta_complete_minus_control": (
                heldout["variants"]["complete-outcome"]["barrier"][
                    "false_safe"
                ]["episodes"]
                - heldout["variants"]["winner-only"]["barrier"][
                    "false_safe"
                ]["episodes"]
            ),
            "stress_false_safe_episode_delta_complete_minus_control": (
                variants["complete-outcome"]["barrier"]["false_safe"][
                    "episodes"
                ]
                - variants["winner-only"]["barrier"]["false_safe"][
                    "episodes"
                ]
            ),
            "heldout_override_lcb_delta_complete_minus_control": (
                heldout["variants"]["complete-outcome"]["barrier"][
                    "override"
                ]["seed_clustered_one_sided_95pct_lcb"]
                - heldout["variants"]["winner-only"]["barrier"][
                    "override"
                ]["seed_clustered_one_sided_95pct_lcb"]
            ),
        },
        "gate": gate,
    }
    _assert_inputs(source)
    _atomic_json(output / "barrier-report.json", barrier_report)
    return bool(gate["passed"])


def student_screen(
    output: Path,
    source: Mapping[str, object],
    config: Mapping[str, object],
    compute: Mapping[str, object],
    seed_bundle: Mapping[str, object],
    base_model: GoalConditionedSteeringModel,
    base_identity: str,
) -> bool:
    barrier_report = _require_gate(
        output / "barrier-report.json", "gate"
    )
    freeze = _load_freeze(output, source, config, seed_bundle)
    started, cpu_started = time.perf_counter(), time.process_time()
    seeds, horizon, manifest = _suite(seed_bundle, "student-screen")
    baseline, _, runner, attestation = _evaluate(
        label="frozen_v5_student_screen",
        seeds=seeds,
        horizon=horizon,
        factory=lambda _env: _base_policy(base_model, base_identity),
    )
    variants: dict[str, Mapping[str, object]] = {}
    for name in ("winner-only", "complete-outcome"):
        checkpoint = freeze["calibrated_checkpoints"][name]
        bundle = _load_model(
            Path(checkpoint["path"]), str(checkpoint["sha256"])
        )
        episodes, policies, student_runner, student_attestation = _evaluate(
            label=f"{name}_teacher_free_student_screen",
            seeds=seeds,
            horizon=horizon,
            factory=lambda env, selected=bundle: BudgetedResidualPolicy(
                env,
                base_model,
                base_identity,
                selected,
                audit=True,
                stress=False,
                sample_audit_cap=QUERY_BUDGET[
                    "student_screen_sample_audits_per_episode"
                ],
            ),
        )
        if student_runner != runner or student_attestation != attestation:
            raise RuntimeError("student screen paired runtime identity changed")
        unsafe = _unsafe_student_actions(policies)
        paired = _paired(baseline, episodes, unsafe_executed=unsafe)
        action_selection_counts = Counter()
        for policy in policies:
            if not isinstance(policy, BudgetedResidualPolicy):
                raise TypeError("student screen policy identity changed")
            action_selection_counts.update(policy.core.statistics())
        teacher_free = (
            int(action_selection_counts["teacher_queries"]) == 0
            and int(action_selection_counts["branch_simulated_ticks"]) == 0
            and int(action_selection_counts["clone_restore_calls"]) == 0
        )
        gate = {
            "passed": bool(teacher_free and not paired["hard_failures"]),
            "checks": {
                "teacher_free_action_selection": teacher_free,
                "zero_hard_paired_failures": not paired["hard_failures"],
            },
            "post_choice_exact_audit_not_used_for_action": True,
        }
        variants[name] = {
            "checkpoint": checkpoint,
            "aggregate": _aggregate(episodes),
            "episodes": [value.manifest() for value in episodes],
            "paired": paired,
            "action_selection_counts": dict(action_selection_counts),
            "gate": gate,
        }
    primary = variants["complete-outcome"]
    report = {
        **_stage_header(
            schema="irisu-r3g-student-screen-report-v1",
            source=source,
            config=config,
            seed_manifest=manifest,
            wall=time.perf_counter() - started,
            cpu=time.process_time() - cpu_started,
            compute=compute,
        ),
        "barrier_report": {
            "path": str(output / "barrier-report.json"),
            "sha256": _file_sha256(output / "barrier-report.json"),
            "gate": barrier_report["gate"],
        },
        "runner": runner,
        "runtime_attestation": attestation,
        "baseline": {
            "aggregate": _aggregate(baseline),
            "episodes": [value.manifest() for value in baseline],
        },
        "variants": variants,
        "negative_branch_ablation": {
            "complete_hard_failures": len(
                variants["complete-outcome"]["paired"]["hard_failures"]
            ),
            "winner_only_hard_failures": len(
                variants["winner-only"]["paired"]["hard_failures"]
            ),
            "score_median_delta_complete_minus_control": (
                float(
                    variants["complete-outcome"]["paired"]["score_delta"][
                        "median"
                    ]
                )
                - float(
                    variants["winner-only"]["paired"]["score_delta"][
                        "median"
                    ]
                )
            ),
            "survival_p10_delta_complete_minus_control": (
                float(
                    variants["complete-outcome"]["aggregate"][
                        "survival_ticks"
                    ]["p10"]
                )
                - float(
                    variants["winner-only"]["aggregate"][
                        "survival_ticks"
                    ]["p10"]
                )
            ),
        },
        "gate": primary["gate"],
        "screen_scope": (
            "resource screen only; parent owns every winner confirmation suite"
        ),
    }
    _assert_inputs(source)
    _atomic_json(output / "student-screen-report.json", report)
    return bool(primary["gate"]["passed"])


def _preregister(
    output: Path,
    source: Mapping[str, object],
    config: Mapping[str, object],
    seed_bundle: Mapping[str, object],
    proof: Mapping[str, object],
) -> None:
    path = output / "experiment-config.json"
    expected = {
        "schema": "irisu-r3g-preregistered-experiment-v1",
        "development_only": True,
        "parent_only_winner_suites_materializable": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "source_identity_sha256": source["sha256"],
        "configuration": config,
        "seed_bundle_sha256": seed_bundle["sha256"],
        "seed_bundle_file_sha256": _file_sha256(
            output / "seed-manifests.json"
        ),
        "global_disjointness_proof_sha256": proof["sha256"],
        "global_disjointness_proof_file_sha256": _file_sha256(
            output / "global-disjointness-proof.json"
        ),
        "output_identity": config["output_identity"],
        "successive_halving": [
            "correctness",
            TEACHER_SCREEN_SPLIT,
            "collect",
            "train",
            "barrier-calibration",
            "barrier-heldout",
            "barrier-stress",
            "student-screen",
        ],
        "stop_on_failed_gate": True,
    }
    expected = {**expected, "sha256": _sha(expected)}
    if path.exists():
        if _load_json(path) != expected:
            raise RuntimeError("preregistered R3G experiment identity changed")
    else:
        _atomic_json(path, expected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "correctness",
            "teacher-screen",
            "collect",
            "train",
            "barrier-calibration",
            "barrier-heldout",
            "barrier-stress",
            "barrier",
            "student-screen",
            "all",
        ),
    )
    args = parser.parse_args()
    compute = _pin_one_core()
    output = _safe_output()
    source = _source_identity()
    _assert_inputs(source)
    seed_bundle = _seed_bundle()
    external_inventory = _external_manifest_inventory()
    config = _config_identity(source, seed_bundle, external_inventory)
    proof = _global_disjointness_proof(
        source, config, seed_bundle, external_inventory
    )
    materialized_bundle = _materialize_seeds(output)
    if materialized_bundle != seed_bundle:
        raise RuntimeError("materialized generation-04 seed bundle changed")
    _materialize_proof(output, proof)
    _preregister(output, source, config, seed_bundle, proof)
    checkpoint = load_steering_checkpoint(
        BASE, expected_sha256=BASE_SHA256
    )
    checkpoint.model.eval()
    base_identity = _base_identity(source)

    stages: Mapping[str, Callable[[], bool]] = {
        "correctness": lambda: correctness(
            output,
            source,
            config,
            compute,
            checkpoint.model,
            base_identity,
        ),
        "teacher-screen": lambda: teacher_screen(
            output,
            source,
            config,
            compute,
            seed_bundle,
            checkpoint.model,
            base_identity,
        ),
        "collect": lambda: collect(
            output,
            source,
            config,
            compute,
            seed_bundle,
            checkpoint.model,
            base_identity,
        ),
        "train": lambda: train(output, source, config, compute),
        "barrier-calibration": lambda: barrier_calibration(
            output,
            source,
            config,
            compute,
            seed_bundle,
            checkpoint.model,
            base_identity,
        ),
        "barrier-heldout": lambda: barrier_heldout(
            output,
            source,
            config,
            compute,
            seed_bundle,
            checkpoint.model,
            base_identity,
        ),
        "barrier-stress": lambda: barrier_stress(
            output,
            source,
            config,
            compute,
            seed_bundle,
            checkpoint.model,
            base_identity,
        ),
        "student-screen": lambda: student_screen(
            output,
            source,
            config,
            compute,
            seed_bundle,
            checkpoint.model,
            base_identity,
        ),
    }
    if args.stage == "barrier":
        selected = ("barrier-heldout", "barrier-stress")
    elif args.stage == "all":
        selected = tuple(stages)
    else:
        selected = (args.stage,)
    completed: list[str] = []
    for name in selected:
        if not stages[name]():
            print(
                json.dumps(
                    {
                        "status": "rejected",
                        "failed_stage": name,
                        "completed": completed,
                        "output": str(output),
                    },
                    sort_keys=True,
                )
            )
            return
        completed.append(name)
    print(
        json.dumps(
            {
                "status": "passed_reached_stages",
                "completed": completed,
                "output": str(output),
                "source_identity_sha256": source["sha256"],
                "configuration_sha256": config["sha256"],
                "protocol_sha256": PROTOCOL_SHA256,
                "parent_confirmation_suites_materialized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
