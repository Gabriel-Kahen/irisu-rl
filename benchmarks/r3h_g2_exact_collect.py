#!/usr/bin/env python3
"""Collect fresh Generation 02 full-board exact labels.

This development-only runner preserves Generation 01 and its evidence.  It
executes only frozen-v5 live actions while the identity-bound Strategy B
evaluator labels alternatives from restored clones.  Every shadow query also
stores the exact pre-query public observation as canonical JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

sys.dont_write_bytecode = True
PYCACHE_PREFIX = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3h-resolution-first-g2r1-20260730/_runtime-pycache-disabled"
)


def _validate_pycache_boundary() -> None:
    if (
        sys.flags.dont_write_bytecode != 1
        or sys.flags.isolated != 1
        or sys.flags.no_site != 1
        or sys.flags.ignore_environment != 1
        or sys.flags.no_user_site != 1
        or sys.flags.safe_path is not True
        or sys.dont_write_bytecode is not True
        or sys.pycache_prefix != str(PYCACHE_PREFIX)
        or PYCACHE_PREFIX.exists()
        or PYCACHE_PREFIX.is_symlink()
    ):
        raise RuntimeError(
            "R3H G2 requires -I -S -B, a dedicated -X pycache_prefix, "
            "and an absent cache-prefix path; PYTHONPYCACHEPREFIX is not enough"
        )


_validate_pycache_boundary()

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"


REPOSITORY = Path("/home/gabe/Documents/irisu")
_STDLIB_ROOT = Path(os.__file__).resolve().parent
CLOSED_RUNTIME_SYS_PATH = (
    str((REPOSITORY / "python").resolve()),
    str(
        (
            REPOSITORY
            / ".venv/lib/python3.14/site-packages"
        ).resolve()
    ),
    str(_STDLIB_ROOT.parent / "python314.zip"),
    str(_STDLIB_ROOT),
    str(_STDLIB_ROOT / "lib-dynload"),
)
EXPERIMENT_ID = "r3h-resolution-first-g2r1-20260730"
EXPERIMENT_ROOT = REPOSITORY / "artifacts/r3/development" / EXPERIMENT_ID
OUTPUT_ROOT = EXPERIMENT_ROOT / "collection"
PROTOCOL_PATH = EXPERIMENT_ROOT / "protocol.md"
SOURCE_IDENTITY_PATH = EXPERIMENT_ROOT / "source-identity.json"
PREFLIGHT_PATH = EXPERIMENT_ROOT / "preflight.json"
PREFLIGHT_RECEIPT_PATH = EXPERIMENT_ROOT / "preflight.receipt.json"
PILOT_PATH = EXPERIMENT_ROOT / "pilot.json"
PILOT_RECEIPT_PATH = EXPERIMENT_ROOT / "pilot.receipt.json"
MODEL_ROOT = EXPERIMENT_ROOT / "model"
TRAINING_PATH = MODEL_ROOT / "training.json"
CHECKPOINT_PATH = MODEL_ROOT / "resolution-first-g2.pt"
CHECKPOINT_RECEIPT_PATH = MODEL_ROOT / "checkpoint.receipt.json"
SUPPORT_PATH = MODEL_ROOT / "support-calibration.json"
SUPPORT_RECEIPT_PATH = MODEL_ROOT / "support-calibration.receipt.json"
MARGIN_PATH = MODEL_ROOT / "margin-calibration.json"
MARGIN_RECEIPT_PATH = MODEL_ROOT / "margin-calibration.receipt.json"
SOURCE_REVISION = "de701b36355d5ec582df30f4223aabde7bc537df"
RUNTIME_SHA256 = "4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79"
FROZEN_V5_SHA256 = (
    "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d"
)
EXPECTED_G1_COLLECTOR_SHA256 = (
    "c54466923c3a8d7f16c70ce2b68f0f6918405e5c84e33e2c1d2f641d61c7637e"
)
EXPECTED_STRATEGY_B_SHA256 = {
    "barrier_core": (
        "b532547fa2e87afe441c8fbc7edaadfdcee48655dc1f71ba56fd279a39953e84"
    ),
    "campaign": (
        "ebb5c0e770fb3722da3c6528a9b26565ead05b16dddcb46f684aa79aad056567"
    ),
    "campaign_metrics": (
        "95df58a0345fa4f80c7fa41eea5b3fff79e70ac1647c05e6157cdc694c880e60"
    ),
}
EXPECTED_EXTERNAL_IDENTITIES = {
    "source_revision": "de701b36355d5ec582df30f4223aabde7bc537df",
    "protocol_sha256": (
        "6dfb2ffa3a76cc00447e3dcf889f6209a17ff5e2f4c3382fe0959bbabbd52991"
    ),
    "runtime_sha256": (
        "4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79"
    ),
    "frozen_v5_sha256": (
        "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d"
    ),
    "joint_v2_sha256": (
        "dc7009fc18a322eca5dace55b9baf982b6ced26c18517af752aab0f6365d362e"
    ),
}
B_SHA256 = EXPECTED_STRATEGY_B_SHA256
_G1_CACHE: ModuleType | None = None
SPLIT_COUNTS = {
    "train-board": 32,
    "support-board": 24,
    "margin-board": 32,
    "offline-board": 16,
}
HORIZON = 2_000
EXACT_HORIZON = 768
QUERY_STRIDE = 6
MAXIMUM_QUERIES = 4
PREREGISTRATION_STATUS = "source_frozen_preflight_authorized"
_CAMPAIGN_VALIDATOR_CACHE: tuple[str, ModuleType] | None = None
_NUMPY_GENERIC_TYPE: type | None = None
_NUMPY_SCALAR_TYPES: tuple[type, ...] | None = None
_GYM_BODY_SEQUENCE_TYPE: type | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_regular(path: Path) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.resolve() != path
    ):
        raise RuntimeError(
            f"R3H G2 evidence is not a direct single-link regular file: {path}"
        )


def _read_json(path: Path) -> dict[str, Any]:
    _require_regular(path)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"R3H G2 JSON artifact is not an object: {path}")
    return value


def _load_g1() -> ModuleType:
    global _G1_CACHE
    _validate_pycache_boundary()
    preregistration = _read_json(
        EXPERIMENT_ROOT / "preregistration.json"
    )
    frozen = preregistration.get("frozen_source_sha256")
    anchor = (
        frozen.get("g1_collector") if isinstance(frozen, Mapping) else None
    )
    if (
        preregistration.get("experiment_id") != EXPERIMENT_ID
        or preregistration.get("source_revision") != SOURCE_REVISION
        or preregistration.get("development_only") is not True
        or preregistration.get("sealed_test_allowed") is not False
        or preregistration.get("status") != PREREGISTRATION_STATUS
        or anchor != EXPECTED_G1_COLLECTOR_SHA256
    ):
        raise RuntimeError("R3H G2 G1 bootstrap anchor is malformed")
    path = (REPOSITORY / "benchmarks/r3h_exact_collect.py").resolve()
    _require_regular(path)
    if _sha256_file(path) != anchor:
        raise RuntimeError("R3H G2 G1 source changed before import")
    if _G1_CACHE is None:
        spec = importlib.util.spec_from_file_location(
            "_irisu_r3h_g2_g1_frozen", path
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load frozen R3H G1 collector")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _G1_CACHE = module
    if (
        Path(str(_G1_CACHE.__file__)).resolve() != path
        or _sha256_file(path) != anchor
    ):
        raise RuntimeError("R3H G2 G1 validator identity changed")
    return _G1_CACHE


def _verify_source_identity(
    *, refresh_runtime: bool = False
) -> tuple[dict[str, Any], str]:
    global _NUMPY_GENERIC_TYPE, _NUMPY_SCALAR_TYPES

    _validate_pycache_boundary()
    identity = _read_json(SOURCE_IDENTITY_PATH)
    preregistration_anchor = _read_json(
        EXPERIMENT_ROOT / "preregistration.json"
    )
    frozen_anchor = preregistration_anchor.get("frozen_source_sha256")
    campaign_anchor = (
        frozen_anchor.get("campaign")
        if isinstance(frozen_anchor, Mapping)
        else None
    )
    if (
        preregistration_anchor.get("experiment_id") != EXPERIMENT_ID
        or preregistration_anchor.get("source_revision") != SOURCE_REVISION
        or preregistration_anchor.get("development_only") is not True
        or preregistration_anchor.get("sealed_test_allowed") is not False
        or preregistration_anchor.get("status") != PREREGISTRATION_STATUS
        or not isinstance(campaign_anchor, str)
        or len(campaign_anchor) != 64
        or any(
            character not in "0123456789abcdef"
            for character in campaign_anchor
        )
    ):
        raise RuntimeError("R3H G2 preregistration campaign anchor is malformed")
    files = identity.get("files")
    if not isinstance(files, Mapping):
        raise RuntimeError("R3H G2 source identity lacks its file closure")
    expected_campaign = (
        REPOSITORY / "benchmarks/rl_r3h_g2.py"
    ).resolve()
    campaign_row = files.get("campaign")
    _require_regular(expected_campaign)
    if (
        not isinstance(campaign_row, Mapping)
        or campaign_row.get("path") != str(expected_campaign)
        or campaign_row.get("sha256") != campaign_anchor
        or _sha256_file(expected_campaign) != campaign_anchor
    ):
        raise RuntimeError("R3H G2 campaign changed before validator load")
    campaign_runtime = _validate_all_incomplete(
        campaign_sha256=str(campaign_row["sha256"])
    )
    if (
        identity.get("schema") != "irisu-r3h-g2-source-identity-v5"
        or identity.get("experiment_id") != EXPERIMENT_ID
        or identity.get("source_revision") != SOURCE_REVISION
        or identity.get("runtime")
        != campaign_runtime.runtime_identity(refresh=refresh_runtime)
    ):
        raise RuntimeError("foreign R3H G2 source identity")
    campaign_runtime.validate_loaded_runtime()
    numpy_module = getattr(campaign_runtime, "np", None)
    numpy_generic = getattr(numpy_module, "generic", None)
    scalar_type_rows = getattr(numpy_module, "ScalarType", None)
    if (
        sys.modules.get("numpy") is not numpy_module
        or not isinstance(numpy_generic, type)
        or type(scalar_type_rows) is not tuple
    ):
        raise RuntimeError("R3H G2 campaign NumPy binding is not trusted")
    numpy_scalar_types = tuple(
        scalar_type
        for scalar_type in scalar_type_rows
        if isinstance(scalar_type, type)
        and issubclass(scalar_type, numpy_generic)
        and numpy_module.dtype(scalar_type).kind in "biuf"
    )
    if not numpy_scalar_types:
        raise RuntimeError("R3H G2 trusted NumPy scalar set is empty")
    if _NUMPY_GENERIC_TYPE is None:
        _NUMPY_GENERIC_TYPE = numpy_generic
    elif _NUMPY_GENERIC_TYPE is not numpy_generic:
        raise RuntimeError("R3H G2 trusted NumPy scalar type changed")
    if _NUMPY_SCALAR_TYPES is None:
        _NUMPY_SCALAR_TYPES = numpy_scalar_types
    elif (
        type(_NUMPY_SCALAR_TYPES) is not tuple
        or len(_NUMPY_SCALAR_TYPES) != len(numpy_scalar_types)
        or any(
            actual is not expected
            for actual, expected in zip(
                tuple.__iter__(_NUMPY_SCALAR_TYPES),
                tuple.__iter__(numpy_scalar_types),
                strict=True,
            )
        )
    ):
        raise RuntimeError("R3H G2 trusted NumPy scalar set changed")
    expected_paths = {
        "campaign": REPOSITORY / "benchmarks/rl_r3h_g2.py",
        "module": REPOSITORY / "python/irisu_pointer/resolution_first_g2.py",
        "collector": Path(__file__).resolve(),
        "protocol": PROTOCOL_PATH,
        "focused_tests": (
            REPOSITORY / "tests/test_pointer_resolution_first_g2.py"
        ),
        "base_tests": (
            REPOSITORY / "tests/test_pointer_resolution_first.py"
        ),
        "preregistration": EXPERIMENT_ROOT / "preregistration.json",
        "resolution_first": (
            REPOSITORY / "python/irisu_pointer/resolution_first.py"
        ),
        "encoding": REPOSITORY / "python/irisu_rl/encoding.py",
        "schema": REPOSITORY / "python/irisu_rl/schema.py",
        "pointer_package_init": (
            REPOSITORY / "python/irisu_pointer/__init__.py"
        ),
        "rl_package_init": REPOSITORY / "python/irisu_rl/__init__.py",
        "g1_collector": REPOSITORY / "benchmarks/r3h_exact_collect.py",
        "pyproject": REPOSITORY / "pyproject.toml",
        "uv_lock": REPOSITORY / "uv.lock",
        "env_package_init": REPOSITORY / "python/irisu_env/__init__.py",
    }
    for dependency in sorted((REPOSITORY / "python/irisu_env").glob("*.py")):
        if dependency.name != "__init__.py":
            expected_paths[f"env_{dependency.stem}"] = dependency
    if (REPOSITORY / "tests/test_r3h_g2_campaign.py").exists():
        expected_paths["campaign_tests"] = (
            REPOSITORY / "tests/test_r3h_g2_campaign.py"
        )
    learner_source = expected_paths["module"].read_text()
    if "from .policy import" in learner_source:
        expected_paths.update(
            {
                "policy": REPOSITORY / "python/irisu_pointer/policy.py",
                "pointer_action": (
                    REPOSITORY / "python/irisu_pointer/action.py"
                ),
                "pointer_experts": (
                    REPOSITORY / "python/irisu_pointer/experts.py"
                ),
                "rl_actions": REPOSITORY / "python/irisu_rl/actions.py",
            }
        )
    if set(files) != set(expected_paths):
        raise RuntimeError("R3H G2 source identity file closure changed")
    for name, expected_path in expected_paths.items():
        path = expected_path.resolve()
        row = files.get(name)
        if not isinstance(row, Mapping):
            raise RuntimeError(f"R3H G2 source identity lacks {name}")
        _require_regular(path)
        if (
            row.get("path") != str(path)
            or row.get("sha256") != _sha256_file(path)
        ):
            raise RuntimeError(f"R3H G2 source dependency changed: {name}")
    preregistration = _read_json(
        Path(str(files["preregistration"]["path"]))
    )
    frozen_source = preregistration.get("frozen_source_sha256")
    if (
        preregistration.get("experiment_id") != EXPERIMENT_ID
        or preregistration.get("source_revision") != SOURCE_REVISION
        or preregistration.get("development_only") is not True
        or preregistration.get("sealed_test_allowed") is not False
        or preregistration.get("outcomes_viewed_before_preregistration")
        is not False
        or preregistration.get("generation_01_outcomes_are_training_data")
        is not False
        or preregistration.get("generation_02_outcomes_collected_before_freeze")
        is not False
        or preregistration.get("status") != PREREGISTRATION_STATUS
        or preregistration.get("protocol_sha256")
        != files["protocol"]["sha256"]
        or frozen_source
        != {
            name: row["sha256"]
            for name, row in files.items()
            if name != "preregistration"
        }
        or preregistration.get("runtime_manifest") != identity.get("runtime")
    ):
        raise RuntimeError("R3H G2 preregistration is malformed")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise RuntimeError("R3H G2 repository revision changed")
    return identity, _sha256_file(SOURCE_IDENTITY_PATH)


def _receipt(
    artifact: Path,
    receipt_path: Path,
    *,
    schema: str,
    source_identity_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _read_json(receipt_path)
    report = _read_json(artifact)
    if (
        receipt.get("schema") != schema
        or receipt.get("artifact") != str(artifact)
        or receipt.get("artifact_sha256") != _sha256_file(artifact)
        or receipt.get("source_identity_sha256") != source_identity_sha256
    ):
        raise RuntimeError(f"R3H G2 receipt mismatch: {receipt_path}")
    return report, receipt


def _collection_identity_sha256(split: str) -> str:
    _validate_collection_split(
        split, allow_writer_temps=False, require_complete=True
    )
    rows: list[dict[str, str]] = []
    for index in range(SPLIT_COUNTS[split]):
        seed = derive_seed(split, index)
        stem = OUTPUT_ROOT / split / f"{seed:010d}"
        for path in (stem.with_suffix(".json"), stem.with_suffix(".queries.jsonl")):
            _require_regular(path)
            rows.append({"path": str(path), "sha256": _sha256_file(path)})
    return _canonical_sha256(
        {
            "schema": "irisu-r3h-g2-collection-identity-v1",
            "experiment_id": EXPERIMENT_ID,
            "split": split,
            "files": rows,
        }
    )


def _require_development_report(
    report: Mapping[str, Any], *, schema: str, source_identity_sha256: str
) -> None:
    if (
        report.get("schema") != schema
        or report.get("development_only") is not True
        or report.get("sealed_test_allowed") is not False
        or report.get("source_identity_sha256") != source_identity_sha256
    ):
        raise RuntimeError(f"foreign R3H G2 {schema} report")


def _validate_preflight(source_identity_sha256: str) -> dict[str, Any]:
    report, receipt = _receipt(
        PREFLIGHT_PATH,
        PREFLIGHT_RECEIPT_PATH,
        schema="irisu-r3h-g2-preflight-receipt-v1",
        source_identity_sha256=source_identity_sha256,
    )
    _require_development_report(
        report,
        schema="irisu-r3h-g2-preflight-v1",
        source_identity_sha256=source_identity_sha256,
    )
    manifests = report.get("seed_manifest_sha256")
    if not isinstance(manifests, Mapping):
        raise RuntimeError("R3H G2 preflight lacks seed manifests")
    for split, count in SPLIT_COUNTS.items():
        path = EXPERIMENT_ROOT / "seed-manifests" / f"{split}.json"
        manifest = _read_json(path)
        expected = [
            {"index": index, "seed": derive_seed(split, index)}
            for index in range(count)
        ]
        if (
            manifests.get(split) != _sha256_file(path)
            or manifest.get("schema") != "irisu-r3h-g2-seed-manifest-v1"
            or manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("split") != split
            or manifest.get("rows") != expected
        ):
            raise RuntimeError(f"R3H G2 {split} seed manifest mismatch")
    return receipt


def _validate_pilot(source_identity_sha256: str) -> dict[str, Any]:
    _validate_preflight(source_identity_sha256)
    report, receipt = _receipt(
        PILOT_PATH,
        PILOT_RECEIPT_PATH,
        schema="irisu-r3h-g2-pilot-receipt-v1",
        source_identity_sha256=source_identity_sha256,
    )
    _require_development_report(
        report,
        schema="irisu-r3h-g2-structural-pilot-v1",
        source_identity_sha256=source_identity_sha256,
    )
    if (
        report.get("passed") is not True
        or receipt.get("passed") is not True
        or not isinstance(report.get("dataset_sha256"), str)
        or receipt.get("dataset_sha256") != report.get("dataset_sha256")
        or report.get("collection_identity_sha256")
        != _collection_identity_sha256("train-board")
        or receipt.get("collection_identity_sha256")
        != report.get("collection_identity_sha256")
    ):
        raise RuntimeError("R3H G2 structural pilot is absent or terminal NO-GO")
    return receipt


def _validate_checkpoint(source_identity_sha256: str) -> dict[str, Any]:
    pilot_receipt = _validate_pilot(source_identity_sha256)
    report, receipt = _receipt(
        TRAINING_PATH,
        CHECKPOINT_RECEIPT_PATH,
        schema="irisu-r3h-g2-checkpoint-receipt-v1",
        source_identity_sha256=source_identity_sha256,
    )
    _require_development_report(
        report,
        schema="irisu-r3h-g2-training-v1",
        source_identity_sha256=source_identity_sha256,
    )
    _require_regular(CHECKPOINT_PATH)
    manifest = report.get("model_manifest")
    if (
        receipt.get("checkpoint") != str(CHECKPOINT_PATH)
        or report.get("checkpoint") != str(CHECKPOINT_PATH)
        or receipt.get("checkpoint_sha256") != _sha256_file(CHECKPOINT_PATH)
        or report.get("checkpoint_sha256") != receipt.get("checkpoint_sha256")
        or receipt.get("dataset_sha256") != pilot_receipt.get("dataset_sha256")
        or report.get("dataset_sha256") != pilot_receipt.get("dataset_sha256")
        or report.get("collection_identity_sha256")
        != _collection_identity_sha256("train-board")
        or receipt.get("collection_identity_sha256")
        != report.get("collection_identity_sha256")
        or not isinstance(manifest, Mapping)
        or receipt.get("model_manifest_sha256")
        != _canonical_sha256(manifest)
    ):
        raise RuntimeError("R3H G2 checkpoint authorization mismatch")
    return receipt


def _validate_support(source_identity_sha256: str) -> dict[str, Any]:
    checkpoint_receipt = _validate_checkpoint(source_identity_sha256)
    report, receipt = _receipt(
        SUPPORT_PATH,
        SUPPORT_RECEIPT_PATH,
        schema="irisu-r3h-g2-support-calibration-receipt-v1",
        source_identity_sha256=source_identity_sha256,
    )
    _require_development_report(
        report,
        schema="irisu-r3h-g2-support-calibration-v1",
        source_identity_sha256=source_identity_sha256,
    )
    auroc = report.get("resolution_auroc")
    support = report.get("support")
    grid = report.get("support_grid")
    passing = (
        [
            point
            for point in grid
            if isinstance(point, Mapping) and point.get("passed") is True
        ]
        if isinstance(grid, list)
        else []
    )
    if (
        report.get("passed") is not True
        or receipt.get("passed") is not True
        or report.get("checkpoint_sha256")
        != checkpoint_receipt.get("checkpoint_sha256")
        or receipt.get("checkpoint_sha256")
        != checkpoint_receipt.get("checkpoint_sha256")
        or not isinstance(auroc, Mapping)
        or auroc.get("passed") is not True
        or type(auroc.get("auroc")) not in (int, float)
        or not math.isfinite(float(auroc["auroc"]))
        or float(auroc["auroc"]) < 0.75
        or receipt.get("resolution_auroc_sha256") != _canonical_sha256(auroc)
        or not isinstance(support, Mapping)
        or receipt.get("support_sha256") != _canonical_sha256(support)
        or not isinstance(grid, list)
        or support.get("grid") != grid
        or support.get("resolution_auroc") != auroc.get("auroc")
        or not passing
        or support.get("threshold") != passing[0].get("threshold")
        or receipt.get("support_dataset_sha256")
        != report.get("support_dataset_sha256")
        or report.get("support_collection_identity_sha256")
        != _collection_identity_sha256("support-board")
        or receipt.get("support_collection_identity_sha256")
        != report.get("support_collection_identity_sha256")
    ):
        raise RuntimeError("R3H G2 support/AUROC authorization is not GO")
    return receipt


def _validate_margin(source_identity_sha256: str) -> dict[str, Any]:
    support_receipt = _validate_support(source_identity_sha256)
    report, receipt = _receipt(
        MARGIN_PATH,
        MARGIN_RECEIPT_PATH,
        schema="irisu-r3h-g2-margin-calibration-receipt-v1",
        source_identity_sha256=source_identity_sha256,
    )
    _require_development_report(
        report,
        schema="irisu-r3h-g2-margin-calibration-v1",
        source_identity_sha256=source_identity_sha256,
    )
    selective = report.get("selective")
    q = selective.get("q") if isinstance(selective, Mapping) else None
    support_receipt_sha = _sha256_file(SUPPORT_RECEIPT_PATH)
    if (
        report.get("passed") is not True
        or receipt.get("passed") is not True
        or receipt.get("finite_q_b2") is not True
        or report.get("calibration_target") != "absolute_b2"
        or receipt.get("calibration_target") != "absolute_b2"
        or report.get("checkpoint_sha256")
        != support_receipt.get("checkpoint_sha256")
        or receipt.get("checkpoint_sha256")
        != support_receipt.get("checkpoint_sha256")
        or report.get("support_calibration_receipt_sha256")
        != support_receipt_sha
        or receipt.get("support_calibration_receipt_sha256")
        != support_receipt_sha
        or report.get("support_calibration_sha256")
        != support_receipt.get("support_sha256")
        or receipt.get("support_calibration_sha256")
        != support_receipt.get("support_sha256")
        or not isinstance(selective, Mapping)
        or selective.get("support_calibration_sha256")
        != support_receipt.get("support_sha256")
        or type(q) not in (int, float)
        or not math.isfinite(float(q))
        or receipt.get("selective_sha256") != _canonical_sha256(selective)
        or receipt.get("margin_dataset_sha256")
        != report.get("margin_dataset_sha256")
        or report.get("margin_collection_identity_sha256")
        != _collection_identity_sha256("margin-board")
        or receipt.get("margin_collection_identity_sha256")
        != report.get("margin_collection_identity_sha256")
    ):
        raise RuntimeError("R3H G2 absolute-B2 margin authorization is not GO")
    return receipt


def _authorize_collection(split: str) -> dict[str, str]:
    _identity, source_sha = _verify_source_identity()
    if split == "train-board":
        receipt_path = PREFLIGHT_RECEIPT_PATH
        _validate_preflight(source_sha)
    elif split == "support-board":
        receipt_path = CHECKPOINT_RECEIPT_PATH
        _validate_checkpoint(source_sha)
    elif split == "margin-board":
        receipt_path = SUPPORT_RECEIPT_PATH
        _validate_support(source_sha)
    elif split == "offline-board":
        receipt_path = MARGIN_RECEIPT_PATH
        _validate_margin(source_sha)
    else:
        raise ValueError(f"forbidden R3H G2 collection split: {split!r}")
    _verify_source_identity()
    return {
        "schema": "irisu-r3h-g2-collection-authorization-v1",
        "split": split,
        "receipt": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "source_identity_sha256": source_sha,
    }


def _verify_external_constants() -> None:
    g1 = _load_g1()
    g1_path = Path(g1.__file__).resolve()
    _require_regular(g1_path)
    if (
        _sha256_file(g1_path) != EXPECTED_G1_COLLECTOR_SHA256
        or dict(g1.B_SHA256) != EXPECTED_STRATEGY_B_SHA256
        or g1.SOURCE_REVISION != EXPECTED_EXTERNAL_IDENTITIES["source_revision"]
        or g1.RUNTIME_SHA256 != EXPECTED_EXTERNAL_IDENTITIES["runtime_sha256"]
        or g1.FROZEN_V5_SHA256
        != EXPECTED_EXTERNAL_IDENTITIES["frozen_v5_sha256"]
        or g1.JOINT_V2_SHA256
        != EXPECTED_EXTERNAL_IDENTITIES["joint_v2_sha256"]
        or g1.PROTOCOL_SHA256
        != EXPECTED_EXTERNAL_IDENTITIES["protocol_sha256"]
    ):
        raise RuntimeError("R3H G2 trusted external identity constants changed")


def _validate_loaded_identities(identities: Mapping[str, object]) -> None:
    for name, expected in EXPECTED_EXTERNAL_IDENTITIES.items():
        if identities.get(name) != expected:
            raise RuntimeError(f"R3H G2 loaded external identity mismatch: {name}")


def _canonical_public_value(value: object) -> object:
    value_type = type(value)
    if (
        value is None
        or value_type is bool
        or value_type is int
        or value_type is float
        or value_type is str
    ):
        return value
    if value_type is dict:
        normalized: dict[str, object] = {}
        for key, item in dict.items(value):
            if type(key) is not str:
                raise TypeError("R3H G2 public mappings require string keys")
            if key in normalized:
                raise TypeError("R3H G2 public mappings require unique keys")
            normalized[key] = _canonical_public_value(item)
        return normalized
    if value_type is list:
        return [
            _canonical_public_value(item)
            for item in list.__iter__(value)
        ]
    if value_type is tuple or value_type is _GYM_BODY_SEQUENCE_TYPE:
        return [
            _canonical_public_value(item)
            for item in tuple.__iter__(value)
        ]
    numpy_generic = _NUMPY_GENERIC_TYPE
    numpy_scalar_types = _NUMPY_SCALAR_TYPES
    if (
        isinstance(numpy_generic, type)
        and type(numpy_scalar_types) is tuple
        and any(
            value_type is scalar_type
            for scalar_type in tuple.__iter__(numpy_scalar_types)
        )
    ):
        resolved = numpy_generic.item(value)
        resolved_type = type(resolved)
        if (
            resolved is not None
            and resolved_type is not bool
            and resolved_type is not int
            and resolved_type is not float
            and resolved_type is not str
        ):
            raise TypeError("R3H G2 NumPy scalar did not yield a native value")
        return resolved
    raise TypeError(
        f"unsupported R3H G2 public value {type(value).__name__}"
    )


def _canonical_observation(
    observation: Mapping[str, Any],
) -> tuple[dict[str, object], bytes, list[int]]:
    """Normalize once while retaining the environment's body-list order."""

    normalized = _canonical_public_value(observation)
    if not isinstance(normalized, dict):
        raise TypeError("pre-query public observation must be a mapping")
    payload = _canonical_bytes(normalized)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise TypeError("pre-query public observation must be a mapping")
    bodies = value.get("bodies")
    if not isinstance(bodies, list):
        raise TypeError("pre-query public observation bodies must be a list")
    body_ids: list[int] = []
    for body in bodies:
        if not isinstance(body, dict) or type(body.get("id")) is not int:
            raise TypeError("pre-query public body must have a plain integer id")
        body_ids.append(int(body["id"]))
    if len(body_ids) != len(set(body_ids)):
        raise ValueError("pre-query public observation has duplicate body ids")
    if type(value.get("tick")) is not int:
        raise TypeError("pre-query public observation tick must be a plain integer")
    return value, payload, body_ids


def derive_seed(split: str, index: int) -> int:
    if split not in SPLIT_COUNTS:
        raise ValueError(f"forbidden R3H G2 collection split: {split!r}")
    if type(index) is not int or not 0 <= index < SPLIT_COUNTS[split]:
        raise ValueError(f"R3H G2 seed index is outside split {split!r}")
    return int.from_bytes(
        hashlib.sha256(f"{EXPERIMENT_ID}|{split}|{index}".encode()).digest()[:4],
        "big",
        signed=False,
    )


def _verify_seed_schedule() -> None:
    seen: dict[int, tuple[str, int]] = {}
    for split, count in SPLIT_COUNTS.items():
        for index in range(count):
            seed = derive_seed(split, index)
            previous = seen.get(seed)
            if previous is not None:
                raise RuntimeError(
                    f"fresh R3H G2 seed collision: {(split, index)} and {previous}"
                )
            seen[seed] = (split, index)


def _validate_request(split: str, start: int, count: int) -> None:
    if split not in SPLIT_COUNTS:
        raise ValueError(f"forbidden R3H G2 collection split: {split!r}")
    if type(start) is not int or type(count) is not int:
        raise TypeError("start and count must be integers")
    if start < 0 or count < 1 or start + count > SPLIT_COUNTS[split]:
        raise ValueError(
            f"requested range [{start}, {start + count}) exceeds "
            f"{split!r} capacity {SPLIT_COUNTS[split]}"
        )


def _load_closed_irisu_env() -> None:
    global _GYM_BODY_SEQUENCE_TYPE

    python_root = (REPOSITORY / "python").resolve()
    package_root = (python_root / "irisu_env").resolve()
    if str(python_root) not in sys.path:
        raise RuntimeError("closed R3H G2 Python root is not active")
    expected = {
        (
            "irisu_env"
            if path.name == "__init__.py"
            else f"irisu_env.{path.stem}"
        ): path.resolve()
        for path in sorted(package_root.glob("*.py"))
    }
    for path in expected.values():
        _require_regular(path)
    for name, path in expected.items():
        existing = sys.modules.get(name)
        if existing is not None:
            existing_path = Path(
                str(getattr(existing, "__file__", ""))
            ).resolve()
            if existing_path != path:
                raise RuntimeError(
                    f"foreign preloaded R3H G2 environment module: {name}"
                )
        module = importlib.import_module(name)
        if Path(str(module.__file__)).resolve() != path:
            raise RuntimeError(
                f"R3H G2 environment module escaped closure: {name}"
            )
    package = sys.modules.get("irisu_env")
    package_path = getattr(package, "__path__", ())
    if [Path(value).resolve() for value in package_path] != [package_root]:
        raise RuntimeError("R3H G2 environment package path escaped closure")
    for name, module in tuple(sys.modules.items()):
        if name != "irisu_env" and not name.startswith("irisu_env."):
            continue
        path = expected.get(name)
        if (
            path is None
            or Path(str(getattr(module, "__file__", ""))).resolve() != path
        ):
            raise RuntimeError(
                f"R3H G2 loaded an unclosed environment module: {name}"
            )
    env_module = sys.modules.get("irisu_env.env")
    gym_body_sequence = getattr(env_module, "_GymBodySequence", None)
    gym_body_mro = getattr(gym_body_sequence, "__mro__", ())
    if (
        not isinstance(gym_body_sequence, type)
        or type(gym_body_sequence.__module__) is not str
        or gym_body_sequence.__module__ != "irisu_env.env"
        or type(gym_body_sequence.__name__) is not str
        or gym_body_sequence.__name__ != "_GymBodySequence"
        or type(gym_body_mro) is not tuple
        or len(gym_body_mro) != 3
        or gym_body_mro[0] is not gym_body_sequence
        or gym_body_mro[1] is not tuple
        or gym_body_mro[2] is not object
    ):
        raise RuntimeError("R3H G2 Gym body-sequence type is not trusted")
    if _GYM_BODY_SEQUENCE_TYPE is None:
        _GYM_BODY_SEQUENCE_TYPE = gym_body_sequence
    elif _GYM_BODY_SEQUENCE_TYPE is not gym_body_sequence:
        raise RuntimeError("R3H G2 Gym body-sequence type changed")


def _validate_closed_sys_path() -> None:
    if tuple(sys.path) != CLOSED_RUNTIME_SYS_PATH:
        raise RuntimeError("R3H G2 import path escaped its frozen closure")


def _load_frozen_b() -> tuple[ModuleType, ModuleType, dict[str, object]]:
    _verify_external_constants()
    _validate_closed_sys_path()
    _load_closed_irisu_env()
    closed_path = tuple(sys.path)
    try:
        core, campaign, identities = _load_g1()._load_frozen_b()
    finally:
        sys.path[:] = closed_path
    _validate_closed_sys_path()
    _load_closed_irisu_env()
    _validate_loaded_identities(identities)
    return core, campaign, identities


def _collector_identity(
    *,
    split: str,
    index: int,
    seed: int,
    identities: Mapping[str, object],
    authorization: Mapping[str, str],
) -> dict[str, object]:
    local_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if local_revision != SOURCE_REVISION:
        raise RuntimeError(
            f"local source revision is {local_revision}, expected {SOURCE_REVISION}"
        )
    source_identity, source_identity_sha = _verify_source_identity()
    files = source_identity["files"]
    protocol_sha = str(files["protocol"]["sha256"])
    collector_sha = str(files["collector"]["sha256"])
    g1_sha = str(files["g1_collector"]["sha256"])
    expected_authorization = _authorize_collection(split)
    if dict(authorization) != expected_authorization:
        raise RuntimeError("Generation 02 collection authorization changed")
    if source_identity_sha != authorization.get("source_identity_sha256"):
        raise RuntimeError("Generation 02 authorization/source mismatch")
    _validate_loaded_identities(identities)
    _verify_external_constants()
    return {
        "schema": "r3h-g2-exact-collector-identity-v1",
        "development_only": True,
        "sealed_test_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "source_revision": local_revision,
        "protocol_path": str(PROTOCOL_PATH),
        "protocol_sha256": protocol_sha,
        "collector_source_sha256": collector_sha,
        "generation_01_collector_source_sha256": g1_sha,
        "strategy_b_source_sha256": dict(EXPECTED_STRATEGY_B_SHA256),
        "strategy_b_identities": dict(identities),
        "authorization": dict(authorization),
        "split": split,
        "index": index,
        "seed": seed,
        "seed_derivation": (
            f'SHA256("{EXPERIMENT_ID}|<split>|<index>")[:4] big-endian'
        ),
        "horizon": HORIZON,
        "exact_horizon": EXACT_HORIZON,
        "query_stride": QUERY_STRIDE,
        "maximum_queries": MAXIMUM_QUERIES,
        "live_policy": "frozen-v5-incumbent-only",
        "alternatives_executed": False,
        "pre_query_observation_encoding": "canonical-json-sort-keys-v1",
        "pre_query_body_order": "environment-emitted-order",
        "pre_query_observation_hash": "sha256-canonical-json",
    }


def _paths(split: str, seed: int) -> tuple[Path, Path]:
    stem = OUTPUT_ROOT / split / f"{seed:010d}"
    return stem.with_suffix(".json"), stem.with_suffix(".queries.jsonl")


def _fsync_directory(path: Path) -> None:
    if (
        not path.is_dir()
        or path.is_symlink()
        or path.resolve() != path
    ):
        raise RuntimeError(f"indirect R3H G2 directory: {path}")
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(path: Path) -> None:
    path = Path(os.path.abspath(path))
    missing: list[Path] = []
    cursor = path
    while not cursor.exists():
        if cursor.is_symlink():
            raise RuntimeError(f"indirect R3H G2 directory: {cursor}")
        missing.append(cursor)
        if cursor.parent == cursor:
            raise RuntimeError(f"cannot create R3H G2 directory: {path}")
        cursor = cursor.parent
    _fsync_directory(cursor)
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            pass
        _fsync_directory(directory)
        _fsync_directory(directory.parent)
    _fsync_directory(path)


def _writer_temp_candidates(path: Path) -> tuple[Path, ...]:
    if not path.parent.exists():
        if path.parent.is_symlink():
            raise RuntimeError(f"indirect R3H G2 artifact parent: {path.parent}")
        return ()
    if (
        not path.parent.is_dir()
        or path.parent.is_symlink()
        or path.parent.resolve() != path.parent
    ):
        raise RuntimeError(f"indirect R3H G2 artifact parent: {path.parent}")
    prefix = f".{path.name}."
    pattern = re.compile(rf"^{re.escape(path.name)}\.tmp-[0-9]+-[0-9]+$")
    output: list[Path] = []
    for candidate in path.parent.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        if not pattern.fullmatch(candidate.name[1:]):
            raise RuntimeError(f"unknown R3H G2 writer remnant: {candidate}")
        output.append(candidate)
    return tuple(sorted(output))


def _quarantine_link(
    path: Path,
    destination_parent: Path,
    *,
    destination_stem: str,
) -> Path:
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise RuntimeError(f"indirect R3H G2 remnant: {path}")
    _mkdir_durable(destination_parent)
    for candidate in destination_parent.iterdir():
        _validate_quarantine_entry(candidate)
        if os.path.samefile(candidate, path):
            current_suffix = (
                candidate.name[len(destination_stem) + 1 :]
                if candidate.name.startswith(f"{destination_stem}.")
                else ""
            )
            if (
                len(current_suffix.split(".")) != 3
                or not all(
                    piece.isdigit() for piece in current_suffix.split(".")
                )
                or path.stat().st_nlink != 2
            ):
                raise RuntimeError(f"foreign R3H G2 quarantine links: {path}")
            _fsync_directory(destination_parent)
            path.unlink()
            _fsync_directory(path.parent)
            if candidate.stat().st_nlink != 1:
                raise RuntimeError(f"foreign R3H G2 quarantine link: {candidate}")
            return candidate
        if candidate.stat().st_nlink != 1:
            raise RuntimeError(f"foreign R3H G2 quarantine entry: {candidate}")
    if path.stat().st_nlink != 1:
        raise RuntimeError(f"foreign R3H G2 remnant links: {path}")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    for sequence in range(1_000):
        destination = destination_parent / (
            f"{destination_stem}.{os.getpid()}.{time.time_ns()}.{sequence}"
        )
        try:
            os.link(path, destination)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError("cannot allocate R3H G2 quarantine identity")
    _fsync_directory(destination_parent)
    path.unlink()
    _fsync_directory(path.parent)
    if (
        destination.is_symlink()
        or not destination.is_file()
        or destination.stat().st_nlink != 1
        or destination.resolve() != destination
    ):
        raise RuntimeError(f"invalid R3H G2 quarantine result: {destination}")
    return destination


def _validate_quarantine_entry(path: Path) -> tuple[str, str, str, str, str]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
    ):
        raise RuntimeError(f"indirect R3H G2 quarantine entry: {path}")
    pieces = path.name.rsplit(".", 4)
    if (
        len(pieces) != 5
        or not pieces[0]
        or len(pieces[1]) != 64
        or any(character not in "0123456789abcdef" for character in pieces[1])
        or any(not piece.isdigit() for piece in pieces[2:])
        or int(pieces[2]) <= 0
        or int(pieces[3]) <= 0
        or _sha256_file(path) != pieces[1]
    ):
        raise RuntimeError(f"unknown R3H G2 quarantine entry: {path}")
    return tuple(pieces)  # type: ignore[return-value]


def _require_interrupted_quarantine_link(
    path: Path,
    destination_parent: Path,
    *,
    destination_stem: str,
) -> None:
    if (
        path.stat().st_nlink != 2
        or not destination_parent.is_dir()
        or destination_parent.is_symlink()
        or destination_parent.resolve() != destination_parent
    ):
        raise RuntimeError(f"foreign R3H G2 remnant links: {path}")
    matches: list[Path] = []
    for candidate in destination_parent.iterdir():
        _validate_quarantine_entry(candidate)
        if not os.path.samefile(candidate, path):
            continue
        suffix = (
            candidate.name[len(destination_stem) + 1 :]
            if candidate.name.startswith(f"{destination_stem}.")
            else ""
        )
        if (
            len(suffix.split(".")) != 3
            or not all(piece.isdigit() for piece in suffix.split("."))
        ):
            raise RuntimeError(f"foreign R3H G2 quarantine links: {path}")
        matches.append(candidate)
    if len(matches) != 1:
        raise RuntimeError(f"foreign R3H G2 remnant links: {path}")


def _expected_collection_names(split: str) -> set[str]:
    names: set[str] = set()
    for index in range(SPLIT_COUNTS[split]):
        seed = derive_seed(split, index)
        names.add(f"{seed:010d}.json")
        names.add(f"{seed:010d}.queries.jsonl")
    return names


def _validate_collection_incomplete() -> None:
    root = OUTPUT_ROOT / "_incomplete"
    if not root.exists():
        if root.is_symlink():
            raise RuntimeError(f"indirect R3H G2 incomplete root: {root}")
        return
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise RuntimeError(f"indirect R3H G2 incomplete root: {root}")
    allowed = {"_atomic", *SPLIT_COUNTS}
    for namespace in root.iterdir():
        if (
            namespace.name not in allowed
            or not namespace.is_dir()
            or namespace.is_symlink()
            or namespace.resolve() != namespace
        ):
            raise RuntimeError(f"unknown R3H G2 incomplete namespace: {namespace}")
        if namespace.name == "_atomic":
            split_directories = tuple(namespace.iterdir())
        else:
            split_directories = (namespace,)
        for directory in split_directories:
            split = (
                directory.name
                if namespace.name == "_atomic"
                else namespace.name
            )
            if (
                split not in SPLIT_COUNTS
                or not directory.is_dir()
                or directory.is_symlink()
                or directory.resolve() != directory
            ):
                raise RuntimeError(
                    f"unknown R3H G2 quarantine namespace: {directory}"
                )
            expected = _expected_collection_names(split)
            for candidate in directory.iterdir():
                head, _digest, _pid, _stamp, _sequence = (
                    _validate_quarantine_entry(candidate)
                )
                if namespace.name == "_atomic":
                    target = head
                    source_pattern = f".{target}.tmp-"
                    identity = None
                else:
                    try:
                        target, identity = head.rsplit(".", 1)
                    except ValueError as exc:
                        raise RuntimeError(
                            f"unknown R3H G2 partial identity: {candidate}"
                        ) from exc
                    if (
                        len(identity) != 64
                        or any(
                            character not in "0123456789abcdef"
                            for character in identity
                        )
                    ):
                        raise RuntimeError(
                            f"unknown R3H G2 partial identity: {candidate}"
                        )
                    source_pattern = target
                if target not in expected:
                    raise RuntimeError(
                        f"unknown R3H G2 quarantine target: {candidate}"
                    )
                if identity is not None:
                    if target.endswith(".queries.jsonl"):
                        rows = [
                            json.loads(line)
                            for line in candidate.read_text().splitlines()
                            if line.strip()
                        ]
                        bound = bool(rows) and all(
                            row.get("collector_identity_sha256") == identity
                            for row in rows
                        )
                    else:
                        value = json.loads(candidate.read_text())
                        bound = (
                            isinstance(value, Mapping)
                            and value.get("collector_identity_sha256") == identity
                        )
                    if not bound:
                        raise RuntimeError(
                            f"foreign R3H G2 partial identity: {candidate}"
                        )
                links = candidate.stat().st_nlink
                if links == 1:
                    continue
                source_parent = OUTPUT_ROOT / split
                if (
                    links != 2
                    or not source_parent.is_dir()
                    or source_parent.is_symlink()
                    or source_parent.resolve() != source_parent
                ):
                    raise RuntimeError(
                        f"foreign R3H G2 quarantine links: {candidate}"
                    )
                matches = [
                    source
                    for source in source_parent.iterdir()
                    if (
                        source.is_file()
                        and not source.is_symlink()
                        and os.path.samefile(source, candidate)
                    )
                ]
                exact_source = (
                    len(matches) == 1
                    and (
                        (
                            namespace.name == "_atomic"
                            and matches[0].name.startswith(source_pattern)
                            and len(
                                matches[0].name[len(source_pattern) :].split("-")
                            )
                            == 2
                            and all(
                                piece.isdigit()
                                for piece in matches[0]
                                .name[len(source_pattern) :]
                                .split("-")
                            )
                        )
                        or (
                            namespace.name != "_atomic"
                            and matches[0].name == source_pattern
                        )
                    )
                )
                if not exact_source:
                    raise RuntimeError(
                        f"foreign R3H G2 quarantine links: {candidate}"
                    )


def _validate_all_incomplete(*, campaign_sha256: str) -> ModuleType:
    global _CAMPAIGN_VALIDATOR_CACHE
    _validate_pycache_boundary()
    _validate_collection_incomplete()
    expected = (REPOSITORY / "benchmarks/rl_r3h_g2.py").resolve()
    _require_regular(expected)
    if (
        len(campaign_sha256) != 64
        or any(character not in "0123456789abcdef" for character in campaign_sha256)
        or _sha256_file(expected) != campaign_sha256
    ):
        raise RuntimeError("R3H G2 campaign changed before validator load")
    cached = _CAMPAIGN_VALIDATOR_CACHE
    if cached is not None and cached[0] == campaign_sha256:
        campaign = cached[1]
    else:
        spec = importlib.util.spec_from_file_location(
            "_irisu_r3h_g2_campaign_validation", expected
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("cannot load exact R3H G2 campaign validator")
        campaign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(campaign)
        _CAMPAIGN_VALIDATOR_CACHE = (campaign_sha256, campaign)
    if Path(str(campaign.__file__)).resolve() != expected:
        raise RuntimeError("R3H G2 campaign validator path changed")
    if campaign.EXPERIMENT.resolve() != EXPERIMENT_ROOT.resolve():
        raise RuntimeError("R3H G2 campaign incomplete root changed")
    campaign._validate_campaign_incomplete()
    return campaign


def _quarantine_writer_temp(path: Path, target: Path) -> Path:
    digest = _sha256_file(path)
    return _quarantine_link(
        path,
        OUTPUT_ROOT
        / "_incomplete"
        / "_atomic"
        / target.parent.name,
        destination_stem=f"{target.name}.{digest}",
    )


def _recover_writer_temps(
    path: Path, *, allow_linked_remnant: bool = False
) -> None:
    linked = path.exists() or path.is_symlink()
    if linked and (path.is_symlink() or not path.is_file()):
        raise RuntimeError(f"indirect R3H G2 collection artifact: {path}")
    for temporary in _writer_temp_candidates(path):
        if temporary.is_symlink() or not temporary.is_file():
            raise RuntimeError(f"indirect R3H G2 writer remnant: {temporary}")
        if linked and os.path.samefile(temporary, path):
            temporary.unlink()
            _fsync_directory(path.parent)
            continue
        _quarantine_writer_temp(temporary, path)
    if (
        linked
        and path.stat().st_nlink != 1
        and not (
            allow_linked_remnant
            and path.stat().st_nlink == 2
        )
    ):
        raise RuntimeError(f"R3H G2 artifact has foreign hardlinks: {path}")


def _validate_collection_split(
    split: str,
    *,
    allow_writer_temps: bool,
    require_complete: bool,
) -> None:
    if split not in SPLIT_COUNTS:
        raise ValueError(f"forbidden R3H G2 collection split: {split!r}")
    _validate_collection_incomplete()
    directory = OUTPUT_ROOT / split
    if not directory.exists():
        if directory.is_symlink():
            raise RuntimeError(f"indirect R3H G2 collection directory: {directory}")
        if require_complete:
            raise RuntimeError(f"absent R3H G2 collection directory: {directory}")
        return
    if (
        not directory.is_dir()
        or directory.is_symlink()
        or directory.resolve() != directory
    ):
        raise RuntimeError(f"indirect R3H G2 collection directory: {directory}")
    expected: set[str] = set()
    for index in range(SPLIT_COUNTS[split]):
        seed = derive_seed(split, index)
        expected.add(f"{seed:010d}.json")
        expected.add(f"{seed:010d}.queries.jsonl")
    temp_pattern = re.compile(
        r"^\.(?P<target>.+)\.tmp-[0-9]+-[0-9]+$"
    )
    finals: dict[str, Path] = {}
    temps: list[tuple[str, Path]] = []
    for entry in directory.iterdir():
        if (
            entry.is_symlink()
            or not entry.is_file()
            or entry.resolve() != entry
        ):
            raise RuntimeError(f"foreign R3H G2 collection entry: {entry}")
        if entry.name in expected:
            finals[entry.name] = entry
            continue
        match = temp_pattern.fullmatch(entry.name)
        target = match.group("target") if match is not None else None
        if target not in expected:
            raise RuntimeError(f"unknown R3H G2 collection entry: {entry}")
        if not allow_writer_temps:
            raise RuntimeError(f"unfinished R3H G2 collection writer: {entry}")
        temps.append((str(target), entry))
    by_target: dict[str, list[Path]] = {}
    for target, temporary in temps:
        by_target.setdefault(target, []).append(temporary)
    for name, final in finals.items():
        links = final.stat().st_nlink
        if links == 1:
            continue
        matching = [
            temporary
            for temporary in by_target.get(name, ())
            if os.path.samefile(final, temporary)
        ]
        partial_parent = OUTPUT_ROOT / "_incomplete" / split
        partial_matching = (
            [
                candidate
                for candidate in partial_parent.iterdir()
                if candidate.is_file()
                and not candidate.is_symlink()
                and os.path.samefile(final, candidate)
            ]
            if partial_parent.is_dir()
            and not partial_parent.is_symlink()
            and partial_parent.resolve() == partial_parent
            else []
        )
        if (
            links != 2
            or (
                len(matching) != 1
                and len(partial_matching) != 1
            )
            or (matching and partial_matching)
        ):
            raise RuntimeError(f"foreign R3H G2 collection links: {final}")
    for target, temporary in temps:
        links = temporary.stat().st_nlink
        if links == 1:
            continue
        final = finals.get(target)
        if (
            links != 2
            or final is None
            or not os.path.samefile(temporary, final)
        ):
            raise RuntimeError(f"foreign R3H G2 writer links: {temporary}")
    if require_complete and set(finals) != expected:
        raise RuntimeError(
            f"incomplete R3H G2 {split} collection: "
            f"missing={sorted(expected - set(finals))}"
        )


def _write_new(path: Path, payload: bytes) -> None:
    _mkdir_durable(path.parent)
    _recover_writer_temps(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite R3H G2 artifact {path}")
    if path.parent.exists() and path.parent.is_symlink():
        raise RuntimeError(f"refusing indirect R3H G2 artifact parent {path.parent}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    linked = False
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        linked = True
        _fsync_directory(path.parent)
    finally:
        if linked:
            temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def _quarantine_partial(path: Path, *, identity_sha256: str) -> Path:
    if not path.is_file() or path.is_symlink() or path.resolve() != path:
        raise RuntimeError(f"cannot quarantine non-regular partial {path}")
    digest = _sha256_file(path)
    return _quarantine_link(
        path,
        OUTPUT_ROOT
        / "_incomplete"
        / path.parent.name,
        destination_stem=f"{path.name}.{identity_sha256}.{digest}",
    )


def _validate_query_row(
    row: Mapping[str, object],
    *,
    query_index: int,
    identity_sha256: str,
    split: str,
    index: int,
    seed: int,
) -> None:
    observation = row.get("pre_query_public_observation")
    if not isinstance(observation, Mapping):
        raise RuntimeError("R3H G2 query lacks its public observation")
    canonical, payload, body_ids = _canonical_observation(observation)
    query = row.get("exact_query")
    if not isinstance(query, Mapping):
        raise RuntimeError("R3H G2 query lacks exact-query evidence")
    tick = canonical["tick"]
    if (
        row.get("schema") != "r3h-g2-exact-shadow-query-v1"
        or row.get("collector_identity_sha256") != identity_sha256
        or int(row.get("query_index", -1)) != query_index
        or row.get("split") != split
        or int(row.get("index", -1)) != index
        or int(row.get("seed", -1)) != seed
        or int(row.get("executed_ordinal", -1)) != 0
        or row.get("alternatives_executed") is not False
        or row.get("live_state_restored_exactly") is not True
        or row.get("public_observation_unchanged_by_oracle") is not True
        or row.get("pre_query_public_observation") != canonical
        or row.get("pre_query_public_observation_sha256")
        != hashlib.sha256(payload).hexdigest()
        or row.get("pre_query_body_ids") != body_ids
        or int(row.get("tick", -1)) != tick
        or int(query.get("seed", -1)) != seed
        or query.get("split") != split
        or int(query.get("start_tick", -1)) != tick
    ):
        raise RuntimeError("foreign or inconsistent R3H G2 shadow-query identity")


def _load_complete(
    episode_path: Path,
    query_path: Path,
    *,
    identity: Mapping[str, object],
    identity_sha256: str,
) -> dict[str, object] | None:
    _recover_writer_temps(episode_path, allow_linked_remnant=True)
    _recover_writer_temps(query_path, allow_linked_remnant=True)
    if not episode_path.exists() and not query_path.exists():
        return None
    if episode_path.exists() != query_path.exists():
        partial = episode_path if episode_path.exists() else query_path
        if (
            not partial.is_file()
            or partial.is_symlink()
            or partial.resolve() != partial
            or partial.stat().st_nlink not in (1, 2)
        ):
            raise RuntimeError(f"indirect R3H G2 partial: {partial}")
        if partial.stat().st_nlink == 2:
            digest = _sha256_file(partial)
            _require_interrupted_quarantine_link(
                partial,
                OUTPUT_ROOT / "_incomplete" / partial.parent.name,
                destination_stem=(
                    f"{partial.name}.{identity_sha256}.{digest}"
                ),
            )
        if partial == episode_path:
            value = json.loads(partial.read_text())
            if (
                value.get("collector_identity_sha256") != identity_sha256
                or int(value.get("query_rows", -1)) < 1
            ):
                raise RuntimeError("foreign partial R3H G2 episode")
        else:
            rows = [
                json.loads(line)
                for line in partial.read_text().splitlines()
                if line.strip()
            ]
            if not rows or any(
                row.get("collector_identity_sha256") != identity_sha256
                for row in rows
            ):
                raise RuntimeError("foreign partial R3H G2 query file")
        _quarantine_partial(partial, identity_sha256=identity_sha256)
        return None
    _require_regular(episode_path)
    _require_regular(query_path)
    value = json.loads(episode_path.read_text())
    if (
        value.get("schema") != "r3h-g2-frozen-v5-shadow-label-episode-v1"
        or value.get("complete") is not True
        or value.get("development_only") is not True
        or value.get("sealed_test_allowed") is not False
        or value.get("collector_identity") != identity
        or value.get("collector_identity_sha256") != identity_sha256
        or type(value.get("logical_cpu")) is not int
        or value.get("query_file_sha256") != _sha256_file(query_path)
        or int(value.get("executed_alternative_count", -1)) != 0
    ):
        raise RuntimeError(f"foreign or incomplete R3H G2 episode: {episode_path}")
    rows = [
        json.loads(line)
        for line in query_path.read_text().splitlines()
        if line.strip()
    ]
    if not rows or len(rows) != int(value.get("query_rows", -1)):
        raise RuntimeError("R3H G2 query row count differs from episode receipt")
    for query_index, row in enumerate(rows):
        _validate_query_row(
            row,
            query_index=query_index,
            identity_sha256=identity_sha256,
            split=str(value["split"]),
            index=int(value["index"]),
            seed=int(value["seed"]),
        )
    return value


def collect_episode(
    split: str,
    index: int,
) -> dict[str, object]:
    if split not in SPLIT_COUNTS:
        raise ValueError(f"forbidden R3H G2 collection split: {split!r}")
    seed = derive_seed(split, index)
    _validate_collection_split(
        split, allow_writer_temps=True, require_complete=False
    )
    _verify_seed_schedule()
    pinned_cpu = _load_g1()._pin_one_cpu()
    if type(pinned_cpu) is not int:
        raise RuntimeError("R3H G2 collector cannot prove one logical CPU")
    _verify_external_constants()
    _verify_source_identity(refresh_runtime=True)
    authorization = _authorize_collection(split)
    core, campaign, identities = _load_frozen_b()
    _verify_source_identity(refresh_runtime=True)
    identity = _collector_identity(
        split=split,
        index=index,
        seed=seed,
        identities=identities,
        authorization=authorization,
    )
    identity_sha256 = _canonical_sha256(identity)
    episode_path, query_path = _paths(split, seed)
    existing = _load_complete(
        episode_path,
        query_path,
        identity=identity,
        identity_sha256=identity_sha256,
    )
    if existing is not None:
        return existing

    _verify_source_identity()
    _validate_loaded_identities(identities)
    policy = campaign.POLICY_FACTORY()
    if getattr(policy, "artifact_sha256", None) != FROZEN_V5_SHA256:
        raise RuntimeError("constructed live policy is not frozen-v5")
    policy.reset(seed)
    evaluator = core.ExactBranchEvaluator(campaign.POLICY_FACTORY)
    _verify_source_identity(refresh_runtime=True)
    query_rows: list[dict[str, object]] = []
    events: Counter[int] = Counter()
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    env_max_ticks = HORIZON + EXACT_HORIZON

    with campaign.IrisuEnv(
        library_path=core.RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": env_max_ticks},
    ) as env:
        _verify_source_identity(refresh_runtime=True)
        if (
            Path(env.library_path).resolve() != core.RUNTIME.resolve()
            or _sha256_file(Path(env.library_path)) != RUNTIME_SHA256
            or env_max_ticks < HORIZON + EXACT_HORIZON
        ):
            raise RuntimeError("foreign or undersized portable environment")
        runner = env.runner_identity_manifest()
        config_hash = int(runner["config_hash"])
        observation, reset_info = env.reset(seed=seed)
        if (
            int(reset_info.get("seed", -1)) != seed
            or int(reset_info.get("config_hash", -1)) != config_hash
        ):
            raise RuntimeError("portable reset seed/config identity mismatch")
        start_tick = int(observation["tick"])
        initial_clears = int(observation.get("qualifying_clear_count", 0))
        terminated = bool(observation.get("terminated", False))
        truncated = bool(observation.get("truncated", False))
        seen_shots = decisions = 0

        def step_unit(action: object) -> None:
            nonlocal observation, terminated, truncated
            before_tick = int(observation["tick"])
            observation, _reward, terminated, truncated, info = env.step(action)
            if (
                int(info.get("config_hash", -1)) != config_hash
                or int(observation["tick"]) != before_tick + 1
            ):
                raise RuntimeError("portable unit-step identity failure")
            for event in info.get("events", ()):
                if isinstance(event, Mapping):
                    kind = campaign.event_kind(event)
                    if kind is not None:
                        events[int(kind)] += 1

        while (
            not terminated
            and not truncated
            and int(observation["tick"]) - start_tick < HORIZON
        ):
            decisions += 1
            if decisions > 2_000_000:
                raise RuntimeError("episode exceeded decision budget")
            incumbent = policy.predict(observation)
            if not isinstance(incumbent, campaign.SteeringDecision):
                raise TypeError("frozen-v5 returned a foreign steering decision")
            if incumbent.is_shot:
                seen_shots += 1
                scheduled = (
                    (seen_shots - 1) % QUERY_STRIDE == 0
                    and len(query_rows) < MAXIMUM_QUERIES
                )
                if scheduled:
                    canonical_observation, observation_payload, body_ids = (
                        _canonical_observation(observation)
                    )
                    query_tick = int(canonical_observation["tick"])
                    snapshot = env.clone_state()
                    state_hash = env.state_hash()
                    query_id = (
                        f"{EXPERIMENT_ID}:{split}:{index}:"
                        f"{query_tick}:q{len(query_rows)}"
                    )
                    query = evaluator.evaluate(
                        env,
                        observation,
                        incumbent,
                        seed=seed,
                        query_id=query_id,
                        split=split,
                        live_policy=policy,
                    )
                    state_restored = (
                        env.clone_state() == snapshot
                        and env.state_hash() == state_hash
                    )
                    _after_observation, after_payload, _after_body_ids = (
                        _canonical_observation(observation)
                    )
                    observation_unchanged = (
                        after_payload == observation_payload
                    )
                    if not state_restored:
                        raise RuntimeError(
                            "exact shadow oracle did not restore live state"
                        )
                    if not observation_unchanged:
                        raise RuntimeError(
                            "exact shadow oracle mutated the public observation"
                        )
                    if (
                        query.snapshot_sha256
                        != hashlib.sha256(snapshot).hexdigest()
                        or query.incumbent.ordinal != 0
                        or query.seed != seed
                        or query.split != split
                        or query.start_tick != query_tick
                        or query.query_id != query_id
                    ):
                        raise RuntimeError(
                            "exact query source/seed/tick/incumbent identity changed"
                        )
                    query_rows.append(
                        {
                            "schema": "r3h-g2-exact-shadow-query-v1",
                            "development_only": True,
                            "sealed_test_allowed": False,
                            "collector_identity_sha256": identity_sha256,
                            "split": split,
                            "index": index,
                            "seed": seed,
                            "query_index": len(query_rows),
                            "shot_index": seen_shots,
                            "tick": query_tick,
                            "executed_ordinal": 0,
                            "alternatives_executed": False,
                            "live_state_restored_exactly": state_restored,
                            "public_observation_unchanged_by_oracle": (
                                observation_unchanged
                            ),
                            "pre_query_public_observation": canonical_observation,
                            "pre_query_public_observation_sha256": (
                                hashlib.sha256(observation_payload).hexdigest()
                            ),
                            "pre_query_body_ids": body_ids,
                            "exact_query": query.manifest(),
                        }
                    )

            for action in incumbent.primitive_actions():
                if terminated or truncated:
                    break
                remaining = HORIZON - (int(observation["tick"]) - start_tick)
                if remaining <= 0:
                    break
                kind = campaign.ActionKind.parse(action.kind)
                duration = int(action.wait_ticks) if kind is campaign.ActionKind.WAIT else 1
                for _ in range(min(duration, remaining)):
                    step_unit(
                        campaign.Action.wait(1)
                        if kind is campaign.ActionKind.WAIT
                        else action
                    )
                    if terminated or truncated:
                        break

        elapsed = int(observation["tick"]) - start_tick
        if truncated and elapsed < HORIZON:
            raise RuntimeError("portable environment truncated before manual censor")
        if not query_rows:
            raise RuntimeError("R3H G2 episode produced no exact shadow query")
        query_payload = b"".join(_canonical_bytes(row) + b"\n" for row in query_rows)
        query_sha256 = hashlib.sha256(query_payload).hexdigest()
        result: dict[str, object] = {
            "schema": "r3h-g2-frozen-v5-shadow-label-episode-v1",
            "complete": True,
            "development_only": True,
            "sealed_test_allowed": False,
            "collector_identity": identity,
            "collector_identity_sha256": identity_sha256,
            "split": split,
            "index": index,
            "seed": seed,
            "logical_cpu": pinned_cpu,
            "horizon": HORIZON,
            "runner_config_max_episode_ticks": env_max_ticks,
            "runner": runner,
            "live_policy": "frozen-v5-incumbent-only",
            "alternatives_executed": False,
            "executed_alternative_count": 0,
            "score": int(observation.get("score", 0)),
            "survival_ticks": min(elapsed, HORIZON),
            "terminal": bool(terminated),
            "final_gauge": int(observation.get("gauge", 0)),
            "final_level": int(observation.get("level", 0)),
            "qualifying_clears": (
                int(observation.get("qualifying_clear_count", 0)) - initial_clears
            ),
            "decisions": decisions,
            "seen_shots": seen_shots,
            "event_counts": {
                str(kind): count for kind, count in sorted(events.items())
            },
            "final_state_sha256": hashlib.sha256(env.clone_state()).hexdigest(),
            "query_file": str(query_path),
            "query_file_sha256": query_sha256,
            "query_rows": len(query_rows),
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
        }

    _verify_source_identity()
    _write_new(query_path, query_payload)
    _write_new(
        episode_path,
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode()
        + b"\n",
    )
    _validate_collection_split(
        split, allow_writer_temps=True, require_complete=False
    )
    _validate_collection_incomplete()
    _verify_source_identity()
    return result


def collect(split: str, start: int, count: int) -> dict[str, object]:
    _validate_request(split, start, count)
    _verify_seed_schedule()
    _validate_collection_split(
        split, allow_writer_temps=True, require_complete=False
    )
    pinned_cpu = _load_g1()._pin_one_cpu()
    authorization = _authorize_collection(split)
    _verify_external_constants()
    episodes = [
        collect_episode(split, index)
        for index in range(start, start + count)
    ]
    _validate_collection_split(
        split, allow_writer_temps=True, require_complete=False
    )
    return {
        "schema": "r3h-g2-exact-collection-batch-v1",
        "development_only": True,
        "sealed_test_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "split": split,
        "start": start,
        "count": count,
        "pinned_cpu": pinned_cpu,
        "seeds": [int(value["seed"]) for value in episodes],
        "query_rows": sum(int(value["query_rows"]) for value in episodes),
        "authorization": authorization,
        "complete": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect fresh R3H G2 full-board shadow labels."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser(
        "collect", help="collect an immutable preregistered split range"
    )
    collect_parser.add_argument("--split", required=True, choices=tuple(SPLIT_COUNTS))
    collect_parser.add_argument("--start", required=True, type=int)
    collect_parser.add_argument("--count", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "collect":
        raise RuntimeError("unsupported command")
    print(
        json.dumps(
            collect(args.split, args.start, args.count),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
