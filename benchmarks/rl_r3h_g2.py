#!/usr/bin/env python3
"""Operate the development-only R3H Generation-02 board learner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import struct
import sys
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
PORTABLE_LIBRARY = (
    ROOT
    / "artifacts/r3/runtime/main-0c48dba-20260723/portable-build"
    / "libirisu_clone.so"
).resolve()
PYCACHE_PREFIX = (
    ROOT
    / "artifacts/r3/development/r3h-resolution-first-g2r1-20260730"
    / "_runtime-pycache-disabled"
)
_BOOTSTRAP_AUDIT_ENV = "IRISU_G2_UNFROZEN_AUDIT"
_BOOTSTRAP_EXPERIMENT = (
    ROOT / "artifacts/r3/development/r3h-resolution-first-g2r1-20260730"
)
_BOOTSTRAP_SOURCE_PATHS = {
    "campaign": Path(__file__).resolve(),
    "module": (
        ROOT / "python/irisu_pointer/resolution_first_g2.py"
    ).resolve(),
    "collector": (ROOT / "benchmarks/r3h_g2_exact_collect.py").resolve(),
    "protocol": (_BOOTSTRAP_EXPERIMENT / "protocol.md").resolve(),
    "focused_tests": (
        ROOT / "tests/test_pointer_resolution_first_g2.py"
    ).resolve(),
    "base_tests": (
        ROOT / "tests/test_pointer_resolution_first.py"
    ).resolve(),
    "resolution_first": (
        ROOT / "python/irisu_pointer/resolution_first.py"
    ).resolve(),
    "encoding": (ROOT / "python/irisu_rl/encoding.py").resolve(),
    "schema": (ROOT / "python/irisu_rl/schema.py").resolve(),
    "pointer_package_init": (
        ROOT / "python/irisu_pointer/__init__.py"
    ).resolve(),
    "rl_package_init": (ROOT / "python/irisu_rl/__init__.py").resolve(),
    "g1_collector": (ROOT / "benchmarks/r3h_exact_collect.py").resolve(),
    "pyproject": (ROOT / "pyproject.toml").resolve(),
    "uv_lock": (ROOT / "uv.lock").resolve(),
    "campaign_tests": (ROOT / "tests/test_r3h_g2_campaign.py").resolve(),
    "env_package_init": (
        ROOT / "python/irisu_env/__init__.py"
    ).resolve(),
}
for _dependency in sorted((ROOT / "python/irisu_env").glob("*.py")):
    if _dependency.name != "__init__.py":
        _BOOTSTRAP_SOURCE_PATHS[f"env_{_dependency.stem}"] = (
            _dependency.resolve()
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


def _bootstrap_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_regular(path: Path) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.resolve() != path
    ):
        raise RuntimeError(f"indirect R3H G2 bootstrap file: {path}")


def _bootstrap_tree(
    root: Path, *, excluded_top_level: Sequence[str] = ()
) -> dict[str, object]:
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"indirect R3H G2 bootstrap tree: {root}")
    paths: list[Path] = []
    for directory, names, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        names[:] = [
            name
            for name in names
            if name != "__pycache__"
            and not (parent == root and name in excluded_top_level)
        ]
        for name in (*names, *filenames):
            path = parent / name
            if path.is_symlink():
                raise RuntimeError(f"indirect R3H G2 bootstrap entry: {path}")
        paths.extend(
            parent / name
            for name in filenames
        )
        direct_bytecode = [
            parent / name
            for name in filenames
            if name.endswith((".pyc", ".pyo"))
        ]
        if direct_bytecode:
            raise RuntimeError(
                f"direct R3H G2 bytecode is forbidden: {direct_bytecode[0]}"
            )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        _bootstrap_regular(path)
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        row = json.dumps(
            {
                "path": relative,
                "size": size,
                "sha256": _bootstrap_sha256(path),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        digest.update(row + b"\n")
        total_bytes += size
    return {
        "root": str(root),
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "manifest_sha256": digest.hexdigest(),
    }


def _bootstrap_machine_identity() -> dict[str, object]:
    fields: dict[str, str] = {}
    allowed = {
        "vendor_id",
        "cpu family",
        "model",
        "stepping",
        "microcode",
        "model name",
        "flags",
    }
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if not line.strip() and fields:
            break
        if ":" not in line:
            continue
        name, value = (part.strip() for part in line.split(":", 1))
        if name in allowed:
            fields[name] = value
    return {"uname": list(os.uname()), "cpu": fields}


def _bootstrap_frozen_environment() -> None:
    preregistration_path = _BOOTSTRAP_EXPERIMENT / "preregistration.json"
    _bootstrap_regular(preregistration_path)
    value = json.loads(preregistration_path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("R3H G2 bootstrap preregistration is malformed")
    if value.get("status") != "source_frozen_preflight_authorized":
        entries = {
            entry.name for entry in _BOOTSTRAP_EXPERIMENT.iterdir()
        }
        if (
            os.environ.get(_BOOTSTRAP_AUDIT_ENV) == "1"
            and value.get("status")
            == "protocol_preregistered_source_not_yet_frozen"
            and entries == {"protocol.md", "preregistration.json"}
        ):
            return
        raise RuntimeError("R3H G2 source is not frozen before local imports")
    frozen = value.get("frozen_source_sha256")
    runtime = value.get("runtime_manifest")
    if (
        value.get("experiment_id") != "r3h-resolution-first-g2r1-20260730"
        or value.get("source_revision")
        != "de701b36355d5ec582df30f4223aabde7bc537df"
        or value.get("development_only") is not True
        or value.get("sealed_test_allowed") is not False
        or value.get("outcomes_viewed_before_preregistration") is not False
        or value.get("generation_01_outcomes_are_training_data") is not False
        or value.get("generation_02_outcomes_collected_before_freeze") is not False
        or not isinstance(frozen, dict)
        or set(frozen) != set(_BOOTSTRAP_SOURCE_PATHS)
        or not isinstance(runtime, dict)
        or runtime.get("schema") != "irisu-r3h-g2-numerical-runtime-v4"
    ):
        raise RuntimeError("R3H G2 bootstrap preregistration is malformed")
    for name, path in _BOOTSTRAP_SOURCE_PATHS.items():
        expected = frozen.get(name)
        _bootstrap_regular(path)
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or _bootstrap_sha256(path) != expected
        ):
            raise RuntimeError(f"R3H G2 source changed before import: {name}")
    if (
        value.get("protocol_sha256") != frozen["protocol"]
        or runtime.get("python_version") != sys.version
        or runtime.get("python_implementation") != sys.implementation.name
        or runtime.get("python_cache_tag") != sys.implementation.cache_tag
        or runtime.get("python_dont_write_bytecode_flag")
        != sys.flags.dont_write_bytecode
        or runtime.get("python_isolated_flag") != sys.flags.isolated
        or runtime.get("python_no_site_flag") != sys.flags.no_site
        or runtime.get("python_ignore_environment_flag")
        != sys.flags.ignore_environment
        or runtime.get("python_safe_path_flag") is not sys.flags.safe_path
        or runtime.get("python_pycache_prefix") != sys.pycache_prefix
        or runtime.get("thread_environment")
        != {
            name: os.environ.get(name)
            for name in (
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONPYCACHEPREFIX",
                "PYTHONHASHSEED",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        }
        or runtime.get("machine") != _bootstrap_machine_identity()
    ):
        raise RuntimeError("R3H G2 bootstrap runtime changed before import")
    runtime_files = runtime.get("files")
    if not isinstance(runtime_files, dict):
        raise RuntimeError("R3H G2 bootstrap runtime file closure is absent")
    for row in runtime_files.values():
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise RuntimeError("R3H G2 bootstrap runtime file is malformed")
        path = Path(row["path"])
        _bootstrap_regular(path)
        if row.get("sha256") != _bootstrap_sha256(path):
            raise RuntimeError("R3H G2 runtime file changed before import")
    trees = runtime.get("distribution_closure")
    if not isinstance(trees, dict) or set(trees) != {
        "python_stdlib",
        "ld_configuration",
        "repository_python",
        "site_packages",
        "numpy",
        "numpy_libs",
        "torch",
    }:
        raise RuntimeError("R3H G2 bootstrap runtime trees are malformed")
    for name, row in trees.items():
        if not isinstance(row, dict) or not isinstance(row.get("root"), str):
            raise RuntimeError("R3H G2 bootstrap runtime tree is malformed")
        actual = _bootstrap_tree(
            Path(row["root"]),
            excluded_top_level=(
                ("site-packages", "dist-packages")
                if name == "python_stdlib"
                else ()
            ),
        )
        if actual != row:
            raise RuntimeError(f"R3H G2 runtime tree changed before import: {name}")
    dynamic = runtime.get("dynamic_closure")
    targets = dynamic.get("targets") if isinstance(dynamic, dict) else None
    dependencies = (
        dynamic.get("dependencies") if isinstance(dynamic, dict) else None
    )
    ldd = dynamic.get("ldd") if isinstance(dynamic, dict) else None
    if (
        not isinstance(targets, dict)
        or not isinstance(dependencies, dict)
        or not isinstance(ldd, dict)
    ):
        raise RuntimeError("R3H G2 dynamic runtime closure is malformed")
    for name, row in {**targets, **dependencies, "ldd": ldd}.items():
        if not isinstance(row, dict):
            raise RuntimeError("R3H G2 dynamic runtime row is malformed")
        path_value = row.get("path") if name == "ldd" else name
        if not isinstance(path_value, str):
            raise RuntimeError("R3H G2 dynamic runtime row is malformed")
        path = Path(path_value)
        _bootstrap_regular(path)
        if (
            row.get("sha256") != _bootstrap_sha256(path)
            or (name != "ldd" and row.get("size") != path.stat().st_size)
        ):
            raise RuntimeError("R3H G2 dynamic runtime changed before import")


_validate_pycache_boundary()
_bootstrap_frozen_environment()
SITE_PACKAGES = (
    ROOT / ".venv/lib/python3.14/site-packages"
).resolve()
if (
    not SITE_PACKAGES.is_dir()
    or SITE_PACKAGES.is_symlink()
    or SITE_PACKAGES.resolve() != SITE_PACKAGES
):
    raise RuntimeError("R3H G2 verified site-packages root is indirect")
if str(SITE_PACKAGES) not in sys.path:
    sys.path.insert(0, str(SITE_PACKAGES))
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from irisu_pointer.resolution_first_g2 import (  # noqa: E402
    BoardResolutionDataset,
    ResolutionFirstG2Config,
    board_branch_records,
    fit_selective_calibration_g2,
    fit_support_calibration_g2,
    load_checkpoint_g2,
    predict_records_g2,
    resolution_auroc_g2,
    save_checkpoint_g2,
    select_candidates_g2,
    train_resolution_first_g2,
    viability_report_g2,
)

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    if torch.get_num_interop_threads() != 1:
        raise


EXPERIMENT_ID = "r3h-resolution-first-g2r1-20260730"
EXPERIMENT = ROOT / "artifacts/r3/development" / EXPERIMENT_ID
PROTOCOL = EXPERIMENT / "protocol.md"
COLLECTOR = ROOT / "benchmarks/r3h_g2_exact_collect.py"
COLLECTION = EXPERIMENT / "collection"
MODEL = EXPERIMENT / "model"
SOURCE_REVISION = "de701b36355d5ec582df30f4223aabde7bc537df"
SOURCE_IDENTITY = EXPERIMENT / "source-identity.json"
PREFLIGHT = EXPERIMENT / "preflight.json"
PREFLIGHT_RECEIPT = EXPERIMENT / "preflight.receipt.json"
PILOT = EXPERIMENT / "pilot.json"
PILOT_RECEIPT = EXPERIMENT / "pilot.receipt.json"
TRAINING = MODEL / "training.json"
CHECKPOINT = MODEL / "resolution-first-g2.pt"
CHECKPOINT_RECEIPT = MODEL / "checkpoint.receipt.json"
SUPPORT_CALIBRATION = MODEL / "support-calibration.json"
SUPPORT_CALIBRATION_RECEIPT = MODEL / "support-calibration.receipt.json"
MARGIN_CALIBRATION = MODEL / "margin-calibration.json"
MARGIN_CALIBRATION_RECEIPT = MODEL / "margin-calibration.receipt.json"
SPLITS = {
    "train-board": 32,
    "support-board": 24,
    "margin-board": 32,
    "offline-board": 16,
}
PILOT_LIMITS = {
    "resolved_alternative_seeds": 24,
    "viable_alternative_seeds": 18,
    "resolved_nonincumbent_rows": 160,
}
PREREGISTRATION_STATUS = "source_frozen_preflight_authorized"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _jsonable(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _artifact_bytes(value: object) -> bytes:
    return (
        json.dumps(
            _jsonable(value), sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode()


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


def _atomic_temp_candidates(path: Path) -> tuple[Path, ...]:
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
    expected = f"{prefix}tmp-"
    output: list[Path] = []
    for candidate in path.parent.iterdir():
        if not candidate.name.startswith(prefix):
            continue
        suffix = (
            candidate.name[len(expected) :]
            if candidate.name.startswith(expected)
            else ""
        )
        pieces = suffix.split("-")
        if len(pieces) != 2 or not all(piece.isdigit() for piece in pieces):
            raise RuntimeError(f"unknown R3H G2 atomic remnant: {candidate}")
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
        or sha256_file(path) != pieces[1]
    ):
        raise RuntimeError(f"unknown R3H G2 quarantine entry: {path}")
    return tuple(pieces)  # type: ignore[return-value]


def _validate_campaign_incomplete() -> tuple[tuple[Path, Path], ...]:
    root = EXPERIMENT / "_incomplete"
    if not root.exists():
        if root.is_symlink():
            raise RuntimeError(f"indirect R3H G2 incomplete root: {root}")
        return ()
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise RuntimeError(f"indirect R3H G2 incomplete root: {root}")
    entries = tuple(root.iterdir())
    if any(entry.name != "_atomic" for entry in entries):
        raise RuntimeError("unknown R3H G2 incomplete namespace")
    if not entries:
        return ()
    atomic = entries[0]
    if not atomic.is_dir() or atomic.is_symlink() or atomic.resolve() != atomic:
        raise RuntimeError("indirect R3H G2 atomic quarantine")
    sources = {
        EXPERIMENT.name: (
            EXPERIMENT,
            {
                "source-identity.json",
                "preflight.json",
                "preflight.receipt.json",
                "pilot.json",
                "pilot.receipt.json",
                "offline-screen.json",
                "offline-screen.receipt.json",
            },
        ),
        "seed-manifests": (
            EXPERIMENT / "seed-manifests",
            {f"{split}.json" for split in SPLITS},
        ),
        "model": (
            MODEL,
            {
                "resolution-first-g2.pt",
                "training.json",
                "checkpoint.receipt.json",
                "support-calibration.json",
                "support-calibration.receipt.json",
                "margin-calibration.json",
                "margin-calibration.receipt.json",
            },
        ),
    }
    output: list[tuple[Path, Path]] = []
    for directory in atomic.iterdir():
        source = sources.get(directory.name)
        if (
            source is None
            or not directory.is_dir()
            or directory.is_symlink()
            or directory.resolve() != directory
        ):
            raise RuntimeError(f"unknown R3H G2 quarantine namespace: {directory}")
        source_parent, allowed_targets = source
        if (
            not source_parent.is_dir()
            or source_parent.is_symlink()
            or source_parent.resolve() != source_parent
        ):
            raise RuntimeError(
                f"indirect R3H G2 quarantine source: {source_parent}"
            )
        for candidate in directory.iterdir():
            target, _digest, _pid, _stamp, _sequence = (
                _validate_quarantine_entry(candidate)
            )
            if target not in allowed_targets:
                raise RuntimeError(
                    f"unknown R3H G2 quarantine target: {candidate}"
                )
            output.append((source_parent / target, candidate))
            links = candidate.stat().st_nlink
            if links == 1:
                continue
            if links != 2:
                raise RuntimeError(f"foreign R3H G2 quarantine links: {candidate}")
            matches = [
                source
                for source in source_parent.iterdir()
                if (
                    source.is_file()
                    and not source.is_symlink()
                    and os.path.samefile(source, candidate)
                )
            ]
            pattern = f".{target}.tmp-"
            if (
                len(matches) != 1
                or not matches[0].name.startswith(pattern)
                or len(matches[0].name[len(pattern) :].split("-")) != 2
                or not all(
                    piece.isdigit()
                    for piece in matches[0].name[len(pattern) :].split("-")
                )
            ):
                raise RuntimeError(f"foreign R3H G2 quarantine links: {candidate}")
    return tuple(sorted(output))


def _validate_all_incomplete(*, collector_sha256: str) -> None:
    _validate_pycache_boundary()
    _validate_campaign_incomplete()
    expected = COLLECTOR.resolve()
    _require_regular(expected)
    if (
        len(collector_sha256) != 64
        or any(character not in "0123456789abcdef" for character in collector_sha256)
        or sha256_file(expected) != collector_sha256
    ):
        raise RuntimeError("R3H G2 collector changed before validator load")
    spec = importlib.util.spec_from_file_location(
        "_irisu_r3h_g2_collector_validation", expected
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load exact R3H G2 collector validator")
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    if Path(str(collector.__file__)).resolve() != expected:
        raise RuntimeError("R3H G2 collector validator path changed")
    if collector.OUTPUT_ROOT.resolve() != COLLECTION.resolve():
        raise RuntimeError("R3H G2 collector incomplete root changed")
    collector._validate_collection_incomplete()


def _quarantine_atomic_temp(path: Path, target: Path) -> Path:
    digest = sha256_file(path)
    return _quarantine_link(
        path,
        EXPERIMENT
        / "_incomplete"
        / "_atomic"
        / target.parent.name,
        destination_stem=f"{target.name}.{digest}",
    )


def _recover_write_once_temps(path: Path, encoded: bytes) -> None:
    linked = path.exists() or path.is_symlink()
    if linked and (path.is_symlink() or not path.is_file()):
        raise RuntimeError(f"indirect R3H G2 artifact: {path}")
    for temporary in _atomic_temp_candidates(path):
        if temporary.is_symlink() or not temporary.is_file():
            raise RuntimeError(f"indirect R3H G2 atomic remnant: {temporary}")
        if linked and os.path.samefile(temporary, path):
            temporary.unlink()
            _fsync_directory(path.parent)
            continue
        if not linked and temporary.read_bytes() == encoded:
            if temporary.stat().st_nlink != 1:
                _quarantine_atomic_temp(temporary, path)
                continue
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, path)
            _fsync_directory(path.parent)
            temporary.unlink()
            _fsync_directory(path.parent)
            linked = True
        else:
            _quarantine_atomic_temp(temporary, path)
    if linked and path.stat().st_nlink != 1:
        raise RuntimeError(f"R3H G2 artifact has foreign hardlinks: {path}")


def write_once(path: Path, value: object) -> str:
    encoded = _artifact_bytes(value)
    _mkdir_durable(path.parent)
    _recover_write_once_temps(path, encoded)
    if path.exists() or path.is_symlink():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to rewrite {path}")
        return sha256_file(path)
    temporary = path.with_name(
        f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}"
    )
    linked = False
    try:
        with temporary.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        linked = True
        _fsync_directory(path.parent)
    finally:
        if linked:
            temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    return sha256_file(path)


def _revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _one_cpu() -> int:
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError("R3H G2 must be pinned to one logical CPU")
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("R3H G2 Torch thread limits changed")
    return affinity[0]


def _runtime_tree_identity(
    root: Path, *, excluded_top_level: Sequence[str] = ()
) -> dict[str, object]:
    root = root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError(f"indirect R3H G2 runtime tree: {root}")
    paths: list[Path] = []
    for directory, names, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        names[:] = [
            name
            for name in names
            if name != "__pycache__"
            and not (parent == root and name in excluded_top_level)
        ]
        for name in (*names, *filenames):
            path = parent / name
            if path.is_symlink():
                raise RuntimeError(f"indirect R3H G2 runtime entry: {path}")
        paths.extend(
            parent / name
            for name in filenames
        )
        direct_bytecode = [
            parent / name
            for name in filenames
            if name.endswith((".pyc", ".pyo"))
        ]
        if direct_bytecode:
            raise RuntimeError(
                f"direct R3H G2 bytecode is forbidden: {direct_bytecode[0]}"
            )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.resolve() != path
        ):
            raise RuntimeError(f"indirect R3H G2 runtime file: {path}")
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_sha256 = sha256_file(path)
        digest.update(
            canonical_bytes(
                {
                    "path": relative,
                    "size": size,
                    "sha256": file_sha256,
                }
            )
            + b"\n"
        )
        total_bytes += size
    return {
        "root": str(root),
        "file_count": len(paths),
        "total_bytes": total_bytes,
        "manifest_sha256": digest.hexdigest(),
    }


def _elf_files(roots: Sequence[Path]) -> tuple[Path, ...]:
    output: set[Path] = set()
    for root in roots:
        candidates = root.rglob("*") if root.is_dir() else (root,)
        for path in candidates:
            if path.is_symlink() or not path.is_file():
                continue
            with path.open("rb") as stream:
                if stream.read(4) == b"\x7fELF":
                    output.add(path.resolve())
    return tuple(sorted(output))


def _elf_has_dynamic_segment(path: Path) -> bool:
    with path.open("rb") as stream:
        header = stream.read(64)
        if len(header) < 52 or header[:4] != b"\x7fELF":
            raise RuntimeError(f"malformed R3H G2 ELF runtime: {path}")
        elf_class = header[4]
        byte_order = header[5]
        endian = "<" if byte_order == 1 else ">" if byte_order == 2 else ""
        if not endian:
            raise RuntimeError(f"unknown R3H G2 ELF byte order: {path}")
        if elf_class == 1:
            program_offset = struct.unpack_from(f"{endian}I", header, 28)[0]
            entry_size = struct.unpack_from(f"{endian}H", header, 42)[0]
            entry_count = struct.unpack_from(f"{endian}H", header, 44)[0]
        elif elf_class == 2:
            if len(header) < 58:
                raise RuntimeError(f"malformed R3H G2 ELF runtime: {path}")
            program_offset = struct.unpack_from(f"{endian}Q", header, 32)[0]
            entry_size = struct.unpack_from(f"{endian}H", header, 54)[0]
            entry_count = struct.unpack_from(f"{endian}H", header, 56)[0]
        else:
            raise RuntimeError(f"unknown R3H G2 ELF class: {path}")
        if entry_size < 4 or entry_count < 1:
            raise RuntimeError(f"malformed R3H G2 ELF program table: {path}")
        for index in range(entry_count):
            stream.seek(program_offset + index * entry_size)
            entry = stream.read(entry_size)
            if len(entry) != entry_size:
                raise RuntimeError(f"truncated R3H G2 ELF program table: {path}")
            if struct.unpack_from(f"{endian}I", entry)[0] == 2:
                return True
    return False


def _loaded_elf_files() -> tuple[Path, ...]:
    maps = Path("/proc/self/maps")
    if not maps.is_file() or maps.is_symlink():
        raise RuntimeError("R3H G2 process-map runtime is unavailable")
    output: set[Path] = set()
    for raw_line in maps.read_text().splitlines():
        fields = raw_line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith("/"):
            continue
        raw_path = fields[5]
        if raw_path.endswith(" (deleted)"):
            raise RuntimeError("R3H G2 loaded runtime was deleted")
        path = Path(raw_path)
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"indirect R3H G2 loaded runtime: {path}")
        with path.open("rb") as stream:
            if stream.read(4) == b"\x7fELF":
                output.add(path.resolve())
    return tuple(sorted(output))


def _dynamic_runtime_identity(roots: Sequence[Path]) -> dict[str, object]:
    ldd = Path("/usr/bin/ldd").resolve()
    if (
        not ldd.is_file()
        or ldd.is_symlink()
        or ldd.stat().st_nlink != 1
        or ldd.resolve() != ldd
    ):
        raise RuntimeError("R3H G2 ldd runtime is indirect")
    dependencies: set[Path] = set()
    loaded_targets = _loaded_elf_files()
    elf_targets = _elf_files(roots)
    targets = tuple(
        target for target in elf_targets if _elf_has_dynamic_segment(target)
    )
    static_targets = tuple(
        target for target in elf_targets if target not in targets
    )
    basename_index: dict[str, list[Path]] = defaultdict(list)
    for candidate in elf_targets:
        basename_index[candidate.name].append(candidate)
    bundled_resolutions: dict[str, dict[str, str]] = {}
    unavailable_targets: dict[str, list[str]] = {}
    stdlib_dynamic_root = (
        Path(os.__file__).resolve().parent / "lib-dynload"
    ).resolve()
    for target in targets:
        result = subprocess.run(
            [str(ldd), str(target)],
            check=False,
            capture_output=True,
            text=True,
        )
        output = f"{result.stdout}\n{result.stderr}"
        if result.returncode != 0:
            raise RuntimeError(f"cannot close R3H G2 dynamic runtime: {target}")
        target_resolutions: dict[str, str] = {}
        target_unavailable: list[str] = []
        for raw_line in result.stdout.splitlines():
            line = raw_line.strip()
            if "=>" in line and line.split("=>", 1)[1].strip() == "not found":
                soname = line.split("=>", 1)[0].strip()
                candidates = basename_index.get(soname, [])
                if not soname or Path(soname).name != soname:
                    raise RuntimeError(
                        f"malformed R3H G2 missing runtime soname: "
                        f"{target}: {soname}"
                    )
                if len(candidates) == 1:
                    resolved = candidates[0]
                    target_resolutions[soname] = str(resolved)
                    dependencies.add(resolved)
                elif (
                    len(candidates) == 0
                    and target.is_relative_to(stdlib_dynamic_root)
                    and target not in loaded_targets
                ):
                    target_unavailable.append(soname)
                else:
                    raise RuntimeError(
                        f"cannot uniquely close R3H G2 bundled runtime: "
                        f"{target}: {soname}"
                    )
                continue
            candidate = ""
            if "=>" in line:
                candidate = line.split("=>", 1)[1].split("(", 1)[0].strip()
            elif line.startswith("/"):
                candidate = line.split("(", 1)[0].strip()
            if candidate.startswith("/"):
                dependencies.add(Path(candidate).resolve())
        if (
            "not found" in output
            and not target_resolutions
            and not target_unavailable
        ):
            raise RuntimeError(f"cannot close R3H G2 dynamic runtime: {target}")
        if target_resolutions:
            bundled_resolutions[str(target)] = target_resolutions
        if target_unavailable:
            unavailable_targets[str(target)] = sorted(target_unavailable)
    files: dict[str, dict[str, object]] = {}
    for path in sorted(dependencies):
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.resolve() != path
        ):
            raise RuntimeError(f"indirect R3H G2 dynamic dependency: {path}")
        files[str(path)] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    target_files = {
        str(path): {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in targets
    }
    loaded_unavailable = sorted(
        set(loaded_targets)
        & {Path(path) for path in unavailable_targets}
    )
    if loaded_unavailable:
        raise RuntimeError(
            f"unavailable R3H G2 runtime was mapped: "
            f"{loaded_unavailable[0]}"
        )
    allowed = set(targets) | set(dependencies)
    escaped = sorted(set(loaded_targets) - allowed)
    if escaped:
        raise RuntimeError(f"unclosed loaded R3H G2 runtime: {escaped[0]}")
    return {
        "ldd": {"path": str(ldd), "sha256": sha256_file(ldd)},
        "elf_target_count": len(elf_targets),
        "static_elf_target_count": len(static_targets),
        "static_elf_target_set_sha256": canonical_sha256(
            [str(path) for path in static_targets]
        ),
        "target_count": len(targets),
        "target_set_sha256": canonical_sha256([str(path) for path in targets]),
        "targets": target_files,
        "bundled_missing_resolutions": bundled_resolutions,
        "unavailable_stdlib_targets": unavailable_targets,
        "dependencies": files,
    }


def _cpu_identity() -> dict[str, object]:
    fields: dict[str, str] = {}
    allowed = {
        "vendor_id",
        "cpu family",
        "model",
        "stepping",
        "microcode",
        "model name",
        "flags",
    }
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if not line.strip() and fields:
            break
        if ":" not in line:
            continue
        name, value = (part.strip() for part in line.split(":", 1))
        if name in allowed:
            fields[name] = value
    return {
        "uname": list(os.uname()),
        "cpu": fields,
    }


def _build_runtime_identity() -> dict[str, object]:
    _validate_pycache_boundary()
    numpy_root = Path(str(np.__file__)).resolve().parent
    torch_root = Path(str(torch.__file__)).resolve().parent
    stdlib_root = Path(os.__file__).resolve().parent
    numpy_libs = numpy_root.parent / "numpy.libs"
    files = {
        "python": Path(sys.executable).resolve(),
        "portable_library": PORTABLE_LIBRARY,
        "ld_so_cache": Path("/etc/ld.so.cache").resolve(),
        "ld_so_conf": Path("/etc/ld.so.conf").resolve(),
        "numpy_init": Path(str(np.__file__)).resolve(),
        "numpy_core": Path(str(np._core._multiarray_umath.__file__)).resolve(),
        "torch_init": Path(str(torch.__file__)).resolve(),
        "torch_c": Path(str(torch._C.__file__)).resolve(),
    }
    for name, path in files.items():
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.resolve() != path
        ):
            raise RuntimeError(f"R3H G2 {name} runtime is indirect")
    return {
        "schema": "irisu-r3h-g2-numerical-runtime-v4",
        "python_version": sys.version,
        "python_implementation": sys.implementation.name,
        "python_cache_tag": sys.implementation.cache_tag,
        "python_dont_write_bytecode_flag": sys.flags.dont_write_bytecode,
        "python_isolated_flag": sys.flags.isolated,
        "python_no_site_flag": sys.flags.no_site,
        "python_ignore_environment_flag": sys.flags.ignore_environment,
        "python_safe_path_flag": sys.flags.safe_path,
        "python_pycache_prefix": sys.pycache_prefix,
        "numpy_version": np.__version__,
        "numpy_config_sha256": canonical_sha256(
            np.__config__.show(mode="dicts")
        ),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_cuda_version": torch.version.cuda,
        "torch_debug": torch.version.debug,
        "torch_config_sha256": hashlib.sha256(
            torch.__config__.show().encode()
        ).hexdigest(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "PYTHONDONTWRITEBYTECODE",
                "PYTHONPYCACHEPREFIX",
                "PYTHONHASHSEED",
                "LD_LIBRARY_PATH",
                "LD_PRELOAD",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        },
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
        "distribution_closure": {
            "python_stdlib": _runtime_tree_identity(
                stdlib_root,
                excluded_top_level=("site-packages", "dist-packages"),
            ),
            "ld_configuration": _runtime_tree_identity(
                Path("/etc/ld.so.conf.d")
            ),
            "repository_python": _runtime_tree_identity(PYTHON),
            "site_packages": _runtime_tree_identity(SITE_PACKAGES),
            "numpy": _runtime_tree_identity(numpy_root),
            "numpy_libs": _runtime_tree_identity(numpy_libs),
            "torch": _runtime_tree_identity(torch_root),
        },
        "dynamic_closure": _dynamic_runtime_identity(
            (
                files["python"],
                files["portable_library"],
                stdlib_root / "lib-dynload",
                SITE_PACKAGES,
                PYTHON,
            )
        ),
        "machine": _cpu_identity(),
    }


_RUNTIME_IDENTITY_BYTES: bytes | None = None


def runtime_identity(*, refresh: bool = False) -> dict[str, object]:
    global _RUNTIME_IDENTITY_BYTES
    if refresh or _RUNTIME_IDENTITY_BYTES is None:
        _RUNTIME_IDENTITY_BYTES = canonical_bytes(_build_runtime_identity())
    value = json.loads(_RUNTIME_IDENTITY_BYTES)
    if not isinstance(value, dict):
        raise RuntimeError("R3H G2 runtime identity is malformed")
    return value


def validate_loaded_runtime() -> int:
    dynamic = runtime_identity().get("dynamic_closure")
    if not isinstance(dynamic, Mapping):
        raise RuntimeError("R3H G2 dynamic runtime closure is absent")
    targets = dynamic.get("targets")
    dependencies = dynamic.get("dependencies")
    if not isinstance(targets, Mapping) or not isinstance(
        dependencies, Mapping
    ):
        raise RuntimeError("R3H G2 dynamic runtime paths are absent")
    allowed = {**targets, **dependencies}
    loaded = _loaded_elf_files()
    for path in loaded:
        row = allowed.get(str(path))
        if (
            not isinstance(row, Mapping)
            or row.get("size") != path.stat().st_size
            or row.get("sha256") != sha256_file(path)
        ):
            raise RuntimeError(f"unclosed loaded R3H G2 runtime: {path}")
    return len(loaded)


def _source_files() -> dict[str, Path]:
    files: dict[str, Path] = {
        "campaign": Path(__file__).resolve(),
        "module": (
            ROOT / "python/irisu_pointer/resolution_first_g2.py"
        ).resolve(),
        "collector": COLLECTOR.resolve(),
        "protocol": PROTOCOL.resolve(),
        "focused_tests": (
            ROOT / "tests/test_pointer_resolution_first_g2.py"
        ).resolve(),
        "base_tests": (
            ROOT / "tests/test_pointer_resolution_first.py"
        ).resolve(),
        "preregistration": (EXPERIMENT / "preregistration.json").resolve(),
        "resolution_first": (
            ROOT / "python/irisu_pointer/resolution_first.py"
        ).resolve(),
        "encoding": (ROOT / "python/irisu_rl/encoding.py").resolve(),
        "schema": (ROOT / "python/irisu_rl/schema.py").resolve(),
        "pointer_package_init": (
            ROOT / "python/irisu_pointer/__init__.py"
        ).resolve(),
        "rl_package_init": (ROOT / "python/irisu_rl/__init__.py").resolve(),
        "g1_collector": (ROOT / "benchmarks/r3h_exact_collect.py").resolve(),
        "pyproject": (ROOT / "pyproject.toml").resolve(),
        "uv_lock": (ROOT / "uv.lock").resolve(),
        "env_package_init": (
            ROOT / "python/irisu_env/__init__.py"
        ).resolve(),
    }
    for dependency in sorted((ROOT / "python/irisu_env").glob("*.py")):
        if dependency.name != "__init__.py":
            files[f"env_{dependency.stem}"] = dependency.resolve()
    learner_source = files["module"].read_text()
    if "from .policy import" in learner_source:
        files.update(
            {
                "policy": (ROOT / "python/irisu_pointer/policy.py").resolve(),
                "pointer_action": (
                    ROOT / "python/irisu_pointer/action.py"
                ).resolve(),
                "pointer_experts": (
                    ROOT / "python/irisu_pointer/experts.py"
                ).resolve(),
                "rl_actions": (ROOT / "python/irisu_rl/actions.py").resolve(),
            }
        )
    campaign_tests = ROOT / "tests/test_r3h_g2_campaign.py"
    if campaign_tests.exists():
        files["campaign_tests"] = campaign_tests.resolve()
    for name, path in files.items():
        if (
            not path.is_file()
            or path.is_symlink()
            or path.stat().st_nlink != 1
            or path.resolve() != path
        ):
            raise RuntimeError(f"R3H G2 {name} source is not a regular file")
    return files


def _preregistered_source_hashes() -> dict[str, str]:
    return {
        name: sha256_file(path)
        for name, path in _source_files().items()
        if name != "preregistration"
    }


def source_identity() -> dict[str, object]:
    files = _source_files()
    return {
        "schema": "irisu-r3h-g2-source-identity-v5",
        "experiment_id": EXPERIMENT_ID,
        "source_revision": SOURCE_REVISION,
        "runtime": runtime_identity(),
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
    }


def verify_source_identity() -> str:
    _one_cpu()
    if _revision() != SOURCE_REVISION:
        raise RuntimeError("R3H G2 repository revision changed")
    _validate_preregistration()
    _validate_all_incomplete(
        collector_sha256=_preregistered_source_hashes()["collector"]
    )
    path = SOURCE_IDENTITY
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("R3H G2 source identity is absent")
    _require_regular(path)
    recorded = json.loads(path.read_text())
    if (
        recorded.get("schema") != "irisu-r3h-g2-source-identity-v5"
        or recorded != source_identity()
    ):
        raise RuntimeError("R3H G2 source bytes changed after preflight")
    return sha256_file(path)


def derive_seeds(split: str, count: int) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(
            hashlib.sha256(
                f"{EXPERIMENT_ID}|{split}|{index}".encode()
            ).digest()[:4],
            "big",
        )
        for index in range(count)
    )


def _seed_manifest(split: str, count: int) -> dict[str, object]:
    return {
        "schema": "irisu-r3h-g2-seed-manifest-v1",
        "experiment_id": EXPERIMENT_ID,
        "derivation": (
            f'SHA256("{EXPERIMENT_ID}|<split>|<index>")[:4] '
            "big-endian"
        ),
        "split": split,
        "rows": [
            {"index": index, "seed": seed}
            for index, seed in enumerate(derive_seeds(split, count))
        ],
    }


def write_manifests() -> dict[str, str]:
    hashes: dict[str, str] = {}
    all_seeds: list[int] = []
    for split, count in SPLITS.items():
        seeds = derive_seeds(split, count)
        all_seeds.extend(seeds)
        hashes[split] = write_once(
            EXPERIMENT / "seed-manifests" / f"{split}.json",
            _seed_manifest(split, count),
        )
    if len(all_seeds) != len(set(all_seeds)):
        raise RuntimeError("R3H G2 seed collision")
    return hashes


def _require_regular(path: Path) -> None:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_nlink != 1
        or path.resolve() != path
    ):
        raise RuntimeError(f"R3H G2 evidence is not a single-link regular file: {path}")


def _collector_hashes() -> tuple[str, str]:
    identity = json.loads(SOURCE_IDENTITY.read_text())
    files = identity["files"]
    return str(files["collector"]["sha256"]), str(files["protocol"]["sha256"])


def _collection_authorization(split: str) -> dict[str, str]:
    path_by_split = {
        "train-board": PREFLIGHT_RECEIPT,
        "support-board": CHECKPOINT_RECEIPT,
        "margin-board": SUPPORT_CALIBRATION_RECEIPT,
        "offline-board": MARGIN_CALIBRATION_RECEIPT,
    }
    try:
        path = path_by_split[split]
    except KeyError as exc:
        raise ValueError(f"unknown R3H G2 split {split!r}") from exc
    _require_regular(path)
    return {
        "schema": "irisu-r3h-g2-collection-authorization-v1",
        "split": split,
        "receipt": str(path),
        "receipt_sha256": sha256_file(path),
        "source_identity_sha256": verify_source_identity(),
    }


def _load_seed_queries(split: str, index: int, seed: int) -> list[dict[str, Any]]:
    stem = COLLECTION / split / f"{seed:010d}"
    episode_path = stem.with_suffix(".json")
    query_path = stem.with_suffix(".queries.jsonl")
    _require_regular(episode_path)
    _require_regular(query_path)
    episode = json.loads(episode_path.read_text())
    identity = episode.get("collector_identity")
    collector_sha, protocol_sha = _collector_hashes()
    authorization = _collection_authorization(split)
    if not isinstance(identity, Mapping):
        raise RuntimeError(f"collector identity is absent in {episode_path}")
    if (
        episode.get("schema")
        != "r3h-g2-frozen-v5-shadow-label-episode-v1"
        or episode.get("complete") is not True
        or episode.get("development_only") is not True
        or episode.get("sealed_test_allowed") is not False
        or episode.get("split") != split
        or int(episode.get("index", -1)) != index
        or int(episode.get("seed", -1)) != seed
        or type(episode.get("logical_cpu")) is not int
        or episode.get("alternatives_executed") is not False
        or int(episode.get("executed_alternative_count", -1)) != 0
        or episode.get("query_file") != str(query_path)
        or episode.get("query_file_sha256") != sha256_file(query_path)
        or identity.get("experiment_id") != EXPERIMENT_ID
        or identity.get("source_revision") != SOURCE_REVISION
        or identity.get("collector_source_sha256") != collector_sha
        or identity.get("protocol_sha256") != protocol_sha
        or identity.get("authorization") != authorization
        or identity.get("split") != split
        or int(identity.get("index", -1)) != index
        or int(identity.get("seed", -1)) != seed
        or episode.get("collector_identity_sha256")
        != canonical_sha256(identity)
    ):
        raise RuntimeError(f"foreign R3H G2 episode evidence: {episode_path}")
    rows = [
        json.loads(line)
        for line in query_path.read_text().splitlines()
        if line.strip()
    ]
    if not rows or len(rows) != int(episode.get("query_rows", -1)):
        raise RuntimeError(f"incomplete R3H G2 query evidence: {query_path}")
    for query_index, row in enumerate(rows):
        observation = row.get("pre_query_public_observation")
        exact = row.get("exact_query")
        if not isinstance(observation, Mapping) or not isinstance(exact, Mapping):
            raise RuntimeError(f"malformed R3H G2 query row in {query_path}")
        if (
            row.get("schema") != "r3h-g2-exact-shadow-query-v1"
            or row.get("collector_identity_sha256")
            != episode.get("collector_identity_sha256")
            or row.get("split") != split
            or int(row.get("index", -1)) != index
            or int(row.get("seed", -1)) != seed
            or int(row.get("query_index", -1)) != query_index
            or row.get("alternatives_executed") is not False
            or int(row.get("executed_ordinal", -1)) != 0
            or row.get("live_state_restored_exactly") is not True
            or row.get("public_observation_unchanged_by_oracle") is not True
            or row.get("pre_query_public_observation_sha256")
            != canonical_sha256(observation)
            or int(row.get("tick", -1)) != int(observation.get("tick", -2))
            or exact.get("split") != split
            or int(exact.get("seed", -1)) != seed
            or int(exact.get("start_tick", -1)) != int(row.get("tick", -2))
        ):
            raise RuntimeError(f"foreign R3H G2 query row in {query_path}")
    return rows


def _require_complete_collection(split: str) -> tuple[int, ...]:
    if split not in SPLITS:
        raise ValueError(f"unknown R3H G2 split {split!r}")
    expected = derive_seeds(split, SPLITS[split])
    expected_names = {f"{seed:010d}.queries.jsonl" for seed in expected}
    expected_episodes = {f"{seed:010d}.json" for seed in expected}
    directory = COLLECTION / split
    if (
        not directory.is_dir()
        or directory.is_symlink()
        or directory.resolve() != directory
    ):
        raise RuntimeError(f"indirect or absent {split} collection directory")
    allowed = expected_names | expected_episodes
    actual: set[str] = set()
    for path in directory.iterdir():
        if path.name not in allowed:
            raise RuntimeError(f"foreign {split} collection entry: {path}")
        _require_regular(path)
        actual.add(path.name)
    actual_names = actual & expected_names
    actual_episodes = actual & expected_episodes
    if actual_names != expected_names or actual_episodes != expected_episodes:
        raise RuntimeError(
            f"incomplete {split} collection: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"extra={sorted(actual_names - expected_names)}, "
            f"missing_episodes={sorted(expected_episodes - actual_episodes)}, "
            f"extra_episodes={sorted(actual_episodes - expected_episodes)}"
        )
    return expected


def load_queries(split: str) -> tuple[dict[str, Any], ...]:
    expected = _require_complete_collection(split)
    rows: list[dict[str, Any]] = []
    for index, seed in enumerate(expected):
        rows.extend(_load_seed_queries(split, index, seed))
    return tuple(rows)


def load_dataset(split: str) -> BoardResolutionDataset:
    verify_source_identity()
    dataset = BoardResolutionDataset(board_branch_records(load_queries(split)))
    expected = set(derive_seeds(split, SPLITS[split]))
    if set(dataset.seeds) != expected:
        raise RuntimeError(f"{split} contains a seed with no exact query")
    verify_source_identity()
    return dataset


def collection_identity_sha256(split: str) -> str:
    seeds = _require_complete_collection(split)
    rows: list[dict[str, str]] = []
    for seed in seeds:
        stem = COLLECTION / split / f"{seed:010d}"
        for path in (stem.with_suffix(".json"), stem.with_suffix(".queries.jsonl")):
            _require_regular(path)
            rows.append({"path": str(path), "sha256": sha256_file(path)})
    return canonical_sha256(
        {
            "schema": "irisu-r3h-g2-collection-identity-v1",
            "experiment_id": EXPERIMENT_ID,
            "split": split,
            "files": rows,
        }
    )


def _artifact_receipt(
    artifact: Path, receipt: Path, *, schema: str
) -> dict[str, Any]:
    _require_regular(artifact)
    _require_regular(receipt)
    value = json.loads(receipt.read_text())
    if (
        value.get("schema") != schema
        or value.get("artifact") != str(artifact)
        or value.get("artifact_sha256") != sha256_file(artifact)
        or value.get("source_identity_sha256") != verify_source_identity()
    ):
        raise RuntimeError(f"R3H G2 receipt mismatch: {receipt}")
    return value


def _read_report(
    path: Path,
    *,
    schema: str,
    development_only: bool = True,
) -> dict[str, Any]:
    _require_regular(path)
    value = json.loads(path.read_text())
    if (
        value.get("schema") != schema
        or (
            development_only
            and (
                value.get("development_only") is not True
                or value.get("sealed_test_allowed") is not False
            )
        )
    ):
        raise RuntimeError(f"foreign R3H G2 report: {path}")
    return value


def _validate_preregistration() -> None:
    path = EXPERIMENT / "preregistration.json"
    _require_regular(path)
    _require_regular(PROTOCOL)
    value = json.loads(path.read_text())
    if (
        value.get("experiment_id") != EXPERIMENT_ID
        or value.get("source_revision") != SOURCE_REVISION
        or value.get("development_only") is not True
        or value.get("sealed_test_allowed") is not False
        or value.get("outcomes_viewed_before_preregistration") is not False
        or value.get("generation_01_outcomes_are_training_data") is not False
        or value.get("generation_02_outcomes_collected_before_freeze") is not False
        or value.get("status") != PREREGISTRATION_STATUS
        or value.get("protocol_sha256") != sha256_file(PROTOCOL)
        or value.get("frozen_source_sha256")
        != _preregistered_source_hashes()
        or value.get("runtime_manifest") != runtime_identity()
    ):
        raise RuntimeError("R3H G2 preregistration identity is malformed")


def _preflight_rows(_core: int) -> tuple[tuple[Path, object], ...]:
    manifests = {
        split: _seed_manifest(split, count)
        for split, count in SPLITS.items()
    }
    manifest_hashes = {
        split: hashlib.sha256(_artifact_bytes(value)).hexdigest()
        for split, value in manifests.items()
    }
    identity = source_identity()
    identity_sha = hashlib.sha256(_artifact_bytes(identity)).hexdigest()
    report = {
        "schema": "irisu-r3h-g2-preflight-v1",
        "development_only": True,
        "sealed_test_allowed": False,
        "source_identity_sha256": identity_sha,
        "seed_manifest_sha256": manifest_hashes,
        "logical_cpu_count": 1,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    report_sha = hashlib.sha256(_artifact_bytes(report)).hexdigest()
    receipt = {
        "schema": "irisu-r3h-g2-preflight-receipt-v1",
        "artifact": str(PREFLIGHT),
        "artifact_sha256": report_sha,
        "source_identity_sha256": identity_sha,
    }
    return (
        *(
            (
                EXPERIMENT / "seed-manifests" / f"{split}.json",
                manifests[split],
            )
            for split in SPLITS
        ),
        (SOURCE_IDENTITY, identity),
        (PREFLIGHT, report),
        (PREFLIGHT_RECEIPT, receipt),
    )


def _validate_preflight_prefix(core: int) -> None:
    if (
        not EXPERIMENT.is_dir()
        or EXPERIMENT.is_symlink()
        or EXPERIMENT.resolve() != EXPERIMENT
    ):
        raise RuntimeError("R3H G2 experiment root is indirect or absent")
    rows = _preflight_rows(core)
    quarantines = _validate_campaign_incomplete()
    expected = {path: _artifact_bytes(value) for path, value in rows}
    root_targets = {
        path.name: path for path in expected if path.parent == EXPERIMENT
    }
    manifest_root = EXPERIMENT / "seed-manifests"
    manifest_targets = {
        path.name: path for path in expected if path.parent == manifest_root
    }
    root_entries = {entry.name: entry for entry in EXPERIMENT.iterdir()}
    allowed_root = {
        PROTOCOL.name,
        "preregistration.json",
        "_incomplete",
        manifest_root.name,
        *root_targets,
    }
    temporary: list[tuple[Path, Path]] = []

    def classify(entries: Mapping[str, Path], targets: Mapping[str, Path]) -> None:
        for name, entry in entries.items():
            if name in targets:
                continue
            matches = [
                target
                for target_name, target in targets.items()
                if name.startswith(f".{target_name}.tmp-")
                and len(name[len(f".{target_name}.tmp-") :].split("-")) == 2
                and all(
                    piece.isdigit()
                    for piece in name[
                        len(f".{target_name}.tmp-") :
                    ].split("-")
                )
            ]
            if len(matches) != 1:
                raise RuntimeError(f"foreign R3H G2 preflight entry: {entry}")
            temporary.append((entry, matches[0]))

    for name, entry in root_entries.items():
        if name not in allowed_root and not name.startswith("."):
            raise RuntimeError(f"foreign R3H G2 preflight entry: {entry}")
    classify(
        {
            name: entry
            for name, entry in root_entries.items()
            if name not in allowed_root
        },
        root_targets,
    )
    _require_regular(PROTOCOL)
    _require_regular(EXPERIMENT / "preregistration.json")
    if manifest_root.exists():
        if (
            not manifest_root.is_dir()
            or manifest_root.is_symlink()
            or manifest_root.resolve() != manifest_root
        ):
            raise RuntimeError("indirect R3H G2 preflight manifest root")
        manifest_entries = {
            entry.name: entry for entry in manifest_root.iterdir()
        }
        classify(manifest_entries, manifest_targets)
    elif manifest_root.is_symlink():
        raise RuntimeError("indirect R3H G2 preflight manifest root")

    present: list[bool] = []
    for path, encoded in expected.items():
        exists = path.exists() or path.is_symlink()
        present.append(exists)
        if not exists:
            continue
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve() != path
            or path.stat().st_nlink not in (1, 2)
            or path.read_bytes() != encoded
        ):
            raise RuntimeError(f"foreign R3H G2 preflight artifact: {path}")
    prefix = 0
    while prefix < len(present) and present[prefix]:
        prefix += 1
    if any(present[prefix:]):
        raise RuntimeError("out-of-order R3H G2 preflight artifact")
    row_paths = [path for path, _value in rows]
    quarantine_links: list[tuple[Path, Path]] = []
    for target, quarantine in quarantines:
        if target not in expected:
            raise RuntimeError(
                f"out-of-order R3H G2 preflight quarantine: {quarantine}"
            )
        index = row_paths.index(target)
        if index > prefix:
            raise RuntimeError(
                f"out-of-order R3H G2 preflight quarantine: {quarantine}"
            )
        if quarantine.stat().st_nlink == 2:
            quarantine_links.append((target, quarantine))
    if len(temporary) > 1:
        raise RuntimeError("multiple R3H G2 preflight writer remnants")
    if len(quarantine_links) > 1:
        raise RuntimeError("multiple R3H G2 preflight quarantine boundaries")
    if temporary:
        remnant, target = temporary[0]
        if (
            remnant.is_symlink()
            or not remnant.is_file()
            or remnant.resolve() != remnant
            or remnant.stat().st_nlink not in (1, 2)
        ):
            raise RuntimeError(f"indirect R3H G2 preflight remnant: {remnant}")
        index = row_paths.index(target)
        if remnant.stat().st_nlink == 1:
            if index > prefix or (
                index == prefix and (target.exists() or target.is_symlink())
            ) or (
                index < prefix and not target.exists()
            ):
                raise RuntimeError("out-of-order R3H G2 preflight remnant")
        else:
            postlink = (
                target.exists()
                and os.path.samefile(remnant, target)
                and index == prefix - 1
            )
            linked_quarantines = [
                quarantine
                for quarantine_target, quarantine in quarantine_links
                if quarantine_target == target
                and os.path.samefile(remnant, quarantine)
            ]
            quarantine_boundary = (
                len(linked_quarantines) == 1
                and index <= prefix
                and (
                    (index == prefix and not target.exists())
                    or (index < prefix and target.exists())
                )
            )
            if not postlink and not quarantine_boundary:
                raise RuntimeError("foreign R3H G2 preflight link boundary")
    elif quarantine_links:
        raise RuntimeError("orphan R3H G2 preflight quarantine boundary")
    for index, (path, _value) in enumerate(rows):
        if (
            path.exists()
            and path.stat().st_nlink == 2
            and (
                not temporary
                or temporary[0][1] != path
                or not os.path.samefile(temporary[0][0], path)
                or index != prefix - 1
            )
        ):
            raise RuntimeError("foreign R3H G2 preflight artifact links")


def command_preflight(_args: argparse.Namespace) -> None:
    runtime_identity(refresh=True)
    if _revision() != SOURCE_REVISION:
        raise RuntimeError("R3H G2 repository revision mismatch")
    _validate_preregistration()
    _validate_all_incomplete(
        collector_sha256=_preregistered_source_hashes()["collector"]
    )
    core = _one_cpu()
    _validate_preflight_prefix(core)
    manifests = write_manifests()
    identity_sha = write_once(
        SOURCE_IDENTITY, source_identity()
    )
    verify_source_identity()
    report = {
        "schema": "irisu-r3h-g2-preflight-v1",
        "development_only": True,
        "sealed_test_allowed": False,
        "source_identity_sha256": identity_sha,
        "seed_manifest_sha256": manifests,
        "logical_cpu_count": 1,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }
    artifact = PREFLIGHT
    artifact_sha = write_once(artifact, report)
    write_once(
        PREFLIGHT_RECEIPT,
        {
            "schema": "irisu-r3h-g2-preflight-receipt-v1",
            "artifact": str(artifact),
            "artifact_sha256": artifact_sha,
            "source_identity_sha256": identity_sha,
        },
    )
    verify_source_identity()
    print(json.dumps({**report, "artifact_sha256": artifact_sha}, sort_keys=True))


def _require_preflight() -> None:
    _artifact_receipt(
        PREFLIGHT,
        PREFLIGHT_RECEIPT,
        schema="irisu-r3h-g2-preflight-receipt-v1",
    )
    report = _read_report(PREFLIGHT, schema="irisu-r3h-g2-preflight-v1")
    if report.get("source_identity_sha256") != verify_source_identity():
        raise RuntimeError("R3H G2 preflight source identity mismatch")
    recorded = report.get("seed_manifest_sha256")
    if not isinstance(recorded, Mapping):
        raise RuntimeError("R3H G2 preflight lacks seed manifest identities")
    for split, count in SPLITS.items():
        path = EXPERIMENT / "seed-manifests" / f"{split}.json"
        _require_regular(path)
        manifest = json.loads(path.read_text())
        expected_rows = [
            {"index": index, "seed": seed}
            for index, seed in enumerate(derive_seeds(split, count))
        ]
        if (
            recorded.get(split) != sha256_file(path)
            or manifest.get("experiment_id") != EXPERIMENT_ID
            or manifest.get("split") != split
            or manifest.get("rows") != expected_rows
        ):
            raise RuntimeError(f"R3H G2 {split} seed manifest changed")


def _pilot_report(dataset: BoardResolutionDataset) -> dict[str, object]:
    groups: dict[int, list[Any]] = defaultdict(list)
    for record in dataset.records:
        groups[int(record.seed)].append(record)
    resolved = {
        seed
        for seed, rows in groups.items()
        if any(
            row.ordinal != 0 and bool(row.candidate_resolved)
            for row in rows
        )
    }
    viable = {
        seed
        for seed, rows in groups.items()
        if any(
            row.ordinal != 0
            and bool(row.candidate_resolved)
            and not row.exact_unsafe
            and row.b2 is not None
            and float(row.b2) >= 0
            and float(row.score_advantage) > 0
            for row in rows
        )
    }
    resolved_rows = sum(
        record.ordinal != 0 and bool(record.candidate_resolved)
        for record in dataset.records
    )
    gates = {
        "all_preregistered_seeds": len(groups) == SPLITS["train-board"],
        "resolved_alternative_seeds": (
            len(resolved) >= PILOT_LIMITS["resolved_alternative_seeds"]
        ),
        "viable_alternative_seeds": (
            len(viable) >= PILOT_LIMITS["viable_alternative_seeds"]
        ),
        "resolved_nonincumbent_rows": (
            resolved_rows >= PILOT_LIMITS["resolved_nonincumbent_rows"]
        ),
    }
    return {
        "schema": "irisu-r3h-g2-structural-pilot-v1",
        "seeds": len(groups),
        "resolved_alternative_seeds": len(resolved),
        "viable_alternative_seeds": len(viable),
        "resolved_nonincumbent_rows": resolved_rows,
        "limits": PILOT_LIMITS,
        "gates": gates,
        "passed": all(gates.values()),
    }


def command_pilot(_args: argparse.Namespace) -> None:
    runtime_identity(refresh=True)
    _require_preflight()
    identity_sha = verify_source_identity()
    dataset = load_dataset("train-board")
    report = {
        **_pilot_report(dataset),
        "development_only": True,
        "sealed_test_allowed": False,
        "dataset_sha256": dataset.sha256,
        "collection_identity_sha256": collection_identity_sha256(
            "train-board"
        ),
        "source_identity_sha256": identity_sha,
    }
    artifact = PILOT
    artifact_sha = write_once(artifact, report)
    write_once(
        PILOT_RECEIPT,
        {
            "schema": "irisu-r3h-g2-pilot-receipt-v1",
            "artifact": str(artifact),
            "artifact_sha256": artifact_sha,
            "dataset_sha256": dataset.sha256,
            "collection_identity_sha256": report[
                "collection_identity_sha256"
            ],
            "passed": report["passed"],
            "source_identity_sha256": identity_sha,
        },
    )
    verify_source_identity()
    print(json.dumps({**report, "artifact_sha256": artifact_sha}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


def _require_pilot() -> dict[str, Any]:
    receipt = _artifact_receipt(
        PILOT,
        PILOT_RECEIPT,
        schema="irisu-r3h-g2-pilot-receipt-v1",
    )
    report = _read_report(PILOT, schema="irisu-r3h-g2-structural-pilot-v1")
    if (
        receipt.get("passed") is not True
        or report.get("passed") is not True
        or receipt.get("dataset_sha256") != report.get("dataset_sha256")
        or receipt.get("collection_identity_sha256")
        != report.get("collection_identity_sha256")
        or report.get("collection_identity_sha256")
        != collection_identity_sha256("train-board")
        or report.get("source_identity_sha256") != verify_source_identity()
    ):
        raise RuntimeError("R3H G2 structural pilot is terminal NO-GO")
    return report


def _manifest(value: object) -> dict[str, object]:
    function = getattr(value, "manifest", None)
    if not callable(function):
        raise TypeError("identity-bound object lacks a manifest")
    manifest = function()
    if not isinstance(manifest, dict):
        raise TypeError("identity-bound manifest is malformed")
    return manifest


def _frozen_model_config_manifest() -> dict[str, object]:
    return ResolutionFirstG2Config().manifest()


def _require_frozen_model_config(
    model: object, metadata: Mapping[str, object]
) -> str:
    expected = _frozen_model_config_manifest()
    expected_sha = canonical_sha256(expected)
    if (
        _manifest(model).get("config") != expected
        or metadata.get("model_config_sha256") != expected_sha
    ):
        raise RuntimeError("R3H G2 checkpoint uses a foreign model config")
    return expected_sha


def command_train(_args: argparse.Namespace) -> None:
    runtime_identity(refresh=True)
    pilot = _require_pilot()
    identity_sha = verify_source_identity()
    dataset = load_dataset("train-board")
    if dataset.sha256 != pilot.get("dataset_sha256"):
        raise RuntimeError("R3H G2 training collection changed after pilot")
    recovered = _recover_training(dataset, identity_sha)
    if recovered is not None:
        verify_source_identity()
        print(json.dumps({**recovered, "recovered": True}, sort_keys=True))
        return
    config = ResolutionFirstG2Config()
    model = train_resolution_first_g2(dataset, config=config)
    checkpoint = CHECKPOINT
    checkpoint_sha = save_checkpoint_g2(
        checkpoint,
        model,
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "dataset_sha256": dataset.sha256,
            "source_identity_sha256": identity_sha,
            "model_config_sha256": canonical_sha256(config.manifest()),
        },
    )
    report = _training_report(model, dataset, checkpoint, checkpoint_sha)
    _write_training_receipt(report, identity_sha=identity_sha)
    verify_source_identity()
    print(json.dumps(report, sort_keys=True))


def _training_report(
    model: object,
    dataset: BoardResolutionDataset,
    checkpoint: Path,
    checkpoint_sha: str,
) -> dict[str, object]:
    model_manifest = _manifest(model)
    config_sha = canonical_sha256(_frozen_model_config_manifest())
    if (
        model_manifest.get("config") != _frozen_model_config_manifest()
    ):
        raise RuntimeError("R3H G2 trained model differs from frozen config")
    predictions = predict_records_g2(model, dataset)
    alternatives = [
        row for row in predictions if int(row["ordinal"]) != 0
    ]
    if not alternatives:
        raise RuntimeError("R3H G2 training data has no nonincumbent rows")
    accuracy = sum(
        (float(row["resolution_mean"]) >= 0.5)
        == bool(
            row.get(
                "exact_candidate_resolved",
                row.get("exact_finite_pair", False),
            )
        )
        for row in alternatives
    ) / len(alternatives)
    return {
        "schema": "irisu-r3h-g2-training-v1",
        "development_only": True,
        "sealed_test_allowed": False,
        "source_identity_sha256": verify_source_identity(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "dataset_sha256": dataset.sha256,
        "collection_identity_sha256": collection_identity_sha256(
            "train-board"
        ),
        "records": len(dataset.records),
        "seeds": len(dataset.seeds),
        "model_config_sha256": config_sha,
        "model_manifest": model_manifest,
        "training_nonincumbent_resolution_accuracy": accuracy,
    }


def _write_training_receipt(
    report: dict[str, object], *, identity_sha: str
) -> str:
    checkpoint_value = report.get("checkpoint")
    checkpoint_sha = report.get("checkpoint_sha256")
    dataset_sha = report.get("dataset_sha256")
    collection_sha = report.get("collection_identity_sha256")
    config_sha = report.get("model_config_sha256")
    model_manifest = report.get("model_manifest")
    if (
        not isinstance(checkpoint_value, str)
        or not isinstance(checkpoint_sha, str)
        or not isinstance(dataset_sha, str)
        or not isinstance(collection_sha, str)
        or config_sha
        != canonical_sha256(_frozen_model_config_manifest())
        or not isinstance(model_manifest, Mapping)
        or model_manifest.get("config")
        != _frozen_model_config_manifest()
    ):
        raise ValueError("R3H G2 training report identities are malformed")
    checkpoint = Path(checkpoint_value)
    if (
        checkpoint != CHECKPOINT
        or not checkpoint.is_file()
        or checkpoint.is_symlink()
        or sha256_file(checkpoint) != checkpoint_sha
    ):
        raise RuntimeError("R3H G2 training report checkpoint identity mismatch")
    artifact = TRAINING
    artifact_sha = write_once(artifact, report)
    receipt_sha = write_once(
        CHECKPOINT_RECEIPT,
        {
            "schema": "irisu-r3h-g2-checkpoint-receipt-v1",
            "artifact": str(artifact),
            "artifact_sha256": artifact_sha,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "dataset_sha256": dataset_sha,
            "collection_identity_sha256": collection_sha,
            "model_config_sha256": config_sha,
            "model_manifest_sha256": canonical_sha256(model_manifest),
            "source_identity_sha256": identity_sha,
        },
    )
    report["artifact_sha256"] = artifact_sha
    report["receipt_sha256"] = receipt_sha
    return receipt_sha


def _recover_training(
    dataset: BoardResolutionDataset, identity_sha: str
) -> dict[str, object] | None:
    """Recover only after the checkpoint's atomic link boundary."""

    checkpoint = CHECKPOINT
    artifact = TRAINING
    receipt = CHECKPOINT_RECEIPT
    _recover_checkpoint_temps(checkpoint, dataset, identity_sha)
    exists = {
        "checkpoint": checkpoint.exists(),
        "artifact": artifact.exists(),
        "receipt": receipt.exists(),
    }
    if not any(exists.values()):
        return None
    if not exists["checkpoint"]:
        raise RuntimeError("R3H G2 training evidence exists without its checkpoint")
    if exists["receipt"] and not exists["artifact"]:
        raise RuntimeError("R3H G2 checkpoint receipt precedes its training artifact")
    _require_regular(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    model, metadata = load_checkpoint_g2(checkpoint)
    if (
        metadata.get("experiment_id") != EXPERIMENT_ID
        or metadata.get("dataset_sha256") != dataset.sha256
        or metadata.get("source_identity_sha256") != identity_sha
    ):
        raise RuntimeError("R3H G2 linked checkpoint cannot be identity-recovered")
    _require_frozen_model_config(model, metadata)
    report = _training_report(model, dataset, checkpoint, checkpoint_sha)
    if exists["artifact"]:
        _require_regular(artifact)
        if json.loads(artifact.read_text()) != report:
            raise RuntimeError("R3H G2 training report differs from checkpoint")
    _write_training_receipt(report, identity_sha=identity_sha)
    return report


def _recover_checkpoint_temps(
    checkpoint: Path,
    dataset: BoardResolutionDataset,
    identity_sha: str,
) -> None:
    _mkdir_durable(checkpoint.parent)
    linked = checkpoint.exists() or checkpoint.is_symlink()
    if linked and (checkpoint.is_symlink() or not checkpoint.is_file()):
        raise RuntimeError("R3H G2 checkpoint is indirect")
    for temporary in _atomic_temp_candidates(checkpoint):
        if temporary.is_symlink() or not temporary.is_file():
            raise RuntimeError(f"indirect R3H G2 checkpoint remnant: {temporary}")
        if linked and os.path.samefile(temporary, checkpoint):
            temporary.unlink()
            _fsync_directory(checkpoint.parent)
            continue
        if temporary.stat().st_nlink != 1:
            _quarantine_atomic_temp(temporary, checkpoint)
            continue
        recoverable = False
        if not linked:
            try:
                model, metadata = load_checkpoint_g2(temporary)
                recoverable = (
                    metadata.get("experiment_id") == EXPERIMENT_ID
                    and metadata.get("dataset_sha256") == dataset.sha256
                    and metadata.get("source_identity_sha256") == identity_sha
                )
                if recoverable:
                    _require_frozen_model_config(model, metadata)
            except Exception:
                recoverable = False
        if recoverable:
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.link(temporary, checkpoint)
            _fsync_directory(checkpoint.parent)
            temporary.unlink()
            _fsync_directory(checkpoint.parent)
            linked = True
        else:
            _quarantine_atomic_temp(temporary, checkpoint)
    if linked and checkpoint.stat().st_nlink != 1:
        raise RuntimeError("R3H G2 checkpoint has foreign hardlinks")


def _load_model():
    identity_sha = verify_source_identity()
    receipt = _artifact_receipt(
        TRAINING,
        CHECKPOINT_RECEIPT,
        schema="irisu-r3h-g2-checkpoint-receipt-v1",
    )
    training = _read_report(TRAINING, schema="irisu-r3h-g2-training-v1")
    checkpoint = Path(str(receipt["checkpoint"]))
    if checkpoint != CHECKPOINT:
        raise RuntimeError("R3H G2 checkpoint receipt names a foreign path")
    _require_regular(checkpoint)
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != receipt.get("checkpoint_sha256"):
        raise RuntimeError("R3H G2 checkpoint hash mismatch")
    model, metadata = load_checkpoint_g2(checkpoint)
    expected_config_sha = _require_frozen_model_config(model, metadata)
    dataset = load_dataset("train-board")
    if (
        metadata.get("experiment_id") != EXPERIMENT_ID
        or metadata.get("source_identity_sha256") != identity_sha
        or metadata.get("dataset_sha256") != receipt.get("dataset_sha256")
        or dataset.sha256 != receipt.get("dataset_sha256")
        or collection_identity_sha256("train-board")
        != receipt.get("collection_identity_sha256")
        or training.get("collection_identity_sha256")
        != receipt.get("collection_identity_sha256")
        or training.get("checkpoint_sha256") != checkpoint_sha
        or training.get("dataset_sha256") != dataset.sha256
        or training.get("source_identity_sha256") != identity_sha
        or training.get("model_config_sha256") != expected_config_sha
        or receipt.get("model_config_sha256") != expected_config_sha
        or training.get("model_manifest", {}).get("config")
        != _frozen_model_config_manifest()
        or receipt.get("artifact_sha256") != sha256_file(TRAINING)
        or canonical_sha256(_manifest(model))
        != receipt.get("model_manifest_sha256")
    ):
        raise RuntimeError("R3H G2 checkpoint metadata mismatch")
    verify_source_identity()
    return model, checkpoint_sha


def _support_grid(
    predictions: Sequence[Mapping[str, object]],
    thresholds: Sequence[float],
) -> list[dict[str, object]]:
    groups: dict[tuple[int, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in predictions:
        groups[(int(row["seed"]), str(row["query_id"]))].append(row)
    alternative_count = sum(
        sum(int(row["ordinal"]) != 0 for row in rows)
        for rows in groups.values()
    )
    grid: list[dict[str, object]] = []
    for threshold in thresholds:
        selected: list[Mapping[str, object]] = []
        covered: set[tuple[int, str]] = set()
        for key, rows in groups.items():
            incumbents = [row for row in rows if int(row["ordinal"]) == 0]
            if len(incumbents) != 1:
                raise RuntimeError(f"R3H G2 query {key!r} lacks one incumbent")
            incumbent_action = str(incumbents[0]["action_id"])
            for row in rows:
                if (
                    int(row["ordinal"]) != 0
                    and bool(row["envelope_supported"])
                    and str(row["action_id"]) != incumbent_action
                    and float(row["support_score"]) > float(threshold)
                    and float(row["score_lcb"]) > 0
                ):
                    selected.append(row)
                    covered.add(key)
        bad = [
            row
            for row in selected
            if not bool(row["exact_candidate_resolved"])
            or bool(row["exact_unsafe"])
        ]
        coverage = len(covered) / len(groups)
        grid.append(
            {
                "threshold": float(threshold),
                "selected_candidates": len(selected),
                "candidate_count": alternative_count,
                "selected_queries": len(covered),
                "query_count": len(groups),
                "coverage": coverage,
                "selected_bad_candidates": len(bad),
                "selected_bad_seeds": len({int(row["seed"]) for row in bad}),
                "passed": not bad and coverage >= 0.05,
            }
        )
    return grid


def _write_stage_receipt(
    artifact: Path,
    receipt_path: Path,
    report: dict[str, object],
    *,
    receipt_schema: str,
    checkpoint_sha: str,
    identity_sha: str,
    extra: Mapping[str, object],
) -> str:
    artifact_sha = write_once(artifact, report)
    receipt = {
        "schema": receipt_schema,
        "artifact": str(artifact),
        "artifact_sha256": artifact_sha,
        "checkpoint_sha256": checkpoint_sha,
        "passed": report["passed"],
        "source_identity_sha256": identity_sha,
        **dict(extra),
    }
    write_once(receipt_path, receipt)
    return artifact_sha


def _compute_support_calibration(model):
    config = model.config
    support_dataset = load_dataset("support-board")
    support_predictions = predict_records_g2(model, support_dataset)
    try:
        auroc = resolution_auroc_g2(support_predictions, minimum=0.75)
    except ValueError as error:
        alternatives = [
            row for row in support_predictions if int(row["ordinal"]) != 0
        ]
        positives = sum(
            bool(row["exact_candidate_resolved"]) for row in alternatives
        )
        auroc = {
            "schema": "irisu-r3h-resolution-auroc-g2",
            "candidates": len(alternatives),
            "positive": positives,
            "negative": len(alternatives) - positives,
            "auroc": None,
            "minimum": 0.75,
            "passed": False,
            "terminal_reason": str(error),
        }
    grid = _support_grid(support_predictions, config.support_thresholds)
    if not auroc["passed"]:
        return support_dataset, None, auroc, grid
    try:
        support = fit_support_calibration_g2(
            support_predictions,
            thresholds=config.support_thresholds,
            minimum_coverage=0.05,
            minimum_auroc=0.75,
        )
    except RuntimeError:
        if any(point["passed"] for point in grid):
            raise
        return support_dataset, None, auroc, grid
    manifest = support.manifest()
    if manifest.get("grid") != grid:
        raise RuntimeError("R3H G2 learner/campaign support grids diverged")
    passing = next(point for point in grid if point["passed"])
    if float(passing["threshold"]) != float(support.threshold):
        raise RuntimeError("R3H G2 support threshold implementation diverged")
    return support_dataset, support, auroc, grid


def command_calibrate_support(_args: argparse.Namespace) -> None:
    runtime_identity(refresh=True)
    model, checkpoint_sha = _load_model()
    identity_sha = verify_source_identity()
    support_dataset, support, auroc, grid = _compute_support_calibration(model)
    failure = None
    if not auroc["passed"]:
        failure = "heldout-resolution-auroc"
    elif support is None:
        failure = "reachable-prefix-support"
    support_manifest = None if support is None else support.manifest()
    report: dict[str, object] = {
        "schema": "irisu-r3h-g2-support-calibration-v1",
        "development_only": True,
        "sealed_test_allowed": False,
        "source_identity_sha256": identity_sha,
        "checkpoint_sha256": checkpoint_sha,
        "support_dataset_sha256": support_dataset.sha256,
        "support_collection_identity_sha256": collection_identity_sha256(
            "support-board"
        ),
        "resolution_auroc": auroc,
        "support_grid": grid,
        "support": support_manifest,
        "terminal_failure": failure,
        "passed": failure is None,
    }
    artifact_sha = _write_stage_receipt(
        SUPPORT_CALIBRATION,
        SUPPORT_CALIBRATION_RECEIPT,
        report,
        receipt_schema="irisu-r3h-g2-support-calibration-receipt-v1",
        checkpoint_sha=checkpoint_sha,
        identity_sha=identity_sha,
        extra={
            "support_dataset_sha256": support_dataset.sha256,
            "support_collection_identity_sha256": report[
                "support_collection_identity_sha256"
            ],
            "resolution_auroc_sha256": canonical_sha256(auroc),
            "support_sha256": (
                None
                if support_manifest is None
                else canonical_sha256(support_manifest)
            ),
        },
    )
    verify_source_identity()
    print(
        json.dumps(
            _jsonable({**report, "artifact_sha256": artifact_sha}),
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


def _support_calibration(model, checkpoint_sha: str):
    receipt = _artifact_receipt(
        SUPPORT_CALIBRATION,
        SUPPORT_CALIBRATION_RECEIPT,
        schema="irisu-r3h-g2-support-calibration-receipt-v1",
    )
    report = _read_report(
        SUPPORT_CALIBRATION,
        schema="irisu-r3h-g2-support-calibration-v1",
    )
    if (
        receipt.get("checkpoint_sha256") != checkpoint_sha
        or report.get("checkpoint_sha256") != checkpoint_sha
        or receipt.get("passed") is not True
        or report.get("passed") is not True
        or report.get("source_identity_sha256") != verify_source_identity()
    ):
        raise RuntimeError("R3H G2 support calibration is absent or terminal NO-GO")
    support_dataset, support, auroc, grid = _compute_support_calibration(model)
    if support is None:
        raise RuntimeError("R3H G2 passed support calibration cannot be reproduced")
    manifest = support.manifest()
    if (
        report.get("support_dataset_sha256") != support_dataset.sha256
        or report.get("support_collection_identity_sha256")
        != collection_identity_sha256("support-board")
        or report.get("resolution_auroc") != auroc
        or report.get("support_grid") != grid
        or report.get("support") != manifest
        or receipt.get("support_dataset_sha256") != support_dataset.sha256
        or receipt.get("support_collection_identity_sha256")
        != report.get("support_collection_identity_sha256")
        or receipt.get("resolution_auroc_sha256") != canonical_sha256(auroc)
        or receipt.get("support_sha256") != canonical_sha256(manifest)
    ):
        raise RuntimeError("R3H G2 support calibration cannot be reproduced")
    return support


def _compute_margin_calibration(model, support):
    margin_dataset = load_dataset("margin-board")
    predictions = predict_records_g2(model, margin_dataset)
    alternatives = [
        row for row in predictions if int(row["ordinal"]) != 0
    ]
    if not alternatives:
        raise RuntimeError("R3H G2 margin data has no alternatives")
    required = {
        "exact_candidate_resolved",
        "exact_b2",
        "b2_q10_mean",
        "b2_q10_std",
        "b2_lcb",
    }
    missing = sorted(
        {
            name
            for row in alternatives
            for name in required
            if name not in row
        }
    )
    if missing:
        raise RuntimeError(
            "R3H G2 learner lacks absolute-B2 calibration fields: "
            + ", ".join(missing)
        )
    selective = fit_selective_calibration_g2(
        predictions,
        support,
        alpha=model.config.conformal_alpha,
        required_episodes=SPLITS["margin-board"],
    )
    return margin_dataset, selective


def command_calibrate_margin(_args: argparse.Namespace) -> None:
    runtime_identity(refresh=True)
    model, checkpoint_sha = _load_model()
    identity_sha = verify_source_identity()
    support = _support_calibration(model, checkpoint_sha)
    support_receipt_sha = sha256_file(SUPPORT_CALIBRATION_RECEIPT)
    margin_dataset, selective = _compute_margin_calibration(model, support)
    selective_manifest = selective.manifest()
    finite = math.isfinite(float(selective.q))
    report: dict[str, object] = {
        "schema": "irisu-r3h-g2-margin-calibration-v1",
        "development_only": True,
        "sealed_test_allowed": False,
        "source_identity_sha256": identity_sha,
        "checkpoint_sha256": checkpoint_sha,
        "support_calibration_receipt_sha256": support_receipt_sha,
        "support_calibration_sha256": support.sha256,
        "margin_dataset_sha256": margin_dataset.sha256,
        "margin_collection_identity_sha256": collection_identity_sha256(
            "margin-board"
        ),
        "calibration_target": "absolute_b2",
        "selective": selective_manifest,
        "terminal_failure": (
            None if finite else "nonfinite-absolute-b2-conformal-q"
        ),
        "passed": finite,
    }
    artifact_sha = _write_stage_receipt(
        MARGIN_CALIBRATION,
        MARGIN_CALIBRATION_RECEIPT,
        report,
        receipt_schema="irisu-r3h-g2-margin-calibration-receipt-v1",
        checkpoint_sha=checkpoint_sha,
        identity_sha=identity_sha,
        extra={
            "support_calibration_receipt_sha256": support_receipt_sha,
            "support_calibration_sha256": support.sha256,
            "margin_dataset_sha256": margin_dataset.sha256,
            "margin_collection_identity_sha256": report[
                "margin_collection_identity_sha256"
            ],
            "calibration_target": "absolute_b2",
            "selective_sha256": canonical_sha256(
                _jsonable(selective_manifest)
            ),
            "finite_q_b2": finite,
        },
    )
    verify_source_identity()
    print(
        json.dumps(
            _jsonable({**report, "artifact_sha256": artifact_sha}),
            sort_keys=True,
        )
    )
    if not report["passed"]:
        raise SystemExit(2)


def _margin_calibration(model, checkpoint_sha: str, support):
    receipt = _artifact_receipt(
        MARGIN_CALIBRATION,
        MARGIN_CALIBRATION_RECEIPT,
        schema="irisu-r3h-g2-margin-calibration-receipt-v1",
    )
    report = _read_report(
        MARGIN_CALIBRATION,
        schema="irisu-r3h-g2-margin-calibration-v1",
    )
    if (
        receipt.get("checkpoint_sha256") != checkpoint_sha
        or report.get("checkpoint_sha256") != checkpoint_sha
        or receipt.get("passed") is not True
        or report.get("passed") is not True
        or receipt.get("finite_q_b2") is not True
        or receipt.get("calibration_target") != "absolute_b2"
        or report.get("calibration_target") != "absolute_b2"
        or receipt.get("support_calibration_receipt_sha256")
        != sha256_file(SUPPORT_CALIBRATION_RECEIPT)
        or report.get("support_calibration_receipt_sha256")
        != sha256_file(SUPPORT_CALIBRATION_RECEIPT)
        or receipt.get("support_calibration_sha256") != support.sha256
        or report.get("support_calibration_sha256") != support.sha256
        or report.get("source_identity_sha256") != verify_source_identity()
    ):
        raise RuntimeError("R3H G2 margin calibration is absent or terminal NO-GO")
    margin_dataset, selective = _compute_margin_calibration(model, support)
    manifest = selective.manifest()
    if (
        not math.isfinite(float(selective.q))
        or report.get("margin_dataset_sha256") != margin_dataset.sha256
        or report.get("margin_collection_identity_sha256")
        != collection_identity_sha256("margin-board")
        or report.get("selective") != manifest
        or receipt.get("margin_dataset_sha256") != margin_dataset.sha256
        or receipt.get("margin_collection_identity_sha256")
        != report.get("margin_collection_identity_sha256")
        or receipt.get("selective_sha256") != canonical_sha256(manifest)
    ):
        raise RuntimeError("R3H G2 margin calibration cannot be reproduced")
    return selective


def command_screen(_args: argparse.Namespace) -> None:
    runtime_identity(refresh=True)
    model, checkpoint_sha = _load_model()
    support = _support_calibration(model, checkpoint_sha)
    selective = _margin_calibration(model, checkpoint_sha, support)
    dataset = load_dataset("offline-board")
    predictions = predict_records_g2(model, dataset)
    selections = select_candidates_g2(predictions, support, selective)
    report = viability_report_g2(
        selections, selective, minimum_coverage=0.05
    )
    report.update(
        {
            "development_only": True,
            "sealed_test_allowed": False,
            "checkpoint_sha256": checkpoint_sha,
            "dataset_sha256": dataset.sha256,
            "collection_identity_sha256": collection_identity_sha256(
                "offline-board"
            ),
            "support_calibration_sha256": support.sha256,
            "selective_calibration_sha256": selective.sha256,
            "support_calibration_receipt_sha256": sha256_file(
                SUPPORT_CALIBRATION_RECEIPT
            ),
            "margin_calibration_receipt_sha256": sha256_file(
                MARGIN_CALIBRATION_RECEIPT
            ),
            "source_identity_sha256": verify_source_identity(),
            "selections": [
                {**asdict(selection), "reasons": list(selection.reasons)}
                for selection in selections
            ],
        }
    )
    artifact = EXPERIMENT / "offline-screen.json"
    artifact_sha = write_once(artifact, report)
    write_once(
        EXPERIMENT / "offline-screen.receipt.json",
        {
            "schema": "irisu-r3h-g2-offline-screen-receipt-v1",
            "artifact": str(artifact),
            "artifact_sha256": artifact_sha,
            "checkpoint_sha256": checkpoint_sha,
            "dataset_sha256": dataset.sha256,
            "collection_identity_sha256": report[
                "collection_identity_sha256"
            ],
            "support_calibration_receipt_sha256": sha256_file(
                SUPPORT_CALIBRATION_RECEIPT
            ),
            "margin_calibration_receipt_sha256": sha256_file(
                MARGIN_CALIBRATION_RECEIPT
            ),
            "passed": report["passed"],
            "source_identity_sha256": verify_source_identity(),
        },
    )
    verify_source_identity()
    print(json.dumps({**report, "artifact_sha256": artifact_sha}, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    subparsers = value.add_subparsers(dest="command", required=True)
    for name, function in (
        ("preflight", command_preflight),
        ("pilot", command_pilot),
        ("train", command_train),
        ("calibrate-support", command_calibrate_support),
        ("calibrate-margin", command_calibrate_margin),
        ("screen", command_screen),
    ):
        subparser = subparsers.add_parser(name)
        subparser.set_defaults(function=function)
    return value


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
