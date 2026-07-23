"""Immutable attestation for disposable original-game measurement runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SHA256 = re.compile(r"[0-9a-f]{64}")
REQUIRED_RUNTIME_FILES: Mapping[str, str] = {
    "irisu.exe": "game_executable_sha256",
    "data/dll/Box2D.dll": "box2d_sha256",
    "data/dll/DxLib.dll": "dxlib_sha256",
    "data/doc/irisu.ini": "game_config_sha256",
    "data/doc/irisu.dat": "config_data_sha256",
    "data/dat.dxa": "dat_dxa_sha256",
    "data/img.dxa": "img_dxa_sha256",
    "data/snd.dxa": "snd_dxa_sha256",
}
CANONICAL_RUNTIME_SHA256: Mapping[str, str] = {
    "game_executable_sha256": (
        "0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255"
    ),
    "box2d_sha256": (
        "34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd"
    ),
    "dxlib_sha256": (
        "d8ef638a078a8b4d24b53b174ca179623fed3690027d3f4acfe71a7d61c8b5c9"
    ),
    "game_config_sha256": (
        "1e29431fe8209c25784d4741f7972737561281169bbb5a56f62e3e0f0b63de35"
    ),
    "config_data_sha256": (
        "f30b36df3ec09d2ecefc71dbd07199c77c88d571c4c47e161d9b7a14a88c3234"
    ),
    "dat_dxa_sha256": (
        "b36ef6864bf2d0e626d5087edb5b571ef548ebd5dde9fbc9b87f7b4ac3e89d4a"
    ),
    "img_dxa_sha256": (
        "7ffdf24de7d9465296e14cbee086ed04927c5e8a7e442d6be597984a71e03c50"
    ),
    "snd_dxa_sha256": (
        "65617d2e2692bb5481e68005745cea8146a5021e78664b336325bd7ab2d4c51d"
    ),
}
CANONICAL_WINE_SHA256 = (
    "3d3ec18b80e54eb09477ab9022f69508dc36fc1e10cbc0d1b9fa7a5251cf270a"
)


class RuntimeAttestationError(ValueError):
    """A candidate run cannot safely be used for original-game measurement."""


def _sha256_descriptor(descriptor: int, label: str) -> str:
    digest = hashlib.sha256()
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeAttestationError(
            f"required runtime file is not a private regular file: {label}"
        )
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 1 << 20):
        digest.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise RuntimeAttestationError(
            f"cannot open required runtime file: {path.name}"
        ) from exc
    try:
        return _sha256_descriptor(descriptor, path.name)
    finally:
        os.close(descriptor)


def _marker(path: Path) -> tuple[dict[str, str], str]:
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise RuntimeAttestationError("missing disposable-run marker") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeAttestationError(
                "disposable-run marker is not a private regular file"
            )
        payload = os.read(descriptor, 8193)
        if len(payload) > 8192 or os.read(descriptor, 1):
            raise RuntimeAttestationError("disposable-run marker is too large")
    finally:
        os.close(descriptor)
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise RuntimeAttestationError("disposable-run marker is malformed") from exc
    result: dict[str, str] = {}
    for line in lines:
        key, separator, value = line.partition("=")
        if not separator or not key or key in result or "\0" in value:
            raise RuntimeAttestationError("disposable-run marker is malformed")
        result[key] = value
    allowed = {"created_by", "created_utc", "experiment_id", "irisu_exe_sha256"}
    if not {"created_by", "created_utc", "irisu_exe_sha256"} <= set(result) <= allowed:
        raise RuntimeAttestationError("disposable-run marker fields disagree")
    if result["created_by"] not in {
        "tools/create-reference-run.sh",
        "tools/prepare-reference-capture.py",
    }:
        raise RuntimeAttestationError("disposable run has an unknown creator")
    if SHA256.fullmatch(result["irisu_exe_sha256"]) is None:
        raise RuntimeAttestationError("marker executable SHA-256 is malformed")
    return result, hashlib.sha256(payload).hexdigest()


def _within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


@dataclass(frozen=True, slots=True)
class DisposableRunAttestation:
    experiment_id: str
    marker_sha256: str
    runtime_sha256: Mapping[str, str]

    @property
    def provenance(self) -> dict[str, str]:
        return {
            key: self.runtime_sha256[key]
            for key in (
                "game_executable_sha256",
                "box2d_sha256",
                "dxlib_sha256",
                "game_config_sha256",
            )
        }

    @property
    def runtime_identity_sha256(self) -> str:
        payload = {
            "experiment_id": self.experiment_id,
            "marker_sha256": self.marker_sha256,
            "runtime_sha256": dict(self.runtime_sha256),
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest()


def attest_disposable_run(
    repo_root: Path,
    run_dir: Path,
    *,
    expected_experiment_id: str,
    canonical_runtime_sha256: Mapping[str, str] = CANONICAL_RUNTIME_SHA256,
) -> DisposableRunAttestation:
    """Hash a no-link run under ``reference/runs`` without trusting its path text."""

    if IDENTIFIER.fullmatch(expected_experiment_id) is None:
        raise RuntimeAttestationError("experiment ID is unsafe")
    root = repo_root.resolve(strict=True)
    runs_root = (root / "reference" / "runs").resolve(strict=True)
    if run_dir.is_symlink():
        raise RuntimeAttestationError("run directory must not be a symlink")
    try:
        run = run_dir.resolve(strict=True)
    except OSError as exc:
        raise RuntimeAttestationError("run directory does not exist") from exc
    if (
        not run.is_dir()
        or not _within(run, runs_root)
        or run.parent != runs_root
        or run.name != expected_experiment_id
    ):
        raise RuntimeAttestationError(
            "measurement run must be the exact named child of reference/runs"
        )
    preserved = (root / "reference" / "game").resolve(strict=False)
    if _within(run, preserved):
        raise RuntimeAttestationError(
            "preserved reference game is never a measurement run"
        )
    for directory, names, files in os.walk(run, followlinks=False):
        parent = Path(directory)
        for name in (*names, *files):
            if (parent / name).is_symlink():
                raise RuntimeAttestationError("disposable run contains a symlink")

    marker_path = run / ".irisu-reference-run"
    marker, marker_sha256 = _marker(marker_path)
    marker_experiment = marker.get("experiment_id")
    if marker_experiment is not None and marker_experiment != expected_experiment_id:
        raise RuntimeAttestationError("marker experiment ID disagrees")
    if set(canonical_runtime_sha256) != set(CANONICAL_RUNTIME_SHA256):
        raise RuntimeAttestationError("canonical runtime profile fields disagree")
    hashes: dict[str, str] = {}
    for relative, key in REQUIRED_RUNTIME_FILES.items():
        path = run / relative
        if not path.is_file():
            raise RuntimeAttestationError(f"missing required runtime file: {relative}")
        hashes[key] = _sha256(path)
        if hashes[key] != canonical_runtime_sha256[key]:
            raise RuntimeAttestationError(
                f"required runtime file is not the canonical v2.03 target: {relative}"
            )
    if hashes["game_executable_sha256"] != marker["irisu_exe_sha256"]:
        raise RuntimeAttestationError("run executable changed after preparation")
    return DisposableRunAttestation(
        expected_experiment_id,
        marker_sha256,
        hashes,
    )


def attest_wine_runtime(path: Path) -> str:
    """Require the exact preregistered Wine executable used by the R4 lab."""

    observed = _sha256(path)
    if observed != CANONICAL_WINE_SHA256:
        raise RuntimeAttestationError("Wine runtime SHA-256 is not preregistered")
    return observed


def attest_wine_runtime_descriptor(descriptor: int) -> str:
    """Attest the already-open Wine inode that the launcher will execute."""

    metadata = os.fstat(descriptor)
    if (
        metadata.st_uid not in {0, os.getuid()}
        or metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
        or not metadata.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    ):
        raise RuntimeAttestationError(
            "Wine executable ownership or permissions are unsafe"
        )
    observed = _sha256_descriptor(descriptor, "Wine executable")
    if observed != CANONICAL_WINE_SHA256:
        raise RuntimeAttestationError("Wine runtime SHA-256 is not preregistered")
    return observed


def verify_attestation_unchanged(
    expected: DisposableRunAttestation,
    repo_root: Path,
    run_dir: Path,
) -> None:
    observed = attest_disposable_run(
        repo_root,
        run_dir,
        expected_experiment_id=expected.experiment_id,
        canonical_runtime_sha256=expected.runtime_sha256,
    )
    if observed != expected:
        raise RuntimeAttestationError("disposable runtime changed during measurement")
