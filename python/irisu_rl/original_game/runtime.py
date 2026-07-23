"""Immutable attestation for disposable original-game measurement runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

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
MUTABLE_RUNTIME_FILES = frozenset({"photo.png", "replay/new.rpy", "save.dat"})
ALLOWED_WINDOWS_MODULES = frozenset(
    {"irisu.exe", "data/dll/Box2D.dll", "data/dll/DxLib.dll"}
)
WINDOWS_MODULE_SUFFIXES = frozenset(
    {".acm", ".cpl", ".dll", ".drv", ".exe", ".ocx", ".sys"}
)
RUNTIME_IDENTITY_SCHEMA = "r4b-runtime-identity-v2"
WINE_PREFIX_SCHEMA = "r4b-wine-prefix-v1"
ALLOWED_PREFIX_SYMLINK_ROOTS = ("dosdevices/", "drive_c/users/")


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


def _relative_name(prefix: str, name: str) -> str:
    return f"{prefix}/{name}" if prefix else name


def _validate_runtime_metadata(
    metadata: os.stat_result,
    relative: str,
    *,
    directory: bool,
) -> None:
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(metadata.st_mode) or (
        not directory and metadata.st_nlink != 1
    ):
        kind = "private directory" if directory else "private regular file"
        raise RuntimeAttestationError(f"runtime entry is not a {kind}: {relative}")
    if metadata.st_uid != os.getuid() or metadata.st_mode & (
        stat.S_IWGRP | stat.S_IWOTH
    ):
        raise RuntimeAttestationError(
            f"runtime entry ownership or permissions are unsafe: {relative}"
        )


def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _entry_identity(
    relative: str, metadata: os.stat_result, *, directory: bool
) -> tuple[str, str, int, int, int, int, int, int, int]:
    return (
        relative,
        "directory" if directory else "file",
        metadata.st_dev,
        metadata.st_ino,
        0 if directory else metadata.st_size,
        0 if directory else metadata.st_mtime_ns,
        0 if directory else metadata.st_ctime_ns,
        metadata.st_mode,
        metadata.st_uid,
    )


def _scan_runtime_tree(
    run: Path,
    *,
    expected_identity: tuple[
        tuple[str, str, int, int, int, int, int, int, int], ...
    ]
    | None = None,
    expected_hashes: tuple[tuple[str, str], ...] | None = None,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str, int, int, int, int, int, int, int], ...],
]:
    """Snapshot immutable files through descriptor-relative no-follow traversal.

    Revalidation reuses a prior digest only when the file's inode, size,
    timestamps, mode, and owner are unchanged.  Linux ctime is not
    user-restorable, so this preserves fail-closed mutation detection without
    rehashing the packed game assets before every observation and input.
    """

    try:
        root_descriptor = os.open(
            run, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
    except OSError as exc:
        raise RuntimeAttestationError(
            "cannot open disposable runtime directory"
        ) from exc
    root_metadata = os.fstat(root_descriptor)
    _validate_runtime_metadata(root_metadata, ".", directory=True)
    baseline: list[tuple[str, str]] = []
    identities = [_entry_identity(".", root_metadata, directory=True)]
    reusable_hashes = dict(expected_hashes or ())
    reusable_identities = {
        identity[0]: identity for identity in (expected_identity or ())
    }

    def visit(directory_descriptor: int, prefix: str) -> None:
        try:
            entries = sorted(
                os.scandir(directory_descriptor), key=lambda item: item.name
            )
        except OSError as exc:
            raise RuntimeAttestationError(
                "cannot enumerate disposable runtime"
            ) from exc
        for entry in entries:
            relative = _relative_name(prefix, entry.name)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeAttestationError(
                    f"cannot inspect runtime entry: {relative}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise RuntimeAttestationError("disposable run contains a symlink")
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    child_descriptor = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=directory_descriptor,
                    )
                except OSError as exc:
                    raise RuntimeAttestationError(
                        f"cannot open runtime directory: {relative}"
                    ) from exc
                try:
                    opened = os.fstat(child_descriptor)
                    if not _same_inode(metadata, opened):
                        raise RuntimeAttestationError(
                            f"runtime directory changed while opening: {relative}"
                        )
                    _validate_runtime_metadata(opened, relative, directory=True)
                    identities.append(
                        _entry_identity(relative, opened, directory=True)
                    )
                    visit(child_descriptor, relative)
                finally:
                    os.close(child_descriptor)
                continue
            if (
                Path(relative).suffix.casefold() in WINDOWS_MODULE_SUFFIXES
                and relative not in ALLOWED_WINDOWS_MODULES
            ):
                raise RuntimeAttestationError(
                    f"disposable run contains an unapproved Windows module: {relative}"
                )
            try:
                file_descriptor = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise RuntimeAttestationError(
                    f"cannot open runtime file: {relative}"
                ) from exc
            try:
                opened = os.fstat(file_descriptor)
                if not _same_inode(metadata, opened):
                    raise RuntimeAttestationError(
                        f"runtime file changed while opening: {relative}"
                    )
                _validate_runtime_metadata(opened, relative, directory=False)
                identity = _entry_identity(relative, opened, directory=False)
                if identity == reusable_identities.get(relative):
                    try:
                        digest = reusable_hashes[relative]
                    except KeyError as exc:
                        raise RuntimeAttestationError(
                            f"runtime digest baseline is incomplete: {relative}"
                        ) from exc
                else:
                    digest = _sha256_descriptor(file_descriptor, relative)
            finally:
                os.close(file_descriptor)
            if relative not in MUTABLE_RUNTIME_FILES:
                identities.append(identity)
                baseline.append((relative, digest))

    try:
        visit(root_descriptor, "")
    finally:
        os.close(root_descriptor)
    result_identity = tuple(identities)
    if expected_identity is not None and result_identity != expected_identity:
        raise RuntimeAttestationError("disposable runtime tree identity changed")
    return tuple(baseline), result_identity


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
    immutable_file_sha256: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    immutable_file_identity: tuple[
        tuple[str, str, int, int, int, int, int, int, int], ...
    ] = field(default_factory=tuple, repr=False)

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
            "schema": RUNTIME_IDENTITY_SCHEMA,
            "experiment_id": self.experiment_id,
            "marker_sha256": self.marker_sha256,
            "runtime_sha256": dict(self.runtime_sha256),
            "immutable_file_sha256": dict(self.immutable_file_sha256),
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


@dataclass(frozen=True, slots=True)
class WinePrefixAttestation:
    sha256: str
    entries: tuple[tuple[str, str, str], ...] = field(repr=False)
    identity: tuple[
        tuple[str, str, int, int, int, int, int, int, int], ...
    ] = field(repr=False)


def _prefix_entry_identity(
    relative: str,
    kind: str,
    metadata: os.stat_result,
) -> tuple[str, str, int, int, int, int, int, int, int]:
    return (
        relative,
        kind,
        metadata.st_dev,
        metadata.st_ino,
        0 if kind == "directory" else metadata.st_size,
        0 if kind == "directory" else metadata.st_mtime_ns,
        0 if kind == "directory" else metadata.st_ctime_ns,
        metadata.st_mode,
        metadata.st_uid,
    )


def _scan_wine_prefix(
    prefix: Path,
    *,
    expected: WinePrefixAttestation | None = None,
) -> WinePrefixAttestation:
    try:
        root_descriptor = os.open(
            prefix,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise RuntimeAttestationError("cannot open Wine prefix") from exc
    entries: list[tuple[str, str, str]] = []
    identities: list[
        tuple[str, str, int, int, int, int, int, int, int]
    ] = []
    expected_entries = {
        (path, kind): value
        for path, kind, value in (expected.entries if expected else ())
    }
    expected_identities = {
        identity[0]: identity for identity in (expected.identity if expected else ())
    }

    def validate_owned(metadata: os.stat_result, relative: str) -> None:
        if metadata.st_uid != os.getuid() or metadata.st_mode & (
            stat.S_IWGRP | stat.S_IWOTH
        ):
            raise RuntimeAttestationError(
                f"Wine prefix entry ownership or permissions are unsafe: {relative}"
            )

    def visit(directory_descriptor: int, relative_root: str) -> None:
        try:
            directory_entries = sorted(
                os.scandir(directory_descriptor), key=lambda item: item.name
            )
        except OSError as exc:
            raise RuntimeAttestationError("cannot enumerate Wine prefix") from exc
        for entry in directory_entries:
            relative = _relative_name(relative_root, entry.name)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise RuntimeAttestationError(
                    f"cannot inspect Wine prefix entry: {relative}"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                if metadata.st_uid != os.getuid():
                    raise RuntimeAttestationError(
                        f"Wine prefix symlink owner is unsafe: {relative}"
                    )
                if not relative.startswith(ALLOWED_PREFIX_SYMLINK_ROOTS):
                    raise RuntimeAttestationError(
                        f"Wine prefix contains an unsafe symlink: {relative}"
                    )
                try:
                    target = os.readlink(entry.name, dir_fd=directory_descriptor)
                except OSError as exc:
                    raise RuntimeAttestationError(
                        f"cannot read Wine prefix symlink: {relative}"
                    ) from exc
                entries.append((relative, "symlink", target))
                identities.append(
                    _prefix_entry_identity(relative, "symlink", metadata)
                )
                continue
            validate_owned(metadata, relative)
            if stat.S_ISDIR(metadata.st_mode):
                try:
                    child = os.open(
                        entry.name,
                        os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                        dir_fd=directory_descriptor,
                    )
                except OSError as exc:
                    raise RuntimeAttestationError(
                        f"cannot open Wine prefix directory: {relative}"
                    ) from exc
                try:
                    opened = os.fstat(child)
                    if not _same_inode(metadata, opened):
                        raise RuntimeAttestationError(
                            f"Wine prefix directory changed while opening: {relative}"
                        )
                    validate_owned(opened, relative)
                    identities.append(
                        _prefix_entry_identity(relative, "directory", opened)
                    )
                    entries.append((relative, "directory", ""))
                    visit(child, relative)
                finally:
                    os.close(child)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeAttestationError(
                    f"Wine prefix entry is not a private regular file: {relative}"
                )
            try:
                descriptor = os.open(
                    entry.name,
                    os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
                    dir_fd=directory_descriptor,
                )
            except OSError as exc:
                raise RuntimeAttestationError(
                    f"cannot open Wine prefix file: {relative}"
                ) from exc
            try:
                opened = os.fstat(descriptor)
                if not _same_inode(metadata, opened):
                    raise RuntimeAttestationError(
                        f"Wine prefix file changed while opening: {relative}"
                    )
                validate_owned(opened, relative)
                identity = _prefix_entry_identity(relative, "file", opened)
                if identity == expected_identities.get(relative):
                    try:
                        digest = expected_entries[(relative, "file")]
                    except KeyError as exc:
                        raise RuntimeAttestationError(
                            f"Wine prefix digest baseline is incomplete: {relative}"
                        ) from exc
                else:
                    digest = _sha256_descriptor(descriptor, relative)
            finally:
                os.close(descriptor)
            identities.append(identity)
            entries.append((relative, "file", digest))

    try:
        root_metadata = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise RuntimeAttestationError("Wine prefix root is not a directory")
        validate_owned(root_metadata, ".")
        identities.append(_prefix_entry_identity(".", "directory", root_metadata))
        visit(root_descriptor, "")
    finally:
        os.close(root_descriptor)
    result_entries = tuple(entries)
    result_identity = tuple(identities)
    if expected is not None and (
        result_entries != expected.entries or result_identity != expected.identity
    ):
        raise RuntimeAttestationError("Wine prefix changed during measurement")
    payload = {
        "schema": WINE_PREFIX_SCHEMA,
        "entries": result_entries,
    }
    return WinePrefixAttestation(
        hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
        ).hexdigest(),
        result_entries,
        result_identity,
    )


def attest_wine_prefix(path: Path) -> WinePrefixAttestation:
    """Bind every file and permitted standard Wine-prefix symlink."""

    if not path.is_absolute() or path.is_symlink():
        raise RuntimeAttestationError(
            "Wine prefix must be an absolute non-symlink directory"
        )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RuntimeAttestationError("Wine prefix does not exist") from exc
    if resolved != path or not resolved.is_dir():
        raise RuntimeAttestationError(
            "Wine prefix ancestry must not contain symlinks"
        )
    return _scan_wine_prefix(resolved)


def verify_wine_prefix_unchanged(
    expected: WinePrefixAttestation,
    path: Path,
) -> None:
    if not isinstance(expected, WinePrefixAttestation):
        raise TypeError("expected must be a WinePrefixAttestation")
    observed = _scan_wine_prefix(path, expected=expected)
    if observed != expected:
        raise RuntimeAttestationError("Wine prefix changed during measurement")


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
    immutable_baseline, immutable_identity = _scan_runtime_tree(run)
    immutable_hashes = dict(immutable_baseline)

    marker_path = run / ".irisu-reference-run"
    marker, marker_sha256 = _marker(marker_path)
    marker_experiment = marker.get("experiment_id")
    if marker_experiment is not None and marker_experiment != expected_experiment_id:
        raise RuntimeAttestationError("marker experiment ID disagrees")
    if set(canonical_runtime_sha256) != set(CANONICAL_RUNTIME_SHA256):
        raise RuntimeAttestationError("canonical runtime profile fields disagree")
    hashes: dict[str, str] = {}
    for relative, key in REQUIRED_RUNTIME_FILES.items():
        observed = immutable_hashes.get(relative)
        if observed is None:
            raise RuntimeAttestationError(f"missing required runtime file: {relative}")
        hashes[key] = observed
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
        immutable_baseline,
        immutable_identity,
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
    root = repo_root.resolve(strict=True)
    runs_root = (root / "reference" / "runs").resolve(strict=True)
    if run_dir.is_symlink():
        raise RuntimeAttestationError("run directory must not be a symlink")
    run = run_dir.resolve(strict=True)
    if (
        not run.is_dir()
        or not _within(run, runs_root)
        or run.parent != runs_root
        or run.name != expected.experiment_id
    ):
        raise RuntimeAttestationError(
            "measurement run must remain the exact named child of reference/runs"
        )
    if not expected.immutable_file_identity:
        observed = attest_disposable_run(
            root,
            run,
            expected_experiment_id=expected.experiment_id,
            canonical_runtime_sha256=expected.runtime_sha256,
        )
        if observed != expected:
            raise RuntimeAttestationError(
                "disposable runtime changed during measurement"
            )
        return
    immutable_baseline, immutable_identity = _scan_runtime_tree(
        run,
        expected_identity=expected.immutable_file_identity,
        expected_hashes=expected.immutable_file_sha256,
    )
    marker, marker_sha256 = _marker(run / ".irisu-reference-run")
    marker_experiment = marker.get("experiment_id")
    if marker_experiment is not None and marker_experiment != expected.experiment_id:
        raise RuntimeAttestationError("disposable-run experiment ID mismatch")
    observed = DisposableRunAttestation(
        expected.experiment_id,
        marker_sha256,
        {
            key: dict(immutable_baseline)[relative]
            for relative, key in REQUIRED_RUNTIME_FILES.items()
        },
        immutable_baseline,
        immutable_identity,
    )
    if observed != expected:
        raise RuntimeAttestationError("disposable runtime changed during measurement")
