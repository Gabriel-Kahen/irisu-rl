"""No-follow, no-replace private artifact publication."""

from __future__ import annotations

import fcntl
import os
import re
import secrets
import stat
from pathlib import Path

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class PrivateArtifactError(OSError):
    """A private artifact cannot be created without weakening its boundary."""


def open_private_directory(path: Path) -> int:
    if path.is_symlink():
        raise PrivateArtifactError("private artifact directory must not be a symlink")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise PrivateArtifactError("cannot open private artifact directory") from exc
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o700
    ):
        os.close(descriptor)
        raise PrivateArtifactError(
            "private artifact directory must be owned by this user with mode 0700"
        )
    return descriptor


def _safe_name(name: str) -> str:
    if SAFE_NAME.fullmatch(name) is None or name in {".", ".."}:
        raise PrivateArtifactError("artifact name is unsafe")
    return name


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise PrivateArtifactError("artifact write made no progress")
        view = view[written:]


def publish_private_noreplace(directory: Path, name: str, payload: bytes) -> None:
    """Atomically publish one 0600 file beneath an already-private directory."""

    if not isinstance(payload, bytes):
        raise PrivateArtifactError("artifact payload must be bytes")
    target_name = _safe_name(name)
    directory_fd = open_private_directory(directory)
    descriptor = -1
    temporary_name = f"_r4b-{secrets.token_hex(16)}.tmp"
    try:
        descriptor = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            raise PrivateArtifactError("new artifact failed its ownership/mode check")
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.link(
            temporary_name,
            target_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        os.unlink(temporary_name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    except FileExistsError as exc:
        raise PrivateArtifactError("artifact already exists") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        os.close(directory_fd)


class PrivateJournal:
    """Single-writer durable journal created exclusively for one live run."""

    def __init__(self, directory: Path, name: str) -> None:
        self._directory_fd = open_private_directory(directory)
        self._closed = False
        self._descriptor = -1
        try:
            self._descriptor = os.open(
                _safe_name(name),
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC
                | os.O_APPEND,
                0o600,
                dir_fd=self._directory_fd,
            )
            metadata = os.fstat(self._descriptor)
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) != 0o600
                or metadata.st_nlink != 1
            ):
                raise PrivateArtifactError(
                    "new journal failed its ownership/mode check"
                )
            fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except FileExistsError as exc:
            os.close(self._directory_fd)
            raise PrivateArtifactError("journal already exists") from exc
        except Exception:
            if self._descriptor >= 0:
                os.close(self._descriptor)
            os.close(self._directory_fd)
            raise

    def append(self, payload: bytes) -> None:
        if self._closed:
            raise PrivateArtifactError("journal is closed")
        if not payload or not payload.endswith(b"\n"):
            raise PrivateArtifactError(
                "journal record must be a nonempty complete line"
            )
        _write_all(self._descriptor, payload)
        os.fsync(self._descriptor)

    def close(self) -> None:
        if not self._closed:
            os.fsync(self._descriptor)
            os.fsync(self._directory_fd)
            os.close(self._descriptor)
            os.close(self._directory_fd)
            self._closed = True

    def __enter__(self) -> PrivateJournal:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
