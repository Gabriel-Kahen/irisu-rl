"""No-follow, no-replace private artifact publication."""

from __future__ import annotations

import fcntl
import os
import re
import secrets
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class PrivateArtifactError(OSError):
    """A private artifact cannot be created without weakening its boundary."""


def open_private_directory(path: Path) -> int:
    if not path.is_absolute():
        raise PrivateArtifactError("private artifact directory must be absolute")
    try:
        if path.resolve(strict=True) != path:
            raise PrivateArtifactError(
                "private artifact directory ancestry contains a symlink"
            )
    except OSError as exc:
        raise PrivateArtifactError(
            "private artifact directory does not exist"
        ) from exc
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


def _stable_file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _open_private_source(path: Path, maximum_bytes: int) -> tuple[int, int]:
    if not path.is_absolute() or type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise PrivateArtifactError(
            "private input path and maximum size must be absolute/positive"
        )
    try:
        if path.parent.resolve(strict=True) != path.parent:
            raise PrivateArtifactError("private input ancestry contains a symlink")
    except OSError as exc:
        raise PrivateArtifactError("private input directory does not exist") from exc
    directory_descriptor = open_private_directory(path.parent)
    try:
        descriptor = os.open(
            _safe_name(path.name),
            os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=directory_descriptor,
        )
    except Exception:
        os.close(directory_descriptor)
        raise
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
            or metadata.st_size > maximum_bytes
        ):
            raise PrivateArtifactError(
                "private input must be an owned 0600 single-link regular file "
                "within the size limit"
            )
        fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except Exception:
        os.close(descriptor)
        os.close(directory_descriptor)
        raise
    return descriptor, directory_descriptor


@contextmanager
def snapshot_private_files(
    sources: Mapping[str, tuple[Path, int]],
) -> Iterator[dict[str, Path]]:
    """Single-open private sources into one immutable, coherent temp snapshot."""

    if not sources:
        raise PrivateArtifactError("at least one private input is required")
    temporary = tempfile.TemporaryDirectory(prefix="irisu-r4b-input-")
    directory = Path(temporary.name)
    try:
        directory.chmod(0o700)
        directory_descriptor = open_private_directory(directory)
        snapshots: dict[str, Path] = {}
        try:
            for name, source in sources.items():
                target_name = _safe_name(name)
                if (
                    not isinstance(source, tuple)
                    or len(source) != 2
                    or not isinstance(source[0], Path)
                ):
                    raise PrivateArtifactError("private input specification is invalid")
                source_descriptor, source_directory = _open_private_source(*source)
                target_descriptor = -1
                try:
                    before = os.fstat(source_descriptor)
                    target_descriptor = os.open(
                        target_name,
                        os.O_WRONLY
                        | os.O_CREAT
                        | os.O_EXCL
                        | os.O_NOFOLLOW
                        | os.O_CLOEXEC,
                        0o600,
                        dir_fd=directory_descriptor,
                    )
                    copied = 0
                    while chunk := os.read(source_descriptor, 1 << 20):
                        copied += len(chunk)
                        if copied > source[1]:
                            raise PrivateArtifactError(
                                "private input exceeds the size limit"
                            )
                        _write_all(target_descriptor, chunk)
                    after = os.fstat(source_descriptor)
                    if copied != before.st_size or _stable_file_identity(
                        before
                    ) != _stable_file_identity(after):
                        raise PrivateArtifactError(
                            "private input changed while it was snapshotted"
                        )
                    os.fsync(target_descriptor)
                    snapshots[name] = directory / target_name
                finally:
                    if target_descriptor >= 0:
                        os.close(target_descriptor)
                    os.close(source_descriptor)
                    os.close(source_directory)
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        yield snapshots
    finally:
        temporary.cleanup()


def publish_private_noreplace(directory: Path, name: str, payload: bytes) -> None:
    """Atomically publish one 0600 file beneath an already-private directory."""

    if not isinstance(payload, bytes):
        raise PrivateArtifactError("artifact payload must be bytes")
    target_name = _safe_name(name)
    directory_fd = open_private_directory(directory)
    descriptor = -1
    temporary_name = f"_r4b-{secrets.token_hex(16)}.tmp"
    failure: BaseException | None = None
    cleanup_errors: list[OSError] = []
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
        failure = PrivateArtifactError("artifact already exists")
        failure.__cause__ = exc
    except BaseException as exc:
        failure = exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as exc:
                cleanup_errors.append(exc)
        try:
            os.unlink(temporary_name, dir_fd=directory_fd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            cleanup_errors.append(exc)
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            cleanup_errors.append(exc)
        try:
            os.close(directory_fd)
        except OSError as exc:
            cleanup_errors.append(exc)
    if failure is not None:
        raise failure
    if cleanup_errors:
        raise PrivateArtifactError("private artifact cleanup was not durable") from (
            cleanup_errors[0]
        )


def publish_private_bundle_noreplace(
    directory: Path,
    bundle_name: str,
    artifacts: Mapping[str, bytes],
) -> Path:
    """Crash-atomically publish a complete private artifact directory."""

    target_name = _safe_name(bundle_name)
    if not artifacts:
        raise PrivateArtifactError("artifact bundle must not be empty")
    payloads: dict[str, bytes] = {}
    for name, payload in artifacts.items():
        safe_name = _safe_name(name)
        if safe_name in payloads or not isinstance(payload, bytes):
            raise PrivateArtifactError("artifact bundle contents are invalid")
        payloads[safe_name] = payload

    parent_descriptor = open_private_directory(directory)
    temporary_name = f"_r4b-bundle-{secrets.token_hex(16)}.tmp"
    bundle_descriptor = -1
    created_files: list[str] = []
    published = False
    failure: BaseException | None = None
    cleanup_errors: list[OSError] = []
    try:
        try:
            fcntl.flock(parent_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PrivateArtifactError(
                "another artifact publication owns the output directory"
            ) from exc
        os.mkdir(temporary_name, 0o700, dir_fd=parent_descriptor)
        bundle_descriptor = os.open(
            temporary_name,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        metadata = os.fstat(bundle_descriptor)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.getuid()
            or stat.S_IMODE(metadata.st_mode) != 0o700
        ):
            raise PrivateArtifactError("new artifact bundle is not private")
        for name, payload in payloads.items():
            descriptor = os.open(
                name,
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | os.O_CLOEXEC,
                0o600,
                dir_fd=bundle_descriptor,
            )
            created_files.append(name)
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_uid != os.getuid()
                    or stat.S_IMODE(opened.st_mode) != 0o600
                    or opened.st_nlink != 1
                ):
                    raise PrivateArtifactError("new bundle artifact is not private")
                _write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        os.fsync(bundle_descriptor)
        try:
            os.stat(target_name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise PrivateArtifactError("artifact bundle already exists")
        os.rename(
            temporary_name,
            target_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        published = True
        os.fsync(parent_descriptor)
    except BaseException as exc:
        failure = exc
    finally:
        if not published and bundle_descriptor >= 0:
            for name in created_files:
                try:
                    os.unlink(name, dir_fd=bundle_descriptor)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    cleanup_errors.append(exc)
        if bundle_descriptor >= 0:
            try:
                os.close(bundle_descriptor)
            except OSError as exc:
                cleanup_errors.append(exc)
        if not published:
            try:
                os.rmdir(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
            except OSError as exc:
                cleanup_errors.append(exc)
        try:
            os.close(parent_descriptor)
        except OSError as exc:
            cleanup_errors.append(exc)
    if failure is not None:
        raise failure
    if cleanup_errors:
        raise PrivateArtifactError("private bundle cleanup was incomplete") from (
            cleanup_errors[0]
        )
    return directory / target_name


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
            error: OSError | None = None
            try:
                os.fsync(self._descriptor)
                os.fsync(self._directory_fd)
            except OSError as exc:
                error = exc
            finally:
                try:
                    os.close(self._descriptor)
                finally:
                    os.close(self._directory_fd)
                    self._descriptor = -1
                    self._directory_fd = -1
                    self._closed = True
            if error is not None:
                raise PrivateArtifactError(
                    "cannot durably close private journal"
                ) from error

    def __enter__(self) -> PrivateJournal:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
