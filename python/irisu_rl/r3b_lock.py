"""Exclusive process ownership for one operational R3 run."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import stat

from irisu_env.exact_ipc import (
    clear_exact_worker_lease,
    register_exact_worker_lease,
)


def _acquire(path: Path, operation: int = fcntl.LOCK_EX) -> int:
    descriptor = os.open(
        path,
        os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            raise ValueError("R3 operator lock metadata is unsafe")
        try:
            fcntl.flock(descriptor, operation | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process is operating an R3 run") from exc
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def evaluator_lease_path(global_directory: str | Path | None = None) -> Path:
    """Return the host-wide evaluator lease shared by operators and workers."""

    lock_root = Path("/tmp" if global_directory is None else global_directory).resolve(
        strict=True
    )
    if not lock_root.is_dir():
        raise ValueError("R3 global lock directory is not a directory")
    return lock_root / f".irisu-r3b-evaluator-{os.geteuid()}.lock"


def hold_evaluator_lease(path: str | Path) -> int:
    """Hold a shared worker lease until the returned descriptor is closed."""

    return _acquire(Path(path), fcntl.LOCK_SH)


class R3BRunLock:
    """Hold a no-follow advisory lock for an entire operator command."""

    def __init__(
        self,
        run_directory: str | Path,
        *,
        global_directory: str | Path | None = None,
    ) -> None:
        self.root = Path(run_directory).resolve(strict=True)
        self.path = self.root / "operator.lock"
        lock_root = Path(
            "/tmp" if global_directory is None else global_directory
        ).resolve(strict=True)
        if not lock_root.is_dir():
            raise ValueError("R3 global lock directory is not a directory")
        self.global_path = lock_root / f".irisu-r3b-operator-{os.geteuid()}.lock"
        self.evaluator_path = evaluator_lease_path(lock_root)
        self._descriptors: list[int] = []
        self._lease_descriptor: int | None = None

    def __enter__(self) -> R3BRunLock:
        if self._descriptors or self._lease_descriptor is not None:
            raise RuntimeError("R3 operator lock instance is already held")
        lease_guard: int | None = None
        try:
            self._descriptors.append(_acquire(self.global_path))
            lease_guard = _acquire(self.evaluator_path)
            self._descriptors.append(_acquire(self.path))
            register_exact_worker_lease(lease_guard, self.evaluator_path)
            self._lease_descriptor = lease_guard
            lease_guard = None
            fcntl.flock(self._lease_descriptor, fcntl.LOCK_SH)
        except BaseException:
            self.__exit__()
            raise
        finally:
            if lease_guard is not None:
                fcntl.flock(lease_guard, fcntl.LOCK_UN)
                os.close(lease_guard)
        return self

    def __exit__(self, *exc_info: object) -> None:
        cleanup_error: BaseException | None = None
        if self._lease_descriptor is not None:
            descriptor = self._lease_descriptor
            self._lease_descriptor = None
            try:
                clear_exact_worker_lease(descriptor)
            except BaseException as exc:
                cleanup_error = exc
            finally:
                # Do not explicitly unlock: an exact worker may hold an
                # inherited duplicate of this same open file description.
                os.close(descriptor)
        while self._descriptors:
            descriptor = self._descriptors.pop()
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        if cleanup_error is not None:
            if exc_info and isinstance(exc_info[1], BaseException):
                exc_info[1].add_note(
                    f"exact-worker lease cleanup also failed: {cleanup_error}"
                )
            else:
                raise cleanup_error


__all__ = ["R3BRunLock", "evaluator_lease_path", "hold_evaluator_lease"]
