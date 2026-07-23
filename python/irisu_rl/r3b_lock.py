"""Exclusive process ownership for one operational R3 run."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path
import stat


class R3BRunLock:
    """Hold a no-follow advisory lock for an entire operator command."""

    def __init__(self, run_directory: str | Path) -> None:
        self.root = Path(run_directory).resolve(strict=True)
        self.path = self.root / "operator.lock"
        self._descriptor: int | None = None

    def __enter__(self) -> R3BRunLock:
        descriptor = os.open(
            self.path,
            os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or stat.S_IMODE(metadata.st_mode) != 0o600
            or metadata.st_nlink != 1
        ):
            os.close(descriptor)
            raise ValueError("R3 operator lock metadata is unsafe")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise RuntimeError("another process is operating this R3 run") from exc
        self._descriptor = descriptor
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._descriptor is not None:
            fcntl.flock(self._descriptor, fcntl.LOCK_UN)
            os.close(self._descriptor)
            self._descriptor = None


__all__ = ["R3BRunLock"]
