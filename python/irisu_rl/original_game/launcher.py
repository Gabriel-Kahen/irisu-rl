"""Exclusive launcher for one disposable R4 original-game process."""

from __future__ import annotations

import fcntl
import hashlib
import os
import secrets
import signal
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .runtime import (
    DisposableRunAttestation,
    attest_disposable_run,
    attest_wine_runtime_descriptor,
    verify_attestation_unchanged,
)


class LaunchError(RuntimeError):
    """The disposable original game cannot be launched exclusively and safely."""


def _open_global_lock(runs_root: Path) -> int:
    try:
        root_fd = os.open(runs_root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as exc:
        raise LaunchError("cannot open the reference-runs directory") from exc
    try:
        descriptor = os.open(
            ".r4b-measurement.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC,
            0o600,
            dir_fd=root_fd,
        )
    finally:
        os.close(root_fd)
    metadata = os.fstat(descriptor)
    if (
        not stat.S_ISREG(metadata.st_mode)
        or metadata.st_uid != os.getuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
    ):
        os.close(descriptor)
        raise LaunchError("measurement lock ownership or mode is unsafe")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise LaunchError("another R4 original-game measurement owns the lock") from exc
    return descriptor


@dataclass(frozen=True, slots=True)
class LaunchAttestation:
    run: DisposableRunAttestation
    wine_executable_sha256: str
    launch_nonce_sha256: str
    launcher_process_id: int
    launcher_process_start_ticks: int


class MeasurementProcess:
    """One exclusively locked process; never targets the shared wineserver."""

    def __init__(
        self,
        *,
        repo_root: Path,
        run_dir: Path,
        experiment_id: str,
        wine_executable: Path,
        wine_prefix: Path,
        session_environment: Mapping[str, str] | None = None,
    ) -> None:
        self._root = repo_root.resolve(strict=True)
        self._run_dir = run_dir
        self._attestation = attest_disposable_run(
            self._root, run_dir, expected_experiment_id=experiment_id
        )
        if not wine_executable.is_absolute():
            raise LaunchError("Wine executable must be an absolute path")
        if not wine_prefix.is_absolute() or wine_prefix.is_symlink():
            raise LaunchError("Wine prefix must be an absolute non-symlink directory")
        try:
            self._wine_prefix = wine_prefix.resolve(strict=True)
        except OSError as exc:
            raise LaunchError("Wine prefix does not exist") from exc
        if not self._wine_prefix.is_dir():
            raise LaunchError("Wine prefix is not a directory")
        try:
            self._wine_descriptor = os.open(
                wine_executable, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
        except OSError as exc:
            raise LaunchError("cannot open the Wine executable") from exc
        try:
            self._wine_sha256 = attest_wine_runtime_descriptor(self._wine_descriptor)
            self._lock = _open_global_lock(self._root / "reference" / "runs")
        except Exception:
            os.close(self._wine_descriptor)
            self._wine_descriptor = -1
            raise
        self._process: subprocess.Popen[bytes] | None = None
        self._process_start_ticks = 0
        self._nonce = secrets.token_hex(32)
        allowed = {
            "DISPLAY",
            "WAYLAND_DISPLAY",
            "XDG_RUNTIME_DIR",
            "HYPRLAND_INSTANCE_SIGNATURE",
            "LANG",
        }
        source = os.environ if session_environment is None else session_environment
        self._environment = {
            key: value for key, value in source.items() if key in allowed
        }
        self._environment.update(
            {
                "WINEPREFIX": os.fspath(self._wine_prefix),
                "WINEDEBUG": "-all",
                "WINEDLLOVERRIDES": "mscoree,mshtml=",
                "IRISU_R4B_LAUNCH_NONCE": self._nonce,
            }
        )

    @property
    def attestation(self) -> LaunchAttestation:
        if self._process is None:
            raise LaunchError("measurement process has not started")
        return LaunchAttestation(
            self._attestation,
            self._wine_sha256,
            hashlib.sha256(self._nonce.encode("ascii")).hexdigest(),
            self._process.pid,
            self._process_start_ticks,
        )

    @property
    def launch_nonce(self) -> str:
        """Secret discovery nonce; callers must not persist or log it."""

        return self._nonce

    def start(self) -> MeasurementProcess:
        if self._process is not None:
            raise LaunchError("measurement process was already started")
        try:
            self._process = subprocess.Popen(
                [f"/proc/self/fd/{self._wine_descriptor}", "irisu.exe"],
                cwd=self._run_dir,
                env=self._environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                close_fds=True,
                pass_fds=(self._wine_descriptor,),
            )
        except OSError as exc:
            self.close()
            raise LaunchError("cannot launch the disposable original game") from exc
        try:
            stat_line = Path(f"/proc/{self._process.pid}/stat").read_text(
                encoding="ascii"
            )
            stat_fields = stat_line[stat_line.rfind(")") + 2 :].split()
            self._process_start_ticks = int(stat_fields[19])
        except (OSError, ValueError, IndexError) as exc:
            self.close()
            raise LaunchError("cannot attest launched process generation") from exc
        return self

    def close(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Kill only the process we launched.  Never signal a shared
                # wineserver or enumerate unrelated Wine processes.
                process.kill()
                process.wait(timeout=5)
        self._process = None
        try:
            verify_attestation_unchanged(self._attestation, self._root, self._run_dir)
        finally:
            if self._lock >= 0:
                os.close(self._lock)
                self._lock = -1
            if self._wine_descriptor >= 0:
                os.close(self._wine_descriptor)
                self._wine_descriptor = -1
            self._nonce = ""

    def __enter__(self) -> MeasurementProcess:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.close()
