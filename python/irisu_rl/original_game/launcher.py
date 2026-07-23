"""Exclusive launcher for one disposable R4 original-game process."""

from __future__ import annotations

import fcntl
import hashlib
import os
import secrets
import select
import signal
import stat
import subprocess
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from .runtime import (
    DisposableRunAttestation,
    attest_disposable_run,
    attest_wine_prefix,
    attest_wine_runtime_descriptor,
    verify_attestation_unchanged,
    verify_wine_prefix_unchanged,
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
    wine_prefix_sha256: str
    launch_nonce_sha256: str
    launcher_process_id: int
    launcher_process_start_ticks: int


@dataclass(frozen=True, slots=True)
class TargetLifecycleBinding:
    """Exact discovered process generation owned by this launch session."""

    process_id: int
    process_start_ticks: int
    session_id: int


def _process_stat(process_id: int) -> tuple[int, int]:
    try:
        stat_line = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
        fields = stat_line[stat_line.rfind(")") + 2 :].split()
        return int(fields[3]), int(fields[19])
    except (OSError, ValueError, IndexError) as exc:
        raise LaunchError("cannot attest process generation and session") from exc


def _process_has_launch_nonce(process_id: int, nonce: str) -> bool:
    expected = f"IRISU_R4B_LAUNCH_NONCE={nonce}".encode("ascii")
    try:
        descriptor = os.open(
            f"/proc/{process_id}/environ",
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise LaunchError("cannot inspect target launch environment") from exc
    try:
        payload = os.read(descriptor, 1_048_577)
        if len(payload) > 1_048_576 or os.read(descriptor, 1):
            raise LaunchError("target launch environment exceeds the size limit")
    finally:
        os.close(descriptor)
    return expected in payload.split(b"\0")


def _signal_pidfd(descriptor: int, sig: signal.Signals) -> None:
    with suppress(ProcessLookupError):
        signal.pidfd_send_signal(descriptor, sig)


def _wait_pidfd(descriptor: int, timeout_seconds: int) -> bool:
    poller = select.poll()
    poller.register(descriptor, select.POLLIN)
    return bool(poller.poll(timeout_seconds * 1000))


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
        try:
            self._wine_prefix_attestation = attest_wine_prefix(wine_prefix)
        except Exception as exc:
            raise LaunchError("Wine prefix attestation failed") from exc
        self._wine_prefix = wine_prefix
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
        self._target: TargetLifecycleBinding | None = None
        self._target_pidfd = -1
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
            self._wine_prefix_attestation.sha256,
            hashlib.sha256(self._nonce.encode("ascii")).hexdigest(),
            self._process.pid,
            self._process_start_ticks,
        )

    @property
    def launch_nonce(self) -> str:
        """Secret discovery nonce; callers must not persist or log it."""

        return self._nonce

    @property
    def target_binding(self) -> TargetLifecycleBinding | None:
        return self._target

    def verify_runtime_unchanged(self) -> None:
        """Re-attest the immutable baseline immediately around live operations."""

        verify_attestation_unchanged(self._attestation, self._root, self._run_dir)

    def verify_input_environment_unchanged(self) -> None:
        """Re-attest the Wine prefix immediately around input."""

        verify_wine_prefix_unchanged(
            self._wine_prefix_attestation,
            self._wine_prefix,
        )

    def bind_target(
        self, process_id: int, process_start_ticks: int
    ) -> TargetLifecycleBinding:
        """Bind broker discovery to this nonce and newly-created process session."""

        process = self._process
        if process is None or process.poll() is not None:
            raise LaunchError("measurement process is not running")
        if self._target is not None:
            raise LaunchError("measurement target was already bound")
        if type(process_id) is not int or process_id <= 0:
            raise LaunchError("target process ID must be positive")
        if type(process_start_ticks) is not int or process_start_ticks <= 0:
            raise LaunchError("target process start ticks must be positive")
        self.verify_runtime_unchanged()
        try:
            descriptor = os.pidfd_open(process_id, 0)
        except OSError as exc:
            raise LaunchError("cannot pin the discovered target process") from exc
        try:
            session_id, observed_start_ticks = _process_stat(process_id)
            if (
                observed_start_ticks != process_start_ticks
                or session_id != process.pid
                or not _process_has_launch_nonce(process_id, self._nonce)
            ):
                raise LaunchError(
                    "discovered target is not owned by this exact launch session"
                )
            # Close the stat/environ race: the pidfd pins the process, and a
            # second generation read must still name the same process.
            if _process_stat(process_id) != (session_id, observed_start_ticks):
                raise LaunchError("target process changed while it was being bound")
        except Exception:
            os.close(descriptor)
            raise
        binding = TargetLifecycleBinding(process_id, observed_start_ticks, session_id)
        self._target = binding
        self._target_pidfd = descriptor
        return binding

    def verify_target_binding(
        self, process_id: int, process_start_ticks: int
    ) -> None:
        """Prove the exact nonce-bound target generation is still alive."""

        target = self._target
        if (
            target is None
            or self._target_pidfd < 0
            or target.process_id != process_id
            or target.process_start_ticks != process_start_ticks
            or _wait_pidfd(self._target_pidfd, 0)
        ):
            raise LaunchError("live target is not the bound process generation")
        session_id, observed_start_ticks = _process_stat(process_id)
        if (
            session_id != target.session_id
            or observed_start_ticks != target.process_start_ticks
            or not _process_has_launch_nonce(process_id, self._nonce)
        ):
            raise LaunchError("live target binding changed")

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
            session_id, self._process_start_ticks = _process_stat(self._process.pid)
            if session_id != self._process.pid:
                raise LaunchError("launcher did not create an isolated process session")
        except LaunchError as exc:
            self.close()
            raise LaunchError("cannot attest launched process generation") from exc
        return self

    def close(self) -> None:
        process = self._process
        errors: list[str] = []
        try:
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"cannot stop launcher process: {exc}")
        try:
            if (
                self._target_pidfd >= 0
                and self._target is not None
                and not _wait_pidfd(self._target_pidfd, 0)
            ):
                _signal_pidfd(self._target_pidfd, signal.SIGTERM)
                if not _wait_pidfd(self._target_pidfd, 5):
                    _signal_pidfd(self._target_pidfd, signal.SIGKILL)
                    if not _wait_pidfd(self._target_pidfd, 5):
                        raise LaunchError("bound target did not exit after SIGKILL")
        except (OSError, LaunchError) as exc:
            errors.append(f"cannot stop bound target: {exc}")
        self._process = None
        try:
            self.verify_runtime_unchanged()
            self.verify_input_environment_unchanged()
        except Exception as exc:
            errors.append(f"runtime changed during measurement: {exc}")
        finally:
            try:
                if self._target_pidfd >= 0:
                    os.close(self._target_pidfd)
            finally:
                self._target_pidfd = -1
                self._target = None
            if self._lock >= 0:
                os.close(self._lock)
                self._lock = -1
            if self._wine_descriptor >= 0:
                os.close(self._wine_descriptor)
                self._wine_descriptor = -1
            self._nonce = ""
        if errors:
            raise LaunchError("; ".join(errors))

    def __enter__(self) -> MeasurementProcess:
        return self.start()

    def __exit__(self, *args: object) -> None:
        self.close()
