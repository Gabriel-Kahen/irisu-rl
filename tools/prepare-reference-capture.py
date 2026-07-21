#!/usr/bin/env python3
"""Prepare, but never launch, a reproducible original-game capture bundle."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
HEADER = struct.Struct("<5i")
REQUIRED_RUNTIME_FILES: Mapping[str, str] = {
    "irisu.exe": "executable_sha256",
    "data/dll/Box2D.dll": "box2d_sha256",
    "data/dll/DxLib.dll": "dxlib_sha256",
    "data/doc/irisu.ini": "config_sha256",
    "data/doc/irisu.dat": "config_data_sha256",
    "data/dat.dxa": "dat_dxa_sha256",
    "data/img.dxa": "img_dxa_sha256",
    "data/snd.dxa": "snd_dxa_sha256",
}
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class PreparationError(ValueError):
    """The requested bundle cannot be prepared safely."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_replay(data: bytes, layout: str) -> dict[str, Any]:
    if len(data) < HEADER.size:
        raise PreparationError(f"replay is shorter than its {HEADER.size}-byte header")
    seed, highest_level, final_score, highest_chain, mode = HEADER.unpack_from(data)
    if mode not in (0, 1):
        raise PreparationError(f"unsupported replay mode {mode}; expected 0 or 1")
    if layout not in ("auto", "legacy", "padded"):
        raise PreparationError(f"unknown replay layout: {layout!r}")
    zero_block = len(data) >= 52 and data[20:52] == bytes(32)
    if layout == "auto" and zero_block:
        raise PreparationError(
            "replay layout is ambiguous: 32 zero bytes can be v2.03 padding or "
            "eight legacy input records; pass --layout legacy or --layout padded"
        )
    resolved_layout = "legacy" if layout == "auto" else layout
    frame_offset = 52 if resolved_layout == "padded" else 20
    if len(data) < frame_offset:
        raise PreparationError(
            f"{resolved_layout} layout requires at least {frame_offset} bytes"
        )
    if (len(data) - frame_offset) % 4:
        raise PreparationError("replay contains a partial four-byte input record")
    return {
        "seed": seed,
        "highest_level": highest_level,
        "header_final_score": final_score,
        "highest_chain": highest_chain,
        "mode": mode,
        "layout": resolved_layout,
        "layout_selection": "explicit" if layout != "auto" else "auto_unambiguous",
        "frame_offset_bytes": frame_offset,
        "frame_count": (len(data) - frame_offset) // 4,
    }


def find_symlink(root: Path) -> Path | None:
    if root.is_symlink():
        return root
    for directory, names, files in os.walk(root, followlinks=False):
        parent = Path(directory)
        for name in (*names, *files):
            candidate = parent / name
            if candidate.is_symlink():
                return candidate
    return None


def publish_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish one staged directory without replacing any target."""

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise PreparationError("atomic no-replace publication is unavailable") from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.EEXIST, errno.ENOTEMPTY):
        raise PreparationError(f"destination appeared during preparation: {destination}")
    raise OSError(error, os.strerror(error), destination)


def snapshot_files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare_capture(
    experiment_id: str,
    replay_path: Path,
    *,
    repo_root: Path = ROOT,
    source_dir: Path | None = None,
    runs_dir: Path | None = None,
    captures_dir: Path | None = None,
    now: datetime | None = None,
    layout: str = "auto",
) -> dict[str, Any]:
    if IDENTIFIER.fullmatch(experiment_id) is None:
        raise PreparationError(
            "experiment ID must start with an alphanumeric and contain only "
            "letters, numbers, dot, underscore, and hyphen (maximum 128 characters)"
        )

    repo_root = repo_root.resolve()
    source_path = source_dir or repo_root / "reference/game/irisu-v2.03-en"
    if source_path.is_symlink():
        raise PreparationError(f"source tree must not be a symlink: {source_path}")
    source_dir = source_path.resolve()
    runs_dir = (runs_dir or repo_root / "reference/runs").resolve()
    captures_dir = (captures_dir or repo_root / "reference/captures").resolve()
    launcher = repo_root / "tools/launch-reference-game.sh"
    try:
        replay_path = replay_path.resolve(strict=True)
    except OSError as exc:
        raise PreparationError(f"cannot read replay: {exc}") from exc
    if not replay_path.is_file():
        raise PreparationError(f"replay is not a regular file: {replay_path}")
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise PreparationError(f"missing executable reference launcher: {launcher}")

    try:
        replay_bytes = replay_path.read_bytes()
    except OSError as exc:
        raise PreparationError(f"cannot snapshot replay: {exc}") from exc
    replay_summary = inspect_replay(replay_bytes, layout)
    replay_hash = sha256_bytes(replay_bytes)
    prepared_at = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    timestamp = prepared_at.isoformat().replace("+00:00", "Z")
    run_dir = runs_dir / experiment_id
    capture_dir = captures_dir / experiment_id
    if run_dir.exists() or run_dir.is_symlink():
        raise PreparationError(f"run directory already exists: {run_dir}")
    if capture_dir.exists() or capture_dir.is_symlink():
        raise PreparationError(f"capture directory already exists: {capture_dir}")

    source_symlink = find_symlink(source_dir)
    if source_symlink is not None:
        raise PreparationError(
            "source tree contains a symlink and cannot be copied safely: "
            f"{source_symlink.relative_to(source_dir)}"
        )

    required = tuple(REQUIRED_RUNTIME_FILES) + ("save.dat", "replay/new.rpy")
    missing = [relative for relative in required if not (source_dir / relative).is_file()]
    if missing:
        raise PreparationError(
            f"source tree is missing required file(s): {', '.join(missing)}"
        )

    runs_dir.mkdir(parents=True, exist_ok=True)
    captures_dir.mkdir(parents=True, exist_ok=True)
    run_stage_parent = Path(tempfile.mkdtemp(prefix=f".{experiment_id}.", dir=runs_dir))
    capture_stage_parent = Path(
        tempfile.mkdtemp(prefix=f".{experiment_id}.", dir=captures_dir)
    )
    staged_run = run_stage_parent / "run"
    staged_capture = capture_stage_parent / "capture"
    try:
        # Dereference as a second line of defense if the source changes after
        # the preflight scan; no link into the preserved tree may reach a run.
        shutil.copytree(source_dir, staged_run, symlinks=False)
        staged_symlink = find_symlink(staged_run)
        if staged_symlink is not None:
            raise PreparationError(
                f"copied run unexpectedly contains a symlink: {staged_symlink}"
            )
        staged_capture.mkdir()
        (staged_capture / "frames").mkdir()

        adjustments: list[dict[str, str]] = []
        copied_launcher = staged_run / "launch-irisu.sh"
        if copied_launcher.exists():
            if not copied_launcher.is_file():
                raise PreparationError(
                    f"copied historical launcher is not a regular file: {copied_launcher}"
                )
            launcher_hash = sha256(copied_launcher)
            copied_launcher.unlink()
            adjustments.append(
                {
                    "path": "launch-irisu.sh",
                    "operation": "removed_from_disposable_copy",
                    "copied_file_sha256": launcher_hash,
                    "reason": (
                        "historical launcher targets a non-disposable game tree; "
                        "the workspace launcher is authoritative"
                    ),
                }
            )

        run_replay = staged_run / "replay" / f"{experiment_id}.rpy"
        if run_replay.exists() or run_replay.is_symlink():
            raise PreparationError(f"source tree already contains replay target: {run_replay}")
        bundle_replay = staged_capture / "input.rpy"
        run_replay.write_bytes(replay_bytes)
        bundle_replay.write_bytes(replay_bytes)
        for copied in (run_replay, bundle_replay):
            if copied.read_bytes() != replay_bytes:
                raise PreparationError(f"replay verification failed after copying to {copied}")

        runtime_hashes: dict[str, str] = {}
        for relative, key in REQUIRED_RUNTIME_FILES.items():
            source_hash = sha256(source_dir / relative)
            run_hash = sha256(staged_run / relative)
            if source_hash != run_hash:
                raise PreparationError(f"runtime verification failed after copying {relative}")
            runtime_hashes[key] = run_hash
        for relative in ("save.dat", "replay/new.rpy"):
            if sha256(source_dir / relative) != sha256(staged_run / relative):
                raise PreparationError(f"baseline verification failed after copying {relative}")

        (staged_run / ".irisu-reference-run").write_text(
            "created_by=tools/prepare-reference-capture.py\n"
            f"created_utc={timestamp}\n"
            f"experiment_id={experiment_id}\n"
            f"irisu_exe_sha256={runtime_hashes['executable_sha256']}\n",
            encoding="utf-8",
        )
        final_run_replay = run_dir / run_replay.relative_to(staged_run)
        final_bundle_replay = capture_dir / bundle_replay.relative_to(staged_capture)
        launch_command = (
            f"IRISU_GAME_DIR={shlex.quote(str(run_dir))} "
            f"{shlex.quote(str(launcher))}"
        )
        metadata = {
            "experiment_id": experiment_id,
            "date": prepared_at.date().isoformat(),
            "status": "prepared_non_golden",
            "status_reason": (
                "Preparation records provenance only; no original-game outcome has "
                "been observed or measured."
            ),
            "hypothesis": None,
            "expected_discriminating_outcomes": [],
            "prepared_at_utc": timestamp,
            "environment": {"capture_at_run_time": "pending"},
            "game": {
                "version": "v2.03 with English-patched data",
                "executable": str(run_dir / "irisu.exe"),
                **runtime_hashes,
                "runtime_file_sha256": {
                    relative: runtime_hashes[key]
                    for relative, key in REQUIRED_RUNTIME_FILES.items()
                },
            },
            "input_replay": {
                "preserved_source": str(replay_path),
                "disposable_copy": str(final_run_replay),
                "bundle_copy": str(final_bundle_replay),
                "sha256": replay_hash,
                "size_bytes": len(replay_bytes),
                "snapshot_read_at_utc": timestamp,
                "snapshot_provenance": (
                    "summary, hash, size, and both copies derive from one in-memory read"
                ),
                **replay_summary,
            },
            "run": {
                "path": str(run_dir),
                "source_tree": str(source_dir),
                "launch_command": launch_command,
                "initial_save_sha256": sha256(staged_run / "save.dat"),
                "initial_new_replay_sha256": sha256(staged_run / "replay/new.rpy"),
                "initial_tree_sha256": snapshot_files(staged_run),
                "source_tree_adjustments": adjustments,
                "generated_run_files": [".irisu-reference-run"],
                "final_save_sha256": None,
                "final_new_replay_sha256": None,
                "changed_game_tree_files": None,
            },
            "window": {"capture_at_run_time": "pending"},
            "input_policy": {"capture_at_run_time": "pending"},
            "capture": {
                "path": str(capture_dir),
                "golden_eligible": False,
                "completion_state": "pending_original_game_observation",
                "required_after_run": [
                    "result.rpy",
                    "measurements.json with observed results",
                    "exact-window frame(s)",
                    "final run-tree hashes",
                    "complete environment, window, timing, and action records",
                ],
            },
        }
        write_json(staged_capture / "metadata.json", metadata)
        actions: list[dict[str, Any]] = [
            {
                "realtime_utc": timestamp,
                "action": "create_disposable_run",
                "source": str(source_dir),
                "destination": str(run_dir),
                "result": "prepared but not launched",
            },
        ]
        if adjustments:
            actions.append(
                {
                    "realtime_utc": timestamp,
                    "action": "remove_copied_historical_launcher",
                    **adjustments[0],
                    "result": "preserved source untouched; disposable copy removed",
                }
            )
        actions.append(
            {
                "realtime_utc": timestamp,
                "action": "copy_input_replay",
                "source": str(replay_path),
                "disposable_copy": str(final_run_replay),
                "bundle_copy": str(final_bundle_replay),
                "sha256": replay_hash,
                "result": "both copies are byte-identical to the single source snapshot",
            }
        )
        for sequence, action in enumerate(actions, 1):
            action["monotonic_sequence"] = sequence
        (staged_capture / "actions.jsonl").write_text(
            "".join(json.dumps(action, sort_keys=True) + "\n" for action in actions),
            encoding="utf-8",
        )
        write_json(
            staged_capture / "measurements.json",
            {
                "status": "pending_original_game_observation",
                "valid_mechanics_measurements": [],
            },
        )
        (staged_capture / "notes.md").write_text(
            f"# {experiment_id}\n\n"
            "Status: `prepared_non_golden`. This bundle contains provenance, not an "
            "original-game observation.\n\n"
            "Fill in the hypothesis and discriminating outcomes in `metadata.json` before "
            "launch. Append every preflight, capture, input, delay, and result to "
            "`actions.jsonl`.\n\n"
            "The copied historical `launch-irisu.sh`, when present, was removed from the "
            "disposable tree; the preserved source was not changed.\n\n"
            "Launch only this disposable run:\n\n"
            f"```bash\n{launch_command}\n```\n\n"
            "Immediately after the run, copy its generated `replay/new.rpy` to "
            "`result.rpy`, record final save/replay and changed-tree hashes, add exact-window "
            "frames and measurements, and replace the prepared status with the actual outcome. "
            "Do not promote the bundle to the golden manifest until all required evidence is "
            "complete.\n",
            encoding="utf-8",
        )

        publish_noreplace(staged_run, run_dir)
        # A failure here deliberately leaves the already published, marked run
        # intact. Removing the pathname would risk deleting a concurrent
        # replacement, and two directories cannot be published transactionally.
        try:
            publish_noreplace(staged_capture, capture_dir)
        except Exception as exc:
            raise PreparationError(
                f"run was published at {run_dir}, but capture publication failed; "
                f"the run was left intact: {exc}"
            ) from exc
    finally:
        shutil.rmtree(run_stage_parent, ignore_errors=True)
        shutil.rmtree(capture_stage_parent, ignore_errors=True)

    return {
        "experiment_id": experiment_id,
        "status": "prepared_non_golden",
        "run_dir": str(run_dir),
        "capture_dir": str(capture_dir),
        "replay_sha256": replay_hash,
        "launch_command": launch_command,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a disposable original-game run and non-golden capture bundle."
    )
    parser.add_argument("experiment_id")
    parser.add_argument("replay", type=Path)
    parser.add_argument("--layout", choices=("auto", "legacy", "padded"), default="auto")
    args = parser.parse_args()
    try:
        report = prepare_capture(args.experiment_id, args.replay, layout=args.layout)
    except (OSError, PreparationError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
