#!/usr/bin/env python3
"""Evaluate exact-forward replays against headers and observed v2.03 oracles.

Replay headers are unverified compatibility metadata.  Captured playback from
the bundled executable is the authoritative scoring oracle when available.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "reference" / "replays" / "raw" / "internet"
DEFAULT_ORACLE_ROOT = ROOT / "reference" / "runs"
DEFAULT_CONTROL_WORD = 0x027F
CANONICAL_EXE_SHA256 = (
    "0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255"
)
CANONICAL_BOX2D_SHA256 = (
    "34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd"
)
ORACLE_STATUSES = {
    "valid_full_original_replay_event_oracle",
    "valid_repeated_original_replay_event_oracle_header_incompatible",
}


def _load_parser() -> Any:
    spec = importlib.util.spec_from_file_location(
        "irisu_inspect_rpy_exact_corpus", ROOT / "tools" / "inspect-rpy.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load tools/inspect-rpy.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


INSPECT_RPY = _load_parser()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _header(replay: Any) -> dict[str, int]:
    return {
        "seed": int(replay.header.seed),
        "highest_level": int(replay.header.highest_level),
        "final_score": int(replay.header.final_score),
        "highest_chain": int(replay.header.highest_chain),
        "mode": int(replay.header.mode),
    }


def inventory(paths: Iterable[Path]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in sorted((value.resolve() for value in paths), key=str):
        data = path.read_bytes()
        replay = INSPECT_RPY.parse_replay(data, "auto")
        reasons: list[str] = []
        if replay.header.mode != 0:
            reasons.append("not_normal_mode")
        if replay.layout != "padded":
            reasons.append("legacy_layout_predates_v203_mechanics")
        entries.append(
            {
                "path": str(path),
                "sha256": _sha256(data),
                "size_bytes": len(data),
                "layout": replay.layout,
                "frame_offset": replay.frame_offset,
                "frame_count": len(replay.frames),
                "header": _header(replay),
                "eligible": not reasons,
                "exclusion_reasons": reasons,
            }
        )
    return entries


def _runner_result(
    runner: Path, replay: Path, *, control_word: int, timeout: float
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["IRISU_EXACT_CW"] = hex(control_word)
    completed = subprocess.run(
        [str(runner), str(replay)],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"runner exited {completed.returncode}: {detail}")
    try:
        result = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"runner did not emit one JSON object: {error}") from error
    if not isinstance(result, dict):
        raise RuntimeError("runner JSON must be an object")
    required = ("tick", "score", "level", "highest_chain", "terminal_frame")
    for key in required:
        if type(result.get(key)) is not int:
            raise RuntimeError(f"runner field {key!r} must be an integer")
    return result


def _comparison(expected: int, actual: int) -> dict[str, Any]:
    return {
        "expected": expected,
        "actual": actual,
        "delta": actual - expected,
        "matches": actual == expected,
    }


def _compact_runner_output(result: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for key, value in result.items():
        if not isinstance(value, list):
            compact[key] = value
            continue
        encoded = json.dumps(value, separators=(",", ":")).encode()
        compact[f"{key}_count"] = len(value)
        compact[f"{key}_sha256"] = _sha256(encoded)
    return compact


def discover_oracle_metadata(root: Path) -> list[Path]:
    paths: list[Path] = []
    if not root.is_dir():
        return paths
    for path in sorted(root.rglob("metadata.json")):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict) and value.get("status") in ORACLE_STATUSES:
            paths.append(path)
    return paths


def _object(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a JSON object")
    return value


def _integer(value: Any, where: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{where} must be an integer")
    return value


def _timeline(value: Any, width: int, where: str) -> list[list[int]]:
    if not isinstance(value, list):
        raise ValueError(f"{where} must be an array")
    result: list[list[int]] = []
    for index, row in enumerate(value):
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"{where}[{index}] must contain {width} integers")
        result.append(
            [_integer(item, f"{where}[{index}]") for item in row]
        )
    return result


def _timeline_summary(values: list[list[int]]) -> dict[str, Any]:
    encoded = json.dumps(values, separators=(",", ":")).encode()
    return {"count": len(values), "sha256": _sha256(encoded)}


def _load_oracle(path: Path) -> dict[str, Any]:
    path = path.resolve()
    try:
        metadata_bytes = path.read_bytes()
        metadata = _object(json.loads(metadata_bytes), str(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read oracle metadata {path}: {error}") from error
    if metadata.get("schema") != 1 or metadata.get("status") not in ORACLE_STATUSES:
        raise ValueError(f"unsupported oracle schema/status in {path}")

    inputs = _object(metadata.get("inputs"), f"{path}:inputs")
    if inputs.get("irisu_exe_sha256") != CANONICAL_EXE_SHA256:
        raise ValueError(f"oracle does not identify the canonical v2.03 executable: {path}")
    if inputs.get("box2d_dll_sha256") != CANONICAL_BOX2D_SHA256:
        raise ValueError(f"oracle does not identify the shipped Box2D DLL: {path}")
    replay_sha = inputs.get("replay_sha256")
    if not isinstance(replay_sha, str) or len(replay_sha) != 64:
        raise ValueError(f"oracle replay SHA-256 is invalid: {path}")
    replay_name = inputs.get("replay_path")
    if isinstance(replay_name, str) and replay_name:
        candidates = [(path.parent / replay_name).resolve()]
    else:
        candidates = sorted((path.parent / "replay").glob("*.rpy"))
    replay_path: Path | None = None
    replay_data = b""
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            candidate.resolve().relative_to(path.parent.resolve())
        except ValueError as error:
            raise ValueError(f"oracle replay path escapes its bundle: {path}") from error
        data = candidate.read_bytes()
        if _sha256(data) == replay_sha:
            replay_path = candidate.resolve()
            replay_data = data
            break
    if replay_path is None:
        raise ValueError(f"oracle bundle has no replay matching {replay_sha}: {path}")
    frame_offset = _integer(inputs.get("frame_offset", 52), "oracle frame_offset")
    if frame_offset < 0 or len(replay_data) < frame_offset:
        raise ValueError(f"oracle replay frame offset is invalid: {path}")
    frame_count = (len(replay_data) - frame_offset) // 4
    if inputs.get("frame_count", frame_count) != frame_count:
        raise ValueError(f"oracle replay frame count mismatch: {path}")

    events_path = path.with_name("events.jsonl")
    events_bytes_hash = hashlib.sha256()
    score_timeline: list[list[int]] = []
    rot_timeline: list[list[int]] = []
    try:
        with events_path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                events_bytes_hash.update(raw)
                event = _object(
                    json.loads(raw), f"{events_path}:{line_number}"
                )
                if event.get("event") == "score":
                    score_timeline.append(
                        [
                            _integer(event.get("tick"), "score tick"),
                            _integer(event.get("delta"), "score delta"),
                            _integer(event.get("score"), "score total"),
                        ]
                    )
                elif event.get("event") == "rot_penalty":
                    rot_timeline.append(
                        [
                            _integer(event.get("tick"), "rot tick"),
                            _integer(event.get("delta"), "rot delta"),
                            _integer(event.get("gauge"), "rot gauge"),
                        ]
                    )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read oracle events {events_path}: {error}") from error
    events_sha = events_bytes_hash.hexdigest()
    artifacts = _object(metadata.get("artifacts"), f"{path}:artifacts")
    if artifacts.get("events_jsonl_sha256") != events_sha:
        raise ValueError(f"oracle event hash mismatch: {events_path}")
    repeat = metadata.get("repeat")
    repeated = repeat is not None
    if repeated:
        repeat = _object(repeat, f"{path}:repeat")
        if repeat.get("normalized_events_byte_identical") is not True:
            raise ValueError(f"oracle repeat is not byte-identical: {path}")
        if repeat.get("normalized_events_sha256") != events_sha:
            raise ValueError(f"oracle repeat hash mismatch: {path}")

    result = _object(metadata.get("result"), f"{path}:result")
    replay_exhaustion = result.get("natural_gauge_death") is False
    tick_key = "tick_at_replay_exhaustion" if replay_exhaustion else "tick"
    gauge_key = (
        "gauge_at_replay_exhaustion"
        if replay_exhaustion
        else "gauge_after_terminal_actor_pass"
    )
    terminal = {
        "tick": _integer(result.get(tick_key), f"oracle result {tick_key}"),
        "score": _integer(result.get("score"), "oracle result score"),
        "level": _integer(result.get("level"), "oracle result level"),
        "highest_chain": _integer(
            result.get("highest_chain"), "oracle result highest_chain"
        ),
        "clears": _integer(
            result.get("qualifying_clears"), "oracle result qualifying_clears"
        ),
        "score_calls": _integer(
            result.get("score_calls"), "oracle result score_calls"
        ),
        "gauge": _integer(
            result.get(gauge_key),
            f"oracle result {gauge_key}",
        ),
    }
    if not replay_exhaustion:
        terminal["terminal_frame"] = _integer(
            result.get("terminal_input_frame", terminal["tick"] - 1),
            "oracle terminal input frame",
        )
    if terminal["score_calls"] != len(score_timeline):
        raise ValueError(f"oracle score-call count disagrees with events: {path}")
    if result.get("rot_penalties", len(rot_timeline)) != len(rot_timeline):
        raise ValueError(f"oracle rot count disagrees with events: {path}")

    return {
        "replay_sha256": replay_sha,
        "frame_count": frame_count,
        "checkpoint_kind": (
            "replay_exhaustion" if replay_exhaustion else "natural_terminal"
        ),
        "terminal": terminal,
        "score_timeline": score_timeline,
        "rot_timeline": rot_timeline,
        "evidence": {
            "authority": "observed_bundled_v203_playback",
            "metadata_path": str(path),
            "metadata_sha256": _sha256(metadata_bytes),
            "events_path": str(events_path),
            "events_sha256": events_sha,
            "repeated_byte_identically": repeated,
            "executable_sha256": CANONICAL_EXE_SHA256,
            "box2d_sha256": CANONICAL_BOX2D_SHA256,
        },
    }


def load_oracles(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    oracles: dict[str, dict[str, Any]] = {}
    for path in paths:
        oracle = _load_oracle(path)
        replay_sha = str(oracle["replay_sha256"])
        if replay_sha in oracles:
            raise ValueError(f"multiple authoritative oracles for replay {replay_sha}")
        oracles[replay_sha] = oracle
    return oracles


def _timeline_comparison(
    expected: list[list[int]], actual: list[list[int]]
) -> dict[str, Any]:
    first_bad = next(
        (
            index
            for index in range(min(len(expected), len(actual)))
            if expected[index] != actual[index]
        ),
        None,
    )
    if first_bad is None and len(expected) != len(actual):
        first_bad = min(len(expected), len(actual))
    return {
        "expected": _timeline_summary(expected),
        "actual": _timeline_summary(actual),
        "matches": first_bad is None,
        "first_mismatch_index": first_bad,
    }


def _compare_oracle(raw: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    field_map = {
        "tick": "tick",
        "score": "score",
        "level": "level",
        "highest_chain": "highest_chain",
        "clears": "clears",
        "score_calls": "score_calls",
        "gauge": "gauge",
    }
    terminal = {
        name: _comparison(
            int(oracle["terminal"][name]),
            _integer(raw.get(raw_name), f"runner {raw_name}"),
        )
        for name, raw_name in field_map.items()
    }
    terminal_frame = _integer(raw.get("terminal_frame"), "runner terminal_frame")
    if oracle["checkpoint_kind"] == "natural_terminal":
        terminal["terminal_frame"] = _comparison(
            int(oracle["terminal"]["terminal_frame"]), terminal_frame
        )
        checkpoint = {
            "kind": "natural_terminal",
            "matches": terminal["terminal_frame"]["matches"],
        }
    else:
        frame_count = int(oracle["frame_count"])
        actual_tick = _integer(raw.get("tick"), "runner tick")
        checkpoint = {
            "kind": "replay_exhaustion",
            "expected_frame_count": frame_count,
            "actual_tick": actual_tick,
            "actual_terminal_frame": terminal_frame,
            "all_replay_frames_consumed": actual_tick == frame_count,
            "no_natural_terminal_before_exhaustion": terminal_frame >= frame_count,
            "matches": (
                actual_tick == frame_count and terminal_frame >= frame_count
            ),
        }
    scores = _timeline(raw.get("score_timeline"), 3, "runner score_timeline")
    gauge = _timeline(raw.get("gauge_timeline"), 5, "runner gauge_timeline")
    rot = [row[:3] for row in gauge if row[4] == 1]
    score_comparison = _timeline_comparison(oracle["score_timeline"], scores)
    rot_comparison = _timeline_comparison(oracle["rot_timeline"], rot)
    terminal_exact = (
        all(value["matches"] for value in terminal.values())
        and checkpoint["matches"]
    )
    return {
        "evidence": oracle["evidence"],
        "checkpoint": checkpoint,
        "terminal": terminal,
        "terminal_state_exact": terminal_exact,
        "score_timeline": score_comparison,
        "rot_penalty_timeline": rot_comparison,
        "full_scoring_parity": (
            terminal_exact
            and score_comparison["matches"]
            and rot_comparison["matches"]
        ),
    }


def evaluate(
    paths: Iterable[Path],
    *,
    runner: Path,
    oracle_paths: Iterable[Path] = (),
    control_word: int = DEFAULT_CONTROL_WORD,
    timeout: float = 300.0,
) -> dict[str, Any]:
    runner = runner.resolve()
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise ValueError(f"runner is not an executable file: {runner}")
    entries = inventory(paths)
    oracles = load_oracles(oracle_paths)
    errors = 0
    for entry in entries:
        if not entry["eligible"]:
            entry["evaluation"] = None
            continue
        try:
            raw = _runner_result(
                runner,
                Path(entry["path"]),
                control_word=control_word,
                timeout=timeout,
            )
            frames = int(entry["frame_count"])
            terminal_frame = int(raw["terminal_frame"])
            terminal_tick = terminal_frame + 1 if 0 <= terminal_frame < frames else None
            expected = entry["header"]
            comparisons = {
                "score": _comparison(int(expected["final_score"]), int(raw["score"])),
                "level": _comparison(int(expected["highest_level"]), int(raw["level"])),
                "highest_chain": _comparison(
                    int(expected["highest_chain"]), int(raw["highest_chain"])
                ),
                "record_count_tick": _comparison(frames, int(raw["tick"])),
                "terminal_tick": {
                    "expected_from_record_count": frames,
                    "actual_first_terminal": terminal_tick,
                    "matches": terminal_tick == frames,
                    "note": "the replay header has no tick field",
                },
            }
            oracle = oracles.get(str(entry["sha256"]))
            entry["evaluation"] = {
                "status": "ok",
                "runner_output": _compact_runner_output(raw),
                "unverified_header_diagnostic": {
                    "authority": "unverified_replay_header",
                    "comparisons": comparisons,
                    "all_header_scalars_match": all(
                        comparisons[key]["matches"]
                        for key in ("score", "level", "highest_chain")
                    ),
                },
                "observed_v203_oracle": (
                    None if oracle is None else _compare_oracle(raw, oracle)
                ),
            }
        except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
            errors += 1
            entry["evaluation"] = {"status": "error", "error": str(error)}

    successful = [
        entry
        for entry in entries
        if entry["evaluation"] is not None
        and entry["evaluation"]["status"] == "ok"
    ]
    score_deltas = [
        int(
            entry["evaluation"]["unverified_header_diagnostic"]["comparisons"]
            ["score"]["delta"]
        )
        for entry in successful
    ]
    relative_errors = [
        abs(delta) / int(entry["header"]["final_score"])
        for entry, delta in zip(successful, score_deltas, strict=True)
        if int(entry["header"]["final_score"]) != 0
    ]
    expected_score_total = sum(int(entry["header"]["final_score"]) for entry in successful)
    actual_score_total = sum(
        int(
            entry["evaluation"]["unverified_header_diagnostic"]["comparisons"]
            ["score"]["actual"]
        )
        for entry in successful
    )
    observed = [
        entry["evaluation"]["observed_v203_oracle"]
        for entry in successful
        if entry["evaluation"]["observed_v203_oracle"] is not None
    ]
    return {
        "schema_version": 2,
        "scope": {
            "target": "bundled v2.03 normal-mode scoring",
            "evidence_order": [
                "observed bundled-v2.03 playback oracle",
                "unverified replay-header compatibility diagnostic",
            ],
            "warning": "padded layout does not prove the replay's generating build",
            "eligibility": "mode 0 and padded offset-52 layout",
        },
        "runner": {
            "path": str(runner),
            "sha256": _sha256(runner.read_bytes()),
            "control_word": f"0x{control_word:04x}",
        },
        "inventory": entries,
        "summary": {
            "discovered": len(entries),
            "eligible": sum(bool(entry["eligible"]) for entry in entries),
            "excluded": sum(not entry["eligible"] for entry in entries),
            "evaluated": len(successful),
            "runner_errors": errors,
            "observed_v203_oracles": {
                "available_for_corpus": len(observed),
                "terminal_state_exact": sum(
                    value["terminal_state_exact"] for value in observed
                ),
                "score_timeline_exact": sum(
                    value["score_timeline"]["matches"] for value in observed
                ),
                "rot_penalty_timeline_exact": sum(
                    value["rot_penalty_timeline"]["matches"] for value in observed
                ),
                "full_scoring_parity": sum(
                    value["full_scoring_parity"] for value in observed
                ),
            },
            "unverified_header_diagnostics": {
                "exact_score": sum(
                    entry["evaluation"]["unverified_header_diagnostic"]
                    ["comparisons"]["score"]["matches"]
                    for entry in successful
                ),
                "exact_all_header_scalars": sum(
                    entry["evaluation"]["unverified_header_diagnostic"]
                    ["all_header_scalars_match"]
                    for entry in successful
                ),
                "score_mean_absolute_error": (
                    statistics.fmean(abs(value) for value in score_deltas)
                    if score_deltas
                    else None
                ),
                "score_mean_absolute_percentage_error": (
                    statistics.fmean(relative_errors) if relative_errors else None
                ),
                "score_weighted_absolute_percentage_error": (
                    sum(abs(value) for value in score_deltas) / expected_score_total
                    if expected_score_total
                    else None
                ),
                "score_expected_total": expected_score_total,
                "score_actual_total": actual_score_total,
                "not_a_fidelity_verdict": True,
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, type=Path)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--oracle-root",
        type=Path,
        default=DEFAULT_ORACLE_ROOT,
        help="discover validated original-playback metadata below this directory",
    )
    parser.add_argument(
        "--no-oracles", action="store_true", help="run header diagnostics only"
    )
    parser.add_argument(
        "--control-word", type=lambda value: int(value, 0), default=DEFAULT_CONTROL_WORD
    )
    parser.add_argument("--timeout", type=float, default=300.0, help="seconds per replay")
    parser.add_argument("--output", type=Path, help="also write the JSON report here")
    parser.add_argument("--compact", action="store_true")
    parser.add_argument(
        "--require-observed-parity",
        action="store_true",
        help="fail unless every discovered original-playback oracle is exact",
    )
    args = parser.parse_args()
    try:
        paths = sorted(args.corpus.glob("*.rpy"))
        if not paths:
            raise ValueError(f"no .rpy files found in {args.corpus}")
        report = evaluate(
            paths,
            runner=args.runner,
            oracle_paths=(
                []
                if args.no_oracles
                else discover_oracle_metadata(args.oracle_root)
            ),
            control_word=args.control_word,
            timeout=args.timeout,
        )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    encoded = json.dumps(report, indent=None if args.compact else 2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if report["summary"]["runner_errors"]:
        raise SystemExit(1)
    if args.require_observed_parity:
        observed = report["summary"]["observed_v203_oracles"]
        available = observed["available_for_corpus"]
        if not available:
            raise SystemExit("no observed v2.03 replay oracle matched the corpus")
        required = (
            "full_scoring_parity",
            "rot_penalty_timeline_exact",
            "score_timeline_exact",
            "terminal_state_exact",
        )
        failed = [key for key in required if observed[key] != available]
        if failed:
            raise SystemExit(
                "exact replay parity failed for one or more observed oracles: "
                + ", ".join(failed)
            )


if __name__ == "__main__":
    main()
