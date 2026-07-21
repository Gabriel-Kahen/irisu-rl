#!/usr/bin/env python3
"""Run a conservatively parsed normal-mode replay against the headless clone.

This is a diagnostic bridge, not a replay-fidelity oracle. The recovered game
loop proves that each record advances one 0.020-second gameplay update; fast
forward skips rendering rather than multiplying simulation time.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Literal


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from irisu_env import Action, ActionKind, IrisuEnv  # noqa: E402
from irisu_env.native import NativeError  # noqa: E402


def _load_replay_parser() -> Any:
    name = "irisu_inspect_rpy"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "inspect-rpy.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the conservative replay parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


INSPECT_RPY = _load_replay_parser()
MappedKind = Literal["wait", "weak_shot", "strong_shot", "both_shots"]


@dataclass(frozen=True, slots=True)
class MappedFrame:
    index: int
    raw_word: int
    kind: MappedKind
    cursor_x: int
    cursor_y: int
    left_level: bool
    right_level: bool
    left_edge: bool
    right_edge: bool
    suppressed_left_edge: bool = False
    suppressed_right_edge: bool = False
    unrepresented_right_edge: bool = False

    def trace_record(self) -> dict[str, object]:
        shots: list[str] = []
        if self.left_edge:
            shots.append("weak_shot")
        if self.right_edge:
            shots.append("strong_shot")
        return {
            "frame": self.index,
            "raw_word": self.raw_word,
            "kind": self.kind,
            "cursor_x": self.cursor_x,
            "cursor_y": self.cursor_y,
            "left_level": self.left_level,
            "right_level": self.right_level,
            "left_edge": self.left_edge,
            "right_edge": self.right_edge,
            "suppressed_left_edge": self.suppressed_left_edge,
            "suppressed_right_edge": self.suppressed_right_edge,
            "shot_order": shots,
            "unrepresented_right_edge": self.unrepresented_right_edge,
        }


def both_shots_available() -> bool:
    """Return whether this Python/native API generation exposes one-tick dual shots."""

    return hasattr(ActionKind, "BOTH_SHOTS")


def map_frames(
    frames: Iterable[Any], *, support_both_shots: bool
) -> tuple[MappedFrame, ...]:
    """Map replay levels to the fresh shots visible to Field.update."""

    previous_left = False
    previous_right = False
    mapped: list[MappedFrame] = []
    for index, frame in enumerate(frames):
        raw_left_edge = bool(frame.left and not previous_left)
        raw_right_edge = bool(frame.right and not previous_right)
        # Input.update increments the replay index, then clears both fresh-edge
        # bytes while index < 3. Held history is still updated, so records 0
        # and 1 suppress new edges independently.
        suppress_edges = index < 2
        suppressed_left_edge = raw_left_edge and suppress_edges
        suppressed_right_edge = raw_right_edge and suppress_edges
        left_edge = raw_left_edge and not suppress_edges
        right_edge = raw_right_edge and not suppress_edges
        unrepresented_right = False
        left_level = bool(frame.left)
        right_level = bool(frame.right)
        if left_edge and right_edge:
            if support_both_shots:
                kind: MappedKind = "both_shots"
            else:
                # Preserve one record == one tick.  The older API cannot issue
                # the second shot without advancing an extra tick, so retain
                # the original game's left-first order and report the omission.
                kind = "weak_shot"
                unrepresented_right = right_edge
        elif left_edge:
            kind = "weak_shot"
        elif right_edge:
            kind = "strong_shot"
        else:
            kind = "wait"
        mapped.append(
            MappedFrame(
                index=index,
                raw_word=int(frame.raw_word),
                kind=kind,
                cursor_x=int(frame.x),
                cursor_y=int(frame.y),
                left_level=left_level,
                right_level=right_level,
                left_edge=left_edge,
                right_edge=right_edge,
                suppressed_left_edge=suppressed_left_edge,
                suppressed_right_edge=suppressed_right_edge,
                unrepresented_right_edge=unrepresented_right,
            )
        )
        previous_left = bool(frame.left)
        previous_right = bool(frame.right)
    return tuple(mapped)


def _env_action(frame: MappedFrame) -> Action:
    if frame.kind == "weak_shot":
        return Action.weak(frame.cursor_x, frame.cursor_y)
    if frame.kind == "strong_shot":
        return Action.strong(frame.cursor_x, frame.cursor_y)
    if frame.kind == "both_shots":
        both = getattr(Action, "both", None)
        if both is None:
            raise RuntimeError("mapped both_shots but the environment API does not expose it")
        return both(frame.cursor_x, frame.cursor_y)
    # Cursor coordinates remain in the action even when neither edge fires.
    return Action(ActionKind.WAIT, frame.cursor_x, frame.cursor_y, 1)


def _trace_sha256(frames: Iterable[MappedFrame]) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        encoded = json.dumps(
            frame.trace_record(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest.update(encoded)
        digest.update(b"\n")
    return digest.hexdigest()


def _u64_hex(value: int) -> str:
    return f"0x{value & 0xFFFFFFFFFFFFFFFF:016x}"


def evaluate_bytes(
    data: bytes,
    *,
    source: str = "<memory>",
    layout: str = "auto",
    library_path: str | None = None,
    env_factory: Callable[..., Any] = IrisuEnv,
    support_both_shots: bool | None = None,
) -> dict[str, object]:
    """Evaluate replay bytes and return a stable, machine-readable report."""

    replay = INSPECT_RPY.parse_replay(data, layout)
    if replay.header.mode != 0:
        raise ValueError("the headless clone only evaluates normal-mode (mode 0) replays")

    supports_both = both_shots_available() if support_both_shots is None else support_both_shots
    if supports_both and not both_shots_available() and env_factory is IrisuEnv:
        raise ValueError("support_both_shots=True requires an API with ActionKind.BOTH_SHOTS")
    mapped = map_frames(replay.frames, support_both_shots=supports_both)

    mapped_counts = Counter(frame.kind for frame in mapped)
    event_counts: Counter[str] = Counter()
    first_event_occurrences: dict[str, dict[str, object]] = {}
    invalid_action_frames: list[int] = []
    terminal_frame: int | None = None
    clone_observation: dict[str, Any]
    step_diagnostics: dict[str, Any] = {}

    kwargs = {"library_path": library_path} if library_path is not None else {}
    with env_factory(**kwargs) as env:
        # A signed header field is serialized as 32 raw bits.  Zero-extension
        # preserves those bits for the clone's uint64 reset API.
        seed_u32 = int(replay.header.seed) & 0xFFFFFFFF
        initial_observation, _ = env.reset(seed=seed_u32)
        initial_state_hash = int(env.state_hash())
        initial_tick = int(initial_observation["tick"])
        for frame in mapped:
            clone_observation, _, terminated, truncated, info = env.step(_env_action(frame))
            raw_diagnostics = info.get("diagnostics", {})
            if isinstance(raw_diagnostics, dict):
                step_diagnostics = dict(raw_diagnostics)
            events = info.get("events", ())
            for event in events:
                name = str(
                    event.get("kind_name", f"kind_{event.get('kind', 'unknown')}")
                )
                event_counts[name] += 1
                if name not in first_event_occurrences:
                    first_event_occurrences[name] = {
                        "frame": frame.index,
                        "tick": int(event.get("tick", frame.index + 1)),
                        "a": int(event.get("a", 0)),
                        "b": int(event.get("b", 0)),
                        "value": int(event.get("value", 0)),
                        "detail": str(event.get("detail", "")),
                    }
            if info.get("invalid_action"):
                invalid_action_frames.append(frame.index)
            if terminal_frame is None and (terminated or truncated):
                terminal_frame = frame.index
        if not mapped:
            clone_observation = initial_observation
        final_snapshot = env.clone_state()
        final_state_hash = int(env.state_hash())
        config_hash = int(env.config_hash())
        build_info = dict(env.build_info)

    final_tick = int(clone_observation["tick"])
    live_score = int(clone_observation["score"])
    live_level = int(clone_observation["level"])
    live_highest_chain = int(clone_observation.get("highest_chain", 0))
    recorded = bool(step_diagnostics.get("terminal_metadata_recorded", False))
    final_score = int(
        step_diagnostics.get("recorded_final_score", live_score)
        if recorded else live_score
    )
    final_level = int(
        step_diagnostics.get("recorded_final_level", live_level)
        if recorded else live_level
    )
    final_highest_chain = int(
        step_diagnostics.get("recorded_final_highest_chain", live_highest_chain)
        if recorded else live_highest_chain
    )
    unrepresented = [frame.index for frame in mapped if frame.unrepresented_right_edge]
    simultaneous_edges = [
        frame.index for frame in mapped if frame.left_edge and frame.right_edge
    ]
    level_frames = {
        "left": sum(bool(frame.left) for frame in replay.frames),
        "right": sum(bool(frame.right) for frame in replay.frames),
        "simultaneous": sum(bool(frame.left and frame.right) for frame in replay.frames),
    }
    edge_counts = {
        "left": sum(frame.left_edge for frame in mapped),
        "right": sum(frame.right_edge for frame in mapped),
        "simultaneous": len(simultaneous_edges),
    }

    expected_score = int(replay.header.final_score)
    expected_level = int(replay.header.highest_level)
    return {
        "schema_version": 1,
        "status": {
            "purpose": "diagnostic replay-to-clone comparison",
            "golden_fidelity_verdict": None,
            "mismatches_are_failures": False,
        },
        "assumptions": {
            "cadence": "PROVEN: one replay record is exactly one 0.020-second gameplay update",
            "buttons": (
                "stored levels drive an adapter-held history; original playback "
                "suppresses fresh edges on records 0 and 1"
            ),
            "mapping": "fresh left edge -> weak action; fresh right edge -> strong; both -> both",
            "simultaneous_order": "left weak shot, then right strong shot",
            "seed_mapping": "signed int32 header bits zero-extended to uint32 for reset",
        },
        "replay": {
            "source": source,
            "size_bytes": len(data),
            "layout": replay.layout,
            "frame_offset": replay.frame_offset,
            "frame_count": len(replay.frames),
            "header": {
                "seed_int32": int(replay.header.seed),
                "seed_uint32": int(replay.header.seed) & 0xFFFFFFFF,
                "highest_level": expected_level,
                "final_score": expected_score,
                "highest_chain": int(replay.header.highest_chain),
                "mode": int(replay.header.mode),
            },
        },
        "mapping": {
            "both_shots_api_available": both_shots_available(),
            "both_shots_mapping_enabled": supports_both,
            "simultaneous_fresh_edge_frames": simultaneous_edges,
            "startup_suppressed_edge_frames": [
                frame.index
                for frame in mapped
                if frame.suppressed_left_edge or frame.suppressed_right_edge
            ],
            "unrepresented_right_edge_frames": unrepresented,
            "limitation": (
                None
                if not unrepresented
                else "base API lacks a one-tick dual-shot action; left was applied and right omitted"
            ),
            "cursor_coordinates_preserved_on_every_record": True,
        },
        "action_counts": {
            "button_level_frames": level_frames,
            "fresh_edges": edge_counts,
            "mapped_actions": {
                name: mapped_counts.get(name, 0)
                for name in ("wait", "weak_shot", "strong_shot", "both_shots")
            },
            "unrepresented_shot_edges": len(unrepresented),
            "invalid_clone_action_frames": invalid_action_frames,
            "events": dict(sorted(event_counts.items())),
            "first_event_occurrences": dict(sorted(first_event_occurrences.items())),
        },
        "cadence": {
            "requested_tick_steps": len(mapped),
            "actual_tick_delta": final_tick - initial_tick,
            "terminal_frame": terminal_frame,
            "post_terminal_record_count": (
                0 if terminal_frame is None else len(mapped) - terminal_frame - 1
            ),
        },
        "outcome": {
            "score": {
                "expected_header_final": expected_score,
                "clone_final": final_score,
                "delta": final_score - expected_score,
                "matches": final_score == expected_score,
            },
            "level": {
                "expected_header_highest": expected_level,
                "clone_final": final_level,
                "delta": final_level - expected_level,
                "matches": final_level == expected_level,
            },
            "highest_chain": {
                "expected_header_highest": int(replay.header.highest_chain),
                "clone_final": final_highest_chain,
                "matches": final_highest_chain == int(replay.header.highest_chain),
            },
            "clone": {
                "tick": final_tick,
                "terminated": bool(clone_observation["terminated"]),
                "truncated": bool(clone_observation["truncated"]),
                "terminal_metadata_recorded": recorded,
                "finish_call_count": int(step_diagnostics.get("finish_call_count", 0)),
                "live_score_after_final_update": live_score,
                "live_level_after_final_update": live_level,
                "live_highest_chain_after_final_update": live_highest_chain,
                "latest_final_score": int(
                    step_diagnostics.get("latest_final_score", live_score)
                ),
                "latest_final_level": int(
                    step_diagnostics.get("latest_final_level", live_level)
                ),
                "latest_final_highest_chain": int(
                    step_diagnostics.get(
                        "latest_final_highest_chain", live_highest_chain
                    )
                ),
            },
        },
        "hashes": {
            "replay_sha256": hashlib.sha256(data).hexdigest(),
            "mapped_actions_sha256": _trace_sha256(mapped),
            "final_snapshot_sha256": hashlib.sha256(final_snapshot).hexdigest(),
            "initial_state_u64": _u64_hex(initial_state_hash),
            "final_state_u64": _u64_hex(final_state_hash),
            "config_u64": _u64_hex(config_hash),
        },
        "clone_build": build_info,
    }


def evaluate_path(
    path: Path,
    *,
    layout: str = "auto",
    library_path: str | None = None,
) -> dict[str, object]:
    return evaluate_bytes(
        path.read_bytes(), source=str(path), layout=layout, library_path=library_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="diagnostically run a normal .rpy input trace against the headless clone"
    )
    parser.add_argument("replay", type=Path)
    parser.add_argument("--layout", choices=("auto", "legacy", "padded"), default="auto")
    parser.add_argument("--library", help="explicit libirisu_clone shared-library path")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    args = parser.parse_args()

    try:
        report = evaluate_path(args.replay, layout=args.layout, library_path=args.library)
    except (OSError, ValueError, RuntimeError, NativeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=None if args.compact else 2, sort_keys=True))


if __name__ == "__main__":
    main()
