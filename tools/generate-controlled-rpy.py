#!/usr/bin/env python3
"""Generate padded v2.03 replays for controlled reference probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))


HEADER = struct.Struct("<5i")
PADDING = bytes(32)
BUTTON_BITS = {"weak": 1, "strong": 2, "both": 3}
SCORE_PROBE_NAME = "score-seed41-parity"
SCORE_PROBE_SEED = 41
SCORE_PROBE_FRAMES = 520
SCORE_PROBE_SHA256 = "1ce501febe8f3f6291e4b82736542179bd9808e412d38e0e1fb1c92d05797657"
SCORE_PROBE_STATE_HASH = 4_642_958_100_242_870_413
SCORE_PROBE_SHOTS = (
    ("strong", 4, 453, 380),
    ("strong", 272, 367, 380),
    ("strong", 364, 303, 380),
    ("strong", 384, 303, 380),
)


@dataclass(frozen=True, slots=True)
class Shot:
    kind: str
    frame: int
    x: int
    y: int


def parse_shot(value: str) -> Shot:
    try:
        kind, frame, x, y = value.split(":")
        shot = Shot(kind.lower(), int(frame), int(x), int(y))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "shot must be KIND:FRAME:X:Y"
        ) from exc
    if shot.kind not in BUTTON_BITS:
        raise argparse.ArgumentTypeError("shot kind must be weak, strong, or both")
    if shot.frame < 0:
        raise argparse.ArgumentTypeError("shot frame must be nonnegative")
    if not 0 <= shot.x <= 1023 or not 0 <= shot.y <= 511:
        raise argparse.ArgumentTypeError("shot coordinates exceed replay bit fields")
    return shot


def build_replay(
    *,
    seed: int,
    frame_count: int,
    shots: list[Shot],
    highest_level: int = 0,
    final_score: int = 0,
    highest_chain: int = 0,
    mode: int = 0,
) -> bytes:
    if type(seed) is not int or not 0 <= seed <= 0xFFFF_FFFF:
        raise ValueError("seed must fit in uint32")
    replay_seed = seed if seed < 1 << 31 else seed - (1 << 32)
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    metadata = (highest_level, final_score, highest_chain, mode)
    if any(not -(1 << 31) <= value < (1 << 31) for value in metadata):
        raise ValueError("header metadata must fit in signed int32")

    words = [0] * frame_count
    coordinates: dict[int, tuple[int, int]] = {}
    for shot in shots:
        if shot.kind not in BUTTON_BITS:
            raise ValueError(f"unknown shot kind: {shot.kind}")
        if shot.frame < 0:
            raise ValueError("shot frame must be nonnegative")
        if not 0 <= shot.x <= 1023 or not 0 <= shot.y <= 511:
            raise ValueError("shot coordinates exceed replay bit fields")
        if shot.frame >= frame_count:
            raise ValueError(f"shot frame {shot.frame} is outside the replay")
        prior = coordinates.get(shot.frame)
        if prior is not None and prior != (shot.x, shot.y):
            raise ValueError(f"frame {shot.frame} has conflicting coordinates")
        if words[shot.frame] & BUTTON_BITS[shot.kind]:
            raise ValueError(f"frame {shot.frame} repeats button levels")
        coordinates[shot.frame] = (shot.x, shot.y)
        words[shot.frame] = (
            (shot.y << 12)
            | (shot.x << 2)
            | (words[shot.frame] & 3)
            | BUTTON_BITS[shot.kind]
        )

    return (
        HEADER.pack(replay_seed, highest_level, final_score, highest_chain, mode)
        + PADDING
        + struct.pack(f"<{frame_count}I", *words)
    )


def validate_score_probe(
    data: bytes, *, library_path: str | os.PathLike[str] | None = None
) -> dict[str, object]:
    """Replay the original-observed seed-41 score probe."""

    from irisu_env import Action, ActionKind, IrisuEnv

    expected_size = HEADER.size + len(PADDING) + SCORE_PROBE_FRAMES * 4
    if len(data) != expected_size or hashlib.sha256(data).hexdigest() != SCORE_PROBE_SHA256:
        raise RuntimeError("score probe bytes changed")
    if HEADER.unpack_from(data) != (SCORE_PROBE_SEED, 0, 0, 0, 0):
        raise RuntimeError("score probe must keep its recording-time header")

    score_events: list[tuple[int, int]] = []
    clear_actors: list[int] = []
    counts = {"weak": 0, "strong": 0, "both": 0}
    with IrisuEnv(library_path=library_path) as env:
        observation, _ = env.reset(seed=SCORE_PROBE_SEED)
        for (word,) in struct.iter_unpack("<I", data[52:]):
            buttons = word & 3
            x, y = (word >> 2) & 0x3FF, (word >> 12) & 0x1FF
            if buttons == 1:
                action, name = Action.weak(x, y), "weak"
            elif buttons == 2:
                action, name = Action.strong(x, y), "strong"
            elif buttons == 3:
                action, name = Action.both(x, y), "both"
            else:
                action, name = Action(ActionKind.WAIT, x, y, 1), None
            if name is not None:
                counts[name] += 1
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                raise RuntimeError("score probe terminated early")
            for event in info["events"]:
                if event["kind_name"] == "score_changed":
                    score_events.append((int(event["tick"]), int(event["value"])))
                elif event["kind_name"] == "cleared":
                    clear_actors.append(int(event["a"]))
        final_state_hash = int(env.state_hash())

    final = {
        name: int(observation[name])
        for name in (
            "tick",
            "score",
            "gauge",
            "level",
            "highest_chain",
            "qualifying_clear_count",
        )
    }
    expected_final = {
        "tick": 520,
        "score": 16,
        "gauge": 3_180,
        "level": 1,
        "highest_chain": 2,
        "qualifying_clear_count": 1,
    }
    if (
        score_events != [(304, 8), (304, 8)]
        or clear_actors != [10, 12]
        or counts != {"weak": 0, "strong": 4, "both": 0}
        or final != expected_final
        or final_state_hash != SCORE_PROBE_STATE_HASH
    ):
        raise RuntimeError("score probe clone oracle changed")
    return {
        "score_events": score_events,
        "clear_actors": clear_actors,
        "click_counts": counts,
        "final": final,
        "state_hash": final_state_hash,
    }


def build_score_probe(
    *, library_path: str | os.PathLike[str] | None = None
) -> tuple[bytes, dict[str, object]]:
    shots = [Shot(*values) for values in SCORE_PROBE_SHOTS]
    data = build_replay(seed=SCORE_PROBE_SEED, frame_count=SCORE_PROBE_FRAMES, shots=shots)
    return data, validate_score_probe(data, library_path=library_path)


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed", type=int)
    source.add_argument("--preset", choices=(SCORE_PROBE_NAME,))
    parser.add_argument("--frames", type=int)
    parser.add_argument("--shot", action="append", default=[], type=parse_shot)
    parser.add_argument("--highest-level", type=int, default=0)
    parser.add_argument("--final-score", type=int, default=0)
    parser.add_argument("--highest-chain", type=int, default=0)
    parser.add_argument("--mode", type=int, default=0)
    parser.add_argument("--library", help="clone library used to generate/validate presets")
    args = parser.parse_args()
    try:
        if args.preset:
            if args.frames is not None or args.shot or any(
                (args.highest_level, args.final_score, args.highest_chain, args.mode)
            ):
                parser.error("preset controls frames, actions, mode, and zeroed outcome metadata")
            data, oracle = build_score_probe(library_path=args.library)
            seed = SCORE_PROBE_SEED
            frame_count = SCORE_PROBE_FRAMES
            shot_count = sum(oracle["click_counts"].values())
        else:
            if args.frames is None:
                parser.error("--frames is required with --seed")
            if args.library is not None:
                parser.error("--library is only used with --preset")
            data = build_replay(
                seed=args.seed,
                frame_count=args.frames,
                shots=args.shot,
                highest_level=args.highest_level,
                final_score=args.final_score,
                highest_chain=args.highest_chain,
                mode=args.mode,
            )
            oracle = None
            seed = args.seed
            frame_count = args.frames
            shot_count = len(args.shot)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    write_atomic(args.output, data)
    report = {
        "frame_count": frame_count,
        "output": str(args.output),
        "seed": seed,
        "sha256": hashlib.sha256(data).hexdigest(),
        "shot_count": shot_count,
        "size_bytes": len(data),
    }
    if args.preset:
        report.update(
            {
                "preset": args.preset,
                "status": "original_observed_score_parity",
                "clone_oracle": oracle,
            }
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
