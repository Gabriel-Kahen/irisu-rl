#!/usr/bin/env python3
"""Parse and summarize IriSu Syndrome replay input traces.

The implementation is independent of the unlicensed public Kaitai schema.  It
is based on the v2.03 executable analysis documented in
``reference/binary-analysis.md``.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


HEADER = struct.Struct("<5i")
FRAME = struct.Struct("<I")
PADDING_SIZE = 0x20
PADDED_FRAME_OFFSET = HEADER.size + PADDING_SIZE
X_MASK = (1 << 10) - 1
Y_MASK = (1 << 9) - 1
RESERVED_MASK = (1 << 11) - 1
MODE_NAMES = {0: "normal", 1: "Metsu"}

Layout = Literal["auto", "legacy", "padded"]
ResolvedLayout = Literal["legacy", "padded"]


@dataclass(frozen=True, slots=True)
class ReplayHeader:
    seed: int
    highest_level: int
    final_score: int
    highest_chain: int
    mode: int


@dataclass(frozen=True, slots=True)
class InputFrame:
    raw_word: int
    left: bool
    right: bool
    x: int
    y: int
    reserved: int


@dataclass(frozen=True, slots=True)
class Replay:
    header: ReplayHeader
    layout: ResolvedLayout
    frame_offset: int
    layout_reason: str
    padding: bytes
    frames: tuple[InputFrame, ...]


def decode_frame(word: int) -> InputFrame:
    """Decode one unsigned 32-bit input word without discarding high bits."""

    if not 0 <= word <= 0xFFFFFFFF:
        raise ValueError("frame word must fit in uint32")
    return InputFrame(
        raw_word=word,
        left=bool(word & 1),
        right=bool((word >> 1) & 1),
        x=(word >> 2) & X_MASK,
        y=(word >> 12) & Y_MASK,
        reserved=(word >> 21) & RESERVED_MASK,
    )


def encode_frame(
    *, left: bool = False, right: bool = False, x: int = 0, y: int = 0, reserved: int = 0
) -> int:
    """Encode one frame, primarily for clean-room fixtures and trace tools."""

    for name, value, maximum in (
        ("x", x, X_MASK),
        ("y", y, Y_MASK),
        ("reserved", reserved, RESERVED_MASK),
    ):
        if type(value) is not int or not 0 <= value <= maximum:
            raise ValueError(f"{name} must be an integer in [0, {maximum}]")
    return (
        (reserved << 21)
        | (y << 12)
        | (x << 2)
        | (int(bool(right)) << 1)
        | int(bool(left))
    )


def _resolve_frame_offset(data: bytes, layout: Layout) -> tuple[int, ResolvedLayout, str]:
    """Resolve a requested layout; auto mode intentionally remains heuristic."""

    if layout == "legacy":
        return HEADER.size, "legacy", "forced legacy layout"
    if layout == "padded":
        return PADDED_FRAME_OFFSET, "padded", "forced padded layout"
    if layout != "auto":
        raise ValueError(f"unknown replay layout: {layout!r}")

    padding = data[HEADER.size:PADDED_FRAME_OFFSET]
    if len(padding) == PADDING_SIZE and padding == bytes(PADDING_SIZE):
        return (
            PADDED_FRAME_OFFSET,
            "padded",
            "32 zero bytes after header; padded layout assumed",
        )
    return HEADER.size, "legacy", "no v2.03-style zero block; legacy layout assumed"


def choose_frame_offset(data: bytes, layout: Layout) -> tuple[int, str]:
    """Return the offset and rationale, preserving the original helper API."""

    offset, _, reason = _resolve_frame_offset(data, layout)
    return offset, reason


def parse_replay(data: bytes, layout: Layout = "auto") -> Replay:
    """Parse replay bytes, rejecting truncated headers and partial frames."""

    if len(data) < HEADER.size:
        raise ValueError(f"file is shorter than the {HEADER.size}-byte header")

    header = ReplayHeader(*HEADER.unpack_from(data))
    if header.mode not in MODE_NAMES:
        raise ValueError(f"unsupported replay mode {header.mode}; expected 0 or 1")

    frame_offset, resolved_layout, reason = _resolve_frame_offset(data, layout)
    if len(data) < frame_offset:
        raise ValueError(
            f"{resolved_layout} layout requires at least {frame_offset} bytes; got {len(data)}"
        )

    frame_bytes = data[frame_offset:]
    if len(frame_bytes) % FRAME.size:
        raise ValueError(f"{len(frame_bytes)} frame bytes is not divisible by four")

    padding = data[HEADER.size:frame_offset] if resolved_layout == "padded" else b""
    frames = tuple(decode_frame(word) for (word,) in struct.iter_unpack("<I", frame_bytes))
    return Replay(header, resolved_layout, frame_offset, reason, padding, frames)


def inspect(path: Path, layout: Layout = "auto", include_frames: bool = False) -> dict[str, object]:
    data = path.read_bytes()
    replay = parse_replay(data, layout)
    clicked = [frame for frame in replay.frames if frame.left or frame.right]
    stats: dict[str, object] = {
        "left_click_frames": sum(frame.left for frame in replay.frames),
        "right_click_frames": sum(frame.right for frame in replay.frames),
        "simultaneous_click_frames": sum(frame.left and frame.right for frame in replay.frames),
        "nonzero_reserved_frames": sum(bool(frame.reserved) for frame in replay.frames),
        "first_click_frame": next(
            (index for index, frame in enumerate(replay.frames) if frame.left or frame.right),
            None,
        ),
        "clicked_x_range": [min(f.x for f in clicked), max(f.x for f in clicked)]
        if clicked
        else None,
        "clicked_y_range": [min(f.y for f in clicked), max(f.y for f in clicked)]
        if clicked
        else None,
    }
    report: dict[str, object] = {
        "path": str(path),
        "size_bytes": len(data),
        "header": {
            "seed": replay.header.seed,
            "highest_level": replay.header.highest_level,
            "final_score": replay.header.final_score,
            "highest_chain": replay.header.highest_chain,
            "mode": replay.header.mode,
            "mode_name": MODE_NAMES[replay.header.mode],
        },
        "layout": {
            "requested": layout,
            "resolved": replay.layout,
            "frame_offset": replay.frame_offset,
            "reason": replay.layout_reason,
            "padding_hex": replay.padding.hex(),
            "warning": (
                "Auto-detection is heuristic: a legacy trace whose first eight frames are zero "
                "is indistinguishable from the padded form. Force --layout when provenance is known."
            ),
        },
        "frame_count": len(replay.frames),
        "input_stats": stats,
    }
    if include_frames:
        report["frames"] = [
            {
                "index": index,
                "raw_word": frame.raw_word,
                "left": frame.left,
                "right": frame.right,
                "x": frame.x,
                "y": frame.y,
                "reserved": frame.reserved,
            }
            for index, frame in enumerate(replay.frames)
        ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("replay", type=Path)
    parser.add_argument("--layout", choices=("auto", "legacy", "padded"), default="auto")
    parser.add_argument(
        "--include-frames",
        action="store_true",
        help="include every decoded frame (large for normal gameplay traces)",
    )
    args = parser.parse_args()

    try:
        report = inspect(args.replay, args.layout, args.include_frames)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
