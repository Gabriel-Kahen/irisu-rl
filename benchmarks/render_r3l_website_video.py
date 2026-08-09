#!/usr/bin/env python3
"""Render the verified R3L exhibition in the playable website's visual style."""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from PIL import Image, ImageColor, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "artifacts/r3/development/r3l-current-model-10k-exhibition-20260808-003"
)
V3_SOURCE = ROOT / "benchmarks/rl_r3l_current_model_10k_v3.py"
TEST_SOURCE = ROOT / "tests/test_render_r3l_website_video.py"
WEB_APP = ROOT / "apps/web/static/app.js"
WEB_COLORS = ROOT / "apps/web/static/colors.mjs"
WEB_STYLES = ROOT / "apps/web/static/styles.css"
OUTPUT = RUN_ROOT / "current-model-10000plus-website.mp4"
MANIFEST = RUN_ROOT / "website-video-manifest.json"

EXPECTED_EPISODE_ID = (
    "f1eaf54b82d1c62141229a2b344804a3045e70ae49f9ba7128dda807099e2f4a"
)
EXPECTED_VERIFICATION_ID = (
    "b28ada6d6148490c1b9e4667342a6746b64600e8c7f726776001a311213fe1ff"
)
EXPECTED_ACTION_SHA = (
    "929e888c30bc3ef192a8e5cdac171fbfdd5ee4e715c09ee8f22428a8eb849953"
)

PAGE_WIDTH = 840
PAGE_HEIGHT = 680
CANVAS_X = 21.0
CANVAS_Y = 56.0
CANVAS_WIDTH = 798.0
CANVAS_HEIGHT = 598.0
FPS = 60
PALETTE = (
    "#861f00", "#0005a4", "#9a9000", "#b227b5",
    "#52aba7", "#ae6311", "#1b747a", "#92335f",
)
ACTIVATED = (
    "#e44717", "#2945ff", "#eee116", "#b227b5",
    "#52aba7", "#ef9c27", "#35bdc4", "#e35b98",
)
BONUS = ACTIVATED[:5]
TRAIL_ALPHAS = (0.08, 0.13, 0.2, 0.3)
INTER_REGULAR = Path("/home/gabe/.local/share/fonts/inter/InterVariable.ttf")
INTER_ITALIC = Path("/home/gabe/.local/share/fonts/inter/InterVariable-Italic.ttf")
SERIF_BLACK = Path("/usr/share/fonts/noto/NotoSerif-Black.ttf")


def _load_v3() -> ModuleType:
    name = "irisu_r3l_exhibition_v3_for_website_render"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, V3_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen exhibition-v3 helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V3 = _load_v3()
BASE = V3.BASE


def _fmt(value: object) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("website renderer received a non-finite number")
    return f"{number:.5f}".rstrip("0").rstrip(".") or "0"


def _color(body: Mapping[str, object], time_ms: float) -> str:
    kind = body["kind"]
    if kind == "projectile":
        return "#d9dcda"
    if kind == "bonus":
        return BONUS[int(time_ms // 400) % len(BONUS)]
    index = int(body["color"]) % len(PALETTE)
    activated = kind == "piece" and body["lifecycle"] in {
        "dynamic_fresh", "confirmed",
    }
    return (ACTIVATED if activated else PALETTE)[index]


def _body_shape(body: Mapping[str, object], size: float) -> str:
    half = size / 2.0
    shape = body["shape"]
    if shape == "circle":
        return f'<circle cx="0" cy="0" r="{_fmt(half)}"/>'
    if shape == "triangle":
        points = f"{-half},{-half} {-half},{half} {half},{half}"
        return f'<polygon points="{points}"/>'
    return (
        f'<rect x="{_fmt(-half)}" y="{_fmt(-half)}" '
        f'width="{_fmt(size)}" height="{_fmt(size)}"/>'
    )


def _body_svg(
    body: Mapping[str, object], pose: Mapping[str, object], color: str, alpha: float
) -> str:
    size = max(2.0, float(body["size"]))
    degrees = float(pose.get("angle", 0.0)) * 180.0 / math.pi
    return (
        f'<g transform="translate({_fmt(pose["x"])} {_fmt(pose["y"])}) '
        f'rotate({_fmt(degrees)})" fill="{color}" opacity="{_fmt(alpha)}">'
        f'{_body_shape(body, size)}</g>'
    )


def update_trails(
    trails: dict[int, list[dict[str, float]]], bodies: Sequence[Mapping[str, object]]
) -> None:
    active: set[int] = set()
    for body in bodies:
        if body["kind"] != "piece" or body["lifecycle"] != "confirmed":
            continue
        identifier = int(body["id"])
        active.add(identifier)
        trail = trails.setdefault(identifier, [])
        trail.append(
            {
                "x": float(body["x"]),
                "y": float(body["y"]),
                "angle": float(body.get("angle", 0.0)),
            }
        )
        if len(trail) > len(TRAIL_ALPHAS) + 1:
            del trail[0]
    for identifier in tuple(trails):
        if identifier not in active:
            del trails[identifier]


def interpolated_bodies(
    previous: Mapping[str, object] | None,
    current: Mapping[str, object],
    video_time_ms: float,
) -> list[Mapping[str, object]]:
    current_bodies = list(current["bodies"])
    if previous is None or int(previous["tick"]) == int(current["tick"]):
        return current_bodies
    gap = int(current["tick"]) - int(previous["tick"])
    if gap <= 0 or gap > 4:
        return current_bodies
    snapshot_time_ms = int(current["tick"]) * 20.0
    alpha = min(1.0, max(0.0, (video_time_ms - snapshot_time_ms) / min(60.0, max(20.0, gap * 20.0))))
    old_bodies = {int(body["id"]): body for body in previous["bodies"]}
    rendered: list[Mapping[str, object]] = []
    for body in current_bodies:
        old = old_bodies.get(int(body["id"]))
        if old is None:
            rendered.append(body)
            continue
        angle_delta = (float(body["angle"]) - float(old["angle"])) % (math.pi * 2.0)
        if angle_delta > math.pi:
            angle_delta -= math.pi * 2.0
        if angle_delta < -math.pi:
            angle_delta += math.pi * 2.0
        pose = dict(body)
        pose["x"] = float(old["x"]) + (float(body["x"]) - float(old["x"])) * alpha
        pose["y"] = float(old["y"]) + (float(body["y"]) - float(old["y"])) * alpha
        pose["angle"] = float(old["angle"]) + angle_delta * alpha
        rendered.append(pose)
    return rendered


def website_svg(
    observation: Mapping[str, object],
    trails: Mapping[int, Sequence[Mapping[str, object]]],
    aim: tuple[int, int] | None,
    *,
    video_time_ms: float | None = None,
) -> str:
    tick = int(observation["tick"])
    render_time_ms = tick * 20.0 if video_time_ms is None else video_time_ms
    bodies = sorted(observation["bodies"], key=lambda body: int(body["id"]))
    body_rows: list[str] = []
    for body in bodies:
        identifier = int(body["id"])
        if body["kind"] == "piece" and body["lifecycle"] == "confirmed":
            echoes = list(trails.get(identifier, ()))[:-1]
            offset = len(TRAIL_ALPHAS) - len(echoes)
            for index, pose in enumerate(echoes):
                body_rows.append(
                    _body_svg(body, pose, _color(body, render_time_ms), TRAIL_ALPHAS[offset + index])
                )
        body_rows.append(
            _body_svg(
                body,
                body,
                _color(body, render_time_ms),
                0.62 if body["lifecycle"] == "scripted_falling" else 1.0,
            )
        )

    field = observation["field"]
    field_x, field_y = float(field["x"]), float(field["y"])
    field_w, field_h = float(field["width"]), float(field["height"])
    gauge_ratio = max(
        0.0,
        min(1.0, float(observation["gauge"]) / float(observation["gauge_max"])),
    )
    aim_row = ""
    if aim is not None:
        x, y = aim
        aim_row = f"""
        <g fill="none" stroke="#ece8dd" stroke-width="1">
          <circle cx="{x}" cy="{y}" r="9"/>
          <path d="M{x-15} {y}H{x-5}M{x+5} {y}H{x+15}M{x} {y-15}V{y-5}M{x} {y+5}V{y+15}"/>
        </g>"""
    score = str(int(observation["score"])).zfill(8)
    level = str(int(observation["level"]))
    scale_x, scale_y = CANVAS_WIDTH / 640.0, CANVAS_HEIGHT / 480.0
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_WIDTH}" height="{PAGE_HEIGHT}" viewBox="0 0 {PAGE_WIDTH} {PAGE_HEIGHT}">
  <defs>
    <linearGradient id="gauge" x1="0" x2="1"><stop offset="0" stop-color="#7c1b31"/><stop offset="1" stop-color="#b02a3f"/></linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="160%"><feDropShadow dx="0" dy="20" stdDeviation="30" flood-color="#000" flood-opacity=".72"/></filter>
    <clipPath id="canvas-clip"><rect x="0" y="0" width="640" height="480"/></clipPath>
  </defs>
  <rect width="840" height="680" fill="#070a0b"/>
  <text x="20" y="38" fill="#eee9df" font-family="Inter,system-ui,sans-serif" font-size="24" font-weight="700" letter-spacing="-1">irisu</text>
  <g font-family="Inter,system-ui,sans-serif" text-transform="uppercase">
    <rect x="665" y="17" width="76" height="24" fill="#101617" fill-opacity=".67" stroke="#303838"/>
    <text x="675" y="32" fill="#b4b8b5" font-size="9">PAUSE</text><text x="716" y="32" fill="#858c8c" font-size="7">SPACE</text>
    <rect x="749" y="17" width="70" height="24" fill="#101617" fill-opacity=".67" stroke="#303838"/>
    <text x="759" y="32" fill="#b4b8b5" font-size="9">RESTART</text><text x="808" y="32" fill="#858c8c" font-size="7">R</text>
  </g>
  <rect x="20" y="55" width="800" height="600" fill="#080d0f" stroke="#364142" filter="url(#shadow)"/>
  <g transform="translate({_fmt(CANVAS_X)} {_fmt(CANVAS_Y)}) scale({_fmt(scale_x)} {_fmt(scale_y)})" clip-path="url(#canvas-clip)">
    <rect width="640" height="480" fill="#0c1517"/>
    <g fill="#f3f3ef">
      <rect x="{_fmt(field_x)}" y="{_fmt(field_y)}" width="16" height="{_fmt(field_h)}"/>
      <rect x="{_fmt(field_x + field_w + 8)}" y="{_fmt(field_y)}" width="16" height="{_fmt(field_h)}"/>
      <rect x="{_fmt(field_x)}" y="{_fmt(field_y + field_h + 40)}" width="{_fmt(field_w + 32)}" height="16"/>
      <rect x="{_fmt(field_x + 16)}" y="0" width="{_fmt(field_w)}" height="10"/>
    </g>
    <g fill="#cad0cd" opacity=".333">
      <rect x="{_fmt(field_x)}" y="{_fmt(field_y)}" width="3" height="{_fmt(field_h)}"/>
      <rect x="{_fmt(field_x + field_w + 8)}" y="{_fmt(field_y)}" width="3" height="{_fmt(field_h)}"/>
    </g>
    {''.join(body_rows)}
    <g font-family="Trebuchet MS,sans-serif" font-style="italic" font-weight="900" paint-order="stroke" stroke="#55152c" stroke-width="5" stroke-linejoin="round" fill="#f0e2a6">
      <text x="21" y="428" font-size="28">Level</text>
      <text x="52" y="458" font-size="25" text-anchor="middle">{level}</text>
    </g>
    <rect x="151" y="437" width="312" height="15" fill="#33161e" fill-opacity=".667"/>
    <rect x="151" y="437" width="{_fmt(312 * gauge_ratio)}" height="15" fill="url(#gauge)"/>
    <rect x="151" y="437" width="312" height="3" fill="#fff" fill-opacity=".094"/>
    <text x="320" y="462" text-anchor="middle" dominant-baseline="middle" font-family="Georgia,serif" font-size="26" font-weight="900" paint-order="stroke" stroke="#681a38" stroke-width="5" fill="#eee0a4">{score}</text>
    {aim_row}
  </g>
  <rect x="25" y="60" width="790" height="590" fill="none" stroke="#0b1011" stroke-width="8" opacity=".8" pointer-events="none"/>
</svg>"""


@functools.lru_cache(maxsize=32)
def _font(path: Path, size: int, weight: float | None = None) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(path, size=size)
    if weight is not None:
        try:
            font.set_variation_by_axes([weight])
        except (AttributeError, OSError, ValueError):
            pass
    return font


@functools.lru_cache(maxsize=1)
def _static_page() -> Image.Image:
    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "#070a0b")
    shadow = Image.new("RGBA", page.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.rectangle((35, 75, 805, 665), fill=(0, 0, 0, 190))
    shadow = shadow.filter(ImageFilter.GaussianBlur(25))
    page.paste(shadow, (0, 0), shadow)
    draw = ImageDraw.Draw(page)
    draw.text((20, 16), "irisu", font=_font(INTER_REGULAR, 24, 700), fill="#eee9df")
    draw.rectangle((665, 17, 741, 41), fill="#101617", outline="#303838", width=1)
    draw.text((675, 23), "PAUSE", font=_font(INTER_REGULAR, 9), fill="#b4b8b5")
    draw.text((716, 25), "SPACE", font=_font(INTER_REGULAR, 7), fill="#858c8c")
    draw.rectangle((749, 17, 819, 41), fill="#101617", outline="#303838", width=1)
    draw.text((759, 23), "RESTART", font=_font(INTER_REGULAR, 9), fill="#b4b8b5")
    draw.text((808, 25), "R", font=_font(INTER_REGULAR, 7), fill="#858c8c")
    draw.rectangle((20, 55, 819, 654), fill="#080d0f", outline="#364142", width=1)
    return page


@functools.lru_cache(maxsize=1)
def _gauge_gradient(width: int, height: int) -> Image.Image:
    gradient = Image.new("RGB", (width, height), "#7c1b31")
    draw = ImageDraw.Draw(gradient)
    for x in range(width):
        amount = x / max(1, width - 1)
        color = (
            round(124 + (176 - 124) * amount),
            round(27 + (42 - 27) * amount),
            round(49 + (63 - 49) * amount),
        )
        draw.line((x, 0, x, height), fill=color)
    return gradient


def _page_point(x: float, y: float) -> tuple[float, float]:
    return (
        CANVAS_X + x * CANVAS_WIDTH / 640.0,
        CANVAS_Y + y * CANVAS_HEIGHT / 480.0,
    )


def _draw_body_pillow(
    page: Image.Image,
    body: Mapping[str, object],
    pose: Mapping[str, object],
    color: str,
    alpha: float,
) -> None:
    draw = ImageDraw.Draw(page, "RGBA")
    size = max(2.0, float(body["size"]))
    half = size / 2.0
    cx, cy = float(pose["x"]), float(pose["y"])
    angle = float(pose.get("angle", 0.0))
    fill = ImageColor.getrgb(color) + (round(255 * alpha),)
    if body["shape"] == "circle":
        left, top = _page_point(cx - half, cy - half)
        right, bottom = _page_point(cx + half, cy + half)
        draw.ellipse((left, top, right, bottom), fill=fill)
    else:
        local = (
            ((-half, -half), (-half, half), (half, half))
            if body["shape"] == "triangle"
            else ((-half, -half), (half, -half), (half, half), (-half, half))
        )
        cosine, sine = math.cos(angle), math.sin(angle)
        points = [
            _page_point(cx + x * cosine - y * sine, cy + x * sine + y * cosine)
            for x, y in local
        ]
        draw.polygon(points, fill=fill)


def website_image(
    observation: Mapping[str, object],
    trails: Mapping[int, Sequence[Mapping[str, object]]],
    aim: tuple[int, int] | None,
    *,
    video_time_ms: float,
) -> Image.Image:
    chrome = _static_page().copy()
    page = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "#0c1517")
    draw = ImageDraw.Draw(page)
    draw.rectangle(
        (CANVAS_X, CANVAS_Y, CANVAS_X + CANVAS_WIDTH, CANVAS_Y + CANVAS_HEIGHT),
        fill="#0c1517",
    )
    field = observation["field"]
    fx, fy = float(field["x"]), float(field["y"])
    fw, fh = float(field["width"]), float(field["height"])

    def rectangle(x: float, y: float, width: float, height: float, fill: str) -> None:
        left, top = _page_point(x, y)
        right, bottom = _page_point(x + width, y + height)
        draw.rectangle((left, top, right, bottom), fill=fill)

    rectangle(fx, fy, 16, fh, "#f3f3ef")
    rectangle(fx + fw + 8, fy, 16, fh, "#f3f3ef")
    rectangle(fx, fy + fh + 40, fw + 32, 16, "#f3f3ef")
    rectangle(fx + 16, 0, fw, 10, "#f3f3ef")
    wall_draw = ImageDraw.Draw(page, "RGBA")
    for wall_x in (fx, fx + fw + 8):
        left, top = _page_point(wall_x, fy)
        right, bottom = _page_point(wall_x + 3, fy + fh)
        wall_draw.rectangle((left, top, right, bottom), fill=(202, 208, 205, 85))

    for body in sorted(observation["bodies"], key=lambda row: int(row["id"])):
        identifier = int(body["id"])
        if body["kind"] == "piece" and body["lifecycle"] == "confirmed":
            echoes = list(trails.get(identifier, ()))[:-1]
            offset = len(TRAIL_ALPHAS) - len(echoes)
            for index, pose in enumerate(echoes):
                _draw_body_pillow(
                    page, body, pose, _color(body, video_time_ms),
                    TRAIL_ALPHAS[offset + index],
                )
        _draw_body_pillow(
            page, body, body, _color(body, video_time_ms),
            0.62 if body["lifecycle"] == "scripted_falling" else 1.0,
        )

    scale = CANVAS_HEIGHT / 480.0
    level_font = _font(INTER_ITALIC, round(28 * scale), 900)
    level_number_font = _font(INTER_ITALIC, round(25 * scale), 900)
    level_xy = _page_point(21, 428)
    draw.text(
        level_xy, "Level", font=level_font, anchor="ls", fill="#f0e2a6",
        stroke_width=round(5 * scale), stroke_fill="#55152c",
    )
    number_xy = _page_point(52, 458)
    draw.text(
        number_xy, str(int(observation["level"])), font=level_number_font,
        anchor="ms", fill="#f0e2a6", stroke_width=round(5 * scale),
        stroke_fill="#55152c",
    )
    track_left, track_top = _page_point(151, 437)
    track_right, track_bottom = _page_point(463, 452)
    draw.rectangle((track_left, track_top, track_right, track_bottom), fill="#33161e")
    ratio = max(0.0, min(1.0, float(observation["gauge"]) / float(observation["gauge_max"])))
    gauge_width = max(0, round((track_right - track_left) * ratio))
    if gauge_width:
        full_width = round(track_right - track_left)
        height = max(1, round(track_bottom - track_top))
        gradient = _gauge_gradient(full_width, height).crop((0, 0, gauge_width, height))
        page.paste(gradient, (round(track_left), round(track_top)))
    highlight_bottom = _page_point(151, 440)[1]
    ImageDraw.Draw(page, "RGBA").rectangle(
        (track_left, track_top, track_right, highlight_bottom), fill=(255, 255, 255, 24)
    )
    score_xy = _page_point(320, 462)
    draw.text(
        score_xy, str(int(observation["score"])).zfill(8),
        font=_font(SERIF_BLACK, round(26 * scale)), anchor="mm", fill="#eee0a4",
        stroke_width=round(5 * scale), stroke_fill="#681a38",
    )
    if aim is not None:
        x, y = _page_point(*aim)
        radius = 9 * scale
        cursor = ImageDraw.Draw(page)
        cursor.ellipse((x - radius, y - radius, x + radius, y + radius), outline="#ece8dd", width=1)
        for x1, y1, x2, y2 in (
            (x - 15 * scale, y, x - 5 * scale, y),
            (x + 5 * scale, y, x + 15 * scale, y),
            (x, y - 15 * scale, x, y - 5 * scale),
            (x, y + 5 * scale, x, y + 15 * scale),
        ):
            cursor.line((x1, y1, x2, y2), fill="#ece8dd", width=1)
    clip_box = (
        round(CANVAS_X), round(CANVAS_Y),
        round(CANVAS_X + CANVAS_WIDTH), round(CANVAS_Y + CANVAS_HEIGHT),
    )
    chrome.paste(page.crop(clip_box), clip_box[:2])
    ImageDraw.Draw(chrome).rectangle((25, 60, 815, 650), outline="#0b1011", width=8)
    return chrome


def _load_evidence() -> tuple[dict[str, Any], dict[str, Any], tuple[int, ...]]:
    V3._validate(RUN_ROOT)
    episode = BASE._read_json(RUN_ROOT / "episode.json")
    verification = BASE._read_json(RUN_ROOT / "verification.json")
    BASE._verify_self_hash(episode, "website-render episode")
    BASE._verify_self_hash(verification, "website-render verification")
    if (
        episode["sha256"] != EXPECTED_EPISODE_ID
        or verification["sha256"] != EXPECTED_VERIFICATION_ID
        or verification.get("verified") is not True
        or episode["action_file_sha256"] != EXPECTED_ACTION_SHA
    ):
        raise RuntimeError("website-render evidence identity differs")
    data = (RUN_ROOT / str(episode["action_file"])).read_bytes()
    if hashlib.sha256(data).hexdigest() != EXPECTED_ACTION_SHA:
        raise RuntimeError("website-render action trace differs")
    words = tuple(word for (word,) in struct.iter_unpack("<I", data))
    if len(words) != episode["action_count"]:
        raise RuntimeError("website-render action count differs")
    return episode, verification, words


def render() -> dict[str, object]:
    episode, verification, words = _load_evidence()
    if MANIFEST.exists():
        manifest = BASE._read_json(MANIFEST)
        BASE._verify_self_hash(manifest, "website video manifest")
        if BASE._sha256_file(OUTPUT) != manifest["video_sha256"]:
            raise RuntimeError("website video file differs")
        return manifest
    screen = BASE._load_screen()
    core, campaign = screen._load_external()
    expected = {int(row["tick"]): row for row in episode["checkpoints"]}
    trails: dict[int, list[dict[str, float]]] = {}
    aim: tuple[int, int] | None = None
    frame_count = math.floor(len(words) * 20 * FPS / 1000) + 1
    temporary_output = RUN_ROOT / f".{OUTPUT.name}.{os.getpid()}.tmp.mp4"
    encoder = subprocess.Popen(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-s:v", f"{PAGE_WIDTH}x{PAGE_HEIGHT}",
            "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264",
            "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", str(temporary_output),
        ],
        stdin=subprocess.PIPE,
        cwd=ROOT,
    )
    try:
        if encoder.stdin is None:
            raise RuntimeError("website video encoder has no input stream")
        with campaign.IrisuEnv(
            library_path=screen.RUNTIME,
            physics_backend="portable",
            config={"max_episode_ticks": BASE.MAX_TICKS},
        ) as env:
            current, _ = env.reset(seed=BASE.SEED)
            previous: Mapping[str, object] | None = None
            action_index = 0
            last_report = 0
            for frame_index in range(frame_count):
                video_time_ms = frame_index * 1000.0 / FPS
                target_tick = min(len(words), (frame_index * 1000) // (FPS * 20))
                while action_index < target_tick:
                    word = words[action_index]
                    buttons = word & 3
                    if buttons:
                        aim = ((word >> 2) & 0x3FF, (word >> 12) & 0x1FF)
                    previous = current
                    current, _reward, _terminated, _truncated, _info = env.step(
                        BASE.decode_action_word(core, word)
                    )
                    action_index += 1
                    update_trails(trails, current["bodies"])
                    tick = int(current["tick"])
                    checkpoint = expected.get(tick)
                    if checkpoint is not None and BASE._public_checkpoint(env, current) != checkpoint:
                        raise RuntimeError(f"website render replay diverged at tick {tick}")
                    if tick // 2_500 > last_report:
                        last_report = tick // 2_500
                        print(
                            json.dumps(
                                {"event": "website-render-progress", "tick": tick,
                                 "score": int(current["score"]), "frame": frame_index},
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                display = dict(current)
                display["bodies"] = interpolated_bodies(previous, current, video_time_ms)
                encoder.stdin.write(
                    website_image(
                        display, trails, aim, video_time_ms=video_time_ms
                    ).tobytes()
                )
            if action_index != len(words):
                raise RuntimeError("website renderer did not consume the full action trace")
            observation = current
            final = BASE._public_checkpoint(env, observation)
            snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
        if final != episode["final"] or snapshot_sha != episode["final_snapshot_sha256"]:
            raise RuntimeError("website render final replay closure differs")
        encoder.stdin.close()
        return_code = encoder.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, encoder.args)
        print(
            json.dumps(
                {"event": "website-encoding-complete", "frames": frame_count},
                sort_keys=True,
            ),
            flush=True,
        )
        os.replace(temporary_output, OUTPUT)
    except BaseException:
        if encoder.stdin is not None and not encoder.stdin.closed:
            encoder.stdin.close()
        if encoder.poll() is None:
            encoder.kill()
        encoder.wait()
        if temporary_output.exists():
            temporary_output.unlink()
        raise
    manifest = BASE._with_sha(
        {
            "schema": "irisu-r3l-website-style-video-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "renderer": "faithful offline port of the playable website canvas and page chrome",
            "website_source_sha256": {
                str(path): BASE._sha256_file(path)
                for path in (WEB_APP, WEB_COLORS, WEB_STYLES)
            },
            "renderer_source_sha256": BASE._sha256_file(Path(__file__).resolve()),
            "renderer_test_sha256": BASE._sha256_file(TEST_SOURCE),
            "font_sha256": {
                str(path): BASE._sha256_file(path)
                for path in (INTER_REGULAR, INTER_ITALIC, SERIF_BLACK)
            },
            "episode_sha256": episode["sha256"],
            "verification_sha256": verification["sha256"],
            "action_file_sha256": episode["action_file_sha256"],
            "video": OUTPUT.name,
            "video_sha256": BASE._sha256_file(OUTPUT),
            "width": PAGE_WIDTH,
            "height": PAGE_HEIGHT,
            "fps": FPS,
            "frame_count": frame_count,
            "duration_seconds": frame_count / FPS,
            "simulation_snapshot_hz": 50,
            "website_interpolation": "60fps virtual time with the app.js 20ms pose interpolation",
            "cursor": "last recorded model shot coordinate",
            "final": final,
        }
    )
    BASE._write_new(MANIFEST, manifest)
    return manifest


def sample(tick: int, output: Path) -> None:
    _episode, _verification, words = _load_evidence()
    if not 0 <= tick <= len(words):
        raise ValueError("sample tick is outside the recorded episode")
    screen = BASE._load_screen()
    core, campaign = screen._load_external()
    trails: dict[int, list[dict[str, float]]] = {}
    aim: tuple[int, int] | None = None
    with campaign.IrisuEnv(
        library_path=screen.RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": BASE.MAX_TICKS},
    ) as env:
        observation, _ = env.reset(seed=BASE.SEED)
        for word in words[:tick]:
            if word & 3:
                aim = ((word >> 2) & 0x3FF, (word >> 12) & 0x1FF)
            observation, *_ = env.step(BASE.decode_action_word(core, word))
            update_trails(trails, observation["bodies"])
    if output.suffix.lower() == ".png":
        website_image(
            observation, trails, aim, video_time_ms=int(observation["tick"]) * 20.0
        ).save(output)
    else:
        output.write_text(website_svg(observation, trails, aim))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("render")
    sample_parser = sub.add_parser("sample")
    sample_parser.add_argument("--tick", type=int, required=True)
    sample_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "sample":
        sample(args.tick, args.output)
        value: object = {"tick": args.tick, "output": str(args.output)}
    else:
        value = render()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
