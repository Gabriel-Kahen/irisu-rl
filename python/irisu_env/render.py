"""Original-asset-free deterministic diagnostic rendering."""

from __future__ import annotations

import math
from html import escape
from typing import Any, Mapping


_PALETTE = ("#55c2ff", "#ff6b75", "#ffd166", "#70d48b", "#ba8cff", "#ff9f50", "#5dd4c0", "#ef7bd2")


def _number(value: object) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "0"
    if abs(number) < 0.0000005:
        number = 0.0
    return f"{number:.6f}".rstrip("0").rstrip(".")


def _body_color(body: Mapping[str, Any]) -> str:
    kind = body.get("kind")
    if kind == "projectile":
        return "#f2f5f7"
    if kind == "bonus":
        return "#ffe66d"
    return _PALETTE[int(body.get("color", 0)) % len(_PALETTE)]


def render_svg(observation: Mapping[str, Any]) -> str:
    """Render one observation to stable, self-contained SVG text."""

    gauge_max = max(1, int(observation.get("gauge_max", 10_000)))
    gauge = max(0, min(gauge_max, int(observation.get("gauge", 0))))
    gauge_height = 350.0 * gauge / gauge_max
    field = observation.get("field", {})
    field_x = float(field.get("x", 94.0))
    field_y = float(field.get("y", 20.0))
    field_width = float(field.get("width", 420.0))
    field_height = float(field.get("height", 370.0))
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480">',
        '<rect width="640" height="480" fill="#111820"/>',
        f'<rect x="{_number(field_x)}" y="{_number(field_y)}" '
        f'width="{_number(field_width)}" height="{_number(field_height)}" '
        'fill="#18232d" stroke="#8aa0b2" stroke-width="2"/>',
        '<rect x="34" y="100" width="20" height="350" fill="#263540" stroke="#8aa0b2"/>',
        f'<rect x="34" y="{_number(450.0 - gauge_height)}" width="20" height="{_number(gauge_height)}" fill="#70d48b"/>',
    ]

    bodies = sorted(observation.get("bodies", ()), key=lambda body: int(body["id"]))
    for body in bodies:
        body_id = int(body["id"])
        x = float(body["x"])
        y = float(body["y"])
        size = max(1.0, float(body["size"]))
        half = size * 0.5
        angle = math.degrees(float(body.get("angle", 0.0)))
        color = _body_color(body)
        lifecycle = escape(str(body.get("lifecycle", "unknown")), quote=True)
        common = (
            f'data-id="{body_id}" data-lifecycle="{lifecycle}" fill="{color}" '
            'stroke="#071015" stroke-width="1.5"'
        )
        shape = body.get("shape")
        if shape == "circle":
            lines.append(
                f'<circle cx="{_number(x)}" cy="{_number(y)}" r="{_number(half)}" {common}/>'
            )
        elif shape == "triangle":
            # Measured wrapper fixture: upper-left, lower-left, lower-right.
            points = (
                f"{_number(-half)},{_number(-half)} "
                f"{_number(-half)},{_number(half)} "
                f"{_number(half)},{_number(half)}"
            )
            lines.append(
                f'<polygon points="{points}" transform="translate({_number(x)} {_number(y)}) rotate({_number(angle)})" {common}/>'
            )
        else:
            lines.append(
                f'<rect x="{_number(x - half)}" y="{_number(y - half)}" '
                f'width="{_number(size)}" height="{_number(size)}" '
                f'transform="rotate({_number(angle)} {_number(x)} {_number(y)})" {common}/>'
            )

    tick = int(observation.get("tick", 0))
    score = int(observation.get("score", 0))
    level = int(observation.get("level", 1))
    lines.append(
        f'<text x="82" y="458" fill="#dbe7ef" font-family="monospace" font-size="14">'
        f'tick {tick}  score {score}  level {level}</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"
