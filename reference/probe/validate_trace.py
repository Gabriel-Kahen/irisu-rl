#!/usr/bin/env python3
"""Validate the shipped Box2D wrapper probe's JSONL schema and key invariants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPORTS = {
    "b2d_init",
    "b2d_dispose",
    "b2d_create_box",
    "b2d_create_triangle",
    "b2d_create_circle",
    "b2d_destroy_body",
    "b2d_step",
    "b2d_get_contact",
    "b2d_get_x",
    "b2d_get_y",
    "b2d_get_r",
    "b2d_get_v",
    "b2d_set_v",
    "b2d_set_user_data",
    "b2d_set_position",
    "b2d_test",
}


def close(actual: float, expected: float, tolerance: float = 1e-5) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, got {actual}")


def one(rows: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    matches = [r for r in rows if all(r.get(k) == v for k, v in fields.items())]
    if len(matches) != 1:
        raise AssertionError(f"expected one row matching {fields}, got {len(matches)}")
    return matches[0]


def validate(path: Path) -> int:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise AssertionError(f"invalid JSON on line {line_number}: {error}") from error
            if not isinstance(row, dict):
                raise AssertionError(f"line {line_number} is not a JSON object")
            rows.append(row)

    if not rows:
        raise AssertionError("trace is empty")
    if [r.get("seq") for r in rows] != list(range(len(rows))):
        raise AssertionError("seq values are not contiguous and zero-based")
    if any(r.get("type") == "error" for r in rows):
        raise AssertionError("trace contains an error record")

    meta = one(rows, type="meta")
    assert meta["schema"] == 2
    assert meta["architecture"] == "x86"
    exports = [r for r in rows if r.get("type") == "export"]
    assert {r["name"] for r in exports} == EXPORTS
    assert all(r["resolved"] is True for r in exports)

    inits = [r for r in rows if r.get("type") == "init"]
    assert {r["scenario"] for r in inits} == {
        "transforms",
        "fall_and_contacts",
        "restitution",
        "friction_response",
        "sleep_timing",
        "triangle_orientation",
        "dimension_skin",
        "gravity_g100_m100",
        "gravity_g250_m100",
        "gravity_g100_m50",
        "gravity_gminus250_m100",
        "contact_ordering",
    }
    assert all(r["result"] == 1 for r in inits)

    bodies = [r for r in rows if r.get("type") == "body"]
    bit_fields = {"x_bits", "y_bits", "r_bits", "vx_bits", "vy_bits"}
    assert all(bit_fields <= r.keys() for r in bodies)

    set_velocity = one(rows, type="body", scenario="transforms", id=101,
                       shape="box_after_set_v")
    close(set_velocity["vx_world"], 2.5)
    close(set_velocity["vy_world"], -5.0)
    stepped = one(rows, type="body", scenario="transforms", id=101,
                  shape="box_after_step")
    close(stepped["x"], 128.0)
    close(stepped["y"], 224.0)
    positioned = one(rows, type="body", scenario="transforms", id=101,
                     shape="box_after_set_position")
    close(positioned["x"], 321.0)
    close(positioned["y"], 222.0)
    close(positioned["r"], -0.5)
    close(positioned["vx_world"], 0.0)
    close(positioned["vy_world"], 0.0)

    gravity_tick = one(rows, type="body", scenario="fall_and_contacts", tick=1,
                       id=201)
    close(gravity_tick["y"], 100.04, tolerance=2e-5)
    close(gravity_tick["vy_world"], 0.02)
    contacts = [r for r in rows if r.get("type") == "contact" and
                r.get("scenario") == "fall_and_contacts"]
    pairs = {(r["user_a"], r["user_b"]) for r in contacts}
    assert {(9001, 201), (9001, 202), (9001, 203)} <= pairs
    first_by_body = {
        body: min(r["tick"] for r in contacts if r["user_b"] == body)
        for body in (201, 202, 203)
    }
    assert first_by_body == {201: 96, 202: 95, 203: 95}

    rested_y = {body: one(rows, type="body", scenario="fall_and_contacts",
                          tick=200, id=body)["y"] for body in (201, 202, 203)}
    close(rested_y[201], 280.500183, tolerance=5e-5)
    close(rested_y[202], 278.500061, tolerance=5e-5)
    close(rested_y[203], 275.500153, tolerance=5e-5)

    rebound = one(rows, type="body", scenario="restitution", tick=20, id=301)
    close(rebound["vx_world"], -5.0)
    wall_contact = one(rows, type="contact", scenario="restitution")
    assert (wall_contact["tick"], wall_contact["user_a"], wall_contact["user_b"]) == (
        19, 9002, 301)

    friction_at_20 = {
        body: one(rows, type="body", scenario="friction_response", tick=20,
                  id=body) for body in (401, 402, 403, 404)
    }
    close(friction_at_20[401]["vx_world"], 1.0)
    close(friction_at_20[402]["vx_world"], 0.8)
    close(friction_at_20[403]["vx_world"], 0.6)
    assert friction_at_20[402]["vx_bits"] == friction_at_20[404]["vx_bits"]

    sleep_control_24 = one(rows, type="body", scenario="sleep_timing", tick=24,
                           id=505)
    sleep_control_25 = one(rows, type="body", scenario="sleep_timing", tick=25,
                           id=505)
    sleep_control_26 = one(rows, type="body", scenario="sleep_timing", tick=26,
                           id=505)
    state_fields = ("x_bits", "y_bits", "r_bits", "vx_bits", "vy_bits")
    assert tuple(sleep_control_24[k] for k in state_fields) != tuple(
        sleep_control_25[k] for k in state_fields)
    assert tuple(sleep_control_25[k] for k in state_fields) == tuple(
        sleep_control_26[k] for k in state_fields)
    awake_24 = one(rows, type="body", scenario="sleep_timing", tick=24,
                   id=501, shape="probe_23")
    asleep_25 = one(rows, type="body", scenario="sleep_timing", tick=25,
                    id=503, shape="probe_25")
    asleep_26 = one(rows, type="body", scenario="sleep_timing", tick=26,
                    id=503, shape="probe_25")
    close(awake_24["x"], -398.0)
    close(asleep_25["vx_world"], 1.0)
    assert tuple(asleep_25[k] for k in state_fields) == tuple(
        asleep_26[k] for k in state_fields)

    triangle_contacts = [r for r in rows if r.get("type") == "contact" and
                         r.get("scenario") == "triangle_orientation"]
    assert [r["user_b"] for r in triangle_contacts] == [607, 605, 602, 601]
    assert all(r["user_a"] == 9301 for r in triangle_contacts)

    dimension_contacts = [r for r in rows if r.get("type") == "contact" and
                          r.get("scenario") == "dimension_skin"]
    first_dimension_contact = {
        body: min(r["tick"] for r in dimension_contacts if r["user_b"] == body)
        for body in (801, 802, 803, 804)
    }
    assert first_dimension_contact == {801: 96, 802: 93, 803: 96, 804: 93}
    dimension_y = {
        body: one(rows, type="body", scenario="dimension_skin", tick=200,
                  id=body)["y"] for body in (801, 802, 803, 804)
    }
    close(dimension_y[801], 280.500122, tolerance=5e-5)
    close(dimension_y[802], 270.499969, tolerance=5e-5)
    close(dimension_y[803], 280.500061, tolerance=5e-5)
    close(dimension_y[804], 270.500031, tolerance=5e-5)

    gravity_100_100 = one(rows, type="body", scenario="gravity_g100_m100",
                          tick=1, id=901)
    gravity_250_100 = one(rows, type="body", scenario="gravity_g250_m100",
                          tick=1, id=902)
    gravity_100_50 = one(rows, type="body", scenario="gravity_g100_m50",
                         tick=1, id=903)
    gravity_minus = one(rows, type="body",
                        scenario="gravity_gminus250_m100", tick=1, id=904)
    close(gravity_100_100["y"], 100.04, tolerance=2e-5)
    close(gravity_100_100["vy_world"], 0.02)
    close(gravity_250_100["y"], 100.1, tolerance=2e-5)
    close(gravity_250_100["vy_world"], 0.05)
    assert gravity_100_100["y_bits"] == gravity_100_50["y_bits"]
    close(gravity_100_50["vy_world"], 0.04)
    close(gravity_minus["y"], 99.9, tolerance=2e-5)
    close(gravity_minus["vy_world"], -0.05)

    ordering = [r for r in rows if r.get("type") == "contact" and
                r.get("scenario") == "contact_ordering"]
    by_tick = {
        tick: [r["user_b"] for r in ordering if r["tick"] == tick]
        for tick in (1, 2, 3)
    }
    assert by_tick == {
        1: [703, 702, 701],
        2: [704, 703, 702, 701],
        3: [706, 705, 704, 703, 702, 701],
    }

    disposals = [r for r in rows if r.get("type") == "lifecycle" and
                 r.get("operation") == "dispose"]
    assert {r["scenario"] for r in disposals} == {
        "transforms",
        "fall_and_contacts",
        "restitution",
        "friction_response",
        "sleep_timing",
        "triangle_orientation",
        "dimension_skin",
        "gravity_g100_m100",
        "gravity_g250_m100",
        "gravity_g100_m50",
        "gravity_gminus250_m100",
        "contact_ordering",
    }
    one(rows, type="lifecycle", scenario="transforms", operation="destroy_null")
    one(rows, type="lifecycle", scenario="probe", operation="FreeLibrary")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    args = parser.parse_args()
    try:
        count = validate(args.trace)
    except (AssertionError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(f"validated {count} JSONL records: {args.trace}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
