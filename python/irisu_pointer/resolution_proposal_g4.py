"""Dual-head, seed-equal proposal learning over the hardened G3R3 core."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import stat
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from irisu_rl.encoding import TeacherStateEncoder

from . import resolution_first_g3r3 as _core
from . import resolution_first_g2 as _g2
from .resolution_first import FEATURE_NAMES
from .resolution_first_g2 import BoardBranchRecord, BoardResolutionDataset
from .resolution_first_g3r3 import (
    BoostConfigR3,
    HistogramNewtonBoostR3,
    WIDE_FEATURE_NAMES,
    WIDE_FEATURE_WIDTH,
    WideBoard,
    wide_board_from_records,
)


MODEL_SCHEMA = "irisu-r3i-resolution-proposal-g4-v2"
CHECKPOINT_SCHEMA = "irisu-r3i-resolution-proposal-checkpoint-g4-v2"
PARTITION_SCHEMA = "irisu-r3i-resolution-proposal-partition-g4-v2"
ABSOLUTE_SAFE_FORMULA = (
    "candidate_resolved&&!exact_unsafe&&!severe_unsafe&&finite(b2)&&b2>=0"
)
SOLVENCY_FORMULA = (
    "absolute_safe*(0.5+0.5*tanh(b2/"
    "(1800+20*min(max(level,0),99))))"
)
GROWTH_FORMULA = (
    "absolute_safe*I(b2>=reserve)*tanh(max(score_advantage,0)/64)"
)
PROPOSAL_POLICY = (
    "regime-active-ceil-half-then-other-floor-half-then-"
    "active-other-alternating-dedupe-fill-v2"
)
INFERENCE_SCHEMA = "irisu-r3i-resolution-public-query-g4-v1"
FEATURE_INVENTORY_SCHEMA = "irisu-r3i-resolution-feature-inventory-g4-v1"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_DEPENDENCY_RE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
REQUIRED_DEPENDENCIES = frozenset(
    {
        "g4-source",
        "g4-tests",
        "g3r3-source",
        "g3r3-tests",
        "g3r2-features",
        "g2-board",
        "g4-protocol",
    }
)
_LEVEL_INDEX = TeacherStateEncoder.schema.global_features.index("level_log1p")
_ID_BODY_COLUMNS = tuple(
    TeacherStateEncoder.schema.body_features.index(name)
    for name in ("id_scaled", "chain_id_scaled")
)
_FORBIDDEN_PREEXACT_KEYS = frozenset(
    {
        "target",
        "targets",
        "outcome",
        "outcomes",
        "ledger",
        "certificate",
        "certificates",
        "b2",
        "b2_margin",
        "delta_b2",
        "score_advantage",
        "candidate_resolved",
        "finite_pair",
        "exact_unsafe",
        "severe_unsafe",
        "source_sha256",
    }
)
_CLIENT_WIDTH = 640.0
_CLIENT_HEIGHT = 480.0
_GEOMETRY = (
    ("analytic-strong", "strong", 0.50, 0.75),
    ("close-strong", "strong", 0.25, 0.75),
    ("wide-strong", "strong", 0.75, 0.75),
    ("deep-strong", "strong", 0.50, 1.00),
    ("analytic-weak", "weak", 0.50, 0.75),
)
_CATEGORY_INTENT = {
    "rotten-hazard": "match_rotten",
    "viable-anchor": "extend_anchor",
    "fresh-match": "steer_match",
}
_OBSERVATION_KEYS = {
    "bodies",
    "difficulty",
    "field",
    "gauge",
    "gauge_max",
    "highest_chain",
    "left_held",
    "level",
    "qualifying_clear_count",
    "right_held",
    "score",
    "terminated",
    "tick",
    "truncated",
}
_DIFFICULTY_KEYS = {"active_colors", "spawn_interval_ticks"}
_FIELD_KEYS = {"height", "side_wall_bottom", "side_wall_top", "width", "x", "y"}
_BODY_KEYS = {
    "age_ticks",
    "angle",
    "angular_velocity",
    "chain_id",
    "color",
    "id",
    "kind",
    "lifecycle",
    "projectile_hits",
    "remaining_lifetime",
    "rot_timer",
    "shape",
    "size",
    "vx",
    "vy",
    "x",
    "y",
}


def _sha256(value: object) -> str:
    return hashlib.sha256(_core._canonical_bytes(value)).hexdigest()


def _exact_str(value: Any, field: str, *, nonempty: bool = True) -> str:
    if type(value) is not str or (nonempty and not value):
        raise RuntimeError(f"{field} must be an exact string")
    return value


def _same_manifest(left: object, right: object) -> bool:
    try:
        return _core._canonical_bytes(left) == _core._canonical_bytes(right)
    except (RuntimeError, TypeError, ValueError):
        return False


def _exact_dependencies(value: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    if type(value) is not dict or set(value) != REQUIRED_DEPENDENCIES:
        raise ValueError("G4 dependencies differ from the required identity set")
    output: list[tuple[str, str]] = []
    for name, digest in value.items():
        if (
            type(name) is not str
            or _DEPENDENCY_RE.fullmatch(name) is None
            or type(digest) is not str
            or _SHA256_RE.fullmatch(digest) is None
        ):
            raise ValueError("G4 dependency identity is malformed")
        output.append((name, digest))
    return tuple(sorted(output))


def _finite(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


@dataclass(frozen=True, slots=True)
class G4CandidateIdentity:
    seed: int
    query_id: str
    ordinal: int
    candidate_id: str
    action_id: str

    def __post_init__(self) -> None:
        if (
            type(self.seed) is not int
            or self.seed < 0
            or type(self.query_id) is not str
            or not self.query_id
            or type(self.ordinal) is not int
            or self.ordinal < 0
            or type(self.candidate_id) is not str
            or _SHA256_RE.fullmatch(self.candidate_id) is None
            or type(self.action_id) is not str
            or _SHA256_RE.fullmatch(self.action_id) is None
        ):
            raise ValueError("G4 candidate identity is malformed")

    @property
    def query_key(self) -> tuple[int, str]:
        return self.seed, self.query_id

    def manifest(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "query_id": self.query_id,
            "ordinal": self.ordinal,
            "candidate_id": self.candidate_id,
            "action_id": self.action_id,
        }

    @classmethod
    def from_manifest(cls, value: Any) -> G4CandidateIdentity:
        raw = _core._exact_dict(value, "G4 candidate identity")
        if set(raw) != {
            "seed",
            "query_id",
            "ordinal",
            "candidate_id",
            "action_id",
        }:
            raise RuntimeError("G4 candidate identity is malformed")
        try:
            result = cls(
                _core._exact_int(raw["seed"], "seed"),
                _exact_str(raw["query_id"], "query_id"),
                _core._exact_int(raw["ordinal"], "ordinal"),
                _exact_str(raw["candidate_id"], "candidate_id"),
                _exact_str(raw["action_id"], "action_id"),
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError("G4 candidate identity is malformed") from exc
        if not _same_manifest(result.manifest(), raw):
            raise RuntimeError("G4 candidate identity is malformed")
        return result


def _preexact_bytes(value: object) -> bytes:
    def validate(item: object, field: str) -> None:
        kind = type(item)
        if item is None or kind in (bool, int, str):
            if kind is str:
                try:
                    item.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise ValueError(f"{field} is not UTF-8") from exc
            return
        if kind is float:
            if not math.isfinite(item):
                raise ValueError(f"{field} is nonfinite")
            return
        if kind is list:
            for index, child in enumerate(item):
                validate(child, f"{field}[{index}]")
            return
        if kind is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError(f"{field} has a non-string key")
                validate(child, f"{field}.{key}")
            return
        raise ValueError(f"{field} has inexact type {kind.__name__}")

    validate(value, "$")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _preexact_sha256(value: object) -> str:
    return hashlib.sha256(_preexact_bytes(value)).hexdigest()


def _reject_exact_fields(value: object, field: str = "$") -> None:
    if type(value) is dict:
        for key, child in value.items():
            if key in _FORBIDDEN_PREEXACT_KEYS:
                raise ValueError(f"{field}.{key} is forbidden before exact evaluation")
            _reject_exact_fields(child, f"{field}.{key}")
    elif type(value) is list:
        for index, child in enumerate(value):
            _reject_exact_fields(child, f"{field}[{index}]")


def _plain_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{field} must be an exact integer >= {minimum}")
    return value


def _plain_float(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{field} must be an exact finite float")
    return value


def _public_candidate_manifest(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("G4 public candidate must be an exact dictionary")
    required = {
        "ordinal",
        "pair_ordinal",
        "geometry_ordinal",
        "pair",
        "geometry",
        "action",
    }
    if set(value) != required:
        raise ValueError("G4 public candidate keys differ from joint-v2")
    pair = value["pair"]
    geometry = value["geometry"]
    action = value["action"]
    if (
        type(pair) is not dict
        or set(pair)
        != {
            "source_body_id",
            "destination_body_id",
            "destination_chain_id",
            "category",
            "intent",
            "distance_sizes",
            "incumbent",
        }
        or type(geometry) is not dict
        or set(geometry)
        != {"name", "strength", "side_sizes", "below_sizes"}
        or type(action) is not dict
        or set(action) != {"kind", "x_norm", "y_norm"}
    ):
        raise ValueError("G4 public candidate sections differ from joint-v2")
    _plain_int(value["ordinal"], "candidate ordinal")
    _plain_int(value["pair_ordinal"], "pair ordinal")
    _plain_int(value["geometry_ordinal"], "geometry ordinal")
    source = _plain_int(pair["source_body_id"], "source body")
    destination = _plain_int(pair["destination_body_id"], "destination body")
    if source == destination:
        raise ValueError("G4 public candidate endpoints must differ")
    _plain_int(pair["destination_chain_id"], "destination chain")
    if (
        type(pair["category"]) is not str
        or pair["category"] not in _CATEGORY_INTENT
        or type(pair["intent"]) is not str
        or pair["intent"] != _CATEGORY_INTENT[pair["category"]]
        or type(pair["incumbent"]) is not bool
    ):
        raise ValueError("G4 public pair semantics are malformed")
    _plain_float(pair["distance_sizes"], "pair distance")
    if (
        type(geometry["name"]) is not str
        or geometry["name"] not in {row[0] for row in _GEOMETRY}
        or type(geometry["strength"]) is not str
        or geometry["strength"] not in {"weak", "strong"}
        or (
            geometry["name"].endswith("-weak")
            != (geometry["strength"] == "weak")
        )
    ):
        raise ValueError("G4 public geometry semantics are malformed")
    for name in ("side_sizes", "below_sizes"):
        if _plain_float(geometry[name], f"geometry {name}") <= 0.0:
            raise ValueError("G4 public geometry scales must be positive")
    action_kind = _plain_int(action["kind"], "action kind", minimum=1)
    if action_kind not in {1, 2} or action_kind != (
        1 if geometry["strength"] == "weak" else 2
    ):
        raise ValueError("G4 public action must be a weak or strong shot")
    for name in ("x_norm", "y_norm"):
        coordinate = _plain_float(action[name], f"action {name}")
        if not 0.0 <= coordinate <= 1.0:
            raise ValueError("G4 public action coordinate is outside [0,1]")
    canonical = json.loads(_preexact_bytes(value).decode("utf-8"))
    if type(canonical) is not dict:
        raise RuntimeError("G4 public candidate canonicalization failed")
    return canonical


def _candidate_identity_from_public(
    seed: int, query_id: str, candidate: Mapping[str, object]
) -> G4CandidateIdentity:
    stable = {
        key: value
        for key, value in candidate.items()
        if key not in {"ordinal", "pair_ordinal", "geometry_ordinal"}
    }
    return G4CandidateIdentity(
        seed,
        query_id,
        _plain_int(candidate["ordinal"], "candidate ordinal"),
        _preexact_sha256(stable),
        _preexact_sha256(candidate["action"]),
    )


def _public_bodies(
    observation: Mapping[str, object],
) -> dict[int, Mapping[str, object]]:
    raw = observation.get("bodies")
    if type(raw) is not list or not raw:
        raise ValueError("G4 public observation bodies must be a nonempty list")
    output: dict[int, Mapping[str, object]] = {}
    for item in raw:
        if type(item) is not dict:
            raise ValueError("G4 public body must be an exact dictionary")
        identifier = _plain_int(item.get("id"), "public body id")
        if identifier in output:
            raise ValueError("G4 public body identities are duplicated")
        output[identifier] = item
    return output


def _validate_public_observation_shape(
    observation: Mapping[str, object],
) -> None:
    difficulty = observation.get("difficulty")
    field = observation.get("field")
    if (
        type(observation) is not dict
        or set(observation) != _OBSERVATION_KEYS
        or type(difficulty) is not dict
        or set(difficulty) != _DIFFICULTY_KEYS
        or type(field) is not dict
        or set(field) != _FIELD_KEYS
        or any(
            type(observation[name]) is not int
            or observation[name] not in {0, 1}
            for name in ("left_held", "right_held", "terminated", "truncated")
        )
    ):
        raise ValueError("G4 public observation schema differs from train-board")
    bodies = observation["bodies"]
    if (
        type(bodies) is not list
        or not bodies
        or any(type(body) is not dict or set(body) != _BODY_KEYS for body in bodies)
    ):
        raise ValueError("G4 public body schema differs from train-board")
    for name in (
        "tick",
        "score",
        "gauge",
        "gauge_max",
        "level",
        "highest_chain",
        "qualifying_clear_count",
    ):
        _plain_int(observation[name], f"observation {name}")
    assert type(difficulty) is dict and type(field) is dict
    _plain_int(difficulty["active_colors"], "active colors", minimum=1)
    _plain_int(difficulty["spawn_interval_ticks"], "spawn interval", minimum=1)
    for name in _FIELD_KEYS:
        _plain_float(field[name], f"field {name}")
    for body in bodies:
        assert type(body) is dict
        for name in (
            "id",
            "chain_id",
            "projectile_hits",
            "age_ticks",
            "remaining_lifetime",
            "rot_timer",
        ):
            _plain_int(body[name], f"body {name}")
        _plain_int(body["color"], "body color", minimum=-1)
        for name in ("x", "y", "vx", "vy", "angle", "angular_velocity", "size"):
            _plain_float(body[name], f"body {name}")
        if (
            type(body["kind"]) is not str
            or type(body["shape"]) is not str
            or type(body["lifecycle"]) is not str
            or not body["kind"]
            or not body["shape"]
            or not body["lifecycle"]
        ):
            raise ValueError("G4 public body categorical field is malformed")


def _position(
    body: Mapping[str, object],
) -> tuple[float, float, float]:
    size = _number(body.get("size"), "body size")
    if size <= 0.0:
        raise ValueError("G4 public body size must be positive")
    return (
        _number(body.get("effect_x", body.get("x")), "body x"),
        _number(body.get("effect_y", body.get("y")), "body y"),
        size,
    )


def _validate_public_candidate_inventory(
    observation: Mapping[str, object],
    candidates: tuple[dict[str, object], ...],
) -> None:
    if len(candidates) > 15:
        raise ValueError("G4 public candidate inventory exceeds joint-v2 cap")
    bodies = _public_bodies(observation)
    pair_ordinals = tuple(
        _plain_int(candidate["pair_ordinal"], "pair ordinal")
        for candidate in candidates
    )
    if (
        pair_ordinals[0] != 0
        or tuple(dict.fromkeys(pair_ordinals))
        != tuple(range(max(pair_ordinals) + 1))
        or max(pair_ordinals) > 2
        or tuple(sorted(pair_ordinals)) != pair_ordinals
    ):
        raise ValueError("G4 public pair ordinals differ from joint-v2")
    grouped: dict[int, list[dict[str, object]]] = {}
    for candidate in candidates:
        grouped.setdefault(
            _plain_int(candidate["pair_ordinal"], "pair ordinal"), []
        ).append(candidate)
    endpoint_pairs: set[tuple[int, int]] = set()
    for pair_ordinal, rows in grouped.items():
        pair_manifests = [row["pair"] for row in rows]
        if any(type(pair) is not dict for pair in pair_manifests):
            raise RuntimeError("validated G4 public pair disappeared")
        if any(pair != pair_manifests[0] for pair in pair_manifests[1:]):
            raise ValueError("G4 public pair group metadata differs")
        geometry_ordinals = tuple(
            _plain_int(row["geometry_ordinal"], "geometry ordinal")
            for row in rows
        )
        if (
            not geometry_ordinals
            or geometry_ordinals[0] != 0
            or tuple(sorted(set(geometry_ordinals))) != geometry_ordinals
            or geometry_ordinals[-1] >= len(_GEOMETRY)
        ):
            raise ValueError("G4 public geometry ordinals differ from joint-v2")
        pair = pair_manifests[0]
        assert type(pair) is dict
        if pair["incumbent"] is not (pair_ordinal == 0):
            raise ValueError("G4 public incumbent pair binding differs")
        source_id = _plain_int(pair["source_body_id"], "source body")
        destination_id = _plain_int(pair["destination_body_id"], "destination body")
        endpoints = (source_id, destination_id)
        if endpoints in endpoint_pairs:
            raise ValueError("G4 public pair endpoints repeat across pair groups")
        endpoint_pairs.add(endpoints)
        try:
            source = bodies[source_id]
            destination = bodies[destination_id]
        except KeyError as exc:
            raise ValueError("G4 public pair endpoint is absent") from exc
        destination_chain = _plain_int(
            destination.get("chain_id", 0), "observed destination chain"
        )
        if pair["destination_chain_id"] != destination_chain:
            raise ValueError("G4 public destination chain binding differs")
        sx, sy, source_size = _position(source)
        dx, dy, destination_size = _position(destination)
        distance = math.hypot(dx - sx, dy - sy) / max(
            (source_size + destination_size) / 2.0, 1e-9
        )
        if pair["distance_sizes"] != distance:
            raise ValueError("G4 public pair distance binding differs")
        direction = 1.0 if dx > sx else -1.0
        if math.isclose(dx, sx, abs_tol=1e-9):
            direction = 1.0 if sx <= _CLIENT_WIDTH / 2.0 else -1.0
        lifecycle = source.get("lifecycle", "")
        velocity = source.get("vx_display_per_second")
        if velocity is None:
            velocity = _number(source.get("vx"), "source vx") * (
                50.0
                if lifecycle in {"scripted_falling", "falling"}
                else 10.0
            )
        else:
            velocity = _number(velocity, "source display vx")
        for row in rows:
            geometry_ordinal = _plain_int(
                row["geometry_ordinal"], "geometry ordinal"
            )
            geometry = row["geometry"]
            action = row["action"]
            if type(geometry) is not dict or type(action) is not dict:
                raise RuntimeError("validated G4 candidate section disappeared")
            expected_geometry = _GEOMETRY[geometry_ordinal]
            supplied_geometry = (
                geometry["name"],
                geometry["strength"],
                geometry["side_sizes"],
                geometry["below_sizes"],
            )
            if supplied_geometry != expected_geometry:
                raise ValueError("G4 public geometry table binding differs")
            side = float(expected_geometry[2])
            below = float(expected_geometry[3])
            expected_x = (
                sx + float(velocity) / 50.0 - direction * side * source_size
            ) / _CLIENT_WIDTH
            expected_y = (sy + below * source_size) / _CLIENT_HEIGHT
            if (
                not 0.0 <= expected_x <= 1.0
                or not 0.0 <= expected_y <= 1.0
                or action["x_norm"] != expected_x
                or action["y_norm"] != expected_y
            ):
                raise ValueError("G4 public action lowering differs from joint-v2")
    if (
        candidates[0]["pair_ordinal"] != 0
        or candidates[0]["geometry_ordinal"] != 0
    ):
        raise ValueError("G4 public candidate zero differs from joint-v2")


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _public_body_features(
    body: Mapping[str, object], observation: Mapping[str, object]
) -> tuple[float, ...]:
    field = observation.get("field")
    if type(field) is not dict:
        raise ValueError("G4 public field must be an exact dictionary")
    width = max(_number(field.get("width", 1280.0), "field width"), 1.0)
    height = max(_number(field.get("height", 720.0), "field height"), 1.0)
    return (
        _number(body.get("effect_x", body.get("x", 0.0)), "body effect x")
        / width,
        _number(body.get("effect_y", body.get("y", 0.0)), "body effect y")
        / height,
        _number(body.get("vx", 0.0), "body vx") / 100.0,
        _number(body.get("vy", 0.0), "body vy") / 100.0,
        _number(body.get("size", 0.0), "body size") / max(width, height),
        _number(body.get("rot_timer", 0), "body rot timer") / 41.0,
        _number(
            body.get("remaining_lifetime", 0), "body remaining lifetime"
        )
        / 1000.0,
        float(_plain_int(body.get("chain_id", 0), "body chain") != 0),
    )


def _public_branch_features(
    observation: Mapping[str, object],
    candidate: Mapping[str, object],
) -> tuple[float, ...]:
    bodies = _public_bodies(observation)
    pair = candidate["pair"]
    geometry = candidate["geometry"]
    if type(pair) is not dict or type(geometry) is not dict:
        raise RuntimeError("validated G4 candidate sections disappeared")
    source_id = _plain_int(pair["source_body_id"], "source body")
    destination_id = _plain_int(pair["destination_body_id"], "destination body")
    try:
        source = bodies[source_id]
        destination = bodies[destination_id]
    except KeyError as exc:
        raise ValueError("G4 public candidate endpoint is absent") from exc
    gauge_max = max(_plain_int(observation.get("gauge_max"), "gauge max", minimum=1), 1)
    gauge = _plain_int(observation.get("gauge"), "gauge")
    if gauge > gauge_max:
        raise ValueError("G4 public gauge exceeds its maximum")
    level = _plain_int(observation.get("level"), "level")
    difficulty = observation.get("difficulty")
    if type(difficulty) is not dict:
        raise ValueError("G4 public difficulty must be an exact dictionary")
    visible_debt = sum(
        one_rot_liability(level)
        for body in bodies.values()
        if body.get("kind") == "piece"
        and body.get("lifecycle") != "rotten"
        and _plain_int(body.get("rot_timer", 0), "body rot timer") > 0
    )
    source_x = _number(
        source.get("effect_x", source.get("x", 0.0)), "source x"
    )
    destination_x = _number(
        destination.get("effect_x", destination.get("x", 0.0)),
        "destination x",
    )
    direction = 1.0 if destination_x > source_x else -1.0
    if math.isclose(destination_x, source_x, abs_tol=1e-9):
        direction = 1.0 if source_x <= _CLIENT_WIDTH / 2.0 else -1.0
    category = pair["category"]
    geometry_name = geometry["name"]
    score = _plain_int(observation.get("score"), "score")
    clears = _plain_int(
        observation.get("qualifying_clear_count"), "qualifying clears"
    )
    values = (
        gauge / gauge_max,
        float(gauge > gauge_max / 2),
        min(level, 99) / 99.0,
        math.log1p(score) / 12.0,
        math.log1p(clears) / 8.0,
        len(bodies) / 196.0,
        visible_debt / gauge_max,
        _number(
            difficulty.get("spawn_interval_ticks", 0),
            "spawn interval",
        )
        / 100.0,
        *_public_body_features(source, observation),
        *_public_body_features(destination, observation),
        _plain_float(pair["distance_sizes"], "pair distance") / 20.0,
        -direction * _plain_float(geometry["side_sizes"], "geometry side"),
        _plain_float(geometry["below_sizes"], "geometry below"),
        float(geometry["strength"] == "weak"),
        float(category == "rotten-hazard"),
        float(category == "viable-anchor"),
        float(category == "fresh-match"),
        float(geometry_name == "analytic-strong"),
        float(geometry_name == "close-strong"),
        float(geometry_name == "wide-strong"),
        float(geometry_name == "deep-strong"),
        float(geometry_name == "analytic-weak"),
        float(pair["incumbent"]),
    )
    if len(values) != len(FEATURE_NAMES) or not all(
        math.isfinite(value) for value in values
    ):
        raise RuntimeError("G4 public branch feature extraction failed")
    return tuple(float(value) for value in values)


@dataclass(frozen=True, slots=True)
class _G4PublicRow:
    identity: G4CandidateIdentity
    candidate_features: tuple[float, ...]
    global_features: tuple[float, ...]
    phase_features: tuple[float, float, float]
    body_features: tuple[tuple[float, ...], ...]
    body_chain_groups: tuple[int, ...]
    body_grouped_flags: tuple[bool, ...]
    body_color_groups: tuple[int, ...]
    source_index: int
    destination_index: int
    incumbent_source_index: int
    incumbent_destination_index: int
    observation_sha256: str

    @property
    def seed(self) -> int:
        return self.identity.seed

    @property
    def query_id(self) -> str:
        return self.identity.query_id

    @property
    def query_key(self) -> tuple[int, str]:
        return self.identity.query_key

    @property
    def ordinal(self) -> int:
        return self.identity.ordinal

    @property
    def features(self) -> tuple[float, ...]:
        return self.candidate_features

    @property
    def model_global_features(self) -> tuple[float, ...]:
        values = list(self.global_features)
        values[0] = 0.0
        return (*values, *self.phase_features)


def _wide_features_from_public_rows(
    rows: tuple[_G4PublicRow, ...],
) -> np.ndarray:
    dataset = BoardResolutionDataset(rows)  # type: ignore[arg-type]
    (
        global_features,
        bodies,
        mask,
        candidate,
        incumbent,
        source,
        destination,
        incumbent_pair,
    ) = dataset.tensors()
    active = bodies.masked_fill(~mask[:, :, None], 0.0)
    for column in _ID_BODY_COLUMNS:
        if bool((active[:, :, column] != 0).any()):
            raise RuntimeError("absolute body identity leaked into G4 inference")
    if bool((global_features[:, 0] != 0).any()):
        raise RuntimeError("absolute tick leaked into G4 inference")
    count = mask.sum(1, keepdim=True).float()
    mean = active.sum(1) / count
    variance = (active.square().sum(1) / count - mean.square()).clamp_min(0.0)
    maximum = bodies.masked_fill(~mask[:, :, None], -math.inf).amax(1)
    minimum = bodies.masked_fill(~mask[:, :, None], math.inf).amin(1)
    torch = __import__("torch")
    torch_rows = torch.arange(len(rows))
    endpoints = torch.cat(
        (
            bodies[torch_rows, source],
            bodies[torch_rows, destination],
            bodies[torch_rows, incumbent_pair[:, 0]],
            bodies[torch_rows, incumbent_pair[:, 1]],
            bodies[torch_rows, source] - bodies[torch_rows, destination],
        ),
        dim=1,
    )
    paired = torch.cat((candidate, incumbent, candidate - incumbent), dim=1)
    features = torch.cat(
        (
            global_features,
            paired,
            count / 30.0,
            mean,
            variance.sqrt(),
            minimum,
            maximum,
            endpoints,
        ),
        dim=1,
    ).numpy()
    if features.shape != (len(rows), WIDE_FEATURE_WIDTH) or not np.isfinite(
        features
    ).all():
        raise RuntimeError("G4 public wide feature extraction failed")
    result = np.array(features, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class G4QueryInventory:
    query_key: tuple[int, str]
    level: int
    reserve: int
    observation_sha256: str
    incumbent_identity: G4CandidateIdentity
    identities: tuple[G4CandidateIdentity, ...]

    def __post_init__(self) -> None:
        if (
            type(self.query_key) is not tuple
            or len(self.query_key) != 2
            or type(self.query_key[0]) is not int
            or self.query_key[0] < 0
            or type(self.query_key[1]) is not str
            or not self.query_key[1]
            or type(self.level) is not int
            or self.level < 0
            or type(self.reserve) is not int
            or self.reserve != one_rot_liability(self.level)
            or type(self.observation_sha256) is not str
            or _SHA256_RE.fullmatch(self.observation_sha256) is None
            or type(self.incumbent_identity) is not G4CandidateIdentity
            or self.incumbent_identity.ordinal != 0
            or self.incumbent_identity.query_key != self.query_key
            or type(self.identities) is not tuple
            or not self.identities
            or any(
                type(identity) is not G4CandidateIdentity
                or identity.query_key != self.query_key
                for identity in self.identities
            )
            or tuple(identity.ordinal for identity in self.identities)
            != tuple(range(1, len(self.identities) + 1))
            or len(set(self.identities)) != len(self.identities)
            or len({identity.candidate_id for identity in self.identities})
            != len(self.identities)
            or self.incumbent_identity.candidate_id
            in {identity.candidate_id for identity in self.identities}
        ):
            raise ValueError("G4 query inventory is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3i-resolution-proposal-inventory-g4-v2",
            "query_key": list(self.query_key),
            "level": self.level,
            "reserve": self.reserve,
            "observation_sha256": self.observation_sha256,
            "incumbent_identity": self.incumbent_identity.manifest(),
            "identities": [identity.manifest() for identity in self.identities],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())


def _validate_exact_record(record: BoardBranchRecord) -> None:
    if type(record) is not BoardBranchRecord:
        raise ValueError("G4 supervision requires an exact board branch record")
    if (
        any(
            type(value) is not bool
            for value in (
                record.candidate_resolved,
                record.finite_pair,
                record.exact_unsafe,
                record.severe_unsafe,
            )
        )
        or type(record.seed) is not int
        or record.seed < 0
        or type(record.query_id) is not str
        or not record.query_id
        or type(record.ordinal) is not int
        or record.ordinal < 0
        or type(record.score_advantage) is not float
        or not math.isfinite(record.score_advantage)
        or (
            record.b2 is not None
            and (
                type(record.b2) is not float
                or not math.isfinite(record.b2)
            )
        )
        or (
            record.delta_b2 is not None
            and (
                type(record.delta_b2) is not float
                or not math.isfinite(record.delta_b2)
            )
        )
        or type(record.source_sha256) is not str
        or _SHA256_RE.fullmatch(record.source_sha256) is None
        or type(record.observation_sha256) is not str
        or _SHA256_RE.fullmatch(record.observation_sha256) is None
    ):
        raise ValueError("G4 exact supervision has inexact types")


def level_from_record(record: BoardBranchRecord) -> int:
    _validate_exact_record(record)
    if (
        type(record.global_features) is not tuple
        or len(record.global_features)
        != len(TeacherStateEncoder.schema.global_features)
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in record.global_features
        )
    ):
        raise ValueError("G4 bound query state has inexact global features")
    encoded = record.global_features[_LEVEL_INDEX]
    if encoded < 0.0:
        raise ValueError("G4 bound query level is negative")
    decoded = math.expm1(encoded * 8.0)
    if not math.isfinite(decoded):
        raise ValueError("G4 bound query level is nonfinite")
    level = int(round(decoded))
    rebound = float(np.float32(math.log1p(max(float(level), 0.0)) / 8.0))
    if level < 0 or rebound != encoded:
        raise ValueError("G4 bound query level is not an exact encoded integer")
    return level


def one_rot_liability(level: int) -> int:
    if type(level) is not int:
        raise ValueError("G4 level must be an exact integer")
    return 1800 + 20 * min(max(level, 0), 99)


def absolute_safe(record: BoardBranchRecord) -> bool:
    _validate_exact_record(record)
    b2 = _finite(record.b2)
    return bool(
        record.candidate_resolved
        and not record.exact_unsafe
        and not record.severe_unsafe
        and b2 is not None
        and b2 >= 0.0
    )


def solvency_target(record: BoardBranchRecord) -> float:
    if not absolute_safe(record):
        return 0.0
    assert record.b2 is not None
    reserve = one_rot_liability(level_from_record(record))
    return float(0.5 + 0.5 * math.tanh(record.b2 / reserve))


def growth_target(record: BoardBranchRecord) -> float:
    if not absolute_safe(record):
        return 0.0
    assert record.b2 is not None
    reserve = one_rot_liability(level_from_record(record))
    if record.b2 < reserve:
        return 0.0
    score = record.score_advantage
    return float(math.tanh(max(score, 0.0) / 64.0))


def _feature_inventory_sha256(
    features: np.ndarray,
    identities: tuple[G4CandidateIdentity, ...],
    incumbents: tuple[G4CandidateIdentity, ...],
    levels: np.ndarray,
    observations: tuple[str, ...],
) -> str:
    return _sha256(
        {
            "schema": FEATURE_INVENTORY_SCHEMA,
            "feature_names": list(WIDE_FEATURE_NAMES),
            "rows": [
                {
                    "features": [float(value) for value in features[index]],
                    "identity": identity.manifest(),
                    "level": int(levels[index]),
                    "observation_sha256": observations[index],
                }
                for index, identity in enumerate(identities)
            ],
            "incumbent_identities": [
                identity.manifest() for identity in incumbents
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class G4Board:
    wide: WideBoard
    identities: tuple[G4CandidateIdentity, ...]
    incumbent_identities: tuple[G4CandidateIdentity, ...]
    levels: np.ndarray
    observation_sha256: tuple[str, ...]
    solvency: np.ndarray
    growth: np.ndarray

    def __post_init__(self) -> None:
        if (
            type(self.wide) is not WideBoard
            or type(self.identities) is not tuple
            or type(self.incumbent_identities) is not tuple
            or type(self.levels) is not np.ndarray
            or self.levels.dtype != np.dtype(np.int64)
            or type(self.observation_sha256) is not tuple
            or type(self.solvency) is not np.ndarray
            or self.solvency.dtype != np.dtype(np.float64)
            or type(self.growth) is not np.ndarray
            or self.growth.dtype != np.dtype(np.float64)
        ):
            raise ValueError("G4 board has inexact component types")
        count = len(self.wide.features)
        if (
            len(self.identities) != count
            or self.levels.shape != (count,)
            or len(self.observation_sha256) != count
            or self.solvency.shape != (count,)
            or self.growth.shape != (count,)
            or count == 0
            or np.any(self.levels < 0)
            or any(
                type(value) is not str
                or _SHA256_RE.fullmatch(value) is None
                for value in self.observation_sha256
            )
            or not np.isfinite(self.solvency).all()
            or not np.isfinite(self.growth).all()
            or np.any((self.solvency < 0.0) | (self.solvency > 1.0))
            or np.any((self.growth < 0.0) | (self.growth > 1.0))
            or np.any((self.solvency > 0.0) & ~self.wide.labels)
            or np.any((self.growth > 0.0) & ~self.wide.labels)
            or np.any(self.wide.labels & (self.solvency < 0.5))
        ):
            raise ValueError("G4 board supervision is malformed")
        seen: set[G4CandidateIdentity] = set()
        grouped: dict[
            tuple[int, str], list[tuple[G4CandidateIdentity, int, str]]
        ] = {}
        for index, identity in enumerate(self.identities):
            if type(identity) is not G4CandidateIdentity or identity.ordinal < 1:
                raise ValueError("G4 board identity is malformed")
            if (
                identity.seed != int(self.wide.seeds[index])
                or identity.query_id != str(self.wide.query_ids[index])
                or identity.ordinal != int(self.wide.ordinals[index])
                or identity in seen
            ):
                raise ValueError("G4 board identity binding differs")
            seen.add(identity)
            grouped.setdefault(identity.query_key, []).append(
                (
                    identity,
                    int(self.levels[index]),
                    self.observation_sha256[index],
                )
            )
        for rows in grouped.values():
            identities = tuple(row[0] for row in rows)
            if (
                tuple(identity.ordinal for identity in identities)
                != tuple(range(1, len(identities) + 1))
                or len({identity.candidate_id for identity in identities})
                != len(identities)
                or len({row[1] for row in rows}) != 1
                or len({row[2] for row in rows}) != 1
            ):
                raise ValueError("G4 query inventory is not closed and bound")
        query_keys = tuple(grouped)
        if (
            len(self.incumbent_identities) != len(query_keys)
            or any(
                type(row) is not G4CandidateIdentity or row.ordinal != 0
                for row in self.incumbent_identities
            )
            or tuple(row.query_key for row in self.incumbent_identities)
            != query_keys
            or any(
                incumbent.candidate_id
                in {
                    identity.candidate_id
                    for identity in self.identities
                    if identity.query_key == incumbent.query_key
                }
                for incumbent in self.incumbent_identities
            )
        ):
            raise ValueError("G4 incumbent inventory is not closed and bound")
        levels_copy = np.array(self.levels, dtype=np.int64, copy=True)
        solvency_copy = np.array(self.solvency, dtype=np.float64, copy=True)
        growth_copy = np.array(self.growth, dtype=np.float64, copy=True)
        levels_copy.setflags(write=False)
        solvency_copy.setflags(write=False)
        growth_copy.setflags(write=False)
        object.__setattr__(self, "levels", levels_copy)
        object.__setattr__(self, "solvency", solvency_copy)
        object.__setattr__(self, "growth", growth_copy)

    @property
    def features(self) -> np.ndarray:
        return self.wide.features

    @property
    def seeds(self) -> np.ndarray:
        return self.wide.seeds

    @property
    def unique_seeds(self) -> tuple[int, ...]:
        return self.wide.unique_seeds

    @property
    def query_keys(self) -> tuple[tuple[int, str], ...]:
        return tuple(dict.fromkeys(row.query_key for row in self.identities))

    @property
    def feature_inventory_sha256(self) -> str:
        return _feature_inventory_sha256(
            self.features,
            self.identities,
            self.incumbent_identities,
            self.levels,
            self.observation_sha256,
        )

    def inventory(self, query_key: tuple[int, str]) -> G4QueryInventory:
        if (
            type(query_key) is not tuple
            or len(query_key) != 2
            or type(query_key[0]) is not int
            or type(query_key[1]) is not str
        ):
            raise ValueError("G4 query inventory request is malformed")
        indices = [
            index
            for index, identity in enumerate(self.identities)
            if identity.query_key == query_key
        ]
        if not indices:
            raise ValueError("G4 query inventory is absent")
        return G4QueryInventory(
            query_key,
            int(self.levels[indices[0]]),
            one_rot_liability(int(self.levels[indices[0]])),
            self.observation_sha256[indices[0]],
            next(
                row
                for row in self.incumbent_identities
                if row.query_key == query_key
            ),
            tuple(self.identities[index] for index in indices),
        )

    @property
    def sha256(self) -> str:
        return _sha256(
            {
                "schema": "irisu-r3i-resolution-proposal-board-g4-v2",
                "wide_board_sha256": self.wide.sha256,
                "identities": [row.manifest() for row in self.identities],
                "incumbent_identities": [
                    row.manifest() for row in self.incumbent_identities
                ],
                "levels": [int(value) for value in self.levels],
                "observation_sha256": list(self.observation_sha256),
                "solvency": [float(value) for value in self.solvency],
                "growth": [float(value) for value in self.growth],
                "absolute_safe_formula": ABSOLUTE_SAFE_FORMULA,
                "solvency_formula": SOLVENCY_FORMULA,
                "growth_formula": GROWTH_FORMULA,
            }
        )


@dataclass(frozen=True, slots=True)
class G4InferenceBoard:
    """Label-free, pre-exact candidates in the training feature space."""

    features: np.ndarray
    identities: tuple[G4CandidateIdentity, ...]
    incumbent_identities: tuple[G4CandidateIdentity, ...]
    levels: np.ndarray
    observation_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.features) is not np.ndarray
            or self.features.dtype != np.dtype(np.float64)
            or self.features.ndim != 2
            or self.features.shape[1] != WIDE_FEATURE_WIDTH
            or not np.isfinite(self.features).all()
            or type(self.identities) is not tuple
            or type(self.incumbent_identities) is not tuple
            or type(self.levels) is not np.ndarray
            or self.levels.dtype != np.dtype(np.int64)
            or type(self.observation_sha256) is not tuple
        ):
            raise ValueError("G4 inference board has inexact component types")
        count = len(self.features)
        if (
            count == 0
            or len(self.identities) != count
            or self.levels.shape != (count,)
            or len(self.observation_sha256) != count
            or np.any(self.levels < 0)
            or any(
                type(value) is not str
                or _SHA256_RE.fullmatch(value) is None
                for value in self.observation_sha256
            )
        ):
            raise ValueError("G4 inference inventory is malformed")
        grouped: dict[
            tuple[int, str], list[tuple[G4CandidateIdentity, int, str]]
        ] = {}
        seen: set[G4CandidateIdentity] = set()
        for index, identity in enumerate(self.identities):
            if (
                type(identity) is not G4CandidateIdentity
                or identity.ordinal < 1
                or identity in seen
            ):
                raise ValueError("G4 inference identity is malformed")
            seen.add(identity)
            grouped.setdefault(identity.query_key, []).append(
                (
                    identity,
                    int(self.levels[index]),
                    self.observation_sha256[index],
                )
            )
        for rows in grouped.values():
            alternatives = tuple(row[0] for row in rows)
            if (
                tuple(row.ordinal for row in alternatives)
                != tuple(range(1, len(alternatives) + 1))
                or len({row.candidate_id for row in alternatives})
                != len(alternatives)
                or len({row[1] for row in rows}) != 1
                or len({row[2] for row in rows}) != 1
            ):
                raise ValueError("G4 inference query inventory is not closed")
        query_keys = tuple(grouped)
        if (
            len(self.incumbent_identities) != len(query_keys)
            or any(
                type(row) is not G4CandidateIdentity or row.ordinal != 0
                for row in self.incumbent_identities
            )
            or tuple(row.query_key for row in self.incumbent_identities)
            != query_keys
            or any(
                incumbent.candidate_id
                in {
                    identity.candidate_id
                    for identity in self.identities
                    if identity.query_key == incumbent.query_key
                }
                for incumbent in self.incumbent_identities
            )
        ):
            raise ValueError("G4 inference incumbent inventory is not closed")
        feature_copy = np.array(self.features, dtype=np.float64, copy=True)
        level_copy = np.array(self.levels, dtype=np.int64, copy=True)
        feature_copy.setflags(write=False)
        level_copy.setflags(write=False)
        object.__setattr__(self, "features", feature_copy)
        object.__setattr__(self, "levels", level_copy)

    @property
    def seeds(self) -> np.ndarray:
        result = np.asarray(
            [identity.seed for identity in self.identities], dtype=np.int64
        )
        result.setflags(write=False)
        return result

    @property
    def query_keys(self) -> tuple[tuple[int, str], ...]:
        return tuple(dict.fromkeys(row.query_key for row in self.identities))

    def inventory(self, query_key: tuple[int, str]) -> G4QueryInventory:
        if (
            type(query_key) is not tuple
            or len(query_key) != 2
            or type(query_key[0]) is not int
            or type(query_key[1]) is not str
        ):
            raise ValueError("G4 inference inventory request is malformed")
        indices = [
            index
            for index, identity in enumerate(self.identities)
            if identity.query_key == query_key
        ]
        if not indices:
            raise ValueError("G4 inference inventory is absent")
        level = int(self.levels[indices[0]])
        return G4QueryInventory(
            query_key,
            level,
            one_rot_liability(level),
            self.observation_sha256[indices[0]],
            next(
                row
                for row in self.incumbent_identities
                if row.query_key == query_key
            ),
            tuple(self.identities[index] for index in indices),
        )

    @property
    def feature_inventory_sha256(self) -> str:
        return _feature_inventory_sha256(
            self.features,
            self.identities,
            self.incumbent_identities,
            self.levels,
            self.observation_sha256,
        )

    @property
    def sha256(self) -> str:
        return self.feature_inventory_sha256


def g4_inference_board_from_entries(
    entries: Iterable[Mapping[str, object]],
    *,
    encoder: TeacherStateEncoder | None = None,
) -> G4InferenceBoard:
    """Build deployable G4 features without accepting any exact outcome field."""

    resolved_encoder = TeacherStateEncoder() if encoder is None else encoder
    if (
        type(resolved_encoder) is not TeacherStateEncoder
        or resolved_encoder.schema.sha256 != TeacherStateEncoder.schema.sha256
    ):
        raise ValueError("G4 inference requires the exact teacher-v1 encoder")
    supplied = tuple(entries)
    if not supplied:
        raise ValueError("G4 inference requires at least one public query")
    public_rows: list[_G4PublicRow] = []
    for raw_entry in supplied:
        if type(raw_entry) is not dict:
            raise ValueError("G4 public query must be an exact dictionary")
        _preexact_bytes(raw_entry)
        _reject_exact_fields(raw_entry)
        required = {
            "schema",
            "seed",
            "query_id",
            "query_index",
            "shot_index",
            "tick",
            "pre_query_public_observation",
            "pre_query_public_observation_sha256",
            "candidates",
        }
        if set(raw_entry) != required or raw_entry["schema"] != INFERENCE_SCHEMA:
            raise ValueError("G4 public query envelope is malformed")
        seed = _plain_int(raw_entry["seed"], "query seed")
        query_id = raw_entry["query_id"]
        if type(query_id) is not str or not query_id:
            raise ValueError("G4 public query identity is malformed")
        query_index = _plain_int(raw_entry["query_index"], "query index")
        shot_index = _plain_int(raw_entry["shot_index"], "shot index", minimum=1)
        tick = _plain_int(raw_entry["tick"], "query tick")
        if query_index > 3 or shot_index > 19:
            raise ValueError("G4 public phase indices exceed frozen bounds")
        observation = raw_entry["pre_query_public_observation"]
        if type(observation) is not dict:
            raise ValueError("G4 public observation must be an exact dictionary")
        _validate_public_observation_shape(observation)
        observation_payload = _preexact_bytes(observation)
        observation_sha256 = hashlib.sha256(observation_payload).hexdigest()
        if (
            type(raw_entry["pre_query_public_observation_sha256"]) is not str
            or raw_entry["pre_query_public_observation_sha256"]
            != observation_sha256
            or type(observation.get("tick")) is not int
            or observation["tick"] != tick
        ):
            raise ValueError("G4 public observation identity differs")
        level = _plain_int(observation.get("level"), "level")
        difficulty = observation.get("difficulty")
        if type(difficulty) is not dict:
            raise ValueError("G4 public difficulty must be an exact dictionary")
        interval = _number(
            difficulty.get("spawn_interval_ticks"), "spawn interval"
        )
        if interval <= 0.0:
            raise ValueError("G4 public spawn interval must be positive")
        candidates_raw = raw_entry["candidates"]
        if type(candidates_raw) is not list or len(candidates_raw) < 2:
            raise ValueError("G4 public query needs incumbent and alternatives")
        candidates = tuple(
            _public_candidate_manifest(candidate)
            for candidate in candidates_raw
        )
        _validate_public_candidate_inventory(observation, candidates)
        if tuple(
            _plain_int(candidate["ordinal"], "candidate ordinal")
            for candidate in candidates
        ) != tuple(range(len(candidates))):
            raise ValueError("G4 public candidate ordinals are not closed")
        pair_zero = candidates[0]["pair"]
        if type(pair_zero) is not dict or pair_zero.get("incumbent") is not True:
            raise ValueError("G4 public candidate zero is not the incumbent")
        identities = tuple(
            _candidate_identity_from_public(seed, query_id, candidate)
            for candidate in candidates
        )
        if (
            len(set(identities)) != len(identities)
            or len({identity.candidate_id for identity in identities})
            != len(identities)
        ):
            raise ValueError("G4 public candidate identities are duplicated")

        encoded = resolved_encoder.encode([observation])
        active = np.flatnonzero(encoded.body_mask[0])
        identifiers = _g2._encoded_body_ids(encoded, observation)
        bound_ids = tuple(identifiers[index] for index in active)
        if any(identifier is None for identifier in bound_ids):
            raise RuntimeError("G4 active body failed public identity binding")
        concrete_ids = tuple(
            int(identifier) for identifier in bound_ids if identifier is not None
        )
        body_index = {
            identifier: position for position, identifier in enumerate(concrete_ids)
        }
        if len(body_index) != len(active):
            raise RuntimeError("G4 encoded body identities are not unique")
        body_matrix = np.array(encoded.body_features[0, active], copy=True)
        for column in _ID_BODY_COLUMNS:
            body_matrix[:, column] = 0.0
        body_rows = tuple(
            tuple(float(value) for value in row) for row in body_matrix
        )
        chain_groups, grouped_flags, color_groups = _g2._raw_body_metadata(
            observation, concrete_ids
        )
        global_row = tuple(float(value) for value in encoded.global_features[0])
        phase = (
            query_index / 3.0,
            shot_index / 19.0,
            (tick % interval) / interval,
        )
        incumbent_pair = _g2._pair_indices(
            candidates[0], body_index  # type: ignore[arg-type]
        )
        for identity, candidate in zip(identities, candidates, strict=True):
            source, destination = _g2._pair_indices(
                candidate, body_index  # type: ignore[arg-type]
            )
            public_rows.append(
                _G4PublicRow(
                    identity,
                    _public_branch_features(observation, candidate),
                    global_row,
                    phase,
                    body_rows,
                    chain_groups,
                    grouped_flags,
                    color_groups,
                    source,
                    destination,
                    incumbent_pair[0],
                    incumbent_pair[1],
                    observation_sha256,
                )
            )
        if level_from_public_global(global_row) != level:
            raise RuntimeError("G4 public level does not match its encoded state")
    features = _wide_features_from_public_rows(tuple(public_rows))
    alternative_mask = np.asarray(
        [row.ordinal != 0 for row in public_rows], dtype=np.bool_
    )
    alternatives = tuple(row for row in public_rows if row.ordinal != 0)
    incumbents = tuple(row.identity for row in public_rows if row.ordinal == 0)
    return G4InferenceBoard(
        np.array(features[alternative_mask], dtype=np.float64, copy=True),
        tuple(row.identity for row in alternatives),
        incumbents,
        np.asarray(
            [level_from_public_global(row.global_features) for row in alternatives],
            dtype=np.int64,
        ),
        tuple(row.observation_sha256 for row in alternatives),
    )


def level_from_public_global(global_features: tuple[float, ...]) -> int:
    if (
        type(global_features) is not tuple
        or len(global_features) != len(TeacherStateEncoder.schema.global_features)
        or any(
            type(value) is not float or not math.isfinite(value)
            for value in global_features
        )
    ):
        raise ValueError("G4 public global features are malformed")
    encoded = global_features[_LEVEL_INDEX]
    if encoded < 0.0:
        raise ValueError("G4 public encoded level is negative")
    decoded = math.expm1(encoded * 8.0)
    if not math.isfinite(decoded):
        raise ValueError("G4 public encoded level is nonfinite")
    level = int(round(decoded))
    rebound = float(np.float32(math.log1p(float(level)) / 8.0))
    if level < 0 or rebound != encoded:
        raise ValueError("G4 public level is not an exact encoded integer")
    return level


def g4_board_from_records(records: Iterable[BoardBranchRecord]) -> G4Board:
    supplied = tuple(records)
    if not supplied or any(type(row) is not BoardBranchRecord for row in supplied):
        raise ValueError("G4 requires exact board branch records")
    grouped: dict[tuple[int, str], list[BoardBranchRecord]] = {}
    for row in supplied:
        _validate_exact_record(row)
        grouped.setdefault(row.query_key, []).append(row)
    for rows in grouped.values():
        if (
            tuple(row.ordinal for row in rows) != tuple(range(len(rows)))
            or len({row.candidate_id for row in rows}) != len(rows)
            or len({row.observation_sha256 for row in rows}) != 1
            or len({level_from_record(row) for row in rows}) != 1
        ):
            raise ValueError("G4 exact query records are not closed and bound")
    wide = wide_board_from_records(supplied)
    alternatives = tuple(row for row in supplied if row.ordinal != 0)
    expected = tuple(
        (row.seed, row.query_id, row.ordinal) for row in alternatives
    )
    actual = tuple(
        (int(seed), str(query), int(ordinal))
        for seed, query, ordinal in zip(
            wide.seeds, wide.query_ids, wide.ordinals, strict=True
        )
    )
    if expected != actual:
        raise RuntimeError("G4 feature rows differ from exact record order")
    safe = np.asarray(
        [absolute_safe(row) for row in alternatives], dtype=np.bool_
    )
    levels = np.asarray(
        [level_from_record(row) for row in alternatives], dtype=np.int64
    )
    solvency = np.asarray(
        [solvency_target(row) for row in alternatives], dtype=np.float64
    )
    growth = np.asarray(
        [growth_target(row) for row in alternatives], dtype=np.float64
    )
    rebound = WideBoard(
        wide.features,
        safe,
        wide.seeds,
        wide.query_ids,
        wide.source_indices,
        wide.destination_indices,
        wide.ordinals,
    )
    identities = tuple(
        G4CandidateIdentity(
            row.seed,
            row.query_id,
            row.ordinal,
            row.candidate_id,
            row.action_id,
        )
        for row in alternatives
    )
    incumbents = tuple(
        G4CandidateIdentity(
            row.seed,
            row.query_id,
            row.ordinal,
            row.candidate_id,
            row.action_id,
        )
        for row in supplied
        if row.ordinal == 0
    )
    return G4Board(
        rebound,
        identities,
        incumbents,
        levels,
        tuple(row.observation_sha256 for row in alternatives),
        solvency,
        growth,
    )


_SOLVENCY_BOOST = BoostConfigR3(
    rounds=300,
    depth=3,
    learning_rate=0.05,
    l2=8.0,
    minimum_leaf=18,
    maximum_features=180,
    preserved_features=126,
    bins=16,
    balance_classes=False,
)
_GROWTH_BOOST = BoostConfigR3(
    rounds=300,
    depth=3,
    learning_rate=0.05,
    l2=8.0,
    minimum_leaf=18,
    maximum_features=180,
    preserved_features=126,
    bins=16,
    balance_classes=False,
)


@dataclass(frozen=True, slots=True)
class G4Config:
    folds: int = 8
    partition_id: str = ""
    seed_partition: tuple[tuple[int, ...], ...] = ()
    solvency: BoostConfigR3 = _SOLVENCY_BOOST
    growth: BoostConfigR3 = _GROWTH_BOOST

    def __post_init__(self) -> None:
        if (
            type(self.folds) is not int
            or self.folds != 8
            or type(self.partition_id) is not str
            or not self.partition_id
            or type(self.seed_partition) is not tuple
            or len(self.seed_partition) != self.folds
            or type(self.solvency) is not BoostConfigR3
            or type(self.growth) is not BoostConfigR3
            or self.solvency != _SOLVENCY_BOOST
            or self.growth != _GROWTH_BOOST
        ):
            raise ValueError("G4 configuration is malformed")
        flattened: list[int] = []
        for fold in self.seed_partition:
            if (
                type(fold) is not tuple
                or not fold
                or tuple(sorted(set(fold))) != fold
                or any(type(seed) is not int or seed < 0 for seed in fold)
            ):
                raise ValueError("G4 partition is malformed")
            flattened.extend(fold)
        if len(flattened) != len(set(flattened)):
            raise ValueError("G4 partition repeats a seed")

    def manifest(self) -> dict[str, object]:
        return {
            "folds": self.folds,
            "partition_id": self.partition_id,
            "seed_partition": [list(row) for row in self.seed_partition],
            "solvency": self.solvency.manifest(),
            "growth": self.growth.manifest(),
        }

    @classmethod
    def from_manifest(cls, value: Any) -> G4Config:
        raw = _core._exact_dict(value, "G4 config")
        if set(raw) != {
            "folds",
            "partition_id",
            "seed_partition",
            "solvency",
            "growth",
        }:
            raise RuntimeError("G4 configuration manifest is malformed")
        try:
            result = cls(
                _core._exact_int(raw["folds"], "folds"),
                _exact_str(raw["partition_id"], "partition_id"),
                tuple(
                    tuple(
                        _core._exact_int(seed, "partition seed")
                        for seed in _core._exact_list(fold, "partition fold")
                    )
                    for fold in _core._exact_list(
                        raw["seed_partition"], "seed partition"
                    )
                ),
                BoostConfigR3.from_manifest(raw["solvency"]),
                BoostConfigR3.from_manifest(raw["growth"]),
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError("G4 configuration manifest is malformed") from exc
        if not _same_manifest(result.manifest(), raw):
            raise RuntimeError("G4 configuration manifest is malformed")
        return result


def _partition(
    seeds: Sequence[int], config: G4Config
) -> tuple[tuple[int, ...], ...]:
    inventory = tuple(sorted(set(seeds)))
    supplied = config.seed_partition
    if (
        any(type(seed) is not int or seed < 0 for seed in seeds)
        or tuple(sorted(seed for fold in supplied for seed in fold)) != inventory
    ):
        raise ValueError("G4 explicit partition does not cover training seeds")
    return supplied


def _partition_sha(
    partition_id: str, partition: tuple[tuple[int, ...], ...]
) -> str:
    return _sha256(
        {
            "schema": PARTITION_SCHEMA,
            "partition_id": partition_id,
            "folds": [list(row) for row in partition],
        }
    )


@dataclass(frozen=True, slots=True)
class G4Fold:
    heldout_seeds: tuple[int, ...]
    solvency: HistogramNewtonBoostR3
    growth: HistogramNewtonBoostR3

    def __post_init__(self) -> None:
        if (
            type(self.heldout_seeds) is not tuple
            or not self.heldout_seeds
            or tuple(sorted(set(self.heldout_seeds))) != self.heldout_seeds
            or any(type(seed) is not int for seed in self.heldout_seeds)
            or type(self.solvency) is not HistogramNewtonBoostR3
            or type(self.growth) is not HistogramNewtonBoostR3
        ):
            raise ValueError("G4 fold is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "heldout_seeds": list(self.heldout_seeds),
            "solvency": self.solvency.manifest(),
            "growth": self.growth.manifest(),
        }

    @classmethod
    def from_manifest(cls, value: Any) -> G4Fold:
        raw = _core._exact_dict(value, "G4 fold")
        if set(raw) != {"heldout_seeds", "solvency", "growth"}:
            raise RuntimeError("G4 fold manifest is malformed")
        try:
            result = cls(
                tuple(
                    _core._exact_int(seed, "heldout seed")
                    for seed in _core._exact_list(
                        raw["heldout_seeds"], "heldout seeds"
                    )
                ),
                HistogramNewtonBoostR3.from_manifest(raw["solvency"]),
                HistogramNewtonBoostR3.from_manifest(raw["growth"]),
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError("G4 fold manifest is malformed") from exc
        if not _same_manifest(result.manifest(), raw):
            raise RuntimeError("G4 fold manifest is malformed")
        return result


@dataclass(frozen=True, slots=True)
class G4Prediction:
    identity: G4CandidateIdentity
    solvency_mean: float
    solvency_std: float
    growth_mean: float
    growth_std: float

    def __post_init__(self) -> None:
        values = (
            self.solvency_mean,
            self.solvency_std,
            self.growth_mean,
            self.growth_std,
        )
        if (
            type(self.identity) is not G4CandidateIdentity
            or any(type(value) is not float or not math.isfinite(value) for value in values)
            or not 0.0 <= self.solvency_mean <= 1.0
            or self.solvency_std < 0.0
            or not 0.0 <= self.growth_mean <= 1.0
            or self.growth_std < 0.0
        ):
            raise ValueError("G4 prediction is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "identity": self.identity.manifest(),
            "solvency_mean": self.solvency_mean,
            "solvency_std": self.solvency_std,
            "growth_mean": self.growth_mean,
            "growth_std": self.growth_std,
        }


@dataclass(frozen=True, slots=True)
class G4PredictionBatch:
    inventory: G4QueryInventory
    model_sha256: str
    feature_inventory_sha256: str
    predictions: tuple[G4Prediction, ...]

    def __post_init__(self) -> None:
        if (
            type(self.inventory) is not G4QueryInventory
            or type(self.model_sha256) is not str
            or _SHA256_RE.fullmatch(self.model_sha256) is None
            or type(self.feature_inventory_sha256) is not str
            or _SHA256_RE.fullmatch(self.feature_inventory_sha256) is None
            or type(self.predictions) is not tuple
            or any(type(row) is not G4Prediction for row in self.predictions)
            or tuple(row.identity for row in self.predictions)
            != self.inventory.identities
        ):
            raise ValueError("G4 prediction batch is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3i-resolution-prediction-batch-g4-v2",
            "inventory": self.inventory.manifest(),
            "inventory_sha256": self.inventory.sha256,
            "model_sha256": self.model_sha256,
            "feature_inventory_sha256": self.feature_inventory_sha256,
            "predictions": [row.manifest() for row in self.predictions],
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class ResolutionProposalG4:
    config: G4Config
    folds: tuple[G4Fold, ...]
    fold_by_seed: tuple[tuple[int, int], ...]
    training_seeds: tuple[int, ...]
    partition_sha256: str
    training_dataset_sha256: str
    training_feature_inventory_sha256: str
    dependency_identities: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if (
            type(self.config) is not G4Config
            or type(self.folds) is not tuple
            or len(self.folds) != self.config.folds
            or any(type(fold) is not G4Fold for fold in self.folds)
            or type(self.fold_by_seed) is not tuple
            or type(self.training_seeds) is not tuple
            or tuple(sorted(set(self.training_seeds))) != self.training_seeds
            or any(type(seed) is not int for seed in self.training_seeds)
            or type(self.partition_sha256) is not str
            or _SHA256_RE.fullmatch(self.partition_sha256) is None
            or type(self.training_dataset_sha256) is not str
            or _SHA256_RE.fullmatch(self.training_dataset_sha256) is None
            or type(self.training_feature_inventory_sha256) is not str
            or _SHA256_RE.fullmatch(self.training_feature_inventory_sha256)
            is None
            or type(self.dependency_identities) is not tuple
        ):
            raise ValueError("G4 model is malformed")
        assignments = tuple(
            sorted(
                (seed, index)
                for index, fold in enumerate(self.folds)
                for seed in fold.heldout_seeds
            )
        )
        if (
            any(
                type(row) is not tuple
                or len(row) != 2
                or type(row[0]) is not int
                or type(row[1]) is not int
                for row in self.fold_by_seed
            )
            or self.fold_by_seed != assignments
            or tuple(seed for seed, _index in assignments)
            != self.training_seeds
            or tuple(fold.heldout_seeds for fold in self.folds)
            != self.config.seed_partition
            or self.partition_sha256
            != _partition_sha(
                self.config.partition_id, self.config.seed_partition
            )
            or any(
                fold.solvency.config != self.config.solvency
                or fold.growth.config != self.config.growth
                for fold in self.folds
            )
        ):
            raise ValueError("G4 model partition or heads are malformed")
        try:
            dependencies = _exact_dependencies(
                {name: digest for name, digest in self.dependency_identities}
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("G4 model dependencies are malformed") from exc
        if dependencies != self.dependency_identities:
            raise ValueError("G4 model dependencies are malformed")

    @property
    def seed_folds(self) -> dict[int, int]:
        return dict(self.fold_by_seed)

    def manifest(self) -> dict[str, object]:
        return {
            "schema": MODEL_SCHEMA,
            "feature_names": list(WIDE_FEATURE_NAMES),
            "feature_width": WIDE_FEATURE_WIDTH,
            "training_dataset_sha256": self.training_dataset_sha256,
            "training_feature_inventory_sha256": (
                self.training_feature_inventory_sha256
            ),
            "training_seeds": list(self.training_seeds),
            "partition_sha256": self.partition_sha256,
            "dependency_identities": [
                list(row) for row in self.dependency_identities
            ],
            "absolute_safe_formula": ABSOLUTE_SAFE_FORMULA,
            "solvency_formula": SOLVENCY_FORMULA,
            "growth_formula": GROWTH_FORMULA,
            "training_population": "all-nonincumbent-alternatives",
            "feature_exclusions": [
                "seed",
                "query_id",
                "ordinal",
                "candidate_id",
                "action_id",
                "exact_labels",
                "certificate_fields",
                "source_sha256",
                "observation_sha256",
            ],
            "seed_weighting": "exact-equal-total-mass-per-training-seed",
            "cross_fit_unit": "whole-seed",
            "proposal_policy": PROPOSAL_POLICY,
            "fold_by_seed": [list(row) for row in self.fold_by_seed],
            "folds": [fold.manifest() for fold in self.folds],
            "config": self.config.manifest(),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())

    @classmethod
    def from_manifest(cls, value: Any) -> ResolutionProposalG4:
        raw = _core._exact_dict(value, "G4 model")
        required = {
            "schema",
            "feature_names",
            "feature_width",
            "training_dataset_sha256",
            "training_feature_inventory_sha256",
            "training_seeds",
            "partition_sha256",
            "dependency_identities",
            "absolute_safe_formula",
            "solvency_formula",
            "growth_formula",
            "training_population",
            "feature_exclusions",
            "seed_weighting",
            "cross_fit_unit",
            "proposal_policy",
            "fold_by_seed",
            "folds",
            "config",
        }
        expected_exclusions = [
            "seed",
            "query_id",
            "ordinal",
            "candidate_id",
            "action_id",
            "exact_labels",
            "certificate_fields",
            "source_sha256",
            "observation_sha256",
        ]
        if set(raw) != required:
            raise RuntimeError("G4 model feature identity is malformed")
        try:
            feature_names = [
                _exact_str(name, "feature name")
                for name in _core._exact_list(
                    raw["feature_names"], "feature names"
                )
            ]
            feature_width = _core._exact_int(
                raw["feature_width"], "feature width"
            )
            exclusions = [
                _exact_str(name, "feature exclusion")
                for name in _core._exact_list(
                    raw["feature_exclusions"], "feature exclusions"
                )
            ]
            if (
                raw["schema"] != MODEL_SCHEMA
                or feature_names != list(WIDE_FEATURE_NAMES)
                or feature_width != WIDE_FEATURE_WIDTH
                or raw["absolute_safe_formula"] != ABSOLUTE_SAFE_FORMULA
                or raw["solvency_formula"] != SOLVENCY_FORMULA
                or raw["growth_formula"] != GROWTH_FORMULA
                or raw["training_population"]
                != "all-nonincumbent-alternatives"
                or exclusions != expected_exclusions
                or raw["seed_weighting"]
                != "exact-equal-total-mass-per-training-seed"
                or raw["cross_fit_unit"] != "whole-seed"
                or raw["proposal_policy"] != PROPOSAL_POLICY
            ):
                raise RuntimeError("G4 model feature identity is malformed")
            config = G4Config.from_manifest(raw["config"])
            folds = tuple(
                G4Fold.from_manifest(row)
                for row in _core._exact_list(raw["folds"], "G4 folds")
            )
            assignments_list: list[tuple[int, int]] = []
            for item in _core._exact_list(
                raw["fold_by_seed"], "fold assignments"
            ):
                row = _core._exact_list(item, "fold assignment")
                if len(row) != 2:
                    raise RuntimeError("G4 fold assignment is malformed")
                assignments_list.append(
                    (
                        _core._exact_int(row[0], "assignment seed"),
                        _core._exact_int(row[1], "assignment fold"),
                    )
                )
            assignments = tuple(assignments_list)
            seeds = tuple(
                _core._exact_int(seed, "training seed")
                for seed in _core._exact_list(
                    raw["training_seeds"], "training seeds"
                )
            )
            dependency_list: list[tuple[str, str]] = []
            dependency_rows = _core._exact_list(
                raw["dependency_identities"], "dependencies"
            )
            for item in dependency_rows:
                row = _core._exact_list(item, "dependency identity")
                if len(row) != 2:
                    raise RuntimeError("G4 dependency identity is malformed")
                dependency_list.append(
                    (
                        _exact_str(row[0], "dependency name"),
                        _exact_str(row[1], "dependency SHA-256"),
                    )
                )
            dependencies = tuple(dependency_list)
            result = cls(
                config,
                folds,
                assignments,
                seeds,
                _core._require_sha256(
                    raw["partition_sha256"], "partition sha"
                ),
                _core._require_sha256(
                    raw["training_dataset_sha256"], "dataset sha"
                ),
                _core._require_sha256(
                    raw["training_feature_inventory_sha256"],
                    "feature inventory sha",
                ),
                dependencies,
            )
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError("G4 model manifest is malformed") from exc
        expected_assignments = tuple(
            sorted(
                (seed, index)
                for index, fold in enumerate(folds)
                for seed in fold.heldout_seeds
            )
        )
        try:
            partition = _partition(seeds, config)
            dependency_map = {name: digest for name, digest in dependencies}
            exact_dependencies = _exact_dependencies(dependency_map)
        except ValueError as exc:
            raise RuntimeError("G4 model manifest is malformed") from exc
        if (
            dependencies != exact_dependencies
            or len(folds) != config.folds
            or assignments != expected_assignments
            or seeds != tuple(sorted(set(seeds)))
            or tuple(seed for seed, _ in assignments) != seeds
            or tuple(fold.heldout_seeds for fold in folds) != partition
            or result.partition_sha256
            != _partition_sha(config.partition_id, partition)
            or any(
                fold.solvency.config != config.solvency
                or fold.growth.config != config.growth
                for fold in folds
            )
            or not _same_manifest(result.manifest(), raw)
        ):
            raise RuntimeError("G4 model manifest is malformed")
        return result

    def predict(
        self,
        board: G4InferenceBoard,
        *,
        oof: bool = False,
    ) -> tuple[G4Prediction, ...]:
        if type(board) is not G4InferenceBoard or type(oof) is not bool:
            raise ValueError("G4 prediction request is malformed")
        if (
            oof
            and board.feature_inventory_sha256
            != self.training_feature_inventory_sha256
        ):
            raise ValueError("G4 OOF feature inventory differs from training")
        known = self.seed_folds
        solvency: list[list[float]] = [[] for _ in board.identities]
        growth: list[list[float]] = [[] for _ in board.identities]
        for fold_index, fold in enumerate(self.folds):
            solvency_score = fold.solvency.probabilities(board.features)
            growth_score = fold.growth.probabilities(board.features)
            for index, seed_value in enumerate(board.seeds):
                seed = int(seed_value)
                if oof:
                    if seed not in known:
                        raise ValueError("G4 OOF prediction has an unknown seed")
                    if known[seed] != fold_index:
                        continue
                solvency[index].append(float(solvency_score[index]))
                growth[index].append(float(growth_score[index]))
        output: list[G4Prediction] = []
        for identity, solvent, growing in zip(
            board.identities, solvency, growth, strict=True
        ):
            if not solvent or not growing:
                raise RuntimeError("G4 prediction has no eligible fold")
            output.append(
                G4Prediction(
                    identity,
                    float(np.mean(solvent)),
                    float(np.std(solvent)),
                    float(np.mean(growing)),
                    float(np.std(growing)),
                )
            )
        return tuple(output)

    def predict_batches(
        self,
        board: G4InferenceBoard,
        *,
        oof: bool = False,
    ) -> tuple[G4PredictionBatch, ...]:
        predictions = self.predict(board, oof=oof)
        feature_inventory_sha256 = board.feature_inventory_sha256
        model_sha256 = self.sha256
        output: list[G4PredictionBatch] = []
        for query_key in board.query_keys:
            inventory = board.inventory(query_key)
            rows = tuple(
                row
                for row in predictions
                if row.identity.query_key == query_key
            )
            output.append(
                G4PredictionBatch(
                    inventory,
                    model_sha256,
                    feature_inventory_sha256,
                    rows,
                )
            )
        return tuple(output)


def train_resolution_proposal_g4(
    records: Iterable[BoardBranchRecord],
    *,
    config: G4Config,
    dependencies: Mapping[str, str],
) -> ResolutionProposalG4:
    if (
        type(records) in {G4Board, G4InferenceBoard}
        or type(config) is not G4Config
    ):
        raise ValueError("G4 training inputs have inexact types")
    try:
        board = g4_board_from_records(records)
    except TypeError as exc:
        raise ValueError(
            "G4 training requires exact board branch records"
        ) from exc
    dependency_identities = _exact_dependencies(dependencies)
    partition = _partition(board.unique_seeds, config)
    all_seeds = set(board.unique_seeds)
    folds: list[G4Fold] = []
    for heldout in partition:
        training_seeds = all_seeds - set(heldout)
        mask = np.asarray(
            [int(seed) in training_seeds for seed in board.seeds],
            dtype=np.bool_,
        )
        folds.append(
            G4Fold(
                heldout,
                HistogramNewtonBoostR3.fit(
                    board.features[mask],
                    board.solvency[mask],
                    board.seeds[mask],
                    config.solvency,
                ),
                HistogramNewtonBoostR3.fit(
                    board.features[mask],
                    board.growth[mask],
                    board.seeds[mask],
                    config.growth,
                ),
            )
        )
    assignments = tuple(
        sorted(
            (seed, index)
            for index, heldout in enumerate(partition)
            for seed in heldout
        )
    )
    return ResolutionProposalG4(
        config,
        tuple(folds),
        assignments,
        board.unique_seeds,
        _partition_sha(config.partition_id, partition),
        board.sha256,
        board.feature_inventory_sha256,
        dependency_identities,
    )


@dataclass(frozen=True, slots=True)
class G4Proposal:
    query_key: tuple[int, str]
    mode: str
    level: int
    reserve: int
    requested_budget: int
    effective_budget: int
    candidate_count: int
    inventory_sha256: str
    prediction_batch_sha256: str
    model_sha256: str
    feature_inventory_sha256: str
    incumbent_outcome_sha256: str
    identities: tuple[G4CandidateIdentity, ...]
    admission_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.query_key) is not tuple
            or len(self.query_key) != 2
            or type(self.query_key[0]) is not int
            or type(self.query_key[1]) is not str
            or not self.query_key[1]
            or self.mode not in {"rescue", "growth"}
            or type(self.level) is not int
            or self.level < 0
            or type(self.reserve) is not int
            or self.reserve != one_rot_liability(self.level)
            or type(self.requested_budget) is not int
            or self.requested_budget not in {1, 2, 4, 8}
            or type(self.effective_budget) is not int
            or type(self.candidate_count) is not int
            or self.candidate_count < 1
            or self.effective_budget
            != min(self.requested_budget, self.candidate_count)
            or any(
                type(value) is not str or _SHA256_RE.fullmatch(value) is None
                for value in (
                    self.inventory_sha256,
                    self.prediction_batch_sha256,
                    self.model_sha256,
                    self.feature_inventory_sha256,
                    self.incumbent_outcome_sha256,
                )
            )
            or type(self.identities) is not tuple
            or len(self.identities) != self.effective_budget
            or len(set(self.identities)) != len(self.identities)
            or len({row.candidate_id for row in self.identities})
            != len(self.identities)
            or type(self.admission_sources) is not tuple
            or len(self.admission_sources) != len(self.identities)
            or any(
                type(row) is not G4CandidateIdentity
                or row.ordinal < 1
                or row.query_key != self.query_key
                for row in self.identities
            )
            or any(
                source
                not in {
                    "solvency-quota",
                    "growth-quota",
                    "solvency-fill",
                    "growth-fill",
                }
                for source in self.admission_sources
            )
        ):
            raise ValueError("G4 proposal is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3i-resolution-proposal-g4-v2",
            "query_key": list(self.query_key),
            "mode": self.mode,
            "level": self.level,
            "reserve": self.reserve,
            "requested_budget": self.requested_budget,
            "effective_budget": self.effective_budget,
            "exact_simulation_cost": self.effective_budget + 1,
            "candidate_count": self.candidate_count,
            "inventory_sha256": self.inventory_sha256,
            "prediction_batch_sha256": self.prediction_batch_sha256,
            "model_sha256": self.model_sha256,
            "feature_inventory_sha256": self.feature_inventory_sha256,
            "incumbent_outcome_sha256": self.incumbent_outcome_sha256,
            "identities": [row.manifest() for row in self.identities],
            "admission_sources": list(self.admission_sources),
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())


def propose_g4(
    predictions: G4PredictionBatch,
    incumbent: G4ExactOutcome,
    *,
    budget: int,
) -> G4Proposal:
    if type(budget) is not int or budget not in {1, 2, 4, 8}:
        raise ValueError("G4 proposal budget must be one of 1, 2, 4, 8")
    if (
        type(predictions) is not G4PredictionBatch
        or type(incumbent) is not G4ExactOutcome
        or incumbent.identity != predictions.inventory.incumbent_identity
        or incumbent.level != predictions.inventory.level
        or incumbent.observation_sha256
        != predictions.inventory.observation_sha256
    ):
        raise ValueError("G4 incumbent and frozen prediction batch differ")
    supplied = predictions.predictions
    mode = (
        "rescue"
        if (
            not incumbent.absolute_safe
            or incumbent.b2 is None
            or incumbent.b2 < predictions.inventory.reserve
        )
        else "growth"
    )
    target = min(budget, len(supplied))
    solvency_rank = sorted(
        supplied, key=lambda row: (-row.solvency_mean, row.identity.ordinal)
    )
    growth_rank = sorted(
        supplied, key=lambda row: (-row.growth_mean, row.identity.ordinal)
    )
    ranks = {"solvency": solvency_rank, "growth": growth_rank}
    active = "solvency" if mode == "rescue" else "growth"
    other = "growth" if mode == "rescue" else "solvency"
    selected: list[G4CandidateIdentity] = []
    sources: list[str] = []
    seen: set[G4CandidateIdentity] = set()
    cursors = {"solvency": 0, "growth": 0}

    def admit(
        rank: list[G4Prediction], head: str, count: int, source: str
    ) -> None:
        admitted = 0
        while admitted < count and cursors[head] < len(rank):
            identity = rank[cursors[head]].identity
            cursors[head] += 1
            if identity in seen:
                continue
            seen.add(identity)
            selected.append(identity)
            sources.append(source)
            admitted += 1

    active_quota = (target + 1) // 2
    other_quota = target // 2
    admit(ranks[active], active, active_quota, f"{active}-quota")
    admit(ranks[other], other, other_quota, f"{other}-quota")
    while len(selected) < target:
        before = len(selected)
        admit(ranks[active], active, 1, f"{active}-fill")
        if len(selected) < target:
            admit(ranks[other], other, 1, f"{other}-fill")
        if len(selected) == before:
            raise RuntimeError("G4 proposal ranks could not fill the budget")
    return G4Proposal(
        predictions.inventory.query_key,
        mode,
        predictions.inventory.level,
        predictions.inventory.reserve,
        budget,
        target,
        len(supplied),
        predictions.inventory.sha256,
        predictions.sha256,
        predictions.model_sha256,
        predictions.feature_inventory_sha256,
        incumbent.sha256,
        tuple(selected),
        tuple(sources),
    )


@dataclass(frozen=True, slots=True)
class G4ExactCertificate:
    identity: G4CandidateIdentity
    exact_score_advantage: float
    b2: float
    level: int
    source_sha256: str
    observation_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.identity) is not G4CandidateIdentity
            or type(self.exact_score_advantage) is not float
            or not math.isfinite(self.exact_score_advantage)
            or type(self.b2) is not float
            or not math.isfinite(self.b2)
            or self.b2 < 0.0
            or type(self.level) is not int
            or self.level < 0
            or type(self.source_sha256) is not str
            or _SHA256_RE.fullmatch(self.source_sha256) is None
            or type(self.observation_sha256) is not str
            or _SHA256_RE.fullmatch(self.observation_sha256) is None
        ):
            raise ValueError("G4 exact certificate is malformed")


@dataclass(frozen=True, slots=True)
class G4ExactOutcome:
    """One terminal exact evaluation bound to a query observation."""

    identity: G4CandidateIdentity
    terminal: bool
    level: int
    candidate_resolved: bool
    finite_pair: bool
    exact_unsafe: bool
    severe_unsafe: bool
    b2: float | None
    delta_b2: float | None
    exact_score_advantage: float
    source_sha256: str
    observation_sha256: str

    def __post_init__(self) -> None:
        flags = (
            self.candidate_resolved,
            self.finite_pair,
            self.exact_unsafe,
            self.severe_unsafe,
        )
        if (
            type(self.identity) is not G4CandidateIdentity
            or self.terminal is not True
            or type(self.level) is not int
            or self.level < 0
            or any(type(value) is not bool for value in flags)
            or type(self.exact_score_advantage) is not float
            or not math.isfinite(self.exact_score_advantage)
            or (
                self.b2 is not None
                and (type(self.b2) is not float or not math.isfinite(self.b2))
            )
            or (
                self.delta_b2 is not None
                and (
                    type(self.delta_b2) is not float
                    or not math.isfinite(self.delta_b2)
                )
            )
            or (
                self.finite_pair
                and (
                    not self.candidate_resolved
                    or self.b2 is None
                    or self.delta_b2 is None
                )
            )
            or type(self.source_sha256) is not str
            or _SHA256_RE.fullmatch(self.source_sha256) is None
            or type(self.observation_sha256) is not str
            or _SHA256_RE.fullmatch(self.observation_sha256) is None
        ):
            raise ValueError("G4 exact terminal outcome is malformed")

    @property
    def absolute_safe(self) -> bool:
        return bool(
            self.candidate_resolved
            and not self.exact_unsafe
            and not self.severe_unsafe
            and self.b2 is not None
            and self.b2 >= 0.0
        )

    @property
    def certificate(self) -> G4ExactCertificate | None:
        if not self.absolute_safe:
            return None
        assert self.b2 is not None
        return G4ExactCertificate(
            self.identity,
            self.exact_score_advantage,
            self.b2,
            self.level,
            self.source_sha256,
            self.observation_sha256,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3i-resolution-exact-outcome-g4-v2",
            "identity": self.identity.manifest(),
            "terminal": self.terminal,
            "level": self.level,
            "candidate_resolved": self.candidate_resolved,
            "finite_pair": self.finite_pair,
            "exact_unsafe": self.exact_unsafe,
            "severe_unsafe": self.severe_unsafe,
            "b2": self.b2,
            "delta_b2": self.delta_b2,
            "exact_score_advantage": self.exact_score_advantage,
            "source_sha256": self.source_sha256,
            "observation_sha256": self.observation_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())


def exact_outcome_g4(record: BoardBranchRecord) -> G4ExactOutcome:
    _validate_exact_record(record)
    return G4ExactOutcome(
        G4CandidateIdentity(
            record.seed,
            record.query_id,
            record.ordinal,
            record.candidate_id,
            record.action_id,
        ),
        True,
        level_from_record(record),
        record.candidate_resolved,
        record.finite_pair,
        record.exact_unsafe,
        record.severe_unsafe,
        record.b2,
        record.delta_b2,
        record.score_advantage,
        record.source_sha256,
        record.observation_sha256,
    )


def exact_certificate_g4(
    record: BoardBranchRecord,
) -> G4ExactCertificate | None:
    return exact_outcome_g4(record).certificate


@dataclass(frozen=True, slots=True)
class G4Selection:
    proposal_sha256: str
    mode: str
    status: str
    identity: G4CandidateIdentity
    outcome_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.proposal_sha256) is not str
            or _SHA256_RE.fullmatch(self.proposal_sha256) is None
            or self.mode not in {"rescue", "growth"}
            or self.status
            not in {
                "selected-rescue",
                "selected-growth",
                "incumbent-retained",
                "no-safe-proposal-abstention",
            }
            or type(self.identity) is not G4CandidateIdentity
            or type(self.outcome_sha256) is not str
            or _SHA256_RE.fullmatch(self.outcome_sha256) is None
            or (
                self.status in {"incumbent-retained", "no-safe-proposal-abstention"}
                and self.identity.ordinal != 0
            )
            or (
                self.status in {"selected-rescue", "selected-growth"}
                and self.identity.ordinal < 1
            )
        ):
            raise ValueError("G4 exact selection is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3i-resolution-selection-g4-v2",
            "proposal_sha256": self.proposal_sha256,
            "mode": self.mode,
            "status": self.status,
            "identity": self.identity.manifest(),
            "outcome_sha256": self.outcome_sha256,
        }

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())


def select_exact_g4(
    proposal: G4Proposal,
    outcomes: Sequence[G4ExactOutcome],
) -> G4Selection:
    if type(proposal) is not G4Proposal:
        raise ValueError("G4 exact selection context is malformed")
    supplied = tuple(outcomes)
    if any(type(row) is not G4ExactOutcome for row in supplied):
        raise ValueError("G4 exact terminal outcomes are malformed")
    expected = (0, *tuple(row.ordinal for row in proposal.identities))
    if (
        len(supplied) != len(proposal.identities) + 1
        or tuple(row.identity.ordinal for row in supplied) != expected
        or tuple(row.identity for row in supplied[1:]) != proposal.identities
        or supplied[0].identity.ordinal != 0
        or supplied[0].sha256 != proposal.incumbent_outcome_sha256
        or any(row.identity.query_key != proposal.query_key for row in supplied)
        or len({row.identity for row in supplied}) != len(supplied)
        or len({row.identity.candidate_id for row in supplied}) != len(supplied)
        or {row.observation_sha256 for row in supplied}
        != {supplied[0].observation_sha256}
        or supplied[0].observation_sha256 == ""
        or supplied[0].level != proposal.level
        or any(row.level != proposal.level for row in supplied)
        or one_rot_liability(proposal.level) != proposal.reserve
    ):
        raise ValueError("G4 exact terminal outcome closure differs")
    incumbent, alternatives = supplied[0], supplied[1:]
    mode = (
        "rescue"
        if (
            not incumbent.absolute_safe
            or incumbent.b2 is None
            or incumbent.b2 < proposal.reserve
        )
        else "growth"
    )
    if mode != proposal.mode:
        raise ValueError("G4 exact selection regime differs from proposal")
    if mode == "rescue":
        eligible = [
            row
            for row in alternatives
            if row.absolute_safe
            and row.b2 is not None
            and (
                not incumbent.absolute_safe
                or incumbent.b2 is None
                or row.b2 > incumbent.b2
            )
        ]
        selected = (
            None
            if not eligible
            else min(
                eligible,
                key=lambda row: (
                    -float(row.b2),
                    -row.exact_score_advantage,
                    row.identity.ordinal,
                ),
            )
        )
        selected_status = "selected-rescue"
    else:
        eligible = [
            row
            for row in alternatives
            if row.absolute_safe
            and row.b2 is not None
            and row.b2 >= proposal.reserve
            and row.exact_score_advantage > 0.0
        ]
        selected = (
            None
            if not eligible
            else min(
                eligible,
                key=lambda row: (
                    -row.exact_score_advantage,
                    -float(row.b2),
                    row.identity.ordinal,
                ),
            )
        )
        selected_status = "selected-growth"
    if selected is not None:
        return G4Selection(
            proposal.sha256,
            mode,
            selected_status,
            selected.identity,
            selected.sha256,
        )
    status = (
        "no-safe-proposal-abstention"
        if mode == "rescue" and not incumbent.absolute_safe
        else "incumbent-retained"
    )
    return G4Selection(
        proposal.sha256,
        mode,
        status,
        incumbent.identity,
        incumbent.sha256,
    )


def _checkpoint(
    model: ResolutionProposalG4,
    metadata: Mapping[str, Any],
    dependencies: Mapping[str, str],
) -> dict[str, object]:
    metadata_value = _core._canonical_dict(metadata, "G4 metadata")
    dependency_value = dict(_exact_dependencies(dependencies))
    body = {
        "schema": CHECKPOINT_SCHEMA,
        "model_sha256": model.sha256,
        "partition_sha256": model.partition_sha256,
        "dataset_sha256": model.training_dataset_sha256,
        "feature_inventory_sha256": model.training_feature_inventory_sha256,
        "metadata": metadata_value,
        "metadata_sha256": _sha256(metadata_value),
        "dependencies": dependency_value,
        "dependencies_sha256": _sha256(dependency_value),
        "model": model.manifest(),
    }
    return {**body, "checkpoint_sha256": _sha256(body)}


def _decode_checkpoint(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in items:
            if key in output:
                raise RuntimeError("G4 checkpoint has duplicate keys")
            output[key] = value
        return output

    def reject(value: str) -> None:
        raise RuntimeError(f"G4 checkpoint has non-finite {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=reject,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("G4 checkpoint JSON is malformed") from exc
    if type(value) is not dict or raw != _core._canonical_bytes(value):
        raise RuntimeError("G4 checkpoint encoding is noncanonical")
    return value


def _validate_checkpoint(
    value: Any,
    *,
    expected_metadata: Mapping[str, Any],
    expected_model_sha256: str,
    expected_partition_sha256: str,
    expected_dataset_sha256: str,
    expected_feature_inventory_sha256: str,
    expected_dependencies: Mapping[str, str],
) -> tuple[ResolutionProposalG4, dict[str, Any]]:
    raw = _core._exact_dict(value, "G4 checkpoint")
    required = {
        "schema",
        "model_sha256",
        "partition_sha256",
        "dataset_sha256",
        "feature_inventory_sha256",
        "metadata",
        "metadata_sha256",
        "dependencies",
        "dependencies_sha256",
        "model",
        "checkpoint_sha256",
    }
    if set(raw) != required or raw.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("G4 checkpoint envelope is malformed")
    expected_model = _core._require_sha256(
        expected_model_sha256, "expected G4 model"
    )
    expected_partition = _core._require_sha256(
        expected_partition_sha256, "expected G4 partition"
    )
    expected_dataset = _core._require_sha256(
        expected_dataset_sha256, "expected G4 dataset"
    )
    expected_feature_inventory = _core._require_sha256(
        expected_feature_inventory_sha256,
        "expected G4 feature inventory",
    )
    metadata = _core._canonical_dict(expected_metadata, "expected G4 metadata")
    try:
        dependencies = dict(_exact_dependencies(expected_dependencies))
    except ValueError as exc:
        raise RuntimeError("expected G4 dependencies are malformed") from exc
    stored_metadata = _core._canonical_dict(raw["metadata"], "G4 metadata")
    stored_dependencies = dict(
        _exact_dependencies(
            _core._exact_dict(raw["dependencies"], "G4 dependencies")
        )
    )
    body = {key: raw[key] for key in raw if key != "checkpoint_sha256"}
    if (
        _core._require_sha256(raw["checkpoint_sha256"], "checkpoint sha")
        != _sha256(body)
        or _core._require_sha256(raw["model_sha256"], "model sha")
        != expected_model
        or _core._require_sha256(raw["partition_sha256"], "partition sha")
        != expected_partition
        or _core._require_sha256(raw["dataset_sha256"], "dataset sha")
        != expected_dataset
        or _core._require_sha256(
            raw["feature_inventory_sha256"], "feature inventory sha"
        )
        != expected_feature_inventory
        or _core._require_sha256(raw["metadata_sha256"], "metadata sha")
        != _sha256(stored_metadata)
        or not _same_manifest(stored_metadata, metadata)
        or _core._require_sha256(raw["dependencies_sha256"], "dependencies sha")
        != _sha256(stored_dependencies)
        or not _same_manifest(stored_dependencies, dependencies)
    ):
        raise RuntimeError("G4 checkpoint expectation or identity mismatch")
    model = ResolutionProposalG4.from_manifest(raw["model"])
    if (
        model.sha256 != expected_model
        or model.partition_sha256 != expected_partition
        or model.training_dataset_sha256 != expected_dataset
        or model.training_feature_inventory_sha256
        != expected_feature_inventory
        or dict(model.dependency_identities) != dependencies
    ):
        raise RuntimeError("G4 checkpoint model binding differs")
    return model, metadata


def load_checkpoint_g4(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str],
    expected_metadata: Mapping[str, Any],
    expected_model_sha256: str,
    expected_partition_sha256: str,
    expected_dataset_sha256: str,
    expected_feature_inventory_sha256: str,
    expected_dependencies: Mapping[str, str],
) -> tuple[ResolutionProposalG4, dict[str, Any]]:
    _target, directory, name = _core._absolute_direct(path, root)
    directory_fd, directory_before = _core._open_directory(directory)
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        raw, opened = _core._read_descriptor(descriptor)
        model, metadata = _validate_checkpoint(
            _decode_checkpoint(raw),
            expected_metadata=expected_metadata,
            expected_model_sha256=expected_model_sha256,
            expected_partition_sha256=expected_partition_sha256,
            expected_dataset_sha256=expected_dataset_sha256,
            expected_feature_inventory_sha256=(
                expected_feature_inventory_sha256
            ),
            expected_dependencies=expected_dependencies,
        )
        live_root = _core._lstat_path(directory)
        live_target = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        final_descriptor = os.fstat(descriptor)
        directory_after = os.fstat(directory_fd)
        if (
            stat.S_ISLNK(live_root.st_mode)
            or (live_root.st_dev, live_root.st_ino)
            != (directory_before.st_dev, directory_before.st_ino)
            or not _core._same_stat(opened, live_target, _core._FILE_FIELDS)
            or not _core._same_stat(
                opened, final_descriptor, _core._FILE_FIELDS
            )
            or not _core._same_stat(
                directory_before, directory_after, _core._DIRECTORY_FIELDS
            )
        ):
            raise RuntimeError("G4 checkpoint path changed or ABA occurred")
    except OSError as exc:
        raise RuntimeError("G4 checkpoint cannot be read safely") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)
    return model, metadata


def save_checkpoint_g4(
    path: str | os.PathLike[str],
    model: ResolutionProposalG4,
    *,
    root: str | os.PathLike[str],
    metadata: Mapping[str, Any],
    expected_model_sha256: str,
    expected_partition_sha256: str,
    expected_dataset_sha256: str,
    expected_feature_inventory_sha256: str,
    dependencies: Mapping[str, str],
) -> str:
    if type(model) is not ResolutionProposalG4:
        raise RuntimeError("G4 checkpoint model has the wrong type")
    expected_model = _core._require_sha256(
        expected_model_sha256, "expected G4 model"
    )
    expected_partition = _core._require_sha256(
        expected_partition_sha256, "expected G4 partition"
    )
    expected_dataset = _core._require_sha256(
        expected_dataset_sha256, "expected G4 dataset"
    )
    expected_feature_inventory = _core._require_sha256(
        expected_feature_inventory_sha256,
        "expected G4 feature inventory",
    )
    try:
        expected_dependencies = dict(_exact_dependencies(dependencies))
    except ValueError as exc:
        raise RuntimeError("expected G4 dependencies are malformed") from exc
    if (
        model.sha256 != expected_model
        or model.partition_sha256 != expected_partition
        or model.training_dataset_sha256 != expected_dataset
        or model.training_feature_inventory_sha256
        != expected_feature_inventory
        or dict(model.dependency_identities) != expected_dependencies
    ):
        raise RuntimeError("G4 save expectations do not bind the model")
    encoded = _core._canonical_bytes(
        _checkpoint(model, metadata, expected_dependencies)
    )
    _target, directory, name = _core._absolute_direct(path, root)
    directory_fd, _opened_directory = _core._open_directory(directory)
    temporary = f".{name}.tmp-{os.getpid()}-{secrets.token_hex(12)}"
    temporary_fd = target_fd = -1
    temporary_live = published = False
    try:
        try:
            os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("G4 checkpoint destination already exists")
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=directory_fd,
        )
        temporary_live = True
        view = memoryview(encoded)
        while view:
            written = os.write(temporary_fd, view)
            if written <= 0:
                raise RuntimeError("G4 checkpoint write was incomplete")
            view = view[written:]
        os.fchmod(temporary_fd, 0o444)
        os.fsync(temporary_fd)
        os.link(
            temporary,
            name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
        published = True
        os.unlink(temporary, dir_fd=directory_fd)
        temporary_live = False
        os.fsync(directory_fd)
        directory_baseline = os.fstat(directory_fd)
        target_fd = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        raw, opened = _core._read_descriptor(target_fd)
        live_root = _core._lstat_path(directory)
        live_target = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        final_target = os.fstat(target_fd)
        final_temporary = os.fstat(temporary_fd)
        directory_after = os.fstat(directory_fd)
        if (
            raw != encoded
            or stat.S_ISLNK(live_root.st_mode)
            or (live_root.st_dev, live_root.st_ino)
            != (directory_baseline.st_dev, directory_baseline.st_ino)
            or not _core._same_stat(opened, live_target, _core._FILE_FIELDS)
            or not _core._same_stat(opened, final_target, _core._FILE_FIELDS)
            or not _core._same_stat(
                opened, final_temporary, _core._FILE_FIELDS
            )
            or not _core._same_stat(
                directory_baseline, directory_after, _core._DIRECTORY_FIELDS
            )
        ):
            raise RuntimeError("G4 checkpoint publication changed or ABA occurred")
    except OSError as exc:
        raise RuntimeError("G4 checkpoint cannot be published safely") from exc
    finally:
        if target_fd >= 0:
            os.close(target_fd)
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_live:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        os.close(directory_fd)
    if not published:
        raise RuntimeError("G4 checkpoint publication failed")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ABSOLUTE_SAFE_FORMULA",
    "CHECKPOINT_SCHEMA",
    "FEATURE_INVENTORY_SCHEMA",
    "GROWTH_FORMULA",
    "G4Board",
    "G4CandidateIdentity",
    "G4Config",
    "G4ExactCertificate",
    "G4ExactOutcome",
    "G4Fold",
    "G4InferenceBoard",
    "G4Prediction",
    "G4PredictionBatch",
    "G4Proposal",
    "G4QueryInventory",
    "G4Selection",
    "INFERENCE_SCHEMA",
    "MODEL_SCHEMA",
    "PROPOSAL_POLICY",
    "REQUIRED_DEPENDENCIES",
    "ResolutionProposalG4",
    "SOLVENCY_FORMULA",
    "absolute_safe",
    "exact_certificate_g4",
    "exact_outcome_g4",
    "g4_board_from_records",
    "g4_inference_board_from_entries",
    "growth_target",
    "level_from_record",
    "level_from_public_global",
    "load_checkpoint_g4",
    "one_rot_liability",
    "propose_g4",
    "save_checkpoint_g4",
    "select_exact_g4",
    "solvency_target",
    "train_resolution_proposal_g4",
]
