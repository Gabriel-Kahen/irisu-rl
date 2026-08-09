"""Development-only hierarchical options over the public IriSu observation.

The hierarchy chooses a board-level option only at observable events.  Every
shot is still produced by :class:`ClosedLoopSteeringExpert`; option projections
only restrict which legal public bodies its analytic controller may consider.
No snapshot, future RNG, sealed artifact, or canonical-run interface is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np

from irisu_pointer.steering import (
    ClosedLoopSteeringExpert,
    SteeringDecision,
    SteeringExpertConfig,
    SteeringIntent,
)


class Option(str, Enum):
    """Portable macro-actions fitted and selected by the option-value model."""

    BUILD_ANCHOR = "build_anchor"
    EXTEND_ANCHOR = "extend_anchor"
    HARVEST = "harvest"
    ROTTEN_TRIAGE = "rotten_triage"
    EJECT = "eject"
    WAIT = "wait"
    BONUS_CLEAR = "bonus_clear"


OPTION_ORDER: tuple[Option, ...] = tuple(Option)

_LIVE_LIFECYCLES = frozenset(
    {
        "scripted_falling",
        "dynamic_fresh",
        "falling",
        "fresh",
        "confirmed",
        "rotten",
    }
)
_SOURCE_LIFECYCLES = frozenset(
    {"scripted_falling", "dynamic_fresh", "falling", "fresh"}
)
_MODEL_SCHEMA = "irisu-hierarchical-option-values-v1"
_CLEARS_PER_LEVEL = 10


def _int(value: Any, name: str, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    item = getattr(value, "item", None)
    if callable(item):
        return _int(item(), name, default)
    raise TypeError(f"{name} must be an integer")


def _float(value: Any, name: str, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        item = getattr(value, "item", None)
        if not callable(item):
            raise TypeError(f"{name} must be numeric")
        result = float(item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _bodies(observation: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    raw = observation.get("bodies", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("observation bodies must be a sequence")
    if any(not isinstance(body, Mapping) for body in raw):
        raise TypeError("every public body must be a mapping")
    result = tuple(body for body in raw if isinstance(body, Mapping))
    identifiers = [_int(body.get("id"), "body.id", -1) for body in result]
    if any(identifier < 0 for identifier in identifiers):
        raise ValueError("public body IDs must be nonnegative")
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("public body IDs must be unique")
    return result


def _pieces(observation: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        body
        for body in _bodies(observation)
        if body.get("kind") == "piece"
        and str(body.get("lifecycle", "")) in _LIVE_LIFECYCLES
    )


def _source(body: Mapping[str, Any]) -> bool:
    return (
        body.get("kind") == "piece"
        and str(body.get("lifecycle", "")) in _SOURCE_LIFECYCLES
        and _int(body.get("chain_id"), "body.chain_id") == 0
    )


def _groups(
    pieces: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int], tuple[Mapping[str, Any], ...]]:
    grouped: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for body in pieces:
        chain = _int(body.get("chain_id"), "body.chain_id")
        if chain:
            grouped[(_int(body.get("color"), "body.color", -1), chain)].append(
                body
            )
    return {
        key: tuple(sorted(value, key=lambda body: _int(body.get("id"), "body.id")))
        for key, value in grouped.items()
    }


def _field_geometry(
    observation: Mapping[str, Any],
) -> tuple[float, float, float, float]:
    field = observation.get("field", {})
    if not isinstance(field, Mapping):
        raise TypeError("observation field must be a mapping")
    left = _float(field.get("x"), "field.x")
    top = _float(field.get("y"), "field.y")
    width = _float(field.get("width"), "field.width", 640.0)
    height = _float(field.get("height"), "field.height", 480.0)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("field extents must be positive")
    return left, top, width, height


def _difficulty(observation: Mapping[str, Any]) -> tuple[int, int]:
    value = observation.get("difficulty", {})
    if not isinstance(value, Mapping):
        raise TypeError("observation difficulty must be a mapping")
    active_colors = _int(value.get("active_colors"), "active_colors", 0)
    interval = _int(value.get("spawn_interval_ticks"), "spawn interval", 100)
    if active_colors < 0 or interval <= 0:
        raise ValueError("difficulty counters are invalid")
    return active_colors, interval


@dataclass(frozen=True, slots=True)
class EconomicState:
    """Visible renewable-survival economics at one decision boundary."""

    tick: int
    score: int
    level: int
    gauge: int
    gauge_max: int
    gauge_fraction: float
    drain_unit: int
    drain_per_tick: int
    rot_penalty: int
    score_scale: float
    qualifying_clears: int
    clears_per_level: int
    next_level_clear_threshold: int
    clears_to_next_level: int
    level_progress: float
    highest_chain: int
    active_colors: int
    spawn_interval_ticks: int
    ticks_to_spawn: int
    group_gauge_potential: int
    rotten_liability: int
    expected_ticks_to_empty: float
    terminated: bool
    truncated: bool

    @classmethod
    def from_observation(
        cls, observation: Mapping[str, Any]
    ) -> "EconomicState":
        """Extract only public counters and visible group/rot state."""

        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        tick = _int(observation.get("tick"), "tick")
        score = _int(observation.get("score"), "score")
        level = max(1, _int(observation.get("level"), "level", 1))
        gauge = _int(observation.get("gauge"), "gauge")
        gauge_max = _int(observation.get("gauge_max"), "gauge_max", 40_000)
        clears = _int(
            observation.get("qualifying_clear_count"), "qualifying clears"
        )
        highest_chain = _int(
            observation.get("highest_chain"), "highest chain"
        )
        if min(tick, score, clears, highest_chain) < 0:
            raise ValueError("public counters other than gauge must be nonnegative")
        if gauge_max <= 0 or gauge > gauge_max:
            raise ValueError("gauge must not exceed gauge_max")
        difficulty = observation.get("difficulty", {})
        clears_per_level = _CLEARS_PER_LEVEL
        if isinstance(difficulty, Mapping):
            clears_per_level = _int(
                difficulty.get("qualifying_clears_per_level"),
                "qualifying clears per level",
                _CLEARS_PER_LEVEL,
            )
        if clears_per_level <= 0:
            raise ValueError("qualifying clears per level must be positive")
        active_colors, spawn_interval = _difficulty(observation)
        remainder = tick % spawn_interval
        ticks_to_spawn = 0 if remainder == 0 else spawn_interval - remainder

        # Executable normal-mode economics.  Above half gauge the first native
        # branch wins, making the nominal 5D branch unreachable.
        drain_unit = level // 10 + 1
        drain_per_tick = drain_unit * (3 if gauge > gauge_max / 2.0 else 1)
        rot_penalty = 1_800 + 20 * level
        score_scale = 4.0 * level**0.7

        grouped = _groups(_pieces(observation))
        group_potential = sum(
            700 * len(members) * (len(members) + 1) // 2
            for members in grouped.values()
        )
        rotten_count = sum(
            str(body.get("lifecycle", "")) == "rotten"
            or _int(body.get("rot_timer"), "body.rot_timer") > 0
            for body in _pieces(observation)
        )
        next_threshold = level * clears_per_level
        clears_to_next = max(0, next_threshold - clears)
        completed_in_level = clears - (level - 1) * clears_per_level
        level_progress = min(
            1.0, max(0.0, completed_in_level / clears_per_level)
        )
        return cls(
            tick=tick,
            score=score,
            level=level,
            gauge=gauge,
            gauge_max=gauge_max,
            gauge_fraction=max(0, gauge) / gauge_max,
            drain_unit=drain_unit,
            drain_per_tick=drain_per_tick,
            rot_penalty=rot_penalty,
            score_scale=score_scale,
            qualifying_clears=clears,
            clears_per_level=clears_per_level,
            next_level_clear_threshold=next_threshold,
            clears_to_next_level=clears_to_next,
            level_progress=level_progress,
            highest_chain=highest_chain,
            active_colors=active_colors,
            spawn_interval_ticks=spawn_interval,
            ticks_to_spawn=ticks_to_spawn,
            group_gauge_potential=group_potential,
            rotten_liability=rot_penalty * rotten_count,
            expected_ticks_to_empty=max(0, gauge) / drain_per_tick,
            terminated=bool(observation.get("terminated", False)),
            truncated=bool(observation.get("truncated", False)),
        )

    @property
    def alive(self) -> bool:
        return not self.terminated


@dataclass(frozen=True, slots=True)
class _BoardSummary:
    live_pieces: int
    ungrouped_fresh: int
    ungrouped_rotten: int
    active_rot: int
    group_count: int
    grouped_pieces: int
    largest_group: int
    pairable_colors: int
    extendable_colors: int
    viable_anchors: int
    imminent_expiry: int
    floor_hazards: int
    fragile_groups: int
    unmatched_hazards: int
    bonus_count: int
    safe_hit_budget: int


def _board_summary(
    observation: Mapping[str, Any], economics: EconomicState
) -> _BoardSummary:
    pieces = _pieces(observation)
    grouped = _groups(pieces)
    sources = tuple(body for body in pieces if _source(body))
    by_color: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for body in sources:
        by_color[_int(body.get("color"), "body.color", -1)].append(body)
    grouped_colors = {color for color, _ in grouped}
    _, top, _, height = _field_geometry(observation)
    floor_y = top + 0.8 * height
    imminent_limit = max(1, 2 * economics.spawn_interval_ticks)

    def on_floor(body: Mapping[str, Any]) -> bool:
        return (
            _float(body.get("y"), "body.y")
            + 0.5 * max(0.0, _float(body.get("size"), "body.size"))
            >= floor_y
        )

    viable = 0
    fragile = 0
    safe_budget = 0
    for members in grouped.values():
        lifetimes = [
            _int(body.get("remaining_lifetime"), "remaining lifetime")
            for body in members
        ]
        rotten = any(
            str(body.get("lifecycle", "")) == "rotten" for body in members
        )
        viable += (
            len(members) >= 2
            and not rotten
            and not any(on_floor(body) for body in members)
            and min(lifetimes, default=0) > imminent_limit
        )
        group_budget = sum(
            max(0, 1 - _int(body.get("projectile_hits"), "projectile hits"))
            for body in members
        )
        safe_budget += group_budget
        fragile += group_budget == 0
    unmatched = sum(
        len(members) == 1 and color not in grouped_colors
        for color, members in by_color.items()
    )
    return _BoardSummary(
        live_pieces=len(pieces),
        ungrouped_fresh=len(sources),
        ungrouped_rotten=sum(
            str(body.get("lifecycle", "")) == "rotten"
            and _int(body.get("chain_id"), "body.chain_id") == 0
            for body in pieces
        ),
        active_rot=sum(
            _int(body.get("rot_timer"), "body.rot_timer") > 0 for body in pieces
        ),
        group_count=len(grouped),
        grouped_pieces=sum(map(len, grouped.values())),
        largest_group=max(map(len, grouped.values()), default=0),
        pairable_colors=sum(len(members) >= 2 for members in by_color.values()),
        extendable_colors=sum(
            bool(members) and color in grouped_colors
            for color, members in by_color.items()
        ),
        viable_anchors=viable,
        imminent_expiry=sum(
            _int(body.get("remaining_lifetime"), "remaining lifetime")
            <= imminent_limit
            for body in pieces
        ),
        floor_hazards=sum(
            on_floor(body)
            and (
                _int(body.get("chain_id"), "body.chain_id") == 0
                or str(body.get("lifecycle", "")) == "rotten"
            )
            for body in pieces
        ),
        fragile_groups=fragile,
        unmatched_hazards=unmatched,
        bonus_count=sum(
            body.get("kind") == "bonus"
            and str(body.get("lifecycle", ""))
            in {"scripted_falling", "dynamic_fresh", "falling", "fresh"}
            for body in _bodies(observation)
        ),
        safe_hit_budget=safe_budget,
    )


FEATURE_NAMES: tuple[str, ...] = (
    "level_fraction",
    "log_score",
    "gauge_fraction",
    "gauge_above_half",
    "drain_unit",
    "drain_fraction_per_spawn",
    "log_runway_spawns",
    "rot_penalty_fraction",
    "group_gauge_potential_fraction",
    "renewable_margin_per_spawn",
    "score_scale",
    "level_progress",
    "clears_remaining_fraction",
    "highest_chain",
    "active_colors",
    "spawn_phase",
    "ticks_to_spawn_fraction",
    "live_piece_fraction",
    "ungrouped_fresh_fraction",
    "ungrouped_rotten_fraction",
    "active_rot_fraction",
    "group_count_fraction",
    "grouped_piece_fraction",
    "largest_group_fraction",
    "pairable_color_fraction",
    "extendable_color_fraction",
    "viable_anchor_fraction",
    "imminent_expiry_fraction",
    "floor_hazard_fraction",
    "fragile_group_fraction",
    "unmatched_hazard_fraction",
    "bonus_fraction",
    "safe_hit_budget_fraction",
    "terminal",
)


def feature_vector(observation: Mapping[str, Any]) -> np.ndarray:
    """Return deterministic, bounded-scale public option-value features."""

    economics = EconomicState.from_observation(observation)
    board = _board_summary(observation, economics)
    interval = economics.spawn_interval_ticks
    runway_spawns = economics.expected_ticks_to_empty / interval
    drain_per_spawn = economics.drain_per_tick * interval
    renewable_margin = (
        economics.group_gauge_potential - drain_per_spawn
    ) / economics.gauge_max
    values = (
        economics.level / 100.0,
        math.log1p(economics.score) / 16.0,
        economics.gauge_fraction,
        float(economics.gauge_fraction > 0.5),
        economics.drain_unit / 11.0,
        drain_per_spawn / economics.gauge_max,
        math.log1p(max(0.0, runway_spawns)) / 8.0,
        economics.rot_penalty / economics.gauge_max,
        economics.group_gauge_potential / economics.gauge_max,
        renewable_margin,
        economics.score_scale / 100.0,
        economics.level_progress,
        economics.clears_to_next_level / economics.clears_per_level,
        math.log1p(economics.highest_chain) / 8.0,
        economics.active_colors / 6.0,
        (economics.tick % interval) / interval,
        economics.ticks_to_spawn / interval,
        board.live_pieces / 20.0,
        board.ungrouped_fresh / 20.0,
        board.ungrouped_rotten / 20.0,
        board.active_rot / 20.0,
        board.group_count / 10.0,
        board.grouped_pieces / 20.0,
        board.largest_group / 20.0,
        board.pairable_colors / 6.0,
        board.extendable_colors / 6.0,
        board.viable_anchors / 10.0,
        board.imminent_expiry / 20.0,
        board.floor_hazards / 20.0,
        board.fragile_groups / 10.0,
        board.unmatched_hazards / 10.0,
        board.bonus_count / 5.0,
        board.safe_hit_budget / 20.0,
        float(economics.terminated),
    )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(FEATURE_NAMES),) or not np.isfinite(result).all():
        raise ValueError("option feature vector is invalid")
    return result


def _fresh_by_color(
    observation: Mapping[str, Any],
) -> dict[int, tuple[Mapping[str, Any], ...]]:
    result: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for body in _pieces(observation):
        if _source(body):
            result[_int(body.get("color"), "body.color", -1)].append(body)
    return {
        color: tuple(
            sorted(members, key=lambda body: _int(body.get("id"), "body.id"))
        )
        for color, members in result.items()
    }


def _ejectable(
    observation: Mapping[str, Any],
    body: Mapping[str, Any],
    config: SteeringExpertConfig,
) -> bool:
    if not _source(body):
        return False
    remaining = _int(body.get("remaining_lifetime"), "remaining lifetime")
    if _int(body.get("rot_timer"), "body.rot_timer") > 0 or (
        0 < remaining <= config.hazard_remaining_ticks
    ):
        return True
    left, top, width, height = _field_geometry(observation)
    x = _float(body.get("x"), "body.x")
    y = _float(body.get("y"), "body.y")
    edge = min(x - left, left + width - x) <= config.unmatched_edge_fraction * width
    floor = y >= top + config.unmatched_floor_fraction * height
    color = _int(body.get("color"), "body.color", -1)
    same_color = sum(
        _int(piece.get("color"), "body.color", -1) == color
        for piece in _pieces(observation)
    )
    return edge and floor and same_color < 2


def applicable_options(observation: Mapping[str, Any]) -> tuple[Option, ...]:
    """Return legal options in the canonical deterministic order."""

    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")
    if bool(observation.get("terminated", False)):
        return (Option.WAIT,)
    pieces = _pieces(observation)
    fresh = _fresh_by_color(observation)
    grouped = _groups(pieces)
    grouped_colors = {
        color
        for (color, _), members in grouped.items()
        if any(str(body.get("lifecycle", "")) != "rotten" for body in members)
    }
    rotten_colors = {
        _int(body.get("color"), "body.color", -1)
        for body in pieces
        if str(body.get("lifecycle", "")) == "rotten"
    }
    config = make_expert_config()
    legal = {
        Option.BUILD_ANCHOR: any(len(members) >= 2 for members in fresh.values()),
        Option.EXTEND_ANCHOR: any(
            members and color in grouped_colors
            for color, members in fresh.items()
        ),
        Option.HARVEST: any(
            members and color in grouped_colors
            for color, members in fresh.items()
        ),
        Option.ROTTEN_TRIAGE: any(
            members and color in rotten_colors for color, members in fresh.items()
        ),
        Option.EJECT: any(
            _ejectable(observation, body, config)
            for members in fresh.values()
            for body in members
        ),
        Option.WAIT: True,
        Option.BONUS_CLEAR: bool(
            any(fresh.values())
            and any(
                body.get("kind") == "bonus"
                and str(body.get("lifecycle", ""))
                in {"scripted_falling", "dynamic_fresh", "falling", "fresh"}
                for body in _bodies(observation)
            )
        ),
    }
    return tuple(option for option in OPTION_ORDER if legal[option])


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _option(value: Option | str) -> Option:
    if isinstance(value, Option):
        return value
    if not isinstance(value, str):
        raise TypeError("option must be an Option or option value")
    return Option(value)


def _training_sample(
    value: object,
) -> tuple[np.ndarray, Option, float]:
    features: object
    option: object
    target: object
    if isinstance(value, Mapping):
        observation = value.get("observation")
        features = (
            feature_vector(observation)
            if isinstance(observation, Mapping)
            else value.get("features")
        )
        option = value.get("option")
        target = value.get("utility", value.get("target", value.get("value")))
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
    ):
        first, second, target = value
        if isinstance(first, Mapping):
            features, option = feature_vector(first), second
        elif isinstance(second, Mapping):
            option, features = first, feature_vector(second)
        else:
            features, option = first, second
    else:
        observation = getattr(value, "observation", None)
        features = (
            feature_vector(observation)
            if isinstance(observation, Mapping)
            else getattr(value, "features", None)
        )
        option = getattr(value, "option", None)
        target = getattr(
            value, "utility", getattr(value, "target", getattr(value, "value", None))
        )
    resolved_features = np.asarray(features, dtype=np.float64)
    if (
        resolved_features.shape != (len(FEATURE_NAMES),)
        or not np.isfinite(resolved_features).all()
    ):
        raise ValueError("training sample features differ from the public schema")
    resolved_option = _option(option)  # type: ignore[arg-type]
    resolved_target = _float(target, "training target")
    return resolved_features, resolved_option, resolved_target


class OptionValueModel:
    """A deterministic independent ridge regressor for each macro option."""

    def __init__(
        self,
        coefficients: Mapping[Option | str, Sequence[float]] | None = None,
        *,
        ridge: float = 0.0,
        sample_counts: Mapping[Option | str, int] | None = None,
    ) -> None:
        self._ridge = _float(ridge, "ridge")
        if self._ridge < 0.0:
            raise ValueError("ridge must be nonnegative")
        width = len(FEATURE_NAMES) + 1
        raw_coefficients = {} if coefficients is None else coefficients
        resolved: dict[Option, np.ndarray] = {}
        for option in OPTION_ORDER:
            raw = next(
                (
                    value
                    for key, value in raw_coefficients.items()
                    if _option(key) is option
                ),
                (0.0,) * width,
            )
            vector = np.asarray(tuple(raw), dtype=np.float64)
            if vector.shape != (width,) or not np.isfinite(vector).all():
                raise ValueError(f"invalid coefficients for {option.value}")
            vector = vector.copy()
            vector[vector == 0.0] = 0.0
            vector.setflags(write=False)
            resolved[option] = vector
        self._coefficients = resolved
        raw_counts = {} if sample_counts is None else sample_counts
        self._sample_counts = {
            option: next(
                (
                    _int(value, f"{option.value} sample count")
                    for key, value in raw_counts.items()
                    if _option(key) is option
                ),
                0,
            )
            for option in OPTION_ORDER
        }
        if any(value < 0 for value in self._sample_counts.values()):
            raise ValueError("sample counts must be nonnegative")

    @classmethod
    def fit(
        cls,
        samples: Sequence[object],
        ridge: float = 1.0,
        minimum_samples_per_option: int = 1,
    ) -> "OptionValueModel":
        """Fit one canonical linear value function per option."""

        resolved_ridge = _float(ridge, "ridge")
        if resolved_ridge < 0.0:
            raise ValueError("ridge must be nonnegative")
        if (
            isinstance(minimum_samples_per_option, bool)
            or not isinstance(minimum_samples_per_option, int)
            or minimum_samples_per_option < 1
        ):
            raise ValueError("minimum_samples_per_option must be positive")
        grouped: dict[Option, list[tuple[np.ndarray, float]]] = defaultdict(list)
        for raw in samples:
            features, option, target = _training_sample(raw)
            grouped[option].append((features, target))

        coefficients: dict[Option, np.ndarray] = {}
        width = len(FEATURE_NAMES) + 1
        for option in OPTION_ORDER:
            rows = grouped.get(option, ())
            if len(rows) < minimum_samples_per_option:
                coefficients[option] = np.zeros(width, dtype=np.float64)
                continue
            design = np.column_stack(
                (
                    np.ones(len(rows), dtype=np.float64),
                    np.stack([features for features, _ in rows]),
                )
            )
            targets = np.asarray([target for _, target in rows], dtype=np.float64)
            gram = design.T @ design
            gram.flat[:: width + 1] += np.r_[
                0.0, np.full(width - 1, resolved_ridge)
            ]
            right = design.T @ targets
            try:
                fitted = np.linalg.solve(gram, right)
            except np.linalg.LinAlgError:
                regularizer = np.diag(
                    np.r_[0.0, np.full(width - 1, math.sqrt(resolved_ridge))]
                )
                fitted = np.linalg.lstsq(
                    np.vstack((design, regularizer)),
                    np.r_[targets, np.zeros(width)],
                    rcond=None,
                )[0]
            fitted[fitted == 0.0] = 0.0
            coefficients[option] = fitted
        return cls(
            coefficients,
            ridge=resolved_ridge,
            sample_counts={option: len(grouped.get(option, ())) for option in OPTION_ORDER},
        )

    @property
    def ridge(self) -> float:
        return self._ridge

    @property
    def sample_counts(self) -> dict[str, int]:
        return {
            option.value: self._sample_counts[option] for option in OPTION_ORDER
        }

    @property
    def coefficients(self) -> dict[str, tuple[float, ...]]:
        return {
            option.value: tuple(float(value) for value in self._coefficients[option])
            for option in OPTION_ORDER
        }

    def predict(self, observation: Mapping[str, Any], option: Option | str) -> float:
        return self.predict_features(feature_vector(observation), option)

    def predict_features(
        self, features: Sequence[float] | np.ndarray, option: Option | str
    ) -> float:
        resolved = _option(option)
        values = np.asarray(features, dtype=np.float64)
        if values.shape != (len(FEATURE_NAMES),) or not np.isfinite(values).all():
            raise ValueError("prediction features differ from the public schema")
        return float(self._coefficients[resolved] @ np.r_[1.0, values])

    @property
    def sha256(self) -> str:
        return str(self.manifest()["model_sha256"])

    def _payload(self) -> dict[str, object]:
        return {
            "schema": _MODEL_SCHEMA,
            "development_only": True,
            "observation_contract": "public_only",
            "economics": {
                "drain_unit": "floor(level/10)+1",
                "above_half_drain_multiplier": 3,
                "rot_penalty": "1800+20*level",
                "group_gauge_potential": "700*n*(n+1)/2",
                "score_scale": "4*level**0.7",
                "qualifying_clears_per_level": _CLEARS_PER_LEVEL,
            },
            "feature_names": list(FEATURE_NAMES),
            "option_order": [option.value for option in OPTION_ORDER],
            "ridge": self._ridge,
            "sample_counts": self.sample_counts,
            "coefficients": {
                option.value: [
                    float(value) for value in self._coefficients[option]
                ]
                for option in OPTION_ORDER
            },
        }

    def manifest(self) -> dict[str, object]:
        payload = self._payload()
        return {
            **payload,
            "model_sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest(),
        }

    def save(self, path: str | Path) -> dict[str, object]:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        manifest = self.manifest()
        target.write_bytes(_canonical_bytes(manifest) + b"\n")
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> "OptionValueModel":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or raw.get("schema") != _MODEL_SCHEMA:
            raise ValueError("not a hierarchical option-value model")
        payload = dict(raw)
        identity = payload.pop("model_sha256", None)
        if identity != hashlib.sha256(_canonical_bytes(payload)).hexdigest():
            raise ValueError("option-value model identity mismatch")
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise ValueError("option-value feature schema mismatch")
        if payload.get("option_order") != [
            option.value for option in OPTION_ORDER
        ]:
            raise ValueError("option order mismatch")
        coefficients = payload.get("coefficients")
        counts = payload.get("sample_counts")
        if not isinstance(coefficients, Mapping) or not isinstance(counts, Mapping):
            raise ValueError("model coefficients and sample counts must be mappings")
        return cls(
            coefficients,  # type: ignore[arg-type]
            ridge=_float(payload.get("ridge"), "ridge"),
            sample_counts=counts,  # type: ignore[arg-type]
        )


def make_expert_config(**overrides: Any) -> SteeringExpertConfig:
    """Construct the audited analytic micro-controller configuration."""

    base: dict[str, Any] = {
        "observe_ticks": 16,
        "resolution_wait_ticks": 4,
        "abandon_ticks": 32,
        "impact_side_sizes": 0.50,
        "impact_below_sizes": 0.75,
        "source_velocity_lead_ticks": 1.0,
        "minimum_pair_closure_sizes": 0.05,
        "ticks_per_second": 50.0,
        "hazard_remaining_ticks": 48,
        "enable_bonus": False,
        "enable_rotten_matching": True,
        "enable_hazard_ejection": False,
        "enable_edge_ejection": False,
    }
    base.update(overrides)
    return SteeringExpertConfig(**base)


def _resolve_expert_config(
    value: SteeringExpertConfig | Mapping[str, Any] | None,
) -> SteeringExpertConfig:
    if value is None:
        return make_expert_config()
    if isinstance(value, SteeringExpertConfig):
        return value
    if isinstance(value, Mapping):
        return make_expert_config(**dict(value))
    raise TypeError("expert_config must be a SteeringExpertConfig or mapping")


def _ids(members: Sequence[Mapping[str, Any]]) -> set[int]:
    return {_int(body.get("id"), "body.id") for body in members}


def _selected_piece_ids(
    observation: Mapping[str, Any],
    option: Option,
    config: SteeringExpertConfig,
) -> set[int]:
    pieces = _pieces(observation)
    fresh = _fresh_by_color(observation)
    grouped = _groups(pieces)
    if option is Option.BUILD_ANCHOR:
        candidates = [
            (color, members) for color, members in fresh.items() if len(members) >= 2
        ]
        if not candidates:
            return set()
        _, members = min(
            candidates,
            key=lambda value: (
                -len(value[1]),
                min(
                    _int(body.get("remaining_lifetime"), "remaining lifetime")
                    for body in value[1]
                ),
                value[0],
            ),
        )
        return _ids(members)

    group_candidates: list[
        tuple[int, int, tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]
    ] = []
    for (color, chain), members in grouped.items():
        destinations = tuple(
            body
            for body in members
            if str(body.get("lifecycle", "")) != "rotten"
        )
        if destinations and fresh.get(color):
            group_candidates.append((color, chain, destinations, fresh[color]))
    if option in {Option.EXTEND_ANCHOR, Option.HARVEST}:
        if not group_candidates:
            return set()

        def group_key(
            value: tuple[
                int,
                int,
                tuple[Mapping[str, Any], ...],
                tuple[Mapping[str, Any], ...],
            ]
        ) -> tuple[float, ...]:
            color, chain, destinations, _ = value
            minimum_lifetime = min(
                _int(body.get("remaining_lifetime"), "remaining lifetime")
                for body in destinations
            )
            if option is Option.EXTEND_ANCHOR:
                return (
                    float(len(destinations)),
                    -float(minimum_lifetime),
                    float(color),
                    float(chain),
                )
            return (
                -float(len(destinations)),
                float(minimum_lifetime),
                float(color),
                float(chain),
            )

        _, _, destinations, sources = min(group_candidates, key=group_key)
        return _ids(destinations) | _ids(sources)

    if option is Option.ROTTEN_TRIAGE:
        rotten = [
            body
            for body in pieces
            if str(body.get("lifecycle", "")) == "rotten"
            and fresh.get(_int(body.get("color"), "body.color", -1))
        ]
        if not rotten:
            return set()
        destination = min(
            rotten,
            key=lambda body: (
                _int(body.get("remaining_lifetime"), "remaining lifetime"),
                -_float(body.get("y"), "body.y"),
                _int(body.get("id"), "body.id"),
            ),
        )
        color = _int(destination.get("color"), "body.color", -1)
        return {_int(destination.get("id"), "body.id")} | _ids(fresh[color])

    if option is Option.EJECT:
        candidates = [
            body
            for members in fresh.values()
            for body in members
            if _ejectable(observation, body, config)
        ]
        if not candidates:
            return set()
        _, top, _, height = _field_geometry(observation)
        source = min(
            candidates,
            key=lambda body: (
                0 if _int(body.get("rot_timer"), "body.rot_timer") > 0 else 1,
                _int(body.get("remaining_lifetime"), "remaining lifetime"),
                -(_float(body.get("y"), "body.y") - top) / height,
                _int(body.get("id"), "body.id"),
            ),
        )
        return {_int(source.get("id"), "body.id")}

    if option is Option.BONUS_CLEAR:
        return _ids(tuple(body for members in fresh.values() for body in members))
    return set()


def _project(
    observation: Mapping[str, Any],
    option: Option,
    config: SteeringExpertConfig,
) -> dict[str, Any]:
    selected = _selected_piece_ids(observation, option, config)
    result = dict(observation)
    result["bodies"] = tuple(
        body
        for body in _bodies(observation)
        if body.get("kind") != "piece"
        or _int(body.get("id"), "body.id") in selected
    )
    return result


@dataclass(frozen=True, slots=True)
class _PublicSignature:
    body_ids: tuple[int, ...]
    groups: tuple[tuple[int, int], ...]
    lifecycles: tuple[tuple[int, str], ...]
    rot_active: tuple[int, ...]
    damage: tuple[tuple[int, int], ...]
    bonus_ids: tuple[int, ...]
    score: int
    level: int
    clears: int
    terminal: bool


@dataclass(frozen=True, slots=True)
class _OptionEpoch:
    option: Option
    start_tick: int
    body_ids: frozenset[int]
    chains: tuple[tuple[int, int], ...]
    lifecycles: tuple[tuple[int, str], ...]
    rot_active: frozenset[int]
    bonus_ids: frozenset[int]
    score: int
    clears: int
    gauge: int
    bound_ids: frozenset[int]


def _signature(observation: Mapping[str, Any]) -> _PublicSignature:
    bodies = _bodies(observation)
    pieces = tuple(body for body in bodies if body.get("kind") == "piece")
    return _PublicSignature(
        body_ids=tuple(sorted(_int(body.get("id"), "body.id") for body in bodies)),
        groups=tuple(
            sorted(
                (
                    _int(body.get("id"), "body.id"),
                    _int(body.get("chain_id"), "body.chain_id"),
                )
                for body in pieces
            )
        ),
        lifecycles=tuple(
            sorted(
                (
                    _int(body.get("id"), "body.id"),
                    str(body.get("lifecycle", "")),
                )
                for body in pieces
            )
        ),
        rot_active=tuple(
            sorted(
                _int(body.get("id"), "body.id")
                for body in pieces
                if _int(body.get("rot_timer"), "body.rot_timer") > 0
            )
        ),
        damage=tuple(
            sorted(
                (
                    _int(body.get("id"), "body.id"),
                    _int(body.get("projectile_hits"), "projectile hits"),
                )
                for body in pieces
            )
        ),
        bonus_ids=tuple(
            sorted(
                _int(body.get("id"), "body.id")
                for body in bodies
                if body.get("kind") == "bonus"
            )
        ),
        score=_int(observation.get("score"), "score"),
        level=max(1, _int(observation.get("level"), "level", 1)),
        clears=_int(
            observation.get("qualifying_clear_count"), "qualifying clears"
        ),
        terminal=bool(observation.get("terminated", False)),
    )


def event_signature(observation: Mapping[str, Any]) -> tuple[object, ...]:
    """Return a public-only event key suitable for collection scheduling."""

    value = _signature(observation)
    return (
        value.body_ids,
        value.groups,
        value.lifecycles,
        value.rot_active,
        value.damage,
        value.bonus_ids,
        value.score,
        value.level,
        value.clears,
        value.terminal,
    )


def _expected_gauge_after(before: EconomicState, elapsed: int) -> int:
    gauge = before.gauge
    remaining = max(0, elapsed)
    half = before.gauge_max / 2.0
    if gauge > half and remaining:
        above_steps = min(
            remaining,
            math.ceil((gauge - half) / (3 * before.drain_unit)),
        )
        gauge = max(1, gauge - above_steps * 3 * before.drain_unit)
        remaining -= above_steps
    return max(1, gauge - remaining * before.drain_unit) if remaining else gauge


def _observable_events(
    previous: _PublicSignature | None,
    current: _PublicSignature,
    before: EconomicState | None,
    after: EconomicState,
) -> tuple[str, ...]:
    if previous is None or before is None:
        return ("initial",)
    events: list[str] = []
    old_ids, new_ids = set(previous.body_ids), set(current.body_ids)
    if new_ids - old_ids:
        events.append("spawn")
    if old_ids - new_ids:
        events.append("body_removed")
    for name, first, second in (
        ("group_change", previous.groups, current.groups),
        ("lifecycle_change", previous.lifecycles, current.lifecycles),
        ("rot_change", previous.rot_active, current.rot_active),
        ("damage", previous.damage, current.damage),
        ("bonus_change", previous.bonus_ids, current.bonus_ids),
    ):
        if first != second:
            events.append(name)
    if previous.score != current.score:
        events.append("score")
    if previous.level != current.level:
        events.append("level")
    if previous.clears != current.clears:
        events.append("qualifying_clear")
    if previous.terminal != current.terminal:
        events.append("terminal")
    elapsed = after.tick - before.tick
    if after.gauge != _expected_gauge_after(before, elapsed):
        events.append("gauge_reward_or_penalty")
    return tuple(events)


class HierarchicalOptionPolicy:
    """Event-driven option selection with an explicitly abstaining residual."""

    def __init__(
        self,
        model: OptionValueModel,
        expert_config: SteeringExpertConfig | Mapping[str, Any] | None = None,
        exploration_epsilon: float = 0.0,
        forced_option: Option | str | None = None,
        *,
        option_commit_ticks: int = 96,
        minimum_option_dwell_ticks: int | None = None,
        maximum_option_dwell_ticks: int = 384,
        low_gauge_fraction: float = 0.20,
        conservative_advantage_margin: float = 0.0,
        minimum_option_samples: int = 4,
        residual_mode: str = "high_advantage_override",
        default_controller: object | None = None,
        base_wait_only: bool = True,
    ) -> None:
        if not isinstance(model, OptionValueModel):
            raise TypeError("model must be an OptionValueModel")
        epsilon = _float(exploration_epsilon, "exploration epsilon")
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("exploration epsilon must be in [0, 1]")
        if (
            isinstance(option_commit_ticks, bool)
            or not isinstance(option_commit_ticks, int)
            or option_commit_ticks < 1
        ):
            raise ValueError("option_commit_ticks must be a positive integer")
        minimum_dwell = (
            option_commit_ticks
            if minimum_option_dwell_ticks is None
            else minimum_option_dwell_ticks
        )
        if (
            isinstance(minimum_dwell, bool)
            or not isinstance(minimum_dwell, int)
            or minimum_dwell < 1
            or isinstance(maximum_option_dwell_ticks, bool)
            or not isinstance(maximum_option_dwell_ticks, int)
            or maximum_option_dwell_ticks < minimum_dwell
        ):
            raise ValueError("option dwell bounds are invalid")
        low_gauge = _float(low_gauge_fraction, "low gauge fraction")
        margin = _float(
            conservative_advantage_margin, "conservative advantage margin"
        )
        if not 0.0 <= low_gauge <= 1.0 or margin < 0.0:
            raise ValueError("hierarchy gauge threshold or margin is invalid")
        if (
            isinstance(minimum_option_samples, bool)
            or not isinstance(minimum_option_samples, int)
            or minimum_option_samples < 1
        ):
            raise ValueError("minimum option samples must be positive")
        if residual_mode not in {
            "agreement_only",
            "high_advantage_override",
        }:
            raise ValueError("residual mode is unsupported")
        if not isinstance(base_wait_only, bool):
            raise TypeError("base_wait_only must be bool")
        if default_controller is not None and (
            not callable(getattr(default_controller, "reset", None))
            or not callable(getattr(default_controller, "predict", None))
        ):
            raise TypeError("default controller must implement reset and predict")
        self.model = model
        self.expert_config = _resolve_expert_config(expert_config)
        self.exploration_epsilon = epsilon
        self.option_commit_ticks = minimum_dwell
        self.minimum_option_dwell_ticks = minimum_dwell
        self.maximum_option_dwell_ticks = maximum_option_dwell_ticks
        self.low_gauge_fraction = low_gauge
        self.conservative_advantage_margin = margin
        self.minimum_option_samples = minimum_option_samples
        self.residual_mode = residual_mode
        self.base_wait_only = base_wait_only
        self._forced_option = (
            None if forced_option is None else _option(forced_option)
        )
        self._controllers = self._make_controllers()
        self._external_default = default_controller is not None
        self._default_controller = (
            ClosedLoopSteeringExpert(config=self.expert_config)
            if default_controller is None
            else default_controller
        )
        self._rng = random.Random(0)
        self.reset()

    def _make_controllers(self) -> dict[Option, ClosedLoopSteeringExpert]:
        base = self.expert_config
        configs = {
            Option.BUILD_ANCHOR: replace(
                base,
                enable_bonus=False,
                enable_rotten_matching=False,
                enable_hazard_ejection=False,
                enable_edge_ejection=False,
            ),
            Option.EXTEND_ANCHOR: replace(
                base,
                enable_bonus=False,
                enable_rotten_matching=False,
                enable_hazard_ejection=False,
                enable_edge_ejection=False,
            ),
            Option.HARVEST: replace(
                base,
                enable_bonus=False,
                enable_rotten_matching=False,
                enable_hazard_ejection=False,
                enable_edge_ejection=False,
            ),
            Option.ROTTEN_TRIAGE: replace(
                base,
                enable_bonus=False,
                enable_rotten_matching=True,
                enable_hazard_ejection=False,
                enable_edge_ejection=False,
            ),
            Option.EJECT: replace(
                base,
                enable_bonus=False,
                enable_rotten_matching=False,
                enable_hazard_ejection=True,
                enable_edge_ejection=True,
            ),
            Option.WAIT: replace(
                base,
                enable_bonus=False,
                enable_rotten_matching=False,
                enable_hazard_ejection=False,
                enable_edge_ejection=False,
            ),
            Option.BONUS_CLEAR: replace(
                base,
                enable_bonus=True,
                enable_rotten_matching=False,
                enable_hazard_ejection=False,
                enable_edge_ejection=False,
            ),
        }
        return {
            option: ClosedLoopSteeringExpert(config=config)
            for option, config in configs.items()
        }

    @property
    def forced_option(self) -> Option | None:
        return self._forced_option

    @forced_option.setter
    def forced_option(self, value: Option | str | None) -> None:
        self.set_forced_option(value)

    @property
    def current_option(self) -> Option | None:
        return self._current_option

    def set_forced_option(self, value: Option | str | None) -> None:
        resolved = None if value is None else _option(value)
        if resolved is self._forced_option:
            return
        self._forced_option = resolved
        self._current_option = None
        self._epoch = None
        self._last_tick = None
        self._last_decision = None

    def reset(self, seed: int = 0) -> None:
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 0xFFFFFFFF
        ):
            raise ValueError("policy seed must fit in uint32")
        self._rng.seed(seed)
        self._seed = seed
        for index, option in enumerate(OPTION_ORDER):
            self._controllers[option].reset((seed + index) & 0xFFFFFFFF)
        getattr(self._default_controller, "reset")(seed)
        self._current_option: Option | None = None
        self._epoch: _OptionEpoch | None = None
        self._selected_tick = 0
        self._last_termination_reason = "reset"
        self._selection_values: dict[Option, float] = {}
        self._override_used_for_selection = False
        self._last_tick: int | None = None
        self._last_decision: SteeringDecision | None = None
        self._previous_signature: _PublicSignature | None = None
        self._previous_economics: EconomicState | None = None
        self._counts: Counter[str] = Counter()
        self._events: Counter[str] = Counter()
        self._selections: Counter[Option] = Counter()
        self._executions: Counter[Option] = Counter()
        self._occupancy: Counter[Option] = Counter()
        self._abstentions: Counter[str] = Counter()

    def _choose(
        self, observation: Mapping[str, Any], legal: tuple[Option, ...]
    ) -> Option:
        if self._forced_option is not None:
            return self._forced_option
        if (
            self.exploration_epsilon > 0.0
            and self._rng.random() < self.exploration_epsilon
        ):
            return legal[self._rng.randrange(len(legal))]
        economics = EconomicState.from_observation(observation)
        urgent = frozenset(
            {
                Option.EXTEND_ANCHOR,
                Option.HARVEST,
                Option.ROTTEN_TRIAGE,
                Option.EJECT,
                Option.BONUS_CLEAR,
            }
        )
        considered = legal
        if economics.gauge_fraction <= self.low_gauge_fraction:
            pressure_options = tuple(option for option in legal if option in urgent)
            if pressure_options:
                considered = pressure_options
                self._counts["pressure_restrictions"] += 1
        self._selection_values = {
            option: self.model.predict(observation, option)
            for option in considered
        }
        values = [
            (
                self._selection_values[option],
                -OPTION_ORDER.index(option),
                option,
            )
            for option in considered
        ]
        self._counts["value_queries"] += len(values)
        return max(values, key=lambda value: (value[0], value[1]))[2]

    def _make_epoch(
        self,
        observation: Mapping[str, Any],
        option: Option,
    ) -> _OptionEpoch:
        signature = _signature(observation)
        return _OptionEpoch(
            option=option,
            start_tick=_int(observation.get("tick"), "tick"),
            body_ids=frozenset(signature.body_ids),
            chains=signature.groups,
            lifecycles=signature.lifecycles,
            rot_active=frozenset(signature.rot_active),
            bonus_ids=frozenset(signature.bonus_ids),
            score=signature.score,
            clears=signature.clears,
            gauge=_int(observation.get("gauge"), "gauge"),
            bound_ids=frozenset(
                _selected_piece_ids(
                    observation,
                    option,
                    self._controllers[option].config,
                )
            ),
        )

    def _termination_reason(
        self,
        *,
        observation: Mapping[str, Any],
        signature: _PublicSignature,
        legal: Sequence[Option],
    ) -> str | None:
        assert self._current_option is not None and self._epoch is not None
        if self._current_option not in legal:
            return "option_became_inapplicable"
        tick = _int(observation.get("tick"), "tick")
        dwell = tick - self._selected_tick
        epoch = self._epoch
        current_ids = frozenset(signature.body_ids)
        current_chains = dict(signature.groups)
        start_chains = dict(epoch.chains)
        current_lifecycles = dict(signature.lifecycles)
        start_lifecycles = dict(epoch.lifecycles)
        bound_removed = bool(epoch.bound_ids - current_ids)
        bound_chain_changed = any(
            identifier in current_chains
            and current_chains[identifier] != start_chains.get(identifier)
            for identifier in epoch.bound_ids
        )
        bound_lifecycle_changed = any(
            identifier in current_lifecycles
            and current_lifecycles[identifier]
            != start_lifecycles.get(identifier)
            for identifier in epoch.bound_ids
        )
        progress: str | None = None
        if epoch.option in {Option.BUILD_ANCHOR, Option.EXTEND_ANCHOR}:
            if bound_removed or bound_chain_changed:
                progress = "bound_join_or_removal"
        elif epoch.option is Option.HARVEST:
            if (
                bound_removed
                or bound_chain_changed
                or signature.clears > epoch.clears
                or signature.score > epoch.score
            ):
                progress = "bound_harvest_progress"
        elif epoch.option is Option.ROTTEN_TRIAGE:
            if (
                bound_removed
                or bound_lifecycle_changed
                or bool(epoch.rot_active - frozenset(signature.rot_active))
            ):
                progress = "bound_rot_progress"
        elif epoch.option is Option.EJECT:
            if bound_removed or bound_lifecycle_changed:
                progress = "bound_eject_progress"
        elif epoch.option is Option.WAIT:
            if current_ids - epoch.body_ids:
                progress = "new_body_after_wait"
        elif epoch.option is Option.BONUS_CLEAR:
            if (
                epoch.bonus_ids - frozenset(signature.bonus_ids)
                or signature.clears > epoch.clears
                or _int(observation.get("gauge"), "gauge") > epoch.gauge
            ):
                progress = "bound_bonus_progress"
        if dwell < self.minimum_option_dwell_ticks:
            if progress is not None:
                self._counts["early_termination_events_held"] += 1
            return None
        if progress is not None:
            return progress
        if dwell >= self.maximum_option_dwell_ticks:
            return "maximum_dwell_timeout"
        return None

    def _safe_execute(
        self, observation: Mapping[str, Any], option: Option
    ) -> SteeringDecision:
        controller = self._controllers[option]
        decision = controller.predict(
            _project(observation, option, controller.config)
        )
        if not decision.is_shot or decision.source_body_id is None:
            return decision
        by_id = {
            _int(body.get("id"), "body.id"): body for body in _bodies(observation)
        }
        source = by_id.get(decision.source_body_id)
        unsafe = (
            source is None
            or (
                source.get("kind") == "piece"
                and (
                    _int(source.get("chain_id"), "body.chain_id") != 0
                    or str(source.get("lifecycle", "")) == "rotten"
                )
            )
        )
        if not unsafe:
            return decision
        self._counts["safety_blocks"] += 1
        wait_controller = self._controllers[Option.WAIT]
        return wait_controller.predict(
            _project(observation, Option.WAIT, wait_controller.config)
        )

    @staticmethod
    def _default_option(decision: SteeringDecision) -> Option:
        return {
            SteeringIntent.STEER_MATCH: Option.BUILD_ANCHOR,
            SteeringIntent.EXTEND_ANCHOR: Option.EXTEND_ANCHOR,
            SteeringIntent.MATCH_ROTTEN: Option.ROTTEN_TRIAGE,
            SteeringIntent.EJECT_HAZARD: Option.EJECT,
            SteeringIntent.ACTIVATE_BONUS: Option.BONUS_CLEAR,
        }.get(decision.intent, Option.WAIT)

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        """Return one idempotent expert decision and update public audit stats."""

        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping")
        economics = EconomicState.from_observation(observation)
        tick = economics.tick
        if self._last_tick is not None:
            if tick < self._last_tick:
                raise RuntimeError("observation tick moved backwards; reset the policy")
            if tick == self._last_tick:
                assert self._last_decision is not None
                return self._last_decision

        signature = _signature(observation)
        events = _observable_events(
            self._previous_signature,
            signature,
            self._previous_economics,
            economics,
        )
        self._events.update(events)
        legal = applicable_options(observation)
        termination_reason = (
            "initial"
            if self._current_option is None
            else self._termination_reason(
                observation=observation,
                signature=signature,
                legal=legal,
            )
        )
        should_select = termination_reason is not None
        if self._forced_option is not None and self._current_option is not self._forced_option:
            should_select = True
            termination_reason = "forced_option_changed"
        if should_select:
            previous = self._current_option
            self._current_option = self._choose(observation, legal)
            self._selected_tick = tick
            self._epoch = self._make_epoch(
                observation, self._current_option
            )
            controller_index = OPTION_ORDER.index(self._current_option)
            self._controllers[self._current_option].reset(
                (self._seed + controller_index) & 0xFFFFFFFF
            )
            self._last_termination_reason = str(termination_reason)
            self._override_used_for_selection = False
            self._counts["option_decisions"] += 1
            self._counts[f"termination:{termination_reason}"] += 1
            self._selections[self._current_option] += 1
            if previous is not None and previous is not self._current_option:
                self._counts["option_switches"] += 1
        assert self._current_option is not None
        if self._forced_option is not None:
            decision = self._safe_execute(observation, self._current_option)
            self._executions[self._current_option] += 1
        else:
            raw_default = getattr(self._default_controller, "predict")(
                observation
            )
            if not isinstance(raw_default, SteeringDecision):
                raise TypeError("default controller must return SteeringDecision")
            default_decision = raw_default
            default_option = self._default_option(default_decision)
            permit_override = False
            abstention_reason: str | None = None
            if not should_select:
                self._counts["dwell_passthroughs"] += 1
            elif self._current_option is default_option:
                self._counts["default_option_matches"] += 1
                candidate = self._safe_execute(
                    observation, self._current_option
                )
                same_binding = (
                    candidate.is_shot
                    and default_decision.is_shot
                    and candidate.source_body_id
                    == default_decision.source_body_id
                    and candidate.destination_body_id
                    == default_decision.destination_body_id
                )
                if same_binding:
                    decision = candidate
                    self._counts["agreement_actuator_substitutions"] += 1
                    self._executions[self._current_option] += 1
                elif not candidate.is_shot and not default_decision.is_shot:
                    decision = default_decision
                    self._counts["agreement_passthroughs"] += 1
                else:
                    decision = default_decision
                    abstention_reason = "binding_disagreement"
            elif self.residual_mode == "agreement_only":
                abstention_reason = "explicit_option_disagreement"
            elif self.base_wait_only and default_decision.is_shot:
                abstention_reason = "base_not_waiting"
            else:
                selected_value = self._selection_values[
                    self._current_option
                ]
                default_value = self._selection_values.get(default_option)
                if default_value is None:
                    default_value = self.model.predict(
                        observation, default_option
                    )
                    self._counts["value_queries"] += 1
                advantage = selected_value - default_value
                self._counts["advantage_checks"] += 1
                sample_counts = self.model.sample_counts
                enough_support = (
                    sample_counts[self._current_option.value]
                    >= self.minimum_option_samples
                    and sample_counts[default_option.value]
                    >= self.minimum_option_samples
                )
                if not enough_support:
                    abstention_reason = "insufficient_sample_support"
                elif advantage >= self.conservative_advantage_margin:
                    permit_override = True
                else:
                    abstention_reason = "insufficient_advantage"
            if permit_override:
                candidate = self._safe_execute(
                    observation, self._current_option
                )
                if candidate.is_shot:
                    decision = candidate
                    self._override_used_for_selection = True
                    self._counts["option_overrides"] += 1
                    self._executions[self._current_option] += 1
                else:
                    decision = default_decision
                    abstention_reason = "projected_controller_waited"
                    self._counts["default_micro_fallbacks"] += 1
                    self._counts["abstentions"] += 1
                    self._counts["selection_abstentions"] += 1
                    self._abstentions[abstention_reason] += 1
            elif not (
                should_select
                and self._current_option is default_option
                and abstention_reason is None
            ):
                decision = default_decision
                if abstention_reason is not None:
                    self._counts["default_micro_fallbacks"] += 1
                    self._counts["abstentions"] += 1
                    self._counts["selection_abstentions"] += 1
                    self._abstentions[abstention_reason] += 1
        self._occupancy[self._current_option] += 1
        self._counts["micro_decisions"] += 1
        if decision.is_shot:
            self._counts["shots"] += 1
            if decision.correction_index > 1:
                self._counts["corrections"] += 1
        else:
            self._counts["waits"] += 1

        self._last_tick = tick
        self._last_decision = decision
        self._previous_signature = signature
        self._previous_economics = economics
        return decision

    def statistics(self) -> dict[str, object]:
        """Return JSON-ready selection, query, correction, and event counts."""

        return {
            "decision_count": self._counts["micro_decisions"],
            "option_decision_count": self._counts["option_decisions"],
            "query_count": self._counts["value_queries"],
            "correction_count": self._counts["corrections"],
            "shot_count": self._counts["shots"],
            "wait_count": self._counts["waits"],
            "option_switch_count": self._counts["option_switches"],
            "safety_block_count": self._counts["safety_blocks"],
            "pressure_restriction_count": self._counts[
                "pressure_restrictions"
            ],
            "advantage_fallback_count": self._counts[
                "advantage_fallbacks"
            ],
            "abstention_count": self._counts["abstentions"],
            "selection_abstention_count": self._counts[
                "selection_abstentions"
            ],
            "agreement_passthrough_count": self._counts[
                "agreement_passthroughs"
            ],
            "agreement_actuator_substitution_count": self._counts[
                "agreement_actuator_substitutions"
            ],
            "dwell_passthrough_count": self._counts[
                "dwell_passthroughs"
            ],
            "default_option_match_count": self._counts[
                "default_option_matches"
            ],
            "default_micro_fallback_count": self._counts[
                "default_micro_fallbacks"
            ],
            "option_override_count": self._counts["option_overrides"],
            "early_termination_events_held": self._counts[
                "early_termination_events_held"
            ],
            "option_queries": self._counts["value_queries"],
            "event_replans": self._counts["option_decisions"],
            "micro_corrections": self._counts["corrections"],
            "option_fallbacks": (
                self._counts["safety_blocks"]
                + self._counts["advantage_fallbacks"]
                + self._counts["default_micro_fallbacks"]
            ),
            "current_option": (
                None if self._current_option is None else self._current_option.value
            ),
            "forced_option": (
                None if self._forced_option is None else self._forced_option.value
            ),
            "residual_mode": self.residual_mode,
            "base_controller": (
                "external" if self._external_default else "analytic"
            ),
            "minimum_option_dwell_ticks": self.minimum_option_dwell_ticks,
            "maximum_option_dwell_ticks": self.maximum_option_dwell_ticks,
            "minimum_option_samples": self.minimum_option_samples,
            "last_termination_reason": self._last_termination_reason,
            "abstention_reasons": dict(sorted(self._abstentions.items())),
            "option_selections": {
                option.value: self._selections[option] for option in OPTION_ORDER
            },
            "option_executions": {
                option.value: self._executions[option] for option in OPTION_ORDER
            },
            "option_occupancy": {
                option.value: self._occupancy[option] for option in OPTION_ORDER
            },
            "events": dict(sorted(self._events.items())),
        }


def _event_total(event_counts: Mapping[object, object], needle: str) -> int:
    total = 0
    for raw_key, raw_value in event_counts.items():
        key = getattr(raw_key, "value", raw_key)
        name = str(key).lower()
        value = _int(raw_value, f"{name} event count")
        if value < 0:
            raise ValueError("event counts must be nonnegative")
        if needle in name:
            total += value
    return total


@dataclass(frozen=True, slots=True)
class UtilityWeights:
    """Auditable coefficients for the offline multi-spawn branch target."""

    alive_horizon_bonus: float = 100_000.0
    survival_tick: float = 8.0
    gauge: float = 1.0
    renewable_group_credit: float = 0.25
    rot_liability: float = -1.0
    score_delta: float = 8.0
    qualifying_clear_delta: float = 100.0
    rotten_event: float = -2_200.0
    ejected_event: float = -20.0
    terminal_lost_tick: float = -24.0

    def __post_init__(self) -> None:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError("utility weights must be finite numeric values")


def branch_utility(
    before: Mapping[str, Any] | EconomicState,
    after: Mapping[str, Any] | EconomicState,
    elapsed_ticks: int,
    horizon_ticks: int,
    event_counts: Mapping[object, object],
    weights: UtilityWeights | None = None,
) -> float:
    """Score a portable public branch with explicit renewable economics."""

    if (
        isinstance(elapsed_ticks, bool)
        or not isinstance(elapsed_ticks, int)
        or elapsed_ticks < 0
    ):
        raise ValueError("elapsed_ticks must be a nonnegative integer")
    if (
        isinstance(horizon_ticks, bool)
        or not isinstance(horizon_ticks, int)
        or horizon_ticks < 1
    ):
        raise ValueError("horizon_ticks must be a positive integer")
    if not isinstance(event_counts, Mapping):
        raise TypeError("event_counts must be a mapping")
    start = (
        before
        if isinstance(before, EconomicState)
        else EconomicState.from_observation(before)
    )
    end = (
        after
        if isinstance(after, EconomicState)
        else EconomicState.from_observation(after)
    )
    resolved = UtilityWeights() if weights is None else weights
    if not isinstance(resolved, UtilityWeights):
        raise TypeError("weights must be UtilityWeights")
    survived_horizon = not end.terminated and elapsed_ticks >= horizon_ticks
    lost_ticks = max(0, horizon_ticks - elapsed_ticks) if end.terminated else 0
    return float(
        resolved.alive_horizon_bonus * survived_horizon
        + resolved.survival_tick * elapsed_ticks
        + resolved.gauge * (end.gauge - start.gauge)
        + resolved.renewable_group_credit
        * (end.group_gauge_potential - start.group_gauge_potential)
        + resolved.rot_liability
        * (end.rotten_liability - start.rotten_liability)
        + resolved.score_delta * (end.score - start.score)
        + resolved.qualifying_clear_delta
        * (end.qualifying_clears - start.qualifying_clears)
        + resolved.rotten_event * _event_total(event_counts, "rotten")
        + resolved.ejected_event * _event_total(event_counts, "eject")
        + resolved.terminal_lost_tick * lost_ticks
    )


__all__ = [
    "EconomicState",
    "FEATURE_NAMES",
    "HierarchicalOptionPolicy",
    "OPTION_ORDER",
    "Option",
    "OptionValueModel",
    "UtilityWeights",
    "applicable_options",
    "branch_utility",
    "event_signature",
    "feature_vector",
    "make_expert_config",
]
