"""Bounded development-only joint pair and shot-geometry planning.

The planner consumes public observations and opaque portable snapshots.  It
never exposes snapshot bytes or future RNG state to a policy or learner.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch

from irisu_env import Action, ActionKind, EventKind
from irisu_rl.actions import ActionSpec, SemanticAction, SemanticActionKind
from irisu_rl.encoding import TeacherStateEncoder

from .action import PointerActionSpec
from .policy import encoded_body_ids
from .steering import SteeringDecision, SteeringIntent
from .steering_learning import GoalConditionedSteeringModel


JOINT_PLANNER_VERSION = "irisu-joint-pair-geometry-planner-v2"
_SOURCE_LIFECYCLES = frozenset(
    {"scripted_falling", "dynamic_fresh", "falling", "fresh"}
)
_DESTINATION_LIFECYCLES = _SOURCE_LIFECYCLES | frozenset(
    {"confirmed", "rotten"}
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _plain_int(value: Any, name: str, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    item = getattr(value, "item", None)
    if callable(item):
        return _plain_int(item(), name, default)
    raise TypeError(f"{name} must be an integer")


def _plain_float(value: Any, name: str, default: float = 0.0) -> float:
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


def _event_kind(event: Mapping[str, Any]) -> int | None:
    raw = event.get("kind")
    if isinstance(raw, Integral) and not isinstance(raw, bool):
        return int(raw)
    name = event.get("kind_name")
    if isinstance(name, str):
        try:
            return int(EventKind[name.upper()])
        except KeyError:
            return None
    return None


def _bodies(observation: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    supplied = observation.get("bodies", ())
    if not isinstance(supplied, Sequence) or isinstance(
        supplied, (str, bytes)
    ):
        raise TypeError("public bodies must be a sequence")
    values = tuple(value for value in supplied if isinstance(value, Mapping))
    identifiers = [_plain_int(value.get("id"), "body id", -1) for value in values]
    if any(value < 0 for value in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("public body IDs must be unique and nonnegative")
    return values


def _by_id(
    observation: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    return {
        _plain_int(body.get("id"), "body id"): body
        for body in _bodies(observation)
    }


def _position(body: Mapping[str, Any]) -> tuple[float, float, float]:
    size = _plain_float(body.get("size"), "body size")
    if size <= 0.0:
        raise ValueError("public body size must be positive")
    return (
        _plain_float(body.get("effect_x", body.get("x")), "body x"),
        _plain_float(body.get("effect_y", body.get("y")), "body y"),
        size,
    )


def _public_signature(observation: Mapping[str, Any]) -> str:
    body_values = []
    for body in _bodies(observation):
        body_values.append(
            (
                _plain_int(body.get("id"), "body id"),
                str(body.get("kind", "")),
                str(body.get("shape", "")),
                str(body.get("lifecycle", "")),
                _plain_int(body.get("color"), "body color", -1),
                _plain_float(body.get("x"), "body x"),
                _plain_float(body.get("y"), "body y"),
                _plain_float(body.get("vx"), "body vx"),
                _plain_float(body.get("vy"), "body vy"),
                _plain_float(body.get("angle"), "body angle"),
                _plain_float(
                    body.get("angular_velocity"), "body angular velocity"
                ),
                _plain_float(body.get("size"), "body size"),
                _plain_int(body.get("chain_id"), "chain id"),
                _plain_int(body.get("projectile_hits"), "projectile hits"),
                _plain_int(
                    body.get("remaining_lifetime"), "remaining lifetime"
                ),
                _plain_int(body.get("rot_timer"), "rot timer"),
            )
        )
    body_values.sort()
    return _canonical_sha256(
        {
            "tick": _plain_int(observation.get("tick"), "tick"),
            "score": _plain_int(observation.get("score"), "score"),
            "gauge": _plain_int(observation.get("gauge"), "gauge"),
            "gauge_max": _plain_int(
                observation.get("gauge_max"), "gauge max"
            ),
            "level": _plain_int(observation.get("level"), "level"),
            "highest_chain": _plain_int(
                observation.get("highest_chain"), "highest chain"
            ),
            "qualifying_clears": _plain_int(
                observation.get("qualifying_clear_count"),
                "qualifying clear count",
            ),
            "terminated": bool(observation.get("terminated", False)),
            "truncated": bool(observation.get("truncated", False)),
            "bodies": body_values,
        }
    )


@dataclass(frozen=True, slots=True)
class GeometryOption:
    name: str
    strength: str
    side_sizes: float
    below_sizes: float

    def __post_init__(self) -> None:
        if not self.name or self.strength not in {"weak", "strong"}:
            raise ValueError("joint geometry option is malformed")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (self.side_sizes, self.below_sizes)
        ):
            raise ValueError("joint geometry scales must be positive")

    def manifest(self) -> dict[str, object]:
        return {
            "name": self.name,
            "strength": self.strength,
            "side_sizes": float(self.side_sizes),
            "below_sizes": float(self.below_sizes),
        }


COMPACT_GEOMETRY: tuple[GeometryOption, ...] = (
    GeometryOption("analytic-strong", "strong", 0.50, 0.75),
    GeometryOption("close-strong", "strong", 0.25, 0.75),
    GeometryOption("wide-strong", "strong", 0.75, 0.75),
    GeometryOption("deep-strong", "strong", 0.50, 1.00),
    GeometryOption("analytic-weak", "weak", 0.50, 0.75),
)


@dataclass(frozen=True, slots=True)
class JointPlannerConfig:
    pair_cap: int = 3
    geometry_cap: int = 4
    horizons: tuple[int, ...] = (48, 160)
    cooldown_ticks: int = 16
    velocity_lead_ticks: float = 1.0
    ticks_per_second: float = 50.0
    require_pristine_source: bool = True

    def __post_init__(self) -> None:
        if type(self.pair_cap) is not int or not 1 <= self.pair_cap <= 8:
            raise ValueError("joint pair cap must be in [1, 8]")
        if (
            type(self.geometry_cap) is not int
            or not 1 <= self.geometry_cap <= len(COMPACT_GEOMETRY)
        ):
            raise ValueError("joint geometry cap is invalid")
        if (
            not self.horizons
            or tuple(sorted(set(self.horizons))) != self.horizons
            or any(type(value) is not int or value < 2 for value in self.horizons)
        ):
            raise ValueError("joint horizons must be unique and increasing")
        if (
            type(self.cooldown_ticks) is not int
            or self.cooldown_ticks < 2
            or self.cooldown_ticks > self.horizons[0]
        ):
            raise ValueError("joint cooldown must fit before the first horizon")
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (self.ticks_per_second,)
        ):
            raise ValueError("joint tick rate must be positive")
        if (
            isinstance(self.velocity_lead_ticks, bool)
            or not isinstance(self.velocity_lead_ticks, Real)
            or not math.isfinite(float(self.velocity_lead_ticks))
            or float(self.velocity_lead_ticks) < 0.0
        ):
            raise ValueError("joint velocity lead must be nonnegative")
        if type(self.require_pristine_source) is not bool:
            raise TypeError("joint pristine-source switch must be boolean")

    @property
    def branch_cap(self) -> int:
        return self.pair_cap * self.geometry_cap

    def manifest(self) -> dict[str, object]:
        return {
            "version": JOINT_PLANNER_VERSION,
            "pair_cap": self.pair_cap,
            "geometry_cap": self.geometry_cap,
            "horizons": list(self.horizons),
            "cooldown_ticks": self.cooldown_ticks,
            "velocity_lead_ticks": float(self.velocity_lead_ticks),
            "ticks_per_second": float(self.ticks_per_second),
            "require_pristine_source": self.require_pristine_source,
            "geometry": [
                value.manifest()
                for value in COMPACT_GEOMETRY[: self.geometry_cap]
            ],
            "geometry_lowering": (
                "nominal body-size offsets plus deterministic public velocity "
                "lead; student scores the nearest pointer template to each "
                "fully lowered action"
            ),
            "branch_cap": self.branch_cap,
            "selection": (
                "all-horizon invalid/alive/survival/final-gauge "
                "nondominance; long-to-short survival, qualifying clears, "
                "positive gauge renewal, final gauge, score, causal control"
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class PairOption:
    source_body_id: int
    destination_body_id: int
    destination_chain_id: int
    category: str
    intent: SteeringIntent
    distance_sizes: float
    incumbent: bool = False

    def manifest(self) -> dict[str, object]:
        return {
            "source_body_id": self.source_body_id,
            "destination_body_id": self.destination_body_id,
            "destination_chain_id": self.destination_chain_id,
            "category": self.category,
            "intent": self.intent.value,
            "distance_sizes": self.distance_sizes,
            "incumbent": self.incumbent,
        }


def _viable_anchor_ids(
    observation: Mapping[str, Any],
) -> frozenset[tuple[int, int]]:
    difficulty = observation.get("difficulty", {})
    field = observation.get("field", {})
    if not isinstance(difficulty, Mapping) or not isinstance(field, Mapping):
        raise TypeError("public difficulty and field must be mappings")
    interval = _plain_int(
        difficulty.get("spawn_interval_ticks"), "spawn interval"
    )
    field_y = _plain_float(field.get("y"), "field y")
    height = _plain_float(field.get("height"), "field height")
    if interval < 1 or height <= 0.0:
        raise ValueError("public cadence or field is invalid")
    floor = field_y + 0.8 * height
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for body in _bodies(observation):
        if (
            body.get("kind") != "piece"
            or str(body.get("lifecycle", "")) not in _DESTINATION_LIFECYCLES
        ):
            continue
        chain = _plain_int(body.get("chain_id"), "chain id")
        if chain:
            key = (_plain_int(body.get("color"), "color", -1), chain)
            groups.setdefault(key, []).append(body)
    output = set()
    for key, members in groups.items():
        non_rotten = any(
            str(body.get("lifecycle", "")) != "rotten" for body in members
        )
        on_floor = any(
            _position(body)[1] + _position(body)[2] / 2.0 >= floor
            for body in members
        )
        lifetime = min(
            _plain_int(
                body.get("remaining_lifetime"), "remaining lifetime"
            )
            for body in members
        )
        if (
            len(members) >= 2
            and non_rotten
            and not on_floor
            and lifetime > 2 * interval
        ):
            output.add(key)
    return frozenset(output)


def _source_safe(
    body: Mapping[str, Any], *, require_pristine: bool
) -> bool:
    return (
        body.get("kind") == "piece"
        and str(body.get("shape", "")) in {"circle", "box", "triangle"}
        and str(body.get("lifecycle", "")) in _SOURCE_LIFECYCLES
        and _plain_int(body.get("chain_id"), "chain id") == 0
        and (
            not require_pristine
            or _plain_int(body.get("projectile_hits"), "projectile hits") == 0
        )
    )


def _destination_safe(body: Mapping[str, Any]) -> bool:
    return (
        body.get("kind") == "piece"
        and str(body.get("shape", "")) in {"circle", "box", "triangle"}
        and str(body.get("lifecycle", "")) in _DESTINATION_LIFECYCLES
    )


def _analytic_decision(
    observation: Mapping[str, Any],
    pair: PairOption,
    geometry: GeometryOption,
    config: JointPlannerConfig,
    action_spec: ActionSpec,
    *,
    reason: str,
) -> SteeringDecision | None:
    bodies = _by_id(observation)
    source = bodies.get(pair.source_body_id)
    destination = bodies.get(pair.destination_body_id)
    if source is None or destination is None:
        return None
    sx, sy, size = _position(source)
    dx, _dy, _destination_size = _position(destination)
    direction = 1.0 if dx > sx else -1.0
    if math.isclose(dx, sx, abs_tol=1e-9):
        direction = 1.0 if sx <= action_spec.client_width / 2.0 else -1.0
    lifecycle = str(source.get("lifecycle", ""))
    vx = source.get("vx_display_per_second")
    if vx is None:
        vx = _plain_float(source.get("vx"), "source vx") * (
            50.0 if lifecycle in {"scripted_falling", "falling"} else 10.0
        )
    impact_x = -direction * geometry.side_sizes
    raw_x = (
        sx
        + _plain_float(vx, "source display vx")
        * config.velocity_lead_ticks
        / config.ticks_per_second
        + impact_x * size
    )
    raw_y = sy + geometry.below_sizes * size
    if not (
        0.0 <= raw_x <= action_spec.client_width
        and 0.0 <= raw_y <= action_spec.client_height
    ):
        return None
    constructor = (
        SemanticAction.weak
        if geometry.strength == "weak"
        else SemanticAction.strong
    )
    return SteeringDecision(
        action_spec.validate(
            constructor(
                raw_x / action_spec.client_width,
                raw_y / action_spec.client_height,
            )
        ),
        pair.intent,
        source_body_id=pair.source_body_id,
        destination_body_id=pair.destination_body_id,
        destination_chain_id=pair.destination_chain_id,
        impact_x_sizes=impact_x,
        impact_y_sizes=geometry.below_sizes,
        reason=reason,
    )


def shortlist_pairs(
    observation: Mapping[str, Any],
    incumbent: SteeringDecision,
    *,
    config: JointPlannerConfig | None = None,
    action_spec: ActionSpec | None = None,
) -> tuple[PairOption, ...]:
    """Return a deterministic category-stratified legal pair shortlist."""

    resolved = JointPlannerConfig() if config is None else config
    spec = ActionSpec() if action_spec is None else action_spec
    bodies = _bodies(observation)
    viable = _viable_anchor_ids(observation)
    options: list[PairOption] = []
    for source in bodies:
        source_identifier = _plain_int(source.get("id"), "source id")
        if not _source_safe(
            source,
            require_pristine=(
                resolved.require_pristine_source
                and source_identifier != incumbent.source_body_id
            ),
        ):
            continue
        source_id = source_identifier
        color = _plain_int(source.get("color"), "source color", -1)
        sx, sy, ssize = _position(source)
        for destination in bodies:
            destination_id = _plain_int(
                destination.get("id"), "destination id"
            )
            if (
                destination_id == source_id
                or not _destination_safe(destination)
                or _plain_int(destination.get("color"), "destination color", -2)
                != color
            ):
                continue
            chain = _plain_int(destination.get("chain_id"), "chain id")
            lifecycle = str(destination.get("lifecycle", ""))
            source_rot_active = (
                _plain_int(source.get("rot_timer"), "source rot timer") > 0
            )
            if lifecycle == "rotten":
                category = "rotten-hazard"
                intent = SteeringIntent.MATCH_ROTTEN
            elif chain:
                is_incumbent = (
                    incumbent.source_body_id == source_id
                    and incumbent.destination_body_id == destination_id
                )
                if (
                    ((color, chain) not in viable and not is_incumbent)
                    or lifecycle == "rotten"
                ):
                    continue
                category = (
                    "rotten-hazard" if source_rot_active else "viable-anchor"
                )
                intent = (
                    SteeringIntent.MATCH_ROTTEN
                    if source_rot_active
                    else SteeringIntent.EXTEND_ANCHOR
                )
            else:
                category = (
                    "rotten-hazard" if source_rot_active else "fresh-match"
                )
                intent = (
                    SteeringIntent.MATCH_ROTTEN
                    if source_rot_active
                    else SteeringIntent.STEER_MATCH
                )
            dx, dy, dsize = _position(destination)
            pair = PairOption(
                source_id,
                destination_id,
                chain,
                category,
                intent,
                math.hypot(dx - sx, dy - sy) / max((ssize + dsize) / 2.0, 1e-9),
                incumbent=(
                    incumbent.source_body_id == source_id
                    and incumbent.destination_body_id == destination_id
                ),
            )
            if (
                _analytic_decision(
                    observation,
                    pair,
                    COMPACT_GEOMETRY[0],
                    resolved,
                    spec,
                    reason="joint shortlist reachability check",
                )
                is not None
            ):
                options.append(pair)
    options.sort(
        key=lambda value: (
            value.distance_sizes,
            value.source_body_id,
            value.destination_body_id,
        )
    )
    selected: list[PairOption] = []
    incumbent_option = next((value for value in options if value.incumbent), None)
    if incumbent_option is None:
        raise ValueError(
            "incoming incumbent is absent from the legal joint shortlist"
        )
    selected.append(incumbent_option)
    for category in ("rotten-hazard", "viable-anchor", "fresh-match"):
        for value in options:
            if (
                len(selected) >= resolved.pair_cap
                or value.category != category
                or value in selected
            ):
                continue
            selected.append(value)
            break
    for value in options:
        if len(selected) >= resolved.pair_cap:
            break
        if value not in selected:
            selected.append(value)
    if not selected:
        raise ValueError("joint planner found no legal source/destination pair")
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class JointCandidate:
    ordinal: int
    pair_ordinal: int
    geometry_ordinal: int
    pair: PairOption
    geometry: GeometryOption
    decision: SteeringDecision

    def manifest(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "pair_ordinal": self.pair_ordinal,
            "geometry_ordinal": self.geometry_ordinal,
            "pair": self.pair.manifest(),
            "geometry": self.geometry.manifest(),
            "action": {
                "kind": int(self.decision.action.kind),
                "x_norm": float(self.decision.action.x_norm),
                "y_norm": float(self.decision.action.y_norm),
            },
        }


@dataclass(frozen=True, slots=True)
class JointMilestoneOutcome:
    horizon_ticks: int
    alive: bool
    survival_ticks: int
    score_gain: int
    final_gauge: int
    final_level: int
    qualifying_clear_gain: int
    cleared_events: int
    rotten_events: int
    positive_gauge_renewal: int
    invalid_actions: int
    intended_source_hits: int
    intended_pair_joined: bool
    pair_closure_sizes: float

    def objective(self) -> tuple[int | float, ...]:
        return (
            int(self.alive),
            self.survival_ticks,
            self.qualifying_clear_gain,
            self.positive_gauge_renewal,
            self.final_gauge,
            self.score_gain,
            self.cleared_events,
            -self.rotten_events,
            int(self.intended_pair_joined),
            self.intended_source_hits,
            self.pair_closure_sizes,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "horizon_ticks": self.horizon_ticks,
            "alive": self.alive,
            "survival_ticks": self.survival_ticks,
            "score_gain": self.score_gain,
            "final_gauge": self.final_gauge,
            "final_level": self.final_level,
            "qualifying_clear_gain": self.qualifying_clear_gain,
            "cleared_events": self.cleared_events,
            "rotten_events": self.rotten_events,
            "positive_gauge_renewal": self.positive_gauge_renewal,
            "invalid_actions": self.invalid_actions,
            "intended_source_hits": self.intended_source_hits,
            "intended_pair_joined": self.intended_pair_joined,
            "pair_closure_sizes": self.pair_closure_sizes,
        }


@dataclass(frozen=True, slots=True)
class JointBranchOutcome:
    candidate: JointCandidate
    milestones: tuple[JointMilestoneOutcome, ...]
    simulated_ticks: int

    def selectable_against(self, incumbent: JointBranchOutcome) -> bool:
        if len(self.milestones) != len(incumbent.milestones):
            return False
        return all(
            value.invalid_actions == 0
            and int(value.alive) >= int(base.alive)
            and value.survival_ticks >= base.survival_ticks
            and value.final_gauge >= base.final_gauge
            for value, base in zip(
                self.milestones, incumbent.milestones, strict=True
            )
        )

    @property
    def objective(self) -> tuple[int | float, ...]:
        return tuple(
            component
            for milestone in reversed(self.milestones)
            for component in milestone.objective()
        )

    def manifest(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.manifest(),
            "milestones": [value.manifest() for value in self.milestones],
            "simulated_ticks": self.simulated_ticks,
            "objective": list(self.objective),
        }


@dataclass(frozen=True, slots=True)
class JointSearchResult:
    config_sha256: str
    planner_sha256: str
    snapshot_sha256: str
    selected_candidate: JointCandidate
    strictly_improved: bool
    outcomes: tuple[JointBranchOutcome, ...]
    restore_checks: int
    wall_seconds: float
    cpu_seconds: float

    @property
    def decision(self) -> SteeringDecision:
        return self.selected_candidate.decision

    @property
    def simulated_ticks(self) -> int:
        return sum(value.simulated_ticks for value in self.outcomes)

    def manifest(self) -> dict[str, object]:
        return {
            "version": JOINT_PLANNER_VERSION,
            "config_sha256": self.config_sha256,
            "planner_sha256": self.planner_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "selected_ordinal": self.selected_candidate.ordinal,
            "strictly_improved": self.strictly_improved,
            "restore_checks": self.restore_checks,
            "simulated_ticks": self.simulated_ticks,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "outcomes": [value.manifest() for value in self.outcomes],
        }

    def identity_manifest(self) -> dict[str, object]:
        value = self.manifest()
        value.pop("wall_seconds")
        value.pop("cpu_seconds")
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())


def _step_duration(action: Action) -> int:
    return (
        int(action.wait_ticks)
        if ActionKind.parse(action.kind) is ActionKind.WAIT
        else 1
    )


def _pair_distance(
    observation: Mapping[str, Any], source_id: int, destination_id: int
) -> float | None:
    values = _by_id(observation)
    if source_id not in values or destination_id not in values:
        return None
    sx, sy, ssize = _position(values[source_id])
    dx, dy, dsize = _position(values[destination_id])
    return math.hypot(dx - sx, dy - sy) / max((ssize + dsize) / 2.0, 1e-9)


class JointPairGeometrySearch:
    """Transactional multi-horizon search over a bounded joint action set."""

    def __init__(
        self,
        continuation_factory: Callable[[], object],
        *,
        config: JointPlannerConfig | None = None,
        action_spec: ActionSpec | None = None,
        continuation_identity_sha256: str | None = None,
    ) -> None:
        if not callable(continuation_factory):
            raise TypeError("joint continuation factory must be callable")
        self.continuation_factory = continuation_factory
        self.config = JointPlannerConfig() if config is None else config
        self.action_spec = ActionSpec() if action_spec is None else action_spec
        inferred_identity = getattr(
            continuation_factory, "artifact_sha256", None
        )
        identity = (
            continuation_identity_sha256
            if continuation_identity_sha256 is not None
            else inferred_identity
        )
        if identity is not None and (
            not isinstance(identity, str)
            or len(identity) != 64
            or any(value not in "0123456789abcdef" for value in identity)
        ):
            raise ValueError(
                "joint continuation identity must be a lowercase SHA-256"
            )
        self.continuation_identity_sha256 = identity

    def identity_manifest(self) -> dict[str, object]:
        return {
            "version": JOINT_PLANNER_VERSION,
            "config": self.config.manifest(),
            "action_spec_sha256": self.action_spec.sha256,
            "continuation_identity_sha256": (
                self.continuation_identity_sha256
            ),
            "continuation_identity_bound": (
                self.continuation_identity_sha256 is not None
            ),
            "backend": "trusted-portable-clone-only",
            "snapshot_rng": (
                "opaque identical snapshot bytes restored per branch; "
                "never exposed to policy"
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())

    def _candidates(
        self,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
    ) -> tuple[JointCandidate, ...]:
        pairs = shortlist_pairs(
            observation,
            incumbent,
            config=self.config,
            action_spec=self.action_spec,
        )
        output: list[JointCandidate] = []
        ordinal = 0
        for pair_index, pair in enumerate(pairs):
            if pair_index == 0 and not pair.incumbent:
                raise RuntimeError(
                    "joint shortlist did not retain the incumbent first"
                )
            for geometry_index, geometry in enumerate(
                COMPACT_GEOMETRY[: self.config.geometry_cap]
            ):
                decision = (
                    incumbent
                    if pair_index == 0 and geometry_index == 0
                    else _analytic_decision(
                        observation,
                        pair,
                        geometry,
                        self.config,
                        self.action_spec,
                        reason=(
                            f"joint planner {pair.category}/{geometry.name}"
                        ),
                    )
                )
                if decision is None:
                    continue
                output.append(
                    JointCandidate(
                        ordinal,
                        pair_index,
                        geometry_index,
                        pair,
                        geometry,
                        decision,
                    )
                )
                ordinal += 1
        if not output:
            raise ValueError("joint planner produced no executable candidate")
        if (
            output[0].pair_ordinal != 0
            or output[0].geometry_ordinal != 0
            or output[0].decision is not incumbent
        ):
            raise RuntimeError(
                "joint candidate zero is not the exact incoming incumbent"
            )
        return tuple(output)

    def _evaluate(
        self,
        env: Any,
        initial: Mapping[str, Any],
        candidate: JointCandidate,
    ) -> JointBranchOutcome:
        start_tick = _plain_int(initial.get("tick"), "initial tick")
        start_score = _plain_int(initial.get("score"), "initial score")
        start_clears = _plain_int(
            initial.get("qualifying_clear_count"), "initial qualifying clears"
        )
        initial_distance = _pair_distance(
            initial,
            candidate.pair.source_body_id,
            candidate.pair.destination_body_id,
        )
        current = initial
        terminated = bool(initial.get("terminated", False))
        truncated = bool(initial.get("truncated", False))
        events: list[Mapping[str, Any]] = []
        first_projectiles: set[int] = set()
        first_phase = True

        def step(action: Action) -> None:
            nonlocal current, terminated, truncated, first_phase
            current, _reward, terminated, truncated, info = env.step(action)
            if not isinstance(current, Mapping) or not isinstance(info, Mapping):
                raise TypeError("joint portable transition is malformed")
            supplied = tuple(
                value
                for value in info.get("events", ())
                if isinstance(value, Mapping)
            )
            events.extend(supplied)
            if first_phase:
                first_projectiles.update(
                    _plain_int(value.get("a"), "projectile id", -1)
                    for value in supplied
                    if _event_kind(value) == int(EventKind.SHOT_FIRED)
                )

        for action in candidate.decision.primitive_actions(self.action_spec):
            if terminated or truncated:
                break
            step(action)
        first_phase = False
        elapsed = _plain_int(current.get("tick"), "current tick") - start_tick
        if not (terminated or truncated) and elapsed < self.config.cooldown_ticks:
            step(Action.wait(self.config.cooldown_ticks - elapsed))

        policy = self.continuation_factory()
        if self.continuation_identity_sha256 is not None:
            actual_identity = getattr(policy, "artifact_sha256", None)
            if actual_identity != self.continuation_identity_sha256:
                raise RuntimeError(
                    "joint continuation policy identity changed"
                )
        reset = getattr(policy, "reset", None)
        if not callable(reset):
            raise TypeError("joint continuation policy lacks reset")
        reset(0)
        milestones: list[JointMilestoneOutcome] = []
        for horizon in self.config.horizons:
            while not (terminated or truncated):
                elapsed = _plain_int(current.get("tick"), "current tick") - start_tick
                if elapsed >= horizon:
                    break
                predict = getattr(policy, "predict", None)
                decision = (
                    predict(current)
                    if callable(predict)
                    else getattr(policy, "act")(current)
                )
                actions = (
                    decision.primitive_actions(self.action_spec)
                    if isinstance(decision, SteeringDecision)
                    else ((decision,) if isinstance(decision, Action) else decision)
                )
                if not isinstance(actions, tuple):
                    actions = tuple(actions)
                for action in actions:
                    if terminated or truncated:
                        break
                    remaining = horizon - (
                        _plain_int(current.get("tick"), "current tick")
                        - start_tick
                    )
                    if remaining <= 0:
                        break
                    duration = _step_duration(action)
                    if duration > remaining:
                        if ActionKind.parse(action.kind) is not ActionKind.WAIT:
                            raise RuntimeError(
                                "joint shot crossed a milestone boundary"
                            )
                        action = Action.wait(remaining)
                    step(action)
            current_tick = _plain_int(current.get("tick"), "current tick")
            counts = Counter(
                kind
                for value in events
                if (kind := _event_kind(value)) is not None
            )
            source_hits = {
                _plain_int(value.get("a"), "hit projectile", -1)
                for value in events
                if _event_kind(value) == int(EventKind.PROJECTILE_HIT)
                and _plain_int(value.get("a"), "hit projectile", -1)
                in first_projectiles
                and _plain_int(value.get("b"), "hit target", -1)
                == candidate.pair.source_body_id
            }
            intended_destination_ids = {
                _plain_int(body.get("id"), "chain member id")
                for body in _bodies(initial)
                if (
                    candidate.pair.destination_chain_id
                    and _plain_int(body.get("chain_id"), "chain id")
                    == candidate.pair.destination_chain_id
                )
            } or {candidate.pair.destination_body_id}
            joined = any(
                _event_kind(value) == int(EventKind.CHAIN_JOINED)
                and candidate.pair.source_body_id
                in {
                    _plain_int(value.get("a"), "join a", -1),
                    _plain_int(value.get("b"), "join b", -1),
                }
                and bool(
                    intended_destination_ids
                    & {
                        _plain_int(value.get("a"), "join a", -1),
                        _plain_int(value.get("b"), "join b", -1),
                    }
                )
                for value in events
            )
            final_distance = _pair_distance(
                current,
                candidate.pair.source_body_id,
                candidate.pair.destination_body_id,
            )
            milestones.append(
                JointMilestoneOutcome(
                    horizon,
                    not (
                        terminated
                        or truncated
                        or bool(current.get("terminated", False))
                        or bool(current.get("truncated", False))
                    ),
                    current_tick - start_tick,
                    _plain_int(current.get("score"), "final score")
                    - start_score,
                    _plain_int(current.get("gauge"), "final gauge"),
                    _plain_int(current.get("level"), "final level"),
                    _plain_int(
                        current.get("qualifying_clear_count"),
                        "final qualifying clears",
                    )
                    - start_clears,
                    counts[int(EventKind.CLEARED)],
                    counts[int(EventKind.ROTTEN)],
                    sum(
                        max(0, _plain_int(value.get("value"), "gauge delta"))
                        for value in events
                        if _event_kind(value) == int(EventKind.GAUGE_CHANGED)
                    ),
                    counts[int(EventKind.INVALID_ACTION)],
                    len(source_hits),
                    joined,
                    (
                        0.0
                        if initial_distance is None or final_distance is None
                        else initial_distance - final_distance
                    ),
                )
            )
        return JointBranchOutcome(
            candidate,
            tuple(milestones),
            _plain_int(current.get("tick"), "current tick") - start_tick,
        )

    def search(
        self,
        env: Any,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
    ) -> JointSearchResult:
        if getattr(env, "physics_backend", None) != "portable":
            raise ValueError("joint search requires the portable backend")
        if not incumbent.is_shot:
            raise ValueError("joint search requires an incumbent shot")
        candidates = self._candidates(observation, incumbent)
        snapshot = env.clone_state()
        snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
        expected_signature = _public_signature(observation)
        state_hash = getattr(env, "state_hash", None)
        expected_state_hash = state_hash() if callable(state_hash) else None
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        outcomes: list[JointBranchOutcome] = []
        restore_checks = 0
        try:
            for candidate in candidates:
                restored = env.restore_state(snapshot)
                restore_checks += 1
                if (
                    _public_signature(restored) != expected_signature
                    or env.clone_state() != snapshot
                    or (
                        expected_state_hash is not None
                        and state_hash() != expected_state_hash
                    )
                ):
                    raise RuntimeError(
                        "joint branch did not receive the identical portable state"
                    )
                outcomes.append(self._evaluate(env, restored, candidate))
        finally:
            restored = env.restore_state(snapshot)
            restore_checks += 1
            if (
                _public_signature(restored) != expected_signature
                or env.clone_state() != snapshot
                or (
                    expected_state_hash is not None
                    and state_hash() != expected_state_hash
                )
            ):
                raise RuntimeError("joint search failed to restore its source state")
        incumbent_outcome = outcomes[0]
        eligible = [
            value
            for value in outcomes
            if value.selectable_against(incumbent_outcome)
        ]
        winner = max(
            eligible,
            key=lambda value: (value.objective, -value.candidate.ordinal),
        )
        improved = winner.objective > incumbent_outcome.objective
        if not improved:
            winner = incumbent_outcome
        return JointSearchResult(
            self.config.sha256,
            self.sha256,
            snapshot_sha256,
            winner.candidate,
            improved,
            tuple(outcomes),
            restore_checks,
            time.perf_counter() - wall_started,
            time.process_time() - cpu_started,
        )


def _commit_base_decision(
    base_policy: object,
    observation: Mapping[str, Any],
    incumbent: SteeringDecision,
    selected: SteeringDecision,
) -> bool:
    """Rebind frozen-v5 progress without changing its chosen cooldown."""

    if selected is incumbent:
        return True
    hook = getattr(base_policy, "commit_external_decision", None)
    if callable(hook):
        hook(observation, selected)
        return True
    pair_changed = (
        selected.source_body_id != incumbent.source_body_id
        or selected.destination_body_id != incumbent.destination_body_id
    )
    if pair_changed:
        tracker = getattr(base_policy, "_progress", None)
        begin = getattr(tracker, "begin", None)
        pending = getattr(tracker, "pending_pair", None)
        if not callable(begin):
            return False
        pending_pair = (
            None
            if pending is None
            else (pending.source_id, pending.destination_id)
        )
        incumbent_pair = (
            incumbent.source_body_id,
            incumbent.destination_body_id,
        )
        selected_pair = (
            selected.source_body_id,
            selected.destination_body_id,
        )
        if pending_pair == selected_pair:
            pass
        elif pending_pair == incumbent_pair and hasattr(tracker, "_attempt"):
            previous_attempt = tracker._attempt
            tracker._attempt = None
            try:
                begin(
                    observation,
                    int(selected.source_body_id),
                    int(selected.destination_body_id),
                )
            except (RuntimeError, TypeError, ValueError):
                tracker._attempt = previous_attempt
                return False
        else:
            return False
    if hasattr(base_policy, "_last_decision"):
        base_policy._last_decision = selected
    return True


class JointTeacherPolicy:
    """Preserve frozen-v5 cadence while querying bounded joint replacements."""

    def __init__(
        self,
        env: Any,
        base_policy: object,
        searcher: JointPairGeometrySearch,
        *,
        query_stride_shots: int = 2,
        maximum_queries: int = 64,
    ) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (query_stride_shots, maximum_queries)
        ):
            raise ValueError("joint teacher query budgets must be positive")
        self.env = env
        self.base_policy = base_policy
        self.searcher = searcher
        self.query_stride_shots = query_stride_shots
        self.maximum_queries = maximum_queries
        self._results: list[JointSearchResult] = []
        self._attempts: list[JointSearchResult] = []
        self._counts: Counter[str] = Counter()

    def reset(self, seed: int = 0) -> None:
        getattr(self.base_policy, "reset")(seed)
        self._results.clear()
        self._attempts.clear()
        self._counts.clear()

    @property
    def results(self) -> tuple[JointSearchResult, ...]:
        return tuple(self._results)

    @property
    def attempts(self) -> tuple[JointSearchResult, ...]:
        return tuple(self._attempts)

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = getattr(self.base_policy, "predict")(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("joint teacher base policy returned a non-decision")
        if not incumbent.is_shot:
            return incumbent
        self._counts["seen_shots"] += 1
        if (
            (self._counts["seen_shots"] - 1) % self.query_stride_shots
            or self._counts["search_attempts"] >= self.maximum_queries
        ):
            self._counts["budget_fallbacks"] += 1
            return incumbent
        self._counts["search_attempts"] += 1
        try:
            result = self.searcher.search(self.env, observation, incumbent)
        except ValueError:
            self._counts["unsupported_fallbacks"] += 1
            return incumbent
        self._attempts.append(result)
        self._counts["branch_outcomes"] += len(result.outcomes)
        self._counts["restore_checks"] += result.restore_checks
        self._counts["simulated_branch_ticks"] += result.simulated_ticks
        if not _commit_base_decision(
            self.base_policy, observation, incumbent, result.decision
        ):
            self._counts["progress_rebind_fallbacks"] += 1
            return incumbent
        self._results.append(result)
        self._counts["search_queries"] += 1
        self._counts["strict_improvements"] += int(result.strictly_improved)
        self._counts[
            f"selected_pair/{result.selected_candidate.pair.category}"
        ] += 1
        self._counts[
            f"selected_geometry/{result.selected_candidate.geometry.name}"
        ] += 1
        self._counts["pair_corrections"] += int(
            result.selected_candidate.pair_ordinal != 0
        )
        self._counts["geometry_corrections"] += int(
            result.selected_candidate.geometry_ordinal != 0
        )
        return result.decision

    def statistics(self) -> dict[str, int | float]:
        return {
            **dict(sorted(self._counts.items())),
            "search_wall_seconds": sum(value.wall_seconds for value in self._attempts),
            "search_cpu_seconds": sum(value.cpu_seconds for value in self._attempts),
        }


class JointPlannerStudentPolicy:
    """Hierarchical residual pair/geometry heads over frozen-v5 cadence."""

    def __init__(
        self,
        base_policy: object,
        model: GoalConditionedSteeringModel,
        *,
        config: JointPlannerConfig | None = None,
        action_spec: ActionSpec | None = None,
        pair_confidence: float = 0.0,
        pair_margin: float = 0.05,
        geometry_confidence: float = 0.0,
        geometry_margin: float = 0.05,
    ) -> None:
        self.base_policy = base_policy
        self.model = model
        self.model.eval()
        self.config = JointPlannerConfig() if config is None else config
        self.action_spec = ActionSpec() if action_spec is None else action_spec
        self.encoder = TeacherStateEncoder()
        if self.encoder.schema.sha256 != model.schema.sha256:
            raise ValueError("joint student model/schema identity mismatch")
        for value in (
            pair_confidence,
            pair_margin,
            geometry_confidence,
            geometry_margin,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError("joint student confidence settings are invalid")
        self.pair_confidence = float(pair_confidence)
        self.pair_margin = float(pair_margin)
        self.geometry_confidence = float(geometry_confidence)
        self.geometry_margin = float(geometry_margin)
        self._counts: Counter[str] = Counter()

    def reset(self, seed: int = 0) -> None:
        getattr(self.base_policy, "reset")(seed)
        self._counts.clear()

    @staticmethod
    def _indices(
        identifiers: Sequence[int | None], pair: PairOption
    ) -> tuple[int, int] | None:
        try:
            source = identifiers.index(pair.source_body_id)
            destination = identifiers.index(pair.destination_body_id)
        except ValueError:
            return None
        return source, destination

    @staticmethod
    def _action_indices(
        spec: PointerActionSpec,
        action_spec: ActionSpec,
        decision: SteeringDecision,
        observation: Mapping[str, Any],
    ) -> tuple[int, int]:
        if decision.source_body_id is None:
            raise ValueError("joint geometry decision lacks a source")
        values = _by_id(observation)
        source = values[decision.source_body_id]
        sx, sy, size = _position(source)
        width = _plain_float(source.get("width", size), "source width")
        height = _plain_float(source.get("height", size), "source height")
        if width <= 0.0 or height <= 0.0:
            raise ValueError("joint source extents must be positive")
        target = (
            (
                float(decision.action.x_norm) * action_spec.client_width - sx
            )
            / (width / 2.0),
            (
                float(decision.action.y_norm) * action_spec.client_height - sy
            )
            / (height / 2.0),
        )
        template_index = min(
            range(spec.template_count),
            key=lambda index: (
                (spec.templates[index][0] - target[0]) ** 2
                + (spec.templates[index][1] - target[1]) ** 2,
                index,
            ),
        )
        kind = SemanticActionKind(decision.action.kind)
        if kind is SemanticActionKind.FIRE_WEAK:
            kind_index = 0
        elif kind is SemanticActionKind.FIRE_STRONG:
            kind_index = 1
        else:
            raise ValueError("joint geometry candidate is not a shot")
        return kind_index, template_index

    @torch.no_grad()
    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = getattr(self.base_policy, "predict")(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("joint student base policy returned a non-decision")
        if not incumbent.is_shot:
            return incumbent
        self._counts["seen_shots"] += 1
        try:
            pairs = shortlist_pairs(
                observation,
                incumbent,
                config=self.config,
                action_spec=self.action_spec,
            )
        except ValueError:
            self._counts["shortlist_fallbacks"] += 1
            return incumbent
        encoded = self.encoder.encode([observation])
        active = np.flatnonzero(encoded.body_mask[0])
        width = int(active[-1]) + 1 if active.size else 1
        device = next(self.model.parameters()).device
        output = self.model(
            torch.from_numpy(encoded.global_features).to(device),
            torch.from_numpy(encoded.body_features[:, :width]).to(device),
            torch.from_numpy(encoded.body_mask[:, :width]).to(device),
        )
        identifiers = list(encoded_body_ids(encoded, observation))[:width]
        bound = [self._indices(identifiers, value) for value in pairs]
        valid = [
            index
            for index, pair_indices in enumerate(bound)
            if pair_indices is not None
            and bool(output.legal_pair_mask[(0, *pair_indices)])
        ]
        if not valid or 0 not in valid:
            self._counts["binding_fallbacks"] += 1
            return incumbent
        pair_scores = torch.stack(
            [
                output.pair_logits[(0, *bound[index])]  # type: ignore[index]
                for index in valid
            ]
        )
        pair_probabilities = torch.softmax(pair_scores, dim=0)
        best_position = int(pair_probabilities.argmax())
        best_pair_index = valid[best_position]
        incumbent_position = valid.index(0)
        pair_probability = float(pair_probabilities[best_position])
        pair_delta = pair_probability - float(
            pair_probabilities[incumbent_position]
        )
        pair = pairs[0]
        pair_corrected = False
        if (
            best_pair_index != 0
            and pair_probability >= self.pair_confidence
            and pair_delta >= self.pair_margin
        ):
            pair = pairs[best_pair_index]
            pair_corrected = True
        elif best_pair_index != 0:
            self._counts["pair_confidence_fallbacks"] += 1
        selected_indices = self._indices(identifiers, pair)
        if selected_indices is None:
            self._counts["binding_fallbacks"] += 1
            return incumbent
        source_index, destination_index = selected_indices
        kind_log = torch.log_softmax(
            output.kind_logits[0, source_index, destination_index], dim=-1
        )
        template_log = torch.log_softmax(
            output.template_logits[0, source_index, destination_index], dim=-1
        )
        geometry_scores = []
        geometry_decisions: list[SteeringDecision | None] = []
        for geometry_index, geometry in enumerate(
            COMPACT_GEOMETRY[: self.config.geometry_cap]
        ):
            candidate_decision = (
                incumbent
                if pair is pairs[0] and geometry_index == 0
                else _analytic_decision(
                    observation,
                    pair,
                    geometry,
                    self.config,
                    self.action_spec,
                    reason=(
                        "distilled hierarchical joint pair/geometry residual"
                    ),
                )
            )
            geometry_decisions.append(candidate_decision)
            if candidate_decision is None:
                geometry_scores.append(
                    torch.full(
                        (),
                        -torch.inf,
                        dtype=kind_log.dtype,
                        device=kind_log.device,
                    )
                )
                continue
            kind_index, template_index = self._action_indices(
                self.model.pointer_spec,
                self.action_spec,
                candidate_decision,
                observation,
            )
            geometry_scores.append(
                kind_log[kind_index] + template_log[template_index]
            )
        probabilities = torch.softmax(torch.stack(geometry_scores), dim=0)
        geometry_index = int(probabilities.argmax())
        geometry_probability = float(probabilities[geometry_index])
        geometry_delta = geometry_probability - float(probabilities[0])
        if (
            geometry_index != 0
            and (
                geometry_probability < self.geometry_confidence
                or geometry_delta < self.geometry_margin
            )
        ):
            geometry_index = 0
            self._counts["geometry_confidence_fallbacks"] += 1
        decision = geometry_decisions[geometry_index]
        if decision is None:
            self._counts["reachability_fallbacks"] += 1
            return incumbent
        if not _commit_base_decision(
            self.base_policy, observation, incumbent, decision
        ):
            self._counts["progress_rebind_fallbacks"] += 1
            return incumbent
        self._counts["pair_corrections"] += int(pair_corrected)
        self._counts["geometry_corrections"] += int(geometry_index != 0)
        self._counts[
            f"selected_pair/{pair.category}"
        ] += 1
        self._counts[
            f"selected_geometry/{COMPACT_GEOMETRY[geometry_index].name}"
        ] += 1
        return decision

    def statistics(self) -> dict[str, int]:
        return dict(sorted(self._counts.items()))


__all__ = [
    "COMPACT_GEOMETRY",
    "JOINT_PLANNER_VERSION",
    "GeometryOption",
    "JointBranchOutcome",
    "JointCandidate",
    "JointMilestoneOutcome",
    "JointPairGeometrySearch",
    "JointPlannerConfig",
    "JointPlannerStudentPolicy",
    "JointSearchResult",
    "JointTeacherPolicy",
    "PairOption",
    "shortlist_pairs",
]
