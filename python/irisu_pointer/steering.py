"""Closed-loop, destination-conditioned steering from public observations.

The older pointer vocabulary chooses a body and an absolute launch row.  That
cannot express the basic IriSu control primitive: hit a moving piece from the
side opposite its destination and from just below its current position.  This
module keeps that geometry continuous and lowers it directly to the existing
deployment-v1 press/release contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any

from irisu_env import Action
from irisu_rl.actions import ActionSpec, SemanticAction, SemanticActionKind

from .steering_progress import DirectedPairProgressTracker


class SteeringIntent(str, Enum):
    """Auditable high-level reason for a steering decision."""

    WAIT = "wait"
    STEER_MATCH = "steer_match"
    EXTEND_ANCHOR = "extend_anchor"
    MATCH_ROTTEN = "match_rotten"
    EJECT_HAZARD = "eject_hazard"
    ACTIVATE_BONUS = "activate_bonus"
    PRESERVE_GROUP = "preserve_group"


@dataclass(frozen=True, slots=True)
class SteeringExpertConfig:
    """Conservative defaults for closed-loop strong-shot steering."""

    observe_ticks: int = 16
    resolution_wait_ticks: int = 4
    abandon_ticks: int = 32
    impact_side_sizes: float = 0.50
    impact_below_sizes: float = 0.75
    source_velocity_lead_ticks: float = 1.0
    minimum_pair_closure_sizes: float = 0.05
    ticks_per_second: float = 50.0
    hazard_remaining_ticks: int = 48
    unmatched_edge_fraction: float = 0.12
    unmatched_floor_fraction: float = 0.55
    projectile_x_sizes: float = 0.80
    enable_bonus: bool = False
    enable_rotten_matching: bool = True
    enable_hazard_ejection: bool = False
    enable_edge_ejection: bool = False

    def __post_init__(self) -> None:
        integer_fields = (
            self.observe_ticks,
            self.resolution_wait_ticks,
            self.abandon_ticks,
            self.hazard_remaining_ticks,
        )
        if any(type(value) is not int or value < 1 for value in integer_fields):
            raise ValueError("steering timing and count fields must be positive integers")
        finite_positive = (
            self.impact_side_sizes,
            self.impact_below_sizes,
            self.minimum_pair_closure_sizes,
            self.ticks_per_second,
            self.projectile_x_sizes,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in finite_positive
        ):
            raise ValueError("steering geometry scales must be finite and positive")
        if (
            isinstance(self.source_velocity_lead_ticks, bool)
            or not isinstance(self.source_velocity_lead_ticks, Real)
            or not math.isfinite(float(self.source_velocity_lead_ticks))
            or float(self.source_velocity_lead_ticks) < 0.0
        ):
            raise ValueError("source velocity lead must be finite and nonnegative")
        for value in (self.unmatched_edge_fraction, self.unmatched_floor_fraction):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError("steering field fractions must be in [0, 1]")
        switches = (
            self.enable_bonus,
            self.enable_rotten_matching,
            self.enable_hazard_ejection,
            self.enable_edge_ejection,
        )
        if any(type(value) is not bool for value in switches):
            raise TypeError("steering feature switches must be booleans")


@dataclass(frozen=True, slots=True)
class SteeringDecision:
    """One semantic action with its public source/destination binding."""

    action: SemanticAction
    intent: SteeringIntent
    source_body_id: int | None = None
    destination_body_id: int | None = None
    destination_chain_id: int | None = None
    impact_x_sizes: float = 0.0
    impact_y_sizes: float = 0.0
    correction_index: int = 0
    reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.action, SemanticAction):
            raise TypeError("steering action must be semantic")
        if not isinstance(self.intent, SteeringIntent):
            raise TypeError("steering intent must be a SteeringIntent")
        for name in ("source_body_id", "destination_body_id", "destination_chain_id"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a nonnegative integer or None")
        for name in ("impact_x_sizes", "impact_y_sizes"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if (
            isinstance(self.correction_index, bool)
            or not isinstance(self.correction_index, int)
            or self.correction_index < 0
        ):
            raise ValueError("correction index must be nonnegative")
        if not isinstance(self.reason, str):
            raise TypeError("steering reason must be text")
        kind = SemanticActionKind(self.action.kind)
        if kind is not SemanticActionKind.WAIT and self.source_body_id is None:
            raise ValueError("a steering shot requires a public source body ID")

    @property
    def is_shot(self) -> bool:
        return SemanticActionKind(self.action.kind) is not SemanticActionKind.WAIT

    def primitive_actions(
        self, action_spec: ActionSpec | None = None
    ) -> tuple[Action, ...]:
        """Lower to one wait or an explicitly released shot macro."""

        resolved = ActionSpec() if action_spec is None else action_spec
        action = resolved.validate(self.action)
        if SemanticActionKind(action.kind) is SemanticActionKind.WAIT:
            return (resolved.press(action),)
        return resolved.press(action), resolved.release()


@dataclass(frozen=True, slots=True)
class _Body:
    identifier: int
    kind: str
    lifecycle: str
    color: int
    x: float
    y: float
    vx: float
    vy: float
    size: float
    chain_id: int
    projectile_hits: int
    remaining_lifetime: int
    rot_timer: int


@dataclass(slots=True)
class _TrackedGoal:
    source_id: int
    destination_id: int
    destination_chain_id: int
    intent: SteeringIntent
    corrections: int = 0


_SOURCE_LIFECYCLES = frozenset(
    {"scripted_falling", "dynamic_fresh", "falling", "fresh", "rotten"}
)
_DESTINATION_LIFECYCLES = _SOURCE_LIFECYCLES | frozenset({"confirmed"})


def _plain_int(value: Any, name: str, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    item = getattr(value, "item", None)
    if callable(item):
        return _plain_int(item(), name, default=default)
    raise TypeError(f"public body {name} must be an integer")


def _plain_float(value: Any, name: str, *, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        item = getattr(value, "item", None)
        if not callable(item):
            raise TypeError(f"public body {name} must be numeric")
        result = float(item())
    if not math.isfinite(result):
        raise ValueError(f"public body {name} must be finite")
    return result


def _body(value: Mapping[str, Any]) -> _Body:
    identifier = _plain_int(value.get("id"), "id", default=-1)
    if identifier < 0:
        raise ValueError("public body ID must be nonnegative")
    size = _plain_float(value.get("size"), "size")
    if size <= 0.0:
        raise ValueError("public body size must be positive")
    lifecycle = str(value.get("lifecycle", ""))
    velocity_factor = (
        50.0 if lifecycle in {"scripted_falling", "falling"} else 10.0
    )
    vx = (
        _plain_float(value.get("vx_display_per_second"), "vx")
        if value.get("vx_display_per_second") is not None
        else _plain_float(value.get("vx"), "vx") * velocity_factor
    )
    vy = (
        _plain_float(value.get("vy_display_per_second"), "vy")
        if value.get("vy_display_per_second") is not None
        else _plain_float(value.get("vy"), "vy") * velocity_factor
    )
    return _Body(
        identifier=identifier,
        kind=str(value.get("kind", "")),
        lifecycle=lifecycle,
        color=_plain_int(value.get("color"), "color", default=-1),
        x=_plain_float(value.get("effect_x", value.get("x")), "x"),
        y=_plain_float(value.get("effect_y", value.get("y")), "y"),
        vx=vx,
        vy=vy,
        size=size,
        chain_id=_plain_int(value.get("chain_id"), "chain_id"),
        projectile_hits=_plain_int(
            value.get("projectile_hits"), "projectile_hits"
        ),
        remaining_lifetime=_plain_int(
            value.get("remaining_lifetime"), "remaining_lifetime"
        ),
        rot_timer=_plain_int(value.get("rot_timer"), "rot_timer"),
    )


def _public_bodies(observation: Mapping[str, Any]) -> tuple[_Body, ...]:
    values = observation.get("bodies", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("public observation bodies must be a sequence")
    bodies = tuple(_body(value) for value in values if isinstance(value, Mapping))
    ids = [body.identifier for body in bodies]
    if len(ids) != len(set(ids)):
        raise ValueError("public observation body IDs must be unique")
    return bodies


def _distance(first: _Body, second: _Body) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


class ClosedLoopSteeringExpert:
    """Track and repeatedly steer one public source toward a public destination.

    Grouped pieces are destination anchors only.  They are never direct-shot
    targets.  The controller also waits while its previous projectile is in
    flight and abandons a source after a bounded number of corrections.
    """

    def __init__(
        self,
        *,
        config: SteeringExpertConfig | None = None,
        action_spec: ActionSpec | None = None,
    ) -> None:
        self.config = SteeringExpertConfig() if config is None else config
        self.action_spec = ActionSpec() if action_spec is None else action_spec
        self._goal: _TrackedGoal | None = None
        self._cooldown_until = 0
        self._suppressed_until: dict[int, int] = {}
        self._last_tick: int | None = None
        self._last_decision: SteeringDecision | None = None
        self._progress = DirectedPairProgressTracker(
            minimum_closure_sizes=self.config.minimum_pair_closure_sizes
        )

    def reset(self, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("steering seed must fit in uint32")
        self._goal = None
        self._cooldown_until = 0
        self._suppressed_until.clear()
        self._last_tick = None
        self._last_decision = None
        self._progress.reset()

    @property
    def tracked_source_body_id(self) -> int | None:
        return None if self._goal is None else self._goal.source_id

    @property
    def tracked_destination_body_id(self) -> int | None:
        return None if self._goal is None else self._goal.destination_id

    def _wait(
        self,
        desired_ticks: int,
        intent: SteeringIntent = SteeringIntent.WAIT,
        *,
        source: _Body | None = None,
        destination: _Body | None = None,
        reason: str,
    ) -> SteeringDecision:
        choices = tuple(
            value for value in self.action_spec.wait_choices if value <= desired_ticks
        )
        ticks = max(choices) if choices else min(self.action_spec.wait_choices)
        return SteeringDecision(
            action=self.action_spec.validate(SemanticAction.wait(ticks)),
            intent=intent,
            source_body_id=None if source is None else source.identifier,
            destination_body_id=(
                None if destination is None else destination.identifier
            ),
            destination_chain_id=(
                None if destination is None else destination.chain_id
            ),
            correction_index=0 if self._goal is None else self._goal.corrections,
            reason=reason,
        )

    def _impact_point(
        self, source: _Body, destination: _Body
    ) -> tuple[float, float, float]:
        direction = 1.0 if destination.x > source.x else -1.0
        if math.isclose(destination.x, source.x, abs_tol=1e-9):
            direction = (
                1.0
                if source.x <= self.action_spec.client_width / 2.0
                else -1.0
            )
        impact_x = -direction * self.config.impact_side_sizes
        lead_seconds = (
            self.config.source_velocity_lead_ticks
            / self.config.ticks_per_second
        )
        return (
            impact_x,
            source.x + source.vx * lead_seconds + impact_x * source.size,
            source.y + self.config.impact_below_sizes * source.size,
        )

    def _pair_is_reachable(self, source: _Body, destination: _Body) -> bool:
        _, x, y = self._impact_point(source, destination)
        return (
            0.0 <= x <= self.action_spec.client_width
            and 0.0 <= y <= self.action_spec.client_height
        )

    def _shot(
        self,
        observation: Mapping[str, Any],
        source: _Body,
        destination: _Body,
        intent: SteeringIntent,
        *,
        tick: int,
        track_progress: bool = True,
    ) -> SteeringDecision:
        impact_x, x, y = self._impact_point(source, destination)
        x_norm = min(1.0, max(0.0, x / self.action_spec.client_width))
        y_norm = min(1.0, max(0.0, y / self.action_spec.client_height))
        action = self.action_spec.validate(SemanticAction.strong(x_norm, y_norm))
        if self._goal is None or self._goal.source_id != source.identifier:
            self._goal = _TrackedGoal(
                source.identifier,
                destination.identifier,
                destination.chain_id,
                intent,
            )
        else:
            self._goal.destination_id = destination.identifier
            self._goal.destination_chain_id = destination.chain_id
            self._goal.intent = intent
        self._goal.corrections += 1
        self._cooldown_until = tick + self.config.observe_ticks
        retargeted = bool(self._progress.stalled_pairs)
        if track_progress:
            self._progress.begin(
                observation, source.identifier, destination.identifier
            )
        return SteeringDecision(
            action=action,
            intent=intent,
            source_body_id=source.identifier,
            destination_body_id=destination.identifier,
            destination_chain_id=destination.chain_id,
            impact_x_sizes=impact_x,
            impact_y_sizes=self.config.impact_below_sizes,
            correction_index=self._goal.corrections,
            reason=(
                "progress tracker selected the next public directed pair"
                if retargeted
                else "target-relative correction toward destination"
            ),
        )

    def _eject(
        self,
        observation: Mapping[str, Any],
        source: _Body,
        center_x: float,
        *,
        tick: int,
    ) -> SteeringDecision:
        # Hit from the inner side, transferring momentum toward the free wall.
        destination_x = 0.0 if source.x < center_x else center_x * 2.0
        proxy = _Body(
            identifier=source.identifier + 1,
            kind="boundary",
            lifecycle="",
            color=source.color,
            x=destination_x,
            y=source.y,
            vx=0.0,
            vy=0.0,
            size=source.size,
            chain_id=0,
            projectile_hits=0,
            remaining_lifetime=0,
            rot_timer=0,
        )
        decision = self._shot(
            observation,
            source,
            proxy,
            SteeringIntent.EJECT_HAZARD,
            tick=tick,
            track_progress=False,
        )
        return SteeringDecision(
            action=decision.action,
            intent=SteeringIntent.EJECT_HAZARD,
            source_body_id=source.identifier,
            impact_x_sizes=decision.impact_x_sizes,
            impact_y_sizes=decision.impact_y_sizes,
            correction_index=decision.correction_index,
            reason="inner-side hit ejects unmatched hazard toward nearest wall",
        )

    def _projectile_is_tracking(self, source: _Body, bodies: tuple[_Body, ...]) -> bool:
        radius = max(8.0, source.size * self.config.projectile_x_sizes)
        return any(
            body.kind == "projectile"
            and abs(body.x - source.x) <= radius
            and body.y >= source.y - source.size
            for body in bodies
        )

    def _destination(
        self, goal: _TrackedGoal, source: _Body, pieces: tuple[_Body, ...]
    ) -> _Body | None:
        exact = next(
            (
                body
                for body in pieces
                if body.identifier == goal.destination_id
                and body.color == source.color
                and body.lifecycle in _DESTINATION_LIFECYCLES
            ),
            None,
        )
        if exact is not None:
            return exact
        if goal.destination_chain_id:
            anchors = [
                body
                for body in pieces
                if body.chain_id == goal.destination_chain_id
                and body.color == source.color
                and body.lifecycle != "rotten"
            ]
            if anchors:
                return min(
                    anchors,
                    key=lambda body: (
                        _distance(source, body),
                        body.x,
                        body.y,
                        body.size,
                    ),
                )
        return None

    def _pair(
        self,
        observation: Mapping[str, Any],
        pieces: tuple[_Body, ...],
        tick: int,
    ) -> tuple[_Body, _Body, SteeringIntent] | None:
        group_sizes: dict[int, int] = {}
        for body in pieces:
            if body.chain_id:
                group_sizes[body.chain_id] = group_sizes.get(body.chain_id, 0) + 1
        candidates: list[tuple[tuple[float, ...], _Body, _Body, SteeringIntent]] = []
        for source in pieces:
            if (
                source.chain_id != 0
                or source.lifecycle not in _SOURCE_LIFECYCLES
                or source.lifecycle == "rotten"
                or self._suppressed_until.get(source.identifier, 0) > tick
            ):
                continue
            for destination in pieces:
                if (
                    destination.identifier == source.identifier
                    or destination.color != source.color
                    or destination.lifecycle not in _DESTINATION_LIFECYCLES
                    or (
                        destination.lifecycle == "rotten"
                        and not self.config.enable_rotten_matching
                    )
                    or not self._pair_is_reachable(source, destination)
                    or self._progress.is_stalled(
                        observation,
                        source.identifier,
                        destination.identifier,
                    )
                ):
                    continue
                anchor_size = group_sizes.get(destination.chain_id, 0)
                destination_rotten = destination.lifecycle == "rotten"
                source_active_rot = source.rot_timer > 0
                intent = (
                    SteeringIntent.MATCH_ROTTEN
                    if destination_rotten or source_active_rot
                    else (
                        SteeringIntent.EXTEND_ANCHOR
                        if anchor_size
                        else SteeringIntent.STEER_MATCH
                    )
                )
                # Native contact order makes fresh -> rotten group and burst
                # immediately. Never direct-hit the rotten piece itself.
                priority = (
                    0.0 if destination_rotten else 1.0,
                    0.0 if source_active_rot else 1.0,
                    0.0 if anchor_size else 1.0,
                    -float(anchor_size),
                    _distance(source, destination),
                    -source.y,
                    source.x,
                    source.size,
                    destination.x,
                    destination.y,
                    destination.size,
                )
                candidates.append((priority, source, destination, intent))
        if not candidates:
            return None
        _, source, destination, intent = min(candidates, key=lambda value: value[0])
        return source, destination, intent

    def _bonus_goal(
        self, bodies: tuple[_Body, ...], pieces: tuple[_Body, ...]
    ) -> tuple[_Body, _Body] | None:
        if not self.config.enable_bonus:
            return None
        bonuses = [
            body
            for body in bodies
            if body.kind == "bonus"
            and body.lifecycle in {"scripted_falling", "falling"}
        ]
        targets = [
            body
            for body in pieces
            if body.chain_id == 0
            and body.lifecycle in _SOURCE_LIFECYCLES
            and body.lifecycle != "rotten"
        ]
        if not bonuses or not targets:
            return None
        return min(
            (
                (
                    _distance(bonus, target),
                    bonus.x,
                    bonus.y,
                    target.x,
                    target.y,
                    bonus,
                    target,
                )
                for bonus in bonuses
                for target in targets
            ),
            key=lambda value: value[:5],
        )[5:]

    def _hazard(
        self,
        pieces: tuple[_Body, ...],
        observation: Mapping[str, Any],
        tick: int,
    ) -> _Body | None:
        if not self.config.enable_hazard_ejection:
            return None
        field = observation.get("field", {})
        if not isinstance(field, Mapping):
            raise TypeError("public field geometry must be a mapping")
        left = _plain_float(field.get("x"), "field.x")
        width = _plain_float(
            field.get("width"), "field.width", default=self.action_spec.client_width
        )
        top = _plain_float(field.get("y"), "field.y")
        height = _plain_float(
            field.get("height"), "field.height", default=self.action_spec.client_height
        )
        if width <= 0.0 or height <= 0.0:
            raise ValueError("public field extents must be positive")
        edge = self.config.unmatched_edge_fraction * width
        floor_y = top + self.config.unmatched_floor_fraction * height
        paired_colors = {
            color
            for color in {body.color for body in pieces}
            if sum(body.color == color for body in pieces) >= 2
        }
        hazards: list[tuple[tuple[float, ...], _Body]] = []
        for body in pieces:
            if (
                body.chain_id != 0
                or body.lifecycle not in _SOURCE_LIFECYCLES
                or self._suppressed_until.get(body.identifier, 0) > tick
            ):
                continue
            rotten = body.lifecycle == "rotten" or body.rot_timer > 0
            expiring = (
                0 < body.remaining_lifetime <= self.config.hazard_remaining_ticks
            )
            outer = min(body.x - left, left + width - body.x) <= edge
            edge_clutter = (
                self.config.enable_edge_ejection
                and body.color not in paired_colors
                and outer
                and body.y >= floor_y
            )
            if not (rotten or expiring or edge_clutter):
                continue
            priority = (
                0.0 if rotten else 1.0,
                0.0 if expiring else 1.0,
                min(body.x - left, left + width - body.x),
                -body.y,
                body.x,
            )
            hazards.append((priority, body))
        return None if not hazards else min(hazards, key=lambda value: value[0])[1]

    def _compute(
        self, observation: Mapping[str, Any], bodies: tuple[_Body, ...], tick: int
    ) -> SteeringDecision:
        pieces = tuple(body for body in bodies if body.kind == "piece")
        by_id = {body.identifier: body for body in pieces}
        expired = [
            identifier
            for identifier, until in self._suppressed_until.items()
            if until <= tick
        ]
        for identifier in expired:
            del self._suppressed_until[identifier]
        self._progress.prune(observation)
        if tick >= self._cooldown_until:
            self._progress.assess(observation)

        if bool(observation.get("terminated", False)) or bool(
            observation.get("truncated", False)
        ):
            self._goal = None
            return self._wait(1, reason="episode is terminal")

        if tick < self._cooldown_until:
            source = (
                None if self._goal is None else by_id.get(self._goal.source_id)
            )
            destination = (
                None
                if self._goal is None or source is None
                else self._destination(self._goal, source, pieces)
            )
            return self._wait(
                self._cooldown_until - tick,
                source=source,
                destination=destination,
                reason="observe the previous correction before acting again",
            )

        if self._goal is not None:
            source = by_id.get(self._goal.source_id)
            if source is None:
                self._goal = None
            elif source.chain_id != 0 or source.lifecycle == "confirmed":
                destination = self._destination(self._goal, source, pieces)
                self._suppressed_until[source.identifier] = tick + self.config.abandon_ticks
                self._goal = None
                return self._wait(
                    self.config.resolution_wait_ticks,
                    SteeringIntent.PRESERVE_GROUP,
                    source=source,
                    destination=destination,
                    reason="tracked source joined or confirmed; do not direct-hit it",
                )
            else:
                # Re-score the whole board at every safe boundary. If the same
                # source/destination remains best, the next shot is naturally
                # a correction; otherwise a stale miss cannot monopolize play.
                self._goal = None

        bonus = self._bonus_goal(bodies, pieces)
        if bonus is not None:
            source, destination = bonus
            return self._shot(
                observation,
                source,
                destination,
                SteeringIntent.ACTIVATE_BONUS,
                tick=tick,
            )

        pair = self._pair(observation, pieces, tick)
        if pair is not None:
            source, destination, intent = pair
            return self._shot(
                observation, source, destination, intent, tick=tick
            )

        hazard = self._hazard(pieces, observation, tick)
        if hazard is not None and not self._projectile_is_tracking(hazard, bodies):
            field = observation.get("field", {})
            assert isinstance(field, Mapping)
            left = _plain_float(field.get("x"), "field.x")
            width = _plain_float(
                field.get("width"),
                "field.width",
                default=self.action_spec.client_width,
            )
            return self._eject(
                observation, hazard, left + width / 2.0, tick=tick
            )

        return self._wait(
            self.config.resolution_wait_ticks,
            reason=(
                "progress tracker is waiting for public pair geometry to change"
                if self._progress.stalled_pairs
                else "no safe destination-conditioned correction is available"
            ),
        )

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        """Return an idempotent decision for the current public observation."""

        if not isinstance(observation, Mapping):
            raise TypeError("steering observation must be a public mapping")
        tick = _plain_int(observation.get("tick"), "tick")
        if tick < 0:
            raise ValueError("public observation tick must be nonnegative")
        if self._last_tick is not None:
            if tick < self._last_tick:
                raise RuntimeError("observation tick moved backwards; reset the expert")
            if tick == self._last_tick:
                assert self._last_decision is not None
                return self._last_decision
        bodies = _public_bodies(observation)
        decision = self._compute(observation, bodies, tick)
        self._last_tick = tick
        self._last_decision = decision
        return decision

    def semantic_act(self, observation: Mapping[str, Any]) -> SemanticAction:
        """Return the continuous semantic action for learning code."""

        return self.predict(observation).action

    def act(self, observation: Mapping[str, Any]) -> tuple[Action, ...]:
        """Return executable deployment-v1 primitives for evaluation."""

        return self.predict(observation).primitive_actions(self.action_spec)


__all__ = [
    "ClosedLoopSteeringExpert",
    "SteeringDecision",
    "SteeringExpertConfig",
    "SteeringIntent",
]
