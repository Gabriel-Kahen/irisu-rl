"""Event-level renewable MPC and a candidate-local uncertainty barrier.

This module is development-only.  Simulator snapshots are used only to make
labels; the fitted policy consumes public observations and corrected joint-v2
candidates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from numbers import Integral, Real
from typing import Any

import numpy as np

from irisu_env import Action, ActionKind, EventKind
from irisu_rl.actions import ActionSpec

from .steering import SteeringDecision


EVENT_MPC_VERSION = "r3g-event-world-model-renewable-mpc-v1"
RENEWABLE_DETAILS = frozenset({"normal burst landing", "special color clear"})
TARGET_NAMES = (
    "shot_hit",
    "pair_joined",
    "first_renewal_reached",
    "second_renewal_reached",
    "time_to_first_renewal",
    "time_to_second_renewal",
    "first_renewal_gain",
    "second_renewal_gain",
    "rot_events",
    "rot_payment",
    "passive_drain",
    "level_delta",
    "b2",
    "second_cycle_margin",
    "final_gauge",
    "score_gain",
    "alive",
    "survival_ticks",
    "delta_b2",
    "delta_final_gauge",
    "delta_score_gain",
    "delta_survival_ticks",
    "catastrophic",
)
TARGET_INDEX = {name: index for index, name in enumerate(TARGET_NAMES)}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


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
    value = event.get("kind")
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    name = event.get("kind_name")
    if isinstance(name, str):
        try:
            return int(EventKind[name.upper()])
        except KeyError:
            return None
    return None


def _ordered_events(
    events: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        value
        for _, value in sorted(
            enumerate(events),
            key=lambda item: (
                _plain_int(item[1].get("tick"), "event tick", -1),
                _plain_int(item[1].get("sequence"), "event sequence", item[0]),
                item[0],
            ),
        )
    )


def _public_signature(observation: Mapping[str, Any]) -> str:
    bodies = []
    for raw in observation.get("bodies", ()):
        if not isinstance(raw, Mapping):
            continue
        bodies.append(
            (
                _plain_int(raw.get("id"), "body id", -1),
                str(raw.get("kind", "")),
                str(raw.get("shape", "")),
                str(raw.get("lifecycle", "")),
                _plain_int(raw.get("color"), "body color", -1),
                _plain_float(raw.get("x"), "body x"),
                _plain_float(raw.get("y"), "body y"),
                _plain_float(raw.get("vx"), "body vx"),
                _plain_float(raw.get("vy"), "body vy"),
                _plain_float(raw.get("angle"), "body angle"),
                _plain_float(raw.get("angular_velocity"), "body angular velocity"),
                _plain_float(raw.get("size"), "body size"),
                _plain_int(raw.get("chain_id"), "chain id"),
                _plain_int(raw.get("projectile_hits"), "projectile hits"),
                _plain_int(raw.get("remaining_lifetime"), "remaining lifetime"),
                _plain_int(raw.get("rot_timer"), "rot timer"),
            )
        )
    return _canonical_sha256(
        {
            "tick": _plain_int(observation.get("tick"), "tick"),
            "score": _plain_int(observation.get("score"), "score"),
            "gauge": _plain_int(observation.get("gauge"), "gauge"),
            "gauge_max": _plain_int(observation.get("gauge_max"), "gauge max"),
            "level": _plain_int(observation.get("level"), "level"),
            "qualifying_clears": _plain_int(
                observation.get("qualifying_clear_count"), "qualifying clears"
            ),
            "terminated": bool(observation.get("terminated", False)),
            "truncated": bool(observation.get("truncated", False)),
            "bodies": sorted(bodies),
        }
    )


def _step_duration(action: Action) -> int:
    return (
        int(action.wait_ticks)
        if ActionKind.parse(action.kind) is ActionKind.WAIT
        else 1
    )


@dataclass(frozen=True, slots=True)
class EventMPCConfig:
    renewal_events: int = 2
    maximum_event_ticks: int = 1_600
    cooldown_ticks: int = 16
    rot_delay_ticks: int = 40
    neighbor_count: int = 24
    calibration_fraction: float = 0.50
    conformal_alpha: float = 0.05
    risk_upper_limit: float = 0.25
    minimum_score_advantage: float = 0.0
    continuation_checkpoint_ticks: tuple[int, ...] = (
        2_000,
        10_000,
        20_000,
        50_000,
    )

    def __post_init__(self) -> None:
        if self.renewal_events != 2:
            raise ValueError("Strategy C binds planning to exactly two renewals")
        if not 64 <= self.maximum_event_ticks <= 10_000:
            raise ValueError("event horizon must be in [64, 10000]")
        if not 2 <= self.cooldown_ticks <= self.maximum_event_ticks:
            raise ValueError("cooldown must fit inside the event horizon")
        if self.rot_delay_ticks != 40:
            raise ValueError("the exact normal rot delay is 40 ticks")
        if not 3 <= self.neighbor_count <= 256:
            raise ValueError("neighbor count must be in [3, 256]")
        if not 0.1 <= self.calibration_fraction <= 0.5:
            raise ValueError("calibration fraction must be in [0.1, 0.5]")
        if not 0.0 < self.conformal_alpha < 0.5:
            raise ValueError("conformal alpha must be in (0, 0.5)")
        if not 0.0 < self.risk_upper_limit < 1.0:
            raise ValueError("risk limit must be in (0, 1)")
        if not math.isfinite(self.minimum_score_advantage):
            raise ValueError("score threshold must be finite")
        if (
            not self.continuation_checkpoint_ticks
            or tuple(sorted(set(self.continuation_checkpoint_ticks)))
            != self.continuation_checkpoint_ticks
            or self.continuation_checkpoint_ticks[0] < 1
        ):
            raise ValueError("continuation checkpoints must be positive and sorted")

    def manifest(self) -> dict[str, object]:
        values = asdict(self)
        values["continuation_checkpoint_ticks"] = list(
            self.continuation_checkpoint_ticks
        )
        return {
            "version": EVENT_MPC_VERSION,
            **values,
            "renewable_details": sorted(RENEWABLE_DETAILS),
            "classification": (
                "each candidate independently compared with exact incumbent "
                "candidate zero; branch-set membership is not an input"
            ),
            "evidence_scope": "development-only",
            "sealed_test_allowed": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class Liability:
    body_id: int
    created_tick: int
    deadline_tick: int
    discovery_level: int


@dataclass(slots=True)
class CandidateDebtLedger:
    """ID-keyed local liabilities with exact-once payment retirement."""

    rot_delay_ticks: int = 40
    active: dict[int, Liability] = field(default_factory=dict)
    created: dict[int, Liability] = field(default_factory=dict)
    paid: dict[int, int] = field(default_factory=dict)
    retired: set[int] = field(default_factory=set)
    duplicate_payments: int = 0
    unmatched_payments: int = 0
    current_level: int = 1

    @staticmethod
    def _penalty(level: int) -> int:
        return 1_800 + 20 * min(max(level, 1), 99)

    def observe(self, observation: Mapping[str, Any]) -> None:
        tick = _plain_int(observation.get("tick"), "ledger tick")
        level = _plain_int(observation.get("level"), "ledger level")
        self.current_level = level
        visible: set[int] = set()
        for raw in observation.get("bodies", ()):
            if not isinstance(raw, Mapping) or raw.get("kind") != "piece":
                continue
            body_id = _plain_int(raw.get("id"), "liability body id", -1)
            if body_id < 0:
                raise ValueError("liability body ID must be nonnegative")
            visible.add(body_id)
            if body_id in self.paid or body_id in self.retired:
                continue
            timer = _plain_int(raw.get("rot_timer"), "rot timer")
            if str(raw.get("lifecycle", "")) == "rotten" or timer <= 0:
                continue
            deadline = tick + max(1, self.rot_delay_ticks + 1 - timer)
            liability = Liability(
                body_id, tick, deadline, level
            )
            previous = self.active.get(body_id)
            if previous is not None:
                liability = Liability(
                    body_id,
                    previous.created_tick,
                    min(previous.deadline_tick, deadline),
                    previous.discovery_level,
                )
            self.active[body_id] = liability
            self.created.setdefault(body_id, liability)
        for body_id in tuple(self.active):
            if body_id not in visible:
                self.active.pop(body_id)
                if body_id not in self.paid:
                    self.retired.add(body_id)

    def apply(self, events: Sequence[Mapping[str, Any]]) -> None:
        by_tick: dict[int, list[Mapping[str, Any]]] = {}
        for event in _ordered_events(events):
            by_tick.setdefault(
                _plain_int(event.get("tick"), "event tick"), []
            ).append(event)
        for values in by_tick.values():
            rotten = {
                _plain_int(event.get("a"), "rotten body id", -1)
                for event in values
                if _event_kind(event) == int(EventKind.ROTTEN)
            }
            penalties = [
                event
                for event in values
                if _event_kind(event) == int(EventKind.GAUGE_CHANGED)
                and str(event.get("detail", "")) == "normal rot penalty"
                and _plain_int(event.get("value"), "rot payment") < 0
            ]
            paid_this_tick: set[int] = set()
            for event in penalties:
                body_id = _plain_int(event.get("a"), "payment body id", -1)
                if body_id in self.paid or body_id in paid_this_tick:
                    self.duplicate_payments += 1
                    continue
                if body_id not in rotten:
                    self.unmatched_payments += 1
                    continue
                self.paid[body_id] = -_plain_int(
                    event.get("value"), "rot payment"
                )
                paid_this_tick.add(body_id)
                self.active.pop(body_id, None)
                self.retired.discard(body_id)
            self.unmatched_payments += len(rotten - paid_this_tick)
            retired = {
                _plain_int(event.get("a"), "retired body id", -1)
                for event in values
                if _event_kind(event)
                in {
                    int(EventKind.CLEARED),
                    int(EventKind.DESTROYED),
                    int(EventKind.EJECTED),
                }
            }
            # A same-tick rot penalty wins over teardown: CLEARED is not proof
            # that the liability vanished before actor-update ordering.
            for body_id in retired - paid_this_tick:
                if body_id in self.active:
                    self.active.pop(body_id)
                    self.retired.add(body_id)

    def liabilities_due_by(self, tick: int) -> tuple[Liability, ...]:
        """Return active liabilities due by ``tick`` in stable deadline order."""

        return tuple(
            sorted(
                (
                    value
                    for value in self.active.values()
                    if value.deadline_tick <= tick
                ),
                key=lambda value: (value.deadline_tick, value.body_id),
            )
        )

    def debt_due_by(
        self,
        tick: int,
        *,
        level_at_payment: int | Callable[[Liability], int] | None = None,
    ) -> int:
        """Price debt, with a per-liability resolver for future certification.

        Omitting ``level_at_payment`` preserves the current-observation
        projection used by local exact-once ledger checks. Safety certificates
        must supply the realized/prospective payment-time level.
        """

        liabilities = self.liabilities_due_by(tick)
        if level_at_payment is None:
            levels = (self.current_level for _ in liabilities)
        elif callable(level_at_payment):
            levels = (
                _plain_int(
                    level_at_payment(liability),
                    "liability payment level",
                )
                for liability in liabilities
            )
        else:
            level = _plain_int(level_at_payment, "liability payment level")
            levels = (level for _ in liabilities)
        return sum(self._penalty(level) for level in levels)


@dataclass(frozen=True, slots=True)
class RenewableCycle:
    ordinal: int
    start_tick: int
    end_tick: int
    completed: bool
    start_gauge: int
    renewal_gain: int
    passive_drain: int
    rot_payment: int
    rot_events: int
    realized_rot_liability: int
    minimum_gauge_before_renewal: int
    solvency_surplus: int

    def manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CashflowReplay:
    cycles: tuple[RenewableCycle, ...]
    renewal_ticks: tuple[int, ...]
    net_renewal_gains: tuple[int, ...]
    b2: int
    final_replayed_gauge: int
    rot_events: int
    rot_payment: int
    passive_drain: int
    terminal_before_second: bool


def replay_two_renewal_cashflow(
    *,
    initial_gauge: int,
    gauge_max: int,
    initial_level: int,
    start_tick: int,
    final_tick: int,
    events: Sequence[Mapping[str, Any]],
) -> CashflowReplay:
    """Replay the protocol cash-flow recurrence in exact event/tick order."""

    if gauge_max <= 0 or not 0 <= initial_gauge <= gauge_max:
        raise ValueError("cash-flow gauge state is invalid")
    by_tick: dict[int, list[Mapping[str, Any]]] = {}
    for event in _ordered_events(events):
        tick = _plain_int(event.get("tick"), "cash-flow event tick")
        if not start_tick < tick <= final_tick:
            raise RuntimeError("cash-flow event lies outside the branch")
        by_tick.setdefault(tick, []).append(event)
    q = initial_gauge
    actual_q = initial_gauge
    level = initial_level
    cycle_start_tick = start_tick
    cycle_start_gauge = q
    cycle_minimum_margin = q - 1
    cycle_passive = cycle_rot = cycle_rot_events = 0
    cycles: list[RenewableCycle] = []
    renewal_ticks: list[int] = []
    renewal_gains: list[int] = []
    total_passive = total_rot = total_rot_events = 0
    terminal_before_second = q <= 0
    for tick in range(start_tick + 1, final_tick + 1):
        values = by_tick.get(tick, ())
        if q <= 0:
            terminal_before_second = len(renewal_ticks) < 2
            break
        entry = q
        cycle_minimum_margin = min(cycle_minimum_margin, entry - 1)
        raw_gain = sum(
            _plain_int(event.get("value"), "renewable gain")
            for event in values
            if _event_kind(event) == int(EventKind.GAUGE_CHANGED)
            and _plain_int(event.get("value"), "renewable gain") > 0
            and str(event.get("detail", "")) in RENEWABLE_DETAILS
        )
        renewable_epoch = raw_gain > 0
        x = min(max(q + raw_gain, 0), gauge_max)
        net_gain = x - q
        cycle_minimum_margin = min(cycle_minimum_margin, x - 1)
        level_events = [
            _plain_int(event.get("value"), "level value")
            for event in values
            if _event_kind(event) == int(EventKind.LEVEL_CHANGED)
        ]
        if level_events:
            level = level_events[-1]
        parameter_level = min(max(level, 1), 99)
        drain = (parameter_level // 10 + 1) * (
            3 if x > gauge_max / 2 else 1
        )
        after_drain = max(1, x - drain)
        penalties = [
            event
            for event in values
            if _event_kind(event) == int(EventKind.GAUGE_CHANGED)
            and str(event.get("detail", "")) == "normal rot penalty"
            and _plain_int(event.get("value"), "rot payment") < 0
        ]
        expected_penalty = 1_800 + 20 * parameter_level
        if any(
            -_plain_int(event.get("value"), "rot payment") != expected_penalty
            for event in penalties
        ):
            raise RuntimeError("rot payment disagrees with level-at-rot")
        rot_payment = expected_penalty * len(penalties)
        rotten = sum(
            _event_kind(event) == int(EventKind.ROTTEN) for event in values
        )
        if rotten != len(penalties):
            raise RuntimeError("rot event/payment pairing is not exact")
        q = after_drain - rot_payment
        actual_q += sum(
            _plain_int(event.get("value"), "observed gauge delta")
            for event in values
            if _event_kind(event) == int(EventKind.GAUGE_CHANGED)
        )
        if q != actual_q:
            raise RuntimeError(
                "event-ordered cash-flow recurrence disagrees with exact gauge"
            )
        cycle_passive += drain
        cycle_rot += rot_payment
        cycle_rot_events += rotten
        total_passive += drain
        total_rot += rot_payment
        total_rot_events += rotten
        cycle_minimum_margin = min(cycle_minimum_margin, q - 1)
        if renewable_epoch and len(cycles) < 2:
            cycles.append(
                RenewableCycle(
                    len(cycles) + 1,
                    cycle_start_tick,
                    tick,
                    True,
                    cycle_start_gauge,
                    net_gain,
                    cycle_passive,
                    cycle_rot,
                    cycle_rot_events,
                    cycle_rot,
                    min(entry, x, q),
                    cycle_minimum_margin,
                )
            )
            renewal_ticks.append(tick)
            renewal_gains.append(net_gain)
            cycle_start_tick = tick
            cycle_start_gauge = q
            cycle_minimum_margin = q - 1
            cycle_passive = cycle_rot = cycle_rot_events = 0
        if len(cycles) >= 2:
            # The target ends at the second distinct renewal epoch.  Events
            # later in an aggregated wait are irrelevant to B2.
            break
    while len(cycles) < 2:
        cycles.append(
            RenewableCycle(
                len(cycles) + 1,
                cycle_start_tick,
                min(final_tick, max(by_tick, default=final_tick)),
                False,
                cycle_start_gauge,
                0,
                cycle_passive,
                cycle_rot,
                cycle_rot_events,
                cycle_rot,
                q,
                cycle_minimum_margin,
            )
        )
        cycle_start_tick = final_tick
        cycle_start_gauge = q
        cycle_minimum_margin = q - 1
        cycle_passive = cycle_rot = cycle_rot_events = 0
    return CashflowReplay(
        tuple(cycles),
        tuple(renewal_ticks),
        tuple(renewal_gains),
        min(value.solvency_surplus for value in cycles),
        q,
        total_rot_events,
        total_rot,
        total_passive,
        terminal_before_second,
    )


@dataclass(frozen=True, slots=True)
class ExactEventOutcome:
    candidate_ordinal: int
    candidate_name: str
    pair_ordinal: int
    geometry_ordinal: int
    pair_category: str
    start_tick: int
    survival_ticks: int
    alive: bool
    invalid_actions: int
    score_gain: int
    final_gauge: int
    level_delta: int
    qualifying_clear_gain: int
    shot_hit: bool
    pair_joined: bool
    renewable_ticks: tuple[int, ...]
    renewable_gains: tuple[int, ...]
    cycles: tuple[RenewableCycle, ...]
    rot_events: int
    rot_payment: int
    passive_drain: int
    liabilities_created: int
    liabilities_paid: int
    liabilities_retired: int
    duplicate_payments: int
    unmatched_payments: int
    snapshot_sha256: str
    exact_state_hash: str | None
    full_action_terminal: bool = False
    full_action_gauge_failure: bool = False
    controller_rebind_valid: bool = True
    action_equivalent_to_incumbent: bool = False

    @property
    def renewals_reached(self) -> int:
        return len(self.renewable_ticks)

    @property
    def two_renewal_complete(self) -> bool:
        return self.renewals_reached >= 2

    @property
    def minimum_surplus(self) -> int:
        return min(
            (cycle.solvency_surplus for cycle in self.cycles),
            default=self.final_gauge,
        )

    @property
    def b2(self) -> int:
        return self.minimum_surplus

    @property
    def hard_negative(self) -> bool:
        return (
            not self.controller_rebind_valid
            or not self.alive
            or self.final_gauge <= 1
            or self.minimum_surplus < 0
        )

    def catastrophic_against(self, incumbent: ExactEventOutcome) -> bool:
        return (
            not self.controller_rebind_valid
            or self.invalid_actions > 0
            or (not self.alive and incumbent.alive)
            or (
                self.renewals_reached < incumbent.renewals_reached
                and incumbent.two_renewal_complete
            )
            or (
                self.minimum_surplus < 0 <= incumbent.minimum_surplus
            )
        )

    def target_vector(self, incumbent: ExactEventOutcome) -> np.ndarray:
        first = self.cycles[0] if self.cycles else None
        second = self.cycles[1] if len(self.cycles) > 1 else None
        values = (
            float(self.shot_hit),
            float(self.pair_joined),
            float(self.renewals_reached >= 1),
            float(self.renewals_reached >= 2),
            float(
                self.renewable_ticks[0] - self.start_tick
                if self.renewable_ticks
                else self.survival_ticks
            ),
            float(
                self.renewable_ticks[1] - self.start_tick
                if len(self.renewable_ticks) > 1
                else self.survival_ticks
            ),
            float(self.renewable_gains[0] if self.renewable_gains else 0),
            float(self.renewable_gains[1] if len(self.renewable_gains) > 1 else 0),
            float(self.rot_events),
            float(self.rot_payment),
            float(self.passive_drain),
            float(self.level_delta),
            float(self.minimum_surplus),
            float(second.solvency_surplus if second else self.minimum_surplus),
            float(self.final_gauge),
            float(self.score_gain),
            float(self.alive),
            float(self.survival_ticks),
            float(self.minimum_surplus - incumbent.minimum_surplus),
            float(self.final_gauge - incumbent.final_gauge),
            float(self.score_gain - incumbent.score_gain),
            float(self.survival_ticks - incumbent.survival_ticks),
            float(self.catastrophic_against(incumbent)),
        )
        return np.asarray(values, dtype=np.float64)

    def manifest(self) -> dict[str, object]:
        return {
            **asdict(self),
            "renewals_reached": self.renewals_reached,
            "two_renewal_complete": self.two_renewal_complete,
            "minimum_surplus": self.minimum_surplus,
            "b2": self.b2,
            "hard_negative": self.hard_negative,
            "accounting_horizon": (
                "B2, score, gauge, and event targets stop at tau2; "
                "full-action terminal flags include the tail of an already "
                "issued atomic action"
            ),
            "cycles": [value.manifest() for value in self.cycles],
        }


@dataclass(frozen=True, slots=True)
class ExactSearchResult:
    query_id: str
    snapshot_sha256: str
    candidates: tuple[Any, ...]
    outcomes: tuple[ExactEventOutcome, ...]
    restore_checks: int
    wall_seconds: float
    cpu_seconds: float

    def __post_init__(self) -> None:
        if len(self.candidates) != len(self.outcomes) or not self.outcomes:
            raise ValueError("exact search must retain every candidate outcome")
        if self.outcomes[0].candidate_ordinal != 0:
            raise ValueError("candidate zero must be the exact incumbent")
        candidate_ordinals = tuple(
            getattr(value, "ordinal", None) for value in self.candidates
        )
        outcome_ordinals = tuple(
            value.candidate_ordinal for value in self.outcomes
        )
        if (
            len(set(outcome_ordinals)) != len(outcome_ordinals)
            or (
                all(value is not None for value in candidate_ordinals)
                and (
                    len({int(value) for value in candidate_ordinals})
                    != len(candidate_ordinals)
                    or outcome_ordinals
                    != tuple(int(value) for value in candidate_ordinals)
                )
            )
        ):
            raise ValueError("candidate/outcome ordinals changed")

    def action_equivalent_to_incumbent(self, ordinal: int) -> bool:
        return (
            ordinal == 0
            or self.outcome_for_ordinal(ordinal).action_equivalent_to_incumbent
        )

    def outcome_for_ordinal(self, ordinal: int) -> ExactEventOutcome:
        matches = [
            value
            for value in self.outcomes
            if value.candidate_ordinal == ordinal
        ]
        if len(matches) != 1:
            raise RuntimeError("candidate outcome ordinal is not unique")
        return matches[0]

    def candidate_for_ordinal(self, ordinal: int) -> Any:
        matches = [
            value
            for value in self.candidates
            if int(getattr(value, "ordinal", -1)) == ordinal
        ]
        if len(matches) != 1:
            raise RuntimeError("candidate ordinal is not unique")
        return matches[0]

    @property
    def incumbent(self) -> ExactEventOutcome:
        return self.outcomes[0]

    def safe(self, outcome: ExactEventOutcome) -> bool:
        base = self.incumbent
        return (
            outcome.controller_rebind_valid
            and outcome.invalid_actions == 0
            and outcome.alive >= base.alive
            and outcome.two_renewal_complete
            and base.two_renewal_complete
            and outcome.minimum_surplus >= base.minimum_surplus
            and outcome.minimum_surplus >= 0
            and not self.action_equivalent_to_incumbent(
                outcome.candidate_ordinal
            )
        )

    @property
    def selected_ordinal(self) -> int:
        eligible = [
            value
            for value in self.outcomes
            if self.safe(value) and value.score_gain > self.incumbent.score_gain
        ]
        if not eligible:
            return 0
        objective = lambda value: (
            value.score_gain,
            value.minimum_surplus,
            value.final_gauge,
        )
        best = max(objective(value) for value in eligible)
        winners = [value for value in eligible if objective(value) == best]
        return winners[0].candidate_ordinal if len(winners) == 1 else 0

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    def manifest(self) -> dict[str, object]:
        return {
            "version": EVENT_MPC_VERSION,
            "query_id": self.query_id,
            "snapshot_sha256": self.snapshot_sha256,
            "restore_checks": self.restore_checks,
            "selected_ordinal": self.selected_ordinal,
            "outcomes": [value.manifest() for value in self.outcomes],
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
        }


class ExactEventPlanner:
    """Generate exact two-renewal labels from byte-identical clone states."""

    def __init__(
        self,
        continuation_factory: Callable[[], object],
        candidate_provider: Callable[
            [Mapping[str, Any], SteeringDecision], Sequence[Any]
        ],
        *,
        config: EventMPCConfig | None = None,
        action_spec: ActionSpec | None = None,
        continuation_identity_sha256: str | None = None,
        continuation_rebind: Callable[
            [object, Mapping[str, Any], SteeringDecision, SteeringDecision],
            bool,
        ]
        | None = None,
    ) -> None:
        self.continuation_factory = continuation_factory
        self.candidate_provider = candidate_provider
        self.config = EventMPCConfig() if config is None else config
        self.action_spec = ActionSpec() if action_spec is None else action_spec
        self.continuation_identity_sha256 = continuation_identity_sha256
        self.continuation_rebind = continuation_rebind

    def identity_manifest(self) -> dict[str, object]:
        return {
            "version": EVENT_MPC_VERSION,
            "config": self.config.manifest(),
            "action_spec_sha256": self.action_spec.sha256,
            "continuation_identity_sha256": self.continuation_identity_sha256,
            "continuation_state_fields": [
                "_cooldown_until",
                "_last_tick",
                "_last_decision",
                "_progress",
            ],
            "continuation_rebind_required": True,
            "branching": (
                "byte-identical portable snapshot, public signature, and native "
                "state hash verified before every branch and in finally; live "
                "frozen-v5 controller state cloned and candidate-rebound"
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())

    def _clone_continuation(self, source: object) -> object:
        continuation = self.continuation_factory()
        if source is None and continuation is None:
            if self.continuation_identity_sha256 is not None:
                raise RuntimeError("event continuation policy is required")
            return continuation
        if type(continuation) is not type(source):
            raise RuntimeError("event continuation controller type changed")
        if self.continuation_identity_sha256 is not None and (
            getattr(source, "artifact_sha256", None)
            != self.continuation_identity_sha256
            or getattr(continuation, "artifact_sha256", None)
            != self.continuation_identity_sha256
        ):
            raise RuntimeError("event continuation identity changed")
        for name in (
            "_cooldown_until",
            "_last_tick",
            "_last_decision",
            "_progress",
        ):
            if not hasattr(source, name) or not hasattr(continuation, name):
                raise RuntimeError(
                    f"event continuation lacks required state {name}"
                )
            setattr(continuation, name, copy.deepcopy(getattr(source, name)))
        return continuation

    def _evaluate(
        self,
        env: Any,
        initial: Mapping[str, Any],
        candidate: Any,
        continuation: object,
        snapshot_sha256: str,
        exact_state_hash: str | None,
    ) -> ExactEventOutcome:
        start_tick = _plain_int(initial.get("tick"), "initial tick")
        start_score = _plain_int(initial.get("score"), "initial score")
        start_level = _plain_int(initial.get("level"), "initial level")
        start_clears = _plain_int(
            initial.get("qualifying_clear_count"), "initial clears"
        )
        current = initial
        terminated = bool(initial.get("terminated", False))
        truncated = bool(initial.get("truncated", False))
        events: list[Mapping[str, Any]] = []
        observations: list[Mapping[str, Any]] = [initial]
        first_projectiles: set[int] = set()
        first_phase = True
        ledger = CandidateDebtLedger(self.config.rot_delay_ticks)
        ledger.observe(initial)

        def step(action: Action) -> None:
            nonlocal current, terminated, truncated, first_phase
            current, _reward, terminated, truncated, info = env.step(action)
            if not isinstance(current, Mapping) or not isinstance(info, Mapping):
                raise TypeError("event branch transition is malformed")
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
            ledger.apply(supplied)
            ledger.observe(current)
            observations.append(current)

        for action in candidate.decision.primitive_actions(self.action_spec):
            if terminated or truncated:
                break
            step(action)
        first_phase = False

        def renewal_count() -> int:
            return len(
                {
                    _plain_int(value.get("tick"), "renewal tick")
                    for value in events
                    if _event_kind(value) == int(EventKind.GAUGE_CHANGED)
                    and _plain_int(value.get("value"), "gauge delta") > 0
                    and str(value.get("detail", "")) in RENEWABLE_DETAILS
                }
            )

        while not (terminated or truncated):
            elapsed = _plain_int(current.get("tick"), "current tick") - start_tick
            if (
                elapsed >= self.config.maximum_event_ticks
                or renewal_count() >= self.config.renewal_events
            ):
                break
            decision = getattr(continuation, "predict")(current)
            if not isinstance(decision, SteeringDecision):
                raise TypeError("event continuation returned a non-decision")
            issued_actions = decision.primitive_actions(self.action_spec)
            for action_index, action in enumerate(issued_actions):
                if terminated or truncated:
                    break
                remaining = self.config.maximum_event_ticks - (
                    _plain_int(current.get("tick"), "current tick") - start_tick
                )
                issued_tail = action_index > 0
                if remaining <= 0 and not issued_tail:
                    break
                if _step_duration(action) > remaining:
                    if issued_tail:
                        # Match the real runner: once a shot decision is issued,
                        # its release primitive completes even if press reached
                        # tau2 or the event cap.
                        pass
                    elif ActionKind.parse(action.kind) is not ActionKind.WAIT:
                        raise RuntimeError("event shot crossed the planning horizon")
                    else:
                        action = Action.wait(remaining)
                current_tick = _plain_int(
                    current.get("tick"), "current tick"
                )
                checkpoints = [
                    value
                    for value in self.config.continuation_checkpoint_ticks
                    if value > current_tick
                ]
                if (
                    ActionKind.parse(action.kind) is ActionKind.WAIT
                    and checkpoints
                    and current_tick + int(action.wait_ticks) > checkpoints[0]
                ):
                    action = Action.wait(checkpoints[0] - current_tick)
                step(action)

        final_tick = _plain_int(current.get("tick"), "final tick")
        ordered = _ordered_events(events)
        accounting_final_tick = min(
            final_tick, start_tick + self.config.maximum_event_ticks
        )
        accounting_events = tuple(
            event
            for event in ordered
            if _plain_int(event.get("tick"), "accounting event tick")
            <= accounting_final_tick
        )
        cashflow = replay_two_renewal_cashflow(
            initial_gauge=_plain_int(initial.get("gauge"), "initial gauge"),
            gauge_max=_plain_int(initial.get("gauge_max"), "gauge maximum"),
            initial_level=start_level,
            start_tick=start_tick,
            final_tick=accounting_final_tick,
            events=accounting_events,
        )
        gauge = cashflow.final_replayed_gauge
        cycles = cashflow.cycles
        renewals = cashflow.renewal_ticks
        gains = cashflow.net_renewal_gains
        target_end_tick = (
            renewals[1] if len(renewals) >= 2 else accounting_final_tick
        )
        target_events = tuple(
            event
            for event in accounting_events
            if _plain_int(event.get("tick"), "target event tick")
            <= target_end_tick
        )
        counts = Counter(
            value
            for event in target_events
            if (value := _event_kind(event)) is not None
        )
        full_counts = Counter(
            value
            for event in ordered
            if (value := _event_kind(event)) is not None
        )
        full_action_terminal = bool(
            full_counts[int(EventKind.GAME_OVER)] or terminated
        )
        hit = any(
            _event_kind(event) == int(EventKind.PROJECTILE_HIT)
            and _plain_int(event.get("a"), "hit projectile", -1)
            in first_projectiles
            and _plain_int(event.get("b"), "hit body", -1)
            == candidate.pair.source_body_id
            for event in target_events
        )
        destination_ids = {
            _plain_int(raw.get("id"), "destination body id", -1)
            for raw in initial.get("bodies", ())
            if isinstance(raw, Mapping)
            and candidate.pair.destination_chain_id
            and _plain_int(raw.get("chain_id"), "chain id")
            == candidate.pair.destination_chain_id
        } or {candidate.pair.destination_body_id}
        joined = any(
            _event_kind(event) == int(EventKind.CHAIN_JOINED)
            and candidate.pair.source_body_id
            in {
                _plain_int(event.get("a"), "join a", -1),
                _plain_int(event.get("b"), "join b", -1),
            }
            and bool(
                destination_ids
                & {
                    _plain_int(event.get("a"), "join a", -1),
                    _plain_int(event.get("b"), "join b", -1),
                }
            )
            for event in target_events
        )
        level_events = [
            _plain_int(event.get("value"), "final target level")
            for event in target_events
            if _event_kind(event) == int(EventKind.LEVEL_CHANGED)
        ]
        return ExactEventOutcome(
            candidate.ordinal,
            (
                f"{candidate.pair.category}/"
                f"{candidate.geometry.name}"
            ),
            candidate.pair_ordinal,
            candidate.geometry_ordinal,
            candidate.pair.category,
            start_tick,
            target_end_tick - start_tick,
            not full_action_terminal,
            counts[int(EventKind.INVALID_ACTION)],
            sum(
                _plain_int(event.get("value"), "score delta")
                for event in target_events
                if _event_kind(event) == int(EventKind.SCORE_CHANGED)
            ),
            gauge,
            (level_events[-1] if level_events else start_level) - start_level,
            sum(
                _event_kind(event) == int(EventKind.CONFIRMED)
                and str(event.get("detail", "")) == "normal burst qualified"
                for event in target_events
            ),
            hit,
            joined,
            tuple(renewals),
            tuple(gains),
            tuple(cycles),
            cashflow.rot_events,
            cashflow.rot_payment,
            cashflow.passive_drain,
            len(ledger.created),
            len(ledger.paid),
            len(ledger.retired),
            ledger.duplicate_payments,
            ledger.unmatched_payments,
            snapshot_sha256,
            exact_state_hash,
            full_action_terminal,
            bool(full_counts[int(EventKind.GAME_OVER)]),
            True,
            False,
        )

    def search(
        self,
        env: Any,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
        *,
        continuation_policy: object | None = None,
        query_id: str,
    ) -> ExactSearchResult:
        if getattr(env, "physics_backend", None) != "portable":
            raise ValueError("event MPC requires the portable backend")
        if not incumbent.is_shot:
            raise ValueError("event MPC requires an incumbent shot")
        candidates = tuple(self.candidate_provider(observation, incumbent))
        if not candidates or candidates[0].decision is not incumbent:
            raise RuntimeError("candidate zero is not the incoming frozen-v5 shot")
        if (
            continuation_policy is None
            and self.continuation_identity_sha256 is not None
        ):
            raise RuntimeError("live frozen-v5 continuation policy is required")
        if (
            hasattr(continuation_policy, "_last_decision")
            and getattr(continuation_policy, "_last_decision") is not incumbent
        ):
            raise RuntimeError(
                "live frozen-v5 controller does not own the incumbent decision"
            )
        snapshot = env.clone_state()
        snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
        signature = _public_signature(observation)
        state_hash = getattr(env, "state_hash", None)
        exact_hash = state_hash() if callable(state_hash) else None
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        outcomes: list[ExactEventOutcome] = []
        restore_checks = 0

        def restore_checked() -> Mapping[str, Any]:
            nonlocal restore_checks
            restored = env.restore_state(snapshot)
            restore_checks += 1
            if (
                _public_signature(restored) != signature
                or env.clone_state() != snapshot
                or (
                    exact_hash is not None
                    and callable(state_hash)
                    and state_hash() != exact_hash
                )
            ):
                raise RuntimeError("event branch restore was not exact")
            return restored

        try:
            incumbent_actions = incumbent.primitive_actions(self.action_spec)
            for candidate in candidates:
                restored = restore_checked()
                continuation = self._clone_continuation(continuation_policy)
                equivalent = (
                    candidate.ordinal != 0
                    and candidate.decision.primitive_actions(self.action_spec)
                    == incumbent_actions
                )
                rebound = True
                if candidate.ordinal != 0 and not equivalent:
                    if self.continuation_rebind is None:
                        rebound = False
                    else:
                        try:
                            rebound = bool(
                                self.continuation_rebind(
                                    continuation,
                                    observation,
                                    incumbent,
                                    candidate.decision,
                                )
                            )
                        except (RuntimeError, TypeError, ValueError):
                            rebound = False
                if not rebound:
                    if not outcomes:
                        raise RuntimeError(
                            "incumbent controller rebind failed structurally"
                        )
                    outcomes.append(
                        replace(
                            outcomes[0],
                            candidate_ordinal=candidate.ordinal,
                            candidate_name=(
                                f"{candidate.pair.category}/"
                                f"{candidate.geometry.name}"
                            ),
                            pair_ordinal=candidate.pair_ordinal,
                            geometry_ordinal=candidate.geometry_ordinal,
                            pair_category=candidate.pair.category,
                            invalid_actions=outcomes[0].invalid_actions + 1,
                            controller_rebind_valid=False,
                            action_equivalent_to_incumbent=False,
                        )
                    )
                    continue
                outcome = self._evaluate(
                    env,
                    restored,
                    candidate,
                    continuation,
                    snapshot_sha256,
                    exact_hash,
                )
                outcomes.append(
                    replace(
                        outcome,
                        action_equivalent_to_incumbent=equivalent,
                    )
                )
        finally:
            restore_checked()
        if any(
            value.duplicate_payments or value.unmatched_payments
            for value in outcomes
        ):
            raise RuntimeError("event ledger did not retire payments exactly once")
        return ExactSearchResult(
            query_id,
            snapshot_sha256,
            candidates,
            tuple(outcomes),
            restore_checks,
            time.perf_counter() - wall_started,
            time.process_time() - cpu_started,
        )


FEATURE_NAMES = (
    "gauge_fraction",
    "gauge_half_surplus_fraction",
    "level_fraction",
    "log_score",
    "tick_spawn_phase",
    "qualifying_clears_fraction",
    "body_count_fraction",
    "rotten_count_fraction",
    "grouped_count_fraction",
    "liability_count_fraction",
    "earliest_deadline_fraction",
    "visible_debt_fraction",
    "source_rot_fraction",
    "source_lifetime_fraction",
    "source_y_fraction",
    "destination_y_fraction",
    "pair_distance_fraction",
    "destination_chain_fraction",
    "category_rotten",
    "category_anchor",
    "category_fresh",
    "geometry_weak",
    "geometry_side",
    "geometry_below",
    "pair_changed",
    "geometry_changed",
    "source_falling",
    "destination_rotten",
    "source_projectile_hits",
)


def candidate_features(
    observation: Mapping[str, Any],
    candidate: Any,
) -> np.ndarray:
    gauge = _plain_int(observation.get("gauge"), "gauge")
    gauge_max = max(1, _plain_int(observation.get("gauge_max"), "gauge max"))
    level = min(max(_plain_int(observation.get("level"), "level"), 1), 99)
    bodies = tuple(
        value
        for value in observation.get("bodies", ())
        if isinstance(value, Mapping)
    )
    by_id = {
        _plain_int(value.get("id"), "body id", -1): value for value in bodies
    }
    source = by_id.get(candidate.pair.source_body_id, {})
    destination = by_id.get(candidate.pair.destination_body_id, {})
    liabilities = []
    for value in bodies:
        timer = _plain_int(value.get("rot_timer"), "rot timer")
        if (
            value.get("kind") == "piece"
            and str(value.get("lifecycle", "")) != "rotten"
            and timer > 0
        ):
            liabilities.append(max(1, 41 - timer))
    penalty = 1_800 + 20 * level
    visible_debt = len(liabilities) * penalty
    difficulty = observation.get("difficulty", {})
    field = observation.get("field", {})
    spawn = (
        _plain_int(difficulty.get("spawn_interval_ticks"), "spawn interval", 1)
        if isinstance(difficulty, Mapping)
        else 1
    )
    spawn = max(spawn, 1)
    field_y = (
        _plain_float(field.get("y"), "field y")
        if isinstance(field, Mapping)
        else 0.0
    )
    field_height = (
        _plain_float(field.get("height"), "field height", 600.0)
        if isinstance(field, Mapping)
        else 600.0
    )
    field_height = max(field_height, 1.0)

    def y_fraction(body: Mapping[str, Any]) -> float:
        return (
            _plain_float(body.get("effect_y", body.get("y")), "body y")
            - field_y
        ) / field_height

    category = str(candidate.pair.category)
    geometry = candidate.geometry
    values = (
        gauge / gauge_max,
        (gauge - gauge_max / 2.0) / gauge_max,
        level / 99.0,
        math.log1p(max(0, _plain_int(observation.get("score"), "score"))) / 16.0,
        (_plain_int(observation.get("tick"), "tick") % spawn) / spawn,
        _plain_int(
            observation.get("qualifying_clear_count"), "qualifying clears"
        )
        / 100.0,
        len(bodies) / 64.0,
        sum(str(value.get("lifecycle", "")) == "rotten" for value in bodies)
        / 16.0,
        sum(_plain_int(value.get("chain_id"), "chain id") > 0 for value in bodies)
        / 32.0,
        len(liabilities) / 16.0,
        (min(liabilities) if liabilities else 1_600) / 1_600.0,
        visible_debt / gauge_max,
        _plain_int(source.get("rot_timer"), "source rot timer") / 40.0,
        _plain_int(
            source.get("remaining_lifetime"), "source remaining lifetime"
        )
        / 1_000.0,
        y_fraction(source),
        y_fraction(destination),
        float(candidate.pair.distance_sizes) / 20.0,
        float(candidate.pair.destination_chain_id > 0),
        float(category == "rotten-hazard"),
        float(category == "viable-anchor"),
        float(category == "fresh-match"),
        float(str(geometry.strength) == "weak"),
        float(geometry.side_sizes),
        float(geometry.below_sizes),
        float(
            not bool(
                getattr(
                    candidate.pair,
                    "incumbent",
                    candidate.pair_ordinal == 0,
                )
            )
        ),
        float(str(geometry.name) != "analytic-strong"),
        float(
            str(source.get("lifecycle", ""))
            in {"scripted_falling", "falling"}
        ),
        float(str(destination.get("lifecycle", "")) == "rotten"),
        _plain_int(source.get("projectile_hits"), "projectile hits") / 4.0,
    )
    if len(values) != len(FEATURE_NAMES):
        raise RuntimeError("event feature schema changed")
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ModelExample:
    split: str
    seed: int
    query_id: str
    decision_tick: int
    candidate_ordinal: int
    features: tuple[float, ...]
    targets: tuple[float, ...]
    outcome: Mapping[str, object]

    def __post_init__(self) -> None:
        if len(self.features) != len(FEATURE_NAMES):
            raise ValueError("model example feature width changed")
        if len(self.targets) != len(TARGET_NAMES):
            raise ValueError("model example target width changed")

    def manifest(self) -> dict[str, object]:
        return asdict(self)


def search_examples(
    result: ExactSearchResult,
    observation: Mapping[str, Any],
    *,
    split: str,
    seed: int,
) -> tuple[ModelExample, ...]:
    incumbent = result.incumbent
    return tuple(
        ModelExample(
            split,
            seed,
            result.query_id,
            _plain_int(observation.get("tick"), "decision tick"),
            outcome.candidate_ordinal,
            tuple(
                float(value)
                for value in candidate_features(observation, candidate)
            ),
            tuple(float(value) for value in outcome.target_vector(incumbent)),
            outcome.manifest(),
        )
        for candidate, outcome in zip(
            result.candidates, result.outcomes, strict=True
        )
    )


@dataclass(frozen=True, slots=True)
class ModelPrediction:
    mean: tuple[float, ...]
    lower: tuple[float, ...]
    local_std: tuple[float, ...]
    neighbor_indices: tuple[int, ...]
    neighbor_distances: tuple[float, ...]
    catastrophic_probability: float
    catastrophic_upper: float
    all_neighbors_two_renewal: bool

    def value(self, name: str) -> float:
        return self.mean[TARGET_INDEX[name]]

    def lower_value(self, name: str) -> float:
        return self.lower[TARGET_INDEX[name]]


@dataclass(frozen=True, slots=True)
class CandidateCertificate:
    certified: bool
    reasons: tuple[str, ...]
    lower_b2: float
    lower_delta_b2: float
    lower_delta_final_gauge: float
    predicted_score_advantage: float
    catastrophic_probability: float
    catastrophic_upper: float
    all_neighbors_two_renewal: bool

    def manifest(self) -> dict[str, object]:
        return asdict(self)


class KNNEventWorldModel:
    """Compact local event model with split-conformal lower residuals."""

    def __init__(self, config: EventMPCConfig | None = None) -> None:
        self.config = EventMPCConfig() if config is None else config
        self.center = np.empty(0, dtype=np.float64)
        self.scale = np.empty(0, dtype=np.float64)
        self.features = np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
        self.targets = np.empty((0, len(TARGET_NAMES)), dtype=np.float64)
        self.lower_offsets = np.zeros(len(TARGET_NAMES), dtype=np.float64)
        self.fit_seeds: tuple[int, ...] = ()
        self.calibration_seeds: tuple[int, ...] = ()
        self.conformal_q = math.inf
        self.conformal_rank = 0
        self.conformal_records: tuple[dict[str, int | float], ...] = ()
        self.calibration_residual_seeds: tuple[int, ...] = ()
        self.calibration_residuals = np.empty(
            (0, len(TARGET_NAMES)), dtype=np.float64
        )

    @property
    def fitted(self) -> bool:
        return bool(self.features.shape[0])

    @staticmethod
    def _seed_partition(
        examples: Sequence[ModelExample], calibration_fraction: float
    ) -> tuple[set[int], set[int]]:
        seeds = sorted({value.seed for value in examples})
        if len(seeds) < 8:
            raise ValueError("event model needs at least eight whole-seed clusters")
        ranked = sorted(
            seeds,
            key=lambda value: hashlib.sha256(
                f"r3g-event-calibration|{value}".encode()
            ).digest(),
        )
        count = max(1, round(len(ranked) * calibration_fraction))
        count = min(count, len(ranked) - 2)
        return set(ranked[count:]), set(ranked[:count])

    def fit(
        self,
        examples: Sequence[ModelExample],
        *,
        extra_training: Sequence[ModelExample] = (),
    ) -> None:
        if any(
            value.split in {"barrier-heldout", "barrier-stress"}
            for value in (*examples, *extra_training)
        ):
            raise ValueError("held-out branches cannot fit the event model")
        fit_seeds, calibration_seeds = self._seed_partition(
            examples, self.config.calibration_fraction
        )
        fitting = [
            value for value in examples if value.seed in fit_seeds
        ] + list(extra_training)
        calibration = [
            value for value in examples if value.seed in calibration_seeds
        ]
        x = np.asarray([value.features for value in fitting], dtype=np.float64)
        y = np.asarray([value.targets for value in fitting], dtype=np.float64)
        self.center = np.median(x, axis=0)
        q75, q25 = np.percentile(x, [75, 25], axis=0)
        self.scale = np.where(q75 - q25 > 1e-9, q75 - q25, 1.0)
        self.features = (x - self.center) / self.scale
        self.targets = y
        self.fit_seeds = tuple(sorted(fit_seeds))
        self.calibration_seeds = tuple(sorted(calibration_seeds))
        residuals = []
        overprediction_by_seed: dict[int, list[float]] = {}
        for example in calibration:
            prediction = self._raw(np.asarray(example.features, dtype=np.float64))
            residuals.append(np.asarray(example.targets) - prediction[0])
            overprediction_by_seed.setdefault(example.seed, []).append(
                float(
                    prediction[0][TARGET_INDEX["delta_b2"]]
                    - example.targets[TARGET_INDEX["delta_b2"]]
                )
            )
        self.calibration_residuals = np.asarray(residuals, dtype=np.float64)
        self.calibration_residual_seeds = tuple(
            int(example.seed) for example in calibration
        )
        if not len(residuals):
            raise ValueError("event model calibration partition is empty")
        self.lower_offsets = np.min(self.calibration_residuals, axis=0)
        self.conformal_records = tuple(
            {
                "seed": int(seed),
                "candidate_count": len(overprediction_by_seed[seed]),
                "r_j": float(max(overprediction_by_seed[seed])),
            }
            for seed in sorted(overprediction_by_seed)
        )
        episode_residuals = sorted(
            float(value["r_j"]) for value in self.conformal_records
        )
        rank = math.ceil(
            (len(episode_residuals) + 1)
            * (1.0 - self.config.conformal_alpha)
        )
        self.conformal_rank = rank
        self.conformal_q = (
            episode_residuals[rank - 1]
            if rank <= len(episode_residuals)
            else math.inf
        )

    def fit_provisional(self, examples: Sequence[ModelExample]) -> None:
        """Fit only the predetermined training half for DAgger collection.

        No target from the reserved whole-seed conformal half is evaluated or
        used until the final one-shot ``fit`` after DAgger data are frozen.
        """

        if any(
            value.split in {"barrier-heldout", "barrier-stress"}
            for value in examples
        ):
            raise ValueError("held-out branches cannot fit the event model")
        fit_seeds, calibration_seeds = self._seed_partition(
            examples, self.config.calibration_fraction
        )
        fitting = [value for value in examples if value.seed in fit_seeds]
        x = np.asarray([value.features for value in fitting], dtype=np.float64)
        self.center = np.median(x, axis=0)
        q75, q25 = np.percentile(x, [75, 25], axis=0)
        self.scale = np.where(q75 - q25 > 1e-9, q75 - q25, 1.0)
        self.features = (x - self.center) / self.scale
        self.targets = np.asarray(
            [value.targets for value in fitting], dtype=np.float64
        )
        self.fit_seeds = tuple(sorted(fit_seeds))
        self.calibration_seeds = tuple(sorted(calibration_seeds))
        self.lower_offsets = np.zeros(len(TARGET_NAMES), dtype=np.float64)
        self.conformal_q = 0.0
        self.conformal_rank = 0
        self.conformal_records = ()
        self.calibration_residual_seeds = ()
        self.calibration_residuals = np.empty(
            (0, len(TARGET_NAMES)), dtype=np.float64
        )

    def _raw(
        self, features: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("event model is not fitted")
        normalized = (features - self.center) / self.scale
        distances = np.sqrt(np.mean((self.features - normalized) ** 2, axis=1))
        count = min(self.config.neighbor_count, len(distances))
        indices = np.argpartition(distances, count - 1)[:count]
        indices = indices[np.argsort(distances[indices], kind="stable")]
        local_distance = distances[indices]
        weights = 1.0 / np.maximum(local_distance, 1e-6)
        weights /= weights.sum()
        local = self.targets[indices]
        mean = np.sum(local * weights[:, None], axis=0)
        variance = np.sum(
            weights[:, None] * (local - mean[None, :]) ** 2, axis=0
        )
        return mean, np.sqrt(variance), indices, local_distance, local

    @staticmethod
    def _wilson_upper(successes: float, total: int, z: float = 1.96) -> float:
        if total < 1:
            return 1.0
        probability = successes / total
        denominator = 1.0 + z * z / total
        center = probability + z * z / (2.0 * total)
        radius = z * math.sqrt(
            probability * (1.0 - probability) / total
            + z * z / (4.0 * total * total)
        )
        return min(1.0, (center + radius) / denominator)

    def predict(self, features: Sequence[float]) -> ModelPrediction:
        mean, local_std, indices, distances, local = self._raw(
            np.asarray(features, dtype=np.float64)
        )
        lower = mean + self.lower_offsets
        lower[TARGET_INDEX["delta_b2"]] = (
            mean[TARGET_INDEX["delta_b2"]] - self.conformal_q
        )
        catastrophic = local[:, TARGET_INDEX["catastrophic"]]
        probability = float(np.mean(catastrophic))
        upper = self._wilson_upper(float(catastrophic.sum()), len(catastrophic))
        return ModelPrediction(
            tuple(float(value) for value in mean),
            tuple(float(value) for value in lower),
            tuple(float(value) for value in local_std),
            tuple(int(value) for value in indices),
            tuple(float(value) for value in distances),
            probability,
            upper,
            bool(
                np.all(
                    local[:, TARGET_INDEX["second_renewal_reached"]] >= 1.0
                )
            ),
        )

    def certify(self, features: Sequence[float]) -> tuple[
        ModelPrediction, CandidateCertificate
    ]:
        prediction = self.predict(features)
        reasons = []
        checks = {
            "two_renewal_neighbor_support": prediction.all_neighbors_two_renewal,
            "absolute_solvency_lower_bound": (
                prediction.lower_value("b2") >= 0.0
            ),
            "relative_solvency_lower_bound": (
                prediction.lower_value("delta_b2") >= 0.0
            ),
            "relative_final_gauge_lower_bound": (
                prediction.lower_value("delta_final_gauge") >= 0.0
            ),
            "catastrophe_risk_upper_bound": (
                prediction.catastrophic_upper <= self.config.risk_upper_limit
            ),
            "positive_score_advantage": (
                prediction.value("delta_score_gain")
                > self.config.minimum_score_advantage
            ),
        }
        reasons.extend(name for name, passed in checks.items() if not passed)
        certificate = CandidateCertificate(
            not reasons,
            tuple(reasons),
            prediction.lower_value("b2"),
            prediction.lower_value("delta_b2"),
            prediction.lower_value("delta_final_gauge"),
            prediction.value("delta_score_gain"),
            prediction.catastrophic_probability,
            prediction.catastrophic_upper,
            prediction.all_neighbors_two_renewal,
        )
        return prediction, certificate

    def manifest(self) -> dict[str, object]:
        if not self.fitted:
            raise RuntimeError("cannot serialize an unfitted event model")
        return {
            "version": EVENT_MPC_VERSION,
            "config": self.config.manifest(),
            "feature_names": list(FEATURE_NAMES),
            "target_names": list(TARGET_NAMES),
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "features": self.features.tolist(),
            "targets": self.targets.tolist(),
            "lower_offsets": self.lower_offsets.tolist(),
            "fit_seeds": list(self.fit_seeds),
            "calibration_seeds": list(self.calibration_seeds),
            "conformal_q": (
                self.conformal_q
                if math.isfinite(self.conformal_q)
                else "inf"
            ),
            "conformal_n": len(self.conformal_records),
            "conformal_rank": self.conformal_rank,
            "conformal_records": list(self.conformal_records),
            "ordered_episode_residuals": sorted(
                float(value["r_j"]) for value in self.conformal_records
            ),
            "calibration_residual_seeds": list(
                self.calibration_residual_seeds
            ),
            "calibration_residuals": self.calibration_residuals.tolist(),
            "calibration_residual_sha256": _canonical_sha256(
                {
                    "seeds": list(self.calibration_residual_seeds),
                    "residuals": self.calibration_residuals.tolist(),
                }
            ),
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> KNNEventWorldModel:
        if value.get("version") != EVENT_MPC_VERSION:
            raise ValueError("event model version changed")
        config_value = value.get("config")
        if not isinstance(config_value, Mapping):
            raise ValueError("event model config is missing")
        if config_value.get("version") != EVENT_MPC_VERSION:
            raise ValueError("event model config version changed")
        config_arguments = {
            key: config_value[key]
            for key in EventMPCConfig.__dataclass_fields__
        }
        config_arguments["continuation_checkpoint_ticks"] = tuple(
            int(item)
            for item in config_arguments["continuation_checkpoint_ticks"]
        )
        config = EventMPCConfig(**config_arguments)
        model = cls(config)
        if tuple(value.get("feature_names", ())) != FEATURE_NAMES:
            raise ValueError("event model feature identity changed")
        if tuple(value.get("target_names", ())) != TARGET_NAMES:
            raise ValueError("event model target identity changed")
        model.center = np.asarray(value["center"], dtype=np.float64)
        model.scale = np.asarray(value["scale"], dtype=np.float64)
        model.features = np.asarray(value["features"], dtype=np.float64)
        model.targets = np.asarray(value["targets"], dtype=np.float64)
        model.lower_offsets = np.asarray(
            value["lower_offsets"], dtype=np.float64
        )
        model.fit_seeds = tuple(int(item) for item in value["fit_seeds"])
        model.calibration_seeds = tuple(
            int(item) for item in value["calibration_seeds"]
        )
        model.conformal_q = float(value["conformal_q"])
        model.conformal_rank = int(value["conformal_rank"])
        model.conformal_records = tuple(
            {
                "seed": int(item["seed"]),
                "candidate_count": int(item["candidate_count"]),
                "r_j": float(item["r_j"]),
            }
            for item in value["conformal_records"]
        )
        model.calibration_residual_seeds = tuple(
            int(item) for item in value["calibration_residual_seeds"]
        )
        if (
            model.center.shape != (len(FEATURE_NAMES),)
            or model.scale.shape != (len(FEATURE_NAMES),)
            or model.features.ndim != 2
            or model.features.shape[1:] != (len(FEATURE_NAMES),)
            or model.targets.shape
            != (model.features.shape[0], len(TARGET_NAMES))
            or model.lower_offsets.shape != (len(TARGET_NAMES),)
            or not all(
                np.all(np.isfinite(array))
                for array in (
                    model.center,
                    model.scale,
                    model.features,
                    model.targets,
                    model.lower_offsets,
                )
            )
            or np.any(model.scale <= 0)
        ):
            raise ValueError("event model array state changed")
        if (
            len(set(model.fit_seeds)) != len(model.fit_seeds)
            or len(set(model.calibration_seeds))
            != len(model.calibration_seeds)
            or set(model.fit_seeds) & set(model.calibration_seeds)
        ):
            raise ValueError("whole-seed model partition changed")
        if int(value["conformal_n"]) != len(model.conformal_records):
            raise ValueError("conformal episode count changed")
        record_seeds = tuple(
            int(item["seed"]) for item in model.conformal_records
        )
        if (
            len(set(record_seeds)) != len(record_seeds)
            or any(
                int(item["candidate_count"]) < 1
                or not math.isfinite(float(item["r_j"]))
                for item in model.conformal_records
            )
        ):
            raise ValueError("conformal episode records changed")
        ordered = sorted(
            float(item["r_j"]) for item in model.conformal_records
        )
        if list(value["ordered_episode_residuals"]) != ordered:
            raise ValueError("ordered conformal residuals changed")
        model.calibration_residuals = np.asarray(
            value["calibration_residuals"], dtype=np.float64
        )
        if model.calibration_residuals.size == 0:
            model.calibration_residuals = model.calibration_residuals.reshape(
                0, len(TARGET_NAMES)
            )
        if (
            model.calibration_residuals.ndim != 2
            or model.calibration_residuals.shape[1:]
            != (len(TARGET_NAMES),)
            or not np.all(np.isfinite(model.calibration_residuals))
            or len(model.calibration_residual_seeds)
            != model.calibration_residuals.shape[0]
            or sum(
                int(item["candidate_count"])
                for item in model.conformal_records
            )
            != model.calibration_residuals.shape[0]
        ):
            raise ValueError("conformal candidate evidence changed")
        if value.get("calibration_residual_sha256") != _canonical_sha256(
            {
                "seeds": list(model.calibration_residual_seeds),
                "residuals": model.calibration_residuals.tolist(),
            }
        ):
            raise ValueError("calibration residual identity changed")
        count = len(model.conformal_records)
        if count == 0:
            if (
                model.conformal_rank != 0
                or model.conformal_q != 0.0
                or model.calibration_residuals.shape[0] != 0
                or model.calibration_residual_seeds
                or np.any(model.lower_offsets != 0.0)
            ):
                raise ValueError("provisional conformal state changed")
        else:
            if set(record_seeds) != set(model.calibration_seeds):
                raise ValueError("conformal whole-seed evidence changed")
            expected_offsets = np.min(
                model.calibration_residuals, axis=0
            )
            if not np.array_equal(model.lower_offsets, expected_offsets):
                raise ValueError("calibration lower offsets changed")
            expected_records = tuple(
                {
                    "seed": seed,
                    "candidate_count": sum(
                        row_seed == seed
                        for row_seed in model.calibration_residual_seeds
                    ),
                    "r_j": max(
                        -float(
                            model.calibration_residuals[index][
                                TARGET_INDEX["delta_b2"]
                            ]
                        )
                        for index, row_seed in enumerate(
                            model.calibration_residual_seeds
                        )
                        if row_seed == seed
                    ),
                }
                for seed in sorted(set(model.calibration_residual_seeds))
            )
            if expected_records != model.conformal_records:
                raise ValueError("conformal residual binding changed")
            expected_rank = math.ceil(
                (count + 1) * (1.0 - model.config.conformal_alpha)
            )
            if model.conformal_rank != expected_rank:
                raise ValueError("conformal rank changed")
            if model.conformal_rank <= count:
                expected_q = ordered[model.conformal_rank - 1]
                if model.conformal_q != expected_q:
                    raise ValueError("conformal threshold changed")
            elif not (
                math.isinf(model.conformal_q)
                and model.conformal_q > 0
            ):
                raise ValueError("conformal threshold changed")
        return model


class ModelBarrierPolicy:
    """Frozen-v5 cadence with at most one certified override per episode."""

    def __init__(
        self,
        base_policy: object,
        candidate_provider: Callable[
            [Mapping[str, Any], SteeringDecision], Sequence[Any]
        ],
        commit: Callable[
            [object, Mapping[str, Any], SteeringDecision, SteeringDecision], bool
        ],
        model: KNNEventWorldModel,
        *,
        query_stride_shots: int = 1,
        maximum_overrides: int | None = None,
        proposal_audit_hook: Callable[
            [
                Mapping[str, Any],
                object,
                SteeringDecision,
                Sequence[Any],
            ],
            None,
        ]
        | None = None,
        audit_hook: Callable[
            [
                Mapping[str, Any],
                object,
                SteeringDecision,
                Any,
                CandidateCertificate,
            ],
            object,
        ]
        | None = None,
        audit_commit_hook: Callable[[object], None] | None = None,
    ) -> None:
        if query_stride_shots < 1:
            raise ValueError("query stride must be positive")
        if maximum_overrides is not None and maximum_overrides < 1:
            raise ValueError("override budget must be positive or None")
        self.base_policy = base_policy
        self.candidate_provider = candidate_provider
        self.commit = commit
        self.model = model
        self.query_stride_shots = query_stride_shots
        self.maximum_overrides = maximum_overrides
        self.proposal_audit_hook = proposal_audit_hook
        self.audit_hook = audit_hook
        self.audit_commit_hook = audit_commit_hook
        self.counts: Counter[str] = Counter()
        self.query_ticks: list[int] = []
        self.override_ticks: list[int] = []
        self._overrode = False

    def reset(self, seed: int = 0) -> None:
        getattr(self.base_policy, "reset")(seed)
        self.counts.clear()
        self.query_ticks.clear()
        self.override_ticks.clear()
        self._overrode = False

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = getattr(self.base_policy, "predict")(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("barrier base policy returned a non-decision")
        if not incumbent.is_shot:
            return incumbent
        self.counts["seen_shots"] += 1
        if self._overrode:
            self.counts["permanent_episode_abstentions"] += 1
            return incumbent
        if (self.counts["seen_shots"] - 1) % self.query_stride_shots:
            self.counts["stride_abstentions"] += 1
            return incumbent
        budget_usage = (
            self.counts["audited_proposals"]
            if self.audit_hook is not None
            else self.counts["overrides"]
        )
        if self.maximum_overrides is not None and budget_usage >= (
            self.maximum_overrides
        ):
            self.counts["query_cap_abstentions"] += 1
            return incumbent
        self.counts["queries"] += 1
        query_tick = _plain_int(observation.get("tick"), "query tick")
        self.query_ticks.append(query_tick)
        self.counts[f"query_tick_bucket/{query_tick // 1_000}"] += 1
        try:
            candidates = tuple(self.candidate_provider(observation, incumbent))
        except ValueError:
            self.counts["shortlist_abstentions"] += 1
            return incumbent
        if not candidates or candidates[0].decision is not incumbent:
            raise RuntimeError(
                "model barrier candidate zero is not frozen-v5"
            )
        self.counts["candidates"] += len(candidates)
        alternatives = []
        incumbent_actions = incumbent.primitive_actions()
        for candidate in candidates[1:]:
            if candidate.decision.primitive_actions() == incumbent_actions:
                self.counts["action_tie_abstentions"] += 1
                continue
            alternatives.append(candidate)
        if alternatives:
            self.counts["eligible_decision_states"] += 1
            if self.proposal_audit_hook is not None:
                self.proposal_audit_hook(
                    observation,
                    self.base_policy,
                    incumbent,
                    candidates,
                )
                self.counts["proposal_audit_calls"] += 1
        eligible = []
        for candidate in alternatives:
            _prediction, certificate = self.model.certify(
                candidate_features(observation, candidate)
            )
            self.counts["certified_candidates"] += int(certificate.certified)
            for reason in certificate.reasons:
                self.counts[f"rejected/{reason}"] += 1
            if certificate.certified:
                eligible.append((certificate, candidate))
        if not eligible:
            self.counts["barrier_abstentions"] += 1
            return incumbent
        objective = lambda item: (
                item[0].predicted_score_advantage,
                item[0].lower_delta_b2,
                item[0].lower_b2,
                item[0].lower_delta_final_gauge,
            )
        best = max(objective(item) for item in eligible)
        winners = [item for item in eligible if objective(item) == best]
        if len(winners) != 1:
            self.counts["top_tie_abstentions"] += 1
            return incumbent
        certificate, selected = winners[0]
        audit_token: object | None = None
        if self.audit_hook is not None:
            self.counts["audited_proposals"] += 1
            audit_token = self.audit_hook(
                observation,
                self.base_policy,
                incumbent,
                selected,
                certificate,
            )
        if not self.commit(
            self.base_policy, observation, incumbent, selected.decision
        ):
            self.counts["progress_rebind_abstentions"] += 1
            return incumbent
        if self.audit_commit_hook is not None:
            self.audit_commit_hook(audit_token)
        self._overrode = True
        self.counts["overrides"] += 1
        self.counts["pair_corrections"] += int(selected.pair_ordinal != 0)
        self.counts["geometry_corrections"] += int(
            selected.geometry_ordinal != 0
        )
        self.override_ticks.append(
            _plain_int(observation.get("tick"), "override tick")
        )
        self.counts[
            f"override_tick_bucket/{self.override_ticks[-1] // 1_000}"
        ] += 1
        return selected.decision

    def statistics(self) -> dict[str, int | float]:
        return {
            **dict(sorted(self.counts.items())),
            "first_query_tick": min(self.query_ticks) if self.query_ticks else -1,
            "last_query_tick": max(self.query_ticks) if self.query_ticks else -1,
            "first_override_tick": (
                min(self.override_ticks) if self.override_ticks else -1
            ),
            "last_override_tick": (
                max(self.override_ticks) if self.override_ticks else -1
            ),
        }


__all__ = [
    "CashflowReplay",
    "CandidateCertificate",
    "CandidateDebtLedger",
    "EVENT_MPC_VERSION",
    "EventMPCConfig",
    "ExactEventOutcome",
    "ExactEventPlanner",
    "ExactSearchResult",
    "FEATURE_NAMES",
    "KNNEventWorldModel",
    "ModelBarrierPolicy",
    "ModelExample",
    "ModelPrediction",
    "RENEWABLE_DETAILS",
    "RenewableCycle",
    "TARGET_NAMES",
    "candidate_features",
    "replay_two_renewal_cashflow",
    "search_examples",
]
