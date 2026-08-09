"""Causal shot accounting from steering decisions and public events only."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import groupby
from numbers import Integral
from typing import Any

from irisu_env import EventKind
from irisu_rl.actions import SemanticActionKind

from .steering import SteeringDecision


CAUSAL_CONTROL_VERSION = "r3d-causal-control-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _integer(value: object, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"event {name} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"event {name} must be at least {minimum}")
    return result


def _kind(value: Mapping[str, Any]) -> int:
    raw = value.get("kind")
    name = value.get("kind_name")
    parsed_name: int | None = None
    if name is not None:
        if not isinstance(name, str):
            raise TypeError("event kind_name must be text")
        try:
            parsed_name = int(EventKind[name.upper()])
        except KeyError as exc:
            raise ValueError("event kind_name is unknown") from exc
    if raw is None:
        if parsed_name is None:
            raise ValueError("event requires kind or kind_name")
        return parsed_name
    parsed = _integer(raw, "kind", minimum=0)
    try:
        EventKind(parsed)
    except ValueError as exc:
        raise ValueError("event kind is unknown") from exc
    if parsed_name is not None and parsed_name != parsed:
        raise ValueError("event kind and kind_name disagree")
    return parsed


@dataclass(frozen=True, slots=True)
class _PublicEvent:
    tick: int
    kind: int
    a: int
    b: int
    value: int
    sequence: int | None
    ordinal: int

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], ordinal: int
    ) -> _PublicEvent:
        if not isinstance(value, Mapping):
            raise TypeError("events must be public event mappings")
        sequence = value.get("sequence")
        return cls(
            tick=_integer(value.get("tick", 0), "tick", minimum=0),
            kind=_kind(value),
            a=_integer(value.get("a", 0), "a", minimum=0),
            b=_integer(value.get("b", 0), "b", minimum=0),
            value=_integer(value.get("value", 0), "value"),
            sequence=(
                None
                if sequence is None
                else _integer(sequence, "sequence", minimum=0)
            ),
            ordinal=ordinal,
        )

    @property
    def order(self) -> tuple[int, int, int]:
        return (
            self.tick,
            self.sequence if self.sequence is not None else self.ordinal,
            self.ordinal,
        )


@dataclass(frozen=True, slots=True)
class CausalShotOutcome:
    """Auditable public evidence associated with one fired projectile."""

    binding_index: int
    projectile_id: int
    fired_tick: int
    fired_sequence: int | None
    shot_strength: str
    intent: str
    source_body_id: int
    destination_body_id: int | None
    destination_chain_id: int | None
    first_hit_body_id: int | None
    first_hit_tick: int | None
    first_hit_was_intended_source: bool | None
    any_intended_source_hit: bool
    intended_source_hit_events: int
    intended_pair_join_events: int
    intended_pair_confirmation_events: int
    intended_pair_clear_events: int
    projectile_ejection_events: int
    source_ejection_events: int
    source_rotten_events: int
    destination_rotten_events: int
    invalid_action_events: int

    def manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CausalControlCounters:
    events_consumed: int
    shot_decisions: int
    wait_decisions: int
    shot_events: int
    bound_shots: int
    unbound_shot_events: int
    shot_decisions_without_fire: int
    invalid_action_events: int
    invalid_shot_decisions: int
    projectile_hit_events: int
    bound_projectile_hit_events: int
    orphan_projectile_hit_events: int
    shots_with_first_hit: int
    first_hits_on_intended_source: int
    shots_with_any_intended_source_hit: int
    intended_source_hit_events: int
    chain_joined_events: int
    intended_pair_join_events: int
    shots_with_intended_pair_join: int
    confirmed_events: int
    intended_pair_confirmation_events: int
    shots_with_intended_pair_confirmation: int
    cleared_events: int
    intended_pair_clear_events: int
    shots_with_intended_pair_clear: int
    ejected_events: int
    projectile_ejections: int
    source_ejections: int
    other_ejections: int
    rotten_events: int
    source_rotten_events: int
    destination_rotten_events: int
    other_rotten_events: int

    def manifest(self) -> dict[str, int]:
        return asdict(self)


@dataclass(slots=True)
class _ShotState:
    binding_index: int
    projectile_id: int
    fired_tick: int
    fired_sequence: int | None
    shot_strength: str
    intent: str
    source_body_id: int
    destination_body_id: int | None
    destination_chain_id: int | None
    first_hit_body_id: int | None = None
    first_hit_tick: int | None = None
    first_hit_was_intended_source: bool | None = None
    intended_source_hit_events: int = 0
    last_source_hit_order: tuple[int, int, int] | None = None
    intended_pair_join_events: int = 0
    intended_pair_confirmation_events: int = 0
    last_confirmation_order: tuple[int, int, int] | None = None
    intended_pair_clear_events: int = 0
    projectile_ejection_events: int = 0
    source_ejection_events: int = 0
    source_rotten_events: int = 0
    destination_rotten_events: int = 0
    invalid_action_events: int = 0

    def outcome(self) -> CausalShotOutcome:
        return CausalShotOutcome(
            binding_index=self.binding_index,
            projectile_id=self.projectile_id,
            fired_tick=self.fired_tick,
            fired_sequence=self.fired_sequence,
            shot_strength=self.shot_strength,
            intent=self.intent,
            source_body_id=self.source_body_id,
            destination_body_id=self.destination_body_id,
            destination_chain_id=self.destination_chain_id,
            first_hit_body_id=self.first_hit_body_id,
            first_hit_tick=self.first_hit_tick,
            first_hit_was_intended_source=self.first_hit_was_intended_source,
            any_intended_source_hit=self.intended_source_hit_events > 0,
            intended_source_hit_events=self.intended_source_hit_events,
            intended_pair_join_events=self.intended_pair_join_events,
            intended_pair_confirmation_events=(
                self.intended_pair_confirmation_events
            ),
            intended_pair_clear_events=self.intended_pair_clear_events,
            projectile_ejection_events=self.projectile_ejection_events,
            source_ejection_events=self.source_ejection_events,
            source_rotten_events=self.source_rotten_events,
            destination_rotten_events=self.destination_rotten_events,
            invalid_action_events=self.invalid_action_events,
        )


class CausalShotTracker:
    """Attribute public outcomes without snapshots, hidden state, or lookahead.

    Events are processed by tick. Within one public tick, projectile hits are
    resolved before pair outcomes because native contact callback order is not
    a physical sub-tick ordering. Events from later ticks are never used.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Start an empty episode and permit public tick/sequence restart."""

        self._shots: dict[int, _ShotState] = {}
        self._shot_order: list[int] = []
        self._seen_sequences: set[int] = set()
        self._last_tick: int | None = None
        self._next_event_ordinal = 0
        self._shot_decisions = 0
        self._wait_decisions = 0
        self._shot_events = 0
        self._unbound_shot_events = 0
        self._shot_decisions_without_fire = 0
        self._invalid_action_events = 0
        self._invalid_shot_decisions = 0
        self._events_consumed = 0
        self._projectile_hit_events = 0
        self._bound_projectile_hit_events = 0
        self._orphan_projectile_hit_events = 0
        self._chain_joined_events = 0
        self._confirmed_events = 0
        self._cleared_events = 0
        self._ejected_events = 0
        self._projectile_ejections = 0
        self._source_ejections = 0
        self._other_ejections = 0
        self._rotten_events = 0
        self._source_rotten_events = 0
        self._destination_rotten_events = 0
        self._other_rotten_events = 0

    @property
    def outcomes(self) -> tuple[CausalShotOutcome, ...]:
        return tuple(self._shots[value].outcome() for value in self._shot_order)

    @property
    def counters(self) -> CausalControlCounters:
        outcomes = self.outcomes
        return CausalControlCounters(
            events_consumed=self._events_consumed,
            shot_decisions=self._shot_decisions,
            wait_decisions=self._wait_decisions,
            shot_events=self._shot_events,
            bound_shots=len(outcomes),
            unbound_shot_events=self._unbound_shot_events,
            shot_decisions_without_fire=self._shot_decisions_without_fire,
            invalid_action_events=self._invalid_action_events,
            invalid_shot_decisions=self._invalid_shot_decisions,
            projectile_hit_events=self._projectile_hit_events,
            bound_projectile_hit_events=self._bound_projectile_hit_events,
            orphan_projectile_hit_events=self._orphan_projectile_hit_events,
            shots_with_first_hit=sum(
                value.first_hit_body_id is not None for value in outcomes
            ),
            first_hits_on_intended_source=sum(
                value.first_hit_was_intended_source is True for value in outcomes
            ),
            shots_with_any_intended_source_hit=sum(
                value.any_intended_source_hit for value in outcomes
            ),
            intended_source_hit_events=sum(
                value.intended_source_hit_events for value in outcomes
            ),
            chain_joined_events=self._chain_joined_events,
            intended_pair_join_events=sum(
                value.intended_pair_join_events for value in outcomes
            ),
            shots_with_intended_pair_join=sum(
                value.intended_pair_join_events > 0 for value in outcomes
            ),
            confirmed_events=self._confirmed_events,
            intended_pair_confirmation_events=sum(
                value.intended_pair_confirmation_events for value in outcomes
            ),
            shots_with_intended_pair_confirmation=sum(
                value.intended_pair_confirmation_events > 0 for value in outcomes
            ),
            cleared_events=self._cleared_events,
            intended_pair_clear_events=sum(
                value.intended_pair_clear_events for value in outcomes
            ),
            shots_with_intended_pair_clear=sum(
                value.intended_pair_clear_events > 0 for value in outcomes
            ),
            ejected_events=self._ejected_events,
            projectile_ejections=self._projectile_ejections,
            source_ejections=self._source_ejections,
            other_ejections=self._other_ejections,
            rotten_events=self._rotten_events,
            source_rotten_events=self._source_rotten_events,
            destination_rotten_events=self._destination_rotten_events,
            other_rotten_events=self._other_rotten_events,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "version": CAUSAL_CONTROL_VERSION,
            "evidence": "steering_decisions_and_public_events_only",
            "same_tick_rule": "hits_before_pair_outcomes",
            "counters": self.counters.manifest(),
            "shots": [value.manifest() for value in self.outcomes],
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.manifest())).hexdigest()

    def _pair_candidate(
        self, event: _PublicEvent, *, require_confirmation: bool = False
    ) -> _ShotState | None:
        candidates: list[_ShotState] = []
        for state in self._shots.values():
            destination = state.destination_body_id
            if (
                destination is None
                or state.fired_tick > event.tick
                or state.last_source_hit_order is None
            ):
                continue
            exact_pair = (
                event.a == state.source_body_id and event.b == destination
            ) or (event.a == destination and event.b == state.source_body_id)
            clear_member = event.b == 0 and event.a in (
                state.source_body_id,
                destination,
            )
            if not exact_pair and not (
                require_confirmation
                and clear_member
                and state.last_confirmation_order is not None
                and state.last_confirmation_order[0] <= event.tick
            ):
                continue
            if require_confirmation and state.last_confirmation_order is None:
                continue
            candidates.append(state)
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda state: (
                state.last_confirmation_order
                if require_confirmation and state.last_confirmation_order is not None
                else state.last_source_hit_order,
                state.binding_index,
            ),
        )

    def _latest_body_shot(
        self, body_id: int, tick: int, *, destination: bool = False
    ) -> _ShotState | None:
        candidates = [
            state
            for state in self._shots.values()
            if state.fired_tick <= tick
            and (
                state.destination_body_id == body_id
                if destination
                else state.source_body_id == body_id
            )
        ]
        return max(candidates, key=lambda value: value.binding_index, default=None)

    def _bind(
        self, decision: SteeringDecision | None, shot_events: list[_PublicEvent]
    ) -> tuple[_ShotState, ...]:
        if decision is not None and not isinstance(decision, SteeringDecision):
            raise TypeError("decision must be a SteeringDecision or None")
        if decision is None:
            self._unbound_shot_events += len(shot_events)
            return ()
        if decision.is_shot:
            self._shot_decisions += 1
        else:
            self._wait_decisions += 1
        if not decision.is_shot:
            self._unbound_shot_events += len(shot_events)
            return ()
        if len(shot_events) > 1:
            raise ValueError("one steering decision emitted multiple projectiles")
        if not shot_events:
            self._shot_decisions_without_fire += 1
            return ()
        event = shot_events[0]
        expected = (
            0
            if SemanticActionKind(decision.action.kind)
            is SemanticActionKind.FIRE_WEAK
            else 1
        )
        if event.value != expected:
            raise ValueError("SHOT_FIRED strength disagrees with steering decision")
        if event.a == 0:
            raise ValueError("SHOT_FIRED has a zero projectile ID")
        if event.a in self._shots:
            raise ValueError("projectile ID was already bound")
        state = _ShotState(
            binding_index=len(self._shot_order),
            projectile_id=event.a,
            fired_tick=event.tick,
            fired_sequence=event.sequence,
            shot_strength="weak" if event.value == 0 else "strong",
            intent=decision.intent.value,
            source_body_id=int(decision.source_body_id),
            destination_body_id=decision.destination_body_id,
            destination_chain_id=decision.destination_chain_id,
        )
        self._shots[event.a] = state
        self._shot_order.append(event.a)
        return (state,)

    def consume(
        self,
        events: Iterable[Mapping[str, Any]],
        *,
        decision: SteeringDecision | None = None,
    ) -> tuple[CausalShotOutcome, ...]:
        """Consume one public transition and return shots bound by it."""

        raw_events = tuple(events)
        normalized = [
            _PublicEvent.from_mapping(value, self._next_event_ordinal + index)
            for index, value in enumerate(raw_events)
        ]
        normalized.sort(key=lambda value: value.order)
        if (
            normalized
            and self._last_tick is not None
            and normalized[0].tick < self._last_tick
        ):
            raise ValueError("event ticks moved backward; reset the tracker")
        sequences = [
            event.sequence for event in normalized if event.sequence is not None
        ]
        if len(sequences) != len(set(sequences)) or any(
            value in self._seen_sequences for value in sequences
        ):
            raise ValueError("event sequence was repeated; reset or de-duplicate input")
        shot_events = [
            event
            for event in normalized
            if event.kind == int(EventKind.SHOT_FIRED)
        ]
        if decision is not None and not isinstance(decision, SteeringDecision):
            raise TypeError("decision must be a SteeringDecision or None")
        if decision is not None and decision.is_shot and len(shot_events) > 1:
            raise ValueError("one steering decision emitted multiple projectiles")
        projectile_ids = [event.a for event in shot_events]
        if any(value == 0 for value in projectile_ids):
            raise ValueError("SHOT_FIRED has a zero projectile ID")
        if len(projectile_ids) != len(set(projectile_ids)) or any(
            value in self._shots for value in projectile_ids
        ):
            raise ValueError("projectile ID was already bound")
        if any(event.value not in (0, 1) for event in shot_events):
            raise ValueError("SHOT_FIRED has an unknown strength")
        if decision is not None and decision.is_shot and shot_events:
            expected = (
                0
                if SemanticActionKind(decision.action.kind)
                is SemanticActionKind.FIRE_WEAK
                else 1
            )
            if shot_events[0].value != expected:
                raise ValueError(
                    "SHOT_FIRED strength disagrees with steering decision"
                )

        self._next_event_ordinal += len(normalized)
        self._events_consumed += len(normalized)
        self._shot_events += len(shot_events)
        self._seen_sequences.update(sequences)
        if normalized:
            self._last_tick = normalized[-1].tick
        bound = self._bind(decision, shot_events)

        invalid = sum(
            event.kind == int(EventKind.INVALID_ACTION) for event in normalized
        )
        self._invalid_action_events += invalid
        if decision is not None and decision.is_shot and invalid:
            self._invalid_shot_decisions += 1
            for state in bound:
                state.invalid_action_events += invalid

        for tick, tick_values in groupby(normalized, key=lambda value: value.tick):
            current = list(tick_values)
            for event in current:
                if event.kind != int(EventKind.PROJECTILE_HIT):
                    continue
                self._projectile_hit_events += 1
                state = self._shots.get(event.a)
                if state is None or state.fired_tick > tick:
                    self._orphan_projectile_hit_events += 1
                    continue
                self._bound_projectile_hit_events += 1
                if state.first_hit_body_id is None:
                    state.first_hit_body_id = event.b
                    state.first_hit_tick = event.tick
                    state.first_hit_was_intended_source = (
                        event.b == state.source_body_id
                    )
                if event.b == state.source_body_id:
                    state.intended_source_hit_events += 1
                    state.last_source_hit_order = event.order

            for event in current:
                if event.kind == int(EventKind.CHAIN_JOINED):
                    self._chain_joined_events += 1
                    state = self._pair_candidate(event)
                    if state is not None:
                        state.intended_pair_join_events += 1
                elif event.kind == int(EventKind.CONFIRMED):
                    self._confirmed_events += 1
                    state = self._pair_candidate(event)
                    if state is not None:
                        state.intended_pair_confirmation_events += 1
                        state.last_confirmation_order = event.order

            for event in current:
                if event.kind == int(EventKind.CLEARED):
                    self._cleared_events += 1
                    state = self._pair_candidate(event, require_confirmation=True)
                    if state is not None:
                        state.intended_pair_clear_events += 1
                elif event.kind == int(EventKind.EJECTED):
                    self._ejected_events += 1
                    projectile = self._shots.get(event.a)
                    source = self._latest_body_shot(event.a, tick)
                    if projectile is not None and projectile.fired_tick <= tick:
                        self._projectile_ejections += 1
                        projectile.projectile_ejection_events += 1
                    elif source is not None:
                        self._source_ejections += 1
                        source.source_ejection_events += 1
                    else:
                        self._other_ejections += 1
                elif event.kind == int(EventKind.ROTTEN):
                    self._rotten_events += 1
                    source = self._latest_body_shot(event.a, tick)
                    destination = self._latest_body_shot(
                        event.a, tick, destination=True
                    )
                    if source is not None:
                        self._source_rotten_events += 1
                        source.source_rotten_events += 1
                    elif destination is not None:
                        self._destination_rotten_events += 1
                    else:
                        self._other_rotten_events += 1
                    if destination is not None:
                        destination.destination_rotten_events += 1

        return tuple(state.outcome() for state in bound)


__all__ = [
    "CAUSAL_CONTROL_VERSION",
    "CausalControlCounters",
    "CausalShotOutcome",
    "CausalShotTracker",
]
