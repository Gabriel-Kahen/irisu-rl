"""Development-only renewable-reserve runway teacher.

The teacher branches only the fixed R3e geometry vocabulary on the trusted
portable simulator.  It is intentionally nondeployable: future public states
from cloned branches are labels, never policy inputs.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from irisu_env import EventKind
from irisu_rl.actions import ActionSpec

from .geometry_search import (
    GeometryBranchOutcome,
    GeometrySearchConfig,
    _public_state_signature,
    enumerate_geometry_candidates,
    evaluate_geometry_candidate,
)
from .runway_search import RunwaySearchResult
from .steering import SteeringDecision


RESERVE_BAND_SEARCH_VERSION = "r3e-renewable-reserve-band-teacher-v2"
SelectionMode = Literal["score_first", "pure_gauge", "reserve_band"]
_MODES = frozenset({"score_first", "pure_gauge", "reserve_band"})


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class ReserveBandSearchConfig:
    """Identity-bound objective over one fixed full-runway branch set."""

    mode: SelectionMode = "reserve_band"
    runway_ticks: int = 256
    candidate_config: GeometrySearchConfig = field(
        default_factory=GeometrySearchConfig
    )
    efficiency_ceiling_numerator: int = 1
    efficiency_ceiling_denominator: int = 2
    rot_delay_ticks: int = 40
    minimum_contingency_rot_events: int = 1

    def __post_init__(self) -> None:
        if self.mode not in _MODES:
            raise ValueError("reserve-band selection mode is unsupported")
        if type(self.runway_ticks) is not int or not 2 <= self.runway_ticks <= 100_000:
            raise ValueError("reserve-band runway ticks must be in [2, 100000]")
        if not isinstance(self.candidate_config, GeometrySearchConfig):
            raise TypeError("reserve-band candidate config is invalid")
        if (
            type(self.efficiency_ceiling_numerator) is not int
            or type(self.efficiency_ceiling_denominator) is not int
            or self.efficiency_ceiling_numerator != 1
            or self.efficiency_ceiling_denominator != 2
        ):
            raise ValueError(
                "v1 binds the efficiency ceiling to the exact half-gauge boundary"
            )
        if type(self.rot_delay_ticks) is not int or self.rot_delay_ticks != 40:
            raise ValueError("v2 binds the exact strict rot threshold to 40")
        if (
            type(self.minimum_contingency_rot_events) is not int
            or self.minimum_contingency_rot_events != 1
        ):
            raise ValueError("v2 binds one contingency rot liability")

    def manifest(self) -> dict[str, object]:
        return {
            "version": RESERVE_BAND_SEARCH_VERSION,
            "mode": self.mode,
            "runway_ticks": self.runway_ticks,
            "candidate_config": self.candidate_config.manifest(),
            "candidate_slot_count": self.candidate_config.slot_count,
            "efficiency_ceiling_fraction": {
                "numerator": self.efficiency_ceiling_numerator,
                "denominator": self.efficiency_ceiling_denominator,
            },
            "reserve_formula": (
                "for each branch i: min(gauge_max/2, "
                "max(one contingency, visible nonrotten piece rot timers due "
                "by i's first renewable gauge event) * current-level rot "
                "penalty + single-rate passive drain until that event; use "
                "the full runway when no renewable event occurs)"
            ),
            "rot_delay_ticks": self.rot_delay_ticks,
            "minimum_contingency_rot_events": (
                self.minimum_contingency_rot_events
            ),
            "rot_liability_scope": (
                "public nonrotten pieces with nonzero rot_timer and a strict "
                "rot deadline inside the runway; conservative because hidden "
                "rule guards are not policy inputs"
            ),
            "renewable_event_details": [
                "normal burst landing",
                "special color clear",
            ],
            "mechanic": {
                "passive_drain_at_or_below_half": "D",
                "passive_drain_above_half": "3D",
                "boundary_comparison": "gauge > gauge_max / 2",
            },
            "eligibility": (
                "valid and alive/survival-tick nondominated by incumbent; "
                "gauge may be spent down to the renewable reserve"
            ),
            "score_first": (
                "score, qualifying clears, negative rot, final gauge capped "
                "at half, minimum gauge, renewable recovery, causal control"
            ),
            "pure_gauge": (
                "final gauge, minimum gauge, qualifying clears, renewable "
                "recovery, score, causal control"
            ),
            "reserve_band": (
                "score-first only when every selectable branch survives the "
                "full runway and remains at or above its branch-specific "
                "debt-aware reserve; otherwise alive, survival, capped "
                "minimum gauge, capped final gauge, renewable recovery, "
                "qualifying clears, negative rot, score, causal control"
            ),
            "hysteresis": "none; regime is recomputed from every branch set",
            "evidence_scope": "development-teacher-only",
            "deployable": False,
            "canonical_evidence": False,
            "sealed_test_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class ReserveBandBranchOutcome(GeometryBranchOutcome):
    """A base public branch outcome with the active comparator objective."""

    minimum_gauge: int
    gauge_tick_sum: int
    gauge_tick_count: int
    gross_gauge_recovery: int
    first_renewable_recovery_ticks: int
    rot_count: int
    cleared_events: int
    imminent_rot_liability_count: int
    reserve_rot_liability_count: int
    selection_mode: str
    selection_regime: str
    reserve_target_gauge: int
    efficiency_ceiling_gauge: int
    selection_objective: tuple[int | float, ...]

    def survival_nondominated_by(
        self, incumbent: GeometryBranchOutcome
    ) -> bool:
        return (
            int(self.alive) >= int(incumbent.alive)
            and self.survival_ticks >= incumbent.survival_ticks
        )

    @property
    def objective(self) -> tuple[int | float, ...]:
        return self.selection_objective

    def manifest(self) -> dict[str, object]:
        return {
            **super().manifest(),
            "selection_mode": self.selection_mode,
            "selection_regime": self.selection_regime,
            "reserve_target_gauge": self.reserve_target_gauge,
            "efficiency_ceiling_gauge": self.efficiency_ceiling_gauge,
            "selection_objective": list(self.selection_objective),
            "minimum_gauge": self.minimum_gauge,
            "gauge_tick_sum": self.gauge_tick_sum,
            "gauge_tick_count": self.gauge_tick_count,
            "gross_gauge_recovery": self.gross_gauge_recovery,
            "first_renewable_recovery_ticks": (
                self.first_renewable_recovery_ticks
            ),
            "rot_count": self.rot_count,
            "cleared_events": self.cleared_events,
            "imminent_rot_liability_count": (
                self.imminent_rot_liability_count
            ),
            "reserve_rot_liability_count": (
                self.reserve_rot_liability_count
            ),
        }


def _objective(
    outcome: ReserveBandBranchOutcome,
    *,
    regime: str,
    ceiling: int,
) -> tuple[int | float, ...]:
    causal = (
        int(outcome.intended_pair_joined),
        outcome.intended_source_hits,
        outcome.pair_closure_sizes,
        outcome.highest_chain_gain,
    )
    if regime == "score":
        return (
            outcome.score_gain,
            outcome.qualifying_clear_gain,
            -outcome.rot_count,
            min(outcome.final_gauge, ceiling),
            outcome.minimum_gauge,
            outcome.gross_gauge_recovery,
            *causal,
        )
    if regime == "gauge":
        return (
            outcome.final_gauge,
            outcome.minimum_gauge,
            outcome.qualifying_clear_gain,
            outcome.gross_gauge_recovery,
            outcome.score_gain,
            *causal,
        )
    if regime == "recovery":
        return (
            int(outcome.alive),
            outcome.survival_ticks,
            min(outcome.minimum_gauge, outcome.reserve_target_gauge),
            min(outcome.final_gauge, outcome.reserve_target_gauge),
            outcome.gross_gauge_recovery,
            outcome.qualifying_clear_gain,
            -outcome.rot_count,
            outcome.score_gain,
            *causal,
        )
    raise ValueError("reserve-band objective regime is unsupported")


class _TracingEnvironment:
    def __init__(self, env: Any) -> None:
        self._env = env
        self.events: list[Mapping[str, Any]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: Any) -> Any:
        result = self._env.step(action)
        info = result[4]
        if not isinstance(info, Mapping):
            raise TypeError("reserve-band transition info must be a mapping")
        self.events.extend(
            event
            for event in info.get("events", ())
            if isinstance(event, Mapping)
        )
        return result


def _event_kind(event: Mapping[str, Any]) -> int | None:
    raw = event.get("kind")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    name = event.get("kind_name")
    if isinstance(name, str):
        try:
            return int(EventKind[name.upper()])
        except KeyError:
            return None
    return None


def _trace_metrics(
    *,
    initial_gauge: int,
    start_tick: int,
    final_gauge: int,
    final_tick: int,
    runway_ticks: int,
    events: list[Mapping[str, Any]],
) -> tuple[int, int, int, int, int, int, int]:
    gauge = initial_gauge
    minimum = gauge
    recovery = 0
    first_recovery = runway_ticks
    end_by_tick: dict[int, int] = {}
    ordered = sorted(
        enumerate(events),
        key=lambda item: (
            int(item[1].get("tick", -1)),
            int(item[1].get("sequence", item[0])),
            item[0],
        ),
    )
    for _index, event in ordered:
        kind = _event_kind(event)
        if kind != int(EventKind.GAUGE_CHANGED):
            continue
        tick = int(event.get("tick", -1))
        delta = int(event.get("value", 0))
        if not start_tick < tick <= final_tick:
            raise RuntimeError("reserve-band gauge event lies outside its runway")
        gauge += delta
        minimum = min(minimum, gauge)
        if (
            delta > 0
            and str(event.get("detail", ""))
            in {"normal burst landing", "special color clear"}
        ):
            recovery += delta
            first_recovery = min(first_recovery, tick - start_tick)
        end_by_tick[tick] = gauge
    tick_sum = 0
    samples = max(0, final_tick - start_tick)
    gauge = initial_gauge
    for tick in range(start_tick + 1, final_tick + 1):
        gauge = end_by_tick.get(tick, gauge)
        tick_sum += gauge
    if gauge != final_gauge:
        raise RuntimeError("reserve-band gauge events do not reconstruct final gauge")
    rot = sum(
        _event_kind(event) == int(EventKind.ROTTEN) for event in events
    )
    cleared = sum(
        _event_kind(event) == int(EventKind.CLEARED) for event in events
    )
    return minimum, tick_sum, samples, recovery, first_recovery, rot, cleared


def _imminent_rot_deadlines(
    observation: Mapping[str, Any],
    *,
    runway_ticks: int,
    rot_delay_ticks: int,
) -> tuple[int, ...]:
    bodies = observation.get("bodies", ())
    if not isinstance(bodies, (list, tuple)):
        raise TypeError("reserve-band public bodies must be a sequence")
    deadlines: list[int] = []
    for body in bodies:
        if not isinstance(body, Mapping):
            continue
        timer = int(body.get("rot_timer", 0))
        if (
            body.get("kind") != "piece"
            or body.get("lifecycle") == "rotten"
            or timer <= 0
        ):
            continue
        deadline = max(1, rot_delay_ticks + 1 - timer)
        if deadline <= runway_ticks:
            deadlines.append(deadline)
    return tuple(sorted(deadlines))


class ReserveBandGeometrySearch:
    """Transactional score/gauge/reserve-band oracle over identical futures."""

    def __init__(
        self,
        *,
        config: ReserveBandSearchConfig | None = None,
        action_spec: ActionSpec | None = None,
    ) -> None:
        self.config = ReserveBandSearchConfig() if config is None else config
        self.action_spec = ActionSpec() if action_spec is None else action_spec
        if not isinstance(self.config, ReserveBandSearchConfig):
            raise TypeError("reserve-band config is invalid")
        if not isinstance(self.action_spec, ActionSpec):
            raise TypeError("reserve-band action spec is invalid")

    def identity_manifest(self) -> dict[str, object]:
        return {
            "version": RESERVE_BAND_SEARCH_VERSION,
            "config": self.config.manifest(),
            "action_spec": self.action_spec.manifest(),
            "policy_inputs": [
                "initial public observation",
                "incumbent public directed-pair decision",
            ],
            "teacher_only_future": (
                "public observations/events from identical restored portable "
                "snapshots for the fixed candidate vocabulary"
            ),
            "hidden_policy_inputs": [],
            "evidence_scope": "development-teacher-only",
            "deployable": False,
            "canonical_evidence": False,
            "sealed_test_allowed": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())

    def search(
        self,
        env: Any,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
    ) -> RunwaySearchResult:
        if getattr(env, "physics_backend", None) != "portable":
            raise ValueError("reserve-band search requires portable backend")
        if not isinstance(observation, Mapping):
            raise TypeError("reserve-band observation must be a public mapping")
        if bool(observation.get("terminated", False)) or bool(
            observation.get("truncated", False)
        ):
            raise ValueError("cannot reserve-band search a terminal observation")
        clone = getattr(env, "clone_state", None)
        restore = getattr(env, "restore_state", None)
        if not callable(clone) or not callable(restore):
            raise TypeError("portable reserve-band environment lacks clone/restore")

        gauge_max = int(observation.get("gauge_max", 0))
        gauge = int(observation.get("gauge", -1))
        if gauge_max <= 0 or not 0 <= gauge <= gauge_max:
            raise ValueError("reserve-band observation has invalid public gauge")
        ceiling = (
            gauge_max
            * self.config.efficiency_ceiling_numerator
            // self.config.efficiency_ceiling_denominator
        )
        level = min(int(observation.get("level", 0)), 99)
        if level < 1:
            raise ValueError("reserve-band observation has invalid public level")
        passive_drain = level // 10 + 1
        rot_penalty = 1_800 + 20 * level
        rot_deadlines = _imminent_rot_deadlines(
            observation,
            runway_ticks=self.config.runway_ticks,
            rot_delay_ticks=self.config.rot_delay_ticks,
        )
        candidate_set = enumerate_geometry_candidates(
            observation,
            incumbent,
            config=self.config.candidate_config,
            action_spec=self.action_spec,
        )
        expected = _public_state_signature(observation)
        snapshot = clone()
        base_outcomes: list[
            tuple[
                GeometryBranchOutcome,
                tuple[int, int, int, int, int, int, int],
            ]
        ] = []
        try:
            for candidate in candidate_set.candidates:
                restored = restore(snapshot)
                if not isinstance(restored, Mapping):
                    raise TypeError("portable restore must return a public mapping")
                if _public_state_signature(restored) != expected:
                    raise RuntimeError(
                        "reserve-band restore disagrees with supplied public state"
                    )
                tracing = _TracingEnvironment(env)
                outcome = evaluate_geometry_candidate(
                    tracing,
                    restored,
                    candidate,
                    horizon_ticks=self.config.runway_ticks,
                    action_spec=self.action_spec,
                )
                metrics = _trace_metrics(
                    initial_gauge=int(restored.get("gauge", -1)),
                    start_tick=int(restored.get("tick", -1)),
                    final_gauge=outcome.final_gauge,
                    final_tick=(
                        int(restored.get("tick", -1)) + outcome.survival_ticks
                    ),
                    runway_ticks=self.config.runway_ticks,
                    events=tracing.events,
                )
                base_outcomes.append((outcome, metrics))
        finally:
            restore(snapshot)

        incumbent_outcome = base_outcomes[0][0]
        if not incumbent_outcome.selectable:
            raise RuntimeError("reserve-band incumbent emitted an invalid action")
        branch_reserves: list[tuple[int, int]] = []
        for _outcome, metrics in base_outcomes:
            first_recovery = metrics[4]
            due_liabilities = sum(
                deadline <= first_recovery for deadline in rot_deadlines
            )
            reserve_liabilities = max(
                self.config.minimum_contingency_rot_events,
                due_liabilities,
            )
            branch_reserves.append(
                (
                    min(
                        ceiling,
                        reserve_liabilities * rot_penalty
                        + first_recovery * passive_drain,
                    ),
                    reserve_liabilities,
                )
            )
        if self.config.mode == "score_first":
            regime = "score"
        elif self.config.mode == "pure_gauge":
            regime = "gauge"
        else:
            regime = (
                "score"
                if all(
                    outcome.selectable
                    and outcome.alive
                    and outcome.survival_ticks == self.config.runway_ticks
                    and metrics[0] >= reserve[0]
                    for (outcome, metrics), reserve in zip(
                        base_outcomes, branch_reserves, strict=True
                    )
                    if outcome.selectable
                )
                else "recovery"
            )
        staged = tuple(
            ReserveBandBranchOutcome(
                outcome.candidate,
                outcome.score_gain,
                outcome.alive,
                outcome.survival_ticks,
                outcome.final_gauge,
                outcome.qualifying_clear_gain,
                outcome.highest_chain_gain,
                outcome.intended_source_hits,
                outcome.intended_pair_joined,
                outcome.pair_closure_sizes,
                outcome.invalid_actions,
                *metrics,
                len(rot_deadlines),
                reserve[1],
                self.config.mode,
                regime,
                reserve[0],
                ceiling,
                (),
            )
            for (outcome, metrics), reserve in zip(
                base_outcomes, branch_reserves, strict=True
            )
        )
        outcomes = tuple(
            ReserveBandBranchOutcome(
                *(
                    getattr(outcome, field_name)
                    for field_name in GeometryBranchOutcome.__dataclass_fields__
                ),
                outcome.minimum_gauge,
                outcome.gauge_tick_sum,
                outcome.gauge_tick_count,
                outcome.gross_gauge_recovery,
                outcome.first_renewable_recovery_ticks,
                outcome.rot_count,
                outcome.cleared_events,
                outcome.imminent_rot_liability_count,
                outcome.reserve_rot_liability_count,
                outcome.selection_mode,
                outcome.selection_regime,
                outcome.reserve_target_gauge,
                outcome.efficiency_ceiling_gauge,
                _objective(
                    outcome,
                    regime=regime,
                    ceiling=ceiling,
                ),
            )
            for outcome in staged
        )
        incumbent_bound = outcomes[0]
        eligible = tuple(
            outcome
            for outcome in outcomes
            if outcome.selectable
            and outcome.survival_nondominated_by(incumbent_bound)
        )
        winner = max(
            eligible,
            key=lambda outcome: (outcome.objective, -outcome.candidate.ordinal),
        )
        strictly_improved = winner.objective > incumbent_bound.objective
        if not strictly_improved:
            winner = incumbent_bound
        return RunwaySearchResult(
            self.sha256,
            candidate_set,
            self.config.runway_ticks,
            winner.candidate,
            strictly_improved,
            outcomes,
        )

    def act(
        self,
        env: Any,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
    ) -> SteeringDecision:
        return self.search(env, observation, incumbent).decision

    choose = act


__all__ = [
    "RESERVE_BAND_SEARCH_VERSION",
    "ReserveBandBranchOutcome",
    "ReserveBandGeometrySearch",
    "ReserveBandSearchConfig",
    "SelectionMode",
]
