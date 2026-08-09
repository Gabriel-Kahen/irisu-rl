#!/usr/bin/env python3
"""Development-only synthetic checks for the reserve-band teacher."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_pointer import reserve_band_search as reserve
from irisu_pointer.geometry_search import (
    GeometryBranchOutcome,
    GeometryCandidate,
    GeometryCandidateSet,
    GeometrySearchConfig,
)
from irisu_pointer.geometry_ranking import geometry_outcome_ordering
from irisu_pointer.reserve_band_search import (
    RESERVE_BAND_SEARCH_VERSION,
    ReserveBandBranchOutcome,
    ReserveBandGeometrySearch,
    ReserveBandSearchConfig,
    _objective,
)
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import ActionSpec, SemanticAction


class CheckFailure(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailure(message)


def require_raises(
    error_type: type[BaseException],
    message: str,
    operation: Callable[[], object],
) -> None:
    try:
        operation()
    except error_type as error:
        require(message in str(error), f"unexpected error: {error}")
    else:
        raise CheckFailure(f"expected {error_type.__name__}: {message}")


def branch(ordinal: int, **changes: object) -> GeometryBranchOutcome:
    values = {
        "score": 0,
        "alive": True,
        "survival": 2,
        "gauge": 50,
        "gauge_max": 10_000,
        "clears": 0,
        "invalid": 0,
    }
    values.update(changes)
    decision = SteeringDecision(
        SemanticAction.strong(0.5, 0.5),
        SteeringIntent.STEER_MATCH,
        source_body_id=1,
        destination_body_id=2,
        destination_chain_id=3,
        reason=f"synthetic-{ordinal}",
    )
    candidate = GeometryCandidate(
        f"candidate-{ordinal}",
        "incumbent" if ordinal == 0 else "synthetic",
        ordinal,
        decision,
        320,
        240,
    )
    return GeometryBranchOutcome(
        candidate=candidate,
        score_gain=int(values["score"]),
        alive=bool(values["alive"]),
        survival_ticks=int(values["survival"]),
        final_gauge=int(values["gauge"]),
        gauge_max=int(values["gauge_max"]),
        qualifying_clear_gain=int(values["clears"]),
        highest_chain_gain=0,
        intended_source_hits=0,
        intended_pair_joined=False,
        pair_closure_sizes=0.0,
        invalid_actions=int(values["invalid"]),
    )


def ranked(
    *,
    minimum: int | None = None,
    recovery: int = 0,
    first_recovery: int = 2,
    target: int = 50,
    **changes: object,
) -> ReserveBandBranchOutcome:
    base = branch(0, **changes)
    minimum = base.final_gauge if minimum is None else minimum
    return ReserveBandBranchOutcome(
        *(getattr(base, name) for name in GeometryBranchOutcome.__dataclass_fields__),
        minimum,
        base.final_gauge * base.survival_ticks,
        base.survival_ticks,
        recovery,
        first_recovery,
        0,
        base.qualifying_clear_gain,
        0,
        1,
        "reserve_band",
        "recovery",
        target,
        50,
        (),
    )


def synthetic_search(
    branches: tuple[GeometryBranchOutcome, ...],
    mode: str,
    *,
    level: int = 1,
    gauge: int = 5_000,
    gauge_max: int = 10_000,
    runway_ticks: int = 2,
    bodies: tuple[dict[str, object], ...] = (),
    first_recovery_ticks: int | None = None,
):
    geometry_config = GeometrySearchConfig(max_candidates=7)
    action_spec = ActionSpec()
    candidates = GeometryCandidateSet(
        geometry_config,
        action_spec.sha256,
        {"id": 1},
        {"id": 2},
        0.0,
        tuple(value.candidate for value in branches),
    )
    observation = {
        "tick": 100,
        "gauge": gauge,
        "gauge_max": gauge_max,
        "level": level,
        "bodies": bodies,
    }

    class PortableFixture:
        physics_backend = "portable"

        def clone_state(self) -> object:
            return object()

        def restore_state(self, snapshot: object) -> dict[str, object]:
            del snapshot
            return dict(observation)

    saved = (
        reserve.enumerate_geometry_candidates,
        reserve.evaluate_geometry_candidate,
        reserve._trace_metrics,
    )
    reserve.enumerate_geometry_candidates = lambda *_args, **_kwargs: candidates
    reserve.evaluate_geometry_candidate = (
        lambda _env, _observation, selected, **_kwargs: branches[selected.ordinal]
    )
    reserve._trace_metrics = (
        lambda *,
        initial_gauge,
        start_tick,
        final_gauge,
        final_tick,
        runway_ticks,
        events: (
            final_gauge,
            final_gauge * (final_tick - start_tick),
            final_tick - start_tick,
            max(0, final_gauge - initial_gauge),
            (
                runway_ticks
                if first_recovery_ticks is None
                else first_recovery_ticks
            ),
            0,
            0,
        )
    )
    try:
        teacher = ReserveBandGeometrySearch(
            config=ReserveBandSearchConfig(
                mode=mode,
                runway_ticks=runway_ticks,
                candidate_config=geometry_config,
            ),
            action_spec=action_spec,
        )
        return teacher.search(
            PortableFixture(), observation, branches[0].candidate.decision
        )
    finally:
        (
            reserve.enumerate_geometry_candidates,
            reserve.evaluate_geometry_candidate,
            reserve._trace_metrics,
        ) = saved


def check_config_and_boundary() -> None:
    config = ReserveBandSearchConfig()
    manifest = config.manifest()
    require(
        (
            config.efficiency_ceiling_numerator,
            config.efficiency_ceiling_denominator,
            config.rot_delay_ticks,
            config.minimum_contingency_rot_events,
        )
        == (1, 2, 40, 1),
        "ceiling/reserve binding changed",
    )
    require(
        manifest["mechanic"]
        == {
            "passive_drain_at_or_below_half": "D",
            "passive_drain_above_half": "3D",
            "boundary_comparison": "gauge > gauge_max / 2",
        },
        "passive-drain boundary changed",
    )

    branches = (
        branch(0, score=1, gauge=5_000),
        branch(1, score=10, gauge=1_822),
    )
    at_reserve = synthetic_search(branches, "reserve_band")
    require(at_reserve.winner_ordinal == 1, "score regime rejected exact reserve")
    require(
        {
            (
                value.selection_regime,
                value.reserve_target_gauge,
                value.efficiency_ceiling_gauge,
            )
            for value in at_reserve.outcomes
        }
        == {("score", 1_822, 5_000)},
        "dynamic reserve or half-gauge ceiling is wrong",
    )

    below = list(branches)
    below[1] = replace(below[1], final_gauge=1_821)
    below_reserve = synthetic_search(tuple(below), "reserve_band")
    require(below_reserve.winner_ordinal == 0, "recovery regime spent reserve")
    require(
        {value.selection_regime for value in below_reserve.outcomes} == {"recovery"},
        "below-reserve branch did not trigger recovery",
    )
    ordering = geometry_outcome_ordering(below_reserve.outcomes)
    require(
        (
            ordering.winner_ordinal,
            ordering.strictly_improved,
        )
        == (
            below_reserve.winner_ordinal,
            below_reserve.strictly_improved,
        ),
        "all-branch distillation ordering changed the reserve winner",
    )
    require(
        synthetic_search(tuple(below), "score_first").winner_ordinal == 1,
        "score-first comparator did not prefer score",
    )
    require(
        synthetic_search(tuple(below), "pure_gauge").winner_ordinal == 0,
        "pure-gauge comparator did not prefer gauge",
    )

    for unsafe in (
        branch(2, alive=False, survival=2, gauge=5_000),
        branch(2, alive=True, survival=1, gauge=5_000),
    ):
        result = synthetic_search((*branches, unsafe), "reserve_band")
        require(
            {value.selection_regime for value in result.outcomes} == {"recovery"},
            "selectable non-full-runway branch did not force recovery",
        )
    invalid = branch(2, alive=False, survival=1, gauge=0, invalid=1)
    require(
        {
            value.selection_regime
            for value in synthetic_search((*branches, invalid), "reserve_band").outcomes
        }
        == {"score"},
        "nonselectable branch incorrectly forced recovery",
    )

    capped = synthetic_search(
        (branch(0, gauge=10_000, survival=17),),
        "reserve_band",
        level=150,
        gauge=10_000,
        gauge_max=20_000,
        runway_ticks=17,
    )
    require(
        {value.reserve_target_gauge for value in capped.outcomes} == {3_950},
        "reserve formula did not clamp level to 99 or include the runway",
    )

    debt = synthetic_search(
        branches,
        "reserve_band",
        bodies=tuple(
            {
                "id": index,
                "kind": "piece",
                "shape": "box",
                "lifecycle": lifecycle,
                "color": 1,
                "x": float(index),
                "y": 1.0,
                "vx": 0.0,
                "vy": 0.0,
                "angle": 0.0,
                "angular_velocity": 0.0,
                "size": 32.0,
                "chain_id": 0,
                "rot_timer": timer,
            }
            for index, lifecycle, timer in (
                (10, "dynamic_fresh", 40),
                (11, "confirmed", 39),
                (12, "rotten", 1),
            )
        ),
        first_recovery_ticks=2,
    )
    require(
        {
            (
                value.imminent_rot_liability_count,
                value.reserve_rot_liability_count,
                value.reserve_target_gauge,
            )
            for value in debt.outcomes
        }
        == {(2, 2, 3_642)},
        "visible rot debt or time-to-renewal reserve is wrong",
    )


def check_objectives() -> None:
    sample = ranked(
        score=11,
        gauge=13,
        minimum=7,
        clears=2,
        recovery=5,
        target=10,
        survival=20,
    )
    causal = (0, 0, 0.0, 0)
    expected = {
        "score": (11, 2, 0, 10, 7, 5, *causal),
        "gauge": (13, 7, 2, 5, 11, *causal),
        "recovery": (1, 20, 7, 10, 5, 2, 0, 11, *causal),
    }
    for regime, objective in expected.items():
        require(
            _objective(sample, regime=regime, ceiling=10)
            == objective,
            f"{regime} objective fields are out of order",
        )
    require(
        _objective(
            ranked(score=11, gauge=0),
            regime="score",
            ceiling=50,
        )
        > _objective(
            ranked(score=10, gauge=100),
            regime="score",
            ceiling=50,
        ),
        "score objective does not lead with score",
    )
    require(
        _objective(
            ranked(score=10, gauge=100),
            regime="gauge",
            ceiling=50,
        )
        > _objective(
            ranked(score=11, gauge=0),
            regime="gauge",
            ceiling=50,
        ),
        "gauge objective does not lead with gauge",
    )
    require_raises(
        ValueError,
        "objective regime is unsupported",
        lambda: _objective(sample, regime="unknown", ceiling=50),
    )


def check_eligibility() -> None:
    result = synthetic_search(
        (
            branch(0, score=1, gauge=90),
            branch(1, score=10, gauge=50),
            branch(2, score=1000, alive=False, survival=100, gauge=100),
            branch(3, score=1000, survival=1, gauge=100),
            branch(4, score=1000, survival=100, gauge=100, invalid=1),
        ),
        "score_first",
    )
    incumbent = result.outcomes[0]
    eligible = tuple(
        value.candidate.ordinal
        for value in result.outcomes
        if value.selectable and value.survival_nondominated_by(incumbent)
    )
    require(eligible == (0, 1), "eligibility changed gauge/alive/survival rules")
    require(result.winner_ordinal == 1, "eligible lower-gauge score branch lost")


def check_renewable_recovery() -> None:
    gauge_changed = int(reserve.EventKind.GAUGE_CHANGED)
    events = [
        {
            "kind": gauge_changed,
            "tick": 1,
            "sequence": 0,
            "value": -100,
            "detail": "passive drain",
        },
        {
            "kind": gauge_changed,
            "tick": 2,
            "sequence": 0,
            "value": 30,
            "detail": "normal burst landing",
        },
        {
            "kind": gauge_changed,
            "tick": 2,
            "sequence": 1,
            "value": 40,
            "detail": "special color clear",
        },
        {
            "kind": gauge_changed,
            "tick": 3,
            "value": 50,
            "detail": "terminal clamp",
        },
        {
            "kind": gauge_changed,
            "tick": 4,
            "value": 10,
            "detail": "other positive adjustment",
        },
        {"kind": int(reserve.EventKind.ROTTEN), "tick": 2},
        {"kind": int(reserve.EventKind.CLEARED), "tick": 2},
    ]
    require(
        reserve._trace_metrics(
            initial_gauge=1_000,
            start_tick=0,
            final_gauge=1_030,
            final_tick=4,
            runway_ticks=4,
            events=events,
        )
        == (900, 3_920, 4, 70, 2, 1, 1),
        "renewable recovery counted terminal/nonrenewable positive gauge",
    )


def check_identity_and_fail_closed_config() -> None:
    config = ReserveBandSearchConfig(
        runway_ticks=17,
        candidate_config=GeometrySearchConfig(max_candidates=7),
    )
    action_spec = ActionSpec(wait_choices=(1, 2))
    first = ReserveBandGeometrySearch(config=config, action_spec=action_spec)
    second = ReserveBandGeometrySearch(config=config, action_spec=action_spec)
    encoded = json.dumps(
        first.identity_manifest(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    require(first.identity_manifest() == second.identity_manifest(), "manifest drift")
    require(first.sha256 == second.sha256, "identical teachers changed identity")
    require(first.sha256 == hashlib.sha256(encoded).hexdigest(), "noncanonical SHA")
    require(
        first.sha256
        != ReserveBandGeometrySearch(
            config=replace(config, mode="pure_gauge"),
            action_spec=action_spec,
        ).sha256,
        "selection mode is absent from identity",
    )

    cases = (
        (ValueError, "selection mode", lambda: ReserveBandSearchConfig(mode="other")),
        (ValueError, "runway ticks", lambda: ReserveBandSearchConfig(runway_ticks=True)),
        (TypeError, "candidate config", lambda: ReserveBandSearchConfig(candidate_config={})),
        (
            ValueError,
            "exact half-gauge",
            lambda: ReserveBandSearchConfig(
                efficiency_ceiling_numerator=2,
                efficiency_ceiling_denominator=4,
            ),
        ),
        (
            ValueError,
            "exact half-gauge",
            lambda: ReserveBandSearchConfig(efficiency_ceiling_numerator=True),
        ),
        (
            ValueError,
            "one contingency rot liability",
            lambda: ReserveBandSearchConfig(
                minimum_contingency_rot_events=2
            ),
        ),
        (
            ValueError,
            "strict rot threshold",
            lambda: ReserveBandSearchConfig(rot_delay_ticks=41),
        ),
        (TypeError, "reserve-band config", lambda: ReserveBandGeometrySearch(config={})),
        (TypeError, "action spec", lambda: ReserveBandGeometrySearch(action_spec={})),
    )
    for error_type, message, operation in cases:
        require_raises(error_type, message, operation)


def main() -> None:
    checks = (
        check_config_and_boundary,
        check_objectives,
        check_eligibility,
        check_renewable_recovery,
        check_identity_and_fail_closed_config,
    )
    for check in checks:
        check()
    source_path = Path(reserve.__file__).resolve()
    print(
        json.dumps(
            {
                "status": "PASS",
                "checks": len(checks),
                "source_version": RESERVE_BAND_SEARCH_VERSION,
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "default_teacher_sha256": ReserveBandGeometrySearch().sha256,
                "check_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
