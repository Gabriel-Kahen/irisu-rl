from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from irisu_env import ActionKind, EventKind
from irisu_pointer.event_mpc import (
    CandidateCertificate,
    CandidateDebtLedger,
    EventMPCConfig,
    ExactEventOutcome,
    ExactEventPlanner,
    ExactSearchResult,
    FEATURE_NAMES,
    KNNEventWorldModel,
    ModelBarrierPolicy,
    ModelExample,
    RenewableCycle,
    TARGET_NAMES,
    replay_two_renewal_cashflow,
)
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import SemanticAction


def _decision(reason: str = "incumbent") -> SteeringDecision:
    return SteeringDecision(
        SemanticAction.strong(0.25, 0.25),
        SteeringIntent.STEER_MATCH,
        source_body_id=11,
        destination_body_id=22,
        reason=reason,
    )


def _candidate(
    ordinal: int, decision: SteeringDecision | None = None
) -> SimpleNamespace:
    generated = SteeringDecision(
        SemanticAction.strong(0.25 + 0.01 * ordinal, 0.25),
        SteeringIntent.STEER_MATCH,
        source_body_id=11,
        destination_body_id=22,
        reason=f"candidate {ordinal}",
    )
    return SimpleNamespace(
        ordinal=ordinal,
        pair_ordinal=int(ordinal > 0),
        geometry_ordinal=ordinal,
        decision=generated if decision is None else decision,
        pair=SimpleNamespace(
            source_body_id=11,
            destination_body_id=22,
            destination_chain_id=0,
            category="fresh-match",
            distance_sizes=1.0,
        ),
        geometry=SimpleNamespace(
            name=f"geometry-{ordinal}",
            strength="strong",
            side_sizes=0.5 + 0.1 * ordinal,
            below_sizes=0.75,
        ),
    )


def _observation(*, bodies: tuple[dict[str, object], ...] = ()) -> dict[str, object]:
    return {
        "tick": 100,
        "score": 1_000,
        "gauge": 10_000,
        "gauge_max": 20_000,
        "level": 5,
        "qualifying_clear_count": 2,
        "terminated": False,
        "truncated": False,
        "field": {"x": 0.0, "y": 0.0, "width": 640.0, "height": 480.0},
        "difficulty": {"spawn_interval_ticks": 100},
        "bodies": bodies,
    }


def _piece(body_id: int, *, rot_timer: int = 1) -> dict[str, object]:
    return {
        "id": body_id,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": "dynamic_fresh",
        "color": 0,
        "x": float(body_id),
        "y": 180.0,
        "vx": 0.0,
        "vy": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 40.0,
        "chain_id": 0,
        "projectile_hits": 0,
        "remaining_lifetime": 1_000,
        "rot_timer": rot_timer,
    }


def _cycle(
    ordinal: int, *, completed: bool = True, surplus: int = 1_000
) -> RenewableCycle:
    start = (ordinal - 1) * 20
    return RenewableCycle(
        ordinal=ordinal,
        start_tick=start,
        end_tick=start + 20,
        completed=completed,
        start_gauge=10_000,
        renewal_gain=2_000 if completed else 0,
        passive_drain=100,
        rot_payment=0,
        rot_events=0,
        realized_rot_liability=0,
        minimum_gauge_before_renewal=9_900,
        solvency_surplus=surplus,
    )


def _outcome(
    ordinal: int,
    *,
    complete: bool = True,
    score_gain: int = 0,
    surplus: int = 1_000,
    invalid_actions: int = 0,
    duplicate_payments: int = 0,
    unmatched_payments: int = 0,
) -> ExactEventOutcome:
    renewals = (20, 40) if complete else (20,)
    gains = (2_000, 2_000) if complete else (2_000,)
    return ExactEventOutcome(
        candidate_ordinal=ordinal,
        candidate_name=f"candidate-{ordinal}",
        pair_ordinal=int(ordinal > 0),
        geometry_ordinal=ordinal,
        pair_category="fresh-match",
        start_tick=0,
        survival_ticks=40,
        alive=True,
        invalid_actions=invalid_actions,
        score_gain=score_gain,
        final_gauge=10_000,
        level_delta=0,
        qualifying_clear_gain=len(renewals),
        shot_hit=True,
        pair_joined=True,
        renewable_ticks=renewals,
        renewable_gains=gains,
        cycles=(
            _cycle(1, surplus=surplus),
            _cycle(2, completed=complete, surplus=surplus),
        ),
        rot_events=0,
        rot_payment=0,
        passive_drain=200,
        liabilities_created=0,
        liabilities_paid=0,
        liabilities_retired=0,
        duplicate_payments=duplicate_payments,
        unmatched_payments=unmatched_payments,
        snapshot_sha256="snapshot",
        exact_state_hash="exact",
    )


def _targets(*, second_renewal: float = 1.0) -> tuple[float, ...]:
    values = np.zeros(len(TARGET_NAMES), dtype=np.float64)
    values[TARGET_NAMES.index("first_renewal_reached")] = 1.0
    values[TARGET_NAMES.index("second_renewal_reached")] = second_renewal
    values[TARGET_NAMES.index("b2")] = 1_000.0
    values[TARGET_NAMES.index("second_cycle_margin")] = 1_000.0
    values[TARGET_NAMES.index("final_gauge")] = 10_000.0
    values[TARGET_NAMES.index("alive")] = 1.0
    values[TARGET_NAMES.index("survival_ticks")] = 40.0
    values[TARGET_NAMES.index("delta_b2")] = 100.0
    values[TARGET_NAMES.index("delta_final_gauge")] = 100.0
    values[TARGET_NAMES.index("delta_score_gain")] = 10.0
    return tuple(float(value) for value in values)


def _fitted_model(*, second_renewal: float = 1.0) -> KNNEventWorldModel:
    examples = tuple(
        ModelExample(
            split="development",
            seed=index,
            query_id=f"query-{index}",
            decision_tick=100 + index,
            candidate_ordinal=1,
            features=tuple(0.0 for _ in FEATURE_NAMES),
            targets=_targets(second_renewal=second_renewal),
            outcome={},
        )
        # Twenty calibration clusters are required for a finite alpha=.05
        # split-conformal order statistic; the other half fits the model.
        for index in range(40)
    )
    model = KNNEventWorldModel(
        EventMPCConfig(neighbor_count=3, risk_upper_limit=0.95)
    )
    model.fit(examples)
    return model


def _gauge_event(
    tick: int,
    value: int,
    detail: str,
    *,
    sequence: int,
    body_id: int = -1,
) -> dict[str, object]:
    return {
        "kind": int(EventKind.GAUGE_CHANGED),
        "tick": tick,
        "sequence": sequence,
        "a": body_id,
        "value": value,
        "detail": detail,
    }


def _baseline_cashflow_events(
    *, offset: int = 0, split_first_gain: bool = False, second: bool = True
) -> tuple[dict[str, object], ...]:
    first_gain = (
        (
            _gauge_event(
                offset + 2,
                400,
                "special color clear",
                sequence=10,
            ),
            _gauge_event(
                offset + 2,
                600,
                "special color clear",
                sequence=11,
            ),
        )
        if split_first_gain
        else (
            _gauge_event(
                offset + 2,
                1_000,
                "special color clear",
                sequence=10,
            ),
        )
    )
    second_gain = (
        (
            _gauge_event(
                offset + 4,
                500,
                "normal burst landing",
                sequence=10,
            ),
        )
        if second
        else ()
    )
    return (
        _gauge_event(
            offset + 1,
            -1,
            "scene clamp and passive drain",
            sequence=20,
        ),
        *first_gain,
        _gauge_event(
            offset + 2,
            -3,
            "scene clamp and passive drain",
            sequence=20,
        ),
        _gauge_event(
            offset + 3,
            -3,
            "scene clamp and passive drain",
            sequence=20,
        ),
        *second_gain,
        _gauge_event(
            offset + 4,
            -3,
            "scene clamp and passive drain",
            sequence=20,
        ),
    )


def _baseline_cashflow(
    *, offset: int = 0, split_first_gain: bool = False, second: bool = True
):
    return replay_two_renewal_cashflow(
        initial_gauge=5_000,
        gauge_max=10_000,
        initial_level=9,
        start_tick=offset,
        final_tick=offset + 4,
        events=_baseline_cashflow_events(
            offset=offset,
            split_first_gain=split_first_gain,
            second=second,
        ),
    )


class _RestoreEnv:
    physics_backend = "portable"

    def __init__(self) -> None:
        self.snapshot = b"event-mpc-snapshot"
        self.observation = _observation()

    def clone_state(self) -> bytes:
        return self.snapshot

    def restore_state(self, snapshot: bytes) -> dict[str, object]:
        if snapshot != self.snapshot:
            raise AssertionError("wrong snapshot")
        return self.observation

    def state_hash(self) -> str:
        return "exact-state"


class _ScriptedEnv:
    def __init__(
        self,
        rows: tuple[
            tuple[dict[str, object], bool, tuple[dict[str, object], ...]],
            ...,
        ],
    ) -> None:
        self.rows = list(rows)
        self.action_kinds: list[ActionKind] = []

    def step(self, action):
        self.action_kinds.append(ActionKind.parse(action.kind))
        observation, terminated, events = self.rows.pop(0)
        return observation, 0.0, terminated, False, {"events": events}


class _CapTailEnv:
    def __init__(self) -> None:
        self.tick = 100
        self.gauge = 10_000
        self.action_kinds: list[ActionKind] = []

    def step(self, action):
        kind = ActionKind.parse(action.kind)
        self.action_kinds.append(kind)
        duration = int(action.wait_ticks) if kind is ActionKind.WAIT else 1
        events = []
        terminated = False
        for _ in range(duration):
            self.tick += 1
            sequence = 1
            if self.tick in {101, 165}:
                if self.tick == 165:
                    events.append(
                        {
                            "kind": int(EventKind.GAME_OVER),
                            "tick": self.tick,
                            "sequence": sequence,
                        }
                    )
                    sequence += 1
                    terminated = True
                events.append(
                    _gauge_event(
                        self.tick,
                        1_000,
                        "normal burst landing",
                        sequence=sequence,
                    )
                )
                sequence += 1
                self.gauge += 1_000
            events.append(
                _gauge_event(
                    self.tick,
                    -3,
                    "scene clamp and passive drain",
                    sequence=sequence,
                )
            )
            self.gauge -= 3
        observation = {
            **_observation(),
            "tick": self.tick,
            "gauge": self.gauge,
            "terminated": terminated,
        }
        return observation, 0.0, terminated, False, {"events": tuple(events)}


class _CapContinuation:
    def predict(self, observation: dict[str, object]) -> SteeringDecision:
        if int(observation["tick"]) == 102:
            return SteeringDecision(
                SemanticAction.wait(61),
                SteeringIntent.WAIT,
                reason="reach the event cap",
            )
        return _decision("continuation cap shot")


class _BasePolicy:
    def __init__(self, incumbent: SteeringDecision) -> None:
        self.incumbent = incumbent

    def reset(self, seed: int = 0) -> None:
        self.seed = seed

    def predict(self, observation: object) -> SteeringDecision:
        del observation
        return self.incumbent


class _StatefulContinuation(_BasePolicy):
    artifact_sha256 = "continuation"

    def __init__(self, incumbent: SteeringDecision) -> None:
        super().__init__(incumbent)
        self._cooldown_until = 116
        self._last_tick = 100
        self._last_decision = incumbent
        self._progress = {"pending": "incumbent"}


class _OrdinalCertificateModel:
    """Test double whose certificate depends only on candidate features."""

    def certify(
        self, features: np.ndarray
    ) -> tuple[None, CandidateCertificate]:
        ordinal = round(
            (float(features[FEATURE_NAMES.index("geometry_side")]) - 0.5)
            / 0.1
        )
        certified = ordinal == 1
        return None, CandidateCertificate(
            certified=certified,
            reasons=() if certified else ("absolute_solvency_lower_bound",),
            lower_b2=1_000.0 if certified else -1.0,
            lower_delta_b2=100.0 if certified else -1.0,
            lower_delta_final_gauge=100.0 if certified else -1.0,
            predicted_score_advantage=10.0 if certified else 10_000.0,
            catastrophic_probability=0.0 if certified else 1.0,
            catastrophic_upper=0.1 if certified else 1.0,
            all_neighbors_two_renewal=certified,
        )


class EventMPCDebtLedgerTests(unittest.TestCase):
    def test_exact_planner_clones_and_rebinds_live_controller_state(self) -> None:
        incumbent = _decision()
        candidates = (_candidate(0, incumbent), _candidate(1))
        live = _StatefulContinuation(incumbent)
        seen: list[tuple[int, dict[str, str]]] = []

        def rebind(base, observation, old, new):
            del observation, old, new
            base._progress["pending"] = "candidate"
            return True

        planner = ExactEventPlanner(
            lambda: _StatefulContinuation(incumbent),
            lambda observation, decision: candidates,
            continuation_identity_sha256="continuation",
            continuation_rebind=rebind,
        )

        def evaluate(env, observation, candidate, continuation, *identity):
            del env, observation, identity
            self.assertEqual(continuation._cooldown_until, 116)
            self.assertEqual(continuation._last_tick, 100)
            seen.append((candidate.ordinal, dict(continuation._progress)))
            continuation._progress["mutated"] = "branch-only"
            return _outcome(candidate.ordinal)

        with patch.object(planner, "_evaluate", side_effect=evaluate):
            result = planner.search(
                _RestoreEnv(),
                _observation(),
                incumbent,
                continuation_policy=live,
                query_id="controller-clone",
            )

        self.assertEqual(
            seen,
            [
                (0, {"pending": "incumbent"}),
                (1, {"pending": "candidate"}),
            ],
        )
        self.assertEqual(live._progress, {"pending": "incumbent"})
        self.assertEqual(result.restore_checks, 3)

    def test_body_id_payment_is_retired_exactly_once(self) -> None:
        ledger = CandidateDebtLedger()
        ledger.observe(_observation(bodies=(_piece(17), _piece(18))))

        self.assertEqual(ledger.debt_due_by(139), 0)
        self.assertEqual(ledger.debt_due_by(140), 3_800)
        ledger.apply(
            (
                {
                    "kind": int(EventKind.GAUGE_CHANGED),
                    "tick": 140,
                    "sequence": 2,
                    "a": 17,
                    "value": -1_900,
                    "detail": "normal rot penalty",
                },
                {
                    "kind": int(EventKind.ROTTEN),
                    "tick": 140,
                    "sequence": 1,
                    "a": 17,
                },
            )
        )

        self.assertEqual(ledger.paid, {17: 1_900})
        self.assertNotIn(17, ledger.active)
        self.assertIn(18, ledger.active)
        self.assertEqual(ledger.debt_due_by(140), 1_900)
        self.assertEqual(ledger.duplicate_payments, 0)
        self.assertEqual(ledger.unmatched_payments, 0)

    def test_debt_prices_each_liability_at_its_payment_level(self) -> None:
        ledger = CandidateDebtLedger()
        ledger.observe(
            _observation(
                bodies=(
                    _piece(17, rot_timer=40),
                    _piece(18, rot_timer=39),
                )
            )
        )
        levels = {101: 5, 102: 10}

        self.assertEqual(
            tuple(
                value.deadline_tick
                for value in ledger.liabilities_due_by(102)
            ),
            (101, 102),
        )
        self.assertEqual(
            ledger.debt_due_by(
                102,
                level_at_payment=lambda liability: levels[
                    liability.deadline_tick
                ],
            ),
            3_900,
        )

    def test_duplicate_and_unmatched_payments_enter_fail_state(self) -> None:
        duplicate = CandidateDebtLedger()
        duplicate.observe(_observation(bodies=(_piece(17),)))
        duplicate.apply(
            (
                {"kind": int(EventKind.ROTTEN), "a": 17},
                {
                    "kind": int(EventKind.GAUGE_CHANGED),
                    "a": 17,
                    "value": -1_900,
                    "detail": "normal rot penalty",
                },
            )
        )
        duplicate.apply(
            (
                {
                    "kind": int(EventKind.GAUGE_CHANGED),
                    "a": 17,
                    "value": -1_900,
                    "detail": "normal rot penalty",
                },
            )
        )
        unmatched = CandidateDebtLedger()
        unmatched.apply(
            (
                {
                    "kind": int(EventKind.GAUGE_CHANGED),
                    "a": 99,
                    "value": -1_900,
                    "detail": "normal rot penalty",
                },
            )
        )

        self.assertEqual(duplicate.duplicate_payments, 1)
        self.assertEqual(unmatched.unmatched_payments, 1)

    def test_exact_planner_fails_closed_on_any_ledger_mismatch(self) -> None:
        incumbent = _decision()
        candidate = _candidate(0, incumbent)
        for field in ("duplicate_payments", "unmatched_payments"):
            with self.subTest(field=field):
                planner = ExactEventPlanner(
                    lambda: None,
                    lambda observation, decision: (candidate,),
                )
                bad = _outcome(0, **{field: 1})
                with (
                    patch.object(planner, "_evaluate", return_value=bad),
                    self.assertRaisesRegex(RuntimeError, "exactly once"),
                ):
                    planner.search(
                        _RestoreEnv(),
                        _observation(),
                        incumbent,
                        query_id=f"bad-{field}",
                    )


class EventMPCCashflowTests(unittest.TestCase):
    def test_continuation_release_tail_after_tau2_is_not_hidden(self) -> None:
        initial = _observation()

        def observed(tick: int, gauge: int) -> dict[str, object]:
            return {**initial, "tick": tick, "gauge": gauge}

        rows = (
            (
                observed(101, 10_997),
                False,
                (
                    _gauge_event(
                        101,
                        1_000,
                        "normal burst landing",
                        sequence=1,
                    ),
                    _gauge_event(
                        101,
                        -3,
                        "scene clamp and passive drain",
                        sequence=2,
                    ),
                ),
            ),
            (
                observed(102, 10_994),
                False,
                (
                    _gauge_event(
                        102,
                        -3,
                        "scene clamp and passive drain",
                        sequence=1,
                    ),
                ),
            ),
            (
                observed(103, 11_991),
                False,
                (
                    _gauge_event(
                        103,
                        1_000,
                        "special color clear",
                        sequence=1,
                    ),
                    _gauge_event(
                        103,
                        -3,
                        "scene clamp and passive drain",
                        sequence=2,
                    ),
                ),
            ),
            (
                observed(104, 11_991),
                True,
                (
                    {
                        "kind": int(EventKind.GAME_OVER),
                        "tick": 104,
                        "sequence": 1,
                    },
                ),
            ),
        )
        environment = _ScriptedEnv(rows)
        incumbent = _decision()
        planner = ExactEventPlanner(
            lambda: _BasePolicy(_decision("continuation")),
            lambda observation, decision: (_candidate(0, decision),),
        )
        outcome = planner._evaluate(
            environment,
            initial,
            _candidate(0, incumbent),
            _BasePolicy(_decision("continuation")),
            "snapshot",
            "exact",
        )

        self.assertEqual(
            environment.action_kinds,
            [
                ActionKind.STRONG_SHOT,
                ActionKind.WAIT,
                ActionKind.STRONG_SHOT,
                ActionKind.WAIT,
            ],
        )
        self.assertEqual(outcome.renewable_ticks, (101, 103))
        self.assertEqual(outcome.survival_ticks, 3)
        self.assertEqual(outcome.final_gauge, 11_991)
        self.assertTrue(outcome.full_action_terminal)
        self.assertTrue(outcome.full_action_gauge_failure)
        self.assertFalse(outcome.alive)

    def test_post_cap_release_renewal_is_not_counted(self) -> None:
        initial = _observation()
        environment = _CapTailEnv()
        incumbent = _decision()
        planner = ExactEventPlanner(
            lambda: _CapContinuation(),
            lambda observation, decision: (_candidate(0, decision),),
            config=EventMPCConfig(maximum_event_ticks=64),
        )
        outcome = planner._evaluate(
            environment,
            initial,
            _candidate(0, incumbent),
            _CapContinuation(),
            "snapshot",
            "exact",
        )

        self.assertEqual(environment.tick, 165)
        self.assertEqual(
            environment.action_kinds[-2:],
            [ActionKind.STRONG_SHOT, ActionKind.WAIT],
        )
        self.assertEqual(outcome.renewable_ticks, (101,))
        self.assertFalse(outcome.two_renewal_complete)
        self.assertEqual(outcome.survival_ticks, 64)
        self.assertEqual(outcome.final_gauge, 10_808)
        self.assertTrue(outcome.full_action_terminal)
        self.assertTrue(outcome.full_action_gauge_failure)

    def test_distinct_renewals_and_same_tick_gain_split_are_invariant(
        self,
    ) -> None:
        coalesced = _baseline_cashflow()
        split = _baseline_cashflow(split_first_gain=True)

        self.assertEqual(split, coalesced)
        self.assertEqual(coalesced.renewal_ticks, (2, 4))
        self.assertEqual(coalesced.net_renewal_gains, (1_000, 500))
        self.assertTrue(all(cycle.completed for cycle in coalesced.cycles))
        self.assertEqual(coalesced.b2, 4_998)
        self.assertEqual(coalesced.final_replayed_gauge, 6_490)
        self.assertEqual(coalesced.passive_drain, 10)

    def test_absolute_tick_offset_does_not_change_cashflow(self) -> None:
        original = _baseline_cashflow()
        shifted = _baseline_cashflow(offset=1_000)

        self.assertEqual(
            shifted.renewal_ticks,
            tuple(tick + 1_000 for tick in original.renewal_ticks),
        )
        self.assertEqual(shifted.net_renewal_gains, original.net_renewal_gains)
        self.assertEqual(shifted.b2, original.b2)
        self.assertEqual(
            shifted.final_replayed_gauge, original.final_replayed_gauge
        )
        self.assertEqual(shifted.passive_drain, original.passive_drain)
        for before, after in zip(
            original.cycles, shifted.cycles, strict=True
        ):
            before_values = before.manifest()
            after_values = after.manifest()
            self.assertEqual(
                after_values.pop("start_tick"),
                before_values.pop("start_tick") + 1_000,
            )
            self.assertEqual(
                after_values.pop("end_tick"),
                before_values.pop("end_tick") + 1_000,
            )
            self.assertEqual(after_values, before_values)

    def test_event_and_body_order_do_not_change_accounting(self) -> None:
        events = _baseline_cashflow_events()
        reversed_replay = replay_two_renewal_cashflow(
            initial_gauge=5_000,
            gauge_max=10_000,
            initial_level=9,
            start_tick=0,
            final_tick=4,
            events=tuple(reversed(events)),
        )
        self.assertEqual(reversed_replay, _baseline_cashflow())

        forward = CandidateDebtLedger()
        reverse = CandidateDebtLedger()
        bodies = (_piece(17), _piece(18))
        forward.observe(_observation(bodies=bodies))
        reverse.observe(_observation(bodies=tuple(reversed(bodies))))
        payments = (
            {"kind": int(EventKind.ROTTEN), "tick": 140, "a": 17},
            _gauge_event(
                140,
                -1_900,
                "normal rot penalty",
                sequence=2,
                body_id=17,
            ),
            {"kind": int(EventKind.ROTTEN), "tick": 140, "a": 18},
            _gauge_event(
                140,
                -1_900,
                "normal rot penalty",
                sequence=4,
                body_id=18,
            ),
        )
        forward.apply(payments)
        reverse.apply(tuple(reversed(payments)))

        self.assertEqual(reverse.created, forward.created)
        self.assertEqual(reverse.active, forward.active)
        self.assertEqual(reverse.paid, forward.paid)
        self.assertEqual(reverse.duplicate_payments, 0)
        self.assertEqual(reverse.unmatched_payments, 0)

    def test_level_band_drain_and_rot_follow_exact_order(self) -> None:
        events = (
            {
                "kind": int(EventKind.LEVEL_CHANGED),
                "tick": 1,
                "sequence": 1,
                "value": 10,
            },
            _gauge_event(
                1, 4_000, "special color clear", sequence=2
            ),
            _gauge_event(
                1,
                -6,
                "scene clamp and passive drain",
                sequence=3,
            ),
            {
                "kind": int(EventKind.ROTTEN),
                "tick": 1,
                "sequence": 4,
                "a": 17,
            },
            _gauge_event(
                1,
                -2_000,
                "normal rot penalty",
                sequence=5,
                body_id=17,
            ),
            _gauge_event(
                2,
                -6,
                "scene clamp and passive drain",
                sequence=1,
            ),
            {
                "kind": int(EventKind.ROTTEN),
                "tick": 2,
                "sequence": 2,
                "a": 18,
            },
            _gauge_event(
                2,
                -2_000,
                "normal rot penalty",
                sequence=3,
                body_id=18,
            ),
            _gauge_event(
                3,
                -6,
                "scene clamp and passive drain",
                sequence=1,
            ),
            {
                "kind": int(EventKind.ROTTEN),
                "tick": 3,
                "sequence": 2,
                "a": 19,
            },
            _gauge_event(
                3,
                -2_000,
                "normal rot penalty",
                sequence=3,
                body_id=19,
            ),
            _gauge_event(
                4,
                -2,
                "scene clamp and passive drain",
                sequence=1,
            ),
            _gauge_event(
                5, 1_000, "normal burst landing", sequence=1
            ),
            _gauge_event(
                5,
                -2,
                "scene clamp and passive drain",
                sequence=2,
            ),
        )
        replay = replay_two_renewal_cashflow(
            initial_gauge=6_000,
            gauge_max=10_000,
            initial_level=9,
            start_tick=0,
            final_tick=5,
            events=events,
        )

        self.assertEqual(replay.renewal_ticks, (1, 5))
        self.assertEqual(replay.net_renewal_gains, (4_000, 1_000))
        self.assertEqual(replay.passive_drain, 22)
        self.assertEqual(replay.rot_events, 3)
        self.assertEqual(replay.rot_payment, 6_000)
        self.assertEqual(replay.final_replayed_gauge, 4_978)
        self.assertEqual(replay.b2, 3_979)
        self.assertEqual(replay.cycles[0].passive_drain, 6)
        self.assertEqual(replay.cycles[0].rot_payment, 2_000)
        self.assertEqual(replay.cycles[1].passive_drain, 16)
        self.assertEqual(replay.cycles[1].rot_payment, 4_000)

        wrong_level_penalty = [dict(event) for event in events]
        wrong_level_penalty[4]["value"] = -1_980
        with self.assertRaisesRegex(RuntimeError, "level-at-rot"):
            replay_two_renewal_cashflow(
                initial_gauge=6_000,
                gauge_max=10_000,
                initial_level=9,
                start_tick=0,
                final_tick=5,
                events=wrong_level_penalty,
            )

    def test_unresolved_second_renewal_stays_incomplete(self) -> None:
        replay = _baseline_cashflow(second=False)

        self.assertEqual(replay.renewal_ticks, (2,))
        self.assertTrue(replay.cycles[0].completed)
        self.assertFalse(replay.cycles[1].completed)
        self.assertFalse(replay.terminal_before_second)
        self.assertEqual(replay.final_replayed_gauge, 5_990)


class EventMPCBarrierTests(unittest.TestCase):
    def test_action_identical_candidate_abstains_to_frozen_v5(self) -> None:
        incumbent = _decision()
        policy = ModelBarrierPolicy(
            _BasePolicy(incumbent),
            lambda observed, decision: (
                _candidate(0, incumbent),
                _candidate(1, _decision("physically identical")),
            ),
            lambda base, observed, old, new: True,
            _OrdinalCertificateModel(),  # type: ignore[arg-type]
        )
        policy.reset(7)

        self.assertIs(policy.predict(_observation()), incumbent)
        self.assertEqual(
            policy.statistics()["action_tie_abstentions"], 1
        )

    def test_action_tie_does_not_consume_proposal_audit(self) -> None:
        incumbent = _decision()
        calls: list[tuple[object, ...]] = []
        policy = ModelBarrierPolicy(
            _BasePolicy(incumbent),
            lambda observed, decision: (
                _candidate(0, incumbent),
                _candidate(1, _decision("physically identical")),
            ),
            lambda base, observed, old, new: True,
            _OrdinalCertificateModel(),  # type: ignore[arg-type]
            proposal_audit_hook=lambda *args: calls.append(args),
        )
        policy.reset(7)

        self.assertIs(policy.predict(_observation()), incumbent)
        self.assertEqual(calls, [])
        self.assertEqual(
            policy.statistics().get("proposal_audit_calls", 0), 0
        )

    def test_failed_commit_never_marks_an_unsafe_execution(self) -> None:
        incumbent = _decision()
        committed: list[object] = []
        policy = ModelBarrierPolicy(
            _BasePolicy(incumbent),
            lambda observed, decision: (
                _candidate(0, incumbent),
                _candidate(1),
            ),
            lambda base, observed, old, new: False,
            _OrdinalCertificateModel(),  # type: ignore[arg-type]
            maximum_overrides=1,
            audit_hook=lambda *args: {"safe": False},
            audit_commit_hook=committed.append,
        )
        policy.reset(7)

        self.assertIs(policy.predict(_observation()), incumbent)
        self.assertEqual(committed, [])
        self.assertEqual(policy.statistics()["audited_proposals"], 1)
        self.assertEqual(policy.statistics().get("overrides", 0), 0)
        self.assertIs(policy.predict(_observation()), incumbent)
        self.assertEqual(policy.statistics()["query_cap_abstentions"], 1)

    def test_faster_second_renewal_is_not_a_survival_regression(self) -> None:
        incumbent = _outcome(0, score_gain=0, surplus=100)
        faster = _outcome(1, score_gain=1, surplus=100)
        object.__setattr__(incumbent, "survival_ticks", 800)
        object.__setattr__(faster, "survival_ticks", 100)
        result = ExactSearchResult(
            "query",
            "snapshot",
            (_candidate(0), _candidate(1)),
            (incumbent, faster),
            3,
            0.0,
            0.0,
        )

        self.assertFalse(faster.catastrophic_against(incumbent))
        self.assertTrue(result.safe(faster))

    def test_candidate_order_does_not_change_action_equivalence(self) -> None:
        incumbent = _outcome(0, score_gain=0, surplus=100)
        safe = _outcome(2, score_gain=10, surplus=100)
        equivalent = _outcome(1, score_gain=1_000, surplus=100)
        object.__setattr__(
            equivalent, "action_equivalent_to_incumbent", True
        )
        incumbent_candidate = _candidate(0)
        safe_candidate = _candidate(2)
        equivalent_candidate = _candidate(1)
        result = ExactSearchResult(
            "permuted",
            "snapshot",
            (
                incumbent_candidate,
                safe_candidate,
                equivalent_candidate,
            ),
            (incumbent, safe, equivalent),
            4,
            0.0,
            0.0,
        )

        self.assertTrue(result.safe(safe))
        self.assertFalse(result.safe(equivalent))
        self.assertEqual(result.selected_ordinal, 2)
        self.assertIs(result.candidate_for_ordinal(2), safe_candidate)
        self.assertIs(result.outcome_for_ordinal(1), equivalent)

    def test_certification_is_invariant_to_appended_unsafe_candidate(self) -> None:
        incumbent = _decision()
        incumbent_candidate = _candidate(0, incumbent)
        safe = _candidate(1)
        unsafe = _candidate(2)
        observation = _observation(
            bodies=(_piece(11, rot_timer=0), _piece(22, rot_timer=0))
        )

        def run(candidates: tuple[SimpleNamespace, ...]) -> tuple[
            SteeringDecision, dict[str, int | float | list[int]]
        ]:
            policy = ModelBarrierPolicy(
                _BasePolicy(incumbent),
                lambda observed, decision: candidates,
                lambda base, observed, old, new: True,
                _OrdinalCertificateModel(),  # type: ignore[arg-type]
            )
            policy.reset(7)
            return policy.predict(observation), policy.statistics()

        selected_before, stats_before = run((incumbent_candidate, safe))
        selected_after, stats_after = run(
            (incumbent_candidate, safe, unsafe)
        )
        unsafe_first = _candidate(1)
        unsafe_first.geometry.side_sizes = 0.7
        safe_reindexed = _candidate(2)
        safe_reindexed.geometry.side_sizes = 0.6
        selected_permuted, stats_permuted = run(
            (incumbent_candidate, unsafe_first, safe_reindexed)
        )

        self.assertEqual(selected_before, safe.decision)
        self.assertEqual(selected_after, safe.decision)
        self.assertEqual(selected_permuted, safe_reindexed.decision)
        self.assertEqual(stats_before["certified_candidates"], 1)
        self.assertEqual(stats_after["certified_candidates"], 1)
        self.assertEqual(stats_permuted["certified_candidates"], 1)
        self.assertEqual(
            stats_after["rejected/absolute_solvency_lower_bound"], 1
        )

    def test_incomplete_two_renewal_support_is_rejected(self) -> None:
        incumbent = _outcome(0)
        incomplete = _outcome(1, complete=False, score_gain=1_000)
        result = ExactSearchResult(
            "incomplete",
            "snapshot",
            (object(), object()),
            (incumbent, incomplete),
            3,
            0.0,
            0.0,
        )
        model = _fitted_model(second_renewal=0.0)
        _prediction, certificate = model.certify(
            tuple(0.0 for _ in FEATURE_NAMES)
        )

        self.assertFalse(result.safe(incomplete))
        self.assertEqual(result.selected_ordinal, 0)
        self.assertFalse(certificate.certified)
        self.assertIn("two_renewal_neighbor_support", certificate.reasons)

    def test_catastrophic_target_is_available_for_false_safe_accounting(
        self,
    ) -> None:
        incumbent = _outcome(0)
        catastrophic = _outcome(1, complete=False, score_gain=1_000)

        self.assertTrue(catastrophic.catastrophic_against(incumbent))
        self.assertEqual(
            catastrophic.target_vector(incumbent)[
                TARGET_NAMES.index("catastrophic")
            ],
            1.0,
        )
        self.assertEqual(len(TARGET_NAMES), len(set(TARGET_NAMES)))
        self.assertEqual(
            len(catastrophic.target_vector(incumbent)),
            len(TARGET_NAMES),
        )

    def test_fitted_model_serialization_round_trip(self) -> None:
        model = _fitted_model()
        self.assertEqual(len(model.fit_seeds), 20)
        self.assertEqual(len(model.calibration_seeds), 20)
        self.assertTrue(
            set(model.fit_seeds).isdisjoint(model.calibration_seeds)
        )
        self.assertTrue(np.isfinite(model.conformal_q))
        features = tuple(0.0 for _ in FEATURE_NAMES)
        original_prediction, original_certificate = model.certify(features)
        payload = json.loads(json.dumps(model.manifest()))

        restored = KNNEventWorldModel.from_manifest(payload)
        restored_prediction, restored_certificate = restored.certify(features)

        self.assertEqual(restored.manifest(), payload)
        self.assertEqual(restored_prediction, original_prediction)
        self.assertEqual(restored_certificate, original_certificate)

    def test_provisional_and_nonfinite_models_round_trip_fail_closed(
        self,
    ) -> None:
        examples = tuple(
            ModelExample(
                split="barrier-calibration",
                seed=index,
                query_id=f"small-{index}",
                decision_tick=100,
                candidate_ordinal=1,
                features=tuple(0.0 for _ in FEATURE_NAMES),
                targets=_targets(),
                outcome={},
            )
            for index in range(8)
        )
        config = EventMPCConfig(neighbor_count=3)

        provisional = KNNEventWorldModel(config)
        provisional.fit_provisional(examples)
        provisional_payload = json.loads(
            json.dumps(provisional.manifest(), allow_nan=False)
        )
        restored_provisional = KNNEventWorldModel.from_manifest(
            provisional_payload
        )
        self.assertEqual(restored_provisional.manifest(), provisional_payload)
        self.assertEqual(restored_provisional.conformal_rank, 0)
        self.assertEqual(restored_provisional.conformal_records, ())
        self.assertEqual(restored_provisional.conformal_q, 0.0)

        nonfinite = KNNEventWorldModel(config)
        nonfinite.fit(examples)
        self.assertTrue(np.isinf(nonfinite.conformal_q))
        nonfinite_payload = json.loads(
            json.dumps(nonfinite.manifest(), allow_nan=False)
        )
        restored_nonfinite = KNNEventWorldModel.from_manifest(
            nonfinite_payload
        )
        self.assertTrue(np.isinf(restored_nonfinite.conformal_q))
        _prediction, certificate = restored_nonfinite.certify(
            tuple(0.0 for _ in FEATURE_NAMES)
        )
        self.assertFalse(certificate.certified)

    def test_serialized_conformal_evidence_is_internally_consistent(
        self,
    ) -> None:
        payload = json.loads(json.dumps(_fitted_model().manifest()))
        records = payload["conformal_records"]
        self.assertEqual(
            {value["seed"] for value in records},
            set(payload["calibration_seeds"]),
        )
        self.assertEqual(
            sum(value["candidate_count"] for value in records),
            len(payload["calibration_residuals"]),
        )

        mutations = {}
        changed = copy.deepcopy(payload)
        changed["version"] = "foreign"
        mutations["version"] = changed
        changed = copy.deepcopy(payload)
        changed["conformal_records"][1]["seed"] = changed[
            "conformal_records"
        ][0]["seed"]
        mutations["duplicate seed"] = changed
        changed = copy.deepcopy(payload)
        changed["conformal_records"][0]["candidate_count"] += 1
        mutations["candidate count"] = changed
        changed = copy.deepcopy(payload)
        changed["calibration_residuals"].pop()
        mutations["residual count"] = changed
        changed = copy.deepcopy(payload)
        changed["ordered_episode_residuals"][0] += 1.0
        mutations["ordered residuals"] = changed
        changed = copy.deepcopy(payload)
        changed["lower_offsets"][TARGET_NAMES.index("b2")] += 12_345.0
        mutations["lower offsets"] = changed
        changed = copy.deepcopy(payload)
        changed["calibration_residuals"][0][
            TARGET_NAMES.index("b2")
        ] += 12_345.0
        mutations["residual contents"] = changed
        changed = copy.deepcopy(payload)
        for record in changed["conformal_records"]:
            record["r_j"] = -1_000.0
        changed["ordered_episode_residuals"] = [
            -1_000.0
            for _ in changed["ordered_episode_residuals"]
        ]
        changed["conformal_q"] = -1_000.0
        mutations["forged conformal bound"] = changed

        for name, changed in mutations.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                KNNEventWorldModel.from_manifest(changed)


if __name__ == "__main__":
    unittest.main()
