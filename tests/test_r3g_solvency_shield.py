from __future__ import annotations

from dataclasses import replace
from itertools import permutations
import unittest

from irisu_env import EventKind
from irisu_pointer.joint_planner import (
    COMPACT_GEOMETRY,
    JointCandidate,
    PairOption,
)
from irisu_pointer.solvency_shield import (
    CandidateLocalSolvencySearch,
    CandidateSolvencyOutcome,
    MarginPoint,
    RotLiabilityLedger,
    SCORE_RESIDUAL_FEATURES,
    ScoreResidualModel,
    SolvencySearchResult,
    certify_candidate,
    passive_drain_unit,
    renewal_epoch,
    rot_penalty,
    select_with_score_residual,
    visible_liability_ids,
)
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import SemanticAction


NORMAL_RENEWAL = "normal burst landing"
SPECIAL_RENEWAL = "special color clear"
ROT_PENALTY = "normal rot penalty"


def _body(
    identifier: int,
    *,
    lifecycle: str = "dynamic_fresh",
    timer: int = 0,
    kind: str = "piece",
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": kind,
        "lifecycle": lifecycle,
        "rot_timer": timer,
    }


def _observation(
    tick: int,
    bodies: tuple[dict[str, object], ...],
    *,
    level: int = 1,
) -> dict[str, object]:
    return {"tick": tick, "level": level, "bodies": bodies}


def _event(kind: EventKind, **values: object) -> dict[str, object]:
    return {"kind": int(kind), **values}


def _candidate(ordinal: int) -> JointCandidate:
    pair = PairOption(
        10,
        20,
        1,
        "fixture",
        SteeringIntent.STEER_MATCH,
        2.0,
        ordinal == 0,
    )
    decision = SteeringDecision(
        SemanticAction.strong(0.25 + 0.01 * ordinal, 0.5),
        SteeringIntent.STEER_MATCH,
        source_body_id=10,
        destination_body_id=20,
        destination_chain_id=1,
    )
    return JointCandidate(
        ordinal,
        ordinal,
        0,
        pair,
        COMPACT_GEOMETRY[0],
        decision,
    )


def _outcome(
    ordinal: int,
    minimum_margin: int,
    *,
    score_gain: int = 10,
    hard_valid: bool = True,
) -> CandidateSolvencyOutcome:
    candidate = _candidate(ordinal)
    final_gauge = max(1, minimum_margin + 1) if hard_valid else 0
    point = MarginPoint(
        20,
        final_gauge,
        2,
        0,
        0,
        minimum_margin,
        2,
        score_gain,
        1,
        int(not hard_valid),
    )
    return CandidateSolvencyOutcome(
        candidate=candidate,
        start_tick=10,
        end_tick=20,
        initial_gauge=100,
        gauge_max=1_000,
        initial_level=1,
        initial_liabilities=0,
        resolved_two_renewals=True,
        renewal_ticks=(14, 20),
        gross_renewal=500,
        minimum_margin=minimum_margin,
        final_margin=minimum_margin,
        final_gauge=final_gauge,
        final_level=2,
        score_gain=score_gain,
        qualifying_clear_gain=1,
        cleared_events=1,
        rotten_events=int(not hard_valid),
        invalid_actions=0,
        game_over=not hard_valid,
        terminated=not hard_valid,
        truncated=False,
        continuation_rebound=True,
        ledger={
            "outstanding": [],
            "paid": [],
            "discharged": [],
            "ever_seen": [],
            "additions": 0,
        },
        margin_curve=(point,),
    )


class GaugeRecurrenceTests(unittest.TestCase):
    def test_half_boundary_and_level_repricing(self) -> None:
        self.assertEqual(
            [passive_drain_unit(level) for level in (9, 10, 99, 100)],
            [1, 2, 10, 10],
        )
        self.assertEqual(
            [rot_penalty(level) for level in (9, 10, 99, 100)],
            [1_980, 2_000, 3_780, 3_780],
        )

        for level, gauge, expected in (
            (9, 500, 499),
            (9, 501, 498),
            (10, 500, 498),
            (10, 501, 495),
        ):
            with self.subTest(level=level, gauge=gauge):
                CandidateLocalSolvencySearch._check_gauge_recurrence(
                    {"gauge": gauge},
                    {"gauge": expected, "gauge_max": 1_000, "level": level},
                    (
                        _event(
                            EventKind.GAUGE_CHANGED,
                            value=expected - gauge,
                            detail="scene clamp and passive drain",
                        ),
                    ),
                )

        ledger = RotLiabilityLedger()
        ledger.observe_initial(
            _observation(
                100,
                (_body(1, timer=1), _body(2, timer=1)),
            )
        )
        self.assertEqual(ledger.reserve(9), 2 * 1_980)
        self.assertEqual(ledger.reserve(10), 2 * 2_000)

        CandidateLocalSolvencySearch._check_gauge_recurrence(
            {"gauge": 1},
            {"gauge": 1, "gauge_max": 1_000, "level": 1},
            (),
        )
        with self.assertRaisesRegex(RuntimeError, "after drain"):
            CandidateLocalSolvencySearch._check_gauge_recurrence(
                {"gauge": 100},
                {"gauge": 109, "gauge_max": 1_000, "level": 1},
                (
                    _event(
                        EventKind.GAUGE_CHANGED,
                        value=-1,
                        detail="scene clamp and passive drain",
                    ),
                    _event(
                        EventKind.GAUGE_CHANGED,
                        value=10,
                        detail=NORMAL_RENEWAL,
                    ),
                ),
            )
        with self.assertRaisesRegex(RuntimeError, "more than once"):
            CandidateLocalSolvencySearch._check_gauge_recurrence(
                {"gauge": 100},
                {"gauge": 98, "gauge_max": 1_000, "level": 1},
                (
                    _event(
                        EventKind.GAUGE_CHANGED,
                        value=-1,
                        detail="scene clamp and passive drain",
                    ),
                    _event(
                        EventKind.GAUGE_CHANGED,
                        value=-1,
                        detail="scene clamp and passive drain",
                    ),
                ),
            )


class RotLiabilityTests(unittest.TestCase):
    def test_visible_filters_and_body_order(self) -> None:
        bodies = (
            _body(5, timer=1),
            _body(2, lifecycle="confirmed", timer=40),
            _body(3, lifecycle="rotten", timer=40),
            _body(4, lifecycle="deleted", timer=40),
            _body(6, timer=0),
            _body(7, kind="projectile", timer=40),
        )
        forward = _observation(100, bodies)
        reverse = _observation(100, tuple(reversed(bodies)))
        self.assertEqual(visible_liability_ids(forward), frozenset({2, 5}))
        self.assertEqual(visible_liability_ids(reverse), frozenset({2, 5}))

        left, right = RotLiabilityLedger(), RotLiabilityLedger()
        left.observe_initial(forward)
        right.observe_initial(reverse)
        self.assertEqual(left.manifest(), right.manifest())
        self.assertEqual(left.outstanding, {2: 101, 5: 140})

    def test_payment_once_discharge_and_new_timer(self) -> None:
        initial = _observation(
            100,
            (_body(7, timer=40), _body(8, timer=10)),
            level=9,
        )
        paid_tick = _observation(
            101,
            (
                _body(7, lifecycle="rotten", timer=41),
                _body(8, timer=11),
                _body(9, timer=1),
            ),
            level=10,
        )
        payment = (
            _event(EventKind.ROTTEN, a=7),
            _event(
                EventKind.GAUGE_CHANGED,
                a=7,
                value=-2_000,
                detail=ROT_PENALTY,
            ),
        )
        ledger = RotLiabilityLedger()
        ledger.observe_initial(initial)
        ledger.reconcile_tick(initial, paid_tick, payment)
        self.assertEqual(ledger.paid, {7: 101})
        self.assertEqual(ledger.outstanding, {8: 131, 9: 141})
        self.assertEqual(ledger.additions, 3)

        discharged_tick = _observation(102, (_body(9, timer=2),), level=10)
        ledger.reconcile_tick(paid_tick, discharged_tick, ())
        self.assertEqual(ledger.discharged, {8: 102})
        self.assertEqual(ledger.outstanding, {9: 141})

        with self.assertRaisesRegex(RuntimeError, "outstanding liability"):
            ledger.reconcile_tick(paid_tick, paid_tick, payment)
        reappeared = _observation(
            103,
            (_body(8, timer=1), _body(9, timer=3)),
            level=10,
        )
        with self.assertRaisesRegex(RuntimeError, "resolved rot liability"):
            ledger.reconcile_tick(discharged_tick, reappeared, ())

    def test_absolute_tick_offset_does_not_change_liability_margin(self) -> None:
        manifests = []
        reserves = []
        for offset in (0, 10_000):
            initial = _observation(
                100 + offset,
                (_body(7, timer=40), _body(8, timer=10)),
                level=9,
            )
            paid_tick = _observation(
                101 + offset,
                (
                    _body(7, lifecycle="rotten", timer=41),
                    _body(8, timer=11),
                ),
                level=10,
            )
            ledger = RotLiabilityLedger()
            ledger.observe_initial(initial)
            ledger.reconcile_tick(
                initial,
                paid_tick,
                (
                    _event(EventKind.ROTTEN, a=7),
                    _event(
                        EventKind.GAUGE_CHANGED,
                        a=7,
                        value=-2_000,
                        detail=ROT_PENALTY,
                    ),
                ),
            )
            manifests.append(ledger.manifest())
            reserves.append(ledger.reserve(10))
        self.assertEqual(reserves, [2_000, 2_000])
        self.assertEqual(
            [
                row["due_tick"] - manifest["paid"][0]["tick"]
                for manifest in manifests
                for row in manifest["outstanding"]
            ],
            [30, 30],
        )

    def test_payment_deadline_order_and_value_are_exact(self) -> None:
        initial = _observation(100, (_body(7, timer=40),), level=9)
        due = _observation(
            101,
            (_body(7, lifecycle="rotten", timer=41),),
            level=10,
        )
        correct_penalty = _event(
            EventKind.GAUGE_CHANGED,
            a=7,
            value=-2_000,
            detail=ROT_PENALTY,
        )
        malformed = (
            (_event(EventKind.ROTTEN, a=7),),
            (correct_penalty, _event(EventKind.ROTTEN, a=7)),
            (
                _event(EventKind.ROTTEN, a=7),
                {**correct_penalty, "a": 8},
            ),
            (
                _event(EventKind.ROTTEN, a=7),
                {**correct_penalty, "value": -1_980},
            ),
            (
                _event(EventKind.ROTTEN, a=7),
                {**correct_penalty, "detail": "wrong"},
            ),
        )
        for events in malformed:
            with self.subTest(events=events):
                ledger = RotLiabilityLedger()
                ledger.observe_initial(initial)
                with self.assertRaises(RuntimeError):
                    ledger.reconcile_tick(initial, due, events)

        late = _observation(
            102,
            (_body(7, lifecycle="rotten", timer=42),),
            level=10,
        )
        ledger = RotLiabilityLedger()
        ledger.observe_initial(initial)
        with self.assertRaisesRegex(RuntimeError, "exact deadline"):
            ledger.reconcile_tick(
                initial,
                late,
                (_event(EventKind.ROTTEN, a=7), correct_penalty),
            )

    def test_cleared_event_alone_does_not_cancel_rot_debt(self) -> None:
        initial = _observation(100, (_body(7, timer=10),), level=9)
        after = _observation(101, (_body(7, timer=11),), level=9)
        ledger = RotLiabilityLedger()
        ledger.observe_initial(initial)
        ledger.reconcile_tick(
            initial,
            after,
            (_event(EventKind.CLEARED, a=7),),
        )
        self.assertEqual(ledger.outstanding, {7: 131})
        self.assertEqual(ledger.discharged, {})


class RenewalTests(unittest.TestCase):
    def test_special_gain_is_anchored_and_coalesced_per_tick(self) -> None:
        anchor = _event(EventKind.CLEARED, detail=SPECIAL_RENEWAL)
        normal = _event(
            EventKind.GAUGE_CHANGED,
            value=30,
            detail=NORMAL_RENEWAL,
        )
        merged = (
            anchor,
            _event(
                EventKind.GAUGE_CHANGED,
                value=200,
                detail=SPECIAL_RENEWAL,
            ),
            normal,
        )
        split = (
            normal,
            _event(
                EventKind.GAUGE_CHANGED,
                value=80,
                detail=SPECIAL_RENEWAL,
            ),
            anchor,
            _event(
                EventKind.GAUGE_CHANGED,
                value=120,
                detail=SPECIAL_RENEWAL,
            ),
        )
        self.assertEqual(renewal_epoch(merged), (True, 230))
        self.assertEqual(renewal_epoch(split), (True, 230))
        self.assertEqual(
            renewal_epoch((_event(EventKind.CLEARED, detail="rotten"),)),
            (False, 0),
        )
        with self.assertRaisesRegex(RuntimeError, "causal clear"):
            renewal_epoch(
                (
                    _event(
                        EventKind.GAUGE_CHANGED,
                        value=200,
                        detail=SPECIAL_RENEWAL,
                    ),
                )
            )


class CertificateTests(unittest.TestCase):
    @staticmethod
    def _certificates(
        outcomes: tuple[CandidateSolvencyOutcome, ...],
    ) -> dict[int, dict[str, object]]:
        incumbent = next(
            value for value in outcomes if value.candidate.ordinal == 0
        )
        return {
            value.candidate.ordinal: certify_candidate(
                value, incumbent
            ).manifest()
            for value in outcomes
        }

    def test_candidate_local_under_permutation_and_unsafe_append(self) -> None:
        incumbent = _outcome(0, 10)
        safe = _outcome(1, 20, score_gain=20)
        unsafe = _outcome(2, -5, score_gain=1_000, hard_valid=False)
        expected = self._certificates((incumbent, safe))[1]
        self.assertTrue(expected["eligible"])

        for ordered in permutations((incumbent, safe, unsafe)):
            with self.subTest(order=[x.candidate.ordinal for x in ordered]):
                actual = self._certificates(ordered)
                self.assertEqual(actual[1], expected)
                self.assertFalse(actual[2]["eligible"])
        self.assertEqual(
            self._certificates((incumbent, safe, unsafe))[1],
            self._certificates((incumbent, safe))[1],
        )

    def test_candidate_zero_and_exact_delta_ties_abstain(self) -> None:
        incumbent = _outcome(0, 10, score_gain=10)
        same_margin = _outcome(1, 10, score_gain=100)
        zero = certify_candidate(incumbent, incumbent)
        tied = certify_candidate(same_margin, incumbent)
        self.assertFalse(zero.eligible)
        self.assertIn("frozen_v5_tie", zero.reasons)
        self.assertIn("delta_b2_tie", zero.reasons)
        self.assertFalse(tied.eligible)
        self.assertEqual(tied.reasons, ("delta_b2_tie",))

        unresolved = replace(same_margin, resolved_two_renewals=False)
        unresolved_certificate = certify_candidate(unresolved, incumbent)
        self.assertFalse(unresolved_certificate.eligible)
        self.assertIn(
            "candidate_hard_invalid", unresolved_certificate.reasons
        )

    def test_residual_selection_is_order_invariant_and_ties_abstain(self) -> None:
        incumbent = _outcome(0, 10)
        safe = _outcome(1, 20)
        unsafe = _outcome(2, -5, hard_valid=False)
        outcomes = (incumbent, safe, unsafe)
        certificates = tuple(
            certify_candidate(value, incumbent) for value in outcomes
        )
        weights = [0.0] * (len(SCORE_RESIDUAL_FEATURES) + 1)
        weights[1 + SCORE_RESIDUAL_FEATURES.index(
            "minimum_margin_fraction"
        )] = 1.0
        model = ScoreResidualModel(
            SCORE_RESIDUAL_FEATURES,
            (0.0,) * len(SCORE_RESIDUAL_FEATURES),
            (1.0,) * len(SCORE_RESIDUAL_FEATURES),
            tuple(weights),
            1e-3,
            1,
            "0" * 64,
        )
        identities = set()
        for order in permutations(range(3)):
            search = SolvencySearchResult(
                "a" * 64,
                "b" * 64,
                "c" * 64,
                "d" * 64,
                tuple(outcomes[index] for index in order),
                tuple(certificates[index] for index in reversed(order)),
                0,
                4,
                0.0,
                0.0,
            )
            selection = select_with_score_residual(search, model)
            self.assertEqual(selection.proposed_ordinal, 1)
            self.assertEqual(selection.selected_ordinal, 1)
            self.assertIs(selection.decision, safe.candidate.decision)
            identities.add(selection.sha256)
        self.assertEqual(len(identities), 1)

        unsafe_weights = [0.0] * (
            len(SCORE_RESIDUAL_FEATURES) + 1
        )
        unsafe_weights[0] = 0.01
        unsafe_weights[
            1 + SCORE_RESIDUAL_FEATURES.index("rotten_events_10")
        ] = 10.0
        unsafe_model = replace(model, weights=tuple(unsafe_weights))
        for order in permutations(range(3)):
            search = SolvencySearchResult(
                "a" * 64,
                "b" * 64,
                "c" * 64,
                "d" * 64,
                tuple(outcomes[index] for index in order),
                tuple(certificates[index] for index in reversed(order)),
                0,
                4,
                0.0,
                0.0,
            )
            selection = select_with_score_residual(
                search, unsafe_model
            )
            self.assertEqual(selection.proposed_ordinal, 2)
            self.assertEqual(selection.selected_ordinal, 1)

        tied_safe = _outcome(2, 20)
        tied_outcomes = (incumbent, safe, tied_safe)
        tied_search = SolvencySearchResult(
            "a" * 64,
            "b" * 64,
            "c" * 64,
            "d" * 64,
            tied_outcomes,
            tuple(
                certify_candidate(value, incumbent)
                for value in tied_outcomes
            ),
            0,
            4,
            0.0,
            0.0,
        )
        tied_selection = select_with_score_residual(tied_search, model)
        self.assertEqual(tied_selection.proposed_ordinal, 0)
        self.assertEqual(tied_selection.selected_ordinal, 0)

    def test_outcome_hash_is_compact_canonical_and_material(self) -> None:
        outcome = _outcome(1, 20, score_gain=20)
        reordered_ledger = dict(reversed(tuple(outcome.ledger.items())))
        self.assertEqual(
            outcome.sha256,
            replace(outcome, ledger=reordered_ledger).sha256,
        )
        self.assertNotEqual(
            outcome.sha256,
            replace(outcome, score_gain=21).sha256,
        )
        self.assertEqual(
            outcome.sha256,
            "a9d0e42227efd9c9b5824288f772d891e5426d6e4aa0d9c651791fd1cb324736",
        )


if __name__ == "__main__":
    unittest.main()
