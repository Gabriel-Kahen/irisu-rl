from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import EventKind
from irisu_pointer import conservative_offline as offline
from irisu_pointer import joint_planner
from irisu_pointer.joint_planner import (
    JointBranchOutcome,
    JointMilestoneOutcome,
)
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import SemanticAction


TRUSTED_JOINT = Path(
    "/home/gabe/.codex/worktrees/e2f8/irisu/"
    "python/irisu_pointer/joint_planner.py"
)
TRUSTED_JOINT_SHA256 = (
    "dc7009fc18a322eca5dace55b9baf982b6ced26c18517af752aab0f6365d362e"
)


def _event(
    tick: int,
    sequence: int,
    kind: EventKind,
    *,
    value: int = 0,
    body_id: int = -1,
    detail: str = "",
    level: int,
) -> dict[str, object]:
    return {
        "tick": tick,
        "sequence": sequence,
        "kind": int(kind),
        "value": value,
        "a": body_id,
        "detail": detail,
        "level": level,
    }


def _body(identifier: int, timer: int) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "lifecycle": "dynamic_fresh",
        "rot_timer": timer,
    }


def _two_renewal_trace(
    *,
    tick_offset: int = 0,
    split_first_gain: bool = False,
    reverse_events: bool = False,
    reverse_bodies: bool = False,
) -> offline.BranchTrace:
    t0 = 100 + tick_offset
    first_gain = (
        [
            _event(
                t0 + 1,
                0,
                EventKind.GAUGE_CHANGED,
                value=400,
                detail="normal burst landing",
                level=9,
            ),
            _event(
                t0 + 1,
                1,
                EventKind.GAUGE_CHANGED,
                value=600,
                detail="normal burst landing",
                level=9,
            ),
        ]
        if split_first_gain
        else [
            _event(
                t0 + 1,
                0,
                EventKind.GAUGE_CHANGED,
                value=1000,
                detail="normal burst landing",
                level=9,
            )
        ]
    )
    shift = int(split_first_gain)
    events = [
        *first_gain,
        _event(
            t0 + 1,
            1 + shift,
            EventKind.CLEARED,
            body_id=10,
            level=9,
        ),
        _event(
            t0 + 1,
            2 + shift,
            EventKind.GAUGE_CHANGED,
            value=-503,
            detail="scene clamp and passive drain",
            level=9,
        ),
        _event(
            t0 + 1,
            3 + shift,
            EventKind.ROTTEN,
            body_id=10,
            level=9,
        ),
        _event(
            t0 + 1,
            4 + shift,
            EventKind.GAUGE_CHANGED,
            value=-(1800 + 20 * 9),
            body_id=10,
            detail="normal rot penalty",
            level=9,
        ),
        _event(
            t0 + 2,
            0,
            EventKind.LEVEL_CHANGED,
            value=10,
            level=10,
        ),
        _event(
            t0 + 2,
            1,
            EventKind.GAUGE_CHANGED,
            value=1000,
            detail="special color clear",
            level=10,
        ),
        _event(
            t0 + 2,
            2,
            EventKind.GAUGE_CHANGED,
            value=-6,
            detail="scene clamp and passive drain",
            level=10,
        ),
        _event(
            t0 + 2,
            3,
            EventKind.ROTTEN,
            body_id=20,
            level=10,
        ),
        _event(
            t0 + 2,
            4,
            EventKind.GAUGE_CHANGED,
            value=-(1800 + 20 * 10),
            body_id=20,
            detail="normal rot penalty",
            level=10,
        ),
    ]
    bodies = [_body(10, 40), _body(20, 39)]
    return offline.BranchTrace(
        {
            "tick": t0,
            "gauge": 3500,
            "gauge_max": 4000,
            "level": 9,
            "bodies": list(reversed(bodies)) if reverse_bodies else bodies,
        },
        {
            "tick": t0 + 2,
            "gauge": 1011,
            "gauge_max": 4000,
            "level": 10,
            "bodies": (),
        },
        tuple(reversed(events)) if reverse_events else tuple(events),
    )


def _solvency_signature(value: offline.SolvencyOutcome) -> tuple[object, ...]:
    return (
        value.minimum_surplus,
        value.minimum_gauge,
        value.renewal_count,
        value.first_renewal_ticks,
        value.second_renewal_ticks,
        value.liability_ids,
        value.paid_liability_ids,
        value.emergent_paid_ids,
        value.censored_before_second_renewal,
        value.negative_through_renewal,
    )


def _milestone(
    *, gauge: int, score: int, alive: bool = True
) -> JointMilestoneOutcome:
    return JointMilestoneOutcome(
        horizon_ticks=2,
        alive=alive,
        survival_ticks=2 if alive else 1,
        score_gain=score,
        final_gauge=gauge,
        final_level=10,
        qualifying_clear_gain=1,
        cleared_events=1,
        rotten_events=0,
        positive_gauge_renewal=1,
        invalid_actions=0,
        intended_source_hits=1,
        intended_pair_joined=True,
        pair_closure_sizes=0.0,
    )


def _outcome(*, gauge: int, score: int, alive: bool = True) -> JointBranchOutcome:
    return JointBranchOutcome(
        candidate=None,  # branch_labels deliberately has no cross-candidate input
        milestones=(_milestone(gauge=gauge, score=score, alive=alive),),
        simulated_ticks=2,
    )


def _solvency(
    minimum: int, *, censored: bool = False
) -> offline.SolvencyOutcome:
    return offline.SolvencyOutcome(
        minimum_surplus=minimum,
        minimum_gauge=max(minimum, 0),
        renewal_count=1 if censored else 2,
        first_renewal_ticks=1,
        second_renewal_ticks=2,
        liability_ids=(),
        paid_liability_ids=(),
        emergent_paid_ids=(),
        censored_before_second_renewal=censored,
        negative_through_renewal=minimum <= 0,
        event_sha256="0" * 64,
    )


class _BasePolicy:
    def __init__(self, decision: SteeringDecision) -> None:
        self.decision = decision
        self.commits: list[SteeringDecision] = []

    def reset(self, seed: int = 0) -> None:
        del seed

    def predict(self, observation: object) -> SteeringDecision:
        del observation
        return self.decision

    def commit_external_decision(
        self, observation: object, decision: SteeringDecision
    ) -> None:
        del observation
        self.commits.append(decision)


class _Ensemble:
    def predict_full(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        count = features.shape[0]
        return {
            "delta_cvar_mean": torch.ones(count),
            "delta_cvar_std": torch.zeros(count),
            "absolute_cvar_mean": torch.ones(count),
            "absolute_cvar_std": torch.zeros(count),
            "unsafe_mean": torch.zeros(count),
            "score_mean": torch.ones(count),
            "score_std": torch.zeros(count),
        }


def _decision(x: float, *, reason: str) -> SteeringDecision:
    return SteeringDecision(
        SemanticAction.strong(x, 0.4),
        SteeringIntent.STEER_MATCH,
        source_body_id=1,
        destination_body_id=2,
        reason=reason,
    )


def _candidate(
    ordinal: int, decision: SteeringDecision
) -> SimpleNamespace:
    return SimpleNamespace(
        ordinal=ordinal,
        decision=decision,
        geometry=SimpleNamespace(name="analytic-strong"),
        pair=SimpleNamespace(category="fresh-match"),
    )


def _calibration() -> offline.BarrierCalibration:
    return offline.BarrierCalibration(
        margin_threshold=0.0,
        probability_threshold=0.1,
        score_threshold=0.0,
        isotonic=offline.IsotonicCalibration((1.0,), (0.0,)),
        report={},
    )


def _benchmark_module():
    name = "r3g_benchmark_test_module"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "benchmarks/rl_r3g_conservative_offline.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("R3G benchmark test module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _AuditedBasePolicy(_BasePolicy):
    def __init__(
        self, decision: SteeringDecision, events: list[str]
    ) -> None:
        super().__init__(decision)
        self.events = events
        self.continuation_state = object()

    def predict(self, observation: object) -> SteeringDecision:
        self.events.append("base.predict")
        return super().predict(observation)

    def commit_external_decision(
        self, observation: object, decision: SteeringDecision
    ) -> None:
        self.events.append("base.commit")
        self.continuation_state = object()
        super().commit_external_decision(observation, decision)


class _StaticSearch:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        result: object | None = None,
        rows: list[dict[str, object]] | None = None,
        error: Exception | None = None,
        rows_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.events = events
        self.result = result
        self.rows = rows
        self.error = error
        self.rows_error = rows_error
        self.records: list[object] = []
        self.calls = 0

    def reset_records(self, seed: int) -> None:
        del seed
        self.records.clear()

    def search(
        self,
        env: object,
        observation: object,
        incumbent: SteeringDecision,
    ) -> object:
        del env, observation, incumbent
        self.calls += 1
        self.events.append(f"{self.name}.search")
        if self.error is not None:
            raise self.error
        if self.result is None:
            raise AssertionError("static search result is absent")
        if self.rows is not None:
            rows = self.rows

            def materialize_rows() -> list[dict[str, object]]:
                if self.rows_error is not None:
                    raise self.rows_error
                return list(rows)

            self.records.append(
                SimpleNamespace(
                    result=self.result,
                    rows=materialize_rows,
                )
            )
        return self.result


def _audited_candidate(
    ordinal: int, decision: SteeringDecision
) -> SimpleNamespace:
    return SimpleNamespace(
        ordinal=ordinal,
        decision=decision,
        pair_ordinal=ordinal,
        geometry_ordinal=ordinal,
        geometry=SimpleNamespace(name="analytic-strong"),
        pair=SimpleNamespace(
            category="fresh-match",
            incumbent=ordinal == 0,
        ),
    )


def _search_result(
    candidates: tuple[SimpleNamespace, ...],
    selected: int,
    *,
    strictly_improved: bool = True,
    top_tie: bool = False,
) -> SimpleNamespace:
    outcomes = []
    for index, candidate in enumerate(candidates):
        objective = (
            (0.0,)
            if index == 0 or not strictly_improved
            else (
                (2.0,)
                if index == selected or (top_tie and index > 0)
                else (1.0,)
            )
        )
        outcomes.append(
            SimpleNamespace(
                candidate=candidate,
                objective=objective,
                selectable_against=lambda _incumbent: True,
            )
        )
    return SimpleNamespace(
        selected_candidate=candidates[selected],
        decision=candidates[selected].decision,
        strictly_improved=strictly_improved,
        outcomes=tuple(outcomes),
        restore_checks=len(candidates),
        simulated_ticks=2 * len(candidates),
        wall_seconds=0.0,
        cpu_seconds=0.0,
    )


def _audit_row(
    candidate: SimpleNamespace,
    labels: dict[str, object],
) -> dict[str, object]:
    incumbent = candidate.ordinal == 0
    risk = float(labels["risk_margin"])
    absolute = float(labels["absolute_solvency"])
    minimum = (
        1_000
        if incumbent
        else (-100 if absolute < 0.0 else (500 if risk < 0.0 else 2_000))
    )
    survival = 500 if bool(labels["catastrophic"]) else 2_000
    return {
        "seed": 7,
        "query_index": 0,
        "tick": 11,
        "search_sha256": "1" * 64,
        "snapshot_sha256": "2" * 64,
        "ordinal": candidate.ordinal,
        "incumbent": incumbent,
        "selected": candidate.ordinal == 1,
        "features": np.zeros(offline.INPUT_DIM, dtype=np.float32),
        "support": np.zeros(1, dtype=np.float32),
        "signature": "known",
        "labels": labels,
        "candidate": {"ordinal": candidate.ordinal},
        "outcome": {
            "milestones": [
                {
                    "survival_ticks": survival,
                    "invalid_actions": 0,
                }
            ]
        },
        "solvency": {
            "minimum_surplus": minimum,
            "censored_before_second_renewal": not bool(
                labels["resolved"]
            ),
            "negative_through_renewal": bool(
                labels["severe_renewable"]
            ),
            "game_over": bool(labels["new_terminal"]),
            "gauge_failure": bool(labels["new_gauge_failure"]),
        },
    }


def _exact_labels(
    *,
    unsafe: bool = False,
    resolved: bool = True,
    risk_margin: float = 0.25,
    absolute_solvency: float = 0.25,
) -> dict[str, object]:
    return {
        "risk_margin": risk_margin,
        "absolute_solvency": absolute_solvency,
        "score_advantage": 1.0,
        "unsafe": unsafe,
        "hard_catastrophe": False,
        "severe_renewable": False,
        "censored": not resolved,
        "resolved": resolved,
        "new_terminal": False,
        "new_gauge_failure": False,
        "catastrophic": False,
    }


def _audited_teacher_harness(
    module: object,
    *,
    teacher_result: object,
    audit_result: object | None,
    audit_rows: list[dict[str, object]] | None,
    audit_error: Exception | None = None,
    audit_rows_error: Exception | None = None,
) -> tuple[object, _AuditedBasePolicy, list[str]]:
    incumbent = teacher_result.outcomes[0].candidate.decision
    events: list[str] = []
    base = _AuditedBasePolicy(incumbent, events)
    teacher_searcher = _StaticSearch(
        "teacher", events, result=teacher_result
    )
    audit_searcher = _StaticSearch(
        "audit",
        events,
        result=audit_result,
        rows=audit_rows,
        error=audit_error,
        rows_error=audit_rows_error,
    )
    policy = module.AuditedTeacherPolicy.__new__(
        module.AuditedTeacherPolicy
    )
    policy.env = object()
    policy.base_policy = base
    policy.teacher_searcher = teacher_searcher
    policy.audit_searcher = audit_searcher
    # Keep the legacy wrapper populated so this test detects its prohibited
    # commit-before-audit flow until AuditedTeacherPolicy is fully migrated.
    policy.teacher = joint_planner.JointTeacherPolicy(
        policy.env,
        base,
        teacher_searcher,
        query_stride_shots=1,
        maximum_queries=48,
    )
    policy.query_stride_shots = 1
    policy.maximum_queries = 48
    policy._counts = Counter()
    policy.counts = policy._counts
    policy._attempts = []
    policy._results = []
    policy.audits = []
    policy.seed = 7
    return policy, base, events


class ConservativeOfflineTests(unittest.TestCase):
    def test_locked_protocol_runtime_and_checkpoint_identities(self) -> None:
        actual = offline.verify_trusted_identities()
        self.assertEqual(
            actual[str(offline.LOCKED_PROTOCOL_PATH)],
            offline.LOCKED_PROTOCOL_SHA256,
        )
        self.assertEqual(
            actual[str(offline.TRUSTED_RUNTIME_PATH)],
            offline.TRUSTED_RUNTIME_SHA256,
        )
        self.assertEqual(
            actual[str(offline.FROZEN_V5_PATH)],
            offline.FROZEN_V5_SHA256,
        )

    def test_joint_planner_import_is_bound_to_exact_trusted_source(self) -> None:
        self.assertEqual(joint_planner._TRUSTED_SOURCE, TRUSTED_JOINT)
        self.assertEqual(joint_planner._TRUSTED_SHA256, TRUSTED_JOINT_SHA256)
        self.assertEqual(
            hashlib.sha256(TRUSTED_JOINT.read_bytes()).hexdigest(),
            TRUSTED_JOINT_SHA256,
        )
        self.assertEqual(
            Path(
                joint_planner.JointPairGeometrySearch.search.__code__.co_filename
            ).resolve(),
            TRUSTED_JOINT.resolve(),
        )

    def test_two_liabilities_and_two_renewals_pay_once_in_event_order(self) -> None:
        result = offline.trace_solvency(
            _two_renewal_trace(reverse_events=True), horizon_ticks=2
        )
        # q1=max(1, clamp(3500+1000)-3)-1980=2017;
        # q2=max(1, 2017+1000-6)-2000=1011.
        self.assertEqual(result.minimum_gauge, 1011)
        self.assertEqual(result.minimum_surplus, 1011)
        self.assertEqual(result.renewal_count, 2)
        self.assertEqual(
            (result.first_renewal_ticks, result.second_renewal_ticks), (1, 2)
        )
        self.assertEqual(result.liability_ids, (10, 20))
        self.assertEqual(result.paid_liability_ids, (10, 20))
        self.assertFalse(result.censored_before_second_renewal)

    def test_duplicate_liability_payment_fails_closed(self) -> None:
        trace = _two_renewal_trace()
        duplicate = (
            _event(
                101,
                5,
                EventKind.ROTTEN,
                body_id=10,
                level=9,
            ),
            _event(
                101,
                6,
                EventKind.GAUGE_CHANGED,
                value=-1980,
                body_id=10,
                detail="normal rot penalty",
                level=9,
            ),
        )
        malformed = offline.BranchTrace(
            trace.initial, trace.final, trace.events + duplicate
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate|more than once"):
            offline.trace_solvency(malformed, horizon_ticks=2)

    def test_trace_is_invariant_to_permutations_offset_and_split_gain(self) -> None:
        expected = _solvency_signature(
            offline.trace_solvency(_two_renewal_trace(), horizon_ticks=2)
        )
        variants = (
            _two_renewal_trace(reverse_events=True),
            _two_renewal_trace(reverse_bodies=True),
            _two_renewal_trace(tick_offset=10_000),
            _two_renewal_trace(split_first_gain=True),
        )
        for trace in variants:
            with self.subTest(trace=trace.initial["tick"], events=len(trace.events)):
                actual = offline.trace_solvency(trace, horizon_ticks=2)
                self.assertEqual(_solvency_signature(actual), expected)

    def test_cleared_does_not_cancel_same_tick_rot(self) -> None:
        result = offline.trace_solvency(_two_renewal_trace(), horizon_ticks=2)
        self.assertIn(10, result.paid_liability_ids)
        self.assertEqual(result.minimum_gauge, 1011)

    def test_entry_failure_cannot_be_rescued_by_same_tick_clear(self) -> None:
        trace = offline.BranchTrace(
            {
                "tick": 0,
                "gauge": 0,
                "gauge_max": 4000,
                "level": 1,
                "bodies": (),
            },
            {"tick": 2, "gauge": 1998, "gauge_max": 4000, "level": 1},
            (
                _event(
                    1,
                    0,
                    EventKind.GAUGE_CHANGED,
                    value=1000,
                    detail="special color clear",
                    level=1,
                ),
                _event(
                    1,
                    1,
                    EventKind.GAUGE_CHANGED,
                    value=-1,
                    detail="scene clamp and passive drain",
                    level=1,
                ),
                _event(
                    2,
                    0,
                    EventKind.GAUGE_CHANGED,
                    value=1000,
                    detail="normal burst landing",
                    level=1,
                ),
                _event(
                    2,
                    1,
                    EventKind.GAUGE_CHANGED,
                    value=-1,
                    detail="scene clamp and passive drain",
                    level=1,
                ),
            ),
        )
        result = offline.trace_solvency(trace, horizon_ticks=2)
        self.assertTrue(result.negative_through_renewal)
        self.assertLessEqual(result.minimum_surplus, 0)

    def test_saturated_gross_gain_is_not_a_renewal(self) -> None:
        trace = offline.BranchTrace(
            {
                "tick": 0,
                "gauge": 4000,
                "gauge_max": 4000,
                "level": 1,
                "bodies": (),
            },
            {"tick": 1, "gauge": 3997, "gauge_max": 4000, "level": 1},
            (
                _event(
                    1,
                    0,
                    EventKind.GAUGE_CHANGED,
                    value=1000,
                    detail="normal burst landing",
                    level=1,
                ),
                _event(
                    1,
                    1,
                    EventKind.GAUGE_CHANGED,
                    value=-1003,
                    detail="scene clamp and passive drain",
                    level=1,
                ),
            ),
        )
        result = offline.trace_solvency(trace, horizon_ticks=1)
        self.assertEqual(result.renewal_count, 0)
        self.assertTrue(result.censored_before_second_renewal)

    def test_live_floor_with_zero_scene_delta_emits_no_gauge_event(self) -> None:
        trace = offline.BranchTrace(
            {
                "tick": 0,
                "gauge": 1,
                "gauge_max": 4000,
                "level": 1,
                "bodies": (),
            },
            {"tick": 1, "gauge": 1, "gauge_max": 4000, "level": 1},
            (),
        )
        result = offline.trace_solvency(trace, horizon_ticks=1)
        self.assertEqual(result.minimum_gauge, 1)
        self.assertFalse(result.gauge_failure)
        self.assertTrue(result.censored_before_second_renewal)

    def test_post_second_renewal_game_over_is_retained(self) -> None:
        source = _two_renewal_trace()
        extra = (
            _event(
                102,
                5,
                EventKind.ROTTEN,
                body_id=30,
                level=10,
            ),
            _event(
                102,
                6,
                EventKind.GAUGE_CHANGED,
                value=-2000,
                body_id=30,
                detail="normal rot penalty",
                level=10,
            ),
            _event(103, 0, EventKind.GAME_OVER, level=10),
            _event(
                103,
                1,
                EventKind.GAUGE_CHANGED,
                value=1000,
                detail="special color clear",
                level=10,
            ),
            _event(
                103,
                2,
                EventKind.GAUGE_CHANGED,
                value=-2,
                detail="scene clamp and passive drain",
                level=10,
            ),
        )
        trace = offline.BranchTrace(
            source.initial,
            {
                "tick": 103,
                "gauge": 9,
                "gauge_max": 4000,
                "level": 10,
                "bodies": (),
            },
            source.events + extra,
        )
        result = offline.trace_solvency(trace, horizon_ticks=3)
        self.assertEqual(result.renewal_count, 2)
        self.assertTrue(result.game_over)
        self.assertTrue(result.gauge_failure)

    def test_prior_tick_clear_cancels_liability_without_readding_it(self) -> None:
        trace = offline.BranchTrace(
            {
                "tick": 0,
                "gauge": 1000,
                "gauge_max": 4000,
                "level": 1,
                "bodies": (_body(10, 39),),
            },
            {"tick": 2, "gauge": 998, "gauge_max": 4000, "level": 1},
            (
                _event(
                    1,
                    0,
                    EventKind.CLEARED,
                    body_id=10,
                    detail="normal burst actor teardown",
                    level=1,
                ),
                _event(
                    1,
                    1,
                    EventKind.GAUGE_CHANGED,
                    value=-1,
                    detail="scene clamp and passive drain",
                    level=1,
                ),
                _event(
                    2,
                    0,
                    EventKind.GAUGE_CHANGED,
                    value=-1,
                    detail="scene clamp and passive drain",
                    level=1,
                ),
            ),
        )
        result = offline.trace_solvency(trace, horizon_ticks=2)
        self.assertEqual(result.cancelled_liability_ids, (10,))
        self.assertEqual(result.paid_liability_ids, ())
        malformed = offline.BranchTrace(
            trace.initial,
            trace.final,
            trace.events
            + (
                _event(
                    2,
                    1,
                    EventKind.ROTTEN,
                    body_id=10,
                    level=1,
                ),
                _event(
                    2,
                    2,
                    EventKind.GAUGE_CHANGED,
                    value=-1820,
                    body_id=10,
                    detail="normal rot penalty",
                    level=1,
                ),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "after cancellation"):
            offline.trace_solvency(malformed, horizon_ticks=2)

    def test_branch_labels_are_candidate_local_and_censor_fails_closed(self) -> None:
        incumbent = _outcome(gauge=1000, score=100)
        candidate = _outcome(gauge=1100, score=150)
        unsafe_irrelevant = _outcome(gauge=1, score=10_000, alive=False)
        base_solvency = _solvency(800)
        candidate_solvency = _solvency(900)

        expected = offline.branch_labels(
            candidate,
            incumbent,
            candidate_solvency,
            base_solvency,
            gauge_max=4000,
        )
        offline.branch_labels(
            unsafe_irrelevant,
            incumbent,
            _solvency(-1),
            base_solvency,
            gauge_max=4000,
        )
        self.assertEqual(
            offline.branch_labels(
                candidate,
                incumbent,
                candidate_solvency,
                base_solvency,
                gauge_max=4000,
            ),
            expected,
        )
        unresolved = offline.branch_labels(
            candidate,
            incumbent,
            _solvency(900, censored=True),
            base_solvency,
            gauge_max=4000,
        )
        self.assertTrue(unresolved["censored"])
        self.assertTrue(unresolved["unsafe"])
        terminal_floor = offline.branch_labels(
            candidate,
            incumbent,
            _solvency(0),
            base_solvency,
            gauge_max=4000,
        )
        self.assertTrue(terminal_floor["unsafe"])

    def test_incumbent_tie_and_out_of_support_actions_abstain(self) -> None:
        incumbent = _decision(0.3, reason="v5")
        duplicate = _decision(0.3, reason="same action")
        known_support = offline.SupportEnvelope(
            {"known": torch.zeros(1)},
            {"known": torch.ones(1)},
            {"known": 1.0},
            minimum_groups=1,
        )

        def run(
            alternative: SteeringDecision, support: np.ndarray, signature: str
        ) -> tuple[SteeringDecision, _BasePolicy, dict[str, int]]:
            base = _BasePolicy(incumbent)
            candidates = (
                _candidate(0, incumbent),
                _candidate(1, alternative),
            )
            searcher = SimpleNamespace(_candidates=lambda *_: candidates)
            policy = offline.ConservativeResidualPolicy(
                base,
                searcher,
                _Ensemble(),
                known_support,
                _calibration(),
            )

            def encode(_observation: object, candidate: object):
                if candidate.ordinal == 0:
                    return np.zeros(offline.INPUT_DIM), np.zeros(1), "known"
                return np.zeros(offline.INPUT_DIM), support, signature

            with patch.object(offline, "encode_candidate", side_effect=encode):
                selected = policy.predict({})
            return selected, base, policy.statistics()

        selected, base, counts = run(duplicate, np.zeros(1), "known")
        self.assertIs(selected, incumbent)
        self.assertEqual(base.commits, [])
        self.assertEqual(counts.get("overrides", 0), 0)

        for support, signature in (
            (np.zeros(1), "unseen"),
            (np.full(1, 100.0), "known"),
        ):
            with self.subTest(signature=signature, support=float(support[0])):
                selected, base, counts = run(
                    _decision(0.5, reason="alternative"), support, signature
                )
                self.assertIs(selected, incumbent)
                self.assertEqual(base.commits, [])
                self.assertEqual(counts.get("support_abstentions"), 1)

        module = _benchmark_module()
        audit_candidates = (
            _audited_candidate(0, incumbent),
            _audited_candidate(1, duplicate),
        )
        audit_result = _search_result(audit_candidates, 0)
        audit_result.sha256 = "1" * 64
        audit_result.snapshot_sha256 = "2" * 64
        core = SimpleNamespace(
            base_policy=_BasePolicy(incumbent),
            counts=Counter(),
            last_candidates=audit_candidates,
            last_selected_ordinal=0,
            last_supported_alternatives=1,
            last_certified_alternatives=0,
            last_incumbent=incumbent,
        )

        def core_predict(_observation: object) -> SteeringDecision:
            core.counts["candidate_queries"] += 1
            return incumbent

        core.predict = core_predict
        student = module.BudgetedResidualPolicy.__new__(
            module.BudgetedResidualPolicy
        )
        student.env = object()
        student.bundle = object()
        student.core = core
        student.audit_searcher = _StaticSearch(
            "student-audit",
            [],
            result=audit_result,
            rows=[{}],
        )
        student.stress = False
        student.sample_audit_cap = 1
        student.audits = []
        student.proposals = []
        student.counts = Counter()
        student.seed = 7
        certified = [
            {"ordinal": 0, "certified": False, "exact_unsafe": False},
            {"ordinal": 1, "certified": False, "exact_unsafe": False},
        ]
        with patch.object(module, "_certify_rows", return_value=certified):
            audited = student.predict({"tick": 11})
        self.assertIs(audited, incumbent)
        self.assertEqual(student.audits[0]["selected_ordinal"], 0)
        self.assertFalse(student.audits[0]["actual_override"])

    def test_missing_conformal_heads_abstains_without_spread_fallback(self) -> None:
        incumbent = _decision(0.3, reason="v5")
        alternative = _decision(0.5, reason="alternative")
        candidates = (
            _candidate(0, incumbent),
            _candidate(1, alternative),
        )

        class SpreadOnly:
            def predict(self, features: torch.Tensor):
                count = features.shape[0]
                return {
                    "cvar_mean": torch.ones(count),
                    "cvar_std": torch.zeros(count),
                    "unsafe_mean": torch.zeros(count),
                    "score_mean": torch.ones(count),
                    "score_std": torch.zeros(count),
                }

        base = _BasePolicy(incumbent)
        policy = offline.ConservativeResidualPolicy(
            base,
            SimpleNamespace(_candidates=lambda *_: candidates),
            SpreadOnly(),
            offline.SupportEnvelope(
                {"known": torch.zeros(1)},
                {"known": torch.ones(1)},
                {"known": 1.0},
                minimum_groups=1,
            ),
            _calibration(),
        )
        with patch.object(
            offline,
            "encode_candidate",
            return_value=(
                np.zeros(offline.INPUT_DIM),
                np.zeros(1),
                "known",
            ),
        ):
            selected = policy.predict({})
        self.assertIs(selected, incumbent)
        self.assertEqual(base.commits, [])
        self.assertEqual(
            policy.statistics().get("conformal_interface_abstentions"), 1
        )

    def test_equal_lexicographic_alternatives_abstain(self) -> None:
        incumbent = _decision(0.3, reason="v5")
        candidates = (
            _candidate(0, incumbent),
            _candidate(1, _decision(0.5, reason="first")),
            _candidate(2, _decision(0.7, reason="second")),
        )
        base = _BasePolicy(incumbent)
        policy = offline.ConservativeResidualPolicy(
            base,
            SimpleNamespace(_candidates=lambda *_: candidates),
            _Ensemble(),
            offline.SupportEnvelope(
                {"known": torch.zeros(1)},
                {"known": torch.ones(1)},
                {"known": 1.0},
                minimum_groups=1,
            ),
            _calibration(),
        )
        with patch.object(
            offline,
            "encode_candidate",
            return_value=(
                np.zeros(offline.INPUT_DIM),
                np.zeros(1),
                "known",
            ),
        ):
            selected = policy.predict({})
        self.assertIs(selected, incumbent)
        self.assertEqual(base.commits, [])
        self.assertEqual(
            policy.statistics().get("lexicographic_tie_abstentions"), 1
        )

    def test_audited_teacher_failures_abstain_before_base_commit(self) -> None:
        module = _benchmark_module()
        incumbent = _decision(0.3, reason="v5")
        alternative = _decision(0.5, reason="teacher alternative")
        teacher_candidates = (
            _audited_candidate(0, incumbent),
            _audited_candidate(1, alternative),
        )

        scenarios: dict[str, dict[str, object]] = {
            "exact-unsafe": {
                "teacher": _search_result(teacher_candidates, 1),
                "audit_candidates": teacher_candidates,
                "rows": [
                    _audit_row(
                        teacher_candidates[0], _exact_labels()
                    ),
                    _audit_row(
                        teacher_candidates[1],
                        _exact_labels(unsafe=True, risk_margin=-0.1),
                    ),
                ],
            },
            "unresolved-second-renewal": {
                "teacher": _search_result(teacher_candidates, 1),
                "audit_candidates": teacher_candidates,
                "rows": [
                    _audit_row(
                        teacher_candidates[0], _exact_labels()
                    ),
                    _audit_row(
                        teacher_candidates[1],
                        _exact_labels(resolved=False),
                    ),
                ],
            },
            "negative-absolute-solvency": {
                "teacher": _search_result(teacher_candidates, 1),
                "audit_candidates": teacher_candidates,
                "rows": [
                    _audit_row(
                        teacher_candidates[0], _exact_labels()
                    ),
                    _audit_row(
                        teacher_candidates[1],
                        _exact_labels(absolute_solvency=-0.1),
                    ),
                ],
            },
            "missing-audit-candidate": {
                "teacher": _search_result(teacher_candidates, 1),
                "audit_candidates": (teacher_candidates[0],),
                "rows": [
                    _audit_row(
                        teacher_candidates[0], _exact_labels()
                    )
                ],
            },
            "unsupported-audit": {
                "teacher": _search_result(teacher_candidates, 1),
                "audit_error": ValueError("unsupported audit state"),
            },
            "unsupported-audit-rows": {
                "teacher": _search_result(teacher_candidates, 1),
                "audit_candidates": teacher_candidates,
                "rows": [
                    _audit_row(candidate, _exact_labels())
                    for candidate in teacher_candidates
                ],
                "rows_error": ValueError("unsupported encoded candidate"),
            },
            "teacher-objective-tie": {
                "teacher": _search_result(
                    teacher_candidates, 1, strictly_improved=False
                ),
                "audit_candidates": teacher_candidates,
                "rows": [
                    _audit_row(candidate, _exact_labels())
                    for candidate in teacher_candidates
                ],
            },
        }

        scenarios["ambiguous-audit-match"] = {
            "teacher": _search_result(teacher_candidates, 1),
            "audit_candidates": (
                *teacher_candidates,
                _audited_candidate(1, alternative),
            ),
            "rows": [
                _audit_row(candidate, _exact_labels())
                for candidate in (
                    *teacher_candidates,
                    _audited_candidate(1, alternative),
                )
            ],
        }
        third = _audited_candidate(
            2, _decision(0.7, reason="equal top alternative")
        )
        scenarios["equal-top-teacher-alternatives"] = {
            "teacher": _search_result(
                (*teacher_candidates, third), 1, top_tie=True
            ),
            "audit_candidates": (*teacher_candidates, third),
            "rows": [
                _audit_row(candidate, _exact_labels())
                for candidate in (*teacher_candidates, third)
            ],
        }

        for name, scenario in scenarios.items():
            with self.subTest(name=name):
                audit_candidates = scenario.get("audit_candidates")
                audit_result = (
                    None
                    if audit_candidates is None
                    else _search_result(
                        tuple(audit_candidates),
                        min(1, len(audit_candidates) - 1),
                    )
                )
                policy, base, events = _audited_teacher_harness(
                    module,
                    teacher_result=scenario["teacher"],
                    audit_result=audit_result,
                    audit_rows=scenario.get("rows"),
                    audit_error=scenario.get("audit_error"),
                    audit_rows_error=scenario.get("rows_error"),
                )
                state = base.continuation_state
                selected = policy.predict({"tick": 11})
                self.assertIs(selected, incumbent)
                self.assertIs(base.decision, incumbent)
                self.assertIs(base.continuation_state, state)
                self.assertEqual(base.commits, [])
                self.assertNotIn("base.commit", events)

    def test_audited_teacher_incumbent_identical_action_abstains(self) -> None:
        module = _benchmark_module()
        incumbent = _decision(0.3, reason="v5")
        identical = SteeringDecision(
            SemanticAction.strong(0.3, 0.4),
            SteeringIntent.STEER_MATCH,
            source_body_id=3,
            destination_body_id=4,
            reason="same primitive action with different metadata",
        )
        candidates = (
            _audited_candidate(0, incumbent),
            _audited_candidate(1, identical),
        )
        result = _search_result(candidates, 1)
        policy, base, events = _audited_teacher_harness(
            module,
            teacher_result=result,
            audit_result=result,
            audit_rows=[
                _audit_row(candidate, _exact_labels())
                for candidate in candidates
            ],
        )
        state = base.continuation_state

        selected = policy.predict({"tick": 11})

        self.assertIs(selected, incumbent)
        self.assertIs(base.continuation_state, state)
        self.assertEqual(base.commits, [])
        self.assertNotIn("base.commit", events)

    def test_audited_teacher_safe_exact_alternative_commits_after_audit(
        self,
    ) -> None:
        module = _benchmark_module()
        incumbent = _decision(0.3, reason="v5")
        alternative = _decision(0.5, reason="safe teacher alternative")
        candidates = (
            _audited_candidate(0, incumbent),
            _audited_candidate(1, alternative),
        )
        result = _search_result(candidates, 1)
        policy, base, events = _audited_teacher_harness(
            module,
            teacher_result=result,
            audit_result=result,
            audit_rows=[
                _audit_row(candidate, _exact_labels())
                for candidate in candidates
            ],
        )
        state = base.continuation_state

        selected = policy.predict({"tick": 11})

        self.assertIs(selected, alternative)
        self.assertEqual(base.commits, [alternative])
        self.assertIsNot(base.continuation_state, state)
        self.assertLess(
            events.index("audit.search"), events.index("base.commit")
        )
        self.assertEqual(module._unsafe_teacher_actions([policy]), {})
        policy.audits[0]["row"]["labels"]["new_terminal"] = True
        self.assertIn(7, module._unsafe_teacher_actions([policy]))
        zero_labels = _exact_labels(
            risk_margin=0.0, absolute_solvency=0.0
        )
        zero_selected = _audit_row(candidates[1], zero_labels)
        zero_incumbent = _audit_row(candidates[0], zero_labels)
        for row in (zero_selected, zero_incumbent):
            row["solvency"]["minimum_surplus"] = 0
            row["solvency"]["negative_through_renewal"] = False
        self.assertFalse(
            module._exact_teacher_row_gate(
                zero_selected, zero_incumbent
            )[0]
        )

    def test_audit_certification_does_not_peek_at_exact_resolution(self) -> None:
        module = _benchmark_module()
        barrier = _calibration()
        bundle = SimpleNamespace(
            barrier=barrier,
            ensemble=_Ensemble(),
            support=offline.SupportEnvelope(
                {"known": torch.zeros(1)},
                {"known": torch.ones(1)},
                {"known": 1.0},
                minimum_groups=1,
            ),
        )
        rows = [
            {
                "features": np.zeros(offline.INPUT_DIM, dtype=np.float32),
                "support": np.zeros(1, dtype=np.float32),
                "signature": "known",
                "incumbent": False,
                "ordinal": 1,
                "labels": {
                    "risk_margin": 1.0,
                    "absolute_solvency": 1.0,
                    "score_advantage": 1.0,
                    "resolved": False,
                    "unsafe": True,
                    "censored": True,
                },
                "candidate": {},
                "outcome": {},
                "solvency": {},
                "search_sha256": "0" * 64,
            }
        ]
        certified = module._certify_rows(bundle, rows)
        self.assertTrue(certified[0]["certified"])
        self.assertTrue(certified[0]["exact_unsafe"])

    def test_support_requires_disjoint_groups_and_model_trains(self) -> None:
        generator = torch.Generator().manual_seed(7)
        count = 16
        tensors = {
            "features": torch.randn(count, offline.INPUT_DIM, generator=generator),
            "support": torch.randn(count, 4, generator=generator),
            "selected": torch.tensor([index % 2 == 0 for index in range(count)]),
            "risk_margin": torch.linspace(-1.0, 1.0, count),
            "unsafe": torch.tensor([index % 3 == 0 for index in range(count)]),
            "score_advantage": torch.linspace(-0.5, 0.5, count),
            "group": torch.arange(count) // 2,
        }
        indices = torch.arange(count)
        signatures = ["known"] * count
        support = offline.fit_support(
            tensors,
            indices,
            signatures,
            winner_only=False,
            minimum_groups=8,
        )
        self.assertIn("known", support.centers)

        one_group = dict(tensors)
        one_group["group"] = torch.zeros(count, dtype=torch.long)
        self.assertNotIn(
            "known",
            offline.fit_support(
                one_group,
                indices,
                signatures,
                winner_only=False,
                minimum_groups=8,
            ).centers,
        )

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            complete = offline.train_ensemble(
                tensors, indices, winner_only=False, steps=2, ensemble_size=1
            )
            winners = offline.train_ensemble(
                tensors, indices, winner_only=True, steps=2, ensemble_size=1
            )
        finally:
            torch.set_num_threads(previous_threads)
        self.assertEqual(complete.training_report["examples"], count)
        self.assertEqual(winners.training_report["examples"], count // 2)
        prediction = complete.predict(tensors["features"][:3])
        self.assertEqual(
            set(prediction),
            {
                "cvar_mean",
                "cvar_std",
                "unsafe_mean",
                "score_mean",
                "score_std",
            },
        )
        for value in prediction.values():
            self.assertEqual(value.shape, (3,))
            self.assertTrue(torch.isfinite(value).all())

    def test_barrier_uses_whole_seed_conformal_order_statistic(self) -> None:
        count = 64
        residual = torch.arange(count, dtype=torch.float32) / 100.0

        class Ensemble:
            def predict_full(self, features: torch.Tensor):
                size = features.shape[0]
                return {
                    "delta_cvar_mean": residual[:size],
                    "delta_cvar_std": torch.zeros(size),
                    "absolute_cvar_mean": torch.ones(size) + residual[:size],
                    "absolute_cvar_std": torch.zeros(size),
                    "unsafe_mean": torch.cat(
                        (torch.full((8,), 0.9), torch.full((size - 8,), 0.1))
                    ),
                    "score_mean": torch.ones(size),
                    "score_std": torch.zeros(size),
                }

        tensors = {
            "features": torch.zeros(count, offline.INPUT_DIM),
            "support": torch.zeros(count, 1),
            "incumbent": torch.zeros(count, dtype=torch.bool),
            "selected": torch.ones(count, dtype=torch.bool),
            "resolved": torch.ones(count, dtype=torch.bool),
            "unsafe": torch.tensor([index < 8 for index in range(count)]),
            "risk_margin": torch.zeros(count),
            "absolute_solvency": torch.ones(count),
            "score_advantage": torch.ones(count),
            "seed": torch.arange(count),
            "group": torch.arange(count),
        }
        support = offline.SupportEnvelope(
            {"known": torch.zeros(1)},
            {"known": torch.ones(1)},
            {"known": 1.0},
            minimum_groups=1,
        )
        signatures = ["known"] * count
        signatures[61] = "unsupported"
        tensors["resolved"][61] = False
        calibration = offline.fit_barrier(
            Ensemble(),
            support,
            tensors,
            torch.arange(count),
            signatures,
        )
        self.assertEqual(calibration.episode_count, 64)
        self.assertEqual(calibration.report["conformal_order"], 62)
        self.assertAlmostEqual(calibration.conformal_q, 0.61, places=5)
        self.assertAlmostEqual(
            calibration.absolute_conformal_q, 0.61, places=5
        )
        with self.assertRaisesRegex(RuntimeError, "whole-seed"):
            offline.fit_barrier(
                Ensemble(),
                support,
                tensors,
                torch.arange(58),
                ["known"] * count,
            )


if __name__ == "__main__":
    unittest.main()
