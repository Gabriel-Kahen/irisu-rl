from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "artifacts/r3/development/r3j-g4-onpolicy-screen-20260801-004"
    / "live_tau2_lease.py"
)
SPEC = importlib.util.spec_from_file_location("r3j_live_tau2_lease_tested", SOURCE)
assert SPEC is not None and SPEC.loader is not None
LEASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = LEASE
SPEC.loader.exec_module(LEASE)


def canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TensorSchema:
    version: str = "teacher-test-v1"
    source: str = "teacher_state"
    capacity: int = 196

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "source": self.source,
            "capacity": self.capacity,
            "global_features": ["gauge"],
            "body_features": ["id"],
            "preprocessing": ["test-only"],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha(self.manifest())


@dataclass(frozen=True)
class ActionSpec:
    version: str = "test-action-v1"

    def manifest(self) -> dict[str, object]:
        return {"version": self.version, "wait_choices": [1, 2, 4, 8, 16]}

    @property
    def sha256(self) -> str:
        return canonical_sha(self.manifest())


@dataclass(frozen=True)
class PointerActionSpec:
    version: str = "test-pointer-v1"

    def manifest(self) -> dict[str, object]:
        return {"version": self.version, "templates": [[0.0, 1.0]]}

    @property
    def sha256(self) -> str:
        return canonical_sha(self.manifest())


class TeacherStateEncoder:
    schema = TensorSchema()


class _EvalLeaf:
    def __init__(self) -> None:
        self.training = False


class GoalConditionedSteeringModel:
    def __init__(self, decisions: list[Decision] | None = None) -> None:
        self.schema = TeacherStateEncoder.schema
        self.pointer_spec = PointerActionSpec()
        self.training = False
        self.dropout = _EvalLeaf()
        self.decisions = list(decisions or [])
        self.mutate_during_deepcopy = False
        self.weight = torch.tensor([1.0, -2.0], dtype=torch.float32)

    def manifest(self) -> dict[str, object]:
        return {
            "architecture": "goal-conditioned-directed-pair-steering-test-v1",
            "schema_sha256": self.schema.sha256,
            "pointer_action_sha256": self.pointer_spec.sha256,
        }

    @property
    def architecture_sha256(self) -> str:
        return canonical_sha(self.manifest())

    def named_modules(self) -> tuple[tuple[str, object], ...]:
        return (("", self), ("dropout", self.dropout))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"weight": self.weight}


@dataclass(frozen=True)
class DirectedPair:
    source_id: int
    destination_id: int


@dataclass(frozen=True)
class _Attempt:
    pair: DirectedPair
    gap: float
    minimum_closure: float


@dataclass(frozen=True)
class _Stall:
    best_gap: float
    minimum_closure: float


class DirectedPairProgressTracker:
    def __init__(self) -> None:
        self.minimum_closure_sizes = 0.05
        self._attempt: _Attempt | None = None
        self._stalled: dict[DirectedPair, _Stall] = {}


@dataclass(frozen=True)
class Action:
    kind: str
    wait_ticks: int = 1
    x: float = 0.0
    y: float = 0.0

    @classmethod
    def wait(cls, ticks: int = 1) -> Action:
        return cls("wait", ticks)

    @classmethod
    def shot(cls) -> Action:
        return cls("shot")


@dataclass(frozen=True)
class Decision:
    actions: tuple[Action, ...]
    name: str

    def primitive_actions(self) -> tuple[Action, ...]:
        return self.actions


class GoalConditionedSteeringPolicy:
    def __init__(self, decisions: list[Decision] | None = None) -> None:
        self.model = GoalConditionedSteeringModel(decisions)
        self.encoder = TeacherStateEncoder()
        self.action_spec = ActionSpec()
        self.pointer_spec = self.model.pointer_spec
        self.artifact_sha256 = "a" * 64
        self.schema_sha256 = self.encoder.schema.sha256
        self.pointer_action_sha256 = self.pointer_spec.sha256
        self.cooldown_ticks = 16
        self.minimum_pair_closure_sizes = 0.05
        self.impact_side_sizes = 0.5
        self.impact_below_sizes = 0.75
        self.source_velocity_lead_ticks = 1.0
        self.ticks_per_second = 50.0
        self.act_logit_bias = 0.0
        self._cooldown_until = 0
        self._last_tick: int | None = None
        self._last_decision: Decision | None = None
        self._progress = DirectedPairProgressTracker()

    def __deepcopy__(self, memo: dict[int, object]) -> GoalConditionedSteeringPolicy:
        if self.model.mutate_during_deepcopy:
            self._cooldown_until += 1
        duplicate = type(self).__new__(type(self))
        memo[id(self)] = duplicate
        for name, value in vars(self).items():
            setattr(duplicate, name, copy.deepcopy(value, memo))
        duplicate.model.mutate_during_deepcopy = False
        return duplicate

    def predict(self, observation: dict[str, object]) -> Decision:
        if not self.model.decisions:
            raise RuntimeError("fake continuation exhausted")
        decision = self.model.decisions.pop(0)
        self._last_tick = int(observation["tick"])
        self._last_decision = decision
        return decision


@dataclass(frozen=True)
class Cashflow:
    tick: int
    entry_gauge: int
    gross_renewable_gain: int
    exit_gauge: int
    renewal: bool

    def manifest(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "entry_gauge": self.entry_gauge,
            "gross_renewable_gain": self.gross_renewable_gain,
            "exit_gauge": self.exit_gauge,
            "renewal": self.renewal,
        }


class FakeLedger:
    def __init__(self, observation: dict[str, object]) -> None:
        self.start_tick = int(observation["tick"])
        self.q = int(observation["gauge"])
        self.gauge_max = int(observation["gauge_max"])
        self.renewal_ticks: list[int] = []
        self.records: list[Cashflow] = []
        self.minimum_margin = self.q - 1
        self.lost = False
        self.unresolved: list[str] = []

    def advance(
        self,
        previous: dict[str, object],
        current: dict[str, object],
        events: tuple[dict[str, object], ...],
    ) -> Cashflow:
        relative = int(current["tick"]) - self.start_tick
        gains = [
            int(event["value"])
            for event in events
            if event.get("kind") == "gain"
        ]
        gross = sum(gains)
        entry = self.q
        self.q = min(self.gauge_max, self.q + gross) - 1
        if any(event.get("kind") == "loss" for event in events):
            self.q = 0
        if int(current["gauge"]) != self.q:
            raise RuntimeError("fake cashflow/public gauge mismatch")
        self.minimum_margin = min(self.minimum_margin, self.q - 1)
        renewal = gross > 0
        if renewal and (
            not self.renewal_ticks or self.renewal_ticks[-1] != relative
        ):
            self.renewal_ticks.append(relative)
        if any(event.get("kind") == "unresolved" for event in events):
            self.unresolved.append(f"unresolved@{relative}")
        if any(event.get("kind") == "resolved" for event in events):
            self.unresolved.clear()
        self.lost = self.q <= 0
        record = Cashflow(relative, entry, gross, self.q, renewal)
        self.records.append(record)
        return record

    def finalize(self, _cap: int) -> dict[str, object]:
        rows = [row.manifest() for row in self.records]
        resolved = len(self.renewal_ticks) >= 2
        return {
            "b2_margin": self.minimum_margin if resolved else None,
            "renewal_ticks": self.renewal_ticks[:2],
            "renewals_resolved": resolved,
            "unresolved": list(self.unresolved),
            "cashflow_lost": self.lost,
            "cashflow_sha256": canonical_sha(rows),
        }


class FakeCore:
    RenewableDebtLedger = FakeLedger
    INVALID_ACTION = "invalid"

    @staticmethod
    def _event_kind(event: dict[str, object]) -> object:
        return event.get("kind")


def observation(tick: int = 0, gauge: int = 100) -> dict[str, object]:
    return {
        "tick": tick,
        "gauge": gauge,
        "gauge_max": 200,
        "score": 0,
        "level": 1,
        "bodies": [],
        "terminated": False,
        "truncated": False,
    }


class FakeEnv:
    def __init__(
        self,
        *,
        events: dict[int, list[dict[str, object]]] | None = None,
        terminal_tick: int | None = None,
        truncate_tick: int | None = None,
    ) -> None:
        self.current = observation()
        self.events = events or {}
        self.terminal_tick = terminal_tick
        self.truncate_tick = truncate_tick
        self.actions: list[Action] = []

    def step(
        self, action: Action
    ) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
        self.actions.append(action)
        tick = int(self.current["tick"]) + 1
        events = copy.deepcopy(self.events.get(tick, []))
        gains = sum(
            int(event["value"])
            for event in events
            if event.get("kind") == "gain"
        )
        gauge = min(int(self.current["gauge_max"]), int(self.current["gauge"]) + gains) - 1
        if any(event.get("kind") == "loss" for event in events):
            gauge = 0
        terminated = tick == self.terminal_tick
        truncated = tick == self.truncate_tick
        self.current = {
            **self.current,
            "tick": tick,
            "gauge": gauge,
            "terminated": terminated,
            "truncated": truncated,
        }
        return self.current, 0.0, terminated, truncated, {"events": events}


def is_wait(action: object) -> bool:
    return isinstance(action, Action) and action.kind == "wait"


def public(value: dict[str, Any]) -> dict[str, object]:
    return {key: copy.deepcopy(value[key]) for key in sorted(value)}


def bound_lease(
    policy: GoalConditionedSteeringPolicy,
) -> tuple[LEASE.LiveTau2Lease, list[int]]:
    lease = LEASE.LiveTau2Lease(FakeCore, observation(), public_observation=public)
    calls: list[int] = []

    def commit(
        target: GoalConditionedSteeringPolicy,
        _obs: dict[str, object],
        _incumbent: object,
        _selected: object,
    ) -> bool:
        calls.append(1)
        target._cooldown_until = 16
        return True

    pre = LEASE.canonical_policy_state_hash(policy)
    lease.commit_live_policy(
        policy,
        observation(),
        "incumbent",
        "selected",
        commit,
        expected_pre_state_sha256=pre,
    )
    return lease, calls


class LiveTau2LeaseTests(unittest.TestCase):
    def test_raw_numpy_tick_is_canonicalized_before_strict_integer_check(
        self,
    ) -> None:
        def numpy_public(value: dict[str, Any]) -> dict[str, object]:
            return {
                key: (
                    child.item()
                    if callable(getattr(child, "item", None))
                    else copy.deepcopy(child)
                )
                for key, child in sorted(value.items())
            }

        initial = observation()
        initial["tick"] = np.uint64(0)
        shadow = LEASE.ShadowEnvStepTracer(
            FakeCore,
            FakeEnv(),
            initial,
            public_observation=numpy_public,
            is_wait_action=is_wait,
        )
        self.assertEqual(shadow._start_tick, 0)

        policy = GoalConditionedSteeringPolicy()
        lease = LEASE.LiveTau2Lease(
            FakeCore,
            initial,
            public_observation=numpy_public,
        )
        lease.commit_live_policy(
            policy,
            initial,
            "incumbent",
            "incumbent",
            lambda *_args: True,
        )
        current = {
            **initial,
            "tick": np.uint64(1),
            "gauge": 99,
        }
        lease.begin_macro("selected", 1)
        row = lease.advance(
            initial,
            current,
            (),
            action=Action.wait(1),
            terminated=False,
            truncated=False,
        )
        lease.end_macro()
        self.assertEqual(row["relative_tick"], 1)
        self.assertEqual(lease.elapsed_ticks, 1)
        self.assertEqual(lease.status, "bound")

    def test_shot_then_fourteen_tick_pad_matches_exact_chronology(self) -> None:
        policy = GoalConditionedSteeringPolicy(
            [Decision((Action.wait(3),), "continuation")]
        )
        lease, calls = bound_lease(policy)
        env = FakeEnv(
            events={
                17: [{"kind": "gain", "value": 20}],
                19: [{"kind": "gain", "value": 20}],
            }
        )
        selected = (Action.shot(), Action.wait(1))
        lease.execute_selected_and_pad(
            env, selected, is_wait_action=is_wait, wait_action=Action.wait
        )
        self.assertFalse(lease.query_allowed)
        phases = [row["phase"] for row in lease.trace]
        self.assertEqual(phases.count("selected"), 2)
        self.assertEqual(phases.count("pad"), 14)
        observed: list[tuple[int, str]] = []

        def observer(
            _policy: object,
            current: dict[str, object],
            decision: Decision,
        ) -> dict[str, object]:
            observed.append((int(current["tick"]), decision.name))
            return {"scheduled_query_shot": False, "suppressed": False}

        lease.continue_until_closed(
            env,
            policy,
            predict=lambda target, obs: target.predict(dict(obs)),
            primitive_actions=lambda decision: decision.primitive_actions(),
            is_wait_action=is_wait,
            wait_action=Action.wait,
            decision_observer=observer,
        )
        self.assertTrue(lease.query_allowed)
        self.assertEqual(calls, [1])
        certificate = lease.certificate()
        self.assertEqual(certificate["renewal_ticks"], [17, 19])
        self.assertEqual(observed, [(16, "continuation")])
        self.assertEqual(certificate["continuation_decision_count"], 1)
        self.assertEqual(
            certificate["continuation_decisions"][0]["observer"],
            {"scheduled_query_shot": False, "suppressed": False},
        )

    def test_wait_selected_macro_gets_exact_remaining_pad(self) -> None:
        policy = GoalConditionedSteeringPolicy()
        lease, _calls = bound_lease(policy)
        env = FakeEnv(
            events={
                10: [{"kind": "gain", "value": 10}],
                12: [{"kind": "gain", "value": 10}],
            }
        )
        lease.execute_selected_and_pad(
            env,
            (Action.wait(4),),
            is_wait_action=is_wait,
            wait_action=Action.wait,
        )
        self.assertTrue(lease.query_allowed)
        phases = [row["phase"] for row in lease.trace]
        self.assertEqual(phases.count("selected"), 4)
        self.assertEqual(phases.count("pad"), 12)
        self.assertEqual(lease.elapsed_ticks, 16)

    def test_same_tick_gains_are_one_renewal_epoch(self) -> None:
        policy = GoalConditionedSteeringPolicy()
        lease, _calls = bound_lease(policy)
        env = FakeEnv(
            events={
                1: [
                    {"kind": "gain", "value": 5},
                    {"kind": "gain", "value": 7},
                ],
                2: [{"kind": "gain", "value": 5}],
            }
        )
        lease.execute_selected_and_pad(
            env,
            (Action.wait(2),),
            is_wait_action=is_wait,
            wait_action=Action.wait,
        )
        self.assertEqual(lease.certificate()["renewal_ticks"], [1, 2])

    def test_tau2_mid_macro_closes_only_after_macro_release(self) -> None:
        policy = GoalConditionedSteeringPolicy(
            [Decision((Action.wait(4),), "long continuation")]
        )
        lease, _calls = bound_lease(policy)
        env = FakeEnv(
            events={
                17: [{"kind": "gain", "value": 10}],
                18: [{"kind": "gain", "value": 10}],
                19: [{"kind": "gain", "value": 10}],
            }
        )
        lease.execute_selected_and_pad(
            env,
            (Action.wait(1),),
            is_wait_action=is_wait,
            wait_action=Action.wait,
        )
        lease.continue_until_closed(
            env,
            policy,
            predict=lambda target, obs: target.predict(dict(obs)),
            primitive_actions=lambda decision: decision.primitive_actions(),
            is_wait_action=is_wait,
            wait_action=Action.wait,
        )
        certificate = lease.certificate()
        self.assertEqual(certificate["renewal_ticks"], [17, 18])
        self.assertEqual(certificate["end_tick"], 20)
        self.assertNotEqual(
            certificate["cashflow_sha256"],
            certificate["release_cashflow_sha256"],
        )
        self.assertEqual(
            certificate["tau2_ledger"]["renewal_ticks"], [17, 18]
        )
        tau2 = lease.trace[17]
        self.assertEqual(tau2["relative_tick"], 18)
        self.assertFalse(tau2["macro_complete"])

    def test_policy_seal_detects_deepcopy_mutating_original(self) -> None:
        policy = GoalConditionedSteeringPolicy()
        duplicate, seal = LEASE.guarded_deepcopy_policy(policy)
        seal.verify(policy)
        self.assertEqual(
            LEASE.canonical_policy_state_hash(duplicate),
            LEASE.canonical_policy_state_hash(policy),
        )
        policy.model.mutate_during_deepcopy = True
        with self.assertRaisesRegex(RuntimeError, "state changed"):
            seal.guarded_deepcopy(policy)

    def test_failed_commit_is_before_steps_and_cannot_retry(self) -> None:
        policy = GoalConditionedSteeringPolicy()
        lease = LEASE.LiveTau2Lease(
            FakeCore, observation(), public_observation=public
        )
        env = FakeEnv()
        calls: list[int] = []

        def reject(*_args: object) -> bool:
            calls.append(1)
            return False

        with self.assertRaisesRegex(RuntimeError, "sole live policy commit"):
            lease.commit_live_policy(
                policy, observation(), "incumbent", "selected", reject
            )
        self.assertEqual(calls, [1])
        self.assertEqual(env.actions, [])
        self.assertEqual(lease.status, "failed")
        with self.assertRaises(RuntimeError):
            lease.commit_live_policy(
                policy, observation(), "incumbent", "selected", reject
            )
        self.assertEqual(calls, [1])

    def test_overlapping_macro_is_rejected(self) -> None:
        policy = GoalConditionedSteeringPolicy()
        lease, _calls = bound_lease(policy)
        lease.begin_macro("selected", 1)
        with self.assertRaisesRegex(RuntimeError, "overlapped"):
            lease.begin_macro("selected", 1)

    def test_terminal_censor_loss_invalid_and_unresolved_fail_closed(self) -> None:
        cases = {
            "terminal": (FakeEnv(terminal_tick=1), None),
            "censor": (FakeEnv(truncate_tick=1), None),
            "loss": (FakeEnv(events={1: [{"kind": "loss"}]}), None),
            "invalid": (FakeEnv(events={1: [{"kind": "invalid"}]}), None),
            "unresolved": (
                FakeEnv(
                    events={
                        1: [
                            {"kind": "gain", "value": 10},
                            {"kind": "unresolved"},
                        ],
                        2: [{"kind": "gain", "value": 10}],
                    }
                ),
                None,
            ),
        }
        for name, (env, _unused) in cases.items():
            with self.subTest(name=name):
                policy = GoalConditionedSteeringPolicy()
                lease, _calls = bound_lease(policy)
                with self.assertRaises(RuntimeError):
                    lease.execute_selected_and_pad(
                        env,
                        (Action.wait(2),),
                        is_wait_action=is_wait,
                        wait_action=Action.wait,
                    )
                self.assertEqual(lease.status, "failed")
                self.assertFalse(lease.query_allowed)

    def test_exact_live_parity_detects_trace_and_summary_tampering(self) -> None:
        policy = GoalConditionedSteeringPolicy()
        lease, _calls = bound_lease(policy)
        env = FakeEnv(
            events={
                1: [{"kind": "gain", "value": 10}],
                2: [{"kind": "gain", "value": 10}],
            }
        )
        lease.execute_selected_and_pad(
            env,
            (Action.wait(2),),
            is_wait_action=is_wait,
            wait_action=Action.wait,
        )
        live = lease.certificate()
        exact = copy.deepcopy(live)
        LEASE.assert_exact_parity(exact, live)
        tampered = copy.deepcopy(live)
        tampered["b2_margin"] = int(tampered["b2_margin"]) + 1
        with self.assertRaisesRegex(RuntimeError, "self-hash"):
            LEASE.assert_exact_parity(exact, tampered)
        tampered = copy.deepcopy(live)
        tampered["trace"]["rows"][0]["action"]["fields"]["kind"] = "shot"
        with self.assertRaisesRegex(RuntimeError, "self-hash|trace"):
            LEASE.assert_exact_parity(exact, tampered)

    def test_transparent_shadow_full_branch_accepts_live_release_prefix(self) -> None:
        events = {
            17: [{"kind": "gain", "value": 10}],
            18: [{"kind": "gain", "value": 10}],
        }
        exact_env = FakeEnv(events=events)
        shadow = LEASE.ShadowEnvStepTracer(
            FakeCore,
            exact_env,
            observation(),
            public_observation=public,
            is_wait_action=is_wait,
        )
        exact_actions = (
            Action.shot(),
            Action.wait(1),
            *(Action.wait(1) for _ in range(14)),
            *(Action.wait(1) for _ in range(4)),
            *(Action.wait(1) for _ in range(8)),
        )
        for action in exact_actions:
            returned = shadow.step(action)
            self.assertIs(returned[0], exact_env.current)
        exact = shadow.certificate()

        policy = GoalConditionedSteeringPolicy(
            [Decision((Action.wait(4),), "continuation")]
        )
        live, _calls = bound_lease(policy)
        live_env = FakeEnv(events=events)
        live.execute_selected_and_pad(
            live_env,
            (Action.shot(), Action.wait(1)),
            is_wait_action=is_wait,
            wait_action=Action.wait,
        )
        live.continue_until_closed(
            live_env,
            policy,
            predict=lambda target, obs: target.predict(dict(obs)),
            primitive_actions=lambda decision: decision.primitive_actions(),
            is_wait_action=is_wait,
            wait_action=Action.wait,
        )
        LEASE.assert_exact_parity(exact, live.certificate())
        self.assertGreater(
            len(exact["unit_trace"]["rows"]),
            len(live.certificate()["unit_trace"]["rows"]),
        )

    def test_frozen_wait_macro_is_truncated_at_exact_horizon(self) -> None:
        policy = GoalConditionedSteeringPolicy(
            [Decision((Action.wait(10),), "crossing wait")]
        )
        lease = LEASE.LiveTau2Lease(
            FakeCore,
            observation(),
            public_observation=public,
            horizon_ticks=20,
        )
        calls: list[int] = []

        def commit(*_args: object) -> bool:
            calls.append(1)
            return True

        lease.commit_live_policy(
            policy, observation(), "incumbent", "selected", commit
        )
        env = FakeEnv(
            events={
                18: [{"kind": "gain", "value": 10}],
                20: [{"kind": "gain", "value": 10}],
            }
        )
        lease.execute_selected_and_pad(
            env,
            (Action.wait(1),),
            is_wait_action=is_wait,
            wait_action=Action.wait,
        )
        lease.continue_until_closed(
            env,
            policy,
            predict=lambda target, obs: target.predict(dict(obs)),
            primitive_actions=lambda decision: decision.primitive_actions(),
            is_wait_action=is_wait,
            wait_action=Action.wait,
        )
        self.assertEqual(lease.elapsed_ticks, 20)
        continuation = [
            row for row in lease.trace if row["phase"] == "continuation"
        ]
        self.assertEqual(len(continuation), 4)
        self.assertEqual(calls, [1])

    def test_policy_state_rejects_unknown_mutable_field(self) -> None:
        policy = GoalConditionedSteeringPolicy()
        policy.foreign_state = 1
        with self.assertRaisesRegex(RuntimeError, "fields differ"):
            LEASE.canonical_policy_state_manifest(policy)

    def test_policy_state_rejects_unknown_nested_progress_field(self) -> None:
        policy = GoalConditionedSteeringPolicy()
        before = LEASE.canonical_policy_state_hash(policy)
        policy._progress.foreign_mutable_state = {"unbound": 1}
        with self.assertRaisesRegex(RuntimeError, "progress tracker state fields"):
            LEASE.canonical_policy_state_manifest(policy)
        del policy._progress.foreign_mutable_state
        self.assertEqual(LEASE.canonical_policy_state_hash(policy), before)

    def test_policy_state_binds_components_modes_and_tensor_bytes(self) -> None:
        def check_rejected(
            mutate: Any,
        ) -> None:
            policy = GoalConditionedSteeringPolicy()
            seal = LEASE.seal_policy_state(policy)
            mutate(policy)
            with self.assertRaises((RuntimeError, TypeError, ValueError)):
                seal.verify(policy)

        cases = {
            "action-spec": lambda policy: setattr(
                policy, "action_spec", ActionSpec("other-action")
            ),
            "pointer-spec": lambda policy: setattr(
                policy, "pointer_spec", PointerActionSpec("other-pointer")
            ),
            "encoder": lambda policy: setattr(policy, "encoder", object()),
            "root-train-mode": lambda policy: setattr(
                policy.model, "training", True
            ),
            "child-train-mode": lambda policy: setattr(
                policy.model.dropout, "training", True
            ),
            "model-tensor": lambda policy: policy.model.weight.add_(1.0),
            "cached-schema": lambda policy: setattr(
                policy, "schema_sha256", "f" * 64
            ),
            "cached-pointer": lambda policy: setattr(
                policy, "pointer_action_sha256", "e" * 64
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                check_rejected(mutate)

    def test_expected_exact_mismatch_never_publishes_closed_state(self) -> None:
        exact_policy = GoalConditionedSteeringPolicy()
        exact_lease, _calls = bound_lease(exact_policy)
        exact_env = FakeEnv(
            events={
                1: [{"kind": "gain", "value": 10}],
                2: [{"kind": "gain", "value": 10}],
            }
        )
        exact_lease.execute_selected_and_pad(
            exact_env,
            (Action.wait(2),),
            is_wait_action=is_wait,
            wait_action=Action.wait,
        )

        live_policy = GoalConditionedSteeringPolicy()
        live = LEASE.LiveTau2Lease(
            FakeCore,
            observation(),
            public_observation=public,
            expected_exact=exact_lease.certificate(),
        )
        live.commit_live_policy(
            live_policy,
            observation(),
            "incumbent",
            "selected",
            lambda *_args: True,
        )
        live_env = FakeEnv(
            events={
                1: [{"kind": "gain", "value": 20}],
                2: [{"kind": "gain", "value": 20}],
            }
        )
        with self.assertRaisesRegex(RuntimeError, "exact/live tau2 parity"):
            live.execute_selected_and_pad(
                live_env,
                (Action.wait(2),),
                is_wait_action=is_wait,
                wait_action=Action.wait,
            )
        self.assertEqual(live.status, "failed")
        self.assertFalse(live.query_allowed)
        self.assertIn("parity", str(live.failure))
        with self.assertRaisesRegex(RuntimeError, "only a closed"):
            live.certificate()

    def test_post_tau2_pre_release_failures_remain_terminal(self) -> None:
        cases = {
            "terminal": {
                "events": {},
                "terminal_tick": 19,
            },
            "censor": {
                "events": {},
                "truncate_tick": 19,
            },
            "loss": {
                "events": {19: [{"kind": "loss"}]},
            },
            "invalid": {
                "events": {19: [{"kind": "invalid"}]},
            },
            "unresolved": {
                "events": {19: [{"kind": "unresolved"}]},
            },
        }
        for name, options in cases.items():
            with self.subTest(name=name):
                events = {
                    17: [{"kind": "gain", "value": 10}],
                    18: [{"kind": "gain", "value": 10}],
                    **options.get("events", {}),
                }
                env = FakeEnv(
                    events=events,
                    terminal_tick=options.get("terminal_tick"),
                    truncate_tick=options.get("truncate_tick"),
                )
                policy = GoalConditionedSteeringPolicy(
                    [Decision((Action.wait(4),), "continuation")]
                )
                lease, _calls = bound_lease(policy)
                lease.execute_selected_and_pad(
                    env,
                    (Action.wait(1),),
                    is_wait_action=is_wait,
                    wait_action=Action.wait,
                )
                with self.assertRaises(RuntimeError):
                    lease.continue_until_closed(
                        env,
                        policy,
                        predict=lambda target, obs: target.predict(dict(obs)),
                        primitive_actions=lambda decision: decision.primitive_actions(),
                        is_wait_action=is_wait,
                        wait_action=Action.wait,
                    )
                self.assertEqual(lease.status, "failed")
                self.assertFalse(lease.query_allowed)
                with self.assertRaisesRegex(RuntimeError, "only a closed"):
                    lease.certificate()

    def test_missing_second_renewal_at_horizon_fails_terminally(self) -> None:
        policy = GoalConditionedSteeringPolicy(
            [Decision((Action.wait(10),), "horizon wait")]
        )
        lease = LEASE.LiveTau2Lease(
            FakeCore,
            observation(),
            public_observation=public,
            horizon_ticks=20,
        )
        lease.commit_live_policy(
            policy,
            observation(),
            "incumbent",
            "selected",
            lambda *_args: True,
        )
        env = FakeEnv(events={18: [{"kind": "gain", "value": 10}]})
        lease.execute_selected_and_pad(
            env,
            (Action.wait(1),),
            is_wait_action=is_wait,
            wait_action=Action.wait,
        )
        with self.assertRaisesRegex(RuntimeError, "second renewal.*horizon"):
            lease.continue_until_closed(
                env,
                policy,
                predict=lambda target, obs: target.predict(dict(obs)),
                primitive_actions=lambda decision: decision.primitive_actions(),
                is_wait_action=is_wait,
                wait_action=Action.wait,
            )
        self.assertEqual(lease.elapsed_ticks, 20)
        self.assertEqual(lease.status, "failed")
        self.assertFalse(lease.query_allowed)

    def test_nonwait_macro_cannot_be_partially_executed_at_horizon(self) -> None:
        policy = GoalConditionedSteeringPolicy(
            [Decision((Action.shot(), Action.wait(1)), "crossing shot")]
        )
        lease = LEASE.LiveTau2Lease(
            FakeCore,
            observation(),
            public_observation=public,
            horizon_ticks=17,
        )
        lease.commit_live_policy(
            policy,
            observation(),
            "incumbent",
            "selected",
            lambda *_args: True,
        )
        env = FakeEnv()
        lease.execute_selected_and_pad(
            env,
            (Action.wait(1),),
            is_wait_action=is_wait,
            wait_action=Action.wait,
        )
        with self.assertRaisesRegex(RuntimeError, "non-wait.*horizon"):
            lease.continue_until_closed(
                env,
                policy,
                predict=lambda target, obs: target.predict(dict(obs)),
                primitive_actions=lambda decision: decision.primitive_actions(),
                is_wait_action=is_wait,
                wait_action=Action.wait,
            )
        self.assertEqual(len(env.actions), 16)

    def test_shadow_rejects_terminal_start_and_step_beyond_horizon(self) -> None:
        terminal = observation()
        terminal["terminated"] = True
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            LEASE.ShadowEnvStepTracer(
                FakeCore,
                FakeEnv(),
                terminal,
                public_observation=public,
                is_wait_action=is_wait,
            )
        env = FakeEnv()
        shadow = LEASE.ShadowEnvStepTracer(
            FakeCore,
            env,
            observation(),
            public_observation=public,
            is_wait_action=is_wait,
            horizon_ticks=1,
        )
        shadow.step(Action.wait(1))
        with self.assertRaisesRegex(RuntimeError, "horizon"):
            shadow.step(Action.wait(1))
        self.assertEqual(len(env.actions), 1)

    def test_shadow_negative_witness_binds_terminal_trace_and_reason(self) -> None:
        env = FakeEnv()
        shadow = LEASE.ShadowEnvStepTracer(
            FakeCore,
            env,
            observation(),
            public_observation=public,
            is_wait_action=is_wait,
        )
        shadow.step(Action.wait(1))
        reason = "shadow branch did not reach a second renewal"
        with self.assertRaisesRegex(RuntimeError, "second renewal"):
            shadow.certificate()
        witness = shadow.terminal_witness(reason)
        self.assertEqual(
            set(witness),
            {
                "schema",
                "status",
                "certificate_mode",
                "failure_reason",
                "start_tick",
                "end_tick",
                "elapsed_ticks",
                "horizon_ticks",
                "start_public_observation",
                "end_public_observation",
                "start_public_sha256",
                "end_public_sha256",
                "unit_trace_root_sha256",
                "unit_trace_sha256",
                "unit_trace",
                "tau2",
                "tau2_ledger",
                "full_ledger",
                "full_cashflow_sha256",
                "sha256",
            },
        )
        self.assertEqual(
            witness["schema"],
            "irisu-r3j-live-tau2-negative-witness-v1",
        )
        self.assertEqual(witness["status"], "failed")
        self.assertEqual(
            witness["certificate_mode"],
            "exact-shadow-terminal-failure",
        )
        self.assertEqual(witness["failure_reason"], reason)
        self.assertEqual(witness["elapsed_ticks"], 1)
        self.assertEqual(
            witness["start_public_sha256"],
            LEASE._sha256(witness["start_public_observation"]),
        )
        self.assertEqual(
            witness["end_public_sha256"],
            LEASE._sha256(witness["end_public_observation"]),
        )
        self.assertEqual(
            witness["unit_trace_root_sha256"],
            witness["unit_trace"]["rows"][-1]["sha256"],
        )
        self.assertEqual(
            witness["full_cashflow_sha256"],
            witness["full_ledger"]["cashflow_sha256"],
        )
        self.assertIsNone(witness["tau2"])
        self.assertIsNone(witness["tau2_ledger"])
        body = dict(witness)
        supplied = body.pop("sha256")
        self.assertEqual(supplied, LEASE._sha256(body))
        with self.assertRaisesRegex(RuntimeError, "differs from its trace"):
            shadow.terminal_witness(
                "shadow branch has structural failure evidence"
            )

    def test_shadow_negative_witness_allows_zero_step_failure(self) -> None:
        shadow = LEASE.ShadowEnvStepTracer(
            FakeCore,
            FakeEnv(),
            observation(),
            public_observation=public,
            is_wait_action=is_wait,
        )
        witness = shadow.terminal_witness(
            "shadow branch did not reach a second renewal"
        )
        self.assertEqual(witness["elapsed_ticks"], 0)
        self.assertEqual(witness["unit_trace"]["rows"], [])
        self.assertEqual(witness["unit_trace_root_sha256"], "0" * 64)
        self.assertIsNone(witness["tau2"])
        self.assertIsNone(witness["tau2_ledger"])
        self.assertEqual(
            witness["start_public_sha256"],
            witness["end_public_sha256"],
        )
        self.assertEqual(
            witness["start_public_observation"],
            witness["end_public_observation"],
        )

    def test_shadow_negative_witness_rejects_successful_branch(self) -> None:
        env = FakeEnv(
            events={
                1: [{"kind": "gain", "value": 10}],
                2: [{"kind": "gain", "value": 10}],
            }
        )
        shadow = LEASE.ShadowEnvStepTracer(
            FakeCore,
            env,
            observation(),
            public_observation=public,
            is_wait_action=is_wait,
        )
        shadow.step(Action.wait(1))
        shadow.step(Action.wait(1))
        self.assertEqual(shadow.certificate()["status"], "closed")
        with self.assertRaisesRegex(RuntimeError, "no negative witness"):
            shadow.terminal_witness(
                "shadow branch did not reach a second renewal"
            )

    def test_shadow_negative_witness_derives_structural_failure(self) -> None:
        env = FakeEnv(
            events={
                1: [{"kind": "gain", "value": 10}],
                2: [{"kind": "gain", "value": 10}],
                3: [{"kind": "invalid"}],
            }
        )
        shadow = LEASE.ShadowEnvStepTracer(
            FakeCore,
            env,
            observation(),
            public_observation=public,
            is_wait_action=is_wait,
        )
        for _ in range(3):
            shadow.step(Action.wait(1))
        reason = "shadow branch has structural failure evidence"
        with self.assertRaisesRegex(RuntimeError, "structural failure"):
            shadow.certificate()
        witness = shadow.terminal_witness(reason)
        self.assertEqual(witness["failure_reason"], reason)
        self.assertEqual(
            witness["unit_trace"]["rows"][-1]["events"],
            [{"kind": "invalid"}],
        )
        self.assertIsInstance(witness["tau2"], dict)
        self.assertIsInstance(witness["tau2_ledger"], dict)

    def test_shadow_negative_witness_freezes_tau2_liability_evidence(self) -> None:
        env = FakeEnv(
            events={
                1: [
                    {"kind": "gain", "value": 10},
                    {"kind": "unresolved"},
                ],
                2: [{"kind": "gain", "value": 10}],
                3: [{"kind": "resolved"}],
            }
        )
        shadow = LEASE.ShadowEnvStepTracer(
            FakeCore,
            env,
            observation(),
            public_observation=public,
            is_wait_action=is_wait,
        )
        for _ in range(3):
            shadow.step(Action.wait(1))
        reason = "shadow branch has unresolved liability evidence"
        with self.assertRaisesRegex(RuntimeError, "unresolved liability"):
            shadow.certificate()
        witness = shadow.terminal_witness(reason)
        self.assertEqual(witness["failure_reason"], reason)
        self.assertEqual(witness["tau2"]["relative_tick"], 2)
        self.assertEqual(
            witness["tau2"]["unit_trace_root_sha256"],
            witness["unit_trace"]["rows"][1]["sha256"],
        )
        self.assertEqual(
            witness["tau2_ledger"]["unresolved"],
            ["unresolved@1"],
        )
        self.assertEqual(witness["full_ledger"]["unresolved"], [])
        self.assertNotEqual(
            witness["tau2_ledger"]["cashflow_sha256"],
            witness["full_ledger"]["cashflow_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
