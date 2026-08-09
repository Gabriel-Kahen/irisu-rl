from __future__ import annotations

import copy
import unittest
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import torch
from irisu_env import Action, ActionKind
from irisu_pointer.geometry_dagger import (
    GeometryDaggerConfig,
    LearnerVisitedGeometryDaggerPolicy,
)
from irisu_pointer.geometry_learning import GeometryModelConfig, GeometrySelectorModel
from irisu_pointer.geometry_policy import (
    GeometryPolicyConfig,
    GeometrySelectorEnsemble,
    SafeguardedGeometryPolicy,
    geometry_candidate_vocabulary_sha256,
)
from irisu_pointer.geometry_search import (
    GeometrySearchConfig,
    enumerate_geometry_candidates,
)
from irisu_pointer.runway_search import RunwayGeometrySearch, RunwaySearchConfig
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
    SteeringModelConfig,
)
from irisu_rl.actions import ActionSpec, SemanticAction
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.schema import TEACHER_V1

_BASE_SHA256 = "b" * 64
_SOURCE_SHA256 = "c" * 64
_SELECTOR_SHA256 = "d" * 64
_RUNTIME_SHA256 = "e" * 64


def _body(identifier: int, x: float) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": "dynamic_fresh",
        "color": 1,
        "x": x,
        "y": 140.0,
        "vx": 0.0,
        "vy": 0.0,
        "vx_display_per_second": 0.0,
        "vy_display_per_second": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 40.0,
        "chain_id": 0,
        "projectile_hits": 0,
        "age_ticks": 20,
        "remaining_lifetime": 1000,
        "rot_timer": 0,
    }


def _observation(*, tick: int = 1, gauge: int = 20) -> dict[str, Any]:
    return {
        "tick": tick,
        "score": 0,
        "gauge": gauge,
        "gauge_max": 100,
        "level": 1,
        "highest_chain": 0,
        "qualifying_clear_count": 0,
        "left_held": False,
        "right_held": False,
        "terminated": False,
        "truncated": False,
        "field": {"x": 16.0, "y": 0.0, "width": 576.0, "height": 480.0},
        "difficulty": {"active_colors": 4, "spawn_interval_ticks": 5},
        "bodies": [_body(1, 200.0), _body(2, 360.0)],
    }


def _incumbent() -> SteeringDecision:
    return SteeringDecision(
        SemanticAction.strong(180.0 / 640.0, 170.0 / 480.0),
        SteeringIntent.STEER_MATCH,
        source_body_id=1,
        destination_body_id=2,
        destination_chain_id=0,
        impact_x_sizes=-0.5,
        impact_y_sizes=0.75,
        reason="fixed incumbent",
    )


class _BasePolicy(GoalConditionedSteeringPolicy):
    def __init__(self) -> None:
        model = GoalConditionedSteeringModel(
            TEACHER_V1,
            config=SteeringModelConfig(body_hidden=8, global_hidden=8, pair_hidden=8),
        )
        model.eval()
        super().__init__(model, artifact_sha256=_BASE_SHA256)
        self.fixed_decision = _incumbent()
        self.calls = 0
        self.reset_calls = 0

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        self.calls += 1
        return self.fixed_decision

    def reset(self, seed: int = 0) -> None:
        self.reset_calls += 1
        super().reset(seed)


def _selector(
    config: GeometrySearchConfig, biases: Mapping[int, float]
) -> GeometrySelectorModel:
    model = GeometrySelectorModel(
        TEACHER_V1,
        candidate_count=config.max_candidates,
        candidate_set_sha256=geometry_candidate_vocabulary_sha256(config),
        config=GeometryModelConfig(body_hidden=8, pair_hidden=8),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for slot, value in biases.items():
            model.selector[-1].bias[slot] = value
    model.eval()
    return model


def _student(
    base: _BasePolicy,
    geometry: GeometrySearchConfig,
    *,
    biases: Mapping[int, float] | None = None,
    ensemble_biases: tuple[Mapping[int, float], ...] | None = None,
    policy_config: GeometryPolicyConfig | None = None,
) -> SafeguardedGeometryPolicy:
    if ensemble_biases is None:
        selector: GeometrySelectorModel | GeometrySelectorEnsemble = _selector(
            geometry, {2: 10.0} if biases is None else biases
        )
        identity = _SELECTOR_SHA256
        safeguards = GeometryPolicyConfig() if policy_config is None else policy_config
    else:
        selector = GeometrySelectorEnsemble(
            tuple(_selector(geometry, value) for value in ensemble_biases),
            artifact_sha256s=tuple(
                f"{index + 1:064x}" for index in range(len(ensemble_biases))
            ),
        )
        identity = selector.sha256
        safeguards = (
            GeometryPolicyConfig(
                minimum_confidence=0.0,
                minimum_logit_margin=0.0,
                minimum_ensemble_members=len(ensemble_biases),
                minimum_ensemble_agreement=1.0,
            )
            if policy_config is None
            else policy_config
        )
    return SafeguardedGeometryPolicy(
        base,
        selector,
        geometry_config=geometry,
        policy_config=safeguards,
        selector_artifact_sha256=identity,
        source_identity=_SOURCE_SHA256,
    )


def _action_key(decision: SteeringDecision) -> tuple[int, int, int]:
    action = decision.primitive_actions(ActionSpec())[0]
    return (
        int(ActionKind.parse(action.kind)),
        int(action.cursor_x),
        int(action.cursor_y),
    )


class _PortableEnv:
    physics_backend = "portable"

    def __init__(
        self,
        observation: Mapping[str, Any],
        behaviors: Mapping[tuple[int, int, int], int],
    ) -> None:
        self.observation = copy.deepcopy(dict(observation))
        self.behaviors = dict(behaviors)
        self.branch_key: tuple[int, int, int] | None = None
        self.clone_calls = 0
        self.fail_clone_at: int | None = None

    def set_observation(self, observation: Mapping[str, Any]) -> None:
        self.observation = copy.deepcopy(dict(observation))
        self.branch_key = None

    def clone_state(self) -> tuple[dict[str, Any], tuple[int, int, int] | None]:
        self.clone_calls += 1
        if self.clone_calls == self.fail_clone_at:
            raise RuntimeError("synthetic clone failure")
        return copy.deepcopy(self.observation), self.branch_key

    def restore_state(
        self,
        snapshot: tuple[dict[str, Any], tuple[int, int, int] | None],
    ) -> dict[str, Any]:
        self.observation, self.branch_key = copy.deepcopy(snapshot)
        return copy.deepcopy(self.observation)

    def step(
        self, action: Action
    ) -> tuple[dict[str, Any], int, bool, bool, dict[str, Any]]:
        kind = ActionKind.parse(action.kind)
        duration = int(action.wait_ticks) if kind is ActionKind.WAIT else 1
        reward = 0
        if kind is not ActionKind.WAIT:
            self.branch_key = (
                int(kind),
                int(action.cursor_x),
                int(action.cursor_y),
            )
            reward = self.behaviors.get(self.branch_key, 0)
            self.observation["score"] = int(self.observation["score"]) + reward
        self.observation["tick"] = int(self.observation["tick"]) + duration
        return (
            copy.deepcopy(self.observation),
            reward,
            False,
            False,
            {"events": [], "invalid_action": False},
        )


def _fixture(
    *,
    observation: Mapping[str, Any] | None = None,
    ensemble_biases: tuple[Mapping[int, float], ...] | None = None,
    policy_config: GeometryPolicyConfig | None = None,
) -> tuple[
    dict[str, Any],
    _PortableEnv,
    _BasePolicy,
    SafeguardedGeometryPolicy,
    RunwayGeometrySearch,
    SteeringDecision,
]:
    current = (
        _observation() if observation is None else copy.deepcopy(dict(observation))
    )
    geometry = GeometrySearchConfig()
    base = _BasePolicy()
    student = _student(
        base,
        geometry,
        ensemble_biases=ensemble_biases,
        policy_config=policy_config,
    )
    teacher = RunwayGeometrySearch(
        config=RunwaySearchConfig(runway_ticks=8, candidate_config=geometry)
    )
    candidates = enumerate_geometry_candidates(
        current, base.fixed_decision, config=geometry
    )
    preferred = candidates.candidate_at(1)
    assert preferred is not None
    env = _PortableEnv(current, {_action_key(preferred.decision): 100})
    return current, env, base, student, teacher, preferred.decision


def _dagger(
    env: _PortableEnv,
    base: _BasePolicy,
    student: SafeguardedGeometryPolicy,
    teacher: Any,
    config: GeometryDaggerConfig,
) -> LearnerVisitedGeometryDaggerPolicy:
    return LearnerVisitedGeometryDaggerPolicy(
        env=env,
        base_policy=base,
        student_policy=student,
        teacher=teacher,
        source_identity=_SOURCE_SHA256,
        runtime_sha256=_RUNTIME_SHA256,
        config=config,
    )


class GeometryDaggerTests(unittest.TestCase):
    def test_select_from_incumbent_never_advances_base_policy(self) -> None:
        observation, _env, base, student, _teacher, _preferred = _fixture()
        student.reset_geometry_state()

        first = student.select_from_incumbent(observation, base.fixed_decision)
        second = student.select_from_incumbent(observation, base.fixed_decision)

        self.assertIs(first, second)
        self.assertEqual(base.calls, 0)
        with self.assertRaisesRegex(RuntimeError, "changed within one tick"):
            student.select_from_incumbent(
                observation,
                replace(base.fixed_decision, reason="changed incumbent"),
            )
        with self.assertRaisesRegex(TypeError, "SteeringDecision"):
            student.select_from_incumbent(observation, object())  # type: ignore[arg-type]
        student.reset_geometry_state()
        student.predict(observation)
        self.assertEqual(base.calls, 1)

    def test_low_gauge_collects_exact_pre_action_label_but_executes_student(
        self,
    ) -> None:
        observation, env, base, student, teacher, preferred = _fixture()
        policy = _dagger(
            env,
            base,
            student,
            teacher,
            GeometryDaggerConfig(
                execution_mode="student",
                cadence_shots=None,
                low_gauge_fraction=0.25,
                query_on_rejection=False,
                query_on_disagreement=False,
            ),
        )
        before = env.clone_state()
        policy.reset(7)

        decision = policy.predict(observation)
        repeated = policy.predict({**observation, "gauge": 1})

        candidates = enumerate_geometry_candidates(
            observation, base.fixed_decision, config=student.geometry_config
        )
        expected_student = candidates.candidate_at(2)
        assert expected_student is not None
        self.assertEqual(decision, expected_student.decision)
        self.assertIs(repeated, decision)
        self.assertNotEqual(decision, preferred)
        self.assertEqual(base.calls, 1)
        self.assertEqual(env.clone_state(), before)
        self.assertEqual(len(policy.ranking_examples), 1)
        self.assertEqual(policy.ranking_examples[0].winner_index, 1)
        expected = TeacherStateEncoder().encode([observation])
        np.testing.assert_array_equal(
            policy.ranking_examples[0].observation.global_features,
            expected.global_features,
        )
        report = policy.statistics()
        self.assertEqual(report["search_queries"], 1)
        self.assertEqual(report["transactional_restore_checks"], 1)
        self.assertEqual(report["low_gauge_triggers"], 1)
        self.assertEqual(report["student_actions_executed"], 1)
        self.assertEqual(report["oracle_actions_executed"], 0)
        self.assertEqual(len(policy.dataset()), 1)
        self.assertEqual(base.reset_calls, 1)

    def test_oracle_execution_is_explicit(self) -> None:
        observation, env, base, student, teacher, preferred = _fixture()
        policy = _dagger(
            env,
            base,
            student,
            teacher,
            GeometryDaggerConfig(
                execution_mode="oracle",
                cadence_shots=1,
                low_gauge_fraction=None,
                query_on_rejection=False,
                query_on_disagreement=False,
            ),
        )
        policy.reset(8)

        self.assertEqual(policy.predict(observation), preferred)
        report = policy.statistics()
        self.assertEqual(report["oracle_actions_executed"], 1)
        self.assertEqual(report["student_actions_executed"], 0)
        self.assertEqual(policy.query_records[0].execution_mode, "oracle")

    def test_rejection_and_disagreement_form_one_or_query(self) -> None:
        observation, env, base, student, teacher, _preferred = _fixture(
            observation=_observation(gauge=100),
            ensemble_biases=({1: 10.0}, {2: 10.0}),
        )
        policy = _dagger(
            env,
            base,
            student,
            teacher,
            GeometryDaggerConfig(
                cadence_shots=None,
                low_gauge_fraction=None,
                query_on_rejection=True,
                query_on_disagreement=True,
                disagreement_below=1.0,
            ),
        )
        policy.reset(9)

        self.assertIs(policy.predict(observation), base.fixed_decision)
        self.assertEqual(
            policy.query_records[0].reasons,
            ("safeguard_rejection", "ensemble_disagreement"),
        )
        report = policy.statistics()
        self.assertEqual(report["search_queries"], 1)
        self.assertEqual(report["safeguard_rejection_triggers"], 1)
        self.assertEqual(report["ensemble_disagreement_triggers"], 1)

    def test_gauge_reserve_rejection_is_queried_as_safety_dagger(self) -> None:
        observation, env, base, student, teacher, _preferred = _fixture(
            observation=_observation(gauge=7),
            policy_config=GeometryPolicyConfig(
                minimum_confidence=0.0,
                minimum_logit_margin=0.0,
                minimum_gauge_fraction=0.075,
            ),
        )
        policy = _dagger(
            env,
            base,
            student,
            teacher,
            GeometryDaggerConfig(
                cadence_shots=None,
                low_gauge_fraction=None,
                query_on_rejection=True,
                query_on_disagreement=False,
            ),
        )
        policy.reset(17)

        self.assertIs(policy.predict(observation), base.fixed_decision)
        self.assertEqual(policy.query_records[0].reasons, ("safeguard_rejection",))
        self.assertEqual(policy.statistics()["search_queries"], 1)
        self.assertEqual(student.statistics()["gauge_reserve_rejections"], 1)

    def test_exhausted_progress_credit_is_queried_as_safety_dagger(self) -> None:
        _observation_unused, env, base, student, teacher, _preferred = _fixture(
            observation=_observation(gauge=100),
            policy_config=GeometryPolicyConfig(
                minimum_confidence=0.0,
                minimum_logit_margin=0.0,
                maximum_unverified_corrections=2,
            ),
        )
        policy = _dagger(
            env,
            base,
            student,
            teacher,
            GeometryDaggerConfig(
                cadence_shots=None,
                low_gauge_fraction=None,
                query_on_rejection=True,
                query_on_disagreement=False,
            ),
        )
        policy.reset(18)

        for tick in (1, 2, 3):
            observation = _observation(tick=tick, gauge=100)
            env.set_observation(observation)
            policy.predict(observation)

        self.assertEqual(len(policy.query_records), 1)
        self.assertEqual(policy.query_records[0].reasons, ("safeguard_rejection",))
        self.assertEqual(policy.query_records[0].eligible_shot_index, 3)
        self.assertEqual(student.statistics()["progress_credit_rejections"], 1)

    def test_cadence_is_deterministic_across_learner_visited_ticks(self) -> None:
        _observation_unused, env, base, student, teacher, _preferred = _fixture(
            observation=_observation(gauge=100)
        )
        policy = _dagger(
            env,
            base,
            student,
            teacher,
            GeometryDaggerConfig(
                cadence_shots=2,
                low_gauge_fraction=None,
                query_on_rejection=False,
                query_on_disagreement=False,
            ),
        )
        policy.reset(10)

        for tick in (1, 2, 3):
            current = _observation(tick=tick, gauge=100)
            env.set_observation(current)
            policy.predict(current)

        self.assertEqual(base.calls, 3)
        self.assertEqual(
            [record.eligible_shot_index for record in policy.query_records],
            [1, 3],
        )
        report = policy.statistics()
        self.assertEqual(report["search_queries"], 2)
        self.assertEqual(report["cadence_triggers"], 2)
        self.assertEqual(report["student_actions_executed"], 3)

    def test_mutating_teacher_latches_fail_closed(self) -> None:
        observation, env, base, student, teacher, _preferred = _fixture()

        class MutatingTeacher:
            config = teacher.config
            action_spec = teacher.action_spec
            sha256 = teacher.sha256

            @staticmethod
            def search(
                target: _PortableEnv,
                current: Mapping[str, Any],
                incumbent: SteeringDecision,
            ):
                result = teacher.search(target, current, incumbent)
                target.step(Action.wait(1))
                return result

        policy = _dagger(
            env,
            base,
            student,
            MutatingTeacher(),
            GeometryDaggerConfig(
                cadence_shots=1,
                low_gauge_fraction=None,
                query_on_rejection=False,
                query_on_disagreement=False,
            ),
        )
        policy.reset(11)

        with self.assertRaisesRegex(RuntimeError, "changed its source state"):
            policy.predict(observation)
        calls = base.calls
        with self.assertRaisesRegex(RuntimeError, "latched"):
            policy.predict(observation)
        with self.assertRaisesRegex(RuntimeError, "latch is permanent"):
            policy.reset(12)
        self.assertEqual(base.calls, calls)
        self.assertTrue(policy.statistics()["failure_latched"])

    def test_spoofed_per_state_candidate_set_is_rejected(self) -> None:
        observation, env, base, student, teacher, _preferred = _fixture()

        class SpoofedTeacher:
            config = teacher.config
            action_spec = teacher.action_spec
            sha256 = teacher.sha256

            @staticmethod
            def search(
                target: _PortableEnv,
                current: Mapping[str, Any],
                incumbent: SteeringDecision,
            ):
                result = teacher.search(target, current, incumbent)
                spoofed = replace(
                    result.candidate_set,
                    predicted_surface_gap_sizes=(
                        result.candidate_set.predicted_surface_gap_sizes + 1.0
                    ),
                )
                return replace(result, candidate_set=spoofed)

        policy = _dagger(
            env,
            base,
            student,
            SpoofedTeacher(),
            GeometryDaggerConfig(cadence_shots=1, low_gauge_fraction=None),
        )
        policy.reset(14)

        with self.assertRaisesRegex(RuntimeError, "inconsistent evidence"):
            policy.predict(observation)
        self.assertTrue(policy.statistics()["failure_latched"])

    def test_unverifiable_post_search_restore_latches_fail_closed(self) -> None:
        observation, env, base, student, teacher, _preferred = _fixture()
        env.fail_clone_at = 3
        policy = _dagger(
            env,
            base,
            student,
            teacher,
            GeometryDaggerConfig(cadence_shots=1, low_gauge_fraction=None),
        )
        policy.reset(15)

        with self.assertRaisesRegex(RuntimeError, "could not be verified"):
            policy.predict(observation)
        with self.assertRaisesRegex(RuntimeError, "latch is permanent"):
            policy.reset(16)

    def test_reset_advances_base_once_and_clears_collected_data(self) -> None:
        observation, env, base, student, teacher, _preferred = _fixture()
        policy = _dagger(
            env,
            base,
            student,
            teacher,
            GeometryDaggerConfig(cadence_shots=1, low_gauge_fraction=None),
        )
        policy.reset(12)
        policy.predict(observation)
        self.assertEqual(base.reset_calls, 1)
        self.assertEqual(len(policy.ranking_examples), 1)

        policy.reset(13)

        self.assertEqual(base.reset_calls, 2)
        self.assertEqual(policy.ranking_examples, ())
        self.assertEqual(policy.query_records, ())
        self.assertEqual(policy.statistics()["decisions"], 0)

    def test_constructor_rejects_foreign_owner_and_vocabulary(self) -> None:
        _observation_unused, env, base, student, _teacher, _preferred = _fixture()
        foreign_base = _BasePolicy()
        with self.assertRaisesRegex(ValueError, "exactly one owner"):
            _dagger(
                env,
                foreign_base,
                student,
                RunwayGeometrySearch(),
                GeometryDaggerConfig(),
            )

        foreign_geometry = GeometrySearchConfig(max_candidates=25)
        foreign_teacher = RunwayGeometrySearch(
            config=RunwaySearchConfig(runway_ticks=8, candidate_config=foreign_geometry)
        )
        with self.assertRaisesRegex(ValueError, "vocabularies differ"):
            _dagger(
                env,
                base,
                student,
                foreign_teacher,
                GeometryDaggerConfig(),
            )

        credit_student = _student(
            base,
            GeometrySearchConfig(),
            policy_config=GeometryPolicyConfig(maximum_unverified_corrections=2),
        )
        with self.assertRaisesRegex(ValueError, "causally account"):
            _dagger(
                env,
                base,
                credit_student,
                RunwayGeometrySearch(),
                GeometryDaggerConfig(execution_mode="oracle"),
            )


if __name__ == "__main__":
    unittest.main()
