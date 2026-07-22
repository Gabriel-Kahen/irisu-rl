from __future__ import annotations

import struct
import unittest

from irisu_env import Action
from irisu_rl.actions import ActionSpec
from irisu_rl.curriculum import SnapshotBlobStore
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.r3b_evaluation import (
    EvaluationSuite,
    ScriptedBaselineSpec,
    RecurrentSemanticPolicy,
    evaluate_recurrent_policy,
    evaluate_scripted_baseline,
    semantic_from_native,
)
from irisu_rl.schema import TEACHER_V1
from tests.test_rl_vector_adapter import observation

import torch
from tests.test_r3b_snapshot_initializer import (
    _RUNTIME_SHA256,
    _fixture,
)


_SNAPSHOT = struct.Struct("<qqqQ")


class FakeSingleSimulator:
    def __init__(self) -> None:
        self.tick = 0
        self.score = 0
        self.gauge = 100
        self.hash = 0

    def _observation(self):
        return {
            "tick": self.tick,
            "score": self.score,
            "gauge": self.gauge,
            "gauge_max": 1000,
            "bodies": (),
        }

    def restore_state(self, snapshot: bytes):
        self.tick, self.score, self.gauge, self.hash = _SNAPSHOT.unpack(snapshot)
        return self._observation()

    def state_hash(self) -> int:
        return int(self.hash)

    def step(self, action: Action):
        delta = action.wait_ticks if int(action.kind) == 0 else 1
        self.tick += int(delta)
        self.score += int(delta)
        self.gauge = max(0, self.gauge - int(delta))
        self.hash += int(delta)
        return self._observation(), int(delta), False, False, {"invalid_action": False}


class R3BEvaluationTests(unittest.TestCase):
    def test_fixed_cells_use_raw_score_delta_and_macro_bounds(self) -> None:
        spec, blobs = _fixture()
        store = SnapshotBlobStore(spec.library, blobs)
        suite = EvaluationSuite(
            "validation-v1",
            "validation",
            ("validation",),
            2,
            17,
            3,
            3,
            _RUNTIME_SHA256,
            spec.assignment_sha256,
        )
        baseline = ScriptedBaselineSpec("no_action_long_wait", (("wait_ticks", 1),))
        report = evaluate_scripted_baseline(
            FakeSingleSimulator(),
            store,
            suite,
            baseline,
            evaluator_sha256="e" * 64,
        )
        self.assertEqual(len(report.episodes), 2)
        self.assertEqual([value.raw_score for value in report.episodes], [3, 3])
        self.assertTrue(all(value.truncated for value in report.episodes))
        self.assertTrue(all(value.invalid_actions == 0 for value in report.episodes))
        self.assertNotEqual(report.sha256, "0" * 64)

    def test_split_runtime_and_simultaneous_action_fail_closed(self) -> None:
        spec, blobs = _fixture()
        store = SnapshotBlobStore(spec.library, blobs)
        wrong_split = EvaluationSuite(
            "wrong-split-v1",
            "test",
            ("validation",),
            1,
            19,
            1,
            1,
            _RUNTIME_SHA256,
            spec.assignment_sha256,
        )
        with self.assertRaisesRegex(ValueError, "wrong split"):
            evaluate_scripted_baseline(
                FakeSingleSimulator(),
                store,
                wrong_split,
                ScriptedBaselineSpec("matcher_shot_policy"),
                evaluator_sha256="e" * 64,
            )
        with self.assertRaisesRegex(ValueError, "simultaneous"):
            semantic_from_native(Action.both(10, 20), ActionSpec())

    def test_all_required_scripted_baselines_are_buildable(self) -> None:
        baseline_ids = (
            "no_action_long_wait",
            "seeded_legal_random",
            "matcher_shot_policy",
            "scripted_direct_matcher",
            "scripted_side_ejector",
            "scripted_imminent_rot_hazard",
        )
        for baseline_id in baseline_ids:
            policy = ScriptedBaselineSpec(baseline_id).build(23)
            self.assertTrue(callable(policy.act))

    def test_recurrent_evaluator_is_deterministic_and_critic_alpha_is_zero(
        self,
    ) -> None:
        torch.manual_seed(31)
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1, critic_condition_features=1),
        )
        encoded = TeacherStateEncoder().encode((observation(0),))
        kind = torch.ones((1, 3), dtype=torch.bool)
        wait = torch.ones((1, len(ActionSpec().wait_choices)), dtype=torch.bool)
        policy = RecurrentSemanticPolicy(model)
        policy.reset(1)
        first = policy.act(encoded, kind, wait)
        policy.reset(1)
        second = policy.act(encoded, kind, wait)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)

        spec, blobs = _fixture()
        store = SnapshotBlobStore(spec.library, blobs)
        suite = EvaluationSuite(
            "recurrent-validation-v1",
            "validation",
            ("validation",),
            1,
            31,
            2,
            2,
            _RUNTIME_SHA256,
            spec.assignment_sha256,
        )
        eval_wait = torch.zeros_like(wait)
        eval_wait[:, 0] = True
        report = evaluate_recurrent_policy(
            FakeSingleSimulator(),
            store,
            suite,
            model,
            TeacherStateEncoder(),
            kind,
            eval_wait,
            evaluator_sha256="e" * 64,
        )
        self.assertEqual(len(report.episodes), 1)
        self.assertEqual(report.episodes[0].raw_score, 2)


if __name__ == "__main__":
    unittest.main()
