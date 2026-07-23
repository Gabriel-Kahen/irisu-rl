from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

import torch

from irisu_rl.actions import ActionSpec
from irisu_rl.curriculum import SnapshotBlobStore
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.r3b_evaluation import (
    EvaluationReport,
    EvaluationSuite,
    ScriptedBaselineSpec,
    evaluate_recurrent_policy,
    evaluate_scripted_baseline,
)
from irisu_rl.r3b_evaluation_shards import (
    EvaluationShardPlan,
    EvaluationShardReport,
    evaluate_recurrent_shard,
    evaluate_scripted_shard,
    evaluation_cells,
    merge_evaluation_shards,
    plan_evaluation_shards,
)
from irisu_rl.schema import TEACHER_V1
from tests.test_r3b_evaluation import FakeSingleSimulator
from tests.test_r3b_snapshot_initializer import _RUNTIME_SHA256, _fixture


def _identity(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _suite(
    store: SnapshotBlobStore,
    assignment_sha256: str,
    *,
    action_spec_sha256: str,
) -> EvaluationSuite:
    recipe = store.library["validation"]
    return EvaluationSuite(
        "sharded-validation-v1",
        "validation",
        ("validation",),
        5,
        47,
        2,
        2,
        _RUNTIME_SHA256,
        assignment_sha256,
        store.library.sha256,
        store.sha256,
        action_spec_sha256,
        (recipe.sha256,),
    )


class EvaluationShardTests(unittest.TestCase):
    def setUp(self) -> None:
        spec, blobs = _fixture()
        self.spec = spec
        self.store = SnapshotBlobStore(spec.library, blobs)
        self.suite = _suite(
            self.store,
            spec.assignment_sha256,
            action_spec_sha256=ActionSpec().sha256,
        )

    def test_partition_is_deterministic_complete_and_non_overlapping(self) -> None:
        plans = plan_evaluation_shards(self.suite, 3)
        self.assertEqual(plans, plan_evaluation_shards(self.suite, 3))
        self.assertEqual([value.shard_index for value in plans], [0, 1, 2])
        self.assertTrue(all(value.suite_sha256 == self.suite.sha256 for value in plans))
        assigned = tuple(cell for plan in plans for cell in plan.cells)
        self.assertEqual(set(assigned), set(evaluation_cells(self.suite)))
        self.assertEqual(len(assigned), len(set(assigned)))
        with self.assertRaisesRegex(ValueError, "shard count"):
            plan_evaluation_shards(self.suite, 0)
        with self.assertRaisesRegex(ValueError, "shard count"):
            plan_evaluation_shards(self.suite, 6)

    def test_scripted_shards_merge_to_single_process_result(self) -> None:
        baseline = ScriptedBaselineSpec("no_action_long_wait", (("wait_ticks", 1),))
        plans = plan_evaluation_shards(self.suite, 3)
        shards = tuple(
            evaluate_scripted_shard(
                FakeSingleSimulator(),
                self.store,
                self.suite,
                baseline,
                plan,
                evaluator_sha256=_identity("scripted-evaluator"),
                expected_assignment_sha256=self.spec.assignment_sha256,
                execution_identity_sha256=_identity(
                    f"scripted-worker:{plan.shard_index}"
                ),
            )
            for plan in plans
        )
        merged = merge_evaluation_shards(self.suite, tuple(reversed(shards)))
        replay = merge_evaluation_shards(self.suite, shards)
        single = evaluate_scripted_baseline(
            FakeSingleSimulator(),
            self.store,
            self.suite,
            baseline,
            evaluator_sha256=_identity("scripted-evaluator"),
            expected_assignment_sha256=self.spec.assignment_sha256,
            execution_identity_sha256=_identity("single-process"),
        )
        self.assertEqual(merged, replay)
        self.assertEqual(merged.episodes, single.episodes)
        self.assertEqual(merged.policy_sha256, single.policy_sha256)
        self.assertNotEqual(
            merged.execution_identity_sha256, single.execution_identity_sha256
        )

        changed_report = replace(
            shards[0].report,
            execution_identity_sha256=_identity("replacement-worker"),
        )
        changed = (
            EvaluationShardReport(shards[0].shard, changed_report),
            *shards[1:],
        )
        self.assertNotEqual(
            merge_evaluation_shards(self.suite, changed).execution_identity_sha256,
            merged.execution_identity_sha256,
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            merge_evaluation_shards(self.suite, shards[:-1])
        with self.assertRaisesRegex(ValueError, "duplicated"):
            merge_evaluation_shards(self.suite, (shards[0], shards[0], shards[2]))

    def test_merge_rejects_foreign_cells_and_identity_mismatches(self) -> None:
        baseline = ScriptedBaselineSpec("no_action_long_wait")
        plans = plan_evaluation_shards(self.suite, 2)
        shards = tuple(
            evaluate_scripted_shard(
                FakeSingleSimulator(),
                self.store,
                self.suite,
                baseline,
                plan,
                evaluator_sha256=_identity("evaluator"),
                expected_assignment_sha256=self.spec.assignment_sha256,
                execution_identity_sha256=_identity(f"worker:{plan.shard_index}"),
            )
            for plan in plans
        )
        foreign_episode = replace(shards[0].report.episodes[0], repetition=99)
        foreign_report = replace(
            shards[0].report,
            episodes=(foreign_episode, *shards[0].report.episodes[1:]),
        )
        with self.assertRaisesRegex(ValueError, "disagrees with its plan"):
            EvaluationShardReport(shards[0].shard, foreign_report)

        wrong_policy = EvaluationShardReport(
            shards[0].shard,
            replace(shards[0].report, policy_sha256=_identity("wrong-policy")),
        )
        with self.assertRaisesRegex(ValueError, "identities disagree"):
            merge_evaluation_shards(self.suite, (wrong_policy, shards[1]))
        wrong_seed = EvaluationShardReport(
            shards[0].shard,
            replace(
                shards[0].report,
                episodes=(
                    replace(
                        shards[0].report.episodes[0],
                        policy_seed=shards[0].report.episodes[0].policy_seed + 1,
                    ),
                    *shards[0].report.episodes[1:],
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "seed disagrees"):
            merge_evaluation_shards(self.suite, (wrong_seed, shards[1]))

        foreign_plan = EvaluationShardPlan(
            self.suite.sha256,
            2,
            0,
            (plans[0].cells[1], plans[0].cells[0], *plans[0].cells[2:]),
        )
        with self.assertRaisesRegex(ValueError, "full suite"):
            evaluate_scripted_shard(
                FakeSingleSimulator(),
                self.store,
                self.suite,
                baseline,
                foreign_plan,
                evaluator_sha256=_identity("evaluator"),
                expected_assignment_sha256=self.spec.assignment_sha256,
                execution_identity_sha256=_identity("foreign-plan"),
            )

        with self.assertRaisesRegex(ValueError, "malformed or foreign"):
            evaluate_scripted_baseline(
                FakeSingleSimulator(),
                self.store,
                self.suite,
                baseline,
                evaluator_sha256=_identity("evaluator"),
                expected_assignment_sha256=self.spec.assignment_sha256,
                execution_identity_sha256=_identity("foreign-cell"),
                cells=(("validation", 99),),
            )

    def test_recurrent_shards_use_the_existing_learned_evaluator(self) -> None:
        torch.manual_seed(53)
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1, critic_condition_features=1),
        )
        suite = replace(self.suite, action_spec_sha256=model.action_spec.sha256)
        kind_mask = torch.zeros((1, 3), dtype=torch.bool)
        kind_mask[:, 0] = True
        wait_mask = torch.zeros(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
        wait_mask[:, 0] = True
        plans = plan_evaluation_shards(suite, 2)
        shards = tuple(
            evaluate_recurrent_shard(
                FakeSingleSimulator(),
                self.store,
                suite,
                model,
                TeacherStateEncoder(),
                kind_mask,
                wait_mask,
                plan,
                evaluator_sha256=_identity("learned-evaluator"),
                expected_assignment_sha256=self.spec.assignment_sha256,
                execution_identity_sha256=_identity(
                    f"learned-worker:{plan.shard_index}"
                ),
            )
            for plan in plans
        )
        merged = merge_evaluation_shards(suite, shards)
        single = evaluate_recurrent_policy(
            FakeSingleSimulator(),
            self.store,
            suite,
            model,
            TeacherStateEncoder(),
            kind_mask,
            wait_mask,
            evaluator_sha256=_identity("learned-evaluator"),
            expected_assignment_sha256=self.spec.assignment_sha256,
            execution_identity_sha256=_identity("learned-single"),
        )
        self.assertEqual(merged.episodes, single.episodes)
        self.assertEqual(merged.policy_sha256, single.policy_sha256)
        self.assertIsInstance(merged, EvaluationReport)


if __name__ == "__main__":
    unittest.main()
