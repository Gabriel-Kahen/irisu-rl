from __future__ import annotations

import unittest

import torch

from irisu_rl.actions import ActionSpec
from irisu_rl.curriculum import SnapshotBlobStore
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.r3b_evaluation import (
    EvaluationSuite,
    evaluate_recurrent_policy,
    evaluate_recurrent_policy_vectorized,
)
from irisu_rl.schema import TEACHER_V1
from tests.test_r3b_evaluation import (
    FakeSingleSimulator,
    FakeTerminalUnderflowSimulator,
)
from tests.test_r3b_snapshot_initializer import _RUNTIME_SHA256, _fixture


class FakePaddedVectorSimulator:
    """Subset-compatible vector façade with one independent simulator per lane."""

    def __init__(self, lanes: int, lane_factory=FakeSingleSimulator) -> None:
        self.envs = tuple(lane_factory() for _ in range(lanes))
        self.num_envs = lanes
        self.initialized = [False] * lanes
        self.reset_many_calls: list[tuple[int, ...]] = []
        self.restore_many_calls: list[tuple[int, ...]] = []
        self.step_many_calls: list[tuple[int, ...]] = []

    def restore_many(self, indices, snapshots):
        lanes = tuple(indices)
        supplied = tuple(snapshots)
        if len(lanes) != len(supplied) or len(set(lanes)) != len(lanes):
            raise ValueError("invalid restore subset")
        if any(not self.initialized[lane] for lane in lanes):
            raise RuntimeError("restore requires initialized rollback state")
        self.restore_many_calls.append(lanes)
        return [
            self.envs[lane].restore_state(snapshot)
            for lane, snapshot in zip(lanes, supplied)
        ]

    def reset_many(self, indices, *, seeds):
        lanes = tuple(indices)
        supplied = tuple(seeds)
        if len(lanes) != len(supplied) or len(set(lanes)) != len(lanes):
            raise ValueError("invalid reset subset")
        self.reset_many_calls.append(lanes)
        for lane in lanes:
            self.initialized[lane] = True
        return [self.envs[lane]._observation() for lane in lanes]

    def step_many(self, indices, actions):
        lanes = tuple(indices)
        supplied = tuple(actions)
        if len(lanes) != len(supplied) or len(set(lanes)) != len(lanes):
            raise ValueError("invalid step subset")
        self.step_many_calls.append(lanes)
        results = [
            self.envs[lane].step(action) for lane, action in zip(lanes, supplied)
        ]
        observations, rewards, terminated, truncated, infos = zip(*results)
        return (
            list(observations),
            list(rewards),
            list(terminated),
            list(truncated),
            list(infos),
        )

    def state_hash_many(self, indices):
        return tuple(self.envs[lane].state_hash() for lane in indices)

    def config_hash_many(self, indices):
        return tuple(self.envs[lane].config_hash() for lane in indices)


class WrongStateVectorSimulator(FakePaddedVectorSimulator):
    def state_hash_many(self, indices):
        values = list(super().state_hash_many(indices))
        values[-1] += 1
        return tuple(values)


def _model() -> RecurrentActorCritic:
    torch.manual_seed(20260723)
    return RecurrentActorCritic(
        TEACHER_V1,
        config=RecurrentModelConfig(8, 8, 12, 12, 1),
    )


def _suite(
    store: SnapshotBlobStore,
    *,
    repetitions: int,
    max_decisions: int = 4,
    max_ticks: int = 4,
) -> EvaluationSuite:
    recipe = store.library["validation"]
    spec, _ = _fixture()
    return EvaluationSuite(
        "vector-validation-v1",
        "validation",
        ("validation",),
        repetitions,
        20260723,
        max_decisions,
        max_ticks,
        _RUNTIME_SHA256,
        spec.assignment_sha256,
        store.library.sha256,
        store.sha256,
        ActionSpec().sha256,
        (recipe.sha256,),
    )


def _evaluate(
    simulator,
    store: SnapshotBlobStore,
    suite: EvaluationSuite,
    model: RecurrentActorCritic,
    kind_mask: torch.Tensor,
    wait_mask: torch.Tensor,
    *,
    vector: bool,
    cells=None,
):
    evaluator = (
        evaluate_recurrent_policy_vectorized if vector else evaluate_recurrent_policy
    )
    return evaluator(
        simulator,
        store,
        suite,
        model,
        TeacherStateEncoder(),
        kind_mask,
        wait_mask,
        evaluator_sha256="e" * 64,
        expected_assignment_sha256=suite.assignment_sha256,
        execution_identity_sha256="f" * 64,
        cells=cells,
    )


class R3BVectorEvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        spec, blobs = _fixture()
        self.store = SnapshotBlobStore(spec.library, blobs)

    def test_wait_only_matches_single_lane_across_batches_and_order(self) -> None:
        model = _model()
        suite = _suite(self.store, repetitions=5, max_decisions=4, max_ticks=4)
        kind_mask = torch.zeros((1, 3), dtype=torch.bool)
        kind_mask[:, 0] = True
        wait_mask = torch.zeros(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
        wait_mask[:, 0] = True
        cells = (
            ("validation", 4),
            ("validation", 0),
            ("validation", 3),
        )
        single = _evaluate(
            FakeSingleSimulator(),
            self.store,
            suite,
            model,
            kind_mask,
            wait_mask,
            vector=False,
            cells=cells,
        )
        vector_simulator = FakePaddedVectorSimulator(2)
        vector = _evaluate(
            vector_simulator,
            self.store,
            suite,
            model,
            kind_mask,
            wait_mask,
            vector=True,
            cells=cells,
        )
        self.assertEqual(vector, single)
        self.assertEqual(
            tuple((value.snapshot_id, value.repetition) for value in vector.episodes),
            (("validation", 0), ("validation", 3), ("validation", 4)),
        )
        self.assertEqual(vector_simulator.restore_many_calls, [(0, 1), (0,)])
        self.assertEqual(vector_simulator.reset_many_calls, [(0, 1)])

    def test_shot_macro_subset_and_signed_gauge_match_single_lane(self) -> None:
        model = _model()
        suite = _suite(self.store, repetitions=3, max_decisions=4, max_ticks=4)
        kind_mask = torch.zeros((1, 3), dtype=torch.bool)
        kind_mask[:, 1] = True
        wait_mask = torch.ones(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
        single = _evaluate(
            FakeTerminalUnderflowSimulator(),
            self.store,
            suite,
            model,
            kind_mask,
            wait_mask,
            vector=False,
        )
        vector_simulator = FakePaddedVectorSimulator(2, FakeTerminalUnderflowSimulator)
        vector = _evaluate(
            vector_simulator,
            self.store,
            suite,
            model,
            kind_mask,
            wait_mask,
            vector=True,
        )
        self.assertEqual(vector, single)
        self.assertTrue(all(value.terminated for value in vector.episodes))
        self.assertTrue(all(value.minimum_gauge == -48 for value in vector.episodes))
        self.assertIn((0, 1), vector_simulator.step_many_calls)
        self.assertIn((0,), vector_simulator.step_many_calls)

    def test_rejects_nonvector_and_identity_mismatch_before_execution(self) -> None:
        model = _model()
        suite = _suite(self.store, repetitions=1)
        kind_mask = torch.ones((1, 3), dtype=torch.bool)
        wait_mask = torch.ones(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
        with self.assertRaisesRegex(ValueError, "subset-capable"):
            _evaluate(
                FakeSingleSimulator(),
                self.store,
                suite,
                model,
                kind_mask,
                wait_mask,
                vector=True,
            )
        vector = FakePaddedVectorSimulator(2)
        with self.assertRaisesRegex(ValueError, "assignment identity"):
            evaluate_recurrent_policy_vectorized(
                vector,
                self.store,
                suite,
                model,
                TeacherStateEncoder(),
                kind_mask,
                wait_mask,
                evaluator_sha256="e" * 64,
                expected_assignment_sha256="a" * 64,
                execution_identity_sha256="f" * 64,
            )
        self.assertEqual(vector.restore_many_calls, [])

        wrong_state = WrongStateVectorSimulator(2)
        with self.assertRaisesRegex(ValueError, "state hash mismatch"):
            _evaluate(
                wrong_state,
                self.store,
                suite,
                model,
                kind_mask,
                wait_mask,
                vector=True,
            )
        self.assertEqual(wrong_state.step_many_calls, [])


if __name__ == "__main__":
    unittest.main()
