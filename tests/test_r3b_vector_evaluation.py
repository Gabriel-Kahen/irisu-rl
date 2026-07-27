from __future__ import annotations

import hashlib
import struct
import unittest
from dataclasses import replace

import torch
from irisu_rl.actions import ActionSpec
from irisu_rl.curriculum import SnapshotBlobStore, SnapshotLibrary
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.r3b_evaluation import (
    EvaluationSuite,
    evaluate_recurrent_policy,
    evaluate_recurrent_policy_vectorized,
)
from irisu_rl.schema import TEACHER_V1

from tests.test_r3b_evaluation import (
    FakeHorizonUnderflowSimulator,
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
        self.step_calls = 0
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

    def _step(self, lanes, actions):
        supplied = tuple(actions)
        if len(lanes) != len(supplied) or len(set(lanes)) != len(lanes):
            raise ValueError("invalid step subset")
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

    def step(self, actions):
        self.step_calls += 1
        return self._step(tuple(range(self.num_envs)), actions)

    def step_many(self, indices, actions):
        lanes = tuple(indices)
        self.step_many_calls.append(lanes)
        return self._step(lanes, actions)

    def state_hash_many(self, indices):
        return tuple(self.envs[lane].state_hash() for lane in indices)

    def config_hash_many(self, indices):
        return tuple(self.envs[lane].config_hash() for lane in indices)


class WrongStateVectorSimulator(FakePaddedVectorSimulator):
    def state_hash_many(self, indices):
        values = list(super().state_hash_many(indices))
        values[-1] += 1
        return tuple(values)


class FakeDurationSimulator(FakeSingleSimulator):
    def restore_state(self, snapshot: bytes):
        observation = super().restore_state(snapshot)
        self.remaining = int(self.gauge) - 100
        return observation

    def step(self, action):
        self.tick += 1
        self.score += 1
        self.gauge = max(0, self.gauge - 1)
        self.remaining -= 1
        return (
            self._observation(),
            1,
            self.remaining == 0,
            False,
            {"invalid_action": False},
        )


class RecordingTeacherEncoder(TeacherStateEncoder):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def encode(self, observations):
        self.batch_sizes.append(len(observations))
        return super().encode(observations)


def _duration_store() -> SnapshotBlobStore:
    spec, _ = _fixture()
    template = spec.library["validation"]
    snapshot_struct = struct.Struct("<qqqQ")
    definitions = (
        ("a-slow", 3, 101),
        ("b-fast", 1, 102),
        ("c-fast", 1, 103),
        ("d-fast", 1, 104),
    )
    blobs = {
        snapshot_id: snapshot_struct.pack(0, 0, 100 + duration, state_hash)
        for snapshot_id, duration, state_hash in definitions
    }
    recipes = tuple(
        replace(
            template,
            snapshot_id=snapshot_id,
            scenario_family=f"family-{snapshot_id}",
            reset_seed=1000 + index,
            expected_tick=0,
            expected_score=0,
            expected_state_hash=state_hash,
            snapshot_sha256=hashlib.sha256(blobs[snapshot_id]).hexdigest(),
        )
        for index, (snapshot_id, _duration, state_hash) in enumerate(definitions)
    )
    return SnapshotBlobStore(SnapshotLibrary(recipes), blobs)


def _duration_suite(
    store: SnapshotBlobStore,
    model: RecurrentActorCritic,
    *,
    snapshot_ids: tuple[str, ...] | None = None,
) -> EvaluationSuite:
    spec, _ = _fixture()
    selected = (
        tuple(recipe.snapshot_id for recipe in store.library.recipes)
        if snapshot_ids is None
        else snapshot_ids
    )
    return EvaluationSuite(
        "work-conserving-validation-v1",
        "validation",
        selected,
        1,
        20260724,
        4,
        4,
        _RUNTIME_SHA256,
        spec.assignment_sha256,
        store.library.sha256,
        store.sha256,
        model.action_spec.sha256,
        tuple(store.library[snapshot_id].sha256 for snapshot_id in selected),
    )


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
    encoder=None,
):
    evaluator = (
        evaluate_recurrent_policy_vectorized if vector else evaluate_recurrent_policy
    )
    return evaluator(
        simulator,
        store,
        suite,
        model,
        TeacherStateEncoder() if encoder is None else encoder,
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
        self.assertTrue(all(value.minimum_gauge == -96 for value in vector.episodes))
        self.assertTrue(all(value.final_gauge == -96 for value in vector.episodes))
        self.assertGreater(vector_simulator.step_calls, 0)
        self.assertIn((0,), vector_simulator.step_many_calls)

    def test_horizon_underflow_matches_single_lane(self) -> None:
        model = _model()
        suite = _suite(self.store, repetitions=3, max_decisions=1, max_ticks=1)
        kind_mask = torch.zeros((1, 3), dtype=torch.bool)
        kind_mask[:, 0] = True
        wait_mask = torch.zeros(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
        wait_mask[:, 0] = True
        single = _evaluate(
            FakeHorizonUnderflowSimulator(),
            self.store,
            suite,
            model,
            kind_mask,
            wait_mask,
            vector=False,
        )
        vector_simulator = FakePaddedVectorSimulator(2, FakeHorizonUnderflowSimulator)
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
        self.assertTrue(all(not value.terminated for value in vector.episodes))
        self.assertTrue(all(value.truncated for value in vector.episodes))
        self.assertTrue(all(value.minimum_gauge == -48 for value in vector.episodes))
        self.assertTrue(all(value.final_gauge == -48 for value in vector.episodes))

    def test_completed_lane_is_refilled_without_waiting_for_slow_lane(self) -> None:
        store = _duration_store()
        model = _model()
        suite = _duration_suite(store, model)
        kind_mask = torch.zeros((1, 3), dtype=torch.bool)
        kind_mask[:, 0] = True
        wait_mask = torch.zeros(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
        wait_mask[:, 0] = True
        single = _evaluate(
            FakeDurationSimulator(),
            store,
            suite,
            model,
            kind_mask,
            wait_mask,
            vector=False,
        )
        vector_simulator = FakePaddedVectorSimulator(2, FakeDurationSimulator)
        vector = _evaluate(
            vector_simulator,
            store,
            suite,
            model,
            kind_mask,
            wait_mask,
            vector=True,
        )
        self.assertEqual(vector, single)
        self.assertEqual(
            vector_simulator.restore_many_calls,
            [(0, 1), (1,), (1,)],
        )

    def test_work_conserving_reports_match_across_lane_counts(self) -> None:
        store = _duration_store()
        model = _model()
        suite = _duration_suite(store, model)
        kind_mask = torch.zeros((1, 3), dtype=torch.bool)
        kind_mask[:, 0] = True
        wait_mask = torch.zeros(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
        wait_mask[:, 0] = True
        expected = _evaluate(
            FakeDurationSimulator(),
            store,
            suite,
            model,
            kind_mask,
            wait_mask,
            vector=False,
        )
        for lanes in (1, 2, 4):
            with self.subTest(lanes=lanes):
                actual = _evaluate(
                    FakePaddedVectorSimulator(lanes, FakeDurationSimulator),
                    store,
                    suite,
                    model,
                    kind_mask,
                    wait_mask,
                    vector=True,
                )
                self.assertEqual(actual, expected)

    def test_inference_excludes_finished_lanes(self) -> None:
        store = _duration_store()
        model = _model()
        snapshot_ids = tuple(recipe.snapshot_id for recipe in store.library.recipes[:2])
        suite = _duration_suite(store, model, snapshot_ids=snapshot_ids)
        kind_mask = torch.zeros((1, 3), dtype=torch.bool)
        kind_mask[:, 0] = True
        wait_mask = torch.zeros(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
        wait_mask[:, 0] = True
        encoder = RecordingTeacherEncoder()
        vector_simulator = FakePaddedVectorSimulator(2, FakeDurationSimulator)
        report = _evaluate(
            vector_simulator,
            store,
            suite,
            model,
            kind_mask,
            wait_mask,
            vector=True,
            encoder=encoder,
        )
        self.assertEqual(len(report.episodes), 2)
        self.assertEqual(encoder.batch_sizes, [2, 1, 1])
        self.assertEqual(vector_simulator.step_calls, 1)
        self.assertEqual(vector_simulator.step_many_calls, [(0,), (0,)])

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
