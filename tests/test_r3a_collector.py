from __future__ import annotations

import tempfile
import unittest

import torch

from irisu_rl.actions import ActionSpec
from irisu_rl.collector import (
    CollectorConfig,
    PolicySampler,
    R3ATrainingSession,
    RecurrentCollector,
    ScoreTaskContract,
)
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.ppo import PPOConfig, PPOTrainer
from irisu_rl.schema import TEACHER_V1
from irisu_rl.torch_distribution import TorchConditionalActionDistribution
from irisu_rl.vector_adapter import MacroVectorAdapter
from tests.test_rl_vector_adapter import FakeActiveVector, FakeTruncatingVector


class FixedBranchTask(ScoreTaskContract):
    def action_masks(self, action_spec: ActionSpec):
        kind = torch.zeros((self.lanes, 3), dtype=torch.bool)
        kind[0, 0] = True
        kind[1, 0] = True
        kind[2, 1] = True
        kind[3, 2] = True
        wait = torch.zeros(
            (self.lanes, len(action_spec.wait_choices)), dtype=torch.bool
        )
        wait[0, action_spec.wait_choices.index(1)] = True
        wait[1, action_spec.wait_choices.index(8)] = True
        return kind, wait


def small_model() -> RecurrentActorCritic:
    return RecurrentActorCritic(
        TEACHER_V1,
        config=RecurrentModelConfig(8, 8, 12, 12, 1),
    )


def make_collector(model, env, *, decisions: int, sampler_seed: int = 13):
    adapter = MacroVectorAdapter(env, encoder=TeacherStateEncoder())
    task = FixedBranchTask(4)
    return RecurrentCollector(
        model,
        adapter,
        task,
        config=CollectorConfig(max_decisions=decisions, lambda_tick=0.95),
        policy_sampler_seed=sampler_seed,
    )


class R3ACollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(101)

    def test_mixed_macros_flow_through_production_buffer_and_reset_once(self) -> None:
        collector = make_collector(small_model(), FakeActiveVector(), decisions=2)
        collector.initialize()
        rollout = collector.collect()
        batch = rollout.batch
        self.assertEqual(batch.global_features.shape[:2], (2, 4))
        self.assertTrue(torch.all(batch.reset_before[0]))
        self.assertEqual(batch.reset_before[1].tolist(), [False, False, False, True])
        self.assertEqual(
            rollout.audit.decisions[0].elapsed_ticks,
            (1, 8, 2, 1),
        )
        self.assertEqual(
            [int(action.kind) for action in rollout.audit.decisions[0].actions],
            [0, 0, 1, 2],
        )
        self.assertEqual(rollout.audit.invalid_actions, 0)
        self.assertEqual(rollout.audit.completed_episodes, 2)
        self.assertTrue(torch.all(batch.train_mask))

    def test_rollout_chunk_boundary_does_not_reset_recurrence_or_resample(self) -> None:
        base = small_model()
        state = {key: value.clone() for key, value in base.state_dict().items()}
        whole_model = small_model()
        whole_model.load_state_dict(state)
        split_model = small_model()
        split_model.load_state_dict(state)
        whole = make_collector(whole_model, FakeActiveVector(), decisions=2)
        split = make_collector(split_model, FakeActiveVector(), decisions=1)
        whole.initialize()
        split.initialize()
        whole_rollout = whole.collect()
        first = split.collect()
        second = split.collect()
        self.assertEqual(
            whole_rollout.audit.decisions,
            first.audit.decisions + second.audit.decisions,
        )
        self.assertFalse(torch.any(second.batch.reset_before[0, :3]))
        self.assertTrue(second.batch.reset_before[0, 3])
        torch.testing.assert_close(
            whole_rollout.batch.global_features[1], second.batch.global_features[0]
        )

    def test_truncation_bootstraps_neutral_wait_and_censors_held_shot(self) -> None:
        collector = make_collector(small_model(), FakeTruncatingVector(), decisions=1)
        collector.initialize()
        batch = collector.collect().batch
        self.assertTrue(batch.train_mask[0, 1])
        self.assertFalse(batch.train_mask[0, 2])
        # Lane 1 is a neutral WAIT truncation and therefore contains a value
        # bootstrap. Lane 2 is interrupted while held and is excluded entirely.
        self.assertFalse(torch.equal(batch.returns[0, 1], batch.advantages[0, 1]))
        self.assertEqual(float(batch.advantages[0, 2]), 0.0)
        self.assertEqual(float(batch.returns[0, 2]), 0.0)

    def test_tick_budget_stops_only_after_a_complete_synchronous_row(self) -> None:
        model = small_model()
        adapter = MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder())
        collector = RecurrentCollector(
            model,
            adapter,
            FixedBranchTask(4),
            config=CollectorConfig(
                max_decisions=10, target_simulated_ticks=10, lambda_tick=0.9
            ),
            policy_sampler_seed=5,
        )
        collector.initialize()
        result = collector.collect()
        self.assertEqual(result.audit.decision_rows, 1)
        self.assertEqual(result.audit.simulated_ticks, 12)
        self.assertEqual(result.audit.tick_target_overshoot, 2)
        self.assertEqual(result.audit.transitions, 4)

    def test_policy_sampler_does_not_change_global_torch_rng(self) -> None:
        torch.manual_seed(29)
        expected_state = torch.get_rng_state().clone()
        logits = torch.zeros((1, 2, 3))
        wait = torch.zeros((1, 2, len(ActionSpec().wait_choices)))
        concentration = torch.full((1, 2, 2, 2), 2.0)
        distribution = TorchConditionalActionDistribution(
            logits, wait, concentration, concentration
        )
        sampler = PolicySampler(31)
        first = sampler.sample(distribution)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_state))
        state = sampler.state_dict()
        expected = sampler.sample(distribution)
        restored = PolicySampler(0)
        restored.load_state_dict(state)
        actual = restored.sample(distribution)
        torch.testing.assert_close(actual.kind, expected.kind)
        torch.testing.assert_close(actual.wait_index, expected.wait_index)
        torch.testing.assert_close(actual.xy, expected.xy)
        self.assertFalse(torch.equal(first.xy, expected.xy))

    def test_failure_after_environment_mutation_poisons_collector(self) -> None:
        class BrokenTask(FixedBranchTask):
            def rewards(self, transitions):
                raise ValueError("broken reward")

        model = small_model()
        adapter = MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder())
        collector = RecurrentCollector(
            model,
            adapter,
            BrokenTask(4),
            config=CollectorConfig(max_decisions=1),
            policy_sampler_seed=1,
        )
        collector.initialize()
        with self.assertRaisesRegex(ValueError, "broken reward"):
            collector.collect()
        self.assertTrue(collector.poisoned)
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            collector.collect()

    def test_ppo_budget_is_checked_before_collecting_an_extra_rollout(self) -> None:
        model = small_model()
        env = FakeActiveVector()
        adapter = MacroVectorAdapter(env, encoder=TeacherStateEncoder())
        task = FixedBranchTask(4)
        collector = RecurrentCollector(
            model,
            adapter,
            task,
            config=CollectorConfig(max_decisions=1),
            policy_sampler_seed=2,
        )
        trainer = PPOTrainer(
            model,
            config=PPOConfig(epochs=1, lane_minibatch_size=4, target_kl=1.0),
            total_updates=1,
            sampler_seed=3,
        )
        session = R3ATrainingSession(collector, trainer, numpy_seed=4)
        session.initialize()
        session.run_update()
        ticks = tuple(env.ticks)
        with self.assertRaisesRegex(RuntimeError, "PPO update budget"):
            session.run_update()
        self.assertEqual(tuple(env.ticks), ticks)
        self.assertFalse(session.poisoned)

    def test_session_rejects_a_rollout_collected_outside_its_transaction(self) -> None:
        model = small_model()
        collector = make_collector(model, FakeActiveVector(), decisions=1)
        trainer = PPOTrainer(
            model,
            config=PPOConfig(epochs=1, lane_minibatch_size=4, target_kl=1.0),
            total_updates=1,
            sampler_seed=3,
        )
        session = R3ATrainingSession(collector, trainer, numpy_seed=4)
        session.initialize()
        collector.collect()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "unconsumed rollout"):
                session.save(directory, "orphan", identity={"test": "orphan"})
        with self.assertRaisesRegex(RuntimeError, "unconsumed rollout"):
            session.run_update()
        self.assertFalse(session.poisoned)

    def test_session_rejects_direct_adapter_mutation(self) -> None:
        model = small_model()
        collector = make_collector(model, FakeActiveVector(), decisions=1)
        trainer = PPOTrainer(
            model,
            config=PPOConfig(epochs=1, lane_minibatch_size=4, target_kl=1.0),
            total_updates=1,
            sampler_seed=3,
        )
        session = R3ATrainingSession(collector, trainer, numpy_seed=4)
        session.initialize()
        collector.adapter.reset()
        with self.assertRaisesRegex(RuntimeError, "unconsumed rollout"):
            session.run_update()

    def test_collector_checks_actual_adapter_event_capture(self) -> None:
        class EventTask(FixedBranchTask):
            capture_events = True

        with self.assertRaisesRegex(ValueError, "actual adapter"):
            RecurrentCollector(
                small_model(),
                MacroVectorAdapter(
                    FakeActiveVector(),
                    encoder=TeacherStateEncoder(),
                    capture_events=False,
                ),
                EventTask(4),
                config=CollectorConfig(max_decisions=1),
                policy_sampler_seed=1,
            )

    def test_reset_failure_poisons_collector(self) -> None:
        env = FakeActiveVector()

        def fail(*_args, **_kwargs):
            raise OSError("reset transport")

        env.reset = fail  # type: ignore[method-assign]
        collector = make_collector(small_model(), env, decisions=1)
        with self.assertRaisesRegex(RuntimeError, "vector coordinator is poisoned"):
            collector.initialize()
        self.assertTrue(collector.poisoned)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_collection_returns_a_device_local_training_batch(self) -> None:
        model = small_model().cuda()
        collector = make_collector(model, FakeActiveVector(), decisions=1)
        collector.initialize()
        batch = collector.collect().batch
        self.assertEqual(batch.global_features.device.type, "cuda")
        self.assertEqual(batch.initial_state.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
