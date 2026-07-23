from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest

import torch

from irisu_rl.actions import ActionSpec
from irisu_rl.collector import (
    CollectorConfig,
    CurriculumTaskContract,
    PolicySampler,
    R3ATrainingSession,
    RecurrentCollector,
    ScoreTaskContract,
)
from irisu_rl.curriculum import (
    CurriculumCoordinator,
    CurriculumSpec,
    SnapshotLibrary,
    ValidationResult,
)
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.models import PolicyValueOutput
from irisu_rl.ppo import PPOConfig, PPOTrainer
from irisu_rl.rewards import (
    LinearGaugePotential,
    RewardComposer,
)
from irisu_rl.schema import TEACHER_V1
from irisu_rl.torch_distribution import TorchConditionalActionDistribution
from irisu_rl.vector_adapter import MacroVectorAdapter
from tests.test_rl_vector_adapter import (
    FakeActiveVector,
    FakeTruncatingVector,
    observation,
)
from tests.test_r3a_curriculum import curriculum, validation_report


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


class TickValueModel(RecurrentActorCritic):
    """Deterministic critic: V(s) is the encoded simulator tick."""

    def forward(
        self,
        global_features,
        body_features,
        body_mask,
        recurrent_state,
        *,
        reset_before=None,
    ):
        time, batch, _ = global_features.shape
        device = global_features.device
        return PolicyValueOutput(
            torch.zeros((time, batch, 3), device=device),
            torch.zeros(
                (time, batch, len(self.action_spec.wait_choices)), device=device
            ),
            torch.full((time, batch, 2, 2), 2.0, device=device),
            torch.full((time, batch, 2, 2), 2.0, device=device),
            global_features[..., 0] * 100_000.0,
            recurrent_state,
        )


class AllHeldTruncatingVector(FakeActiveVector):
    def _step(self, indices, actions):
        output, rewards, terminated, truncated, infos = [], [], [], [], []
        self.calls.append(
            (tuple(indices), tuple(int(action.kind) for action in actions))
        )
        for lane, _action in zip(indices, actions):
            self.ticks[lane] += 1
            output.append(observation(self.ticks[lane], truncated=True))
            rewards.append(1)
            terminated.append(False)
            truncated.append(True)
            infos.append({"events": (), "invalid_action": False, "config_hash": 99})
        return output, rewards, terminated, truncated, infos

    def clone_state(self):
        return tuple(int(tick).to_bytes(8, "little") for tick in self.ticks)

    def state_hash(self):
        return tuple(self.ticks)

    def restore_state(self, snapshots):
        self.ticks = [int.from_bytes(snapshot, "little") for snapshot in snapshots]
        return tuple(observation(tick) for tick in self.ticks)


class BootstrapMatrixVector(FakeActiveVector):
    """Live, neutral truncation, terminal, and completed-shot truncation."""

    def _step(self, indices, actions):
        output, rewards, terminated, truncated, infos = [], [], [], [], []
        self.calls.append(
            (tuple(indices), tuple(int(action.kind) for action in actions))
        )
        release = tuple(indices) == (3,)
        for lane, action in zip(indices, actions):
            if release:
                delta, terminal, cut = 1, False, True
            elif lane == 1:
                delta, terminal, cut = 2, False, True
            elif lane == 2:
                delta, terminal, cut = 1, True, False
            else:
                delta = action.wait_ticks if lane == 0 else 1
                terminal, cut = False, False
            self.ticks[lane] += delta
            output.append(
                observation(self.ticks[lane], terminated=terminal, truncated=cut)
            )
            rewards.append(delta)
            terminated.append(terminal)
            truncated.append(cut)
            infos.append({"events": (), "invalid_action": False, "config_hash": 99})
        return output, rewards, terminated, truncated, infos


class StaggeredTerminalVector(FakeActiveVector):
    thresholds = (3, 2, 2, 1)

    def _step(self, indices, actions):
        output, rewards, terminated, truncated, infos = [], [], [], [], []
        self.calls.append(
            (tuple(indices), tuple(int(action.kind) for action in actions))
        )
        for lane, action in zip(indices, actions):
            delta = action.wait_ticks if int(action.kind) == 0 else 1
            self.ticks[lane] += delta
            terminal = self.ticks[lane] >= self.thresholds[lane]
            output.append(observation(self.ticks[lane], terminated=terminal))
            rewards.append(delta)
            terminated.append(terminal)
            truncated.append(False)
            infos.append({"events": (), "invalid_action": False, "config_hash": 99})
        return output, rewards, terminated, truncated, infos


class MixedGaugeResetVector(FakeActiveVector):
    """Gauge-changing lanes with lane zero ending on each semantic wait."""

    def _step(self, indices, actions):
        output, rewards, terminated, truncated, infos = super()._step(indices, actions)
        for offset, lane in enumerate(indices):
            output[offset].gauge = max(1, 100 - self.ticks[lane] * (lane + 1))
            if lane == 0:
                output[offset].terminated = True
                terminated[offset] = True
        return output, rewards, terminated, truncated, infos


class WeakOnlyTask(ScoreTaskContract):
    def action_masks(self, action_spec: ActionSpec):
        kind = torch.zeros((self.lanes, 3), dtype=torch.bool)
        kind[:, 1] = True
        wait = torch.zeros(
            (self.lanes, len(action_spec.wait_choices)), dtype=torch.bool
        )
        return kind, wait


class WaitOnlyTask(ScoreTaskContract):
    def action_masks(self, action_spec: ActionSpec):
        kind = torch.zeros((self.lanes, 3), dtype=torch.bool)
        kind[:, 0] = True
        wait = torch.ones((self.lanes, len(action_spec.wait_choices)), dtype=torch.bool)
        return kind, wait


class DeterministicWaitModel(RecurrentActorCritic):
    def __init__(self, *, prefer_long: bool) -> None:
        super().__init__(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1),
        )
        self.prefer_long = prefer_long

    def forward(
        self,
        global_features,
        body_features,
        body_mask,
        recurrent_state,
        *,
        reset_before=None,
    ):
        time, batch, _ = global_features.shape
        device = global_features.device
        direction = 1.0 if self.prefer_long else -1.0
        waits = (
            torch.arange(
                len(self.action_spec.wait_choices),
                dtype=torch.float32,
                device=device,
            )
            * direction
        ).expand(time, batch, -1)
        kind = torch.full((time, batch, 3), -100.0, device=device)
        kind[..., 0] = 100.0
        concentration = torch.full((time, batch, 2, 2), 2.0, device=device)
        return PolicyValueOutput(
            kind,
            waits,
            concentration,
            concentration,
            torch.zeros((time, batch), device=device),
            recurrent_state,
        )


class BudgetVector(FakeActiveVector):
    def _step(self, indices, actions):
        output, rewards, terminated, truncated, infos = [], [], [], [], []
        self.calls.append(
            (tuple(indices), tuple(int(action.kind) for action in actions))
        )
        for lane, action in zip(indices, actions):
            delta = action.wait_ticks if int(action.kind) == 0 else 1
            self.ticks[lane] += delta
            output.append(observation(self.ticks[lane]))
            rewards.append(delta)
            terminated.append(False)
            truncated.append(False)
            infos.append({"events": (), "invalid_action": False, "config_hash": 99})
        return output, rewards, terminated, truncated, infos


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
    def test_gauge_shaping_requires_conditioned_critic_and_supports_mixed_weights(
        self,
    ) -> None:
        base = curriculum()
        composer = RewardComposer(shaping_spec=LinearGaugePotential())
        coordinator = CurriculumCoordinator(base, 4, learner_seed=19)
        task = CurriculumTaskContract(coordinator, composer, capture_events=False)
        adapter = MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder())
        with self.assertRaisesRegex(ValueError, "critic-condition width"):
            RecurrentCollector(
                small_model(),
                adapter,
                task,
                policy_sampler_seed=7,
            )

        conditioned = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1, critic_condition_features=1),
        )
        RecurrentCollector(
            conditioned,
            adapter,
            task,
            policy_sampler_seed=7,
        )
        coordinator.lane_shaping_weight_ppm = [500_000, 0, 250_000, 100_000]
        torch.testing.assert_close(
            task.critic_condition(),
            torch.tensor([[0.5], [0.0], [0.25], [0.1]]),
            rtol=0,
            atol=0,
        )

    def test_reward_identity_mismatch_is_rejected_on_task_restore(self) -> None:
        base = curriculum()
        shaped = CurriculumTaskContract(
            CurriculumCoordinator(base, 4, learner_seed=19),
            RewardComposer(shaping_spec=LinearGaugePotential()),
            capture_events=False,
        )
        score_only = CurriculumTaskContract(
            CurriculumCoordinator(base, 4, learner_seed=19),
            RewardComposer(),
            capture_events=False,
        )
        with self.assertRaisesRegex(ValueError, "reward identity mismatch"):
            score_only.load_state_dict(shaped.state_dict())

    def test_collector_binds_critic_condition_to_composed_reward_weight(self) -> None:
        base = curriculum()
        library = SnapshotLibrary(
            tuple(replace(recipe, config_hash=99) for recipe in base.library.recipes)
        )
        spec = replace(base, curriculum_id="condition-bind-v1", library=library)

        class IncorrectConditionTask(CurriculumTaskContract):
            def critic_condition(self):
                return torch.zeros((self.coordinator.lanes, 1), dtype=torch.float32)

        task = IncorrectConditionTask(
            CurriculumCoordinator(spec, 4, learner_seed=19),
            RewardComposer(shaping_spec=LinearGaugePotential()),
            capture_events=False,
        )
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1, critic_condition_features=1),
        )
        collector = RecurrentCollector(
            model,
            MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder()),
            task,
            config=CollectorConfig(max_decisions=1),
            policy_sampler_seed=7,
        )
        collector.initialize()
        with self.assertRaisesRegex(ValueError, "composed reward weight"):
            collector.collect()

    def test_collector_rejects_condition_change_across_rollout_boundary(self) -> None:
        base = curriculum()
        library = SnapshotLibrary(
            tuple(replace(recipe, config_hash=99) for recipe in base.library.recipes)
        )
        spec = replace(base, curriculum_id="condition-continuity-v1", library=library)
        coordinator = CurriculumCoordinator(spec, 4, learner_seed=19)
        task = CurriculumTaskContract(
            coordinator,
            RewardComposer(shaping_spec=LinearGaugePotential()),
            capture_events=False,
        )
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1, critic_condition_features=1),
        )
        collector = RecurrentCollector(
            model,
            MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder()),
            task,
            config=CollectorConfig(max_decisions=1),
            policy_sampler_seed=7,
        )
        collector.initialize()
        collector.collect()
        coordinator.lane_shaping_weight_ppm = [250_000] * 4
        with self.assertRaisesRegex(ValueError, "continuing episode"):
            collector.collect()

    def test_mixed_coefficients_collect_reset_and_update_end_to_end(self) -> None:
        base = curriculum()
        library = SnapshotLibrary(
            tuple(replace(recipe, config_hash=99) for recipe in base.library.recipes)
        )
        spec = replace(base, curriculum_id="mixed-alpha-update-v1", library=library)
        coordinator = CurriculumCoordinator(spec, 4, learner_seed=19)
        coordinator.lane_shaping_weight_ppm = [250_000, 0, 500_000, 100_000]
        task = CurriculumTaskContract(
            coordinator,
            RewardComposer(shaping_spec=LinearGaugePotential()),
            capture_events=False,
        )
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1, critic_condition_features=1),
        )
        collector = RecurrentCollector(
            model,
            MacroVectorAdapter(MixedGaugeResetVector(), encoder=TeacherStateEncoder()),
            task,
            config=CollectorConfig(max_decisions=2),
            policy_sampler_seed=7,
        )
        collector.initialize()
        rollout = collector.collect()
        expected = torch.tensor(
            [
                [[0.25], [0.0], [0.5], [0.1]],
                [[0.5], [0.0], [0.5], [0.1]],
            ]
        )
        torch.testing.assert_close(
            rollout.batch.critic_condition, expected, rtol=0, atol=0
        )
        self.assertTrue(rollout.batch.reset_before[1, 0])
        self.assertEqual(
            rollout.audit.decisions[0].shaping_weight_ppm,
            (250_000, 0, 500_000, 100_000),
        )
        self.assertTrue(
            any(
                value != 0
                for decision in rollout.audit.decisions
                for value in decision.shaping_rewards
            )
        )
        trainer = PPOTrainer(
            model,
            config=PPOConfig(epochs=1, lane_minibatch_size=4, target_kl=1.0),
            total_updates=1,
            sampler_seed=11,
        )
        self.assertGreater(trainer.update(rollout.batch).optimizer_steps, 0)

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
        self.assertEqual(rollout.audit.decisions[0].start_gauges, (100,) * 4)
        self.assertEqual(rollout.audit.decisions[0].end_gauges, (100,) * 4)
        self.assertEqual(rollout.audit.decisions[0].gauge_maxes, (1000,) * 4)
        self.assertEqual(len(rollout.audit.reward_sha256), 64)
        self.assertEqual(rollout.audit.shaping_id, "none")
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
        collector = make_collector(
            TickValueModel(TEACHER_V1, config=RecurrentModelConfig(8, 8, 12, 12, 1)),
            FakeTruncatingVector(),
            decisions=1,
        )
        collector.initialize()
        batch = collector.collect().batch
        self.assertTrue(batch.train_mask[0, 1])
        self.assertFalse(batch.train_mask[0, 2])
        # V(s) equals tick. These exact returns prove live lanes bootstrap from
        # tick 1 and the neutral truncation from retained tick 2, not autoreset
        # tick 0. The held-shot truncation is censored.
        torch.testing.assert_close(batch.returns[0], torch.tensor([2.0, 4.0, 0.0, 4.0]))
        self.assertEqual(float(batch.advantages[0, 2]), 0.0)
        self.assertEqual(float(batch.returns[0, 2]), 0.0)

    def test_all_censored_rollout_is_a_clean_skipped_update(self) -> None:
        model = small_model()
        collector = RecurrentCollector(
            model,
            MacroVectorAdapter(
                AllHeldTruncatingVector(), encoder=TeacherStateEncoder()
            ),
            WeakOnlyTask(4),
            config=CollectorConfig(max_decisions=1),
            policy_sampler_seed=2,
        )
        trainer = PPOTrainer(
            model,
            config=PPOConfig(epochs=1, lane_minibatch_size=4, target_kl=1.0),
            total_updates=1,
            sampler_seed=3,
        )
        session = R3ATrainingSession(
            collector, trainer, numpy_seed=4, max_consecutive_skips=2
        )
        session.initialize()
        result = session.run_update()
        self.assertIsNone(result.optimizer)
        self.assertIn("no trainable", result.skipped_reason or "")
        self.assertEqual(collector.completed_updates, 0)
        self.assertEqual(trainer.schedule.completed_updates, 0)
        self.assertFalse(session.poisoned)
        self.assertIsNone(session.run_update().optimizer)
        with self.assertRaisesRegex(RuntimeError, "safety limit"):
            session.run_update()

    def test_skipped_rollout_boundary_resumes_exactly(self) -> None:
        def build(seed):
            torch.manual_seed(seed)
            model = small_model()
            collector = RecurrentCollector(
                model,
                MacroVectorAdapter(
                    AllHeldTruncatingVector(), encoder=TeacherStateEncoder()
                ),
                WeakOnlyTask(4),
                config=CollectorConfig(max_decisions=1),
                policy_sampler_seed=2,
            )
            trainer = PPOTrainer(
                model,
                config=PPOConfig(epochs=1, lane_minibatch_size=4, target_kl=1.0),
                total_updates=1,
                sampler_seed=3,
            )
            return R3ATrainingSession(
                collector, trainer, numpy_seed=4, max_consecutive_skips=4
            )

        with tempfile.TemporaryDirectory() as directory:
            source = build(101)
            source.initialize()
            source.run_update()
            source.save(directory, "skip", identity={"test": "skip-resume"})
            expected = source.run_update()

            restored = build(999)
            restored.restore(directory, identity={"test": "skip-resume"})
            actual = restored.run_update()
            self.assertEqual(actual, expected)
            self.assertEqual(restored.attempted_rollouts, 2)
            self.assertEqual(restored.skipped_rollouts, 2)
            self.assertEqual(restored.consecutive_skips, 2)

    def test_bootstrap_matrix_covers_terminal_and_completed_shot_cut(self) -> None:
        collector = make_collector(
            TickValueModel(TEACHER_V1, config=RecurrentModelConfig(8, 8, 12, 12, 1)),
            BootstrapMatrixVector(),
            decisions=1,
        )
        collector.initialize()
        rollout = collector.collect()
        decision = rollout.audit.decisions[0]
        self.assertEqual(decision.bootstrap_mask, (True, True, False, True))
        self.assertEqual(decision.trace_mask, (True, False, False, False))
        self.assertEqual(rollout.batch.train_mask[0].tolist(), [True] * 4)
        torch.testing.assert_close(
            rollout.batch.returns[0], torch.tensor([2.0, 4.0, 1.0, 4.0])
        )

    def test_activation_rollouts_are_drain_only_until_every_lane_resets(self) -> None:
        base = curriculum()
        library = SnapshotLibrary(
            tuple(replace(recipe, config_hash=99) for recipe in base.library.recipes)
        )
        spec = CurriculumSpec(
            "activation-session-v1",
            library,
            (replace(base.stages[0], enabled_wait_ticks=(1,)), base.stages[1]),
            base.evaluation_seed,
            base.prior_stage_mix_ppm,
        )
        coordinator = CurriculumCoordinator(spec, 4, learner_seed=19)
        for policy in ("a" * 64, "b" * 64):
            coordinator.record_validation(
                validation_report(
                    coordinator, policy, (ValidationResult("wait", 8, 10),)
                )
            )
        self.assertEqual(coordinator.phase, "activation")
        model = small_model()
        task = CurriculumTaskContract(
            coordinator,
            RewardComposer(
                shaping_id="zero-v1",
                shaping=lambda transitions: torch.zeros(
                    len(transitions), dtype=torch.float32
                ),
            ),
            capture_events=False,
        )
        collector = RecurrentCollector(
            model,
            MacroVectorAdapter(
                StaggeredTerminalVector(), encoder=TeacherStateEncoder()
            ),
            task,
            config=CollectorConfig(max_decisions=1),
            policy_sampler_seed=2,
        )
        trainer = PPOTrainer(
            model,
            config=PPOConfig(epochs=1, lane_minibatch_size=4, target_kl=1.0),
            total_updates=2,
            sampler_seed=3,
        )
        session = R3ATrainingSession(
            collector, trainer, numpy_seed=4, max_consecutive_skips=3
        )
        session.initialize()
        before = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
        for _ in range(3):
            result = session.run_update()
            self.assertIsNone(result.optimizer)
            self.assertIn("activation drain", result.skipped_reason or "")
            self.assertEqual(trainer.schedule.completed_updates, 0)
            self.assertEqual(collector.completed_updates, 0)
            self.assertEqual(task.completed_updates, 0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertEqual(coordinator.phase, "normal")
        trained = session.run_update()
        self.assertIsNotNone(trained.optimizer)
        self.assertEqual(trainer.schedule.completed_updates, 1)

    def test_tick_budget_rejects_a_task_without_a_bounded_action(self) -> None:
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
        with self.assertRaisesRegex(ValueError, "no legal action"):
            collector.collect()

    def test_tick_budget_masks_long_waits_and_reaches_target_at_capacity(self) -> None:
        for prefer_long in (False, True):
            collector = RecurrentCollector(
                DeterministicWaitModel(prefer_long=prefer_long),
                MacroVectorAdapter(BudgetVector(), encoder=TeacherStateEncoder()),
                WaitOnlyTask(4),
                config=CollectorConfig(
                    max_decisions=16,
                    target_simulated_ticks=64,
                    lambda_tick=0.9,
                ),
                policy_sampler_seed=5,
            )
            collector.initialize()
            result = collector.collect()
            self.assertGreaterEqual(result.audit.simulated_ticks, 64)
            self.assertLess(result.audit.simulated_ticks - 64, 4)
            self.assertLessEqual(result.audit.decision_rows, 16)

    def test_tick_budget_handles_remaining_less_than_lane_count(self) -> None:
        collector = RecurrentCollector(
            DeterministicWaitModel(prefer_long=True),
            MacroVectorAdapter(BudgetVector(), encoder=TeacherStateEncoder()),
            WaitOnlyTask(4),
            config=CollectorConfig(
                max_decisions=18,
                target_simulated_ticks=70,
                lambda_tick=0.9,
            ),
            policy_sampler_seed=5,
        )
        collector.initialize()
        result = collector.collect()
        self.assertEqual(result.audit.simulated_ticks, 71)
        self.assertEqual(result.audit.tick_target_overshoot, 1)
        self.assertEqual(
            result.audit.decisions[-1].elapsed_ticks,
            (1, 1, 1, 1),
        )

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

    def test_restore_rejects_wrong_recurrent_dtype(self) -> None:
        source = make_collector(small_model(), FakeActiveVector(), decisions=1)
        source.initialize()
        state = source.state_dict()
        state["recurrent_state"] = state["recurrent_state"].to(torch.float64)
        restored = make_collector(small_model(), FakeActiveVector(), decisions=1)
        with self.assertRaisesRegex(ValueError, "recurrent/reset state"):
            restored.load_state_dict(state)

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
