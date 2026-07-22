from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import astuple, replace

import torch

from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.checkpoints import load_checkpoint, save_checkpoint
from irisu_rl.ppo import PPOConfig, PPOTrainer, RecurrentTrainingBatch
from irisu_rl.schema import TEACHER_V1
from irisu_rl.torch_distribution import TorchConditionalActionDistribution


class RecurrentModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(4)
        self.model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(16, 16, 24, 24, 1),
        )

    def observations(self, time=3, batch=2):
        global_features = torch.randn(time, batch, len(TEACHER_V1.global_features))
        bodies = torch.randn(
            time,
            batch,
            TEACHER_V1.capacity,
            len(TEACHER_V1.body_features),
        )
        mask = torch.zeros(time, batch, TEACHER_V1.capacity, dtype=torch.bool)
        mask[..., :3] = True
        return global_features, bodies, mask

    def test_body_permutation_and_padded_nan_are_inert(self) -> None:
        global_features, bodies, mask = self.observations(time=1)
        hidden = self.model.initial_state(2)
        baseline = self.model(global_features, bodies, mask, hidden)
        permutation = torch.arange(TEACHER_V1.capacity)
        permutation[:3] = torch.tensor([2, 0, 1])
        permuted = self.model(
            global_features, bodies[:, :, permutation], mask[:, :, permutation], hidden
        )
        torch.testing.assert_close(permuted.kind_logits, baseline.kind_logits)
        torch.testing.assert_close(permuted.values, baseline.values)
        poisoned = bodies.clone()
        poisoned[..., 3:, :] = float("nan")
        ignored = self.model(global_features, poisoned, mask, hidden)
        torch.testing.assert_close(ignored.kind_logits, baseline.kind_logits)
        torch.testing.assert_close(ignored.recurrent_state, baseline.recurrent_state)

    def test_full_sequence_matches_repeated_single_steps_and_reset_clears_history(
        self,
    ) -> None:
        global_features, bodies, mask = self.observations()
        initial = self.model.initial_state(2)
        full = self.model(global_features, bodies, mask, initial)
        hidden = initial
        logits = []
        for index in range(global_features.shape[0]):
            step = self.model(
                global_features[index : index + 1],
                bodies[index : index + 1],
                mask[index : index + 1],
                hidden,
            )
            logits.append(step.kind_logits)
            hidden = step.recurrent_state
        torch.testing.assert_close(torch.cat(logits), full.kind_logits)
        torch.testing.assert_close(hidden, full.recurrent_state)
        reset = torch.zeros((3, 2), dtype=torch.bool)
        reset[0] = True
        from_large = self.model(
            global_features,
            bodies,
            mask,
            torch.full_like(initial, 1e6),
            reset_before=reset,
        )
        from_zero = self.model(
            global_features, bodies, mask, initial, reset_before=reset
        )
        torch.testing.assert_close(from_large.kind_logits, from_zero.kind_logits)


class PPOTrainerTests(unittest.TestCase):
    def make_batch(self):
        torch.manual_seed(8)
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1),
        )
        time, lanes = 2, 2
        global_features = torch.randn(time, lanes, len(TEACHER_V1.global_features))
        bodies = torch.zeros(
            time,
            lanes,
            TEACHER_V1.capacity,
            len(TEACHER_V1.body_features),
        )
        body_mask = torch.zeros(time, lanes, TEACHER_V1.capacity, dtype=torch.bool)
        initial = model.initial_state(lanes)
        reset = torch.zeros((time, lanes), dtype=torch.bool)
        output = model(global_features, bodies, body_mask, initial)
        distribution = TorchConditionalActionDistribution(
            output.kind_logits,
            output.wait_logits,
            output.coordinate_alpha,
            output.coordinate_beta,
        )
        actions = distribution.deterministic()
        actions = type(actions)(
            actions.kind.detach(),
            actions.wait_index.detach(),
            actions.xy.detach(),
        )
        old_log_prob = distribution.log_prob(actions).detach()
        old_values = output.values.detach()
        valid = torch.ones((time, lanes), dtype=torch.bool)
        batch = RecurrentTrainingBatch(
            global_features,
            bodies,
            body_mask,
            reset,
            initial,
            actions,
            old_log_prob,
            old_values,
            torch.tensor([[1.0, -0.5], [0.25, 0.75]]),
            old_values + 1.0,
            valid,
            valid.clone(),
            torch.ones((time, lanes, 3), dtype=torch.bool),
            torch.ones((time, lanes, 100), dtype=torch.bool),
        )
        return model, batch

    def refresh_batch(self, model, batch):
        with torch.no_grad():
            output = model(
                batch.global_features,
                batch.body_features,
                batch.body_mask,
                batch.initial_state,
                reset_before=batch.reset_before,
            )
            distribution = TorchConditionalActionDistribution(
                output.kind_logits,
                output.wait_logits,
                output.coordinate_alpha,
                output.coordinate_beta,
                kind_mask=batch.kind_mask,
                wait_mask=batch.wait_mask,
            )
            actions = distribution.deterministic()
            return replace(
                batch,
                actions=actions,
                old_log_prob=distribution.log_prob(actions),
                old_values=output.values,
                returns=output.values + 1.0,
            )

    def test_update_is_finite_changes_parameters_and_reports_used_lr(self) -> None:
        model, batch = self.make_batch()
        trainer = PPOTrainer(
            model,
            config=PPOConfig(
                learning_rate=3e-4,
                epochs=1,
                lane_minibatch_size=2,
                entropy_coefficient=0.0,
                target_kl=1.0,
            ),
            total_updates=3,
            sampler_seed=11,
        )
        before = copy.deepcopy(model.state_dict())
        stats = trainer.update(batch)
        self.assertEqual(stats.learning_rate, 3e-4)
        self.assertTrue(
            all(
                torch.isfinite(torch.tensor(value))
                for value in astuple(stats)
                if isinstance(value, float)
            )
        )
        self.assertTrue(
            any(
                not torch.equal(before[name], value)
                for name, value in model.state_dict().items()
            )
        )
        self.assertAlmostEqual(trainer.schedule.learning_rate, 1.65e-4)

    def test_collection_policy_mismatch_fails_before_mutation(self) -> None:
        model, batch = self.make_batch()
        trainer = PPOTrainer(model, total_updates=2, sampler_seed=1)
        bad_log_prob = batch.old_log_prob.clone()
        bad_log_prob[0, 0] += 1
        bad = replace(batch, old_log_prob=bad_log_prob)
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, "likelihoods"):
            trainer.update(bad)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_trainer_state_resumes_next_update_bit_exactly(self) -> None:
        model, batch = self.make_batch()
        config = PPOConfig(
            learning_rate=1e-4,
            epochs=2,
            lane_minibatch_size=1,
            entropy_coefficient=0.0,
            target_kl=1.0,
        )
        trainer = PPOTrainer(model, config=config, total_updates=4, sampler_seed=29)
        trainer.update(batch)
        next_batch = self.refresh_batch(model, batch)
        model_state = copy.deepcopy(model.state_dict())
        trainer_state = copy.deepcopy(trainer.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            save_checkpoint(
                directory,
                "update-1",
                identity={"model": model.manifest()},
                state={"model": model_state, "trainer": trainer_state},
            )
            loaded, _, _ = load_checkpoint(
                directory, expected_identity={"model": model.manifest()}
            )
        expected_stats = trainer.update(next_batch)
        expected_model = copy.deepcopy(model.state_dict())

        restored_model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1),
        )
        restored = PPOTrainer(
            restored_model, config=config, total_updates=4, sampler_seed=999
        )
        restored_model.load_state_dict(loaded["model"], strict=True)
        restored.load_state_dict(loaded["trainer"])
        actual_stats = restored.update(next_batch)
        self.assertEqual(actual_stats, expected_stats)
        for name, value in restored_model.state_dict().items():
            self.assertTrue(torch.equal(value, expected_model[name]), name)


if __name__ == "__main__":
    unittest.main()
