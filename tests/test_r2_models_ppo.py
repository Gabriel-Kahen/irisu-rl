from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import astuple, replace
from unittest.mock import patch

import torch
from irisu_rl.checkpoints import load_checkpoint, save_checkpoint
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.ppo import (
    PPOConfig,
    PPOTrainer,
    RecurrentTrainingBatch,
    clipped_surrogate_loss,
    quantile_huber_loss,
)
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

    def test_masked_body_prefix_preserves_inference_outputs(self) -> None:
        global_features, bodies, mask = self.observations(time=1)
        hidden = self.model.initial_state(2)
        full = self.model(global_features, bodies, mask, hidden)
        prefix = self.model(
            global_features,
            bodies[..., :3, :].contiguous(),
            mask[..., :3].contiguous(),
            hidden,
        )
        for name in (
            "kind_logits",
            "wait_logits",
            "coordinate_alpha",
            "coordinate_beta",
            "values",
            "recurrent_state",
        ):
            torch.testing.assert_close(
                getattr(prefix, name),
                getattr(full, name),
                rtol=1e-5,
                atol=1e-6,
            )
        self.assertTrue(
            torch.equal(prefix.kind_logits.argmax(-1), full.kind_logits.argmax(-1))
        )
        self.assertTrue(
            torch.equal(prefix.wait_logits.argmax(-1), full.wait_logits.argmax(-1))
        )
        with self.assertRaisesRegex(ValueError, "model schema"):
            self.model(
                global_features,
                bodies[..., :0, :],
                mask[..., :0],
                hidden,
            )

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

    def test_default_model_preserves_legacy_architecture_and_manifest(self) -> None:
        config = RecurrentModelConfig()
        self.assertEqual(
            config.manifest(),
            {
                "global_hidden": 96,
                "body_hidden": 96,
                "fused_hidden": 192,
                "recurrent_hidden": 192,
                "recurrent_layers": 1,
                "minimum_concentration": 1.001,
            },
        )
        model = RecurrentActorCritic(TEACHER_V1)
        self.assertEqual(model.manifest()["architecture"], "recurrent-actor-critic-v1")
        self.assertIsNone(model.value_quantile_head)
        self.assertFalse(
            any(name.startswith("value_quantile_head.") for name in model.state_dict())
        )
        global_features, bodies, mask = self.observations(time=1)
        output = model(global_features, bodies, mask, model.initial_state(2))
        self.assertIsNone(output.value_quantiles)

    def test_quantile_count_validation_and_enabled_manifest(self) -> None:
        for count in (-1, 1, 2, 4, 102, True, 3.0):
            with (
                self.subTest(count=count),
                self.assertRaisesRegex(ValueError, "quantile count"),
            ):
                RecurrentModelConfig(value_quantile_count=count)  # type: ignore[arg-type]
        config = RecurrentModelConfig(value_quantile_count=51)
        self.assertEqual(config.manifest()["value_quantile_count"], 51)

    def test_enabled_model_emits_conditioned_critic_only_quantiles(self) -> None:
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(
                16,
                16,
                24,
                24,
                1,
                critic_condition_features=1,
                value_quantile_count=5,
            ),
        )
        global_features, bodies, mask = self.observations()
        hidden = model.initial_state(2)
        zero = model(
            global_features,
            bodies,
            mask,
            hidden,
            critic_condition=torch.zeros((3, 2, 1)),
        )
        one = model(
            global_features,
            bodies,
            mask,
            hidden,
            critic_condition=torch.ones((3, 2, 1)),
        )
        self.assertIsNotNone(zero.value_quantiles)
        self.assertEqual(zero.value_quantiles.shape, (3, 2, 5))
        self.assertTrue(torch.isfinite(zero.value_quantiles).all())
        for name in (
            "kind_logits",
            "wait_logits",
            "coordinate_alpha",
            "coordinate_beta",
            "recurrent_state",
        ):
            self.assertTrue(torch.equal(getattr(zero, name), getattr(one, name)), name)
        value_shift = one.values - zero.values
        self.assertTrue(torch.any(value_shift != 0))
        torch.testing.assert_close(
            one.value_quantiles - zero.value_quantiles,
            value_shift.unsqueeze(-1).expand_as(one.value_quantiles),
        )

    def test_mixed_lane_resets_use_maximal_reset_free_gru_spans(self) -> None:
        global_features, bodies, mask = self.observations(time=6)
        initial = self.model.initial_state(2)
        reset = torch.zeros((6, 2), dtype=torch.bool)
        reset[0, 0] = True
        reset[2, 1] = True
        reset[5, 0] = True
        hidden = initial
        repeated = []
        for index in range(6):
            step = self.model(
                global_features[index : index + 1],
                bodies[index : index + 1],
                mask[index : index + 1],
                hidden,
                reset_before=reset[index : index + 1],
            )
            repeated.append(step)
            hidden = step.recurrent_state
        calls = 0

        def count_forward(*_: object) -> None:
            nonlocal calls
            calls += 1

        handle = self.model.recurrent.register_forward_hook(count_forward)
        try:
            segmented = self.model(
                global_features,
                bodies,
                mask,
                initial,
                reset_before=reset,
            )
        finally:
            handle.remove()
        self.assertEqual(calls, 3)
        for name in (
            "kind_logits",
            "wait_logits",
            "coordinate_alpha",
            "coordinate_beta",
            "values",
        ):
            torch.testing.assert_close(
                getattr(segmented, name),
                torch.cat([getattr(step, name) for step in repeated]),
                rtol=0,
                atol=2e-7,
                msg=name,
            )
        torch.testing.assert_close(
            segmented.recurrent_state, hidden, rtol=0, atol=2e-7
        )

    def test_segmented_gru_gradient_difference_is_bounded(self) -> None:
        global_features, bodies, mask = self.observations(time=6)
        reset = torch.zeros((6, 2), dtype=torch.bool)
        reset[0, 0] = True
        reset[2, 1] = True
        reset[5, 0] = True
        segmented_model = copy.deepcopy(self.model)
        repeated_model = copy.deepcopy(self.model)

        segmented = segmented_model(
            global_features,
            bodies,
            mask,
            segmented_model.initial_state(2),
            reset_before=reset,
        )
        output_names = (
            "kind_logits",
            "wait_logits",
            "coordinate_alpha",
            "coordinate_beta",
            "values",
        )
        segmented_loss = sum(
            getattr(segmented, name).mean()
            for name in output_names
        )
        segmented_loss.backward()

        hidden = repeated_model.initial_state(2)
        repeated_loss = torch.zeros(())
        for index in range(6):
            step = repeated_model(
                global_features[index : index + 1],
                bodies[index : index + 1],
                mask[index : index + 1],
                hidden,
                reset_before=reset[index : index + 1],
            )
            hidden = step.recurrent_state
            repeated_loss = repeated_loss + sum(
                getattr(step, name).mean() / 6
                for name in output_names
            )
        repeated_loss.backward()
        maximum_difference = max(
            float((left.grad - right.grad).abs().max())
            for left, right in zip(
                segmented_model.parameters(), repeated_model.parameters()
            )
        )
        self.assertLessEqual(maximum_difference, 2e-7)


class PPOTrainerTests(unittest.TestCase):
    def assert_nested_equal(self, left, right, path="state"):
        self.assertIs(type(left), type(right), path)
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right), path)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys(), path)
            for key in left:
                self.assert_nested_equal(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), path)
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                self.assert_nested_equal(left_item, right_item, f"{path}[{index}]")
        else:
            self.assertEqual(left, right, path)

    def make_batch(self, config=None):
        torch.manual_seed(8)
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=config or RecurrentModelConfig(8, 8, 12, 12, 1),
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
        old_components = distribution.log_prob_components(actions)
        old_log_prob = old_components.total.detach()
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
            old_components.kind.detach(),
            old_components.wait.detach(),
            old_components.coordinates.detach(),
            old_values,
            torch.tensor([[1.0, -0.5], [0.25, 0.75]]),
            old_values + 1.0,
            valid,
            valid.clone(),
            torch.ones((time, lanes, 3), dtype=torch.bool),
            torch.ones((time, lanes, 100), dtype=torch.bool),
            torch.zeros((time, lanes, 0), dtype=torch.float32),
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
            components = distribution.log_prob_components(actions)
            return replace(
                batch,
                actions=actions,
                old_log_prob=components.total,
                old_kind_log_prob=components.kind,
                old_wait_log_prob=components.wait,
                old_coordinate_log_prob=components.coordinates,
                old_values=output.values,
                returns=output.values + 1.0,
            )

    def test_clipped_surrogate_matches_hand_calculated_sign_cases(self) -> None:
        ratio = torch.tensor([[1.5, 0.5, 1.1, 9.0, 0.1]])
        advantages = torch.tensor([[1.0, -1.0, 1.0, 100.0, 0.0]])
        mask = torch.tensor([[True, True, True, False, True]])
        loss = clipped_surrogate_loss(ratio, advantages, mask, 0.2)
        # Objectives: 1.2 (positive clipped high), -0.8 (negative clipped
        # low), 1.1 (inside clip), and 0.0 (zero advantage).
        self.assertAlmostEqual(float(loss), -(1.2 - 0.8 + 1.1) / 4)

    def test_quantile_huber_loss_zero_large_and_masked_cases(self) -> None:
        audited = quantile_huber_loss(
            torch.tensor([[[-2.0, -1.0, 3.0]]]),
            torch.tensor([[0.0]]),
            torch.tensor([[True]]),
            kappa=2.0,
        )
        self.assertAlmostEqual(float(audited), 5 / 24)

        targets = torch.tensor([[2.0, -3.0], [1.0, -1.0]])
        predictions = targets.unsqueeze(-1).expand(2, 2, 5).clone()
        mask = torch.tensor([[True, True], [True, False]])
        self.assertEqual(float(quantile_huber_loss(predictions, targets, mask)), 0.0)

        predictions[1, 1] = torch.tensor([-1e12, -1e6, 0.0, 1e6, 1e12])
        unchanged = quantile_huber_loss(predictions, targets, mask)
        predictions[1, 1] *= -1
        torch.testing.assert_close(
            quantile_huber_loss(predictions, targets, mask), unchanged
        )
        large = quantile_huber_loss(
            torch.zeros((1, 2, 5)),
            torch.tensor([[1e30, -1e30]]),
            torch.ones((1, 2), dtype=torch.bool),
        )
        self.assertTrue(torch.isfinite(large))

    def test_quantile_huber_loss_rejects_malformed_inputs(self) -> None:
        predictions = torch.zeros((1, 2, 5))
        targets = torch.zeros((1, 2))
        mask = torch.ones((1, 2), dtype=torch.bool)
        bad_cases = (
            (predictions[0], targets[0], mask[0], 1.0),
            (predictions[..., :4], targets, mask, 1.0),
            (predictions, targets, mask.float(), 1.0),
            (predictions, targets, torch.zeros_like(mask), 1.0),
            (predictions, targets, mask, 0.0),
            (predictions, targets, mask, True),
        )
        for arguments in bad_cases:
            with (
                self.subTest(shapes=[value.shape for value in arguments[:3]]),
                self.assertRaises(ValueError),
            ):
                quantile_huber_loss(*arguments)
        nonfinite = predictions.clone()
        nonfinite[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid"):
            quantile_huber_loss(nonfinite, targets, mask)
        with self.assertRaisesRegex(FloatingPointError, "reduction"):
            quantile_huber_loss(
                torch.zeros((1, 16, 5)),
                torch.full((1, 16), 5e37),
                torch.ones((1, 16), dtype=torch.bool),
            )

    def test_quantile_head_and_loss_must_be_enabled_together(self) -> None:
        head_model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1, value_quantile_count=5),
        )
        head_before = copy.deepcopy(head_model.state_dict())
        with patch("irisu_rl.ppo.torch.optim.Adam") as adam:
            with self.assertRaisesRegex(ValueError, "enabled together"):
                PPOTrainer(head_model, total_updates=1, sampler_seed=1)
            adam.assert_not_called()
        self.assert_nested_equal(head_model.state_dict(), head_before, "head_model")

        scalar_model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1),
        )
        scalar_before = copy.deepcopy(scalar_model.state_dict())
        with patch("irisu_rl.ppo.torch.optim.Adam") as adam:
            with self.assertRaisesRegex(ValueError, "enabled together"):
                PPOTrainer(
                    scalar_model,
                    config=PPOConfig(quantile_value_coefficient=0.5),
                    total_updates=1,
                    sampler_seed=1,
                )
            adam.assert_not_called()
        self.assert_nested_equal(
            scalar_model.state_dict(), scalar_before, "scalar_model"
        )

    def test_quantile_update_changes_head_and_reports_loss(self) -> None:
        model, batch = self.make_batch(
            RecurrentModelConfig(8, 8, 12, 12, 1, value_quantile_count=5)
        )
        trainer = PPOTrainer(
            model,
            config=PPOConfig(
                epochs=1,
                lane_minibatch_size=2,
                entropy_coefficient=0.0,
                target_kl=1.0,
                quantile_value_coefficient=0.5,
            ),
            total_updates=1,
            sampler_seed=11,
        )
        before = model.value_quantile_head.weight.detach().clone()
        stats = trainer.update(batch)
        self.assertGreater(stats.quantile_value_loss, 0)
        self.assertFalse(torch.equal(model.value_quantile_head.weight, before))

    def test_malformed_quantile_output_fails_before_trainer_mutation(self) -> None:
        model, batch = self.make_batch(
            RecurrentModelConfig(8, 8, 12, 12, 1, value_quantile_count=5)
        )
        trainer = PPOTrainer(
            model,
            config=PPOConfig(quantile_value_coefficient=0.5),
            total_updates=2,
            sampler_seed=11,
        )
        with torch.no_grad():
            malformed = replace(
                model(
                    batch.global_features,
                    batch.body_features,
                    batch.body_mask,
                    batch.initial_state,
                    reset_before=batch.reset_before,
                ),
                value_quantiles=torch.zeros((*batch.returns.shape, 3)),
            )
        before_model = copy.deepcopy(model.state_dict())
        before_trainer = copy.deepcopy(trainer.state_dict())
        with (
            patch.object(model, "forward", return_value=malformed),
            self.assertRaisesRegex(ValueError, "configured shape"),
        ):
            trainer.update(batch)
        self.assert_nested_equal(model.state_dict(), before_model, "model")
        self.assert_nested_equal(trainer.state_dict(), before_trainer, "trainer")

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

    def test_update_avoids_redundant_full_batch_validation_forward(self) -> None:
        model, batch = self.make_batch()
        trainer = PPOTrainer(
            model,
            config=PPOConfig(
                epochs=1,
                lane_minibatch_size=2,
                entropy_coefficient=0.0,
                target_kl=1.0,
            ),
            total_updates=1,
            sampler_seed=11,
        )
        forward_calls = 0

        def count_forward(*_: object) -> None:
            nonlocal forward_calls
            forward_calls += 1

        handle = model.register_forward_hook(count_forward)
        try:
            trainer.update(batch)
        finally:
            handle.remove()
        # One full-batch policy verification plus one optimizer minibatch.
        self.assertEqual(forward_calls, 2)

    def test_batch_validation_checks_schema_and_recurrent_state_without_forward(
        self,
    ) -> None:
        model, batch = self.make_batch()
        with self.assertRaisesRegex(ValueError, "model schema"):
            replace(batch, body_features=batch.body_features[..., :-1, :]).validate(
                model
            )
        with self.assertRaisesRegex(ValueError, "recurrent state"):
            replace(batch, initial_state=batch.initial_state[..., :-1]).validate(model)

    def test_collection_policy_mismatch_fails_before_mutation(self) -> None:
        model, batch = self.make_batch()
        trainer = PPOTrainer(model, total_updates=2, sampler_seed=1)
        bad_log_prob = batch.old_log_prob.clone()
        bad_log_prob[0, 0] += 1
        bad = replace(batch, old_log_prob=bad_log_prob)
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, "likelihood"):
            trainer.update(bad)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_exhausted_update_budget_fails_before_any_mutation(self) -> None:
        model, batch = self.make_batch()
        trainer = PPOTrainer(
            model,
            config=PPOConfig(
                epochs=1,
                lane_minibatch_size=2,
                entropy_coefficient=0.0,
                target_kl=1.0,
            ),
            total_updates=1,
            sampler_seed=7,
        )
        trainer.update(batch)
        before_model = copy.deepcopy(model.state_dict())
        before_trainer = copy.deepcopy(trainer.state_dict())
        with self.assertRaisesRegex(RuntimeError, "budget is exhausted"):
            trainer.update(self.refresh_batch(model, batch))
        self.assert_nested_equal(model.state_dict(), before_model, "model")
        self.assert_nested_equal(trainer.state_dict(), before_trainer, "trainer")

    def assert_trainer_resume_is_bit_exact(
        self, model_config: RecurrentModelConfig, config: PPOConfig
    ) -> None:
        model, batch = self.make_batch(model_config)
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
            config=model_config,
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
        self.assert_nested_equal(restored.state_dict(), trainer.state_dict(), "trainer")

    def test_legacy_scalar_trainer_state_resumes_next_update_bit_exactly(self) -> None:
        self.assert_trainer_resume_is_bit_exact(
            RecurrentModelConfig(8, 8, 12, 12, 1),
            PPOConfig(
                learning_rate=1e-4,
                epochs=2,
                lane_minibatch_size=1,
                entropy_coefficient=0.0,
                target_kl=1.0,
            ),
        )

    def test_distributional_trainer_state_resumes_next_update_bit_exactly(
        self,
    ) -> None:
        self.assert_trainer_resume_is_bit_exact(
            RecurrentModelConfig(8, 8, 12, 12, 1, value_quantile_count=5),
            PPOConfig(
                learning_rate=1e-4,
                epochs=2,
                lane_minibatch_size=1,
                entropy_coefficient=0.0,
                target_kl=1.0,
                quantile_value_coefficient=0.5,
            ),
        )


if __name__ == "__main__":
    unittest.main()
