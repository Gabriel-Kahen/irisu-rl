from __future__ import annotations

import copy
import unittest

import torch

from irisu_pointer.action import PointerActionTensor
from irisu_pointer.model import EntityPointerActorCritic, PointerModelConfig
from irisu_pointer.sequence import (
    PointerSequenceConfig,
    PointerSequenceEpisode,
    PointerSequenceTrainer,
    forward_pointer_sequence,
    pad_pointer_episodes,
    pointer_sequence_loss,
)
from irisu_rl.schema import TEACHER_V1


def _small_model() -> EntityPointerActorCritic:
    torch.manual_seed(71)
    return EntityPointerActorCritic(
        TEACHER_V1,
        config=PointerModelConfig(
            global_hidden=12,
            body_hidden=12,
            attention_hidden=24,
            attention_heads=4,
            attention_layers=1,
            feedforward_hidden=48,
            relation_hidden=12,
            recurrent_hidden=20,
        ),
    )


def _episode(
    identity: str, length: int, bodies: int = 3, signal: float = 0.0
) -> PointerSequenceEpisode:
    schema = TEACHER_V1
    global_features = torch.zeros(length, len(schema.global_features))
    global_features[0, 0] = signal
    body_features = torch.zeros(length, bodies, len(schema.body_features))
    body_mask = torch.ones(length, bodies, dtype=torch.bool)
    feature = {name: schema.body_features.index(name) for name in schema.body_features}
    for index in range(bodies):
        body_features[:, index, feature["kind_piece"]] = 1.0
        body_features[:, index, feature[f"color_{index % 3}"]] = 1.0
        body_features[:, index, feature["lifecycle_confirmed"]] = 1.0
        body_features[:, index, feature["effect_x_norm"]] = 0.2 + index * 0.25
        body_features[:, index, feature["effect_y_norm"]] = 0.4 + index * 0.1
        body_features[:, index, feature["width_norm"]] = 0.08
        body_features[:, index, feature["height_norm"]] = 0.10
    pattern = torch.tensor([0, 1, 2, 0], dtype=torch.long)[:length]
    target = torch.tensor([0, 1, min(2, bodies - 1), 0], dtype=torch.long)[:length]
    return PointerSequenceEpisode(
        identity,
        global_features,
        body_features,
        body_mask,
        PointerActionTensor(
            pattern,
            torch.tensor([1, 0, 0, 3], dtype=torch.long)[:length],
            target,
            torch.tensor([0, 4, 10, 0], dtype=torch.long)[:length],
        ),
        torch.linspace(-0.5, 1.0, length),
        schema,
    )


class PointerSequenceDataTests(unittest.TestCase):
    def test_padding_masks_resets_and_burn_in(self) -> None:
        batch = pad_pointer_episodes(
            (_episode("long", 4), _episode("short", 2, bodies=2))
        )
        self.assertEqual(batch.global_features.shape[:2], (4, 2))
        self.assertEqual(batch.body_features.shape[:3], (4, 2, 3))
        self.assertEqual(batch.valid_mask.sum(dim=0).tolist(), [4, 2])
        self.assertTrue(batch.reset_before[0].all())
        self.assertFalse(batch.reset_before[1:].any())
        self.assertFalse(batch.valid_mask[2:, 1].any())
        self.assertFalse(batch.body_mask[2:, 1].any())
        self.assertEqual(batch.training_mask(1).sum(dim=0).tolist(), [3, 1])

    def test_masked_sequence_target_fails_closed(self) -> None:
        episode = _episode("bad", 3)
        episode.body_mask[1, 1] = False
        with self.assertRaisesRegex(ValueError, "masked target"):
            PointerSequenceEpisode(
                episode.identity,
                episode.global_features,
                episode.body_features,
                episode.body_mask,
                episode.actions,
                episode.returns,
                episode.schema,
            )


class PointerSequenceTrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_chunked_hidden_carry_matches_full_and_initial_state_resets(self) -> None:
        model = _small_model().eval()
        batch = pad_pointer_episodes(
            (_episode("a", 4, signal=-1.0), _episode("b", 3, signal=1.0))
        )
        random_state = torch.randn_like(model.initial_state(2))
        with torch.no_grad():
            full = forward_pointer_sequence(
                model, batch, initial_state=random_state
            )
            chunked = forward_pointer_sequence(
                model,
                batch,
                initial_state=random_state,
                chunk_steps=2,
                detach_between_chunks=True,
            )
            zero = forward_pointer_sequence(model, batch)
        for left, right in (
            (full.kind_logits, chunked.kind_logits),
            (full.target_logits, chunked.target_logits),
            (full.value_quantiles, chunked.value_quantiles),
            (full.recurrent_state, chunked.recurrent_state),
            (full.kind_logits, zero.kind_logits),
            (full.recurrent_state, zero.recurrent_state),
        ):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)

    def test_branch_conditioned_loss_entropy_and_burn_in_mask(self) -> None:
        model = _small_model()
        batch = pad_pointer_episodes((_episode("a", 4), _episode("b", 3)))
        config = PointerSequenceConfig(burn_in_steps=1, entropy_coefficient=0.01)
        loss, metrics, state = pointer_sequence_loss(
            model, batch, config=config
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics.examples, 5)
        self.assertGreater(metrics.wait_examples, 0)
        self.assertGreater(metrics.actionable_examples, 0)
        self.assertGreater(metrics.entropy, 0.0)
        self.assertEqual(state.shape, model.initial_state(2).shape)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(value is not None for value in gradients))
        self.assertTrue(
            all(torch.isfinite(value).all() for value in gradients if value is not None)
        )

    def test_all_wait_sequence_has_finite_zero_branch_losses(self) -> None:
        episode = _episode("all-wait", 4)
        episode.actions.kind.zero_()
        batch = pad_pointer_episodes((episode,))
        loss, metrics, _ = pointer_sequence_loss(
            _small_model(),
            batch,
            config=PointerSequenceConfig(entropy_coefficient=0.0),
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics.actionable_examples, 0)
        self.assertEqual(metrics.target_loss, 0.0)
        self.assertEqual(metrics.template_loss, 0.0)
        self.assertGreaterEqual(metrics.predicted_actionable_rate, 0.0)

    def test_search_weights_do_not_change_balanced_kind_mass(self) -> None:
        ordinary = _episode("ordinary", 4)
        corrected = PointerSequenceEpisode(
            identity="corrected",
            global_features=ordinary.global_features.clone(),
            body_features=ordinary.body_features.clone(),
            body_mask=ordinary.body_mask.clone(),
            actions=PointerActionTensor(
                ordinary.actions.kind.clone(),
                ordinary.actions.wait_index.clone(),
                ordinary.actions.target_index.clone(),
                ordinary.actions.template_index.clone(),
            ),
            returns=ordinary.returns.clone(),
            schema=ordinary.schema,
            policy_weight=torch.tensor([1.0, 8.0, 8.0, 1.0]),
        )
        model = _small_model()
        plain_loss, plain_metrics, _ = pointer_sequence_loss(
            model,
            pad_pointer_episodes((ordinary,)),
            config=PointerSequenceConfig(entropy_coefficient=0.0),
        )
        weighted_loss, weighted_metrics, _ = pointer_sequence_loss(
            model,
            pad_pointer_episodes((corrected,)),
            config=PointerSequenceConfig(entropy_coefficient=0.0),
        )
        self.assertTrue(torch.isfinite(plain_loss))
        self.assertTrue(torch.isfinite(weighted_loss))
        self.assertAlmostEqual(
            plain_metrics.kind_loss, weighted_metrics.kind_loss, places=6
        )

    def test_tbptt_clipping_and_training_are_deterministic(self) -> None:
        first = _small_model()
        second = copy.deepcopy(first)
        batch = pad_pointer_episodes(
            (_episode("negative", 4, signal=-1.0), _episode("positive", 4, signal=1.0))
        )
        config = PointerSequenceConfig(
            learning_rate=1e-3,
            entropy_coefficient=0.0,
            burn_in_steps=1,
            tbptt_steps=2,
            max_gradient_norm=0.05,
            seed=101,
        )
        first_trainer = PointerSequenceTrainer(first, config=config)
        second_trainer = PointerSequenceTrainer(second, config=config)
        first_metrics = first_trainer.step(batch)
        second_metrics = second_trainer.step(batch)
        self.assertEqual(first_metrics.optimizer_steps, 2)
        self.assertGreater(first_metrics.gradient_norm, 0.0)
        self.assertLessEqual(
            first_metrics.clipped_gradient_norm, config.max_gradient_norm
        )
        self.assertEqual(first_metrics, second_metrics)
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name])


if __name__ == "__main__":
    unittest.main()
