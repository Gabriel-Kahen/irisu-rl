from __future__ import annotations

import unittest

import numpy as np
import torch

from irisu_pointer.action import PointerActionSpec, PointerActionTensor
from irisu_pointer.dataset import PointerDataset
from irisu_pointer.distill import PointerBCConfig, PointerBCTrainer
from irisu_pointer.model import (
    EntityPointerActorCritic,
    PointerModelConfig,
)
from irisu_rl.encoding import EncodedBatch
from irisu_rl.schema import TEACHER_V1


def _same_color_batch(examples: int = 12) -> tuple[
    EncodedBatch, PointerActionTensor, tuple[str, ...], torch.Tensor
]:
    schema = TEACHER_V1
    global_features = np.zeros(
        (examples, len(schema.global_features)), dtype=np.float32
    )
    body_features = np.zeros(
        (examples, schema.capacity, len(schema.body_features)), dtype=np.float32
    )
    body_mask = np.zeros((examples, schema.capacity), dtype=np.bool_)
    tick = np.arange(examples, dtype=np.uint64)
    health = np.zeros(examples, dtype=np.uint32)
    index = {name: schema.body_features.index(name) for name in schema.body_features}
    gauge_index = schema.global_features.index("gauge_fraction")
    kinds: list[int] = []
    waits: list[int] = []
    targets: list[int] = []
    templates: list[int] = []
    values: list[float] = []
    for row in range(examples):
        body_mask[row, :3] = True
        for body in range(3):
            body_features[row, body, index["kind_piece"]] = 1.0
            body_features[row, body, index["shape_box"]] = 1.0
            body_features[row, body, index["lifecycle_falling"]] = 1.0
            body_features[row, body, index["width_norm"]] = 0.08
            body_features[row, body, index["height_norm"]] = 0.10
        # Rows 0 and 1 are the same-color pair; row 2 is a distractor.
        body_features[row, 0:2, index["color_0"]] = 1.0
        body_features[row, 2, index["color_1"]] = 1.0
        left_x = 0.20 + 0.01 * (row % 4)
        right_x = 0.70 - 0.01 * (row % 4)
        body_features[row, 0, index["effect_x_norm"]] = left_x
        body_features[row, 1, index["effect_x_norm"]] = right_x
        body_features[row, 2, index["effect_x_norm"]] = 0.48
        if row % 2:
            first_y, second_y, target = 0.72, 0.32, 0
        else:
            first_y, second_y, target = 0.32, 0.72, 1
        body_features[row, 0, index["effect_y_norm"]] = first_y
        body_features[row, 1, index["effect_y_norm"]] = second_y
        body_features[row, 2, index["effect_y_norm"]] = 0.50
        mode = row % 3
        global_features[row, gauge_index] = (0.95, 0.55, 0.10)[mode]
        kind = (0, 1, 2)[mode]
        kinds.append(kind)
        waits.append(2)
        targets.append(target if kind else 0)
        target_x = float(body_features[row, target, index["effect_x_norm"]])
        templates.append(6 if target_x < 0.5 else 8)
        values.append((kind - 1) * 0.5 + max(first_y, second_y))
    return (
        EncodedBatch(
            global_features,
            body_features,
            body_mask,
            tick,
            health,
            schema,
        ),
        PointerActionTensor(
            torch.tensor(kinds),
            torch.tensor(waits),
            torch.tensor(targets),
            torch.tensor(templates),
        ),
        tuple(f"episode-{row // 2}" for row in range(examples)),
        torch.tensor(values, dtype=torch.float32),
    )


def _small_model() -> EntityPointerActorCritic:
    return EntityPointerActorCritic(
        TEACHER_V1,
        config=PointerModelConfig(
            global_hidden=16,
            body_hidden=16,
            attention_hidden=32,
            attention_heads=4,
            attention_layers=1,
            feedforward_hidden=64,
            recurrent_hidden=32,
        ),
    )


class PointerDatasetTests(unittest.TestCase):
    def test_split_is_deterministic_and_episode_disjoint(self) -> None:
        encoded, labels, identities, values = _same_color_batch()
        dataset = PointerDataset.from_encoded_batch(
            encoded, labels, identities, values
        )
        first_train, first_validation = dataset.split_by_episode(
            1 / 3, salt="fixed"
        )
        second_train, second_validation = dataset.split_by_episode(
            1 / 3, salt="fixed"
        )
        self.assertEqual(first_train.episode_identities, second_train.episode_identities)
        self.assertEqual(
            first_validation.episode_identities,
            second_validation.episode_identities,
        )
        self.assertTrue(
            set(first_train.episode_identities).isdisjoint(
                first_validation.episode_identities
            )
        )

    def test_active_masked_or_missing_target_fails_closed(self) -> None:
        encoded, labels, identities, values = _same_color_batch(3)
        encoded.body_mask[0, 1] = False
        labels.target_index[0] = 1
        labels.kind[0] = 1
        with self.assertRaisesRegex(ValueError, "masked body"):
            PointerDataset.from_encoded_batch(
                encoded, labels, identities, values
            )


class PointerDistillationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(19)
        torch.set_num_threads(1)

    def test_tiny_same_color_dataset_overfits(self) -> None:
        encoded, labels, identities, values = _same_color_batch()
        dataset = PointerDataset.from_encoded_batch(
            encoded, labels, identities, values
        )
        trainer = PointerBCTrainer(
            _small_model(),
            config=PointerBCConfig(
                learning_rate=3e-3,
                value_coefficient=0.05,
            ),
        )
        initial = trainer.evaluate(dataset)
        final = trainer.fit(dataset, 300)
        evaluated = trainer.evaluate(dataset)
        self.assertLess(evaluated.total_loss, initial.total_loss * 0.1)
        self.assertGreaterEqual(evaluated.kind_accuracy, 0.99)
        self.assertGreaterEqual(evaluated.wait_accuracy, 0.99)
        self.assertGreaterEqual(evaluated.target_accuracy, 0.99)
        self.assertGreaterEqual(evaluated.template_accuracy, 0.99)
        self.assertEqual(evaluated.actionable_recall, 1.0)
        self.assertAlmostEqual(evaluated.wait_only_rate, 1 / 3)
        self.assertTrue(np.isfinite(final.gradient_norm))

    def test_model_and_labels_remain_permutation_consistent(self) -> None:
        encoded, labels, identities, values = _same_color_batch()
        dataset = PointerDataset.from_encoded_batch(
            encoded, labels, identities, values
        )
        model = _small_model().eval()
        trainer = PointerBCTrainer(
            model, config=PointerBCConfig(learning_rate=3e-3)
        )
        trainer.fit(dataset, 20)
        tensors = dataset.as_tensors()
        permutation = torch.tensor([2, 0, 1])
        inverse = torch.argsort(permutation)
        body = tensors.body_features[:, :3]
        mask = tensors.body_mask[:, :3]
        with torch.no_grad():
            first = model(
                tensors.global_features.unsqueeze(0),
                body.unsqueeze(0),
                mask.unsqueeze(0),
                model.initial_state(len(dataset)),
                reset_before=torch.ones((1, len(dataset)), dtype=torch.bool),
            )
            second = model(
                tensors.global_features.unsqueeze(0),
                body[:, permutation].unsqueeze(0),
                mask[:, permutation].unsqueeze(0),
                model.initial_state(len(dataset)),
                reset_before=torch.ones((1, len(dataset)), dtype=torch.bool),
            )
        torch.testing.assert_close(first.kind_logits, second.kind_logits)
        torch.testing.assert_close(first.wait_logits, second.wait_logits)
        torch.testing.assert_close(first.value_quantiles, second.value_quantiles)
        torch.testing.assert_close(
            first.target_logits,
            second.target_logits[..., inverse],
            atol=2e-5,
            rtol=2e-5,
        )
        torch.testing.assert_close(
            first.template_logits,
            second.template_logits[..., inverse, :],
            atol=2e-5,
            rtol=2e-5,
        )
