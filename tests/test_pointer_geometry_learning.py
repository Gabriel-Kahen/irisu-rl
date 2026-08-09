from __future__ import annotations

import hashlib
import unittest

import torch

from irisu_pointer.geometry_learning import (
    GeometryDataset,
    GeometryModelConfig,
    GeometrySelectorModel,
    geometry_example,
    train_geometry_selector,
)
from irisu_rl.schema import TEACHER_V1


_PROVENANCE = hashlib.sha256(b"geometry-learning-test").hexdigest()
_CANDIDATES = hashlib.sha256(b"geometry-candidates").hexdigest()


def _body(identifier: int, x: float, color: int = 0) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": "dynamic_fresh",
        "color": color,
        "x": x,
        "y": 180.0,
        "vx": 0.0,
        "vy": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 40.0,
        "chain_id": 0,
        "projectile_hits": 0,
        "age_ticks": 20,
        "remaining_lifetime": 1000,
        "rot_timer": 0,
    }


def _observation(*, reverse: bool = False, gauge: int = 1000):
    bodies = [_body(11, 160.0), _body(22, 300.0), _body(33, 480.0, 1)]
    if reverse:
        bodies.reverse()
    return {
        "tick": 0,
        "score": 0,
        "gauge": gauge,
        "gauge_max": 1000,
        "level": 1,
        "highest_chain": 0,
        "qualifying_clear_count": 0,
        "left_held": False,
        "right_held": False,
        "terminated": False,
        "truncated": False,
        "field": {"x": 0.0, "y": 0.0, "width": 640.0, "height": 480.0},
        "difficulty": {"active_colors": 4, "spawn_interval_ticks": 100},
        "bodies": bodies,
    }


def _example(index: int, *, improved: bool, gauge: int = 1000):
    return geometry_example(
        _observation(gauge=gauge),
        source_body_id=11,
        destination_body_id=22,
        candidate_index=index,
        candidate_count=3,
        improved_over_incumbent=improved,
        episode_identity=f"episode-{index}-{gauge}",
        provenance_sha256=_PROVENANCE,
        candidate_set_sha256=_CANDIDATES,
    )


def _selector_inputs():
    example = geometry_example(
        _observation(),
        source_body_id=11,
        destination_body_id=22,
        candidate_index=0,
        candidate_count=3,
        improved_over_incumbent=False,
        episode_identity="selector-inputs",
        provenance_sha256=_PROVENANCE,
        candidate_set_sha256=_CANDIDATES,
    )
    return GeometryDataset((example,)).tensors()[:-1]


class GeometryLearningTests(unittest.TestCase):
    def test_example_binds_public_body_ids_and_dataset_identity(self) -> None:
        example = _example(2, improved=True)
        dataset = GeometryDataset((example,))

        self.assertEqual(example.candidate_index, 2)
        self.assertNotEqual(example.source_index, example.destination_index)
        self.assertEqual(dataset.candidate_count, 3)
        self.assertEqual(len(dataset.sha256), 64)

    def test_selector_is_invariant_to_body_order_with_rebound_indices(self) -> None:
        first = geometry_example(
            _observation(),
            source_body_id=11,
            destination_body_id=22,
            candidate_index=0,
            candidate_count=3,
            improved_over_incumbent=False,
            episode_identity="first",
            provenance_sha256=_PROVENANCE,
            candidate_set_sha256=_CANDIDATES,
        )
        second = geometry_example(
            _observation(reverse=True),
            source_body_id=11,
            destination_body_id=22,
            candidate_index=0,
            candidate_count=3,
            improved_over_incumbent=False,
            episode_identity="second",
            provenance_sha256=_PROVENANCE,
            candidate_set_sha256=_CANDIDATES,
        )
        torch.manual_seed(7)
        model = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=3,
            candidate_set_sha256=_CANDIDATES,
            config=GeometryModelConfig(body_hidden=16, pair_hidden=24),
        )

        outputs = []
        for example in (first, second):
            tensors = GeometryDataset((example,)).tensors()
            outputs.append(model(*tensors[:-1]))
        torch.testing.assert_close(outputs[0], outputs[1])

    def test_disabled_context_preserves_legacy_manifest_and_state_dict(self) -> None:
        legacy_config = {
            "body_hidden": 16,
            "pair_hidden": 24,
            "dropout": 0.0,
        }
        torch.manual_seed(31)
        first = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=3,
            candidate_set_sha256=_CANDIDATES,
            config=GeometryModelConfig(**legacy_config),
        )
        manifest = first.manifest()

        self.assertEqual(
            manifest["architecture"], "directed-pair-geometry-selector-v1"
        )
        self.assertEqual(manifest["config"], legacy_config)
        self.assertNotIn("board_context", manifest)
        self.assertFalse(
            any("board_context" in name for name in first.state_dict())
        )

        reconstructed = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=int(manifest["candidate_count"]),
            candidate_set_sha256=str(manifest["candidate_set_sha256"]),
            config=GeometryModelConfig(**manifest["config"]),
        )
        reconstructed.load_state_dict(first.state_dict(), strict=True)
        self.assertEqual(reconstructed.manifest(), manifest)
        torch.testing.assert_close(
            reconstructed(*_selector_inputs()),
            first(*_selector_inputs()),
        )

    def test_board_context_is_invariant_to_body_row_permutation(self) -> None:
        torch.manual_seed(37)
        model = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=3,
            candidate_set_sha256=_CANDIDATES,
            config=GeometryModelConfig(
                body_hidden=16,
                pair_hidden=24,
                board_context_hidden=12,
            ),
        )
        global_features, bodies, mask, source, destination = (
            _selector_inputs()
        )
        bodies = bodies.clone()
        third = next(
            index
            for index in range(bodies.shape[1])
            if index not in {int(source[0]), int(destination[0])}
        )
        chain_index = TEACHER_V1.body_features.index("chain_id_scaled")
        bodies[0, destination[0], chain_index] = 7 / 2**32
        bodies[0, third, chain_index] = 7 / 2**32
        permutation = torch.tensor([2, 0, 1])
        source_permuted = torch.tensor(
            [int((permutation == source[0]).nonzero()[0])]
        )
        destination_permuted = torch.tensor(
            [int((permutation == destination[0]).nonzero()[0])]
        )

        original = model(
            global_features, bodies, mask, source, destination
        )
        permuted = model(
            global_features,
            bodies[:, permutation],
            mask[:, permutation],
            source_permuted,
            destination_permuted,
        )

        torch.testing.assert_close(original, permuted)

    def test_board_context_masks_padding_and_ablates_ids(self) -> None:
        torch.manual_seed(41)
        model = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=3,
            candidate_set_sha256=_CANDIDATES,
            config=GeometryModelConfig(
                body_hidden=16,
                pair_hidden=24,
                board_context_hidden=12,
            ),
        )
        global_features, bodies, mask, source, destination = (
            _selector_inputs()
        )
        baseline = model(
            global_features, bodies, mask, source, destination
        )
        padding = torch.randn(
            bodies.shape[0], 2, bodies.shape[2]
        ) * 10_000.0
        padded = model(
            global_features,
            torch.cat((bodies, padding), dim=1),
            torch.cat(
                (
                    mask,
                    torch.zeros(
                        mask.shape[0], 2, dtype=torch.bool
                    ),
                ),
                dim=1,
            ),
            source,
            destination,
        )
        changed_ids = bodies.clone()
        id_index = TEACHER_V1.body_features.index("id_scaled")
        changed_ids[..., id_index] = torch.randn_like(
            changed_ids[..., id_index]
        ) * 1000.0

        torch.testing.assert_close(baseline, padded)
        torch.testing.assert_close(
            baseline,
            model(
                global_features,
                changed_ids,
                mask,
                source,
                destination,
            ),
        )
        third = next(
            index
            for index in range(bodies.shape[1])
            if index not in {int(source[0]), int(destination[0])}
        )
        chain_index = TEACHER_V1.body_features.index("chain_id_scaled")
        first_labels = bodies.clone()
        first_labels[0, destination[0], chain_index] = 7 / 2**32
        first_labels[0, third, chain_index] = 7 / 2**32
        second_labels = first_labels.clone()
        second_labels[0, destination[0], chain_index] = 997 / 2**32
        second_labels[0, third, chain_index] = 997 / 2**32
        torch.testing.assert_close(
            model(
                global_features,
                first_labels,
                mask,
                source,
                destination,
            ),
            model(
                global_features,
                second_labels,
                mask,
                source,
                destination,
            ),
        )

    def test_board_context_exposes_strategy_relations_without_chain_ids(
        self,
    ) -> None:
        torch.manual_seed(47)
        model = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=3,
            candidate_set_sha256=_CANDIDATES,
            config=GeometryModelConfig(
                body_hidden=16,
                pair_hidden=24,
                board_context_hidden=12,
            ),
        )
        manifest = model.manifest()
        self.assertEqual(
            manifest["architecture"],
            "directed-pair-geometry-selector-v3-strategic-board-context",
        )
        self.assertEqual(
            manifest["board_context"]["pair_relative_features"][-4:],
            [
                "same_color_as_source",
                "same_color_as_destination",
                "grouped",
                "same_chain_as_destination",
            ],
        )
        global_features, bodies, mask, source, destination = (
            _selector_inputs()
        )
        bodies = bodies.clone()
        third = next(
            index
            for index in range(bodies.shape[1])
            if index not in {int(source[0]), int(destination[0])}
        )
        colors = [
            TEACHER_V1.body_features.index(f"color_{index}")
            for index in range(6)
        ]
        bodies[..., colors] = 0.0
        bodies[0, source[0], colors[0]] = 1.0
        bodies[0, destination[0], colors[1]] = 1.0
        bodies[0, third, colors[0]] = 1.0
        chain_index = TEACHER_V1.body_features.index("chain_id_scaled")
        bodies[0, destination[0], chain_index] = 7 / 2**32
        bodies[0, third, chain_index] = 7 / 2**32

        flags = model._board_context_relations(
            bodies, source, destination
        )[..., -4:]
        torch.testing.assert_close(
            flags[0, source[0]], torch.tensor([1.0, 0.0, 0.0, 0.0])
        )
        torch.testing.assert_close(
            flags[0, destination[0]],
            torch.tensor([0.0, 1.0, 1.0, 1.0]),
        )
        torch.testing.assert_close(
            flags[0, third], torch.tensor([1.0, 0.0, 1.0, 1.0])
        )

        different_chain = bodies.clone()
        different_chain[0, third, chain_index] = 11 / 2**32
        ungrouped = different_chain.clone()
        ungrouped[0, third, chain_index] = 0.0
        same_output = model(
            global_features, bodies, mask, source, destination
        )
        different_output = model(
            global_features,
            different_chain,
            mask,
            source,
            destination,
        )
        ungrouped_output = model(
            global_features, ungrouped, mask, source, destination
        )

        self.assertFalse(torch.allclose(same_output, different_output))
        self.assertFalse(torch.allclose(different_output, ungrouped_output))

    def test_board_context_has_finite_gradients_and_uses_third_body(self) -> None:
        torch.manual_seed(43)
        model = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=3,
            candidate_set_sha256=_CANDIDATES,
            config=GeometryModelConfig(
                body_hidden=16,
                pair_hidden=24,
                board_context_hidden=12,
            ),
        )
        global_features, bodies, mask, source, destination = (
            _selector_inputs()
        )
        bodies = bodies.clone().requires_grad_(True)
        baseline = model(
            global_features, bodies, mask, source, destination
        )
        self.assertEqual(baseline.shape, (1, 3))
        changed = bodies.detach().clone()
        third = next(
            index
            for index in range(changed.shape[1])
            if index not in {int(source[0]), int(destination[0])}
        )
        x_index = TEACHER_V1.body_features.index("effect_x_norm")
        changed[0, third, x_index] -= 0.35
        contextual = model(
            global_features, changed, mask, source, destination
        )

        self.assertFalse(torch.allclose(baseline, contextual))
        baseline.square().mean().backward()
        self.assertIsNotNone(bodies.grad)
        self.assertTrue(torch.isfinite(bodies.grad).all())
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(value is not None for value in gradients))
        self.assertTrue(
            all(torch.isfinite(value).all() for value in gradients)
        )

    def test_balanced_training_learns_improved_and_incumbent_examples(self) -> None:
        examples = tuple(
            _example(2 if gauge < 500 else 0, improved=gauge < 500, gauge=gauge)
            for gauge in (100, 200, 300, 400, 600, 700, 800, 900)
        )
        dataset = GeometryDataset(examples)
        torch.manual_seed(9)
        model = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=3,
            candidate_set_sha256=_CANDIDATES,
            config=GeometryModelConfig(body_hidden=24, pair_hidden=32),
        )

        report = train_geometry_selector(
            model,
            dataset,
            steps=120,
            batch_size=8,
            learning_rate=1e-3,
            seed=11,
        )

        self.assertLess(report.final_loss, report.initial_loss)
        self.assertGreaterEqual(report.accuracy, 0.75)
        self.assertGreaterEqual(report.improved_accuracy, 0.75)
        self.assertGreaterEqual(report.incumbent_accuracy, 0.75)


if __name__ == "__main__":
    unittest.main()
