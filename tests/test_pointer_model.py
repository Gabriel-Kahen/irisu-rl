from __future__ import annotations

import unittest

import torch

from irisu_pointer.action import (
    PointerActionSpec,
    PointerActionTensor,
    decode_pointer_action,
)
from irisu_pointer.model import EntityPointerActorCritic, PointerModelConfig
from irisu_rl.actions import SemanticAction, SemanticActionKind
from irisu_rl.schema import TEACHER_V1


def _small_model() -> EntityPointerActorCritic:
    torch.manual_seed(19)
    return EntityPointerActorCritic(
        TEACHER_V1,
        config=PointerModelConfig(
            global_hidden=12,
            body_hidden=12,
            attention_hidden=24,
            attention_heads=4,
            attention_layers=2,
            feedforward_hidden=48,
            recurrent_hidden=20,
        ),
    )


def _observations(
    time: int = 2, batch: int = 2, bodies: int = 4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(23)
    global_features = torch.randn(
        time, batch, len(TEACHER_V1.global_features), generator=generator
    )
    body_features = torch.randn(
        time,
        batch,
        bodies,
        len(TEACHER_V1.body_features),
        generator=generator,
    )
    x = TEACHER_V1.body_features.index("effect_x_norm")
    y = TEACHER_V1.body_features.index("effect_y_norm")
    width = TEACHER_V1.body_features.index("width_norm")
    kind_indices = [
        TEACHER_V1.body_features.index(name)
        for name in (
            "kind_piece",
            "kind_projectile",
            "kind_bonus",
            "kind_unknown",
        )
    ]
    color_indices = [
        TEACHER_V1.body_features.index(f"color_{index}") for index in range(6)
    ] + [
        TEACHER_V1.body_features.index("color_bonus"),
        TEACHER_V1.body_features.index("color_unknown"),
    ]
    lifecycle_indices = [
        TEACHER_V1.body_features.index(name)
        for name in (
            "lifecycle_falling",
            "lifecycle_fresh",
            "lifecycle_confirmed",
            "lifecycle_rotten",
            "lifecycle_ambiguous",
            "lifecycle_unknown",
        )
    ]
    height = TEACHER_V1.body_features.index("height_norm")
    body_features[..., kind_indices] = 0.0
    body_features[..., kind_indices[0]] = 1.0
    body_features[..., color_indices] = 0.0
    body_features[..., lifecycle_indices] = 0.0
    body_features[..., lifecycle_indices[0]] = 1.0
    for index in range(bodies):
        body_features[..., index, color_indices[index % 2]] = 1.0
    body_features[..., x] = torch.linspace(0.2, 0.8, bodies)
    body_features[..., y] = 0.5
    body_features[..., width] = 0.1
    body_features[..., height] = 0.1
    return (
        global_features,
        body_features,
        torch.ones((time, batch, bodies), dtype=torch.bool),
    )


class PointerActionTests(unittest.TestCase):
    def test_spec_and_tensor_validation(self) -> None:
        spec = PointerActionSpec()
        self.assertEqual(spec.wait_choices, (1, 2, 4, 8, 16))
        self.assertEqual(spec.template_count, 21)
        self.assertEqual(spec.templates[0], (-1.0, 1.5))
        self.assertEqual(spec.templates[-1], (1.5, 2.5))
        self.assertEqual(
            spec.manifest()["coordinate_frame"],
            "selected_effect_center_and_half_extents-v1",
        )
        self.assertNotIn("launch_y_norms", spec.manifest())
        labels = PointerActionTensor(
            kind=torch.tensor([0, 1, 2]),
            wait_index=torch.tensor([4, 0, 0]),
            target_index=torch.tensor([0, 1, 3]),
            template_index=torch.tensor([0, 10, 20]),
        )
        labels.validate(torch.Size((3,)), 4, spec)

    def test_decoder_uses_selected_row_and_validates_action(self) -> None:
        row = torch.zeros(len(TEACHER_V1.body_features))
        row[TEACHER_V1.body_features.index("effect_x_norm")] = 0.4
        row[TEACHER_V1.body_features.index("effect_y_norm")] = 0.6
        row[TEACHER_V1.body_features.index("width_norm")] = 0.2
        row[TEACHER_V1.body_features.index("height_norm")] = 0.2
        spec = PointerActionSpec()
        weak = decode_pointer_action(
            kind=int(SemanticActionKind.FIRE_WEAK),
            template_index=spec.templates.index((-1.0, 1.5)),
            selected_body_row=row,
            schema=TEACHER_V1,
            pointer_spec=spec,
        )
        self.assertEqual(weak.kind, SemanticActionKind.FIRE_WEAK)
        self.assertAlmostEqual(weak.x_norm, 0.3)
        self.assertAlmostEqual(weak.y_norm, 0.75)

        row[TEACHER_V1.body_features.index("effect_y_norm")] = 0.95
        clipped = decode_pointer_action(
            kind=int(SemanticActionKind.FIRE_STRONG),
            template_index=spec.templates.index((1.0, 2.5)),
            selected_body_row=row,
            schema=TEACHER_V1,
            pointer_spec=spec,
        )
        self.assertAlmostEqual(clipped.x_norm, 0.5)
        self.assertAlmostEqual(clipped.y_norm, 1.0)
        wait = decode_pointer_action(
            kind=int(SemanticActionKind.WAIT),
            wait_index=4,
            selected_body_row=None,
            schema=TEACHER_V1,
            pointer_spec=spec,
        )
        self.assertEqual(wait, SemanticAction.wait(16))


class EntityPointerModelTests(unittest.TestCase):
    def test_matcher_kind_prior_bootstraps_strength_and_is_neutral_without_pair(
        self,
    ) -> None:
        model = _small_model().eval()
        _, bodies, mask = _observations(time=1, batch=2, bodies=2)
        y = TEACHER_V1.body_features.index("effect_y_norm")
        colors = [
            TEACHER_V1.body_features.index(f"color_{index}")
            for index in range(6)
        ]
        bodies[..., colors] = 0.0
        bodies[:, 0, :, colors[0]] = 1.0
        bodies[:, 1, 0, colors[0]] = 1.0
        bodies[:, 1, 1, colors[1]] = 1.0
        bodies[:, 0, :, y] = torch.tensor([0.3, 0.2])
        prior = model._matcher_prior(
            bodies, model._semantic_piece_mask(bodies, mask)
        )
        residual = torch.tensor([[[8.0, 8.0, -8.0], [1.0, -1.0, 0.5]]])
        gated = model._gated_kind_logits(
            residual,
            prior,
            bodies,
            torch.ones((1, 2, 1)),
        )
        self.assertEqual(int(gated[0, 0].argmax()), 2)
        torch.testing.assert_close(
            gated[0, 1],
            model.config.kind_residual_scale * torch.tanh(residual[0, 1]),
        )
        self.assertGreater(
            float(torch.sigmoid(model.matcher_kind_gate_head.bias).item()), 0.8
        )

    def test_matcher_gate_mixes_prior_with_bounded_branch_residual(self) -> None:
        model = _small_model().eval()
        raw_residual = torch.tensor(
            [[[[3.0, -3.0, 0.0], [3.0, -3.0, 0.0]]]]
        )
        prior = torch.tensor([[[0.0, 1.0, 0.0]]])
        branch_gate = torch.tensor([[[1.0, 0.0]]])
        mixed = model._gated_target_logits(
            raw_residual, prior, branch_gate
        )
        self.assertEqual(int(mixed[0, 0, 0].argmax()), 1)
        self.assertEqual(int(mixed[0, 0, 1].argmax()), 0)
        self.assertGreaterEqual(
            float(mixed.min()), -model.config.target_residual_scale
        )
        self.assertLessEqual(
            float(mixed.max()), model.config.matcher_prior_scale
        )

        neutral = model._gated_target_logits(
            raw_residual, torch.zeros_like(prior), torch.ones_like(branch_gate)
        )
        torch.testing.assert_close(
            neutral,
            model.config.target_residual_scale * torch.tanh(raw_residual),
        )
        self.assertTrue((model.matcher_gate_head.bias > 0).all())
        self.assertGreater(
            float(torch.sigmoid(model.matcher_gate_head.bias.detach()).min()), 0.8
        )

    def test_matcher_prior_selects_closest_pair_then_lower_member(self) -> None:
        model = _small_model().eval()
        _, bodies, mask = _observations(time=1, batch=1)
        x = TEACHER_V1.body_features.index("effect_x_norm")
        y = TEACHER_V1.body_features.index("effect_y_norm")
        color_indices = [
            TEACHER_V1.body_features.index(f"color_{index}")
            for index in range(6)
        ]
        bodies[..., color_indices] = 0.0
        bodies[..., (0, 2), color_indices[0]] = 1.0
        bodies[..., (1, 3), color_indices[1]] = 1.0
        bodies[..., x] = torch.tensor([0.10, 0.60, 0.40, 0.62])
        bodies[..., y] = torch.tensor([0.10, 0.40, 0.70, 0.50])
        prior = model._matcher_prior(
            bodies, model._semantic_piece_mask(bodies, mask)
        )
        torch.testing.assert_close(
            prior, torch.tensor([[[0.0, 0.0, 0.0, 1.0]]])
        )

    def test_matcher_prior_uses_rightmost_for_equal_y(self) -> None:
        model = _small_model().eval()
        _, bodies, mask = _observations(time=1, batch=1, bodies=2)
        x = TEACHER_V1.body_features.index("effect_x_norm")
        y = TEACHER_V1.body_features.index("effect_y_norm")
        color_indices = [
            TEACHER_V1.body_features.index(f"color_{index}")
            for index in range(6)
        ]
        bodies[..., color_indices] = 0.0
        bodies[..., color_indices[2]] = 1.0
        bodies[..., x] = torch.tensor([0.40, 0.60])
        bodies[..., y] = 0.55
        prior = model._matcher_prior(
            bodies, model._semantic_piece_mask(bodies, mask)
        )
        torch.testing.assert_close(prior, torch.tensor([[[0.0, 1.0]]]))

    def test_matcher_prior_is_equivariant_padding_safe_and_neutral_without_pair(
        self,
    ) -> None:
        model = _small_model().eval()
        _, bodies, mask = _observations(time=1, batch=1)
        permutation = torch.tensor([2, 0, 3, 1])
        prior = model._matcher_prior(
            bodies, model._semantic_piece_mask(bodies, mask)
        )
        permuted_bodies = bodies[..., permutation, :]
        permuted_mask = mask[..., permutation]
        permuted = model._matcher_prior(
            permuted_bodies,
            model._semantic_piece_mask(permuted_bodies, permuted_mask),
        )
        torch.testing.assert_close(prior[..., permutation], permuted)

        padding = torch.randn(
            1,
            1,
            2,
            len(TEACHER_V1.body_features),
            generator=torch.Generator().manual_seed(37),
        )
        padded_bodies = torch.cat((bodies, padding), dim=-2)
        padded_mask = torch.cat(
            (mask, torch.zeros((1, 1, 2), dtype=torch.bool)), dim=-1
        )
        padded = model._matcher_prior(
            padded_bodies,
            model._semantic_piece_mask(padded_bodies, padded_mask),
        )
        torch.testing.assert_close(prior, padded[..., :4])
        self.assertTrue((padded[..., 4:] == 0).all())

        no_pair_bodies = bodies[..., :2, :].clone()
        no_pair_mask = mask[..., :2]
        colors = [
            TEACHER_V1.body_features.index(f"color_{index}")
            for index in range(6)
        ]
        no_pair_bodies[..., colors] = 0.0
        no_pair_bodies[..., 0, colors[0]] = 1.0
        no_pair_bodies[..., 1, colors[1]] = 1.0
        neutral = model._matcher_prior(
            no_pair_bodies,
            model._semantic_piece_mask(no_pair_bodies, no_pair_mask),
        )
        self.assertTrue((neutral == 0).all())

    def test_teacher_id_is_ablated_from_every_model_output(self) -> None:
        model = _small_model().eval()
        manifest = model.manifest()
        self.assertEqual(manifest["input_ablations"], ["id_scaled"])
        self.assertEqual(
            manifest["config"]["matcher_prior_scale"],
            model.config.matcher_prior_scale,
        )
        self.assertEqual(
            manifest["config"]["target_residual_scale"],
            model.config.target_residual_scale,
        )
        global_features, bodies, mask = _observations()
        id_index = TEACHER_V1.body_features.index("id_scaled")
        changed = bodies.clone()
        bodies[..., id_index] = torch.linspace(0.0, 1.0, bodies.shape[-2])
        changed[..., id_index] = torch.linspace(1.0, 0.0, bodies.shape[-2])
        with torch.no_grad():
            baseline = model(
                global_features, bodies, mask, model.initial_state(2)
            )
            renumbered = model(
                global_features, changed, mask, model.initial_state(2)
            )
        for left, right in (
            (baseline.kind_logits, renumbered.kind_logits),
            (baseline.wait_logits, renumbered.wait_logits),
            (baseline.target_logits, renumbered.target_logits),
            (baseline.template_logits, renumbered.template_logits),
            (baseline.values, renumbered.values),
            (baseline.value_quantiles, renumbered.value_quantiles),
            (baseline.recurrent_state, renumbered.recurrent_state),
        ):
            torch.testing.assert_close(left, right)

    def test_relation_context_is_equivariant_and_padding_safe(self) -> None:
        model = _small_model().eval()
        _, bodies, mask = _observations()
        permutation = torch.tensor([2, 0, 3, 1])
        padding = torch.randn(
            2, 2, 2, len(TEACHER_V1.body_features),
            generator=torch.Generator().manual_seed(31),
        )
        padded_bodies = torch.cat((bodies, padding), dim=2)
        padded_mask = torch.cat(
            (mask, torch.zeros((2, 2, 2), dtype=torch.bool)), dim=2
        )
        with torch.no_grad():
            piece_mask = model._semantic_piece_mask(bodies, mask)
            baseline = model._relation_context(bodies, piece_mask)
            permuted_bodies = bodies[:, :, permutation]
            permuted_mask = mask[:, :, permutation]
            permuted = model._relation_context(
                permuted_bodies,
                model._semantic_piece_mask(permuted_bodies, permuted_mask),
            )
            padded = model._relation_context(
                padded_bodies,
                model._semantic_piece_mask(padded_bodies, padded_mask),
            )
        torch.testing.assert_close(
            baseline[..., permutation, :], permuted, rtol=1e-5, atol=1e-6
        )
        torch.testing.assert_close(
            baseline, padded[..., :4, :], rtol=1e-5, atol=1e-6
        )
        self.assertTrue((padded[..., 4:, :] == 0).all())

    def test_relation_context_prefers_near_same_color_and_excludes_nonpieces(
        self,
    ) -> None:
        model = _small_model().eval()
        _, bodies, mask = _observations(time=1, batch=1, bodies=3)
        colors = [
            TEACHER_V1.body_features.index(f"color_{index}")
            for index in range(6)
        ]
        x = TEACHER_V1.body_features.index("effect_x_norm")
        bodies[..., colors] = 0.0
        bodies[..., 0, colors[0]] = 1.0
        bodies[..., 1, colors[0]] = 1.0
        bodies[..., 2, colors[1]] = 1.0
        bodies[..., 0, x] = 0.50
        bodies[..., 1, x] = 0.55
        bodies[..., 2, x] = 0.85
        with torch.no_grad():
            same_color_near = model._relation_context(
                bodies, model._semantic_piece_mask(bodies, mask)
            )
            swapped = bodies.clone()
            swapped[..., 1, colors[0]] = 0.0
            swapped[..., 1, colors[1]] = 1.0
            swapped[..., 2, colors[1]] = 0.0
            swapped[..., 2, colors[0]] = 1.0
            same_color_far = model._relation_context(
                swapped, model._semantic_piece_mask(swapped, mask)
            )
        self.assertGreater(
            float((same_color_near[..., 0, :] - same_color_far[..., 0, :]).norm()),
            1e-5,
        )

        projectile = TEACHER_V1.body_features.index("kind_projectile")
        piece = TEACHER_V1.body_features.index("kind_piece")
        nonpieces = bodies.clone()
        nonpieces[..., 1:, piece] = 0.0
        nonpieces[..., 1:, projectile] = 1.0
        with torch.no_grad():
            excluded = model._relation_context(
                nonpieces, model._semantic_piece_mask(nonpieces, mask)
            )
        self.assertTrue((excluded == 0).all())

    def test_semantic_target_mask_excludes_nonpieces(self) -> None:
        model = _small_model().eval()
        global_features, bodies, mask = _observations(time=1, batch=1)
        kinds = {
            name: TEACHER_V1.body_features.index(name)
            for name in (
                "kind_piece",
                "kind_projectile",
                "kind_bonus",
                "kind_unknown",
            )
        }
        bodies[..., 1, list(kinds.values())] = 0.0
        bodies[..., 1, kinds["kind_projectile"]] = 1.0
        bodies[..., 2, list(kinds.values())] = 0.0
        bodies[..., 2, kinds["kind_bonus"]] = 1.0
        bodies[..., 3, list(kinds.values())] = 0.25
        output = model(
            global_features, bodies, mask, model.initial_state(1)
        )
        floor = torch.finfo(output.target_logits.dtype).min
        self.assertTrue((output.target_logits[..., 0] > floor).all())
        self.assertTrue((output.template_logits[..., 0, :] > floor).all())
        self.assertTrue((output.target_logits[..., 1:] == floor).all())
        self.assertTrue((output.template_logits[..., 1:, :] == floor).all())
        self.assertTrue((output.kind_logits[..., 1:] > floor).all())

    def test_no_piece_forces_wait_and_uses_quantile_mean_value(self) -> None:
        model = _small_model().eval()
        global_features, bodies, mask = _observations(time=1, batch=2)
        piece = TEACHER_V1.body_features.index("kind_piece")
        projectile = TEACHER_V1.body_features.index("kind_projectile")
        bodies[..., piece] = 0.0
        bodies[..., projectile] = 1.0
        output = model(
            global_features, bodies, mask, model.initial_state(2)
        )
        floor = torch.finfo(output.kind_logits.dtype).min
        self.assertTrue((output.kind_logits[..., 1:] == floor).all())
        self.assertTrue((output.kind_logits.argmax(dim=-1) == 0).all())
        self.assertTrue((output.target_logits == floor).all())
        self.assertTrue((output.template_logits == floor).all())
        torch.testing.assert_close(
            output.values, output.value_quantiles.mean(dim=-1)
        )
        self.assertFalse(
            any(name.startswith("value_head.") for name in model.state_dict())
        )

    def test_explicit_target_mask_overrides_semantic_kind_and_is_validated(self) -> None:
        model = _small_model().eval()
        global_features, bodies, mask = _observations(time=1, batch=1)
        piece = TEACHER_V1.body_features.index("kind_piece")
        projectile = TEACHER_V1.body_features.index("kind_projectile")
        bodies[..., 1, piece] = 0.0
        bodies[..., 1, projectile] = 1.0
        explicit = torch.tensor([[[False, True, False, False]]])
        output = model(
            global_features,
            bodies,
            mask,
            model.initial_state(1),
            target_mask=explicit,
        )
        floor = torch.finfo(output.target_logits.dtype).min
        self.assertTrue((output.target_logits[..., 1] > floor).all())
        self.assertTrue((output.target_logits[..., (0, 2, 3)] == floor).all())
        with self.assertRaisesRegex(ValueError, "target mask"):
            model(
                global_features,
                bodies,
                mask,
                model.initial_state(1),
                target_mask=explicit.float(),
            )
        with self.assertRaisesRegex(ValueError, "target mask"):
            model(
                global_features,
                bodies,
                mask,
                model.initial_state(1),
                target_mask=explicit[..., :3],
            )

    def test_body_permutation_is_equivariant_and_decoding_is_invariant(self) -> None:
        model = _small_model().eval()
        global_features, bodies, mask = _observations()
        permutation = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            baseline = model(
                global_features, bodies, mask, model.initial_state(2)
            )
            permuted = model(
                global_features,
                bodies[:, :, permutation],
                mask[:, :, permutation],
                model.initial_state(2),
            )
        for left, right in (
            (baseline.kind_logits, permuted.kind_logits),
            (baseline.wait_logits, permuted.wait_logits),
            (baseline.values, permuted.values),
            (baseline.value_quantiles, permuted.value_quantiles),
            (baseline.recurrent_state, permuted.recurrent_state),
        ):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(
            baseline.target_logits[..., permutation],
            permuted.target_logits,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            baseline.template_logits[..., permutation, :],
            permuted.template_logits,
            rtol=1e-5,
            atol=1e-6,
        )

        branch = 0
        base_target = int(baseline.target_logits[0, 0, branch].argmax())
        perm_target = int(permuted.target_logits[0, 0, branch].argmax())
        base_template = int(
            baseline.template_logits[0, 0, branch, base_target].argmax()
        )
        perm_template = int(
            permuted.template_logits[0, 0, branch, perm_target].argmax()
        )
        base_action = decode_pointer_action(
            kind=int(SemanticActionKind.FIRE_WEAK),
            template_index=base_template,
            selected_body_row=bodies[0, 0, base_target],
            schema=TEACHER_V1,
        )
        perm_action = decode_pointer_action(
            kind=int(SemanticActionKind.FIRE_WEAK),
            template_index=perm_template,
            selected_body_row=bodies[0, 0, permutation[perm_target]],
            schema=TEACHER_V1,
        )
        self.assertEqual(base_action, perm_action)

    def test_padding_cannot_change_real_entity_or_global_outputs(self) -> None:
        model = _small_model().eval()
        global_features, bodies, mask = _observations(bodies=3)
        generator = torch.Generator().manual_seed(29)
        padding = torch.randn(
            2,
            2,
            2,
            len(TEACHER_V1.body_features),
            generator=generator,
        )
        padded_bodies = torch.cat((bodies, padding), dim=2)
        padded_mask = torch.cat(
            (mask, torch.zeros((2, 2, 2), dtype=torch.bool)), dim=2
        )
        with torch.no_grad():
            short = model(
                global_features, bodies, mask, model.initial_state(2)
            )
            padded = model(
                global_features,
                padded_bodies,
                padded_mask,
                model.initial_state(2),
            )
        for left, right in (
            (short.kind_logits, padded.kind_logits),
            (short.wait_logits, padded.wait_logits),
            (short.values, padded.values),
            (short.value_quantiles, padded.value_quantiles),
            (short.recurrent_state, padded.recurrent_state),
            (short.target_logits, padded.target_logits[..., :3]),
            (short.template_logits, padded.template_logits[..., :3, :]),
        ):
            torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)

    def test_masked_targets_have_finite_floor_logits(self) -> None:
        model = _small_model().eval()
        global_features, bodies, mask = _observations(time=1, batch=1)
        mask[..., 1] = False
        mask[..., 3] = False
        output = model(
            global_features, bodies, mask, model.initial_state(1)
        )
        floor = torch.finfo(output.target_logits.dtype).min
        self.assertTrue(torch.isfinite(output.target_logits).all())
        self.assertTrue(torch.isfinite(output.template_logits).all())
        self.assertTrue((output.target_logits[..., (1, 3)] == floor).all())
        self.assertTrue((output.template_logits[..., (1, 3), :] == floor).all())
        self.assertTrue((output.target_logits[..., (0, 2)] > floor).all())

    def test_all_parameters_receive_finite_gradients(self) -> None:
        model = _small_model().train()
        global_features, bodies, mask = _observations()
        output = model(
            global_features,
            bodies,
            mask,
            model.initial_state(2),
            reset_before=torch.tensor([[True, True], [False, False]]),
        )
        loss = (
            output.kind_logits.square().mean()
            + output.wait_logits.square().mean()
            + output.target_logits[..., 0].square().mean()
            + output.template_logits[..., 0, :].square().mean()
            + output.values.square().mean()
            + output.value_quantiles.square().mean()
        )
        loss.backward()
        missing = [
            name for name, parameter in model.named_parameters() if parameter.grad is None
        ]
        nonfinite = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
        ]
        zero = [
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
            and float(parameter.grad.detach().abs().sum()) == 0.0
        ]
        self.assertEqual(missing, [])
        self.assertEqual(nonfinite, [])
        self.assertEqual(zero, [])


if __name__ == "__main__":
    unittest.main()
