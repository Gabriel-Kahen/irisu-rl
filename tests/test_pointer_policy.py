from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from irisu_env import ActionKind
from irisu_pointer.action import PointerActionSpec
from irisu_pointer.checkpoint import (
    load_actor_pointer_policy,
    load_pointer_checkpoint,
    load_teacher_pointer_policy,
    save_pointer_checkpoint,
)
from irisu_pointer.model import EntityPointerActorCritic, PointerModelConfig
from irisu_pointer.policy import (
    RecurrentActorPointerPolicy,
    RecurrentPointerPolicy,
    encoded_body_ids,
    target_index_for_decision,
)
from irisu_rl.actions import ActionSpec, SemanticActionKind
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.schema import ACTOR_VISION_V1, TEACHER_V1


def _observation() -> dict[str, object]:
    return {
        "tick": 17,
        "score": 0,
        "gauge": 500,
        "gauge_max": 1000,
        "level": 0,
        "highest_chain": 0,
        "qualifying_clear_count": 0,
        "left_held": False,
        "right_held": False,
        "terminated": False,
        "truncated": False,
        "difficulty": {"active_colors": 3, "spawn_interval_ticks": 300},
        "field": {
            "x": 16.0,
            "y": 8.0,
            "width": 608.0,
            "height": 464.0,
            "side_wall_top": 0.0,
            "side_wall_bottom": 480.0,
        },
        "bodies": [
            {
                "id": 41,
                "kind": "piece",
                "shape": "box",
                "lifecycle": "scripted_falling",
                "color": 2,
                "x": 220.0,
                "y": 160.0,
                "vx": 0.0,
                "vy": 0.0,
                "angle": 0.0,
                "angular_velocity": 0.0,
                "size": 24.0,
                "chain_id": 0,
                "projectile_hits": 0,
                "age_ticks": 0,
                "remaining_lifetime": 500,
                "rot_timer": 0,
            },
            {
                "id": 73,
                "kind": "piece",
                "shape": "circle",
                "lifecycle": "dynamic_fresh",
                "color": 2,
                "x": 238.0,
                "y": 260.0,
                "vx": 0.0,
                "vy": 0.0,
                "angle": 0.0,
                "angular_velocity": 0.0,
                "size": 24.0,
                "chain_id": 0,
                "projectile_hits": 0,
                "age_ticks": 20,
                "remaining_lifetime": 500,
                "rot_timer": 0,
            },
        ],
    }


def _model() -> EntityPointerActorCritic:
    torch.manual_seed(4)
    return EntityPointerActorCritic(
        TEACHER_V1,
        config=PointerModelConfig(
            global_hidden=12,
            body_hidden=12,
            attention_hidden=24,
            attention_heads=4,
            attention_layers=1,
            feedforward_hidden=48,
            recurrent_hidden=20,
        ),
    ).eval()


def _actor_record(*, body_id: int = 999) -> dict[str, object]:
    return {
        "global": {
            "level": 4,
            "level_confidence": 1.0,
            "gauge": 500,
            "gauge_max": 1000,
            "gauge_confidence": 1.0,
        },
        "tracks": [
            {
                "id": body_id,
                "kind": "piece",
                "shape": "box",
                "color": 2,
                "lifecycle": "confirmed",
                "effect_x": 180.0,
                "effect_y": 220.0,
                "vx_display_per_second": 20.0,
                "vy_display_per_second": -30.0,
                "size": 24.0,
                "confidence": 0.95,
            }
        ],
    }


def _actor_model() -> EntityPointerActorCritic:
    torch.manual_seed(7)
    return EntityPointerActorCritic(
        ACTOR_VISION_V1,
        config=PointerModelConfig(
            global_hidden=12,
            body_hidden=12,
            attention_hidden=24,
            attention_heads=4,
            attention_layers=1,
            feedforward_hidden=48,
            recurrent_hidden=20,
        ),
    ).eval()


class RecurrentPointerPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_public_id_binding_and_greedy_pointer_lowering(self) -> None:
        observation = _observation()
        encoded = TeacherStateEncoder().encode([observation])
        ids = encoded_body_ids(encoded, observation)
        self.assertEqual({value for value in ids if value is not None}, {41, 73})

        model = _model()
        with torch.no_grad():
            model.kind_head.weight.zero_()
            model.kind_head.bias.copy_(torch.tensor([-8.0, -8.0, 8.0]))
            model.template_head.weight.zero_()
            model.template_head.bias.zero_()
            model.template_head.bias[11] = 8.0
        policy = RecurrentPointerPolicy(model)
        policy.reset(11)
        prediction = policy.predict(observation)
        self.assertEqual(prediction.decision.kind, 1)
        self.assertEqual(prediction.decision.target_body_id, 73)
        self.assertEqual(prediction.decision.template_index, 11)
        self.assertEqual(
            target_index_for_decision(
                prediction.encoded, observation, prediction.decision
            ),
            prediction.target_index,
        )
        self.assertGreater(prediction.confidence, 0.5)

    def test_wait_branch_and_episode_reset(self) -> None:
        model = _model()
        with torch.no_grad():
            model.kind_head.weight.zero_()
            model.kind_head.bias.copy_(torch.tensor([8.0, -8.0, -8.0]))
            model.wait_head.weight.zero_()
            model.wait_head.bias.zero_()
            model.wait_head.bias[3] = 8.0
        policy = RecurrentPointerPolicy(model)
        observation = _observation()
        observation["bodies"][1]["color"] = 4  # type: ignore[index]
        first = policy.predict(observation)
        self.assertEqual(first.decision.kind, 0)
        self.assertEqual(first.decision.wait_ticks, 8)
        self.assertIsNotNone(policy._state)
        policy.reset(19)
        self.assertIsNone(policy._state)


class RecurrentActorPointerPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)

    def test_actor_shot_uses_geometry_not_id_and_lowers_press_release(self) -> None:
        model = _actor_model()
        with torch.no_grad():
            model.kind_head.weight.zero_()
            model.kind_head.bias.copy_(torch.tensor([-8.0, -8.0, 8.0]))
            model.template_head.weight.zero_()
            model.template_head.bias.zero_()
            model.template_head.bias[11] = 8.0
        policy = RecurrentActorPointerPolicy(model)
        first = policy.predict(_actor_record(body_id=17))
        policy.reset(0)
        second = policy.predict(_actor_record(body_id=4_000_000_000))

        self.assertEqual(first.encoded.schema.sha256, ACTOR_VISION_V1.sha256)
        self.assertNotIn("id_scaled", first.encoded.schema.body_features)
        torch.testing.assert_close(
            torch.from_numpy(first.encoded.body_features),
            torch.from_numpy(second.encoded.body_features),
        )
        self.assertEqual(first.kind_index, int(SemanticActionKind.FIRE_STRONG))
        self.assertEqual(first.target_index, 0)
        self.assertEqual(first.template_index, 11)
        self.assertEqual(first.primitive_actions, second.primitive_actions)
        self.assertEqual(len(first.primitive_actions), 2)
        press, release = first.primitive_actions
        self.assertEqual(press.kind, ActionKind.STRONG_SHOT)
        self.assertAlmostEqual(press.cursor_x, 186.0)
        self.assertAlmostEqual(press.cursor_y, 250.0)
        self.assertEqual(release.kind, ActionKind.WAIT)
        self.assertEqual(release.wait_ticks, 1)

    def test_actor_wait_lowers_to_one_legal_primitive(self) -> None:
        model = _actor_model()
        with torch.no_grad():
            model.kind_head.weight.zero_()
            model.kind_head.bias.copy_(torch.tensor([8.0, -8.0, -8.0]))
            model.wait_head.weight.zero_()
            model.wait_head.bias.zero_()
            model.wait_head.bias[3] = 8.0
        policy = RecurrentActorPointerPolicy(model)
        decision = policy.predict(_actor_record())
        self.assertEqual(decision.semantic_action.kind, SemanticActionKind.WAIT)
        self.assertEqual(decision.primitive_actions, policy.act(_actor_record()))
        self.assertEqual(len(decision.primitive_actions), 1)
        self.assertEqual(decision.primitive_actions[0].kind, ActionKind.WAIT)
        self.assertEqual(decision.primitive_actions[0].wait_ticks, 8)

    def test_actor_policy_fails_closed_on_identity_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "actor-vision-v1"):
            RecurrentActorPointerPolicy(_model())
        with self.assertRaisesRegex(ValueError, "pointer action identities"):
            RecurrentActorPointerPolicy(
                _actor_model(),
                pointer_spec=PointerActionSpec(wait_choices=(1, 2)),
            )
        with self.assertRaisesRegex(ValueError, "primitive action identities"):
            RecurrentActorPointerPolicy(
                _actor_model(),
                action_spec=ActionSpec(timing_status="measured"),
            )
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            RecurrentActorPointerPolicy(
                _actor_model(),
                artifact_sha256="A" * 64,
            )

    def test_actor_checkpoint_loader_validates_file_and_schema_identity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="irisu-actor-policy-") as directory:
            actor_path = Path(directory) / "actor.pt"
            actor_digest = save_pointer_checkpoint(actor_path, _actor_model())
            policy = load_actor_pointer_policy(
                actor_path, expected_sha256=actor_digest
            )
            self.assertEqual(policy.artifact_sha256, actor_digest)
            self.assertEqual(policy.schema_sha256, ACTOR_VISION_V1.sha256)
            self.assertEqual(
                policy.action_schema_sha256, policy.model.action_spec.sha256
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_actor_pointer_policy(
                    actor_path, expected_sha256="0" * 64
                )
            with self.assertRaisesRegex(ValueError, "teacher policy loader"):
                load_teacher_pointer_policy(
                    actor_path, expected_sha256=actor_digest
                )

            teacher_path = Path(directory) / "teacher.pt"
            teacher_digest = save_pointer_checkpoint(teacher_path, _model())
            with self.assertRaisesRegex(ValueError, "actor policy loader"):
                load_actor_pointer_policy(
                    teacher_path, expected_sha256=teacher_digest
                )

    def test_checkpoint_binds_relative_action_coordinate_frame(self) -> None:
        spec = PointerActionSpec(y_radius_offsets=(1.25, 1.75))
        model = EntityPointerActorCritic(
            ACTOR_VISION_V1,
            pointer_spec=spec,
            config=PointerModelConfig(
                global_hidden=12,
                body_hidden=12,
                attention_hidden=24,
                attention_heads=4,
                attention_layers=1,
                feedforward_hidden=48,
                recurrent_hidden=20,
            ),
        ).eval()
        with tempfile.TemporaryDirectory(prefix="irisu-relative-checkpoint-") as directory:
            path = Path(directory) / "relative.pt"
            digest = save_pointer_checkpoint(path, model)
            loaded = load_pointer_checkpoint(path, expected_sha256=digest)
            self.assertEqual(loaded.model.pointer_spec, spec)
            self.assertEqual(
                loaded.model.pointer_spec.manifest()["coordinate_frame"],
                "selected_effect_center_and_half_extents-v1",
            )

            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["pointer_spec"]["coordinate_frame"] = "absolute-client-v0"
            tampered_path = Path(directory) / "tampered-frame.pt"
            torch.save(payload, tampered_path)
            with self.assertRaisesRegex(
                ValueError, "pointer action identity differs"
            ):
                load_pointer_checkpoint(tampered_path)

            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["pointer_spec"]["launch_y_norms"] = [350 / 480]
            del payload["pointer_spec"]["y_radius_offsets"]
            legacy_path = Path(directory) / "legacy-semantics.pt"
            torch.save(payload, legacy_path)
            with self.assertRaisesRegex(
                ValueError, "model configuration is invalid"
            ):
                load_pointer_checkpoint(legacy_path)


if __name__ == "__main__":
    unittest.main()
