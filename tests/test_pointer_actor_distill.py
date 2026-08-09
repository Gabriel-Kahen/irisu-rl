from __future__ import annotations

import unittest

import torch

from irisu_pointer.actor_distill import (
    ActorAlignmentError,
    ActorDistillConfig,
    ActorTeacherStep,
    build_actor_sequence_batch,
    build_actor_sequence_episode,
    perfect_actor_record,
    sanitize_actor_record,
)
from irisu_pointer.experts import PointerExpertDecision
from irisu_pointer.model import EntityPointerActorCritic, PointerModelConfig
from irisu_pointer.sequence import PointerSequenceConfig, PointerSequenceTrainer
from irisu_rl.schema import ACTOR_VISION_V1


_PRIVILEGED_TEST_KEYS = {
    "id",
    "chain_id",
    "projectile_hits",
    "age_ticks",
    "remaining_lifetime",
    "rot_timer",
    "score",
    "highest_chain",
}


def _track(
    *,
    x: float = 180.0,
    y: float = 220.0,
    color: int = 2,
    confidence: float = 0.95,
) -> dict[str, object]:
    return {
        "kind": "piece",
        "shape": "box",
        "color": color,
        "lifecycle": "confirmed",
        "effect_x": x,
        "effect_y": y,
        "vx_display_per_second": 20.0,
        "vy_display_per_second": -30.0,
        "size": 24.0,
        "confidence": confidence,
    }


def _teacher_body(
    body_id: int,
    *,
    x: float = 180.0,
    y: float = 220.0,
    color: int = 2,
) -> dict[str, object]:
    return {
        "id": body_id,
        "kind": "piece",
        "shape": "box",
        "color": color,
        "lifecycle": "confirmed",
        "effect_x": x,
        "effect_y": y,
        "size": 24.0,
    }


def _step(
    decision: PointerExpertDecision,
    *,
    body_id: int = 17,
    actor_record: dict[str, object] | None = None,
    teacher_body: dict[str, object] | None = None,
    value: float = 1.0,
) -> ActorTeacherStep:
    return ActorTeacherStep(
        actor_record or {
            "global": {"level": 4, "level_confidence": 1.0},
            "tracks": [_track()],
        },
        {"bodies": [teacher_body or _teacher_body(body_id)]},
        decision,
        value,
    )


def _small_actor_model() -> EntityPointerActorCritic:
    torch.manual_seed(83)
    return EntityPointerActorCritic(
        ACTOR_VISION_V1,
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


class ActorAlignmentTests(unittest.TestCase):
    def test_perfect_detector_bridge_strips_privileged_truth(self) -> None:
        observation = {
            "tick": 50,
            "score": 999,
            "highest_chain": 8,
            "gauge": 500,
            "gauge_max": 1000,
            "level": 4,
            "bodies": [
                {
                    **_teacher_body(17),
                    "vx": 2.0,
                    "vy": -3.0,
                    "angle": 0.5,
                    "angular_velocity": 0.25,
                    "chain_id": 99,
                    "rot_timer": 44,
                }
            ],
        }
        first = perfect_actor_record(observation)
        observation["score"] = -123
        observation["highest_chain"] = 100
        observation["bodies"][0]["id"] = 4_000_000_000  # type: ignore[index]
        observation["bodies"][0]["chain_id"] = 1  # type: ignore[index]
        observation["bodies"][0]["rot_timer"] = 999  # type: ignore[index]
        second = perfect_actor_record(observation)
        self.assertEqual(first, second)
        self.assertFalse(
            _PRIVILEGED_TEST_KEYS
            & set(first["global"])
        )
        self.assertFalse(
            _PRIVILEGED_TEST_KEYS
            & set(first["tracks"][0])
        )

    def test_builds_actor_schema_sequence_and_trains_with_existing_tbptt(self) -> None:
        steps = (
            _step(PointerExpertDecision.wait(1), value=0.0),
            _step(PointerExpertDecision.weak(17, template_index=4), value=0.5),
            _step(PointerExpertDecision.strong(17, template_index=10), value=1.0),
        )
        episode = build_actor_sequence_episode("episode-a", steps)
        self.assertEqual(episode.schema.sha256, ACTOR_VISION_V1.sha256)
        self.assertEqual(episode.actions.kind.tolist(), [0, 1, 2])
        self.assertEqual(episode.actions.target_index.tolist(), [0, 0, 0])
        self.assertTrue(episode.body_mask.all())
        batch = build_actor_sequence_batch({"episode-a": steps})
        trainer = PointerSequenceTrainer(
            _small_actor_model(),
            config=PointerSequenceConfig(
                learning_rate=1e-3,
                burn_in_steps=1,
                tbptt_steps=1,
                entropy_coefficient=0.0,
                seed=5,
            ),
        )
        metrics = trainer.step(batch)
        self.assertEqual(metrics.examples, 2)
        self.assertEqual(metrics.actionable_examples, 2)
        self.assertEqual(metrics.optimizer_steps, 2)
        self.assertTrue(torch.isfinite(torch.tensor(metrics.total_loss)))

    def test_ids_and_privileged_fields_cannot_change_tensors_or_labels(self) -> None:
        baseline = _step(PointerExpertDecision.weak(17, template_index=7))
        privileged_track = {
            **_track(),
            "id": 9999,
            "chain_id": 888,
            "remaining_lifetime": -1,
            "rot_timer": 999,
            "projectile_hits": 42,
            "future_spawns": ["secret"],
        }
        changed_body = {
            **_teacher_body(9001),
            "chain_id": 123,
            "remaining_lifetime": 1,
            "rot_timer": 456,
            "projectile_hits": 7,
        }
        changed = _step(
            PointerExpertDecision.weak(9001, template_index=7),
            body_id=9001,
            actor_record={
                "global": {
                    "level": 4,
                    "level_confidence": 1.0,
                    "score": 999999,
                    "highest_chain": 99,
                    "future_spawns": [1, 2, 3],
                    "rng_state": "secret",
                },
                "tracks": [privileged_track],
                "snapshot": "secret",
            },
            teacher_body=changed_body,
        )
        first = build_actor_sequence_episode("same", (baseline,))
        second = build_actor_sequence_episode("same", (changed,))
        for left, right in (
            (first.global_features, second.global_features),
            (first.body_features, second.body_features),
            (first.body_mask, second.body_mask),
            (first.actions.kind, second.actions.kind),
            (first.actions.target_index, second.actions.target_index),
            (first.actions.template_index, second.actions.template_index),
        ):
            torch.testing.assert_close(left, right)
        forbidden = {
            "id_scaled",
            "chain_id_scaled",
            "remaining_lifetime_signed_log1p",
            "rot_timer_log1p",
        }
        self.assertFalse(forbidden & set(first.schema.body_features))

    def test_alignment_rejects_ambiguous_or_low_confidence_tracks(self) -> None:
        ambiguous_record = {
            "tracks": [_track(), _track()],
        }
        with self.assertRaisesRegex(ActorAlignmentError, "not unique"):
            build_actor_sequence_episode(
                "ambiguous",
                (
                    _step(
                        PointerExpertDecision.weak(17),
                        actor_record=ambiguous_record,
                    ),
                ),
            )
        with self.assertRaisesRegex(ActorAlignmentError, "too low"):
            build_actor_sequence_episode(
                "low-confidence",
                (
                    _step(
                        PointerExpertDecision.weak(17),
                        actor_record={"tracks": [_track(confidence=0.2)]},
                    ),
                ),
            )

    def test_noise_is_deterministic_bounded_and_does_not_rebind_target(self) -> None:
        record = {"tracks": [_track()]}
        config = ActorDistillConfig(
            position_noise_pixels=2.0,
            velocity_noise_pixels_per_second=5.0,
            confidence_noise=0.02,
            seed=77,
        )
        first_record = sanitize_actor_record(
            record,
            config=config,
            episode_identity="noise",
            step_index=0,
        )
        second_record = sanitize_actor_record(
            record,
            config=config,
            episode_identity="noise",
            step_index=0,
        )
        self.assertEqual(first_record, second_record)
        noisy = first_record["tracks"][0]
        original = record["tracks"][0]
        self.assertLessEqual(abs(noisy["effect_x"] - original["effect_x"]), 2.0)
        self.assertLessEqual(abs(noisy["effect_y"] - original["effect_y"]), 2.0)
        self.assertLessEqual(
            abs(
                noisy["vx_display_per_second"]
                - original["vx_display_per_second"]
            ),
            5.0,
        )
        episode = build_actor_sequence_episode(
            "noise",
            (_step(PointerExpertDecision.weak(17), actor_record=record),),
            config=config,
        )
        self.assertEqual(episode.actions.target_index.item(), 0)


if __name__ == "__main__":
    unittest.main()
