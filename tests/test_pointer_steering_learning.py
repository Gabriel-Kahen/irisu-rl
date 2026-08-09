from __future__ import annotations

import hashlib
import math
import unittest

import torch

from irisu_pointer.action import PointerActionSpec
from irisu_pointer.policy import encoded_body_ids
from irisu_pointer.steering import ClosedLoopSteeringExpert
from irisu_pointer.steering import SteeringExpertConfig
from irisu_pointer.steering import SteeringIntent
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
    SteeringDataset,
    SteeringModelConfig,
    steering_example_from_decision,
    steering_imitation_loss,
    train_goal_conditioned_steering,
)
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.schema import TEACHER_V1


_PROVENANCE = hashlib.sha256(b"steering-test").hexdigest()


def _body(identifier: int, color: int, x: float, y: float) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": "dynamic_fresh",
        "color": color,
        "x": x,
        "y": y,
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


def _observation(tick: int, offset: float = 0.0) -> dict[str, object]:
    return {
        "tick": tick,
        "score": 0,
        "gauge": 1000,
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
        "bodies": (
            _body(1, 0, 160.0 + offset, 210.0),
            _body(2, 0, 250.0 + offset, 190.0),
            _body(3, 1, 400.0 - offset, 220.0),
            _body(4, 1, 500.0 - offset, 180.0),
        ),
    }


class SteeringLearningTests(unittest.TestCase):
    def examples(self):
        expert = ClosedLoopSteeringExpert()
        examples = []
        for index in range(12):
            expert.reset(index)
            observation = _observation(index, float(index % 3) * 5.0)
            decision = expert.predict(observation)
            example = steering_example_from_decision(
                observation,
                decision,
                episode_identity=f"episode-{index}",
                provenance_sha256=_PROVENANCE,
            )
            self.assertIsNotNone(example)
            examples.append(example)
        return examples

    def mixed_examples(self):
        examples = []
        for index in range(8):
            expert = ClosedLoopSteeringExpert()
            expert.reset(index)
            shot_observation = _observation(index * 32, float(index % 3) * 5.0)
            shot = steering_example_from_decision(
                shot_observation,
                expert.predict(shot_observation),
                episode_identity=f"mixed-shot-{index}",
                provenance_sha256=_PROVENANCE,
            )
            self.assertIsNotNone(shot)
            examples.append(shot)
            wait_observation = _observation(
                index * 32 + 1, float(index % 3) * 5.0
            )
            bodies = list(wait_observation["bodies"])
            bodies.append(
                {
                    **_body(100 + index, -1, 160.0, 260.0),
                    "kind": "projectile",
                    "shape": "triangle",
                    "lifecycle": "unknown",
                }
            )
            wait_observation["bodies"] = tuple(bodies)
            wait = steering_example_from_decision(
                wait_observation,
                expert.predict(wait_observation),
                episode_identity=f"mixed-wait-{index}",
                provenance_sha256=_PROVENANCE,
            )
            self.assertIsNotNone(wait)
            examples.append(wait)
        return examples

    def test_pair_mask_rejects_cross_color_pairs(self) -> None:
        dataset = SteeringDataset(self.examples()[:1])
        batch = dataset.as_tensors()
        model = GoalConditionedSteeringModel(
            TEACHER_V1,
            config=SteeringModelConfig(body_hidden=16, global_hidden=8, pair_hidden=24),
        )
        output = model(
            batch.global_features, batch.body_features, batch.body_mask
        )
        source = int(batch.source_index[0])
        destination = int(batch.destination_index[0])
        self.assertTrue(bool(output.legal_pair_mask[0, source, destination]))
        color_columns = [
            TEACHER_V1.body_features.index(f"color_{index}") for index in range(6)
        ]
        labels = batch.body_features[0, :, color_columns].argmax(-1)
        cross = next(
            (i, j)
            for i in range(batch.body_features.shape[1])
            for j in range(batch.body_features.shape[1])
            if i != j and labels[i] != labels[j]
        )
        self.assertFalse(bool(output.legal_pair_mask[0, cross[0], cross[1]]))

    def test_pair_mask_rejects_grouped_confirmed_and_rotten_sources(self) -> None:
        model = GoalConditionedSteeringModel(
            TEACHER_V1,
            config=SteeringModelConfig(body_hidden=16, global_hidden=8, pair_hidden=24),
        )
        for lifecycle, chain_id in (
            ("dynamic_fresh", 4),
            ("confirmed", 0),
            ("rotten", 0),
        ):
            observation = _observation(0)
            bodies = [dict(value) for value in observation["bodies"]]
            bodies[0]["lifecycle"] = lifecycle
            bodies[0]["chain_id"] = chain_id
            observation["bodies"] = tuple(bodies)
            encoded = TeacherStateEncoder().encode([observation])
            identifiers = encoded_body_ids(encoded, observation)
            width = max(index for index, value in enumerate(identifiers) if value) + 1
            output = model(
                torch.from_numpy(encoded.global_features),
                torch.from_numpy(encoded.body_features[:, :width]),
                torch.from_numpy(encoded.body_mask[:, :width]),
            )
            source = identifiers.index(1)
            self.assertFalse(bool(output.legal_pair_mask[0, source].any()))

    def test_pair_mask_allows_rotten_destination_but_not_rotten_source(
        self,
    ) -> None:
        observation = _observation(0)
        bodies = [dict(value) for value in observation["bodies"]]
        bodies[1]["lifecycle"] = "rotten"
        observation["bodies"] = tuple(bodies)
        encoded = TeacherStateEncoder().encode([observation])
        identifiers = encoded_body_ids(encoded, observation)
        width = max(index for index, value in enumerate(identifiers) if value) + 1
        model = GoalConditionedSteeringModel(TEACHER_V1)
        output = model(
            torch.from_numpy(encoded.global_features),
            torch.from_numpy(encoded.body_features[:, :width]),
            torch.from_numpy(encoded.body_mask[:, :width]),
        )
        fresh = identifiers.index(1)
        rotten = identifiers.index(2)
        self.assertTrue(bool(output.legal_pair_mask[0, fresh, rotten]))
        self.assertFalse(bool(output.legal_pair_mask[0, rotten].any()))

        decision = ClosedLoopSteeringExpert().predict(observation)
        example = steering_example_from_decision(
            observation,
            decision,
            episode_identity="fresh-to-rotten",
            provenance_sha256=_PROVENANCE,
            require_representable_template=False,
        )
        self.assertIsNotNone(example)
        assert example is not None
        self.assertEqual(
            example.intent_index,
            tuple(SteeringIntent).index(SteeringIntent.MATCH_ROTTEN),
        )
        batch = SteeringDataset((example,)).as_tensors()
        loss = steering_imitation_loss(
            model(
                batch.global_features,
                batch.body_features,
                batch.body_mask,
            ),
            batch,
        )
        self.assertTrue(bool(torch.isfinite(loss.total)))

    def test_pair_relations_expose_rotten_anchor_size_and_source_rot(self) -> None:
        observation = _observation(0)
        bodies = [dict(value) for value in observation["bodies"]]
        bodies[0]["rot_timer"] = 5
        bodies[1].update(lifecycle="rotten", chain_id=7)
        bodies[2].update(color=0, lifecycle="confirmed", chain_id=7)
        bodies[3].update(
            kind="projectile",
            shape="triangle",
            color=-1,
            lifecycle="unknown",
            chain_id=7,
        )
        observation["bodies"] = tuple(bodies)
        encoded = TeacherStateEncoder().encode([observation])
        identifiers = encoded_body_ids(encoded, observation)
        width = max(index for index, value in enumerate(identifiers) if value) + 1
        model = GoalConditionedSteeringModel(TEACHER_V1)
        features = torch.from_numpy(encoded.body_features[:, :width])
        mask = torch.from_numpy(encoded.body_mask[:, :width])
        relation = model._public_pair_relations(features, mask)
        source = identifiers.index(1)
        rotten = identifiers.index(2)
        ungrouped = identifiers.index(1)
        self.assertEqual(float(relation[0, source, rotten, 5]), 1.0)
        self.assertAlmostEqual(
            float(relation[0, source, rotten, 6]),
            math.log1p(2) / math.log1p(TEACHER_V1.capacity),
            places=6,
        )
        self.assertEqual(float(relation[0, source, rotten, 7]), 1.0)
        self.assertEqual(float(relation[0, source, rotten, 8]), 1.0)
        self.assertEqual(float(relation[0, source, ungrouped, 5]), 0.0)
        self.assertEqual(float(relation[0, source, ungrouped, 6]), 0.0)

    def test_wait_restraint_is_supervised_and_learnable(self) -> None:
        torch.manual_seed(11)
        dataset = SteeringDataset(self.mixed_examples())
        self.assertEqual(sum(value.is_shot for value in dataset), 8)
        model = GoalConditionedSteeringModel(
            TEACHER_V1,
            config=SteeringModelConfig(body_hidden=24, global_hidden=12, pair_hidden=32),
        )
        report = train_goal_conditioned_steering(
            model,
            dataset,
            steps=160,
            batch_size=16,
            learning_rate=2e-3,
            seed=9,
        )
        self.assertEqual(report.act_accuracy, 1.0)
        self.assertEqual(report.wait_accuracy, 1.0)

    def test_nonrepresentable_expert_shot_is_not_quantized_into_a_label(
        self,
    ) -> None:
        observation = _observation(0)
        decision = ClosedLoopSteeringExpert(
            config=SteeringExpertConfig(impact_side_sizes=0.6)
        ).predict(observation)
        self.assertIsNone(
            steering_example_from_decision(
                observation,
                decision,
                episode_identity="nonrepresentable",
                provenance_sha256=_PROVENANCE,
            )
        )
        self.assertIsNotNone(
            steering_example_from_decision(
                observation,
                decision,
                episode_identity="pair-only-nonrepresentable",
                provenance_sha256=_PROVENANCE,
                require_representable_template=False,
            )
        )

    def test_small_velocity_lead_is_bounded_into_the_pointer_vocabulary(
        self,
    ) -> None:
        observation = _observation(0)
        bodies = [dict(value) for value in observation["bodies"]]
        bodies[0]["lifecycle"] = "scripted_falling"
        bodies[0]["vx"] = 1.0
        observation["bodies"] = tuple(bodies)
        decision = ClosedLoopSteeringExpert().predict(observation)
        self.assertIsNotNone(
            steering_example_from_decision(
                observation,
                decision,
                episode_identity="bounded-velocity-lead",
                provenance_sha256=_PROVENANCE,
            )
        )

    def test_dataset_identity_binds_encoded_observations(self) -> None:
        first, second = self.examples()[:2]
        self.assertNotEqual(first.sha256, second.sha256)
        self.assertNotEqual(
            SteeringDataset((first,)).sha256,
            SteeringDataset((second,)).sha256,
        )

    def test_supervised_pair_model_fits_toy_expert(self) -> None:
        torch.manual_seed(3)
        dataset = SteeringDataset(self.examples())
        model = GoalConditionedSteeringModel(
            TEACHER_V1,
            pointer_spec=PointerActionSpec(),
            config=SteeringModelConfig(body_hidden=24, global_hidden=12, pair_hidden=32),
        )
        report = train_goal_conditioned_steering(
            model,
            dataset,
            steps=120,
            batch_size=12,
            learning_rate=2e-3,
            seed=7,
        )
        self.assertLess(report.final_loss, report.initial_loss * 0.25)
        self.assertEqual(report.pair_accuracy, 1.0)
        self.assertEqual(report.kind_accuracy, 1.0)
        self.assertEqual(report.template_accuracy, 1.0)

    def test_learned_policy_observes_cooldown(self) -> None:
        torch.manual_seed(4)
        dataset = SteeringDataset(self.examples())
        model = GoalConditionedSteeringModel(
            TEACHER_V1,
            config=SteeringModelConfig(body_hidden=24, global_hidden=12, pair_hidden=32),
        )
        train_goal_conditioned_steering(
            model, dataset, steps=80, batch_size=12, learning_rate=2e-3, seed=1
        )
        policy = GoalConditionedSteeringPolicy(model, cooldown_ticks=16)
        policy.reset(0)
        observation = _observation(0)
        first = policy.predict(observation)
        self.assertTrue(first.is_shot)
        bodies = {
            int(value["id"]): value for value in observation["bodies"]
        }
        source = bodies[first.source_body_id]
        self.assertAlmostEqual(
            first.action.x_norm,
            (
                float(source["x"])
                + first.impact_x_sizes * float(source["size"])
            )
            / 640.0,
        )
        self.assertAlmostEqual(
            first.action.y_norm,
            (
                float(source["y"])
                + first.impact_y_sizes * float(source["size"])
            )
            / 480.0,
        )
        second = policy.predict(_observation(1))
        self.assertFalse(second.is_shot)
        self.assertIn("previous learned correction", second.reason)
        safe_boundary = policy.predict(_observation(16))
        if safe_boundary.is_shot:
            self.assertNotEqual(
                (
                    safe_boundary.source_body_id,
                    safe_boundary.destination_body_id,
                ),
                (first.source_body_id, first.destination_body_id),
            )

    def test_model_ablates_body_and_chain_id_magnitudes(self) -> None:
        model = GoalConditionedSteeringModel(TEACHER_V1)
        self.assertEqual(
            model.manifest()["architecture"],
            "goal-conditioned-directed-pair-steering-v4",
        )
        self.assertIn("id_scaled", model.manifest()["input_ablations"])
        self.assertIn("chain_id_scaled", model.manifest()["input_ablations"])
        encoded = TeacherStateEncoder().encode([_observation(0)])
        batch = SteeringDataset(self.examples()[:1]).as_tensors()
        id_index = TEACHER_V1.body_features.index("id_scaled")
        chain_index = TEACHER_V1.body_features.index("chain_id_scaled")
        first_features = encoded.body_features.copy()
        first_features[0, 0, id_index] = 0.11
        first_features[0, 0, chain_index] = 0.2
        second_features = encoded.body_features.copy()
        second_features[0, 0, id_index] = 0.99
        second_features[0, 0, chain_index] = 0.8
        with torch.no_grad():
            first = model(
                torch.from_numpy(encoded.global_features),
                torch.from_numpy(
                    first_features[:, : batch.body_features.shape[1]]
                ),
                torch.from_numpy(encoded.body_mask[:, : batch.body_features.shape[1]]),
            )
            second = model(
                torch.from_numpy(encoded.global_features),
                torch.from_numpy(
                    second_features[:, : batch.body_features.shape[1]]
                ),
                torch.from_numpy(encoded.body_mask[:, : batch.body_features.shape[1]]),
            )
        for name in (
            "act_logits",
            "wait_logits",
            "pair_logits",
            "kind_logits",
            "template_logits",
            "intent_logits",
            "legal_pair_mask",
        ):
            self.assertTrue(
                torch.equal(getattr(first, name), getattr(second, name)),
                name,
            )


if __name__ == "__main__":
    unittest.main()
