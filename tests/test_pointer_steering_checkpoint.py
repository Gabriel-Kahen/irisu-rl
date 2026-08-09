from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from irisu_pointer.steering_checkpoint import (
    load_goal_conditioned_steering_policy,
    load_steering_checkpoint,
    save_steering_checkpoint,
)
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    SteeringModelConfig,
)
from irisu_rl.schema import TEACHER_V1


class SteeringCheckpointTests(unittest.TestCase):
    def model(self) -> GoalConditionedSteeringModel:
        torch.manual_seed(7)
        return GoalConditionedSteeringModel(
            TEACHER_V1,
            config=SteeringModelConfig(
                body_hidden=16, global_hidden=8, pair_hidden=24
            ),
        )

    def test_round_trip_is_identity_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steering.pt"
            digest = save_steering_checkpoint(
                path,
                self.model(),
                metadata={"source_revision": "deadbee", "development_only": True},
            )
            loaded = load_steering_checkpoint(path, expected_sha256=digest)
            self.assertEqual(loaded.sha256, digest)
            self.assertEqual(loaded.metadata["source_revision"], "deadbee")
            policy = load_goal_conditioned_steering_policy(
                path, expected_sha256=digest, act_logit_bias=1.0
            )
            self.assertEqual(policy.artifact_sha256, digest)
            self.assertEqual(policy.act_logit_bias, 1.0)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_steering_checkpoint(path, expected_sha256="0" * 64)

    def test_rejects_legacy_absolute_y_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steering.pt"
            save_steering_checkpoint(path, self.model())
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["pointer_spec"]["launch_y_norms"] = payload["pointer_spec"].pop(
                "y_radius_offsets"
            )
            torch.save(payload, path)
            with self.assertRaisesRegex(
                ValueError, "configuration|action identity"
            ):
                load_steering_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
