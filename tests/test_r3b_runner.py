from __future__ import annotations

import unittest
from pathlib import Path

from irisu_rl.collector import CollectorConfig
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.ppo import PPOConfig
from irisu_rl.r3b_experiments import load_plan
from irisu_rl.r3b_runner import R3BRunBuilder
from irisu_rl.schema import TEACHER_V1
from tests.test_r3b_snapshot_initializer import (
    FakeSnapshotVector,
    _RUNTIME_SHA256,
    _fixture,
)
from tests.test_r3a_curriculum import curriculum as multistage_curriculum
from irisu_rl.curriculum import SnapshotBlobStore


ROOT = Path(__file__).resolve().parents[1]


class R3BRunBuilderTests(unittest.TestCase):
    def test_arms_share_initial_model_assignments_and_seed_plan(self) -> None:
        plan = load_plan(ROOT / "configs/rl/experiments/r3b-completion-v1.toml")
        curriculum, blobs = _fixture()
        store = SnapshotBlobStore(curriculum.library, blobs)

        def model_factory():
            return RecurrentActorCritic(
                TEACHER_V1,
                config=RecurrentModelConfig(
                    8, 8, 12, 12, 1, critic_condition_features=1
                ),
            )

        builder = R3BRunBuilder(
            plan,
            curriculum,
            store,
            runtime_identity_sha256=_RUNTIME_SHA256,
            lanes=2,
            collector_config=CollectorConfig(
                max_decisions=1,
                target_simulated_ticks=plan.ticks_per_update,
            ),
            ppo_config=PPOConfig(
                learning_rate=1e-4,
                final_learning_rate_fraction=plan.final_learning_rate_fraction,
                epochs=1,
                lane_minibatch_size=2,
            ),
            model_factory=model_factory,
            environment_factory=FakeSnapshotVector,
        )
        control_arm = plan.arms[0]
        shaped_arm = next(arm for arm in plan.arms if arm.alpha_weight_ppm == 100_000)
        with (
            builder.build(control_arm, 1103) as control,
            builder.build(shaped_arm, 1103) as shaped,
        ):
            self.assertEqual(
                control.manifest.initial_model_sha256,
                shaped.manifest.initial_model_sha256,
            )
            self.assertEqual(
                control.manifest.assignment_sha256,
                shaped.manifest.assignment_sha256,
            )
            self.assertEqual(
                control.manifest.seed_plan_sha256,
                shaped.manifest.seed_plan_sha256,
            )
            self.assertEqual(
                control.manifest.reward_sha256, shaped.manifest.reward_sha256
            )
            self.assertNotEqual(
                control.manifest.curriculum_sha256,
                shaped.manifest.curriculum_sha256,
            )
            self.assertFalse(control.manifest.deployable)
            manifest_sha256 = control.manifest.sha256
            with self.assertRaises(TypeError):
                control.manifest.collector["max_decisions"] = 99  # type: ignore[index]
            self.assertEqual(control.manifest.sha256, manifest_sha256)
            control.session.initialize()
            shaped.session.initialize()
            self.assertEqual(
                control.session.task.coordinator.lane_snapshot_id,
                shaped.session.task.coordinator.lane_snapshot_id,
            )

    def test_adaptive_multi_stage_sweep_is_rejected(self) -> None:
        plan = load_plan(ROOT / "configs/rl/experiments/r3b-completion-v1.toml")
        multi_stage = multistage_curriculum()
        with self.assertRaises(ValueError):
            R3BRunBuilder(
                plan,
                multi_stage,
                None,  # type: ignore[arg-type]
                runtime_identity_sha256=_RUNTIME_SHA256,
                lanes=2,
                collector_config=CollectorConfig(
                    target_simulated_ticks=plan.ticks_per_update
                ),
                ppo_config=PPOConfig(
                    final_learning_rate_fraction=plan.final_learning_rate_fraction
                ),
                model_factory=lambda: None,  # type: ignore[arg-type]
                environment_factory=FakeSnapshotVector,
            )


if __name__ == "__main__":
    unittest.main()
