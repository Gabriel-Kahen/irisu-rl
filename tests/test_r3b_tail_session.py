from __future__ import annotations

from dataclasses import replace
import unittest

from irisu_rl.collector import (
    CollectorConfig,
    CurriculumTaskContract,
    R3ATrainingSession,
    RecurrentCollector,
)
from irisu_rl.curriculum import CurriculumCoordinator, SnapshotLibrary
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.ppo import PPOConfig, PPOTrainer
from irisu_rl.r3b_tail import ScoreOnlyTailController
from irisu_rl.rewards import (
    LinearGaugePotential,
    RewardComposer,
    RewardKnot,
    RewardSchedule,
)
from irisu_rl.schema import TEACHER_V1
from irisu_rl.vector_adapter import MacroVectorAdapter
from tests.test_r3a_collector import AllHeldTruncatingVector
from tests.test_r3a_curriculum import curriculum


class TailSessionIntegrationTests(unittest.TestCase):
    def test_drain_rollout_cannot_advance_optimizer_clock(self) -> None:
        base = curriculum()
        library = SnapshotLibrary(
            tuple(replace(recipe, config_hash=99) for recipe in base.library.recipes)
        )
        schedule = RewardSchedule(
            "tail-session-v1", (RewardKnot(0, 500_000), RewardKnot(1, 0))
        )
        spec = replace(
            base,
            curriculum_id="tail-session-v1",
            library=library,
            stages=(replace(base.stages[0], reward_schedule=schedule), base.stages[1]),
        )
        coordinator = CurriculumCoordinator(spec, 4, learner_seed=11)
        task = CurriculumTaskContract(
            coordinator,
            RewardComposer(shaping_spec=LinearGaugePotential()),
            capture_events=False,
        )
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1, critic_condition_features=1),
        )
        collector = RecurrentCollector(
            model,
            MacroVectorAdapter(
                AllHeldTruncatingVector(), encoder=TeacherStateEncoder()
            ),
            task,
            config=CollectorConfig(max_decisions=1),
            policy_sampler_seed=7,
        )
        trainer = PPOTrainer(
            model,
            config=PPOConfig(epochs=1, lane_minibatch_size=4, target_kl=1.0),
            total_updates=401,
            sampler_seed=13,
        )
        tail = ScoreOnlyTailController(1)
        session = R3ATrainingSession(
            collector,
            trainer,
            numpy_seed=17,
            tail_controller=tail,
        )
        session.initialize()

        shaped = session.run_update()
        self.assertIsNotNone(shaped.optimizer)
        self.assertEqual(trainer.schedule.completed_updates, 1)

        drain = session.run_update()
        self.assertIsNone(drain.optimizer)
        self.assertEqual(drain.skipped_reason, "score-only tail episode drain")
        self.assertEqual(trainer.schedule.completed_updates, 1)
        self.assertEqual(tail.drain_collections, 1)
        self.assertEqual(coordinator.shaping_weights_ppm().tolist(), [0, 0, 0, 0])

        score_only = session.run_update()
        self.assertIsNotNone(score_only.optimizer)
        self.assertEqual(trainer.schedule.completed_updates, 2)
        self.assertEqual(tail.score_only_updates, 1)
        self.assertTrue(
            all(
                weight == 0
                for decision in score_only.collection.decisions
                for weight in decision.shaping_weight_ppm
            )
        )


if __name__ == "__main__":
    unittest.main()
