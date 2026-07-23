from __future__ import annotations

from dataclasses import replace
import hashlib
import tempfile
import unittest

import torch

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
from irisu_rl.r3b_runner import verify_exact_resume_continuation
from irisu_rl.r3b_experiments import TrainingCheckpointArtifact
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


def build_tail_session() -> R3ATrainingSession:
    torch.manual_seed(101)
    base = curriculum()
    library = SnapshotLibrary(
        tuple(replace(recipe, config_hash=99) for recipe in base.library.recipes)
    )
    schedule = RewardSchedule(
        "tail-resume-v1",
        (RewardKnot(0, 500_000), RewardKnot(1, 500_000), RewardKnot(2, 0)),
    )
    spec = replace(
        base,
        curriculum_id="tail-resume-v1",
        library=library,
        stages=(
            replace(base.stages[0], reward_schedule=schedule, max_updates=402),
            base.stages[1],
        ),
    )
    coordinator = CurriculumCoordinator(spec, 4, learner_seed=11)
    composer = RewardComposer(shaping_spec=LinearGaugePotential())
    task = CurriculumTaskContract(coordinator, composer, capture_events=False)
    model = RecurrentActorCritic(
        TEACHER_V1,
        config=RecurrentModelConfig(8, 8, 12, 12, 1, critic_condition_features=1),
    )
    collector = RecurrentCollector(
        model,
        MacroVectorAdapter(AllHeldTruncatingVector(), encoder=TeacherStateEncoder()),
        task,
        config=CollectorConfig(max_decisions=1),
        policy_sampler_seed=7,
    )
    trainer = PPOTrainer(
        model,
        config=PPOConfig(epochs=1, lane_minibatch_size=4, target_kl=1.0),
        total_updates=402,
        sampler_seed=13,
    )
    tail = ScoreOnlyTailController(
        2,
        reward_scale=composer.reward_scale,
        reward_sha256=composer.sha256,
    )
    return R3ATrainingSession(
        collector,
        trainer,
        numpy_seed=17,
        tail_controller=tail,
    )


class TailSessionIntegrationTests(unittest.TestCase):
    def test_resume_is_exact_mid_sweep_drain_and_score_only(self) -> None:
        identity = {
            "test": "r3b-tail-boundary-resume",
            "trial_manifest_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = build_tail_session()
            source.initialize()
            source.run_update()
            closed_restored_environments: list[str] = []

            def assert_next_update_exact(generation: str) -> None:
                destination = source.save(directory, generation, identity=identity)
                manifest_sha256 = hashlib.sha256(
                    (destination / "manifest.json").read_bytes()
                ).hexdigest()
                checkpoint = TrainingCheckpointArtifact(
                    11,
                    source.trainer.schedule.completed_updates,
                    source.collector.simulated_ticks,
                    source.collector.simulated_ticks,
                    "b" * 64,
                    "c" * 64,
                    "a" * 64,
                    "d" * 64,
                    manifest_sha256,
                    source.policy_sha256,
                    "e" * 64,
                )

                def restored_factory() -> R3ATrainingSession:
                    restored = build_tail_session()
                    restored.collector.adapter.env.close = lambda: (
                        closed_restored_environments.append(generation)
                    )
                    return restored

                artifact = verify_exact_resume_continuation(
                    trial_manifest_sha256="a" * 64,
                    checkpoint=checkpoint,
                    checkpoint_directory=directory,
                    generation=generation,
                    checkpoint_identity=identity,
                    source=source,
                    restored_factory=restored_factory,
                )
                self.assertEqual(
                    artifact.source_next_update_sha256,
                    artifact.restored_next_update_sha256,
                )

            # One of two shaped updates has completed: the next update is still sweep.
            assert_next_update_exact("mid-sweep")
            # The shaped sweep is complete: the next collection is the no-update drain.
            assert_next_update_exact("sweep-boundary")
            # One drain completed and all lanes reset: the next update starts score-only.
            assert_next_update_exact("mid-drain")
            # One score-only update completed: the following optimizer step is exact.
            assert_next_update_exact("mid-score-only")
            self.assertEqual(
                closed_restored_environments,
                ["mid-sweep", "sweep-boundary", "mid-drain", "mid-score-only"],
            )

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
            stages=(
                replace(base.stages[0], reward_schedule=schedule, max_updates=401),
                base.stages[1],
            ),
        )
        coordinator = CurriculumCoordinator(spec, 4, learner_seed=11)
        composer = RewardComposer(shaping_spec=LinearGaugePotential())
        task = CurriculumTaskContract(coordinator, composer, capture_events=False)
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
        tail = ScoreOnlyTailController(
            1,
            reward_scale=composer.reward_scale,
            reward_sha256=composer.sha256,
        )
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
        for _ in range(399):
            update = session.run_update()
            self.assertIsNotNone(update.optimizer)
        self.assertEqual(tail.phase, "complete")
        self.assertEqual(tail.score_only_updates, 400)
        self.assertEqual(trainer.schedule.completed_updates, 401)
        session.assert_evidence_ready()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            session.run_update()


if __name__ == "__main__":
    unittest.main()
