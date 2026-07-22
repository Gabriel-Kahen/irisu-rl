from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import hashlib

import torch

from irisu_env import PaddedVectorEnv
from irisu_rl.collector import (
    CollectorConfig,
    CurriculumTaskContract,
    R3ATrainingSession,
    RecurrentCollector,
)
from irisu_rl.checkpoints import load_checkpoint, save_checkpoint
from irisu_rl.actions import ActionSpec, SemanticAction
from irisu_rl.curriculum import (
    CurriculumCoordinator,
    CurriculumSpec,
    SnapshotLibrary,
    SnapshotRecipe,
    StageSpec,
    ValidationEpisodeOutcome,
    ValidationReport,
    ValidationResult,
)
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.ppo import PPOConfig, PPOTrainer
from irisu_rl.rewards import (
    LinearGaugePotential,
    RewardComposer,
    RewardKnot,
    RewardSchedule,
)
from irisu_rl.schema import TEACHER_V1
from irisu_rl.vector_adapter import MacroVectorAdapter


ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT / "build-physics-integration-portable" / "libirisu_clone.so"
EXACT = ROOT / "build-physics-integration-exact-multiworld-2" / "irisu-exact-worker"


def build_session(
    *, exact: bool, construction_seed: int, shaping_weight_ppm: int = 0
) -> tuple[R3ATrainingSession, PaddedVectorEnv]:
    torch.manual_seed(construction_seed)
    vector = PaddedVectorEnv(
        2,
        physics_backend="exact" if exact else "portable",
        worker_path=EXACT if exact else None,
        library_path=None if exact else PORTABLE,
        config={"max_episode_ticks": 30},
    )
    model = RecurrentActorCritic(
        TEACHER_V1,
        config=RecurrentModelConfig(
            8,
            8,
            12,
            12,
            1,
            critic_condition_features=1 if shaping_weight_ppm else 0,
        ),
    )
    adapter = MacroVectorAdapter(vector, encoder=TeacherStateEncoder())
    action_spec = ActionSpec()
    config_hash = int(vector.envs[0].config_hash())

    def recipe(split: str, suffix: str) -> SnapshotRecipe:
        identity = int.from_bytes(hashlib.sha256(suffix.encode()).digest()[:4], "big")
        return SnapshotRecipe(
            f"nominal-{split}-{suffix}",
            "nominal",
            split,
            f"nominal-{split}",
            "nominal-pool",
            "1" * 64,
            config_hash,
            identity,
            action_spec.sha256,
            (action_spec.serialize(SemanticAction.wait(1)).hex(),),
            1,
            0,
            identity,
            hashlib.sha256(suffix.encode()).hexdigest(),
            "2" * 64,
            "resume-fixture-v1",
        )

    train_recipe = recipe("train", "train")
    validation_recipe = recipe("validation", "validation")
    curriculum = CurriculumSpec(
        "resume-fixture-v1",
        SnapshotLibrary((train_recipe, validation_recipe)),
        (
            StageSpec(
                "nominal",
                0,
                "nominal-pool",
                (train_recipe.snapshot_id,),
                (validation_recipe.snapshot_id,),
                (0, 1, 2),
                action_spec.wait_choices,
                1,
                1,
                1,
                1,
                1,
                100,
                RewardSchedule(
                    "gauge-fixed-v1" if shaping_weight_ppm else "score-only-v1",
                    (RewardKnot(0, shaping_weight_ppm),),
                ),
            ),
        ),
        0xEAA1,
    )
    coordinator = CurriculumCoordinator(curriculum, 2, learner_seed=83)
    task = CurriculumTaskContract(
        coordinator,
        RewardComposer(
            reward_scale=100.0,
            shaping_spec=(
                LinearGaugePotential() if shaping_weight_ppm else None
            ),
        ),
        capture_events=False,
    )
    collector = RecurrentCollector(
        model,
        adapter,
        task,
        config=CollectorConfig(max_decisions=3, lambda_tick=0.95),
        policy_sampler_seed=71,
    )
    trainer = PPOTrainer(
        model,
        config=PPOConfig(
            learning_rate=1e-4,
            final_learning_rate_fraction=0.5,
            epochs=1,
            lane_minibatch_size=2,
            target_kl=1.0,
        ),
        total_updates=4,
        sampler_seed=73,
    )
    return R3ATrainingSession(collector, trainer, numpy_seed=79), vector


class R3ASessionResumeMixin:
    exact = False

    def assert_resume(self, *, shaping_weight_ppm: int = 0) -> None:
        identity = {
            "test": "r3a-full-resume",
            "backend": "exact" if self.exact else "portable",
            "shaping_weight_ppm": shaping_weight_ppm,
        }
        with tempfile.TemporaryDirectory() as directory:
            source, source_vector = build_session(
                exact=self.exact,
                construction_seed=101,
                shaping_weight_ppm=shaping_weight_ppm,
            )
            try:
                source.initialize()
                source.run_update()
                source.save(directory, "update-0001", identity=identity)
                expected = source.run_update()
                expected_hashes = source_vector.state_hash()
                expected_parameters = {
                    name: value.detach().cpu().clone()
                    for name, value in source.model.state_dict().items()
                }
                expected_trainer = source.trainer.state_dict()
                expected_collector = source.collector.state_dict()
            finally:
                source_vector.close()

            restored, restored_vector = build_session(
                exact=self.exact,
                construction_seed=999,
                shaping_weight_ppm=shaping_weight_ppm,
            )
            try:
                restored.restore(directory, identity=identity)
                actual = restored.run_update()
                actual_hashes = restored_vector.state_hash()
                self.assertEqual(actual.collection, expected.collection)
                if shaping_weight_ppm:
                    self.assertTrue(
                        any(
                            any(value != 0 for value in row.shaping_rewards)
                            for row in actual.collection.decisions
                        )
                    )
                self.assertEqual(actual.optimizer, expected.optimizer)
                self.assertEqual(actual_hashes, expected_hashes)
                self.assertEqual(
                    restored.collector.state_dict()["completed_updates"],
                    expected_collector["completed_updates"],
                )
                self.assertEqual(
                    restored.collector.state_dict()["decision_rows"],
                    expected_collector["decision_rows"],
                )
                for name, value in restored.model.state_dict().items():
                    torch.testing.assert_close(
                        value.detach().cpu(), expected_parameters[name], rtol=0, atol=0
                    )
                self.assertEqual(
                    restored.trainer.state_dict()["schedule"],
                    expected_trainer["schedule"],
                )
                torch.testing.assert_close(
                    restored.trainer.state_dict()["sampler_state"],
                    expected_trainer["sampler_state"],
                    rtol=0,
                    atol=0,
                )
            finally:
                restored_vector.close()

    def assert_validation_binds_loaded_model(self) -> None:
        session, vector = build_session(exact=self.exact, construction_seed=202)
        try:
            session.initialize()
            request = session.request_validation(evaluator_identity_sha256="e" * 64)
            self.assertEqual(request.policy_sha256, session.policy_sha256)
            stage = request.stages[0]
            seed = request.episode_seed(stage.stage_id, stage.snapshot_ids[0], 0)
            report = ValidationReport(
                request.request_id,
                request.policy_sha256,
                request.evaluator_identity_sha256,
                (
                    ValidationResult(
                        stage.stage_id,
                        1,
                        1,
                        stage.snapshot_ids,
                        (
                            ValidationEpisodeOutcome(
                                stage.snapshot_ids[0], 0, seed, True
                            ),
                        ),
                    ),
                ),
            )
            parameter = next(session.model.parameters())
            original = parameter.detach().clone()
            with torch.no_grad():
                parameter.add_(1.0)
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, "pending validation"):
                    session.save(
                        directory, "mutated", identity={"test": "pending-policy"}
                    )
            with self.assertRaisesRegex(ValueError, "loaded model state"):
                session.record_validation(report)
            with torch.no_grad():
                parameter.copy_(original)
            self.assertEqual(session.record_validation(report).phase, "complete")
        finally:
            vector.close()

    def assert_restore_rejects_pending_policy_mismatch(self) -> None:
        identity = {"test": "pending-policy-restore"}
        with tempfile.TemporaryDirectory() as directory:
            source, source_vector = build_session(
                exact=self.exact, construction_seed=303
            )
            try:
                source.initialize()
                source.request_validation(evaluator_identity_sha256="e" * 64)
                source.save(directory, "valid", identity=identity)
                checkpoint_identity = {
                    **identity,
                    "r3a_payload": source.version,
                }
                state, blobs, _ = load_checkpoint(
                    directory,
                    generation="valid",
                    expected_identity=checkpoint_identity,
                )
                parameter_name = next(iter(state["model"]))
                state["model"][parameter_name] = (
                    state["model"][parameter_name].clone() + 1.0
                )
                save_checkpoint(
                    directory,
                    "mismatch",
                    identity=checkpoint_identity,
                    state=state,
                    blobs=blobs,
                )
            finally:
                source_vector.close()

            restored, restored_vector = build_session(
                exact=self.exact, construction_seed=404
            )
            try:
                with self.assertRaisesRegex(ValueError, "pending validation"):
                    restored.restore(
                        directory, generation="mismatch", identity=identity
                    )
                self.assertTrue(restored.poisoned)
            finally:
                restored_vector.close()


@unittest.skipUnless(PORTABLE.exists(), "portable integration library not built")
class R3APortableSessionResumeTests(R3ASessionResumeMixin, unittest.TestCase):
    def test_next_rollout_gae_and_update_are_exact_after_resume(self) -> None:
        self.assert_resume()

    def test_nonzero_gauge_shaping_is_exact_after_resume(self) -> None:
        self.assert_resume(shaping_weight_ppm=250_000)

    def test_validation_is_bound_to_loaded_model(self) -> None:
        self.assert_validation_binds_loaded_model()

    def test_restore_rejects_pending_policy_mismatch(self) -> None:
        self.assert_restore_rejects_pending_policy_mismatch()


@unittest.skipUnless(EXACT.exists(), "exact integration worker not built")
class R3AExactSessionResumeTests(R3ASessionResumeMixin, unittest.TestCase):
    exact = True

    def test_next_rollout_gae_and_update_are_exact_after_resume(self) -> None:
        self.assert_resume()

    def test_nonzero_gauge_shaping_is_exact_after_resume(self) -> None:
        self.assert_resume(shaping_weight_ppm=250_000)

    def test_validation_is_bound_to_loaded_model(self) -> None:
        self.assert_validation_binds_loaded_model()


if __name__ == "__main__":
    unittest.main()
