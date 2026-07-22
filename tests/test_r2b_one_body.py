from __future__ import annotations

import copy
import json
import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np
import torch

from benchmarks.rl_r2b import acceptance_predicates
from irisu_env import EventKind, PaddedVectorEnv
from irisu_rl import MacroVectorAdapter, TeacherStateEncoder
from irisu_rl.actions import ActionSpec
from irisu_rl.checkpoints import (
    capture_rng_state,
    load_checkpoint,
    pack_adapter_checkpoint,
    restore_rng_state,
    save_checkpoint,
    unpack_adapter_checkpoint,
)
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.one_body import (
    OneBodySpec,
    OneBodyTask,
    expert_actions,
    one_body_training_batch,
    policy_distribution,
)
from irisu_rl.ppo import PPOConfig, PPOTrainer
from irisu_rl.recurrent_buffer import RecurrentRolloutBuffer
from irisu_rl.schema import TEACHER_V1
from irisu_rl.seeds import SeedAllocator


ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT / "build-physics-integration-portable" / "libirisu_clone.so"
SMALL = RecurrentModelConfig(
    8, 8, 12, 12, 1, coordinate_parameterization="mean-log-concentration"
)


class OneBodyContractTests(unittest.TestCase):
    def test_spec_config_and_checked_experiment_are_disjoint(self) -> None:
        spec = OneBodySpec()
        groups = (
            spec.train_heights,
            spec.calibration_heights,
            spec.validation_heights,
            spec.test_heights,
        )
        self.assertEqual(sum(map(len, groups)), len(set().union(*map(set, groups))))
        config = tomllib.loads(
            (ROOT / "configs/rl/experiments/r2b-one-body-v1.toml").read_text()
        )
        self.assertEqual(config["task_version"], spec.version)
        self.assertEqual(config["mechanics"]["train_heights"], list(spec.train_heights))
        self.assertFalse(config["deployable"])
        self.assertEqual(config["observation_provenance"], "privileged_simulator")

    def test_mean_concentration_head_is_finite_and_explicit(self) -> None:
        model = RecurrentActorCritic(TEACHER_V1, config=SMALL)
        output = model(
            torch.zeros((1, 2, len(TEACHER_V1.global_features))),
            torch.zeros((1, 2, TEACHER_V1.capacity, len(TEACHER_V1.body_features))),
            torch.zeros((1, 2, TEACHER_V1.capacity), dtype=torch.bool),
            model.initial_state(2),
        )
        self.assertTrue(torch.all(output.coordinate_alpha > 1))
        self.assertTrue(torch.all(output.coordinate_beta > 1))
        self.assertEqual(model.manifest()["architecture"], "recurrent-actor-critic-v2")

    def test_default_head_keeps_r2a_v1_identity(self) -> None:
        model = RecurrentActorCritic(
            TEACHER_V1, config=RecurrentModelConfig(8, 8, 12, 12, 1)
        )
        self.assertEqual(model.manifest()["architecture"], "recurrent-actor-critic-v1")
        self.assertNotIn("coordinate_parameterization", model.config.manifest())

    def test_checked_learning_artifact_passes_without_claiming_deployment(self) -> None:
        artifact = json.loads(
            (ROOT / "benchmarks/results/rl-r2b-one-body-2026-07-22.json").read_text()
        )
        predicates = acceptance_predicates(artifact)
        self.assertEqual(artifact["acceptance"]["predicates"], predicates)
        self.assertEqual(artifact["acceptance"]["pass"], all(predicates.values()))
        self.assertTrue(artifact["acceptance"]["pass"])
        self.assertEqual(artifact["selected_learning_rate"], 1e-4)
        self.assertFalse(artifact["source"]["dirty"])
        self.assertFalse(artifact["deployable"])
        self.assertEqual(artifact["task"]["sha256"], OneBodySpec().sha256)

    @unittest.skipUnless(PORTABLE.exists(), "portable integration library not built")
    def test_expert_hits_and_raw_reward_remains_separate(self) -> None:
        with OneBodyTask(32, 100.0, library_path=PORTABLE) as task:
            observations = task.reset(range(32))
            outcome = task.step(expert_actions(task.target_xy))
        self.assertEqual(observations.body_mask.sum(), 32)
        self.assertTrue(torch.all(outcome.hit))
        self.assertTrue(torch.all(outcome.optimizer_reward == 1))
        self.assertTrue(torch.all(outcome.raw_reward == 0))

    @unittest.skipUnless(PORTABLE.exists(), "portable integration library not built")
    def test_adapter_owns_press_and_release_events_before_expiry(self) -> None:
        with PaddedVectorEnv(
            2,
            library_path=PORTABLE,
            config=OneBodySpec.mechanics_config(100.0),
        ) as vector:
            adapter = MacroVectorAdapter(
                vector, encoder=TeacherStateEncoder(), capture_events=True
            )
            current = adapter.reset()
            x = current.body_features[
                :, 0, TEACHER_V1.body_features.index("effect_x_norm")
            ]
            y = current.body_features[
                :, 0, TEACHER_V1.body_features.index("effect_y_norm")
            ]
            transitions = adapter.step(
                tuple(
                    ActionSpec().decode(1, 0, float(xy[0]), float(xy[1]))
                    for xy in zip(x, y)
                )
            )
        for transition in transitions:
            phases = {event.primitive_phase for event in transition.diagnostics.events}
            self.assertEqual(phases, {"press", "release"})
            self.assertTrue(
                any(
                    event.kind == int(EventKind.PROJECTILE_HIT)
                    for event in transition.diagnostics.events
                )
            )
            self.assertEqual(
                transition.diagnostics.event_count,
                len(transition.diagnostics.events),
            )

    @unittest.skipUnless(PORTABLE.exists(), "portable integration library not built")
    def test_adapter_counts_events_without_copying_them_by_default(self) -> None:
        with PaddedVectorEnv(
            2,
            library_path=PORTABLE,
            config=OneBodySpec.mechanics_config(100.0),
        ) as vector:
            adapter = MacroVectorAdapter(vector, encoder=TeacherStateEncoder())
            current = adapter.reset()
            x_index = TEACHER_V1.body_features.index("effect_x_norm")
            y_index = TEACHER_V1.body_features.index("effect_y_norm")
            transitions = adapter.step(
                tuple(
                    ActionSpec().decode(
                        1,
                        0,
                        float(current.body_features[lane, 0, x_index]),
                        float(current.body_features[lane, 0, y_index]),
                    )
                    for lane in range(2)
                )
            )
        for transition in transitions:
            self.assertGreater(transition.diagnostics.event_count, 0)
            self.assertEqual(transition.diagnostics.events, ())

    @unittest.skipUnless(PORTABLE.exists(), "portable integration library not built")
    def test_buffer_accepts_task_reward_without_overwriting_raw_score(self) -> None:
        with PaddedVectorEnv(
            2,
            library_path=PORTABLE,
            config=OneBodySpec.mechanics_config(100.0),
        ) as vector:
            adapter = MacroVectorAdapter(vector, encoder=TeacherStateEncoder())
            current = adapter.reset()
            x_index = TEACHER_V1.body_features.index("effect_x_norm")
            y_index = TEACHER_V1.body_features.index("effect_y_norm")
            transitions = adapter.step(
                tuple(
                    ActionSpec().decode(
                        1,
                        0,
                        float(current.body_features[lane, 0, x_index]),
                        float(current.body_features[lane, 0, y_index]),
                    )
                    for lane in range(2)
                )
            )
        buffer = RecurrentRolloutBuffer(1, 2, TEACHER_V1, torch.zeros((1, 2, 4)))
        buffer.append(
            current,
            transitions,
            torch.zeros(2),
            torch.zeros(2),
            reset_before=torch.zeros(2, dtype=torch.bool),
            optimizer_reward=torch.ones(2),
        )
        torch.testing.assert_close(
            buffer.raw_reward[0], torch.zeros(2, dtype=torch.int64)
        )
        torch.testing.assert_close(buffer.optimizer_reward[0], torch.ones(2))


@unittest.skipUnless(PORTABLE.exists(), "portable integration library not built")
class IntegratedResumeTests(unittest.TestCase):
    @staticmethod
    def continue_once(adapter, observations, model, trainer):
        with torch.no_grad():
            distribution, values = policy_distribution(model, observations)
            actions = distribution.sample()
            log_prob = distribution.log_prob(actions)
        semantic = tuple(
            model.action_spec.decode(1, 0, float(x), float(y)) for x, y in actions.xy[0]
        )
        transitions = adapter.step(semantic)
        x_index = TEACHER_V1.body_features.index("effect_x_norm")
        y_index = TEACHER_V1.body_features.index("effect_y_norm")
        target = torch.from_numpy(
            observations.body_features[:, 0, [x_index, y_index]].copy()
        )
        hit = torch.tensor(
            [
                any(
                    event.kind == int(EventKind.PROJECTILE_HIT)
                    for event in transition.diagnostics.events
                )
                for transition in transitions
            ]
        )
        aim = torch.exp(
            -(actions.xy[0] - target).square().sum(dim=-1)
            / (2 * OneBodySpec().aim_sigma ** 2)
        )
        reward = 0.75 * hit.float() + 0.25 * aim
        batch = one_body_training_batch(
            model, observations, actions, log_prob, values, reward
        )
        stats = trainer.update(batch)
        return (
            copy.deepcopy(actions),
            tuple(adapter.env.state_hash()),
            stats,
            copy.deepcopy(model.state_dict()),
        )

    def test_sampled_action_environment_and_update_resume_exactly(self) -> None:
        identity = {"test": "r2b-integrated-resume-v1"}
        numpy_generator = np.random.default_rng(73)
        torch.manual_seed(73)
        with PaddedVectorEnv(
            2,
            library_path=PORTABLE,
            config=OneBodySpec.mechanics_config(100.0),
        ) as vector:
            adapter = MacroVectorAdapter(
                vector,
                encoder=TeacherStateEncoder(),
                seed_allocator=SeedAllocator("train", key=73),
                capture_events=True,
            )
            observations = adapter.reset()
            model = RecurrentActorCritic(TEACHER_V1, config=SMALL)
            trainer = PPOTrainer(
                model,
                config=PPOConfig(epochs=1, lane_minibatch_size=2, target_kl=1.0),
                total_updates=2,
                sampler_seed=73,
            )
            adapter_state, blobs = pack_adapter_checkpoint(adapter.checkpoint())
            with tempfile.TemporaryDirectory() as directory:
                save_checkpoint(
                    directory,
                    "before-next-update",
                    identity=identity,
                    state={
                        "adapter": adapter_state,
                        "model": model.state_dict(),
                        "trainer": trainer.state_dict(),
                        "rng": capture_rng_state(numpy_generator),
                    },
                    blobs=blobs,
                )
                loaded, loaded_blobs, _ = load_checkpoint(
                    directory, expected_identity=identity
                )
            expected = self.continue_once(adapter, observations, model, trainer)

        restored_checkpoint = unpack_adapter_checkpoint(
            loaded["adapter"],
            loaded_blobs,
            schema=TEACHER_V1,
            action_spec=ActionSpec(),
        )
        with PaddedVectorEnv(
            2,
            library_path=PORTABLE,
            config=OneBodySpec.mechanics_config(100.0),
        ) as vector:
            adapter = MacroVectorAdapter(
                vector,
                encoder=TeacherStateEncoder(),
                seed_allocator=SeedAllocator("train", key=73),
                capture_events=True,
            )
            adapter.reset()
            observations = adapter.restore_checkpoint(restored_checkpoint)
            model = RecurrentActorCritic(TEACHER_V1, config=SMALL)
            trainer = PPOTrainer(
                model,
                config=PPOConfig(epochs=1, lane_minibatch_size=2, target_kl=1.0),
                total_updates=2,
                sampler_seed=999,
            )
            model.load_state_dict(loaded["model"], strict=True)
            trainer.load_state_dict(loaded["trainer"])
            restore_rng_state(loaded["rng"], numpy_generator)
            actual = self.continue_once(adapter, observations, model, trainer)

        self.assertTrue(torch.equal(actual[0].kind, expected[0].kind))
        self.assertTrue(torch.equal(actual[0].xy, expected[0].xy))
        self.assertEqual(actual[1:3], expected[1:3])
        for name, value in actual[3].items():
            self.assertTrue(torch.equal(value, expected[3][name]), name)


if __name__ == "__main__":
    unittest.main()
