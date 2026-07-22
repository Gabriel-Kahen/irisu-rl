from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tomllib
import unittest

import torch

from irisu_env import PaddedVectorEnv
from irisu_rl.actions import SemanticAction
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.returns import smdp_gae
from irisu_rl.rewards import LinearGaugePotential, RewardComposer
from irisu_rl.schema import TEACHER_V1
from irisu_rl.vector_adapter import MacroVectorAdapter, MacroTransition
from tests.test_rl_vector_adapter import FakeActiveVector


ROOT = Path(__file__).resolve().parents[1]


class GaugePotentialTests(unittest.TestCase):
    def setUp(self) -> None:
        adapter = MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder())
        adapter.reset()
        self.base = adapter.step((SemanticAction.wait(1),) * 4)[0]
        self.potential = LinearGaugePotential()

    def transition(
        self,
        start: int,
        end: int,
        *,
        gauge_max: int = 40_000,
        terminated: bool = False,
        truncated: bool = False,
    ) -> MacroTransition:
        return replace(
            self.base,
            start_gauge=start,
            end_gauge=end,
            gauge_max=gauge_max,
            terminated=terminated,
            truncated=truncated,
        )

    def shaping(self, *transitions: MacroTransition) -> torch.Tensor:
        return self.potential(transitions)

    def test_loss_recovery_and_passive_drain_have_linear_signals(self) -> None:
        values = self.shaping(
            self.transition(3_000, 2_999),
            self.transition(3_000, 1_179),
            self.transition(3_000, 3_699),
            self.transition(3_000, 3_000),
        )
        torch.testing.assert_close(
            values,
            torch.tensor([-1 / 40_000, -1_821 / 40_000, 699 / 40_000, 0.0]),
        )

    def test_clamping_terminal_and_truncation_rules_are_explicit(self) -> None:
        values = self.shaping(
            self.transition(999, -822, gauge_max=10_000),
            self.transition(-822, 1, gauge_max=10_000, terminated=True),
            self.transition(9_999, 12_000, gauge_max=10_000),
            self.transition(1_000, 900, gauge_max=10_000, truncated=True),
            self.transition(8_000, 8_500, gauge_max=10_000, terminated=True),
        )
        torch.testing.assert_close(
            values,
            torch.tensor([-0.0999, 0.0, 0.0001, -0.01, -0.8]),
        )

    def test_native_delayed_terminal_uses_final_gauge_not_autoreset_gauge(
        self,
    ) -> None:
        library = ROOT / "build-physics-integration-portable/libirisu_clone.so"
        if not library.is_file():
            self.skipTest("portable native integration library is unavailable")
        vector = PaddedVectorEnv(
            1,
            library_path=library,
            config={"max_episode_ticks": 10_000},
        )
        try:
            adapter = MacroVectorAdapter(vector, encoder=TeacherStateEncoder())
            adapter.reset()
            terminal = None
            for _ in range(30):
                candidate = adapter.step((SemanticAction.wait(100),))[0]
                if candidate.terminated:
                    terminal = candidate
                    break
            self.assertIsNotNone(terminal)
            assert terminal is not None
            self.assertFalse(terminal.truncated)
            self.assertEqual(terminal.start_gauge, 1)
            self.assertEqual(terminal.end_gauge, 1)
            gauge_index = TEACHER_V1.global_features.index("gauge_fraction")
            self.assertAlmostEqual(
                float(
                    terminal.transition_next_observation.global_features[
                        0, gauge_index
                    ]
                ),
                1 / 40_000,
            )
            self.assertAlmostEqual(
                float(terminal.next_policy_observation.global_features[0, gauge_index]),
                3_000 / 40_000,
            )
            torch.testing.assert_close(
                self.shaping(terminal),
                torch.tensor([-1 / 40_000]),
                rtol=0,
                atol=1e-9,
            )
        finally:
            vector.close()

    def test_potential_telescopes_across_macro_partitions(self) -> None:
        whole = self.shaping(self.transition(3_000, 2_900)).sum()
        split = self.shaping(
            self.transition(3_000, 2_950), self.transition(2_950, 2_900)
        ).sum()
        torch.testing.assert_close(whole, split, rtol=0, atol=1e-8)

        episode = self.shaping(
            self.transition(3_000, 2_000),
            self.transition(2_000, 3_700),
            self.transition(3_700, 1, terminated=True),
        ).sum()
        torch.testing.assert_close(
            episode, torch.tensor(-3_000 / 40_000), rtol=0, atol=1e-8
        )

    def test_composer_preserves_score_and_applies_episode_coefficient(self) -> None:
        composer = RewardComposer(
            reward_scale=1_000.0, shaping_spec=self.potential
        )
        transition = replace(
            self.transition(3_000, 1_179), raw_reward=16
        )
        reward = composer.compose(
            (transition,), torch.tensor([250_000], dtype=torch.int64)
        )
        torch.testing.assert_close(reward.scaled_raw_reward, torch.tensor([0.016]))
        torch.testing.assert_close(reward.shaping_reward, torch.tensor([-0.045525]))
        torch.testing.assert_close(
            reward.optimizer_reward, torch.tensor([0.016 - 0.25 * 0.045525])
        )
        self.assertEqual(reward.raw_reward.item(), 16)

    def test_invalid_authoritative_fields_fail_closed(self) -> None:
        with self.assertRaisesRegex(TypeError, "canonical integer"):
            self.shaping(replace(self.base, start_gauge=True))
        with self.assertRaisesRegex(ValueError, "maximum"):
            self.shaping(replace(self.base, gauge_max=0))
        with self.assertRaisesRegex(ValueError, "positive elapsed"):
            self.shaping(replace(self.base, elapsed_ticks=0))

    def test_manifest_binds_formula_without_changing_legacy_identity(self) -> None:
        legacy = RewardComposer(reward_scale=1_000.0)
        self.assertEqual(
            legacy.manifest(),
            {
                "version": "reward-composer-v1",
                "reward_scale": 1_000.0,
                "raw_reward": "score_after - score_before",
                "shaping_id": "none",
                "requires_events": False,
                "clip": False,
            },
        )
        shaped = RewardComposer(reward_scale=1_000.0, shaping_spec=self.potential)
        self.assertEqual(shaped.manifest()["shaping_spec"], self.potential.manifest())
        self.assertNotEqual(shaped.sha256, legacy.sha256)
        with self.assertRaisesRegex(ValueError, "potential_scale=1"):
            LinearGaugePotential(potential_scale=0.5)
        with self.assertRaisesRegex(ValueError, "shaping_spec argument"):
            RewardComposer(
                shaping_id=self.potential.shaping_id,
                shaping=self.potential,
            )

    def test_composer_freezes_and_revalidates_custom_spec_identity(self) -> None:
        class MutableSpec:
            shaping_id = "mutable-gauge-test-v1"
            requires_events = False
            gamma_tick = 1.0
            critic_condition_features = 1

            def __init__(self):
                self.version = "mutable-gauge-test-v1"

            def manifest(self):
                result = self.potential.manifest()
                result["version"] = self.version
                result["shaping_id"] = self.shaping_id
                return result

            def __call__(self, transitions):
                return self.potential(transitions)

            potential = LinearGaugePotential()

        spec = MutableSpec()
        composer = RewardComposer(shaping_spec=spec)
        original = composer.manifest()
        spec.version = "mutated"
        self.assertEqual(composer.manifest(), original)
        with self.assertRaisesRegex(RuntimeError, "manifest changed"):
            composer.validate_identity()

    def test_config_is_an_exact_checked_reward_decision_record(self) -> None:
        config = tomllib.loads(
            (
                ROOT
                / "configs/rl/rewards/r3b-linear-gauge-potential-v1.toml"
            ).read_text()
        )
        self.assertEqual(config["potential"], self.potential.manifest())
        self.assertEqual(
            config["coefficient"]["candidate_weight_ppm"],
            [0, 100_000, 250_000, 500_000],
        )
        self.assertEqual(config["score_only"]["final_weight_ppm"], 0)
        self.assertEqual(
            config["coefficient"]["starting_hypothesis_weight_ppm"], 100_000
        )

    def test_critic_condition_cannot_change_actor_or_recurrent_state(self) -> None:
        torch.manual_seed(41)
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(
                8, 8, 12, 12, 1, critic_condition_features=1
            ),
        )
        observation = self.base.observation
        global_features = torch.from_numpy(observation.global_features).repeat(2, 1)
        global_features[1, 0] += 0.25
        global_features = global_features.unsqueeze(0)
        body_features = torch.from_numpy(observation.body_features).repeat(2, 1, 1)
        body_features = body_features.unsqueeze(0)
        body_mask = torch.from_numpy(observation.body_mask).repeat(2, 1).unsqueeze(0)
        state = model.initial_state(2)
        common = (global_features, body_features, body_mask, state)
        zero = model(*common, critic_condition=torch.zeros((1, 2, 1)))
        one = model(*common, critic_condition=torch.ones((1, 2, 1)))
        for left, right in (
            (zero.kind_logits, one.kind_logits),
            (zero.wait_logits, one.wait_logits),
            (zero.coordinate_alpha, one.coordinate_alpha),
            (zero.coordinate_beta, one.coordinate_beta),
            (zero.recurrent_state, one.recurrent_state),
        ):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertFalse(torch.equal(zero.values, one.values))
        slopes = (one.values - zero.values).detach()
        self.assertNotEqual(float(slopes[0, 0]), float(slopes[0, 1]))

    def test_shifted_critic_keeps_smdp_gae_invariant(self) -> None:
        alpha = 0.25
        raw_rewards = torch.tensor([[0.0], [0.016], [0.0]])
        phi = torch.tensor([[0.075], [0.05], [0.04]])
        next_phi = torch.tensor([[0.05], [0.09], [0.0]])
        shaping = next_phi - phi
        raw_values = torch.tensor([[1.0], [2.0], [3.0]])
        raw_bootstrap = torch.tensor([[2.0], [3.0], [0.0]])
        shaped_values = raw_values - alpha * phi
        shaped_bootstrap = raw_bootstrap - alpha * next_phi
        elapsed = torch.tensor([[3], [7], [2]], dtype=torch.int64)
        bootstrap = torch.tensor([[True], [True], [False]])
        # Row one is a time-limit truncation: retain and bootstrap its outgoing
        # potential, but stop the trace before the reset-created next episode.
        trace = torch.tensor([[True], [False], [False]])
        valid = torch.ones((3, 1), dtype=torch.bool)

        raw = smdp_gae(
            raw_rewards,
            raw_values,
            raw_bootstrap,
            elapsed,
            bootstrap,
            trace,
            valid,
            lambda_tick=0.91,
        )
        shaped = smdp_gae(
            raw_rewards + alpha * shaping,
            shaped_values,
            shaped_bootstrap,
            elapsed,
            bootstrap,
            trace,
            valid,
            lambda_tick=0.91,
        )
        torch.testing.assert_close(shaped.deltas, raw.deltas)
        torch.testing.assert_close(shaped.advantages, raw.advantages)


if __name__ == "__main__":
    unittest.main()
