from __future__ import annotations

import unittest
import json
import tomllib
from pathlib import Path

import numpy as np

from irisu_env import ActionKind
from irisu_rl.actions import (
    ActionSpec,
    ConditionalActionDistribution,
    SemanticAction,
)
from irisu_rl.seeds import SEED_SPLITS_V1, SeedAllocator, validate_seed_splits
from irisu_rl.runtime_identity import ACCEPTED_EXACT_RUNTIME_2026_07_21

ROOT = Path(__file__).resolve().parents[1]


class ActionAndSeedTests(unittest.TestCase):
    def test_codec_validates_before_lowering_and_never_emits_both(self) -> None:
        spec = ActionSpec()
        self.assertEqual(spec.press(SemanticAction.wait(8)).kind, ActionKind.WAIT)
        self.assertEqual(
            spec.press(SemanticAction.weak(0.5, 0.25)).kind, ActionKind.WEAK_SHOT
        )
        self.assertEqual(spec.release().kind, ActionKind.WAIT)
        for invalid in (
            SemanticAction.wait(0),
            SemanticAction.wait(101),
            SemanticAction.weak(float("nan"), 0.5),
            SemanticAction.strong(-0.1, 0.5),
        ):
            with self.assertRaises(ValueError):
                spec.validate(invalid)
        for action in (
            SemanticAction.wait(37),
            SemanticAction.weak(0.2, 0.8),
            SemanticAction.strong(1.0, 0.0),
        ):
            self.assertEqual(spec.deserialize(spec.serialize(action)), action)

    def test_conditional_log_prob_ignores_inactive_heads(self) -> None:
        kind = np.array([[0.1, 0.2, 0.3], [0.3, -0.1, 0.2]])
        wait = np.zeros((2, 100))
        alpha = np.full((2, 2, 2), 2.0)
        beta = np.full((2, 2, 2), 3.0)
        actions = (SemanticAction.wait(4), SemanticAction.strong(0.2, 0.8))
        baseline = ConditionalActionDistribution(kind, wait, alpha, beta).log_prob(
            actions
        )
        changed = alpha.copy()
        changed[0] = 100.0  # wait row has no active coordinate head
        changed[1, 0] = 100.0  # strong row has no active weak head
        actual = ConditionalActionDistribution(kind, wait, changed, beta).log_prob(
            actions
        )
        np.testing.assert_allclose(actual, baseline, rtol=0, atol=1e-12)

    def test_distribution_sampling_is_reproducible_and_finite(self) -> None:
        distribution = ConditionalActionDistribution(
            np.zeros((16, 3)),
            np.zeros((16, 100)),
            np.full((16, 2, 2), 2.0),
            np.full((16, 2, 2), 2.0),
        )
        left = distribution.sample(np.random.default_rng(7))
        right = distribution.sample(np.random.default_rng(7))
        self.assertEqual(left, right)
        self.assertTrue(np.all(np.isfinite(distribution.log_prob(left))))
        self.assertTrue(np.all(np.isfinite(distribution.entropy())))
        uniform = ConditionalActionDistribution(
            np.zeros((1, 3)), np.zeros((1, 100)), np.ones((1, 2, 2)), np.ones((1, 2, 2))
        )
        self.assertAlmostEqual(uniform.entropy()[0], np.log(3) + np.log(100) / 3)

    def test_distribution_owns_caller_parameters_and_masks(self) -> None:
        kind = np.zeros((1, 3))
        wait = np.zeros((1, 100))
        alpha = np.full((1, 2, 2), 2.0)
        beta = np.full((1, 2, 2), 3.0)
        kind_mask = np.ones((1, 3), dtype=bool)
        wait_mask = np.ones((1, 100), dtype=bool)
        distribution = ConditionalActionDistribution(
            kind,
            wait,
            alpha,
            beta,
            kind_mask=kind_mask,
            wait_mask=wait_mask,
        )
        action = (SemanticAction.weak(0.25, 0.75),)
        baseline = distribution.log_prob(action).copy()
        kind[:] = 100
        wait[:] = 100
        alpha[:] = 100
        beta[:] = 100
        kind_mask[:] = False
        wait_mask[:] = False
        np.testing.assert_array_equal(distribution.log_prob(action), baseline)

    def test_masks_and_deterministic_evaluation(self) -> None:
        wait = np.zeros((1, 100))
        wait[0, 36] = 5
        distribution = ConditionalActionDistribution(
            np.array([[9.0, 8.0, 7.0]]),
            wait,
            np.full((1, 2, 2), 2.0),
            np.full((1, 2, 2), 2.0),
            kind_mask=np.array([[False, True, True]]),
        )
        self.assertEqual(distribution.deterministic(), (SemanticAction.weak(0.5, 0.5),))
        with self.assertRaisesRegex(ValueError, "all-masked"):
            ConditionalActionDistribution(
                np.zeros((1, 3)),
                np.zeros((1, 100)),
                np.ones((1, 2, 2)),
                np.ones((1, 2, 2)),
                kind_mask=np.zeros((1, 3), dtype=bool),
            )
        inactive_wait = ConditionalActionDistribution(
            np.zeros((1, 3)),
            np.zeros((1, 100)),
            np.ones((1, 2, 2)),
            np.ones((1, 2, 2)),
            kind_mask=np.array([[False, True, True]]),
            wait_mask=np.zeros((1, 100), dtype=bool),
        )
        self.assertTrue(np.isfinite(inactive_wait.entropy()[0]))

    def test_seed_allocator_reservation_resume_and_disjointness(self) -> None:
        validate_seed_splits(SEED_SPLITS_V1)
        allocator = SeedAllocator("train", key=41)
        pending = allocator.reserve(1000)
        self.assertEqual(allocator.cursor, 0)
        self.assertEqual(len(set(pending.seeds)), 1000)
        allocator.commit(pending)
        state = allocator.state_dict()
        resumed = SeedAllocator("train", key=41)
        resumed.load_state_dict(state)
        self.assertEqual(allocator.take(50), resumed.take(50))
        for seed in pending.seeds:
            self.assertLess(seed, SEED_SPLITS_V1["validation"].start)
        with self.assertRaises(ValueError):
            allocator.commit(pending)

    def test_checked_in_action_seed_reward_and_runtime_configs_match_code(self) -> None:
        action = tomllib.loads(
            (ROOT / "configs/rl/actions/deployment-v1.toml").read_text()
        )
        spec = ActionSpec()
        self.assertEqual(action["wait_min_ticks"], spec.wait_choices[0])
        self.assertEqual(action["wait_max_ticks"], spec.wait_choices[-1])
        self.assertEqual(action["click_macro"]["press_ticks"], spec.press_ticks)
        self.assertEqual(action["click_macro"]["release_ticks"], spec.release_ticks)
        seeds = json.loads((ROOT / "configs/rl/seeds/v1.json").read_text())
        self.assertEqual(
            seeds["splits"],
            {
                name: {"start": value.start, "size": value.size}
                for name, value in SEED_SPLITS_V1.items()
            },
        )
        runtime = json.loads(
            (ROOT / "configs/rl/runtime/exact-worker-2026-07-21.json").read_text()
        )
        self.assertEqual(
            runtime["worker_sha256"], ACCEPTED_EXACT_RUNTIME_2026_07_21.worker_sha256
        )
        self.assertEqual(
            runtime["exact_library_sha256"],
            ACCEPTED_EXACT_RUNTIME_2026_07_21.exact_library_sha256,
        )
        r0 = tomllib.loads((ROOT / "configs/rl/r0-r1.toml").read_text())
        self.assertEqual(r0["reward"]["raw"], "score_after - score_before")
        self.assertEqual(r0["smdp"]["gamma_tick"], 1.0)
        self.assertTrue(r0["smdp"]["neutral_wait_truncation_bootstrap"])
        self.assertFalse(r0["smdp"]["held_press_truncation_bootstrap"])


if __name__ == "__main__":
    unittest.main()
