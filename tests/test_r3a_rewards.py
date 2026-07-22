from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from irisu_rl.rewards import RewardComposer, RewardKnot, RewardSchedule
from irisu_rl.vector_adapter import MacroVectorAdapter
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.actions import SemanticAction
from tests.test_rl_vector_adapter import FakeActiveVector


class R3ARewardTests(unittest.TestCase):
    def transitions(self):
        adapter = MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder())
        adapter.reset()
        return adapter.step((SemanticAction.wait(1),) * 4)

    def test_integer_schedule_is_monotone_and_reaches_exact_zero(self) -> None:
        schedule = RewardSchedule(
            "decay-v1",
            (RewardKnot(0, 1_000_000), RewardKnot(3, 500_000), RewardKnot(10, 0)),
        )
        values = [schedule.weight_ppm(update) for update in range(15)]
        self.assertEqual(values[0], 1_000_000)
        self.assertEqual(values[3], 500_000)
        self.assertEqual(values[10:], [0] * 5)
        self.assertTrue(all(left >= right for left, right in zip(values, values[1:])))
        self.assertEqual(len(schedule.sha256), 64)

    def test_zero_weight_skips_shaping_and_is_exactly_score_only(self) -> None:
        calls = 0

        def shaping(transitions):
            nonlocal calls
            calls += 1
            return torch.full((len(transitions),), 1000.0)

        composer = RewardComposer(
            reward_scale=2.0, shaping_id="fixture-v1", shaping=shaping
        )
        result = composer.compose(self.transitions(), torch.zeros(4, dtype=torch.int64))
        self.assertEqual(calls, 0)
        torch.testing.assert_close(
            result.optimizer_reward, result.raw_reward.float() / 2
        )
        self.assertTrue(torch.equal(result.optimizer_reward, result.scaled_raw_reward))

    def test_components_remain_separate_and_tampering_is_detected(self) -> None:
        observed_lanes = []

        def shaping(values):
            observed_lanes.append(len(values))
            return torch.arange(len(values), dtype=torch.float32)

        composer = RewardComposer(
            reward_scale=2.0,
            shaping_id="constant-v1",
            shaping=shaping,
        )
        result = composer.compose(
            self.transitions(), torch.tensor([0, 250_000, 500_000, 1_000_000])
        )
        expected = result.scaled_raw_reward + torch.tensor([0.0, 0.0, 0.5, 2.0])
        torch.testing.assert_close(result.optimizer_reward, expected)
        self.assertEqual(observed_lanes, [3])
        self.assertEqual(result.raw_reward.dtype, torch.int64)
        self.assertEqual(result.optimizer_reward.dtype, torch.float32)
        with self.assertRaisesRegex(ValueError, "audited components"):
            replace(result, optimizer_reward=result.optimizer_reward + 1).validate(
                4, reward_scale=2.0
            )

    def test_invalid_or_increasing_schedules_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "begin"):
            RewardSchedule("bad", (RewardKnot(1, 0),))
        with self.assertRaisesRegex(ValueError, "nonincreasing"):
            RewardSchedule("bad", (RewardKnot(0, 0), RewardKnot(1, 1)))


if __name__ == "__main__":
    unittest.main()
