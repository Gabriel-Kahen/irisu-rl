from __future__ import annotations

import unittest

import torch

from irisu_rl.returns import lambda_tick_from_half_life, smdp_gae


class SmdpReturnsTests(unittest.TestCase):
    def test_hand_calculated_mixed_duration_and_truncation_fixture(self) -> None:
        rewards = torch.tensor([[1.0], [2.0]], dtype=torch.float64)
        values = torch.tensor([[5.0], [7.0]], dtype=torch.float64)
        bootstrap = torch.tensor([[7.0], [11.0]], dtype=torch.float64)
        ticks = torch.tensor([[2], [3]])
        bootstrap_mask = torch.tensor([[True], [True]])
        trace_mask = torch.tensor([[True], [False]])
        valid = torch.ones((2, 1), dtype=torch.bool)
        result = smdp_gae(
            rewards,
            values,
            bootstrap,
            ticks,
            bootstrap_mask,
            trace_mask,
            valid,
            gamma_tick=0.9,
            lambda_tick=0.8,
            reward_is_event_discounted=True,
        )
        expected_delta = torch.tensor([[1.67], [3.019]], dtype=torch.float64)
        expected_advantage = torch.tensor(
            [[1.67 + (0.9 * 0.8) ** 2 * 3.019], [3.019]], dtype=torch.float64
        )
        torch.testing.assert_close(result.deltas, expected_delta)
        torch.testing.assert_close(result.advantages, expected_advantage)
        torch.testing.assert_close(result.returns, expected_advantage + values)

    def test_terminal_ignores_bootstrap_and_padding_is_inert(self) -> None:
        result = smdp_gae(
            torch.tensor([[3.0, 999.0], [4.0, 999.0]]),
            torch.tensor([[2.0, 999.0], [5.0, 999.0]]),
            torch.full((2, 2), 1e20),
            torch.tensor([[100, 0], [1, 0]]),
            torch.tensor([[False, False], [False, False]]),
            torch.tensor([[False, False], [False, False]]),
            torch.tensor([[True, False], [True, False]]),
            lambda_tick=0.99,
        )
        torch.testing.assert_close(result.deltas[:, 0], torch.tensor([1.0, -1.0]))
        torch.testing.assert_close(result.advantages[:, 1], torch.zeros(2))
        torch.testing.assert_close(result.returns[:, 1], torch.zeros(2))

    def test_invalid_discount_and_mask_contracts_fail_closed(self) -> None:
        values = torch.zeros((1, 1))
        ticks = torch.ones((1, 1), dtype=torch.long)
        true = torch.ones((1, 1), dtype=torch.bool)
        false = torch.zeros((1, 1), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "per-event"):
            smdp_gae(
                values,
                values,
                values,
                ticks,
                true,
                true,
                true,
                gamma_tick=0.99,
                lambda_tick=0.9,
            )
        with self.assertRaisesRegex(ValueError, "continuing trace"):
            smdp_gae(values, values, values, ticks, false, true, true, lambda_tick=0.9)
        self.assertAlmostEqual(lambda_tick_from_half_life(1.0), 2**-0.02)


if __name__ == "__main__":
    unittest.main()
