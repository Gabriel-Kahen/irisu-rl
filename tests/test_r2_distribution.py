from __future__ import annotations

import unittest

import numpy as np
import torch

from irisu_rl.actions import ConditionalActionDistribution, SemanticAction
from irisu_rl.torch_distribution import ActionTensor, TorchConditionalActionDistribution


class TorchDistributionTests(unittest.TestCase):
    def fixture(self):
        kind = np.array([[0.2, -0.1, 0.4], [0.5, 0.7, -0.2], [-0.4, 0.1, 0.9]])
        wait = np.linspace(-1, 1, 300).reshape(3, 100)
        alpha = np.array(
            [
                [[2.0, 3.0], [4.0, 5.0]],
                [[1.5, 2.5], [3.5, 4.5]],
                [[5.0, 2.0], [2.0, 5.0]],
            ]
        )
        beta = alpha + 0.75
        actions = (
            SemanticAction.wait(4),
            SemanticAction.weak(0.0, 1.0),
            SemanticAction.strong(0.25, 0.8),
        )
        tensor_actions = ActionTensor(
            torch.tensor([0, 1, 2]),
            torch.tensor([3, 0, 0]),
            torch.tensor([[0.0, 0.0], [0.0, 1.0], [0.25, 0.8]], dtype=torch.float64),
        )
        return kind, wait, alpha, beta, actions, tensor_actions

    def test_log_probability_and_entropy_match_numpy_oracle(self) -> None:
        kind, wait, alpha, beta, actions, tensor_actions = self.fixture()
        expected = ConditionalActionDistribution(kind, wait, alpha, beta)
        actual = TorchConditionalActionDistribution(
            torch.tensor(kind),
            torch.tensor(wait),
            torch.tensor(alpha),
            torch.tensor(beta),
        )
        components = actual.log_prob_components(tensor_actions)
        torch.testing.assert_close(
            components.total,
            torch.tensor(expected.log_prob(actions)),
            rtol=0,
            atol=1e-11,
        )
        torch.testing.assert_close(
            actual.entropy(),
            torch.tensor(expected.entropy()),
            rtol=0,
            atol=2e-9,
        )
        torch.testing.assert_close(
            components.total,
            components.kind + components.wait + components.coordinates,
        )

    def test_selected_likelihood_has_zero_inactive_branch_gradients(self) -> None:
        kind = torch.zeros((3, 3), dtype=torch.float64, requires_grad=True)
        wait = torch.zeros((3, 100), dtype=torch.float64, requires_grad=True)
        alpha = torch.full((3, 2, 2), 2.0, dtype=torch.float64, requires_grad=True)
        beta = torch.full((3, 2, 2), 3.0, dtype=torch.float64, requires_grad=True)
        distribution = TorchConditionalActionDistribution(kind, wait, alpha, beta)
        actions = ActionTensor(
            torch.tensor([0, 1, 2]),
            torch.tensor([0, 0, 0]),
            torch.full((3, 2), 0.5, dtype=torch.float64),
        )
        distribution.log_prob(actions).sum().backward()
        torch.testing.assert_close(alpha.grad[0], torch.zeros_like(alpha.grad[0]))
        torch.testing.assert_close(beta.grad[0], torch.zeros_like(beta.grad[0]))
        torch.testing.assert_close(wait.grad[1:], torch.zeros_like(wait.grad[1:]))
        torch.testing.assert_close(alpha.grad[1, 1], torch.zeros_like(alpha.grad[1, 1]))
        torch.testing.assert_close(alpha.grad[2, 0], torch.zeros_like(alpha.grad[2, 0]))

    def test_masks_reject_corrupt_actions_and_sampling_is_canonical(self) -> None:
        kind_mask = torch.tensor([[False, True, False], [True, False, False]])
        wait_mask = torch.zeros((2, 100), dtype=torch.bool)
        wait_mask[1, 7] = True
        distribution = TorchConditionalActionDistribution(
            torch.zeros((2, 3)),
            torch.zeros((2, 100)),
            torch.full((2, 2, 2), 2.0),
            torch.full((2, 2, 2), 2.0),
            kind_mask=kind_mask,
            wait_mask=wait_mask,
        )
        torch.manual_seed(19)
        sampled = distribution.sample()
        self.assertEqual(sampled.kind.tolist(), [1, 0])
        self.assertEqual(sampled.wait_index.tolist(), [0, 7])
        torch.testing.assert_close(sampled.xy[1], torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "masked kind"):
            distribution.log_prob(
                ActionTensor(
                    torch.tensor([2, 0]),
                    torch.tensor([0, 7]),
                    torch.zeros((2, 2)),
                )
            )
        with self.assertRaisesRegex(ValueError, "masked wait"):
            distribution.log_prob(
                ActionTensor(
                    torch.tensor([1, 0]),
                    torch.tensor([0, 8]),
                    torch.full((2, 2), 0.5),
                )
            )


if __name__ == "__main__":
    unittest.main()
