from __future__ import annotations

import unittest
from dataclasses import replace

import torch

from irisu_rl.actions import ActionSpec, SemanticAction
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.recurrent_buffer import RecurrentRolloutBuffer
from irisu_rl.torch_distribution import LogProbabilityComponents
from irisu_rl.vector_adapter import MacroVectorAdapter
from tests.test_rl_vector_adapter import FakeTruncatingVector
from tests.test_rl_vector_adapter import FakeActiveVector


class RecurrentBufferTests(unittest.TestCase):
    @staticmethod
    def zero_components(lanes: int) -> LogProbabilityComponents:
        zeros = torch.zeros(lanes)
        return LogProbabilityComponents(zeros, zeros.clone(), zeros.clone())

    def _censored_predecessor_advantage(
        self, censored_reward: int, value: float
    ) -> float:
        adapter = MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder())
        current = adapter.reset()
        actions = tuple(SemanticAction.wait(1) for _ in range(4))
        first = adapter.step(actions)
        buffer = RecurrentRolloutBuffer(2, 4, current.schema, torch.zeros((1, 4, 3)))
        buffer.append(
            current,
            first,
            torch.zeros(4),
            torch.zeros(4),
            old_log_prob_components=self.zero_components(4),
            reset_before=torch.zeros(4, dtype=torch.bool),
        )
        second_observation = adapter.current_observation
        second = list(adapter.step(actions))
        second[0] = replace(
            second[0],
            raw_reward=censored_reward,
            truncated=True,
            macro_interrupted=True,
            bootstrap_mask=False,
            trace_mask=False,
        )
        old_values = torch.zeros(4)
        old_values[0] = value
        buffer.append(
            second_observation,
            second,
            torch.zeros(4),
            old_values,
            old_log_prob_components=self.zero_components(4),
            reset_before=torch.zeros(4, dtype=torch.bool),
        )
        return float(
            buffer.finalize(torch.zeros((2, 4)), lambda_tick=0.99).advantages[0, 0]
        )

    def test_censored_tail_cannot_change_preceding_advantage(self) -> None:
        baseline = self._censored_predecessor_advantage(0, 0.0)
        poisoned = self._censored_predecessor_advantage(1_000_000, -500_000.0)
        self.assertEqual(baseline, poisoned)

    def test_censored_press_truncation_is_audited_but_not_trained(self) -> None:
        adapter = MacroVectorAdapter(
            FakeTruncatingVector(), encoder=TeacherStateEncoder()
        )
        current = adapter.reset()
        actions = (
            SemanticAction.wait(1),
            SemanticAction.wait(8),
            SemanticAction.weak(0.5, 0.5),
            SemanticAction.wait(1),
        )
        transitions = adapter.step(actions)
        buffer = RecurrentRolloutBuffer(
            2,
            4,
            current.schema,
            torch.zeros((1, 4, 12)),
            reward_scale=2.0,
        )
        buffer.append(
            current,
            transitions,
            torch.zeros(4),
            torch.zeros(4),
            old_log_prob_components=self.zero_components(4),
            reset_before=torch.zeros(4, dtype=torch.bool),
        )
        batch = buffer.finalize(torch.zeros((1, 4)), lambda_tick=0.99)
        self.assertTrue(buffer.truncated[0, 1])
        self.assertTrue(buffer.bootstrap_mask[0, 1])
        self.assertTrue(batch.train_mask[0, 1])
        self.assertTrue(buffer.truncated[0, 2])
        self.assertTrue(buffer.macro_interrupted[0, 2])
        self.assertFalse(buffer.bootstrap_mask[0, 2])
        self.assertFalse(batch.train_mask[0, 2])
        torch.testing.assert_close(
            buffer.optimizer_reward[0], buffer.raw_reward[0].float() / 2
        )
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            buffer.append(
                current,
                transitions,
                torch.zeros(4),
                torch.zeros(4),
                old_log_prob_components=self.zero_components(4),
                reset_before=torch.zeros(4, dtype=torch.bool),
            )

    def test_action_encoding_and_observations_are_owned(self) -> None:
        adapter = MacroVectorAdapter(
            FakeTruncatingVector(), encoder=TeacherStateEncoder()
        )
        current = adapter.reset()
        actions = tuple(SemanticAction.wait(1) for _ in range(4))
        transitions = adapter.step(actions)
        buffer = RecurrentRolloutBuffer(1, 4, current.schema, torch.zeros((1, 4, 3)))
        buffer.append(
            current,
            transitions,
            torch.zeros(4),
            torch.zeros(4),
            old_log_prob_components=self.zero_components(4),
            reset_before=torch.zeros(4, dtype=torch.bool),
        )
        saved = buffer.global_features.clone()
        current.global_features[:] = -100
        torch.testing.assert_close(buffer.global_features, saved)
        self.assertEqual(
            buffer.action_wait_index[0, 0], ActionSpec().wait_choices.index(1)
        )

    def test_failed_append_does_not_leave_stale_masks(self) -> None:
        adapter = MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder())
        current = adapter.reset()
        actions = tuple(SemanticAction.wait(1) for _ in range(4))
        valid_transitions = list(adapter.step(actions))
        transitions = valid_transitions.copy()
        transitions[-1] = replace(
            transitions[-1],
            diagnostics=replace(transitions[-1].diagnostics, config_hash=2**64),
        )
        buffer = RecurrentRolloutBuffer(1, 4, current.schema, torch.zeros((1, 4, 3)))
        custom = torch.zeros((4, 3), dtype=torch.bool)
        custom[:, 0] = True
        with self.assertRaisesRegex(ValueError, "config hash"):
            buffer.append(
                current,
                transitions,
                torch.zeros(4),
                torch.zeros(4),
                old_log_prob_components=self.zero_components(4),
                reset_before=torch.zeros(4, dtype=torch.bool),
                kind_mask=custom,
            )
        self.assertEqual(buffer.size, 0)
        buffer.append(
            current,
            valid_transitions,
            torch.zeros(4),
            torch.zeros(4),
            old_log_prob_components=self.zero_components(4),
            reset_before=torch.zeros(4, dtype=torch.bool),
        )
        self.assertTrue(torch.all(buffer.kind_mask[0]))

    def test_shot_only_lane_does_not_require_a_wait_duration(self) -> None:
        adapter = MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder())
        current = adapter.reset()
        transitions = adapter.step(
            tuple(SemanticAction.weak(0.5, 0.5) for _ in range(4))
        )
        buffer = RecurrentRolloutBuffer(1, 4, current.schema, torch.zeros((1, 4, 3)))
        kind_mask = torch.zeros((4, 3), dtype=torch.bool)
        kind_mask[:, 1] = True
        wait_mask = torch.zeros((4, len(ActionSpec().wait_choices)), dtype=torch.bool)
        buffer.append(
            current,
            transitions,
            torch.zeros(4),
            torch.zeros(4),
            old_log_prob_components=self.zero_components(4),
            reset_before=torch.zeros(4, dtype=torch.bool),
            kind_mask=kind_mask,
            wait_mask=wait_mask,
        )
        self.assertFalse(torch.any(buffer.wait_mask[0]))

    def test_reset_mask_is_enforced_across_episode_boundaries(self) -> None:
        adapter = MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder())
        current = adapter.reset()
        first_actions = [SemanticAction.wait(1) for _ in range(4)]
        first_actions[3] = SemanticAction.weak(0.5, 0.5)
        first = adapter.step(first_actions)
        buffer = RecurrentRolloutBuffer(2, 4, current.schema, torch.zeros((1, 4, 3)))
        buffer.append(
            current,
            first,
            torch.zeros(4),
            torch.zeros(4),
            old_log_prob_components=self.zero_components(4),
            reset_before=torch.zeros(4, dtype=torch.bool),
        )
        second_observation = adapter.current_observation
        second = adapter.step(tuple(SemanticAction.wait(1) for _ in range(4)))
        with self.assertRaisesRegex(ValueError, "reset-before"):
            buffer.append(
                second_observation,
                second,
                torch.zeros(4),
                torch.zeros(4),
                old_log_prob_components=self.zero_components(4),
                reset_before=torch.zeros(4, dtype=torch.bool),
            )
        reset = torch.zeros(4, dtype=torch.bool)
        reset[3] = True
        buffer.append(
            second_observation,
            second,
            torch.zeros(4),
            torch.zeros(4),
            old_log_prob_components=self.zero_components(4),
            reset_before=reset,
        )
        self.assertTrue(buffer.reset_before[1, 3])
        self.assertEqual(buffer.episode_id[1, 3], buffer.episode_id[0, 3] + 1)

    def test_critic_condition_is_constant_inside_each_episode(self) -> None:
        adapter = MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder())
        current = adapter.reset()
        actions = tuple(SemanticAction.wait(1) for _ in range(4))
        first = adapter.step(actions)
        buffer = RecurrentRolloutBuffer(
            2,
            4,
            current.schema,
            torch.zeros((1, 4, 3)),
            critic_condition_features=1,
        )
        condition = torch.full((4, 1), 0.5)
        buffer.append(
            current,
            first,
            torch.zeros(4),
            torch.zeros(4),
            old_log_prob_components=self.zero_components(4),
            reset_before=torch.zeros(4, dtype=torch.bool),
            critic_condition=condition,
        )
        second_observation = adapter.current_observation
        second = adapter.step(actions)
        changed = condition.clone()
        changed[0] = 0.25
        with self.assertRaisesRegex(ValueError, "continuing episode"):
            buffer.append(
                second_observation,
                second,
                torch.zeros(4),
                torch.zeros(4),
                old_log_prob_components=self.zero_components(4),
                reset_before=torch.zeros(4, dtype=torch.bool),
                critic_condition=changed,
            )
        self.assertEqual(buffer.size, 1)


if __name__ == "__main__":
    unittest.main()
