from __future__ import annotations

import types
import unittest

import numpy as np

from irisu_env import ActionKind
from irisu_rl.actions import SemanticAction
from irisu_rl.rollout_buffer import RolloutBuffer
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.seeds import SeedAllocator
from irisu_rl.vector_adapter import MacroVectorAdapter


def observation(
    tick: int,
    *,
    gauge: int = 100,
    gauge_max: int = 1000,
    terminated: bool = False,
    truncated: bool = False,
):
    return types.SimpleNamespace(
        tick=tick,
        score=tick,
        gauge=gauge,
        gauge_max=gauge_max,
        qualifying_clear_count=0,
        level=1,
        active_colors=3,
        spawn_interval_ticks=50,
        highest_chain=0,
        left_held=False,
        right_held=False,
        terminated=terminated,
        truncated=truncated,
        body_count=0,
        bodies=(),
    )


class FakeActiveVector:
    num_envs = 4

    def __init__(self) -> None:
        self.ticks = [0] * self.num_envs
        self.seeds = [0] * self.num_envs
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def reset(self, *, seed):
        self.seeds = list(seed)
        self.ticks = [0] * self.num_envs
        return [observation(0) for _ in self.ticks], [
            {"seed": value, "config_hash": 99} for value in seed
        ]

    def _step(self, indices, actions):
        output, rewards, terminated, truncated, infos = [], [], [], [], []
        self.calls.append(
            (tuple(indices), tuple(int(action.kind) for action in actions))
        )
        for lane, action in zip(indices, actions):
            delta = action.wait_ticks if action.kind == ActionKind.WAIT else 1
            self.ticks[lane] += delta
            terminal = lane == 3 and action.kind != ActionKind.WAIT
            output.append(observation(self.ticks[lane], terminated=terminal))
            rewards.append(delta)
            terminated.append(terminal)
            truncated.append(False)
            infos.append(
                {"events": [object()], "invalid_action": False, "config_hash": 99}
            )
        return output, rewards, terminated, truncated, infos

    def step(self, actions):
        return self._step(range(self.num_envs), actions)

    def step_many(self, indices, actions):
        return self._step(indices, actions)

    def reset_many(self, indices, *, seeds):
        result = []
        for lane, seed in zip(indices, seeds):
            self.seeds[lane] = seed
            self.ticks[lane] = 0
            result.append(observation(0))
        return result


class FakeTruncatingVector(FakeActiveVector):
    def _step(self, indices, actions):
        output, rewards, terminated, truncated, infos = [], [], [], [], []
        self.calls.append(
            (tuple(indices), tuple(int(action.kind) for action in actions))
        )
        for lane, action in zip(indices, actions):
            is_wait_cut = lane == 1 and action.kind == ActionKind.WAIT
            is_press_cut = lane == 2 and action.kind != ActionKind.WAIT
            delta = (
                2
                if is_wait_cut
                else (action.wait_ticks if action.kind == ActionKind.WAIT else 1)
            )
            self.ticks[lane] += delta
            output.append(
                observation(self.ticks[lane], truncated=is_wait_cut or is_press_cut)
            )
            rewards.append(delta)
            terminated.append(False)
            truncated.append(is_wait_cut or is_press_cut)
            infos.append({"events": (), "invalid_action": False, "config_hash": 99})
        return output, rewards, terminated, truncated, infos


class AdapterTests(unittest.TestCase):
    def test_mixed_macros_release_only_shots_and_preserve_final_observation(
        self,
    ) -> None:
        env = FakeActiveVector()
        adapter = MacroVectorAdapter(
            env, encoder=TeacherStateEncoder(), seed_allocator=SeedAllocator(key=3)
        )
        adapter.reset()
        transitions = adapter.step(
            (
                SemanticAction.wait(1),
                SemanticAction.wait(8),
                SemanticAction.weak(0.25, 0.5),
                SemanticAction.strong(0.75, 0.5),
            )
        )
        self.assertEqual([value.elapsed_ticks for value in transitions], [1, 8, 2, 1])
        self.assertEqual(transitions[2].primitive_trace, ("press", "release"))
        self.assertEqual(transitions[2].raw_reward, 2)
        self.assertEqual(transitions[2].start_gauge, 100)
        self.assertEqual(transitions[2].end_gauge, 100)
        self.assertEqual(transitions[2].gauge_max, 1000)
        self.assertEqual(transitions[3].primitive_trace, ("press",))
        self.assertTrue(transitions[3].terminated)
        self.assertTrue(transitions[3].macro_interrupted)
        self.assertFalse(transitions[3].bootstrap_mask)
        self.assertEqual(env.calls[1][0], (2,))
        final = transitions[3].final_observation.global_features.copy()
        adapter.step((SemanticAction.wait(1),) * 4)
        np.testing.assert_array_equal(
            transitions[3].final_observation.global_features, final
        )
        self.assertNotEqual(transitions[3].seed, env.seeds[3])

    def test_gauge_maximum_drift_poisons_after_backend_mutation(self) -> None:
        class GaugeDriftVector(FakeActiveVector):
            def _step(self, indices, actions):
                result = super()._step(indices, actions)
                observations = result[0]
                observations[0].gauge_max = 999
                return result

        adapter = MacroVectorAdapter(
            GaugeDriftVector(), encoder=TeacherStateEncoder()
        )
        adapter.reset()
        with self.assertRaisesRegex(RuntimeError, "gauge maximum changed"):
            adapter.step((SemanticAction.wait(1),) * 4)
        self.assertTrue(adapter.poisoned)

        env = FakeActiveVector()
        malformed = MacroVectorAdapter(env, encoder=TeacherStateEncoder())
        malformed.reset()
        original = env._step

        def inexact_gauge(indices, actions):
            result = original(indices, actions)
            result[0][0].gauge = 99.5
            return result

        env._step = inexact_gauge  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            malformed.step((SemanticAction.wait(1),) * 4)
        self.assertTrue(malformed.poisoned)

    def test_terminal_gauge_boundary_excludes_autoreset_state(self) -> None:
        class TerminalGaugeVector(FakeActiveVector):
            def _step(self, indices, actions):
                result = super()._step(indices, actions)
                for raw, terminal in zip(result[0], result[2]):
                    if terminal:
                        raw.gauge = 1
                return result

            def reset_many(self, indices, *, seeds):
                reset = super().reset_many(indices, seeds=seeds)
                for raw in reset:
                    raw.gauge = 900
                return reset

        adapter = MacroVectorAdapter(
            TerminalGaugeVector(), encoder=TeacherStateEncoder()
        )
        adapter.reset()
        transition = adapter.step(
            (
                SemanticAction.wait(1),
                SemanticAction.wait(1),
                SemanticAction.wait(1),
                SemanticAction.strong(0.5, 0.5),
            )
        )[3]
        self.assertTrue(transition.terminated)
        self.assertEqual(transition.start_gauge, 100)
        self.assertEqual(transition.end_gauge, 1)
        self.assertAlmostEqual(
            float(transition.transition_next_observation.global_features[0, 2]),
            0.001,
        )
        self.assertAlmostEqual(
            float(transition.next_policy_observation.global_features[0, 2]), 0.9
        )

    def test_preflight_failure_does_not_advance_and_backend_failure_poisons(
        self,
    ) -> None:
        env = FakeActiveVector()
        adapter = MacroVectorAdapter(env, encoder=TeacherStateEncoder())
        adapter.reset()
        with self.assertRaises(ValueError):
            adapter.step((SemanticAction.wait(101),) * 4)
        self.assertEqual(env.ticks, [0, 0, 0, 0])

        def fail(*_args, **_kwargs):
            raise OSError("transport")

        env.step = fail  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            adapter.step((SemanticAction.wait(1),) * 4)
        self.assertTrue(adapter.poisoned)

    def test_rollout_buffer_copies_complete_transition(self) -> None:
        adapter = MacroVectorAdapter(FakeActiveVector(), encoder=TeacherStateEncoder())
        current = adapter.reset()
        transition = adapter.step((SemanticAction.wait(2),) * 4)[0]
        buffer = RolloutBuffer(2, current.schema)
        self.assertEqual(buffer.append(transition), 0)
        saved = buffer.observations_global.copy()
        transition.observation.global_features[:] = -10
        np.testing.assert_array_equal(buffer.observations_global, saved)
        self.assertEqual(buffer.size, 1)
        buffer.seal(adapter.current_observation)
        self.assertEqual(buffer.rollout_end_observation.global_features.shape[0], 4)
        with self.assertRaisesRegex(RuntimeError, "sealed"):
            buffer.append(transition)
        with self.assertRaisesRegex(ValueError, "positive"):
            RolloutBuffer(2.5, current.schema)

    def test_encoder_schema_change_after_dispatch_poisons_adapter(self) -> None:
        class DriftingEncoder:
            def __init__(self) -> None:
                self.base = TeacherStateEncoder()
                self.calls = 0

            def encode(self, observations):
                batch = self.base.encode(observations)
                self.calls += 1
                if self.calls > 1:
                    batch.schema = types.SimpleNamespace(
                        capacity=batch.schema.capacity,
                        global_features=batch.schema.global_features,
                        body_features=batch.schema.body_features,
                    )
                return batch

        adapter = MacroVectorAdapter(FakeActiveVector(), encoder=DriftingEncoder())
        adapter.reset()
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            adapter.step((SemanticAction.wait(1),) * 4)
        self.assertTrue(adapter.poisoned)

    def test_transform_receives_lane_phase_context_and_current_isolation(self) -> None:
        calls = []

        def transform(inputs):
            calls.append(
                tuple(
                    (value.lane_id, value.phase, value.episode_reset)
                    for value in inputs
                )
            )
            return [value.raw_observation for value in inputs]

        adapter = MacroVectorAdapter(
            FakeActiveVector(),
            encoder=TeacherStateEncoder(),
            observation_transform=transform,
        )
        reset = adapter.reset()
        reset.global_features[:] = -100
        transitions = adapter.step(
            (
                SemanticAction.wait(1),
                SemanticAction.wait(1),
                SemanticAction.weak(0.5, 0.5),
                SemanticAction.strong(0.5, 0.5),
            )
        )
        self.assertFalse(np.all(transitions[0].observation.global_features == -100))
        self.assertEqual(calls[0], tuple((lane, "reset", True) for lane in range(4)))
        self.assertEqual(calls[2], ((2, "release", False),))
        self.assertEqual(calls[3], ((3, "reset", True),))

    def test_malformed_backend_batch_poisons_adapter(self) -> None:
        env = FakeActiveVector()
        adapter = MacroVectorAdapter(env, encoder=TeacherStateEncoder())
        adapter.reset()
        original = env.step

        def short(actions):
            result = original(actions)
            return tuple(values[:-1] for values in result)

        env.step = short  # type: ignore[method-assign]
        with self.assertRaisesRegex(RuntimeError, "poisoned"):
            adapter.step((SemanticAction.wait(1),) * 4)
        self.assertTrue(adapter.poisoned)

    def test_truncation_bootstrap_distinguishes_neutral_wait_from_held_press(
        self,
    ) -> None:
        adapter = MacroVectorAdapter(
            FakeTruncatingVector(), encoder=TeacherStateEncoder()
        )
        adapter.reset()
        transitions = adapter.step(
            (
                SemanticAction.wait(1),
                SemanticAction.wait(8),
                SemanticAction.weak(0.5, 0.5),
                SemanticAction.wait(1),
            )
        )
        self.assertTrue(transitions[1].macro_interrupted)
        self.assertTrue(transitions[1].bootstrap_mask)
        self.assertTrue(transitions[2].macro_interrupted)
        self.assertFalse(transitions[2].bootstrap_mask)


if __name__ == "__main__":
    unittest.main()
