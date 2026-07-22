from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from irisu_env import PaddedVectorEnv
from irisu_rl import MacroVectorAdapter, SemanticAction, TeacherStateEncoder
from irisu_rl.runtime_identity import ACCEPTED_EXACT_RUNTIME_2026_07_21

ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT / "build-physics-integration-portable" / "libirisu_clone.so"
EXACT = ROOT / "build-physics-integration-exact-multiworld-2" / "irisu-exact-worker"


@unittest.skipUnless(PORTABLE.exists(), "portable integration library not built")
class PortableRlIntegrationTests(unittest.TestCase):
    def test_active_lane_step_does_not_advance_sibling(self) -> None:
        from irisu_env import Action

        with PaddedVectorEnv(2, library_path=PORTABLE) as vector:
            vector.reset(seed=[10, 11])
            before = vector.state_hash()
            observations, _, _, _, _ = vector.step_many([1], [Action.wait(7)])
            after = vector.state_hash()
        self.assertEqual(before[0], after[0])
        self.assertNotEqual(before[1], after[1])
        self.assertEqual(observations[0].tick, 7)

    def test_two_semantic_clicks_fire_and_end_neutral(self) -> None:
        with PaddedVectorEnv(1, library_path=PORTABLE) as vector:
            adapter = MacroVectorAdapter(vector, encoder=TeacherStateEncoder())
            adapter.reset()
            first = adapter.step((SemanticAction.weak(0.5, 0.75),))[0]
            second = adapter.step((SemanticAction.weak(0.5, 0.75),))[0]
        self.assertEqual(first.elapsed_ticks, 2)
        self.assertEqual(second.elapsed_ticks, 2)
        self.assertFalse(first.diagnostics.invalid_action)
        self.assertFalse(second.diagnostics.invalid_action)
        names = first.transition_next_observation.schema.global_features
        left = names.index("left_held")
        right = names.index("right_held")
        self.assertEqual(first.transition_next_observation.global_features[0, left], 0)
        self.assertEqual(first.transition_next_observation.global_features[0, right], 0)

    def test_long_random_rollout_is_reproducible_and_has_no_invalid_actions(self) -> None:
        def collect():
            with PaddedVectorEnv(
                4, library_path=PORTABLE, config={"max_episode_ticks": 500}
            ) as vector:
                adapter = MacroVectorAdapter(vector, encoder=TeacherStateEncoder())
                adapter.reset()
                trace = []
                for step in range(80):
                    actions = tuple(
                        SemanticAction.wait((1, 2, 4, 8, 16, 32)[(step + lane) % 6])
                        if (step + lane) % 4
                        else SemanticAction.strong(((step * 7 + lane) % 100) / 100, 0.7)
                        for lane in range(4)
                    )
                    transitions = adapter.step(actions)
                    trace.extend(
                        (
                            value.lane_id,
                            value.episode_id,
                            value.seed,
                            value.raw_reward,
                            value.elapsed_ticks,
                            value.terminated,
                            value.truncated,
                            value.diagnostics.invalid_action,
                        )
                        for value in transitions
                    )
                return trace, adapter.seed_allocator.cursor

        left, left_cursor = collect()
        right, right_cursor = collect()
        self.assertEqual(left, right)
        self.assertEqual(left_cursor, right_cursor)
        self.assertFalse(any(row[-1] for row in left))
        assignments = {(row[0], row[1]): row[2] for row in left}
        self.assertGreater(len(assignments), 4)
        self.assertEqual(len(assignments), len(set(assignments.values())))


@unittest.skipUnless(PORTABLE.exists() and EXACT.exists(), "exact integration artifacts not built")
class ExactRlIntegrationTests(unittest.TestCase):
    def test_portable_exact_and_dictionary_teacher_tensor_parity(self) -> None:
        with PaddedVectorEnv(1, library_path=PORTABLE) as portable, PaddedVectorEnv(
            1, physics_backend="exact", worker_path=EXACT
        ) as exact:
            portable_observation, _ = portable.reset(seed=[41])
            exact_observation, _ = exact.reset(seed=[41])
            encoder = TeacherStateEncoder()
            portable_tensor = encoder.encode(portable_observation)
            exact_tensor = encoder.encode(exact_observation)
            dictionary_tensor = encoder.encode([portable_observation[0].to_dict()])
        np.testing.assert_array_equal(portable_tensor.global_features, exact_tensor.global_features)
        np.testing.assert_array_equal(portable_tensor.body_features, exact_tensor.body_features)
        np.testing.assert_array_equal(portable_tensor.body_mask, exact_tensor.body_mask)
        np.testing.assert_array_equal(portable_tensor.global_features, dictionary_tensor.global_features)
        np.testing.assert_array_equal(portable_tensor.body_features, dictionary_tensor.body_features)

    def test_accepted_exact_runtime_attests_live_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "absolute"):
            ACCEPTED_EXACT_RUNTIME_2026_07_21.attest(
                Path("build-physics-integration-exact-multiworld-2/irisu-exact-worker")
            )
        report = ACCEPTED_EXACT_RUNTIME_2026_07_21.attest(EXACT)
        self.assertEqual(
            report["build_info"]["worker_executable_sha256"],
            ACCEPTED_EXACT_RUNTIME_2026_07_21.worker_sha256,
        )


if __name__ == "__main__":
    unittest.main()
