from __future__ import annotations

import os
import sys
import unittest
from contextlib import ExitStack
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import (  # noqa: E402
    Action,
    ExactWorkerNotFoundError,
    NativeError,
    PaddedVectorEnv,
    SyncVectorEnv,
    ThreadVectorEnv,
    find_exact_worker,
)


try:
    EXACT_WORKER = find_exact_worker()
except ExactWorkerNotFoundError:
    EXACT_WORKER = None


def _observation(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else value.to_dict()


def _step_result(value: tuple[Any, ...]) -> tuple[Any, ...]:
    observations, rewards, terminated, truncated, infos = value
    canonical_infos = []
    for info in infos:
        diagnostics = info["diagnostics"]
        if not isinstance(diagnostics, dict):
            diagnostics = diagnostics.diagnostics()
        canonical_infos.append(
            {
                "events": [
                    event if isinstance(event, dict) else event.to_dict()
                    for event in info["events"]
                ],
                "invalid_action": bool(info["invalid_action"]),
                "config_hash": int(info["config_hash"]),
                "diagnostics": diagnostics,
            }
        )
    return (
        [_observation(value) for value in observations],
        list(rewards),
        list(terminated),
        list(truncated),
        canonical_infos,
    )


def _actions(round_index: int) -> list[Action]:
    actions = []
    for lane in range(4):
        phase = (round_index + 2 * lane) % 8
        x = 150 + (round_index * 37 + lane * 83) % 300
        y = 160 + (round_index * 29 + lane * 47) % 190
        if phase == 0:
            actions.append(Action.weak(x, y))
        elif phase == 3:
            actions.append(Action.strong(x, y))
        else:
            actions.append(Action.wait(3 + (round_index + lane) % 7))
    return actions


@unittest.skipUnless(
    EXACT_WORKER and os.name == "posix",
    "requires the exact worker",
)
class ExactVectorSnapshotConcurrencyTests(unittest.TestCase):
    def test_vector_backends_restore_contact_future_and_rollback_atomically(
        self,
    ) -> None:
        config = {
            "gauge_initial": 40_000,
            "passive_gauge_decay_per_tick": 0,
            "spawn_interval_ticks": 5,
            "piece_life_ticks": 5_000,
            # The saved future crosses this boundary at different rounds per
            # lane, exercising truncation and frozen terminal state as well as
            # active contact/scoring state.
            "max_episode_ticks": 430,
            "rotten_penalty": 0,
        }
        common = {
            "physics_backend": "exact",
            "worker_path": EXACT_WORKER,
            "config": config,
        }
        with ExitStack() as stack:
            vectors = (
                stack.enter_context(SyncVectorEnv(4, **common)),
                stack.enter_context(ThreadVectorEnv(4, workers=2, **common)),
                stack.enter_context(PaddedVectorEnv(4, workers=2, **common)),
            )

            resets = []
            for vector in vectors:
                observations, infos = vector.reset(seed=[41] * 4)
                resets.append(
                    (
                        [_observation(value) for value in observations],
                        [int(info["config_hash"]) for info in infos],
                    )
                )
            self.assertEqual(resets[1:], resets[:1] * 2)

            event_kinds: set[str] = set()

            def step_all(round_index: int) -> tuple[Any, ...]:
                results = [
                    _step_result(vector.step(_actions(round_index)))
                    for vector in vectors
                ]
                self.assertEqual(results[1:], results[:1] * 2)
                for info in results[0][4]:
                    event_kinds.update(event["kind_name"] for event in info["events"])
                return results[0]

            for round_index in range(64):
                step_all(round_index)

            checkpoint = vectors[0].clone_state()
            checkpoint_hashes = vectors[0].state_hash()
            self.assertEqual(len(set(checkpoint_hashes)), 4)
            for vector in vectors[1:]:
                self.assertEqual(vector.clone_state(), checkpoint)
                self.assertEqual(vector.state_hash(), checkpoint_hashes)

            # From here onward, require the checkpointed future itself (and
            # then its restored replay) to exercise contacts and scoring.
            event_kinds.clear()
            expected_future = [step_all(index) for index in range(64, 96)]
            expected_final = vectors[0].clone_state()
            expected_final_hashes = vectors[0].state_hash()
            for vector in vectors[1:]:
                self.assertEqual(vector.clone_state(), expected_final)
                self.assertEqual(vector.state_hash(), expected_final_hashes)

            restored = []
            for vector in vectors:
                restored.append(
                    [_observation(value) for value in vector.restore_state(checkpoint)]
                )
                self.assertEqual(vector.clone_state(), checkpoint)
                self.assertEqual(vector.state_hash(), checkpoint_hashes)
            self.assertEqual(restored[1:], restored[:1] * 2)

            for round_index, expected in zip(range(64, 96), expected_future):
                self.assertEqual(step_all(round_index), expected)
            for vector in vectors:
                self.assertEqual(vector.clone_state(), expected_final)
                self.assertEqual(vector.state_hash(), expected_final_hashes)

            self.assertIn("contact", event_kinds)
            self.assertIn("score_changed", event_kinds)
            self.assertEqual(expected_future[-1][2], [False] * 4)
            self.assertEqual(expected_future[-1][3], [True] * 4)

            corrupted = bytearray(checkpoint[2])
            corrupted[-1] ^= 1
            invalid_target = (
                checkpoint[0],
                checkpoint[1],
                bytes(corrupted),
                checkpoint[3],
            )
            before = vectors[0].clone_state()
            before_hashes = vectors[0].state_hash()
            for vector in vectors:
                with self.assertRaisesRegex(NativeError, "checksum"):
                    vector.restore_state(invalid_target)
                self.assertEqual(vector.clone_state(), before)
                self.assertEqual(vector.state_hash(), before_hashes)

            # Rollback rebuilt some lanes in fresh workers. Their next complete
            # transition must still equal the serial exact oracle.
            step_all(96)


if __name__ == "__main__":
    unittest.main()
