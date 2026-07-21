from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import (  # noqa: E402
    Action,
    ExactWorkerNotFoundError,
    NativeError,
    PaddedVectorEnv,
    RandomPolicy,
    find_exact_worker,
)


try:
    EXACT_WORKER = find_exact_worker()
except ExactWorkerNotFoundError:
    EXACT_WORKER = None


@unittest.skipIf(EXACT_WORKER is None, "build the exact physics worker")
class ExactPaddedStressTests(unittest.TestCase):
    def test_repeated_episode_resets_replace_worker_and_avoid_r58_crash(self) -> None:
        """Regression for b2PairManager::Commit crashing on the sixth reset."""

        base = 0x52574157
        expected_lengths = (470, 798, 3932, 884, 784, 1053)
        process_ids: list[int] = []
        with PaddedVectorEnv(
            1, physics_backend="exact", worker_path=EXACT_WORKER
        ) as vector:
            for episode, expected_length in enumerate(expected_lengths):
                seed = 41 + episode * 2
                policy = RandomPolicy(base + episode * 2, max_wait_ticks=1)
                if episode == 0:
                    vector.reset(seed=[seed])
                else:
                    vector.reset_at(0, seed=seed)
                process_ids.append(vector.envs[0].build_info()["worker_pid"])
                for frame in range(10_000):
                    result = vector.step([policy.act({})])
                    if result[2][0] or result[3][0]:
                        break
                self.assertEqual(frame + 1, expected_length)
        self.assertEqual(len(set(process_ids)), len(process_ids))

    def test_lazy_event_view_is_bound_to_originating_worker(self) -> None:
        with PaddedVectorEnv(
            1, physics_backend="exact", worker_path=EXACT_WORKER
        ) as vector:
            vector.reset(seed=[41])
            first = vector.step([Action.wait()])[-1][0]["events"]
            materialized = [event.to_dict() for event in first.materialize()]
            stale = vector.step([Action.wait()])[-1][0]["events"]
            origin = vector.envs[0]._client

            vector.reset_at(0, seed=42)
            self.assertIsNot(vector.envs[0]._client, origin)
            vector.step([Action.wait()])
            self.assertEqual(
                [event.to_dict() for event in first.materialize()], materialized
            )
            with self.assertRaisesRegex(NativeError, "events expired"):
                stale.materialize()


if __name__ == "__main__":
    unittest.main()
