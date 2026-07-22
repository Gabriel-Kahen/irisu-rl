from __future__ import annotations

import hashlib
import os
import sys
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import (  # noqa: E402
    Action,
    ExactWorkerNotFoundError,
    NativeError,
    PaddedVectorEnv,
    find_exact_worker,
)
from irisu_env.rollout import ExactActorRolloutPool  # noqa: E402


try:
    EXACT_WORKER = find_exact_worker()
except ExactWorkerNotFoundError:
    EXACT_WORKER = None


CONFIG = {
    "gauge_initial": 1_000_000_000_000,
    "gauge_max": 1_000_000_000_000,
    "passive_gauge_decay_per_tick": 0,
    "qualifying_clears_per_level": 0xFFFFFFFF,
    "rotten_penalty": 0,
    "max_episode_ticks": 100_000,
}


class StatePolicy:
    def __init__(self, lane: int) -> None:
        self.lane = lane

    def act(self, observation: Any) -> Action:
        tick = int(observation.tick)
        body_count = int(observation.body_count)
        phase = (tick + body_count + 3 * self.lane) % 12
        x = 150.0 + float((tick * 37 + self.lane * 83) % 300)
        y = 160.0 + float((tick * 29 + self.lane * 47) % 190)
        if phase == 0:
            return Action.weak(x, y)
        if phase == 4:
            return Action.strong(x, y)
        return Action.wait(1)


def digest(payloads: list[bytes]) -> str:
    value = hashlib.sha256()
    for payload in payloads:
        value.update(payload)
    return value.hexdigest()


@unittest.skipUnless(EXACT_WORKER and os.name == "posix", "requires exact worker")
class ExactActorRolloutTests(unittest.TestCase):
    def test_rollout_payload_state_and_delayed_events_match_sync_vector(self) -> None:
        lanes = 4
        horizon = 48
        sync_payloads: list[list[bytes]] = [[] for _ in range(lanes)]
        sync_actions: list[list[Action]] = [[] for _ in range(lanes)]
        sync_events: list[list[list[dict[str, Any]]]] = [
            [] for _ in range(lanes)
        ]

        with PaddedVectorEnv(
            lanes,
            physics_backend="exact",
            worker_path=EXACT_WORKER,
            workers=lanes,
            config=CONFIG,
        ) as sync:
            observations, _ = sync.reset(seed=41)
            reset_observations = [value.to_dict() for value in observations]
            policies = [StatePolicy(lane) for lane in range(lanes)]
            for _ in range(horizon):
                actions = [
                    policy.act(observation)
                    for policy, observation in zip(policies, observations)
                ]
                observations, _, _, _, infos = sync.step(actions)
                for lane, (env, info) in enumerate(zip(sync.envs, infos)):
                    sync_actions[lane].append(actions[lane])
                    raw = env._raw_observation
                    assert raw is not None
                    sync_payloads[lane].append(raw[0])
                    sync_events[lane].append(
                        [event.to_dict() for event in info["events"].materialize()]
                    )
            sync_hashes = sync.state_hash()

        with ExactActorRolloutPool(
            lanes,
            worker_path=EXACT_WORKER,
            workers=lanes,
            config=CONFIG,
            event_mode="full",
        ) as actors:
            actors.reset(seed=41)
            rollouts = actors.collect(
                [StatePolicy(lane) for lane in range(lanes)], horizon
            )
            self.assertEqual(actors.state_hash(), sync_hashes)
            for lane, rollout in enumerate(rollouts):
                self.assertEqual(rollout.lane, lane)
                self.assertEqual(
                    rollout.initial_observation.to_dict(), reset_observations[lane]
                )
                self.assertEqual(
                    [step.action for step in rollout.steps], sync_actions[lane]
                )
                self.assertEqual(rollout.final_state_hash, sync_hashes[lane])
                self.assertEqual(
                    digest([step.payload for step in rollout.steps]),
                    digest(sync_payloads[lane]),
                )
                # These details are decoded only now, after the worker has
                # advanced well past every event generation.
                self.assertEqual(
                    [
                        [event.to_dict() for event in step.events.materialize()]
                        for step in rollout.steps
                    ],
                    sync_events[lane],
                )

    def test_snapshot_replay_and_failure_order_are_deterministic(self) -> None:
        with ExactActorRolloutPool(
            3,
            worker_path=EXACT_WORKER,
            workers=3,
            config=CONFIG,
        ) as actors:
            actors.reset(seed=[41, 42, 43])
            actors.collect([StatePolicy(lane) for lane in range(3)], 20)
            checkpoint = actors.clone_state()
            expected = actors.collect([StatePolicy(lane) for lane in range(3)], 12)
            detached_initial = expected[0].initial_observation.to_dict()
            expected_hashes = actors.state_hash()

            actors.restore_state(checkpoint)
            actual = actors.collect([StatePolicy(lane) for lane in range(3)], 12)
            self.assertEqual(actors.state_hash(), expected_hashes)
            self.assertEqual(
                [[step.payload for step in lane.steps] for lane in actual],
                [[step.payload for step in lane.steps] for lane in expected],
            )
            actors.collect([StatePolicy(lane) for lane in range(3)], 1)
            self.assertEqual(expected[0].initial_observation.to_dict(), detached_initial)

            before_failure = actors.state_hash()

            shared = StatePolicy(0)
            with self.assertRaisesRegex(ValueError, "distinct lane-local"):
                actors.collect([shared, shared, shared], 1)
            self.assertEqual(actors.state_hash(), before_failure)

            def lane_zero(_: Any) -> Action:
                raise ValueError("lane zero failure")

            def lane_one(_: Any) -> Action:
                raise RuntimeError("lane one failure")

            with self.assertRaisesRegex(ValueError, "lane zero failure"):
                actors.collect([lane_zero, lane_one, StatePolicy(2)], 1)
            # Successful sibling work is committed, but the lowest failing lane
            # is deterministic and every future has drained before returning.
            after_failure = actors.state_hash()
            self.assertEqual(after_failure[:2], before_failure[:2])
            self.assertNotEqual(after_failure[2], before_failure[2])
            actors.collect([StatePolicy(lane) for lane in range(3)], 1)

    def test_failed_reset_invalidates_rollout_until_restore(self) -> None:
        class FailingReset:
            def __init__(self, target: Any) -> None:
                self.target = target

            def __getattr__(self, name: str) -> Any:
                return getattr(self.target, name)

            def reset_typed(self, seed: int) -> Any:
                del seed
                raise RuntimeError("injected reset failure")

        with ExactActorRolloutPool(
            2, worker_path=EXACT_WORKER, workers=2, config=CONFIG
        ) as actors:
            actors.reset(seed=41)
            actors.collect([StatePolicy(0), StatePolicy(1)], 4)
            checkpoint = actors.clone_state()
            expected_hashes = actors.state_hash()
            original_envs = actors._envs
            actors._envs = (original_envs[0], FailingReset(original_envs[1]))
            try:
                with self.assertRaisesRegex(RuntimeError, "injected reset failure"):
                    actors.reset(seed=90)
            finally:
                actors._envs = original_envs

            with self.assertRaisesRegex(RuntimeError, "reset must be called"):
                actors.collect([StatePolicy(0), StatePolicy(1)], 1)
            actors.restore_state(checkpoint)
            self.assertEqual(actors.state_hash(), expected_hashes)
            self.assertTrue(
                all(
                    rollout.steps
                    for rollout in actors.collect(
                        [StatePolicy(0), StatePolicy(1)], 1
                    )
                )
            )

    def test_terminal_lane_stops_until_explicit_reset(self) -> None:
        calls = 0

        def wait_policy(_: Any) -> Action:
            nonlocal calls
            calls += 1
            return Action.wait(1)

        config = dict(CONFIG)
        config["max_episode_ticks"] = 3
        with ExactActorRolloutPool(
            1,
            worker_path=EXACT_WORKER,
            config=config,
        ) as actors:
            actors.reset(seed=41)
            rollout = actors.collect([wait_policy], 20)[0]
            self.assertEqual(len(rollout.steps), 3)
            self.assertTrue(rollout.steps[-1].truncated)
            self.assertEqual(calls, 3)
            terminal_hash = actors.state_hash()
            terminal_snapshot = actors.clone_state()

            frozen = actors.collect([wait_policy], 20)[0]
            self.assertEqual(frozen.steps, ())
            self.assertEqual(actors.state_hash(), terminal_hash)
            self.assertEqual(calls, 3)

            actors.reset_at(0, seed=42)
            restarted = actors.collect([wait_policy], 1)[0]
            self.assertEqual(len(restarted.steps), 1)
            self.assertFalse(restarted.steps[0].truncated)
            self.assertEqual(calls, 4)

            actors.restore_state(terminal_snapshot)
            restored_terminal = actors.collect([wait_policy], 1)[0]
            self.assertEqual(restored_terminal.steps, ())
            self.assertEqual(actors.state_hash(), terminal_hash)
            self.assertEqual(calls, 4)

    def test_terminal_state_survives_full_event_fetch_failure(self) -> None:
        class FailingFetch:
            def __init__(self, target: Any) -> None:
                self.target = target

            def __getattr__(self, name: str) -> Any:
                return getattr(self.target, name)

            def fetch_padded_events_raw(self, generation: int, count: int) -> bytes:
                del generation, count
                raise NativeError("injected event fetch failure")

        calls = 0

        def wait_policy(_: Any) -> Action:
            nonlocal calls
            calls += 1
            return Action.wait(1)

        config = dict(CONFIG)
        config["max_episode_ticks"] = 1
        with ExactActorRolloutPool(
            1,
            worker_path=EXACT_WORKER,
            config=config,
            event_mode="full",
        ) as actors:
            actors.reset(seed=41)
            original_envs = actors._envs
            actors._envs = (FailingFetch(original_envs[0]),)
            try:
                with self.assertRaisesRegex(NativeError, "event fetch failure"):
                    actors.collect([wait_policy], 2)
            finally:
                actors._envs = original_envs
            self.assertEqual(calls, 1)
            self.assertEqual(actors.collect([wait_policy], 1)[0].steps, ())
            self.assertEqual(calls, 1)

    def test_count_mode_retains_counts_without_fetching_details(self) -> None:
        with ExactActorRolloutPool(
            1,
            worker_path=EXACT_WORKER,
            config=CONFIG,
            event_mode="count",
        ) as actors:
            actors.reset(seed=41)
            rollout = actors.collect([StatePolicy(0)], 16)[0]
            populated = next(step.events for step in rollout.steps if len(step.events))
            with self.assertRaisesRegex(NativeError, "not retained"):
                populated.materialize()


if __name__ == "__main__":
    unittest.main()
