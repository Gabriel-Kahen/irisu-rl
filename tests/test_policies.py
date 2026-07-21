from __future__ import annotations

from collections import Counter
import hashlib
import struct
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import (  # noqa: E402
    Action,
    ActionKind,
    IrisuEnv,
    MatcherShotPolicy,
    NativeError,
    RandomPolicy,
    ParameterRange,
    SplitMix64,
    SyncVectorEnv,
    TransferRanges,
    TransferRobustnessEnv,
    find_library,
    randomized_config,
)

try:
    import gymnasium as _gym
    from gymnasium.utils.env_checker import check_env as _check_env
except ImportError:
    _gym = None
    _check_env = None


_FakeBaseEnv = _gym.Env if _gym is not None else object


try:
    LIBRARY = find_library()
except NativeError:
    LIBRARY = None


class PolicyUnitTests(unittest.TestCase):
    def test_splitmix64_has_a_fixed_reference_sequence(self) -> None:
        generator = SplitMix64(0)
        self.assertEqual(
            [generator.next_u64() for _ in range(3)],
            [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F],
        )

    def test_random_policy_is_seeded_and_legal(self) -> None:
        first = RandomPolicy(123)
        second = RandomPolicy(123)
        other = RandomPolicy(124)
        first_actions = [first.act({}) for _ in range(100)]

        self.assertEqual(first_actions, [second.act({}) for _ in range(100)])
        self.assertNotEqual(first_actions, [other.act({}) for _ in range(100)])
        for action in first_actions:
            kind = ActionKind.parse(action.kind)
            self.assertGreaterEqual(action.wait_ticks, 1)
            if kind is not ActionKind.WAIT:
                self.assertGreaterEqual(action.cursor_x, 94.0)
                self.assertLessEqual(action.cursor_x, 514.0)
                self.assertGreaterEqual(action.cursor_y, 260.0)
                self.assertLessEqual(action.cursor_y, 380.0)

    def test_matcher_targets_lower_member_of_nearest_same_color_pair(self) -> None:
        observation = {
            "tick": 8,
            "bodies": [
                {"id": 8, "kind": "piece", "lifecycle": "scripted_falling", "color": 2, "x": 200, "y": 40, "size": 30},
                {"id": 3, "kind": "piece", "lifecycle": "scripted_falling", "color": 2, "x": 210, "y": 90, "size": 48},
                {"id": 9, "kind": "piece", "lifecycle": "scripted_falling", "color": 1, "x": 500, "y": 100, "size": 30},
            ],
        }
        action = MatcherShotPolicy().act(observation)
        self.assertEqual(ActionKind.parse(action.kind), ActionKind.STRONG_SHOT)
        self.assertEqual(action.cursor_x, 210.0)
        self.assertEqual(action.cursor_y, 380.0)

    def test_mechanics_randomization_is_seeded_and_bounded(self) -> None:
        ranges = {
            "gravity_y": ParameterRange(300.0, 400.0),
            "click_cooldown_ticks": ParameterRange(1.0, 3.0, integer=True),
        }
        first = randomized_config(123, ranges)
        self.assertEqual(first, randomized_config(123, ranges))
        self.assertNotEqual(first, randomized_config(124, ranges))
        self.assertLessEqual(300.0, first["gravity_y"])
        self.assertLessEqual(first["gravity_y"], 400.0)
        self.assertIn(first["click_cooldown_ticks"], (1, 2, 3))

    def test_default_randomization_preserves_recovered_nominal_values(self) -> None:
        sampled = randomized_config(123)
        self.assertEqual(sampled["gravity_y"], 160.0)
        self.assertEqual(sampled["scripted_fall_speed"], 0.2)
        self.assertEqual(sampled["weak_projectile_vy"], -250.0)
        self.assertNotIn("click_cooldown_ticks", sampled)


def _public_body(identifier: int, x: float, *, color: int = 1) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "box",
        "lifecycle": "dynamic_fresh",
        "color": color,
        "x": x,
        "y": 50.0,
        "vx": 4.0,
        "vy": 5.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 30.0,
        "chain_id": 0,
        "projectile_hits": 0,
        "age_ticks": 10,
        "remaining_lifetime": 100,
        "rot_timer": 0,
    }


class _PublicContractEnv(_FakeBaseEnv):
    metadata: dict[str, object] = {}
    action_space = None
    observation_space = None
    render_mode = None

    def __init__(self, *, terminal_tick: int = 10_000) -> None:
        self.terminal_tick = terminal_tick
        self.tick = 0
        self.actions: list[Action] = []
        self.reset_options: object = None
        self.closed = False

    def _observation(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "score": self.tick * 10,
            "terminated": self.tick >= self.terminal_tick,
            "truncated": False,
            "bodies": [
                _public_body(9, 100.0 + self.tick),
                _public_body(3, 102.0 + self.tick),
                _public_body(20, 300.0 + self.tick, color=2),
            ],
        }

    def reset(self, *, seed: int | None = None, options: object = None):
        self.tick = 0
        self.actions.clear()
        self.reset_options = options
        return self._observation(), {"seed": seed}

    def step(self, action: Action):
        self.actions.append(action)
        requested = (
            action.wait_ticks
            if ActionKind.parse(action.kind) is ActionKind.WAIT
            else 1
        )
        advanced = min(requested, max(0, self.terminal_tick - self.tick))
        self.tick += advanced
        terminated = self.tick >= self.terminal_tick
        return (
            self._observation(),
            advanced,
            terminated,
            False,
            {
                "events": [{"tick": self.tick, "kind": int(ActionKind.parse(action.kind))}],
                "invalid_action": False,
                "diagnostics": {"tick": self.tick},
            },
        )

    def render(self, mode: str | None = None) -> str:
        return mode or "render"

    def close(self) -> None:
        self.closed = True


class TransferRobustnessUnitTests(unittest.TestCase):
    def test_nominal_wrapper_preserves_public_reset_and_step_results(self) -> None:
        wrapped_native = _PublicContractEnv()
        control = _PublicContractEnv()
        wrapped = TransferRobustnessEnv(wrapped_native, transfer_seed=91)
        self.assertEqual(wrapped.reset(seed=7), control.reset(seed=7))
        wrapped_result = wrapped.step(Action.strong(200.0, 300.0))
        control_result = control.step(Action.strong(200.0, 300.0))
        self.assertEqual(wrapped_result, control_result)
        self.assertEqual(wrapped_native.actions, control.actions)

    def test_large_wait_bulk_forwards_except_for_required_history_tail(self) -> None:
        nominal_native = _PublicContractEnv()
        nominal = TransferRobustnessEnv(nominal_native)
        nominal.reset(seed=0)
        nominal.step(Action.wait(50_000))
        self.assertEqual([action.wait_ticks for action in nominal_native.actions], [50_000])

        delayed_native = _PublicContractEnv(terminal_tick=100_000)
        delayed = TransferRobustnessEnv(
            delayed_native,
            TransferRanges(
                observation_delay_ticks=ParameterRange(3, 3, integer=True)
            ),
        )
        delayed.reset(seed=0)
        observation, *_ = delayed.step(Action.wait(50_000))
        self.assertEqual(
            [action.wait_ticks for action in delayed_native.actions],
            [49_997, 1, 1, 1],
        )
        self.assertEqual(observation["tick"], 49_997)

    def test_fixed_action_delay_cursor_error_and_reward_aggregation(self) -> None:
        native = _PublicContractEnv()
        ranges = TransferRanges(
            action_delay_ticks=ParameterRange(2, 2, integer=True),
            cursor_error_x=ParameterRange(5.0, 5.0),
            cursor_error_y=ParameterRange(-3.0, -3.0),
        )
        wrapped = TransferRobustnessEnv(native, ranges)
        wrapped.reset(seed=1)
        observation, reward, terminated, truncated, info = wrapped.step(
            Action.strong(100.0, 200.0)
        )
        self.assertEqual(
            [ActionKind.parse(action.kind) for action in native.actions],
            [ActionKind.WAIT, ActionKind.STRONG_SHOT],
        )
        self.assertEqual(native.actions[0].wait_ticks, 2)
        self.assertEqual(
            (native.actions[-1].cursor_x, native.actions[-1].cursor_y),
            (105.0, 197.0),
        )
        self.assertEqual(observation["tick"], 3)
        self.assertEqual(reward, 3)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(len(info["events"]), 2)
        self.assertEqual(info["diagnostics"], {"tick": 3})

    def test_observation_delay_noise_and_ambiguous_merge_use_only_public_state(self) -> None:
        native = _PublicContractEnv()
        ranges = TransferRanges(
            observation_delay_ticks=ParameterRange(2, 2, integer=True),
            position_noise=ParameterRange(1.0, 1.0),
            velocity_noise=ParameterRange(-2.0, -2.0),
            merge_distance=3.0,
        )
        wrapped = TransferRobustnessEnv(native, ranges)
        reset_observation, _ = wrapped.reset(seed=4)
        self.assertEqual(len(reset_observation["bodies"]), 2)
        merged = reset_observation["bodies"][0]
        self.assertEqual(merged["id"], 3)
        self.assertEqual(merged["lifecycle"], "ambiguous")
        self.assertEqual((merged["x"], merged["y"]), (102.0, 51.0))
        self.assertEqual((merged["vx"], merged["vy"]), (2.0, 3.0))

        observation, reward, terminated, truncated, info = wrapped.step(Action.wait(3))
        self.assertEqual(observation["tick"], 1)
        self.assertEqual(observation["bodies"][0]["x"], 103.0)
        self.assertEqual(reward, 3)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertNotIn("current_observation", info)
        self.assertEqual(info["diagnostics"], {"tick": 3})

    def test_caller_cannot_mutate_buffered_delayed_observations(self) -> None:
        wrapped = TransferRobustnessEnv(
            _PublicContractEnv(),
            TransferRanges(
                observation_delay_ticks=ParameterRange(1, 1, integer=True)
            ),
        )
        reset_observation, _ = wrapped.reset(seed=0)
        reset_observation["bodies"][0]["x"] = -999.0
        delayed, *_ = wrapped.step(Action.wait())
        self.assertEqual(delayed["tick"], 0)
        self.assertEqual(delayed["bodies"][0]["x"], 100.0)

    def test_dropped_detections_are_bounded_and_seeded(self) -> None:
        all_dropped = TransferRobustnessEnv(
            _PublicContractEnv(),
            TransferRanges(detection_drop_probability=1.0),
        )
        observation, _ = all_dropped.reset(seed=0)
        self.assertEqual(observation["bodies"], [])

        ranges = TransferRanges(
            position_noise=ParameterRange(-2.0, 2.0),
            velocity_noise=ParameterRange(-1.0, 1.0),
            detection_drop_probability=0.4,
        )

        def trace(transfer_seed: int) -> list[object]:
            native = _PublicContractEnv()
            wrapped = TransferRobustnessEnv(native, ranges, transfer_seed=transfer_seed)
            values: list[object] = [wrapped.reset(seed=5)[0]]
            for _ in range(4):
                values.append(wrapped.step(Action.wait())[0])
            return values

        self.assertEqual(trace(19), trace(19))
        self.assertNotEqual(trace(19), trace(20))

    def test_action_jitter_is_seeded_and_cursor_stays_in_client(self) -> None:
        ranges = TransferRanges(
            action_delay_ticks=ParameterRange(0, 3, integer=True),
            cursor_error_x=ParameterRange(-20.0, 20.0),
            cursor_error_y=ParameterRange(-20.0, 20.0),
        )

        def trace(transfer_seed: int) -> list[Action]:
            native = _PublicContractEnv()
            wrapped = TransferRobustnessEnv(native, ranges, transfer_seed=transfer_seed)
            wrapped.reset(seed=1)
            for _ in range(8):
                wrapped.step(Action.strong(1.0, 479.0))
            return native.actions

        first = trace(41)
        self.assertEqual(first, trace(41))
        self.assertNotEqual(first, trace(42))
        for action in first:
            if ActionKind.parse(action.kind) is ActionKind.STRONG_SHOT:
                self.assertLessEqual(0.0, action.cursor_x)
                self.assertLessEqual(action.cursor_x, 640.0)
                self.assertLessEqual(0.0, action.cursor_y)
                self.assertLessEqual(action.cursor_y, 480.0)

    def test_terminal_during_latency_does_not_execute_the_delayed_shot(self) -> None:
        native = _PublicContractEnv(terminal_tick=2)
        wrapped = TransferRobustnessEnv(
            native,
            TransferRanges(action_delay_ticks=ParameterRange(5, 5, integer=True)),
        )
        wrapped.reset(seed=0)
        observation, reward, terminated, truncated, info = wrapped.step(
            Action.weak(100.0, 200.0)
        )
        self.assertEqual(
            [ActionKind.parse(action.kind) for action in native.actions],
            [ActionKind.WAIT],
        )
        self.assertEqual(native.actions[0].wait_ticks, 5)
        self.assertEqual(observation["tick"], 2)
        self.assertEqual(reward, 2)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(len(info["events"]), 1)

    def test_terminal_transition_flushes_observation_delay(self) -> None:
        wrapped = TransferRobustnessEnv(
            _PublicContractEnv(terminal_tick=2),
            TransferRanges(
                observation_delay_ticks=ParameterRange(3, 3, integer=True)
            ),
        )
        wrapped.reset(seed=0)
        observation, reward, terminated, truncated, _ = wrapped.step(
            Action.wait(100)
        )
        self.assertEqual(observation["tick"], 2)
        self.assertTrue(observation["terminated"])
        self.assertEqual(reward, 2)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_transfer_seed_option_is_local_and_ranges_are_validated(self) -> None:
        native = _PublicContractEnv()
        wrapped = TransferRobustnessEnv(native, transfer_seed=7)
        wrapped.reset(seed=2, options={"transfer_seed": 9, "caller": "kept"})
        self.assertEqual(native.reset_options, {"caller": "kept"})
        states = (
            wrapped._action_rng.state,
            wrapped._delay_rng.state,
            wrapped._perception_rng.state,
        )

        def reject_reset(*, seed=None, options=None):
            del seed, options
            raise ValueError("synthetic reset rejection")

        native.reset = reject_reset
        with self.assertRaisesRegex(ValueError, "synthetic reset rejection"):
            wrapped.reset(seed=3, options={"transfer_seed": 10})
        self.assertEqual(
            (
                wrapped._action_rng.state,
                wrapped._delay_rng.state,
                wrapped._perception_rng.state,
            ),
            states,
        )
        with self.assertRaises(ValueError):
            TransferRanges(action_delay_ticks=ParameterRange(0, 1))
        with self.assertRaises(ValueError):
            TransferRanges(detection_drop_probability=1.1)
        with self.assertRaises(ValueError):
            TransferRanges(merge_distance=-1.0)
        with self.assertRaises(ValueError):
            TransferRanges(position_noise=ParameterRange(-1e308, 1e308))


@unittest.skipIf(LIBRARY is None, "build the native shared library before policy tests")
class PolicyIntegrationTests(unittest.TestCase):
    @unittest.skipIf(_check_env is None, "Gymnasium optional dependency is not installed")
    def test_transfer_wrapper_passes_gym_checker_with_ambiguous_bodies(self) -> None:
        ranges = TransferRanges(
            observation_delay_ticks=ParameterRange(0, 2, integer=True),
            position_noise=ParameterRange(-0.25, 0.25),
            velocity_noise=ParameterRange(-0.05, 0.05),
            detection_drop_probability=0.05,
            merge_distance=1_000.0,
        )
        with TransferRobustnessEnv(
            IrisuEnv(library_path=LIBRARY), ranges, transfer_seed=8
        ) as env:
            _check_env(env, skip_render_check=True)
            observation, _ = env.reset(seed=0)
            saw_ambiguous = False
            for _ in range(300):
                observation, *_ = env.step(Action.wait())
                self.assertTrue(env.observation_space.contains(observation))
                saw_ambiguous = saw_ambiguous or any(
                    body["lifecycle"] == "ambiguous"
                    for body in observation["bodies"]
                )
                if saw_ambiguous:
                    break
            self.assertTrue(saw_ambiguous)

    def test_nominal_transfer_wrapper_is_native_trace_equivalent(self) -> None:
        candidate = IrisuEnv(library_path=LIBRARY)
        wrapped = TransferRobustnessEnv(candidate, transfer_seed=123)
        with wrapped, IrisuEnv(library_path=LIBRARY) as control:
            wrapped.reset(seed=29)
            control.reset(seed=29)
            self.assertEqual(candidate.state_hash(), control.state_hash())
            for action in (
                Action.wait(7),
                Action.strong(250.0, 360.0),
                Action.wait(3),
                Action.weak(410.0, 370.0),
                Action.wait(11),
            ):
                wrapped_result = wrapped.step(action)
                control_result = control.step(action)
                self.assertEqual(wrapped_result[1:], control_result[1:])
                self.assertEqual(candidate.state_hash(), control.state_hash())

    def test_delayed_wait_tail_preserves_native_transition_semantics(self) -> None:
        candidate = IrisuEnv(library_path=LIBRARY)
        wrapped = TransferRobustnessEnv(
            candidate,
            TransferRanges(
                observation_delay_ticks=ParameterRange(2, 2, integer=True)
            ),
        )
        with wrapped, IrisuEnv(library_path=LIBRARY) as control:
            wrapped.reset(seed=37)
            control.reset(seed=37)
            wrapped_result = wrapped.step(Action.wait(19))
            control_result = control.step(Action.wait(19))
            self.assertEqual(wrapped_result[0]["tick"], 17)
            self.assertEqual(wrapped_result[1:], control_result[1:])
            self.assertEqual(candidate.state_hash(), control.state_hash())

    def test_matcher_rollout_is_deterministic_and_useful(self) -> None:
        def rollout(env: IrisuEnv) -> dict[str, object]:
            policy = MatcherShotPolicy(shot_period_ticks=2)
            policy.reset(77)
            observation, _ = env.reset(seed=0)
            action_digest = hashlib.sha256()
            trajectory_digest = hashlib.sha256()
            actions: Counter[str] = Counter()
            events: Counter[str] = Counter()
            total_reward = 0
            for _ in range(300):
                action = policy.act(observation)
                kind = ActionKind.parse(action.kind)
                self.assertGreaterEqual(action.wait_ticks, 1)
                if kind is not ActionKind.WAIT:
                    self.assertGreaterEqual(action.cursor_x, 0.0)
                    self.assertLessEqual(action.cursor_x, 640.0)
                    self.assertGreaterEqual(action.cursor_y, 0.0)
                    self.assertLessEqual(action.cursor_y, 480.0)
                action_bytes = struct.pack(
                    "<BddI",
                    int(kind),
                    float(action.cursor_x),
                    float(action.cursor_y),
                    int(action.wait_ticks),
                )
                action_digest.update(action_bytes)
                observation, reward, terminated, truncated, info = env.step(action)
                self.assertFalse(info["invalid_action"])
                state_bytes = struct.pack(
                    "<q??Q", reward, terminated, truncated, env.state_hash()
                )
                trajectory_digest.update(action_bytes + state_bytes)
                actions[kind.name.lower()] += 1
                events.update(event["kind_name"] for event in info["events"])
                total_reward += reward
                if terminated or truncated:
                    break
            return {
                "action_counts": dict(actions),
                "actions_sha256": action_digest.hexdigest(),
                "event_counts": dict(events),
                "final_state_hash": env.state_hash(),
                "steps": sum(actions.values()),
                "total_reward": total_reward,
                "trajectory_sha256": trajectory_digest.hexdigest(),
            }

        with IrisuEnv(library_path=LIBRARY) as env:
            first = rollout(env)
            second = rollout(env)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["steps"], 100)
        self.assertLessEqual(first["steps"], 300)
        self.assertEqual(len(first["actions_sha256"]), 64)
        self.assertEqual(len(first["trajectory_sha256"]), 64)
        self.assertGreater(first["action_counts"].get("wait", 0), 0)
        self.assertGreater(first["event_counts"].get("spawned", 0), 0)
        self.assertGreater(first["final_state_hash"], 0)

    def test_vector_lane_matches_isolated_control_when_other_lane_is_perturbed(self) -> None:
        with SyncVectorEnv(2, library_path=LIBRARY) as vector, IrisuEnv(
            library_path=LIBRARY
        ) as control:
            observations, _ = vector.reset(seed=[31, 31])
            control_observation, _ = control.reset(seed=31)
            self.assertEqual(observations[1], control_observation)
            lane_policy = RandomPolicy(5)
            control_policy = RandomPolicy(5)
            perturbation = RandomPolicy(9, shot_probability=1.0)
            for _ in range(20):
                lane_action = lane_policy.act(observations[1])
                control_action = control_policy.act(control_observation)
                self.assertEqual(lane_action, control_action)
                vector_result = vector.step(
                    [perturbation.act(observations[0]), lane_action]
                )
                control_result = control.step(control_action)
                observations = vector_result[0]
                control_observation = control_result[0]
                lane_result = tuple(values[1] for values in vector_result)
                self.assertEqual(lane_result, control_result)
                self.assertEqual(vector.state_hash()[1], control.state_hash())
            self.assertEqual(lane_policy.rng_state, control_policy.rng_state)
            self.assertNotEqual(vector.state_hash()[0], vector.state_hash()[1])


if __name__ == "__main__":
    unittest.main()
