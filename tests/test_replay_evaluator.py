from __future__ import annotations

import importlib.util
import struct
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_rpy", ROOT / "tools" / "evaluate-rpy.py"
)
assert SPEC and SPEC.loader
evaluate_rpy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate_rpy
SPEC.loader.exec_module(evaluate_rpy)


class Frame:
    def __init__(self, word: int, left: bool, right: bool, x: int, y: int) -> None:
        self.raw_word = word
        self.left = left
        self.right = right
        self.x = x
        self.y = y


class FakeEnv:
    last_seed: int | None = None

    def __init__(self, **_: object) -> None:
        self.tick = 0
        self.score = 0
        self.level = 1

    def __enter__(self) -> FakeEnv:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    @property
    def build_info(self) -> dict[str, object]:
        return {"implementation": "synthetic-test"}

    def _observation(self) -> dict[str, object]:
        return {
            "tick": self.tick,
            "score": self.score,
            "level": self.level,
            "terminated": False,
            "truncated": False,
        }

    def reset(self, *, seed: int) -> tuple[dict[str, object], dict[str, int]]:
        type(self).last_seed = seed
        return self._observation(), {"seed": seed}

    def step(self, action: object) -> tuple[dict[str, object], int, bool, bool, dict[str, object]]:
        self.tick += 1
        kind = evaluate_rpy.ActionKind.parse(action.kind)
        reward = {
            evaluate_rpy.ActionKind.WAIT: 0,
            evaluate_rpy.ActionKind.WEAK_SHOT: 1,
            evaluate_rpy.ActionKind.STRONG_SHOT: 2,
            evaluate_rpy.ActionKind.BOTH_SHOTS: 3,
        }[kind]
        self.score += reward
        return self._observation(), reward, False, False, {"events": [], "invalid_action": False}

    def clone_state(self) -> bytes:
        return struct.pack("<qq", self.tick, self.score)

    def config_hash(self) -> int:
        return 0xCAFE

    def state_hash(self) -> int:
        return 0x1000 + self.tick + self.score


class DiagnosticEnv(FakeEnv):
    def step(self, action: object) -> tuple[dict[str, object], int, bool, bool, dict[str, object]]:
        observation, reward, terminated, truncated, info = super().step(action)
        info["diagnostics"] = {
            "config_hash": 0xCAFE,
            "finish_call_count": 2,
            "terminal_metadata_recorded": True,
            "recorded_final_score": 123,
            "recorded_final_level": 4,
            "recorded_final_highest_chain": 5,
            "latest_final_score": 999,
            "latest_final_level": 100,
            "latest_final_highest_chain": 8,
        }
        return observation, reward, terminated, truncated, info


class TerminalCauseEnv(FakeEnv):
    def step(self, action: object) -> tuple[dict[str, object], int, bool, bool, dict[str, object]]:
        observation, reward, _, truncated, info = super().step(action)
        terminated = self.tick >= 2
        observation["terminated"] = terminated
        info["events"] = [
            {
                "tick": self.tick,
                "kind_name": "rotten" if self.tick == 1 else "game_over",
                "a": 17 if self.tick == 1 else 0,
                "b": 0,
                "value": 0,
                "detail": "test terminal cause",
            }
        ]
        return observation, reward, terminated, truncated, info


class TimelineEnv(FakeEnv):
    last_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        type(self).last_kwargs = kwargs
        self.gauge = 100
        self.clears = 0
        self.highest_chain = 0
        self.physics_backend = "exact"

    def exact_library_provenance(self) -> dict[str, object]:
        return {"sha256": "a" * 64, "mapped_path": "/verified/exact.so"}

    def _observation(self) -> dict[str, object]:
        result = super()._observation()
        result.update(
            {
                "gauge": self.gauge,
                "qualifying_clear_count": self.clears,
                "highest_chain": self.highest_chain,
            }
        )
        return result

    def step(self, action: object) -> tuple[dict[str, object], int, bool, bool, dict[str, object]]:
        del action
        self.tick += 1
        self.gauge += 9
        self.clears += 1
        self.level = 2
        self.score += 8
        self.highest_chain = 2
        events = [
            {
                "tick": self.tick,
                "kind_name": "gauge_changed",
                "a": 7,
                "value": 10,
                "detail": "normal burst landing",
            },
            {
                "tick": self.tick,
                "kind_name": "confirmed",
                "value": 2,
                "detail": "normal burst qualified",
            },
            {
                "tick": self.tick,
                "kind_name": "level_changed",
                "value": 2,
                "detail": "qualifying normal clears",
            },
            {
                "tick": self.tick,
                "kind_name": "score_changed",
                "value": 8,
                "detail": "normal burst block",
            },
            {
                "tick": self.tick,
                "kind_name": "gauge_changed",
                "value": -1,
                "detail": "scene clamp and passive drain",
            },
        ]
        return self._observation(), 8, False, False, {
            "events": events,
            "invalid_action": False,
        }


class ReplayEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.frames = (
            Frame(0, False, False, 11, 21),
            Frame(1, True, False, 12, 22),
            Frame(3, True, True, 13, 23),
            Frame(3, True, True, 14, 24),
            Frame(0, False, False, 15, 25),
            Frame(3, True, True, 16, 26),
        )

    def test_level_inputs_remain_levels_and_keep_every_cursor(self) -> None:
        mapped = evaluate_rpy.map_frames(self.frames, support_both_shots=True)

        self.assertEqual(
            [frame.kind for frame in mapped],
            ["wait", "wait", "strong_shot", "wait", "wait", "both_shots"],
        )
        self.assertEqual(
            [(frame.cursor_x, frame.cursor_y) for frame in mapped],
            [(11, 21), (12, 22), (13, 23), (14, 24), (15, 25), (16, 26)],
        )
        self.assertEqual(
            [(action.cursor_x, action.cursor_y) for action in map(evaluate_rpy._env_action, mapped)],
            [(11, 21), (12, 22), (13, 23), (14, 24), (15, 25), (16, 26)],
        )
        self.assertEqual(mapped[-1].trace_record()["shot_order"], ["weak_shot", "strong_shot"])
        self.assertEqual(mapped[2].trace_record()["shot_order"], ["strong_shot"])
        self.assertEqual(mapped[3].trace_record()["shot_order"], [])

    def test_first_two_records_suppress_independent_edges_and_keep_history(self) -> None:
        frames = (
            Frame(1, True, False, 10, 20),
            Frame(3, True, True, 11, 21),
            Frame(3, True, True, 12, 22),
            Frame(0, False, False, 13, 23),
            Frame(3, True, True, 14, 24),
        )
        mapped = evaluate_rpy.map_frames(frames, support_both_shots=True)

        self.assertEqual(
            [frame.kind for frame in mapped],
            ["wait", "wait", "wait", "wait", "both_shots"],
        )
        self.assertTrue(mapped[0].suppressed_left_edge)
        self.assertFalse(mapped[0].suppressed_right_edge)
        self.assertFalse(mapped[1].suppressed_left_edge)
        self.assertTrue(mapped[1].suppressed_right_edge)
        self.assertFalse(mapped[2].left_edge or mapped[2].right_edge)

    def test_old_api_fallback_is_left_first_and_reports_omitted_right(self) -> None:
        mapped = evaluate_rpy.map_frames(self.frames, support_both_shots=False)

        self.assertEqual(mapped[-1].kind, "weak_shot")
        self.assertTrue(mapped[-1].unrepresented_right_edge)
        self.assertEqual(
            [frame.index for frame in mapped if frame.unrepresented_right_edge],
            [5],
        )

    def test_mapping_hash_is_deterministic_and_cursor_sensitive(self) -> None:
        first = evaluate_rpy.map_frames(self.frames, support_both_shots=False)
        second = evaluate_rpy.map_frames(self.frames, support_both_shots=False)
        changed_frames = (*self.frames[:-1], Frame(3, True, True, 17, 26))
        changed = evaluate_rpy.map_frames(changed_frames, support_both_shots=False)

        self.assertEqual(evaluate_rpy._trace_sha256(first), evaluate_rpy._trace_sha256(second))
        self.assertNotEqual(evaluate_rpy._trace_sha256(first), evaluate_rpy._trace_sha256(changed))

    def test_synthetic_report_preserves_seed_cadence_and_non_golden_mismatch(self) -> None:
        words = [
            evaluate_rpy.INSPECT_RPY.encode_frame(x=1, y=2),
            evaluate_rpy.INSPECT_RPY.encode_frame(left=True, right=True, x=3, y=4),
            evaluate_rpy.INSPECT_RPY.encode_frame(left=True, right=True, x=5, y=6),
            evaluate_rpy.INSPECT_RPY.encode_frame(x=7, y=8),
            evaluate_rpy.INSPECT_RPY.encode_frame(left=True, right=True, x=9, y=10),
        ]
        data = struct.pack("<5i", -1, 2, 99, 7, 0) + bytes(32)
        data += b"".join(struct.pack("<I", word) for word in words)

        report = evaluate_rpy.evaluate_bytes(
            data,
            layout="padded",
            env_factory=FakeEnv,
            support_both_shots=False,
        )

        self.assertEqual(FakeEnv.last_seed, 0xFFFFFFFF)
        self.assertEqual(report["cadence"]["requested_tick_steps"], 5)
        self.assertEqual(report["cadence"]["actual_tick_delta"], 5)
        self.assertEqual(report["mapping"]["startup_suppressed_edge_frames"], [1])
        self.assertEqual(report["mapping"]["unrepresented_right_edge_frames"], [4])
        self.assertEqual(report["action_counts"]["mapped_actions"]["weak_shot"], 1)
        self.assertFalse(report["outcome"]["score"]["matches"])
        self.assertFalse(report["status"]["mismatches_are_failures"])
        self.assertIsNone(report["status"]["golden_fidelity_verdict"])

    def test_metsu_replay_is_rejected_before_clone_start(self) -> None:
        data = struct.pack("<5i", 1, 1, 0, 0, 1) + bytes(32)
        with self.assertRaisesRegex(ValueError, "normal-mode"):
            evaluate_rpy.evaluate_bytes(data, layout="padded", env_factory=FakeEnv)

    def test_report_uses_first_recorded_terminal_diagnostics(self) -> None:
        data = struct.pack("<5i", 1, 4, 123, 5, 0) + bytes(32)
        data += struct.pack("<I", evaluate_rpy.INSPECT_RPY.encode_frame(x=1, y=2))
        report = evaluate_rpy.evaluate_bytes(
            data,
            layout="padded",
            env_factory=DiagnosticEnv,
            support_both_shots=False,
        )

        self.assertEqual(report["outcome"]["score"]["clone_final"], 123)
        self.assertEqual(report["outcome"]["level"]["clone_final"], 4)
        self.assertEqual(report["outcome"]["highest_chain"]["clone_final"], 5)
        self.assertEqual(report["outcome"]["clone"]["latest_final_score"], 999)

    def test_report_preserves_first_causal_event_occurrences(self) -> None:
        words = [evaluate_rpy.INSPECT_RPY.encode_frame(x=1, y=2)] * 3
        data = struct.pack("<5i", 1, 1, 0, 0, 0) + bytes(32)
        data += b"".join(struct.pack("<I", word) for word in words)
        report = evaluate_rpy.evaluate_bytes(
            data,
            layout="padded",
            env_factory=TerminalCauseEnv,
            support_both_shots=False,
        )

        occurrences = report["action_counts"]["first_event_occurrences"]
        self.assertEqual(occurrences["rotten"]["frame"], 0)
        self.assertEqual(occurrences["rotten"]["a"], 17)
        self.assertEqual(occurrences["game_over"]["frame"], 1)
        self.assertEqual(report["cadence"]["terminal_frame"], 1)
        self.assertEqual(report["cadence"]["post_terminal_record_count"], 1)

    def test_worker_mode_emits_reconstructible_oracle_streams_and_provenance(self) -> None:
        data = struct.pack("<5i", 1, 2, 8, 2, 0) + bytes(32)
        data += struct.pack("<I", evaluate_rpy.INSPECT_RPY.encode_frame(x=1, y=2))

        report = evaluate_rpy.evaluate_bytes(
            data,
            layout="padded",
            worker_path="/tmp/exact-worker",
            include_timelines=True,
            env_factory=TimelineEnv,
            support_both_shots=False,
        )

        self.assertEqual(
            TimelineEnv.last_kwargs,
            {"physics_backend": "exact", "worker_path": "/tmp/exact-worker"},
        )
        self.assertEqual(report["exact_runtime_provenance"]["sha256"], "a" * 64)
        output = report["oracle_output"]
        self.assertEqual(output["score_timeline"], [[1, 8, 8]])
        self.assertEqual(output["score_checkpoints"], [[1, 8, 8, 110, 2, 1]])
        self.assertEqual(output["clear_checkpoints"], [[1, 2, 0, 110, 2, 1]])
        self.assertEqual(output["level_checkpoints"], [[1, 2, 0, 110, 1]])
        self.assertEqual(output["terminal_frame"], 1)

    def test_library_and_worker_paths_are_mutually_exclusive(self) -> None:
        data = struct.pack("<5i", 1, 1, 0, 0, 0) + bytes(32)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            evaluate_rpy.evaluate_bytes(
                data,
                layout="padded",
                library_path="portable.so",
                worker_path="exact-worker",
                env_factory=FakeEnv,
            )


if __name__ == "__main__":
    unittest.main()
