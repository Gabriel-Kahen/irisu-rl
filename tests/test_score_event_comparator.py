from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_score_events", ROOT / "tools" / "compare-score-events.py"
)
assert SPEC and SPEC.loader
compare_score_events = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = compare_score_events
SPEC.loader.exec_module(compare_score_events)


def original_event(
    tick: int, before: int, delta: int, after: int, ordinal: int
) -> list[dict[str, object]]:
    return [
        {
            "event": "qualifying_clear",
            "tick": tick,
            "score": before,
            "clears": ordinal,
            "chain": 2,
            "group_num": 1,
        },
        {
            "event": "score",
            "tick": tick,
            "score": after,
            "delta": delta,
            "chain": 2,
            "group_num": 1,
        },
    ]


def clone_event(
    tick: int, delta: int, after: int, group_num: int = 1
) -> list[dict[str, object]]:
    return [
        {
            "kind": "confirmed",
            "tick": tick,
            "sequence": 2 * tick,
            "a": tick,
            "b": 0,
            "value": group_num,
            "score_after_tick": after,
        },
        {
            "kind": "score_changed",
            "tick": tick,
            "sequence": 2 * tick + 1,
            "value": delta,
            "score_after_tick": after,
        },
    ]


class ScoreEventComparatorTests(unittest.TestCase):
    def test_separates_transient_timing_from_lasting_value_divergence(self) -> None:
        original = [
            *original_event(10, 0, 5, 5, 1),
            *original_event(20, 5, 3, 8, 2),
            *original_event(30, 8, 4, 12, 3),
            {"event": "terminal_after_actor", "tick": 50},
        ]
        clone = {
            "causal_events": [
                *clone_event(11, 5, 5),
                *clone_event(20, 3, 8),
                *clone_event(40, 6, 14, group_num=2),
                {"kind": "game_over", "tick": 50},
            ],
            "final": {"tick": 50},
        }

        report = compare_score_events.compare_documents(original, clone)

        self.assertEqual(report["exact_prefix"]["score_calls"], 2)
        self.assertEqual(report["exact_prefix"]["cumulative_score"], 8)
        self.assertEqual(report["exact_prefix"]["score_calls_including_tick"], 0)
        transient = report["first_transient_timing_mismatch"]
        self.assertEqual(
            (transient["start_tick"], transient["rejoined_tick"]), (10, 11)
        )
        self.assertEqual(transient["original_episodes"][0]["deltas"], [5])
        self.assertEqual(transient["clone_episodes"][0]["deltas"], [5])
        lasting = report["first_lasting_cumulative_mismatch"]
        self.assertEqual(lasting["start_tick"], 30)
        self.assertEqual(lasting["original_at_start"]["deltas"], [4])
        self.assertIsNone(lasting["clone_at_start"])
        self.assertEqual(lasting["first_clone_episode"]["tick"], 40)
        self.assertEqual(
            lasting["first_clone_episode"]["clear_inputs"][0]["group_num"], 2
        )
        mismatch = report["first_score_call_value_mismatch"]
        self.assertEqual(mismatch["ordinal"], 3)
        self.assertEqual(mismatch["differing_fields"], ["delta", "cumulative_after"])

    def test_exact_timelines_have_no_mismatch_sections(self) -> None:
        original = [
            *original_event(10, 0, 5, 5, 1),
            *original_event(20, 5, 3, 8, 2),
            {"event": "terminal_after_actor", "tick": 30},
        ]
        clone = {
            "causal_events": [
                *clone_event(10, 5, 5),
                *clone_event(20, 3, 8),
                {"kind": "game_over", "tick": 30},
            ],
            "final": {"tick": 30},
        }

        report = compare_score_events.compare_documents(original, clone)

        self.assertEqual(report["exact_prefix"]["score_calls"], 2)
        self.assertEqual(report["exact_prefix"]["score_calls_including_tick"], 2)
        self.assertIsNone(report["first_transient_timing_mismatch"])
        self.assertIsNone(report["first_score_call_value_mismatch"])
        self.assertIsNone(report["first_lasting_cumulative_mismatch"])

    def test_rejects_an_inconsistent_recorded_cumulative_score(self) -> None:
        original = [
            {"event": "score", "tick": 1, "delta": 2, "score": 2},
            {"event": "score", "tick": 1, "delta": 2, "score": 5},
        ]
        with self.assertRaisesRegex(
            compare_score_events.ComparisonError, "inconsistent cumulative"
        ):
            compare_score_events.normalize_trace(original, "broken")

    def test_run004_evidence_localizes_the_known_divergences(self) -> None:
        run = ROOT / "reference" / "runs" / "replay-41449-full-event-gdb-20260720-004"
        original_path = run / "events.jsonl"
        clone_path = run / "clone-timeline-release-current.json"
        self.assertTrue(original_path.is_file())
        self.assertTrue(clone_path.is_file())

        report = compare_score_events.compare_traces(
            compare_score_events.normalize_trace(
                compare_score_events.read_document(original_path), "run004-original"
            ),
            compare_score_events.normalize_trace(
                compare_score_events.read_document(clone_path), "current-clone"
            ),
        )

        prefix = report["exact_prefix"]
        self.assertEqual(prefix["score_calls"], 31)
        self.assertEqual(prefix["score_episodes"], 24)
        self.assertEqual(prefix["cumulative_score"], 457)
        self.assertEqual(prefix["score_calls_including_tick"], 5)

        transient = report["first_transient_timing_mismatch"]
        self.assertEqual(
            (transient["start_tick"], transient["rejoined_tick"]),
            (1336, 1337),
        )
        self.assertEqual(transient["original_episodes"][0]["deltas"], [8, 8])
        self.assertEqual(transient["first_tick_offset_clone_minus_original"], 1)

        mismatch = report["first_score_call_value_mismatch"]
        self.assertEqual(mismatch["ordinal"], 32)
        self.assertEqual(mismatch["original"]["delta"], 17)
        self.assertEqual(mismatch["original"]["cumulative_after"], 474)
        self.assertEqual(mismatch["clone"]["delta"], 25)
        self.assertEqual(mismatch["clone"]["cumulative_after"], 482)

        lasting = report["first_lasting_cumulative_mismatch"]
        self.assertEqual(lasting["start_tick"], 4035)
        self.assertEqual(lasting["observed_through_tick"], 8228)
        self.assertEqual(lasting["original_at_start"]["deltas"], [17, 17])
        clear = lasting["original_at_start"]["clear_inputs"][0]
        self.assertEqual((clear["chain"], clear["group_num"]), (2, 1))
        self.assertIsNone(lasting["clone_at_start"])
        self.assertEqual(lasting["first_clone_episode"]["tick"], 4379)
        self.assertEqual(lasting["first_clone_episode"]["deltas"], [25])


if __name__ == "__main__":
    unittest.main()
