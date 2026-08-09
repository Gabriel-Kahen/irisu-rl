from __future__ import annotations

import unittest

from irisu_pointer.steering_progress import (
    DirectedPair,
    DirectedPairProgressTracker,
    PairProgressStatus,
)


def _body(
    identifier: int,
    x: float,
    *,
    size: float = 40.0,
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "effect_x": x,
        "effect_y": 100.0,
        "size": size,
    }


def _observation(*bodies: dict[str, object]) -> dict[str, object]:
    return {"bodies": list(bodies)}


class DirectedPairProgressTrackerTests(unittest.TestCase):
    def test_material_surface_closure_is_progress(self) -> None:
        tracker = DirectedPairProgressTracker(minimum_closure_sizes=0.05)
        tracker.begin(_observation(_body(1, 100.0), _body(2, 300.0)), 1, 2)

        result = tracker.assess(
            _observation(_body(1, 104.0), _body(2, 300.0))
        )

        assert result is not None
        self.assertEqual(result.status, PairProgressStatus.PROGRESSED)
        self.assertAlmostEqual(result.minimum_closure, 2.0)
        self.assertEqual(tracker.stalled_pairs, ())

    def test_subthreshold_closure_stalls_only_the_directed_pair(self) -> None:
        observation = _observation(
            _body(1, 100.0), _body(2, 300.0), _body(3, 500.0)
        )
        tracker = DirectedPairProgressTracker(minimum_closure_sizes=0.05)
        tracker.begin(observation, 1, 2)

        result = tracker.assess(
            _observation(
                _body(1, 101.0), _body(2, 300.0), _body(3, 500.0)
            )
        )

        assert result is not None
        self.assertEqual(result.status, PairProgressStatus.STALLED)
        self.assertTrue(
            tracker.is_stalled(observation, 1, 2)
        )
        self.assertFalse(tracker.is_stalled(observation, 2, 1))
        self.assertFalse(tracker.is_stalled(observation, 1, 3))

    def test_later_public_closure_releases_a_stalled_pair(self) -> None:
        tracker = DirectedPairProgressTracker(minimum_closure_sizes=0.05)
        initial = _observation(_body(1, 100.0), _body(2, 300.0))
        tracker.begin(initial, 1, 2)
        tracker.assess(initial)

        still = _observation(_body(1, 101.0), _body(2, 300.0))
        recovered = _observation(_body(1, 103.0), _body(2, 300.0))
        self.assertTrue(tracker.is_stalled(still, 1, 2))
        self.assertFalse(tracker.is_stalled(recovered, 1, 2))
        self.assertEqual(tracker.stalled_pairs, ())

    def test_stall_has_no_tick_or_attempt_count_expiry(self) -> None:
        tracker = DirectedPairProgressTracker()
        initial = _observation(_body(1, 100.0), _body(2, 300.0))
        tracker.begin(initial, 1, 2)
        tracker.assess(initial)
        unchanged = {**initial, "tick": 1_000_000}

        self.assertTrue(tracker.is_stalled(unchanged, 1, 2))
        with self.assertRaisesRegex(RuntimeError, "stalled pair"):
            tracker.begin(unchanged, 1, 2)

    def test_missing_body_resolves_pending_and_prunes_stall(self) -> None:
        tracker = DirectedPairProgressTracker()
        initial = _observation(_body(1, 100.0), _body(2, 300.0))
        tracker.begin(initial, 1, 2)
        resolved = tracker.assess(_observation(_body(2, 300.0)))

        assert resolved is not None
        self.assertEqual(resolved.status, PairProgressStatus.RESOLVED)
        self.assertIsNone(resolved.observed_gap)

        tracker.begin(initial, 1, 2)
        tracker.assess(initial)
        tracker.prune(_observation(_body(2, 300.0)))
        self.assertEqual(tracker.stalled_pairs, ())

    def test_reset_clears_pending_and_stalled_state(self) -> None:
        tracker = DirectedPairProgressTracker()
        initial = _observation(_body(1, 100.0), _body(2, 300.0))
        tracker.begin(initial, 1, 2)
        tracker.assess(initial)
        tracker.reset()

        self.assertIsNone(tracker.pending_pair)
        self.assertEqual(tracker.stalled_pairs, ())
        self.assertEqual(DirectedPair(1, 2), DirectedPair(1, 2))


if __name__ == "__main__":
    unittest.main()
