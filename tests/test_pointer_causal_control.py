from __future__ import annotations

import unittest

from irisu_env import EventKind
from irisu_pointer.causal_control import CausalShotTracker
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import SemanticAction


def _decision(
    source: int = 10, destination: int | None = 20
) -> SteeringDecision:
    return SteeringDecision(
        action=SemanticAction.strong(0.5, 0.5),
        intent=SteeringIntent.MATCH_ROTTEN,
        source_body_id=source,
        destination_body_id=destination,
        destination_chain_id=7 if destination is not None else None,
        reason="causal-test",
    )


def _event(
    tick: int,
    sequence: int,
    kind: EventKind,
    *,
    a: int = 0,
    b: int = 0,
    value: int = 0,
) -> dict[str, int]:
    return {
        "tick": tick,
        "sequence": sequence,
        "kind": int(kind),
        "a": a,
        "b": b,
        "value": value,
    }


class CausalShotTrackerTests(unittest.TestCase):
    def test_binds_projectile_and_tracks_first_and_later_source_hits(self) -> None:
        tracker = CausalShotTracker()
        bound = tracker.consume(
            [_event(1, 1, EventKind.SHOT_FIRED, a=100, value=1)],
            decision=_decision(),
        )
        self.assertEqual(len(bound), 1)
        self.assertEqual(bound[0].projectile_id, 100)
        self.assertEqual(bound[0].source_body_id, 10)
        self.assertEqual(bound[0].destination_body_id, 20)

        tracker.consume(
            [_event(2, 2, EventKind.PROJECTILE_HIT, a=100, b=99)]
        )
        tracker.consume(
            [_event(3, 3, EventKind.PROJECTILE_HIT, a=100, b=10)]
        )
        outcome = tracker.outcomes[0]
        self.assertEqual(outcome.first_hit_body_id, 99)
        self.assertFalse(outcome.first_hit_was_intended_source)
        self.assertTrue(outcome.any_intended_source_hit)
        self.assertEqual(outcome.intended_source_hit_events, 1)
        self.assertEqual(tracker.counters.first_hits_on_intended_source, 0)
        self.assertEqual(tracker.counters.shots_with_any_intended_source_hit, 1)

    def test_same_tick_contact_order_does_not_hide_causal_pair_outcome(self) -> None:
        tracker = CausalShotTracker()
        tracker.consume(
            [
                _event(4, 10, EventKind.SHOT_FIRED, a=100, value=1),
                # Native callback order can report the pair before the direct hit.
                _event(4, 11, EventKind.CHAIN_JOINED, a=10, b=20),
                _event(4, 12, EventKind.CONFIRMED, a=20, b=10),
                _event(4, 13, EventKind.PROJECTILE_HIT, a=100, b=10),
                _event(4, 14, EventKind.CLEARED, a=10),
            ],
            decision=_decision(),
        )
        outcome = tracker.outcomes[0]
        self.assertEqual(outcome.intended_pair_join_events, 1)
        self.assertEqual(outcome.intended_pair_confirmation_events, 1)
        self.assertEqual(outcome.intended_pair_clear_events, 1)
        self.assertEqual(tracker.counters.shots_with_intended_pair_clear, 1)

    def test_pair_events_require_source_hit_and_exact_pair_evidence(self) -> None:
        tracker = CausalShotTracker()
        tracker.consume(
            [_event(1, 1, EventKind.SHOT_FIRED, a=100, value=1)],
            decision=_decision(),
        )
        tracker.consume(
            [
                _event(2, 2, EventKind.CHAIN_JOINED, a=10, b=20),
                _event(2, 3, EventKind.CONFIRMED, a=10, b=20),
                _event(2, 4, EventKind.CLEARED, a=10),
            ]
        )
        self.assertEqual(tracker.outcomes[0].intended_pair_join_events, 0)

        tracker.consume(
            [_event(3, 5, EventKind.PROJECTILE_HIT, a=100, b=10)]
        )
        tracker.consume(
            [
                _event(4, 6, EventKind.CHAIN_JOINED, a=10, b=30),
                _event(4, 7, EventKind.CONFIRMED, a=20, b=10),
                _event(4, 8, EventKind.CLEARED, a=20),
            ]
        )
        outcome = tracker.outcomes[0]
        self.assertEqual(outcome.intended_pair_join_events, 0)
        self.assertEqual(outcome.intended_pair_confirmation_events, 1)
        self.assertEqual(outcome.intended_pair_clear_events, 1)

    def test_ejection_rot_and_invalid_counters_are_identity_split(self) -> None:
        tracker = CausalShotTracker()
        tracker.consume(
            [_event(1, 1, EventKind.SHOT_FIRED, a=100, value=1)],
            decision=_decision(),
        )
        tracker.consume(
            [
                _event(2, 2, EventKind.EJECTED, a=100),
                _event(2, 3, EventKind.EJECTED, a=10),
                _event(2, 4, EventKind.EJECTED, a=20),
                _event(2, 5, EventKind.EJECTED, a=88),
                _event(2, 6, EventKind.ROTTEN, a=10),
                _event(2, 7, EventKind.ROTTEN, a=20),
                _event(2, 8, EventKind.ROTTEN, a=88),
            ]
        )
        tracker.consume(
            [_event(3, 9, EventKind.INVALID_ACTION)],
            decision=_decision(source=30, destination=40),
        )
        counters = tracker.counters
        self.assertEqual(counters.projectile_ejections, 1)
        self.assertEqual(counters.source_ejections, 1)
        self.assertEqual(counters.other_ejections, 2)
        self.assertEqual(counters.source_rotten_events, 1)
        self.assertEqual(counters.destination_rotten_events, 1)
        self.assertEqual(counters.other_rotten_events, 1)
        self.assertEqual(counters.invalid_action_events, 1)
        self.assertEqual(counters.invalid_shot_decisions, 1)
        self.assertEqual(counters.shot_decisions_without_fire, 1)
        outcome = tracker.outcomes[0]
        self.assertEqual(outcome.projectile_ejection_events, 1)
        self.assertEqual(outcome.source_ejection_events, 1)
        self.assertEqual(outcome.source_rotten_events, 1)
        self.assertEqual(outcome.destination_rotten_events, 1)

    def test_latest_source_hit_gets_repeated_pair_credit(self) -> None:
        tracker = CausalShotTracker()
        tracker.consume(
            [
                _event(1, 1, EventKind.SHOT_FIRED, a=100, value=1),
                _event(1, 2, EventKind.PROJECTILE_HIT, a=100, b=10),
            ],
            decision=_decision(),
        )
        tracker.consume(
            [
                _event(2, 3, EventKind.SHOT_FIRED, a=101, value=1),
                _event(2, 4, EventKind.PROJECTILE_HIT, a=101, b=10),
                _event(2, 5, EventKind.CHAIN_JOINED, a=20, b=10),
            ],
            decision=_decision(),
        )
        first, second = tracker.outcomes
        self.assertEqual(first.intended_pair_join_events, 0)
        self.assertEqual(second.intended_pair_join_events, 1)

    def test_manifest_is_deterministic_and_accepts_kind_name(self) -> None:
        def build() -> CausalShotTracker:
            value = CausalShotTracker()
            value.consume(
                [
                    {
                        "tick": 1,
                        "sequence": 8,
                        "kind_name": "shot_fired",
                        "a": 100,
                        "b": 0,
                        "value": 1,
                    }
                ],
                decision=_decision(),
            )
            return value

        first, second = build(), build()
        self.assertEqual(first.manifest(), second.manifest())
        self.assertEqual(first.sha256, second.sha256)
        self.assertEqual(len(first.sha256), 64)
        self.assertEqual(
            first.manifest()["evidence"],
            "steering_decisions_and_public_events_only",
        )

    def test_reset_clears_identity_and_allows_episode_clock_restart(self) -> None:
        tracker = CausalShotTracker()
        tracker.consume(
            [_event(10, 10, EventKind.SHOT_FIRED, a=100, value=1)],
            decision=_decision(),
        )
        with self.assertRaisesRegex(ValueError, "moved backward"):
            tracker.consume([_event(9, 11, EventKind.ROTTEN, a=10)])
        with self.assertRaisesRegex(ValueError, "sequence was repeated"):
            tracker.consume([_event(10, 10, EventKind.ROTTEN, a=10)])

        tracker.reset()
        self.assertEqual(tracker.outcomes, ())
        self.assertEqual(tracker.counters.events_consumed, 0)
        tracker.consume(
            [_event(1, 1, EventKind.SHOT_FIRED, a=100, value=1)],
            decision=_decision(),
        )
        self.assertEqual(tracker.counters.bound_shots, 1)

    def test_mismatched_strength_fails_before_recording_evidence(self) -> None:
        tracker = CausalShotTracker()
        with self.assertRaisesRegex(ValueError, "strength disagrees"):
            tracker.consume(
                [_event(1, 1, EventKind.SHOT_FIRED, a=100, value=0)],
                decision=_decision(),
            )
        self.assertEqual(tracker.outcomes, ())
        self.assertEqual(tracker.counters.events_consumed, 0)


if __name__ == "__main__":
    unittest.main()
