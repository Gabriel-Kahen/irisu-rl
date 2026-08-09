from __future__ import annotations

import unittest

from irisu_env import ActionKind
from irisu_pointer.steering import (
    ClosedLoopSteeringExpert,
    SteeringExpertConfig,
    SteeringIntent,
)
from irisu_rl.actions import SemanticActionKind


def _body(
    identifier: int,
    *,
    kind: str = "piece",
    lifecycle: str = "dynamic_fresh",
    color: int = 1,
    x: float = 200.0,
    y: float = 180.0,
    vx: float = 0.0,
    size: float = 40.0,
    chain_id: int = 0,
    rot_timer: int = 0,
    remaining_lifetime: int = 500,
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": kind,
        "lifecycle": lifecycle,
        "color": color,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": 0.0,
        "size": size,
        "chain_id": chain_id,
        "projectile_hits": 0,
        "remaining_lifetime": remaining_lifetime,
        "rot_timer": rot_timer,
    }


def _observation(
    tick: int,
    bodies: list[dict[str, object]],
    *,
    terminated: bool = False,
) -> dict[str, object]:
    return {
        "tick": tick,
        "gauge": 500,
        "gauge_max": 1000,
        "terminated": terminated,
        "truncated": False,
        "field": {"x": 16.0, "y": 8.0, "width": 608.0, "height": 464.0},
        "bodies": bodies,
    }


class ClosedLoopSteeringExpertTests(unittest.TestCase):
    def test_match_uses_target_relative_side_and_height(self) -> None:
        expert = ClosedLoopSteeringExpert()
        observation = _observation(
            10,
            [
                _body(1, x=180.0, y=220.0),
                _body(2, x=320.0, y=120.0),
            ],
        )

        decision = expert.predict(observation)

        self.assertEqual(decision.intent, SteeringIntent.STEER_MATCH)
        self.assertEqual(decision.source_body_id, 1)
        self.assertEqual(decision.destination_body_id, 2)
        self.assertEqual(decision.action.kind, SemanticActionKind.FIRE_STRONG)
        self.assertAlmostEqual(decision.action.x_norm * 640.0, 160.0)
        self.assertAlmostEqual(decision.action.y_norm * 480.0, 250.0)
        self.assertEqual(decision.impact_x_sizes, -0.5)
        self.assertEqual(decision.impact_y_sizes, 0.75)
        press, release = decision.primitive_actions()
        self.assertEqual(press.kind, ActionKind.STRONG_SHOT)
        self.assertEqual((press.cursor_x, press.cursor_y), (160, 250))
        self.assertEqual(release.kind, ActionKind.WAIT)

    def test_impact_side_reverses_for_destination_on_left(self) -> None:
        expert = ClosedLoopSteeringExpert()
        decision = expert.predict(
            _observation(
                1,
                [
                    _body(4, x=360.0, y=240.0),
                    _body(5, x=100.0, y=100.0),
                ],
            )
        )
        self.assertEqual(decision.source_body_id, 4)
        self.assertAlmostEqual(decision.action.x_norm * 640.0, 380.0)
        self.assertEqual(decision.impact_x_sizes, 0.5)

    def test_equal_x_geometry_is_invariant_to_body_id_renumbering(self) -> None:
        first = ClosedLoopSteeringExpert().predict(
            _observation(
                1,
                [
                    _body(1, x=200.0, y=260.0),
                    _body(2, x=200.0, y=100.0),
                ],
            )
        )
        second = ClosedLoopSteeringExpert().predict(
            _observation(
                1,
                [
                    _body(900, x=200.0, y=260.0),
                    _body(3, x=200.0, y=100.0),
                ],
            )
        )
        self.assertEqual(first.action, second.action)
        self.assertEqual(first.impact_x_sizes, second.impact_x_sizes)

    def test_velocity_lead_uses_public_lifecycle_units_and_50hz_ticks(self) -> None:
        config = SteeringExpertConfig(source_velocity_lead_ticks=2.0)
        dynamic = ClosedLoopSteeringExpert(config=config).predict(
            _observation(
                1,
                [
                    _body(1, x=180.0, y=220.0, vx=5.0),
                    _body(2, x=320.0, y=120.0),
                ],
            )
        )
        scripted = ClosedLoopSteeringExpert(config=config).predict(
            _observation(
                1,
                [
                    _body(
                        1,
                        lifecycle="scripted_falling",
                        x=180.0,
                        y=220.0,
                        vx=5.0,
                    ),
                    _body(2, x=320.0, y=120.0),
                ],
            )
        )
        self.assertAlmostEqual(dynamic.action.x_norm * 640.0, 162.0)
        self.assertAlmostEqual(scripted.action.x_norm * 640.0, 170.0)

    def test_explicit_display_velocity_is_not_scaled_twice(self) -> None:
        source = _body(1, x=180.0, y=220.0, vx=999.0)
        source["vx_display_per_second"] = 100.0
        decision = ClosedLoopSteeringExpert(
            config=SteeringExpertConfig(source_velocity_lead_ticks=2.0)
        ).predict(
            _observation(
                1,
                [source, _body(2, x=320.0, y=120.0)],
            )
        )
        self.assertAlmostEqual(decision.action.x_norm * 640.0, 164.0)

    def test_observes_then_globally_replans_the_correction(self) -> None:
        expert = ClosedLoopSteeringExpert(
            config=SteeringExpertConfig(observe_ticks=16)
        )
        bodies = [_body(1, x=180.0, y=220.0), _body(2, x=320.0, y=120.0)]
        first = expert.predict(_observation(10, bodies))
        repeated = expert.predict(_observation(10, bodies))
        cooldown = expert.predict(_observation(12, bodies))
        moved = [
            _body(1, x=210.0, y=215.0),
            _body(2, x=320.0, y=120.0),
        ]
        correction = expert.predict(_observation(26, moved))

        self.assertIs(first, repeated)
        self.assertEqual(cooldown.action.kind, SemanticActionKind.WAIT)
        self.assertEqual(cooldown.action.wait_ticks, 14)
        self.assertEqual(correction.source_body_id, 1)
        self.assertEqual(correction.destination_body_id, 2)
        self.assertEqual(correction.correction_index, 1)

    def test_visible_projectile_does_not_stall_after_safe_cooldown(self) -> None:
        expert = ClosedLoopSteeringExpert()
        bodies = [
            _body(1, x=180.0, y=220.0),
            _body(2, x=320.0, y=120.0),
            _body(
                8,
                kind="projectile",
                lifecycle="dynamic_fresh",
                color=-1,
                x=185.0,
                y=300.0,
                size=10.0,
            ),
        ]
        decision = expert.predict(_observation(1, bodies))
        self.assertTrue(decision.is_shot)
        self.assertIn(decision.source_body_id, {1, 2})
        self.assertIn(decision.destination_body_id, {1, 2})

    def test_grouped_destination_is_anchor_but_never_shot_source(self) -> None:
        expert = ClosedLoopSteeringExpert()
        decision = expert.predict(
            _observation(
                1,
                [
                    _body(1, x=190.0, y=230.0),
                    _body(
                        2,
                        lifecycle="confirmed",
                        x=260.0,
                        y=300.0,
                        chain_id=17,
                    ),
                    _body(
                        3,
                        lifecycle="confirmed",
                        x=285.0,
                        y=300.0,
                        chain_id=17,
                    ),
                ],
            )
        )
        self.assertEqual(decision.intent, SteeringIntent.EXTEND_ANCHOR)
        self.assertEqual(decision.source_body_id, 1)
        self.assertIn(decision.destination_body_id, {2, 3})
        self.assertEqual(decision.destination_chain_id, 17)

    def test_grouped_or_confirmed_tracked_source_causes_restraint(self) -> None:
        expert = ClosedLoopSteeringExpert(
            config=SteeringExpertConfig(observe_ticks=2)
        )
        initial = [_body(1, x=190.0, y=230.0), _body(2, x=280.0, y=120.0)]
        expert.predict(_observation(1, initial))
        grouped = [
            _body(1, lifecycle="confirmed", x=250.0, y=220.0, chain_id=9),
            _body(2, lifecycle="confirmed", x=270.0, y=220.0, chain_id=9),
        ]
        decision = expert.predict(_observation(3, grouped))
        self.assertEqual(decision.action.kind, SemanticActionKind.WAIT)
        self.assertEqual(decision.intent, SteeringIntent.PRESERVE_GROUP)
        self.assertIsNone(expert.tracked_source_body_id)

    def test_fresh_source_is_steered_to_rotten_same_color_destination(self) -> None:
        expert = ClosedLoopSteeringExpert()
        decision = expert.predict(
            _observation(
                1,
                [
                    _body(
                        1,
                        lifecycle="rotten",
                        x=160.0,
                        y=250.0,
                        rot_timer=20,
                    ),
                    _body(2, x=260.0, y=270.0),
                    _body(3, color=2, x=100.0, y=240.0),
                    _body(4, color=2, x=120.0, y=180.0),
                ],
            )
        )
        self.assertEqual(decision.intent, SteeringIntent.MATCH_ROTTEN)
        self.assertEqual((decision.source_body_id, decision.destination_body_id), (2, 1))

    def test_unmatched_rotten_hazard_is_ejected_from_inner_side(self) -> None:
        expert = ClosedLoopSteeringExpert(
            config=SteeringExpertConfig(enable_hazard_ejection=True)
        )
        decision = expert.predict(
            _observation(
                1,
                [
                    _body(
                        10,
                        lifecycle="rotten",
                        color=3,
                        x=70.0,
                        y=340.0,
                        rot_timer=9,
                    )
                ],
            )
        )
        self.assertEqual(decision.intent, SteeringIntent.EJECT_HAZARD)
        self.assertEqual(decision.source_body_id, 10)
        self.assertGreater(decision.impact_x_sizes, 0.0)
        self.assertIsNone(decision.destination_body_id)

    def test_bonus_is_activated_toward_ungrouped_piece(self) -> None:
        expert = ClosedLoopSteeringExpert(
            config=SteeringExpertConfig(enable_bonus=True)
        )
        decision = expert.predict(
            _observation(
                1,
                [
                    _body(
                        30,
                        kind="bonus",
                        lifecycle="scripted_falling",
                        color=-1,
                        x=200.0,
                        y=150.0,
                    ),
                    _body(4, color=2, x=320.0, y=230.0),
                ],
            )
        )
        self.assertEqual(decision.intent, SteeringIntent.ACTIVATE_BONUS)
        self.assertEqual((decision.source_body_id, decision.destination_body_id), (30, 4))
        self.assertEqual(decision.action.kind, SemanticActionKind.FIRE_STRONG)

    def test_safe_boundary_retargets_then_waits_when_all_pairs_stall(self) -> None:
        expert = ClosedLoopSteeringExpert(
            config=SteeringExpertConfig(
                observe_ticks=1,
                resolution_wait_ticks=4,
            )
        )
        bodies = [_body(1, x=180.0, y=220.0), _body(2, x=320.0, y=120.0)]
        self.assertTrue(expert.predict(_observation(1, bodies)).is_shot)
        second = expert.predict(_observation(2, bodies))
        self.assertTrue(second.is_shot)
        self.assertEqual(
            (second.source_body_id, second.destination_body_id), (2, 1)
        )
        decision = expert.predict(_observation(3, bodies))
        self.assertFalse(decision.is_shot)
        self.assertIn("progress tracker", decision.reason)

    def test_no_match_and_no_hazard_waits(self) -> None:
        expert = ClosedLoopSteeringExpert()
        decision = expert.predict(
            _observation(
                1,
                [
                    _body(1, color=1, x=300.0, y=100.0),
                    _body(2, color=2, x=350.0, y=110.0),
                ],
            )
        )
        self.assertEqual(decision.intent, SteeringIntent.WAIT)
        self.assertEqual(decision.action.kind, SemanticActionKind.WAIT)
        self.assertEqual(len(expert.act(_observation(1, []))), 1)

    def test_lingering_projectile_does_not_block_other_pairs(self) -> None:
        expert = ClosedLoopSteeringExpert()
        decision = expert.predict(
            _observation(
                1,
                [
                    _body(1, color=1, x=180.0, y=220.0),
                    _body(2, color=1, x=260.0, y=120.0),
                    _body(3, color=2, x=380.0, y=230.0),
                    _body(4, color=2, x=460.0, y=130.0),
                    _body(
                        20,
                        kind="projectile",
                        color=-1,
                        x=182.0,
                        y=290.0,
                        size=10.0,
                    ),
                ],
            )
        )
        self.assertTrue(decision.is_shot)
        self.assertIn(decision.source_body_id, {1, 2, 3, 4})
        self.assertIn(decision.destination_body_id, {1, 2, 3, 4})

    def test_tick_rewind_requires_explicit_reset(self) -> None:
        expert = ClosedLoopSteeringExpert()
        expert.predict(_observation(5, []))
        with self.assertRaisesRegex(RuntimeError, "moved backwards"):
            expert.predict(_observation(4, []))
        expert.reset(7)
        self.assertEqual(
            expert.predict(_observation(0, [])).action.kind,
            SemanticActionKind.WAIT,
        )


if __name__ == "__main__":
    unittest.main()
