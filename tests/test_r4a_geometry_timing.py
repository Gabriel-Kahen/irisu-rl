from __future__ import annotations

import unittest

from irisu_rl.original_game.geometry import (
    Affine2D,
    Calibration,
    GeometryError,
    Rect,
    WindowGeometry,
    fit_affine,
)
from irisu_rl.original_game.timing import (
    ActionTiming,
    CadenceConfig,
    GameplayClock,
    TimingError,
)


class GeometryTests(unittest.TestCase):
    def test_affine_compose_inverse_and_least_squares_fit(self) -> None:
        transform = Affine2D(2.0, 0.25, -0.5, 3.0, 7.0, -4.0)
        inverse = transform.inverse()
        point = (13.0, 19.0)
        self.assertEqual(transform.compose(inverse).apply(point), point)

        sources = ((0.0, 0.0), (640.0, 0.0), (0.0, 480.0), (640.0, 480.0))
        fitted, residuals = fit_affine(
            (source, transform.apply(source)) for source in sources
        )
        for source in sources:
            expected = transform.apply(source)
            actual = fitted.apply(source)
            self.assertAlmostEqual(actual[0], expected[0], places=9)
            self.assertAlmostEqual(actual[1], expected[1], places=9)
        self.assertEqual(residuals.count, 4)
        self.assertLess(residuals.worst, 1e-9)

    def test_fit_rejects_invalid_and_collinear_pairs(self) -> None:
        with self.assertRaises(GeometryError):
            fit_affine([((0.0, 0.0), (0.0, 0.0))] * 2)
        with self.assertRaises(GeometryError):
            fit_affine(
                [
                    ((0.0, 0.0), (0.0, 0.0)),
                    ((1.0, 1.0), (1.0, 1.0)),
                    ((2.0, 2.0), (2.0, 2.0)),
                ]
            )
        with self.assertRaises(GeometryError):
            Rect(0.0, 0.0, float("nan"), 1.0)
        with self.assertRaises(TypeError):
            Rect(True, 0.0, 1.0, 1.0)

    def test_calibration_is_bound_to_claim_age_geometry_and_anchors(self) -> None:
        geometry = WindowGeometry(
            "window-1",
            "capture-1",
            Rect(100.0, 50.0, 644.0, 484.0),
            Rect(102.0, 52.0, 640.0, 480.0),
            Rect(0.0, 0.0, 1920.0, 1080.0),
        )
        pairs = [
            ((0.0, 0.0), (0.0, 0.0)),
            ((640.0, 0.0), (640.0, 0.0)),
            ((0.0, 480.0), (0.0, 480.0)),
            ((640.0, 480.0), (640.0, 480.0)),
        ]
        calibration = Calibration.fit(
            pairs,
            geometry,
            created_at=10.0,
            max_age=5.0,
            max_geometry_drift=0.5,
            max_anchor_drift=1.0,
            max_residual=1e-6,
        )
        self.assertEqual(
            calibration.to_window_local(
                (320.0, 240.0),
                now=12.0,
                geometry=geometry,
                anchor_drift=0.25,
            ),
            (322.0, 242.0),
        )
        changed_identity = WindowGeometry(
            "other-window",
            "capture-1",
            geometry.outer,
            geometry.client,
            geometry.capture,
        )
        for kwargs in (
            {"now": 16.0, "geometry": geometry, "anchor_drift": 0.0},
            {"now": 12.0, "geometry": geometry, "anchor_drift": 1.01},
            {"now": 12.0, "geometry": changed_identity, "anchor_drift": 0.0},
        ):
            with self.assertRaises(GeometryError):
                calibration.to_client((10.0, 10.0), **kwargs)
        with self.assertRaises(GeometryError):
            calibration.to_client(
                (641.0, 10.0),
                now=12.0,
                geometry=geometry,
                anchor_drift=0.0,
            )

    def test_bounds_are_half_open_and_bad_fit_residual_is_rejected(self) -> None:
        rectangle = Rect(0.0, 0.0, 640.0, 480.0)
        self.assertTrue(rectangle.contains((639.999, 479.999)))
        self.assertFalse(rectangle.contains((640.0, 100.0)))
        self.assertFalse(rectangle.contains((100.0, 480.0)))

        geometry = WindowGeometry(
            "window-1",
            "capture-1",
            Rect(0.0, 0.0, 644.0, 484.0),
            Rect(2.0, 2.0, 640.0, 480.0),
            Rect(0.0, 0.0, 640.0, 480.0),
        )
        good = Calibration.fit(
            [
                ((0.0, 0.0), (0.0, 0.0)),
                ((639.0, 0.0), (639.0, 0.0)),
                ((0.0, 479.0), (0.0, 479.0)),
            ],
            geometry,
            created_at=0.0,
            max_age=5.0,
            max_geometry_drift=0.0,
            max_anchor_drift=0.0,
            max_residual=1e-6,
        )
        with self.assertRaisesRegex(GeometryError, "captured surface"):
            good.to_client(
                (640.0, 100.0),
                now=1.0,
                geometry=geometry,
                anchor_drift=0.0,
            )
        with self.assertRaisesRegex(GeometryError, "residual"):
            Calibration.fit(
                [
                    ((0.0, 0.0), (0.0, 0.0)),
                    ((639.0, 0.0), (639.0, 0.0)),
                    ((0.0, 479.0), (0.0, 479.0)),
                    ((639.0, 479.0), (620.0, 450.0)),
                ],
                geometry,
                created_at=0.0,
                max_age=5.0,
                max_geometry_drift=0.0,
                max_anchor_drift=0.0,
                max_residual=0.5,
            )


class GameplayClockTests(unittest.TestCase):
    @staticmethod
    def replay(
        frames: list[tuple[float, str]], config: CadenceConfig | None = None
    ) -> tuple[GameplayClock, list[str], list[bool]]:
        clock = GameplayClock(config or CadenceConfig())
        classifications: list[str] = []
        safety: list[bool] = []
        for timestamp, content_hash in frames:
            clock, assessment = clock.observe(timestamp, content_hash)
            classifications.append(assessment.classification)
            safety.append(assessment.safe_to_act)
        return clock, classifications, safety

    def test_faster_compositor_duplicates_recover_gameplay_cadence(self) -> None:
        clock, classes, safety = self.replay(
            [
                (0.000, "a"),
                (0.010, "a"),
                (0.020, "b"),
                (0.030, "b"),
                (0.040, "c"),
                (0.050, "c"),
                (0.060, "d"),
            ]
        )
        self.assertEqual(
            classes,
            ["warmup", "duplicate", "unique", "duplicate", "unique", "duplicate", "unique"],
        )
        self.assertTrue(safety[-1])
        posterior = clock.posterior()
        self.assertAlmostEqual(posterior.period, 0.020)
        self.assertEqual(posterior.sample_count, 3)

    def test_dropped_and_delayed_frames_are_not_safe_decisions(self) -> None:
        dropped, classes, safety = self.replay(
            [(0.000, "a"), (0.020, "b"), (0.060, "c")]
        )
        self.assertEqual(classes[-1], "dropped")
        self.assertFalse(safety[-1])
        self.assertEqual(len(dropped.period_samples), 2)
        self.assertAlmostEqual(dropped.period_samples[0], 0.020)
        self.assertAlmostEqual(dropped.period_samples[1], 0.040)

        _, classes, safety = self.replay(
            [(0.000, "a"), (0.020, "b"), (0.040, "c"), (0.065, "d")]
        )
        self.assertEqual(classes[-1], "delayed")
        self.assertFalse(safety[-1])

    def test_out_of_order_stall_duplicate_limit_and_restart_fail_closed(self) -> None:
        clock, _ = GameplayClock().observe(1.000, "a")
        failed, assessment = clock.observe(0.999, "b")
        self.assertEqual(assessment.classification, "out_of_order")
        self.assertFalse(assessment.safe_to_act)
        with self.assertRaises(TimingError):
            failed.posterior()
        latched, assessment = failed.observe(1.010, "c")
        self.assertEqual(assessment.classification, "latched_failure")
        restarted, assessment = latched.observe(1.020, "d", restart=True)
        self.assertEqual(assessment.classification, "restart")
        self.assertEqual(restarted.generation, 1)
        self.assertIsNone(restarted.failed_reason)

        stalled, _, safety = self.replay([(0.0, "a"), (0.3, "b")])
        self.assertEqual(stalled.failed_reason, "capture stall")
        self.assertFalse(safety[-1])

        config = CadenceConfig(max_duplicate_run=2)
        duplicated, classes, safety = self.replay(
            [(0.0, "a"), (0.01, "a"), (0.02, "a"), (0.03, "a")], config
        )
        self.assertEqual(classes[-1], "duplicate_limit")
        self.assertFalse(safety[-1])
        self.assertIsNotNone(duplicated.failed_reason)

    def test_effect_boundary_is_independent_of_visible_render_delay(self) -> None:
        clock, _, _ = self.replay(
            [(0.000, "a"), (0.020, "b"), (0.040, "c"), (0.060, "d")]
        )
        effect = clock.infer_poll_effect(request_at=0.061, injected_at=0.062)
        self.assertLessEqual(effect.earliest_effect_at, 0.080)
        self.assertGreaterEqual(effect.latest_effect_at, 0.080)
        self.assertGreaterEqual(
            effect.latest_effect_at - effect.earliest_effect_at, effect.period
        )

        fast_visible = ActionTiming(effect).confirm_visible(
            effect.latest_effect_at + 0.020
        )
        delayed_visible = ActionTiming(effect).confirm_visible(
            effect.latest_effect_at + 0.200
        )
        self.assertEqual(fast_visible.effect, delayed_visible.effect)
        self.assertLess(
            fast_visible.first_visible.latest_effect_to_visible,
            delayed_visible.first_visible.latest_effect_to_visible,
        )

    def test_ambiguous_thirty_millisecond_changes_are_not_aliased(self) -> None:
        clock, classes, _ = self.replay(
            [(0.000, "a"), (0.030, "b"), (0.060, "c"), (0.090, "d")]
        )
        self.assertEqual(classes[1:], ["unique", "unique", "unique"])
        posterior = clock.posterior()
        self.assertAlmostEqual(posterior.period, 0.030)
        self.assertIn(0.015, posterior.plausible_periods)
        self.assertGreaterEqual(posterior.phase_uncertainty, 0.015)

    def test_perfect_presentation_does_not_imply_exact_poll_phase(self) -> None:
        clock, _, _ = self.replay(
            [(0.000, "a"), (0.020, "b"), (0.040, "c"), (0.060, "d")]
        )
        posterior = clock.posterior()
        self.assertGreaterEqual(posterior.phase_uncertainty, posterior.period / 2)
        effect = clock.infer_poll_effect(request_at=0.061, injected_at=0.062)
        self.assertGreaterEqual(
            effect.latest_effect_at - effect.earliest_effect_at,
            posterior.period,
        )

    def test_timestamp_and_action_validation_is_strict(self) -> None:
        clock = GameplayClock()
        for invalid in (float("nan"), float("inf"), True, "0.0"):
            expected = TypeError if invalid is True or isinstance(invalid, str) else TimingError
            with self.assertRaises(expected):
                clock.observe(invalid, "frame")
        with self.assertRaises(TypeError):
            clock.observe(0.0, 7)

        ready, _, _ = self.replay(
            [(0.000, "a"), (0.020, "b"), (0.040, "c"), (0.060, "d")]
        )
        with self.assertRaises(TimingError):
            ready.infer_poll_effect(request_at=0.050, injected_at=0.061)
        with self.assertRaises(TimingError):
            ready.infer_poll_effect(request_at=0.070, injected_at=0.069)
        with self.assertRaises(TimingError):
            ready.infer_poll_effect(request_at=0.400, injected_at=0.401)


if __name__ == "__main__":
    unittest.main()
