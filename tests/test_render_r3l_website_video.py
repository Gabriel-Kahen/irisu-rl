from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/render_r3l_website_video.py"
SPEC = importlib.util.spec_from_file_location("website_video_tested", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class WebsiteVideoTest(unittest.TestCase):
    def test_palette_matches_website(self) -> None:
        rotten = {"kind": "piece", "lifecycle": "rotten", "color": 0}
        active = {"kind": "piece", "lifecycle": "confirmed", "color": 0}
        projectile = {"kind": "projectile", "lifecycle": "confirmed", "color": 0}
        self.assertEqual(MODULE._color(rotten, 0), "#861f00")
        self.assertEqual(MODULE._color(active, 0), "#e44717")
        self.assertEqual(MODULE._color(projectile, 0), "#d9dcda")

    def test_bonus_cycle_matches_400ms_website_period(self) -> None:
        bonus = {"kind": "bonus", "lifecycle": "confirmed", "color": 0}
        self.assertEqual(MODULE._color(bonus, 0), MODULE.BONUS[0])
        self.assertEqual(MODULE._color(bonus, 399), MODULE.BONUS[0])
        self.assertEqual(MODULE._color(bonus, 400), MODULE.BONUS[1])

    def test_trail_retains_current_plus_four_echoes(self) -> None:
        trails = {}
        for tick in range(7):
            MODULE.update_trails(
                trails,
                [{"id": 7, "kind": "piece", "lifecycle": "confirmed",
                  "x": tick, "y": 1, "angle": 0}],
            )
        self.assertEqual([row["x"] for row in trails[7]], [2, 3, 4, 5, 6])
        MODULE.update_trails(trails, [])
        self.assertEqual(trails, {})

    def test_svg_contains_website_chrome_and_hud(self) -> None:
        observation = {
            "tick": 20, "level": 4, "score": 10055, "gauge": 50,
            "gauge_max": 100, "bodies": [],
            "field": {"x": 130, "y": 120, "width": 320, "height": 250},
        }
        svg = MODULE.website_svg(observation, {}, (320, 240))
        for text in ("irisu", "PAUSE", "RESTART", "00010055", "Level"):
            self.assertIn(text, svg)
        self.assertIn('width="156"', svg)

    def test_interpolation_matches_website_wraparound(self) -> None:
        old = {"tick": 4, "bodies": [{"id": 1, "x": 0, "y": 10, "angle": 6.2}]}
        new = {"tick": 5, "bodies": [{"id": 1, "x": 20, "y": 30, "angle": 0.1}]}
        at_half = MODULE.interpolated_bodies(old, new, 110)[0]
        self.assertEqual(at_half["x"], 10)
        self.assertEqual(at_half["y"], 20)
        self.assertGreater(at_half["angle"], 6.2)

    def test_pixel_renderer_has_website_dimensions(self) -> None:
        observation = {
            "tick": 0, "level": 1, "score": 0, "gauge": 100,
            "gauge_max": 100, "bodies": [],
            "field": {"x": 130, "y": 120, "width": 320, "height": 250},
        }
        image = MODULE.website_image(observation, {}, None, video_time_ms=0)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (840, 680))


if __name__ == "__main__":
    unittest.main()
