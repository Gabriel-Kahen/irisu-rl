from __future__ import annotations

import unittest

from irisu_pointer.shot_necessity import (
    ProbeOutcome,
    WaitDominanceConfig,
    choose_shot,
)


def outcome(**changes: int | bool) -> ProbeOutcome:
    values: dict[str, int | bool] = {
        "survival_ticks": 128,
        "score": 100,
        "clears": 4,
        "final_gauge": 8_000,
        "minimum_gauge": 7_000,
        "terminated": False,
        "truncated": False,
    }
    values.update(changes)
    return ProbeOutcome(**values)  # type: ignore[arg-type]


class ShotNecessityTests(unittest.TestCase):
    def test_exact_tie_prefers_wait(self) -> None:
        self.assertEqual(
            choose_shot(outcome(), outcome()),
            (False, "wait-tie-or-no-benefit"),
        )

    def test_strict_benefits_execute_shot(self) -> None:
        cases = (
            ({"survival_ticks": 128}, "shot-survival"),
            ({"clears": 5}, "shot-clear"),
            ({"score": 101}, "shot-score"),
            ({"final_gauge": 8_017}, "shot-gauge"),
        )
        for changes, reason in cases:
            with self.subTest(reason=reason):
                wait = outcome(survival_ticks=127) if reason == "shot-survival" else outcome()
                self.assertEqual(choose_shot(outcome(**changes), wait), (True, reason))

    def test_small_gauge_difference_prefers_wait(self) -> None:
        self.assertEqual(
            choose_shot(outcome(final_gauge=8_016), outcome()),
            (False, "wait-tie-or-no-benefit"),
        )

    def test_wait_survival_dominates_score(self) -> None:
        self.assertEqual(
            choose_shot(
                outcome(survival_ticks=64, score=1_000),
                outcome(survival_ticks=128),
            ),
            (False, "wait-survival"),
        )

    def test_config_rejects_nonpositive_values(self) -> None:
        for field in ("probe_ticks", "wait_ticks", "gauge_advantage"):
            with self.subTest(field=field):
                values = {"probe_ticks": 128, "wait_ticks": 16, "gauge_advantage": 16}
                values[field] = 0
                with self.assertRaises(ValueError):
                    WaitDominanceConfig(**values)


if __name__ == "__main__":
    unittest.main()
