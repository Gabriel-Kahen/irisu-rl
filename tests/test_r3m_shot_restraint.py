from __future__ import annotations

import unittest

import benchmarks.rl_r3m_shot_restraint as runner


class ShotRestraintRunnerTests(unittest.TestCase):
    def test_seed_derivation_is_frozen_and_unique(self) -> None:
        expected = tuple(
            int.from_bytes(
                __import__("hashlib").sha256(
                    f"{runner.RUN_ID}|matched-development|{index}".encode()
                ).digest()[:4],
                "big",
            )
            for index in range(len(runner.SEEDS))
        )
        self.assertEqual(runner.SEEDS, expected)
        self.assertEqual(len(set(runner.SEEDS)), len(runner.SEEDS))

    def test_action_word_round_trip_fields(self) -> None:
        class Action:
            kind = 2
            cursor_x = 123.0
            cursor_y = 456.0

        word = runner.encode_action(Action())
        self.assertEqual(word & 3, 2)
        self.assertEqual((word >> 2) & 1023, 123)
        self.assertEqual((word >> 12) & 511, 456)


if __name__ == "__main__":
    unittest.main()
