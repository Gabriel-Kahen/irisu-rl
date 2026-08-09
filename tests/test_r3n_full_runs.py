from __future__ import annotations

import hashlib
import unittest

import benchmarks.rl_r3n_full_runs as runner


class FullRunTests(unittest.TestCase):
    def test_frozen_seed_derivation(self) -> None:
        expected = tuple(
            int.from_bytes(
                hashlib.sha256(
                    f"{runner.RUN_ID}|full-development|{index}".encode()
                ).digest()[:4],
                "big",
            )
            for index in range(runner.EPISODE_COUNT)
        )
        self.assertEqual(runner.SEEDS, expected)
        self.assertEqual(len(set(expected)), runner.EPISODE_COUNT)

    def test_operational_ceiling_is_long(self) -> None:
        self.assertGreaterEqual(runner.MAX_TICKS, 100_000)


if __name__ == "__main__":
    unittest.main()
