#!/usr/bin/env python3
"""Deterministic vectors measured from the shipped v2.03 DxLib.dll."""

import unittest

from dxlib_rng import DxLibRng


MAXIMA = [100, 12, 69, 404, 1000, 3, 100, 1000, 5, 100]
VECTORS = {
    0: [11, 11, 36, 264, 666, 0, 28, 813, 0, 71],
    1: [83, 12, 3, 55, 782, 1, 82, 879, 1, 99],
    0x1234_5678: [47, 9, 0, 302, 168, 1, 78, 895, 0, 59],
    0x3FFF_FFFF: [64, 8, 6, 353, 16, 2, 2, 684, 5, 59],
}


class DxLibRngTests(unittest.TestCase):
    def test_measured_vectors(self) -> None:
        for seed, expected in VECTORS.items():
            with self.subTest(seed=seed):
                rng = DxLibRng(seed)
                actual = [rng.get_rand(maximum) for maximum in MAXIMA]
                self.assertEqual(actual, expected)

    def test_range_is_inclusive(self) -> None:
        rng = DxLibRng(0)
        for maximum in (0, 1, 3, 12, 69, 100, 404, 1000):
            for _ in range(2000):
                self.assertLessEqual(rng.get_rand(maximum), maximum)


if __name__ == "__main__":
    unittest.main()
