#!/usr/bin/env python3
"""Clean-room model of the RNG shipped in DxLib 3.24f.

This is evidence tooling, not a dependency on the original DLL.  Integer
operations are deliberately masked to the 32-bit behavior of the x86 binary.
"""

from __future__ import annotations

import argparse


MASK32 = 0xFFFF_FFFF
STATE_WORDS = 624
PERIOD_OFFSET = 397


def u32(value: int) -> int:
    return value & MASK32


class DxLibRng:
    def __init__(self, seed: int):
        self.state = [0] * STATE_WORDS
        self.index = STATE_WORDS
        self.seed(seed)

    def seed(self, seed: int) -> None:
        value = u32(seed)
        for index in range(STATE_WORDS):
            following = u32(69_069 * value + 1)
            self.state[index] = (value & 0xFFFF_0000) | (following >> 16)
            value = u32(69_069 * following + 1)
        self.index = STATE_WORDS

    def _twist(self) -> None:
        # DxLib performs the canonical in-place MT twist.  The wrapped half
        # intentionally reads words already replaced by the first half.
        for index in range(STATE_WORDS - PERIOD_OFFSET):
            joined = ((self.state[index] & 0x8000_0000) |
                      (self.state[index + 1] & 0x7FFF_FFFF))
            word = self.state[index + PERIOD_OFFSET] ^ (joined >> 1)
            if joined & 1:
                word ^= 0x9908_B0DF
            self.state[index] = u32(word)
        for index in range(STATE_WORDS - PERIOD_OFFSET, STATE_WORDS - 1):
            joined = ((self.state[index] & 0x8000_0000) |
                      (self.state[index + 1] & 0x7FFF_FFFF))
            word = self.state[index + PERIOD_OFFSET - STATE_WORDS] ^ (joined >> 1)
            if joined & 1:
                word ^= 0x9908_B0DF
            self.state[index] = u32(word)
        joined = ((self.state[-1] & 0x8000_0000) |
                  (self.state[0] & 0x7FFF_FFFF))
        word = self.state[PERIOD_OFFSET - 1] ^ (joined >> 1)
        if joined & 1:
            word ^= 0x9908_B0DF
        self.state[-1] = u32(word)
        self.index = 0

    def raw_u32(self) -> int:
        if self.index >= STATE_WORDS:
            self._twist()
        value = self.state[self.index]
        self.index += 1
        value ^= value >> 11
        value ^= (value << 7) & 0x9D2C_5680
        value ^= (value << 15) & 0xEFC6_0000
        value ^= value >> 18
        return u32(value)

    def get_rand(self, maximum: int) -> int:
        if not 0 <= maximum <= 0x7FFF_FFFF:
            raise ValueError("maximum must be a nonnegative signed 32-bit int")
        # The DLL takes the high word of a 32x32 -> 64-bit product.  This is
        # range scaling, not remainder/modulo reduction.
        return (self.raw_u32() * (maximum + 1)) >> 32


def parse_int(text: str) -> int:
    return int(text, 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=parse_int, required=True)
    parser.add_argument("maxima", nargs="+", type=parse_int)
    args = parser.parse_args()
    rng = DxLibRng(args.seed)
    for maximum in args.maxima:
        print(rng.get_rand(maximum))


if __name__ == "__main__":
    main()
