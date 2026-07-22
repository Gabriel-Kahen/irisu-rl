"""Disjoint, checkpointable, nonrepeating uint32 seed schedules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from typing import Mapping


@dataclass(frozen=True, slots=True)
class SeedSplit:
    name: str
    start: int
    size: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.size <= 0 or self.start + self.size > 2**32:
            raise ValueError("seed split must fit uint32")
        if self.size & (self.size - 1):
            raise ValueError("seed split size must be a power of two")


SEED_SPLITS_V1 = {
    "train": SeedSplit("train", 0x00000000, 1 << 30),
    "validation": SeedSplit("validation", 0x40000000, 1 << 28),
    "calibration": SeedSplit("calibration", 0x50000000, 1 << 28),
    "test": SeedSplit("test", 0x60000000, 1 << 29),
}


def validate_seed_splits(splits: Mapping[str, SeedSplit]) -> None:
    ordered = sorted(splits.values(), key=lambda value: value.start)
    if len({value.name for value in ordered}) != len(ordered):
        raise ValueError("duplicate seed split name")
    for left, right in zip(ordered, ordered[1:]):
        if left.start + left.size > right.start:
            raise ValueError("seed splits overlap")


validate_seed_splits(SEED_SPLITS_V1)


@dataclass(frozen=True, slots=True)
class SeedReservation:
    cursor: int
    seeds: tuple[int, ...]


class SeedAllocator:
    """Affine permutation of a power-of-two split, with transactional reserve."""

    version = "seed-allocator-v1"

    def __init__(self, split: str = "train", *, key: int = 0, cursor: int = 0) -> None:
        if split not in SEED_SPLITS_V1:
            raise ValueError(f"unknown seed split: {split}")
        if not isinstance(key, Integral) or isinstance(key, bool) or not 0 <= int(key) < 2**64:
            raise ValueError("allocator key must fit uint64")
        self.split = SEED_SPLITS_V1[split]
        self.key = int(key)
        self._multiplier = ((self.key * 0x9E3779B97F4A7C15) | 1) & (self.split.size - 1)
        self._offset = ((self.key ^ (self.key >> 32)) * 0x85EBCA6B) & (self.split.size - 1)
        self._cursor = 0
        self.load_state_dict({"version": self.version, "split": split, "key": self.key, "cursor": cursor})

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def manifest_sha256(self) -> str:
        payload = {
            name: {"start": value.start, "size": value.size}
            for name, value in sorted(SEED_SPLITS_V1.items())
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def reserve(self, count: int) -> SeedReservation:
        if not isinstance(count, Integral) or isinstance(count, bool) or count < 0:
            raise ValueError("reservation count must be nonnegative")
        count = int(count)
        if self._cursor + count > self.split.size:
            raise RuntimeError("seed split exhausted")
        mask = self.split.size - 1
        seeds = tuple(
            self.split.start + ((self._multiplier * index + self._offset) & mask)
            for index in range(self._cursor, self._cursor + count)
        )
        return SeedReservation(self._cursor, seeds)

    def commit(self, reservation: SeedReservation) -> None:
        if reservation.cursor != self._cursor:
            raise ValueError("stale or out-of-order seed reservation")
        if reservation != self.reserve(len(reservation.seeds)):
            raise ValueError("seed reservation does not match allocator schedule")
        self._cursor += len(reservation.seeds)

    def take(self, count: int) -> tuple[int, ...]:
        reservation = self.reserve(count)
        self.commit(reservation)
        return reservation.seeds

    def state_dict(self) -> dict[str, int | str]:
        return {"version": self.version, "split": self.split.name, "key": self.key, "cursor": self._cursor}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {"version", "split", "key", "cursor"}
        if set(state) != expected:
            raise ValueError("seed allocator state keys do not match version")
        if state["version"] != self.version or state["split"] != self.split.name or state["key"] != self.key:
            raise ValueError("seed allocator state identity mismatch")
        cursor = state["cursor"]
        if not isinstance(cursor, Integral) or isinstance(cursor, bool) or not 0 <= int(cursor) <= self.split.size:
            raise ValueError("invalid seed allocator cursor")
        self._cursor = int(cursor)
