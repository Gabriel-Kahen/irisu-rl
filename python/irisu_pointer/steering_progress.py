"""Public-observation progress tracking for directed steering pairs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from typing import Any


@dataclass(frozen=True, order=True, slots=True)
class DirectedPair:
    """One source-to-destination identity from public body IDs."""

    source_id: int
    destination_id: int

    def __post_init__(self) -> None:
        for value in (self.source_id, self.destination_id):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError("directed pair IDs must be nonnegative integers")
        if self.source_id == self.destination_id:
            raise ValueError("directed pair source and destination must differ")


class PairProgressStatus(str, Enum):
    """Result of observing one attempted pair at a safe boundary."""

    PROGRESSED = "progressed"
    STALLED = "stalled"
    RESOLVED = "resolved"


@dataclass(frozen=True, slots=True)
class PairProgressAssessment:
    pair: DirectedPair
    status: PairProgressStatus
    baseline_gap: float
    observed_gap: float | None
    minimum_closure: float


@dataclass(frozen=True, slots=True)
class _Attempt:
    pair: DirectedPair
    gap: float
    minimum_closure: float


@dataclass(frozen=True, slots=True)
class _Stall:
    best_gap: float
    minimum_closure: float


def _plain_id(value: Any) -> int:
    if isinstance(value, Integral) and not isinstance(value, bool):
        result = int(value)
    else:
        item = getattr(value, "item", None)
        if not callable(item):
            raise TypeError("public body ID must be an integer")
        return _plain_id(item())
    if result < 0:
        raise ValueError("public body ID must be nonnegative")
    return result


def _plain_float(value: Any, name: str) -> float:
    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        item = getattr(value, "item", None)
        if not callable(item):
            raise TypeError(f"public body {name} must be numeric")
        result = float(item())
    if not math.isfinite(result):
        raise ValueError(f"public body {name} must be finite")
    return result


def _public_bodies(
    observation: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    values = observation.get("bodies", ())
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise TypeError("public observation bodies must be a sequence")
    result: dict[int, Mapping[str, Any]] = {}
    for value in values:
        if not isinstance(value, Mapping):
            continue
        identifier = _plain_id(value.get("id"))
        if identifier in result:
            raise ValueError("public observation body IDs must be unique")
        result[identifier] = value
    return result


def _surface_gap(
    source: Mapping[str, Any], destination: Mapping[str, Any]
) -> tuple[float, float]:
    source_x = _plain_float(source.get("effect_x", source.get("x")), "x")
    source_y = _plain_float(source.get("effect_y", source.get("y")), "y")
    destination_x = _plain_float(
        destination.get("effect_x", destination.get("x")), "x"
    )
    destination_y = _plain_float(
        destination.get("effect_y", destination.get("y")), "y"
    )
    source_size = _plain_float(source.get("size"), "size")
    destination_size = _plain_float(destination.get("size"), "size")
    if source_size <= 0.0 or destination_size <= 0.0:
        raise ValueError("public body size must be positive")
    gap = max(
        0.0,
        math.hypot(destination_x - source_x, destination_y - source_y)
        - (source_size + destination_size) / 2.0,
    )
    return gap, min(source_size, destination_size)


class DirectedPairProgressTracker:
    """Retarget pairs that fail to make observable geometric progress."""

    def __init__(self, *, minimum_closure_sizes: float = 0.05) -> None:
        if (
            isinstance(minimum_closure_sizes, bool)
            or not isinstance(minimum_closure_sizes, Real)
            or not math.isfinite(float(minimum_closure_sizes))
            or float(minimum_closure_sizes) <= 0.0
        ):
            raise ValueError("minimum pair closure must be finite and positive")
        self.minimum_closure_sizes = float(minimum_closure_sizes)
        self._attempt: _Attempt | None = None
        self._stalled: dict[DirectedPair, _Stall] = {}

    @property
    def pending_pair(self) -> DirectedPair | None:
        return None if self._attempt is None else self._attempt.pair

    @property
    def stalled_pairs(self) -> tuple[DirectedPair, ...]:
        return tuple(sorted(self._stalled))

    def reset(self) -> None:
        self._attempt = None
        self._stalled.clear()

    def begin(
        self,
        observation: Mapping[str, Any],
        source_id: int,
        destination_id: int,
    ) -> None:
        """Record the public geometry immediately before one correction."""

        if self._attempt is not None:
            raise RuntimeError("previous directed pair attempt is still pending")
        pair = DirectedPair(source_id, destination_id)
        bodies = _public_bodies(observation)
        try:
            gap, scale = _surface_gap(
                bodies[pair.source_id], bodies[pair.destination_id]
            )
        except KeyError as exc:
            raise ValueError("directed pair body is absent from the observation") from exc
        if self.is_stalled(observation, pair.source_id, pair.destination_id):
            raise RuntimeError("cannot begin a correction for a stalled pair")
        self._attempt = _Attempt(
            pair,
            gap,
            self.minimum_closure_sizes * scale,
        )

    def assess(
        self, observation: Mapping[str, Any]
    ) -> PairProgressAssessment | None:
        """Assess and consume the pending correction at a safe boundary."""

        attempt = self._attempt
        if attempt is None:
            return None
        self._attempt = None
        bodies = _public_bodies(observation)
        source = bodies.get(attempt.pair.source_id)
        destination = bodies.get(attempt.pair.destination_id)
        if source is None or destination is None:
            self._stalled.pop(attempt.pair, None)
            return PairProgressAssessment(
                attempt.pair,
                PairProgressStatus.RESOLVED,
                attempt.gap,
                None,
                attempt.minimum_closure,
            )
        gap, _ = _surface_gap(source, destination)
        if gap <= attempt.gap - attempt.minimum_closure:
            self._stalled.pop(attempt.pair, None)
            status = PairProgressStatus.PROGRESSED
        else:
            self._stalled[attempt.pair] = _Stall(
                min(attempt.gap, gap),
                attempt.minimum_closure,
            )
            status = PairProgressStatus.STALLED
        return PairProgressAssessment(
            attempt.pair,
            status,
            attempt.gap,
            gap,
            attempt.minimum_closure,
        )

    def is_stalled(
        self,
        observation: Mapping[str, Any],
        source_id: int,
        destination_id: int,
    ) -> bool:
        """Return whether the pair remains stalled in the current geometry."""

        pair = DirectedPair(source_id, destination_id)
        stall = self._stalled.get(pair)
        if stall is None:
            return False
        bodies = _public_bodies(observation)
        source = bodies.get(pair.source_id)
        destination = bodies.get(pair.destination_id)
        if source is None or destination is None:
            del self._stalled[pair]
            return False
        gap, _ = _surface_gap(source, destination)
        if gap <= stall.best_gap - stall.minimum_closure:
            del self._stalled[pair]
            return False
        return True

    def prune(self, observation: Mapping[str, Any]) -> None:
        """Forget stalled identities no longer present in public state."""

        live = set(_public_bodies(observation))
        for pair in tuple(self._stalled):
            if pair.source_id not in live or pair.destination_id not in live:
                del self._stalled[pair]


__all__ = [
    "DirectedPair",
    "DirectedPairProgressTracker",
    "PairProgressAssessment",
    "PairProgressStatus",
]
