"""Deterministic baselines for smoke tests, transfer probes, and benchmarks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from .env import Action


_MASK64 = (1 << 64) - 1


class Policy(Protocol):
    def reset(self, seed: int = 0) -> None: ...

    def act(self, observation: Mapping[str, Any]) -> Action: ...


class SplitMix64:
    """Tiny fully specified PRNG for cross-run baseline reproducibility."""

    def __init__(self, seed: int = 0) -> None:
        self.seed(seed)

    def seed(self, seed: int) -> None:
        if type(seed) is not int or not 0 <= seed <= _MASK64:
            raise ValueError("policy seed must fit in uint64")
        self.state = seed

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (value ^ (value >> 31)) & _MASK64

    def unit(self) -> float:
        return (self.next_u64() >> 11) * (1.0 / (1 << 53))

    def bounded(self, bound: int) -> int:
        if type(bound) is not int or not 1 <= bound <= 1 << 64:
            raise ValueError("bound must be an integer in [1, 2**64]")
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound


class RandomPolicy:
    """Seeded legal-action baseline with no privileged future information."""

    def __init__(
        self,
        seed: int = 0,
        *,
        shot_probability: float = 0.25,
        strong_probability: float = 0.5,
        max_wait_ticks: int = 5,
    ) -> None:
        if not 0.0 <= shot_probability <= 1.0:
            raise ValueError("shot_probability must be in [0, 1]")
        if not 0.0 <= strong_probability <= 1.0:
            raise ValueError("strong_probability must be in [0, 1]")
        if type(max_wait_ticks) is not int or not 1 <= max_wait_ticks <= 0xFFFFFFFF:
            raise ValueError("max_wait_ticks must fit the action uint32")
        self.shot_probability = float(shot_probability)
        self.strong_probability = float(strong_probability)
        self.max_wait_ticks = max_wait_ticks
        self._rng = SplitMix64(seed)

    def reset(self, seed: int = 0) -> None:
        self._rng.seed(seed)

    @property
    def rng_state(self) -> int:
        return self._rng.state

    def act(self, observation: Mapping[str, Any]) -> Action:
        del observation
        if self._rng.unit() >= self.shot_probability:
            return Action.wait(1 + self._rng.bounded(self.max_wait_ticks))
        strong = self._rng.unit() < self.strong_probability
        x = 94.0 + self._rng.unit() * 420.0
        y = 260.0 + self._rng.unit() * 120.0
        return Action.strong(x, y) if strong else Action.weak(x, y)


class MatcherShotPolicy:
    """Body-aware heuristic that prioritizes close same-color falling pairs."""

    def __init__(self, *, shot_period_ticks: int = 4, cursor_y: float = 380.0) -> None:
        if type(shot_period_ticks) is not int or shot_period_ticks < 1:
            raise ValueError("shot_period_ticks must be positive")
        if not 0.0 <= cursor_y <= 480.0:
            raise ValueError("cursor_y must be in the 640x480 client")
        self.shot_period_ticks = shot_period_ticks
        self.cursor_y = float(cursor_y)

    def reset(self, seed: int = 0) -> None:
        if type(seed) is not int or not 0 <= seed <= _MASK64:
            raise ValueError("policy seed must fit in uint64")

    def act(self, observation: Mapping[str, Any]) -> Action:
        tick = int(observation.get("tick", 0))
        until_shot = (-tick) % self.shot_period_ticks
        if until_shot:
            return Action.wait(until_shot)

        pieces = [
            body
            for body in observation.get("bodies", ())
            if body.get("kind") == "piece"
            and body.get("lifecycle") in ("scripted_falling", "dynamic_fresh")
        ]
        scripted = [body for body in pieces if body["lifecycle"] == "scripted_falling"]
        if not pieces:
            return Action.wait(self.shot_period_ticks)

        pairs: list[tuple[float, int, int, Mapping[str, Any], Mapping[str, Any]]] = []
        for index, first in enumerate(pieces):
            for second in pieces[index + 1 :]:
                if first.get("color") != second.get("color"):
                    continue
                distance = abs(float(first["x"]) - float(second["x"])) + 0.25 * abs(
                    float(first["y"]) - float(second["y"])
                )
                pairs.append(
                    (
                        distance,
                        min(int(first["id"]), int(second["id"])),
                        max(int(first["id"]), int(second["id"])),
                        first,
                        second,
                    )
                )

        if pairs:
            _, _, _, first, second = min(pairs, key=lambda value: value[:3])
            target = max(
                (first, second), key=lambda body: (float(body["y"]), -int(body["id"]))
            )
        elif scripted:
            target = max(
                scripted, key=lambda body: (float(body["y"]), -int(body["id"]))
            )
        else:
            target = min(
                pieces,
                key=lambda body: (abs(float(body["x"]) - 304.0), int(body["id"])),
            )

        target_x = min(640.0, max(0.0, float(target["x"])))
        projectiles = [
            body
            for body in observation.get("bodies", ())
            if body.get("kind") == "projectile"
            and abs(float(body["x"]) - target_x)
            <= max(8.0, float(target["size"]) * 0.5)
            and float(body["y"]) >= float(target["y"])
        ]
        if projectiles:
            return Action.wait(self.shot_period_ticks)
        distance_up = self.cursor_y - float(target["y"])
        return (
            Action.strong(target_x, self.cursor_y)
            if distance_up > 150.0
            else Action.weak(target_x, self.cursor_y)
        )


class LongWaitPolicy:
    """Deterministic no-click diagnostic baseline."""

    def __init__(self, wait_ticks: int = 100) -> None:
        if type(wait_ticks) is not int or not 1 <= wait_ticks <= 0xFFFFFFFF:
            raise ValueError("wait_ticks must fit the action uint32")
        self.wait_ticks = wait_ticks

    def reset(self, seed: int = 0) -> None:
        if type(seed) is not int or not 0 <= seed <= _MASK64:
            raise ValueError("policy seed must fit in uint64")

    def act(self, observation: Mapping[str, Any]) -> Action:
        del observation
        return Action.wait(self.wait_ticks)


class DirectMatcherPolicy(MatcherShotPolicy):
    """Named direct-matcher baseline used by the frozen R3b protocol."""


class SideEjectorPolicy:
    """Strong-shot outer pieces to encourage safe lateral ejection."""

    def __init__(self, *, shot_period_ticks: int = 3, cursor_y: float = 390.0) -> None:
        if type(shot_period_ticks) is not int or shot_period_ticks < 1:
            raise ValueError("shot_period_ticks must be positive")
        if not 0.0 <= cursor_y <= 480.0:
            raise ValueError("cursor_y must be in the 640x480 client")
        self.shot_period_ticks = shot_period_ticks
        self.cursor_y = float(cursor_y)

    def reset(self, seed: int = 0) -> None:
        if type(seed) is not int or not 0 <= seed <= _MASK64:
            raise ValueError("policy seed must fit in uint64")

    def act(self, observation: Mapping[str, Any]) -> Action:
        tick = int(observation.get("tick", 0))
        until_shot = (-tick) % self.shot_period_ticks
        if until_shot:
            return Action.wait(until_shot)
        pieces = [
            body
            for body in observation.get("bodies", ())
            if body.get("kind") == "piece"
            and body.get("lifecycle")
            in ("scripted_falling", "dynamic_fresh", "dynamic_rotten")
        ]
        if not pieces:
            return Action.wait(self.shot_period_ticks)
        target = max(
            pieces,
            key=lambda body: (
                abs(float(body["x"]) - 304.0),
                float(body["y"]),
                -int(body["id"]),
            ),
        )
        return Action.strong(min(640.0, max(0.0, float(target["x"]))), self.cursor_y)


class ImminentRotHazardPolicy:
    """Escalate to rapid strong shots when visible gauge safety is low."""

    def __init__(
        self,
        *,
        gauge_threshold: float = 0.35,
        cursor_y: float = 390.0,
    ) -> None:
        if not 0.0 < gauge_threshold < 1.0:
            raise ValueError("gauge_threshold must lie in (0, 1)")
        if not 0.0 <= cursor_y <= 480.0:
            raise ValueError("cursor_y must be in the 640x480 client")
        self.gauge_threshold = float(gauge_threshold)
        self.cursor_y = float(cursor_y)
        self._normal = MatcherShotPolicy(shot_period_ticks=4, cursor_y=cursor_y)

    def reset(self, seed: int = 0) -> None:
        self._normal.reset(seed)

    def act(self, observation: Mapping[str, Any]) -> Action:
        gauge = int(observation.get("gauge", 0))
        gauge_max = max(1, int(observation.get("gauge_max", 1)))
        if gauge / gauge_max >= self.gauge_threshold:
            return self._normal.act(observation)
        pieces = [
            body
            for body in observation.get("bodies", ())
            if body.get("kind") == "piece"
            and body.get("lifecycle")
            in ("scripted_falling", "dynamic_fresh", "dynamic_rotten")
        ]
        if not pieces:
            return Action.wait(1)
        target = max(
            pieces,
            key=lambda body: (float(body["y"]), -int(body["id"])),
        )
        return Action.strong(min(640.0, max(0.0, float(target["x"]))), self.cursor_y)


ScriptedPolicy = MatcherShotPolicy
