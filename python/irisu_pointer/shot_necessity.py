"""Exact wait-dominance gate for learned steering shots.

The learned policy proposes a semantic shot.  This module compares that shot
with one cooldown of restraint from the identical simulator and policy state.
It executes a shot only when the short counterfactual shows strict utility or
survival benefit; exact ties are waits.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import ActionSpec, SemanticAction, SemanticActionKind


@dataclass(frozen=True, slots=True)
class WaitDominanceConfig:
    probe_ticks: int = 128
    wait_ticks: int = 16
    gauge_advantage: int = 16

    def __post_init__(self) -> None:
        for name in ("probe_ticks", "wait_ticks", "gauge_advantage"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def manifest(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    survival_ticks: int
    score: int
    clears: int
    final_gauge: int
    minimum_gauge: int
    terminated: bool
    truncated: bool

    def manifest(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GateVerdict:
    execute_shot: bool
    reason: str
    shot: ProbeOutcome
    wait: ProbeOutcome
    restore_checks: int

    def manifest(self) -> dict[str, object]:
        return {
            "execute_shot": self.execute_shot,
            "reason": self.reason,
            "shot": self.shot.manifest(),
            "wait": self.wait.manifest(),
            "restore_checks": self.restore_checks,
        }


def choose_shot(
    shot: ProbeOutcome,
    wait: ProbeOutcome,
    *,
    gauge_advantage: int = 16,
) -> tuple[bool, str]:
    """Prefer restraint unless the proposed shot has strict causal benefit."""

    if shot.survival_ticks != wait.survival_ticks:
        return (
            shot.survival_ticks > wait.survival_ticks,
            "shot-survival" if shot.survival_ticks > wait.survival_ticks else "wait-survival",
        )
    shot_failed = shot.terminated or shot.truncated
    wait_failed = wait.terminated or wait.truncated
    if shot_failed != wait_failed:
        return (not shot_failed, "shot-rescue" if wait_failed else "wait-safer")
    if shot.clears > wait.clears:
        return True, "shot-clear"
    if shot.score > wait.score:
        return True, "shot-score"
    if shot.final_gauge > wait.final_gauge + gauge_advantage:
        return True, "shot-gauge"
    return False, "wait-tie-or-no-benefit"


class ExactWaitDominanceGate:
    """Transactional exact probe around an environment and learned policy."""

    def __init__(
        self,
        primitive_actions: Callable[[object], tuple[object, ...]],
        *,
        config: WaitDominanceConfig | None = None,
        action_spec: ActionSpec | None = None,
    ) -> None:
        self.primitive_actions = primitive_actions
        self.config = WaitDominanceConfig() if config is None else config
        self.action_spec = ActionSpec() if action_spec is None else action_spec

    def wait_decision(self, reason: str) -> SteeringDecision:
        return SteeringDecision(
            self.action_spec.validate(SemanticAction.wait(self.config.wait_ticks)),
            SteeringIntent.WAIT,
            reason=reason,
        )

    def _advance(
        self,
        env: object,
        observation: Mapping[str, Any],
        policy: object,
        first: object,
    ) -> ProbeOutcome:
        start = int(observation["tick"])
        current = observation
        minimum_gauge = int(current["gauge"])
        terminated = truncated = False
        decision: object | None = first
        while int(current["tick"]) - start < self.config.probe_ticks and not (
            terminated or truncated
        ):
            if decision is None:
                decision = policy.predict(current)
            for action in self.primitive_actions(decision):
                kind = SemanticActionKind(int(action.kind))
                duration = int(action.wait_ticks) if kind is SemanticActionKind.WAIT else 1
                remaining = self.config.probe_ticks - (int(current["tick"]) - start)
                if remaining <= 0:
                    break
                if duration > remaining:
                    if kind is not SemanticActionKind.WAIT:
                        break
                    action = self.action_spec.press(SemanticAction.wait(remaining))
                    duration = remaining
                for _ in range(duration):
                    primitive = (
                        self.action_spec.press(SemanticAction.wait(1))
                        if kind is SemanticActionKind.WAIT
                        else action
                    )
                    current, _reward, terminated, truncated, _info = env.step(primitive)
                    minimum_gauge = min(minimum_gauge, int(current["gauge"]))
                    if terminated or truncated:
                        break
                if terminated or truncated or int(current["tick"]) - start >= self.config.probe_ticks:
                    break
            decision = None
        return ProbeOutcome(
            int(current["tick"]) - start,
            int(current["score"]),
            int(current.get("qualifying_clear_count", 0)),
            int(current["gauge"]),
            minimum_gauge,
            bool(terminated or current.get("terminated", False)),
            bool(truncated or current.get("truncated", False)),
        )

    def evaluate(
        self,
        env: object,
        observation: Mapping[str, Any],
        policy_before: object,
        policy_after_shot: object,
        shot_decision: SteeringDecision,
    ) -> GateVerdict:
        if not shot_decision.is_shot:
            raise ValueError("wait-dominance gate requires a proposed shot")
        snapshot = env.clone_state()
        expected_hash = env.state_hash()
        restore_checks = 0

        def restore() -> Mapping[str, Any]:
            nonlocal restore_checks
            restored = env.restore_state(snapshot)
            restore_checks += 1
            if env.clone_state() != snapshot or env.state_hash() != expected_hash:
                raise RuntimeError("wait-dominance transactional restore mismatch")
            return restored

        try:
            shot = self._advance(
                env,
                restore(),
                copy.deepcopy(policy_after_shot),
                shot_decision,
            )
            wait = self._advance(
                env,
                restore(),
                copy.deepcopy(policy_before),
                self.wait_decision("counterfactual restraint probe"),
            )
        finally:
            restore()
        execute, reason = choose_shot(
            shot,
            wait,
            gauge_advantage=self.config.gauge_advantage,
        )
        return GateVerdict(execute, reason, shot, wait, restore_checks)


__all__ = [
    "ExactWaitDominanceGate",
    "GateVerdict",
    "ProbeOutcome",
    "WaitDominanceConfig",
    "choose_shot",
]
