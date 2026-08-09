"""Development-only long-horizon authority for the R3K pointer policy.

R3K deliberately keeps the frozen G4 proposal model and the calibrated
768-tick solvency certificate.  It changes the *final* decision authority:
two deterministic, exactly-safe alternatives are extended from the same
snapshot under the frozen continuation policy and compared through 2,048
ticks.  This makes score-neutral runway moves admissible without allowing a
known survival or long-score regression.

The module contains no sealed/test/canonical-run paths and performs no I/O.
Campaign runners bind all source and artifact identities externally.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


R3K_SCHEMA = "irisu-r3k-sustainable-runway-v1"
SHORT_HORIZON = 768
CHECKPOINT_HORIZONS = (128, 512, 2_048)


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _public_count(observation: Mapping[str, Any], name: str) -> int:
    value = observation.get(name, 0)
    return int(value) if type(value) in {int, float} else 0


@dataclass(frozen=True, slots=True)
class RunwayCheckpoint:
    horizon_ticks: int
    elapsed_ticks: int
    survived: bool
    score_gain: int
    clear_gain: int
    gauge: int
    minimum_gauge: int
    level: int

    def __post_init__(self) -> None:
        if (
            type(self.horizon_ticks) is not int
            or self.horizon_ticks not in CHECKPOINT_HORIZONS
            or type(self.elapsed_ticks) is not int
            or not 0 <= self.elapsed_ticks <= self.horizon_ticks
            or type(self.survived) is not bool
            or self.survived != (self.elapsed_ticks == self.horizon_ticks)
            or any(
                type(value) is not int
                for value in (
                    self.score_gain,
                    self.clear_gain,
                    self.gauge,
                    self.minimum_gauge,
                    self.level,
                )
            )
            or self.minimum_gauge > self.gauge
        ):
            raise ValueError("R3K runway checkpoint is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "horizon_ticks": self.horizon_ticks,
            "elapsed_ticks": self.elapsed_ticks,
            "survived": self.survived,
            "score_gain": self.score_gain,
            "clear_gain": self.clear_gain,
            "gauge": self.gauge,
            "minimum_gauge": self.minimum_gauge,
            "level": self.level,
        }


@dataclass(frozen=True, slots=True)
class RunwayOutcome:
    identity: object
    observation_sha256: str
    checkpoints: tuple[RunwayCheckpoint, ...]
    survival_ticks: int
    terminal: bool
    gauge_failure: bool
    invalid_actions: int
    continuation_rebind_failed: bool

    def __post_init__(self) -> None:
        ordinal = getattr(self.identity, "ordinal", None)
        if (
            type(ordinal) is not int
            or ordinal < 0
            or type(self.observation_sha256) is not str
            or len(self.observation_sha256) != 64
            or type(self.checkpoints) is not tuple
            or tuple(row.horizon_ticks for row in self.checkpoints)
            != CHECKPOINT_HORIZONS
            or any(type(row) is not RunwayCheckpoint for row in self.checkpoints)
            or type(self.survival_ticks) is not int
            or not 0 <= self.survival_ticks <= CHECKPOINT_HORIZONS[-1]
            or any(
                type(value) is not bool
                for value in (
                    self.terminal,
                    self.gauge_failure,
                    self.continuation_rebind_failed,
                )
            )
            or type(self.invalid_actions) is not int
            or self.invalid_actions < 0
        ):
            raise ValueError("R3K runway outcome is malformed")

    @property
    def final(self) -> RunwayCheckpoint:
        return self.checkpoints[-1]

    @property
    def tail_score_gain(self) -> int:
        return self.checkpoints[-1].score_gain - self.checkpoints[1].score_gain

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    def manifest(self) -> dict[str, object]:
        identity = getattr(self.identity, "manifest")()
        return {
            "schema": "irisu-r3k-runway-outcome-v1",
            "identity": identity,
            "observation_sha256": self.observation_sha256,
            "checkpoints": [row.manifest() for row in self.checkpoints],
            "survival_ticks": self.survival_ticks,
            "terminal": self.terminal,
            "gauge_failure": self.gauge_failure,
            "invalid_actions": self.invalid_actions,
            "continuation_rebind_failed": self.continuation_rebind_failed,
        }


def rollout_candidate(
    core: Any,
    env: Any,
    initial: Mapping[str, Any],
    candidate: Any,
    incumbent: Any,
    continuation_template: object,
    identity: object,
    *,
    observation_sha256: str,
    action_spec: object | None = None,
) -> RunwayOutcome:
    """Roll one candidate once and retain nested 128/512/2,048 targets.

    ``env`` must already be restored to the query snapshot.  The caller owns
    snapshot restoration between candidates.  The live continuation template
    is deep-copied and therefore cannot be mutated here.
    """

    if not isinstance(initial, Mapping):
        raise TypeError("R3K initial observation must be a mapping")
    resolved_spec = core.JOINT.ActionSpec() if action_spec is None else action_spec
    if resolved_spec.sha256 != core.JOINT.ActionSpec().sha256:
        raise ValueError("R3K requires the default action specification")
    start_tick = int(initial["tick"])
    start_score = _public_count(initial, "score")
    start_clears = _public_count(initial, "qualifying_clears")
    current = initial
    terminated = bool(initial.get("terminated", False))
    truncated = bool(initial.get("truncated", False))
    minimum_gauge = _public_count(initial, "gauge")
    invalid_actions = 0
    game_over = False

    def event_kind(event: Mapping[str, Any]) -> object:
        helper = getattr(core, "_event_kind", None)
        return helper(event) if callable(helper) else event.get("kind")

    def unit_step(action: Any) -> None:
        nonlocal current, terminated, truncated
        nonlocal minimum_gauge, invalid_actions, game_over
        kind = core.JOINT.ActionKind.parse(action.kind)
        duration = int(action.wait_ticks) if kind is core.JOINT.ActionKind.WAIT else 1
        for _ in range(duration):
            if terminated or truncated:
                break
            primitive = (
                core.JOINT.Action.wait(1)
                if kind is core.JOINT.ActionKind.WAIT
                else action
            )
            current, _reward, terminated, truncated, info = env.step(primitive)
            if not isinstance(current, Mapping) or not isinstance(info, Mapping):
                raise TypeError("R3K portable transition is malformed")
            minimum_gauge = min(minimum_gauge, _public_count(current, "gauge"))
            events = tuple(
                row for row in info.get("events", ()) if isinstance(row, Mapping)
            )
            invalid_actions += sum(
                event_kind(row) == getattr(core, "INVALID_ACTION", object())
                for row in events
            )
            game_over |= any(
                event_kind(row) == getattr(core, "GAME_OVER", object())
                for row in events
            )

    for action in candidate.decision.primitive_actions(resolved_spec):
        unit_step(action)
    elapsed = int(current["tick"]) - start_tick
    cooldown = int(core.BarrierConfig().cooldown_ticks)
    if not (terminated or truncated) and elapsed < cooldown:
        unit_step(core.JOINT.Action.wait(cooldown - elapsed))

    policy = copy.deepcopy(continuation_template)
    expected_identity = getattr(core, "BASE_SHA256", None)
    if (
        expected_identity is not None
        and getattr(policy, "artifact_sha256", None) != expected_identity
    ):
        raise RuntimeError("R3K continuation identity changed")
    rebound = core.commit_base_decision(
        policy, initial, incumbent.decision, candidate.decision
    )
    if type(rebound) is not bool:
        raise RuntimeError("R3K continuation rebind returned an inexact value")

    checkpoints: list[RunwayCheckpoint] = []

    def capture(horizon: int) -> RunwayCheckpoint:
        elapsed_now = min(int(current["tick"]) - start_tick, horizon)
        return RunwayCheckpoint(
            horizon,
            elapsed_now,
            elapsed_now == horizon,
            _public_count(current, "score") - start_score,
            _public_count(current, "qualifying_clears") - start_clears,
            _public_count(current, "gauge"),
            minimum_gauge,
            _public_count(current, "level"),
        )

    for horizon in CHECKPOINT_HORIZONS:
        while rebound and not (terminated or truncated or game_over):
            elapsed = int(current["tick"]) - start_tick
            if elapsed >= horizon:
                break
            decision = policy.predict(current)
            actions = core.ExactBranchEvaluator._primitive_tuple(
                decision, resolved_spec
            )
            for action in actions:
                remaining = horizon - (int(current["tick"]) - start_tick)
                if remaining <= 0 or terminated or truncated or game_over:
                    break
                kind = core.JOINT.ActionKind.parse(action.kind)
                duration = (
                    int(action.wait_ticks)
                    if kind is core.JOINT.ActionKind.WAIT
                    else 1
                )
                if duration > remaining:
                    if kind is not core.JOINT.ActionKind.WAIT:
                        raise RuntimeError("R3K action crossed a checkpoint horizon")
                    action = core.JOINT.Action.wait(remaining)
                unit_step(action)
        checkpoints.append(capture(horizon))

    return RunwayOutcome(
        identity,
        observation_sha256,
        tuple(checkpoints),
        min(int(current["tick"]) - start_tick, CHECKPOINT_HORIZONS[-1]),
        bool(terminated or truncated or game_over),
        bool(game_over or minimum_gauge <= 0),
        invalid_actions,
        not rebound,
    )


def extension_identities(
    proposal: object,
    short_outcomes: Sequence[object],
    predictions: Sequence[object],
) -> tuple[object, ...]:
    """Return incumbent plus a deterministic rescue/growth shortlist."""

    outcomes = tuple(short_outcomes)
    if not outcomes or getattr(outcomes[0].identity, "ordinal", None) != 0:
        raise ValueError("R3K exact outcomes must be incumbent-first")
    reserve = int(getattr(proposal, "reserve"))
    prediction_by_identity = {row.identity: row for row in predictions}
    eligible = [
        row
        for row in outcomes[1:]
        if bool(getattr(row, "absolute_safe"))
        and getattr(row, "b2") is not None
        and float(row.b2) >= reserve
        and float(getattr(row, "exact_score_advantage")) >= 0.0
    ]
    rescue = sorted(
        eligible,
        key=lambda row: (
            -float(row.b2),
            -float(row.exact_score_advantage),
            row.identity.ordinal,
        ),
    )

    def growth_key(row: object) -> tuple[float, float, float, int]:
        prediction = prediction_by_identity.get(row.identity)
        if prediction is None:
            conservative = -math.inf
        else:
            conservative = float(prediction.growth_mean) - float(
                prediction.growth_std
            )
        return (
            -conservative,
            -float(row.exact_score_advantage),
            -float(row.b2),
            row.identity.ordinal,
        )

    growth = sorted(eligible, key=growth_key)
    selected = [outcomes[0].identity]
    for rank in (rescue, growth):
        if rank and rank[0].identity not in selected:
            selected.append(rank[0].identity)
    return tuple(selected)


@dataclass(frozen=True, slots=True)
class SustainableSelection:
    mode: str
    status: str
    identity: object
    short_outcome_sha256: str
    runway_outcome_sha256: str
    reason: str

    def manifest(self) -> dict[str, object]:
        return {
            "schema": "irisu-r3k-sustainable-selection-v1",
            "mode": self.mode,
            "status": self.status,
            "identity": getattr(self.identity, "manifest")(),
            "short_outcome_sha256": self.short_outcome_sha256,
            "runway_outcome_sha256": self.runway_outcome_sha256,
            "reason": self.reason,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def select_sustainable(
    proposal: object,
    short_outcomes: Sequence[object],
    runway_outcomes: Sequence[RunwayOutcome],
) -> SustainableSelection:
    """Select with exact safety, survival, late score, then runway authority."""

    short = tuple(short_outcomes)
    long = tuple(runway_outcomes)
    if not short or not long or short[0].identity != long[0].identity:
        raise ValueError("R3K selection inputs are not incumbent-first")
    short_by_identity = {row.identity: row for row in short}
    long_by_identity = {row.identity: row for row in long}
    if set(long_by_identity) - set(short_by_identity):
        raise ValueError("R3K runway outcome lacks its short certificate")
    if len(short_by_identity) != len(short) or len(long_by_identity) != len(long):
        raise ValueError("R3K selection repeats an identity")
    observation_hashes = {row.observation_sha256 for row in long}
    if len(observation_hashes) != 1:
        raise ValueError("R3K runway outcomes cross query snapshots")

    incumbent_short = short[0]
    incumbent_long = long[0]
    reserve = int(getattr(proposal, "reserve"))
    mode = str(getattr(proposal, "mode"))
    eligible: list[tuple[object, RunwayOutcome]] = []
    for outcome in short[1:]:
        runway = long_by_identity.get(outcome.identity)
        if runway is None:
            continue
        if (
            not bool(outcome.absolute_safe)
            or outcome.b2 is None
            or runway.continuation_rebind_failed
            or runway.invalid_actions
            or runway.gauge_failure
            or runway.survival_ticks < incumbent_long.survival_ticks
        ):
            continue
        if mode == "growth" and (
            float(outcome.b2) < reserve
            or float(outcome.exact_score_advantage) < 0.0
            or runway.final.score_gain < incumbent_long.final.score_gain
        ):
            continue
        if (
            mode == "growth"
            and runway.final.score_gain == incumbent_long.final.score_gain
            and (
                runway.tail_score_gain < incumbent_long.tail_score_gain
                or float(outcome.b2) <= float(incumbent_short.b2)
            )
        ):
            continue
        eligible.append((outcome, runway))

    if mode == "rescue":
        eligible.sort(
            key=lambda pair: (
                -pair[1].survival_ticks,
                -float(pair[0].b2),
                -pair[1].final.score_gain,
                pair[0].identity.ordinal,
            )
        )
    else:
        eligible.sort(
            key=lambda pair: (
                -pair[1].final.score_gain,
                -pair[1].tail_score_gain,
                -float(pair[0].b2),
                -pair[1].final.clear_gain,
                pair[0].identity.ordinal,
            )
        )
    if eligible:
        outcome, runway = eligible[0]
        return SustainableSelection(
            mode,
            "selected-runway",
            outcome.identity,
            outcome.sha256,
            runway.sha256,
            "survival-nonregression-long-score-runway-lexicographic",
        )
    return SustainableSelection(
        mode,
        "incumbent-retained",
        incumbent_short.identity,
        incumbent_short.sha256,
        incumbent_long.sha256,
        "no-extended-alternative-cleared-all-guards",
    )
