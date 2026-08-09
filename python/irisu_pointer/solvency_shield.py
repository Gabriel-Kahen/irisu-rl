"""Candidate-local exact two-renewal solvency shielding.

This module is development-only.  It evaluates the corrected joint-v2
directed-pair × geometry candidates transactionally, then exposes score only
after the exact renewable-solvency certificate has passed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

from irisu_env import Action, ActionKind, EventKind
from irisu_rl.actions import ActionSpec

from .joint_planner import (
    JOINT_PLANNER_VERSION,
    JointCandidate,
    JointPairGeometrySearch,
    JointPlannerConfig,
    _commit_base_decision,
    _public_signature,
)
from .steering import SteeringDecision


SOLVENCY_SHIELD_VERSION = "r3g-analytic-candidate-local-b2-v1"
_NORMAL_RENEWAL = "normal burst landing"
_SPECIAL_RENEWAL = "special color clear"
_ROT_PENALTY = "normal rot penalty"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _plain_int(value: Any, name: str, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    item = getattr(value, "item", None)
    if callable(item):
        return _plain_int(item(), name, default)
    raise TypeError(f"{name} must be an integer")


def _event_kind(event: Mapping[str, Any]) -> int | None:
    raw = event.get("kind")
    if isinstance(raw, Integral) and not isinstance(raw, bool):
        return int(raw)
    name = event.get("kind_name")
    if isinstance(name, str):
        try:
            return int(EventKind[name.upper()])
        except KeyError:
            return None
    return None


def _body_map(
    observation: Mapping[str, Any],
) -> dict[int, Mapping[str, Any]]:
    supplied = observation.get("bodies", ())
    if not isinstance(supplied, Sequence) or isinstance(supplied, (str, bytes)):
        raise TypeError("public bodies must be a sequence")
    output: dict[int, Mapping[str, Any]] = {}
    for body in supplied:
        if not isinstance(body, Mapping):
            continue
        identifier = _plain_int(body.get("id"), "body id", -1)
        if identifier < 0 or identifier in output:
            raise ValueError("public body IDs must be unique and nonnegative")
        output[identifier] = body
    return output


def passive_drain_unit(level: int) -> int:
    if type(level) is not int or level < 1:
        raise ValueError("level must be a positive integer")
    return min(level, 99) // 10 + 1


def rot_penalty(level: int) -> int:
    if type(level) is not int or level < 1:
        raise ValueError("level must be a positive integer")
    return 1_800 + 20 * min(level, 99)


def visible_liability_ids(
    observation: Mapping[str, Any],
) -> frozenset[int]:
    return frozenset(
        identifier
        for identifier, body in _body_map(observation).items()
        if body.get("kind") == "piece"
        and str(body.get("lifecycle", "")) not in {"rotten", "deleted"}
        and _plain_int(body.get("rot_timer"), "rot timer") > 0
    )


def renewal_epoch(
    events: Sequence[Mapping[str, Any]],
) -> tuple[bool, int]:
    """Return one coalesced renewable epoch and its gross positive credits."""

    normal = [
        _plain_int(event.get("value"), "normal renewal")
        for event in events
        if _event_kind(event) == int(EventKind.GAUGE_CHANGED)
        and str(event.get("detail", "")) == _NORMAL_RENEWAL
        and _plain_int(event.get("value"), "normal renewal") > 0
    ]
    special = [
        _plain_int(event.get("value"), "special renewal")
        for event in events
        if _event_kind(event) == int(EventKind.GAUGE_CHANGED)
        and str(event.get("detail", "")) == _SPECIAL_RENEWAL
        and _plain_int(event.get("value"), "special renewal") > 0
    ]
    anchored_special = bool(special) and any(
        _event_kind(event) == int(EventKind.CLEARED)
        and str(event.get("detail", "")) == _SPECIAL_RENEWAL
        for event in events
    )
    if special and not anchored_special:
        raise RuntimeError("special renewal credits lack their causal clear")
    # The protocol defines an epoch as a distinct tick, so all same-tick
    # normal and special credits coalesce to one event.
    return bool(normal or anchored_special), sum(normal) + sum(special)


def _primitive_action_key(
    decision: SteeringDecision,
    action_spec: ActionSpec,
) -> tuple[tuple[int, str, str, int], ...]:
    return tuple(
        (
            int(ActionKind.parse(action.kind)),
            float(action.cursor_x).hex(),
            float(action.cursor_y).hex(),
            int(action.wait_ticks),
        )
        for action in decision.primitive_actions(action_spec)
    )


@dataclass(frozen=True, slots=True)
class SolvencyBarrierConfig:
    required_renewal_epochs: int = 2
    branch_tick_cap: int = 1_024
    query_interval_ticks: int = 256
    terminal_floor: int = 1

    def __post_init__(self) -> None:
        if self.required_renewal_epochs != 2:
            raise ValueError("R3G requires exactly two certification renewals")
        if (
            type(self.branch_tick_cap) is not int
            or self.branch_tick_cap < 2
            or type(self.query_interval_ticks) is not int
            or self.query_interval_ticks < 1
            or self.terminal_floor != 1
        ):
            raise ValueError("invalid R3G barrier budget")

    def manifest(self) -> dict[str, object]:
        return {
            "version": SOLVENCY_SHIELD_VERSION,
            "required_renewal_epochs": self.required_renewal_epochs,
            "branch_tick_cap": self.branch_tick_cap,
            "query_interval_ticks": self.query_interval_ticks,
            "terminal_floor": self.terminal_floor,
            "renewal_epoch": (
                "distinct tick with positive normal-burst or special-color "
                "gauge change; same-tick gains coalesced"
            ),
            "margin": (
                "post-tick gauge - terminal_floor - dynamically repriced "
                "unpaid visible rot liabilities"
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(slots=True)
class RotLiabilityLedger:
    """Candidate-private visible rot liabilities with exact terminal states."""

    outstanding: dict[int, int] = field(default_factory=dict)
    paid: dict[int, int] = field(default_factory=dict)
    discharged: dict[int, int] = field(default_factory=dict)
    ever_seen: set[int] = field(default_factory=set)
    additions: int = 0

    def observe_initial(self, observation: Mapping[str, Any]) -> None:
        tick = _plain_int(observation.get("tick"), "initial tick")
        if self.outstanding or self.paid or self.discharged or self.ever_seen:
            raise RuntimeError("liability ledger was initialized twice")
        for identifier in sorted(visible_liability_ids(observation)):
            body = _body_map(observation)[identifier]
            timer = _plain_int(body.get("rot_timer"), "rot timer")
            self.outstanding[identifier] = tick + 41 - timer
            self.ever_seen.add(identifier)
            self.additions += 1

    def reconcile_tick(
        self,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
    ) -> None:
        """Apply payments in event order, then discharge/add from post-state."""

        tick = _plain_int(after.get("tick"), "post tick")
        level = _plain_int(after.get("level"), "post level")
        expected_penalty = -rot_penalty(level)
        index = 0
        while index < len(events):
            event = events[index]
            if _event_kind(event) != int(EventKind.ROTTEN):
                index += 1
                continue
            identifier = _plain_int(event.get("a"), "rotten body id", -1)
            if identifier not in self.outstanding:
                raise RuntimeError("rot payment lacks an outstanding liability")
            if identifier in self.paid or identifier in self.discharged:
                raise RuntimeError("rot liability was removed more than once")
            if index + 1 >= len(events):
                raise RuntimeError("rot event lacks its ordered gauge penalty")
            penalty = events[index + 1]
            if (
                _event_kind(penalty) != int(EventKind.GAUGE_CHANGED)
                or _plain_int(penalty.get("a"), "penalty body id", -1)
                != identifier
                or str(penalty.get("detail", "")) != _ROT_PENALTY
                or _plain_int(penalty.get("value"), "rot penalty")
                != expected_penalty
            ):
                raise RuntimeError("rot payment and gauge penalty disagree")
            due = self.outstanding.pop(identifier)
            if tick != due:
                raise RuntimeError("rot liability paid outside its exact deadline")
            self.paid[identifier] = tick
            index += 2

        bodies = _body_map(after)
        for identifier in tuple(sorted(self.outstanding)):
            body = bodies.get(identifier)
            if body is None or str(body.get("lifecycle", "")) == "deleted":
                self.outstanding.pop(identifier)
                self.discharged[identifier] = tick
                continue
            if str(body.get("lifecycle", "")) == "rotten":
                raise RuntimeError("rotten transition lacked an ordered payment")
            timer = _plain_int(body.get("rot_timer"), "rot timer")
            if timer <= 0:
                raise RuntimeError("visible rot timer regressed before resolution")
            if tick + 41 - timer != self.outstanding[identifier]:
                raise RuntimeError("visible rot deadline changed")
            if tick >= self.outstanding[identifier]:
                raise RuntimeError("rot liability remained visible past its deadline")

        for identifier in sorted(visible_liability_ids(after)):
            if identifier in self.outstanding:
                continue
            if identifier in self.paid or identifier in self.discharged:
                raise RuntimeError("resolved rot liability reappeared")
            if identifier in self.ever_seen:
                raise RuntimeError("rot liability was re-added")
            timer = _plain_int(bodies[identifier].get("rot_timer"), "rot timer")
            self.outstanding[identifier] = tick + 41 - timer
            self.ever_seen.add(identifier)
            self.additions += 1

    def reserve(self, level: int) -> int:
        return len(self.outstanding) * rot_penalty(level)

    def manifest(self) -> dict[str, object]:
        return {
            "outstanding": [
                {"body_id": key, "due_tick": value}
                for key, value in sorted(self.outstanding.items())
            ],
            "paid": [
                {"body_id": key, "tick": value}
                for key, value in sorted(self.paid.items())
            ],
            "discharged": [
                {"body_id": key, "tick": value}
                for key, value in sorted(self.discharged.items())
            ],
            "ever_seen": sorted(self.ever_seen),
            "additions": self.additions,
        }


@dataclass(frozen=True, slots=True)
class FrozenPolicyState:
    cooldown_until: int
    last_tick: int | None
    last_decision: SteeringDecision | None
    progress: object

    @classmethod
    def capture(cls, policy: object) -> FrozenPolicyState:
        required = ("_cooldown_until", "_last_tick", "_last_decision", "_progress")
        if any(not hasattr(policy, name) for name in required):
            raise TypeError("frozen-v5 policy state is not clonable")
        return cls(
            int(getattr(policy, "_cooldown_until")),
            getattr(policy, "_last_tick"),
            getattr(policy, "_last_decision"),
            copy.deepcopy(getattr(policy, "_progress")),
        )

    def restore(self, policy: object) -> None:
        policy._cooldown_until = self.cooldown_until
        policy._last_tick = self.last_tick
        policy._last_decision = self.last_decision
        policy._progress = copy.deepcopy(self.progress)

    def manifest(self) -> dict[str, object]:
        attempt = getattr(self.progress, "_attempt", None)
        stalled = getattr(self.progress, "_stalled", {})
        return {
            "cooldown_until": self.cooldown_until,
            "last_tick": self.last_tick,
            "last_decision": (
                None
                if self.last_decision is None
                else {
                    "kind": int(self.last_decision.action.kind),
                    "source": self.last_decision.source_body_id,
                    "destination": self.last_decision.destination_body_id,
                }
            ),
            "pending_pair": (
                None
                if attempt is None
                else {
                    "source": int(attempt.pair.source_id),
                    "destination": int(attempt.pair.destination_id),
                    "gap": float(attempt.gap),
                    "minimum_closure": float(attempt.minimum_closure),
                }
            ),
            "stalled": [
                {
                    "source": int(pair.source_id),
                    "destination": int(pair.destination_id),
                    "best_gap": float(value.best_gap),
                    "minimum_closure": float(value.minimum_closure),
                }
                for pair, value in sorted(stalled.items())
            ],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class MarginPoint:
    tick: int
    gauge: int
    level: int
    outstanding_liabilities: int
    reserve: int
    margin: int
    renewal_epoch: int
    score: int
    qualifying_clears: int
    rotten_events: int

    def manifest(self) -> dict[str, int]:
        return {
            name: int(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class CandidateSolvencyOutcome:
    candidate: JointCandidate
    start_tick: int
    end_tick: int
    initial_gauge: int
    gauge_max: int
    initial_level: int
    initial_liabilities: int
    resolved_two_renewals: bool
    renewal_ticks: tuple[int, ...]
    gross_renewal: int
    minimum_margin: int
    final_margin: int
    final_gauge: int
    final_level: int
    score_gain: int
    qualifying_clear_gain: int
    cleared_events: int
    rotten_events: int
    invalid_actions: int
    game_over: bool
    terminated: bool
    truncated: bool
    continuation_rebound: bool
    ledger: Mapping[str, object]
    margin_curve: tuple[MarginPoint, ...]
    error: str | None = None

    @property
    def simulated_ticks(self) -> int:
        return self.end_tick - self.start_tick

    @property
    def hard_valid(self) -> bool:
        return (
            self.error is None
            and self.continuation_rebound
            and self.resolved_two_renewals
            and not self.game_over
            and not self.terminated
            and not self.truncated
            and self.invalid_actions == 0
            and self.final_gauge > 0
        )

    def manifest(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.manifest(),
            "start_tick": self.start_tick,
            "end_tick": self.end_tick,
            "simulated_ticks": self.simulated_ticks,
            "initial_gauge": self.initial_gauge,
            "gauge_max": self.gauge_max,
            "initial_level": self.initial_level,
            "initial_liabilities": self.initial_liabilities,
            "resolved_two_renewals": self.resolved_two_renewals,
            "renewal_ticks": list(self.renewal_ticks),
            "gross_renewal": self.gross_renewal,
            "minimum_margin": self.minimum_margin,
            "final_margin": self.final_margin,
            "final_gauge": self.final_gauge,
            "final_level": self.final_level,
            "score_gain": self.score_gain,
            "qualifying_clear_gain": self.qualifying_clear_gain,
            "cleared_events": self.cleared_events,
            "rotten_events": self.rotten_events,
            "invalid_actions": self.invalid_actions,
            "game_over": self.game_over,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "continuation_rebound": self.continuation_rebound,
            "hard_valid": self.hard_valid,
            "ledger": self.ledger,
            "margin_curve": [value.manifest() for value in self.margin_curve],
            "error": self.error,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class CandidateCertificate:
    ordinal: int
    exact_b2: int
    exact_delta_b2: int
    predicted_delta_b2: float
    conformal_q: float
    lower_bound_delta_b2: float
    eligible: bool
    reasons: tuple[str, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "ordinal": self.ordinal,
            "exact_b2": self.exact_b2,
            "exact_delta_b2": self.exact_delta_b2,
            "predicted_delta_b2": self.predicted_delta_b2,
            "conformal_q": self.conformal_q,
            "lower_bound_delta_b2": self.lower_bound_delta_b2,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
        }


def certify_candidate(
    outcome: CandidateSolvencyOutcome,
    incumbent: CandidateSolvencyOutcome,
    *,
    predicted_delta_b2: float | None = None,
    conformal_q: float = 0.0,
) -> CandidateCertificate:
    """Classify one candidate using only itself and candidate zero."""

    exact_delta = outcome.minimum_margin - incumbent.minimum_margin
    predicted = float(exact_delta if predicted_delta_b2 is None else predicted_delta_b2)
    conformal = float(conformal_q)
    if (
        not math.isfinite(predicted)
        or not math.isfinite(conformal)
        or conformal < 0.0
    ):
        raise ValueError("candidate uncertainty inputs are invalid")
    lower = predicted - conformal
    reasons: list[str] = []
    if outcome.candidate.ordinal == incumbent.candidate.ordinal:
        reasons.append("frozen_v5_tie")
    if not incumbent.hard_valid:
        reasons.append("incumbent_unresolved")
    if not outcome.hard_valid:
        reasons.append("candidate_hard_invalid")
    if outcome.minimum_margin < 0:
        reasons.append("negative_b2")
    if lower < 0.0:
        reasons.append("negative_delta_b2_lcb")
    if exact_delta == 0:
        reasons.append("delta_b2_tie")
    return CandidateCertificate(
        outcome.candidate.ordinal,
        outcome.minimum_margin,
        exact_delta,
        predicted,
        conformal,
        lower,
        not reasons,
        tuple(reasons),
    )


@dataclass(frozen=True, slots=True)
class SolvencySearchResult:
    config_sha256: str
    candidate_generator_sha256: str
    snapshot_sha256: str
    policy_state_sha256: str
    outcomes: tuple[CandidateSolvencyOutcome, ...]
    certificates: tuple[CandidateCertificate, ...]
    selected_ordinal: int
    restore_checks: int
    wall_seconds: float
    cpu_seconds: float

    def __post_init__(self) -> None:
        ordinals = [value.candidate.ordinal for value in self.outcomes]
        certificate_ordinals = [
            value.ordinal for value in self.certificates
        ]
        if (
            len(self.outcomes) != len(self.certificates)
            or sorted(ordinals) != list(range(len(ordinals)))
            or sorted(certificate_ordinals)
            != list(range(len(certificate_ordinals)))
            or self.selected_ordinal not in ordinals
        ):
            raise ValueError("solvency search ordinal accounting is malformed")

    def outcome_for(self, ordinal: int) -> CandidateSolvencyOutcome:
        return next(
            value
            for value in self.outcomes
            if value.candidate.ordinal == ordinal
        )

    def certificate_for(self, ordinal: int) -> CandidateCertificate:
        return next(
            value for value in self.certificates if value.ordinal == ordinal
        )

    def ordered_pairs(
        self,
    ) -> tuple[
        tuple[CandidateSolvencyOutcome, CandidateCertificate], ...
    ]:
        return tuple(
            (self.outcome_for(ordinal), self.certificate_for(ordinal))
            for ordinal in range(len(self.outcomes))
        )

    @property
    def incumbent(self) -> CandidateSolvencyOutcome:
        return self.outcome_for(0)

    @property
    def selected(self) -> CandidateSolvencyOutcome:
        return self.outcome_for(self.selected_ordinal)

    @property
    def decision(self) -> SteeringDecision:
        return self.selected.candidate.decision

    @property
    def override(self) -> bool:
        return self.selected_ordinal != 0

    def manifest(self) -> dict[str, object]:
        return {
            "version": SOLVENCY_SHIELD_VERSION,
            "config_sha256": self.config_sha256,
            "candidate_generator_sha256": self.candidate_generator_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "policy_state_sha256": self.policy_state_sha256,
            "selected_ordinal": self.selected_ordinal,
            "override": self.override,
            "restore_checks": self.restore_checks,
            "wall_seconds": self.wall_seconds,
            "cpu_seconds": self.cpu_seconds,
            "outcomes": [
                value.manifest()
                for value in sorted(
                    self.outcomes, key=lambda item: item.candidate.ordinal
                )
            ],
            "certificates": [
                value.manifest()
                for value in sorted(
                    self.certificates, key=lambda item: item.ordinal
                )
            ],
        }

    def identity_manifest(self) -> dict[str, object]:
        value = self.manifest()
        value.pop("wall_seconds")
        value.pop("cpu_seconds")
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())


SCORE_RESIDUAL_FEATURES: tuple[str, ...] = (
    "initial_gauge_fraction",
    "initial_level_99",
    "initial_liabilities_8",
    "pair_distance_10",
    "pair_rotten_hazard",
    "pair_viable_anchor",
    "pair_fresh_match",
    "geometry_analytic_strong",
    "geometry_close_strong",
    "geometry_wide_strong",
    "geometry_deep_strong",
    "geometry_analytic_weak",
    "minimum_margin_fraction",
    "exact_delta_b2_fraction",
    "final_margin_fraction",
    "renewal_duration_1024",
    "gross_renewal_fraction",
    "qualifying_clear_gain_10",
    "rotten_events_10",
)


def score_residual_features(
    outcome: CandidateSolvencyOutcome,
    certificate: CandidateCertificate,
) -> np.ndarray:
    """Public/action/analytic-safety features; realized score is excluded."""

    scale = max(outcome.gauge_max, 1)
    category = outcome.candidate.pair.category
    geometry = outcome.candidate.geometry.name
    values = (
        outcome.initial_gauge / scale,
        min(outcome.initial_level, 99) / 99.0,
        outcome.initial_liabilities / 8.0,
        outcome.candidate.pair.distance_sizes / 10.0,
        float(category == "rotten-hazard"),
        float(category == "viable-anchor"),
        float(category == "fresh-match"),
        float(geometry == "analytic-strong"),
        float(geometry == "close-strong"),
        float(geometry == "wide-strong"),
        float(geometry == "deep-strong"),
        float(geometry == "analytic-weak"),
        outcome.minimum_margin / scale,
        certificate.exact_delta_b2 / scale,
        outcome.final_margin / scale,
        outcome.simulated_ticks / 1_024.0,
        outcome.gross_renewal / scale,
        outcome.qualifying_clear_gain / 10.0,
        outcome.rotten_events / 10.0,
    )
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (len(SCORE_RESIDUAL_FEATURES),) or not np.isfinite(
        result
    ).all():
        raise ValueError("score-residual features are malformed")
    return result


@dataclass(frozen=True, slots=True)
class ScoreResidualModel:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    weights: tuple[float, ...]
    ridge: float
    training_rows: int
    training_manifest_sha256: str

    def __post_init__(self) -> None:
        width = len(SCORE_RESIDUAL_FEATURES)
        if (
            self.feature_names != SCORE_RESIDUAL_FEATURES
            or len(self.means) != width
            or len(self.scales) != width
            or len(self.weights) != width + 1
            or self.training_rows < 1
            or len(self.training_manifest_sha256) != 64
            or any(value <= 0.0 for value in self.scales)
            or not all(
                math.isfinite(value)
                for value in (*self.means, *self.scales, *self.weights)
            )
        ):
            raise ValueError("score-residual model is malformed")

    @classmethod
    def fit(
        cls,
        searches: Sequence[SolvencySearchResult],
        *,
        ridge: float,
        training_manifest_sha256: str,
    ) -> ScoreResidualModel:
        if not math.isfinite(ridge) or ridge <= 0.0:
            raise ValueError("ridge must be finite and positive")
        rows: list[np.ndarray] = []
        labels: list[float] = []
        for search in searches:
            incumbent_score = search.incumbent.score_gain
            for outcome, certificate in search.ordered_pairs():
                if outcome.candidate.ordinal == 0 or not certificate.eligible:
                    continue
                rows.append(score_residual_features(outcome, certificate))
                labels.append(float(outcome.score_gain - incumbent_score))
        return cls.fit_rows(
            rows,
            labels,
            ridge=ridge,
            training_manifest_sha256=training_manifest_sha256,
        )

    @classmethod
    def fit_rows(
        cls,
        rows: Sequence[Sequence[float] | np.ndarray],
        labels: Sequence[float],
        *,
        ridge: float,
        training_manifest_sha256: str,
    ) -> ScoreResidualModel:
        if len(rows) < 2 or len(rows) != len(labels):
            raise ValueError("score-residual fitting needs two aligned rows")
        matrix = np.asarray(rows, dtype=np.float64)
        target = np.asarray(labels, dtype=np.float64)
        if (
            matrix.shape != (len(rows), len(SCORE_RESIDUAL_FEATURES))
            or target.shape != (len(rows),)
            or not np.isfinite(matrix).all()
            or not np.isfinite(target).all()
        ):
            raise ValueError("score-residual training rows are malformed")
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales[scales < 1e-12] = 1.0
        normalized = (matrix - means) / scales
        design = np.concatenate(
            (np.ones((normalized.shape[0], 1)), normalized), axis=1
        )
        penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
        penalty[0, 0] = 0.0
        weights = np.linalg.solve(
            design.T @ design + penalty, design.T @ target
        )
        return cls(
            SCORE_RESIDUAL_FEATURES,
            tuple(float(value) for value in means),
            tuple(float(value) for value in scales),
            tuple(float(value) for value in weights),
            float(ridge),
            len(rows),
            training_manifest_sha256,
        )

    def predict(
        self,
        outcome: CandidateSolvencyOutcome,
        certificate: CandidateCertificate,
    ) -> float:
        values = score_residual_features(outcome, certificate)
        normalized = (
            values - np.asarray(self.means)
        ) / np.asarray(self.scales)
        weights = np.asarray(self.weights)
        return float(weights[0] + normalized @ weights[1:])

    def manifest(self) -> dict[str, object]:
        return {
            "version": "r3g-certified-safe-score-ridge-v1",
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "weights": list(self.weights),
            "ridge": self.ridge,
            "training_rows": self.training_rows,
            "training_manifest_sha256": self.training_manifest_sha256,
            "target": (
                "candidate score gain minus frozen-v5 score gain; fit only "
                "on exact certified-safe alternatives"
            ),
            "forbidden_features": ["score_gain", "realized_score_advantage"],
        }

    @classmethod
    def from_manifest(cls, value: Mapping[str, Any]) -> ScoreResidualModel:
        if value.get("version") != "r3g-certified-safe-score-ridge-v1":
            raise ValueError("score-residual version mismatch")
        return cls(
            tuple(value["feature_names"]),
            tuple(float(item) for item in value["means"]),
            tuple(float(item) for item in value["scales"]),
            tuple(float(item) for item in value["weights"]),
            float(value["ridge"]),
            int(value["training_rows"]),
            str(value["training_manifest_sha256"]),
        )

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class ResidualSelection:
    search: SolvencySearchResult
    model_sha256: str
    predicted_score_advantages: tuple[float | None, ...]
    proposed_ordinal: int
    selected_ordinal: int

    @property
    def decision(self) -> SteeringDecision:
        return self.search.outcome_for(
            self.selected_ordinal
        ).candidate.decision

    @property
    def override(self) -> bool:
        return self.selected_ordinal != 0

    def manifest(self) -> dict[str, object]:
        return {
            "search": self.search.manifest(),
            "model_sha256": self.model_sha256,
            "predicted_score_advantages": list(
                self.predicted_score_advantages
            ),
            "proposed_ordinal": self.proposed_ordinal,
            "selected_ordinal": self.selected_ordinal,
            "override": self.override,
        }

    @property
    def sha256(self) -> str:
        value = self.manifest()
        value["search"] = self.search.identity_manifest()
        return _canonical_sha256(value)


def select_with_score_residual(
    search: SolvencySearchResult,
    model: ScoreResidualModel,
) -> ResidualSelection:
    predictions: list[float | None] = []
    proposed: list[tuple[float, CandidateSolvencyOutcome]] = []
    eligible: list[tuple[float, CandidateSolvencyOutcome]] = []
    ordered = sorted(
        search.ordered_pairs(),
        key=lambda value: value[0].candidate.ordinal,
    )
    for outcome, certificate in ordered:
        if outcome.candidate.ordinal == 0:
            predictions.append(None)
            continue
        prediction = model.predict(outcome, certificate)
        if not math.isfinite(prediction):
            raise ValueError("score-residual prediction is nonfinite")
        predictions.append(prediction)
        if prediction > 0.0:
            proposed.append((prediction, outcome))
        if certificate.eligible and prediction > 0.0:
            eligible.append((prediction, outcome))
    if not proposed:
        proposed_ordinal = 0
    else:
        best_proposal = max(value[0] for value in proposed)
        proposal_winners = [
            outcome.candidate.ordinal
            for prediction, outcome in proposed
            if prediction == best_proposal
        ]
        proposed_ordinal = (
            proposal_winners[0] if len(proposal_winners) == 1 else 0
        )
    if not eligible:
        selected = 0
    else:
        best_prediction = max(value[0] for value in eligible)
        winners = [
            outcome
            for prediction, outcome in eligible
            if prediction == best_prediction
        ]
        selected = (
            winners[0].candidate.ordinal if len(winners) == 1 else 0
        )
    return ResidualSelection(
        search,
        model.sha256,
        tuple(predictions),
        proposed_ordinal,
        selected,
    )


class CandidateLocalSolvencySearch:
    """Exact branch replay with candidate-private policy and liability state."""

    def __init__(
        self,
        continuation_factory: Callable[[], object],
        *,
        joint_config: JointPlannerConfig | None = None,
        barrier_config: SolvencyBarrierConfig | None = None,
        action_spec: ActionSpec | None = None,
        continuation_identity_sha256: str | None = None,
        conformal_q: float = 0.0,
    ) -> None:
        self.continuation_factory = continuation_factory
        self.joint_config = (
            JointPlannerConfig(geometry_cap=5)
            if joint_config is None
            else joint_config
        )
        if self.joint_config.geometry_cap != 5:
            raise ValueError("R3G reconstruction requires all five geometries")
        self.barrier_config = (
            SolvencyBarrierConfig()
            if barrier_config is None
            else barrier_config
        )
        self.action_spec = ActionSpec() if action_spec is None else action_spec
        if not math.isfinite(conformal_q) or conformal_q < 0.0:
            raise ValueError("conformal q must be finite and nonnegative")
        self.conformal_q = float(conformal_q)
        self.generator = JointPairGeometrySearch(
            continuation_factory,
            config=self.joint_config,
            action_spec=self.action_spec,
            continuation_identity_sha256=continuation_identity_sha256,
        )

    def identity_manifest(self) -> dict[str, object]:
        return {
            "version": SOLVENCY_SHIELD_VERSION,
            "joint_version": JOINT_PLANNER_VERSION,
            "joint_config": self.joint_config.manifest(),
            "joint_config_sha256": self.joint_config.sha256,
            "barrier_config": self.barrier_config.manifest(),
            "barrier_config_sha256": self.barrier_config.sha256,
            "candidate_generator": self.generator.identity_manifest(),
            "safety_prediction": "exact analytic delta_B2",
            "conformal_q": self.conformal_q,
            "selection": (
                "exact hard certificate first; among strictly positive-score "
                "certified alternatives; candidate zero on every tie"
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())

    @staticmethod
    def _check_gauge_recurrence(
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        events: Sequence[Mapping[str, Any]],
    ) -> None:
        gauge = _plain_int(before.get("gauge"), "pre gauge")
        for event in events:
            if _event_kind(event) == int(EventKind.GAUGE_CHANGED):
                gauge += _plain_int(event.get("value"), "gauge delta")
        observed = _plain_int(after.get("gauge"), "post gauge")
        if gauge != observed:
            raise RuntimeError("ordered gauge events do not reproduce public gauge")

        level = _plain_int(after.get("level"), "post level")
        gauge_max = _plain_int(after.get("gauge_max"), "gauge max")
        pre_scene = _plain_int(before.get("gauge"), "pre gauge")
        scene_event: Mapping[str, Any] | None = None
        after_scene = False
        for event in events:
            if _event_kind(event) != int(EventKind.GAUGE_CHANGED):
                continue
            detail = str(event.get("detail", ""))
            if detail == "scene clamp and passive drain":
                if scene_event is not None or after_scene:
                    raise RuntimeError("scene gauge event occurred more than once")
                scene_event = event
                after_scene = True
                continue
            if detail in {_NORMAL_RENEWAL, _SPECIAL_RENEWAL}:
                if after_scene:
                    raise RuntimeError("renewable recovery occurred after drain")
                if _plain_int(event.get("value"), "renewal gauge delta") <= 0:
                    raise RuntimeError("renewable recovery was not positive")
                pre_scene += _plain_int(
                    event.get("value"), "pre-scene gauge delta"
                )
                continue
            if detail == _ROT_PENALTY:
                # At gauge==terminal_floor the scene floor can make its net
                # delta zero, so the native runtime omits the scene event.
                after_scene = True
                if _plain_int(event.get("value"), "rot gauge delta") >= 0:
                    raise RuntimeError("rot penalty was not negative")
                continue
            raise RuntimeError("unknown ordered gauge event")
        clamped = min(max(pre_scene, 0), gauge_max)
        unit = passive_drain_unit(level)
        drained = clamped - (3 * unit if clamped > gauge_max // 2 else unit)
        expected_scene = max(1, drained)
        actual_scene = pre_scene + (
            0
            if scene_event is None
            else _plain_int(scene_event.get("value"), "scene gauge delta")
        )
        if actual_scene != expected_scene:
            raise RuntimeError("passive-drain event violates exact gauge recurrence")

    def _evaluate(
        self,
        env: Any,
        initial: Mapping[str, Any],
        candidate: JointCandidate,
        incumbent: SteeringDecision,
        policy_state: FrozenPolicyState,
        baseline_action_tie: bool,
    ) -> CandidateSolvencyOutcome:
        start_tick = _plain_int(initial.get("tick"), "initial tick")
        start_score = _plain_int(initial.get("score"), "initial score")
        start_clears = _plain_int(
            initial.get("qualifying_clear_count"), "initial clears"
        )
        current = initial
        ledger = RotLiabilityLedger()
        ledger.observe_initial(initial)
        initial_level = _plain_int(initial.get("level"), "initial level")
        initial_gauge = _plain_int(initial.get("gauge"), "initial gauge")
        initial_margin = (
            initial_gauge
            - self.barrier_config.terminal_floor
            - ledger.reserve(initial_level)
        )
        minimum_margin = initial_margin
        previous_liabilities = len(ledger.outstanding)
        curve = [
            MarginPoint(
                start_tick,
                initial_gauge,
                initial_level,
                len(ledger.outstanding),
                ledger.reserve(initial_level),
                initial_margin,
                0,
                start_score,
                start_clears,
                0,
            )
        ]
        renewal_ticks: list[int] = []
        gross_renewal = 0
        counts: Counter[int] = Counter()
        terminated = bool(initial.get("terminated", False))
        truncated = bool(initial.get("truncated", False))
        game_over = terminated and initial_gauge <= 0
        continuation_rebound = False
        error: str | None = None

        policy = self.continuation_factory()
        expected_identity = self.generator.continuation_identity_sha256
        if (
            expected_identity is not None
            and getattr(policy, "artifact_sha256", None)
            != expected_identity
        ):
            raise RuntimeError("frozen-v5 continuation identity changed")
        policy_state.restore(policy)
        if candidate.ordinal == 0 or baseline_action_tie:
            continuation_rebound = True
        else:
            continuation_rebound = _commit_base_decision(
                policy, initial, incumbent, candidate.decision
            )

        def unit_step(action: Action) -> None:
            nonlocal current, terminated, truncated, game_over, gross_renewal
            nonlocal minimum_margin, previous_liabilities
            if _plain_int(current.get("gauge"), "entry gauge") <= 0:
                game_over = True
            before = current
            current, _reward, terminated, truncated, info = env.step(action)
            if not isinstance(current, Mapping) or not isinstance(info, Mapping):
                raise TypeError("portable branch transition is malformed")
            events = tuple(
                event
                for event in info.get("events", ())
                if isinstance(event, Mapping)
            )
            self._check_gauge_recurrence(before, current, events)
            ledger.reconcile_tick(before, current, events)
            for event in events:
                kind = _event_kind(event)
                if kind is not None:
                    counts[kind] += 1
                if kind == int(EventKind.GAME_OVER):
                    game_over = True
            renewed, gain = renewal_epoch(events)
            tick = _plain_int(current.get("tick"), "current tick")
            if renewed and (not renewal_ticks or renewal_ticks[-1] != tick):
                renewal_ticks.append(tick)
                gross_renewal += gain
            level = _plain_int(current.get("level"), "current level")
            gauge = _plain_int(current.get("gauge"), "current gauge")
            reserve = ledger.reserve(level)
            margin = gauge - self.barrier_config.terminal_floor - reserve
            new_minimum = margin < minimum_margin
            minimum_margin = min(minimum_margin, margin)
            liabilities = len(ledger.outstanding)
            point = MarginPoint(
                tick,
                gauge,
                level,
                liabilities,
                reserve,
                margin,
                len(renewal_ticks),
                _plain_int(current.get("score"), "current score"),
                _plain_int(
                    current.get("qualifying_clear_count"),
                    "current qualifying clears",
                ),
                counts[int(EventKind.ROTTEN)],
            )
            if (
                renewed
                or new_minimum
                or liabilities != previous_liabilities
                or any(
                    _event_kind(event) == int(EventKind.ROTTEN)
                    for event in events
                )
            ):
                curve.append(point)
            previous_liabilities = liabilities

        def execute(action: Action) -> None:
            if (
                terminated
                or truncated
                or len(renewal_ticks)
                >= self.barrier_config.required_renewal_epochs
                or _plain_int(current.get("tick"), "current tick") - start_tick
                >= self.barrier_config.branch_tick_cap
            ):
                return
            if ActionKind.parse(action.kind) is ActionKind.WAIT:
                for _ in range(int(action.wait_ticks)):
                    if (
                        terminated
                        or truncated
                        or len(renewal_ticks)
                        >= self.barrier_config.required_renewal_epochs
                        or _plain_int(current.get("tick"), "current tick")
                        - start_tick
                        >= self.barrier_config.branch_tick_cap
                    ):
                        break
                    unit_step(Action.wait(1))
            else:
                unit_step(action)

        try:
            for action in candidate.decision.primitive_actions(self.action_spec):
                if (
                    terminated
                    or truncated
                    or len(renewal_ticks)
                    >= self.barrier_config.required_renewal_epochs
                    or _plain_int(current.get("tick"), "current tick")
                    - start_tick
                    >= self.barrier_config.branch_tick_cap
                ):
                    break
                execute(action)
            while (
                not terminated
                and not truncated
                and len(renewal_ticks)
                < self.barrier_config.required_renewal_epochs
                and _plain_int(current.get("tick"), "current tick") - start_tick
                < self.barrier_config.branch_tick_cap
            ):
                decision = getattr(policy, "predict")(current)
                if not isinstance(decision, SteeringDecision):
                    raise TypeError("frozen-v5 continuation returned a non-decision")
                for action in decision.primitive_actions(self.action_spec):
                    if (
                        terminated
                        or truncated
                        or len(renewal_ticks)
                        >= self.barrier_config.required_renewal_epochs
                        or _plain_int(current.get("tick"), "current tick")
                        - start_tick
                        >= self.barrier_config.branch_tick_cap
                    ):
                        break
                    execute(action)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        final_tick = _plain_int(current.get("tick"), "final tick")
        final_gauge = _plain_int(current.get("gauge"), "final gauge")
        final_level = _plain_int(current.get("level"), "final level")
        final_reserve = ledger.reserve(final_level)
        final_margin = (
            final_gauge
            - self.barrier_config.terminal_floor
            - final_reserve
        )
        if curve[-1].tick != final_tick:
            curve.append(
                MarginPoint(
                    final_tick,
                    final_gauge,
                    final_level,
                    len(ledger.outstanding),
                    final_reserve,
                    final_margin,
                    len(renewal_ticks),
                    _plain_int(current.get("score"), "final score"),
                    _plain_int(
                        current.get("qualifying_clear_count"), "final clears"
                    ),
                    counts[int(EventKind.ROTTEN)],
                )
            )
        return CandidateSolvencyOutcome(
            candidate,
            start_tick,
            final_tick,
            initial_gauge,
            _plain_int(initial.get("gauge_max"), "initial gauge max"),
            initial_level,
            curve[0].outstanding_liabilities,
            len(renewal_ticks) >= self.barrier_config.required_renewal_epochs,
            tuple(renewal_ticks),
            gross_renewal,
            minimum_margin,
            final_margin,
            final_gauge,
            final_level,
            _plain_int(current.get("score"), "final score") - start_score,
            _plain_int(
                current.get("qualifying_clear_count"), "final clears"
            )
            - start_clears,
            counts[int(EventKind.CLEARED)],
            counts[int(EventKind.ROTTEN)],
            counts[int(EventKind.INVALID_ACTION)],
            game_over,
            terminated,
            truncated,
            continuation_rebound,
            ledger.manifest(),
            tuple(curve),
            error,
        )

    def search(
        self,
        env: Any,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
        policy_state: FrozenPolicyState,
        *,
        teacher_score_selection: bool = True,
    ) -> SolvencySearchResult:
        if getattr(env, "physics_backend", None) != "portable":
            raise ValueError("R3G shield requires the portable backend")
        if not incumbent.is_shot:
            raise ValueError("R3G shield requires an incumbent shot")
        candidates = self.generator._candidates(observation, incumbent)
        incumbent_action_key = _primitive_action_key(
            incumbent, self.action_spec
        )
        baseline_action_ties = {
            candidate.ordinal
            for candidate in candidates
            if candidate.ordinal != 0
            and _primitive_action_key(candidate.decision, self.action_spec)
            == incumbent_action_key
        }
        snapshot = env.clone_state()
        snapshot_sha256 = hashlib.sha256(snapshot).hexdigest()
        expected_signature = _public_signature(observation)
        state_hash = getattr(env, "state_hash", None)
        expected_state_hash = state_hash() if callable(state_hash) else None
        policy_state_sha256 = policy_state.sha256
        wall_started, cpu_started = time.perf_counter(), time.process_time()
        outcomes: list[CandidateSolvencyOutcome] = []
        restore_checks = 0
        try:
            for candidate in candidates:
                restored = env.restore_state(snapshot)
                restore_checks += 1
                if (
                    _public_signature(restored) != expected_signature
                    or env.clone_state() != snapshot
                    or (
                        expected_state_hash is not None
                        and state_hash() != expected_state_hash
                    )
                    or policy_state.sha256 != policy_state_sha256
                ):
                    raise RuntimeError("R3G candidate did not receive exact state")
                outcomes.append(
                    self._evaluate(
                        env,
                        restored,
                        candidate,
                        incumbent,
                        policy_state,
                        candidate.ordinal in baseline_action_ties,
                    )
                )
        finally:
            restored = env.restore_state(snapshot)
            restore_checks += 1
            if (
                _public_signature(restored) != expected_signature
                or env.clone_state() != snapshot
                or (
                    expected_state_hash is not None
                    and state_hash() != expected_state_hash
                )
                or policy_state.sha256 != policy_state_sha256
            ):
                raise RuntimeError("R3G search failed transactional restore")

        incumbent_outcome = next(
            value for value in outcomes if value.candidate.ordinal == 0
        )
        invariant_fields = (
            "start_tick",
            "end_tick",
            "initial_gauge",
            "gauge_max",
            "initial_level",
            "initial_liabilities",
            "resolved_two_renewals",
            "renewal_ticks",
            "gross_renewal",
            "minimum_margin",
            "final_margin",
            "final_gauge",
            "final_level",
            "score_gain",
            "qualifying_clear_gain",
            "cleared_events",
            "rotten_events",
            "invalid_actions",
            "game_over",
            "terminated",
            "truncated",
            "continuation_rebound",
            "ledger",
            "margin_curve",
            "error",
        )
        for ordinal in baseline_action_ties:
            tied = next(
                value
                for value in outcomes
                if value.candidate.ordinal == ordinal
            )
            if any(
                getattr(tied, name) != getattr(incumbent_outcome, name)
                for name in invariant_fields
            ):
                raise RuntimeError(
                    "action-identical candidate changed frozen-v5 dynamics"
                )
        certificates = tuple(
            certify_candidate(
                value,
                incumbent_outcome,
                predicted_delta_b2=(
                    value.minimum_margin
                    - incumbent_outcome.minimum_margin
                ),
                conformal_q=self.conformal_q,
            )
            for value in outcomes
        )
        eligible = (
            [
                value
                for value, certificate in zip(
                    outcomes, certificates, strict=True
                )
                if certificate.eligible
                and value.score_gain > incumbent_outcome.score_gain
            ]
            if teacher_score_selection
            else []
        )
        best_score = (
            max(value.score_gain for value in eligible)
            if eligible
            else incumbent_outcome.score_gain
        )
        winners = [
            value for value in eligible if value.score_gain == best_score
        ]
        winner = winners[0] if len(winners) == 1 else incumbent_outcome
        return SolvencySearchResult(
            self.sha256,
            self.generator.sha256,
            snapshot_sha256,
            policy_state_sha256,
            tuple(outcomes),
            certificates,
            winner.candidate.ordinal,
            restore_checks,
            time.perf_counter() - wall_started,
            time.process_time() - cpu_started,
        )


class AnalyticSolvencyTeacherPolicy:
    """Frozen-v5 cadence with one exact candidate-local query per tick epoch."""

    def __init__(
        self,
        env: Any,
        base_policy: object,
        searcher: CandidateLocalSolvencySearch,
    ) -> None:
        self.env = env
        self.base_policy = base_policy
        self.searcher = searcher
        self._next_query_tick = 0
        self._results: list[SolvencySearchResult] = []
        self._executed_ordinals: list[int] = []
        self._counts: Counter[str] = Counter()

    def reset(self, seed: int = 0) -> None:
        getattr(self.base_policy, "reset")(seed)
        self._next_query_tick = 0
        self._results.clear()
        self._executed_ordinals.clear()
        self._counts.clear()

    @property
    def results(self) -> tuple[SolvencySearchResult, ...]:
        return tuple(self._results)

    @property
    def executed_ordinals(self) -> tuple[int, ...]:
        return tuple(self._executed_ordinals)

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = getattr(self.base_policy, "predict")(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("R3G base policy returned a non-decision")
        if not incumbent.is_shot:
            return incumbent
        self._counts["seen_shots"] += 1
        tick = _plain_int(observation.get("tick"), "decision tick")
        if tick < self._next_query_tick:
            self._counts["interval_abstentions"] += 1
            return incumbent
        self._next_query_tick = tick + self.searcher.barrier_config.query_interval_ticks
        self._counts["queries"] += 1
        state = FrozenPolicyState.capture(self.base_policy)
        try:
            result = self.searcher.search(
                self.env, observation, incumbent, state
            )
        except ValueError:
            self._counts["unsupported_abstentions"] += 1
            return incumbent
        self._results.append(result)
        self._executed_ordinals.append(0)
        self._counts["candidates"] += len(result.outcomes)
        self._counts["certified_candidates"] += sum(
            value.eligible for value in result.certificates
        )
        self._counts["states_with_certified_override"] += int(
            any(value.eligible for value in result.certificates)
        )
        self._counts["simulated_ticks"] += sum(
            value.simulated_ticks for value in result.outcomes
        )
        self._counts["restore_checks"] += result.restore_checks
        if not result.override:
            self._counts["barrier_abstentions"] += 1
            return incumbent
        if not _commit_base_decision(
            self.base_policy, observation, incumbent, result.decision
        ):
            self._counts["progress_rebind_abstentions"] += 1
            return incumbent
        self._executed_ordinals[-1] = result.selected_ordinal
        self._counts["overrides"] += 1
        self._counts[
            f"override_pair/{result.selected.candidate.pair.category}"
        ] += 1
        self._counts[
            f"override_geometry/{result.selected.candidate.geometry.name}"
        ] += 1
        return result.decision

    def statistics(self) -> dict[str, object]:
        return {
            **dict(sorted(self._counts.items())),
            "search_wall_seconds": sum(value.wall_seconds for value in self._results),
            "search_cpu_seconds": sum(value.cpu_seconds for value in self._results),
        }


class LearnedScoreResidualPolicy:
    """Exact analytic shield followed by a learned certified-safe score ranker."""

    def __init__(
        self,
        env: Any,
        base_policy: object,
        searcher: CandidateLocalSolvencySearch,
        model: ScoreResidualModel,
        *,
        query_predicate: Callable[[Mapping[str, Any]], bool] | None = None,
        maximum_queries: int | None = None,
    ) -> None:
        self.env = env
        self.base_policy = base_policy
        self.searcher = searcher
        self.model = model
        self.query_predicate = query_predicate
        if maximum_queries is not None and maximum_queries < 1:
            raise ValueError("maximum queries must be positive")
        self.maximum_queries = maximum_queries
        self._next_query_tick = 0
        self._results: list[ResidualSelection] = []
        self._executed_ordinals: list[int] = []
        self._counts: Counter[str] = Counter()

    def reset(self, seed: int = 0) -> None:
        getattr(self.base_policy, "reset")(seed)
        self._next_query_tick = 0
        self._results.clear()
        self._executed_ordinals.clear()
        self._counts.clear()

    @property
    def results(self) -> tuple[ResidualSelection, ...]:
        return tuple(self._results)

    @property
    def executed_ordinals(self) -> tuple[int, ...]:
        return tuple(self._executed_ordinals)

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = getattr(self.base_policy, "predict")(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("R3G base policy returned a non-decision")
        if not incumbent.is_shot:
            return incumbent
        self._counts["seen_shots"] += 1
        if (
            self.query_predicate is not None
            and not self.query_predicate(observation)
        ):
            self._counts["target_abstentions"] += 1
            return incumbent
        if (
            self.maximum_queries is not None
            and self._counts["queries"] >= self.maximum_queries
        ):
            self._counts["query_budget_abstentions"] += 1
            return incumbent
        tick = _plain_int(observation.get("tick"), "decision tick")
        if tick < self._next_query_tick:
            self._counts["interval_abstentions"] += 1
            return incumbent
        self._next_query_tick = tick + self.searcher.barrier_config.query_interval_ticks
        self._counts["queries"] += 1
        state = FrozenPolicyState.capture(self.base_policy)
        try:
            search = self.searcher.search(
                self.env,
                observation,
                incumbent,
                state,
                teacher_score_selection=False,
            )
        except ValueError:
            self._counts["unsupported_abstentions"] += 1
            return incumbent
        result = select_with_score_residual(search, self.model)
        self._results.append(result)
        self._executed_ordinals.append(0)
        self._counts["candidates"] += len(search.outcomes)
        self._counts["certified_candidates"] += sum(
            value.eligible for value in search.certificates
        )
        self._counts["states_with_certified_override"] += int(
            any(value.eligible for value in search.certificates)
        )
        self._counts["score_proposals"] += int(result.proposed_ordinal != 0)
        if result.proposed_ordinal:
            proposed_certificate = search.certificate_for(
                result.proposed_ordinal
            )
            self._counts["uncertified_score_proposals"] += int(
                not proposed_certificate.eligible
            )
        self._counts["simulated_ticks"] += sum(
            value.simulated_ticks for value in search.outcomes
        )
        self._counts["restore_checks"] += search.restore_checks
        if not result.override:
            self._counts["barrier_or_score_abstentions"] += 1
            return incumbent
        if not _commit_base_decision(
            self.base_policy, observation, incumbent, result.decision
        ):
            self._counts["progress_rebind_abstentions"] += 1
            return incumbent
        self._executed_ordinals[-1] = result.selected_ordinal
        self._counts["overrides"] += 1
        chosen = search.outcome_for(result.selected_ordinal).candidate
        self._counts[f"override_pair/{chosen.pair.category}"] += 1
        self._counts[f"override_geometry/{chosen.geometry.name}"] += 1
        return result.decision

    def statistics(self) -> dict[str, object]:
        return {
            **dict(sorted(self._counts.items())),
            "model_sha256": self.model.sha256,
            "search_wall_seconds": sum(
                value.search.wall_seconds for value in self._results
            ),
            "search_cpu_seconds": sum(
                value.search.cpu_seconds for value in self._results
            ),
        }


__all__ = [
    "SOLVENCY_SHIELD_VERSION",
    "AnalyticSolvencyTeacherPolicy",
    "CandidateCertificate",
    "CandidateLocalSolvencySearch",
    "CandidateSolvencyOutcome",
    "FrozenPolicyState",
    "LearnedScoreResidualPolicy",
    "MarginPoint",
    "RotLiabilityLedger",
    "SCORE_RESIDUAL_FEATURES",
    "ScoreResidualModel",
    "SolvencyBarrierConfig",
    "SolvencySearchResult",
    "certify_candidate",
    "passive_drain_unit",
    "renewal_epoch",
    "rot_penalty",
    "score_residual_features",
    "select_with_score_residual",
    "visible_liability_ids",
]
