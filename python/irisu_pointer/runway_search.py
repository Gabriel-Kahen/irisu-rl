"""Development-only full-runway teacher for directed-pair shot geometry.

Unlike causal geometry search, this teacher deliberately crosses future spawn
boundaries.  It is suitable for development labels and diagnostics only: it is
not a deployable policy and its outcomes are not canonical or sealed evidence.
Candidate policy inputs remain the initial public observation and incumbent
pair decision; every future is generated from the same restored portable
snapshot, including identical RNG state.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from irisu_rl.actions import ActionSpec

from .geometry_search import (
    GeometryBranchOutcome,
    GeometryCandidate,
    GeometryCandidateSet,
    GeometrySearchConfig,
    _public_state_signature,
    enumerate_geometry_candidates,
    evaluate_geometry_candidate,
)
from .steering import SteeringDecision


RUNWAY_SEARCH_VERSION = "r3e-runway-geometry-teacher-v4"


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class RunwaySearchConfig:
    """Fixed geometry vocabulary and deliberately noncausal rollout length."""

    runway_ticks: int = 256
    candidate_config: GeometrySearchConfig = field(
        default_factory=GeometrySearchConfig
    )

    def __post_init__(self) -> None:
        if (
            type(self.runway_ticks) is not int
            or not 2 <= self.runway_ticks <= 100_000
        ):
            raise ValueError("runway ticks must be in [2, 100000]")
        if not isinstance(self.candidate_config, GeometrySearchConfig):
            raise TypeError(
                "runway candidate config must be a GeometrySearchConfig"
            )

    def manifest(self) -> dict[str, object]:
        return {
            "version": RUNWAY_SEARCH_VERSION,
            "runway_ticks": self.runway_ticks,
            "candidate_config": self.candidate_config.manifest(),
            "candidate_slot_count": self.candidate_config.slot_count,
            "branch_protocol": (
                "restore the identical portable snapshot and RNG, execute one "
                "candidate, then no-action coast for the complete runway"
            ),
            "spawn_policy": "deliberately cross cadence spawns",
            "selection_rule": (
                "valid branches survival-nondominated by incumbent; "
                "reserve-band objective protects min(final gauge, half maximum), "
                "then prefers score and causal control; "
                "incumbent wins exact ties"
            ),
            "evidence_scope": "development-teacher-only",
            "deployable": False,
            "canonical_evidence": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class RunwaySearchResult:
    """Winner plus every public full-runway branch outcome."""

    teacher_identity_sha256: str
    candidate_set: GeometryCandidateSet
    runway_ticks: int
    selected_candidate: GeometryCandidate
    strictly_improved: bool
    outcomes: tuple[GeometryBranchOutcome, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.teacher_identity_sha256, str)
            or len(self.teacher_identity_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.teacher_identity_sha256
            )
        ):
            raise ValueError("runway teacher identity must be a lowercase SHA-256")
        if type(self.runway_ticks) is not int or self.runway_ticks < 2:
            raise ValueError("runway result ticks must be at least two")
        if len(self.outcomes) != len(self.candidate_set.candidates):
            raise ValueError("runway result must retain every branch outcome")
        if not any(
            outcome.candidate.ordinal == self.selected_candidate.ordinal
            for outcome in self.outcomes
        ):
            raise ValueError("runway winner is absent from branch outcomes")

    @property
    def decision(self) -> SteeringDecision:
        return self.selected_candidate.decision

    @property
    def winner_ordinal(self) -> int:
        return self.selected_candidate.ordinal

    def manifest(self) -> dict[str, object]:
        return {
            "version": RUNWAY_SEARCH_VERSION,
            "teacher_identity_sha256": self.teacher_identity_sha256,
            "candidate_set_sha256": self.candidate_set.sha256,
            "runway_ticks": self.runway_ticks,
            "winner_ordinal": self.winner_ordinal,
            "selected_candidate": self.selected_candidate.name,
            "strictly_improved": self.strictly_improved,
            "outcomes": [outcome.manifest() for outcome in self.outcomes],
            "evidence_scope": "development-teacher-only",
            "deployable": False,
            "canonical_evidence": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


class RunwayGeometrySearch:
    """Portable branch teacher over a complete, spawn-crossing runway."""

    def __init__(
        self,
        *,
        config: RunwaySearchConfig | None = None,
        action_spec: ActionSpec | None = None,
    ) -> None:
        self.config = RunwaySearchConfig() if config is None else config
        self.action_spec = ActionSpec() if action_spec is None else action_spec
        if not isinstance(self.config, RunwaySearchConfig):
            raise TypeError("runway config must be a RunwaySearchConfig")
        if not isinstance(self.action_spec, ActionSpec):
            raise TypeError("runway action spec must be an ActionSpec")

    def identity_manifest(self) -> dict[str, object]:
        return {
            "version": RUNWAY_SEARCH_VERSION,
            "config": self.config.manifest(),
            "action_spec": self.action_spec.manifest(),
            "policy_inputs": [
                "initial public observation",
                "incumbent public directed-pair decision",
            ],
            "teacher_only_future": (
                "public observations/events produced after restoring the same "
                "portable snapshot for each fixed candidate"
            ),
            "hidden_policy_inputs": [],
            "evidence_scope": "development-teacher-only",
            "deployable": False,
            "canonical_evidence": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())

    def search(
        self,
        env: Any,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
    ) -> RunwaySearchResult:
        if getattr(env, "physics_backend", None) != "portable":
            raise ValueError("runway geometry search requires portable backend")
        if not isinstance(observation, Mapping):
            raise TypeError("runway observation must be a public mapping")
        if bool(observation.get("terminated", False)) or bool(
            observation.get("truncated", False)
        ):
            raise ValueError("cannot runway-search a terminal observation")
        clone = getattr(env, "clone_state", None)
        restore = getattr(env, "restore_state", None)
        if not callable(clone) or not callable(restore):
            raise TypeError("portable runway environment lacks clone/restore")

        candidate_set = enumerate_geometry_candidates(
            observation,
            incumbent,
            config=self.config.candidate_config,
            action_spec=self.action_spec,
        )
        expected = _public_state_signature(observation)
        snapshot = clone()
        outcomes: list[GeometryBranchOutcome] = []
        try:
            for candidate in candidate_set.candidates:
                restored = restore(snapshot)
                if not isinstance(restored, Mapping):
                    raise TypeError("portable restore must return a public mapping")
                if _public_state_signature(restored) != expected:
                    raise RuntimeError(
                        "portable restore disagrees with the supplied public state"
                    )
                outcomes.append(
                    evaluate_geometry_candidate(
                        env,
                        restored,
                        candidate,
                        horizon_ticks=self.config.runway_ticks,
                        action_spec=self.action_spec,
                    )
                )
        finally:
            restore(snapshot)

        incumbent_outcome = outcomes[0]
        if not incumbent_outcome.selectable:
            raise RuntimeError("incumbent runway branch emitted an invalid action")
        eligible = tuple(
            outcome
            for outcome in outcomes
            if outcome.selectable
            and outcome.survival_nondominated_by(incumbent_outcome)
        )
        winner = max(
            eligible,
            key=lambda outcome: (outcome.objective, -outcome.candidate.ordinal),
        )
        strictly_improved = winner.objective > incumbent_outcome.objective
        if not strictly_improved:
            winner = incumbent_outcome
        return RunwaySearchResult(
            self.sha256,
            candidate_set,
            self.config.runway_ticks,
            winner.candidate,
            strictly_improved,
            tuple(outcomes),
        )

    def act(
        self,
        env: Any,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
    ) -> SteeringDecision:
        return self.search(env, observation, incumbent).decision

    choose = act


__all__ = [
    "RUNWAY_SEARCH_VERSION",
    "RunwayGeometrySearch",
    "RunwaySearchConfig",
    "RunwaySearchResult",
]
