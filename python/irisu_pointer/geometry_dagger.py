"""Development-only learner-visited safety DAgger for shot geometry."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np

from .geometry_policy import (
    SafeguardedGeometryPolicy,
    geometry_candidate_vocabulary_sha256,
)
from .geometry_ranking import (
    GeometryRankingDataset,
    GeometryRankingExample,
    geometry_ranking_example,
)
from .geometry_search import GeometrySearchConfig, enumerate_geometry_candidates
from .runway_search import RunwaySearchConfig, RunwaySearchResult
from .steering import SteeringDecision
from .steering_learning import GoalConditionedSteeringPolicy

GEOMETRY_DAGGER_VERSION = "r3e-learner-visited-geometry-dagger-v1"
_QUERY_REASONS = (
    "low_gauge",
    "safeguard_rejection",
    "ensemble_disagreement",
    "cadence",
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _plain_int(value: Any, name: str) -> int:
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    item = getattr(value, "item", None)
    if callable(item):
        return _plain_int(item(), name)
    raise TypeError(f"{name} must be an integer")


def _plain_float(value: Any, name: str) -> float:
    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        item = getattr(value, "item", None)
        if not callable(item):
            raise TypeError(f"{name} must be numeric")
        result = float(item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _exact_state_equal(left: Any, right: Any) -> bool:
    """Compare clone-state payloads without accepting approximate equality."""

    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C")
        )
    if isinstance(left, Mapping):
        return left.keys() == right.keys() and all(
            _exact_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _exact_state_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    try:
        equal = left == right
    except Exception:  # noqa: BLE001 - opaque clone payloads own equality
        return False
    return bool(equal) if isinstance(equal, (bool, np.bool_)) else False


def _is_piece_pair(observation: Mapping[str, Any], decision: SteeringDecision) -> bool:
    if (
        not decision.is_shot
        or decision.source_body_id is None
        or decision.destination_body_id is None
        or decision.source_body_id == decision.destination_body_id
    ):
        return False
    target = {decision.source_body_id, decision.destination_body_id}
    selected: dict[int, Mapping[str, Any]] = {}
    bodies = observation.get("bodies", ())
    if not isinstance(bodies, Sequence) or isinstance(bodies, (str, bytes)):
        return False
    for body in bodies:
        if not isinstance(body, Mapping):
            continue
        try:
            identifier = _plain_int(body.get("id"), "public body id")
        except (TypeError, ValueError):
            return False
        if identifier in selected:
            return False
        if identifier in target:
            selected[identifier] = body
    return (
        set(selected) == target
        and all(body.get("kind") == "piece" for body in selected.values())
        and all(
            body.get("shape") in {"circle", "box", "triangle"}
            for body in selected.values()
        )
    )


@dataclass(frozen=True, slots=True)
class GeometryDaggerConfig:
    """Deterministic OR-gated oracle-query and execution policy."""

    execution_mode: str = "student"
    cadence_shots: int | None = 16
    low_gauge_fraction: float | None = 0.25
    query_on_rejection: bool = True
    query_on_disagreement: bool = True
    disagreement_below: float = 1.0
    maximum_queries: int | None = None

    def __post_init__(self) -> None:
        if self.execution_mode not in {"student", "oracle"}:
            raise ValueError("geometry DAgger execution mode must be student or oracle")
        if self.cadence_shots is not None and (
            type(self.cadence_shots) is not int or self.cadence_shots < 1
        ):
            raise ValueError("geometry DAgger cadence must be positive or None")
        if self.low_gauge_fraction is not None and (
            isinstance(self.low_gauge_fraction, bool)
            or not isinstance(self.low_gauge_fraction, Real)
            or not math.isfinite(float(self.low_gauge_fraction))
            or not 0.0 <= float(self.low_gauge_fraction) <= 1.0
        ):
            raise ValueError("geometry DAgger low-gauge fraction must be in [0, 1]")
        if (
            type(self.query_on_rejection) is not bool
            or type(self.query_on_disagreement) is not bool
        ):
            raise TypeError("geometry DAgger query switches must be booleans")
        if (
            isinstance(self.disagreement_below, bool)
            or not isinstance(self.disagreement_below, Real)
            or not math.isfinite(float(self.disagreement_below))
            or not 0.0 <= float(self.disagreement_below) <= 1.0
        ):
            raise ValueError("geometry DAgger disagreement threshold must be in [0, 1]")
        if self.maximum_queries is not None and (
            type(self.maximum_queries) is not int or self.maximum_queries < 0
        ):
            raise ValueError("geometry DAgger query cap must be nonnegative or None")

    def manifest(self) -> dict[str, object]:
        return {
            **asdict(self),
            "query_rule": (
                "OR over configured low gauge, rejected non-incumbent proposal, "
                "ensemble agreement below threshold, and deterministic shot cadence"
            ),
            "cadence_origin": "first eligible directed-pair shot",
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class GeometryDaggerQueryRecord:
    """Auditable identity and disposition of one completed teacher query."""

    episode_identity: str
    provenance_sha256: str
    tick: int
    eligible_shot_index: int
    query_index: int
    reasons: tuple[str, ...]
    search_result_sha256: str
    winner_slot: int
    strictly_improved: bool
    execution_mode: str

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["reasons"] = list(self.reasons)
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


class LearnerVisitedGeometryDaggerPolicy:
    """Collect runway labels on states visited by a safeguarded learner."""

    def __init__(
        self,
        *,
        env: Any,
        base_policy: GoalConditionedSteeringPolicy,
        student_policy: SafeguardedGeometryPolicy,
        teacher: Any,
        source_identity: str,
        runtime_sha256: str,
        config: GeometryDaggerConfig | None = None,
    ) -> None:
        if getattr(env, "physics_backend", None) != "portable":
            raise ValueError("geometry DAgger requires a portable environment")
        if not callable(getattr(env, "clone_state", None)) or not callable(
            getattr(env, "restore_state", None)
        ):
            raise TypeError("geometry DAgger environment lacks clone/restore")
        if not isinstance(base_policy, GoalConditionedSteeringPolicy):
            raise TypeError("geometry DAgger base policy has an invalid type")
        if not isinstance(student_policy, SafeguardedGeometryPolicy):
            raise TypeError("geometry DAgger student policy has an invalid type")
        if student_policy.base_policy is not base_policy:
            raise ValueError("geometry DAgger base policy must have exactly one owner")
        resolved = GeometryDaggerConfig() if config is None else config
        if not isinstance(resolved, GeometryDaggerConfig):
            raise TypeError("geometry DAgger config has an invalid type")
        if (
            resolved.execution_mode == "oracle"
            and student_policy.policy_config.maximum_unverified_corrections is not None
        ):
            raise ValueError(
                "oracle execution cannot causally account student progress-credit "
                "corrections"
            )

        teacher_config = getattr(teacher, "config", None)
        candidate_config = getattr(teacher_config, "candidate_config", None)
        if not isinstance(teacher_config, RunwaySearchConfig) or not isinstance(
            candidate_config, GeometrySearchConfig
        ):
            raise TypeError("geometry DAgger teacher lacks a runway vocabulary")
        if not callable(getattr(teacher, "search", None)):
            raise TypeError("geometry DAgger teacher lacks transactional search")
        if candidate_config != student_policy.geometry_config:
            raise ValueError("geometry DAgger teacher and student vocabularies differ")
        vocabulary_sha256 = geometry_candidate_vocabulary_sha256(candidate_config)
        if student_policy.candidate_vocabulary_sha256 != vocabulary_sha256:
            raise ValueError("geometry DAgger student vocabulary identity differs")
        teacher_action_spec = getattr(teacher, "action_spec", None)
        if (
            teacher_action_spec is None
            or getattr(teacher_action_spec, "sha256", None)
            != student_policy.action_spec.sha256
        ):
            raise ValueError("geometry DAgger teacher and student actions differ")

        source_sha256 = _sha256(source_identity, "geometry DAgger source identity")
        if source_sha256 != student_policy.source_identity:
            raise ValueError("geometry DAgger student source identity differs")
        base_sha256 = _sha256(
            base_policy.artifact_sha256, "geometry DAgger base-policy checkpoint"
        )
        if base_sha256 != student_policy.base_policy_checkpoint_sha256:
            raise ValueError("geometry DAgger base-policy identity differs")

        self.env = env
        self.base_policy = base_policy
        self.student_policy = student_policy
        self.teacher = teacher
        self.config = resolved
        self.source_identity = source_sha256
        self.runtime_sha256 = _sha256(
            runtime_sha256, "geometry DAgger runtime identity"
        )
        self.base_policy_checkpoint_sha256 = base_sha256
        self.student_policy_sha256 = student_policy.sha256
        self.teacher_sha256 = _sha256(
            getattr(teacher, "sha256", None), "geometry DAgger teacher identity"
        )
        self.candidate_config = candidate_config
        self.candidate_vocabulary_sha256 = vocabulary_sha256
        self.runway_ticks = teacher_config.runway_ticks
        self.action_spec = student_policy.action_spec
        self._episode_seed: int | None = None
        self._last_tick: int | None = None
        self._last_decision: SteeringDecision | None = None
        self._failure: str | None = None
        self._ranking_examples: list[GeometryRankingExample] = []
        self._query_records: list[GeometryDaggerQueryRecord] = []
        self._selected = Counter[str]()
        self._counts = Counter[str]()
        self._reason_counts = Counter[str]()

    def identity_manifest(self) -> dict[str, object]:
        return {
            "version": GEOMETRY_DAGGER_VERSION,
            "source_identity": self.source_identity,
            "runtime_sha256": self.runtime_sha256,
            "base_policy_checkpoint_sha256": self.base_policy_checkpoint_sha256,
            "student_policy_sha256": self.student_policy_sha256,
            "teacher_sha256": self.teacher_sha256,
            "candidate_vocabulary_sha256": self.candidate_vocabulary_sha256,
            "action_spec_sha256": self.action_spec.sha256,
            "runway_ticks": self.runway_ticks,
            "config": self.config.manifest(),
            "base_state_ownership": "wrapper advances base policy exactly once per tick",
            "execution_rule": (
                "queried runway winner"
                if self.config.execution_mode == "oracle"
                else "safeguarded student decision"
            ),
            "policy_inputs": ["current public observation"],
            "teacher_only_future": (
                "public branch observations/events after exact portable restore"
            ),
            "hidden_policy_inputs": [],
            "evidence_scope": "development-teacher-only",
            "deployable": False,
            "canonical_evidence": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())

    @property
    def ranking_examples(self) -> tuple[GeometryRankingExample, ...]:
        return tuple(self._ranking_examples)

    @property
    def examples(self) -> tuple[GeometryRankingExample, ...]:
        return self.ranking_examples

    @property
    def query_records(self) -> tuple[GeometryDaggerQueryRecord, ...]:
        return tuple(self._query_records)

    def dataset(self) -> GeometryRankingDataset:
        return GeometryRankingDataset(self._ranking_examples)

    def reset(self, seed: int = 0) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "geometry DAgger integrity latch is permanent; construct a new "
                "policy after explicitly resetting the environment"
            )
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed <= 0xFFFFFFFF
        ):
            raise ValueError("geometry DAgger seed must fit uint32")
        self.base_policy.reset(seed)
        self.student_policy.reset_geometry_state()
        self._episode_seed = seed
        self._last_tick = None
        self._last_decision = None
        self._ranking_examples.clear()
        self._query_records.clear()
        self._selected.clear()
        self._counts.clear()
        self._reason_counts.clear()

    def _query_reasons(self, observation: Mapping[str, Any]) -> tuple[str, ...]:
        selection = self.student_policy.last_selection
        if selection is None:
            raise RuntimeError("geometry DAgger student omitted selection evidence")
        reasons: list[str] = []
        low = self.config.low_gauge_fraction
        if low is not None:
            gauge = _plain_float(observation.get("gauge"), "public gauge")
            gauge_max = _plain_float(
                observation.get("gauge_max"), "public maximum gauge"
            )
            if gauge < 0.0 or gauge_max <= 0.0:
                raise ValueError("public gauge values are invalid")
            if gauge / gauge_max <= float(low):
                reasons.append("low_gauge")
        if (
            self.config.query_on_rejection
            and selection.proposed_slot not in (None, 0)
            and not selection.used_learned_geometry
        ):
            reasons.append("safeguard_rejection")
        if (
            self.config.query_on_disagreement
            and selection.selector_member_count > 1
            and selection.ensemble_agreement is not None
            and selection.ensemble_agreement < float(self.config.disagreement_below)
        ):
            reasons.append("ensemble_disagreement")
        cadence = self.config.cadence_shots
        if cadence is not None and (self._counts["eligible_shots"] - 1) % cadence == 0:
            reasons.append("cadence")
        if any(reason not in _QUERY_REASONS for reason in reasons):
            raise RuntimeError("geometry DAgger produced an unknown query reason")
        return tuple(reasons)

    def _validate_result(
        self,
        result: Any,
        incumbent: SteeringDecision,
        expected_candidate_set_sha256: str,
    ) -> RunwaySearchResult:
        if not isinstance(result, RunwaySearchResult):
            raise TypeError("geometry DAgger teacher returned an invalid result")
        winner = result.candidate_set.candidate_at(result.winner_ordinal)
        if (
            result.teacher_identity_sha256 != self.teacher_sha256
            or result.runway_ticks != self.runway_ticks
            or result.candidate_set.config != self.candidate_config
            or result.candidate_set.sha256 != expected_candidate_set_sha256
            or result.candidate_set.action_spec_sha256 != self.action_spec.sha256
            or geometry_candidate_vocabulary_sha256(result.candidate_set.config)
            != self.candidate_vocabulary_sha256
            or not result.outcomes
            or winner is None
            or result.selected_candidate != winner
            or result.candidate_set.candidates[0].decision != incumbent
        ):
            raise RuntimeError("geometry DAgger teacher returned inconsistent evidence")
        return result

    def _execute(
        self, tick: int, student: SteeringDecision, oracle: SteeringDecision | None
    ) -> SteeringDecision:
        if oracle is not None and self.config.execution_mode == "oracle":
            decision = oracle
            self._counts["oracle_actions_executed"] += 1
        else:
            decision = student
            self._counts["student_actions_executed"] += 1
        self._counts["decisions"] += 1
        self._last_tick = tick
        self._last_decision = decision
        return decision

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        if self._episode_seed is None:
            raise RuntimeError("geometry DAgger must be reset before prediction")
        if self._failure is not None:
            raise RuntimeError(
                f"geometry DAgger is latched after evidence failure: {self._failure}"
            )
        if not isinstance(observation, Mapping):
            raise TypeError("geometry DAgger observation must be a public mapping")
        tick = _plain_int(observation.get("tick", 0), "public observation tick")
        if tick < 0:
            raise ValueError("public observation tick must be nonnegative")
        if self._last_tick is not None:
            if tick < self._last_tick:
                raise RuntimeError("geometry DAgger observation tick moved backwards")
            if tick == self._last_tick:
                assert self._last_decision is not None
                return self._last_decision

        incumbent = self.base_policy.predict(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("geometry DAgger base policy returned an invalid decision")
        student = self.student_policy.select_from_incumbent(observation, incumbent)
        if not incumbent.is_shot:
            self._counts["nonshot_decisions"] += 1
            return self._execute(tick, student, None)
        self._counts["seen_shots"] += 1
        if not _is_piece_pair(observation, incumbent):
            self._counts["unsupported_pairs"] += 1
            return self._execute(tick, student, None)

        self._counts["eligible_shots"] += 1
        reasons = self._query_reasons(observation)
        for reason in reasons:
            self._reason_counts[reason] += 1
        if not reasons:
            return self._execute(tick, student, None)
        maximum = self.config.maximum_queries
        if maximum is not None and self._counts["query_attempts"] >= maximum:
            self._counts["query_budget_skips"] += 1
            return self._execute(tick, student, None)

        pre_action = copy.deepcopy(dict(observation))
        expected_candidates = enumerate_geometry_candidates(
            pre_action,
            incumbent,
            config=self.candidate_config,
            action_spec=self.action_spec,
        )
        selection = self.student_policy.last_selection
        if (
            selection is None
            or selection.candidate_set_sha256 is None
            or selection.candidate_set_sha256 != expected_candidates.sha256
        ):
            self._failure = "student candidate-set evidence changed before query"
            raise RuntimeError(self._failure)
        teacher_observation = copy.deepcopy(pre_action)
        self._counts["query_attempts"] += 1
        before = self.env.clone_state()
        try:
            result = self.teacher.search(self.env, teacher_observation, incumbent)
        except Exception as exc:
            try:
                restored = _exact_state_equal(before, self.env.clone_state())
            except Exception as verification_error:
                self._failure = "teacher failed and source state could not be verified"
                raise RuntimeError(self._failure) from verification_error
            self._counts["transactional_restore_checks"] += 1
            self._failure = (
                "teacher search failed after restoring source state"
                if restored
                else "teacher search failed and changed source state"
            )
            if not restored:
                raise RuntimeError(self._failure) from exc
            raise

        self._counts["transactional_restore_checks"] += 1
        try:
            after = self.env.clone_state()
        except Exception as verification_error:
            self._failure = "teacher source state could not be verified"
            raise RuntimeError(self._failure) from verification_error
        if not _exact_state_equal(before, after):
            self._failure = "teacher changed its source state"
            raise RuntimeError(self._failure)
        if not _exact_state_equal(pre_action, teacher_observation):
            self._failure = "teacher changed its public pre-action observation"
            raise RuntimeError(self._failure)
        try:
            result = self._validate_result(
                result, incumbent, expected_candidates.sha256
            )
            provenance = _canonical_sha256(
                {
                    "dagger_policy_sha256": self.sha256,
                    "source_identity": self.source_identity,
                    "runtime_sha256": self.runtime_sha256,
                    "base_policy_checkpoint_sha256": (
                        self.base_policy_checkpoint_sha256
                    ),
                    "student_policy_sha256": self.student_policy_sha256,
                    "teacher_sha256": self.teacher_sha256,
                    "candidate_vocabulary_sha256": (self.candidate_vocabulary_sha256),
                    "execution_mode": self.config.execution_mode,
                    "seed": self._episode_seed,
                    "tick": tick,
                    "eligible_shot_index": self._counts["eligible_shots"],
                    "query_index": self._counts["search_queries"] + 1,
                    "query_reasons": list(reasons),
                    "search_result_sha256": result.sha256,
                }
            )
            episode_identity = (
                f"geometry-dagger-{self.config.execution_mode}:"
                f"{self._episode_seed:08x}:{tick}:"
                f"{self._counts['eligible_shots']}:"
                f"{self._counts['search_queries'] + 1}"
            )
            example = geometry_ranking_example(
                pre_action,
                result,
                episode_identity=episode_identity,
                provenance_sha256=provenance,
                encoder=self.base_policy.encoder,
            )
            if (
                example.winner_index != result.winner_ordinal
                or example.improved_over_incumbent != result.strictly_improved
                or example.candidate_vocabulary_sha256
                != self.candidate_vocabulary_sha256
            ):
                raise RuntimeError(
                    "geometry DAgger ranking conversion changed the oracle label"
                )
        except Exception:
            self._failure = "teacher evidence failed identity validation"
            raise

        self._counts["search_queries"] += 1
        self._counts["strict_improvements"] += int(result.strictly_improved)
        self._counts["branch_outcomes"] += len(result.outcomes)
        self._selected[result.selected_candidate.name] += 1
        self._ranking_examples.append(example)
        self._query_records.append(
            GeometryDaggerQueryRecord(
                episode_identity,
                provenance,
                tick,
                self._counts["eligible_shots"],
                self._counts["search_queries"],
                reasons,
                result.sha256,
                result.winner_ordinal,
                result.strictly_improved,
                self.config.execution_mode,
            )
        )
        return self._execute(tick, student, result.decision)

    def act(self, observation: Mapping[str, Any]) -> tuple[Any, ...]:
        return self.predict(observation).primitive_actions(self.action_spec)

    def statistics(self) -> dict[str, object]:
        counts = {
            name: int(self._counts[name])
            for name in (
                "decisions",
                "nonshot_decisions",
                "seen_shots",
                "eligible_shots",
                "unsupported_pairs",
                "query_attempts",
                "search_queries",
                "query_budget_skips",
                "transactional_restore_checks",
                "strict_improvements",
                "branch_outcomes",
                "student_actions_executed",
                "oracle_actions_executed",
            )
        }
        counts.update(
            {
                f"{reason}_triggers": int(self._reason_counts[reason])
                for reason in _QUERY_REASONS
            }
        )
        return {
            **counts,
            "query_reason_counts": dict(sorted(self._reason_counts.items())),
            "selected_candidate_counts": dict(sorted(self._selected.items())),
            "execution_mode": self.config.execution_mode,
            "failure_latched": self._failure is not None,
        }


__all__ = [
    "GEOMETRY_DAGGER_VERSION",
    "GeometryDaggerConfig",
    "GeometryDaggerQueryRecord",
    "LearnerVisitedGeometryDaggerPolicy",
]
