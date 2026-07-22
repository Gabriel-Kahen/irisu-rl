"""Deterministic, checkpointable curriculum identity and promotion state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Mapping, Sequence

import torch
from torch import Tensor

from .actions import ActionSpec
from .rewards import RewardSchedule


_SHA256_ZERO = "0" * 64
_SPLITS = {"train", "validation", "calibration", "test"}
_PHASES = {
    "normal",
    "remediation",
    "budget_validation",
    "complete",
    "budget_exhausted",
}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class SnapshotRecipe:
    """Authority for constructing one reachable curriculum decision boundary.

    Snapshot bytes are caches. The reset seed plus serialized legal semantic
    trace is the provenance that must reproduce the declared state hash.
    """

    snapshot_id: str
    stage_id: str
    split: str
    scenario_family: str
    environment_pool: str
    config_sha256: str
    config_hash: int
    reset_seed: int
    action_spec_sha256: str
    semantic_actions_hex: tuple[str, ...]
    expected_tick: int
    expected_score: int
    expected_state_hash: int
    snapshot_sha256: str
    runtime_identity_sha256: str
    generator_version: str
    version: str = "curriculum-snapshot-recipe-v1"

    def __post_init__(self) -> None:
        identifiers = (
            self.snapshot_id,
            self.stage_id,
            self.scenario_family,
            self.environment_pool,
            self.generator_version,
        )
        if any(not value or not value.isascii() for value in identifiers):
            raise ValueError("recipe identifiers must be nonempty ASCII")
        if self.split not in _SPLITS:
            raise ValueError("unknown curriculum recipe split")
        for value in (
            self.config_sha256,
            self.action_spec_sha256,
            self.snapshot_sha256,
            self.runtime_identity_sha256,
        ):
            if not _is_sha256(value):
                raise ValueError("recipe identities must be lowercase SHA-256 values")
        integers = (
            self.config_hash,
            self.reset_seed,
            self.expected_tick,
            self.expected_score,
            self.expected_state_hash,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in integers
        ):
            raise TypeError("recipe numeric identity fields must be integers")
        if (
            not 0 <= self.config_hash < 2**64
            or not 0 <= self.expected_state_hash < 2**64
        ):
            raise ValueError("recipe state/config hashes must fit uint64")
        if not 0 <= self.reset_seed < 2**32 or self.expected_tick < 0:
            raise ValueError("recipe reset seed or tick is invalid")
        action_spec = ActionSpec()
        if self.action_spec_sha256 != action_spec.sha256:
            raise ValueError("snapshot recipe uses an unsupported action schema")
        for payload in self.semantic_actions_hex:
            try:
                action_spec.deserialize(bytes.fromhex(payload))
            except (ValueError, TypeError) as exc:
                raise ValueError("recipe contains a malformed semantic action") from exc

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["semantic_actions_hex"] = list(self.semantic_actions_hex)
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


class SnapshotLibrary:
    """Immutable recipe catalog with source-family split protection."""

    version = "curriculum-snapshot-library-v1"

    def __init__(self, recipes: Sequence[SnapshotRecipe]) -> None:
        if not recipes:
            raise ValueError("snapshot library cannot be empty")
        ordered = tuple(sorted(recipes, key=lambda recipe: recipe.snapshot_id))
        if len({recipe.snapshot_id for recipe in ordered}) != len(ordered):
            raise ValueError("snapshot ids must be unique")
        if len({recipe.snapshot_sha256 for recipe in ordered}) != len(ordered):
            raise ValueError("snapshot blobs must not be duplicated under new ids")
        pool_identities: dict[str, tuple[str, int, str]] = {}
        for recipe in ordered:
            identity = (
                recipe.config_sha256,
                int(recipe.config_hash),
                recipe.runtime_identity_sha256,
            )
            previous = pool_identities.setdefault(recipe.environment_pool, identity)
            if previous != identity:
                raise ValueError(
                    "one environment pool cannot mix mechanics or runtime identities"
                )
        train_families = {
            recipe.scenario_family for recipe in ordered if recipe.split == "train"
        }
        validation_families = {
            recipe.scenario_family for recipe in ordered if recipe.split == "validation"
        }
        if train_families & validation_families:
            raise ValueError("training and validation scenario families overlap")
        self.recipes = ordered
        self._by_id = {recipe.snapshot_id: recipe for recipe in ordered}

    def __getitem__(self, snapshot_id: str) -> SnapshotRecipe:
        try:
            return self._by_id[snapshot_id]
        except KeyError as exc:
            raise KeyError(f"unknown curriculum snapshot: {snapshot_id}") from exc

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "recipes": [recipe.manifest() for recipe in self.recipes],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    def verify_snapshot_blob(self, snapshot_id: str, payload: bytes) -> None:
        recipe = self[snapshot_id]
        if hashlib.sha256(bytes(payload)).hexdigest() != recipe.snapshot_sha256:
            raise ValueError("curriculum snapshot blob hash mismatch")


@dataclass(frozen=True, slots=True)
class StageSpec:
    stage_id: str
    rank: int
    environment_pool: str
    train_snapshot_ids: tuple[str, ...]
    validation_snapshot_ids: tuple[str, ...]
    enabled_action_kinds: tuple[int, ...]
    enabled_wait_ticks: tuple[int, ...]
    promotion_successes: int
    promotion_trials: int
    regression_successes: int
    regression_trials: int
    required_consecutive_passes: int
    max_updates: int
    reward_schedule: RewardSchedule
    version: str = "curriculum-stage-v1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stage_id, str)
            or not self.stage_id
            or not self.stage_id.isascii()
            or not self.environment_pool
        ):
            raise ValueError("stage identity must be nonempty")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, Integral)
            or self.rank < 0
        ):
            raise ValueError("stage rank must be a nonnegative integer")
        if not self.train_snapshot_ids or not self.validation_snapshot_ids:
            raise ValueError("stage requires training and validation snapshots")
        if len(set(self.train_snapshot_ids)) != len(self.train_snapshot_ids) or len(
            set(self.validation_snapshot_ids)
        ) != len(self.validation_snapshot_ids):
            raise ValueError("stage snapshot ids must be unique")
        if any(
            isinstance(kind, bool) or not isinstance(kind, Integral)
            for kind in self.enabled_action_kinds
        ):
            raise TypeError("enabled action kinds must be integers")
        kinds = tuple(sorted(set(self.enabled_action_kinds)))
        if (
            kinds != self.enabled_action_kinds
            or not kinds
            or any(kind not in (0, 1, 2) for kind in kinds)
        ):
            raise ValueError(
                "enabled action kinds must be sorted unique values in [0, 2]"
            )
        if any(
            isinstance(wait, bool) or not isinstance(wait, Integral)
            for wait in self.enabled_wait_ticks
        ):
            raise TypeError("enabled wait durations must be integers")
        waits = tuple(sorted(set(self.enabled_wait_ticks)))
        if waits != self.enabled_wait_ticks or (0 in kinds and not waits):
            raise ValueError("WAIT-enabled stages require sorted unique wait durations")
        if any(wait not in ActionSpec().wait_choices for wait in waits):
            raise ValueError("stage wait duration is absent from the action schema")
        counts = (
            self.promotion_successes,
            self.promotion_trials,
            self.regression_successes,
            self.regression_trials,
            self.required_consecutive_passes,
            self.max_updates,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value <= 0
            for value in counts
        ):
            raise ValueError("stage gates and budgets must be positive integers")
        if (
            self.promotion_successes > self.promotion_trials
            or self.regression_successes > self.regression_trials
        ):
            raise ValueError("stage success requirements cannot exceed trial counts")
        if min(self.promotion_trials, self.regression_trials) < len(
            self.validation_snapshot_ids
        ):
            raise ValueError(
                "validation trials must cover every declared snapshot recipe"
            )

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "stage_id": self.stage_id,
            "rank": int(self.rank),
            "environment_pool": self.environment_pool,
            "train_snapshot_ids": list(self.train_snapshot_ids),
            "validation_snapshot_ids": list(self.validation_snapshot_ids),
            "enabled_action_kinds": list(self.enabled_action_kinds),
            "enabled_wait_ticks": list(self.enabled_wait_ticks),
            "promotion_successes": int(self.promotion_successes),
            "promotion_trials": int(self.promotion_trials),
            "regression_successes": int(self.regression_successes),
            "regression_trials": int(self.regression_trials),
            "required_consecutive_passes": int(self.required_consecutive_passes),
            "max_updates": int(self.max_updates),
            "reward_schedule": self.reward_schedule.manifest(),
        }


@dataclass(frozen=True, slots=True)
class CurriculumSpec:
    curriculum_id: str
    library: SnapshotLibrary
    stages: tuple[StageSpec, ...]
    prior_stage_mix_ppm: int = 200_000
    version: str = "curriculum-v1"

    def __post_init__(self) -> None:
        if (
            not self.curriculum_id
            or not self.curriculum_id.isascii()
            or not self.stages
        ):
            raise ValueError("curriculum identity and stages are required")
        ranks = tuple(stage.rank for stage in self.stages)
        if ranks != tuple(range(len(self.stages))):
            raise ValueError("curriculum stage ranks must be contiguous and ordered")
        if len({stage.stage_id for stage in self.stages}) != len(self.stages):
            raise ValueError("curriculum stage ids must be unique")
        if isinstance(self.prior_stage_mix_ppm, bool) or not isinstance(
            self.prior_stage_mix_ppm, Integral
        ):
            raise TypeError("prior-stage mix must be an integer ppm value")
        if not 0 <= self.prior_stage_mix_ppm <= 1_000_000:
            raise ValueError("prior-stage mix must be within [0, 1_000_000] ppm")
        for stage in self.stages:
            for snapshot_id in stage.train_snapshot_ids:
                recipe = self.library[snapshot_id]
                if recipe.stage_id != stage.stage_id or recipe.split != "train":
                    raise ValueError("stage training snapshot identity mismatch")
                if recipe.environment_pool != stage.environment_pool:
                    raise ValueError(
                        "stage snapshot belongs to a different environment pool"
                    )
            for snapshot_id in stage.validation_snapshot_ids:
                recipe = self.library[snapshot_id]
                if recipe.stage_id != stage.stage_id or recipe.split != "validation":
                    raise ValueError("stage validation snapshot identity mismatch")
                if recipe.environment_pool != stage.environment_pool:
                    raise ValueError(
                        "stage snapshot belongs to a different environment pool"
                    )

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "curriculum_id": self.curriculum_id,
            "library_sha256": self.library.sha256,
            "prior_stage_mix_ppm": self.prior_stage_mix_ppm,
            "stages": [stage.manifest() for stage in self.stages],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class EpisodeAssignment:
    lane_id: int
    episode_ordinal: int
    stage_id: str
    snapshot_id: str


@dataclass(frozen=True, slots=True)
class AssignmentReservation:
    assignments: tuple[EpisodeAssignment, ...]
    state_hash: str


@dataclass(frozen=True, slots=True)
class ValidationEpisodeOutcome:
    """One evaluator result bound to a requested recipe and repetition."""

    snapshot_id: str
    repetition: int
    policy_seed: int
    success: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.snapshot_id, str)
            or not self.snapshot_id
            or isinstance(self.repetition, bool)
            or not isinstance(self.repetition, Integral)
            or self.repetition < 0
            or isinstance(self.policy_seed, bool)
            or not isinstance(self.policy_seed, Integral)
            or not 0 <= self.policy_seed < 2**64
            or not isinstance(self.success, bool)
        ):
            raise ValueError("validation episode outcome is malformed")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    stage_id: str
    successes: int
    episodes: int
    snapshot_ids: tuple[str, ...] = ()
    outcomes: tuple[ValidationEpisodeOutcome, ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stage_id, str)
            or not self.stage_id
            or isinstance(self.successes, bool)
            or isinstance(self.episodes, bool)
            or not isinstance(self.successes, Integral)
            or not isinstance(self.episodes, Integral)
            or self.episodes <= 0
            or not 0 <= self.successes <= self.episodes
            or len(set(self.snapshot_ids)) != len(self.snapshot_ids)
            or any(
                not isinstance(snapshot_id, str) or not snapshot_id
                for snapshot_id in self.snapshot_ids
            )
            or not isinstance(self.outcomes, tuple)
            or any(
                not isinstance(outcome, ValidationEpisodeOutcome)
                for outcome in self.outcomes
            )
            or (self.outcomes and len(self.outcomes) != self.episodes)
            or (
                self.outcomes
                and sum(outcome.success for outcome in self.outcomes) != self.successes
            )
        ):
            raise ValueError("validation result is malformed")


@dataclass(frozen=True, slots=True)
class ValidationStageRequest:
    stage_id: str
    snapshot_ids: tuple[str, ...]
    episodes: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.stage_id, str)
            or not self.stage_id
            or not self.snapshot_ids
            or not isinstance(self.snapshot_ids, tuple)
            or any(
                not isinstance(snapshot_id, str) or not snapshot_id
                for snapshot_id in self.snapshot_ids
            )
            or len(set(self.snapshot_ids)) != len(self.snapshot_ids)
            or isinstance(self.episodes, bool)
            or not isinstance(self.episodes, Integral)
            or self.episodes <= 0
        ):
            raise ValueError("validation stage request is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "snapshot_ids": list(self.snapshot_ids),
            "episodes": int(self.episodes),
        }


@dataclass(frozen=True, slots=True)
class ValidationRequest:
    curriculum_sha256: str
    gate_ordinal: int
    completed_update: int
    policy_sha256: str
    evaluator_identity_sha256: str
    stages: tuple[ValidationStageRequest, ...]
    version: str = "curriculum-validation-request-v1"

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.curriculum_sha256)
            or not _is_sha256(self.policy_sha256)
            or not _is_sha256(self.evaluator_identity_sha256)
            or isinstance(self.gate_ordinal, bool)
            or not isinstance(self.gate_ordinal, Integral)
            or self.gate_ordinal < 0
            or isinstance(self.completed_update, bool)
            or not isinstance(self.completed_update, Integral)
            or self.completed_update < 0
            or not self.stages
            or len({stage.stage_id for stage in self.stages}) != len(self.stages)
        ):
            raise ValueError("validation request identity is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "curriculum_sha256": self.curriculum_sha256,
            "gate_ordinal": int(self.gate_ordinal),
            "completed_update": int(self.completed_update),
            "policy_sha256": self.policy_sha256,
            "evaluator_identity_sha256": self.evaluator_identity_sha256,
            "stages": [stage.manifest() for stage in self.stages],
        }

    @property
    def request_id(self) -> str:
        return _canonical_sha256(self.manifest())

    def episode_seed(self, stage_id: str, snapshot_id: str, repetition: int) -> int:
        """Derive the evaluator policy RNG seed for one requested episode."""

        if (
            not isinstance(stage_id, str)
            or not stage_id
            or not isinstance(snapshot_id, str)
            or not snapshot_id
            or isinstance(repetition, bool)
            or not isinstance(repetition, Integral)
            or repetition < 0
        ):
            raise ValueError("validation episode seed coordinates are malformed")
        payload = (
            f"{self.curriculum_sha256}:{self.gate_ordinal}:"
            f"{self.completed_update}:{self.policy_sha256}:"
            f"{self.evaluator_identity_sha256}:{stage_id}:{snapshot_id}:"
            f"{repetition}"
        ).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> ValidationRequest:
        expected = {
            "version",
            "curriculum_sha256",
            "gate_ordinal",
            "completed_update",
            "policy_sha256",
            "evaluator_identity_sha256",
            "stages",
        }
        if (
            set(value) != expected
            or value["version"] != "curriculum-validation-request-v1"
        ):
            raise ValueError("validation request checkpoint identity mismatch")
        stages = value["stages"]
        if not isinstance(stages, list):
            raise ValueError("validation request stages are malformed")
        parsed = []
        for stage in stages:
            if not isinstance(stage, Mapping) or set(stage) != {
                "stage_id",
                "snapshot_ids",
                "episodes",
            }:
                raise ValueError("validation stage request is malformed")
            snapshot_ids = stage["snapshot_ids"]
            if not isinstance(snapshot_ids, list):
                raise ValueError("validation stage snapshot ids are malformed")
            parsed.append(
                ValidationStageRequest(
                    stage["stage_id"],  # type: ignore[arg-type]
                    tuple(snapshot_ids),  # type: ignore[arg-type]
                    stage["episodes"],  # type: ignore[arg-type]
                )
            )
        return cls(
            value["curriculum_sha256"],  # type: ignore[arg-type]
            value["gate_ordinal"],  # type: ignore[arg-type]
            value["completed_update"],  # type: ignore[arg-type]
            value["policy_sha256"],  # type: ignore[arg-type]
            value["evaluator_identity_sha256"],  # type: ignore[arg-type]
            tuple(parsed),
        )


@dataclass(frozen=True, slots=True)
class ValidationReport:
    request_id: str
    policy_sha256: str
    evaluator_identity_sha256: str
    results: tuple[ValidationResult, ...]

    def __post_init__(self) -> None:
        if (
            not _is_sha256(self.request_id)
            or not _is_sha256(self.policy_sha256)
            or not _is_sha256(self.evaluator_identity_sha256)
            or not self.results
        ):
            raise ValueError("validation report identity is malformed")
        if len({result.stage_id for result in self.results}) != len(self.results):
            raise ValueError("validation report repeats a stage")

    def manifest(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "policy_sha256": self.policy_sha256,
            "evaluator_identity_sha256": self.evaluator_identity_sha256,
            "results": [
                asdict(result)
                for result in sorted(self.results, key=lambda item: item.stage_id)
            ],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class GateDecision:
    phase: str
    highest_unlocked_stage: str
    focus_stage: str
    promoted: bool
    remediation_stage: str | None
    reason: str


class CurriculumCoordinator:
    """Transactional assignment sampler and monotone promotion state machine."""

    version = "curriculum-coordinator-v1"

    def __init__(self, spec: CurriculumSpec, lanes: int, *, learner_seed: int) -> None:
        if isinstance(lanes, bool) or not isinstance(lanes, Integral) or lanes <= 0:
            raise ValueError("curriculum lane count must be a positive integer")
        if (
            isinstance(learner_seed, bool)
            or not isinstance(learner_seed, Integral)
            or not 0 <= learner_seed < 2**64
        ):
            raise ValueError("curriculum learner seed must fit uint64")
        self.spec = spec
        self.lanes = int(lanes)
        self.learner_seed = int(learner_seed)
        self.highest_unlocked = 0
        self.focus = 0
        self.phase = "normal"
        self.completed_updates = 0
        self.unlock_updates = [0] + [-1] * (len(spec.stages) - 1)
        self.promotion_streak = 0
        self.lane_stage = [0] * self.lanes
        initial_weight = spec.stages[0].reward_schedule.weight_ppm(0)
        self.lane_shaping_weight_ppm = [initial_weight] * self.lanes
        self.episode_ordinals = [0] * self.lanes
        self._outstanding: AssignmentReservation | None = None
        self._submitted_reports: dict[str, str] = {}
        self._evaluated_policies: set[str] = set()
        self._evaluation_ordinal = 0
        self._pending_validation: ValidationRequest | None = None
        self._event_head = _SHA256_ZERO

    @property
    def current_stage(self) -> StageSpec:
        return self.spec.stages[self.focus]

    @property
    def event_head(self) -> str:
        return self._event_head

    @property
    def validation_pending(self) -> bool:
        return self._pending_validation is not None

    def _sample_index(self, lane: int, ordinal: int, size: int) -> int:
        counter = 0
        limit = (1 << 256) - ((1 << 256) % size)
        while True:
            payload = (
                f"{self.spec.sha256}:{self.learner_seed}:{lane}:{ordinal}:{counter}"
            ).encode()
            value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
            if value < limit:
                return value % size
            counter += 1

    def reserve_assignments(self, lane_ids: Sequence[int]) -> AssignmentReservation:
        if self._outstanding is not None:
            raise RuntimeError("an assignment reservation is already outstanding")
        lanes = tuple(int(lane) for lane in lane_ids)
        if len(set(lanes)) != len(lanes) or any(
            not 0 <= lane < self.lanes for lane in lanes
        ):
            raise ValueError("assignment lanes must be unique and in range")
        assignments = []
        for lane in lanes:
            ordinal = self.episode_ordinals[lane]
            stage_index = self.focus
            # Retain a deterministic fraction of earlier stages after promotion.
            if self.phase == "normal" and self.highest_unlocked > 0:
                selector = self._sample_index(lane, ordinal, 1_000_000)
                if selector < self.spec.prior_stage_mix_ppm:
                    stage_index = self._sample_index(
                        lane, ordinal + 1, self.highest_unlocked
                    )
            stage = self.spec.stages[stage_index]
            snapshot_index = self._sample_index(
                lane, ordinal + 2, len(stage.train_snapshot_ids)
            )
            assignments.append(
                EpisodeAssignment(
                    lane,
                    ordinal,
                    stage.stage_id,
                    stage.train_snapshot_ids[snapshot_index],
                )
            )
        reservation = AssignmentReservation(
            tuple(assignments),
            _canonical_sha256(
                {
                    "spec": self.spec.sha256,
                    "assignments": [asdict(value) for value in assignments],
                }
            ),
        )
        self._outstanding = reservation
        return reservation

    def commit_assignments(self, reservation: AssignmentReservation) -> None:
        if reservation != self._outstanding:
            raise ValueError("stale or foreign assignment reservation")
        for assignment in reservation.assignments:
            if assignment.episode_ordinal != self.episode_ordinals[assignment.lane_id]:
                raise ValueError("assignment episode ordinal changed before commit")
        for assignment in reservation.assignments:
            stage_index = next(
                index
                for index, stage in enumerate(self.spec.stages)
                if stage.stage_id == assignment.stage_id
            )
            self.lane_stage[assignment.lane_id] = stage_index
            unlocked = self.unlock_updates[stage_index]
            self.lane_shaping_weight_ppm[assignment.lane_id] = self.spec.stages[
                stage_index
            ].reward_schedule.weight_ppm(self.completed_updates - unlocked)
            self.episode_ordinals[assignment.lane_id] += 1
        self._outstanding = None

    def action_masks(self, action_spec: ActionSpec) -> tuple[Tensor, Tensor]:
        kind = torch.zeros((self.lanes, 3), dtype=torch.bool)
        wait = torch.zeros(
            (self.lanes, len(action_spec.wait_choices)), dtype=torch.bool
        )
        for lane, stage_index in enumerate(self.lane_stage):
            stage = self.spec.stages[stage_index]
            kind[lane, list(stage.enabled_action_kinds)] = True
            for ticks in stage.enabled_wait_ticks:
                try:
                    wait[lane, action_spec.wait_choices.index(ticks)] = True
                except ValueError as exc:
                    raise ValueError(
                        "curriculum wait is absent from the action schema"
                    ) from exc
        return kind, wait

    def shaping_weights_ppm(self) -> Tensor:
        return torch.tensor(self.lane_shaping_weight_ppm, dtype=torch.int64)

    def activate_focus_for_new_episodes(self, reset_mask: Tensor) -> None:
        if reset_mask.shape != (self.lanes,) or reset_mask.dtype != torch.bool:
            raise ValueError("curriculum reset mask must be boolean [B]")
        for lane in torch.nonzero(reset_mask.cpu(), as_tuple=False).flatten().tolist():
            self.lane_stage[lane] = self.focus
            unlocked = self.unlock_updates[self.focus]
            if unlocked < 0:
                raise RuntimeError("cannot assign a locked curriculum stage")
            self.lane_shaping_weight_ppm[lane] = (
                self.current_stage.reward_schedule.weight_ppm(
                    self.completed_updates - unlocked
                )
            )

    def advance_update(self) -> None:
        if self._pending_validation is not None:
            raise RuntimeError("curriculum cannot train while validation is pending")
        if self.phase in {"budget_validation", "complete", "budget_exhausted"}:
            raise RuntimeError("closed curriculum cannot advance its update clock")
        self.completed_updates += 1
        stage = self.spec.stages[self.highest_unlocked]
        unlock = self.unlock_updates[self.highest_unlocked]
        if self.completed_updates - unlock >= stage.max_updates:
            # Training is closed, but the policy produced by the final allowed
            # update still receives exactly one bound gate evaluation.
            self.phase = "budget_validation"
            self._append_event("budget_closed", {"stage": stage.stage_id})

    def _append_event(self, kind: str, payload: Mapping[str, object]) -> None:
        event = {
            "previous": self._event_head,
            "kind": kind,
            "completed_updates": self.completed_updates,
            "payload": dict(payload),
        }
        self._event_head = _canonical_sha256(event)

    def request_validation(
        self, *, policy_sha256: str, evaluator_identity_sha256: str
    ) -> ValidationRequest:
        if self.phase in {"complete", "budget_exhausted"}:
            raise RuntimeError("terminal curriculum cannot request validation")
        if self._pending_validation is not None:
            raise RuntimeError("a validation request is already pending")
        if not _is_sha256(policy_sha256) or not _is_sha256(evaluator_identity_sha256):
            raise ValueError("validation policy/evaluator identity is malformed")
        if policy_sha256 in self._evaluated_policies:
            raise ValueError("one policy checkpoint cannot satisfy multiple gates")
        stages = self._expected_validation_stages()
        request = ValidationRequest(
            self.spec.sha256,
            self._evaluation_ordinal,
            self.completed_updates,
            policy_sha256,
            evaluator_identity_sha256,
            stages,
        )
        self._evaluation_ordinal += 1
        self._pending_validation = request
        return request

    def _expected_validation_stages(
        self, highest_unlocked: int | None = None
    ) -> tuple[ValidationStageRequest, ...]:
        highest = (
            self.highest_unlocked if highest_unlocked is None else highest_unlocked
        )
        stages = []
        for index, stage in enumerate(self.spec.stages[: highest + 1]):
            episodes = (
                stage.promotion_trials if index == highest else stage.regression_trials
            )
            stages.append(
                ValidationStageRequest(
                    stage.stage_id, stage.validation_snapshot_ids, episodes
                )
            )
        return tuple(stages)

    @staticmethod
    def _passes(result: ValidationResult, required: int, trials: int) -> bool:
        return (
            result.episodes >= trials
            and result.successes * trials >= required * result.episodes
        )

    def record_validation(self, report: ValidationReport) -> GateDecision:
        prior_hash = self._submitted_reports.get(report.request_id)
        if prior_hash is not None:
            if prior_hash != report.sha256:
                raise ValueError("conflicting validation report reuses a request id")
            return self._decision(False, None, "idempotent replay")
        if self.phase == "budget_exhausted":
            raise RuntimeError("budget-exhausted curriculum cannot accept a new gate")
        request = self._pending_validation
        if request is None or report.request_id != request.request_id:
            raise ValueError("validation report does not match a pending request")
        if (
            report.policy_sha256 != request.policy_sha256
            or report.evaluator_identity_sha256 != request.evaluator_identity_sha256
            or request.completed_update != self.completed_updates
        ):
            raise ValueError(
                "validation report policy/evaluator/update identity mismatch"
            )
        results = {result.stage_id: result for result in report.results}
        required_ids = {
            stage.stage_id for stage in self.spec.stages[: self.highest_unlocked + 1]
        }
        if set(results) != required_ids:
            raise ValueError(
                "validation report must cover every unlocked stage exactly"
            )
        requested = {stage.stage_id: stage for stage in request.stages}
        for stage_id in required_ids:
            result = results[stage_id]
            stage_request = requested[stage_id]
            expected_outcomes = tuple(
                (
                    stage_request.snapshot_ids[index % len(stage_request.snapshot_ids)],
                    index // len(stage_request.snapshot_ids),
                    request.episode_seed(
                        stage_id,
                        stage_request.snapshot_ids[
                            index % len(stage_request.snapshot_ids)
                        ],
                        index // len(stage_request.snapshot_ids),
                    ),
                )
                for index in range(stage_request.episodes)
            )
            actual_outcomes = tuple(
                (outcome.snapshot_id, outcome.repetition, outcome.policy_seed)
                for outcome in result.outcomes
            )
            if (
                result.episodes != stage_request.episodes
                or result.snapshot_ids != stage_request.snapshot_ids
                or actual_outcomes != expected_outcomes
            ):
                raise ValueError(
                    "validation report recipe outcomes or trial count differ from "
                    "its request"
                )
        final_budget_gate = self.phase == "budget_validation"
        self._submitted_reports[report.request_id] = report.sha256
        self._evaluated_policies.add(report.policy_sha256)
        self._pending_validation = None

        current = self.spec.stages[self.highest_unlocked]
        failing_prior: int | None = None
        for index in range(self.highest_unlocked):
            stage = self.spec.stages[index]
            if not self._passes(
                results[stage.stage_id],
                stage.regression_successes,
                stage.regression_trials,
            ):
                failing_prior = index
                break
        if failing_prior is not None:
            if final_budget_gate:
                self.phase = "budget_exhausted"
                self.promotion_streak = 0
                self._append_event(
                    "budget_exhausted",
                    {"stage": current.stage_id, "report": report.sha256},
                )
                return self._decision(
                    False, None, "final-budget prior-stage regression floor failed"
                )
            self.phase = "remediation"
            self.focus = failing_prior
            self.promotion_streak = 0
            self._append_event(
                "remediation",
                {
                    "stage": self.spec.stages[failing_prior].stage_id,
                    "report": report.sha256,
                },
            )
            return self._decision(
                False,
                self.spec.stages[failing_prior].stage_id,
                "prior-stage regression floor failed",
            )

        if self.phase == "remediation":
            self.phase = "normal"
            self.focus = self.highest_unlocked
            self._append_event("remediation_cleared", {"report": report.sha256})
        elif final_budget_gate and self.focus != self.highest_unlocked:
            # Budget closure replaces the remediation phase label, but a
            # passing all-stage report still clears its stale focus.
            self.focus = self.highest_unlocked
            self._append_event("remediation_cleared", {"report": report.sha256})

        passed = self._passes(
            results[current.stage_id],
            current.promotion_successes,
            current.promotion_trials,
        )
        self.promotion_streak = self.promotion_streak + 1 if passed else 0
        if self.promotion_streak < current.required_consecutive_passes:
            if final_budget_gate:
                self.phase = "budget_exhausted"
                self._append_event(
                    "budget_exhausted",
                    {"stage": current.stage_id, "report": report.sha256},
                )
            return self._decision(False, None, "promotion evidence is insufficient")

        self.promotion_streak = 0
        if self.highest_unlocked + 1 == len(self.spec.stages):
            schedule_weight = current.reward_schedule.weight_ppm(
                self.completed_updates - self.unlock_updates[self.highest_unlocked]
            )
            if schedule_weight != 0 or any(
                weight != 0 for weight in self.lane_shaping_weight_ppm
            ):
                if final_budget_gate:
                    self.phase = "budget_exhausted"
                    self._append_event(
                        "budget_exhausted",
                        {"stage": current.stage_id, "report": report.sha256},
                    )
                return self._decision(
                    False,
                    None,
                    "final performance gate passed but shaping is not exactly zero",
                )
            self.phase = "complete"
            self._append_event(
                "complete", {"stage": current.stage_id, "report": report.sha256}
            )
            return self._decision(False, None, "final stage passed")
        self.highest_unlocked += 1
        self.focus = self.highest_unlocked
        self.phase = "normal"
        self.unlock_updates[self.highest_unlocked] = self.completed_updates
        promoted = self.spec.stages[self.highest_unlocked]
        self._append_event(
            "promote", {"stage": promoted.stage_id, "report": report.sha256}
        )
        return self._decision(True, None, "promotion gate passed")

    def _decision(
        self, promoted: bool, remediation_stage: str | None, reason: str
    ) -> GateDecision:
        return GateDecision(
            self.phase,
            self.spec.stages[self.highest_unlocked].stage_id,
            self.spec.stages[self.focus].stage_id,
            promoted,
            remediation_stage,
            reason,
        )

    def state_dict(self) -> dict[str, object]:
        if self._outstanding is not None:
            raise RuntimeError("checkpoint cannot capture an uncommitted assignment")
        core = {
            "version": self.version,
            "spec_sha256": self.spec.sha256,
            "lanes": self.lanes,
            "learner_seed": self.learner_seed,
            "highest_unlocked": self.highest_unlocked,
            "focus": self.focus,
            "phase": self.phase,
            "completed_updates": self.completed_updates,
            "unlock_updates": list(self.unlock_updates),
            "promotion_streak": self.promotion_streak,
            "lane_stage": list(self.lane_stage),
            "lane_shaping_weight_ppm": list(self.lane_shaping_weight_ppm),
            "episode_ordinals": list(self.episode_ordinals),
            "submitted_reports": dict(self._submitted_reports),
            "evaluated_policies": sorted(self._evaluated_policies),
            "evaluation_ordinal": self._evaluation_ordinal,
            "pending_validation": (
                None
                if self._pending_validation is None
                else self._pending_validation.manifest()
            ),
            "event_head": self._event_head,
        }
        return {**core, "state_sha256": _canonical_sha256(core)}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "version",
            "spec_sha256",
            "lanes",
            "learner_seed",
            "highest_unlocked",
            "focus",
            "phase",
            "completed_updates",
            "unlock_updates",
            "promotion_streak",
            "lane_stage",
            "lane_shaping_weight_ppm",
            "episode_ordinals",
            "submitted_reports",
            "evaluated_policies",
            "evaluation_ordinal",
            "pending_validation",
            "event_head",
            "state_sha256",
        }
        if set(state) != expected:
            raise ValueError("curriculum checkpoint keys do not match the version")
        core = {key: state[key] for key in expected if key != "state_sha256"}
        if state["state_sha256"] != _canonical_sha256(core):
            raise ValueError("curriculum checkpoint state hash mismatch")
        if (
            state["version"] != self.version
            or state["spec_sha256"] != self.spec.sha256
            or state["lanes"] != self.lanes
            or state["learner_seed"] != self.learner_seed
        ):
            raise ValueError("curriculum checkpoint identity mismatch")
        highest = state["highest_unlocked"]
        focus = state["focus"]
        completed = state["completed_updates"]
        streak = state["promotion_streak"]
        unlocks = state["unlock_updates"]
        lane_stage = state["lane_stage"]
        lane_weights = state["lane_shaping_weight_ppm"]
        ordinals = state["episode_ordinals"]
        reports = state["submitted_reports"]
        evaluated_policies = state["evaluated_policies"]
        evaluation_ordinal = state["evaluation_ordinal"]
        pending_value = state["pending_validation"]
        if (
            any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in (highest, focus, completed, streak)
            )
            or not 0 <= highest < len(self.spec.stages)
            or not 0 <= focus <= highest
            or completed < 0
            or streak < 0
            or state["phase"] not in _PHASES
            or not isinstance(unlocks, list)
            or len(unlocks) != len(self.spec.stages)
            or not isinstance(lane_stage, list)
            or len(lane_stage) != self.lanes
            or not isinstance(lane_weights, list)
            or len(lane_weights) != self.lanes
            or not isinstance(ordinals, list)
            or len(ordinals) != self.lanes
            or any(not isinstance(value, int) or value < 0 for value in ordinals)
            or any(
                not isinstance(value, int) or not 0 <= value <= highest
                for value in lane_stage
            )
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 1_000_000
                for value in lane_weights
            )
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in unlocks
            )
            or not isinstance(reports, dict)
            or any(
                not isinstance(key, str) or not _is_sha256(value)
                for key, value in reports.items()
            )
            or not _is_sha256(state["event_head"])
            or not isinstance(evaluated_policies, list)
            or len(set(evaluated_policies)) != len(evaluated_policies)
            or any(not _is_sha256(value) for value in evaluated_policies)
            or isinstance(evaluation_ordinal, bool)
            or not isinstance(evaluation_ordinal, int)
            or evaluation_ordinal < 0
        ):
            raise ValueError("curriculum checkpoint state is malformed")
        pending = (
            None
            if pending_value is None
            else ValidationRequest.from_manifest(pending_value)
            if isinstance(pending_value, Mapping)
            else None
        )
        if pending_value is not None and pending is None:
            raise ValueError("pending validation request is malformed")
        if pending is not None and (
            pending.curriculum_sha256 != self.spec.sha256
            or pending.gate_ordinal + 1 != evaluation_ordinal
            or pending.completed_update != completed
            or pending.policy_sha256 in evaluated_policies
            or pending.stages != self._expected_validation_stages(int(highest))
        ):
            raise ValueError(
                "pending validation request disagrees with coordinator state"
            )
        if (
            unlocks[0] != 0
            or any(not 0 <= value <= completed for value in unlocks[: highest + 1])
            or any(value != -1 for value in unlocks[highest + 1 :])
        ):
            raise ValueError("curriculum unlock schedule is malformed")
        stage_elapsed = completed - unlocks[highest]
        stage_budget = self.spec.stages[highest].max_updates
        phase = state["phase"]
        if (
            stage_elapsed < 0
            or (phase in {"normal", "remediation"} and stage_elapsed >= stage_budget)
            or (
                phase in {"budget_validation", "budget_exhausted"}
                and stage_elapsed != stage_budget
            )
            or (phase in {"complete", "budget_exhausted"} and pending is not None)
            or (phase == "complete" and highest + 1 != len(self.spec.stages))
            or (phase == "complete" and focus != highest)
            or (phase == "normal" and focus != highest)
        ):
            raise ValueError("curriculum checkpoint phase semantics are malformed")
        prepared_unlocks = list(unlocks)
        prepared_lane_stage = list(lane_stage)
        prepared_lane_weights = list(lane_weights)
        prepared_ordinals = list(ordinals)
        prepared_reports = dict(reports)
        prepared_policies = set(evaluated_policies)
        self.highest_unlocked = int(highest)
        self.focus = int(focus)
        self.phase = str(state["phase"])
        self.completed_updates = int(completed)
        self.unlock_updates = prepared_unlocks
        self.promotion_streak = int(streak)
        self.lane_stage = prepared_lane_stage
        self.lane_shaping_weight_ppm = prepared_lane_weights
        self.episode_ordinals = prepared_ordinals
        self._submitted_reports = prepared_reports
        self._evaluated_policies = prepared_policies
        self._evaluation_ordinal = evaluation_ordinal
        self._pending_validation = pending
        self._event_head = str(state["event_head"])
        self._outstanding = None
