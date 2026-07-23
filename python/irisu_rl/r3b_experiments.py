"""Immutable R3b experiment design and fail-closed acceptance logic.

This module deliberately does not run training.  It binds an experiment runner to
the checked-in plan, makes failed arms first-class records, and keeps model
selection separate from the one-shot sealed-test decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import secrets
import sqlite3
import stat
import statistics
import tomllib
from contextlib import closing
from dataclasses import InitVar, asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence


_RESULT_STATUSES = frozenset(
    {"complete", "eliminated", "pre_start_failure", "post_start_failure"}
)
_PHASES = frozenset({"calibration", "validation", "test"})
_EXACT_RESUME_VERIFICATION_TOKEN = object()
_COMMITTED_BASELINE_EVIDENCE_TOKEN = object()
_SEALED_AUTHORIZATION_KIND = "irisu.r3b.sealed-test-run-authorization"
_SEALED_BASELINE_EVIDENCE_KIND = "irisu.r3b.sealed-baseline-evidence"
_SEALED_OUTCOME_REFERENCE_KIND = "irisu.r3b.sealed-learner-outcome-reference"
_SEALED_CONFIRMATION_KIND = "irisu.r3b.sealed-confirmation-report"
_CONFIRMATION_GATE_ORDER = (
    "complete_exact_test_results",
    "engineering_audits",
    "required_baselines",
    "relative_auc_gain_lcb",
    "final_mean_retention_lcb",
    "p10_noninferiority_lcb",
    "trivial_baseline_margin_lcb",
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_nonzero_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value != "0" * 64
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _bearer_sha256(token: str) -> str:
    if not _is_nonzero_sha256(token):
        raise ValueError("bearer token must be a nonzero lowercase SHA-256 value")
    return hashlib.sha256(token.encode()).hexdigest()


def _assert_private_ledger_path(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError("sealed-test ledger is missing") from exc
    if (
        not stat.S_ISREG(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or stat.S_IMODE(metadata.st_mode) != 0o600
        or metadata.st_nlink != 1
    ):
        raise RuntimeError(
            "sealed-test ledger must be an owned, private, singly-linked regular file"
        )


def _connect_private_ledger(path: str | Path) -> sqlite3.Connection:
    ledger = Path(path)
    _assert_private_ledger_path(ledger)
    before = ledger.stat()
    connection = sqlite3.connect(ledger, timeout=30.0)
    after = ledger.stat()
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        connection.close()
        raise RuntimeError("sealed-test ledger changed while opening")
    connection.execute("PRAGMA foreign_keys=ON")
    check = connection.execute("PRAGMA quick_check").fetchone()
    if check != ("ok",):
        connection.close()
        raise RuntimeError("sealed-test ledger integrity check failed")
    return connection


def _require_keys(
    value: object, expected: set[str], *, location: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{location} must be a string-keyed table")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{location} keys differ: missing={missing}, extra={extra}")
    return value


def _manifest_list(value: object, *, location: str) -> list[object]:
    if type(value) is not list:
        raise ValueError(f"{location} must be an array")
    return value


def _require_manifest_round_trip(
    original: Mapping[str, object],
    reconstructed: Mapping[str, object],
    *,
    location: str,
) -> None:
    try:
        left = json.dumps(
            original, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        right = json.dumps(
            reconstructed, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{location} must be finite JSON") from exc
    if left != right:
        raise ValueError(f"{location} is not a canonical serialization")


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _string_tuple(value: object, *, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"{name} must be an array of nonempty strings")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} entries must be unique")
    return result


def _seed_tuple(value: object, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(
        isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
        for seed in value
    ):
        raise ValueError(f"{name} must contain uint64 learner seeds")
    result = tuple(value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} seeds must be unique")
    return result


@dataclass(frozen=True, slots=True)
class CandidateArm:
    alpha_weight_ppm: int
    learning_rate: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.alpha_weight_ppm, bool)
            or not isinstance(self.alpha_weight_ppm, int)
            or not 0 <= self.alpha_weight_ppm <= 1_000_000
        ):
            raise ValueError("alpha_weight_ppm must be an integer in [0, 1000000]")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(float(self.learning_rate))
            or self.learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive")
        object.__setattr__(self, "learning_rate", float(self.learning_rate))

    @property
    def arm_id(self) -> str:
        return f"alpha-{self.alpha_weight_ppm:07d}-lr-{self.learning_rate:.12g}"

    def manifest(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "alpha_weight_ppm": self.alpha_weight_ppm,
            "learning_rate": self.learning_rate,
        }

    @classmethod
    def from_manifest(cls, value: object) -> CandidateArm:
        manifest = _require_keys(
            value,
            {"arm_id", "alpha_weight_ppm", "learning_rate"},
            location="candidate arm",
        )
        if (
            type(manifest["arm_id"]) is not str
            or type(manifest["alpha_weight_ppm"]) is not int
            or type(manifest["learning_rate"]) is not float
        ):
            raise ValueError("candidate arm field types are malformed")
        try:
            result = cls(
                manifest["alpha_weight_ppm"],  # type: ignore[arg-type]
                manifest["learning_rate"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("candidate arm is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="candidate arm"
        )
        return result


@dataclass(frozen=True, slots=True)
class TrialSeedPlan:
    """Domain-separated RNG identities shared by every arm for one learner seed."""

    learner_seed: int
    model_initialization: int
    policy_sampling: int
    ppo_minibatching: int
    assignment: int
    session_numpy: int

    @classmethod
    def derive(cls, experiment_sha256: str, learner_seed: int) -> TrialSeedPlan:
        if (
            not isinstance(experiment_sha256, str)
            or len(experiment_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in experiment_sha256
            )
            or isinstance(learner_seed, bool)
            or not isinstance(learner_seed, int)
            or not 0 <= learner_seed < 2**64
        ):
            raise ValueError(
                "seed derivation requires a SHA-256 and uint64 learner seed"
            )

        def derive(domain: str) -> int:
            payload = (
                f"irisu-r3b-seed-v1:{experiment_sha256}:{learner_seed}:{domain}"
            ).encode()
            return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

        return cls(
            learner_seed,
            derive("model-initialization"),
            derive("policy-sampling"),
            derive("ppo-minibatching"),
            derive("assignment"),
            derive("session-numpy"),
        )

    def manifest(self) -> dict[str, int | str]:
        return {
            "version": "r3b-trial-seed-plan-v2",
            "learner_seed": self.learner_seed,
            "model_initialization": self.model_initialization,
            "policy_sampling": self.policy_sampling,
            "ppo_minibatching": self.ppo_minibatching,
            "assignment": self.assignment,
            "session_numpy": self.session_numpy,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class R3BExperimentPlan:
    version: str
    status: str
    alpha_weight_ppm: tuple[int, ...]
    learning_rates: tuple[float, ...]
    tie_break: tuple[str, ...]
    total_updates: int
    shaped_updates: int
    zero_tail_updates: int
    ticks_per_update: int
    checkpoint_interval_updates: int
    alpha_lifetime: str
    tail_transition: str
    optimizer_updates_while_draining: bool
    optimizer_state_reset_at_tail: bool
    optimizer_schedule: str
    final_learning_rate_fraction: float
    zero_control: str
    calibration_learner_seeds: tuple[int, ...]
    validation_learner_seeds: tuple[int, ...]
    test_learner_seeds: tuple[int, ...]
    calibration_evaluation_seed: int
    validation_evaluation_seed: int
    test_evaluation_seed: int
    calibration_budgets_updates: tuple[int, ...]
    calibration_elimination_metric: str
    validation_updates: int
    validation_episodes_per_policy: int
    validation_selection_data: str
    test_updates: int
    test_episodes_per_policy: int
    test_sealed: bool
    test_reuse_after_rejection: bool
    auc_definition: str
    maximum_cumulative_tick_overshoot_fraction: float
    bootstrap_unit: str
    bootstrap_samples: int
    bootstrap_seed: int
    confidence_level: float
    relative_score_denominator_floor: float
    minimum_relative_auc_gain: float
    minimum_final_mean_retention: float
    minimum_p10_retention: float
    minimum_p10_absolute_delta_when_control_near_zero: float
    minimum_trivial_baseline_margin: float
    required_baselines: tuple[str, ...]
    optional_baselines: tuple[str, ...]
    minimum_baseline_episodes: int
    retain_every_arm_record: bool
    missing_arm_behavior: str
    missing_seed_behavior: str
    pre_start_failure_behavior: str
    post_start_failure_behavior: str
    test_failure_behavior: str

    def __post_init__(self) -> None:
        tuple_fields = (
            self.alpha_weight_ppm,
            self.learning_rates,
            self.tie_break,
            self.calibration_learner_seeds,
            self.validation_learner_seeds,
            self.test_learner_seeds,
            self.calibration_budgets_updates,
            self.required_baselines,
            self.optional_baselines,
        )
        if any(not isinstance(value, tuple) for value in tuple_fields):
            raise ValueError("R3b plan sequence fields must be immutable tuples")
        if self.version != "r3b-completion-v1":
            raise ValueError("unsupported R3b experiment version")
        if self.status != "design_only_no_empirical_results":
            raise ValueError("R3b plan must not claim empirical results")
        if self.alpha_weight_ppm != (0, 100_000, 250_000, 500_000) or any(
            isinstance(alpha, bool)
            or not isinstance(alpha, int)
            or not 0 <= alpha <= 1_000_000
            for alpha in self.alpha_weight_ppm
        ):
            raise ValueError("R3b alpha grid differs from the frozen design")
        if self.learning_rates != (3e-5, 1e-4, 3e-4) or any(
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(float(rate))
            or rate <= 0
            for rate in self.learning_rates
        ):
            raise ValueError("R3b learning-rate grid differs from the frozen design")
        expected_tie_break = (
            "higher_median_auc",
            "higher_final_mean_raw_score",
            "lower_learning_rate",
        )
        if self.tie_break != expected_tie_break:
            raise ValueError("calibration tie-break policy is not canonical")
        integer_fields = {
            "total_updates": self.total_updates,
            "shaped_updates": self.shaped_updates,
            "zero_tail_updates": self.zero_tail_updates,
            "ticks_per_update": self.ticks_per_update,
            "checkpoint_interval_updates": self.checkpoint_interval_updates,
            "validation_updates": self.validation_updates,
            "validation_episodes_per_policy": self.validation_episodes_per_policy,
            "test_updates": self.test_updates,
            "test_episodes_per_policy": self.test_episodes_per_policy,
            "bootstrap_samples": self.bootstrap_samples,
            "minimum_baseline_episodes": self.minimum_baseline_episodes,
        }
        for name, value in integer_fields.items():
            _positive_int(value, name=name)
        if self.total_updates != self.shaped_updates + self.zero_tail_updates:
            raise ValueError("total updates must equal shaped updates plus zero tail")
        if self.zero_tail_updates < 400:
            raise ValueError("R3b requires at least 400 pure score-only tail updates")
        if (
            self.total_updates != 1000
            or self.shaped_updates != 600
            or self.zero_tail_updates != 400
            or self.ticks_per_update != 2048
            or self.checkpoint_interval_updates != 50
            or self.calibration_budgets_updates != (100, 300)
        ):
            raise ValueError("training schedule differs from the frozen R3b design")
        if (
            self.validation_updates != self.total_updates
            or self.test_updates != self.total_updates
        ):
            raise ValueError("validation and test must evaluate full-budget runs")
        if (
            self.alpha_lifetime != "one_complete_episode"
            or self.tail_transition != "drain_nonzero_episodes_before_zero_tail"
            or not isinstance(self.optimizer_updates_while_draining, bool)
            or not isinstance(self.optimizer_state_reset_at_tail, bool)
            or self.optimizer_updates_while_draining
            or self.optimizer_state_reset_at_tail
            or self.optimizer_schedule != "linear_decay"
            or self.final_learning_rate_fraction != 0.1
            or self.zero_control
            != "same_conditioned_architecture_composer_and_schedule"
        ):
            raise ValueError("score-only tail and drain semantics are not canonical")
        if (
            not self.calibration_budgets_updates
            or tuple(sorted(set(self.calibration_budgets_updates)))
            != self.calibration_budgets_updates
            or self.calibration_budgets_updates[-1] >= self.total_updates
            or any(
                _positive_int(value, name="calibration budget")
                % self.checkpoint_interval_updates
                for value in self.calibration_budgets_updates
            )
            or self.total_updates % self.checkpoint_interval_updates
        ):
            raise ValueError("experiment budgets must lie on the checkpoint grid")
        if (
            len(self.calibration_learner_seeds) < 3
            or len(self.validation_learner_seeds) < 8
            or len(self.test_learner_seeds) < 12
        ):
            raise ValueError("learner-seed phases do not meet minimum replication")
        all_seeds = (
            self.calibration_learner_seeds
            + self.validation_learner_seeds
            + self.test_learner_seeds
        )
        if len(all_seeds) != len(set(all_seeds)):
            raise ValueError(
                "calibration, validation, and test learner seeds must be disjoint"
            )
        if any(
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**64
            for seed in all_seeds
        ):
            raise ValueError("learner seeds must be uint64")
        evaluation_seeds = (
            self.calibration_evaluation_seed,
            self.validation_evaluation_seed,
            self.test_evaluation_seed,
        )
        if (
            len(set(evaluation_seeds)) != 3
            or set(evaluation_seeds) & set(all_seeds)
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed < 2**64
                for seed in evaluation_seeds
            )
        ):
            raise ValueError(
                "phase evaluation seeds must be distinct uint64 values "
                "disjoint from learner seeds"
            )
        if (
            self.calibration_elimination_metric
            != "final_rung_paired_tick_aligned_raw_score_auc_no_early_elimination"
            or self.auc_definition
            != "linear_interpolation_to_target_tick_grid_normalized_by_horizon"
            or self.maximum_cumulative_tick_overshoot_fraction != 0.01
            or self.bootstrap_unit != "paired_learner_seed"
            or self.validation_selection_data != "fresh_unsealed_validation"
            or not isinstance(self.test_sealed, bool)
            or not isinstance(self.test_reuse_after_rejection, bool)
            or not self.test_sealed
            or self.test_reuse_after_rejection
        ):
            raise ValueError("selection or sealed-test protocol is not canonical")
        if (
            self.validation_episodes_per_policy < 512
            or self.test_episodes_per_policy < 512
            or self.minimum_baseline_episodes < 512
            or self.bootstrap_samples < 4096
        ):
            raise ValueError("R3b evidence or resampling budget is too small")
        if (
            isinstance(self.bootstrap_seed, bool)
            or not isinstance(self.bootstrap_seed, int)
            or not 0 <= self.bootstrap_seed < 2**64
        ):
            raise ValueError("bootstrap seed must be uint64")
        numeric_fields = {
            "confidence_level": self.confidence_level,
            "final_learning_rate_fraction": self.final_learning_rate_fraction,
            "relative_score_denominator_floor": self.relative_score_denominator_floor,
            "maximum_cumulative_tick_overshoot_fraction": self.maximum_cumulative_tick_overshoot_fraction,
            "minimum_relative_auc_gain": self.minimum_relative_auc_gain,
            "minimum_final_mean_retention": self.minimum_final_mean_retention,
            "minimum_p10_retention": self.minimum_p10_retention,
            "minimum_p10_absolute_delta_when_control_near_zero": self.minimum_p10_absolute_delta_when_control_near_zero,
            "minimum_trivial_baseline_margin": self.minimum_trivial_baseline_margin,
        }
        for name, value in numeric_fields.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence level must lie in (0, 1)")
        if self.relative_score_denominator_floor <= 0:
            raise ValueError("relative-score denominator floor must be positive")
        if (
            self.confidence_level != 0.95
            or self.minimum_relative_auc_gain != 0.05
            or self.minimum_final_mean_retention != 0.95
            or self.minimum_p10_retention != 0.90
            or self.minimum_p10_absolute_delta_when_control_near_zero != 0.0
            or self.minimum_trivial_baseline_margin != 0.0
        ):
            raise ValueError("acceptance gates differ from the frozen R3b design")
        if set(self.required_baselines) & set(self.optional_baselines):
            raise ValueError("required and optional baselines must be disjoint")
        all_baselines = self.required_baselines + self.optional_baselines
        if (
            self.required_baselines
            != (
                "no_action_long_wait",
                "seeded_legal_random",
                "matcher_shot_policy",
                "scripted_direct_matcher",
                "scripted_side_ejector",
                "scripted_imminent_rot_hazard",
            )
            or self.optional_baselines != ("one_step_greedy",)
            or len(all_baselines) != len(set(all_baselines))
            or any(
                not isinstance(baseline, str) or not baseline
                for baseline in all_baselines
            )
        ):
            raise ValueError("baseline set differs from the frozen R3b design")
        if (
            not isinstance(self.retain_every_arm_record, bool)
            or not self.retain_every_arm_record
            or self.missing_arm_behavior != "reject_phase"
            or self.missing_seed_behavior != "reject_arm"
            or self.pre_start_failure_behavior != "record_and_rank_ineligible"
            or self.post_start_failure_behavior != "record_and_rank_ineligible"
            or self.test_failure_behavior
            != "reject_candidate_without_testing_runner_up"
        ):
            raise ValueError("failure handling must be fail-closed and retain all arms")

    @property
    def arms(self) -> tuple[CandidateArm, ...]:
        return tuple(
            CandidateArm(alpha, rate)
            for alpha in self.alpha_weight_ppm
            for rate in self.learning_rates
        )

    def tick_grid(self, updates: int) -> tuple[int, ...]:
        _positive_int(updates, name="updates")
        if updates % self.checkpoint_interval_updates:
            raise ValueError("updates must end on a checkpoint boundary")
        return tuple(
            update * self.ticks_per_update
            for update in range(0, updates + 1, self.checkpoint_interval_updates)
        )

    def evaluation_seed(self, phase: str) -> int:
        try:
            return {
                "calibration": self.calibration_evaluation_seed,
                "validation": self.validation_evaluation_seed,
                "test": self.test_evaluation_seed,
            }[phase]
        except KeyError as exc:
            raise ValueError("unknown evaluation phase") from exc

    def trial_jobs(
        self,
        phase: str,
        arms: (
            Sequence[CandidateArm]
            | ValidationRunAuthorization
            | SealedTestRunAuthorization
        ) = (),
    ) -> tuple[TrialJob, ...]:
        """Enumerate the exact paired jobs allowed to enter one phase."""

        supplied = (
            ()
            if isinstance(
                arms,
                (
                    CalibrationSelectionAuthorization,
                    ValidationRunAuthorization,
                    SealedTestRunAuthorization,
                ),
            )
            else tuple(arms)
        )
        if phase == "calibration":
            if supplied or isinstance(
                arms,
                (
                    CalibrationSelectionAuthorization,
                    ValidationRunAuthorization,
                    SealedTestRunAuthorization,
                ),
            ):
                raise ValueError("calibration arms come only from the frozen grid")
            selected = self.arms
            seeds = self.calibration_learner_seeds
            budgets = self.calibration_budgets_updates
            sealed = False
        elif phase == "validation":
            if (
                not isinstance(arms, ValidationRunAuthorization)
                or arms.plan.sha256 != self.sha256
            ):
                raise ValueError("validation jobs require calibrated-arm authorization")
            selected = arms.authorization.arms
            seeds = self.validation_learner_seeds
            budgets = (self.validation_updates,)
            sealed = False
        elif phase == "test":
            if (
                not isinstance(arms, SealedTestRunAuthorization)
                or arms.plan.sha256 != self.sha256
            ):
                raise ValueError("test jobs require a validation-bound authorization")
            arms.assert_authorized()
            indexed = {arm.arm_id: arm for arm in self.arms}
            try:
                selected = (
                    indexed[arms.authorization.control_arm_id],
                    indexed[arms.authorization.candidate_arm_id],
                )
            except KeyError as exc:
                raise ValueError("test authorization names an unplanned arm") from exc
            seeds = self.test_learner_seeds
            budgets = (self.test_updates,)
            sealed = True
        else:
            raise ValueError("unknown R3b experiment phase")
        return tuple(
            TrialJob(
                self.sha256,
                phase,
                arm,
                seed,
                budget,
                sealed,
                TrialSeedPlan.derive(self.sha256, seed).sha256,
                (
                    arms.sha256
                    if isinstance(
                        arms,
                        (
                            ValidationRunAuthorization,
                            SealedTestRunAuthorization,
                        ),
                    )
                    else None
                ),
            )
            for arm in selected
            for seed in seeds
            for budget in budgets
        )

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "status": self.status,
            "grid": {
                "alpha_weight_ppm": list(self.alpha_weight_ppm),
                "learning_rates": list(self.learning_rates),
                "tie_break": list(self.tie_break),
            },
            "schedule": {
                "total_updates": self.total_updates,
                "shaped_updates": self.shaped_updates,
                "zero_tail_updates": self.zero_tail_updates,
                "ticks_per_update": self.ticks_per_update,
                "checkpoint_interval_updates": self.checkpoint_interval_updates,
                "alpha_lifetime": self.alpha_lifetime,
                "tail_transition": self.tail_transition,
                "optimizer_updates_while_draining": self.optimizer_updates_while_draining,
                "optimizer_state_reset_at_tail": self.optimizer_state_reset_at_tail,
                "optimizer_schedule": self.optimizer_schedule,
                "final_learning_rate_fraction": self.final_learning_rate_fraction,
                "zero_control": self.zero_control,
            },
            "seeds": {
                "calibration_learner": list(self.calibration_learner_seeds),
                "validation_learner": list(self.validation_learner_seeds),
                "test_learner": list(self.test_learner_seeds),
                "calibration_evaluation": self.calibration_evaluation_seed,
                "validation_evaluation": self.validation_evaluation_seed,
                "test_evaluation": self.test_evaluation_seed,
            },
            "calibration": {
                "budgets_updates": list(self.calibration_budgets_updates),
                "elimination_metric": self.calibration_elimination_metric,
            },
            "validation": {
                "updates": self.validation_updates,
                "episodes_per_policy": self.validation_episodes_per_policy,
                "selection_data": self.validation_selection_data,
            },
            "test": {
                "updates": self.test_updates,
                "episodes_per_policy": self.test_episodes_per_policy,
                "sealed_until_one_candidate_is_selected": self.test_sealed,
                "reuse_after_rejection": self.test_reuse_after_rejection,
            },
            "statistics": {
                "auc": self.auc_definition,
                "maximum_cumulative_tick_overshoot_fraction": self.maximum_cumulative_tick_overshoot_fraction,
                "bootstrap_unit": self.bootstrap_unit,
                "bootstrap_samples": self.bootstrap_samples,
                "bootstrap_seed": self.bootstrap_seed,
                "confidence_level": self.confidence_level,
                "relative_score_denominator_floor": self.relative_score_denominator_floor,
            },
            "gates": {
                "minimum_relative_auc_gain": self.minimum_relative_auc_gain,
                "minimum_final_mean_retention": self.minimum_final_mean_retention,
                "minimum_p10_retention": self.minimum_p10_retention,
                "minimum_p10_absolute_delta_when_control_near_zero": self.minimum_p10_absolute_delta_when_control_near_zero,
                "minimum_trivial_baseline_margin": self.minimum_trivial_baseline_margin,
            },
            "baselines": {
                "required": list(self.required_baselines),
                "optional": list(self.optional_baselines),
                "minimum_episodes_per_policy": self.minimum_baseline_episodes,
            },
            "failures": {
                "retain_every_arm_record": self.retain_every_arm_record,
                "missing_arm": self.missing_arm_behavior,
                "missing_seed": self.missing_seed_behavior,
                "pre_start_failure": self.pre_start_failure_behavior,
                "post_start_failure": self.post_start_failure_behavior,
                "test_failure": self.test_failure_behavior,
            },
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    @classmethod
    def from_manifest(cls, value: object) -> R3BExperimentPlan:
        result = cls.from_mapping(value)
        manifest = _require_keys(
            value,
            set(result.manifest()),
            location="R3b experiment plan",
        )
        _require_manifest_round_trip(
            manifest, result.manifest(), location="R3b experiment plan"
        )
        return result

    @classmethod
    def from_mapping(cls, value: object) -> R3BExperimentPlan:
        root = _require_keys(
            value,
            {
                "version",
                "status",
                "grid",
                "schedule",
                "seeds",
                "calibration",
                "validation",
                "test",
                "statistics",
                "gates",
                "baselines",
                "failures",
            },
            location="root",
        )
        grid = _require_keys(
            root["grid"],
            {"alpha_weight_ppm", "learning_rates", "tie_break"},
            location="grid",
        )
        schedule = _require_keys(
            root["schedule"],
            {
                "total_updates",
                "shaped_updates",
                "zero_tail_updates",
                "ticks_per_update",
                "checkpoint_interval_updates",
                "alpha_lifetime",
                "tail_transition",
                "optimizer_updates_while_draining",
                "optimizer_state_reset_at_tail",
                "optimizer_schedule",
                "final_learning_rate_fraction",
                "zero_control",
            },
            location="schedule",
        )
        seeds = _require_keys(
            root["seeds"],
            {
                "calibration_learner",
                "validation_learner",
                "test_learner",
                "calibration_evaluation",
                "validation_evaluation",
                "test_evaluation",
            },
            location="seeds",
        )
        calibration = _require_keys(
            root["calibration"],
            {"budgets_updates", "elimination_metric"},
            location="calibration",
        )
        validation = _require_keys(
            root["validation"],
            {"updates", "episodes_per_policy", "selection_data"},
            location="validation",
        )
        test = _require_keys(
            root["test"],
            {
                "updates",
                "episodes_per_policy",
                "sealed_until_one_candidate_is_selected",
                "reuse_after_rejection",
            },
            location="test",
        )
        statistics_table = _require_keys(
            root["statistics"],
            {
                "auc",
                "maximum_cumulative_tick_overshoot_fraction",
                "bootstrap_unit",
                "bootstrap_samples",
                "bootstrap_seed",
                "confidence_level",
                "relative_score_denominator_floor",
            },
            location="statistics",
        )
        gates = _require_keys(
            root["gates"],
            {
                "minimum_relative_auc_gain",
                "minimum_final_mean_retention",
                "minimum_p10_retention",
                "minimum_p10_absolute_delta_when_control_near_zero",
                "minimum_trivial_baseline_margin",
            },
            location="gates",
        )
        baselines = _require_keys(
            root["baselines"],
            {"required", "optional", "minimum_episodes_per_policy"},
            location="baselines",
        )
        failures = _require_keys(
            root["failures"],
            {
                "retain_every_arm_record",
                "missing_arm",
                "missing_seed",
                "pre_start_failure",
                "post_start_failure",
                "test_failure",
            },
            location="failures",
        )

        def integer_tuple(source: object, *, name: str) -> tuple[int, ...]:
            if not isinstance(source, list) or any(
                isinstance(item, bool) or not isinstance(item, int) for item in source
            ):
                raise ValueError(f"{name} must be an integer array")
            return tuple(source)

        def float_tuple(source: object, *, name: str) -> tuple[float, ...]:
            if not isinstance(source, list):
                raise ValueError(f"{name} must be a numeric array")
            return tuple(_finite_float(item, name=name) for item in source)

        return cls(
            version=_nonempty_string(root["version"], name="version"),
            status=_nonempty_string(root["status"], name="status"),
            alpha_weight_ppm=integer_tuple(
                grid["alpha_weight_ppm"], name="alpha weights"
            ),
            learning_rates=float_tuple(grid["learning_rates"], name="learning rates"),
            tie_break=_string_tuple(grid["tie_break"], name="tie break"),
            total_updates=_positive_int(
                schedule["total_updates"], name="total updates"
            ),
            shaped_updates=_positive_int(
                schedule["shaped_updates"], name="shaped updates"
            ),
            zero_tail_updates=_positive_int(
                schedule["zero_tail_updates"], name="zero tail updates"
            ),
            ticks_per_update=_positive_int(
                schedule["ticks_per_update"], name="ticks per update"
            ),
            checkpoint_interval_updates=_positive_int(
                schedule["checkpoint_interval_updates"], name="checkpoint interval"
            ),
            alpha_lifetime=_nonempty_string(
                schedule["alpha_lifetime"], name="alpha lifetime"
            ),
            tail_transition=_nonempty_string(
                schedule["tail_transition"], name="tail transition"
            ),
            optimizer_updates_while_draining=_boolean(
                schedule["optimizer_updates_while_draining"],
                name="optimizer updates while draining",
            ),
            optimizer_state_reset_at_tail=_boolean(
                schedule["optimizer_state_reset_at_tail"],
                name="optimizer state reset at tail",
            ),
            optimizer_schedule=_nonempty_string(
                schedule["optimizer_schedule"], name="optimizer schedule"
            ),
            final_learning_rate_fraction=_finite_float(
                schedule["final_learning_rate_fraction"],
                name="final learning-rate fraction",
            ),
            zero_control=_nonempty_string(
                schedule["zero_control"], name="zero control"
            ),
            calibration_learner_seeds=_seed_tuple(
                seeds["calibration_learner"], name="calibration"
            ),
            validation_learner_seeds=_seed_tuple(
                seeds["validation_learner"], name="validation"
            ),
            test_learner_seeds=_seed_tuple(seeds["test_learner"], name="test"),
            calibration_evaluation_seed=_positive_int(
                seeds["calibration_evaluation"], name="calibration evaluation seed"
            ),
            validation_evaluation_seed=_positive_int(
                seeds["validation_evaluation"], name="validation evaluation seed"
            ),
            test_evaluation_seed=_positive_int(
                seeds["test_evaluation"], name="test evaluation seed"
            ),
            calibration_budgets_updates=integer_tuple(
                calibration["budgets_updates"], name="calibration budgets"
            ),
            calibration_elimination_metric=_nonempty_string(
                calibration["elimination_metric"], name="calibration elimination metric"
            ),
            validation_updates=_positive_int(
                validation["updates"], name="validation updates"
            ),
            validation_episodes_per_policy=_positive_int(
                validation["episodes_per_policy"], name="validation episodes"
            ),
            validation_selection_data=_nonempty_string(
                validation["selection_data"], name="validation selection data"
            ),
            test_updates=_positive_int(test["updates"], name="test updates"),
            test_episodes_per_policy=_positive_int(
                test["episodes_per_policy"], name="test episodes"
            ),
            test_sealed=_boolean(
                test["sealed_until_one_candidate_is_selected"], name="sealed test"
            ),
            test_reuse_after_rejection=_boolean(
                test["reuse_after_rejection"], name="test reuse"
            ),
            auc_definition=_nonempty_string(
                statistics_table["auc"], name="AUC definition"
            ),
            maximum_cumulative_tick_overshoot_fraction=_finite_float(
                statistics_table["maximum_cumulative_tick_overshoot_fraction"],
                name="maximum cumulative tick overshoot fraction",
            ),
            bootstrap_unit=_nonempty_string(
                statistics_table["bootstrap_unit"], name="bootstrap unit"
            ),
            bootstrap_samples=_positive_int(
                statistics_table["bootstrap_samples"], name="bootstrap samples"
            ),
            bootstrap_seed=statistics_table["bootstrap_seed"],
            confidence_level=_finite_float(
                statistics_table["confidence_level"], name="confidence level"
            ),
            relative_score_denominator_floor=_finite_float(
                statistics_table["relative_score_denominator_floor"],
                name="relative-score denominator floor",
            ),
            minimum_relative_auc_gain=_finite_float(
                gates["minimum_relative_auc_gain"], name="minimum relative AUC gain"
            ),
            minimum_final_mean_retention=_finite_float(
                gates["minimum_final_mean_retention"],
                name="minimum final mean retention",
            ),
            minimum_p10_retention=_finite_float(
                gates["minimum_p10_retention"], name="minimum p10 retention"
            ),
            minimum_p10_absolute_delta_when_control_near_zero=_finite_float(
                gates["minimum_p10_absolute_delta_when_control_near_zero"],
                name="minimum p10 absolute delta",
            ),
            minimum_trivial_baseline_margin=_finite_float(
                gates["minimum_trivial_baseline_margin"],
                name="minimum trivial baseline margin",
            ),
            required_baselines=_string_tuple(
                baselines["required"], name="required baselines"
            ),
            optional_baselines=_string_tuple(
                baselines["optional"], name="optional baselines"
            ),
            minimum_baseline_episodes=_positive_int(
                baselines["minimum_episodes_per_policy"],
                name="minimum baseline episodes",
            ),
            retain_every_arm_record=_boolean(
                failures["retain_every_arm_record"], name="retain every arm record"
            ),
            missing_arm_behavior=_nonempty_string(
                failures["missing_arm"], name="missing arm behavior"
            ),
            missing_seed_behavior=_nonempty_string(
                failures["missing_seed"], name="missing seed behavior"
            ),
            pre_start_failure_behavior=_nonempty_string(
                failures["pre_start_failure"], name="pre-start failure behavior"
            ),
            post_start_failure_behavior=_nonempty_string(
                failures["post_start_failure"], name="post-start failure behavior"
            ),
            test_failure_behavior=_nonempty_string(
                failures["test_failure"], name="test failure behavior"
            ),
        )


def load_plan(path: str | Path) -> R3BExperimentPlan:
    with Path(path).open("rb") as handle:
        return R3BExperimentPlan.from_mapping(tomllib.load(handle))


@dataclass(frozen=True, slots=True)
class TrialJob:
    plan_sha256: str
    phase: str
    arm: CandidateArm
    learner_seed: int
    budget_updates: int
    sealed: bool
    seed_plan_sha256: str
    authorization_sha256: str | None = None
    version: str = "r3b-trial-job-v2"

    def __post_init__(self) -> None:
        if (
            self.phase not in _PHASES
            or self.version != "r3b-trial-job-v2"
            or not isinstance(self.arm, CandidateArm)
            or isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or not 0 <= self.learner_seed < 2**64
            or isinstance(self.budget_updates, bool)
            or not isinstance(self.budget_updates, int)
            or self.budget_updates <= 0
            or not isinstance(self.sealed, bool)
            or (self.phase == "calibration" and self.authorization_sha256 is not None)
            or (
                self.phase != "calibration"
                and not _is_nonzero_sha256(self.authorization_sha256)
            )
        ):
            raise ValueError("trial job identity or budget is invalid")
        for name in ("plan_sha256", "seed_plan_sha256"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or value == "0" * 64
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"trial job {name} is invalid")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "plan_sha256": self.plan_sha256,
            "phase": self.phase,
            "arm": self.arm.manifest(),
            "learner_seed": self.learner_seed,
            "budget_updates": self.budget_updates,
            "sealed": self.sealed,
            "seed_plan_sha256": self.seed_plan_sha256,
            "authorization_sha256": self.authorization_sha256,
        }

    @classmethod
    def from_manifest(cls, value: object) -> TrialJob:
        manifest = _require_keys(
            value,
            {
                "version",
                "plan_sha256",
                "phase",
                "arm",
                "learner_seed",
                "budget_updates",
                "sealed",
                "seed_plan_sha256",
                "authorization_sha256",
            },
            location="trial job",
        )
        if (
            any(
                type(manifest[name]) is not str
                for name in ("version", "plan_sha256", "phase", "seed_plan_sha256")
            )
            or type(manifest["learner_seed"]) is not int
            or type(manifest["budget_updates"]) is not int
            or type(manifest["sealed"]) is not bool
            or (
                manifest["authorization_sha256"] is not None
                and type(manifest["authorization_sha256"]) is not str
            )
        ):
            raise ValueError("trial job field types are malformed")
        try:
            result = cls(
                plan_sha256=manifest["plan_sha256"],  # type: ignore[arg-type]
                phase=manifest["phase"],  # type: ignore[arg-type]
                arm=CandidateArm.from_manifest(manifest["arm"]),
                learner_seed=manifest["learner_seed"],  # type: ignore[arg-type]
                budget_updates=manifest["budget_updates"],  # type: ignore[arg-type]
                sealed=manifest["sealed"],  # type: ignore[arg-type]
                seed_plan_sha256=manifest["seed_plan_sha256"],  # type: ignore[arg-type]
                authorization_sha256=manifest["authorization_sha256"],  # type: ignore[arg-type]
                version=manifest["version"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("trial job is malformed") from exc
        _require_manifest_round_trip(manifest, result.manifest(), location="trial job")
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class CalibrationSelectionAuthorization:
    """The single LR-per-alpha selection allowed to enter validation."""

    plan_sha256: str
    arms: tuple[CandidateArm, ...]
    calibration_results_sha256: str
    runner_spec_sha256: str
    evaluation_suite_sha256: str
    version: str = "r3b-calibration-selection-authorization-v2"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-calibration-selection-authorization-v2"
            or not _is_nonzero_sha256(self.plan_sha256)
            or not _is_nonzero_sha256(self.calibration_results_sha256)
            or not _is_nonzero_sha256(self.runner_spec_sha256)
            or not _is_nonzero_sha256(self.evaluation_suite_sha256)
            or not isinstance(self.arms, tuple)
            or not self.arms
            or any(not isinstance(arm, CandidateArm) for arm in self.arms)
        ):
            raise ValueError("calibration-selection authorization is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "plan_sha256": self.plan_sha256,
            "arms": [arm.manifest() for arm in self.arms],
            "calibration_results_sha256": self.calibration_results_sha256,
            "runner_spec_sha256": self.runner_spec_sha256,
            "evaluation_suite_sha256": self.evaluation_suite_sha256,
        }

    @classmethod
    def from_manifest(cls, value: object) -> CalibrationSelectionAuthorization:
        manifest = _require_keys(
            value,
            {
                "version",
                "plan_sha256",
                "arms",
                "calibration_results_sha256",
                "runner_spec_sha256",
                "evaluation_suite_sha256",
            },
            location="calibration-selection authorization",
        )
        arms = _manifest_list(
            manifest["arms"], location="calibration-selection authorization arms"
        )
        if any(
            type(manifest[name]) is not str
            for name in (
                "version",
                "plan_sha256",
                "calibration_results_sha256",
                "runner_spec_sha256",
                "evaluation_suite_sha256",
            )
        ):
            raise ValueError(
                "calibration-selection authorization field types are malformed"
            )
        try:
            result = cls(
                plan_sha256=manifest["plan_sha256"],  # type: ignore[arg-type]
                arms=tuple(CandidateArm.from_manifest(arm) for arm in arms),
                calibration_results_sha256=manifest["calibration_results_sha256"],  # type: ignore[arg-type]
                runner_spec_sha256=manifest["runner_spec_sha256"],  # type: ignore[arg-type]
                evaluation_suite_sha256=manifest["evaluation_suite_sha256"],  # type: ignore[arg-type]
                version=manifest["version"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "calibration-selection authorization is malformed"
            ) from exc
        _require_manifest_round_trip(
            manifest,
            result.manifest(),
            location="calibration-selection authorization",
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class TestSuiteCommitment:
    """Durable pre-validation commitment to the single sealed test suite."""

    plan_sha256: str
    test_suite_sha256: str
    ledger_nonce: str
    version: str = "r3b-test-suite-commitment-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-test-suite-commitment-v1"
            or not _is_nonzero_sha256(self.plan_sha256)
            or not _is_nonzero_sha256(self.test_suite_sha256)
            or not _is_nonzero_sha256(self.ledger_nonce)
        ):
            raise ValueError("sealed test-suite commitment is malformed")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_manifest(cls, value: object) -> TestSuiteCommitment:
        expected = {"plan_sha256", "test_suite_sha256", "ledger_nonce", "version"}
        manifest = _require_keys(value, expected, location="test-suite commitment")
        if any(type(manifest[name]) is not str for name in expected):
            raise ValueError("test-suite commitment field types are malformed")
        try:
            result = cls(**manifest)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("test-suite commitment is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="test-suite commitment"
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class CurvePoint:
    simulated_ticks: int
    mean_raw_score: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.simulated_ticks, bool)
            or not isinstance(self.simulated_ticks, int)
            or self.simulated_ticks < 0
            or isinstance(self.mean_raw_score, bool)
            or not isinstance(self.mean_raw_score, (int, float))
            or not math.isfinite(float(self.mean_raw_score))
            or self.mean_raw_score < 0
        ):
            raise ValueError("curve points require nonnegative ticks and raw score")
        object.__setattr__(self, "mean_raw_score", float(self.mean_raw_score))


def tick_aligned_raw_score_auc(
    points: Sequence[CurvePoint], expected_ticks: Sequence[int]
) -> float:
    """Interpolate observed checkpoints to the target tick grid and normalize."""

    ticks = tuple(expected_ticks)
    if (
        len(ticks) < 2
        or ticks[0] != 0
        or any(
            isinstance(tick, bool) or not isinstance(tick, int) or tick < 0
            for tick in ticks
        )
    ):
        raise ValueError("expected tick grid must start at zero and have a horizon")
    if any(right <= left for left, right in zip(ticks, ticks[1:])):
        raise ValueError("expected tick grid must be strictly increasing")
    if len(points) != len(ticks):
        raise ValueError("raw-score curve must match the target-grid cardinality")
    observed_ticks = tuple(point.simulated_ticks for point in points)
    if (
        observed_ticks[0] != 0
        or any(right <= left for left, right in zip(observed_ticks, observed_ticks[1:]))
        or any(
            not (observed_ticks[index - 1] < target <= observed_ticks[index])
            for index, target in enumerate(ticks[1:], start=1)
        )
    ):
        raise ValueError(
            "each target tick must be bracketed by adjacent observed checkpoints"
        )
    aligned = [points[0]]
    for index, target in enumerate(ticks[1:], start=1):
        left = points[index - 1]
        right = points[index]
        fraction = (target - left.simulated_ticks) / (
            right.simulated_ticks - left.simulated_ticks
        )
        aligned.append(
            CurvePoint(
                target,
                left.mean_raw_score
                + fraction * (right.mean_raw_score - left.mean_raw_score),
            )
        )
    area = sum(
        (right.simulated_ticks - left.simulated_ticks)
        * (left.mean_raw_score + right.mean_raw_score)
        / 2.0
        for left, right in zip(aligned, aligned[1:])
    )
    return area / ticks[-1]


@dataclass(frozen=True, slots=True)
class TrainingCheckpointArtifact:
    """Session-produced checkpoint identity used for one metric-grid point."""

    learner_seed: int
    completed_updates: int
    simulated_ticks: int
    target_simulated_ticks: int
    plan_sha256: str
    job_sha256: str
    trial_manifest_sha256: str
    runner_spec_sha256: str
    checkpoint_manifest_sha256: str
    model_sha256: str
    deployment_policy_sha256: str
    version: str = "r3b-training-checkpoint-artifact-v2"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-training-checkpoint-artifact-v2"
            or isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or not 0 <= self.learner_seed < 2**64
            or isinstance(self.completed_updates, bool)
            or not isinstance(self.completed_updates, int)
            or self.completed_updates < 0
            or isinstance(self.simulated_ticks, bool)
            or not isinstance(self.simulated_ticks, int)
            or self.simulated_ticks < 0
            or isinstance(self.target_simulated_ticks, bool)
            or not isinstance(self.target_simulated_ticks, int)
            or self.target_simulated_ticks < 0
            or self.simulated_ticks < self.target_simulated_ticks
            or any(
                not _is_nonzero_sha256(value)
                for value in (
                    self.plan_sha256,
                    self.job_sha256,
                    self.trial_manifest_sha256,
                    self.runner_spec_sha256,
                    self.checkpoint_manifest_sha256,
                    self.model_sha256,
                    self.deployment_policy_sha256,
                )
            )
        ):
            raise ValueError("training checkpoint artifact is malformed")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_manifest(cls, value: object) -> TrainingCheckpointArtifact:
        expected = {
            "learner_seed",
            "completed_updates",
            "simulated_ticks",
            "target_simulated_ticks",
            "plan_sha256",
            "job_sha256",
            "trial_manifest_sha256",
            "runner_spec_sha256",
            "checkpoint_manifest_sha256",
            "model_sha256",
            "deployment_policy_sha256",
            "version",
        }
        manifest = _require_keys(
            value, expected, location="training checkpoint artifact"
        )
        if any(
            type(manifest[name]) is not int
            for name in (
                "learner_seed",
                "completed_updates",
                "simulated_ticks",
                "target_simulated_ticks",
            )
        ) or any(
            type(manifest[name]) is not str
            for name in expected
            - {
                "learner_seed",
                "completed_updates",
                "simulated_ticks",
                "target_simulated_ticks",
            }
        ):
            raise ValueError("training checkpoint artifact field types are malformed")
        try:
            result = cls(**manifest)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("training checkpoint artifact is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="training checkpoint artifact"
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class CheckpointEvaluation:
    """One evaluated training checkpoint with an explicit clock/policy binding."""

    checkpoint: TrainingCheckpointArtifact
    report: object
    version: str = "r3b-checkpoint-evaluation-v2"

    def __post_init__(self) -> None:
        from .r3b_evaluation import EvaluationReport

        if (
            self.version != "r3b-checkpoint-evaluation-v2"
            or not isinstance(self.checkpoint, TrainingCheckpointArtifact)
            or not isinstance(self.report, EvaluationReport)
            or self.report.policy_sha256 != self.checkpoint.deployment_policy_sha256
        ):
            raise ValueError("checkpoint evaluation identity is malformed")

    @property
    def completed_updates(self) -> int:
        return self.checkpoint.completed_updates

    @property
    def simulated_ticks(self) -> int:
        return self.checkpoint.simulated_ticks

    @property
    def target_simulated_ticks(self) -> int:
        return self.checkpoint.target_simulated_ticks

    @property
    def checkpoint_artifact_sha256(self) -> str:
        return self.checkpoint.sha256

    @property
    def policy_sha256(self) -> str:
        return self.checkpoint.deployment_policy_sha256

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "checkpoint": self.checkpoint.manifest(),
            "report_sha256": self.report.sha256,
        }

    @classmethod
    def from_manifest(cls, value: object, *, report: object) -> CheckpointEvaluation:
        from .r3b_evaluation import EvaluationReport

        manifest = _require_keys(
            value,
            {"version", "checkpoint", "report_sha256"},
            location="checkpoint evaluation",
        )
        if (
            type(manifest["version"]) is not str
            or type(manifest["report_sha256"]) is not str
            or not isinstance(report, EvaluationReport)
            or manifest["report_sha256"] != report.sha256
        ):
            raise ValueError("checkpoint evaluation report reference mismatch")
        try:
            result = cls(
                TrainingCheckpointArtifact.from_manifest(manifest["checkpoint"]),
                report,
                manifest["version"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("checkpoint evaluation is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="checkpoint evaluation"
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class RawScoreMetricsArtifact:
    """Canonical checkpoint reports from which every selection metric is derived."""

    learner_seed: int
    curve_suite: object
    final_suite: object
    checkpoints: tuple[CheckpointEvaluation, ...]
    final_report: object
    version: str = "r3b-raw-score-metrics-v3"

    def __post_init__(self) -> None:
        from .r3b_evaluation import EvaluationReport, EvaluationSuite

        if (
            self.version != "r3b-raw-score-metrics-v3"
            or isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or not 0 <= self.learner_seed < 2**64
            or not isinstance(self.curve_suite, EvaluationSuite)
            or not isinstance(self.final_suite, EvaluationSuite)
            or not isinstance(self.final_report, EvaluationReport)
            or not isinstance(self.checkpoints, tuple)
            or len(self.checkpoints) < 2
        ):
            raise ValueError("raw-score metrics artifact is malformed")
        if (
            self.curve_suite.split != self.final_suite.split
            or self.curve_suite.backend != self.final_suite.backend
            or self.curve_suite.runtime_identity_sha256
            != self.final_suite.runtime_identity_sha256
            or self.curve_suite.assignment_sha256 != self.final_suite.assignment_sha256
            or self.curve_suite.action_spec_sha256
            != self.final_suite.action_spec_sha256
            or self.curve_suite.policy_seed != self.final_suite.policy_seed
            or self.curve_suite.repetitions != self.final_suite.repetitions
            or not set(self.curve_suite.logical_cell_ids).issubset(
                self.final_suite.logical_cell_ids
            )
            or self.final_report.suite_sha256 != self.final_suite.sha256
            or self.final_report.backend_identity_sha256
            != self.final_suite.runtime_identity_sha256
        ):
            raise ValueError("curve and final evaluation suites disagree")
        ticks: list[int] = []
        target_ticks: list[int] = []
        reports: list[EvaluationReport] = []
        expected_cells = {
            (snapshot_id, repetition)
            for snapshot_id in self.curve_suite.snapshot_ids
            for repetition in range(self.curve_suite.repetitions)
        }
        updates: list[int] = []
        for checkpoint in self.checkpoints:
            if (
                not isinstance(checkpoint, CheckpointEvaluation)
                or checkpoint.checkpoint.learner_seed != self.learner_seed
            ):
                raise ValueError("checkpoint report entry is malformed")
            tick = checkpoint.simulated_ticks
            report = checkpoint.report
            cells = {
                (episode.snapshot_id, episode.repetition) for episode in report.episodes
            }
            if (
                report.suite_sha256 != self.curve_suite.sha256
                or report.backend_identity_sha256
                != self.curve_suite.runtime_identity_sha256
                or cells != expected_cells
                or any(
                    episode.policy_seed
                    != self.curve_suite.episode_seed(
                        episode.snapshot_id, episode.repetition
                    )
                    or episode.decisions > self.curve_suite.max_decisions
                    or episode.elapsed_ticks > self.curve_suite.max_simulated_ticks
                    or not (episode.terminated or episode.truncated)
                    or episode.invalid_actions != 0
                    for episode in report.episodes
                )
            ):
                raise ValueError("checkpoint report cells do not match the suite")
            updates.append(checkpoint.completed_updates)
            ticks.append(tick)
            target_ticks.append(checkpoint.target_simulated_ticks)
            reports.append(report)
        final_cells = {
            (episode.snapshot_id, episode.repetition)
            for episode in self.final_report.episodes
        }
        expected_final_cells = {
            (snapshot_id, repetition)
            for snapshot_id in self.final_suite.snapshot_ids
            for repetition in range(self.final_suite.repetitions)
        }
        if (
            final_cells != expected_final_cells
            or self.final_report.policy_sha256 != self.checkpoints[-1].policy_sha256
            or self.final_report.evaluator_sha256
            != self.checkpoints[-1].report.evaluator_sha256
            or any(
                episode.policy_seed
                != self.final_suite.episode_seed(
                    episode.snapshot_id, episode.repetition
                )
                or episode.decisions > self.final_suite.max_decisions
                or episode.elapsed_ticks > self.final_suite.max_simulated_ticks
                or not (episode.terminated or episode.truncated)
                or episode.invalid_actions != 0
                for episode in self.final_report.episodes
            )
        ):
            raise ValueError("final checkpoint report cells do not match its suite")
        if (
            updates[0] != 0
            or ticks[0] != 0
            or target_ticks[0] != 0
            or any(right <= left for left, right in zip(updates, updates[1:]))
            or any(right <= left for left, right in zip(ticks, ticks[1:]))
            or any(right <= left for left, right in zip(target_ticks, target_ticks[1:]))
            or any(
                ticks[index - 1] >= target_ticks[index]
                or target_ticks[index] > ticks[index]
                for index in range(1, len(ticks))
            )
        ):
            raise ValueError(
                "checkpoint reports must bracket an increasing target tick grid"
            )
        if (
            len({report.evaluator_sha256 for report in reports}) != 1
            or len({report.backend_identity_sha256 for report in reports}) != 1
            or len(
                {
                    checkpoint.checkpoint_artifact_sha256
                    for checkpoint in self.checkpoints
                }
            )
            != len(self.checkpoints)
        ):
            raise ValueError(
                "checkpoint reports changed evaluator/backend or reused a checkpoint"
            )

    @property
    def tick_reports(self) -> tuple[tuple[int, str, object], ...]:
        """Compatibility view; authoritative metadata lives in ``checkpoints``."""

        return tuple(
            (
                checkpoint.simulated_ticks,
                checkpoint.checkpoint_artifact_sha256,
                checkpoint.report,
            )
            for checkpoint in self.checkpoints
        )

    @property
    def points(self) -> tuple[CurvePoint, ...]:
        return tuple(
            CurvePoint(
                tick,
                statistics.fmean(episode.raw_score for episode in report.episodes),
            )
            for checkpoint in self.checkpoints
            for tick, report in ((checkpoint.simulated_ticks, checkpoint.report),)
        )

    @property
    def suite(self):
        """Compatibility view: selection gates use the full final suite."""

        return self.final_suite

    @property
    def raw_score_auc(self) -> float:
        points = self.points
        return tick_aligned_raw_score_auc(
            points,
            tuple(checkpoint.target_simulated_ticks for checkpoint in self.checkpoints),
        )

    @property
    def final_mean_raw_score(self) -> float:
        return statistics.fmean(
            episode.raw_score for episode in self.final_report.episodes
        )

    @property
    def p10_raw_score(self) -> float:
        ordered = sorted(episode.raw_score for episode in self.final_report.episodes)
        return float(ordered[max(0, math.ceil(0.1 * len(ordered)) - 1)])

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "learner_seed": self.learner_seed,
            "curve_suite_sha256": self.curve_suite.sha256,
            "final_suite_sha256": self.final_suite.sha256,
            "checkpoints": [checkpoint.manifest() for checkpoint in self.checkpoints],
            "final_report_sha256": self.final_report.sha256,
        }

    @classmethod
    def from_manifest(
        cls,
        value: object,
        *,
        curve_suite: object,
        final_suite: object,
        reports: Sequence[object],
        final_report: object,
    ) -> RawScoreMetricsArtifact:
        from .r3b_evaluation import EvaluationReport, EvaluationSuite

        manifest = _require_keys(
            value,
            {
                "version",
                "learner_seed",
                "curve_suite_sha256",
                "final_suite_sha256",
                "checkpoints",
                "final_report_sha256",
            },
            location="raw-score metrics artifact",
        )
        checkpoints = _manifest_list(
            manifest["checkpoints"], location="raw-score metrics checkpoints"
        )
        if (
            type(manifest["version"]) is not str
            or type(manifest["learner_seed"]) is not int
            or type(manifest["curve_suite_sha256"]) is not str
            or type(manifest["final_suite_sha256"]) is not str
            or type(manifest["final_report_sha256"]) is not str
            or not isinstance(curve_suite, EvaluationSuite)
            or not isinstance(final_suite, EvaluationSuite)
            or not isinstance(final_report, EvaluationReport)
            or manifest["curve_suite_sha256"] != curve_suite.sha256
            or manifest["final_suite_sha256"] != final_suite.sha256
            or manifest["final_report_sha256"] != final_report.sha256
        ):
            raise ValueError("raw-score metrics suite reference mismatch")
        supplied = tuple(reports)
        if any(not isinstance(report, EvaluationReport) for report in supplied):
            raise TypeError("raw-score metrics reports must be evaluation reports")
        reports_by_sha256 = {report.sha256: report for report in supplied}
        if len(reports_by_sha256) != len(supplied):
            raise ValueError("raw-score metrics supplied duplicate reports")
        referenced: list[str] = []
        for checkpoint in checkpoints:
            table = _require_keys(
                checkpoint,
                {"version", "checkpoint", "report_sha256"},
                location="raw-score checkpoint evaluation",
            )
            report_sha256 = table["report_sha256"]
            if type(report_sha256) is not str:
                raise ValueError("raw-score checkpoint report hash is malformed")
            referenced.append(report_sha256)
        if set(referenced) != set(reports_by_sha256) or len(referenced) != len(
            reports_by_sha256
        ):
            raise ValueError("raw-score metrics report references mismatch")
        try:
            result = cls(
                learner_seed=manifest["learner_seed"],  # type: ignore[arg-type]
                curve_suite=curve_suite,
                final_suite=final_suite,
                checkpoints=tuple(
                    CheckpointEvaluation.from_manifest(
                        checkpoint,
                        report=reports_by_sha256[report_sha256],
                    )
                    for checkpoint, report_sha256 in zip(checkpoints, referenced)
                ),
                final_report=final_report,
                version=manifest["version"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("raw-score metrics artifact is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="raw-score metrics artifact"
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class ExactResumeArtifact:
    """Independent checkpoint restore whose next update matches uninterrupted work."""

    trial_manifest_sha256: str
    checkpoint_manifest_sha256: str
    checkpoint_model_sha256: str
    source_next_update_sha256: str
    restored_next_update_sha256: str
    source_after_state_sha256: str
    restored_after_state_sha256: str
    version: str = "r3b-exact-resume-artifact-v2"
    _verification_token: InitVar[object] = None

    def __post_init__(self, _verification_token: object) -> None:
        hashes = (
            self.trial_manifest_sha256,
            self.checkpoint_manifest_sha256,
            self.checkpoint_model_sha256,
            self.source_next_update_sha256,
            self.restored_next_update_sha256,
            self.source_after_state_sha256,
            self.restored_after_state_sha256,
        )
        if (
            _verification_token is not _EXACT_RESUME_VERIFICATION_TOKEN
            or self.version != "r3b-exact-resume-artifact-v2"
            or any(not _is_nonzero_sha256(value) for value in hashes)
            or self.source_next_update_sha256 != self.restored_next_update_sha256
            or self.source_after_state_sha256 != self.restored_after_state_sha256
        ):
            raise ValueError("exact-resume artifact does not prove equal continuation")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def _verified_exact_resume_artifact(
    trial_manifest_sha256: str,
    checkpoint_manifest_sha256: str,
    checkpoint_model_sha256: str,
    source_next_update_sha256: str,
    restored_next_update_sha256: str,
    source_after_state_sha256: str,
    restored_after_state_sha256: str,
) -> ExactResumeArtifact:
    """Internal constructor used only after an actual restore continuation audit."""

    return ExactResumeArtifact(
        trial_manifest_sha256,
        checkpoint_manifest_sha256,
        checkpoint_model_sha256,
        source_next_update_sha256,
        restored_next_update_sha256,
        source_after_state_sha256,
        restored_after_state_sha256,
        _verification_token=_EXACT_RESUME_VERIFICATION_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class EngineeringEvidence:
    """Artifact identities required before an outcome may enter selection."""

    phase: str
    completed_updates: int
    plan_sha256: str
    job_sha256: str
    arm_id: str
    learner_seed: int
    authorization_sha256: str | None
    sealed_job_lease_sha256: str | None
    policy_sha256: str
    trial_manifest_sha256: str
    runner_spec_sha256: str
    pairing_sha256: str
    metrics_sha256: str
    evaluation_suite_sha256: str
    evaluation_report_sha256: str
    final_checkpoint_artifact: TrainingCheckpointArtifact
    resume_checkpoint_artifact: TrainingCheckpointArtifact
    checkpoint_resume_artifact: ExactResumeArtifact
    exact_backend_parity_artifact: object
    tail_state_sha256: str | None = None
    tail_phase: str | None = None
    score_only_updates: int = 0
    version: str = "r3b-engineering-evidence-v2"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-engineering-evidence-v2"
            or self.phase not in _PHASES
            or isinstance(self.completed_updates, bool)
            or not isinstance(self.completed_updates, int)
            or self.completed_updates <= 0
            or not self.arm_id
            or isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or not 0 <= self.learner_seed < 2**64
            or isinstance(self.score_only_updates, bool)
            or not isinstance(self.score_only_updates, int)
            or self.score_only_updates < 0
        ):
            raise ValueError("engineering evidence phase or counters are invalid")
        from .r3b_evaluation import LearnedPolicyBackendParityArtifact

        if (
            not isinstance(self.final_checkpoint_artifact, TrainingCheckpointArtifact)
            or not isinstance(
                self.resume_checkpoint_artifact, TrainingCheckpointArtifact
            )
            or not isinstance(self.checkpoint_resume_artifact, ExactResumeArtifact)
            or not isinstance(
                self.exact_backend_parity_artifact,
                LearnedPolicyBackendParityArtifact,
            )
        ):
            raise ValueError("engineering audit artifacts are malformed")
        hashes = (
            self.plan_sha256,
            self.job_sha256,
            self.policy_sha256,
            self.trial_manifest_sha256,
            self.runner_spec_sha256,
            self.pairing_sha256,
            self.metrics_sha256,
            self.evaluation_suite_sha256,
            self.evaluation_report_sha256,
            self.final_checkpoint_artifact.sha256,
            self.resume_checkpoint_artifact.sha256,
            self.checkpoint_resume_artifact.sha256,
            self.exact_backend_parity_artifact.sha256,
        )
        if self.tail_state_sha256 is not None:
            hashes += (self.tail_state_sha256,)
        if self.authorization_sha256 is not None:
            hashes += (self.authorization_sha256,)
        if self.sealed_job_lease_sha256 is not None:
            hashes += (self.sealed_job_lease_sha256,)
        if any(not _is_nonzero_sha256(value) for value in hashes):
            raise ValueError("engineering evidence must bind nonzero SHA-256 artifacts")
        if (
            self.final_checkpoint_artifact.learner_seed != self.learner_seed
            or self.final_checkpoint_artifact.completed_updates
            != self.completed_updates
            or self.final_checkpoint_artifact.plan_sha256 != self.plan_sha256
            or self.final_checkpoint_artifact.job_sha256 != self.job_sha256
            or self.final_checkpoint_artifact.trial_manifest_sha256
            != self.trial_manifest_sha256
            or self.final_checkpoint_artifact.runner_spec_sha256
            != self.runner_spec_sha256
            or self.final_checkpoint_artifact.deployment_policy_sha256
            != self.policy_sha256
            or self.resume_checkpoint_artifact.learner_seed != self.learner_seed
            or self.resume_checkpoint_artifact.completed_updates
            >= self.completed_updates
            or self.resume_checkpoint_artifact.plan_sha256 != self.plan_sha256
            or self.resume_checkpoint_artifact.job_sha256 != self.job_sha256
            or self.resume_checkpoint_artifact.trial_manifest_sha256
            != self.trial_manifest_sha256
            or self.resume_checkpoint_artifact.runner_spec_sha256
            != self.runner_spec_sha256
            or self.checkpoint_resume_artifact.trial_manifest_sha256
            != self.trial_manifest_sha256
            or self.checkpoint_resume_artifact.checkpoint_manifest_sha256
            != self.resume_checkpoint_artifact.checkpoint_manifest_sha256
            or self.checkpoint_resume_artifact.checkpoint_model_sha256
            != self.resume_checkpoint_artifact.model_sha256
            or self.exact_backend_parity_artifact.policy_sha256 != self.policy_sha256
            or self.exact_backend_parity_artifact.exact_suite.sha256
            != self.evaluation_suite_sha256
            or self.exact_backend_parity_artifact.exact_report.sha256
            != self.evaluation_report_sha256
        ):
            raise ValueError("engineering audit artifacts disagree with the trial")
        if self.phase == "calibration":
            if (
                self.authorization_sha256 is not None
                or self.sealed_job_lease_sha256 is not None
                or self.tail_state_sha256 is not None
                or self.tail_phase is not None
                or self.score_only_updates != 0
            ):
                raise ValueError("calibration evidence cannot claim tail completion")
        elif self.authorization_sha256 is None:
            raise ValueError("full-budget evidence must bind its phase authorization")
        if self.phase == "test" and self.sealed_job_lease_sha256 is None:
            raise ValueError("test evidence must bind its one-time job lease")
        if self.phase != "test" and self.sealed_job_lease_sha256 is not None:
            raise ValueError("only sealed-test evidence may bind a job lease")
        if self.phase != "calibration" and (
            self.tail_state_sha256 is None
            or self.tail_phase != "complete"
            or self.score_only_updates != 400
        ):
            raise ValueError(
                "full-budget evidence requires a completed score-only tail"
            )

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["final_checkpoint_artifact"] = self.final_checkpoint_artifact.manifest()
        value["resume_checkpoint_artifact"] = self.resume_checkpoint_artifact.manifest()
        value["checkpoint_resume_artifact"] = self.checkpoint_resume_artifact.manifest()
        value["exact_backend_parity_artifact"] = (
            self.exact_backend_parity_artifact.manifest()
        )
        return value

    @classmethod
    def from_manifest(
        cls,
        value: object,
        *,
        checkpoint_resume_artifact: ExactResumeArtifact,
        exact_backend_parity_artifact: object,
    ) -> EngineeringEvidence:
        expected = {
            "phase",
            "completed_updates",
            "plan_sha256",
            "job_sha256",
            "arm_id",
            "learner_seed",
            "authorization_sha256",
            "sealed_job_lease_sha256",
            "policy_sha256",
            "trial_manifest_sha256",
            "runner_spec_sha256",
            "pairing_sha256",
            "metrics_sha256",
            "evaluation_suite_sha256",
            "evaluation_report_sha256",
            "final_checkpoint_artifact",
            "resume_checkpoint_artifact",
            "checkpoint_resume_artifact",
            "exact_backend_parity_artifact",
            "tail_state_sha256",
            "tail_phase",
            "score_only_updates",
            "version",
        }
        manifest = _require_keys(value, expected, location="engineering evidence")
        if (
            manifest["checkpoint_resume_artifact"]
            != checkpoint_resume_artifact.manifest()
            or manifest["exact_backend_parity_artifact"]
            != exact_backend_parity_artifact.manifest()
        ):
            raise ValueError("engineering evidence audit dependencies differ")
        arguments = dict(manifest)
        arguments["final_checkpoint_artifact"] = (
            TrainingCheckpointArtifact.from_manifest(
                manifest["final_checkpoint_artifact"]
            )
        )
        arguments["resume_checkpoint_artifact"] = (
            TrainingCheckpointArtifact.from_manifest(
                manifest["resume_checkpoint_artifact"]
            )
        )
        arguments["checkpoint_resume_artifact"] = checkpoint_resume_artifact
        arguments["exact_backend_parity_artifact"] = exact_backend_parity_artifact
        try:
            result = cls(**arguments)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("engineering evidence is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="engineering evidence"
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class LearnerOutcome:
    learner_seed: int
    raw_score_auc: float
    final_mean_raw_score: float
    p10_raw_score: float
    initial_model_sha256: str
    assignment_sha256: str
    seed_plan_sha256: str
    metrics_artifact: RawScoreMetricsArtifact
    engineering_evidence: EngineeringEvidence | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or not 0 <= self.learner_seed < 2**64
        ):
            raise ValueError("learner seed must be uint64")
        for name in (
            "raw_score_auc",
            "final_mean_raw_score",
            "p10_raw_score",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
            object.__setattr__(self, name, float(value))
        if self.engineering_evidence is not None and not isinstance(
            self.engineering_evidence, EngineeringEvidence
        ):
            raise ValueError("engineering evidence is malformed")
        if (
            not isinstance(self.metrics_artifact, RawScoreMetricsArtifact)
            or self.metrics_artifact.learner_seed != self.learner_seed
        ):
            raise ValueError("learner outcome requires its typed metrics artifact")
        derived = (
            self.metrics_artifact.raw_score_auc,
            self.metrics_artifact.final_mean_raw_score,
            self.metrics_artifact.p10_raw_score,
        )
        supplied = (
            self.raw_score_auc,
            self.final_mean_raw_score,
            self.p10_raw_score,
        )
        if any(
            not math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)
            for left, right in zip(supplied, derived)
        ):
            raise ValueError("learner aggregates disagree with checkpoint reports")
        if (
            self.engineering_evidence is not None
            and self.engineering_evidence.learner_seed != self.learner_seed
        ):
            raise ValueError("engineering evidence learner seed mismatch")
        if self.engineering_evidence is not None and (
            self.engineering_evidence.metrics_sha256 != self.metrics_artifact.sha256
            or self.engineering_evidence.evaluation_suite_sha256
            != self.metrics_artifact.suite.sha256
            or self.engineering_evidence.evaluation_report_sha256
            != self.metrics_artifact.final_report.sha256
            or self.engineering_evidence.policy_sha256
            != self.metrics_artifact.final_report.policy_sha256
            or self.engineering_evidence.final_checkpoint_artifact.sha256
            != self.metrics_artifact.checkpoints[-1].checkpoint_artifact_sha256
            or self.engineering_evidence.resume_checkpoint_artifact.sha256
            != self.metrics_artifact.checkpoints[-2].checkpoint_artifact_sha256
        ):
            raise ValueError("engineering evidence disagrees with metrics reports")
        for name in (
            "initial_model_sha256",
            "assignment_sha256",
            "seed_plan_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or value == "0" * 64
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a nonzero lowercase SHA-256")

    def manifest(self) -> dict[str, object]:
        return {
            "learner_seed": self.learner_seed,
            "raw_score_auc": self.raw_score_auc,
            "final_mean_raw_score": self.final_mean_raw_score,
            "p10_raw_score": self.p10_raw_score,
            "initial_model_sha256": self.initial_model_sha256,
            "assignment_sha256": self.assignment_sha256,
            "seed_plan_sha256": self.seed_plan_sha256,
            "metrics_artifact": self.metrics_artifact.manifest(),
            "engineering_evidence": (
                None
                if self.engineering_evidence is None
                else self.engineering_evidence.manifest()
            ),
        }

    @classmethod
    def from_manifest(
        cls,
        value: object,
        *,
        metrics_artifact: RawScoreMetricsArtifact,
        engineering_evidence: EngineeringEvidence | None,
    ) -> LearnerOutcome:
        manifest = _require_keys(
            value,
            {
                "learner_seed",
                "raw_score_auc",
                "final_mean_raw_score",
                "p10_raw_score",
                "initial_model_sha256",
                "assignment_sha256",
                "seed_plan_sha256",
                "metrics_artifact",
                "engineering_evidence",
            },
            location="learner outcome",
        )
        if manifest["metrics_artifact"] != metrics_artifact.manifest() or manifest[
            "engineering_evidence"
        ] != (
            None if engineering_evidence is None else engineering_evidence.manifest()
        ):
            raise ValueError("learner outcome dependencies differ")
        arguments = dict(manifest)
        arguments["metrics_artifact"] = metrics_artifact
        arguments["engineering_evidence"] = engineering_evidence
        try:
            result = cls(**arguments)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("learner outcome is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="learner outcome"
        )
        return result

    @property
    def engineering_pass(self) -> bool:
        return self.engineering_evidence is not None

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class ArmPhaseResult:
    arm_id: str
    phase: str
    status: str
    budget_updates: int
    outcomes: tuple[LearnerOutcome, ...] = ()
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.arm_id, str) or not self.arm_id:
            raise ValueError("arm_id is required")
        if self.phase not in _PHASES or self.status not in _RESULT_STATUSES:
            raise ValueError("unknown experiment phase or result status")
        _positive_int(self.budget_updates, name="result budget")
        if not isinstance(self.outcomes, tuple) or any(
            not isinstance(item, LearnerOutcome) for item in self.outcomes
        ):
            raise ValueError("outcomes must contain LearnerOutcome values")
        seeds = tuple(outcome.learner_seed for outcome in self.outcomes)
        if len(seeds) != len(set(seeds)):
            raise ValueError("learner outcomes must have unique seeds")
        if self.status == "complete":
            if not self.outcomes or self.failure_reason is not None:
                raise ValueError(
                    "complete results require outcomes and no failure reason"
                )
            if any(
                outcome.engineering_evidence is not None
                and (
                    outcome.engineering_evidence.phase != self.phase
                    or outcome.engineering_evidence.completed_updates
                    != self.budget_updates
                    or outcome.engineering_evidence.arm_id != self.arm_id
                )
                for outcome in self.outcomes
            ):
                raise ValueError("engineering evidence disagrees with its phase result")
        elif not isinstance(self.failure_reason, str) or not self.failure_reason:
            raise ValueError("non-complete results must retain a failure reason")

    def manifest(self) -> dict[str, object]:
        return {
            "arm_id": self.arm_id,
            "phase": self.phase,
            "status": self.status,
            "budget_updates": self.budget_updates,
            "outcomes": [outcome.manifest() for outcome in self.outcomes],
            "failure_reason": self.failure_reason,
        }

    @classmethod
    def from_manifest(
        cls,
        value: object,
        *,
        outcomes: Sequence[LearnerOutcome],
    ) -> ArmPhaseResult:
        manifest = _require_keys(
            value,
            {
                "arm_id",
                "phase",
                "status",
                "budget_updates",
                "outcomes",
                "failure_reason",
            },
            location="arm phase result",
        )
        supplied = tuple(outcomes)
        if manifest["outcomes"] != [outcome.manifest() for outcome in supplied]:
            raise ValueError("arm phase result outcome dependencies differ")
        arguments = dict(manifest)
        arguments["outcomes"] = supplied
        try:
            result = cls(**arguments)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("arm phase result is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="arm phase result"
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def _index_exact_results(
    results: Sequence[ArmPhaseResult], expected_ids: set[str], *, phase: str
) -> dict[str, ArmPhaseResult]:
    indexed: dict[str, ArmPhaseResult] = {}
    for result in results:
        if result.phase != phase or result.arm_id in indexed:
            raise ValueError(f"duplicate or wrong-phase {phase} result")
        indexed[result.arm_id] = result
    if set(indexed) != expected_ids:
        raise ValueError(f"{phase} results must retain exactly every expected arm")
    return indexed


def _require_exact_seeds(
    result: ArmPhaseResult, expected_seeds: tuple[int, ...]
) -> None:
    if result.status == "complete" and {
        outcome.learner_seed for outcome in result.outcomes
    } != set(expected_seeds):
        raise ValueError(
            f"complete result {result.arm_id} does not contain exact seeds"
        )


def _require_paired_identities(
    results: Sequence[ArmPhaseResult], expected_seeds: tuple[int, ...]
) -> None:
    by_seed: dict[int, tuple[str, str, str, str, str]] = {}
    for result in results:
        if result.status != "complete":
            continue
        for outcome in result.outcomes:
            if outcome.learner_seed not in expected_seeds:
                raise ValueError("result contains an unexpected learner seed")
            identity = (
                outcome.initial_model_sha256,
                outcome.assignment_sha256,
                outcome.seed_plan_sha256,
                (
                    outcome.engineering_evidence.pairing_sha256
                    if outcome.engineering_evidence is not None
                    else ""
                ),
                (
                    outcome.engineering_evidence.evaluation_suite_sha256
                    if outcome.engineering_evidence is not None
                    else ""
                ),
            )
            previous = by_seed.setdefault(outcome.learner_seed, identity)
            if previous != identity:
                raise ValueError(
                    "paired arms disagree on model, assignment, seed, runner, or suite identity"
                )


def _require_runner_spec(results: Sequence[ArmPhaseResult]) -> str:
    identities = {
        outcome.engineering_evidence.runner_spec_sha256
        for result in results
        for outcome in result.outcomes
        if outcome.engineering_evidence is not None
    }
    if len(identities) != 1:
        raise ValueError(
            "phase outcomes lack or disagree on the canonical runner specification"
        )
    return next(iter(identities))


def _require_evaluation_suite(
    plan: R3BExperimentPlan, results: Sequence[ArmPhaseResult]
) -> str:
    for result in results:
        for outcome in result.outcomes:
            suite = outcome.metrics_artifact.suite
            if suite.policy_seed != plan.evaluation_seed(result.phase):
                raise ValueError("evaluation suite policy seed differs from the plan")
    identities = {
        outcome.metrics_artifact.suite.sha256
        for result in results
        for outcome in result.outcomes
        if outcome.engineering_evidence is not None
    }
    if len(identities) != 1:
        raise ValueError(
            "phase outcomes lack or disagree on the canonical evaluation suite"
        )
    return next(iter(identities))


def _require_metric_artifacts(
    plan: R3BExperimentPlan, results: Sequence[ArmPhaseResult]
) -> None:
    for result in results:
        if result.status != "complete":
            continue
        expected_ticks = plan.tick_grid(result.budget_updates)
        expected_updates = tuple(
            range(
                0,
                result.budget_updates + 1,
                plan.checkpoint_interval_updates,
            )
        )
        for outcome in result.outcomes:
            artifact = outcome.metrics_artifact
            evidence = outcome.engineering_evidence
            observed_ticks = tuple(
                checkpoint.simulated_ticks for checkpoint in artifact.checkpoints
            )
            target_ticks = tuple(
                checkpoint.target_simulated_ticks for checkpoint in artifact.checkpoints
            )
            if (
                evidence is None
                or artifact.suite.split != result.phase
                or tuple(
                    checkpoint.completed_updates for checkpoint in artifact.checkpoints
                )
                != expected_updates
                or target_ticks != expected_ticks
                or any(
                    observed
                    > target * (1.0 + plan.maximum_cumulative_tick_overshoot_fraction)
                    for observed, target in zip(observed_ticks[1:], target_ticks[1:])
                )
                or any(
                    checkpoint.checkpoint.plan_sha256 != plan.sha256
                    or checkpoint.checkpoint.job_sha256 != evidence.job_sha256
                    or checkpoint.checkpoint.trial_manifest_sha256
                    != evidence.trial_manifest_sha256
                    or checkpoint.checkpoint.runner_spec_sha256
                    != evidence.runner_spec_sha256
                    for checkpoint in artifact.checkpoints
                )
            ):
                raise ValueError(
                    "learner metrics do not cover the exact phase checkpoint grid"
                )


def _require_plan_evidence(
    plan: R3BExperimentPlan, results: Sequence[ArmPhaseResult]
) -> None:
    if any(
        outcome.engineering_evidence is not None
        and outcome.engineering_evidence.plan_sha256 != plan.sha256
        for result in results
        for outcome in result.outcomes
    ):
        raise ValueError("engineering evidence belongs to another experiment plan")


def select_calibrated_learning_rates(
    plan: R3BExperimentPlan, results: Sequence[ArmPhaseResult]
) -> tuple[CandidateArm, ...]:
    """Select one LR per alpha from the final calibration rung."""

    arms = plan.arms
    indexed = _index_exact_results(
        results, {arm.arm_id for arm in arms}, phase="calibration"
    )
    _require_plan_evidence(plan, tuple(indexed.values()))
    _require_runner_spec(tuple(indexed.values()))
    _require_evaluation_suite(plan, tuple(indexed.values()))
    _require_metric_artifacts(plan, tuple(indexed.values()))
    _require_paired_identities(tuple(indexed.values()), plan.calibration_learner_seeds)
    selected: list[CandidateArm] = []
    final_budget = plan.calibration_budgets_updates[-1]
    for alpha in plan.alpha_weight_ppm:
        eligible: list[tuple[tuple[float, float, float], CandidateArm]] = []
        for arm in (
            candidate for candidate in arms if candidate.alpha_weight_ppm == alpha
        ):
            result = indexed[arm.arm_id]
            if result.budget_updates not in plan.calibration_budgets_updates:
                raise ValueError("calibration result is not from a planned rung")
            if result.status != "complete" or not all(
                outcome.engineering_pass for outcome in result.outcomes
            ):
                continue
            if {outcome.learner_seed for outcome in result.outcomes} != set(
                plan.calibration_learner_seeds
            ):
                continue
            if result.budget_updates != final_budget:
                raise ValueError(
                    "complete calibration result is not from the final rung"
                )
            aucs = [outcome.raw_score_auc for outcome in result.outcomes]
            finals = [outcome.final_mean_raw_score for outcome in result.outcomes]
            key = (
                statistics.median(aucs),
                statistics.fmean(finals),
                -arm.learning_rate,
            )
            eligible.append((key, arm))
        if not eligible:
            raise ValueError(f"alpha {alpha} has no complete calibration arm")
        selected.append(max(eligible, key=lambda item: item[0])[1])
    return tuple(selected)


def authorize_validation(
    plan: R3BExperimentPlan, results: Sequence[ArmPhaseResult]
) -> CalibrationSelectionAuthorization:
    retained = tuple(results)
    arms = select_calibrated_learning_rates(plan, retained)
    runner_spec_sha256 = _require_runner_spec(retained)
    evaluation_suite_sha256 = _require_evaluation_suite(plan, retained)
    return CalibrationSelectionAuthorization(
        plan.sha256,
        arms,
        _canonical_sha256(
            [result.sha256 for result in sorted(retained, key=lambda x: x.arm_id)]
        ),
        runner_spec_sha256,
        evaluation_suite_sha256,
    )


@dataclass(frozen=True, slots=True)
class ValidationRunAuthorization:
    """Calibration artifacts whose derived selection may enter validation."""

    plan: R3BExperimentPlan
    authorization: CalibrationSelectionAuthorization
    calibration_results: tuple[ArmPhaseResult, ...]
    test_commitment: TestSuiteCommitment
    version: str = "r3b-validation-run-authorization-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-validation-run-authorization-v1"
            or not isinstance(self.plan, R3BExperimentPlan)
            or not isinstance(self.authorization, CalibrationSelectionAuthorization)
            or not isinstance(self.calibration_results, tuple)
            or not isinstance(self.test_commitment, TestSuiteCommitment)
            or self.test_commitment.plan_sha256 != self.plan.sha256
        ):
            raise ValueError("validation-run authorization is malformed")
        expected = authorize_validation(self.plan, self.calibration_results)
        if expected != self.authorization:
            raise ValueError("validation-run calibration artifacts disagree")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "plan_sha256": self.plan.sha256,
            "authorization": self.authorization.manifest(),
            "calibration_result_sha256s": [
                result.sha256 for result in self.calibration_results
            ],
            "test_commitment": self.test_commitment.manifest(),
        }

    @classmethod
    def from_manifest(
        cls,
        value: object,
        *,
        plan: R3BExperimentPlan,
        calibration_results: Sequence[ArmPhaseResult],
    ) -> ValidationRunAuthorization:
        manifest = _require_keys(
            value,
            {
                "version",
                "plan_sha256",
                "authorization",
                "calibration_result_sha256s",
                "test_commitment",
            },
            location="validation-run authorization",
        )
        retained = tuple(calibration_results)
        result_sha256s = _manifest_list(
            manifest["calibration_result_sha256s"],
            location="validation-run calibration result identities",
        )
        if (
            type(manifest["version"]) is not str
            or type(manifest["plan_sha256"]) is not str
            or manifest["plan_sha256"] != plan.sha256
            or result_sha256s != [result.sha256 for result in retained]
        ):
            raise ValueError("validation-run authorization references differ")
        try:
            result = cls(
                plan=plan,
                authorization=CalibrationSelectionAuthorization.from_manifest(
                    manifest["authorization"]
                ),
                calibration_results=retained,
                test_commitment=TestSuiteCommitment.from_manifest(
                    manifest["test_commitment"]
                ),
                version=manifest["version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("validation-run authorization is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="validation-run authorization"
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "version": self.version,
                "plan_sha256": self.plan.sha256,
                "authorization_sha256": self.authorization.sha256,
                "calibration_results_sha256": self.authorization.calibration_results_sha256,
                "test_commitment_sha256": self.test_commitment.sha256,
            }
        )


def bind_validation_run(
    plan: R3BExperimentPlan,
    results: Sequence[ArmPhaseResult],
    test_commitment: TestSuiteCommitment,
) -> ValidationRunAuthorization:
    retained = tuple(results)
    return ValidationRunAuthorization(
        plan,
        authorize_validation(plan, retained),
        retained,
        test_commitment,
    )


def select_validation_candidate(
    plan: R3BExperimentPlan,
    validation_run: ValidationRunAuthorization,
    results: Sequence[ArmPhaseResult],
) -> CandidateArm | None:
    """Choose one shaped candidate using validation only; never consult test data."""

    if (
        not isinstance(validation_run, ValidationRunAuthorization)
        or validation_run.plan.sha256 != plan.sha256
    ):
        raise ValueError("validation selection lacks calibration authorization")
    authorization = validation_run.authorization
    arms = authorization.arms
    if (
        len(arms) != len(plan.alpha_weight_ppm)
        or tuple(arm.alpha_weight_ppm for arm in arms) != plan.alpha_weight_ppm
    ):
        raise ValueError("validation requires one calibrated arm per alpha")
    indexed = _index_exact_results(
        results, {arm.arm_id for arm in arms}, phase="validation"
    )
    _require_plan_evidence(plan, tuple(indexed.values()))
    runner_spec_sha256 = _require_runner_spec(tuple(indexed.values()))
    _require_evaluation_suite(plan, tuple(indexed.values()))
    _require_metric_artifacts(plan, tuple(indexed.values()))
    _require_paired_identities(tuple(indexed.values()), plan.validation_learner_seeds)
    if any(
        outcome.engineering_evidence is not None
        and outcome.engineering_evidence.authorization_sha256 != validation_run.sha256
        for result in indexed.values()
        for outcome in result.outcomes
    ):
        raise ValueError("validation evidence disagrees with calibration authorization")
    if runner_spec_sha256 != authorization.runner_spec_sha256:
        raise ValueError("validation runner differs from calibrated runner")
    for result in indexed.values():
        if result.budget_updates != plan.validation_updates:
            raise ValueError("validation result is not a full-budget run")
    control_arm = arms[0]
    control = indexed[control_arm.arm_id]
    if (
        control.status != "complete"
        or not all(outcome.engineering_pass for outcome in control.outcomes)
        or {outcome.learner_seed for outcome in control.outcomes}
        != set(plan.validation_learner_seeds)
    ):
        raise ValueError("complete score-only validation control is mandatory")
    control_auc = statistics.fmean(
        outcome.raw_score_auc for outcome in control.outcomes
    )
    control_final = statistics.fmean(
        outcome.final_mean_raw_score for outcome in control.outcomes
    )
    candidates: list[tuple[tuple[float, float, int, float], CandidateArm]] = []
    for arm in arms[1:]:
        result = indexed[arm.arm_id]
        if result.status != "complete" or not all(
            outcome.engineering_pass for outcome in result.outcomes
        ):
            continue
        if {outcome.learner_seed for outcome in result.outcomes} != set(
            plan.validation_learner_seeds
        ):
            continue
        mean_auc = statistics.fmean(
            outcome.raw_score_auc for outcome in result.outcomes
        )
        mean_final = statistics.fmean(
            outcome.final_mean_raw_score for outcome in result.outcomes
        )
        relative_auc_gain = (mean_auc - control_auc) / max(
            abs(control_auc), plan.relative_score_denominator_floor
        )
        final_retention = (
            mean_final / control_final
            if control_final > plan.relative_score_denominator_floor
            else mean_final - control_final
        )
        final_threshold = (
            plan.minimum_final_mean_retention
            if control_final > plan.relative_score_denominator_floor
            else 0.0
        )
        if (
            relative_auc_gain < plan.minimum_relative_auc_gain
            or final_retention < final_threshold
        ):
            continue
        candidates.append(
            (
                (
                    relative_auc_gain,
                    mean_final,
                    -arm.alpha_weight_ppm,
                    -arm.learning_rate,
                ),
                arm,
            )
        )
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


@dataclass(frozen=True, slots=True)
class BaselineEvidence:
    baseline_id: str
    status: str
    episodes: int
    mean_raw_score: float
    invalid_actions: int
    suite_sha256: str
    report_sha256: str
    replay_report_sha256: str
    diagnostic_report_sha256: str
    portable_backend_sha256: str
    exact_backend_sha256: str
    cross_backend_diagnostics_sha256: str
    version: str = "r3b-baseline-evidence-v2"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-baseline-evidence-v2"
            or not isinstance(self.baseline_id, str)
            or not self.baseline_id
        ):
            raise ValueError("baseline_id is required")
        if self.status not in {"complete", "failure"}:
            raise ValueError("baseline status must be complete or failure")
        if (
            isinstance(self.episodes, bool)
            or not isinstance(self.episodes, int)
            or self.episodes < 0
            or isinstance(self.invalid_actions, bool)
            or not isinstance(self.invalid_actions, int)
            or self.invalid_actions < 0
        ):
            raise ValueError("baseline counts must be nonnegative integers")
        if (
            isinstance(self.mean_raw_score, bool)
            or not isinstance(self.mean_raw_score, (int, float))
            or not math.isfinite(float(self.mean_raw_score))
            or self.mean_raw_score < 0
        ):
            raise ValueError("baseline raw-score mean must be finite and nonnegative")
        object.__setattr__(self, "mean_raw_score", float(self.mean_raw_score))
        if any(
            not _is_nonzero_sha256(value)
            for value in (
                self.suite_sha256,
                self.report_sha256,
                self.replay_report_sha256,
                self.diagnostic_report_sha256,
                self.portable_backend_sha256,
                self.exact_backend_sha256,
                self.cross_backend_diagnostics_sha256,
            )
        ):
            raise ValueError("baseline evidence must bind nonzero report SHA-256s")
        if (
            len(
                {
                    self.report_sha256,
                    self.replay_report_sha256,
                    self.diagnostic_report_sha256,
                }
            )
            != 3
            or self.portable_backend_sha256 == self.exact_backend_sha256
        ):
            raise ValueError("baseline evidence requires independent backend reports")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_manifest(cls, value: object) -> BaselineEvidence:
        expected = {
            "baseline_id",
            "status",
            "episodes",
            "mean_raw_score",
            "invalid_actions",
            "suite_sha256",
            "report_sha256",
            "replay_report_sha256",
            "diagnostic_report_sha256",
            "portable_backend_sha256",
            "exact_backend_sha256",
            "cross_backend_diagnostics_sha256",
            "version",
        }
        manifest = _require_keys(value, expected, location="baseline evidence")
        if (
            any(
                type(manifest[name]) is not str
                for name in expected - {"episodes", "mean_raw_score", "invalid_actions"}
            )
            or type(manifest["episodes"]) is not int
            or type(manifest["mean_raw_score"]) is not float
            or type(manifest["invalid_actions"]) is not int
        ):
            raise ValueError("baseline evidence field types are malformed")
        try:
            result = cls(**manifest)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("baseline evidence is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="baseline evidence"
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class CommittedBaselineEvidence:
    """Authority-bearing evidence reconstructed from the ledger and artifact store."""

    plan_sha256: str
    authorization_sha256: str
    commitment_sha256: str
    evidence: tuple[BaselineEvidence, ...]
    _verification_token: InitVar[object]

    def __post_init__(self, _verification_token: object) -> None:
        if (
            _verification_token is not _COMMITTED_BASELINE_EVIDENCE_TOKEN
            or any(
                not _is_nonzero_sha256(value)
                for value in (
                    self.plan_sha256,
                    self.authorization_sha256,
                    self.commitment_sha256,
                )
            )
            or not isinstance(self.evidence, tuple)
            or not self.evidence
            or any(not isinstance(item, BaselineEvidence) for item in self.evidence)
        ):
            raise ValueError(
                "committed baseline evidence requires verified durable authority"
            )


def baseline_requirements_pass(
    plan: R3BExperimentPlan,
    evidence: Sequence[BaselineEvidence],
    *,
    expected_suite_sha256: str | None = None,
) -> bool:
    indexed: dict[str, BaselineEvidence] = {}
    allowed = set(plan.required_baselines) | set(plan.optional_baselines)
    for item in evidence:
        if item.baseline_id not in allowed or item.baseline_id in indexed:
            return False
        indexed[item.baseline_id] = item
    if not set(plan.required_baselines) <= set(indexed):
        return False
    return all(
        item.status == "complete"
        and item.episodes >= plan.minimum_baseline_episodes
        and item.invalid_actions == 0
        and (
            expected_suite_sha256 is None or item.suite_sha256 == expected_suite_sha256
        )
        for item in (indexed[name] for name in plan.required_baselines)
    )


def _resolve_baseline_artifacts(
    artifacts: Sequence[object],
) -> tuple[BaselineEvidence, ...]:
    from .r3b_evaluation import BaselineArtifactBundle

    retained = tuple(artifacts)
    if len(retained) == 1 and isinstance(retained[0], CommittedBaselineEvidence):
        return retained[0].evidence
    if any(not isinstance(item, BaselineArtifactBundle) for item in artifacts):
        raise TypeError("sealed confirmation requires typed baseline artifacts")
    return tuple(item.evidence() for item in artifacts)


@dataclass(frozen=True, slots=True)
class SealedBaselineBatchCommitment:
    """Pre-execution contract for the one permitted required-baseline batch."""

    plan_sha256: str
    test_suite_sha256: str
    logical_manifest_sha256: str
    required_baselines: tuple[tuple[str, str], ...]
    episodes_per_baseline: int
    primary_backend: str = "exact"
    replay_backend: str = "exact"
    diagnostic_backend: str = "portable"
    version: str = "r3b-sealed-baseline-batch-commitment-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-sealed-baseline-batch-commitment-v1"
            or not _is_nonzero_sha256(self.plan_sha256)
            or not _is_nonzero_sha256(self.test_suite_sha256)
            or not _is_nonzero_sha256(self.logical_manifest_sha256)
            or not isinstance(self.required_baselines, tuple)
            or not self.required_baselines
            or any(
                not isinstance(item, tuple)
                or len(item) != 2
                or not item[0]
                or not _is_nonzero_sha256(item[1])
                for item in self.required_baselines
            )
            or len({item[0] for item in self.required_baselines})
            != len(self.required_baselines)
            or isinstance(self.episodes_per_baseline, bool)
            or not isinstance(self.episodes_per_baseline, int)
            or self.episodes_per_baseline <= 0
            or (
                self.primary_backend,
                self.replay_backend,
                self.diagnostic_backend,
            )
            != ("exact", "exact", "portable")
        ):
            raise ValueError("sealed baseline-batch commitment is malformed")

    @classmethod
    def from_plan(
        cls, plan: R3BExperimentPlan, test_suite: object
    ) -> SealedBaselineBatchCommitment:
        from .r3b_evaluation import EvaluationSuite, ScriptedBaselineSpec

        if (
            not isinstance(plan, R3BExperimentPlan)
            or not isinstance(test_suite, EvaluationSuite)
            or test_suite.split != "test"
            or not _is_nonzero_sha256(test_suite.logical_manifest_sha256)
            or len(test_suite.snapshot_ids) * test_suite.repetitions
            != plan.test_episodes_per_policy
        ):
            raise ValueError("baseline batch requires the sealed test cells")
        return cls(
            plan.sha256,
            test_suite.sha256,
            test_suite.logical_manifest_sha256,
            tuple(
                (baseline_id, ScriptedBaselineSpec(baseline_id).sha256)
                for baseline_id in plan.required_baselines
            ),
            plan.minimum_baseline_episodes,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "plan_sha256": self.plan_sha256,
            "test_suite_sha256": self.test_suite_sha256,
            "logical_manifest_sha256": self.logical_manifest_sha256,
            "required_baselines": [
                {"baseline_id": baseline_id, "policy_sha256": policy_sha256}
                for baseline_id, policy_sha256 in self.required_baselines
            ],
            "episodes_per_baseline": self.episodes_per_baseline,
            "reports": {
                "primary_backend": self.primary_backend,
                "deterministic_replay_backend": self.replay_backend,
                "diagnostic_backend": self.diagnostic_backend,
            },
        }

    @classmethod
    def from_manifest(cls, value: object) -> SealedBaselineBatchCommitment:
        manifest = _require_keys(
            value,
            {
                "version",
                "plan_sha256",
                "test_suite_sha256",
                "logical_manifest_sha256",
                "required_baselines",
                "episodes_per_baseline",
                "reports",
            },
            location="sealed baseline-batch commitment",
        )
        required = _manifest_list(
            manifest["required_baselines"],
            location="sealed baseline-batch required baselines",
        )
        reports = _require_keys(
            manifest["reports"],
            {
                "primary_backend",
                "deterministic_replay_backend",
                "diagnostic_backend",
            },
            location="sealed baseline-batch reports",
        )
        if (
            any(
                type(manifest[name]) is not str
                for name in (
                    "version",
                    "plan_sha256",
                    "test_suite_sha256",
                    "logical_manifest_sha256",
                )
            )
            or type(manifest["episodes_per_baseline"]) is not int
            or any(type(reports[name]) is not str for name in reports)
        ):
            raise ValueError("sealed baseline-batch field types are malformed")
        parsed: list[tuple[str, str]] = []
        for item in required:
            table = _require_keys(
                item,
                {"baseline_id", "policy_sha256"},
                location="sealed baseline-batch baseline",
            )
            if any(type(table[name]) is not str for name in table):
                raise ValueError("sealed baseline-batch baseline is malformed")
            parsed.append((table["baseline_id"], table["policy_sha256"]))  # type: ignore[arg-type]
        try:
            result = cls(
                plan_sha256=manifest["plan_sha256"],  # type: ignore[arg-type]
                test_suite_sha256=manifest["test_suite_sha256"],  # type: ignore[arg-type]
                logical_manifest_sha256=manifest["logical_manifest_sha256"],  # type: ignore[arg-type]
                required_baselines=tuple(parsed),
                episodes_per_baseline=manifest["episodes_per_baseline"],  # type: ignore[arg-type]
                primary_backend=reports["primary_backend"],  # type: ignore[arg-type]
                replay_backend=reports["deterministic_replay_backend"],  # type: ignore[arg-type]
                diagnostic_backend=reports["diagnostic_backend"],  # type: ignore[arg-type]
                version=manifest["version"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("sealed baseline-batch commitment is malformed") from exc
        _require_manifest_round_trip(
            manifest,
            result.manifest(),
            location="sealed baseline-batch commitment",
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def _resolve_sealed_baseline_batch(
    plan: R3BExperimentPlan,
    commitment: SealedBaselineBatchCommitment,
    artifacts: Sequence[object],
) -> tuple[BaselineEvidence, ...]:
    from .r3b_evaluation import BaselineArtifactBundle

    bundles = tuple(artifacts)
    if not isinstance(commitment, SealedBaselineBatchCommitment):
        raise TypeError("sealed baseline batch requires its typed commitment")
    if len(bundles) == 1 and isinstance(bundles[0], CommittedBaselineEvidence):
        committed = bundles[0]
        if (
            committed.plan_sha256 != plan.sha256
            or committed.commitment_sha256 != commitment.sha256
        ):
            raise ValueError("committed baseline evidence belongs to another batch")
        expected = tuple(item[0] for item in commitment.required_baselines)
        if tuple(
            item.baseline_id for item in committed.evidence
        ) != expected or not baseline_requirements_pass(
            plan,
            committed.evidence,
            expected_suite_sha256=commitment.test_suite_sha256,
        ):
            raise ValueError("committed baseline evidence violates its commitment")
        return committed.evidence
    if commitment.plan_sha256 != plan.sha256 or any(
        not isinstance(item, BaselineArtifactBundle) for item in bundles
    ):
        raise TypeError("sealed baseline batch requires typed baseline artifacts")
    indexed = {item.baseline.baseline_id: item for item in bundles}
    expected = tuple(item[0] for item in commitment.required_baselines)
    if len(indexed) != len(bundles) or set(indexed) != set(expected):
        raise ValueError("sealed baseline batch differs from its required baseline set")
    ordered = tuple(indexed[baseline_id] for baseline_id in expected)
    if any(
        bundle.baseline.sha256 != dict(commitment.required_baselines)[baseline_id]
        or bundle.primary_suite.sha256 != commitment.test_suite_sha256
        or bundle.primary_suite.backend != commitment.primary_backend
        or bundle.diagnostic_suite.backend != commitment.diagnostic_backend
        or bundle.primary_suite.split != "test"
        or bundle.diagnostic_suite.split != "test"
        or bundle.primary_suite.logical_manifest_sha256
        != commitment.logical_manifest_sha256
        or bundle.diagnostic_suite.logical_manifest_sha256
        != commitment.logical_manifest_sha256
        or len(bundle.primary_suite.snapshot_ids) * bundle.primary_suite.repetitions
        != commitment.episodes_per_baseline
        or len(bundle.diagnostic_suite.snapshot_ids)
        * bundle.diagnostic_suite.repetitions
        != commitment.episodes_per_baseline
        or bundle.primary_report.backend_identity_sha256
        != bundle.primary_suite.runtime_identity_sha256
        or bundle.primary_replay_report.backend_identity_sha256
        != bundle.primary_suite.runtime_identity_sha256
        or bundle.diagnostic_report.backend_identity_sha256
        != bundle.diagnostic_suite.runtime_identity_sha256
        for baseline_id, bundle in zip(expected, ordered)
    ):
        raise ValueError("sealed baseline batch violates its backend/report commitment")
    evidence = tuple(bundle.evidence() for bundle in ordered)
    if not baseline_requirements_pass(
        plan, evidence, expected_suite_sha256=commitment.test_suite_sha256
    ):
        raise ValueError("sealed baseline batch evidence does not pass requirements")
    return evidence


@dataclass(frozen=True, slots=True)
class SealedTestAuthorization:
    """One candidate selection bound to validation and a sealed test suite."""

    plan_sha256: str
    validation_run_sha256: str
    control_arm_id: str
    candidate_arm_id: str
    validation_results_sha256: str
    validation_suite_sha256: str
    test_suite_sha256: str
    runner_spec_sha256: str
    baseline_batch_commitment_sha256: str
    attempt: int = 1
    version: str = "r3b-sealed-test-authorization-v5"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-sealed-test-authorization-v5"
            or not self.control_arm_id
            or not self.candidate_arm_id
            or self.control_arm_id == self.candidate_arm_id
            or self.attempt != 1
            or any(
                not _is_nonzero_sha256(value)
                for value in (
                    self.plan_sha256,
                    self.validation_run_sha256,
                    self.validation_results_sha256,
                    self.validation_suite_sha256,
                    self.test_suite_sha256,
                    self.runner_spec_sha256,
                    self.baseline_batch_commitment_sha256,
                )
            )
        ):
            raise ValueError("sealed-test authorization identity is invalid")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_manifest(cls, value: object) -> SealedTestAuthorization:
        expected = {
            "plan_sha256",
            "validation_run_sha256",
            "control_arm_id",
            "candidate_arm_id",
            "validation_results_sha256",
            "validation_suite_sha256",
            "test_suite_sha256",
            "runner_spec_sha256",
            "baseline_batch_commitment_sha256",
            "attempt",
            "version",
        }
        manifest = _require_keys(value, expected, location="sealed-test authorization")
        if type(manifest["attempt"]) is not int or any(
            type(manifest[name]) is not str for name in expected - {"attempt"}
        ):
            raise ValueError("sealed-test authorization field types are malformed")
        try:
            result = cls(**manifest)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("sealed-test authorization is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="sealed-test authorization"
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def _authorize_sealed_test(
    plan: R3BExperimentPlan,
    validation_run: ValidationRunAuthorization,
    validation_results: Sequence[ArmPhaseResult],
    validation_suite: object,
    test_suite: object,
) -> SealedTestAuthorization:
    """Select exactly once from validation data and bind the sealed test cells."""

    from .r3b_evaluation import EvaluationSuite

    if (
        not isinstance(validation_suite, EvaluationSuite)
        or not isinstance(test_suite, EvaluationSuite)
        or validation_suite.split != "validation"
        or test_suite.split != "test"
        or validation_suite.policy_seed != plan.validation_evaluation_seed
        or test_suite.policy_seed != plan.test_evaluation_seed
        or len(validation_suite.snapshot_ids) * validation_suite.repetitions
        != plan.validation_episodes_per_policy
        or len(test_suite.snapshot_ids) * test_suite.repetitions
        != plan.test_episodes_per_policy
        or set(validation_suite.snapshot_ids) & set(test_suite.snapshot_ids)
        or set(validation_suite.logical_cell_ids) & set(test_suite.logical_cell_ids)
        or (
            validation_suite.runtime_identity_sha256,
            validation_suite.library_sha256,
            validation_suite.snapshot_store_sha256,
            validation_suite.action_spec_sha256,
            validation_suite.assignment_sha256,
            validation_suite.backend,
        )
        != (
            test_suite.runtime_identity_sha256,
            test_suite.library_sha256,
            test_suite.snapshot_store_sha256,
            test_suite.action_spec_sha256,
            test_suite.assignment_sha256,
            test_suite.backend,
        )
    ):
        raise ValueError("validation/test evaluation suites violate the sealed split")
    results = tuple(validation_results)
    if any(
        outcome.engineering_evidence is None
        or outcome.engineering_evidence.evaluation_suite_sha256
        != validation_suite.sha256
        for result in results
        for outcome in result.outcomes
    ):
        raise ValueError("validation outcomes disagree with their evaluation suite")
    if (
        not isinstance(validation_run, ValidationRunAuthorization)
        or validation_run.plan.sha256 != plan.sha256
        or validation_run.test_commitment.test_suite_sha256 != test_suite.sha256
    ):
        raise ValueError("sealed test lacks verified calibration artifacts")
    calibration_authorization = validation_run.authorization
    arms = calibration_authorization.arms
    candidate = select_validation_candidate(plan, validation_run, results)
    if candidate is None:
        raise ValueError("validation selected no shaped candidate")
    control = arms[0]
    if control.alpha_weight_ppm != 0 or candidate.alpha_weight_ppm <= 0:
        raise ValueError("sealed authorization requires control and shaped arms")
    result_identity = _canonical_sha256(
        [result.sha256 for result in sorted(results, key=lambda x: x.arm_id)]
    )
    baseline_commitment = SealedBaselineBatchCommitment.from_plan(plan, test_suite)
    return SealedTestAuthorization(
        plan.sha256,
        validation_run.sha256,
        control.arm_id,
        candidate.arm_id,
        result_identity,
        validation_suite.sha256,
        test_suite.sha256,
        calibration_authorization.runner_spec_sha256,
        baseline_commitment.sha256,
    )


@dataclass(frozen=True, slots=True)
class SealedTestRunAuthorization:
    """Resolved upstream artifacts required to create or confirm test jobs."""

    plan: R3BExperimentPlan
    authorization: SealedTestAuthorization
    validation_run: ValidationRunAuthorization
    validation_results: tuple[ArmPhaseResult, ...]
    validation_suite: object
    test_suite: object
    baseline_batch_commitment: SealedBaselineBatchCommitment
    ledger_path: str
    ledger_receipt_token: str
    version: str = "r3b-sealed-test-run-authorization-v4"
    allow_finalized: InitVar[bool] = False

    def __post_init__(self, allow_finalized: bool) -> None:
        if (
            self.version != "r3b-sealed-test-run-authorization-v4"
            or not isinstance(self.plan, R3BExperimentPlan)
            or not isinstance(self.validation_results, tuple)
            or not isinstance(
                self.baseline_batch_commitment, SealedBaselineBatchCommitment
            )
            or not isinstance(self.ledger_path, str)
            or not Path(self.ledger_path).is_absolute()
            or not _is_nonzero_sha256(self.ledger_receipt_token)
        ):
            raise ValueError("sealed-test run authorization is malformed")
        expected = _authorize_sealed_test(
            self.plan,
            self.validation_run,
            self.validation_results,
            self.validation_suite,
            self.test_suite,
        )
        if expected != self.authorization:
            raise ValueError("sealed-test run authorization artifacts disagree")
        expected_baselines = SealedBaselineBatchCommitment.from_plan(
            self.plan, self.test_suite
        )
        if (
            self.baseline_batch_commitment != expected_baselines
            or self.authorization.baseline_batch_commitment_sha256
            != expected_baselines.sha256
        ):
            raise ValueError("sealed-test run baseline commitment disagrees")
        result_sha256 = _canonical_sha256(
            [
                result.sha256
                for result in sorted(
                    self.validation_results, key=lambda value: value.arm_id
                )
            ]
        )
        self.assert_authorized(
            expected_validation_results_sha256=result_sha256,
            allow_finalized=allow_finalized,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "plan_sha256": self.plan.sha256,
            "authorization": self.authorization.manifest(),
            "validation_run_sha256": self.validation_run.sha256,
            "validation_result_sha256s": [
                result.sha256 for result in self.validation_results
            ],
            "validation_suite_sha256": self.validation_suite.sha256,
            "test_suite_sha256": self.test_suite.sha256,
            "baseline_batch_commitment": self.baseline_batch_commitment.manifest(),
            "ledger_path": self.ledger_path,
            # The receipt is a bearer secret. Its artifact store is required to be
            # private (0700 root, 0600 files), just like the ledger itself.
            "ledger_receipt_token": self.ledger_receipt_token,
        }

    @classmethod
    def from_manifest(
        cls,
        value: object,
        *,
        plan: R3BExperimentPlan,
        validation_run: ValidationRunAuthorization,
        validation_results: Sequence[ArmPhaseResult],
        validation_suite: object,
        test_suite: object,
        allow_finalized: bool = False,
    ) -> SealedTestRunAuthorization:
        manifest = _require_keys(
            value,
            {
                "version",
                "plan_sha256",
                "authorization",
                "validation_run_sha256",
                "validation_result_sha256s",
                "validation_suite_sha256",
                "test_suite_sha256",
                "baseline_batch_commitment",
                "ledger_path",
                "ledger_receipt_token",
            },
            location="sealed-test run authorization",
        )
        retained = tuple(validation_results)
        result_sha256s = _manifest_list(
            manifest["validation_result_sha256s"],
            location="sealed-test validation result identities",
        )
        string_fields = (
            "version",
            "plan_sha256",
            "validation_run_sha256",
            "validation_suite_sha256",
            "test_suite_sha256",
            "ledger_path",
            "ledger_receipt_token",
        )
        if (
            any(type(manifest[name]) is not str for name in string_fields)
            or manifest["plan_sha256"] != plan.sha256
            or manifest["validation_run_sha256"] != validation_run.sha256
            or result_sha256s != [result.sha256 for result in retained]
            or manifest["validation_suite_sha256"]
            != getattr(validation_suite, "sha256", None)
            or manifest["test_suite_sha256"] != getattr(test_suite, "sha256", None)
        ):
            raise ValueError("sealed-test run authorization references differ")
        try:
            result = cls(
                plan=plan,
                authorization=SealedTestAuthorization.from_manifest(
                    manifest["authorization"]
                ),
                validation_run=validation_run,
                validation_results=retained,
                validation_suite=validation_suite,
                test_suite=test_suite,
                baseline_batch_commitment=(
                    SealedBaselineBatchCommitment.from_manifest(
                        manifest["baseline_batch_commitment"]
                    )
                ),
                ledger_path=manifest["ledger_path"],
                ledger_receipt_token=manifest["ledger_receipt_token"],
                version=manifest["version"],
                allow_finalized=allow_finalized,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("sealed-test run authorization is malformed") from exc
        _require_manifest_round_trip(
            manifest,
            result.manifest(),
            location="sealed-test run authorization",
        )
        return result

    def publish(self, store: object) -> str:
        from .r3b_artifacts import ArtifactStore

        if not isinstance(store, ArtifactStore):
            raise TypeError("sealed authorization requires an ArtifactStore")
        return store.publish(
            kind=_SEALED_AUTHORIZATION_KIND,
            version=self.version,
            payload=self.manifest(),
        ).artifact_id

    @classmethod
    def load(
        cls,
        store: object,
        artifact_id: str,
        *,
        plan: R3BExperimentPlan,
        validation_run: ValidationRunAuthorization,
        validation_results: Sequence[ArmPhaseResult],
        validation_suite: object,
        test_suite: object,
        allow_finalized: bool = False,
    ) -> SealedTestRunAuthorization:
        from .r3b_artifacts import ArtifactStore

        if not isinstance(store, ArtifactStore):
            raise TypeError("sealed authorization requires an ArtifactStore")
        envelope = store.load(
            artifact_id,
            expected_kind=_SEALED_AUTHORIZATION_KIND,
            expected_version="r3b-sealed-test-run-authorization-v4",
        )
        return cls.from_manifest(
            envelope.payload,
            plan=plan,
            validation_run=validation_run,
            validation_results=validation_results,
            validation_suite=validation_suite,
            test_suite=test_suite,
            allow_finalized=allow_finalized,
        )

    @property
    def ledger_receipt_sha256(self) -> str:
        return hashlib.sha256(self.ledger_receipt_token.encode()).hexdigest()

    def assert_authorized(
        self,
        *,
        expected_validation_results_sha256: str | None = None,
        allow_finalized: bool = False,
    ) -> None:
        """Verify this opaque receipt against the durable ledger."""

        result_sha256 = expected_validation_results_sha256 or _canonical_sha256(
            [
                result.sha256
                for result in sorted(
                    self.validation_results, key=lambda value: value.arm_id
                )
            ]
        )
        jobs_sha256 = _canonical_sha256(
            list(_sealed_job_sha256s(self.plan, self.authorization))
        )
        try:
            with closing(_connect_private_ledger(self.ledger_path)) as connection:
                row = connection.execute(
                    "SELECT state,receipt_token_sha256,validation_run_sha256,"
                    "validation_results_sha256,validation_suite_sha256,"
                    "authorization_sha256,jobs_sha256,baseline_batch_sha256 "
                    "FROM sealed_test_attempt "
                    "WHERE plan_sha256=?",
                    (self.plan.sha256,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError("sealed-test ledger is unavailable") from exc
        expected_tail = (
            _bearer_sha256(self.ledger_receipt_token),
            self.validation_run.sha256,
            result_sha256,
            self.validation_suite.sha256,
            self.authorization.sha256,
            jobs_sha256,
            self.baseline_batch_commitment.sha256,
        )
        if (
            row is None
            or row[0]
            not in ({"authorized", "finalized"} if allow_finalized else {"authorized"})
            or tuple(row[1:]) != expected_tail
        ):
            raise RuntimeError("sealed-test receipt is not active in its ledger")

    @property
    def sha256(self) -> str:
        return self.authorization.sha256


def _bind_sealed_test_run(
    plan: R3BExperimentPlan,
    validation_run: ValidationRunAuthorization,
    validation_results: Sequence[ArmPhaseResult],
    validation_suite: object,
    test_suite: object,
    ledger_path: str,
    ledger_receipt_token: str,
) -> SealedTestRunAuthorization:
    results = tuple(validation_results)
    authorization = _authorize_sealed_test(
        plan,
        validation_run,
        results,
        validation_suite,
        test_suite,
    )
    return SealedTestRunAuthorization(
        plan,
        authorization,
        validation_run,
        results,
        validation_suite,
        test_suite,
        SealedBaselineBatchCommitment.from_plan(plan, test_suite),
        ledger_path,
        ledger_receipt_token,
    )


@dataclass(frozen=True, slots=True)
class SealedBaselineEvidenceArtifact:
    """Reloadable summaries for the baseline batch consumed by the ledger."""

    plan_sha256: str
    authorization_sha256: str
    commitment_sha256: str
    evidence: tuple[BaselineEvidence, ...]
    version: str = "r3b-sealed-baseline-evidence-artifact-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-sealed-baseline-evidence-artifact-v1"
            or any(
                not _is_nonzero_sha256(value)
                for value in (
                    self.plan_sha256,
                    self.authorization_sha256,
                    self.commitment_sha256,
                )
            )
            or not isinstance(self.evidence, tuple)
            or not self.evidence
            or any(not isinstance(item, BaselineEvidence) for item in self.evidence)
            or len({item.baseline_id for item in self.evidence}) != len(self.evidence)
        ):
            raise ValueError("sealed baseline evidence artifact is malformed")

    @classmethod
    def from_artifacts(
        cls,
        sealed_run: SealedTestRunAuthorization,
        baseline_artifacts: Sequence[object],
    ) -> SealedBaselineEvidenceArtifact:
        if not isinstance(sealed_run, SealedTestRunAuthorization):
            raise TypeError("sealed baseline evidence requires its authorization")
        evidence = _resolve_sealed_baseline_batch(
            sealed_run.plan,
            sealed_run.baseline_batch_commitment,
            tuple(baseline_artifacts),
        )
        return cls(
            sealed_run.plan.sha256,
            sealed_run.authorization.sha256,
            sealed_run.baseline_batch_commitment.sha256,
            evidence,
        )

    def assert_for(self, sealed_run: SealedTestRunAuthorization) -> None:
        expected = tuple(
            baseline_id
            for baseline_id, _policy_sha256 in (
                sealed_run.baseline_batch_commitment.required_baselines
            )
        )
        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or self.plan_sha256 != sealed_run.plan.sha256
            or self.authorization_sha256 != sealed_run.authorization.sha256
            or self.commitment_sha256 != sealed_run.baseline_batch_commitment.sha256
            or tuple(item.baseline_id for item in self.evidence) != expected
            or not baseline_requirements_pass(
                sealed_run.plan,
                self.evidence,
                expected_suite_sha256=(
                    sealed_run.baseline_batch_commitment.test_suite_sha256
                ),
            )
        ):
            raise ValueError("sealed baseline evidence references another run")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "plan_sha256": self.plan_sha256,
            "authorization_sha256": self.authorization_sha256,
            "commitment_sha256": self.commitment_sha256,
            "evidence": [item.manifest() for item in self.evidence],
        }

    @classmethod
    def from_manifest(
        cls,
        value: object,
        *,
        sealed_run: SealedTestRunAuthorization,
    ) -> SealedBaselineEvidenceArtifact:
        manifest = _require_keys(
            value,
            {
                "version",
                "plan_sha256",
                "authorization_sha256",
                "commitment_sha256",
                "evidence",
            },
            location="sealed baseline evidence artifact",
        )
        evidence = _manifest_list(
            manifest["evidence"], location="sealed baseline evidence"
        )
        if any(
            type(manifest[name]) is not str
            for name in (
                "version",
                "plan_sha256",
                "authorization_sha256",
                "commitment_sha256",
            )
        ):
            raise ValueError("sealed baseline evidence fields are malformed")
        try:
            result = cls(
                manifest["plan_sha256"],  # type: ignore[arg-type]
                manifest["authorization_sha256"],  # type: ignore[arg-type]
                manifest["commitment_sha256"],  # type: ignore[arg-type]
                tuple(BaselineEvidence.from_manifest(item) for item in evidence),
                manifest["version"],  # type: ignore[arg-type]
            )
            result.assert_for(sealed_run)
        except (TypeError, ValueError) as exc:
            raise ValueError("sealed baseline evidence artifact is malformed") from exc
        _require_manifest_round_trip(
            manifest,
            result.manifest(),
            location="sealed baseline evidence artifact",
        )
        return result

    def publish(self, store: object) -> str:
        from .r3b_artifacts import ArtifactStore

        if not isinstance(store, ArtifactStore):
            raise TypeError("sealed baseline evidence requires an ArtifactStore")
        return store.publish(
            kind=_SEALED_BASELINE_EVIDENCE_KIND,
            version=self.version,
            payload=self.manifest(),
        ).artifact_id

    @classmethod
    def load_committed(
        cls,
        store: object,
        artifact_id: str,
        *,
        ledger: SealedTestLedger,
        sealed_run: SealedTestRunAuthorization,
    ) -> CommittedBaselineEvidence:
        from .r3b_artifacts import ArtifactStore

        if not isinstance(store, ArtifactStore):
            raise TypeError("sealed baseline evidence requires an ArtifactStore")
        envelope = store.load(
            artifact_id,
            expected_kind=_SEALED_BASELINE_EVIDENCE_KIND,
            expected_version="r3b-sealed-baseline-evidence-artifact-v1",
        )
        result = cls.from_manifest(envelope.payload, sealed_run=sealed_run)
        if not ledger.verify_completed_baseline_batch(
            sealed_run, tuple(item.sha256 for item in result.evidence)
        ):
            raise RuntimeError("baseline evidence is not committed in the ledger")
        return CommittedBaselineEvidence(
            result.plan_sha256,
            result.authorization_sha256,
            result.commitment_sha256,
            result.evidence,
            _COMMITTED_BASELINE_EVIDENCE_TOKEN,
        )


@dataclass(frozen=True, slots=True)
class SealedLearnerOutcomeReference:
    """Content-addressed link from a ledger job to its rich outcome package."""

    plan_sha256: str
    authorization_sha256: str
    job_sha256: str
    learner_outcome_sha256: str
    output_artifact_sha256: str
    version: str = "r3b-sealed-learner-outcome-reference-v1"

    def __post_init__(self) -> None:
        if self.version != "r3b-sealed-learner-outcome-reference-v1" or any(
            not _is_nonzero_sha256(value)
            for value in (
                self.plan_sha256,
                self.authorization_sha256,
                self.job_sha256,
                self.learner_outcome_sha256,
                self.output_artifact_sha256,
            )
        ):
            raise ValueError("sealed learner-outcome reference is malformed")

    @classmethod
    def capture(
        cls,
        sealed_run: SealedTestRunAuthorization,
        job: TrialJob,
        outcome: LearnerOutcome,
        output_artifact_sha256: str,
    ) -> SealedLearnerOutcomeReference:
        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or job not in sealed_run.plan.trial_jobs("test", sealed_run)
            or not isinstance(outcome, LearnerOutcome)
            or outcome.engineering_evidence is None
            or outcome.engineering_evidence.job_sha256 != job.sha256
            or outcome.engineering_evidence.authorization_sha256
            != sealed_run.authorization.sha256
        ):
            raise ValueError("sealed learner outcome differs from its authorization")
        return cls(
            sealed_run.plan.sha256,
            sealed_run.authorization.sha256,
            job.sha256,
            outcome.sha256,
            output_artifact_sha256,
        )

    def manifest(self) -> dict[str, str]:
        return asdict(self)

    @classmethod
    def from_manifest(cls, value: object) -> SealedLearnerOutcomeReference:
        expected = {
            "version",
            "plan_sha256",
            "authorization_sha256",
            "job_sha256",
            "learner_outcome_sha256",
            "output_artifact_sha256",
        }
        manifest = _require_keys(
            value, expected, location="sealed learner-outcome reference"
        )
        if any(type(manifest[name]) is not str for name in expected):
            raise ValueError("sealed learner-outcome reference fields are malformed")
        try:
            result = cls(**manifest)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("sealed learner-outcome reference is malformed") from exc
        _require_manifest_round_trip(
            manifest,
            result.manifest(),
            location="sealed learner-outcome reference",
        )
        return result

    def publish(self, store: object) -> str:
        from .r3b_artifacts import ArtifactStore

        if not isinstance(store, ArtifactStore):
            raise TypeError("sealed learner outcome requires an ArtifactStore")
        return store.publish(
            kind=_SEALED_OUTCOME_REFERENCE_KIND,
            version=self.version,
            payload=self.manifest(),
        ).artifact_id

    @classmethod
    def load(cls, store: object, artifact_id: str) -> SealedLearnerOutcomeReference:
        from .r3b_artifacts import ArtifactStore

        if not isinstance(store, ArtifactStore):
            raise TypeError("sealed learner outcome requires an ArtifactStore")
        envelope = store.load(
            artifact_id,
            expected_kind=_SEALED_OUTCOME_REFERENCE_KIND,
            expected_version="r3b-sealed-learner-outcome-reference-v1",
        )
        return cls.from_manifest(envelope.payload)

    def resolve(
        self,
        *,
        ledger: SealedTestLedger,
        sealed_run: SealedTestRunAuthorization,
        job: TrialJob,
        loader: Callable[[str], LearnerOutcome],
    ) -> LearnerOutcome:
        if (
            self.plan_sha256 != sealed_run.plan.sha256
            or self.authorization_sha256 != sealed_run.authorization.sha256
            or self.job_sha256 != job.sha256
        ):
            raise ValueError("sealed learner-outcome reference belongs elsewhere")
        outcome = loader(self.output_artifact_sha256)
        if (
            not isinstance(outcome, LearnerOutcome)
            or outcome.sha256 != self.learner_outcome_sha256
            or outcome.engineering_evidence is None
            or outcome.engineering_evidence.job_sha256 != job.sha256
            or not ledger.verify_completed_job(sealed_run, job, outcome.sha256)
        ):
            raise RuntimeError("sealed learner outcome is not ledger-authorized")
        return outcome


def _sealed_job_sha256s(
    plan: R3BExperimentPlan, authorization: SealedTestAuthorization
) -> tuple[str, ...]:
    indexed = {arm.arm_id: arm for arm in plan.arms}
    arms = (
        indexed[authorization.control_arm_id],
        indexed[authorization.candidate_arm_id],
    )
    return tuple(
        TrialJob(
            plan.sha256,
            "test",
            arm,
            seed,
            plan.test_updates,
            True,
            TrialSeedPlan.derive(plan.sha256, seed).sha256,
            authorization.sha256,
        ).sha256
        for arm in arms
        for seed in plan.test_learner_seeds
    )


@dataclass(frozen=True, slots=True)
class SealedTestJobLease:
    """Opaque, restart-safe permission to execute exactly one sealed job."""

    sealed_run: SealedTestRunAuthorization
    job: TrialJob
    lease_token: str
    version: str = "r3b-sealed-test-job-lease-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-sealed-test-job-lease-v1"
            or not isinstance(self.sealed_run, SealedTestRunAuthorization)
            or not isinstance(self.job, TrialJob)
            or self.job.phase != "test"
            or self.job.authorization_sha256 != self.sealed_run.sha256
            or self.job.sha256
            not in _sealed_job_sha256s(
                self.sealed_run.plan, self.sealed_run.authorization
            )
            or not _is_nonzero_sha256(self.lease_token)
        ):
            raise ValueError("sealed-test job lease is malformed")
        self.assert_active()

    def assert_active(self) -> None:
        self._assert_state({"leased", "running"})

    def assert_running(self) -> None:
        self._assert_state({"running"})

    def _assert_state(self, allowed: set[str]) -> None:
        self.sealed_run.assert_authorized()
        try:
            with closing(
                _connect_private_ledger(self.sealed_run.ledger_path)
            ) as connection:
                row = connection.execute(
                    "SELECT state,lease_token_sha256 FROM sealed_test_job "
                    "WHERE plan_sha256=? AND job_sha256=?",
                    (self.sealed_run.plan.sha256, self.job.sha256),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError("sealed-test job ledger is unavailable") from exc
        if (
            row is None
            or row[0] not in allowed
            or row[1] != _bearer_sha256(self.lease_token)
        ):
            raise RuntimeError("sealed-test job lease is not active")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "version": self.version,
                "authorization_sha256": self.sealed_run.authorization.sha256,
                "job_sha256": self.job.sha256,
                "lease_token_sha256": hashlib.sha256(
                    self.lease_token.encode()
                ).hexdigest(),
            }
        )


@dataclass(frozen=True, slots=True)
class SealedBaselineBatchLease:
    """Opaque permission to execute the required baselines exactly once."""

    sealed_run: SealedTestRunAuthorization
    lease_token: str
    version: str = "r3b-sealed-baseline-batch-lease-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-sealed-baseline-batch-lease-v1"
            or not isinstance(self.sealed_run, SealedTestRunAuthorization)
            or not _is_nonzero_sha256(self.lease_token)
        ):
            raise ValueError("sealed baseline-batch lease is malformed")
        self.assert_active()

    def assert_active(self) -> None:
        self._assert_state({"leased", "running"})

    def assert_running(self) -> None:
        self._assert_state({"running"})

    def _assert_state(self, allowed: set[str]) -> None:
        self.sealed_run.assert_authorized()
        try:
            with closing(
                _connect_private_ledger(self.sealed_run.ledger_path)
            ) as connection:
                row = connection.execute(
                    "SELECT state,lease_token_sha256 FROM sealed_baseline_batch "
                    "WHERE plan_sha256=? AND commitment_sha256=?",
                    (
                        self.sealed_run.plan.sha256,
                        self.sealed_run.baseline_batch_commitment.sha256,
                    ),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError("sealed baseline-batch ledger is unavailable") from exc
        if (
            row is None
            or row[0] not in allowed
            or row[1] != _bearer_sha256(self.lease_token)
        ):
            raise RuntimeError("sealed baseline-batch lease is not active")

    @property
    def sha256(self) -> str:
        return _canonical_sha256(
            {
                "version": self.version,
                "authorization_sha256": self.sealed_run.authorization.sha256,
                "commitment_sha256": (self.sealed_run.baseline_batch_commitment.sha256),
                "lease_token_sha256": hashlib.sha256(
                    self.lease_token.encode()
                ).hexdigest(),
            }
        )


@dataclass(frozen=True, slots=True)
class SealedTestJobRecord:
    job_sha256: str
    arm_id: str
    learner_seed: int
    state: str
    outcome_sha256: str | None
    failure_reason: str | None

    def __post_init__(self) -> None:
        if (
            not _is_nonzero_sha256(self.job_sha256)
            or not self.arm_id
            or isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or not 0 <= self.learner_seed < 2**64
            or self.state not in {"complete", "failure"}
            or (
                self.state == "complete"
                and (
                    not _is_nonzero_sha256(self.outcome_sha256)
                    or self.failure_reason is not None
                )
            )
            or (
                self.state == "failure"
                and (
                    self.outcome_sha256 is not None
                    or not isinstance(self.failure_reason, str)
                    or not self.failure_reason
                )
            )
        ):
            raise ValueError("sealed-test job record is malformed")


class SealedTestLedger:
    """Restart-safe single-attempt ledger for a precommitted test suite."""

    version = "r3b-sealed-test-ledger-v3"
    _SCHEMA = {
        "sealed_test_attempt": {
            "plan_sha256",
            "version",
            "test_suite_sha256",
            "ledger_nonce",
            "state",
            "validation_run_sha256",
            "validation_results_sha256",
            "validation_suite_sha256",
            "authorization_sha256",
            "jobs_sha256",
            "baseline_batch_sha256",
            "receipt_token_sha256",
            "confirmation_report_sha256",
            "accepted",
        },
        "sealed_test_job": {
            "plan_sha256",
            "job_sha256",
            "arm_id",
            "learner_seed",
            "state",
            "lease_token_sha256",
            "outcome_sha256",
            "failure_reason",
        },
        "sealed_baseline_batch": {
            "plan_sha256",
            "commitment_sha256",
            "state",
            "lease_token_sha256",
            "evidence_sha256s",
            "failure_reason",
        },
    }

    def __init__(self, path: str | Path) -> None:
        supplied = Path(path).expanduser()
        if not supplied.is_absolute():
            raise ValueError("sealed-test ledger path must be absolute")
        self.path = supplied
        self._schema_ready = False
        current = Path(self.path.anchor)
        for component in self.path.parent.parts[1:]:
            current /= component
            if current.is_symlink():
                raise ValueError("sealed-test ledger path crosses a symbolic link")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parent = self.path.parent.lstat()
        if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.geteuid():
            raise ValueError("sealed-test ledger parent must be an owned directory")
        os.chmod(self.path.parent, 0o700)
        if self.path.is_symlink():
            raise ValueError("sealed-test ledger cannot be a symbolic link")
        if not self.path.exists():
            descriptor = os.open(
                self.path,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            os.close(descriptor)
        _assert_private_ledger_path(self.path)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sealed_test_attempt (
                    plan_sha256 TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    test_suite_sha256 TEXT NOT NULL,
                    ledger_nonce TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('ready','authorized','finalized')),
                    validation_run_sha256 TEXT,
                    validation_results_sha256 TEXT,
                    validation_suite_sha256 TEXT,
                    authorization_sha256 TEXT,
                    jobs_sha256 TEXT,
                    baseline_batch_sha256 TEXT,
                    receipt_token_sha256 TEXT UNIQUE,
                    confirmation_report_sha256 TEXT,
                    accepted INTEGER
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sealed_test_job (
                    plan_sha256 TEXT NOT NULL,
                    job_sha256 TEXT NOT NULL,
                    arm_id TEXT NOT NULL,
                    learner_seed TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(
                        state IN ('pending','leased','running','complete','failure')
                    ),
                    lease_token_sha256 TEXT UNIQUE,
                    outcome_sha256 TEXT,
                    failure_reason TEXT,
                    PRIMARY KEY(plan_sha256, job_sha256),
                    UNIQUE(plan_sha256, arm_id, learner_seed),
                    FOREIGN KEY(plan_sha256) REFERENCES sealed_test_attempt(plan_sha256)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS sealed_baseline_batch (
                    plan_sha256 TEXT PRIMARY KEY,
                    commitment_sha256 TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(
                        state IN ('pending','leased','running','complete','failure')
                    ),
                    lease_token_sha256 TEXT UNIQUE,
                    evidence_sha256s TEXT,
                    failure_reason TEXT,
                    FOREIGN KEY(plan_sha256)
                        REFERENCES sealed_test_attempt(plan_sha256)
                )
                """
            )
            connection.commit()
            self._verify_connection(connection)
            self._schema_ready = True

    def _connect(self) -> sqlite3.Connection:
        connection = _connect_private_ledger(self.path)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        if self._schema_ready:
            self._verify_connection(connection)
        return connection

    def _verify_connection(self, connection: sqlite3.Connection) -> None:
        tables = {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            ).fetchall()
        }
        columns = {
            table: {
                str(row[1])
                for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for table in self._SCHEMA
        }
        versions = {
            str(row[0])
            for row in connection.execute(
                "SELECT DISTINCT version FROM sealed_test_attempt"
            ).fetchall()
        }
        if (
            tables != set(self._SCHEMA)
            or columns != self._SCHEMA
            or versions - {self.version}
            or connection.execute("PRAGMA foreign_key_check").fetchall()
        ):
            raise RuntimeError(
                "sealed-test ledger schema, version, or referential integrity "
                "is unsupported; in-place migrations are forbidden"
            )
        attempts = connection.execute(
            "SELECT state,validation_run_sha256,validation_results_sha256,"
            "validation_suite_sha256,authorization_sha256,jobs_sha256,"
            "baseline_batch_sha256,receipt_token_sha256,"
            "confirmation_report_sha256,accepted FROM sealed_test_attempt"
        ).fetchall()
        for row in attempts:
            state, *fields = row
            authorization_fields = fields[:7]
            confirmation_fields = fields[7:]
            if (
                (state == "ready" and any(value is not None for value in fields))
                or (
                    state == "authorized"
                    and (
                        any(value is None for value in authorization_fields)
                        or any(value is not None for value in confirmation_fields)
                    )
                )
                or (
                    state == "finalized"
                    and (
                        any(value is None for value in fields)
                        or confirmation_fields[1] not in (0, 1)
                    )
                )
            ):
                raise RuntimeError("sealed-test attempt state is inconsistent")
        for state, token, outcome, failure in connection.execute(
            "SELECT state,lease_token_sha256,outcome_sha256,failure_reason "
            "FROM sealed_test_job"
        ):
            if (
                (state == "pending" and any((token, outcome, failure)))
                or (
                    state in {"leased", "running"}
                    and (token is None or outcome or failure)
                )
                or (
                    state == "complete"
                    and (token is None or outcome is None or failure)
                )
                or (state == "failure" and (token is None or outcome or not failure))
            ):
                raise RuntimeError("sealed-test job state is inconsistent")
        for state, token, evidence, failure in connection.execute(
            "SELECT state,lease_token_sha256,evidence_sha256s,failure_reason "
            "FROM sealed_baseline_batch"
        ):
            if (
                (state == "pending" and any((token, evidence, failure)))
                or (
                    state in {"leased", "running"}
                    and (token is None or evidence or failure)
                )
                or (
                    state == "complete"
                    and (token is None or evidence is None or failure)
                )
                or (state == "failure" and (token is None or evidence or not failure))
            ):
                raise RuntimeError("sealed baseline-batch state is inconsistent")

    def verify(self) -> None:
        """Fail closed on path, SQLite, schema, and state corruption."""

        with closing(self._connect()) as connection:
            self._verify_connection(connection)

    def precommit(
        self, plan: R3BExperimentPlan, test_suite: object
    ) -> TestSuiteCommitment:
        from .r3b_evaluation import EvaluationSuite

        if (
            not isinstance(plan, R3BExperimentPlan)
            or not isinstance(test_suite, EvaluationSuite)
            or test_suite.split != "test"
            or test_suite.policy_seed != plan.test_evaluation_seed
            or len(test_suite.snapshot_ids) * test_suite.repetitions
            != plan.test_episodes_per_policy
        ):
            raise ValueError("test-suite precommit violates the frozen plan")
        nonce = secrets.token_hex(32)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT version,test_suite_sha256,ledger_nonce FROM sealed_test_attempt "
                "WHERE plan_sha256=?",
                (plan.sha256,),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO sealed_test_attempt "
                    "(plan_sha256,version,test_suite_sha256,ledger_nonce,state) "
                    "VALUES (?,?,?,?, 'ready')",
                    (plan.sha256, self.version, test_suite.sha256, nonce),
                )
            else:
                if row[0] != self.version or row[1] != test_suite.sha256:
                    raise RuntimeError(
                        "sealed-test ledger already commits another suite or version"
                    )
                nonce = str(row[2])
            connection.commit()
        return TestSuiteCommitment(plan.sha256, test_suite.sha256, nonce)

    def authorize_once(
        self,
        plan: R3BExperimentPlan,
        validation_run: ValidationRunAuthorization,
        validation_results: Sequence[ArmPhaseResult],
        validation_suite: object,
        test_suite: object,
    ) -> SealedTestRunAuthorization:
        results = tuple(validation_results)
        authorization = _authorize_sealed_test(
            plan,
            validation_run,
            results,
            validation_suite,
            test_suite,
        )
        result_sha256 = _canonical_sha256(
            [
                result.sha256
                for result in sorted(results, key=lambda value: value.arm_id)
            ]
        )
        jobs = tuple(
            TrialJob(
                plan.sha256,
                "test",
                arm,
                seed,
                plan.test_updates,
                True,
                TrialSeedPlan.derive(plan.sha256, seed).sha256,
                authorization.sha256,
            )
            for arm in (
                next(
                    value
                    for value in plan.arms
                    if value.arm_id == authorization.control_arm_id
                ),
                next(
                    value
                    for value in plan.arms
                    if value.arm_id == authorization.candidate_arm_id
                ),
            )
            for seed in plan.test_learner_seeds
        )
        jobs_sha256 = _canonical_sha256([job.sha256 for job in jobs])
        baseline_commitment = SealedBaselineBatchCommitment.from_plan(plan, test_suite)
        if authorization.baseline_batch_commitment_sha256 != baseline_commitment.sha256:
            raise RuntimeError("sealed baseline-batch commitment disagrees")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT test_suite_sha256,ledger_nonce,state,validation_run_sha256,"
                "validation_results_sha256,validation_suite_sha256,authorization_sha256,"
                "jobs_sha256,baseline_batch_sha256,receipt_token_sha256 "
                "FROM sealed_test_attempt WHERE plan_sha256=?",
                (plan.sha256,),
            ).fetchone()
            if row is None:
                raise RuntimeError("sealed test suite was not precommitted")
            if (
                row[0] != test_suite.sha256
                or row[1] != validation_run.test_commitment.ledger_nonce
            ):
                raise RuntimeError("sealed test does not match its durable commitment")
            expected = (
                validation_run.sha256,
                result_sha256,
                validation_suite.sha256,
                authorization.sha256,
                jobs_sha256,
                baseline_commitment.sha256,
            )
            receipt_token = _canonical_sha256(
                {
                    "domain": "r3b-sealed-test-receipt-v1",
                    "plan_sha256": plan.sha256,
                    "ledger_nonce": row[1],
                    "authorization_sha256": authorization.sha256,
                    "jobs_sha256": jobs_sha256,
                    "baseline_batch_sha256": baseline_commitment.sha256,
                }
            )
            receipt_token_sha256 = _bearer_sha256(receipt_token)
            if row[2] == "ready":
                updated = connection.execute(
                    "UPDATE sealed_test_attempt SET state='authorized',"
                    "validation_run_sha256=?,validation_results_sha256=?,"
                    "validation_suite_sha256=?,authorization_sha256=?,jobs_sha256=?,"
                    "baseline_batch_sha256=?,receipt_token_sha256=? "
                    "WHERE plan_sha256=? AND state='ready'",
                    (*expected, receipt_token_sha256, plan.sha256),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("sealed-test authorization race was lost")
                connection.executemany(
                    "INSERT INTO sealed_test_job "
                    "(plan_sha256,job_sha256,arm_id,learner_seed,state) "
                    "VALUES (?,?,?,?, 'pending')",
                    (
                        (
                            plan.sha256,
                            job.sha256,
                            job.arm.arm_id,
                            str(job.learner_seed),
                        )
                        for job in jobs
                    ),
                )
                connection.execute(
                    "INSERT INTO sealed_baseline_batch "
                    "(plan_sha256,commitment_sha256,state) VALUES (?,?,'pending')",
                    (plan.sha256, baseline_commitment.sha256),
                )
            elif (
                row[2] != "authorized"
                or tuple(row[3:9]) != expected
                or row[9] != receipt_token_sha256
            ):
                raise RuntimeError("sealed-test attempt is consumed or disagrees")
            connection.commit()
        return _bind_sealed_test_run(
            plan,
            validation_run,
            results,
            validation_suite,
            test_suite,
            str(self.path),
            receipt_token,
        )

    def claim_baseline_batch(
        self,
        sealed_run: SealedTestRunAuthorization,
        *,
        lease_token: str | None = None,
    ) -> SealedBaselineBatchLease:
        """Atomically claim the sole required-baseline execution."""

        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
        ):
            raise ValueError("sealed baseline batch belongs to another ledger")
        sealed_run.assert_authorized()
        token = secrets.token_hex(32) if lease_token is None else lease_token
        if not _is_nonzero_sha256(token):
            raise ValueError("sealed baseline lease token must be a nonzero SHA-256")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE sealed_baseline_batch SET state='leased',lease_token_sha256=? "
                "WHERE plan_sha256=? AND commitment_sha256=? AND state='pending'",
                (
                    _bearer_sha256(token),
                    sealed_run.plan.sha256,
                    sealed_run.baseline_batch_commitment.sha256,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError(
                    "sealed baseline batch is already claimed or terminal"
                )
            connection.commit()
        return SealedBaselineBatchLease(sealed_run, token)

    def resume_baseline_batch(
        self,
        sealed_run: SealedTestRunAuthorization,
        *,
        lease_token: str,
    ) -> SealedBaselineBatchLease:
        """Recover only an unstarted batch lease held by the same bearer."""

        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
            or not _is_nonzero_sha256(lease_token)
        ):
            raise ValueError("sealed baseline-batch lease recovery is malformed")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state,lease_token_sha256 FROM sealed_baseline_batch "
                "WHERE plan_sha256=? AND commitment_sha256=?",
                (
                    sealed_run.plan.sha256,
                    sealed_run.baseline_batch_commitment.sha256,
                ),
            ).fetchone()
        if row != ("leased", _bearer_sha256(lease_token)):
            raise RuntimeError("sealed baseline-batch recovery token is invalid")
        return SealedBaselineBatchLease(sealed_run, lease_token)

    def begin_baseline_batch(self, lease: SealedBaselineBatchLease) -> None:
        """Irrevocably consume a baseline-batch lease into execution."""

        if (
            not isinstance(lease, SealedBaselineBatchLease)
            or Path(lease.sealed_run.ledger_path) != self.path
        ):
            raise ValueError("sealed baseline-batch lease belongs to another ledger")
        lease._assert_state({"leased"})
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE sealed_baseline_batch SET state='running' "
                "WHERE plan_sha256=? AND commitment_sha256=? AND state='leased' "
                "AND lease_token_sha256=?",
                (
                    lease.sealed_run.plan.sha256,
                    lease.sealed_run.baseline_batch_commitment.sha256,
                    _bearer_sha256(lease.lease_token),
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("sealed baseline-batch lease is stale or consumed")
            connection.commit()

    def complete_baseline_batch(
        self,
        lease: SealedBaselineBatchLease,
        baseline_artifacts: Sequence[object],
    ) -> tuple[BaselineEvidence, ...]:
        """Persist exact evidence identities for the sole successful batch."""

        if (
            not isinstance(lease, SealedBaselineBatchLease)
            or Path(lease.sealed_run.ledger_path) != self.path
        ):
            raise ValueError("sealed baseline-batch lease belongs to another ledger")
        evidence = _resolve_sealed_baseline_batch(
            lease.sealed_run.plan,
            lease.sealed_run.baseline_batch_commitment,
            tuple(baseline_artifacts),
        )
        evidence_sha256s = json.dumps(
            [item.sha256 for item in evidence],
            separators=(",", ":"),
        )
        lease.assert_running()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE sealed_baseline_batch SET state='complete',"
                "evidence_sha256s=? WHERE plan_sha256=? AND commitment_sha256=? "
                "AND state='running' AND lease_token_sha256=?",
                (
                    evidence_sha256s,
                    lease.sealed_run.plan.sha256,
                    lease.sealed_run.baseline_batch_commitment.sha256,
                    _bearer_sha256(lease.lease_token),
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("sealed baseline-batch lease is stale or consumed")
            connection.commit()
        return evidence

    def fail_baseline_batch(
        self, lease: SealedBaselineBatchLease, failure_reason: str
    ) -> None:
        """Terminalize a started baseline execution without allowing a retry."""

        if (
            not isinstance(lease, SealedBaselineBatchLease)
            or Path(lease.sealed_run.ledger_path) != self.path
            or not isinstance(failure_reason, str)
            or not failure_reason.strip()
        ):
            raise ValueError("sealed baseline-batch failure record is malformed")
        lease.assert_running()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE sealed_baseline_batch SET state='failure',failure_reason=? "
                "WHERE plan_sha256=? AND commitment_sha256=? AND state='running' "
                "AND lease_token_sha256=?",
                (
                    failure_reason.strip(),
                    lease.sealed_run.plan.sha256,
                    lease.sealed_run.baseline_batch_commitment.sha256,
                    _bearer_sha256(lease.lease_token),
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("sealed baseline-batch lease is stale or consumed")
            connection.commit()

    def claim_job(
        self,
        sealed_run: SealedTestRunAuthorization,
        job: TrialJob,
        *,
        lease_token: str | None = None,
    ) -> SealedTestJobLease:
        """Atomically lease one pending test job; a lease cannot be reassigned."""

        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
            or job not in sealed_run.plan.trial_jobs("test", sealed_run)
        ):
            raise ValueError("sealed-test job does not belong to this ledger run")
        token = secrets.token_hex(32) if lease_token is None else lease_token
        if not _is_nonzero_sha256(token):
            raise ValueError("sealed-test lease token must be a nonzero SHA-256")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state,lease_token_sha256 FROM sealed_test_job "
                "WHERE plan_sha256=? AND job_sha256=?",
                (sealed_run.plan.sha256, job.sha256),
            ).fetchone()
            if row is None:
                raise RuntimeError("sealed-test job is absent from the authorization")
            if row[0] == "pending":
                updated = connection.execute(
                    "UPDATE sealed_test_job SET state='leased',lease_token_sha256=? "
                    "WHERE plan_sha256=? AND job_sha256=? AND state='pending'",
                    (
                        _bearer_sha256(token),
                        sealed_run.plan.sha256,
                        job.sha256,
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("sealed-test job claim race was lost")
            else:
                raise RuntimeError("sealed-test job is already claimed or terminal")
            connection.commit()
        return SealedTestJobLease(sealed_run, job, token)

    def resume_job(
        self,
        sealed_run: SealedTestRunAuthorization,
        job: TrialJob,
        *,
        lease_token: str,
    ) -> SealedTestJobLease:
        """Recover an unstarted lease only when its bearer presents the token."""

        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
            or not isinstance(job, TrialJob)
            or not _is_nonzero_sha256(lease_token)
        ):
            raise ValueError("sealed-test lease recovery is malformed")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state,lease_token_sha256 FROM sealed_test_job "
                "WHERE plan_sha256=? AND job_sha256=?",
                (sealed_run.plan.sha256, job.sha256),
            ).fetchone()
        if row != ("leased", _bearer_sha256(lease_token)):
            raise RuntimeError("sealed-test lease recovery token is invalid")
        return SealedTestJobLease(sealed_run, job, lease_token)

    def begin_job(self, lease: SealedTestJobLease) -> None:
        """Consume a lease into one running execution."""

        if (
            not isinstance(lease, SealedTestJobLease)
            or Path(lease.sealed_run.ledger_path) != self.path
        ):
            raise ValueError("sealed-test lease belongs to another ledger")
        lease._assert_state({"leased"})
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE sealed_test_job SET state='running' "
                "WHERE plan_sha256=? AND job_sha256=? AND state='leased' "
                "AND lease_token_sha256=?",
                (
                    lease.sealed_run.plan.sha256,
                    lease.job.sha256,
                    _bearer_sha256(lease.lease_token),
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("sealed-test job lease is stale or consumed")
            connection.commit()

    def complete_job(self, lease: SealedTestJobLease, outcome: LearnerOutcome) -> None:
        """Persist the sole admissible result for one leased test job."""

        if (
            not isinstance(lease, SealedTestJobLease)
            or Path(lease.sealed_run.ledger_path) != self.path
            or not isinstance(outcome, LearnerOutcome)
            or outcome.learner_seed != lease.job.learner_seed
            or outcome.seed_plan_sha256 != lease.job.seed_plan_sha256
            or outcome.engineering_evidence is None
            or outcome.engineering_evidence.phase != "test"
            or outcome.engineering_evidence.job_sha256 != lease.job.sha256
            or outcome.engineering_evidence.arm_id != lease.job.arm.arm_id
            or outcome.engineering_evidence.authorization_sha256
            != lease.sealed_run.authorization.sha256
            or outcome.engineering_evidence.sealed_job_lease_sha256 != lease.sha256
            or outcome.engineering_evidence.completed_updates
            != lease.job.budget_updates
        ):
            raise ValueError("sealed-test outcome disagrees with its job lease")
        lease.assert_running()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE sealed_test_job SET state='complete',outcome_sha256=? "
                "WHERE plan_sha256=? AND job_sha256=? AND state='running' "
                "AND lease_token_sha256=?",
                (
                    outcome.sha256,
                    lease.sealed_run.plan.sha256,
                    lease.job.sha256,
                    _bearer_sha256(lease.lease_token),
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("sealed-test job lease is stale or consumed")
            connection.commit()

    def fail_job(self, lease: SealedTestJobLease, failure_reason: str) -> None:
        """Record a terminal execution failure without permitting a retry."""

        if (
            not isinstance(lease, SealedTestJobLease)
            or Path(lease.sealed_run.ledger_path) != self.path
            or not isinstance(failure_reason, str)
            or not failure_reason.strip()
        ):
            raise ValueError("sealed-test failure record is malformed")
        lease.assert_running()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            updated = connection.execute(
                "UPDATE sealed_test_job SET state='failure',failure_reason=? "
                "WHERE plan_sha256=? AND job_sha256=? AND state='running' "
                "AND lease_token_sha256=?",
                (
                    failure_reason.strip(),
                    lease.sealed_run.plan.sha256,
                    lease.job.sha256,
                    _bearer_sha256(lease.lease_token),
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("sealed-test job lease is stale or consumed")
            connection.commit()

    def terminalize_orphaned_execution(
        self,
        sealed_run: SealedTestRunAuthorization,
        *,
        failure_reason: str,
    ) -> int:
        """Reject started work whose bearer process was irrecoverably lost.

        This never makes a job retryable. It only converts durable ``running``
        rows to terminal failures so the one-shot attempt can be finalized.
        """

        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
            or not isinstance(failure_reason, str)
            or not failure_reason.strip()
        ):
            raise ValueError("sealed orphan reconciliation is malformed")
        sealed_run.assert_authorized()
        reason = failure_reason.strip()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            jobs = connection.execute(
                "UPDATE sealed_test_job SET state='failure',failure_reason=? "
                "WHERE plan_sha256=? AND state='running'",
                (reason, sealed_run.plan.sha256),
            ).rowcount
            baseline = connection.execute(
                "UPDATE sealed_baseline_batch SET state='failure',failure_reason=? "
                "WHERE plan_sha256=? AND state='running'",
                (reason, sealed_run.plan.sha256),
            ).rowcount
            connection.commit()
        return jobs + baseline

    def verify_completed_job(
        self,
        sealed_run: SealedTestRunAuthorization,
        job: TrialJob,
        outcome_sha256: str,
    ) -> bool:
        """Verify the immutable ledger result used by workflow reconciliation."""

        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
            or job not in sealed_run.plan.trial_jobs("test", sealed_run)
            or not _is_nonzero_sha256(outcome_sha256)
        ):
            return False
        sealed_run.assert_authorized()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state,outcome_sha256 FROM sealed_test_job "
                "WHERE plan_sha256=? AND job_sha256=?",
                (sealed_run.plan.sha256, job.sha256),
            ).fetchone()
        return row == ("complete", outcome_sha256)

    def job_state(
        self,
        sealed_run: SealedTestRunAuthorization,
        job: TrialJob,
    ) -> tuple[str, str | None, str | None]:
        """Return one ledger row for fail-closed process supervision."""

        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
            or job not in sealed_run.plan.trial_jobs("test", sealed_run)
        ):
            raise ValueError("sealed-test job belongs to another ledger")
        sealed_run.assert_authorized()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state,outcome_sha256,failure_reason "
                "FROM sealed_test_job WHERE plan_sha256=? AND job_sha256=?",
                (sealed_run.plan.sha256, job.sha256),
            ).fetchone()
        if row is None or row[0] not in {
            "pending",
            "leased",
            "running",
            "complete",
            "failure",
        }:
            raise RuntimeError("sealed-test job state is absent or malformed")
        return str(row[0]), row[1], row[2]

    def verify_failed_job(
        self,
        sealed_run: SealedTestRunAuthorization,
        job: TrialJob,
        failure_reason: str,
    ) -> bool:
        """Verify a terminal failure without making the sealed job retryable."""

        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
            or job not in sealed_run.plan.trial_jobs("test", sealed_run)
            or not isinstance(failure_reason, str)
            or not failure_reason
        ):
            return False
        sealed_run.assert_authorized()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state,failure_reason FROM sealed_test_job "
                "WHERE plan_sha256=? AND job_sha256=?",
                (sealed_run.plan.sha256, job.sha256),
            ).fetchone()
        return row == ("failure", failure_reason)

    def verify_completed_baseline_batch(
        self,
        sealed_run: SealedTestRunAuthorization,
        evidence_sha256s: Sequence[str],
    ) -> bool:
        """Verify the ordered evidence identities consumed by the one-shot batch."""

        identities = tuple(evidence_sha256s)
        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
            or not identities
            or any(not _is_nonzero_sha256(value) for value in identities)
        ):
            return False
        sealed_run.assert_authorized()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state,evidence_sha256s FROM sealed_baseline_batch "
                "WHERE plan_sha256=? AND commitment_sha256=?",
                (
                    sealed_run.plan.sha256,
                    sealed_run.baseline_batch_commitment.sha256,
                ),
            ).fetchone()
        return row == (
            "complete",
            json.dumps(list(identities), separators=(",", ":")),
        )

    def terminal_job_records(
        self, sealed_run: SealedTestRunAuthorization
    ) -> tuple[SealedTestJobRecord, ...]:
        """Return the complete immutable test result index, or fail if unfinished."""

        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
        ):
            raise ValueError("sealed-test result index belongs to another ledger")
        sealed_run.assert_authorized()
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT job_sha256,arm_id,learner_seed,state,outcome_sha256,"
                "failure_reason FROM sealed_test_job WHERE plan_sha256=? "
                "ORDER BY arm_id,learner_seed",
                (sealed_run.plan.sha256,),
            ).fetchall()
        expected = {
            job.sha256 for job in sealed_run.plan.trial_jobs("test", sealed_run)
        }
        if (
            len(rows) != len(expected)
            or {str(row[0]) for row in rows} != expected
            or any(row[3] not in {"complete", "failure"} for row in rows)
        ):
            raise RuntimeError("sealed-test jobs are not all terminal")
        return tuple(
            SealedTestJobRecord(
                str(row[0]),
                str(row[1]),
                int(row[2]),
                str(row[3]),
                None if row[4] is None else str(row[4]),
                None if row[5] is None else str(row[5]),
            )
            for row in rows
        )

    def finalized_confirmation_sha256(
        self, sealed_run: SealedTestRunAuthorization
    ) -> str | None:
        """Return the immutable report identity after finalization."""

        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
        ):
            raise ValueError("sealed confirmation belongs to another ledger")
        sealed_run.assert_authorized(allow_finalized=True)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state,confirmation_report_sha256 FROM sealed_test_attempt "
                "WHERE plan_sha256=?",
                (sealed_run.plan.sha256,),
            ).fetchone()
        if row is None:
            raise RuntimeError("sealed-test attempt is absent")
        return str(row[1]) if row[0] == "finalized" else None

    def finalize_once(
        self,
        sealed_run: SealedTestRunAuthorization,
        candidate_result: ArmPhaseResult | None,
        control_result: ArmPhaseResult | None,
        baseline_artifacts: Sequence[object],
    ) -> SealedConfirmationReport:
        """Recompute and persist the decision from exactly the recorded job results."""

        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
        ):
            raise ValueError("sealed-test finalization belongs to another ledger")
        sealed_run.assert_authorized()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            attempt = connection.execute(
                "SELECT state,authorization_sha256,receipt_token_sha256 "
                "FROM sealed_test_attempt "
                "WHERE plan_sha256=?",
                (sealed_run.plan.sha256,),
            ).fetchone()
            if attempt != (
                "authorized",
                sealed_run.authorization.sha256,
                _bearer_sha256(sealed_run.ledger_receipt_token),
            ):
                raise RuntimeError(
                    "sealed-test attempt is absent, mismatched, or consumed"
                )
            rows = connection.execute(
                "SELECT job_sha256,arm_id,learner_seed,state,outcome_sha256,"
                "failure_reason FROM sealed_test_job WHERE plan_sha256=? "
                "ORDER BY arm_id,learner_seed",
                (sealed_run.plan.sha256,),
            ).fetchall()
            expected_jobs = sealed_run.plan.trial_jobs("test", sealed_run)
            if (
                len(rows) != len(expected_jobs)
                or {row[0] for row in rows} != {job.sha256 for job in expected_jobs}
                or any(row[3] not in {"complete", "failure"} for row in rows)
            ):
                raise RuntimeError("sealed-test jobs are incomplete or disagree")
            baseline_row = connection.execute(
                "SELECT state,evidence_sha256s FROM sealed_baseline_batch "
                "WHERE plan_sha256=? AND commitment_sha256=?",
                (
                    sealed_run.plan.sha256,
                    sealed_run.baseline_batch_commitment.sha256,
                ),
            ).fetchone()
            if baseline_row is None or baseline_row[0] not in {
                "complete",
                "failure",
            }:
                raise RuntimeError("sealed baseline batch is absent or incomplete")
            if baseline_row[0] == "complete":
                baseline_evidence = _resolve_sealed_baseline_batch(
                    sealed_run.plan,
                    sealed_run.baseline_batch_commitment,
                    tuple(baseline_artifacts),
                )
                expected_evidence_sha256s = json.dumps(
                    [item.sha256 for item in baseline_evidence],
                    separators=(",", ":"),
                )
                if baseline_row[1] != expected_evidence_sha256s:
                    raise ValueError(
                        "supplied baseline evidence differs from the sealed batch"
                    )
            elif tuple(baseline_artifacts):
                raise ValueError("failed sealed baseline batch accepts no evidence")

            def recorded_result(
                arm_id: str, supplied: ArmPhaseResult | None
            ) -> ArmPhaseResult:
                arm_rows = [row for row in rows if row[1] == arm_id]
                failures = tuple(
                    (str(row[0]), str(row[5]))
                    for row in arm_rows
                    if row[3] == "failure"
                )
                if failures:
                    reason = "; ".join(
                        f"{job_sha256}:{message}" for job_sha256, message in failures
                    )
                    return ArmPhaseResult(
                        arm_id,
                        "test",
                        "post_start_failure",
                        sealed_run.plan.test_updates,
                        failure_reason=reason,
                    )
                if (
                    not isinstance(supplied, ArmPhaseResult)
                    or supplied.arm_id != arm_id
                    or supplied.phase != "test"
                    or supplied.status != "complete"
                    or supplied.budget_updates != sealed_run.plan.test_updates
                ):
                    raise ValueError("complete sealed jobs require their typed result")
                recorded = {str(row[0]): str(row[4]) for row in arm_rows}
                presented = {
                    outcome.engineering_evidence.job_sha256: outcome.sha256
                    for outcome in supplied.outcomes
                    if outcome.engineering_evidence is not None
                }
                if recorded != presented:
                    raise ValueError(
                        "sealed result does not exactly match persisted job outcomes"
                    )
                return supplied

            candidate = recorded_result(
                sealed_run.authorization.candidate_arm_id, candidate_result
            )
            control = recorded_result(
                sealed_run.authorization.control_arm_id, control_result
            )
            report = build_sealed_confirmation_report(
                sealed_run.plan,
                sealed_run,
                candidate,
                control,
                tuple(baseline_artifacts),
            )
            updated = connection.execute(
                "UPDATE sealed_test_attempt SET state='finalized',"
                "confirmation_report_sha256=?,accepted=? "
                "WHERE plan_sha256=? AND state='authorized'",
                (report.sha256, int(report.accepted), sealed_run.plan.sha256),
            )
            if updated.rowcount != 1:
                raise RuntimeError("sealed-test finalization race was lost")
            connection.commit()
        return report

    def verify_finalized(self, report: SealedConfirmationReport) -> bool:
        if not isinstance(report, SealedConfirmationReport):
            return False
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT state,confirmation_report_sha256,accepted "
                "FROM sealed_test_attempt WHERE plan_sha256=?",
                (report.plan_sha256,),
            ).fetchone()
        return row == ("finalized", report.sha256, int(report.accepted))


@dataclass(frozen=True, slots=True)
class ConfirmationDecision:
    accepted: bool
    gates: tuple[tuple[str, bool], ...]
    relative_auc_gain_lower: float | None = None
    final_mean_retention_lower: float | None = None
    final_mean_mode: str | None = None
    p10_lower: float | None = None
    p10_mode: str | None = None
    trivial_baseline_margin_lower: float | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.accepted, bool)
            or not self.gates
            or not isinstance(self.gates, tuple)
            or any(
                not isinstance(gate, tuple)
                or len(gate) != 2
                or not isinstance(gate[0], str)
                or not gate[0]
                or not isinstance(gate[1], bool)
                for gate in self.gates
            )
            or len({name for name, _ in self.gates}) != len(self.gates)
        ):
            raise ValueError("confirmation gates must be immutable and uniquely named")
        for value in (
            self.relative_auc_gain_lower,
            self.final_mean_retention_lower,
            self.p10_lower,
            self.trivial_baseline_margin_lower,
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError("confirmation bounds must be finite when present")
        if self.accepted != all(passed for _, passed in self.gates):
            raise ValueError("acceptance must equal the conjunction of all gates")
        if self.p10_mode not in {None, "ratio", "absolute_delta"} or (
            self.final_mean_mode not in {None, "ratio", "absolute_delta"}
        ):
            raise ValueError("confirmation comparison mode is invalid")

    def manifest(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "gates": {name: passed for name, passed in self.gates},
            "relative_auc_gain_lower": self.relative_auc_gain_lower,
            "final_mean_retention_lower": self.final_mean_retention_lower,
            "final_mean_mode": self.final_mean_mode,
            "p10_lower": self.p10_lower,
            "p10_mode": self.p10_mode,
            "trivial_baseline_margin_lower": self.trivial_baseline_margin_lower,
        }

    @classmethod
    def from_manifest(cls, value: object) -> ConfirmationDecision:
        manifest = _require_keys(
            value,
            {
                "accepted",
                "gates",
                "relative_auc_gain_lower",
                "final_mean_retention_lower",
                "final_mean_mode",
                "p10_lower",
                "p10_mode",
                "trivial_baseline_margin_lower",
            },
            location="confirmation decision",
        )
        gates = manifest["gates"]
        if (
            type(manifest["accepted"]) is not bool
            or not isinstance(gates, Mapping)
            or not gates
            or any(
                type(name) is not str or not name or type(passed) is not bool
                for name, passed in gates.items()
            )
            or any(name not in _CONFIRMATION_GATE_ORDER for name in gates)
        ):
            raise ValueError("confirmation decision fields are malformed")
        for name in (
            "relative_auc_gain_lower",
            "final_mean_retention_lower",
            "p10_lower",
            "trivial_baseline_margin_lower",
        ):
            if manifest[name] is not None and type(manifest[name]) is not float:
                raise ValueError("confirmation decision bounds must be floats or null")
        for name in ("final_mean_mode", "p10_mode"):
            if manifest[name] is not None and type(manifest[name]) is not str:
                raise ValueError("confirmation decision modes must be strings or null")
        try:
            result = cls(
                accepted=manifest["accepted"],  # type: ignore[arg-type]
                gates=tuple(
                    (name, gates[name])
                    for name in _CONFIRMATION_GATE_ORDER
                    if name in gates
                ),  # type: ignore[arg-type]
                relative_auc_gain_lower=manifest["relative_auc_gain_lower"],  # type: ignore[arg-type]
                final_mean_retention_lower=manifest["final_mean_retention_lower"],  # type: ignore[arg-type]
                final_mean_mode=manifest["final_mean_mode"],  # type: ignore[arg-type]
                p10_lower=manifest["p10_lower"],  # type: ignore[arg-type]
                p10_mode=manifest["p10_mode"],  # type: ignore[arg-type]
                trivial_baseline_margin_lower=manifest["trivial_baseline_margin_lower"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("confirmation decision is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="confirmation decision"
        )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class SealedConfirmationReport:
    plan_sha256: str
    authorization_sha256: str
    ledger_receipt_sha256: str
    candidate_arm_id: str
    control_arm_id: str
    candidate_result_sha256: str
    control_result_sha256: str
    baseline_evidence_sha256: tuple[str, ...]
    decision: ConfirmationDecision
    version: str = "r3b-sealed-confirmation-report-v3"

    def __post_init__(self) -> None:
        hashes = (
            self.plan_sha256,
            self.authorization_sha256,
            self.ledger_receipt_sha256,
            self.candidate_result_sha256,
            self.control_result_sha256,
            *self.baseline_evidence_sha256,
        )
        if (
            self.version != "r3b-sealed-confirmation-report-v3"
            or not self.candidate_arm_id
            or not self.control_arm_id
            or not isinstance(self.decision, ConfirmationDecision)
            or any(
                not isinstance(value, str)
                or value == "0" * 64
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in hashes
            )
        ):
            raise ValueError("sealed confirmation report identity is invalid")

    @property
    def accepted(self) -> bool:
        return self.decision.accepted

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "plan_sha256": self.plan_sha256,
            "authorization_sha256": self.authorization_sha256,
            "ledger_receipt_sha256": self.ledger_receipt_sha256,
            "candidate_arm_id": self.candidate_arm_id,
            "control_arm_id": self.control_arm_id,
            "candidate_result_sha256": self.candidate_result_sha256,
            "control_result_sha256": self.control_result_sha256,
            "baseline_evidence_sha256": list(self.baseline_evidence_sha256),
            "decision": self.decision.manifest(),
        }

    @classmethod
    def from_manifest(cls, value: object) -> SealedConfirmationReport:
        manifest = _require_keys(
            value,
            {
                "version",
                "plan_sha256",
                "authorization_sha256",
                "ledger_receipt_sha256",
                "candidate_arm_id",
                "control_arm_id",
                "candidate_result_sha256",
                "control_result_sha256",
                "baseline_evidence_sha256",
                "decision",
            },
            location="sealed confirmation report",
        )
        evidence = _manifest_list(
            manifest["baseline_evidence_sha256"],
            location="sealed confirmation baseline identities",
        )
        string_fields = (
            "version",
            "plan_sha256",
            "authorization_sha256",
            "ledger_receipt_sha256",
            "candidate_arm_id",
            "control_arm_id",
            "candidate_result_sha256",
            "control_result_sha256",
        )
        if any(type(manifest[name]) is not str for name in string_fields) or any(
            type(item) is not str for item in evidence
        ):
            raise ValueError("sealed confirmation report fields are malformed")
        try:
            result = cls(
                plan_sha256=manifest["plan_sha256"],  # type: ignore[arg-type]
                authorization_sha256=manifest["authorization_sha256"],  # type: ignore[arg-type]
                ledger_receipt_sha256=manifest["ledger_receipt_sha256"],  # type: ignore[arg-type]
                candidate_arm_id=manifest["candidate_arm_id"],  # type: ignore[arg-type]
                control_arm_id=manifest["control_arm_id"],  # type: ignore[arg-type]
                candidate_result_sha256=manifest["candidate_result_sha256"],  # type: ignore[arg-type]
                control_result_sha256=manifest["control_result_sha256"],  # type: ignore[arg-type]
                baseline_evidence_sha256=tuple(evidence),  # type: ignore[arg-type]
                decision=ConfirmationDecision.from_manifest(manifest["decision"]),
                version=manifest["version"],  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("sealed confirmation report is malformed") from exc
        _require_manifest_round_trip(
            manifest, result.manifest(), location="sealed confirmation report"
        )
        return result

    def publish(self, store: object) -> str:
        from .r3b_artifacts import ArtifactStore

        if not isinstance(store, ArtifactStore):
            raise TypeError("sealed confirmation requires an ArtifactStore")
        return store.publish(
            kind=_SEALED_CONFIRMATION_KIND,
            version=self.version,
            payload=self.manifest(),
        ).artifact_id

    @classmethod
    def load_finalized(
        cls,
        store: object,
        artifact_id: str,
        *,
        ledger: SealedTestLedger,
    ) -> SealedConfirmationReport:
        from .r3b_artifacts import ArtifactStore

        if not isinstance(store, ArtifactStore):
            raise TypeError("sealed confirmation requires an ArtifactStore")
        envelope = store.load(
            artifact_id,
            expected_kind=_SEALED_CONFIRMATION_KIND,
            expected_version="r3b-sealed-confirmation-report-v3",
        )
        report = cls.from_manifest(envelope.payload)
        if not ledger.verify_finalized(report):
            raise RuntimeError("sealed confirmation is not finalized in the ledger")
        return report

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def _percentile_lower(values: list[float], confidence_level: float) -> float:
    ordered = sorted(values)
    tail_probability = 1.0 - confidence_level
    index = max(0, math.ceil(tail_probability * len(ordered)) - 1)
    return ordered[index]


def confirm_on_sealed_test(
    plan: R3BExperimentPlan,
    sealed_run: SealedTestRunAuthorization,
    candidate_result: ArmPhaseResult,
    control_result: ArmPhaseResult,
    baseline_artifacts: Sequence[object],
) -> ConfirmationDecision:
    """Apply one-sided paired-bootstrap gates to one preselected candidate."""

    if not isinstance(sealed_run, SealedTestRunAuthorization):
        return ConfirmationDecision(False, (("complete_exact_test_results", False),))
    authorization_valid = sealed_run.plan.sha256 == plan.sha256
    authorization = sealed_run.authorization
    arm_index = {arm.arm_id: arm for arm in plan.arms}
    candidate_arm = arm_index.get(authorization.candidate_arm_id)
    control_arm = arm_index.get(authorization.control_arm_id)
    preconditions = (
        authorization_valid
        and authorization.plan_sha256 == plan.sha256
        and candidate_arm is not None
        and control_arm is not None
        and candidate_arm.alpha_weight_ppm > 0
        and control_arm.alpha_weight_ppm == 0
        and candidate_result.arm_id == authorization.candidate_arm_id
        and control_result.arm_id == authorization.control_arm_id
        and candidate_result.phase == control_result.phase == "test"
        and candidate_result.budget_updates
        == control_result.budget_updates
        == plan.test_updates
        and candidate_result.status == control_result.status == "complete"
    )
    if not preconditions:
        return ConfirmationDecision(False, (("complete_exact_test_results", False),))
    try:
        _require_exact_seeds(candidate_result, plan.test_learner_seeds)
        _require_exact_seeds(control_result, plan.test_learner_seeds)
        _require_paired_identities(
            (candidate_result, control_result), plan.test_learner_seeds
        )
        if (
            _require_runner_spec((candidate_result, control_result))
            != authorization.runner_spec_sha256
        ):
            raise ValueError("test runner differs from the authorized runner")
        _require_evaluation_suite(plan, (candidate_result, control_result))
        _require_metric_artifacts(plan, (candidate_result, control_result))
    except ValueError:
        return ConfirmationDecision(False, (("complete_exact_test_results", False),))
    candidate = {outcome.learner_seed: outcome for outcome in candidate_result.outcomes}
    control = {outcome.learner_seed: outcome for outcome in control_result.outcomes}
    exact_test = set(candidate) == set(control) == set(plan.test_learner_seeds)
    engineering = all(
        outcome.engineering_evidence is not None
        and outcome.engineering_evidence.plan_sha256 == plan.sha256
        and outcome.engineering_evidence.phase == "test"
        and outcome.engineering_evidence.authorization_sha256 == authorization.sha256
        and outcome.engineering_evidence.evaluation_suite_sha256
        == authorization.test_suite_sha256
        for outcome in (*candidate_result.outcomes, *control_result.outcomes)
    )
    try:
        baseline_evidence = _resolve_sealed_baseline_batch(
            plan,
            sealed_run.baseline_batch_commitment,
            baseline_artifacts,
        )
        baseline_pass = True
    except (TypeError, ValueError):
        baseline_evidence = ()
        baseline_pass = False
    if not exact_test:
        return ConfirmationDecision(
            False,
            (
                ("complete_exact_test_results", False),
                ("engineering_audits", engineering),
                ("required_baselines", baseline_pass),
            ),
        )
    seed_order = plan.test_learner_seeds
    strongest_baseline_mean = (
        max(
            item.mean_raw_score
            for item in baseline_evidence
            if item.baseline_id in plan.required_baselines
        )
        if baseline_pass
        else 0.0
    )
    control_p10_mean = statistics.fmean(
        control[seed].p10_raw_score for seed in seed_order
    )
    control_final_mean = statistics.fmean(
        control[seed].final_mean_raw_score for seed in seed_order
    )
    final_mean_mode = (
        "ratio"
        if control_final_mean > plan.relative_score_denominator_floor
        else "absolute_delta"
    )
    p10_mode = (
        "ratio"
        if control_p10_mean > plan.relative_score_denominator_floor
        else "absolute_delta"
    )
    rng = random.Random(plan.bootstrap_seed)
    auc_samples: list[float] = []
    final_samples: list[float] = []
    p10_samples: list[float] = []
    trivial_samples: list[float] = []
    for _ in range(plan.bootstrap_samples):
        sampled = [seed_order[rng.randrange(len(seed_order))] for _ in seed_order]
        candidate_auc = statistics.fmean(
            candidate[seed].raw_score_auc for seed in sampled
        )
        control_auc = statistics.fmean(control[seed].raw_score_auc for seed in sampled)
        candidate_final = statistics.fmean(
            candidate[seed].final_mean_raw_score for seed in sampled
        )
        control_final = statistics.fmean(
            control[seed].final_mean_raw_score for seed in sampled
        )
        candidate_p10 = statistics.fmean(
            candidate[seed].p10_raw_score for seed in sampled
        )
        control_p10 = statistics.fmean(control[seed].p10_raw_score for seed in sampled)
        auc_samples.append(
            (candidate_auc - control_auc)
            / max(abs(control_auc), plan.relative_score_denominator_floor)
        )
        final_samples.append(
            candidate_final / max(control_final, plan.relative_score_denominator_floor)
            if final_mean_mode == "ratio"
            else candidate_final - control_final
        )
        p10_samples.append(
            candidate_p10 / max(control_p10, plan.relative_score_denominator_floor)
            if p10_mode == "ratio"
            else candidate_p10 - control_p10
        )
        trivial_samples.append(candidate_final - strongest_baseline_mean)
    auc_lower = _percentile_lower(auc_samples, plan.confidence_level)
    final_lower = _percentile_lower(final_samples, plan.confidence_level)
    p10_lower = _percentile_lower(p10_samples, plan.confidence_level)
    trivial_lower = _percentile_lower(trivial_samples, plan.confidence_level)
    p10_threshold = (
        plan.minimum_p10_retention
        if p10_mode == "ratio"
        else plan.minimum_p10_absolute_delta_when_control_near_zero
    )
    final_threshold = (
        plan.minimum_final_mean_retention if final_mean_mode == "ratio" else 0.0
    )
    gates = (
        ("complete_exact_test_results", True),
        ("engineering_audits", engineering),
        ("required_baselines", baseline_pass),
        ("relative_auc_gain_lcb", auc_lower > plan.minimum_relative_auc_gain),
        ("final_mean_retention_lcb", final_lower >= final_threshold),
        ("p10_noninferiority_lcb", p10_lower >= p10_threshold),
        (
            "trivial_baseline_margin_lcb",
            trivial_lower > plan.minimum_trivial_baseline_margin,
        ),
    )
    return ConfirmationDecision(
        accepted=all(passed for _, passed in gates),
        gates=gates,
        relative_auc_gain_lower=auc_lower,
        final_mean_retention_lower=final_lower,
        final_mean_mode=final_mean_mode,
        p10_lower=p10_lower,
        p10_mode=p10_mode,
        trivial_baseline_margin_lower=trivial_lower,
    )


def build_sealed_confirmation_report(
    plan: R3BExperimentPlan,
    sealed_run: SealedTestRunAuthorization,
    candidate_result: ArmPhaseResult,
    control_result: ArmPhaseResult,
    baseline_artifacts: Sequence[object],
) -> SealedConfirmationReport:
    """Bind the canonical decision to every result/evidence artifact identity."""

    artifacts = tuple(baseline_artifacts)
    authorization = sealed_run.authorization
    try:
        evidence = _resolve_baseline_artifacts(artifacts)
    except (TypeError, ValueError):
        evidence = ()
    decision = confirm_on_sealed_test(
        plan,
        sealed_run,
        candidate_result,
        control_result,
        artifacts,
    )
    return SealedConfirmationReport(
        plan.sha256,
        authorization.sha256,
        sealed_run.ledger_receipt_sha256,
        authorization.candidate_arm_id,
        authorization.control_arm_id,
        candidate_result.sha256,
        control_result.sha256,
        tuple(item.sha256 for item in evidence),
        decision,
    )


@dataclass(frozen=True, slots=True)
class DurableSealedFinalization:
    """A finalized ledger decision and its immutable report artifact."""

    report: SealedConfirmationReport
    report_artifact_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(
            self.report, SealedConfirmationReport
        ) or not _is_nonzero_sha256(self.report_artifact_sha256):
            raise ValueError("durable sealed finalization is malformed")


def _find_finalized_report_artifact(
    store: object,
    ledger: SealedTestLedger,
    report_sha256: str,
) -> DurableSealedFinalization:
    from .r3b_artifacts import ArtifactStore, ArtifactTypeError

    if not isinstance(store, ArtifactStore):
        raise TypeError("sealed finalization requires an ArtifactStore")
    matches: list[tuple[str, SealedConfirmationReport]] = []
    for artifact_id in store.list():
        try:
            envelope = store.load(
                artifact_id,
                expected_kind=_SEALED_CONFIRMATION_KIND,
                expected_version="r3b-sealed-confirmation-report-v3",
            )
        except ArtifactTypeError:
            continue
        report = SealedConfirmationReport.from_manifest(envelope.payload)
        if report.sha256 == report_sha256:
            matches.append((artifact_id, report))
    if len(matches) != 1 or not ledger.verify_finalized(matches[0][1]):
        raise RuntimeError("finalized ledger report has no unique immutable artifact")
    return DurableSealedFinalization(matches[0][1], matches[0][0])


def finalize_persisted_sealed_test(
    *,
    store: object,
    workflow: object,
    ledger: SealedTestLedger,
    sealed_run: SealedTestRunAuthorization,
    authorization_artifact_sha256: str,
    baseline_artifact_sha256: str | None,
    outcome_reference_sha256s: Sequence[str],
    outcome_loader: Callable[[str], LearnerOutcome],
) -> DurableSealedFinalization:
    """Reload, reconcile, and finalize one sealed test after arbitrary restarts.

    Rich outcome packages remain owned by the canonical runner; ``outcome_loader``
    must perform that strict dependency reconstruction. This coordinator then
    binds each loaded outcome to its immutable reference, ledger row, and workflow
    row before the one-shot decision is allowed to finalize.
    """

    from .r3b_artifacts import ArtifactStore
    from .r3b_operational import R3BWorkflow

    if (
        not isinstance(store, ArtifactStore)
        or not isinstance(workflow, R3BWorkflow)
        or not isinstance(ledger, SealedTestLedger)
        or not isinstance(sealed_run, SealedTestRunAuthorization)
        or not callable(outcome_loader)
    ):
        raise TypeError("sealed finalization dependencies are malformed")
    reloaded_run = SealedTestRunAuthorization.load(
        store,
        authorization_artifact_sha256,
        plan=sealed_run.plan,
        validation_run=sealed_run.validation_run,
        validation_results=sealed_run.validation_results,
        validation_suite=sealed_run.validation_suite,
        test_suite=sealed_run.test_suite,
        allow_finalized=True,
    )
    if reloaded_run != sealed_run:
        raise ValueError("sealed authorization artifact differs from the active run")
    finalized_sha256 = ledger.finalized_confirmation_sha256(reloaded_run)
    if finalized_sha256 is not None:
        return _find_finalized_report_artifact(store, ledger, finalized_sha256)

    committed_baselines: tuple[object, ...]
    if baseline_artifact_sha256 is None:
        committed_baselines = ()
    else:
        committed_baselines = (
            SealedBaselineEvidenceArtifact.load_committed(
                store,
                baseline_artifact_sha256,
                ledger=ledger,
                sealed_run=reloaded_run,
            ),
        )
    records = ledger.terminal_job_records(reloaded_run)
    jobs = {
        job.sha256: job for job in reloaded_run.plan.trial_jobs("test", reloaded_run)
    }
    references = tuple(
        SealedLearnerOutcomeReference.load(store, artifact_id)
        for artifact_id in outcome_reference_sha256s
    )
    indexed_references = {reference.job_sha256: reference for reference in references}
    completed_ids = {
        record.job_sha256 for record in records if record.state == "complete"
    }
    if (
        len(indexed_references) != len(references)
        or set(indexed_references) != completed_ids
    ):
        raise ValueError(
            "sealed outcome references must cover exactly the completed ledger jobs"
        )
    outcomes_by_arm: dict[str, list[LearnerOutcome]] = {}
    failures_by_arm: dict[str, list[tuple[str, str]]] = {}
    for record in records:
        job = jobs[record.job_sha256]
        if record.state == "failure":
            assert record.failure_reason is not None
            failures_by_arm.setdefault(record.arm_id, []).append(
                (record.job_sha256, record.failure_reason)
            )
            workflow.reconcile_sealed_failure(
                ledger=ledger,
                sealed_run=reloaded_run,
                job=job,
                failure_reason=record.failure_reason,
            )
            continue
        reference = indexed_references[record.job_sha256]
        outcome = reference.resolve(
            ledger=ledger,
            sealed_run=reloaded_run,
            job=job,
            loader=outcome_loader,
        )
        workflow.reconcile_sealed_completion(
            ledger=ledger,
            sealed_run=reloaded_run,
            job=job,
            outcome_sha256=outcome.sha256,
            output_sha256=reference.output_artifact_sha256,
        )
        outcomes_by_arm.setdefault(record.arm_id, []).append(outcome)

    def arm_result(arm_id: str) -> ArmPhaseResult:
        if arm_id in failures_by_arm:
            reason = "; ".join(
                f"{job_sha256}:{message}"
                for job_sha256, message in failures_by_arm[arm_id]
            )
            return ArmPhaseResult(
                arm_id,
                "test",
                "post_start_failure",
                reloaded_run.plan.test_updates,
                failure_reason=reason,
            )
        outcomes = tuple(
            sorted(
                outcomes_by_arm.get(arm_id, ()),
                key=lambda outcome: outcome.learner_seed,
            )
        )
        expected_seeds = set(reloaded_run.plan.test_learner_seeds)
        if {outcome.learner_seed for outcome in outcomes} != expected_seeds:
            raise ValueError("sealed arm outcome references omit learner seeds")
        return ArmPhaseResult(
            arm_id,
            "test",
            "complete",
            reloaded_run.plan.test_updates,
            outcomes,
        )

    candidate = arm_result(reloaded_run.authorization.candidate_arm_id)
    control = arm_result(reloaded_run.authorization.control_arm_id)
    preview = build_sealed_confirmation_report(
        reloaded_run.plan,
        reloaded_run,
        candidate,
        control,
        committed_baselines,
    )
    report_artifact = preview.publish(store)
    report = ledger.finalize_once(
        reloaded_run,
        candidate,
        control,
        committed_baselines,
    )
    if report != preview:
        raise RuntimeError("sealed finalization differs from its published preview")
    return DurableSealedFinalization(report, report_artifact)
