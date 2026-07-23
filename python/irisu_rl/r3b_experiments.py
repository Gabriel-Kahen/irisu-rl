"""Immutable R3b experiment design and fail-closed acceptance logic.

This module deliberately does not run training.  It binds an experiment runner to
the checked-in plan, makes failed arms first-class records, and keeps model
selection separate from the one-shot sealed-test decision.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import secrets
import sqlite3
import statistics
import tomllib
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence


_RESULT_STATUSES = frozenset(
    {"complete", "eliminated", "pre_start_failure", "post_start_failure"}
)
_PHASES = frozenset({"calibration", "validation", "test"})


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


def _require_keys(
    value: object, expected: set[str], *, location: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} must be a table")
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{location} keys differ: missing={missing}, extra={extra}")
    return value


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


@dataclass(frozen=True, slots=True)
class TrialSeedPlan:
    """Domain-separated RNG identities shared by every arm for one learner seed."""

    learner_seed: int
    model_initialization: int
    policy_sampling: int
    ppo_minibatching: int
    assignment: int
    session_numpy: int
    evaluation: int

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
            derive("evaluation"),
        )

    def manifest(self) -> dict[str, int | str]:
        return {
            "version": "r3b-trial-seed-plan-v1",
            "learner_seed": self.learner_seed,
            "model_initialization": self.model_initialization,
            "policy_sampling": self.policy_sampling,
            "ppo_minibatching": self.ppo_minibatching,
            "assignment": self.assignment,
            "session_numpy": self.session_numpy,
            "evaluation": self.evaluation,
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
            or self.ticks_per_update != 32768
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
        if (
            self.calibration_elimination_metric != "paired_tick_aligned_raw_score_auc"
            or self.auc_definition
            != "trapezoid_on_exact_simulated_tick_grid_normalized_by_horizon"
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
            {"calibration_learner", "validation_learner", "test_learner"},
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
    """Trapezoidal AUC divided by horizon; interpolation is forbidden."""

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
    if (
        len(points) != len(ticks)
        or tuple(point.simulated_ticks for point in points) != ticks
    ):
        raise ValueError("raw-score curve must exactly match the planned tick grid")
    area = sum(
        (right.simulated_ticks - left.simulated_ticks)
        * (left.mean_raw_score + right.mean_raw_score)
        / 2.0
        for left, right in zip(points, points[1:])
    )
    return area / ticks[-1]


@dataclass(frozen=True, slots=True)
class TrainingCheckpointArtifact:
    """Session-produced checkpoint identity used for one metric-grid point."""

    learner_seed: int
    completed_updates: int
    simulated_ticks: int
    plan_sha256: str
    job_sha256: str
    trial_manifest_sha256: str
    runner_spec_sha256: str
    checkpoint_manifest_sha256: str
    model_sha256: str
    deployment_policy_sha256: str
    version: str = "r3b-training-checkpoint-artifact-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-training-checkpoint-artifact-v1"
            or isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or not 0 <= self.learner_seed < 2**64
            or isinstance(self.completed_updates, bool)
            or not isinstance(self.completed_updates, int)
            or self.completed_updates < 0
            or isinstance(self.simulated_ticks, bool)
            or not isinstance(self.simulated_ticks, int)
            or self.simulated_ticks < 0
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

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class RawScoreMetricsArtifact:
    """Canonical checkpoint reports from which every selection metric is derived."""

    learner_seed: int
    suite: object
    checkpoints: tuple[CheckpointEvaluation, ...]
    version: str = "r3b-raw-score-metrics-v2"

    def __post_init__(self) -> None:
        from .r3b_evaluation import EvaluationReport, EvaluationSuite

        if (
            self.version != "r3b-raw-score-metrics-v2"
            or isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or not 0 <= self.learner_seed < 2**64
            or not isinstance(self.suite, EvaluationSuite)
            or not isinstance(self.checkpoints, tuple)
            or len(self.checkpoints) < 2
        ):
            raise ValueError("raw-score metrics artifact is malformed")
        ticks: list[int] = []
        reports: list[EvaluationReport] = []
        expected_cells = {
            (snapshot_id, repetition)
            for snapshot_id in self.suite.snapshot_ids
            for repetition in range(self.suite.repetitions)
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
                report.suite_sha256 != self.suite.sha256
                or report.backend_identity_sha256 != self.suite.runtime_identity_sha256
                or cells != expected_cells
                or any(
                    episode.policy_seed
                    != self.suite.episode_seed(episode.snapshot_id, episode.repetition)
                    or episode.decisions > self.suite.max_decisions
                    or episode.elapsed_ticks > self.suite.max_simulated_ticks
                    or not (episode.terminated or episode.truncated)
                    or episode.invalid_actions != 0
                    for episode in report.episodes
                )
            ):
                raise ValueError("checkpoint report cells do not match the suite")
            updates.append(checkpoint.completed_updates)
            ticks.append(tick)
            reports.append(report)
        if (
            updates[0] != 0
            or ticks[0] != 0
            or any(right <= left for left, right in zip(updates, updates[1:]))
            or any(right <= left for left, right in zip(ticks, ticks[1:]))
        ):
            raise ValueError("checkpoint report ticks must start at zero and increase")
        if (
            len({report.evaluator_sha256 for report in reports}) != 1
            or len({report.backend_identity_sha256 for report in reports}) != 1
            or len({report.sha256 for report in reports}) != len(reports)
            or len(
                {
                    checkpoint.checkpoint_artifact_sha256
                    for checkpoint in self.checkpoints
                }
            )
            != len(self.checkpoints)
        ):
            raise ValueError(
                "checkpoint reports changed evaluator/backend or reused an artifact"
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
    def final_report(self):
        return self.checkpoints[-1].report

    @property
    def raw_score_auc(self) -> float:
        points = self.points
        return tick_aligned_raw_score_auc(
            points, tuple(point.simulated_ticks for point in points)
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
            "suite_sha256": self.suite.sha256,
            "checkpoints": [checkpoint.manifest() for checkpoint in self.checkpoints],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class ExactResumeArtifact:
    """Independent checkpoint restore whose next update matches uninterrupted work."""

    trial_manifest_sha256: str
    checkpoint_manifest_sha256: str
    checkpoint_policy_sha256: str
    source_next_update_sha256: str
    restored_next_update_sha256: str
    source_after_state_sha256: str
    restored_after_state_sha256: str
    version: str = "r3b-exact-resume-artifact-v1"

    def __post_init__(self) -> None:
        hashes = (
            self.trial_manifest_sha256,
            self.checkpoint_manifest_sha256,
            self.checkpoint_policy_sha256,
            self.source_next_update_sha256,
            self.restored_next_update_sha256,
            self.source_after_state_sha256,
            self.restored_after_state_sha256,
        )
        if (
            self.version != "r3b-exact-resume-artifact-v1"
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
    checkpoint_resume_artifact: ExactResumeArtifact
    exact_backend_parity_artifact: object
    tail_state_sha256: str | None = None
    tail_phase: str | None = None
    score_only_updates: int = 0
    version: str = "r3b-engineering-evidence-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-engineering-evidence-v1"
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

        if not isinstance(
            self.checkpoint_resume_artifact, ExactResumeArtifact
        ) or not isinstance(
            self.exact_backend_parity_artifact,
            LearnedPolicyBackendParityArtifact,
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
            self.checkpoint_resume_artifact.trial_manifest_sha256
            != self.trial_manifest_sha256
            or self.exact_backend_parity_artifact.policy_sha256 != self.policy_sha256
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
        value["checkpoint_resume_artifact"] = self.checkpoint_resume_artifact.manifest()
        value["exact_backend_parity_artifact"] = (
            self.exact_backend_parity_artifact.manifest()
        )
        return value

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


def _require_evaluation_suite(results: Sequence[ArmPhaseResult]) -> str:
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
            if (
                evidence is None
                or artifact.suite.split != result.phase
                or tuple(
                    checkpoint.completed_updates for checkpoint in artifact.checkpoints
                )
                != expected_updates
                or tuple(
                    checkpoint.simulated_ticks for checkpoint in artifact.checkpoints
                )
                != expected_ticks
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
    _require_evaluation_suite(tuple(indexed.values()))
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
    evaluation_suite_sha256 = _require_evaluation_suite(retained)
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
    _require_evaluation_suite(tuple(indexed.values()))
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
    exact_report_sha256: str
    portable_backend_sha256: str
    exact_backend_sha256: str
    episode_content_sha256: str
    version: str = "r3b-baseline-evidence-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-baseline-evidence-v1"
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
                self.exact_report_sha256,
                self.portable_backend_sha256,
                self.exact_backend_sha256,
                self.episode_content_sha256,
            )
        ):
            raise ValueError("baseline evidence must bind nonzero report SHA-256s")
        if (
            len(
                {
                    self.report_sha256,
                    self.replay_report_sha256,
                    self.exact_report_sha256,
                }
            )
            != 3
            or self.portable_backend_sha256 == self.exact_backend_sha256
        ):
            raise ValueError("baseline evidence requires independent backend reports")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


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

    if any(not isinstance(item, BaselineArtifactBundle) for item in artifacts):
        raise TypeError("sealed confirmation requires typed baseline artifacts")
    return tuple(item.evidence() for item in artifacts)


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
    attempt: int = 1
    version: str = "r3b-sealed-test-authorization-v4"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-sealed-test-authorization-v4"
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
                )
            )
        ):
            raise ValueError("sealed-test authorization identity is invalid")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

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
    return SealedTestAuthorization(
        plan.sha256,
        validation_run.sha256,
        control.arm_id,
        candidate.arm_id,
        result_identity,
        validation_suite.sha256,
        test_suite.sha256,
        calibration_authorization.runner_spec_sha256,
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
    ledger_path: str
    ledger_receipt_token: str
    version: str = "r3b-sealed-test-run-authorization-v3"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-sealed-test-run-authorization-v3"
            or not isinstance(self.plan, R3BExperimentPlan)
            or not isinstance(self.validation_results, tuple)
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
        result_sha256 = _canonical_sha256(
            [
                result.sha256
                for result in sorted(
                    self.validation_results, key=lambda value: value.arm_id
                )
            ]
        )
        self.assert_authorized(expected_validation_results_sha256=result_sha256)

    @property
    def ledger_receipt_sha256(self) -> str:
        return hashlib.sha256(self.ledger_receipt_token.encode()).hexdigest()

    def assert_authorized(
        self, *, expected_validation_results_sha256: str | None = None
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
            with closing(sqlite3.connect(self.ledger_path, timeout=30.0)) as connection:
                row = connection.execute(
                    "SELECT state,receipt_token,validation_run_sha256,"
                    "validation_results_sha256,validation_suite_sha256,"
                    "authorization_sha256,jobs_sha256 FROM sealed_test_attempt "
                    "WHERE plan_sha256=?",
                    (self.plan.sha256,),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError("sealed-test ledger is unavailable") from exc
        expected = (
            "authorized",
            self.ledger_receipt_token,
            self.validation_run.sha256,
            result_sha256,
            self.validation_suite.sha256,
            self.authorization.sha256,
            jobs_sha256,
        )
        if row != expected:
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
        ledger_path,
        ledger_receipt_token,
    )


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
                sqlite3.connect(self.sealed_run.ledger_path, timeout=30.0)
            ) as connection:
                row = connection.execute(
                    "SELECT state,lease_token FROM sealed_test_job "
                    "WHERE plan_sha256=? AND job_sha256=?",
                    (self.sealed_run.plan.sha256, self.job.sha256),
                ).fetchone()
        except sqlite3.Error as exc:
            raise RuntimeError("sealed-test job ledger is unavailable") from exc
        if row is None or row[0] not in allowed or row[1] != self.lease_token:
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


class SealedTestLedger:
    """Restart-safe single-attempt ledger for a precommitted test suite."""

    version = "r3b-sealed-test-ledger-v1"

    def __init__(self, path: str | Path) -> None:
        supplied = Path(path).expanduser()
        if not supplied.is_absolute():
            raise ValueError("sealed-test ledger path must be absolute")
        self.path = supplied
        self.path.parent.mkdir(parents=True, exist_ok=True)
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
                    receipt_token TEXT UNIQUE,
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
                    lease_token TEXT UNIQUE,
                    outcome_sha256 TEXT,
                    failure_reason TEXT,
                    PRIMARY KEY(plan_sha256, job_sha256),
                    UNIQUE(plan_sha256, arm_id, learner_seed),
                    FOREIGN KEY(plan_sha256) REFERENCES sealed_test_attempt(plan_sha256)
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30.0)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def precommit(
        self, plan: R3BExperimentPlan, test_suite: object
    ) -> TestSuiteCommitment:
        from .r3b_evaluation import EvaluationSuite

        if (
            not isinstance(plan, R3BExperimentPlan)
            or not isinstance(test_suite, EvaluationSuite)
            or test_suite.split != "test"
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
        receipt_token = secrets.token_hex(32)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT test_suite_sha256,ledger_nonce,state,validation_run_sha256,"
                "validation_results_sha256,validation_suite_sha256,authorization_sha256,"
                "jobs_sha256,receipt_token FROM sealed_test_attempt WHERE plan_sha256=?",
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
            )
            if row[2] == "ready":
                updated = connection.execute(
                    "UPDATE sealed_test_attempt SET state='authorized',"
                    "validation_run_sha256=?,validation_results_sha256=?,"
                    "validation_suite_sha256=?,authorization_sha256=?,jobs_sha256=?,"
                    "receipt_token=? "
                    "WHERE plan_sha256=? AND state='ready'",
                    (*expected, receipt_token, plan.sha256),
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
            elif row[2] != "authorized" or tuple(row[3:8]) != expected:
                raise RuntimeError("sealed-test attempt is consumed or disagrees")
            else:
                receipt_token = str(row[8])
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

    def claim_job(
        self, sealed_run: SealedTestRunAuthorization, job: TrialJob
    ) -> SealedTestJobLease:
        """Atomically lease one pending test job; a lease cannot be reassigned."""

        if (
            not isinstance(sealed_run, SealedTestRunAuthorization)
            or Path(sealed_run.ledger_path) != self.path
            or job not in sealed_run.plan.trial_jobs("test", sealed_run)
        ):
            raise ValueError("sealed-test job does not belong to this ledger run")
        lease_token = secrets.token_hex(32)
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT state,lease_token FROM sealed_test_job "
                "WHERE plan_sha256=? AND job_sha256=?",
                (sealed_run.plan.sha256, job.sha256),
            ).fetchone()
            if row is None:
                raise RuntimeError("sealed-test job is absent from the authorization")
            if row[0] == "pending":
                updated = connection.execute(
                    "UPDATE sealed_test_job SET state='leased',lease_token=? "
                    "WHERE plan_sha256=? AND job_sha256=? AND state='pending'",
                    (lease_token, sealed_run.plan.sha256, job.sha256),
                )
                if updated.rowcount != 1:
                    raise RuntimeError("sealed-test job claim race was lost")
            else:
                raise RuntimeError("sealed-test job is already claimed or terminal")
            connection.commit()
        return SealedTestJobLease(sealed_run, job, lease_token)

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
                "SELECT state,lease_token FROM sealed_test_job "
                "WHERE plan_sha256=? AND job_sha256=?",
                (sealed_run.plan.sha256, job.sha256),
            ).fetchone()
        if row != ("leased", lease_token):
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
                "AND lease_token=?",
                (
                    lease.sealed_run.plan.sha256,
                    lease.job.sha256,
                    lease.lease_token,
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
                "AND lease_token=?",
                (
                    outcome.sha256,
                    lease.sealed_run.plan.sha256,
                    lease.job.sha256,
                    lease.lease_token,
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
                "AND lease_token=?",
                (
                    failure_reason.strip(),
                    lease.sealed_run.plan.sha256,
                    lease.job.sha256,
                    lease.lease_token,
                ),
            )
            if updated.rowcount != 1:
                raise RuntimeError("sealed-test job lease is stale or consumed")
            connection.commit()

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
                "SELECT state,authorization_sha256,receipt_token FROM sealed_test_attempt "
                "WHERE plan_sha256=?",
                (sealed_run.plan.sha256,),
            ).fetchone()
            if attempt != (
                "authorized",
                sealed_run.authorization.sha256,
                sealed_run.ledger_receipt_token,
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
        _require_evaluation_suite((candidate_result, control_result))
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
        baseline_evidence = _resolve_baseline_artifacts(baseline_artifacts)
        baseline_pass = baseline_requirements_pass(
            plan,
            baseline_evidence,
            expected_suite_sha256=authorization.test_suite_sha256,
        )
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
