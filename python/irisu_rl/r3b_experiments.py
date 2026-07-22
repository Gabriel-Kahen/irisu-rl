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
import statistics
import tomllib
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
        arms: Sequence[CandidateArm] = (),
    ) -> tuple[TrialJob, ...]:
        """Enumerate the exact paired jobs allowed to enter one phase."""

        supplied = tuple(arms)
        if phase == "calibration":
            if supplied:
                raise ValueError("calibration arms come only from the frozen grid")
            selected = self.arms
            seeds = self.calibration_learner_seeds
            budget = self.calibration_budgets_updates[-1]
            sealed = False
        elif phase == "validation":
            if (
                len(supplied) != len(self.alpha_weight_ppm)
                or tuple(arm.alpha_weight_ppm for arm in supplied)
                != self.alpha_weight_ppm
                or any(arm not in self.arms for arm in supplied)
            ):
                raise ValueError("validation requires one grid arm per alpha")
            selected = supplied
            seeds = self.validation_learner_seeds
            budget = self.validation_updates
            sealed = False
        elif phase == "test":
            if (
                len(supplied) != 2
                or supplied[0].alpha_weight_ppm != 0
                or supplied[1].alpha_weight_ppm <= 0
                or any(arm not in self.arms for arm in supplied)
            ):
                raise ValueError("test requires one control and one shaped grid arm")
            selected = supplied
            seeds = self.test_learner_seeds
            budget = self.test_updates
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
            )
            for arm in selected
            for seed in seeds
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
    version: str = "r3b-trial-job-v1"

    def __post_init__(self) -> None:
        if (
            self.phase not in _PHASES
            or self.version != "r3b-trial-job-v1"
            or not isinstance(self.arm, CandidateArm)
            or isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or not 0 <= self.learner_seed < 2**64
            or isinstance(self.budget_updates, bool)
            or not isinstance(self.budget_updates, int)
            or self.budget_updates <= 0
            or not isinstance(self.sealed, bool)
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
        }

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
class LearnerOutcome:
    learner_seed: int
    raw_score_auc: float
    final_mean_raw_score: float
    p10_raw_score: float
    initial_model_sha256: str
    assignment_sha256: str
    seed_plan_sha256: str
    engineering_pass: bool = True

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
        if not isinstance(self.engineering_pass, bool):
            raise ValueError("engineering_pass must be boolean")
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
        return asdict(self)


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
    by_seed: dict[int, tuple[str, str, str]] = {}
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
            )
            previous = by_seed.setdefault(outcome.learner_seed, identity)
            if previous != identity:
                raise ValueError(
                    "paired arms disagree on initial model, assignment, or seed identity"
                )


def select_calibrated_learning_rates(
    plan: R3BExperimentPlan, results: Sequence[ArmPhaseResult]
) -> tuple[CandidateArm, ...]:
    """Select one LR per alpha from the final calibration rung."""

    arms = plan.arms
    indexed = _index_exact_results(
        results, {arm.arm_id for arm in arms}, phase="calibration"
    )
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


def select_validation_candidate(
    plan: R3BExperimentPlan,
    calibrated_arms: Sequence[CandidateArm],
    results: Sequence[ArmPhaseResult],
) -> CandidateArm | None:
    """Choose one shaped candidate using validation only; never consult test data."""

    arms = tuple(calibrated_arms)
    if (
        len(arms) != len(plan.alpha_weight_ppm)
        or tuple(arm.alpha_weight_ppm for arm in arms) != plan.alpha_weight_ppm
    ):
        raise ValueError("validation requires one calibrated arm per alpha")
    indexed = _index_exact_results(
        results, {arm.arm_id for arm in arms}, phase="validation"
    )
    _require_paired_identities(tuple(indexed.values()), plan.validation_learner_seeds)
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
            mean_final / control_final if control_final > 0 else float(mean_final >= 0)
        )
        if (
            relative_auc_gain < plan.minimum_relative_auc_gain
            or final_retention < plan.minimum_final_mean_retention
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
    deterministic_replay: bool
    raw_score_identity: bool
    portable_exact_parity: bool
    report_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.baseline_id, str) or not self.baseline_id:
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
        if (
            not isinstance(self.report_sha256, str)
            or self.report_sha256 == "0" * 64
            or len(self.report_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.report_sha256
            )
        ):
            raise ValueError("baseline evidence must bind a nonzero report SHA-256")
        if any(
            not isinstance(value, bool)
            for value in (
                self.deterministic_replay,
                self.raw_score_identity,
                self.portable_exact_parity,
            )
        ):
            raise ValueError("baseline audit fields must be boolean")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def baseline_requirements_pass(
    plan: R3BExperimentPlan, evidence: Sequence[BaselineEvidence]
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
        and item.deterministic_replay
        and item.raw_score_identity
        and item.portable_exact_parity
        for item in (indexed[name] for name in plan.required_baselines)
    )


@dataclass(frozen=True, slots=True)
class ConfirmationDecision:
    accepted: bool
    gates: tuple[tuple[str, bool], ...]
    relative_auc_gain_lower: float | None = None
    final_mean_retention_lower: float | None = None
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
        if self.p10_mode not in {None, "ratio", "absolute_delta"}:
            raise ValueError("confirmation p10 mode is invalid")

    def manifest(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "gates": {name: passed for name, passed in self.gates},
            "relative_auc_gain_lower": self.relative_auc_gain_lower,
            "final_mean_retention_lower": self.final_mean_retention_lower,
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
    candidate_arm_id: str
    control_arm_id: str
    candidate_result_sha256: str
    control_result_sha256: str
    baseline_evidence_sha256: tuple[str, ...]
    decision: ConfirmationDecision
    version: str = "r3b-sealed-confirmation-report-v1"

    def __post_init__(self) -> None:
        hashes = (
            self.plan_sha256,
            self.candidate_result_sha256,
            self.control_result_sha256,
            *self.baseline_evidence_sha256,
        )
        if (
            self.version != "r3b-sealed-confirmation-report-v1"
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
    candidate_arm: CandidateArm,
    candidate_result: ArmPhaseResult,
    control_arm: CandidateArm,
    control_result: ArmPhaseResult,
    baseline_evidence: Sequence[BaselineEvidence],
) -> ConfirmationDecision:
    """Apply one-sided paired-bootstrap gates to one preselected candidate."""

    preconditions = (
        candidate_arm.alpha_weight_ppm > 0
        and control_arm.alpha_weight_ppm == 0
        and candidate_result.arm_id == candidate_arm.arm_id
        and control_result.arm_id == control_arm.arm_id
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
    except ValueError:
        return ConfirmationDecision(False, (("complete_exact_test_results", False),))
    candidate = {outcome.learner_seed: outcome for outcome in candidate_result.outcomes}
    control = {outcome.learner_seed: outcome for outcome in control_result.outcomes}
    exact_test = set(candidate) == set(control) == set(plan.test_learner_seeds)
    engineering = all(
        outcome.engineering_pass
        for outcome in (*candidate_result.outcomes, *control_result.outcomes)
    )
    baseline_pass = baseline_requirements_pass(plan, baseline_evidence)
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
            candidate_final / control_final
            if control_final > 0
            else float(candidate_final >= 0)
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
    gates = (
        ("complete_exact_test_results", True),
        ("engineering_audits", engineering),
        ("required_baselines", baseline_pass),
        ("relative_auc_gain_lcb", auc_lower > plan.minimum_relative_auc_gain),
        ("final_mean_retention_lcb", final_lower >= plan.minimum_final_mean_retention),
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
        p10_lower=p10_lower,
        p10_mode=p10_mode,
        trivial_baseline_margin_lower=trivial_lower,
    )


def build_sealed_confirmation_report(
    plan: R3BExperimentPlan,
    candidate_arm: CandidateArm,
    candidate_result: ArmPhaseResult,
    control_arm: CandidateArm,
    control_result: ArmPhaseResult,
    baseline_evidence: Sequence[BaselineEvidence],
) -> SealedConfirmationReport:
    """Bind the canonical decision to every result/evidence artifact identity."""

    evidence = tuple(baseline_evidence)
    decision = confirm_on_sealed_test(
        plan,
        candidate_arm,
        candidate_result,
        control_arm,
        control_result,
        evidence,
    )
    return SealedConfirmationReport(
        plan.sha256,
        candidate_arm.arm_id,
        control_arm.arm_id,
        candidate_result.sha256,
        control_result.sha256,
        tuple(item.sha256 for item in evidence),
        decision,
    )
