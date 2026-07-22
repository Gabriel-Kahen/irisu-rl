"""Construction of identity-bound R3b training trials."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch

from .collector import (
    CollectorConfig,
    CurriculumTaskContract,
    R3ATrainingSession,
    RecurrentCollector,
    model_state_sha256,
)
from .curriculum import (
    CurriculumCoordinator,
    CurriculumSnapshotInitializer,
    CurriculumSpec,
    SnapshotBlobStore,
)
from .encoding import TeacherStateEncoder
from .models import RecurrentActorCritic
from .ppo import PPOConfig, PPOTrainer
from .r3b_experiments import CandidateArm, R3BExperimentPlan, TrialSeedPlan
from .r3b_tail import ScoreOnlyTailController
from .rewards import (
    LinearGaugePotential,
    RewardComposer,
    RewardKnot,
    RewardSchedule,
)
from .vector_adapter import MacroVectorAdapter


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("trial manifest object keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("trial manifest values must be JSON-compatible")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class TrialManifest:
    plan_sha256: str
    arm_id: str
    learner_seed: int
    seed_plan_sha256: str
    initial_model_sha256: str
    curriculum_sha256: str
    assignment_sha256: str
    snapshot_store_sha256: str
    runtime_identity_sha256: str
    reward_sha256: str
    collector: Mapping[str, object]
    ppo: Mapping[str, object]
    lanes: int
    deployable: bool = False
    observation_provenance: str = "privileged_simulator"
    transfer_gate: str = "R4 causal tracker and input calibration pending"
    version: str = "r3b-trial-manifest-v1"

    def __post_init__(self) -> None:
        hashes = (
            self.plan_sha256,
            self.seed_plan_sha256,
            self.initial_model_sha256,
            self.curriculum_sha256,
            self.assignment_sha256,
            self.snapshot_store_sha256,
            self.runtime_identity_sha256,
            self.reward_sha256,
        )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or value == "0" * 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in hashes
        ):
            raise ValueError(
                "trial identities must be nonzero lowercase SHA-256 values"
            )
        if (
            self.version != "r3b-trial-manifest-v1"
            or not self.arm_id
            or isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or not 0 <= self.learner_seed < 2**64
            or isinstance(self.lanes, bool)
            or not isinstance(self.lanes, int)
            or self.lanes <= 0
        ):
            raise ValueError("trial manifest identity or dimensions are invalid")
        if (
            self.deployable
            or self.observation_provenance != "privileged_simulator"
            or not self.transfer_gate
        ):
            raise ValueError("R3b trial manifests must remain pre-transfer")
        object.__setattr__(self, "collector", _freeze_json(self.collector))
        object.__setattr__(self, "ppo", _freeze_json(self.ppo))

    def manifest(self) -> dict[str, object]:
        return {
            "plan_sha256": self.plan_sha256,
            "arm_id": self.arm_id,
            "learner_seed": self.learner_seed,
            "seed_plan_sha256": self.seed_plan_sha256,
            "initial_model_sha256": self.initial_model_sha256,
            "curriculum_sha256": self.curriculum_sha256,
            "assignment_sha256": self.assignment_sha256,
            "snapshot_store_sha256": self.snapshot_store_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "reward_sha256": self.reward_sha256,
            "collector": _thaw_json(self.collector),
            "ppo": _thaw_json(self.ppo),
            "lanes": self.lanes,
            "deployable": self.deployable,
            "observation_provenance": self.observation_provenance,
            "transfer_gate": self.transfer_gate,
            "version": self.version,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(slots=True)
class BuiltTrial:
    session: R3ATrainingSession
    manifest: TrialManifest
    environment: Any

    def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if close is not None:
            close()

    def __enter__(self) -> BuiltTrial:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class R3BRunBuilder:
    """Build paired trials without allowing arm-specific hidden configuration."""

    def __init__(
        self,
        plan: R3BExperimentPlan,
        curriculum: CurriculumSpec,
        snapshots: SnapshotBlobStore,
        *,
        runtime_identity_sha256: str,
        lanes: int,
        collector_config: CollectorConfig,
        ppo_config: PPOConfig,
        model_factory: Callable[[], RecurrentActorCritic],
        environment_factory: Callable[[], Any],
        reward_scale: float = 1000.0,
        max_consecutive_skips: int = 64,
    ) -> None:
        if len(curriculum.stages) != 1:
            raise ValueError(
                "paired R3b sweeps require one frozen stage; adaptive promotion is forbidden"
            )
        if snapshots.library is not curriculum.library:
            raise ValueError("snapshot store and curriculum library must be identical")
        if collector_config.target_simulated_ticks != plan.ticks_per_update:
            raise ValueError("collector tick budget disagrees with the experiment plan")
        if ppo_config.final_learning_rate_fraction != plan.final_learning_rate_fraction:
            raise ValueError("PPO learning-rate schedule disagrees with the plan")
        if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes <= 0:
            raise ValueError("trial lane count must be positive")
        if (
            isinstance(reward_scale, bool)
            or not isinstance(reward_scale, (int, float))
            or reward_scale <= 0
        ):
            raise ValueError("reward scale must be positive")
        if (
            isinstance(max_consecutive_skips, bool)
            or not isinstance(max_consecutive_skips, int)
            or max_consecutive_skips <= 0
        ):
            raise ValueError("maximum consecutive skips must be positive")
        self.plan = plan
        self.base_curriculum = curriculum
        self.snapshots = snapshots
        self.runtime_identity_sha256 = runtime_identity_sha256
        self.lanes = lanes
        self.collector_config = collector_config
        self.ppo_config = ppo_config
        self.model_factory = model_factory
        self.environment_factory = environment_factory
        self.reward_scale = float(reward_scale)
        self.max_consecutive_skips = max_consecutive_skips

    def _curriculum(self, arm: CandidateArm) -> CurriculumSpec:
        if arm not in self.plan.arms:
            raise ValueError("trial arm is absent from the frozen plan")
        alpha = arm.alpha_weight_ppm
        knots = (
            (RewardKnot(0, 0),)
            if alpha == 0
            else (
                RewardKnot(0, alpha),
                RewardKnot(self.plan.shaped_updates - 1, alpha),
                RewardKnot(self.plan.shaped_updates, 0),
            )
        )
        schedule = RewardSchedule(f"r3b-{arm.arm_id}", knots)
        stage = replace(
            self.base_curriculum.stages[0],
            max_updates=self.plan.total_updates,
            reward_schedule=schedule,
        )
        return replace(self.base_curriculum, stages=(stage,))

    def build(self, arm: CandidateArm, learner_seed: int) -> BuiltTrial:
        seed_plan = TrialSeedPlan.derive(self.plan.sha256, learner_seed)
        curriculum = self._curriculum(arm)
        environment = self.environment_factory()
        try:
            if int(environment.num_envs) != self.lanes:
                raise ValueError("environment factory returned the wrong lane count")
            coordinator = CurriculumCoordinator(
                curriculum, self.lanes, learner_seed=seed_plan.assignment
            )
            initializer = CurriculumSnapshotInitializer(
                coordinator,
                self.snapshots,
                environment_pool=curriculum.stages[0].environment_pool,
                runtime_identity_sha256=self.runtime_identity_sha256,
            )
            with torch.random.fork_rng(devices=[], enabled=True):
                torch.manual_seed(seed_plan.model_initialization)
                model = self.model_factory()
            if model.config.critic_condition_features != 1:
                raise ValueError("every R3b arm requires the same conditioned critic")
            initial_model_sha256 = model_state_sha256(model)
            composer = RewardComposer(
                reward_scale=self.reward_scale,
                shaping_spec=LinearGaugePotential(
                    gamma_tick=self.collector_config.gamma_tick
                ),
            )
            task = CurriculumTaskContract(
                coordinator,
                composer,
                capture_events=False,
                snapshot_initializer=initializer,
            )
            adapter = MacroVectorAdapter(
                environment,
                encoder=TeacherStateEncoder(),
                capture_events=False,
                episode_initializer=initializer,
            )
            collector = RecurrentCollector(
                model,
                adapter,
                task,
                config=self.collector_config,
                policy_sampler_seed=seed_plan.policy_sampling,
            )
            trial_ppo = replace(self.ppo_config, learning_rate=arm.learning_rate)
            trainer = PPOTrainer(
                model,
                config=trial_ppo,
                total_updates=self.plan.total_updates,
                sampler_seed=seed_plan.ppo_minibatching,
            )
            tail = ScoreOnlyTailController(
                self.plan.shaped_updates,
                minimum_score_only_updates=self.plan.zero_tail_updates,
            )
            session = R3ATrainingSession(
                collector,
                trainer,
                numpy_seed=seed_plan.session_numpy,
                max_consecutive_skips=self.max_consecutive_skips,
                tail_controller=tail,
            )
            manifest = TrialManifest(
                self.plan.sha256,
                arm.arm_id,
                learner_seed,
                seed_plan.sha256,
                initial_model_sha256,
                curriculum.sha256,
                curriculum.assignment_sha256,
                self.snapshots.sha256,
                self.runtime_identity_sha256,
                composer.sha256,
                self.collector_config.manifest(),
                trial_ppo.manifest(),
                self.lanes,
            )
            return BuiltTrial(session, manifest, environment)
        except BaseException:
            close = getattr(environment, "close", None)
            if close is not None:
                close()
            raise
