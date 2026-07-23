"""Construction of identity-bound R3b training trials."""

from __future__ import annotations

import hashlib
import inspect
import json
import marshal
import math
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any, Callable, Mapping

import numpy as np
import torch

from .collector import (
    CollectorConfig,
    CurriculumTaskContract,
    R3ATrainingSession,
    RecurrentCollector,
    model_state_sha256,
)
from .checkpoints import capture_rng_state, restore_rng_state
from .curriculum import (
    CurriculumCoordinator,
    CurriculumSnapshotInitializer,
    CurriculumSpec,
    SnapshotBlobStore,
)
from .encoding import TeacherStateEncoder
from .models import RecurrentActorCritic
from .ppo import PPOConfig, PPOTrainer
from .r3b_evaluation import (
    DeploymentPolicyIdentity,
    LearnedPolicyBackendParityArtifact,
    behavior_build_identity_sha256,
    encoder_instance_manifest,
)
from .r3b_experiments import (
    CandidateArm,
    EngineeringEvidence,
    ExactResumeArtifact,
    RawScoreMetricsArtifact,
    R3BExperimentPlan,
    SealedTestLedger,
    SealedTestJobLease,
    TrainingCheckpointArtifact,
    TrialJob,
    TrialSeedPlan,
    ValidationRunAuthorization,
    _verified_exact_resume_artifact,
)
from .r3b_tail import ScoreOnlyTailController
from .rewards import (
    LinearGaugePotential,
    RewardComposer,
    RewardKnot,
    RewardSchedule,
)
from .vector_adapter import MacroVectorAdapter
from .runtime_identity import SimulatorRuntimeAttestation, attest_simulator_runtime


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


def _audit_normalize(value: object) -> object:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        return {
            "tensor_dtype": str(tensor.dtype),
            "tensor_shape": list(tensor.shape),
            "tensor_sha256": hashlib.sha256(
                tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            ).hexdigest(),
        }
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "numpy_dtype": array.dtype.str,
            "numpy_shape": list(array.shape),
            "numpy_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _audit_normalize(getattr(value, field.name))
                for field in fields(value)
                if field.init
            },
        }
    if isinstance(value, Mapping):
        return {
            str(key): _audit_normalize(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_audit_normalize(item) for item in value]
    if isinstance(value, bytes):
        return {
            "bytes": len(value),
            "sha256": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _audit_normalize(value.item())
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise TypeError(f"resume audit cannot normalize {type(value).__name__}")


def _audit_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            _audit_normalize(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _implementation_identity(
    value: object, *, bind_callable_state: bool = False
) -> dict[str, str]:
    """Bind a concrete callable/type to executable Python implementation bytes."""

    if inspect.ismethod(value):
        target = value.__func__
        bound_state: object = (
            getattr(value.__self__, "__dict__", {}) if bind_callable_state else {}
        )
    elif inspect.isfunction(value) or inspect.isclass(value):
        target = value
        bound_state = {}
    else:
        target = type(value)
        bound_state = getattr(value, "__dict__", {}) if bind_callable_state else {}
    code = getattr(target, "__code__", None)
    try:
        source = inspect.getsource(target).encode()
    except (OSError, TypeError):
        source = b""
    try:
        source_path = inspect.getsourcefile(target)
    except TypeError:
        source_path = None
    source_file_path = Path(source_path) if source_path else None
    source_file = (
        source_file_path.read_bytes()
        if source_file_path is not None and source_file_path.is_file()
        else b""
    )
    if not source and code is None and not source_file:
        raise TypeError("runner implementation identity is not inspectable")

    def capture(value: object) -> object:
        if inspect.isfunction(value) or inspect.isclass(value):
            return _implementation_identity(value)
        try:
            return _audit_normalize(value)
        except TypeError:
            return {
                "opaque_type": f"{type(value).__module__}.{type(value).__qualname__}"
            }

    closure_state = (
        tuple(capture(cell.cell_contents) for cell in (target.__closure__ or ()))
        if bind_callable_state and inspect.isfunction(target)
        else ()
    )
    return {
        "qualified_name": f"{target.__module__}.{target.__qualname__}",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "source_file_sha256": hashlib.sha256(source_file).hexdigest(),
        "code_sha256": hashlib.sha256(
            b"" if code is None else marshal.dumps(code)
        ).hexdigest(),
        "state_sha256": _audit_sha256(
            {
                "defaults": getattr(target, "__defaults__", None),
                "keyword_defaults": getattr(target, "__kwdefaults__", None),
                "closure": closure_state,
                "bound_state": capture(bound_state),
            }
        ),
    }


def _session_continuation_state(session: R3ATrainingSession) -> dict[str, object]:
    return {
        "model": session.model.state_dict(),
        "trainer": session.trainer.state_dict(),
        "collector": session.collector.state_dict(),
        "task": session.task.state_dict(),
        "adapter": session.collector.adapter.checkpoint(),
        "tail": (
            None
            if session.tail_controller is None
            else session.tail_controller.state_dict()
        ),
        "attempted_rollouts": session.attempted_rollouts,
        "skipped_rollouts": session.skipped_rollouts,
        "consecutive_skips": session.consecutive_skips,
        "max_consecutive_skips": session.max_consecutive_skips,
        "optimizer_update_limit": session.optimizer_update_limit,
        "rng": capture_rng_state(session.numpy_generator),
    }


def verify_exact_resume_continuation(
    *,
    trial_manifest_sha256: str,
    checkpoint: TrainingCheckpointArtifact,
    checkpoint_directory: str | Path,
    generation: str,
    checkpoint_identity: Mapping[str, object],
    source: R3ATrainingSession,
    restored_factory: Callable[[], R3ATrainingSession],
) -> ExactResumeArtifact:
    """Restore a bound checkpoint, then prove its next update is exact."""

    if (
        not isinstance(checkpoint, TrainingCheckpointArtifact)
        or checkpoint.trial_manifest_sha256 != trial_manifest_sha256
        or not isinstance(checkpoint_identity, Mapping)
        or checkpoint_identity.get("trial_manifest_sha256") != trial_manifest_sha256
        or source.policy_sha256 != checkpoint.model_sha256
        or source.trainer.schedule.completed_updates != checkpoint.completed_updates
        or source.collector.simulated_ticks != checkpoint.simulated_ticks
    ):
        raise ValueError("exact-resume source disagrees with the typed checkpoint")
    manifest_path = Path(checkpoint_directory) / generation / "manifest.json"
    if (
        not manifest_path.is_file()
        or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        != checkpoint.checkpoint_manifest_sha256
    ):
        raise ValueError("exact-resume checkpoint manifest bytes disagree")
    restored = restored_factory()
    if not isinstance(restored, R3ATrainingSession) or restored is source:
        raise ValueError("exact-resume verification requires an independent session")
    restored.restore(
        checkpoint_directory,
        generation=generation,
        identity=checkpoint_identity,
    )
    source_policy = source.policy_sha256
    if restored.policy_sha256 != source_policy:
        raise ValueError("restored checkpoint policy differs before continuation")
    initial_rng = capture_rng_state(source.numpy_generator)
    initial_source = _audit_sha256(_session_continuation_state(source))
    initial_restored = _audit_sha256(_session_continuation_state(restored))
    if initial_source != initial_restored:
        raise ValueError("restored checkpoint state differs before continuation")
    source_update = source.run_update()
    source_update_sha256 = _audit_sha256(source_update)
    source_after_sha256 = _audit_sha256(_session_continuation_state(source))
    restore_rng_state(initial_rng, restored.numpy_generator)
    restored_update = restored.run_update()
    restored_update_sha256 = _audit_sha256(restored_update)
    restored_after_sha256 = _audit_sha256(_session_continuation_state(restored))
    return _verified_exact_resume_artifact(
        trial_manifest_sha256,
        checkpoint.checkpoint_manifest_sha256,
        source_policy,
        source_update_sha256,
        restored_update_sha256,
        source_after_sha256,
        restored_after_sha256,
    )


@dataclass(frozen=True, slots=True)
class TrialManifest:
    plan_sha256: str
    job_sha256: str
    phase: str
    arm_id: str
    learner_seed: int
    budget_updates: int
    sealed: bool
    authorization_sha256: str | None
    seed_plan_sha256: str
    initial_model_sha256: str
    curriculum_sha256: str
    assignment_sha256: str
    library_sha256: str
    snapshot_store_sha256: str
    runtime_identity_sha256: str
    action_spec_sha256: str
    reward_sha256: str
    runner_spec_sha256: str
    pairing_sha256: str
    collector: Mapping[str, object]
    ppo: Mapping[str, object]
    lanes: int
    deployable: bool = False
    observation_provenance: str = "privileged_simulator"
    transfer_gate: str = "R4 causal tracker and input calibration pending"
    version: str = "r3b-trial-manifest-v4"

    def __post_init__(self) -> None:
        hashes = (
            self.plan_sha256,
            self.job_sha256,
            self.seed_plan_sha256,
            self.initial_model_sha256,
            self.curriculum_sha256,
            self.assignment_sha256,
            self.library_sha256,
            self.snapshot_store_sha256,
            self.runtime_identity_sha256,
            self.action_spec_sha256,
            self.reward_sha256,
            self.runner_spec_sha256,
            self.pairing_sha256,
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
            self.version != "r3b-trial-manifest-v4"
            or self.phase not in {"calibration", "validation", "test"}
            or not self.arm_id
            or isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or not 0 <= self.learner_seed < 2**64
            or isinstance(self.lanes, bool)
            or not isinstance(self.lanes, int)
            or self.lanes <= 0
            or isinstance(self.budget_updates, bool)
            or not isinstance(self.budget_updates, int)
            or self.budget_updates <= 0
            or not isinstance(self.sealed, bool)
            or self.sealed != (self.phase == "test")
            or (
                self.phase != "calibration"
                and (
                    not isinstance(self.authorization_sha256, str)
                    or len(self.authorization_sha256) != 64
                    or self.authorization_sha256 == "0" * 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in self.authorization_sha256
                    )
                )
            )
            or (self.phase == "calibration" and self.authorization_sha256 is not None)
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
            "job_sha256": self.job_sha256,
            "phase": self.phase,
            "arm_id": self.arm_id,
            "learner_seed": self.learner_seed,
            "budget_updates": self.budget_updates,
            "sealed": self.sealed,
            "authorization_sha256": self.authorization_sha256,
            "seed_plan_sha256": self.seed_plan_sha256,
            "initial_model_sha256": self.initial_model_sha256,
            "curriculum_sha256": self.curriculum_sha256,
            "assignment_sha256": self.assignment_sha256,
            "library_sha256": self.library_sha256,
            "snapshot_store_sha256": self.snapshot_store_sha256,
            "runtime_identity_sha256": self.runtime_identity_sha256,
            "action_spec_sha256": self.action_spec_sha256,
            "reward_sha256": self.reward_sha256,
            "runner_spec_sha256": self.runner_spec_sha256,
            "pairing_sha256": self.pairing_sha256,
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
    sealed_job_lease_sha256: str | None = None

    def close(self) -> None:
        close = getattr(self.environment, "close", None)
        if close is not None:
            close()

    def engineering_evidence(
        self,
        *,
        metrics_artifact: RawScoreMetricsArtifact,
        deployment_identity: DeploymentPolicyIdentity,
        checkpoint_resume_artifact: ExactResumeArtifact,
        exact_backend_parity_artifact: LearnedPolicyBackendParityArtifact,
    ) -> EngineeringEvidence:
        """Bind completed session state to the external audit artifacts."""

        self.session.assert_evidence_ready()
        completed = self.session.trainer.schedule.completed_updates
        model_sha256 = self.session.policy_sha256
        final_checkpoint = (
            metrics_artifact.checkpoints[-1].checkpoint
            if isinstance(metrics_artifact, RawScoreMetricsArtifact)
            else None
        )
        resume_checkpoint = (
            metrics_artifact.checkpoints[-2].checkpoint
            if isinstance(metrics_artifact, RawScoreMetricsArtifact)
            and len(metrics_artifact.checkpoints) >= 2
            else None
        )
        if (
            not isinstance(metrics_artifact, RawScoreMetricsArtifact)
            or not isinstance(deployment_identity, DeploymentPolicyIdentity)
            or metrics_artifact.suite.split != self.manifest.phase
            or metrics_artifact.final_report.policy_sha256 != deployment_identity.sha256
            or deployment_identity.model_sha256 != model_sha256
            or final_checkpoint is None
            or resume_checkpoint is None
            or final_checkpoint.completed_updates != completed
            or final_checkpoint.simulated_ticks
            != self.session.collector.simulated_ticks
            or final_checkpoint.plan_sha256 != self.manifest.plan_sha256
            or final_checkpoint.job_sha256 != self.manifest.job_sha256
            or final_checkpoint.trial_manifest_sha256 != self.manifest.sha256
            or final_checkpoint.runner_spec_sha256 != self.manifest.runner_spec_sha256
            or final_checkpoint.model_sha256 != model_sha256
            or final_checkpoint.deployment_policy_sha256 != deployment_identity.sha256
            or metrics_artifact.suite.runtime_identity_sha256
            != self.manifest.runtime_identity_sha256
            or metrics_artifact.suite.assignment_sha256
            != self.manifest.assignment_sha256
            or metrics_artifact.suite.library_sha256 != self.manifest.library_sha256
            or metrics_artifact.suite.snapshot_store_sha256
            != self.manifest.snapshot_store_sha256
            or metrics_artifact.suite.action_spec_sha256
            != self.manifest.action_spec_sha256
            or deployment_identity.action_spec_sha256
            != self.manifest.action_spec_sha256
            or not isinstance(checkpoint_resume_artifact, ExactResumeArtifact)
            or checkpoint_resume_artifact.trial_manifest_sha256 != self.manifest.sha256
            or checkpoint_resume_artifact.checkpoint_manifest_sha256
            != resume_checkpoint.checkpoint_manifest_sha256
            or checkpoint_resume_artifact.checkpoint_model_sha256
            != resume_checkpoint.model_sha256
            or not isinstance(
                exact_backend_parity_artifact,
                LearnedPolicyBackendParityArtifact,
            )
            or exact_backend_parity_artifact.portable_suite.sha256
            != metrics_artifact.suite.sha256
            or exact_backend_parity_artifact.portable_report.sha256
            != metrics_artifact.final_report.sha256
            or exact_backend_parity_artifact.policy_sha256 != deployment_identity.sha256
        ):
            raise ValueError(
                "evaluation artifacts do not belong to the completed trial"
            )
        tail_state = None
        tail_state_sha256 = None
        tail_phase = None
        score_only_updates = 0
        if self.session.tail_controller is not None:
            tail_state = self.session.tail_controller.state_dict()
            tail_phase = str(tail_state["phase"])
            score_only_updates = int(tail_state["score_only_updates"])
            tail_state_sha256 = hashlib.sha256(
                json.dumps(
                    tail_state,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
        return EngineeringEvidence(
            phase=self.manifest.phase,
            completed_updates=completed,
            plan_sha256=self.manifest.plan_sha256,
            job_sha256=self.manifest.job_sha256,
            arm_id=self.manifest.arm_id,
            learner_seed=self.manifest.learner_seed,
            authorization_sha256=self.manifest.authorization_sha256,
            sealed_job_lease_sha256=self.sealed_job_lease_sha256,
            policy_sha256=deployment_identity.sha256,
            trial_manifest_sha256=self.manifest.sha256,
            runner_spec_sha256=self.manifest.runner_spec_sha256,
            pairing_sha256=self.manifest.pairing_sha256,
            metrics_sha256=metrics_artifact.sha256,
            evaluation_suite_sha256=metrics_artifact.suite.sha256,
            evaluation_report_sha256=metrics_artifact.final_report.sha256,
            final_checkpoint_artifact=final_checkpoint,
            resume_checkpoint_artifact=resume_checkpoint,
            checkpoint_resume_artifact=checkpoint_resume_artifact,
            exact_backend_parity_artifact=exact_backend_parity_artifact,
            tail_state_sha256=tail_state_sha256,
            tail_phase=tail_phase,
            score_only_updates=score_only_updates,
        )

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
        runtime_attestation: SimulatorRuntimeAttestation,
        lanes: int,
        collector_config: CollectorConfig,
        ppo_config: PPOConfig,
        model_factory: Callable[[], RecurrentActorCritic],
        environment_factory: Callable[[], Any],
        sealed_test_ledger: SealedTestLedger | None = None,
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
        if not isinstance(runtime_attestation, SimulatorRuntimeAttestation):
            raise TypeError("runner requires a measured simulator runtime attestation")
        self.runtime_attestation = runtime_attestation
        self.runtime_identity_sha256 = runtime_attestation.sha256
        self.lanes = lanes
        self.collector_config = collector_config
        self.ppo_config = ppo_config
        self.model_factory = model_factory
        self.environment_factory = environment_factory
        if (
            sealed_test_ledger is not None
            and type(sealed_test_ledger) is not SealedTestLedger
        ):
            raise TypeError("sealed-test ledger must be a trusted ledger instance")
        self._sealed_test_ledger = sealed_test_ledger
        self.reward_scale = float(reward_scale)
        self.max_consecutive_skips = max_consecutive_skips
        self._runner_spec_sha256: str | None = None
        self._runner_spec_lock = Lock()

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

    def _validate_job(
        self,
        job: TrialJob,
        authorization: ValidationRunAuthorization | SealedTestJobLease | None,
        sealed_test_ledger: SealedTestLedger | None,
    ) -> None:
        if not isinstance(job, TrialJob) or job.plan_sha256 != self.plan.sha256:
            raise ValueError("trial job does not belong to the frozen plan")
        if job.arm not in self.plan.arms:
            raise ValueError("trial job arm is absent from the frozen plan")
        phase_contract = {
            "calibration": (
                self.plan.calibration_learner_seeds,
                self.plan.calibration_budgets_updates,
                False,
            ),
            "validation": (
                self.plan.validation_learner_seeds,
                (self.plan.validation_updates,),
                False,
            ),
            "test": (
                self.plan.test_learner_seeds,
                (self.plan.test_updates,),
                True,
            ),
        }
        seeds, budgets, sealed = phase_contract[job.phase]
        expected_seed_plan = TrialSeedPlan.derive(
            self.plan.sha256, job.learner_seed
        ).sha256
        if (
            job.learner_seed not in seeds
            or job.budget_updates not in budgets
            or job.sealed != sealed
            or job.seed_plan_sha256 != expected_seed_plan
            or (job.phase != "calibration" and job.authorization_sha256 is None)
        ):
            raise ValueError("trial job violates its frozen phase contract")
        if job.phase == "calibration":
            valid_authorization = authorization is None
        elif job.phase == "validation":
            valid_authorization = (
                isinstance(authorization, ValidationRunAuthorization)
                and authorization.plan.sha256 == self.plan.sha256
                and job.arm in authorization.authorization.arms
                and job.authorization_sha256 == authorization.sha256
            )
        else:
            valid_authorization = (
                isinstance(authorization, SealedTestJobLease)
                and authorization.sealed_run.plan.sha256 == self.plan.sha256
                and authorization.job == job
                and job.arm.arm_id
                in {
                    authorization.sealed_run.authorization.control_arm_id,
                    authorization.sealed_run.authorization.candidate_arm_id,
                }
                and job.authorization_sha256 == authorization.sealed_run.sha256
                and sealed_test_ledger is not None
                and sealed_test_ledger.path
                == Path(authorization.sealed_run.ledger_path)
            )
            if valid_authorization:
                authorization.assert_active()
        if not valid_authorization:
            raise ValueError("trial job lacks its phase-selection authorization")

    def build(
        self,
        job: TrialJob,
        *,
        authorization: ValidationRunAuthorization | SealedTestJobLease | None = None,
    ) -> BuiltTrial:
        sealed_test_ledger = self._sealed_test_ledger
        self._validate_job(job, authorization, sealed_test_ledger)
        arm = job.arm
        learner_seed = job.learner_seed
        seed_plan = TrialSeedPlan.derive(self.plan.sha256, learner_seed)
        curriculum = self._curriculum(arm)
        environment = None
        lease_started = False
        try:
            if isinstance(authorization, SealedTestJobLease):
                assert sealed_test_ledger is not None
                sealed_test_ledger.begin_job(authorization)
                lease_started = True
            environment = self.environment_factory()
            if int(environment.num_envs) != self.lanes:
                raise ValueError("environment factory returned the wrong lane count")
            actual_runtime = attest_simulator_runtime(environment)
            if (
                actual_runtime.sha256 != self.runtime_attestation.sha256
                or actual_runtime.verified_lanes != self.lanes
            ):
                raise RuntimeError("environment factory returned an unattested runtime")
            coordinator = CurriculumCoordinator(
                curriculum, self.lanes, learner_seed=seed_plan.assignment
            )
            initializer = CurriculumSnapshotInitializer(
                coordinator,
                self.snapshots,
                environment_pool=curriculum.stages[0].environment_pool,
                runtime_attestation=actual_runtime,
            )
            cuda_devices = (
                list(range(torch.cuda.device_count()))
                if torch.cuda.is_available()
                else []
            )
            with torch.random.fork_rng(devices=cuda_devices, enabled=True):
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
            encoder = TeacherStateEncoder()
            encoder_manifest = encoder_instance_manifest(encoder)
            implementation_identity = {
                "model": _implementation_identity(model),
                "model_factory": _implementation_identity(
                    self.model_factory, bind_callable_state=True
                ),
                "environment": _implementation_identity(environment),
                "environment_factory": _implementation_identity(
                    self.environment_factory, bind_callable_state=True
                ),
            }
            curriculum_shared = {
                **{
                    key: value
                    for key, value in curriculum.manifest().items()
                    if key not in {"stages", "assignment_sha256"}
                },
                "assignment_sha256": curriculum.assignment_sha256,
                "stages": [
                    {
                        key: value
                        for key, value in stage.manifest().items()
                        if key != "reward_schedule"
                    }
                    for stage in curriculum.stages
                ],
            }
            build_identity_sha256 = behavior_build_identity_sha256(
                {
                    "purpose": "r3b-training-runner-v1",
                    "model": model.manifest(),
                    "implementations": implementation_identity,
                    "encoder": encoder_manifest,
                    "collector": self.collector_config.manifest(),
                    "ppo_without_learning_rate": {
                        key: value
                        for key, value in self.ppo_config.manifest().items()
                        if key != "learning_rate"
                    },
                    "curriculum_shared": curriculum_shared,
                }
            )
            runner_spec_sha256 = hashlib.sha256(
                json.dumps(
                    {
                        "version": "r3b-runner-spec-v1",
                        "plan_sha256": self.plan.sha256,
                        "model": model.manifest(),
                        "implementations": implementation_identity,
                        "encoder": encoder_manifest,
                        "collector": self.collector_config.manifest(),
                        "ppo_without_learning_rate": {
                            key: value
                            for key, value in self.ppo_config.manifest().items()
                            if key != "learning_rate"
                        },
                        "lanes": self.lanes,
                        "reward_scale": self.reward_scale,
                        "max_consecutive_skips": self.max_consecutive_skips,
                        "runtime_identity_sha256": actual_runtime.sha256,
                        "snapshot_store_sha256": self.snapshots.sha256,
                        "curriculum_shared": curriculum_shared,
                        "build_identity_sha256": build_identity_sha256,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            with self._runner_spec_lock:
                if self._runner_spec_sha256 is None:
                    self._runner_spec_sha256 = runner_spec_sha256
                elif self._runner_spec_sha256 != runner_spec_sha256:
                    raise RuntimeError(
                        "runner implementation changed the frozen runner specification"
                    )
            task = CurriculumTaskContract(
                coordinator,
                composer,
                capture_events=False,
                snapshot_initializer=initializer,
            )
            adapter = MacroVectorAdapter(
                environment,
                encoder=encoder,
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
            tail = (
                ScoreOnlyTailController(
                    self.plan.shaped_updates,
                    minimum_score_only_updates=self.plan.zero_tail_updates,
                    reward_scale=composer.reward_scale,
                    reward_sha256=composer.sha256,
                )
                if job.budget_updates == self.plan.total_updates
                else None
            )
            session = R3ATrainingSession(
                collector,
                trainer,
                numpy_seed=seed_plan.session_numpy,
                max_consecutive_skips=self.max_consecutive_skips,
                tail_controller=tail,
                optimizer_update_limit=job.budget_updates,
            )
            pairing_payload = {
                "version": "r3b-pairing-v1",
                "plan_sha256": self.plan.sha256,
                "phase": job.phase,
                "learner_seed": learner_seed,
                "budget_updates": job.budget_updates,
                "authorization_sha256": job.authorization_sha256,
                "seed_plan_sha256": seed_plan.sha256,
                "initial_model_sha256": initial_model_sha256,
                "assignment_sha256": curriculum.assignment_sha256,
                "curriculum_shared": curriculum_shared,
                "library_sha256": self.snapshots.library.sha256,
                "snapshot_store_sha256": self.snapshots.sha256,
                "runtime_identity_sha256": self.runtime_identity_sha256,
                "action_spec_sha256": model.action_spec.sha256,
                "reward_sha256": composer.sha256,
                "runner_spec_sha256": runner_spec_sha256,
                "collector": self.collector_config.manifest(),
                "ppo_without_learning_rate": {
                    key: value
                    for key, value in trial_ppo.manifest().items()
                    if key != "learning_rate"
                },
                "lanes": self.lanes,
                "max_consecutive_skips": self.max_consecutive_skips,
            }
            pairing_sha256 = hashlib.sha256(
                json.dumps(
                    pairing_payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
            ).hexdigest()
            manifest = TrialManifest(
                self.plan.sha256,
                job.sha256,
                job.phase,
                arm.arm_id,
                learner_seed,
                job.budget_updates,
                job.sealed,
                job.authorization_sha256,
                seed_plan.sha256,
                initial_model_sha256,
                curriculum.sha256,
                curriculum.assignment_sha256,
                self.snapshots.library.sha256,
                self.snapshots.sha256,
                self.runtime_identity_sha256,
                model.action_spec.sha256,
                composer.sha256,
                runner_spec_sha256,
                pairing_sha256,
                self.collector_config.manifest(),
                trial_ppo.manifest(),
                self.lanes,
            )
            return BuiltTrial(
                session,
                manifest,
                environment,
                authorization.sha256
                if isinstance(authorization, SealedTestJobLease)
                else None,
            )
        except BaseException as error:
            close = getattr(environment, "close", None)
            if close is not None:
                close()
            if lease_started:
                assert isinstance(authorization, SealedTestJobLease)
                assert sealed_test_ledger is not None
                try:
                    sealed_test_ledger.fail_job(
                        authorization,
                        f"runner construction failed: {type(error).__name__}: {error}",
                    )
                except BaseException as ledger_error:
                    error.add_note(
                        "sealed-test failure recording also failed: "
                        f"{type(ledger_error).__name__}: {ledger_error}"
                    )
            raise
