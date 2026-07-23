"""Bounded local execution of identity-bound R3b smoke training jobs."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
from dataclasses import dataclass
from pathlib import Path

import torch

from irisu_env import IrisuEnv, PaddedVectorEnv

from .actions import ActionSpec
from .collector import CollectorConfig
from .curriculum import CurriculumSpec, StageSpec
from .encoding import TeacherStateEncoder
from .models import RecurrentActorCritic, RecurrentModelConfig
from .ppo import PPOConfig
from .r3b_artifacts import ArtifactStore
from .r3b_evaluation import DeploymentPolicyIdentity
from .r3b_experiments import (
    R3BExperimentPlan,
    SealedTestJobLease,
    SealedTestLedger,
    TrainingCheckpointArtifact,
    TrialJob,
    ValidationRunAuthorization,
)
from .r3b_operational import JobClaim, R3BOperationalConfig, R3BWorkflow
from .r3b_runner import R3BRunBuilder
from .r3b_snapshots import load_snapshot_bundle
from .rewards import RewardKnot, RewardSchedule
from .runtime_identity import attest_simulator_runtime
from .schema import TEACHER_V1


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _source_identity(project_root: Path) -> str:
    inputs = (
        sorted((project_root / "python" / "irisu_env").glob("*.py"))
        + sorted((project_root / "python" / "irisu_rl").glob("*.py"))
        + [project_root / "pyproject.toml", project_root / "uv.lock"]
    )
    manifest = []
    for path in inputs:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"source identity input is missing or unsafe: {path}")
        payload = path.read_bytes()
        manifest.append(
            {
                "path": path.relative_to(project_root).as_posix(),
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    return hashlib.sha256(_canonical_bytes(manifest)).hexdigest()


def _clean_source_revision(project_root: Path) -> str:
    status = subprocess.run(
        (
            "git",
            "status",
            "--porcelain",
            "--untracked-files=all",
            "--",
            "python",
            "configs/rl",
            "pyproject.toml",
            "uv.lock",
        ),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = subprocess.run(
        ("git", "rev-parse", "--verify", "HEAD"),
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = revision.stdout.strip()
    if (
        status.returncode
        or status.stdout
        or revision.returncode
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(
            "canonical execution requires its clean reviewed source revision"
        )
    return value


def _read_resolved_run(root: Path) -> dict[str, object]:
    path = root / "resolved-run.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("resolved run manifest is missing or unsafe")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("resolved run manifest is malformed") from exc
    if (
        not isinstance(value, dict)
        or value.get("version") != "r3b-resolved-run-v1"
        or payload != _canonical_bytes(value) + b"\n"
    ):
        raise ValueError("resolved run manifest is noncanonical or unsupported")
    return value


def _write_claim(path: Path, claim: JobClaim) -> None:
    path.parent.mkdir(mode=0o700, exist_ok=True)
    os.chmod(path.parent, 0o700)
    payload = (
        _canonical_bytes(
            {
                "version": "r3b-local-claim-v1",
                "job_sha256": claim.job_sha256,
                "phase": claim.phase,
                "token": claim.token,
                "owner": claim.owner,
            }
        )
        + b"\n"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _load_claim(path: Path) -> JobClaim:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("claim secret is missing, linked, or not private")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("claim secret is malformed") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "job_sha256", "phase", "token", "owner"}
        or value["version"] != "r3b-local-claim-v1"
        or payload != _canonical_bytes(value) + b"\n"
    ):
        raise ValueError("claim secret schema differs")
    return JobClaim(
        value["job_sha256"],
        value["phase"],
        value["token"],
        value["owner"],
        0,
        None,
    )


def _write_claim_intent(path: Path, *, phase: str, token: str, owner: str) -> None:
    path.parent.mkdir(mode=0o700, exist_ok=True)
    os.chmod(path.parent, 0o700)
    payload = (
        _canonical_bytes(
            {
                "version": "r3b-claim-intent-v1",
                "phase": phase,
                "token": token,
                "owner": owner,
            }
        )
        + b"\n"
    )
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _load_claim_intent(path: Path) -> tuple[str, str, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("claim intent is missing, linked, or not private")
    payload = path.read_bytes()
    value = json.loads(payload)
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "phase", "token", "owner"}
        or value["version"] != "r3b-claim-intent-v1"
        or payload != _canonical_bytes(value) + b"\n"
        or any(type(value[name]) is not str for name in value)
    ):
        raise ValueError("claim intent schema differs")
    return value["phase"], value["token"], value["owner"]


def _curriculum(bundle: object, total_updates: int) -> CurriculumSpec:
    library = bundle.library
    recipes = library.recipes
    train = tuple(value.snapshot_id for value in recipes if value.split == "train")
    validation = tuple(
        value.snapshot_id for value in recipes if value.split == "validation"
    )
    pools = {value.environment_pool for value in recipes}
    stages = {value.stage_id for value in recipes}
    if len(pools) != 1 or len(stages) != 1 or not train or not validation:
        raise ValueError("snapshot bundle cannot form one frozen full-game stage")
    trials = len(validation)
    stage = StageSpec(
        next(iter(stages)),
        0,
        next(iter(pools)),
        train,
        validation,
        (0, 1, 2),
        ActionSpec().wait_choices,
        trials,
        trials,
        trials,
        trials,
        1,
        total_updates,
        RewardSchedule("r3b-builder-placeholder", (RewardKnot(0, 0),)),
    )
    return CurriculumSpec(
        "r3b-full-game-v1",
        library,
        (stage,),
        evaluation_seed=2026072201,
        prior_stage_mix_ppm=0,
    )


def _builder(
    *,
    plan: R3BExperimentPlan,
    config: R3BOperationalConfig,
    bundle: object,
    worker_path: Path,
    sealed_test_ledger: SealedTestLedger | None = None,
) -> R3BRunBuilder:
    if config.collector_max_decisions * config.lanes < plan.ticks_per_update:
        raise ValueError("collector decision capacity cannot guarantee its tick target")
    torch.set_num_threads(config.torch_threads)
    model_config = RecurrentModelConfig(
        config.model_global_hidden,
        config.model_body_hidden,
        config.model_fused_hidden,
        config.model_recurrent_hidden,
        config.model_recurrent_layers,
        critic_condition_features=1,
    )
    worker = str(worker_path)
    lanes = config.lanes
    workers = config.workers

    def model_factory() -> RecurrentActorCritic:
        return RecurrentActorCritic(TEACHER_V1, config=model_config)

    def environment_factory() -> PaddedVectorEnv:
        return PaddedVectorEnv(
            lanes,
            workers=workers,
            physics_backend="exact",
            worker_path=worker,
        )

    preflight = environment_factory()
    try:
        runtime = attest_simulator_runtime(preflight)
    finally:
        preflight.close()
    return R3BRunBuilder(
        plan,
        _curriculum(bundle, plan.total_updates),
        bundle.store,
        runtime_attestation=runtime,
        lanes=lanes,
        collector_config=CollectorConfig(
            max_decisions=config.collector_max_decisions,
            target_simulated_ticks=plan.ticks_per_update,
            gamma_tick=1.0,
            lambda_tick=config.collector_lambda_tick,
        ),
        ppo_config=PPOConfig(
            learning_rate=plan.learning_rates[0],
            final_learning_rate_fraction=plan.final_learning_rate_fraction,
            epochs=config.ppo_epochs,
            lane_minibatch_size=config.ppo_lane_minibatch_size,
            clip_ratio=config.ppo_clip_ratio,
            value_clip=config.ppo_value_clip,
            value_coefficient=config.ppo_value_coefficient,
            entropy_coefficient=config.ppo_entropy_coefficient,
            max_gradient_norm=config.ppo_max_gradient_norm,
            target_kl=config.ppo_target_kl,
        ),
        model_factory=model_factory,
        environment_factory=environment_factory,
        sealed_test_ledger=sealed_test_ledger,
        reward_scale=1000.0,
        max_consecutive_skips=config.max_consecutive_skips,
    )


def _checkpoint(
    *,
    run_root: Path,
    workflow: R3BWorkflow,
    claim: JobClaim,
    built: object,
    identity: dict[str, object],
    plan: R3BExperimentPlan,
) -> str:
    session = built.session
    completed = session.trainer.schedule.completed_updates
    generation = f"update-{completed:04d}-{secrets.token_hex(8)}"
    checkpoint_root = run_root / "jobs" / claim.job_sha256 / "checkpoints"
    directory = session.save(checkpoint_root, generation, identity=identity)
    manifest_payload = (directory / "manifest.json").read_bytes()
    manifest = json.loads(manifest_payload)
    model = session.model
    kind_mask = torch.ones((1, 3), dtype=torch.bool)
    wait_mask = torch.ones((1, len(model.action_spec.wait_choices)), dtype=torch.bool)
    deployment = DeploymentPolicyIdentity.from_components(
        model,
        TeacherStateEncoder(),
        kind_mask,
        wait_mask,
    )
    checkpoint_manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    artifact = TrainingCheckpointArtifact(
        learner_seed=built.manifest.learner_seed,
        completed_updates=completed,
        simulated_ticks=session.collector.simulated_ticks,
        target_simulated_ticks=completed * plan.ticks_per_update,
        plan_sha256=plan.sha256,
        job_sha256=claim.job_sha256,
        trial_manifest_sha256=built.manifest.sha256,
        runner_spec_sha256=built.manifest.runner_spec_sha256,
        checkpoint_manifest_sha256=checkpoint_manifest_sha256,
        model_sha256=session.policy_sha256,
        deployment_policy_sha256=deployment.sha256,
    )
    envelope = ArtifactStore(run_root / "artifacts").publish(
        kind="irisu.r3b.training-checkpoint",
        version="r3b-training-checkpoint-package-v2",
        payload={
            "job_sha256": claim.job_sha256,
            "trial_manifest_sha256": built.manifest.sha256,
            "runner_spec_sha256": built.manifest.runner_spec_sha256,
            "completed_updates": completed,
            "simulated_ticks": session.collector.simulated_ticks,
            "model_sha256": session.policy_sha256,
            "deployment_policy_sha256": deployment.sha256,
            "checkpoint_artifact": artifact.manifest(),
            "generation": generation,
            "checkpoint_manifest_sha256": checkpoint_manifest_sha256,
            "checkpoint_files": manifest["files"],
        },
    )
    workflow.record_checkpoint(claim, completed, envelope.artifact_id)
    return envelope.artifact_id


@dataclass(frozen=True, slots=True)
class LocalTrainingResult:
    job_sha256: str
    completed_updates: int
    budget_updates: int
    simulated_ticks: int
    checkpoint_artifact_sha256: str
    training_complete: bool
    version: str = "r3b-local-training-result-v1"

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "job_sha256": self.job_sha256,
            "completed_updates": self.completed_updates,
            "budget_updates": self.budget_updates,
            "simulated_ticks": self.simulated_ticks,
            "checkpoint_artifact_sha256": self.checkpoint_artifact_sha256,
            "training_complete": self.training_complete,
            "acceptance_eligible": False,
            "transfer_eligible": False,
        }


def _run_local_training_updates(
    run_directory: str | Path,
    *,
    worker_path: str | Path,
    max_new_updates: int,
    owner: str,
    run_class: str,
    phase: str,
    authorization: ValidationRunAuthorization | SealedTestJobLease | None,
    sealed_test_ledger: SealedTestLedger | None,
) -> LocalTrainingResult:
    """Run one bounded exact-backend segment and publish a durable resume point."""

    if (
        isinstance(max_new_updates, bool)
        or not isinstance(max_new_updates, int)
        or max_new_updates <= 0
        or run_class not in {"smoke", "canonical"}
        or phase not in {"calibration", "validation", "test"}
    ):
        raise ValueError("training segment arguments are invalid")
    if run_class == "smoke" and (
        phase != "calibration"
        or authorization is not None
        or sealed_test_ledger is not None
    ):
        raise ValueError("smoke execution is limited to unsealed calibration")
    if phase == "calibration" and authorization is not None:
        raise ValueError("calibration execution cannot carry an authorization")
    if phase == "validation" and not isinstance(
        authorization, ValidationRunAuthorization
    ):
        raise ValueError("validation execution requires its typed authorization")
    if phase == "test" and (
        not isinstance(authorization, SealedTestJobLease)
        or not isinstance(sealed_test_ledger, SealedTestLedger)
    ):
        raise ValueError("test execution requires its sealed lease and ledger")
    root = Path(run_directory).resolve(strict=True)
    resolved = _read_resolved_run(root)
    workflow = R3BWorkflow(root / "workflow.sqlite3")
    workflow_manifest = workflow.verify()
    if resolved.get("workflow") != workflow_manifest:
        raise ValueError("resolved run manifest differs from durable workflow metadata")
    if (
        workflow_manifest["run_class"] != run_class
        or workflow_manifest["acceptance_eligible"] is not (run_class == "canonical")
        or workflow_manifest["transfer_eligible"] is not False
    ):
        raise ValueError("training runner workflow class or eligibility differs")
    project_root = Path(__file__).resolve().parents[2]
    if workflow_manifest["source_identity_sha256"] != _source_identity(project_root):
        raise ValueError("source tree changed after the run was initialized")
    if run_class == "canonical" and workflow_manifest.get(
        "source_revision"
    ) != _clean_source_revision(project_root):
        raise ValueError("canonical source revision changed after initialization")
    plan_value = resolved["plan"]
    config_value = resolved["operational_config"]
    if not isinstance(plan_value, dict) or not isinstance(config_value, dict):
        raise ValueError("resolved plan or operational config is malformed")
    plan = R3BExperimentPlan.from_mapping(plan_value)
    config = R3BOperationalConfig.from_manifest(config_value)
    if (
        plan.sha256 != workflow_manifest["plan_sha256"]
        or config.sha256 != workflow_manifest["operational_config_sha256"]
    ):
        raise ValueError("resolved plan or config differs from the durable workflow")
    snapshot_path = resolved["snapshot_bundle_path"]
    if not isinstance(snapshot_path, str) or not Path(snapshot_path).is_absolute():
        raise ValueError("resolved snapshot bundle path is malformed")
    worker = Path(worker_path)
    if not worker.is_absolute() or worker.is_symlink() or not worker.is_file():
        raise ValueError("exact worker path must be an absolute regular file")

    secret_root = root / "secrets"
    active: tuple[Path, JobClaim] | None = None
    if secret_root.exists():
        for path in sorted(secret_root.glob("*.claim.json")):
            candidate = _load_claim(path)
            record = workflow.job_record(candidate.job_sha256)
            if record["status"] in {"claimed", "running", "trained"}:
                if active is not None:
                    raise RuntimeError("multiple active local claim secrets exist")
                active = (path, candidate)
    if active is None:
        intent_path = secret_root / f"{phase}.intent.json"
        if intent_path.exists():
            intent_phase, token, intent_owner = _load_claim_intent(intent_path)
            if intent_phase != phase or intent_owner != owner:
                raise RuntimeError("claim intent belongs to another worker")
            claim = workflow.resume_unstarted_claim(phase, owner=owner, token=token)
        else:
            token = secrets.token_hex(32)
            _write_claim_intent(
                intent_path,
                phase=phase,
                token=token,
                owner=owner,
            )
            claim = None
        if claim is None:
            claim = workflow.claim_next(phase, owner=owner, token=token)
        if claim is None:
            intent_path.unlink()
            raise RuntimeError(f"no pending {phase} training job exists")
        secret_path = secret_root / f"{claim.job_sha256}.claim.json"
        _write_claim(secret_path, claim)
        intent_path.unlink()
        parent_fd = os.open(secret_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    else:
        secret_path, claim = active
        if claim.owner != owner:
            raise RuntimeError("active training claim belongs to another owner")

    record = workflow.job_record(claim.job_sha256)
    if record["status"] == "trained":
        raise RuntimeError("training is complete; evaluation output is still pending")
    if phase == "test" and record["status"] == "running":
        raise RuntimeError(
            "sealed test training cannot resume across an execution boundary"
        )
    try:
        job = TrialJob.from_manifest(record["manifest"])
    except (TypeError, ValueError) as exc:
        raise ValueError("claimed job manifest is invalid") from exc
    if job.phase != phase or job.sha256 != claim.job_sha256:
        raise ValueError("claimed job is foreign to the requested phase")
    if phase == "test" and max_new_updates < job.budget_updates:
        raise ValueError(
            "sealed test training must reach its full budget in one process"
        )
    simulator = IrisuEnv(physics_backend="exact", worker_path=worker)
    try:
        bundle = load_snapshot_bundle(snapshot_path, simulator)
    finally:
        simulator.close()
    if bundle.sha256 != workflow_manifest["snapshot_bundle_sha256"]:
        raise ValueError("loaded snapshot bundle differs from the initialized run")
    builder = _builder(
        plan=plan,
        config=config,
        bundle=bundle,
        worker_path=worker,
        sealed_test_ledger=sealed_test_ledger,
    )
    built = None
    began = record["status"] == "running"
    try:
        built = (
            builder.build_under_running_sealed_lease(job, authorization=authorization)
            if isinstance(authorization, SealedTestJobLease)
            and record["status"] == "running"
            else builder.build(job, authorization=authorization)
        )
        if not began:
            workflow.begin(claim)
            began = True
        identity = {
            "trial_manifest_sha256": built.manifest.sha256,
            "job_sha256": job.sha256,
            "runner_spec_sha256": built.manifest.runner_spec_sha256,
            "source_identity_sha256": workflow_manifest["source_identity_sha256"],
            "snapshot_bundle_sha256": workflow_manifest["snapshot_bundle_sha256"],
        }
        latest = workflow.job_record(claim.job_sha256)["latest_checkpoint"]
        if latest is None:
            built.session.initialize()
            checkpoint_sha = _checkpoint(
                run_root=root,
                workflow=workflow,
                claim=claim,
                built=built,
                identity=identity,
                plan=plan,
            )
        else:
            update = int(latest["completed_updates"])
            envelope = ArtifactStore(root / "artifacts").load(
                str(latest["artifact_sha256"]),
                expected_kind="irisu.r3b.training-checkpoint",
                expected_version="r3b-training-checkpoint-package-v2",
            )
            payload = envelope.payload
            if (
                not isinstance(payload, dict)
                or set(payload)
                != {
                    "job_sha256",
                    "trial_manifest_sha256",
                    "runner_spec_sha256",
                    "completed_updates",
                    "simulated_ticks",
                    "model_sha256",
                    "deployment_policy_sha256",
                    "checkpoint_artifact",
                    "generation",
                    "checkpoint_manifest_sha256",
                    "checkpoint_files",
                }
                or payload.get("completed_updates") != update
                or payload.get("job_sha256") != claim.job_sha256
                or payload.get("trial_manifest_sha256") != built.manifest.sha256
                or payload.get("runner_spec_sha256")
                != built.manifest.runner_spec_sha256
            ):
                raise ValueError("latest checkpoint receipt differs from the job")
            checkpoint_artifact = TrainingCheckpointArtifact.from_manifest(
                payload["checkpoint_artifact"]
            )
            if (
                checkpoint_artifact.learner_seed != job.learner_seed
                or checkpoint_artifact.completed_updates != update
                or checkpoint_artifact.plan_sha256 != plan.sha256
                or checkpoint_artifact.job_sha256 != job.sha256
                or checkpoint_artifact.trial_manifest_sha256 != built.manifest.sha256
                or checkpoint_artifact.runner_spec_sha256
                != built.manifest.runner_spec_sha256
                or checkpoint_artifact.simulated_ticks != payload["simulated_ticks"]
                or checkpoint_artifact.model_sha256 != payload["model_sha256"]
                or checkpoint_artifact.deployment_policy_sha256
                != payload["deployment_policy_sha256"]
            ):
                raise ValueError("typed checkpoint receipt differs from the job")
            generation = payload.get("generation")
            checkpoint_manifest_sha256 = payload.get("checkpoint_manifest_sha256")
            if not isinstance(generation, str) or not isinstance(
                checkpoint_manifest_sha256, str
            ):
                raise ValueError("latest checkpoint receipt lacks restore identity")
            manifest_path = (
                root
                / "jobs"
                / claim.job_sha256
                / "checkpoints"
                / generation
                / "manifest.json"
            )
            if (
                manifest_path.is_symlink()
                or not manifest_path.is_file()
                or hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                != checkpoint_manifest_sha256
            ):
                raise ValueError("checkpoint manifest differs from its receipt")
            built.session.restore(
                root / "jobs" / claim.job_sha256 / "checkpoints",
                generation=generation,
                identity=identity,
            )
            if (
                built.session.trainer.schedule.completed_updates != update
                or built.session.collector.simulated_ticks
                != payload.get("simulated_ticks")
                or built.session.policy_sha256 != payload.get("model_sha256")
            ):
                raise ValueError("restored checkpoint state differs from its receipt")
            checkpoint_sha = envelope.artifact_id

        start = built.session.trainer.schedule.completed_updates
        target = min(job.budget_updates, start + max_new_updates)
        if phase == "test" and target != job.budget_updates:
            raise ValueError(
                "sealed test training must reach its full budget in one process"
            )
        if (
            run_class == "canonical"
            and target != job.budget_updates
            and target % plan.checkpoint_interval_updates
        ):
            raise ValueError(
                "canonical segments must stop on the frozen checkpoint grid"
            )
        last_published_update = int(
            workflow.job_record(claim.job_sha256)["latest_checkpoint"][
                "completed_updates"
            ]
        )
        while built.session.trainer.schedule.completed_updates < target:
            built.session.run_update()
            completed_updates = built.session.trainer.schedule.completed_updates
            if (
                completed_updates % plan.checkpoint_interval_updates == 0
                or completed_updates == target
            ):
                checkpoint_sha = _checkpoint(
                    run_root=root,
                    workflow=workflow,
                    claim=claim,
                    built=built,
                    identity=identity,
                    plan=plan,
                )
                last_published_update = completed_updates
        if (
            built.session.trainer.schedule.completed_updates != start
            and last_published_update
            != built.session.trainer.schedule.completed_updates
        ):
            raise RuntimeError("the final training update was not checkpointed")
        complete = (
            built.session.trainer.schedule.completed_updates == job.budget_updates
        )
        if complete:
            workflow.mark_trained(claim)
        return LocalTrainingResult(
            claim.job_sha256,
            built.session.trainer.schedule.completed_updates,
            job.budget_updates,
            built.session.collector.simulated_ticks,
            checkpoint_sha,
            complete,
        )
    except Exception as error:
        reason = f"{type(error).__name__}: {error}"
        if (
            isinstance(authorization, SealedTestJobLease)
            and sealed_test_ledger is not None
        ):
            try:
                authorization.assert_running()
            except RuntimeError:
                pass
            else:
                sealed_test_ledger.fail_job(authorization, reason)
                workflow.reconcile_sealed_failure(
                    ledger=sealed_test_ledger,
                    sealed_run=authorization.sealed_run,
                    job=authorization.job,
                    failure_reason=reason,
                )
        if began:
            current = workflow.job_record(claim.job_sha256)["status"]
            if current in {"claimed", "running"}:
                workflow.fail(claim, reason)
        raise
    finally:
        if built is not None:
            built.close()


def run_local_smoke_updates(
    run_directory: str | Path,
    *,
    worker_path: str | Path,
    max_new_updates: int,
    owner: str,
) -> LocalTrainingResult:
    """Run a bounded diagnostic calibration segment."""

    return _run_local_training_updates(
        run_directory,
        worker_path=worker_path,
        max_new_updates=max_new_updates,
        owner=owner,
        run_class="smoke",
        phase="calibration",
        authorization=None,
        sealed_test_ledger=None,
    )


def run_local_canonical_updates(
    run_directory: str | Path,
    *,
    worker_path: str | Path,
    max_new_updates: int,
    owner: str,
    phase: str = "calibration",
    authorization: ValidationRunAuthorization | SealedTestJobLease | None = None,
    sealed_test_ledger: SealedTestLedger | None = None,
) -> LocalTrainingResult:
    """Run a bounded, restartable segment of an installed canonical job."""

    return _run_local_training_updates(
        run_directory,
        worker_path=worker_path,
        max_new_updates=max_new_updates,
        owner=owner,
        run_class="canonical",
        phase=phase,
        authorization=authorization,
        sealed_test_ledger=sealed_test_ledger,
    )
