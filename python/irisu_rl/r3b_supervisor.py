"""Executable, restartable supervision for one trained canonical R3 job."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from irisu_env import IrisuEnv, PaddedVectorEnv

from .encoding import TeacherStateEncoder
from .r3b_artifacts import ArtifactStore
from .r3b_canonical_runner import (
    CanonicalRunInputs,
    PairedEvaluationSuites,
    assemble_and_publish_outcome,
    audit_penultimate_checkpoint,
    complete_nonsealed_workflow_job,
    complete_sealed_workflow_job,
    evaluate_recurrent_policy_sharded,
)
from .r3b_evaluation import DeploymentPolicyIdentity
from .r3b_experiments import (
    CheckpointEvaluation,
    SealedTestJobLease,
    SealedTestLedger,
    TrainingCheckpointArtifact,
    TrialJob,
    ValidationRunAuthorization,
)
from .r3b_local_runner import (
    _builder,
    _load_claim,
    _read_resolved_run,
)
from .r3b_operational import JobClaim, R3BWorkflow


_CHECKPOINT_KIND = "irisu.r3b.training-checkpoint"
_CHECKPOINT_VERSION = "r3b-training-checkpoint-package-v2"


def _deployment(
    model: Any,
) -> tuple[TeacherStateEncoder, torch.Tensor, torch.Tensor, DeploymentPolicyIdentity]:
    encoder = TeacherStateEncoder()
    kind_mask = torch.ones((1, 3), dtype=torch.bool)
    wait_mask = torch.ones((1, len(model.action_spec.wait_choices)), dtype=torch.bool)
    identity = DeploymentPolicyIdentity.from_components(
        model, encoder, kind_mask, wait_mask
    )
    return encoder, kind_mask, wait_mask, identity


def _active_claim(
    root: Path, workflow: R3BWorkflow, phase: str
) -> tuple[Path, JobClaim]:
    secrets = root / "secrets"
    active: list[tuple[Path, JobClaim]] = []
    if secrets.is_dir():
        for path in sorted(secrets.glob("*.claim.json")):
            claim = _load_claim(path)
            record = workflow.job_record(claim.job_sha256)
            if record["status"] in {"claimed", "running", "trained"}:
                active.append((path, claim))
    if len(active) != 1:
        raise RuntimeError("canonical evaluation requires exactly one active claim")
    path, claim = active[0]
    if claim.phase != phase:
        raise ValueError("active claim belongs to a different phase")
    return path, claim


def _checkpoint_package(
    *,
    root: Path,
    store: ArtifactStore,
    artifact_sha256: str,
    built: Any,
    job: TrialJob,
    target_update: int,
) -> tuple[TrainingCheckpointArtifact, str, dict[str, object]]:
    envelope = store.load(
        artifact_sha256,
        expected_kind=_CHECKPOINT_KIND,
        expected_version=_CHECKPOINT_VERSION,
    )
    payload = envelope.payload
    expected = {
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
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("training checkpoint package schema differs")
    checkpoint = TrainingCheckpointArtifact.from_manifest(
        payload["checkpoint_artifact"]
    )
    generation = payload["generation"]
    if (
        not isinstance(generation, str)
        or not generation
        or payload["job_sha256"] != job.sha256
        or payload["trial_manifest_sha256"] != built.manifest.sha256
        or payload["runner_spec_sha256"] != built.manifest.runner_spec_sha256
        or payload["completed_updates"] != target_update
        or checkpoint.completed_updates != target_update
        or checkpoint.learner_seed != job.learner_seed
        or checkpoint.job_sha256 != job.sha256
        or checkpoint.plan_sha256 != job.plan_sha256
        or checkpoint.trial_manifest_sha256 != built.manifest.sha256
        or checkpoint.runner_spec_sha256 != built.manifest.runner_spec_sha256
        or checkpoint.checkpoint_manifest_sha256
        != payload["checkpoint_manifest_sha256"]
        or checkpoint.simulated_ticks != payload["simulated_ticks"]
        or checkpoint.model_sha256 != payload["model_sha256"]
        or checkpoint.deployment_policy_sha256 != payload["deployment_policy_sha256"]
    ):
        raise ValueError("training checkpoint package is foreign to the job")
    directory = root / "jobs" / job.sha256 / "checkpoints" / generation
    manifest = directory / "manifest.json"
    if (
        not directory.is_dir()
        or manifest.is_symlink()
        or not manifest.is_file()
        or hashlib.sha256(manifest.read_bytes()).hexdigest()
        != checkpoint.checkpoint_manifest_sha256
    ):
        raise ValueError("training checkpoint files are missing or unsafe")
    try:
        manifest_value = json.loads(manifest.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("training checkpoint manifest is malformed") from exc
    if not isinstance(manifest_value, dict) or payload[
        "checkpoint_files"
    ] != manifest_value.get("files"):
        raise ValueError("training checkpoint file inventory differs")
    return checkpoint, generation, payload


def _restore(
    *,
    root: Path,
    built: Any,
    job: TrialJob,
    generation: str,
    identity: dict[str, object],
    checkpoint: TrainingCheckpointArtifact,
) -> None:
    built.session.restore(
        root / "jobs" / job.sha256 / "checkpoints",
        generation=generation,
        identity=identity,
    )
    encoder, kind_mask, wait_mask, deployment = _deployment(built.session.model)
    del encoder, kind_mask, wait_mask
    if (
        built.session.trainer.schedule.completed_updates != checkpoint.completed_updates
        or built.session.collector.simulated_ticks != checkpoint.simulated_ticks
        or built.session.policy_sha256 != checkpoint.model_sha256
        or deployment.sha256 != checkpoint.deployment_policy_sha256
    ):
        raise ValueError("restored session differs from its typed checkpoint")


def _build_trial_for_evaluation(
    builder: Any,
    job: TrialJob,
    authorization: ValidationRunAuthorization | SealedTestJobLease | None,
) -> Any:
    return (
        builder.build_under_running_sealed_lease(job, authorization=authorization)
        if isinstance(authorization, SealedTestJobLease)
        else builder.build(job, authorization=authorization)
    )


def _fresh_restored_checkpoint(
    *,
    builder: Any,
    authorization: ValidationRunAuthorization | SealedTestJobLease | None,
    root: Path,
    store: ArtifactStore,
    artifact_sha256: str,
    job: TrialJob,
    target_update: int,
    identity: dict[str, object],
) -> tuple[Any, TrainingCheckpointArtifact, str]:
    """Restore one checkpoint into a new caller-owned trial session."""

    built = _build_trial_for_evaluation(builder, job, authorization)
    try:
        checkpoint, generation, _ = _checkpoint_package(
            root=root,
            store=store,
            artifact_sha256=artifact_sha256,
            built=built,
            job=job,
            target_update=target_update,
        )
        _restore(
            root=root,
            built=built,
            job=job,
            generation=generation,
            identity=identity,
            checkpoint=checkpoint,
        )
    except BaseException:
        built.close()
        raise
    return built, checkpoint, generation


@dataclass(frozen=True, slots=True)
class CanonicalEvaluationResult:
    job_sha256: str
    phase: str
    completed_updates: int
    output_artifact_sha256: str
    outcome_sha256: str
    outcome_reference_sha256: str | None = None
    version: str = "r3b-canonical-evaluation-result-v1"

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "job_sha256": self.job_sha256,
            "phase": self.phase,
            "completed_updates": self.completed_updates,
            "output_artifact_sha256": self.output_artifact_sha256,
            "outcome_sha256": self.outcome_sha256,
            "outcome_reference_sha256": self.outcome_reference_sha256,
            "acceptance_eligible": True,
            "transfer_eligible": False,
        }


def evaluate_trained_canonical_job(
    run_directory: str | Path,
    *,
    exact_worker_path: str | Path,
    portable_library_path: str | Path,
    phase: str,
    authorization: ValidationRunAuthorization | SealedTestJobLease | None = None,
    sealed_test_ledger: SealedTestLedger | None = None,
) -> CanonicalEvaluationResult:
    """Evaluate, audit, publish, and complete one fully trained canonical job."""

    if phase not in {"calibration", "validation", "test"}:
        raise ValueError("canonical evaluation phase is invalid")
    if phase == "calibration" and authorization is not None:
        raise ValueError("calibration evaluation cannot carry an authorization")
    if phase == "validation" and not isinstance(
        authorization, ValidationRunAuthorization
    ):
        raise ValueError("validation evaluation requires its authorization")
    if phase == "test" and (
        not isinstance(authorization, SealedTestJobLease)
        or not isinstance(sealed_test_ledger, SealedTestLedger)
    ):
        raise ValueError("test evaluation requires its sealed lease and ledger")

    root = Path(run_directory).resolve(strict=True)
    supplied_worker = Path(exact_worker_path)
    supplied_library = Path(portable_library_path)
    if (
        not supplied_worker.is_absolute()
        or supplied_worker.is_symlink()
        or not supplied_worker.is_file()
    ):
        raise ValueError("exact worker must be a regular file")
    if (
        not supplied_library.is_absolute()
        or supplied_library.is_symlink()
        or not supplied_library.is_file()
    ):
        raise ValueError("portable library must be a regular file")
    worker = supplied_worker.resolve(strict=True)
    library = supplied_library.resolve(strict=True)
    _read_resolved_run(root)
    workflow = R3BWorkflow(root / "workflow.sqlite3")
    _, claim = _active_claim(root, workflow, phase)
    record = workflow.job_record(claim.job_sha256)
    if record["status"] != "trained":
        raise RuntimeError("canonical evaluation requires completed training")
    job = TrialJob.from_manifest(record["manifest"])
    if job.phase != phase or job.sha256 != claim.job_sha256:
        raise ValueError("claimed canonical job identity differs")

    with ExitStack() as stack:
        exact_loader = stack.enter_context(
            IrisuEnv(physics_backend="exact", worker_path=worker)
        )
        portable_loader = stack.enter_context(
            IrisuEnv(physics_backend="portable", library_path=library)
        )
        inputs = CanonicalRunInputs.load(
            root,
            exact_simulator=exact_loader,
            portable_simulator=portable_loader,
        )
    plan = inputs.plan
    config = inputs.config
    store = ArtifactStore(root / "artifacts")
    builder = _builder(
        plan=plan,
        config=config,
        bundle=inputs.exact_bundle,
        worker_path=worker,
        sealed_test_ledger=sealed_test_ledger,
    )
    built = None
    initial_built = None
    resume_built = None
    try:
        initial_built = _build_trial_for_evaluation(builder, job, authorization)
        identity = {
            "trial_manifest_sha256": initial_built.manifest.sha256,
            "job_sha256": job.sha256,
            "runner_spec_sha256": initial_built.manifest.runner_spec_sha256,
            "source_identity_sha256": inputs.workflow.verify()[
                "source_identity_sha256"
            ],
            "snapshot_bundle_sha256": inputs.exact_bundle.sha256,
        }
        expected_updates = tuple(
            range(0, job.budget_updates + 1, plan.checkpoint_interval_updates)
        )
        indexed = {
            int(value["completed_updates"]): str(value["artifact_sha256"])
            for value in workflow.job_checkpoints(job.sha256)
            if int(value["completed_updates"]) in expected_updates
        }
        if set(indexed) != set(expected_updates):
            raise RuntimeError("canonical job lacks its complete checkpoint grid")
        packages: dict[int, tuple[TrainingCheckpointArtifact, str]] = {}
        checkpoint_evaluations: list[CheckpointEvaluation] = []
        curve_suites = PairedEvaluationSuites.build(
            inputs,
            phase=phase,
            learner_seed=job.learner_seed,
            assignment_sha256=initial_built.manifest.assignment_sha256,
            purpose="curve",
        )
        final_suites = PairedEvaluationSuites.build(
            inputs,
            phase=phase,
            learner_seed=job.learner_seed,
            assignment_sha256=initial_built.manifest.assignment_sha256,
        )
        initial_built.close()
        initial_built = None
        with PaddedVectorEnv(
            config.lanes,
            workers=config.workers,
            physics_backend="exact",
            worker_path=worker,
        ) as exact_vector:
            for update in expected_updates:
                checkpoint_built, checkpoint, generation = (
                    _fresh_restored_checkpoint(
                        builder=builder,
                        authorization=authorization,
                        root=root,
                        store=store,
                        artifact_sha256=indexed[update],
                        job=job,
                        target_update=update,
                        identity=identity,
                    )
                )
                retain = False
                try:
                    packages[update] = (checkpoint, generation)
                    encoder, kind_mask, wait_mask, deployment = _deployment(
                        checkpoint_built.session.model
                    )
                    report = evaluate_recurrent_policy_sharded(
                        inputs=inputs,
                        simulator=exact_vector,
                        store=inputs.exact_bundle.store,
                        suite=curve_suites.exact,
                        model=checkpoint_built.session.model,
                        encoder=encoder,
                        kind_mask=kind_mask,
                        wait_mask=wait_mask,
                        policy_sha256=deployment.sha256,
                        artifact_store=store,
                    )
                    checkpoint_evaluations.append(
                        CheckpointEvaluation(checkpoint, report)
                    )
                    if update == job.budget_updates:
                        built = checkpoint_built
                        retain = True
                finally:
                    if not retain:
                        checkpoint_built.close()
            if built is None:
                raise RuntimeError("canonical evaluation lacks its final session")
            encoder, kind_mask, wait_mask, deployment = _deployment(built.session.model)
            exact_final_report = evaluate_recurrent_policy_sharded(
                inputs=inputs,
                simulator=exact_vector,
                store=inputs.exact_bundle.store,
                suite=final_suites.exact,
                model=built.session.model,
                encoder=encoder,
                kind_mask=kind_mask,
                wait_mask=wait_mask,
                policy_sha256=deployment.sha256,
                artifact_store=store,
            )

        with PaddedVectorEnv(
            config.lanes,
            workers=config.workers,
            physics_backend="portable",
            library_path=library,
        ) as portable_vector:
            portable_final_report = evaluate_recurrent_policy_sharded(
                inputs=inputs,
                simulator=portable_vector,
                store=inputs.portable_bundle.store,
                suite=final_suites.portable,
                model=built.session.model,
                encoder=encoder,
                kind_mask=kind_mask,
                wait_mask=wait_mask,
                policy_sha256=deployment.sha256,
                artifact_store=store,
            )

        resume_update = job.budget_updates - plan.checkpoint_interval_updates
        resume_checkpoint, resume_generation = packages[resume_update]
        resume_built, restored_resume_checkpoint, restored_resume_generation = (
            _fresh_restored_checkpoint(
                builder=builder,
                authorization=authorization,
                root=root,
                store=store,
                artifact_sha256=indexed[resume_update],
                job=job,
                target_update=resume_update,
                identity=identity,
            )
        )
        if (
            restored_resume_checkpoint != resume_checkpoint
            or restored_resume_generation != resume_generation
        ):
            raise RuntimeError("resume checkpoint restoration changed identity")

        def restored_factory():
            if isinstance(authorization, SealedTestJobLease):
                return builder.build_resume_audit_session(
                    job, authorization=authorization
                )
            return builder.build(job, authorization=authorization).session

        resume_artifact = audit_penultimate_checkpoint(
            job=job,
            checkpoint=resume_checkpoint,
            checkpoint_directory=root / "jobs" / job.sha256 / "checkpoints",
            generation=resume_generation,
            checkpoint_identity=identity,
            source=resume_built.session,
            restored_factory=restored_factory,
            plan=plan,
            sealed_job_lease=(
                authorization if isinstance(authorization, SealedTestJobLease) else None
            ),
        )
        resume_built.close()
        resume_built = None
        outcome, published = assemble_and_publish_outcome(
            inputs=inputs,
            store=store,
            built=built,
            job=job,
            suites=final_suites,
            curve_suites=curve_suites,
            checkpoint_evaluations=tuple(checkpoint_evaluations),
            exact_final_report=exact_final_report,
            portable_final_report=portable_final_report,
            deployment_identity=deployment,
            exact_resume_artifact=resume_artifact,
            checkpoint_interval_updates=plan.checkpoint_interval_updates,
            plan=plan,
            workflow_claim=claim,
            sealed_job_lease=(
                authorization if isinstance(authorization, SealedTestJobLease) else None
            ),
        )
        if isinstance(authorization, SealedTestJobLease):
            assert sealed_test_ledger is not None
            outcome_reference_sha256 = complete_sealed_workflow_job(
                inputs=inputs,
                store=store,
                workflow=workflow,
                claim=claim,
                ledger=sealed_test_ledger,
                lease=authorization,
                outcome=outcome,
                published=published,
            )
        else:
            outcome_reference_sha256 = None
            complete_nonsealed_workflow_job(
                inputs=inputs,
                store=store,
                workflow=workflow,
                claim=claim,
                job=job,
                published=published,
            )
        return CanonicalEvaluationResult(
            job.sha256,
            phase,
            job.budget_updates,
            published.output_artifact_sha256,
            outcome.sha256,
            outcome_reference_sha256,
        )
    except Exception as error:
        if (
            isinstance(authorization, SealedTestJobLease)
            and sealed_test_ledger is not None
        ):
            reason = f"{type(error).__name__}: {error}"
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
        raise
    finally:
        if initial_built is not None:
            initial_built.close()
        if resume_built is not None:
            resume_built.close()
        if built is not None:
            built.close()
