"""Executable, restartable supervision for one trained canonical R3 job."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import signal
import sys
import time
from ctypes import CDLL, get_errno
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from irisu_env import IrisuEnv

from .collector import model_state_sha256
from .encoding import TeacherStateEncoder
from .r3b_artifacts import ArtifactLookupIndex, ArtifactStore
from .r3b_canonical_runner import (
    CanonicalRunInputs,
    PairedEvaluationSuites,
    assemble_and_publish_outcome,
    audit_penultimate_checkpoint,
    canonical_evaluation_identities,
    complete_nonsealed_workflow_job,
    complete_sealed_workflow_job,
)
from .r3b_evaluation import (
    DeploymentPolicyIdentity,
    deployment_policy_identity_for_threads,
)
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
from .r3b_lock import evaluator_lease_path, hold_evaluator_lease
from .r3b_operational import JobClaim, R3BWorkflow
from .r3b_parallel_evaluation import (
    CanonicalEvaluationTask,
    CanonicalEvaluationTaskResult,
    evaluate_canonical_task,
    serialize_evaluation_model,
)

_CHECKPOINT_KIND = "irisu.r3b.training-checkpoint"
_CHECKPOINT_VERSION = "r3b-training-checkpoint-package-v2"
_EVALUATION_SHUTDOWN_SECONDS = 5.0
_PR_SET_PDEATHSIG = 1
_EVALUATION_LEASE_DESCRIPTOR: int | None = None


def _terminate_evaluation_process_group(_signum: int, _frame: object | None) -> None:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.killpg(os.getpgrp(), signal.SIGTERM)


def _evaluation_worker_initializer(parent_pid: int, lease_path: str) -> None:
    """Make evaluator descendants die as one group when their owner disappears."""

    global _EVALUATION_LEASE_DESCRIPTOR
    if sys.platform != "linux" or parent_pid <= 1:
        raise RuntimeError("canonical evaluator isolation requires Linux")
    os.setsid()
    signal.signal(signal.SIGTERM, _terminate_evaluation_process_group)
    libc = CDLL(None, use_errno=True)
    if libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0) != 0:
        error = get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != parent_pid:
        _terminate_evaluation_process_group(signal.SIGTERM, None)
    _EVALUATION_LEASE_DESCRIPTOR = hold_evaluator_lease(lease_path)


def _evaluation_processes(executor: ProcessPoolExecutor) -> tuple[Any, ...]:
    processes = getattr(executor, "_processes", None)
    return tuple(processes.values()) if isinstance(processes, dict) else ()


def _signal_evaluation_process(process: Any, sig: signal.Signals) -> None:
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return
    if os.name == "posix":
        try:
            if os.getpgid(pid) == pid:
                os.killpg(pid, sig)
        except ProcessLookupError:
            pass
    try:
        if sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except (ProcessLookupError, ValueError):
        pass


def _stop_evaluation_executor(
    executor: ProcessPoolExecutor,
    *,
    timeout_seconds: float = _EVALUATION_SHUTDOWN_SECONDS,
) -> None:
    """Cancel queued work and bound teardown of evaluators and descendants."""

    processes = _evaluation_processes(executor)
    if not processes:
        executor.shutdown(wait=False, cancel_futures=True)
        return
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        _signal_evaluation_process(process, signal.SIGTERM)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        active = tuple(process for process in processes if process.is_alive())
        if not active:
            break
        for process in active:
            process.join(min(0.05, max(0.0, deadline - time.monotonic())))
    survivors = tuple(process for process in processes if process.is_alive())
    for process in survivors:
        _signal_evaluation_process(process, signal.SIGKILL)
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        active = tuple(process for process in survivors if process.is_alive())
        if not active:
            break
        for process in active:
            process.join(min(0.05, max(0.0, deadline - time.monotonic())))
    remaining = tuple(process for process in survivors if process.is_alive())
    if remaining:
        details = tuple(
            (getattr(process, "pid", None), getattr(process, "exitcode", None))
            for process in remaining
        )
        raise RuntimeError(
            f"isolated canonical evaluator cleanup did not finish: {details}"
        )


def _deployment(
    model: Any, *, torch_threads: int | None = None
) -> tuple[TeacherStateEncoder, torch.Tensor, torch.Tensor, DeploymentPolicyIdentity]:
    encoder = TeacherStateEncoder()
    kind_mask = torch.ones((1, 3), dtype=torch.bool)
    wait_mask = torch.ones((1, len(model.action_spec.wait_choices)), dtype=torch.bool)
    identity = (
        DeploymentPolicyIdentity.from_components(model, encoder, kind_mask, wait_mask)
        if torch_threads is None
        else deployment_policy_identity_for_threads(
            model,
            encoder,
            kind_mask,
            wait_mask,
            torch_threads=torch_threads,
        )
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
    evaluation_torch_threads: int | None = None,
) -> None:
    built.session.restore(
        root / "jobs" / job.sha256 / "checkpoints",
        generation=generation,
        identity=identity,
    )
    encoder, kind_mask, wait_mask, deployment = _deployment(
        built.session.model,
        torch_threads=evaluation_torch_threads,
    )
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
    evaluation_torch_threads: int | None = None,
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
            evaluation_torch_threads=evaluation_torch_threads,
        )
    except BaseException:
        built.close()
        raise
    return built, checkpoint, generation


def _evaluation_task(
    *,
    inputs: CanonicalRunInputs,
    worker: Path,
    library: Path,
    job: TrialJob,
    assignment_sha256: str,
    purpose: str,
    backend: str,
    suite: Any,
    checkpoint: TrainingCheckpointArtifact,
    model: Any,
    model_transport: bytes | None = None,
) -> CanonicalEvaluationTask:
    """Bind one verified restored model to one immutable evaluation suite."""

    _, _, _, deployment = _deployment(
        model,
        torch_threads=inputs.config.evaluation_torch_threads,
    )
    if (
        model_state_sha256(model) != checkpoint.model_sha256
        or deployment.sha256 != checkpoint.deployment_policy_sha256
    ):
        raise ValueError("evaluation model differs from its typed checkpoint")
    transport = (
        serialize_evaluation_model(model)
        if model_transport is None
        else model_transport
    )
    evaluator_sha256, worker_identity_sha256 = canonical_evaluation_identities(
        inputs, suite
    )
    return CanonicalEvaluationTask(
        run_directory=str(inputs.root),
        exact_worker_path=str(worker),
        portable_library_path=str(library),
        phase=job.phase,
        job_sha256=job.sha256,
        learner_seed=job.learner_seed,
        completed_updates=checkpoint.completed_updates,
        authorization_sha256=job.authorization_sha256,
        assignment_sha256=assignment_sha256,
        workflow_manifest_sha256=inputs.workflow_manifest_sha256,
        operational_config_sha256=inputs.config.sha256,
        purpose=purpose,
        backend=backend,
        suite_sha256=suite.sha256,
        model_sha256=checkpoint.model_sha256,
        deployment_policy_sha256=checkpoint.deployment_policy_sha256,
        evaluator_sha256=evaluator_sha256,
        worker_identity_sha256=worker_identity_sha256,
        evaluation_shards=inputs.config.evaluation_shards,
        model_transport_sha256=hashlib.sha256(transport).hexdigest(),
        model_transport=transport,
    )


def _collect_evaluation_results(
    futures: dict[Future[CanonicalEvaluationTaskResult], CanonicalEvaluationTask],
) -> dict[str, CanonicalEvaluationTaskResult]:
    """Collect every task exactly once, failing before workflow completion."""

    results: dict[str, CanonicalEvaluationTaskResult] = {}
    for future in as_completed(futures):
        task = futures[future]
        try:
            result = future.result()
        except BaseException as exc:
            raise RuntimeError(
                "isolated canonical evaluation task "
                f"{task.sha256} ({task.purpose}/{task.backend}, "
                f"update {task.completed_updates}) failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if isinstance(result, CanonicalEvaluationTaskResult):
            result.__post_init__()
        if (
            not isinstance(result, CanonicalEvaluationTaskResult)
            or result.task_sha256 != task.sha256
            or result.task_sha256 in results
        ):
            raise ValueError("isolated canonical evaluation result identity differs")
        results[result.task_sha256] = result
    expected = {task.sha256 for task in futures.values()}
    if set(results) != expected:
        raise RuntimeError("isolated canonical evaluation results are incomplete")
    return results


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
        curve_tasks: dict[int, CanonicalEvaluationTask] = {}
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
        final_checkpoint: TrainingCheckpointArtifact | None = None
        final_transport: bytes | None = None
        for update in expected_updates:
            checkpoint_built, checkpoint, generation = _fresh_restored_checkpoint(
                builder=builder,
                authorization=authorization,
                root=root,
                store=store,
                artifact_sha256=indexed[update],
                job=job,
                target_update=update,
                identity=identity,
                evaluation_torch_threads=config.evaluation_torch_threads,
            )
            retain = False
            try:
                packages[update] = (checkpoint, generation)
                transport = serialize_evaluation_model(checkpoint_built.session.model)
                curve_tasks[update] = _evaluation_task(
                    inputs=inputs,
                    worker=worker,
                    library=library,
                    job=job,
                    assignment_sha256=curve_suites.exact.assignment_sha256,
                    purpose="curve",
                    backend="exact",
                    suite=curve_suites.exact,
                    checkpoint=checkpoint,
                    model=checkpoint_built.session.model,
                    model_transport=transport,
                )
                if update == job.budget_updates:
                    built = checkpoint_built
                    final_checkpoint = checkpoint
                    final_transport = transport
                    retain = True
            finally:
                if not retain:
                    checkpoint_built.close()
        if built is None or final_checkpoint is None or final_transport is None:
            raise RuntimeError("canonical evaluation lacks its final session")
        _, _, _, deployment = _deployment(
            built.session.model,
            torch_threads=config.evaluation_torch_threads,
        )
        exact_final_task = _evaluation_task(
            inputs=inputs,
            worker=worker,
            library=library,
            job=job,
            assignment_sha256=final_suites.exact.assignment_sha256,
            purpose="final",
            backend="exact",
            suite=final_suites.exact,
            checkpoint=final_checkpoint,
            model=built.session.model,
            model_transport=final_transport,
        )
        portable_final_task = _evaluation_task(
            inputs=inputs,
            worker=worker,
            library=library,
            job=job,
            assignment_sha256=final_suites.portable.assignment_sha256,
            purpose="final",
            backend="portable",
            suite=final_suites.portable,
            checkpoint=final_checkpoint,
            model=built.session.model,
            model_transport=final_transport,
        )
        evaluation_tasks = (
            *(curve_tasks[update] for update in expected_updates),
            exact_final_task,
            portable_final_task,
        )
        if len({task.sha256 for task in evaluation_tasks}) != len(evaluation_tasks):
            raise RuntimeError("canonical evaluation task grid contains duplicates")
        ArtifactLookupIndex(root / "evaluation-index.sqlite3")
        executor = ProcessPoolExecutor(
            max_workers=min(config.evaluation_processes, len(evaluation_tasks)),
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_evaluation_worker_initializer,
            initargs=(os.getpid(), str(evaluator_lease_path())),
        )
        futures: dict[
            Future[CanonicalEvaluationTaskResult], CanonicalEvaluationTask
        ] = {}
        failed = True
        try:
            for task in evaluation_tasks:
                futures[executor.submit(evaluate_canonical_task, task)] = task
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
                    evaluation_torch_threads=config.evaluation_torch_threads,
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
                    authorization
                    if isinstance(authorization, SealedTestJobLease)
                    else None
                ),
            )
            resume_built.close()
            resume_built = None
            results = _collect_evaluation_results(futures)
            failed = False
        finally:
            if failed:
                for future in futures:
                    future.cancel()
                _stop_evaluation_executor(executor)
            else:
                executor.shutdown(wait=True)

        checkpoint_evaluations = tuple(
            CheckpointEvaluation(
                packages[update][0],
                results[curve_tasks[update].sha256].report_for(
                    curve_tasks[update],
                    curve_suites.exact,
                ),
            )
            for update in expected_updates
        )
        exact_final_report = results[exact_final_task.sha256].report_for(
            exact_final_task,
            final_suites.exact,
        )
        portable_final_report = results[portable_final_task.sha256].report_for(
            portable_final_task,
            final_suites.portable,
        )
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
