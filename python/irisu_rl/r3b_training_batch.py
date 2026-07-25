"""CPU-budgeted concurrent training with sequential canonical evaluation."""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import secrets
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .cpu_parallelism import TrainingCpuPlan
from .r3b_artifacts import ensure_private_directory, publish_private_file
from .r3b_experiments import ValidationRunAuthorization
from .r3b_local_runner import (
    LocalTrainingResult,
    _load_claim,
    _load_claim_intent,
    _write_claim,
    _write_claim_intent,
    run_local_canonical_updates,
)
from .r3b_lock import evaluator_lease_path
from .r3b_operational import JobClaim, R3BWorkflow
from .r3b_supervisor import (
    CanonicalEvaluationResult,
    _capture_evaluation_process_groups,
    _evaluation_worker_initializer,
    _stop_evaluation_executor,
    evaluate_trained_canonical_job,
)

_SAFE_OWNER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")


def _train_one(
    run_directory: str,
    worker_path: str,
    phase: str,
    owner: str,
    job_sha256: str,
    training_cpu_ids: tuple[int, ...],
    authorization: ValidationRunAuthorization | None,
) -> LocalTrainingResult:
    os.sched_setaffinity(0, training_cpu_ids)
    return run_local_canonical_updates(
        run_directory,
        worker_path=worker_path,
        max_new_updates=2**31 - 1,
        owner=owner,
        phase=phase,
        authorization=authorization,
        allow_parallel_claims=True,
        expected_job_sha256=job_sha256,
    )


@dataclass(frozen=True, slots=True)
class CanonicalBatchResult:
    cpu_plan: TrainingCpuPlan
    training: tuple[LocalTrainingResult, ...]
    evaluation: tuple[CanonicalEvaluationResult, ...]
    version: str = "r3b-canonical-training-batch-v1"

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "cpu_plan": self.cpu_plan.manifest(),
            "cpu_plan_sha256": self.cpu_plan.sha256,
            "training": [value.manifest() for value in self.training],
            "evaluation": [value.manifest() for value in self.evaluation],
            "acceptance_eligible": True,
            "transfer_eligible": False,
        }


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _wave_path(root: Path, phase: str) -> Path:
    secret_root = ensure_private_directory(root / "secrets")
    return secret_root / f"training-wave-{phase}.json"


def _wave_identity_path(root: Path, phase: str) -> Path:
    secret_root = ensure_private_directory(root / "secrets")
    return secret_root / f"training-wave-{phase}.identity.json"


def _bind_cpu_wave_identity(
    root: Path,
    *,
    phase: str,
    owner: str,
    plan: TrainingCpuPlan,
) -> None:
    """Bind plan and owner before the first durable job claim is attempted."""

    path = _wave_identity_path(root, phase)
    payload = _canonical_bytes(
        {
            "version": "r3b-training-cpu-wave-identity-v1",
            "phase": phase,
            "owner": owner,
            "cpu_plan": plan.manifest(),
        }
    )
    if path.exists():
        metadata = path.stat(follow_symlinks=False)
        if (
            path.is_symlink()
            or not path.is_file()
            or metadata.st_mode & 0o077
            or metadata.st_uid != os.geteuid()
            or metadata.st_nlink != 1
        ):
            raise ValueError("training CPU wave identity is unsafe")
        if path.read_bytes() != payload:
            raise ValueError("training CPU wave changed across resume")
    else:
        publish_private_file(path, payload)


def _load_cpu_wave(
    root: Path,
    *,
    phase: str,
    owner: str,
    plan: TrainingCpuPlan,
) -> tuple[JobClaim, ...] | None:
    """Load one immutable in-progress wave and require its resume identity."""

    path = _wave_path(root, phase)
    if not path.exists():
        return None
    metadata = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not path.is_file()
        or metadata.st_mode & 0o077
        or metadata.st_uid != os.geteuid()
        or metadata.st_nlink != 1
    ):
        raise ValueError("training CPU wave is missing, linked, or not private")
    try:
        value = json.loads(path.read_bytes())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("training CPU wave is malformed") from exc
    if (
        not isinstance(value, dict)
        or set(value) != {"version", "phase", "owner", "cpu_plan", "jobs"}
        or value.get("version") != "r3b-training-cpu-wave-v1"
        or value.get("phase") != phase
        or value.get("owner") != owner
        or value.get("cpu_plan") != plan.manifest()
        or path.read_bytes() != _canonical_bytes(value)
        or not isinstance(value.get("jobs"), list)
    ):
        raise ValueError("training CPU wave changed across resume")
    claims: list[JobClaim] = []
    for item in value["jobs"]:
        if (
            not isinstance(item, dict)
            or set(item) != {"job_sha256", "owner"}
            or not isinstance(item["job_sha256"], str)
            or not isinstance(item["owner"], str)
        ):
            raise ValueError("training CPU wave job identity is malformed")
        claim = _load_claim(root / "secrets" / f"{item['job_sha256']}.claim.json")
        if claim.owner != item["owner"] or claim.phase != phase:
            raise ValueError("training CPU wave claim identity changed")
        claims.append(claim)
    if not claims:
        raise ValueError("training CPU wave has no jobs")
    return tuple(claims)


def _publish_cpu_wave(
    root: Path,
    *,
    phase: str,
    owner: str,
    plan: TrainingCpuPlan,
    claims: tuple[JobClaim, ...],
) -> None:
    value = {
        "version": "r3b-training-cpu-wave-v1",
        "phase": phase,
        "owner": owner,
        "cpu_plan": plan.manifest(),
        "jobs": [
            {"job_sha256": claim.job_sha256, "owner": claim.owner} for claim in claims
        ],
    }
    publish_private_file(_wave_path(root, phase), _canonical_bytes(value))


def _clear_cpu_wave(root: Path, phase: str) -> None:
    paths = (_wave_path(root, phase), _wave_identity_path(root, phase))
    for path in paths:
        path.unlink(missing_ok=True)
    parent_fd = os.open(paths[0].parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _prepare_claims(
    root: Path,
    *,
    workflow: R3BWorkflow,
    phase: str,
    owner: str,
    count: int,
) -> tuple[JobClaim, ...]:
    """Durably claim a deterministic wave before any child is spawned."""

    secret_root = ensure_private_directory(root / "secrets")
    active: dict[str, JobClaim] = {}
    for path in sorted(secret_root.glob("*.claim.json")):
        claim = _load_claim(path)
        record = workflow.job_record(claim.job_sha256)
        if record["status"] in {"claimed", "running", "trained"}:
            if claim.owner in active:
                raise RuntimeError("parallel training owner has multiple active claims")
            active[claim.owner] = claim
    prepared: list[JobClaim] = []
    for index in range(count):
        worker_owner = f"{owner}-{index + 1}"
        claim = active.get(worker_owner)
        if claim is not None:
            if claim.phase != phase:
                raise ValueError("parallel training claim belongs to another phase")
            prepared.append(claim)
            continue
        intent = secret_root / f"{phase}.{worker_owner}.intent.json"
        if intent.exists():
            intent_phase, token, intent_owner = _load_claim_intent(intent)
            if intent_phase != phase or intent_owner != worker_owner:
                raise RuntimeError("parallel claim intent belongs to another worker")
            claim = workflow.resume_unstarted_claim(
                phase, owner=worker_owner, token=token
            )
            if claim is None:
                claim = workflow.claim_next(phase, owner=worker_owner, token=token)
        else:
            token = secrets.token_hex(32)
            _write_claim_intent(
                intent,
                phase=phase,
                token=token,
                owner=worker_owner,
            )
            claim = workflow.claim_next(phase, owner=worker_owner, token=token)
        if claim is None:
            intent.unlink()
            continue
        _write_claim(secret_root / f"{claim.job_sha256}.claim.json", claim)
        intent.unlink()
        parent_fd = os.open(secret_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        prepared.append(claim)
    return tuple(prepared)


def run_canonical_training_batch(
    run_directory: str | Path,
    *,
    exact_worker_path: str | Path,
    portable_library_path: str | Path,
    phase: str,
    owner: str,
    cpu_plan: TrainingCpuPlan,
    authorization: ValidationRunAuthorization | None = None,
) -> CanonicalBatchResult:
    """Train a bounded wave concurrently, then evaluate without oversubscription."""

    if phase not in {"calibration", "validation"}:
        raise ValueError("parallel training is limited to nonsealed phases")
    if phase == "calibration" and authorization is not None:
        raise ValueError("calibration batch cannot carry an authorization")
    if phase == "validation" and not isinstance(
        authorization, ValidationRunAuthorization
    ):
        raise ValueError("validation batch requires its authorization")
    root = Path(run_directory).resolve(strict=True)
    worker = Path(exact_worker_path).resolve(strict=True)
    library = Path(portable_library_path).resolve(strict=True)
    workflow = R3BWorkflow(root / "workflow.sqlite3")
    if not _SAFE_OWNER.fullmatch(owner):
        raise ValueError("batch owner is unsafe")
    _bind_cpu_wave_identity(
        root,
        phase=phase,
        owner=owner,
        plan=cpu_plan,
    )
    wave_claims = _load_cpu_wave(
        root,
        phase=phase,
        owner=owner,
        plan=cpu_plan,
    )
    if wave_claims is None:
        wave_claims = _prepare_claims(
            root,
            workflow=workflow,
            phase=phase,
            owner=owner,
            count=cpu_plan.parallel_jobs,
        )
        if wave_claims:
            _publish_cpu_wave(
                root,
                phase=phase,
                owner=owner,
                plan=cpu_plan,
                claims=wave_claims,
            )
        else:
            _clear_cpu_wave(root, phase)
            raise RuntimeError(f"no unfinished {phase} jobs remain")
    records = {
        claim.job_sha256: workflow.job_record(claim.job_sha256) for claim in wave_claims
    }
    failed = [
        claim.job_sha256
        for claim in wave_claims
        if records[claim.job_sha256]["status"] == "failed"
    ]
    if failed:
        raise RuntimeError(f"training CPU wave contains failed jobs: {failed}")
    claims = tuple(
        claim
        for claim in wave_claims
        if records[claim.job_sha256]["status"] != "completed"
    )
    jobs = len(claims)
    if jobs <= 0:
        _clear_cpu_wave(root, phase)
        return CanonicalBatchResult(cpu_plan, (), ())

    context = multiprocessing.get_context("spawn")
    ready_gate = context.Event()
    lease_path = str(evaluator_lease_path())
    executor: ProcessPoolExecutor | None = None
    futures = []
    try:
        executor = ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=context,
            initializer=_evaluation_worker_initializer,
            initargs=(os.getpid(), lease_path, ready_gate),
        )
        executor._irisu_ready_gate = ready_gate
        executor._irisu_evaluator_lease_path = lease_path
        futures = [
            executor.submit(
                _train_one,
                str(root),
                str(worker),
                phase,
                claim.owner,
                claim.job_sha256,
                cpu_plan.training_cpu_ids,
                authorization,
            )
            for claim in claims
        ]
        _capture_evaluation_process_groups(executor)
        training = tuple(
            sorted(
                (future.result() for future in futures),
                key=lambda value: value.job_sha256,
            )
        )
        if not all(result.training_complete for result in training):
            raise RuntimeError("parallel canonical training stopped before completion")
    except BaseException as primary:
        for future in futures:
            future.cancel()
        if executor is not None:
            try:
                _stop_evaluation_executor(executor)
            except BaseException as cleanup:
                primary.add_note(
                    "parallel training cleanup also failed: "
                    f"{type(cleanup).__name__}: {cleanup}"
                )
        raise
    else:
        assert executor is not None
        _stop_evaluation_executor(executor)

    evaluation = tuple(
        evaluate_trained_canonical_job(
            root,
            exact_worker_path=worker,
            portable_library_path=library,
            phase=phase,
            authorization=authorization,
            job_sha256=result.job_sha256,
        )
        for result in training
    )
    if not all(
        workflow.job_record(claim.job_sha256)["status"] == "completed"
        for claim in wave_claims
    ):
        raise RuntimeError("training CPU wave did not complete every job")
    _clear_cpu_wave(root, phase)
    return CanonicalBatchResult(cpu_plan, training, evaluation)
