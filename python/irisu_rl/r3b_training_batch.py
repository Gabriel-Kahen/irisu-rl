"""CPU-budgeted training with shard-parallel canonical evaluation."""

from __future__ import annotations

import json
import multiprocessing
import os
import re
import secrets
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
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


def _task_ids(task_root: Path = Path("/proc/self/task")) -> tuple[int, ...]:
    try:
        task_ids = tuple(
            sorted(
                int(entry.name)
                for entry in task_root.iterdir()
                if entry.name.isdecimal()
            )
        )
    except OSError as exc:
        raise RuntimeError("training process threads could not be inspected") from exc
    if not task_ids:
        raise RuntimeError("training process has no inspectable threads")
    return task_ids


def _set_process_affinity(
    cpu_ids: tuple[int, ...],
    *,
    task_root: Path = Path("/proc/self/task"),
) -> None:
    """Restrict every existing thread and verify a stable process-wide mask."""

    target = frozenset(cpu_ids)
    if not target:
        raise ValueError("training CPU affinity cannot be empty")
    os.sched_setaffinity(0, target)
    for _attempt in range(16):
        before = _task_ids(task_root)
        for task_id in before:
            with suppress(ProcessLookupError):
                os.sched_setaffinity(task_id, target)
        after = _task_ids(task_root)
        if after != before:
            continue
        for task_id in after:
            try:
                actual = os.sched_getaffinity(task_id)
            except ProcessLookupError:
                break
            if actual != target:
                raise RuntimeError(
                    "training CPU affinity did not apply to every thread"
                )
        else:
            if _task_ids(task_root) == after:
                return
    raise RuntimeError("training process threads changed while applying CPU affinity")


def _train_one(
    run_directory: str,
    worker_path: str,
    phase: str,
    owner: str,
    job_sha256: str,
    training_cpu_ids: tuple[int, ...],
    authorization: ValidationRunAuthorization | None,
) -> LocalTrainingResult:
    _set_process_affinity(training_cpu_ids)
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
    wave_complete: bool
    remaining_jobs: int
    version: str = "r3b-canonical-training-batch-v2"

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "cpu_plan": self.cpu_plan.manifest(),
            "cpu_plan_sha256": self.cpu_plan.sha256,
            "training": [value.manifest() for value in self.training],
            "evaluation": [value.manifest() for value in self.evaluation],
            "wave_complete": self.wave_complete,
            "remaining_jobs": self.remaining_jobs,
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
            "version": "r3b-training-cpu-wave-identity-v2",
            "phase": phase,
            "owner": owner,
            "cpu_policy": plan.policy_manifest(),
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
        or set(value) != {"version", "phase", "owner", "cpu_policy", "jobs"}
        or value.get("version") != "r3b-training-cpu-wave-v2"
        or value.get("phase") != phase
        or value.get("owner") != owner
        or value.get("cpu_policy") != plan.policy_manifest()
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
        "version": "r3b-training-cpu-wave-v2",
        "phase": phase,
        "owner": owner,
        "cpu_policy": plan.policy_manifest(),
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _batch_owner_pattern(owner: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(owner)}-([1-9][0-9]*)$")


def _phase_intents(
    secret_root: Path,
    phase: str,
) -> tuple[tuple[Path, str, str], ...]:
    intents: list[tuple[Path, str, str]] = []
    for path in sorted(secret_root.glob(f"{phase}*.intent.json")):
        intent_phase, token, intent_owner = _load_claim_intent(path)
        if intent_phase == phase:
            intents.append((path, token, intent_owner))
    return tuple(intents)


def _preflight_batch_recovery(
    root: Path,
    *,
    workflow: R3BWorkflow,
    phase: str,
    owner: str,
) -> None:
    """Reject foreign or unrecoverable active work before binding a batch."""

    secret_root = ensure_private_directory(root / "secrets")
    owner_pattern = _batch_owner_pattern(owner)
    intent_owners = {
        intent_owner
        for _path, _token, intent_owner in _phase_intents(secret_root, phase)
    }
    foreign = sorted(
        intent_owner
        for intent_owner in intent_owners
        if owner_pattern.fullmatch(intent_owner) is None
    )
    active = [
        record
        for record in workflow.phase_job_records(phase)
        if record["status"] in {"claimed", "running", "trained"}
    ]
    foreign.extend(
        str(record["owner"])
        for record in active
        if owner_pattern.fullmatch(str(record["owner"])) is None
    )
    if foreign:
        raise RuntimeError(
            "parallel training cannot adopt active claims owned by "
            + ", ".join(sorted(set(foreign)))
        )
    for record in active:
        job_sha256 = str(record["job_sha256"])
        record_owner = str(record["owner"])
        secret = secret_root / f"{job_sha256}.claim.json"
        if not secret.exists() and record_owner not in intent_owners:
            raise RuntimeError("active batch claim has no durable recovery secret")


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
    owner_pattern = _batch_owner_pattern(owner)
    batch_active: dict[int, JobClaim] = {}
    foreign: list[str] = []
    for active_owner, claim in active.items():
        match = owner_pattern.fullmatch(active_owner)
        if match is None:
            foreign.append(active_owner)
            continue
        slot = int(match.group(1))
        if slot in batch_active:
            raise RuntimeError("parallel training slot has multiple active claims")
        batch_active[slot] = claim
    if foreign:
        raise RuntimeError(
            "parallel training cannot adopt active claims owned by "
            + ", ".join(sorted(foreign))
        )
    intent_slots: dict[int, tuple[Path, str, str]] = {}
    for path, token, intent_owner in _phase_intents(secret_root, phase):
        match = owner_pattern.fullmatch(intent_owner)
        if match is None:
            raise RuntimeError(
                f"parallel training cannot adopt active claims owned by {intent_owner}"
            )
        slot = int(match.group(1))
        if slot in intent_slots:
            raise RuntimeError("parallel training slot has multiple claim intents")
        intent_slots[slot] = (path, token, intent_owner)
    for slot, (intent, token, intent_owner) in sorted(intent_slots.items()):
        claim = batch_active.get(slot)
        if claim is None:
            claim = workflow.resume_unstarted_claim(
                phase,
                owner=intent_owner,
                token=token,
            )
            if claim is None:
                claim = workflow.claim_next(
                    phase,
                    owner=intent_owner,
                    token=token,
                )
            if claim is not None:
                _write_claim(secret_root / f"{claim.job_sha256}.claim.json", claim)
                batch_active[slot] = claim
        elif (
            claim.phase != phase or claim.owner != intent_owner or claim.token != token
        ):
            raise RuntimeError("parallel claim intent differs from active claim")
        intent.unlink()
        _fsync_directory(secret_root)
    for _slot, claim in sorted(batch_active.items()):
        if claim.phase != phase:
            raise ValueError("parallel training claim belongs to another phase")
    for slot in range(1, count + 1):
        if slot in batch_active:
            continue
        worker_owner = f"{owner}-{slot}"
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
        _fsync_directory(secret_root)
        batch_active[slot] = claim
    return tuple(claim for _slot, claim in sorted(batch_active.items()))


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
    """Train a bounded wave concurrently, then evaluate with a bounded shard pool."""

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
    _preflight_batch_recovery(
        root,
        workflow=workflow,
        phase=phase,
        owner=owner,
    )
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
    unfinished = tuple(
        claim
        for claim in wave_claims
        if records[claim.job_sha256]["status"] != "completed"
    )
    claims = unfinished[: cpu_plan.parallel_jobs]
    jobs = len(claims)
    if jobs <= 0:
        _clear_cpu_wave(root, phase)
        return CanonicalBatchResult(cpu_plan, (), (), True, 0)

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
    final_statuses = tuple(
        workflow.job_record(claim.job_sha256)["status"] for claim in wave_claims
    )
    remaining_jobs = sum(status != "completed" for status in final_statuses)
    wave_complete = remaining_jobs == 0
    if wave_complete:
        _clear_cpu_wave(root, phase)
    return CanonicalBatchResult(
        cpu_plan,
        training,
        evaluation,
        wave_complete,
        remaining_jobs,
    )
