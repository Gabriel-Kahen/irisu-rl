"""Durable phase transitions for the canonical R3 experiment."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import secrets

from .r3b_artifacts import (
    ArtifactStore,
    ArtifactTypeError,
    ensure_private_directory,
    publish_private_file,
)
from .r3b_canonical_runner import (
    CanonicalRunInputs,
    PairedEvaluationSuites,
    PublishedCanonicalOutcome,
    load_published_canonical_outcome,
)
from .r3b_experiments import (
    ArmPhaseResult,
    SealedLearnerOutcomeReference,
    SealedTestLedger,
    SealedTestJobLease,
    SealedTestRunAuthorization,
    ValidationRunAuthorization,
    bind_validation_run,
)
from .r3b_local_runner import (
    _canonical_bytes,
    _load_claim,
    _load_claim_intent,
    _write_claim,
    _write_claim_intent,
)
from .r3b_operational import JobClaim


_VALIDATION_KIND = "irisu.r3b.validation-run-authorization"
_VALIDATION_VERSION = "r3b-validation-run-authorization-package-v1"
_SEALED_PACKAGE_VERSION = "r3b-sealed-test-run-package-v1"


@dataclass(frozen=True, slots=True)
class LoadedPhaseResults:
    phase: str
    results: tuple[ArmPhaseResult, ...]
    outputs: tuple[PublishedCanonicalOutcome, ...]


def load_completed_phase_results(
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
    phase: str,
) -> LoadedPhaseResults:
    """Strictly reconstruct every successful outcome in one installed phase."""

    if phase not in {"calibration", "validation", "test"}:
        raise ValueError("unknown experiment phase")
    records = inputs.workflow.phase_job_records(phase)
    if not records or any(record["status"] != "completed" for record in records):
        raise RuntimeError(f"{phase} is not completely successful")
    grouped: dict[str, list[object]] = {}
    published: list[PublishedCanonicalOutcome] = []
    budgets: dict[str, int] = {}
    for record in records:
        output_sha256 = record["output_sha256"]
        if not isinstance(output_sha256, str):
            raise ValueError("completed workflow job lacks its output artifact")
        outcome, output = load_published_canonical_outcome(
            inputs=inputs,
            store=store,
            output_artifact_sha256=output_sha256,
        )
        evidence = outcome.engineering_evidence
        if (
            evidence is None
            or evidence.job_sha256 != record["job_sha256"]
            or evidence.phase != phase
        ):
            raise ValueError("workflow output is foreign to its phase job")
        grouped.setdefault(evidence.arm_id, []).append(outcome)
        budget = int(record["budget_updates"])
        previous = budgets.setdefault(evidence.arm_id, budget)
        if previous != budget:
            raise ValueError("one arm cannot mix phase budgets")
        published.append(output)
    results = tuple(
        ArmPhaseResult(
            arm_id,
            phase,
            "complete",
            budgets[arm_id],
            tuple(sorted(outcomes, key=lambda value: value.learner_seed)),
        )
        for arm_id, outcomes in sorted(grouped.items())
    )
    return LoadedPhaseResults(phase, results, tuple(published))


@dataclass(frozen=True, slots=True)
class PublishedValidationAuthorization:
    authorization: ValidationRunAuthorization
    artifact_sha256: str


def prepare_validation_phase(
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
) -> PublishedValidationAuthorization:
    """Select calibrated learning rates, precommit test cells, and install jobs."""

    phase = load_completed_phase_results(inputs, store, "calibration")
    assignment_sha256s = {
        outcome.assignment_sha256
        for result in phase.results
        for outcome in result.outcomes
    }
    if len(assignment_sha256s) != 1:
        raise ValueError("calibration outcomes do not share one assignment")
    assignment_sha256 = next(iter(assignment_sha256s))
    test_suites = PairedEvaluationSuites.build(
        inputs,
        phase="test",
        learner_seed=inputs.plan.test_learner_seeds[0],
        assignment_sha256=assignment_sha256,
    )
    ledger = SealedTestLedger(inputs.root / "sealed-test.sqlite3")
    commitment = ledger.precommit(inputs.plan, test_suites.exact)
    authorization = bind_validation_run(
        inputs.plan,
        phase.results,
        commitment,
    )
    payload = {
        "version": _VALIDATION_VERSION,
        "workflow_manifest_sha256": inputs.workflow_manifest_sha256,
        "calibration_output_artifact_sha256s": [
            output.output_artifact_sha256 for output in phase.outputs
        ],
        "authorization": authorization.manifest(),
        "test_suite": test_suites.exact.manifest(),
    }
    envelope = store.publish(
        kind=_VALIDATION_KIND,
        version=_VALIDATION_VERSION,
        payload=payload,
    )
    existing = inputs.workflow.phase_job_records("validation")
    jobs = inputs.plan.trial_jobs("validation", authorization)
    if existing:
        if {record["job_sha256"] for record in existing} != {
            job.sha256 for job in jobs
        }:
            raise RuntimeError("installed validation jobs differ from authorization")
    else:
        inputs.workflow.append_jobs(jobs, authorization=authorization)
    return PublishedValidationAuthorization(authorization, envelope.artifact_id)


def load_validation_authorization(
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
    artifact_sha256: str,
) -> ValidationRunAuthorization:
    envelope = store.load(
        artifact_sha256,
        expected_kind=_VALIDATION_KIND,
        expected_version=_VALIDATION_VERSION,
    )
    payload = envelope.payload
    if not isinstance(payload, dict) or set(payload) != {
        "version",
        "workflow_manifest_sha256",
        "calibration_output_artifact_sha256s",
        "authorization",
        "test_suite",
    }:
        raise ValueError("validation authorization package schema differs")
    references = payload["calibration_output_artifact_sha256s"]
    if (
        payload["version"] != _VALIDATION_VERSION
        or payload["workflow_manifest_sha256"] != inputs.workflow_manifest_sha256
        or not isinstance(references, list)
        or any(type(value) is not str for value in references)
    ):
        raise ValueError("validation authorization package identity differs")
    phase = load_completed_phase_results(inputs, store, "calibration")
    if references != [output.output_artifact_sha256 for output in phase.outputs]:
        raise ValueError("validation authorization calibration references differ")
    authorization = ValidationRunAuthorization.from_manifest(
        payload["authorization"],
        plan=inputs.plan,
        calibration_results=phase.results,
    )
    from .r3b_evaluation import EvaluationSuite

    test_suite = EvaluationSuite.from_manifest(payload["test_suite"])
    if (
        authorization.test_commitment.test_suite_sha256 != test_suite.sha256
        or test_suite.split != "test"
        or test_suite.backend != "exact"
    ):
        raise ValueError("validation authorization test commitment differs")
    return authorization


@dataclass(frozen=True, slots=True)
class PublishedSealedAuthorization:
    authorization: SealedTestRunAuthorization
    artifact_sha256: str
    validation_artifact_sha256: str


@dataclass(frozen=True, slots=True)
class AcquiredSealedJob:
    claim: JobClaim
    lease: SealedTestJobLease
    ledger: SealedTestLedger


def _write_sealed_lease_secret(
    path: Path,
    *,
    job_sha256: str,
    authorization_artifact_sha256: str,
    token: str,
) -> None:
    payload = (
        _canonical_bytes(
            {
                "version": "r3b-sealed-job-secret-v1",
                "job_sha256": job_sha256,
                "authorization_artifact_sha256": authorization_artifact_sha256,
                "lease_token": token,
            }
        )
        + b"\n"
    )
    publish_private_file(path, payload)


def _load_sealed_lease_secret(
    path: Path,
    *,
    sealed: PublishedSealedAuthorization,
) -> tuple[str, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("sealed job secret is missing, linked, or not private")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("sealed job secret is malformed") from exc
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "version",
            "job_sha256",
            "authorization_artifact_sha256",
            "lease_token",
        }
        or value["version"] != "r3b-sealed-job-secret-v1"
        or value["authorization_artifact_sha256"] != sealed.artifact_sha256
        or payload != _canonical_bytes(value) + b"\n"
        or any(type(item) is not str for item in value.values())
    ):
        raise ValueError("sealed job secret schema or authorization differs")
    return value["job_sha256"], value["lease_token"]


def acquire_sealed_job(
    inputs: CanonicalRunInputs,
    sealed: PublishedSealedAuthorization,
    *,
    owner: str,
) -> AcquiredSealedJob:
    """Crash-safely acquire matching workflow and one-shot ledger authorities."""

    if sealed.authorization.plan.sha256 != inputs.plan.sha256:
        raise ValueError("sealed authorization belongs to another run")
    root = inputs.root
    workflow = inputs.workflow
    secret_root = ensure_private_directory(root / "secrets")
    active: list[JobClaim] = []
    for path in sorted(secret_root.glob("*.claim.json")):
        candidate = _load_claim(path)
        record = workflow.job_record(candidate.job_sha256)
        if record["status"] in {"claimed", "running", "trained"}:
            active.append(candidate)
    if len(active) > 1:
        raise RuntimeError("multiple active workflow claims exist")
    if active:
        claim = active[0]
        if claim.phase != "test" or claim.owner != owner:
            raise RuntimeError("active test claim belongs to another owner")
    else:
        intent = secret_root / "test.intent.json"
        if intent.exists():
            intent_phase, token, intent_owner = _load_claim_intent(intent)
            if intent_phase != "test" or intent_owner != owner:
                raise RuntimeError("test claim intent belongs to another owner")
            claim = workflow.resume_unstarted_claim("test", owner=owner, token=token)
        else:
            token = secrets.token_hex(32)
            _write_claim_intent(intent, phase="test", token=token, owner=owner)
            claim = None
        if claim is None:
            claim = workflow.claim_next("test", owner=owner, token=token)
        if claim is None:
            intent.unlink()
            raise RuntimeError("no pending sealed test job exists")
        _write_claim(secret_root / f"{claim.job_sha256}.claim.json", claim)
        intent.unlink()

    jobs = {
        job.sha256: job for job in inputs.plan.trial_jobs("test", sealed.authorization)
    }
    try:
        job = jobs[claim.job_sha256]
    except KeyError as exc:
        raise ValueError("workflow claim is foreign to sealed authorization") from exc
    ledger = SealedTestLedger(sealed.authorization.ledger_path)
    lease_secret = secret_root / f"{job.sha256}.sealed.json"
    lease_intent = secret_root / f"{job.sha256}.sealed.intent.json"
    if lease_secret.exists():
        secret_job, lease_token = _load_sealed_lease_secret(lease_secret, sealed=sealed)
        if secret_job != job.sha256:
            raise ValueError("sealed lease secret belongs to another job")
        state, outcome_sha256, failure_reason = ledger.job_state(
            sealed.authorization, job
        )
        if state == "complete":
            store = ArtifactStore(inputs.root / "artifacts")
            matches: list[tuple[SealedLearnerOutcomeReference, object]] = []
            for artifact_id in store.list():
                try:
                    reference = SealedLearnerOutcomeReference.load(store, artifact_id)
                except ArtifactTypeError:
                    continue
                if (
                    reference.plan_sha256 != inputs.plan.sha256
                    or reference.authorization_sha256
                    != sealed.authorization.authorization.sha256
                    or reference.job_sha256 != job.sha256
                    or reference.learner_outcome_sha256 != outcome_sha256
                ):
                    continue
                outcome = reference.resolve(
                    ledger=ledger,
                    sealed_run=sealed.authorization,
                    job=job,
                    loader=lambda output_id: load_published_canonical_outcome(
                        inputs=inputs,
                        store=store,
                        output_artifact_sha256=output_id,
                    )[0],
                )
                matches.append((reference, outcome))
            if len(matches) != 1:
                raise RuntimeError(
                    "completed sealed job lacks one recoverable outcome reference"
                )
            reference, outcome = matches[0]
            workflow.reconcile_sealed_completion(
                ledger=ledger,
                sealed_run=sealed.authorization,
                job=job,
                outcome_sha256=outcome.sha256,
                output_sha256=reference.output_artifact_sha256,
            )
            return acquire_sealed_job(inputs, sealed, owner=owner)
        if state == "failure":
            if not isinstance(failure_reason, str):
                raise RuntimeError("failed sealed job lacks its reason")
            workflow.reconcile_sealed_failure(
                ledger=ledger,
                sealed_run=sealed.authorization,
                job=job,
                failure_reason=failure_reason,
            )
            return acquire_sealed_job(inputs, sealed, owner=owner)
        lease = SealedTestJobLease(sealed.authorization, job, lease_token)
        if state == "running":
            reason = "orphaned sealed execution found after process restart"
            ledger.fail_job(lease, reason)
            workflow.reconcile_sealed_failure(
                ledger=ledger,
                sealed_run=sealed.authorization,
                job=job,
                failure_reason=reason,
            )
            raise RuntimeError(reason)
        if state != "leased":
            raise RuntimeError("sealed lease secret and ledger state disagree")
    else:
        if lease_intent.exists():
            if (
                lease_intent.is_symlink()
                or not lease_intent.is_file()
                or lease_intent.stat().st_mode & 0o077
            ):
                raise ValueError(
                    "sealed lease intent is missing, linked, or not private"
                )
            payload = lease_intent.read_bytes()
            try:
                value = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise ValueError("sealed lease intent is malformed") from exc
            if (
                not isinstance(value, dict)
                or set(value) != {"version", "job_sha256", "lease_token"}
                or value["version"] != "r3b-sealed-lease-intent-v1"
                or value["job_sha256"] != job.sha256
                or payload != _canonical_bytes(value) + b"\n"
            ):
                raise ValueError("sealed lease intent is malformed")
            lease_token = value["lease_token"]
            try:
                lease = ledger.resume_job(
                    sealed.authorization,
                    job,
                    lease_token=lease_token,
                )
            except RuntimeError:
                lease = ledger.claim_job(
                    sealed.authorization,
                    job,
                    lease_token=lease_token,
                )
        else:
            lease_token = secrets.token_hex(32)
            publish_private_file(
                lease_intent,
                _canonical_bytes(
                    {
                        "version": "r3b-sealed-lease-intent-v1",
                        "job_sha256": job.sha256,
                        "lease_token": lease_token,
                    }
                )
                + b"\n",
            )
            lease = ledger.claim_job(
                sealed.authorization,
                job,
                lease_token=lease_token,
            )
        _write_sealed_lease_secret(
            lease_secret,
            job_sha256=job.sha256,
            authorization_artifact_sha256=sealed.artifact_sha256,
            token=lease.lease_token,
        )
        lease_intent.unlink()
        parent = os.open(secret_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    return AcquiredSealedJob(claim, lease, ledger)


def load_sealed_authorization(
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
    artifact_sha256: str,
    *,
    allow_finalized: bool = False,
) -> PublishedSealedAuthorization:
    package = store.load(
        artifact_sha256,
        expected_kind="irisu.r3b.sealed-test-run-package",
        expected_version=_SEALED_PACKAGE_VERSION,
    ).payload
    if not isinstance(package, dict) or set(package) != {
        "version",
        "workflow_manifest_sha256",
        "validation_authorization_artifact_sha256",
        "validation_output_artifact_sha256s",
        "sealed_authorization_artifact_sha256",
    }:
        raise ValueError("sealed authorization package schema differs")
    validation_artifact = package["validation_authorization_artifact_sha256"]
    validation_outputs = package["validation_output_artifact_sha256s"]
    sealed_artifact = package["sealed_authorization_artifact_sha256"]
    if (
        package["version"] != _SEALED_PACKAGE_VERSION
        or package["workflow_manifest_sha256"] != inputs.workflow_manifest_sha256
        or type(validation_artifact) is not str
        or type(sealed_artifact) is not str
        or not isinstance(validation_outputs, list)
        or any(type(value) is not str for value in validation_outputs)
    ):
        raise ValueError("sealed authorization package identity differs")
    validation_run = load_validation_authorization(inputs, store, validation_artifact)
    phase = load_completed_phase_results(inputs, store, "validation")
    if validation_outputs != [
        output.output_artifact_sha256 for output in phase.outputs
    ]:
        raise ValueError("sealed authorization validation references differ")
    assignments = {
        outcome.assignment_sha256
        for result in phase.results
        for outcome in result.outcomes
    }
    if len(assignments) != 1:
        raise ValueError("validation outcomes do not share one assignment")
    assignment = next(iter(assignments))
    validation_suite = PairedEvaluationSuites.build(
        inputs,
        phase="validation",
        learner_seed=inputs.plan.validation_learner_seeds[0],
        assignment_sha256=assignment,
    ).exact
    test_suite = PairedEvaluationSuites.build(
        inputs,
        phase="test",
        learner_seed=inputs.plan.test_learner_seeds[0],
        assignment_sha256=assignment,
    ).exact
    sealed = SealedTestRunAuthorization.load(
        store,
        sealed_artifact,
        plan=inputs.plan,
        validation_run=validation_run,
        validation_results=phase.results,
        validation_suite=validation_suite,
        test_suite=test_suite,
        allow_finalized=allow_finalized,
    )
    return PublishedSealedAuthorization(
        sealed,
        artifact_sha256,
        validation_artifact,
    )


def prepare_sealed_test_phase(
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
    *,
    validation_artifact_sha256: str,
) -> PublishedSealedAuthorization:
    """Select the sole candidate, authorize once, and install sealed jobs."""

    existing_packages: list[str] = []
    for artifact_id in store.list():
        try:
            payload = store.load(
                artifact_id,
                expected_kind="irisu.r3b.sealed-test-run-package",
                expected_version=_SEALED_PACKAGE_VERSION,
            ).payload
        except ArtifactTypeError:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("workflow_manifest_sha256")
            == inputs.workflow_manifest_sha256
            and payload.get("validation_authorization_artifact_sha256")
            == validation_artifact_sha256
        ):
            existing_packages.append(artifact_id)
    if len(existing_packages) > 1:
        raise RuntimeError("multiple sealed authorization packages disagree")
    if existing_packages:
        existing = load_sealed_authorization(inputs, store, existing_packages[0])
        jobs = inputs.plan.trial_jobs("test", existing.authorization)
        records = inputs.workflow.phase_job_records("test")
        if records:
            if {record["job_sha256"] for record in records} != {
                job.sha256 for job in jobs
            }:
                raise RuntimeError(
                    "installed test jobs differ from sealed authorization"
                )
        else:
            inputs.workflow.append_jobs(jobs, authorization=existing.authorization)
        return existing

    validation_run = load_validation_authorization(
        inputs, store, validation_artifact_sha256
    )
    phase = load_completed_phase_results(inputs, store, "validation")
    assignments = {
        outcome.assignment_sha256
        for result in phase.results
        for outcome in result.outcomes
    }
    if len(assignments) != 1:
        raise ValueError("validation outcomes do not share one assignment")
    assignment = next(iter(assignments))
    validation_suites = PairedEvaluationSuites.build(
        inputs,
        phase="validation",
        learner_seed=inputs.plan.validation_learner_seeds[0],
        assignment_sha256=assignment,
    )
    test_suites = PairedEvaluationSuites.build(
        inputs,
        phase="test",
        learner_seed=inputs.plan.test_learner_seeds[0],
        assignment_sha256=assignment,
    )
    ledger = SealedTestLedger(inputs.root / "sealed-test.sqlite3")
    sealed = ledger.authorize_once(
        inputs.plan,
        validation_run,
        phase.results,
        validation_suites.exact,
        test_suites.exact,
    )
    sealed_artifact = sealed.publish(store)
    package = store.publish(
        kind="irisu.r3b.sealed-test-run-package",
        version=_SEALED_PACKAGE_VERSION,
        payload={
            "version": _SEALED_PACKAGE_VERSION,
            "workflow_manifest_sha256": inputs.workflow_manifest_sha256,
            "validation_authorization_artifact_sha256": (validation_artifact_sha256),
            "validation_output_artifact_sha256s": [
                output.output_artifact_sha256 for output in phase.outputs
            ],
            "sealed_authorization_artifact_sha256": sealed_artifact,
        },
    )
    jobs = inputs.plan.trial_jobs("test", sealed)
    existing = inputs.workflow.phase_job_records("test")
    if existing:
        if {record["job_sha256"] for record in existing} != {
            job.sha256 for job in jobs
        }:
            raise RuntimeError("installed test jobs differ from sealed authorization")
    else:
        inputs.workflow.append_jobs(jobs, authorization=sealed)
    return PublishedSealedAuthorization(
        sealed,
        package.artifact_id,
        validation_artifact_sha256,
    )
