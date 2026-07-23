"""Restartable, one-shot execution of the sealed R3 scripted baselines."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import secrets
from typing import Any

from irisu_env import IrisuEnv

from .r3b_artifacts import (
    ArtifactLookupIndex,
    ArtifactStore,
    ensure_private_directory,
    publish_private_file,
)
from .r3b_canonical_runner import CanonicalRunInputs, PairedEvaluationSuites
from .r3b_evaluation import (
    BaselineArtifactBundle,
    EvaluationReport,
    EvaluationSuite,
    ScriptedBaselineSpec,
)
from .r3b_evaluation_shards import (
    EvaluationShardPlan,
    EvaluationShardReport,
    evaluate_scripted_shard,
    merge_evaluation_shards,
    plan_evaluation_shards,
)
from .r3b_experiments import (
    SealedBaselineBatchLease,
    SealedBaselineEvidenceArtifact,
    SealedTestLedger,
)
from .r3b_phases import PublishedSealedAuthorization


_SHARD_KIND = "irisu.r3b.scripted-baseline-shard-package"
_SHARD_VERSION = "r3b-scripted-baseline-shard-package-v1"
_BUNDLE_KIND = "irisu.r3b.scripted-baseline-bundle-package"
_BUNDLE_VERSION = "r3b-scripted-baseline-bundle-package-v1"
_EVIDENCE_KIND = "irisu.r3b.sealed-baseline-evidence"
_EVIDENCE_VERSION = "r3b-sealed-baseline-evidence-artifact-v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _lease_index(store: ArtifactStore, lease: SealedBaselineBatchLease) -> ArtifactLookupIndex:
    return ArtifactLookupIndex(
        store.root.parent / f"sealed-baseline-index-{lease.sha256}.sqlite3"
    )


def _baseline_from_manifest(value: object) -> ScriptedBaselineSpec:
    if not isinstance(value, dict) or set(value) != {
        "version",
        "baseline_id",
        "parameters",
    }:
        raise ValueError("scripted baseline manifest schema differs")
    parameters = value["parameters"]
    if (
        type(value["version"]) is not str
        or type(value["baseline_id"]) is not str
        or not isinstance(parameters, dict)
        or any(type(name) is not str for name in parameters)
    ):
        raise ValueError("scripted baseline manifest fields are malformed")
    try:
        baseline = ScriptedBaselineSpec(
            value["baseline_id"],
            tuple(sorted(parameters.items())),
            value["version"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("scripted baseline manifest is malformed") from exc
    if baseline.manifest() != value:
        raise ValueError("scripted baseline manifest is noncanonical")
    return baseline


def _scripted_report(
    *,
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
    simulator: Any,
    suite: EvaluationSuite,
    baseline: ScriptedBaselineSpec,
    purpose: str,
    lease: SealedBaselineBatchLease,
) -> EvaluationReport:
    if purpose not in {"primary", "replay", "diagnostic"}:
        raise ValueError("scripted baseline execution purpose is invalid")
    lease.assert_running()
    evaluator_sha256 = _sha256(
        {
            "version": "r3b-scripted-baseline-evaluator-v1",
            "workflow_manifest_sha256": inputs.workflow_manifest_sha256,
            "purpose": purpose,
            "algorithm": "scripted-semantic-fixed-cell-v1",
            "sealed_baseline_lease_sha256": lease.sha256,
        }
    )
    worker_identity_sha256 = _sha256(
        {
            "version": "r3b-scripted-baseline-worker-v1",
            "workflow_manifest_sha256": inputs.workflow_manifest_sha256,
            "runtime_identity_sha256": suite.runtime_identity_sha256,
            "backend": suite.backend,
            "shards": inputs.config.evaluation_shards,
            "sealed_baseline_lease_sha256": lease.sha256,
        }
    )
    plans = plan_evaluation_shards(suite, inputs.config.evaluation_shards)
    index = _lease_index(store, lease)
    reports: list[EvaluationShardReport] = []
    for shard in plans:
        lease.assert_running()
        execution_identity_sha256 = _sha256(
            {
                "version": "r3b-scripted-baseline-shard-execution-v1",
                "suite_sha256": suite.sha256,
                "baseline_sha256": baseline.sha256,
                "purpose": purpose,
                "shard_plan_sha256": shard.sha256,
                "evaluator_sha256": evaluator_sha256,
                "worker_identity_sha256": worker_identity_sha256,
                "sealed_baseline_lease_sha256": lease.sha256,
            }
        )
        lookup_key = _sha256(
            {
                "version": "r3b-scripted-baseline-shard-lookup-v1",
                "execution_identity_sha256": execution_identity_sha256,
            }
        )
        envelope = index.lookup(
            lookup_key,
            store,
            expected_kind=_SHARD_KIND,
            expected_version=_SHARD_VERSION,
        )
        if envelope is None:
            report = evaluate_scripted_shard(
                simulator,
                (
                    inputs.exact_bundle.store
                    if suite.backend == "exact"
                    else inputs.portable_bundle.store
                ),
                suite,
                baseline,
                shard,
                evaluator_sha256=evaluator_sha256,
                expected_assignment_sha256=suite.assignment_sha256,
                execution_identity_sha256=execution_identity_sha256,
            )
            envelope = store.publish(
                kind=_SHARD_KIND,
                version=_SHARD_VERSION,
                payload={
                    "baseline": baseline.manifest(),
                    "suite": suite.manifest(),
                    "purpose": purpose,
                    "shard_plan": shard.manifest(),
                    "report": report.report.manifest(),
                    "shard_report": report.manifest(),
                    "evaluator_sha256": evaluator_sha256,
                    "worker_identity_sha256": worker_identity_sha256,
                    "execution_identity_sha256": execution_identity_sha256,
                    "workflow_manifest_sha256": inputs.workflow_manifest_sha256,
                    "sealed_baseline_lease_sha256": lease.sha256,
                },
            )
            index.record(lookup_key, envelope)
        payload = envelope.payload
        expected = {
            "baseline",
            "suite",
            "purpose",
            "shard_plan",
            "report",
            "shard_report",
            "evaluator_sha256",
            "worker_identity_sha256",
            "execution_identity_sha256",
            "workflow_manifest_sha256",
            "sealed_baseline_lease_sha256",
        }
        if not isinstance(payload, dict) or set(payload) != expected:
            raise ValueError("scripted baseline shard package schema differs")
        stored_baseline = _baseline_from_manifest(payload["baseline"])
        stored_suite = EvaluationSuite.from_manifest(payload["suite"])
        stored_shard = EvaluationShardPlan.from_manifest(payload["shard_plan"])
        if (
            payload["workflow_manifest_sha256"] != inputs.workflow_manifest_sha256
            or stored_baseline != baseline
            or stored_suite != suite
            or stored_shard != shard
            or payload["purpose"] != purpose
            or payload["evaluator_sha256"] != evaluator_sha256
            or payload["worker_identity_sha256"] != worker_identity_sha256
            or payload["execution_identity_sha256"] != execution_identity_sha256
            or payload["sealed_baseline_lease_sha256"] != lease.sha256
        ):
            raise ValueError("indexed scripted baseline shard differs")
        stored_report = EvaluationReport.from_manifest(payload["report"], suite=suite)
        report = EvaluationShardReport.from_manifest(
            payload["shard_report"],
            shard=shard,
            report=stored_report,
        )
        if (
            stored_report.policy_sha256 != baseline.sha256
            or stored_report.evaluator_sha256 != evaluator_sha256
            or stored_report.execution_identity_sha256 != execution_identity_sha256
        ):
            raise ValueError("indexed scripted baseline report differs")
        reports.append(report)
    return merge_evaluation_shards(suite, tuple(reports))


def _load_baseline_bundle(
    *,
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
    suites: PairedEvaluationSuites,
    baseline: ScriptedBaselineSpec,
    lease: SealedBaselineBatchLease,
) -> tuple[str, BaselineArtifactBundle] | None:
    lease.assert_running()
    lookup_key = _sha256(
        {
            "version": "r3b-scripted-baseline-bundle-lookup-v1",
            "workflow_manifest_sha256": inputs.workflow_manifest_sha256,
            "baseline_sha256": baseline.sha256,
            "exact_suite_sha256": suites.exact.sha256,
            "portable_suite_sha256": suites.portable.sha256,
            "sealed_baseline_lease_sha256": lease.sha256,
        }
    )
    envelope = _lease_index(store, lease).lookup(
        lookup_key,
        store,
        expected_kind=_BUNDLE_KIND,
        expected_version=_BUNDLE_VERSION,
    )
    if envelope is None:
        return None
    payload = envelope.payload
    expected = {
        "baseline",
        "primary_suite",
        "primary_report",
        "primary_replay_report",
        "diagnostic_suite",
        "diagnostic_report",
        "logical_manifest_sha256",
        "workflow_manifest_sha256",
        "sealed_baseline_lease_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("scripted baseline bundle package schema differs")
    stored_baseline = _baseline_from_manifest(payload["baseline"])
    primary_suite = EvaluationSuite.from_manifest(payload["primary_suite"])
    diagnostic_suite = EvaluationSuite.from_manifest(payload["diagnostic_suite"])
    if (
        payload["workflow_manifest_sha256"] != inputs.workflow_manifest_sha256
        or stored_baseline != baseline
        or primary_suite != suites.exact
        or diagnostic_suite != suites.portable
        or payload["logical_manifest_sha256"] != suites.logical_manifest.sha256
        or payload["sealed_baseline_lease_sha256"] != lease.sha256
    ):
        raise ValueError("scripted baseline bundle suites differ")
    bundle = BaselineArtifactBundle(
        baseline,
        primary_suite,
        EvaluationReport.from_manifest(payload["primary_report"], suite=primary_suite),
        EvaluationReport.from_manifest(
            payload["primary_replay_report"], suite=primary_suite
        ),
        diagnostic_suite,
        EvaluationReport.from_manifest(
            payload["diagnostic_report"], suite=diagnostic_suite
        ),
        suites.logical_manifest,
        suites.portable_library,
        suites.exact_library,
    )
    bundle.evidence()
    return envelope.artifact_id, bundle


def _publish_baseline_bundle(
    *,
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
    suites: PairedEvaluationSuites,
    baseline: ScriptedBaselineSpec,
    primary: EvaluationReport,
    replay: EvaluationReport,
    diagnostic: EvaluationReport,
    lease: SealedBaselineBatchLease,
) -> tuple[str, BaselineArtifactBundle]:
    lease.assert_running()
    bundle = BaselineArtifactBundle(
        baseline,
        suites.exact,
        primary,
        replay,
        suites.portable,
        diagnostic,
        suites.logical_manifest,
        suites.portable_library,
        suites.exact_library,
    )
    bundle.evidence()
    artifact = store.publish(
        kind=_BUNDLE_KIND,
        version=_BUNDLE_VERSION,
        payload={
            "baseline": baseline.manifest(),
            "primary_suite": suites.exact.manifest(),
            "primary_report": primary.manifest(),
            "primary_replay_report": replay.manifest(),
            "diagnostic_suite": suites.portable.manifest(),
            "diagnostic_report": diagnostic.manifest(),
            "logical_manifest_sha256": suites.logical_manifest.sha256,
            "workflow_manifest_sha256": inputs.workflow_manifest_sha256,
            "sealed_baseline_lease_sha256": lease.sha256,
        },
    )
    lookup_key = _sha256(
        {
            "version": "r3b-scripted-baseline-bundle-lookup-v1",
            "workflow_manifest_sha256": inputs.workflow_manifest_sha256,
            "baseline_sha256": baseline.sha256,
            "exact_suite_sha256": suites.exact.sha256,
            "portable_suite_sha256": suites.portable.sha256,
            "sealed_baseline_lease_sha256": lease.sha256,
        }
    )
    _lease_index(store, lease).record(lookup_key, artifact)
    return artifact.artifact_id, bundle


def _secret_payload(
    *,
    authorization_artifact_sha256: str,
    token: str,
) -> bytes:
    return (
        _canonical_bytes(
            {
                "version": "r3b-sealed-baseline-secret-v1",
                "authorization_artifact_sha256": authorization_artifact_sha256,
                "lease_token": token,
            }
        )
        + b"\n"
    )


def _write_private(path: Path, payload: bytes) -> None:
    publish_private_file(path, payload)


def _load_private(path: Path) -> dict[str, str]:
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o077:
        raise ValueError("sealed baseline secret is missing, linked, or not private")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError("sealed baseline secret is malformed") from exc
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "version",
            "authorization_artifact_sha256",
            "lease_token",
        }
        or value["version"] != "r3b-sealed-baseline-secret-v1"
        or any(type(item) is not str for item in value.values())
        or payload != _canonical_bytes(value) + b"\n"
    ):
        raise ValueError("sealed baseline secret schema differs")
    return value


def _acquire_baseline_lease(
    *,
    inputs: CanonicalRunInputs,
    sealed: PublishedSealedAuthorization,
    ledger: SealedTestLedger,
) -> SealedBaselineBatchLease:
    secret_root = ensure_private_directory(inputs.root / "secrets")
    secret = secret_root / "sealed-baseline.json"
    intent = secret_root / "sealed-baseline.intent.json"
    if secret.exists():
        value = _load_private(secret)
        if value["authorization_artifact_sha256"] != sealed.artifact_sha256:
            raise ValueError("sealed baseline secret belongs to another authorization")
        lease = SealedBaselineBatchLease(sealed.authorization, value["lease_token"])
        try:
            lease.assert_running()
        except RuntimeError:
            return lease
        reason = "orphaned sealed baseline execution found after process restart"
        ledger.fail_baseline_batch(lease, reason)
        raise RuntimeError(reason)
    if intent.exists():
        value = _load_private(intent)
        if value["authorization_artifact_sha256"] != sealed.artifact_sha256:
            raise ValueError("sealed baseline intent belongs to another authorization")
        token = value["lease_token"]
        try:
            lease = ledger.resume_baseline_batch(
                sealed.authorization, lease_token=token
            )
        except RuntimeError:
            lease = ledger.claim_baseline_batch(sealed.authorization, lease_token=token)
    else:
        token = secrets.token_hex(32)
        payload = _secret_payload(
            authorization_artifact_sha256=sealed.artifact_sha256,
            token=token,
        )
        _write_private(intent, payload)
        lease = ledger.claim_baseline_batch(sealed.authorization, lease_token=token)
    _write_private(
        secret,
        _secret_payload(
            authorization_artifact_sha256=sealed.artifact_sha256,
            token=lease.lease_token,
        ),
    )
    intent.unlink()
    parent = os.open(secret_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return lease


@dataclass(frozen=True, slots=True)
class SealedBaselineRunResult:
    evidence_artifact_sha256: str
    bundle_artifact_sha256s: tuple[str, ...]
    evidence_sha256s: tuple[str, ...]
    version: str = "r3b-sealed-baseline-run-result-v1"

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "evidence_artifact_sha256": self.evidence_artifact_sha256,
            "bundle_artifact_sha256s": list(self.bundle_artifact_sha256s),
            "evidence_sha256s": list(self.evidence_sha256s),
            "acceptance_eligible": True,
            "transfer_eligible": False,
        }


def _recover_committed_evidence(
    *,
    store: ArtifactStore,
    ledger: SealedTestLedger,
    sealed: PublishedSealedAuthorization,
) -> SealedBaselineRunResult | None:
    """Recover the unique immutable summary authorized by the completed ledger."""

    matches: list[tuple[str, SealedBaselineEvidenceArtifact]] = []
    for artifact_id in store.list():
        envelope = store.load(artifact_id)
        if envelope.kind != _EVIDENCE_KIND or envelope.version != _EVIDENCE_VERSION:
            continue
        try:
            artifact = SealedBaselineEvidenceArtifact.from_manifest(
                envelope.payload,
                sealed_run=sealed.authorization,
            )
        except ValueError:
            continue
        if ledger.verify_completed_baseline_batch(
            sealed.authorization,
            tuple(item.sha256 for item in artifact.evidence),
        ):
            matches.append((artifact_id, artifact))
    if not matches:
        return None
    if len(matches) != 1:
        raise RuntimeError("multiple committed sealed baseline summaries exist")
    artifact_id, artifact = matches[0]
    SealedBaselineEvidenceArtifact.load_committed(
        store,
        artifact_id,
        ledger=ledger,
        sealed_run=sealed.authorization,
    )
    return SealedBaselineRunResult(
        artifact_id,
        (),
        tuple(item.sha256 for item in artifact.evidence),
    )


def run_sealed_baselines(
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
    sealed: PublishedSealedAuthorization,
    *,
    exact_worker_path: str | Path,
    portable_library_path: str | Path,
) -> SealedBaselineRunResult:
    """Run or resume the committed baseline batch and publish durable evidence."""

    if sealed.authorization.plan.sha256 != inputs.plan.sha256:
        raise ValueError("sealed baseline authorization belongs to another run")
    ledger = SealedTestLedger(sealed.authorization.ledger_path)
    recovered = _recover_committed_evidence(
        store=store,
        ledger=ledger,
        sealed=sealed,
    )
    if recovered is not None:
        return recovered
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
    exact_worker = supplied_worker.resolve(strict=True)
    portable_library = supplied_library.resolve(strict=True)
    test_suite = sealed.authorization.test_suite
    if not isinstance(test_suite, EvaluationSuite):
        raise TypeError("sealed test suite is not an evaluation suite")
    suites = PairedEvaluationSuites.build(
        inputs,
        phase="test",
        learner_seed=inputs.plan.test_learner_seeds[0],
        assignment_sha256=test_suite.assignment_sha256,
    )
    if suites.exact != test_suite:
        raise ValueError("reconstructed baseline test suite differs from its seal")
    required = tuple(
        ScriptedBaselineSpec(baseline_id)
        for baseline_id, _ in (
            sealed.authorization.baseline_batch_commitment.required_baselines
        )
    )

    lease = _acquire_baseline_lease(
        inputs=inputs,
        sealed=sealed,
        ledger=ledger,
    )
    try:
        try:
            lease.assert_running()
        except RuntimeError:
            ledger.begin_baseline_batch(lease)
        completed: list[tuple[str, BaselineArtifactBundle]] = []
        for baseline in required:
            loaded = _load_baseline_bundle(
                inputs=inputs,
                store=store,
                suites=suites,
                baseline=baseline,
                lease=lease,
            )
            if loaded is None:
                with IrisuEnv(
                    physics_backend="exact", worker_path=exact_worker
                ) as simulator:
                    primary = _scripted_report(
                        inputs=inputs,
                        store=store,
                        simulator=simulator,
                        suite=suites.exact,
                        baseline=baseline,
                        purpose="primary",
                        lease=lease,
                    )
                with IrisuEnv(
                    physics_backend="exact", worker_path=exact_worker
                ) as simulator:
                    replay = _scripted_report(
                        inputs=inputs,
                        store=store,
                        simulator=simulator,
                        suite=suites.exact,
                        baseline=baseline,
                        purpose="replay",
                        lease=lease,
                    )
                with IrisuEnv(
                    physics_backend="portable", library_path=portable_library
                ) as simulator:
                    diagnostic = _scripted_report(
                        inputs=inputs,
                        store=store,
                        simulator=simulator,
                        suite=suites.portable,
                        baseline=baseline,
                        purpose="diagnostic",
                        lease=lease,
                    )
                loaded = _publish_baseline_bundle(
                    inputs=inputs,
                    store=store,
                    suites=suites,
                    baseline=baseline,
                    primary=primary,
                    replay=replay,
                    diagnostic=diagnostic,
                    lease=lease,
                )
            completed.append(loaded)
        summary = SealedBaselineEvidenceArtifact.from_artifacts(
            sealed.authorization,
            tuple(bundle for _, bundle in completed),
        )
        summary_id = summary.publish(store)
        ledger.complete_baseline_batch(lease, tuple(bundle for _, bundle in completed))
        SealedBaselineEvidenceArtifact.load_committed(
            store,
            summary_id,
            ledger=ledger,
            sealed_run=sealed.authorization,
        )
        return SealedBaselineRunResult(
            summary_id,
            tuple(artifact_id for artifact_id, _ in completed),
            tuple(item.sha256 for item in summary.evidence),
        )
    except Exception as error:
        try:
            lease.assert_running()
        except RuntimeError:
            pass
        else:
            ledger.fail_baseline_batch(lease, f"{type(error).__name__}: {error}")
        raise


__all__ = [
    "SealedBaselineRunResult",
    "run_sealed_baselines",
]
