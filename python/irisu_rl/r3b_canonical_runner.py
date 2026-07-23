"""Fail-closed execution primitives for canonical R3b jobs.

This module deliberately separates deterministic construction and evidence
assembly from process supervision.  A supervisor may restart training and
evaluation shards, but it cannot alter the resolved run, evaluation cells,
checkpoint grid, or sealed-test authority.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from torch import Tensor

from .curriculum import SnapshotLibrary
from .models import RecurrentActorCritic
from .r3b_artifacts import (
    ArtifactLookupIndex,
    ArtifactStore,
    ExactResumeVerificationReceipt,
)
from .r3b_evaluation import (
    CrossBackendEvaluationManifest,
    DeploymentPolicyIdentity,
    EvaluationReport,
    EvaluationSuite,
    LearnedPolicyBackendParityArtifact,
)
from .r3b_evaluation_shards import (
    EvaluationShardPlan,
    EvaluationShardReport,
    evaluate_recurrent_vector_shard,
    merge_evaluation_shards,
    plan_evaluation_shards,
)
from .r3b_experiments import (
    CheckpointEvaluation,
    EngineeringEvidence,
    ExactResumeArtifact,
    LearnerOutcome,
    RawScoreMetricsArtifact,
    R3BExperimentPlan,
    SealedTestLedger,
    SealedTestJobLease,
    SealedLearnerOutcomeReference,
    TrainingCheckpointArtifact,
    TrialJob,
)
from .r3b_operational import (
    JobClaim,
    R3BOperationalConfig,
    R3BWorkflow,
)
from .r3b_local_runner import _source_identity
from .r3b_runner import (
    BuiltTrial,
    verify_exact_resume_continuation,
)
from .r3b_snapshots import (
    SnapshotBundle,
    load_snapshot_bundle,
    pair_snapshot_bundles,
)


_OUTPUT_KIND = "irisu.r3b.canonical-job-output"
_OUTPUT_VERSION = "r3b-canonical-job-output-v1"
_SHARD_KIND = "irisu.r3b.evaluation-shard-package"
_SHARD_VERSION = "r3b-evaluation-shard-package-v1"


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _resume_verifier_identity(
    *,
    workflow_manifest_sha256: str,
    source_identity_sha256: str,
    runner_spec_sha256: str,
) -> str:
    return _sha256(
        {
            "version": "r3b-canonical-resume-verifier-v1",
            "workflow_manifest_sha256": workflow_manifest_sha256,
            "source_identity_sha256": source_identity_sha256,
            "runner_spec_sha256": runner_spec_sha256,
        }
    )


def _is_nonzero_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value != "0" * 64
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
        status.returncode != 0
        or status.stdout
        or revision.returncode != 0
        or len(value) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(
            "canonical execution requires its clean reviewed source revision"
        )
    return value


def _read_resolved_run(root: Path) -> Mapping[str, object]:
    path = root / "resolved-run.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("resolved run manifest is missing or unsafe")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("resolved run manifest is malformed") from error
    expected = {
        "version",
        "workflow",
        "plan",
        "operational_config",
        "snapshot_bundle_path",
        "portable_snapshot_bundle_path",
        "pairing_manifests",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected
        or value.get("version") != "r3b-resolved-run-v1"
        or payload != _canonical_bytes(value) + b"\n"
    ):
        raise ValueError("resolved run manifest is noncanonical or unsupported")
    return value


@dataclass(frozen=True, slots=True)
class CanonicalRunInputs:
    """Fully verified immutable inputs of one initialized canonical run."""

    root: Path
    workflow: R3BWorkflow
    plan: R3BExperimentPlan
    config: R3BOperationalConfig
    exact_bundle: SnapshotBundle
    portable_bundle: SnapshotBundle
    pairings: Mapping[str, CrossBackendEvaluationManifest]
    workflow_manifest_sha256: str

    def __post_init__(self) -> None:
        pairings = dict(self.pairings)
        if (
            not self.root.is_absolute()
            or not isinstance(self.workflow, R3BWorkflow)
            or not isinstance(self.plan, R3BExperimentPlan)
            or not isinstance(self.config, R3BOperationalConfig)
            or self.exact_bundle.runtime_backend != "exact"
            or self.portable_bundle.runtime_backend != "portable"
            or not _is_nonzero_sha256(self.workflow_manifest_sha256)
            or set(pairings) != {"calibration", "validation", "test"}
            or pairings
            != pair_snapshot_bundles(self.portable_bundle, self.exact_bundle)
        ):
            raise ValueError("canonical run inputs disagree")

    @classmethod
    def load(
        cls,
        run_directory: str | Path,
        *,
        exact_simulator: Any,
        portable_simulator: Any,
    ) -> CanonicalRunInputs:
        root = Path(run_directory).resolve(strict=True)
        resolved = _read_resolved_run(root)
        workflow = R3BWorkflow(root / "workflow.sqlite3")
        workflow_manifest = workflow.verify()
        if (
            resolved["workflow"] != workflow_manifest
            or workflow_manifest.get("run_class") != "canonical"
            or workflow_manifest.get("acceptance_eligible") is not True
            or workflow_manifest.get("transfer_eligible") is not False
        ):
            raise ValueError("resolved workflow is not an acceptance-eligible R3 run")
        project_root = Path(__file__).resolve().parents[2]
        if workflow_manifest.get("source_identity_sha256") != _source_identity(
            project_root
        ) or workflow_manifest.get("source_revision") != _clean_source_revision(
            project_root
        ):
            raise ValueError("source tree changed after the canonical run was frozen")
        plan = R3BExperimentPlan.from_mapping(resolved["plan"])
        config = R3BOperationalConfig.from_manifest(resolved["operational_config"])
        if (
            plan.sha256 != workflow_manifest.get("plan_sha256")
            or config.sha256 != workflow_manifest.get("operational_config_sha256")
            or config.primary_backend != "exact"
            or config.transfer_eligible
            or config.collector_max_decisions * config.lanes < plan.ticks_per_update
        ):
            raise ValueError("resolved plan or operational config identity differs")

        exact_path = resolved["snapshot_bundle_path"]
        portable_path = resolved["portable_snapshot_bundle_path"]
        if (
            not isinstance(exact_path, str)
            or not Path(exact_path).is_absolute()
            or not isinstance(portable_path, str)
            or not Path(portable_path).is_absolute()
        ):
            raise ValueError("canonical snapshot paths must be absolute")
        exact_bundle = load_snapshot_bundle(exact_path, exact_simulator)
        portable_bundle = load_snapshot_bundle(portable_path, portable_simulator)
        pairings_value = resolved["pairing_manifests"]
        if not isinstance(pairings_value, Mapping) or set(pairings_value) != {
            "calibration",
            "validation",
            "test",
        }:
            raise ValueError("resolved run lacks all pairing manifests")
        pairings = {
            split: CrossBackendEvaluationManifest.from_manifest(value)
            for split, value in pairings_value.items()
        }
        recomputed = pair_snapshot_bundles(portable_bundle, exact_bundle)
        if (
            exact_bundle.sha256 != workflow_manifest.get("snapshot_bundle_sha256")
            or portable_bundle.sha256
            != workflow_manifest.get("portable_snapshot_bundle_sha256")
            or pairings != recomputed
            or {split: manifest.sha256 for split, manifest in pairings.items()}
            != workflow_manifest.get("pairing_sha256s")
        ):
            raise ValueError("loaded canonical snapshots differ from the frozen run")
        return cls(
            root,
            workflow,
            plan,
            config,
            exact_bundle,
            portable_bundle,
            pairings,
            _sha256(workflow_manifest),
        )


@dataclass(frozen=True, slots=True)
class PairedEvaluationSuites:
    """Exact-primary and portable-diagnostic views of identical logical cells."""

    exact: EvaluationSuite
    portable: EvaluationSuite
    logical_manifest: CrossBackendEvaluationManifest
    exact_library: SnapshotLibrary
    portable_library: SnapshotLibrary
    workflow_manifest_sha256: str

    def __post_init__(self) -> None:
        if (
            self.exact.backend != "exact"
            or self.portable.backend != "portable"
            or self.exact.logical_manifest_sha256 != self.logical_manifest.sha256
            or self.portable.logical_manifest_sha256 != self.logical_manifest.sha256
            or self.exact.library_sha256 != self.exact_library.sha256
            or self.portable.library_sha256 != self.portable_library.sha256
            or not _is_nonzero_sha256(self.workflow_manifest_sha256)
        ):
            raise ValueError("paired evaluation suite identities disagree")
        pairs = self.logical_manifest.pairs
        logical_ids = tuple(pair.logical_cell.sha256 for pair in pairs)
        if (
            self.exact.snapshot_ids != tuple(pair.exact_snapshot_id for pair in pairs)
            or self.portable.snapshot_ids
            != tuple(pair.portable_snapshot_id for pair in pairs)
            or self.exact.recipe_sha256s
            != tuple(pair.exact_recipe_sha256 for pair in pairs)
            or self.portable.recipe_sha256s
            != tuple(pair.portable_recipe_sha256 for pair in pairs)
            or self.exact.logical_cell_ids != logical_ids
            or self.portable.logical_cell_ids != logical_ids
            or (
                self.exact.split,
                self.exact.repetitions,
                self.exact.policy_seed,
                self.exact.max_decisions,
                self.exact.max_simulated_ticks,
                self.exact.assignment_sha256,
                self.exact.action_spec_sha256,
            )
            != (
                self.portable.split,
                self.portable.repetitions,
                self.portable.policy_seed,
                self.portable.max_decisions,
                self.portable.max_simulated_ticks,
                self.portable.assignment_sha256,
                self.portable.action_spec_sha256,
            )
        ):
            raise ValueError("paired evaluation suites do not share logical cells")

    @classmethod
    def build(
        cls,
        inputs: CanonicalRunInputs,
        *,
        phase: str,
        learner_seed: int,
        assignment_sha256: str,
        purpose: str = "final",
    ) -> PairedEvaluationSuites:
        if phase not in {"calibration", "validation", "test"}:
            raise ValueError("unknown evaluation phase")
        if purpose not in {"curve", "final"}:
            raise ValueError("evaluation suite purpose must be curve or final")
        if not _is_nonzero_sha256(assignment_sha256):
            raise ValueError("evaluation assignment identity must be a SHA-256")
        phase_seeds = {
            "calibration": inputs.plan.calibration_learner_seeds,
            "validation": inputs.plan.validation_learner_seeds,
            "test": inputs.plan.test_learner_seeds,
        }[phase]
        if learner_seed not in phase_seeds:
            raise ValueError("learner seed is outside the frozen phase")
        repetitions = {
            "calibration": inputs.config.calibration_repetitions,
            "validation": inputs.config.validation_repetitions,
            "test": inputs.config.test_repetitions,
        }[phase]
        expected_episodes = {
            "calibration": (inputs.config.minimum_calibration_snapshots * repetitions),
            "validation": inputs.plan.validation_episodes_per_policy,
            "test": inputs.plan.test_episodes_per_policy,
        }[phase]
        complete_manifest = inputs.pairings[phase]
        complete_pairs = complete_manifest.pairs
        if len(complete_pairs) * repetitions != expected_episodes:
            raise ValueError("evaluation cell count differs from the frozen plan")
        pairs = (
            complete_pairs[: inputs.config.curve_snapshots]
            if purpose == "curve"
            else complete_pairs
        )
        logical_manifest = CrossBackendEvaluationManifest(tuple(pairs))
        if any(pair.logical_cell.split != phase for pair in pairs):
            raise ValueError("pairing manifest phase differs")
        if purpose == "curve" and len(pairs) != inputs.config.curve_snapshots:
            raise ValueError("curve suite cannot meet its frozen snapshot count")
        action_spec_sha256 = pairs[0].logical_cell.action_spec_sha256
        logical_ids = tuple(pair.logical_cell.sha256 for pair in pairs)

        def suite(backend: str) -> EvaluationSuite:
            bundle = (
                inputs.exact_bundle if backend == "exact" else inputs.portable_bundle
            )
            snapshot_ids = tuple(
                getattr(pair, f"{backend}_snapshot_id") for pair in pairs
            )
            recipe_sha256s = tuple(
                getattr(pair, f"{backend}_recipe_sha256") for pair in pairs
            )
            return EvaluationSuite(
                suite_id=(
                    f"r3b-{phase}-{purpose}-{backend}-"
                    f"workflow-{inputs.workflow_manifest_sha256}"
                ),
                split=phase,
                snapshot_ids=snapshot_ids,
                repetitions=repetitions,
                policy_seed=inputs.plan.evaluation_seed(phase),
                max_decisions=inputs.config.evaluation_max_decisions,
                max_simulated_ticks=(inputs.config.evaluation_max_simulated_ticks),
                runtime_identity_sha256=bundle.runtime_identity_sha256,
                assignment_sha256=assignment_sha256,
                library_sha256=bundle.library.sha256,
                snapshot_store_sha256=bundle.store.sha256,
                action_spec_sha256=action_spec_sha256,
                recipe_sha256s=recipe_sha256s,
                logical_cell_ids=logical_ids,
                backend=backend,
                logical_manifest_sha256=logical_manifest.sha256,
            )

        return cls(
            suite("exact"),
            suite("portable"),
            logical_manifest,
            inputs.exact_bundle.library,
            inputs.portable_bundle.library,
            inputs.workflow_manifest_sha256,
        )


def evaluate_recurrent_policy_sharded(
    *,
    inputs: CanonicalRunInputs,
    simulator: Any,
    store: Any,
    suite: EvaluationSuite,
    model: RecurrentActorCritic,
    encoder: Any,
    kind_mask: Tensor,
    wait_mask: Tensor,
    policy_sha256: str,
    artifact_store: ArtifactStore,
    shard_evaluator: Callable[..., EvaluationShardReport] = (
        evaluate_recurrent_vector_shard
    ),
) -> EvaluationReport:
    """Resume, execute, persist, and merge a deterministic shard partition."""

    if (
        not isinstance(inputs, CanonicalRunInputs)
        or not _is_nonzero_sha256(policy_sha256)
        or not isinstance(artifact_store, ArtifactStore)
    ):
        raise ValueError("canonical evaluation inputs are malformed")
    bundle = inputs.exact_bundle if suite.backend == "exact" else inputs.portable_bundle
    if (
        suite.backend not in {"exact", "portable"}
        or suite.runtime_identity_sha256 != bundle.runtime_identity_sha256
        or suite.library_sha256 != bundle.library.sha256
        or suite.snapshot_store_sha256 != bundle.store.sha256
    ):
        raise ValueError("evaluation suite is foreign to the canonical bundles")
    evaluator_sha256 = _sha256(
        {
            "version": "r3b-canonical-evaluator-v1",
            "workflow_manifest_sha256": inputs.workflow_manifest_sha256,
            "action_spec_sha256": suite.action_spec_sha256,
            "algorithm": "recurrent-semantic-fixed-cell-vector-v1",
        }
    )
    worker_identity_sha256 = _sha256(
        {
            "version": "r3b-evaluation-worker-topology-v1",
            "workflow_manifest_sha256": inputs.workflow_manifest_sha256,
            "evaluator_sha256": evaluator_sha256,
            "runtime_identity_sha256": suite.runtime_identity_sha256,
            "lanes": inputs.config.lanes,
            "workers": inputs.config.workers,
        }
    )
    deployment_identity = DeploymentPolicyIdentity.from_components(
        model, encoder, kind_mask, wait_mask
    )
    if deployment_identity.sha256 != policy_sha256:
        raise ValueError("evaluation policy identity differs from the model")
    plans = plan_evaluation_shards(suite, inputs.config.evaluation_shards)
    index = ArtifactLookupIndex(artifact_store.root.parent / "evaluation-index.sqlite3")
    reports: list[EvaluationShardReport] = []
    for shard in plans:
        execution_identity_sha256 = _sha256(
            {
                "version": "r3b-evaluation-shard-execution-v1",
                "suite_sha256": suite.sha256,
                "shard_plan_sha256": shard.sha256,
                "evaluator_sha256": evaluator_sha256,
                "policy_sha256": policy_sha256,
                "worker_identity_sha256": worker_identity_sha256,
            }
        )
        lookup_key = _sha256(
            {
                "version": "r3b-evaluation-shard-lookup-v1",
                "execution_identity_sha256": execution_identity_sha256,
            }
        )
        envelope = index.lookup(
            lookup_key,
            artifact_store,
            expected_kind=_SHARD_KIND,
            expected_version=_SHARD_VERSION,
        )
        if envelope is None:
            report = shard_evaluator(
                simulator,
                store,
                suite,
                model,
                encoder,
                kind_mask,
                wait_mask,
                shard,
                evaluator_sha256=evaluator_sha256,
                expected_assignment_sha256=suite.assignment_sha256,
                execution_identity_sha256=execution_identity_sha256,
            )
            if (
                report.shard != shard
                or report.report.suite_sha256 != suite.sha256
                or report.report.policy_sha256 != policy_sha256
                or report.report.evaluator_sha256 != evaluator_sha256
                or report.report.backend_identity_sha256
                != suite.runtime_identity_sha256
                or report.report.execution_identity_sha256 != execution_identity_sha256
            ):
                raise ValueError("fresh evaluation shard identities differ")
            envelope = artifact_store.publish(
                kind=_SHARD_KIND,
                version=_SHARD_VERSION,
                payload={
                    "suite": suite.manifest(),
                    "shard_plan": shard.manifest(),
                    "report": report.report.manifest(),
                    "shard_report": report.manifest(),
                    "worker_identity_sha256": worker_identity_sha256,
                    "execution_identity_sha256": execution_identity_sha256,
                    "policy_sha256": policy_sha256,
                },
            )
            index.record(lookup_key, envelope)
        payload = envelope.payload
        if not isinstance(payload, dict) or set(payload) != {
            "suite",
            "shard_plan",
            "report",
            "shard_report",
            "worker_identity_sha256",
            "execution_identity_sha256",
            "policy_sha256",
        }:
            raise ValueError("persisted evaluation shard package schema differs")
        stored_suite = EvaluationSuite.from_manifest(payload["suite"])
        stored_shard = EvaluationShardPlan.from_manifest(payload["shard_plan"])
        if (
            stored_suite != suite
            or stored_shard != shard
            or payload["worker_identity_sha256"] != worker_identity_sha256
            or payload["execution_identity_sha256"] != execution_identity_sha256
            or payload["policy_sha256"] != policy_sha256
        ):
            raise ValueError("indexed evaluation shard identity differs")
        stored_report = EvaluationReport.from_manifest(payload["report"], suite=suite)
        report = EvaluationShardReport.from_manifest(
            payload["shard_report"],
            shard=shard,
            report=stored_report,
        )
        if (
            stored_report.execution_identity_sha256 != execution_identity_sha256
            or stored_report.policy_sha256 != policy_sha256
            or stored_report.evaluator_sha256 != evaluator_sha256
            or stored_report.backend_identity_sha256 != suite.runtime_identity_sha256
        ):
            raise ValueError("indexed evaluation report identity differs")
        reports.append(report)
    return merge_evaluation_shards(suite, tuple(reports))


def audit_penultimate_checkpoint(
    *,
    job: TrialJob,
    checkpoint: TrainingCheckpointArtifact,
    checkpoint_directory: str | Path,
    generation: str,
    checkpoint_identity: Mapping[str, object],
    source: Any,
    restored_factory: Callable[[], Any],
    plan: R3BExperimentPlan,
    sealed_job_lease: SealedTestJobLease | None = None,
) -> ExactResumeArtifact:
    """Prove an independent exact restore before the final grid interval."""

    if not isinstance(plan, R3BExperimentPlan):
        raise TypeError("resume audit requires the frozen experiment plan")
    expected_update = job.budget_updates - plan.checkpoint_interval_updates
    if (
        job.plan_sha256 != plan.sha256
        or job.budget_updates % plan.checkpoint_interval_updates
        or checkpoint.completed_updates != expected_update
        or checkpoint.target_simulated_ticks != expected_update * plan.ticks_per_update
        or checkpoint.job_sha256 != job.sha256
    ):
        raise ValueError("resume audit is not at the penultimate checkpoint")
    if job.phase == "test":
        if (
            not isinstance(sealed_job_lease, SealedTestJobLease)
            or sealed_job_lease.job != job
        ):
            raise ValueError("sealed test resume audit requires its job lease")
        sealed_job_lease.assert_running()
    elif sealed_job_lease is not None:
        raise ValueError("non-test resume audit must not carry a sealed lease")
    return verify_exact_resume_continuation(
        trial_manifest_sha256=checkpoint.trial_manifest_sha256,
        checkpoint=checkpoint,
        checkpoint_directory=checkpoint_directory,
        generation=generation,
        checkpoint_identity=checkpoint_identity,
        source=source,
        restored_factory=restored_factory,
    )


@dataclass(frozen=True, slots=True)
class PublishedCanonicalOutcome:
    output_artifact_sha256: str
    metrics_artifact_sha256: str
    parity_artifact_sha256: str
    resume_receipt_sha256: str
    engineering_evidence_sha256: str
    outcome_sha256: str
    version: str = "r3b-published-canonical-outcome-v1"

    def __post_init__(self) -> None:
        if any(
            not _is_nonzero_sha256(value)
            for value in (
                self.output_artifact_sha256,
                self.metrics_artifact_sha256,
                self.parity_artifact_sha256,
                self.resume_receipt_sha256,
                self.engineering_evidence_sha256,
                self.outcome_sha256,
            )
        ):
            raise ValueError("published canonical outcome identities are malformed")


def load_published_canonical_outcome(
    *,
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
    output_artifact_sha256: str,
) -> tuple[LearnerOutcome, PublishedCanonicalOutcome]:
    """Strictly reconstruct one result from immutable, dependency-complete packages."""

    current = inputs.workflow.verify()
    if (
        not isinstance(store, ArtifactStore)
        or store.root.resolve() != (inputs.root / "artifacts").resolve()
        or _sha256(current) != inputs.workflow_manifest_sha256
        or _source_identity(Path(__file__).resolve().parents[2])
        != current.get("source_identity_sha256")
    ):
        raise ValueError("canonical result loader inputs differ from the frozen run")
    output = store.load(
        output_artifact_sha256,
        expected_kind=_OUTPUT_KIND,
        expected_version=_OUTPUT_VERSION,
    )
    payload = output.payload
    if (
        not isinstance(payload, dict)
        or set(payload)
        != {
            "job_sha256",
            "learner_outcome_sha256",
            "learner_outcome",
            "metrics_artifact_sha256",
            "parity_artifact_sha256",
            "resume_receipt_sha256",
            "engineering_evidence_sha256",
            "workflow_manifest_sha256",
            "acceptance_eligible",
            "transfer_eligible",
        }
        or (
            payload["workflow_manifest_sha256"] != inputs.workflow_manifest_sha256
            or payload["acceptance_eligible"] is not True
            or payload["transfer_eligible"] is not False
        )
    ):
        raise ValueError("canonical output package schema or workflow differs")

    metrics_package = store.load(
        payload["metrics_artifact_sha256"],
        expected_kind="irisu.r3b.raw-score-metrics-package",
        expected_version="r3b-raw-score-metrics-package-v1",
    ).payload
    if (
        not isinstance(metrics_package, dict)
        or set(metrics_package)
        != {
            "curve_suite",
            "final_suite",
            "reports",
            "final_report",
            "metrics",
            "workflow_manifest_sha256",
        }
        or metrics_package["workflow_manifest_sha256"]
        != inputs.workflow_manifest_sha256
    ):
        raise ValueError("canonical metrics package schema differs")
    curve_exact_suite = EvaluationSuite.from_manifest(metrics_package["curve_suite"])
    exact_suite = EvaluationSuite.from_manifest(metrics_package["final_suite"])
    if not isinstance(metrics_package["reports"], list):
        raise ValueError("canonical metrics reports must be an array")
    exact_reports = tuple(
        EvaluationReport.from_manifest(value, suite=curve_exact_suite)
        for value in metrics_package["reports"]
    )
    exact_final_report = EvaluationReport.from_manifest(
        metrics_package["final_report"], suite=exact_suite
    )
    metrics = RawScoreMetricsArtifact.from_manifest(
        metrics_package["metrics"],
        curve_suite=curve_exact_suite,
        final_suite=exact_suite,
        reports=exact_reports,
        final_report=exact_final_report,
    )

    parity_package = store.load(
        payload["parity_artifact_sha256"],
        expected_kind="irisu.r3b.learned-policy-backend-parity-package",
        expected_version="r3b-learned-policy-backend-parity-package-v1",
    ).payload
    if (
        not isinstance(parity_package, dict)
        or set(parity_package)
        != {
            "portable_suite",
            "portable_report",
            "exact_suite",
            "exact_report",
            "logical_manifest",
            "portable_library",
            "exact_library",
            "parity",
            "workflow_manifest_sha256",
        }
        or parity_package["workflow_manifest_sha256"] != inputs.workflow_manifest_sha256
    ):
        raise ValueError("canonical parity package schema differs")
    portable_suite = EvaluationSuite.from_manifest(parity_package["portable_suite"])
    parity_exact_suite = EvaluationSuite.from_manifest(parity_package["exact_suite"])
    if parity_exact_suite != exact_suite:
        raise ValueError("canonical metrics and parity suites differ")
    portable_report = EvaluationReport.from_manifest(
        parity_package["portable_report"], suite=portable_suite
    )
    exact_report = EvaluationReport.from_manifest(
        parity_package["exact_report"], suite=exact_suite
    )
    if exact_report != metrics.final_report:
        raise ValueError("canonical metrics and parity final reports differ")
    logical_manifest = CrossBackendEvaluationManifest.from_manifest(
        parity_package["logical_manifest"]
    )
    portable_library = SnapshotLibrary.from_manifest(parity_package["portable_library"])
    exact_library = SnapshotLibrary.from_manifest(parity_package["exact_library"])
    parity = LearnedPolicyBackendParityArtifact.from_manifest(
        parity_package["parity"],
        portable_suite=portable_suite,
        portable_report=portable_report,
        exact_suite=exact_suite,
        exact_report=exact_report,
        logical_manifest=logical_manifest,
        portable_library=portable_library,
        exact_library=exact_library,
    )

    evidence_package = store.load(
        payload["engineering_evidence_sha256"],
        expected_kind="irisu.r3b.engineering-evidence-package",
        expected_version="r3b-engineering-evidence-package-v1",
    ).payload
    if (
        not isinstance(evidence_package, dict)
        or set(evidence_package)
        != {
            "engineering_evidence",
            "metrics_artifact_sha256",
            "parity_artifact_sha256",
            "resume_receipt_sha256",
            "workflow_manifest_sha256",
        }
        or (
            evidence_package["metrics_artifact_sha256"]
            != payload["metrics_artifact_sha256"]
            or evidence_package["parity_artifact_sha256"]
            != payload["parity_artifact_sha256"]
            or evidence_package["resume_receipt_sha256"]
            != payload["resume_receipt_sha256"]
            or evidence_package["workflow_manifest_sha256"]
            != inputs.workflow_manifest_sha256
        )
    ):
        raise ValueError("canonical engineering package references differ")
    evidence_manifest = evidence_package["engineering_evidence"]
    if not isinstance(evidence_manifest, dict):
        raise ValueError("canonical engineering evidence is malformed")
    job_sha256 = evidence_manifest.get("job_sha256")
    if not isinstance(job_sha256, str) or not inputs.workflow.verify_resume_audit(
        job_sha256, payload["resume_receipt_sha256"]
    ):
        raise ValueError("canonical resume receipt lacks workflow authority")
    exact_resume = ExactResumeVerificationReceipt.load_verified_artifact(
        store,
        payload["resume_receipt_sha256"],
        expected_verifier_identity_sha256=_resume_verifier_identity(
            workflow_manifest_sha256=inputs.workflow_manifest_sha256,
            source_identity_sha256=str(current["source_identity_sha256"]),
            runner_spec_sha256=str(evidence_manifest.get("runner_spec_sha256")),
        ),
        expected_build_identity_sha256=inputs.exact_bundle.runtime_identity_sha256,
    )
    evidence = EngineeringEvidence.from_manifest(
        evidence_manifest,
        checkpoint_resume_artifact=exact_resume,
        exact_backend_parity_artifact=parity,
    )
    outcome = LearnerOutcome.from_manifest(
        payload["learner_outcome"],
        metrics_artifact=metrics,
        engineering_evidence=evidence,
    )
    if (
        outcome.sha256 != payload["learner_outcome_sha256"]
        or evidence.job_sha256 != payload["job_sha256"]
    ):
        raise ValueError("canonical learner outcome identity differs")
    job = TrialJob.from_manifest(
        inputs.workflow.job_record(evidence.job_sha256)["manifest"]
    )
    expected_suites = PairedEvaluationSuites.build(
        inputs,
        phase=job.phase,
        learner_seed=job.learner_seed,
        assignment_sha256=outcome.assignment_sha256,
    )
    expected_curve_suites = PairedEvaluationSuites.build(
        inputs,
        phase=job.phase,
        learner_seed=job.learner_seed,
        assignment_sha256=outcome.assignment_sha256,
        purpose="curve",
    )
    if (
        exact_suite != expected_suites.exact
        or portable_suite != expected_suites.portable
        or curve_exact_suite != expected_curve_suites.exact
    ):
        raise ValueError("canonical learner outcome suites differ from the run")
    published = PublishedCanonicalOutcome(
        output.artifact_id,
        payload["metrics_artifact_sha256"],
        payload["parity_artifact_sha256"],
        payload["resume_receipt_sha256"],
        payload["engineering_evidence_sha256"],
        outcome.sha256,
    )
    return outcome, published


def assemble_and_publish_outcome(
    *,
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
    built: BuiltTrial,
    job: TrialJob,
    suites: PairedEvaluationSuites,
    curve_suites: PairedEvaluationSuites,
    checkpoint_evaluations: tuple[CheckpointEvaluation, ...],
    exact_final_report: EvaluationReport,
    portable_final_report: EvaluationReport,
    deployment_identity: DeploymentPolicyIdentity,
    exact_resume_artifact: ExactResumeArtifact,
    checkpoint_interval_updates: int,
    plan: R3BExperimentPlan,
    workflow_claim: JobClaim,
    sealed_job_lease: SealedTestJobLease | None = None,
) -> tuple[LearnerOutcome, PublishedCanonicalOutcome]:
    """Assemble typed evidence and atomically publish dependency-complete packages."""

    if (
        not isinstance(inputs, CanonicalRunInputs)
        or not isinstance(built, BuiltTrial)
        or not isinstance(job, TrialJob)
        or not isinstance(workflow_claim, JobClaim)
        or workflow_claim.job_sha256 != job.sha256
        or workflow_claim.phase != job.phase
    ):
        raise TypeError(
            "canonical outcome assembly requires verified inputs, trial, and job"
        )
    current_workflow_manifest = inputs.workflow.verify()
    expected_suites = PairedEvaluationSuites.build(
        inputs,
        phase=job.phase,
        learner_seed=job.learner_seed,
        assignment_sha256=built.manifest.assignment_sha256,
    )
    expected_curve_suites = PairedEvaluationSuites.build(
        inputs,
        phase=job.phase,
        learner_seed=job.learner_seed,
        assignment_sha256=built.manifest.assignment_sha256,
        purpose="curve",
    )
    if (
        _sha256(current_workflow_manifest) != inputs.workflow_manifest_sha256
        or current_workflow_manifest.get("acceptance_eligible") is not True
        or current_workflow_manifest.get("run_class") != "canonical"
        or _source_identity(Path(__file__).resolve().parents[2])
        != current_workflow_manifest.get("source_identity_sha256")
        or not isinstance(store, ArtifactStore)
        or store.root.resolve() != (inputs.root / "artifacts").resolve()
        or suites != expected_suites
        or curve_suites != expected_curve_suites
        or suites.workflow_manifest_sha256 != inputs.workflow_manifest_sha256
        or curve_suites.workflow_manifest_sha256 != inputs.workflow_manifest_sha256
        or built.manifest.runtime_identity_sha256
        != inputs.exact_bundle.runtime_identity_sha256
        or built.manifest.library_sha256 != inputs.exact_bundle.library.sha256
        or built.manifest.snapshot_store_sha256 != inputs.exact_bundle.store.sha256
        or built.manifest.job_sha256 != job.sha256
        or built.manifest.phase != job.phase
        or not checkpoint_evaluations
        or not isinstance(exact_final_report, EvaluationReport)
        or exact_final_report.suite_sha256 != suites.exact.sha256
        or exact_final_report.policy_sha256 != checkpoint_evaluations[-1].policy_sha256
        or not isinstance(plan, R3BExperimentPlan)
        or plan.sha256 != job.plan_sha256
        or built.manifest.plan_sha256 != plan.sha256
        or checkpoint_interval_updates <= 0
        or checkpoint_interval_updates != plan.checkpoint_interval_updates
        or job.budget_updates % checkpoint_interval_updates
    ):
        raise ValueError("canonical outcome assembly inputs disagree")
    expected_grid = tuple(range(0, job.budget_updates + 1, checkpoint_interval_updates))
    actual_grid = tuple(value.completed_updates for value in checkpoint_evaluations)
    if actual_grid != expected_grid:
        raise ValueError("checkpoint evaluations do not cover the frozen grid")
    expected_target_ticks = tuple(
        update * plan.ticks_per_update for update in expected_grid
    )
    actual_target_ticks = tuple(
        value.target_simulated_ticks for value in checkpoint_evaluations
    )
    if actual_target_ticks != expected_target_ticks:
        raise ValueError("checkpoint evaluations do not cover the nominal tick grid")
    if job.phase == "test":
        if (
            not isinstance(sealed_job_lease, SealedTestJobLease)
            or sealed_job_lease.job != job
        ):
            raise ValueError("sealed test outcome requires its one-time job lease")
        sealed_job_lease.assert_running()
    elif sealed_job_lease is not None:
        raise ValueError("non-test outcome must not carry a sealed lease")

    from .r3b_experiments import RawScoreMetricsArtifact

    metrics = RawScoreMetricsArtifact(
        job.learner_seed,
        curve_suites.exact,
        suites.exact,
        checkpoint_evaluations,
        exact_final_report,
    )
    parity = LearnedPolicyBackendParityArtifact(
        suites.portable,
        portable_final_report,
        suites.exact,
        metrics.final_report,
        suites.logical_manifest,
        suites.portable_library,
        suites.exact_library,
    )
    evidence = built.engineering_evidence(
        metrics_artifact=metrics,
        deployment_identity=deployment_identity,
        checkpoint_resume_artifact=exact_resume_artifact,
        exact_backend_parity_artifact=parity,
    )
    outcome = LearnerOutcome(
        learner_seed=job.learner_seed,
        raw_score_auc=metrics.raw_score_auc,
        final_mean_raw_score=metrics.final_mean_raw_score,
        p10_raw_score=metrics.p10_raw_score,
        initial_model_sha256=built.manifest.initial_model_sha256,
        assignment_sha256=built.manifest.assignment_sha256,
        seed_plan_sha256=built.manifest.seed_plan_sha256,
        metrics_artifact=metrics,
        engineering_evidence=evidence,
    )

    report_manifests = [
        checkpoint.report.manifest() for checkpoint in checkpoint_evaluations
    ]
    metrics_envelope = store.publish(
        kind="irisu.r3b.raw-score-metrics-package",
        version="r3b-raw-score-metrics-package-v1",
        payload={
            "curve_suite": curve_suites.exact.manifest(),
            "final_suite": suites.exact.manifest(),
            "reports": report_manifests,
            "final_report": exact_final_report.manifest(),
            "metrics": metrics.manifest(),
            "workflow_manifest_sha256": suites.workflow_manifest_sha256,
        },
    )
    parity_envelope = store.publish(
        kind="irisu.r3b.learned-policy-backend-parity-package",
        version="r3b-learned-policy-backend-parity-package-v1",
        payload={
            "portable_suite": suites.portable.manifest(),
            "portable_report": portable_final_report.manifest(),
            "exact_suite": suites.exact.manifest(),
            "exact_report": metrics.final_report.manifest(),
            "logical_manifest": suites.logical_manifest.manifest(),
            "portable_library": suites.portable_library.manifest(),
            "exact_library": suites.exact_library.manifest(),
            "parity": parity.manifest(),
            "workflow_manifest_sha256": suites.workflow_manifest_sha256,
        },
    )
    verifier_identity_sha256 = _resume_verifier_identity(
        workflow_manifest_sha256=inputs.workflow_manifest_sha256,
        source_identity_sha256=str(current_workflow_manifest["source_identity_sha256"]),
        runner_spec_sha256=built.manifest.runner_spec_sha256,
    )
    build_identity_sha256 = inputs.exact_bundle.runtime_identity_sha256
    resume_receipt_sha256 = inputs.workflow.publish_resume_audit(
        workflow_claim,
        exact_resume_artifact,
        store=store,
        verifier_identity_sha256=verifier_identity_sha256,
        build_identity_sha256=build_identity_sha256,
    )
    evidence_envelope = store.publish(
        kind="irisu.r3b.engineering-evidence-package",
        version="r3b-engineering-evidence-package-v1",
        payload={
            "engineering_evidence": evidence.manifest(),
            "metrics_artifact_sha256": metrics_envelope.artifact_id,
            "parity_artifact_sha256": parity_envelope.artifact_id,
            "resume_receipt_sha256": resume_receipt_sha256,
            "workflow_manifest_sha256": suites.workflow_manifest_sha256,
        },
    )
    output = store.publish(
        kind=_OUTPUT_KIND,
        version=_OUTPUT_VERSION,
        payload={
            "job_sha256": job.sha256,
            "learner_outcome_sha256": outcome.sha256,
            "learner_outcome": outcome.manifest(),
            "metrics_artifact_sha256": metrics_envelope.artifact_id,
            "parity_artifact_sha256": parity_envelope.artifact_id,
            "resume_receipt_sha256": resume_receipt_sha256,
            "engineering_evidence_sha256": evidence_envelope.artifact_id,
            "workflow_manifest_sha256": suites.workflow_manifest_sha256,
            "acceptance_eligible": True,
            "transfer_eligible": False,
        },
    )
    published = PublishedCanonicalOutcome(
        output.artifact_id,
        metrics_envelope.artifact_id,
        parity_envelope.artifact_id,
        resume_receipt_sha256,
        evidence_envelope.artifact_id,
        outcome.sha256,
    )
    return outcome, published


def complete_nonsealed_workflow_job(
    *,
    workflow: R3BWorkflow,
    claim: JobClaim,
    job: TrialJob,
    published: PublishedCanonicalOutcome,
    inputs: CanonicalRunInputs | None = None,
    store: ArtifactStore | None = None,
) -> None:
    """Commit an assembled calibration/validation output to the workflow.

    Sealed-test completion additionally mutates the one-shot test ledger.  It is
    intentionally not approximated by two non-atomic writes here.
    """

    if (
        not isinstance(workflow, R3BWorkflow)
        or not isinstance(claim, JobClaim)
        or not isinstance(job, TrialJob)
        or not isinstance(published, PublishedCanonicalOutcome)
        or claim.job_sha256 != job.sha256
        or claim.phase != job.phase
    ):
        raise ValueError("workflow completion inputs disagree")
    if job.phase == "test" or job.sealed:
        raise RuntimeError(
            "sealed-test completion requires a ledger/workflow transaction coordinator"
        )
    if (
        not isinstance(inputs, CanonicalRunInputs)
        or not isinstance(store, ArtifactStore)
        or inputs.workflow.path != workflow.path
    ):
        raise ValueError("workflow completion requires its verified canonical store")
    outcome, verified = load_published_canonical_outcome(
        inputs=inputs,
        store=store,
        output_artifact_sha256=published.output_artifact_sha256,
    )
    if verified != published or outcome.engineering_evidence.job_sha256 != job.sha256:
        raise ValueError("published outcome is foreign to the completed job")
    workflow.complete(claim, published.output_artifact_sha256)


def complete_sealed_workflow_job(
    *,
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
    workflow: R3BWorkflow,
    claim: JobClaim,
    ledger: SealedTestLedger,
    lease: SealedTestJobLease,
    outcome: LearnerOutcome,
    published: PublishedCanonicalOutcome,
) -> str:
    """Commit ledger first; a crash leaves a safely reconcilable workflow row."""

    if (
        not isinstance(inputs, CanonicalRunInputs)
        or not isinstance(store, ArtifactStore)
        or inputs.workflow.path != workflow.path
        or not isinstance(claim, JobClaim)
        or claim.job_sha256 != lease.job.sha256
        or claim.phase != "test"
        or outcome.sha256 != published.outcome_sha256
    ):
        raise ValueError("sealed workflow completion inputs disagree")
    loaded, verified = load_published_canonical_outcome(
        inputs=inputs,
        store=store,
        output_artifact_sha256=published.output_artifact_sha256,
    )
    if (
        loaded != outcome
        or verified != published
        or loaded.engineering_evidence.job_sha256 != lease.job.sha256
    ):
        raise ValueError("sealed published outcome is foreign to the job")
    reference_artifact_sha256 = SealedLearnerOutcomeReference.capture(
        lease.sealed_run,
        lease.job,
        loaded,
        published.output_artifact_sha256,
    ).publish(store)
    if ledger.verify_completed_job(lease.sealed_run, lease.job, loaded.sha256):
        workflow.reconcile_sealed_completion(
            ledger=ledger,
            sealed_run=lease.sealed_run,
            job=lease.job,
            outcome_sha256=loaded.sha256,
            output_sha256=published.output_artifact_sha256,
        )
    else:
        ledger.complete_job(lease, loaded)
        workflow.complete(claim, published.output_artifact_sha256)
    return reference_artifact_sha256


def reconcile_sealed_workflow_job(
    *,
    inputs: CanonicalRunInputs,
    store: ArtifactStore,
    workflow: R3BWorkflow,
    claim: JobClaim,
    ledger: SealedTestLedger,
    sealed_run: object,
    job: TrialJob,
    outcome: LearnerOutcome,
    published: PublishedCanonicalOutcome,
) -> None:
    """Finish the workflow half after a crash following authoritative ledger commit."""

    from .r3b_experiments import SealedTestRunAuthorization

    if (
        not isinstance(inputs, CanonicalRunInputs)
        or not isinstance(store, ArtifactStore)
        or inputs.workflow.path != workflow.path
        or not isinstance(sealed_run, SealedTestRunAuthorization)
        or outcome.sha256 != published.outcome_sha256
        or not ledger.verify_completed_job(sealed_run, job, outcome.sha256)
    ):
        raise ValueError("sealed workflow reconciliation is not ledger-authorized")
    loaded, verified = load_published_canonical_outcome(
        inputs=inputs,
        store=store,
        output_artifact_sha256=published.output_artifact_sha256,
    )
    if (
        loaded != outcome
        or verified != published
        or loaded.engineering_evidence.job_sha256 != job.sha256
    ):
        raise ValueError("sealed reconciliation artifact is foreign to the job")
    workflow.reconcile_sealed_completion(
        ledger=ledger,
        sealed_run=sealed_run,
        job=job,
        outcome_sha256=loaded.sha256,
        output_sha256=published.output_artifact_sha256,
    )


__all__ = [
    "CanonicalRunInputs",
    "PairedEvaluationSuites",
    "PublishedCanonicalOutcome",
    "assemble_and_publish_outcome",
    "audit_penultimate_checkpoint",
    "complete_nonsealed_workflow_job",
    "complete_sealed_workflow_job",
    "evaluate_recurrent_policy_sharded",
    "load_published_canonical_outcome",
    "reconcile_sealed_workflow_job",
]
