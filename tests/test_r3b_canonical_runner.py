from __future__ import annotations

import tempfile
import unittest
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from irisu_rl.curriculum import (
    SnapshotBlobStore,
    SnapshotLibrary,
    SnapshotRecipe,
)
from irisu_rl.actions import ActionSpec
from irisu_rl.r3b_artifacts import ArtifactStore
from irisu_rl.r3b_canonical_runner import (
    CanonicalRunInputs,
    PairedEvaluationSuites,
    PublishedCanonicalOutcome,
    complete_nonsealed_workflow_job,
    evaluate_recurrent_policy_sharded,
    load_published_canonical_outcome,
)
from irisu_rl.r3b_evaluation import (
    CrossBackendCellPair,
    CrossBackendEvaluationManifest,
    EpisodeMetrics,
    EvaluationReport,
    EvaluationSuite,
    LogicalEvaluationCell,
)
from irisu_rl.r3b_evaluation_shards import (
    EvaluationShardPlan,
    EvaluationShardReport,
)
from irisu_rl.r3b_experiments import (
    CandidateArm,
    R3BExperimentPlan,
    TrialJob,
    load_plan,
)
from irisu_rl.r3b_operational import (
    JobClaim,
    R3BOperationalConfig,
    R3BWorkflow,
)
from irisu_rl.r3b_snapshots import (
    SnapshotBundle,
    SnapshotIntent,
    SnapshotSourceManifest,
)


def _hash(character: str) -> str:
    return character * 64


def _sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _recipe(backend: str, split: str, index: int, runtime: str) -> SnapshotRecipe:
    split_offset = {"calibration": 100, "validation": 200, "test": 300}[split]
    return SnapshotRecipe(
        snapshot_id=f"{backend}-{split}-{index}",
        stage_id="full-game",
        split=split,
        scenario_family=f"{split}-family",
        environment_pool="full-game",
        reset_seed=split_offset + index,
        config_sha256=_hash("1"),
        config_hash=7,
        action_spec_sha256=ActionSpec().sha256,
        semantic_actions_hex=(),
        expected_tick=0,
        expected_score=0,
        expected_state_hash=split_offset + index,
        snapshot_sha256=_hash("3" if backend == "exact" else "4"),
        runtime_identity_sha256=runtime,
        generator_version="r3b-full-game-generator-v1",
    )


def _bundle(backend: str, count: int) -> SnapshotBundle:
    runtime = _hash("5" if backend == "exact" else "6")
    recipes = tuple(
        _recipe(backend, split, index, runtime)
        for split in ("calibration", "validation", "test")
        for index in range(count)
    )
    blobs = {
        recipe.snapshot_id: recipe.snapshot_id.encode().ljust(64, b"\0")
        for recipe in recipes
    }
    # SnapshotBlobStore verifies blob hashes, so use recipes rebuilt with real hashes.
    recipes = tuple(
        SnapshotRecipe(
            **{
                **recipe.manifest(),
                "semantic_actions_hex": tuple(recipe.semantic_actions_hex),
                "snapshot_sha256": __import__("hashlib")
                .sha256(blobs[recipe.snapshot_id])
                .hexdigest(),
            }
        )
        for recipe in recipes
    )
    library = SnapshotLibrary(recipes)
    store = SnapshotBlobStore(library, blobs)
    source = SnapshotSourceManifest(
        "test-source",
        ActionSpec().sha256,
        tuple(
            SnapshotIntent(
                value.snapshot_id,
                value.stage_id,
                value.split,
                value.scenario_family,
                value.environment_pool,
                value.reset_seed,
                value.semantic_actions_hex,
            )
            for value in recipes
        ),
    )
    return SnapshotBundle(source, library, store, backend, runtime)


def _plan(count: int) -> R3BExperimentPlan:
    config = Path("configs/rl/experiments/r3b-completion-v1.toml")
    plan = load_plan(config)
    object.__setattr__(plan, "validation_episodes_per_policy", count)
    object.__setattr__(plan, "test_episodes_per_policy", count)
    return plan


def _config() -> R3BOperationalConfig:
    return R3BOperationalConfig(
        lanes=2,
        workers=2,
        torch_threads=1,
        max_consecutive_skips=2,
        collector_max_decisions=2,
        collector_lambda_tick=0.9,
        model_global_hidden=2,
        model_body_hidden=2,
        model_fused_hidden=2,
        model_recurrent_hidden=2,
        model_recurrent_layers=1,
        ppo_epochs=1,
        ppo_lane_minibatch_size=1,
        ppo_clip_ratio=0.2,
        ppo_value_clip=0.2,
        ppo_value_coefficient=0.5,
        ppo_entropy_coefficient=0.0,
        ppo_max_gradient_norm=1.0,
        ppo_target_kl=0.1,
        curve_snapshots=2,
        evaluation_shards=2,
        calibration_repetitions=1,
        validation_repetitions=1,
        test_repetitions=1,
        evaluation_max_decisions=10,
        evaluation_max_simulated_ticks=10,
        snapshot_generator_version="r3b-full-game-generator-v1",
        minimum_train_snapshots=1,
        minimum_calibration_snapshots=2,
        minimum_validation_snapshots=2,
        minimum_test_snapshots=2,
        checkpoint_retention="all_planned_boundaries",
        primary_backend="exact",
        transfer_eligible=False,
    )


def _inputs(count: int = 2) -> CanonicalRunInputs:
    exact = _bundle("exact", count)
    portable = _bundle("portable", count)
    pairings: dict[str, CrossBackendEvaluationManifest] = {}
    for split in ("calibration", "validation", "test"):
        exact_recipes = {
            LogicalEvaluationCell.from_recipe(value).sha256: value
            for value in exact.library.recipes
            if value.split == split
        }
        portable_recipes = {
            LogicalEvaluationCell.from_recipe(value).sha256: value
            for value in portable.library.recipes
            if value.split == split
        }
        pairings[split] = CrossBackendEvaluationManifest(
            tuple(
                CrossBackendCellPair.from_recipes(
                    portable_recipes[key], exact_recipes[key]
                )
                for key in sorted(exact_recipes)
            )
        )
    return CanonicalRunInputs(
        Path("/tmp/r3-canonical"),
        R3BWorkflow("/tmp/r3-canonical-workflow.sqlite3"),
        _plan(count),
        _config(),
        exact,
        portable,
        pairings,
        _hash("8"),
    )


class CanonicalRunnerTests(unittest.TestCase):
    def test_builds_exact_primary_and_portable_diagnostic_suites(self) -> None:
        inputs = _inputs()
        suites = PairedEvaluationSuites.build(
            inputs,
            phase="validation",
            learner_seed=inputs.plan.validation_learner_seeds[0],
            assignment_sha256=_hash("a"),
        )
        self.assertEqual(suites.exact.backend, "exact")
        self.assertEqual(suites.portable.backend, "portable")
        self.assertEqual(
            suites.exact.logical_cell_ids, suites.portable.logical_cell_ids
        )
        self.assertEqual(len(suites.exact.snapshot_ids), 2)
        self.assertNotEqual(
            suites.exact.runtime_identity_sha256,
            suites.portable.runtime_identity_sha256,
        )
        self.assertEqual(
            suites.exact.policy_seed,
            inputs.plan.evaluation_seed("validation"),
        )

    def test_rejects_suite_cell_count_that_differs_from_plan(self) -> None:
        inputs = _inputs()
        object.__setattr__(inputs.plan, "validation_episodes_per_policy", 3)
        with self.assertRaisesRegex(ValueError, "cell count"):
            PairedEvaluationSuites.build(
                inputs,
                phase="validation",
                learner_seed=inputs.plan.validation_learner_seeds[0],
                assignment_sha256=_hash("a"),
            )

    def test_sharded_evaluation_merges_in_canonical_cell_order(self) -> None:
        inputs = _inputs()
        suite = PairedEvaluationSuites.build(
            inputs,
            phase="validation",
            learner_seed=inputs.plan.validation_learner_seeds[0],
            assignment_sha256=_hash("a"),
        ).exact

        calls = 0

        def evaluator(
            _simulator: object,
            _store: object,
            suite_value: object,
            _model: object,
            _encoder: object,
            _kind_mask: object,
            _wait_mask: object,
            shard: object,
            *,
            evaluator_sha256: str,
            expected_assignment_sha256: str,
            execution_identity_sha256: str,
        ) -> EvaluationShardReport:
            nonlocal calls
            calls += 1
            self.assertEqual(expected_assignment_sha256, suite.assignment_sha256)
            episodes = tuple(
                EpisodeMetrics(
                    snapshot_id,
                    repetition,
                    suite.episode_seed(snapshot_id, repetition),
                    0,
                    1,
                    1,
                    1,
                    1,
                    True,
                    False,
                    0,
                    100,
                    100,
                )
                for snapshot_id, repetition in shard.cells
            )
            return EvaluationShardReport(
                shard,
                EvaluationReport(
                    suite_value.sha256,
                    _hash("b"),
                    evaluator_sha256,
                    suite_value.runtime_identity_sha256,
                    execution_identity_sha256,
                    episodes,
                ),
            )

        with tempfile.TemporaryDirectory() as directory:
            artifacts = ArtifactStore(Path(directory) / "artifacts")
            arguments = {
                "inputs": inputs,
                "simulator": object(),
                "store": object(),
                "suite": suite,
                "model": object(),
                "encoder": object(),
                "kind_mask": object(),
                "wait_mask": object(),
                "policy_sha256": _hash("b"),
                "artifact_store": artifacts,
                "shard_evaluator": evaluator,
            }
            with patch(
                "irisu_rl.r3b_canonical_runner."
                "DeploymentPolicyIdentity.from_components",
                return_value=SimpleNamespace(sha256=_hash("b")),
            ):
                report = evaluate_recurrent_policy_sharded(**arguments)
                replay = evaluate_recurrent_policy_sharded(**arguments)
                source = artifacts.load(artifacts.list()[0]).payload
                assert isinstance(source, dict)
                foreign_worker = _hash("f")
                shard = EvaluationShardPlan.from_manifest(source["shard_plan"])
                execution = _sha256(
                    {
                        "version": "r3b-evaluation-shard-execution-v1",
                        "suite_sha256": suite.sha256,
                        "shard_plan_sha256": shard.sha256,
                        "evaluator_sha256": source["report"]["evaluator_sha256"],
                        "policy_sha256": _hash("b"),
                        "worker_identity_sha256": foreign_worker,
                    }
                )
                foreign_report = EvaluationReport.from_manifest(
                    {
                        **source["report"],
                        "execution_identity_sha256": execution,
                    },
                    suite=suite,
                )
                foreign_shard = EvaluationShardReport(shard, foreign_report)
                artifacts.publish(
                    kind="irisu.r3b.evaluation-shard-package",
                    version="r3b-evaluation-shard-package-v1",
                    payload={
                        **source,
                        "report": foreign_report.manifest(),
                        "shard_report": foreign_shard.manifest(),
                        "worker_identity_sha256": foreign_worker,
                        "execution_identity_sha256": execution,
                    },
                )
                indexed_replay = evaluate_recurrent_policy_sharded(**arguments)
                self.assertEqual(indexed_replay, report)
        self.assertEqual(report, replay)
        self.assertEqual(calls, 2)
        self.assertEqual(
            tuple(value.snapshot_id for value in report.episodes),
            suite.snapshot_ids,
        )

    def test_sealed_workflow_completion_fails_closed(self) -> None:
        arm = CandidateArm(0, 0.0001)
        job = TrialJob(
            _hash("a"),
            "test",
            arm,
            1,
            1000,
            True,
            _hash("b"),
            _hash("c"),
        )
        claim = JobClaim(job.sha256, "test", _hash("d"), "runner", 0, None)
        published = PublishedCanonicalOutcome(
            _hash("1"),
            _hash("2"),
            _hash("3"),
            _hash("4"),
            _hash("5"),
            _hash("6"),
        )
        with self.assertRaisesRegex(RuntimeError, "transaction coordinator"):
            complete_nonsealed_workflow_job(
                workflow=R3BWorkflow("/tmp/unopened-r3-workflow.sqlite3"),
                claim=claim,
                job=job,
                published=published,
            )

    def test_nonsealed_completion_requires_verified_artifact_for_exact_job(
        self,
    ) -> None:
        inputs = _inputs()
        workflow = inputs.workflow
        job = workflow.calibration_jobs(inputs.plan)[0]
        claim = JobClaim(job.sha256, "calibration", _hash("d"), "runner", 0, None)
        published = PublishedCanonicalOutcome(
            _hash("1"),
            _hash("2"),
            _hash("3"),
            _hash("4"),
            _hash("5"),
            _hash("6"),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "artifacts")
            with (
                patch(
                    "irisu_rl.r3b_canonical_runner.load_published_canonical_outcome",
                    return_value=(
                        SimpleNamespace(
                            engineering_evidence=SimpleNamespace(job_sha256=job.sha256)
                        ),
                        published,
                    ),
                ) as load,
                patch.object(workflow, "complete") as complete,
            ):
                complete_nonsealed_workflow_job(
                    workflow=workflow,
                    claim=claim,
                    job=job,
                    published=published,
                    inputs=inputs,
                    store=store,
                )
            load.assert_called_once_with(
                inputs=inputs,
                store=store,
                output_artifact_sha256=published.output_artifact_sha256,
            )
            complete.assert_called_once_with(claim, published.output_artifact_sha256)

            with (
                patch(
                    "irisu_rl.r3b_canonical_runner.load_published_canonical_outcome",
                    side_effect=ValueError("artifact does not exist"),
                ),
                patch.object(workflow, "complete") as complete,
                self.assertRaisesRegex(ValueError, "does not exist"),
            ):
                complete_nonsealed_workflow_job(
                    workflow=workflow,
                    claim=claim,
                    job=job,
                    published=published,
                    inputs=inputs,
                    store=store,
                )
            complete.assert_not_called()

    def test_nonsealed_completion_rejects_foreign_job_artifact(self) -> None:
        inputs = _inputs()
        workflow = inputs.workflow
        job = workflow.calibration_jobs(inputs.plan)[0]
        claim = JobClaim(job.sha256, "calibration", _hash("d"), "runner", 0, None)
        published = PublishedCanonicalOutcome(
            _hash("1"),
            _hash("2"),
            _hash("3"),
            _hash("4"),
            _hash("5"),
            _hash("6"),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ArtifactStore(Path(directory) / "artifacts")
            with (
                patch(
                    "irisu_rl.r3b_canonical_runner.load_published_canonical_outcome",
                    return_value=(
                        SimpleNamespace(
                            engineering_evidence=SimpleNamespace(job_sha256=_hash("f"))
                        ),
                        published,
                    ),
                ),
                patch.object(workflow, "complete") as complete,
                self.assertRaisesRegex(ValueError, "foreign"),
            ):
                complete_nonsealed_workflow_job(
                    workflow=workflow,
                    claim=claim,
                    job=job,
                    published=published,
                    inputs=inputs,
                    store=store,
                )
            complete.assert_not_called()

    def test_canonical_loader_rejects_smoke_workflow_before_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resolved = {
                "version": "r3b-resolved-run-v1",
                "workflow": {"run_class": "smoke"},
                "plan": {},
                "operational_config": {},
                "snapshot_bundle_path": "/exact",
                "portable_snapshot_bundle_path": "/portable",
                "pairing_manifests": {},
            }
            (root / "resolved-run.json").write_bytes(
                json.dumps(
                    resolved,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode()
                + b"\n"
            )
            fake_workflow = SimpleNamespace(
                verify=lambda: {
                    "run_class": "smoke",
                    "acceptance_eligible": False,
                    "transfer_eligible": False,
                }
            )
            with (
                patch(
                    "irisu_rl.r3b_canonical_runner.R3BWorkflow",
                    return_value=fake_workflow,
                ),
                self.assertRaisesRegex(ValueError, "acceptance-eligible"),
            ):
                CanonicalRunInputs.load(
                    root,
                    exact_simulator=object(),
                    portable_simulator=object(),
                )

    def test_canonical_loader_requires_exact_workflow_anchored_resume_receipt(
        self,
    ) -> None:
        current = {"source_identity_sha256": _hash("7")}
        workflow_hash = _sha256(current)
        job_sha256 = _hash("8")
        output_id, metrics_id, parity_id = _hash("1"), _hash("2"), _hash("3")
        receipt_id, evidence_id, outcome_id = _hash("4"), _hash("5"), _hash("6")
        curve_suite = SimpleNamespace(name="curve")
        exact_suite = SimpleNamespace(name="exact")
        portable_suite = SimpleNamespace(name="portable")
        exact_report = SimpleNamespace(name="exact-report")
        portable_report = SimpleNamespace(name="portable-report")
        metrics = SimpleNamespace(final_report=exact_report)
        parity = object()
        resume = object()
        evidence = SimpleNamespace(job_sha256=job_sha256)
        outcome = SimpleNamespace(
            sha256=outcome_id,
            engineering_evidence=evidence,
            assignment_sha256=_hash("9"),
        )
        job = SimpleNamespace(phase="calibration", learner_seed=11)
        final_suites = SimpleNamespace(exact=exact_suite, portable=portable_suite)
        curve_suites = SimpleNamespace(exact=curve_suite)
        output_payload = {
            "job_sha256": job_sha256,
            "learner_outcome_sha256": outcome_id,
            "learner_outcome": {},
            "metrics_artifact_sha256": metrics_id,
            "parity_artifact_sha256": parity_id,
            "resume_receipt_sha256": receipt_id,
            "engineering_evidence_sha256": evidence_id,
            "workflow_manifest_sha256": workflow_hash,
            "acceptance_eligible": True,
            "transfer_eligible": False,
        }
        packages = {
            output_id: SimpleNamespace(payload=output_payload, artifact_id=output_id),
            metrics_id: SimpleNamespace(
                payload={
                    "curve_suite": "curve",
                    "final_suite": "exact",
                    "reports": [],
                    "final_report": "exact-report",
                    "metrics": {},
                    "workflow_manifest_sha256": workflow_hash,
                }
            ),
            parity_id: SimpleNamespace(
                payload={
                    "portable_suite": "portable",
                    "portable_report": "portable-report",
                    "exact_suite": "exact",
                    "exact_report": "exact-report",
                    "logical_manifest": {},
                    "portable_library": {},
                    "exact_library": {},
                    "parity": {},
                    "workflow_manifest_sha256": workflow_hash,
                }
            ),
            evidence_id: SimpleNamespace(
                payload={
                    "engineering_evidence": {
                        "job_sha256": job_sha256,
                        "runner_spec_sha256": _hash("a"),
                    },
                    "metrics_artifact_sha256": metrics_id,
                    "parity_artifact_sha256": parity_id,
                    "resume_receipt_sha256": receipt_id,
                    "workflow_manifest_sha256": workflow_hash,
                }
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ArtifactStore(root / "artifacts")
            workflow = SimpleNamespace(
                verify=lambda: current,
                verify_resume_audit=Mock(return_value=False),
                job_record=lambda _job: {"manifest": {}},
            )
            inputs = SimpleNamespace(
                root=root,
                workflow=workflow,
                workflow_manifest_sha256=workflow_hash,
                exact_bundle=SimpleNamespace(runtime_identity_sha256=_hash("b")),
                plan=SimpleNamespace(),
            )

            def suite(value: object) -> object:
                return {
                    "curve": curve_suite,
                    "exact": exact_suite,
                    "portable": portable_suite,
                }[value]  # type: ignore[index]

            def report(value: object, **_kwargs: object) -> object:
                return {
                    "exact-report": exact_report,
                    "portable-report": portable_report,
                }[value]  # type: ignore[index]

            with (
                patch.object(store, "load", side_effect=lambda identity, **_: packages[identity]),
                patch(
                    "irisu_rl.r3b_canonical_runner._source_identity",
                    return_value=_hash("7"),
                ),
                patch.object(EvaluationSuite, "from_manifest", side_effect=suite),
                patch.object(EvaluationReport, "from_manifest", side_effect=report),
                patch(
                    "irisu_rl.r3b_canonical_runner.RawScoreMetricsArtifact.from_manifest",
                    return_value=metrics,
                ),
                patch.object(
                    CrossBackendEvaluationManifest,
                    "from_manifest",
                    return_value=object(),
                ),
                patch.object(SnapshotLibrary, "from_manifest", return_value=object()),
                patch(
                    "irisu_rl.r3b_canonical_runner.LearnedPolicyBackendParityArtifact.from_manifest",
                    return_value=parity,
                ),
                patch(
                    "irisu_rl.r3b_canonical_runner.ExactResumeVerificationReceipt.load_verified_artifact",
                    return_value=resume,
                ) as load_receipt,
                patch(
                    "irisu_rl.r3b_canonical_runner.EngineeringEvidence.from_manifest",
                    return_value=evidence,
                ),
                patch(
                    "irisu_rl.r3b_canonical_runner.LearnerOutcome.from_manifest",
                    return_value=outcome,
                ),
                patch.object(TrialJob, "from_manifest", return_value=job),
                patch.object(
                    PairedEvaluationSuites,
                    "build",
                    side_effect=(final_suites, curve_suites),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "workflow authority"):
                    load_published_canonical_outcome(
                        inputs=inputs,  # type: ignore[arg-type]
                        store=store,
                        output_artifact_sha256=output_id,
                    )
                load_receipt.assert_not_called()

                workflow.verify_resume_audit.return_value = True
                loaded, published = load_published_canonical_outcome(
                    inputs=inputs,  # type: ignore[arg-type]
                    store=store,
                    output_artifact_sha256=output_id,
                )
            self.assertIs(loaded, outcome)
            self.assertEqual(published.resume_receipt_sha256, receipt_id)
            load_receipt.assert_called_once()


if __name__ == "__main__":
    unittest.main()
