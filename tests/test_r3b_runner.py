from __future__ import annotations

from dataclasses import replace
import functools
import hashlib
import unittest
import tempfile
from pathlib import Path

from irisu_rl.collector import CollectorConfig
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.ppo import PPOConfig
from irisu_rl.r3b_experiments import (
    SealedTestLedger,
    TrainingCheckpointArtifact,
    TrialJob,
    bind_validation_run,
    load_plan,
)
from irisu_rl.r3b_runner import (
    R3BRunBuilder,
    _environment_implementation_identity,
    verify_exact_resume_continuation,
)
from irisu_env.vector import SyncVectorEnv
from irisu_rl.schema import TEACHER_V1
from tests.test_r3b_snapshot_initializer import (
    FakeSnapshotVector,
    FakeRuntimeLane,
    _RUNTIME_ATTESTATION,
    _fixture,
)
from tests.test_r3b_experiments import (
    TEST_EVALUATION_SUITE,
    VALIDATION_EVALUATION_SUITE,
    authorization_validation_results,
    phase_result,
    validation_authorization,
)
from tests.test_r3a_curriculum import curriculum as multistage_curriculum
from irisu_rl.curriculum import SnapshotBlobStore


ROOT = Path(__file__).resolve().parents[1]


class R3BRunBuilderTests(unittest.TestCase):
    def test_real_portable_vector_identity_excludes_live_handles(self) -> None:
        library = ROOT / "build/libirisu_clone.so"
        first = SyncVectorEnv(1, library_path=library)
        second = SyncVectorEnv(1, library_path=library)
        try:
            self.assertEqual(
                _environment_implementation_identity(first),
                _environment_implementation_identity(second),
            )
        finally:
            first.close()
            second.close()

    def test_runner_identity_binds_model_and_vector_implementations(self) -> None:
        plan = load_plan(ROOT / "configs/rl/experiments/r3b-completion-v1.toml")
        curriculum, blobs = _fixture()
        store = SnapshotBlobStore(curriculum.library, blobs)
        job = plan.trial_jobs("calibration")[0]

        class AlteredModel(RecurrentActorCritic):
            def forward(self, *args, **kwargs):
                return super().forward(*args, **kwargs)

        class AlteredVector(FakeSnapshotVector):
            pass

        changed = {"model": False, "vector": False}

        def model_factory():
            model_type = AlteredModel if changed["model"] else RecurrentActorCritic
            return model_type(
                TEACHER_V1,
                config=RecurrentModelConfig(
                    8, 8, 12, 12, 1, critic_condition_features=1
                ),
            )

        def environment_factory():
            vector_type = AlteredVector if changed["vector"] else FakeSnapshotVector
            return vector_type()

        def make_builder() -> R3BRunBuilder:
            return R3BRunBuilder(
                plan,
                curriculum,
                store,
                runtime_attestation=_RUNTIME_ATTESTATION,
                lanes=2,
                collector_config=CollectorConfig(
                    max_decisions=1,
                    target_simulated_ticks=plan.ticks_per_update,
                ),
                ppo_config=PPOConfig(
                    learning_rate=1e-4,
                    final_learning_rate_fraction=plan.final_learning_rate_fraction,
                    epochs=1,
                    lane_minibatch_size=2,
                ),
                model_factory=model_factory,
                environment_factory=environment_factory,
            )

        builder = make_builder()
        with builder.build(job):
            pass
        changed["model"] = True
        with self.assertRaisesRegex(TypeError, "must return RecurrentActorCritic"):
            builder.build(job)

        changed["model"] = False
        builder = make_builder()
        with builder.build(job):
            pass
        changed["vector"] = True
        with self.assertRaisesRegex(RuntimeError, "runner implementation changed"):
            builder.build(job)

        modes = iter((1, 2))

        def configured_environment_factory():
            environment = FakeSnapshotVector()
            environment.behavior_mode = next(modes)
            return environment

        configured = R3BRunBuilder(
            plan,
            curriculum,
            store,
            runtime_attestation=_RUNTIME_ATTESTATION,
            lanes=2,
            collector_config=CollectorConfig(
                max_decisions=1,
                target_simulated_ticks=plan.ticks_per_update,
            ),
            ppo_config=PPOConfig(
                learning_rate=1e-4,
                final_learning_rate_fraction=plan.final_learning_rate_fraction,
                epochs=1,
                lane_minibatch_size=2,
            ),
            model_factory=model_factory,
            environment_factory=configured_environment_factory,
        )
        with configured.build(job):
            pass
        with self.assertRaisesRegex(RuntimeError, "runner implementation changed"):
            configured.build(job)

        with self.assertRaisesRegex(TypeError, "plain functions or classes"):
            R3BRunBuilder(
                plan,
                curriculum,
                store,
                runtime_attestation=_RUNTIME_ATTESTATION,
                lanes=2,
                collector_config=configured.collector_config,
                ppo_config=configured.ppo_config,
                model_factory=model_factory,
                environment_factory=functools.partial(FakeSnapshotVector),
            )

    def test_arms_share_initial_model_assignments_and_seed_plan(self) -> None:
        plan = load_plan(ROOT / "configs/rl/experiments/r3b-completion-v1.toml")
        curriculum, blobs = _fixture()
        store = SnapshotBlobStore(curriculum.library, blobs)

        def model_factory():
            return RecurrentActorCritic(
                TEACHER_V1,
                config=RecurrentModelConfig(
                    8, 8, 12, 12, 1, critic_condition_features=1
                ),
            )

        builder = R3BRunBuilder(
            plan,
            curriculum,
            store,
            runtime_attestation=_RUNTIME_ATTESTATION,
            lanes=2,
            collector_config=CollectorConfig(
                max_decisions=1,
                target_simulated_ticks=plan.ticks_per_update,
            ),
            ppo_config=PPOConfig(
                learning_rate=1e-4,
                final_learning_rate_fraction=plan.final_learning_rate_fraction,
                epochs=1,
                lane_minibatch_size=2,
            ),
            model_factory=model_factory,
            environment_factory=FakeSnapshotVector,
        )
        control_arm = plan.arms[0]
        shaped_arm = next(arm for arm in plan.arms if arm.alpha_weight_ppm == 100_000)
        jobs = plan.trial_jobs("calibration")
        control_job = next(
            job
            for job in jobs
            if job.arm == control_arm
            and job.learner_seed == 1103
            and job.budget_updates == plan.calibration_budgets_updates[-1]
        )
        shaped_job = next(
            job
            for job in jobs
            if job.arm == shaped_arm
            and job.learner_seed == 1103
            and job.budget_updates == plan.calibration_budgets_updates[-1]
        )
        with (
            builder.build(control_job) as control,
            builder.build(shaped_job) as shaped,
        ):
            self.assertEqual(
                control.manifest.initial_model_sha256,
                shaped.manifest.initial_model_sha256,
            )
            self.assertEqual(
                control.manifest.assignment_sha256,
                shaped.manifest.assignment_sha256,
            )
            self.assertEqual(
                control.manifest.seed_plan_sha256,
                shaped.manifest.seed_plan_sha256,
            )
            self.assertEqual(
                control.manifest.pairing_sha256,
                shaped.manifest.pairing_sha256,
            )
            self.assertEqual(
                control.manifest.runner_spec_sha256,
                shaped.manifest.runner_spec_sha256,
            )
            self.assertEqual(
                control.session.optimizer_update_limit,
                plan.calibration_budgets_updates[-1],
            )
            self.assertIsNone(control.session.tail_controller)
            self.assertEqual(
                control.manifest.reward_sha256, shaped.manifest.reward_sha256
            )
            self.assertNotEqual(
                control.manifest.curriculum_sha256,
                shaped.manifest.curriculum_sha256,
            )
            self.assertFalse(control.manifest.deployable)
            manifest_sha256 = control.manifest.sha256
            with self.assertRaises(TypeError):
                control.manifest.collector["max_decisions"] = 99  # type: ignore[index]
            self.assertEqual(control.manifest.sha256, manifest_sha256)
            control.session.initialize()
            shaped.session.initialize()
            self.assertEqual(
                control.session.task.coordinator.lane_snapshot_id,
                shaped.session.task.coordinator.lane_snapshot_id,
            )

        fabricated = TrialJob(
            plan.sha256,
            "calibration",
            control_arm,
            999_999_999,
            plan.calibration_budgets_updates[-1],
            False,
            "a" * 64,
        )
        with self.assertRaisesRegex(ValueError, "phase contract"):
            builder.build(fabricated)

        first_rung = next(
            job
            for job in jobs
            if job.arm == control_arm
            and job.learner_seed == 1103
            and job.budget_updates == plan.calibration_budgets_updates[0]
        )
        with builder.build(first_rung) as trial:
            self.assertEqual(
                trial.session.optimizer_update_limit,
                plan.calibration_budgets_updates[0],
            )
            self.assertIsNone(trial.session.tail_controller)

        validation_auth = validation_authorization(plan, plan.learning_rates[0])
        validation_job = plan.trial_jobs("validation", validation_auth)[0]
        with builder.build(validation_job, authorization=validation_auth) as trial:
            self.assertEqual(trial.session.optimizer_update_limit, plan.total_updates)
            self.assertIsNotNone(trial.session.tail_controller)

        with tempfile.TemporaryDirectory() as ledger_directory:
            ledger = SealedTestLedger(Path(ledger_directory) / "sealed.sqlite3")
            commitment = ledger.precommit(plan, TEST_EVALUATION_SUITE)
            calibration = validation_authorization(
                plan, control_arm.learning_rate
            ).calibration_results
            test_validation = bind_validation_run(plan, calibration, commitment)
            validation_results = authorization_validation_results(
                plan,
                control_arm,
                shaped_arm,
                validation_run=test_validation,
            )
            test_auth = ledger.authorize_once(
                plan,
                test_validation,
                validation_results,
                VALIDATION_EVALUATION_SUITE,
                TEST_EVALUATION_SUITE,
            )
            test_job = plan.trial_jobs("test", test_auth)[0]
            with self.assertRaisesRegex(ValueError, "phase-selection authorization"):
                builder.build(  # type: ignore[arg-type]
                    test_job, authorization=test_auth
                )
            lease = ledger.claim_job(test_auth, test_job)
            with self.assertRaisesRegex(RuntimeError, "already claimed"):
                ledger.claim_job(test_auth, test_job)
            resumed_lease = ledger.resume_job(
                test_auth, test_job, lease_token=lease.lease_token
            )
            self.assertEqual(resumed_lease, lease)
            sealed_builder = R3BRunBuilder(
                plan,
                curriculum,
                store,
                runtime_attestation=_RUNTIME_ATTESTATION,
                lanes=2,
                collector_config=builder.collector_config,
                ppo_config=builder.ppo_config,
                model_factory=model_factory,
                environment_factory=FakeSnapshotVector,
                sealed_test_ledger=ledger,
            )
            with sealed_builder.build(test_job, authorization=lease) as trial:
                self.assertTrue(trial.manifest.sealed)
                self.assertEqual(trial.manifest.authorization_sha256, test_auth.sha256)
                self.assertEqual(trial.sealed_job_lease_sha256, lease.sha256)
            with self.assertRaisesRegex(RuntimeError, "lease is not active"):
                sealed_builder.build(test_job, authorization=lease)

            failed_job = plan.trial_jobs("test", test_auth)[1]
            failed_lease = ledger.claim_job(test_auth, failed_job)

            def failing_environment_factory():
                raise RuntimeError("factory failed")

            failing_builder = R3BRunBuilder(
                plan,
                curriculum,
                store,
                runtime_attestation=_RUNTIME_ATTESTATION,
                lanes=2,
                collector_config=builder.collector_config,
                ppo_config=builder.ppo_config,
                model_factory=model_factory,
                environment_factory=failing_environment_factory,
                sealed_test_ledger=ledger,
            )
            with self.assertRaisesRegex(RuntimeError, "factory failed"):
                failing_builder.build(failed_job, authorization=failed_lease)
            with self.assertRaisesRegex(RuntimeError, "lease is not active"):
                ledger.fail_job(failed_lease, "must already be terminal")

            cleanup_job = plan.trial_jobs("test", test_auth)[2]
            cleanup_lease = ledger.claim_job(test_auth, cleanup_job)

            class BadLaneVector(FakeSnapshotVector):
                def __init__(self):
                    super().__init__()
                    self.num_envs = 1

                def close(self):
                    raise RuntimeError("close failed")

            cleanup_builder = R3BRunBuilder(
                plan,
                curriculum,
                store,
                runtime_attestation=_RUNTIME_ATTESTATION,
                lanes=2,
                collector_config=builder.collector_config,
                ppo_config=builder.ppo_config,
                model_factory=model_factory,
                environment_factory=BadLaneVector,
                sealed_test_ledger=ledger,
            )
            with self.assertRaisesRegex(ValueError, "wrong lane count") as raised:
                cleanup_builder.build(cleanup_job, authorization=cleanup_lease)
            self.assertTrue(
                any(
                    "cleanup also failed" in note for note in raised.exception.__notes__
                )
            )
            with self.assertRaisesRegex(RuntimeError, "lease is not active"):
                ledger.fail_job(cleanup_lease, "must already be terminal")

        with tempfile.TemporaryDirectory() as directory:
            with builder.build(validation_job, authorization=validation_auth) as source:
                source.session.initialize()
                identity = {"trial_manifest_sha256": source.manifest.sha256}
                source.session.save(directory, "r3b", identity=identity)
                expected_policy = source.session.policy_sha256
                assert source.session.tail_controller is not None
                expected_tail = source.session.tail_controller.state_dict()
            with builder.build(
                validation_job, authorization=validation_auth
            ) as restored:
                restored.session.restore(
                    directory,
                    generation="r3b",
                    identity={"trial_manifest_sha256": restored.manifest.sha256},
                )
                self.assertEqual(restored.session.policy_sha256, expected_policy)
                assert restored.session.tail_controller is not None
                self.assertEqual(
                    restored.session.tail_controller.state_dict(), expected_tail
                )

    def test_adaptive_multi_stage_sweep_is_rejected(self) -> None:
        plan = load_plan(ROOT / "configs/rl/experiments/r3b-completion-v1.toml")
        multi_stage = multistage_curriculum()
        with self.assertRaises(ValueError):
            R3BRunBuilder(
                plan,
                multi_stage,
                None,  # type: ignore[arg-type]
                runtime_attestation=_RUNTIME_ATTESTATION,
                lanes=2,
                collector_config=CollectorConfig(
                    target_simulated_ticks=plan.ticks_per_update
                ),
                ppo_config=PPOConfig(
                    final_learning_rate_fraction=plan.final_learning_rate_fraction
                ),
                model_factory=lambda: None,  # type: ignore[arg-type]
                environment_factory=FakeSnapshotVector,
            )

    def test_sealed_resume_audit_uses_running_lease_without_outcome_authority(
        self,
    ) -> None:
        plan = load_plan(ROOT / "configs/rl/experiments/r3b-completion-v1.toml")
        curriculum, blobs = _fixture()
        store = SnapshotBlobStore(curriculum.library, blobs)

        def model_factory():
            return RecurrentActorCritic(
                TEACHER_V1,
                config=RecurrentModelConfig(
                    8, 8, 12, 12, 1, critic_condition_features=1
                ),
            )

        with tempfile.TemporaryDirectory() as directory:
            ledger = SealedTestLedger(Path(directory) / "sealed.sqlite3")
            commitment = ledger.precommit(plan, TEST_EVALUATION_SUITE)
            calibration = validation_authorization(
                plan, plan.learning_rates[0]
            ).calibration_results
            validation_run = bind_validation_run(plan, calibration, commitment)
            validation_results = authorization_validation_results(
                plan,
                plan.arms[0],
                next(arm for arm in plan.arms if arm.alpha_weight_ppm > 0),
                validation_run=validation_run,
            )
            sealed_run = ledger.authorize_once(
                plan,
                validation_run,
                validation_results,
                VALIDATION_EVALUATION_SUITE,
                TEST_EVALUATION_SUITE,
            )
            jobs = plan.trial_jobs("test", sealed_run)
            job = jobs[0]
            lease = ledger.claim_job(sealed_run, job)
            builder = R3BRunBuilder(
                plan,
                curriculum,
                store,
                runtime_attestation=_RUNTIME_ATTESTATION,
                lanes=2,
                collector_config=CollectorConfig(
                    max_decisions=1,
                    target_simulated_ticks=plan.ticks_per_update,
                ),
                ppo_config=PPOConfig(
                    learning_rate=1e-4,
                    final_learning_rate_fraction=plan.final_learning_rate_fraction,
                    epochs=1,
                    lane_minibatch_size=2,
                ),
                model_factory=model_factory,
                environment_factory=FakeSnapshotVector,
                sealed_test_ledger=ledger,
            )

            checkpoint_root = Path(directory) / "checkpoint"
            with builder.build(job, authorization=lease) as primary:
                lease.assert_running()
                primary.session.initialize()
                identity = {"trial_manifest_sha256": primary.manifest.sha256}
                destination = primary.session.save(
                    checkpoint_root, "resume-source", identity=identity
                )
                checkpoint = TrainingCheckpointArtifact(
                    job.learner_seed,
                    primary.session.trainer.schedule.completed_updates,
                    primary.session.collector.simulated_ticks,
                    primary.session.collector.simulated_ticks,
                    plan.sha256,
                    job.sha256,
                    primary.manifest.sha256,
                    primary.manifest.runner_spec_sha256,
                    hashlib.sha256(
                        (destination / "manifest.json").read_bytes()
                    ).hexdigest(),
                    primary.session.policy_sha256,
                    "e" * 64,
                )
                standalone_audit = builder.build_resume_audit_session(
                    job, authorization=lease
                )
                self.assertIsNot(standalone_audit, primary.session)
                self.assertIsNot(
                    standalone_audit.collector.adapter.env,
                    primary.environment,
                )
                self.assertFalse(hasattr(standalone_audit, "engineering_evidence"))
                close = getattr(standalone_audit.collector.adapter.env, "close", None)
                if close is not None:
                    close()

                artifact = verify_exact_resume_continuation(
                    trial_manifest_sha256=primary.manifest.sha256,
                    checkpoint=checkpoint,
                    checkpoint_directory=checkpoint_root,
                    generation="resume-source",
                    checkpoint_identity=identity,
                    source=primary.session,
                    restored_factory=lambda: builder.build_resume_audit_session(
                        job, authorization=lease
                    ),
                )
                self.assertEqual(
                    artifact.source_next_update_sha256,
                    artifact.restored_next_update_sha256,
                )
                lease.assert_running()

            unstarted_job = jobs[1]
            unstarted = ledger.claim_job(sealed_run, unstarted_job)
            with self.assertRaisesRegex(RuntimeError, "lease is not active"):
                builder.build_resume_audit_session(
                    unstarted_job, authorization=unstarted
                )
            self.assertEqual(
                ledger.resume_job(
                    sealed_run,
                    unstarted_job,
                    lease_token=unstarted.lease_token,
                ),
                unstarted,
            )
            with self.assertRaisesRegex(ValueError, "phase-selection authorization"):
                builder.build_resume_audit_session(job, authorization=unstarted)

            synthetic = phase_result(
                job.arm,
                "test",
                (job.learner_seed,),
                budget=plan.test_updates,
                auc=100,
                final=100,
                authorization_sha256=sealed_run.sha256,
            ).outcomes[0]
            assert synthetic.engineering_evidence is not None
            metrics = replace(
                synthetic.metrics_artifact,
                checkpoints=tuple(
                    replace(
                        evaluated,
                        checkpoint=replace(
                            evaluated.checkpoint,
                            job_sha256=job.sha256,
                        ),
                    )
                    for evaluated in synthetic.metrics_artifact.checkpoints
                ),
            )
            completed = replace(
                synthetic,
                seed_plan_sha256=job.seed_plan_sha256,
                metrics_artifact=metrics,
                engineering_evidence=replace(
                    synthetic.engineering_evidence,
                    job_sha256=job.sha256,
                    sealed_job_lease_sha256=lease.sha256,
                    metrics_sha256=metrics.sha256,
                    final_checkpoint_artifact=metrics.checkpoints[-1].checkpoint,
                    resume_checkpoint_artifact=metrics.checkpoints[-2].checkpoint,
                ),
            )
            ledger.complete_job(lease, completed)
            with self.assertRaisesRegex(RuntimeError, "lease is not active"):
                builder.build_resume_audit_session(job, authorization=lease)

    def test_runtime_and_model_factory_are_frozen(self) -> None:
        plan = load_plan(ROOT / "configs/rl/experiments/r3b-completion-v1.toml")
        curriculum, blobs = _fixture()
        store = SnapshotBlobStore(curriculum.library, blobs)
        collector = CollectorConfig(
            max_decisions=1,
            target_simulated_ticks=plan.ticks_per_update,
        )
        ppo = PPOConfig(
            final_learning_rate_fraction=plan.final_learning_rate_fraction,
            epochs=1,
            lane_minibatch_size=2,
        )
        job = plan.trial_jobs("calibration")[0]

        class WrongRuntimeLane(FakeRuntimeLane):
            build_info = {
                **FakeRuntimeLane.build_info,
                "clone_version": "different-runtime",
            }

        class WrongRuntimeVector(FakeSnapshotVector):
            def __init__(self) -> None:
                super().__init__()
                self.envs = (WrongRuntimeLane(), WrongRuntimeLane())

        def stable_model():
            return RecurrentActorCritic(
                TEACHER_V1,
                config=RecurrentModelConfig(
                    8, 8, 12, 12, 1, critic_condition_features=1
                ),
            )

        wrong_runtime = R3BRunBuilder(
            plan,
            curriculum,
            store,
            runtime_attestation=_RUNTIME_ATTESTATION,
            lanes=2,
            collector_config=collector,
            ppo_config=ppo,
            model_factory=stable_model,
            environment_factory=WrongRuntimeVector,
        )
        with self.assertRaisesRegex(RuntimeError, "unattested runtime"):
            wrong_runtime.build(job)

        widths = iter((8, 9))

        def changing_model():
            width = next(widths)
            return RecurrentActorCritic(
                TEACHER_V1,
                config=RecurrentModelConfig(
                    width, 8, 12, 12, 1, critic_condition_features=1
                ),
            )

        changing = R3BRunBuilder(
            plan,
            curriculum,
            store,
            runtime_attestation=_RUNTIME_ATTESTATION,
            lanes=2,
            collector_config=collector,
            ppo_config=ppo,
            model_factory=changing_model,
            environment_factory=FakeSnapshotVector,
        )
        with changing.build(job):
            pass
        with self.assertRaisesRegex(RuntimeError, "runner specification"):
            changing.build(plan.trial_jobs("calibration")[1])


if __name__ == "__main__":
    unittest.main()
