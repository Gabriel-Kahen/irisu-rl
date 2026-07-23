from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from irisu_rl.collector import CollectorConfig
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.ppo import PPOConfig
from irisu_rl.r3b_experiments import (
    SealedTestLedger,
    TrialJob,
    bind_validation_run,
    load_plan,
)
from irisu_rl.r3b_runner import R3BRunBuilder
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
    validation_authorization,
)
from tests.test_r3a_curriculum import curriculum as multistage_curriculum
from irisu_rl.curriculum import SnapshotBlobStore


ROOT = Path(__file__).resolve().parents[1]


class R3BRunBuilderTests(unittest.TestCase):
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
            builder.sealed_test_ledger = ledger
            with builder.build(test_job, authorization=lease) as trial:
                self.assertTrue(trial.manifest.sealed)
                self.assertEqual(trial.manifest.authorization_sha256, test_auth.sha256)
                self.assertEqual(trial.sealed_job_lease_sha256, lease.sha256)
            with self.assertRaisesRegex(RuntimeError, "lease is not active"):
                builder.build(test_job, authorization=lease)

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
