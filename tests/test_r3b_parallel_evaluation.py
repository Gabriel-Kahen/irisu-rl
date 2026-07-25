from __future__ import annotations

import hashlib
import multiprocessing
import os
import pickle
import signal
import subprocess
import tempfile
import time
import unittest
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch
from irisu_rl.collector import model_state_sha256
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.r3b_canonical_runner import (
    PairedEvaluationSuites,
    canonical_evaluation_identities,
    canonical_shard_execution_identity,
)
from irisu_rl.r3b_evaluation import (
    EpisodeMetrics,
    EvaluationReport,
    deployment_policy_identity_for_threads,
)
from irisu_rl.r3b_parallel_evaluation import (
    CanonicalEvaluationTask,
    CanonicalEvaluationTaskResult,
    _model_from_task,
    serialize_evaluation_model,
)
from irisu_rl.r3b_evaluation_shards import (
    EvaluationShardReport,
    merge_evaluation_shards,
    plan_evaluation_shards,
)
from irisu_rl.r3b_supervisor import (
    _collect_evaluation_results,
    _evaluation_worker_initializer,
    _stop_evaluation_executor,
)
from irisu_rl.r3b_lock import R3BRunLock, evaluator_lease_path
from irisu_rl.schema import TEACHER_V1

from tests.test_r3b_canonical_runner import _inputs


def _hash(character: str) -> str:
    return character * 64


def _spawn_blocking_descendant(path: str) -> None:
    child = subprocess.Popen(["sleep", "60"])
    Path(path).write_text(f"{os.getpid()} {child.pid}\n", encoding="utf-8")
    child.wait()


def _task_fixture() -> tuple[
    CanonicalEvaluationTask,
    object,
    RecurrentActorCritic,
]:
    inputs = _inputs()
    config = inputs.config
    model = RecurrentActorCritic(
        TEACHER_V1,
        config=RecurrentModelConfig(
            config.model_global_hidden,
            config.model_body_hidden,
            config.model_fused_hidden,
            config.model_recurrent_hidden,
            config.model_recurrent_layers,
            critic_condition_features=1,
        ),
    )
    encoder = TeacherStateEncoder()
    kind_mask = torch.ones((1, 3), dtype=torch.bool)
    wait_mask = torch.ones((1, len(model.action_spec.wait_choices)), dtype=torch.bool)
    deployment = deployment_policy_identity_for_threads(
        model,
        encoder,
        kind_mask,
        wait_mask,
        torch_threads=config.evaluation_torch_threads,
    )
    assignment = _hash("a")
    suite = PairedEvaluationSuites.build(
        inputs,
        phase="calibration",
        learner_seed=inputs.plan.calibration_learner_seeds[0],
        assignment_sha256=assignment,
        purpose="curve",
    ).exact
    transport = serialize_evaluation_model(model)
    evaluator_sha256, worker_identity_sha256 = canonical_evaluation_identities(
        inputs, suite
    )
    task = CanonicalEvaluationTask(
        run_directory="/tmp/r3-canonical-run",
        exact_worker_path="/tmp/irisu-exact-worker",
        portable_library_path="/tmp/libirisu_clone.so",
        phase="calibration",
        job_sha256=_hash("b"),
        learner_seed=inputs.plan.calibration_learner_seeds[0],
        completed_updates=0,
        authorization_sha256=None,
        assignment_sha256=assignment,
        workflow_manifest_sha256=inputs.workflow_manifest_sha256,
        operational_config_sha256=inputs.config.sha256,
        purpose="curve",
        backend="exact",
        suite_sha256=suite.sha256,
        model_sha256=model_state_sha256(model),
        deployment_policy_sha256=deployment.sha256,
        evaluator_sha256=evaluator_sha256,
        worker_identity_sha256=worker_identity_sha256,
        evaluation_shards=inputs.config.evaluation_shards,
        model_transport_sha256=hashlib.sha256(transport).hexdigest(),
        model_transport=transport,
    )
    return task, suite, model


class R3BParallelEvaluationTests(unittest.TestCase):
    def test_deployment_identity_binds_threads_and_restores_training_setting(
        self,
    ) -> None:
        task, _suite, model = _task_fixture()
        encoder = TeacherStateEncoder()
        kind_mask = torch.ones((1, 3), dtype=torch.bool)
        wait_mask = torch.ones(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
        previous = torch.get_num_threads()
        alternate_threads = 2 if previous != 2 else 3
        alternate = deployment_policy_identity_for_threads(
            model,
            encoder,
            kind_mask,
            wait_mask,
            torch_threads=alternate_threads,
        )
        self.assertEqual(torch.get_num_threads(), previous)
        self.assertNotEqual(alternate.sha256, task.deployment_policy_sha256)

    def test_model_transport_is_bounded_picklable_and_identity_checked(self) -> None:
        task, _suite, expected_model = _task_fixture()
        restored_task = pickle.loads(pickle.dumps(task))
        self.assertEqual(restored_task, task)
        inputs = _inputs()
        model, _encoder, _kind_mask, _wait_mask = _model_from_task(task, inputs)
        self.assertEqual(
            model_state_sha256(model),
            model_state_sha256(expected_model),
        )

        tampered = bytearray(task.model_transport)
        tampered[-1] ^= 1
        with self.assertRaisesRegex(ValueError, "malformed"):
            replace(task, model_transport=bytes(tampered))
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            _model_from_task(replace(task, model_sha256=_hash("f")), inputs)

    def test_result_round_trip_rejects_foreign_task_or_policy(self) -> None:
        task, suite, _model = _task_fixture()
        episodes = tuple(
            EpisodeMetrics(
                snapshot_id,
                0,
                suite.episode_seed(snapshot_id, 0),
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
            for snapshot_id in suite.snapshot_ids
        )
        shards = tuple(
            EvaluationShardReport(
                shard,
                EvaluationReport(
                    suite.sha256,
                    task.deployment_policy_sha256,
                    task.evaluator_sha256,
                    suite.runtime_identity_sha256,
                    canonical_shard_execution_identity(
                        suite=suite,
                        shard=shard,
                        evaluator_sha256=task.evaluator_sha256,
                        policy_sha256=task.deployment_policy_sha256,
                        worker_identity_sha256=task.worker_identity_sha256,
                    ),
                    tuple(
                        episode
                        for episode in episodes
                        if (episode.snapshot_id, episode.repetition) in shard.cells
                    ),
                ),
            )
            for shard in plan_evaluation_shards(suite, task.evaluation_shards)
        )
        report = merge_evaluation_shards(suite, shards)
        result = CanonicalEvaluationTaskResult(
            task.sha256,
            task.suite_sha256,
            task.deployment_policy_sha256,
            report.manifest(),
        )
        self.assertEqual(result.report_for(task, suite), report)
        with self.assertRaisesRegex(ValueError, "another task"):
            result.report_for(replace(task, job_sha256=_hash("e")), suite)
        with self.assertRaisesRegex(ValueError, "another task"):
            replace(result, policy_sha256=_hash("f")).report_for(task, suite)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            replace(
                result,
                report_manifest=replace(report, evaluator_sha256=_hash("c")).manifest(),
            ).report_for(task, suite)
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            replace(
                result,
                report_manifest=replace(
                    report, execution_identity_sha256=_hash("d")
                ).manifest(),
            ).report_for(task, suite)

    def test_parent_collection_is_complete_and_fail_closed(self) -> None:
        task, _suite, _model = _task_fixture()
        result = CanonicalEvaluationTaskResult(
            task.sha256,
            task.suite_sha256,
            task.deployment_policy_sha256,
            {},
        )
        completed: Future[CanonicalEvaluationTaskResult] = Future()
        completed.set_result(result)
        self.assertEqual(
            _collect_evaluation_results({completed: task}),
            {task.sha256: result},
        )

        failed: Future[CanonicalEvaluationTaskResult] = Future()
        failed.set_exception(ValueError("worker failed"))
        with self.assertRaisesRegex(
            RuntimeError, f"{task.sha256}.*ValueError: worker failed"
        ):
            _collect_evaluation_results({failed: task})

        duplicate: Future[CanonicalEvaluationTaskResult] = Future()
        duplicate.set_result(result)
        with self.assertRaisesRegex(ValueError, "identity differs"):
            _collect_evaluation_results({completed: task, duplicate: task})

    @unittest.skipUnless(os.name == "posix", "process-group cleanup is POSIX-only")
    def test_failed_executor_bounds_descendant_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pids"
            executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_evaluation_worker_initializer,
                initargs=(os.getpid(), str(evaluator_lease_path(directory))),
            )
            executor.submit(_spawn_blocking_descendant, str(path))
            deadline = time.monotonic() + 10
            while not path.is_file() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(path.is_file())
            pids = tuple(int(value) for value in path.read_text().split())
            try:
                with self.assertRaisesRegex(RuntimeError, "another process"):
                    with R3BRunLock(directory, global_directory=directory):
                        pass
                started = time.monotonic()
                _stop_evaluation_executor(executor, timeout_seconds=1)
                self.assertLess(time.monotonic() - started, 3)
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    if all(not Path(f"/proc/{pid}").exists() for pid in pids):
                        break
                    time.sleep(0.01)
                self.assertTrue(all(not Path(f"/proc/{pid}").exists() for pid in pids))
            finally:
                for pid in pids:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
