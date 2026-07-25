"""Spawn-isolated execution of immutable canonical evaluation tasks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path

import torch
from irisu_env import IrisuEnv, PaddedVectorEnv

from .collector import model_state_sha256
from .encoding import TeacherStateEncoder
from .models import RecurrentActorCritic, RecurrentModelConfig
from .r3b_artifacts import ArtifactStore
from .r3b_canonical_runner import (
    CanonicalRunInputs,
    PairedEvaluationSuites,
    evaluate_recurrent_policy_sharded,
)
from .r3b_evaluation import (
    EvaluationReport,
    EvaluationSuite,
    deployment_policy_identity_for_threads,
)
from .r3b_experiments import TrialJob
from .runtime_identity import attest_simulator_runtime
from .schema import TEACHER_V1

_MAX_MODEL_TRANSPORT_BYTES = 64 * 1024 * 1024
_SHA256_ZERO = "0" * 64
_INPUT_CACHE: dict[tuple[str, str, str], CanonicalRunInputs] = {}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_nonzero_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value != _SHA256_ZERO
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class CanonicalEvaluationTask:
    """One checkpoint/suite evaluation with every trusted identity bound."""

    run_directory: str
    exact_worker_path: str
    portable_library_path: str
    phase: str
    job_sha256: str
    learner_seed: int
    completed_updates: int
    authorization_sha256: str | None
    assignment_sha256: str
    workflow_manifest_sha256: str
    operational_config_sha256: str
    purpose: str
    backend: str
    suite_sha256: str
    model_sha256: str
    deployment_policy_sha256: str
    model_transport_sha256: str
    model_transport: bytes = field(repr=False, compare=False)
    version: str = "r3b-canonical-evaluation-task-v1"

    def __post_init__(self) -> None:
        paths = (
            Path(self.run_directory),
            Path(self.exact_worker_path),
            Path(self.portable_library_path),
        )
        hashes = (
            self.job_sha256,
            self.assignment_sha256,
            self.workflow_manifest_sha256,
            self.operational_config_sha256,
            self.suite_sha256,
            self.model_sha256,
            self.deployment_policy_sha256,
            self.model_transport_sha256,
        )
        if (
            self.version != "r3b-canonical-evaluation-task-v1"
            or self.phase not in {"calibration", "validation", "test"}
            or self.purpose not in {"curve", "final"}
            or self.backend not in {"exact", "portable"}
            or (self.purpose == "curve" and self.backend != "exact")
            or any(not path.is_absolute() for path in paths)
            or isinstance(self.learner_seed, bool)
            or not isinstance(self.learner_seed, int)
            or not 0 <= self.learner_seed < 2**64
            or isinstance(self.completed_updates, bool)
            or not isinstance(self.completed_updates, int)
            or self.completed_updates < 0
            or any(not _is_nonzero_sha256(value) for value in hashes)
            or (
                self.authorization_sha256 is not None
                and not _is_nonzero_sha256(self.authorization_sha256)
            )
            or (self.phase == "calibration") != (self.authorization_sha256 is None)
            or not isinstance(self.model_transport, bytes)
            or not 0 < len(self.model_transport) <= _MAX_MODEL_TRANSPORT_BYTES
            or hashlib.sha256(self.model_transport).hexdigest()
            != self.model_transport_sha256
        ):
            raise ValueError("canonical evaluation task is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "run_directory": self.run_directory,
            "exact_worker_path": self.exact_worker_path,
            "portable_library_path": self.portable_library_path,
            "phase": self.phase,
            "job_sha256": self.job_sha256,
            "learner_seed": self.learner_seed,
            "completed_updates": self.completed_updates,
            "authorization_sha256": self.authorization_sha256,
            "assignment_sha256": self.assignment_sha256,
            "workflow_manifest_sha256": self.workflow_manifest_sha256,
            "operational_config_sha256": self.operational_config_sha256,
            "purpose": self.purpose,
            "backend": self.backend,
            "suite_sha256": self.suite_sha256,
            "model_sha256": self.model_sha256,
            "deployment_policy_sha256": self.deployment_policy_sha256,
            "model_transport_sha256": self.model_transport_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class CanonicalEvaluationTaskResult:
    """Pickle-safe report returned from one isolated evaluator process."""

    task_sha256: str
    suite_sha256: str
    policy_sha256: str
    report_manifest: dict[str, object]
    version: str = "r3b-canonical-evaluation-task-result-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-canonical-evaluation-task-result-v1"
            or not _is_nonzero_sha256(self.task_sha256)
            or not _is_nonzero_sha256(self.suite_sha256)
            or not _is_nonzero_sha256(self.policy_sha256)
            or not isinstance(self.report_manifest, dict)
        ):
            raise ValueError("canonical evaluation task result is malformed")

    def report_for(
        self,
        task: CanonicalEvaluationTask,
        suite: EvaluationSuite,
    ) -> EvaluationReport:
        if (
            not isinstance(task, CanonicalEvaluationTask)
            or self.task_sha256 != task.sha256
            or self.suite_sha256 != task.suite_sha256
            or self.policy_sha256 != task.deployment_policy_sha256
        ):
            raise ValueError("canonical evaluation result belongs to another task")
        report = EvaluationReport.from_manifest(self.report_manifest, suite=suite)
        if (
            report.suite_sha256 != task.suite_sha256
            or report.policy_sha256 != task.deployment_policy_sha256
        ):
            raise ValueError("canonical evaluation report identity differs")
        return report


def serialize_evaluation_model(model: RecurrentActorCritic) -> bytes:
    """Copy a learned model into a bounded, process-safe transport payload."""

    if not isinstance(model, RecurrentActorCritic):
        raise TypeError("evaluation model must be recurrent actor-critic")
    state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    buffer = BytesIO()
    torch.save(state, buffer)
    payload = buffer.getvalue()
    if not 0 < len(payload) <= _MAX_MODEL_TRANSPORT_BYTES:
        raise ValueError("serialized evaluation model exceeds the safety bound")
    return payload


def _model_from_task(
    task: CanonicalEvaluationTask,
    inputs: CanonicalRunInputs,
) -> tuple[
    RecurrentActorCritic,
    TeacherStateEncoder,
    torch.Tensor,
    torch.Tensor,
]:
    if (
        not isinstance(task.model_transport, bytes)
        or hashlib.sha256(task.model_transport).hexdigest()
        != task.model_transport_sha256
    ):
        raise ValueError("evaluation model transport hash differs")
    try:
        state = torch.load(
            BytesIO(task.model_transport),
            map_location="cpu",
            weights_only=True,
        )
    except Exception as exc:
        raise ValueError("evaluation model transport cannot be decoded") from exc
    if not isinstance(state, Mapping) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("evaluation model transport is not a tensor state mapping")
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
    try:
        model.load_state_dict(state, strict=True)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ValueError(
            "evaluation model transport differs from the architecture"
        ) from exc
    if model_state_sha256(model) != task.model_sha256:
        raise ValueError("evaluation model transport differs from the checkpoint")
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
    if deployment.sha256 != task.deployment_policy_sha256:
        raise ValueError("evaluation deployment identity differs from the checkpoint")
    model.eval()
    return model, encoder, kind_mask, wait_mask


def _load_inputs(task: CanonicalEvaluationTask) -> CanonicalRunInputs:
    key = (
        task.run_directory,
        task.exact_worker_path,
        task.portable_library_path,
    )
    inputs = _INPUT_CACHE.get(key)
    if inputs is None:
        worker = Path(task.exact_worker_path)
        library = Path(task.portable_library_path)
        if (
            worker.is_symlink()
            or not worker.is_file()
            or library.is_symlink()
            or not library.is_file()
        ):
            raise ValueError("canonical evaluation runtime path is missing or unsafe")
        with ExitStack() as stack:
            exact_loader = stack.enter_context(
                IrisuEnv(physics_backend="exact", worker_path=worker)
            )
            portable_loader = stack.enter_context(
                IrisuEnv(physics_backend="portable", library_path=library)
            )
            inputs = CanonicalRunInputs.load(
                task.run_directory,
                exact_simulator=exact_loader,
                portable_simulator=portable_loader,
            )
        _INPUT_CACHE[key] = inputs
    if (
        inputs.workflow_manifest_sha256 != task.workflow_manifest_sha256
        or inputs.config.sha256 != task.operational_config_sha256
    ):
        raise ValueError("canonical evaluation task belongs to another frozen run")
    return inputs


def evaluate_canonical_task(
    task: CanonicalEvaluationTask,
) -> CanonicalEvaluationTaskResult:
    """Execute one task in its own spawned process without mutating workflow state."""

    if not isinstance(task, CanonicalEvaluationTask):
        raise TypeError("canonical evaluator requires a typed task")
    task.__post_init__()
    inputs = _load_inputs(task)
    record = inputs.workflow.job_record(task.job_sha256)
    if record["status"] != "trained":
        raise RuntimeError("canonical evaluation task requires a trained job")
    job = TrialJob.from_manifest(record["manifest"])
    if (
        job.sha256 != task.job_sha256
        or job.phase != task.phase
        or job.learner_seed != task.learner_seed
        or job.authorization_sha256 != task.authorization_sha256
        or task.completed_updates > job.budget_updates
        or task.completed_updates % inputs.plan.checkpoint_interval_updates
        or (
            task.purpose == "final"
            and task.completed_updates != job.budget_updates
        )
    ):
        raise ValueError("canonical evaluation task job identity differs")
    suites = PairedEvaluationSuites.build(
        inputs,
        phase=task.phase,
        learner_seed=task.learner_seed,
        assignment_sha256=task.assignment_sha256,
        purpose=task.purpose,
    )
    suite = suites.exact if task.backend == "exact" else suites.portable
    if suite.sha256 != task.suite_sha256:
        raise ValueError("canonical evaluation task suite identity differs")
    torch.set_num_threads(inputs.config.evaluation_torch_threads)
    model, encoder, kind_mask, wait_mask = _model_from_task(task, inputs)
    vector_arguments: dict[str, object] = {
        "workers": inputs.config.evaluation_workers,
        "physics_backend": task.backend,
    }
    if task.backend == "exact":
        vector_arguments["worker_path"] = task.exact_worker_path
        snapshot_store = inputs.exact_bundle.store
    else:
        vector_arguments["library_path"] = task.portable_library_path
        snapshot_store = inputs.portable_bundle.store
    with PaddedVectorEnv(
        inputs.config.evaluation_lanes,
        **vector_arguments,
    ) as simulator:
        runtime = attest_simulator_runtime(simulator.envs[0])
        if runtime.sha256 != suite.runtime_identity_sha256:
            raise ValueError("canonical evaluation runtime identity differs")
        report = evaluate_recurrent_policy_sharded(
            inputs=inputs,
            simulator=simulator,
            store=snapshot_store,
            suite=suite,
            model=model,
            encoder=encoder,
            kind_mask=kind_mask,
            wait_mask=wait_mask,
            policy_sha256=task.deployment_policy_sha256,
            artifact_store=ArtifactStore(inputs.root / "artifacts"),
        )
    if (
        report.suite_sha256 != task.suite_sha256
        or report.policy_sha256 != task.deployment_policy_sha256
    ):
        raise ValueError("isolated canonical evaluation produced a foreign report")
    return CanonicalEvaluationTaskResult(
        task.sha256,
        task.suite_sha256,
        task.deployment_policy_sha256,
        report.manifest(),
    )
