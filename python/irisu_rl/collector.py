"""Production recurrent SMDP collection and update-boundary exact resume."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import numpy as np
import torch
from torch import Tensor

from .actions import ActionSpec, SemanticAction
from .checkpoints import (
    capture_rng_state,
    load_checkpoint,
    pack_adapter_checkpoint,
    restore_rng_state,
    save_checkpoint,
    unpack_adapter_checkpoint,
)
from .curriculum import (
    CurriculumCoordinator,
    GateDecision,
    ValidationReport,
    ValidationRequest,
)
from .encoding import EncodedBatch
from .models import RecurrentActorCritic
from .ppo import PPOTrainer, PPOUpdateStats, RecurrentTrainingBatch
from .recurrent_buffer import RecurrentRolloutBuffer
from .rewards import RewardBatch, RewardComposer
from .torch_distribution import ActionTensor, TorchConditionalActionDistribution
from .vector_adapter import MacroTransition, MacroVectorAdapter


@dataclass(frozen=True, slots=True)
class CollectorConfig:
    """Bound one synchronous collection by decisions and simulated ticks."""

    max_decisions: int = 128
    target_simulated_ticks: int | None = None
    gamma_tick: float = 1.0
    lambda_tick: float = 0.99
    version: str = "recurrent-collector-config-v1"

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_decisions, bool)
            or not isinstance(self.max_decisions, Integral)
            or self.max_decisions <= 0
        ):
            raise ValueError("collector decision cap must be a positive integer")
        if self.target_simulated_ticks is not None and (
            isinstance(self.target_simulated_ticks, bool)
            or not isinstance(self.target_simulated_ticks, Integral)
            or self.target_simulated_ticks <= 0
        ):
            raise ValueError("collector tick target must be a positive integer or None")
        if self.gamma_tick != 1.0:
            raise ValueError(
                "R3a requires gamma_tick=1 until event-timed rewards exist"
            )
        if not math.isfinite(self.lambda_tick) or not 0 < self.lambda_tick <= 1:
            raise ValueError("collector lambda_tick must be in (0, 1]")

    def manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DecisionAudit:
    actions: tuple[SemanticAction, ...]
    raw_rewards: tuple[int, ...]
    scaled_raw_rewards: tuple[float, ...]
    shaping_rewards: tuple[float, ...]
    shaping_weight_ppm: tuple[int, ...]
    optimizer_rewards: tuple[float, ...]
    elapsed_ticks: tuple[int, ...]
    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    bootstrap_mask: tuple[bool, ...]
    trace_mask: tuple[bool, ...]
    episode_ids: tuple[int, ...]
    seeds: tuple[int, ...]
    config_hashes: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CollectionAudit:
    decision_rows: int
    transitions: int
    simulated_ticks: int
    tick_target: int | None
    tick_target_overshoot: int
    raw_reward: int
    optimizer_reward: float
    completed_episodes: int
    invalid_actions: int
    decisions: tuple[DecisionAudit, ...]


@dataclass(frozen=True, slots=True)
class CollectedRollout:
    batch: RecurrentTrainingBatch
    audit: CollectionAudit


@dataclass(frozen=True, slots=True)
class TrainingUpdate:
    collection: CollectionAudit
    optimizer: PPOUpdateStats | None
    skipped_reason: str | None = None


def model_state_sha256(model: RecurrentActorCritic) -> str:
    """Hash one exact model state without pickle/container nondeterminism."""

    digest = hashlib.sha256(b"irisu-model-state-v1\0")
    manifest = json.dumps(
        model.manifest(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    digest.update(len(manifest).to_bytes(8, "big"))
    digest.update(manifest)
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(value, Tensor) or value.layout != torch.strided:
            raise TypeError("model state must contain dense tensors")
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {"name": name, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        payload = tensor.view(torch.uint8).reshape(-1).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


class TaskContract(Protocol):
    """Collection-time masks and reward state, external to actor features."""

    def manifest(self) -> Mapping[str, object]: ...

    @property
    def training_allowed(self) -> bool: ...

    @property
    def completed_updates(self) -> int: ...

    def action_masks(self, action_spec: ActionSpec) -> tuple[Tensor, Tensor]: ...

    def rewards(self, transitions: Sequence[MacroTransition]) -> RewardBatch: ...

    def after_transitions(self, transitions: Sequence[MacroTransition]) -> None: ...

    def advance_update(self) -> None: ...

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: Mapping[str, object]) -> None: ...


class ScoreTaskContract:
    """Unmasked raw-score task used for nominal and integration rollouts."""

    version = "score-task-v1"

    def __init__(self, lanes: int, *, reward_scale: float = 1.0) -> None:
        if isinstance(lanes, bool) or not isinstance(lanes, Integral) or lanes <= 0:
            raise ValueError("task lane count must be a positive integer")
        self.lanes = int(lanes)
        self.composer = RewardComposer(reward_scale=reward_scale)
        self._completed_updates = 0

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "lanes": self.lanes,
            "reward": self.composer.manifest(),
        }

    @property
    def training_allowed(self) -> bool:
        return True

    @property
    def completed_updates(self) -> int:
        return self._completed_updates

    def action_masks(self, action_spec: ActionSpec) -> tuple[Tensor, Tensor]:
        return (
            torch.ones((self.lanes, 3), dtype=torch.bool),
            torch.ones((self.lanes, len(action_spec.wait_choices)), dtype=torch.bool),
        )

    def rewards(self, transitions: Sequence[MacroTransition]) -> RewardBatch:
        return self.composer.compose(
            transitions, torch.zeros(self.lanes, dtype=torch.int64)
        )

    def after_transitions(self, transitions: Sequence[MacroTransition]) -> None:
        if len(transitions) != self.lanes:
            raise ValueError("task transition count does not match its lanes")

    def advance_update(self) -> None:
        self._completed_updates += 1

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "lanes": self.lanes,
            "reward_sha256": self.composer.sha256,
            "completed_updates": self._completed_updates,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {"version", "lanes", "reward_sha256", "completed_updates"}
        if set(state) != expected or state["version"] != self.version:
            raise ValueError("score task checkpoint identity mismatch")
        if (
            state["lanes"] != self.lanes
            or state["reward_sha256"] != self.composer.sha256
        ):
            raise ValueError("score task checkpoint configuration mismatch")
        updates = state["completed_updates"]
        if (
            isinstance(updates, bool)
            or not isinstance(updates, Integral)
            or updates < 0
        ):
            raise ValueError("score task update count is malformed")
        self._completed_updates = int(updates)


class CurriculumTaskContract:
    """Same-pool curriculum masks and episode-stable shaping integration.

    Environment initialization remains owned by the adapter/pool. This contract
    therefore rejects curricula spanning more than one environment pool; a
    snapshot initializer must commit recipe assignments before such curricula
    can be mixed.
    """

    version = "curriculum-task-v1"

    def __init__(
        self,
        coordinator: CurriculumCoordinator,
        composer: RewardComposer,
        *,
        capture_events: bool,
    ) -> None:
        if not isinstance(capture_events, bool):
            raise TypeError("capture_events must be boolean")
        pools = {stage.environment_pool for stage in coordinator.spec.stages}
        if len(pools) != 1:
            raise ValueError(
                "one collector can only train stages from one environment pool"
            )
        if composer.requires_events and not capture_events:
            raise ValueError("event-dependent curriculum reward requires event capture")
        self.coordinator = coordinator
        self.composer = composer
        self.capture_events = capture_events
        self.environment_pool = pools.pop()
        pool_recipes = tuple(
            recipe
            for recipe in coordinator.spec.library.recipes
            if recipe.environment_pool == self.environment_pool
        )
        self.expected_config_hash = pool_recipes[0].config_hash

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "curriculum_sha256": self.coordinator.spec.sha256,
            "environment_pool": self.environment_pool,
            "expected_config_hash": self.expected_config_hash,
            "capture_events": self.capture_events,
            "reward": self.composer.manifest(),
        }

    @property
    def training_allowed(self) -> bool:
        return (
            self.coordinator.phase
            not in {"budget_validation", "complete", "budget_exhausted"}
            and not self.coordinator.validation_pending
        )

    @property
    def completed_updates(self) -> int:
        return self.coordinator.completed_updates

    def action_masks(self, action_spec: ActionSpec) -> tuple[Tensor, Tensor]:
        return self.coordinator.action_masks(action_spec)

    def rewards(self, transitions: Sequence[MacroTransition]) -> RewardBatch:
        if any(
            transition.diagnostics.config_hash != self.expected_config_hash
            for transition in transitions
        ):
            raise ValueError("collected transition does not match its environment pool")
        return self.composer.compose(
            transitions, self.coordinator.shaping_weights_ppm()
        )

    def after_transitions(self, transitions: Sequence[MacroTransition]) -> None:
        if len(transitions) != self.coordinator.lanes:
            raise ValueError("curriculum transition count does not match its lanes")
        done = torch.tensor(
            [value.terminated or value.truncated for value in transitions],
            dtype=torch.bool,
        )
        self.coordinator.activate_focus_for_new_episodes(done)

    def advance_update(self) -> None:
        self.coordinator.advance_update()

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "reward_sha256": self.composer.sha256,
            "coordinator": self.coordinator.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            set(state) != {"version", "reward_sha256", "coordinator"}
            or state["version"] != self.version
        ):
            raise ValueError("curriculum task checkpoint identity mismatch")
        if state["reward_sha256"] != self.composer.sha256:
            raise ValueError("curriculum task reward identity mismatch")
        coordinator_state = state["coordinator"]
        if not isinstance(coordinator_state, Mapping):
            raise ValueError("curriculum task coordinator state is malformed")
        self.coordinator.load_state_dict(coordinator_state)


class PolicySampler:
    """Isolate policy sampling from global Torch RNG streams."""

    version = "policy-sampler-v1"
    _lock = threading.Lock()

    def __init__(self, seed: int, *, device: torch.device | str = "cpu") -> None:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("policy sampler seed must be a nonnegative integer")
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("policy sampler supports CPU or CUDA")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA policy sampler requires an available CUDA runtime")
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        self._state = generator.get_state()

    def sample(self, distribution: TorchConditionalActionDistribution) -> ActionTensor:
        if distribution.kind_logits.device != self.device:
            raise ValueError("policy sampler and distribution devices differ")
        devices = (
            []
            if self.device.type == "cpu"
            else [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
        )
        with self._lock, torch.random.fork_rng(devices=devices, enabled=True):
            if self.device.type == "cpu":
                torch.set_rng_state(self._state)
            else:
                torch.cuda.set_rng_state(self._state, self.device)
            actions = distribution.sample()
            self._state = (
                torch.get_rng_state()
                if self.device.type == "cpu"
                else torch.cuda.get_rng_state(self.device)
            )
        return actions

    def state_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "device": str(self.device),
            "state": self._state.clone(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            set(state) != {"version", "device", "state"}
            or state["version"] != self.version
        ):
            raise ValueError("policy sampler checkpoint identity mismatch")
        value = state["state"]
        if state["device"] != str(self.device) or not isinstance(value, Tensor):
            raise ValueError("policy sampler checkpoint device/state mismatch")
        generator = torch.Generator(device=self.device)
        try:
            generator.set_state(value.cpu())
        except RuntimeError as exc:
            raise ValueError("policy sampler RNG state is malformed") from exc
        self._state = generator.get_state()


def _encoded_torch(
    observation: EncodedBatch, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    observation.validate()
    return (
        torch.from_numpy(observation.global_features).to(device).unsqueeze(0),
        torch.from_numpy(observation.body_features).to(device).unsqueeze(0),
        torch.from_numpy(observation.body_mask).to(device).unsqueeze(0),
    )


def _concatenate_encoded(rows: Sequence[EncodedBatch]) -> EncodedBatch:
    if not rows:
        raise ValueError("encoded concatenation requires at least one row")
    schema = rows[0].schema
    if any(row.schema != schema or row.global_features.shape[0] != 1 for row in rows):
        raise ValueError("bootstrap observations must be one-row batches of one schema")
    return EncodedBatch(
        np.concatenate([row.global_features for row in rows]),
        np.concatenate([row.body_features for row in rows]),
        np.concatenate([row.body_mask for row in rows]),
        np.concatenate([row.source_tick for row in rows]),
        np.concatenate([row.health_flags for row in rows]),
        schema,
    )


def _batch_to_device(
    batch: RecurrentTrainingBatch, device: torch.device
) -> RecurrentTrainingBatch:
    """Move one sealed CPU rollout without rebuilding any likelihood values."""

    return RecurrentTrainingBatch(
        batch.global_features.to(device),
        batch.body_features.to(device),
        batch.body_mask.to(device),
        batch.reset_before.to(device),
        batch.initial_state.to(device),
        ActionTensor(
            batch.actions.kind.to(device),
            batch.actions.wait_index.to(device),
            batch.actions.xy.to(device),
        ),
        batch.old_log_prob.to(device),
        batch.old_kind_log_prob.to(device),
        batch.old_wait_log_prob.to(device),
        batch.old_coordinate_log_prob.to(device),
        batch.old_values.to(device),
        batch.advantages.to(device),
        batch.returns.to(device),
        batch.valid.to(device),
        batch.train_mask.to(device),
        batch.kind_mask.to(device),
        batch.wait_mask.to(device),
    )


class RecurrentCollector:
    """Collect complete semantic decisions without breaking recurrent history."""

    version = "recurrent-collector-v1"

    def __init__(
        self,
        model: RecurrentActorCritic,
        adapter: MacroVectorAdapter,
        task: TaskContract,
        *,
        config: CollectorConfig | None = None,
        policy_sampler_seed: int,
    ) -> None:
        if adapter.num_envs <= 0:
            raise ValueError("collector adapter must contain at least one lane")
        declared_capture = getattr(task, "capture_events", adapter.capture_events)
        if declared_capture != adapter.capture_events:
            raise ValueError(
                "task event-capture declaration disagrees with the actual adapter"
            )
        composer = getattr(task, "composer", None)
        if getattr(composer, "requires_events", False) and not adapter.capture_events:
            raise ValueError("event-dependent reward requires adapter event capture")
        self.model = model
        self.adapter = adapter
        self.task = task
        self.config = config or CollectorConfig()
        self.lanes = adapter.num_envs
        parameter = next(model.parameters())
        self.device = parameter.device
        self.sampler = PolicySampler(policy_sampler_seed, device=self.device)
        self._initialized = False
        self._poisoned = False
        self._collecting = False
        self._recurrent_state: Tensor | None = None
        self._reset_before: Tensor | None = None
        self.completed_updates = 0
        self.decision_rows = 0
        self.simulated_ticks = 0

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> EncodedBatch:
        if self._poisoned:
            raise RuntimeError("poisoned collector must be recreated")
        if self._initialized:
            raise RuntimeError("collector is already initialized")
        try:
            observation = self.adapter.reset()
            if observation.schema != self.model.schema:
                raise ValueError("adapter observation schema does not match the model")
        except BaseException:
            self._poisoned = True
            raise
        self._recurrent_state = self.model.initial_state(self.lanes).detach()
        self._reset_before = torch.ones(
            self.lanes, dtype=torch.bool, device=self.device
        )
        self._initialized = True
        return observation

    def _require_ready(self) -> tuple[Tensor, Tensor]:
        if self._poisoned:
            raise RuntimeError("poisoned collector must be recreated")
        if (
            not self._initialized
            or self._recurrent_state is None
            or self._reset_before is None
        ):
            raise RuntimeError("collector must be initialized before use")
        if self._collecting:
            raise RuntimeError("collector is already collecting")
        return self._recurrent_state, self._reset_before

    def collect(self) -> CollectedRollout:
        incoming, reset_before = self._require_ready()
        self._collecting = True
        prior_mode = self.model.training
        self.model.eval()
        observation = self.adapter.current_observation
        buffer = RecurrentRolloutBuffer(
            self.config.max_decisions,
            self.lanes,
            observation.schema,
            incoming,
            action_spec=self.model.action_spec,
            reward_scale=1.0,
        )
        bootstrap_rows: list[Tensor] = []
        pending_live_bootstrap: Tensor | None = None
        audits: list[DecisionAudit] = []
        collected_ticks = 0
        raw_total = 0
        optimizer_total = 0.0
        completed_episodes = 0
        try:
            for _ in range(self.config.max_decisions):
                kind_mask, wait_mask = self.task.action_masks(self.model.action_spec)
                expected_wait = (self.lanes, len(self.model.action_spec.wait_choices))
                if kind_mask.shape != (self.lanes, 3) or kind_mask.dtype != torch.bool:
                    raise ValueError("task kind mask must be boolean [B, 3]")
                if wait_mask.shape != expected_wait or wait_mask.dtype != torch.bool:
                    raise ValueError("task wait mask does not match the action schema")
                global_features, body_features, body_mask = _encoded_torch(
                    observation, self.device
                )
                device_kind = kind_mask.to(self.device).unsqueeze(0)
                device_wait = wait_mask.to(self.device).unsqueeze(0)
                with torch.no_grad():
                    output = self.model(
                        global_features,
                        body_features,
                        body_mask,
                        incoming,
                        reset_before=reset_before.unsqueeze(0),
                    )
                    distribution = TorchConditionalActionDistribution(
                        output.kind_logits,
                        output.wait_logits,
                        output.coordinate_alpha,
                        output.coordinate_beta,
                        spec=self.model.action_spec,
                        kind_mask=device_kind,
                        wait_mask=device_wait,
                    )
                    tensor_actions = self.sampler.sample(distribution)
                    components = distribution.log_prob_components(tensor_actions)
                    log_prob = components.total
                sampled_kind = tensor_actions.kind[0].detach().cpu()
                sampled_wait = tensor_actions.wait_index[0].detach().cpu()
                sampled_xy = tensor_actions.xy[0].detach().cpu()
                if pending_live_bootstrap is not None:
                    bootstrap_rows[-1][pending_live_bootstrap.cpu()] = (
                        output.values[0, pending_live_bootstrap].detach().cpu()
                    )
                    pending_live_bootstrap = None
                semantic = tuple(
                    self.model.action_spec.decode(
                        int(sampled_kind[lane]),
                        int(sampled_wait[lane]),
                        float(sampled_xy[lane, 0]),
                        float(sampled_xy[lane, 1]),
                    )
                    for lane in range(self.lanes)
                )
                transitions = self.adapter.step(semantic)
                rewards = self.task.rewards(transitions)
                rewards.validate(self.lanes, reward_scale=self._reward_scale())

                bootstrap_values = torch.zeros(self.lanes, dtype=torch.float32)
                bootstrap_mask_values = torch.tensor(
                    [value.bootstrap_mask for value in transitions], dtype=torch.bool
                )
                trace_mask_values = torch.tensor(
                    [value.trace_mask for value in transitions], dtype=torch.bool
                )
                retained_final = bootstrap_mask_values & ~trace_mask_values
                self._evaluate_bootstrap_subset(
                    transitions,
                    retained_final,
                    output.recurrent_state,
                    bootstrap_values,
                )
                buffer.append(
                    observation,
                    transitions,
                    log_prob[0].detach(),
                    output.values[0].detach(),
                    old_log_prob_components=type(components)(
                        components.kind[0].detach(),
                        components.wait[0].detach(),
                        components.coordinates[0].detach(),
                    ),
                    reset_before=reset_before.detach(),
                    optimizer_reward=rewards.optimizer_reward,
                    kind_mask=kind_mask,
                    wait_mask=wait_mask,
                )
                bootstrap_rows.append(bootstrap_values)
                done = torch.tensor(
                    [value.terminated or value.truncated for value in transitions],
                    dtype=torch.bool,
                    device=self.device,
                )
                self.task.after_transitions(transitions)
                elapsed = tuple(value.elapsed_ticks for value in transitions)
                row_ticks = sum(elapsed)
                collected_ticks += row_ticks
                raw_total += int(rewards.raw_reward.sum())
                optimizer_total += float(rewards.optimizer_reward.sum())
                completed_episodes += int(done.sum())
                audits.append(
                    DecisionAudit(
                        semantic,
                        tuple(int(value) for value in rewards.raw_reward.tolist()),
                        tuple(
                            float(value) for value in rewards.scaled_raw_reward.tolist()
                        ),
                        tuple(
                            float(value) for value in rewards.shaping_reward.tolist()
                        ),
                        tuple(
                            int(value) for value in rewards.shaping_weight_ppm.tolist()
                        ),
                        tuple(
                            float(value) for value in rewards.optimizer_reward.tolist()
                        ),
                        elapsed,
                        tuple(value.terminated for value in transitions),
                        tuple(value.truncated for value in transitions),
                        tuple(value.bootstrap_mask for value in transitions),
                        tuple(value.trace_mask for value in transitions),
                        tuple(value.episode_id for value in transitions),
                        tuple(value.seed for value in transitions),
                        tuple(value.diagnostics.config_hash for value in transitions),
                    )
                )
                incoming = output.recurrent_state.detach()
                reset_before = done
                observation = self.adapter.current_observation
                should_stop = (
                    self.config.target_simulated_ticks is not None
                    and collected_ticks >= self.config.target_simulated_ticks
                ) or buffer.size >= self.config.max_decisions
                if should_stop:
                    self._evaluate_bootstrap_subset(
                        transitions,
                        trace_mask_values,
                        output.recurrent_state,
                        bootstrap_values,
                    )
                else:
                    pending_live_bootstrap = trace_mask_values.to(self.device)
                if should_stop:
                    break

            batch = _batch_to_device(
                buffer.finalize(
                    torch.stack(bootstrap_rows),
                    gamma_tick=self.config.gamma_tick,
                    lambda_tick=self.config.lambda_tick,
                ),
                self.device,
            )
            self._recurrent_state = incoming
            self._reset_before = reset_before
            self.decision_rows += buffer.size
            self.simulated_ticks += collected_ticks
            target = self.config.target_simulated_ticks
            overshoot = 0 if target is None else max(0, collected_ticks - target)
            audit = CollectionAudit(
                buffer.size,
                buffer.size * self.lanes,
                collected_ticks,
                target,
                overshoot,
                raw_total,
                optimizer_total,
                completed_episodes,
                0,
                tuple(audits),
            )
            return CollectedRollout(batch, audit)
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._collecting = False
            self.model.train(prior_mode)

    def _evaluate_bootstrap_subset(
        self,
        transitions: Sequence[MacroTransition],
        mask: Tensor,
        recurrent_state: Tensor,
        destination: Tensor,
    ) -> None:
        """Shadow-evaluate only values unavailable from the next policy pass."""

        indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        if not indices:
            return
        observation = _concatenate_encoded(
            [transitions[index].transition_next_observation for index in indices]
        )
        global_features, body_features, body_mask = _encoded_torch(
            observation, self.device
        )
        device_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        with torch.no_grad():
            output = self.model(
                global_features,
                body_features,
                body_mask,
                recurrent_state[:, device_indices],
                reset_before=torch.zeros(
                    (1, len(indices)), dtype=torch.bool, device=self.device
                ),
            )
        destination[torch.tensor(indices, dtype=torch.long)] = (
            output.values[0].detach().cpu()
        )

    def _reward_scale(self) -> float:
        composer = getattr(self.task, "composer", None)
        scale = getattr(composer, "reward_scale", None)
        if not isinstance(scale, float):
            raise TypeError("task must expose a RewardComposer with a reward scale")
        return scale

    def mark_update_complete(self) -> None:
        self._require_ready()
        self.completed_updates += 1

    def state_dict(self) -> dict[str, object]:
        incoming, reset_before = self._require_ready()
        return {
            "version": self.version,
            "config": self.config.manifest(),
            "model_manifest": self.model.manifest(),
            "lanes": self.lanes,
            "recurrent_state": incoming.detach().cpu().clone(),
            "reset_before": reset_before.detach().cpu().clone(),
            "completed_updates": self.completed_updates,
            "decision_rows": self.decision_rows,
            "simulated_ticks": self.simulated_ticks,
            "sampler": self.sampler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if self._collecting or self._poisoned:
            raise RuntimeError("collector cannot restore in its current state")
        expected = {
            "version",
            "config",
            "model_manifest",
            "lanes",
            "recurrent_state",
            "reset_before",
            "completed_updates",
            "decision_rows",
            "simulated_ticks",
            "sampler",
        }
        if set(state) != expected or state["version"] != self.version:
            raise ValueError("collector checkpoint identity mismatch")
        if (
            state["config"] != self.config.manifest()
            or state["model_manifest"] != self.model.manifest()
            or state["lanes"] != self.lanes
        ):
            raise ValueError("collector checkpoint configuration mismatch")
        recurrent = state["recurrent_state"]
        reset = state["reset_before"]
        expected_state = self.model.initial_state(self.lanes).shape
        if (
            not isinstance(recurrent, Tensor)
            or recurrent.shape != expected_state
            or recurrent.dtype != next(self.model.parameters()).dtype
            or recurrent.device.type != "cpu"
            or not torch.isfinite(recurrent).all()
            or not isinstance(reset, Tensor)
            or reset.shape != (self.lanes,)
            or reset.dtype != torch.bool
            or reset.device.type != "cpu"
        ):
            raise ValueError("collector recurrent/reset state is malformed")
        counters = (
            state["completed_updates"],
            state["decision_rows"],
            state["simulated_ticks"],
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 0
            for value in counters
        ):
            raise ValueError("collector counters are malformed")
        sampler_state = state["sampler"]
        if not isinstance(sampler_state, Mapping):
            raise ValueError("collector sampler state is malformed")
        self.sampler.load_state_dict(sampler_state)
        self._recurrent_state = recurrent.to(self.device).detach().clone()
        self._reset_before = reset.to(self.device).detach().clone()
        self.completed_updates = int(counters[0])
        self.decision_rows = int(counters[1])
        self.simulated_ticks = int(counters[2])
        self._initialized = True


class R3ATrainingSession:
    """One clean-boundary collect/update/checkpoint state machine."""

    version = "r3a-training-session-v1"

    def __init__(
        self,
        collector: RecurrentCollector,
        trainer: PPOTrainer,
        *,
        numpy_seed: int,
        max_consecutive_skips: int = 64,
    ) -> None:
        if collector.model is not trainer.model:
            raise ValueError("collector and trainer must share one model instance")
        if (
            isinstance(max_consecutive_skips, bool)
            or not isinstance(max_consecutive_skips, Integral)
            or max_consecutive_skips <= 0
        ):
            raise ValueError("maximum consecutive skips must be a positive integer")
        self.collector = collector
        self.trainer = trainer
        self.model = trainer.model
        self.task = collector.task
        self.numpy_generator = np.random.default_rng(numpy_seed)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.attempted_rollouts = 0
        self.skipped_rollouts = 0
        self.consecutive_skips = 0
        self._busy = False
        self._poisoned = False
        self._clean_collection_counters: tuple[int, int, int] | None = None

    @property
    def poisoned(self) -> bool:
        return (
            self._poisoned or self.collector.poisoned or self.collector.adapter.poisoned
        )

    def initialize(self) -> EncodedBatch:
        if self._busy or self.poisoned:
            raise RuntimeError("training session cannot initialize")
        observation = self.collector.initialize()
        self._mark_clean_collection_boundary()
        return observation

    def run_update(self) -> TrainingUpdate:
        if self._busy or self.poisoned:
            raise RuntimeError("training session is not at a clean update boundary")
        if not self.task.training_allowed:
            raise RuntimeError("training is currently closed by the task")
        self._validate_clean_collection_boundary()
        self._validate_update_clocks()
        activating = (
            isinstance(self.task, CurriculumTaskContract)
            and self.task.coordinator.phase == "activation"
        )
        if self.consecutive_skips >= self.max_consecutive_skips:
            raise RuntimeError("consecutive skipped-rollout safety limit is exhausted")
        if (
            not activating
            and self.trainer.schedule.completed_updates
            >= self.trainer.schedule.total_updates
        ):
            raise RuntimeError("PPO update budget is exhausted")
        self._busy = True
        try:
            rollout = self.collector.collect()
            self.attempted_rollouts += 1
            if activating:
                return self._finish_skipped_rollout(
                    rollout, "curriculum stage activation drain"
                )
            if not torch.any(rollout.batch.train_mask):
                # A fully censored held-shot truncation is a valid environment
                # outcome. Preserve the advanced rollout state but do not
                # fabricate an optimizer/update-clock step.
                return self._finish_skipped_rollout(
                    rollout, "rollout contained no trainable decisions"
                )
            self.model.train()
            stats = self.trainer.update(rollout.batch)
            self.task.advance_update()
            self.collector.mark_update_complete()
            self.consecutive_skips = 0
            self._mark_clean_collection_boundary()
            return TrainingUpdate(rollout.audit, stats)
        except BaseException:
            self._poisoned = True
            raise
        finally:
            self._busy = False

    def _finish_skipped_rollout(
        self, rollout: CollectedRollout, reason: str
    ) -> TrainingUpdate:
        self.skipped_rollouts += 1
        activation_completed = (
            reason == "curriculum stage activation drain"
            and isinstance(self.task, CurriculumTaskContract)
            and self.task.coordinator.phase != "activation"
        )
        self.consecutive_skips = (
            0 if activation_completed else self.consecutive_skips + 1
        )
        self._mark_clean_collection_boundary()
        return TrainingUpdate(rollout.audit, None, reason)

    def _identity(self, identity: Mapping[str, object]) -> dict[str, object]:
        if "r3a_payload" in identity:
            raise ValueError("checkpoint identity uses reserved r3a_payload key")
        return {**dict(identity), "r3a_payload": self.version}

    def _validate_update_clocks(self) -> None:
        clocks = {
            self.collector.completed_updates,
            self.trainer.schedule.completed_updates,
            self.task.completed_updates,
        }
        if len(clocks) != 1:
            raise ValueError("trainer, collector, and task update clocks disagree")
        if self.attempted_rollouts != (
            self.skipped_rollouts + self.collector.completed_updates
        ):
            raise ValueError("session rollout and update counters disagree")

    def _validate_pending_policy(self) -> None:
        if isinstance(self.task, CurriculumTaskContract):
            request = self.task.coordinator.pending_validation_request
            if request is not None and request.policy_sha256 != model_state_sha256(
                self.model
            ):
                raise ValueError(
                    "pending validation request does not match the loaded model state"
                )

    @property
    def policy_sha256(self) -> str:
        if self._busy or self.poisoned:
            raise RuntimeError("policy identity requires a clean, healthy session")
        return model_state_sha256(self.model)

    def request_validation(
        self, *, evaluator_identity_sha256: str
    ) -> ValidationRequest:
        """Atomically bind validation to the current frozen model state."""

        if not isinstance(self.task, CurriculumTaskContract):
            raise TypeError("validation requests require a curriculum task")
        if self._busy or self.poisoned:
            raise RuntimeError("validation requires a clean, healthy session")
        self._validate_clean_collection_boundary()
        self._validate_update_clocks()
        return self.task.coordinator.request_validation(
            policy_sha256=model_state_sha256(self.model),
            evaluator_identity_sha256=evaluator_identity_sha256,
        )

    def record_validation(self, report: ValidationReport) -> GateDecision:
        """Accept evidence only while the requested model remains loaded."""

        if not isinstance(self.task, CurriculumTaskContract):
            raise TypeError("validation reports require a curriculum task")
        if self._busy or self.poisoned:
            raise RuntimeError("validation requires a clean, healthy session")
        self._validate_clean_collection_boundary()
        self._validate_update_clocks()
        if report.policy_sha256 != model_state_sha256(self.model):
            raise ValueError("validation report does not match the loaded model state")
        return self.task.coordinator.record_validation(report)

    def _mark_clean_collection_boundary(self) -> None:
        self._clean_collection_counters = (
            self.collector.decision_rows,
            self.collector.simulated_ticks,
            self.collector.adapter.mutation_generation,
        )

    def _validate_clean_collection_boundary(self) -> None:
        current = (
            self.collector.decision_rows,
            self.collector.simulated_ticks,
            self.collector.adapter.mutation_generation,
        )
        if (
            self._clean_collection_counters is None
            or current != self._clean_collection_counters
        ):
            raise RuntimeError(
                "collector has an unconsumed rollout outside the training session"
            )

    def save(
        self,
        root: str | Path,
        generation: str,
        *,
        identity: Mapping[str, object],
    ) -> Path:
        if self._busy or self.poisoned:
            raise RuntimeError("checkpoint requires a clean, healthy update boundary")
        self._validate_clean_collection_boundary()
        self._validate_update_clocks()
        self._validate_pending_policy()
        adapter_state, blobs = pack_adapter_checkpoint(
            self.collector.adapter.checkpoint()
        )
        state = {
            "version": self.version,
            "session": {
                "max_consecutive_skips": self.max_consecutive_skips,
                "attempted_rollouts": self.attempted_rollouts,
                "skipped_rollouts": self.skipped_rollouts,
                "consecutive_skips": self.consecutive_skips,
            },
            "model": copy.deepcopy(self.model.state_dict()),
            "trainer": self.trainer.state_dict(),
            "collector": self.collector.state_dict(),
            "task": self.task.state_dict(),
            "adapter": adapter_state,
            "rng": capture_rng_state(self.numpy_generator),
        }
        return save_checkpoint(
            root,
            generation,
            identity=self._identity(identity),
            state=state,
            blobs=blobs,
        )

    def restore(
        self,
        root: str | Path,
        *,
        identity: Mapping[str, object],
        generation: str | None = None,
    ) -> None:
        if self._busy or self.poisoned or self.collector.initialized:
            raise RuntimeError("restore requires a fresh training session")
        state, blobs, _ = load_checkpoint(
            root,
            generation=generation,
            expected_identity=self._identity(identity),
        )
        expected = {
            "version",
            "session",
            "model",
            "trainer",
            "collector",
            "task",
            "adapter",
            "rng",
        }
        if set(state) != expected or state["version"] != self.version:
            raise ValueError("training-session checkpoint identity mismatch")
        session_state = state["session"]
        if not isinstance(session_state, Mapping) or set(session_state) != {
            "max_consecutive_skips",
            "attempted_rollouts",
            "skipped_rollouts",
            "consecutive_skips",
        }:
            raise ValueError("training-session skip state is malformed")
        skip_values = tuple(session_state.values())
        if (
            any(
                isinstance(value, bool) or not isinstance(value, Integral) or value < 0
                for value in skip_values
            )
            or session_state["max_consecutive_skips"] != self.max_consecutive_skips
            or session_state["attempted_rollouts"] < session_state["skipped_rollouts"]
            or session_state["consecutive_skips"] > session_state["skipped_rollouts"]
            or session_state["consecutive_skips"] > self.max_consecutive_skips
        ):
            raise ValueError("training-session skip state is malformed")
        try:
            self.collector.adapter.reset()
            adapter_checkpoint = unpack_adapter_checkpoint(
                state["adapter"],
                blobs,
                schema=self.model.schema,
                action_spec=self.model.action_spec,
            )
            self.model.load_state_dict(state["model"], strict=True)
            self.trainer.load_state_dict(state["trainer"])
            self.task.load_state_dict(state["task"])
            self.collector.adapter.restore_checkpoint(adapter_checkpoint)
            self.collector.load_state_dict(state["collector"])
            self.attempted_rollouts = int(session_state["attempted_rollouts"])
            self.skipped_rollouts = int(session_state["skipped_rollouts"])
            self.consecutive_skips = int(session_state["consecutive_skips"])
            self._validate_update_clocks()
            self._validate_pending_policy()
            restore_rng_state(state["rng"], self.numpy_generator)
            self._mark_clean_collection_boundary()
        except BaseException:
            self._poisoned = True
            raise
