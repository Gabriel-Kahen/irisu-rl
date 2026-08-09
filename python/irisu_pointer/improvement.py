"""Offline DAgger collection and recurrent policy-improvement orchestration."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from irisu_env import EventKind
from irisu_env.policies import SplitMix64
from irisu_rl.encoding import TeacherStateEncoder

from .action import PointerActionSpec, PointerActionTensor
from .actor_distill import (
    ActorDistillConfig,
    ActorTeacherStep,
    build_actor_sequence_episode,
    perfect_actor_record,
)
from .experts import PointerExpertDecision, matcher_anchor
from .policy import RecurrentPointerPolicy, target_index_for_decision
from .search import lower_expert_decision
from .sequence import (
    PointerSequenceEpisode,
    PointerSequenceMetrics,
    PointerSequenceTrainer,
    pad_pointer_episodes,
)
from .trajectory import (
    DelayedRewardSpec,
    TrajectoryEpisode,
    TrajectoryProvenance,
    TrajectoryStep,
    discounted_return_targets,
)


class SearchTeacher(Protocol):
    def act(
        self, env: Any, observation: Mapping[str, Any]
    ) -> PointerExpertDecision: ...


FallbackTeacher = Callable[[Mapping[str, Any], PointerActionSpec], PointerExpertDecision]
EnvironmentFactory = Callable[[], Any]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DistributionalLeafEvaluator:
    """Conservative stateless leaf value from the lower critic quantiles."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        lower_quantile_fraction: float = 0.25,
        maximum_absolute_value: float = 5_000.0,
    ) -> None:
        if (
            not 0.0 < float(lower_quantile_fraction) <= 1.0
            or not math.isfinite(float(maximum_absolute_value))
            or maximum_absolute_value <= 0.0
        ):
            raise ValueError("leaf critic risk or clipping configuration is invalid")
        self.model = model
        self.lower_quantile_fraction = float(lower_quantile_fraction)
        self.maximum_absolute_value = float(maximum_absolute_value)
        self.enabled = False
        self._recurrent_state: torch.Tensor | None = None
        self.encoder = TeacherStateEncoder()
        schema = getattr(model, "schema", None)
        if schema is None or schema.sha256 != self.encoder.schema.sha256:
            raise ValueError("leaf critic requires the teacher observation schema")

    def enable(self) -> None:
        self.enabled = True

    def set_recurrent_state(self, state: torch.Tensor | None) -> None:
        if state is None:
            self._recurrent_state = None
            return
        if state.ndim != 3 or state.shape[1] < 1:
            raise ValueError("leaf critic recurrent context is malformed")
        expected = self.model.initial_state(
            int(state.shape[1]), device=state.device
        )
        if (
            state.shape != expected.shape
            or state.dtype != expected.dtype
            or not bool(torch.isfinite(state).all())
        ):
            raise ValueError("leaf critic recurrent context is malformed")
        self._recurrent_state = state.detach().clone()

    @torch.no_grad()
    def __call__(self, observation: Mapping[str, Any]) -> float:
        if not self.enabled:
            return 0.0
        encoded = self.encoder.encode([observation])
        device = next(self.model.parameters()).device
        active = np.flatnonzero(encoded.body_mask[0])
        width = int(active[-1]) + 1 if active.size else 1
        prior = self.model.training
        self.model.eval()
        try:
            state = (
                self.model.initial_state(1, device=device)
                if self._recurrent_state is None
                else self._recurrent_state.to(device=device)
            )
            if state.shape[1] != 1:
                raise ValueError("leaf critic context must contain one episode lane")
            output = self.model(
                torch.from_numpy(encoded.global_features).to(device).unsqueeze(0),
                torch.from_numpy(encoded.body_features[:, :width])
                .to(device)
                .unsqueeze(0),
                torch.from_numpy(encoded.body_mask[:, :width])
                .to(device)
                .unsqueeze(0),
                state,
            )
        finally:
            self.model.train(prior)
        quantiles = output.value_quantiles[0, 0].sort().values
        count = max(1, math.ceil(quantiles.numel() * self.lower_quantile_fraction))
        value = float(quantiles[:count].mean())
        return min(
            self.maximum_absolute_value,
            max(-self.maximum_absolute_value, value),
        )


@dataclass(frozen=True, slots=True)
class DaggerCollectionConfig:
    max_decisions: int = 512
    teacher_beta_ppm: int = 1_000_000
    uncertainty_threshold: float = 0.55
    search_query_stride: int = 16
    maximum_search_queries: int = 32
    gamma_tick: float = 0.9995
    search_label_weight: float = 8.0

    def __post_init__(self) -> None:
        integers = (
            self.max_decisions,
            self.teacher_beta_ppm,
            self.search_query_stride,
            self.maximum_search_queries,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise TypeError("DAgger collection integer fields must be integers")
        if (
            self.max_decisions < 1
            or not 0 <= self.teacher_beta_ppm <= 1_000_000
            or self.search_query_stride < 1
            or self.maximum_search_queries < 0
        ):
            raise ValueError("DAgger collection integer field is out of range")
        for name, value, lower, upper in (
            ("uncertainty_threshold", self.uncertainty_threshold, 0.0, 1.0),
            ("gamma_tick", self.gamma_tick, 0.0, 1.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not lower < float(value) <= upper
            ):
                raise ValueError(f"{name} must lie in ({lower}, {upper}]")
        if (
            isinstance(self.search_label_weight, bool)
            or not isinstance(self.search_label_weight, (int, float))
            or not math.isfinite(float(self.search_label_weight))
            or float(self.search_label_weight) < 1.0
        ):
            raise ValueError("search label weight must be finite and at least one")


@dataclass(frozen=True, slots=True)
class CollectedPointerEpisode:
    trajectory: TrajectoryEpisode
    supervision: PointerSequenceEpisode
    behavior: PointerSequenceEpisode
    actor_supervision: PointerSequenceEpisode
    search_queries: int
    fallback_labels: int
    policy_actions: int
    teacher_actions: int
    disagreements: int
    collector_cut: bool
    search_selections: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.trajectory.episode_identity != self.supervision.identity
            or self.trajectory.episode_identity != self.behavior.identity
            or self.trajectory.episode_identity != self.actor_supervision.identity
        ):
            raise ValueError("trajectory and supervision episode identities differ")
        counts = (
            self.search_queries,
            self.fallback_labels,
            self.policy_actions,
            self.teacher_actions,
            self.disagreements,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counts):
            raise ValueError("collection counts must be nonnegative integers")
        if (
            len(self.search_selections) != self.search_queries
            or any(not isinstance(value, str) or not value for value in self.search_selections)
        ):
            raise ValueError("search selection audit differs from query count")

    @property
    def final_score(self) -> int:
        return self.trajectory.steps[-1].score_after

    @property
    def failure_events(self) -> int:
        failures = {
            int(EventKind.ROTTEN),
            int(EventKind.INVALID_ACTION),
            int(EventKind.GAME_OVER),
        }
        return sum(
            event.kind in failures
            for step in self.trajectory.steps
            for event in step.events
        )


def _decision_key(value: PointerExpertDecision) -> tuple[int, int, int | None, int]:
    return value.kind, value.wait_ticks, value.target_body_id, value.template_index


def _events_info(events: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    return {"events": tuple(dict(event) for event in events)}


def _execute_decision(
    env: Any,
    observation: Mapping[str, Any],
    decision: PointerExpertDecision,
    spec: PointerActionSpec,
) -> tuple[Mapping[str, Any], bool, bool, dict[str, object]]:
    current = observation
    terminated = bool(current.get("terminated", False))
    truncated = bool(current.get("truncated", False))
    events: list[Mapping[str, Any]] = []
    for action in lower_expert_decision(decision, current, spec):
        if terminated or truncated:
            break
        current, _reward, terminated, truncated, info = env.step(action)
        events.extend(info.get("events", ()))
    return current, terminated, truncated, _events_info(events)


def _sequence_episode(
    identity: str,
    encoded_rows: Sequence[Any],
    labels: Sequence[PointerExpertDecision],
    observations: Sequence[Mapping[str, Any]],
    trajectory: TrajectoryEpisode,
    reward_spec: DelayedRewardSpec,
    spec: PointerActionSpec,
    gamma_tick: float,
    policy_weights: Sequence[float] | None = None,
) -> PointerSequenceEpisode:
    if not (len(encoded_rows) == len(labels) == len(observations) == len(trajectory.steps)):
        raise ValueError("collected observation, label, and transition counts differ")
    target_indices = [
        target_index_for_decision(encoded, observation, decision)
        for encoded, observation, decision in zip(encoded_rows, observations, labels)
    ]
    last_visible = max(
        (
            int(np.flatnonzero(encoded.body_mask[0])[-1]) + 1
            for encoded in encoded_rows
            if bool(encoded.body_mask[0].any())
        ),
        default=1,
    )
    globals_tensor = torch.from_numpy(
        np.concatenate([encoded.global_features for encoded in encoded_rows])
    )
    bodies = np.concatenate(
        [encoded.body_features[:, :last_visible] for encoded in encoded_rows]
    )
    id_column = encoded_rows[0].schema.body_features.index("id_scaled")
    bodies[:, :, id_column] = 0.0
    if np.any(bodies[:, :, id_column]):
        raise RuntimeError("trajectory ID ablation failed")
    body_tensor = torch.from_numpy(np.ascontiguousarray(bodies))
    mask_tensor = torch.from_numpy(
        np.concatenate([encoded.body_mask[:, :last_visible] for encoded in encoded_rows])
    )
    wait_indices = [
        spec.wait_choices.index(decision.wait_ticks)
        if decision.kind == 0
        else 0
        for decision in labels
    ]
    actions = PointerActionTensor(
        torch.tensor([decision.kind for decision in labels], dtype=torch.long),
        torch.tensor(wait_indices, dtype=torch.long),
        torch.tensor(target_indices, dtype=torch.long),
        torch.tensor(
            [decision.template_index for decision in labels],
            dtype=torch.long,
        ),
    )
    returns = discounted_return_targets(
        trajectory,
        reward_spec,
        gamma_tick=gamma_tick,
    )
    return PointerSequenceEpisode(
        identity=identity,
        global_features=globals_tensor,
        body_features=body_tensor,
        body_mask=mask_tensor,
        actions=actions,
        returns=torch.tensor(returns.scalar_returns, dtype=torch.float32),
        schema=encoded_rows[0].schema,
        pointer_spec=spec,
        policy_weight=(
            None
            if policy_weights is None
            else torch.tensor(policy_weights, dtype=torch.float32)
        ),
    )


def collect_dagger_episode(
    env: Any,
    *,
    seed: int,
    episode_identity: str,
    source_revision: str,
    runtime_sha256: str,
    collector_id: str,
    expensive_teacher: SearchTeacher | None = None,
    fallback_teacher: FallbackTeacher = matcher_anchor,
    policy: RecurrentPointerPolicy | None = None,
    pointer_spec: PointerActionSpec | None = None,
    reward_spec: DelayedRewardSpec | None = None,
    actor_distill_config: ActorDistillConfig | None = None,
    config: DaggerCollectionConfig | None = None,
) -> CollectedPointerEpisode:
    """Collect one policy-visited, teacher-labeled episode from a real env."""

    resolved_spec = PointerActionSpec() if pointer_spec is None else pointer_spec
    resolved_reward = DelayedRewardSpec() if reward_spec is None else reward_spec
    resolved = DaggerCollectionConfig() if config is None else config
    observation, reset_info = env.reset(seed=seed)
    provenance = TrajectoryProvenance(
        source_revision=source_revision,
        runtime_sha256=runtime_sha256,
        config_hash=int(reset_info["config_hash"]),
        observation_schema_sha256=TeacherStateEncoder.schema.sha256,
        pointer_spec_sha256=resolved_spec.sha256,
        reward_spec_sha256=resolved_reward.sha256,
        collector_id=collector_id,
    )
    if policy is not None:
        policy.reset(seed)
    rng = SplitMix64(seed ^ 0xDADDA66E)
    encoder = TeacherStateEncoder()
    transitions: list[TrajectoryStep] = []
    encoded_rows: list[Any] = []
    label_observations: list[Mapping[str, Any]] = []
    labels: list[PointerExpertDecision] = []
    behaviors: list[PointerExpertDecision] = []
    search_queries = fallback_labels = policy_actions = teacher_actions = 0
    disagreements = 0
    previous_failure = False
    search_selections: list[str] = []
    label_weights: list[float] = []
    terminated = truncated = False
    for decision_index in range(resolved.max_decisions):
        if terminated or truncated:
            break
        encoded = encoder.encode([observation])
        prediction = policy.predict(observation) if policy is not None else None
        context_setter = getattr(
            expensive_teacher, "set_recurrent_context", None
        )
        if callable(context_setter):
            context_setter(
                None if policy is None else policy.recurrent_state
            )
        periodic = decision_index % resolved.search_query_stride == 0
        uncertain = (
            prediction is not None
            and prediction.confidence < resolved.uncertainty_threshold
        )
        use_search = (
            expensive_teacher is not None
            and search_queries < resolved.maximum_search_queries
            and (periodic or uncertain or previous_failure)
        )
        if use_search:
            search_method = getattr(expensive_teacher, "search", None)
            if callable(search_method):
                result = search_method(env, observation)
                label = result.decision
                search_selections.append(str(result.selected_name))
            else:
                label = expensive_teacher.act(env, observation)
                search_selections.append(type(expensive_teacher).__qualname__)
            search_queries += 1
            label_weights.append(float(resolved.search_label_weight))
        else:
            label = fallback_teacher(observation, resolved_spec)
            fallback_labels += 1
            label_weights.append(1.0)
        if label.kind == 0 and label.wait_ticks not in resolved_spec.wait_choices:
            raise ValueError("teacher emitted a wait outside the pointer vocabulary")
        if not 0 <= label.template_index < resolved_spec.template_count:
            raise ValueError("teacher emitted a template outside the pointer vocabulary")
        if prediction is not None:
            disagreements += int(_decision_key(prediction.decision) != _decision_key(label))
        teacher_behavior = (
            prediction is None
            or rng.bounded(1_000_000) < resolved.teacher_beta_ppm
        )
        behavior = label if teacher_behavior else prediction.decision
        teacher_actions += int(teacher_behavior)
        policy_actions += int(not teacher_behavior)

        before = observation
        observation, terminated, truncated, info = _execute_decision(
            env, before, behavior, resolved_spec
        )
        step = TrajectoryStep.from_public_transition(
            episode_identity=episode_identity,
            provenance_sha256=provenance.sha256,
            decision_index=decision_index,
            decision=behavior,
            before=before,
            after=observation,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )
        transitions.append(step)
        encoded_rows.append(encoded)
        label_observations.append(before)
        labels.append(label)
        behaviors.append(behavior)
        previous_failure = any(
            event.kind
            in {
                int(EventKind.ROTTEN),
                int(EventKind.INVALID_ACTION),
                int(EventKind.GAME_OVER),
            }
            for event in step.events
        )
    if not transitions:
        raise RuntimeError("DAgger collector produced an empty episode")
    trajectory = TrajectoryEpisode(
        episode_identity=episode_identity,
        seed=seed,
        provenance=provenance,
        steps=tuple(transitions),
    )
    supervision = _sequence_episode(
        episode_identity,
        encoded_rows,
        labels,
        label_observations,
        trajectory,
        resolved_reward,
        resolved_spec,
        resolved.gamma_tick,
        label_weights,
    )
    behavior_sequence = _sequence_episode(
        episode_identity,
        encoded_rows,
        behaviors,
        label_observations,
        trajectory,
        resolved_reward,
        resolved_spec,
        resolved.gamma_tick,
    )
    actor_supervision = build_actor_sequence_episode(
        episode_identity,
        tuple(
            ActorTeacherStep(
                perfect_actor_record(observation),
                observation,
                label,
                float(value),
            )
            for observation, label, value in zip(
                label_observations,
                labels,
                supervision.returns.tolist(),
            )
        ),
        config=actor_distill_config,
        pointer_spec=resolved_spec,
    )
    actor_supervision = replace(
        actor_supervision,
        policy_weight=supervision.policy_weight.clone(),
    )
    return CollectedPointerEpisode(
        trajectory=trajectory,
        supervision=supervision,
        behavior=behavior_sequence,
        actor_supervision=actor_supervision,
        search_queries=search_queries,
        fallback_labels=fallback_labels,
        policy_actions=policy_actions,
        teacher_actions=teacher_actions,
        disagreements=disagreements,
        collector_cut=not (terminated or truncated),
        search_selections=tuple(search_selections),
    )


@dataclass(frozen=True, slots=True)
class DaggerLoopConfig:
    teacher_beta_ppm: tuple[int, ...] = (1_000_000, 650_000, 350_000, 100_000)
    sequence_updates_per_iteration: int = 8
    sequence_minibatch_episodes: int = 4
    replay_capacity_episodes: int = 64
    elite_fraction: float = 0.25
    failure_fraction: float = 0.25
    awr_temperature: float = 100.0
    awr_maximum_weight: float = 20.0
    awr_behavior_coefficient: float = 0.5

    def __post_init__(self) -> None:
        if (
            not self.teacher_beta_ppm
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 1_000_000
                for value in self.teacher_beta_ppm
            )
            or any(
                right > left
                for left, right in zip(
                    self.teacher_beta_ppm, self.teacher_beta_ppm[1:]
                )
            )
        ):
            raise ValueError("teacher beta schedule must be nonincreasing ppm")
        if (
            isinstance(self.sequence_updates_per_iteration, bool)
            or not isinstance(self.sequence_updates_per_iteration, int)
            or self.sequence_updates_per_iteration < 1
            or isinstance(self.sequence_minibatch_episodes, bool)
            or not isinstance(self.sequence_minibatch_episodes, int)
            or self.sequence_minibatch_episodes < 1
            or isinstance(self.replay_capacity_episodes, bool)
            or not isinstance(self.replay_capacity_episodes, int)
            or self.replay_capacity_episodes < 2
        ):
            raise ValueError("DAgger update and replay counts must be positive")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in (self.elite_fraction, self.failure_fraction)
        ):
            raise ValueError("replay fractions must lie in [0, 1]")
        if self.elite_fraction + self.failure_fraction > 1.0:
            raise ValueError("elite and failure replay fractions exceed one")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
            for value in (
                self.awr_temperature,
                self.awr_maximum_weight,
                self.awr_behavior_coefficient,
            )
        ) or self.awr_maximum_weight < 1.0:
            raise ValueError("AWR configuration must be finite and positive")


def advantage_weighted_behavior(
    episodes: Sequence[CollectedPointerEpisode],
    *,
    temperature: float,
    maximum_weight: float,
    coefficient: float,
) -> tuple[PointerSequenceEpisode, ...]:
    """Weight logged behavior by delayed return while retaining teacher BC."""

    supplied = tuple(episodes)
    if not supplied:
        raise ValueError("AWR requires collected behavior episodes")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
        for value in (temperature, maximum_weight, coefficient)
    ) or maximum_weight < 1.0:
        raise ValueError("AWR configuration must be finite and positive")
    returns = torch.cat([value.behavior.returns for value in supplied])
    baseline = returns.median()
    maximum_log = math.log(float(maximum_weight))
    output: list[PointerSequenceEpisode] = []
    for value in supplied:
        advantage = (value.behavior.returns - baseline) / float(temperature)
        weights = torch.exp(
            advantage.clamp(min=-maximum_log, max=maximum_log)
        ).clamp_max(float(maximum_weight))
        weights = weights * float(coefficient)
        output.append(replace(value.behavior, policy_weight=weights))
    return tuple(output)


def prioritized_replay(
    episodes: Sequence[CollectedPointerEpisode],
    *,
    capacity: int,
    elite_fraction: float,
    failure_fraction: float,
) -> tuple[CollectedPointerEpisode, ...]:
    supplied = tuple(episodes)
    if capacity >= len(supplied):
        return supplied
    elite_count = int(round(capacity * elite_fraction))
    failure_count = int(round(capacity * failure_fraction))
    selected: dict[str, CollectedPointerEpisode] = {}

    def add(values: Sequence[CollectedPointerEpisode], count: int) -> None:
        for value in values:
            if len(selected) >= capacity or count <= 0:
                break
            identity = value.trajectory.episode_identity
            if identity not in selected:
                selected[identity] = value
                count -= 1

    add(sorted(supplied, key=lambda value: (-value.final_score, value.trajectory.episode_identity)), elite_count)
    add(
        sorted(
            supplied,
            key=lambda value: (
                -value.failure_events,
                value.final_score,
                value.trajectory.episode_identity,
            ),
        ),
        failure_count,
    )
    add(tuple(reversed(supplied)), capacity - len(selected))
    return tuple(selected.values())


def _spread_indices(count: int, slots: int) -> tuple[int, ...]:
    if slots == 0:
        return ()
    if slots == 1:
        return (count // 2,)
    if slots <= count:
        return tuple(
            index * (count - 1) // (slots - 1)
            for index in range(slots)
        )
    return tuple(index % count for index in range(slots))


def stratified_replay_minibatches(
    supervision: Sequence[PointerSequenceEpisode],
    behavior: Sequence[PointerSequenceEpisode],
    *,
    updates: int,
    batch_size: int,
) -> tuple[tuple[PointerSequenceEpisode, ...], ...]:
    """Interleave both replay pools while spreading limited slots end to end."""

    teacher = tuple(supervision)
    awr = tuple(behavior)
    if (
        not teacher
        or len(teacher) != len(awr)
        or any(
            left.identity != right.identity
            for left, right in zip(teacher, awr)
        )
    ):
        raise ValueError("stratified replay requires aligned nonempty pools")
    if (
        isinstance(updates, bool)
        or not isinstance(updates, int)
        or updates < 1
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= len(teacher) + len(awr)
    ):
        raise ValueError("stratified replay update or minibatch count is invalid")

    total_slots = updates * batch_size
    teacher_order = _spread_indices(len(teacher), (total_slots + 1) // 2)
    awr_order = _spread_indices(len(awr), total_slots // 2)
    teacher_cursor = awr_cursor = 0
    schedule: list[PointerSequenceEpisode] = []
    for slot in range(total_slots):
        if slot % 2 == 0:
            schedule.append(teacher[teacher_order[teacher_cursor]])
            teacher_cursor += 1
        else:
            schedule.append(awr[awr_order[awr_cursor]])
            awr_cursor += 1
    return tuple(
        tuple(schedule[start : start + batch_size])
        for start in range(0, total_slots, batch_size)
    )


def spread_sequence_minibatches(
    episodes: Sequence[PointerSequenceEpisode],
    *,
    updates: int,
    batch_size: int,
) -> tuple[tuple[PointerSequenceEpisode, ...], ...]:
    """Spread a bounded update budget across one episode pool."""

    supplied = tuple(episodes)
    if (
        not supplied
        or isinstance(updates, bool)
        or not isinstance(updates, int)
        or updates < 1
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= len(supplied)
    ):
        raise ValueError("sequence replay update or minibatch count is invalid")
    order = _spread_indices(len(supplied), updates * batch_size)
    return tuple(
        tuple(supplied[index] for index in order[start : start + batch_size])
        for start in range(0, len(order), batch_size)
    )


@dataclass(frozen=True, slots=True)
class DaggerIterationMetrics:
    iteration: int
    beta_ppm: int
    collected_episodes: int
    replay_episodes: int
    decisions: int
    search_queries: int
    policy_actions: int
    teacher_actions: int
    disagreements: int
    mean_score: float
    maximum_score: int
    mean_behavior_weight: float
    maximum_behavior_weight: float
    continuation_critic_enabled_for_next_wave: bool
    sequence_metric_scope: str
    search_selection_counts: Mapping[str, int]
    sequence: PointerSequenceMetrics

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["sequence"] = asdict(self.sequence)
        return value


class DaggerPolicyImprover:
    """Accumulate policy-visited episodes and update a recurrent pointer model."""

    def __init__(
        self,
        model: torch.nn.Module,
        trainer: PointerSequenceTrainer,
        *,
        source_revision: str,
        runtime_sha256: str,
        expensive_teacher: SearchTeacher | None = None,
        fallback_teacher: FallbackTeacher = matcher_anchor,
        pointer_spec: PointerActionSpec | None = None,
        reward_spec: DelayedRewardSpec | None = None,
        actor_distill_config: ActorDistillConfig | None = None,
        collection_config: DaggerCollectionConfig | None = None,
        loop_config: DaggerLoopConfig | None = None,
        leaf_evaluator: DistributionalLeafEvaluator | None = None,
    ) -> None:
        if trainer.model is not model:
            raise ValueError("DAgger trainer must own the supplied model")
        self.model = model
        self.trainer = trainer
        self.source_revision = source_revision
        self.runtime_sha256 = runtime_sha256
        self.expensive_teacher = expensive_teacher
        self.fallback_teacher = fallback_teacher
        self.pointer_spec = PointerActionSpec() if pointer_spec is None else pointer_spec
        self.reward_spec = DelayedRewardSpec() if reward_spec is None else reward_spec
        self.actor_distill_config = (
            ActorDistillConfig()
            if actor_distill_config is None
            else actor_distill_config
        )
        self.collection_config = (
            DaggerCollectionConfig() if collection_config is None else collection_config
        )
        self.loop_config = DaggerLoopConfig() if loop_config is None else loop_config
        if leaf_evaluator is not None and leaf_evaluator.model is not model:
            raise ValueError("leaf evaluator must use the improved model")
        self.leaf_evaluator = leaf_evaluator
        self.replay: list[CollectedPointerEpisode] = []
        self.policy = RecurrentPointerPolicy(model, pointer_spec=self.pointer_spec)

    def run(
        self,
        env_factory: EnvironmentFactory,
        seed_waves: Sequence[Sequence[int]],
    ) -> tuple[DaggerIterationMetrics, ...]:
        waves = tuple(tuple(values) for values in seed_waves)
        if len(waves) != len(self.loop_config.teacher_beta_ppm):
            raise ValueError("seed waves must match the DAgger beta schedule")
        metrics: list[DaggerIterationMetrics] = []
        seen_seeds: set[int] = set()
        for iteration, (beta, seeds) in enumerate(
            zip(self.loop_config.teacher_beta_ppm, waves)
        ):
            if not seeds:
                raise ValueError("each DAgger iteration needs at least one seed")
            if seen_seeds.intersection(seeds):
                raise ValueError("DAgger seed waves must be disjoint")
            seen_seeds.update(seeds)
            collection = replace(self.collection_config, teacher_beta_ppm=beta)
            current: list[CollectedPointerEpisode] = []
            for seed in seeds:
                with env_factory() as env:
                    episode = collect_dagger_episode(
                        env,
                        seed=seed,
                        episode_identity=f"dagger-{iteration:02d}-seed-{seed}",
                        source_revision=self.source_revision,
                        runtime_sha256=self.runtime_sha256,
                        collector_id="r3c-policy-improvement-v1",
                        expensive_teacher=self.expensive_teacher,
                        fallback_teacher=self.fallback_teacher,
                        policy=None if iteration == 0 else self.policy,
                        pointer_spec=self.pointer_spec,
                        reward_spec=self.reward_spec,
                        actor_distill_config=self.actor_distill_config,
                        config=collection,
                    )
                current.append(episode)
            self.replay.extend(current)
            selected = prioritized_replay(
                self.replay,
                capacity=self.loop_config.replay_capacity_episodes,
                elite_fraction=self.loop_config.elite_fraction,
                failure_fraction=self.loop_config.failure_fraction,
            )
            weighted_behavior = advantage_weighted_behavior(
                selected,
                temperature=self.loop_config.awr_temperature,
                maximum_weight=self.loop_config.awr_maximum_weight,
                coefficient=self.loop_config.awr_behavior_coefficient,
            )
            minibatch_size = min(
                self.loop_config.sequence_minibatch_episodes,
                2 * len(selected),
            )
            sequence_metrics: PointerSequenceMetrics | None = None
            minibatches = stratified_replay_minibatches(
                [value.supervision for value in selected],
                weighted_behavior,
                updates=self.loop_config.sequence_updates_per_iteration,
                batch_size=minibatch_size,
            )
            for sequences in minibatches:
                sequence_metrics = self.trainer.step(
                    pad_pointer_episodes(sequences)
                )
            assert sequence_metrics is not None
            if self.leaf_evaluator is not None:
                self.leaf_evaluator.enable()
            scores = [value.final_score for value in current]
            behavior_weights = torch.cat(
                [
                    value.policy_weight
                    for value in weighted_behavior
                    if value.policy_weight is not None
                ]
            )
            metrics.append(
                DaggerIterationMetrics(
                    iteration=iteration,
                    beta_ppm=beta,
                    collected_episodes=len(current),
                    replay_episodes=len(selected),
                    decisions=sum(len(value.trajectory.steps) for value in current),
                    search_queries=sum(value.search_queries for value in current),
                    policy_actions=sum(value.policy_actions for value in current),
                    teacher_actions=sum(value.teacher_actions for value in current),
                    disagreements=sum(value.disagreements for value in current),
                    mean_score=sum(scores) / len(scores),
                    maximum_score=max(scores),
                    mean_behavior_weight=float(behavior_weights.mean()),
                    maximum_behavior_weight=float(behavior_weights.max()),
                    continuation_critic_enabled_for_next_wave=(
                        self.leaf_evaluator is not None
                        and self.leaf_evaluator.enabled
                    ),
                    sequence_metric_scope=(
                        "last_deterministic_stratified_episode_minibatch"
                    ),
                    search_selection_counts=dict(
                        sorted(
                            Counter(
                                name
                                for value in current
                                for name in value.search_selections
                            ).items()
                        )
                    ),
                    sequence=sequence_metrics,
                )
            )
        return tuple(metrics)


__all__ = [
    "CollectedPointerEpisode",
    "DaggerCollectionConfig",
    "DaggerIterationMetrics",
    "DaggerLoopConfig",
    "DaggerPolicyImprover",
    "DistributionalLeafEvaluator",
    "SearchTeacher",
    "collect_dagger_episode",
    "file_sha256",
    "advantage_weighted_behavior",
    "prioritized_replay",
    "stratified_replay_minibatches",
]
