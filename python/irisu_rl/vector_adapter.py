"""Synchronous SMDP action macros over the active-lane padded vector API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .actions import ActionSpec, SemanticAction, SemanticActionKind
from .encoding import EncodedBatch
from .schema import TensorSchema
from .seeds import SeedAllocator


class Encoder(Protocol):
    def encode(self, observations: Sequence[object]) -> EncodedBatch: ...


@dataclass(frozen=True, slots=True)
class ObservationInput:
    lane_id: int
    phase: str
    raw_observation: object
    semantic_action: SemanticAction | None
    episode_reset: bool


@dataclass(frozen=True, slots=True)
class OwnedDiagnostics:
    config_hash: int
    invalid_action: bool
    event_count: int


@dataclass(frozen=True, slots=True)
class MacroTransition:
    lane_id: int
    episode_id: int
    seed: int
    observation: EncodedBatch
    action: SemanticAction
    primitive_trace: tuple[str, ...]
    raw_reward: int
    elapsed_ticks: int
    start_tick: int
    end_tick: int
    terminated: bool
    truncated: bool
    macro_interrupted: bool
    transition_next_observation: EncodedBatch
    final_observation: EncodedBatch | None
    next_policy_observation: EncodedBatch
    bootstrap_mask: bool
    trace_mask: bool
    diagnostics: OwnedDiagnostics


@dataclass(frozen=True, slots=True)
class AdapterCheckpoint:
    version: str
    schema_sha256: str
    action_sha256: str
    num_envs: int
    current: EncodedBatch
    raw_ticks: tuple[int, ...]
    raw_scores: tuple[int, ...]
    seeds: tuple[int, ...]
    episode_ids: tuple[int, ...]
    seed_allocator: dict[str, int | str]
    snapshots: tuple[bytes, ...]
    state_hashes: tuple[int, ...]


class MacroVectorAdapter:
    """Complete one legal semantic macro per lane and return owned tensors.

    The first primitive is full-width and batched. Only shot lanes then execute
    a concurrent active-lane neutral release, so wait lanes never receive dummy
    transitions and every public call ends at a policy/update boundary.
    """

    def __init__(
        self,
        env: Any,
        *,
        encoder: Encoder,
        observation_transform: Callable[[Sequence[ObservationInput]], Sequence[object]]
        | None = None,
        seed_allocator: SeedAllocator | None = None,
        action_spec: ActionSpec | None = None,
    ) -> None:
        self.env = env
        self.encoder = encoder
        self.observation_transform = observation_transform
        self.seed_allocator = seed_allocator or SeedAllocator()
        self.action_spec = action_spec or ActionSpec()
        self.num_envs = int(env.num_envs)
        self._poisoned = False
        self._initialized = False
        self._current: EncodedBatch | None = None
        self._schema: TensorSchema | None = None
        self._raw_ticks = [0] * self.num_envs
        self._raw_scores = [0] * self.num_envs
        self._seeds = [0] * self.num_envs
        self._episode_ids = [0] * self.num_envs

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    @property
    def current_observation(self) -> EncodedBatch:
        if not self._initialized or self._current is None:
            raise RuntimeError("adapter must be reset first")
        return self._current.copy()

    def _encode(
        self,
        observations: Sequence[object],
        *,
        lane_ids: Sequence[int],
        phase: str,
        actions: Sequence[SemanticAction | None] | None = None,
    ) -> EncodedBatch:
        if len(observations) != len(lane_ids):
            raise ValueError("observation and lane-id counts differ")
        semantic_actions = (None,) * len(observations) if actions is None else actions
        if len(semantic_actions) != len(observations):
            raise ValueError("observation and action counts differ")
        inputs = tuple(
            ObservationInput(int(lane_id), phase, observation, action, phase == "reset")
            for lane_id, observation, action in zip(
                lane_ids, observations, semantic_actions
            )
        )
        values = (
            self.observation_transform(inputs)
            if self.observation_transform is not None
            else observations
        )
        if len(values) != len(observations):
            raise ValueError("observation transform changed batch length")
        encoded = self.encoder.encode(values)
        encoded.validate()
        if encoded.global_features.shape[0] != len(observations):
            raise ValueError("encoder changed batch length")
        if self._schema is not None and encoded.schema != self._schema:
            raise ValueError("encoder changed schema during a rollout")
        return encoded

    @staticmethod
    def _require_lengths(expected: int, *values: Sequence[object]) -> None:
        if any(len(value) != expected for value in values):
            raise ValueError("backend returned a malformed lane batch")

    def _fail_closed(self, exc: BaseException) -> None:
        self._poisoned = True
        raise RuntimeError(
            "vector coordinator is poisoned after a possibly partial backend operation"
        ) from exc

    def reset(self) -> EncodedBatch:
        if self._poisoned:
            raise RuntimeError("poisoned adapter must be recreated")
        reservation = self.seed_allocator.reserve(self.num_envs)
        try:
            observations, infos = self.env.reset(seed=reservation.seeds)
            self._require_lengths(self.num_envs, observations, infos)
            encoded = self._encode(
                observations, lane_ids=range(self.num_envs), phase="reset"
            )
            if any(
                int(info.get("seed", seed)) != seed
                for info, seed in zip(infos, reservation.seeds)
            ):
                raise RuntimeError("backend reset did not honor explicit seeds")
        except BaseException as exc:
            self._fail_closed(exc)
        self.seed_allocator.commit(reservation)
        if self._schema is None:
            self._schema = encoded.schema
        self._current = encoded
        self._raw_ticks = [int(getattr(value, "tick", 0)) for value in observations]
        self._raw_scores = [int(getattr(value, "score", 0)) for value in observations]
        self._seeds = list(reservation.seeds)
        self._episode_ids = [0] * self.num_envs
        self._initialized = True
        return encoded.copy()

    def checkpoint(self) -> AdapterCheckpoint:
        """Capture a complete clean-boundary coordinator/environment state."""

        if self._poisoned:
            raise RuntimeError("cannot checkpoint a poisoned adapter")
        if not self._initialized or self._current is None or self._schema is None:
            raise RuntimeError("adapter must be reset before checkpointing")
        if self.observation_transform is not None:
            raise RuntimeError(
                "stateful observation transforms need an explicit checkpoint contract"
            )
        snapshots = tuple(self.env.clone_state())
        state_hashes = tuple(int(value) for value in self.env.state_hash())
        self._require_lengths(self.num_envs, snapshots, state_hashes)
        return AdapterCheckpoint(
            "macro-vector-adapter-checkpoint-v1",
            self._schema.sha256,
            self.action_spec.sha256,
            self.num_envs,
            self._current.copy(),
            tuple(self._raw_ticks),
            tuple(self._raw_scores),
            tuple(self._seeds),
            tuple(self._episode_ids),
            self.seed_allocator.state_dict(),
            snapshots,
            state_hashes,
        )

    def restore_checkpoint(self, checkpoint: AdapterCheckpoint) -> EncodedBatch:
        """Restore after a disposable reset and verify observations/state hashes."""

        if self._poisoned:
            raise RuntimeError("cannot restore a poisoned adapter")
        if not self._initialized:
            raise RuntimeError("reset the fresh backend once before checkpoint restore")
        if checkpoint.version != "macro-vector-adapter-checkpoint-v1":
            raise ValueError("adapter checkpoint version mismatch")
        if checkpoint.num_envs != self.num_envs:
            raise ValueError("adapter checkpoint lane count mismatch")
        if checkpoint.action_sha256 != self.action_spec.sha256:
            raise ValueError("adapter checkpoint action identity mismatch")
        checkpoint.current.validate()
        if checkpoint.current.schema.sha256 != checkpoint.schema_sha256:
            raise ValueError("adapter checkpoint schema identity mismatch")
        if self._schema is None or checkpoint.schema_sha256 != self._schema.sha256:
            raise ValueError(
                "adapter checkpoint does not match the fresh encoder schema"
            )
        expected_lengths = (
            checkpoint.raw_ticks,
            checkpoint.raw_scores,
            checkpoint.seeds,
            checkpoint.episode_ids,
            checkpoint.snapshots,
            checkpoint.state_hashes,
        )
        self._require_lengths(self.num_envs, *expected_lengths)
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for values in (
                checkpoint.raw_ticks,
                checkpoint.raw_scores,
                checkpoint.seeds,
                checkpoint.episode_ids,
                checkpoint.state_hashes,
            )
            for value in values
        ):
            raise TypeError(
                "adapter checkpoint metadata must contain canonical integers"
            )
        if any(not isinstance(value, bytes) for value in checkpoint.snapshots):
            raise TypeError("adapter checkpoint snapshots must be owned bytes")
        if not np.array_equal(
            checkpoint.current.source_tick,
            np.asarray(checkpoint.raw_ticks, dtype=np.int64),
        ):
            raise ValueError("checkpoint raw ticks disagree with encoded observations")
        allocator = SeedAllocator(
            self.seed_allocator.split.name, key=self.seed_allocator.key
        )
        allocator.load_state_dict(checkpoint.seed_allocator)
        try:
            observations = self.env.restore_state(checkpoint.snapshots)
            self._require_lengths(self.num_envs, observations)
            restored_ticks = tuple(
                int(getattr(value, "tick", 0)) for value in observations
            )
            restored_scores = tuple(
                int(getattr(value, "score", 0)) for value in observations
            )
            if restored_ticks != checkpoint.raw_ticks:
                raise ValueError(
                    "restored raw ticks do not match checkpoint bookkeeping"
                )
            if restored_scores != checkpoint.raw_scores:
                raise ValueError(
                    "restored raw scores do not match checkpoint bookkeeping"
                )
            encoded = self._encode(
                observations, lane_ids=range(self.num_envs), phase="restore"
            )
            fields = (
                "global_features",
                "body_features",
                "body_mask",
                "source_tick",
                "health_flags",
            )
            if any(
                not np.array_equal(
                    getattr(encoded, field), getattr(checkpoint.current, field)
                )
                for field in fields
            ):
                raise ValueError("restored observation does not match checkpoint")
            state_hashes = tuple(int(value) for value in self.env.state_hash())
            if state_hashes != checkpoint.state_hashes:
                raise ValueError("restored environment state hash mismatch")
        except BaseException as exc:
            self._fail_closed(exc)
        self.seed_allocator.load_state_dict(checkpoint.seed_allocator)
        self._current = encoded
        self._raw_ticks = list(checkpoint.raw_ticks)
        self._raw_scores = list(checkpoint.raw_scores)
        self._seeds = list(checkpoint.seeds)
        self._episode_ids = list(checkpoint.episode_ids)
        self._initialized = True
        return encoded.copy()

    def step(self, actions: Sequence[SemanticAction]) -> tuple[MacroTransition, ...]:
        if self._poisoned:
            raise RuntimeError("poisoned adapter must be recreated")
        if not self._initialized or self._current is None:
            raise RuntimeError("adapter must be reset first")
        if len(actions) != self.num_envs:
            raise ValueError(f"actions must contain exactly {self.num_envs} items")
        # Entire-batch validation happens before any backend mutation.
        validated = tuple(self.action_spec.validate(action) for action in actions)
        start_observations = tuple(
            self._current.row(index) for index in range(self.num_envs)
        )
        start_ticks = tuple(self._raw_ticks)
        start_scores = tuple(self._raw_scores)
        primitive_actions = [self.action_spec.press(action) for action in validated]
        try:
            observations, rewards, terminated, truncated, infos = self.env.step(
                primitive_actions
            )
            self._require_lengths(
                self.num_envs, observations, rewards, terminated, truncated, infos
            )
            first_encoded = self._encode(
                observations,
                lane_ids=range(self.num_envs),
                phase="macro_first",
                actions=validated,
            )
        except BaseException as exc:
            self._fail_closed(exc)

        try:
            final_raw = list(observations)
            final_encoded = [first_encoded.row(index) for index in range(self.num_envs)]
            total_rewards = [int(value) for value in rewards]
            total_events = [len(info.get("events", ())) for info in infos]
            invalid = [bool(info.get("invalid_action", False)) for info in infos]
            config_hashes = [int(info.get("config_hash", 0)) for info in infos]
            final_terminated = [bool(value) for value in terminated]
            final_truncated = [bool(value) for value in truncated]
            traces: list[list[str]] = [
                ["wait" if action.kind is SemanticActionKind.WAIT else "press"]
                for action in validated
            ]
        except BaseException as exc:
            self._fail_closed(exc)

        release_lanes = [
            index
            for index, action in enumerate(validated)
            if action.kind is not SemanticActionKind.WAIT
            and not final_terminated[index]
            and not final_truncated[index]
        ]
        if release_lanes:
            try:
                release_result = self.env.step_many(
                    release_lanes,
                    [self.action_spec.release() for _ in release_lanes],
                )
                (
                    release_observations,
                    release_rewards,
                    release_terminated,
                    release_truncated,
                    release_infos,
                ) = release_result
                self._require_lengths(
                    len(release_lanes),
                    release_observations,
                    release_rewards,
                    release_terminated,
                    release_truncated,
                    release_infos,
                )
                release_encoded = self._encode(
                    release_observations,
                    lane_ids=release_lanes,
                    phase="release",
                    actions=[validated[lane] for lane in release_lanes],
                )
                release_event_counts = [
                    len(info.get("events", ())) for info in release_infos
                ]
                release_invalid = [
                    bool(info.get("invalid_action", False)) for info in release_infos
                ]
                release_hashes = [
                    int(info.get("config_hash", 0)) for info in release_infos
                ]
            except BaseException as exc:
                self._fail_closed(exc)
            for offset, lane in enumerate(release_lanes):
                final_raw[lane] = release_observations[offset]
                final_encoded[lane] = release_encoded.row(offset)
                total_rewards[lane] += int(release_rewards[offset])
                total_events[lane] += release_event_counts[offset]
                invalid[lane] |= release_invalid[offset]
                config_hashes[lane] = release_hashes[offset]
                final_terminated[lane] = bool(release_terminated[offset])
                final_truncated[lane] = bool(release_truncated[offset])
                traces[lane].append("release")

        if any(invalid):
            self._poisoned = True
            raise RuntimeError(
                "validated semantic action produced a native invalid action"
            )

        end_ticks = [int(getattr(value, "tick", 0)) for value in final_raw]
        end_scores = [int(getattr(value, "score", 0)) for value in final_raw]
        elapsed = [end - start for start, end in zip(start_ticks, end_ticks)]
        if any(value <= 0 for value in elapsed):
            self._poisoned = True
            raise RuntimeError(
                "semantic macro did not advance a positive number of ticks"
            )
        if any(
            reward != end - start
            for reward, start, end in zip(total_rewards, start_scores, end_scores)
        ):
            self._poisoned = True
            raise RuntimeError("macro reward does not equal raw score delta")

        done_lanes = [
            index
            for index in range(self.num_envs)
            if final_terminated[index] or final_truncated[index]
        ]
        reset_encoded_by_lane: dict[int, EncodedBatch] = {}
        new_seed_by_lane: dict[int, int] = {}
        reset_tick_by_lane: dict[int, int] = {}
        reset_score_by_lane: dict[int, int] = {}
        if done_lanes:
            reservation = self.seed_allocator.reserve(len(done_lanes))
            try:
                reset_observations = self.env.reset_many(
                    done_lanes, seeds=reservation.seeds
                )
                self._require_lengths(len(done_lanes), reset_observations)
                reset_encoded = self._encode(
                    reset_observations, lane_ids=done_lanes, phase="reset"
                )
            except BaseException as exc:
                self._fail_closed(exc)
            self.seed_allocator.commit(reservation)
            for offset, lane in enumerate(done_lanes):
                reset_encoded_by_lane[lane] = reset_encoded.row(offset)
                new_seed_by_lane[lane] = reservation.seeds[offset]
                reset_tick_by_lane[lane] = int(
                    getattr(reset_observations[offset], "tick", 0)
                )
                reset_score_by_lane[lane] = int(
                    getattr(reset_observations[offset], "score", 0)
                )

        try:
            transitions: list[MacroTransition] = []
            for lane, action in enumerate(validated):
                interrupted = False
                if final_terminated[lane] or final_truncated[lane]:
                    if action.kind is SemanticActionKind.WAIT:
                        interrupted = elapsed[lane] < action.wait_ticks
                    else:
                        interrupted = "release" not in traces[lane]
                episode_done = final_terminated[lane] or final_truncated[lane]
                next_policy = reset_encoded_by_lane.get(lane, final_encoded[lane])
                transitions.append(
                    MacroTransition(
                        lane_id=lane,
                        episode_id=self._episode_ids[lane],
                        seed=self._seeds[lane],
                        observation=start_observations[lane],
                        action=action,
                        primitive_trace=tuple(traces[lane]),
                        raw_reward=total_rewards[lane],
                        elapsed_ticks=elapsed[lane],
                        start_tick=start_ticks[lane],
                        end_tick=end_ticks[lane],
                        terminated=final_terminated[lane],
                        truncated=final_truncated[lane],
                        macro_interrupted=interrupted,
                        transition_next_observation=final_encoded[lane],
                        final_observation=(
                            final_encoded[lane] if episode_done else None
                        ),
                        next_policy_observation=next_policy,
                        bootstrap_mask=(
                            not final_terminated[lane]
                            and (
                                not interrupted
                                or action.kind is SemanticActionKind.WAIT
                            )
                        ),
                        trace_mask=not episode_done,
                        diagnostics=OwnedDiagnostics(
                            config_hashes[lane], invalid[lane], total_events[lane]
                        ),
                    )
                )
            for lane, transition in enumerate(transitions):
                next_policy = transition.next_policy_observation
                self._current.global_features[lane] = next_policy.global_features[0]
                self._current.body_features[lane] = next_policy.body_features[0]
                self._current.body_mask[lane] = next_policy.body_mask[0]
                self._current.source_tick[lane] = next_policy.source_tick[0]
                self._current.health_flags[lane] = next_policy.health_flags[0]
                if transition.terminated or transition.truncated:
                    self._seeds[lane] = new_seed_by_lane[lane]
                    self._episode_ids[lane] += 1
                    self._raw_ticks[lane] = reset_tick_by_lane[lane]
                    self._raw_scores[lane] = reset_score_by_lane[lane]
                else:
                    self._raw_ticks[lane] = end_ticks[lane]
                    self._raw_scores[lane] = end_scores[lane]
        except BaseException as exc:
            self._fail_closed(exc)
        return tuple(transitions)
