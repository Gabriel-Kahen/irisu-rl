"""Preallocated owned core storage for R1 random-rollout smoke tests."""

from __future__ import annotations

from numbers import Integral

import numpy as np

from .encoding import EncodedBatch
from .vector_adapter import MacroTransition


class RolloutBuffer:
    def __init__(self, capacity: int, schema) -> None:
        if (
            not isinstance(capacity, Integral)
            or isinstance(capacity, bool)
            or capacity <= 0
        ):
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.schema = schema
        g, f, b = (
            len(schema.global_features),
            len(schema.body_features),
            schema.capacity,
        )
        self.observations_global = np.zeros((self.capacity, g), dtype=np.float32)
        self.observations_body = np.zeros((self.capacity, b, f), dtype=np.float32)
        self.observations_mask = np.zeros((self.capacity, b), dtype=np.bool_)
        self.action_kind = np.zeros(self.capacity, dtype=np.uint8)
        self.action_wait_ticks = np.zeros(self.capacity, dtype=np.uint32)
        self.action_xy = np.zeros((self.capacity, 2), dtype=np.float32)
        self.raw_reward = np.zeros(self.capacity, dtype=np.int64)
        self.elapsed_ticks = np.zeros(self.capacity, dtype=np.uint32)
        self.terminated = np.zeros(self.capacity, dtype=np.bool_)
        self.truncated = np.zeros(self.capacity, dtype=np.bool_)
        self.macro_interrupted = np.zeros(self.capacity, dtype=np.bool_)
        self.bootstrap_mask = np.zeros(self.capacity, dtype=np.bool_)
        self.trace_mask = np.zeros(self.capacity, dtype=np.bool_)
        self.lane_id = np.zeros(self.capacity, dtype=np.uint32)
        self.episode_id = np.zeros(self.capacity, dtype=np.uint64)
        self.seed = np.zeros(self.capacity, dtype=np.uint32)
        self.config_hash = np.zeros(self.capacity, dtype=np.uint64)
        self.event_count = np.zeros(self.capacity, dtype=np.uint32)
        # R1 keeps episode-ending observations and one per-lane rollout-end
        # batch without duplicating every ordinary next observation. R2 extends
        # this core with recurrent state and PPO-specific training tensors.
        self.final_observations: dict[int, EncodedBatch] = {}
        self.rollout_end_observation: EncodedBatch | None = None
        self.size = 0

    def append(self, transition: MacroTransition) -> int:
        if self.rollout_end_observation is not None:
            raise RuntimeError("cannot append after rollout buffer is sealed")
        if self.size >= self.capacity:
            raise BufferError("rollout buffer is full")
        if transition.observation.schema != self.schema:
            raise ValueError("rollout schema mismatch")
        index = self.size
        self.observations_global[index] = transition.observation.global_features[0]
        self.observations_body[index] = transition.observation.body_features[0]
        self.observations_mask[index] = transition.observation.body_mask[0]
        self.action_kind[index] = int(transition.action.kind)
        self.action_wait_ticks[index] = transition.action.wait_ticks
        self.action_xy[index] = (transition.action.x_norm, transition.action.y_norm)
        self.raw_reward[index] = transition.raw_reward
        self.elapsed_ticks[index] = transition.elapsed_ticks
        self.terminated[index] = transition.terminated
        self.truncated[index] = transition.truncated
        self.macro_interrupted[index] = transition.macro_interrupted
        self.bootstrap_mask[index] = transition.bootstrap_mask
        self.trace_mask[index] = transition.trace_mask
        self.lane_id[index] = transition.lane_id
        self.episode_id[index] = transition.episode_id
        self.seed[index] = transition.seed
        self.config_hash[index] = transition.diagnostics.config_hash
        self.event_count[index] = transition.diagnostics.event_count
        if transition.final_observation is not None:
            self.final_observations[index] = transition.final_observation.row(0)
        self.size += 1
        return index

    def seal(self, next_policy_observation: EncodedBatch) -> None:
        """Retain the per-lane bootstrap batch at a nonterminal rollout cut."""

        if next_policy_observation.schema != self.schema:
            raise ValueError("rollout-end schema mismatch")
        if self.rollout_end_observation is not None:
            raise RuntimeError("rollout buffer is already sealed")
        self.rollout_end_observation = next_policy_observation.copy()

    def clear(self) -> None:
        self.size = 0
        self.final_observations.clear()
        self.rollout_end_observation = None
