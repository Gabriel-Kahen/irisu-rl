"""Barrier-free exact rollout collection over independent worker lanes."""

from __future__ import annotations

import ctypes
import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from numbers import Integral
from os import PathLike
from threading import RLock
from typing import Any

from .env import Action, _action
from .exact_ipc import ExactSimulator
from .native import NativeError, PaddedEvent
from .padded import (
    ExactPaddedObservation,
    ExactPaddedTransition,
    _copy_exact_observation,
    _decode_exact_events,
    _decode_exact_transition,
    _default_exact_workers,
)


Policy = Callable[[ExactPaddedObservation], Action | Mapping[str, Any]]


class ExactRolloutEvents(Sequence[PaddedEvent]):
    """A rollout-owned event batch decoded only when details are accessed."""

    def __init__(self, count: int, payload: bytes | None) -> None:
        self._count = count
        self._payload = payload
        self._cache: ctypes.Array[PaddedEvent] | None = None

    def _values(self) -> ctypes.Array[PaddedEvent]:
        if self._cache is None:
            if self._count and self._payload is None:
                raise NativeError(
                    "event details were not retained; use event_mode='full'"
                )
            if self._payload is None:
                self._cache = (PaddedEvent * 0)()
            else:
                self._cache = _decode_exact_events(self._payload, self._count)
        return self._cache

    def __len__(self) -> int:
        return self._count

    def __getitem__(self, index: int | slice) -> PaddedEvent | list[PaddedEvent]:
        values = self._values()
        if isinstance(index, slice):
            return [values[position] for position in range(*index.indices(self._count))]
        return values[index]

    def __iter__(self):
        values = self._values()
        return (values[index] for index in range(self._count))

    def materialize(self) -> tuple[PaddedEvent, ...]:
        return tuple(self)


@dataclass(frozen=True)
class ExactRolloutStep:
    """One validated padded transition and its rollout-owned lazy events."""

    action: Action
    payload: bytes
    event_generation: int
    events: ExactRolloutEvents

    @cached_property
    def transition(self) -> ExactPaddedTransition:
        transition, generation = _decode_exact_transition(
            self.payload, ExactPaddedTransition()
        )
        if generation != self.event_generation:
            raise NativeError("exact rollout event generation changed")
        return transition

    @property
    def observation(self) -> ExactPaddedObservation:
        return self.transition.observation

    @property
    def reward(self) -> int:
        return int(self.transition.reward)

    @property
    def terminated(self) -> bool:
        return bool(self.transition.terminated)

    @property
    def truncated(self) -> bool:
        return bool(self.transition.truncated)


@dataclass(frozen=True)
class ExactLaneRollout:
    """A lane-ordered exact trajectory segment."""

    lane: int
    initial_observation: ExactPaddedObservation
    steps: tuple[ExactRolloutStep, ...]
    final_state_hash: int


class ExactActorRolloutPool:
    """Collect independent exact lanes without a barrier at every decision.

    The existing worker frames and one-world-per-process lifecycle are unchanged.
    Results and failures are always joined in lane order. A failed collection may
    have committed a prefix on every lane, matching the worker's forward-only
    semantics. Reset or restore can realign lanes after policy/action failures;
    a transport or protocol failure can close a worker and requires recreating
    the pool.

    ``event_mode='count'`` keeps the fast StepPadded contract and retains only
    event counts. ``event_mode='full'`` fetches each event batch before that lane
    advances, but defers Python decoding until the caller accesses the batch.
    A lane stops at its first terminated or truncated transition and produces
    empty rollouts until :meth:`reset_at` starts its next episode.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        worker_path: str | PathLike[str] | None = None,
        config: Mapping[str, Any] | None = None,
        workers: int | None = None,
        event_mode: str = "count",
    ) -> None:
        if (
            not isinstance(num_envs, Integral)
            or isinstance(num_envs, bool)
            or num_envs <= 0
        ):
            raise ValueError("num_envs must be a positive integer")
        if workers is not None and (
            not isinstance(workers, Integral)
            or isinstance(workers, bool)
            or workers <= 0
        ):
            raise ValueError("workers must be a positive integer or None")
        normalized_mode = event_mode.strip().lower() if isinstance(event_mode, str) else ""
        if normalized_mode not in ("count", "full"):
            raise ValueError("event_mode must be 'count' or 'full'")

        self._lock = RLock()
        self._num_envs = int(num_envs)
        self._workers = min(
            self._num_envs,
            int(workers)
            if workers is not None
            else _default_exact_workers(self._num_envs),
        )
        self._event_mode = normalized_mode
        self._envs: tuple[ExactSimulator, ...] = ()
        self._executor: ThreadPoolExecutor | None = None
        self._has_reset = False
        self._lane_has_reset = [False] * self._num_envs
        self._done = [False] * self._num_envs
        self._buffers = [ExactPaddedTransition() for _ in range(self._num_envs)]

        created: list[ExactSimulator] = []
        try:
            for _ in range(self._num_envs):
                created.append(ExactSimulator(worker_path, config=config))
            self._envs = tuple(created)
            self._executor = ThreadPoolExecutor(
                max_workers=self._workers,
                thread_name_prefix="irisu-exact-actor",
            )
        except BaseException:
            for env in created:
                env.close()
            raise

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def workers(self) -> int:
        return self._workers

    @property
    def event_mode(self) -> str:
        return self._event_mode

    def _require_open(self) -> ThreadPoolExecutor:
        if self._executor is None:
            raise RuntimeError("exact actor rollout pool is closed")
        return self._executor

    def _items(self, values: Sequence[Any], label: str) -> tuple[Any, ...]:
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes, bytearray))
            or len(values) != self._num_envs
        ):
            raise ValueError(f"{label} must contain exactly {self._num_envs} items")
        return tuple(values[index] for index in range(self._num_envs))

    @staticmethod
    def _seed(value: int | None) -> int:
        resolved = 0 if value is None else value
        if not isinstance(resolved, Integral) or isinstance(resolved, bool):
            raise TypeError("seed must be an integer or None")
        seed = int(resolved)
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("normal-mode seed must fit in uint32")
        return seed

    def _seeds(self, seed: int | Sequence[int | None] | None) -> tuple[int, ...]:
        if seed is None:
            supplied: Sequence[int | None] = [None] * self._num_envs
        elif isinstance(seed, Integral) and not isinstance(seed, bool):
            supplied = [int(seed) + index for index in range(self._num_envs)]
        else:
            supplied = self._items(seed, "seed")
        return tuple(self._seed(value) for value in supplied)

    @staticmethod
    def _drain(futures: Sequence[Future[Any]]) -> list[Any]:
        results: list[Any] = []
        failure: BaseException | None = None
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as exc:
                results.append(None)
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure
        return results

    @staticmethod
    def _copy_observation(
        source: ExactPaddedObservation,
    ) -> ExactPaddedObservation:
        destination = ExactPaddedObservation()
        ctypes.memmove(
            ctypes.addressof(destination),
            ctypes.addressof(source),
            ctypes.sizeof(destination),
        )
        return destination

    def reset(
        self, *, seed: int | Sequence[int | None] | None = None
    ) -> list[ExactPaddedObservation]:
        with self._lock:
            executor = self._require_open()
            seeds = self._seeds(seed)
            try:
                observations = self._drain(
                    [
                        executor.submit(env.reset_typed, lane_seed)
                        for env, lane_seed in zip(self._envs, seeds)
                    ]
                )
            except BaseException:
                # Some lanes may already have committed their fresh worker.
                # Refuse rollout until every lane is reset again or a complete
                # snapshot restore realigns the pool.
                self._has_reset = False
                self._lane_has_reset = [False] * self._num_envs
                raise
            for source, buffer in zip(observations, self._buffers):
                _copy_exact_observation(source, buffer.observation)
            self._has_reset = True
            self._lane_has_reset = [True] * self._num_envs
            self._done = [False] * self._num_envs
            return [buffer.observation for buffer in self._buffers]

    def reset_at(self, lane: int, *, seed: int | None = None) -> ExactPaddedObservation:
        """Start one lane's next episode in a fresh identity-matched worker."""

        with self._lock:
            self._require_open()
            if not isinstance(lane, Integral) or isinstance(lane, bool):
                raise TypeError("lane must be an integer")
            lane = int(lane)
            if not 0 <= lane < self._num_envs:
                raise IndexError(lane)
            source = self._envs[lane].reset_typed(self._seed(seed))
            destination = self._buffers[lane].observation
            _copy_exact_observation(source, destination)
            self._done[lane] = False
            self._lane_has_reset[lane] = True
            self._has_reset = all(self._lane_has_reset)
            return destination

    @staticmethod
    def _policy_action(
        policy: Policy | Any, observation: ExactPaddedObservation
    ) -> Action:
        act = getattr(policy, "act", None)
        value = act(observation) if callable(act) else policy(observation)
        kind, x, y, wait_ticks = _action(value)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("cursor coordinates must be finite")
        return Action(kind, x, y, wait_ticks)

    def _collect_lane(self, lane: int, policy: Policy | Any, horizon: int) -> ExactLaneRollout:
        env = self._envs[lane]
        buffer = self._buffers[lane]
        initial_observation = self._copy_observation(buffer.observation)
        steps: list[ExactRolloutStep] = []
        if self._done[lane]:
            return ExactLaneRollout(
                lane, initial_observation, (), env.state_hash()
            )
        for _ in range(horizon):
            action = self._policy_action(policy, buffer.observation)
            env.send_step_padded(
                int(action.kind),
                action.cursor_x,
                action.cursor_y,
                action.wait_ticks,
            )
            payload, generation = env.receive_step_padded_raw()
            transition, decoded_generation = _decode_exact_transition(payload, buffer)
            if generation != decoded_generation:
                raise NativeError("exact rollout event generation changed")
            done = bool(transition.terminated or transition.truncated)
            if done:
                # The step is already committed even if optional event fetching
                # fails, so preserve the terminal state before another request.
                self._done[lane] = True
            count = int(transition.event_count)
            event_payload = (
                env.fetch_padded_events_raw(generation, count)
                if self._event_mode == "full" and count
                else None
            )
            steps.append(
                ExactRolloutStep(
                    action,
                    payload,
                    generation,
                    ExactRolloutEvents(count, event_payload),
                )
            )
            if done:
                break
        return ExactLaneRollout(
            lane, initial_observation, tuple(steps), env.state_hash()
        )

    def collect(
        self,
        policies: Sequence[Policy | Any],
        horizon: int,
    ) -> tuple[ExactLaneRollout, ...]:
        """Collect ``horizon`` consecutive decisions independently per lane."""

        if (
            not isinstance(horizon, Integral)
            or isinstance(horizon, bool)
            or horizon <= 0
        ):
            raise ValueError("horizon must be a positive integer")
        with self._lock:
            executor = self._require_open()
            if not self._has_reset:
                raise RuntimeError("reset must be called before collect")
            lane_policies = self._items(policies, "policies")
            if len({id(policy) for policy in lane_policies}) != self._num_envs:
                raise ValueError(
                    "policies must be distinct lane-local objects; shared mutable "
                    "policy state is schedule-dependent"
                )
            futures = [
                executor.submit(self._collect_lane, lane, policy, int(horizon))
                for lane, policy in enumerate(lane_policies)
            ]
            return tuple(self._drain(futures))

    def clone_state(self) -> tuple[bytes, ...]:
        with self._lock:
            self._require_open()
            if not self._has_reset:
                raise RuntimeError("reset must be called before clone_state")
            return tuple(env.clone_state() for env in self._envs)

    def restore_state(
        self, snapshots: Sequence[bytes]
    ) -> list[ExactPaddedObservation]:
        """Restore every lane transactionally after all actor work is quiescent."""

        with self._lock:
            executor = self._require_open()
            supplied = self._items(snapshots, "snapshots")
            backups = tuple(env.clone_state() for env in self._envs)
            futures = [
                executor.submit(env.restore_state_typed, snapshot)
                for env, snapshot in zip(self._envs, supplied)
            ]
            try:
                observations = self._drain(futures)
            except BaseException:
                rollback = [
                    executor.submit(env.restore_state_typed, snapshot)
                    for env, snapshot in zip(self._envs, backups)
                ]
                try:
                    self._drain(rollback)
                except BaseException as rollback_error:
                    raise RuntimeError(
                        "exact actor snapshot rollback failed"
                    ) from rollback_error
                raise
            for source, buffer in zip(observations, self._buffers):
                _copy_exact_observation(source, buffer.observation)
            self._has_reset = True
            self._lane_has_reset = [True] * self._num_envs
            self._done = [
                bool(buffer.observation.terminated or buffer.observation.truncated)
                for buffer in self._buffers
            ]
            return [buffer.observation for buffer in self._buffers]

    def state_hash(self) -> tuple[int, ...]:
        with self._lock:
            self._require_open()
            if not self._has_reset:
                raise RuntimeError("reset must be called before state_hash")
            return tuple(env.state_hash() for env in self._envs)

    def close(self) -> None:
        with self._lock:
            executor = self._executor
            if executor is None:
                return
            executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
            for env in self._envs:
                env.close()

    def __enter__(self) -> ExactActorRolloutPool:
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __copy__(self) -> ExactActorRolloutPool:
        raise TypeError("ExactActorRolloutPool owns mutable workers and cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> ExactActorRolloutPool:
        del memo
        raise TypeError("ExactActorRolloutPool owns mutable workers and cannot be copied")
