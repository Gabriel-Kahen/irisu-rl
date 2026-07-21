"""Typed padded vector execution without JSON serialization."""

from __future__ import annotations

import ctypes
import math
import select
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from numbers import Integral
from os import PathLike
from threading import RLock
from typing import Any

from .env import Action, _action
from .exact_ipc import (
    ExactObservation,
    ExactSimulator,
    _BODY,
    _EVENT,
    _EVENT_COUNT,
    _EVENT_GENERATION,
    _OBSERVATION_HEADER,
    _TRANSITION,
)
from .native import (
    EVENT_DETAIL_CAPACITY,
    NativeError,
    NativeSimulator,
    PADDED_BODY_CAPACITY,
    PaddedBody,
    PaddedEvent,
    PaddedObservation,
    PaddedTransition,
)


class ExactPaddedBody(ctypes.Structure):
    """Packed exact-worker body with the public :class:`PaddedBody` contract."""

    _layout_ = "ms"
    _pack_ = 1
    _fields_ = PaddedBody._fields_
    to_dict = PaddedBody.to_dict


class ExactPaddedObservation(ctypes.Structure):
    """Allocation-light exact observation using the worker's packed wire layout."""

    _layout_ = "ms"
    _pack_ = 1
    _fields_ = [
        ("tick", ctypes.c_uint64),
        ("score", ctypes.c_int64),
        ("gauge", ctypes.c_int64),
        ("gauge_max", ctypes.c_int64),
        ("qualifying_clear_count", ctypes.c_uint64),
        ("field_x", ctypes.c_double),
        ("field_y", ctypes.c_double),
        ("field_width", ctypes.c_double),
        ("field_height", ctypes.c_double),
        ("side_wall_top", ctypes.c_double),
        ("side_wall_bottom", ctypes.c_double),
        ("level", ctypes.c_uint32),
        ("active_colors", ctypes.c_uint32),
        ("spawn_interval_ticks", ctypes.c_uint32),
        ("highest_chain", ctypes.c_uint32),
        ("body_count", ctypes.c_uint32),
        ("terminated", ctypes.c_uint8),
        ("truncated", ctypes.c_uint8),
        ("left_held", ctypes.c_uint8),
        ("right_held", ctypes.c_uint8),
        ("bodies", ExactPaddedBody * PADDED_BODY_CAPACITY),
    ]
    body_mask = PaddedObservation.body_mask
    to_dict = PaddedObservation.to_dict


class ExactPaddedTransition(ctypes.Structure):
    """Exact worker transition with the public typed diagnostics contract."""

    _layout_ = "ms"
    _pack_ = 1
    _fields_ = [
        ("observation", ExactPaddedObservation),
        ("reward", ctypes.c_int64),
        ("event_count", ctypes.c_uint64),
        ("config_hash", ctypes.c_uint64),
        ("finish_call_count", ctypes.c_uint64),
        ("recorded_final_score", ctypes.c_int64),
        ("recorded_final_clears", ctypes.c_uint64),
        ("latest_final_score", ctypes.c_int64),
        ("latest_final_clears", ctypes.c_uint64),
        ("recorded_final_highest_chain", ctypes.c_uint32),
        ("recorded_final_level", ctypes.c_uint32),
        ("latest_final_highest_chain", ctypes.c_uint32),
        ("latest_final_level", ctypes.c_uint32),
        ("terminated", ctypes.c_uint8),
        ("truncated", ctypes.c_uint8),
        ("terminal_metadata_recorded", ctypes.c_uint8),
        ("invalid_action", ctypes.c_uint8),
    ]
    diagnostics = PaddedTransition.diagnostics


if (
    ctypes.sizeof(ExactPaddedBody) != _BODY.size
    or ExactPaddedObservation.bodies.offset != _OBSERVATION_HEADER.size
    or ExactPaddedTransition.reward.offset != ctypes.sizeof(ExactPaddedObservation)
):
    raise RuntimeError("exact padded ctypes layout does not match the worker protocol")


_OBSERVATION_FIELDS = tuple(name for name, _ in ExactPaddedObservation._fields_[:-1])
_BODY_FIELDS = tuple(name for name, _ in ExactPaddedBody._fields_)
_TRANSITION_FIELDS = (
    "reward",
    "event_count",
    "config_hash",
    "finish_call_count",
    "recorded_final_score",
    "recorded_final_clears",
    "latest_final_score",
    "latest_final_clears",
    "recorded_final_highest_chain",
    "recorded_final_level",
    "latest_final_highest_chain",
    "latest_final_level",
    "terminated",
    "truncated",
    "terminal_metadata_recorded",
    "invalid_action",
)


def _copy_exact_observation(
    source: ExactObservation, destination: ExactPaddedObservation
) -> ExactPaddedObservation:
    if len(source.bodies) > PADDED_BODY_CAPACITY:
        raise NativeError("exact observation exceeds padded body capacity")
    for name in _OBSERVATION_FIELDS:
        if name == "body_count":
            setattr(destination, name, len(source.bodies))
        else:
            setattr(destination, name, getattr(source, name))
    for source_body, destination_body in zip(source.bodies, destination.bodies):
        for name in _BODY_FIELDS:
            setattr(
                destination_body,
                name,
                0 if name == "reserved" else getattr(source_body, name),
            )
    return destination


def _decode_exact_transition(
    payload: bytes, destination: ExactPaddedTransition
) -> tuple[ExactPaddedTransition, int]:
    """Copy a prevalidated worker payload into reusable packed typed storage."""

    observation_address = ctypes.addressof(destination.observation)
    ctypes.memmove(observation_address, payload, _OBSERVATION_HEADER.size)
    body_count = int(destination.observation.body_count)
    body_size = body_count * ctypes.sizeof(ExactPaddedBody)
    body_end = _OBSERVATION_HEADER.size + body_size
    if body_size:
        ctypes.memmove(
            observation_address + _OBSERVATION_HEADER.size,
            payload[_OBSERVATION_HEADER.size : body_end],
            body_size,
        )

    values = _TRANSITION.unpack_from(payload, body_end)
    for name, value in zip(_TRANSITION_FIELDS, values):
        setattr(destination, name, value)
    generation = _EVENT_GENERATION.unpack_from(
        payload, body_end + _TRANSITION.size
    )[0]
    return destination, int(generation)


def _decode_exact_events(
    payload: bytes, count: int
) -> ctypes.Array[PaddedEvent]:
    offset = _EVENT_COUNT.size
    event_type = PaddedEvent * count
    events = event_type()
    for output in events:
        event = _EVENT.unpack_from(payload, offset)
        offset += _EVENT.size
        detail_size = event[5]
        detail = payload[offset : offset + detail_size]
        offset += detail_size
        output.tick = event[0]
        output.sequence = event[1]
        output.value = event[2]
        output.a = event[3]
        output.b = event[4]
        output.detail_size = detail_size
        output.kind = event[6]
        output.reserved = 0
        output.detail = detail
    return events


class ExactPaddedEvents(Sequence[PaddedEvent]):
    """Lazy event view tied to one exact worker generation."""

    def __init__(
        self, simulator: ExactSimulator, count: int, generation: int
    ) -> None:
        self._simulator = simulator
        self._client = simulator._client
        self._count = count
        self._generation = generation
        self._cache: ctypes.Array[PaddedEvent] | None = None

    def _values(self) -> ctypes.Array[PaddedEvent]:
        if self._cache is None:
            with self._simulator._lock:
                if self._simulator._client is not self._client:
                    raise NativeError("lazy padded events expired")
                payload = self._simulator.fetch_padded_events_raw(
                    self._generation, self._count
                )
            self._cache = _decode_exact_events(payload, self._count)
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


class PaddedVectorEnv:
    """Run independent worlds in parallel and return fixed-capacity typed state.

    Body slots ``[0, body_count)`` are live and the remaining slots are masked.
    Call ``observation.to_dict()`` only when canonical dictionary materialization
    is needed; training code can read the typed fields and body array directly.
    Returned typed views are double-buffered and should be consumed before the
    next call. Event views are lazy and must be materialized before that lane
    advances again if they need to be retained.
    """

    body_capacity = PADDED_BODY_CAPACITY

    def __init__(
        self,
        num_envs: int,
        *,
        library_path: str | PathLike[str] | None = None,
        config: Mapping[str, Any] | None = None,
        workers: int | None = None,
        physics_backend: str = "portable",
        worker_path: str | PathLike[str] | None = None,
    ) -> None:
        if not isinstance(num_envs, Integral) or isinstance(num_envs, bool) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        if workers is not None and (
            not isinstance(workers, Integral) or isinstance(workers, bool) or workers <= 0
        ):
            raise ValueError("workers must be a positive integer or None")
        if not isinstance(physics_backend, str):
            raise TypeError("physics_backend must be a string")
        aliases = {
            "portable": "portable",
            "exact": "exact",
            "exact-msvc": "exact",
            "exact-msvc9": "exact",
        }
        normalized = physics_backend.strip().lower().replace("_", "-")
        if normalized not in aliases:
            raise ValueError("physics_backend must be 'portable' or 'exact'")
        self._physics_backend = aliases[normalized]
        if self._physics_backend == "portable" and worker_path is not None:
            raise ValueError("worker_path is only valid for physics_backend='exact'")
        if self._physics_backend == "exact" and library_path is not None:
            raise ValueError("library_path is only valid for physics_backend='portable'")
        self._lock = RLock()
        self._envs: tuple[NativeSimulator | ExactSimulator, ...] = ()
        self._num_envs = int(num_envs)
        requested_workers = (
            int(workers) if workers is not None else self._num_envs
        )
        # Portable lanes share one native process and retain the eight-thread
        # ceiling. Exact lanes are independent worker processes, so an explicit
        # request may use every lane; the default remains the conservative cap.
        worker_ceiling = (
            self._num_envs
            if self._physics_backend == "exact" and workers is not None
            else 8
        )
        self._workers = min(self._num_envs, worker_ceiling, requested_workers)
        self._executor: ThreadPoolExecutor | None = None
        self._has_reset = [False] * self._num_envs
        self._config_hashes: tuple[int, ...] = ()
        self._batch_buffers: dict[str, Any] | None = None
        self._batch_library: Any = None
        self._batch_lock_order: tuple[NativeSimulator, ...] = ()
        self._exact_transition_buffers = (
            [(ExactPaddedTransition * self._num_envs)() for _ in range(2)]
            if self._physics_backend == "exact"
            else []
        )
        self._exact_observation_buffers = (
            [(ExactPaddedObservation * self._num_envs)() for _ in range(2)]
            if self._physics_backend == "exact"
            else []
        )
        self._exact_transition_slot = 0
        self._exact_observation_slot = 0
        created: list[NativeSimulator | ExactSimulator] = []
        try:
            for _ in range(self._num_envs):
                if self._physics_backend == "exact":
                    created.append(ExactSimulator(worker_path, config=config))
                else:
                    created.append(NativeSimulator(library_path, config=config))
            self._envs = tuple(created)
            self._config_hashes = tuple(env.config_hash() for env in self._envs)
            if self._physics_backend == "portable":
                portable_envs = tuple(
                    env for env in self._envs if isinstance(env, NativeSimulator)
                )
                self._batch_library, self._batch_lock_order = (
                    NativeSimulator._padded_batch_topology(portable_envs)
                )
            self._executor = ThreadPoolExecutor(
                max_workers=self._workers, thread_name_prefix="irisu-padded"
            )
        except Exception:
            for env in created:
                env.close()
            raise

    @property
    def envs(self) -> tuple[NativeSimulator | ExactSimulator, ...]:
        """The vector's immutable simulator topology."""

        return self._envs

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def workers(self) -> int:
        return self._workers

    @property
    def physics_backend(self) -> str:
        return self._physics_backend

    def _require_open(self) -> ThreadPoolExecutor:
        executor = self._executor
        if executor is None:
            raise RuntimeError("padded vector environment is closed")
        return executor

    def _items(self, values: Sequence[Any], label: str) -> tuple[Any, ...]:
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes, bytearray))
            or len(values) != self._num_envs
        ):
            raise ValueError(f"{label} must contain exactly {self._num_envs} items")
        try:
            return tuple(values[index] for index in range(self._num_envs))
        except IndexError as exc:
            raise ValueError(
                f"{label} must contain exactly {self._num_envs} items"
            ) from exc

    @staticmethod
    def _seed(value: int | None) -> int:
        resolved = 0 if value is None else value
        if not isinstance(resolved, Integral) or isinstance(resolved, bool):
            raise TypeError("seed must be an integer or None")
        result = int(resolved)
        if not 0 <= result <= 0xFFFFFFFF:
            raise ValueError("normal-mode seed must fit in uint32")
        return result

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

    def _pack_exact_observations(
        self, values: Sequence[ExactObservation]
    ) -> list[ExactPaddedObservation]:
        self._exact_observation_slot ^= 1
        buffer = self._exact_observation_buffers[self._exact_observation_slot]
        for source, destination in zip(values, buffer):
            _copy_exact_observation(source, destination)
        return [buffer[index] for index in range(self._num_envs)]

    def _step_exact(
        self, encoded: Sequence[tuple[Any, float, float, int]]
    ) -> tuple[
        list[ExactPaddedTransition],
        list[ExactPaddedEvents],
    ]:
        self._exact_transition_slot ^= 1
        buffer = self._exact_transition_buffers[self._exact_transition_slot]
        transitions: list[ExactPaddedTransition | None] = [None] * self._num_envs
        events: list[ExactPaddedEvents | None] = [None] * self._num_envs
        failures: dict[int, BaseException] = {}

        def receive(index: int, env: ExactSimulator) -> None:
            try:
                payload, generation = env.receive_step_padded_raw()
                transition, decoded_generation = _decode_exact_transition(
                    payload, buffer[index]
                )
                if decoded_generation != generation:
                    raise NativeError("exact padded event generation changed")
                transitions[index] = transition
                events[index] = ExactPaddedEvents(
                    env, int(transition.event_count), generation
                )
            except BaseException as exc:
                failures.setdefault(index, exc)

        # Pipe writes do not block for these small requests. Send a capped wave
        # first, then drain ready responses so a slow low-numbered lane cannot
        # hold completed workers behind it. A fresh poll set costs only a few
        # microseconds at the maximum useful width and remains correct when a
        # reset or restore replaces a lane's worker descriptor.
        for start in range(0, self._num_envs, self._workers):
            stop = min(start + self._workers, self._num_envs)
            # Create the poll set before dispatch. If the platform cannot
            # provide one, no lane is left with an outstanding request.
            poller = select.poll()
            sent: list[tuple[int, ExactSimulator]] = []
            for index in range(start, stop):
                env = self._envs[index]
                assert isinstance(env, ExactSimulator)
                kind, x, y, wait_ticks = encoded[index]
                try:
                    env.send_step_padded(int(kind), x, y, wait_ticks)
                    sent.append((index, env))
                except BaseException as exc:
                    failures.setdefault(index, exc)

            pending: dict[int, tuple[int, ExactSimulator]] = {}
            setup_failed = False
            for index, env in sent:
                try:
                    descriptor = env._pending_response_fd()
                    if descriptor in pending:
                        raise NativeError(
                            "exact padded workers share a response descriptor"
                        )
                    poller.register(
                        descriptor,
                        select.POLLIN
                        | select.POLLERR
                        | select.POLLHUP
                        | select.POLLNVAL,
                    )
                    pending[descriptor] = (index, env)
                except BaseException as exc:
                    failures.setdefault(index, exc)
                    setup_failed = True
                    break

            if setup_failed:
                for index, env in sent:
                    receive(index, env)
                continue

            while pending:
                try:
                    ready = poller.poll()
                    if not ready:
                        raise NativeError(
                            "exact padded response poll returned no ready lanes"
                        )
                except BaseException as exc:
                    failures.setdefault(
                        min(index for index, _ in pending.values()), exc
                    )
                    for index, env in pending.values():
                        receive(index, env)
                    pending.clear()
                    break
                poll_aborted = False
                for descriptor, _ in ready:
                    lane = pending.pop(descriptor, None)
                    if lane is not None:
                        try:
                            poller.unregister(descriptor)
                        except BaseException as exc:
                            failures.setdefault(lane[0], exc)
                            receive(*lane)
                            for index, env in pending.values():
                                receive(index, env)
                            pending.clear()
                            poll_aborted = True
                            break
                        receive(*lane)
                if poll_aborted:
                    break
        if failures:
            raise failures[min(failures)]
        return (
            [value for value in transitions if value is not None],
            [value for value in events if value is not None],
        )

    def reset(
        self, *, seed: int | Sequence[int | None] | None = None
    ) -> tuple[
        list[PaddedObservation | ExactPaddedObservation],
        list[dict[str, Any]],
    ]:
        with self._lock:
            executor = self._require_open()
            if seed is None:
                seeds: Sequence[int | None] = [None] * self._num_envs
            elif isinstance(seed, Integral) and not isinstance(seed, bool):
                seeds = [int(seed) + index for index in range(self._num_envs)]
            else:
                seeds = self._items(seed, "seed")
            try:
                resolved = [self._seed(seeds[index]) for index in range(self._num_envs)]
            except IndexError as exc:
                raise ValueError(
                    f"seed must contain exactly {self._num_envs} items"
                ) from exc
            if self._physics_backend == "exact":
                futures = [
                    executor.submit(env.reset_typed, value)
                    for env, value in zip(self._envs, resolved)
                    if isinstance(env, ExactSimulator)
                ]
                observations = self._pack_exact_observations(self._drain(futures))
            else:
                futures = [
                    executor.submit(env.reset_padded, value)
                    for env, value in zip(self._envs, resolved)
                    if isinstance(env, NativeSimulator)
                ]
                observations = self._drain(futures)
            self._has_reset = [True] * self._num_envs
            infos = [
                {"seed": value, "config_hash": config_hash}
                for value, config_hash in zip(resolved, self._config_hashes)
            ]
            return observations, infos

    def reset_at(
        self, index: int, *, seed: int | None = None
    ) -> PaddedObservation | ExactPaddedObservation:
        with self._lock:
            self._require_open()
            if not 0 <= index < self._num_envs:
                raise IndexError(index)
            env = self._envs[index]
            if isinstance(env, ExactSimulator):
                source = env.reset_typed(self._seed(seed))
                observation = _copy_exact_observation(
                    source, ExactPaddedObservation()
                )
            else:
                observation = env.reset_padded(self._seed(seed))
            self._has_reset[index] = True
            return observation

    def step(
        self, actions: Sequence[Action | Mapping[str, Any]]
    ) -> tuple[
        list[PaddedObservation | ExactPaddedObservation],
        list[int],
        list[bool],
        list[bool],
        list[dict[str, Any]],
    ]:
        with self._lock:
            self._require_open()
            if not all(self._has_reset):
                raise RuntimeError("reset must be called before step")
            encoded = [_action(action) for action in self._items(actions, "actions")]
            if len(encoded) != self._num_envs:
                raise ValueError(f"actions must contain exactly {self._num_envs} items")
            if any(
                not math.isfinite(x) or not math.isfinite(y)
                for _, x, y, _ in encoded
            ):
                raise ValueError("cursor coordinates must be finite")
            if self._physics_backend == "exact":
                transitions, events = self._step_exact(encoded)
            else:
                portable_envs = tuple(
                    env for env in self._envs if isinstance(env, NativeSimulator)
                )
                results, self._batch_buffers = (
                    NativeSimulator._step_padded_batch_prevalidated(
                        portable_envs,
                        encoded,
                        self._batch_buffers,
                        self._workers,
                        self._batch_library,
                        self._batch_lock_order,
                    )
                )
                transitions = [result[0] for result in results]
                events = [result[1] for result in results]
            observations = [transition.observation for transition in transitions]
            rewards = [int(transition.reward) for transition in transitions]
            terminated = [bool(transition.terminated) for transition in transitions]
            truncated = [bool(transition.truncated) for transition in transitions]
            infos = [
                {
                    "events": lane_events,
                    "invalid_action": bool(transition.invalid_action),
                    "config_hash": int(transition.config_hash),
                    "diagnostics": transition,
                }
                for transition, lane_events in zip(transitions, events)
            ]
            return observations, rewards, terminated, truncated, infos

    def clone_state(self) -> tuple[bytes, ...]:
        with self._lock:
            self._require_open()
            if not all(self._has_reset):
                raise RuntimeError("reset must be called before clone_state")
            return tuple(env.clone_state() for env in self._envs)

    def restore_state(
        self, snapshots: Sequence[bytes]
    ) -> list[PaddedObservation | ExactPaddedObservation]:
        with self._lock:
            executor = self._require_open()
            supplied = self._items(snapshots, "snapshots")
            try:
                snapshots = [supplied[index] for index in range(self._num_envs)]
            except IndexError as exc:
                raise ValueError(
                    f"snapshots must contain exactly {self._num_envs} items"
                ) from exc
            backups = tuple(env.clone_state() for env in self._envs)
            prior_has_reset = self._has_reset.copy()

            def restore(
                env: NativeSimulator | ExactSimulator, snapshot: bytes
            ) -> PaddedObservation | ExactObservation:
                if isinstance(env, ExactSimulator):
                    return env.restore_state_typed(snapshot)
                return env.restore_state_padded(snapshot)

            futures = [
                executor.submit(restore, env, snapshot)
                for env, snapshot in zip(self._envs, snapshots)
            ]
            try:
                observations = self._drain(futures)
            except BaseException:
                rollback_futures = [
                    executor.submit(restore, env, snapshot)
                    for env, snapshot in zip(self._envs, backups)
                ]
                try:
                    self._drain(rollback_futures)
                except BaseException as rollback_error:
                    raise RuntimeError(
                        "padded vector snapshot rollback failed"
                    ) from rollback_error
                self._has_reset = prior_has_reset
                raise
            self._has_reset = [True] * self._num_envs
            if self._physics_backend == "exact":
                observations = self._pack_exact_observations(observations)
            return observations

    def state_hash(self) -> tuple[int, ...]:
        with self._lock:
            self._require_open()
            if not all(self._has_reset):
                raise RuntimeError("reset must be called before state_hash")
            return tuple(env.state_hash() for env in self._envs)

    def close(self) -> None:
        lock = getattr(self, "_lock", None)
        if lock is None:
            return
        with lock:
            executor = getattr(self, "_executor", None)
            if executor is None:
                return
            self._executor = None
            executor.shutdown(wait=True, cancel_futures=True)
            for env in self._envs:
                env.close()
            self._has_reset = [False] * self._num_envs

    def __enter__(self) -> PaddedVectorEnv:
        with self._lock:
            self._require_open()
            return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __copy__(self) -> PaddedVectorEnv:
        raise TypeError("PaddedVectorEnv owns mutable simulator state and cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> PaddedVectorEnv:
        del memo
        raise TypeError("PaddedVectorEnv owns mutable simulator state and cannot be copied")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


FastVectorEnv = PaddedVectorEnv
