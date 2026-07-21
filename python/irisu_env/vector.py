"""Small dependency-free synchronous vector wrapper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from numbers import Integral
from os import PathLike
from typing import Any

from .env import Action, IrisuEnv, _action


class SyncVectorEnv:
    """Run independent native environments sequentially in one process."""

    def __init__(
        self,
        num_envs: int,
        *,
        library_path: str | PathLike[str] | None = None,
        render_mode: str | None = None,
        config: Mapping[str, Any] | None = None,
        physics_backend: str = "portable",
        worker_path: str | PathLike[str] | None = None,
    ) -> None:
        if not isinstance(num_envs, Integral) or isinstance(num_envs, bool) or num_envs <= 0:
            raise ValueError("num_envs must be a positive integer")
        created: list[IrisuEnv] = []
        try:
            for _ in range(int(num_envs)):
                created.append(
                    IrisuEnv(
                        library_path=library_path,
                        render_mode=render_mode,
                        config=config,
                        physics_backend=physics_backend,
                        worker_path=worker_path,
                    )
                )
        except Exception:
            for env in created:
                env.close()
            raise
        self.envs = tuple(created)
        self.num_envs = len(self.envs)

    def _items(self, values: Sequence[Any], label: str) -> tuple[Any, ...]:
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes, bytearray))
            or len(values) != self.num_envs
        ):
            raise ValueError(f"{label} must contain exactly {self.num_envs} items")
        try:
            return tuple(values[index] for index in range(self.num_envs))
        except IndexError as exc:
            raise ValueError(
                f"{label} must contain exactly {self.num_envs} items"
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

    def _seeds(
        self, seed: int | Sequence[int | None] | None
    ) -> tuple[int, ...]:
        if seed is None:
            supplied: Sequence[int | None] = [None] * self.num_envs
        elif isinstance(seed, Integral) and not isinstance(seed, bool):
            supplied = [int(seed) + index for index in range(self.num_envs)]
        else:
            supplied = self._items(seed, "seed")
        return tuple(self._seed(value) for value in supplied)

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        seeds = self._seeds(seed)
        results = [env.reset(seed=value, options=options) for env, value in zip(self.envs, seeds)]
        return [value[0] for value in results], [value[1] for value in results]

    def step(
        self, actions: Sequence[Action | Mapping[str, Any]]
    ) -> tuple[
        list[dict[str, Any]],
        list[int],
        list[bool],
        list[bool],
        list[dict[str, Any]],
    ]:
        supplied = self._items(actions, "actions")
        actions = tuple(Action(*_action(action)) for action in supplied)
        results = [env.step(action) for env, action in zip(self.envs, actions)]
        observations, rewards, terminated, truncated, infos = zip(*results)
        return (
            list(observations),
            list(rewards),
            list(terminated),
            list(truncated),
            list(infos),
        )

    def clone_state(self) -> tuple[bytes, ...]:
        return tuple(env.clone_state() for env in self.envs)

    def restore_state(self, snapshots: Sequence[bytes]) -> list[dict[str, Any]]:
        snapshots = self._items(snapshots, "snapshots")
        backups = tuple(
            (env._native.clone_state(), env._has_reset) for env in self.envs
        )
        try:
            return [
                env.restore_state(snapshot)
                for env, snapshot in zip(self.envs, snapshots)
            ]
        except BaseException:
            try:
                for env, (snapshot, had_reset) in zip(self.envs, backups):
                    env._native.restore_state(snapshot)
                    env._has_reset = had_reset
            except BaseException as rollback_error:
                raise RuntimeError("vector snapshot rollback failed") from rollback_error
            raise

    def state_hash(self) -> tuple[int, ...]:
        return tuple(env.state_hash() for env in self.envs)

    def render(self, mode: str | None = None) -> tuple[str, ...]:
        return tuple(env.render(mode) for env in self.envs)

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def __enter__(self) -> SyncVectorEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __copy__(self) -> SyncVectorEnv:
        raise TypeError("SyncVectorEnv owns mutable simulator state and cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> SyncVectorEnv:
        del memo
        raise TypeError("SyncVectorEnv owns mutable simulator state and cannot be copied")


IrisuSyncVectorEnv = SyncVectorEnv


class ThreadVectorEnv(SyncVectorEnv):
    """Step independent native worlds concurrently with a fixed thread pool."""

    def __init__(
        self,
        num_envs: int,
        *,
        library_path: str | PathLike[str] | None = None,
        render_mode: str | None = None,
        config: Mapping[str, Any] | None = None,
        workers: int | None = None,
        physics_backend: str = "portable",
        worker_path: str | PathLike[str] | None = None,
    ) -> None:
        super().__init__(
            num_envs,
            library_path=library_path,
            render_mode=render_mode,
            config=config,
            physics_backend=physics_backend,
            worker_path=worker_path,
        )
        if workers is not None and (
            not isinstance(workers, Integral) or isinstance(workers, bool) or workers <= 0
        ):
            super().close()
            raise ValueError("workers must be a positive integer or None")
        self._worker_count = min(
            self.num_envs,
            int(workers) if workers is not None else self.num_envs,
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self._worker_count,
            thread_name_prefix="irisu-env",
        )

    @staticmethod
    def _drain(futures: Sequence[Future[Any]]) -> list[Any]:
        """Wait every lane, then deterministically raise the first lane error."""

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

    def reset(
        self,
        *,
        seed: int | Sequence[int | None] | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        seeds = self._seeds(seed)
        futures = [
            self._executor.submit(env.reset, seed=value, options=options)
            for env, value in zip(self.envs, seeds)
        ]
        results = self._drain(futures)
        return [value[0] for value in results], [value[1] for value in results]

    def step(
        self, actions: Sequence[Action | Mapping[str, Any]]
    ) -> tuple[
        list[dict[str, Any]],
        list[int],
        list[bool],
        list[bool],
        list[dict[str, Any]],
    ]:
        supplied = self._items(actions, "actions")
        actions = tuple(Action(*_action(action)) for action in supplied)
        if all(env.physics_backend == "exact" for env in self.envs):
            results: list[Any] = [None] * self.num_envs
            failures: dict[int, BaseException] = {}
            for start in range(0, self.num_envs, self._worker_count):
                stop = min(start + self._worker_count, self.num_envs)
                sent: list[tuple[int, IrisuEnv]] = []
                for index in range(start, stop):
                    try:
                        self.envs[index]._send_exact_step(actions[index])
                        sent.append((index, self.envs[index]))
                    except BaseException as exc:
                        failures.setdefault(index, exc)
                for index, env in sent:
                    try:
                        results[index] = env._receive_exact_step()
                    except BaseException as exc:
                        failures.setdefault(index, exc)
            if failures:
                raise failures[min(failures)]
            observations, rewards, terminated, truncated, infos = zip(*results)
            return (
                list(observations),
                list(rewards),
                list(terminated),
                list(truncated),
                list(infos),
            )
        futures = [
            self._executor.submit(env.step, action)
            for env, action in zip(self.envs, actions)
        ]
        results = self._drain(futures)
        observations, rewards, terminated, truncated, infos = zip(*results)
        return (
            list(observations), list(rewards), list(terminated), list(truncated), list(infos)
        )

    def restore_state(self, snapshots: Sequence[bytes]) -> list[dict[str, Any]]:
        snapshots = self._items(snapshots, "snapshots")
        backups = tuple(
            (env._native.clone_state(), env._has_reset) for env in self.envs
        )
        futures = [
            self._executor.submit(env.restore_state, snapshot)
            for env, snapshot in zip(self.envs, snapshots)
        ]
        try:
            return self._drain(futures)
        except BaseException:
            def rollback(env: IrisuEnv, backup: tuple[bytes, bool]) -> None:
                env._native.restore_state(backup[0])
                env._has_reset = backup[1]

            rollback_futures = [
                self._executor.submit(rollback, env, backup)
                for env, backup in zip(self.envs, backups)
            ]
            try:
                self._drain(rollback_futures)
            except BaseException as rollback_error:
                raise RuntimeError("vector snapshot rollback failed") from rollback_error
            raise

    def close(self) -> None:
        executor = getattr(self, "_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        super().close()


IrisuThreadVectorEnv = ThreadVectorEnv
ParallelVectorEnv = ThreadVectorEnv
