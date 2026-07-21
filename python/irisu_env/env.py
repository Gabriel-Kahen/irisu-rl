"""Gymnasium-shaped Python environment over the deterministic native clone."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum
from numbers import Integral, Real
from os import PathLike
from string import ascii_lowercase
from typing import Any

from .exact_ipc import ExactFastCheckpoint, ExactSimulator
from .native import NativeError, NativeSimulator
from .render import render_svg

try:
    import gymnasium as _gym
except ImportError:
    _gym = None


_BaseEnv = _gym.Env if _gym is not None else object


class ActionKind(IntEnum):
    WAIT = 0
    WEAK_SHOT = 1
    STRONG_SHOT = 2
    BOTH_SHOTS = 3

    @classmethod
    def parse(cls, value: ActionKind | int | str) -> ActionKind:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_")
            aliases = {
                "wait": cls.WAIT,
                "weak": cls.WEAK_SHOT,
                "weak_shot": cls.WEAK_SHOT,
                "strong": cls.STRONG_SHOT,
                "strong_shot": cls.STRONG_SHOT,
                "both": cls.BOTH_SHOTS,
                "both_shots": cls.BOTH_SHOTS,
            }
            if normalized in aliases:
                return aliases[normalized]
            raise ValueError(f"unknown action kind: {value!r}")
        if isinstance(value, Integral) and not isinstance(value, bool):
            try:
                return cls(int(value))
            except ValueError as exc:
                raise ValueError(f"unknown action kind: {value!r}") from exc
        raise TypeError("action kind must be an ActionKind, integer, or string")


@dataclass(frozen=True, slots=True)
class Action:
    kind: ActionKind | int | str = ActionKind.WAIT
    cursor_x: float = 0.0
    cursor_y: float = 0.0
    wait_ticks: int = 1

    @classmethod
    def wait(cls, ticks: int = 1) -> Action:
        return cls(ActionKind.WAIT, wait_ticks=ticks)

    @classmethod
    def weak(cls, x: float, y: float) -> Action:
        return cls(ActionKind.WEAK_SHOT, x, y)

    @classmethod
    def strong(cls, x: float, y: float) -> Action:
        return cls(ActionKind.STRONG_SHOT, x, y)

    @classmethod
    def both(cls, x: float, y: float) -> Action:
        """Hold both button levels for this update; fresh edges fire left first."""

        return cls(ActionKind.BOTH_SHOTS, x, y)


class EventKind(IntEnum):
    INVALID_ACTION = 0
    SPAWNED = 1
    SHOT_FIRED = 2
    ACTIVATED = 3
    CONTACT = 4
    CONFIRMED = 5
    CHAIN_JOINED = 6
    CLEARED = 7
    ROTTEN = 8
    EJECTED = 9
    DESTROYED = 10
    GAUGE_CHANGED = 11
    SCORE_CHANGED = 12
    LEVEL_CHANGED = 13
    GAME_OVER = 14
    PROJECTILE_HIT = 15
    PROJECTILE_CONTACT = 16
    HELD_INPUT_IGNORED = 17
    LEVEL_COMPLETED = 18


def _plain_scalar(value: Any) -> Any:
    """Normalize NumPy-style zero-dimensional samples without importing NumPy."""

    if getattr(value, "shape", None) == ():
        item = getattr(value, "item", None)
        if callable(item):
            return item()
    return value


def _action(value: Action | Mapping[str, Any]) -> tuple[ActionKind, float, float, int]:
    if isinstance(value, Action):
        kind = ActionKind.parse(_plain_scalar(value.kind))
        x, y, wait_ticks = value.cursor_x, value.cursor_y, value.wait_ticks
    elif isinstance(value, Mapping):
        if "kind" not in value:
            raise ValueError("action mapping requires kind")
        kind = ActionKind.parse(_plain_scalar(value["kind"]))
        x = value.get("cursor_x", 0.0)
        y = value.get("cursor_y", 0.0)
        wait_ticks = value.get("wait_ticks", 1)
    else:
        raise TypeError("action must be an Action or mapping")

    x = _plain_scalar(x)
    y = _plain_scalar(y)
    wait_ticks = _plain_scalar(wait_ticks)
    if not isinstance(x, Real) or isinstance(x, bool):
        raise TypeError("cursor_x must be a real number")
    if not isinstance(y, Real) or isinstance(y, bool):
        raise TypeError("cursor_y must be a real number")
    if not isinstance(wait_ticks, Integral) or isinstance(wait_ticks, bool):
        raise TypeError("wait_ticks must be an integer")
    wait_ticks = int(wait_ticks)
    if not 0 <= wait_ticks <= 0xFFFFFFFF:
        raise ValueError("wait_ticks must fit in uint32")
    return kind, float(x), float(y), wait_ticks


def _events(raw: object) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise NativeError("native step result has invalid events")
    result: list[dict[str, Any]] = []
    for value in raw:
        if not isinstance(value, dict) or not isinstance(value.get("kind"), int):
            raise NativeError("native step result contains an invalid event")
        event = dict(value)
        try:
            event["kind_name"] = EventKind(event["kind"]).name.lower()
        except ValueError:
            event["kind_name"] = f"unknown_{event['kind']}"
        result.append(event)
    return result


def _gym_observation(value: dict[str, Any]) -> dict[str, Any]:
    if _gym is None:
        return value
    import numpy as np

    result = dict(value)
    for key in (
        "tick",
        "qualifying_clear_count",
    ):
        result[key] = np.asarray(result[key], dtype=np.uint64)
    for key in (
        "score",
        "gauge",
        "gauge_max",
    ):
        result[key] = np.asarray(result[key], dtype=np.int64)
    for key in (
        "level",
        "highest_chain",
    ):
        result[key] = np.asarray(result[key], dtype=np.uint32)
    for key in ("terminated", "truncated", "left_held", "right_held"):
        result[key] = int(bool(result[key]))
    result["bodies"] = tuple(
        {
            **body,
            "id": np.asarray(body["id"], dtype=np.uint32),
            "color": np.asarray(body["color"], dtype=np.int32),
            "chain_id": np.asarray(body["chain_id"], dtype=np.uint32),
            "projectile_hits": np.asarray(body["projectile_hits"], dtype=np.uint32),
            "age_ticks": np.asarray(body["age_ticks"], dtype=np.uint64),
            "remaining_lifetime": np.asarray(
                body["remaining_lifetime"], dtype=np.int64
            ),
            "rot_timer": np.asarray(body["rot_timer"], dtype=np.uint64),
            **{
                key: np.asarray(body[key], dtype=np.float64)
                for key in (
                    "x", "y", "vx", "vy", "angle", "angular_velocity", "size"
                )
            },
        }
        for body in result["bodies"]
    )
    result["field"] = {
        key: np.asarray(number, dtype=np.float64)
        for key, number in result["field"].items()
    }
    result["difficulty"] = {
        key: np.asarray(number, dtype=np.uint32)
        for key, number in result["difficulty"].items()
    }
    return result


class IrisuFastCheckpoint:
    """IrisuEnv-shaped owner for one exact fork/COW checkpoint."""

    def __init__(
        self, source: IrisuEnv, checkpoint: ExactFastCheckpoint
    ) -> None:
        self._checkpoint = checkpoint
        self._render_mode = source.render_mode
        self._diagnostic_hashes = source.diagnostic_hashes
        self._worker_path = source.worker_path
        self._last_seed = source._last_seed

    @property
    def closed(self) -> bool:
        return self._checkpoint.closed

    @property
    def keeper_pid(self) -> int:
        return self._checkpoint.keeper_pid

    def branch(self) -> IrisuEnv:
        """Return an independently owned environment at the checkpoint state."""

        native = self._checkpoint.branch()
        try:
            return IrisuEnv._from_exact_branch(
                native,
                render_mode=self._render_mode,
                diagnostic_hashes=self._diagnostic_hashes,
                worker_path=self._worker_path,
                last_seed=self._last_seed,
            )
        except BaseException:
            native.close()
            raise

    def close(self) -> None:
        self._checkpoint.close()

    release = close

    def __enter__(self) -> IrisuFastCheckpoint:
        if self.closed:
            raise NativeError("Irisu fast checkpoint is closed")
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __copy__(self) -> IrisuFastCheckpoint:
        raise TypeError("IrisuFastCheckpoint is a unique process capability")

    def __deepcopy__(self, memo: dict[int, object]) -> IrisuFastCheckpoint:
        del memo
        raise TypeError("IrisuFastCheckpoint is a unique process capability")

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class IrisuEnv(_BaseEnv):
    """One headless normal-mode clone with Gymnasium-compatible method shapes."""

    metadata = {"render_modes": ["svg"], "render_fps": 50}

    def __init__(
        self,
        *,
        library_path: str | PathLike[str] | None = None,
        render_mode: str | None = None,
        config: Mapping[str, Any] | None = None,
        diagnostic_hashes: bool = False,
        physics_backend: str = "portable",
        worker_path: str | PathLike[str] | None = None,
    ) -> None:
        if _gym is not None:
            super().__init__()
        if render_mode not in (None, "svg"):
            raise ValueError("render_mode must be None or 'svg'")
        if not isinstance(physics_backend, str):
            raise TypeError("physics_backend must be a string")
        normalized_backend = physics_backend.strip().lower().replace("_", "-")
        aliases = {
            "portable": "portable",
            "exact": "exact",
            "exact-msvc": "exact",
            "exact-msvc9": "exact",
        }
        if normalized_backend not in aliases:
            raise ValueError("physics_backend must be 'portable' or 'exact'")
        self._physics_backend = aliases[normalized_backend]
        if self._physics_backend == "portable" and worker_path is not None:
            raise ValueError("worker_path is only valid for physics_backend='exact'")
        if self._physics_backend == "exact" and library_path is not None:
            raise ValueError(
                "library_path is only valid for the portable backend; use worker_path"
            )
        self.render_mode = render_mode
        self.diagnostic_hashes = bool(diagnostic_hashes)
        self._library_path = library_path
        self._worker_path = worker_path
        self._native = self._make_simulator(config)
        self._has_reset = False
        self._last_seed: int | None = None
        self.observation_space = self._make_observation_space()
        self.action_space = self._make_action_space()

    @classmethod
    def _from_exact_branch(
        cls,
        native: ExactSimulator,
        *,
        render_mode: str | None,
        diagnostic_hashes: bool,
        worker_path: str | PathLike[str] | None,
        last_seed: int | None,
    ) -> IrisuEnv:
        self = cls.__new__(cls)
        if _gym is not None:
            _gym.Env.__init__(self)
        self._physics_backend = "exact"
        self.render_mode = render_mode
        self.diagnostic_hashes = diagnostic_hashes
        self._library_path = None
        self._worker_path = worker_path
        self._native = native
        self._has_reset = True
        self._last_seed = last_seed
        self.observation_space = self._make_observation_space()
        self.action_space = self._make_action_space()
        return self

    def _make_simulator(
        self, config: Mapping[str, Any] | None
    ) -> NativeSimulator | ExactSimulator:
        if self._physics_backend == "exact":
            return ExactSimulator(self._worker_path, config=config)
        return NativeSimulator(self._library_path, config=config)

    @staticmethod
    def _make_action_space() -> object | None:
        if _gym is None:
            return None
        import numpy as np

        return _gym.spaces.Dict(
            {
                "kind": _gym.spaces.Discrete(4),
                "cursor_x": _gym.spaces.Box(0.0, 640.0, shape=(), dtype=np.float64),
                "cursor_y": _gym.spaces.Box(0.0, 480.0, shape=(), dtype=np.float64),
                "wait_ticks": _gym.spaces.Box(1, 100_000, shape=(), dtype=np.uint32),
            }
        )

    @staticmethod
    def _make_observation_space() -> object | None:
        if _gym is None:
            return None
        import numpy as np

        scalar_f64 = _gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float64)
        scalar_u32 = _gym.spaces.Box(0, np.iinfo(np.uint32).max, shape=(), dtype=np.uint32)
        scalar_u64 = _gym.spaces.Box(0, np.iinfo(np.uint64).max, shape=(), dtype=np.uint64)
        scalar_i32 = _gym.spaces.Box(np.iinfo(np.int32).min, np.iinfo(np.int32).max,
                                     shape=(), dtype=np.int32)
        scalar_i64 = _gym.spaces.Box(np.iinfo(np.int64).min, np.iinfo(np.int64).max,
                                     shape=(), dtype=np.int64)
        body = _gym.spaces.Dict(
            {
                "id": scalar_u32,
                "kind": _gym.spaces.Text(max_length=10, charset=ascii_lowercase + "_"),
                "shape": _gym.spaces.Text(max_length=8, charset=ascii_lowercase + "_"),
                "lifecycle": _gym.spaces.Text(
                    max_length=24, charset=ascii_lowercase + "_"
                ),
                "color": scalar_i32,
                "x": scalar_f64,
                "y": scalar_f64,
                "vx": scalar_f64,
                "vy": scalar_f64,
                "angle": scalar_f64,
                "angular_velocity": scalar_f64,
                "size": scalar_f64,
                "chain_id": scalar_u32,
                "projectile_hits": scalar_u32,
                "age_ticks": scalar_u64,
                "remaining_lifetime": scalar_i64,
                "rot_timer": scalar_u64,
            }
        )
        field = _gym.spaces.Dict(
            {
                key: scalar_f64
                for key in (
                    "x", "y", "width", "height", "side_wall_top", "side_wall_bottom"
                )
            }
        )
        difficulty = _gym.spaces.Dict(
            {"active_colors": scalar_u32, "spawn_interval_ticks": scalar_u32}
        )
        return _gym.spaces.Dict(
            {
                "tick": scalar_u64,
                "score": scalar_i64,
                "gauge": scalar_i64,
                "gauge_max": scalar_i64,
                "level": scalar_u32,
                "terminated": _gym.spaces.Discrete(2),
                "truncated": _gym.spaces.Discrete(2),
                "left_held": _gym.spaces.Discrete(2),
                "right_held": _gym.spaces.Discrete(2),
                "highest_chain": scalar_u32,
                "qualifying_clear_count": scalar_u64,
                "field": field,
                "difficulty": difficulty,
                "bodies": _gym.spaces.Sequence(body),
            }
        )

    @property
    def library_path(self) -> str:
        return self._native.library_path

    @property
    def physics_backend(self) -> str:
        return self._physics_backend

    @property
    def worker_path(self) -> str | None:
        if self._physics_backend != "exact":
            return None
        return self._native.worker_path

    @property
    def config(self) -> dict[str, Any]:
        return self._native.config()

    @property
    def build_info(self) -> dict[str, Any]:
        return self._native.build_info()

    def exact_library_provenance(self) -> dict[str, Any]:
        """Capture the library mapped by this environment's exact worker."""

        if self._physics_backend != "exact":
            raise NativeError(
                "exact library provenance requires physics_backend='exact'"
            )
        assert isinstance(self._native, ExactSimulator)
        return self._native.exact_library_provenance()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        resolved_seed = 0 if seed is None else seed
        if not isinstance(resolved_seed, Integral) or isinstance(resolved_seed, bool):
            raise TypeError("seed must be an integer or None")
        resolved_seed = int(resolved_seed)
        if not 0 <= resolved_seed <= 0xFFFFFFFF:
            raise ValueError("normal-mode seed must fit in uint32")

        requested_config: Mapping[str, Any] | None = None
        if options is not None and "config" in options:
            raw_config = options["config"]
            if not isinstance(raw_config, Mapping):
                raise TypeError("options['config'] must be a mapping")
            requested_config = raw_config

        replacement = None
        if requested_config is not None:
            if isinstance(self._native, ExactSimulator):
                self._native._require_replacement_safe("reset")
            replacement = self._make_simulator(requested_config)
        try:
            if _gym is not None:
                super().reset(seed=resolved_seed)
            target = replacement if replacement is not None else self._native
            raw_observation = target.reset(resolved_seed)
        except Exception:
            if replacement is not None:
                replacement.close()
            raise
        if replacement is not None:
            previous = self._native
            self._native = replacement
            previous.close()
            self.observation_space = self._make_observation_space()
            self.action_space = self._make_action_space()

        observation = _gym_observation(raw_observation)
        self._has_reset = True
        self._last_seed = resolved_seed
        info = {
            "seed": resolved_seed,
            "config_hash": self._native.config_hash(),
        }
        if self.diagnostic_hashes:
            info["state_hash"] = self._native.state_hash()
        return observation, info

    def step(
        self, action: Action | Mapping[str, Any]
    ) -> tuple[dict[str, Any], int, bool, bool, dict[str, Any]]:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        kind, x, y, wait_ticks = _action(action)
        result = self._native.step(int(kind), x, y, wait_ticks)
        return self._step_result(result)

    def _send_exact_step(self, action: Action) -> None:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        if not isinstance(self._native, ExactSimulator):
            raise RuntimeError("split steps require the exact worker backend")
        kind, x, y, wait_ticks = _action(action)
        self._native.send_step(int(kind), x, y, wait_ticks)

    def _receive_exact_step(
        self,
    ) -> tuple[dict[str, Any], int, bool, bool, dict[str, Any]]:
        if not isinstance(self._native, ExactSimulator):
            raise RuntimeError("split steps require the exact worker backend")
        return self._step_result(self._native.receive_step())

    def _step_result(
        self, result: dict[str, Any]
    ) -> tuple[dict[str, Any], int, bool, bool, dict[str, Any]]:
        observation = _gym_observation(self._native.observation())
        events = _events(result.get("events"))
        raw_diagnostics = result.get("diagnostics")
        if not isinstance(raw_diagnostics, dict) or not isinstance(
            raw_diagnostics.get("config_hash"), int
        ):
            raise NativeError("native step result has invalid diagnostics")
        diagnostics = dict(raw_diagnostics)
        reward = int(result.get("reward", 0))
        terminated = bool(result.get("terminated", False))
        truncated = bool(result.get("truncated", False))
        info = {
            "events": events,
            "invalid_action": any(
                event["kind"] == int(EventKind.INVALID_ACTION) for event in events
            ),
            "config_hash": int(diagnostics["config_hash"]),
            "diagnostics": diagnostics,
        }
        if self.diagnostic_hashes:
            info["state_hash"] = self._native.state_hash()
        return observation, reward, terminated, truncated, info

    def clone_state(self) -> bytes:
        if not self._has_reset:
            raise RuntimeError("reset must be called before clone_state")
        return self._native.clone_state()

    def fast_checkpoint(self) -> IrisuFastCheckpoint:
        """Create a reusable Linux-local exact checkpoint for fast branches."""

        if self._physics_backend != "exact" or not isinstance(
            self._native, ExactSimulator
        ):
            raise NativeError(
                "fast checkpoints require physics_backend='exact'"
            )
        if not self._has_reset:
            raise RuntimeError("reset must be called before fast_checkpoint")
        return IrisuFastCheckpoint(self, self._native.fast_checkpoint())

    def restore_state(self, snapshot: bytes | bytearray | memoryview) -> dict[str, Any]:
        observation = _gym_observation(self._native.restore_state(snapshot))
        self._has_reset = True
        return observation

    def state_hash(self) -> int:
        if not self._has_reset:
            raise RuntimeError("reset must be called before state_hash")
        return self._native.state_hash()

    def config_hash(self) -> int:
        if not self._has_reset:
            raise RuntimeError("reset must be called before config_hash")
        return self._native.config_hash()

    def render(self, mode: str | None = None) -> str:
        if not self._has_reset:
            raise RuntimeError("reset must be called before render")
        resolved_mode = mode or self.render_mode or "svg"
        if resolved_mode != "svg":
            raise ValueError("only deterministic diagnostic mode 'svg' is supported")
        return render_svg(self._native.observation())

    def close(self) -> None:
        self._native.close()
        self._has_reset = False

    def __enter__(self) -> IrisuEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __copy__(self) -> IrisuEnv:
        raise TypeError("IrisuEnv owns mutable simulator state and cannot be copied")

    def __deepcopy__(self, memo: dict[int, object]) -> IrisuEnv:
        del memo
        raise TypeError("IrisuEnv owns mutable simulator state and cannot be copied")


IriSuEnv = IrisuEnv
