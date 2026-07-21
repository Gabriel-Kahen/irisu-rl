"""Seeded action and perception perturbations for transfer-robust training.

The wrapper only consumes the public action/observation contract.  It never
reads snapshots, simulator hashes, RNG state, contacts, or other diagnostics.
All default ranges are zero-width, so wrapping an environment is nominal until
the caller explicitly supplies uncertainty bounds.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import math
from numbers import Real
from typing import Any

from .env import Action, ActionKind, _action, _gym
from .policies import SplitMix64
from .randomization import ParameterRange


_ZERO_FLOAT = ParameterRange(0.0, 0.0)
_ZERO_TICKS = ParameterRange(0.0, 0.0, integer=True)
_MAX_WAIT_TICKS = 100_000


def _is_zero(parameter: ParameterRange) -> bool:
    return parameter.low == 0.0 and parameter.high == 0.0


def _validate_range(name: str, value: object, *, integer: bool = False) -> None:
    if not isinstance(value, ParameterRange):
        raise TypeError(f"{name} must be a ParameterRange")
    if integer and not value.integer:
        raise ValueError(f"{name} must be an integer ParameterRange")
    if not value.integer and not math.isfinite(value.high - value.low):
        raise ValueError(f"{name} span must be finite")


@dataclass(frozen=True, slots=True)
class TransferRanges:
    """Explicit bounds for action and public-observation perturbations.

    Delay bounds are gameplay ticks.  A non-singleton ``action_delay_ticks``
    models timing jitter around its lower-bound latency.  Noise bounds are
    additive and sampled independently for every coordinate and detection.
    ``merge_distance`` is an Euclidean distance in display units.
    """

    action_delay_ticks: ParameterRange = _ZERO_TICKS
    observation_delay_ticks: ParameterRange = _ZERO_TICKS
    cursor_error_x: ParameterRange = _ZERO_FLOAT
    cursor_error_y: ParameterRange = _ZERO_FLOAT
    position_noise: ParameterRange = _ZERO_FLOAT
    velocity_noise: ParameterRange = _ZERO_FLOAT
    detection_drop_probability: float = 0.0
    merge_distance: float = 0.0

    def __post_init__(self) -> None:
        _validate_range("action_delay_ticks", self.action_delay_ticks, integer=True)
        _validate_range(
            "observation_delay_ticks", self.observation_delay_ticks, integer=True
        )
        for name in (
            "cursor_error_x",
            "cursor_error_y",
            "position_noise",
            "velocity_noise",
        ):
            _validate_range(name, getattr(self, name))
        for name in ("action_delay_ticks", "observation_delay_ticks"):
            value = getattr(self, name)
            if value.low < 0.0 or value.high > _MAX_WAIT_TICKS:
                raise ValueError(f"{name} must stay in [0, {_MAX_WAIT_TICKS}]")
        probability = self.detection_drop_probability
        if (
            isinstance(probability, bool)
            or not isinstance(probability, Real)
            or not math.isfinite(float(probability))
            or not 0.0 <= float(probability) <= 1.0
        ):
            raise ValueError("detection_drop_probability must be in [0, 1]")
        distance = self.merge_distance
        if (
            isinstance(distance, bool)
            or not isinstance(distance, Real)
            or not math.isfinite(float(distance))
            or float(distance) < 0.0
        ):
            raise ValueError("merge_distance must be a finite nonnegative number")

    @property
    def perception_is_nominal(self) -> bool:
        return (
            _is_zero(self.position_noise)
            and _is_zero(self.velocity_noise)
            and self.detection_drop_probability == 0.0
            and self.merge_distance == 0.0
        )


NOMINAL_TRANSFER_RANGES = TransferRanges()
_BaseWrapper = _gym.Wrapper if _gym is not None else object


def _sample(rng: SplitMix64, parameter: ParameterRange) -> float | int:
    if parameter.integer:
        low = int(parameter.low)
        high = int(parameter.high)
        if low == high:
            return low
        return low + rng.bounded(high - low + 1)
    if parameter.low == parameter.high:
        return float(parameter.low)
    return parameter.low + (parameter.high - parameter.low) * rng.unit()


def _scalar(value: object) -> object:
    if getattr(value, "shape", None) == ():
        item = getattr(value, "item", None)
        if callable(item):
            return item()
    return value


def _number(value: object, name: str) -> float:
    value = _scalar(value)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"public body {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"public body {name} must be finite")
    return result


def _same(values: Sequence[object]) -> bool:
    first = _scalar(values[0])
    return all(_scalar(value) == first for value in values[1:])


def _merge_component(
    bodies: list[dict[str, Any]], members: list[int]
) -> dict[str, Any]:
    ranked = sorted(
        members,
        key=lambda index: (int(_scalar(bodies[index].get("id", 0))), index),
    )
    merged = dict(bodies[ranked[0]])
    selected = [bodies[index] for index in members]
    merged["id"] = min(int(_scalar(body.get("id", 0))) for body in selected)
    for key in ("x", "y", "vx", "vy", "angle", "angular_velocity"):
        merged[key] = sum(_number(body[key], key) for body in selected) / len(selected)
    merged["size"] = max(_number(body["size"], "size") for body in selected)

    for key, ambiguous in (
        ("kind", "ambiguous"),
        ("shape", "unknown"),
        ("color", -1),
        ("chain_id", 0),
    ):
        values = [body.get(key) for body in selected]
        merged[key] = _scalar(values[0]) if _same(values) else ambiguous
    merged["lifecycle"] = "ambiguous"
    merged["projectile_hits"] = 0
    for key in ("age_ticks", "remaining_lifetime", "rot_timer"):
        merged[key] = min(int(_scalar(body.get(key, 0))) for body in selected)
    return merged


def _merge_nearby(
    bodies: list[dict[str, Any]], distance: float
) -> list[dict[str, Any]]:
    if distance <= 0.0 or len(bodies) < 2:
        return bodies
    parents = list(range(len(bodies)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parents[max(first_root, second_root)] = min(first_root, second_root)

    squared_limit = distance * distance
    positions = [(_number(body["x"], "x"), _number(body["y"], "y")) for body in bodies]
    for first in range(len(bodies)):
        for second in range(first + 1, len(bodies)):
            dx = positions[first][0] - positions[second][0]
            dy = positions[first][1] - positions[second][1]
            if dx * dx + dy * dy <= squared_limit:
                union(first, second)

    components: dict[int, list[int]] = {}
    for index in range(len(bodies)):
        components.setdefault(find(index), []).append(index)
    result: list[dict[str, Any]] = []
    for members in sorted(components.values(), key=min):
        if len(members) == 1:
            result.append(bodies[members[0]])
        else:
            result.append(_merge_component(bodies, members))
    return result


class TransferRobustnessEnv(_BaseWrapper):
    """Partially observed training wrapper with deterministic bounded noise.

    Rewards and termination flags always come from the current native state.
    Observation delay selects a prior *public, already perturbed* observation;
    the undisclosed current observation is never attached to ``info``.
    Terminal transitions flush observation delay so the final observation and
    returned termination flags describe the same state.
    """

    def __init__(
        self,
        env: Any,
        ranges: TransferRanges | None = None,
        *,
        transfer_seed: int = 0,
    ) -> None:
        if ranges is not None and not isinstance(ranges, TransferRanges):
            raise TypeError("ranges must be TransferRanges or None")
        if _gym is not None:
            super().__init__(env)
        else:
            self.env = env
        self.ranges = NOMINAL_TRANSFER_RANGES if ranges is None else ranges
        self._default_seed = self._validate_seed(transfer_seed)
        self._action_rng, self._delay_rng, self._perception_rng = self._streams(
            self._default_seed
        )
        self._history: deque[tuple[int, dict[str, Any]]] = deque()
        self._has_reset = False

        if _gym is None:
            self.action_space = getattr(env, "action_space", None)
            self.observation_space = getattr(env, "observation_space", None)
            self.metadata = getattr(env, "metadata", {})
            self.render_mode = getattr(env, "render_mode", None)

    @staticmethod
    def _validate_seed(seed: object) -> int:
        if type(seed) is not int or not 0 <= seed <= (1 << 64) - 1:
            raise ValueError("transfer seed must fit in uint64")
        return seed

    @staticmethod
    def _streams(seed: int) -> tuple[SplitMix64, SplitMix64, SplitMix64]:
        master = SplitMix64(seed)
        return (
            SplitMix64(master.next_u64()),
            SplitMix64(master.next_u64()),
            SplitMix64(master.next_u64()),
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        transfer_seed = self._default_seed
        forwarded_options = options
        if options is not None:
            if not isinstance(options, Mapping):
                raise TypeError("options must be a mapping or None")
            if "transfer_seed" in options:
                transfer_seed = self._validate_seed(options["transfer_seed"])
                forwarded = dict(options)
                del forwarded["transfer_seed"]
                forwarded_options = forwarded
        streams = self._streams(transfer_seed)
        observation, info = self.env.reset(seed=seed, options=forwarded_options)
        self._action_rng, self._delay_rng, self._perception_rng = streams
        self._history.clear()
        self._record(observation)
        self._has_reset = True
        public_observation = self._history[-1][1]
        if self.ranges.observation_delay_ticks.high > 0.0:
            public_observation = deepcopy(public_observation)
        return public_observation, info

    def _perceive(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise TypeError("wrapped environment observation must be a mapping")
        source = (
            deepcopy(observation)
            if self.ranges.observation_delay_ticks.high > 0.0
            else observation
        )
        if self.ranges.perception_is_nominal:
            return source if isinstance(source, dict) else dict(source)
        raw_bodies = source.get("bodies", ())
        if not isinstance(raw_bodies, Sequence) or isinstance(raw_bodies, (str, bytes)):
            raise TypeError("public observation bodies must be a sequence")

        detected: list[dict[str, Any]] = []
        probability = float(self.ranges.detection_drop_probability)
        for raw_body in raw_bodies:
            if not isinstance(raw_body, Mapping):
                raise TypeError("public observation body must be a mapping")
            drop = probability > 0.0 and self._perception_rng.unit() < probability
            body = dict(raw_body)
            for key, parameter in (
                ("x", self.ranges.position_noise),
                ("y", self.ranges.position_noise),
                ("vx", self.ranges.velocity_noise),
                ("vy", self.ranges.velocity_noise),
            ):
                body[key] = _number(body[key], key) + float(
                    _sample(self._perception_rng, parameter)
                )
            if not drop:
                detected.append(body)

        detected = _merge_nearby(detected, float(self.ranges.merge_distance))
        result = dict(source)
        result["bodies"] = (
            tuple(detected) if isinstance(raw_bodies, tuple) else detected
        )
        return result

    def _record(self, observation: Mapping[str, Any]) -> None:
        tick = int(_scalar(observation.get("tick", 0)))
        if self._history and tick < self._history[-1][0]:
            raise RuntimeError("wrapped observation tick moved backwards")
        perceived = self._perceive(observation)
        self._history.append((tick, perceived))
        oldest_needed = tick - int(self.ranges.observation_delay_ticks.high)
        while len(self._history) > 1 and self._history[1][0] <= oldest_needed:
            self._history.popleft()

    def _delayed_observation(self, *, flush: bool = False) -> dict[str, Any]:
        delay = (
            0
            if flush
            else int(_sample(self._delay_rng, self.ranges.observation_delay_ticks))
        )
        target = self._history[-1][0] - delay
        for tick, observation in reversed(self._history):
            if tick <= target:
                selected = observation
                break
        else:
            selected = self._history[0][1]
        if self.ranges.observation_delay_ticks.high > 0.0:
            return deepcopy(selected)
        return selected

    def _run(
        self, actions: Iterable[Action]
    ) -> tuple[dict[str, Any], Any, bool, bool, dict[str, Any]]:
        reward: Any = 0
        terminated = False
        truncated = False
        last_info: dict[str, Any] = {}
        all_events: list[Any] = []
        saw_events = False
        invalid_action = False
        for action in actions:
            observation, delta, terminated, truncated, info = self.env.step(action)
            reward += delta
            self._record(observation)
            last_info = dict(info)
            if "events" in info:
                saw_events = True
                all_events.extend(info["events"])
            invalid_action = invalid_action or bool(info.get("invalid_action", False))
            if terminated or truncated:
                break
        if saw_events:
            last_info["events"] = all_events
        if "invalid_action" in last_info or invalid_action:
            last_info["invalid_action"] = invalid_action
        return (
            self._delayed_observation(flush=terminated or truncated),
            reward,
            terminated,
            truncated,
            last_info,
        )

    def step(
        self, action: Action | Mapping[str, Any]
    ) -> tuple[dict[str, Any], Any, bool, bool, dict[str, Any]]:
        if not self._has_reset:
            raise RuntimeError("reset must be called before step")
        kind, x, y, wait_ticks = _action(action)
        if kind is ActionKind.WAIT:
            if not 1 <= wait_ticks <= _MAX_WAIT_TICKS:
                return self._run((Action(kind, x, y, wait_ticks),))
            history_ticks = min(
                wait_ticks, int(self.ranges.observation_delay_ticks.high)
            )

            def wait_actions() -> Iterable[Action]:
                if wait_ticks > history_ticks:
                    yield Action.wait(wait_ticks - history_ticks)
                for _ in range(history_ticks):
                    yield Action.wait(1)

            return self._run(wait_actions())

        delay = int(_sample(self._action_rng, self.ranges.action_delay_ticks))
        perturbed_x = min(
            640.0,
            max(0.0, x + float(_sample(self._action_rng, self.ranges.cursor_error_x))),
        )
        perturbed_y = min(
            480.0,
            max(0.0, y + float(_sample(self._action_rng, self.ranges.cursor_error_y))),
        )
        history_ticks = min(
            delay,
            max(0, int(self.ranges.observation_delay_ticks.high) - 1),
        )

        def delayed_shot() -> Iterable[Action]:
            if delay > history_ticks:
                yield Action.wait(delay - history_ticks)
            for _ in range(history_ticks):
                yield Action.wait(1)
            yield Action(kind, perturbed_x, perturbed_y, 1)

        return self._run(delayed_shot())

    def render(self, mode: str | None = None) -> Any:
        return self.env.render(mode=mode)

    def close(self) -> None:
        self.env.close()
        self._history.clear()
        self._has_reset = False

    def __enter__(self) -> TransferRobustnessEnv:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __copy__(self) -> TransferRobustnessEnv:
        raise TypeError(
            "TransferRobustnessEnv owns mutable history and cannot be copied"
        )

    def __deepcopy__(self, memo: dict[int, object]) -> TransferRobustnessEnv:
        del memo
        raise TypeError(
            "TransferRobustnessEnv owns mutable history and cannot be copied"
        )
