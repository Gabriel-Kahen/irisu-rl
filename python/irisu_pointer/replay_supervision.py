"""Identity-bound steering supervision recovered from public replay transitions."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from irisu_env import Action, ActionKind, EventKind

from .action import PointerActionSpec


_COLLECTOR_VERSION = "replay-steering-supervision-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _identity(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or not value.isascii()
    ):
        raise ValueError(f"{name} must be nonempty NUL-free ASCII")
    return value


def _event_kind(event: Mapping[str, Any]) -> int | None:
    raw = event.get("kind")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return int(raw)
    name = event.get("kind_name")
    if isinstance(name, str):
        try:
            return int(EventKind[name.upper()])
        except KeyError:
            return None
    return None


def _effect_geometry(body: Mapping[str, Any]) -> tuple[float, float, float, float]:
    x = float(body.get("effect_x", body.get("x", 0.0)))
    y = float(body.get("effect_y", body.get("y", 0.0)))
    size = float(body.get("size", 0.0))
    width = float(body.get("width", size))
    height = float(body.get("height", size))
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("body effect geometry must be finite")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("body effect extents must be positive")
    return x, y, width, height


def _copy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Own only public fields needed by the learner and identity audit."""

    output = {
        key: observation[key]
        for key in (
            "tick",
            "score",
            "gauge",
            "gauge_max",
            "level",
            "highest_chain",
            "qualifying_clear_count",
            "left_held",
            "right_held",
            "terminated",
            "truncated",
            "field",
            "difficulty",
        )
        if key in observation
    }
    output["bodies"] = tuple(
        dict(body)
        for body in observation.get("bodies", ())
        if isinstance(body, Mapping)
    )
    return output


def _json_value(value: Any) -> Any:
    item = getattr(value, "item", None)
    if callable(item):
        return _json_value(item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("observation identity contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item_value)
            for key, item_value in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_value(item_value) for item_value in value]
    raise TypeError("observation identity contains unsupported public data")


@dataclass(frozen=True, slots=True)
class ReplayInputFrame:
    """One replay input level; fresh edges are reconstructed by the collector."""

    index: int
    left: bool
    right: bool
    x: int
    y: int
    raw_word: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.index, bool)
            or not isinstance(self.index, int)
            or self.index < 0
        ):
            raise ValueError("replay frame index must be nonnegative")
        for name, maximum in (("x", 1023), ("y", 511), ("raw_word", 0xFFFFFFFF)):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= maximum
            ):
                raise ValueError(f"replay frame {name} is outside its packed range")
        object.__setattr__(self, "left", bool(self.left))
        object.__setattr__(self, "right", bool(self.right))

    @classmethod
    def from_object(cls, value: object, index: int | None = None) -> ReplayInputFrame:
        resolved_index = (
            int(getattr(value, "index"))
            if index is None and hasattr(value, "index")
            else int(index if index is not None else 0)
        )
        return cls(
            resolved_index,
            bool(getattr(value, "left")),
            bool(getattr(value, "right")),
            int(getattr(value, "x")),
            int(getattr(value, "y")),
            int(getattr(value, "raw_word", 0)),
        )


@dataclass(frozen=True, slots=True)
class ReplayEvidenceIdentity:
    source_revision: str
    replay_sha256: str
    runtime_sha256: str
    config_hash: int
    observation_schema_sha256: str
    pointer_spec_sha256: str
    mapped_runtime_sha256: str | None = None
    collector_version: str = _COLLECTOR_VERSION

    def __post_init__(self) -> None:
        _identity(self.source_revision, "source revision")
        _digest(self.replay_sha256, "replay identity")
        _digest(self.runtime_sha256, "runtime identity")
        _digest(self.observation_schema_sha256, "observation schema identity")
        _digest(self.pointer_spec_sha256, "pointer action identity")
        if self.mapped_runtime_sha256 is not None:
            _digest(self.mapped_runtime_sha256, "mapped runtime identity")
        _identity(self.collector_version, "collector version")
        if (
            isinstance(self.config_hash, bool)
            or not isinstance(self.config_hash, int)
            or not 0 <= self.config_hash < 2**64
        ):
            raise ValueError("config hash must fit uint64")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.manifest())).hexdigest()


@dataclass(frozen=True, slots=True)
class ReplayShotSupervision:
    """One fired projectile with its first observed public hit."""

    frame_index: int
    shot_tick: int
    hit_frame_index: int
    hit_tick: int
    projectile_id: int
    action_kind: int
    cursor_x: float
    cursor_y: float
    target_body_id: int
    destination_body_id: int | None
    target_color: int
    target_lifecycle: str
    target_grouped: bool
    cursor_x_radius_offset: float
    cursor_y_radius_offset: float
    template_index: int

    def __post_init__(self) -> None:
        integer_fields = (
            "frame_index",
            "shot_tick",
            "hit_frame_index",
            "hit_tick",
            "projectile_id",
            "action_kind",
            "target_body_id",
            "target_color",
            "template_index",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"shot {name} must be an integer")
        if (
            min(
                self.frame_index,
                self.shot_tick,
                self.hit_frame_index,
                self.hit_tick,
                self.projectile_id,
                self.target_body_id,
                self.template_index,
            )
            < 0
            or self.hit_frame_index < self.frame_index
            or self.hit_tick < self.shot_tick
        ):
            raise ValueError("shot timing, IDs, and template must be nonnegative")
        if ActionKind.parse(self.action_kind) not in (
            ActionKind.WEAK_SHOT,
            ActionKind.STRONG_SHOT,
        ):
            raise ValueError("replay supervision requires a weak or strong shot")
        if self.destination_body_id is not None and (
            isinstance(self.destination_body_id, bool)
            or not isinstance(self.destination_body_id, int)
            or self.destination_body_id < 0
            or self.destination_body_id == self.target_body_id
        ):
            raise ValueError("destination must be a distinct nonnegative body ID")
        for name in (
            "cursor_x",
            "cursor_y",
            "cursor_x_radius_offset",
            "cursor_y_radius_offset",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"shot {name} must be finite")
        _identity(self.target_lifecycle, "target lifecycle")

    @property
    def first_hit_delay_ticks(self) -> int:
        return self.hit_tick - self.shot_tick

    def manifest(self) -> dict[str, object]:
        return {
            **asdict(self),
            "first_hit_delay_ticks": self.first_hit_delay_ticks,
            "destination_semantics": "nearest-visible-same-color-peer-inference-v1",
        }


@dataclass(frozen=True, slots=True)
class SteeringConversionMetrics:
    frames: int
    survival_ticks: int
    requested_shots: int
    shots_fired: int
    shots_hit: int
    projectile_hit_events: int
    chain_joins: int
    clears: int
    rotten: int
    ejected: int
    invalid_actions: int
    final_score: int
    terminated: bool
    truncated: bool

    def __post_init__(self) -> None:
        for name in (
            "frames",
            "survival_ticks",
            "requested_shots",
            "shots_fired",
            "shots_hit",
            "projectile_hit_events",
            "chain_joins",
            "clears",
            "rotten",
            "ejected",
            "invalid_actions",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"metric {name} must be a nonnegative integer")
        if isinstance(self.final_score, bool) or not isinstance(self.final_score, int):
            raise TypeError("final score must be an integer")
        if self.shots_hit > self.shots_fired:
            raise ValueError("hit projectile count cannot exceed fired projectile count")

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return float(numerator / denominator) if denominator else 0.0

    @property
    def hit_rate(self) -> float:
        return self._rate(self.shots_hit, self.shots_fired)

    @property
    def joins_per_shot(self) -> float:
        return self._rate(self.chain_joins, self.shots_fired)

    @property
    def clears_per_shot(self) -> float:
        return self._rate(self.clears, self.shots_fired)

    def manifest(self) -> dict[str, object]:
        return {
            **asdict(self),
            "hit_rate": self.hit_rate,
            "joins_per_shot": self.joins_per_shot,
            "clears_per_shot": self.clears_per_shot,
        }


@dataclass(frozen=True, slots=True)
class ReplaySteeringCollection:
    identity: ReplayEvidenceIdentity
    shots: tuple[ReplayShotSupervision, ...]
    shot_observations: tuple[Mapping[str, Any], ...]
    metrics: SteeringConversionMetrics

    def __post_init__(self) -> None:
        if len(self.shots) != len(self.shot_observations):
            raise ValueError("every supervised shot requires its pre-shot observation")
        if len({shot.projectile_id for shot in self.shots}) != len(self.shots):
            raise ValueError("supervision contains duplicate projectile IDs")
        owned = tuple(
            _json_value(_copy_observation(observation))
            for observation in self.shot_observations
        )
        object.__setattr__(self, "shot_observations", owned)

    def manifest(self) -> dict[str, object]:
        return {
            "format": "irisu-replay-steering-collection-v1",
            "identity": {**self.identity.manifest(), "sha256": self.identity.sha256},
            "shots": [shot.manifest() for shot in self.shots],
            "shot_observation_sha256s": [
                hashlib.sha256(_canonical_json(observation)).hexdigest()
                for observation in self.shot_observations
            ],
            "metrics": self.metrics.manifest(),
        }

    @property
    def sha256(self) -> str:
        return hashlib.sha256(_canonical_json(self.manifest())).hexdigest()


class ReplayEnvironment(Protocol):
    def reset(
        self, *, seed: int
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...

    def step(
        self, action: Action
    ) -> tuple[Mapping[str, Any], int, bool, bool, Mapping[str, Any]]: ...


@dataclass(slots=True)
class _PendingShot:
    frame_index: int
    shot_tick: int
    projectile_id: int
    action_kind: int
    cursor_x: float
    cursor_y: float
    observation: dict[str, Any]


def _nearest_same_color_peer(
    bodies: Sequence[Mapping[str, Any]], target: Mapping[str, Any]
) -> int | None:
    target_id = int(target["id"])
    target_color = int(target.get("color", -1))
    target_x, target_y, _, _ = _effect_geometry(target)
    candidates: list[tuple[float, float, int]] = []
    for body in bodies:
        if (
            body.get("kind") != "piece"
            or int(body.get("id", -1)) == target_id
            or int(body.get("color", -2)) != target_color
            or body.get("lifecycle") == "deleted"
        ):
            continue
        x, y, _, _ = _effect_geometry(body)
        candidates.append(((x - target_x) ** 2 + (y - target_y) ** 2, x, int(body["id"])))
    return min(candidates)[2] if candidates else None


def _template_index(
    spec: PointerActionSpec, x_offset: float, y_offset: float
) -> int:
    return min(
        range(spec.template_count),
        key=lambda index: (
            (spec.templates[index][0] - x_offset) ** 2
            + (spec.templates[index][1] - y_offset) ** 2,
            index,
        ),
    )


def collect_replay_steering_supervision(
    env: ReplayEnvironment,
    frames: Iterable[ReplayInputFrame | object],
    *,
    seed: int,
    identity: ReplayEvidenceIdentity,
    pointer_spec: PointerActionSpec | None = None,
) -> ReplaySteeringCollection:
    """Replay input levels and bind each projectile to its first public hit.

    The first two replay records update held history but suppress fresh edges,
    matching the recovered v2.03 input loop. Destination labels are explicitly
    behavioral inferences; hit targets and projectile identities come directly
    from public events.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
        raise ValueError("normal replay seed must fit uint32")
    spec = pointer_spec or PointerActionSpec()
    if identity.pointer_spec_sha256 != spec.sha256:
        raise ValueError("replay identity and pointer action identity differ")
    normalized = tuple(
        value
        if isinstance(value, ReplayInputFrame)
        else ReplayInputFrame.from_object(value, index)
        for index, value in enumerate(frames)
    )
    if any(frame.index != index for index, frame in enumerate(normalized)):
        raise ValueError("replay frame indices must be contiguous from zero")

    observation, reset_info = env.reset(seed=seed)
    if int(reset_info.get("config_hash", -1)) != identity.config_hash:
        raise RuntimeError("replay reset config identity mismatch")
    initial_tick = int(observation.get("tick", 0))
    previous_left = False
    previous_right = False
    pending: dict[int, _PendingShot] = {}
    completed: list[tuple[ReplayShotSupervision, Mapping[str, Any]]] = []
    event_counts: Counter[int] = Counter()
    requested_shots = 0
    hit_projectiles: set[int] = set()
    terminated = bool(observation.get("terminated", False))
    truncated = bool(observation.get("truncated", False))

    for frame in normalized:
        if terminated or truncated:
            break
        pre_observation = _copy_observation(observation)
        raw_left = frame.left and not previous_left
        raw_right = frame.right and not previous_right
        suppress = frame.index < 2
        left_edge = raw_left and not suppress
        right_edge = raw_right and not suppress
        requested_shots += int(left_edge) + int(right_edge)
        if left_edge and right_edge:
            action = Action.both(frame.x, frame.y)
        elif left_edge:
            action = Action.weak(frame.x, frame.y)
        elif right_edge:
            action = Action.strong(frame.x, frame.y)
        else:
            action = Action(ActionKind.WAIT, frame.x, frame.y, 1)
        previous_left, previous_right = frame.left, frame.right
        observation, _, terminated, truncated, info = env.step(action)
        events = tuple(info.get("events", ()))
        for event in events:
            if not isinstance(event, Mapping):
                continue
            kind = _event_kind(event)
            if kind is None:
                continue
            event_counts[kind] += 1
            if kind == int(EventKind.SHOT_FIRED):
                projectile_id = int(event.get("a", -1))
                if projectile_id < 0 or projectile_id in pending:
                    raise RuntimeError("shot event has an invalid or duplicate projectile ID")
                action_kind = (
                    int(ActionKind.STRONG_SHOT)
                    if int(event.get("value", 0))
                    else int(ActionKind.WEAK_SHOT)
                )
                pending[projectile_id] = _PendingShot(
                    frame.index,
                    int(event.get("tick", pre_observation.get("tick", 0))),
                    projectile_id,
                    action_kind,
                    float(frame.x),
                    float(frame.y),
                    pre_observation,
                )
            elif kind == int(EventKind.PROJECTILE_HIT):
                projectile_id = int(event.get("a", -1))
                target_id = int(event.get("b", -1))
                if projectile_id in hit_projectiles:
                    continue
                hit_projectiles.add(projectile_id)
                shot = pending.get(projectile_id)
                if shot is None:
                    continue
                bodies = tuple(
                    body
                    for body in shot.observation.get("bodies", ())
                    if isinstance(body, Mapping)
                )
                targets = [body for body in bodies if int(body.get("id", -1)) == target_id]
                if len(targets) != 1:
                    continue
                target = targets[0]
                x, y, width, height = _effect_geometry(target)
                x_offset = (shot.cursor_x - x) / (width / 2.0)
                y_offset = (shot.cursor_y - y) / (height / 2.0)
                supervision = ReplayShotSupervision(
                    frame_index=shot.frame_index,
                    shot_tick=shot.shot_tick,
                    hit_frame_index=frame.index,
                    hit_tick=int(event.get("tick", observation.get("tick", 0))),
                    projectile_id=projectile_id,
                    action_kind=shot.action_kind,
                    cursor_x=shot.cursor_x,
                    cursor_y=shot.cursor_y,
                    target_body_id=target_id,
                    destination_body_id=_nearest_same_color_peer(bodies, target),
                    target_color=int(target.get("color", -1)),
                    target_lifecycle=str(target.get("lifecycle", "unknown")),
                    target_grouped=bool(int(target.get("chain_id", 0))),
                    cursor_x_radius_offset=x_offset,
                    cursor_y_radius_offset=y_offset,
                    template_index=_template_index(spec, x_offset, y_offset),
                )
                completed.append((supervision, shot.observation))

    final_tick = int(observation.get("tick", initial_tick))
    shots_fired = event_counts[int(EventKind.SHOT_FIRED)]
    metrics = SteeringConversionMetrics(
        frames=min(len(normalized), max(0, final_tick - initial_tick)),
        survival_ticks=max(0, final_tick - initial_tick),
        requested_shots=requested_shots,
        shots_fired=shots_fired,
        shots_hit=len(hit_projectiles & set(pending)),
        projectile_hit_events=event_counts[int(EventKind.PROJECTILE_HIT)],
        chain_joins=event_counts[int(EventKind.CHAIN_JOINED)],
        clears=event_counts[int(EventKind.CLEARED)],
        rotten=event_counts[int(EventKind.ROTTEN)],
        ejected=event_counts[int(EventKind.EJECTED)],
        invalid_actions=event_counts[int(EventKind.INVALID_ACTION)],
        final_score=int(observation.get("score", 0)),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )
    return ReplaySteeringCollection(
        identity,
        tuple(value[0] for value in completed),
        tuple(value[1] for value in completed),
        metrics,
    )


__all__ = [
    "ReplayEvidenceIdentity",
    "ReplayInputFrame",
    "ReplayShotSupervision",
    "ReplaySteeringCollection",
    "SteeringConversionMetrics",
    "collect_replay_steering_supervision",
]
