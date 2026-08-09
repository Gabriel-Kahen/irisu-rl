"""Identity-bound Phase-2 trajectories, delayed rewards, and sequence targets."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from irisu_env import ActionKind, EventKind

from .experts import PointerExpertDecision


_SERIALIZATION_FORMAT = "irisu-pointer-trajectories-v1"
_MAX_SERIALIZED_BYTES = 64 * 1024 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _identity(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or not value.isascii()
    ):
        raise ValueError(f"{name} must be nonempty NUL-free ASCII")
    return value


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True, slots=True)
class DelayedRewardSpec:
    """Auditable delayed-game reward weights over public transition data."""

    raw_score_delta: float = 1.0
    survival_tick: float = 0.02
    chain_continuation: float = 100.0
    chain_join: float = 25.0
    clear: float = 100.0
    gauge_delta: float = 1.0
    rot_penalty: float = -100.0
    terminal_penalty: float = -1_000.0
    invalid_penalty: float = -1_000.0
    version: str = "pointer-delayed-reward-v1"

    def __post_init__(self) -> None:
        _identity(self.version, "reward version")
        for name in (
            "raw_score_delta",
            "survival_tick",
            "chain_continuation",
            "chain_join",
            "clear",
            "gauge_delta",
            "rot_penalty",
            "terminal_penalty",
            "invalid_penalty",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"reward weight {name} must be finite")
            object.__setattr__(self, name, float(value))
        for name in (
            "raw_score_delta",
            "survival_tick",
            "chain_continuation",
            "chain_join",
            "clear",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"reward weight {name} must be nonnegative")
        for name in ("rot_penalty", "terminal_penalty", "invalid_penalty"):
            if getattr(self, name) > 0.0:
                raise ValueError(f"reward penalty {name} must be nonpositive")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())

    def decompose(self, step: TrajectoryStep) -> RewardDecomposition:
        counts: dict[int, int] = {}
        for event in step.events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        score_delta = step.score_after - step.score_before
        gauge_delta = step.gauge_after - step.gauge_before
        continuation = max(
            0, step.highest_chain_after - step.highest_chain_before
        )
        joins = counts.get(int(EventKind.CHAIN_JOINED), 0)
        clears = counts.get(int(EventKind.CLEARED), 0)
        rot = counts.get(int(EventKind.ROTTEN), 0)
        invalid = counts.get(int(EventKind.INVALID_ACTION), 0)
        total = (
            self.raw_score_delta * score_delta
            + self.survival_tick * step.elapsed_ticks
            + self.chain_continuation * continuation
            + self.chain_join * joins
            + self.clear * clears
            + self.gauge_delta * gauge_delta
            + self.rot_penalty * rot
            + self.terminal_penalty * int(step.terminated)
            + self.invalid_penalty * invalid
        )
        return RewardDecomposition(
            raw_score_delta=score_delta,
            survival_ticks=step.elapsed_ticks,
            chain_continuation=continuation,
            chain_joins=joins,
            clears=clears,
            gauge_delta=gauge_delta,
            rot_events=rot,
            terminal=step.terminated,
            invalid_events=invalid,
            total=float(total),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryProvenance:
    """Immutable source/runtime/model identities for a trajectory collection."""

    source_revision: str
    runtime_sha256: str
    config_hash: int
    observation_schema_sha256: str
    pointer_spec_sha256: str
    reward_spec_sha256: str
    collector_id: str
    version: str = "pointer-trajectory-provenance-v1"

    def __post_init__(self) -> None:
        _identity(self.source_revision, "source revision")
        _digest(self.runtime_sha256, "runtime identity")
        _digest(self.observation_schema_sha256, "observation schema identity")
        _digest(self.pointer_spec_sha256, "pointer spec identity")
        _digest(self.reward_spec_sha256, "reward spec identity")
        _identity(self.collector_id, "collector identity")
        _identity(self.version, "provenance version")
        if (
            isinstance(self.config_hash, bool)
            or not isinstance(self.config_hash, int)
            or not 0 <= self.config_hash < 2**64
        ):
            raise ValueError("config_hash must fit uint64")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class PublicEvent:
    tick: int
    kind: int
    a: int = 0
    b: int = 0
    value: int = 0

    def __post_init__(self) -> None:
        for name in ("tick", "kind", "a", "b", "value"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"event {name} must be an integer")
        if self.tick < 0 or self.a < 0 or self.b < 0:
            raise ValueError("event tick and body handles must be nonnegative")
        try:
            EventKind(self.kind)
        except ValueError as exc:
            raise ValueError("event kind is unknown") from exc

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PublicEvent:
        return cls(
            tick=int(value.get("tick", 0)),
            kind=int(value["kind"]),
            a=int(value.get("a", 0)),
            b=int(value.get("b", 0)),
            value=int(value.get("value", 0)),
        )


@dataclass(frozen=True, slots=True)
class TrajectoryStep:
    """One semantic decision and its complete public delayed outcome."""

    episode_identity: str
    provenance_sha256: str
    decision_index: int
    tick_start: int
    tick_end: int
    action_kind: int
    wait_ticks: int
    target_body_id: int | None
    template_index: int
    score_before: int
    score_after: int
    gauge_before: int
    gauge_after: int
    highest_chain_before: int
    highest_chain_after: int
    events: tuple[PublicEvent, ...] = ()
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        _identity(self.episode_identity, "episode identity")
        _digest(self.provenance_sha256, "step provenance identity")
        integer_fields = (
            "decision_index",
            "tick_start",
            "tick_end",
            "action_kind",
            "wait_ticks",
            "template_index",
            "score_before",
            "score_after",
            "gauge_before",
            "gauge_after",
            "highest_chain_before",
            "highest_chain_after",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"step {name} must be an integer")
        if self.decision_index < 0 or self.tick_start < 0:
            raise ValueError("step index and start tick must be nonnegative")
        if self.tick_end <= self.tick_start:
            raise ValueError("a trajectory step must advance at least one tick")
        if self.tick_end >= 2**64 or self.decision_index >= 2**64:
            raise ValueError("step counters must fit uint64")
        if (
            self.wait_ticks < 1
            or self.wait_ticks > 100_000
            or self.template_index < 0
        ):
            raise ValueError("wait ticks and template index must be valid")
        if any(
            not -(2**63) <= getattr(self, name) < 2**63
            for name in (
                "score_before",
                "score_after",
                "gauge_before",
                "gauge_after",
            )
        ):
            raise ValueError("score and gauge values must fit int64")
        if self.highest_chain_before < 0 or self.highest_chain_after < 0:
            raise ValueError("highest-chain values must be nonnegative")
        if self.highest_chain_before >= 2**32 or self.highest_chain_after >= 2**32:
            raise ValueError("highest-chain values must fit uint32")
        kind = ActionKind.parse(self.action_kind)
        if kind not in (
            ActionKind.WAIT,
            ActionKind.WEAK_SHOT,
            ActionKind.STRONG_SHOT,
        ):
            raise ValueError("trajectory action kind is unsupported")
        if self.target_body_id is not None and (
            isinstance(self.target_body_id, bool)
            or not isinstance(self.target_body_id, int)
            or not 0 <= self.target_body_id < 2**32
        ):
            raise ValueError("trajectory body target must fit uint32")
        if kind is ActionKind.WAIT and self.target_body_id is not None:
            raise ValueError("wait trajectory steps cannot carry a body target")
        if kind is not ActionKind.WAIT and self.target_body_id is None:
            raise ValueError("shot trajectory steps require a body target")
        if kind is not ActionKind.WAIT and self.wait_ticks != 1:
            raise ValueError("shot trajectory decisions require wait_ticks=1")
        if not isinstance(self.terminated, bool) or not isinstance(
            self.truncated, bool
        ):
            raise TypeError("terminal flags must be booleans")
        if self.terminated and self.truncated:
            raise ValueError("a step cannot be both terminated and truncated")
        events = tuple(self.events)
        if any(not isinstance(event, PublicEvent) for event in events):
            raise TypeError("trajectory events must be PublicEvent values")
        if any(
            not self.tick_start < event.tick <= self.tick_end for event in events
        ):
            raise ValueError("event tick lies outside its trajectory step")
        object.__setattr__(self, "events", events)

    @property
    def elapsed_ticks(self) -> int:
        return self.tick_end - self.tick_start

    @classmethod
    def from_public_transition(
        cls,
        *,
        episode_identity: str,
        provenance_sha256: str,
        decision_index: int,
        decision: PointerExpertDecision,
        before: Mapping[str, Any],
        after: Mapping[str, Any],
        info: Mapping[str, Any],
        terminated: bool,
        truncated: bool,
    ) -> TrajectoryStep:
        return cls(
            episode_identity=episode_identity,
            provenance_sha256=provenance_sha256,
            decision_index=decision_index,
            tick_start=int(before["tick"]),
            tick_end=int(after["tick"]),
            action_kind=int(decision.kind),
            wait_ticks=int(decision.wait_ticks),
            target_body_id=decision.target_body_id,
            template_index=int(decision.template_index),
            score_before=int(before["score"]),
            score_after=int(after["score"]),
            gauge_before=int(before["gauge"]),
            gauge_after=int(after["gauge"]),
            highest_chain_before=int(before.get("highest_chain", 0)),
            highest_chain_after=int(after.get("highest_chain", 0)),
            events=tuple(
                PublicEvent.from_mapping(event)
                for event in info.get("events", ())
            ),
            terminated=terminated,
            truncated=truncated,
        )


@dataclass(frozen=True, slots=True)
class TrajectoryEpisode:
    episode_identity: str
    seed: int
    provenance: TrajectoryProvenance
    steps: tuple[TrajectoryStep, ...]

    def __post_init__(self) -> None:
        _identity(self.episode_identity, "episode identity")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or not 0 <= self.seed <= 0xFFFFFFFF
        ):
            raise ValueError("episode seed must fit uint32")
        if not isinstance(self.provenance, TrajectoryProvenance):
            raise TypeError("episode provenance has the wrong type")
        steps = tuple(self.steps)
        if not steps:
            raise ValueError("trajectory episodes must contain at least one step")
        for index, step in enumerate(steps):
            if not isinstance(step, TrajectoryStep):
                raise TypeError("episode contains a non-trajectory step")
            if (
                step.episode_identity != self.episode_identity
                or step.provenance_sha256 != self.provenance.sha256
                or step.decision_index != index
            ):
                raise ValueError("step identity, provenance, or index differs")
            if index and step.tick_start != steps[index - 1].tick_end:
                raise ValueError("episode step ticks are not contiguous")
            if index and (
                step.score_before != steps[index - 1].score_after
                or step.gauge_before != steps[index - 1].gauge_after
                or step.highest_chain_before
                != steps[index - 1].highest_chain_after
            ):
                raise ValueError("episode public state scalars are discontinuous")
            if index < len(steps) - 1 and (step.terminated or step.truncated):
                raise ValueError("terminal trajectory step must be last")
        object.__setattr__(self, "steps", steps)


@dataclass(frozen=True, slots=True)
class RewardDecomposition:
    raw_score_delta: int
    survival_ticks: int
    chain_continuation: int
    chain_joins: int
    clears: int
    gauge_delta: int
    rot_events: int
    terminal: bool
    invalid_events: int
    total: float


@dataclass(frozen=True, slots=True)
class ReturnTargets:
    scalar_returns: tuple[float, ...]
    quantile_targets: tuple[tuple[float, ...], ...]
    quantile_fractions: tuple[float, ...]


def discounted_return_targets(
    episode: TrajectoryEpisode,
    reward_spec: DelayedRewardSpec,
    *,
    gamma_tick: float = 0.999,
    bootstrap_value: float = 0.0,
    quantile_count: int = 51,
) -> ReturnTargets:
    """Build duration-aware scalar returns and critic-compatible quantile targets."""

    if (
        isinstance(gamma_tick, bool)
        or not isinstance(gamma_tick, (int, float))
        or not math.isfinite(float(gamma_tick))
        or not 0.0 < float(gamma_tick) <= 1.0
    ):
        raise ValueError("gamma_tick must be finite and in (0, 1]")
    if (
        isinstance(bootstrap_value, bool)
        or not isinstance(bootstrap_value, (int, float))
        or not math.isfinite(float(bootstrap_value))
    ):
        raise ValueError("bootstrap value must be finite")
    if (
        isinstance(quantile_count, bool)
        or not isinstance(quantile_count, int)
        or not 3 <= quantile_count <= 101
        or quantile_count % 2 != 1
    ):
        raise ValueError("quantile count must be odd and within [3, 101]")
    if episode.provenance.reward_spec_sha256 != reward_spec.sha256:
        raise ValueError("episode and delayed reward identities differ")
    last = episode.steps[-1]
    accumulator = 0.0 if last.terminated else float(bootstrap_value)
    values = [0.0] * len(episode.steps)
    for index in range(len(episode.steps) - 1, -1, -1):
        step = episode.steps[index]
        reward = reward_spec.decompose(step).total
        accumulator = reward + float(gamma_tick) ** step.elapsed_ticks * accumulator
        if not math.isfinite(accumulator):
            raise FloatingPointError("discounted trajectory return overflowed")
        values[index] = accumulator
    fractions = tuple((index + 0.5) / quantile_count for index in range(quantile_count))
    quantiles = tuple(tuple(value for _ in fractions) for value in values)
    return ReturnTargets(tuple(values), quantiles, fractions)


@dataclass(frozen=True, slots=True)
class SequenceWindow:
    episode_identity: str
    start_decision_index: int
    steps: tuple[TrajectoryStep | None, ...]
    burn_in_mask: tuple[bool, ...]
    unroll_mask: tuple[bool, ...]
    valid_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        _identity(self.episode_identity, "window episode identity")
        if self.start_decision_index < 0:
            raise ValueError("window start must be nonnegative")
        width = len(self.steps)
        if (
            width == 0
            or len(self.burn_in_mask) != width
            or len(self.unroll_mask) != width
            or len(self.valid_mask) != width
        ):
            raise ValueError("sequence-window fields must share a nonempty width")
        for index, step in enumerate(self.steps):
            valid = self.valid_mask[index]
            if valid != (step is not None):
                raise ValueError("window valid mask differs from step padding")
            if self.burn_in_mask[index] and self.unroll_mask[index]:
                raise ValueError("burn-in and unroll masks must be disjoint")
            if (self.burn_in_mask[index] or self.unroll_mask[index]) and not valid:
                raise ValueError("padded window positions cannot train or burn in")
            if step is not None and step.episode_identity != self.episode_identity:
                raise ValueError("a sequence window crosses episode identity")


def episode_sequence_windows(
    episodes: Sequence[TrajectoryEpisode],
    *,
    burn_in: int,
    unroll: int,
    stride: int | None = None,
    drop_incomplete: bool = False,
) -> tuple[SequenceWindow, ...]:
    """Create fixed-width, episode-disjoint recurrent windows."""

    for name, value in (("burn_in", burn_in), ("unroll", unroll)):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or (value < 0 if name == "burn_in" else value < 1)
        ):
            raise ValueError(f"{name} has an invalid width")
    resolved_stride = unroll if stride is None else stride
    if (
        isinstance(resolved_stride, bool)
        or not isinstance(resolved_stride, int)
        or resolved_stride < 1
    ):
        raise ValueError("window stride must be positive")
    if not isinstance(drop_incomplete, bool):
        raise TypeError("drop_incomplete must be boolean")
    supplied = tuple(episodes)
    if not supplied or any(
        not isinstance(episode, TrajectoryEpisode) for episode in supplied
    ):
        raise ValueError("sequence windows require trajectory episodes")
    identities = [episode.episode_identity for episode in supplied]
    if len(set(identities)) != len(identities):
        raise ValueError("trajectory collection repeats an episode identity")

    width = burn_in + unroll
    windows: list[SequenceWindow] = []
    for episode in supplied:
        for start in range(0, len(episode.steps), resolved_stride):
            unroll_steps = episode.steps[start : start + unroll]
            if drop_incomplete and len(unroll_steps) < unroll:
                continue
            context_start = max(0, start - burn_in)
            context = episode.steps[context_start:start]
            left_padding = burn_in - len(context)
            right_padding = unroll - len(unroll_steps)
            values: tuple[TrajectoryStep | None, ...] = (
                (None,) * left_padding
                + context
                + unroll_steps
                + (None,) * right_padding
            )
            burn_mask = (
                (False,) * left_padding
                + (True,) * len(context)
                + (False,) * (unroll + right_padding)
            )[:width]
            unroll_mask = (
                (False,) * burn_in
                + (True,) * len(unroll_steps)
                + (False,) * right_padding
            )
            valid_mask = tuple(value is not None for value in values)
            windows.append(
                SequenceWindow(
                    episode_identity=episode.episode_identity,
                    start_decision_index=start,
                    steps=values,
                    burn_in_mask=burn_mask,
                    unroll_mask=unroll_mask,
                    valid_mask=valid_mask,
                )
            )
    return tuple(windows)


def split_episodes(
    episodes: Sequence[TrajectoryEpisode],
    *,
    validation_fraction: float = 0.2,
    salt: str = "irisu-pointer-trajectory-split-v1",
) -> tuple[tuple[TrajectoryEpisode, ...], tuple[TrajectoryEpisode, ...]]:
    """Deterministically split whole episodes without trajectory leakage."""

    supplied = tuple(episodes)
    if (
        len(supplied) < 2
        or any(not isinstance(value, TrajectoryEpisode) for value in supplied)
        or len({value.episode_identity for value in supplied}) != len(supplied)
    ):
        raise ValueError("episode split requires distinct episode identities")
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not 0.0 < float(validation_fraction) < 1.0
    ):
        raise ValueError("validation fraction must lie in (0, 1)")
    _identity(salt, "split salt")
    count = min(
        len(supplied) - 1,
        max(1, round(len(supplied) * float(validation_fraction))),
    )
    ordered = sorted(
        supplied,
        key=lambda episode: (
            hashlib.sha256(
                salt.encode() + b"\0" + episode.episode_identity.encode()
            ).digest(),
            episode.episode_identity,
        ),
    )
    validation_ids = {value.episode_identity for value in ordered[:count]}
    training = tuple(
        value for value in supplied if value.episode_identity not in validation_ids
    )
    validation = tuple(
        value for value in supplied if value.episode_identity in validation_ids
    )
    return training, validation


def _event_payload(event: PublicEvent) -> dict[str, int]:
    return asdict(event)


def _step_payload(step: TrajectoryStep) -> dict[str, object]:
    value = asdict(step)
    value["events"] = [_event_payload(event) for event in step.events]
    return value


def serialize_episodes(episodes: Sequence[TrajectoryEpisode]) -> bytes:
    """Serialize homogeneous trajectories with a canonical payload checksum."""

    supplied = tuple(episodes)
    if not supplied:
        raise ValueError("cannot serialize an empty trajectory collection")
    if any(not isinstance(episode, TrajectoryEpisode) for episode in supplied):
        raise TypeError("serialized collection contains a non-trajectory episode")
    provenance = supplied[0].provenance
    if any(
        episode.provenance != provenance
        for episode in supplied
    ):
        raise ValueError("serialized trajectories must share exact provenance")
    if len({episode.episode_identity for episode in supplied}) != len(supplied):
        raise ValueError("serialized trajectories repeat an episode identity")
    payload = {
        "format": _SERIALIZATION_FORMAT,
        "provenance": provenance.manifest(),
        "episodes": [
            {
                "episode_identity": episode.episode_identity,
                "seed": episode.seed,
                "steps": [_step_payload(step) for step in episode.steps],
            }
            for episode in supplied
        ],
    }
    envelope = {
        "format": f"{_SERIALIZATION_FORMAT}-envelope",
        "payload_sha256": _sha256(payload),
        "payload": payload,
    }
    encoded = _canonical_json(envelope)
    if len(encoded) > _MAX_SERIALIZED_BYTES:
        raise ValueError("serialized trajectories exceed the size limit")
    return encoded


def _require_keys(
    value: object, expected: set[str], name: str
) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} has unknown or missing fields")
    return value


def deserialize_episodes(
    encoded: bytes | bytearray | memoryview,
) -> tuple[TrajectoryEpisode, ...]:
    """Verify and reconstruct a canonical trajectory envelope."""

    raw = bytes(encoded)
    if not raw or len(raw) > _MAX_SERIALIZED_BYTES:
        raise ValueError("serialized trajectory size is invalid")
    try:
        envelope = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("serialized trajectories are not canonical JSON") from exc
    envelope = _require_keys(
        envelope,
        {"format", "payload_sha256", "payload"},
        "trajectory envelope",
    )
    if envelope["format"] != f"{_SERIALIZATION_FORMAT}-envelope":
        raise ValueError("trajectory envelope format is unsupported")
    _digest(envelope["payload_sha256"], "payload checksum")
    if _sha256(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError("trajectory payload checksum differs")
    payload = _require_keys(
        envelope["payload"],
        {"format", "provenance", "episodes"},
        "trajectory payload",
    )
    if payload["format"] != _SERIALIZATION_FORMAT:
        raise ValueError("trajectory payload format is unsupported")
    provenance_fields = {
        "source_revision",
        "runtime_sha256",
        "config_hash",
        "observation_schema_sha256",
        "pointer_spec_sha256",
        "reward_spec_sha256",
        "collector_id",
        "version",
    }
    provenance_value = _require_keys(
        payload["provenance"], provenance_fields, "trajectory provenance"
    )
    provenance = TrajectoryProvenance(**provenance_value)
    if not isinstance(payload["episodes"], list) or not payload["episodes"]:
        raise ValueError("trajectory payload has no episodes")
    episodes: list[TrajectoryEpisode] = []
    step_fields = {
        "episode_identity",
        "provenance_sha256",
        "decision_index",
        "tick_start",
        "tick_end",
        "action_kind",
        "wait_ticks",
        "target_body_id",
        "template_index",
        "score_before",
        "score_after",
        "gauge_before",
        "gauge_after",
        "highest_chain_before",
        "highest_chain_after",
        "events",
        "terminated",
        "truncated",
    }
    event_fields = {"tick", "kind", "a", "b", "value"}
    for raw_episode in payload["episodes"]:
        episode_value = _require_keys(
            raw_episode,
            {"episode_identity", "seed", "steps"},
            "trajectory episode",
        )
        if not isinstance(episode_value["steps"], list):
            raise ValueError("trajectory episode steps must be a list")
        steps: list[TrajectoryStep] = []
        for raw_step in episode_value["steps"]:
            step_value = dict(
                _require_keys(raw_step, step_fields, "trajectory step")
            )
            if not isinstance(step_value["events"], list):
                raise ValueError("trajectory step events must be a list")
            step_value["events"] = tuple(
                PublicEvent(
                    **_require_keys(event, event_fields, "trajectory event")
                )
                for event in step_value["events"]
            )
            steps.append(TrajectoryStep(**step_value))
        episodes.append(
            TrajectoryEpisode(
                episode_identity=episode_value["episode_identity"],
                seed=episode_value["seed"],
                provenance=provenance,
                steps=tuple(steps),
            )
        )
    result = tuple(episodes)
    if serialize_episodes(result) != raw:
        raise ValueError("trajectory encoding is not canonical")
    return result


__all__ = [
    "DelayedRewardSpec",
    "PublicEvent",
    "ReturnTargets",
    "RewardDecomposition",
    "SequenceWindow",
    "TrajectoryEpisode",
    "TrajectoryProvenance",
    "TrajectoryStep",
    "deserialize_episodes",
    "discounted_return_targets",
    "episode_sequence_windows",
    "serialize_episodes",
    "split_episodes",
]
