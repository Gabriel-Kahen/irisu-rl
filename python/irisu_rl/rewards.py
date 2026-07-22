"""Auditable score scaling and episode-stable curriculum shaping."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Callable, ClassVar, Protocol, Sequence, runtime_checkable

import torch
from torch import Tensor

from .vector_adapter import MacroTransition


@dataclass(frozen=True, slots=True)
class RewardKnot:
    """One integer-valued point in a shaping schedule."""

    completed_update: int
    shaping_weight_ppm: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.completed_update, bool)
            or not isinstance(self.completed_update, Integral)
            or self.completed_update < 0
        ):
            raise ValueError("reward-knot update must be a nonnegative integer")
        if (
            isinstance(self.shaping_weight_ppm, bool)
            or not isinstance(self.shaping_weight_ppm, Integral)
            or not 0 <= self.shaping_weight_ppm <= 1_000_000
        ):
            raise ValueError("shaping weight must be within [0, 1_000_000] ppm")


@dataclass(frozen=True, slots=True)
class RewardSchedule:
    """Versioned, monotone, piecewise-linear shaping schedule.

    Integer parts-per-million weights make the score-only endpoint exact and
    keep schedule state independent of platform floating-point formatting.
    """

    schedule_id: str
    knots: tuple[RewardKnot, ...]
    version: str = "reward-schedule-v1"

    def __post_init__(self) -> None:
        if not self.schedule_id or not self.schedule_id.isascii():
            raise ValueError("reward schedule id must be nonempty ASCII")
        if not self.knots or self.knots[0].completed_update != 0:
            raise ValueError("reward schedule must begin at update zero")
        updates = tuple(knot.completed_update for knot in self.knots)
        weights = tuple(knot.shaping_weight_ppm for knot in self.knots)
        if updates != tuple(sorted(set(updates))):
            raise ValueError("reward-knot updates must be strictly increasing")
        if any(right > left for left, right in zip(weights, weights[1:])):
            raise ValueError("shaping schedule must be monotone nonincreasing")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "schedule_id": self.schedule_id,
            "knots": [asdict(knot) for knot in self.knots],
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def weight_ppm(self, completed_update: int) -> int:
        if (
            isinstance(completed_update, bool)
            or not isinstance(completed_update, Integral)
            or completed_update < 0
        ):
            raise ValueError("completed update must be a nonnegative integer")
        update = int(completed_update)
        if update >= self.knots[-1].completed_update:
            return self.knots[-1].shaping_weight_ppm
        for left, right in zip(self.knots, self.knots[1:]):
            if left.completed_update <= update <= right.completed_update:
                offset = update - left.completed_update
                span = right.completed_update - left.completed_update
                delta = right.shaping_weight_ppm - left.shaping_weight_ppm
                # Python's floor division preserves monotonicity for a negative
                # delta and both declared endpoints are handled exactly.
                return left.shaping_weight_ppm + delta * offset // span
        raise AssertionError("reward schedule interval lookup failed")


@dataclass(frozen=True, slots=True)
class RewardBatch:
    """Detached reward components for one synchronous semantic decision."""

    raw_reward: Tensor
    scaled_raw_reward: Tensor
    shaping_reward: Tensor
    shaping_weight_ppm: Tensor
    optimizer_reward: Tensor

    def validate(self, lanes: int, *, reward_scale: float) -> None:
        shape = (lanes,)
        if self.raw_reward.shape != shape or self.raw_reward.dtype != torch.int64:
            raise ValueError("raw reward must be int64 [B]")
        for name, value in (
            ("scaled raw reward", self.scaled_raw_reward),
            ("shaping reward", self.shaping_reward),
            ("optimizer reward", self.optimizer_reward),
        ):
            if value.shape != shape or value.dtype != torch.float32:
                raise ValueError(f"{name} must be float32 [B]")
        if (
            self.shaping_weight_ppm.shape != shape
            or self.shaping_weight_ppm.dtype != torch.int64
        ):
            raise ValueError("shaping weights must be int64 [B]")
        tensors = (
            self.raw_reward,
            self.scaled_raw_reward,
            self.shaping_reward,
            self.shaping_weight_ppm,
            self.optimizer_reward,
        )
        if any(value.device.type != "cpu" or value.requires_grad for value in tensors):
            raise ValueError("reward audit tensors must be detached CPU values")
        if not all(
            torch.isfinite(value).all()
            for value in (
                self.scaled_raw_reward,
                self.shaping_reward,
                self.optimizer_reward,
            )
        ):
            raise ValueError("reward batch contains nonfinite values")
        if torch.any(
            (self.shaping_weight_ppm < 0) | (self.shaping_weight_ppm > 1_000_000)
        ):
            raise ValueError("shaping weight is outside the declared ppm range")
        expected_raw = self.raw_reward.to(torch.float32) / float(reward_scale)
        if not torch.equal(self.scaled_raw_reward, expected_raw):
            raise ValueError("scaled raw reward does not match raw score delta")
        expected = self.scaled_raw_reward + self.shaping_reward * (
            self.shaping_weight_ppm.to(torch.float32) / 1_000_000.0
        )
        if not torch.equal(self.optimizer_reward, expected):
            raise ValueError("optimizer reward does not match its audited components")
        score_only = self.shaping_weight_ppm == 0
        if torch.any(score_only) and not torch.equal(
            self.optimizer_reward[score_only], self.scaled_raw_reward[score_only]
        ):
            raise ValueError("zero shaping must be exactly score-only")


ShapingFunction = Callable[[Sequence[MacroTransition]], Tensor]


@runtime_checkable
class ShapingSpec(Protocol):
    shaping_id: str
    requires_events: bool
    gamma_tick: float
    critic_condition_features: int

    def manifest(self) -> dict[str, object]: ...

    def __call__(self, transitions: Sequence[MacroTransition]) -> Tensor: ...


@dataclass(frozen=True, slots=True)
class LinearGaugePotential:
    """Bounded, policy-invariant gauge feedback for score pretraining.

    The potential is evaluated from authoritative transition scalars rather
    than learner-visible observations.  It is deliberately linear: every
    retained gauge unit has one interpretable value, including near zero.
    """

    potential_scale: float = 1.0
    gamma_tick: float = 1.0
    version: ClassVar[str] = "linear-gauge-potential-v1"
    shaping_id: ClassVar[str] = "linear-gauge-potential-v1"
    requires_events: ClassVar[bool] = False
    critic_condition_features: ClassVar[int] = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.potential_scale, bool)
            or not isinstance(self.potential_scale, Real)
            or not math.isfinite(self.potential_scale)
            or self.potential_scale != 1.0
        ):
            raise ValueError("linear-gauge-potential-v1 requires potential_scale=1")
        if isinstance(self.gamma_tick, bool) or self.gamma_tick != 1.0:
            raise ValueError("linear-gauge-potential-v1 requires gamma_tick=1")
        object.__setattr__(self, "potential_scale", float(self.potential_scale))
        object.__setattr__(self, "gamma_tick", float(self.gamma_tick))

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "shaping_id": self.shaping_id,
            "source": "authoritative_transition_int64",
            "potential": "potential_scale*clamp(gauge,0,gauge_max)/gauge_max",
            "potential_scale": self.potential_scale,
            "gamma_tick": self.gamma_tick,
            "macro_discount": "gamma_tick**elapsed_ticks",
            "terminal": "zero_on_true_termination",
            "truncation": "retain_transition_end_potential",
            "autoreset": "excluded",
            "requires_events": self.requires_events,
            "critic_condition": "shaping_weight_ppm/1000000",
            "critic_condition_features": self.critic_condition_features,
            "actor_condition_features": 0,
            "output": "detached_cpu_float32",
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def _potential(self, gauge: int, gauge_max: int) -> float:
        if (
            isinstance(gauge, bool)
            or not isinstance(gauge, int)
            or isinstance(gauge_max, bool)
            or not isinstance(gauge_max, int)
        ):
            raise TypeError("gauge potential requires canonical integer fields")
        if gauge_max <= 0:
            raise ValueError("gauge maximum must be positive")
        retained = min(max(gauge, 0), gauge_max)
        return self.potential_scale * retained / gauge_max

    def __call__(self, transitions: Sequence[MacroTransition]) -> Tensor:
        values: list[float] = []
        for transition in transitions:
            if transition.elapsed_ticks <= 0:
                raise ValueError("gauge shaping requires positive elapsed ticks")
            before = self._potential(transition.start_gauge, transition.gauge_max)
            after = (
                0.0
                if transition.terminated
                else self._potential(transition.end_gauge, transition.gauge_max)
            )
            discount = self.gamma_tick**transition.elapsed_ticks
            values.append(discount * after - before)
        return torch.tensor(values, dtype=torch.float32)


class RewardComposer:
    """Compose optimizer rewards while preserving raw-score authority."""

    version = "reward-composer-v1"
    __slots__ = (
        "_reward_scale",
        "_shaping_id",
        "_shaping",
        "_requires_events",
        "_shaping_spec",
        "_shaping_spec_manifest",
        "_manifest",
        "_sha256",
    )

    def __init__(
        self,
        *,
        reward_scale: float = 1.0,
        shaping_id: str = "none",
        shaping: ShapingFunction | None = None,
        requires_events: bool = False,
        shaping_spec: ShapingSpec | None = None,
    ) -> None:
        if (
            isinstance(reward_scale, bool)
            or not isinstance(reward_scale, Real)
            or not math.isfinite(reward_scale)
            or reward_scale <= 0
        ):
            raise ValueError("reward scale must be finite and positive")
        if (
            not isinstance(shaping_id, str)
            or not shaping_id
            or not shaping_id.isascii()
        ):
            raise ValueError("shaping id must be nonempty ASCII")
        if shaping_spec is not None and (
            shaping is not None or shaping_id != "none" or requires_events
        ):
            raise ValueError(
                "a shaping spec cannot be combined with legacy shaping arguments"
            )
        if shaping_spec is None and isinstance(shaping, ShapingSpec):
            raise ValueError("manifested shaping must use the shaping_spec argument")
        if shaping_spec is not None:
            if not isinstance(shaping_spec, ShapingSpec):
                raise TypeError("shaping spec does not implement its runtime contract")
            shaping_id = shaping_spec.shaping_id
            shaping = shaping_spec
            requires_events = shaping_spec.requires_events
            if (
                not isinstance(shaping_spec.critic_condition_features, int)
                or isinstance(shaping_spec.critic_condition_features, bool)
                or shaping_spec.critic_condition_features not in (0, 1)
            ):
                raise ValueError("shaping spec critic-condition width is unsupported")
            if not isinstance(requires_events, bool):
                raise TypeError("shaping spec event requirement must be boolean")
            if (
                isinstance(shaping_spec.gamma_tick, bool)
                or not isinstance(shaping_spec.gamma_tick, Real)
                or not math.isfinite(shaping_spec.gamma_tick)
                or shaping_spec.gamma_tick <= 0
            ):
                raise ValueError("shaping spec gamma_tick must be finite and positive")
        if (
            not isinstance(shaping_id, str)
            or not shaping_id
            or not shaping_id.isascii()
        ):
            raise ValueError("shaping id must be nonempty ASCII")
        if shaping is None and shaping_id != "none":
            raise ValueError("a nontrivial shaping id requires a shaping function")
        if shaping is not None and shaping_id == "none":
            raise ValueError("a shaping function requires a versioned shaping id")
        shaping_spec_manifest = (
            None
            if shaping_spec is None
            else self._canonical_shaping_manifest(shaping_spec)
        )
        if shaping_spec_manifest is not None:
            declared = {
                "shaping_id": shaping_id,
                "requires_events": requires_events,
                "gamma_tick": float(shaping_spec.gamma_tick),
                "critic_condition_features": shaping_spec.critic_condition_features,
            }
            if any(
                shaping_spec_manifest.get(key) != value
                for key, value in declared.items()
            ):
                raise ValueError(
                    "shaping spec manifest contradicts its runtime contract"
                )
        self._reward_scale = float(reward_scale)
        self._shaping_id = shaping_id
        self._shaping = shaping
        self._requires_events = bool(requires_events)
        self._shaping_spec = shaping_spec
        self._shaping_spec_manifest = shaping_spec_manifest
        manifest: dict[str, object] = {
            "version": self.version,
            "reward_scale": self._reward_scale,
            "raw_reward": "score_after - score_before",
            "shaping_id": self._shaping_id,
            "requires_events": self._requires_events,
            "clip": False,
        }
        if shaping_spec_manifest is not None:
            manifest["shaping_spec"] = shaping_spec_manifest
        self._manifest = manifest
        payload = json.dumps(
            manifest, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        self._sha256 = hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _canonical_shaping_manifest(spec: ShapingSpec) -> dict[str, object]:
        try:
            manifest = json.loads(
                json.dumps(
                    spec.manifest(),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("shaping spec manifest must be canonical JSON") from exc
        if not isinstance(manifest, dict):
            raise TypeError("shaping spec manifest must be an object")
        return manifest

    @property
    def reward_scale(self) -> float:
        return self._reward_scale

    @property
    def shaping_id(self) -> str:
        return self._shaping_id

    @property
    def shaping(self) -> ShapingFunction | None:
        return self._shaping

    @property
    def requires_events(self) -> bool:
        return self._requires_events

    @property
    def shaping_spec(self) -> ShapingSpec | None:
        return self._shaping_spec

    @property
    def critic_condition_features(self) -> int:
        return (
            0
            if self._shaping_spec_manifest is None
            else int(self._shaping_spec_manifest["critic_condition_features"])
        )

    @property
    def shaping_gamma_tick(self) -> float | None:
        return (
            None
            if self._shaping_spec_manifest is None
            else float(self._shaping_spec_manifest["gamma_tick"])
        )

    def manifest(self) -> dict[str, object]:
        return json.loads(json.dumps(self._manifest))

    def validate_identity(self) -> None:
        """Fail before use if an owned shaping spec changed its declaration."""

        if self._shaping_spec is not None:
            manifest = self._canonical_shaping_manifest(self._shaping_spec)
            contract = {
                "shaping_id": self._shaping_spec.shaping_id,
                "requires_events": self._shaping_spec.requires_events,
                "gamma_tick": float(self._shaping_spec.gamma_tick),
                "critic_condition_features": (
                    self._shaping_spec.critic_condition_features
                ),
            }
            if manifest != self._shaping_spec_manifest or any(
                self._shaping_spec_manifest.get(key) != value
                for key, value in contract.items()
            ):
                raise RuntimeError(
                    "shaping spec manifest changed after composition setup"
                )

    @property
    def sha256(self) -> str:
        return self._sha256

    def compose(
        self,
        transitions: Sequence[MacroTransition],
        shaping_weight_ppm: Tensor,
    ) -> RewardBatch:
        self.validate_identity()
        lanes = len(transitions)
        if lanes <= 0:
            raise ValueError("reward composition requires at least one transition")
        if (
            shaping_weight_ppm.shape != (lanes,)
            or shaping_weight_ppm.dtype != torch.int64
            or shaping_weight_ppm.device.type != "cpu"
            or shaping_weight_ppm.requires_grad
        ):
            raise ValueError("shaping weights must be detached CPU int64 [B]")
        raw = torch.tensor(
            [transition.raw_reward for transition in transitions], dtype=torch.int64
        )
        scaled = raw.to(torch.float32) / self.reward_scale
        active = shaping_weight_ppm != 0
        if not torch.any(active):
            shaping = torch.zeros(lanes, dtype=torch.float32)
            optimizer = scaled.clone()
        else:
            if self.shaping is None:
                raise ValueError("nonzero shaping weight has no shaping function")
            active_indices = torch.nonzero(active, as_tuple=False).flatten().tolist()
            active_transitions = tuple(transitions[index] for index in active_indices)
            if self.requires_events and any(
                transition.diagnostics.event_count > 0
                and not transition.diagnostics.events
                for transition in active_transitions
            ):
                raise ValueError(
                    "event-dependent reward requires captured event payloads"
                )
            active_shaping = self.shaping(active_transitions)
            if (
                not isinstance(active_shaping, Tensor)
                or active_shaping.shape != (len(active_indices),)
                or active_shaping.dtype != torch.float32
                or active_shaping.device.type != "cpu"
                or active_shaping.requires_grad
                or not torch.isfinite(active_shaping).all()
            ):
                raise ValueError(
                    "shaping function returned an invalid active-lane batch"
                )
            shaping = torch.zeros(lanes, dtype=torch.float32)
            shaping[active] = active_shaping
            optimizer = scaled + shaping * (
                shaping_weight_ppm.to(torch.float32) / 1_000_000.0
            )
        result = RewardBatch(
            raw,
            scaled,
            shaping,
            shaping_weight_ppm.clone(),
            optimizer,
        )
        result.validate(lanes, reward_scale=self.reward_scale)
        return result
