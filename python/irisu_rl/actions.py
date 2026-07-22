"""Validated semantic actions and conditional reference distribution math."""

from __future__ import annotations

import math
import hashlib
import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from numbers import Real
from typing import Any

import numpy as np

from irisu_env import Action

ACTION_COORDINATE_EPSILON = 1e-6


class SemanticActionKind(IntEnum):
    WAIT = 0
    FIRE_WEAK = 1
    FIRE_STRONG = 2


@dataclass(frozen=True, slots=True)
class SemanticAction:
    kind: SemanticActionKind | int
    wait_ticks: int = 1
    x_norm: float = 0.0
    y_norm: float = 0.0

    @classmethod
    def wait(cls, ticks: int) -> SemanticAction:
        return cls(SemanticActionKind.WAIT, wait_ticks=ticks)

    @classmethod
    def weak(cls, x_norm: float, y_norm: float) -> SemanticAction:
        return cls(SemanticActionKind.FIRE_WEAK, x_norm=x_norm, y_norm=y_norm)

    @classmethod
    def strong(cls, x_norm: float, y_norm: float) -> SemanticAction:
        return cls(SemanticActionKind.FIRE_STRONG, x_norm=x_norm, y_norm=y_norm)


@dataclass(frozen=True, slots=True)
class ActionSpec:
    version: str = "deployment-v1"
    wait_choices: tuple[int, ...] = tuple(range(1, 101))
    client_width: float = 640.0
    client_height: float = 480.0
    press_ticks: int = 1
    release_ticks: int = 1
    timing_status: str = "provisional"
    allow_both: bool = False
    coordinate_log_prob_epsilon: float = ACTION_COORDINATE_EPSILON

    def __post_init__(self) -> None:
        if (
            not self.wait_choices
            or tuple(sorted(set(self.wait_choices))) != self.wait_choices
        ):
            raise ValueError("wait choices must be unique and strictly increasing")
        if self.wait_choices[0] < 1 or self.wait_choices[-1] > 100_000:
            raise ValueError("wait choices must fit the native legal range")
        if self.press_ticks != 1 or self.release_ticks < 1:
            raise ValueError("R1 supports one-tick press and positive release")
        if self.release_ticks > 100_000:
            raise ValueError("release duration exceeds native wait limit")
        if (
            not math.isfinite(self.client_width)
            or not math.isfinite(self.client_height)
            or self.client_width <= 0
            or self.client_height <= 0
        ):
            raise ValueError("client dimensions must be finite and positive")
        if self.timing_status not in {"provisional", "measured"}:
            raise ValueError("invalid timing status")
        if self.allow_both:
            raise ValueError("simultaneous shots are outside deployment-v1")
        if (
            not math.isfinite(self.coordinate_log_prob_epsilon)
            or not 0 < self.coordinate_log_prob_epsilon < 0.5
        ):
            raise ValueError("coordinate likelihood epsilon must be in (0, 0.5)")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "wait_choices": list(self.wait_choices),
            "client_width": self.client_width,
            "client_height": self.client_height,
            "press_ticks": self.press_ticks,
            "release_ticks": self.release_ticks,
            "timing_status": self.timing_status,
            "allow_both": self.allow_both,
            "coordinate_log_prob_epsilon": self.coordinate_log_prob_epsilon,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def validate(self, action: SemanticAction) -> SemanticAction:
        try:
            kind = SemanticActionKind(action.kind)
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown semantic action kind") from exc
        if kind is SemanticActionKind.WAIT:
            if (
                isinstance(action.wait_ticks, bool)
                or action.wait_ticks not in self.wait_choices
            ):
                raise ValueError("wait duration is not in the declared choices")
            return SemanticAction(kind, int(action.wait_ticks), 0.0, 0.0)
        if (
            not isinstance(action.x_norm, Real)
            or isinstance(action.x_norm, bool)
            or not isinstance(action.y_norm, Real)
            or isinstance(action.y_norm, bool)
        ):
            raise TypeError("shot coordinates must be real numbers")
        x, y = float(action.x_norm), float(action.y_norm)
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("shot coordinates must be finite")
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError("shot coordinates must be normalized to [0, 1]")
        return SemanticAction(kind, 1, x, y)

    def press(self, action: SemanticAction) -> Action:
        value = self.validate(action)
        if value.kind is SemanticActionKind.WAIT:
            return Action.wait(value.wait_ticks)
        x = value.x_norm * self.client_width
        y = value.y_norm * self.client_height
        if value.kind is SemanticActionKind.FIRE_WEAK:
            return Action.weak(x, y)
        return Action.strong(x, y)

    def release(self) -> Action:
        return Action.wait(self.release_ticks)

    def encode(self, action: SemanticAction) -> tuple[int, int, float, float]:
        value = self.validate(action)
        wait_index = (
            self.wait_choices.index(value.wait_ticks)
            if value.kind is SemanticActionKind.WAIT
            else 0
        )
        return int(value.kind), wait_index, value.x_norm, value.y_norm

    def decode(self, kind: int, wait_index: int, x: float, y: float) -> SemanticAction:
        parsed = SemanticActionKind(kind)
        if parsed is SemanticActionKind.WAIT:
            if not 0 <= wait_index < len(self.wait_choices):
                raise ValueError("wait index out of range")
            return self.validate(SemanticAction.wait(self.wait_choices[wait_index]))
        constructor = (
            SemanticAction.weak
            if parsed is SemanticActionKind.FIRE_WEAK
            else SemanticAction.strong
        )
        return self.validate(constructor(x, y))

    def serialize(self, action: SemanticAction) -> bytes:
        kind, wait_index, x, y = self.encode(action)
        return struct.pack("<BIdd", kind, wait_index, x, y)

    def deserialize(self, payload: bytes) -> SemanticAction:
        if len(payload) != struct.calcsize("<BIdd"):
            raise ValueError("invalid serialized semantic action size")
        return self.decode(*struct.unpack("<BIdd", payload))


def _masked_log_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != logits.shape:
        raise ValueError("action mask shape mismatch")
    if not np.all(mask.any(axis=-1)):
        raise ValueError("all-masked action branch")
    masked = np.where(mask, logits, -np.inf)
    maximum = np.max(masked, axis=-1, keepdims=True)
    shifted = masked - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def _digamma(value: np.ndarray) -> np.ndarray:
    """Accurate positive-domain digamma without a SciPy dependency."""

    x = np.asarray(value, dtype=np.float64).copy()
    result = np.zeros_like(x)
    while np.any(x < 8.0):
        mask = x < 8.0
        result[mask] -= 1.0 / x[mask]
        x[mask] += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    result += (
        np.log(x)
        - 0.5 * inv
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 / 252.0))
    )
    return result


def _beta_log_prob(
    value: np.ndarray, alpha: np.ndarray, beta: np.ndarray, epsilon: float
) -> np.ndarray:
    x = np.clip(value, epsilon, 1.0 - epsilon)
    log_norm = np.vectorize(math.lgamma)(alpha) + np.vectorize(math.lgamma)(beta)
    log_norm -= np.vectorize(math.lgamma)(alpha + beta)
    return (alpha - 1) * np.log(x) + (beta - 1) * np.log1p(-x) - log_norm


def _beta_entropy(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
    log_b = np.vectorize(math.lgamma)(alpha) + np.vectorize(math.lgamma)(beta)
    log_b -= np.vectorize(math.lgamma)(alpha + beta)
    return (
        log_b
        - (alpha - 1) * _digamma(alpha)
        - (beta - 1) * _digamma(beta)
        + (alpha + beta - 2) * _digamma(alpha + beta)
    )


class ConditionalActionDistribution:
    """Framework-independent likelihood oracle for the R2 policy head.

    Coordinate parameters have shape ``[batch, 2 shot kinds, 2 coordinates]``.
    Only the selected branch contributes to likelihood and entropy.
    """

    def __init__(
        self,
        kind_logits: Any,
        wait_logits: Any,
        coordinate_alpha: Any,
        coordinate_beta: Any,
        *,
        spec: ActionSpec | None = None,
        kind_mask: Any | None = None,
        wait_mask: Any | None = None,
    ) -> None:
        self.spec = spec or ActionSpec()
        self.kind_logits = np.array(kind_logits, dtype=np.float64, copy=True)
        self.wait_logits = np.array(wait_logits, dtype=np.float64, copy=True)
        self.alpha = np.array(coordinate_alpha, dtype=np.float64, copy=True)
        self.beta = np.array(coordinate_beta, dtype=np.float64, copy=True)
        batch = self.kind_logits.shape[0] if self.kind_logits.ndim == 2 else -1
        if batch <= 0 or self.kind_logits.shape != (batch, 3):
            raise ValueError("kind_logits must have shape [batch, 3]")
        if self.wait_logits.shape != (batch, len(self.spec.wait_choices)):
            raise ValueError("wait_logits shape does not match wait choices")
        if self.alpha.shape != (batch, 2, 2) or self.beta.shape != self.alpha.shape:
            raise ValueError("coordinate parameters must have shape [batch, 2, 2]")
        if not all(
            np.all(np.isfinite(value))
            for value in (self.kind_logits, self.wait_logits, self.alpha, self.beta)
        ):
            raise ValueError("distribution parameters must be finite")
        if np.any(self.alpha <= 0) or np.any(self.beta <= 0):
            raise ValueError("Beta concentrations must be positive")
        self.kind_mask = (
            np.ones_like(self.kind_logits, dtype=np.bool_)
            if kind_mask is None
            else np.array(kind_mask, dtype=np.bool_, copy=True)
        )
        self.wait_mask = (
            np.ones_like(self.wait_logits, dtype=np.bool_)
            if wait_mask is None
            else np.array(wait_mask, dtype=np.bool_, copy=True)
        )
        self._kind_logp = _masked_log_softmax(self.kind_logits, self.kind_mask)
        if self.wait_mask.shape != self.wait_logits.shape:
            raise ValueError("action mask shape mismatch")
        missing_wait = ~self.wait_mask.any(axis=1)
        if np.any(missing_wait & self.kind_mask[:, 0]):
            raise ValueError("all-masked active wait branch")
        effective_wait_mask = self.wait_mask.copy()
        effective_wait_mask[missing_wait, 0] = True
        self._wait_logp = _masked_log_softmax(self.wait_logits, effective_wait_mask)

    @property
    def batch_size(self) -> int:
        return self.kind_logits.shape[0]

    def log_prob(
        self, actions: list[SemanticAction] | tuple[SemanticAction, ...]
    ) -> np.ndarray:
        if len(actions) != self.batch_size:
            raise ValueError("action batch length mismatch")
        result = np.empty(self.batch_size, dtype=np.float64)
        for lane, supplied in enumerate(actions):
            action = self.spec.validate(supplied)
            kind = int(action.kind)
            value = self._kind_logp[lane, kind]
            if action.kind is SemanticActionKind.WAIT:
                index = self.spec.wait_choices.index(action.wait_ticks)
                value += self._wait_logp[lane, index]
            else:
                branch = kind - 1
                xy = np.asarray((action.x_norm, action.y_norm), dtype=np.float64)
                value += _beta_log_prob(
                    xy,
                    self.alpha[lane, branch],
                    self.beta[lane, branch],
                    self.spec.coordinate_log_prob_epsilon,
                ).sum()
            result[lane] = value
        return result

    def entropy(self) -> np.ndarray:
        kind_p = np.exp(self._kind_logp)
        kind_terms = np.zeros_like(kind_p)
        np.multiply(kind_p, self._kind_logp, out=kind_terms, where=self.kind_mask)
        result = -kind_terms.sum(axis=1)
        wait_p = np.exp(self._wait_logp)
        wait_terms = np.zeros_like(wait_p)
        np.multiply(wait_p, self._wait_logp, out=wait_terms, where=self.wait_mask)
        wait_entropy = -wait_terms.sum(axis=1)
        result += kind_p[:, 0] * wait_entropy
        coordinate_entropy = _beta_entropy(self.alpha, self.beta).sum(axis=2)
        result += kind_p[:, 1] * coordinate_entropy[:, 0]
        result += kind_p[:, 2] * coordinate_entropy[:, 1]
        return result

    def sample(self, rng: np.random.Generator) -> tuple[SemanticAction, ...]:
        actions: list[SemanticAction] = []
        for lane in range(self.batch_size):
            kind = int(rng.choice(3, p=np.exp(self._kind_logp[lane])))
            if kind == 0:
                index = int(
                    rng.choice(
                        len(self.spec.wait_choices), p=np.exp(self._wait_logp[lane])
                    )
                )
                actions.append(SemanticAction.wait(self.spec.wait_choices[index]))
            else:
                branch = kind - 1
                xy = rng.beta(self.alpha[lane, branch], self.beta[lane, branch])
                constructor = (
                    SemanticAction.weak if kind == 1 else SemanticAction.strong
                )
                actions.append(constructor(float(xy[0]), float(xy[1])))
        return tuple(actions)

    def deterministic(self) -> tuple[SemanticAction, ...]:
        actions: list[SemanticAction] = []
        for lane in range(self.batch_size):
            kind = int(np.argmax(self._kind_logp[lane]))
            if kind == 0:
                index = int(np.argmax(self._wait_logp[lane]))
                actions.append(SemanticAction.wait(self.spec.wait_choices[index]))
            else:
                branch = kind - 1
                xy = self.alpha[lane, branch] / (
                    self.alpha[lane, branch] + self.beta[lane, branch]
                )
                constructor = (
                    SemanticAction.weak if kind == 1 else SemanticAction.strong
                )
                actions.append(constructor(float(xy[0]), float(xy[1])))
        return tuple(actions)
