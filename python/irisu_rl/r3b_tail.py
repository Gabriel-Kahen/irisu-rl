"""Checkpointable proof that final PPO updates are exactly score-only."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Literal

import torch
from torch import Tensor


CollectionMode = Literal["train", "drain", "closed"]
TailPhase = Literal["sweep", "draining", "score_only", "complete"]

_ZERO_SHA256 = "0" * 64
_PHASES = {"sweep", "draining", "score_only", "complete"}


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_nonnegative(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return int(value)


def _finite_rows(value: object, width: int, label: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"decision audit is missing {label}")
    if len(value) != width:
        raise ValueError(f"decision audit {label} width mismatch")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"decision audit {label} must be numeric")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"decision audit {label} contains a nonfinite value")
        result.append(number)
    return tuple(result)


def _integer_rows(value: object, width: int, label: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"decision audit is missing {label}")
    if len(value) != width or any(
        isinstance(item, bool) or not isinstance(item, Integral) for item in value
    ):
        raise ValueError(f"decision audit {label} must contain {width} integers")
    return tuple(int(item) for item in value)


def _weights(value: object, *, label: str) -> tuple[int, ...]:
    if isinstance(value, Tensor):
        if (
            value.ndim != 1
            or value.numel() == 0
            or value.dtype != torch.int64
            or value.device.type != "cpu"
            or value.requires_grad
        ):
            raise ValueError(f"{label} must be detached CPU int64 [B]")
        supplied: Sequence[object] = value.tolist()
    else:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError(f"{label} must be a nonempty integer sequence")
        supplied = value
    if not supplied:
        raise ValueError(f"{label} must be a nonempty integer sequence")
    result = []
    for item in supplied:
        if (
            isinstance(item, bool)
            or not isinstance(item, Integral)
            or not 0 <= int(item) <= 1_000_000
        ):
            raise ValueError(f"{label} must contain canonical ppm integers")
        result.append(int(item))
    return tuple(result)


def _audit_manifest(
    audit: object, *, reward_scale: float, reward_sha256: str
) -> dict[str, object]:
    """Validate and reduce one collection audit to score-only evidence."""

    if audit is None:
        raise ValueError("collection audit is required")
    decision_rows = _canonical_nonnegative(
        getattr(audit, "decision_rows", None), "audit decision_rows"
    )
    transitions = _canonical_nonnegative(
        getattr(audit, "transitions", None), "audit transitions"
    )
    if decision_rows == 0 or transitions == 0:
        raise ValueError("collection audit must contain decisions and transitions")
    decisions = getattr(audit, "decisions", None)
    if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
        raise ValueError("collection audit is missing decision audits")
    if len(decisions) != decision_rows:
        raise ValueError("collection decision-row count mismatch")
    if getattr(audit, "reward_sha256", None) != reward_sha256:
        raise ValueError("collection reward identity mismatch")

    optimizer_reward = getattr(audit, "optimizer_reward", None)
    if (
        isinstance(optimizer_reward, bool)
        or not isinstance(optimizer_reward, Real)
        or not math.isfinite(float(optimizer_reward))
    ):
        raise ValueError("collection optimizer reward must be finite")
    raw_reward = getattr(audit, "raw_reward", None)
    invalid_actions = getattr(audit, "invalid_actions", None)
    if isinstance(raw_reward, bool) or not isinstance(raw_reward, Integral):
        raise ValueError("collection raw reward must be an integer")
    if (
        isinstance(invalid_actions, bool)
        or not isinstance(invalid_actions, Integral)
        or invalid_actions < 0
    ):
        raise ValueError("collection invalid-action count is malformed")
    if invalid_actions != 0:
        raise ValueError("collection contains invalid actions")

    rows = []
    lane_width: int | None = None
    raw_total = 0
    optimizer_total = 0.0
    for decision in decisions:
        weights = _weights(
            getattr(decision, "shaping_weight_ppm", None),
            label="decision shaping weights",
        )
        if lane_width is None:
            lane_width = len(weights)
        elif len(weights) != lane_width:
            raise ValueError("decision audit lane width changed within a collection")
        raw = _integer_rows(
            getattr(decision, "raw_rewards", None),
            len(weights),
            "raw_rewards",
        )
        scaled = _finite_rows(
            getattr(decision, "scaled_raw_rewards", None),
            len(weights),
            "scaled_raw_rewards",
        )
        shaping = _finite_rows(
            getattr(decision, "shaping_rewards", None),
            len(weights),
            "shaping_rewards",
        )
        optimizer = _finite_rows(
            getattr(decision, "optimizer_rewards", None),
            len(weights),
            "optimizer_rewards",
        )
        expected_scaled = tuple(
            float(
                torch.tensor(int(raw_value), dtype=torch.int64)
                .to(torch.float32)
                .div(reward_scale)
            )
            for raw_value in raw
        )
        expected_optimizer = tuple(
            float(value)
            for value in (
                torch.tensor(expected_scaled, dtype=torch.float32)
                + torch.tensor(shaping, dtype=torch.float32)
                * (
                    torch.tensor(weights, dtype=torch.int64).to(torch.float32)
                    / 1_000_000.0
                )
            ).tolist()
        )
        if scaled != expected_scaled:
            raise ValueError("scaled raw reward does not match raw reward and scale")
        if optimizer != expected_optimizer:
            raise ValueError("optimizer reward does not match declared composition")
        for weight, shaping_value, raw_value, optimizer_value in zip(
            weights, shaping, scaled, optimizer
        ):
            if weight == 0 and (shaping_value != 0.0 or optimizer_value != raw_value):
                raise ValueError(
                    "zero-weight decision reward is not exactly score-only"
                )
        raw_total += sum(raw)
        optimizer_total += float(torch.tensor(optimizer, dtype=torch.float32).sum())
        rows.append(
            {
                "raw_rewards": list(raw),
                "shaping_weight_ppm": list(weights),
                "scaled_raw_rewards": list(scaled),
                "shaping_rewards": list(shaping),
                "optimizer_rewards": list(optimizer),
            }
        )
    if transitions != decision_rows * int(lane_width or 0):
        raise ValueError("collection transition count does not match decision audits")
    if int(raw_reward) != raw_total:
        raise ValueError("collection raw reward does not match decision rows")
    if float(optimizer_reward) != optimizer_total:
        raise ValueError("collection optimizer reward does not match decision rows")
    return {
        "decision_rows": decision_rows,
        "transitions": transitions,
        "raw_reward": int(raw_reward),
        "optimizer_reward": float(optimizer_reward),
        "invalid_actions": int(invalid_actions),
        "reward_sha256": reward_sha256,
        "decisions": rows,
    }


class ScoreOnlyTailController:
    """Enforce a sweep, drain, and bounded exact-score training lifecycle.

    ``completed_updates`` always refers to the trainer's optimizer-update clock.
    A drain collection leaves that clock unchanged. Optimizer updates use a
    two-step protocol so their audit is checked before mutation and matched
    again when the completed clock advances.
    """

    version = "score-only-tail-controller-v2"

    def __init__(
        self,
        sweep_updates: int,
        *,
        reward_scale: float,
        reward_sha256: str,
        minimum_score_only_updates: int = 400,
    ) -> None:
        if (
            isinstance(sweep_updates, bool)
            or not isinstance(sweep_updates, Integral)
            or sweep_updates <= 0
        ):
            raise ValueError("sweep update count must be a positive integer")
        if (
            isinstance(minimum_score_only_updates, bool)
            or not isinstance(minimum_score_only_updates, Integral)
            or minimum_score_only_updates < 400
        ):
            raise ValueError("score-only tail must contain at least 400 updates")
        if (
            isinstance(reward_scale, bool)
            or not isinstance(reward_scale, Real)
            or not math.isfinite(float(reward_scale))
            or reward_scale <= 0
            or not _is_sha256(reward_sha256)
            or reward_sha256 == _ZERO_SHA256
        ):
            raise ValueError("tail reward scale and identity are invalid")
        self.sweep_updates = int(sweep_updates)
        self.reward_scale = float(reward_scale)
        self.reward_sha256 = reward_sha256
        self.minimum_score_only_updates = int(minimum_score_only_updates)
        self.phase: TailPhase = "sweep"
        self.completed_updates = 0
        self.drain_collections = 0
        self.score_only_updates = 0
        self.event_head = _ZERO_SHA256
        self._pending_update: tuple[int, str] | None = None

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "sweep_updates": self.sweep_updates,
            "minimum_score_only_updates": self.minimum_score_only_updates,
            "reward_scale": self.reward_scale,
            "reward_sha256": self.reward_sha256,
            "score_only_weight_ppm": 0,
            "drain_advances_optimizer_clock": False,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    def _clock(self, completed_updates: object) -> int:
        value = _canonical_nonnegative(completed_updates, "completed update clock")
        if value != self.completed_updates:
            raise ValueError("tail and optimizer update clocks are discontinuous")
        return value

    def _append_event(self, kind: str, payload: Mapping[str, object]) -> None:
        self.event_head = _canonical_sha256(
            {
                "previous": self.event_head,
                "kind": kind,
                "completed_updates": self.completed_updates,
                "payload": dict(payload),
            }
        )

    def collection_mode(
        self,
        *,
        completed_updates: int,
        lane_shaping_weight_ppm: Sequence[int] | Tensor,
    ) -> CollectionMode:
        """Return whether the next collection trains, drains, or is closed."""

        if self._pending_update is not None:
            raise RuntimeError("an optimizer update is awaiting completion")
        self._clock(completed_updates)
        weights = _weights(lane_shaping_weight_ppm, label="lane shaping weights")
        all_zero = all(weight == 0 for weight in weights)
        if self.phase == "complete":
            if not all_zero:
                raise ValueError("shaping weight became nonzero after tail completion")
            return "closed"
        if self.phase == "sweep" and self.completed_updates < self.sweep_updates:
            return "train"
        if self.phase == "sweep":
            self.phase = "draining"
            self._append_event(
                "tail_drain_started",
                {"lane_shaping_weight_ppm": list(weights)},
            )
        if self.phase == "draining" and all_zero:
            self.phase = "score_only"
            self._append_event(
                "score_only_started", {"lane_shaping_weight_ppm": list(weights)}
            )
        elif self.phase == "score_only" and not all_zero:
            raise ValueError("shaping weight became nonzero during the score-only tail")
        return "drain" if self.phase == "draining" else "train"

    def record_drain(self, audit: object, *, completed_updates: int) -> str:
        """Record a collection that intentionally performed no PPO update."""

        if self._pending_update is not None:
            raise RuntimeError("an optimizer update is awaiting completion")
        self._clock(completed_updates)
        if self.phase != "draining":
            raise RuntimeError("drain evidence is only valid during tail draining")
        manifest = _audit_manifest(
            audit,
            reward_scale=self.reward_scale,
            reward_sha256=self.reward_sha256,
        )
        weights = [
            weight
            for row in manifest["decisions"]
            for weight in row["shaping_weight_ppm"]
        ]
        if not any(weight != 0 for weight in weights):
            raise ValueError("an all-zero collection must not be recorded as a drain")
        audit_sha256 = _canonical_sha256(manifest)
        self.drain_collections += 1
        self._append_event(
            "drain_collection",
            {
                "audit_sha256": audit_sha256,
                "drain_collections": self.drain_collections,
            },
        )
        return audit_sha256

    def validate_optimizer_update(
        self, audit: object, *, completed_updates: int
    ) -> str:
        """Validate a collection before allowing its PPO optimizer mutation."""

        if self._pending_update is not None:
            raise RuntimeError("an optimizer update is already awaiting completion")
        before = self._clock(completed_updates)
        if self.phase in {"draining", "complete"}:
            raise RuntimeError(f"optimizer updates are forbidden during {self.phase}")
        if self.phase == "sweep" and before >= self.sweep_updates:
            raise RuntimeError("tail collection mode must be resolved after the sweep")
        manifest = _audit_manifest(
            audit,
            reward_scale=self.reward_scale,
            reward_sha256=self.reward_sha256,
        )
        if self.phase == "score_only" and any(
            weight != 0
            for row in manifest["decisions"]
            for weight in row["shaping_weight_ppm"]
        ):
            raise ValueError("score-only optimizer audit contains nonzero shaping")
        audit_sha256 = _canonical_sha256(manifest)
        self._pending_update = (before, audit_sha256)
        return audit_sha256

    def record_optimizer_update(self, audit: object, *, completed_updates: int) -> str:
        """Commit one previously validated optimizer update at clock ``before+1``."""

        pending = self._pending_update
        if pending is None:
            raise RuntimeError("optimizer update was not validated before mutation")
        after = _canonical_nonnegative(completed_updates, "completed update clock")
        if after != pending[0] + 1 or pending[0] != self.completed_updates:
            raise ValueError("optimizer update clock did not advance by exactly one")
        manifest = _audit_manifest(
            audit,
            reward_scale=self.reward_scale,
            reward_sha256=self.reward_sha256,
        )
        audit_sha256 = _canonical_sha256(manifest)
        if audit_sha256 != pending[1]:
            raise ValueError("optimizer update audit changed after validation")
        self._pending_update = None
        self.completed_updates = after
        if self.phase == "score_only":
            self.score_only_updates += 1
        self._append_event(
            "optimizer_update",
            {
                "audit_sha256": audit_sha256,
                "phase": self.phase,
                "score_only_updates": self.score_only_updates,
            },
        )
        if self.phase == "score_only" and (
            self.score_only_updates >= self.minimum_score_only_updates
        ):
            self.phase = "complete"
            self._append_event(
                "score_only_complete",
                {"score_only_updates": self.score_only_updates},
            )
        return audit_sha256

    def state_dict(self) -> dict[str, object]:
        if self._pending_update is not None:
            raise RuntimeError("cannot checkpoint inside an optimizer transaction")
        self._validate_state()
        return {
            "version": self.version,
            "identity_sha256": self.sha256,
            "phase": self.phase,
            "completed_updates": self.completed_updates,
            "drain_collections": self.drain_collections,
            "score_only_updates": self.score_only_updates,
            "event_head": self.event_head,
        }

    def _validate_state(self) -> None:
        if self.phase not in _PHASES:
            raise ValueError("tail checkpoint phase is invalid")
        if not _is_sha256(self.event_head):
            raise ValueError("tail event-chain head is malformed")
        counters = (
            self.completed_updates,
            self.drain_collections,
            self.score_only_updates,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counters
        ):
            raise ValueError("tail checkpoint counters are malformed")
        if self.completed_updates > (
            self.sweep_updates + self.minimum_score_only_updates
        ):
            raise ValueError("tail checkpoint exceeds its optimizer budget")
        if self.score_only_updates != max(
            self.completed_updates - self.sweep_updates, 0
        ):
            raise ValueError("tail score-only and optimizer clocks disagree")
        if self.phase == "sweep" and (
            self.completed_updates > self.sweep_updates
            or self.drain_collections != 0
            or self.score_only_updates != 0
        ):
            raise ValueError("sweep checkpoint state is inconsistent")
        if self.phase == "draining" and (
            self.completed_updates != self.sweep_updates or self.score_only_updates != 0
        ):
            raise ValueError("draining checkpoint state is inconsistent")
        if self.phase == "score_only" and not (
            self.sweep_updates
            <= self.completed_updates
            < self.sweep_updates + self.minimum_score_only_updates
        ):
            raise ValueError("score-only checkpoint state is inconsistent")
        if self.phase == "complete" and (
            self.score_only_updates < self.minimum_score_only_updates
            or self.completed_updates != self.sweep_updates + self.score_only_updates
        ):
            raise ValueError("complete tail checkpoint state is inconsistent")

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        expected = {
            "version",
            "identity_sha256",
            "phase",
            "completed_updates",
            "drain_collections",
            "score_only_updates",
            "event_head",
        }
        if set(state) != expected or state["version"] != self.version:
            raise ValueError("tail checkpoint keys or version do not match")
        if state["identity_sha256"] != self.sha256:
            raise ValueError("tail checkpoint identity mismatch")
        if self._pending_update is not None:
            raise RuntimeError("cannot restore inside an optimizer transaction")
        prior = (
            self.phase,
            self.completed_updates,
            self.drain_collections,
            self.score_only_updates,
            self.event_head,
        )
        try:
            self.phase = state["phase"]  # type: ignore[assignment]
            self.completed_updates = state["completed_updates"]  # type: ignore[assignment]
            self.drain_collections = state["drain_collections"]  # type: ignore[assignment]
            self.score_only_updates = state["score_only_updates"]  # type: ignore[assignment]
            self.event_head = state["event_head"]  # type: ignore[assignment]
            self._validate_state()
        except BaseException:
            (
                self.phase,
                self.completed_updates,
                self.drain_collections,
                self.score_only_updates,
                self.event_head,
            ) = prior
            raise
