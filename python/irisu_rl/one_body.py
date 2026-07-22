"""Versioned one-body curriculum used by the R2b learning proof."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import Tensor

from irisu_env import EventKind, PaddedVectorEnv

from .actions import ActionSpec, SemanticAction, SemanticActionKind
from .encoding import EncodedBatch, TeacherStateEncoder
from .models import RecurrentActorCritic
from .ppo import RecurrentTrainingBatch
from .torch_distribution import (
    ActionTensor,
    LogProbabilityComponents,
    TorchConditionalActionDistribution,
)


@dataclass(frozen=True, slots=True)
class OneBodySpec:
    version: str = "one-body-direct-hit-v1"
    train_heights: tuple[float, ...] = (60.0, 80.0, 100.0, 120.0, 140.0)
    calibration_heights: tuple[float, ...] = (50.0, 150.0)
    validation_heights: tuple[float, ...] = (70.0, 130.0)
    test_heights: tuple[float, ...] = (90.0, 110.0)
    hit_weight: float = 0.75
    aim_weight: float = 0.25
    aim_sigma: float = 0.20

    def __post_init__(self) -> None:
        groups = (
            self.train_heights,
            self.calibration_heights,
            self.validation_heights,
            self.test_heights,
        )
        flattened = [height for group in groups for height in group]
        if any(not math.isfinite(height) for height in flattened):
            raise ValueError("one-body heights must be finite")
        if len(flattened) != len(set(flattened)):
            raise ValueError("one-body height families must be disjoint")
        if (
            not math.isclose(self.hit_weight + self.aim_weight, 1.0)
            or self.hit_weight <= 0
            or self.aim_weight <= 0
            or not 0 < self.aim_sigma < 1
        ):
            raise ValueError("one-body reward parameters are invalid")

    def manifest(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.manifest(), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def mechanics_config(height: float) -> dict[str, int | float]:
        if not math.isfinite(height):
            raise ValueError("initial height must be finite")
        return {
            "initial_rotten_count": 0,
            "initial_falling_count": 1,
            "initial_falling_y": float(height),
            "spawn_y": -200.0,
            "max_episode_ticks": 500,
        }


@dataclass(frozen=True, slots=True)
class OneBodyOutcome:
    raw_reward: Tensor
    optimizer_reward: Tensor
    hit: Tensor
    aim_score: Tensor
    target_xy: Tensor
    action_xy: Tensor
    elapsed_ticks: Tensor


def concatenate_encoded(batches: Sequence[EncodedBatch]) -> EncodedBatch:
    if not batches:
        raise ValueError("at least one encoded batch is required")
    schema = batches[0].schema
    if any(batch.schema != schema for batch in batches):
        raise ValueError("encoded batches use different schemas")
    return EncodedBatch(
        np.concatenate([batch.global_features for batch in batches]),
        np.concatenate([batch.body_features for batch in batches]),
        np.concatenate([batch.body_mask for batch in batches]),
        np.concatenate([batch.source_tick for batch in batches]),
        np.concatenate([batch.health_flags for batch in batches]),
        schema,
    )


def encoded_to_torch(batch: EncodedBatch) -> tuple[Tensor, Tensor, Tensor]:
    batch.validate()
    return (
        torch.from_numpy(batch.global_features).unsqueeze(0),
        torch.from_numpy(batch.body_features).unsqueeze(0),
        torch.from_numpy(batch.body_mask).unsqueeze(0),
    )


def weak_only_masks(
    leading_shape: tuple[int, ...], action_spec: ActionSpec
) -> tuple[Tensor, Tensor]:
    kind = torch.zeros((*leading_shape, 3), dtype=torch.bool)
    kind[..., int(SemanticActionKind.FIRE_WEAK)] = True
    wait = torch.ones((*leading_shape, len(action_spec.wait_choices)), dtype=torch.bool)
    return kind, wait


def policy_distribution(
    model: RecurrentActorCritic, observations: EncodedBatch
) -> tuple[TorchConditionalActionDistribution, Tensor]:
    global_features, body_features, body_mask = encoded_to_torch(observations)
    lanes = global_features.shape[1]
    output = model(
        global_features,
        body_features,
        body_mask,
        model.initial_state(lanes),
        reset_before=torch.ones((1, lanes), dtype=torch.bool),
    )
    kind_mask, wait_mask = weak_only_masks((1, lanes), model.action_spec)
    return (
        TorchConditionalActionDistribution(
            output.kind_logits,
            output.wait_logits,
            output.coordinate_alpha,
            output.coordinate_beta,
            spec=model.action_spec,
            kind_mask=kind_mask,
            wait_mask=wait_mask,
        ),
        output.values,
    )


def expert_actions(target_xy: Tensor) -> ActionTensor:
    if target_xy.ndim != 2 or target_xy.shape[-1] != 2:
        raise ValueError("one-body targets must have shape [B, 2]")
    lanes = target_xy.shape[0]
    return ActionTensor(
        torch.ones((1, lanes), dtype=torch.long),
        torch.zeros((1, lanes), dtype=torch.long),
        target_xy.unsqueeze(0).clamp(1e-6, 1.0 - 1e-6),
    )


def one_body_training_batch(
    model: RecurrentActorCritic,
    observations: EncodedBatch,
    actions: ActionTensor,
    old_log_prob: Tensor,
    old_log_prob_components: LogProbabilityComponents,
    old_values: Tensor,
    rewards: Tensor,
) -> RecurrentTrainingBatch:
    global_features, body_features, body_mask = encoded_to_torch(observations)
    lanes = global_features.shape[1]
    expected = (1, lanes)
    if rewards.shape != (lanes,) or rewards.dtype != torch.float32:
        raise ValueError("one-body optimizer rewards must be float32 [B]")
    kind_mask, wait_mask = weak_only_masks(expected, model.action_spec)
    valid = torch.ones(expected, dtype=torch.bool)
    return RecurrentTrainingBatch(
        global_features,
        body_features,
        body_mask,
        torch.ones(expected, dtype=torch.bool),
        model.initial_state(lanes),
        ActionTensor(
            actions.kind.detach(), actions.wait_index.detach(), actions.xy.detach()
        ),
        old_log_prob.detach(),
        old_log_prob_components.kind.detach(),
        old_log_prob_components.wait.detach(),
        old_log_prob_components.coordinates.detach(),
        old_values.detach(),
        rewards.unsqueeze(0) - old_values.detach(),
        rewards.unsqueeze(0),
        valid,
        valid.clone(),
        kind_mask,
        wait_mask,
    )


class OneBodyTask:
    """One legal weak click against one reset-created body, then task reset."""

    def __init__(
        self,
        lanes: int,
        height: float,
        *,
        library_path: str | Path | None = None,
        worker_path: str | Path | None = None,
        physics_backend: str = "portable",
        spec: OneBodySpec | None = None,
    ) -> None:
        self.spec = spec or OneBodySpec()
        self.action_spec = ActionSpec()
        self.encoder = TeacherStateEncoder()
        self.height = float(height)
        self.env = PaddedVectorEnv(
            lanes,
            library_path=library_path,
            worker_path=worker_path,
            physics_backend=physics_backend,
            config=self.spec.mechanics_config(height),
        )
        self.lanes = lanes
        hashes = {int(env.config_hash()) for env in self.env.envs}
        if len(hashes) != 1:
            self.env.close()
            raise RuntimeError("one-body lanes disagree on mechanics config hash")
        self.config_hash = hashes.pop()
        self._target_xy: Tensor | None = None
        self._target_ids: tuple[int, ...] = ()
        self._start_ticks: tuple[int, ...] = ()
        self._start_scores: tuple[int, ...] = ()
        self._poisoned = False

    def close(self) -> None:
        self.env.close()

    def __enter__(self) -> OneBodyTask:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def reset(self, seeds: Sequence[int]) -> EncodedBatch:
        if self._poisoned:
            raise RuntimeError("poisoned one-body task must be recreated")
        if len(seeds) != self.lanes:
            raise ValueError("one-body seed count must equal lane count")
        observations, _ = self.env.reset(seed=seeds)
        encoded = self.encoder.encode(observations)
        if not np.all(encoded.body_mask.sum(axis=1) == 1):
            raise RuntimeError("one-body reset did not produce exactly one body")
        x_index = encoded.schema.body_features.index("effect_x_norm")
        y_index = encoded.schema.body_features.index("effect_y_norm")
        self._target_xy = torch.from_numpy(
            encoded.body_features[:, 0, [x_index, y_index]].copy()
        )
        self._target_ids = tuple(int(value.bodies[0].id) for value in observations)
        self._start_ticks = tuple(int(value.tick) for value in observations)
        self._start_scores = tuple(int(value.score) for value in observations)
        return encoded

    def _fail_closed(self, message: str) -> None:
        self._poisoned = True
        raise RuntimeError(message)

    @property
    def target_xy(self) -> Tensor:
        if self._target_xy is None:
            raise RuntimeError("reset one-body task before reading targets")
        return self._target_xy.clone()

    @staticmethod
    def _event_fields(info: dict[str, object]) -> tuple[tuple[int, int], ...]:
        # Materialize lazy views before any following primitive invalidates them.
        return tuple(
            (int(getattr(event, "kind", -1)), int(getattr(event, "b", 0)))
            for event in info.get("events", ())
        )

    def step(self, actions: ActionTensor) -> OneBodyOutcome:
        if self._target_xy is None:
            raise RuntimeError("reset one-body task before stepping")
        actions.validate(torch.Size((1, self.lanes)))
        if torch.any(actions.kind != int(SemanticActionKind.FIRE_WEAK)):
            raise ValueError("one-body-v1 permits only weak shots")
        semantic = tuple(
            SemanticAction.weak(float(x), float(y)) for x, y in actions.xy[0]
        )
        press = [self.action_spec.press(action) for action in semantic]
        try:
            _, press_reward, terminated, truncated, infos = self.env.step(press)
            press_events = [self._event_fields(info) for info in infos]
        except Exception:
            self._poisoned = True
            raise
        if any(
            bool(info.get("invalid_action", False))
            or int(info.get("config_hash", -1)) != self.config_hash
            for info in infos
        ):
            self._fail_closed("one-body press violated action/config identity")
        if any(terminated) or any(truncated):
            self._fail_closed("one-body task ended during the shot press")
        release = [self.action_spec.release() for _ in range(self.lanes)]
        try:
            final, release_reward, terminated, truncated, infos = self.env.step(release)
            release_events = [self._event_fields(info) for info in infos]
        except Exception:
            self._poisoned = True
            raise
        if any(
            bool(info.get("invalid_action", False))
            or int(info.get("config_hash", -1)) != self.config_hash
            for info in infos
        ):
            self._fail_closed("one-body release violated action/config identity")
        if any(terminated) or any(truncated):
            self._fail_closed("one-body task ended during release")
        hit_kind = int(EventKind.PROJECTILE_HIT)
        hit = torch.tensor(
            [
                any(
                    kind == hit_kind and body == target
                    for kind, body in press_events[lane] + release_events[lane]
                )
                for lane, target in enumerate(self._target_ids)
            ],
            dtype=torch.bool,
        )
        action_xy = actions.xy[0].detach().cpu()
        target_xy = self._target_xy
        error = (action_xy - target_xy).square().sum(dim=-1)
        aim = torch.exp(-error / (2 * self.spec.aim_sigma**2))
        reward = self.spec.hit_weight * hit.float() + self.spec.aim_weight * aim
        raw = torch.tensor(
            [left + right for left, right in zip(press_reward, release_reward)],
            dtype=torch.int64,
        )
        score_delta = torch.tensor(
            [int(value.score) - start for value, start in zip(final, self._start_scores)],
            dtype=torch.int64,
        )
        if not torch.equal(raw, score_delta):
            self._fail_closed("one-body raw reward does not equal score delta")
        elapsed = torch.tensor(
            [int(value.tick) - start for value, start in zip(final, self._start_ticks)],
            dtype=torch.int64,
        )
        if torch.any(elapsed != 2):
            self._fail_closed("one-body shot macro did not advance exactly two ticks")
        self._target_xy = None
        return OneBodyOutcome(raw, reward, hit, aim, target_xy, action_xy, elapsed)
