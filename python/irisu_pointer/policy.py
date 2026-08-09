"""Stateful inference and identity-safe lowering for pointer policies."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from irisu_env import Action
from irisu_rl.actions import ActionSpec, SemanticAction, SemanticActionKind
from irisu_rl.encoding import ActorTrackEncoder, EncodedBatch, TeacherStateEncoder
from irisu_rl.schema import ACTOR_VISION_V1

from .action import PointerActionSpec, decode_pointer_action
from .experts import PointerExpertDecision


@dataclass(frozen=True, slots=True)
class PointerPolicyDecision:
    """One recurrent decision plus confidence and its encoded row binding."""

    decision: PointerExpertDecision
    encoded: EncodedBatch
    kind_index: int
    wait_index: int
    target_index: int
    template_index: int
    kind_confidence: float
    branch_confidence: float
    confidence: float

    def __post_init__(self) -> None:
        for name in ("kind_confidence", "branch_confidence", "confidence"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


@dataclass(frozen=True, slots=True)
class ActorPointerPolicyDecision:
    """One actor-only pointer decision lowered to a legal primitive macro."""

    semantic_action: SemanticAction
    primitive_actions: tuple[Action, ...]
    encoded: EncodedBatch
    kind_index: int
    wait_index: int
    target_index: int
    template_index: int
    kind_confidence: float
    branch_confidence: float
    confidence: float

    def __post_init__(self) -> None:
        if not self.primitive_actions:
            raise ValueError("actor pointer macro cannot be empty")
        for name in ("kind_confidence", "branch_confidence", "confidence"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")


def encoded_body_ids(
    encoded: EncodedBatch,
    observation: Mapping[str, Any],
    *,
    row: int = 0,
) -> tuple[int | None, ...]:
    """Bind teacher rows back to current public IDs without exposing IDs to the model."""

    encoded.validate()
    if not 0 <= row < encoded.global_features.shape[0]:
        raise IndexError("encoded row is outside the batch")
    try:
        id_column = encoded.schema.body_features.index("id_scaled")
    except ValueError as exc:
        raise ValueError("identity binding requires a teacher schema with id_scaled") from exc
    public_ids = {
        int(body["id"])
        for body in observation.get("bodies", ())
        if isinstance(body, Mapping) and "id" in body
    }
    output: list[int | None] = []
    bound: set[int] = set()
    for index in range(encoded.schema.capacity):
        if not bool(encoded.body_mask[row, index]):
            output.append(None)
            continue
        body_id = round(float(encoded.body_features[row, index, id_column]) * 2**32)
        if body_id not in public_ids:
            raise RuntimeError("encoded body row does not bind to a current public body")
        if body_id in bound:
            raise RuntimeError("encoded body rows collide on one public body ID")
        bound.add(body_id)
        output.append(body_id)
    return tuple(output)


def target_index_for_decision(
    encoded: EncodedBatch,
    observation: Mapping[str, Any],
    decision: PointerExpertDecision,
    *,
    row: int = 0,
) -> int:
    """Return the unique encoded target row for a public-ID expert decision."""

    if decision.target_body_id is None:
        return 0
    matches = [
        index
        for index, body_id in enumerate(encoded_body_ids(encoded, observation, row=row))
        if body_id == decision.target_body_id
    ]
    if len(matches) != 1:
        raise RuntimeError("pointer target did not bind to exactly one encoded body row")
    return matches[0]


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("pointer model has no parameters") from exc


def _tensor(value: np.ndarray, device: torch.device) -> Tensor:
    return torch.from_numpy(value).to(device=device)


class RecurrentPointerPolicy:
    """Greedy recurrent policy whose hidden state is reset at episode boundaries."""

    def __init__(
        self,
        model: nn.Module,
        *,
        encoder: TeacherStateEncoder | None = None,
        pointer_spec: PointerActionSpec | None = None,
        artifact_sha256: str | None = None,
    ) -> None:
        self.model = model
        self.encoder = TeacherStateEncoder() if encoder is None else encoder
        self.pointer_spec = (
            getattr(model, "pointer_spec", PointerActionSpec())
            if pointer_spec is None
            else pointer_spec
        )
        schema = getattr(model, "schema", None)
        if schema is None or schema.sha256 != self.encoder.schema.sha256:
            raise ValueError("pointer model and inference encoder schema identities differ")
        model_spec = getattr(model, "pointer_spec", None)
        if (
            model_spec is not None
            and getattr(model_spec, "sha256", None) != self.pointer_spec.sha256
        ):
            raise ValueError("model and inference pointer action identities differ")
        if artifact_sha256 is not None and re.fullmatch(
            r"[0-9a-f]{64}", artifact_sha256
        ) is None:
            raise ValueError("policy artifact identity must be a lowercase SHA-256")
        self.artifact_sha256 = artifact_sha256
        self.schema_sha256 = self.encoder.schema.sha256
        self.pointer_action_sha256 = self.pointer_spec.sha256
        self._state: Tensor | None = None

    def reset(self, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("policy seed must fit in uint32")
        self._state = None

    @property
    def recurrent_state(self) -> Tensor | None:
        return None if self._state is None else self._state.detach().clone()

    @torch.no_grad()
    def predict(self, observation: Mapping[str, Any]) -> PointerPolicyDecision:
        encoded = self.encoder.encode([observation])
        device = _model_device(self.model)
        global_features = _tensor(encoded.global_features, device).unsqueeze(0)
        active = np.flatnonzero(encoded.body_mask[0])
        width = int(active[-1]) + 1 if active.size else 1
        body_features = _tensor(
            encoded.body_features[:, :width], device
        ).unsqueeze(0)
        body_mask = _tensor(encoded.body_mask[:, :width], device).unsqueeze(0)
        if self._state is None:
            self._state = self.model.initial_state(1, device=device)
        output = self.model(
            global_features,
            body_features,
            body_mask,
            self._state,
        )
        self._state = output.recurrent_state.detach()

        kind_probability = torch.softmax(output.kind_logits[0, 0], dim=-1)
        kind = int(kind_probability.argmax())
        kind_confidence = float(kind_probability[kind])
        wait_index = int(output.wait_logits[0, 0].argmax())
        target_index = 0
        template_index = 0
        if kind == 0:
            branch_probability = torch.softmax(output.wait_logits[0, 0], dim=-1)
            branch_confidence = float(branch_probability[wait_index])
            decision = PointerExpertDecision.wait(
                self.pointer_spec.wait_choices[wait_index]
            )
        else:
            branch = kind - 1
            target_probability = torch.softmax(
                output.target_logits[0, 0, branch], dim=-1
            )
            target_index = int(target_probability.argmax())
            template_probability = torch.softmax(
                output.template_logits[0, 0, branch, target_index], dim=-1
            )
            template_index = int(template_probability.argmax())
            branch_confidence = min(
                float(target_probability[target_index]),
                float(template_probability[template_index]),
            )
            body_ids = encoded_body_ids(encoded, observation)
            body_id = body_ids[target_index]
            if body_id is None:
                raise RuntimeError("model selected a masked target row")
            constructor = (
                PointerExpertDecision.weak
                if kind == 1
                else PointerExpertDecision.strong
            )
            decision = constructor(body_id, template_index=template_index)
        return PointerPolicyDecision(
            decision=decision,
            encoded=encoded,
            kind_index=kind,
            wait_index=wait_index,
            target_index=target_index,
            template_index=template_index,
            kind_confidence=kind_confidence,
            branch_confidence=branch_confidence,
            confidence=min(kind_confidence, branch_confidence),
        )

    def act(self, observation: Mapping[str, Any]) -> PointerExpertDecision:
        return self.predict(observation).decision


class RecurrentActorPointerPolicy:
    """Greedy actor-track policy that never reconstructs privileged body IDs."""

    def __init__(
        self,
        model: nn.Module,
        *,
        encoder: ActorTrackEncoder | None = None,
        pointer_spec: PointerActionSpec | None = None,
        action_spec: ActionSpec | None = None,
        artifact_sha256: str | None = None,
    ) -> None:
        self.model = model
        self.encoder = ActorTrackEncoder() if encoder is None else encoder
        self.pointer_spec = (
            getattr(model, "pointer_spec", PointerActionSpec())
            if pointer_spec is None
            else pointer_spec
        )
        self.action_spec = (
            getattr(model, "action_spec", ActionSpec())
            if action_spec is None
            else action_spec
        )
        schema = getattr(model, "schema", None)
        if (
            schema is None
            or schema.sha256 != ACTOR_VISION_V1.sha256
            or self.encoder.schema.sha256 != ACTOR_VISION_V1.sha256
        ):
            raise ValueError("actor policy requires the exact actor-vision-v1 schema")
        if "id_scaled" in schema.body_features:
            raise ValueError("actor policy schema cannot contain privileged body IDs")
        model_pointer = getattr(model, "pointer_spec", None)
        if (
            model_pointer is None
            or model_pointer.sha256 != self.pointer_spec.sha256
        ):
            raise ValueError("model and actor pointer action identities differ")
        model_action = getattr(model, "action_spec", None)
        if (
            model_action is None
            or model_action.sha256 != self.action_spec.sha256
        ):
            raise ValueError("model and actor primitive action identities differ")
        if artifact_sha256 is not None and re.fullmatch(
            r"[0-9a-f]{64}", artifact_sha256
        ) is None:
            raise ValueError("policy artifact identity must be a lowercase SHA-256")
        self.artifact_sha256 = artifact_sha256
        self.schema_sha256 = self.encoder.schema.sha256
        self.pointer_action_sha256 = self.pointer_spec.sha256
        self.action_schema_sha256 = self.action_spec.sha256
        self._state: Tensor | None = None

    def reset(self, seed: int = 0) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("policy seed must fit in uint32")
        self._state = None

    @torch.no_grad()
    def predict(self, actor_record: Mapping[str, Any]) -> ActorPointerPolicyDecision:
        encoded = self.encoder.encode([actor_record])
        if encoded.schema.sha256 != self.schema_sha256:
            raise RuntimeError("actor encoder returned an unexpected schema identity")
        device = _model_device(self.model)
        global_features = _tensor(encoded.global_features, device).unsqueeze(0)
        active = np.flatnonzero(encoded.body_mask[0])
        width = int(active[-1]) + 1 if active.size else 1
        body_features = _tensor(
            encoded.body_features[:, :width], device
        ).unsqueeze(0)
        body_mask = _tensor(encoded.body_mask[:, :width], device).unsqueeze(0)
        if self._state is None:
            self._state = self.model.initial_state(1, device=device)
        output = self.model(
            global_features,
            body_features,
            body_mask,
            self._state,
        )
        self._state = output.recurrent_state.detach()

        kind_probability = torch.softmax(output.kind_logits[0, 0], dim=-1)
        kind = int(kind_probability.argmax())
        kind_confidence = float(kind_probability[kind])
        wait_index = int(output.wait_logits[0, 0].argmax())
        target_index = 0
        template_index = 0
        selected_body_row: np.ndarray | None = None
        if kind == int(SemanticActionKind.WAIT):
            branch_probability = torch.softmax(output.wait_logits[0, 0], dim=-1)
            branch_confidence = float(branch_probability[wait_index])
        else:
            branch = kind - 1
            target_probability = torch.softmax(
                output.target_logits[0, 0, branch], dim=-1
            )
            target_index = int(target_probability.argmax())
            if not bool(encoded.body_mask[0, target_index]):
                raise RuntimeError("actor model selected a masked target row")
            template_probability = torch.softmax(
                output.template_logits[0, 0, branch, target_index], dim=-1
            )
            template_index = int(template_probability.argmax())
            branch_confidence = min(
                float(target_probability[target_index]),
                float(template_probability[template_index]),
            )
            selected_body_row = encoded.body_features[0, target_index]
        semantic = decode_pointer_action(
            kind=kind,
            wait_index=wait_index,
            template_index=template_index,
            selected_body_row=selected_body_row,
            schema=encoded.schema,
            pointer_spec=self.pointer_spec,
            action_spec=self.action_spec,
        )
        primitives = [self.action_spec.press(semantic)]
        if semantic.kind is not SemanticActionKind.WAIT:
            primitives.append(self.action_spec.release())
        return ActorPointerPolicyDecision(
            semantic_action=semantic,
            primitive_actions=tuple(primitives),
            encoded=encoded,
            kind_index=kind,
            wait_index=wait_index,
            target_index=target_index,
            template_index=template_index,
            kind_confidence=kind_confidence,
            branch_confidence=branch_confidence,
            confidence=min(kind_confidence, branch_confidence),
        )

    def act(self, actor_record: Mapping[str, Any]) -> tuple[Action, ...]:
        return self.predict(actor_record).primitive_actions


__all__ = [
    "ActorPointerPolicyDecision",
    "PointerPolicyDecision",
    "RecurrentActorPointerPolicy",
    "RecurrentPointerPolicy",
    "encoded_body_ids",
    "target_index_for_decision",
]
