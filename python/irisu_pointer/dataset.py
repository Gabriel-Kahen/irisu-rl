"""Identity-bound in-memory datasets for entity-pointer supervision."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import Tensor

from irisu_rl.encoding import EncodedBatch
from irisu_rl.schema import TensorSchema

from .action import PointerActionSpec, PointerActionTensor


def _spec_sha256(spec: PointerActionSpec) -> str:
    value = getattr(spec, "sha256", None)
    if isinstance(value, str) and len(value) == 64:
        return value
    manifest = getattr(spec, "manifest", None)
    payload = manifest() if callable(manifest) else repr(spec)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _single_label(
    labels: PointerActionTensor, index: int
) -> PointerActionTensor:
    return PointerActionTensor(
        labels.kind.reshape(-1)[index : index + 1].detach().cpu().clone(),
        labels.wait_index.reshape(-1)[index : index + 1].detach().cpu().clone(),
        labels.target_index.reshape(-1)[index : index + 1].detach().cpu().clone(),
        labels.template_index.reshape(-1)[index : index + 1].detach().cpu().clone(),
    )


def _copy_observation(value: EncodedBatch) -> EncodedBatch:
    value.validate()
    return value.copy()


@dataclass(frozen=True, slots=True)
class PointerExample:
    """One independent pointer decision and its immutable supervision."""

    episode_identity: str
    observation: EncodedBatch
    label: PointerActionTensor
    value_target: float
    pointer_spec: PointerActionSpec = field(default_factory=PointerActionSpec)
    schema_sha256: str = field(init=False)
    pointer_spec_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.episode_identity, str) or not self.episode_identity:
            raise ValueError("episode identity must be a nonempty string")
        if "\x00" in self.episode_identity:
            raise ValueError("episode identity must not contain NUL")
        if (
            isinstance(self.value_target, bool)
            or not isinstance(self.value_target, (int, float))
            or not math.isfinite(float(self.value_target))
        ):
            raise ValueError("pointer value target must be finite")
        observation = _copy_observation(self.observation)
        if observation.global_features.shape[0] != 1:
            raise ValueError("one pointer example must contain exactly one observation")
        if not isinstance(self.label, PointerActionTensor):
            raise TypeError("pointer example label has the wrong type")
        label_fields = (
            self.label.kind,
            self.label.wait_index,
            self.label.target_index,
            self.label.template_index,
        )
        if any(value.numel() != 1 for value in label_fields):
            raise ValueError("one pointer example must contain exactly one label")
        label = _single_label(self.label, 0)
        label.validate(
            torch.Size((1,)),
            observation.schema.capacity,
            self.pointer_spec,
        )
        kind = int(label.kind.item())
        if kind != 0:
            target = int(label.target_index.item())
            if not bool(observation.body_mask[0, target]):
                raise ValueError("active pointer target refers to a masked body")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "value_target", float(self.value_target))
        object.__setattr__(self, "schema_sha256", observation.schema.sha256)
        object.__setattr__(self, "pointer_spec_sha256", _spec_sha256(self.pointer_spec))

    @classmethod
    def from_batch(
        cls,
        observations: EncodedBatch,
        labels: PointerActionTensor,
        index: int,
        *,
        episode_identity: str,
        value_target: float,
        pointer_spec: PointerActionSpec | None = None,
    ) -> PointerExample:
        observations.validate()
        batch = observations.global_features.shape[0]
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < batch:
            raise IndexError("pointer example row is outside the encoded batch")
        return cls(
            episode_identity,
            observations.row(index),
            _single_label(labels, index),
            value_target,
            pointer_spec or PointerActionSpec(),
        )

    from_encoded_batch = from_batch


@dataclass(frozen=True, slots=True)
class PointerTensorBatch:
    global_features: Tensor
    body_features: Tensor
    body_mask: Tensor
    kind: Tensor
    wait_index: Tensor
    target_index: Tensor
    template_index: Tensor
    value_target: Tensor

    @property
    def size(self) -> int:
        return int(self.kind.numel())


class PointerDataset(Sequence[PointerExample]):
    """A homogeneous, owned collection that cannot mix model identities."""

    def __init__(self, examples: Sequence[PointerExample]) -> None:
        supplied = tuple(examples)
        if not supplied:
            raise ValueError("pointer dataset must not be empty")
        if any(not isinstance(value, PointerExample) for value in supplied):
            raise TypeError("pointer dataset contains a non-pointer example")
        schema_sha256s = {value.schema_sha256 for value in supplied}
        pointer_sha256s = {value.pointer_spec_sha256 for value in supplied}
        if len(schema_sha256s) != 1 or len(pointer_sha256s) != 1:
            raise ValueError("pointer dataset mixes schema or action identities")
        self._examples = supplied
        self.schema: TensorSchema = supplied[0].observation.schema
        self.pointer_spec: PointerActionSpec = supplied[0].pointer_spec
        self.schema_sha256 = supplied[0].schema_sha256
        self.pointer_spec_sha256 = supplied[0].pointer_spec_sha256

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int | slice) -> PointerExample | tuple[PointerExample, ...]:
        return self._examples[index]

    def __iter__(self) -> Iterator[PointerExample]:
        return iter(self._examples)

    @property
    def episode_identities(self) -> tuple[str, ...]:
        return tuple(value.episode_identity for value in self._examples)

    @classmethod
    def from_encoded_batch(
        cls,
        observations: EncodedBatch,
        labels: PointerActionTensor,
        episode_identities: Sequence[str],
        value_targets: Sequence[float] | Tensor | np.ndarray,
        *,
        pointer_spec: PointerActionSpec | None = None,
    ) -> PointerDataset:
        observations.validate()
        batch = observations.global_features.shape[0]
        spec = pointer_spec or PointerActionSpec()
        labels.validate(torch.Size((batch,)), observations.schema.capacity, spec)
        if len(episode_identities) != batch:
            raise ValueError("episode identity count differs from encoded batch")
        values = torch.as_tensor(value_targets, dtype=torch.float32).reshape(-1)
        if values.shape != (batch,) or not bool(torch.isfinite(values).all()):
            raise ValueError("value targets must be one finite scalar per example")
        return cls(
            tuple(
                PointerExample.from_batch(
                    observations,
                    labels,
                    index,
                    episode_identity=episode_identities[index],
                    value_target=float(values[index]),
                    pointer_spec=spec,
                )
                for index in range(batch)
            )
        )

    from_batch = from_encoded_batch

    def split_by_episode(
        self,
        validation_fraction: float = 0.2,
        *,
        salt: str = "irisu-pointer-validation-v1",
    ) -> tuple[PointerDataset, PointerDataset]:
        if (
            isinstance(validation_fraction, bool)
            or not isinstance(validation_fraction, (int, float))
            or not 0 < float(validation_fraction) < 1
            or not isinstance(salt, str)
            or not salt
        ):
            raise ValueError("pointer split fraction or salt is invalid")
        identities = sorted(set(self.episode_identities))
        if len(identities) < 2:
            raise ValueError("episode split requires at least two distinct identities")
        validation_count = round(len(identities) * float(validation_fraction))
        validation_count = min(len(identities) - 1, max(1, validation_count))

        def key(identity: str) -> tuple[bytes, str]:
            digest = hashlib.sha256(
                salt.encode() + b"\0" + identity.encode()
            ).digest()
            return digest, identity

        validation_ids = set(sorted(identities, key=key)[:validation_count])
        training = tuple(
            value for value in self._examples
            if value.episode_identity not in validation_ids
        )
        validation = tuple(
            value for value in self._examples
            if value.episode_identity in validation_ids
        )
        if not training or not validation:
            raise RuntimeError("deterministic pointer split produced an empty partition")
        return PointerDataset(training), PointerDataset(validation)

    deterministic_split = split_by_episode

    def as_tensors(
        self, *, device: torch.device | str | None = None
    ) -> PointerTensorBatch:
        last_visible = max(
            (
                int(indices[-1]) + 1
                for value in self._examples
                for indices in [np.flatnonzero(value.observation.body_mask[0])]
                if indices.size
            ),
            default=1,
        )
        global_features = torch.from_numpy(
            np.concatenate(
                [value.observation.global_features for value in self._examples],
                axis=0,
            )
        )
        body_features = torch.from_numpy(
            np.concatenate(
                [
                    value.observation.body_features[:, :last_visible]
                    for value in self._examples
                ],
                axis=0,
            )
        )
        body_mask = torch.from_numpy(
            np.concatenate(
                [
                    value.observation.body_mask[:, :last_visible]
                    for value in self._examples
                ],
                axis=0,
            )
        )

        def labels(name: str) -> Tensor:
            return torch.cat(
                [getattr(value.label, name).reshape(1) for value in self._examples]
            ).to(torch.long)

        output = PointerTensorBatch(
            global_features,
            body_features,
            body_mask,
            labels("kind"),
            labels("wait_index"),
            labels("target_index"),
            labels("template_index"),
            torch.tensor(
                [value.value_target for value in self._examples],
                dtype=torch.float32,
            ),
        )
        if device is None:
            return output
        target = torch.device(device)
        return PointerTensorBatch(
            output.global_features.to(target),
            output.body_features.to(target),
            output.body_mask.to(target),
            output.kind.to(target),
            output.wait_index.to(target),
            output.target_index.to(target),
            output.template_index.to(target),
            output.value_target.to(target),
        )
