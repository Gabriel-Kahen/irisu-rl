"""Neural-ready contracts and rollout plumbing for IriSu training."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ACTOR_VISION_V1": ("schema", "ACTOR_VISION_V1"),
    "ACCEPTED_EXACT_RUNTIME_2026_07_21": (
        "runtime_identity",
        "ACCEPTED_EXACT_RUNTIME_2026_07_21",
    ),
    "ActionSpec": ("actions", "ActionSpec"),
    "ActorTrackEncoder": ("encoding", "ActorTrackEncoder"),
    "ConditionalActionDistribution": ("actions", "ConditionalActionDistribution"),
    "EncodedBatch": ("encoding", "EncodedBatch"),
    "ExactRuntimeIdentity": ("runtime_identity", "ExactRuntimeIdentity"),
    "MacroTransition": ("vector_adapter", "MacroTransition"),
    "MacroVectorAdapter": ("vector_adapter", "MacroVectorAdapter"),
    "ObservationInput": ("vector_adapter", "ObservationInput"),
    "OwnedEvent": ("vector_adapter", "OwnedEvent"),
    "RolloutBuffer": ("rollout_buffer", "RolloutBuffer"),
    "SEED_SPLITS_V1": ("seeds", "SEED_SPLITS_V1"),
    "SemanticAction": ("actions", "SemanticAction"),
    "SemanticActionKind": ("actions", "SemanticActionKind"),
    "SimulatorRuntimeAttestation": (
        "runtime_identity",
        "SimulatorRuntimeAttestation",
    ),
    "SeedAllocator": ("seeds", "SeedAllocator"),
    "SeedReservation": ("seeds", "SeedReservation"),
    "TEACHER_V1": ("schema", "TEACHER_V1"),
    "TeacherStateEncoder": ("encoding", "TeacherStateEncoder"),
    "TensorSchema": ("schema", "TensorSchema"),
    "attest_simulator_runtime": ("runtime_identity", "attest_simulator_runtime"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value
