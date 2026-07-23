"""Neural-ready contracts and rollout plumbing for IriSu training."""

from .actions import (
    ActionSpec,
    ConditionalActionDistribution,
    SemanticAction,
    SemanticActionKind,
)
from .encoding import ActorTrackEncoder, EncodedBatch, TeacherStateEncoder
from .rollout_buffer import RolloutBuffer
from .runtime_identity import (
    ACCEPTED_EXACT_RUNTIME_2026_07_21,
    ExactRuntimeIdentity,
    SimulatorRuntimeAttestation,
    attest_simulator_runtime,
)
from .schema import ACTOR_VISION_V1, TEACHER_V1, TensorSchema
from .seeds import SEED_SPLITS_V1, SeedAllocator, SeedReservation
from .vector_adapter import (
    MacroTransition,
    MacroVectorAdapter,
    ObservationInput,
    OwnedEvent,
)

__all__ = [
    "ACTOR_VISION_V1",
    "ACCEPTED_EXACT_RUNTIME_2026_07_21",
    "ActionSpec",
    "ActorTrackEncoder",
    "ConditionalActionDistribution",
    "EncodedBatch",
    "ExactRuntimeIdentity",
    "MacroTransition",
    "MacroVectorAdapter",
    "ObservationInput",
    "OwnedEvent",
    "RolloutBuffer",
    "SEED_SPLITS_V1",
    "SemanticAction",
    "SemanticActionKind",
    "SimulatorRuntimeAttestation",
    "SeedAllocator",
    "SeedReservation",
    "TEACHER_V1",
    "TeacherStateEncoder",
    "TensorSchema",
    "attest_simulator_runtime",
]
