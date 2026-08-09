"""Fail-closed compatibility for the frozen legacy R3e geometry students."""

from __future__ import annotations

import hashlib
import json
import stat
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from irisu_rl.actions import ActionSpec
from irisu_rl.encoding import TeacherStateEncoder
from irisu_rl.schema import TEACHER_V1

from .geometry_learning import GeometryModelConfig, GeometrySelectorModel
from .geometry_search import (
    GeometryCandidateSet,
    GeometrySearchConfig,
    enumerate_geometry_candidates,
    geometry_candidate_slots,
)
from .policy import encoded_body_ids
from .steering import SteeringDecision
from .steering_learning import GoalConditionedSteeringPolicy

LEGACY_R3E_BASE_POLICY_SHA256 = (
    "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d"
)
LEGACY_R3E_BASE_POLICY_STATE_DICT_SHA256 = (
    "17c26b2beda17e85f5dab1b3a92dad5fccbf8210433dc98fd0b38641783b453a"
)
LEGACY_R3E_POINTER_ACTION_SHA256 = (
    "2daa7c0817ddffed1a8dbebc1b04b9ad5ce4ea6bd4429f0204bc33e76d646a6c"
)
LEGACY_R3E_RUNTIME_SHA256 = (
    "4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79"
)
LEGACY_R3E_SOURCE_IDENTITY_SHA256 = (
    "91101b048d5fb4b6c44b78fe1ed01165fefff90e81808cd446c5e8a95b0b8c67"
)
LEGACY_R3E_TEACHER_SHA256 = (
    "c4a4f109e04b6c16978769f099519191e68fa88c5df339a3dc8f3df7931ace8a"
)
LEGACY_R3E_CANDIDATE_VOCABULARY_SHA256 = (
    "e6cd65d87e6ca4fa5a2bcb89cd9e61186f29eac265af25e867eaf69f0346d7a4"
)
LEGACY_R3E_COLLECTION_SHA256 = (
    "906eb8c468d87efe4c26b1d139a4b76c99e86b6c643f5b1ca81e188a74dac08c"
)
LEGACY_R3E_DATASET_SHA256 = (
    "d1147b91d681606c1077afd4e37af4f60ba263622cb7cd31fe6c23a3cb350805"
)
LEGACY_R3E_MINIMUM_CONFIDENCE = 0.55
LEGACY_R3E_MINIMUM_PROBABILITY_MARGIN = 0.05

_FORMAT = "irisu-r3e-geometry-selector-checkpoint-v1"
_VOCABULARY_VERSION = "r3d-fixed-slot-geometry-vocabulary-v2"
_SEARCH_VERSION = "r3d-directed-pair-geometry-search-v2"
_TEACHER_VERSION = "r3e-runway-geometry-teacher-v2"
_SELECTION_RULE = (
    "valid branches survival-nondominated by incumbent; "
    "lexicographic public objective; incumbent wins exact ties"
)
_TRAINING_SELECTION_RULE = (
    "minimum_gauge_failures",
    "maximum_p10_survival_ticks",
    "maximum_median_survival_ticks",
    "maximum_median_raw_score",
    "minimum_training_steps",
)
_KNOWN_CHECKPOINTS = {
    "5db3b5cc3fe7583d98e294561d3928677ee708bc62327b67bfb9e7da46eaaefe": (
        30,
        "f23bfc6c82f183c6f7f350e5f4a85c697befe58c6de453ba42c8459365a2cce5",
    ),
    "6752f1c7be5a05e75d7bd85f0464e58c327acfc5b0155461e851ac13451c931a": (
        960,
        "7dc96cfd576fc622b57bdfaed9ed9658bb56a00baaeeca3b03ac11d464bf0826",
    ),
}


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _json_mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    try:
        return json.loads(_canonical_bytes(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be canonical JSON") from exc


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str


def _snapshot(path: str | Path, expected_sha256: str) -> _FileSnapshot:
    expected = _sha256(expected_sha256, "legacy R3e checkpoint")
    supplied = Path(path).expanduser()
    if supplied.is_symlink():
        raise ValueError("legacy R3e checkpoint must not be a symbolic link")
    source = supplied.resolve(strict=True)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("legacy R3e checkpoint must be a regular file")
    digest = _file_sha256(source)
    after = source.stat()
    identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise RuntimeError("legacy R3e checkpoint changed while hashing")
    if digest != expected:
        raise ValueError("legacy R3e checkpoint SHA-256 mismatch")
    return _FileSnapshot(source, *map(int, identity), digest)


def _require_unchanged(snapshot: _FileSnapshot) -> None:
    current = _snapshot(snapshot.path, snapshot.sha256)
    if current != snapshot:
        raise RuntimeError("legacy R3e checkpoint changed while loading")


def _state_dict_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode())
        digest.update(_canonical_bytes(list(tensor.shape)))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _current_geometry_config(legacy: Mapping[str, Any]) -> GeometrySearchConfig:
    required = {
        "version",
        "horizon_ticks",
        "velocity_lead_ticks",
        "ticks_per_second",
        "support_fractions",
        "support_clearance_sizes",
        "grid_x_fractions",
        "grid_y_sizes",
        "max_candidates",
        "candidate_slots",
        "causal_horizon",
        "selection_rule",
    }
    if set(legacy) != required:
        raise ValueError("legacy R3e search config fields differ")
    if (
        legacy["version"] != _SEARCH_VERSION
        or legacy["causal_horizon"] != "min(configured ticks, ticks before next spawn)"
        or legacy["selection_rule"] != _SELECTION_RULE
    ):
        raise ValueError("legacy R3e search contract differs")
    try:
        config = GeometrySearchConfig(
            horizon_ticks=legacy["horizon_ticks"],
            velocity_lead_ticks=legacy["velocity_lead_ticks"],
            ticks_per_second=legacy["ticks_per_second"],
            support_fractions=tuple(legacy["support_fractions"]),
            support_clearance_sizes=legacy["support_clearance_sizes"],
            grid_x_fractions=tuple(legacy["grid_x_fractions"]),
            grid_y_sizes=tuple(legacy["grid_y_sizes"]),
            max_candidates=legacy["max_candidates"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("legacy R3e search parameters are invalid") from exc
    old_slots = tuple(
        _json_mapping(value, "legacy R3e candidate slot")
        for value in legacy["candidate_slots"]
    )
    current_slots = geometry_candidate_slots(config)
    if old_slots != current_slots:
        raise ValueError(
            "legacy R3e candidate slots are not semantically compatible "
            "with the current enumerator"
        )
    legacy_vocabulary = {
        "version": _VOCABULARY_VERSION,
        "geometry_config_sha256": _canonical_sha256(dict(legacy)),
        "candidate_count": legacy["max_candidates"],
        "slots": list(old_slots),
    }
    if _canonical_sha256(legacy_vocabulary) != LEGACY_R3E_CANDIDATE_VOCABULARY_SHA256:
        raise ValueError("legacy R3e candidate vocabulary identity differs")
    return config


def _search_contract(value: object) -> tuple[dict[str, Any], GeometrySearchConfig]:
    search = _json_mapping(value, "legacy R3e search identity")
    if _canonical_sha256(search) != LEGACY_R3E_TEACHER_SHA256:
        raise ValueError("legacy R3e teacher identity differs")
    required = {
        "version",
        "config",
        "action_spec",
        "deployable",
        "canonical_evidence",
        "evidence_scope",
        "policy_inputs",
        "hidden_policy_inputs",
        "teacher_only_future",
    }
    if set(search) != required:
        raise ValueError("legacy R3e teacher fields differ")
    if (
        search["version"] != _TEACHER_VERSION
        or search["deployable"] is not False
        or search["canonical_evidence"] is not False
        or search["evidence_scope"] != "development-teacher-only"
        or search["hidden_policy_inputs"] != []
        or search["action_spec"] != ActionSpec().manifest()
    ):
        raise ValueError("legacy R3e teacher contract differs")
    teacher = _json_mapping(search["config"], "legacy R3e teacher config")
    if (
        teacher.get("version") != _TEACHER_VERSION
        or teacher.get("candidate_slot_count") != 32
        or teacher.get("deployable") is not False
        or teacher.get("canonical_evidence") is not False
        or teacher.get("evidence_scope") != "development-teacher-only"
    ):
        raise ValueError("legacy R3e runway teacher contract differs")
    config = _current_geometry_config(
        _json_mapping(
            teacher.get("candidate_config"),
            "legacy R3e candidate config",
        )
    )
    if teacher["candidate_slot_count"] != config.slot_count:
        raise ValueError("legacy R3e teacher slot count differs")
    return search, config


def _metadata(
    value: object,
    *,
    checkpoint_sha256: str,
    expected_base_policy_sha256: str,
    expected_runtime_sha256: str,
) -> dict[str, Any]:
    metadata = _json_mapping(value, "legacy R3e checkpoint metadata")
    required = {
        "base_policy_sha256",
        "candidate_vocabulary_sha256",
        "collection_sha256",
        "dataset_sha256",
        "runtime_sha256",
        "selected_training_steps",
        "selection_rule",
        "source_identity_sha256",
        "teacher_sha256",
        "development_only",
        "canonical_r3_evidence",
        "sealed_test_material_used",
    }
    if set(metadata) != required:
        raise ValueError("legacy R3e checkpoint metadata fields differ")
    selected_steps, _ = _KNOWN_CHECKPOINTS[checkpoint_sha256]
    expected = {
        "base_policy_sha256": expected_base_policy_sha256,
        "candidate_vocabulary_sha256": LEGACY_R3E_CANDIDATE_VOCABULARY_SHA256,
        "collection_sha256": LEGACY_R3E_COLLECTION_SHA256,
        "dataset_sha256": LEGACY_R3E_DATASET_SHA256,
        "runtime_sha256": expected_runtime_sha256,
        "selected_training_steps": selected_steps,
        "selection_rule": list(_TRAINING_SELECTION_RULE),
        "source_identity_sha256": LEGACY_R3E_SOURCE_IDENTITY_SHA256,
        "teacher_sha256": LEGACY_R3E_TEACHER_SHA256,
        "development_only": True,
        "canonical_r3_evidence": False,
        "sealed_test_material_used": False,
    }
    if metadata != expected:
        raise ValueError("legacy R3e checkpoint metadata identity differs")
    return metadata


@dataclass(frozen=True, slots=True)
class LegacyR3eGeometryCheckpoint:
    path: Path
    sha256: str
    state_dict_sha256: str
    model: GeometrySelectorModel
    geometry_config: GeometrySearchConfig
    search_identity: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def manifest(self) -> dict[str, object]:
        return {
            "format": _FORMAT,
            "path": str(self.path),
            "sha256": self.sha256,
            "state_dict_sha256": self.state_dict_sha256,
            "base_policy_sha256": self.metadata["base_policy_sha256"],
            "runtime_sha256": self.metadata["runtime_sha256"],
            "source_identity_sha256": self.metadata["source_identity_sha256"],
            "teacher_sha256": self.metadata["teacher_sha256"],
            "legacy_candidate_vocabulary_sha256": (
                self.metadata["candidate_vocabulary_sha256"]
            ),
            "current_semantic_slot_manifest_sha256": _canonical_sha256(
                list(geometry_candidate_slots(self.geometry_config))
            ),
            "selected_training_steps": self.metadata["selected_training_steps"],
        }


def load_legacy_r3e_geometry_checkpoint(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_base_policy_sha256: str = LEGACY_R3E_BASE_POLICY_SHA256,
    expected_runtime_sha256: str = LEGACY_R3E_RUNTIME_SHA256,
    device: torch.device | str = "cpu",
) -> LegacyR3eGeometryCheckpoint:
    """Load only the two known frozen R3e selector artifacts."""

    expected = _sha256(expected_sha256, "legacy R3e checkpoint")
    expected_base = _sha256(expected_base_policy_sha256, "legacy R3e base policy")
    expected_runtime = _sha256(expected_runtime_sha256, "legacy R3e portable runtime")
    if expected not in _KNOWN_CHECKPOINTS:
        raise ValueError("legacy R3e checkpoint is not a known frozen student")
    if expected_base != LEGACY_R3E_BASE_POLICY_SHA256:
        raise ValueError("legacy R3e base-policy identity is unsupported")
    if expected_runtime != LEGACY_R3E_RUNTIME_SHA256:
        raise ValueError("legacy R3e runtime identity is unsupported")
    snapshot = _snapshot(path, expected)
    payload = torch.load(snapshot.path, map_location="cpu", weights_only=True)
    required = {
        "format",
        "metadata",
        "model_manifest",
        "search_identity",
        "state_dict",
        "state_dict_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("legacy R3e checkpoint fields differ")
    if payload["format"] != _FORMAT:
        raise ValueError("legacy R3e checkpoint format differs")
    metadata = _metadata(
        payload["metadata"],
        checkpoint_sha256=expected,
        expected_base_policy_sha256=expected_base,
        expected_runtime_sha256=expected_runtime,
    )
    search, geometry_config = _search_contract(payload["search_identity"])
    if metadata["teacher_sha256"] != _canonical_sha256(search):
        raise ValueError("legacy R3e metadata and teacher identity differ")

    manifest = _json_mapping(payload["model_manifest"], "legacy R3e model manifest")
    try:
        model = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=manifest["candidate_count"],
            candidate_set_sha256=manifest["candidate_set_sha256"],
            config=GeometryModelConfig(**manifest["config"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("legacy R3e model configuration is invalid") from exc
    if (
        model.manifest() != manifest
        or model.candidate_count != geometry_config.slot_count
        or model.candidate_set_sha256 != LEGACY_R3E_CANDIDATE_VOCABULARY_SHA256
    ):
        raise ValueError("legacy R3e model identity differs")
    state_dict = payload["state_dict"]
    if not isinstance(state_dict, dict) or any(
        not isinstance(name, str) or not isinstance(value, torch.Tensor)
        for name, value in state_dict.items()
    ):
        raise ValueError("legacy R3e state dictionary is malformed")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise ValueError("legacy R3e model parameters differ") from exc
    state_sha256 = _sha256(payload["state_dict_sha256"], "legacy R3e state dictionary")
    _, known_state_sha256 = _KNOWN_CHECKPOINTS[expected]
    if state_sha256 != known_state_sha256 or _state_dict_sha256(model) != state_sha256:
        raise ValueError("legacy R3e state-dictionary identity differs")
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    _require_unchanged(snapshot)
    return LegacyR3eGeometryCheckpoint(
        snapshot.path,
        snapshot.sha256,
        state_sha256,
        model,
        geometry_config,
        search,
        metadata,
    )


def _piece_pair(observation: Mapping[str, Any], decision: SteeringDecision) -> bool:
    if (
        not decision.is_shot
        or decision.source_body_id is None
        or decision.destination_body_id is None
    ):
        return False
    target = {decision.source_body_id, decision.destination_body_id}
    selected = {
        int(body.get("id", -1)): body
        for body in observation.get("bodies", ())
        if isinstance(body, Mapping) and int(body.get("id", -1)) in target
    }
    return (
        set(selected) == target
        and all(body.get("kind") == "piece" for body in selected.values())
        and all(
            body.get("shape") in {"circle", "box", "triangle"}
            for body in selected.values()
        )
    )


@dataclass(frozen=True, slots=True)
class LegacyR3eGeometryProposal:
    """One incumbent-conditioned legacy selection with reusable model inputs."""

    incumbent: SteeringDecision
    decision: SteeringDecision
    candidates: GeometryCandidateSet | None
    base_logits: torch.Tensor | None
    available_mask: torch.Tensor | None
    source_index: int | None
    destination_index: int | None
    argmax_slot: int | None
    selected_slot: int
    confidence: float | None
    probability_margin_over_incumbent: float | None
    status: str

    @property
    def applied(self) -> bool:
        return self.selected_slot != 0

    def candidate_at(self, slot: int) -> SteeringDecision | None:
        if self.candidates is None:
            return None
        candidate = self.candidates.candidate_at(slot)
        return None if candidate is None else candidate.decision


class LegacyR3eGeometryPolicy:
    """Exact legacy confidence-gated selector over current equivalent slots."""

    def __init__(
        self,
        base_policy: GoalConditionedSteeringPolicy,
        checkpoint: LegacyR3eGeometryCheckpoint,
    ) -> None:
        if not isinstance(base_policy, GoalConditionedSteeringPolicy):
            raise TypeError("legacy R3e base policy must be goal-conditioned steering")
        if not isinstance(checkpoint, LegacyR3eGeometryCheckpoint):
            raise TypeError("legacy R3e checkpoint has the wrong type")
        if base_policy.artifact_sha256 != LEGACY_R3E_BASE_POLICY_SHA256:
            raise ValueError("legacy R3e base-policy checkpoint identity differs")
        if (
            _state_dict_sha256(base_policy.model)
            != LEGACY_R3E_BASE_POLICY_STATE_DICT_SHA256
            or base_policy.model.training
        ):
            raise ValueError("legacy R3e base-policy parameters differ")
        expected_options = (
            (base_policy.cooldown_ticks, 16),
            (base_policy.minimum_pair_closure_sizes, 0.05),
            (base_policy.impact_side_sizes, 0.5),
            (base_policy.impact_below_sizes, 0.75),
            (base_policy.source_velocity_lead_ticks, 1.0),
            (base_policy.ticks_per_second, 50.0),
            (base_policy.act_logit_bias, 1.0),
        )
        if any(actual != expected for actual, expected in expected_options):
            raise ValueError("legacy R3e base-policy inference options differ")
        if (
            base_policy.pointer_spec.sha256 != LEGACY_R3E_POINTER_ACTION_SHA256
            or base_policy.pointer_action_sha256 != LEGACY_R3E_POINTER_ACTION_SHA256
        ):
            raise ValueError("legacy R3e base-policy pointer vocabulary differs")
        if base_policy.encoder.schema.sha256 != checkpoint.model.schema.sha256:
            raise ValueError("legacy R3e base policy and selector schemas differ")
        if (
            checkpoint.model.training
            or any(
                parameter.requires_grad for parameter in checkpoint.model.parameters()
            )
            or _state_dict_sha256(checkpoint.model) != checkpoint.state_dict_sha256
        ):
            raise ValueError("legacy R3e selector parameters changed after loading")
        action_spec = ActionSpec()
        if (
            base_policy.action_spec.manifest() != action_spec.manifest()
            or checkpoint.search_identity["action_spec"] != action_spec.manifest()
        ):
            raise ValueError("legacy R3e semantic action identity differs")
        self.base_policy = base_policy
        self.checkpoint = checkpoint
        self.model = checkpoint.model
        self.geometry_config = checkpoint.geometry_config
        self.action_spec = base_policy.action_spec
        self.encoder = TeacherStateEncoder()
        self.artifact_sha256 = checkpoint.sha256
        self.schema_sha256 = checkpoint.model.schema.sha256
        self.pointer_action_sha256 = base_policy.pointer_action_sha256
        self.corrections = 0
        self.incumbent_selections = 0
        self.confidence_fallbacks = 0
        self.unsupported_pairs = 0

    def identity_manifest(self) -> dict[str, object]:
        return {
            "version": "irisu-sequence-replay-legacy-r3e-policy-v1",
            "selector_checkpoint_sha256": self.checkpoint.sha256,
            "selector_state_dict_sha256": self.checkpoint.state_dict_sha256,
            "base_policy_checkpoint_sha256": LEGACY_R3E_BASE_POLICY_SHA256,
            "base_policy_state_dict_sha256": (LEGACY_R3E_BASE_POLICY_STATE_DICT_SHA256),
            "runtime_sha256": LEGACY_R3E_RUNTIME_SHA256,
            "source_identity_sha256": LEGACY_R3E_SOURCE_IDENTITY_SHA256,
            "legacy_teacher_sha256": LEGACY_R3E_TEACHER_SHA256,
            "legacy_candidate_vocabulary_sha256": (
                LEGACY_R3E_CANDIDATE_VOCABULARY_SHA256
            ),
            "current_semantic_slot_manifest_sha256": _canonical_sha256(
                list(geometry_candidate_slots(self.geometry_config))
            ),
            "minimum_confidence": LEGACY_R3E_MINIMUM_CONFIDENCE,
            "minimum_probability_margin_over_incumbent": (
                LEGACY_R3E_MINIMUM_PROBABILITY_MARGIN
            ),
            "selection_rule": (
                "softmax over available legacy-equivalent slots; argmax; "
                "non-incumbent requires confidence >= 0.55 and probability "
                "margin over incumbent >= 0.05"
            ),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())

    def reset(self, seed: int = 0) -> None:
        self.base_policy.reset(seed)
        self.corrections = 0
        self.incumbent_selections = 0
        self.confidence_fallbacks = 0
        self.unsupported_pairs = 0

    @torch.no_grad()
    def proposal(
        self,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
    ) -> LegacyR3eGeometryProposal:
        """Score geometry for a supplied incumbent without touching the base policy."""

        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("legacy R3e incumbent must be a steering decision")
        if not _piece_pair(observation, incumbent):
            if incumbent.is_shot:
                self.unsupported_pairs += 1
            return LegacyR3eGeometryProposal(
                incumbent=incumbent,
                decision=incumbent,
                candidates=None,
                base_logits=None,
                available_mask=None,
                source_index=None,
                destination_index=None,
                argmax_slot=None,
                selected_slot=0,
                confidence=None,
                probability_margin_over_incumbent=None,
                status="unsupported_pair" if incumbent.is_shot else "not_shot",
            )
        candidates = enumerate_geometry_candidates(
            observation,
            incumbent,
            config=self.geometry_config,
            action_spec=self.action_spec,
        )
        encoded = self.encoder.encode([observation])
        identifiers = encoded_body_ids(encoded, observation)

        def bound_index(identifier: int | None) -> int:
            matches = [
                index for index, value in enumerate(identifiers) if value == identifier
            ]
            if len(matches) != 1:
                raise RuntimeError(
                    "legacy R3e pair did not bind to exactly one encoded row"
                )
            return matches[0]

        source = bound_index(incumbent.source_body_id)
        destination = bound_index(incumbent.destination_body_id)
        active = np.flatnonzero(encoded.body_mask[0])
        width = int(active[-1]) + 1 if active.size else 1
        device = next(self.model.parameters()).device
        logits = self.model(
            torch.from_numpy(encoded.global_features).to(device),
            torch.from_numpy(encoded.body_features[:, :width]).to(device),
            torch.from_numpy(encoded.body_mask[:, :width]).to(device),
            torch.tensor([source], dtype=torch.long, device=device),
            torch.tensor([destination], dtype=torch.long, device=device),
        )[0]
        available_mask = torch.tensor(
            candidates.availability_mask,
            dtype=torch.bool,
            device=device,
        )
        available = available_mask.nonzero(as_tuple=False).reshape(-1)
        probabilities = torch.softmax(logits[available], dim=-1)
        best_position = int(probabilities.argmax())
        best_slot = int(available[best_position])
        incumbent_position = int((available == 0).nonzero()[0, 0])
        confidence = float(probabilities[best_position])
        margin = confidence - float(probabilities[incumbent_position])
        if best_slot == 0:
            self.incumbent_selections += 1
            decision = incumbent
            selected_slot = 0
            status = "incumbent_argmax"
        else:
            selected = candidates.candidate_at(best_slot)
            if selected is None:
                raise RuntimeError("available legacy R3e candidate is missing")
            if (
                confidence < LEGACY_R3E_MINIMUM_CONFIDENCE
                or margin < LEGACY_R3E_MINIMUM_PROBABILITY_MARGIN
            ):
                self.confidence_fallbacks += 1
                decision = incumbent
                selected_slot = 0
                status = "confidence_fallback"
            else:
                self.corrections += 1
                decision = selected.decision
                selected_slot = best_slot
                status = "geometry_correction"
        return LegacyR3eGeometryProposal(
            incumbent=incumbent,
            decision=decision,
            candidates=candidates,
            base_logits=logits.detach().clone(),
            available_mask=available_mask.detach().clone(),
            source_index=source,
            destination_index=destination,
            argmax_slot=best_slot,
            selected_slot=selected_slot,
            confidence=confidence,
            probability_margin_over_incumbent=margin,
            status=status,
        )

    @torch.no_grad()
    def select_from_incumbent(
        self,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
    ) -> SteeringDecision:
        """Apply the exact legacy selector without calling the base policy."""

        return self.proposal(observation, incumbent).decision

    @torch.no_grad()
    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = self.base_policy.predict(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("legacy R3e base policy did not return steering")
        return self.select_from_incumbent(observation, incumbent)

    def act(self, observation: Mapping[str, Any]) -> tuple[Any, ...]:
        return self.predict(observation).primitive_actions(self.action_spec)

    def statistics(self) -> dict[str, int]:
        return {
            "geometry_corrections": self.corrections,
            "incumbent_selections": self.incumbent_selections,
            "confidence_fallbacks": self.confidence_fallbacks,
            "unsupported_pairs": self.unsupported_pairs,
        }


@dataclass(frozen=True, slots=True)
class LegacyR3eGeometryPolicyFactory:
    checkpoint: LegacyR3eGeometryCheckpoint
    base_policy_factory: Callable[[], GoalConditionedSteeringPolicy]

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, LegacyR3eGeometryCheckpoint):
            raise TypeError("legacy R3e factory checkpoint has the wrong type")
        if not callable(self.base_policy_factory):
            raise TypeError("legacy R3e base-policy factory must be callable")

    def __call__(self) -> LegacyR3eGeometryPolicy:
        return LegacyR3eGeometryPolicy(
            self.base_policy_factory(),
            self.checkpoint,
        )

    def identity_manifest(self) -> dict[str, object]:
        return {
            "version": "irisu-sequence-replay-legacy-r3e-factory-v1",
            "selector_checkpoint_sha256": self.checkpoint.sha256,
            "selector_state_dict_sha256": self.checkpoint.state_dict_sha256,
            "legacy_candidate_vocabulary_sha256": (
                LEGACY_R3E_CANDIDATE_VOCABULARY_SHA256
            ),
            "current_semantic_slot_manifest_sha256": _canonical_sha256(
                list(geometry_candidate_slots(self.checkpoint.geometry_config))
            ),
            "base_policy_checkpoint_sha256": LEGACY_R3E_BASE_POLICY_SHA256,
            "base_policy_state_dict_sha256": (LEGACY_R3E_BASE_POLICY_STATE_DICT_SHA256),
            "runtime_sha256": LEGACY_R3E_RUNTIME_SHA256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())


__all__ = [
    "LEGACY_R3E_BASE_POLICY_SHA256",
    "LEGACY_R3E_BASE_POLICY_STATE_DICT_SHA256",
    "LEGACY_R3E_CANDIDATE_VOCABULARY_SHA256",
    "LEGACY_R3E_COLLECTION_SHA256",
    "LEGACY_R3E_DATASET_SHA256",
    "LEGACY_R3E_MINIMUM_CONFIDENCE",
    "LEGACY_R3E_MINIMUM_PROBABILITY_MARGIN",
    "LEGACY_R3E_POINTER_ACTION_SHA256",
    "LEGACY_R3E_RUNTIME_SHA256",
    "LEGACY_R3E_SOURCE_IDENTITY_SHA256",
    "LEGACY_R3E_TEACHER_SHA256",
    "LegacyR3eGeometryCheckpoint",
    "LegacyR3eGeometryPolicy",
    "LegacyR3eGeometryPolicyFactory",
    "LegacyR3eGeometryProposal",
    "load_legacy_r3e_geometry_checkpoint",
]
