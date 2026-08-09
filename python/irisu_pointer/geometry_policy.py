"""Safeguarded deployment of learned directed-pair shot geometry."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .geometry_learning import GeometrySelectorModel
from .geometry_search import (
    GeometrySearchConfig,
    enumerate_geometry_candidates,
    geometry_candidate_slots,
)
from .policy import encoded_body_ids
from .steering import SteeringDecision
from .steering_learning import GoalConditionedSteeringPolicy

GEOMETRY_POLICY_VERSION = "r3d-safeguarded-geometry-policy-v2"
GEOMETRY_VOCABULARY_VERSION = "r3d-fixed-slot-geometry-vocabulary-v2"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _plain_int(value: Any, name: str) -> int:
    if isinstance(value, Integral) and not isinstance(value, bool):
        return int(value)
    item = getattr(value, "item", None)
    if callable(item):
        return _plain_int(item(), name)
    raise TypeError(f"{name} must be an integer")


def _plain_float(value: Any, name: str) -> float:
    if isinstance(value, Real) and not isinstance(value, bool):
        result = float(value)
    else:
        item = getattr(value, "item", None)
        if not callable(item):
            raise TypeError(f"{name} must be numeric")
        result = float(item())
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _supported_piece_pair(
    observation: Mapping[str, Any], decision: SteeringDecision
) -> bool:
    source = decision.source_body_id
    destination = decision.destination_body_id
    if not decision.is_shot or source is None or destination is None:
        return False
    values = observation.get("bodies")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return False
    selected: dict[int, Mapping[str, Any]] = {}
    for body in values:
        if not isinstance(body, Mapping):
            continue
        try:
            identifier = _plain_int(body.get("id"), "public body id")
        except TypeError:
            continue
        if identifier in {source, destination}:
            selected[identifier] = body
    return (
        set(selected) == {source, destination}
        and all(body.get("kind") == "piece" for body in selected.values())
        and all(
            body.get("shape") in {"circle", "box", "triangle"}
            for body in selected.values()
        )
    )


def geometry_candidate_vocabulary_manifest(
    config: GeometrySearchConfig | None = None,
) -> dict[str, object]:
    """Identity of fixed selector slots, including permanently unused slots."""

    resolved = GeometrySearchConfig() if config is None else config
    if not isinstance(resolved, GeometrySearchConfig):
        raise TypeError("geometry config must be a GeometrySearchConfig")
    slots = {
        int(value["slot"]): dict(value) for value in geometry_candidate_slots(resolved)
    }
    return {
        "version": GEOMETRY_VOCABULARY_VERSION,
        "geometry_config_sha256": resolved.sha256,
        "candidate_count": resolved.max_candidates,
        "slots": [
            slots.get(index, {"slot": index, "family": "unassigned", "name": None})
            for index in range(resolved.max_candidates)
        ],
    }


def geometry_candidate_vocabulary_sha256(
    config: GeometrySearchConfig | None = None,
) -> str:
    return _canonical_sha256(geometry_candidate_vocabulary_manifest(config))


@dataclass(frozen=True, slots=True)
class GeometryPolicyConfig:
    """Conservative gates for replacing the incumbent geometry."""

    minimum_confidence: float = 0.70
    minimum_logit_margin: float = 1.0
    minimum_ensemble_members: int = 1
    minimum_ensemble_agreement: float = 0.0
    minimum_member_incumbent_logit_margin: float | None = None
    minimum_gauge_fraction: float | None = None
    maximum_unverified_corrections: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_confidence, bool)
            or not isinstance(self.minimum_confidence, Real)
            or not math.isfinite(float(self.minimum_confidence))
            or not 0.0 <= float(self.minimum_confidence) <= 1.0
        ):
            raise ValueError("geometry confidence threshold must be in [0, 1]")
        if (
            isinstance(self.minimum_logit_margin, bool)
            or not isinstance(self.minimum_logit_margin, Real)
            or not math.isfinite(float(self.minimum_logit_margin))
            or float(self.minimum_logit_margin) < 0.0
        ):
            raise ValueError("geometry logit margin must be nonnegative")
        if (
            isinstance(self.minimum_ensemble_members, bool)
            or not isinstance(self.minimum_ensemble_members, int)
            or self.minimum_ensemble_members < 1
        ):
            raise ValueError("minimum ensemble members must be positive")
        if (
            isinstance(self.minimum_ensemble_agreement, bool)
            or not isinstance(self.minimum_ensemble_agreement, Real)
            or not math.isfinite(float(self.minimum_ensemble_agreement))
            or not 0.0 <= float(self.minimum_ensemble_agreement) <= 1.0
        ):
            raise ValueError("minimum ensemble agreement must be in [0, 1]")
        member_margin = self.minimum_member_incumbent_logit_margin
        if member_margin is not None and (
            isinstance(member_margin, bool)
            or not isinstance(member_margin, Real)
            or not math.isfinite(float(member_margin))
            or float(member_margin) < 0.0
        ):
            raise ValueError(
                "minimum member incumbent logit margin must be nonnegative"
            )
        gauge_fraction = self.minimum_gauge_fraction
        if gauge_fraction is not None and (
            isinstance(gauge_fraction, bool)
            or not isinstance(gauge_fraction, Real)
            or not math.isfinite(float(gauge_fraction))
            or not 0.0 <= float(gauge_fraction) <= 1.0
        ):
            raise ValueError("minimum gauge fraction must be in [0, 1]")
        maximum_unverified = self.maximum_unverified_corrections
        if maximum_unverified is not None and (
            isinstance(maximum_unverified, bool)
            or not isinstance(maximum_unverified, int)
            or maximum_unverified < 0
        ):
            raise ValueError(
                "maximum unverified learned corrections must be nonnegative"
            )

    def manifest(self) -> dict[str, object]:
        result: dict[str, object] = {
            "minimum_confidence": float(self.minimum_confidence),
            "minimum_logit_margin": float(self.minimum_logit_margin),
        }
        # Omit disabled ensemble gates so the default checkpoint/config
        # identity remains backward compatible.
        if self.minimum_ensemble_members != 1:
            result["minimum_ensemble_members"] = self.minimum_ensemble_members
        if self.minimum_ensemble_agreement != 0.0:
            result["minimum_ensemble_agreement"] = float(
                self.minimum_ensemble_agreement
            )
        if self.minimum_member_incumbent_logit_margin is not None:
            result["minimum_member_incumbent_logit_margin"] = float(
                self.minimum_member_incumbent_logit_margin
            )
        if self.minimum_gauge_fraction is not None:
            result["minimum_gauge_fraction"] = float(self.minimum_gauge_fraction)
        if self.maximum_unverified_corrections is not None:
            result["maximum_unverified_corrections"] = (
                self.maximum_unverified_corrections
            )
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


class GeometrySelectorEnsemble(nn.Module):
    """Identity-bound selectors whose agreement can gate deployment."""

    def __init__(
        self,
        selectors: Sequence[GeometrySelectorModel],
        *,
        artifact_sha256s: Sequence[str],
    ) -> None:
        super().__init__()
        values = tuple(selectors)
        artifacts = tuple(
            _sha256(value, "geometry ensemble member checkpoint")
            for value in artifact_sha256s
        )
        if len(values) < 2:
            raise ValueError("geometry ensemble requires at least two selectors")
        if len(artifacts) != len(values):
            raise ValueError("geometry ensemble selectors and identities differ")
        if len(set(artifacts)) != len(artifacts):
            raise ValueError("geometry ensemble checkpoint identities must be unique")
        if any(not isinstance(value, GeometrySelectorModel) for value in values):
            raise TypeError("geometry ensemble members must be GeometrySelectorModels")
        first = values[0]
        for value in values[1:]:
            if (
                value.schema.sha256 != first.schema.sha256
                or value.candidate_count != first.candidate_count
                or value.candidate_set_sha256 != first.candidate_set_sha256
            ):
                raise ValueError("geometry ensemble member contracts do not match")
        devices = {next(value.parameters()).device for value in values}
        if len(devices) != 1:
            raise ValueError("geometry ensemble members must share one device")
        self.members = nn.ModuleList(values)
        self.artifact_sha256s = artifacts
        self.schema = first.schema
        self.candidate_count = first.candidate_count
        self.candidate_set_sha256 = first.candidate_set_sha256

    @property
    def member_count(self) -> int:
        return len(self.members)

    def manifest(self) -> dict[str, object]:
        return {
            "format": "irisu-geometry-selector-ensemble-v1",
            "schema_sha256": self.schema.sha256,
            "candidate_set_sha256": self.candidate_set_sha256,
            "candidate_count": self.candidate_count,
            "aggregation": "arithmetic-mean-logits",
            "members": [
                {
                    "ordinal": ordinal,
                    "artifact_sha256": artifact,
                    "architecture_sha256": selector.architecture_sha256,
                }
                for ordinal, (selector, artifact) in enumerate(
                    zip(
                        self.members,
                        self.artifact_sha256s,
                        strict=True,
                    )
                )
            ],
        }

    @property
    def architecture_sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    @property
    def sha256(self) -> str:
        return self.architecture_sha256

    def member_logits(
        self,
        global_features: Tensor,
        body_features: Tensor,
        body_mask: Tensor,
        source_index: Tensor,
        destination_index: Tensor,
    ) -> Tensor:
        return torch.stack(
            [
                selector(
                    global_features,
                    body_features,
                    body_mask,
                    source_index,
                    destination_index,
                )
                for selector in self.members
            ],
            dim=0,
        )

    def forward(
        self,
        global_features: Tensor,
        body_features: Tensor,
        body_mask: Tensor,
        source_index: Tensor,
        destination_index: Tensor,
    ) -> Tensor:
        return self.member_logits(
            global_features,
            body_features,
            body_mask,
            source_index,
            destination_index,
        ).mean(dim=0)


@dataclass(frozen=True, slots=True)
class GeometryPolicySelection:
    tick: int
    candidate_set_sha256: str | None
    available_slots: tuple[int, ...]
    proposed_slot: int | None
    proposed_confidence: float | None
    incumbent_logit_margin: float | None
    deployed_slot: int | None
    used_learned_geometry: bool
    reason: str
    selector_member_count: int = 1
    ensemble_agreement: float | None = None
    minimum_member_incumbent_logit_margin: float | None = None
    gauge_fraction: float | None = None
    unverified_learned_corrections: int = 0
    progress_credit_replenished_by: tuple[str, ...] = ()

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["available_slots"] = list(self.available_slots)
        value["progress_credit_replenished_by"] = list(
            self.progress_credit_replenished_by
        )
        return value


class SafeguardedGeometryPolicy:
    """Wrap a frozen pair policy with confidence-gated geometry selection."""

    def __init__(
        self,
        base_policy: GoalConditionedSteeringPolicy,
        selector: GeometrySelectorModel | GeometrySelectorEnsemble,
        *,
        geometry_config: GeometrySearchConfig | None = None,
        policy_config: GeometryPolicyConfig | None = None,
        selector_artifact_sha256: str,
        source_identity: str,
    ) -> None:
        if not isinstance(base_policy, GoalConditionedSteeringPolicy):
            raise TypeError("base policy must be goal-conditioned steering")
        if not isinstance(selector, (GeometrySelectorModel, GeometrySelectorEnsemble)):
            raise TypeError("geometry selector must be a model or bound ensemble")
        resolved_geometry = (
            GeometrySearchConfig() if geometry_config is None else geometry_config
        )
        resolved_policy = (
            GeometryPolicyConfig() if policy_config is None else policy_config
        )
        if not isinstance(resolved_geometry, GeometrySearchConfig):
            raise TypeError("geometry config must be a GeometrySearchConfig")
        if not isinstance(resolved_policy, GeometryPolicyConfig):
            raise TypeError("policy config must be a GeometryPolicyConfig")
        base_sha256 = _sha256(base_policy.artifact_sha256, "base-policy checkpoint")
        vocabulary_sha256 = geometry_candidate_vocabulary_sha256(resolved_geometry)
        if selector.schema.sha256 != base_policy.encoder.schema.sha256:
            raise ValueError("base policy and geometry selector schemas differ")
        if selector.candidate_count != resolved_geometry.max_candidates:
            raise ValueError("selector width and fixed-slot vocabulary differ")
        if selector.candidate_set_sha256 != vocabulary_sha256:
            raise ValueError("selector and geometry vocabulary identities differ")

        self.base_policy = base_policy
        self.selector = selector
        self.geometry_config = resolved_geometry
        self.policy_config = resolved_policy
        selector_identity = _sha256(
            selector_artifact_sha256, "geometry-selector checkpoint"
        )
        if (
            isinstance(selector, GeometrySelectorEnsemble)
            and selector_identity != selector.sha256
        ):
            raise ValueError("geometry ensemble identity binding differs")
        self.selector_artifact_sha256 = selector_identity
        self.source_identity = _sha256(source_identity, "source identity")
        self.base_policy_checkpoint_sha256 = base_sha256
        self.schema_sha256 = selector.schema.sha256
        self.candidate_vocabulary_sha256 = vocabulary_sha256
        self.action_spec = base_policy.action_spec
        self.selector.eval()
        self.base_policy.model.eval()
        self.selector.requires_grad_(False)
        self.base_policy.model.requires_grad_(False)
        self._last_tick: int | None = None
        self._last_incumbent: SteeringDecision | None = None
        self._last_decision: SteeringDecision | None = None
        self._last_selection: GeometryPolicySelection | None = None
        self._last_score: int | None = None
        self._last_qualifying_clear_count: int | None = None
        self._unverified_learned_corrections = 0
        self._progress_credit_replenished_by: tuple[str, ...] = ()
        self._counts = Counter[str]()

    @property
    def last_selection(self) -> GeometryPolicySelection | None:
        return self._last_selection

    def identity_manifest(self) -> dict[str, object]:
        if isinstance(self.selector, GeometrySelectorEnsemble):
            selector_kind = "ensemble"
            member_artifacts = list(self.selector.artifact_sha256s)
        else:
            selector_kind = "single"
            member_artifacts = [self.selector_artifact_sha256]
        selection_rule = (
            "mean-logit argmax over available fixed slots; non-incumbent "
            "requires confidence, aggregate incumbent margin, configured "
            "ensemble agreement, and optional all-member incumbent margin; "
            "unsupported non-piece directed pairs retain incumbent geometry"
        )
        if self.policy_config.minimum_gauge_fraction is not None:
            selection_rule += (
                "; learned geometry requires the configured public gauge reserve"
            )
        if self.policy_config.maximum_unverified_corrections is not None:
            selection_rule += (
                "; learned correction credit is replenished only by a public "
                "score or qualifying-clear gain"
            )
        return {
            "version": GEOMETRY_POLICY_VERSION,
            "source_identity": self.source_identity,
            "schema_sha256": self.schema_sha256,
            "base_policy_checkpoint_sha256": self.base_policy_checkpoint_sha256,
            "selector_artifact_sha256": self.selector_artifact_sha256,
            "selector_architecture_sha256": self.selector.architecture_sha256,
            "selector_kind": selector_kind,
            "selector_member_count": len(member_artifacts),
            "selector_member_artifact_sha256s": member_artifacts,
            "geometry_config_sha256": self.geometry_config.sha256,
            "candidate_vocabulary_sha256": self.candidate_vocabulary_sha256,
            "action_spec_sha256": self.action_spec.sha256,
            "policy_config": self.policy_config.manifest(),
            "selection_rule": selection_rule,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())

    def reset(self, seed: int = 0) -> None:
        self.base_policy.reset(seed)
        self.reset_geometry_state()

    def reset_geometry_state(self) -> None:
        """Reset only geometry caches when another owner resets the base policy."""

        self._last_tick = None
        self._last_incumbent = None
        self._last_decision = None
        self._last_selection = None
        self._last_score = None
        self._last_qualifying_clear_count = None
        self._unverified_learned_corrections = 0
        self._progress_credit_replenished_by = ()
        self._counts.clear()

    def _observe_progress(self, observation: Mapping[str, Any]) -> None:
        self._progress_credit_replenished_by = ()
        if self.policy_config.maximum_unverified_corrections is None:
            return
        score = _plain_int(observation.get("score"), "public score")
        qualifying = _plain_int(
            observation.get("qualifying_clear_count"),
            "public qualifying clear count",
        )
        if score < 0 or qualifying < 0:
            raise ValueError("public progress counters must be nonnegative")
        if self._last_score is not None:
            assert self._last_qualifying_clear_count is not None
            if (
                score < self._last_score
                or qualifying < self._last_qualifying_clear_count
            ):
                raise RuntimeError("public progress counters moved backwards")
            sources: list[str] = []
            if score > self._last_score:
                sources.append("score")
                self._counts["score_progress_events"] += 1
            if qualifying > self._last_qualifying_clear_count:
                sources.append("qualifying_clear")
                self._counts["qualifying_clear_progress_events"] += 1
            if sources and self._unverified_learned_corrections:
                self._unverified_learned_corrections = 0
                self._progress_credit_replenished_by = tuple(sources)
                self._counts["progress_credit_replenishments"] += 1
        self._last_score = score
        self._last_qualifying_clear_count = qualifying

    def _gauge_fraction(self, observation: Mapping[str, Any]) -> float | None:
        if self.policy_config.minimum_gauge_fraction is None:
            return None
        try:
            gauge = _plain_float(observation.get("gauge"), "public gauge")
            gauge_max = _plain_float(
                observation.get("gauge_max"), "public maximum gauge"
            )
        except (TypeError, ValueError):
            return None
        if gauge < 0.0 or gauge_max <= 0.0 or gauge > gauge_max:
            return None
        return gauge / gauge_max

    def statistics(self) -> dict[str, int]:
        return {
            name: int(self._counts[name])
            for name in (
                "learned_geometry_deployments",
                "safeguard_fallbacks",
                "unsupported_pair_fallbacks",
                "gauge_reserve_rejections",
                "progress_credit_rejections",
                "progress_credit_replenishments",
                "score_progress_events",
                "qualifying_clear_progress_events",
            )
        } | {"unverified_learned_corrections": (self._unverified_learned_corrections)}

    def _indices(
        self, observation: Mapping[str, Any], decision: SteeringDecision
    ) -> tuple[Any, int, int, int]:
        encoded = self.base_policy.encoder.encode([observation])
        identifiers = encoded_body_ids(encoded, observation)

        def find(identifier: int | None, name: str) -> int:
            matches = [
                index for index, value in enumerate(identifiers) if value == identifier
            ]
            if len(matches) != 1:
                raise ValueError(f"geometry {name} did not bind exactly once")
            return matches[0]

        source_index = find(decision.source_body_id, "source")
        destination_index = find(decision.destination_body_id, "destination")
        active = np.flatnonzero(encoded.body_mask[0])
        width = int(active[-1]) + 1 if active.size else 1
        return encoded, source_index, destination_index, width

    def _fallback(
        self,
        tick: int,
        incumbent: SteeringDecision,
        reason: str,
        *,
        candidate_set_sha256: str | None = None,
        available_slots: tuple[int, ...] = (),
        proposed_slot: int | None = None,
        confidence: float | None = None,
        margin: float | None = None,
        ensemble_agreement: float | None = None,
        member_margin: float | None = None,
        gauge_fraction: float | None = None,
    ) -> SteeringDecision:
        member_count = (
            self.selector.member_count
            if isinstance(self.selector, GeometrySelectorEnsemble)
            else 1
        )
        self._last_selection = GeometryPolicySelection(
            tick,
            candidate_set_sha256,
            available_slots,
            proposed_slot,
            confidence,
            margin,
            0 if incumbent.is_shot else None,
            False,
            reason,
            member_count,
            ensemble_agreement,
            member_margin,
            gauge_fraction,
            self._unverified_learned_corrections,
            self._progress_credit_replenished_by,
        )
        return incumbent

    @torch.no_grad()
    def select_from_incumbent(
        self,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
    ) -> SteeringDecision:
        """Apply geometry once to a caller-owned incumbent without advancing base."""

        raw_tick = observation.get("tick", 0)
        if isinstance(raw_tick, bool) or not isinstance(raw_tick, Integral):
            raise TypeError("geometry observation tick must be an integer")
        tick = int(raw_tick)
        if tick < 0:
            raise ValueError("geometry observation tick must be nonnegative")
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("geometry incumbent must be a SteeringDecision")
        if self._last_tick is not None:
            if tick < self._last_tick:
                raise RuntimeError("geometry observation tick moved backwards")
            if tick == self._last_tick:
                if incumbent != self._last_incumbent:
                    raise RuntimeError("geometry incumbent changed within one tick")
                assert self._last_decision is not None
                return self._last_decision
        self._observe_progress(observation)
        if not incumbent.is_shot:
            decision = self._fallback(tick, incumbent, "base policy selected restraint")
        elif incumbent.destination_body_id is None:
            decision = self._fallback(
                tick, incumbent, "shot has no directed-pair destination"
            )
        elif not _supported_piece_pair(observation, incumbent):
            self._counts["unsupported_pair_fallbacks"] += 1
            decision = self._fallback(
                tick,
                incumbent,
                "base directed pair is unsupported by piece-only geometry",
            )
        else:
            candidate_set = enumerate_geometry_candidates(
                observation,
                incumbent,
                config=self.geometry_config,
                action_spec=self.action_spec,
            )
            by_slot = {
                candidate.ordinal: candidate for candidate in candidate_set.candidates
            }
            available_slots = tuple(sorted(by_slot))
            if 0 not in by_slot:
                raise RuntimeError("geometry candidate set lost its incumbent")
            if any(
                slot < 0 or slot >= self.selector.candidate_count
                for slot in available_slots
            ):
                raise RuntimeError("geometry candidate exceeds selector vocabulary")

            encoded, source_index, destination_index, width = self._indices(
                observation, incumbent
            )
            device = next(self.selector.parameters()).device
            arguments = (
                torch.from_numpy(encoded.global_features).to(device),
                torch.from_numpy(encoded.body_features[:, :width]).to(device),
                torch.from_numpy(encoded.body_mask[:, :width]).to(device),
                torch.tensor([source_index], dtype=torch.long, device=device),
                torch.tensor([destination_index], dtype=torch.long, device=device),
            )
            raw_logits = (
                self.selector.member_logits(*arguments)
                if isinstance(self.selector, GeometrySelectorEnsemble)
                else self.selector(*arguments).unsqueeze(0)
            )
            member_count = int(raw_logits.shape[0])
            if raw_logits.shape != (
                member_count,
                1,
                self.selector.candidate_count,
            ):
                raise RuntimeError("geometry selector returned an invalid shape")
            member_logits = raw_logits[:, 0]
            logits = member_logits.mean(dim=0)
            available = torch.zeros_like(logits, dtype=torch.bool)
            available[list(available_slots)] = True
            if not bool(torch.isfinite(member_logits[:, available]).all()):
                decision = self._fallback(
                    tick,
                    incumbent,
                    "selector member produced non-finite available logits",
                    candidate_set_sha256=candidate_set.sha256,
                    available_slots=available_slots,
                )
            else:
                masked = logits.masked_fill(~available, -torch.inf)
                member_masked = member_logits.masked_fill(
                    ~available.unsqueeze(0), -torch.inf
                )
                proposed = int(masked.argmax())
                confidence = float(torch.softmax(masked, dim=0)[proposed])
                margin = float(masked[proposed] - masked[0])
                votes = member_masked.argmax(dim=1)
                agreement = float((votes == proposed).float().mean())
                member_margin = float(
                    (member_masked[:, proposed] - member_masked[:, 0]).min()
                )
                required_member_margin = (
                    self.policy_config.minimum_member_incumbent_logit_margin
                )
                required_gauge = self.policy_config.minimum_gauge_fraction
                gauge_fraction = (
                    self._gauge_fraction(observation) if proposed != 0 else None
                )
                gauge_allowed = (
                    required_gauge is None
                    or gauge_fraction is not None
                    and gauge_fraction >= float(required_gauge)
                )
                maximum_unverified = self.policy_config.maximum_unverified_corrections
                credit_allowed = (
                    maximum_unverified is None
                    or self._unverified_learned_corrections < maximum_unverified
                )
                if proposed != 0 and not gauge_allowed:
                    self._counts["gauge_reserve_rejections"] += 1
                if proposed != 0 and not credit_allowed:
                    self._counts["progress_credit_rejections"] += 1
                guarded = (
                    proposed != 0
                    and confidence >= float(self.policy_config.minimum_confidence)
                    and margin >= float(self.policy_config.minimum_logit_margin)
                    and member_count >= self.policy_config.minimum_ensemble_members
                    and agreement
                    >= float(self.policy_config.minimum_ensemble_agreement)
                    and (
                        required_member_margin is None
                        or member_margin >= float(required_member_margin)
                    )
                    and gauge_allowed
                    and credit_allowed
                )
                if guarded:
                    decision = by_slot[proposed].decision
                    if maximum_unverified is not None:
                        self._unverified_learned_corrections += 1
                    self._counts["learned_geometry_deployments"] += 1
                    passed_reason = (
                        "learned geometry passed confidence, margin, "
                        "and ensemble safeguards"
                        if required_gauge is None and maximum_unverified is None
                        else (
                            "learned geometry passed confidence, margin, "
                            "ensemble, gauge-reserve, and progress-credit safeguards"
                        )
                    )
                    self._last_selection = GeometryPolicySelection(
                        tick,
                        candidate_set.sha256,
                        available_slots,
                        proposed,
                        confidence,
                        margin,
                        proposed,
                        True,
                        passed_reason,
                        member_count,
                        agreement,
                        member_margin,
                        gauge_fraction,
                        self._unverified_learned_corrections,
                        self._progress_credit_replenished_by,
                    )
                else:
                    if proposed == 0:
                        reason = "selector retained incumbent geometry"
                    elif confidence < float(self.policy_config.minimum_confidence):
                        reason = "selector confidence safeguard rejected geometry"
                    elif margin < float(self.policy_config.minimum_logit_margin):
                        reason = "selector margin safeguard rejected geometry"
                    elif member_count < self.policy_config.minimum_ensemble_members:
                        reason = "selector ensemble-size safeguard rejected geometry"
                    elif agreement < float(
                        self.policy_config.minimum_ensemble_agreement
                    ):
                        reason = (
                            "selector ensemble-agreement safeguard rejected geometry"
                        )
                    elif required_member_margin is not None and member_margin < float(
                        required_member_margin
                    ):
                        reason = "selector member-incumbent safeguard rejected geometry"
                    elif not gauge_allowed:
                        reason = (
                            "minimum gauge reserve safeguard rejected geometry"
                            if gauge_fraction is not None
                            else "public gauge reserve unavailable; safeguard rejected geometry"
                        )
                    else:
                        reason = (
                            "unverified learned-correction progress-credit "
                            "safeguard rejected geometry"
                        )
                    if proposed != 0:
                        self._counts["safeguard_fallbacks"] += 1
                    decision = self._fallback(
                        tick,
                        incumbent,
                        reason,
                        candidate_set_sha256=candidate_set.sha256,
                        available_slots=available_slots,
                        proposed_slot=proposed,
                        confidence=confidence,
                        margin=margin,
                        ensemble_agreement=agreement,
                        member_margin=member_margin,
                        gauge_fraction=gauge_fraction,
                    )

        self._last_tick = tick
        self._last_incumbent = incumbent
        self._last_decision = decision
        return decision

    @torch.no_grad()
    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        raw_tick = observation.get("tick", 0)
        if isinstance(raw_tick, bool) or not isinstance(raw_tick, Integral):
            raise TypeError("geometry observation tick must be an integer")
        tick = int(raw_tick)
        if tick < 0:
            raise ValueError("geometry observation tick must be nonnegative")
        if self._last_tick is not None:
            if tick < self._last_tick:
                raise RuntimeError("geometry observation tick moved backwards")
            if tick == self._last_tick:
                assert self._last_decision is not None
                return self._last_decision
        incumbent = self.base_policy.predict(observation)
        return self.select_from_incumbent(observation, incumbent)

    def act(self, observation: Mapping[str, Any]) -> tuple[Any, ...]:
        return self.predict(observation).primitive_actions(self.action_spec)


__all__ = [
    "GEOMETRY_POLICY_VERSION",
    "GEOMETRY_VOCABULARY_VERSION",
    "GeometryPolicyConfig",
    "GeometryPolicySelection",
    "GeometrySelectorEnsemble",
    "SafeguardedGeometryPolicy",
    "geometry_candidate_vocabulary_manifest",
    "geometry_candidate_vocabulary_sha256",
]
