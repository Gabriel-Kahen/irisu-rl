"""Streaming lowering for the recurrent sequence-replay residual."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch

from irisu_rl.encoding import TeacherStateEncoder

from .geometry_search import enumerate_geometry_candidates
from .policy import encoded_body_ids
from .sequence_replay import (
    SEQUENCE_REPLAY_EVENT_FEATURES,
    SequenceReplayModel,
    SequenceReplayStream,
)
from .sequence_replay_legacy import LegacyR3eGeometryCheckpoint
from .steering import SteeringDecision, SteeringIntent
from .steering_learning import GoalConditionedSteeringPolicy


_INTENTS = tuple(SteeringIntent)
_EVENT_INDEX = {
    name: index for index, name in enumerate(SEQUENCE_REPLAY_EVENT_FEATURES)
}


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _body_map(observation: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {
        int(value["id"]): value
        for value in observation.get("bodies", ())
        if isinstance(value, Mapping) and isinstance(value.get("id"), int)
    }


def _position(body: Mapping[str, Any]) -> tuple[float, float, float]:
    return (
        float(body.get("effect_x", body.get("x", 0.0))),
        float(body.get("effect_y", body.get("y", 0.0))),
        max(float(body.get("size", 0.0)), 1e-6),
    )


def _distance_sizes(
    bodies: Mapping[int, Mapping[str, Any]],
    source: int | None,
    destination: int | None,
) -> float | None:
    if source not in bodies or destination not in bodies:
        return None
    sx, sy, ss = _position(bodies[int(source)])
    dx, dy, ds = _position(bodies[int(destination)])
    return math.hypot(dx - sx, dy - sy) / max((ss + ds) / 2.0, 1e-6)


class CausalEventHistory:
    """Build an ID-free token from the previous completed transition."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.previous: dict[str, Any] | None = None
        self.previous_decision: SteeringDecision | None = None
        self.last_shot_tick: int | None = None

    def _snapshot(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "tick": int(observation.get("tick", 0)),
            "score": int(observation.get("score", 0)),
            "gauge": int(observation.get("gauge", 0)),
            "gauge_max": int(observation.get("gauge_max", 0)),
            "highest_chain": int(observation.get("highest_chain", 0)),
            "qualifying_clear_count": int(
                observation.get("qualifying_clear_count", 0)
            ),
            "terminated": bool(observation.get("terminated", False)),
            "truncated": bool(observation.get("truncated", False)),
            "spawn_interval": int(
                observation.get("difficulty", {}).get(
                    "spawn_interval_ticks", 1
                )
            ),
            "bodies": {
                identifier: dict(body)
                for identifier, body in _body_map(observation).items()
            },
        }

    def features(self, observation: Mapping[str, Any]) -> torch.Tensor:
        values = torch.zeros(len(SEQUENCE_REPLAY_EVENT_FEATURES))
        previous = self.previous
        decision = self.previous_decision
        if previous is None or decision is None:
            return values
        current = self._snapshot(observation)
        tick = current["tick"]
        elapsed = max(0, tick - previous["tick"])
        values[_EVENT_INDEX["previous_action_wait"]] = float(
            not decision.is_shot
        )
        values[_EVENT_INDEX["previous_action_shot"]] = float(decision.is_shot)
        values[_EVENT_INDEX[f"previous_intent_{decision.intent.value}"]] = 1.0
        values[_EVENT_INDEX["elapsed_ticks_log_scaled"]] = (
            math.log1p(elapsed) / 4.0
        )
        since_shot = (
            0 if self.last_shot_tick is None else max(0, tick - self.last_shot_tick)
        )
        values[_EVENT_INDEX["time_since_shot_log_scaled"]] = (
            math.log1p(since_shot) / 8.0
        )
        score_delta = current["score"] - previous["score"]
        values[_EVENT_INDEX["delta_score_scaled"]] = (
            math.copysign(math.log1p(abs(score_delta)), score_delta) / 8.0
        )
        values[_EVENT_INDEX["delta_gauge"]] = (
            current["gauge"] - previous["gauge"]
        ) / max(current["gauge_max"], 1)
        clear_delta = max(
            0,
            current["qualifying_clear_count"]
            - previous["qualifying_clear_count"],
        )
        values[_EVENT_INDEX["delta_clears_scaled"]] = math.log1p(clear_delta) / 4.0
        chain_delta = current["highest_chain"] - previous["highest_chain"]
        values[_EVENT_INDEX["delta_highest_chain_scaled"]] = (
            math.copysign(math.log1p(abs(chain_delta)), chain_delta) / 4.0
        )
        values[_EVENT_INDEX["shot_fired_count_log_scaled"]] = float(
            decision.is_shot
        )
        previous_bodies = previous["bodies"]
        current_bodies = current["bodies"]
        source = decision.source_body_id
        destination = decision.destination_body_id
        source_present = source in current_bodies
        destination_present = destination in current_bodies
        values[_EVENT_INDEX["source_present"]] = float(source_present)
        values[_EVENT_INDEX["destination_present"]] = float(destination_present)
        hit_gain = 0
        if source_present and source in previous_bodies:
            hit_gain = max(
                0,
                int(current_bodies[source].get("projectile_hits", 0))
                - int(previous_bodies[source].get("projectile_hits", 0)),
            )
        values[_EVENT_INDEX["projectile_hit_count_log_scaled"]] = (
            math.log1p(hit_gain) / 4.0
        )
        values[_EVENT_INDEX["intended_source_hit_count_log_scaled"]] = (
            math.log1p(hit_gain) / 4.0
        )
        joined = False
        if source_present and destination_present:
            source_chain = int(current_bodies[source].get("chain_id", 0))
            destination_chain = int(
                current_bodies[destination].get("chain_id", 0)
            )
            joined = source_chain > 0 and source_chain == destination_chain
        values[_EVENT_INDEX["pair_joined"]] = float(joined)
        values[_EVENT_INDEX["exact_pair_join_count_log_scaled"]] = float(joined)
        confirmed = joined and all(
            str(current_bodies[value].get("lifecycle", "")) == "confirmed"
            for value in (source, destination)
        )
        values[_EVENT_INDEX["chain_confirmed_count_log_scaled"]] = float(
            confirmed
        )
        pair_cleared = (
            decision.is_shot
            and not source_present
            and not destination_present
            and clear_delta > 0
        )
        values[_EVENT_INDEX["chain_cleared_count_log_scaled"]] = float(
            pair_cleared
        )
        before_distance = _distance_sizes(previous_bodies, source, destination)
        after_distance = _distance_sizes(current_bodies, source, destination)
        closure = (
            before_distance is not None
            and after_distance is not None
            and before_distance - after_distance >= 0.05
        )
        values[_EVENT_INDEX["closure_observed"]] = float(closure)
        values[_EVENT_INDEX["progress_failed"]] = float(
            decision.is_shot and elapsed >= 16 and not (closure or joined)
        )
        rotten_before = sum(
            str(body.get("lifecycle", "")) == "rotten"
            for body in previous_bodies.values()
        )
        rotten_after = sum(
            str(body.get("lifecycle", "")) == "rotten"
            for body in current_bodies.values()
        )
        values[_EVENT_INDEX["rotten_count_log_scaled"]] = (
            math.log1p(max(0, rotten_after - rotten_before)) / 4.0
        )
        vanished = max(0, len(previous_bodies) - len(current_bodies))
        values[_EVENT_INDEX["ejected_count_log_scaled"]] = (
            math.log1p(max(0, vanished - 2 * clear_delta)) / 4.0
        )
        values[_EVENT_INDEX["game_over"]] = float(
            current["terminated"] or current["truncated"]
        )
        interval = max(1, current["spawn_interval"])
        values[_EVENT_INDEX["spawn_boundary"]] = float(
            previous["tick"] // interval != tick // interval
        )
        return values

    def commit(
        self, observation: Mapping[str, Any], decision: SteeringDecision
    ) -> None:
        self.previous = self._snapshot(observation)
        self.previous_decision = decision
        if decision.is_shot:
            self.last_shot_tick = int(observation.get("tick", 0))


class SequenceReplayPolicy:
    """Conservative recurrent residual with frozen-v5 fallback."""

    def __init__(
        self,
        model: SequenceReplayModel,
        base_policy: GoalConditionedSteeringPolicy,
        geometry: LegacyR3eGeometryCheckpoint,
        *,
        checkpoint_sha256: str,
        source_identity: str,
        minimum_restraint_probability: float = 0.90,
        minimum_pair_probability: float = 0.80,
        minimum_geometry_probability: float = 0.55,
        minimum_geometry_margin: float = 0.05,
    ) -> None:
        if model.base_model is not base_policy.model:
            raise ValueError("sequence and lowering policy must share frozen v5")
        if (
            model.geometry_candidate_count != geometry.model.candidate_count
            or model.geometry_candidate_set_sha256
            != geometry.model.candidate_set_sha256
        ):
            raise ValueError("sequence and legacy geometry identities differ")
        thresholds = (
            minimum_restraint_probability,
            minimum_pair_probability,
            minimum_geometry_probability,
            minimum_geometry_margin,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in thresholds
        ):
            raise ValueError("sequence deployment thresholds must lie in [0,1]")
        self.model = model.eval()
        self.base_policy = base_policy
        self.geometry = geometry
        self.encoder = TeacherStateEncoder()
        self.stream = SequenceReplayStream(model)
        self.history = CausalEventHistory()
        self.checkpoint_sha256 = checkpoint_sha256
        self.source_identity = source_identity
        self.minimum_restraint_probability = float(
            minimum_restraint_probability
        )
        self.minimum_pair_probability = float(minimum_pair_probability)
        self.minimum_geometry_probability = float(
            minimum_geometry_probability
        )
        self.minimum_geometry_margin = float(minimum_geometry_margin)
        self._last_tick: int | None = None
        self._last_decision: SteeringDecision | None = None
        self._counts: Counter[str] = Counter()

    def identity_manifest(self) -> dict[str, object]:
        return {
            "version": "irisu-sequence-replay-policy-v1",
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_identity": self.source_identity,
            "base_checkpoint_sha256": self.model.base_checkpoint_sha256,
            "legacy_geometry_checkpoint_sha256": self.geometry.sha256,
            "model_architecture_sha256": self.model.architecture_sha256,
            "thresholds": {
                "minimum_restraint_probability": (
                    self.minimum_restraint_probability
                ),
                "minimum_pair_probability": self.minimum_pair_probability,
                "minimum_geometry_probability": (
                    self.minimum_geometry_probability
                ),
                "minimum_geometry_margin": self.minimum_geometry_margin,
            },
            "safety": [
                "never accelerate a frozen-v5 wait",
                "hard public legal-pair mask",
                "geometry slots masked by public availability",
                "frozen-v5 analytic fallback",
            ],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.identity_manifest())

    def reset(self, seed: int = 0) -> None:
        self.base_policy.reset(seed)
        self.stream.reset()
        self.history.reset()
        self._last_tick = None
        self._last_decision = None
        self._counts.clear()

    def _geometry_inputs(
        self,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
        encoded: Any,
        width: int,
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = enumerate_geometry_candidates(
            observation,
            incumbent,
            config=self.geometry.geometry_config,
            action_spec=self.base_policy.action_spec,
        )
        identifiers = encoded_body_ids(encoded, observation)

        def index(identifier: int | None) -> int:
            matches = [
                position
                for position, value in enumerate(identifiers)
                if value == identifier
            ]
            if len(matches) != 1:
                raise ValueError("sequence geometry pair failed row binding")
            return matches[0]

        source = index(incumbent.source_body_id)
        destination = index(incumbent.destination_body_id)
        device = next(self.model.parameters()).device
        logits = self.geometry.model(
            torch.from_numpy(encoded.global_features).to(device),
            torch.from_numpy(encoded.body_features[:, :width]).to(device),
            torch.from_numpy(encoded.body_mask[:, :width]).to(device),
            torch.tensor([source], dtype=torch.long, device=device),
            torch.tensor([destination], dtype=torch.long, device=device),
        )
        available = torch.tensor(
            [
                candidates.candidate_at(index) is not None
                for index in range(self.model.geometry_candidate_count)
            ],
            dtype=torch.bool,
            device=device,
        ).unsqueeze(0)
        return (
            candidates,
            logits,
            torch.tensor([source], dtype=torch.long, device=device),
            torch.tensor([destination], dtype=torch.long, device=device),
            available,
        )

    def _pair_override(
        self,
        observation: Mapping[str, Any],
        output: Any,
        encoded: Any,
        width: int,
        incumbent: SteeringDecision,
    ) -> SteeringDecision:
        legal = output.legal_pair_mask[0, 0]
        legal_flat = legal.flatten().nonzero(as_tuple=False).reshape(-1)
        if not legal_flat.numel():
            return incumbent
        logits = output.pair_logits[0, 0].flatten()[legal_flat]
        probabilities = torch.softmax(logits, dim=0)
        position = int(probabilities.argmax())
        if float(probabilities[position]) < self.minimum_pair_probability:
            self._counts["pair_confidence_fallbacks"] += 1
            return incumbent
        source_index, destination_index = divmod(
            int(legal_flat[position]), width
        )
        identifiers = encoded_body_ids(encoded, observation)
        source_id = identifiers[source_index]
        destination_id = identifiers[destination_index]
        if (
            source_id is None
            or destination_id is None
            or (
                source_id == incumbent.source_body_id
                and destination_id == incumbent.destination_body_id
            )
        ):
            return incumbent
        bodies = _body_map(observation)
        source = bodies.get(source_id)
        destination = bodies.get(destination_id)
        if source is None or destination is None:
            return incumbent
        analytic = self.base_policy._analytic_action(  # noqa: SLF001
            source, destination
        )
        if analytic is None or self.base_policy._progress.is_stalled(  # noqa: SLF001
            observation, source_id, destination_id
        ):
            return incumbent
        semantic, impact_x, impact_y = analytic
        intent_index = int(
            output.intent_logits[0, 0, source_index, destination_index].argmax()
        )
        decision = SteeringDecision(
            semantic,
            _INTENTS[intent_index],
            source_body_id=source_id,
            destination_body_id=destination_id,
            destination_chain_id=int(destination.get("chain_id", 0)),
            impact_x_sizes=impact_x,
            impact_y_sizes=impact_y,
            reason="high-confidence recurrent directed-pair residual",
        )
        self._counts["pair_overrides"] += 1
        return decision

    def _commit_base_state(
        self,
        observation: Mapping[str, Any],
        decision: SteeringDecision,
        *,
        tick: int,
        safe_boundary: bool,
        previous_cooldown: int,
        settled_progress: Any,
    ) -> None:
        """Commit only the action actually returned, not the frozen proposal."""

        self.base_policy._progress = settled_progress  # noqa: SLF001
        if safe_boundary:
            self.base_policy._cooldown_until = tick  # noqa: SLF001
            if decision.is_shot:
                if (
                    decision.source_body_id is None
                    or decision.destination_body_id is None
                ):
                    raise RuntimeError("sequence shot lacks a directed pair")
                settled_progress.begin(
                    observation,
                    decision.source_body_id,
                    decision.destination_body_id,
                )
                self.base_policy._cooldown_until = (  # noqa: SLF001
                    tick + self.base_policy.cooldown_ticks
                )
        else:
            if decision.is_shot:
                raise RuntimeError("sequence policy accelerated a frozen wait")
            self.base_policy._cooldown_until = previous_cooldown  # noqa: SLF001
        self.base_policy._last_tick = tick  # noqa: SLF001
        self.base_policy._last_decision = decision  # noqa: SLF001

    @torch.no_grad()
    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        tick = int(observation.get("tick", 0))
        if self._last_tick is not None:
            if tick < self._last_tick:
                raise RuntimeError("sequence observation tick moved backwards")
            if tick == self._last_tick:
                assert self._last_decision is not None
                return self._last_decision
        event = self.history.features(observation)
        previous_cooldown = self.base_policy._cooldown_until  # noqa: SLF001
        safe_boundary = tick >= previous_cooldown
        if safe_boundary:
            self.base_policy._progress.prune(observation)  # noqa: SLF001
            self.base_policy._progress.assess(observation)  # noqa: SLF001
        settled_progress = deepcopy(self.base_policy._progress)  # noqa: SLF001
        incumbent = self.base_policy.predict(observation)
        encoded = self.encoder.encode([observation])
        active = np.flatnonzero(encoded.body_mask[0])
        width = int(active[-1]) + 1 if active.size else 1
        device = next(self.model.parameters()).device
        kwargs: dict[str, Any] = {}
        geometry_data: tuple[Any, ...] | None = None
        if (
            incumbent.is_shot
            and incumbent.source_body_id is not None
            and incumbent.destination_body_id is not None
        ):
            try:
                geometry_data = self._geometry_inputs(
                    observation, incumbent, encoded, width
                )
            except (TypeError, ValueError):
                self._counts["unsupported_geometry_pairs"] += 1
            if geometry_data is not None:
                candidates, logits, source, destination, available = geometry_data
                kwargs = {
                    "base_geometry_logits": logits,
                    "geometry_source_index": source,
                    "geometry_destination_index": destination,
                    "geometry_pair_mask": torch.ones(
                        1, dtype=torch.bool, device=device
                    ),
                    "geometry_candidate_mask": available,
                }
                self._counts["geometry_queries"] += 1
        output = self.stream.step(
            torch.from_numpy(encoded.global_features).to(device),
            torch.from_numpy(encoded.body_features[:, :width]).to(device),
            torch.from_numpy(encoded.body_mask[:, :width]).to(device),
            event.to(device).unsqueeze(0),
            **kwargs,
        )
        decision = incumbent
        finite = all(
            bool(torch.isfinite(value).all())
            for value in (
                output.act_logits,
                output.wait_logits,
                output.pair_logits,
                output.intent_logits,
            )
        ) and all(
            value is None or bool(torch.isfinite(value).all())
            for value in (
                output.geometry_logits,
                output.geometry_gate_logit,
            )
        )
        if not finite:
            self.stream.reset()
            self._counts["nonfinite_output_fallbacks"] += 1
        else:
            act_logits = output.act_logits[0, 0].clone()
            act_logits[1] += self.base_policy.act_logit_bias
            act_probability = torch.softmax(act_logits, dim=0)
            if (
                incumbent.is_shot
                and int(act_probability.argmax()) == 0
                and float(act_probability[0])
                >= self.minimum_restraint_probability
            ):
                wait_index = int(output.wait_logits[0, 0].argmax())
                decision = self.base_policy._wait(  # noqa: SLF001
                    self.model.pointer_spec.wait_choices[wait_index],
                    "high-confidence recurrent restraint",
                )
                self._counts["learned_restraints"] += 1
            elif incumbent.is_shot:
                decision = self._pair_override(
                    observation, output, encoded, width, incumbent
                )
                if (
                    decision.source_body_id == incumbent.source_body_id
                    and decision.destination_body_id
                    == incumbent.destination_body_id
                    and geometry_data is not None
                    and output.geometry_logits is not None
                    and output.geometry_apply_mask is not None
                    and bool(output.geometry_apply_mask[0, 0])
                ):
                    candidates = geometry_data[0]
                    logits = output.geometry_logits[0, 0]
                    available = geometry_data[4][0]
                    probabilities = torch.softmax(
                        logits.masked_fill(~available, -torch.inf), dim=0
                    )
                    slot = int(probabilities.argmax())
                    confidence = float(probabilities[slot])
                    margin = confidence - float(probabilities[0])
                    candidate = candidates.candidate_at(slot)
                    if (
                        slot != 0
                        and candidate is not None
                        and confidence >= self.minimum_geometry_probability
                        and margin >= self.minimum_geometry_margin
                    ):
                        decision = candidate.decision
                        self._counts["geometry_corrections"] += 1
                        self._counts[f"geometry_slot_{slot}"] += 1
                    elif slot != 0:
                        self._counts["geometry_confidence_fallbacks"] += 1
                elif geometry_data is not None:
                    self._counts["geometry_gate_fallbacks"] += 1
            else:
                self._counts["base_restraints"] += 1
        self._commit_base_state(
            observation,
            decision,
            tick=tick,
            safe_boundary=safe_boundary,
            previous_cooldown=previous_cooldown,
            settled_progress=settled_progress,
        )
        self.history.commit(observation, decision)
        self._last_tick = tick
        self._last_decision = decision
        self._counts["decisions"] += 1
        self._counts[f"intent_{decision.intent.value}"] += 1
        return decision

    def act(self, observation: Mapping[str, Any]) -> tuple[Any, ...]:
        return self.predict(observation).primitive_actions(
            self.base_policy.action_spec
        )

    def statistics(self) -> dict[str, int]:
        return dict(sorted(self._counts.items()))


__all__ = [
    "CausalEventHistory",
    "SequenceReplayPolicy",
]
