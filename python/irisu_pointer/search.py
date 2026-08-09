"""Spawn-censored branch search over public IriSu observations and events."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from irisu_env import Action, ActionKind, EventKind

from .action import PointerActionSpec
from .experts import (
    PointerExpertDecision,
    SearchCandidate,
    diverse_template_indices,
    generate_candidates,
    target_body,
)


def ticks_before_next_spawn(observation: Mapping[str, Any]) -> int:
    """Return executable ticks before the next cadence-spawn tick.

    Spawning is tested at scene entry.  At tick 0 (and every exact multiple of
    the interval) the next primitive tick is therefore already a spawn tick.
    """

    tick = int(observation.get("tick", 0))
    difficulty = observation.get("difficulty", {})
    interval = int(difficulty.get("spawn_interval_ticks", 0))
    if tick < 0:
        raise ValueError("observation tick must be nonnegative")
    if interval <= 0:
        raise ValueError("spawn_interval_ticks must be positive")
    remainder = tick % interval
    return 0 if remainder == 0 else interval - remainder


@dataclass(frozen=True, slots=True)
class SearchUtilityWeights:
    """All utility terms are explicit so search evidence remains auditable."""

    score: float = 1.0
    gauge_fraction: float = 0.10
    chain_joined: float = 1.0
    cleared: float = 1.0
    projectile_hit: float = 0.10
    ejected: float = 0.20
    rotten: float = -1.0
    terminal: float = -4.0
    confirmed_body_destructive_hit: float = -2.0
    invalid_action: float = -4.0


@dataclass(frozen=True, slots=True)
class UtilityBreakdown:
    score_delta: int
    gauge_delta: int
    gauge_fraction_delta: float
    chain_joined: int
    cleared: int
    projectile_hit: int
    ejected: int
    rotten: int
    terminal: bool
    confirmed_body_destructive_hit: int
    invalid_action: int
    total: float


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    ordinal: int
    candidate: SearchCandidate
    utility: UtilityBreakdown
    final_observation: Mapping[str, Any]
    primitive_ticks: int
    continuation_value: float = 0.0
    search_score: float = 0.0


@dataclass(frozen=True, slots=True)
class SearchResult:
    decision: PointerExpertDecision
    selected_name: str
    evaluations: tuple[CandidateEvaluation, ...]
    safe_tick_budget: int
    evaluated_tick_budget: int


def _event_kind(event: Mapping[str, Any]) -> int | None:
    raw = event.get("kind")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    name = event.get("kind_name")
    if isinstance(name, str):
        try:
            return int(EventKind[name.upper()])
        except KeyError:
            return None
    return None


def utility_breakdown(
    initial_observation: Mapping[str, Any],
    final_observation: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    terminated: bool,
    truncated: bool,
    weights: SearchUtilityWeights | None = None,
) -> UtilityBreakdown:
    """Score one public branch outcome without hidden hashes or RNG state."""

    resolved = SearchUtilityWeights() if weights is None else weights
    counts = Counter(
        kind for event in events if (kind := _event_kind(event)) is not None
    )
    confirmed_ids = {
        int(body["id"])
        for body in initial_observation.get("bodies", ())
        if body.get("kind") == "piece" and body.get("lifecycle") == "confirmed"
    }
    hit_pairs = {
        (int(event.get("a", -1)), int(event.get("b", -1)))
        for event in events
        if _event_kind(event) == int(EventKind.PROJECTILE_HIT)
    }
    destructive_hits = len(
        {
            body_id
            for _, body_id in hit_pairs
            if body_id in confirmed_ids
            and any(
                _event_kind(event) == int(EventKind.PROJECTILE_HIT)
                and int(event.get("b", -1)) == body_id
                and int(event.get("value", 0)) >= 1
                for event in events
            )
        }
    )
    score_delta = int(final_observation.get("score", 0)) - int(
        initial_observation.get("score", 0)
    )
    gauge_delta = int(final_observation.get("gauge", 0)) - int(
        initial_observation.get("gauge", 0)
    )
    initial_gauge_max = max(int(initial_observation.get("gauge_max", 1)), 1)
    final_gauge_max = max(int(final_observation.get("gauge_max", 1)), 1)
    gauge_fraction_delta = (
        int(final_observation.get("gauge", 0)) / final_gauge_max
        - int(initial_observation.get("gauge", 0)) / initial_gauge_max
    )
    chain_joined = counts[int(EventKind.CHAIN_JOINED)]
    cleared = counts[int(EventKind.CLEARED)]
    # The clone reports sustained projectile contact every physics tick. Count
    # unique projectile/body pairs so a wedged projectile is not an artificial
    # source of teacher reward.
    projectile_hit = len(hit_pairs)
    ejected = counts[int(EventKind.EJECTED)]
    rotten = counts[int(EventKind.ROTTEN)]
    invalid_action = counts[int(EventKind.INVALID_ACTION)]
    is_terminal = bool(terminated or truncated)
    total = (
        resolved.score * score_delta
        + resolved.gauge_fraction * gauge_fraction_delta
        + resolved.chain_joined * chain_joined
        + resolved.cleared * cleared
        + resolved.projectile_hit * projectile_hit
        + resolved.ejected * ejected
        + resolved.rotten * rotten
        + resolved.terminal * int(is_terminal)
        + resolved.confirmed_body_destructive_hit * destructive_hits
        + resolved.invalid_action * invalid_action
    )
    return UtilityBreakdown(
        score_delta=score_delta,
        gauge_delta=gauge_delta,
        gauge_fraction_delta=gauge_fraction_delta,
        chain_joined=chain_joined,
        cleared=cleared,
        projectile_hit=projectile_hit,
        ejected=ejected,
        rotten=rotten,
        terminal=is_terminal,
        confirmed_body_destructive_hit=destructive_hits,
        invalid_action=invalid_action,
        total=float(total),
    )


def _template(
    spec: PointerActionSpec, template_index: int
) -> tuple[float, float]:
    if not 0 <= template_index < spec.template_count:
        raise ValueError("pointer template index is out of range")
    x_offset, y_offset = spec.templates[template_index]
    return float(x_offset), float(y_offset)


def lower_expert_decision(
    decision: PointerExpertDecision,
    observation: Mapping[str, Any],
    spec: PointerActionSpec,
    *,
    client_width: float = 640.0,
    client_height: float = 480.0,
) -> tuple[Action, ...]:
    """Lower an ID-targeted decision to deployment-v1 primitive actions."""

    kind = ActionKind.parse(decision.kind)
    if kind is ActionKind.WAIT:
        if decision.wait_ticks < 1:
            raise ValueError("wait decision must consume at least one tick")
        return (Action.wait(decision.wait_ticks),)
    if kind not in (ActionKind.WEAK_SHOT, ActionKind.STRONG_SHOT):
        raise ValueError("pointer experts support only wait, weak, and strong")
    if decision.target_body_id is None:
        raise ValueError("shot decision requires a target body ID")
    body = target_body(observation, decision.target_body_id)
    if body is None:
        raise ValueError("target body ID is absent from the public observation")
    x_offset, y_offset = _template(spec, decision.template_index)
    radius = float(body["size"]) / 2.0
    x = float(body["x"]) + x_offset * radius
    x = min(float(client_width), max(0.0, x))
    y = float(body["y"]) + y_offset * radius
    y = min(float(client_height), max(0.0, y))
    press = (
        Action.weak(x, y)
        if kind is ActionKind.WEAK_SHOT
        else Action.strong(x, y)
    )
    # The release tick is part of the semantic action contract.
    return press, Action.wait(1)


def _run_candidate(
    env: Any,
    initial_observation: Mapping[str, Any],
    candidate: SearchCandidate,
    spec: PointerActionSpec,
    weights: SearchUtilityWeights,
    tick_budget: int,
) -> tuple[UtilityBreakdown, Mapping[str, Any], int]:
    observation = initial_observation
    events: list[Mapping[str, Any]] = []
    terminated = bool(initial_observation.get("terminated", False))
    truncated = bool(initial_observation.get("truncated", False))
    for action in lower_expert_decision(candidate.decision, observation, spec):
        if terminated or truncated:
            break
        observation, _, terminated, truncated, info = env.step(action)
        events.extend(info.get("events", ()))
    primitive_ticks = int(observation.get("tick", 0)) - int(
        initial_observation.get("tick", 0)
    )
    if primitive_ticks < 0 or primitive_ticks > tick_budget:
        raise RuntimeError("branch tick progression exceeded its spawn-censored budget")
    if not terminated and not truncated and primitive_ticks < tick_budget:
        observation, _, terminated, truncated, info = env.step(
            Action.wait(tick_budget - primitive_ticks)
        )
        events.extend(info.get("events", ()))
        primitive_ticks = int(observation.get("tick", 0)) - int(
            initial_observation.get("tick", 0)
        )
        if primitive_ticks < 0 or primitive_ticks > tick_budget:
            raise RuntimeError(
                "branch tick progression exceeded its spawn-censored budget"
            )
    return (
        utility_breakdown(
            initial_observation,
            observation,
            events,
            terminated=terminated,
            truncated=truncated,
            weights=weights,
        ),
        observation,
        primitive_ticks,
    )


class SpawnCensoredSearchTeacher:
    """One-ply deterministic teacher that cannot inspect the next spawn."""

    def __init__(
        self,
        *,
        pointer_spec: PointerActionSpec | None = None,
        weights: SearchUtilityWeights | None = None,
        max_target_bodies: int = 8,
        max_rollout_ticks: int = 64,
        max_candidates: int = 64,
        candidate_generator: Callable[
            [Mapping[str, Any], PointerActionSpec], Sequence[SearchCandidate]
        ]
        | None = None,
        continuation_evaluator: Callable[[Mapping[str, Any]], float] | None = None,
        continuation_scale: float = 1.0,
    ) -> None:
        if type(max_target_bodies) is not int or max_target_bodies < 1:
            raise ValueError("max_target_bodies must be a positive integer")
        if type(max_rollout_ticks) is not int or max_rollout_ticks < 1:
            raise ValueError("max_rollout_ticks must be a positive integer")
        self.pointer_spec = (
            PointerActionSpec() if pointer_spec is None else pointer_spec
        )
        minimum_candidates = 4 + len(self.pointer_spec.wait_choices)
        if type(max_candidates) is not int or max_candidates < minimum_candidates:
            raise ValueError(
                "max_candidates must fit all four anchors and every wait choice"
            )
        self.weights = SearchUtilityWeights() if weights is None else weights
        self.max_target_bodies = max_target_bodies
        self.max_rollout_ticks = max_rollout_ticks
        self.max_candidates = max_candidates
        self._candidate_generator = candidate_generator
        if (
            continuation_evaluator is not None
            and not callable(continuation_evaluator)
        ):
            raise TypeError("continuation evaluator must be callable")
        if (
            isinstance(continuation_scale, bool)
            or not isinstance(continuation_scale, (int, float))
            or not math.isfinite(float(continuation_scale))
            or float(continuation_scale) < 0.0
        ):
            raise ValueError("continuation scale must be finite and nonnegative")
        self._continuation_evaluator = continuation_evaluator
        self.continuation_scale = float(continuation_scale)

    def set_recurrent_context(self, state: Any | None) -> None:
        """Bind a policy-history state when the continuation critic supports it."""

        setter = getattr(
            self._continuation_evaluator, "set_recurrent_state", None
        )
        if callable(setter):
            setter(state)

    def candidates(
        self, observation: Mapping[str, Any]
    ) -> tuple[SearchCandidate, ...]:
        if self._candidate_generator is not None:
            return tuple(self._candidate_generator(observation, self.pointer_spec))
        return generate_candidates(
            observation,
            self.pointer_spec,
            max_target_bodies=self.max_target_bodies,
        )

    def _capped_candidates(
        self, observation: Mapping[str, Any]
    ) -> tuple[SearchCandidate, ...]:
        candidates = self.candidates(observation)
        if len(candidates) <= self.max_candidates:
            return candidates
        mandatory = {
            index
            for index, candidate in enumerate(candidates)
            if candidate.name.startswith(("anchor/", "wait/"))
        }
        if len(mandatory) > self.max_candidates:
            raise ValueError("candidate cap cannot retain all anchors and waits")
        selected = list(sorted(mandatory))
        selected_set = set(selected)
        body_ids: list[int] = []
        shot_indices: dict[tuple[int, int, int], list[int]] = {}
        for index, candidate in enumerate(candidates):
            if index in mandatory:
                continue
            decision = candidate.decision
            if (
                decision.kind
                in (int(ActionKind.WEAK_SHOT), int(ActionKind.STRONG_SHOT))
                and decision.target_body_id is not None
            ):
                body_id = int(decision.target_body_id)
                if body_id not in body_ids:
                    body_ids.append(body_id)
                shot_indices.setdefault(
                    (decision.template_index, body_id, decision.kind), []
                ).append(index)
        for template_index in diverse_template_indices(self.pointer_spec):
            for body_id in body_ids:
                for kind in (
                    int(ActionKind.WEAK_SHOT),
                    int(ActionKind.STRONG_SHOT),
                ):
                    matches = shot_indices.get((template_index, body_id, kind), ())
                    for index in matches:
                        if len(selected) == self.max_candidates:
                            break
                        if index not in selected_set:
                            selected.append(index)
                            selected_set.add(index)
                    if len(selected) == self.max_candidates:
                        break
                if len(selected) == self.max_candidates:
                    break
            if len(selected) == self.max_candidates:
                break
        for index in range(len(candidates)):
            if len(selected) == self.max_candidates:
                break
            if index not in selected_set:
                selected.append(index)
                selected_set.add(index)
        return tuple(candidates[index] for index in selected)

    def _evaluation(
        self,
        ordinal: int,
        candidate: SearchCandidate,
        utility: UtilityBreakdown,
        final_observation: Mapping[str, Any],
        primitive_ticks: int,
    ) -> CandidateEvaluation:
        continuation = (
            0.0
            if self._continuation_evaluator is None
            else float(self._continuation_evaluator(final_observation))
        )
        if not math.isfinite(continuation):
            raise FloatingPointError("search continuation value is nonfinite")
        return CandidateEvaluation(
            ordinal,
            candidate,
            utility,
            final_observation,
            primitive_ticks,
            continuation,
            utility.total + self.continuation_scale * continuation,
        )

    def search(self, env: Any, observation: Mapping[str, Any]) -> SearchResult:
        safe_ticks = ticks_before_next_spawn(observation)
        evaluated_ticks = min(safe_ticks, self.max_rollout_ticks)
        candidates = self._capped_candidates(observation)
        eligible = [
            (ordinal, candidate)
            for ordinal, candidate in enumerate(candidates)
            if candidate.decision.primitive_ticks <= evaluated_ticks
        ]
        if not eligible:
            return SearchResult(
                decision=PointerExpertDecision.wait(1),
                selected_name="censored/no-safe-candidate",
                evaluations=(),
                safe_tick_budget=safe_ticks,
                evaluated_tick_budget=evaluated_ticks,
            )

        evaluations: list[CandidateEvaluation] = []
        if getattr(env, "physics_backend", None) == "exact":
            with env.fast_checkpoint() as checkpoint:
                for ordinal, candidate in eligible:
                    with checkpoint.branch() as branch:
                        utility, final_observation, primitive_ticks = _run_candidate(
                            branch,
                            observation,
                            candidate,
                            self.pointer_spec,
                            self.weights,
                            evaluated_ticks,
                        )
                    evaluations.append(
                        self._evaluation(
                            ordinal,
                            candidate,
                            utility,
                            final_observation,
                            primitive_ticks,
                        )
                    )
        else:
            snapshot = env.clone_state()
            try:
                for ordinal, candidate in eligible:
                    env.restore_state(snapshot)
                    utility, final_observation, primitive_ticks = _run_candidate(
                        env,
                        observation,
                        candidate,
                        self.pointer_spec,
                        self.weights,
                        evaluated_ticks,
                    )
                    evaluations.append(
                        self._evaluation(
                            ordinal,
                            candidate,
                            utility,
                            final_observation,
                            primitive_ticks,
                        )
                    )
            finally:
                env.restore_state(snapshot)

        # Earlier deterministic candidate order wins exact utility ties.
        winner = max(
            evaluations,
            key=lambda evaluation: (evaluation.search_score, -evaluation.ordinal),
        )
        return SearchResult(
            decision=winner.candidate.decision,
            selected_name=winner.candidate.name,
            evaluations=tuple(evaluations),
            safe_tick_budget=safe_ticks,
            evaluated_tick_budget=evaluated_ticks,
        )

    def act(
        self, env: Any, observation: Mapping[str, Any]
    ) -> PointerExpertDecision:
        return self.search(env, observation).decision

    choose = act


__all__ = [
    "CandidateEvaluation",
    "SearchResult",
    "SearchUtilityWeights",
    "SpawnCensoredSearchTeacher",
    "UtilityBreakdown",
    "lower_expert_decision",
    "ticks_before_next_spawn",
    "utility_breakdown",
]
