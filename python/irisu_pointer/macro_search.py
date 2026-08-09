"""Bounded multi-action search that stops before the next random spawn."""

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
    expert_anchors,
    matcher_anchor,
)
from .search import lower_expert_decision, ticks_before_next_spawn


CandidateGenerator = Callable[
    [Mapping[str, Any], PointerActionSpec], Sequence[SearchCandidate]
]
FallbackTeacher = Callable[
    [Mapping[str, Any], PointerActionSpec], PointerExpertDecision
]
ContinuationEvaluator = Callable[[Mapping[str, Any]], float]


@dataclass(frozen=True, slots=True)
class MacroUtilityWeights:
    """Raw-score-scaled public utility for pruning and leaf selection."""

    score: float = 1.0
    gauge_delta: float = 0.01
    chain_joined: float = 4.0
    chain_potential: float = 1.0
    highest_chain: float = 4.0
    qualifying_clear: float = 2.0
    ejected: float = -8.0
    rotten: float = -512.0
    destructive_confirmed_hit: float = -64.0
    invalid_action: float = -100_000.0
    terminal: float = -100_000.0


@dataclass(frozen=True, slots=True)
class MacroUtility:
    score_delta: int
    gauge_delta: int
    chain_joined: int
    chain_potential_delta: float
    highest_chain_delta: int
    qualifying_clear_delta: int
    ejected: int
    rotten: int
    destructive_confirmed_hit: int
    invalid_action: int
    terminal: bool
    total: float


@dataclass(frozen=True, slots=True)
class MacroEvaluation:
    path: tuple[str, ...]
    first_decision: PointerExpertDecision
    final_observation: Mapping[str, Any]
    utility: MacroUtility
    primitive_ticks: int
    continuation_value: float = 0.0
    search_score: float = 0.0


@dataclass(frozen=True, slots=True)
class MacroSearchResult:
    decision: PointerExpertDecision
    selected_name: str
    evaluations: tuple[MacroEvaluation, ...]
    safe_tick_budget: int
    evaluated_tick_budget: int
    branches_evaluated: int


@dataclass(slots=True)
class _Node:
    snapshot: bytes
    observation: Mapping[str, Any]
    path: tuple[str, ...]
    first_decision: PointerExpertDecision
    events: tuple[Mapping[str, Any], ...]
    primitive_ticks: int
    utility: MacroUtility
    ordinal: int


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


def _central_template(spec: PointerActionSpec, offset: float = 0.0) -> int:
    x_index = min(
        range(len(spec.x_radius_offsets)),
        key=lambda index: (
            abs(float(spec.x_radius_offsets[index]) - float(offset)),
            index,
        ),
    )
    y_index = min(
        range(len(spec.y_radius_offsets)),
        key=lambda index: (
            abs(float(spec.y_radius_offsets[index]) - 2.0),
            index,
        ),
    )
    return x_index * len(spec.y_radius_offsets) + y_index


def generate_macro_candidates(
    observation: Mapping[str, Any],
    spec: PointerActionSpec,
    *,
    max_target_bodies: int = 4,
) -> tuple[SearchCandidate, ...]:
    """Generate compact public candidates for forming one group per color."""

    if type(max_target_bodies) is not int or max_target_bodies < 1:
        raise ValueError("max_target_bodies must be a positive integer")
    result = list(expert_anchors(observation, spec))
    result.extend(
        SearchCandidate(f"wait/{ticks}", PointerExpertDecision.wait(ticks))
        for ticks in spec.wait_choices
    )
    pieces = [
        body
        for body in observation.get("bodies", ())
        if body.get("kind") == "piece"
        and body.get("lifecycle")
        in {"scripted_falling", "dynamic_fresh", "rotten"}
    ]
    color_counts = Counter(int(body.get("color", -1)) for body in pieces)
    ordered = sorted(
        pieces,
        key=lambda body: (
            color_counts[int(body.get("color", -1))] < 2,
            body.get("lifecycle") != "rotten",
            -int(body.get("rot_timer", 0)),
            -float(body.get("y", 0.0)),
            -float(body.get("size", 0.0)),
            int(body.get("color", -1)),
            float(body.get("x", 0.0)),
            int(body.get("id", 0)),
        ),
    )[:max_target_bodies]
    central = _central_template(spec)
    for body in ordered:
        body_id = int(body["id"])
        peers = [
            peer
            for peer in pieces
            if int(peer["id"]) != body_id
            and int(peer.get("color", -1)) == int(body.get("color", -1))
        ]
        templates = [central]
        if peers:
            peer = min(
                peers,
                key=lambda value: (
                    abs(float(value["x"]) - float(body["x"])),
                    abs(float(value["y"]) - float(body["y"])),
                    int(value["id"]),
                ),
            )
            desired_dx = float(peer["x"]) - float(body["x"])
            # Impact transfers momentum away from the launch point.
            steering_offset = -1.0 if desired_dx > 0.0 else 1.0
            templates.append(_central_template(spec, steering_offset))
        for template in dict.fromkeys(templates):
            result.append(
                SearchCandidate(
                    f"macro/weak/{body_id}/{template}",
                    PointerExpertDecision.weak(
                        body_id, template_index=template
                    ),
                )
            )
            result.append(
                SearchCandidate(
                    f"macro/strong/{body_id}/{template}",
                    PointerExpertDecision.strong(
                        body_id, template_index=template
                    ),
                )
            )
    return tuple(result)


def chain_score_potential(observation: Mapping[str, Any]) -> float:
    """Optimistic public score if each currently visible group lands together."""

    counts = Counter(
        int(body.get("chain_id", 0))
        for body in observation.get("bodies", ())
        if body.get("kind") == "piece"
        and int(body.get("chain_id", 0)) != 0
    )
    level = max(1, int(observation.get("level", 1)))
    scale = float(level) ** 0.7
    return sum(2.0 * count**3 * scale for count in counts.values())


def macro_utility(
    initial: Mapping[str, Any],
    final: Mapping[str, Any],
    events: Sequence[Mapping[str, Any]],
    *,
    terminated: bool,
    truncated: bool,
    weights: MacroUtilityWeights | None = None,
) -> MacroUtility:
    """Evaluate only public observations and public events."""

    resolved = MacroUtilityWeights() if weights is None else weights
    counts = Counter(
        kind for event in events if (kind := _event_kind(event)) is not None
    )
    confirmed_ids = {
        int(body["id"])
        for body in initial.get("bodies", ())
        if body.get("kind") == "piece"
        and body.get("lifecycle") == "confirmed"
    }
    destructive = len(
        {
            int(event.get("b", -1))
            for event in events
            if _event_kind(event) == int(EventKind.PROJECTILE_HIT)
            and int(event.get("b", -1)) in confirmed_ids
            and int(event.get("value", 0)) >= 1
        }
    )
    score_delta = int(final.get("score", 0)) - int(initial.get("score", 0))
    gauge_delta = int(final.get("gauge", 0)) - int(initial.get("gauge", 0))
    potential_delta = chain_score_potential(final) - chain_score_potential(initial)
    highest_chain_delta = int(final.get("highest_chain", 0)) - int(
        initial.get("highest_chain", 0)
    )
    qualifying_clear_delta = int(final.get("qualifying_clear_count", 0)) - int(
        initial.get("qualifying_clear_count", 0)
    )
    joined = counts[int(EventKind.CHAIN_JOINED)]
    ejected = counts[int(EventKind.EJECTED)]
    rotten = counts[int(EventKind.ROTTEN)]
    invalid = counts[int(EventKind.INVALID_ACTION)]
    terminal = bool(terminated or truncated)
    total = (
        resolved.score * score_delta
        + resolved.gauge_delta * gauge_delta
        + resolved.chain_joined * joined
        + resolved.chain_potential * potential_delta
        + resolved.highest_chain * highest_chain_delta
        + resolved.qualifying_clear * qualifying_clear_delta
        + resolved.ejected * ejected
        + resolved.rotten * rotten
        + resolved.destructive_confirmed_hit * destructive
        + resolved.invalid_action * invalid
        + resolved.terminal * int(terminal)
    )
    return MacroUtility(
        score_delta=score_delta,
        gauge_delta=gauge_delta,
        chain_joined=joined,
        chain_potential_delta=potential_delta,
        highest_chain_delta=highest_chain_delta,
        qualifying_clear_delta=qualifying_clear_delta,
        ejected=ejected,
        rotten=rotten,
        destructive_confirmed_hit=destructive,
        invalid_action=invalid,
        terminal=terminal,
        total=float(total),
    )


class SpawnCensoredMacroBeamTeacher:
    """Portable snapshot beam search over several actions in one spawn window."""

    def __init__(
        self,
        *,
        pointer_spec: PointerActionSpec | None = None,
        weights: MacroUtilityWeights | None = None,
        max_depth: int = 3,
        beam_width: int = 5,
        max_candidates: int = 25,
        max_target_bodies: int = 4,
        max_rollout_ticks: int = 64,
        settle_ticks: tuple[int, ...] = (0, 8, 24),
        max_branch_evaluations: int = 768,
        candidate_generator: CandidateGenerator | None = None,
        continuation_evaluator: ContinuationEvaluator | None = None,
        continuation_scale: float = 1.0,
        fallback_teacher: FallbackTeacher = matcher_anchor,
        minimum_search_score: float = 0.0,
    ) -> None:
        integers = (
            max_depth,
            beam_width,
            max_candidates,
            max_target_bodies,
            max_rollout_ticks,
            max_branch_evaluations,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise TypeError("macro search bounds must be integers")
        if (
            max_depth < 1
            or beam_width < 1
            or max_candidates < 1
            or max_target_bodies < 1
            or max_rollout_ticks < 1
            or max_branch_evaluations < 1
        ):
            raise ValueError("macro search bounds must be positive")
        if (
            not settle_ticks
            or tuple(sorted(set(settle_ticks))) != settle_ticks
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in settle_ticks
            )
        ):
            raise ValueError("settle ticks must be unique nonnegative integers")
        self.pointer_spec = PointerActionSpec() if pointer_spec is None else pointer_spec
        self.weights = MacroUtilityWeights() if weights is None else weights
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.max_candidates = max_candidates
        self.max_target_bodies = max_target_bodies
        self.max_rollout_ticks = max_rollout_ticks
        self.settle_ticks = settle_ticks
        self.max_branch_evaluations = max_branch_evaluations
        self._candidate_generator = candidate_generator
        if continuation_evaluator is not None and not callable(
            continuation_evaluator
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
        if not callable(fallback_teacher):
            raise TypeError("macro fallback teacher must be callable")
        if (
            isinstance(minimum_search_score, bool)
            or not isinstance(minimum_search_score, (int, float))
            or not math.isfinite(float(minimum_search_score))
        ):
            raise ValueError("minimum macro search score must be finite")
        self.fallback_teacher = fallback_teacher
        self.minimum_search_score = float(minimum_search_score)

    def set_recurrent_context(self, state: Any | None) -> None:
        setter = getattr(
            self._continuation_evaluator, "set_recurrent_state", None
        )
        if callable(setter):
            setter(state)

    def candidates(
        self, observation: Mapping[str, Any]
    ) -> tuple[SearchCandidate, ...]:
        generated = (
            generate_macro_candidates(
                observation,
                self.pointer_spec,
                max_target_bodies=self.max_target_bodies,
            )
            if self._candidate_generator is None
            else tuple(self._candidate_generator(observation, self.pointer_spec))
        )
        return tuple(generated[: self.max_candidates])

    def _macro_variants(
        self,
        observation: Mapping[str, Any],
        remaining_ticks: int,
    ) -> tuple[tuple[SearchCandidate, int, str], ...]:
        variants: list[tuple[SearchCandidate, int, str]] = []
        for candidate in self.candidates(observation):
            decision = candidate.decision
            if decision.kind == int(ActionKind.WAIT):
                if decision.primitive_ticks <= remaining_ticks:
                    variants.append((candidate, 0, candidate.name))
                continue
            for settle in self.settle_ticks:
                if decision.primitive_ticks + settle <= remaining_ticks:
                    variants.append(
                        (
                            candidate,
                            settle,
                            f"{candidate.name};settle={settle}",
                        )
                    )
        return tuple(variants)

    def search(
        self, env: Any, observation: Mapping[str, Any]
    ) -> MacroSearchResult:
        if getattr(env, "physics_backend", None) != "portable":
            raise ValueError("macro beam search currently requires the portable backend")
        safe_ticks = ticks_before_next_spawn(observation)
        budget = min(safe_ticks, self.max_rollout_ticks)
        if budget == 0:
            return MacroSearchResult(
                self.fallback_teacher(observation, self.pointer_spec),
                "censored/no-safe-macro",
                (),
                safe_ticks,
                0,
                0,
            )
        source_snapshot = env.clone_state()
        branches = 0
        ordinal = 0
        beam: list[_Node] = []
        try:
            for depth in range(self.max_depth):
                parents: Sequence[_Node | None] = beam if depth else (None,)
                expanded: list[_Node] = []
                for parent in parents:
                    parent_observation = observation if parent is None else parent.observation
                    parent_snapshot = source_snapshot if parent is None else parent.snapshot
                    parent_ticks = 0 if parent is None else parent.primitive_ticks
                    parent_events = () if parent is None else parent.events
                    parent_path = () if parent is None else parent.path
                    for candidate, settle, name in self._macro_variants(
                        parent_observation, budget - parent_ticks
                    ):
                        if branches >= self.max_branch_evaluations:
                            break
                        env.restore_state(parent_snapshot)
                        current = parent_observation
                        events = list(parent_events)
                        terminated = bool(current.get("terminated", False))
                        truncated = bool(current.get("truncated", False))
                        for action in lower_expert_decision(
                            candidate.decision, current, self.pointer_spec
                        ):
                            if terminated or truncated:
                                break
                            current, _, terminated, truncated, info = env.step(action)
                            events.extend(info.get("events", ()))
                        if settle and not (terminated or truncated):
                            current, _, terminated, truncated, info = env.step(
                                Action.wait(settle)
                            )
                            events.extend(info.get("events", ()))
                        elapsed = int(current.get("tick", 0)) - int(
                            observation.get("tick", 0)
                        )
                        if elapsed < 0 or elapsed > budget:
                            raise RuntimeError(
                                "macro branch crossed its spawn-censored budget"
                            )
                        utility = macro_utility(
                            observation,
                            current,
                            events,
                            terminated=terminated,
                            truncated=truncated,
                            weights=self.weights,
                        )
                        expanded.append(
                            _Node(
                                env.clone_state(),
                                current,
                                parent_path + (name,),
                                (
                                    candidate.decision
                                    if parent is None
                                    else parent.first_decision
                                ),
                                tuple(events),
                                elapsed,
                                utility,
                                ordinal,
                            )
                        )
                        ordinal += 1
                        branches += 1
                    if branches >= self.max_branch_evaluations:
                        break
                if not expanded:
                    break
                expanded.sort(
                    key=lambda node: (
                        node.utility.total,
                        node.utility.score_delta,
                        node.utility.chain_potential_delta,
                        -node.ordinal,
                    ),
                    reverse=True,
                )
                beam = expanded[: self.beam_width]
                if branches >= self.max_branch_evaluations:
                    break
            leaves: list[MacroEvaluation] = []
            for node in beam:
                env.restore_state(node.snapshot)
                current = node.observation
                events = list(node.events)
                terminated = bool(current.get("terminated", False))
                truncated = bool(current.get("truncated", False))
                remaining = budget - node.primitive_ticks
                if remaining and not (terminated or truncated):
                    current, _, terminated, truncated, info = env.step(
                        Action.wait(remaining)
                    )
                    events.extend(info.get("events", ()))
                elapsed = int(current.get("tick", 0)) - int(
                    observation.get("tick", 0)
                )
                if elapsed < 0 or elapsed > budget:
                    raise RuntimeError("macro leaf crossed its spawn-censored budget")
                utility = macro_utility(
                    observation,
                    current,
                    events,
                    terminated=terminated,
                    truncated=truncated,
                    weights=self.weights,
                )
                continuation = (
                    0.0
                    if self._continuation_evaluator is None
                    else float(self._continuation_evaluator(current))
                )
                if not math.isfinite(continuation):
                    raise FloatingPointError(
                        "macro continuation value is nonfinite"
                    )
                leaves.append(
                    MacroEvaluation(
                        node.path,
                        node.first_decision,
                        current,
                        utility,
                        elapsed,
                        continuation,
                        utility.total + self.continuation_scale * continuation,
                    )
                )
            if not leaves:
                return MacroSearchResult(
                    PointerExpertDecision.wait(1),
                    "censored/no-safe-macro",
                    (),
                    safe_ticks,
                    budget,
                    branches,
                )
            winner = max(
                enumerate(leaves),
                key=lambda item: (
                    item[1].search_score,
                    item[1].utility.score_delta,
                    item[1].utility.chain_potential_delta,
                    -item[0],
                ),
            )[1]
            if winner.search_score <= self.minimum_search_score:
                return MacroSearchResult(
                    self.fallback_teacher(observation, self.pointer_spec),
                    "macro-fallback/nonpositive-plan",
                    tuple(leaves),
                    safe_ticks,
                    budget,
                    branches,
                )
            return MacroSearchResult(
                winner.first_decision,
                "macro-plan/" + " > ".join(winner.path),
                tuple(leaves),
                safe_ticks,
                budget,
                branches,
            )
        finally:
            env.restore_state(source_snapshot)

    def act(
        self, env: Any, observation: Mapping[str, Any]
    ) -> PointerExpertDecision:
        return self.search(env, observation).decision

    choose = act


__all__ = [
    "MacroEvaluation",
    "MacroSearchResult",
    "MacroUtility",
    "MacroUtilityWeights",
    "SpawnCensoredMacroBeamTeacher",
    "chain_score_potential",
    "generate_macro_candidates",
    "macro_utility",
]
