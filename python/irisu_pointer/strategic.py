"""Public-state strategic abstractions for long-horizon IriSu training.

This module deliberately consumes only the policy observation.  Snapshot
bytes, simulator internals, and future random state are outside its API.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from statistics import median
from typing import Any


_LIVE_LIFECYCLES = frozenset(
    {"scripted_falling", "dynamic_fresh", "confirmed", "rotten"}
)


class StrategicIntent(str, Enum):
    """High-level options executed by a closed-loop steering controller."""

    STEER_MATCH = "steer_match"
    EXTEND_ANCHOR = "extend_anchor"
    PRESERVE_GROUP = "preserve_group"
    HARVEST = "harvest"
    MATCH_ROTTEN = "match_rotten"
    EJECT_HAZARD = "eject_hazard"
    ORB_CLEAN = "orb_clean"
    TRIAGE = "triage"


class CurriculumStage(str, Enum):
    """Fail-closed progression from steering to strategic distillation."""

    MICRO_CONTROL = "micro_control"
    GROUP_FORMATION = "group_formation"
    SURVIVAL_OPTIONS = "survival_options"
    ARCHIVE_PLANNING = "archive_planning"
    DISTILLATION = "distillation"


@dataclass(frozen=True, slots=True)
class StrategicFeatureConfig:
    """Thresholds used to summarize the visible board."""

    floor_zone_fraction: float = 0.20
    imminent_lifetime_spawns: float = 2.0
    low_gauge_fraction: float = 0.20

    def __post_init__(self) -> None:
        for name, value in (
            ("floor_zone_fraction", self.floor_zone_fraction),
            ("imminent_lifetime_spawns", self.imminent_lifetime_spawns),
            ("low_gauge_fraction", self.low_gauge_fraction),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.floor_zone_fraction < 1.0:
            raise ValueError("floor_zone_fraction must be in (0, 1)")
        if self.imminent_lifetime_spawns <= 0.0:
            raise ValueError("imminent_lifetime_spawns must be positive")
        if not 0.0 <= self.low_gauge_fraction <= 1.0:
            raise ValueError("low_gauge_fraction must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class GroupFeatures:
    """One visible group.  Chain IDs are public but never enter archive cells."""

    color: int
    chain_id: int
    members: int
    non_rotten_members: int
    floor_members: int
    minimum_remaining_lifetime: int
    deletion_hit_budget: int
    safe_direct_hit_budget: int
    center_x: float
    center_y: float
    viable_anchor: bool


@dataclass(frozen=True, slots=True)
class ColorFeatures:
    """Permutation-stable state for one public color."""

    color: int
    ungrouped_fresh: int
    ungrouped_rotten: int
    floor_ungrouped: int
    groups: tuple[GroupFeatures, ...]

    @property
    def ungrouped_total(self) -> int:
        return self.ungrouped_fresh + self.ungrouped_rotten

    @property
    def viable_anchor_count(self) -> int:
        return sum(group.viable_anchor for group in self.groups)

    @property
    def largest_group(self) -> int:
        return max((group.members for group in self.groups), default=0)

    @property
    def can_steer_match(self) -> bool:
        return self.ungrouped_total >= 2

    @property
    def can_extend_anchor(self) -> bool:
        return self.ungrouped_total > 0 and self.viable_anchor_count > 0


@dataclass(frozen=True, slots=True)
class StrategicFeatures:
    """Deterministic features extracted solely from the current observation."""

    tick: int
    raw_score: int
    level: int
    gauge: int
    gauge_max: int
    gauge_fraction: float
    low_gauge: bool
    highest_chain: int
    qualifying_clears: int
    active_colors: int
    spawn_interval_ticks: int
    ticks_to_spawn: int
    spawn_phase: float
    colors: tuple[ColorFeatures, ...]
    live_piece_count: int
    bonus_count: int
    viable_anchor_count: int
    largest_group: int
    grouped_piece_count: int
    total_deletion_hit_budget: int
    total_safe_direct_hit_budget: int
    fragile_group_count: int
    rotten_piece_count: int
    rot_active_count: int
    imminent_expiry_count: int
    floor_hazard_count: int
    unmatched_hazard_count: int
    terminated: bool
    truncated: bool

    @property
    def alive(self) -> bool:
        return not self.terminated and not self.truncated

    @property
    def under_pressure(self) -> bool:
        return (
            self.low_gauge
            or self.rotten_piece_count > 0
            or self.floor_hazard_count >= 3
        )

    def color(self, color: int) -> ColorFeatures | None:
        return next((value for value in self.colors if value.color == color), None)


def _pieces(observation: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = observation.get("bodies", ())
    if not isinstance(raw, Sequence):
        raise TypeError("observation bodies must be a sequence")
    result: list[Mapping[str, Any]] = []
    for body in raw:
        if not isinstance(body, Mapping):
            raise TypeError("observation body must be a mapping")
        if (
            body.get("kind") == "piece"
            and body.get("lifecycle") in _LIVE_LIFECYCLES
        ):
            result.append(body)
    return result


def _ticks_to_spawn(tick: int, interval: int) -> int:
    remainder = tick % interval
    return 0 if remainder == 0 else interval - remainder


def extract_strategic_features(
    observation: Mapping[str, Any],
    config: StrategicFeatureConfig | None = None,
) -> StrategicFeatures:
    """Summarize public groups, hazards, endurance, and spawn cadence."""

    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")
    resolved = StrategicFeatureConfig() if config is None else config
    tick = int(observation.get("tick", 0))
    score = int(observation.get("score", 0))
    level = int(observation.get("level", 0))
    gauge = int(observation.get("gauge", 0))
    gauge_max = int(observation.get("gauge_max", 0))
    highest_chain = int(observation.get("highest_chain", 0))
    clears = int(observation.get("qualifying_clear_count", 0))
    difficulty = observation.get("difficulty", {})
    field = observation.get("field", {})
    if not isinstance(difficulty, Mapping) or not isinstance(field, Mapping):
        raise TypeError("difficulty and field must be mappings")
    active_colors = int(difficulty.get("active_colors", 0))
    interval = int(difficulty.get("spawn_interval_ticks", 0))
    if tick < 0 or score < 0 or level < 0 or highest_chain < 0 or clears < 0:
        raise ValueError("public counters must be nonnegative")
    if gauge_max <= 0 or not 0 <= gauge <= gauge_max:
        raise ValueError("gauge must be in [0, gauge_max] with positive gauge_max")
    if active_colors < 0:
        raise ValueError("active_colors must be nonnegative")
    if interval <= 0:
        raise ValueError("spawn_interval_ticks must be positive")

    field_y = float(field.get("y", 0.0))
    field_height = float(field.get("height", 480.0))
    if (
        not math.isfinite(field_y)
        or not math.isfinite(field_height)
        or field_height <= 0.0
    ):
        raise ValueError("field geometry must be finite with positive height")
    floor_threshold = (
        field_y + field_height * (1.0 - resolved.floor_zone_fraction)
    )
    imminent_ticks = max(
        1, math.ceil(interval * resolved.imminent_lifetime_spawns)
    )

    pieces = _pieces(observation)
    by_group: dict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    ungrouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    floor_ids: set[int] = set()
    rotten_ids: set[int] = set()
    active_rot_ids: set[int] = set()
    expiry_ids: set[int] = set()
    for body in pieces:
        identifier = int(body.get("id", -1))
        color = int(body.get("color", -1))
        chain_id = int(body.get("chain_id", 0))
        y = float(body.get("y", 0.0))
        size = max(0.0, float(body.get("size", 0.0)))
        remaining = int(body.get("remaining_lifetime", 0))
        if not math.isfinite(y) or not math.isfinite(size):
            raise ValueError("piece geometry must be finite")
        if y + size / 2.0 >= floor_threshold:
            floor_ids.add(identifier)
        if body.get("lifecycle") == "rotten":
            rotten_ids.add(identifier)
        if int(body.get("rot_timer", 0)) > 0:
            active_rot_ids.add(identifier)
        if remaining <= imminent_ticks:
            expiry_ids.add(identifier)
        if chain_id:
            by_group[(color, chain_id)].append(body)
        else:
            ungrouped[color].append(body)

    groups_by_color: dict[int, list[GroupFeatures]] = defaultdict(list)
    for (color, chain_id), members in sorted(by_group.items()):
        hit_budgets = [
            max(0, 2 - int(body.get("projectile_hits", 0))) for body in members
        ]
        safe_budgets = [
            max(0, 1 - int(body.get("projectile_hits", 0))) for body in members
        ]
        floor_members = sum(
            int(body.get("id", -1)) in floor_ids for body in members
        )
        non_rotten = sum(body.get("lifecycle") != "rotten" for body in members)
        minimum_lifetime = min(
            (int(body.get("remaining_lifetime", 0)) for body in members),
            default=0,
        )
        center_x = sum(float(body.get("x", 0.0)) for body in members) / len(
            members
        )
        center_y = sum(float(body.get("y", 0.0)) for body in members) / len(
            members
        )
        viable = (
            len(members) >= 2
            and non_rotten > 0
            and floor_members == 0
            and minimum_lifetime > imminent_ticks
        )
        groups_by_color[color].append(
            GroupFeatures(
                color=color,
                chain_id=chain_id,
                members=len(members),
                non_rotten_members=non_rotten,
                floor_members=floor_members,
                minimum_remaining_lifetime=minimum_lifetime,
                deletion_hit_budget=sum(hit_budgets),
                safe_direct_hit_budget=sum(safe_budgets),
                center_x=center_x,
                center_y=center_y,
                viable_anchor=viable,
            )
        )

    observed_colors = set(ungrouped) | set(groups_by_color)
    colors = tuple(
        ColorFeatures(
            color=color,
            ungrouped_fresh=sum(
                body.get("lifecycle") != "rotten"
                for body in ungrouped.get(color, ())
            ),
            ungrouped_rotten=sum(
                body.get("lifecycle") == "rotten"
                for body in ungrouped.get(color, ())
            ),
            floor_ungrouped=sum(
                int(body.get("id", -1)) in floor_ids
                for body in ungrouped.get(color, ())
            ),
            groups=tuple(
                sorted(
                    groups_by_color.get(color, ()),
                    key=lambda group: (group.chain_id, group.center_x, group.center_y),
                )
            ),
        )
        for color in sorted(observed_colors)
    )
    grouped_count = sum(group.members for color in colors for group in color.groups)
    deletion_budget = sum(
        group.deletion_hit_budget for color in colors for group in color.groups
    )
    safe_budget = sum(
        group.safe_direct_hit_budget for color in colors for group in color.groups
    )
    fragile_groups = sum(
        group.safe_direct_hit_budget == 0
        for color in colors
        for group in color.groups
    )
    unmatched_hazards = 0
    for color in colors:
        compatible = color.ungrouped_total + sum(
            group.viable_anchor for group in color.groups
        )
        if compatible < 2:
            unmatched_hazards += sum(
                body.get("lifecycle") == "rotten"
                or int(body.get("id", -1)) in floor_ids
                for body in ungrouped.get(color.color, ())
            )

    floor_hazard_ids = {
        int(body.get("id", -1))
        for body in pieces
        if int(body.get("id", -1)) in floor_ids
        and (
            int(body.get("chain_id", 0)) == 0
            or body.get("lifecycle") == "rotten"
        )
    }

    bodies = observation.get("bodies", ())
    bonus_count = sum(
        isinstance(body, Mapping)
        and body.get("kind") == "bonus"
        and body.get("lifecycle") in _LIVE_LIFECYCLES
        for body in bodies
    )
    ticks_to_spawn = _ticks_to_spawn(tick, interval)
    return StrategicFeatures(
        tick=tick,
        raw_score=score,
        level=level,
        gauge=gauge,
        gauge_max=gauge_max,
        gauge_fraction=gauge / gauge_max,
        low_gauge=gauge / gauge_max <= resolved.low_gauge_fraction,
        highest_chain=highest_chain,
        qualifying_clears=clears,
        active_colors=active_colors,
        spawn_interval_ticks=interval,
        ticks_to_spawn=ticks_to_spawn,
        spawn_phase=(tick % interval) / interval,
        colors=colors,
        live_piece_count=len(pieces),
        bonus_count=bonus_count,
        viable_anchor_count=sum(
            color.viable_anchor_count for color in colors
        ),
        largest_group=max((color.largest_group for color in colors), default=0),
        grouped_piece_count=grouped_count,
        total_deletion_hit_budget=deletion_budget,
        total_safe_direct_hit_budget=safe_budget,
        fragile_group_count=fragile_groups,
        rotten_piece_count=len(rotten_ids),
        rot_active_count=len(active_rot_ids),
        imminent_expiry_count=len(expiry_ids),
        floor_hazard_count=len(floor_hazard_ids),
        unmatched_hazard_count=unmatched_hazards,
        terminated=bool(observation.get("terminated", False)),
        truncated=bool(observation.get("truncated", False)),
    )


def available_intents(features: StrategicFeatures) -> tuple[StrategicIntent, ...]:
    """Return applicable intents in deterministic safety-first order."""

    intents: list[StrategicIntent] = []
    if features.under_pressure:
        intents.append(StrategicIntent.TRIAGE)
    if features.bonus_count and (
        features.rotten_piece_count or features.floor_hazard_count
    ):
        intents.append(StrategicIntent.ORB_CLEAN)
    if any(
        color.ungrouped_rotten
        and (color.ungrouped_total >= 2 or color.viable_anchor_count)
        for color in features.colors
    ):
        intents.append(StrategicIntent.MATCH_ROTTEN)
    if features.unmatched_hazard_count:
        intents.append(StrategicIntent.EJECT_HAZARD)
    if features.fragile_group_count or any(
        group.floor_members
        for color in features.colors
        for group in color.groups
    ):
        intents.append(StrategicIntent.PRESERVE_GROUP)
    if any(color.can_extend_anchor for color in features.colors):
        intents.append(StrategicIntent.EXTEND_ANCHOR)
    if any(color.can_steer_match for color in features.colors):
        intents.append(StrategicIntent.STEER_MATCH)
    if features.grouped_piece_count:
        intents.append(StrategicIntent.HARVEST)
    if StrategicIntent.TRIAGE not in intents:
        intents.append(StrategicIntent.TRIAGE)
    return tuple(dict.fromkeys(intents))


@dataclass(frozen=True, slots=True)
class PotentialWeights:
    """Diagnostic potential weights; raw score remains the selection objective."""

    chain_score: float = 1.0
    matchable_pair: float = 4.0
    gauge_fraction: float = 64.0
    safe_hit_budget: float = 1.0
    rot_hazard: float = -32.0
    floor_hazard: float = -16.0
    expiry_hazard: float = -8.0


@dataclass(frozen=True, slots=True)
class StrategicPotential:
    """Auditable decomposition of a public diagnostic potential."""

    chain_score: float
    matchable_pairs: int
    gauge_fraction: float
    safe_hit_budget: int
    rot_hazards: int
    floor_hazards: int
    expiry_hazards: int
    total: float


def strategic_potential(
    features: StrategicFeatures,
    weights: PotentialWeights | None = None,
) -> StrategicPotential:
    """Compute a diagnostic, policy-invariant shaping potential."""

    resolved = PotentialWeights() if weights is None else weights
    chain_score = sum(
        2.0 * group.members**3 * max(features.level, 1) ** 0.7
        for color in features.colors
        for group in color.groups
    )
    matchable_pairs = sum(color.ungrouped_total // 2 for color in features.colors)
    total = (
        resolved.chain_score * chain_score
        + resolved.matchable_pair * matchable_pairs
        + resolved.gauge_fraction * features.gauge_fraction
        + resolved.safe_hit_budget * features.total_safe_direct_hit_budget
        + resolved.rot_hazard * features.rotten_piece_count
        + resolved.floor_hazard * features.floor_hazard_count
        + resolved.expiry_hazard * features.imminent_expiry_count
    )
    return StrategicPotential(
        chain_score=chain_score,
        matchable_pairs=matchable_pairs,
        gauge_fraction=features.gauge_fraction,
        safe_hit_budget=features.total_safe_direct_hit_budget,
        rot_hazards=features.rotten_piece_count,
        floor_hazards=features.floor_hazard_count,
        expiry_hazards=features.imminent_expiry_count,
        total=total,
    )


def potential_shaping_delta(
    before: StrategicPotential,
    after: StrategicPotential,
    *,
    gamma: float,
    duration_ticks: int = 1,
    terminal: bool = False,
) -> float:
    """Return ``gamma**duration * Phi(s') - Phi(s)``.

    A terminal transition uses zero terminal potential.  This function is a
    training diagnostic only and must not rank archive elites.
    """

    if (
        isinstance(gamma, bool)
        or not isinstance(gamma, (int, float))
        or not math.isfinite(float(gamma))
        or not 0.0 <= gamma <= 1.0
    ):
        raise ValueError("gamma must be finite and in [0, 1]")
    if (
        isinstance(duration_ticks, bool)
        or not isinstance(duration_ticks, int)
        or duration_ticks < 1
    ):
        raise ValueError("duration_ticks must be a positive integer")
    terminal_value = 0.0 if terminal else after.total
    return float(gamma) ** duration_ticks * terminal_value - before.total


@dataclass(frozen=True, slots=True)
class CurriculumMetrics:
    """Evaluation counts used by curriculum gates without rounded rates."""

    episodes: int
    shots: int
    projectile_hits: int
    chain_joins: int
    qualifying_clears: int
    raw_scores: tuple[int, ...]
    survival_ticks: tuple[int, ...]
    highest_chains: tuple[int, ...]
    gauge_failures: int = 0
    baseline_median_score: float = 0.0
    baseline_median_survival: float = 0.0

    def __post_init__(self) -> None:
        counters = (
            self.episodes,
            self.shots,
            self.projectile_hits,
            self.chain_joins,
            self.qualifying_clears,
            self.gauge_failures,
        )
        if any(type(value) is not int or value < 0 for value in counters):
            raise ValueError("curriculum counters must be nonnegative integers")
        if not (
            len(self.raw_scores)
            == len(self.survival_ticks)
            == len(self.highest_chains)
            == self.episodes
        ):
            raise ValueError("episode metric tuples must match episodes")
        if any(type(value) is not int or value < 0 for value in self.raw_scores):
            raise ValueError("raw scores must be nonnegative integers")
        if any(
            type(value) is not int or value < 0 for value in self.survival_ticks
        ):
            raise ValueError("survival ticks must be nonnegative integers")
        if any(
            type(value) is not int or value < 0 for value in self.highest_chains
        ):
            raise ValueError("highest chains must be nonnegative integers")
        if self.gauge_failures > self.episodes:
            raise ValueError("gauge failures cannot exceed episodes")
        for value in (
            self.baseline_median_score,
            self.baseline_median_survival,
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError("baseline medians must be finite and nonnegative")

    @property
    def hit_rate(self) -> float:
        return self.projectile_hits / self.shots if self.shots else 0.0

    @property
    def joins_per_shot(self) -> float:
        return self.chain_joins / self.shots if self.shots else 0.0

    @property
    def clears_per_shot(self) -> float:
        return self.qualifying_clears / self.shots if self.shots else 0.0

    @property
    def median_score(self) -> float:
        return float(median(self.raw_scores)) if self.raw_scores else 0.0

    @property
    def median_survival(self) -> float:
        return float(median(self.survival_ticks)) if self.survival_ticks else 0.0

    @property
    def max_highest_chain(self) -> int:
        return max(self.highest_chains, default=0)

    @property
    def gauge_failure_rate(self) -> float:
        return self.gauge_failures / self.episodes if self.episodes else 1.0


@dataclass(frozen=True, slots=True)
class CurriculumGate:
    """Explicit thresholds for one promotion boundary."""

    stage: CurriculumStage
    min_episodes: int
    min_shots: int = 0
    min_hit_rate: float = 0.0
    min_joins_per_shot: float = 0.0
    min_clears_per_shot: float = 0.0
    min_score_ratio: float = 0.0
    min_survival_ratio: float = 0.0
    min_highest_chain: int = 0
    max_gauge_failure_rate: float = 1.0

    def __post_init__(self) -> None:
        if type(self.min_episodes) is not int or self.min_episodes < 1:
            raise ValueError("min_episodes must be a positive integer")
        if type(self.min_shots) is not int or self.min_shots < 0:
            raise ValueError("min_shots must be a nonnegative integer")
        if type(self.min_highest_chain) is not int or self.min_highest_chain < 0:
            raise ValueError("min_highest_chain must be a nonnegative integer")
        for name, value in (
            ("min_hit_rate", self.min_hit_rate),
            ("min_joins_per_shot", self.min_joins_per_shot),
            ("min_clears_per_shot", self.min_clears_per_shot),
            ("min_score_ratio", self.min_score_ratio),
            ("min_survival_ratio", self.min_survival_ratio),
            ("max_gauge_failure_rate", self.max_gauge_failure_rate),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0.0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_gauge_failure_rate > 1.0:
            raise ValueError("max_gauge_failure_rate must be at most one")

    def evaluate(self, metrics: CurriculumMetrics) -> GateResult:
        failures: list[str] = []
        if metrics.episodes < self.min_episodes:
            failures.append("episodes")
        if metrics.shots < self.min_shots:
            failures.append("shots")
        if metrics.hit_rate < self.min_hit_rate:
            failures.append("hit_rate")
        if metrics.joins_per_shot < self.min_joins_per_shot:
            failures.append("joins_per_shot")
        if metrics.clears_per_shot < self.min_clears_per_shot:
            failures.append("clears_per_shot")
        score_floor = metrics.baseline_median_score * self.min_score_ratio
        if metrics.median_score < score_floor:
            failures.append("median_score")
        survival_floor = (
            metrics.baseline_median_survival * self.min_survival_ratio
        )
        if metrics.median_survival < survival_floor:
            failures.append("median_survival")
        if metrics.max_highest_chain < self.min_highest_chain:
            failures.append("highest_chain")
        if metrics.gauge_failure_rate > self.max_gauge_failure_rate:
            failures.append("gauge_failure_rate")
        return GateResult(
            stage=self.stage,
            passed=not failures,
            failed_metrics=tuple(failures),
        )


@dataclass(frozen=True, slots=True)
class GateResult:
    stage: CurriculumStage
    passed: bool
    failed_metrics: tuple[str, ...]


DEFAULT_CURRICULUM_GATES: tuple[CurriculumGate, ...] = (
    CurriculumGate(
        CurriculumStage.MICRO_CONTROL,
        min_episodes=8,
        min_shots=100,
        min_hit_rate=0.90,
    ),
    CurriculumGate(
        CurriculumStage.GROUP_FORMATION,
        min_episodes=8,
        min_shots=100,
        min_hit_rate=0.90,
        min_joins_per_shot=0.20,
        min_clears_per_shot=0.10,
    ),
    CurriculumGate(
        CurriculumStage.SURVIVAL_OPTIONS,
        min_episodes=16,
        min_score_ratio=1.0,
        min_survival_ratio=0.95,
        max_gauge_failure_rate=0.25,
    ),
    CurriculumGate(
        CurriculumStage.ARCHIVE_PLANNING,
        min_episodes=16,
        min_score_ratio=3.0,
        min_survival_ratio=1.0,
        min_highest_chain=4,
        max_gauge_failure_rate=0.20,
    ),
    CurriculumGate(
        CurriculumStage.DISTILLATION,
        min_episodes=16,
        min_score_ratio=3.0,
        min_survival_ratio=1.0,
        min_highest_chain=4,
        max_gauge_failure_rate=0.20,
    ),
)


def evaluate_curriculum(
    metrics: CurriculumMetrics,
    gates: Sequence[CurriculumGate] = DEFAULT_CURRICULUM_GATES,
) -> tuple[GateResult, ...]:
    """Evaluate gates in order and fail closed after the first failed stage."""

    results: list[GateResult] = []
    blocked = False
    for gate in gates:
        if blocked:
            result = GateResult(
                stage=gate.stage,
                passed=False,
                failed_metrics=("prerequisite_stage",),
            )
        else:
            result = gate.evaluate(metrics)
            blocked = not result.passed
        results.append(result)
    return tuple(results)


__all__ = [
    "ColorFeatures",
    "CurriculumGate",
    "CurriculumMetrics",
    "CurriculumStage",
    "DEFAULT_CURRICULUM_GATES",
    "GateResult",
    "GroupFeatures",
    "PotentialWeights",
    "StrategicFeatureConfig",
    "StrategicFeatures",
    "StrategicIntent",
    "StrategicPotential",
    "available_intents",
    "evaluate_curriculum",
    "extract_strategic_features",
    "potential_shaping_delta",
    "strategic_potential",
]
