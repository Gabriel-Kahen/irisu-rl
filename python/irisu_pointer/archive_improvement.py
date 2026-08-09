"""Evidence-bound branch improvement from portable strategic archive states."""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from irisu_env import Action, ActionKind, EventKind
from irisu_rl.actions import ActionSpec, SemanticAction
from irisu_rl.encoding import TeacherStateEncoder

from .action import PointerActionSpec
from .archive import ArchiveElite, StrategicArchive, archive_cell_key
from .steering import (
    ClosedLoopSteeringExpert,
    SteeringDecision,
    SteeringExpertConfig,
    SteeringIntent,
)
from .steering_learning import SteeringExample, steering_example_from_decision
from .strategic import StrategicFeatures, extract_strategic_features


EnvironmentFactory = Callable[[], Any]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_environment_identity(
    env: Any, binding: ArchiveImprovementBinding
) -> None:
    runner_manifest = getattr(env, "runner_identity_manifest", None)
    config = getattr(env, "config", None)
    if not callable(runner_manifest) or not isinstance(config, Mapping):
        raise ValueError("archive environment lacks identity surfaces")
    if _canonical_sha256(runner_manifest()) != binding.runner_identity_sha256:
        raise ValueError("archive environment runner identity mismatch")
    if _canonical_sha256(config) != binding.environment_config_sha256:
        raise ValueError("archive environment config identity mismatch")
    runtime_sha256 = getattr(env, "runtime_sha256", None)
    if runtime_sha256 is None:
        library_path = getattr(env, "library_path", None)
        if not isinstance(library_path, (str, Path)):
            raise ValueError("archive environment lacks a runtime identity")
        runtime_sha256 = _file_sha256(Path(library_path).resolve(strict=True))
    if runtime_sha256 != binding.runtime_sha256:
        raise ValueError("archive environment runtime identity mismatch")


@dataclass(frozen=True, slots=True)
class ArchiveImprovementBinding:
    """Scientific inputs that authorize one archive improvement collection."""

    archive_sha256: str
    archive_source_identity: str
    environment_config_sha256: str
    runner_identity_sha256: str
    runtime_sha256: str
    version: str = "r3d-archive-improvement-v1"

    def __post_init__(self) -> None:
        for name in (
            "archive_sha256",
            "environment_config_sha256",
            "runner_identity_sha256",
            "runtime_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        if (
            not isinstance(self.archive_source_identity, str)
            or not self.archive_source_identity
            or "\x00" in self.archive_source_identity
        ):
            raise ValueError("archive source identity must be nonempty and NUL-free")
        if self.version != "r3d-archive-improvement-v1":
            raise ValueError("unsupported archive improvement binding version")

    @classmethod
    def create(
        cls,
        archive: StrategicArchive,
        *,
        environment_config_sha256: str,
        runner_identity_sha256: str,
        runtime_sha256: str,
    ) -> ArchiveImprovementBinding:
        return cls(
            archive.sha256,
            archive.source_identity,
            environment_config_sha256,
            runner_identity_sha256,
            runtime_sha256,
        )

    def manifest(self) -> dict[str, str]:
        return {
            "version": self.version,
            "archive_sha256": self.archive_sha256,
            "archive_source_identity": self.archive_source_identity,
            "environment_config_sha256": self.environment_config_sha256,
            "runner_identity_sha256": self.runner_identity_sha256,
            "runtime_sha256": self.runtime_sha256,
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    def verify(self, archive: StrategicArchive) -> None:
        if (
            self.archive_sha256 != archive.sha256
            or self.archive_source_identity != archive.source_identity
        ):
            raise ValueError("archive improvement binding does not match the archive")


@dataclass(frozen=True, slots=True)
class SteeringBranchCandidate:
    name: str
    expert_config: SteeringExpertConfig

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name
            or "\x00" in self.name
        ):
            raise ValueError("steering branch name must be nonempty and NUL-free")
        if not isinstance(self.expert_config, SteeringExpertConfig):
            raise TypeError("steering branch config must be a SteeringExpertConfig")

    def manifest(self) -> dict[str, object]:
        return {"name": self.name, "expert_config": asdict(self.expert_config)}


def default_steering_branches() -> tuple[SteeringBranchCandidate, ...]:
    """Small deterministic set of physically distinct first-shot proposals."""

    baseline = SteeringExpertConfig()
    return (
        SteeringBranchCandidate("baseline", baseline),
        SteeringBranchCandidate(
            "close-contact", replace(baseline, impact_side_sizes=0.25)
        ),
        SteeringBranchCandidate(
            "wide-contact", replace(baseline, impact_side_sizes=0.75)
        ),
        SteeringBranchCandidate(
            "lower-contact", replace(baseline, impact_below_sizes=1.0)
        ),
        SteeringBranchCandidate(
            "rotten-destination-disabled",
            replace(baseline, enable_rotten_matching=False),
        ),
    )


@dataclass(frozen=True, slots=True)
class ArchiveImprovementConfig:
    """Finite public rollouts used to improve one first steering decision."""

    horizon_ticks: int = 128
    max_elites: int = 32
    wait_candidates: tuple[int, ...] = (4, 8, 16)
    steering_branches: tuple[SteeringBranchCandidate, ...] = field(
        default_factory=default_steering_branches
    )
    continuation_config: SteeringExpertConfig = field(
        default_factory=SteeringExpertConfig
    )

    def __post_init__(self) -> None:
        if (
            type(self.horizon_ticks) is not int
            or self.horizon_ticks < 2
            or type(self.max_elites) is not int
            or self.max_elites < 1
        ):
            raise ValueError("archive horizon and elite limit must be positive")
        if (
            not self.wait_candidates
            or any(type(value) is not int or value < 1 for value in self.wait_candidates)
            or tuple(sorted(set(self.wait_candidates))) != self.wait_candidates
            or self.wait_candidates[-1] > self.horizon_ticks
        ):
            raise ValueError(
                "archive waits must be unique increasing positive values within the horizon"
            )
        if (
            not self.steering_branches
            or any(
                not isinstance(value, SteeringBranchCandidate)
                for value in self.steering_branches
            )
        ):
            raise ValueError("at least one steering branch is required")
        names = tuple(value.name for value in self.steering_branches)
        if len(names) != len(set(names)):
            raise ValueError("steering branch names must be unique")
        if "baseline" not in names:
            raise ValueError("archive branches require a baseline incumbent")
        if not isinstance(self.continuation_config, SteeringExpertConfig):
            raise TypeError("continuation config must be a SteeringExpertConfig")

    @property
    def branch_count_per_elite(self) -> int:
        return len(self.steering_branches) + len(self.wait_candidates)

    def manifest(self) -> dict[str, object]:
        return {
            "horizon_ticks": self.horizon_ticks,
            "max_elites": self.max_elites,
            "selection_rule": (
                "maximize_score_only_among_branches_nondecreasing_baseline_"
                "alive_survival_and_final_gauge"
            ),
            "wait_candidates": list(self.wait_candidates),
            "steering_branches": [
                value.manifest() for value in self.steering_branches
            ],
            "continuation_config": asdict(self.continuation_config),
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class ArchiveBranchOutcome:
    """Public terminal measurements for one deterministic branch."""

    candidate_name: str
    candidate_ordinal: int
    first_decision: SteeringDecision
    score_gain: int
    alive: bool
    survival_ticks: int
    final_gauge: int
    clear_gain: int
    highest_chain_gain: int
    invalid_actions: int

    @property
    def selectable(self) -> bool:
        return self.invalid_actions == 0

    def survival_nondominated_by(self, incumbent: ArchiveBranchOutcome) -> bool:
        """Require every public survival measure to match or beat baseline."""

        return (
            int(self.alive) >= int(incumbent.alive)
            and self.survival_ticks >= incumbent.survival_ticks
            and self.final_gauge >= incumbent.final_gauge
        )

    @property
    def objective(self) -> tuple[int, int, int, int, int, int]:
        return (
            self.score_gain,
            self.survival_ticks,
            int(self.alive),
            self.final_gauge,
            self.clear_gain,
            self.highest_chain_gain,
        )

    def manifest(self) -> dict[str, object]:
        return {
            "candidate_name": self.candidate_name,
            "candidate_ordinal": self.candidate_ordinal,
            "decision": _decision_manifest(self.first_decision),
            "score_gain": self.score_gain,
            "alive": self.alive,
            "survival_ticks": self.survival_ticks,
            "final_gauge": self.final_gauge,
            "clear_gain": self.clear_gain,
            "highest_chain_gain": self.highest_chain_gain,
            "invalid_actions": self.invalid_actions,
            "objective": list(self.objective),
        }


@dataclass(frozen=True, slots=True)
class ArchiveImprovementSelection:
    elite_snapshot_sha256: str
    elite_trajectory_identity: str
    incumbent_candidate: str | None
    selected_candidate: str | None
    strictly_improved: bool
    score_gain: int
    score_advantage_over_incumbent: int
    branch_count: int
    selectable_branch_count: int
    emitted_example: bool
    provenance_sha256: str | None
    outcomes: tuple[ArchiveBranchOutcome, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "elite_snapshot_sha256": self.elite_snapshot_sha256,
            "elite_trajectory_identity": self.elite_trajectory_identity,
            "incumbent_candidate": self.incumbent_candidate,
            "selected_candidate": self.selected_candidate,
            "strictly_improved": self.strictly_improved,
            "score_gain": self.score_gain,
            "score_advantage_over_incumbent": (
                self.score_advantage_over_incumbent
            ),
            "branch_count": self.branch_count,
            "selectable_branch_count": self.selectable_branch_count,
            "emitted_example": self.emitted_example,
            "provenance_sha256": self.provenance_sha256,
            "outcomes": [value.manifest() for value in self.outcomes],
        }


@dataclass(frozen=True, slots=True)
class ArchiveImprovementReport:
    binding: ArchiveImprovementBinding
    config: ArchiveImprovementConfig
    action_spec_sha256: str
    pointer_spec_sha256: str
    observation_schema_sha256: str
    collection_identity_sha256: str
    archive_elite_count: int
    evaluated_elite_count: int
    branch_count: int
    selectable_branch_count: int
    strict_improvement_count: int
    emitted_example_count: int
    positive_score_gain_count: int
    total_selected_score_gain: int
    maximum_selected_score_gain: int
    total_score_advantage_over_incumbent: int
    maximum_score_advantage_over_incumbent: int
    selections: tuple[ArchiveImprovementSelection, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "binding": self.binding.manifest(),
            "binding_sha256": self.binding.sha256,
            "config": self.config.manifest(),
            "config_sha256": self.config.sha256,
            "action_spec_sha256": self.action_spec_sha256,
            "pointer_spec_sha256": self.pointer_spec_sha256,
            "observation_schema_sha256": self.observation_schema_sha256,
            "collection_identity_sha256": self.collection_identity_sha256,
            "archive_elite_count": self.archive_elite_count,
            "evaluated_elite_count": self.evaluated_elite_count,
            "branch_count": self.branch_count,
            "selectable_branch_count": self.selectable_branch_count,
            "strict_improvement_count": self.strict_improvement_count,
            "emitted_example_count": self.emitted_example_count,
            "positive_score_gain_count": self.positive_score_gain_count,
            "total_selected_score_gain": self.total_selected_score_gain,
            "maximum_selected_score_gain": self.maximum_selected_score_gain,
            "total_score_advantage_over_incumbent": (
                self.total_score_advantage_over_incumbent
            ),
            "maximum_score_advantage_over_incumbent": (
                self.maximum_score_advantage_over_incumbent
            ),
            "selections": [value.manifest() for value in self.selections],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


@dataclass(frozen=True, slots=True)
class ArchiveImprovementResult:
    examples: tuple[SteeringExample, ...]
    report: ArchiveImprovementReport


def _decision_manifest(decision: SteeringDecision) -> dict[str, object]:
    return {
        "kind": int(decision.action.kind),
        "wait_ticks": int(decision.action.wait_ticks),
        "x_norm": float(decision.action.x_norm),
        "y_norm": float(decision.action.y_norm),
        "intent": decision.intent.value,
        "source_body_id": decision.source_body_id,
        "destination_body_id": decision.destination_body_id,
        "destination_chain_id": decision.destination_chain_id,
        "impact_x_sizes": float(decision.impact_x_sizes),
        "impact_y_sizes": float(decision.impact_y_sizes),
        "reason": decision.reason,
    }


def _verify_restored_elite(
    archive: StrategicArchive,
    elite: ArchiveElite,
    observation: Mapping[str, Any],
) -> StrategicFeatures:
    features = extract_strategic_features(observation)
    if (
        archive_cell_key(features, archive.cell_config) != elite.cell
        or features.raw_score != elite.raw_score
        or features.tick != elite.survival_ticks
        or features.alive != elite.alive
        or features.qualifying_clears != elite.qualifying_clears
        or features.highest_chain != elite.highest_chain
        or features.gauge != elite.gauge
    ):
        raise RuntimeError("restored public state does not match its archive elite")
    return features


def _limited_primitives(
    decision: SteeringDecision,
    action_spec: ActionSpec,
    remaining: int,
) -> tuple[Action, ...]:
    primitives = decision.primitive_actions(action_spec)
    if decision.is_shot and remaining < len(primitives):
        return (Action.wait(remaining),)
    output: list[Action] = []
    budget = remaining
    for action in primitives:
        if budget <= 0:
            break
        kind = ActionKind.parse(action.kind)
        duration = int(action.wait_ticks) if kind is ActionKind.WAIT else 1
        if duration > budget:
            output.append(Action.wait(budget))
            budget = 0
        else:
            output.append(action)
            budget -= duration
    return tuple(output)


def _rollout(
    env: Any,
    initial: Mapping[str, Any],
    first: SteeringDecision,
    *,
    horizon_ticks: int,
    continuation_config: SteeringExpertConfig,
    action_spec: ActionSpec,
) -> tuple[Mapping[str, Any], int]:
    current = initial
    start_tick = int(initial.get("tick", 0))
    invalid_actions = 0
    continuation = ClosedLoopSteeringExpert(
        config=continuation_config, action_spec=action_spec
    )
    continuation.reset(0)
    decision: SteeringDecision | None = first
    first_executed = False
    cooldown_until = (
        start_tick + continuation_config.observe_ticks if first.is_shot else start_tick
    )
    while (
        int(current.get("tick", 0)) - start_tick < horizon_ticks
        and not bool(current.get("terminated", False))
        and not bool(current.get("truncated", False))
    ):
        remaining = horizon_ticks - (int(current.get("tick", 0)) - start_tick)
        if first_executed and int(current.get("tick", 0)) < cooldown_until:
            desired = min(
                remaining, cooldown_until - int(current.get("tick", 0))
            )
            eligible = tuple(
                ticks for ticks in action_spec.wait_choices if ticks <= desired
            )
            decision = SteeringDecision(
                SemanticAction.wait(
                    max(eligible) if eligible else min(action_spec.wait_choices)
                ),
                SteeringIntent.WAIT,
                reason="observe the archive candidate shot before continuation",
            )
        elif first_executed:
            decision = continuation.predict(current)
        assert decision is not None
        for action in _limited_primitives(decision, action_spec, remaining):
            previous_tick = int(current.get("tick", 0))
            current, _reward, terminated, truncated, info = env.step(action)
            if int(current.get("tick", 0)) <= previous_tick:
                raise RuntimeError("archive branch action did not advance public time")
            if (
                bool(current.get("terminated", False)) != bool(terminated)
                or bool(current.get("truncated", False)) != bool(truncated)
            ):
                raise RuntimeError(
                    "archive branch transition flags disagree with its observation"
                )
            event_invalid = sum(
                isinstance(event, Mapping)
                and (
                    event.get("kind") == int(EventKind.INVALID_ACTION)
                    or event.get("kind_name") == "invalid_action"
                )
                for event in info.get("events", ())
            )
            invalid_actions += max(
                int(bool(info.get("invalid_action", False))), event_invalid
            )
            if terminated or truncated:
                break
        first_executed = True
        elapsed = int(current.get("tick", 0)) - start_tick
        if elapsed < 0 or elapsed > horizon_ticks:
            raise RuntimeError("archive branch violated its fixed public horizon")
        if (
            elapsed >= horizon_ticks
            or bool(current.get("terminated", False))
            or bool(current.get("truncated", False))
        ):
            break
    return current, invalid_actions


def _candidate_decisions(
    observation: Mapping[str, Any],
    config: ArchiveImprovementConfig,
    action_spec: ActionSpec,
) -> tuple[tuple[str, SteeringDecision], ...]:
    candidates: list[tuple[str, SteeringDecision]] = []
    for branch in config.steering_branches:
        expert = ClosedLoopSteeringExpert(
            config=branch.expert_config, action_spec=action_spec
        )
        expert.reset(0)
        candidates.append((f"steering/{branch.name}", expert.predict(observation)))
    candidates.extend(
        (
            f"wait/{ticks}",
            SteeringDecision(
                action_spec.validate(SemanticAction.wait(ticks)),
                SteeringIntent.WAIT,
                reason="deliberate archive branch restraint",
            ),
        )
        for ticks in config.wait_candidates
    )
    return tuple(candidates)


def collect_archive_improvement(
    archive: StrategicArchive,
    env_factory: EnvironmentFactory,
    *,
    binding: ArchiveImprovementBinding,
    config: ArchiveImprovementConfig | None = None,
    action_spec: ActionSpec | None = None,
    pointer_spec: PointerActionSpec | None = None,
    encoder: TeacherStateEncoder | None = None,
) -> ArchiveImprovementResult:
    """Branch from archive elites and return deterministic steering labels."""

    if not isinstance(archive, StrategicArchive):
        raise TypeError("archive must be a StrategicArchive")
    if not callable(env_factory):
        raise TypeError("environment factory must be callable")
    if not isinstance(binding, ArchiveImprovementBinding):
        raise TypeError("binding must be an ArchiveImprovementBinding")
    binding.verify(archive)
    resolved = ArchiveImprovementConfig() if config is None else config
    resolved_action = ActionSpec() if action_spec is None else action_spec
    resolved_pointer = PointerActionSpec() if pointer_spec is None else pointer_spec
    resolved_encoder = TeacherStateEncoder() if encoder is None else encoder
    for ticks in resolved.wait_candidates:
        resolved_action.validate(SemanticAction.wait(ticks))
        if ticks not in resolved_pointer.wait_choices:
            raise ValueError(
                "archive wait is absent from the steering supervision vocabulary"
            )
    collection_identity_sha256 = _canonical_sha256(
        {
            "binding_sha256": binding.sha256,
            "improvement_config_sha256": resolved.sha256,
            "action_spec_sha256": resolved_action.sha256,
            "pointer_spec_sha256": resolved_pointer.sha256,
            "observation_schema_sha256": resolved_encoder.schema.sha256,
        }
    )

    examples: list[SteeringExample] = []
    selections: list[ArchiveImprovementSelection] = []
    elites = tuple(
        sorted(
            archive,
            key=lambda value: (
                -value.raw_score,
                -value.survival_ticks,
                -int(value.alive),
                value.cell.sha256,
                value.snapshot_sha256,
            ),
        )
    )[: resolved.max_elites]
    for elite in elites:
        if not elite.trajectory_identity.endswith(":pre-shot"):
            raise ValueError(
                "archive elite was not captured at a pre-shot safe boundary"
            )
        env = env_factory()
        if getattr(env, "physics_backend", None) != "portable":
            close = getattr(env, "close", None)
            if callable(close):
                close()
            raise ValueError("archive improvement requires the portable backend")
        try:
            _verify_environment_identity(env, binding)
        except Exception:
            close = getattr(env, "close", None)
            if callable(close):
                close()
            raise
        outcomes: list[ArchiveBranchOutcome] = []
        initial: Mapping[str, Any] | None = None
        label_observation: Mapping[str, Any] | None = None
        try:
            initial = env.restore_state(elite.snapshot)
            if not isinstance(initial, Mapping):
                raise TypeError("portable restore_state must return a public mapping")
            before = _verify_restored_elite(archive, elite, initial)
            label_observation = copy.deepcopy(initial)
            for ordinal, (name, decision) in enumerate(
                _candidate_decisions(initial, resolved, resolved_action)
            ):
                restored = env.restore_state(elite.snapshot)
                if not isinstance(restored, Mapping):
                    raise TypeError(
                        "portable restore_state must return a public mapping"
                    )
                _verify_restored_elite(archive, elite, restored)
                final, invalid = _rollout(
                    env,
                    restored,
                    decision,
                    horizon_ticks=resolved.horizon_ticks,
                    continuation_config=resolved.continuation_config,
                    action_spec=resolved_action,
                )
                after = extract_strategic_features(final)
                elapsed = after.tick - before.tick
                outcomes.append(
                    ArchiveBranchOutcome(
                        name,
                        ordinal,
                        decision,
                        after.raw_score - before.raw_score,
                        after.alive,
                        elapsed,
                        after.gauge,
                        after.qualifying_clears - before.qualifying_clears,
                        after.highest_chain - before.highest_chain,
                        invalid,
                    )
                )
        finally:
            try:
                if initial is not None:
                    env.restore_state(elite.snapshot)
            finally:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

        selectable = tuple(value for value in outcomes if value.selectable)
        incumbent = next(
            (
                value
                for value in selectable
                if value.candidate_name == "steering/baseline"
            ),
            None,
        )
        eligible = (
            selectable
            if incumbent is None
            else tuple(
                value
                for value in selectable
                if value.survival_nondominated_by(incumbent)
            )
        )
        winner = (
            None
            if not eligible
            else max(
                eligible,
                key=lambda value: (value.objective, -value.candidate_ordinal),
            )
        )
        strictly_improved = bool(
            winner is not None
            and incumbent is not None
            and winner.objective > incumbent.objective
        )
        score_advantage = (
            0
            if winner is None or incumbent is None
            else winner.score_gain - incumbent.score_gain
        )
        provenance: str | None = None
        example: SteeringExample | None = None
        if winner is not None:
            assert label_observation is not None
            provenance = _canonical_sha256(
                {
                    "collection_identity_sha256": collection_identity_sha256,
                    "elite_snapshot_sha256": elite.snapshot_sha256,
                    "elite_trajectory_identity": elite.trajectory_identity,
                    "incumbent": (
                        None if incumbent is None else incumbent.manifest()
                    ),
                    "winner": winner.manifest(),
                    "strictly_improved": strictly_improved,
                }
            )
            if strictly_improved:
                example = steering_example_from_decision(
                    label_observation,
                    winner.first_decision,
                    episode_identity=(
                        f"archive-improvement/{binding.sha256[:16]}/"
                        f"{elite.snapshot_sha256[:16]}"
                    ),
                    provenance_sha256=provenance,
                    encoder=resolved_encoder,
                    pointer_spec=resolved_pointer,
                    action_spec=resolved_action,
                    require_representable_template=False,
                )
                if example is not None:
                    examples.append(example)
        selections.append(
            ArchiveImprovementSelection(
                elite_snapshot_sha256=elite.snapshot_sha256,
                elite_trajectory_identity=elite.trajectory_identity,
                incumbent_candidate=(
                    None if incumbent is None else incumbent.candidate_name
                ),
                selected_candidate=(
                    None if winner is None else winner.candidate_name
                ),
                strictly_improved=strictly_improved,
                score_gain=0 if winner is None else winner.score_gain,
                score_advantage_over_incumbent=score_advantage,
                branch_count=len(outcomes),
                selectable_branch_count=len(selectable),
                emitted_example=example is not None,
                provenance_sha256=provenance,
                outcomes=tuple(outcomes),
            )
        )

    score_gains = [selection.score_gain for selection in selections]
    score_advantages = [
        selection.score_advantage_over_incumbent for selection in selections
    ]
    report = ArchiveImprovementReport(
        binding=binding,
        config=resolved,
        action_spec_sha256=resolved_action.sha256,
        pointer_spec_sha256=resolved_pointer.sha256,
        observation_schema_sha256=resolved_encoder.schema.sha256,
        collection_identity_sha256=collection_identity_sha256,
        archive_elite_count=len(archive),
        evaluated_elite_count=len(selections),
        branch_count=sum(value.branch_count for value in selections),
        selectable_branch_count=sum(
            value.selectable_branch_count for value in selections
        ),
        strict_improvement_count=sum(
            value.strictly_improved for value in selections
        ),
        emitted_example_count=len(examples),
        positive_score_gain_count=sum(value > 0 for value in score_gains),
        total_selected_score_gain=sum(score_gains),
        maximum_selected_score_gain=max(score_gains, default=0),
        total_score_advantage_over_incumbent=sum(score_advantages),
        maximum_score_advantage_over_incumbent=max(
            score_advantages, default=0
        ),
        selections=tuple(selections),
    )
    return ArchiveImprovementResult(tuple(examples), report)


__all__ = [
    "ArchiveBranchOutcome",
    "ArchiveImprovementBinding",
    "ArchiveImprovementConfig",
    "ArchiveImprovementReport",
    "ArchiveImprovementResult",
    "ArchiveImprovementSelection",
    "SteeringBranchCandidate",
    "collect_archive_improvement",
    "default_steering_branches",
]
