"""Raw-score-only fixed-cell evaluation for R3b policies and baselines."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any, Mapping

import torch
from torch import Tensor

from irisu_env import Action, ActionKind
from irisu_env.policies import (
    DirectMatcherPolicy,
    ImminentRotHazardPolicy,
    LongWaitPolicy,
    MatcherShotPolicy,
    RandomPolicy,
    SideEjectorPolicy,
)

from .actions import ActionSpec, SemanticAction, SemanticActionKind
from .collector import model_state_sha256
from .curriculum import SnapshotBlobStore
from .encoding import EncodedBatch
from .models import RecurrentActorCritic


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class EvaluationSuite:
    suite_id: str
    split: str
    snapshot_ids: tuple[str, ...]
    repetitions: int
    policy_seed: int
    max_decisions: int
    max_simulated_ticks: int
    runtime_identity_sha256: str
    assignment_sha256: str
    library_sha256: str
    snapshot_store_sha256: str
    action_spec_sha256: str
    version: str = "r3b-evaluation-suite-v2"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-evaluation-suite-v2"
            or not self.suite_id
            or not self.suite_id.isascii()
            or self.split not in {"calibration", "validation", "test"}
            or not isinstance(self.snapshot_ids, tuple)
            or not self.snapshot_ids
            or len(set(self.snapshot_ids)) != len(self.snapshot_ids)
            or any(not value for value in self.snapshot_ids)
        ):
            raise ValueError("evaluation suite identity is invalid")
        for name in ("repetitions", "max_decisions", "max_simulated_ticks"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.policy_seed, bool)
            or not isinstance(self.policy_seed, Integral)
            or not 0 <= self.policy_seed < 2**64
            or not _is_sha256(self.runtime_identity_sha256)
            or any(
                not _is_sha256(value) or value == "0" * 64
                for value in (
                    self.runtime_identity_sha256,
                    self.assignment_sha256,
                    self.library_sha256,
                    self.snapshot_store_sha256,
                    self.action_spec_sha256,
                )
            )
        ):
            raise ValueError("evaluation suite seed or identity is invalid")

    def manifest(self) -> dict[str, object]:
        value = asdict(self)
        value["snapshot_ids"] = list(self.snapshot_ids)
        return value

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    def episode_seed(self, snapshot_id: str, repetition: int) -> int:
        if (
            snapshot_id not in self.snapshot_ids
            or not 0 <= repetition < self.repetitions
        ):
            raise ValueError("evaluation cell is outside the suite")
        payload = (
            f"{self.sha256}:{self.policy_seed}:{snapshot_id}:{repetition}"
        ).encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    snapshot_id: str
    repetition: int
    policy_seed: int
    initial_score: int
    final_score: int
    raw_score: int
    elapsed_ticks: int
    decisions: int
    terminated: bool
    truncated: bool
    invalid_actions: int
    minimum_gauge: int
    final_gauge: int

    def __post_init__(self) -> None:
        integer_fields = (
            self.repetition,
            self.policy_seed,
            self.initial_score,
            self.final_score,
            self.raw_score,
            self.elapsed_ticks,
            self.decisions,
            self.invalid_actions,
            self.minimum_gauge,
            self.final_gauge,
        )
        if (
            not self.snapshot_id
            or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in integer_fields
            )
            or self.repetition < 0
            or not 0 <= self.policy_seed < 2**64
            or self.elapsed_ticks < 0
            or self.decisions < 0
            or self.invalid_actions < 0
            or self.minimum_gauge < 0
            or self.final_gauge < 0
            or self.raw_score != self.final_score - self.initial_score
            or not isinstance(self.terminated, bool)
            or not isinstance(self.truncated, bool)
        ):
            raise ValueError("evaluation episode metrics are malformed")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    suite_sha256: str
    policy_sha256: str
    evaluator_sha256: str
    backend_identity_sha256: str
    execution_identity_sha256: str
    episodes: tuple[EpisodeMetrics, ...]
    version: str = "r3b-evaluation-report-v2"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-evaluation-report-v2"
            or not isinstance(self.episodes, tuple)
            or not all(
                _is_sha256(value)
                for value in (
                    self.suite_sha256,
                    self.policy_sha256,
                    self.evaluator_sha256,
                    self.backend_identity_sha256,
                    self.execution_identity_sha256,
                )
            )
            or self.backend_identity_sha256 == "0" * 64
            or self.execution_identity_sha256 == "0" * 64
            or not self.episodes
            or len({(value.snapshot_id, value.repetition) for value in self.episodes})
            != len(self.episodes)
        ):
            raise ValueError("evaluation report identity or cells are malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "suite_sha256": self.suite_sha256,
            "policy_sha256": self.policy_sha256,
            "evaluator_sha256": self.evaluator_sha256,
            "backend_identity_sha256": self.backend_identity_sha256,
            "execution_identity_sha256": self.execution_identity_sha256,
            "episodes": [asdict(value) for value in self.episodes],
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    @property
    def episode_content_sha256(self) -> str:
        return _canonical_sha256([asdict(value) for value in self.episodes])


@dataclass(frozen=True, slots=True)
class ScriptedBaselineSpec:
    baseline_id: str
    parameters: tuple[tuple[str, int | float], ...] = ()
    version: str = "r3b-scripted-baseline-v1"

    def __post_init__(self) -> None:
        supported = {
            "no_action_long_wait",
            "seeded_legal_random",
            "matcher_shot_policy",
            "scripted_direct_matcher",
            "scripted_side_ejector",
            "scripted_imminent_rot_hazard",
        }
        if (
            self.version != "r3b-scripted-baseline-v1"
            or self.baseline_id not in supported
            or not isinstance(self.parameters, tuple)
            or any(
                not isinstance(item, tuple) or len(item) != 2
                for item in self.parameters
            )
        ):
            raise ValueError("unknown scripted baseline")
        names = tuple(name for name, _ in self.parameters)
        if len(names) != len(set(names)) or any(not name for name in names):
            raise ValueError("baseline parameter names must be unique")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for _, value in self.parameters
        ):
            raise ValueError("baseline parameters must be finite numbers")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "baseline_id": self.baseline_id,
            "parameters": {name: value for name, value in self.parameters},
        }

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    def build(self, seed: int) -> Any:
        parameters = dict(self.parameters)
        if self.baseline_id == "no_action_long_wait":
            policy = LongWaitPolicy(**parameters)
        elif self.baseline_id == "seeded_legal_random":
            policy = RandomPolicy(seed=seed, **parameters)
        elif self.baseline_id == "matcher_shot_policy":
            policy = MatcherShotPolicy(**parameters)
        elif self.baseline_id == "scripted_direct_matcher":
            policy = DirectMatcherPolicy(**parameters)
        elif self.baseline_id == "scripted_side_ejector":
            policy = SideEjectorPolicy(**parameters)
        else:
            policy = ImminentRotHazardPolicy(**parameters)
        policy.reset(seed)
        return policy


def _mapping(observation: object) -> Mapping[str, Any]:
    if isinstance(observation, Mapping):
        return observation
    converter = getattr(observation, "to_dict", None)
    if converter is None:
        raise TypeError("scripted evaluation requires a mapping-capable observation")
    value = converter()
    if not isinstance(value, Mapping):
        raise TypeError("observation to_dict() did not return a mapping")
    return value


def semantic_from_native(action: Action, spec: ActionSpec) -> SemanticAction:
    kind = ActionKind.parse(action.kind)
    if kind is ActionKind.WAIT:
        return spec.validate(SemanticAction.wait(int(action.wait_ticks)))
    if kind not in {ActionKind.WEAK_SHOT, ActionKind.STRONG_SHOT}:
        raise ValueError("scripted policy emitted an unsupported simultaneous shot")
    constructor = (
        SemanticAction.weak if kind is ActionKind.WEAK_SHOT else SemanticAction.strong
    )
    return spec.validate(
        constructor(
            float(action.cursor_x) / spec.client_width,
            float(action.cursor_y) / spec.client_height,
        )
    )


class RecurrentSemanticPolicy:
    """Deterministic deployment-style recurrent inference at semantic boundaries."""

    def __init__(self, model: RecurrentActorCritic) -> None:
        self.model = model
        self.device = next(model.parameters()).device
        self._state: Tensor | None = None
        self._reset_before: Tensor | None = None

    def reset(self, lanes: int) -> None:
        if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes <= 0:
            raise ValueError("policy lane count must be positive")
        self._state = self.model.initial_state(lanes).detach()
        self._reset_before = torch.ones(lanes, dtype=torch.bool, device=self.device)

    def act(
        self,
        observation: EncodedBatch,
        kind_mask: Tensor,
        wait_mask: Tensor,
    ) -> tuple[SemanticAction, ...]:
        observation.validate()
        lanes = observation.global_features.shape[0]
        if self._state is None or self._reset_before is None:
            raise RuntimeError("recurrent evaluation policy must be reset")
        if kind_mask.shape != (lanes, 3) or kind_mask.dtype != torch.bool:
            raise ValueError("evaluation kind mask must be boolean [B, 3]")
        expected_wait = (lanes, len(self.model.action_spec.wait_choices))
        if wait_mask.shape != expected_wait or wait_mask.dtype != torch.bool:
            raise ValueError("evaluation wait mask does not match the action schema")
        if not bool(torch.all(kind_mask.any(dim=1))):
            raise ValueError("evaluation kind mask contains an all-masked lane")
        wait_lanes = kind_mask[:, int(SemanticActionKind.WAIT)]
        if bool(torch.any(wait_lanes & ~wait_mask.any(dim=1))):
            raise ValueError("WAIT is enabled without a legal wait duration")
        global_features = (
            torch.from_numpy(observation.global_features).to(self.device).unsqueeze(0)
        )
        body_features = (
            torch.from_numpy(observation.body_features).to(self.device).unsqueeze(0)
        )
        body_mask = torch.from_numpy(observation.body_mask).to(self.device).unsqueeze(0)
        arguments: dict[str, Tensor] = {}
        if self.model.config.critic_condition_features:
            arguments["critic_condition"] = torch.zeros(
                (1, lanes, self.model.config.critic_condition_features),
                dtype=torch.float32,
                device=self.device,
            )
        prior_mode = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(
                    global_features,
                    body_features,
                    body_mask,
                    self._state,
                    reset_before=self._reset_before.unsqueeze(0),
                    **arguments,
                )
        finally:
            self.model.train(prior_mode)
        kind = (
            output.kind_logits[0]
            .masked_fill(~kind_mask.to(self.device), -torch.inf)
            .argmax(-1)
        )
        wait = (
            output.wait_logits[0]
            .masked_fill(~wait_mask.to(self.device), -torch.inf)
            .argmax(-1)
        )
        coordinate_mean = output.coordinate_alpha[0] / (
            output.coordinate_alpha[0] + output.coordinate_beta[0]
        )
        actions = []
        for lane in range(lanes):
            kind_value = int(kind[lane])
            xy = (
                coordinate_mean[lane, kind_value - 1]
                if kind_value > 0
                else torch.zeros(2, device=self.device)
            )
            actions.append(
                self.model.action_spec.decode(
                    kind_value,
                    int(wait[lane]),
                    float(xy[0]),
                    float(xy[1]),
                )
            )
        self._state = output.recurrent_state.detach()
        self._reset_before = torch.zeros(lanes, dtype=torch.bool, device=self.device)
        return tuple(actions)


def evaluate_scripted_baseline(
    simulator: Any,
    store: SnapshotBlobStore,
    suite: EvaluationSuite,
    baseline: ScriptedBaselineSpec,
    *,
    evaluator_sha256: str,
    expected_assignment_sha256: str,
    actual_runtime_identity_sha256: str,
    backend_identity_sha256: str,
    execution_identity_sha256: str,
) -> EvaluationReport:
    """Evaluate fixed snapshot/repetition cells using deployment macro semantics."""

    action_spec = ActionSpec()

    def factory(seed: int):
        policy = baseline.build(seed)
        return lambda observation: semantic_from_native(
            policy.act(_mapping(observation)), action_spec
        )

    return _evaluate_semantic_policy(
        simulator,
        store,
        suite,
        policy_sha256=baseline.sha256,
        evaluator_sha256=evaluator_sha256,
        expected_assignment_sha256=expected_assignment_sha256,
        actual_runtime_identity_sha256=actual_runtime_identity_sha256,
        backend_identity_sha256=backend_identity_sha256,
        execution_identity_sha256=execution_identity_sha256,
        action_spec=action_spec,
        policy_factory=factory,
    )


def evaluate_recurrent_policy(
    simulator: Any,
    store: SnapshotBlobStore,
    suite: EvaluationSuite,
    model: RecurrentActorCritic,
    encoder: Any,
    kind_mask: Tensor,
    wait_mask: Tensor,
    *,
    evaluator_sha256: str,
    expected_assignment_sha256: str,
    actual_runtime_identity_sha256: str,
    backend_identity_sha256: str,
    execution_identity_sha256: str,
) -> EvaluationReport:
    """Evaluate a learned policy on the same fixed cells and macro semantics."""

    if kind_mask.shape != (1, 3) or kind_mask.dtype != torch.bool:
        raise ValueError("recurrent evaluation kind mask must be boolean [1, 3]")
    expected_wait = (1, len(model.action_spec.wait_choices))
    if wait_mask.shape != expected_wait or wait_mask.dtype != torch.bool:
        raise ValueError(
            "recurrent evaluation wait mask disagrees with the action schema"
        )
    if not bool(torch.all(kind_mask.any(dim=1))):
        raise ValueError("recurrent evaluation kind mask is all-masked")
    if bool(kind_mask[0, int(SemanticActionKind.WAIT)]) and not bool(wait_mask.any()):
        raise ValueError("WAIT is enabled without a legal wait duration")
    kind_mask = kind_mask.detach().cpu().clone()
    wait_mask = wait_mask.detach().cpu().clone()

    def factory(seed: int):
        del seed
        policy = RecurrentSemanticPolicy(model)
        policy.reset(1)

        def act(observation: object) -> SemanticAction:
            encoded = encoder.encode((observation,))
            return policy.act(encoded, kind_mask, wait_mask)[0]

        return act

    return _evaluate_semantic_policy(
        simulator,
        store,
        suite,
        policy_sha256=model_state_sha256(model),
        evaluator_sha256=evaluator_sha256,
        expected_assignment_sha256=expected_assignment_sha256,
        actual_runtime_identity_sha256=actual_runtime_identity_sha256,
        backend_identity_sha256=backend_identity_sha256,
        execution_identity_sha256=execution_identity_sha256,
        action_spec=model.action_spec,
        policy_factory=factory,
    )


def _evaluate_semantic_policy(
    simulator: Any,
    store: SnapshotBlobStore,
    suite: EvaluationSuite,
    *,
    policy_sha256: str,
    evaluator_sha256: str,
    expected_assignment_sha256: str,
    actual_runtime_identity_sha256: str,
    backend_identity_sha256: str,
    execution_identity_sha256: str,
    action_spec: ActionSpec,
    policy_factory: Any,
) -> EvaluationReport:
    """Shared fixed-cell evaluator after a policy has entered semantic space."""

    if (
        not _is_sha256(evaluator_sha256)
        or not _is_sha256(policy_sha256)
        or not _is_sha256(expected_assignment_sha256)
        or not _is_sha256(actual_runtime_identity_sha256)
        or not _is_sha256(backend_identity_sha256)
        or not _is_sha256(execution_identity_sha256)
        or "0" * 64
        in (
            evaluator_sha256,
            policy_sha256,
            expected_assignment_sha256,
            actual_runtime_identity_sha256,
            backend_identity_sha256,
            execution_identity_sha256,
        )
    ):
        raise ValueError("policy and evaluator identities must be lowercase SHA-256")
    if actual_runtime_identity_sha256 != suite.runtime_identity_sha256:
        raise ValueError("evaluated simulator runtime identity mismatch")
    if suite.assignment_sha256 != expected_assignment_sha256:
        raise ValueError("evaluation suite assignment identity mismatch")
    if suite.library_sha256 != store.library.sha256:
        raise ValueError("evaluation suite snapshot-library identity mismatch")
    if suite.snapshot_store_sha256 != store.sha256:
        raise ValueError("evaluation suite snapshot-store identity mismatch")
    if suite.action_spec_sha256 != action_spec.sha256:
        raise ValueError("evaluation suite action identity mismatch")
    episodes: list[EpisodeMetrics] = []
    for snapshot_id in suite.snapshot_ids:
        recipe = store.library[snapshot_id]
        if recipe.split != suite.split:
            raise ValueError("evaluation snapshot belongs to the wrong split")
        if recipe.runtime_identity_sha256 != suite.runtime_identity_sha256:
            raise ValueError("evaluation snapshot runtime identity mismatch")
        for repetition in range(suite.repetitions):
            observation = simulator.restore_state(store[snapshot_id])
            if int(simulator.config_hash()) != recipe.config_hash:
                raise ValueError("evaluated simulator config hash mismatch")
            if _canonical_sha256(simulator.config()) != recipe.config_sha256:
                raise ValueError("evaluated simulator canonical config mismatch")
            if int(simulator.state_hash()) != recipe.expected_state_hash:
                raise ValueError("evaluation snapshot state hash mismatch")
            restored = _mapping(observation)
            gauge = restored.get("gauge")
            gauge_max = restored.get("gauge_max")
            if (
                int(restored.get("tick", -1)) != recipe.expected_tick
                or int(restored.get("score", -1)) != recipe.expected_score
                or bool(restored.get("terminated", False))
                or bool(restored.get("truncated", False))
                or isinstance(gauge, bool)
                or not isinstance(gauge, Integral)
                or isinstance(gauge_max, bool)
                or not isinstance(gauge_max, Integral)
                or gauge_max <= 0
                or not 0 <= gauge <= gauge_max
            ):
                raise ValueError(
                    "evaluation snapshot is not the declared live boundary"
                )
            seed = suite.episode_seed(snapshot_id, repetition)
            act = policy_factory(seed)
            if not callable(act):
                raise TypeError("evaluation policy factory must return a callable")
            initial_tick = int(_mapping(observation)["tick"])
            initial_score = int(_mapping(observation)["score"])
            minimum_gauge = int(_mapping(observation)["gauge"])
            decisions = 0
            invalid_actions = 0
            accumulated_reward = 0
            terminated = False
            truncated = False
            while (
                not terminated
                and not truncated
                and decisions < suite.max_decisions
                and int(_mapping(observation)["tick"]) - initial_tick
                < suite.max_simulated_ticks
            ):
                semantic = action_spec.validate(act(observation))
                elapsed = int(_mapping(observation)["tick"]) - initial_tick
                remaining = suite.max_simulated_ticks - elapsed
                if remaining <= 0:
                    break
                first = (
                    Action.wait(min(semantic.wait_ticks, remaining))
                    if semantic.kind is SemanticActionKind.WAIT
                    else action_spec.press(semantic)
                )
                primitives = [first]
                if semantic.kind is not SemanticActionKind.WAIT:
                    primitives.append(action_spec.release())
                for primitive in primitives:
                    elapsed = int(_mapping(observation)["tick"]) - initial_tick
                    remaining = suite.max_simulated_ticks - elapsed
                    if remaining <= 0:
                        break
                    if (
                        ActionKind.parse(primitive.kind) is ActionKind.WAIT
                        and primitive.wait_ticks > remaining
                    ):
                        primitive = Action.wait(remaining)
                    observation, reward, terminated, truncated, info = simulator.step(
                        primitive
                    )
                    accumulated_reward += int(reward)
                    invalid_actions += int(bool(info.get("invalid_action", False)))
                    minimum_gauge = min(
                        minimum_gauge, int(_mapping(observation)["gauge"])
                    )
                    if terminated or truncated:
                        break
                decisions += 1
            budget_cut = not terminated and not truncated
            final = _mapping(observation)
            final_score = int(final["score"])
            if int(final["tick"]) - initial_tick > suite.max_simulated_ticks:
                raise ValueError("evaluation exceeded its simulated-tick horizon")
            if accumulated_reward != final_score - initial_score:
                raise ValueError("evaluation raw reward does not equal score delta")
            episodes.append(
                EpisodeMetrics(
                    snapshot_id,
                    repetition,
                    seed,
                    initial_score,
                    final_score,
                    final_score - initial_score,
                    int(final["tick"]) - initial_tick,
                    decisions,
                    terminated,
                    truncated or budget_cut,
                    invalid_actions,
                    minimum_gauge,
                    int(final["gauge"]),
                )
            )
    return EvaluationReport(
        suite.sha256,
        policy_sha256,
        evaluator_sha256,
        backend_identity_sha256,
        execution_identity_sha256,
        tuple(episodes),
    )


def build_baseline_evidence(
    baseline: ScriptedBaselineSpec,
    suite: EvaluationSuite,
    report: EvaluationReport,
    replay_report: EvaluationReport,
    exact_backend_report: EvaluationReport,
):
    """Build acceptance evidence only from matching evaluated report artifacts."""

    from .r3b_experiments import BaselineEvidence

    reports = (report, replay_report, exact_backend_report)
    if any(value.suite_sha256 != suite.sha256 for value in reports):
        raise ValueError("baseline reports disagree with the evaluation suite")
    if any(value.policy_sha256 != baseline.sha256 for value in reports):
        raise ValueError("baseline reports disagree with the scripted policy")
    if (
        report.backend_identity_sha256 != replay_report.backend_identity_sha256
        or exact_backend_report.backend_identity_sha256
        == report.backend_identity_sha256
        or len({value.execution_identity_sha256 for value in reports}) != 3
        or len({value.sha256 for value in reports}) != 3
    ):
        raise ValueError("baseline evidence lacks independent backend executions")
    expected_cells = {
        (snapshot_id, repetition)
        for snapshot_id in suite.snapshot_ids
        for repetition in range(suite.repetitions)
    }
    for value in reports:
        if {
            (episode.snapshot_id, episode.repetition) for episode in value.episodes
        } != expected_cells or any(
            episode.policy_seed
            != suite.episode_seed(episode.snapshot_id, episode.repetition)
            or episode.decisions > suite.max_decisions
            or episode.elapsed_ticks > suite.max_simulated_ticks
            or not (episode.terminated or episode.truncated)
            for episode in value.episodes
        ):
            raise ValueError("baseline report cells do not exactly match the suite")
    if report.episode_content_sha256 != replay_report.episode_content_sha256:
        raise ValueError("baseline replay is not deterministic")
    if report.episode_content_sha256 != exact_backend_report.episode_content_sha256:
        raise ValueError("portable and exact baseline episode metrics differ")
    episodes = len(report.episodes)
    return BaselineEvidence(
        baseline.baseline_id,
        "complete",
        episodes,
        sum(value.raw_score for value in report.episodes) / episodes,
        sum(value.invalid_actions for value in report.episodes),
        suite.sha256,
        report.sha256,
        replay_report.sha256,
        exact_backend_report.sha256,
        report.backend_identity_sha256,
        exact_backend_report.backend_identity_sha256,
        report.episode_content_sha256,
    )


@dataclass(frozen=True, slots=True)
class BaselineArtifactBundle:
    """Typed primary/replay/exact artifacts consumed by sealed confirmation."""

    baseline: ScriptedBaselineSpec
    suite: EvaluationSuite
    report: EvaluationReport
    replay_report: EvaluationReport
    exact_backend_report: EvaluationReport

    def evidence(self):
        return build_baseline_evidence(
            self.baseline,
            self.suite,
            self.report,
            self.replay_report,
            self.exact_backend_report,
        )
