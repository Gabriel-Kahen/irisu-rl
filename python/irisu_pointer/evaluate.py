"""Fail-closed, development-only full-game pointer evaluation."""

from __future__ import annotations

import hashlib
import json
import math
import re
import stat
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from irisu_env import Action, EventKind, IrisuEnv
from irisu_rl.schema import ACTOR_VISION_V1, TEACHER_V1

from .action import PointerActionSpec
from .experts import PointerExpertDecision
from .search import lower_expert_decision


UNSEEN_DEVELOPMENT_SEEDS = (
    0x13579BDF,
    0x2468ACE0,
    0x31415926,
    0x5A17C0DE,
    0x6C8E9CF1,
    0x7B1D3F59,
    0x8D2E4A60,
    0xA5C31E79,
)
_FORBIDDEN = re.compile(r"(?:^|[^a-z0-9])(?:sealed|test)(?:$|[^a-z0-9])")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DevelopmentEvaluationError(RuntimeError):
    """Raised when evidence identity or complete-game execution fails closed."""


class EvaluatedPolicy(Protocol):
    artifact_sha256: str
    schema_sha256: str
    pointer_action_sha256: str

    def reset(self, seed: int = 0) -> None: ...

    def act(
        self, observation: Mapping[str, Any]
    ) -> PointerExpertDecision | Action | tuple[Action, ...]: ...


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def identity_sha256(manifest: Mapping[str, Any]) -> str:
    """Return the canonical identity used for runner provenance binding."""

    if not isinstance(manifest, Mapping):
        raise TypeError("identity manifest must be a mapping")
    return _canonical_sha256(dict(manifest))


def _reject_prohibited(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a nonempty string")
    if _FORBIDDEN.search(value.lower().replace("\\", "/")):
        raise ValueError(f"{name} must not reference sealed or test material")


def _sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    path: Path
    device: int
    inode: int
    size: int
    mtime_ns: int
    sha256: str


def _snapshot_file(path: Path, expected_sha256: str, *, name: str) -> _FileSnapshot:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise DevelopmentEvaluationError(f"{name} must not be a symbolic link")
    resolved = expanded.resolve(strict=True)
    _reject_prohibited(str(resolved), name=f"{name} path")
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise DevelopmentEvaluationError(f"{name} must be a regular file")
    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
    metadata = resolved.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    ):
        raise DevelopmentEvaluationError(f"{name} changed while hashing")
    if digest != expected_sha256:
        raise DevelopmentEvaluationError(f"{name} SHA-256 mismatch")
    return _FileSnapshot(
        resolved,
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        digest,
    )


def _verify_unchanged(snapshot: _FileSnapshot, *, name: str) -> None:
    current = _snapshot_file(snapshot.path, snapshot.sha256, name=name)
    if current != snapshot:
        raise DevelopmentEvaluationError(f"{name} changed during evaluation")


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    """Expected policy, runtime, schema, action, and runner identities."""

    label: str
    policy_path: Path
    policy_sha256: str
    runtime_path: Path
    runtime_sha256: str
    runner_identity_sha256: str
    schema_sha256: str = TEACHER_V1.sha256
    pointer_action_sha256: str = field(
        default_factory=lambda: PointerActionSpec().sha256
    )

    def __post_init__(self) -> None:
        _reject_prohibited(self.label, name="artifact label")
        object.__setattr__(self, "policy_path", Path(self.policy_path))
        object.__setattr__(self, "runtime_path", Path(self.runtime_path))
        for name in (
            "policy_sha256",
            "runtime_sha256",
            "runner_identity_sha256",
            "schema_sha256",
            "pointer_action_sha256",
        ):
            object.__setattr__(
                self, name, _sha256(getattr(self, name), name=name)
            )
        _reject_prohibited(str(self.policy_path), name="policy path")
        _reject_prohibited(str(self.runtime_path), name="runtime path")
        if self.schema_sha256 not in {
            TEACHER_V1.sha256,
            ACTOR_VISION_V1.sha256,
        }:
            raise ValueError("artifact binding uses an unknown observation schema")
        if self.pointer_action_sha256 != PointerActionSpec().sha256:
            raise ValueError("artifact binding uses an unknown pointer action schema")

    def manifest(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "policy_path": str(self.policy_path),
            "policy_sha256": self.policy_sha256,
            "runtime_path": str(self.runtime_path),
            "runtime_sha256": self.runtime_sha256,
            "runner_identity_sha256": self.runner_identity_sha256,
            "schema_sha256": self.schema_sha256,
            "pointer_action_sha256": self.pointer_action_sha256,
        }


@dataclass(frozen=True, slots=True)
class DevelopmentSuite:
    """A fixed unsealed seed suite and complete-game safety budget."""

    label: str = "r3c-full-game-unseen-development-v1"
    seeds: tuple[int, ...] = UNSEEN_DEVELOPMENT_SEEDS
    config: tuple[tuple[str, Any], ...] = ()
    max_decisions_per_episode: int = 1_000_000

    def __post_init__(self) -> None:
        _reject_prohibited(self.label, name="suite label")
        if (
            not self.seeds
            or len(set(self.seeds)) != len(self.seeds)
            or any(
                isinstance(seed, bool)
                or not isinstance(seed, int)
                or not 0 <= seed <= 0xFFFFFFFF
                for seed in self.seeds
            )
        ):
            raise ValueError("development seeds must be unique uint32 values")
        if (
            isinstance(self.max_decisions_per_episode, bool)
            or not isinstance(self.max_decisions_per_episode, int)
            or self.max_decisions_per_episode < 1
        ):
            raise ValueError("maximum decision count must be positive")
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            for item in self.config
        ):
            raise ValueError("suite config must contain key/value pairs")
        if len({key for key, _ in self.config}) != len(self.config):
            raise ValueError("suite config contains duplicate keys")
        try:
            json.dumps(
                dict(self.config),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("suite config must be canonical JSON data") from exc

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())

    def manifest(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "seeds": list(self.seeds),
            "config": dict(self.config),
            "max_decisions_per_episode": self.max_decisions_per_episode,
            "sealed": False,
            "development_only": True,
        }


DEFAULT_DEVELOPMENT_SUITE = DevelopmentSuite()


@dataclass(frozen=True, slots=True)
class PromotionCriteria:
    """Explicit deterministic gates for another development iteration."""

    minimum_median_raw_score: float = 100_000.0
    minimum_p10_raw_score: float = 25_000.0
    minimum_median_survival_ticks: float = 10_000.0
    minimum_median_unique_hit_pairs: float = 25.0
    minimum_completion_rate: float = 1.0
    maximum_invalid_actions: int = 0

    def __post_init__(self) -> None:
        numeric = (
            self.minimum_median_raw_score,
            self.minimum_p10_raw_score,
            self.minimum_median_survival_ticks,
            self.minimum_median_unique_hit_pairs,
            self.minimum_completion_rate,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
            for value in numeric
        ):
            raise ValueError("promotion thresholds must be finite and nonnegative")
        if not 0.0 <= self.minimum_completion_rate <= 1.0:
            raise ValueError("minimum completion rate must be in [0, 1]")
        if (
            isinstance(self.maximum_invalid_actions, bool)
            or not isinstance(self.maximum_invalid_actions, int)
            or self.maximum_invalid_actions < 0
        ):
            raise ValueError("maximum invalid actions must be nonnegative")

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpisodeOutcome:
    seed: int
    raw_score: int
    survival_ticks: int
    decisions: int
    primitive_actions: int
    valid_actions: int
    invalid_actions: int
    chain_joined: int
    cleared: int
    rotten: int
    max_chain: int
    projectile_hit_events: int
    unique_projectile_hit_pairs: int
    unique_projectiles_hitting: int
    unique_bodies_hit: int
    terminated: bool
    truncated: bool

    def manifest(self) -> dict[str, Any]:
        return asdict(self)


def _percentile(values: Sequence[int], fraction: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(int(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _distribution(values: Sequence[int]) -> dict[str, float | int]:
    return {
        "minimum": min(values),
        "p10": _percentile(values, 0.10),
        "p25": _percentile(values, 0.25),
        "median": _percentile(values, 0.50),
        "p75": _percentile(values, 0.75),
        "p90": _percentile(values, 0.90),
        "maximum": max(values),
    }


@dataclass(frozen=True, slots=True)
class FullGameEvaluationReport:
    binding: ArtifactBinding
    suite: DevelopmentSuite
    criteria: PromotionCriteria
    episodes: tuple[EpisodeOutcome, ...]
    aggregate: Mapping[str, Any]
    gates: Mapping[str, bool]
    runner_identity: Mapping[str, Any]

    @property
    def promoted(self) -> bool:
        return bool(self.gates.get("all_passed", False))

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "irisu-pointer-full-game-development-evaluation-v1",
            "development_only": True,
            "sealed_test_material_used": False,
            "binding": self.binding.manifest(),
            "suite": {**self.suite.manifest(), "sha256": self.suite.sha256},
            "criteria": self.criteria.manifest(),
            "runner_identity": dict(self.runner_identity),
            "episodes": [episode.manifest() for episode in self.episodes],
            "aggregate": dict(self.aggregate),
            "gates": dict(self.gates),
            "promoted": self.promoted,
        }


def _event_kind(event: Mapping[str, Any]) -> int | None:
    raw = event.get("kind")
    if isinstance(raw, int) and not isinstance(raw, bool):
        return int(raw)
    name = event.get("kind_name")
    if isinstance(name, str):
        try:
            return int(EventKind[name.upper()])
        except KeyError:
            return None
    return None


def _policy_actions(
    decision: PointerExpertDecision | Action | tuple[Action, ...],
    observation: Mapping[str, Any],
    pointer_spec: PointerActionSpec,
) -> tuple[Action, ...]:
    if isinstance(decision, PointerExpertDecision):
        return lower_expert_decision(decision, observation, pointer_spec)
    if isinstance(decision, Action):
        return (decision,)
    if (
        isinstance(decision, tuple)
        and decision
        and all(isinstance(value, Action) for value in decision)
    ):
        return decision
    raise DevelopmentEvaluationError("policy returned an unsupported decision type")


def _policy(
    factory: Callable[[], EvaluatedPolicy],
    binding: ArtifactBinding,
    seed: int,
) -> EvaluatedPolicy:
    policy = factory()
    identities = {
        "artifact_sha256": binding.policy_sha256,
        "schema_sha256": binding.schema_sha256,
        "pointer_action_sha256": binding.pointer_action_sha256,
    }
    if any(getattr(policy, name, None) != value for name, value in identities.items()):
        raise DevelopmentEvaluationError("loaded policy identity differs from binding")
    policy.reset(seed)
    return policy


def _run_episode(
    env: Any,
    policy: EvaluatedPolicy,
    seed: int,
    suite: DevelopmentSuite,
    pointer_spec: PointerActionSpec,
    expected_config_hash: int,
) -> EpisodeOutcome:
    observation, reset_info = env.reset(seed=seed)
    if (
        type(reset_info.get("config_hash")) is not int
        or int(reset_info["config_hash"]) != expected_config_hash
    ):
        raise DevelopmentEvaluationError(
            f"seed {seed} reset config identity mismatch"
        )
    initial_tick = int(observation.get("tick", 0))
    event_counts: Counter[int] = Counter()
    hit_pairs: set[tuple[int, int]] = set()
    decisions = 0
    primitive_actions = 0
    invalid_actions = 0
    max_chain = int(observation.get("highest_chain", 0))
    terminated = bool(observation.get("terminated", False))
    truncated = bool(observation.get("truncated", False))
    while not terminated and not truncated:
        if decisions >= suite.max_decisions_per_episode:
            raise DevelopmentEvaluationError(
                f"seed {seed} exceeded the complete-game decision budget"
            )
        try:
            decision = policy.act(observation)
            actions = _policy_actions(decision, observation, pointer_spec)
        except Exception as exc:
            raise DevelopmentEvaluationError(
                f"seed {seed} policy decision failed"
            ) from exc
        decisions += 1
        if not actions:
            raise DevelopmentEvaluationError("policy decision lowered to no actions")
        for action in actions:
            if terminated or truncated:
                break
            before_tick = int(observation.get("tick", 0))
            observation, _, terminated, truncated, info = env.step(action)
            primitive_actions += 1
            events = tuple(info.get("events", ()))
            invalid = bool(info.get("invalid_action", False)) or any(
                _event_kind(event) == int(EventKind.INVALID_ACTION)
                for event in events
            )
            invalid_actions += int(invalid)
            for event in events:
                kind = _event_kind(event)
                if kind is None:
                    continue
                event_counts[kind] += 1
                if kind == int(EventKind.PROJECTILE_HIT):
                    hit_pairs.add(
                        (int(event.get("a", -1)), int(event.get("b", -1)))
                    )
            tick = int(observation.get("tick", 0))
            if tick <= before_tick:
                raise DevelopmentEvaluationError(
                    f"seed {seed} did not advance after an action"
                )
            max_chain = max(max_chain, int(observation.get("highest_chain", 0)))
    final_tick = int(observation.get("tick", 0))
    if final_tick < initial_tick:
        raise DevelopmentEvaluationError("episode tick moved backwards")
    projectiles = {projectile for projectile, _ in hit_pairs}
    bodies = {body for _, body in hit_pairs}
    return EpisodeOutcome(
        seed=seed,
        raw_score=int(observation.get("score", 0)),
        survival_ticks=final_tick - initial_tick,
        decisions=decisions,
        primitive_actions=primitive_actions,
        valid_actions=primitive_actions - invalid_actions,
        invalid_actions=invalid_actions,
        chain_joined=event_counts[int(EventKind.CHAIN_JOINED)],
        cleared=event_counts[int(EventKind.CLEARED)],
        rotten=event_counts[int(EventKind.ROTTEN)],
        max_chain=max_chain,
        projectile_hit_events=event_counts[int(EventKind.PROJECTILE_HIT)],
        unique_projectile_hit_pairs=len(hit_pairs),
        unique_projectiles_hitting=len(projectiles),
        unique_bodies_hit=len(bodies),
        terminated=bool(terminated),
        truncated=bool(truncated),
    )


def _default_env_factory(runtime: Path, config: Mapping[str, Any]) -> IrisuEnv:
    return IrisuEnv(
        library_path=runtime,
        physics_backend="portable",
        config=config or None,
    )


def evaluate_full_games(
    policy_factory: Callable[[], EvaluatedPolicy],
    binding: ArtifactBinding,
    *,
    suite: DevelopmentSuite = DEFAULT_DEVELOPMENT_SUITE,
    criteria: PromotionCriteria | None = None,
    pointer_spec: PointerActionSpec | None = None,
    env_factory: Callable[[Path, Mapping[str, Any]], Any] | None = None,
) -> FullGameEvaluationReport:
    """Evaluate complete games without accepting sealed identities or paths."""

    if not isinstance(binding, ArtifactBinding):
        raise TypeError("binding must be an ArtifactBinding")
    if not isinstance(suite, DevelopmentSuite):
        raise TypeError("suite must be a DevelopmentSuite")
    resolved_criteria = criteria or PromotionCriteria()
    resolved_pointer = pointer_spec or PointerActionSpec()
    if resolved_pointer.sha256 != binding.pointer_action_sha256:
        raise DevelopmentEvaluationError("pointer action identity mismatch")
    policy_file = _snapshot_file(
        binding.policy_path, binding.policy_sha256, name="policy artifact"
    )
    runtime_file = _snapshot_file(
        binding.runtime_path, binding.runtime_sha256, name="portable runtime"
    )
    factory = env_factory or _default_env_factory
    outcomes: list[EpisodeOutcome] = []
    with factory(runtime_file.path, dict(suite.config)) as env:
        if getattr(env, "physics_backend", None) != "portable":
            raise DevelopmentEvaluationError("development evaluator requires portable")
        loaded_runtime = Path(getattr(env, "library_path", "")).resolve()
        if loaded_runtime != runtime_file.path:
            raise DevelopmentEvaluationError("environment loaded a different runtime")
        runner_identity = dict(env.runner_identity_manifest())
        if identity_sha256(runner_identity) != binding.runner_identity_sha256:
            raise DevelopmentEvaluationError("runner provenance identity mismatch")
        runner_config_hash = runner_identity.get("config_hash")
        if type(runner_config_hash) is not int:
            raise DevelopmentEvaluationError(
                "runner identity lacks an integer config hash"
            )
        for seed in suite.seeds:
            outcomes.append(
                _run_episode(
                    env,
                    _policy(policy_factory, binding, seed),
                    seed,
                    suite,
                    resolved_pointer,
                    runner_config_hash,
                )
            )
    _verify_unchanged(policy_file, name="policy artifact")
    _verify_unchanged(runtime_file, name="portable runtime")
    scores = [outcome.raw_score for outcome in outcomes]
    ticks = [outcome.survival_ticks for outcome in outcomes]
    unique_hits = [outcome.unique_projectile_hit_pairs for outcome in outcomes]
    invalid_actions = sum(outcome.invalid_actions for outcome in outcomes)
    completion_rate = sum(
        outcome.terminated or outcome.truncated for outcome in outcomes
    ) / len(outcomes)
    aggregate = {
        "episodes": len(outcomes),
        "raw_score": _distribution(scores),
        "survival_ticks": _distribution(ticks),
        "decisions": _distribution(
            [outcome.decisions for outcome in outcomes]
        ),
        "max_chain": _distribution(
            [outcome.max_chain for outcome in outcomes]
        ),
        "chain_joined": _distribution(
            [outcome.chain_joined for outcome in outcomes]
        ),
        "cleared": _distribution([outcome.cleared for outcome in outcomes]),
        "rotten": _distribution([outcome.rotten for outcome in outcomes]),
        "projectile_hit_events": _distribution(
            [outcome.projectile_hit_events for outcome in outcomes]
        ),
        "unique_projectile_hit_pairs": _distribution(unique_hits),
        "total_decisions": sum(outcome.decisions for outcome in outcomes),
        "total_primitive_actions": sum(
            outcome.primitive_actions for outcome in outcomes
        ),
        "total_valid_actions": sum(outcome.valid_actions for outcome in outcomes),
        "total_invalid_actions": invalid_actions,
        "total_chain_joined": sum(outcome.chain_joined for outcome in outcomes),
        "total_cleared": sum(outcome.cleared for outcome in outcomes),
        "total_rotten": sum(outcome.rotten for outcome in outcomes),
        "total_projectile_hit_events": sum(
            outcome.projectile_hit_events for outcome in outcomes
        ),
        "total_unique_projectile_hit_pairs": sum(unique_hits),
        "completion_rate": completion_rate,
    }
    gates = {
        "median_raw_score": (
            aggregate["raw_score"]["median"]  # type: ignore[index]
            >= resolved_criteria.minimum_median_raw_score
        ),
        "p10_raw_score": (
            aggregate["raw_score"]["p10"]  # type: ignore[index]
            >= resolved_criteria.minimum_p10_raw_score
        ),
        "median_survival_ticks": (
            aggregate["survival_ticks"]["median"]  # type: ignore[index]
            >= resolved_criteria.minimum_median_survival_ticks
        ),
        "median_unique_projectile_hit_pairs": (
            aggregate["unique_projectile_hit_pairs"]["median"]  # type: ignore[index]
            >= resolved_criteria.minimum_median_unique_hit_pairs
        ),
        "completion_rate": (
            completion_rate >= resolved_criteria.minimum_completion_rate
        ),
        "invalid_actions": (
            invalid_actions <= resolved_criteria.maximum_invalid_actions
        ),
        "fixed_unsealed_suite": (
            suite.manifest()["sealed"] is False
            and suite.manifest()["development_only"] is True
        ),
        "artifact_and_runtime_reverified": True,
    }
    gates["all_passed"] = all(gates.values())
    return FullGameEvaluationReport(
        binding,
        suite,
        resolved_criteria,
        tuple(outcomes),
        aggregate,
        gates,
        runner_identity,
    )


__all__ = [
    "ArtifactBinding",
    "DEFAULT_DEVELOPMENT_SUITE",
    "DevelopmentEvaluationError",
    "DevelopmentSuite",
    "EpisodeOutcome",
    "FullGameEvaluationReport",
    "PromotionCriteria",
    "UNSEEN_DEVELOPMENT_SEEDS",
    "evaluate_full_games",
    "identity_sha256",
]
