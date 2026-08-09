"""Conservative all-outcome residual learning for the R3G development screen."""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from irisu_env import EventKind
from irisu_rl.encoding import TeacherStateEncoder

from .joint_planner import (
    COMPACT_GEOMETRY,
    JointBranchOutcome,
    JointCandidate,
    JointPairGeometrySearch,
    JointPlannerConfig,
    JointSearchResult,
    _commit_base_decision,
)
from .policy import encoded_body_ids
from .steering import SteeringDecision


R3G_OFFLINE_VERSION = "irisu-r3g-conservative-offline-residual-v1"
LOCKED_PROTOCOL_PATH = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3g-solvency-barrier-tournament-20260729/shared-protocol.md"
)
LOCKED_PROTOCOL_SHA256 = (
    "6dfb2ffa3a76cc00447e3dcf889f6209a17ff5e2f4c3382fe0959bbabbd52991"
)
TRUSTED_RUNTIME_PATH = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/runtime/"
    "main-0c48dba-20260723/portable-build/libirisu_clone.so"
)
TRUSTED_RUNTIME_SHA256 = (
    "4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79"
)
FROZEN_V5_PATH = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3d-survival-v5-20260729/long-development.pt"
)
FROZEN_V5_SHA256 = (
    "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d"
)
QUANTILES = (0.05, 0.10, 0.25, 0.50)
RENEWABLE_DETAILS = frozenset(
    {"normal burst landing", "special color clear"}
)
DEBT_DETAILS = frozenset(
    {"normal rot penalty", "scene clamp and passive drain"}
)
PAIR_CATEGORIES = ("rotten-hazard", "viable-anchor", "fresh-match")
INPUT_DIM = 12 + 45 + 45 + 18


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_trusted_identities() -> dict[str, str]:
    expected = {
        str(LOCKED_PROTOCOL_PATH): LOCKED_PROTOCOL_SHA256,
        str(TRUSTED_RUNTIME_PATH): TRUSTED_RUNTIME_SHA256,
        str(FROZEN_V5_PATH): FROZEN_V5_SHA256,
    }
    actual = {path: _file_sha256(Path(path)) for path in expected}
    if actual != expected:
        raise RuntimeError("locked protocol/runtime/frozen-v5 identity mismatch")
    return actual


def _event_kind(event: Mapping[str, Any]) -> int | None:
    value = event.get("kind")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    name = event.get("kind_name")
    if isinstance(name, str):
        try:
            return int(EventKind[name.upper()])
        except KeyError:
            return None
    return None


@dataclass(frozen=True, slots=True)
class BranchTrace:
    initial: Mapping[str, Any]
    final: Mapping[str, Any]
    events: tuple[Mapping[str, Any], ...]
    frames: tuple[Mapping[str, Any], ...] = ()


class BranchTraceEnvironment:
    """Trace each exact branch without changing the wrapped environment."""

    def __init__(self, env: Any) -> None:
        self._env = env
        self._active: dict[str, Any] | None = None
        self._completed: list[BranchTrace] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def _close(self) -> None:
        if self._active is None:
            return
        self._completed.append(
            BranchTrace(
                self._active["initial"],
                self._active["final"],
                tuple(self._active["events"]),
                tuple(self._active["frames"]),
            )
        )

    def restore_state(self, snapshot: bytes) -> Mapping[str, Any]:
        self._close()
        restored = self._env.restore_state(snapshot)
        self._active = {
            "initial": restored,
            "final": restored,
            "events": [],
            "frames": [],
        }
        return restored

    def step(self, action: Any) -> Any:
        if self._active is None:
            raise RuntimeError("branch tracing began without a restore")
        result = self._env.step(action)
        observation, _reward, _terminated, _truncated, info = result
        if not isinstance(observation, Mapping) or not isinstance(info, Mapping):
            raise TypeError("traced portable transition is malformed")
        supplied = [
            dict(value)
            for value in info.get("events", ())
            if isinstance(value, Mapping)
        ]
        self._active["events"].extend(supplied)
        self._active["frames"].append(observation)
        self._active["final"] = observation
        return result

    def finish(self, expected: int) -> tuple[BranchTrace, ...]:
        if len(self._completed) != expected:
            raise RuntimeError(
                f"traced {len(self._completed)} branches, expected {expected}"
            )
        return tuple(self._completed)


@dataclass(frozen=True, slots=True)
class SolvencyOutcome:
    minimum_surplus: int
    minimum_gauge: int
    renewal_count: int
    first_renewal_ticks: int
    second_renewal_ticks: int
    liability_ids: tuple[int, ...]
    paid_liability_ids: tuple[int, ...]
    emergent_paid_ids: tuple[int, ...]
    censored_before_second_renewal: bool
    negative_through_renewal: bool
    event_sha256: str
    game_over: bool = False
    gauge_failure: bool = False
    cancelled_liability_ids: tuple[int, ...] = ()

    def manifest(self) -> dict[str, object]:
        return {
            "minimum_surplus": self.minimum_surplus,
            "minimum_gauge": self.minimum_gauge,
            "renewal_count": self.renewal_count,
            "first_renewal_ticks": self.first_renewal_ticks,
            "second_renewal_ticks": self.second_renewal_ticks,
            "liability_ids": list(self.liability_ids),
            "paid_liability_ids": list(self.paid_liability_ids),
            "emergent_paid_ids": list(self.emergent_paid_ids),
            "censored_before_second_renewal": (
                self.censored_before_second_renewal
            ),
            "negative_through_renewal": self.negative_through_renewal,
            "game_over": self.game_over,
            "gauge_failure": self.gauge_failure,
            "cancelled_liability_ids": list(self.cancelled_liability_ids),
            "event_sha256": self.event_sha256,
        }


def trace_solvency(trace: BranchTrace, *, horizon_ticks: int) -> SolvencyOutcome:
    """Replay the locked event-ordered B2 gauge recurrence exactly."""

    if type(horizon_ticks) is not int or horizon_ticks < 1:
        raise ValueError("solvency horizon must be positive")
    start_tick = int(trace.initial.get("tick", 0))
    final_tick = int(trace.final.get("tick", start_tick))
    initial_gauge = int(trace.initial.get("gauge", 0))
    gauge_max = int(trace.initial.get("gauge_max", 0))
    level = int(trace.initial.get("level", 0))
    if gauge_max <= 0 or initial_gauge > gauge_max or level < 1:
        raise ValueError("trace has invalid initial gauge or level")
    if final_tick < start_tick:
        raise ValueError("trace ends before it starts")
    liability_deadlines: dict[int, int] = {}
    for body in trace.initial.get("bodies", ()):
        if (
            not isinstance(body, Mapping)
            or body.get("kind") != "piece"
            or body.get("lifecycle") == "rotten"
        ):
            continue
        timer = int(body.get("rot_timer", 0))
        if timer <= 0:
            continue
        body_id = int(body.get("id", -1))
        if body_id < 0 or body_id in liability_deadlines:
            raise ValueError("trace liability body IDs are invalid")
        liability_deadlines[body_id] = start_tick + max(1, 41 - timer)
    liability_ids = frozenset(liability_deadlines)
    ordered = sorted(
        enumerate(trace.events),
        key=lambda item: (
            int(item[1].get("tick", -1)),
            int(item[1].get("sequence", item[0])),
            item[0],
        ),
    )
    by_tick: dict[int, list[Mapping[str, Any]]] = {}
    for _source, event in ordered:
        tick = int(event.get("tick", -1))
        if not start_tick < tick <= min(
            start_tick + horizon_ticks, final_tick
        ):
            continue
        by_tick.setdefault(tick, []).append(event)
    q = initial_gauge
    observed_gauge = initial_gauge
    minimum = q
    paid: set[int] = set()
    cancelled: set[int] = set()
    emergent: set[int] = set()
    renewals: list[int] = []
    game_over = q <= 0
    gauge_failure = q <= 0
    processed_tick = start_tick
    normalized_events: list[dict[str, object]] = []
    for tick in range(
        start_tick + 1,
        min(start_tick + horizon_ticks, final_tick) + 1,
    ):
        events = by_tick.get(tick, [])
        processed_tick = tick
        b2_active = len(renewals) < 2
        entry_failed = q <= 0 or any(
            _event_kind(event) == int(EventKind.GAME_OVER)
            for event in events
        )
        game_over = game_over or any(
            _event_kind(event) == int(EventKind.GAME_OVER)
            for event in events
        )
        gauge_failure = gauge_failure or entry_failed
        for event in events:
            if _event_kind(event) == int(EventKind.LEVEL_CHANGED):
                level = int(event.get("value", level))
        parameter_level = min(level, 99)
        renewable_gain = sum(
            int(event.get("value", 0))
            for event in events
            if _event_kind(event) == int(EventKind.GAUGE_CHANGED)
            and int(event.get("value", 0)) > 0
            and str(event.get("detail", "")) in RENEWABLE_DETAILS
        )
        x = min(max(q + renewable_gain, 0), gauge_max)
        net_post_clamp_recovery = max(0, x - min(max(q, 0), gauge_max))
        unit = parameter_level // 10 + 1
        drain = unit * (3 if x > gauge_max / 2 else 1)
        post_drain = max(1, x - drain)
        scene_events = [
            event
            for event in events
            if _event_kind(event) == int(EventKind.GAUGE_CHANGED)
            and str(event.get("detail", ""))
            == "scene clamp and passive drain"
        ]
        expected_scene_delta = post_drain - (q + renewable_gain)
        if (
            expected_scene_delta == 0
            and scene_events
        ) or (
            expected_scene_delta != 0
            and (
                len(scene_events) != 1
                or int(scene_events[0].get("value", 0))
                != expected_scene_delta
            )
        ):
            raise RuntimeError("scene gauge event violates exact recurrence")
        q = post_drain
        if b2_active:
            minimum = min(minimum, q)
        rotten_ids = [
            int(event.get("a", -1))
            for event in events
            if _event_kind(event) == int(EventKind.ROTTEN)
        ]
        penalty_events = [
            event
            for event in events
            if _event_kind(event) == int(EventKind.GAUGE_CHANGED)
            and str(event.get("detail", "")) == "normal rot penalty"
        ]
        if rotten_ids != [
            int(event.get("a", -1)) for event in penalty_events
        ]:
            raise RuntimeError("rot events and gauge penalties do not match")
        if len(rotten_ids) != len(set(rotten_ids)):
            raise RuntimeError("duplicate rot liability payment in one tick")
        penalty = 1800 + 20 * parameter_level
        for body_id, event in zip(
            rotten_ids, penalty_events, strict=True
        ):
            if (
                body_id < 0
                or body_id in paid
                or body_id in emergent
                or body_id in cancelled
            ):
                raise RuntimeError(
                    "rot liability was paid more than once or after cancellation"
                )
            if int(event.get("value", 0)) != -penalty:
                raise RuntimeError("rot penalty violates exact level recurrence")
            deadline = liability_deadlines.pop(body_id, None)
            if deadline is not None:
                if tick < deadline:
                    raise RuntimeError("rot liability paid before its deadline")
                paid.add(body_id)
            else:
                emergent.add(body_id)
            q -= penalty
            if b2_active:
                minimum = min(minimum, q)
        prevented_ids = {
            int(event.get("a", -1))
            for event in events
            if _event_kind(event)
            in (int(EventKind.CLEARED), int(EventKind.DESTROYED))
        }
        for body_id in sorted(prevented_ids & liability_ids - paid):
            if body_id in cancelled:
                continue
            liability_deadlines.pop(body_id, None)
            cancelled.add(body_id)
        observed_delta = sum(
            int(event.get("value", 0))
            for event in events
            if _event_kind(event) == int(EventKind.GAUGE_CHANGED)
        )
        observed_gauge += observed_delta
        if q != observed_gauge:
            raise RuntimeError("gauge events do not match exact recurrence")
        if (
            b2_active
            and renewable_gain > 0
            and net_post_clamp_recovery > 0
            and not entry_failed
        ):
            renewals.append(tick)
        normalized_events.extend(
            {
                **dict(event),
                "tick": tick - start_tick,
            }
            for event in events
        )
    if processed_tick == final_tick and q != int(trace.final.get("gauge", q)):
        raise RuntimeError("exact recurrence does not reconstruct final gauge")
    censored = len(renewals) < 2
    return SolvencyOutcome(
        minimum_surplus=minimum,
        minimum_gauge=minimum,
        renewal_count=len(renewals),
        first_renewal_ticks=(
            horizon_ticks if not renewals else renewals[0] - start_tick
        ),
        second_renewal_ticks=(
            horizon_ticks if censored else renewals[1] - start_tick
        ),
        liability_ids=tuple(sorted(liability_ids)),
        paid_liability_ids=tuple(sorted(paid)),
        emergent_paid_ids=tuple(sorted(emergent)),
        censored_before_second_renewal=censored,
        negative_through_renewal=minimum <= 0,
        event_sha256=canonical_sha256(normalized_events),
        game_over=game_over,
        gauge_failure=gauge_failure,
        cancelled_liability_ids=tuple(sorted(cancelled)),
    )


def branch_labels(
    outcome: JointBranchOutcome,
    incumbent: JointBranchOutcome,
    solvency: SolvencyOutcome,
    incumbent_solvency: SolvencyOutcome,
    *,
    gauge_max: int,
) -> dict[str, float | bool]:
    if gauge_max <= 0:
        raise ValueError("gauge maximum must be positive")
    invalid = any(value.invalid_actions > 0 for value in outcome.milestones)
    candidate_survival = outcome.milestones[-1].survival_ticks
    base_survival = incumbent.milestones[-1].survival_ticks
    material = base_survival - candidate_survival >= 1000
    catastrophic = material and (
        base_survival > 0
        and candidate_survival / base_survival <= 0.5
    )
    new_terminal = solvency.game_over and not incumbent_solvency.game_over
    new_gauge_failure = (
        solvency.gauge_failure and not incumbent_solvency.gauge_failure
    )
    hard = invalid or catastrophic or new_terminal or new_gauge_failure
    relative_solvency = (
        solvency.minimum_surplus - incumbent_solvency.minimum_surplus
    ) / gauge_max
    score_advantage = (
        outcome.milestones[-1].score_gain
        - incumbent.milestones[-1].score_gain
    )
    resolved = (
        not solvency.censored_before_second_renewal
        and not incumbent_solvency.censored_before_second_renewal
    )
    unsafe = (
        not resolved
        or solvency.minimum_surplus <= 0
        or relative_solvency < 0.0
        or hard
    )
    return {
        "risk_margin": float(relative_solvency),
        "absolute_solvency": solvency.minimum_surplus / gauge_max,
        "score_advantage": math.copysign(
            math.log1p(abs(score_advantage)), score_advantage
        )
        / 8.0,
        "unsafe": unsafe,
        "hard_catastrophe": hard,
        "severe_renewable": solvency.negative_through_renewal,
        "censored": solvency.censored_before_second_renewal,
        "resolved": resolved,
        "new_terminal": new_terminal,
        "new_gauge_failure": new_gauge_failure,
        "catastrophic": catastrophic,
    }


def _candidate_tail(candidate: JointCandidate) -> np.ndarray:
    category = [float(candidate.pair.category == value) for value in PAIR_CATEGORIES]
    geometry = [
        float(candidate.geometry_ordinal == index)
        for index in range(len(COMPACT_GEOMETRY))
    ]
    kind = int(candidate.decision.action.kind)
    return np.asarray(
        [
            float(candidate.pair.incumbent),
            0.0,  # pair order is deliberately not a learner input
            *geometry,
            *category,
            math.log1p(candidate.pair.distance_sizes) / 4.0,
            float(kind == 1),
            float(kind == 2),
            float(candidate.decision.action.x_norm),
            float(candidate.decision.action.y_norm),
            float(candidate.pair.destination_chain_id != 0),
            float(candidate.geometry.side_sizes),
            float(candidate.geometry.below_sizes),
        ],
        dtype=np.float32,
    )


def encode_candidate(
    observation: Mapping[str, Any], candidate: JointCandidate
) -> tuple[np.ndarray, np.ndarray, str]:
    encoded = TeacherStateEncoder().encode([observation])
    identifiers = encoded_body_ids(encoded, observation)
    try:
        source = identifiers.index(candidate.pair.source_body_id)
        destination = identifiers.index(candidate.pair.destination_body_id)
    except ValueError as exc:
        raise ValueError("candidate body is absent from encoded state") from exc
    source_features = encoded.body_features[0, source].copy()
    destination_features = encoded.body_features[0, destination].copy()
    # IDs bind the row but are not learner inputs.
    source_features[39:41] = 0.0
    destination_features[39:41] = 0.0
    tail = _candidate_tail(candidate)
    global_features = encoded.global_features[0].copy()
    global_features[0] = 0.0  # absolute tick offset cannot affect certification
    row = np.concatenate(
        (
            global_features,
            source_features,
            destination_features,
            tail,
        )
    ).astype(np.float32, copy=False)
    if row.shape != (INPUT_DIM,) or not np.all(np.isfinite(row)):
        raise ValueError("candidate encoding is malformed")
    support = np.concatenate(
        (
            global_features[[2, 3, 4, 6, 7]],
            source_features[[16, 17, 18, 19, 22, 23, 24, 25, 43, 44]],
            destination_features[[16, 17, 18, 19, 22, 23, 24, 25, 43, 44]],
            tail,
        )
    ).astype(np.float32, copy=False)
    signature = (
        f"{candidate.pair.category}|g{candidate.geometry_ordinal}|"
        f"i{int(candidate.pair.incumbent)}"
    )
    return row, support, signature


@dataclass(slots=True)
class RecordedSearch:
    seed: int
    query_index: int
    result: JointSearchResult
    traces: tuple[BranchTrace, ...]

    def rows(self) -> list[dict[str, Any]]:
        if len(self.traces) != len(self.result.outcomes):
            raise RuntimeError("trace/outcome count mismatch")
        initial = self.traces[0].initial
        gauge_max = int(initial.get("gauge_max", 0))
        horizon = self.result.outcomes[0].milestones[-1].horizon_ticks
        solvencies = [
            trace_solvency(trace, horizon_ticks=horizon) for trace in self.traces
        ]
        incumbent = self.result.outcomes[0]
        incumbent_solvency = solvencies[0]
        rows: list[dict[str, Any]] = []
        for outcome, solvency in zip(
            self.result.outcomes, solvencies, strict=True
        ):
            features, support, signature = encode_candidate(
                initial, outcome.candidate
            )
            rows.append(
                {
                    "seed": self.seed,
                    "query_index": self.query_index,
                    "tick": int(initial.get("tick", 0)),
                    "search_sha256": self.result.sha256,
                    "snapshot_sha256": self.result.snapshot_sha256,
                    "ordinal": outcome.candidate.ordinal,
                    "incumbent": outcome.candidate.ordinal == 0,
                    "selected": (
                        outcome.candidate.ordinal
                        == self.result.selected_candidate.ordinal
                    ),
                    "features": features,
                    "support": support,
                    "signature": signature,
                    "labels": branch_labels(
                        outcome,
                        incumbent,
                        solvency,
                        incumbent_solvency,
                        gauge_max=gauge_max,
                    ),
                    "candidate": outcome.candidate.manifest(),
                    "outcome": outcome.manifest(),
                    "solvency": solvency.manifest(),
                }
            )
        return rows


class RecordingJointSearch(JointPairGeometrySearch):
    """Exact joint-v2 search with all-candidate trace retention."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.continuation_identity_sha256 is None:
            raise ValueError("recording search requires a bound continuation")
        self.records: list[RecordedSearch] = []
        self.seed = 0

    def reset_records(self, seed: int) -> None:
        self.records.clear()
        self.seed = seed

    def search(
        self,
        env: Any,
        observation: Mapping[str, Any],
        incumbent: SteeringDecision,
    ) -> JointSearchResult:
        traced = BranchTraceEnvironment(env)
        result = super().search(traced, observation, incumbent)
        traces = traced.finish(len(result.outcomes))
        self.records.append(
            RecordedSearch(self.seed, len(self.records), result, traces)
        )
        return result


class AllOutcomeCollectorPolicy:
    """Query the exact teacher at frozen-v5 states but always execute v5."""

    def __init__(
        self,
        env: Any,
        base_policy: object,
        searcher: RecordingJointSearch,
        *,
        query_stride_shots: int = 2,
        maximum_queries: int = 48,
    ) -> None:
        if (
            type(query_stride_shots) is not int
            or query_stride_shots < 1
            or type(maximum_queries) is not int
            or maximum_queries < 1
        ):
            raise ValueError("collector query settings must be positive integers")
        self.env = env
        self.base_policy = base_policy
        self.searcher = searcher
        self.query_stride_shots = query_stride_shots
        self.maximum_queries = maximum_queries
        self.counts: Counter[str] = Counter()

    def reset(self, seed: int = 0) -> None:
        getattr(self.base_policy, "reset")(seed)
        self.searcher.reset_records(seed)
        self.counts.clear()

    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = getattr(self.base_policy, "predict")(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("collector base returned a non-decision")
        if not incumbent.is_shot:
            return incumbent
        self.counts["seen_shots"] += 1
        if (
            (self.counts["seen_shots"] - 1) % self.query_stride_shots
            or self.counts["search_queries"] >= self.maximum_queries
        ):
            self.counts["query_abstentions"] += 1
            return incumbent
        try:
            result = self.searcher.search(self.env, observation, incumbent)
        except ValueError:
            self.counts["unsupported_queries"] += 1
            return incumbent
        self.counts["search_queries"] += 1
        self.counts["candidate_outcomes"] += len(result.outcomes)
        self.counts["restore_checks"] += result.restore_checks
        self.counts["simulated_branch_ticks"] += result.simulated_ticks
        self.counts["teacher_winner_overrides"] += int(
            result.selected_candidate.ordinal != 0
        )
        return incumbent

    def statistics(self) -> dict[str, int]:
        return dict(sorted(self.counts.items()))


class ResidualValueNet(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.SiLU(),
            nn.Linear(96, 64),
            nn.SiLU(),
        )
        self.quantiles = nn.Linear(64, 2 * len(QUANTILES))
        self.unsafe = nn.Linear(64, 1)
        self.score = nn.Linear(64, 1)
        self.value = nn.Sequential(
            nn.Linear(12, 32), nn.SiLU(), nn.Linear(32, 1)
        )

    def forward(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trunk(features)
        return (
            self.quantiles(hidden),
            self.unsafe(hidden).squeeze(-1),
            self.score(hidden).squeeze(-1),
            self.value(features[:, :12]).squeeze(-1),
        )


def _pinball(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    error = target[:, None] - prediction
    tau = prediction.new_tensor(QUANTILES)
    return torch.maximum(tau * error, (tau - 1.0) * error).mean()


def _expectile(error: torch.Tensor, tau: float) -> torch.Tensor:
    weight = torch.where(error > 0, tau, 1.0 - tau)
    return (weight * error.square()).mean()


@dataclass(frozen=True, slots=True)
class TrainedEnsemble:
    models: tuple[ResidualValueNet, ...]
    mean: torch.Tensor
    scale: torch.Tensor
    training_report: Mapping[str, Any]

    @torch.no_grad()
    def predict_full(
        self, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        normalized = (features - self.mean) / self.scale
        outputs = [model(normalized) for model in self.models]
        quantiles = torch.stack([value[0] for value in outputs])
        delta, absolute = quantiles.split(len(QUANTILES), dim=-1)
        delta, _ = delta.sort(dim=-1)
        absolute, _ = absolute.sort(dim=-1)
        delta_cvar = delta[..., :3].mean(dim=-1)
        absolute_cvar = absolute[..., :3].mean(dim=-1)
        unsafe = torch.stack([value[1].sigmoid() for value in outputs])
        score = torch.stack([value[2] - value[3] for value in outputs])
        return {
            "delta_cvar_mean": delta_cvar.mean(0),
            "delta_cvar_std": delta_cvar.std(0, unbiased=False),
            "absolute_cvar_mean": absolute_cvar.mean(0),
            "absolute_cvar_std": absolute_cvar.std(0, unbiased=False),
            "unsafe_mean": unsafe.mean(0),
            "score_mean": score.mean(0),
            "score_std": score.std(0, unbiased=False),
        }

    @torch.no_grad()
    def predict(
        self, features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        full = self.predict_full(features)
        conservative = torch.minimum(
            full["delta_cvar_mean"], full["absolute_cvar_mean"]
        )
        spread = torch.maximum(
            full["delta_cvar_std"], full["absolute_cvar_std"]
        )
        return {
            "cvar_mean": conservative,
            "cvar_std": spread,
            "unsafe_mean": full["unsafe_mean"],
            "score_mean": full["score_mean"],
            "score_std": full["score_std"],
        }


def train_ensemble(
    tensors: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    *,
    winner_only: bool,
    steps: int = 500,
    ensemble_size: int = 3,
    seed: int = 2026072905,
) -> TrainedEnsemble:
    if winner_only:
        indices = indices[tensors["selected"][indices]]
    if indices.numel() == 0:
        raise ValueError("offline training selection is empty")
    features = tensors["features"][indices].float()
    mean = features.mean(0)
    scale = features.std(0, unbiased=False).clamp_min(1e-4)
    normalized = (features - mean) / scale
    risk = tensors["risk_margin"][indices].float()
    absolute = tensors.get("absolute_solvency", tensors["risk_margin"])[
        indices
    ].float()
    unsafe = tensors["unsafe"][indices].float()
    resolved = tensors.get("resolved", ~tensors["unsafe"].bool())[indices].bool()
    score = tensors["score_advantage"][indices].float()
    positive = unsafe.sum().clamp_min(1.0)
    negative = (1.0 - unsafe).sum().clamp_min(1.0)
    positive_weight = (negative / positive).clamp(0.25, 20.0)
    models: list[ResidualValueNet] = []
    reports: list[dict[str, float]] = []
    for member in range(ensemble_size):
        torch.manual_seed(seed + member)
        model = ResidualValueNet()
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=5e-4, weight_decay=1e-4
        )
        generator = torch.Generator().manual_seed(seed + 100 + member)
        initial = final = 0.0
        for step in range(steps):
            batch = torch.randint(
                0,
                normalized.shape[0],
                (min(256, normalized.shape[0]),),
                generator=generator,
            )
            q, logit, score_q, value = model(normalized[batch])
            delta_q, absolute_q = q.split(len(QUANTILES), dim=-1)
            target_score = score[batch]
            risk_mask = resolved[batch]
            risk_loss = (
                _pinball(delta_q[risk_mask], risk[batch][risk_mask])
                + _pinball(
                    absolute_q[risk_mask], absolute[batch][risk_mask]
                )
                if bool(risk_mask.any())
                else q.sum() * 0.0
            )
            loss = (
                2.0 * risk_loss
                + F.binary_cross_entropy_with_logits(
                    logit,
                    unsafe[batch],
                    pos_weight=positive_weight,
                )
                + F.smooth_l1_loss(score_q, target_score)
                + _expectile(target_score - value, 0.70)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            final = float(loss.detach())
            if step == 0:
                initial = final
        model.eval()
        models.append(model)
        reports.append({"initial_loss": initial, "final_loss": final})
    return TrainedEnsemble(
        tuple(models),
        mean,
        scale,
        {
            "winner_only": winner_only,
            "examples": int(indices.numel()),
            "steps": steps,
            "ensemble_size": ensemble_size,
            "members": reports,
        },
    )


@dataclass(frozen=True, slots=True)
class SupportEnvelope:
    centers: Mapping[str, torch.Tensor]
    scales: Mapping[str, torch.Tensor]
    thresholds: Mapping[str, float]
    minimum_groups: int = 8

    def passes(self, signature: str, value: torch.Tensor) -> bool:
        if signature not in self.centers:
            return False
        distance = torch.sqrt(
            (((value - self.centers[signature]) / self.scales[signature]) ** 2).mean()
        )
        return float(distance) <= self.thresholds[signature]


def fit_support(
    tensors: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    signatures: Sequence[str],
    *,
    winner_only: bool,
    minimum_groups: int = 8,
) -> SupportEnvelope:
    if winner_only:
        indices = indices[tensors["selected"][indices]]
    centers: dict[str, torch.Tensor] = {}
    scales: dict[str, torch.Tensor] = {}
    thresholds: dict[str, float] = {}
    group_ids = tensors["group"][indices]
    for signature in sorted(set(signatures[index] for index in indices.tolist())):
        chosen = torch.tensor(
            [
                index
                for index in indices.tolist()
                if signatures[index] == signature
            ],
            dtype=torch.long,
        )
        if torch.unique(tensors["group"][chosen]).numel() < minimum_groups:
            continue
        values = tensors["support"][chosen].float()
        center = values.median(0).values
        scale = (values - center).abs().median(0).values.clamp_min(1e-3)
        distance = torch.sqrt((((values - center) / scale) ** 2).mean(1))
        centers[signature] = center
        scales[signature] = scale
        thresholds[signature] = float(torch.quantile(distance, 0.99)) + 1e-6
    return SupportEnvelope(centers, scales, thresholds, minimum_groups)


@dataclass(frozen=True, slots=True)
class IsotonicCalibration:
    bounds: tuple[float, ...]
    values: tuple[float, ...]

    def predict(self, score: float) -> float:
        if not self.bounds:
            return 1.0
        index = min(bisect.bisect_left(self.bounds, score), len(self.values) - 1)
        return self.values[index]


def fit_isotonic(scores: Sequence[float], labels: Sequence[bool]) -> IsotonicCalibration:
    ordered = sorted(zip(scores, labels, strict=True))
    blocks: list[list[float]] = []
    for score, label in ordered:
        if blocks and score == blocks[-1][1]:
            blocks[-1][2] += 1.0
            blocks[-1][3] += float(label)
        else:
            blocks.append([score, score, 1.0, float(label)])
        while len(blocks) >= 2:
            left, right = blocks[-2], blocks[-1]
            if left[3] / left[2] <= right[3] / right[2]:
                break
            blocks[-2:] = [
                [
                    left[0],
                    right[1],
                    left[2] + right[2],
                    left[3] + right[3],
                ]
            ]
    calibrated: list[float] = []
    for value in blocks:
        smoothed = (value[3] + 1.0) / (value[2] + 2.0)
        calibrated.append(
            smoothed if not calibrated else max(calibrated[-1], smoothed)
        )
    return IsotonicCalibration(
        tuple(value[1] for value in blocks), tuple(calibrated)
    )


@dataclass(frozen=True, slots=True)
class BarrierCalibration:
    margin_threshold: float
    probability_threshold: float
    score_threshold: float
    isotonic: IsotonicCalibration
    report: Mapping[str, Any]
    conformal_q: float = 0.0
    absolute_conformal_q: float = 0.0
    alpha: float = 0.05
    episode_count: int = 0


def fit_barrier(
    ensemble: TrainedEnsemble,
    support: SupportEnvelope,
    tensors: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    signatures: Sequence[str],
    *,
    alpha: float = 0.05,
) -> BarrierCalibration:
    if not 0.0 < alpha < 1.0:
        raise ValueError("conformal alpha must lie in (0, 1)")
    if "seed" not in tensors or "resolved" not in tensors:
        raise ValueError("whole-seed conformal tensors are incomplete")
    prediction = ensemble.predict_full(tensors["features"][indices].float())
    delta_prediction = prediction["delta_cvar_mean"]
    absolute_prediction = prediction["absolute_cvar_mean"]
    raw_probability = prediction["unsafe_mean"]
    unsafe = tensors["unsafe"][indices].bool()
    isotonic = fit_isotonic(
        raw_probability.tolist(), unsafe.tolist()
    )
    supported = torch.tensor(
        [
            support.passes(
                signatures[index], tensors["support"][index].float()
            )
            for index in indices.tolist()
        ],
        dtype=torch.bool,
    )
    alternatives = ~tensors["incumbent"][indices].bool()
    seeds = tensors["seed"][indices]
    delta_residuals: list[float] = []
    absolute_residuals: list[float] = []
    for seed in torch.unique(seeds).tolist():
        chosen = seeds == seed
        if not bool(chosen.any()):
            raise RuntimeError("empty whole-seed conformal cluster")
        delta_residuals.append(
            float(
                (
                    delta_prediction[chosen]
                    - tensors["risk_margin"][indices][chosen]
                ).max()
            )
        )
        absolute_residuals.append(
            float(
                (
                    absolute_prediction[chosen]
                    - tensors["absolute_solvency"][indices][chosen]
                ).max()
            )
        )
    episode_count = len(delta_residuals)
    order = math.ceil((episode_count + 1) * (1.0 - alpha))
    if episode_count < 59 or order > episode_count:
        raise RuntimeError("insufficient whole-seed conformal episodes")
    conformal_q = sorted(delta_residuals)[order - 1]
    absolute_q = sorted(absolute_residuals)[order - 1]
    calibrated_probability = torch.tensor(
        [isotonic.predict(value) for value in raw_probability.tolist()]
    )
    unsafe_probability = calibrated_probability[
        supported & alternatives & unsafe
    ]
    if unsafe_probability.numel() == 0:
        raise RuntimeError("calibration contains no supported unsafe outcomes")
    probability_threshold = max(
        0.0, float(unsafe_probability.min()) - 1e-6
    )
    score_threshold = 0.0
    certified = (
        supported
        & alternatives
        & (delta_prediction - conformal_q >= 0.0)
        & (absolute_prediction - absolute_q >= 0.0)
        & (calibrated_probability <= probability_threshold)
        & (prediction["score_mean"] - 2.0 * prediction["score_std"] > score_threshold)
    )
    return BarrierCalibration(
        conformal_q,
        probability_threshold,
        score_threshold,
        isotonic,
        {
            "rows": int(indices.numel()),
            "unsafe": int(unsafe.sum()),
            "supported_alternatives": int((supported & alternatives).sum()),
            "certified_alternatives": int(certified.sum()),
            "false_safe_unsafe": int((certified & unsafe).sum()),
            "conformal_episode_count": episode_count,
            "conformal_order": order,
            "conformal_q": conformal_q,
            "absolute_conformal_q": absolute_q,
            "alpha": alpha,
        },
        conformal_q,
        absolute_q,
        alpha,
        episode_count,
    )


class ConservativeResidualPolicy:
    """Teacher-free fail-closed residual over exact public candidates."""

    def __init__(
        self,
        base_policy: object,
        searcher: JointPairGeometrySearch,
        ensemble: TrainedEnsemble,
        support: SupportEnvelope,
        calibration: BarrierCalibration,
        *,
        minimum_override_gap_shots: int = 1,
    ) -> None:
        if (
            type(minimum_override_gap_shots) is not int
            or minimum_override_gap_shots < 1
        ):
            raise ValueError("override gap must be a positive integer")
        self.base_policy = base_policy
        self.searcher = searcher
        self.ensemble = ensemble
        self.support = support
        self.calibration = calibration
        self.minimum_override_gap_shots = minimum_override_gap_shots
        self.counts: Counter[str] = Counter()
        self._last_override_shot = -10**9
        self.last_incumbent: SteeringDecision | None = None
        self.last_candidates: tuple[JointCandidate, ...] = ()
        self.last_selected_ordinal = 0
        self.last_supported_alternatives = 0
        self.last_certified_alternatives = 0

    def reset(self, seed: int = 0) -> None:
        getattr(self.base_policy, "reset")(seed)
        self.counts.clear()
        self._last_override_shot = -10**9
        self.last_incumbent = None
        self.last_candidates = ()
        self.last_selected_ordinal = 0
        self.last_supported_alternatives = 0
        self.last_certified_alternatives = 0

    @torch.no_grad()
    def predict(self, observation: Mapping[str, Any]) -> SteeringDecision:
        incumbent = getattr(self.base_policy, "predict")(observation)
        if not isinstance(incumbent, SteeringDecision):
            raise TypeError("residual base returned a non-decision")
        if not incumbent.is_shot:
            return incumbent
        self.last_incumbent = incumbent
        self.last_candidates = ()
        self.last_selected_ordinal = 0
        self.last_supported_alternatives = 0
        self.last_certified_alternatives = 0
        self.counts["candidate_queries"] += 1
        try:
            candidates = self.searcher._candidates(observation, incumbent)
        except ValueError:
            self.counts["shortlist_abstentions"] += 1
            return incumbent
        self.last_candidates = candidates
        rows: list[np.ndarray] = []
        supports: list[np.ndarray] = []
        signatures: list[str] = []
        for candidate in candidates:
            row, support, signature = encode_candidate(observation, candidate)
            rows.append(row)
            supports.append(support)
            signatures.append(signature)
        features = torch.from_numpy(np.stack(rows))
        predict_full = getattr(self.ensemble, "predict_full", None)
        if not callable(predict_full):
            self.counts["conformal_interface_abstentions"] += 1
            return incumbent
        full_prediction = predict_full(features)
        delta_lcb = (
            full_prediction["delta_cvar_mean"]
            - self.calibration.conformal_q
        )
        absolute_lcb = (
            full_prediction["absolute_cvar_mean"]
            - self.calibration.absolute_conformal_q
        )
        risk_lcb = torch.minimum(delta_lcb, absolute_lcb)
        prediction = {
            "unsafe_mean": full_prediction["unsafe_mean"],
            "score_mean": full_prediction["score_mean"],
            "score_std": full_prediction["score_std"],
        }
        score_lcb = prediction["score_mean"] - 2.0 * prediction["score_std"]
        eligible: list[int] = []
        for index in range(1, len(candidates)):
            self.counts["alternative_candidates"] += 1
            if (
                candidates[index].decision.primitive_actions()
                == incumbent.primitive_actions()
            ):
                self.counts["identical_action_abstentions"] += 1
                continue
            if not self.support.passes(
                signatures[index], torch.from_numpy(supports[index])
            ):
                self.counts["support_abstentions"] += 1
                continue
            self.last_supported_alternatives += 1
            probability = self.calibration.isotonic.predict(
                float(prediction["unsafe_mean"][index])
            )
            if (
                float(delta_lcb[index]) < 0.0
                or float(absolute_lcb[index]) < 0.0
                or probability > self.calibration.probability_threshold
            ):
                self.counts["solvency_abstentions"] += 1
                continue
            if float(score_lcb[index]) <= self.calibration.score_threshold:
                self.counts["score_abstentions"] += 1
                continue
            eligible.append(index)
        self.last_certified_alternatives = len(eligible)
        self.counts["certified_candidates"] += len(eligible)
        if not eligible:
            self.counts["state_abstentions"] += 1
            return incumbent
        if (
            self.counts["candidate_queries"] - self._last_override_shot
            < self.minimum_override_gap_shots
        ):
            self.counts["rate_abstentions"] += 1
            return incumbent
        best_key = max(
            (
                float(risk_lcb[index]),
                float(score_lcb[index]),
            )
            for index in eligible
        )
        best = [
            index
            for index in eligible
            if math.isclose(
                float(risk_lcb[index]),
                best_key[0],
                rel_tol=1e-7,
                abs_tol=1e-8,
            )
            and math.isclose(
                float(score_lcb[index]),
                best_key[1],
                rel_tol=1e-7,
                abs_tol=1e-8,
            )
        ]
        if len(best) != 1:
            self.counts["lexicographic_tie_abstentions"] += 1
            return incumbent
        selected = best[0]
        decision = candidates[selected].decision
        if not _commit_base_decision(
            self.base_policy, observation, incumbent, decision
        ):
            self.counts["rebind_abstentions"] += 1
            return incumbent
        self._last_override_shot = self.counts["candidate_queries"]
        self.last_selected_ordinal = candidates[selected].ordinal
        self.counts["overrides"] += 1
        self.counts[f"override_geometry/{candidates[selected].geometry.name}"] += 1
        self.counts[f"override_pair/{candidates[selected].pair.category}"] += 1
        return decision

    def statistics(self) -> dict[str, int]:
        return {
            **dict(sorted(self.counts.items())),
            "teacher_queries": 0,
            "branch_simulated_ticks": 0,
            "clone_restore_calls": 0,
        }


def planner_config() -> JointPlannerConfig:
    return JointPlannerConfig(
        pair_cap=3,
        geometry_cap=5,
        horizons=(48, 160, 512),
        cooldown_ticks=16,
        velocity_lead_ticks=1.0,
        ticks_per_second=50.0,
        require_pristine_source=True,
    )


def teacher_planner_config() -> JointPlannerConfig:
    return JointPlannerConfig(
        pair_cap=3,
        geometry_cap=5,
        horizons=(48, 160),
        cooldown_ticks=16,
        velocity_lead_ticks=1.0,
        ticks_per_second=50.0,
        require_pristine_source=True,
    )


__all__ = [
    "AllOutcomeCollectorPolicy",
    "BarrierCalibration",
    "BranchTrace",
    "BranchTraceEnvironment",
    "ConservativeResidualPolicy",
    "FROZEN_V5_PATH",
    "FROZEN_V5_SHA256",
    "INPUT_DIM",
    "IsotonicCalibration",
    "LOCKED_PROTOCOL_PATH",
    "LOCKED_PROTOCOL_SHA256",
    "R3G_OFFLINE_VERSION",
    "RecordedSearch",
    "RecordingJointSearch",
    "ResidualValueNet",
    "SolvencyOutcome",
    "SupportEnvelope",
    "TRUSTED_RUNTIME_PATH",
    "TRUSTED_RUNTIME_SHA256",
    "TrainedEnsemble",
    "branch_labels",
    "canonical_sha256",
    "encode_candidate",
    "fit_barrier",
    "fit_isotonic",
    "fit_support",
    "planner_config",
    "trace_solvency",
    "train_ensemble",
    "teacher_planner_config",
    "verify_trusted_identities",
]
