"""Deterministic, independently resumable R3b evaluation shards."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from torch import Tensor

from .curriculum import SnapshotBlobStore
from .models import RecurrentActorCritic
from .r3b_evaluation import (
    EpisodeMetrics,
    EvaluationReport,
    EvaluationSuite,
    ScriptedBaselineSpec,
    evaluate_recurrent_policy,
    evaluate_recurrent_policy_vectorized,
    evaluate_scripted_baseline,
)

EvaluationCell = tuple[str, int]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        and value != "0" * 64
    )


def evaluation_cells(suite: EvaluationSuite) -> tuple[EvaluationCell, ...]:
    """Return the suite's canonical snapshot-major cell order."""

    if not isinstance(suite, EvaluationSuite):
        raise TypeError("evaluation cells require a typed suite")
    return tuple(
        (snapshot_id, repetition)
        for snapshot_id in suite.snapshot_ids
        for repetition in range(suite.repetitions)
    )


@dataclass(frozen=True, slots=True)
class EvaluationShardPlan:
    suite_sha256: str
    shard_count: int
    shard_index: int
    cells: tuple[EvaluationCell, ...]
    version: str = "r3b-evaluation-shard-plan-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-evaluation-shard-plan-v1"
            or not _is_sha256(self.suite_sha256)
            or isinstance(self.shard_count, bool)
            or not isinstance(self.shard_count, Integral)
            or self.shard_count <= 0
            or isinstance(self.shard_index, bool)
            or not isinstance(self.shard_index, Integral)
            or not 0 <= self.shard_index < self.shard_count
            or not isinstance(self.cells, tuple)
            or not self.cells
            or any(
                not isinstance(cell, tuple)
                or len(cell) != 2
                or not isinstance(cell[0], str)
                or not cell[0]
                or isinstance(cell[1], bool)
                or not isinstance(cell[1], Integral)
                or cell[1] < 0
                for cell in self.cells
            )
            or len(set(self.cells)) != len(self.cells)
        ):
            raise ValueError("evaluation shard plan is malformed")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "suite_sha256": self.suite_sha256,
            "shard_count": int(self.shard_count),
            "shard_index": int(self.shard_index),
            "cells": [
                {"snapshot_id": snapshot_id, "repetition": int(repetition)}
                for snapshot_id, repetition in self.cells
            ],
        }

    @classmethod
    def from_manifest(cls, value: object) -> EvaluationShardPlan:
        if not isinstance(value, dict) or set(value) != {
            "version",
            "suite_sha256",
            "shard_count",
            "shard_index",
            "cells",
        }:
            raise ValueError("evaluation shard plan schema differs")
        cells = value["cells"]
        if not isinstance(cells, list) or any(
            not isinstance(cell, dict)
            or set(cell) != {"snapshot_id", "repetition"}
            or type(cell["snapshot_id"]) is not str
            or type(cell["repetition"]) is not int
            for cell in cells
        ):
            raise ValueError("evaluation shard plan cells are malformed")
        try:
            result = cls(
                value["suite_sha256"],
                value["shard_count"],
                value["shard_index"],
                tuple((cell["snapshot_id"], cell["repetition"]) for cell in cells),
                value["version"],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation shard plan is malformed") from exc
        if result.manifest() != value:
            raise ValueError("evaluation shard plan is noncanonical")
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def plan_evaluation_shards(
    suite: EvaluationSuite, shard_count: int
) -> tuple[EvaluationShardPlan, ...]:
    """Round-robin canonical cells into stable, nonempty shards."""

    cells = evaluation_cells(suite)
    if (
        isinstance(shard_count, bool)
        or not isinstance(shard_count, Integral)
        or not 0 < shard_count <= len(cells)
    ):
        raise ValueError("shard count must fit the number of evaluation cells")
    return tuple(
        EvaluationShardPlan(
            suite.sha256,
            int(shard_count),
            shard_index,
            cells[shard_index::shard_count],
        )
        for shard_index in range(shard_count)
    )


def _validate_plan(suite: EvaluationSuite, shard: EvaluationShardPlan) -> None:
    if not isinstance(shard, EvaluationShardPlan):
        raise TypeError("evaluation shard must be a typed plan")
    expected = plan_evaluation_shards(suite, shard.shard_count)[shard.shard_index]
    if shard != expected:
        raise ValueError("evaluation shard disagrees with its full suite")


@dataclass(frozen=True, slots=True)
class EvaluationShardReport:
    shard: EvaluationShardPlan
    report: EvaluationReport
    version: str = "r3b-evaluation-shard-report-v1"

    def __post_init__(self) -> None:
        if (
            self.version != "r3b-evaluation-shard-report-v1"
            or not isinstance(self.shard, EvaluationShardPlan)
            or not isinstance(self.report, EvaluationReport)
            or self.report.suite_sha256 != self.shard.suite_sha256
            or tuple(
                (episode.snapshot_id, int(episode.repetition))
                for episode in self.report.episodes
            )
            != self.shard.cells
        ):
            raise ValueError("evaluation shard report disagrees with its plan")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "shard_plan_sha256": self.shard.sha256,
            "report_sha256": self.report.sha256,
        }

    @classmethod
    def from_manifest(
        cls,
        value: object,
        *,
        shard: EvaluationShardPlan,
        report: EvaluationReport,
    ) -> EvaluationShardReport:
        if (
            not isinstance(value, dict)
            or set(value) != {"version", "shard_plan_sha256", "report_sha256"}
            or value.get("shard_plan_sha256") != shard.sha256
            or value.get("report_sha256") != report.sha256
        ):
            raise ValueError("evaluation shard report references differ")
        try:
            result = cls(shard, report, value["version"])
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation shard report is malformed") from exc
        if result.manifest() != value:
            raise ValueError("evaluation shard report is noncanonical")
        return result

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.manifest())


def evaluate_scripted_shard(
    simulator: Any,
    store: SnapshotBlobStore,
    suite: EvaluationSuite,
    baseline: ScriptedBaselineSpec,
    shard: EvaluationShardPlan,
    *,
    evaluator_sha256: str,
    expected_assignment_sha256: str,
    execution_identity_sha256: str,
) -> EvaluationShardReport:
    _validate_plan(suite, shard)
    report = evaluate_scripted_baseline(
        simulator,
        store,
        suite,
        baseline,
        evaluator_sha256=evaluator_sha256,
        expected_assignment_sha256=expected_assignment_sha256,
        execution_identity_sha256=execution_identity_sha256,
        cells=shard.cells,
    )
    return EvaluationShardReport(shard, report)


def evaluate_recurrent_shard(
    simulator: Any,
    store: SnapshotBlobStore,
    suite: EvaluationSuite,
    model: RecurrentActorCritic,
    encoder: Any,
    kind_mask: Tensor,
    wait_mask: Tensor,
    shard: EvaluationShardPlan,
    *,
    evaluator_sha256: str,
    expected_assignment_sha256: str,
    execution_identity_sha256: str,
) -> EvaluationShardReport:
    _validate_plan(suite, shard)
    report = evaluate_recurrent_policy(
        simulator,
        store,
        suite,
        model,
        encoder,
        kind_mask,
        wait_mask,
        evaluator_sha256=evaluator_sha256,
        expected_assignment_sha256=expected_assignment_sha256,
        execution_identity_sha256=execution_identity_sha256,
        cells=shard.cells,
    )
    return EvaluationShardReport(shard, report)


def evaluate_recurrent_vector_shard(
    simulator: Any,
    store: SnapshotBlobStore,
    suite: EvaluationSuite,
    model: RecurrentActorCritic,
    encoder: Any,
    kind_mask: Tensor,
    wait_mask: Tensor,
    shard: EvaluationShardPlan,
    *,
    evaluator_sha256: str,
    expected_assignment_sha256: str,
    execution_identity_sha256: str,
) -> EvaluationShardReport:
    """Evaluate one frozen shard across a subset-capable simulator vector."""

    _validate_plan(suite, shard)
    report = evaluate_recurrent_policy_vectorized(
        simulator,
        store,
        suite,
        model,
        encoder,
        kind_mask,
        wait_mask,
        evaluator_sha256=evaluator_sha256,
        expected_assignment_sha256=expected_assignment_sha256,
        execution_identity_sha256=execution_identity_sha256,
        cells=shard.cells,
    )
    return EvaluationShardReport(shard, report)


def merge_evaluation_shards(
    suite: EvaluationSuite,
    shard_reports: tuple[EvaluationShardReport, ...],
) -> EvaluationReport:
    """Merge one complete deterministic partition into a canonical report."""

    if not isinstance(suite, EvaluationSuite):
        raise TypeError("shard merge requires a typed evaluation suite")
    if (
        not isinstance(shard_reports, tuple)
        or not shard_reports
        or any(not isinstance(value, EvaluationShardReport) for value in shard_reports)
    ):
        raise ValueError("shard merge requires typed shard reports")
    shard_count = shard_reports[0].shard.shard_count
    expected_plans = plan_evaluation_shards(suite, shard_count)
    indexed: dict[int, EvaluationShardReport] = {}
    for value in shard_reports:
        index = value.shard.shard_index
        if index in indexed:
            raise ValueError("evaluation shard index is duplicated")
        if (
            value.shard.shard_count != shard_count
            or index >= len(expected_plans)
            or value.shard != expected_plans[index]
        ):
            raise ValueError("evaluation shard is foreign to the merge plan")
        indexed[index] = value
    if set(indexed) != set(range(shard_count)):
        raise ValueError("evaluation shard set is incomplete")

    ordered = tuple(indexed[index] for index in range(shard_count))
    identities = {
        (
            value.report.suite_sha256,
            value.report.policy_sha256,
            value.report.evaluator_sha256,
            value.report.backend_identity_sha256,
        )
        for value in ordered
    }
    expected_identity = {
        (
            suite.sha256,
            ordered[0].report.policy_sha256,
            ordered[0].report.evaluator_sha256,
            suite.runtime_identity_sha256,
        )
    }
    if identities != expected_identity:
        raise ValueError("evaluation shard report identities disagree")

    episodes: dict[EvaluationCell, EpisodeMetrics] = {}
    for value in ordered:
        for episode in value.report.episodes:
            cell = (episode.snapshot_id, int(episode.repetition))
            if cell in episodes:
                raise ValueError("evaluation shard cells overlap")
            if episode.policy_seed != suite.episode_seed(*cell):
                raise ValueError("evaluation shard cell seed disagrees with the suite")
            episodes[cell] = episode
    cells = evaluation_cells(suite)
    if set(episodes) != set(cells):
        raise ValueError("evaluation shard cells are missing or foreign")

    execution_identity_sha256 = _canonical_sha256(
        {
            "version": "r3b-evaluation-shard-merge-execution-v1",
            "suite_sha256": suite.sha256,
            "shard_count": shard_count,
            "shard_reports": [
                {
                    "shard_index": value.shard.shard_index,
                    "shard_report_sha256": value.sha256,
                }
                for value in ordered
            ],
        }
    )
    first = ordered[0].report
    return EvaluationReport(
        suite.sha256,
        first.policy_sha256,
        first.evaluator_sha256,
        first.backend_identity_sha256,
        execution_identity_sha256,
        tuple(episodes[cell] for cell in cells),
    )
