"""Correct, query-denominated threshold fitting and frozen-threshold audit.

This module is intentionally separate from :mod:`r3i_g3_evidence`: that
evidence core is immutable while it is audited.  Every rule and report binds
the exact live evidence-core source used for canonical JSON, hashing, and
sealing.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from . import r3i_g3_evidence as _core


EvidenceError = _core.EvidenceError

DEPENDENCY_SCHEMA = "irisu.r3i.g3.evidence-core-dependency.v2"
SOURCE_BINDING_SCHEMA = "irisu.r3i.g3.threshold-module-source-binding.v2"
OBSERVATION_SET_SCHEMA = "irisu.r3i.g3.threshold-observations.v2"
THRESHOLD_RULE_SCHEMA = "irisu.r3i.g3.threshold-rule.v2"
FIT_REPORT_SCHEMA = "irisu.r3i.g3.threshold-fit.v2"
AUDIT_REPORT_SCHEMA = "irisu.r3i.g3.threshold-fixed-audit.v2"

_CORE_API = "canonical-json-sha256-seal-record-v1"
_MODULE_API = "query-denominated-fit-and-fixed-audit-v2"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_QUERY_KEYS = {"seed", "query_id"}
_ITEM_KEYS = {"seed", "query_id", "item_id"}
_OBSERVATION_KEYS = {
    "seed",
    "query_id",
    "item_id",
    "score",
    "eligible",
    "acceptable",
}
_EMPTY_SHA256 = _core.sha256_json([])


def _require_exact(value: Any, expected: type, field: str) -> None:
    if type(value) is not expected:
        raise EvidenceError(f"{field} must be exact {expected.__name__}")


def _require_sha256(value: Any, field: str) -> str:
    _require_exact(value, str, field)
    if _SHA256_RE.fullmatch(value) is None:
        raise EvidenceError(f"{field} must be a lowercase SHA-256")
    return value


def _require_nonnegative_int(value: Any, field: str) -> int:
    _require_exact(value, int, field)
    if value < 0:
        raise EvidenceError(f"{field} must be non-negative")
    return value


def _identifier(value: Any, field: str) -> str:
    _require_exact(value, str, field)
    if not value or len(value) > 512 or "\x00" in value:
        raise EvidenceError(f"{field} is empty or unsafe")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise EvidenceError(f"{field} is not valid UTF-8 text") from exc
    return value


def _seed(value: Any, field: str) -> int:
    _require_exact(value, int, field)
    if value < 0:
        raise EvidenceError(f"{field} must be non-negative")
    return value


def _score(value: Any, field: str) -> float:
    _require_exact(value, float, field)
    if not math.isfinite(value):
        raise EvidenceError(f"{field} must be finite")
    # >= cannot distinguish the two signed zeros, so accepting both encodings
    # would make exact-float tie blocks ambiguous.
    if value == 0.0 and math.copysign(1.0, value) < 0.0:
        raise EvidenceError(f"{field} cannot be negative zero")
    return value


def _dependency() -> dict[str, Any]:
    source = Path(_core.__file__).resolve()
    binding = _core.validate_regular_file(source, root=source.parent)
    return {
        "schema": DEPENDENCY_SCHEMA,
        "module": "irisu_pointer.r3i_g3_evidence",
        "api": _CORE_API,
        "source_sha256": binding["sha256"],
        "canonical_json_encoding": _core.CANONICAL_JSON_ENCODING,
    }


def _validate_dependency(value: Any) -> dict[str, Any]:
    _require_exact(value, dict, "evidence_core_dependency")
    if value != _dependency():
        raise EvidenceError("evidence-core dependency identity/version mismatch")
    return dict(value)


def build_threshold_source_binding_v2(
    *, expected_source_sha256: str
) -> dict[str, Any]:
    """Bind caller-supplied expected bytes to the descriptor-safe live module."""

    expected_source_sha256 = _require_sha256(
        expected_source_sha256, "expected_source_sha256"
    )
    source = Path(__file__).resolve()
    binding = _core.validate_regular_file(
        source,
        root=source.parent,
        expected_sha256=expected_source_sha256,
    )
    return _core.seal_record(
        {
            "schema": SOURCE_BINDING_SCHEMA,
            "module": "irisu_pointer.r3i_g3_threshold_v2",
            "api": _MODULE_API,
            "source_filename": source.name,
            "source_size": binding["size"],
            "source_sha256": binding["sha256"],
            "evidence_core_dependency": _dependency(),
        },
        "binding_sha256",
    )


def _validate_source_binding(value: Any) -> dict[str, Any]:
    verified = _core.verify_sealed_record(value, "binding_sha256")
    required = {
        "schema",
        "module",
        "api",
        "source_filename",
        "source_size",
        "source_sha256",
        "evidence_core_dependency",
        "binding_sha256",
    }
    if set(verified) != required:
        raise EvidenceError("threshold module source binding has missing or extra fields")
    _require_sha256(verified["source_sha256"], "source binding SHA-256")
    _require_nonnegative_int(verified["source_size"], "source binding size")
    _validate_dependency(verified["evidence_core_dependency"])
    if (
        verified["schema"] != SOURCE_BINDING_SCHEMA
        or verified["module"] != "irisu_pointer.r3i_g3_threshold_v2"
        or verified["api"] != _MODULE_API
        or verified["source_filename"] != Path(__file__).name
    ):
        raise EvidenceError("threshold module source binding identity/version mismatch")
    source = Path(__file__).resolve()
    live = _core.validate_regular_file(
        source,
        root=source.parent,
        expected_sha256=verified["source_sha256"],
        expected_size=verified["source_size"],
    )
    expected = _core.seal_record(
        {
            "schema": SOURCE_BINDING_SCHEMA,
            "module": "irisu_pointer.r3i_g3_threshold_v2",
            "api": _MODULE_API,
            "source_filename": source.name,
            "source_size": live["size"],
            "source_sha256": live["sha256"],
            "evidence_core_dependency": _dependency(),
        },
        "binding_sha256",
    )
    if verified != expected:
        raise EvidenceError("threshold module source binding is not canonical/live")
    return verified


def build_threshold_rule_v2(
    *, source_binding: Any
) -> dict[str, Any]:
    """Build the single accepted zero-error, five-percent-coverage rule."""

    source_binding = _validate_source_binding(source_binding)
    return _core.seal_record(
        {
            "schema": THRESHOLD_RULE_SCHEMA,
            "evidence_core_dependency": _dependency(),
            "threshold_module_source_binding": source_binding,
            "comparator": ">=",
            "candidate_set": "every-eligible-exact-float-tie-block",
            "tie_policy": "whole-block-no-split",
            "objective": "longest-zero-error-prefix",
            "coverage_unit": "unique-query",
            "coverage_denominator": "complete-query-universe",
            "minimum_query_coverage_numerator": 1,
            "minimum_query_coverage_denominator": 20,
            "maximum_bad_candidates": 0,
            "maximum_bad_seeds": 0,
        },
        "rule_sha256",
    )


def _validate_rule(value: Any) -> dict[str, Any]:
    verified = _core.verify_sealed_record(value, "rule_sha256")
    if "threshold_module_source_binding" not in verified:
        raise EvidenceError("threshold-v2 rule is missing its module source binding")
    source_binding = _validate_source_binding(
        verified["threshold_module_source_binding"]
    )
    if set(verified) != set(build_threshold_rule_v2(source_binding=source_binding)):
        raise EvidenceError("threshold-v2 rule has missing or extra fields")
    _validate_dependency(verified["evidence_core_dependency"])
    if verified != build_threshold_rule_v2(source_binding=source_binding):
        raise EvidenceError("threshold-v2 rule is not the fixed rule")
    return verified


def _query_key(row: dict[str, Any]) -> tuple[int, str]:
    return row["seed"], row["query_id"]


def _item_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return row["seed"], row["query_id"], row["item_id"]


def _validate_id_row(
    row: Any,
    *,
    keys: set[str],
    field: str,
) -> dict[str, Any]:
    _require_exact(row, dict, field)
    if set(row) != keys:
        raise EvidenceError(f"{field} has missing or extra identity fields")
    result = {
        "seed": _seed(row["seed"], f"{field}.seed"),
        "query_id": _identifier(row["query_id"], f"{field}.query_id"),
    }
    if "item_id" in keys:
        result["item_id"] = _identifier(row["item_id"], f"{field}.item_id")
    return result


def _observation_bundle(
    observations: Any,
    *,
    expected_items: Any,
    query_universe: Any,
) -> dict[str, Any]:
    _require_exact(query_universe, list, "query_universe")
    if not query_universe:
        raise EvidenceError("query universe cannot be empty")
    queries: list[dict[str, Any]] = []
    seen_queries: set[tuple[int, str]] = set()
    for index, row in enumerate(query_universe):
        normalized = _validate_id_row(
            row, keys=_QUERY_KEYS, field=f"query_universe[{index}]"
        )
        key = _query_key(normalized)
        if key in seen_queries:
            raise EvidenceError("query universe contains a duplicate query")
        seen_queries.add(key)
        queries.append(normalized)
    queries.sort(key=_query_key)

    _require_exact(expected_items, list, "expected_items")
    if not expected_items:
        raise EvidenceError("expected item universe cannot be empty")
    items: list[dict[str, Any]] = []
    seen_items: set[tuple[int, str, str]] = set()
    queries_with_items: set[tuple[int, str]] = set()
    for index, row in enumerate(expected_items):
        normalized = _validate_id_row(
            row, keys=_ITEM_KEYS, field=f"expected_items[{index}]"
        )
        query = _query_key(normalized)
        key = _item_key(normalized)
        if query not in seen_queries:
            raise EvidenceError("expected item names a query outside the universe")
        if key in seen_items:
            raise EvidenceError("expected item universe contains a duplicate item")
        seen_items.add(key)
        queries_with_items.add(query)
        items.append(normalized)
    if queries_with_items != seen_queries:
        raise EvidenceError("every query must have at least one expected item")
    items.sort(key=_item_key)

    _require_exact(observations, list, "observations")
    rows: list[dict[str, Any]] = []
    seen_observations: set[tuple[int, str, str]] = set()
    for index, row in enumerate(observations):
        field = f"observations[{index}]"
        _require_exact(row, dict, field)
        if set(row) != _OBSERVATION_KEYS:
            raise EvidenceError(f"{field} has missing or extra fields")
        normalized = {
            "seed": _seed(row["seed"], f"{field}.seed"),
            "query_id": _identifier(row["query_id"], f"{field}.query_id"),
            "item_id": _identifier(row["item_id"], f"{field}.item_id"),
            "score": _score(row["score"], f"{field}.score"),
            "eligible": row["eligible"],
            "acceptable": row["acceptable"],
        }
        _require_exact(normalized["eligible"], bool, f"{field}.eligible")
        _require_exact(normalized["acceptable"], bool, f"{field}.acceptable")
        key = _item_key(normalized)
        if key in seen_observations:
            raise EvidenceError("observations contain a duplicate item")
        seen_observations.add(key)
        rows.append(normalized)
    if seen_observations != seen_items:
        missing = sorted(seen_items - seen_observations)
        extra = sorted(seen_observations - seen_items)
        raise EvidenceError(
            f"observations do not close over expected items "
            f"(missing={missing}, extra={extra})"
        )
    rows.sort(key=_item_key)

    payload = {
        "schema": OBSERVATION_SET_SCHEMA,
        "query_universe": queries,
        "expected_items": items,
        "observations": rows,
    }
    seeds = sorted({query["seed"] for query in queries})
    return {
        "seeds": seeds,
        "queries": queries,
        "items": items,
        "observations": rows,
        "query_keys": seen_queries,
        "item_keys": seen_items,
        "identity_sha256": _core.sha256_json(payload),
        "seed_inventory_sha256": _core.sha256_json(seeds),
        "query_universe_sha256": _core.sha256_json(queries),
        "item_universe_sha256": _core.sha256_json(items),
        "observations_sha256": _core.sha256_json(rows),
    }


def _tie_blocks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        if row["eligible"]:
            groups.setdefault(row["score"], []).append(row)
    result = []
    for score in sorted(groups, reverse=True):
        block_rows = sorted(groups[score], key=_item_key)
        items = [
            {
                "seed": row["seed"],
                "query_id": row["query_id"],
                "item_id": row["item_id"],
                "acceptable": row["acceptable"],
            }
            for row in block_rows
        ]
        result.append(
            {
                "score": score,
                "score_hex": score.hex(),
                "rows": block_rows,
                "count": len(block_rows),
                "acceptable_count": sum(row["acceptable"] for row in block_rows),
                "bad_count": sum(not row["acceptable"] for row in block_rows),
                "sha256": _core.sha256_json(
                    {"score_hex": score.hex(), "items": items}
                ),
            }
        )
    return result


def _selection_metrics(
    selected: list[dict[str, Any]],
    *,
    query_count: int,
) -> dict[str, Any]:
    bad = [row for row in selected if not row["acceptable"]]
    selected_queries = {_query_key(row) for row in selected}
    selected_seeds = {row["seed"] for row in selected}
    bad_seeds = {row["seed"] for row in bad}
    identities = [
        {"seed": row["seed"], "query_id": row["query_id"], "item_id": row["item_id"]}
        for row in sorted(selected, key=_item_key)
    ]
    coverage_passed = len(selected_queries) * 20 >= query_count
    return {
        "selected_candidate_count": len(selected),
        "selected_acceptable_count": len(selected) - len(bad),
        "selected_bad_candidate_count": len(bad),
        "selected_seed_count": len(selected_seeds),
        "selected_bad_seed_count": len(bad_seeds),
        "selected_query_count": len(selected_queries),
        "coverage_numerator": len(selected_queries),
        "coverage_denominator": query_count,
        "coverage_passed": coverage_passed,
        "selection_sha256": _core.sha256_json(identities),
    }


def _failure_reasons(
    *,
    has_threshold: bool,
    metrics: dict[str, Any],
) -> list[str]:
    reasons = []
    if not has_threshold:
        reasons.append("no-zero-error-eligible-prefix")
    if metrics["selected_bad_candidate_count"]:
        reasons.append("bad-candidate-selected")
    if metrics["selected_bad_seed_count"]:
        reasons.append("bad-seed-selected")
    if not metrics["coverage_passed"]:
        reasons.append("unique-query-coverage-below-five-percent")
    return reasons


def fit_threshold_calibration(
    observations: Any,
    *,
    expected_items: Any,
    query_universe: Any,
    rule: Any,
) -> dict[str, Any]:
    """Fit the longest zero-error prefix over every eligible score tie block."""

    fixed_rule = _validate_rule(rule)
    bundle = _observation_bundle(
        observations,
        expected_items=expected_items,
        query_universe=query_universe,
    )
    blocks = _tie_blocks(bundle["observations"])

    prefix: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    best_block: dict[str, Any] | None = None
    for block in blocks:
        prefix.extend(block["rows"])
        metrics = _selection_metrics(prefix, query_count=len(bundle["queries"]))
        if (
            metrics["selected_bad_candidate_count"]
            <= fixed_rule["maximum_bad_candidates"]
            and metrics["selected_bad_seed_count"]
            <= fixed_rule["maximum_bad_seeds"]
        ):
            # Each whole block strictly lengthens the prefix, so the last valid
            # assignment is the unique deterministic longest valid prefix.
            best_rows = list(prefix)
            best_block = block

    metrics = _selection_metrics(best_rows, query_count=len(bundle["queries"]))
    has_threshold = best_block is not None
    passed = (
        has_threshold
        and metrics["selected_bad_candidate_count"] == 0
        and metrics["selected_bad_seed_count"] == 0
        and metrics["coverage_passed"]
    )
    reasons = _failure_reasons(has_threshold=has_threshold, metrics=metrics)
    eligible_count = sum(row["eligible"] for row in bundle["observations"])
    return _core.seal_record(
        {
            "schema": FIT_REPORT_SCHEMA,
            "evidence_core_dependency": _dependency(),
            "threshold_module_source_binding": fixed_rule[
                "threshold_module_source_binding"
            ],
            "passed": passed,
            "terminal": not passed,
            "failure_reasons": reasons,
            "rule": fixed_rule,
            "input_identity_sha256": bundle["identity_sha256"],
            "seed_inventory": bundle["seeds"],
            "seed_inventory_sha256": bundle["seed_inventory_sha256"],
            "query_inventory": bundle["queries"],
            "item_inventory": bundle["items"],
            "query_universe_sha256": bundle["query_universe_sha256"],
            "item_universe_sha256": bundle["item_universe_sha256"],
            "observations_sha256": bundle["observations_sha256"],
            "seed_count": len(bundle["seeds"]),
            "query_count": len(bundle["queries"]),
            "item_count": len(bundle["items"]),
            "observation_count": len(bundle["observations"]),
            "eligible_count": eligible_count,
            "unique_eligible_score_count": len(blocks),
            "evaluated_tie_block_count": len(blocks),
            "tie_blocks_sha256": _core.sha256_json(
                [
                    {
                        "score_hex": block["score_hex"],
                        "count": block["count"],
                        "acceptable_count": block["acceptable_count"],
                        "bad_count": block["bad_count"],
                        "sha256": block["sha256"],
                    }
                    for block in blocks
                ]
            ),
            "threshold_hex": None if best_block is None else best_block["score_hex"],
            "threshold_tie_block_score_hex": (
                None if best_block is None else best_block["score_hex"]
            ),
            "threshold_tie_block_count": (
                0 if best_block is None else best_block["count"]
            ),
            "threshold_tie_block_sha256": (
                _EMPTY_SHA256 if best_block is None else best_block["sha256"]
            ),
            **metrics,
        },
        "calibration_sha256",
    )


def _validate_fit_report(report: Any) -> dict[str, Any]:
    verified = _core.verify_sealed_record(report, "calibration_sha256")
    required = {
        "schema",
        "evidence_core_dependency",
        "threshold_module_source_binding",
        "passed",
        "terminal",
        "failure_reasons",
        "rule",
        "input_identity_sha256",
        "seed_inventory",
        "seed_inventory_sha256",
        "query_inventory",
        "item_inventory",
        "query_universe_sha256",
        "item_universe_sha256",
        "observations_sha256",
        "seed_count",
        "query_count",
        "item_count",
        "observation_count",
        "eligible_count",
        "unique_eligible_score_count",
        "evaluated_tie_block_count",
        "tie_blocks_sha256",
        "threshold_hex",
        "threshold_tie_block_score_hex",
        "threshold_tie_block_count",
        "threshold_tie_block_sha256",
        "selected_candidate_count",
        "selected_acceptable_count",
        "selected_bad_candidate_count",
        "selected_seed_count",
        "selected_bad_seed_count",
        "selected_query_count",
        "coverage_numerator",
        "coverage_denominator",
        "coverage_passed",
        "selection_sha256",
        "calibration_sha256",
    }
    if set(verified) != required or verified["schema"] != FIT_REPORT_SCHEMA:
        raise EvidenceError("fit calibration has missing, extra, or wrong-schema fields")
    _validate_dependency(verified["evidence_core_dependency"])
    source_binding = _validate_source_binding(
        verified["threshold_module_source_binding"]
    )
    fixed_rule = _validate_rule(verified["rule"])
    if source_binding != fixed_rule["threshold_module_source_binding"]:
        raise EvidenceError("fit calibration source/rule binding mismatch")
    for field in (
        "input_identity_sha256",
        "seed_inventory_sha256",
        "query_universe_sha256",
        "item_universe_sha256",
        "observations_sha256",
        "tie_blocks_sha256",
        "threshold_tie_block_sha256",
        "selection_sha256",
        "calibration_sha256",
    ):
        _require_sha256(verified[field], field)
    for field in (
        "seed_count",
        "query_count",
        "item_count",
        "observation_count",
        "eligible_count",
        "unique_eligible_score_count",
        "evaluated_tie_block_count",
        "threshold_tie_block_count",
        "selected_candidate_count",
        "selected_acceptable_count",
        "selected_bad_candidate_count",
        "selected_seed_count",
        "selected_bad_seed_count",
        "selected_query_count",
        "coverage_numerator",
        "coverage_denominator",
    ):
        _require_nonnegative_int(verified[field], field)

    _require_exact(verified["seed_inventory"], list, "seed_inventory")
    seeds = [
        _seed(value, f"seed_inventory[{index}]")
        for index, value in enumerate(verified["seed_inventory"])
    ]
    if seeds != sorted(set(seeds)):
        raise EvidenceError("fit seed inventory must be sorted and unique")
    _require_exact(verified["query_inventory"], list, "query_inventory")
    queries = [
        _validate_id_row(row, keys=_QUERY_KEYS, field=f"query_inventory[{index}]")
        for index, row in enumerate(verified["query_inventory"])
    ]
    if queries != sorted(queries, key=_query_key):
        raise EvidenceError("fit query inventory must be canonically sorted")
    if len({_query_key(row) for row in queries}) != len(queries):
        raise EvidenceError("fit query inventory contains duplicates")
    _require_exact(verified["item_inventory"], list, "item_inventory")
    items = [
        _validate_id_row(row, keys=_ITEM_KEYS, field=f"item_inventory[{index}]")
        for index, row in enumerate(verified["item_inventory"])
    ]
    if items != sorted(items, key=_item_key):
        raise EvidenceError("fit item inventory must be canonically sorted")
    if len({_item_key(row) for row in items}) != len(items):
        raise EvidenceError("fit item inventory contains duplicates")
    query_keys = {_query_key(row) for row in queries}
    if (
        set(seeds) != {row["seed"] for row in queries}
        or any(_query_key(row) not in query_keys for row in items)
        or {_query_key(row) for row in items} != query_keys
        or verified["seed_count"] != len(seeds)
        or verified["query_count"] != len(queries)
        or verified["item_count"] != len(items)
        or verified["seed_inventory_sha256"] != _core.sha256_json(seeds)
        or verified["query_universe_sha256"] != _core.sha256_json(queries)
        or verified["item_universe_sha256"] != _core.sha256_json(items)
    ):
        raise EvidenceError("fit inventory counts, closure, or hashes are inconsistent")
    if verified["query_count"] == 0:
        raise EvidenceError("fit calibration has an ambiguous zero-query denominator")
    if (
        verified["item_count"] != verified["observation_count"]
        or verified["eligible_count"] > verified["item_count"]
        or verified["unique_eligible_score_count"]
        != verified["evaluated_tie_block_count"]
        or verified["selected_candidate_count"] > verified["eligible_count"]
        or verified["selected_acceptable_count"]
        + verified["selected_bad_candidate_count"]
        != verified["selected_candidate_count"]
        or verified["selected_query_count"] > verified["query_count"]
        or verified["selected_seed_count"] > verified["selected_candidate_count"]
        or verified["selected_bad_seed_count"] > verified["selected_seed_count"]
        or verified["coverage_numerator"] != verified["selected_query_count"]
        or verified["coverage_denominator"] != verified["query_count"]
    ):
        raise EvidenceError("fit calibration counts are inconsistent")
    for field in ("passed", "terminal", "coverage_passed"):
        _require_exact(verified[field], bool, field)
    _require_exact(verified["failure_reasons"], list, "failure_reasons")
    if any(type(reason) is not str for reason in verified["failure_reasons"]):
        raise EvidenceError("failure reasons must be exact strings")

    threshold_hex = verified["threshold_hex"]
    if threshold_hex is None:
        if (
            verified["threshold_tie_block_score_hex"] is not None
            or verified["threshold_tie_block_count"] != 0
            or verified["threshold_tie_block_sha256"] != _EMPTY_SHA256
            or verified["selected_candidate_count"] != 0
        ):
            raise EvidenceError("absent threshold has inconsistent tie/selection evidence")
        has_threshold = False
    else:
        _require_exact(threshold_hex, str, "threshold_hex")
        try:
            threshold = float.fromhex(threshold_hex)
        except ValueError as exc:
            raise EvidenceError("threshold_hex is not a float.hex value") from exc
        _score(threshold, "threshold")
        if threshold.hex() != threshold_hex:
            raise EvidenceError("threshold_hex is not canonical")
        if (
            verified["threshold_tie_block_score_hex"] != threshold_hex
            or verified["threshold_tie_block_count"] <= 0
        ):
            raise EvidenceError("threshold tie block is inconsistent")
        has_threshold = True

    coverage_passed = (
        verified["coverage_numerator"] * 20 >= verified["coverage_denominator"]
    )
    if verified["coverage_passed"] != coverage_passed:
        raise EvidenceError("fit calibration coverage decision is inconsistent")
    passed = (
        has_threshold
        and verified["selected_bad_candidate_count"] == 0
        and verified["selected_bad_seed_count"] == 0
        and coverage_passed
    )
    if verified["passed"] != passed or verified["terminal"] == passed:
        raise EvidenceError("fit calibration pass/terminal decision is inconsistent")
    expected_reasons = _failure_reasons(
        has_threshold=has_threshold, metrics=verified
    )
    if verified["failure_reasons"] != expected_reasons:
        raise EvidenceError("fit calibration failure reasons are inconsistent")
    return verified


def verify_fit_threshold_calibration(
    observations: Any,
    report: Any,
    *,
    expected_items: Any,
    query_universe: Any,
    rule: Any,
) -> dict[str, Any]:
    """Recompute and exactly verify a fit calibration from its source inputs."""

    _validate_fit_report(report)
    expected = fit_threshold_calibration(
        observations,
        expected_items=expected_items,
        query_universe=query_universe,
        rule=rule,
    )
    if report != expected:
        raise EvidenceError("fit calibration does not reproduce exactly")
    return dict(report)


def apply_fixed_threshold_audit(
    observations: Any,
    *,
    expected_items: Any,
    query_universe: Any,
    fit_calibration: Any,
    fit_observations: Any,
    fit_expected_items: Any,
    fit_query_universe: Any,
) -> dict[str, Any]:
    """Apply a passed fit threshold to a different audit set without refitting."""

    structurally_valid_fit = _validate_fit_report(fit_calibration)
    fit = verify_fit_threshold_calibration(
        fit_observations,
        fit_calibration,
        expected_items=fit_expected_items,
        query_universe=fit_query_universe,
        rule=structurally_valid_fit["rule"],
    )
    if not fit["passed"]:
        raise EvidenceError("fixed-threshold audit requires a passed fit calibration")
    bundle = _observation_bundle(
        observations,
        expected_items=expected_items,
        query_universe=query_universe,
    )
    if bundle["identity_sha256"] == fit["input_identity_sha256"]:
        raise EvidenceError("fit and audit observation identities must differ")
    if bundle["item_universe_sha256"] == fit["item_universe_sha256"]:
        raise EvidenceError("fit and audit item universes must differ")
    fit_seeds = set(fit["seed_inventory"])
    audit_seeds = set(bundle["seeds"])
    fit_queries = {_query_key(row) for row in fit["query_inventory"]}
    audit_queries = set(bundle["query_keys"])
    fit_items = {_item_key(row) for row in fit["item_inventory"]}
    audit_items = set(bundle["item_keys"])
    if fit_seeds & audit_seeds:
        raise EvidenceError("fit and audit seed inventories overlap")
    if fit_queries & audit_queries:
        raise EvidenceError("fit and audit query-key inventories overlap")
    if fit_items & audit_items:
        raise EvidenceError("fit and audit item-key inventories overlap")

    threshold_hex = fit["threshold_hex"]
    if type(threshold_hex) is not str:
        raise EvidenceError("passed fit calibration has no threshold")
    threshold = float.fromhex(threshold_hex)
    selected = [
        row
        for row in bundle["observations"]
        if row["eligible"] and row["score"] >= threshold
    ]
    metrics = _selection_metrics(selected, query_count=len(bundle["queries"]))
    passed = (
        metrics["selected_bad_candidate_count"] == 0
        and metrics["selected_bad_seed_count"] == 0
        and metrics["coverage_passed"]
    )
    threshold_ties = [
        row
        for row in bundle["observations"]
        if row["eligible"] and row["score"] == threshold
    ]
    tie_identities = [
        {
            "seed": row["seed"],
            "query_id": row["query_id"],
            "item_id": row["item_id"],
            "acceptable": row["acceptable"],
        }
        for row in sorted(threshold_ties, key=_item_key)
    ]
    return _core.seal_record(
        {
            "schema": AUDIT_REPORT_SCHEMA,
            "evidence_core_dependency": _dependency(),
            "threshold_module_source_binding": fit[
                "threshold_module_source_binding"
            ],
            "application_mode": "fixed-fit-threshold-no-search",
            "passed": passed,
            "terminal": not passed,
            "failure_reasons": _failure_reasons(
                has_threshold=True, metrics=metrics
            ),
            "rule_sha256": fit["rule"]["rule_sha256"],
            "fit_calibration_sha256": fit["calibration_sha256"],
            "fit_input_identity_sha256": fit["input_identity_sha256"],
            "fit_seed_inventory": fit["seed_inventory"],
            "fit_seed_inventory_sha256": fit["seed_inventory_sha256"],
            "fit_query_inventory": fit["query_inventory"],
            "fit_item_inventory": fit["item_inventory"],
            "audit_input_identity_sha256": bundle["identity_sha256"],
            "audit_seed_inventory": bundle["seeds"],
            "audit_seed_inventory_sha256": bundle["seed_inventory_sha256"],
            "audit_query_inventory": bundle["queries"],
            "audit_item_inventory": bundle["items"],
            "audit_query_universe_sha256": bundle["query_universe_sha256"],
            "audit_item_universe_sha256": bundle["item_universe_sha256"],
            "audit_observations_sha256": bundle["observations_sha256"],
            "frozen_threshold_hex": threshold_hex,
            "comparator": ">=",
            "query_count": len(bundle["queries"]),
            "item_count": len(bundle["items"]),
            "observation_count": len(bundle["observations"]),
            "eligible_count": sum(
                row["eligible"] for row in bundle["observations"]
            ),
            "frozen_threshold_tie_count": len(threshold_ties),
            "frozen_threshold_tie_sha256": _core.sha256_json(tie_identities),
            "seed_overlap_count": 0,
            "query_key_overlap_count": 0,
            "item_key_overlap_count": 0,
            **metrics,
        },
        "audit_sha256",
    )


def verify_fixed_threshold_audit(
    observations: Any,
    report: Any,
    *,
    expected_items: Any,
    query_universe: Any,
    fit_calibration: Any,
    fit_observations: Any,
    fit_expected_items: Any,
    fit_query_universe: Any,
) -> dict[str, Any]:
    """Recompute and exactly verify a frozen-threshold audit report."""

    _core.verify_sealed_record(report, "audit_sha256")
    expected = apply_fixed_threshold_audit(
        observations,
        expected_items=expected_items,
        query_universe=query_universe,
        fit_calibration=fit_calibration,
        fit_observations=fit_observations,
        fit_expected_items=fit_expected_items,
        fit_query_universe=fit_query_universe,
    )
    if report != expected:
        raise EvidenceError("fixed-threshold audit report does not reproduce exactly")
    return dict(report)


__all__ = [
    "AUDIT_REPORT_SCHEMA",
    "DEPENDENCY_SCHEMA",
    "EvidenceError",
    "FIT_REPORT_SCHEMA",
    "OBSERVATION_SET_SCHEMA",
    "SOURCE_BINDING_SCHEMA",
    "THRESHOLD_RULE_SCHEMA",
    "apply_fixed_threshold_audit",
    "build_threshold_source_binding_v2",
    "build_threshold_rule_v2",
    "fit_threshold_calibration",
    "verify_fit_threshold_calibration",
    "verify_fixed_threshold_audit",
]
