from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_pointer import r3i_g3_evidence as core
from irisu_pointer.r3i_g3_threshold_v3 import (
    AUDIT_REPORT_SCHEMA,
    FIT_REPORT_SCHEMA,
    EvidenceError,
    apply_fixed_threshold_audit_v3,
    build_threshold_rule_v3,
    build_threshold_source_binding_v3,
    fit_threshold_calibration_v3,
    verify_fit_threshold_calibration_v3,
    verify_fixed_threshold_audit_v3,
)


def _sha(label: str) -> str:
    return core.sha256_bytes(label.encode("utf-8"))


def _source_sha() -> str:
    source = ROOT / "python/irisu_pointer/r3i_g3_threshold_v3.py"
    return core.validate_regular_file(source, root=source.parent)["sha256"]


def _inputs(
    query_count: int = 20,
    *,
    seed_offset: int = 0,
    extra_first_items: int = 0,
) -> tuple[list[dict], list[dict], list[dict]]:
    queries = []
    items = []
    observations = []
    for index in range(query_count):
        seed = seed_offset + index
        query_id = f"q-{index}"
        queries.append({"seed": seed, "query_id": query_id})
        count = 1 + (extra_first_items if index == 0 else 0)
        for item_index in range(count):
            item_id = f"i-{item_index}"
            item = {
                "seed": seed,
                "query_id": query_id,
                "item_id": item_id,
            }
            items.append(item)
            observations.append(
                {
                    **item,
                    "score": 0.1,
                    "eligible": False,
                    "acceptable": False,
                }
            )
    return queries, items, observations


def _set(
    observations: list[dict],
    query_index: int,
    item_index: int = 0,
    *,
    score: float,
    eligible: bool,
    acceptable: bool,
) -> None:
    matches = [
        row
        for row in observations
        if row["query_id"] == f"q-{query_index}"
        and row["item_id"] == f"i-{item_index}"
    ]
    if len(matches) != 1:
        raise AssertionError("test observation lookup is not unique")
    matches[0].update(
        score=score,
        eligible=eligible,
        acceptable=acceptable,
    )


def _inventory_shas(
    queries: list[dict],
    items: list[dict],
) -> tuple[str, str, str]:
    canonical_queries = sorted(queries, key=lambda row: (row["seed"], row["query_id"]))
    canonical_items = sorted(
        items,
        key=lambda row: (row["seed"], row["query_id"], row["item_id"]),
    )
    seeds = sorted({row["seed"] for row in canonical_queries})
    return (
        core.sha256_json(seeds),
        core.sha256_json(canonical_queries),
        core.sha256_json(canonical_items),
    )


def _lineage(label: str = "a") -> dict[str, str]:
    return {
        "model_sha256": _sha(f"{label}-model"),
        "partition_sha256": _sha(f"{label}-partition"),
        "training_dataset_sha256": _sha(f"{label}-training-dataset"),
    }


def _rule(lineage: dict[str, str]) -> dict:
    return build_threshold_rule_v3(
        expected_threshold_source_sha256=_source_sha(),
        expected_model_sha256=lineage["model_sha256"],
        expected_partition_sha256=lineage["partition_sha256"],
        expected_training_dataset_sha256=lineage["training_dataset_sha256"],
    )


def _fit(
    queries: list[dict],
    items: list[dict],
    observations: list[dict],
    lineage: dict[str, str],
) -> dict:
    seed_sha, query_sha, item_sha = _inventory_shas(queries, items)
    return fit_threshold_calibration_v3(
        observations,
        expected_items=items,
        query_universe=queries,
        rule=_rule(lineage),
        **lineage,
        expected_threshold_source_sha256=_source_sha(),
        expected_model_sha256=lineage["model_sha256"],
        expected_partition_sha256=lineage["partition_sha256"],
        expected_training_dataset_sha256=lineage["training_dataset_sha256"],
        expected_fit_seed_inventory_sha256=seed_sha,
        expected_fit_query_inventory_sha256=query_sha,
        expected_fit_item_inventory_sha256=item_sha,
    )


def _audit_expectations(
    fit: dict,
    fit_queries: list[dict],
    fit_items: list[dict],
    audit_queries: list[dict],
    audit_items: list[dict],
    lineage: dict[str, str],
) -> dict[str, str]:
    fit_seed, fit_query, fit_item = _inventory_shas(fit_queries, fit_items)
    audit_seed, audit_query, audit_item = _inventory_shas(
        audit_queries,
        audit_items,
    )
    return {
        "expected_fit_calibration_sha256": fit["calibration_sha256"],
        "expected_threshold_source_sha256": _source_sha(),
        "expected_model_sha256": lineage["model_sha256"],
        "expected_partition_sha256": lineage["partition_sha256"],
        "expected_training_dataset_sha256": lineage[
            "training_dataset_sha256"
        ],
        "expected_fit_seed_inventory_sha256": fit_seed,
        "expected_fit_query_inventory_sha256": fit_query,
        "expected_fit_item_inventory_sha256": fit_item,
        "expected_audit_seed_inventory_sha256": audit_seed,
        "expected_audit_query_inventory_sha256": audit_query,
        "expected_audit_item_inventory_sha256": audit_item,
    }


def _apply(
    audit_queries: list[dict],
    audit_items: list[dict],
    audit_observations: list[dict],
    fit: dict,
    fit_queries: list[dict],
    fit_items: list[dict],
    fit_observations: list[dict],
    expectations: dict[str, str],
) -> dict:
    return apply_fixed_threshold_audit_v3(
        audit_observations,
        expected_items=audit_items,
        query_universe=audit_queries,
        fit_calibration=fit,
        fit_observations=fit_observations,
        fit_expected_items=fit_items,
        fit_query_universe=fit_queries,
        **expectations,
    )


class R3IG3ThresholdV3Tests(unittest.TestCase):
    def test_source_rule_and_lineage_are_caller_bound(self) -> None:
        lineage = _lineage()
        source_sha = _source_sha()
        binding = build_threshold_source_binding_v3(
            expected_threshold_source_sha256=source_sha
        )
        self.assertEqual(binding["source_sha256"], source_sha)
        rule = _rule(lineage)
        self.assertEqual(rule["model_sha256"], lineage["model_sha256"])
        self.assertEqual(
            rule["partition_sha256"],
            lineage["partition_sha256"],
        )
        self.assertEqual(
            rule["training_dataset_sha256"],
            lineage["training_dataset_sha256"],
        )
        with self.assertRaises(EvidenceError):
            build_threshold_source_binding_v3(
                expected_threshold_source_sha256="0" * 64
            )
        with self.assertRaises(EvidenceError):
            build_threshold_rule_v3(
                expected_threshold_source_sha256=source_sha,
                expected_model_sha256=True,
                expected_partition_sha256=lineage["partition_sha256"],
                expected_training_dataset_sha256=lineage[
                    "training_dataset_sha256"
                ],
            )

    def test_fit_is_query_denominated_tie_safe_and_exactly_reproducible(self) -> None:
        lineage = _lineage()
        queries, items, observations = _inputs(extra_first_items=1)
        _set(
            observations,
            0,
            0,
            score=0.9,
            eligible=True,
            acceptable=True,
        )
        _set(
            observations,
            0,
            1,
            score=0.9,
            eligible=True,
            acceptable=True,
        )
        _set(
            observations,
            1,
            score=0.8,
            eligible=True,
            acceptable=False,
        )
        fit = _fit(queries, items, observations, lineage)
        self.assertEqual(fit["schema"], FIT_REPORT_SCHEMA)
        self.assertTrue(fit["passed"])
        self.assertFalse(fit["terminal"])
        self.assertEqual(fit["threshold_hex"], (0.9).hex())
        self.assertEqual(fit["threshold_tie_block_count"], 2)
        self.assertEqual(fit["selected_candidate_count"], 2)
        self.assertEqual(fit["selected_query_count"], 1)
        self.assertEqual(fit["coverage_denominator"], 20)

        seed_sha, query_sha, item_sha = _inventory_shas(queries, items)
        verified = verify_fit_threshold_calibration_v3(
            list(reversed(observations)),
            fit,
            expected_items=list(reversed(items)),
            query_universe=list(reversed(queries)),
            rule=_rule(lineage),
            **lineage,
            expected_fit_calibration_sha256=fit["calibration_sha256"],
            expected_threshold_source_sha256=_source_sha(),
            expected_model_sha256=lineage["model_sha256"],
            expected_partition_sha256=lineage["partition_sha256"],
            expected_training_dataset_sha256=lineage[
                "training_dataset_sha256"
            ],
            expected_fit_seed_inventory_sha256=seed_sha,
            expected_fit_query_inventory_sha256=query_sha,
            expected_fit_item_inventory_sha256=item_sha,
        )
        self.assertEqual(verified, fit)

    def test_fit_rejects_actual_lineage_and_inventory_mismatches(self) -> None:
        lineage = _lineage()
        queries, items, observations = _inputs()
        _set(
            observations,
            0,
            score=0.9,
            eligible=True,
            acceptable=True,
        )
        seed_sha, query_sha, item_sha = _inventory_shas(queries, items)
        common = {
            "expected_items": items,
            "query_universe": queries,
            "rule": _rule(lineage),
            **lineage,
            "expected_threshold_source_sha256": _source_sha(),
            "expected_model_sha256": lineage["model_sha256"],
            "expected_partition_sha256": lineage["partition_sha256"],
            "expected_training_dataset_sha256": lineage[
                "training_dataset_sha256"
            ],
            "expected_fit_seed_inventory_sha256": seed_sha,
            "expected_fit_query_inventory_sha256": query_sha,
            "expected_fit_item_inventory_sha256": item_sha,
        }
        for field in (
            "model_sha256",
            "partition_sha256",
            "training_dataset_sha256",
            "expected_fit_seed_inventory_sha256",
            "expected_fit_query_inventory_sha256",
            "expected_fit_item_inventory_sha256",
        ):
            damaged = dict(common)
            damaged[field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(EvidenceError):
                fit_threshold_calibration_v3(observations, **damaged)

    def test_fit_verification_requires_the_expected_calibration_sha(self) -> None:
        lineage = _lineage()
        queries, items, observations = _inputs()
        _set(
            observations,
            0,
            score=0.9,
            eligible=True,
            acceptable=True,
        )
        fit = _fit(queries, items, observations, lineage)
        seed_sha, query_sha, item_sha = _inventory_shas(queries, items)
        with self.assertRaises(EvidenceError):
            verify_fit_threshold_calibration_v3(
                observations,
                fit,
                expected_items=items,
                query_universe=queries,
                rule=_rule(lineage),
                **lineage,
                expected_fit_calibration_sha256="0" * 64,
                expected_threshold_source_sha256=_source_sha(),
                expected_model_sha256=lineage["model_sha256"],
                expected_partition_sha256=lineage["partition_sha256"],
                expected_training_dataset_sha256=lineage[
                    "training_dataset_sha256"
                ],
                expected_fit_seed_inventory_sha256=seed_sha,
                expected_fit_query_inventory_sha256=query_sha,
                expected_fit_item_inventory_sha256=item_sha,
            )

    def test_fixed_audit_binds_the_complete_caller_contract(self) -> None:
        lineage = _lineage()
        fit_queries, fit_items, fit_observations = _inputs(seed_offset=0)
        _set(
            fit_observations,
            0,
            score=0.9,
            eligible=True,
            acceptable=True,
        )
        fit = _fit(fit_queries, fit_items, fit_observations, lineage)

        audit_queries, audit_items, audit_observations = _inputs(seed_offset=100)
        _set(
            audit_observations,
            0,
            score=0.95,
            eligible=True,
            acceptable=True,
        )
        expectations = _audit_expectations(
            fit,
            fit_queries,
            fit_items,
            audit_queries,
            audit_items,
            lineage,
        )
        report = _apply(
            audit_queries,
            audit_items,
            audit_observations,
            fit,
            fit_queries,
            fit_items,
            fit_observations,
            expectations,
        )
        self.assertEqual(report["schema"], AUDIT_REPORT_SCHEMA)
        self.assertTrue(report["passed"])
        self.assertEqual(
            report["caller_identity_contract_sha256"],
            core.sha256_json(report["caller_identity_contract"]),
        )
        self.assertEqual(
            report["fit_calibration_sha256"],
            expectations["expected_fit_calibration_sha256"],
        )
        self.assertEqual(
            verify_fixed_threshold_audit_v3(
                audit_observations,
                report,
                expected_items=audit_items,
                query_universe=audit_queries,
                fit_calibration=fit,
                fit_observations=fit_observations,
                fit_expected_items=fit_items,
                fit_query_universe=fit_queries,
                **expectations,
            ),
            report,
        )

    def test_every_caller_expectation_is_enforced(self) -> None:
        lineage = _lineage()
        fit_queries, fit_items, fit_observations = _inputs(seed_offset=0)
        _set(
            fit_observations,
            0,
            score=0.9,
            eligible=True,
            acceptable=True,
        )
        fit = _fit(fit_queries, fit_items, fit_observations, lineage)
        audit_queries, audit_items, audit_observations = _inputs(seed_offset=100)
        _set(
            audit_observations,
            0,
            score=0.95,
            eligible=True,
            acceptable=True,
        )
        expectations = _audit_expectations(
            fit,
            fit_queries,
            fit_items,
            audit_queries,
            audit_items,
            lineage,
        )
        for field in expectations:
            damaged = dict(expectations)
            damaged[field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(EvidenceError):
                _apply(
                    audit_queries,
                    audit_items,
                    audit_observations,
                    fit,
                    fit_queries,
                    fit_items,
                    fit_observations,
                    damaged,
                )

    def test_whole_valid_bundle_substitution_is_rejected(self) -> None:
        lineage = _lineage("shared")

        fit_a_queries, fit_a_items, fit_a_observations = _inputs(seed_offset=0)
        _set(
            fit_a_observations,
            0,
            score=0.9,
            eligible=True,
            acceptable=True,
        )
        fit_a = _fit(
            fit_a_queries,
            fit_a_items,
            fit_a_observations,
            lineage,
        )
        audit_a_queries, audit_a_items, audit_a_observations = _inputs(
            seed_offset=100
        )
        _set(
            audit_a_observations,
            0,
            score=0.95,
            eligible=True,
            acceptable=True,
        )
        expectations_a = _audit_expectations(
            fit_a,
            fit_a_queries,
            fit_a_items,
            audit_a_queries,
            audit_a_items,
            lineage,
        )
        report_a = _apply(
            audit_a_queries,
            audit_a_items,
            audit_a_observations,
            fit_a,
            fit_a_queries,
            fit_a_items,
            fit_a_observations,
            expectations_a,
        )
        self.assertTrue(report_a["passed"])

        fit_b_queries, fit_b_items, fit_b_observations = _inputs(seed_offset=200)
        _set(
            fit_b_observations,
            0,
            score=0.91,
            eligible=True,
            acceptable=True,
        )
        fit_b = _fit(
            fit_b_queries,
            fit_b_items,
            fit_b_observations,
            lineage,
        )
        audit_b_queries, audit_b_items, audit_b_observations = _inputs(
            seed_offset=300
        )
        _set(
            audit_b_observations,
            0,
            score=0.96,
            eligible=True,
            acceptable=True,
        )
        expectations_b = _audit_expectations(
            fit_b,
            fit_b_queries,
            fit_b_items,
            audit_b_queries,
            audit_b_items,
            lineage,
        )
        report_b = _apply(
            audit_b_queries,
            audit_b_items,
            audit_b_observations,
            fit_b,
            fit_b_queries,
            fit_b_items,
            fit_b_observations,
            expectations_b,
        )
        self.assertTrue(report_b["passed"])

        # The replacement is entirely self-consistent and independently
        # verifies under B's contract, but cannot satisfy A's caller contract.
        self.assertEqual(
            verify_fixed_threshold_audit_v3(
                audit_b_observations,
                report_b,
                expected_items=audit_b_items,
                query_universe=audit_b_queries,
                fit_calibration=fit_b,
                fit_observations=fit_b_observations,
                fit_expected_items=fit_b_items,
                fit_query_universe=fit_b_queries,
                **expectations_b,
            ),
            report_b,
        )
        with self.assertRaises(EvidenceError):
            verify_fixed_threshold_audit_v3(
                audit_b_observations,
                report_b,
                expected_items=audit_b_items,
                query_universe=audit_b_queries,
                fit_calibration=fit_b,
                fit_observations=fit_b_observations,
                fit_expected_items=fit_b_items,
                fit_query_universe=fit_b_queries,
                **expectations_a,
            )

        # Even accepting B's fit hash separately does not waive A's exact
        # fit/audit inventory expectations.
        mixed = dict(expectations_a)
        mixed["expected_fit_calibration_sha256"] = fit_b["calibration_sha256"]
        with self.assertRaises(EvidenceError):
            _apply(
                audit_b_queries,
                audit_b_items,
                audit_b_observations,
                fit_b,
                fit_b_queries,
                fit_b_items,
                fit_b_observations,
                mixed,
            )

    def test_resealed_fit_or_audit_tampering_is_rejected(self) -> None:
        lineage = _lineage()
        fit_queries, fit_items, fit_observations = _inputs(seed_offset=0)
        _set(
            fit_observations,
            0,
            score=0.9,
            eligible=True,
            acceptable=True,
        )
        fit = _fit(fit_queries, fit_items, fit_observations, lineage)
        audit_queries, audit_items, audit_observations = _inputs(seed_offset=100)
        _set(
            audit_observations,
            0,
            score=0.95,
            eligible=True,
            acceptable=True,
        )
        expectations = _audit_expectations(
            fit,
            fit_queries,
            fit_items,
            audit_queries,
            audit_items,
            lineage,
        )
        report = _apply(
            audit_queries,
            audit_items,
            audit_observations,
            fit,
            fit_queries,
            fit_items,
            fit_observations,
            expectations,
        )

        damaged_fit = copy.deepcopy(fit)
        damaged_fit["model_sha256"] = "0" * 64
        damaged_fit.pop("calibration_sha256")
        damaged_fit = core.seal_record(damaged_fit, "calibration_sha256")
        with self.assertRaises(EvidenceError):
            _apply(
                audit_queries,
                audit_items,
                audit_observations,
                damaged_fit,
                fit_queries,
                fit_items,
                fit_observations,
                expectations,
            )

        damaged_report = copy.deepcopy(report)
        damaged_report["model_sha256"] = "0" * 64
        damaged_report.pop("audit_sha256")
        damaged_report = core.seal_record(damaged_report, "audit_sha256")
        with self.assertRaises(EvidenceError):
            verify_fixed_threshold_audit_v3(
                audit_observations,
                damaged_report,
                expected_items=audit_items,
                query_universe=audit_queries,
                fit_calibration=fit,
                fit_observations=fit_observations,
                fit_expected_items=fit_items,
                fit_query_universe=fit_queries,
                **expectations,
            )


if __name__ == "__main__":
    unittest.main()
