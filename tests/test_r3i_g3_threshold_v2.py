from __future__ import annotations

import copy
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_pointer import r3i_g3_evidence as core
from irisu_pointer.r3i_g3_threshold_v2 import (
    AUDIT_REPORT_SCHEMA,
    FIT_REPORT_SCHEMA,
    EvidenceError,
    apply_fixed_threshold_audit,
    build_threshold_rule_v2,
    build_threshold_source_binding_v2,
    fit_threshold_calibration,
    verify_fit_threshold_calibration,
    verify_fixed_threshold_audit,
)


class _FloatSubclass(float):
    pass


class _IntSubclass(int):
    pass


def _inputs(
    query_count: int,
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
            items.append(
                {"seed": seed, "query_id": query_id, "item_id": item_id}
            )
            observations.append(
                {
                    "seed": seed,
                    "query_id": query_id,
                    "item_id": item_id,
                    "score": 0.1,
                    "eligible": False,
                    "acceptable": False,
                }
            )
    return queries, items, observations


def _set(
    observations: list[dict],
    query_index: int,
    item_index: int,
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


def _source_binding() -> dict:
    source = ROOT / "python/irisu_pointer/r3i_g3_threshold_v2.py"
    sha256 = core.validate_regular_file(source, root=source.parent)["sha256"]
    return build_threshold_source_binding_v2(expected_source_sha256=sha256)


def _rule() -> dict:
    return build_threshold_rule_v2(source_binding=_source_binding())


class R3IG3ThresholdV2FitTests(unittest.TestCase):
    def test_rule_binds_live_evidence_core_and_rejects_tampering(self) -> None:
        rule = _rule()
        self.assertEqual(rule["comparator"], ">=")
        self.assertEqual(rule["minimum_query_coverage_numerator"], 1)
        self.assertEqual(rule["minimum_query_coverage_denominator"], 20)
        self.assertEqual(rule["maximum_bad_candidates"], 0)
        self.assertEqual(rule["maximum_bad_seeds"], 0)
        self.assertEqual(
            rule["evidence_core_dependency"]["source_sha256"],
            core.validate_regular_file(
                Path(core.__file__).resolve(),
                root=Path(core.__file__).resolve().parent,
            )["sha256"],
        )
        self.assertEqual(
            rule["threshold_module_source_binding"], _source_binding()
        )
        with self.assertRaises(EvidenceError):
            build_threshold_source_binding_v2(
                expected_source_sha256="0" * 64
            )

        queries, items, observations = _inputs(20)
        _set(
            observations,
            0,
            0,
            score=0.8,
            eligible=True,
            acceptable=True,
        )
        with self.assertRaises(TypeError):
            fit_threshold_calibration(
                observations,
                expected_items=items,
                query_universe=queries,
            )
        tampered = copy.deepcopy(rule)
        tampered["comparator"] = ">"
        with self.assertRaises(EvidenceError):
            fit_threshold_calibration(
                observations,
                expected_items=items,
                query_universe=queries,
                rule=tampered,
            )
        tampered.pop("rule_sha256")
        tampered = core.seal_record(tampered, "rule_sha256")
        with self.assertRaises(EvidenceError):
            fit_threshold_calibration(
                observations,
                expected_items=items,
                query_universe=queries,
                rule=tampered,
            )

        dependency_tamper = copy.deepcopy(rule)
        dependency_tamper["evidence_core_dependency"]["source_sha256"] = "0" * 64
        dependency_tamper.pop("rule_sha256")
        dependency_tamper = core.seal_record(
            dependency_tamper, "rule_sha256"
        )
        with self.assertRaises(EvidenceError):
            fit_threshold_calibration(
                observations,
                expected_items=items,
                query_universe=queries,
                rule=dependency_tamper,
            )

        source_tamper = copy.deepcopy(rule["threshold_module_source_binding"])
        source_tamper["source_sha256"] = "0" * 64
        source_tamper.pop("binding_sha256")
        source_tamper = core.seal_record(source_tamper, "binding_sha256")
        with self.assertRaises(EvidenceError):
            build_threshold_rule_v2(source_binding=source_tamper)

    def test_longest_safe_prefix_uses_unique_query_coverage_and_float_hex(self) -> None:
        queries, items, observations = _inputs(20)
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
            1,
            0,
            score=0.8,
            eligible=True,
            acceptable=True,
        )
        _set(
            observations,
            2,
            0,
            score=0.7,
            eligible=True,
            acceptable=False,
        )
        report = fit_threshold_calibration(
            observations,
            expected_items=items,
            query_universe=queries,
            rule=_rule(),
        )
        self.assertEqual(report["schema"], FIT_REPORT_SCHEMA)
        self.assertTrue(report["passed"])
        self.assertFalse(report["terminal"])
        self.assertEqual(report["threshold_hex"], (0.8).hex())
        self.assertEqual(report["threshold_tie_block_score_hex"], (0.8).hex())
        self.assertEqual(report["selected_candidate_count"], 2)
        self.assertEqual(report["selected_query_count"], 2)
        self.assertEqual(report["coverage_denominator"], 20)
        self.assertEqual(report["seed_inventory"], list(range(20)))
        self.assertEqual(report["query_inventory"], queries)
        self.assertEqual(report["item_inventory"], items)
        self.assertEqual(report["evaluated_tie_block_count"], 3)
        self.assertEqual(
            verify_fit_threshold_calibration(
                observations,
                report,
                expected_items=items,
                query_universe=queries,
                rule=_rule(),
            ),
            report,
        )

        # Input order is not scientific identity and cannot alter the result.
        reordered = fit_threshold_calibration(
            list(reversed(observations)),
            expected_items=list(reversed(items)),
            query_universe=list(reversed(queries)),
            rule=_rule(),
        )
        self.assertEqual(reordered, report)

    def test_equal_scores_are_one_whole_tie_block_and_never_split(self) -> None:
        queries, items, observations = _inputs(20, extra_first_items=1)
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
            0,
            score=0.8,
            eligible=True,
            acceptable=False,
        )
        report = fit_threshold_calibration(
            observations,
            expected_items=items,
            query_universe=queries,
            rule=_rule(),
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["threshold_hex"], (0.9).hex())
        self.assertEqual(report["threshold_tie_block_count"], 2)
        self.assertEqual(report["selected_candidate_count"], 2)
        self.assertEqual(report["selected_query_count"], 1)
        self.assertEqual(report["unique_eligible_score_count"], 2)

        # A mixed-label top tie cannot be split to keep only its good member.
        _set(
            observations,
            0,
            1,
            score=0.9,
            eligible=True,
            acceptable=False,
        )
        failed = fit_threshold_calibration(
            observations,
            expected_items=items,
            query_universe=queries,
            rule=_rule(),
        )
        self.assertFalse(failed["passed"])
        self.assertIsNone(failed["threshold_hex"])
        self.assertEqual(failed["selected_candidate_count"], 0)
        self.assertIn(
            "no-zero-error-eligible-prefix", failed["failure_reasons"]
        )

    def test_many_candidates_in_one_query_cannot_fake_query_coverage(self) -> None:
        queries, items, observations = _inputs(21, extra_first_items=9)
        for item_index in range(10):
            _set(
                observations,
                0,
                item_index,
                score=0.9,
                eligible=True,
                acceptable=True,
            )
        report = fit_threshold_calibration(
            observations,
            expected_items=items,
            query_universe=queries,
            rule=_rule(),
        )
        self.assertFalse(report["passed"])
        self.assertTrue(report["terminal"])
        self.assertEqual(report["selected_candidate_count"], 10)
        self.assertEqual(report["selected_query_count"], 1)
        self.assertEqual(report["coverage_denominator"], 21)
        self.assertFalse(report["coverage_passed"])
        self.assertIn(
            "unique-query-coverage-below-five-percent",
            report["failure_reasons"],
        )

    def test_zero_eligible_queries_stay_in_denominator(self) -> None:
        queries, items, observations = _inputs(40)
        _set(
            observations,
            0,
            0,
            score=0.5,
            eligible=True,
            acceptable=True,
        )
        report = fit_threshold_calibration(
            observations,
            expected_items=items,
            query_universe=queries,
            rule=_rule(),
        )
        self.assertEqual(report["coverage_denominator"], 40)
        self.assertEqual(report["selected_query_count"], 1)
        self.assertFalse(report["coverage_passed"])
        self.assertFalse(report["passed"])

        none_eligible = fit_threshold_calibration(
            _inputs(20)[2],
            expected_items=_inputs(20)[1],
            query_universe=_inputs(20)[0],
            rule=_rule(),
        )
        self.assertIsNone(none_eligible["threshold_hex"])
        self.assertFalse(none_eligible["passed"])

    def test_missing_duplicate_and_zero_query_closure_fail_closed(self) -> None:
        queries, items, observations = _inputs(2)
        with self.assertRaises(EvidenceError):
            fit_threshold_calibration(
                observations[:-1],
                expected_items=items,
                query_universe=queries,
                rule=_rule(),
            )
        with self.assertRaises(EvidenceError):
            fit_threshold_calibration(
                observations + [copy.deepcopy(observations[0])],
                expected_items=items,
                query_universe=queries,
                rule=_rule(),
            )
        with self.assertRaises(EvidenceError):
            fit_threshold_calibration(
                observations,
                expected_items=items + [copy.deepcopy(items[0])],
                query_universe=queries,
                rule=_rule(),
            )
        with self.assertRaises(EvidenceError):
            fit_threshold_calibration(
                observations,
                expected_items=items,
                query_universe=queries + [copy.deepcopy(queries[0])],
                rule=_rule(),
            )
        with self.assertRaises(EvidenceError):
            fit_threshold_calibration(
                [], expected_items=[], query_universe=[], rule=_rule()
            )
        with self.assertRaises(EvidenceError):
            fit_threshold_calibration(
                observations[:1],
                expected_items=items[:1],
                query_universe=queries,
                rule=_rule(),
            )

    def test_nonfinite_spoof_and_ambiguous_scores_fail_closed(self) -> None:
        mutations = (
            ("score", math.nan),
            ("score", math.inf),
            ("score", -math.inf),
            ("score", 1),
            ("score", _FloatSubclass(0.5)),
            ("score", -0.0),
            ("eligible", 1),
            ("acceptable", 0),
            ("seed", _IntSubclass(0)),
        )
        for field, value in mutations:
            with self.subTest(field=field, value=repr(value)):
                queries, items, observations = _inputs(1)
                observations[0][field] = value
                with self.assertRaises(EvidenceError):
                    fit_threshold_calibration(
                        observations,
                        expected_items=items,
                        query_universe=queries,
                        rule=_rule(),
                    )

    def test_fit_report_hash_structure_and_resealed_tamper_fail(self) -> None:
        queries, items, observations = _inputs(20)
        _set(
            observations,
            0,
            0,
            score=0.8,
            eligible=True,
            acceptable=True,
        )
        report = fit_threshold_calibration(
            observations,
            expected_items=items,
            query_universe=queries,
            rule=_rule(),
        )
        damaged = copy.deepcopy(report)
        damaged["threshold_tie_block_count"] = 2
        with self.assertRaises(EvidenceError):
            verify_fit_threshold_calibration(
                observations,
                damaged,
                expected_items=items,
                query_universe=queries,
                rule=_rule(),
            )
        damaged.pop("calibration_sha256")
        damaged = core.seal_record(damaged, "calibration_sha256")
        with self.assertRaises(EvidenceError):
            verify_fit_threshold_calibration(
                observations,
                damaged,
                expected_items=items,
                query_universe=queries,
                rule=_rule(),
            )
        wrong_hash = copy.deepcopy(report)
        wrong_hash["calibration_sha256"] = "0" * 64
        with self.assertRaises(EvidenceError):
            verify_fit_threshold_calibration(
                observations,
                wrong_hash,
                expected_items=items,
                query_universe=queries,
                rule=_rule(),
            )


class R3IG3ThresholdV2AuditTests(unittest.TestCase):
    def _passed_fit(self) -> tuple[list[dict], list[dict], list[dict], dict]:
        queries, items, observations = _inputs(20)
        _set(
            observations,
            0,
            0,
            score=0.8,
            eligible=True,
            acceptable=True,
        )
        fit = fit_threshold_calibration(
            observations,
            expected_items=items,
            query_universe=queries,
            rule=_rule(),
        )
        self.assertTrue(fit["passed"])
        return queries, items, observations, fit

    def test_audit_applies_frozen_threshold_when_audit_optimum_differs(self) -> None:
        fit_queries, fit_items, fit_observations, fit = self._passed_fit()
        queries, items, observations = _inputs(20, seed_offset=100)
        # On audit alone, 0.9 is the safe passing threshold.  The frozen 0.8
        # fit threshold must also select the 0.85 bad candidate and therefore
        # fail; searching/refitting here would incorrectly pass.
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
            1,
            0,
            score=0.85,
            eligible=True,
            acceptable=False,
        )
        _set(
            observations,
            2,
            0,
            score=0.7,
            eligible=True,
            acceptable=True,
        )
        report = apply_fixed_threshold_audit(
            observations,
            expected_items=items,
            query_universe=queries,
            fit_calibration=fit,
            fit_observations=fit_observations,
            fit_expected_items=fit_items,
            fit_query_universe=fit_queries,
        )
        self.assertEqual(report["schema"], AUDIT_REPORT_SCHEMA)
        self.assertEqual(
            report["application_mode"], "fixed-fit-threshold-no-search"
        )
        self.assertEqual(report["frozen_threshold_hex"], (0.8).hex())
        self.assertEqual(report["selected_candidate_count"], 2)
        self.assertEqual(report["selected_bad_candidate_count"], 1)
        self.assertEqual(report["selected_bad_seed_count"], 1)
        self.assertEqual(report["fit_seed_inventory"], list(range(20)))
        self.assertEqual(report["audit_seed_inventory"], list(range(100, 120)))
        self.assertEqual(report["seed_overlap_count"], 0)
        self.assertEqual(report["query_key_overlap_count"], 0)
        self.assertEqual(report["item_key_overlap_count"], 0)
        self.assertFalse(report["passed"])
        self.assertTrue(report["terminal"])
        self.assertEqual(
            verify_fixed_threshold_audit(
                observations,
                report,
                expected_items=items,
                query_universe=queries,
                fit_calibration=fit,
                fit_observations=fit_observations,
                fit_expected_items=fit_items,
                fit_query_universe=fit_queries,
            ),
            report,
        )

    def test_audit_passes_only_zero_bad_and_five_percent_query_coverage(self) -> None:
        fit_queries, fit_items, fit_observations, fit = self._passed_fit()
        queries, items, observations = _inputs(20, seed_offset=100)
        _set(
            observations,
            0,
            0,
            score=0.8,
            eligible=True,
            acceptable=True,
        )
        report = apply_fixed_threshold_audit(
            observations,
            expected_items=items,
            query_universe=queries,
            fit_calibration=fit,
            fit_observations=fit_observations,
            fit_expected_items=fit_items,
            fit_query_universe=fit_queries,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["selected_query_count"], 1)
        self.assertEqual(report["coverage_denominator"], 20)

        queries_21, items_21, observations_21 = _inputs(
            21, seed_offset=200, extra_first_items=9
        )
        for item_index in range(10):
            _set(
                observations_21,
                0,
                item_index,
                score=0.8,
                eligible=True,
                acceptable=True,
            )
        coverage_failure = apply_fixed_threshold_audit(
            observations_21,
            expected_items=items_21,
            query_universe=queries_21,
            fit_calibration=fit,
            fit_observations=fit_observations,
            fit_expected_items=fit_items,
            fit_query_universe=fit_queries,
        )
        self.assertEqual(coverage_failure["selected_candidate_count"], 10)
        self.assertEqual(coverage_failure["selected_query_count"], 1)
        self.assertFalse(coverage_failure["passed"])

    def test_same_fit_audit_identity_and_universe_are_rejected(self) -> None:
        queries, items, observations, fit = self._passed_fit()
        with self.assertRaises(EvidenceError):
            apply_fixed_threshold_audit(
                observations,
                expected_items=items,
                query_universe=queries,
                fit_calibration=fit,
                fit_observations=observations,
                fit_expected_items=items,
                fit_query_universe=queries,
            )

        # Exact audit counterexample: all 20 fit seeds/query keys are reused
        # while every item is renamed.  A different item/content hash cannot
        # launder query or seed overlap.
        renamed_items = copy.deepcopy(items)
        renamed_observations = copy.deepcopy(observations)
        for row in renamed_items:
            row["item_id"] = f"renamed-{row['item_id']}"
        for row in renamed_observations:
            row["item_id"] = f"renamed-{row['item_id']}"
            row["score"] = 0.82
        with self.assertRaisesRegex(EvidenceError, "seed inventories overlap"):
            apply_fixed_threshold_audit(
                renamed_observations,
                expected_items=renamed_items,
                query_universe=queries,
                fit_calibration=fit,
                fit_observations=observations,
                fit_expected_items=items,
                fit_query_universe=queries,
            )

        # Scores differ, but recycling the exact fit item universe is still
        # forbidden rather than treated as a new audit split.
        changed = copy.deepcopy(observations)
        changed[0]["score"] = 0.81
        with self.assertRaises(EvidenceError):
            apply_fixed_threshold_audit(
                changed,
                expected_items=items,
                query_universe=queries,
                fit_calibration=fit,
                fit_observations=observations,
                fit_expected_items=items,
                fit_query_universe=queries,
            )

    def test_failed_or_tampered_fit_and_tampered_audit_fail_closed(self) -> None:
        queries, items, observations = _inputs(20)
        failed_fit = fit_threshold_calibration(
            observations,
            expected_items=items,
            query_universe=queries,
            rule=_rule(),
        )
        audit_queries, audit_items, audit_observations = _inputs(
            20, seed_offset=100
        )
        with self.assertRaises(EvidenceError):
            apply_fixed_threshold_audit(
                audit_observations,
                expected_items=audit_items,
                query_universe=audit_queries,
                fit_calibration=failed_fit,
                fit_observations=observations,
                fit_expected_items=items,
                fit_query_universe=queries,
            )

        (
            passed_fit_queries,
            passed_fit_items,
            passed_fit_observations,
            fit,
        ) = self._passed_fit()
        # Coherently reseal the formerly structurally acceptable 0.8 -> 0.9
        # counterexample.  Direct replay of the original fit inputs must reject
        # it even though its outer record hash and local threshold fields agree.
        tampered_fit = copy.deepcopy(fit)
        tampered_fit["threshold_hex"] = (0.9).hex()
        tampered_fit["threshold_tie_block_score_hex"] = (0.9).hex()
        tampered_fit.pop("calibration_sha256")
        tampered_fit = core.seal_record(
            tampered_fit, "calibration_sha256"
        )
        with self.assertRaisesRegex(
            EvidenceError, "does not reproduce exactly"
        ):
            apply_fixed_threshold_audit(
                audit_observations,
                expected_items=audit_items,
                query_universe=audit_queries,
                fit_calibration=tampered_fit,
                fit_observations=passed_fit_observations,
                fit_expected_items=passed_fit_items,
                fit_query_universe=passed_fit_queries,
            )

        _set(
            audit_observations,
            0,
            0,
            score=0.9,
            eligible=True,
            acceptable=True,
        )
        audit = apply_fixed_threshold_audit(
            audit_observations,
            expected_items=audit_items,
            query_universe=audit_queries,
            fit_calibration=fit,
            fit_observations=passed_fit_observations,
            fit_expected_items=passed_fit_items,
            fit_query_universe=passed_fit_queries,
        )
        tampered_audit = copy.deepcopy(audit)
        tampered_audit["passed"] = False
        with self.assertRaises(EvidenceError):
            verify_fixed_threshold_audit(
                audit_observations,
                tampered_audit,
                expected_items=audit_items,
                query_universe=audit_queries,
                fit_calibration=fit,
                fit_observations=passed_fit_observations,
                fit_expected_items=passed_fit_items,
                fit_query_universe=passed_fit_queries,
            )

        tampered_audit.pop("audit_sha256")
        tampered_audit = core.seal_record(tampered_audit, "audit_sha256")
        with self.assertRaises(EvidenceError):
            verify_fixed_threshold_audit(
                audit_observations,
                tampered_audit,
                expected_items=audit_items,
                query_universe=audit_queries,
                fit_calibration=fit,
                fit_observations=passed_fit_observations,
                fit_expected_items=passed_fit_items,
                fit_query_universe=passed_fit_queries,
            )


if __name__ == "__main__":
    unittest.main()
