from __future__ import annotations

import copy
import dataclasses
import unittest

from irisu_pointer.resolution_first import (
    FEATURE_NAMES,
    SelectiveCalibration,
    SelectedCandidate,
    SupportCalibration,
    branch_records,
    fit_selective_calibration,
    fit_support_calibration,
    fit_support_envelope,
    pilot_report,
    select_candidates,
    viability_report,
)


def _outcome(
    ordinal: int,
    *,
    tag: str,
    category: str = "fresh_match",
    geometry: str | None = None,
    resolved: bool = True,
    b2: float | None = 2.0,
    delta: float | None = 0.0,
    score: float = 0.0,
    unsafe: bool = False,
    feature_offset: float = 0.0,
) -> dict[str, object]:
    return {
        "candidate": {
            "ordinal": ordinal,
            "pair_ordinal": ordinal,
            "geometry_ordinal": ordinal,
            "tag": tag,
            "action": {
                "kind": 2,
                "x_norm": 0.1 + ordinal / 100.0,
                "y_norm": 0.2,
            },
            "pair": {
                "category": category,
                "source_body_id": 100 + ord(tag[-1]) if tag[-1].isalpha() else 100,
                "destination_body_id": 200 + ordinal,
            },
            "geometry": {"name": geometry or f"geometry-{tag}"},
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_vector": [
            float(index) / 10.0 + feature_offset for index in range(len(FEATURE_NAMES))
        ],
        "targets": {
            "b2_margin": b2,
            "delta_b2": delta,
            "score_advantage": score,
            "exact_unsafe": unsafe,
            "severe_unsafe": unsafe,
        },
        "outcome": {
            "renewals_resolved": resolved,
            "exact_unsafe": unsafe,
            "severe_unsafe": unsafe,
        },
        "ledger": {
            "unresolved": [] if resolved else [f"unresolved-{tag}"],
            "continuation_rebind_failed": False,
        },
    }


def _query(
    seed: int,
    query_id: str,
    alternatives: list[dict[str, object]],
) -> dict[str, object]:
    incumbent = _outcome(0, tag="incumbent", delta=0.0)
    outcomes = [incumbent, *alternatives]
    for ordinal, outcome in enumerate(outcomes):
        candidate = outcome["candidate"]
        assert isinstance(candidate, dict)
        candidate["ordinal"] = ordinal
        candidate["pair_ordinal"] = ordinal
        candidate["geometry_ordinal"] = ordinal
    return {"seed": seed, "query_id": query_id, "outcomes": outcomes}


def _prediction(
    seed: int,
    query_id: str,
    ordinal: int,
    *,
    support_score: float = 1.0,
    supported: bool = True,
    finite: bool = True,
    unsafe: bool = False,
    delta_lcb: float = 1.0,
    exact_delta: float | None = 0.5,
    score_lcb: float = 1.0,
    exact_score: float = 1.0,
    action_id: str | None = None,
) -> dict[str, object]:
    return {
        "seed": seed,
        "query_id": query_id,
        "ordinal": ordinal,
        "candidate_id": f"{query_id}-{ordinal}",
        "action_id": action_id or f"{query_id}-action-{ordinal}",
        "envelope_supported": supported,
        "support_score": support_score,
        "exact_finite_pair": finite,
        "exact_unsafe": unsafe,
        "delta_lcb": delta_lcb,
        "exact_delta_b2": exact_delta,
        "score_lcb": score_lcb,
        "exact_score_advantage": exact_score,
    }


def _support(threshold: float = 0.5) -> SupportCalibration:
    return SupportCalibration(
        threshold=threshold,
        selected_candidates=1,
        candidate_count=1,
        coverage=1.0,
        selected_bad_candidates=0,
        selected_bad_seeds=0,
        grid=(),
    )


class ExactRecordTests(unittest.TestCase):
    def test_unresolved_rows_are_retained_but_not_finite_pairs(self) -> None:
        query = _query(
            11,
            "unresolved",
            [
                _outcome(
                    1,
                    tag="a",
                    resolved=False,
                    b2=None,
                    delta=None,
                    score=7.0,
                )
            ],
        )

        records = branch_records([query])

        self.assertEqual(len(records), 2)
        unresolved = records[1]
        self.assertFalse(unresolved.candidate_resolved)
        self.assertFalse(unresolved.finite_pair)
        self.assertIsNone(unresolved.b2)
        self.assertIsNone(unresolved.delta_b2)
        self.assertEqual(unresolved.score_advantage, 7.0)
        self.assertEqual(len(unresolved.features), len(FEATURE_NAMES))
        self.assertEqual(len(unresolved.source_sha256), 64)

    def test_stable_candidate_identity_survives_alternative_permutation(self) -> None:
        left = _outcome(1, tag="a", geometry="left")
        right = _outcome(2, tag="b", geometry="right")
        first = _query(
            21,
            "first",
            [copy.deepcopy(left), copy.deepcopy(right)],
        )
        second = _query(
            22,
            "second",
            [copy.deepcopy(right), copy.deepcopy(left)],
        )

        first_records = branch_records([first])
        second_records = branch_records([second])
        first_ids = {record.signature: record.candidate_id for record in first_records[1:]}
        second_ids = {
            record.signature: record.candidate_id for record in second_records[1:]
        }

        self.assertEqual(first_ids, second_ids)
        self.assertNotEqual(
            first_records[1].source_sha256,
            second_records[2].source_sha256,
            "raw evidence identity should still bind the supplied ordinal",
        )

    def test_support_envelope_rejects_unknown_candidate_signature(self) -> None:
        records = branch_records(
            [
                _query(
                    31,
                    "support",
                    [_outcome(1, tag="a", category="known", geometry="known")],
                )
            ]
        )
        envelope = fit_support_envelope(records, minimum_signature_seeds=1)
        known = records[1]
        unknown = dataclasses.replace(known, signature="unknown|unknown")

        self.assertTrue(envelope.contains(known))
        self.assertFalse(envelope.contains(unknown))
        self.assertNotIn(unknown.signature, envelope.radius_by_signature)


class CalibrationTests(unittest.TestCase):
    def test_support_threshold_must_exclude_bad_rows(self) -> None:
        predictions = [
            _prediction(1, "q1", 0),
            _prediction(1, "q1", 1, support_score=0.2, finite=False),
            _prediction(2, "q2", 0),
            _prediction(2, "q2", 1, support_score=0.9),
            _prediction(3, "q3", 0),
            _prediction(3, "q3", 1, support_score=-1.0),
            _prediction(4, "q4", 0),
            _prediction(4, "q4", 1, support_score=-1.0),
        ]

        calibration = fit_support_calibration(
            predictions,
            thresholds=(0.0, 0.5),
            minimum_coverage=0.20,
        )

        self.assertEqual(calibration.threshold, 0.5)
        self.assertEqual(calibration.selected_candidates, 1)
        self.assertEqual(calibration.selected_bad_candidates, 0)
        self.assertFalse(calibration.grid[0]["passed"])
        self.assertTrue(calibration.grid[1]["passed"])

    def test_unselected_unresolved_does_not_poison_selective_quantile(self) -> None:
        support = _support()
        predictions = [
            _prediction(1, "q1", 0),
            _prediction(
                1,
                "q1",
                1,
                delta_lcb=0.9,
                exact_delta=0.5,
            ),
            _prediction(2, "q2", 0),
            _prediction(
                2,
                "q2",
                1,
                support_score=0.1,
                finite=False,
                exact_delta=None,
            ),
            _prediction(3, "q3", 0),
        ]

        calibration = fit_selective_calibration(
            predictions,
            support,
            alpha=0.25,
            required_episodes=3,
        )

        self.assertAlmostEqual(calibration.q, 0.4)
        self.assertEqual(calibration.selected_candidates, 1)
        self.assertEqual(calibration.selected_bad_candidates, 0)
        self.assertEqual(calibration.nonempty_episodes, 1)
        self.assertEqual(dict(calibration.episode_residuals)[2], 0.0)
        self.assertEqual(dict(calibration.episode_residuals)[3], 0.0)

    def test_selected_unresolved_forces_infinite_selective_quantile(self) -> None:
        support = _support()
        predictions = [
            _prediction(1, "q1", 0),
            _prediction(1, "q1", 1, delta_lcb=0.9, exact_delta=0.5),
            _prediction(2, "q2", 0),
            _prediction(
                2,
                "q2",
                1,
                support_score=0.9,
                finite=False,
                exact_delta=None,
            ),
            _prediction(3, "q3", 0),
        ]

        calibration = fit_selective_calibration(
            predictions,
            support,
            alpha=0.25,
        )

        self.assertEqual(calibration.q, float("inf"))
        self.assertEqual(calibration.selected_bad_candidates, 1)
        self.assertEqual(dict(calibration.episode_residuals)[2], float("inf"))
        self.assertEqual(len(calibration.sha256), 64)


class SelectionTests(unittest.TestCase):
    def test_selection_requires_all_gates_and_ties_fall_back(self) -> None:
        support = _support()
        calibration = SelectiveCalibration(
            alpha=0.05,
            q=0.1,
            episode_count=5,
            rank=5,
            selected_candidates=5,
            selected_bad_candidates=0,
            nonempty_episodes=5,
            episode_residuals=(),
            support_calibration_sha256=support.sha256,
        )
        predictions: list[dict[str, object]] = []
        cases = (
            ("support-fail", [_prediction(1, "support-fail", 1, support_score=0.5)]),
            ("delta-fail", [_prediction(2, "delta-fail", 1, delta_lcb=0.09)]),
            ("score-fail", [_prediction(3, "score-fail", 1, score_lcb=0.0)]),
            ("pass", [_prediction(4, "pass", 1, delta_lcb=0.2, score_lcb=0.3)]),
            (
                "tie",
                [
                    _prediction(5, "tie", 1, delta_lcb=0.2, score_lcb=0.3),
                    _prediction(5, "tie", 2, delta_lcb=0.9, score_lcb=0.3),
                ],
            ),
            (
                "same-action",
                [
                    _prediction(
                        6,
                        "same-action",
                        1,
                        delta_lcb=0.9,
                        score_lcb=0.9,
                        action_id="same-action-action-0",
                    )
                ],
            ),
        )
        for index, (query_id, alternatives) in enumerate(cases, start=1):
            predictions.append(_prediction(index, query_id, 0))
            predictions.extend(alternatives)

        selections = {
            selection.query_id: selection
            for selection in select_candidates(predictions, support, calibration)
        }

        for query_id in (
            "support-fail",
            "delta-fail",
            "score-fail",
            "same-action",
        ):
            self.assertFalse(selections[query_id].override)
            self.assertEqual(
                selections[query_id].reasons, ("no-certified-alternative",)
            )
        self.assertTrue(selections["pass"].override)
        self.assertEqual(selections["pass"].ordinal, 1)
        self.assertFalse(selections["tie"].override)
        self.assertEqual(selections["tie"].ordinal, 0)
        self.assertEqual(selections["tie"].reasons, ("conservative-tie",))


class ReportTests(unittest.TestCase):
    def test_structural_pilot_reports_preregistered_counts(self) -> None:
        queries = []
        for seed in range(18):
            alternatives = [
                _outcome(
                    ordinal,
                    tag=chr(ord("a") + ordinal),
                    delta=0.5,
                    score=2.0 if seed < 12 else -1.0,
                    feature_offset=seed / 100.0,
                )
                for ordinal in range(1, 7)
            ]
            queries.append(_query(1000 + seed, f"pilot-{seed}", alternatives))

        report = pilot_report(branch_records(queries))

        self.assertEqual(report["seeds"], 18)
        self.assertEqual(report["resolved_alternative_seeds"], 18)
        self.assertEqual(report["viable_alternative_seeds"], 12)
        self.assertEqual(report["resolved_nonincumbent_rows"], 108)
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["gates"].values()))

    def test_viability_report_passes_only_with_safe_useful_overrides(self) -> None:
        support = _support()
        calibration = SelectiveCalibration(
            alpha=0.05,
            q=0.0,
            episode_count=8,
            rank=8,
            selected_candidates=8,
            selected_bad_candidates=0,
            nonempty_episodes=8,
            episode_residuals=tuple((seed, 0.0) for seed in range(8)),
            support_calibration_sha256=support.sha256,
        )
        selections = tuple(
            SelectedCandidate(
                seed=seed,
                query_id=f"viable-{seed}",
                ordinal=1,
                candidate_id=f"candidate-{seed}",
                override=True,
                exact_finite_pair=True,
                exact_unsafe=False,
                exact_delta_b2=0.5,
                exact_score_advantage=2.0,
                reasons=(),
            )
            for seed in range(8)
        )

        report = viability_report(selections, calibration)

        self.assertEqual(report["overrides"], 8)
        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["false_safe_episodes"], 0)
        self.assertEqual(report["median_exact_score_advantage"], 2.0)
        self.assertEqual(report["median_exact_delta_b2"], 0.5)
        self.assertTrue(report["passed"])
        self.assertTrue(all(report["gates"].values()))

        unsafe = dataclasses.replace(
            selections[0],
            exact_finite_pair=False,
            exact_unsafe=True,
        )
        failed = viability_report((unsafe, *selections[1:]), calibration)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["false_safe_episodes"], 1)
        self.assertFalse(failed["gates"]["zero_selected_bad_candidates"])


if __name__ == "__main__":
    unittest.main()
