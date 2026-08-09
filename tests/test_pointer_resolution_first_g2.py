from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

import torch

from irisu_pointer.resolution_first import FEATURE_NAMES
from irisu_pointer.resolution_first_g2 import (
    BoardResolutionDataset,
    G2B2Calibration,
    G2SelectedCandidate,
    G2SupportCalibration,
    RepresentationEnvelope,
    ResolutionFirstG2Config,
    RobustSignatureProfile,
    _record_stratum,
    board_branch_records,
    fit_selective_calibration_g2,
    fit_support_calibration_g2,
    load_checkpoint_g2,
    predict_records_g2,
    resolution_auroc_g2,
    save_checkpoint_g2,
    select_candidates_g2,
    train_resolution_first_g2,
    viability_report_g2,
)


def _body(
    identifier: int,
    x: float,
    y: float,
    *,
    chain: int,
    color: int,
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": "dynamic_fresh",
        "color": color,
        "x": x,
        "y": y,
        "vx": 0.0,
        "vy": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 40.0,
        "chain_id": chain,
        "projectile_hits": 0,
        "age_ticks": 20,
        "remaining_lifetime": 1000,
        "rot_timer": 0,
    }


def _observation(
    bodies: list[dict[str, object]], *, tick: int = 101
) -> dict[str, object]:
    return {
        "tick": tick,
        "score": 50,
        "gauge": 800,
        "gauge_max": 1000,
        "level": 2,
        "highest_chain": 1,
        "qualifying_clear_count": 2,
        "difficulty": {"active_colors": 3, "spawn_interval_ticks": 50},
        "bodies": bodies,
    }


def _outcome(
    ordinal: int,
    *,
    source: int = 1,
    destination: int = 2,
    resolved: bool = True,
    unsafe: bool = False,
    b2: float | None = 2.0,
    delta: float | None = 0.0,
    score: float = 0.0,
    action_x: float | None = None,
) -> dict[str, object]:
    features = [
        index / 10.0 + ordinal / 100.0
        for index in range(len(FEATURE_NAMES))
    ]
    features[FEATURE_NAMES.index("source_chain")] = 1.0
    features[FEATURE_NAMES.index("destination_chain")] = 1.0
    return {
        "candidate": {
            "ordinal": ordinal,
            "pair_ordinal": ordinal,
            "geometry_ordinal": ordinal,
            "action": {
                "kind": 2,
                "x_norm": 0.2 + ordinal / 100 if action_x is None else action_x,
                "y_norm": 0.3,
            },
            "pair": {
                "category": "fresh_match",
                "source_body_id": source,
                "destination_body_id": destination,
            },
            "geometry": {"name": "direct"},
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_vector": features,
        "targets": {
            "b2_margin": b2 if resolved else None,
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
            "unresolved": [] if resolved else ["censored"],
            "continuation_rebind_failed": False,
        },
    }


def _entry(
    seed: int,
    *,
    order: tuple[int, ...] = (1, 2, 3, 4),
    id_map: dict[int, int] | None = None,
    chain_map: dict[int, int] | None = None,
    tick: int = 101,
    incumbent_resolved: bool = True,
    all_resolved: bool = False,
) -> dict[str, object]:
    id_map = {value: value for value in order} if id_map is None else id_map
    chain_map = {10: 10, 20: 20} if chain_map is None else chain_map
    physical = {
        1: (200.0, 171.0, 10, 0),
        2: (340.0, 172.0, 10, 0),
        3: (470.0, 173.0, 20, 1),
        4: (520.0, 174.0, 20, 2),
    }
    bodies = [
        _body(
            id_map[original],
            physical[original][0],
            physical[original][1],
            chain=chain_map[physical[original][2]],
            color=physical[original][3],
        )
        for original in order
    ]
    outcomes = [
        _outcome(
            0,
            source=id_map[1],
            destination=id_map[2],
            resolved=incumbent_resolved,
            b2=1.0 if incumbent_resolved else None,
            delta=0.0 if incumbent_resolved else None,
        ),
        _outcome(
            1,
            source=id_map[1],
            destination=id_map[3],
            resolved=True,
            b2=3.0,
            delta=0.8 if incumbent_resolved else None,
            score=3.0,
        ),
        _outcome(
            2,
            source=id_map[2],
            destination=id_map[4],
            resolved=all_resolved,
            unsafe=not all_resolved,
            b2=2.0 if all_resolved else None,
            delta=0.2 if all_resolved and incumbent_resolved else None,
            score=-2.0,
        ),
    ]
    return {
        "exact_query": {
            "seed": seed,
            "query_id": f"q-{seed}",
            "outcomes": outcomes,
        },
        "pre_query_public_observation": _observation(bodies, tick=tick),
        "query_index": 1,
        "shot_index": 7,
        "tick": tick,
    }


def _dataset(*, incumbent_resolved: bool = True) -> BoardResolutionDataset:
    return BoardResolutionDataset(
        board_branch_records(
            [
                _entry(seed, incumbent_resolved=incumbent_resolved)
                for seed in range(1, 9)
            ]
        )
    )


def _config() -> ResolutionFirstG2Config:
    return ResolutionFirstG2Config(
        members=2,
        body_width=8,
        hidden_width=12,
        training_steps=2,
        batch_size=12,
        minimum_signature_seeds=8,
        envelope_top_k=4,
        training_seeds=(41, 43),
    )


def _prediction(
    seed: int,
    ordinal: int,
    *,
    support_score: float = 0.9,
    score_lcb: float = 1.0,
    resolved: bool = True,
    unsafe: bool = False,
    b2_lcb: float = 2.0,
    exact_b2: float | None = 1.0,
    action: str | None = None,
    envelope: bool = True,
    resolution_mean: float | None = None,
    finite_pair: bool = True,
) -> dict[str, object]:
    return {
        "seed": seed,
        "query_id": f"q-{seed}",
        "ordinal": ordinal,
        "candidate_id": f"candidate-{seed}-{ordinal}",
        "action_id": f"action-{seed}-{ordinal}" if action is None else action,
        "signature": "fresh_match|direct",
        "envelope_supported": envelope,
        "support_score": support_score,
        "resolution_mean": (
            support_score if resolution_mean is None else resolution_mean
        ),
        "exact_candidate_resolved": resolved,
        "exact_finite_pair": finite_pair,
        "exact_unsafe": unsafe,
        "b2_lcb": b2_lcb,
        "exact_b2": exact_b2,
        "delta_lcb": 0.0,
        "exact_delta_b2": 0.0 if finite_pair else None,
        "score_lcb": score_lcb,
        "exact_score_advantage": 2.0,
    }


def _support(threshold: float = 0.5) -> G2SupportCalibration:
    return G2SupportCalibration(
        threshold,
        1,
        1,
        1,
        1,
        1.0,
        0,
        0,
        1.0,
        (),
    )


def _b2_calibration(
    support: G2SupportCalibration, *, q: float = 0.0
) -> G2B2Calibration:
    return G2B2Calibration(
        0.05,
        q,
        8,
        8,
        1,
        0,
        1,
        tuple((seed, 0.0) for seed in range(1, 9)),
        support.sha256,
    )


class BoardRecordTests(unittest.TestCase):
    def test_identity_chain_order_and_tick_are_not_model_inputs(self) -> None:
        original = board_branch_records([_entry(1)])
        transformed = board_branch_records(
            [
                _entry(
                    1,
                    order=(4, 2, 1, 3),
                    id_map={1: 401, 2: 205, 3: 999, 4: 17},
                    chain_map={10: 8000, 20: 3},
                    tick=151,
                )
            ]
        )
        left = BoardResolutionDataset(original).tensors()
        right = BoardResolutionDataset(transformed).tensors()
        for tensor_index in (0, 1, 2, 3, 4):
            if tensor_index == 1:
                self.assertTrue(
                    torch.equal(
                        left[tensor_index].sort(dim=1).values,
                        right[tensor_index].sort(dim=1).values,
                    )
                )
            else:
                self.assertTrue(torch.equal(left[tensor_index], right[tensor_index]))
        for record in (*original, *transformed):
            self.assertTrue(all(row[39] == 0.0 for row in record.body_features))
            self.assertTrue(all(row[40] == 0.0 for row in record.body_features))
            self.assertEqual(record.features[15], 1.0)
            self.assertEqual(record.features[23], 1.0)
            self.assertEqual(record.model_global_features[0], 0.0)

    def test_ungrouped_bodies_are_not_a_shared_chain(self) -> None:
        entry = _entry(1)
        bodies = entry["pre_query_public_observation"]["bodies"]
        bodies[0]["chain_id"] = 0
        bodies[1]["chain_id"] = 0
        records = board_branch_records([entry])
        record = records[0]
        self.assertFalse(record.body_grouped_flags[record.source_index])
        self.assertFalse(record.body_grouped_flags[record.destination_index])
        relational = BoardResolutionDataset(records).tensors()[1][0]
        source = relational[record.source_index, 45:]
        destination = relational[record.destination_index, 45:]
        self.assertEqual(float(source[4]), 0.0)
        self.assertEqual(float(destination[4]), 0.0)
        self.assertEqual(float(source[6]), 0.0)
        self.assertEqual(float(source[7]), 0.0)

    def test_resolved_unpaired_rescue_retains_absolute_b2(self) -> None:
        records = board_branch_records([_entry(1, incumbent_resolved=False)])
        rescue = records[1]
        self.assertTrue(rescue.candidate_resolved)
        self.assertFalse(rescue.finite_pair)
        self.assertIsNone(rescue.delta_b2)
        self.assertEqual(rescue.b2, 3.0)
        self.assertTrue(rescue.safe_score_row_g2)

    def test_unresolved_unsafe_has_its_own_balanced_stratum(self) -> None:
        unsafe = board_branch_records([_entry(1)])[2]
        self.assertFalse(unsafe.candidate_resolved)
        self.assertTrue(unsafe.exact_unsafe)
        self.assertEqual(_record_stratum(unsafe, 0.25), "unsafe")

        entry = _entry(1)
        entry["exact_query"]["outcomes"][2] = _outcome(
            2,
            source=2,
            destination=4,
            resolved=False,
            unsafe=False,
            b2=None,
            delta=None,
        )
        unresolved = board_branch_records([entry])[2]
        self.assertEqual(_record_stratum(unresolved, 0.25), "unresolved")


class LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.training = _dataset()
        cls.model = train_resolution_first_g2(cls.training, config=_config())

    def test_prediction_invariant_to_order_id_chain_and_tick(self) -> None:
        variants = [
            _entry(20),
            _entry(20, order=(4, 3, 2, 1)),
            _entry(
                20,
                order=(3, 1, 4, 2),
                id_map={1: 111, 2: 900, 3: 7, 4: 45},
                chain_map={10: 72, 20: 801},
            ),
            _entry(20, tick=201),
        ]
        predictions = [
            predict_records_g2(
                self.model, BoardResolutionDataset(board_branch_records([entry]))
            )
            for entry in variants
        ]
        keys = (
            "resolution_mean",
            "unsafe_mean",
            "support_score",
            "delta_lcb",
            "b2_lcb",
            "score_lcb",
        )
        for candidate in range(3):
            for key in keys:
                baseline = float(predictions[0][candidate][key])
                for variant in predictions[1:]:
                    self.assertAlmostEqual(
                        baseline, float(variant[candidate][key]), places=6
                    )

    def test_checkpoint_roundtrip_is_prediction_exact(self) -> None:
        before = predict_records_g2(self.model, self.training)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            digest = save_checkpoint_g2(path, self.model, metadata={"x": 1})
            loaded, metadata = load_checkpoint_g2(path)
            after = predict_records_g2(loaded, self.training)
        self.assertEqual(len(digest), 64)
        self.assertEqual(metadata, {"x": 1})
        self.assertEqual(before, after)
        for name, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, loaded.state_dict()[name]))

    def test_checkpoint_rejects_changed_score_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model.pt"
            changed = Path(directory) / "changed.pt"
            save_checkpoint_g2(source, self.model)
            payload = torch.load(source, map_location="cpu", weights_only=True)
            payload["manifest"]["score_target"] = "foreign"
            torch.save(payload, changed)
            with self.assertRaisesRegex(RuntimeError, "feature identity"):
                load_checkpoint_g2(changed)

    def test_checkpoint_rejects_nonfinite_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "model.pt"
            changed = Path(directory) / "nonfinite.pt"
            save_checkpoint_g2(source, self.model)
            payload = torch.load(source, map_location="cpu", weights_only=True)
            name = next(
                key
                for key, value in payload["state_dict"].items()
                if value.is_floating_point()
            )
            payload["state_dict"][name] = payload["state_dict"][name].clone()
            payload["state_dict"][name].reshape(-1)[0] = float("nan")
            torch.save(payload, changed)
            with self.assertRaisesRegex(RuntimeError, "nonfinite"):
                load_checkpoint_g2(changed)

    def test_checkpoint_rejects_indirect_parent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            direct = root / "direct"
            indirect = root / "indirect"
            direct.mkdir()
            indirect.symlink_to(direct, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "indirect"):
                save_checkpoint_g2(indirect / "model.pt", self.model)

    def test_bootstrap_fails_closed_without_resolution_classes(self) -> None:
        dataset = BoardResolutionDataset(
            board_branch_records([_entry(seed, all_resolved=True) for seed in range(1, 9)])
        )
        with self.assertRaisesRegex(ValueError, "both candidate-resolved classes"):
            train_resolution_first_g2(dataset, config=_config())


class EnvelopeTests(unittest.TestCase):
    def test_rms_topk_and_max_gates_are_conjunctive(self) -> None:
        def envelope(rms: float, topk: float, maximum: float) -> RepresentationEnvelope:
            return RepresentationEnvelope(
                (
                    RobustSignatureProfile(
                        "s",
                        (0.0, 0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0, 1.0),
                        rms,
                        topk,
                        maximum,
                        8,
                    ),
                ),
                0.975,
                8,
                2,
            )

        value = (0.6, 0.6, 0.0, 0.0)
        self.assertTrue(envelope(1.0, 1.0, 1.0).contains("s", value))
        self.assertFalse(envelope(0.4, 1.0, 1.0).contains("s", value))
        self.assertFalse(envelope(1.0, 0.5, 1.0).contains("s", value))
        self.assertFalse(envelope(1.0, 1.0, 0.5).contains("s", value))
        self.assertFalse(envelope(1.0, 1.0, 1.0).contains("unknown", value))


class CalibrationTests(unittest.TestCase):
    def test_support_accounts_all_prefix_candidates(self) -> None:
        predictions = [
            _prediction(1, 0, support_score=0.0),
            _prediction(1, 1, support_score=0.9),
            _prediction(
                1,
                2,
                support_score=0.8,
                resolved=False,
                exact_b2=None,
                resolution_mean=0.1,
            ),
        ]
        with self.assertRaisesRegex(RuntimeError, "support threshold"):
            fit_support_calibration_g2(
                predictions,
                thresholds=(0.5,),
                minimum_coverage=0.05,
                minimum_auroc=0.0,
            )

    def test_action_identical_and_score_veto_are_prefix_exclusions(self) -> None:
        incumbent_action = "same-action"
        predictions = [
            _prediction(1, 0, support_score=0.0, action=incumbent_action),
            _prediction(1, 1, action=incumbent_action),
            _prediction(
                1,
                2,
                score_lcb=-0.01,
                resolved=False,
                exact_b2=None,
                resolution_mean=0.1,
            ),
            _prediction(1, 3, support_score=0.8, resolution_mean=0.8),
            _prediction(
                1,
                4,
                support_score=0.1,
                resolved=False,
                exact_b2=None,
                resolution_mean=0.1,
            ),
        ]
        support = fit_support_calibration_g2(
            predictions,
            thresholds=(0.5,),
            minimum_coverage=0.05,
            minimum_auroc=0.0,
        )
        self.assertEqual(support.selected_candidates, 1)
        self.assertEqual(support.selected_queries, 1)
        self.assertEqual(support.selected_bad_candidates, 0)

    def test_auroc_uses_candidate_resolved_and_is_honestly_undefined(self) -> None:
        report = resolution_auroc_g2(
            [
                _prediction(1, 0),
                _prediction(1, 1, resolved=True, finite_pair=False),
            ]
        )
        self.assertIsNone(report["auroc"])
        self.assertFalse(report["passed"])
        self.assertEqual(report["reason"], "undefined-single-class")

        defined = resolution_auroc_g2(
            [
                _prediction(1, 0),
                _prediction(
                    1, 1, resolved=True, finite_pair=False, resolution_mean=0.9
                ),
                _prediction(
                    1,
                    2,
                    resolved=False,
                    exact_b2=None,
                    resolution_mean=0.1,
                ),
            ]
        )
        self.assertEqual(defined["auroc"], 1.0)

    def test_absolute_b2_margin_uses_all_prefix_and_preserves_infinity(self) -> None:
        predictions = [
            _prediction(1, 0, support_score=0.0),
            _prediction(1, 1, b2_lcb=3.0, exact_b2=2.0),
            _prediction(
                1,
                2,
                resolved=False,
                exact_b2=None,
                resolution_mean=0.1,
            ),
            _prediction(2, 0, support_score=0.0),
            _prediction(2, 1, support_score=0.1),
        ]
        support = _support()
        calibration = fit_selective_calibration_g2(
            predictions, support, alpha=0.5, required_episodes=2
        )
        self.assertEqual(calibration.selected_candidates, 2)
        self.assertEqual(calibration.selected_bad_candidates, 1)
        self.assertEqual(calibration.episode_residuals[1][1], 0.0)
        self.assertTrue(math.isinf(calibration.q))
        self.assertEqual(calibration.manifest()["q"], "Infinity")


class SelectionTests(unittest.TestCase):
    def test_absolute_b2_rescues_resolved_candidate_when_incumbent_unresolved(self) -> None:
        support = _support()
        calibration = _b2_calibration(support)
        predictions = [
            _prediction(
                1,
                0,
                support_score=0.0,
                resolved=False,
                exact_b2=None,
                finite_pair=False,
            ),
            _prediction(
                1,
                1,
                b2_lcb=1.0,
                exact_b2=3.0,
                finite_pair=False,
                score_lcb=2.0,
            ),
        ]
        selected = select_candidates_g2(predictions, support, calibration)[0]
        self.assertTrue(selected.override)
        self.assertTrue(selected.exact_candidate_resolved)
        self.assertFalse(selected.exact_finite_pair)
        self.assertEqual(selected.exact_b2, 3.0)

    def test_final_ranks_certified_candidates_and_tie_abstains(self) -> None:
        support = _support()
        calibration = _b2_calibration(support, q=0.5)
        predictions = [
            _prediction(1, 0, support_score=0.0),
            _prediction(1, 1, b2_lcb=1.0, score_lcb=2.0),
            _prediction(1, 2, b2_lcb=1.0, score_lcb=3.0),
        ]
        selected = select_candidates_g2(predictions, support, calibration)[0]
        self.assertEqual(selected.ordinal, 2)

        tied = copy.deepcopy(predictions)
        tied[1]["score_lcb"] = 3.0
        selected = select_candidates_g2(tied, support, calibration)[0]
        self.assertFalse(selected.override)
        self.assertEqual(selected.reasons, ("conservative-score-tie",))

    def test_score_veto_and_action_identical_never_reenter_at_selection(self) -> None:
        support = _support()
        calibration = _b2_calibration(support)
        incumbent_action = "same"
        predictions = [
            _prediction(1, 0, support_score=0.0, action=incumbent_action),
            _prediction(1, 1, action=incumbent_action, score_lcb=10.0),
            _prediction(1, 2, score_lcb=0.0, b2_lcb=10.0),
        ]
        selected = select_candidates_g2(predictions, support, calibration)[0]
        self.assertFalse(selected.override)
        self.assertEqual(selected.reasons, ("no-certified-alternative",))


class ViabilityTests(unittest.TestCase):
    @staticmethod
    def _selection(
        seed: int,
        *,
        override: bool,
        b2: float | None = 1.0,
        score: float = 2.0,
    ) -> G2SelectedCandidate:
        return G2SelectedCandidate(
            seed,
            f"q-{seed}",
            int(override),
            f"candidate-{seed}",
            override,
            True,
            True,
            False,
            b2,
            0.0,
            score,
            (),
        )

    def test_viability_requires_clustered_coverage_and_nonnegative_b2(self) -> None:
        support = _support()
        calibration = _b2_calibration(support)
        sparse = tuple(
            self._selection(seed, override=seed == 1) for seed in range(1, 17)
        )
        sparse_report = viability_report_g2(sparse, calibration)
        self.assertFalse(
            sparse_report["gates"]["positive_clustered_coverage_lcb"]
        )
        self.assertFalse(sparse_report["passed"])

        negative = tuple(
            self._selection(seed, override=True, b2=-0.01 if seed == 1 else 1.0)
            for seed in range(1, 17)
        )
        negative_report = viability_report_g2(negative, calibration)
        self.assertFalse(
            negative_report["gates"]["all_nonnegative_absolute_b2"]
        )
        self.assertFalse(negative_report["passed"])

        missing = tuple(
            self._selection(seed, override=True, b2=None if seed == 1 else 1.0)
            for seed in range(1, 17)
        )
        missing_report = viability_report_g2(missing, calibration)
        self.assertFalse(
            missing_report["gates"]["complete_finite_absolute_b2"]
        )
        self.assertEqual(missing_report["missing_or_nonfinite_absolute_b2"], 1)
        self.assertFalse(missing_report["passed"])

        passing = tuple(
            self._selection(seed, override=True) for seed in range(1, 17)
        )
        passing_report = viability_report_g2(passing, calibration)
        self.assertGreater(
            passing_report["coverage_seed_clustered_lcb_95"], 0
        )
        self.assertTrue(passing_report["passed"])


if __name__ == "__main__":
    unittest.main()
