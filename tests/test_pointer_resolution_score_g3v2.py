from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from irisu_pointer.resolution_first import FEATURE_NAMES
from irisu_pointer.resolution_first_g2 import BoardBranchRecord
from irisu_pointer.resolution_first_g3r2 import (
    WIDE_FEATURE_WIDTH,
    BoostConfig,
    WideBoard,
)
from irisu_pointer.resolution_score_g3 import (
    ResolutionScoreConfig,
    implied_score_advantage,
)
from irisu_pointer.resolution_score_g3v2 import (
    ResolutionScoreBoardV2,
    ResolutionScorePredictionV2,
    ScoreCandidateIdentity,
    StrictShieldCertificate,
    load_resolution_score_checkpoint_v2,
    resolution_score_board_v2_from_records,
    save_resolution_score_checkpoint_v2,
    select_strict_score_candidate,
    strict_certificate,
    strict_shield_certified,
    train_resolution_score_g3v2,
)
from irisu_rl.encoding import TeacherStateEncoder


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _record(
    seed: int,
    ordinal: int,
    *,
    resolved: bool = True,
    finite_pair: bool = True,
    unsafe: bool = False,
    severe: bool = False,
    b2: float | None = 10.0,
    delta: float | None = 2.0,
    score: float = 32.0,
) -> BoardBranchRecord:
    candidate = [0.0] * len(FEATURE_NAMES)
    candidate[0] = ordinal / 10
    bodies = []
    for index in range(4):
        body = [0.0] * len(TeacherStateEncoder.schema.body_features)
        body[TeacherStateEncoder.schema.body_features.index("effect_x_norm")] = (
            0.1 + index * 0.2
        )
        body[TeacherStateEncoder.schema.body_features.index("effect_y_norm")] = (
            0.2 + index * 0.1
        )
        body[TeacherStateEncoder.schema.body_features.index("color_0") + index] = 1.0
        bodies.append(tuple(body))
    return BoardBranchRecord(
        seed=seed,
        query_id=f"q-{seed}",
        ordinal=ordinal,
        candidate_id=f"candidate-{seed}-{ordinal}",
        action_id=f"action-{seed}-{ordinal}",
        signature="fresh-match|analytic-strong",
        features=tuple(candidate),
        candidate_resolved=resolved,
        finite_pair=finite_pair,
        exact_unsafe=unsafe,
        severe_unsafe=severe,
        b2=b2,
        delta_b2=delta,
        score_advantage=score,
        source_sha256=f"{seed + ordinal:064x}"[-64:],
        global_features=tuple([0.25] + [0.0] * 11),
        phase_features=(0.0, 0.5, 0.25),
        body_features=tuple(bodies),
        body_chain_groups=(0, 0, 1, 1),
        body_grouped_flags=(True, True, True, True),
        body_color_groups=(0, 1, 2, 3),
        source_index=0 if ordinal < 2 else 2,
        destination_index=1 if ordinal < 2 else 3,
        incumbent_source_index=0,
        incumbent_destination_index=1,
        observation_sha256=f"{seed + 100:064x}"[-64:],
    )


def _records(seed: int = 1) -> tuple[BoardBranchRecord, ...]:
    return (
        _record(seed, 0, delta=0.0, score=0.0),
        _record(seed, 1, delta=2.0, score=64.0),
        _record(seed, 2, delta=0.0, score=1000.0),
        _record(seed, 3, delta=-2.0, score=-1000.0),
    )


def _synthetic_board(seed_count: int = 11) -> ResolutionScoreBoardV2:
    features, labels, seeds, queries = [], [], [], []
    sources, destinations, ordinals, identities, scores, strict = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for seed in range(1, seed_count + 1):
        for index in range(12):
            value = (index - 5.5) / 5.5
            row = np.zeros(WIDE_FEATURE_WIDTH)
            row[0] = value
            row[1] = value * value
            row[126] = np.sin(seed + index)
            ordinal = index + 1
            identity = ScoreCandidateIdentity(
                seed, f"q-{seed}-{index // 4}", ordinal,
                f"candidate-{seed}-{ordinal}", f"action-{seed}-{ordinal}"
            )
            features.append(row)
            labels.append(index % 3 == 1)
            seeds.append(seed)
            queries.append(identity.query_id)
            sources.append(index * 2)
            destinations.append(index * 2 + 1)
            ordinals.append(ordinal)
            identities.append(identity)
            scores.append(320.0 * value + (seed % 3 - 1) * 4)
            strict.append(index % 3 == 1)
    wide = WideBoard(
        np.asarray(features),
        np.asarray(labels),
        np.asarray(seeds),
        np.asarray(queries, dtype=object),
        np.asarray(sources),
        np.asarray(destinations),
        np.asarray(ordinals),
    )
    return ResolutionScoreBoardV2(
        wide, tuple(identities), np.asarray(scores), np.asarray(strict)
    )


def _config(
    *,
    folds: int = 5,
    partition_id: str = "development-internal-sorted-v1",
    partition: tuple[tuple[int, ...], ...] = (),
) -> ResolutionScoreConfig:
    return ResolutionScoreConfig(
        folds=folds,
        partition_id=partition_id,
        seed_partition=partition,
        boost=BoostConfig(
            rounds=6,
            depth=3,
            learning_rate=0.2,
            l2=0.5,
            minimum_leaf=1,
            maximum_features=128,
            preserved_features=126,
            bins=4,
            balance_classes=False,
        ),
    )


class StrictShieldTests(unittest.TestCase):
    def test_every_strict_condition_and_delta_tie_abstention(self) -> None:
        valid = _record(1, 1)
        self.assertTrue(strict_shield_certified(valid))
        self.assertEqual(strict_certificate(valid).delta_b2, 2.0)
        variants = (
            replace(valid, candidate_resolved=False),
            replace(valid, finite_pair=False),
            replace(valid, exact_unsafe=True),
            replace(valid, severe_unsafe=True),
            replace(valid, b2=None),
            replace(valid, b2=-1.0),
            replace(valid, delta_b2=None),
            replace(valid, delta_b2=0.0),
            replace(valid, delta_b2=-1.0),
        )
        self.assertTrue(all(not strict_shield_certified(row) for row in variants))
        for row in variants:
            with self.assertRaisesRegex(ValueError, "strict shield"):
                strict_certificate(row)

    def test_alignment_and_certificate_fields_do_not_enter_features(self) -> None:
        original = resolution_score_board_v2_from_records(_records())
        changed_rows = list(_records())
        changed_rows[1] = replace(
            changed_rows[1],
            finite_pair=False,
            exact_unsafe=True,
            severe_unsafe=True,
            b2=-10.0,
            delta_b2=-10.0,
        )
        changed = resolution_score_board_v2_from_records(changed_rows)
        np.testing.assert_array_equal(original.features, changed.features)
        self.assertNotEqual(
            original.strict_certified.tolist(), changed.strict_certified.tolist()
        )
        self.assertEqual(original.identities, changed.identities)
        manifest = original.manifest()
        self.assertFalse(manifest["certificate_or_identity_is_model_input"])
        self.assertFalse(manifest["score_target_is_model_input"])
        np.testing.assert_array_equal(original.wide.ordinals, (1, 2, 3))

    def test_variable_candidate_counts_are_supported_per_query(self) -> None:
        rows = (*_records(1), _record(2, 0, delta=0.0), _record(2, 1))
        board = resolution_score_board_v2_from_records(rows)
        self.assertEqual(len(board.features), 4)
        self.assertEqual(
            [(identity.seed, identity.ordinal) for identity in board.identities],
            [(1, 1), (1, 2), (1, 3), (2, 1)],
        )


class LearnerCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = _synthetic_board()
        cls.model = train_resolution_score_g3v2(cls.board, config=_config())

    def test_training_uses_only_strict_rows(self) -> None:
        scores = np.array(self.board.score_advantages, copy=True)
        scores[~self.board.strict_certified] *= -10000
        changed = ResolutionScoreBoardV2(
            self.board.wide,
            self.board.identities,
            scores,
            self.board.strict_certified,
        )
        repeated = train_resolution_score_g3v2(changed, config=_config())
        self.assertEqual(
            [fold.head.manifest() for fold in self.model.folds],
            [fold.head.manifest() for fold in repeated.folds],
        )
        self.assertNotEqual(
            self.model.training_dataset_sha256,
            repeated.training_dataset_sha256,
        )

    def test_whole_seed_oof_permutation_and_determinism(self) -> None:
        predictions = self.model.predict(self.board, oof=True)
        self.assertEqual(len(predictions), len(self.board.features))
        repeated = train_resolution_score_g3v2(self.board, config=_config())
        self.assertEqual(self.model.manifest(), repeated.manifest())
        self.assertEqual(predictions, repeated.predict(self.board, oof=True))
        order = np.arange(len(self.board.features))[::-1]
        permuted = {
            row.identity: row.soft_score_mean
            for row in self.model.predict(self.board.take(order), oof=True)
        }
        original = {row.identity: row.soft_score_mean for row in predictions}
        self.assertEqual(original, permuted)

    def test_eight_fold_production_partition_is_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "configuration"):
            ResolutionScoreConfig(folds=8)
        partition = tuple(
            tuple(self.board.unique_seeds[index::8]) for index in range(8)
        )
        model = train_resolution_score_g3v2(
            self.board,
            config=_config(
                folds=8,
                partition_id="strict-campaign-partition-v1",
                partition=partition,
            ),
        )
        self.assertEqual(
            tuple(fold.heldout_seeds for fold in model.folds), partition
        )

    def test_checkpoint_roundtrip_and_recomputed_tamper_reject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "strict.json"
            save_resolution_score_checkpoint_v2(
                path, self.model, metadata={"stage": "unit"}
            )
            loaded, metadata = load_resolution_score_checkpoint_v2(
                path,
                expected_metadata={"stage": "unit"},
                expected_partition_sha256=self.model.partition_sha256,
                expected_model_sha256=self.model.sha256,
            )
            self.assertEqual(metadata, {"stage": "unit"})
            self.assertEqual(loaded.sha256, self.model.sha256)

            tamper = json.loads(path.read_text())
            tamper["model"]["dependency_identities"][0][1] = "0" * 64
            tamper["model_sha256"] = _digest(tamper["model"])
            body = {
                key: tamper[key] for key in tamper if key != "checkpoint_sha256"
            }
            tamper["checkpoint_sha256"] = _digest(body)
            bad = root / "tamper.json"
            bad.write_bytes(_canonical(tamper))
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                load_resolution_score_checkpoint_v2(bad)


class IdentityBoundSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.incumbent = ScoreCandidateIdentity(7, "q", 0, "inc", "inc-action")
        self.first = ScoreCandidateIdentity(7, "q", 1, "c1", "a1")
        self.second = ScoreCandidateIdentity(7, "q", 2, "c2", "a2")

    @staticmethod
    def prediction(
        identity: ScoreCandidateIdentity,
        soft: float,
        *,
        std: float = 0.0,
        implied: float | None = None,
    ) -> ResolutionScorePredictionV2:
        return ResolutionScorePredictionV2(
            identity,
            soft,
            std,
            implied_score_advantage(soft) if implied is None else implied,
        )

    @staticmethod
    def certificate(identity: ScoreCandidateIdentity) -> StrictShieldCertificate:
        return StrictShieldCertificate(identity, True, True, False, False, 2.0, 1.0)

    def test_selection_uses_full_certificate_identity_and_tie_breaks(self) -> None:
        predictions = (
            self.prediction(self.second, 0.75),
            self.prediction(self.first, 0.75),
        )
        certificates = (
            self.certificate(self.second),
            self.certificate(self.first),
        )
        selected = select_strict_score_candidate(
            predictions, certificates, incumbent=self.incumbent
        )
        changed = select_strict_score_candidate(
            tuple(reversed(predictions)),
            tuple(reversed(certificates)),
            incumbent=self.incumbent,
        )
        self.assertEqual(selected, self.first)
        self.assertEqual(changed, selected)

    def test_uncertified_positive_is_ignored_and_fallback_is_exact_incumbent(self) -> None:
        predictions = (
            self.prediction(self.first, 0.99),
            self.prediction(self.second, 0.49),
        )
        self.assertEqual(
            select_strict_score_candidate(
                predictions,
                (self.certificate(self.second),),
                incumbent=self.incumbent,
            ),
            self.incumbent,
        )
        with self.assertRaisesRegex(ValueError, "ordinal zero"):
            select_strict_score_candidate(
                predictions,
                (),
                incumbent=replace(self.incumbent, ordinal=3),
            )

    def test_missing_duplicate_mixed_and_mismatched_identities_reject(self) -> None:
        predictions = (self.prediction(self.first, 0.75),)
        with self.assertRaisesRegex(ValueError, "lacks an exact prediction"):
            select_strict_score_candidate(
                predictions,
                (self.certificate(self.second),),
                incumbent=self.incumbent,
            )
        duplicate_ordinal = replace(self.first, candidate_id="foreign")
        with self.assertRaisesRegex(ValueError, "duplicate identity"):
            select_strict_score_candidate(
                (
                    self.prediction(self.first, 0.75),
                    self.prediction(duplicate_ordinal, 0.7),
                ),
                (),
                incumbent=self.incumbent,
            )
        mixed = replace(self.first, query_id="other")
        with self.assertRaisesRegex(ValueError, "mix seed or query"):
            select_strict_score_candidate(
                (self.prediction(mixed, 0.75),),
                (),
                incumbent=self.incumbent,
            )
        mismatched_action = replace(self.first, action_id="wrong")
        with self.assertRaisesRegex(ValueError, "lacks an exact prediction"):
            select_strict_score_candidate(
                predictions,
                (self.certificate(mismatched_action),),
                incumbent=self.incumbent,
            )

    def test_nonfinite_or_inconsistent_prediction_rejects(self) -> None:
        for prediction in (
            self.prediction(self.first, float("nan"), implied=0.0),
            self.prediction(self.first, 0.75, std=float("inf")),
            self.prediction(self.first, 0.75, implied=999.0),
        ):
            with self.assertRaisesRegex(ValueError, "inconsistent"):
                select_strict_score_candidate(
                    (prediction,), (), incumbent=self.incumbent
                )


if __name__ == "__main__":
    unittest.main()
