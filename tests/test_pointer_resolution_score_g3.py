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
    ResolutionScoreBoard,
    ResolutionScoreConfig,
    ResolutionScorePrediction,
    implied_score_advantage,
    load_resolution_score_checkpoint,
    resolution_score_board_from_records,
    save_resolution_score_checkpoint,
    select_shielded_score_candidate,
    soft_score_target,
    train_resolution_score_g3,
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
    resolved: bool,
    unsafe: bool,
    score: float,
) -> BoardBranchRecord:
    candidate = [0.0] * len(FEATURE_NAMES)
    candidate[0] = ordinal / 10
    bodies = []
    for index in range(4):
        body = [0.0] * len(TeacherStateEncoder.schema.body_features)
        body[TeacherStateEncoder.schema.body_features.index("effect_x_norm")] = (
            0.1 + 0.2 * index
        )
        body[TeacherStateEncoder.schema.body_features.index("effect_y_norm")] = (
            0.2 + 0.1 * index
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
        finite_pair=resolved,
        exact_unsafe=unsafe,
        severe_unsafe=False,
        b2=2.0 if resolved else None,
        delta_b2=0.0 if resolved else None,
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
        _record(seed, 0, resolved=True, unsafe=False, score=0.0),
        _record(seed, 1, resolved=True, unsafe=False, score=64.0),
        _record(seed, 2, resolved=True, unsafe=True, score=10000.0),
        _record(seed, 3, resolved=False, unsafe=False, score=-10000.0),
    )


def _synthetic_board(seed_count: int = 11) -> ResolutionScoreBoard:
    features: list[np.ndarray] = []
    labels: list[bool] = []
    seeds: list[int] = []
    queries: list[str] = []
    sources: list[int] = []
    destinations: list[int] = []
    ordinals: list[int] = []
    scores: list[float] = []
    shield: list[bool] = []
    for seed in range(1, seed_count + 1):
        for index in range(12):
            value = (index - 5.5) / 5.5
            row = np.zeros(WIDE_FEATURE_WIDTH)
            row[0] = value
            row[1] = value * value
            row[126] = np.sin(seed + index)
            features.append(row)
            labels.append(index % 3 != 0)
            seeds.append(seed)
            queries.append(f"q-{seed}-{index // 4}")
            sources.append((index // 2) * 2)
            destinations.append((index // 2) * 2 + 1)
            ordinals.append(index + 1)
            scores.append(384.0 * value + (seed % 3 - 1) * 4.0)
            shield.append(index % 3 != 0)
    wide = WideBoard(
        np.asarray(features),
        np.asarray(labels),
        np.asarray(seeds),
        np.asarray(queries, dtype=object),
        np.asarray(sources),
        np.asarray(destinations),
        np.asarray(ordinals),
    )
    return ResolutionScoreBoard(wide, np.asarray(scores), np.asarray(shield))


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


class BoardAndTargetTests(unittest.TestCase):
    def test_record_alignment_shield_definition_and_incumbent_exclusion(self) -> None:
        board = resolution_score_board_from_records(_records())
        self.assertEqual(board.features.shape, (3, WIDE_FEATURE_WIDTH))
        np.testing.assert_array_equal(board.ordinals, (1, 2, 3))
        np.testing.assert_array_equal(board.score_advantages, (64, 10000, -10000))
        np.testing.assert_array_equal(
            board.shield_certified, (True, False, False)
        )

    def test_shield_and_score_labels_are_not_feature_inputs(self) -> None:
        original = resolution_score_board_from_records(_records())
        changed_rows = list(_records())
        changed_rows[1] = replace(
            changed_rows[1], candidate_resolved=False, exact_unsafe=True
        )
        changed = resolution_score_board_from_records(changed_rows)
        np.testing.assert_array_equal(original.features, changed.features)
        self.assertNotEqual(
            original.shield_certified.tolist(), changed.shield_certified.tolist()
        )
        manifest = original.manifest()
        self.assertFalse(manifest["shield_certified_is_model_input"])
        self.assertFalse(manifest["score_target_is_model_input"])

    def test_soft_target_is_fixed_monotone_and_invertible(self) -> None:
        scores = np.asarray([-512.0, -1.0, 0.0, 1.0, 512.0])
        target = soft_score_target(scores)
        self.assertTrue(np.all(np.diff(target) > 0))
        self.assertEqual(target[2], 0.5)
        for score, probability in zip(scores, target, strict=True):
            self.assertAlmostEqual(
                implied_score_advantage(float(probability)), float(score), places=8
            )


class LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = _synthetic_board()
        cls.model = train_resolution_score_g3(cls.board, config=_config())

    def test_only_shield_certified_rows_affect_training(self) -> None:
        changed_scores = np.array(self.board.score_advantages, copy=True)
        changed_scores[~self.board.shield_certified] *= -1000.0
        changed = ResolutionScoreBoard(
            self.board.wide, changed_scores, self.board.shield_certified
        )
        repeated = train_resolution_score_g3(changed, config=_config())
        self.assertNotEqual(
            self.model.training_dataset_sha256,
            repeated.training_dataset_sha256,
        )
        self.assertEqual(
            [fold.head.manifest() for fold in self.model.folds],
            [fold.head.manifest() for fold in repeated.folds],
        )
        expected_total = int(self.board.shield_certified.sum())
        for fold in self.model.folds:
            heldout_rows = self.board.shield_certified & np.isin(
                self.board.seeds, fold.heldout_seeds
            )
            self.assertEqual(
                fold.training_shielded_rows,
                expected_total - int(heldout_rows.sum()),
            )

    def test_whole_seed_oof_is_complete_and_deterministic(self) -> None:
        assigned = self.model.seed_folds
        self.assertEqual(set(assigned), set(self.board.unique_seeds))
        predictions = self.model.predict(self.board, oof=True)
        self.assertEqual(len(predictions), len(self.board.features))
        self.assertTrue(all(row.soft_score_std == 0.0 for row in predictions))
        repeated = train_resolution_score_g3(self.board, config=_config())
        self.assertEqual(self.model.manifest(), repeated.manifest())
        self.assertEqual(
            predictions, repeated.predict(self.board, oof=True)
        )

    def test_candidate_permutation_is_prediction_equivariant(self) -> None:
        order = np.arange(len(self.board.features))[::-1]
        permuted = self.board.take(order)
        original = {
            (row.seed, row.query_id, row.ordinal): row.soft_score_mean
            for row in self.model.predict(self.board, oof=True)
        }
        changed = {
            (row.seed, row.query_id, row.ordinal): row.soft_score_mean
            for row in self.model.predict(permuted, oof=True)
        }
        self.assertEqual(original, changed)

    def test_production_eight_fold_requires_explicit_partition(self) -> None:
        with self.assertRaisesRegex(ValueError, "configuration"):
            ResolutionScoreConfig(folds=8)
        partition = tuple(
            tuple(self.board.unique_seeds[index::8]) for index in range(8)
        )
        model = train_resolution_score_g3(
            self.board,
            config=_config(
                folds=8,
                partition_id="campaign-evidence-core-v1",
                partition=partition,
            ),
        )
        self.assertEqual(
            tuple(fold.heldout_seeds for fold in model.folds), partition
        )

    def test_checkpoint_roundtrip_type_and_tamper_rejection(self) -> None:
        before = self.model.predict(self.board, oof=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "score.json"
            digest = save_resolution_score_checkpoint(
                checkpoint, self.model, metadata={"stage": "unit"}
            )
            loaded, metadata = load_resolution_score_checkpoint(
                checkpoint,
                expected_metadata={"stage": "unit"},
                expected_partition_sha256=self.model.partition_sha256,
                expected_model_sha256=self.model.sha256,
            )
            self.assertEqual(len(digest), 64)
            self.assertEqual(metadata, {"stage": "unit"})
            self.assertEqual(loaded.predict(self.board, oof=True), before)

            metadata_tamper = json.loads(checkpoint.read_text())
            metadata_tamper["metadata"]["stage"] = "foreign"
            path = root / "metadata-tamper.json"
            path.write_bytes(_canonical(metadata_tamper))
            with self.assertRaisesRegex(RuntimeError, "identity"):
                load_resolution_score_checkpoint(path)

            type_tamper = json.loads(checkpoint.read_text())
            type_tamper["model"]["config"]["folds"] = True
            body = {
                key: type_tamper[key]
                for key in type_tamper
                if key != "checkpoint_sha256"
            }
            type_tamper["model_sha256"] = _digest(type_tamper["model"])
            body["model_sha256"] = type_tamper["model_sha256"]
            type_tamper["checkpoint_sha256"] = _digest(body)
            path = root / "type-tamper.json"
            path.write_bytes(_canonical(type_tamper))
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                load_resolution_score_checkpoint(path)

            dependency = json.loads(checkpoint.read_text())
            dependency["model"]["dependency_identities"][0][1] = "0" * 64
            dependency["model_sha256"] = _digest(dependency["model"])
            body = {
                key: dependency[key]
                for key in dependency
                if key != "checkpoint_sha256"
            }
            dependency["checkpoint_sha256"] = _digest(body)
            path = root / "dependency-tamper.json"
            path.write_bytes(_canonical(dependency))
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                load_resolution_score_checkpoint(path)


class SelectionTests(unittest.TestCase):
    @staticmethod
    def _prediction(ordinal: int, soft: float) -> ResolutionScorePrediction:
        return ResolutionScorePrediction(
            1,
            "q",
            ordinal,
            soft,
            0.0,
            implied_score_advantage(soft),
        )

    def test_selection_never_chooses_uncertified_and_falls_back(self) -> None:
        predictions = (
            self._prediction(1, 0.99),
            self._prediction(2, 0.75),
            self._prediction(3, 0.49),
        )
        self.assertEqual(
            select_shielded_score_candidate(predictions, {2, 3}), 2
        )
        self.assertEqual(
            select_shielded_score_candidate(predictions, {3}), 0
        )
        self.assertEqual(
            select_shielded_score_candidate(predictions, set()), 0
        )

    def test_selection_is_permutation_invariant_with_ordinal_tie_break(self) -> None:
        predictions = (
            self._prediction(8, 0.75),
            self._prediction(2, 0.75),
            self._prediction(5, 0.70),
        )
        expected = select_shielded_score_candidate(predictions, {2, 5, 8})
        changed = select_shielded_score_candidate(
            tuple(reversed(predictions)), {8, 5, 2}
        )
        self.assertEqual(expected, 2)
        self.assertEqual(changed, expected)


if __name__ == "__main__":
    unittest.main()
