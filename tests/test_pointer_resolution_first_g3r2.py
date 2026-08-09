from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from irisu_pointer.resolution_first import FEATURE_NAMES
from irisu_pointer.resolution_first_g2 import BoardBranchRecord
from irisu_pointer.resolution_first_g3r2 import (
    PRESERVED_BLOCK_WIDTH,
    WIDE_FEATURE_NAMES,
    WIDE_FEATURE_WIDTH,
    BoostConfig,
    FeatureBinner,
    G3R2Config,
    HistogramNewtonBoost,
    PairGroups,
    WideBoard,
    load_checkpoint_g3r2,
    save_checkpoint_g3r2,
    train_resolution_first_g3r2,
    wide_board_from_records,
    whole_seed_folds,
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
    source: int,
    destination: int,
    resolved: bool,
    identity_leak: bool = False,
) -> BoardBranchRecord:
    candidate = [0.0] * len(FEATURE_NAMES)
    candidate[0] = 0.1 * ordinal
    bodies = []
    for index in range(4):
        row = [0.0] * len(TeacherStateEncoder.schema.body_features)
        row[TeacherStateEncoder.schema.body_features.index("effect_x_norm")] = (
            0.1 + 0.2 * index
        )
        row[TeacherStateEncoder.schema.body_features.index("effect_y_norm")] = (
            0.2 + 0.1 * index
        )
        row[TeacherStateEncoder.schema.body_features.index("color_0") + index] = 1.0
        if identity_leak and index == 0:
            row[TeacherStateEncoder.schema.body_features.index("id_scaled")] = 0.4
        bodies.append(tuple(row))
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
        exact_unsafe=not resolved,
        severe_unsafe=False,
        b2=2.0 if resolved else None,
        delta_b2=0.0 if resolved else None,
        score_advantage=1.0 if resolved else -1.0,
        source_sha256=f"{seed:064x}"[-64:],
        global_features=tuple([0.25] + [0.0] * 11),
        phase_features=(0.0, 0.5, 0.25),
        body_features=tuple(bodies),
        body_chain_groups=(0, 0, 1, 1),
        body_grouped_flags=(True, True, True, True),
        body_color_groups=(0, 1, 2, 3),
        source_index=source,
        destination_index=destination,
        incumbent_source_index=0,
        incumbent_destination_index=1,
        observation_sha256=f"{seed + 100:064x}"[-64:],
    )


def _records(seed: int = 1) -> tuple[BoardBranchRecord, ...]:
    return (
        _record(seed, 0, source=0, destination=1, resolved=True),
        _record(seed, 1, source=0, destination=1, resolved=False),
        _record(seed, 2, source=2, destination=3, resolved=True),
    )


def _synthetic_board(seed_count: int = 11) -> WideBoard:
    features: list[np.ndarray] = []
    labels: list[bool] = []
    seeds: list[int] = []
    queries: list[str] = []
    sources: list[int] = []
    destinations: list[int] = []
    ordinals: list[int] = []
    for seed in range(1, seed_count + 1):
        for index in range(12):
            left = float((index // 3) % 2)
            right = float(index % 3 != 0)
            row = np.zeros(WIDE_FEATURE_WIDTH, dtype=np.float64)
            row[0] = left
            row[1] = right
            row[126] = left * right
            row[127] = (seed % 3) / 3
            row[400] = np.sin(seed + index)
            label = bool(left and right)
            features.append(row)
            labels.append(label)
            seeds.append(seed)
            queries.append(f"q-{seed}-{index // 6}")
            pair = (index // 2) % 3
            sources.append(pair * 2)
            destinations.append(pair * 2 + 1)
            ordinals.append(index + 1)
    return WideBoard(
        np.asarray(features),
        np.asarray(labels),
        np.asarray(seeds),
        np.asarray(queries, dtype=object),
        np.asarray(sources),
        np.asarray(destinations),
        np.asarray(ordinals),
    )


def _test_config() -> G3R2Config:
    return G3R2Config(
        folds=5,
        candidate=BoostConfig(
            rounds=8,
            depth=3,
            learning_rate=0.2,
            l2=0.5,
            minimum_leaf=1,
            maximum_features=128,
            bins=4,
            balance_classes=True,
        ),
        pair=BoostConfig(
            rounds=8,
            depth=3,
            learning_rate=0.2,
            l2=0.5,
            minimum_leaf=1,
            maximum_features=128,
            bins=4,
            balance_classes=False,
        ),
    )


class WideFeatureTests(unittest.TestCase):
    def test_exact_width_names_and_absolute_identity_zeroing(self) -> None:
        board = wide_board_from_records(_records())
        self.assertEqual(board.features.shape, (2, WIDE_FEATURE_WIDTH))
        self.assertEqual(len(WIDE_FEATURE_NAMES), WIDE_FEATURE_WIDTH)
        self.assertEqual(
            tuple(WIDE_FEATURE_NAMES[:PRESERVED_BLOCK_WIDTH])[-1],
            f"candidate_minus_incumbent_{FEATURE_NAMES[-1]}",
        )
        for name in WIDE_FEATURE_NAMES:
            if "id_scaled" in name or "chain_id_scaled" in name:
                self.assertTrue(name.endswith("_zero"))
                np.testing.assert_array_equal(
                    board.features[:, WIDE_FEATURE_NAMES.index(name)], 0.0
                )

        leaking = list(_records())
        leaking[0] = _record(
            1,
            0,
            source=0,
            destination=1,
            resolved=True,
            identity_leak=True,
        )
        with self.assertRaisesRegex(RuntimeError, "identity leaked"):
            wide_board_from_records(leaking)

    def test_candidate_permutation_is_row_equivariant(self) -> None:
        left = wide_board_from_records(_records())
        rows = _records()
        right = wide_board_from_records((rows[0], rows[2], rows[1]))
        left_by_ordinal = {
            int(ordinal): row
            for ordinal, row in zip(left.ordinals, left.features, strict=True)
        }
        right_by_ordinal = {
            int(ordinal): row
            for ordinal, row in zip(right.ordinals, right.features, strict=True)
        }
        self.assertEqual(set(left_by_ordinal), set(right_by_ordinal))
        for ordinal in left_by_ordinal:
            np.testing.assert_array_equal(
                left_by_ordinal[ordinal], right_by_ordinal[ordinal]
            )

    def test_pair_partition_only_aggregates_and_expands(self) -> None:
        board = _synthetic_board(6)
        groups = PairGroups.from_board(board)
        self.assertLess(len(groups.keys), len(board.labels))
        values = np.arange(len(groups.keys), dtype=np.float64)
        expanded = groups.expand(values, len(board.labels))
        for index, rows in enumerate(groups.row_groups):
            np.testing.assert_array_equal(expanded[list(rows)], float(index))
        self.assertFalse(
            any("query_id" in name or "source_index" in name for name in WIDE_FEATURE_NAMES)
        )


class BoosterTests(unittest.TestCase):
    def test_feature_screen_is_train_only_and_keeps_complete_prefix(self) -> None:
        generator = np.random.default_rng(7)
        train = generator.normal(size=(200, WIDE_FEATURE_WIDTH))
        target = (train[:, 500] > 0).astype(float)
        config = BoostConfig(
            rounds=2,
            minimum_leaf=2,
            maximum_features=180,
            bins=16,
        )
        first = FeatureBinner.fit(train, target, config)
        heldout = generator.normal(size=(50, WIDE_FEATURE_WIDTH))
        heldout[:, 700] = 1e9
        first.transform(heldout)
        second = FeatureBinner.fit(train, target, config)
        self.assertEqual(first, second)
        self.assertEqual(
            first.selected_columns[:PRESERVED_BLOCK_WIDTH],
            tuple(range(PRESERVED_BLOCK_WIDTH)),
        )
        self.assertIn(500, first.selected_columns)
        self.assertNotIn(700, first.selected_columns)
        self.assertTrue(all(type(column) is int for column in first.selected_columns))
        json.dumps(first.manifest(), allow_nan=False)

    def test_depth_three_tree_learns_an_interaction(self) -> None:
        rows = []
        target = []
        seeds = []
        for repeat in range(40):
            for left, right in ((0, 0), (0, 1), (1, 0), (1, 1)):
                row = np.zeros(WIDE_FEATURE_WIDTH)
                row[0], row[1] = left, right
                rows.append(row)
                target.append(float(left and right))
                seeds.append(repeat)
        model = HistogramNewtonBoost.fit(
            np.asarray(rows),
            np.asarray(target),
            np.asarray(seeds),
            BoostConfig(
                rounds=12,
                depth=3,
                learning_rate=0.3,
                l2=0.25,
                minimum_leaf=4,
                maximum_features=126,
                bins=4,
            ),
        )
        self.assertTrue(any(tree.depth > 1 for tree in model.trees))
        scores = model.probabilities(np.asarray(rows))
        self.assertGreater(scores[np.asarray(target, dtype=bool)].min(), 0.7)
        self.assertLess(scores[~np.asarray(target, dtype=bool)].max(), 0.3)


class CrossFitCheckpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = _synthetic_board()
        cls.model = train_resolution_first_g3r2(
            cls.board, config=_test_config()
        )

    def test_seed_folds_are_complete_disjoint_and_deterministic(self) -> None:
        expected = whole_seed_folds(self.board.unique_seeds, 5)
        self.assertEqual(
            tuple(fold.heldout_seeds for fold in self.model.folds), expected
        )
        assignments = self.model.seed_folds
        self.assertEqual(set(assignments), set(self.board.unique_seeds))
        predictions = self.model.predict(self.board, oof=True)
        self.assertEqual(len(predictions), len(self.board.labels))
        self.assertTrue(all(row.primary_score == row.candidate_mean for row in predictions))

    def test_production_eight_fold_requires_an_explicit_bound_partition(self) -> None:
        with self.assertRaisesRegex(ValueError, "configuration"):
            G3R2Config(folds=8)
        partition = tuple(
            tuple(self.board.unique_seeds[index::8]) for index in range(8)
        )
        config = G3R2Config(
            folds=8,
            partition_id="campaign-evidence-core-v1",
            seed_partition=partition,
            candidate=_test_config().candidate,
            pair=_test_config().pair,
        )
        model = train_resolution_first_g3r2(self.board, config=config)
        self.assertEqual(
            tuple(fold.heldout_seeds for fold in model.folds), partition
        )
        self.assertEqual(len(model.partition_sha256), 64)

    def test_oof_and_checkpoint_are_deterministic(self) -> None:
        repeated = train_resolution_first_g3r2(
            self.board, config=_test_config()
        )
        self.assertEqual(self.model.manifest(), repeated.manifest())
        self.assertEqual(
            self.model.predict(self.board, oof=True),
            repeated.predict(self.board, oof=True),
        )

    def test_checkpoint_roundtrip_and_all_identity_tamper_reject(self) -> None:
        before = self.model.predict(self.board, oof=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "g3r2.json"
            digest = save_checkpoint_g3r2(
                checkpoint, self.model, metadata={"stage": "unit"}
            )
            loaded, metadata = load_checkpoint_g3r2(checkpoint)
            self.assertEqual(len(digest), 64)
            self.assertEqual(metadata, {"stage": "unit"})
            self.assertEqual(loaded.predict(self.board, oof=True), before)
            with self.assertRaisesRegex(RuntimeError, "metadata expectation"):
                load_checkpoint_g3r2(
                    checkpoint, expected_metadata={"stage": "foreign"}
                )
            with self.assertRaisesRegex(RuntimeError, "partition expectation"):
                load_checkpoint_g3r2(
                    checkpoint, expected_partition_sha256="0" * 64
                )
            with self.assertRaisesRegex(RuntimeError, "model expectation"):
                load_checkpoint_g3r2(
                    checkpoint, expected_model_sha256="0" * 64
                )

            metadata_tamper = json.loads(checkpoint.read_text())
            metadata_tamper["metadata"]["stage"] = "foreign"
            path = root / "metadata-tamper.json"
            path.write_bytes(_canonical(metadata_tamper))
            with self.assertRaisesRegex(RuntimeError, "identity"):
                load_checkpoint_g3r2(path)

            tree_tamper = json.loads(checkpoint.read_text())
            tree_tamper["model"]["folds"][0]["candidate"]["trees"][0]["value"] += 0.1
            path = root / "tree-tamper.json"
            path.write_bytes(_canonical(tree_tamper))
            with self.assertRaisesRegex(RuntimeError, "identity"):
                load_checkpoint_g3r2(path)

            partition = json.loads(checkpoint.read_text())
            removed = partition["model"]["folds"][0]["heldout_seeds"].pop()
            partition["model"]["fold_by_seed"] = [
                row
                for row in partition["model"]["fold_by_seed"]
                if row[0] != removed
            ]
            partition["model_sha256"] = _digest(partition["model"])
            body = {
                key: partition[key]
                for key in partition
                if key != "checkpoint_sha256"
            }
            partition["checkpoint_sha256"] = _digest(body)
            path = root / "partition-tamper.json"
            path.write_bytes(_canonical(partition))
            with self.assertRaisesRegex(RuntimeError, "manifest"):
                load_checkpoint_g3r2(path)

            noncanonical = root / "noncanonical.json"
            noncanonical.write_text(json.dumps(json.loads(checkpoint.read_text()), indent=2))
            with self.assertRaisesRegex(RuntimeError, "noncanonical"):
                load_checkpoint_g3r2(noncanonical)


if __name__ == "__main__":
    unittest.main()
