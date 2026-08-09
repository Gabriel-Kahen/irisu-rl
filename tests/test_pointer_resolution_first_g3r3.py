from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from irisu_pointer import resolution_first_g3r3 as g3r3
from irisu_pointer.resolution_first_g3r3 import (
    WIDE_FEATURE_WIDTH,
    BoostConfigR3,
    FeatureBinnerR3,
    G3R3Config,
    HistogramNewtonBoostR3,
    TreeNodeR3,
    WideBoard,
    load_checkpoint_g3r3,
    save_checkpoint_g3r3,
    seed_equal_weights,
    train_resolution_first_g3r3,
    whole_seed_folds_g3r3,
)


DEPENDENCIES = {
    "g3r2-feature-core": hashlib.sha256(b"g3r2-feature-core").hexdigest(),
    "g3r3-learner-source": hashlib.sha256(b"g3r3-learner-source").hexdigest(),
}
METADATA = {"stage": "unit", "purpose": "hardened-roundtrip"}


def _synthetic_board(seed_count: int = 11) -> WideBoard:
    features: list[np.ndarray] = []
    labels: list[bool] = []
    seeds: list[int] = []
    queries: list[str] = []
    sources: list[int] = []
    destinations: list[int] = []
    ordinals: list[int] = []
    for seed in range(1, seed_count + 1):
        for index in range(8):
            left = float((index // 2) % 2)
            right = float(index % 2)
            row = np.zeros(WIDE_FEATURE_WIDTH, dtype=np.float64)
            row[0] = left
            row[1] = right
            row[126] = left * right
            row[127] = (seed % 3) / 3.0
            row[400] = math_value = float(np.sin(seed + index))
            row[500] = math_value * (1.0 if seed % 2 else -1.0)
            features.append(row)
            labels.append(bool(left and right))
            seeds.append(seed)
            queries.append(f"q-{seed}-{index // 4}")
            pair = (index // 2) % 2
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


def _boost(*, balance: bool, rounds: int = 4) -> BoostConfigR3:
    return BoostConfigR3(
        rounds=rounds,
        depth=3,
        learning_rate=0.2,
        l2=0.5,
        minimum_leaf=1,
        maximum_features=128,
        preserved_features=126,
        bins=4,
        balance_classes=balance,
    )


def _config() -> G3R3Config:
    return G3R3Config(
        folds=5,
        candidate=_boost(balance=True),
        pair=_boost(balance=False),
    )


class ExactTypeAndWeightTests(unittest.TestCase):
    def test_integer_config_seed_and_fold_types_are_exact_non_bool(self) -> None:
        for invalid in (True, np.int64(4), 4.0):
            with self.subTest(invalid=repr(invalid)):
                with self.assertRaises(ValueError):
                    BoostConfigR3(rounds=invalid)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            BoostConfigR3(rounds=4, depth=True)
        for invalid in (True, np.int64(5), 5.0):
            with self.subTest(folds=repr(invalid)):
                with self.assertRaises(ValueError):
                    G3R3Config(folds=invalid)  # type: ignore[arg-type]
                with self.assertRaises(ValueError):
                    whole_seed_folds_g3r3(list(range(10)), invalid)  # type: ignore[arg-type]
        for invalid_seed in (True, np.int64(3), 3.0):
            seeds = [1, 2, 4, 5, 6, 7, 8, 9, invalid_seed]
            with self.subTest(seed=repr(invalid_seed)):
                with self.assertRaises(ValueError):
                    whole_seed_folds_g3r3(seeds, 5)
        with self.assertRaises(ValueError):
            G3R3Config(
                folds=5,
                partition_id="explicit",
                seed_partition=((1,), (2,), (3,), (4,), (True,)),
            )

    def test_manifest_float_fields_reject_integer_encodings(self) -> None:
        boost = _boost(balance=True, rounds=1)
        for field in ("learning_rate", "l2"):
            manifest = boost.manifest()
            manifest[field] = int(manifest[field])
            with self.subTest(boost_field=field):
                with self.assertRaises(RuntimeError):
                    BoostConfigR3.from_manifest(manifest)

        board = _synthetic_board()
        head = HistogramNewtonBoostR3.fit(
            board.features,
            board.labels.astype(np.float64),
            board.seeds,
            boost,
        )
        binner_manifest = head.binner.manifest()
        threshold_row = next(
            row for row in binner_manifest["thresholds"] if row
        )
        threshold_row[0] = int(threshold_row[0])
        with self.assertRaises(RuntimeError):
            FeatureBinnerR3.from_manifest(binner_manifest)

        tree_manifest = head.trees[0].manifest()
        tree_manifest["value"] = int(tree_manifest["value"])
        with self.assertRaises(RuntimeError):
            TreeNodeR3.from_manifest(tree_manifest)

        head_manifest = head.manifest()
        head_manifest["base_logit"] = int(head_manifest["base_logit"])
        with self.assertRaises(RuntimeError):
            HistogramNewtonBoostR3.from_manifest(head_manifest)

        config_manifest = _config().manifest()
        config_manifest["blend_pair_weight"] = 0
        with self.assertRaises(RuntimeError):
            G3R3Config.from_manifest(config_manifest)

    def test_model_width_fields_reject_float_encodings(self) -> None:
        model = train_resolution_first_g3r3(
            _synthetic_board(), config=_config()
        )
        for field in ("feature_width", "preserved_block_width"):
            manifest = model.manifest()
            manifest[field] = float(manifest[field])
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError):
                    g3r3.ResolutionFirstG3R3.from_manifest(manifest)

    def test_each_seed_has_equal_loss_mass_without_balancing(self) -> None:
        seeds = np.asarray([1] * 2 + [2] * 5 + [3] * 11)
        targets = np.asarray([0, 1] + [0, 0, 1, 1, 1] + [0, 1] * 5 + [1])
        weights = seed_equal_weights(seeds, targets, False)
        masses = [weights[seeds == seed].sum() for seed in (1, 2, 3)]
        np.testing.assert_allclose(masses, masses[0], rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_each_seed_and_present_class_has_equal_mass_with_balancing(self) -> None:
        seeds = np.asarray([1] * 2 + [2] * 6 + [3] * 10)
        targets = np.asarray(
            [0, 1]
            + [0, 0, 0, 0, 0, 1]
            + [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        )
        weights = seed_equal_weights(seeds, targets, True)
        masses = [weights[seeds == seed].sum() for seed in (1, 2, 3)]
        np.testing.assert_allclose(masses, masses[0], rtol=0.0, atol=1e-12)
        for seed in (1, 2, 3):
            rows = seeds == seed
            positive = weights[rows & (targets == 1)].sum()
            negative = weights[rows & (targets == 0)].sum()
            self.assertAlmostEqual(float(positive), float(negative), places=12)

    def test_weighted_screening_and_quantiles_ignore_uniform_seed_duplication(self) -> None:
        features = np.zeros((8, WIDE_FEATURE_WIDTH), dtype=np.float64)
        seeds = np.asarray([1] * 4 + [2] * 4)
        targets = np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=np.float64)
        features[:, 500] = targets
        features[:, 501] = np.asarray([0, 1, 2, 3, 3, 2, 1, 0])
        config = BoostConfigR3(
            rounds=1,
            depth=1,
            learning_rate=0.1,
            l2=1.0,
            minimum_leaf=1,
            maximum_features=127,
            preserved_features=126,
            bins=4,
            balance_classes=True,
        )
        base = FeatureBinnerR3.fit(features, targets, seeds, config)
        expanded_rows = np.concatenate(
            (np.arange(4), np.repeat(np.arange(4, 8), 5))
        )
        expanded = FeatureBinnerR3.fit(
            features[expanded_rows],
            targets[expanded_rows],
            seeds[expanded_rows],
            config,
        )
        self.assertEqual(base, expanded)
        self.assertEqual(base.selected_columns[-1], 500)


class ScreeningTreeAndCrossFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = _synthetic_board()
        cls.model = train_resolution_first_g3r3(cls.board, config=_config())

    def test_feature_screen_is_train_only_with_deterministic_weighted_fit(self) -> None:
        heldout = self.model.folds[0].heldout_seeds
        mask = np.asarray(
            [int(seed) not in set(heldout) for seed in self.board.seeds]
        )
        expected = FeatureBinnerR3.fit(
            self.board.features[mask],
            self.board.labels[mask].astype(np.float64),
            self.board.seeds[mask],
            self.model.config.candidate,
        )
        self.assertEqual(self.model.folds[0].candidate.binner, expected)

        changed = np.array(self.board.features, copy=True)
        heldout_mask = ~mask
        changed[heldout_mask, 700] = (
            self.board.labels[heldout_mask].astype(float) * 1e9
        )
        expected_again = FeatureBinnerR3.fit(
            changed[mask],
            self.board.labels[mask].astype(np.float64),
            self.board.seeds[mask],
            self.model.config.candidate,
        )
        self.assertEqual(expected, expected_again)

    def test_nonleaf_threshold_must_be_nonnegative_and_inside_binner(self) -> None:
        leaf = TreeNodeR3(-1, -1, 0.0)
        with self.assertRaises(ValueError):
            TreeNodeR3(0, -1, 0.0, leaf, leaf)
        with self.assertRaises(ValueError):
            TreeNodeR3(True, 0, 0.0, leaf, leaf)  # type: ignore[arg-type]

        manifest = self.model.folds[0].candidate.manifest()
        nonleaf = next(
            node
            for tree in manifest["trees"]
            for node in _manifest_nodes(tree)
            if node["feature"] >= 0
        )
        bad_negative = copy.deepcopy(manifest)
        target = _matching_manifest_node(bad_negative["trees"], nonleaf)
        target["threshold"] = -1
        with self.assertRaises(RuntimeError):
            HistogramNewtonBoostR3.from_manifest(bad_negative)

        bad_width = copy.deepcopy(manifest)
        target = _matching_manifest_node(bad_width["trees"], nonleaf)
        target["threshold"] = len(
            bad_width["binner"]["thresholds"][target["feature"]]
        )
        with self.assertRaises(RuntimeError):
            HistogramNewtonBoostR3.from_manifest(bad_width)

    def test_whole_seed_oof_is_complete_disjoint_and_deterministic(self) -> None:
        expected = whole_seed_folds_g3r3(self.board.unique_seeds, 5)
        self.assertEqual(
            tuple(fold.heldout_seeds for fold in self.model.folds), expected
        )
        self.assertEqual(
            set(self.model.seed_folds), set(self.board.unique_seeds)
        )
        predictions = self.model.predict(self.board, oof=True)
        self.assertEqual(len(predictions), len(self.board.labels))
        self.assertTrue(all(row.candidate_std == 0.0 for row in predictions))

        repeated = train_resolution_first_g3r3(self.board, config=_config())
        self.assertEqual(self.model.manifest(), repeated.manifest())
        self.assertEqual(predictions, repeated.predict(self.board, oof=True))

    def test_explicit_eight_fold_partition_is_exact(self) -> None:
        with self.assertRaises(ValueError):
            G3R3Config(folds=8)
        partition = tuple(
            tuple(self.board.unique_seeds[index::8]) for index in range(8)
        )
        config = G3R3Config(
            folds=8,
            partition_id="campaign-whole-seed-v1",
            seed_partition=partition,
            candidate=_boost(balance=True, rounds=1),
            pair=_boost(balance=False, rounds=1),
        )
        model = train_resolution_first_g3r3(self.board, config=config)
        self.assertEqual(
            tuple(fold.heldout_seeds for fold in model.folds), partition
        )


def _manifest_nodes(root: dict) -> tuple[dict, ...]:
    output: list[dict] = []
    stack = [root]
    while stack:
        node = stack.pop()
        output.append(node)
        if node["feature"] >= 0:
            stack.extend((node["left"], node["right"]))
    return tuple(output)


def _matching_manifest_node(trees: list[dict], exemplar: dict) -> dict:
    for tree in trees:
        for node in _manifest_nodes(tree):
            if node == exemplar:
                return node
    raise AssertionError("tree node not found")


class CheckpointHardeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board = _synthetic_board()
        cls.model = train_resolution_first_g3r3(cls.board, config=_config())

    def save(self, root: Path, name: str = "g3r3.json") -> tuple[Path, str]:
        path = root / name
        digest = save_checkpoint_g3r3(
            path,
            self.model,
            root=root,
            metadata=METADATA,
            expected_model_sha256=self.model.sha256,
            expected_partition_sha256=self.model.partition_sha256,
            expected_dataset_sha256=self.model.training_dataset_sha256,
            dependencies=DEPENDENCIES,
        )
        return path, digest

    def load(self, root: Path, path: Path):
        return load_checkpoint_g3r3(
            path,
            root=root,
            expected_metadata=METADATA,
            expected_model_sha256=self.model.sha256,
            expected_partition_sha256=self.model.partition_sha256,
            expected_dataset_sha256=self.model.training_dataset_sha256,
            expected_dependencies=DEPENDENCIES,
        )

    def test_checkpoint_roundtrip_is_deterministic_and_fully_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, first_digest = self.save(root, "first.json")
            second, second_digest = self.save(root, "second.json")
            self.assertEqual(first_digest, second_digest)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            loaded, metadata = self.load(root, first)
            self.assertEqual(metadata, METADATA)
            self.assertEqual(loaded.manifest(), self.model.manifest())
            self.assertEqual(
                loaded.predict(self.board, oof=True),
                self.model.predict(self.board, oof=True),
            )

            expectations = (
                {"expected_metadata": {"stage": "foreign"}},
                {"expected_model_sha256": "0" * 64},
                {"expected_partition_sha256": "1" * 64},
                {"expected_dataset_sha256": "2" * 64},
                {"expected_dependencies": {"foreign": "3" * 64}},
            )
            baseline = {
                "root": root,
                "expected_metadata": METADATA,
                "expected_model_sha256": self.model.sha256,
                "expected_partition_sha256": self.model.partition_sha256,
                "expected_dataset_sha256": self.model.training_dataset_sha256,
                "expected_dependencies": DEPENDENCIES,
            }
            for change in expectations:
                with self.subTest(change=change):
                    arguments = {**baseline, **change}
                    with self.assertRaises(RuntimeError):
                        load_checkpoint_g3r3(first, **arguments)

                with self.assertRaises(RuntimeError):
                    load_checkpoint_g3r3(
                        first,
                        **{
                            **baseline,
                            "expected_model_sha256": self.model.sha256.upper(),
                        },
                    )

    def test_coherently_rehashed_integer_float_substitution_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.save(root)
            payload = json.loads(path.read_text())
            payload["model"]["config"]["blend_pair_weight"] = 0
            body = {
                key: payload[key]
                for key in payload
                if key != "checkpoint_sha256"
            }
            payload["checkpoint_sha256"] = g3r3._sha256(body)
            tampered = root / "tampered.json"
            tampered.write_bytes(g3r3._canonical_bytes(payload))
            with self.assertRaises(RuntimeError):
                self.load(root, tampered)

    def test_coherently_rehashed_float_integer_substitution_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.save(root)
            for field in ("feature_width", "preserved_block_width"):
                payload = json.loads(path.read_text())
                payload["model"][field] = float(payload["model"][field])
                body = {
                    key: payload[key]
                    for key in payload
                    if key != "checkpoint_sha256"
                }
                payload["checkpoint_sha256"] = g3r3._sha256(body)
                tampered = root / f"tampered-{field}.json"
                tampered.write_bytes(g3r3._canonical_bytes(payload))
                with self.subTest(field=field):
                    with self.assertRaises(RuntimeError):
                        self.load(root, tampered)

    def test_save_requires_exact_caller_model_partition_dataset_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field in (
                "expected_model_sha256",
                "expected_partition_sha256",
                "expected_dataset_sha256",
            ):
                arguments = {
                    "root": root,
                    "metadata": METADATA,
                    "expected_model_sha256": self.model.sha256,
                    "expected_partition_sha256": self.model.partition_sha256,
                    "expected_dataset_sha256": self.model.training_dataset_sha256,
                    "dependencies": DEPENDENCIES,
                }
                arguments[field] = "f" * 64
                with self.subTest(field=field):
                    with self.assertRaises(RuntimeError):
                        save_checkpoint_g3r3(
                            root / f"{field}.json", self.model, **arguments
                        )

    def test_load_rejects_live_path_replacement_and_true_aba(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.save(root)
            replacement = root.parent / f"{root.name}-replacement"
            replacement.write_bytes(path.read_bytes())
            original = g3r3._lstat_path
            calls = 0

            def replace_after_root_stat(target: Path):
                nonlocal calls
                info = original(target)
                if target == root:
                    calls += 1
                    if calls == 2:
                        os.replace(replacement, path)
                return info

            with mock.patch.object(
                g3r3, "_lstat_path", side_effect=replace_after_root_stat
            ):
                with self.assertRaisesRegex(RuntimeError, "changed|ABA"):
                    self.load(root, path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path, _ = self.save(root)
            away = root / "away"
            original = g3r3._lstat_path
            calls = 0

            def aba_after_root_stat(target: Path):
                nonlocal calls
                info = original(target)
                if target == root:
                    calls += 1
                    if calls == 2:
                        path.rename(away)
                        away.rename(path)
                return info

            with mock.patch.object(
                g3r3, "_lstat_path", side_effect=aba_after_root_stat
            ):
                with self.assertRaisesRegex(RuntimeError, "changed|ABA"):
                    self.load(root, path)

    def test_save_rejects_postpublication_replacement_and_aba(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "replace.json"
            replacement = root.parent / f"{root.name}-replacement"
            replacement.write_bytes(b"replacement")
            original = g3r3._lstat_path
            calls = 0

            def replace_after_root_stat(target: Path):
                nonlocal calls
                info = original(target)
                if target == root:
                    calls += 1
                    if calls == 2:
                        os.replace(replacement, path)
                return info

            with mock.patch.object(
                g3r3, "_lstat_path", side_effect=replace_after_root_stat
            ):
                with self.assertRaisesRegex(RuntimeError, "changed|ABA"):
                    self.save(root, "replace.json")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "aba.json"
            away = root / "away"
            original = g3r3._lstat_path
            calls = 0

            def aba_after_root_stat(target: Path):
                nonlocal calls
                info = original(target)
                if target == root:
                    calls += 1
                    if calls == 2:
                        path.rename(away)
                        away.rename(path)
                return info

            with mock.patch.object(
                g3r3, "_lstat_path", side_effect=aba_after_root_stat
            ):
                with self.assertRaisesRegex(RuntimeError, "changed|ABA"):
                    self.save(root, "aba.json")


if __name__ == "__main__":
    unittest.main()
