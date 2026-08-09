from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from irisu_pointer.resolution_first import FEATURE_NAMES
from irisu_pointer.resolution_first_g3 import (
    GEOMETRY_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    G3Config,
    G3Dataset,
    HistogramNewtonConfig,
    g3_outcomes,
    load_checkpoint_g3,
    save_checkpoint_g3,
    train_resolution_first_g3,
)


def _body(
    identifier: int,
    x: float,
    y: float,
    *,
    chain: int,
    color: int,
    lifetime: int = 100,
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "triangle",
        "lifecycle": "rotten" if color == 0 else "dynamic_fresh",
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
        "remaining_lifetime": lifetime,
        "rot_timer": 3,
    }


def _outcome(
    ordinal: int,
    *,
    pair_ordinal: int,
    source: int,
    destination: int,
    destination_chain: int,
    geometry: str,
    renewal_count: int,
    bind_ok: bool = True,
    censored: bool = False,
) -> dict[str, object]:
    weak = geometry == "analytic-weak"
    features = [0.1 + index / 100 for index in range(len(FEATURE_NAMES))]
    for name, value in {
        "gauge_fraction": 0.8,
        "source_x": 0.2,
        "source_y": 0.3,
        "source_vx": 0.0,
        "source_vy": 0.0,
        "source_size": 0.4,
        "source_rot_timer": 0.1,
        "destination_x": 0.6,
        "destination_y": 0.3,
        "destination_vx": 0.0,
        "destination_vy": 0.0,
        "destination_size": 0.4,
        "destination_rot_timer": 0.1,
        "pair_distance_sizes": 2.0,
        "impact_x_sizes": 0.5,
        "impact_y_sizes": 0.75,
    }.items():
        features[FEATURE_NAMES.index(name)] = value
    resolved = renewal_count == 2
    horizon = 100
    survival = horizon if resolved or censored else 20
    ticks = [10, 30][:renewal_count]
    action_x = {
        "analytic-strong": 0.8,
        "close-strong": 0.6,
        "analytic-weak": 0.2,
        "deep-strong": 0.1,
    }[geometry]
    pair_category = "rotten-hazard" if pair_ordinal == 0 else "fresh-match"
    intent = "match_rotten" if pair_ordinal == 0 else "steer_match"
    return {
        "candidate": {
            "ordinal": ordinal,
            "pair_ordinal": pair_ordinal,
            "geometry_ordinal": ordinal,
            "action": {
                "kind": 1 if weak else 2,
                "x_norm": action_x,
                "y_norm": 0.3,
            },
            "pair": {
                "category": pair_category,
                "intent": intent,
                "source_body_id": source,
                "destination_body_id": destination,
                "destination_chain_id": destination_chain,
                "distance_sizes": 2.0,
                "incumbent": ordinal == 0,
            },
            "geometry": {
                "name": geometry,
                "strength": "weak" if weak else "strong",
                "side_sizes": 0.5,
                "below_sizes": 0.75,
            },
        },
        "feature_names": list(FEATURE_NAMES),
        "feature_vector": features,
        "targets": {
            "b2_margin": 2.0 if resolved else None,
            "delta_b2": 0.0 if resolved and ordinal == 0 else None,
            "score_advantage": 1.0 if resolved else -1.0,
            "exact_unsafe": not resolved,
            "severe_unsafe": False,
        },
        "outcome": {
            "horizon_ticks": horizon,
            "survival_ticks": survival,
            "renewal_ticks": ticks,
            "renewals_resolved": resolved,
            "terminal": False,
            "gauge_failure": False,
            "new_terminal": False,
            "exact_unsafe": not resolved,
            "severe_unsafe": False,
        },
        "ledger": {
            "unresolved": [] if bind_ok else ["continuation-rebind-failed"],
            "continuation_rebind_failed": not bind_ok,
        },
    }


def _entry(
    seed: int,
    *,
    id_map: dict[int, int] | None = None,
    chain_map: dict[int, int] | None = None,
    tick: int = 101,
) -> dict[str, object]:
    ids = {1: 1, 2: 2, 3: 3, 4: 4} if id_map is None else id_map
    chains = {10: 10, 20: 20} if chain_map is None else chain_map
    bodies = [
        _body(ids[1], 120.0, 170.0, chain=chains[10], color=0, lifetime=50),
        _body(ids[2], 300.0, 172.0, chain=chains[10], color=0, lifetime=200),
        _body(ids[3], 460.0, 174.0, chain=chains[20], color=1, lifetime=80),
        _body(ids[4], 540.0, 176.0, chain=chains[20], color=1, lifetime=120),
    ]
    outcomes = [
        _outcome(
            0,
            pair_ordinal=0,
            source=ids[1],
            destination=ids[2],
            destination_chain=chains[10],
            geometry="analytic-strong",
            renewal_count=2,
        ),
        _outcome(
            1,
            pair_ordinal=0,
            source=ids[1],
            destination=ids[2],
            destination_chain=chains[10],
            geometry="close-strong",
            renewal_count=1,
            censored=True,
        ),
        _outcome(
            2,
            pair_ordinal=1,
            source=ids[3],
            destination=ids[4],
            destination_chain=chains[20],
            geometry="analytic-weak",
            renewal_count=0,
            bind_ok=False,
        ),
        _outcome(
            3,
            pair_ordinal=1,
            source=ids[3],
            destination=ids[4],
            destination_chain=chains[20],
            geometry="deep-strong",
            renewal_count=0,
        ),
    ]
    return {
        "exact_query": {
            "seed": seed,
            "query_id": f"q-{seed}",
            "outcomes": outcomes,
        },
        "pre_query_public_observation": {
            "tick": tick,
            "score": 50,
            "gauge": 800,
            "gauge_max": 1000,
            "level": 2,
            "highest_chain": 1,
            "qualifying_clear_count": 2,
            "difficulty": {"active_colors": 3, "spawn_interval_ticks": 50},
            "bodies": bodies,
        },
        "query_index": 1,
        "shot_index": 7,
        "tick": tick,
    }


def _dataset(seeds: range = range(1, 10)) -> G3Dataset:
    return G3Dataset(g3_outcomes(_entry(seed) for seed in seeds))


def _config() -> G3Config:
    return G3Config(
        folds=3,
        bags_per_fold=1,
        random_seed=17,
        histogram=HistogramNewtonConfig(
            bins=4,
            pair_steps=5,
            geometry_steps=8,
            learning_rate=0.2,
            l2=0.5,
            min_leaf=1,
        ),
    )


class OutcomeTests(unittest.TestCase):
    def test_labels_pair_partition_and_censoring_are_explicit(self) -> None:
        rows = g3_outcomes([_entry(1)])
        self.assertEqual([row.pair_partition for row in rows], [0, 0, 1, 1])
        self.assertEqual([row.renewal_count for row in rows], [2, 1, 0, 0])
        self.assertEqual([row.bind_ok for row in rows], [True, True, False, True])
        self.assertEqual([row.censored for row in rows], [False, True, False, False])
        self.assertEqual([row.deployable for row in rows], [True, False, False, False])
        self.assertEqual(rows[0].second_renewal_fraction, 0.3)

    def test_identity_chain_numbers_and_period_shift_are_not_inputs(self) -> None:
        original = G3Dataset(g3_outcomes([_entry(1)]))
        transformed = G3Dataset(
            g3_outcomes(
                [
                    _entry(
                        1,
                        id_map={1: 900, 2: 17, 3: 441, 4: 3},
                        chain_map={10: 8000, 20: 6},
                        tick=151,
                    )
                ]
            )
        )
        np.testing.assert_array_equal(original.pair_matrix, transformed.pair_matrix)
        np.testing.assert_array_equal(
            original.geometry_matrix, transformed.geometry_matrix
        )

    def test_declared_partition_is_validation_only(self) -> None:
        changed = _entry(1)
        for outcome in changed["exact_query"]["outcomes"]:
            outcome["candidate"]["pair_ordinal"] = (
                91 if outcome["candidate"]["pair_ordinal"] == 0 else 7
            )
        left = G3Dataset(g3_outcomes([_entry(1)]))
        right = G3Dataset(g3_outcomes([changed]))
        np.testing.assert_array_equal(left.pair_matrix, right.pair_matrix)
        np.testing.assert_array_equal(left.geometry_matrix, right.geometry_matrix)

    def test_raw_lifetime_replaces_frozen_normalized_lifetime_proxy(self) -> None:
        changed = _entry(1)
        for outcome in changed["exact_query"]["outcomes"]:
            vector = outcome["feature_vector"]
            vector[FEATURE_NAMES.index("source_remaining_lifetime")] = 999999.0
            vector[FEATURE_NAMES.index("destination_remaining_lifetime")] = -999999.0
        left = G3Dataset(g3_outcomes([_entry(1)]))
        right = G3Dataset(g3_outcomes([changed]))
        np.testing.assert_array_equal(left.pair_matrix, right.pair_matrix)
        source_fraction = PAIR_FEATURE_NAMES.index(
            "source_lifetime_horizon_fraction"
        )
        source_overrun = PAIR_FEATURE_NAMES.index("source_lifetime_log_overrun")
        self.assertEqual(left.pair_matrix[0, source_fraction], 0.5)
        self.assertEqual(left.pair_matrix[0, source_overrun], 0.0)
        destination_fraction = PAIR_FEATURE_NAMES.index(
            "destination_lifetime_horizon_fraction"
        )
        self.assertEqual(left.pair_matrix[0, destination_fraction], 1.0)

    def test_lifetime_sentinel_is_distinct_from_finite_over_horizon(self) -> None:
        sentinel = _entry(1)
        finite = _entry(1)
        sentinel["pre_query_public_observation"]["bodies"][0][
            "remaining_lifetime"
        ] = 99999
        finite["pre_query_public_observation"]["bodies"][0][
            "remaining_lifetime"
        ] = 100000
        sentinel_row = G3Dataset(g3_outcomes([sentinel])).pair_matrix[0]
        finite_row = G3Dataset(g3_outcomes([finite])).pair_matrix[0]
        column = PAIR_FEATURE_NAMES.index("source_lifetime_sentinel")
        self.assertEqual(sentinel_row[column], 1.0)
        self.assertEqual(finite_row[column], 0.0)

    def test_malformed_intent_and_renewal_evidence_fail_closed(self) -> None:
        bad_intent = _entry(1)
        bad_intent["exact_query"]["outcomes"][0]["candidate"]["pair"][
            "intent"
        ] = "foreign"
        with self.assertRaisesRegex(ValueError, "intent/category"):
            g3_outcomes([bad_intent])

        bad_renewal = _entry(1)
        bad_renewal["exact_query"]["outcomes"][0]["outcome"][
            "renewals_resolved"
        ] = False
        with self.assertRaisesRegex(ValueError, "renewals_resolved"):
            g3_outcomes([bad_renewal])

    def test_model_feature_names_are_unique_and_contain_no_identity(self) -> None:
        self.assertEqual(len(PAIR_FEATURE_NAMES), len(set(PAIR_FEATURE_NAMES)))
        self.assertEqual(
            len(GEOMETRY_FEATURE_NAMES), len(set(GEOMETRY_FEATURE_NAMES))
        )
        names = (*PAIR_FEATURE_NAMES, *GEOMETRY_FEATURE_NAMES)
        self.assertFalse(any("body_id" in name or "chain_id" in name for name in names))
        self.assertNotIn("pair_partition", names)


class LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = _dataset()
        cls.model = train_resolution_first_g3(cls.dataset, config=_config())

    def test_cross_fit_is_whole_seed_and_oof_is_complete(self) -> None:
        assigned = self.model.seed_folds
        self.assertEqual(set(assigned), set(self.dataset.seeds))
        for seed in self.dataset.seeds:
            fold = self.model.folds[assigned[seed]]
            self.assertIn(seed, fold.heldout_seeds)
        predictions = self.model.predict(self.dataset, oof=True)
        self.assertEqual(len(predictions), len(self.dataset.outcomes))
        self.assertTrue(
            all(0.0 <= prediction.primary_score <= 1.0 for prediction in predictions)
        )

    def test_training_and_predictions_are_deterministic(self) -> None:
        repeated = train_resolution_first_g3(self.dataset, config=_config())
        self.assertEqual(
            self.model.predict(self.dataset, oof=True),
            repeated.predict(self.dataset, oof=True),
        )
        self.assertEqual(self.model.manifest(), repeated.manifest())

    def test_direct_head_is_primary_and_heads_are_independent(self) -> None:
        manifest = self.model.manifest()
        self.assertEqual(
            manifest["primary_score"], "direct-pair-plus-geometry-residual"
        )
        self.assertFalse(manifest["downstream_multitask_gradients"])
        bag = self.model.folds[0].bags[0]
        self.assertIsNot(bag.deployability.pair, bag.bind.pair)
        counts = dict(
            (name, (negative, positive))
            for name, negative, positive in bag.head_counts
        )
        self.assertGreater(counts["deployability"][0], 0)
        self.assertGreater(counts["deployability"][1], 0)
        self.assertGreater(counts["second_renewal"][0], 0)
        self.assertGreater(counts["second_renewal"][1], 0)

    def test_direct_geometry_residual_separates_synthetic_actions(self) -> None:
        predictions = self.model.predict(self.dataset, oof=True)
        positives = [
            row.deployability_mean for row in predictions if row.ordinal == 0
        ]
        negatives = [
            row.deployability_mean for row in predictions if row.ordinal in {2, 3}
        ]
        self.assertGreater(float(np.mean(positives)), float(np.mean(negatives)))
        self.assertTrue(
            any(
                abs(row.deployability_mean - row.causal_product_mean) > 1e-6
                for row in predictions
            )
        )

    def test_oof_rejects_unknown_seed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown seed"):
            self.model.predict(G3Dataset(g3_outcomes([_entry(999)])), oof=True)

    def test_checkpoint_roundtrip_and_tamper_rejection(self) -> None:
        before = self.model.predict(self.dataset, oof=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "g3.json"
            digest = save_checkpoint_g3(path, self.model, metadata={"stage": "unit"})
            loaded, metadata = load_checkpoint_g3(path)
            self.assertEqual(len(digest), 64)
            self.assertEqual(metadata, {"stage": "unit"})
            self.assertEqual(loaded.sha256, self.model.sha256)
            self.assertEqual(loaded.predict(self.dataset, oof=True), before)

            leaf_tamper = json.loads(path.read_text())
            pair = leaf_tamper["model"]["folds"][0]["bags"][0]["heads"][
                "deployability"
            ]["pair"]
            self.assertTrue(pair["stumps"])
            pair["stumps"][0]["left"] += 0.01
            leaf_path = root / "leaf-tamper.json"
            leaf_path.write_text(
                json.dumps(leaf_tamper, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaisesRegex(RuntimeError, "model identity"):
                load_checkpoint_g3(leaf_path)

            feature_tamper = json.loads(path.read_text())
            feature_tamper["model"]["pair_feature_names"][0] = "foreign"
            feature_path = root / "feature-tamper.json"
            feature_path.write_text(
                json.dumps(feature_tamper, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaisesRegex(RuntimeError, "feature identity"):
                load_checkpoint_g3(feature_path)


if __name__ == "__main__":
    unittest.main()
