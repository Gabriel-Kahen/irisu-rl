from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from unittest import mock
from dataclasses import replace
from pathlib import Path

import numpy as np

from irisu_pointer.resolution_first import FEATURE_NAMES
from irisu_pointer.resolution_first_g2 import (
    BoardBranchRecord,
    board_branch_records,
)
from irisu_pointer.resolution_first_g3r3 import (
    HistogramNewtonBoostR3,
    WIDE_FEATURE_WIDTH,
    WideBoard,
)
from irisu_pointer.resolution_proposal_g4 import (
    G4Board,
    G4CandidateIdentity,
    G4Config,
    G4ExactOutcome,
    G4InferenceBoard,
    G4Prediction,
    G4PredictionBatch,
    G4QueryInventory,
    INFERENCE_SCHEMA,
    ResolutionProposalG4,
    absolute_safe,
    exact_certificate_g4,
    exact_outcome_g4,
    g4_board_from_records,
    g4_inference_board_from_entries,
    growth_target,
    level_from_record,
    load_checkpoint_g4,
    one_rot_liability,
    propose_g4,
    save_checkpoint_g4,
    select_exact_g4,
    solvency_target,
    train_resolution_proposal_g4,
)
from irisu_rl.encoding import TeacherStateEncoder


DEPENDENCIES = {
    "g4-source": hashlib.sha256(b"g4-source").hexdigest(),
    "g4-tests": hashlib.sha256(b"g4-tests").hexdigest(),
    "g3r3-source": (
        "0ee4f907a531f6f790dc342a9668edfb06c161f7630a6c452981f599e9a66fd1"
    ),
    "g3r3-tests": (
        "e527bfd63b3e6cb9b50320baae714c08c78219a505750accd2414eff29948dd5"
    ),
    "g3r2-features": hashlib.sha256(b"g3r2-features").hexdigest(),
    "g2-board": hashlib.sha256(b"g2-board").hexdigest(),
    "g4-protocol": (
        "7fa5974e6aa647462a30809028a8848f715698bcbea3c7bc1bdf3629b40c5e4d"
    ),
}
METADATA = {"stage": "unit", "purpose": "dual-head-roundtrip"}


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(seed: int, query: str, ordinal: int) -> G4CandidateIdentity:
    return G4CandidateIdentity(
        seed,
        query,
        ordinal,
        _digest(f"candidate:{seed}:{query}:{ordinal}"),
        _digest(f"action:{seed}:{query}:{ordinal}"),
    )


def _outcome(
    identity: G4CandidateIdentity,
    *,
    score: float,
    b2: float | None = 2000.0,
    safe: bool = True,
    unsafe: bool = False,
    level: int = 1,
    terminal: bool = True,
    observation_sha256: str | None = None,
) -> G4ExactOutcome:
    return G4ExactOutcome(
        identity,
        terminal,
        level,
        safe,
        safe,
        unsafe,
        False,
        b2 if safe else None,
        (0.0 if identity.ordinal == 0 else 1.0) if safe else None,
        score,
        _digest(f"outcome:{identity.candidate_id}"),
        (
            _digest(f"observation:{identity.seed}:{identity.query_id}")
            if observation_sha256 is None
            else observation_sha256
        ),
    )


def _batch(
    predictions: tuple[G4Prediction, ...],
    *,
    level: int = 1,
) -> G4PredictionBatch:
    identities = tuple(row.identity for row in predictions)
    query_key = identities[0].query_key
    inventory = G4QueryInventory(
        query_key,
        level,
        one_rot_liability(level),
        _digest(f"observation:{query_key[0]}:{query_key[1]}"),
        _identity(query_key[0], query_key[1], 0),
        identities,
    )
    return G4PredictionBatch(
        inventory,
        _digest("unit-model"),
        _digest("unit-board"),
        predictions,
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _manifest_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _config() -> G4Config:
    return G4Config(
        folds=8,
        partition_id="unit-explicit-eightfold-v1",
        seed_partition=tuple((index, index + 8) for index in range(1, 9)),
    )


def _synthetic_board() -> G4Board:
    features: list[np.ndarray] = []
    safe_labels: list[bool] = []
    levels: list[int] = []
    observations: list[str] = []
    solvency: list[float] = []
    growth: list[float] = []
    seeds: list[int] = []
    queries: list[str] = []
    sources: list[int] = []
    destinations: list[int] = []
    ordinals: list[int] = []
    identities: list[G4CandidateIdentity] = []
    incumbents: list[G4CandidateIdentity] = []
    for seed in range(1, 17):
        for query_index, count in enumerate((2 + seed % 3, 3 + seed % 2)):
            query = f"q-{seed}-{query_index}"
            incumbents.append(_identity(seed, query, 0))
            for ordinal in range(1, count + 1):
                viable = (seed + ordinal + query_index) % 3 != 0
                row = np.zeros(WIDE_FEATURE_WIDTH, dtype=np.float64)
                row[0] = float(viable)
                row[1] = ordinal / 8.0
                row[126] = (seed % 5) / 5.0
                row[127] = math.sin(seed + ordinal)
                row[500] = math.cos(seed * ordinal)
                features.append(row)
                b2 = 1900.0 + 10.0 * ordinal
                safe_labels.append(viable)
                levels.append(1)
                observations.append(_digest(f"observation:{seed}:{query}"))
                solvency.append(
                    0.5 + 0.5 * math.tanh(b2 / one_rot_liability(1))
                    if viable
                    else 0.0
                )
                growth.append(
                    math.tanh((8.0 + 7.0 * ordinal) / 64.0)
                    if viable and ordinal % 2
                    else 0.0
                )
                seeds.append(seed)
                queries.append(query)
                sources.append(ordinal % 3)
                destinations.append(3 + ordinal % 3)
                ordinals.append(ordinal)
                identities.append(_identity(seed, query, ordinal))
    wide = WideBoard(
        np.asarray(features),
        np.asarray(safe_labels),
        np.asarray(seeds),
        np.asarray(queries, dtype=object),
        np.asarray(sources),
        np.asarray(destinations),
        np.asarray(ordinals),
    )
    return G4Board(
        wide,
        tuple(identities),
        tuple(incumbents),
        np.asarray(levels, dtype=np.int64),
        tuple(observations),
        np.asarray(solvency, dtype=np.float64),
        np.asarray(growth, dtype=np.float64),
    )


def _inference_board(board: G4Board) -> G4InferenceBoard:
    return G4InferenceBoard(
        np.array(board.features, dtype=np.float64, copy=True),
        board.identities,
        board.incumbent_identities,
        np.array(board.levels, dtype=np.int64, copy=True),
        board.observation_sha256,
    )


_SHARED_BOARD: G4Board | None = None
_SHARED_MODEL: ResolutionProposalG4 | None = None


def _shared_board_model() -> tuple[G4Board, ResolutionProposalG4]:
    global _SHARED_BOARD, _SHARED_MODEL
    if _SHARED_BOARD is None:
        _SHARED_BOARD = g4_board_from_records(_training_records())
    if _SHARED_MODEL is None:
        _SHARED_MODEL = train_resolution_proposal_g4(
            _training_records(),
            config=_config(),
            dependencies=DEPENDENCIES,
        )
    return _SHARED_BOARD, _SHARED_MODEL


def _record(
    seed: int,
    ordinal: int,
    *,
    resolved: bool,
    unsafe: bool,
    b2: float | None,
    delta: float | None,
    score: float,
    level: int = 1,
) -> BoardBranchRecord:
    candidate = [0.0] * len(FEATURE_NAMES)
    candidate[0] = ordinal / 10.0
    candidate[24] = 0.25 + ordinal / 100.0
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
        bodies.append(tuple(row))
    global_features = [0.0] * len(TeacherStateEncoder.schema.global_features)
    global_features[
        TeacherStateEncoder.schema.global_features.index("level_log1p")
    ] = float(np.float32(math.log1p(level) / 8.0))
    return BoardBranchRecord(
        seed=seed,
        query_id=f"q-{seed}",
        ordinal=ordinal,
        candidate_id=_digest(f"candidate-{seed}-{ordinal}"),
        action_id=_digest(f"action-{seed}-{ordinal}"),
        signature="fresh-match|analytic-strong",
        features=tuple(candidate),
        candidate_resolved=resolved,
        finite_pair=resolved and delta is not None,
        exact_unsafe=unsafe,
        severe_unsafe=False,
        b2=b2,
        delta_b2=delta,
        score_advantage=score,
        source_sha256=_digest(f"source-{seed}-{ordinal}"),
        global_features=tuple(global_features),
        phase_features=(0.0, 0.5, 0.25),
        body_features=tuple(bodies),
        body_chain_groups=(0, 0, 1, 1),
        body_grouped_flags=(True, True, True, True),
        body_color_groups=(0, 1, 2, 3),
        source_index=0 if ordinal < 2 else 2,
        destination_index=1 if ordinal < 2 else 3,
        incumbent_source_index=0,
        incumbent_destination_index=1,
        observation_sha256=_digest(f"observation-{seed}"),
    )


def _records(seed: int = 1) -> tuple[BoardBranchRecord, ...]:
    return (
        _record(
            seed,
            0,
            resolved=True,
            unsafe=False,
            b2=1000.0,
            delta=0.0,
            score=0.0,
        ),
        _record(
            seed,
            1,
            resolved=True,
            unsafe=False,
            b2=2000.0,
            delta=1000.0,
            score=64.0,
        ),
        _record(
            seed,
            2,
            resolved=False,
            unsafe=True,
            b2=None,
            delta=None,
            score=-5.0,
        ),
    )


def _training_records() -> tuple[BoardBranchRecord, ...]:
    return tuple(
        record
        for seed in range(1, 17)
        for record in _records(seed)
    )


def _opened_entries(
    split: str, expected_seed_count: int
) -> tuple[dict[str, object], ...]:
    root = (
        Path(__file__).resolve().parents[1]
        / "artifacts/r3/development"
        / f"r3h-resolution-first-g2r1-20260730/collection/{split}"
    )
    paths = tuple(sorted(root.glob("*.queries.jsonl")))
    if len(paths) != expected_seed_count:
        raise RuntimeError(f"opened {split} inventory is unavailable")
    output: list[dict[str, object]] = []
    for path in paths:
        for line in path.read_text().splitlines():
            value = json.loads(line)
            if type(value) is not dict:
                raise RuntimeError("opened train-board query is malformed")
            output.append(value)
    return tuple(output)


def _opened_train_entries() -> tuple[dict[str, object], ...]:
    return _opened_entries("train-board", 32)


def _opened_support_entries() -> tuple[dict[str, object], ...]:
    return _opened_entries("support-board", 24)


def _public_projection(entry: dict[str, object]) -> dict[str, object]:
    query = entry["exact_query"]
    if type(query) is not dict or type(query.get("outcomes")) is not list:
        raise RuntimeError("opened exact query is malformed")
    outcomes = query["outcomes"]
    return {
        "schema": INFERENCE_SCHEMA,
        "seed": query["seed"],
        "query_id": query["query_id"],
        "query_index": entry["query_index"],
        "shot_index": entry["shot_index"],
        "tick": entry["tick"],
        "pre_query_public_observation": entry[
            "pre_query_public_observation"
        ],
        "pre_query_public_observation_sha256": entry[
            "pre_query_public_observation_sha256"
        ],
        "candidates": [
            outcome["candidate"]
            for outcome in outcomes
            if type(outcome) is dict
        ],
    }


class SupervisionAndFeatureBoundaryTests(unittest.TestCase):
    def test_exact_dual_regime_target_formulas(self) -> None:
        incumbent, safe, unsafe = _records()
        reserve = one_rot_liability(1)
        self.assertEqual(level_from_record(safe), 1)
        self.assertEqual(reserve, 1820)
        self.assertEqual(one_rot_liability(0), 1800)
        self.assertEqual(one_rot_liability(99), 3780)
        self.assertEqual(one_rot_liability(100), 3780)
        self.assertTrue(absolute_safe(incumbent))
        self.assertTrue(absolute_safe(safe))
        self.assertFalse(absolute_safe(unsafe))
        self.assertAlmostEqual(
            solvency_target(safe),
            0.5 + 0.5 * math.tanh(2000.0 / reserve),
        )
        self.assertAlmostEqual(growth_target(safe), math.tanh(1.0))
        self.assertEqual(growth_target(incumbent), 0.0)
        self.assertEqual(solvency_target(unsafe), 0.0)
        self.assertEqual(growth_target(unsafe), 0.0)
        for changed in (
            replace(safe, candidate_resolved=False),
            replace(safe, exact_unsafe=True),
            replace(safe, severe_unsafe=True),
            replace(safe, b2=-1.0),
        ):
            with self.subTest(changed=changed):
                self.assertFalse(absolute_safe(changed))
                self.assertEqual(solvency_target(changed), 0.0)
                self.assertEqual(growth_target(changed), 0.0)
        for rescued in (
            replace(safe, finite_pair=False),
            replace(safe, finite_pair=False, delta_b2=None),
        ):
            with self.subTest(rescued=rescued):
                self.assertTrue(absolute_safe(rescued))
                self.assertGreater(solvency_target(rescued), 0.5)
        rescue_board = g4_board_from_records(
            (
                incumbent,
                replace(safe, finite_pair=False, delta_b2=None),
                unsafe,
            )
        )
        self.assertTrue(rescue_board.wide.labels[0])
        self.assertGreater(rescue_board.solvency[0], 0.5)
        self.assertEqual(rescue_board.growth[0], math.tanh(1.0))
        self.assertEqual(growth_target(replace(safe, b2=1819.0)), 0.0)
        zero = replace(safe, b2=0.0)
        self.assertTrue(absolute_safe(zero))
        self.assertEqual(solvency_target(zero), 0.5)
        self.assertEqual(growth_target(zero), 0.0)
        at_reserve = replace(safe, b2=float(reserve))
        self.assertGreater(growth_target(at_reserve), 0.0)
        self.assertEqual(
            growth_target(replace(safe, score_advantage=-9.0)), 0.0
        )
        with self.assertRaises(ValueError):
            one_rot_liability(True)

    def test_labels_identities_and_certificate_fields_never_enter_features(self) -> None:
        original = _records()
        changed = (
            original[0],
            replace(
                original[1],
                candidate_id=_digest("foreign-candidate"),
                action_id=_digest("foreign-action"),
                source_sha256=_digest("foreign-source"),
                candidate_resolved=False,
                finite_pair=False,
                exact_unsafe=True,
                severe_unsafe=True,
                b2=None,
                delta_b2=None,
                score_advantage=-999.0,
            ),
            replace(
                original[2],
                candidate_id=_digest("other-candidate"),
                action_id=_digest("other-action"),
                source_sha256=_digest("other-source"),
                candidate_resolved=True,
                finite_pair=True,
                exact_unsafe=False,
                severe_unsafe=False,
                b2=50.0,
                delta_b2=3.0,
                score_advantage=128.0,
            ),
        )
        left = g4_board_from_records(original)
        right = g4_board_from_records(changed)
        np.testing.assert_array_equal(left.features, right.features)
        self.assertNotEqual(left.identities, right.identities)
        self.assertFalse(np.array_equal(left.wide.labels, right.wide.labels))
        self.assertFalse(np.array_equal(left.solvency, right.solvency))
        self.assertFalse(np.array_equal(left.growth, right.growth))

    def test_variable_query_sizes_are_supported_but_incomplete_ordinals_reject(self) -> None:
        board = _synthetic_board()
        sizes: dict[tuple[int, str], int] = {}
        for identity in board.identities:
            sizes[identity.query_key] = sizes.get(identity.query_key, 0) + 1
        self.assertGreater(len(set(sizes.values())), 1)
        mask = np.arange(len(board.identities)) != 1
        with self.assertRaisesRegex(ValueError, "closed and bound"):
            G4Board(
                board.wide.take(mask),
                tuple(
                    identity
                    for index, identity in enumerate(board.identities)
                    if index != 1
                ),
                board.incumbent_identities,
                board.levels[mask],
                tuple(
                    value
                    for index, value in enumerate(board.observation_sha256)
                    if index != 1
                ),
                board.solvency[mask],
                board.growth[mask],
            )


class PublicInferenceBoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.opened = _opened_train_entries()
        cls.public = tuple(_public_projection(entry) for entry in cls.opened)
        cls.opened_support = _opened_support_entries()
        cls.public_support = tuple(
            _public_projection(entry) for entry in cls.opened_support
        )

    def test_full_opened_splits_are_bit_identical_before_and_after_exact(
        self,
    ) -> None:
        cases = (
            ("train-32", self.opened, self.public, 128, 1777),
            (
                "support-24",
                self.opened_support,
                self.public_support,
                96,
                1329,
            ),
        )
        for name, opened, public, queries, rows in cases:
            with self.subTest(split=name):
                inference = g4_inference_board_from_entries(public)
                labeled = g4_board_from_records(board_branch_records(opened))
                self.assertEqual(len(public), queries)
                self.assertEqual(
                    inference.features.shape,
                    (rows, WIDE_FEATURE_WIDTH),
                )
                np.testing.assert_array_equal(
                    inference.features, labeled.features
                )
                self.assertEqual(inference.identities, labeled.identities)
                self.assertEqual(
                    inference.incumbent_identities,
                    labeled.incumbent_identities,
                )
                np.testing.assert_array_equal(
                    inference.levels, labeled.levels
                )
                self.assertEqual(
                    inference.observation_sha256,
                    labeled.observation_sha256,
                )
                self.assertEqual(
                    inference.feature_inventory_sha256,
                    labeled.feature_inventory_sha256,
                )

    def test_predictions_are_label_invariant_on_public_features(self) -> None:
        public = self.public[0]
        original = self.opened[0]
        changed = copy.deepcopy(original)
        query = changed["exact_query"]
        assert type(query) is dict and type(query["outcomes"]) is list
        alternative = query["outcomes"][1]
        assert type(alternative) is dict
        alternative["targets"] = {
            "b2_margin": None,
            "delta_b2": None,
            "score_advantage": -999.0,
            "exact_unsafe": True,
            "severe_unsafe": True,
        }
        alternative["outcome"] = {
            "renewals_resolved": False,
            "exact_unsafe": True,
            "severe_unsafe": True,
        }
        alternative["ledger"] = {
            "unresolved": ["changed-label-only"],
            "continuation_rebind_failed": False,
        }
        inference = g4_inference_board_from_entries((public,))
        original_board = g4_board_from_records(
            board_branch_records((original,))
        )
        changed_board = g4_board_from_records(board_branch_records((changed,)))
        np.testing.assert_array_equal(
            inference.features, original_board.features
        )
        np.testing.assert_array_equal(
            inference.features, changed_board.features
        )
        self.assertEqual(
            inference.feature_inventory_sha256,
            original_board.feature_inventory_sha256,
        )
        self.assertEqual(
            inference.feature_inventory_sha256,
            changed_board.feature_inventory_sha256,
        )
        self.assertFalse(
            np.array_equal(original_board.solvency, changed_board.solvency)
        )
        _board, model = _shared_board_model()
        self.assertEqual(
            model.predict(inference),
            model.predict(_inference_board(original_board)),
        )
        self.assertEqual(
            model.predict(inference),
            model.predict(_inference_board(changed_board)),
        )

    def test_forbidden_exact_fields_and_malformed_public_bindings_reject(
        self,
    ) -> None:
        baseline = self.public[0]
        mutations: list[tuple[str, dict[str, object]]] = []
        for field in ("targets", "outcome", "ledger", "certificate"):
            changed = copy.deepcopy(baseline)
            changed[field] = {}
            mutations.append((f"top-{field}", changed))
        changed = copy.deepcopy(baseline)
        observation = changed["pre_query_public_observation"]
        assert type(observation) is dict
        observation["survival_ticks"] = 768
        changed["pre_query_public_observation_sha256"] = hashlib.sha256(
            _canonical_bytes(observation)
        ).hexdigest()
        mutations.append(("observation-exact-field", changed))
        changed = copy.deepcopy(baseline)
        candidates = changed["candidates"]
        assert type(candidates) is list and type(candidates[1]) is dict
        candidates[1]["targets"] = {}
        mutations.append(("candidate-target", changed))
        changed = copy.deepcopy(baseline)
        changed["pre_query_public_observation_sha256"] = "0" * 64
        mutations.append(("observation-hash", changed))
        changed = copy.deepcopy(baseline)
        changed["seed"] = np.int64(changed["seed"])
        mutations.append(("numpy-seed", changed))
        changed = copy.deepcopy(baseline)
        candidates = changed["candidates"]
        assert type(candidates) is list
        candidates[1], candidates[2] = candidates[2], candidates[1]
        mutations.append(("candidate-order", changed))
        changed = copy.deepcopy(baseline)
        candidates = changed["candidates"]
        assert type(candidates) is list and type(candidates[1]) is dict
        action = candidates[1]["action"]
        assert type(action) is dict
        action["x_norm"] = float(action["x_norm"]) + 1e-4
        mutations.append(("action-lowering", changed))
        changed = copy.deepcopy(baseline)
        candidates = changed["candidates"]
        assert type(candidates) is list and type(candidates[1]) is dict
        pair = candidates[1]["pair"]
        assert type(pair) is dict
        pair["distance_sizes"] = float(pair["distance_sizes"]) + 1e-4
        mutations.append(("pair-distance", changed))
        changed = copy.deepcopy(baseline)
        candidates = changed["candidates"]
        assert type(candidates) is list and type(candidates[1]) is dict
        pair = candidates[1]["pair"]
        assert type(pair) is dict
        pair["source_body_id"] = 2**31
        mutations.append(("foreign-endpoint", changed))
        for name, value in mutations:
            with self.subTest(name=name), self.assertRaises(ValueError):
                g4_inference_board_from_entries((value,))

    def test_variable_preexact_candidate_counts_remain_closed(self) -> None:
        baseline = self.public[0]
        for count in (2, 3, 5, 6):
            changed = copy.deepcopy(baseline)
            candidates = changed["candidates"]
            assert type(candidates) is list
            changed["candidates"] = candidates[:count]
            with self.subTest(count=count):
                board = g4_inference_board_from_entries((changed,))
                self.assertEqual(len(board.identities), count - 1)

    def test_oof_requires_exact_training_feature_inventory(self) -> None:
        board, model = _shared_board_model()
        inference = G4InferenceBoard(
            np.array(board.features, dtype=np.float64, copy=True),
            board.identities,
            board.incumbent_identities,
            np.array(board.levels, dtype=np.int64, copy=True),
            board.observation_sha256,
        )
        self.assertEqual(
            model.predict(inference, oof=True),
            model.predict(_inference_board(board), oof=True),
        )
        changed_features = np.array(
            inference.features, dtype=np.float64, copy=True
        )
        changed_features[0, 0] += 1e-6
        changed = G4InferenceBoard(
            changed_features,
            inference.identities,
            inference.incumbent_identities,
            np.array(inference.levels, dtype=np.int64, copy=True),
            inference.observation_sha256,
        )
        with self.assertRaisesRegex(ValueError, "feature inventory"):
            model.predict(changed, oof=True)


class ExactTypesTrainingAndProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board, cls.model = _shared_board_model()
        cls.config = _config()

    def test_strict_types_reject_bool_numpy_and_manifest_coercion(self) -> None:
        with self.assertRaisesRegex(ValueError, "inexact types"):
            train_resolution_proposal_g4(
                self.board,
                config=self.config,
                dependencies=DEPENDENCIES,
            )
        with self.assertRaisesRegex(ValueError, "prediction request"):
            self.model.predict(self.board)
        with self.assertRaisesRegex(ValueError, "prediction request"):
            self.model.predict_batches(self.board)
        with self.assertRaises(ValueError):
            G4CandidateIdentity(
                True, "q", 1, _digest("candidate"), _digest("action")
            )
        with self.assertRaises(ValueError):
            G4CandidateIdentity(
                1, "q", np.int64(1), _digest("candidate"), _digest("action")
            )
        with self.assertRaises(ValueError):
            G4Prediction(_identity(1, "q", 1), np.float64(0.5), 0.0, 0.5, 0.0)
        prediction = G4Prediction(
            _identity(1, "q", 1), 0.5, 0.0, 0.5, 0.0
        )
        with self.assertRaises(ValueError):
            propose_g4(
                _batch((prediction,)),
                _outcome(_identity(1, "q", 0), score=0.0, b2=1000.0),
                budget=True,
            )
        manifest = self.config.manifest()
        manifest["solvency"]["learning_rate"] = 1
        with self.assertRaises(RuntimeError):
            G4Config.from_manifest(manifest)
        with self.assertRaisesRegex(ValueError, "configuration"):
            G4Config(
                folds=self.config.folds,
                partition_id=self.config.partition_id,
                seed_partition=self.config.seed_partition,
                solvency=replace(self.config.solvency, rounds=299),
                growth=self.config.growth,
            )
        manifest = self.config.manifest()
        manifest["growth"]["rounds"] = 299
        with self.assertRaisesRegex(RuntimeError, "configuration"):
            G4Config.from_manifest(manifest)
        identity_manifest = _identity(1, "q", 1).manifest()
        identity_manifest["query_id"] = 7
        with self.assertRaises(RuntimeError):
            G4CandidateIdentity.from_manifest(identity_manifest)
        config_manifest = self.config.manifest()
        config_manifest["partition_id"] = 7
        with self.assertRaises(RuntimeError):
            G4Config.from_manifest(config_manifest)
        model_manifest = self.model.manifest()
        model_manifest["feature_width"] = float(WIDE_FEATURE_WIDTH)
        with self.assertRaises(RuntimeError):
            ResolutionProposalG4.from_manifest(model_manifest)
        model_manifest = self.model.manifest()
        model_manifest["dependency_identities"][0][0] = 7
        with self.assertRaises(RuntimeError):
            ResolutionProposalG4.from_manifest(model_manifest)
        with self.assertRaises(ValueError):
            _outcome(
                _identity(1, "q", 1),
                score=1.0,
                terminal=False,
            )
        with self.assertRaises(ValueError):
            G4Board(
                self.board.wide,
                self.board.identities,
                self.board.incumbent_identities,
                self.board.levels.astype(np.float64),
                self.board.observation_sha256,
                self.board.solvency,
                self.board.growth,
            )

    def test_whole_seed_oof_is_exact_complete_and_train_only(self) -> None:
        self.assertEqual(
            tuple(fold.heldout_seeds for fold in self.model.folds),
            self.config.seed_partition,
        )
        self.assertEqual(set(self.model.seed_folds), set(self.board.unique_seeds))
        inference = _inference_board(self.board)
        predictions = self.model.predict(inference, oof=True)
        self.assertEqual(len(predictions), len(self.board.identities))
        self.assertTrue(
            all(
                row.solvency_std == 0.0 and row.growth_std == 0.0
                for row in predictions
            )
        )
        batches = self.model.predict_batches(inference, oof=True)
        self.assertEqual(len(batches), len(self.board.query_keys))
        self.assertTrue(
            all(
                batch.model_sha256 == self.model.sha256
                and batch.feature_inventory_sha256
                == self.board.feature_inventory_sha256
                and batch.inventory.sha256
                == self.board.inventory(batch.inventory.query_key).sha256
                for batch in batches
            )
        )
        heldout = set(self.model.folds[0].heldout_seeds)
        mask = np.asarray([int(seed) not in heldout for seed in self.board.seeds])
        expected_solvency = HistogramNewtonBoostR3.fit(
            self.board.features[mask],
            self.board.solvency[mask],
            self.board.seeds[mask],
            self.config.solvency,
        )
        expected_growth = HistogramNewtonBoostR3.fit(
            self.board.features[mask],
            self.board.growth[mask],
            self.board.seeds[mask],
            self.config.growth,
        )
        self.assertEqual(
            self.model.folds[0].solvency.manifest(),
            expected_solvency.manifest(),
        )
        self.assertEqual(
            self.model.folds[0].growth.manifest(),
            expected_growth.manifest(),
        )

    def test_prediction_batches_hash_feature_inventory_once(self) -> None:
        inference = _inference_board(self.board)
        feature_digest = inference.feature_inventory_sha256
        model_digest = self.model.sha256
        with mock.patch.object(
            G4InferenceBoard,
            "feature_inventory_sha256",
            new_callable=mock.PropertyMock,
            return_value=feature_digest,
        ) as feature_hash, mock.patch.object(
            ResolutionProposalG4,
            "sha256",
            new_callable=mock.PropertyMock,
            return_value=model_digest,
        ) as model_hash:
            batches = self.model.predict_batches(inference)
        self.assertEqual(len(batches), len(self.board.query_keys))
        self.assertEqual(feature_hash.call_count, 1)
        self.assertEqual(model_hash.call_count, 1)

    def test_diversity_union_has_fixed_quotas_dedupe_fill_and_ties(self) -> None:
        identities = tuple(_identity(9, "query", ordinal) for ordinal in range(1, 6))
        solvency = (0.80, 0.99, 0.98, 0.97, 0.10)
        growth = (0.99, 0.98, 0.40, 0.30, 0.20)
        predictions = tuple(
            G4Prediction(identity, solvent, 0.0, growing, 0.0)
            for identity, solvent, growing in zip(
                identities, solvency, growth, strict=True
            )
        )
        batch = _batch(predictions)
        rescue_incumbent = _outcome(
            _identity(9, "query", 0), score=0.0, b2=1000.0
        )
        top1 = propose_g4(batch, rescue_incumbent, budget=1)
        self.assertEqual(top1.mode, "rescue")
        self.assertEqual(top1.identities, (identities[1],))
        self.assertEqual(top1.admission_sources, ("solvency-quota",))
        top4 = propose_g4(batch, rescue_incumbent, budget=4)
        self.assertEqual(
            top4.identities,
            (identities[1], identities[2], identities[0], identities[3]),
        )
        self.assertEqual(
            top4.admission_sources,
            (
                "solvency-quota",
                "solvency-quota",
                "growth-quota",
                "growth-quota",
            ),
        )
        growth_incumbent = _outcome(
            _identity(9, "query", 0), score=0.0, b2=2000.0
        )
        growth_top1 = propose_g4(batch, growth_incumbent, budget=1)
        self.assertEqual(growth_top1.mode, "growth")
        self.assertEqual(growth_top1.identities, (identities[0],))
        self.assertEqual(growth_top1.admission_sources, ("growth-quota",))
        tied = tuple(
            G4Prediction(identity, 0.5, 0.0, 0.5, 0.0)
            for identity in identities
        )
        self.assertEqual(
            propose_g4(
                _batch(tied), rescue_incumbent, budget=2
            ).identities,
            (identities[0], identities[1]),
        )
        with self.assertRaisesRegex(ValueError, "inventory"):
            _batch((predictions[0], predictions[2]))
        duplicate = G4CandidateIdentity(
            9,
            "query",
            2,
            identities[0].candidate_id,
            _digest("distinct-action"),
        )
        with self.assertRaisesRegex(ValueError, "inventory"):
            _batch(
                (
                    predictions[0],
                    G4Prediction(duplicate, 0.5, 0.0, 0.5, 0.0),
                )
            )
        with self.assertRaisesRegex(ValueError, "differ"):
            propose_g4(
                batch,
                replace(
                    rescue_incumbent,
                    observation_sha256=_digest("foreign-observation"),
                ),
                budget=2,
            )
        with self.assertRaisesRegex(ValueError, "differ"):
            propose_g4(
                batch,
                replace(
                    rescue_incumbent,
                    identity=G4CandidateIdentity(
                        9,
                        "query",
                        0,
                        _digest("foreign-incumbent-candidate"),
                        _digest("foreign-incumbent-action"),
                    ),
                ),
                budget=2,
            )

    def test_exact_rescue_selection_and_terminal_closure(self) -> None:
        identities = tuple(_identity(7, "query", ordinal) for ordinal in range(4))
        predictions = tuple(
            G4Prediction(identity, 0.5, 0.0, 0.9 - 0.1 * identity.ordinal, 0.0)
            for identity in identities[1:]
        )
        incumbent = _outcome(identities[0], score=0.0, b2=1000.0)
        proposal = propose_g4(_batch(predictions), incumbent, budget=4)
        by_identity = {
            identities[1]: _outcome(identities[1], score=99.0, b2=900.0),
            identities[2]: _outcome(identities[2], score=-5.0, b2=1500.0),
            identities[3]: _outcome(identities[3], score=7.0, b2=1500.0),
        }
        outcomes = (
            incumbent,
            *(by_identity[identity] for identity in proposal.identities),
        )
        selected = select_exact_g4(proposal, outcomes)
        self.assertEqual(selected.status, "selected-rescue")
        self.assertEqual(selected.identity, identities[3])

        ineligible = (
            incumbent,
            *(
                _outcome(identity, score=99.0, b2=900.0)
                for identity in proposal.identities
            ),
        )
        retained = select_exact_g4(proposal, ineligible)
        self.assertEqual(retained.status, "incumbent-retained")
        self.assertEqual(retained.identity, identities[0])
        tied_b2 = (
            incumbent,
            *(
                _outcome(identity, score=999.0, b2=1000.0)
                for identity in proposal.identities
            ),
        )
        self.assertEqual(
            select_exact_g4(proposal, tied_b2).status,
            "incumbent-retained",
        )

        unsafe_incumbent = _outcome(
            identities[0], score=0.0, safe=False, unsafe=True
        )
        unsafe_proposal = propose_g4(
            _batch(predictions), unsafe_incumbent, budget=4
        )
        safe_low = {
            identity: _outcome(identity, score=-10.0, b2=10.0)
            for identity in unsafe_proposal.identities
        }
        rescued = select_exact_g4(
            unsafe_proposal,
            (
                unsafe_incumbent,
                *(safe_low[identity] for identity in unsafe_proposal.identities),
            ),
        )
        self.assertEqual(rescued.status, "selected-rescue")
        no_safe = select_exact_g4(
            unsafe_proposal,
            (
                unsafe_incumbent,
                *(
                    _outcome(
                        identity, score=99.0, safe=False, unsafe=True
                    )
                    for identity in unsafe_proposal.identities
                ),
            ),
        )
        self.assertEqual(no_safe.status, "no-safe-proposal-abstention")

        with self.assertRaisesRegex(ValueError, "closure"):
            select_exact_g4(proposal, outcomes[:-1])
        with self.assertRaisesRegex(ValueError, "closure"):
            select_exact_g4(
                proposal,
                (outcomes[0], outcomes[1], outcomes[1], outcomes[3]),
            )
        with self.assertRaisesRegex(ValueError, "closure"):
            select_exact_g4(
                proposal,
                (outcomes[0], outcomes[2], outcomes[1], outcomes[3]),
            )
        foreign = replace(
            outcomes[1],
            identity=G4CandidateIdentity(
                7,
                "query",
                outcomes[1].identity.ordinal,
                _digest("foreign-candidate"),
                _digest("foreign-action"),
            ),
        )
        with self.assertRaisesRegex(ValueError, "closure"):
            select_exact_g4(
                proposal, (outcomes[0], foreign, outcomes[2], outcomes[3])
            )
        with self.assertRaisesRegex(ValueError, "closure"):
            select_exact_g4(
                proposal,
                (
                    outcomes[0],
                    replace(
                        outcomes[1],
                        observation_sha256=_digest("mixed-observation"),
                    ),
                    outcomes[2],
                    outcomes[3],
                ),
            )
        with self.assertRaisesRegex(ValueError, "closure"):
            select_exact_g4(
                proposal,
                (
                    replace(
                        outcomes[0], source_sha256=_digest("foreign-incumbent")
                    ),
                    *outcomes[1:],
                ),
            )
        with self.assertRaisesRegex(ValueError, "closure"):
            select_exact_g4(
                proposal,
                (
                    outcomes[0],
                    replace(outcomes[1], level=2),
                    outcomes[2],
                    outcomes[3],
                ),
            )

    def test_exact_growth_selection_respects_reserve_and_score(self) -> None:
        identities = tuple(_identity(8, "query", ordinal) for ordinal in range(4))
        predictions = tuple(
            G4Prediction(identity, 0.5, 0.0, 0.5, 0.0)
            for identity in identities[1:]
        )
        incumbent = _outcome(identities[0], score=0.0, b2=2000.0)
        proposal = propose_g4(_batch(predictions), incumbent, budget=4)
        self.assertEqual(proposal.mode, "growth")
        at_reserve = _outcome(
            identities[0], score=0.0, b2=float(one_rot_liability(1))
        )
        self.assertEqual(
            propose_g4(_batch(predictions), at_reserve, budget=1).mode,
            "growth",
        )
        by_identity = {
            identities[1]: _outcome(identities[1], score=100.0, b2=1819.0),
            identities[2]: _outcome(identities[2], score=7.0, b2=1900.0),
            identities[3]: _outcome(identities[3], score=7.0, b2=2000.0),
        }
        outcomes = (
            incumbent,
            *(by_identity[identity] for identity in proposal.identities),
        )
        selected = select_exact_g4(proposal, outcomes)
        self.assertEqual(selected.status, "selected-growth")
        self.assertEqual(selected.identity, identities[3])
        nonpositive = (
            incumbent,
            *(
                replace(
                    by_identity[identity],
                    exact_score_advantage=-1.0,
                )
                for identity in proposal.identities
            ),
        )
        self.assertEqual(
            select_exact_g4(proposal, nonpositive).status,
            "incumbent-retained",
        )

    def test_record_certificate_is_absolute_safe_and_identity_bound(self) -> None:
        incumbent, safe, unsafe = _records()
        incumbent_outcome = exact_outcome_g4(incumbent)
        safe_outcome = exact_outcome_g4(safe)
        unsafe_outcome = exact_outcome_g4(unsafe)
        self.assertTrue(incumbent_outcome.absolute_safe)
        self.assertTrue(safe_outcome.absolute_safe)
        self.assertFalse(unsafe_outcome.absolute_safe)
        self.assertIsNone(unsafe_outcome.certificate)
        certificate = exact_certificate_g4(safe)
        self.assertIsNotNone(certificate)
        assert certificate is not None
        self.assertEqual(certificate.identity.ordinal, safe.ordinal)
        self.assertEqual(certificate.source_sha256, safe.source_sha256)
        self.assertEqual(certificate.level, 1)
        self.assertIsNone(exact_certificate_g4(unsafe))


class CheckpointExpectationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.board, cls.model = _shared_board_model()

    def test_checkpoint_roundtrip_and_every_expectation_is_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "g4.json"
            digest = save_checkpoint_g4(
                path,
                self.model,
                root=root,
                metadata=METADATA,
                expected_model_sha256=self.model.sha256,
                expected_partition_sha256=self.model.partition_sha256,
                expected_dataset_sha256=self.model.training_dataset_sha256,
                expected_feature_inventory_sha256=(
                    self.model.training_feature_inventory_sha256
                ),
                dependencies=DEPENDENCIES,
            )
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            baseline = {
                "root": root,
                "expected_metadata": METADATA,
                "expected_model_sha256": self.model.sha256,
                "expected_partition_sha256": self.model.partition_sha256,
                "expected_dataset_sha256": self.model.training_dataset_sha256,
                "expected_feature_inventory_sha256": (
                    self.model.training_feature_inventory_sha256
                ),
                "expected_dependencies": DEPENDENCIES,
            }
            loaded, metadata = load_checkpoint_g4(path, **baseline)
            self.assertEqual(metadata, METADATA)
            self.assertEqual(loaded.manifest(), self.model.manifest())
            inference = _inference_board(self.board)
            self.assertEqual(
                loaded.predict(inference, oof=True),
                self.model.predict(inference, oof=True),
            )
            self.assertEqual(
                loaded.predict_batches(inference, oof=True),
                self.model.predict_batches(inference, oof=True),
            )
            real_public = g4_inference_board_from_entries(
                (_public_projection(_opened_train_entries()[0]),)
            )
            self.assertEqual(
                loaded.predict_batches(real_public),
                self.model.predict_batches(real_public),
            )
            changes = (
                {"expected_metadata": {"stage": "foreign"}},
                {"expected_model_sha256": "0" * 64},
                {"expected_partition_sha256": "1" * 64},
                {"expected_dataset_sha256": "2" * 64},
                {"expected_feature_inventory_sha256": "3" * 64},
                {"expected_dependencies": {"foreign": "4" * 64}},
            )
            for change in changes:
                with self.subTest(change=change):
                    with self.assertRaises(RuntimeError):
                        load_checkpoint_g4(path, **{**baseline, **change})
            tampered = json.loads(path.read_text())
            tampered["model"]["feature_width"] = float(WIDE_FEATURE_WIDTH)
            tampered["model_sha256"] = _manifest_sha256(tampered["model"])
            body = {
                key: value
                for key, value in tampered.items()
                if key != "checkpoint_sha256"
            }
            tampered["checkpoint_sha256"] = _manifest_sha256(body)
            typed_path = root / "typed-feature-width.json"
            typed_path.write_bytes(_canonical_bytes(tampered))
            with self.assertRaises(RuntimeError):
                load_checkpoint_g4(
                    typed_path,
                    **{
                        **baseline,
                        "expected_model_sha256": tampered["model_sha256"],
                    },
                )
            with self.assertRaises(RuntimeError):
                save_checkpoint_g4(
                    path,
                    self.model,
                    root=root,
                    metadata=METADATA,
                    expected_model_sha256=self.model.sha256,
                    expected_partition_sha256=self.model.partition_sha256,
                    expected_dataset_sha256=self.model.training_dataset_sha256,
                    expected_feature_inventory_sha256=(
                        self.model.training_feature_inventory_sha256
                    ),
                    dependencies=DEPENDENCIES,
                )

    def test_checkpoint_rejects_symlink_and_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            save_checkpoint_g4(
                source,
                self.model,
                root=root,
                metadata=METADATA,
                expected_model_sha256=self.model.sha256,
                expected_partition_sha256=self.model.partition_sha256,
                expected_dataset_sha256=self.model.training_dataset_sha256,
                expected_feature_inventory_sha256=(
                    self.model.training_feature_inventory_sha256
                ),
                dependencies=DEPENDENCIES,
            )
            symlink = root / "symlink.json"
            symlink.symlink_to(source.name)
            with self.assertRaises(RuntimeError):
                load_checkpoint_g4(
                    symlink,
                    root=root,
                    expected_metadata=METADATA,
                    expected_model_sha256=self.model.sha256,
                    expected_partition_sha256=self.model.partition_sha256,
                    expected_dataset_sha256=self.model.training_dataset_sha256,
                    expected_feature_inventory_sha256=(
                        self.model.training_feature_inventory_sha256
                    ),
                    expected_dependencies=DEPENDENCIES,
                )
            hardlink = root / "hardlink.json"
            hardlink.hardlink_to(source)
            with self.assertRaises(RuntimeError):
                load_checkpoint_g4(
                    hardlink,
                    root=root,
                    expected_metadata=METADATA,
                    expected_model_sha256=self.model.sha256,
                    expected_partition_sha256=self.model.partition_sha256,
                    expected_dataset_sha256=self.model.training_dataset_sha256,
                    expected_feature_inventory_sha256=(
                        self.model.training_feature_inventory_sha256
                    ),
                    expected_dependencies=DEPENDENCIES,
                )


if __name__ == "__main__":
    unittest.main()
