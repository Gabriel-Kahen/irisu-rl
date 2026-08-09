from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

import torch

from irisu_pointer.geometry_learning import (
    GeometryDataset,
    GeometryModelConfig,
    GeometrySelectorModel,
    geometry_example,
)
from irisu_pointer.geometry_ranking import (
    GeometryRankingDataset,
    geometry_ranking_example,
)
from irisu_pointer.geometry_search import (
    GeometryBranchOutcome,
    GeometrySearchConfig,
    enumerate_geometry_candidates,
)
from irisu_pointer.runway_search import (
    RunwayGeometrySearch,
    RunwaySearchConfig,
    RunwaySearchResult,
)
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import SemanticAction
from irisu_rl.schema import TEACHER_V1


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks/rl_r3e_sustainable.py"
SPEC = importlib.util.spec_from_file_location(
    "rl_r3e_sustainable_benchmark", BENCHMARK_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _body(identifier: int, x: float, color: int = 0) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": "dynamic_fresh",
        "color": color,
        "x": x,
        "y": 180.0,
        "vx": 0.0,
        "vy": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 40.0,
        "chain_id": 0,
        "projectile_hits": 0,
        "age_ticks": 20,
        "remaining_lifetime": 1000,
        "rot_timer": 0,
    }


def _observation(tick: int = 10) -> dict[str, object]:
    return {
        "tick": tick,
        "score": 0,
        "gauge": 1000,
        "gauge_max": 1000,
        "level": 1,
        "highest_chain": 0,
        "qualifying_clear_count": 0,
        "left_held": False,
        "right_held": False,
        "terminated": False,
        "truncated": False,
        "field": {
            "x": 0.0,
            "y": 0.0,
            "width": 640.0,
            "height": 480.0,
        },
        "difficulty": {
            "active_colors": 4,
            "spawn_interval_ticks": 100,
        },
        "bodies": (
            _body(11, 160.0),
            _body(22, 300.0),
            _body(33, 480.0, 1),
        ),
    }


def _incumbent() -> SteeringDecision:
    return SteeringDecision(
        SemanticAction.strong(180.0 / 640.0, 150.0 / 480.0),
        SteeringIntent.STEER_MATCH,
        source_body_id=11,
        destination_body_id=22,
        impact_x_sizes=-0.5,
        impact_y_sizes=0.75,
        correction_index=1,
        reason="fixture incumbent",
    )


def _dataset(teacher: RunwayGeometrySearch) -> GeometryDataset:
    vocabulary_sha256 = BENCHMARK.geometry_candidate_vocabulary_sha256(
        teacher.config.candidate_config
    )
    return GeometryDataset(
        (
            geometry_example(
                _observation(),
                source_body_id=11,
                destination_body_id=22,
                candidate_index=2,
                candidate_count=teacher.config.candidate_config.slot_count,
                improved_over_incumbent=True,
                episode_identity="fixture:1",
                provenance_sha256=_sha("fixture provenance"),
                candidate_set_sha256=vocabulary_sha256,
            ),
        )
    )


def _ranking_dataset(
    teacher: _FixtureTeacher | None = None,
    *,
    episode_identity: str = "ranking:1",
    provenance: str = "ranking provenance",
) -> GeometryRankingDataset:
    resolved = _FixtureTeacher() if teacher is None else teacher
    observation = _observation()
    result = resolved.search(_StableEnv(), observation, _incumbent())
    return GeometryRankingDataset(
        (
            geometry_ranking_example(
                observation,
                result,
                episode_identity=episode_identity,
                provenance_sha256=_sha(provenance),
            ),
        )
    )


class _BasePolicy:
    def reset(self, seed: int) -> None:
        self.seed = seed

    def predict(self, observation: object) -> SteeringDecision:
        del observation
        return _incumbent()


class _StableEnv:
    physics_backend = "portable"

    def __init__(self) -> None:
        self.state = b"stable"

    def clone_state(self) -> bytes:
        return self.state


class _FixtureTeacher:
    def __init__(self) -> None:
        real = RunwayGeometrySearch(
            config=RunwaySearchConfig(
                runway_ticks=256,
                candidate_config=GeometrySearchConfig(),
            )
        )
        self.config = real.config
        self.action_spec = real.action_spec
        self.sha256 = real.sha256

    def search(
        self,
        env: _StableEnv,
        observation: dict[str, object],
        incumbent: SteeringDecision,
    ) -> RunwaySearchResult:
        del env
        candidates = enumerate_geometry_candidates(
            observation,
            incumbent,
            config=self.config.candidate_config,
            action_spec=self.action_spec,
        )
        selected = candidates.candidate_at(1)
        assert selected is not None
        outcomes = tuple(
            GeometryBranchOutcome(
                candidate=candidate,
                score_gain=10 if candidate.ordinal == selected.ordinal else 0,
                alive=True,
                survival_ticks=self.config.runway_ticks,
                final_gauge=1000,
                gauge_max=1000,
                qualifying_clear_gain=0,
                highest_chain_gain=0,
                intended_source_hits=int(
                    candidate.ordinal == selected.ordinal
                ),
                intended_pair_joined=(
                    candidate.ordinal == selected.ordinal
                ),
                pair_closure_sizes=1.0,
                invalid_actions=0,
            )
            for candidate in candidates.candidates
        )
        return RunwaySearchResult(
            teacher_identity_sha256=self.sha256,
            candidate_set=candidates,
            runway_ticks=self.config.runway_ticks,
            selected_candidate=selected,
            strictly_improved=True,
            outcomes=outcomes,
        )


def _curve_point(
    *,
    steps: int,
    failures: int,
    p10: float,
    median_survival: float,
    median_score: float,
    new_failures: int = 0,
    new_terminal_failures: int | None = None,
    catastrophic_regressions: int = 0,
    rescued_failures: int = 0,
    minimum_survival_ratio: float = 1.0,
    minimum_survival_delta: int = 0,
    survival_regressions: int = 0,
    horizons: tuple[int, ...] = (10_000,),
) -> dict[str, object]:
    terminal_failures = (
        new_failures
        if new_terminal_failures is None
        else new_terminal_failures
    )

    def paired(horizon: int) -> dict[str, object]:
        return {
            "schema": "irisu-r3e-paired-safety-v1",
            "horizon_ticks": horizon,
            "new_terminal_failures": terminal_failures,
            "new_terminal_failure_seeds": list(range(terminal_failures)),
            "new_gauge_failures": new_failures,
            "new_gauge_failure_seeds": list(range(new_failures)),
            "catastrophic_survival_regressions": catastrophic_regressions,
            "catastrophic_survival_regression_seeds": list(
                range(catastrophic_regressions)
            ),
            "candidate_gauge_failures": failures,
            "rescued_gauge_failures": rescued_failures,
            "minimum_survival_ratio": minimum_survival_ratio,
            "minimum_survival_delta": minimum_survival_delta,
            "survival_regressions": survival_regressions,
        }
    return {
        "training_steps": steps,
        "state_dict_sha256": _sha(f"curve:{steps}"),
        "evaluation": {
            str(horizon): {
                "paired_safety": paired(horizon),
                "aggregate": {
                    "gauge_failures": failures,
                    "survival_ticks": {
                        "p10": p10,
                        "median": median_survival,
                    },
                    "raw_score": {"median": median_score},
                }
            }
            for horizon in horizons
        },
    }


def _base_evaluation(
    *,
    failures: int,
    p10: float,
    median_survival: float,
    median_score: float,
    horizons: tuple[int, ...] = (10_000,),
) -> dict[str, object]:
    return {
        str(horizon): {
            "aggregate": {
                "gauge_failures": failures,
                "survival_ticks": {
                    "p10": p10,
                    "median": median_survival,
                },
                "raw_score": {"median": median_score},
            }
        }
        for horizon in horizons
    }


def _paired_evaluation(
    rows: tuple[tuple[int, int, int, bool], ...],
    *,
    horizon: int = 10_000,
    runner_config_hash: int = 7,
) -> dict[str, object]:
    return {
        "runner": {
            "version": "irisu-env-runner-identity-v1",
            "physics_backend": "portable",
            "config_hash": runner_config_hash,
        },
        "episodes": [
            {
                "seed": seed,
                "final_gauge": 1 if failed else 1000,
                "gauge_max": 1000,
                "gauge_failure": failed,
                "conversion": {
                    "survival_ticks": survival,
                    "final_score": score,
                    "terminated": failed,
                    "truncated": not failed,
                },
            }
            for seed, survival, score, failed in rows
        ],
        "aggregate": {
            "episodes": len(rows),
            "gauge_failures": sum(failed for _, _, _, failed in rows),
        },
    }


class R3eSustainableBenchmarkTests(unittest.TestCase):
    def test_seed_suites_are_fixed_unique_and_disjoint(self) -> None:
        collection = BENCHMARK._derive_seeds(
            "irisu-r3e-policy-visited-collection-v1"
        )
        development = BENCHMARK._derive_seeds(
            "irisu-r3e-fixed-disjoint-development-v1"
        )
        self.assertEqual(collection, BENCHMARK.COLLECTION_SEEDS)
        self.assertEqual(development, BENCHMARK.DEVELOPMENT_SEEDS)
        self.assertEqual(len(set(collection)), len(collection))
        self.assertEqual(len(set(development)), len(development))
        prior = (
            set(BENCHMARK.r3d.DEMONSTRATION_SEEDS)
            | set(BENCHMARK.r3d.UNSEEN_DEVELOPMENT_SEEDS)
            | set(BENCHMARK.r3d.SURVIVAL_HOLDOUT_SEEDS)
        )
        self.assertFalse(set(collection) & set(development))
        self.assertFalse((set(collection) | set(development)) & prior)

    def test_config_defaults_to_spawn_crossing_256_tick_oracle(self) -> None:
        config = tomllib.loads(BENCHMARK.DEFAULT_CONFIG.read_text())
        teacher = BENCHMARK._collection_teacher(
            config["collection_teacher"],
            BENCHMARK._search_config(config["search"]),
        )

        self.assertIsInstance(teacher, RunwayGeometrySearch)
        self.assertEqual(teacher.config.runway_ticks, 256)
        self.assertEqual(config["profiles"]["fast"]["query_stride_shots"], 2)
        self.assertGreaterEqual(
            config["profiles"]["fast"][
                "maximum_search_queries_per_episode"
            ],
            64,
        )
        self.assertFalse(config["sealed_evaluation_allowed"])
        self.assertFalse(config["canonical_r3_evidence"])
        self.assertEqual(
            config["geometry_model"]["board_context_hidden"], 128
        )
        self.assertEqual(
            config["selection"]["eligibility_horizons"],
            "all_profile_evaluation_horizons",
        )
        self.assertEqual(
            config["selection"]["ranking_horizon"],
            "longest_profile_evaluation_horizon",
        )
        self.assertEqual(
            config["selection"]["baseline_tie_policy"],
            "retain_frozen_v5",
        )

    def test_medium_and_deep_profiles_cover_long_evaluation(self) -> None:
        config = tomllib.loads(BENCHMARK.DEFAULT_CONFIG.read_text())
        medium = config["profiles"]["medium"]
        deep = config["profiles"]["deep"]

        self.assertEqual(medium["collection_seeds"], 8)
        self.assertGreaterEqual(max(medium["evaluation_horizons"]), 20_000)
        self.assertEqual(deep["collection_seeds"], 8)
        self.assertGreaterEqual(max(deep["evaluation_horizons"]), 50_000)
        self.assertGreater(
            max(deep["training_steps"]), max(medium["training_steps"])
        )
        self.assertTrue(
            config["training"]["ranking"]["all_available_candidate_outcomes"]
        )

    def test_collection_artifact_round_trip_is_identity_bound(self) -> None:
        teacher = RunwayGeometrySearch()
        dataset = _dataset(teacher)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "collection.pt"
            sha256 = BENCHMARK.save_geometry_collection(
                path,
                dataset,
                metadata={"teacher_sha256": teacher.sha256},
            )
            loaded = BENCHMARK.load_geometry_collection(
                path, expected_sha256=sha256
            )

            self.assertEqual(loaded.dataset.sha256, dataset.sha256)
            self.assertEqual(loaded.dataset[0].sha256, dataset[0].sha256)
            self.assertTrue(loaded.metadata["development_only"])
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                BENCHMARK.load_geometry_collection(
                    path, expected_sha256="0" * 64
                )

    def test_geometry_checkpoint_round_trip_is_identity_bound(self) -> None:
        teacher = RunwayGeometrySearch()
        torch.manual_seed(7)
        model = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=teacher.config.candidate_config.slot_count,
            candidate_set_sha256=(
                BENCHMARK.geometry_candidate_vocabulary_sha256(
                    teacher.config.candidate_config
                )
            ),
            config=GeometryModelConfig(body_hidden=16, pair_hidden=24),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.pt"
            sha256 = BENCHMARK.save_geometry_checkpoint(
                path,
                model,
                search_identity=teacher.identity_manifest(),
            )
            loaded = BENCHMARK.load_geometry_checkpoint(
                path, expected_sha256=sha256
            )

            self.assertEqual(
                loaded.search_identity, teacher.identity_manifest()
            )
            self.assertEqual(
                BENCHMARK._state_dict_sha256(loaded.model),
                BENCHMARK._state_dict_sha256(model),
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                BENCHMARK.load_geometry_checkpoint(
                    path, expected_sha256="0" * 64
                )

    def test_ranking_collection_round_trip_retains_all_branches(self) -> None:
        dataset = _ranking_dataset()
        example = dataset[0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranking.pt"
            sha256 = BENCHMARK.save_geometry_ranking_collection(
                path,
                dataset,
                metadata={"rollout_mode": "oracle_visited"},
            )
            loaded = BENCHMARK.load_geometry_ranking_collection(
                path, expected_sha256=sha256
            )

            self.assertEqual(loaded.dataset.sha256, dataset.sha256)
            self.assertEqual(
                loaded.dataset[0].outcome_sha256s,
                example.outcome_sha256s,
            )
            self.assertEqual(
                loaded.dataset[0].preferences, example.preferences
            )
            self.assertGreater(
                len(example.preferences), len(example.outcome_sha256s)
            )
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                BENCHMARK.load_geometry_ranking_collection(
                    path, expected_sha256="0" * 64
                )

    def test_ranking_merge_deduplicates_and_rejects_conflicts(self) -> None:
        first = _ranking_dataset()
        merged = BENCHMARK.merge_geometry_ranking_datasets((first, first))
        conflict = _ranking_dataset(provenance="different provenance")

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.sha256, first.sha256)
        with self.assertRaisesRegex(ValueError, "conflicting episode"):
            BENCHMARK.merge_geometry_ranking_datasets((first, conflict))

    def test_ranking_plateau_training_uses_all_preferences(self) -> None:
        dataset = _ranking_dataset()
        curves = BENCHMARK.train_ranking_plateau_models(
            dataset,
            budgets=(1, 2),
            model_config=GeometryModelConfig(
                body_hidden=8, pair_hidden=8
            ),
            batch_size=1,
            learning_rate=1e-3,
            listwise_weight=1.0,
            pairwise_weight=1.0,
            seed=17,
        )

        self.assertEqual([curve.steps for curve in curves], [1, 2])
        self.assertTrue(
            all(
                curve.report.preferences
                == len(dataset[0].preferences)
                for curve in curves
            )
        )
        self.assertNotEqual(
            curves[0].state_dict_sha256, curves[1].state_dict_sha256
        )

    def test_collection_executes_queried_oracle_winner(self) -> None:
        teacher = _FixtureTeacher()
        policy = BENCHMARK.OracleVisitedCollectionPolicy(
            env=_StableEnv(),
            base_policy=_BasePolicy(),
            teacher=teacher,
            seed=7,
            episode_ticks=1000,
            query_stride_shots=1,
            maximum_search_queries=4,
            source_identity_sha256=_sha("source"),
            runtime_sha256=_sha("runtime"),
            base_policy_sha256=_sha("base"),
        )
        policy.reset(7)
        incumbent = _incumbent()
        decision = policy.predict(_observation())

        self.assertNotEqual(decision.action, incumbent.action)
        self.assertEqual(len(policy.examples), 1)
        self.assertEqual(policy.examples[0].candidate_index, 1)
        self.assertEqual(
            policy.examples[0].candidate_set_sha256,
            BENCHMARK.geometry_candidate_vocabulary_sha256(
                teacher.config.candidate_config
            ),
        )
        self.assertEqual(policy.statistics()["search_queries"], 1)
        self.assertEqual(
            policy.statistics()["transactional_restore_checks"], 1
        )
        self.assertEqual(len(policy.ranking_examples), 1)
        self.assertGreater(
            len(policy.ranking_examples[0].preferences),
            len(policy.ranking_examples[0].outcome_sha256s),
        )

    def test_learner_visited_hook_labels_without_executing_oracle(self) -> None:
        teacher = _FixtureTeacher()
        policy = BENCHMARK.OracleVisitedCollectionPolicy(
            env=_StableEnv(),
            base_policy=_BasePolicy(),
            teacher=teacher,
            seed=7,
            episode_ticks=1000,
            query_stride_shots=1,
            maximum_search_queries=4,
            source_identity_sha256=_sha("source"),
            runtime_sha256=_sha("runtime"),
            base_policy_sha256=_sha("base"),
            rollout_mode="learner_visited",
        )
        policy.reset(7)

        decision = policy.predict(_observation())

        self.assertEqual(decision, _incumbent())
        self.assertEqual(len(policy.ranking_examples), 1)
        self.assertIn(
            "learner-visited",
            policy.ranking_examples[0].episode_identity,
        )

    def test_collection_skips_queries_without_a_complete_runway(self) -> None:
        policy = BENCHMARK.OracleVisitedCollectionPolicy(
            env=_StableEnv(),
            base_policy=_BasePolicy(),
            teacher=_FixtureTeacher(),
            seed=7,
            episode_ticks=300,
            query_stride_shots=1,
            maximum_search_queries=4,
            source_identity_sha256=_sha("source"),
            runtime_sha256=_sha("runtime"),
            base_policy_sha256=_sha("base"),
        )
        policy.reset(7)

        self.assertEqual(policy.predict(_observation(tick=100)), _incumbent())
        self.assertFalse(policy.examples)
        self.assertEqual(policy.statistics()["episode_boundary_skips"], 1)

    def test_plateau_selection_is_survival_first(self) -> None:
        safe = _curve_point(
            steps=240,
            failures=0,
            p10=8000,
            median_survival=9000,
            median_score=100,
        )
        high_score_failure = _curve_point(
            steps=120,
            failures=1,
            p10=10000,
            median_survival=10000,
            median_score=1_000_000,
        )
        longer_tail = _curve_point(
            steps=480,
            failures=0,
            p10=9000,
            median_survival=9000,
            median_score=50,
        )

        self.assertGreater(
            BENCHMARK._selection_key(safe, (10_000,)),
            BENCHMARK._selection_key(high_score_failure, (10_000,)),
        )
        self.assertGreater(
            BENCHMARK._selection_key(longer_tail, (10_000,)),
            BENCHMARK._selection_key(safe, (10_000,)),
        )

    def test_plateau_selection_rejects_new_paired_failures_first(self) -> None:
        aggregate_winner = _curve_point(
            steps=120,
            failures=1,
            p10=10000,
            median_survival=10000,
            median_score=1_000_000,
            new_failures=1,
        )
        paired_safe = _curve_point(
            steps=240,
            failures=1,
            p10=5000,
            median_survival=7000,
            median_score=100,
        )
        catastrophic = _curve_point(
            steps=480,
            failures=0,
            p10=10000,
            median_survival=10000,
            median_score=1_000_000,
            catastrophic_regressions=1,
        )

        self.assertGreater(
            BENCHMARK._selection_key(paired_safe, (10_000,)),
            BENCHMARK._selection_key(aggregate_winner, (10_000,)),
        )
        self.assertGreater(
            BENCHMARK._selection_key(paired_safe, (10_000,)),
            BENCHMARK._selection_key(catastrophic, (10_000,)),
        )

    def test_paired_safety_reports_failures_rescues_and_catastrophes(
        self,
    ) -> None:
        baseline = _paired_evaluation(
            (
                (1, 10_000, 100, False),
                (2, 4_000, 50, True),
                (3, 8_000, 90, False),
            )
        )
        candidate = _paired_evaluation(
            (
                (1, 4_000, 200, True),
                (2, 10_000, 60, False),
                (3, 7_500, 80, False),
            )
        )

        paired = BENCHMARK._paired_safety_comparison(
            candidate,
            baseline,
            horizon_ticks=10_000,
            catastrophic_survival_ratio=0.5,
            catastrophic_survival_loss_ticks=1_000,
        )

        self.assertEqual(paired["new_gauge_failure_seeds"], [1])
        self.assertEqual(paired["rescued_gauge_failure_seeds"], [2])
        self.assertEqual(
            paired["catastrophic_survival_regression_seeds"], [1]
        )
        self.assertEqual(paired["survival_regressions"], 2)
        self.assertEqual(paired["survival_improvements"], 1)
        self.assertEqual(paired["minimum_survival_delta"], -6_000)
        self.assertEqual(paired["minimum_survival_ratio"], 0.4)
        self.assertEqual(paired["new_terminal_failure_seeds"], [1])
        self.assertEqual(paired["material_survival_regression_seeds"], [1])
        self.assertEqual([row["seed"] for row in paired["pairs"]], [1, 2, 3])

    def test_paired_safety_requires_identical_unique_seeds(self) -> None:
        candidate = _paired_evaluation(((1, 10, 1, False),))
        baseline = _paired_evaluation(((2, 10, 1, False),))
        with self.assertRaisesRegex(ValueError, "different seeds"):
            BENCHMARK._paired_safety_comparison(
                candidate,
                baseline,
                horizon_ticks=10_000,
                catastrophic_survival_ratio=0.5,
                catastrophic_survival_loss_ticks=1_000,
            )
        duplicate = _paired_evaluation(
            ((1, 10, 1, False), (1, 10, 1, False))
        )
        with self.assertRaisesRegex(ValueError, "seeds are invalid"):
            BENCHMARK._paired_safety_comparison(
                duplicate,
                duplicate,
                horizon_ticks=10_000,
                catastrophic_survival_ratio=0.5,
                catastrophic_survival_loss_ticks=1_000,
            )

    def test_paired_safety_validates_runner_and_gauge_manifest(self) -> None:
        baseline = _paired_evaluation(((1, 10_000, 1, False),))
        foreign_runner = _paired_evaluation(
            ((1, 10_000, 1, False),),
            runner_config_hash=8,
        )
        with self.assertRaisesRegex(ValueError, "runner identities"):
            BENCHMARK._paired_safety_comparison(
                foreign_runner,
                baseline,
                horizon_ticks=10_000,
                catastrophic_survival_ratio=0.5,
                catastrophic_survival_loss_ticks=1_000,
            )

        inconsistent = _paired_evaluation(((1, 5_000, 1, True),))
        inconsistent["episodes"][0]["final_gauge"] = 1000
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            BENCHMARK._paired_safety_comparison(
                inconsistent,
                baseline,
                horizon_ticks=10_000,
                catastrophic_survival_ratio=0.5,
                catastrophic_survival_loss_ticks=1_000,
            )

    def test_paired_safety_accepts_signed_terminal_gauge_overshoot(
        self,
    ) -> None:
        baseline = _paired_evaluation(((1, 10_000, 100, False),))
        candidate = _paired_evaluation(((1, 9_995, 95, True),))
        candidate["episodes"][0]["final_gauge"] = -1_879

        paired = BENCHMARK._paired_safety_comparison(
            candidate,
            baseline,
            horizon_ticks=10_000,
            catastrophic_survival_ratio=0.5,
            catastrophic_survival_loss_ticks=1_000,
        )

        self.assertEqual(paired["new_terminal_failure_seeds"], [1])
        self.assertEqual(paired["new_gauge_failure_seeds"], [1])

    def test_catastrophic_threshold_uses_ratio_and_absolute_loss(self) -> None:
        baseline = _paired_evaluation(((1, 10_000, 1, False),))
        at_boundary = _paired_evaluation(((1, 5_000, 1, False),))
        above_boundary = _paired_evaluation(((1, 5_001, 1, False),))

        catastrophic = BENCHMARK._paired_safety_comparison(
            at_boundary,
            baseline,
            horizon_ticks=10_000,
            catastrophic_survival_ratio=0.5,
            catastrophic_survival_loss_ticks=1_000,
        )
        material_only = BENCHMARK._paired_safety_comparison(
            above_boundary,
            baseline,
            horizon_ticks=10_000,
            catastrophic_survival_ratio=0.5,
            catastrophic_survival_loss_ticks=1_000,
        )
        below_absolute_floor = BENCHMARK._paired_safety_comparison(
            _paired_evaluation(((1, 0, 1, False),)),
            _paired_evaluation(((1, 999, 1, False),)),
            horizon_ticks=10_000,
            catastrophic_survival_ratio=0.5,
            catastrophic_survival_loss_ticks=1_000,
        )

        self.assertEqual(catastrophic["catastrophic_survival_regressions"], 1)
        self.assertEqual(material_only["catastrophic_survival_regressions"], 0)
        self.assertEqual(material_only["material_survival_regressions"], 1)
        self.assertEqual(
            below_absolute_floor["catastrophic_survival_regressions"], 0
        )

    def test_missing_paired_selection_evidence_fails_closed(self) -> None:
        point = _curve_point(
            steps=120,
            failures=0,
            p10=10_000,
            median_survival=10_000,
            median_score=100,
        )
        del point["evaluation"]["10000"]["paired_safety"]

        with self.assertRaisesRegex(ValueError, "paired safety"):
            BENCHMARK._selection_key(point, (10_000,))

    def test_campaign_curve_payloads_require_base_pairing(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires paired base-v5"):
            BENCHMARK._curve_payloads(
                (),
                learner="ranking",
                campaign=True,
                library_path=Path("/tmp/not-used.so"),
                seeds=(1,),
                horizons=(10_000,),
                factory=None,
                base_evaluation=None,
            )

    def test_all_horizon_failure_retains_base_with_rejection_metadata(
        self,
    ) -> None:
        horizons = (10_000, 20_000)
        point = _curve_point(
            steps=120,
            failures=1,
            p10=10_000,
            median_survival=10_000,
            median_score=1_000_000,
            horizons=horizons,
        )
        late = point["evaluation"]["20000"]["paired_safety"]
        late["new_terminal_failures"] = 1
        late["new_terminal_failure_seeds"] = [7]
        late["new_gauge_failures"] = 1
        late["new_gauge_failure_seeds"] = [7]
        selection = BENCHMARK._select_plateau_candidate(
            (point,),
            _base_evaluation(
                failures=0,
                p10=10_000,
                median_survival=10_000,
                median_score=10,
                horizons=horizons,
            ),
            horizons,
        )

        self.assertFalse(selection["accepted_learned_candidate"])
        self.assertEqual(selection["retained_policy"], "frozen_v5")
        self.assertIsNone(selection["selected_curve_index"])
        reasons = selection["rejected_candidates"][0]["rejection_reasons"]
        self.assertTrue(
            any(reason.get("horizon_ticks") == 20_000 for reason in reasons)
        )

    def test_longest_horizon_ranks_and_base_wins_exact_tie(self) -> None:
        horizons = (10_000, 20_000)
        short_winner = _curve_point(
            steps=120,
            failures=0,
            p10=10_000,
            median_survival=10_000,
            median_score=200,
            horizons=horizons,
        )
        long_winner = _curve_point(
            steps=240,
            failures=0,
            p10=9_000,
            median_survival=9_000,
            median_score=100,
            horizons=horizons,
        )
        short_winner["evaluation"]["20000"]["aggregate"]["survival_ticks"][
            "p10"
        ] = 15_000
        long_winner["evaluation"]["20000"]["aggregate"]["survival_ticks"][
            "p10"
        ] = 19_000
        base = _base_evaluation(
            failures=0,
            p10=18_000,
            median_survival=19_000,
            median_score=50,
            horizons=horizons,
        )

        selected = BENCHMARK._select_plateau_candidate(
            (short_winner, long_winner),
            base,
            horizons,
        )
        self.assertTrue(selected["accepted_learned_candidate"])
        self.assertEqual(selected["selected_curve_index"], 1)
        self.assertEqual(selected["ranking_horizon_ticks"], 20_000)

        tie = _curve_point(
            steps=120,
            failures=0,
            p10=18_000,
            median_survival=19_000,
            median_score=50,
            horizons=horizons,
        )
        tied = BENCHMARK._select_plateau_candidate((tie,), base, horizons)
        self.assertFalse(tied["accepted_learned_candidate"])
        self.assertEqual(tied["retained_policy"], "frozen_v5")


if __name__ == "__main__":
    unittest.main()
