from __future__ import annotations

import hashlib
import unittest

import torch

from irisu_pointer.geometry_learning import GeometryModelConfig, GeometrySelectorModel
from irisu_pointer.geometry_policy import geometry_candidate_vocabulary_sha256
from irisu_pointer.geometry_ranking import (
    GeometryRankingDataset,
    geometry_outcome_ordering,
    geometry_ranking_example,
    geometry_ranking_loss,
    train_geometry_ranker,
)
from irisu_pointer.geometry_search import (
    GeometryBranchOutcome,
    GeometrySearchConfig,
    GeometrySearchResult,
    enumerate_geometry_candidates,
)
from irisu_pointer.runway_search import RunwaySearchResult
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import SemanticAction
from irisu_rl.schema import TEACHER_V1


_PROVENANCE = hashlib.sha256(b"geometry-ranking-test").hexdigest()


def _body(identifier: int, x: float) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": "dynamic_fresh",
        "color": 1,
        "x": x,
        "y": 140.0,
        "vx": 0.0,
        "vy": 0.0,
        "vx_display_per_second": 0.0,
        "vy_display_per_second": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 40.0,
        "chain_id": 0,
        "projectile_hits": 0,
        "age_ticks": 20,
        "remaining_lifetime": 1000,
        "rot_timer": 0,
    }


def _observation(gauge: int = 900, *, source_x: float = 200.0):
    return {
        "tick": 1,
        "score": 0,
        "gauge": gauge,
        "gauge_max": 1000,
        "level": 1,
        "highest_chain": 0,
        "qualifying_clear_count": 0,
        "left_held": False,
        "right_held": False,
        "terminated": False,
        "truncated": False,
        "field": {"x": 16.0, "y": 0.0, "width": 576.0, "height": 480.0},
        "difficulty": {"active_colors": 4, "spawn_interval_ticks": 100},
        "bodies": [_body(1, source_x), _body(2, 360.0)],
    }


def _incumbent() -> SteeringDecision:
    return SteeringDecision(
        SemanticAction.strong(180.0 / 640.0, 170.0 / 480.0),
        SteeringIntent.STEER_MATCH,
        source_body_id=1,
        destination_body_id=2,
        destination_chain_id=0,
        impact_x_sizes=-0.5,
        impact_y_sizes=0.75,
        reason="ranking incumbent",
    )


def _outcome(
    candidate,
    *,
    score: int = 0,
    gauge: int = 100,
    gauge_max: int = 100,
    alive: bool = True,
    survival: int = 8,
    invalid: int = 0,
) -> GeometryBranchOutcome:
    return GeometryBranchOutcome(
        candidate=candidate,
        score_gain=score,
        alive=alive,
        survival_ticks=survival,
        final_gauge=gauge,
        gauge_max=gauge_max,
        qualifying_clear_gain=0,
        highest_chain_gain=0,
        intended_source_hits=1 if score else 0,
        intended_pair_joined=bool(score),
        pair_closure_sizes=1.0 if score else 0.0,
        invalid_actions=invalid,
    )


def _result(
    observation,
    *,
    winner_slot: int = 0,
    config: GeometrySearchConfig | None = None,
) -> GeometrySearchResult:
    resolved = GeometrySearchConfig(max_candidates=7) if config is None else config
    candidate_set = enumerate_geometry_candidates(
        observation, _incumbent(), config=resolved
    )
    outcomes = tuple(
        _outcome(
            candidate,
            score=10 if candidate.ordinal == winner_slot and winner_slot else 0,
        )
        for candidate in candidate_set.candidates
    )
    selected = next(
        candidate
        for candidate in candidate_set.candidates
        if candidate.ordinal == winner_slot
    )
    return GeometrySearchResult(
        candidate_set,
        resolved.horizon_ticks,
        8,
        selected,
        winner_slot != 0,
        outcomes,
    )


class GeometryOutcomeOrderingTests(unittest.TestCase):
    def test_order_exactly_matches_search_admissibility_objective_and_ties(
        self,
    ) -> None:
        result = _result(_observation())
        candidates = result.candidate_set.candidates
        outcomes = [
            _outcome(candidates[0]),
            _outcome(candidates[1], score=100, gauge=49),
            _outcome(candidates[2], score=30),
            _outcome(candidates[3], score=30),
            _outcome(candidates[4], invalid=1),
            *(_outcome(value) for value in candidates[5:]),
        ]
        ordering = geometry_outcome_ordering(outcomes)

        self.assertEqual(ordering.winner_ordinal, 2)
        self.assertTrue(ordering.strictly_improved)
        self.assertNotIn(1, ordering.eligible_ordinals)
        self.assertNotIn(4, ordering.eligible_ordinals)
        self.assertIn((2, 0), ordering.preferences)
        self.assertIn((0, 1), ordering.preferences)
        self.assertIn((2, 3), ordering.preferences)
        self.assertNotIn((1, 4), ordering.preferences)
        self.assertNotIn((4, 1), ordering.preferences)

    def test_reserve_band_is_gauge_first_below_half_and_score_first_above(
        self,
    ) -> None:
        result = _result(_observation())
        candidates = result.candidate_set.candidates
        below = [
            _outcome(candidates[0], gauge=40),
            _outcome(candidates[1], score=1_000, gauge=40),
            _outcome(candidates[2], score=1, gauge=49),
        ]
        below_ordering = geometry_outcome_ordering(below)
        self.assertEqual(below_ordering.winner_ordinal, 2)

        above = [
            _outcome(candidates[0], gauge=80),
            _outcome(candidates[1], score=100, gauge=50),
            _outcome(candidates[2], score=99, gauge=90),
        ]
        above_ordering = geometry_outcome_ordering(above)
        self.assertEqual(above_ordering.winner_ordinal, 1)

    def test_alive_branch_beats_terminal_branch_when_incumbent_is_terminal(
        self,
    ) -> None:
        result = _result(_observation())
        candidates = result.candidate_set.candidates
        outcomes = [
            _outcome(candidates[0], alive=False, survival=4, gauge=20),
            _outcome(
                candidates[1],
                score=10_000,
                alive=False,
                survival=8,
                gauge=50,
            ),
            _outcome(candidates[2], alive=True, survival=8, gauge=20),
        ]
        ordering = geometry_outcome_ordering(outcomes)
        self.assertEqual(ordering.winner_ordinal, 2)

    def test_reserve_target_uses_integer_half_and_binds_gauge_max(self) -> None:
        result = _result(_observation())
        incumbent = _outcome(
            result.candidate_set.candidates[0],
            gauge=50,
            gauge_max=101,
        )
        exact = _outcome(
            result.candidate_set.candidates[1],
            score=1,
            gauge=50,
            gauge_max=101,
        )
        below = _outcome(
            result.candidate_set.candidates[2],
            score=10_000,
            gauge=49,
            gauge_max=101,
        )
        self.assertEqual(incumbent.reserve_target, 50)
        self.assertEqual(incumbent.protected_reserve, 50)
        self.assertTrue(exact.survival_nondominated_by(incumbent))
        self.assertFalse(below.survival_nondominated_by(incumbent))
        mismatched = _outcome(
            result.candidate_set.candidates[3],
            gauge=50,
            gauge_max=100,
        )
        with self.assertRaisesRegex(ValueError, "different gauge maxima"):
            mismatched.survival_nondominated_by(incumbent)

    def test_negative_terminal_gauge_remains_ordered(self) -> None:
        result = _result(_observation())
        worse = _outcome(
            result.candidate_set.candidates[0],
            gauge=-20,
            alive=False,
            survival=4,
        )
        better = _outcome(
            result.candidate_set.candidates[1],
            gauge=-1,
            alive=False,
            survival=4,
        )
        self.assertLess(worse.protected_reserve, better.protected_reserve)
        self.assertLess(worse.objective, better.objective)

    def test_equal_objective_uses_lower_fixed_slot_and_incumbent_tie(self) -> None:
        result = _result(_observation())
        outcomes = tuple(
            _outcome(candidate) for candidate in result.candidate_set.candidates
        )
        ordering = geometry_outcome_ordering(outcomes)
        self.assertEqual(ordering.winner_ordinal, 0)
        self.assertFalse(ordering.strictly_improved)
        self.assertTrue(all(winner < loser for winner, loser in ordering.preferences))


class GeometryRankingExampleTests(unittest.TestCase):
    def test_conversion_uses_all_outcomes_and_fixed_slot_availability(self) -> None:
        observation = _observation()
        result = _result(observation, winner_slot=1)
        first = geometry_ranking_example(
            observation,
            result,
            episode_identity="runway-1",
            provenance_sha256=_PROVENANCE,
        )
        second = geometry_ranking_example(
            observation,
            result,
            episode_identity="runway-1",
            provenance_sha256=_PROVENANCE,
        )

        self.assertEqual(first.winner_index, 1)
        self.assertTrue(first.improved_over_incumbent)
        self.assertEqual(int(first.available_mask.sum()), len(result.outcomes))
        self.assertEqual(len(first.outcome_sha256s), len(result.outcomes))
        self.assertGreater(len(first.preferences), len(result.outcomes))
        self.assertEqual(first.sha256, second.sha256)
        self.assertFalse(first.available_mask.flags.writeable)
        self.assertEqual(
            first.candidate_vocabulary_sha256,
            geometry_candidate_vocabulary_sha256(result.candidate_set.config),
        )

    def test_conversion_rejects_result_whose_declared_winner_disagrees(self) -> None:
        observation = _observation()
        result = _result(observation, winner_slot=1)
        inconsistent = GeometrySearchResult(
            result.candidate_set,
            result.configured_horizon_ticks,
            result.causal_horizon_ticks,
            result.candidate_set.candidates[0],
            False,
            result.outcomes,
        )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            geometry_ranking_example(
                observation,
                inconsistent,
                episode_identity="invalid",
                provenance_sha256=_PROVENANCE,
            )

    def test_conversion_accepts_full_runway_search_result(self) -> None:
        observation = _observation()
        causal = _result(observation, winner_slot=1)
        runway = RunwaySearchResult(
            teacher_identity_sha256=_PROVENANCE,
            candidate_set=causal.candidate_set,
            runway_ticks=256,
            selected_candidate=causal.selected_candidate,
            strictly_improved=causal.strictly_improved,
            outcomes=causal.outcomes,
        )
        example = geometry_ranking_example(
            observation,
            runway,
            episode_identity="full-runway",
            provenance_sha256=_PROVENANCE,
        )

        self.assertEqual(example.search_result_sha256, runway.sha256)
        self.assertEqual(example.winner_index, runway.winner_ordinal)
        self.assertEqual(len(example.outcome_sha256s), len(runway.outcomes))

    def test_unavailable_slots_are_excluded_from_listwise_normalizer(self) -> None:
        observation = _observation(source_x=4.0)
        result = _result(observation)
        example = geometry_ranking_example(
            observation,
            result,
            episode_identity="offscreen",
            provenance_sha256=_PROVENANCE,
        )
        missing = next(
            index
            for index, available in enumerate(example.available_mask)
            if not available
        )
        model = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=example.candidate_count,
            candidate_set_sha256=example.candidate_vocabulary_sha256,
            config=GeometryModelConfig(body_hidden=8, pair_hidden=8),
        )
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.selector[-1].bias[missing] = 100.0
            model.selector[-1].bias[0] = 10.0
        batch = GeometryRankingDataset((example,)).batch()
        loss = geometry_ranking_loss(model, batch)
        logits = model(
            batch.global_features,
            batch.body_features,
            batch.body_mask,
            batch.source_index,
            batch.destination_index,
        ).masked_fill(~batch.available_mask, -torch.inf)

        self.assertEqual(int(logits.argmax(dim=-1)[0]), 0)
        self.assertTrue(torch.isfinite(loss.total))
        self.assertLess(float(loss.listwise.detach()), 0.01)


class GeometryRankingTrainingTests(unittest.TestCase):
    def test_combined_ranking_loss_learns_incumbent_and_improvements(self) -> None:
        examples = []
        for gauge in (100, 200, 300, 400, 600, 700, 800, 900):
            observation = _observation(gauge)
            result = _result(
                observation, winner_slot=1 if gauge < 500 else 0
            )
            examples.append(
                geometry_ranking_example(
                    observation,
                    result,
                    episode_identity=f"runway-{gauge}",
                    provenance_sha256=_PROVENANCE,
                )
            )
        dataset = GeometryRankingDataset(examples)
        torch.manual_seed(17)
        model = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=dataset.candidate_count,
            candidate_set_sha256=dataset.candidate_vocabulary_sha256,
            config=GeometryModelConfig(body_hidden=16, pair_hidden=24),
        )
        report = train_geometry_ranker(
            model,
            dataset,
            steps=160,
            batch_size=8,
            learning_rate=1e-3,
            seed=19,
        )

        self.assertLess(report.final_loss, report.initial_loss)
        self.assertGreaterEqual(report.top1_accuracy, 0.75)
        self.assertGreaterEqual(report.incumbent_accuracy, 0.75)
        self.assertGreaterEqual(report.improved_accuracy, 0.75)
        self.assertGreaterEqual(report.pairwise_accuracy, 0.75)
        self.assertGreater(
            report.preferences,
            len(dataset) * (dataset.candidate_count - 1),
        )


if __name__ == "__main__":
    unittest.main()
