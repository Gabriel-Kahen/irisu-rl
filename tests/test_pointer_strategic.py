from __future__ import annotations

import copy
import hashlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_pointer.archive import ArchiveElite, StrategicArchive, archive_cell_key
from irisu_pointer.strategic import (
    CurriculumMetrics,
    CurriculumStage,
    StrategicIntent,
    available_intents,
    evaluate_curriculum,
    extract_strategic_features,
    potential_shaping_delta,
    strategic_potential,
)


def _body(
    identifier: int,
    *,
    color: int = 0,
    chain_id: int = 0,
    lifecycle: str = "dynamic_fresh",
    x: float = 200.0,
    y: float = 180.0,
    hits: int = 0,
    remaining: int = 1_000,
    rot_timer: int = 0,
    kind: str = "piece",
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": kind,
        "shape": "box",
        "lifecycle": lifecycle,
        "color": color,
        "x": x,
        "y": y,
        "vx": 0.0,
        "vy": 0.0,
        "size": 40.0,
        "chain_id": chain_id,
        "projectile_hits": hits,
        "age_ticks": 100,
        "remaining_lifetime": remaining,
        "rot_timer": rot_timer,
    }


def _observation(
    *,
    tick: int = 25,
    score: int = 16,
    gauge: int = 8_000,
    bodies: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "tick": tick,
        "score": score,
        "gauge": gauge,
        "gauge_max": 10_000,
        "level": 3,
        "highest_chain": 2,
        "qualifying_clear_count": 2,
        "terminated": False,
        "truncated": False,
        "difficulty": {"active_colors": 3, "spawn_interval_ticks": 100},
        "field": {"x": 130.0, "y": 120.0, "width": 320.0, "height": 250.0},
        "bodies": list(
            bodies
            or [
                _body(1, color=0, chain_id=7, lifecycle="confirmed"),
                _body(
                    2,
                    color=0,
                    chain_id=7,
                    lifecycle="confirmed",
                    x=240.0,
                ),
                _body(3, color=0, x=300.0),
                _body(4, color=1, x=320.0),
                _body(
                    5,
                    color=2,
                    lifecycle="rotten",
                    y=340.0,
                    rot_timer=41,
                ),
                _body(6, kind="bonus", lifecycle="dynamic_fresh"),
            ]
        ),
    }


class StrategicFeatureTests(unittest.TestCase):
    def test_truncated_state_is_not_archive_alive(self) -> None:
        observation = _observation()
        observation["truncated"] = True
        self.assertFalse(extract_strategic_features(observation).alive)

    def test_group_hit_budget_anchor_and_hazards_are_public(self) -> None:
        features = extract_strategic_features(_observation())
        color = features.color(0)
        assert color is not None
        self.assertEqual(color.largest_group, 2)
        self.assertEqual(color.viable_anchor_count, 1)
        self.assertEqual(features.total_deletion_hit_budget, 4)
        self.assertEqual(features.total_safe_direct_hit_budget, 2)
        self.assertEqual(features.rotten_piece_count, 1)
        self.assertEqual(features.rot_active_count, 1)
        self.assertEqual(features.floor_hazard_count, 1)
        self.assertEqual(features.bonus_count, 1)
        self.assertEqual(features.ticks_to_spawn, 75)
        self.assertAlmostEqual(features.spawn_phase, 0.25)

    def test_intents_cover_safety_anchor_and_harvest_options(self) -> None:
        observation = _observation()
        observation["bodies"] = list(observation["bodies"]) + [
            _body(7, color=2, x=280.0)
        ]
        intents = available_intents(extract_strategic_features(observation))
        self.assertEqual(intents[0], StrategicIntent.TRIAGE)
        self.assertIn(StrategicIntent.ORB_CLEAN, intents)
        self.assertIn(StrategicIntent.MATCH_ROTTEN, intents)
        self.assertIn(StrategicIntent.EXTEND_ANCHOR, intents)
        self.assertIn(StrategicIntent.HARVEST, intents)

    def test_archive_key_ignores_hidden_rng_ids_and_color_permutation(self) -> None:
        first = _observation()
        second = copy.deepcopy(first)
        second["_hidden_rng_state"] = "different"
        for body in second["bodies"]:  # type: ignore[index]
            body["id"] = int(body["id"]) + 100  # type: ignore[index]
            body["color"] = {0: 2, 1: 0, 2: 1}.get(  # type: ignore[index]
                int(body["color"]), int(body["color"])  # type: ignore[index]
            )
            if int(body["chain_id"]):  # type: ignore[index]
                body["chain_id"] = 999  # type: ignore[index]
        first_key = archive_cell_key(extract_strategic_features(first))
        second_key = archive_cell_key(extract_strategic_features(second))
        self.assertEqual(first_key, second_key)
        self.assertEqual(first_key.sha256, second_key.sha256)

    def test_potential_delta_is_duration_discounted_and_terminal_safe(self) -> None:
        before = strategic_potential(extract_strategic_features(_observation()))
        after_observation = _observation(score=64, gauge=9_000)
        after_observation["bodies"] = list(after_observation["bodies"]) + [
            _body(10, color=0, chain_id=7, lifecycle="confirmed", x=260.0)
        ]
        after = strategic_potential(extract_strategic_features(after_observation))
        self.assertGreater(after.chain_score, before.chain_score)
        self.assertAlmostEqual(
            potential_shaping_delta(before, after, gamma=0.99, duration_ticks=3),
            0.99**3 * after.total - before.total,
        )
        self.assertEqual(
            potential_shaping_delta(
                before, after, gamma=0.99, duration_ticks=3, terminal=True
            ),
            -before.total,
        )


class StrategicArchiveTests(unittest.TestCase):
    def test_raw_score_then_survival_replaces_elite(self) -> None:
        archive = StrategicArchive(source_identity="source-a")
        observation = _observation()
        first, inserted = archive.capture(
            observation, b"first-snapshot", trajectory_identity="trajectory-1"
        )
        self.assertTrue(inserted)
        worse = ArchiveElite(
            cell=first.cell,
            snapshot=b"worse",
            snapshot_sha256=hashlib.sha256(b"worse").hexdigest(),
            source_identity="source-a",
            trajectory_identity="trajectory-2",
            raw_score=first.raw_score - 1,
            survival_ticks=first.survival_ticks + 1_000,
            alive=True,
            qualifying_clears=100,
            highest_chain=10,
            gauge=10_000,
        )
        self.assertFalse(archive.consider(worse))
        better = ArchiveElite(
            cell=first.cell,
            snapshot=b"better",
            snapshot_sha256=hashlib.sha256(b"better").hexdigest(),
            source_identity="source-a",
            trajectory_identity="trajectory-3",
            raw_score=first.raw_score,
            survival_ticks=first.survival_ticks + 1,
            alive=True,
            qualifying_clears=first.qualifying_clears,
            highest_chain=first.highest_chain,
            gauge=first.gauge,
        )
        self.assertTrue(archive.consider(better))
        self.assertIs(archive.get(first.cell), better)

    def test_snapshot_and_source_identities_fail_closed(self) -> None:
        archive = StrategicArchive(source_identity="source-a")
        elite = ArchiveElite.create(
            _observation(),
            b"snapshot",
            source_identity="source-b",
            trajectory_identity="trajectory",
        )
        with self.assertRaisesRegex(ValueError, "source identity"):
            archive.consider(elite)
        with self.assertRaisesRegex(ValueError, "snapshot identity"):
            ArchiveElite(
                cell=elite.cell,
                snapshot=b"snapshot",
                snapshot_sha256="0" * 64,
                source_identity="source-a",
                trajectory_identity="trajectory",
                raw_score=0,
                survival_ticks=0,
                alive=True,
                qualifying_clears=0,
                highest_chain=0,
                gauge=0,
            )

    def test_manifest_identity_is_insertion_order_independent(self) -> None:
        observations = [_observation(score=16), _observation(score=1_000)]
        forward = StrategicArchive(source_identity="source-a")
        reverse = StrategicArchive(source_identity="source-a")
        for index, observation in enumerate(observations):
            forward.capture(
                observation,
                f"snapshot-{index}".encode(),
                trajectory_identity=f"trajectory-{index}",
            )
        for index in reversed(range(len(observations))):
            reverse.capture(
                observations[index],
                f"snapshot-{index}".encode(),
                trajectory_identity=f"trajectory-{index}",
            )
        self.assertEqual(forward.manifest(), reverse.manifest())
        self.assertEqual(forward.sha256, reverse.sha256)


class CurriculumGateTests(unittest.TestCase):
    def test_default_curriculum_passes_strong_learning_evidence(self) -> None:
        metrics = CurriculumMetrics(
            episodes=16,
            shots=200,
            projectile_hits=190,
            chain_joins=60,
            qualifying_clears=30,
            raw_scores=(320,) * 16,
            survival_ticks=(2_000,) * 16,
            highest_chains=(4,) * 16,
            gauge_failures=1,
            baseline_median_score=100.0,
            baseline_median_survival=2_000.0,
        )
        results = evaluate_curriculum(metrics)
        self.assertTrue(all(result.passed for result in results))

    def test_failed_micro_control_blocks_later_stages(self) -> None:
        metrics = CurriculumMetrics(
            episodes=16,
            shots=200,
            projectile_hits=100,
            chain_joins=60,
            qualifying_clears=30,
            raw_scores=(1_000,) * 16,
            survival_ticks=(3_000,) * 16,
            highest_chains=(8,) * 16,
            baseline_median_score=100.0,
            baseline_median_survival=2_000.0,
        )
        results = evaluate_curriculum(metrics)
        self.assertEqual(results[0].stage, CurriculumStage.MICRO_CONTROL)
        self.assertEqual(results[0].failed_metrics, ("hit_rate",))
        self.assertEqual(results[1].failed_metrics, ("prerequisite_stage",))


if __name__ == "__main__":
    unittest.main()
