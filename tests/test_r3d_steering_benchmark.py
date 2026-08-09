from __future__ import annotations

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

from irisu_env import ActionKind
from irisu_pointer.archive import StrategicArchive
from irisu_pointer.replay_supervision import SteeringConversionMetrics
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_rl.actions import SemanticAction


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks/rl_r3d_steering.py"
SPEC = importlib.util.spec_from_file_location(
    "rl_r3d_steering_benchmark", BENCHMARK_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def _episode(*, qualifying_clears: int, event_clears: int):
    conversion = SteeringConversionMetrics(
        frames=2_000,
        survival_ticks=2_000,
        requested_shots=20,
        shots_fired=20,
        shots_hit=20,
        projectile_hit_events=20,
        chain_joins=5,
        clears=event_clears,
        rotten=0,
        ejected=0,
        invalid_actions=0,
        final_score=100,
        terminated=False,
        truncated=True,
    )
    return BENCHMARK.EpisodeResult(
        policy="fixture",
        seed=1,
        decisions=20,
        primitive_actions=20,
        highest_chain=3,
        qualifying_clears=qualifying_clears,
        final_level=3,
        final_gauge=500,
        gauge_max=1_000,
        archive_insertions=0,
        archive_rejections=0,
        conversion=conversion,
    )


def _observation(tick: int, *, truncated: bool = False):
    body = lambda identifier, x: {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": "dynamic_fresh",
        "color": 0,
        "x": x,
        "y": 200.0,
        "vx": 0.0,
        "vy": 0.0,
        "size": 40.0,
        "chain_id": 0,
        "projectile_hits": 0,
        "remaining_lifetime": 1_000,
        "rot_timer": 0,
    }
    return {
        "tick": tick,
        "score": 0,
        "gauge": 1_000,
        "gauge_max": 1_000,
        "level": 1,
        "highest_chain": 0,
        "qualifying_clear_count": 0,
        "terminated": False,
        "truncated": truncated,
        "field": {"x": 0.0, "y": 0.0, "width": 640.0, "height": 480.0},
        "difficulty": {"active_colors": 2, "spawn_interval_ticks": 100},
        "bodies": (body(1, 180.0), body(2, 280.0)),
    }


class _BoundaryEnv:
    def __init__(self) -> None:
        self.tick = 0
        self.clone_ticks: list[int] = []

    def reset(self, *, seed: int):
        self.tick = 0
        return _observation(0), {"config_hash": 17, "seed": seed}

    def step(self, action):
        duration = (
            int(action.wait_ticks)
            if ActionKind.parse(action.kind) is ActionKind.WAIT
            else 1
        )
        self.tick += duration
        truncated = self.tick >= 10
        return (
            _observation(self.tick, truncated=truncated),
            0,
            False,
            truncated,
            {"config_hash": 17, "events": (), "invalid_action": False},
        )

    def clone_state(self) -> bytes:
        self.clone_ticks.append(self.tick)
        return f"snapshot:{self.tick}".encode()


class _WaitThenShotPolicy:
    def reset(self, seed: int) -> None:
        del seed

    def predict(self, observation):
        if int(observation["tick"]) < 8:
            return SteeringDecision(
                SemanticAction.wait(8),
                SteeringIntent.WAIT,
                reason="synthetic cooldown",
            )
        return SteeringDecision(
            SemanticAction.strong(0.25, 0.5),
            SteeringIntent.STEER_MATCH,
            source_body_id=1,
            destination_body_id=2,
            reason="synthetic safe-boundary shot",
        )


class R3dSteeringBenchmarkTests(unittest.TestCase):
    def test_survival_holdout_suite_is_fixed_unique_and_disjoint(self) -> None:
        derived = tuple(
            int.from_bytes(
                hashlib.sha256(
                    f"irisu-r3d-survival-holdout-v1:{index}".encode()
                ).digest()[:4],
                "big",
            )
            for index in range(16)
        )
        self.assertEqual(derived, BENCHMARK.SURVIVAL_HOLDOUT_SEEDS)
        self.assertEqual(len(set(derived)), 16)
        self.assertFalse(
            set(derived)
            & (
                set(BENCHMARK.DEMONSTRATION_SEEDS)
                | set(BENCHMARK.UNSEEN_DEVELOPMENT_SEEDS)
            )
        )

    def test_extended_demonstration_seeds_are_reproducible(self) -> None:
        derived = tuple(
            int.from_bytes(
                hashlib.sha256(
                    (
                        "irisu-r3d-survival-demonstration-v2:"
                        f"{index}"
                    ).encode()
                ).digest()[:4],
                "big",
            )
            for index in range(16)
        )
        self.assertEqual(derived, BENCHMARK.DEMONSTRATION_SEEDS[8:])
        self.assertEqual(len(set(BENCHMARK.DEMONSTRATION_SEEDS)), 24)
        self.assertFalse(
            set(BENCHMARK.DEMONSTRATION_SEEDS)
            & (
                set(BENCHMARK.UNSEEN_DEVELOPMENT_SEEDS)
                | set(BENCHMARK.SURVIVAL_HOLDOUT_SEEDS)
            )
        )

    def test_episode_decision_budget_fails_at_declared_boundary(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "decision budget"):
            BENCHMARK._run_episode(
                _BoundaryEnv(),
                _WaitThenShotPolicy(),
                label="fixture",
                seed=1,
                config_hash=17,
                max_decisions=1,
            )

    def test_archive_capture_occurs_only_at_pre_shot_boundary(self) -> None:
        env = _BoundaryEnv()
        archive = StrategicArchive(source_identity="fixture-source")
        BENCHMARK._run_episode(
            env,
            _WaitThenShotPolicy(),
            label="fixture",
            seed=1,
            config_hash=17,
            archive=archive,
            archive_stride_ticks=1,
        )
        self.assertEqual(env.clone_ticks, [8])
        elites = tuple(archive)
        self.assertEqual(len(elites), 1)
        self.assertTrue(elites[0].trajectory_identity.endswith(":8:pre-shot"))

    def test_clear_events_cannot_substitute_for_qualifying_clears(self) -> None:
        baseline_episodes = tuple(
            _episode(qualifying_clears=0, event_clears=40)
            for _ in range(8)
        )
        baseline = BENCHMARK._aggregate(baseline_episodes)
        curriculum = BENCHMARK._curriculum(baseline_episodes, baseline)
        self.assertEqual(curriculum["metrics"]["qualifying_clears"], 0)
        self.assertEqual(
            curriculum["metrics"]["clears_per_shot"],
            0.0,
        )
        self.assertEqual(
            curriculum["metrics"]["qualifying_clears_per_shot"],
            0.0,
        )
        self.assertTrue(curriculum["gates"][0]["passed"])
        self.assertIn(
            "clears_per_shot",
            curriculum["gates"][1]["failed_metrics"],
        )
        self.assertEqual(
            baseline["conversion"]["clears"],
            320,
        )
        self.assertEqual(
            baseline["conversion"]["cleared_events_per_shot"],
            2.0,
        )

    def test_symlinked_parent_cannot_reach_canonical_run_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protected = root / "artifacts/r3/runs/run"
            protected.mkdir(parents=True)
            source = protected / "input.toml"
            source.write_text("version = 1\n", encoding="utf-8")
            link = root / "development-link"
            link.symlink_to(protected, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "canonical R3"):
                BENCHMARK._snapshot_file(link / source.name, "fixture input")
            with self.assertRaisesRegex(ValueError, "canonical R3"):
                BENCHMARK._output_path(
                    link / "output.pt", "fixture output", ".pt"
                )


if __name__ == "__main__":
    unittest.main()
