from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import Any

from irisu_env import Action, EventKind, IrisuEnv, NativeError, find_library
from irisu_pointer.action import PointerActionSpec
from irisu_pointer.evaluate import (
    ArtifactBinding,
    DevelopmentEvaluationError,
    DevelopmentSuite,
    PromotionCriteria,
    evaluate_full_games,
    identity_sha256,
)
from irisu_pointer.experts import PointerExpertDecision
from irisu_rl.schema import TEACHER_V1


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _Policy:
    schema_sha256 = TEACHER_V1.sha256
    pointer_action_sha256 = PointerActionSpec().sha256

    def __init__(self, artifact_sha256: str, *, native: bool = False) -> None:
        self.artifact_sha256 = artifact_sha256
        self.native = native
        self.seed = -1

    def reset(self, seed: int = 0) -> None:
        self.seed = seed

    def act(self, observation: dict[str, Any]):
        del observation
        return Action.wait(1) if self.native else PointerExpertDecision.wait(1)


class _MacroPolicy(_Policy):
    def act(self, observation: dict[str, Any]):
        del observation
        return (Action.wait(1), Action.wait(1))


class _FakeEnv:
    physics_backend = "portable"

    def __init__(
        self,
        runtime: Path,
        config: dict[str, Any],
        runner_identity: dict[str, Any],
        *,
        terminal: bool = True,
    ) -> None:
        self.library_path = str(runtime)
        self.config = config
        self.identity = runner_identity
        self.make_terminal = terminal
        self.seed = 0
        self.steps = 0

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def runner_identity_manifest(self) -> dict[str, Any]:
        return dict(self.identity)

    def _observation(self, terminal: bool = False) -> dict[str, Any]:
        return {
            "tick": self.steps,
            "score": self.seed * 10 if terminal else 0,
            "highest_chain": min(self.steps, 2),
            "terminated": terminal,
            "truncated": False,
            "difficulty": {"spawn_interval_ticks": 100},
            "bodies": [],
        }

    def reset(self, *, seed: int):
        self.seed = seed
        self.steps = 0
        return self._observation(), {"seed": seed, "config_hash": 123}

    def step(self, action: Action):
        del action
        self.steps += 1
        events: list[dict[str, int]] = []
        if self.steps == 1:
            events.extend(
                {
                    "kind": int(EventKind.PROJECTILE_HIT),
                    "a": 100,
                    "b": 1,
                    "value": 0,
                }
                for _ in range(50)
            )
            events.append(
                {
                    "kind": int(EventKind.CHAIN_JOINED),
                    "a": 1,
                    "b": 2,
                    "value": 2,
                }
            )
        elif self.steps == 2:
            events.extend(
                (
                    {
                        "kind": int(EventKind.PROJECTILE_HIT),
                        "a": 100,
                        "b": 2,
                        "value": 0,
                    },
                    {
                        "kind": int(EventKind.CLEARED),
                        "a": 1,
                        "b": 2,
                        "value": 2,
                    },
                )
            )
        elif self.steps == 3:
            events.append(
                {
                    "kind": int(EventKind.ROTTEN),
                    "a": 3,
                    "b": 0,
                    "value": 0,
                }
            )
        terminal = self.make_terminal and self.steps == 3
        return (
            self._observation(terminal),
            0,
            terminal,
            False,
            {"events": events, "invalid_action": False},
        )


class FullGameEvaluatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="irisu-dev-")
        root = Path(self.temp.name)
        self.policy_path = root / "candidate.bin"
        self.runtime_path = root / "portable.so"
        self.policy_path.write_bytes(b"policy-v1")
        self.runtime_path.write_bytes(b"runtime-v1")
        self.runner_identity = {
            "version": "fake-runner-v1",
            "physics_backend": "portable",
            "config_hash": 123,
        }
        self.binding = ArtifactBinding(
            "development-candidate-v1",
            self.policy_path,
            _sha256(self.policy_path),
            self.runtime_path,
            _sha256(self.runtime_path),
            identity_sha256(self.runner_identity),
        )
        self.suite = DevelopmentSuite(
            label="unseen-development-mini-v1",
            seeds=(1, 2, 3),
            config=(("max_episode_ticks", 3),),
            max_decisions_per_episode=10,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _factory(self, *, terminal: bool = True):
        return lambda runtime, config: _FakeEnv(
            runtime,
            dict(config),
            self.runner_identity,
            terminal=terminal,
        )

    def test_complete_games_deduplicate_hits_and_summarize_deterministically(
        self,
    ) -> None:
        report = evaluate_full_games(
            lambda: _Policy(self.binding.policy_sha256),
            self.binding,
            suite=self.suite,
            criteria=PromotionCriteria(
                minimum_median_raw_score=20,
                minimum_p10_raw_score=12,
                minimum_median_survival_ticks=3,
                minimum_median_unique_hit_pairs=2,
                minimum_completion_rate=1,
                maximum_invalid_actions=0,
            ),
            env_factory=self._factory(),
        )
        self.assertTrue(report.promoted)
        self.assertEqual([value.raw_score for value in report.episodes], [10, 20, 30])
        for outcome in report.episodes:
            self.assertEqual(outcome.survival_ticks, 3)
            self.assertEqual(outcome.decisions, 3)
            self.assertEqual(outcome.valid_actions, 3)
            self.assertEqual(outcome.invalid_actions, 0)
            self.assertEqual(outcome.projectile_hit_events, 51)
            self.assertEqual(outcome.unique_projectile_hit_pairs, 2)
            self.assertEqual(outcome.unique_projectiles_hitting, 1)
            self.assertEqual(outcome.unique_bodies_hit, 2)
            self.assertEqual(outcome.chain_joined, 1)
            self.assertEqual(outcome.cleared, 1)
            self.assertEqual(outcome.rotten, 1)
            self.assertEqual(outcome.max_chain, 2)
        self.assertEqual(report.aggregate["raw_score"]["p10"], 12.0)
        self.assertEqual(report.aggregate["raw_score"]["median"], 20.0)
        self.assertEqual(report.aggregate["total_projectile_hit_events"], 153)
        self.assertEqual(report.aggregate["total_unique_projectile_hit_pairs"], 6)
        manifest = report.manifest()
        self.assertFalse(manifest["sealed_test_material_used"])
        self.assertTrue(manifest["gates"]["all_passed"])

    def test_actor_style_primitive_macro_is_executed_in_order(self) -> None:
        report = evaluate_full_games(
            lambda: _MacroPolicy(self.binding.policy_sha256),
            self.binding,
            suite=self.suite,
            env_factory=self._factory(),
        )
        self.assertEqual(
            [value.primitive_actions for value in report.episodes],
            [3, 3, 3],
        )
        self.assertEqual(
            [value.decisions for value in report.episodes],
            [2, 2, 2],
        )

    def test_identity_mismatches_fail_closed(self) -> None:
        wrong_hash = "0" * 64
        bad_policy = ArtifactBinding(
            "development-candidate-v1",
            self.policy_path,
            wrong_hash,
            self.runtime_path,
            _sha256(self.runtime_path),
            identity_sha256(self.runner_identity),
        )
        with self.assertRaisesRegex(DevelopmentEvaluationError, "policy.*mismatch"):
            evaluate_full_games(
                lambda: _Policy(wrong_hash),
                bad_policy,
                suite=self.suite,
                env_factory=self._factory(),
            )

        bad_runtime = ArtifactBinding(
            "development-candidate-v1",
            self.policy_path,
            _sha256(self.policy_path),
            self.runtime_path,
            wrong_hash,
            identity_sha256(self.runner_identity),
        )
        with self.assertRaisesRegex(DevelopmentEvaluationError, "runtime.*mismatch"):
            evaluate_full_games(
                lambda: _Policy(bad_runtime.policy_sha256),
                bad_runtime,
                suite=self.suite,
                env_factory=self._factory(),
            )

        bad_runner = ArtifactBinding(
            "development-candidate-v1",
            self.policy_path,
            _sha256(self.policy_path),
            self.runtime_path,
            _sha256(self.runtime_path),
            "1" * 64,
        )
        with self.assertRaisesRegex(DevelopmentEvaluationError, "runner provenance"):
            evaluate_full_games(
                lambda: _Policy(bad_runner.policy_sha256),
                bad_runner,
                suite=self.suite,
                env_factory=self._factory(),
            )

        reset_mismatch_identity = {**self.runner_identity, "config_hash": 124}
        reset_mismatch = ArtifactBinding(
            "development-candidate-v1",
            self.policy_path,
            _sha256(self.policy_path),
            self.runtime_path,
            _sha256(self.runtime_path),
            identity_sha256(reset_mismatch_identity),
        )
        with self.assertRaisesRegex(
            DevelopmentEvaluationError, "reset config identity"
        ):
            evaluate_full_games(
                lambda: _Policy(reset_mismatch.policy_sha256),
                reset_mismatch,
                suite=self.suite,
                env_factory=lambda runtime, config: _FakeEnv(
                    runtime, dict(config), reset_mismatch_identity
                ),
            )

        with self.assertRaisesRegex(DevelopmentEvaluationError, "loaded policy"):
            evaluate_full_games(
                lambda: _Policy("2" * 64),
                self.binding,
                suite=self.suite,
                env_factory=self._factory(),
            )

    def test_sealed_and_test_labels_or_paths_are_rejected(self) -> None:
        for label in ("sealed-suite-v1", "heldout-test-v1"):
            with self.subTest(label=label), self.assertRaisesRegex(
                ValueError, "sealed or test"
            ):
                DevelopmentSuite(label=label)
        forbidden = Path(self.temp.name) / "sealed-model.bin"
        forbidden.write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "sealed or test"):
            ArtifactBinding(
                "development-candidate-v1",
                forbidden,
                _sha256(forbidden),
                self.runtime_path,
                _sha256(self.runtime_path),
                identity_sha256(self.runner_identity),
            )

    def test_nonterminal_decision_budget_fails_instead_of_reporting_partial_game(
        self,
    ) -> None:
        suite = DevelopmentSuite(
            label="unseen-development-budget-v1",
            seeds=(7,),
            max_decisions_per_episode=2,
        )
        with self.assertRaisesRegex(
            DevelopmentEvaluationError, "complete-game decision budget"
        ):
            evaluate_full_games(
                lambda: _Policy(self.binding.policy_sha256),
                self.binding,
                suite=suite,
                env_factory=self._factory(terminal=False),
            )


class RealPortableSmokeTests(unittest.TestCase):
    def test_real_portable_runtime_completes_short_development_game(self) -> None:
        try:
            runtime = Path(find_library()).resolve()
        except NativeError:
            self.skipTest("portable runtime is unavailable")
        config = {"max_episode_ticks": 5}
        with IrisuEnv(
            library_path=runtime,
            physics_backend="portable",
            config=config,
        ) as env:
            runner_sha256 = identity_sha256(env.runner_identity_manifest())
        with tempfile.TemporaryDirectory(prefix="irisu-dev-real-") as directory:
            policy_path = Path(directory) / "candidate.bin"
            policy_path.write_bytes(b"native-wait-policy-v1")
            binding = ArtifactBinding(
                "real-portable-development-smoke-v1",
                policy_path,
                _sha256(policy_path),
                runtime,
                _sha256(runtime),
                runner_sha256,
            )
            report = evaluate_full_games(
                lambda: _Policy(binding.policy_sha256, native=True),
                binding,
                suite=DevelopmentSuite(
                    label="real-portable-unseen-development-smoke-v1",
                    seeds=(123,),
                    config=(("max_episode_ticks", 5),),
                    max_decisions_per_episode=10,
                ),
                criteria=PromotionCriteria(
                    minimum_median_raw_score=0,
                    minimum_p10_raw_score=0,
                    minimum_median_survival_ticks=5,
                    minimum_median_unique_hit_pairs=0,
                    minimum_completion_rate=1,
                    maximum_invalid_actions=0,
                ),
            )
        self.assertTrue(report.promoted)
        self.assertEqual(report.episodes[0].survival_ticks, 5)
        self.assertEqual(report.episodes[0].invalid_actions, 0)


if __name__ == "__main__":
    unittest.main()
