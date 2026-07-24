from __future__ import annotations

import hashlib
import json
import struct
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from irisu_env import Action
from irisu_rl.actions import ActionSpec
from irisu_rl.curriculum import SnapshotBlobStore, SnapshotLibrary
from irisu_rl.encoding import ActorTrackEncoder, TeacherStateEncoder
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.r3b_evaluation import (
    CrossBackendCellPair,
    CrossBackendEvaluationManifest,
    DeploymentPolicyIdentity,
    EpisodeMetrics,
    EvaluationReport,
    EvaluationSuite,
    LearnedPolicyBackendParityArtifact,
    ScriptedBaselineSpec,
    RecurrentSemanticPolicy,
    behavior_build_identity_manifest,
    behavior_build_identity_sha256,
    build_baseline_evidence,
    encoder_instance_manifest,
    evaluate_recurrent_policy,
    evaluate_scripted_baseline,
    semantic_from_native,
)
from irisu_rl.schema import TEACHER_V1
from tests.test_rl_vector_adapter import observation

import torch
from tests.test_r3b_snapshot_initializer import (
    _CONFIG,
    _CONFIG_HASH,
    _RUNTIME_SHA256,
    FakeRuntimeLane,
    _fixture,
)


_SNAPSHOT = struct.Struct("<qqqQ")


def evaluation_identity(*, execution: str = "c"):
    return {"execution_identity_sha256": execution * 64}


class FakeSingleSimulator:
    library_path = FakeRuntimeLane.library_path
    build_info = FakeRuntimeLane.build_info

    def __init__(self) -> None:
        self.tick = 0
        self.score = 0
        self.gauge = 100
        self.hash = 0

    def _observation(self):
        return {
            "tick": self.tick,
            "score": self.score,
            "gauge": self.gauge,
            "gauge_max": 1000,
            "bodies": (),
        }

    def restore_state(self, snapshot: bytes):
        self.tick, self.score, self.gauge, self.hash = _SNAPSHOT.unpack(snapshot)
        return self._observation()

    def state_hash(self) -> int:
        return int(self.hash)

    def config_hash(self) -> int:
        return _CONFIG_HASH

    def config(self):
        return dict(_CONFIG)

    def step(self, action: Action):
        delta = action.wait_ticks if int(action.kind) == 0 else 1
        self.tick += int(delta)
        self.score += int(delta)
        self.gauge = max(0, self.gauge - int(delta))
        self.hash += int(delta)
        return self._observation(), int(delta), False, False, {"invalid_action": False}


class FakeTerminalUnderflowSimulator(FakeSingleSimulator):
    def restore_state(self, snapshot: bytes):
        self.steps = 0
        return super().restore_state(snapshot)

    def step(self, action: Action):
        self.steps += 1
        self.tick += 1
        self.score += 1
        self.gauge = -48 if self.steps == 1 else -96
        return (
            self._observation(),
            1,
            self.steps == 2,
            False,
            {"invalid_action": False},
        )


class FakeHorizonUnderflowSimulator(FakeSingleSimulator):
    def step(self, action: Action):
        self.tick += 1
        self.score += 1
        self.gauge = -48
        return self._observation(), 1, False, False, {"invalid_action": False}


class ConfiguredTeacherEncoder(TeacherStateEncoder):
    def __init__(self, scale: int) -> None:
        self.scale = scale


class R3BEvaluationTests(unittest.TestCase):
    def test_episode_metrics_preserve_signed_final_gauge(self) -> None:
        manifest = {
            "snapshot_id": "r3b-source-v1-exact-calibration-0031",
            "repetition": 0,
            "policy_seed": 1533497378338680030,
            "initial_score": 0,
            "final_score": 16,
            "raw_score": 16,
            "elapsed_ticks": 1877,
            "decisions": 29,
            "terminated": True,
            "truncated": False,
            "invalid_actions": 0,
            "minimum_gauge": -1819,
            "final_gauge": -1819,
        }
        episode = EpisodeMetrics.from_manifest(manifest)
        self.assertEqual(episode.final_gauge, -1819)
        with self.assertRaisesRegex(ValueError, "malformed"):
            replace(
                episode,
                minimum_gauge=-(2**63) - 1,
                final_gauge=-(2**63) - 1,
            )

    def test_build_identity_binds_source_dependencies_runtime_and_configuration(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "package"
            source.mkdir()
            module = source / "helper.py"
            module.write_text("VALUE = 1\n", encoding="utf-8")
            lock = root / "uv.lock"
            project = root / "pyproject.toml"
            lock.write_text("version = 1\n", encoding="utf-8")
            project.write_text("[project]\nname = 'fixture'\n", encoding="utf-8")
            inputs = {"pyproject.toml": project, "uv.lock": lock}
            roots = {"fixture": source}

            first = behavior_build_identity_sha256(
                {"mode": "train"}, source_roots=roots, dependency_inputs=inputs
            )
            module.write_text("VALUE = 2\n", encoding="utf-8")
            changed_source = behavior_build_identity_sha256(
                {"mode": "train"}, source_roots=roots, dependency_inputs=inputs
            )
            changed_config = behavior_build_identity_sha256(
                {"mode": "evaluate"}, source_roots=roots, dependency_inputs=inputs
            )
            lock.write_text("version = 2\n", encoding="utf-8")
            changed_dependency = behavior_build_identity_sha256(
                {"mode": "evaluate"}, source_roots=roots, dependency_inputs=inputs
            )
            manifest = behavior_build_identity_manifest(
                {"mode": "evaluate"},
                source_roots=roots,
                dependency_inputs=inputs,
            )

        self.assertNotEqual(first, changed_source)
        self.assertNotEqual(changed_source, changed_config)
        self.assertNotEqual(changed_config, changed_dependency)
        self.assertEqual(manifest["version"], "r3b-behavior-build-identity-v1")
        self.assertIn("torch", manifest["runtime"])
        self.assertIn("algorithms_enabled", manifest["deterministic_settings"])

    def test_deployment_identity_binds_encoder_instance_and_model_inference(
        self,
    ) -> None:
        torch.manual_seed(17)
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1, critic_condition_features=1),
        )
        kind = torch.ones((1, 3), dtype=torch.bool)
        wait = torch.ones((1, len(ActionSpec().wait_choices)), dtype=torch.bool)
        first_encoder = ConfiguredTeacherEncoder(1)
        second_encoder = ConfiguredTeacherEncoder(2)

        first = DeploymentPolicyIdentity.from_components(
            model, first_encoder, kind, wait
        )
        second = DeploymentPolicyIdentity.from_components(
            model, second_encoder, kind, wait
        )

        self.assertNotEqual(
            encoder_instance_manifest(first_encoder),
            encoder_instance_manifest(second_encoder),
        )
        self.assertNotEqual(
            first.encoder_identity_sha256, second.encoder_identity_sha256
        )
        self.assertNotEqual(first.inference_build_sha256, second.inference_build_sha256)
        self.assertNotEqual(first.sha256, second.sha256)
        self.assertEqual(first.version, "r3b-deployment-policy-v2")
        self.assertNotEqual(first.model_manifest_sha256, "0" * 64)

    def test_fixed_cells_use_raw_score_delta_and_macro_bounds(self) -> None:
        spec, blobs = _fixture()
        store = SnapshotBlobStore(spec.library, blobs)
        suite = EvaluationSuite(
            "validation-v1",
            "validation",
            ("validation",),
            2,
            17,
            3,
            3,
            _RUNTIME_SHA256,
            spec.assignment_sha256,
            store.library.sha256,
            store.sha256,
            ActionSpec().sha256,
            (spec.library["validation"].sha256,),
        )
        portable_recipe = spec.library["validation"]
        exact_recipe = replace(
            portable_recipe,
            snapshot_id="exact-validation",
            environment_pool="exact-validation",
            expected_state_hash=portable_recipe.expected_state_hash + 1,
            snapshot_sha256="c" * 64,
            runtime_identity_sha256="b" * 64,
        )
        logical_manifest = CrossBackendEvaluationManifest(
            (CrossBackendCellPair.from_recipes(portable_recipe, exact_recipe),)
        )
        self.assertNotEqual(
            portable_recipe.expected_state_hash, exact_recipe.expected_state_hash
        )
        exact_library = SnapshotLibrary((exact_recipe,))
        logical_ids = tuple(pair.logical_cell.sha256 for pair in logical_manifest.pairs)
        suite = replace(
            suite,
            logical_cell_ids=logical_ids,
            logical_manifest_sha256=logical_manifest.sha256,
        )
        baseline = ScriptedBaselineSpec("no_action_long_wait", (("wait_ticks", 1),))
        report = evaluate_scripted_baseline(
            FakeSingleSimulator(),
            store,
            suite,
            baseline,
            evaluator_sha256="e" * 64,
            expected_assignment_sha256=spec.assignment_sha256,
            **evaluation_identity(),
        )
        self.assertEqual(len(report.episodes), 2)
        self.assertEqual([value.raw_score for value in report.episodes], [3, 3])
        self.assertTrue(all(value.truncated for value in report.episodes))
        self.assertTrue(all(value.invalid_actions == 0 for value in report.episodes))
        self.assertNotEqual(report.sha256, "0" * 64)
        replay = evaluate_scripted_baseline(
            FakeSingleSimulator(),
            store,
            suite,
            baseline,
            evaluator_sha256="e" * 64,
            expected_assignment_sha256=spec.assignment_sha256,
            **evaluation_identity(execution="d"),
        )
        self.assertEqual(
            report.episode_content_sha256,
            replay.episode_content_sha256,
        )
        exact_suite = replace(
            suite,
            suite_id="validation-exact-v1",
            snapshot_ids=("exact-validation",),
            recipe_sha256s=(exact_recipe.sha256,),
            runtime_identity_sha256="b" * 64,
            library_sha256=exact_library.sha256,
            snapshot_store_sha256="d" * 64,
            assignment_sha256="9" * 64,
            backend="exact",
        )
        exact = EvaluationReport(
            exact_suite.sha256,
            baseline.sha256,
            "e" * 64,
            exact_suite.runtime_identity_sha256,
            "f" * 64,
            tuple(
                replace(
                    value,
                    snapshot_id="exact-validation",
                    final_score=value.final_score + 1,
                    raw_score=value.raw_score + 1,
                )
                for value in report.episodes
            ),
        )
        exact_replay = replace(exact, execution_identity_sha256="a" * 64)
        parity = LearnedPolicyBackendParityArtifact(
            suite,
            report,
            exact_suite,
            exact,
            logical_manifest,
            store.library,
            exact_library,
        )
        self.assertEqual(
            parity.cross_backend_diagnostics[0]["exact_minus_portable"]["raw_score"],
            1,
        )
        self.assertEqual(parity.version, "r3b-learned-policy-backend-parity-v2")
        evidence = build_baseline_evidence(
            baseline,
            exact_suite,
            exact,
            exact_replay,
            suite,
            report,
            logical_manifest,
            store.library,
            exact_library,
        )
        self.assertEqual(evidence.episodes, 2)
        self.assertEqual(evidence.invalid_actions, 0)
        canonical = json.dumps(
            report.manifest(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        self.assertEqual(report.sha256, hashlib.sha256(canonical).hexdigest())
        with self.assertRaisesRegex(ValueError, "canonical provenance"):
            CrossBackendCellPair.from_recipes(
                portable_recipe,
                replace(exact_recipe, reset_seed=exact_recipe.reset_seed + 1),
            )
        with self.assertRaisesRegex(ValueError, "shared recipe provenance"):
            forged_suite = replace(exact_suite, logical_cell_ids=("forged",))
            build_baseline_evidence(
                baseline,
                forged_suite,
                replace(exact, suite_sha256=forged_suite.sha256),
                replace(exact_replay, suite_sha256=forged_suite.sha256),
                suite,
                report,
                logical_manifest,
                store.library,
                exact_library,
            )
        forged_portable_recipe = replace(
            portable_recipe,
            reset_seed=portable_recipe.reset_seed + 1,
            snapshot_sha256="1" * 64,
        )
        forged_exact_recipe = replace(
            exact_recipe,
            reset_seed=exact_recipe.reset_seed + 1,
            snapshot_sha256="2" * 64,
        )
        forged_manifest = CrossBackendEvaluationManifest(
            (
                CrossBackendCellPair.from_recipes(
                    forged_portable_recipe, forged_exact_recipe
                ),
            )
        )
        forged_portable_library = SnapshotLibrary((forged_portable_recipe,))
        forged_exact_library = SnapshotLibrary((forged_exact_recipe,))
        forged_logical_ids = (forged_manifest.pairs[0].logical_cell.sha256,)
        forged_portable_suite = replace(
            suite,
            recipe_sha256s=(forged_portable_recipe.sha256,),
            logical_cell_ids=forged_logical_ids,
            logical_manifest_sha256=forged_manifest.sha256,
            library_sha256=forged_portable_library.sha256,
        )
        forged_exact_suite = replace(
            exact_suite,
            recipe_sha256s=(forged_exact_recipe.sha256,),
            logical_cell_ids=forged_logical_ids,
            logical_manifest_sha256=forged_manifest.sha256,
            library_sha256=forged_exact_library.sha256,
        )
        with self.assertRaisesRegex(ValueError, "shared recipe provenance"):
            build_baseline_evidence(
                baseline,
                forged_exact_suite,
                replace(exact, suite_sha256=forged_exact_suite.sha256),
                replace(exact_replay, suite_sha256=forged_exact_suite.sha256),
                forged_portable_suite,
                replace(report, suite_sha256=forged_portable_suite.sha256),
                forged_manifest,
                store.library,
                exact_library,
            )
        with self.assertRaisesRegex(ValueError, "assignment identity"):
            evaluate_scripted_baseline(
                FakeSingleSimulator(),
                store,
                replace(suite, assignment_sha256="f" * 64),
                baseline,
                evaluator_sha256="e" * 64,
                expected_assignment_sha256=spec.assignment_sha256,
                **evaluation_identity(),
            )
        with self.assertRaisesRegex(ValueError, "snapshot-store identity"):
            evaluate_scripted_baseline(
                FakeSingleSimulator(),
                store,
                replace(suite, snapshot_store_sha256="f" * 64),
                baseline,
                evaluator_sha256="e" * 64,
                expected_assignment_sha256=spec.assignment_sha256,
                **evaluation_identity(),
            )

    def test_split_runtime_and_simultaneous_action_fail_closed(self) -> None:
        spec, blobs = _fixture()
        store = SnapshotBlobStore(spec.library, blobs)
        wrong_split = EvaluationSuite(
            "wrong-split-v1",
            "test",
            ("validation",),
            1,
            19,
            1,
            1,
            _RUNTIME_SHA256,
            spec.assignment_sha256,
            store.library.sha256,
            store.sha256,
            ActionSpec().sha256,
            (spec.library["validation"].sha256,),
        )
        with self.assertRaisesRegex(ValueError, "wrong split"):
            evaluate_scripted_baseline(
                FakeSingleSimulator(),
                store,
                wrong_split,
                ScriptedBaselineSpec("matcher_shot_policy"),
                evaluator_sha256="e" * 64,
                expected_assignment_sha256=spec.assignment_sha256,
                **evaluation_identity(),
            )
        with self.assertRaisesRegex(ValueError, "simultaneous"):
            semantic_from_native(Action.both(10, 20), ActionSpec())

    def test_all_required_scripted_baselines_are_buildable(self) -> None:
        baseline_ids = (
            "no_action_long_wait",
            "seeded_legal_random",
            "matcher_shot_policy",
            "scripted_direct_matcher",
            "scripted_side_ejector",
            "scripted_imminent_rot_hazard",
        )
        for baseline_id in baseline_ids:
            policy = ScriptedBaselineSpec(baseline_id).build(23)
            self.assertTrue(callable(policy.act))

    def test_recurrent_evaluator_is_deterministic_and_critic_alpha_is_zero(
        self,
    ) -> None:
        torch.manual_seed(31)
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1, critic_condition_features=1),
        )
        encoded = TeacherStateEncoder().encode((observation(0),))
        kind = torch.ones((1, 3), dtype=torch.bool)
        wait = torch.ones((1, len(ActionSpec().wait_choices)), dtype=torch.bool)
        policy = RecurrentSemanticPolicy(model)
        policy.reset(1)
        first = policy.act(encoded, kind, wait)
        policy.reset(1)
        second = policy.act(encoded, kind, wait)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)

        spec, blobs = _fixture()
        store = SnapshotBlobStore(spec.library, blobs)
        suite = EvaluationSuite(
            "recurrent-validation-v1",
            "validation",
            ("validation",),
            1,
            31,
            2,
            2,
            _RUNTIME_SHA256,
            spec.assignment_sha256,
            store.library.sha256,
            store.sha256,
            model.action_spec.sha256,
            (spec.library["validation"].sha256,),
        )
        eval_wait = torch.zeros_like(wait)
        eval_wait[:, 0] = True
        report = evaluate_recurrent_policy(
            FakeSingleSimulator(),
            store,
            suite,
            model,
            TeacherStateEncoder(),
            kind,
            eval_wait,
            evaluator_sha256="e" * 64,
            expected_assignment_sha256=spec.assignment_sha256,
            **evaluation_identity(),
        )
        self.assertEqual(len(report.episodes), 1)
        self.assertEqual(report.episodes[0].raw_score, 2)

        wait_only = torch.zeros_like(kind)
        wait_only[:, 0] = True
        restricted = evaluate_recurrent_policy(
            FakeSingleSimulator(),
            store,
            suite,
            model,
            TeacherStateEncoder(),
            wait_only,
            eval_wait,
            evaluator_sha256="e" * 64,
            expected_assignment_sha256=spec.assignment_sha256,
            **evaluation_identity(execution="d"),
        )
        self.assertNotEqual(report.policy_sha256, restricted.policy_sha256)

        with self.assertRaisesRegex(ValueError, "encoder schema"):
            evaluate_recurrent_policy(
                FakeSingleSimulator(),
                store,
                suite,
                model,
                ActorTrackEncoder(),
                kind,
                eval_wait,
                evaluator_sha256="e" * 64,
                expected_assignment_sha256=spec.assignment_sha256,
                **evaluation_identity(),
            )

        with self.assertRaisesRegex(ValueError, "all-masked"):
            evaluate_recurrent_policy(
                FakeSingleSimulator(),
                store,
                suite,
                model,
                TeacherStateEncoder(),
                torch.zeros_like(kind),
                eval_wait,
                evaluator_sha256="e" * 64,
                expected_assignment_sha256=spec.assignment_sha256,
                **evaluation_identity(),
            )

        with self.assertRaisesRegex(ValueError, "runtime identity"):
            evaluate_recurrent_policy(
                FakeSingleSimulator(),
                store,
                replace(suite, runtime_identity_sha256="f" * 64),
                model,
                TeacherStateEncoder(),
                kind,
                eval_wait,
                evaluator_sha256="e" * 64,
                expected_assignment_sha256=spec.assignment_sha256,
                **evaluation_identity(),
            )

    def test_long_wait_is_clipped_to_exact_tick_horizon(self) -> None:
        spec, blobs = _fixture()
        store = SnapshotBlobStore(spec.library, blobs)
        suite = EvaluationSuite(
            "bounded-validation-v1",
            "validation",
            ("validation",),
            1,
            37,
            3,
            3,
            _RUNTIME_SHA256,
            spec.assignment_sha256,
            store.library.sha256,
            store.sha256,
            ActionSpec().sha256,
            (spec.library["validation"].sha256,),
        )
        report = evaluate_scripted_baseline(
            FakeSingleSimulator(),
            store,
            suite,
            ScriptedBaselineSpec("no_action_long_wait"),
            evaluator_sha256="e" * 64,
            expected_assignment_sha256=spec.assignment_sha256,
            **evaluation_identity(),
        )
        self.assertEqual(report.episodes[0].elapsed_ticks, 3)
        self.assertEqual(report.episodes[0].raw_score, 3)

    def test_terminal_underflow_is_preserved_as_signed_final_gauge(self) -> None:
        torch.manual_seed(41)
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1),
        )
        spec, blobs = _fixture()
        store = SnapshotBlobStore(spec.library, blobs)
        suite = EvaluationSuite(
            "terminal-underflow-v1",
            "validation",
            ("validation",),
            1,
            41,
            4,
            4,
            _RUNTIME_SHA256,
            spec.assignment_sha256,
            store.library.sha256,
            store.sha256,
            model.action_spec.sha256,
            (spec.library["validation"].sha256,),
        )
        kind_mask = torch.zeros((1, 3), dtype=torch.bool)
        kind_mask[:, 1] = True
        wait_mask = torch.ones(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
        report = evaluate_recurrent_policy(
            FakeTerminalUnderflowSimulator(),
            store,
            suite,
            model,
            TeacherStateEncoder(),
            kind_mask,
            wait_mask,
            evaluator_sha256="e" * 64,
            expected_assignment_sha256=spec.assignment_sha256,
            **evaluation_identity(),
        )
        episode = report.episodes[0]
        self.assertTrue(episode.terminated)
        self.assertEqual(episode.minimum_gauge, -96)
        self.assertEqual(episode.final_gauge, -96)

    def test_horizon_underflow_is_preserved_as_signed_final_gauge(self) -> None:
        torch.manual_seed(42)
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1),
        )
        spec, blobs = _fixture()
        store = SnapshotBlobStore(spec.library, blobs)
        suite = EvaluationSuite(
            "horizon-underflow-v1",
            "validation",
            ("validation",),
            1,
            42,
            1,
            1,
            _RUNTIME_SHA256,
            spec.assignment_sha256,
            store.library.sha256,
            store.sha256,
            model.action_spec.sha256,
            (spec.library["validation"].sha256,),
        )
        kind_mask = torch.zeros((1, 3), dtype=torch.bool)
        kind_mask[:, 0] = True
        wait_mask = torch.zeros(
            (1, len(model.action_spec.wait_choices)), dtype=torch.bool
        )
        wait_mask[:, 0] = True
        report = evaluate_recurrent_policy(
            FakeHorizonUnderflowSimulator(),
            store,
            suite,
            model,
            TeacherStateEncoder(),
            kind_mask,
            wait_mask,
            evaluator_sha256="e" * 64,
            expected_assignment_sha256=spec.assignment_sha256,
            **evaluation_identity(),
        )
        episode = report.episodes[0]
        self.assertFalse(episode.terminated)
        self.assertTrue(episode.truncated)
        self.assertEqual(episode.minimum_gauge, -48)
        self.assertEqual(episode.final_gauge, -48)


if __name__ == "__main__":
    unittest.main()
