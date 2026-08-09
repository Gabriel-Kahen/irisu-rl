from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from irisu_pointer.geometry_checkpoint import (
    load_geometry_checkpoint,
    load_safeguarded_geometry_ensemble_policy,
    load_safeguarded_geometry_policy,
    save_geometry_checkpoint,
)
from irisu_pointer.geometry_learning import GeometryModelConfig, GeometrySelectorModel
from irisu_pointer.geometry_policy import (
    GeometryPolicyConfig,
    GeometrySelectorEnsemble,
    SafeguardedGeometryPolicy,
    geometry_candidate_vocabulary_manifest,
    geometry_candidate_vocabulary_sha256,
)
from irisu_pointer.geometry_search import GeometrySearchConfig
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.steering_learning import (
    GoalConditionedSteeringModel,
    GoalConditionedSteeringPolicy,
    SteeringModelConfig,
)
from irisu_rl.actions import SemanticAction
from irisu_rl.schema import TEACHER_V1

_BASE_SHA256 = "b" * 64
_SOURCE_SHA256 = "c" * 64
_SELECTOR_SHA256 = "d" * 64


def _body(
    identifier: int,
    x: float,
    *,
    color: int = 1,
    shape: str = "circle",
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": shape,
        "lifecycle": "dynamic_fresh",
        "color": color,
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


def _observation(
    tick: int = 1,
    *,
    source_x: float = 200.0,
    source_shape: str = "circle",
):
    return {
        "tick": tick,
        "score": 0,
        "gauge": 900,
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
        "bodies": [
            _body(1, source_x, shape=source_shape),
            _body(2, 360.0),
        ],
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
        reason="frozen incumbent",
    )


class _FrozenBasePolicy(GoalConditionedSteeringPolicy):
    def __init__(self, decision: SteeringDecision | None = None) -> None:
        model = GoalConditionedSteeringModel(
            TEACHER_V1,
            config=SteeringModelConfig(body_hidden=8, global_hidden=8, pair_hidden=8),
        )
        model.eval()
        super().__init__(model, artifact_sha256=_BASE_SHA256)
        self.fixed_decision = _incumbent() if decision is None else decision
        self.calls = 0
        self.reset_calls = 0

    def predict(self, observation):
        self.calls += 1
        return self.fixed_decision

    def reset(self, seed: int = 0) -> None:
        self.reset_calls += 1
        super().reset(seed)


def _selector(
    config: GeometrySearchConfig,
    biases: dict[int, float],
) -> GeometrySelectorModel:
    model = GeometrySelectorModel(
        TEACHER_V1,
        candidate_count=config.max_candidates,
        candidate_set_sha256=geometry_candidate_vocabulary_sha256(config),
        config=GeometryModelConfig(body_hidden=8, pair_hidden=8),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for slot, value in biases.items():
            model.selector[-1].bias[slot] = value
    model.eval()
    return model


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensemble(
    config: GeometrySearchConfig,
    *biases: dict[int, float],
) -> GeometrySelectorEnsemble:
    return GeometrySelectorEnsemble(
        tuple(_selector(config, value) for value in biases),
        artifact_sha256s=tuple(f"{index + 1:064x}" for index in range(len(biases))),
    )


class GeometryPolicyTests(unittest.TestCase):
    def policy(
        self,
        biases: dict[int, float],
        *,
        policy_config: GeometryPolicyConfig | None = None,
        base: _FrozenBasePolicy | None = None,
    ) -> SafeguardedGeometryPolicy:
        geometry = GeometrySearchConfig()
        return SafeguardedGeometryPolicy(
            base or _FrozenBasePolicy(),
            _selector(geometry, biases),
            geometry_config=geometry,
            policy_config=policy_config,
            selector_artifact_sha256=_SELECTOR_SHA256,
            source_identity=_SOURCE_SHA256,
        )

    def ensemble_policy(
        self,
        *biases: dict[int, float],
        policy_config: GeometryPolicyConfig,
    ) -> SafeguardedGeometryPolicy:
        geometry = GeometrySearchConfig()
        ensemble = _ensemble(geometry, *biases)
        return SafeguardedGeometryPolicy(
            _FrozenBasePolicy(),
            ensemble,
            geometry_config=geometry,
            policy_config=policy_config,
            selector_artifact_sha256=ensemble.sha256,
            source_identity=_SOURCE_SHA256,
        )

    def test_high_confidence_available_slot_replaces_incumbent(self) -> None:
        policy = self.policy({1: 10.0, 27: 100.0})
        decision = policy.predict(_observation(source_shape="triangle"))

        self.assertNotEqual(decision, _incumbent())
        self.assertEqual(decision.reason.split()[0], "shape-support")
        selection = policy.last_selection
        assert selection is not None
        self.assertEqual(selection.proposed_slot, 1)
        self.assertEqual(selection.deployed_slot, 1)
        self.assertTrue(selection.used_learned_geometry)
        self.assertNotIn(27, selection.available_slots)
        self.assertGreater(selection.proposed_confidence or 0.0, 0.99)
        self.assertTrue(
            all(not value.requires_grad for value in policy.selector.parameters())
        )
        self.assertTrue(
            all(
                not value.requires_grad
                for value in policy.base_policy.model.parameters()
            )
        )

    def test_confidence_and_margin_guards_each_fall_back(self) -> None:
        confidence = self.policy(
            {1: 1.0},
            policy_config=GeometryPolicyConfig(
                minimum_confidence=0.8, minimum_logit_margin=0.5
            ),
        )
        self.assertIs(
            confidence.predict(_observation()), confidence.base_policy.fixed_decision
        )
        assert confidence.last_selection is not None
        self.assertIn("confidence", confidence.last_selection.reason)

        margin = self.policy(
            {1: 0.5},
            policy_config=GeometryPolicyConfig(
                minimum_confidence=0.0, minimum_logit_margin=1.0
            ),
        )
        self.assertIs(margin.predict(_observation()), margin.base_policy.fixed_decision)
        assert margin.last_selection is not None
        self.assertIn("margin", margin.last_selection.reason)

    def test_unavailable_candidate_is_masked_without_slot_renumbering(self) -> None:
        policy = self.policy({27: 100.0})
        decision = policy.predict(_observation(source_shape="triangle"))
        selection = policy.last_selection
        assert selection is not None
        self.assertNotIn(27, selection.available_slots)
        self.assertNotEqual(selection.proposed_slot, 27)
        self.assertIs(decision, policy.base_policy.fixed_decision)

    def test_wait_and_same_tick_are_incumbent_idempotent(self) -> None:
        wait = SteeringDecision(
            SemanticAction.wait(8), SteeringIntent.WAIT, reason="frozen restraint"
        )
        base = _FrozenBasePolicy(wait)
        policy = self.policy({1: 10.0}, base=base)
        first = policy.predict(_observation(tick=7))
        changed = _observation(tick=7)
        changed["gauge"] = 1
        second = policy.predict(changed)

        self.assertIs(first, wait)
        self.assertIs(second, first)
        self.assertEqual(base.calls, 1)
        assert policy.last_selection is not None
        self.assertIn("restraint", policy.last_selection.reason)
        policy.reset(9)
        self.assertEqual(base.reset_calls, 1)
        self.assertIs(policy.predict(_observation(tick=7)), wait)
        self.assertEqual(base.calls, 2)
        with self.assertRaisesRegex(RuntimeError, "moved backwards"):
            policy.predict(_observation(tick=6))

    def test_policy_identity_binds_base_selector_schema_and_safeguards(self) -> None:
        policy = self.policy({0: 1.0})
        manifest = policy.identity_manifest()
        self.assertEqual(manifest["base_policy_checkpoint_sha256"], _BASE_SHA256)
        self.assertEqual(
            manifest["candidate_vocabulary_sha256"],
            geometry_candidate_vocabulary_sha256(policy.geometry_config),
        )
        self.assertEqual(len(policy.sha256), 64)
        self.assertEqual(
            geometry_candidate_vocabulary_manifest(policy.geometry_config)["slots"][31][
                "family"
            ],
            "central-interior",
        )

    def test_default_config_preserves_single_selector_behavior_and_manifest(
        self,
    ) -> None:
        self.assertEqual(
            GeometryPolicyConfig().manifest(),
            {
                "minimum_confidence": 0.7,
                "minimum_logit_margin": 1.0,
            },
        )
        policy = self.policy({1: 10.0})
        self.assertEqual(
            policy.predict(_observation()).reason.split()[0],
            "shape-support",
        )
        selection = policy.last_selection
        assert selection is not None
        self.assertEqual(selection.selector_member_count, 1)
        self.assertEqual(selection.ensemble_agreement, 1.0)
        self.assertEqual(selection.unverified_learned_corrections, 0)

    def test_minimum_gauge_reserve_is_an_opt_in_gate(self) -> None:
        policy = self.policy(
            {1: 10.0},
            policy_config=GeometryPolicyConfig(
                minimum_confidence=0.0,
                minimum_logit_margin=0.0,
                minimum_gauge_fraction=0.075,
            ),
        )
        low = _observation(tick=1)
        low["gauge"] = 7
        low["gauge_max"] = 100

        self.assertIs(policy.predict(low), policy.base_policy.fixed_decision)
        selection = policy.last_selection
        assert selection is not None
        self.assertEqual(selection.proposed_slot, 1)
        self.assertAlmostEqual(selection.gauge_fraction or 0.0, 0.07)
        self.assertIn("gauge reserve", selection.reason)

        enough = _observation(tick=2)
        enough["gauge"] = 8
        enough["gauge_max"] = 100
        self.assertIsNot(policy.predict(enough), policy.base_policy.fixed_decision)
        selection = policy.last_selection
        assert selection is not None
        self.assertAlmostEqual(selection.gauge_fraction or 0.0, 0.08)
        self.assertTrue(selection.used_learned_geometry)
        self.assertEqual(
            policy.policy_config.manifest()["minimum_gauge_fraction"], 0.075
        )
        self.assertIn(
            "public gauge reserve", policy.identity_manifest()["selection_rule"]
        )
        self.assertEqual(
            policy.statistics(),
            {
                "learned_geometry_deployments": 1,
                "safeguard_fallbacks": 1,
                "unsupported_pair_fallbacks": 0,
                "gauge_reserve_rejections": 1,
                "progress_credit_rejections": 0,
                "progress_credit_replenishments": 0,
                "score_progress_events": 0,
                "qualifying_clear_progress_events": 0,
                "unverified_learned_corrections": 0,
            },
        )

    def test_progress_credit_requires_public_consequence_before_reuse(self) -> None:
        policy = self.policy(
            {1: 10.0},
            policy_config=GeometryPolicyConfig(
                minimum_confidence=0.0,
                minimum_logit_margin=0.0,
                maximum_unverified_corrections=2,
            ),
        )

        first = policy.predict(_observation(tick=1))
        self.assertIsNot(first, policy.base_policy.fixed_decision)
        selection = policy.last_selection
        assert selection is not None
        self.assertEqual(selection.unverified_learned_corrections, 1)

        second = policy.predict(_observation(tick=2))
        self.assertIsNot(second, policy.base_policy.fixed_decision)
        selection = policy.last_selection
        assert selection is not None
        self.assertEqual(selection.unverified_learned_corrections, 2)

        blocked = policy.predict(_observation(tick=3))
        self.assertIs(blocked, policy.base_policy.fixed_decision)
        selection = policy.last_selection
        assert selection is not None
        self.assertIn("progress-credit", selection.reason)
        self.assertEqual(selection.progress_credit_replenished_by, ())

        progressed = _observation(tick=4)
        progressed["score"] = 1
        self.assertIsNot(policy.predict(progressed), policy.base_policy.fixed_decision)
        selection = policy.last_selection
        assert selection is not None
        self.assertEqual(selection.progress_credit_replenished_by, ("score",))
        self.assertEqual(selection.unverified_learned_corrections, 1)
        self.assertEqual(
            policy.statistics(),
            {
                "learned_geometry_deployments": 3,
                "safeguard_fallbacks": 1,
                "unsupported_pair_fallbacks": 0,
                "gauge_reserve_rejections": 0,
                "progress_credit_rejections": 1,
                "progress_credit_replenishments": 1,
                "score_progress_events": 1,
                "qualifying_clear_progress_events": 0,
                "unverified_learned_corrections": 1,
            },
        )
        with self.assertRaisesRegex(RuntimeError, "progress counters moved backwards"):
            policy.predict(_observation(tick=5))

    def test_qualifying_clear_gain_also_replenishes_progress_credit(self) -> None:
        policy = self.policy(
            {1: 10.0},
            policy_config=GeometryPolicyConfig(
                minimum_confidence=0.0,
                minimum_logit_margin=0.0,
                maximum_unverified_corrections=2,
            ),
        )
        policy.predict(_observation(tick=1))
        progressed = _observation(tick=2)
        progressed["qualifying_clear_count"] = 1

        self.assertIsNot(policy.predict(progressed), policy.base_policy.fixed_decision)
        selection = policy.last_selection
        assert selection is not None
        self.assertEqual(
            selection.progress_credit_replenished_by, ("qualifying_clear",)
        )
        self.assertEqual(policy.statistics()["qualifying_clear_progress_events"], 1)

    def test_unsupported_non_piece_pair_falls_back_without_search(self) -> None:
        policy = self.policy({1: 10.0})
        observation = _observation()
        observation["bodies"][1]["kind"] = "orb"
        observation["bodies"][1]["shape"] = "none"

        decision = policy.predict(observation)

        self.assertIs(decision, policy.base_policy.fixed_decision)
        selection = policy.last_selection
        assert selection is not None
        self.assertIn("unsupported", selection.reason)
        self.assertEqual(selection.available_slots, ())
        self.assertEqual(policy.statistics()["unsupported_pair_fallbacks"], 1)

    def test_new_safeguard_config_validation_is_strict(self) -> None:
        for arguments in (
            {"minimum_gauge_fraction": -0.01},
            {"minimum_gauge_fraction": 1.01},
            {"minimum_gauge_fraction": True},
            {"maximum_unverified_corrections": -1},
            {"maximum_unverified_corrections": True},
        ):
            with (
                self.subTest(arguments=arguments),
                self.assertRaises((TypeError, ValueError)),
            ):
                GeometryPolicyConfig(**arguments)

    def test_ensemble_disagreement_falls_back_to_incumbent(self) -> None:
        policy = self.ensemble_policy(
            {1: 10.0},
            {2: 10.0},
            policy_config=GeometryPolicyConfig(
                minimum_confidence=0.0,
                minimum_logit_margin=0.0,
                minimum_ensemble_members=2,
                minimum_ensemble_agreement=1.0,
            ),
        )
        decision = policy.predict(_observation())
        selection = policy.last_selection
        assert selection is not None
        self.assertIs(decision, policy.base_policy.fixed_decision)
        self.assertEqual(selection.proposed_slot, 1)
        self.assertEqual(selection.selector_member_count, 2)
        self.assertEqual(selection.ensemble_agreement, 0.5)
        self.assertIn("ensemble-agreement", selection.reason)

    def test_agreeing_ensemble_can_deploy_candidate(self) -> None:
        policy = self.ensemble_policy(
            {1: 10.0},
            {1: 8.0},
            policy_config=GeometryPolicyConfig(
                minimum_confidence=0.5,
                minimum_logit_margin=1.0,
                minimum_ensemble_members=2,
                minimum_ensemble_agreement=1.0,
                minimum_member_incumbent_logit_margin=1.0,
            ),
        )
        decision = policy.predict(_observation())
        selection = policy.last_selection
        assert selection is not None
        self.assertIsNot(decision, policy.base_policy.fixed_decision)
        self.assertEqual(selection.deployed_slot, 1)
        self.assertEqual(selection.ensemble_agreement, 1.0)
        self.assertEqual(selection.minimum_member_incumbent_logit_margin, 8.0)
        self.assertTrue(selection.used_learned_geometry)

    def test_each_member_can_be_required_to_beat_incumbent(self) -> None:
        policy = self.ensemble_policy(
            {1: 10.0},
            {0: 8.0, 1: 7.0, 2: 9.0},
            policy_config=GeometryPolicyConfig(
                minimum_confidence=0.0,
                minimum_logit_margin=0.0,
                minimum_ensemble_members=2,
                minimum_ensemble_agreement=0.5,
                minimum_member_incumbent_logit_margin=0.0,
            ),
        )
        decision = policy.predict(_observation())
        selection = policy.last_selection
        assert selection is not None
        self.assertIs(decision, policy.base_policy.fixed_decision)
        self.assertEqual(selection.proposed_slot, 1)
        self.assertEqual(selection.minimum_member_incumbent_logit_margin, -1.0)
        self.assertIn("member-incumbent", selection.reason)

    def test_nonfinite_ensemble_member_fails_closed(self) -> None:
        geometry = GeometrySearchConfig()
        first = _selector(geometry, {1: 10.0})
        second = _selector(geometry, {1: 10.0})
        with torch.no_grad():
            second.selector[-1].bias[1] = torch.nan
        ensemble = GeometrySelectorEnsemble(
            (first, second),
            artifact_sha256s=("1" * 64, "2" * 64),
        )
        policy = SafeguardedGeometryPolicy(
            _FrozenBasePolicy(),
            ensemble,
            geometry_config=geometry,
            policy_config=GeometryPolicyConfig(
                minimum_ensemble_members=2,
                minimum_ensemble_agreement=1.0,
            ),
            selector_artifact_sha256=ensemble.sha256,
            source_identity=_SOURCE_SHA256,
        )
        decision = policy.predict(_observation())
        self.assertIs(decision, policy.base_policy.fixed_decision)
        assert policy.last_selection is not None
        self.assertIn("non-finite", policy.last_selection.reason)

    def test_ensemble_contract_and_identity_mismatches_fail_closed(self) -> None:
        geometry = GeometrySearchConfig()
        first = _selector(geometry, {1: 1.0})
        second = _selector(geometry, {1: 1.0})
        with self.assertRaisesRegex(ValueError, "unique"):
            GeometrySelectorEnsemble(
                (first, second),
                artifact_sha256s=("1" * 64, "1" * 64),
            )
        incompatible_geometry = GeometrySearchConfig(max_candidates=25)
        incompatible = _selector(incompatible_geometry, {1: 1.0})
        with self.assertRaisesRegex(ValueError, "contracts"):
            GeometrySelectorEnsemble(
                (first, incompatible),
                artifact_sha256s=("1" * 64, "2" * 64),
            )
        ensemble = _ensemble(geometry, {1: 1.0}, {1: 1.0})
        with self.assertRaisesRegex(ValueError, "identity binding"):
            SafeguardedGeometryPolicy(
                _FrozenBasePolicy(),
                ensemble,
                geometry_config=geometry,
                selector_artifact_sha256="f" * 64,
                source_identity=_SOURCE_SHA256,
            )


class GeometryCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = GeometrySearchConfig()
        self.safeguards = GeometryPolicyConfig(
            minimum_confidence=0.8,
            minimum_logit_margin=1.5,
            minimum_gauge_fraction=0.075,
            maximum_unverified_corrections=2,
        )

    def save(
        self,
        path: Path,
        *,
        biases: dict[int, float] | None = None,
        policy_config: GeometryPolicyConfig | None = None,
    ) -> str:
        return save_geometry_checkpoint(
            path,
            _selector(
                self.geometry,
                {1: 4.0} if biases is None else biases,
            ),
            geometry_config=self.geometry,
            policy_config=(self.safeguards if policy_config is None else policy_config),
            base_policy_checkpoint_sha256=_BASE_SHA256,
            source_identity=_SOURCE_SHA256,
            metadata={"development_only": True, "wave": 3},
        )

    def test_round_trip_binds_all_deployment_identities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.pt"
            digest = self.save(path)
            loaded = load_geometry_checkpoint(
                path,
                expected_sha256=digest,
                expected_base_policy_checkpoint_sha256=_BASE_SHA256,
                expected_source_identity=_SOURCE_SHA256,
            )
            self.assertEqual(loaded.sha256, digest)
            self.assertEqual(loaded.model.schema.sha256, TEACHER_V1.sha256)
            self.assertEqual(
                loaded.model.candidate_set_sha256,
                geometry_candidate_vocabulary_sha256(self.geometry),
            )
            self.assertEqual(loaded.geometry_config, self.geometry)
            self.assertEqual(loaded.policy_config, self.safeguards)
            self.assertEqual(loaded.metadata["wave"], 3)
            self.assertFalse(loaded.model.training)
            self.assertTrue(
                all(not value.requires_grad for value in loaded.model.parameters())
            )

            policy = load_safeguarded_geometry_policy(
                path,
                base_policy=_FrozenBasePolicy(),
                expected_sha256=digest,
                expected_base_policy_checkpoint_sha256=_BASE_SHA256,
                expected_source_identity=_SOURCE_SHA256,
            )
            self.assertEqual(policy.selector_artifact_sha256, digest)
            self.assertEqual(policy.source_identity, _SOURCE_SHA256)

    def test_ensemble_loader_binds_each_checkpoint_contract(self) -> None:
        safeguards = GeometryPolicyConfig(
            minimum_confidence=0.8,
            minimum_logit_margin=1.5,
            minimum_ensemble_members=2,
            minimum_ensemble_agreement=1.0,
            minimum_member_incumbent_logit_margin=1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.pt"
            second = Path(directory) / "second.pt"
            first_sha = self.save(first, biases={1: 8.0}, policy_config=safeguards)
            second_sha = self.save(second, biases={1: 9.0}, policy_config=safeguards)
            policy = load_safeguarded_geometry_ensemble_policy(
                (first, second),
                base_policy=_FrozenBasePolicy(),
                expected_sha256s=(first_sha, second_sha),
                expected_base_policy_checkpoint_sha256=_BASE_SHA256,
                expected_source_identity=_SOURCE_SHA256,
            )
            self.assertIsInstance(policy.selector, GeometrySelectorEnsemble)
            self.assertEqual(policy.selector.member_count, 2)
            self.assertEqual(policy.selector_artifact_sha256, policy.selector.sha256)
            decision = policy.predict(_observation())
            self.assertIsNot(decision, policy.base_policy.fixed_decision)
            assert policy.last_selection is not None
            self.assertEqual(policy.last_selection.ensemble_agreement, 1.0)

    def test_ensemble_loader_rejects_mixed_policy_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.pt"
            second = Path(directory) / "second.pt"
            first_sha = self.save(first, biases={1: 4.0})
            second_sha = self.save(
                second,
                biases={1: 5.0},
                policy_config=GeometryPolicyConfig(),
            )
            with self.assertRaisesRegex(ValueError, "contracts differ"):
                load_safeguarded_geometry_ensemble_policy(
                    (first, second),
                    base_policy=_FrozenBasePolicy(),
                    expected_sha256s=(first_sha, second_sha),
                    expected_base_policy_checkpoint_sha256=_BASE_SHA256,
                    expected_source_identity=_SOURCE_SHA256,
                )

    def test_expected_file_base_and_source_identities_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.pt"
            digest = self.save(path)
            arguments = {
                "expected_sha256": digest,
                "expected_base_policy_checkpoint_sha256": _BASE_SHA256,
                "expected_source_identity": _SOURCE_SHA256,
            }
            for field, value, message in (
                ("expected_sha256", "0" * 64, "SHA-256 mismatch"),
                (
                    "expected_base_policy_checkpoint_sha256",
                    "0" * 64,
                    "base-policy identity mismatch",
                ),
                (
                    "expected_source_identity",
                    "0" * 64,
                    "source identity mismatch",
                ),
            ):
                changed = dict(arguments)
                changed[field] = value
                with (
                    self.subTest(field=field),
                    self.assertRaisesRegex(ValueError, message),
                ):
                    load_geometry_checkpoint(path, **changed)

    def test_tampered_architecture_binding_is_rejected_after_file_rehash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "geometry.pt"
            self.save(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["bindings"]["architecture_sha256"] = "0" * 64
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "architecture binding"):
                load_geometry_checkpoint(
                    path,
                    expected_sha256=_file_sha256(path),
                    expected_base_policy_checkpoint_sha256=_BASE_SHA256,
                    expected_source_identity=_SOURCE_SHA256,
                )

    def test_save_rejects_model_from_another_vocabulary(self) -> None:
        model = GeometrySelectorModel(
            TEACHER_V1,
            candidate_count=self.geometry.max_candidates,
            candidate_set_sha256="0" * 64,
            config=GeometryModelConfig(body_hidden=8, pair_hidden=8),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(ValueError, "candidate identities differ"),
        ):
            save_geometry_checkpoint(
                Path(directory) / "geometry.pt",
                model,
                geometry_config=self.geometry,
                policy_config=self.safeguards,
                base_policy_checkpoint_sha256=_BASE_SHA256,
                source_identity=_SOURCE_SHA256,
            )


if __name__ == "__main__":
    unittest.main()
