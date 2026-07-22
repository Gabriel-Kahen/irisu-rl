from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from irisu_rl.actions import ActionSpec
from irisu_rl.manifests import SimulatorIdentity, canonical_sha256, runtime_manifest
from irisu_rl.models import RecurrentActorCritic, RecurrentModelConfig
from irisu_rl.ppo import PPOConfig
from irisu_rl.returns import lambda_tick_from_half_life
from irisu_rl.schema import TEACHER_V1
from irisu_rl.seeds import SeedAllocator

ROOT = Path(__file__).resolve().parents[1]


class R2ManifestTests(unittest.TestCase):
    def test_checked_configs_match_code_and_remain_non_deployable(self) -> None:
        action = tomllib.loads(
            (ROOT / "configs/rl/actions/deployment-v1.toml").read_text()
        )
        self.assertEqual(action["sha256"], ActionSpec().sha256)
        experiment = tomllib.loads(
            (ROOT / "configs/rl/experiments/r2a-smoke-v1.toml").read_text()
        )
        self.assertEqual(experiment["model"], RecurrentModelConfig().manifest())
        expected_ppo = PPOConfig().manifest()
        self.assertEqual(
            {key: experiment["ppo"][key] for key in expected_ppo}, expected_ppo
        )
        self.assertAlmostEqual(
            experiment["smdp"]["lambda_tick"], lambda_tick_from_half_life(1.0)
        )
        self.assertFalse(experiment["deployable"])
        self.assertEqual(experiment["observation_provenance"], "privileged_simulator")

    def test_runtime_manifest_binds_lock_model_and_transfer_gate(self) -> None:
        model = RecurrentActorCritic(
            TEACHER_V1,
            config=RecurrentModelConfig(8, 8, 12, 12, 1),
        )
        manifest = runtime_manifest(
            ROOT,
            model=model,
            ppo=PPOConfig(),
            reward_scale=1.0,
            gamma_tick=1.0,
            lambda_tick=lambda_tick_from_half_life(1.0),
            code_revision="test-revision",
            observation_provenance="privileged_simulator",
            simulator=SimulatorIdentity(
                backend="portable",
                worker_executable_sha256=None,
                physics_library_sha256="1" * 64,
                mechanics_config_sha256="2" * 64,
                config_hashes=(3, 4),
                protocol_version=1,
                seed_manifest_sha256=SeedAllocator().manifest_sha256,
            ),
        )
        digest = manifest.pop("manifest_sha256")
        self.assertEqual(digest, canonical_sha256(manifest))
        self.assertFalse(manifest["deployable"])
        self.assertEqual(manifest["model"]["actor_schema_sha256"], TEACHER_V1.sha256)
        self.assertRegex(manifest["uv_lock_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(manifest["simulator"]["config_hashes"], [3, 4])

    def test_simulator_identity_requires_exact_runtime_and_mechanics_hashes(self) -> None:
        with self.assertRaisesRegex(ValueError, "worker hash"):
            SimulatorIdentity(
                backend="exact",
                worker_executable_sha256=None,
                physics_library_sha256="1" * 64,
                mechanics_config_sha256="2" * 64,
                config_hashes=(3,),
                protocol_version=1,
                seed_manifest_sha256="4" * 64,
            )


if __name__ == "__main__":
    unittest.main()
