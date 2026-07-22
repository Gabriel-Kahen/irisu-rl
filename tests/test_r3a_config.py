from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from irisu_rl.actions import ActionSpec
from irisu_rl.collector import CollectorConfig
from irisu_rl.schema import TEACHER_V1


ROOT = Path(__file__).resolve().parents[1]


class R3AConfigTests(unittest.TestCase):
    def test_checked_config_binds_collector_reward_and_transfer_scope(self) -> None:
        config = tomllib.loads(
            (ROOT / "configs/rl/experiments/r3a-multistep-v1.toml").read_text()
        )
        collector = config["collector"]
        expected = CollectorConfig(
            max_decisions=collector["max_decisions"],
            target_simulated_ticks=collector["target_simulated_ticks"],
            gamma_tick=collector["gamma_tick"],
            lambda_tick=collector["lambda_tick"],
        )
        self.assertEqual(expected.manifest()["version"], collector["version"])
        self.assertEqual(config["policy_schema"], TEACHER_V1.version)
        self.assertEqual(config["action_schema"], ActionSpec().version)
        self.assertFalse(config["deployable"])
        self.assertEqual(config["observation_provenance"], "privileged_simulator")
        self.assertEqual(config["reward"]["score_only_weight_ppm"], 0)
        self.assertEqual(
            config["curriculum"]["environment_topology"],
            "one_fixed_config_hash_per_collector_pool",
        )
        self.assertIsInstance(config["curriculum"]["evaluation_seed"], int)
        self.assertEqual(config["ppo"]["learning_rate"], 1e-4)
        self.assertGreater(config["smoke_run"]["max_consecutive_skips"], 0)


if __name__ == "__main__":
    unittest.main()
