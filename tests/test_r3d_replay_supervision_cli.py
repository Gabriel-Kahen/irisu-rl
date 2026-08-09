from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import tempfile
import types
import unittest
from pathlib import Path

from irisu_env import EventKind


ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "benchmarks/r3d_replay_supervision.py"
SPEC = importlib.util.spec_from_file_location("r3d_replay_supervision_cli", CLI_PATH)
assert SPEC is not None and SPEC.loader is not None
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def _body(identifier: int, x: float) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": "dynamic_fresh",
        "color": 2,
        "x": x,
        "y": 200.0,
        "size": 40.0,
        "chain_id": 0,
    }


def _observation(tick: int, score: int = 0) -> dict[str, object]:
    return {
        "tick": tick,
        "score": score,
        "gauge": 1000,
        "gauge_max": 1000,
        "level": 1,
        "highest_chain": 0,
        "qualifying_clear_count": 0,
        "left_held": False,
        "right_held": False,
        "terminated": False,
        "truncated": False,
        "field": {"x": 0.0, "y": 0.0, "width": 640.0, "height": 480.0},
        "difficulty": {"active_colors": 4, "spawn_interval_ticks": 100},
        "bodies": (_body(7, 200.0), _body(8, 260.0)),
    }


class _FakeExactEnv:
    physics_backend = "exact"

    def __init__(self, *, worker_path: str, **_: object) -> None:
        self.worker_sha256 = hashlib.sha256(Path(worker_path).read_bytes()).hexdigest()
        self.tick = 0

    @property
    def build_info(self) -> dict[str, object]:
        return {
            "physics_backend": "fake-exact",
            "protocol_version": 1,
            "pointer_bits": 32,
            "body_capacity": 196,
            "config_hash": 17,
            "x87_control_word": 639,
            "worker_backend": "fake",
            "worker_compiler": "test",
            "exact_library_sha256": "d" * 64,
            "worker_executable_sha256": self.worker_sha256,
        }

    def exact_library_provenance(self) -> dict[str, object]:
        return {
            "status": "captured",
            "path": "/fake/libirisu.so",
            "bytes": 1,
            "sha256": "d" * 64,
        }

    def reset(self, *, seed: int):
        self.tick = 0
        return _observation(0), {"seed": seed, "config_hash": 17}

    def config_hash(self) -> int:
        return 17

    def step(self, action):
        del action
        self.tick += 1
        events: list[dict[str, int]] = []
        if self.tick == 3:
            events.append(
                {
                    "tick": 3,
                    "kind": int(EventKind.SHOT_FIRED),
                    "a": 50,
                    "value": 1,
                }
            )
        if self.tick == 4:
            events.append(
                {
                    "tick": 4,
                    "kind": int(EventKind.PROJECTILE_HIT),
                    "a": 50,
                    "b": 7,
                }
            )
        return _observation(self.tick, 42 if self.tick == 4 else 0), 0, False, False, {
            "events": events
        }

    def __enter__(self):
        return self

    def __exit__(self, *_: object) -> None:
        return None


class ReplaySupervisionCliTests(unittest.TestCase):
    def test_parser_free_identity_bound_conversion(self) -> None:
        frames = (
            types.SimpleNamespace(left=False, right=True, x=200, y=230, raw_word=2),
            types.SimpleNamespace(left=False, right=False, x=200, y=230, raw_word=0),
            types.SimpleNamespace(left=False, right=True, x=180, y=230, raw_word=2),
            types.SimpleNamespace(left=False, right=False, x=180, y=230, raw_word=0),
        )
        replay = types.SimpleNamespace(
            header=types.SimpleNamespace(
                seed=5,
                highest_level=1,
                final_score=42,
                highest_chain=0,
                mode=0,
            ),
            layout="padded",
            frames=frames,
        )
        parser = types.SimpleNamespace(parse_replay=lambda data, layout: replay)
        with tempfile.TemporaryDirectory() as temporary:
            replay_path = Path(temporary) / "replay.rpy"
            worker_path = Path(temporary) / "worker"
            replay_path.write_bytes(b"parser-free-replay")
            worker_path.write_bytes(b"fake-worker")
            os.chmod(worker_path, 0o700)
            args = argparse.Namespace(
                replay=replay_path,
                worker=worker_path,
                layout="padded",
                train_steps=0,
                holdout_fraction=0.2,
                batch_size=2,
                learning_rate=3e-4,
                training_seed=7,
            )
            report = CLI.run_diagnostic(
                args,
                parser_module=parser,
                env_factory=_FakeExactEnv,
                source_paths=(CLI_PATH,),
            )

        self.assertTrue(report["ok"])
        self.assertFalse(report["scope"]["canonical_evidence"])
        runtime = report["identity"]["exact_runtime"]
        self.assertEqual(runtime["build"]["config_hash"], 17)
        self.assertEqual(report["conversion"]["first_hit_supervision"], 1)
        self.assertEqual(report["conversion"]["safe_ungrouped_examples"], 1)
        self.assertIsNone(report["temporal_holdout"])


if __name__ == "__main__":
    unittest.main()
