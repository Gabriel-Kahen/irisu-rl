from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace

from irisu_env import EventKind
from irisu_pointer.action import PointerActionSpec
from irisu_pointer.replay_supervision import (
    ReplayEvidenceIdentity,
    ReplayInputFrame,
    collect_replay_steering_supervision,
)
from irisu_rl.schema import TEACHER_V1


_DIGEST = hashlib.sha256(b"test").hexdigest()


def _observation(tick: int, *, score: int = 0) -> dict[str, object]:
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
        "bodies": (
            {
                "id": 7,
                "kind": "piece",
                "shape": "circle",
                "lifecycle": "dynamic_fresh",
                "color": 2,
                "x": 200.0,
                "y": 200.0,
                "size": 40.0,
                "chain_id": 0,
            },
            {
                "id": 8,
                "kind": "piece",
                "shape": "circle",
                "lifecycle": "dynamic_fresh",
                "color": 2,
                "x": 260.0,
                "y": 190.0,
                "size": 40.0,
                "chain_id": 0,
            },
        ),
    }


class _FakeReplayEnv:
    def __init__(self) -> None:
        self.tick = 0
        self.steps = 0

    def reset(self, *, seed: int):
        self.tick = 0
        self.steps = 0
        return _observation(0), {"config_hash": 17, "seed": seed}

    def step(self, action):
        self.tick += 1
        self.steps += 1
        events: list[dict[str, int]] = []
        # Records 0/1 are startup-suppressed. Record 2 fires projectile 50.
        if self.steps == 3:
            events.append(
                {
                    "tick": self.tick,
                    "kind": int(EventKind.SHOT_FIRED),
                    "a": 50,
                    "value": 1,
                }
            )
        if self.steps == 4:
            events.extend(
                (
                    {
                        "tick": self.tick,
                        "kind": int(EventKind.PROJECTILE_HIT),
                        "a": 50,
                        "b": 7,
                    },
                    {
                        "tick": self.tick,
                        "kind": int(EventKind.CHAIN_JOINED),
                        "a": 7,
                        "b": 8,
                    },
                    {
                        "tick": self.tick,
                        "kind": int(EventKind.CLEARED),
                        "a": 7,
                        "b": 8,
                    },
                )
            )
        return (
            _observation(self.tick, score=42 if self.steps >= 4 else 0),
            0,
            False,
            False,
            {"events": events},
        )


class ReplaySupervisionTests(unittest.TestCase):
    def identity(self, spec: PointerActionSpec) -> ReplayEvidenceIdentity:
        return ReplayEvidenceIdentity(
            source_revision="deadbee",
            replay_sha256=_DIGEST,
            runtime_sha256=_DIGEST,
            config_hash=17,
            observation_schema_sha256=TEACHER_V1.sha256,
            pointer_spec_sha256=spec.sha256,
        )

    def test_collects_first_hit_and_inferred_destination(self) -> None:
        spec = PointerActionSpec()
        frames = (
            ReplayInputFrame(0, False, True, 200, 230),
            ReplayInputFrame(1, False, False, 200, 230),
            ReplayInputFrame(2, False, True, 180, 230),
            ReplayInputFrame(3, False, False, 180, 230),
        )
        collection = collect_replay_steering_supervision(
            _FakeReplayEnv(),
            frames,
            seed=5,
            identity=self.identity(spec),
            pointer_spec=spec,
        )
        self.assertEqual(len(collection.shots), 1)
        shot = collection.shots[0]
        self.assertEqual(shot.projectile_id, 50)
        self.assertEqual(shot.target_body_id, 7)
        self.assertEqual(shot.destination_body_id, 8)
        self.assertEqual(shot.first_hit_delay_ticks, 1)
        self.assertAlmostEqual(shot.cursor_x_radius_offset, -1.0)
        self.assertAlmostEqual(shot.cursor_y_radius_offset, 1.5)
        self.assertEqual(spec.templates[shot.template_index], (-1.0, 1.5))
        self.assertEqual(collection.metrics.requested_shots, 1)
        self.assertEqual(collection.metrics.shots_fired, 1)
        self.assertEqual(collection.metrics.shots_hit, 1)
        self.assertEqual(collection.metrics.chain_joins, 1)
        self.assertEqual(collection.metrics.clears, 1)
        self.assertEqual(collection.metrics.final_score, 42)
        self.assertEqual(len(collection.sha256), 64)

    def test_fails_closed_on_config_or_pointer_identity_mismatch(self) -> None:
        spec = PointerActionSpec()
        wrong = ReplayEvidenceIdentity(
            source_revision="deadbee",
            replay_sha256=_DIGEST,
            runtime_sha256=_DIGEST,
            config_hash=18,
            observation_schema_sha256=TEACHER_V1.sha256,
            pointer_spec_sha256=spec.sha256,
        )
        with self.assertRaisesRegex(RuntimeError, "config identity"):
            collect_replay_steering_supervision(
                _FakeReplayEnv(), (), seed=0, identity=wrong, pointer_spec=spec
            )
        other_spec = PointerActionSpec(y_radius_offsets=(1.0,))
        with self.assertRaisesRegex(ValueError, "pointer action identity"):
            collect_replay_steering_supervision(
                _FakeReplayEnv(),
                (),
                seed=0,
                identity=self.identity(spec),
                pointer_spec=other_spec,
            )

    def test_collection_owns_and_hashes_training_observations(self) -> None:
        spec = PointerActionSpec()
        collection = collect_replay_steering_supervision(
            _FakeReplayEnv(),
            (
                ReplayInputFrame(0, False, False, 0, 0),
                ReplayInputFrame(1, False, False, 0, 0),
                ReplayInputFrame(2, False, True, 180, 230),
                ReplayInputFrame(3, False, False, 180, 230),
            ),
            seed=5,
            identity=self.identity(spec),
            pointer_spec=spec,
        )
        external = _observation(2)
        rebound = replace(collection, shot_observations=(external,))
        digest = rebound.sha256
        external["score"] = 999
        external["bodies"][0]["x"] = 999.0
        self.assertEqual(rebound.sha256, digest)
        self.assertNotEqual(rebound.identity.sha256, rebound.sha256)


if __name__ == "__main__":
    unittest.main()
