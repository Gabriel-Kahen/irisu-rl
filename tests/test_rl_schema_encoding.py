from __future__ import annotations

import unittest
import json
from pathlib import Path

import numpy as np

from irisu_env.native import PaddedObservation
from irisu_rl.encoding import ActorTrackEncoder, EncodedBatch, TeacherStateEncoder
from irisu_rl.schema import ACTOR_VISION_V1, PROHIBITED_ACTOR_FIELDS, TEACHER_V1


def padded_fixture() -> PaddedObservation:
    value = PaddedObservation()
    value.tick = 17
    value.score = -30
    value.gauge = 250
    value.gauge_max = 1000
    value.qualifying_clear_count = 2
    value.level = 3
    value.active_colors = 4
    value.spawn_interval_ticks = 31
    value.highest_chain = 5
    value.body_count = 2
    value.bodies[0].id = 19
    value.bodies[0].kind = 0
    value.bodies[0].shape = 1
    value.bodies[0].lifecycle = 0
    value.bodies[0].color = 2
    value.bodies[0].x = 100
    value.bodies[0].y = 200
    value.bodies[0].vx = 2
    value.bodies[0].vy = 3
    value.bodies[0].angle = 0.25
    value.bodies[0].size = 40
    value.bodies[1].id = 5
    value.bodies[1].kind = 1
    value.bodies[1].shape = 0
    value.bodies[1].lifecycle = 2
    value.bodies[1].color = -1
    value.bodies[1].x = 300
    value.bodies[1].y = 400
    value.bodies[1].vx = 20
    value.bodies[1].vy = -10
    value.bodies[1].size = 20
    return value


class SchemaEncodingTests(unittest.TestCase):
    def test_encoded_batch_rejects_wrong_model_dtypes(self) -> None:
        valid = TeacherStateEncoder().encode([padded_fixture()])
        with self.assertRaisesRegex(ValueError, "float32"):
            EncodedBatch(
                valid.global_features.astype(np.float64),
                valid.body_features,
                valid.body_mask,
                valid.source_tick,
                valid.health_flags,
                valid.schema,
            )

    def test_schema_identities_are_stable_and_actor_excludes_privileged_fields(
        self,
    ) -> None:
        self.assertRegex(ACTOR_VISION_V1.sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(TEACHER_V1.sha256, r"^[0-9a-f]{64}$")
        names = set(ACTOR_VISION_V1.global_features + ACTOR_VISION_V1.body_features)
        self.assertFalse(names & PROHIBITED_ACTOR_FIELDS)
        root = Path(__file__).resolve().parents[1]
        for schema in (ACTOR_VISION_V1, TEACHER_V1):
            manifest = json.loads(
                (
                    root / "configs" / "rl" / "schemas" / f"{schema.version}.json"
                ).read_text()
            )
            digest = manifest.pop("sha256")
            self.assertEqual(manifest, schema.manifest())
            self.assertEqual(digest, schema.sha256)

    def test_teacher_dict_and_padded_parity_and_owned_tail(self) -> None:
        observation = padded_fixture()
        encoder = TeacherStateEncoder()
        typed = encoder.encode([observation])
        dictionary = encoder.encode([observation.to_dict()])
        np.testing.assert_array_equal(typed.global_features, dictionary.global_features)
        np.testing.assert_array_equal(typed.body_features, dictionary.body_features)
        np.testing.assert_array_equal(typed.body_mask, dictionary.body_mask)
        saved = typed.body_features.copy()
        observation.bodies[0].x = -99999
        observation.bodies[2].x = float("nan")
        np.testing.assert_array_equal(typed.body_features, saved)
        self.assertEqual(int(typed.body_mask.sum()), 2)

    def test_adversarial_teacher_sort_and_unknown_shape_parity(self) -> None:
        encoder = TeacherStateEncoder()
        close = padded_fixture()
        close.bodies[0].x = 0.0
        close.bodies[0].y = 1.00002
        close.bodies[1].x = 100.0
        close.bodies[1].y = 1.00001
        typed = encoder.encode([close])
        dictionary = encoder.encode([close.to_dict()])
        np.testing.assert_array_equal(typed.body_features, dictionary.body_features)

        unknown = padded_fixture()
        unknown.body_count = 1
        unknown.bodies[0].shape = 255
        unknown.bodies[0].angle = 0.75
        mapping = padded_fixture().to_dict()
        mapping["bodies"] = [
            {
                "id": int(unknown.bodies[0].id),
                "kind": "piece",
                "shape": "unknown",
                "lifecycle": "scripted_falling",
                "color": int(unknown.bodies[0].color),
                "x": float(unknown.bodies[0].x),
                "y": float(unknown.bodies[0].y),
                "vx": float(unknown.bodies[0].vx),
                "vy": float(unknown.bodies[0].vy),
                "angle": float(unknown.bodies[0].angle),
                "angular_velocity": float(unknown.bodies[0].angular_velocity),
                "size": float(unknown.bodies[0].size),
                "chain_id": int(unknown.bodies[0].chain_id),
                "projectile_hits": int(unknown.bodies[0].projectile_hits),
                "age_ticks": int(unknown.bodies[0].age_ticks),
                "remaining_lifetime": int(unknown.bodies[0].remaining_lifetime),
                "rot_timer": int(unknown.bodies[0].rot_timer),
            }
        ]
        unknown_typed = encoder.encode([unknown])
        unknown_dict = encoder.encode([mapping])
        np.testing.assert_array_equal(
            unknown_typed.body_features, unknown_dict.body_features
        )

    def test_velocity_conversion_and_shape_symmetry(self) -> None:
        encoded = TeacherStateEncoder().encode([padded_fixture()])
        names = TEACHER_V1.body_features
        rows = encoded.body_features[0, :2]
        # Sorting is canonical by y then x: falling body is first.
        self.assertAlmostEqual(
            rows[0, names.index("velocity_x_display_per_second_scaled")], 0.1
        )
        self.assertAlmostEqual(
            rows[1, names.index("velocity_x_display_per_second_scaled")], 0.2
        )
        self.assertEqual(rows[1, names.index("orientation_valid")], 0.0)

    def test_actor_requires_causal_records_and_selects_overflow_deterministically(
        self,
    ) -> None:
        encoder = ActorTrackEncoder()
        with self.assertRaisesRegex(TypeError, "causal mapping"):
            encoder.encode([padded_fixture()])  # type: ignore[list-item]
        tracks = [
            {
                "kind": "piece",
                "shape": "box",
                "color": index % 6,
                "lifecycle": "confirmed",
                "effect_x": float(index % 20) * 20,
                "effect_y": float(index // 20) * 20,
                "vx_display_per_second": 0.0,
                "vy_display_per_second": 0.0,
                "size": 10,
                "confidence": 0.5 + index / 1000,
            }
            for index in range(210)
        ]
        first = encoder.encode([{"tracks": tracks}])
        second = encoder.encode([{"tracks": list(reversed(tracks))}])
        np.testing.assert_array_equal(first.body_features, second.body_features)
        self.assertEqual(int(first.body_mask.sum()), 196)
        self.assertEqual(int(first.health_flags[0]) & 1, 1)

    def test_actor_requires_effect_time_units_and_excludes_fully_occluded_tracks(
        self,
    ) -> None:
        encoder = ActorTrackEncoder()
        with self.assertRaisesRegex(ValueError, "effect-time"):
            encoder.encode([{"tracks": [{"x": 10, "y": 20}]}])
        track = {
            "effect_x": 100.0,
            "effect_y": 100.0,
            "vx_display_per_second": 0.0,
            "vy_display_per_second": 0.0,
            "kind": "piece",
            "shape": "unknown",
            "color": 1,
            "lifecycle": "ambiguous",
            "size": 20,
            "confidence": 1.0,
            "occluded_probability": 1.0,
        }
        self.assertEqual(int(encoder.encode([{"tracks": [track]}]).body_mask.sum()), 0)
        track["occluded_probability"] = 0.0
        encoded = encoder.encode([{"tracks": [track]}])
        names = encoded.schema.body_features
        self.assertEqual(
            encoded.body_features[0, 0, names.index("orientation_valid")], 0
        )
        tied = [
            {**track, "kind": "piece", "shape": "box", "color": 1},
            {**track, "kind": "projectile", "shape": "circle", "color": -1},
        ]
        forward = encoder.encode([{"tracks": tied}])
        reverse = encoder.encode([{"tracks": list(reversed(tied))}])
        np.testing.assert_array_equal(forward.body_features, reverse.body_features)
        probabilistic = {
            **track,
            "kind": "unknown",
            "kind_probabilities": np.array([1.0, 0.0, 0.0, 0.0]),
        }
        probability_tensor = encoder.encode([{"tracks": [probabilistic]}])
        self.assertEqual(probability_tensor.body_features[0, 0, 0], 1.0)
        for unknown_shape in (None, -1, "ambiguous", "typo"):
            uncertain = {**track, "shape": unknown_shape}
            tensor = encoder.encode([{"tracks": [uncertain]}])
            self.assertEqual(
                tensor.body_features[0, 0, names.index("orientation_valid")], 0
            )
        bridge = encoder.encode(
            [
                {
                    "global": {
                        "injection_acknowledged": 1.0,
                        "previous_effect_confirmed": 0.2,
                        "previous_effect_missed": 0.8,
                    },
                    "tracks": [],
                }
            ]
        )
        global_names = bridge.schema.global_features
        self.assertEqual(
            bridge.global_features[0, global_names.index("injection_acknowledged")],
            1.0,
        )
        self.assertAlmostEqual(
            bridge.global_features[0, global_names.index("previous_effect_confirmed")],
            0.2,
        )
        self.assertAlmostEqual(
            bridge.global_features[0, global_names.index("previous_effect_missed")],
            0.8,
        )

    def test_explicit_teacher_display_velocity_is_not_converted_twice(self) -> None:
        body = padded_fixture().bodies[1].to_dict()
        body.pop("vx")
        body.pop("vy")
        body["vx_display_per_second"] = 100.0
        body["vy_display_per_second"] = -50.0
        observation = padded_fixture().to_dict()
        observation["bodies"] = [body]
        encoded = TeacherStateEncoder().encode([observation])
        names = encoded.schema.body_features
        self.assertAlmostEqual(
            encoded.body_features[
                0, 0, names.index("velocity_x_display_per_second_scaled")
            ],
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
