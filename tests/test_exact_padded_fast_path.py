from __future__ import annotations

import ctypes
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import ActionKind  # noqa: E402
from irisu_env.exact_ipc import (  # noqa: E402
    _BODY,
    _EVENT_GENERATION,
    _OBSERVATION_HEADER,
    _TRANSITION,
    _pack_step_request,
)
from irisu_env.padded import (  # noqa: E402
    ExactPaddedTransition,
    _decode_exact_transition,
)


class ExactPaddedFastPathTests(unittest.TestCase):
    def test_canonical_action_encoding_matches_generic_enum_path(self) -> None:
        canonical = _pack_step_request(2, 321.5, -17.25, 0xFFFFFFFF, True)
        generic = _pack_step_request(
            ActionKind.STRONG_SHOT, 321.5, -17.25, 0xFFFFFFFF, True
        )
        self.assertEqual(canonical, generic)

    def test_transition_suffix_copy_preserves_every_wire_field(self) -> None:
        header = _OBSERVATION_HEADER.pack(
            7,
            -11,
            2993,
            40_000,
            2,
            130.0,
            120.0,
            320.0,
            250.0,
            120.0,
            370.0,
            3,
            4,
            96,
            5,
            1,
            1,
            0,
            1,
            0,
        )
        body = _BODY.pack(
            1,
            -2,
            3,
            4.0,
            -5.0,
            6.0,
            -7.0,
            8.0,
            -9.0,
            10.0,
            11,
            -12,
            13,
            14,
            2,
            1,
            4,
            0,
        )
        suffix = (
            -15,
            16,
            17,
            18,
            -19,
            20,
            -21,
            22,
            23,
            24,
            25,
            26,
            1,
            0,
            1,
            0,
        )
        payload = header + body + _TRANSITION.pack(*suffix) + _EVENT_GENERATION.pack(27)
        destination = ExactPaddedTransition()

        transition, generation = _decode_exact_transition(payload, destination)

        self.assertEqual(generation, 27)
        self.assertEqual(transition.observation.tick, 7)
        self.assertEqual(transition.observation.body_count, 1)
        self.assertEqual(transition.observation.bodies[0].id, 11)
        self.assertEqual(
            tuple(getattr(transition, name) for name, _ in transition._fields_[1:]),
            suffix,
        )

    def test_transition_suffix_layout_is_contiguous(self) -> None:
        self.assertEqual(
            ctypes.sizeof(ExactPaddedTransition)
            - ExactPaddedTransition.reward.offset,
            _TRANSITION.size,
        )


if __name__ == "__main__":
    unittest.main()
