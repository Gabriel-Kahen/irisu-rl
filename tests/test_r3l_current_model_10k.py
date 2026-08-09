from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmarks/rl_r3l_current_model_10k.py"
SPEC = importlib.util.spec_from_file_location("r3l_exhibition_test_target", SOURCE)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


class Kind:
    WAIT = 0
    WEAK_SHOT = 1
    STRONG_SHOT = 2
    BOTH_SHOTS = 3

    @classmethod
    def parse(cls, value: object) -> int:
        return int(value)


class Action:
    def __init__(self, kind: int, x: float = 0.0, y: float = 0.0) -> None:
        self.kind = kind
        self.cursor_x = x
        self.cursor_y = y
        self.wait_ticks = 1

    @classmethod
    def wait(cls, _: int = 1) -> "Action":
        return cls(Kind.WAIT)

    @classmethod
    def weak(cls, x: int, y: int) -> "Action":
        return cls(Kind.WEAK_SHOT, x, y)

    @classmethod
    def strong(cls, x: int, y: int) -> "Action":
        return cls(Kind.STRONG_SHOT, x, y)

    @classmethod
    def both(cls, x: int, y: int) -> "Action":
        return cls(Kind.BOTH_SHOTS, x, y)


CORE = types.SimpleNamespace(
    JOINT=types.SimpleNamespace(ActionKind=Kind, Action=Action)
)


class ExhibitionHelpersTests(unittest.TestCase):
    def test_action_words_round_trip(self) -> None:
        actions = (
            Action.wait(),
            Action.weak(123, 45),
            Action.strong(640, 480),
            Action.both(1023, 511),
        )
        for action in actions:
            word = RUNNER.encode_action_word(CORE, action)
            decoded = RUNNER.decode_action_word(CORE, word)
            self.assertEqual(decoded.kind, action.kind)
            if action.kind != Kind.WAIT:
                self.assertEqual(decoded.cursor_x, action.cursor_x)
                self.assertEqual(decoded.cursor_y, action.cursor_y)

    def test_nonintegral_shot_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "integer pixels"):
            RUNNER.encode_action_word(CORE, Action.strong(1.5, 2))

    def test_out_of_range_shot_rejected(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "coordinate fields"):
            RUNNER.encode_action_word(CORE, Action.weak(1024, 0))

    def test_self_hash_round_trip_and_tamper(self) -> None:
        value = RUNNER._with_sha({"schema": "example", "value": 3})
        RUNNER._verify_self_hash(value, "example")
        value["value"] = 4
        with self.assertRaisesRegex(RuntimeError, "self-hash"):
            RUNNER._verify_self_hash(value, "example")

    def test_frozen_source_identity_is_current(self) -> None:
        identity = RUNNER._source_identity()
        RUNNER._verify_self_hash(identity, "source")
        self.assertEqual(
            identity["inherited_source_identity_sha256"],
            RUNNER.EXPECTED_SCREEN_ID,
        )


if __name__ == "__main__":
    unittest.main()
