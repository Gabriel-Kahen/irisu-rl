from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "replays"
SPEC = importlib.util.spec_from_file_location("inspect_rpy", ROOT / "tools" / "inspect-rpy.py")
assert SPEC and SPEC.loader
inspect_rpy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inspect_rpy
SPEC.loader.exec_module(inspect_rpy)


def fixture(name: str) -> bytes:
    return bytes.fromhex((FIXTURES / name).read_text(encoding="ascii"))


class ReplayParserTests(unittest.TestCase):
    def test_padded_header_and_boundary_fields(self) -> None:
        replay = inspect_rpy.parse_replay(fixture("padded.rpy.hex"))

        self.assertEqual(replay.layout, "padded")
        self.assertEqual(replay.frame_offset, 52)
        self.assertEqual(replay.padding, bytes(32))
        self.assertEqual(
            replay.header,
            inspect_rpy.ReplayHeader(-123456, 17, 4321, 6, 0),
        )
        self.assertEqual(len(replay.frames), 4)

        maximum = replay.frames[1]
        self.assertTrue(maximum.left)
        self.assertFalse(maximum.right)
        self.assertEqual((maximum.x, maximum.y), (1023, 511))
        self.assertEqual(maximum.reserved, 0x5A7)

        simultaneous = replay.frames[3]
        self.assertTrue(simultaneous.left and simultaneous.right)
        self.assertEqual((simultaneous.x, simultaneous.y), (640, 480))
        self.assertEqual(simultaneous.reserved, 0x7FF)

    def test_legacy_header_and_frames(self) -> None:
        replay = inspect_rpy.parse_replay(fixture("legacy.rpy.hex"))

        self.assertEqual(replay.layout, "legacy")
        self.assertEqual(replay.frame_offset, 20)
        self.assertEqual(replay.padding, b"")
        self.assertEqual(replay.header.seed, 0x12345678)
        self.assertEqual(replay.header.final_score, 12_345)
        self.assertEqual(len(replay.frames), 3)
        self.assertEqual((replay.frames[0].x, replay.frames[0].y), (3, 88))

    def test_frame_round_trip_preserves_all_fields(self) -> None:
        word = inspect_rpy.encode_frame(
            left=True,
            right=True,
            x=inspect_rpy.X_MASK,
            y=inspect_rpy.Y_MASK,
            reserved=inspect_rpy.RESERVED_MASK,
        )
        frame = inspect_rpy.decode_frame(word)

        self.assertEqual(frame.raw_word, 0xFFFFFFFF)
        self.assertEqual((frame.x, frame.y), (1023, 511))
        self.assertEqual(frame.reserved, 0x7FF)
        self.assertTrue(frame.left and frame.right)

    def test_report_counts_clicks_and_reserved_bits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.rpy"
            path.write_bytes(fixture("padded.rpy.hex"))
            report = inspect_rpy.inspect(path, include_frames=True)

        self.assertEqual(report["frame_count"], 4)
        self.assertEqual(report["input_stats"]["left_click_frames"], 2)
        self.assertEqual(report["input_stats"]["right_click_frames"], 2)
        self.assertEqual(report["input_stats"]["nonzero_reserved_frames"], 2)
        self.assertEqual(report["frames"][1]["reserved"], 0x5A7)

    def test_malformed_sizes_are_rejected(self) -> None:
        cases = (
            (fixture("malformed-header.rpy.hex"), "auto"),
            (fixture("malformed-frame.rpy.hex"), "legacy"),
            (bytes(20), "padded"),
            (bytes(53), "padded"),
        )
        for data, layout in cases:
            with self.subTest(size=len(data), layout=layout):
                with self.assertRaises(ValueError):
                    inspect_rpy.parse_replay(data, layout)

    def test_coordinate_and_reserved_ranges_are_enforced(self) -> None:
        for values in ({"x": 1024}, {"y": 512}, {"reserved": 2048}, {"x": -1}):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    inspect_rpy.encode_frame(**values)

    def test_unknown_layout_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inspect_rpy.parse_replay(fixture("legacy.rpy.hex"), "future")

    def test_unknown_mode_is_rejected(self) -> None:
        data = bytearray(fixture("legacy.rpy.hex"))
        data[16:20] = (2).to_bytes(4, "little", signed=True)
        with self.assertRaisesRegex(ValueError, "unsupported replay mode"):
            inspect_rpy.parse_replay(data)


if __name__ == "__main__":
    unittest.main()
