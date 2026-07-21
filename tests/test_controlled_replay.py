from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import NativeError, find_library  # noqa: E402


try:
    LIBRARY = find_library()
except NativeError:
    LIBRARY = None


def load_generator():
    name = "irisu_generate_controlled_rpy"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools" / "generate-controlled-rpy.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


GENERATOR = load_generator()


class ControlledReplayTests(unittest.TestCase):
    def test_known_ejection_probe_is_byte_exact(self) -> None:
        data = GENERATOR.build_replay(
            seed=17,
            frame_count=80,
            shots=[GENERATOR.Shot("strong", 4, 97, 380)],
        )
        self.assertEqual(len(data), 372)
        self.assertEqual(
            hashlib.sha256(data).hexdigest(),
            "09cfa4cfe939969af1177e97c85036c7c1a94d59cdbabde9b4a35e161d8c56c6",
        )
        self.assertEqual(data[:20], GENERATOR.HEADER.pack(17, 0, 0, 0, 0))
        self.assertEqual(data[20:52], bytes(32))

    @unittest.skipIf(LIBRARY is None, "build the native shared library before score probe test")
    def test_score_preset_is_byte_exact_and_reaches_original_oracle(self) -> None:
        data, oracle = GENERATOR.build_score_probe(library_path=LIBRARY)
        self.assertEqual(len(data), 2_132)
        self.assertEqual(
            hashlib.sha256(data).hexdigest(), GENERATOR.SCORE_PROBE_SHA256
        )
        self.assertEqual(
            data[:20], GENERATOR.HEADER.pack(GENERATOR.SCORE_PROBE_SEED, 0, 0, 0, 0)
        )
        self.assertEqual(oracle["score_events"], [(304, 8), (304, 8)])
        self.assertEqual(oracle["clear_actors"], [10, 12])
        self.assertEqual(
            oracle["click_counts"], {"weak": 0, "strong": 4, "both": 0}
        )
        self.assertEqual(
            oracle["final"],
            {
                "tick": 520,
                "score": 16,
                "gauge": 3_180,
                "level": 1,
                "highest_chain": 2,
                "qualifying_clear_count": 1,
            },
        )
        self.assertEqual(oracle["state_hash"], GENERATOR.SCORE_PROBE_STATE_HASH)

    @unittest.skipIf(LIBRARY is None, "build the native shared library before score probe test")
    def test_generator_cli_finds_the_repo_package_outside_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "score.rpy"
            environment = os.environ.copy()
            environment.pop("PYTHONPATH", None)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools" / "generate-controlled-rpy.py"),
                    str(output),
                    "--preset",
                    GENERATOR.SCORE_PROBE_NAME,
                    "--library",
                    str(LIBRARY),
                ],
                cwd=directory,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(completed.stdout)
            self.assertEqual(report["sha256"], GENERATOR.SCORE_PROBE_SHA256)
            self.assertEqual(
                hashlib.sha256(output.read_bytes()).hexdigest(), report["sha256"]
            )

    def test_button_levels_merge_only_at_identical_coordinates(self) -> None:
        data = GENERATOR.build_replay(
            seed=1,
            frame_count=2,
            shots=[
                GENERATOR.Shot("weak", 1, 300, 380),
                GENERATOR.Shot("strong", 1, 300, 380),
            ],
        )
        self.assertEqual(int.from_bytes(data[56:60], "little") & 3, 3)
        with self.assertRaisesRegex(ValueError, "conflicting coordinates"):
            GENERATOR.build_replay(
                seed=1,
                frame_count=2,
                shots=[
                    GENERATOR.Shot("weak", 1, 300, 380),
                    GENERATOR.Shot("strong", 1, 301, 380),
                ],
            )

    def test_invalid_ranges_are_rejected(self) -> None:
        for kwargs in (
            {"seed": -1, "frame_count": 1, "shots": []},
            {"seed": 2**32, "frame_count": 1, "shots": []},
            {"seed": 0, "frame_count": 0, "shots": []},
            {
                "seed": 0,
                "frame_count": 1,
                "shots": [GENERATOR.Shot("strong", 1, 0, 0)],
            },
        ):
            with self.assertRaises(ValueError):
                GENERATOR.build_replay(**kwargs)

    def test_uint32_seed_is_written_as_the_signed_replay_bit_pattern(self) -> None:
        data = GENERATOR.build_replay(seed=0xFFFF_FFFF, frame_count=1, shots=[])
        self.assertEqual(GENERATOR.HEADER.unpack_from(data)[0], -1)


if __name__ == "__main__":
    unittest.main()
