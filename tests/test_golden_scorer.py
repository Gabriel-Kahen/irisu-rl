from __future__ import annotations

import copy
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from typing import Any
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "score_golden", ROOT / "tools" / "score-golden.py"
)
assert SPEC and SPEC.loader
score_golden = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = score_golden
SPEC.loader.exec_module(score_golden)

from irisu_env import ExactWorkerNotFoundError, find_exact_worker


TARGET = dict(score_golden.CANONICAL_TARGET)

EVENTS = {
    "match": {"kind": "confirmed", "detail": "normal burst qualified"},
    "rot": {"kind": "rotten", "detail": "normal rot timer"},
    "chain": {"kind": "chain_joined", "detail": "normal group membership"},
    "ejection": {
        "kind": "ejected",
        "detail": "normal strict out-of-bounds guard",
    },
    "orb": {"kind": "cleared", "detail": "special color clear"},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _captured_library(path: Path) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return {
        "status": "captured",
        "path": str(resolved),
        "bytes": stat.st_size,
        "sha256": _sha(resolved),
        "file_identity": {
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
        },
        "mapped_identity": {
            "device": score_golden._mount_device(os.getpid(), str(resolved)),
            "inode": stat.st_ino,
        },
    }


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _minimal_png(*, scanline: bytes = b"\x00\x00") -> bytes:
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)),
            _png_chunk(b"IDAT", zlib.compress(scanline)),
            _png_chunk(b"IEND", b""),
        )
    )


class FakeEnv:
    categories: dict[int, str] = {}
    missing_event_seeds: set[int] = set()
    wrong_scalar_seeds: set[int] = set()
    wrong_trajectory_seeds: set[int] = set()

    def __init__(self, **_: object) -> None:
        self.seed = 0
        self.tick = 0
        self.score = 0
        self.gauge = 1000
        self.level = 1

    def __enter__(self) -> FakeEnv:
        return self

    def __exit__(self, *_: object) -> None:
        pass

    @property
    def build_info(self) -> dict[str, object]:
        return {"implementation": "synthetic-golden-test"}

    def config_hash(self) -> int:
        return int(TARGET["clone_config_u64"], 16)

    def _observation(self) -> dict[str, Any]:
        x, y = (30.0, 40.0) if self.seed in self.wrong_trajectory_seeds else (3.0, 4.0)
        return {
            "tick": self.tick,
            "score": self.score,
            "gauge": self.gauge,
            "level": self.level,
            "terminated": False,
            "truncated": False,
            "bodies": [{"id": 1, "x": x, "y": y}],
        }

    def reset(self, *, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        self.seed = seed
        self.tick = 0
        self.score = 0
        self.gauge = 1000
        self.level = 1
        return self._observation(), {}

    def step(
        self, _: object
    ) -> tuple[dict[str, Any], int, bool, bool, dict[str, Any]]:
        self.tick += 1
        self.score = 11 if self.seed in self.wrong_scalar_seeds else 10
        self.gauge = 999
        category = self.categories[self.seed]
        events: list[dict[str, Any]] = []
        if self.tick == 1 and self.seed not in self.missing_event_seeds:
            event = EVENTS[category]
            events.append(
                {
                    "tick": self.tick,
                    "kind_name": event["kind"],
                    "a": 1,
                    "b": 0,
                    "value": 0,
                    "detail": event["detail"],
                }
            )
        return self._observation(), self.score, False, False, {
            "events": events,
            "invalid_action": False,
        }


class CorpusBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.next_seed = 1

    def scenario(
        self,
        scenario_id: str,
        category: str,
        *,
        metadata_status: str = "valid_for_mechanics_calibration",
        measurement_status: str = "observed",
        trajectory_tolerance: float | None = 0.500001,
        event_min_count: int = 1,
        event_max_count: int = 1,
    ) -> dict[str, Any]:
        seed = self.next_seed
        self.next_seed += 1
        FakeEnv.categories[seed] = category
        bundle = self.root / scenario_id
        (bundle / "frames").mkdir(parents=True)

        replay = struct.pack("<5i", seed, 1, 10, 1, 0) + bytes(32)
        replay += struct.pack(
            "<I", score_golden.INSPECT_RPY.encode_frame(x=300, y=200)
        ) * score_golden.MIN_TRAJECTORY_HORIZON_FRAMES
        replay_path = bundle / "result.rpy"
        replay_path.write_bytes(replay)
        replay_sha = _sha(replay_path)

        event = EVENTS[category]
        expected: dict[str, Any] = {
            "events": [
                {
                    "kind": event["kind"],
                    "detail": event["detail"],
                    "min_count": event_min_count,
                    "max_count": event_max_count,
                }
            ],
            "scalar_transition": {
                "from_frame": -1,
                "to_frame": score_golden.MIN_TRAJECTORY_HORIZON_FRAMES - 1,
                "score": {"before": 0, "after": 10, "delta": 10},
                "gauge": {"before": 1000, "after": 999, "delta": -1},
                "level": {"before": 1, "after": 1, "delta": 0},
            },
        }
        if trajectory_tolerance is not None:
            expected["trajectories"] = [
                {
                    "frame": score_golden.MIN_TRAJECTORY_HORIZON_FRAMES - 1,
                    "body_id": 1,
                    "x": 3.3,
                    "y": 4.4,
                    "tolerance": trajectory_tolerance,
                }
            ]

        metadata = {
            "experiment_id": scenario_id,
            "status": metadata_status,
            "game": {
                "version": TARGET["game_version"],
                **{key: TARGET[key] for key in score_golden._HASH_KEYS},
            },
            "run": {"final_replay_sha256": replay_sha},
        }
        measurements = {
            "valid_mechanics_measurements": [
                {
                    "id": f"{scenario_id}-measurement",
                    "status": measurement_status,
                    "category": category,
                    "repeat_count": 2,
                    "replay_window": {
                        "first_frame": 0,
                        "last_frame": score_golden.MIN_TRAJECTORY_HORIZON_FRAMES - 1,
                    },
                    "oracle": expected,
                }
            ]
        }
        (bundle / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True), encoding="utf-8"
        )
        (bundle / "measurements.json").write_text(
            json.dumps(measurements, sort_keys=True), encoding="utf-8"
        )
        (bundle / "actions.jsonl").write_text(
            '{"sequence":1,"action":"wait","result":"recorded"}\n',
            encoding="utf-8",
        )
        (bundle / "frames" / "000.png").write_bytes(_minimal_png())
        (bundle / "notes.md").write_text("Synthetic evidence fixture.\n", encoding="utf-8")

        evidence_files = [
            ("metadata", "metadata.json"),
            ("measurements", "measurements.json"),
            ("actions", "actions.jsonl"),
            ("replay", "result.rpy"),
            ("frame", "frames/000.png"),
            ("notes", "notes.md"),
        ]
        return {
            "id": scenario_id,
            "category": category,
            "evidence": {
                "status": "observed",
                "experiment_id": scenario_id,
                "measurement_id": f"{scenario_id}-measurement",
                "bundle": scenario_id,
                "files": [
                    {"role": role, "path": relative, "sha256": _sha(bundle / relative)}
                    for role, relative in evidence_files
                ],
            },
            "replay": {
                "path": "result.rpy",
                "sha256": replay_sha,
                "layout": "padded",
                "first_frame": 0,
                "last_frame": score_golden.MIN_TRAJECTORY_HORIZON_FRAMES - 1,
            },
            "expected": expected,
            "_seed": seed,
        }

    def manifest(self, scenarios: list[dict[str, Any]], name: str = "manifest.json") -> Path:
        clean = []
        for scenario in scenarios:
            item = dict(scenario)
            item.pop("_seed")
            clean.append(item)
        path = self.root / name
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "threshold_percent": 95,
                    "target": TARGET,
                    "scenarios": clean,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return path


class GoldenScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeEnv.categories.clear()
        FakeEnv.missing_event_seeds.clear()
        FakeEnv.wrong_scalar_seeds.clear()
        FakeEnv.wrong_trajectory_seeds.clear()

    def test_complete_five_category_corpus_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            scenarios = [
                builder.scenario(category, category)
                for category in score_golden.CATEGORIES
            ]
            manifest = builder.manifest(scenarios)
            manifest_sha = _sha(manifest)
            report = score_golden.score_manifest(
                manifest, env_factory=FakeEnv
            )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["manifest"]["sha256"], manifest_sha)
        self.assertTrue(report["gate"]["passed"])
        self.assertTrue(report["gate"]["one_vote_per_scenario"])
        self.assertTrue(report["gate"]["one_vote_per_unique_source_measurement"])
        self.assertTrue(report["gate"]["one_vote_per_nonoverlapping_source_probe"])
        self.assertEqual(report["gate"]["overall"]["total"], 5)
        for category in score_golden.CATEGORIES:
            self.assertEqual(report["gate"]["categories"][category]["total"], 1)
        self.assertEqual(report["scenarios"][0]["trajectories"]["status"], "evaluated")
        self.assertTrue(report["gate"]["trajectory_coverage_complete"])
        self.assertFalse(report["scope"]["full_clone_md_fidelity_gate_met"])
        self.assertEqual(report["clone_library"]["status"], "unavailable")
        evidence = report["scenarios"][0]["evidence"]
        self.assertRegex(evidence["source_probe_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(evidence["reported_repeat_count"], 2)
        self.assertEqual(
            evidence["repeat_count_status"],
            "advisory_not_independently_verified",
        )
        self.assertFalse(evidence["repeat_count_used_for_gate"])

    def test_target_must_match_the_independent_canonical_tuple(self) -> None:
        replacements = {
            "profile": "other-profile",
            "game_version": "v2.03 synthetic target",
            "executable_sha256": "0" * 64,
            "dat_dxa_sha256": "1" * 64,
            "config_sha256": "2" * 64,
            "box2d_sha256": "3" * 64,
            "clone_config_u64": "0x000000000000cafe",
        }
        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            scenario = builder.scenario("canonical-target", "match")
            canonical_path = builder.manifest([scenario])
            canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
            for index, (key, replacement) in enumerate(replacements.items()):
                with self.subTest(key=key):
                    altered = copy.deepcopy(canonical)
                    altered["target"][key] = replacement
                    path = Path(directory) / f"altered-target-{index}.json"
                    path.write_text(json.dumps(altered), encoding="utf-8")
                    with self.assertRaisesRegex(
                        score_golden.GoldenError,
                        "canonical v2.03 target",
                    ):
                        score_golden.validate_manifest(path)

    def test_each_vote_requires_a_unique_source_measurement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            original = builder.scenario("source-measurement", "match")
            duplicate = copy.deepcopy(original)
            duplicate["id"] = "different-scenario-label"
            with self.assertRaisesRegex(
                score_golden.GoldenError,
                "measurement identities must be unique",
            ):
                score_golden.validate_manifest(
                    builder.manifest([original, duplicate])
                )

    def test_relabeling_a_copied_probe_cannot_add_a_vote(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = CorpusBuilder(root)
            original = builder.scenario("source-probe", "match")
            duplicate = copy.deepcopy(original)
            duplicate["id"] = "renamed-copy"
            duplicate["evidence"]["experiment_id"] = "renamed-experiment"
            duplicate["evidence"]["measurement_id"] = "renamed-measurement"
            duplicate["evidence"]["bundle"] = "renamed-copy"
            shutil.copytree(root / "source-probe", root / "renamed-copy")

            metadata_path = root / "renamed-copy" / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["experiment_id"] = "renamed-experiment"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            measurements_path = root / "renamed-copy" / "measurements.json"
            measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
            measurements["valid_mechanics_measurements"][0]["id"] = (
                "renamed-measurement"
            )
            measurements_path.write_text(json.dumps(measurements), encoding="utf-8")
            for file_entry in duplicate["evidence"]["files"]:
                path = root / "renamed-copy" / file_entry["path"]
                file_entry["sha256"] = _sha(path)

            with self.assertRaisesRegex(
                score_golden.GoldenError,
                "source probe replay windows must not overlap",
            ):
                score_golden.validate_manifest(
                    builder.manifest([original, duplicate])
                )

    def test_reserved_bits_cannot_relabel_the_same_behavioral_probe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = CorpusBuilder(root)
            original = builder.scenario("source-probe", "match")
            duplicate = copy.deepcopy(original)
            duplicate["id"] = "reserved-bit-copy"
            duplicate["evidence"]["experiment_id"] = "reserved-bit-experiment"
            duplicate["evidence"]["measurement_id"] = "reserved-bit-measurement"
            duplicate["evidence"]["bundle"] = "reserved-bit-copy"
            copied_bundle = root / "reserved-bit-copy"
            shutil.copytree(root / "source-probe", copied_bundle)

            replay_path = copied_bundle / "result.rpy"
            replay = bytearray(replay_path.read_bytes())
            frame_offset = score_golden.INSPECT_RPY.PADDED_FRAME_OFFSET
            word = struct.unpack_from("<I", replay, frame_offset)[0]
            struct.pack_into("<I", replay, frame_offset, word | (1 << 21))
            replay_path.write_bytes(replay)
            replay_sha = _sha(replay_path)
            duplicate["replay"]["sha256"] = replay_sha

            metadata_path = copied_bundle / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["experiment_id"] = "reserved-bit-experiment"
            metadata["run"]["final_replay_sha256"] = replay_sha
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            measurements_path = copied_bundle / "measurements.json"
            measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
            measurements["valid_mechanics_measurements"][0]["id"] = (
                "reserved-bit-measurement"
            )
            measurements_path.write_text(json.dumps(measurements), encoding="utf-8")
            for file_entry in duplicate["evidence"]["files"]:
                path = copied_bundle / file_entry["path"]
                file_entry["sha256"] = _sha(path)

            self.assertNotEqual(original["replay"]["sha256"], replay_sha)
            with self.assertRaisesRegex(
                score_golden.GoldenError,
                "source probe replay windows must not overlap",
            ):
                score_golden.validate_manifest(
                    builder.manifest([original, duplicate])
                )

    def test_different_window_ends_cannot_relabel_a_behavioral_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = CorpusBuilder(root)
            original = builder.scenario("longer-probe", "match")
            original_bundle = root / "longer-probe"
            replay_path = original_bundle / "result.rpy"
            replay_path.write_bytes(replay_path.read_bytes() + struct.pack("<I", 0))
            replay_sha = _sha(replay_path)
            original["replay"]["sha256"] = replay_sha
            original["replay"]["last_frame"] = 25
            original["expected"]["scalar_transition"]["to_frame"] = 25
            metadata_path = original_bundle / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["run"]["final_replay_sha256"] = replay_sha
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            measurements_path = original_bundle / "measurements.json"
            measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
            measured = measurements["valid_mechanics_measurements"][0]
            measured["replay_window"]["last_frame"] = 25
            measured["oracle"] = original["expected"]
            measurements_path.write_text(json.dumps(measurements), encoding="utf-8")
            for file_entry in original["evidence"]["files"]:
                file_entry["sha256"] = _sha(original_bundle / file_entry["path"])

            duplicate = copy.deepcopy(original)
            duplicate["id"] = "shorter-copy"
            duplicate["evidence"]["experiment_id"] = "shorter-experiment"
            duplicate["evidence"]["measurement_id"] = "shorter-measurement"
            duplicate["evidence"]["bundle"] = "shorter-copy"
            duplicate["replay"]["last_frame"] = 24
            duplicate["expected"]["scalar_transition"]["to_frame"] = 24
            copied_bundle = root / "shorter-copy"
            shutil.copytree(original_bundle, copied_bundle)
            metadata_path = copied_bundle / "metadata.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["experiment_id"] = "shorter-experiment"
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            measurements_path = copied_bundle / "measurements.json"
            measurements = json.loads(measurements_path.read_text(encoding="utf-8"))
            measured = measurements["valid_mechanics_measurements"][0]
            measured["id"] = "shorter-measurement"
            measured["replay_window"]["last_frame"] = 24
            measured["oracle"] = duplicate["expected"]
            measurements_path.write_text(json.dumps(measurements), encoding="utf-8")
            for file_entry in duplicate["evidence"]["files"]:
                file_entry["sha256"] = _sha(copied_bundle / file_entry["path"])

            with self.assertRaisesRegex(
                score_golden.GoldenError,
                "source probe replay windows must not overlap",
            ):
                score_golden.validate_manifest(
                    builder.manifest([original, duplicate])
                )

    def test_category_vote_requires_a_positive_relevant_event(self) -> None:
        for category in score_golden.CATEGORIES:
            with self.subTest(category=category), tempfile.TemporaryDirectory() as directory:
                builder = CorpusBuilder(Path(directory))
                scenario = builder.scenario(
                    category,
                    category,
                    event_min_count=0,
                    event_max_count=0,
                )
                with self.assertRaisesRegex(
                    score_golden.GoldenError,
                    "positive category-relevant event assertion",
                ):
                    score_golden.validate_manifest(builder.manifest([scenario]))

    def test_every_category_requires_bounded_trajectory_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            scenarios = [
                builder.scenario(
                    category,
                    category,
                    trajectory_tolerance=None if category == "orb" else 0.500001,
                )
                for category in score_golden.CATEGORIES
            ]
            report = score_golden.score_manifest(
                builder.manifest(scenarios), env_factory=FakeEnv
            )

        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["gate"]["trajectory_coverage_complete"])
        self.assertEqual(report["gate"]["trajectory_assertions_by_category"]["orb"], 0)

        with self.assertRaisesRegex(score_golden.GoldenError, "25..50-update horizon"):
            score_golden._validate_trajectory(
                {"frame": 23, "body_id": 1, "x": 0, "y": 0, "tolerance": 1},
                "trajectory",
                first_frame=0,
                last_frame=24,
            )
        with self.assertRaisesRegex(score_golden.GoldenError, "15 pixels"):
            score_golden._validate_trajectory(
                {"frame": 24, "body_id": 1, "x": 0, "y": 0, "tolerance": 15.01},
                "trajectory",
                first_frame=0,
                last_frame=24,
            )

    def test_unobserved_or_invalid_evidence_is_rejected(self) -> None:
        cases = (
            {"metadata_status": "blocked_before_playback"},
            {"metadata_status": "invalid_for_mechanics_calibration"},
            {"measurement_status": "inferred"},
        )
        for index, kwargs in enumerate(cases):
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as directory:
                builder = CorpusBuilder(Path(directory))
                scenario = builder.scenario(f"invalid-{index}", "match", **kwargs)
                with self.assertRaises(score_golden.GoldenError):
                    score_golden.validate_manifest(builder.manifest([scenario]))

    def test_frame_evidence_requires_a_structurally_valid_png(self) -> None:
        valid = _minimal_png()
        bad_crc = bytearray(valid)
        bad_crc[-1] ^= 1
        malformed = {
            "signature only": b"\x89PNG\r\n\x1a\n",
            "bad chunk CRC": bytes(bad_crc),
            "invalid zlib stream": b"".join(
                (
                    b"\x89PNG\r\n\x1a\n",
                    _png_chunk(
                        b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
                    ),
                    _png_chunk(b"IDAT", b"not-zlib"),
                    _png_chunk(b"IEND", b""),
                )
            ),
            "invalid scanline filter": _minimal_png(scanline=b"\x05\x00"),
            "missing IEND": valid[:-12],
        }
        for label, png in malformed.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                builder = CorpusBuilder(root)
                scenario = builder.scenario("bad-png", "match")
                frame_path = root / "bad-png" / "frames" / "000.png"
                frame_path.write_bytes(png)
                frame_entry = next(
                    entry
                    for entry in scenario["evidence"]["files"]
                    if entry["role"] == "frame"
                )
                frame_entry["sha256"] = _sha(frame_path)
                with self.assertRaisesRegex(score_golden.GoldenError, "PNG"):
                    score_golden.validate_manifest(builder.manifest([scenario]))

    def test_bundle_roots_are_confined_and_cannot_be_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden = root / "golden"
            golden.mkdir()
            builder = CorpusBuilder(golden)
            scenario = builder.scenario("captured", "match")
            captures = root / "captures"
            captures.mkdir()
            shutil.move(str(golden / "captured"), str(captures / "captured"))
            scenario["evidence"]["bundle"] = "../captures/captured"
            score_golden.validate_manifest(builder.manifest([scenario]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            golden = root / "golden"
            golden.mkdir()
            builder = CorpusBuilder(golden)
            scenario = builder.scenario("outside", "match")
            sibling = root / "arbitrary"
            sibling.mkdir()
            shutil.move(str(golden / "outside"), str(sibling / "outside"))
            scenario["evidence"]["bundle"] = "../arbitrary/outside"
            with self.assertRaisesRegex(
                score_golden.GoldenError, "sibling captures directory"
            ):
                score_golden.validate_manifest(builder.manifest([scenario]))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder = CorpusBuilder(root)
            scenario = builder.scenario("real-bundle", "match")
            (root / "linked-bundle").symlink_to(
                root / "real-bundle", target_is_directory=True
            )
            scenario["evidence"]["bundle"] = "linked-bundle"
            with self.assertRaisesRegex(score_golden.GoldenError, "symlink"):
                score_golden.validate_manifest(builder.manifest([scenario]))

    def test_manifest_and_evidence_changes_during_scoring_are_rejected(self) -> None:
        for target_kind in ("manifest", "evidence"):
            with self.subTest(target=target_kind), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                builder = CorpusBuilder(root)
                scenarios = [
                    builder.scenario(category, category)
                    for category in score_golden.CATEGORIES
                ]
                manifest = builder.manifest(scenarios)
                target = (
                    manifest
                    if target_kind == "manifest"
                    else root / "match" / "notes.md"
                )

                class MutatingInputEnv(FakeEnv):
                    mutated = False

                    def step(self, action: object) -> tuple[
                        dict[str, Any], int, bool, bool, dict[str, Any]
                    ]:
                        if not type(self).mutated:
                            target.write_bytes(target.read_bytes() + b"\n")
                            type(self).mutated = True
                        return super().step(action)

                with self.assertRaisesRegex(
                    score_golden.GoldenError,
                    "validated input changed during scenario scoring",
                ):
                    score_golden.score_manifest(
                        manifest, env_factory=MutatingInputEnv
                    )

    def test_action_records_require_consistent_increasing_sequences_and_results(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actions.jsonl"
            for sequence_field in ("sequence", "monotonic_sequence"):
                path.write_text(
                    "\n".join(
                        (
                            json.dumps(
                                {
                                    sequence_field: 1,
                                    "action": "capture",
                                    "result": "recorded",
                                }
                            ),
                            json.dumps(
                                {
                                    sequence_field: 3,
                                    "action": "click",
                                    "result": "delivered",
                                }
                            ),
                        )
                    )
                    + "\n",
                    encoding="utf-8",
                )
                score_golden._validate_actions(path.read_bytes(), path)

            invalid_records = {
                "missing action": [{"sequence": 1, "result": "recorded"}],
                "blank action": [
                    {"sequence": 1, "action": "   ", "result": "recorded"}
                ],
                "non-string action": [
                    {"sequence": 1, "action": 1, "result": "recorded"}
                ],
                "missing result": [{"sequence": 1, "action": "capture"}],
                "blank result": [
                    {"sequence": 1, "action": "capture", "result": ""}
                ],
                "non-string result": [
                    {"sequence": 1, "action": "capture", "result": True}
                ],
                "nonpositive sequence": [
                    {"sequence": 0, "action": "capture", "result": "recorded"}
                ],
                "non-integer sequence": [
                    {"sequence": True, "action": "capture", "result": "recorded"}
                ],
                "duplicate sequence": [
                    {"sequence": 2, "action": "capture", "result": "recorded"},
                    {"sequence": 2, "action": "click", "result": "delivered"},
                ],
                "mixed sequence fields": [
                    {"sequence": 1, "action": "capture", "result": "recorded"},
                    {
                        "monotonic_sequence": 2,
                        "action": "click",
                        "result": "delivered",
                    },
                ],
                "both sequence fields": [
                    {
                        "sequence": 1,
                        "monotonic_sequence": 1,
                        "action": "capture",
                        "result": "recorded",
                    }
                ],
            }
            for label, records in invalid_records.items():
                with self.subTest(label=label):
                    path.write_text(
                        "".join(json.dumps(record) + "\n" for record in records),
                        encoding="utf-8",
                    )
                    with self.assertRaises(score_golden.GoldenError):
                        score_golden._validate_actions(path.read_bytes(), path)

    def test_clone_library_change_during_scenario_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "libirisu_clone.fake"
            library.write_bytes(b"before")

            class MutatingLibraryEnv(FakeEnv):
                @property
                def library_path(self) -> str:
                    return str(library)

                def step(self, action: object) -> tuple[
                    dict[str, Any], int, bool, bool, dict[str, Any]
                ]:
                    result = super().step(action)
                    library.write_bytes(b"after")
                    return result

            builder = CorpusBuilder(root)
            scenario = builder.scenario("mutating-library", "match")
            with self.assertRaisesRegex(
                score_golden.GoldenError,
                "library changed during scenario scoring",
            ):
                score_golden.score_manifest(
                    builder.manifest([scenario]), env_factory=MutatingLibraryEnv
                )

    def test_file_backed_clone_library_provenance_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "libirisu_clone.fake"
            library.write_bytes(b"stable synthetic library")

            class FileBackedEnv(FakeEnv):
                @property
                def library_path(self) -> str:
                    return str(library)

            builder = CorpusBuilder(root)
            scenarios = [
                builder.scenario(category, category)
                for category in score_golden.CATEGORIES
            ]
            report = score_golden.score_manifest(
                builder.manifest(scenarios), env_factory=FileBackedEnv
            )

            provenance = report["clone_library"]
            self.assertEqual(provenance["status"], "verified")
            self.assertEqual(provenance["stability_check"], "passed")
            self.assertEqual(provenance["path"], str(library.resolve()))
            self.assertEqual(provenance["bytes"], library.stat().st_size)
            self.assertEqual(provenance["sha256"], _sha(library))
            self.assertEqual(provenance["file_identity"]["inode"], library.stat().st_ino)

    def test_exact_worker_backend_and_provenance_are_stable_across_scenarios(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "irisu-exact-worker.fake"
            worker.write_bytes(b"stable synthetic exact worker")
            worker_sha = _sha(worker)
            exact_library = root / "libirisu_box2d_msvc_exact_multiworld.so"
            exact_library.write_bytes(b"stable synthetic exact library")
            exact_library_sha = _sha(exact_library)
            received: list[dict[str, object]] = []

            class ExactWorkerEnv(FakeEnv):
                next_pid = 100

                def __init__(self, **kwargs: object) -> None:
                    received.append(dict(kwargs))
                    self.worker_pid = type(self).next_pid
                    type(self).next_pid += 1
                    super().__init__()

                @property
                def library_path(self) -> str:
                    return str(worker)

                def exact_library_provenance(self) -> dict[str, object]:
                    return _captured_library(exact_library)

                @property
                def build_info(self) -> dict[str, object]:
                    return {
                        "physics_backend": "exact-msvc9-r58-worker",
                        "worker_backend": "exact-msvc9-r58-multiworld-forward",
                        "worker_executable_sha256": worker_sha,
                        "exact_library_sha256": exact_library_sha.upper(),
                        "worker_pid": self.worker_pid,
                    }

            builder = CorpusBuilder(root)
            scenarios = [
                builder.scenario(category, category)
                for category in score_golden.CATEGORIES
            ]
            report = score_golden.score_manifest(
                builder.manifest(scenarios),
                worker_path=str(worker),
                env_factory=ExactWorkerEnv,
            )

        self.assertEqual(report["status"], "pass")
        self.assertNotIn("worker_pid", report["clone_build"])
        self.assertNotIn("clone_library", report)
        self.assertEqual(
            report["clone_worker"]["artifact_role"],
            "exact_worker_executable",
        )
        self.assertEqual(report["clone_worker"]["sha256"], worker_sha)
        self.assertEqual(
            report["clone_worker"]["linked_exact_library_sha256"],
            exact_library_sha,
        )
        linked_library = report["clone_worker"]["linked_exact_library"]
        self.assertEqual(linked_library["status"], "verified")
        self.assertEqual(linked_library["stability_check"], "passed")
        self.assertEqual(linked_library["path"], str(exact_library.resolve()))
        self.assertEqual(linked_library["sha256"], exact_library_sha)
        self.assertEqual(
            report["clone_build"]["exact_library_sha256"], exact_library_sha
        )
        self.assertEqual(
            [scenario["runtime"]["worker_pid"] for scenario in report["scenarios"]],
            list(range(100, 100 + len(score_golden.CATEGORIES))),
        )
        self.assertEqual(
            received,
            [
                {"physics_backend": "exact", "worker_path": str(worker)}
                for _ in score_golden.CATEGORIES
            ],
        )

    def test_exact_worker_reported_hash_must_match_executable_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "irisu-exact-worker.fake"
            worker.write_bytes(b"stable synthetic exact worker")
            exact_library = root / "libirisu_box2d_msvc_exact_multiworld.so"
            exact_library.write_bytes(b"stable synthetic exact library")

            class MismatchedWorkerEnv(FakeEnv):
                @property
                def library_path(self) -> str:
                    return str(worker)

                def exact_library_provenance(self) -> dict[str, object]:
                    return _captured_library(exact_library)

                @property
                def build_info(self) -> dict[str, object]:
                    return {
                        "physics_backend": "exact-msvc9-r58-worker",
                        "worker_backend": "exact-msvc9-r58-multiworld-forward",
                        "worker_executable_sha256": "0" * 64,
                        "exact_library_sha256": _sha(exact_library),
                        "worker_pid": 100,
                    }

            builder = CorpusBuilder(root)
            scenario = builder.scenario("mismatched-worker", "match")
            with self.assertRaisesRegex(
                score_golden.GoldenError,
                "worker bytes disagree with worker-reported provenance",
            ):
                score_golden.score_manifest(
                    builder.manifest([scenario]),
                    worker_path=str(worker),
                    env_factory=MismatchedWorkerEnv,
                )

    def test_exact_worker_library_provenance_fails_closed(self) -> None:
        for case in ("placeholder", "mismatch", "mutation", "device"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                worker = root / "irisu-exact-worker.fake"
                worker.write_bytes(b"stable synthetic exact worker")
                worker_sha = _sha(worker)
                exact_library = root / "libirisu_box2d_msvc_exact_multiworld.so"
                exact_library.write_bytes(b"stable synthetic exact library")
                initial_library_sha = _sha(exact_library)

                class InvalidLibraryEnv(FakeEnv):
                    mutated = False

                    @property
                    def library_path(self) -> str:
                        return str(worker)

                    def exact_library_provenance(self) -> dict[str, object]:
                        captured = _captured_library(exact_library)
                        if case == "device":
                            captured["mapped_identity"] = {
                                **captured["mapped_identity"],
                                "device": "ff:ff",
                            }
                        return captured

                    @property
                    def build_info(self) -> dict[str, object]:
                        reported_sha = {
                            "placeholder": "0" * 64,
                            "mismatch": "a" * 64,
                            "mutation": initial_library_sha,
                            "device": initial_library_sha,
                        }[case]
                        return {
                            "physics_backend": "exact-msvc9-r58-worker",
                            "worker_backend": "exact-msvc9-r58-multiworld-forward",
                            "worker_executable_sha256": worker_sha,
                            "exact_library_sha256": reported_sha,
                            "worker_pid": 100,
                        }

                    def step(self, action: object) -> tuple[
                        dict[str, Any], int, bool, bool, dict[str, Any]
                    ]:
                        result = super().step(action)
                        if case == "mutation" and not type(self).mutated:
                            exact_library.write_bytes(b"changed exact library")
                            type(self).mutated = True
                        return result

                builder = CorpusBuilder(root)
                scenario = builder.scenario(f"invalid-library-{case}", "match")
                expected_error = {
                    "placeholder": "incomplete backend provenance",
                    "mismatch": "mapped exact library bytes disagree",
                    "mutation": "mapped exact library changed",
                    "device": "device disagrees with the client mount",
                }[case]
                with self.assertRaisesRegex(
                    score_golden.GoldenError, expected_error
                ):
                    score_golden.score_manifest(
                        builder.manifest([scenario]),
                        worker_path=str(worker),
                        env_factory=InvalidLibraryEnv,
                    )

    def test_library_and_worker_options_are_mutually_exclusive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            scenario = builder.scenario("exclusive-backend", "match")
            manifest = builder.manifest([scenario])
            with self.assertRaisesRegex(
                score_golden.GoldenError, "mutually exclusive"
            ):
                score_golden.score_manifest(
                    manifest,
                    library_path="portable.so",
                    worker_path="exact-worker",
                    env_factory=FakeEnv,
                )

            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                score_golden.main(
                    [
                        str(manifest),
                        "--library",
                        "portable.so",
                        "--worker",
                        "exact-worker",
                    ]
                )
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("not allowed with argument", stderr.getvalue())

    def test_cli_forwards_exact_worker_backend(self) -> None:
        stdout = io.StringIO()
        report = {"schema_version": 1, "status": "pass"}
        with mock.patch.object(
            score_golden, "score_manifest", return_value=report
        ) as scorer:
            status = score_golden.main(
                ["manifest.json", "--worker", "irisu-exact-worker", "--compact"],
                stdout=stdout,
            )

        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout.getvalue()), report)
        scorer.assert_called_once_with(
            Path("manifest.json"),
            library_path=None,
            worker_path="irisu-exact-worker",
        )

    def test_hash_mismatch_and_unlinked_oracle_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            scenario = builder.scenario("hash-mismatch", "match")
            manifest = builder.manifest([scenario])
            (Path(directory) / "hash-mismatch" / "actions.jsonl").write_text(
                "tampered\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(score_golden.GoldenError, "hash mismatch"):
                score_golden.validate_manifest(manifest)

        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            scenario = builder.scenario("unlinked", "match")
            scenario["expected"] = copy.deepcopy(scenario["expected"])
            scenario["expected"]["events"][0]["max_count"] = 2
            with self.assertRaisesRegex(score_golden.GoldenError, "exact copy"):
                score_golden.validate_manifest(builder.manifest([scenario]))

    def test_all_five_categories_need_nonzero_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            scenarios = [
                builder.scenario(category, category)
                for category in score_golden.CATEGORIES[:-1]
            ]
            report = score_golden.score_manifest(
                builder.manifest(scenarios), env_factory=FakeEnv
            )

        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["gate"]["coverage_complete"])
        self.assertEqual(report["gate"]["categories"]["orb"]["total"], 0)
        self.assertIsNone(report["gate"]["categories"]["orb"]["rate_percent"])

    def test_exact_integer_95_percent_is_applied_per_scenario_and_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            matches = [builder.scenario(f"match-{index}", "match") for index in range(20)]
            others = [
                builder.scenario(category, category)
                for category in score_golden.CATEGORIES[1:]
            ]
            FakeEnv.missing_event_seeds.add(matches[0]["_seed"])
            report = score_golden.score_manifest(
                builder.manifest(matches + others), env_factory=FakeEnv
            )
            self.assertEqual(report["status"], "pass")
            self.assertEqual(report["gate"]["categories"]["match"]["passed"], 19)
            self.assertEqual(report["gate"]["categories"]["match"]["total"], 20)

        self.setUp()
        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            matches = [builder.scenario(f"match-{index}", "match") for index in range(19)]
            others = [
                builder.scenario(category, category)
                for category in score_golden.CATEGORIES[1:]
            ]
            FakeEnv.missing_event_seeds.add(matches[0]["_seed"])
            report = score_golden.score_manifest(
                builder.manifest(matches + others), env_factory=FakeEnv
            )
            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["gate"]["categories"]["match"]["threshold_met"])
            self.assertEqual(report["gate"]["categories"]["match"]["passed"], 18)
            self.assertEqual(report["gate"]["categories"]["match"]["total"], 19)

    def test_scalar_and_trajectory_mismatches_fail_their_scenario(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            scenarios = [
                builder.scenario(
                    category,
                    category,
                    trajectory_tolerance=0.49 if category == "match" else 0.500001,
                )
                for category in score_golden.CATEGORIES
            ]
            FakeEnv.wrong_scalar_seeds.add(scenarios[1]["_seed"])
            report = score_golden.score_manifest(
                builder.manifest(scenarios), env_factory=FakeEnv
            )

        self.assertEqual(report["status"], "fail")
        self.assertFalse(report["gate"]["all_scalar_transitions_exact"])
        self.assertFalse(report["scenarios"][0]["trajectories"]["passed"])
        self.assertFalse(report["scenarios"][1]["scalars"]["passed"])

    def test_exit_codes_are_nonzero_for_failure_and_not_evaluable(self) -> None:
        self.assertEqual(score_golden.report_exit_code({"status": "pass"}), 0)
        self.assertEqual(score_golden.report_exit_code({"status": "fail"}), 1)
        self.assertEqual(score_golden.report_exit_code({"status": "not_evaluable"}), 2)
        self.assertEqual(score_golden.report_exit_code({}), 2)

    def test_real_native_runner_evaluates_an_admissible_synthetic_case_when_available(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            scenario = builder.scenario(
                "native-smoke", "match", trajectory_tolerance=0.1
            )
            manifest = builder.manifest([scenario])
            try:
                report = score_golden.score_manifest(manifest)
            except score_golden.NativeError as exc:
                self.skipTest(str(exc))

        self.assertEqual(report["status"], "fail")
        self.assertTrue(report["scenarios"][0]["execution"]["passed"])
        self.assertEqual(report["clone_config_u64"], "0xec0e8463feaf2670")
        library = report["clone_library"]
        self.assertEqual(library["status"], "verified")
        self.assertEqual(library["stability_check"], "passed")
        library_path = Path(library["path"])
        self.assertTrue(library_path.is_absolute())
        self.assertTrue(library_path.is_file())
        self.assertEqual(library["bytes"], library_path.stat().st_size)
        self.assertEqual(library["sha256"], _sha(library_path))
        self.assertEqual(library["file_identity"]["inode"], library_path.stat().st_ino)

    def test_real_exact_worker_reports_verified_runtime_artifacts_when_available(
        self,
    ) -> None:
        try:
            worker = find_exact_worker()
        except ExactWorkerNotFoundError as exc:
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as directory:
            builder = CorpusBuilder(Path(directory))
            scenario = builder.scenario(
                "exact-native-smoke", "match", trajectory_tolerance=0.1
            )
            report = score_golden.score_manifest(
                builder.manifest([scenario]), worker_path=str(worker)
            )

        self.assertEqual(report["status"], "fail")
        self.assertTrue(report["scenarios"][0]["execution"]["passed"])
        self.assertEqual(report["clone_config_u64"], "0xec0e8463feaf2670")
        self.assertTrue(report["clone_build"]["exact_library_runtime_verified"])
        worker_artifact = report["clone_worker"]
        self.assertEqual(worker_artifact["status"], "verified")
        self.assertEqual(worker_artifact["stability_check"], "passed")
        self.assertEqual(worker_artifact["sha256"], _sha(worker))
        library_artifact = worker_artifact["linked_exact_library"]
        self.assertEqual(library_artifact["status"], "verified")
        self.assertEqual(library_artifact["stability_check"], "passed")
        library_path = Path(library_artifact["path"])
        self.assertEqual(library_artifact["sha256"], _sha(library_path))
        self.assertEqual(
            library_artifact["sha256"],
            worker_artifact["linked_exact_library_sha256"],
        )
        self.assertEqual(
            library_artifact["file_identity"]["inode"],
            library_artifact["mapped_identity"]["inode"],
        )


if __name__ == "__main__":
    unittest.main()
