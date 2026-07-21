from __future__ import annotations

import importlib.util
import hashlib
import json
import stat
import struct
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_exact_replay_corpus",
    ROOT / "tools" / "evaluate-exact-replay-corpus.py",
)
assert SPEC and SPEC.loader
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)


def replay(*, seed: int, level: int, score: int, chain: int, frames: int, padded: bool) -> bytes:
    data = struct.pack("<5i", seed, level, score, chain, 0)
    if padded:
        data += bytes(32)
    return data + struct.pack("<I", 0) * frames


class ExactReplayCorpusEvaluatorTests(unittest.TestCase):
    def test_inventory_excludes_legacy_without_filename_special_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            padded = root / "anything.rpy"
            legacy = root / "also-anything.rpy"
            padded.write_bytes(replay(seed=1, level=2, score=30, chain=4, frames=3, padded=True))
            legacy.write_bytes(replay(seed=2, level=8, score=900, chain=9, frames=2, padded=False))
            entries = evaluator.inventory([legacy, padded])

        by_name = {Path(entry["path"]).name: entry for entry in entries}
        self.assertTrue(by_name["anything.rpy"]["eligible"])
        self.assertFalse(by_name["also-anything.rpy"]["eligible"])
        self.assertEqual(
            by_name["also-anything.rpy"]["exclusion_reasons"],
            ["legacy_layout_predates_v203_mechanics"],
        )

    def test_evaluation_compares_every_scalar_and_uses_pc53(self) -> None:
        runner_source = """#!/usr/bin/env python3
import json, os, struct, sys
data = open(sys.argv[1], 'rb').read()
seed, level, score, chain, mode = struct.unpack_from('<5i', data)
assert os.environ['IRISU_EXACT_CW'] == '0x27f'
frames = (len(data) - 52) // 4
print(json.dumps({'tick': frames, 'score': score + seed - 1, 'level': level,
                  'highest_chain': chain, 'terminal_frame': frames - 1,
                  'gauge': 2990, 'clears': 1, 'score_calls': 1,
                  'score_timeline': [[frames, score + seed - 1, score + seed - 1]],
                  'gauge_timeline': [[2, -10, 2990, 5, 1]]}))
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exact = root / "exact.rpy"
            mismatch = root / "mismatch.rpy"
            legacy = root / "old.rpy"
            exact.write_bytes(replay(seed=1, level=2, score=30, chain=4, frames=3, padded=True))
            mismatch.write_bytes(replay(seed=3, level=5, score=40, chain=6, frames=7, padded=True))
            legacy.write_bytes(replay(seed=9, level=8, score=90, chain=9, frames=2, padded=False))
            runner = root / "runner"
            runner.write_text(runner_source, encoding="utf-8")
            runner.chmod(runner.stat().st_mode | stat.S_IXUSR)

            oracle = root / "oracle"
            (oracle / "replay").mkdir(parents=True)
            oracle_replay = oracle / "replay" / "target.rpy"
            oracle_replay.write_bytes(exact.read_bytes())
            events = (
                '{"event":"score","tick":3,"delta":30,"score":30}\n'
                '{"event":"rot_penalty","tick":2,"delta":-10,"gauge":2990}\n'
            ).encode()
            (oracle / "events.jsonl").write_bytes(events)
            events_sha = hashlib.sha256(events).hexdigest()
            replay_sha = hashlib.sha256(exact.read_bytes()).hexdigest()
            metadata = oracle / "metadata.json"
            metadata.write_text(
                json.dumps(
                    {
                        "schema": 1,
                        "status": "valid_repeated_original_replay_event_oracle_header_incompatible",
                        "inputs": {
                            "irisu_exe_sha256": evaluator.CANONICAL_EXE_SHA256,
                            "box2d_dll_sha256": evaluator.CANONICAL_BOX2D_SHA256,
                            "replay_sha256": replay_sha,
                        },
                        "result": {
                            "tick": 3,
                            "terminal_input_frame": 2,
                            "score": 30,
                            "level": 2,
                            "highest_chain": 4,
                            "qualifying_clears": 1,
                            "score_calls": 1,
                            "rot_penalties": 1,
                            "gauge_after_terminal_actor_pass": 2990,
                        },
                        "repeat": {
                            "normalized_events_byte_identical": True,
                            "normalized_events_sha256": events_sha,
                        },
                        "artifacts": {"events_jsonl_sha256": events_sha},
                    }
                ),
                encoding="utf-8",
            )

            report = evaluator.evaluate(
                [legacy, mismatch, exact], runner=runner, oracle_paths=[metadata]
            )

        self.assertEqual(report["summary"]["eligible"], 2)
        self.assertEqual(report["summary"]["evaluated"], 2)
        headers = report["summary"]["unverified_header_diagnostics"]
        self.assertEqual(headers["exact_score"], 1)
        self.assertEqual(headers["score_mean_absolute_error"], 1.0)
        observed = report["summary"]["observed_v203_oracles"]
        self.assertEqual(observed["available_for_corpus"], 1)
        self.assertEqual(observed["full_scoring_parity"], 1)
        self.assertEqual(report["runner"]["control_word"], "0x027f")
        evaluated = [entry for entry in report["inventory"] if entry["eligible"]]
        self.assertEqual(
            evaluated[0]["evaluation"]["runner_output"]["gauge_timeline_count"],
            1,
        )
        self.assertNotIn(
            "gauge_timeline", evaluated[0]["evaluation"]["runner_output"]
        )
        self.assertTrue(
            all(
                entry["evaluation"]["unverified_header_diagnostic"]
                ["comparisons"]["terminal_tick"]["matches"]
                for entry in evaluated
            )
        )
        authoritative = next(
            entry["evaluation"]["observed_v203_oracle"]
            for entry in evaluated
            if entry["evaluation"]["observed_v203_oracle"] is not None
        )
        self.assertEqual(
            authoritative["evidence"]["authority"],
            "observed_bundled_v203_playback",
        )
        self.assertTrue(authoritative["score_timeline"]["matches"])
        self.assertTrue(authoritative["rot_penalty_timeline"]["matches"])
        excluded = next(entry for entry in report["inventory"] if not entry["eligible"])
        self.assertIsNone(excluded["evaluation"])

    def test_replay_exhaustion_is_compared_at_last_record(self) -> None:
        raw = {
            "tick": 12,
            "terminal_frame": 19,
            "score": 88,
            "level": 1,
            "highest_chain": 2,
            "clears": 6,
            "score_calls": 1,
            "gauge": 1766,
            "score_timeline": [[12, 88, 88]],
            "gauge_timeline": [[8, -10, 1766, 5, 1]],
        }
        oracle = {
            "frame_count": 12,
            "checkpoint_kind": "replay_exhaustion",
            "terminal": {
                "tick": 12,
                "score": 88,
                "level": 1,
                "highest_chain": 2,
                "clears": 6,
                "score_calls": 1,
                "gauge": 1766,
            },
            "score_timeline": [[12, 88, 88]],
            "rot_timeline": [[8, -10, 1766]],
            "evidence": {},
        }

        comparison = evaluator._compare_oracle(raw, oracle)

        self.assertTrue(comparison["full_scoring_parity"])
        self.assertEqual(comparison["checkpoint"]["kind"], "replay_exhaustion")
        self.assertTrue(comparison["checkpoint"]["all_replay_frames_consumed"])
        self.assertTrue(
            comparison["checkpoint"]["no_natural_terminal_before_exhaustion"]
        )


if __name__ == "__main__":
    unittest.main()
