from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_PATH = ROOT / "benchmarks/rl_sequence_replay.py"
SPEC = importlib.util.spec_from_file_location(
    "rl_sequence_replay_benchmark", BENCHMARK_PATH
)
assert SPEC is not None and SPEC.loader is not None
BENCHMARK = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BENCHMARK
SPEC.loader.exec_module(BENCHMARK)


def _episode(
    seed: int,
    *,
    score: int,
    survival: int,
    gauge_failure: bool,
) -> dict[str, object]:
    return {
        "seed": seed,
        "gauge_failure": gauge_failure,
        "conversion": {
            "final_score": score,
            "survival_ticks": survival,
        },
    }


def _evaluation(*episodes: dict[str, object]) -> dict[str, object]:
    return {"episodes": list(episodes)}


class SequenceReplayBenchmarkTests(unittest.TestCase):
    def test_config_is_explicitly_development_only(self) -> None:
        config = BENCHMARK._load_config(BENCHMARK.DEFAULT_CONFIG)
        self.assertEqual(
            config["status"], "development_only_not_canonical_evidence"
        )
        self.assertIs(config["deployable"], False)
        self.assertIs(config["canonical_r3_evidence"], False)
        self.assertIs(config["sealed_evaluation_allowed"], False)
        self.assertIs(config["evaluation"]["sealed_test_allowed"], False)
        self.assertIs(
            config["evaluation"]["training_seed_overlap_allowed"], False
        )

    def test_config_rejects_any_relaxed_evidence_boundary(self) -> None:
        source = BENCHMARK.DEFAULT_CONFIG.read_text(encoding="utf-8")
        mutations = {
            "status": (
                'status = "development_only_not_canonical_evidence"',
                'status = "production"',
            ),
            "deployable": ("deployable = false", "deployable = true"),
            "canonical": (
                "canonical_r3_evidence = false",
                "canonical_r3_evidence = true",
            ),
            "sealed": (
                "sealed_evaluation_allowed = false",
                "sealed_evaluation_allowed = true",
            ),
            "nested sealed": (
                "sealed_test_allowed = false",
                "sealed_test_allowed = true",
            ),
            "training overlap": (
                "training_seed_overlap_allowed = false",
                "training_seed_overlap_allowed = true",
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, (old, new) in mutations.items():
                with self.subTest(name=name):
                    self.assertEqual(source.count(old), 1)
                    path = root / f"{name.replace(' ', '-')}.toml"
                    path.write_text(source.replace(old, new), encoding="utf-8")
                    with self.assertRaisesRegex(
                        ValueError, "development-only"
                    ):
                        BENCHMARK._load_config(path)

    def test_seed_suites_match_preregistered_lists_and_are_disjoint(self) -> None:
        config = BENCHMARK._load_config(BENCHMARK.DEFAULT_CONFIG)
        expected_training = (
            3993225437,
            1210292757,
            3690546926,
            2275139235,
            3457209652,
            4085297243,
            3667357959,
            4166920730,
        )
        expected_selection = (
            3747571761,
            3633074689,
            3165196451,
            1295131450,
            1147578272,
            4251965354,
            886910123,
            2939586874,
        )
        expected_evaluation = (
            3028406002,
            2789482418,
            2666547395,
            2200065417,
            1075831150,
            3122472238,
            1753098387,
            2897648126,
            1770871059,
            29484235,
            962351311,
            3845527485,
            1334142681,
            550658739,
            3738769297,
            3062439664,
        )
        suites = (
            BENCHMARK._derive_seeds(
                config["training"]["development_seed_label"],
                config["training"]["development_seeds"],
            ),
            BENCHMARK._derive_seeds(
                config["selection"]["seed_label"],
                config["selection"]["seeds"],
            ),
            BENCHMARK._derive_seeds(
                config["evaluation"]["suite_label"],
                config["evaluation"]["seed_count"],
            ),
        )
        self.assertEqual(
            suites,
            (expected_training, expected_selection, expected_evaluation),
        )
        self.assertTrue(all(len(set(suite)) == len(suite) for suite in suites))
        self.assertFalse(set(suites[0]) & set(suites[1]))
        self.assertFalse(set(suites[0]) & set(suites[2]))
        self.assertFalse(set(suites[1]) & set(suites[2]))

    def test_seed_derivation_rejects_implicit_or_empty_suites(self) -> None:
        for label, count in (("", 1), ("fixture", 0), ("fixture", -1)):
            with self.subTest(label=label, count=count):
                with self.assertRaisesRegex(ValueError, "seed derivation"):
                    BENCHMARK._derive_seeds(label, count)

    def test_forbidden_output_names_and_source_trees_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            forbidden = (
                root / "sealed" / "report.json",
                root / "test-output.json",
                root / "canonical" / "report.json",
                root / "artifacts/r3/runs/campaign/report.json",
                ROOT / "tests/sequence-replay-report.json",
                ROOT / "benchmarks/sequence-replay-report.json",
            )
            for path in forbidden:
                with self.subTest(path=path):
                    with self.assertRaisesRegex(
                        ValueError, "development-only|overwrite source"
                    ):
                        BENCHMARK._safe_path(
                            path, "fixture output", output=True
                        )

    def test_output_symlinks_and_symlinked_parents_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "development" / "report.json"
            target.parent.mkdir()
            target.write_text("original\n", encoding="utf-8")
            leaf = root / "report-link.json"
            leaf.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "symbolic link"):
                BENCHMARK._safe_path(leaf, "fixture output", output=True)

            forbidden = root / "canonical"
            forbidden.mkdir()
            parent = root / "development-link"
            parent.symlink_to(forbidden, target_is_directory=True)
            with self.assertRaisesRegex(
                ValueError, "development-only|symbolic link"
            ):
                BENCHMARK._safe_path(
                    parent / "report.json", "fixture output", output=True
                )

    def test_paired_comparison_accounts_for_every_regression_and_rescue(
        self,
    ) -> None:
        baseline = _evaluation(
            _episode(1, score=100, survival=10_000, gauge_failure=False),
            _episode(2, score=100, survival=8_000, gauge_failure=False),
            _episode(3, score=100, survival=8_000, gauge_failure=False),
            _episode(4, score=10, survival=2_000, gauge_failure=True),
        )
        candidate = _evaluation(
            _episode(1, score=95, survival=9_000, gauge_failure=True),
            _episode(2, score=50, survival=4_000, gauge_failure=False),
            _episode(3, score=99, survival=7_999, gauge_failure=False),
            _episode(4, score=50, survival=10_000, gauge_failure=False),
        )
        result = BENCHMARK.paired_comparison(
            baseline, candidate, horizon=10_000
        )
        self.assertEqual(result["seeds"], [1, 2, 3, 4])
        self.assertEqual(result["score_regressions"], 3)
        self.assertEqual(result["survival_regressions"], 3)
        self.assertEqual(result["joint_score_survival_regressions"], 3)
        self.assertEqual(result["gauge_regressions"], 1)
        self.assertEqual(result["gauge_rescues"], 1)
        self.assertEqual(result["catastrophic_regressions"], 2)
        self.assertEqual(
            [value["seed"] for value in result["catastrophic_seeds"]],
            [1, 2],
        )
        pairs = {value["seed"]: value for value in result["pairs"]}
        self.assertEqual(
            pairs[1]["catastrophic_reasons"], ["terminal_flip"]
        )
        self.assertEqual(
            pairs[2]["catastrophic_reasons"],
            ["severe_joint_collapse"],
        )
        self.assertFalse(pairs[3]["catastrophic"])
        self.assertTrue(pairs[4]["gauge_rescue"])
        self.assertEqual(result["score_delta"]["median"], -3.0)
        self.assertEqual(result["survival_delta"]["median"], -500.5)

    def test_paired_comparison_rejects_unpaired_or_duplicate_seeds(self) -> None:
        episode = _episode(
            1, score=1, survival=1, gauge_failure=False
        )
        with self.assertRaisesRegex(ValueError, "identical seeds"):
            BENCHMARK.paired_comparison(
                _evaluation(episode),
                _evaluation(
                    _episode(
                        2, score=1, survival=1, gauge_failure=False
                    )
                ),
                horizon=1,
            )
        with self.assertRaisesRegex(ValueError, "repeats a seed"):
            BENCHMARK.paired_comparison(
                _evaluation(episode, dict(episode)),
                _evaluation(episode),
                horizon=1,
            )

    def test_atomic_report_refuses_to_replace_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "development" / "report.json"
            target.parent.mkdir()
            target.write_text("original\n", encoding="utf-8")
            with self.assertRaisesRegex(
                FileExistsError, "refusing to replace"
            ):
                BENCHMARK._atomic_json(
                    target, {"replacement": True}, overwrite=False
                )
            self.assertEqual(target.read_text(encoding="utf-8"), "original\n")
            self.assertEqual(
                tuple(target.parent.glob(f".{target.name}.*.tmp")), ()
            )


if __name__ == "__main__":
    unittest.main()
