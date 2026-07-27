from __future__ import annotations

import io
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from tools import validate


class ValidateRunnerTests(unittest.TestCase):
    def test_available_cpus_uses_effective_affinity_and_quota(self) -> None:
        with mock.patch.object(
            validate,
            "available_cpu_capacity",
            return_value=(16, 3.5, 3.5),
        ):
            self.assertEqual(validate.available_cpus(), 3)

    def test_available_cpus_handles_capacity_and_affinity_errors(self) -> None:
        with (
            mock.patch.object(
                validate,
                "available_cpu_capacity",
                side_effect=RuntimeError("unreadable cgroup"),
            ),
            mock.patch.object(
                os,
                "sched_getaffinity",
                side_effect=OSError("unavailable"),
                create=True,
            ),
            mock.patch.object(os, "cpu_count", return_value=4),
        ):
            self.assertEqual(validate.available_cpus(), 4)

    def test_validation_environment_caps_nested_thread_pools(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.dict(os.environ, {"PYTHONPATH": "existing"}, clear=True):
                environment = validate.validation_environment(root)
        self.assertEqual(
            environment["PYTHONPATH"],
            os.pathsep.join((str(root / "python"), "existing")),
        )
        for variable in validate.THREAD_LIMIT_VARIABLES:
            self.assertEqual(environment[variable], "1")

    def test_python_discovery_is_stable_longest_first(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tests = root / "tests"
            tests.mkdir()
            nested = tests / "nested"
            nested.mkdir()
            (tests / "test_small.py").write_text("pass\n", encoding="utf-8")
            (nested / "test_large.py").write_text(
                "pass\n" * 10, encoding="utf-8"
            )
            tasks = validate.discover_python_tasks(root, "python", jobs=16)
        self.assertEqual(
            [task.label for task in tasks],
            ["python:tests/nested/test_large.py", "python:tests/test_small.py"],
        )

    def test_web_discovery_is_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "apps" / "web" / "tests" / "nested"
            nested.mkdir(parents=True)
            (nested / "color.test.mjs").write_text("", encoding="utf-8")
            with mock.patch.object(shutil, "which", return_value="/node"):
                tasks = validate.discover_web_tasks(root)
        self.assertEqual(
            [task.label for task in tasks],
            ["web:apps/web/tests/nested/color.test.mjs"],
        )

    def test_tasks_overlap_and_results_are_sorted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = (
                "import pathlib,sys,time;"
                "root=pathlib.Path(sys.argv[1]);"
                "(root/sys.argv[2]).touch();"
                "deadline=time.monotonic()+10;"
                "\nwhile len(list(root.iterdir())) < 2 and time.monotonic() < deadline:"
                "\n time.sleep(0.01)"
                "\nsys.exit(0 if len(list(root.iterdir())) >= 2 else 1)"
            )
            tasks = [
                validate.Task(
                    "z-last", (sys.executable, "-c", script, str(root), "z")
                ),
                validate.Task(
                    "a-first", (sys.executable, "-c", script, str(root), "a")
                ),
            ]
            results = validate.run_tasks(tasks, 2)
        self.assertEqual([result.label for result in results], ["a-first", "z-last"])
        self.assertTrue(all(result.returncode == 0 for result in results))

    def test_failure_output_is_aggregated_in_label_order(self) -> None:
        results = [
            validate.Result("b", ("false",), 2, "second\n", 0.2),
            validate.Result("a", ("false",), 1, "first\n", 0.1),
        ]
        stream = io.StringIO()
        with mock.patch("sys.stdout", stream):
            passed = validate.render_results(
                sorted(results, key=lambda result: result.label)
            )
        self.assertFalse(passed)
        self.assertLess(stream.getvalue().index("FAIL a"), stream.getvalue().index("FAIL b"))
        self.assertIn("first", stream.getvalue())
        self.assertIn("second", stream.getvalue())

    def test_missing_command_becomes_a_failed_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            command = str(Path(directory) / "missing-command")
            result = validate.run_tasks(
                [validate.Task("missing", (command,))], 1
            )[0]
        self.assertEqual(result.returncode, 127)
        self.assertTrue(result.output)

    def test_known_nested_concurrency_reserves_slots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tests = root / "tests"
            tests.mkdir()
            (tests / "test_physics_lifecycle.py").write_text("", encoding="utf-8")
            (tests / "test_r3b_supervisor.py").write_text("", encoding="utf-8")
            tasks = validate.discover_python_tasks(root, "python", jobs=16)
        self.assertEqual(
            {task.label: task.slots for task in tasks},
            {
                "python:tests/test_physics_lifecycle.py": 8,
                "python:tests/test_r3b_supervisor.py": 9,
            },
        )

    def test_task_cannot_exceed_the_cpu_budget(self) -> None:
        with self.assertRaisesRegex(ValueError, "fit the validation budget"):
            validate.run_tasks(
                [validate.Task("oversized", (sys.executable,), slots=2)],
                1,
            )

    def test_nonpositive_jobs_are_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            validate.parse_args(["--jobs", "0"])

    def test_no_native_still_builds(self) -> None:
        with (
            mock.patch.object(validate, "_run_build", return_value=True) as build,
            mock.patch.object(validate, "render_results", return_value=True),
        ):
            result = validate.main(
                ["--jobs", "1", "--no-native", "--no-python", "--no-web"]
            )
        self.assertEqual(result, 0)
        build.assert_called_once()


if __name__ == "__main__":
    unittest.main()
