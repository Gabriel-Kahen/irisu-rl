from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from irisu_rl.cpu_parallelism import (
    _finite_cgroup_quota,
    available_cpu_capacity,
    resolve_training_cpu_plan,
)


class CpuParallelismTests(unittest.TestCase):
    def test_host_plan_targets_eighty_percent_without_changing_jobs(self) -> None:
        plan = resolve_training_cpu_plan(
            workers_per_job=16,
            torch_threads_per_job=4,
            affinity_cpus=16,
            target_percent=80,
            reserved_cpus=1,
            max_parallel_jobs=4,
        )
        self.assertEqual(plan.target_cpu_slots, 13)
        self.assertEqual(plan.parallel_jobs, 2)
        self.assertFalse(plan.manifest()["trial_behavior_changed"])

    def test_quota_and_reserve_cap_the_plan(self) -> None:
        plan = resolve_training_cpu_plan(
            workers_per_job=16,
            torch_threads_per_job=4,
            affinity_cpus=32,
            quota_cpus=10.5,
            target_percent=100,
            reserved_cpus=2,
            max_parallel_jobs=8,
        )
        self.assertEqual(plan.effective_cpus, 10.5)
        self.assertEqual(plan.target_cpu_slots, 8)
        self.assertEqual(plan.parallel_jobs, 1)

    def test_invalid_policy_is_rejected(self) -> None:
        for changes in (
            {"target_percent": 0},
            {"target_percent": 101},
            {"reserved_cpus": -1},
            {"max_parallel_jobs": 0},
            {"quota_cpus": float("inf")},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                resolve_training_cpu_plan(
                    workers_per_job=16,
                    torch_threads_per_job=4,
                    affinity_cpus=16,
                    **changes,
                )

    def test_cgroup_uses_tightest_finite_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            leaf = root / "user.slice" / "scope"
            leaf.mkdir(parents=True)
            membership = root / "membership"
            membership.write_text("0::/user.slice/scope\n", encoding="utf-8")
            (root / "cpu.max").write_text("max 100000\n", encoding="ascii")
            (root / "user.slice" / "cpu.max").write_text(
                "600000 100000\n", encoding="ascii"
            )
            (leaf / "cpu.max").write_text("250000 100000\n", encoding="ascii")
            self.assertEqual(_finite_cgroup_quota(root, membership), 2.5)

    def test_affinity_is_combined_with_detected_quota(self) -> None:
        with (
            mock.patch("os.sched_getaffinity", return_value=set(range(8))),
            mock.patch(
                "irisu_rl.cpu_parallelism._finite_cgroup_quota",
                return_value=3.5,
            ),
        ):
            self.assertEqual(available_cpu_capacity(), (8, 3.5, 3.5))


if __name__ == "__main__":
    unittest.main()
