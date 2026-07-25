from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from irisu_rl.cpu_parallelism import (
    TrainingCpuPlan,
    _finite_cgroup_quota,
    _topology_cpu_order,
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

    def test_internally_inconsistent_plan_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "capacity is inconsistent"):
            TrainingCpuPlan(
                affinity_cpus=16,
                quota_cpus=2.0,
                effective_cpus=2.0,
                target_percent=80,
                requested_reserved_cpus=1,
                reserved_cpus=1,
                target_cpu_slots=16,
                training_cpu_ids=tuple(range(16)),
                workers_per_job=16,
                torch_threads_per_job=4,
                estimated_cores_per_job=8.0,
                parallel_jobs=16,
                max_parallel_jobs=16,
            )

    def test_oversubscribed_job_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "capacity is inconsistent"):
            TrainingCpuPlan(
                affinity_cpus=1,
                quota_cpus=None,
                effective_cpus=1.0,
                target_percent=80,
                requested_reserved_cpus=1,
                reserved_cpus=0,
                target_cpu_slots=1,
                training_cpu_ids=(0,),
                workers_per_job=16,
                torch_threads_per_job=4,
                estimated_cores_per_job=8.0,
                parallel_jobs=16,
                max_parallel_jobs=16,
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

    def test_cgroup_v1_uses_tightest_finite_ancestor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = root / "cpu"
            leaf = controller / "tenant" / "job"
            leaf.mkdir(parents=True)
            membership = root / "membership"
            membership.write_text(
                "2:cpu,cpuacct:/tenant/job\n",
                encoding="utf-8",
            )
            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                f"29 23 0:26 / {controller} rw - cgroup cgroup rw,cpu,cpuacct\n",
                encoding="utf-8",
            )
            for path, quota in (
                (controller, -1),
                (controller / "tenant", 600000),
                (leaf, 250000),
            ):
                (path / "cpu.cfs_quota_us").write_text(
                    f"{quota}\n",
                    encoding="ascii",
                )
                (path / "cpu.cfs_period_us").write_text(
                    "100000\n",
                    encoding="ascii",
                )
            self.assertEqual(
                _finite_cgroup_quota(root, membership, mountinfo),
                2.5,
            )

    def test_hybrid_cgroup_uses_the_v1_cpu_controller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            controller = root / "cpu"
            leaf = controller / "job"
            leaf.mkdir(parents=True)
            membership = root / "membership"
            membership.write_text(
                "0::/unified\n2:cpu,cpuacct:/job\n",
                encoding="utf-8",
            )
            mountinfo = root / "mountinfo"
            mountinfo.write_text(
                f"29 23 0:26 / {controller} rw - cgroup cgroup rw,cpu,cpuacct\n",
                encoding="utf-8",
            )
            for path, quota in ((controller, -1), (leaf, 200000)):
                (path / "cpu.cfs_quota_us").write_text(
                    f"{quota}\n",
                    encoding="ascii",
                )
                (path / "cpu.cfs_period_us").write_text(
                    "100000\n",
                    encoding="ascii",
                )
            self.assertEqual(
                _finite_cgroup_quota(root / "unified", membership, mountinfo),
                2.0,
            )

    def test_cpu_order_uses_physical_cores_before_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cpu_id, core_id in ((2, 0), (4, 1), (6, 0), (8, 1)):
                topology = root / f"cpu{cpu_id}" / "topology"
                topology.mkdir(parents=True)
                (topology / "physical_package_id").write_text(
                    "0\n",
                    encoding="ascii",
                )
                (topology / "core_id").write_text(
                    f"{core_id}\n",
                    encoding="ascii",
                )
            self.assertEqual(
                _topology_cpu_order((8, 6, 4, 2), root),
                (2, 4, 6, 8),
            )

    def test_default_job_cap_scales_on_a_sixty_four_cpu_host(self) -> None:
        plan = resolve_training_cpu_plan(
            workers_per_job=16,
            torch_threads_per_job=4,
            affinity_cpus=64,
        )
        self.assertEqual(plan.target_cpu_slots, 51)
        self.assertEqual(plan.parallel_jobs, 7)
        self.assertTrue(plan.manifest()["estimated_target_satisfied"])

    def test_cpu_policy_is_stable_across_host_allocations(self) -> None:
        first = resolve_training_cpu_plan(
            workers_per_job=16,
            torch_threads_per_job=4,
            affinity_cpus=16,
        )
        changed = resolve_training_cpu_plan(
            workers_per_job=16,
            torch_threads_per_job=4,
            affinity_cpus=8,
        )
        self.assertEqual(first.policy_manifest(), changed.policy_manifest())

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
