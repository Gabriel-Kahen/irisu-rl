"""Host-aware CPU budgeting for independent training jobs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


def _finite_cgroup_quota(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    membership: Path = Path("/proc/self/cgroup"),
) -> float | None:
    """Return the tightest cgroup-v2 CPU quota visible to this process."""

    try:
        line = next(
            value
            for value in membership.read_text(encoding="utf-8").splitlines()
            if value.startswith("0::")
        )
    except (FileNotFoundError, OSError, StopIteration, UnicodeError):
        return None
    leaf = cgroup_root / line[3:].lstrip("/")
    quotas: list[float] = []
    while leaf == cgroup_root or cgroup_root in leaf.parents:
        try:
            fields = (leaf / "cpu.max").read_text(encoding="ascii").split()
        except (FileNotFoundError, OSError, UnicodeError):
            fields = []
        if len(fields) == 2 and fields[0] != "max":
            try:
                quota, period = (int(value) for value in fields)
            except ValueError:
                pass
            else:
                if quota > 0 and period > 0:
                    quotas.append(quota / period)
        if leaf == cgroup_root:
            break
        leaf = leaf.parent
    return min(quotas) if quotas else None


def available_cpu_capacity() -> tuple[int, float | None, float]:
    """Return affinity CPUs, finite quota, and effective CPU capacity."""

    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os.cpu_count() or 1
    affinity = max(1, affinity)
    quota = _finite_cgroup_quota()
    return affinity, quota, min(float(affinity), quota or float(affinity))


@dataclass(frozen=True, slots=True)
class TrainingCpuPlan:
    """Resolved phase-specific concurrency without changing trial behavior."""

    affinity_cpus: int
    quota_cpus: float | None
    effective_cpus: float
    target_percent: int
    reserved_cpus: int
    target_cpu_slots: int
    training_cpu_ids: tuple[int, ...]
    estimated_cores_per_job: float
    parallel_jobs: int
    max_parallel_jobs: int
    version: str = "r3b-training-cpu-plan-v1"

    def __post_init__(self) -> None:
        positive_ints = (
            self.affinity_cpus,
            self.target_percent,
            self.target_cpu_slots,
            self.parallel_jobs,
            self.max_parallel_jobs,
        )
        if (
            self.version != "r3b-training-cpu-plan-v1"
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in positive_ints
            )
            or not 1 <= self.target_percent <= 100
            or isinstance(self.reserved_cpus, bool)
            or not isinstance(self.reserved_cpus, int)
            or self.reserved_cpus < 0
            or isinstance(self.effective_cpus, bool)
            or not isinstance(self.effective_cpus, (int, float))
            or not math.isfinite(self.effective_cpus)
            or self.effective_cpus <= 0
            or self.target_cpu_slots > self.affinity_cpus
            or not isinstance(self.training_cpu_ids, tuple)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in self.training_cpu_ids
            )
            or len(self.training_cpu_ids) != self.target_cpu_slots
            or len(set(self.training_cpu_ids)) != len(self.training_cpu_ids)
            or any(value < 0 for value in self.training_cpu_ids)
            or self.parallel_jobs > self.max_parallel_jobs
            or isinstance(self.estimated_cores_per_job, bool)
            or not isinstance(self.estimated_cores_per_job, (int, float))
            or not math.isfinite(self.estimated_cores_per_job)
            or self.estimated_cores_per_job <= 0
            or (
                self.quota_cpus is not None
                and (
                    isinstance(self.quota_cpus, bool)
                    or not isinstance(self.quota_cpus, (int, float))
                    or not math.isfinite(self.quota_cpus)
                    or self.quota_cpus <= 0
                )
            )
        ):
            raise ValueError("training CPU plan is invalid")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "affinity_cpus": self.affinity_cpus,
            "quota_cpus": self.quota_cpus,
            "effective_cpus": self.effective_cpus,
            "target_percent": self.target_percent,
            "reserved_cpus": self.reserved_cpus,
            "target_cpu_slots": self.target_cpu_slots,
            "unallocated_cpu_slots": max(
                0, math.floor(self.effective_cpus) - self.target_cpu_slots
            ),
            "training_cpu_ids": list(self.training_cpu_ids),
            "estimated_cores_per_job": self.estimated_cores_per_job,
            "parallel_jobs": self.parallel_jobs,
            "max_parallel_jobs": self.max_parallel_jobs,
            "scope": "independent-training-jobs-only",
            "trial_behavior_changed": False,
        }

    @property
    def sha256(self) -> str:
        payload = json.dumps(
            self.manifest(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def resolve_training_cpu_plan(
    *,
    workers_per_job: int,
    torch_threads_per_job: int,
    target_percent: int = 80,
    reserved_cpus: int = 1,
    max_parallel_jobs: int = 4,
    affinity_cpus: int | None = None,
    quota_cpus: float | None = None,
) -> TrainingCpuPlan:
    """Resolve a bounded job count from host capacity and frozen job topology."""

    values = (workers_per_job, torch_threads_per_job, max_parallel_jobs)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in values
    ):
        raise ValueError("worker, Torch-thread, and job caps must be positive integers")
    if (
        isinstance(target_percent, bool)
        or not isinstance(target_percent, int)
        or not 1 <= target_percent <= 100
        or isinstance(reserved_cpus, bool)
        or not isinstance(reserved_cpus, int)
        or reserved_cpus < 0
    ):
        raise ValueError("CPU target or reserve is invalid")
    if affinity_cpus is None:
        try:
            available_ids = tuple(sorted(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            available_ids = tuple(range(os.cpu_count() or 1))
        affinity_cpus, detected_quota, _effective = available_cpu_capacity()
        if quota_cpus is None:
            quota_cpus = detected_quota
    else:
        if affinity_cpus <= 0:
            raise ValueError("affinity CPU count must be positive")
        available_ids = tuple(range(affinity_cpus))
    if quota_cpus is not None and (not math.isfinite(quota_cpus) or quota_cpus <= 0):
        raise ValueError("CPU quota must be finite and positive")
    effective = min(float(affinity_cpus), quota_cpus or float(affinity_cpus))

    resolved_reserve = min(reserved_cpus, max(0, math.floor(effective) - 1))
    usable = max(1, math.floor(effective) - resolved_reserve)
    target = max(1, min(usable, math.floor(effective * target_percent / 100 + 0.5)))
    # Synchronous exact stepping leaves roughly half of a lane pool runnable
    # while Torch work uses its explicit intra-op team. This topology-derived
    # estimate is only a scheduler input; the resolved plan reports it.
    estimated = max(float(torch_threads_per_job), workers_per_job / 2)
    jobs = min(max_parallel_jobs, max(1, math.ceil(target / estimated)))
    jobs = min(jobs, max(1, target // torch_threads_per_job))
    return TrainingCpuPlan(
        affinity_cpus,
        quota_cpus,
        effective,
        target_percent,
        resolved_reserve,
        target,
        available_ids[:target],
        estimated,
        jobs,
        max_parallel_jobs,
    )
