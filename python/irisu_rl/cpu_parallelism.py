"""Host-aware CPU budgeting for independent training jobs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


def _quota_from_v2(cgroup_root: Path, membership_path: str) -> float | None:
    parts = PurePosixPath(membership_path).parts
    if ".." in parts:
        raise RuntimeError("cgroup-v2 membership path is unsafe")
    leaf = cgroup_root.joinpath(*parts[1:])
    quotas: list[float] = []
    observed = False
    while leaf == cgroup_root or cgroup_root in leaf.parents:
        try:
            fields = (leaf / "cpu.max").read_text(encoding="ascii").split()
        except FileNotFoundError:
            fields = []
        except (OSError, UnicodeError) as exc:
            raise RuntimeError("cgroup-v2 CPU quota could not be read") from exc
        if fields:
            observed = True
            if len(fields) != 2:
                raise RuntimeError("cgroup-v2 CPU quota is malformed")
            if fields[0] != "max":
                try:
                    quota, period = (int(value) for value in fields)
                except ValueError as exc:
                    raise RuntimeError("cgroup-v2 CPU quota is malformed") from exc
                if quota <= 0 or period <= 0:
                    raise RuntimeError("cgroup-v2 CPU quota is invalid")
                quotas.append(quota / period)
        if leaf == cgroup_root:
            break
        leaf = leaf.parent
    if not observed:
        raise RuntimeError("cgroup-v2 CPU controller could not be resolved")
    return min(quotas) if quotas else None


def _mount_path(value: str) -> Path:
    decoded = re.sub(
        r"\\([0-7]{3})",
        lambda match: chr(int(match.group(1), 8)),
        value,
    )
    return Path(decoded)


def _quota_from_v1(
    membership_path: str,
    *,
    mountinfo: Path,
) -> float | None:
    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError, UnicodeError) as exc:
        raise RuntimeError("cgroup-v1 CPU controller mount could not be read") from exc
    member = PurePosixPath(membership_path)
    leaf: Path | None = None
    mount_point: Path | None = None
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if (
            len(fields) <= separator + 3
            or fields[separator + 1] != "cgroup"
            or "cpu" not in fields[separator + 3].split(",")
        ):
            continue
        root = PurePosixPath(fields[3])
        if member != root and root not in member.parents:
            continue
        mount_point = _mount_path(fields[4])
        relative = member.relative_to(root)
        leaf = mount_point.joinpath(*relative.parts)
        break
    if leaf is None or mount_point is None:
        raise RuntimeError("cgroup-v1 CPU controller mount could not be resolved")
    quotas: list[float] = []
    observed = False
    while leaf == mount_point or mount_point in leaf.parents:
        try:
            quota_text = (leaf / "cpu.cfs_quota_us").read_text(encoding="ascii")
            period_text = (leaf / "cpu.cfs_period_us").read_text(encoding="ascii")
        except FileNotFoundError:
            quota_text = period_text = ""
        except (OSError, UnicodeError) as exc:
            raise RuntimeError("cgroup-v1 CPU quota could not be read") from exc
        if quota_text or period_text:
            observed = True
            try:
                quota = int(quota_text)
                period = int(period_text)
            except ValueError as exc:
                raise RuntimeError("cgroup-v1 CPU quota is malformed") from exc
            if period <= 0 or quota == 0 or quota < -1:
                raise RuntimeError("cgroup-v1 CPU quota is invalid")
            if quota > 0:
                quotas.append(quota / period)
        if leaf == mount_point:
            break
        leaf = leaf.parent
    if not observed:
        raise RuntimeError("cgroup-v1 CPU controller could not be resolved")
    return min(quotas) if quotas else None


def _finite_cgroup_quota(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    membership: Path = Path("/proc/self/cgroup"),
    mountinfo: Path = Path("/proc/self/mountinfo"),
) -> float | None:
    """Return the tightest cgroup-v1/v2 CPU quota visible to this process."""

    try:
        lines = membership.read_text(encoding="utf-8").splitlines()
    except (FileNotFoundError, OSError, UnicodeError):
        return None
    for line in lines:
        fields = line.split(":", 2)
        if len(fields) == 3 and "cpu" in fields[1].split(","):
            return _quota_from_v1(fields[2], mountinfo=mountinfo)
    for line in lines:
        if line.startswith("0::"):
            return _quota_from_v2(cgroup_root, line[3:])
    return None


def _topology_cpu_order(
    cpu_ids: tuple[int, ...],
    topology_root: Path = Path("/sys/devices/system/cpu"),
) -> tuple[int, ...]:
    """Prefer one logical CPU per physical core before SMT siblings."""

    groups: dict[tuple[int, int], list[int]] = {}
    for cpu_id in sorted(cpu_ids):
        topology = topology_root / f"cpu{cpu_id}" / "topology"
        try:
            package = int(
                (topology / "physical_package_id").read_text(encoding="ascii")
            )
            core = int((topology / "core_id").read_text(encoding="ascii"))
        except (FileNotFoundError, OSError, UnicodeError, ValueError):
            return tuple(sorted(cpu_ids))
        groups.setdefault((package, core), []).append(cpu_id)
    ordered: list[int] = []
    for sibling_index in range(
        max((len(group) for group in groups.values()), default=0)
    ):
        ordered.extend(
            group[sibling_index]
            for _identity, group in sorted(groups.items())
            if sibling_index < len(group)
        )
    return tuple(ordered)


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
    requested_reserved_cpus: int
    reserved_cpus: int
    target_cpu_slots: int
    training_cpu_ids: tuple[int, ...]
    workers_per_job: int
    torch_threads_per_job: int
    estimated_cores_per_job: float
    parallel_jobs: int
    max_parallel_jobs: int
    version: str = "r3b-training-cpu-plan-v1"

    def __post_init__(self) -> None:
        positive_ints = (
            self.affinity_cpus,
            self.target_percent,
            self.target_cpu_slots,
            self.workers_per_job,
            self.torch_threads_per_job,
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
            or isinstance(self.requested_reserved_cpus, bool)
            or not isinstance(self.requested_reserved_cpus, int)
            or self.requested_reserved_cpus < 0
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
        expected_effective = min(
            float(self.affinity_cpus),
            self.quota_cpus or float(self.affinity_cpus),
        )
        expected_reserve = min(
            self.requested_reserved_cpus,
            max(0, math.floor(expected_effective) - 1),
        )
        usable = max(1, math.floor(expected_effective) - expected_reserve)
        expected_target = max(
            1,
            min(
                usable,
                math.floor(expected_effective * self.target_percent / 100 + 0.5),
            ),
        )
        expected_estimate = max(
            float(self.torch_threads_per_job),
            self.workers_per_job / 2,
        )
        expected_jobs = min(
            self.max_parallel_jobs,
            max(1, math.ceil(expected_target / expected_estimate)),
        )
        expected_jobs = min(
            expected_jobs,
            max(1, expected_target // self.torch_threads_per_job),
        )
        if (
            self.effective_cpus != expected_effective
            or self.reserved_cpus != expected_reserve
            or self.target_cpu_slots != expected_target
            or self.estimated_cores_per_job != expected_estimate
            or self.parallel_jobs != expected_jobs
        ):
            raise ValueError("training CPU plan capacity is inconsistent")

    def manifest(self) -> dict[str, object]:
        return {
            "version": self.version,
            "affinity_cpus": self.affinity_cpus,
            "quota_cpus": self.quota_cpus,
            "effective_cpus": self.effective_cpus,
            "target_percent": self.target_percent,
            "requested_reserved_cpus": self.requested_reserved_cpus,
            "reserved_cpus": self.reserved_cpus,
            "target_cpu_slots": self.target_cpu_slots,
            "unallocated_cpu_slots": max(
                0, math.floor(self.effective_cpus) - self.target_cpu_slots
            ),
            "training_cpu_ids": list(self.training_cpu_ids),
            "workers_per_job": self.workers_per_job,
            "torch_threads_per_job": self.torch_threads_per_job,
            "estimated_cores_per_job": self.estimated_cores_per_job,
            "parallel_jobs": self.parallel_jobs,
            "max_parallel_jobs": self.max_parallel_jobs,
            "estimated_target_satisfied": (
                self.parallel_jobs * self.estimated_cores_per_job
                >= self.target_cpu_slots
            ),
            "scope": "independent-training-jobs-only",
            "trial_behavior_changed": False,
        }

    def policy_manifest(self) -> dict[str, object]:
        return {
            "version": "r3b-training-cpu-policy-v1",
            "target_percent": self.target_percent,
            "reserved_cpus": self.requested_reserved_cpus,
            "max_parallel_jobs": self.max_parallel_jobs,
            "workers_per_job": self.workers_per_job,
            "torch_threads_per_job": self.torch_threads_per_job,
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
    target_percent: int = 100,
    reserved_cpus: int = 0,
    max_parallel_jobs: int = 16,
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
            available_ids = _topology_cpu_order(tuple(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            available_ids = tuple(range(os.cpu_count() or 1))
        affinity_cpus = max(1, len(available_ids))
        if quota_cpus is None:
            quota_cpus = _finite_cgroup_quota()
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
        affinity_cpus=affinity_cpus,
        quota_cpus=quota_cpus,
        effective_cpus=effective,
        target_percent=target_percent,
        requested_reserved_cpus=reserved_cpus,
        reserved_cpus=resolved_reserve,
        target_cpu_slots=target,
        training_cpu_ids=available_ids[:target],
        workers_per_job=workers_per_job,
        torch_threads_per_job=torch_threads_per_job,
        estimated_cores_per_job=estimated,
        parallel_jobs=jobs,
        max_parallel_jobs=max_parallel_jobs,
    )
