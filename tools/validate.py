#!/usr/bin/env python3
"""Run the repository's independent validation jobs concurrently."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_rl.cpu_parallelism import available_cpu_capacity  # noqa: E402


THREAD_LIMIT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
PYTHON_TASK_SLOTS = {
    "test_physics_lifecycle.py": 8,
    "test_r3b_supervisor.py": 9,
}
NATIVE_CONCURRENCY_SLOTS = 8


@dataclasses.dataclass(frozen=True)
class Task:
    label: str
    command: tuple[str, ...]
    slots: int = 1


@dataclasses.dataclass(frozen=True)
class Result:
    label: str
    command: tuple[str, ...]
    returncode: int
    output: str
    seconds: float


class Capacity:
    def __init__(self, slots: int) -> None:
        self._available = slots
        self._condition = threading.Condition()

    def acquire(self, slots: int) -> None:
        with self._condition:
            self._condition.wait_for(lambda: slots <= self._available)
            self._available -= slots

    def release(self, slots: int) -> None:
        with self._condition:
            self._available += slots
            self._condition.notify_all()


def available_cpus() -> int:
    try:
        _affinity, _quota, effective = available_cpu_capacity()
        return max(1, math.floor(effective))
    except (OSError, RuntimeError):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            return max(1, os.cpu_count() or 1)


def validation_environment(root: Path = ROOT) -> dict[str, str]:
    environment = os.environ.copy()
    python_path = str(root / "python")
    if existing := environment.get("PYTHONPATH"):
        python_path = os.pathsep.join((python_path, existing))
    environment["PYTHONPATH"] = python_path
    for variable in THREAD_LIMIT_VARIABLES:
        environment[variable] = "1"
    return environment


def discover_python_tasks(
    root: Path = ROOT,
    python: str = sys.executable,
    jobs: int | None = None,
) -> list[Task]:
    paths = list((root / "tests").rglob("test_*.py"))
    # Starting large modules first is a cheap, stable approximation of longest-job-first.
    paths.sort(
        key=lambda path: (-path.stat().st_size, path.relative_to(root).as_posix())
    )
    capacity = jobs if jobs is not None else available_cpus()
    return [
        Task(
            f"python:{path.relative_to(root).as_posix()}",
            (python, "-m", "unittest", "-v", str(path.relative_to(root))),
            slots=min(capacity, PYTHON_TASK_SLOTS.get(path.name, 1)),
        )
        for path in paths
    ]


def discover_web_tasks(root: Path = ROOT) -> list[Task]:
    paths = sorted((root / "apps" / "web" / "tests").rglob("*.test.mjs"))
    if not paths:
        return []
    node = shutil.which("node")
    if node is None:
        return [
            Task(
                "web:node-unavailable",
                (
                    sys.executable,
                    "-c",
                    "import sys; sys.exit('Node.js is required for web tests')",
                ),
            )
        ]
    return [
        Task(
            f"web:{path.relative_to(root).as_posix()}",
            (node, "--test", str(path.relative_to(root))),
        )
        for path in paths
    ]


def native_task(build_dir: Path, jobs: int) -> Task:
    native_slots = min(jobs, max(NATIVE_CONCURRENCY_SLOTS, jobs // 4))
    return Task(
        "native:ctest",
        (
            "ctest",
            "--test-dir",
            str(build_dir),
            "--parallel",
            str(native_slots),
            "--output-on-failure",
        ),
        slots=native_slots,
    )


def _run_task(
    task: Task,
    capacity: Capacity,
    environment: dict[str, str],
    root: Path,
) -> Result:
    capacity.acquire(task.slots)
    started = time.monotonic()
    try:
        try:
            completed = subprocess.run(
                task.command,
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            returncode = completed.returncode
            output = completed.stdout
        except OSError as error:
            returncode = 127
            output = f"{error}\n"
    finally:
        seconds = time.monotonic() - started
        capacity.release(task.slots)
    return Result(
        task.label,
        task.command,
        returncode,
        output,
        seconds,
    )


def run_tasks(
    tasks: list[Task],
    jobs: int,
    *,
    root: Path = ROOT,
    environment: dict[str, str] | None = None,
) -> list[Result]:
    if jobs < 1:
        raise ValueError("validation CPU slots must be positive")
    if any(task.slots < 1 or task.slots > jobs for task in tasks):
        raise ValueError("task CPU slots must fit the validation budget")
    if not tasks:
        return []
    capacity = Capacity(jobs)
    environment = environment or validation_environment(root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(_run_task, task, capacity, environment, root)
            for task in tasks
        ]
        results = [future.result() for future in futures]
    return sorted(results, key=lambda result: result.label)


def render_results(results: list[Result], *, verbose: bool = False) -> bool:
    passed = True
    for result in results:
        status = "PASS" if result.returncode == 0 else "FAIL"
        print(f"{status} {result.label} ({result.seconds:.2f}s)")
        if verbose or result.returncode:
            output = result.output.rstrip()
            if output:
                print(f"--- {result.label} output ---")
                print(output)
        passed &= result.returncode == 0
    return passed


def _run_build(
    build_dir: Path,
    jobs: int,
    environment: dict[str, str],
    verbose: bool,
) -> bool:
    if not (build_dir / "CMakeCache.txt").is_file():
        configure = Task(
            "build:configure",
            (
                "cmake",
                "-S",
                str(ROOT),
                "-B",
                str(build_dir),
                "-DCMAKE_BUILD_TYPE=Release",
            ),
        )
        if not render_results(
            run_tasks([configure], 1, environment=environment), verbose=verbose
        ):
            return False
    build = Task(
        "build:compile",
        ("cmake", "--build", str(build_dir), "--parallel", str(jobs)),
        slots=jobs,
    )
    return render_results(
        run_tasks([build], jobs, environment=environment), verbose=verbose
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="tools/validate.py", description=__doc__)
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=available_cpus(),
        help="total CPU slots (default: effective affinity and cgroup quota)",
    )
    parser.add_argument(
        "--build-dir", type=Path, default=ROOT / "build", help="CMake build tree"
    )
    parser.add_argument("--no-build", action="store_true", help="skip native build")
    parser.add_argument("--no-native", action="store_true", help="skip CTest")
    parser.add_argument("--no-python", action="store_true", help="skip Python tests")
    parser.add_argument("--no-web", action="store_true", help="skip web tests")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="show passing job output"
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    args.build_dir = args.build_dir.resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    environment = validation_environment()
    started = time.monotonic()

    if not args.no_build:
        if not _run_build(args.build_dir, args.jobs, environment, args.verbose):
            return 1

    tasks: list[Task] = []
    if not args.no_native:
        tasks.append(native_task(args.build_dir, args.jobs))
    if not args.no_web:
        tasks.extend(discover_web_tasks())
    if not args.no_python:
        tasks.extend(discover_python_tasks(jobs=args.jobs))

    print(f"validation: {len(tasks)} jobs, {args.jobs} CPU slots")
    results = run_tasks(tasks, args.jobs, environment=environment)
    passed = render_results(results, verbose=args.verbose)
    seconds = time.monotonic() - started
    status = "passed" if passed else "failed"
    print(f"validation {status}: {len(results)} jobs in {seconds:.2f}s")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
