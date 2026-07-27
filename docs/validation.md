# Fast validation

Run the complete build and validation suite with:

```bash
uv run --all-extras python tools/validate.py
```

The runner uses the smaller of the process CPU affinity and its cgroup CPU
quota, builds with CMake parallelism, runs CTest concurrently with the web test
and independent Python `unittest` modules, and reports results in stable name
order. Numerical-library thread pools are capped at one thread per Python test
process. Tests with known internal native-thread or subprocess concurrency
reserve their demand in the shared scheduler, up to the selected total budget,
and run alone when that demand reaches the budget. CTest uses matching processor
metadata.

Useful options:

```bash
# Reuse an already-current native build.
uv run --all-extras python tools/validate.py --no-build

# Build, but skip CTest.
uv run --all-extras python tools/validate.py --no-native

# Run only Python and web tests without building.
uv run --all-extras python tools/validate.py --no-build --no-native

# Override the detected CPU budget.
uv run --all-extras python tools/validate.py --jobs 8

# Show the captured output of passing jobs.
uv run --all-extras python tools/validate.py --verbose
```

Failure output is always shown, all scheduled jobs are allowed to finish, and
the runner exits nonzero if the build or any native, Python, or web test fails.
Web tests are included recursively below `apps/web/tests` when files matching
`*.test.mjs` exist.
