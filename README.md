# irisu-rl

This is the complete project repository for IriSu mechanics research and
reinforcement-learning work. It includes a deterministic, asset-free C++20
normal-mode simulator, a dependency-free Python environment, validation and
benchmark tooling. It is intended for policy
search and eventual transfer testing against an authorized local copy of the
original game.

## Project map

- [`clone`](clone): native C++ simulator and C API
- [`python/irisu_env`](python/irisu_env): Python and Gymnasium-shaped environment
- [`python/irisu_rl`](python/irisu_rl): versioned neural encoders, semantic
  actions, exact-runtime attestation, and owned vector rollout plumbing
- [`tests`](tests): native, Python, integration, and fidelity tests
- [`benchmarks`](benchmarks): training-readiness and performance benchmarks
- [`configs`](configs): measured mechanics configurations
- [`tools`](tools): capture, validation, and exact-physics tooling
- [`docs`](docs) and [`reference`](reference/README.md): design and clean-room research records
- [`RL.md`](RL.md) and [`docs/rl-r0-r1.md`](docs/rl-r0-r1.md): transfer roadmap
  and the implemented R0/R1 contract

## Simulator details

The simulator targets IriSu Syndrome v2.03 normal puzzle mode.

The physics layer is the official zlib-licensed Box2D 1.4.3 SourceForge SVN r58
engine behind a measured compatibility adapter. The normal-mode RNG, replay
cadence, input edges, spawning, dispatcher, scoring, gauge, lifetimes, actor
pool, seeded 20-block reset prefill, and level progression are clean-room
implementations of recovered v2.03 behavior. The default configuration is the
executable's mode-0 production table (`0xec0e8463feaf2670`), not the separate
nonzero-mode/Metsu-side table mirrored by the shipped INI.

With the separately generated 32-bit exact-MSVC physics host, all four eligible,
instrumented original v2.03 playbacks now match the headless simulator on every
score and rot event and on score, gauge, level, chain, clears, and tick at the
same terminal or replay-exhaustion checkpoint. This covers 536 score calls and
57,921 ticks; the longest trace is exact through all 47,019 updates. The same
gate now runs through the public production `IrisuEnv` exact-worker backend,
not only the standalone native runner. It matches all 1,111 original state
checkpoints: 536 score, 103 rot, 431 qualifying-clear, and 41 level events,
including the reconstructible score/gauge/level/clear state at each event. The
[production replay artifact](benchmarks/results/exact-production-replay-parity-2026-07-21.json)
has SHA-256
`b0e5def9d05eab34f76a43c0bdc23a2ecb83e414223a5ad06bb1d06c500d1848`.
The
low-level stream comparison covers all 813,412 active mutation, step, and
contact records, including 573,557 contact results. The original trace's
9,810,360 getter-only records were also replayed in their original global call
order against that exact host. All 12,262,950 returned binary32 words match
bit-for-bit through step 47,019, with no getter or contact mismatch. The
ordinary GNU physics build remains the portable default and is not claimed to
preserve chaotic long-horizon trajectories bit-for-bit. The exact host is
available as an opt-in Python backend through one isolated 32-bit worker per
active episode. Production resets replace the worker instead of rebuilding
another pristine-r58 world in the same process. Policy transfer is still
unproven.

## Build and test

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
PYTHONPATH=python python3 -m unittest discover -s tests -p 'test_*.py' -v
```

The supported fidelity-reference profile is an optimized GNU C++ build on
x86/x86-64. CMake reports whether the shipped-DLL bit oracle is enabled or why
it is skipped. The native build metadata records the compiler, configuration,
system processor, pointer width, legacy floating-point mode, and enforced
floating-point environment so results can be compared only across compatible
builds.

### Install and Python package

CMake installs the native CLI, shared C ABI library, and public C header:

```bash
cmake --install build --prefix "$PWD/install"
```

The installed header defines stable `IRISU_*` constants for every integer
action, body, shape, lifecycle, and event kind exposed by the C ABI.

Install the thin Python package into the active environment from the repository
root:

```bash
python3 -m pip install .
```

The `irisu-env` wheel is intentionally thin and pure Python: it neither bundles
nor rebuilds the native simulator. Build or install the native library
separately, then set `IRISU_CLONE_LIBRARY` to its full path (or pass
`library_path=` to the Python API):

```bash
export IRISU_CLONE_LIBRARY="$PWD/install/lib/libirisu_clone.so"
python3 -c 'from irisu_env import IrisuEnv; IrisuEnv().close()'
```

Libraries installed into a standard system search path may be discovered by
the platform loader without the environment override. The explicit path is the
most reproducible choice and also works with a wheel installed in an isolated
environment. Platform-specific library names and install directories may
differ.

The normal test suite needs no original binary or copyrighted asset. The
optional DLL oracle requires the ignored authorized reference copy described in
[`reference/README.md`](reference/README.md):

```bash
reference/probe/test.sh
```

## Run headlessly

The native CLI accepts one action per input line and emits JSONL state summaries:

```bash
printf 'wait 20\nweak 250 350\nwait 20\nstate\nquit\n' | \
  build/irisu-headless --seed 42
```

Python exposes Gymnasium-shaped methods without requiring Gymnasium:

```python
from irisu_env import Action, IrisuEnv

with IrisuEnv() as env:
    observation, info = env.reset(seed=42)
    assert len(observation["bodies"]) == 20
    observation, reward, terminated, truncated, info = env.step(
        Action.strong(300, 360)
    )
    snapshot = env.clone_state()
    env.restore_state(snapshot)
    assert isinstance(env.state_hash(), int)
```

For trajectory-exact physics, build the `irisu-exact-worker` target from a
32-bit `IRISU_PHYSICS_BACKEND=exact-msvc` configuration, then opt in explicitly:

```bash
python3 tools/host-msvc9-box2d-multiworld.py \
  --object-dir /path/to/msvc9-r58-objects \
  --output-dir /new/exact-host-output

cmake -S . -B build-exact -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-m32 -DCMAKE_EXE_LINKER_FLAGS=-m32 \
  -DCMAKE_SHARED_LINKER_FLAGS=-m32 \
  -DIRISU_PHYSICS_BACKEND=exact-msvc \
  -DIRISU_EXACT_BOX2D_LIBRARY=/new/exact-host-output/libirisu_box2d_msvc_exact_multiworld.so
cmake --build build-exact -j
```

The exact host is a local artifact and is not stored in this repository; see
[`reference/native-box2d/multiworld/README.md`](reference/native-box2d/multiworld/README.md)
for the source-build form and validation boundary.
The generated host has a stable `DT_SONAME`, is retained as a `DT_NEEDED`
dependency, and is reopened by the forward wrapper with `RTLD_NOLOAD`. Exact
post-link checks require a nonempty `RPATH`/`RUNPATH` made only of absolute or
`$ORIGIN`-relative components; empty/current-directory and bare relative search
components fail the build. Installed executables use origin-relative lookup and
the exact host is installed in the configured library directory.

```python
from irisu_env import Action, IrisuEnv

with IrisuEnv(
    physics_backend="exact",
    worker_path="build-exact/irisu-exact-worker",
) as env:
    observation, info = env.reset(seed=42)
    observation, reward, terminated, truncated, info = env.step(Action.wait())
```

`IRISU_EXACT_WORKER` can replace `worker_path`. The exact backend supports the
same configuration overrides and public observation/event/diagnostic shapes.
Production Python launches the worker with working directory `/`, removes every
inherited `LD_*` variable and `GLIBC_TUNABLES`, and forces x87 control word
`0x027f`, so caller loader injection and a hostile control-word override do not
cross the process boundary.
Its snapshots store the seed and complete accepted action log, then restore
atomically into a fresh worker by replaying that log. This preserves otherwise
unserializable legacy solver state exactly. Snapshot identity includes the
mechanics configuration, exact Box2D host, and executed worker SHA-256. A
normal worker is hashed through `/proc/<pid>/exe` only after its valid Hello
proves `exec` completed, avoiding a `posix_spawn` race with the parent Python
executable. The client also identifies the exact library actually mapped by
the live worker and checks its path, device, inode, ELF segments, worker and
client mount identities, and bytes against the handshake SHA-256. Before
constructing a simulator, the forward wrapper resolves 15 public `b2d_*`
symbols into typed function pointers; those stored pointers are the actual
physics call path. Every target must be a unique executable address owned by
that one captured host. Its `dladdr1` symbol name and exact symbol-start address
must equal the requested entrypoint, and its process-global binding must equal
the stored pointer. Tests reject both an ordinary `LD_PRELOAD` definition such
as `b2d_world_step` and an interposed-`dlsym` same-library X/Y permutation before
simulation. Worker opcode 13 reports the attested target count and device/inode
mapping, which Python requires to equal its independently captured live library.
The host's private `msvc_b2d_*` bridge calls are separately linked with
`-Bsymbolic-functions`; the host generator and exact build reject every
`R_386_*` relocation naming those helpers. These are fail-closed provenance
checks against the tested loader attacks, not a sandbox for arbitrary code
already executing inside the worker. Restore time still grows linearly with
episode age.
`state_hash()` is therefore a stable restorable-history identity, not a
canonical hash proving that two different action histories reached the same
physical state.
`clone_state()` also refuses while a split exact step is pending, so a request
already accepted by the worker cannot be omitted from the durable action log.

Exact production resets transactionally launch, configure, and identity-check
a fresh worker before replacing the old one. This contains a process-global
pristine-r58 allocator failure found by repeated in-process world teardown.
A 50-episode stress run completed 58,534 decisions and 11,843,733 events with a
fresh worker PID for every reset. Low-level `ExactWorkerClient` users should
likewise treat a worker as single-episode; the protocol now rejects a second
reset and any post-reset reconfiguration. `ExactSimulator`, `IrisuEnv`, and
their worker-backed vector APIs transparently enforce the safe multi-episode
lifecycle. Direct in-process exact C ABI/C++ use remains a one-episode
diagnostic path, not a trainable multi-episode backend.

The retained exact host accelerates the MSVC runtime's observed paired
`FCOS`/`FSIN` sequence with one x87 `FSINCOS`. It keys the cached float-rounded
sine by the raw input bits and falls back to standalone `FSIN` when the next
sine input differs. Of the 13,096,069 measured matching inputs, 833,228
(6.362%) are raw positive zero. `FCOS(+0)` now stores the exact `+0` sine and
returns exact `1` without executing `FSINCOS`; negative zero and ordinary
nonzero inputs retain the general paired path. At raw absolute-angle bits
`>= 0x5f000000` (`|angle| >= 2**63`), where x87 `FSINCOS` cannot reduce the
argument, the shim executes direct `FCOS` then `FSIN` to preserve the original
intrinsic behavior for every finite snapshot angle. A controlled local
dense-core A/B against the paired-only host improved another 1.287%; that
isolated comparison is not a field in the wide pipeline artifact.

The public multiworld bridge remains serialized, so this small runtime cache
cannot interleave across worlds. No Box2D solver iteration, operation order, or
game rule changed. The exact replay gate remains 4/4 through all 47,019 steps.

The retained host passes 14/14 exact Release and 14/14 exact ASAN/UBSAN CTest
targets, including its dedicated trig runtime test, both replay execution paths,
production `IrisuEnv`, packed fast-path, vector snapshot/concurrency, and hostile
`LD_PRELOAD` fixtures, plus the experimental actor-rollout regression. Sanitized
preload tests put the worker-linked ELF32
`libasan` first, then exercise the instrumented fixture. A regression explicitly
records the boundary: the GNU worker, wrapper, API, and fixtures are
ASAN/UBSAN-instrumented; immutable MSVC9 host instructions are not, although
their allocations cross ASAN's interposed allocator. Portable Release and
ASAN/UBSAN builds each pass 8/8 CTest targets. The combined Python suite passes
204 tests with three expected skips (two optional Gym checks and the
sanitizer-only boundary assertion in a normal build).

Linux exact workers also provide an opt-in constant-time local branch path:

```python
from irisu_env import Action, IrisuEnv

with IrisuEnv(
    physics_backend="exact",
    worker_path="build-exact/irisu-exact-worker",
) as env:
    env.reset(seed=42)
    with env.fast_checkpoint() as checkpoint:
        with checkpoint.branch() as branch:
            branch.step(Action.wait(20))
```

The dormant checkpoint can create repeated independent branches while its
source remains open. Release refuses while a branch is alive, and source reset
or restore refuses while it owns an open checkpoint. These fork/COW handles are
Linux-local process capabilities, not serializable state; every branch still
supports the ordinary durable action-log `clone_state()` format. Resetting or
restoring a branch detaches it from its parent keeper. Branch connection
authenticates the checkpoint keeper, verifies direct ancestry and inherited
worker/library file identities, and inherits the launch-verified hashes; an
explicit provenance request rehashes the library mapped by that live branch.
At an action history of
1,000 steps, the current source-manifested local measurement puts checkpoint
creation at 0.171 ms, median branch creation at 0.284 ms, and median durable
restore at 95.479 ms: a 336.274x median branch advantage.

Seeds use the target game's unsigned 32-bit RNG domain. Values outside
`0..2**32-1` are rejected instead of silently aliasing another run.

Authorized normal-mode `.rpy` traces can be mapped into raw button-level
actions (the simulator derives fresh edges) and compared diagnostically with
the clone:

```bash
python3 tools/evaluate-rpy.py path/to/replay.rpy --layout padded
```

The report includes expected header versus clone outcomes, action/state hashes,
and exact action counts. One replay record is one 0.020-second gameplay update,
including during fast-forward. Header offset `+0x10` records mode; the supported
target is mode 0. External score headers remain diagnostic metadata unless the
same bytes have an observed playback on the supported executable.

To exercise the actual production exact backend and compare the full local
corpus with the observed v2.03 event bundles:

```bash
python3 tools/evaluate-exact-replay-corpus.py \
  --worker build-exact/irisu-exact-worker \
  --require-observed-parity
```

This path launches `IrisuEnv(physics_backend="exact")`, verifies the executable
and live mapped host before and after each replay, reconstructs every score,
gauge, level, and qualifying-clear checkpoint, and fails if any available
original-playback checkpoint differs.

Regenerate the exact original-observed seed-41 score regression with:

```bash
python3 tools/generate-controlled-rpy.py /tmp/score-seed41-parity.rpy \
  --preset score-seed41-parity --library build/libirisu_clone.so
```

Observed reference-game scenarios use a separate, fail-closed gate. Its tracked
five-category manifest is currently empty, so this exact-backend command exits
`2` (`not_evaluable`) rather than claiming fidelity:

```bash
PYTHONPATH=python python3 tools/score-golden.py \
  reference/golden/manifest.json \
  --worker build-exact/irisu-exact-worker
```

Use `--library build/libirisu_clone.so` instead to score the portable backend;
`--worker` and `--library` are mutually exclusive. Exact reports hash the
executed worker and the live mapped exact library before and after each
scenario and fail closed on identity, mapping, or byte changes.

See [`reference/golden/README.md`](reference/golden/README.md) for the evidence
admission rules, schema, category coverage, and exact 95% calculation.

Use `SyncVectorEnv` for independent environments. `ThreadVectorEnv` sends exact
worker actions in batches up to its `workers=` cap before draining responses,
so worker processes can advance concurrently without silently oversubscribing
the requested limit. Exact `PaddedVectorEnv` sends each capped wave first, then
polls and drains response-ready lanes so a slow low-numbered lane cannot hold
completed workers behind it; it still drains every sent lane and reports the
lowest-numbered failure deterministically. `PaddedVectorEnv` and its
`FastVectorEnv` alias also
support `physics_backend="exact"`. Their `StepPadded` response contains a packed
typed observation, transition diagnostics, and event count but no event
records; `info["events"]` issues `FetchEvents` only if it is materialized. A
randomized parity regression compared ordinary `Step` with this packed path for
1,906 steps and all 190,501 emitted events. Event materialization has a 4 MiB
wire cap; an oversized fetch returns a bounded error without poisoning the lane.
Numeric mechanics overrides can be passed as `IrisuEnv(config={...})` or through
`reset(options={"config": {...}})`. Reset and step `info` identify the active
configuration hash; incompatible snapshots are rejected atomically. A native
`Simulator` validates its immutable configuration and computes that hash once
at construction, then reuses the cached value in diagnostics, transitions, and
snapshots. This removes repeated serialization from the step path without
changing configuration identity. Policy observations contain neither that hash
nor finish/replay bookkeeping. Terminal metadata is available under step
`info["diagnostics"]`. The hidden-state hash
is available only through `env.state_hash()` (or in reset/step `info` with
`diagnostic_hashes=True`), never as a policy observation.
Previous left/right button levels are included in observations because they
determine whether the next level produces a fresh edge.
Body positions are display units. Physics-owned body velocities deliberately
expose the wrapper's raw Box2D world units (weak/strong shots begin at -25/-50
with magnification 10); scripted falling actors retain their display-unit
float32 descent velocity until activation.

Exact `PaddedVectorEnv` chooses an adaptive default of
`min(num_envs, 4 * process-visible logical CPUs)` because each lane owns an
independent worker process. This is a topology heuristic from one supplemental
scaling sample per width, not a demonstrated universal optimum. Process
affinity is honored when available; `workers=` remains authoritative. Portable
vectors retain their eight-thread
ceiling. A supplemental 16-logical-CPU scaling probe measured 4,606.714
decisions/s for the old eight-worker behavior and 10,975.352/s for 64 workers
at 64 lanes. That probe is scheduling evidence, not the formal performance
gate, and hard CPU affinity is not enabled by default.

The experimental `irisu_env.rollout.ExactActorRolloutPool` can run an
independent policy for several consecutive decisions inside each lane task,
removing the cross-lane barrier at every decision. It preserves exact packed
payloads, state hashes, event counts, lane order, snapshots, and deterministic
failure selection; `event_mode="full"` retains event bytes before advancing and
defers only their Python decode. It is useful for independent actor collection,
not a drop-in batched neural-policy API. Policy calls execute concurrently, so
stateful policies must be lane-private or thread-safe. A failed collection can
commit successful sibling lanes; restore a known checkpoint after a policy
failure, and recreate the pool after a worker transport/protocol failure.

Transfer robustness is opt-in and separately seeded. Mechanics uncertainty is
sampled with `randomized_config`; action/perception uncertainty uses the public
API only through `TransferRobustnessEnv`:

```python
from irisu_env import IrisuEnv, ParameterRange, TransferRanges, TransferRobustnessEnv

ranges = TransferRanges(
    action_delay_ticks=ParameterRange(1, 3, integer=True),
    observation_delay_ticks=ParameterRange(0, 2, integer=True),
    cursor_error_x=ParameterRange(-2.0, 2.0),
    cursor_error_y=ParameterRange(-2.0, 2.0),
    position_noise=ParameterRange(-0.5, 0.5),
    velocity_noise=ParameterRange(-0.1, 0.1),
    detection_drop_probability=0.02,
    merge_distance=4.0,
)
with TransferRobustnessEnv(IrisuEnv(), ranges, transfer_seed=17) as env:
    observation, info = env.reset(seed=42)
```

Delays are gameplay ticks; the width of the action-delay range is timing
jitter. Cursor errors are clamped to the 640x480 client. Nearby detections form
deterministic connected components represented by one visibly ambiguous body.
Delayed observations never expose the undisclosed current observation in
`info`; reward and termination always describe the current native transition.
Terminal transitions flush the delay so their final observation is coherent.
All default transfer ranges are zero, preserving the nominal environment.

Diagnostic rendering returns deterministic, self-contained SVG and never loads
original art. Baseline policies and the reproducible benchmark harness are
available as:

```bash
PYTHONPATH=python python3 benchmarks/throughput.py \
  --library build/libirisu_clone.so \
  --output benchmarks/results/local.json
```

Benchmark results are local engineering measurements, not evidence of gameplay
fidelity or policy-transfer readiness. The benchmark contract and recorded
baseline are documented in [`benchmarks/README.md`](benchmarks/README.md).
The current [adaptive-wide source-manifested exact profile](benchmarks/results/exact-pipeline-adaptive-wide-perf-2026-07-21.json)
measures 1,370.539/1,988.302/3,569.759/5,679.407/7,977.156/8,917.207/
9,685.170 packed/lazy decisions/s for 1/4/8/16/32/48/64 lanes; raw packed IPC
reaches 10,333.568/s at 64 lanes. The 64-lane result is 48.426% of the 20,000
decisions/s RL target, or 2.065x short. Its SHA-256 is
`4067fdff9360989adb696bdc5ad7d98983729f9fa424271fbd7e3e1fb9164eef`.
The artifact records 38/38 current runtime-source hashes, worker SHA-256
`4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261`,
host SHA-256
`ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5`,
and 88/88 true cross-path equivalence leaves. The run measures 1,447.881 dense
native wall decisions/s (1,485.277/s for step plus observation) and 74,853.849
physics ticks/s on the directly comparable 30,000-tick 48-body workload.

The isolated [paired Python hot-path A/B](benchmarks/results/exact-padded-python-hot-path-ab-2026-07-21.json)
measures 7,610.288 to 8,066.670 decisions/s at 32 lanes (+5.997%) on six
interleaved, workload-equivalent 80,000-decision samples. Its SHA-256 is
`0f7a5c6820cd002d190f177f45ba0f0db44c7cf7387c4527d154c8a30299fbbd`;
it attributes the action-packing and packed-suffix changes but does not replace
the full performance gate.

The supplemental [scaling and scheduling probe](benchmarks/results/exact-padded-scaling-ceiling-2026-07-21.json)
scales equal exact lanes/workers from 4,674.992/s at 8 to 10,975.352/s at 64
without reaching a ceiling on this 8-core/16-thread host. Its SHA-256 is
`2938d3e072ee99e39ba408f0dd934e5e5caa82993e8e1d7472a6b3322d4f4657`.
The separate [actor-rollout A/B](benchmarks/results/exact-actor-rollout-ab-2026-07-21.json)
matches synchronous trajectory payloads, state hashes, and event counts at
16/32/64 lanes. All 9 paired samples are exact; with 64-decision horizons, its
median paired speedups are 1.211x/1.097x/1.057x. Its SHA-256 is
`2f247f1222f0423475bdcffc185ea893c0b31eb204a7fb6df55030c396d6fc4f`.
Both artifacts use focused workloads and supplement rather than replace the
source-manifested gate.

A separate [exact-core cache investigation](benchmarks/results/exact-core-trig-cache-investigation-2026-07-21.json)
rejects broader raw-angle memoization: 10,080,004 of 13,096,069 observed inputs
are unique, a 4,096-entry cache improves the dense median by only about 1.04%,
and the immutable contact solver accounts for 58.36% of fresh profile samples.
Two exact-preserving solver-source candidates were also rejected. The
[solver optimization artifact](benchmarks/results/exact-core-solver-source-optimizations-2026-07-21.json)
records only +0.648% for skipping static-body position stores and +0.472% for
caching velocity anchors, below the predeclared 3% integration threshold. Both
matched the full 47,019-step exact replay and 813,508-record wrapper trace, but
the expensive full getter replay was intentionally not run for rejected
candidates and no engine change was retained. Artifact SHA-256:
`6fe2b8c482e8764ff64577261d839d552ae7c4a5a996c538bbead2a135ffcb71`.
The candidate sources and binaries were local, unarchived experiment inputs;
the artifact preserves their hashes and descriptions but is not independently
rebuildable from a clean checkout.

The July 20 [paired-trig artifact](benchmarks/results/exact-pipeline-paired-trig-2026-07-20.json)
remains the historical directly comparable post-trig run: against the
[pre-trig baseline](benchmarks/results/exact-pipeline-final-2026-07-20.json), its
pinned core A/B improved 16.14%, dense native simulation improved 16.82%, and
the 48-body 30,000-tick physics workload improved 14.37%. The retained `+0`
fast path adds 1.287% in a controlled local dense-core A/B against that paired
host.
The recorded action/reset counts and body/event density summaries remain
equivalent across all benchmark paths, but solver work still dominates.
Observed replay physics and reward parity are passing gates, but bulk-RL
readiness is not. The representative exact backend remains below the numerical
20,000 decisions/s target, but the measured limitation was explicitly accepted
as sufficient for RL on 2026-07-21 and is no longer a blocker. The tracked
five-category controlled manifest is empty and `not_evaluable`; no hashed
original-game spawn/difficulty distribution comparison exists; and no scripted
policy has demonstrated qualitative transfer.
The default portable backend also remains unsuitable when replay-exact physics
is required.

## Evidence boundary

- [`clone.md`](clone.md) is the north-star contract and definition of done.
- [`docs/mechanics.md`](docs/mechanics.md) defines units, schemas, provenance,
  and the uncertainty register.
- [`configs/v2.03-normal.toml`](configs/v2.03-normal.toml) records recovered
  normal-path evidence, out-of-scope raw shipped keys, and explicitly ignored
  compatibility-only API fields.
- [`docs/physics-source.md`](docs/physics-source.md) records the exact legacy
  source and local safety adaptations.
- [`docs/fidelity.md`](docs/fidelity.md) says what currently passes, what has
  not been calibrated, and what must not yet be claimed.
- [`reference/probe`](reference/probe) contains the redistributable DLL probe,
  validator, and numerical golden trace. The original DLL itself stays ignored.

The simulator never requires story data, images, audio, saves, archives, or the
original executable. Copyrighted reference-only materials remain outside
distributable source control.
