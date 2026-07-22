# Exact worker IPC

`ipc_worker.cpp` is a persistent-per-episode 32-bit simulator process. The
reusable client in `python/irisu_env/exact_ipc.py` launches it from ordinary
64-bit Python and exposes typed reset, observation, configuration, step, event,
reward, diagnostic, snapshot, and terminal data. `ipc_client.py` is the
corresponding replay/benchmark CLI. `IrisuEnv(physics_backend="exact")` uses
this protocol in production. The 32-bit C ABI can link the exact backend
directly for bounded one-episode diagnostics, but it is not a multi-episode
training path.

## Build and exercise

The exact multi-world library remains a local build artifact. Configure the
whole exact target as 32-bit and point CMake at that artifact:

```sh
cmake -S . -B build-exact-ipc -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-m32 \
  -DCMAKE_EXE_LINKER_FLAGS=-m32 \
  -DCMAKE_SHARED_LINKER_FLAGS=-m32 \
  -DIRISU_PHYSICS_BACKEND=exact-msvc \
  -DIRISU_EXACT_BOX2D_LIBRARY=/path/to/libirisu_box2d_msvc_exact_multiworld.so \
  -DIRISU_BUILD_SHARED=OFF \
  -DIRISU_BUILD_TESTS=OFF
cmake --build build-exact-ipc --target irisu-exact-worker

python3 tools/exact-physics-prototype/ipc_client.py \
  --worker build-exact-ipc/irisu-exact-worker smoke
python3 tools/exact-physics-prototype/ipc_client.py \
  --worker build-exact-ipc/irisu-exact-worker benchmark --workers 4
python3 tools/exact-physics-prototype/ipc_client.py \
  --worker build-exact-ipc/irisu-exact-worker replay replay.rpy
```

The worker reports its pointer width, x87 control word, mechanics config hash,
backend name, compiler, and exact-library SHA-256 during its handshake. The
client rejects incompatible pointer width, protocol version, body capacity,
process model, backend identity, or a missing/placeholder library SHA-256.
Production launch uses working directory `/`, removes every inherited `LD_*`
variable and `GLIBC_TUNABLES`, and forces x87 control word `0x027f`.
Only after a valid Hello proves `exec` completed does the client hash
`/proc/<pid>/exe`, avoiding a `posix_spawn` race that could otherwise identify
the parent Python executable. It also locates the one exact library in the live
worker maps and validates its resolved path, device, inode, required ELF
segments, worker and client mount identities, stable metadata, and complete
bytes against the handshake SHA-256.

Before constructing `Simulator`, the forward wrapper resolves its typed table
of 15 `b2d_*` entrypoints and performs every later physics call through those
stored pointers. Each target must be a unique executable address owned by the
genuine exact-library link-map object. All targets must share one `/proc`
device/inode mapping, and each process-global binding must equal the attested
call target. Each `dladdr1` name and exact symbol-start address must equal the
requested entrypoint. An ordinary `b2d_*` preload therefore fails closed even
when the genuine SONAME is also mapped, and a tested interposed-`dlsym`
same-host X/Y permutation fails the symbol-identity check. The exact host's
internal `msvc_b2d_*` calls are bound locally with `-Bsymbolic-functions`; its
generator and the consuming CMake configuration reject every `R_386_*`
relocation naming those helpers. These are fail-closed provenance checks for
the tested loader attacks, not a general sandbox for arbitrary code already
executing inside the worker.

The generated host has a stable `DT_SONAME`, is retained as a `DT_NEEDED`
dependency, and is reopened only with `RTLD_NOLOAD`. Exact post-link validation
requires a nonempty `RPATH`/`RUNPATH` whose components are absolute or
`$ORIGIN`-relative; empty/current-directory and bare relative components fail
the build. Installed exact executables use origin-relative lookup for the host
installed with the relocatable tree.

## Wire contract

Every little-endian frame has a 16-byte header: magic, protocol version,
opcode, request ID, and payload size. Responses echo the opcode and request ID
and begin with a signed status. Requests and responses have bounded lengths.
The observation prefix and each body are explicitly encoded field by field, so
no native C structure padding crosses the 32/64-bit boundary. Step responses
also carry the full transition diagnostics followed by variable-length UTF-8
events. Configure opcode 6 carries a count followed by length-prefixed UTF-8
key/double pairs; it returns the resulting config hash. Opcode 7 returns the
canonical config JSON. A non-request exception poisons the worker after it
returns one error; bad input requests return an error without mutating the
simulator.

Opcode 8 (`StepPadded`) is the training-oriented packed form. It returns the
observation header, only the live fixed-width bodies, transition diagnostics,
event count, and an event generation; it does not serialize event records.
Opcode 9 (`FetchEvents`) retrieves the event records for that generation only
when a caller materializes the lazy view. Advancing the lane expires an
unmaterialized prior generation, while an already materialized view remains
stable. The fetch is capped at 4 MiB; an oversized materialization returns a
bounded error without poisoning the lane. Opcodes 10--12 implement the Linux-
local fork/COW fast checkpoint, release, and authenticated branch operations
described in `fast_snapshots.md`.

Opcode 13 (`ExactAttestation`) is a backward-compatible protocol-v1 extension.
Its response contains little-endian schema `u32` (currently 1), entrypoint
count `u32` (15), inode `u64`, and a standard length-prefixed ASCII device
string. Python requires valid schema/count/device/inode values and an exact
device/inode match with its independently captured live exact library. Older
workers that do not implement opcode 13 therefore fail closed rather than
silently losing this guarantee. Exact `build_info` records
`exact_call_targets_runtime_verified=true` and `exact_entrypoint_count=15`.

`send_step`/`receive_step` split a round trip. A vector coordinator sends one
action to every process before draining any response, allowing the exact
workers to run concurrently without Python worker threads. Exact
`PaddedVectorEnv` then polls the response descriptors and drains ready lanes,
so a slow low-numbered lane does not hold completed workers behind it. It still
drains every sent request and reports the lowest-numbered failure
deterministically.

## Measured result

On the representative dense `RandomPolicy(max_wait_ticks=1)` workload, explicit
1/4/8/16/32-lane exact packed/lazy-event vectors sustain
1,364.451/2,009.342/3,516.100/5,630.998/7,894.265 decisions/s on the development
host. Raw packed IPC reaches 7,955.692/s at 32 lanes. At eight lanes the
workload averages 110.865 live bodies and 172.954 events per decision and
reaches maxima of 196 bodies and 484 events. Packed
response content averages 11,291 bytes and peaks at 19,804, versus
23,558/54,559 bytes for eager event-bearing responses.

Ten baseline profile runs collected 24,020 samples at 1,267.12 ± 4.43 dense
native decisions/s. The exact host accounted for 92.678% of samples, led by
`SolveVelocityConstraints` (28.426%), `SolvePositionConstraints` (23.393%),
`FCOS` (14.205%), and `FSIN` (11.678%). A 3,000-decision instrumented run found
13,096,069 matching raw-float cosine/sine pairs. The retained runtime shim uses
one x87 `FSINCOS` for such a pair, caches the float-rounded sine under the raw
input key, and retains standalone `FSIN` for a nonmatch. Bridge serialization
prevents cross-world interleaving; solver iterations and operation order are
unchanged.

Of those pair inputs, 833,228 (6.362%) are raw positive zero. The current shim
stores exact sine `+0` and returns exact cosine `1` without running `FSINCOS`
for that case. Raw negative zero and ordinary nonzero inputs retain the general
paired path. Raw absolute-angle bits `>= 0x5f000000` (`|angle| >= 2**63`)
instead use direct `FCOS` then `FSIN`, preserving the original intrinsic
behavior beyond
`FSINCOS`'s argument-reduction range. Boundary vectors and 100,000
full-raw-bit randomized inputs pin the fallback. A controlled dense-core A/B
improves 1.287% over the paired-only host. This is an unarchived local
comparison; the wide pipeline does not isolate that change.

A stable pinned seven-run comparison moved from 1,246.873 to 1,448.173
decisions/s (+16.14%). In the complete source-manifested pipeline, the dense
native simulator improves 16.82%, the 48-body physics workload 14.37%, and
eight-lane padded throughput 13.50%. An attempted combined wrapper getter
changed representative throughput by only +0.009% and the dense case by
+0.187%, both noise-sized, so it remains rejected. The current explicit
64-lane padded result is 48.426% of the 20,000 aggregate-decision/s target, or
2.065x short; solver work remains dominant.

The checked-in source-manifested result is
[`exact-pipeline-adaptive-wide-perf-2026-07-21.json`](../../benchmarks/results/exact-pipeline-adaptive-wide-perf-2026-07-21.json),
SHA-256 `4067fdff9360989adb696bdc5ad7d98983729f9fa424271fbd7e3e1fb9164eef`.
It records 38/38 current runtime-source hashes and 88/88 true equivalence leaves,
worker SHA-256
`4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261`,
host SHA-256
`ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5`,
1,447.881 dense native wall decisions/s (1,485.277/s for step plus observation),
and 74,853.849 ticks/s on the directly comparable 30,000-tick 48-body physics
workload. The July 20
[`exact-pipeline-paired-trig-2026-07-20.json`](../../benchmarks/results/exact-pipeline-paired-trig-2026-07-20.json)
remains the historical comparable post-pair artifact; the
[`exact-pipeline-final-2026-07-20.json`](../../benchmarks/results/exact-pipeline-final-2026-07-20.json)
run is its pre-trig baseline. Timing is a local-host measurement, not a portable
performance guarantee.

All four authoritative v2.03 replays reproduce through the IPC path, including
the active mutation/step/contact stream through all 47,019 physics steps. The
longest reaches score 41,449,
gauge 1, level 38, highest chain 5, 379 clears, 455 score events, and terminal
replay frame 47,018. Score and gauge events reconstruct the observation after
every step in the corpus. Randomized `Step`/`StepPadded` validation additionally
matched 1,906 steps and every one of 190,501 events across five boundary seeds.
The production `IrisuEnv` corpus gate further compares all 1,111 original
score/rot/clear/level state checkpoints and passes 4/4 with an attested worker
and live mapped host. Its report SHA-256 is
`b0e5def9d05eab34f76a43c0bdc23a2ecb83e414223a5ad06bb1d06c500d1848`.
The low-level comparator checks 813,412 active mutation/step/contact records,
including 573,557 contacts. The original trace's 9,810,360 getter-only records
were independently replayed in original global call order against the exact
host. All 12,262,950 returned binary32 words match bit-for-bit through step
47,019, with no getter or contact mismatch.
The retained range-safe positive-zero build passes 14/14 exact Release and
14/14 exact ASAN/UBSAN CTest targets. Portable Release and ASAN/UBSAN builds
each pass 8/8, and the Python suite passes 204 tests with three expected
normal-build skips. Sanitized hostile-preload tests put the worker-linked ELF32
`libasan` first. The GNU layers are instrumented; immutable MSVC9 host
instructions are explicitly verified as uninstrumented.

## Production mapping

`ExactSimulator` in `python/irisu_env/exact_ipc.py` owns the worker and maps
`Observation`, `BodyState`, `EventState`, and `Transition` into the same public
dictionary contract as `NativeSimulator`. `IrisuEnv`, `SyncVectorEnv`, and
`ThreadVectorEnv` accept `physics_backend="exact"`; the threaded vector path
uses split send/drain calls so worker processes advance concurrently.
`PaddedVectorEnv` and its `FastVectorEnv` alias also accept the exact backend
and use packed `StepPadded` responses plus lazy `FetchEvents`; their capped
waves use readiness-ordered response draining.

Exact padded vectors default to
`min(num_envs, 4 * process-visible logical CPUs)` concurrent worker processes;
`os.sched_getaffinity(0)` supplies the CPU count when available. An explicit
exact `workers=` request remains authoritative. Portable padded vectors retain
their conservative eight-worker cap. The exact default is a measured topology
heuristic, not a claim that four workers per logical CPU is universally
optimal.

The new exact library supports independent world handles, but its host bridge
currently serializes calls because pristine r58 contains shared lazy state and
the paired-trig runtime cache is process-local.
One worker process per active lane therefore remains the useful topology for
parallel CPU execution and fault isolation. A later multi-lane worker can save
process overhead, but its worlds will execute serially unless that bridge
constraint is removed.

Configuration overrides now use the same `apply_config_override` core utility
as the C ABI, including flattened array keys and identical numeric validation.
Configuration is atomic: the worker constructs and validates a replacement
simulator before discarding the current one. `ExactWorkerClient.info` remains
the immutable launch handshake, while `current_config_hash` tracks successful
reconfiguration and is checked against config JSON. Snapshot identities must
use the current hash rather than the handshake's default hash.
The underlying native `Simulator` configuration is immutable after validation;
it calculates this hash once at construction and reuses the cached value for
every transition, diagnostic record, and snapshot compatibility check.

After an `ExactSimulator` has been reset once, subsequent resets launch,
configure, and identity-check a fresh worker before atomically swapping
clients. Long multi-episode `StepPadded` stress exposed pristine-r58 process-
global allocator state that eventually crashes when worlds are repeatedly torn
down and recreated inside one process. The production lifecycle completed 50
episodes, 58,534 decisions, and 11,843,733 events with 50 distinct reset worker
PIDs. An open local fast checkpoint must be released before its source
simulator can reset or restore; a branch may reset or restore to detach itself
from its parent keeper. The low-level worker permits exactly one successful
reset and rejects `Configure` afterward. `ExactWorkerClient` callers must use
one worker process per episode; the benchmark CLI now respawns and identity-
checks workers on natural episode boundaries. Direct in-process exact C ABI/C++
execution has the same pristine-r58 process-global hazard and remains a bounded
one-episode diagnostic path.

## Snapshot mapping

The forward exact adapter still cannot export or rebuild contact solver and
broad-phase state directly. Production implements the first two of these three
viable stages:

1. Store seed plus the accepted action log and restore by reset/replay. This is
   exact and portable, but restore time is linear in snapshot age.
2. On Linux, fork only between requests. A dormant copy-on-write keeper is an
   exact process snapshot and can produce repeated authenticated RPC branches.
   Explicit release refuses while branches remain alive; source death
   recursively terminates and reaps its process tree. See `fast_snapshots.md`.
3. Extend the MSVC wrapper with full world export/import, including broad-phase
   ordering, pending destroys, contacts, and accumulated impulses. That is the
   portable long-term solution and needs its own parity campaign.

`ExactSimulator.clone_state()` now emits a checksummed, versioned seed/action
log bound to the mechanics configuration and exact-worker identity.
`restore_state()` validates it, starts a fresh identically configured worker,
replays every accepted action, and swaps workers only after complete success.
`clone_state()` refuses while a split step is pending, preventing a request
already accepted by the worker from being omitted from that action log.
`ExactSimulator.fast_checkpoint()` is the opt-in constant-time local path; its
`ExactFastCheckpoint.branch()` children retain the same durable action history.
`IrisuEnv.fast_checkpoint()` wraps the same capability and returns fully usable
independent `IrisuEnv` branches with the source configuration, render mode,
diagnostic setting, spaces, and exact backend. At a 1,000-action history,
checkpoint creation is 171.215 us, median branch creation is 283.931 us
(3,521.982/s), and median durable restore is 95.479 ms (10.474/s) locally, a
336.274x median branch advantage. The durable snapshot is 28,104 bytes.
Branch connection authenticates the keeper with peer credentials, verifies
direct parentage and inherited executable/library file identities, and then
accepts the launch-verified hashes reported by the inherited worker. An
explicit `exact_library_provenance()` request always rehashes the exact library
mapped by that live branch and compares it with the inherited capture.
Structured r58 export/import remains the portable future option for
constant-time restore without Linux process capabilities.

The raw fast-branch replay regression can use the authoritative replay when it
is supplied locally, but skips cleanly when that optional oracle is absent.
The normal worker, Python, lifecycle, and fast-checkpoint tests remain asset-
free.
