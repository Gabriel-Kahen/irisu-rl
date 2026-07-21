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
Only after a valid Hello proves `exec` completed does the client hash
`/proc/<pid>/exe`, avoiding a `posix_spawn` race that could otherwise identify
the parent Python executable. It also locates the one exact library in the live
worker maps and validates its resolved path, device, inode, required ELF
segments, worker and client mount identities, stable metadata, and complete
bytes against the handshake SHA-256.

Before constructing `Simulator`, the forward wrapper resolves its typed table
of 15 `b2d_*` entrypoints and requires every target to be a unique executable
address owned by the genuine exact-library link-map object. All targets must
share one `/proc` device/inode mapping, and each process-global binding must
equal the attested call target. This rejects symbol interposition even when the
genuine SONAME is also mapped.

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
device/inode match with its independent live-map capture. Older workers that do
not implement opcode 13 therefore fail closed rather than silently losing this
guarantee. Exact `build_info` records
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
1,334.810/1,933.338/3,292.397/5,391.293/7,199.041 decisions/s on the development
host. Raw packed IPC reaches 7,995.455/s at 32 lanes. At eight lanes the
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
32-lane padded result is 35.995% of the 20,000 aggregate-decision/s target, or
2.778x short; solver work remains dominant.

The checked-in source-manifested result is
[`exact-pipeline-range-safe-wide-2026-07-20.json`](../../benchmarks/results/exact-pipeline-range-safe-wide-2026-07-20.json),
SHA-256 `91c8db5feb9d3c8339d101940f05a42d93a4490641745964a0ca427553b8b8e9`.
It records 37/37 current source hashes and 64/64 true equivalence leaves, plus
1,498.136 dense native decisions/s and 75,819.177 ticks/s on the directly
comparable 30,000-tick 48-body physics workload. The earlier
[`exact-pipeline-paired-trig-2026-07-20.json`](../../benchmarks/results/exact-pipeline-paired-trig-2026-07-20.json)
remains the comparable post-pair artifact; the
[`exact-pipeline-final-2026-07-20.json`](../../benchmarks/results/exact-pipeline-final-2026-07-20.json)
run is its pre-trig baseline. Timing is a local-host measurement, not a portable
performance guarantee.

All four authoritative v2.03 replays reproduce through the IPC path, including
the complete 47,019-step physics stream. The longest reaches score 41,449,
gauge 1, level 38, highest chain 5, 379 clears, 455 score events, and terminal
replay frame 47,018. Score and gauge events reconstruct the observation after
every step in the corpus. Randomized `Step`/`StepPadded` validation additionally
matched 1,906 steps and every one of 190,501 events across five boundary seeds.
The retained range-safe positive-zero build passes 10/10 exact CTest targets,
portable Release and ASAN/UBSAN builds each pass 8/8, and the Python suite
passes 159 tests with two optional Gym skips.

## Production mapping

`ExactSimulator` in `python/irisu_env/exact_ipc.py` owns the worker and maps
`Observation`, `BodyState`, `EventState`, and `Transition` into the same public
dictionary contract as `NativeSimulator`. `IrisuEnv`, `SyncVectorEnv`, and
`ThreadVectorEnv` accept `physics_backend="exact"`; the threaded vector path
uses split send/drain calls so worker processes advance concurrently.
`PaddedVectorEnv` and its `FastVectorEnv` alias also accept the exact backend
and use packed `StepPadded` responses plus lazy `FetchEvents`; their capped
waves use readiness-ordered response draining.

Both the default exact path and every portable padded vector use at most eight
concurrent workers. An explicit exact request may use more independent worker
processes—for example,
`PaddedVectorEnv(32, physics_backend="exact", workers=32)`. Merely setting
`num_envs=32` without `workers=32` leaves the conservative eight-worker cap in
place.

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
checkpoint creation is 187.828 us, median branch creation is 479.742 us
(2,056.212/s), and durable restore is 95.933 ms (10.413/s) locally, a 199.968x
median branch advantage.
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
