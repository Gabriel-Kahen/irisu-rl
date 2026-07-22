# Training-readiness baselines and throughput

These tools exercise the **current provisional `v2.03-normal` profile**. They
measure engineering behavior only; they do not certify original-game fidelity
or policy quality.

`rl_r1.py` measures the complete neural-ready collection path: padded vector
execution, validated semantic click macros, active-lane releases, numeric
encoding, autoreset seed ownership, and preallocated rollout writes. Run with
`uv run --extra training python benchmarks/rl_r1.py`.

Build and run the default benchmark from the repository root:

```bash
cmake -S . -B build-benchmark -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build-benchmark
PYTHONPATH=python python3 benchmarks/throughput.py \
  --library build-benchmark/libirisu_clone.so \
  --output benchmarks/results/provisional-local.json
```

A short smoke benchmark is:

```bash
PYTHONPATH=python python3 benchmarks/throughput.py \
  --physics-ticks 200 --physics-warmup 20 --physics-bodies 24 \
  --single-steps 200 --vector-steps 50 --num-envs 2 \
  --snapshot-iterations 200 --warmup 10
```

The JSON report records invocation parameters, seed, host/CPU/Python metadata,
shared-library path and SHA-256, native build information, native configuration
and config hash, profile/config file hashes, the compiler version/flags, exact
Gymnasium/NumPy observation-conversion mode and package versions, the `uv.lock`
identity, and a per-file source manifest. Every policy/API workload embeds its
exact actions as base64 records of little-endian `<BddI>` values plus exact
reset markers and hashes; the physics-only workload records its deterministic
excitation parameters and hashes separately. The benchmark hashes its inputs
before and after measurement and rejects a run if they changed.
State/action/snapshot hashes are reproducible for the same supported build;
elapsed time and rates naturally vary. It also rejects the run unless
sequential JSON, threaded JSON, and typed padded vectors finish the exact same
action/reset trace with the same per-lane state hashes, then replays every reset
and step outside the timed region and compares a canonical digest of all public
transitions and state hashes across the three APIs.

The mechanics configuration of each native `Simulator` is immutable after
validation. Its configuration hash is computed once at construction and cached
for transitions, diagnostics, and snapshots, so benchmarked step loops do not
re-serialize the complete configuration merely to report the same identity.

`SyncVectorEnv` (sequential JSON), `ThreadVectorEnv` (thread-pool JSON), and
`PaddedVectorEnv` (typed native batch) aggregate throughput are measured.
Finished vector lanes reset independently; a terminal lane never resets or
advances another lane. Use `--thread-workers N` to cap parallel execution.
Portable typed vectors remain capped at eight workers. Exact `PaddedVectorEnv`
lanes are separate worker processes, so `workers=None` selects
`min(num_envs, 4 * process-visible logical CPUs)` and an explicit `workers=N`
remains authoritative. The exact pipeline deliberately sets `workers` equal to
every requested `--vector-lanes` width.

`PaddedVectorEnv` removes JSON from the training hot path without changing the
canonical Gym API. It supports both `physics_backend="portable"` and
`physics_backend="exact"`; `FastVectorEnv` is its alias. The portable version-1
C ABI and the exact worker protocol expose all public policy fields in 196 body
slots plus `body_count`, which is the explicit mask boundary. Enum fields are
their documented native numeric values. Observations are double-buffered;
consume a typed view before the next call, or use `to_dict()` to materialize a
stable canonical mapping. Exact `StepPadded` responses contain packed live
bodies, transition diagnostics, an event count, and an event generation, but no
event records. Those records are requested with `FetchEvents` only when the
lazy typed `info["events"]` view is materialized. Materialize that view before
advancing its lane if it must be retained. `info["diagnostics"]` is the typed
transition header; call its `diagnostics()` method only when a stable dictionary
is needed. `len(events)` uses the count without fetching; materialization is
capped at 4 MiB, and an oversized fetch returns a bounded error while the lane
remains usable.

[`results/provisional-training-readiness-2026-07-20.json`](results/provisional-training-readiness-2026-07-20.json)
is the current source-manifested default Release run on the available
eight-core host. It records config `0x06dff74bda7c2070`, snapshot schema 6,
`legacy_fp_mode: "x87"`, and the corrected `nearest,x87-pc53` floating-point
environment. The typed path sustained 4,698.178 aggregate decisions/s and
11,588.056 simulation ticks/s while observing 11/41.005/124
minimum/mean/maximum bodies. Sequential, threaded, and typed paths used the
same trace, finished with the same per-lane state hashes, and produced the same
8,036-record public-transition digest
`cfa60023927bbf31830a9c064f1e6f454a9c0f1c81493aeb6319fbaa5d8b6744`.
An independent second full run reproduced the normalized deterministic payload
exactly (sorted compact JSON after removing the generated timestamp,
git-status hash, elapsed times, and rates; SHA-256
`5cf38c0409c949ae3ad00d9e41934e5f99423a6c030d0a71846cfa16e3b28200`).
The 91-file source manifest SHA-256 is
`34f014a78d7dfce0e2d91c50b856dec251df2874a35df72672fe556e0ac16bdd`;
the artifact SHA-256 is
`074d238c1942c43b2c96bf12bfdeb5b0447a22914697dfa8f53ba425eb49ba23`,
and the measured shared-library SHA-256 is
`72857f6f07c1c255b2ef2b307cb4a56cea3aea32a57f405cfbbf93b67e07d421`.
The 20,000 decisions/s performance gate and fidelity gate are both false, so
the report remains `provisional-not-ready` and is not accepted for bulk
training.

### Exact-worker representative profile

A dense `RandomPolicy(max_wait_ticks=1)` profile on the development host uses
the same actions and resets across eager raw IPC, packed raw IPC,
`ThreadVectorEnv`, and exact `PaddedVectorEnv`. The padded coordinator sends a
capped wave and drains response-ready descriptors instead of waiting in lane
order; every sent lane is still drained and failures are reported by the lowest
lane deterministically. The current explicit
1/4/8/16/32/48/64-lane packed/lazy rates are
1,370.539/1,988.302/3,569.759/5,679.407/7,977.156/8,917.207/9,685.170
decisions/s. Raw packed IPC reaches 10,333.568/s at 64 lanes. At eight lanes the
board averages 110.865 live
bodies and 172.954 events per decision, with maxima of 196 and 484. Packed
response content averages 11,291 bytes and peaks at 19,804; the
`PaddedVectorEnv` frame is exactly `224 + 100 * body_count` bytes and averages
11,311 bytes with a 19,824-byte maximum. Eager event-bearing response content
averages 23,558 bytes and peaks at 54,559. These density, size, action, reset,
and event distributions remain equivalent across the measured APIs.

Ten runs of the pre-optimization dense native core collected 24,020 samples at
1,267.12 ± 4.43 decisions/s. The exact MSVC host accounted for 92.678% of
samples. Its largest symbols were `SolveVelocityConstraints` at 28.426%,
`SolvePositionConstraints` at 23.393%, `FCOS` at 14.205%, and `FSIN` at 11.678%.
A 3,000-decision instrumented run observed 13,096,069 same-raw-float
cosine/sine pairs. The retained runtime shim coalesces matching in-range pairs
with one x87 `FSINCOS`; it caches the float-rounded sine under the raw input
bits and uses standalone `FSIN` for a nonmatching sine. The public multiworld
bridge still serializes every call, and neither solver iterations nor operation
order changed.

Of those matching inputs, 833,228 (6.362%) are raw positive zero. The current
shim stores exact sine `+0` and returns exact cosine `1` without running
`FSINCOS` in that case; raw negative zero and ordinary nonzero inputs retain the
general paired path. At raw absolute-angle bits `>= 0x5f000000`
(`|angle| >= 2**63`),
the shim falls back to direct `FCOS` then `FSIN` because `FSINCOS` cannot reduce
the operand. Boundary vectors and 100,000 full-raw-bit randomized values pin
this behavior. An unarchived controlled local dense-core A/B improved 1.287%
over the paired-only host. The wide pipeline does not isolate this change, so
its artifact must not be presented as that A/B.

The stable pinned seven-run core A/B moved from 1,246.873 to 1,448.173
decisions/s (+16.14%). In the comparable source-manifested pipeline, the dense
native simulator moves from 1,266.212 to 1,479.193/s (+16.82%), the 48-body
physics workload moves from 64,638.838 to 73,928.145 ticks/s (+14.37%), and
packed/lazy vector rates for 1/4/8 lanes moved from
1,155.738/1,626.229/2,902.754 to 1,323.334/1,901.625/3,294.699/s
(+14.50%/+16.93%/+13.50%). These 30,000-tick physics figures remain the direct
pre/post-pair comparison. Solver work remains dominant.

[`results/exact-pipeline-adaptive-wide-perf-2026-07-21.json`](results/exact-pipeline-adaptive-wide-perf-2026-07-21.json)
is the current checked-in 3,000-decision-per-lane adaptive-wide
source-manifested run. Its 64-lane packed/lazy rate is 48.426% of the 20,000
aggregate-decision/s target, or 2.065x short. Its
SHA-256 is
`4067fdff9360989adb696bdc5ad7d98983729f9fa424271fbd7e3e1fb9164eef`.
It records 38/38 current runtime-source hashes, exact worker SHA-256
`4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261`,
host SHA-256
`ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5`,
and 88/88 true workload-equivalence leaves. It measures 1,447.881 dense native
wall decisions/s (1,485.277/s for step plus observation) and 74,853.849 ticks/s
on the directly comparable 30,000-tick 48-body physics workload.
The artifact continues to report the numerical 20,000-decision/s gate as
false. On 2026-07-21 the project owner explicitly accepted this measured
throughput as sufficient for RL, satisfying the separate readiness alternative
in `clone.md`; the benchmark result itself has not been rewritten as a pass.

[`results/exact-padded-python-hot-path-ab-2026-07-21.json`](results/exact-padded-python-hot-path-ab-2026-07-21.json)
isolates the canonical action-packing and packed-transition suffix changes. Six
interleaved 80,000-decision samples move the 32-lane median from 7,610.288 to
8,066.670 decisions/s (+5.997%) with identical captured body/event
distributions. The attribution artifact SHA-256 is
`0f7a5c6820cd002d190f177f45ba0f0db44c7cf7387c4527d154c8a30299fbbd`;
it is not itself the full performance gate.

[`results/exact-padded-scaling-ceiling-2026-07-21.json`](results/exact-padded-scaling-ceiling-2026-07-21.json)
is a supplemental scaling/scheduling probe, not the full performance gate. On
the 8-core/16-thread host, equal lane/worker rates rise from 4,674.992/s at 8 to
10,975.352/s at 64 without reaching a measured ceiling. At 64 lanes the old
eight-worker behavior reaches 4,606.714/s, so the implemented adaptive default
is 2.382x faster in this probe. Hard affinity is not enabled: pinning two workers
per logical CPU loses 17.5%. The artifact SHA-256 is
`2938d3e072ee99e39ba408f0dd934e5e5caa82993e8e1d7472a6b3322d4f4657`.
This is one sample per configuration, the old eight-worker comparator lacks the
full equivalence digest retained for 16-to-64 workers, and `4 * CPUs` is an
implemented heuristic rather than a proven optimum.

[`results/exact-actor-rollout-ab-2026-07-21.json`](results/exact-actor-rollout-ab-2026-07-21.json)
compares the synchronous per-decision barrier with the experimental
`irisu_env.rollout.ExactActorRolloutPool`. Each worker thread runs its lane's
independent policy for a 64-decision horizon and retains every packed payload;
this is not centralized batched policy inference. Three repeats use 128 warmup
and 1,500 timed steps per lane at 16/32/64 lanes. All 9 pairs match trajectory
digests, final state hashes, and event counts. Median synchronous rates are
4,781.824/5,481.509/5,707.713 decisions/s; actor rates are
5,747.609/5,960.250/6,032.712/s, for median paired speedups of
1.211x/1.097x/1.057x. The focused A/B supplements rather than replaces the
formal gate. Artifact SHA-256:
`2f247f1222f0423475bdcffc185ea893c0b31eb204a7fb6df55030c396d6fc4f`.
Policies run concurrently and must keep mutable/RNG state lane-private or
thread-safe. Policy failures may leave successful sibling lanes advanced;
worker transport/protocol failures require pool recreation.

Reproduce that focused actor comparison with:

```bash
PYTHONPATH=python python3 benchmarks/exact_actor_rollout.py \
  --worker build-exact/irisu-exact-worker \
  --lanes 16,32,64 --warmup 128 --steps 1500 \
  --rollout-horizon 64 --repeats 3 \
  --output benchmarks/results/exact-actor-rollout-local.json
```

[`results/exact-core-trig-cache-investigation-2026-07-21.json`](results/exact-core-trig-cache-investigation-2026-07-21.json)
records why no broader angle memoization was retained. Of 13,096,069 canonical
cosine inputs, 10,080,004 are unique; a 4,096-entry direct cache hits only
9.747% of nonzero inputs and improves a controlled dense median by about 1.04%.
The artifact SHA-256 is
`13f4d6b26d5bde00fd900e2c48317b1336bb5918b9209117e1b279214b8ba9b0`.
Its fresh profile places 58.36% of samples in the immutable exact contact
solver, so the cache's process-global complexity is rejected.

[`results/exact-core-solver-source-optimizations-2026-07-21.json`](results/exact-core-solver-source-optimizations-2026-07-21.json)
rejects two further MSVC9-only candidates below the predeclared 3% integration
threshold. Static-body position-update skipping improves the dense median by
0.648%; a velocity-anchor cache improves it by 0.472%. Both match the full
47,019-step replay, 813,508-record wrapper trace, and trig runtime gate. Neither
was retained, and the full 10,777,297-command getter replay was not spent on
already rejected candidates. Artifact SHA-256:
`6fe2b8c482e8764ff64577261d839d552ae7c4a5a996c538bbead2a135ffcb71`.
The candidate source/object/host inputs are local and unarchived. The report
preserves their hashes and audits but is not a clean-checkout rebuild bundle.

The July 20
[`results/exact-pipeline-paired-trig-2026-07-20.json`](results/exact-pipeline-paired-trig-2026-07-20.json)
is retained as the historical directly comparable post-pair artifact, and
[`results/exact-pipeline-final-2026-07-20.json`](results/exact-pipeline-final-2026-07-20.json)
is its directly comparable pre-trig baseline. Final validation is 14/14 exact
Release CTest targets, 14/14 exact ASAN/UBSAN targets, 8/8 portable Release
targets, 8/8 portable ASAN/UBSAN targets, and 204 passing Python tests with
three expected normal-build skips. Sanitized hostile-preload fixtures put the
worker-linked ELF32 `libasan` first. The GNU layers are instrumented; immutable
MSVC9 host instructions are explicitly verified as uninstrumented.

[`results/exact-pipeline-zero-fastpath-wide-2026-07-20.json`](results/exact-pipeline-zero-fastpath-wide-2026-07-20.json)
is retained only as the intermediate pre-range-guard wide run. Its source
manifest is stale against the current runtime and benchmark harness; none of
its artifact identities or rates are current claims.

The exact packed contract has stronger validation than timing alone. A
randomized ordinary `Step`/`StepPadded` comparison matched 1,906 steps and all
190,501 events across five boundary seeds. A separate 50-episode stress run
completed 58,534 decisions and 11,843,733 events, replacing the exact worker
on every reset to contain pristine-r58 process-global allocator state. The
worker now rejects a second reset, and the benchmark's raw client path likewise
respawns and identity-checks a worker for every later episode. Direct in-process
exact C ABI/C++ execution remains one-episode diagnostic-only. An older
sparse wait-only spot check reached higher headline rates; it is not
representative of a dense training board and must not be used for the gate.
The paired artifact's corresponding historical sparse eight-lane
`ThreadVectorEnv` rate is
11,632.97/s, illustrating the density effect rather than overriding it.

Use the exact pipeline profiler to separate physics, full-event worker/pipe
traffic, Python decoding, dictionary materialization, `IrisuEnv`, and vector
orchestration costs:

Configure the 32-bit exact build as documented in
[`../tools/exact-physics-prototype/ipc.md`](../tools/exact-physics-prototype/ipc.md),
with `IRISU_BUILD_BENCHMARKS=ON`, then run:

```bash
cmake --build build-exact --target irisu-exact-worker irisu-physics-benchmark
PYTHONPATH=python python3 benchmarks/exact_pipeline.py \
  --worker build-exact/irisu-exact-worker \
  --physics-benchmark build-exact/irisu-physics-benchmark \
  --vector-lanes 1,4,8,16,32,48,64 \
  --output benchmarks/results/exact-pipeline-adaptive-wide-local.json
```

The command above explicitly fixes exact `PaddedVectorEnv` at each requested
width, making the benchmark independent of host topology. The profiler's
default lane list remains `1,4,8`; this is a benchmark duration choice, not the
public exact worker default.

The profiler records artifact and source SHA-256 identities. It includes both a
sparse `wait(1)` diagnostic and a representative seeded
`RandomPolicy(max_wait_ticks=1)` workload. The latter runs through eager and
packed raw send/drain, fully decoded `ThreadVectorEnv`, and packed/lazy
`PaddedVectorEnv` with the same actions, resets, body counts, and event counts.
The report records those equivalence checks plus response-size and body/event-
density distributions. It also compares a reusable Linux fast checkpoint with
durable action-log restore. At 1,000 history actions, local measurements put
checkpoint creation at 171.215 us, median branch creation at 283.931 us
(3,521.982/s), and median durable restore at 95.479 ms (10.474/s), a 336.274x
median advantage. The durable snapshot is 28,104 bytes. This matters because
eager exact transition payloads can contain
hundreds of contact
events and tens of kilobytes even though a sparse wait benchmark looks fast. A
raw physics rate or sparse worker rate must not be presented as the
20,000-decision training gate.

[`results/provisional-training-readiness-2026-07-19.json`](results/provisional-training-readiness-2026-07-19.json)
is retained as the historical pre-PC53 source-manifested Release run. It used
the former `nearest,x87-extended` floating-point environment. The recorded
plain-Python observation mode had no installed Gymnasium or NumPy conversion
path. The typed path sustained 4,536.442
aggregate decisions/s and 11,189.135 simulation ticks/s while observing
11/41.005/124 minimum/mean/maximum bodies. Sequential, threaded, and typed
paths used the same recorded trace, finished with the same per-lane state
hashes, and produced the same 8,036-record public-transition digest
`cfa60023927bbf31830a9c064f1e6f454a9c0f1c81493aeb6319fbaa5d8b6744`.
A second full run reproduced the entire deterministic payload byte-for-byte
after excluding timestamps and timing/rate fields (sorted compact JSON,
UTF-8, no trailing newline; payload SHA-256
`8eb115bce3f4a10fb662f2ae1cd8d92034a6766bd6ac2164144c1d8fc2fb8de9`).
The 91-file source manifest SHA-256 is
`c493312f9703fe08296e3493a32a18fc2818c03624e17a315d145dd33e8f5c8e`;
the artifact SHA-256 is
`03db6fec657123ed99be35fa886414363103c4e745da7794a29c28055f19f15f`,
and the measured shared-library SHA-256 is
`b263c06f54b3bbebca2213326a14b927152eddf0a8713ff2b4c6b986b3dba208`.
Its failed readiness gates remain part of the historical record.

[`results/provisional-training-readiness-2026-07-18.json`](results/provisional-training-readiness-2026-07-18.json)
is retained as a historical pre-corrected-reset/x87 run. Its 21,002.997
decisions/s result and old configuration/source hashes are not a current
performance claim.

[`results/provisional-short-2026-07-17.json`](results/provisional-short-2026-07-17.json)
is retained as the pre-typed-path baseline. It does not contain the corrected
version-2 native excitation workload or padded-vector result and must not be
used as the current performance gate.

Metrics are defined as:

- `physics_ticks_per_second`: native `PhysicsWorld::step` calls on a fixed
  mixed-shape 48-body board, with no Python, JSON, policy, or game-rule work in
  the timed loop; deterministic periodic excitations prevent a sleeping-board
  microbenchmark, and their trace plus initial/final state hashes are recorded;
- `decision_steps_per_second`: complete Python policy/action, native step, JSON
  observation/event decode calls per second for one environment;
- `aggregate_decision_steps_per_second`: `num_envs × vector calls` per second;
- `padded_vector.aggregate_decision_steps_per_second`: complete seeded policy,
  action tracing, native batch step, typed observation, reward/termination,
  diagnostic header, and exact lazy event-count exposure per second;
- `simulation_ticks_per_second`: native ticks advanced, including multi-tick
  waits;
- snapshot clone/restore operations per second and serialized snapshot size.

The report evaluates a 20,000 aggregate-decision/s performance threshold. A
failed result is recorded as failed, not silently accepted. It records observed
body-count minimum, mean, and maximum so a sparse-board result cannot masquerade
as typical training throughput. Native physics-only performance remains a
separate metric. Passing this engineering gate does not open bulk training while
the fidelity/transfer gate remains open. The tracked five-category controlled
manifest is empty and `not_evaluable`; there is no hashed original-game
spawn/difficulty distribution comparison and no demonstrated qualitative policy
transfer. Four exact replay oracles therefore do not by themselves authorize
bulk RL.

```python
from irisu_env import Action, PaddedVectorEnv

with PaddedVectorEnv(
    8,
    physics_backend="exact",
    worker_path="build-exact/irisu-exact-worker",
) as envs:
    observations, reset_info = envs.reset(seed=100)
    observations, rewards, terminated, truncated, info = envs.step(
        [Action.wait()] * 8
    )
    live_ids = [
        observations[0].bodies[index].id
        for index in range(observations[0].body_count)
    ]
```

## Baselines

`RandomPolicy` uses a repository-defined SplitMix64 stream, so its action trace
does not depend on Python's `random` implementation. `MatcherShotPolicy` is a
nontrivial causal body-aware heuristic: it prioritizes the lower member of the
closest same-color pair and avoids redundant shots already travelling in that
column.

```python
from irisu_env import IrisuEnv, MatcherShotPolicy

with IrisuEnv() as env:
    observation, _ = env.reset(seed=17)
    policy = MatcherShotPolicy()
    policy.reset(77)
    for _ in range(500):
        observation, reward, terminated, truncated, info = env.step(
            policy.act(observation)
        )
        if terminated or truncated:
            break
```

Neither baseline receives RNG state, future spawns, or hidden schedules. Their
purpose is determinism, integration testing, qualitative transfer probes, and a
floor for later learned policies—not a superhuman-performance claim.
