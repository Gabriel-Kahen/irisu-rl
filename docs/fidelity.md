# Fidelity and readiness report

## Current status

The repository contains an end-to-end, asset-free headless implementation of
IriSu Syndrome v2.03 normal mode. The rule layer is no longer the earlier
placeholder vertical slice: its RNG, update cadence, input edges, spawn order,
seeded 20-block reset prefill, level formulas, native-order contact dispatcher,
grouping, burst scoring, special/direct-hit behavior, gauge ordering, lifetimes,
actor allocation, and terminal timing were recovered from the executable and
implemented directly. The nominal configuration now uses the executable's
mode-0 normal table, not the nonzero-mode/Metsu-side table mirrored by the
shipped INI.

The rule/scoring layer is now exact on every observed v2.03 replay event. With
the separately generated 32-bit exact-MSVC physics host, the 41,449-point run
matches all 455 score calls and terminates at the original tick 47,019; a
second long replay matches all 66 score calls and terminates at the original
v2.03 outcome of 1,794 points at tick 8,368. Two short external traces have
also been replayed in the original: one is exact outright, while the other is
exact after reproducing the loader's documented suppression of fresh click
edges on replay frames zero and one. The four-replay exact gate, including the
complete 47,019-step physics stream, remains passing.

The clone is suitable for deterministic research and replay diagnostics. The
portable GNU physics build used by the default C/Python API is still not
long-horizon trajectory-identical; on the 41,449 trace it dies at tick 8,375
with score 1,659. The exact-MSVC host is now available to the Python API through
an opt-in isolated 32-bit worker per active episode.
Exact `PaddedVectorEnv`/`FastVectorEnv` and Linux fork/COW checkpoint branches
are now part of the production Python surface. Policy transfer remains
unproven.

## North-star definition-of-done status

The clone phase is not complete under [`clone.md`](../clone.md). The current
evidence for each required outcome is:

| # | Status | Current evidence or blocker |
|---:|---|---|
| 1 | Met | A native end-to-end regression runs the nominal no-input episode from its seeded 20-block reset state through game over. The CLI and Python API use the same core. |
| 2 | Met for the supported clone build | Same-seed action traces and raw replay-level inputs are deterministic. Portable and exact Sync/Thread/Padded/Fast vectors have isolation/equivalence regressions, including an enforced exact-worker concurrency cap. Randomized ordinary `Step` versus packed `StepPadded` testing matched 1,906 steps and all 190,501 events. A scoped canonical floating-point environment isolates execution from hostile caller rounding state. Cross-build bitwise identity is not claimed. |
| 3 | Met as an engineering gate | Portable object/wire snapshots branch exactly through randomized futures, dense and swept contacts, deferred contact replacement, out-of-range/frozen proxies, allocator churn, and terminal states. Exact durable snapshots reproduce hidden futures by reset/action replay and reject corruption or incompatible configuration atomically. Production Linux fork/COW checkpoints create reusable independent branches in approximately constant time; source reset/restore refuses while a checkpoint is live, while branch reset/restore safely detaches it. Durable restore remains O(history). |
| 4 | Met | Native rule tests cover scoring, gauge, lifecycle, spawning, actor allocation, contacts, difficulty progression, and terminal ordering. |
| 5 | Not met as the formal golden gate | Four external padded replays now have instrumented original v2.03 playbacks. The exact-forward path matches all 536 observed score calls across the four traces (521 in the two long traces). The scorer now accepts the exact worker with `--worker` and records its executed worker plus live mapped host, but the strict manifest remains empty because these replay bundles do not supply the complete five-category controlled-scenario schema; the result is `not_evaluable`, and policy transfer is still unevaluated. |
| 6 | Met | The score formula was exonerated. Causal fixes covered mode selection, exact level descent stores, reentrant level-transition field updates, replay startup edge suppression, and the original's stale actor-pool gauge rewards. No replay-specific score tuning was introduced. |
| 7 | Not met | The current wide source-manifested profile measures 7,199.041 decisions/s for 32 explicit exact packed/lazy-event lanes. That is 35.995% of the 20,000 decision/s target, or 2.778x short. The default and portable concurrency cap remains eight; more than eight exact workers must be requested explicitly. |
| 8 | Met | The simulator and normal test suite are asset-free; the original executable, DLL, replays, and presentation data are optional ignored reference inputs. The raw fast-branch replay test skips cleanly when its optional oracle is absent, while its lifecycle/security coverage still runs. |
| 9 | Not met | `MatcherShotPolicy` is a nontrivial causal baseline in the clone, but it has not been run successfully in controlled original-game trials. |
| 10 | Partially met | Repository instructions reproduce the clean build, tests, bounded oracles, exact-host generator, replay evaluators, and opt-in exact Python worker. Original-playback evidence is preserved with hashes, but the authorized original-game path still lacks a complete admissible scenario bundle. |

Outcomes 5, 7, and 9 remain release blockers for bulk training. Outcome 10
remains partial because the authorized original-game path has not yet produced
a complete admissible scenario bundle.

## Evidence-backed implementation

| Area | Evidence and result |
|---|---|
| Engine | Official zlib Box2D 1.4.3 SVN r58 source, matched to the shipped DLL ABI and numerical probes |
| Fixed update | One replay record is one 0.020-second gameplay update; fast-forward skips rendering only |
| RNG | Exact DxLib 624-word generator, seed expansion, and inclusive range mapping; measured vectors pass |
| Field | Literal normal world bounds, magnification 10, gravity 160, and the four mode-0 fixtures/materials selected by replay mode |
| Input | Button levels with previous-level edge detection; left then right, followed by cadence spawn |
| Spawn | Exact setup/prefill draw order, 10 rotten plus 10 scripted reset blocks, mode-0 99-ticket seven-size distribution, position/rotation/color/shape, special schedule, and 200-slot allocator |
| Rules | Exact contact-list traversal and handler order/predicates, including persistent contacts and monotonic `e0` |
| Score/gauge | Exact group formula and event ordering; 521/521 long-replay score calls and 100/100 long-replay rot events match; 700-unit normal rewards; special clears reward matching stale dead actor slots; mode-0 gauge 40,000/3,000; level-dependent rot/drain; ten-clear progression and level-100 branch |
| Lifecycle | Scripted/native ownership split, captured spawn speed, c8 postlude delay, lifetimes, strict OOB, and deferred native destruction |
| API | C ABI; Python/Gymnasium-shaped environment; portable and exact Sync/Thread/Padded/Fast vectors; durable and Linux-local fork/COW snapshots; replay evaluator; SVG diagnostics; and policy/diagnostic separation |

The reproducible physics oracle is
`reference/probe/golden/box2d-v2.03-wine11.13.jsonl`, SHA-256
`b73f9c74db48a9695450ab04b76661c0d576030f7f428d0e53b953b4dae077b2`.
It contains 1,290 records and is reproduced byte-for-byte by two consecutive
runs of the shipped DLL probe on the recorded Wine 11.13 host. Native clone
regressions cover its discriminating velocity units, integration, dimensions,
contact timing/order, friction/restitution, sleeping, and triangle geometry.
The clean-room DxLib model also matches every local-DLL result for four checked
seeds and the maxima used by the first normal spawn path. These bounded oracles
support the implementation; they do not prove indefinite game trajectories.
The exact rule derivation is documented in
[`reference/game-rules-analysis.md`](../reference/game-rules-analysis.md).

The supported optimized GNU build compiles the legacy engine with a scoped
floating-point environment. It is deterministic within the supported profile,
but the full getter oracle disproves numerical identity: tiny first-step
differences compound into a different long-run board. Public simulator,
physics, and C API entries install round-to-nearest, x87 control word `0x027f`
(PC53), and MXCSR `0x1f80` within a nested thread-local scope, then restore the
caller environment even on exceptions. Tests exercise hostile downward
rounding in scalar execution and an eight-lane padded worker batch.

A separate clean-room host converts user-built MSVC9 r58 objects into an ELF32
library. Its handle-based adapter reproduces the original physics and
rule/event stream on the adjudicated replay corpus and supports independent
worlds. The production Python exact backend keeps one isolated 32-bit worker
per active episode: this isolates failures and lets vector lanes run in
parallel despite the host bridge serializing pristine-r58 calls inside one
process. Production reset launches, configures, and identity-checks a fresh
worker before atomically replacing the old one. This contains a deterministic
pristine-r58 process-global allocator crash exposed by repeated in-process
world teardown. A 50-episode regression completed 58,534 decisions and
11,843,733 events with a fresh worker PID on every reset. The low-level worker
now permits one successful reset and rejects post-reset configuration; the
Python owners transparently replace it for each later episode. Direct in-
process exact C ABI/C++ use does not have that isolation and remains a one-
episode diagnostic, not a trainable multi-episode backend.

Identity checking is tied to the running processes rather than only configured
paths. A normal worker's `/proc/<pid>/exe` is hashed only after a valid Hello
proves `exec` completed. The client locates the one exact library in the live
worker maps and verifies its resolved path, device, inode, required ELF
segments, worker and client mount identities, stable file metadata, and bytes
against the handshake hash. Before constructing a simulator, the worker
attests that all 15 resolved `b2d_*` targets are unique executable addresses
owned by that one host; Python requires the opcode-13 device/inode result to
match its independent map capture. Fork/COW branch connection authenticates the
keeper, direct parentage, and inherited worker/library file identities before
accepting the inherited launch provenance; an explicit provenance call
rehashes the branch's live mapped library.

Durable exact snapshots store the seed and accepted action history and
atomically restore into a fresh configured worker by replaying that history.
They preserve hidden solver state exactly, but restore in time linear in the
episode age. A durable clone refuses while a split exact step is pending, so an
advance already accepted by the worker cannot be omitted from the serialized
action history. Production Linux exact workers additionally expose reusable
fork/COW checkpoints through `ExactSimulator.fast_checkpoint()` and
`IrisuEnv.fast_checkpoint()`. A source refuses reset/restore while it owns a
live checkpoint; resetting/restoring a branch detaches it from the keeper. At a
1,000-action history, local measurements put checkpoint creation at 0.188 ms,
median branch creation at 0.480 ms, and durable restore at 95.933 ms, a 200.0x
median advantage for local branching.

The native `Simulator` validates its mechanics configuration at construction,
after which that configuration is immutable. Its configuration hash is now
computed once and cached for transitions, diagnostics, snapshot identity, and
compatibility checks. This removes repeated configuration serialization from
the step path without changing any hash value or accepted snapshot.

The exact padded step returns only the packed observation, diagnostics, event
count, and event generation. It transfers event records later through
`FetchEvents` only if `info["events"]` is materialized. `len(events)` remains
count-only; materialized fetches are capped at 4 MiB, and an oversized fetch
returns a bounded error without poisoning the lane. Each capped exact wave is
sent first and then drained in descriptor-readiness order, avoiding
head-of-line blocking by a slow low-numbered lane while preserving full-drain
and deterministic lowest-lane failure behavior. On the dense eight-lane
workload this cuts average/max response content from 23,558/54,559 bytes to
11,291/19,804 bytes. The current wide run measures
1,334.810/1,933.338/3,292.397/5,391.293/7,199.041 decisions/s at
1/4/8/16/32 exact packed/lazy lanes; raw packed IPC reaches 7,995.455/s at 32
lanes. Wider-than-eight concurrency is opt-in with, for example,
`PaddedVectorEnv(32, physics_backend="exact", workers=32)`. Omitting `workers=`
and every portable configuration remain capped at eight concurrent workers. A
combined body-state getter experiment was rejected: its
representative and dense deltas were only +0.009% and +0.187%, within noise,
and the final host does not export it.

Ten-run sampling of the pre-optimization dense core collected 24,020 samples at
1,267.12 ± 4.43 decisions/s. The exact MSVC host accounted for 92.678% of
samples: `SolveVelocityConstraints` 28.426%, `SolvePositionConstraints` 23.393%,
`FCOS` 14.205%, and `FSIN` 11.678%. A 3,000-decision instrumented run then found
13,096,069 same-raw-float cosine/sine pairs. The retained runtime shim computes
each pair with one x87 `FSINCOS`, caches the float-rounded sine under the raw
input bits, and falls back to standalone `FSIN` when a sine does not match. Of
those inputs, 833,228 (6.362%) are raw positive zero. The retained fast path
stores exact sine `+0` and returns exact cosine `1` without `FSINCOS` for that
case; raw negative zero and ordinary nonzero values retain the general paired
path. Because x87 `FSINCOS` reports an unreduced operand at `|angle| >= 2**63`,
the shim falls back to direct `FCOS` then `FSIN` for raw absolute-angle bits
`>= 0x5f000000`. Boundary cases and 100,000 full-raw-bit randomized inputs are
checked against the independent instructions. An unarchived controlled local
dense-core A/B improves 1.287% over the paired-only host; the wide pipeline does
not isolate that change.

The public bridge remains serialized, and no solver iteration count or
operation ordering changed. The earlier controlled pinned seven-run comparison
moved from 1,246.873 to 1,448.173 decisions/s (+16.14%) when pairing trig.
Solver work remains the dominant optimization target after both trig reductions.

The current wide exact pipeline artifact records 37/37 current source hashes,
exact host SHA-256
`bf46953217a7bcd49f382d44cb05dd58db373fb9f86dc1e42eb531c12c71908a`,
worker SHA-256
`aa7ba4a6998b6dfeb59d1ea80cd1690cd0e7b727cf9968c38f362e60835e6d57`,
and 64/64 true cross-path equivalence leaves. Its SHA-256 is
`91c8db5feb9d3c8339d101940f05a42d93a4490641745964a0ca427553b8b8e9`;
see
[`exact-pipeline-range-safe-wide-2026-07-20.json`](../benchmarks/results/exact-pipeline-range-safe-wide-2026-07-20.json).
It measures 1,498.136 dense native decisions/s and 75,819.177 ticks/s on the
directly comparable 30,000-tick 48-body physics workload. The prior
[`exact-pipeline-paired-trig-2026-07-20.json`](../benchmarks/results/exact-pipeline-paired-trig-2026-07-20.json)
is retained as the comparable post-trig artifact, while
[`exact-pipeline-final-2026-07-20.json`](../benchmarks/results/exact-pipeline-final-2026-07-20.json)
is its pre-trig baseline.

Final validation passes 10/10 exact CTest targets, 8/8 portable Release CTest
targets, 8/8 portable ASAN/UBSAN targets, and 159 Python tests with two optional
Gym skips. The exact corpus remains 4/4, including the full 47,019-step replay
stream.

The corrected seed-123 reset regression pins the 20-block constructor state:
10 rotten blocks at Y=200, 10 scripted blocks beginning at Y=60, RNG index 102,
next body ID 21, actor cursor 24, level-shape cutoff 63, and special threshold
51. Its first wait update consumes five more draws for cadence body 21. These
are clone/DLL-derived diagnostics, not an original-game outcome scenario.

## Determinism and snapshots

Automated coverage includes exact RNG/rule vectors, same-seed traces, differing
seeds, malformed/wrong-profile rejection, previous-button edge restoration,
actor-slot wrap/capacity cases, dense contacts, sleeping bodies, zero-manifold
contacts, pending native tombstones, and future equivalence after restore.

Schema-7 snapshots preserve the full causal state rather than merely visible
bodies: Dx RNG words/index, group counters, raw Block state, actor slots and
cursor, stale colors for all 200 actor-pool slots, terminal metadata,
native body/proxy/free/destroy order, all broad-phase
bounds and proxy timestamps, contact-list and body-node order, manifolds, and
warm-start impulses. Both the native origin and exact center-of-mass bits are
stored because r58's float transform is not bit-invertible for asymmetric
triangles. Dense contacts, swept zero-manifold contacts, divergent actor/native
state, asymmetric centers of mass, deferred contact replacement, swept proxies
whose current shape is outside creation range, frozen proxies with deferred
contacts, and repeated deferred-delete/proxy-pool churn have exact
future-equivalence property tests. Structural and causal snapshot validators
also reject malformed object and wire inputs atomically. One pinned wire
regression changes a valid zero-manifold identity from bodies `(29,31)` to
`(29,30)` while retaining incompatible saved broad-phase bounds; restore now
rejects it before Box2D commits the fabricated pair. This is a completed
engineering gate with finite test coverage, not a proof against every possible
state or corrupted byte stream.

## Observation and training boundary

Policy observations include visible scalar/body state and the previous left and
right button levels because those levels determine legal fresh edges. The C++,
C JSON, and Python surfaces all use the same filtered body contract; native
transforms, broad-phase/contact caches, allocator fields, pending rule flags,
RNG, configuration identity, finish/replay metadata, and the diagnostic state
hash are intentionally hidden. Python reset and step `info` carry the
configuration hash; step `info["diagnostics"]` carries finish count and
first/latest terminal metadata. `env.state_hash()` remains available for
reproducibility checks and can be mirrored into `info` only when
`diagnostic_hashes=True`.

Default mechanics randomization is now a set of singleton recovered values;
the environment never silently perturbs nominal mechanics. Researchers may
supply explicit non-nominal ranges for robustness work, and those configurations
receive distinct hashes and incompatible snapshots.

`TransferRobustnessEnv` covers the observation/action side of the transfer
boundary without altering the native mechanics profile. Caller-supplied,
seeded bounds support shot latency/timing jitter, cursor error, observation
latency, position/velocity noise, dropped detections, and deterministic
distance-based merges. It transforms only the public policy observation and
does not attach the current clean observation or native hidden state to
`info`. Rewards and termination remain those of the current native state. Its
terminal observations flush the delay for a coherent final transition. Its
singleton defaults are an exact nominal pass-through; these ranges remain
research assumptions until controlled original-game measurements bound them.

Legacy API fields listed as `compatibility_only_ignored` in configuration JSON
remain accepted and hashed but never affect the faithful rule path. They are
not calibration knobs or unresolved original mechanics.

## Replay-corpus diagnostics

Replay headers are not outcome oracles. Every padded external mode-0 file was
therefore replayed in a fresh bundled-v2.03 process under the same `0x027f`
floating-point environment used by the exact-forward runner. The observed
results are:

| File header | Observed bundled-v2.03 outcome | Exact-forward result |
|---:|---:|---:|
| 40 / L1 / C2 | 32 / L1 / C2, 2 clears, tick 1,020 | exact, including 4 score calls |
| 56 / L1 / C2 | 88 / L1 / C2, 6 clears, replay exhaustion tick 1,514 | exact score/event timeline after the replay-start edge rule |
| 41,449 / L38 / C5 | 41,449 / L38 / C5, 379 clears, tick 47,019 | exact, including 455 score and 80 rot calls |
| 43,791 / L32 / C7 | 1,794 / L5 / C6, 44 clears, tick 8,368 | exact, including 66 score and 20 rot calls |

The 40, 56, and 43,791 headers do not describe what the supported executable
does with those bytes; their generating build/runtime is unknown. They remain
useful action traces, while their observed v2.03 playbacks are the comparison
targets. The 214,453-point offset-20 file predates v2.00 and remains excluded.

The final short-trace discrepancy was in replay adaptation rather than game
scoring. `Input.update` loads and retains raw held levels normally, but clears
fresh left/right edge bytes on replay records zero and one. The 56 file begins
with left held for exactly those records. Treating record zero as a normal edge
created an extra weak projectile before the first physics step. Reproducing the
two-record suppression makes the complete score timeline (`0 -> 88`), gauge
events, creates, contacts, destroys, and transforms exact. The original replay
wrapper additionally forces scene finish when records are exhausted; that
loader-specific finish is not a normal RL episode rule.

The controlled seed-41 adjudication remains a smaller independent regression:
original and clone both score `+8,+8` at tick 304 and finish the 520-record
nonterminal trace at 16. None of these results changes the score formula; the
large former discrepancy came from upstream physics/setup, reentrant level
updates, special-clear gauge survival, and replay edge handling.

### Historical pre-mode-table results

The numerical clone results below were recorded before recovery of the
constructor-time prefill and correct mode-0 table. They are retained only as
historical diagnostics. Their hashes, terminal ticks, and outcomes are not
current-source claims.

On 2026-07-18, all six preserved normal-mode traces were evaluated twice with
a fresh GNU 16.1.1 Release build (`clone 0.1.0`, C ABI 1, padded ABI 1,
snapshot schema 6, Box2D 1.4.3 SVN r58). The two passes produced byte-identical
JSON reports. Auto layout detection resolved the five zero-padded files at byte
52 and the January 2009 file at byte 20; this detection remains heuristic for
an otherwise-unknown legacy file whose first eight input words are all zero.

| Trace | Bytes / frames | Layout | Header: seed / level / score / chain / mode | Version boundary |
|---|---:|---:|---:|---|
| `new-2026-07-17.rpy` | 2,704 / 663 | padded, offset 52 | 586667 / 1 / 0 / 0 / 0 | Self-generated v2.03 trace; short and nonterminal. |
| `irisu_00000040_20190417_184328_1.rpy` | 8,352 / 2,075 | padded, offset 52 | 26103349 / 1 / 40 / 2 / 0 | Padded v2-era trace; generating build unverified; attempted v2.03 playback was blocked. |
| `irisu_00000056_20190417_184425_2.rpy` | 6,108 / 1,514 | padded, offset 52 | 26172090 / 1 / 56 / 2 / 0 | Padded v2-era trace; generating build unverified; attempted v2.03 playback was blocked. |
| `irisu_00041449_20100725_182435_7.rpy` | 188,128 / 47,019 | padded, offset 52 | 168175029 / 38 / 41449 / 5 / 0 | Post-v2.03-release padded trace; exact generating build unverified. |
| `irisu_00043791_20111118_222006_26.rpy` | 142,828 / 35,694 | padded, offset 52 | 387338 / 32 / 43791 / 7 / 0 | Post-v2.03-release padded trace; exact generating build unverified. |
| `irisu_00214453_20090104_005708_5.rpy` | 172,608 / 43,147 | legacy, offset 20 | 4765293 / 34 / 214453 / 42 / 0 | Pre-2.00, probably v1.02; exact build unknown and not v2.03 mechanics. |

The obsolete run preserved all encoded cursor coordinates and had no invalid
clone actions or unrepresented shot edges, but its outcome table has been
removed because it used the wrong mode table and is not current mode-0
evidence. A later but still retired portable-physics measurement scored 1,563,
reached level 5/chain 6, and terminated at tick 8,228. Replay headers remain
diagnostic metadata; the oldest
file uses a different mechanics line, generating binaries are not proven for
the other external files, and legacy Box2D trajectories may diverge by machine.
In particular, the v2.03 loader would consume the first eight input words of
the legacy file as padding; this diagnostic intentionally parses that file's
actual legacy input stream from byte 20 instead of simulating broken
cross-version loading.

### Historical alternate-profile DLL prefix — inadmissible for mode 0

The July 18 PE32 driver is not evidence for the 41,449-point mode-0 replay. Its
source hardcodes the nonzero-mode/INI geometry—walls centered at X 102 and 530,
bottom centered at Y 620—and 48-unit initial pieces. The replay header's mode
field is zero, so the executable instead selects the release table at
`0x412560`. The reported frame-310 contact and subpixel trajectory agreement
only compare the DLL and clone under that alternate table; they cannot validate
the current mode-0 prefix or rule out a release-profile mapping/setup error.

The ignored experiment bundle
`reference/runs/replay-prefix-dll-20260718/` contains the exact source, inputs,
commands, hashes, and byte-repeatable 1,361-record trace (SHA-256
`5c0800a5555a598dbce4f405cc270669abefb5a3ca936548dc9f750de96f1850`).
The bundle remains useful as a byte-repeatable alternate-profile component
probe, but it is inadmissible as replay-fidelity evidence for mode 0.

### 2026-07-19 corrected-reset diagnostics

An active compositor output and exact IriSu window capture worked. A fresh
seed-123 one-record replay and images under
`reference/captures/probe-reset1-20260719-001/` visually corroborate the
corrected 10-rotten/10-scripted initial layout. A separate seed-41, 520-record
probe under `reference/captures/probe-b-match-chain-20260719-001/` ended with
score 16 in the original HUD while the then-current x87 clone ended with score
0.
The [seed-41 adjudication](../reference/seed-41-adjudication.md) pins its four
strong-shot records and shows that divergence is already unambiguous in the
first projectile around ticks 9–13. It also records that clone and shipped-DLL
prefixes agree on the actor-16 first-step contacts, both x87 precision settings
produce the same DLL trace, and an actor-16-only exception still scores 0.
Because the original capture reused a process after earlier scenes and lacks
the full metadata, action, measurement, and status records required by the
golden schema, it was then treated as an unresolved high-impact diagnostic
rather than a mechanics fix or golden result.

That paragraph describes the July 19 state. The July 20 fresh-process proxy and
score breakpoint resolved it: mode 0 selects the production table at
`0x412560`, and the corrected clone now matches `+8,+8` at tick 304 and final
16. The score formula was unchanged.

Later on July 19, exact capture of a freshly background-launched window worked,
but the window did not acquire targeted DirectInput without focus. No global
focus or physical-input fallback was used. July 20 avoided that historical
blocker by forcing exact replay selection under debugger control in a fresh
process while tracing the authentic DLL.

The retired historical `orb-seed93-edge` preset remains clone-only evidence and
is not a golden. The active reproducible observed-score input is:

```bash
python3 tools/generate-controlled-rpy.py /tmp/score-seed41-parity.rpy \
  --preset score-seed41-parity --library build/libirisu_clone.so
```

It regenerates the byte-identical 520-record seed-41 replay (SHA-256
`1ce501febe8f3f6291e4b82736542179bd9808e412d38e0e1fb1c92d05797657`)
and requires clone score events `+8,+8` at tick 304 and final score 16.

## Remaining claim boundary

Safe claim: this is a tested clean-room headless implementation of the recovered
v2.03 normal-mode mechanics and legacy physics surface.

Still not claimed:

- indefinite cross-machine trajectory identity for chaotic legacy contacts;
- long-horizon trajectory fidelity from the default portable GNU physics
  backend (the opt-in Python exact-MSVC worker is required for replay parity);
- portable constant-time exact snapshots: durable restore still replays a
  complete seed/action history in time linear in episode age; Linux-local
  fork/COW branches provide the fast non-serializable path;
- fidelity for Metsu, EX-specific paths, menus, story, saves, or endings;
- the 95% controlled discrete-probe fidelity threshold or qualitative
  scripted-policy transfer before controlled original-game evaluation;
- correctness of deprecated compatibility knobs as original mechanics.

Long-horizon replay disagreement in the portable backend remains useful
diagnostic evidence, but the exact-forward comparisons show that it is an
engine-integration issue rather than uncertainty in score arithmetic.

The July 18/19 display attempts remain historical. July 20 produced four
fresh-process external replay oracles plus the controlled seed-41 trace, but
the bundles still lack the complete formal golden artifact set and
five-category controlled coverage. The golden manifest therefore remains
empty. No scripted policy has yet demonstrated qualitative transfer.
