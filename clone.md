# Headless IriSu Puzzle Clone Specification

## Goal

Create a fast, deterministic, testable, clean-room clone of the **normal puzzle mode** of IriSu Syndrome. Its purpose is to train and search for policies that transfer to the original game.

The clone is not a visual remake and does not include story progression, original art, music, menus, endings, file-changing behavior, or Metsu mode. It is a physics-and-rules simulator plus optional diagnostic rendering.

The clone is successful when policies learned in it retain their skills in the original executable. Exact long-duration pixel synchronization is neither expected nor required, but legal actions, short-horizon physics, discrete game events, scoring, gauge behavior, spawning, and difficulty progression must be sufficiently faithful for transfer.

## Implemented evidence update (v2.03)

The original research plan below is retained as project history, but its early
“unknown” and placeholder statements are superseded by the current clean-room
analysis in [`reference/game-rules-analysis.md`](reference/game-rules-analysis.md)
and the implementation contract in [`docs/mechanics.md`](docs/mechanics.md).
The headless clone now uses the exact DxLib RNG, legacy Box2D r58 source,
magnification 10, gravity 160, the four recovered fixtures, 0.020-second replay
cadence, button-level edge semantics, the native-order contact dispatcher,
group/burst/special/direct-hit rules, actor-slot allocation, gauge/score/level
formulas, the executable's mode-0 production table, and causal schema-7
portable snapshots. Placeholder compatibility knobs are not nominal gameplay
formulas.
A fresh-process seed-41 playback now gives exact original/clone score parity:
two `+8` events at tick 304 and final score 16. The score formula was unchanged;
the prior zero came from using the separate nonzero-mode/Metsu-side INI table.
An opt-in 32-bit MSVC9-r58 backend now removes the measured long-horizon
numerical drift: it matches every score/rot event and terminal outcome in all
four observed v2.03 replay oracles, including the full 47,019-step trace. It is
available through `IrisuEnv`, independent worker-backed vector environments,
configuration overrides, exact packed/lazy-event padded vectors, durable
seed/action-log snapshots, and Linux fork/COW checkpoint branches. Production
resets use a fresh worker process to contain a legacy allocator failure found
under repeated in-process world teardown. A 50-episode stress run completed
58,534 decisions and 11,843,733 events, and randomized ordinary-versus-padded
testing matched 1,906 steps and 190,501 events. The five-category controlled
golden gate and qualitative policy transfer remain unmet; the main remaining
training risks are dense exact-backend throughput, formal controlled coverage,
external replay-build provenance, and out-of-scope modes.

The formal scorer can now execute the same exact backend with `--worker` (or
the portable backend with the mutually exclusive `--library`) and records both
the executed worker and live mapped host. Its tracked five-category manifest is
still empty, however, so the controlled gate remains `not_evaluable` rather
than passing by absence of evidence.

Exact worker identity is fail-closed. A valid Hello precedes hashing the
executed `/proc/<pid>/exe`, which avoids observing the parent during
`posix_spawn`; the client then verifies the live mapped host's path, device,
inode, segments, worker/client mount identities, and bytes against the worker's
handshake hash. Before simulator construction, the worker verifies that all 15
resolved `b2d_*` call targets are unique executable addresses owned by that
same host, and Python requires its opcode-13 device/inode attestation to match
the independently captured mapping. Fork/COW branches authenticate their keeper and direct ancestry
before inheriting that launch-verified provenance; an explicit provenance call
rehashes the branch's mapped library.

Durable `clone_state()` refuses while an exact split-step request is in flight.
This prevents a worker-accepted state advance from being serialized before its
action is committed to the durable action history.

The worker-backed `ExactSimulator`, `IrisuEnv`, and padded/vector surfaces are
the supported multi-episode training topology. The protocol permits only one
successful reset per OS process and production resets transparently replace the
worker. Direct in-process exact C ABI/C++ use still shares pristine-r58 process-
global state and is therefore a one-episode diagnostic path, not a trainable
multi-episode backend.

The packed exact path removes event records from the normal step response and
fetches them lazily on demand. It sends each capped worker wave before polling
and draining response-ready lanes, while preserving full-drain and deterministic
lowest-lane failure semantics. The current wide run measures
1,334.810/1,933.338/3,292.397/5,391.293/7,199.041 decisions/s for explicit
1/4/8/16/32-lane exact vectors, with raw packed IPC reaching 7,995.455/s at 32
lanes. The 32-lane packed/lazy result is 35.995% of the 20,000 decision/s gate
(2.778x short). `PaddedVectorEnv` still defaults to at most eight workers, and
portable vectors remain capped at eight; widths above eight require both the
exact backend and an explicit `workers=` request.

The directly comparable paired-trig run improved a stable pinned dense-core A/B
from 1,246.873 to 1,448.173 decisions/s (+16.14%), the dense native simulator
from 1,266.212 to 1,479.193/s (+16.82%), and the 30,000-tick 48-body physics
workload from 64,638.838 to 73,928.145 ticks/s (+14.37%). The retained
positive-zero trig fast path adds another 1.287% in a controlled local A/B
against the paired host; that isolated measurement is not stored in the wide
pipeline artifact. Profiling continues to place the solver/runtime core far
above IPC as the dominant cost.

At a 1,000-action history, a local fork/COW branch takes 0.480 ms median versus
95.933 ms for durable replay restore, a
200.0x advantage, while the durable bytes remain the portable snapshot
representation. The current wide measurements come from
[`exact-pipeline-range-safe-wide-2026-07-20.json`](benchmarks/results/exact-pipeline-range-safe-wide-2026-07-20.json),
SHA-256
`91c8db5feb9d3c8339d101940f05a42d93a4490641745964a0ca427553b8b8e9`.
It measures 1,498.136 dense native decisions/s and 75,819.177 ticks/s on the
directly comparable 30,000-tick 48-body physics workload.
[`exact-pipeline-paired-trig-2026-07-20.json`](benchmarks/results/exact-pipeline-paired-trig-2026-07-20.json)
is the comparable post-trig artifact, and
[`exact-pipeline-final-2026-07-20.json`](benchmarks/results/exact-pipeline-final-2026-07-20.json)
is its pre-trig baseline.

The optimization changes only the checked-in native MSVC runtime shim. An
instrumented 3,000-decision run found 13,096,069 raw-float-matched cosine/sine
pairs. The shim computes each pair with one x87 `FSINCOS`, returns the cosine,
caches the float-rounded sine under the raw input bits, and retains standalone
`FSIN` for a nonmatching sine. Of those inputs, 833,228 (6.362%) are raw `+0`;
that exact case now stores sine `+0` and returns cosine `1` without executing
`FSINCOS`. Raw `-0` and ordinary nonzero inputs retain the general path; raw
absolute-angle bits at or above `0x5f000000` fall back to direct `FCOS` then
`FSIN`, covering all finite snapshot angles without relying on out-of-range
`FSINCOS`. Public multiworld calls remain serialized; no Box2D iteration count,
solver ordering, or game-rule operation changed. The four-replay/full-47,019-
step exact gate remains passing.

The validated mechanics configuration is immutable after `Simulator`
construction. Its configuration hash is therefore computed once and cached for
all later transitions, diagnostics, and snapshots instead of being rebuilt in
the hot step path; the hash value and compatibility checks are unchanged.

The supported mode-0 defaults are field
`(130,120,320,250,blank=40,thickness=16)`, sizes
`[32,46,54,60,72,90,140,5,5,5]` with weights
`[20,28,28,14,5,3,1,0,0,0]`, block life 100,000, strict rot threshold 40,
projectile life 3,000, gauge 40,000/3,000, and ten qualifying clears per level.
Replay header offset `+0x10` records mode; every preserved external replay says
mode 0. Nominal native config hash is `0xec0e8463feaf2670`.

## Sources and Evidence

Use the following as initial documentation, not unquestionable truth:

- Official manual: <https://katatema.main.jp/irisu/manual.html>
- Contemporary Vector review and creator interview: <https://www.vector.co.jp/magazine/softnews/081101/n0811014.html>
- Community system details: <https://w.atwiki.jp/irisu_syndrome/pages/14.html>
- English gameplay guide: <https://gamefaqs.gamespot.com/pc/206701-irisu-syndrome/faqs/75396/gameplay-basics-amp-tips>
- Original developer's replay implementation notes: <https://wtetsu.hatenablog.com/entry/20080927/1222495777>
- High-score strategy and 214,453-point replay: <https://w.atwiki.jp/loveinch/pages/50.html>
- Public replay-structure research: <https://github.com/hoangcaominh/irisu-rpy-struct>

The official manual identifies D 1.030, DX Library, and Box2D, documents weak/strong mouse shots and fast-forward, and describes `.rpy` replay support. The executable/DLL analysis now resolves the normal-mode constants, formulas, dispatcher, replay cadence, and mode-selected parameter table. Seed 41 supplies one exact original score transition; broader controlled probes and qualitative policy transfer remain open validation gates.

For every mechanic or constant, record its provenance as one of:

- `official`: explicitly documented by the author.
- `shipped-config`: present in a configuration file shipped with the target build, but runtime interpretation may remain unverified.
- `binary-derived`: recovered from the target executable or DLL and independently checked where practical.
- `observed`: measured in controlled experiments against the original game.
- `community`: described by a secondary source but not yet measured.
- `inferred`: best current explanation of observations.
- `placeholder`: deliberately temporary and still unvalidated.

Never bury an uncertain guess in code. Put uncertain values in versioned configuration with a provenance note and a test plan.

## Executive Recommendation

Build the clone as a measured behavioral model of **v2.03 normal mode**, not as a visual remake and not as a speculative modern-Box2D approximation.

The highest-probability route is:

1. Treat the original v2.03 executable and its shipped `Box2D.dll` as behavioral oracles.
2. Recover and test the small legacy physics subset first, using the DLL wrapper and the zlib-licensed Box2DJS 1.4.3.1-derived source as references.
3. Build the original-game measurement harness and the deterministic clone vertical slice in parallel.
4. Add lifecycle rules, scoring, gauge, spawning, and difficulty one independently validated mechanic at a time.
5. Convert every accepted measurement into configuration provenance plus a golden regression test.
6. Validate short-horizon trajectories and discrete outcomes; do not require indefinite replay synchronization from a chaotic physics system.
7. Begin large-scale RL only after the clone passes the fidelity, determinism, snapshot, and throughput gates in this document.

Do not begin by watching footage and guessing the whole game loop. Do not use Box2D 2.x defaults and tune around their consequences. Do not train a policy against an unfinished simulator: early policies will discover and amplify its mistakes.

## Historical findings snapshot (superseded)

This section preserves the consolidated starting state that informed the
implementation. Current evidence and resolved values live in the implemented
evidence update above, [`docs/mechanics.md`](docs/mechanics.md), and
[`reference/game-rules-analysis.md`](reference/game-rules-analysis.md).

### Target and version boundary

- The target is the English-patched **v2.03 normal puzzle mode**. The engine is the Japanese v2.03 executable with patched data archives.
- The shipped changelog records 1.00, 1.01, and 2.00–2.03. Historical Vector pages also establish v1.02.
- Community record tables explicitly separate pre-2.00 and v2+ normal-mode records because mechanics changed substantially.
- Historical 1.00–1.02 archives were identified by exact names and byte sizes, but their payloads were not recovered. Do not block v2.03 work on them.
- The January 2009 214,453-point replay predates 2.00, probably came from 1.02, and must not be used as v2.03 golden physics.

### Physics engine and wrapper

- The shipped PE32 x86 `Box2D.dll` is legacy Box2D 1.4.x, very likely the 1.4.3 lineage. Evidence includes `BodyDef.AddShape`, origin-position APIs, object layouts, and `userData` at the legacy body offset.
- The earlier Box2D 2.0.1 timing guess is disproven.
- The DLL exposes only 16 `stdcall` functions: initialize/dispose, step, create box/circle/triangle, destroy body, enumerate contacts, get position/rotation/velocity, set position/velocity/user data, and one unused test function.
- Normal game code calls `b2d_step(world_step * integer_time_multiplier, 10)`. The configured `world_step` is `0.020`, so normal stepping is a strong 50 Hz lead with exactly 10 solver iterations.
- Wrapper dimensions and positions are divided by literal magnification 10 on input and positions are multiplied back on output; the INI value 100 is not the normal world-call argument.
- Velocity conversion is asymmetric: `set_v` divides by 10, while `get_v` returns raw world units. The clone reproduces that observable boundary.
- `set_position` zeros linear velocity while preserving angular velocity.
- Box2DJS 0.1.0 is locally cached under its zlib license. Its official project says it was mechanically converted from Box2DFlashAS3 1.4.3.1. It is the best readable semantics reference currently available, but only the shipped DLL is numerical ground truth.

### Shipped INI table (later identified as nonzero-mode/Metsu-side)

These are historical `shipped-config` facts. Later executable analysis proved
that they match the nonzero-mode/Metsu-side initializer, not mode-0 normal:

| Area | Current values |
|---|---|
| Field | `x=94`, `y=20`, `width=420`, `height=370`, `blank=30`, `thick=16` |
| Extra boundaries | `top=-140`, `top_w=450`, `top_h=300`, `bottom_h=400` |
| Normal block material | density `1`, friction `1`, restitution `0` |
| Normal block size/weight slots | sizes `30, 48, 64`; weights `10, 40, 20` |
| Normal block timers | life span `10000`, death delay `120` |
| Projectile | size `24`, density `8`, friction `1`, restitution `0`, life span `1200` |
| Weak/strong velocity leads | vertical velocities `-250`, `-500`; exact mapping must be confirmed |
| Special/heavy block | size `24`, density `50`, friction `0.1`, restitution `0.6` |
| Wall material | friction `1`, restitution `1` |
| Bottom material | friction `1`, restitution `0` |
| Top material | friction `1`, restitution `0.5` |
| Gauge | maximum `10000`, initial `1000`, plus unit `120`, rotten penalty `5000` |
| World | magnification `100`, nominal step `0.020` |

Unresolved one-letter/configuration keys include normal frequency `a=50`, speed `b=0.1`, color `c=8`, ex frequency `d=1`, and `p=2000`. Embedded executable names include `level_table.txt`, `block_color_kind`, `block_speed`, `tri_ratio`, `score_atozdj`, `yeild_interval`, and `game_life_*`. Do not guess their meanings in production code; correlate them with call sites or disposable config-perturbation experiments.

### Replay format and corpus

- v2.03 writes five signed little-endian 32-bit fields—seed, highest level, final score, highest chain, and mode—followed by 32 bytes and then 4-byte input records. Legacy files begin records after the first 20 bytes.
- The v2.03 loader always consumes 52 header bytes, so it swallows the first eight inputs of a legacy replay during cross-version playback.
- Each input word stores left/right button levels, 10-bit X, 9-bit Y, and 11 decoder-ignored high bits. Real files sometimes set ignored bits.
- One record is emitted per game input update and consumed per 0.020-second simulation update; fast-forward skips rendering only.
- The local corpus contains one self-generated v2.03 replay and five external normal-mode files scoring 40, 56, 41,449, 43,791, and 214,453. Only the last has a very large chain, and it is from the older mechanics line.
- A password-locked 132,117-point replay is visible on the original community uploader. Do not bypass the password.
- The original developer states that replays store gameplay RNG seed plus input and can diverge across computers because of Box2D behavior.

### Archives, footage, and prior code

- The DXArchive key is `shine`. The official decoder successfully extracted the pristine and English-patched `dat.dxa` files. They contain presentation/config resources but did not expose the embedded `level_table.txt` name.
- The ignored corpus contains eight gameplay recordings, including the v2.03 330k level-100 run, the 214k chain-strategy run, a 49-second 40k run, and a 58-minute 100% run.
- A repository named `JosephJeongs/IrisuSyndrome` is only an empty SFML template. The hyphenated `JosephJeongs/Irisu-Syndrome` is a roughly 205-line physics toy, not a game clone; its constants conflict with shipped evidence.
- The prior `Gabriel-Kahen/irisu_blackbox` repository contains useful capture, HUD, OCR, action-grid, Gymnasium, and test code. Its Windows/PyAutoGUI input route and old screen coordinates are not suitable for this host, but its perception and failure-handling code are worth mining.
- Contemporary documentation supports confirmed-chain landing resolution, no reward for confirmed bodies lost out of bounds or destroyed by excess shots, rainbow-orb color clears, and level-dependent color/speed/rot changes. Because early-version clear rules changed, reproduce these on v2.03 before freezing them.

### Remaining high-impact unknowns

- Exact triangle vertices, fixture filters, gravity, damping, sleeping, continuous-collision behavior, and contact ordering.
- Scripted descent rate and the precise event that converts a falling body to dynamic fresh state.
- Input/replay cadence, weak/strong mapping, shot lifetime units, click-rate limit, and fast-forward multipliers.
- Chain graph construction, simultaneous contacts, confirmation/deletion timing, direct-hit counters, and projectile-projectile effects.
- Exact scoring formula, gauge arithmetic/rounding/decay, rot damage scaling, level thresholds, spawn schedule, colors, sizes, shapes, and bonus frequency.
- Boundary openings, side-ejection classification, off-screen lifetime, and upper-opening chain-anchor behavior.
- The original gameplay PRNG algorithm, seeding, number of RNG streams, and exact order of random draws.

These unknowns define the measurement backlog. They are not permission to insert plausible defaults silently.

## Available Local Reference Corpus

Read [`reference/README.md`](./reference/README.md), [`reference/computer-use.md`](./reference/computer-use.md), [`reference/environment.md`](./reference/environment.md), [`reference/mechanics-evidence.md`](./reference/mechanics-evidence.md), [`reference/binary-analysis.md`](./reference/binary-analysis.md), [`reference/version-history.md`](./reference/version-history.md), and [`reference/manifest.md`](./reference/manifest.md) before starting reference work. The Git-ignored corpus includes:

- A runnable English-patched v2.03 working copy and its shipped `Box2D.dll`.
- A plaintext shipped v2.03 configuration matching the nonzero-mode/Metsu-side parameter table; mode-0 normal values come from the executable initializer and proxy trace.
- Pristine v2.03 base and English patch archives.
- One short replay generated locally by v2.03.
- Five external normal-mode replays: padded v2-era traces scoring 40, 56, 41,449, and 43,791, plus a legacy-layout 214,453-point run with max chain 42.
- Eight gameplay recordings, including ordinary normal-mode play, a two-part v2.03 level-100 330k run, the 214k chain-strategy run, a 49-second 40k run, and a 100% run.

Use `tools/launch-reference-game.sh` to start the workspace copy, `tools/create-reference-run.sh` to create a disposable experiment tree, and `tools/inspect-rpy.py` for an initial replay report.

Agents controlling the original game must follow `reference/computer-use.md`: claim only the exact IriSu window, use background capture and targeted input, preserve a complete experiment bundle, and release the claim in cleanup. Do not improvise global desktop automation.

The original developer states that replays store the gameplay RNG seed plus input state and that Box2D replay behavior can diverge between computers. Replays are therefore high-value input traces and metadata, but playback agreement on this Wine environment must be measured rather than assumed.

The shipped `Box2D.dll` is unusually helpful: it exposes a small 16-function stdcall wrapper rather than forcing all investigation through the full game. Its ABI and most semantics have been recovered in `reference/binary-analysis.md`. Build a small Windows probe program that exercises the original DLL under Wine to validate exact fixture construction, legacy solver/contact behavior, gravity, sleeping, and the wrapper's asymmetric velocity units. Keep the original DLL confined to the ignored reference lab; the clean clone must not require it.

Binary layout evidence identifies legacy Box2D 1.4.x, likely the 1.4.3 lineage. The previously cached Box2D 2.0.1 candidate is not the shipped engine. Do not start from 2.x behavior without differential evidence.

Current evidence confirms two replay layouts. Both start with five little-endian signed 32-bit fields: seed, highest level, final score, highest chain, and mode. v2.03-era files have 32 bytes after that header before 4-byte frame records; the older 214,453-point replay starts frame records immediately. A public Kaitai schema describes the padded form, but it has no license and does not cover the observed older layout. `tools/inspect-rpy.py` is an independent conservative parser; extend it with controlled replay tests rather than copying unlicensed schema code.

## Recommended Clone-Build Workflow

### 1. Freeze contracts and evidence before mechanics code

Create the initial repository contracts first:

- explicit units and coordinate transforms;
- typed mechanics configuration with value, provenance, units, uncertainty, and validating experiment IDs;
- body, event, action, observation, snapshot, and replay-trace schemas;
- a mechanics uncertainty register ordered by transfer risk;
- deterministic test/build scaffolding.

Use `v2.03-normal` as a named mechanics profile. Keep legacy replay parsing separate from the nominal environment. The first code review should reject unexplained literals in the mechanics core.

### 2. Build two small oracles before the full game

Build these in parallel:

**Original-DLL probe**

- A minimal 32-bit Windows executable that dynamically loads the ignored shipped `Box2D.dll` under Wine.
- Exercise every recovered wrapper operation with isolated box, circle, and triangle scenarios.
- Emit JSON/CSV traces for transforms, velocities, contacts, sleeping, material mixing, boundary cases, and repeated steps.
- Sweep timestep, iterations, magnification, density, friction, restitution, position, rotation, and velocity.

**Original-game reference harness**

- Launch only disposable run trees.
- Use the native same-session protocol in `reference/computer-use.md` for exact captures and targeted input.
- Record metadata, action JSONL, captures/video, before/after file hashes, measurements, and the resulting `.rpy` in one experiment bundle.
- Begin with replay cadence, coordinate mapping, single weak/strong shots, isolated falls, and easily classified lifecycle events.

These tools are more valuable than a large speculative clone. They turn future disagreements into measurable questions.

### 3. Choose the physics implementation empirically

The preferred starting point is a small C++20 legacy-compatibility core implementing only the used Box2D 1.4-era subset. First try to locate a properly licensed original 1.4.3 C++ source snapshot. If unavailable, port the necessary semantics from the cached zlib-licensed Box2DJS source with attribution, then validate it against the original DLL probe.

Do not implement joints or unused engine features. The required subset appears to be:

- bounded world/AABB broad phase;
- static and dynamic bodies;
- circle and convex polygon fixtures;
- box and exact game triangle geometry;
- mass/inertia and material mixing;
- contact generation/iteration;
- sleeping and fixed-step integration;
- position/velocity/user-data access.

A modern Box2D adapter can be built as a comparison baseline, but should become nominal only if it matches the DLL more closely after measured calibration. A custom physics engine from scratch is the last resort because subtle solver/contact errors are the largest transfer risk.

### 4. Implement one deterministic vertical slice

Before spawning, scoring, or RL, support:

- field fixtures from typed configuration;
- one scripted-falling colored body;
- activation into a dynamic fresh body;
- one weak or strong projectile at a raw cursor coordinate;
- fixed stepping and structured collision events;
- snapshot, restore, canonical state hash, and diagnostic rendering.

Gate this slice with DLL differential tests and short original-game trajectory probes. Snapshot/restore and same-seed/same-actions equivalence are required now, not retrofitted after search is built.

### 5. Add game rules as a deterministic state machine

Keep physics contacts as observations; resolve game rules in a separate, explicitly ordered event phase. Add in this order:

1. fresh activation and floor rot;
2. fresh–rotten clearing;
3. fresh–fresh confirmation;
4. chain membership and expansion;
5. delayed landing resolution/deletion;
6. projectile hit accounting and unrewarded destruction;
7. side exits and off-screen cleanup;
8. gauge, scoring, and game over;
9. bonus orb;
10. projectile-projectile and simultaneous-contact edge cases.

For each mechanic, first write a minimal discriminating reference experiment, then implement, then add its golden test. Do not batch several unknown mechanics into one opaque test.

### 6. Recover spawning and difficulty statistically

Instrument long no-input and controlled-play runs. Recover score/level thresholds, color count, size/shape selection, spawn interval and position, fall speed, bonus frequency, and gauge/rot scaling by phase.

Compare distributions with confidence intervals. Match the PRNG and draw ordering if practical; otherwise implement a deterministic clone PRNG and train with uncertainty-aware randomization while preserving measured marginal and conditional distributions. Never expose future draws to the policy.

### 7. Calibrate nominal and randomized profiles

Maintain:

- one nominal `v2.03-normal` configuration for regression and golden tests;
- an uncertainty table tied to measurements;
- narrowly randomized training profiles drawn from those uncertainty bounds.

Use short-horizon state/trajectory errors, event agreement, exact score/gauge transitions, and distributional spawn agreement. Long replay divergence is diagnostic but not itself failure if local predictions remain calibrated.

### 8. Integrate the Gymnasium/vector API last

Wrap the proven mechanics core with the environment contract in this document. Keep Python out of the per-tick physics loop. Establish random and scripted baselines, vector throughput, complete snapshot cloning, and deterministic trace replay before connecting PPO, SAC, or planning.

### 9. Run continuous real-game transfer checks

The first policy does not need to be strong. A scripted matcher/ejector should already demonstrate that cloned shots and qualitative tactics transfer. After RL begins, replay representative policy states/actions in the original game, add discrepancies to the uncertainty register, and retrain or re-evaluate any policy that could exploit a fixed mismatch.

## First Agent Work Package

The receiving agent should start with these concrete outputs, in order:

1. Create `docs/mechanics.md` with the consolidated known/unknown register and provenance copied from this document and the reference evidence files.
2. Define the typed `v2.03-normal` mechanics configuration and unit conventions without yet asserting meanings for unresolved keys.
3. Add automated replay-parser tests for padded and legacy headers, 10-bit X, 9-bit Y, ignored high bits, and malformed sizes.
4. Scaffold the 32-bit original-DLL probe and reproduce at least initialization, one static boundary, one dynamic box, one circle projectile, stepping, positions, velocities, and contact enumeration.
5. Create the first same-session experiment bundle: window/puzzle coordinates plus one weak and one strong isolated shot.
6. Produce a physics-implementation decision note comparing a legacy subset port with a modern Box2D adapter against the probe.
7. Only then implement the deterministic vertical slice and its snapshot/hash tests.

The first handoff checkpoint is not “the game runs.” It is: the replay parser is tested, the DLL oracle produces machine-readable traces, the first original-game experiment is reproducible, units/configuration are explicit, and the physics choice is backed by measurements.

## Known Mechanics to Represent

The initial mechanics model must support the following, subject to validation:

- Colored bodies spawn above and initially descend in a scripted/kinematic state.
- A relevant collision activates normal physics for a fresh body.
- The mouse cursor defines the origin of a projectile travelling vertically upward.
- Left click produces a weak shot; right click produces a faster or stronger shot.
- Fresh same-color contact can form a confirmed chain that resolves later and receives a chain-dependent score.
- Additional directly connected same-color fresh bodies can join a chain.
- Excess direct projectile hits can destroy confirmed bodies without normal reward.
- Fresh and rotten same-color bodies can clear immediately.
- Rotten and rotten bodies do not normally clear one another.
- Contact with the floor or rotten bodies can make a fresh body rotten and damage the gauge.
- Valid boundary openings can remove bodies without the normal clear or rot result.
- Same-color clears affect score and gauge.
- Gauge depletion terminates the run; normal mode also applies time pressure.
- Score changes level or difficulty phases, including color count and fall/spawn pressure.
- A heavy bonus orb can trigger a color-wide clear or gauge recovery behavior.
- Fast-forward exists but is out of the initial training action space.
- Fired projectiles collide with one another; high-score documentation claims this interaction can change how subsequent direct hits are counted.
- High-score play may wedge large bodies partly into upper side openings to create persistent chain anchors.

Anything more specific must be verified or marked uncertain.

## Environment Contract

Provide a Gymnasium-compatible Python API over a native or otherwise high-performance simulation core.

Minimum operations:

```text
reset(seed, config) -> observation, info
step(action) -> observation, reward, terminated, truncated, info
clone_state() -> snapshot
restore_state(snapshot)
state_hash() -> stable hash
render(mode)
```

`reset` must fully reset physics, RNG, contacts, counters, timers, caches, and observations. Two environments started from the same seed and configuration and given the same action trace must produce matching hashes throughout a run on the supported build platform.

`clone_state` and `restore_state` must include every state component that can affect the future, including RNG, spawn counters, pending contacts/events, chain state, timers, and global difficulty state. Search cannot be trusted until clone/restore tests pass.

Rendering, audio, logging, and Python callbacks must not be part of the physics loop. Diagnostic rendering may be added as a separate consumer of state.

## Observation Contract

The privileged training observation is an object set plus global features.

Each body should expose, as applicable:

```text
stable episode-local ID
body class: colored piece / projectile / bonus
color
shape and dimensions
x, y, angle
vx, vy, angular velocity
lifecycle: scripted-falling / dynamic-fresh / confirmed / rotten
chain or group ID
projectile hit count
age and relevant state timers
```

Global features should include:

```text
gauge
score
level
elapsed simulation ticks
difficulty/spawn phase observable to the player
current legal field boundaries
recent action history if required by the model
```

Do not expose RNG state, future spawns, hidden event schedules, or other information that a causal visual observer cannot infer. Those values may exist inside snapshots but must not enter policy observations.

Use a variable-length set representation or a padded representation with an explicit mask. Define coordinate conventions and normalization in one place.

## Action Contract

The base legal action is:

```text
Action {
  kind: wait | weak_shot | strong_shot
  cursor_x: continuous legal field coordinate
  cursor_y: continuous legal field coordinate
  wait_ticks: positive integer when kind == wait
}
```

Mouse movement itself has no simulated cost unless original-game experiments show otherwise. Shot timing is part of the action. Validate the maximum legal click frequency and enforce it consistently in training and evaluation.

The core must support raw legal cursor actions. Higher-level training helpers may generate candidate actions relative to bodies or contact points, but the simulator must not require those macros. This prevents the action design from excluding unexpected superhuman tactics.

Invalid actions must be handled deterministically and reported in `info`; do not silently reinterpret them.

## Reward and Episode Contract

The canonical environment reward is:

```text
reward = score_after - score_before
```

Expose diagnostic event data in `info`, not as permanent extra reward. Curriculum wrappers may add explicitly configured potential-based shaping, but the core environment must preserve the true score reward.

An episode terminates when the real game-over condition occurs. A training time limit is a truncation, not a termination. Score and gauge arithmetic must use types that cannot silently overflow during exceptional runs.

## Determinism Standards

- Advance physics using a fixed simulation timestep, never wall-clock time.
- Use one explicit, seedable PRNG implementation for game randomness.
- Keep deterministic event ordering for contacts and simultaneous state changes.
- Run each environment's physics single-threaded.
- Pin the Box2D version and toolchain in the build metadata.
- Avoid unordered iteration where it affects simulation outcomes.
- Serialize or hash state in a canonical order.
- Include same-seed/same-actions regression tests of meaningful length.

Bitwise identity is required on the supported reference build. Cross-platform bitwise identity is desirable but should not be claimed without evidence.

## Fidelity Standards

Validate local mechanics rather than demanding indefinite replay synchronization. Repeated collision systems are chaotic, so small numerical differences will eventually produce different long trajectories.

The initial bulk-training gate requires:

- Correct discrete outcomes on at least 95% of controlled match, rot, chain, ejection, and orb probes.
- Exact score, gauge, and level-transition results for all validated golden scenarios.
- Short-horizon trajectories that remain within a documented tolerance, initially targeting less than a fraction of a typical body width over 0.5–1.0 seconds.
- Spawn and difficulty distributions statistically consistent with measured original-game samples.
- No known high-impact discrepancy left undocumented.
- Basic scripted policies demonstrating the same qualitative skills in the clone and original game.

These are initial gates, not excuses to ignore remaining errors. Tighten tolerances as measurement quality improves.

Every discovered discrepancy must produce:

1. A minimal reproduction trace.
2. An explanation or explicit unknown.
3. A regression test where possible.
4. A provenance/configuration update.
5. Re-evaluation of policies that could have exploited it.

## Reference-Game Calibration

Build or maintain a separate reference harness that can:

- Start the original executable in a controlled environment such as Wine.
- Capture puzzle-region frames with stable timestamps.
- Inject cursor positions and left/right clicks at controlled ticks.
- Record the complete input trace and environment metadata.
- Preserve original `.rpy` files without modifying them.
- Run repeated probes and export machine-readable tracking results.
- Exercise the shipped `Box2D.dll` through a minimal probe executable after its small wrapper API has been independently characterized.

Investigate the `.rpy` format early. Determine whether it contains seeds, time steps, cursor positions, buttons, or other state. A parser must reject malformed or unknown variants explicitly.

The recovered v2.03 decoder uses one 4-byte input word per update: left click, right click, 10-bit cursor X, 9-bit cursor Y, and 11 ignored high bits. Independently confirm frame cadence, button level/edge behavior, fast-forward representation, and whether no-input frames still carry cursor position. Preserve the ignored high bits; real replays sometimes set them.

Use computer vision to track original-game bodies through controlled experiments. Fit or validate:

- Field geometry and coordinate transforms.
- Body fixtures, mass, and inertia.
- Gravity and scripted descent.
- Weak and strong projectile speed/impulse.
- Friction, restitution, damping, and collision filtering.
- Activation, confirmation, rot, and deletion timers.
- Chain membership and scoring.
- Gauge decay, damage, and recovery.
- Spawn timing, colors, geometry, bonus objects, and difficulty phases.
- Projectile-projectile collision behavior and direct-hit accounting.
- Large-body behavior at the upper side openings and chain-anchor strategies.

Do not commit the original binary, art, music, or other copyrighted assets. Keep reference data paths configurable and document how an authorized local copy is supplied.

## Robustness for Transfer

Exact calibration alone is insufficient. Training must tolerate a plausible family of original-game dynamics.

Support narrowly randomized:

- Shot impulse and timing.
- Gravity and scripted descent rate.
- Friction, restitution, damping, mass, and geometry.
- Contact/event timing around ambiguous boundaries.
- Observation delay and action delay.
- Cursor coordinate error.
- Missed, noisy, or merged object observations.

Keep randomization ranges tied to measurement uncertainty. Do not use extreme randomization that changes the identity or strategy of the game.

Maintain a nominal calibrated configuration for regression tests and randomized configurations for robust policy training.

## Performance Standards

Before large training runs:

- Benchmark physics-only and Python-vectorized throughput separately.
- Target at least tens of thousands of aggregate decision steps per second on the available multicore machine for typical board states.
- Demonstrate parallel independent environments without shared mutable state.
- Profile before optimizing; do not trade correctness or determinism for speculative speed.
- Keep state cloning efficient enough for hundreds of short planning rollouts per chosen action, or document the limiting cost.

No renderer, sleep, audio, or wall-clock synchronization may run during headless training.

## Engineering Standards

- Prefer a small C++20 simulation core with an explicitly pinned legacy-compatible physics implementation and a thin Python binding, unless repository evidence supports a better stack. Do not silently substitute modern Box2D semantics for the shipped 1.4-era behavior.
- Keep game rules separate from physics-engine callbacks and rendering.
- Keep constants in typed, versioned configuration rather than scattered literals.
- Use explicit units for time, position, velocity, angles, and score/gauge values.
- Make state transitions inspectable through structured events.
- Keep functions and modules focused; avoid premature frameworks.
- Add tests with every implemented mechanic and every fixed discrepancy.
- Run formatters, static analysis, and sanitizers in CI where practical.
- Fail loudly on invalid configuration, incompatible snapshots, or replay versions.
- Record build version, mechanics configuration hash, seed, and action trace for every benchmark result.
- Preserve reproducibility: a reported run must be replayable from its metadata.

## Suggested Repository Layout

The exact names may change, but preserve the separation of concerns:

```text
clone/
  core/           physics and deterministic game rules
  include/        public native interfaces
  bindings/       thin Python binding
  configs/        mechanics configurations and provenance
  render/         optional diagnostic renderer
python/
  irisu_env/      Gymnasium API and vector wrappers
reference/
  tools/          replay parsing, capture, and calibration utilities
  README.md       local original-game setup; no copyrighted assets
tests/
  unit/           isolated rule and state tests
  determinism/    seed, snapshot, and trace tests
  golden/         original-game-derived scenarios
benchmarks/       throughput and fidelity benchmarks
docs/
  mechanics.md    current rules and uncertainty register
  fidelity.md     calibration methods and results
```

Do not create empty architectural layers merely to match this tree. Add structure as real code requires it.

## Milestones

### M0: Contracts and Evidence Ledger

- Define units, coordinates, configuration schema, state, action, observation, and event types.
- Establish tests and reproducible builds.
- Create the mechanics uncertainty/provenance register.
- Add tested padded/legacy replay parsing.

### M1: Reference Oracles and Physics Decision

- Build the original-DLL probe and machine-readable differential cases.
- Build the first reproducible original-game computer-use experiments.
- Compare legacy-subset and modern-adapter candidates.
- Select and pin the implementation stack based on measured behavior.

### M2: Deterministic Physics Vertical Slice

- Field boundaries, one colored body, one weak/strong projectile, fixed-step physics.
- Seeded reset and complete state cleanup.
- Snapshot, restore, and state hash.
- Headless stepping and diagnostic rendering.
- Same-seed and snapshot equivalence tests.
- Short-horizon DLL and original-game trajectory checks.

### M3: Core Lifecycle and Scoring

- Scripted falling and activation.
- Fresh, confirmed, rotten, and deleted states.
- Same-color contact, chains, floor rot, side exits, gauge, score, and termination.
- Structured event trace, unit tests, and golden reference probes.

### M4: Full Normal-Mode Structure and Calibration

- Spawning, colors/geometries, level/difficulty phases, bonus orb, and validated edge cases.
- Python environment and vector execution.
- Random-action and scripted-policy integration tests.
- Fitted mechanics configurations and golden tests.
- Fidelity report with measured tolerances and known gaps.

### M5: Training Readiness

- Fidelity gates pass.
- Determinism and snapshot tests pass.
- Throughput target is met or the measured limitation is accepted explicitly.
- Robustness randomization is supported.
- A scripted policy transfers basic skills to the original game.

## Definition of Done

The clone phase is done only when all of the following are true:

1. The normal puzzle mode can run headlessly from reset to game over.
2. Legal action traces are deterministic and replayable on the supported reference build.
3. Snapshot/restore produces behavior equivalent to uninterrupted execution.
4. Scoring, gauge behavior, lifecycle transitions, spawning, and difficulty are covered by tests.
5. Original-game-derived golden scenarios meet the documented fidelity gates.
6. Unknowns and remaining discrepancies are visible and prioritized.
7. Vectorized throughput is suitable for RL and search.
8. No original copyrighted assets or binaries are required inside the repository.
9. At least one nontrivial scripted policy behaves qualitatively similarly in the clone and original game.
10. Another agent can reproduce tests and benchmarks from repository instructions alone, given an authorized local original-game copy for reference-only checks.

Passing unit tests without original-game calibration does not satisfy this definition.
