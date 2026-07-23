# R4b live calibration and soak operations

R4b turns the provider-independent R4a harness into an executable, fail-closed
measurement boundary. It does **not** claim that live input is currently
qualified. A fresh 2026-07-23 disposable v2.03 run proved exact background
capture of the 644×484 Wine/XWayland window, then released the exact claim
without sending input. The installed same-session provider remains
`input_ineligible`: it exposes one atomic click but no independently
acknowledged down/up, authoritative release deadline, release-all operation, or
claim-expiry neutralization. The asset-free result is recorded in
[`reference/r4b-provider-preflight-20260723.json`](../reference/r4b-provider-preflight-20260723.json);
the raw frame remains ignored.

The checked deployment contract therefore stays
`measurement_status = "provisional_unmeasured"` and
`live_deployment_enabled = false`.

## Why a separate broker is required

IriSu runs through Wine/XWayland and polls mouse-button levels on its native
20 ms update path. Two caller-side operations, a drag, raw `xdotool`, or a
Python watchdog cannot substitute for one authoritative input broker. If the
caller stalls or dies after button-down, that broker must still:

1. release the button by the accepted absolute monotonic deadline;
2. neutralize every button on claim release, expiry, or disconnect;
3. reject a stale fencing generation or changed process/window identity; and
4. attest the exact operation, target, button transition, deadline, broker
   instance, and clock domain in its acknowledgement.

The JSON-lines broker client pins a regular, non-symlink, owner-controlled
executable by SHA-256 and executes the already-open file descriptor, never a
shell command. Its handshake pins protocol, implementation, audited input
backend, broker instance, capability set, and Linux `CLOCK_MONOTONIC` domain.
Every response is bounded, sequence checked, duplicate-key rejecting, and exact
schema validated. Claim tokens and the launch nonce never belong in reports.

The required broker backend is
`hyprland_native_targeted_edges_v1`. Advertising safe capabilities from an
unrecognized backend is rejected. A conforming implementation must still pass
native fault injection—especially caller death after down, claim expiry,
disconnect, broker restart, and deadline expiry—before its build hash may be
placed in a frozen calibration plan.

## Executable and window identity

Only a newly prepared child of `reference/runs/<experiment-id>` is eligible.
R4b rejects the preserved source tree, symlinks, hard-linked required files,
missing or mismatched disposable-run markers, trace-proxy DLLs, and any change
to the executable, Box2D, DxLib, configuration, or packed data during a run.
All eight canonical v2.03 hashes and the preregistered Wine executable are
pinned. Attestation also snapshots every other immutable file through
descriptor-relative no-follow traversal, rejects any unapproved Windows module,
and permits mutation only for the exact known output paths (`save.dat`,
`photo.png`, and `replay/new.rpy`). Live orchestration must re-attest that
baseline immediately around every input/observation boundary.

The launcher strips ambient Wine overrides and binds every owned, non-writable
file plus the permitted standard `dosdevices`/user-shell symlinks in the fixed
Wine prefix. Prefix content is SHA-bound into measurement provenance and its
metadata baseline is re-attested immediately before and after input; any
prefix mutation invalidates the run. The launcher then
holds a cross-process measurement lock, starts one new process session, and
creates a 256-bit launch nonce. It never kills a shared wineserver or scans and
signals unrelated Wine processes. A claimed target binds:

- exact window address and capture identity;
- Unix PID and `/proc` process-start generation;
- launch-nonce hash;
- game executable and complete disposable-runtime identity hashes;
- Wine executable and complete Wine-prefix hashes; and
- broker instance and monotonically increasing claim generation.

The broker must discover exactly one target carrying the raw nonce. Title,
class, process name, or a matching icon is not authority. Every renewal must
preserve the target, token, generation, and broker while strictly extending a
still-live lease.

After discovery, the launcher must bind that PID/start generation to its raw
nonce and newly created Unix session. It pins the process with a pidfd and
requires the bound target to exit during cleanup. The calibration runner
refuses a harness without both runtime and Wine-prefix guards, and the harness
rechecks the exact binding around capture and input. This deliberately does not
signal a shared wineserver; production orchestration must call `bind_target`
before opening the guarded harness.

## Input-free preflight

`tools/run-r4b-preflight.py` only claims, captures, observes the cursor, verifies
runtime identity again, and releases the claim. It never invokes down, up,
click, drag, or release-all. The resulting private report contains hashes,
dimensions, bounded durations, capability booleans, and blockers—not pixels,
tokens, titles, launch nonces, or personal paths.

Reports and journals require an owner-controlled, non-symlink directory with
mode `0700`. Files are created once with `O_NOFOLLOW|O_EXCL`, mode `0600`, an
exclusive writer lock, complete-write loops, file `fsync`, and directory
`fsync`. A partial final journal record is invalid rather than silently
ignored. The R4b finalizer single-opens its private soak report/event/threshold
sources under shared locks, verifies owner/mode/link count and a bounded stable
identity, and works only from one private snapshot so repeated verification
cannot observe different source versions. Evidence and the measured contract
are published inside one fsynced, atomically renamed `0700` bundle directory;
a crash cannot expose only one member as a completed bundle.

Example after an independently audited broker exists:

```bash
mkdir -m 700 /private/r4b-preflight
PYTHONPATH=python python tools/run-r4b-preflight.py \
  <experiment-id> \
  "$PWD/reference/runs/<experiment-id>" \
  <exact-window-address> <exact-capture-id> <launch-nonce-sha256> \
  /absolute/path/to/fixed-wine-prefix \
  /absolute/path/to/audited-broker <broker-sha256> \
  /private/r4b-preflight
```

The current same-session MCP provider cannot be passed to this command because
it does not implement the explicit-edge broker protocol. That is the expected
failure mode.

## Calibration design

The checked R4b plan is preregistration, not observed data. It freezes:

- at least three fresh disposable processes;
- a deterministic hash-shuffled 9×7 interior client-coordinate lattice;
- weak/left and strong/right shots;
- three repetitions per process (1,134 commanded cells);
- half-open execution quantization,
  `floor(normalized × extent)` clamped to `[0, extent-1]`;
- a hard action and wall-time budget;
- a 1,800-second soak and at least 512 confirmed actions on each fresh process,
  with at most one second between required metric samples;
- a registration reliability gate and typed instrument resolutions; and
- moving-block/run-cluster uncertainty instead of treating adjacent frames as
  independent samples.

For each command, the raw safe journal must distinguish request, injected down,
acknowledged down, accepted release deadline, injected up, acknowledged up,
inferred game poll/effect, and first visible confirmation. A failed or
ambiguous registration is a result; it cannot receive fabricated latency or
coordinate measurements.

Before any capture or input attempt, the runner appends and `fsync`s a
write-ahead intent bound to the exact sweep cell, process ID/start generation,
launch-nonce hash, runtime identity, and audited runner/observer bundle. It then
appends one terminal outcome. Capture, fire, observer, effect-recording,
sample-validation, and final runtime/prefix revalidation failures are explicit
terminal states; a crash leaves an unmatched intent. Either condition
permanently taints that no-replace journal and prevents finalization, so an
attempted action cannot disappear from the qualifying record.

The copied `replay/new.rpy` is the authoritative IriSu-side input check. Each
confirmed command must produce exactly one new edge for the correct button and
quantized coordinate, never `BOTH`, with a sampled neutral record before the
same button is fired again. Visual projectile-birth association supplies the
coordinate residual and visible-effect timing. Broker acknowledgement alone
does not prove the game sampled an input.

The sweep order must match the preregistered permutation exactly and cover each
experiment/x/y/button/repetition cell. Missing cells, extra attempts,
one-dimensional coverage, wrong buttons, clipped coordinates, repeated edges,
or runtime/geometry drift fail the run. Detection overlays and associations
remain private and require operator review before promotion.

Every experiment retains one immutable process attestation. Launch nonces and
PID/start generations must be unique across experiment IDs; relabeling one
long-lived IriSu process as several fresh runs is rejected before the next
input and again during journal verification. The measurement provenance is
derived from the runner's observed installed `original_game` source bundle and
the observer's role-bound, owner-controlled artifact files. The runner hashes
those files itself and records the resulting build identities in each
attestation; callers cannot supply provenance strings copied from the plan.

## Statistics and soak

Calibration values are rebuilt from a typed SHA-256 journal, not accepted as
free-form aggregate JSON. Latency/cadence quantiles use deterministic
moving-block bootstrap intervals within a run and treat fresh runs as clusters.
Registration reliability uses a frozen Wilson lower confidence bound.
Instrument resolution is a floor on uncertainty; a constant sample cannot
claim zero uncertainty.

The executable plan must freeze at least one acceptance bound for every
calibration metric. Evidence construction evaluates the conservative 95%
confidence edge—point estimate plus uncertainty for maximum bounds and minus
uncertainty for minimum bounds—and fails before contract finalization if any
bound is missed.

Soak thresholds are evaluated for every experiment as well as in aggregate, so
a good run cannot hide a bad one. Safety totals are intrinsically zero for
stale/out-of-order frames, deadline or renewal failures, release failures,
cross-window routes, buffer overflow, descriptor/thread growth, wrong/repeated
button edges, and coordinate clipping. The report also retains duplicates,
drops, cadence/jitter, ring age, causal and visible latency, crop drift, and
resource growth for transfer-model fitting.

Every raw soak event carries the claimed target's Unix PID, process-start
ticks, launch-nonce hash, complete disposable-runtime identity hash, and
Wine-prefix hash. Those five fields must remain constant within an experiment.
Process generation,
launch nonce, and disposable-runtime identity must each be distinct across
experiment IDs; changing only an event's experiment label cannot manufacture
independent executable runs. The verified report persists this experiment-to-
process mapping alongside the per-experiment statistics.

The numerical latency and coordinate limits in the preregistration are
engineering acceptance proposals, not measurements. If a labeled pilot shows
they are inappropriate, freeze a new plan/version before the qualifying run;
never edit thresholds after the first journal event.

[`configs/rl/original_game/r4b-preregistration-v1.toml`](../configs/rl/original_game/r4b-preregistration-v1.toml)
records those proposals and the unresolved cursor choice. It is deliberately a
draft schema and is not accepted by the calibration-plan loader. The executable
JSON plan can be frozen only after the audited broker build, exact experiment
IDs, order seed, pilot-derived block length, and preregistered soak-threshold
hash exist. Event-stream and report hashes are derived afterward from verified
raw sources, avoiding a circular “hash the future run” requirement.

## RL transfer contract

The policy still produces continuous normalized coordinates and receives PPO
likelihood under that continuous sample. Simulator and live execution share the
manifest-bound integer-pixel lowering, so normalized `1.0` targets `(639,479)`,
not an invalid `(640,480)`.

The current simulator teleport cursor remains provisional. Human-comparable
deployment should use retained client-space cursor state with measured speed
and acceleration, or explicitly establish that a fixed-rate abstract
coordinate interface is the human comparison interface. Window borders and
display scaling must not change a client-space fairness limit.

Measured press/release duration, click-rate, game-poll phase, capture
duplicates/drops/delay, and effect-to-visible tails must become training
distributions—not backend state supplied during live play. At deployment the
student receives only captured pixels/history and emits legal actions through
this broker.

## Promotion boundary

R4b can create only a `measured_pending_review` contract with live deployment
still disabled. Promotion additionally requires:

- an audited broker build and passing native crash/deadline conformance suite;
- successful replay-edge and 2-D projectile-birth calibration;
- a frozen cursor/human-comparison protocol;
- a passing 30-minute production-cadence soak on every declared process;
- operator approval of gameplay state, crop overlays, and associations; and
- an exact reviewed-contract hash allowlisted by the deployment runtime.

No current artifact satisfies those gates. Full R4 still needs the causal
detector/tracker, `actor-vision-v1` adapter, golden corpus,
spawn/difficulty report, and controlled `vision-matcher-v1` transfer report.
