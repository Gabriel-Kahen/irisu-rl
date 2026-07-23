# R4a original-game harness foundation

R4a adds the asset-free, fail-closed core for original-game capture, coordinate
calibration, legal targeted input, causal timing estimation, and sustained-soak
reporting. It does not claim that the live calibration gate has passed. No
original-game frames, cursor/click constants, or latency measurements were
committed with this change.

The checked deployment contract remains
`measurement_status = "provisional_unmeasured"` and
`live_deployment_enabled = false`. In particular, the installed same-session
targeted-click operation observed during development is atomic: it does not
expose independently timestamped button-down and button-up operations. R4a
refuses to reinterpret that operation as a measured press/release macro and
does not add a PyAutoGUI, raw `xdotool`, focus-stealing, or global-input
fallback.

## Runtime boundary

`irisu_rl.original_game` contains provider-independent runtime primitives:

- exact window/capture identity and opaque claimed-window leases;
- completed-frame metadata, duplicate/drop/stale/out-of-order classification,
  and bounded-ring overflow accounting;
- image/client/window-local affine calibration with residual and drift gates;
- a causal gameplay cadence/phase posterior that never exposes an exact native
  tick;
- targeted weak/strong input with pending-depth one, rate/cursor/coordinate
  limits, explicit down/up, and release-all cleanup;
- watchdog state and proposed-versus-executed action records;
- tamper-evident aggregate soak reports and an evidence-backed contract
  finalizer.

The live provider port is deliberately strict. It must expose current native
input safety, exact background capture, exact claim/renew/release, and targeted
button down/up/release-all for the same claimed identity. It must also enforce
an absolute automatic-release deadline independently of the caller and
neutralize held buttons on claim release or expiry. An atomic-click-only
provider—or an explicit-edge provider without those broker guarantees—is
capture-capable but input-ineligible. Every provider exception enters cleanup;
an attempted release-all and claim release are required even when injection or
renewal fails.

Captured pixels stay in ignored `reference/captures/<experiment-id>/frames/`.
Committed reports may contain opaque experiment IDs, hashes, aggregate counts,
quantiles, and uncertainty, but no raw frames, claim tokens, personal paths, or
window titles.

## Causal clocks and geometry

Capture request, start, completion, injection, acknowledgment, inferred
game-poll/effect, and first-visible confirmation all use one injected monotonic
clock. The estimator updates only from frames already completed. Duplicate
compositor frames do not masquerade as gameplay ticks, and out-of-order frames,
long stalls, or capture restarts invalidate confidence until sufficient new
causal evidence exists.

The effect posterior and visible confirmation remain separate:

```text
request -> injection/ack -> latent poll/effect -> first visible confirmation
```

Changing only post-effect render/capture delay cannot move the already-issued
poll/effect estimate. Geometry follows image pixels to the 640×480 game client
and then to window-local input coordinates. A calibration is bound to exact
window/capture identity and geometry, has a maximum age, and fails closed on
move, scale, crop, anchor, or residual drift.

Normalized policy coordinates are lowered once, before simulator or live
execution, with the deployment contract's half-open integer-pixel rule:
`floor(normalized * extent)`, clamped to `[0, extent - 1]`. The continuous
sample remains the PPO likelihood value; quantization is an execution
transform, so `1.0` targets `(639, 479)` rather than the out-of-client
`(640, 480)`.

## Soak and contract promotion

The soak reporter consumes safe JSONL events whose records form a SHA-256
chain. Its thresholds are preregistered before a run, and every sealed event
commits to the exact canonical threshold-config hash. Metric tail directions
are fixed by schema, and every metric in every declared experiment must span
the soak within a declared maximum measurement gap. It reports counts and
p50/p95/p99/worst tails for capture cadence/jitter, ring age, request-to-ack,
inferred poll/effect, effect-to-visible, total latency, deadline misses,
confirmation failures, crop drift, button-release failures, cross-window
misroutes, and resource growth. Missing required samples produce
`not_evaluable`; threshold violations produce `fail`. Passing requires zero
cross-window misroutes and zero release failures.

After a complete private measurement bundle and passing sustained soak exist,
create a new measured contract instead of editing the checked provisional file
in place:

```bash
PYTHONPATH=python uv run --locked --extra training \
  python tools/finalize-r4a-contract.py \
  configs/rl/actions/deployment-v1.toml \
  /private/path/r4a-deployment-measurements.json \
  /private/path/r4a-soak-report.json \
  /private/path/r4a-safe-events.jsonl \
  /private/path/r4a-soak-thresholds.json \
  /private/path/deployment-v1.measured.toml
```

The finalizer rebuilds the report from the raw safe-event chain and threshold
config, then requires game/runtime/tool hashes, opaque hardware/runtime
provenance, experiment and artifact IDs, positive sample counts, uncertainty,
monotone p50/p95/p99/worst values, a two-dimensional click sweep, a frozen
cursor-fairness choice, and a passing soak hash. It publishes with no replace as
`measured_pending_review`, with live deployment still disabled. Only a separate
human review may remove that blocker and replace the repository contract in a
later change.

R4b supersedes the aggregate-measurement input shown above for qualifying live
work. Its authoritative finalizer takes a frozen plan plus the typed raw
calibration journal, deterministically rebuilds every empirical statistic, and
then invokes the same review-blocked contract transition:

```bash
mkdir -m 700 /private/r4b-final
PYTHONPATH=python python tools/finalize-r4b-contract.py \
  configs/rl/actions/deployment-v1.toml \
  /private/path/r4b-calibration-plan.json \
  /private/path/r4b-calibration.jsonl \
  /private/path/r4a-soak-report.json \
  /private/path/r4a-safe-events.jsonl \
  /private/path/r4a-soak-thresholds.json \
  /private/r4b-final
```

The older R4a command remains useful for foundation/schema regression fixtures;
it is not the live promotion path.

## Validation and remaining gate

Deterministic tests use recorded synthetic frame hashes and injected fake
providers. They cover clean 50 Hz presentation, faster duplicate presentation,
drops, delay, out-of-order delivery, restart/stall recovery, phase boundaries,
crop/affine round trips, buffer overflow, forced failures at each input stage,
button/claim cleanup, rate and bounds limits, and a long zero-misroute run.

R4a remains provisional until a current safe provider supports explicit
targeted down/up plus broker release deadlines and claim-end neutralization, a
fresh claimed-window capture/input smoke passes, the 2-D
projectile-birth sweep and click/cadence/latency measurements are complete, and
a production-cadence soak longer than the evaluation episode envelope passes
its preregistered thresholds. Full R4 additionally requires the detector,
tracker, visual adapter, controlled golden corpus, spawn/difficulty report, and
controlled matcher transfer; those are outside this foundation.
