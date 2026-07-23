# R3 operations: snapshots, durable trials, and evidence

R3 is a bounded simulator-learning milestone. Its policies consume privileged
`teacher-v1` state and remain `deployable=false` and `transfer_eligible=false`.
R3 can validate the learning, reward, curriculum, and evaluation machinery; it
cannot establish original-game transfer. R4 freezes the measured deployment
action/capture contract, and R5 builds the causal actor.

## What is operational

The operational layer freezes every choice left open by
`r3b-completion-v1.toml`: vector size, worker and Torch concurrency, model
dimensions, collector limits, PPO settings, evaluation repetitions/horizons,
snapshot minimums, checkpoint retention, and the exact backend as the primary
scientific backend. Unknown keys fail validation.

Long runs use two durable stores:

- a private, content-addressed JSON artifact store for immutable metadata and
  verification receipts;
- a SQLite workflow for atomic job claims, checkpoints, phase state, and
  restart decisions.

Artifact publication is exclusive and crash-safe. Files use canonical finite
JSON, content hashes, mode `0600`, parent mode `0700`, no-follow opens, fsync,
and atomic no-replace publication. Symlinks, hard links, path traversal,
duplicate JSON keys, noncanonical encoding, corruption, overwrite, and type or
version drift are rejected.

The workflow creates one progressive 300-update calibration job per arm and
learner seed. Update 100 remains a required diagnostic rung; it is not a
second independent training run. Every arm reaches update 300 and selection
uses only that final rung. The earlier checkpoint measures learning speed
without introducing a post-hoc elimination rule.

The operator command `canonical-run-job` owns one job from claim through
training, exact-resume audit, exact/portable evaluation, artifact publication,
and workflow completion. A private no-follow process lock prevents two local
operators from restoring or evaluating the same job concurrently. Canonical
training may only stop on the frozen 50-update grid; sealed test jobs must train
and evaluate in one uninterrupted command.

Calibration and validation may recover only from the last identity-bound
checkpoint. A stale worker token immediately loses authority. Test jobs use the
separate one-shot sealed ledger: an unstarted lease is recoverable only with
its bearer token, while any crash after `begin` is terminal. The independent
exact-resume audit session verifies the already-running lease without beginning
it twice and cannot create outcome evidence.

## Snapshot bundles

A snapshot source contains only preregistered construction intent:

- split, stage, scenario family, and environment pool;
- reset seed;
- canonical legal semantic-action trace;
- action schema and generator versions.

Generation runs that intent against a measured simulator, records the observed
config/tick/score/state/blob identities, and replay-verifies every result before
atomically publishing a bundle. An invalid primitive, terminal boundary,
unstable clone, identity mismatch, unsafe output path, or partial bundle aborts
publication. Snapshot bytes remain a cache; seed plus legal trace remains the
construction authority.

Portable and exact snapshots pair on backend-neutral construction provenance.
The native state hash is intentionally not part of the logical cell: snapshot
schemas and physics state are backend-specific, so each recipe retains its own
state hash.

## Backend evidence

The exact backend is the primary R3 score/evaluation backend. Exact evaluation
must replay deterministically.

Portable and exact physics can diverge over long horizons even from the same
logical seed and macro trace. R3 therefore does not require their episode
metrics to be byte-identical. It stores per-cell exact-minus-portable deltas for
score, ticks, decisions, gauge, invalid actions, and terminal outcome. Both
backends must cover the same logical cells, use the same policy seeds, start at
the same raw score, and produce zero invalid actions. Divergence is diagnostic
evidence for R4 fidelity work, not a result that may be discarded.

The required scripted baselines are sealed as one batch with the test attempt.
That batch contains the primary exact report, deterministic exact replay, and
portable diagnostic report for every required baseline. It is one-shot:
crashing after start rejects the confirmation attempt, and finalization accepts
only the exact evidence hashes recorded by the ledger.

## Evaluation throughput

Evaluation is partitioned into deterministic `(snapshot, repetition)` shards.
Every shard binds the complete suite, policy, evaluator, runtime, and its exact
cell set. Merge rejects missing, duplicate, overlapping, or foreign cells and
binds the merged execution identity to all shard report hashes. Shards may be
resumed from immutable per-shard packages without changing policy seeds or the
authoritative full-suite result. The checked runner executes missing shards
sequentially; concurrent distributed workers are not part of the R3 contract.

A durable lookup index maps each frozen suite/policy/worker/shard key to its
immutable artifact. The index is only an accelerator: retrieved artifacts are
still content- and semantics-verified. This avoids repeatedly scanning a
growing artifact store at every checkpoint.

Learning-curve AUC uses a preregistered 32-snapshot exact subset at every
50-update checkpoint. That subset therefore supplies LR selection, validation
AUC nomination, and the sealed relative-AUC gate. Final mean, p10, retention,
and portable diagnostics use the complete phase suite: 64 calibration cells
and 512 validation/test cells. Curve and final suite identities are stored
separately and cannot be substituted.

On the local Ryzen 7 3700X reference host, a real exact 16-cell recurrent
evaluation took 73.431 seconds scalar and 62.107 seconds with the 16-lane
vector evaluator (1.18x). A real 50-update exact canonical segment took about
64 seconds and produced 102,810 simulated ticks. These are engineering
measurements, not R3 acceptance results.

The frozen protocol comprises 66,800 optimizer updates and 116,864 bounded
logical episode cells: 82,816 exact and 34,048 portable. Artifact-cache reuse
can make the number of physical executions slightly lower. Extrapolating the
reference measurements gives a best-observed serial floor of roughly five days
before portable evaluation, persistence, and operational overhead. This is a
capacity estimate, not a deadline or an acceptance result: the reference cells
averaged only 1,576 of the allowed 8,192 ticks. At the same measured tick
throughput, the all-cells-at-cap exact upper estimate is about 19 days before
portable work. Rerun the benchmark and use completed validation durations to
reserve each sealed learner-job window separately; pending sealed jobs are safe
between commands. Reserve the one-shot baseline batch as its own uninterrupted
window (about 35 hours of exact work at the tick-cap estimate, plus portable
work and overhead), and retain enough disk for all immutable checkpoints and
reports. The checked runner intentionally serializes jobs and evaluation shards;
launching competing canonical runners on the same host is rejected by the
global run lock.

Evaluation is a preregistered bounded-horizon benchmark. An episode that reaches
the 8,192-tick bound is explicitly recorded as truncated, and
its score at that bound is the fixed-cell final raw score. Natural game over is
recorded separately as termination. The 8,192-decision cap is a nonbinding
safety cap because every accepted semantic decision advances at least one tick.
Reports may not omit, relabel, or extend a truncated cell, so policy comparisons
remain paired under identical simulated-time bounds.

## Commands

Install the training extra, then validate the frozen configuration:

```bash
uv sync --extra training
uv run irisu-r3 config verify
```

Generate and verify portable and exact snapshot bundles using absolute runtime
paths. The source plan is checked in; generated bundles belong under ignored
`artifacts/`.

```bash
install -d -m 700 artifacts artifacts/r3 artifacts/r3/snapshots

uv run irisu-r3 snapshots build \
  --source-config configs/rl/snapshots/r3b-source-plan-v1.toml \
  --backend portable \
  --library /absolute/path/to/libirisu_clone.so \
  --output artifacts/r3/snapshots/portable

uv run irisu-r3 snapshots verify \
  --bundle artifacts/r3/snapshots/portable \
  --backend portable \
  --library /absolute/path/to/libirisu_clone.so

uv run irisu-r3 snapshots build \
  --source-config configs/rl/snapshots/r3b-source-plan-v1.toml \
  --backend exact \
  --worker /absolute/path/to/irisu-exact-worker \
  --output artifacts/r3/snapshots/exact

uv run irisu-r3 snapshots verify \
  --bundle artifacts/r3/snapshots/exact \
  --backend exact \
  --worker /absolute/path/to/irisu-exact-worker
```

After both bundles verify, initialize a smoke or canonical run:

```bash
uv run irisu-r3 experiment init \
  --run-id r3b-smoke-001 \
  --run-class smoke \
  --snapshots artifacts/r3/snapshots/exact \
  --output artifacts/r3/runs/r3b-smoke-001

uv run irisu-r3 experiment status \
  --run artifacts/r3/runs/r3b-smoke-001

uv run irisu-r3 experiment smoke-update \
  --run artifacts/r3/runs/r3b-smoke-001 \
  --worker /absolute/path/to/irisu-exact-worker \
  --max-new-updates 1

uv run irisu-r3 experiment verify \
  --run artifacts/r3/runs/r3b-smoke-001
```

A smoke run may stop after a bounded number of updates and is structurally
ineligible for selection or confirmation. Canonical mode accepts no grid,
seed, budget, suite, or reward overrides.

Initialize a canonical run with both verified bundles, then run one complete
calibration job per command until `status` shows the phase complete:

```bash
uv run irisu-r3 experiment init \
  --run-id r3b-canonical-001 \
  --run-class canonical \
  --snapshots /absolute/path/to/exact-v2 \
  --portable-snapshots /absolute/path/to/portable-v2 \
  --output /absolute/path/to/r3b-canonical-001

uv run irisu-r3 experiment canonical-run-job \
  --run /absolute/path/to/r3b-canonical-001 \
  --phase calibration \
  --worker /absolute/path/to/irisu-exact-worker \
  --library /absolute/path/to/libirisu_clone.so
```

After calibration, prepare validation once. Reuse the returned authorization
artifact for every validation job:

```bash
uv run irisu-r3 experiment prepare-validation \
  --run /absolute/path/to/r3b-canonical-001 \
  --worker /absolute/path/to/irisu-exact-worker \
  --library /absolute/path/to/libirisu_clone.so

uv run irisu-r3 experiment canonical-run-job \
  --run /absolute/path/to/r3b-canonical-001 \
  --phase validation \
  --authorization VALIDATION_AUTHORIZATION_ARTIFACT \
  --worker /absolute/path/to/irisu-exact-worker \
  --library /absolute/path/to/libirisu_clone.so
```

Prepare the sealed phase only after validation completes. The returned value is
the sealed package artifact used by all remaining commands:

```bash
uv run irisu-r3 experiment prepare-test \
  --run /absolute/path/to/r3b-canonical-001 \
  --authorization VALIDATION_AUTHORIZATION_ARTIFACT \
  --worker /absolute/path/to/irisu-exact-worker \
  --library /absolute/path/to/libirisu_clone.so

uv run irisu-r3 experiment canonical-run-job \
  --run /absolute/path/to/r3b-canonical-001 \
  --phase test \
  --authorization SEALED_PACKAGE_ARTIFACT \
  --worker /absolute/path/to/irisu-exact-worker \
  --library /absolute/path/to/libirisu_clone.so

uv run irisu-r3 experiment run-baselines \
  --run /absolute/path/to/r3b-canonical-001 \
  --authorization SEALED_PACKAGE_ARTIFACT \
  --worker /absolute/path/to/irisu-exact-worker \
  --library /absolute/path/to/libirisu_clone.so

uv run irisu-r3 experiment finalize-test \
  --run /absolute/path/to/r3b-canonical-001 \
  --authorization SEALED_PACKAGE_ARTIFACT \
  --baseline-artifact BASELINE_EVIDENCE_ARTIFACT \
  --worker /absolute/path/to/irisu-exact-worker \
  --library /absolute/path/to/libirisu_clone.so
```

If the baseline batch terminally failed, omit `--baseline-artifact`; the sole
finalization records a rejected confirmation. `status` reports every
non-pending job, owner, checkpoint progress, output artifact, and failure.

## Order of execution

1. Build and replay-verify portable and exact snapshot bundles.
2. Run progressive calibration, retaining update-100 and update-300 evidence.
3. Select one learning rate per alpha, then precommit the test suite and
   baseline batch before installing validation jobs.
4. Authorize and run fresh validation.
5. Select at most one shaped candidate.
6. Authorize the sealed test once.
7. Run every candidate/control learner job without a process boundary.
8. Run the one-shot sealed baseline batch.
9. Finalize once from the ledger-recorded artifacts.

No local smoke result, partial run, portable diagnostic, or unsealed baseline
can enter an R3 acceptance decision.
