# R3b completion: reproducible curriculum training and bounded evaluation

R3b turns the R3a collector into a runnable, auditable experiment. It does not
claim a successful training result, transfer to the original game, or readiness
for bulk compute. The checked protocol is intentionally marked
`design_only_no_empirical_results` until real artifacts satisfy every gate.

## Scope and trust boundary

R3b trains a privileged simulator policy. Its actor input is `teacher-v1`, so a
model produced here is always `deployable=false`. Backend score, gauge, native
snapshots, state hashes, and curriculum identities are training/evaluation
signals only. R4 must still build the causal screen tracker, calibrate input and
latency, and show that the deployment observation/action path matches the
contract. R5 must train or distill the causal/noisy actor.

The purpose of R3b is narrower: demonstrate that the learning system can use
the curriculum correctly, choose reward and optimizer settings without
cherry-picking, end with the real score objective, and beat strong scripted
simulator baselines under a bounded preregistered test.

## Transactional snapshot starts

Each `SnapshotRecipe` binds:

- train/validation/test split and scenario family;
- environment pool and exact mechanics/config hashes;
- reset seed and legal serialized semantic-action trace;
- action schema, generator, and runtime identities;
- expected tick, score, native state hash, and snapshot blob hash.

Snapshot bytes are a cache, not provenance. `replay_snapshot_recipe` rebuilds a
cache entry from its seed and legal macro trace and rejects terminal boundaries
or any identity mismatch. `SnapshotBlobStore` owns and eagerly verifies exactly
one blob for every recipe; incomplete, extra, corrupt, symlinked, or non-owned
payloads fail closed.

Assignments use a domain-separated counter keyed by assignment manifest,
learner seed, lane, and lane-local episode ordinal. Reward schedules are
deliberately excluded from that manifest, so alpha and learning-rate arms see
identical starts. No mutable shared RNG makes assignments depend on lane
completion order.

Initialization is a two-phase transaction:

1. reserve assignments without changing curriculum clocks;
2. clone only the affected vector lanes;
3. restore the selected snapshots concurrently;
4. verify pool/config, tick, score, state hash, liveness, gauge, and encoding;
5. expose the new observations while retaining lane backups;
6. compose the terminal transition with the old episode label and alpha;
7. commit the new assignments only after reward processing succeeds.

Any failure restores the affected lanes and cancels the reservation. Failure to
roll back poisons the vector so collection cannot continue from ambiguous
state. Initial reset uses the same deferred commit: an encoder failure cannot
consume the first assignment. Checkpoints bind active snapshot labels and the
initializer manifest before native state mutation.

The vector remains homogeneous by mechanics/config identity. Snapshot subsets
can differ within that pool; mixing pools requires separate vectors.

## Reward and optimizer protocol

The permanent objective is held-out raw score. Candidate shaping coefficients
are `0`, `0.1`, `0.25`, and `0.5`, represented as integer parts per million.
The learning-rate candidates are `3e-5`, `1e-4`, and `3e-4`. Every arm uses the
same conditioned-critic architecture, reward composer, curriculum assignment
stream, model initialization, schedule shape, and tick budget. Only alpha and
the declared initial learning rate may differ.

The conditioned critic receives alpha; the actor and recurrent state do not.
Alpha is frozen for a complete episode. For shaped candidates it remains fixed
through update 599, changes to zero for newly initialized episodes at update
600, and never becomes nonzero again. The score-only control uses the same code
and architecture with alpha zero from update zero.

At the tail boundary, existing shaped episodes are drained without PPO updates
or optimizer-clock advancement. Once every lane is on a zero-alpha episode,
exactly 400 optimizer updates remain. Before each one, the tail controller
requires:

- every recorded coefficient is exactly integer zero;
- each scaled score equals the integer raw score divided by the bound reward
  scale using the collector's float32 arithmetic;
- each optimizer reward exactly reconstructs from raw score, shaping value, and
  coefficient, and zero-weight rows have both zero shaping and score-only
  optimizer reward;
- row totals equal the collection raw and optimizer totals, and the collection
  reward-composer identity matches the controller;
- decision/transition counts and audit widths agree;
- all reward values are finite;
- optimizer and tail clocks are contiguous.

The controller is checkpointable and hash-chained. Resume cannot skip the
drain, reset its count, or turn a shaped collection into tail evidence. The PPO
optimizer state is retained across the boundary; resetting it would confound
the comparison.

R3b advances the adapter checkpoint to v3 and the training-session payload to
v3 because active snapshot labels, the authorized optimizer-update limit, and
tail state are required identities. Older
R3a checkpoints intentionally fail version validation rather than being loaded
without evidence those fields never carried.

## Reproducible trial construction

`R3BRunBuilder` consumes an immutable `TrialJob` and constructs a complete
session from a frozen experiment plan,
one-stage curriculum, verified snapshot store, measured runtime attestation, model
factory, vector factory, and collector/PPO configurations. It rejects adaptive
promotion in the paired hyperparameter sweep because promotion would make arms
see different training distributions.

Runtime identity is measured from the environment returned by the factory
before any restore or step. Portable runs retain the exact shared-library file
descriptor loaded by `ctypes`, hash those loaded bytes and file identity, and
revalidate that descriptor during attestation; replacing the pathname cannot
relabel the executing binary. Exact runs additionally hash the worker
executable, recapture the mapped exact-library bytes and file identity, and
require every vector lane to attest to the same runtime. Caller-supplied backend
labels are not accepted as evidence.

For each learner seed, `TrialSeedPlan` derives independent SHA-256 streams for:

- model initialization;
- policy sampling;
- PPO minibatch ordering;
- curriculum assignments;
- session NumPy state;
- evaluation.

The derivation is arm-independent. A job binds phase, grid arm, learner seed,
authorized update budget, sealed status, and (for test jobs) the validation
authorization. The session enforces the job budget while retaining the frozen
1,000-update learning-rate horizon, so calibration cannot silently train to
validation length. A seed-independent runner specification binds the model,
encoder, collector and PPO configuration, lane count, reward scale, snapshot
store, runtime, deterministic Torch settings, Python/NumPy/Torch build data,
every Python module in `irisu_rl` and `irisu_env`, and the checked dependency
files; a builder rejects a model factory that changes it. One runner identity
is carried from calibration through validation and sealed test. Selection
rejects results when any seed or paired arm
disagrees on that specification, initial model, assignment, seed plan,
runner-pairing, or evaluation-suite identities. A trial manifest additionally
binds the job, plan, curriculum, snapshot store, runtime, reward, collector,
PPO, lane count, and pre-transfer status. Checkpoint callers must include this
manifest identity in their runtime identity map.

## Frozen selection design

The canonical plan is
`configs/rl/experiments/r3b-completion-v1.toml`. Unknown keys or changes to its
grid, schedules, seed counts, tail length, baselines, statistics, or failure
rules are rejected.

Calibration runs all 12 alpha/LR arms on three paired learner seeds at bounded
100- and 300-update rungs. It selects one learning rate per alpha by median
tick-aligned raw-score AUC, then final raw score, then the lower learning rate.
Every arm retains a record; missing arms reject the phase and failures remain
explicit but ineligible.

Fresh validation uses eight disjoint learner seeds and full 1,000-update runs.
It compares the four calibrated alpha arms and may nominate at most one shaped
candidate. Nomination requires at least 5% mean raw-score AUC gain over the
score-only control and at least 95% final-mean retention. Validation selects;
it does not confirm. Validation jobs require a typed authorization that carries
the complete retained calibration results and recomputes the one-LR-per-alpha
selection; a standalone selection hash cannot authorize work.

The test suite is committed before validation in a durable SQLite ledger. The
validation authorization carries that commitment. After validation fixes one
candidate, an immediate transaction records the candidate, control, complete
validation-result set, validation suite, precommitted test suite, exact test-job
set, an opaque random receipt, and attempt number. That transaction inserts all
24 jobs as pending. Each job can be leased once; its lease is bound into runner
evidence, and its result or failure becomes terminal in the ledger. Reopening
the ledger resumes the same authorization or active lease. An alternate suite
or candidate, a terminal-job retry, and every second finalized attempt fail
closed. Test runners require an active database-verified lease, not a
reconstructible authorization object. Finalization verifies every persisted
job outcome, recomputes and stores the report, and exposes `verify_finalized`;
a caller-created report is never authoritative. Test uses 12 new paired learner seeds, 512 fixed
evaluation episodes per policy, and a one-sided paired bootstrap with the
learner seed as the resampling unit. A candidate passes only if all lower
confidence bounds satisfy:

- raw-score AUC gain greater than 5%;
- final mean raw-score retention at least 95%, or nonnegative absolute change
  when the control mean is near zero;
- p10 raw-score retention at least 90%, or nonnegative absolute change when
  the control p10 is near zero;
- final raw score strictly above the strongest trivial/scripted baseline;
- all engineering and baseline audits pass.

A failed sealed test rejects the candidate. The runner-up is not tried on the
same test set, and the test set cannot be reused after rejection.

## Evaluation and baselines

Evaluation cells are fixed by suite identity, split snapshot ID, typed logical
recipe provenance, repetition, policy seed, measured runtime identity, assignment
identity, snapshot-library identity, snapshot-store identity, action-schema
identity, and decision/tick bounds.
Raw score is always `final_score - initial_score`, and accumulated environment
reward must equal that delta. Gauge, invalid actions, terminal status, and
elapsed ticks are diagnostics, never selection rewards.

The tick horizon is exact: long waits are clipped to the remaining budget and
a shot macro stops after its press when only one tick remains. An all-masked
action branch is an error, not an implicit WAIT action.

The deployment-style recurrent evaluator uses deterministic semantic argmax,
masked wait choices, coordinate means, recurrent state, and a zero critic-only
alpha condition. Its policy identity includes model weights, tensor schema,
encoder implementation, action schema, and both action masks. Scripted policies
are converted through the same semantic press/release macros. Required
baselines are:

- no-action long wait;
- seeded legal random;
- existing matcher-shot heuristic;
- direct matcher;
- side ejector;
- imminent-rot hazard policy.

All required baselines need at least 512 fixed episodes, zero invalid actions,
deterministic replay, exact raw-score accounting, and portable/exact parity.
Parity uses two backend-specific suites and stores because portable and exact
snapshot schemas differ. Their physical library, store, assignment, recipe,
and snapshot identities may differ. A typed manifest pairs recipes only after
their config, reset seed, legal semantic trace, action schema, expected
tick/score, split, and scenario provenance match; arbitrary logical-cell labels
cannot establish parity. Normalized episode metrics must then match exactly.
Each report's backend identity
comes from the live runtime attestation. Acceptance rejects primary evidence
from any suite other than the sealed test suite.
`one_step_greedy` is optional and cannot substitute for a missing required
baseline.

## Failure and evidence policy

The experiment fails closed. A missing arm rejects its phase. A missing learner
seed rejects that arm. Pre-start and post-start failures remain retained and
rank-ineligible. Nonfinite metrics, interpolated learning curves, identity
mismatches, incomplete baselines, malformed tail audits, or checkpoint drift
cannot be ignored.

Raw-score AUC, final mean, and p10 are recomputed from typed fixed-cell reports
at the exact checkpoint update/tick grid. Each point binds completed updates,
simulated ticks, checkpoint artifact, evaluated policy, and report; reused
reports or checkpoints and relabeled policy reports are rejected. Scalar
aggregates cannot be edited independently. No result is an R3b
acceptance result until the complete calibration, validation, sealed test,
exact-resume, snapshot replay, raw-score, and baseline artifacts exist and the
canonical confirmation function returns `accepted`.
Learner outcomes no longer carry a default boolean engineering pass: they must
bind the completed trial manifest, pairing identity, metrics, evaluation,
checkpoint-resume, exact-backend parity, and (for full-budget runs) completed
score-only-tail artifacts.
Unit and integration tests establish implementation behavior only; they are not
empirical learning evidence.

## Running and next gate

The checked plan can be loaded with:

```python
from irisu_rl.r3b_experiments import load_plan

plan = load_plan("configs/rl/experiments/r3b-completion-v1.toml")
```

The next operational work is to generate the versioned full-game snapshot
library, precommit the test suite with `SealedTestLedger.precommit`, run the
bounded calibration jobs, create validation jobs through
`bind_validation_run`, and inspect every retained failure/diagnostic artifact.
Only after validation freezes one candidate may `authorize_once` emit the exact
sealed job set. Each job then follows `claim_job` and `complete_job` (or
`fail_job`), and `finalize_once` consumes the recorded outcomes and returns the
authoritative report. In parallel, R4 must complete the real-game causal observation
and input calibration path. A successful R3b simulator result does not remove
that transfer gate.
