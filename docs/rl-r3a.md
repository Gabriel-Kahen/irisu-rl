# R3a multi-step collector and curriculum foundation

R3a connects the R2 recurrent PPO kernel to the real semantic-action vector
adapter. It is a correctness and orchestration milestone, not a full-game
learning result and not a transfer result. Every model produced through this
path remains `deployable=false` and uses privileged `teacher-v1` simulator
observations until the R4/R5 perception gates replace them.

## What is implemented

### Production recurrent collection

`irisu_rl.collector.RecurrentCollector` owns the policy-boundary state:

- the current adapter observation;
- the recurrent state immediately before that observation is consumed;
- a lane-local `reset_before` mask;
- a dedicated, checkpointable policy-sampling RNG;
- cumulative decision-row, transition, and simulated-tick counters.

Each collection row performs one inference over all lanes, samples the complete
conditional `WAIT`/weak/strong action, executes the existing legal press/release
macro, records the old likelihood components, and appends to
`RecurrentRolloutBuffer`. A rollout can end at a configured simulated-tick
target or decision ceiling. It never drops part of a synchronous row: reported
tick overshoot is therefore bounded by the final row.

If a legitimate rollout is entirely censored (for example, every lane is
truncated during a held shot), the session records a clean skipped update. It
advances environment/recurrent state but does not fabricate PPO statistics or
advance optimizer/curriculum clocks. Every activation-phase rollout is likewise
drain-only, so mixed old/new-stage data never reaches PPO. Consecutive skips are
bounded, counted, and checkpointed; benchmark update targets count completed
optimizer updates rather than collection attempts. CUDA actions cross to CPU
in one batched copy per action tensor before semantic decoding, avoiding
per-lane device synchronization.

Rollout boundaries do not clear recurrent state. Episode boundaries clear it
exactly when the next reset observation is consumed through `reset_before`.

### Correct SMDP bootstrap semantics

The collector evaluates every bootstrap from
`MacroTransition.transition_next_observation` with the outgoing recurrent state.
It never bootstraps from `adapter.current_observation`, because autoreset may
already have replaced that lane with the next episode's observation.

Consequently:

- live transitions bootstrap from their next decision observation;
- neutral `WAIT` truncations bootstrap from the retained final observation;
- true terminals do not bootstrap;
- interrupted held-shot truncations are retained for audit but excluded from
  policy and value losses.

Live-lane bootstrap values are reused from the next policy pass. Shadow
inference is limited to retained truncation observations and live lanes in the
final rollout row, batched across the affected subset. It does not commit the
returned hidden state because the next policy decision must consume that same
observation once through the main recurrent path.

### Reward separation and shaping schedules

`irisu_rl.rewards` keeps five tensors separate:

1. authoritative raw `int64` score delta;
2. scaled raw `float32` reward;
3. task shaping component;
4. integer parts-per-million shaping coefficient;
5. final optimizer reward.

The composer validates the arithmetic on every batch. A zero shaping
coefficient skips the shaping callback entirely and produces a bit-exact copy
of scaled raw score. `RewardSchedule` uses monotone integer knots, and curriculum
weights are frozen per lane until that lane begins a new episode. An optimizer
update can therefore never change the objective halfway through an episode.

### Deterministic curriculum state

`irisu_rl.curriculum` separates three concerns:

- immutable snapshot/trace recipes and their hashes;
- immutable ordered stage definitions and reward schedules;
- mutable coordinator progress.

A snapshot recipe binds its split, scenario family, fixed environment pool,
mechanics hash, reset seed, deployment action schema, serialized legal action
trace, expected tick/score/state hash, snapshot blob hash, and runtime identity.
The recipe is authoritative; snapshot bytes are a verified cache. The library
rejects train/validation overlap in scenario-family labels, legal construction
provenance, and reachable state identity.

Episode assignments use a SHA-256 counter keyed by curriculum identity,
learner seed, lane, and lane-local episode ordinal. Reservations are
transactional, so failed initialization does not consume an assignment. Lane
completion order and batching do not share a mutable RNG stream.

R3a originally stopped at recipe reservations. R3b now supplies the production
initializer: it restores verified snapshot bytes into exactly the completed
lane subset and commits the corresponding assignment only after restore,
identity checks, observation encoding, gauge validation, and reward processing
all succeed. This is an R3b extension, not retroactive evidence for historical
R3a runs.

Promotion requires declared validation counts and consecutive passes. Every
unlocked prior stage is checked against its regression floor. Failure enters
remediation without reducing the highest unlocked stage or rewinding shaping.
After promotion, an activation barrier lets every lane finish its current
episode before the new stage budget and shaping clock begin; this preserves
episode-stable objectives without spending the new stage budget on old-stage
data.
Validation can be recorded only against a content-derived pending request that
binds the policy checkpoint, completed update, evaluator/runtime identity,
exact validation recipe IDs, trial counts, and every recipe/repetition outcome;
each outcome carries a policy RNG seed derived only from the immutable
curriculum evaluation seed and recipe coordinates. This keeps trials paired
across policies and prevents policy/evaluator identity from grinding favorable
randomness. The production session hashes the exact loaded model tensors when
issuing a request, freezes training, and rechecks that hash before accepting a
report. One model state cannot satisfy multiple gates under different labels.
Reaching an update budget first enters a training-closed validation phase so
the policy from the last permitted update receives exactly one gate; a failed
final-budget gate becomes terminal budget exhaustion.
All coordinator checkpoints carry a canonical state hash and a hash-chain head
for promotion, remediation, completion, and budget events.

### Exact training-session resume

`R3ATrainingSession` permits durable checkpoints only at a healthy completed
session-transaction boundary, after either an optimizer update or an explicitly
audited skipped rollout. The immutable checkpoint contains:

- model parameters;
- optimizer, learning-rate schedule, and PPO minibatch RNG;
- adapter observations, bookkeeping, seed allocator, native snapshots, and
  state hashes;
- recurrent hidden state and pending reset mask;
- policy-sampling RNG;
- curriculum/reward state;
- Python, NumPy, Torch CPU, and available CUDA RNG streams;
- counters and all runtime/configuration identities supplied by the run
  manifest.

Pending validation is checked against the exact model tensor hash both before
save and immediately after restore. Skip attempt/consecutive counters are also
restored, so restarting cannot reset the bounded-skip safety gate.

Restore is fail closed. Files and identities are checked before mutation. A
fresh backend is disposable-reset, native states are restored and verified,
and ambient RNG streams are restored last. The integration fixtures require a
resumed session to reproduce the entire next rollout and update: sampled
actions, transition audit, simulator state hashes, SMDP GAE, PPO statistics,
parameters, scheduler state, and sampler state.
The session also rejects checkpointing or training after a rollout was
collected directly outside its collect/update transaction.

## Deliberate topology constraint

`PaddedVectorEnv` is constructed with one immutable mechanics configuration.
R3b can inject arbitrary verified snapshots into lane subsets, but one vector
still cannot mix mechanics/configuration identities. The snapshot initializer
therefore requires one homogeneous `environment_pool` and config hash. A run
that needs multiple pools must use separate vectors and an explicit sampler;
silently mixing them remains forbidden.

## Configuration and acceptance

The checked configuration is
`configs/rl/experiments/r3a-multistep-v1.toml`. Its provisional PPO learning
rate remains `1e-4`, selected by R2b for the early task only; it is not promoted
as a universal full-game optimum. R3b must select any changed optimizer settings
on frozen validation recipes and multiple learner seeds.

R3a acceptance requires:

- mixed WAIT/weak/strong production-path collection;
- variable-duration macros and synchronous tick-budget accounting;
- recurrent continuity across rollout boundaries and one reset per episode;
- correct live/terminal/truncation bootstrap behavior;
- raw-score and zero-shaping audits;
- deterministic assignment, promotion, remediation, and resume state;
- exact portable and exact-worker next-rollout-plus-update resume fixtures;
- the full pre-existing test suite remaining green.

R3a does not establish multi-body learning quality, full-game superiority over
scripted policies, causal pixel observations, robustness to measured transfer
noise, or performance in the original game. Those remain R3b through R5 gates.
