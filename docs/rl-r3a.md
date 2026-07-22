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
rejects train/validation scenario-family overlap.

Episode assignments use a SHA-256 counter keyed by curriculum identity,
learner seed, lane, and lane-local episode ordinal. Reservations are
transactional, so failed initialization does not consume an assignment. Lane
completion order and batching do not share a mutable RNG stream.

The current adapter does **not** consume those recipe reservations. Its
homogeneous task contract changes only action masks and episode-frozen reward
weights after ordinary seeded autoresets. Consequently, snapshot IDs,
lane-episode ordinals, and `prior_stage_mix_ppm` are library/coordinator
contracts awaiting the transactional initializer; they are not claimed as
executed curriculum evidence in R3a.

Promotion requires declared validation counts and consecutive passes. Every
unlocked prior stage is checked against its regression floor. Failure enters
remediation without reducing the highest unlocked stage or rewinding shaping.
Validation can be recorded only against a content-derived pending request that
binds the policy checkpoint, completed update, evaluator/runtime identity,
exact validation recipe IDs, trial counts, and every recipe/repetition outcome;
each outcome carries the policy RNG seed derived from the request identity and
episode coordinates. One policy checkpoint cannot satisfy multiple gates. A
pending request freezes training. Reaching an update budget first enters a
training-closed validation phase so the policy from the last permitted update
receives exactly one gate; a failed final-budget gate becomes terminal budget
exhaustion.
All coordinator checkpoints carry a canonical state hash and a hash-chain head
for promotion, remediation, completion, and budget events.

### Exact training-session resume

`R3ATrainingSession` permits durable checkpoints only at a healthy boundary
after a complete optimizer update. The immutable checkpoint contains:

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

Restore is fail closed. Files and identities are checked before mutation. A
fresh backend is disposable-reset, native states are restored and verified,
and ambient RNG streams are restored last. The integration fixtures require a
resumed session to reproduce the entire next rollout and update: sampled
actions, transition audit, simulator state hashes, SMDP GAE, PPO statistics,
parameters, scheduler state, and sampler state.
The session also rejects checkpointing or training after a rollout was
collected directly outside its collect/update transaction.

## Deliberate topology constraint

`PaddedVectorEnv` is constructed with one immutable mechanics configuration and
the current `MacroVectorAdapter` autoresets completed lanes by seed. R3a does
not claim that it can inject arbitrary snapshot/config starts into individual
lanes. `CurriculumTaskContract` therefore rejects stages spanning multiple
`environment_pool` values.

The next snapshot-initializer change must add transactional subset restore and
make recipe assignment part of adapter autoreset bookkeeping. Until that gate
lands, use one homogeneous collector pool per mechanics/config hash. This keeps
the implemented training path honest and prevents a curriculum manifest from
claiming starts that the simulator did not actually execute.

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
