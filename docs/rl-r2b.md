# R2b one-body learning proof

R2b is a bounded coordinate-learning proof. It shows that behavioral cloning
can fit legal target coordinates and that one-step PPO fine-tuning can preserve
or improve that policy from simulator outcomes on disjoint scenario families.
It does **not** exercise recurrent credit assignment, variable-duration SMDP
returns, learned action kind/WAIT decisions, or the production rollout-buffer
and GAE path. Those components have separate R2a contract tests; joining them in
a multi-step learning run remains R3 work. This is not a full-game result or
evidence that a policy can yet observe or control the original game.

Every R2b model uses privileged `teacher-v1` simulator state and is stamped
`deployable=false`. The real-game gate remains the R4 causal tracker, capture
timing model, coordinate calibration, and input bridge.

## Task contract

`one-body-direct-hit-v1` is an intentionally noncanonical diagnostic:

- zero initial rotten bodies;
- one initial falling body;
- ordinary spawn height moved to `-200` to suppress interference during the
  two-tick task; this ongoing mechanics simplification is not just reset
  randomization and must never be treated as nominal-game evidence;
- one legal weak press followed by the normal forced release;
- native collision, projectile, event, gauge, score, and physics behavior is
  unchanged.

The action is constrained by the ordinary `deployment-v1` action mask to a weak
shot with normalized `(x, y)` coordinates. A native `PROJECTILE_HIT` event
against the reset body is success. Full structured event ownership is opt-in;
the normal high-throughput adapter copies only event counts, while diagnostic
and resume tests copy press/release payloads before lazy views expire. The
capture mode is persisted in adapter checkpoint v2 and mismatched restores are
rejected before environment mutation.

Height families and seed splits are disjoint:

| Use | Initial heights |
|---|---|
| Train | 60, 80, 100, 120, 140 |
| Calibration | 50, 150 |
| Validation / LR selection | 70, 130 |
| Final test | 90, 110 |

The validation and test heights are interpolation tests, intentionally scoped
to an R2 smoke. Height extrapolation, multiple bodies, delayed effects, and
earlier-stage regression belong to R3.

## Reward boundary

The permanent game reward remains exact raw score delta. Direct hits in this
task score zero, so R2b logs raw reward independently and supplies the explicit
curriculum-only optimizer reward:

```text
aim = exp(-||action_xy - target_xy||² / (2 * 0.20²))
optimizer_reward = 0.75 * projectile_hit + 0.25 * aim
```

The rollout buffer accepts a detached versioned optimizer reward without
overwriting its raw `int64` score-delta audit tensor. This shaping is valid only
for this isolated task and must be exactly absent from nominal/full-game
evaluation.

## Policy and optimization

The coordinate policy is parameterized as a Beta mean plus log concentration.
This makes the two distinct jobs auditable: the mean learns where to click, and
concentration controls exploration width. The earlier independent α/β head
could fit a deterministic BC mean while remaining extremely broad when
sampled; PPO then collected mostly misses. The v2 head removes that failure
mode without changing conditional log-probability semantics.

Behavioral cloning uses 320 reset-generated examples across training heights.
Its two-phase objective first fits coordinate mean, then retains a strong mean
constraint while optimizing conditional likelihood and Beta variance.

PPO starts from a fixed 200-step expert warm start. That is deliberate: R2b
tests one-step on-policy fine-tuning, not whether sparse reward can rediscover
the click-target demonstrator. Each episode starts from zero recurrent state;
the task fixes weak-shot kind and release timing, and constructs a one-step PPO
batch directly. Three model/optimizer seeds compare `1e-4`, `3e-4`, and `6e-4`
under identical 120-update and seed-split budgets. Selection uses median
validation hit rate, then median aim score, then lower learning rate. The exact
test split uses allocator key `20260722` and is opened only after selection.

The checked TOML is the executable experiment source, not parallel
documentation. It supplies model/task/reward/PPO/runtime settings, all allocator
and random seeds, and canonical budgets; the result embeds the parsed config and
its SHA-256. Noncanonical overrides can run as diagnostics but cannot pass the
canonical acceptance predicate.

The selected `1e-4` is provisional and scoped only to this diagnostic. It
triggered KL early stopping on 9, 17, and 9 of 120 updates, versus 72–76 at
`3e-4` and 108–110 at `6e-4`. That supports the conservative choice here; it
does not establish a generally optimal learning rate for later curricula.

## Acceptance evidence

The checked [result artifact](../benchmarks/results/rl-r2b-one-body-2026-07-22.json)
is the direct `--summary` output of the recorded reproduction command. It binds
the clean source commit, dependency lock, runtime, portable library, exact
worker/build, mechanics configs, task, model/action/schema, allocator keys, and
seed identities. Acceptance is recomputed in tests rather than trusting a
stored pass bit. The gates are:

- BC validation hit rate at least 90%;
- every selected-LR model seed hits at least 90% on the exact-backend test;
- selected median test hit rate exceeds the matched random policy by at least
  70 percentage points;
- every raw-score audit reports count/min/max/sum and remains exactly zero;
- no invalid action or nonfinite optimization statistic;
- the exact worker and mapped physics library match the accepted runtime
  identity, including protocol, pointer width, capacity, and backend;
- every selected seed's PPO validation hit rate is at least its paired
  warm-start rate;
- all three accepted policy states are published as immutable, checksummed,
  weights-only checkpoints;
- a checkpoint fixture reproduces the complete next sampled action → native
  transition/events → task batch → PPO update exactly on the supported CPU
  stack.

The recorded run selected `1e-4`. The three selected models hit 126/128,
128/128, and 128/128 exact test episodes; random hit 0/128. BC hit 128/128
validation episodes. Paired PPO gains over the warm starts were 3.125, 3.906,
and 5.469 percentage points. The accepted states and identities are stored in
[`benchmarks/results/rl-r2b-one-body-models`](../benchmarks/results/rl-r2b-one-body-models/),
and tests verify every manifest/state hash before strict model loading.

The integrated resume fixture validates a sampled action,
native transition/events, direct one-step task batch, and PPO update, but it
does not claim that the evidence run used the production multi-step collector.
These are task-specific engineering results, not full-game or sim-to-real
claims.

## What remains

R3 must add a checkpointable multi-step curriculum coordinator, delayed task
outcomes, stage regression, multiple-body tasks, score-only fine-tuning, and
bounded nominal full-game baselines. R4/R5 must replace privileged teacher
features with causal pixel-derived tracks, calibrate the action/timing bridge,
and measure policy retention in the authorized original game before any model
can be promoted for deployment.
