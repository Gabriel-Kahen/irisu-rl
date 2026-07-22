# R2b one-body learning proof

R2b is a bounded integration proof for the recurrent training stack. It shows
that scripted behavioral cloning can fit legal target coordinates and that PPO
can preserve/improve that policy using simulator outcomes on disjoint scenario
families. It is **not** a full-game result and is **not** evidence that a policy
can yet observe or control the original game.

Every R2b model uses privileged `teacher-v1` simulator state and is stamped
`deployable=false`. The real-game gate remains the R4 causal tracker, capture
timing model, coordinate calibration, and input bridge.

## Task contract

`one-body-direct-hit-v1` changes only legal reset parameters:

- zero initial rotten bodies;
- one initial falling body;
- ordinary spawn height moved to `-200` so it cannot interfere with the
  two-tick task;
- one legal weak press followed by the normal forced release;
- native collision, projectile, event, gauge, score, and physics behavior is
  unchanged.

The action is constrained by the ordinary `deployment-v1` action mask to a weak
shot with normalized `(x, y)` coordinates. A native `PROJECTILE_HIT` event
against the reset body is success. The adapter now owns every structured press
and release event before the backend's lazy event view can expire.

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
tests the correctness and stability of on-policy improvement, not whether
millions of sparse-reward samples can rediscover the obvious click-target
demonstrator. Three model/optimizer seeds compare `1e-4`, `3e-4`, and `6e-4`
under identical 120-update and seed-split budgets. Selection uses median
validation hit rate, then median aim score, then the lower learning rate. The
test split is opened only after that selection.

## Acceptance evidence

The checked result artifact records source, dependency lock, portable library,
exact worker, mechanics-config, task, action/schema, and seed identities. The
acceptance gates are:

- BC validation hit rate at least 90%;
- every selected-LR model seed hits at least 90% on the exact-backend test;
- selected median test hit rate exceeds the matched random policy by at least
  70 percentage points;
- raw score delta remains independently reported;
- no invalid action or nonfinite optimization statistic;
- a checkpoint fixture reproduces the complete next sampled action → native
  transition/events → task batch → PPO update exactly on the supported CPU
  stack.

The recorded run selected `1e-4`. All three selected models hit 128/128 exact
test episodes; random hit 2/128. BC hit 128/128 validation episodes. These are
task-specific engineering results, not full-game or sim-to-real claims.

## What remains

R3 must add a checkpointable multi-step curriculum coordinator, delayed task
outcomes, stage regression, multiple-body tasks, score-only fine-tuning, and
bounded nominal full-game baselines. R4/R5 must replace privileged teacher
features with causal pixel-derived tracks, calibrate the action/timing bridge,
and measure policy retention in the authorized original game before any model
can be promoted for deployment.
