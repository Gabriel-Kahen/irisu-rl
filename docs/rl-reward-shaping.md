# R3b gauge-potential reward shaping

R3b gives gauge loss an immediate optimizer-facing signal while preserving the
actual objective: maximize held-out raw game score. Raw score remains the
authoritative environment reward, checkpoint audit value, and promotion
metric. Gauge shaping is temporary training assistance, not a new game goal.

## Reward contract

For one completed semantic macro transition:

```text
x(s)   = clamp(gauge(s), 0, gauge_max) / gauge_max
Phi(s) = x(s)

F_t = gamma_tick^elapsed_ticks * Phi(s_next) - Phi(s)
r_t = score_delta / reward_scale + alpha_episode * F_t
```

`linear-gauge-potential-v1` fixes `potential_scale=1` and `gamma_tick=1`.
`alpha_episode` is stored as integer parts per million and converted to a
float only at composition time. It is frozen for each episode by the existing
curriculum assignment state. Different lanes may use different coefficients.

The potential reads exact integer `start_gauge`, `end_gauge`, and `gauge_max`
values captured at the transition boundary. It does not read float32 model
features, event-detail strings, confidence estimates, or an autoreset-created
observation. Invalid or drifting gauge metadata fails closed.

True termination sets the next potential to zero. A time-limit or external
truncation retains the outgoing potential and bootstraps the critic from the
same final observation. Values below zero clamp to zero and values above the
configured maximum clamp to one.

At `gamma_tick=1`, an episode with a fixed coefficient telescopes:

```text
sum(F_t) = Phi(terminal) - Phi(initial) = -Phi(initial)
```

For a given initial state this is constant across policies, so it cannot make
survival or health conservation a second permanent objective. It changes the
timing of learning signals and may improve transient value approximation and
credit learning; it does not guarantee faster learning. A negative-only sum
of damage is deliberately excluded because it would not telescope and could
reward stalling, short episodes, or overly conservative play.

## Critic conditioning and actor isolation

Potential shaping shifts the correct value function:

```text
V_shaped(s, alpha) = V_score(s) - alpha * Phi(s)
```

A critic unaware of `alpha` would receive incompatible targets when curriculum
lanes use different coefficients. R3b therefore adds one critic-only input:

```text
critic_condition = shaping_weight_ppm / 1_000_000
```

The rollout stores the exact condition used during collection and PPO reuses it
during policy verification and every optimization epoch. Collection rejects a
reward/model condition-width mismatch. After the recurrent core, the value
path learns a state-dependent coefficient slope and multiplies it by `alpha`.
That interaction can represent the required `-alpha*Phi(s)` shift; a mere
linear bias on `alpha` could not. The coefficient is not a forward input to the
observation encoders, recurrence, or actor heads. Tests require actor logits,
coordinate distributions, and recurrent state to be bit-identical when only
the critic condition changes.

This permits episode-stable coefficient decay and prior-stage lane mixing
without stale critic targets. Changing a lane's coefficient inside an episode
is still prohibited by design. If that ever becomes necessary, shaping must be
redefined as `gamma^k * alpha_next * Phi(s_next) - alpha_current * Phi(s)`.

## Scale selection

Candidate coefficients are `0`, `0.1`, `0.25`, and `0.5`. `0.1` is the
conservative starting hypothesis, not a selected production value. The choice
must be evaluated jointly with `reward_scale`; changing score normalization
changes the relative strength of the dimensionless potential term.

Selection should use paired learner seeds, identical snapshot assignments, and
held-out raw score. Report learning curves for raw score plus diagnostics for
gauge, rot frequency, episode duration, value loss, explained variance, KL,
entropy, and action cadence. Reject a coefficient if it improves gauge
retention but harms final raw score, produces high seed variance, destabilizes
the critic, or learns conservative/stalling behavior.

The checked config preregisters a planned final coefficient of exactly zero and
a planned minimum of 400 score-only fine-tuning updates. Those are experiment
gates for the forthcoming run builder; this PR records them but does not claim
that an empirical sweep or 400-update run has already occurred.

## Game-specific magnitude

Normal mode starts at 3,000 gauge with a 40,000 maximum. Game update order
matters: a clear gain occurs before scene passive drain, while rot subtraction
occurs afterward.

- A level-1 rot endpoint is `3000 -> 1179`, so `F=-1821/40000`. At
  `alpha=0.25`, the optimizer contribution is about `-0.01138125`.
- An initial qualifying clear endpoint is `3000 -> 3699`, so `F=699/40000`.
  At `alpha=0.25`, the contribution is about `+0.00436875`.
- One hundred below-half passive-drain ticks contribute `-100/40000`; at
  `alpha=0.25`, that is only `-0.000625`.

These are endpoint examples, not a recommendation for `alpha=0.25`. Only net
retained gauge is shaped. Gain and loss inside a macro can cancel, which is
required for macro-partition invariance. Typed events may be collected for
diagnosis, but this reward does not require event capture.

Rot subtraction can make gauge nonpositive before game over is latched on a
later tick. The later update may display gauge as one or even apply recovery.
Forcing terminal potential to zero prevents that update-order artifact from
creating a positive terminal rebound.

## Transfer boundary

The deployed actor neither computes rewards nor reads authoritative transition
gauge fields. Those exist only in the training and audit path. The actor uses
the ordinary observation schema, including gauge information that the transfer
pipeline must ultimately derive from the visible HUD. The shaping coefficient
is critic-only and the critic is not deployed.

The planned final causal/noisy training and game evaluation use `alpha=0`, so
the actual game needs no backend reward feed. R4 tracker and input calibration
remain transfer gates; this PR does not claim that those gates are complete.

## Implemented safeguards

- Raw score is stored unchanged; zero coefficient bypasses the shaping callback
  and is exactly score-only.
- Transition and collection audits bind exact gauge boundaries, shaping
  components, coefficient, shaping ID, and reward manifest hash.
- Reward identity is part of task checkpoint restore and the optional runtime
  manifest composition identity.
- Gauge loss, recovery, clamping, terminal handling, truncation bootstrap,
  autoreset exclusion, and macro telescoping have deterministic tests.
- Shifted-critic SMDP GAE matches score-only GAE, including variable macro
  lengths and a bootstrapped truncation that stops the trace.
- Nonzero shaping resumes exactly on both portable and exact physics backends.
- The critic condition must equal the composed reward coefficient and remain
  constant through an episode, including across rollout chunks and resume.
- A mixed-coefficient rollout with an episode reset completes collection,
  policy verification, and a PPO update.
- Shaping manifests are canonicalized, frozen into reward identity, and
  revalidated before environment mutation.
- A reward/model critic-condition disagreement fails before collection.
- The v1 potential scale and discount are fixed; semantic changes require a new
  shaping ID.

The machine-checked decision record is
`configs/rl/rewards/r3b-linear-gauge-potential-v1.toml`.
