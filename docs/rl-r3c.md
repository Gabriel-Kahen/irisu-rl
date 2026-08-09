# R3c: relational search and distillation

R3c is an exploratory successor to the completed R3b gauge-shaping experiment.
It is deliberately isolated from the canonical R3 package, workflow, configs,
and artifacts. Nothing produced by R3c is R3 calibration, validation, test, or
deployment evidence.

## Implemented development system

The mechanics-targeting slice and the Phase-2 policy-improvement loop now live
in `python/irisu_pointer/`. The implementation includes:

- a permutation-equivariant recurrent entity-pointer actor-critic;
- actor-compatible pair relations for color, relative geometry, gaps,
  overlaps, and distance;
- gated same-color priors for both action kind and target, with bounded learned
  residuals that may override those priors;
- a compact target-relative action vocabulary;
- a spawn-censored tactical teacher plus a bounded depth-3 macro beam that
  searches short shot/wait sequences, values public chain structure, restores
  every branch, and conservatively falls back when no plan is positive;
- identity-bound trajectory records with public delayed-reward decomposition,
  duration-aware returns, 51 quantile targets, episode-disjoint splitting,
  recurrent windows, and hash-verified serialization;
- recurrent padded-sequence training with explicit reset masks, burn-in,
  detached truncated backpropagation through time (TBPTT), branch-conditional
  policy losses, bounded weighted-mass kind balancing, per-step policy weights,
  entropy regularization, and a quantile-Huber critic;
- real-environment DAgger collection on policy-visited states, a decreasing
  teacher-behavior schedule, periodic/uncertainty/failure search queries,
  stronger weights for expensive search corrections, deterministic stratified
  minibatches that cover both teacher and behavior pools from early through
  late replay, prioritized elite/failure replay, and advantage-weighted
  behavior replay;
- recurrent identity-safe inference and atomic, hash-bound model checkpoints;
- a fail-closed teacher-to-`actor-vision-v1` alignment path that removes
  privileged fields, uniquely aligns targets by visible geometry/category,
  applies deterministic bounded track noise, trains a separate recurrent
  student, and lowers selected track geometry to legal press/release actions
  without reconstructing IDs; and
- a fail-closed portable full-episode evaluator that fixes an unsealed
  development seed suite, binds policy/runtime/runner/schema/action identities,
  rejects sealed or test paths and labels, re-verifies files after evaluation,
  and reports score, duration, hits, chains, clears, rot, and promotion gates.

`benchmarks/rl_r3c_phase2.py` connects these pieces into a development-only
run: hash and re-verify every Phase-2 source file, collect DAgger waves from
the real portable environment, train recurrent teacher and noisy actor
minibatches, enable the history-conditioned lower-quantile leaf evaluator,
save and reload both identity-bound checkpoints, then compare the teacher,
actor, and paced matcher on the same fixed unsealed seeds.
`configs/rl/experiments/r3c-policy-iteration-v1.toml` records the intended
larger campaign and the six-figure promotion thresholds.

`id_scaled` is forcibly zeroed inside the model and again when collected
sequences are constructed. Simulator body IDs bind a teacher decision to an
encoded row and audit the resulting collision, but cannot influence a model
output. Non-piece entities remain visible to attention while target masks
exclude them. With no legal piece, shot logits are suppressed and `WAIT` is
the only possible argmax.

## Failure being addressed

R3b usually converged to a no-input policy. Its fixed-size mean/max body pool
discarded target identity, its coordinate distribution could not represent
several separated bodies, its 100-way wait head made passive behavior easy,
and its single shallow snapshot stage supplied almost no chain or late-game
experience.

IriSu is a relational control problem. A useful decision names a body, a
same-color partner or hazard intent, a collision offset, a shot strength, and a
time. R3c represents that structure directly.

## Method

The implemented Phase-2 loop is:

1. Encode bodies as entities and exchange information through masked
   self-attention.
2. Select `WAIT`, `FIRE_WEAK`, or `FIRE_STRONG`.
3. For a shot, point to one live body and select one target-relative shot
   template. For a wait, select from a compact measured duration set.
4. Seed a compact macro beam with direct-matcher, matcher, side-ejector,
   low-gauge hazard, wait, and target-relative shot candidates.
5. Search up to three decisions and 512 branches from cloned simulator states,
   but stop strictly before the next random spawn. Select on raw score and
   public group/chain, gauge, clear, rot, ejection, and destructive-hit terms.
   A nonpositive plan cannot override the paced matcher.
6. Execute a teacher/policy mixture in the real environment, retain both the
   teacher supervision and logged behavior, and compute delayed,
   duration-discounted returns from public transitions.
7. Replay teacher corrections and advantage-weighted behavior as recurrent
   episode minibatches with burn-in and TBPTT.
8. Train the 51-quantile critic, then use the lower quarter of its predicted
   quantiles as a clipped continuation value at later search leaves.
9. Distill the same labels through bounded noisy causal tracks into a separate
   actor model, save both identity-bound checkpoints, and evaluate whole
   configured episodes against the paced matcher on identical unsealed seeds.

Curriculum restarts, long-game raw-score optimization, belief branching across
future spawns, real tracker integration, and exact-backend confirmation remain
the intended next layers; they are not implied by the implemented short
development benchmark.

Both actionable-kind selection and matcher targeting mix a known-mechanics
prior with a learned residual. Branch-specific gates keep the priors active for
ordinary matching, but can turn them off when search labels favor ejection or
hazard control. The residuals are bounded, preventing a small dataset from
memorizing body-row quirks strongly enough to erase correct relational priors.

The structured action is:

```text
wait({1,2,4,8,16})
fire(
  strength={weak,strong},
  target=body_pointer,
  x=target_x + x_radius_offset * target_radius,
  y=target_y + y_radius_offset * target_radius
)
```

This action remains an ordinary legal mouse action after decoding. Target IDs,
snapshot identities, state hashes, RNG state, and future spawns are never model
inputs.

## Spawn-censored search

At public tick `t` with public spawn interval `I`, the next cadence boundary is
known without RNG access. Candidate simulation may advance only through the
prefix strictly before that boundary. A state at the boundary receives no
physics lookahead. Leaves use a history-conditioned conservative critic, while
nonpositive plans fall back to the paced matcher.

This teacher is still local to one spawn window, but it can compare short
multi-shot sequences rather than only one action. Its public chain potential
approximates the score available when a currently visible group lands, and it
avoids directly targeting confirmed grouped members whose second hit deletes
them without score. Longer planning requires a separate belief-branch API that
replaces the real future RNG with the same externally sampled future set for
every candidate.

The utility counts unique projectile/body pairs. The simulator emits
`projectile_hit` on every sustained-contact tick, so raw multiplicity would
reward a projectile for getting wedged against one body. Gauge change is
normalized by `gauge_max`.

## Planned curriculum

The larger campaign is intended to proceed through:

1. single-body targeting;
2. redirection and ejection;
3. one same-color pair;
4. fresh-to-rotten clearing;
5. chain extension and confirmed-chain restraint;
6. bonus-orb activation and targeting;
7. gauge emergencies;
8. increasing colors, cadence, and level;
9. complete games;
10. failure-state restarts.

Each stage must use reachable states from the production mechanics core. Stage
potentials may label isolated skills for the teacher; final fine-tuning and
policy selection must use only real score. The Phase-2 development runs below
sampled short normal-game episodes and did not yet implement this full staged
curriculum or failure-state restart distribution.

## Required behavioral gates

Before complete-game optimization:

- held-out one-body hit rate at least 90%;
- same-color target accuracy at least 90%;
- causal match/chain success at least 80% of the search teacher;
- actionable-state non-wait recall at least 80%;
- wait-only episode rate at most 20%;
- no-action trajectory equality at most 25%;
- no invalid actions or hidden-future-RNG access.

Full-game development reports action kinds, wait durations, projectile hits,
chain joins, clears, rot, level, highest chain, gauge, score, and paired
performance against the no-action and scripted anchors. Its horizon must be
long enough to measure the six-figure objective; the former 8,192-tick R3b
bound is retained only for historical comparison.

## Development evidence

### Phase 1: mechanics targeting

Run:

```bash
PYTHONPATH=python uv run --extra training \
  python benchmarks/rl_r3c_pointer.py \
  > /tmp/rl-r3c-pointer-final-20260728.json
```

The run used the trusted portable artifact, 48 episodes per challenge,
balanced wait/weak/strong classes, episode-disjoint splits, and no
model-visible IDs. It completed in 78.47 seconds. The JSON SHA-256 is
`8fe4a16b72d71f01adedbe90b6907d06d21b0136f0ad1718cb6c8b060ad14d56`.

| Held-out gate | Controlled pair | Crowded multi-pair |
|---|---:|---:|
| kind accuracy | 94.44% | 98.61% |
| exact target accuracy | 100% | 100% |
| template accuracy | 100% | 100% |
| actionable recall | 100% | 100% |
| selected-target hit rate | 91.67% | 87.50% |
| invalid actions | 0 | 0 |

The benchmark preserves and hash-verifies the failed iterations. The original
shortcut had 0% held-out target accuracy; balanced data reached 37.5%; generic
pair attention reached 39.6%; and an unbounded residual later overrode the
correct prior. Those failures motivated ID-free geometric labels, stratified
search candidates, explicit relations, and gated bounded residuals. No
threshold or crowded-board test was relaxed.

### Phase 2: recurrent policy improvement through v8

The Phase-2 benchmark was exercised repeatedly against the trusted portable
library. These are tiny development probes, not comparable to a full campaign:
v1-v3 used only two evaluation seeds, while v5-v6 used four seeds and a
2,000-tick episode cap. A configured truncation is counted as an episode
completion by the evaluator.

| Run | Material change | Last-wave sequence evidence | Learned and paced-matcher scores |
|---|---|---|---|
| v1 | Initial recurrent DAgger path | 11 actionable labels among 288 examples; target accuracy 90.91% on those labels | `[0, 0]` and `[0, 0]` |
| v2 | Bounded inverse-frequency kind balancing | actionable recall 16.58%; target accuracy 97.93% | `[0, 0]` and `[0, 0]` |
| v3 | Added gated weak/strong matcher-kind prior | actionable recall 94.44%; target accuracy 96.67%; policy-driven training wave reached 24 points | `[0, 0]` and `[0, 0]` |
| v4 | First richer recurrent/AWR attempt | intentionally stopped after resident memory grew to about 17.4 GB; no scientific result artifact | not run |
| v5 | Chunked recurrent evaluation, deterministic four-episode minibatches, AWR, replay, and critic leaf; early reduced search mode | three waves held actionable recall at 95.67–97.52% and target accuracy at 96.59–97.87% | `[0, 0, 40, 16]` and `[0, 0, 40, 16]` |
| v6 | Restored the full stratified set of up to 64 search candidates | actionable recall rose from 80.00% to 92.31%; target accuracy rose from 96.00% to 98.46% | `[0, 0, 40, 16]` and `[0, 0, 40, 16]` |
| v7 | Gave expensive full-search corrections 8x policy weight | wait recall collapsed to 0%; predicted-actionable rate was 100%; the policy fired continuously | `[0, 0, 0, 0]` and `[0, 0, 40, 16]` |
| v8 | Balanced kind loss by weighted class mass, so search priority acts within rather than across wait/shot classes | wait recall 100%; actionable recall 95.88%; target accuracy 95.88%; the final policy-driven training wave averaged 10.67 points | `[0, 0, 40, 16]` and `[0, 0, 40, 16]` |

The v4 failure was operationally useful: evaluation had forwarded an entire
collected sequence while retaining recurrent activations. Evaluation now uses
detached TBPTT chunks, and training replay uses bounded deterministic episode
minibatches. The same richer workload completed in v5 below roughly 1 GB
resident memory.

The v3 checkpoint was also evaluated on all eight fixed development seeds for
2,000 ticks. Learned and paced-matcher scores were exactly
`[0, 0, 40, 16, 0, 0, 0, 16]`; the median was zero. Its report SHA-256 is
`e76cc2174a21db0ef83e109f10b1de7165d64cc67443c4b2dcf7dffc202700e6`.

The reproducible Phase-2 artifact identities through v8 are:

| Run | Report SHA-256 | Checkpoint SHA-256 |
|---|---|---|
| v1 | `1e1e0a05cf17cf47d53b02009c6c7aa93b38bbef145cd264239f71281b985c95` | `d2560e10d041e75917e3b0d4b70917a0a190993d26339bd396462a871df49700` |
| v2 | `cec7daace3fc6eec722cfb07f6ea61dd62da2d2495575b64dc7460273e8f81dd` | `7a75ee9c7b5fd4f823dfa68c10f58c1f044c778254011e58c0cc2b0182422147` |
| v3 | `6cf99d57d6aa594bfc0e326d81e31a434610990ceefe1db31590f1484463e90a` | `1bea7f248e9549469ef3c57baf0ba32cd5bbdec0806678e13e4c438f61ed4427` |
| v5 | `7e1cee4d6874ba7c86098505ef16ac353aa3fb3c42b497901c79ff13802e81ef` | `e1ca412fd556dfa10294e1b3f3acc05d7bd02b748ae50da9dcfefce591cac8ee` |
| v6 | `6785ced42b1ac892a0241c29fd40b75c84bceb131854bc4540b6187f9f2826f7` | `910d468aa33e7cd0d4d9b198ee1eea0dfef5e1e1b86c35a9208e3b5c24377eb8` |
| v7 | `24455c15734ecf21b722976344069d48a522a4783e859a95de56e2329c986d9e` | `26f62eebf8320678b1d2b68d78a1ad6af94c5a72a560164a9cbc3895da1d076a` |
| v8 | `a819b80e0c5065bc6471cea686b437b41623bff5f0bab2cde50f76025e0f013b` | `ca782f6a6b86b57ee6d223edb2bd4e6aa3a4072b8dfaa979940f9ac4d2c34ccd` |

Full search did produce genuine corrections instead of merely repeating
the matcher. Across its two waves it selected censored fallbacks, hazard
control, and varied weak/strong body-template actions; the policy-driven wave
disagreed with teacher corrections. The first 8x correction-weight run exposed
an interaction bug: unweighted inverse-frequency class balancing was followed
by per-example search weighting, so rare search shots dominated every wait
example. v7 consequently made 22,207 projectile-hit events but no chain join
or clear, scoring zero on all four seeds. Its median delta from the matcher was
-8.

v8 computed inverse-frequency balancing from weighted class mass. This kept
the expensive correction priority within each action-kind class without
changing the total wait/weak/strong mass. The three collection waves had mean
scores 0, 29.33, and 10.67; the latter two were policy-driven mixtures. The
reloaded greedy policy again exactly matched the paced matcher on all four
development seeds: median 8, maximum 40, and median delta zero. v8 completed
in 770.79 seconds. Every six-figure promotion gate remained closed.

### Post-v8 integration verification

The current implementation adds source-tree binding, stratified teacher/AWR
replay, recurrent critic context, the macro beam, and a separately trained
actor checkpoint. A short end-to-end smoke exercised a real macro query and
all three evaluation paths:

```bash
PYTHONPATH=python uv run --extra training \
  python benchmarks/rl_r3c_phase2.py \
  --iterations 1 --episodes-per-iteration 1 \
  --training-episode-ticks 160 --evaluation-episode-ticks 100 \
  --evaluation-seeds 1 --sequence-updates 1 \
  --maximum-search-queries 2 \
  --policy-out /tmp/r3c-phase2-dev-policy-smoke-v11-20260728.pt
```

The run completed in 1.91 seconds with zero invalid actions for the teacher,
actor, and matcher. Its short horizon is explicitly ineligible for promotion
and is not score evidence. The 17-file source identity is
`7632efcb846995dc9e225d8f8bfa71821ee8be31ce74dd84b841b5be21a95932`.
Artifact identities are:

- report:
  `afc5eb13fc435b1fbbe9912b92f29ba1c9d5b1ee53461e4159980d550a280d81`;
- teacher checkpoint:
  `c87f50de787dfc67c68a941d2352da34ed0faa7b27aa24655af4b784d39ab2ed`;
- actor checkpoint:
  `4774a032c9bbf929a104e801bb7aadb70e23ad7fae37afa41540b9e12dee7d88`.

An initial three-seed, 600-tick strict macro probe showed why a conservative
gate is necessary: 64 safe ticks did not expose delayed multi-spawn payoff,
and periodic tie-winning waits changed scores from `[16, 0, 0]` for the matcher
to `[0, 0, 0]`. Its report hash is
`37bbdc04b3aee6c3bd385c72cb1cc801c4ded7e984fd1233bf0617b71b3ccb7d`.
After requiring a strictly positive public leaf score, the previously harmed
seed retained the matcher result exactly: score 16 and chain 2 at tick 600.
Twelve of thirteen queries declined to override the baseline. The conservative
probe hash is
`4ad0dff4dbf8cae9df4aa996ef0eeca7aa5d0503b6b535c7df892a46a4a491f8`.

A separate diagnostic that was allowed to cross later spawn windows found a
chain-3, score-54 action sequence where the matcher reached chain 2 and score
16. That is motivation for future belief branching, not evidence for the
strict no-future-RNG macro teacher.

The final focused regression sweep passed 113 tests, including real portable
environment smokes, macro boundary/restoration checks, recurrent sequence and
critic tests, actor ID-independence/lowering, and evaluator identity gates.

## What this does and does not establish

The implementation now solves the engineering failures that made the earlier
program structurally unable to learn: it retains target identity
relationally, carries recurrent state, trains delayed outcomes, visits its own
states, asks an expensive teacher for corrections, replays high-value and
failure episodes, and evaluates a reloaded policy under bound identities. It
also fixed the concrete all-wait collapse in sequence classification:
actionable recall moved from 16.58% in v2 to above 92% in the later probes.

It has not established strategic score learning. The strongest completed
Phase-2 evaluations exactly reproduced the low-scoring matcher. The current
macro teacher considers up to three structured decisions, but only within one
64-tick spawn-censored window; it cannot reason across a random spawn. Most
labels still come from the fallback matcher, the collected corpus is tiny and
concentrated on early normal-game states, and the critic has not been
calibrated on long-game returns. The current benchmark trains and reloads a
separate `actor-vision-v1` student and lowers selected track geometry without
reconstructing IDs. Its development evaluator still uses a perfect-detector
track adapter; the real vision tracker is not connected. There has been no
trained-policy confirmation on the exact backend, no long-horizon six-figure
development run, and no sealed test access.

## Next work

The next iteration should improve the teacher and state distribution before
spending substantially more compute on the same tactical target:

1. add belief branching that gives every candidate the same externally sampled
   future-spawn set, allowing fair multi-spawn and multi-shot comparisons
   without reading the real hidden future RNG;
2. collect a much larger episode-disjoint curriculum with chain-building,
   restraint, rot, bonus-orb, gauge emergency, elite restart, and failure
   restart states, while retaining expensive corrections at higher weight;
3. extend macro search across several externally sampled spawn futures and
   rank complete action sequences by eventual groups, clears, survival, and
   raw score rather than only one-window public potential;
4. fit and calibrate the recurrent quantile critic on those longer outcomes,
   then verify that leaf values improve held-out search decisions rather than
   merely changing them;
5. connect the existing actor-schema inference adapter to the real tracker and
   measure teacher-to-actor degradation under recorded perception noise; and
6. run long portable development suites against matcher and no-input anchors,
   promote only on the raw-score gates, then confirm a frozen artifact on the
   exact backend. The sealed test phase remains out of scope until those gates
   pass.
