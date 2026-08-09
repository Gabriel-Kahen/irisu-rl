# R3d: goal-conditioned steering and strategic survival

Status: development-only. R3d is not canonical R3 evidence, does not read
sealed material, and must not be used to prepare or enter the sealed test
phase.

## Why this direction exists

The earlier policy chose a body and an absolute launch row. That representation
discarded the selected body's vertical position, so the learner had to solve
aiming, matching, cadence, and long-horizon survival through one weak label.
Short training horizons also ended near the first useful clear, before score
and gauge consequences could teach a stable strategy.

R3d separates the problem:

1. **Micro-control:** learn whether to act and which directed
   `source -> destination` pair to pursue. Deployment lowers that pair through
   a deterministic continuous strong-shot rule: strike the source from the
   side opposite the destination and from below, with one tick of public
   velocity lead.
2. **Cadence and restraint:** predict whether to shoot or wait and, for waits,
   predict a duration. A hard public-time cooldown still prevents repeated
   shots while a correction resolves. A public progress tracker suppresses a
   directed pair when a shot fails to close at least 0.05
   source/destination sizes and re-enables it after observable closure.
3. **Closed-loop execution:** replan at every safe boundary and never
   direct-hit a grouped, confirmed, rotten, or deleted source. Rotten pieces
   remain legal destinations: native contact ordering lets a fresh source join
   and burst against a same-color rotten destination.
4. **Archive improvement:** restore evidence-bound portable snapshots, compare
   physically distinct steering and deliberate-wait branches, and label the
   first decision of the winning rollout only when it strictly improves on the
   baseline branch. A candidate is eligible only if it does not reduce
   baseline alive state, survival ticks, or final gauge; score and the remaining
   public outcomes rank eligible candidates. Every branch consumes the same
   opaque simulator continuation; future RNG is never a policy input.
5. **Curriculum:** do not optimize six-figure score until control, group
   formation, and survival gates pass on unseen development seeds.

The learner is a directed-pair network. It scores every legal
`source -> destination` pair and predicts strategic intent conditioned on that
pair. Strength and target-relative template heads remain auxiliary training
targets; the frozen v5 deployment deliberately uses analytic continuous
geometry because deploying the earlier template prediction regressed
closed-loop behavior. Separate heads learn act-versus-wait and wait duration.
Public body IDs bind supervision to the current observation but are zeroed
before the model encodes bodies.

## Exact replay supervision

`benchmarks/r3d_replay_supervision.py` replays the trusted v2.03 trace through a
fresh exact worker. It reconstructs held-button edges, including suppression of
fresh edges for the first two records, and binds each `shot_fired` projectile
to its first `projectile_hit`.

The diagnostic fails unless:

- source files, replay bytes, worker bytes, mapped exact library, config, tensor
  schema, and pointer vocabulary have stable identities;
- all 47,019 replay records are consumed;
- the exact result reproduces the recorded score; and
- source/runtime inputs remain unchanged through the run.

The verified 41,449 replay produced:

| Metric | Result |
|---|---:|
| Fired projectiles | 1,037 |
| Projectiles with a public hit | 1,025 (98.84%) |
| Public chain-join events | 500 |
| Invalid actions | 0 |
| Final score | 41,449 |
| First-hit labels with a visible target | 1,020 |
| Strict legal, reachable, direction-consistent examples | 416 |

Destination is the nearest visible same-color peer at shot time and is
explicitly marked as a behavioral inference, not recovered human intent.
Imitation additionally requires legal source/destination lifecycles, an
ungrouped source, contact within 64 ticks, source-to-peer distance at most 200
pixels, and an impact side consistent with pushing toward the inferred peer.
Rejected contacts remain in conversion evidence.

A deterministic temporal holdout is available to test whether pair labels are
learnable without mixing later observations into training. In a short
60-update diagnostic, holdout pair accuracy improved from 10.84% to 71.08% and
loss fell from 8.08 to 3.69. That is evidence that the stricter representation
carries a learnable control signal, not evidence of a high-scoring policy.
The final report is `/tmp/r3d-replay-final-20260728.json`, SHA-256
`3cefd59bcc6b83484143bb5635f9e05b99f95fa4090840736c2e26b4d26a09eb`.

Run the exact conversion:

```bash
PYTHONPATH=python .venv/bin/python \
  benchmarks/r3d_replay_supervision.py
```

Add the temporal learning diagnostic:

```bash
PYTHONPATH=python .venv/bin/python \
  benchmarks/r3d_replay_supervision.py \
  --train-steps 300 --batch-size 64 --torch-threads 4
```

## Development training benchmark

`benchmarks/rl_r3d_steering.py` uses separate fixed seeds for demonstrations
and evaluation. All 16 seeds in `r3d-fixed-unseen-development-v1` are tuning
and development seeds. A second disjoint 16-seed suite,
`r3d-survival-holdout-development-v1`, was consumed once after v5 was frozen.
It is an untouched development holdout, not sealed or promotion evidence. The
benchmark:

- collects public-state decisions from the closed-loop controller;
- records strategic features and evidence-bound archive cells;
- restores archive elites and evaluates bounded steering/wait branches;
- adds only legal winning first decisions to the supervised dataset;
- trains the pair model;
- saves and reloads a fail-closed steering checkpoint; and
- compares the demonstrator, learned imitator, and legacy absolute-row matcher
  on identical unseen seeds.

Every report includes raw score, survival, hit conversion, joins per shot,
`CLEARED` events per shot, qualifying clears per shot, rot, ejection, highest
chain, gauge failures, source/runtime/config identities, archive branch
outcomes, dataset identity, and all curriculum gate failures. `CLEARED` events
and qualifying clears are deliberately separate quantities. The archive
binding covers its source, snapshots, native environment config, runner,
portable runtime, action vocabulary, and observation schema. Fast profiles are
direction-finding runs. The July 29 evidence keeps the fast training budget but
overrides evaluation to 16 episodes and 10,000 ticks; it remains
development-only.

```bash
PYTHONPATH=python .venv/bin/python \
  benchmarks/rl_r3d_steering.py \
  --profile fast \
  --evaluation-seeds 16 \
  --evaluation-ticks 10000 \
  --policy-out artifacts/r3/development/r3d-survival-v5-20260729/long-development.pt \
  --result-out artifacts/r3/development/r3d-survival-v5-20260729/long-development.json
```

The one-shot survival holdout used the frozen source/config/training procedure:

```bash
PYTHONPATH=python .venv/bin/python \
  benchmarks/rl_r3d_steering.py \
  --profile fast \
  --evaluation-suite survival-holdout \
  --evaluation-seeds 16 \
  --evaluation-ticks 10000 \
  --policy-out artifacts/r3/development/r3d-survival-v5-20260729/long-survival-holdout.pt \
  --result-out artifacts/r3/development/r3d-survival-v5-20260729/long-survival-holdout.json
```

## Development evidence

The frozen v5 run used eight 2,000-tick demonstration seeds and 1,236 examples:
826 shot decisions and 410 deliberate waits. It evaluated 24 archive elites
and 216 identity-bound branches, emitting nine strict improvement labels.
Balanced shot/restraint minibatches prevented a wait-only collapse.

| Training metric | Result |
|---|---:|
| Full-dataset loss | 4.623 to 1.654 |
| Balanced act accuracy | 95.03% |
| Shot recall | 91.77% |
| Restraint recall | 98.29% |
| Directed-pair accuracy | 49.52% |
| Auxiliary impact-template accuracy | 70.70% |

Long-horizon results:

| 16 seeds × 10,000 ticks | Learned pair policy | Closed-loop controller | Legacy matcher |
|---|---:|---:|---:|
| Development median score | 577.5 | 776 | 8 |
| Development score p10 | 31 | 49 | 0 |
| Development median survival | 7,536 | 8,528 | 1,541 |
| Development survival p10 | 542.5 | 2,192.5 | 726.5 |
| Development full survivors | 5/16 | 6/16 | 0/16 |
| Development gauge failures | 11/16 | 10/16 | 16/16 |
| Holdout median score | 261 | 349 | 32 |
| Holdout score p10 | 32 | 48 | 0 |
| Holdout median survival | 5,497 | 5,099 | 1,598 |
| Holdout survival p10 | 1,838.5 | 1,690.5 | 830 |
| Holdout full survivors | 4/16 | 6/16 | 0/16 |
| Holdout gauge failures | 12/16 | 10/16 | 16/16 |

On the development suite the learned policy beat legacy score on 16/16 paired
seeds and survival on 14/16; it was jointly noninferior in score and survival
on 14/16. The corresponding holdout counts were 14/16, 13/16, and 13/16.
There were zero invalid actions in either learned evaluation. These results
show a repeatable improvement in both score and survival, but not a solved
survival policy: most episodes still die before 10,000 ticks, and scores remain
orders of magnitude below the six-figure target.

Artifacts and identities:

| Artifact | SHA-256 |
|---|---|
| `long-development.json` | `491b503a03cab24786c0318628597cce8c0afb3ff09d2bbb6f98e16cbc4dfe55` |
| `long-development.pt` | `31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d` |
| `long-survival-holdout.json` | `b892669fb8b9a84f8e3507888d94b17f7cc6f30635f23609c4a6e768e548860d` |
| `long-survival-holdout.pt` | `9f575e179ae79222c9b677a3c5c188c45fa26e1314d1ea3a2cfca53d3003316a` |

Both reports bind revision `de701b3`, source identity
`ee942609e08ad1eee5ffeede4b272feccfcfc42256134f78dd0445c1729829e5`,
dataset identity
`31a8b410b7dc91717e26a35e9e820dea57da89347528132540c6f1b67c1b305c`,
and trusted portable runtime identity
`4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79`.
The two Torch files have different container hashes, but their checkpoint
metadata and every state tensor are equal. Suite identities are
`667dd6abe4aea1bac792e0f7687d8c1743fad01efbe7fcbeef4814852b33e808`
(development) and
`2bbb19d66f87c24b06a581f75c8a5384a70ddbcefb18c3d1ffcf6e1e8c22a92e`
(holdout), with no overlapping seed.

### Rejected and diagnostic follow-ups

A strategic-state stall invalidation was tested as an isolated v6 A/B. At
2,000 ticks it improved learned survival p10 from 542.5 to 1,369, but reduced
median score from 147 to 89 and worsened failures from 3/16 to 4/16. It was
rejected and the source was restored exactly to v5 before the holdout.

A development-only model-predictive diagnostic kept the learned pair choice
but branched 32 legal impact/strength geometries for 64 ticks at each safe shot.
The three v5 failures before tick 1,000 all reached tick 2,000; scores changed
from `38/16/24` to `278/241/104`. Only 19 of 293 selected shots changed
geometry. This is strong causal evidence that inverse shot dynamics—not merely
pair selection or firing cadence—is the next bottleneck. It is not reported as
holdout evidence because clone-and-branch lookahead is not the frozen deployed
policy.

The current `shot_hit_rate` is also only “projectile hit any body,” not
“projectile first hit its intended source,” and raw `CHAIN_JOINED` events can
count both members of one physical join. The next benchmark revision must bind
projectile IDs to intended source/destination IDs before using either metric to
diagnose control conversion.

## Promotion order

The default gates are deliberately sequential:

1. **Micro-control:** at least 8 episodes, 100 shots, and 90% hit rate.
2. **Group formation:** retain 90% hits while reaching at least 0.20 joins and
   0.10 qualifying clears per shot.
3. **Survival options:** at least 16 episodes, baseline score, 95% of baseline
   survival, and at most 25% gauge failures.
4. **Archive planning:** at least 3x baseline median score, baseline survival,
   chain 4, and at most 20% gauge failures.
5. **Distillation:** only after the archive-planning gate passes.

The implemented one-step archive branch teacher with closed-loop continuation
is the first policy-improvement pass; it is not the separate R3c macro beam.
Before interpreting the first two gates again, the benchmark must replace raw
any-body hits and join-event counts with intended-source and intended-pair
conversion. The next learning step is then an identity-bound pair × impact
search teacher, explicit causal contact outcomes, and distillation of its sparse
geometry corrections into the existing strength/template heads. A sealed
evaluation remains out of scope until a separately authorized protocol permits
it.
