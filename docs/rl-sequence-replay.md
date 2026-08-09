# Sequence-replay development branch

This is development-only evidence. It is not deployable or canonical R3
evidence. No sealed, test, or canonical run was read or executed.

## Outcome

The recurrent residual learned useful offline act/restraint, pair, and geometry
signals, but no nonzero checkpoint passed the preregistered catastrophe-first
selection rule. Step 0 (the exact zero-residual frozen-V5 fallback) was selected.
The fully trained step-600 policy was still evaluated through 20k on all 16
fixed development seeds and through 50k on the first eight. It introduced one
20k terminal-flip catastrophe, so it is rejected even though aggregate medians
were unchanged.

This is a negative deployment result and a positive diagnostic result: learned
history is feasible and permutation-safe, but sparse local option labels plus
high-confidence lowering did not produce a sustainable improvement.

## What the memoryless students omit

- Frozen V5 is a feed-forward directed-pair network. Numeric body and chain IDs
  are ablated, but its only deployed state is hand-coded cooldown, progress,
  and same-tick caching. Its intent head is auxiliary and does not provide
  option memory or duration.
- Legacy R3e freezes V5 act/restraint, pair, and cadence. It selects one of 32
  shot geometries from a single snapshot. It cannot learn delayed join/clear
  credit, act versus restraint, option persistence, or miss/rot recovery.
- The new model adds invariant mean/max body pooling, a 96-wide GRU over 31
  previous-completed-transition features, bounded residual act/wait/pair/
  intent/geometry heads, 51 return quantiles, viability and outcome heads, and
  conservative V5 fallback gates. `id_scaled`, `chain_id_scaled`, and absolute
  `tick_scaled` are zeroed. It has 144,366 trainable parameters plus the frozen
  71,590-parameter V5 base.

## Exact identities

- Human replay:
  `73bf5b5d4a478c9bf73b62a6df98f16a01fde2cf97eb751438cd0ae857e3362d`
- Exact worker:
  `4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261`
- Exact mapped runtime:
  `ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5`
- Portable runtime:
  `4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79`
- Frozen V5 checkpoint/state:
  `31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d` /
  `17c26b2beda17e85f5dab1b3a92dad5fccbf8210433dc98fd0b38641783b453a`
- Fast R3e checkpoint/state:
  `5db3b5cc3fe7583d98e294561d3928677ee708bc62327b67bfb9e7da46eaaefe` /
  `f23bfc6c82f183c6f7f350e5f4a85c697befe58c6de453ba42c8459365a2cce5`
- Extended R3e checkpoint/state:
  `6752f1c7be5a05e75d7bd85f0464e58c327acfc5b0155461e851ac13451c931a` /
  `7dc96cfd576fc622b57bdfaed9ed9658bb56a00baaeeca3b03ac11d464bf0826`
- Geometry collection/dataset/vocabulary:
  `906eb8c468d87efe4c26b1d139a4b76c99e86b6c643f5b1ca81e188a74dac08c` /
  `d1147b91d681606c1077afd4e37af4f60ba263622cb7cd31fe6c23a3cb350805` /
  `e6cd65d87e6ca4fa5a2bcb89cd9e61186f29eac265af25e867eaf69f0346d7a4`
- Sequence architecture:
  `70f9b7ee53ecb7a8b98bc5d4802d4ac9adaf713bf58e9c8e34eaa739dfb61ed5`
- Training source identity:
  `d9a11f05c4105691b1ee15fd3200993012946b6eb56a3661ac417b2d5e67390f`
- Finalization source identity:
  `3ae231b851e0ad8bd5dc314cfa79b85d7ab488a0ea8cd8bbb207ebd45ea10260`
  (the only post-training source change repaired comparator identity-key
  binding; checkpoint source binding was not changed).

Checkpoint identities:

| Steps | SHA-256 | Role |
|---:|---|---|
| 0 | `6ab3113d9c792bc481eac7e25950330bddc2d93b570001572e120e18222cbcb7` | selected exact V5 fallback |
| 80 | `539eae4181f1fc8a98e54ddf50b6539abb9afd9d2b3bd455387cad26b68ac3a1` | plateau |
| 240 | `dc1579715e8ba77d648b50fcd90f8a030fa1fa0229820bd701d3036d243d2b88` | rejected: selection catastrophes |
| 600 | `ea80b1d4ce8b9b2b03da9fdc449c2faa91237399322595b14fc3edebca1f1468` | trained diagnostic; rejected at 20k |

## Disjoint development seeds

- Replay seed: `168175029`
- DAgger training:
  `[3993225437, 1210292757, 3690546926, 2275139235, 3457209652, 4085297243, 3667357959, 4166920730]`
- Legacy geometry training:
  `[3643205411, 1211936443, 4184595850, 2573156672]`
- Selection:
  `[3747571761, 3633074689, 3165196451, 1295131450, 1147578272, 4251965354, 886910123, 2939586874]`
- Fixed final development suite:
  `[3028406002, 2789482418, 2666547395, 2200065417, 1075831150, 3122472238, 1753098387, 2897648126, 1770871059, 29484235, 962351311, 3845527485, 1334142681, 550658739, 3738769297, 3062439664]`

All sets are pairwise disjoint.

## Replay and causal labels

The exact replay reproduced 47,019/47,019 frames, score 41,449, level 38,
highest chain 5, 1,025/1,037 hits, 500 joins, 462 clears, 80 rotten events,
235 ejections, and zero invalid actions. It yielded 455 strict chronological
labels. Destination labels are explicitly nearest-visible-same-color
behavioral inference, not recovered human intent.

Eight behavior-visited portable DAgger episodes added 3,014 labels: 1,269
shots and 1,745 restraints. Ninety-six same-state causal queries evaluated 418
branches using a shared frozen-V5 128-tick continuation and produced 13
corrections. The rollouts produced 69 qualifying clears and 9 rotten events.
The immutable R3e collection added 182 geometry labels, 137 marked improved.

## Training curve (offline diagnostics only)

| Steps | Development act | restraint | pair | replay pair | geometry winner | geometry gate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 53.9% | 24.7% | 36.3% | 8.8% | 93.4% | 75.3% |
| 80 | 54.7% | 33.9% | 37.1% | 6.4% | 93.4% | 75.3% |
| 240 | 58.7% | 59.1% | 38.5% | 39.3% | 93.4% | 75.3% |
| 600 | 79.8% | 78.9% | 47.5% | 57.0% | 94.0% | 85.2% |

Training used an 8-step burn-in and 32-step TBPTT. The configured
`baseline_kl_weight` is implemented as an L2 penalty on bounded residual
logits, not a literal KL.

## Real portable results

Score is median / p10 / max. Survival is median / p10. Gauge and level show
median (level also shows maximum).

| Horizon | Policy | Score | Survival | Gauge failures | Gauge median | Level median/max | Qualifying clears | Rot | Decisions/corrections |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|
| 2k | V5 = fast = selected step 0 | 76 / 34 / 190 | 2000 / 1173.5 | 4 | 4290 | 1 / 1 | 78 | 19 | 3855 / 0 |
| 2k | extended R3e | 90 / 24 / 184 | 2000 / 1147.5 | 3 | 3940 | 1 / 1 | 82 | 19 | 3996 / 406 |
| 2k | trained step 600 | 76 / 36 / 190 | 2000 / 1555 | 4 | 4290 | 1 / 1 | 79 | 20 | 3964 / 1 geometry |
| 10k | V5 = fast = selected step 0 | 103 / 34 / 1538 | 3149 / 1173.5 | 11 | 1 | 1 / 5 | 285 | 93 | 10793 / 0 |
| 10k | extended R3e | 418.5 / 42 / 2355 | 6361 / 1147.5 | 11 | 1 | 3 / 7 | 373 | 122 | 13195 / 1332 |
| 10k | trained step 600 | 103 / 36 / 1538 | 3149 / 1555 | 11 | 1 | 1 / 5 | 294 | 89 | 10906 / 1 geometry + 1 restraint |
| 20k | V5 = fast = selected step 0 | 103 / 34 / 5461 | 3149 / 1173.5 | 13 | 1 | 1 / 11 | 507 | 208 | 15483 / 0 |
| 20k | extended R3e | 418.5 / 42 / 3734 | 6361 / 1147.5 | 16 | 1 | 3 / 9 | 493 | 225 | 16185 / 1727 |
| 20k | trained step 600 | 103 / 36 / 5461 | 3149 / 1555 | 14 | 1 | 1 / 11 | 479 | 201 | 14915 / 1 geometry + 1 restraint |

The aggregate p10 improvement of step 600 hides a terminal flip. The selected
step-0 policy had zero score, survival, or gauge differences from V5 at every
2k/10k/20k seed.

At 50k on the first eight final seeds, V5/selected step 0 scored
93 / 54.2 / 7069 with survival 2523.5 / 1338.9, eight gauge failures, 272
qualifying clears, and 123 rotten events. Step 600 scored 93 / 55.4 / 7069
with survival 2523.5 / 1567.8, eight gauge failures, 273 clears, 124 rotten
events, and no paired regression on this smaller suite.

## Every paired catastrophe

Fresh selection catastrophes introduced by step 240:

| Horizon | Seed | Reason | Survival V5→candidate | Score V5→candidate |
|---:|---:|---|---:|---:|
| 2k | 886910123 | terminal flip | 2000→1200 | 74→0 |
| 2k | 3165196451 | terminal flip | 2000→1922 | 74→104 |
| 10k | 886910123 | terminal flip + severe joint collapse | 10000→1200 | 1049→0 |
| 10k | 3165196451 | terminal flip + severe joint collapse | 10000→1922 | 1165→104 |

Fresh final catastrophe introduced by step 600:

| Horizon | Seed | Reason | Survival V5→candidate | Score V5→candidate | Gauge failure |
|---:|---:|---|---:|---:|---|
| 20k | 3738769297 | terminal flip | 20000→14727 | 5422→3302 | false→true |

Extended R3e comparator catastrophes:

| Horizon | Seed | Reason | Survival V5→R3e | Score V5→R3e |
|---:|---:|---|---:|---:|
| 2k | 3062439664 | terminal flip | 2000→1298 | 40→24 |
| 10k | 29484235 | terminal flip | 10000→6986 | 974→497 |
| 10k | 1753098387 | terminal flip | 10000→6296 | 1538→420 |
| 10k | 1770871059 | terminal flip | 10000→6426 | 985→535 |
| 10k | 3062439664 | severe joint collapse | 9753→1298 | 1261→24 |
| 10k | 3738769297 | both | 10000→3790 | 918→114 |
| 20k | 29484235 | severe joint collapse | 16325→6986 | 2868→497 |
| 20k | 1753098387 | both | 20000→6296 | 5461→420 |
| 20k | 2789482418 | terminal flip | 20000→13238 | 4552→2596 |
| 20k | 3062439664 | severe joint collapse | 9753→1298 | 1261→24 |
| 20k | 3738769297 | both | 20000→3790 | 5422→114 |

Fast R3e made zero corrections and was episode-identical to V5. Selected step
0 had no catastrophes. The four step-240 rows are two unique selection seeds;
the 11 extended-R3e rows are six unique final seeds.

## Cost and limitations

- DAgger: 54.42 s wall / 53.10 s CPU.
- Recurrent training: 766.04 s wall / 746.24 s CPU.
- Real selection: 493.64 s wall / 490.26 s CPU.
- Comparator evaluation: 183.32 s wall with three workers / 482.48 s summed
  worker CPU.
- Final selected + trained sequence evaluation, including 50k: 980.69 s wall /
  974.27 s CPU.
- Exact replay reproduction was observed at approximately 55.7 s wall; its CPU
  time was not recorded, so no fabricated total is reported.
- One pre-training attempt was stopped after observing an incorrectly inherited
  four-thread pool: 250 s wall / 953 s CPU, no artifacts written. The successful
  campaign applied the one-thread cap before inference.

The 128-tick branch values are local causal labels under a frozen-V5
continuation, not full-episode counterfactuals. The legacy collection did not
retain per-state candidate availability, so geometry auxiliary training uses
all 32 semantically fixed slots and records that approximation. The recurrent
pair path remains quadratic in active bodies. Most importantly, high offline
accuracy did not translate into a safe nonzero policy.

## Validation and artifacts

Forty-three sequence, policy, benchmark, checkpoint, causality, permutation,
progress, and steering-learning tests pass. The key artifacts are:

- `campaign-report.json`:
  `1dfc5fc4c76d6d0f4a03bd01d74b3241e564305689496bd43596ac682e4a23ab`
- `final-sequence-evaluation.json`:
  `9a84011294c08dfe57ce7105a4c9ce45b06719ef7382e23f62e1b5da96b2c56d`
- `comparator-evaluation.json`:
  `89dc95554c5b220d0989e48c1a209423c018fc9fc25700c622f91fe775354d65`
- `selection.json`:
  `02e3ba356e5b336ad9aae312897f99a9eb9681d1f04a55a6466828ae54518ab6`
- `training-curve.json`:
  `83bf83e8e872ec2220da740095f309d564ade75ae2baef9a1e88567f93999804`
- `training-data-evidence.json`:
  `5c2c74ab9a203fe920a60086e1ffdaaef8e4e8fbe8534ba2f96f8a3bcb677492`

No commit or push was made.
