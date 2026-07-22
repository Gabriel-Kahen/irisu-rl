# R2a recurrent PPO correctness kernel

R2a adds the neural-training kernel without claiming that a transferable agent
exists. The first smoke policy consumes `teacher-v1`, so every model manifest
and checkpoint is marked `deployable=false`, `observation_provenance` is
`privileged_simulator`, and the R4 tracker/input calibration gate remains open.
The architecture is schema-parameterized so a later causal actor can use the
same audited math without treating simulator truth as a deployment input.

## Locked software boundary

The optional training extra pins PyTorch 2.13 and TensorBoard 2.20. PyTorch is
resolved from its explicit CPU wheel index, avoiding several gigabytes of CUDA
packages on CPU collectors and CI. Run manifests record the actual Python,
NumPy, PyTorch, CUDA availability/build, thread counts, deterministic-algorithm
flag, model/action/schema identities, and `uv.lock` hash. The deterministic R2
baseline uses float32 eager execution without AMP or `torch.compile`.

## Recurrent and observation convention

The masked body encoder zeros padded rows before and after its shared MLP, then
combines mean and maximum Deep Sets pools. This makes body ordering irrelevant
and prevents stale or nonfinite padding from reaching the model. A GRU consumes
one encoded observation per semantic policy decision.

`h_t` is the recurrent state before processing `o_t`. The model produces policy
heads, `V(o_t)`, and `h_(t+1)`. A semantic macro does not update the recurrent
state again. Environment reset clears state before the first reset observation;
a rollout chunk boundary does not. PPO minibatches contain complete lane
sequences and never shuffle individual timesteps.

## Conditional action likelihood

The Torch distribution mirrors the independent NumPy oracle. Total likelihood
is kind likelihood plus exactly one active branch: wait duration for `WAIT`, or
the selected weak/strong x/y Beta likelihood for a shot. Selected masked actions
fail before an optimizer step. Sampled actions canonicalize inactive fields and
do not consume random numbers for inactive Beta branches. Analytic entropy is
the action-kind-probability-weighted conditional entropy over all branches.

## SMDP returns and censoring

Raw environment reward remains signed 64-bit score delta and is always retained
for audits. Optimizer scaling is separate, fixed for an experiment family, and
never used in reported evaluation score.

For elapsed ticks `k_t`, R2 computes:

```text
delta_t = r_opt_t
          + bootstrap_mask_t * gamma_tick**k_t * V_bootstrap_t
          - V_t

A_t = delta_t
      + trace_mask_t * (gamma_tick * lambda_tick)**k_t * A_(t+1)
```

`bootstrap_mask` is authoritative; it is not inferred from `terminated`.
Neutral truncation bootstraps from the retained final observation, while
advantage recursion stops at every episode boundary. An interrupted held-press
truncation that cannot bootstrap is censored: it remains in raw diagnostics but
is excluded from policy and value loss instead of being mislabeled as death.

R2 locks `gamma_tick=1`. A value below one is rejected unless rewards were
constructed from per-event tick offsets; discounting one aggregated macro score
delta would not be exact. `lambda_tick` is derived from a declared wall-time
half-life rather than copying a decision-level `0.95` constant.

## PPO safety

Before the first optimizer mutation, stored action likelihoods and values are
recomputed from the collection observations, masks, incoming hidden state, and
current model. Mismatch fails closed. Advantages normalize only across
loss-bearing decisions. Padding is neutralized before exponentiation, all
selected outputs/losses/gradient norms must be finite, gradients are globally
clipped, and KL can stop remaining epochs early. Complete lane sequences—not
individual decisions—form minibatches.

The baseline learning rate is `3e-4`, with an exact linear schedule whose last
budgeted update uses `0.1 * initial`. This value is provisional. R2b compares
`1e-4`, `3e-4`, and `6e-4` under identical simulator/optimizer seeds and tick
budgets before freezing a selection.

## Checkpoint and resume contract

Checkpoints are taken only at a clean semantic boundary after a complete PPO
update. Each generation is an immutable directory with a checksummed manifest,
weights-only Torch state, and per-lane environment snapshots. Files and the
directory are flushed before publication; a small `latest.json` pointer is
updated atomically, and an existing generation is never overwritten.

The format can carry model, optimizer, learning-rate schedule, minibatch
sampler, Python/NumPy/Torch RNGs, recurrent lane state, current encoded
observation, adapter ticks/scores/seeds/episode IDs, seed allocator cursor,
environment snapshots, and state hashes. Restore validates the caller-supplied
run identity and file hashes before deserialization. Portable and exact-worker
fixtures prove exact adapter continuation for a predetermined next semantic
macro; a separate fixture proves an exact next optimizer update from a stored
batch. R2a does **not** yet claim one composed sampled-action → environment →
rollout → update replay. R2b must add that integrated gate for its collector.

Run manifests require producer-supplied observation provenance. Schema shape is
not accepted as evidence that actor-track tensors came from a causal tracker.

The authoritative training lock targets Linux x86-64 with CPU PyTorch. Other
platforms are not an R2 acceptance surface yet.

## R2a exit boundary

R2a establishes training math, recurrent semantics, likelihood parity,
checkpoint integrity, exact adapter continuation, and exact optimizer
continuation. It does not establish an integrated collector resume, learning
quality, or transfer. R2b must demonstrate behavioral cloning and held-out
one-body PPO learning and add collector-level resume; R4 must still produce the
causal visual tracker, capture timing, coordinate transform, and input bridge
before any policy can be called deployable.
