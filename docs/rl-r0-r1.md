# R0/R1 neural-ready foundation

This foundation deliberately stops before PPO. It fixes the contracts that a
trainer must not invent ad hoc: model inputs, semantic actions, click edges,
semi-Markov timing, final observations, seed ownership, runtime identity, and
owned rollout storage.

## Transfer boundary

`ActorTrackEncoder` accepts only causal HUD, timestamp, input-bridge, and visual
track mappings. It rejects native padded state. `TeacherStateEncoder` is the
separate privileged path for simulator truth, teacher search, or an asymmetric
critic. This type-level split prevents exact IDs, timers, native ticks, score,
and future/RNG state from becoming accidental deployment dependencies.

The actor layout is fixed float32 `[B, G]` plus `[B, 196, F]` and a boolean
mask. It includes confidence, missing/merge/occlusion state, timing uncertainty,
requested, injected, and acknowledged input state, separate effect evidence,
and effect-time coordinates so the real detector/tracker can produce the same
contract. Injection acknowledgment means only that the bridge accepted an
input; it is never treated as evidence that the game produced the intended
effect. The simulator cannot yet honestly produce this
actor record directly; R4 must add a render-visible oracle/tracker and measured
latency/input calibration before transfer-oriented bulk training.

## Actions and SMDP transitions

The policy selects `WAIT(k)`, `FIRE_WEAK(x,y)`, or `FIRE_STRONG(x,y)`. A fire is
lowered to one raw press tick and a forced neutral release tick. The vector API
can step active lane subsets, so release completion never injects fake waits
into lanes already at a decision boundary. Every adapter call returns all lanes
at a clean policy/update boundary. Internally this is the atomic equivalent of
the roadmap's READY/RELEASE_PENDING scheduler: only incomplete release lanes
advance before the synchronous decision barrier is returned.

Raw score delta is aggregated without clipping or shaping. Elapsed duration is
the observed raw native tick difference. Terminal observations are encoded and
owned before autoreset; reset observations are separate. Native truncation
during a neutral wait can bootstrap from the retained final observation;
truncation during a held press is marked and disables ordinary bootstrap
because residual release state is not an actor input.
Backend failures after dispatch poison the adapter because sibling lanes may
have committed; they are never converted into training truncations.

## Efficiency and learning-rate handoff

The typed teacher encoder uses a vectorized structured NumPy view for both the
aligned portable ABI and packed exact ABI, then copies into contiguous owned
float32 storage. Collection batches the first primitive across every lane and
executes only required releases concurrently. The R1 smoke buffer stores each
ordinary observation once and keeps additional final tensors only at episode
boundaries, plus the per-lane batch at the rollout cut. R2 must add recurrent
states, action masks, old log-probabilities and values, optimizer rewards,
advantages, returns, and critic-specific tensors before it is a complete PPO
buffer.

R0/R1 contains no optimizer, so claiming a tuned learning rate here would be
false. The R2 handoff starts recurrent PPO at `3e-4`, compares `1e-4`, `3e-4`,
and `6e-4` under identical train/held-out seed budgets, and uses raw held-out
score plus KL, clipping, value fit, and gradient diagnostics. The choice and
schedule must then be frozen in checkpoint metadata.

## Smoke usage

Install the optional dependency group and run the end-to-end benchmark:

```bash
uv sync --extra training
uv run --extra training python benchmarks/rl_r1.py \
  --backend exact --worker /absolute/path/to/irisu-exact-worker
```

Exact training must also attest the absolute worker path against
`configs/rl/runtime/exact-worker-2026-07-21.json`. Automatic build-directory
discovery is not an accepted training identity.
