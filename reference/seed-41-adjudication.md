# Seed-41 score discrepancy adjudication

Adjudication date: 2026-07-20.

Status: **resolved for this replay, exact score parity, non-golden**. The score
formula was already correct. The clone had initialized mode-0 normal play with
the separate nonzero-mode/Metsu-side table that matches `data/doc/irisu.ini`.
Using the executable's actual mode-0 table makes original and clone score
`+8,+8` at tick 304 and finish all 520 records at 16.

## Replay identity

The replay is
[`captures/seed41-score-parity-20260720-001/input.rpy`](./captures/seed41-score-parity-20260720-001/input.rpy),
SHA-256
`1ce501febe8f3f6291e4b82736542179bd9808e412d38e0e1fb1c92d05797657`.
It is a 2,132-byte padded v2.03 replay with seed 41, mode 0, 520 records, and a
deliberately zeroed outcome header. Its only non-wait records are right/strong
button levels:

| Replay frame | Cursor `(x,y)` | Raw word |
|---:|---:|---:|
| 4 | `(453,380)` | `1558294` |
| 272 | `(367,380)` | `1557950` |
| 364 | `(303,380)` | `1557694` |
| 384 | `(303,380)` | `1557694` |

The zero score in this header is intentional and is not the observed outcome.

## Decisive original-game evidence

A fresh original-game process replayed those exact bytes with the authentic
Box2D DLL behind the forwarding trace proxy. The validated clean trace contains
26,973 records and 521 physics steps:

`reference/runs/seed41-score-parity-20260720-001/data/dll/box2d-trace.jsonl`

It records the mode-0 field fixtures, selected spawn dimensions, complete
contact enumeration, and native destruction. The first step destroys newborn
actor IDs in this exact order:

```text
14, 20, 16, 19, 18, 13, 11, 15
```

A second fresh-process Wine-GDB run stopped at the original score routine
`0x4036b8` while retaining the DLL proxy trace. At proxy step 304 it hit twice:

| Call | Actor | Size slot | Color | Group chain | Group num | Score before | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 1 | 0 | 2 | 1 | 0 | 8 |
| 2 | 12 | 2 | 0 | 2 | 1 | 8 | 8 |

The live original score is therefore exactly 16 after the two calls. Native
destruction follows in actor order 10 then 12. The decisive trace is under
`reference/runs/seed41-score-gdb-20260720-001/`.

## Root cause

The parameter selector at `0x4124c0` reads the mode stored at
`SaveData+0x1c`. The replay loader copies the signed mode from header offset
`+0x10` into that state. Mode 0 selects initializer `0x412560`; nonzero mode
selects `0x413648`, whose values match the shipped INI. This is a mode branch,
not unrecorded save-state provenance. Every preserved external replay header in
the current corpus says mode 0.

The important mode-0 values are:

| Family | Mode-0 normal value |
|---|---|
| Field | `x=130,y=120,w=320,h=250,blank=40,thickness=16` |
| Top / bottom / sides | top `(-140,320,300)`, bottom height `16`, sides `120..370` |
| Sizes | `32,46,54,60,72,90,140,5,5,5` |
| Weights | `20,28,28,14,5,3,1,0,0,0` |
| Block life / strict rot threshold | `100000 / 40` |
| Projectile life | `3000` |
| Gauge maximum / initial | `40000 / 3000` |
| Qualifying clears per level | `10` |

The old clone defaults were the nonzero-mode table: a larger, higher field,
70-ticket three-size distribution, gauge 10,000/1,000, rot threshold 120, and
shorter lifetimes. That produced a different initial board, trajectories,
contacts, rot/game-over timing, and ultimately no seed-41 scoring group. The
score routine itself was downstream of that divergence.

## Correction and result

The clone now defaults to the ten-slot mode-0 table and samples all ten weight
slots. Its mechanics schema is 3 and nominal config hash is
`0xec0e8463feaf2670`. No score constant, multiplier, replay special case, or
reward shaping was added.

On the preserved seed-41 replay the clone now matches:

- first-step native destruction order;
- actor 10 then actor 12 scoring at tick 304;
- exact `+8,+8` score deltas, chain 2, and final score 16;
- full 520-record nonterminal playback.

This directly validates the RL reward for this transition because the core
reward remains `score_after - score_before`.

## Claim boundary

Seed 41 remains a controlled, non-golden adjudication of one score transition.
Subsequent fresh-process instrumentation produced observed bundled-v2.03
oracles for all four eligible padded external replays. The exact worker matches
all four on terminal/checkpoint state and every one of their 536 score calls
across 57,921 ticks, including the longest replay's scoring/terminal oracle
through all 47,019 ticks. Their
replay headers remain diagnostic rather than authoritative: only the 41,449
header describes the observed v2.03 outcome, and the 214,453-point legacy replay
still predates v2.00.

The old 48/114/1,563/1,553 clone outcomes were portable-backend diagnostics from
before exact-physics integration and are historical, not current exact-backend
claims. Portable GNU physics can still diverge over long horizons; the exact
worker is required for replay parity. No score constant, multiplier, or
replay-specific branch was introduced.

This result resolves the seed-41 discrepancy but does not populate the empty
five-category controlled manifest, statistically validate original-game
spawn/difficulty distributions, meet the exact-backend throughput gate, or
demonstrate policy transfer.
