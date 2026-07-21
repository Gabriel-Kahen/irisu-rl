# Exact-forward scoring evaluation — 2026-07-20

The exact-forward simulator has full scoring parity on all four authoritative
bundled-v2.03 playback oracles available in the repository. Replay-header
claims are reported separately: three of the four eligible headers do not
describe what the bundled v2.03 executable actually does with those inputs.

## Authoritative observed-v2.03 results

| Replay | Observed v2.03 score / level / chain / checkpoint | Exact-forward | Score events | Rot penalties |
|---:|---:|---:|---:|---:|
| Header 40 | 32 / 1 / 2 / natural tick 1,020 | exact | 4/4 exact | 2/2 exact |
| Header 56 | 88 / 1 / 2 / replay exhaustion at tick 1,514 | exact | 11/11 exact | 1/1 exact |
| Header 41,449 | 41,449 / 38 / 5 / natural tick 47,019 | exact | 455/455 exact | 80/80 exact |
| Header 43,791 | 1,794 / 5 / 6 / natural tick 8,368 | exact | 66/66 exact | 20/20 exact |

Across the four episodes, all 536 score tuples `(tick, delta, cumulative
score)` and all 103 rot-penalty tuples `(tick, delta, post-penalty gauge)`
match over 57,921 playback ticks. Score, level, highest chain, 431 qualifying
clears, score-call count, and gauge at the same checkpoint also match. The
three natural episodes match their exact terminal frame. The header-56 replay
consumes all 1,514 records without an earlier natural death and matches the
original exhaustion gauge of 1,766.

The 40, 56, and 43,791 observations were each repeated in a fresh process with
a byte-identical normalized event trace. The evidence comes from canonical
executable `0636d3e4...ad28255` and shipped Box2D DLL
`34f1387c...d14fcd`, under the observed `0x027f` x87 control word. The
evaluator verifies those identities, replay and event hashes, repeat evidence,
and current-runner results independently.

The header-56 input begins with left held in records 0 and 1. The playback
runner now models the original `Input.update` protocol generically: those two
records establish held history but cannot fire a fresh shot edge. The replay
file is not normalized or special-cased by identity.

This is strong evidence that score, clear, rotation penalty, gauge, level,
chain, and death/replay-exhaustion behavior provide the right reward trajectory
on these four normal-mode episodes. No scoring divergence remains in the
authoritative corpus. The remaining scoring risk for RL is unseen,
policy-generated state/action coverage, not a currently observed formula or
timing mismatch.

## Unverified header diagnostics

| Header score / level / chain / records | Bundled v2.03 and exact-forward | Interpretation |
|---:|---:|---|
| 40 / 1 / 2 / 2,075 | 32 / 1 / 2 / natural tick 1,020 | Header incompatible with observed playback. |
| 56 / 1 / 2 / 1,514 | 88 / 1 / 2 / exhaustion tick 1,514 | Header incompatible with observed playback. |
| 41,449 / 38 / 5 / 47,019 | exact | Header, observed playback, and clone agree. |
| 43,791 / 32 / 7 / 35,694 | 1,794 / 5 / 6 / natural tick 8,368 | Header incompatible with observed playback. |

There are five replay files in the raw internet corpus. Four are normal-mode,
offset-52 inputs eligible for v2.03 evaluation and all four now have observed
oracles. The 214,453-point offset-20 replay is excluded generically because it
predates the v2.03 mechanics line.

Header mismatch is therefore not evidence of clone-scoring error for the 40,
56, or 43,791 files: the bundled original produces the same outcomes as the
clone. Aggregate header-error statistics remain in machine output as
compatibility diagnostics and explicitly are not a fidelity verdict.

## Reproduction

```sh
python3 tools/evaluate-exact-replay-corpus.py \
  --runner /tmp/irisu-exact-final \
  --compact \
  --output /tmp/irisu-scoring-authority-evaluation.json
```

The evaluator discovers validated oracle metadata by schema and status, then
matches it to corpus inputs by replay SHA-256. It has no replay-specific outcome
branches.

Evidence hashes:

- exact runner: `b2fde03afa80c1e16e82592caacfe38fdef6a93070552d71c4331cecb9137062`
- linked exact Box2D host: `14475fa3bf3f93e2a644abaadc12b2d7b981d7569a13db6873a32cafe642995a`
- evaluator output: `a60f0e154f9b10e45cf4ea588d6414f1c200cb0082a7cf4fcd340085a19c7471`
- header-40 repeated events: `0a8222d03f8b7099697ba8c067a7dc56a28fe1fe3af10d26da9f6ab887938f69`
- header-56 repeated events: `bd716e2d94263a7af7da9fc96fc34c3f228641d174df3f2f466e309d5f99b63a`
- 41,449 observed events: `f8ab8fb968543ca7ca0daf30d6b78e07575659e0d87d799cd9e7dc66edf2e878`
- 43,791 repeated events: `9933ba8de1979b8f8171839c5de07ff9e2a72b4928b93722c5352712c35f1d27`

## Full wrapper-stream cross-check

The final runner also produced a complete `IRISU_EXACT_TRACE` for the 41,449
replay. `tools/compare-exact-wrapper-trace.py` streamed the 1.5 GB,
10,777,298-record original getter trace without loading it into memory. After
removing two constructor/bootstrap generations, all 813,412 active native
records match their original per-operation streams through physics step 47,019:
2,368 creates, 2,368 velocity sets, 2,368 user-data sets, 183,387 transform
sets, 2,344 gameplay destroys, 47,019 steps, and 573,557 contact results.

The original-only post-step process teardown contains 153,500 transform writes,
24 destroys, and one dispose, all at step 47,019. Getter observations are not
wrapper mutations and are excluded. Exact bounds and hashes are recorded in
`reference/native-box2d/validation.json`. The final trace SHA-256 is
`cde35ca60b5511678edf128ed8f3ae09c8cf00e240696325c4acc8681f829eb0`;
the streamed comparison output SHA-256 is
`1db375516a07e068f66841c07e295e53eca7bda9b236fcf7b82bd77147e485ff`.
