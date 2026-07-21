# Original-game golden scenarios

This directory is the strict bridge between ignored reference-game captures and
the controlled-scenario fidelity subgate. `manifest.json` is intentionally
empty: no completed capture currently qualifies as observed mechanics evidence,
so the gate is **not evaluable**, not passed.

`schema-v1.json` describes the JSON shape. `tools/score-golden.py` is the
authoritative dependency-free structural and semantic validator. It refuses to
score a scenario unless all of the following hold:

- the manifest exactly matches the scorer's independent, hard-coded v2.03 target
  tuple, and the capture metadata status is `valid_for_mechanics_calibration`
  with that same game version and four artifact hashes;
- the scenario is marked `observed`, and `measurements.json` contains exactly
  one matching `valid_mechanics_measurements` entry also marked `observed`;
- every scenario uses a unique `(experiment_id, measurement_id)` pair, so
  relabeling one captured measurement cannot add votes to the gate. Scenarios
  with the same canonical behavioral prefix (seed, button levels, and cursor
  coordinates only on fresh edges, compared through the shorter window end)
  must also use nonoverlapping frame windows, so editing replay outcome fields,
  reserved bits, or window endpoints cannot add a vote;
- evidence bundle roots are confined to the manifest directory or its sibling
  `captures/` directory. Symlinked bundle roots and arbitrary relative siblings
  are rejected;
- metadata, JSONL actions, measurements, replay, notes, and at least one PNG
  frame are present and every listed SHA-256 matches. PNG chunks, ordering,
  CRCs, zlib stream, dimensions, scanline lengths, and filters are validated;
  the explicit replay layout parses as normal mode, and its hash also matches
  capture metadata;
- the manifest and every evidence file are captured with their SHA-256 and
  filesystem identity during validation, then checked again after all scenario
  scoring. Any change fails closed, and the report records the initially
  validated manifest SHA-256;
- every action record has a nonblank `action`, a nonblank string `result`, and
  one positive, strictly increasing sequence field. A file may use either
  `sequence` or `monotonic_sequence`, but it must use the same field throughout;
- the tracked oracle exactly equals the oracle inside the hashed measurement
  entry. This prevents clone output or an unlinked transcription from silently
  becoming reference truth. Each scenario must include at least one positive
  category-relevant event assertion (`min_count >= 1`); absence assertions may
  be additional checks, but cannot establish category coverage.

## Promotion contract

Keep raw frames, video, and generated replays in the ignored
`reference/captures/<experiment-id>/` bundle. After repeated inspection, add an
entry like this to its `measurements.json`. The numbers below are illustrative
schema examples, not mechanics evidence:

```json
{
  "valid_mechanics_measurements": [{
    "id": "same-color-landing-001",
    "status": "observed",
    "category": "match",
    "repeat_count": 3,
    "replay_window": {"first_frame": 120, "last_frame": 180},
    "oracle": {
      "events": [{
        "kind": "confirmed",
        "min_count": 1,
        "max_count": 1,
        "first_frame": 160,
        "last_frame": 165
      }],
      "scalar_transition": {
        "from_frame": 119,
        "to_frame": 180,
        "score": {"before": 0, "after": 8, "delta": 8},
        "gauge": {"before": 880, "after": 1430, "delta": 550},
        "level": {"before": 1, "after": 1, "delta": 0}
      },
      "trajectories": [{
        "frame": 144,
        "body_id": 21,
        "x": 211.5,
        "y": 83.25,
        "tolerance": 8.0
      }]
    }
  }]
}
```

`repeat_count` is reporter-supplied advisory metadata. The current admission
format does not link it to separately hashed trial bundles, so the validator
only requires a positive integer and reports it as `reported_repeat_count` with
`repeat_count_status: "advisory_not_independently_verified"`. It never weights
a scenario or contributes evidence to the gate. Repetitions may be treated as
gate evidence only after each trial is independently linked and hashed.

Copy that oracle exactly into one manifest scenario, list and hash its evidence
files, and identify the explicit replay window. Frame `-1` means reset state;
frame `0` is the observation after the first replay record. Trajectory points
use display coordinates and Euclidean tolerance. `body_id` must be mapped from
the controlled actor/spawn order and documented in the measurement evidence.
To prevent token trajectory checks, each category needs at least one point
25–50 updates (0.5–1.0 seconds) from its scenario's first frame, and tolerance
cannot exceed 15 display pixels.

## Scoring

```bash
PYTHONPATH=python python3 tools/score-golden.py \
  reference/golden/manifest.json \
  --worker build-exact/irisu-exact-worker
```

That command exercises the exact-MSVC9 backend. Use
`--library build/libirisu_clone.so` instead for the portable backend;
`--worker` and `--library` are mutually exclusive.

The scorer gives one vote to each uniquely linked source measurement, represented
by one scenario, regardless of its assertion count or advisory `repeat_count`.
It requires nonzero positive-event coverage and at least 95% passing scenarios
separately for `match`, `rot`, `chain`, `ejection`, and `orb`, as well as 95%
overall. The comparison is exact integer arithmetic (`passed*100 >= total*95`);
rates are not rounded for the verdict. Every score, gauge, and level transition
must be exact, and every trajectory point must be within its declared tolerance.
Trajectory coverage is required separately for all five categories.

For a real portable `IrisuEnv`, the report's `clone_library` records the
resolved loaded shared-library path, byte length, SHA-256, and local file
identity. The scorer hashes that file before and after every scenario and fails
closed if its identity or contents change. A system loader name that cannot be
resolved to exact bytes is not admissible; pass `--library` with an explicit
path. Custom portable test environments may instead report library provenance
as `unavailable`.

For exact scoring, `clone_worker` records the executed worker artifact and
embeds `linked_exact_library` for the host actually mapped by the live worker.
The environment captures the worker executable only after Hello proves `exec`
completed, then verifies the mapped library's path, device, inode, ELF segments,
worker and client mount identities, metadata, and bytes against the handshake
SHA-256. It also requires the worker's 15-target runtime attestation to match
that independently captured mapping, preventing a mapped genuine host from
masking interposed `b2d_*` calls. The scorer independently rehashes both artifacts before and after each
scenario and rejects unstable or inconsistent provenance.

This tool scores the controlled-scenario subgate only. Even an exit-0 report
explicitly leaves the full `clone.md` fidelity gate false: a separate hashed
statistical comparison of spawn/difficulty distributions and an original-game
policy-transfer result must still be supplied and combined with this report.

Exit status `0` means the controlled-scenario subgate passed, `1` means
admissible scenarios were evaluated and failed, and `2` means the
manifest/evidence was not evaluable. An empty or incomplete corpus can never
pass. The tracked five-category manifest is currently empty, so both backend
commands currently return `2` (`not_evaluable`); no golden-fidelity claim is
made.
