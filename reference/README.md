# Local Reference Lab

This directory supports clean-room measurement of IriSu Syndrome. The tracked files document provenance and usage; third-party binaries, assets, recordings, and raw replays are intentionally ignored by Git.

## Local inventory

The expected local layout is:

```text
reference/
  game/irisu-v2.03-en/       runnable English-patched 2.03 working copy
  archives/                  pristine 2.03 base and English patch archives
  replays/raw/local/         preserved replay files found locally
  replays/raw/internet/      five downloaded third-party replays
  videos/raw/                downloaded gameplay references
  runs/                      disposable per-experiment game copies
  captures/                  frames, traces, and measurements
```

See `manifest.md` for exact hashes and source URLs, [`environment.md`](./environment.md) for the current host/runtime snapshot, [`mechanics-evidence.md`](./mechanics-evidence.md) for the initial shipped-configuration register, [`binary-analysis.md`](./binary-analysis.md) for recovered wrapper/replay/archive details, and [`game-rules-analysis.md`](./game-rules-analysis.md) for promoted normal-mode implementation facts. [`score-scale-binary64.tsv`](./score-scale-binary64.tsv) preserves the 99 exact score-scale values dumped from the v2.03 level setter; the clone consumes those bits instead of relying on host `pow`. The local DLL probes live under [`probe/`](./probe/), the original-game lifecycle forwarding harness under [`trace-proxy/`](./trace-proxy/), and the redistributable exact RNG model under [`rng-oracle/`](./rng-oracle/). [`version-history.md`](./version-history.md) records version boundaries and human benchmarks; [`computer-use.md`](./computer-use.md) is the operating protocol.

The Box2D oracle currently reproduces its 1,290-record Wine 11.13 golden trace
byte-for-byte across two runs. Its SHA-256 is
`b73f9c74db48a9695450ab04b76661c0d576030f7f428d0e53b953b4dae077b2`.
The clean-room DxLib RNG model matches the preserved local DLL for four checked
seeds and the normal first-spawn maxima. These are bounded numerical oracles,
not proof of long-horizon original-game synchronization.

On the supported optimized GNU build, legacy Box2D uses a scoped canonical
floating-point environment. That profile is deterministic but not bit-exact:
the full getter oracle first differs in initial actor 10's rotation after step
1 by two ULP, then in X after step 2 by one ULP. Scoped guards restore hostile
caller state; native regressions cover both single-handle and eight-lane
execution. A separate 32-bit native host for user-built exact MSVC9 objects is
documented under [`native-box2d/`](./native-box2d/). Its forward-only adapter
is the opt-in production physics backend for `IrisuEnv` and the worker-backed
vector APIs. It matches the four observed v2.03 replay oracles and the active
mutation/step/contact wrapper stream through all 47,019 steps; the default
portable GNU backend is not claimed to preserve those long horizons bit-for-bit.

[`golden/`](./golden/) contains the strict scenario manifest, schema, and
scoring contract. Its manifest remains empty until a capture bundle has valid
observed mechanics measurements; diagnostic replay headers and blocked or
invalid captures are rejected rather than counted toward the 95% gate.
Replay parity does not pass that formal gate: the manifest is still empty and
`not_evaluable`, no hashed original-game spawn/difficulty distribution
comparison exists, and no scripted policy has demonstrated qualitative
transfer.

## Launching the workspace copy

Run:

```bash
tools/launch-reference-game.sh
```

The launcher uses the isolated Wine 11.13 runtime and prefix already installed for IriSu. It starts the workspace copy, so generated saves and `replay/new.rpy` changes do not modify `/home/gabe/Games/Irisu Syndrome`.

The game overwrites `replay/new.rpy` after a run. Preserve useful replays under `reference/replays/raw/local/` with a descriptive filename before starting another run.

For controlled replay experiments, prepare the disposable game tree and capture
bundle together:

```bash
tools/prepare-reference-capture.py <experiment-id> <input.rpy> --layout padded
```

Choose `legacy` instead for a known legacy trace. Auto-detection refuses the
ambiguous case where eight leading zero input records are indistinguishable from
v2.03 padding. The tool snapshots the replay once, creates byte-verified copies in the disposable
run and capture bundle, hashes the executable, DLLs, configuration, packed data,
inherited save, and inherited `replay/new.rpy`, and prints the exact launch
command. It also removes the preserved tree's historical `launch-irisu.sh` from
the disposable copy because that script targets a different installation; the
printed workspace launcher remains authoritative. It does not launch the game or
modify the preserved source. The generated metadata is deliberately
`prepared_non_golden`; fill in its hypothesis before launch, append every action,
then preserve `replay/new.rpy` as `result.rpy` and record final hashes and observed
measurements before changing that status. A prepared bundle is never sufficient
for `golden/manifest.json` by itself.

To instrument that prepared disposable run for a decisive Box2D trace, deploy
the validated proxy before launching:

```bash
tools/deploy-box2d-trace-proxy.sh "$PWD/reference/runs/<experiment-id>"
```

For isolated experiments without an input replay, create a disposable game tree
directly:

```bash
tools/create-reference-run.sh one-weak-click
IRISU_GAME_DIR="$PWD/reference/runs/one-weak-click" tools/launch-reference-game.sh
```

Inspect a replay without launching the game:

```bash
tools/inspect-rpy.py reference/replays/raw/local/new-2026-07-17.rpy
```

## Computer-use experiments

Agents may use same-session computer control to inspect and play the reference game. Prefer exact background capture and window-targeted pointer or shortcut delivery so the user's physical pointer, focus, and workspace are not disturbed.

Read [`computer-use.md`](./computer-use.md) before operating the game. It is the authoritative safety sequence, experiment-bundle schema, and measurement backlog.

If a fresh compositor session reports that the native target-pointer component is unloaded, use `tools/ensure-native-input-plugin.sh` and re-check status before claiming the game window.

### Current reference status

The July 18 zero-display attempts for the imported padded 40- and 56-point
traces remain historical. On 2026-07-19 an active output was available and
exact IriSu window capture worked. A fresh seed-123 one-record reset under
`captures/probe-reset1-20260719-001/` visually corroborates the corrected
10-rotten/10-scripted layout. The seed-41, 520-record diagnostic under
`captures/probe-b-match-chain-20260719-001/` initially showed original HUD 16
while the clone scored 0. A July 20 fresh-process proxy and score-breakpoint run
resolved the cause: replay mode 0 selects the production table at `0x412560`,
not the nonzero-mode/Metsu-side table matching the INI. The corrected clone now
matches original `+8,+8` at tick 304 and final 16. The
[`seed-41 adjudication`](./seed-41-adjudication.md) records the exact evidence
and remaining claim boundary.

Those July 19 directories contain useful frames and replays but not the full
metadata, actions, measurements, notes, hashes, and status required by the
golden schema. They are non-golden, and `golden/manifest.json` remains empty.

The July 19 operating blocker was DirectInput acquisition without focus. The
July 20 score trial used exact replay selection under debugger control in a
fresh process while tracing the authentic DLL; it did not require global
physical input.

Regenerate the observed score input with
`python3 tools/generate-controlled-rpy.py /tmp/score-seed41-parity.rpy --preset score-seed41-parity --library build/libirisu_clone.so`.
It has 520 records, SHA-256
`1ce501febe8f3f6291e4b82736542179bd9808e412d38e0e1fb1c92d05797657`,
and checks clone `+8,+8` at tick 304 and final 16. The evidence remains outside
the golden manifest until its capture directory satisfies the full schema.

For each experiment:

1. Claim the exact IriSu window before capture or input.
2. Record game hash/version, window geometry, timestamps, and every input.
3. Capture the puzzle region before and after the action.
4. Use controlled weak/strong clicks at known window-local coordinates.
5. Verify the result with another exact capture.
6. Copy `new.rpy` before it is overwritten.
7. Release the window claim after verification.

Do not inspect unrelated windows, use global physical input, or interrupt the user's active workspace. The native Hyprland target-pointer component was built, loaded, and verified safe on 2026-07-17, but agents must re-check `session_status` at the start of every operating turn. IriSu runs through Wine/XWayland, which is supported, and every automated sequence should begin with a harmless capture/input smoke test.

## Evidence policy

- Treat the original executable as the behavioral reference.
- Keep raw third-party files untracked.
- Commit derived numeric measurements, parsers, experiment definitions, and short test fixtures only when redistribution is permitted.
- Record URLs, creator names, retrieval dates, hashes, and stated restrictions.
- Written guides and videos suggest experiments; they do not override observed behavior.
- Never silently edit the preserved source archives or preserved replay files.
