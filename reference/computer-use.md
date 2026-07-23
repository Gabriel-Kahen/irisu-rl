# Reference-Game Computer Use Protocol

This protocol is for agents measuring the authorized local IriSu Syndrome copy. Use the `same-session-computer-use` skill and its broker tools; do not invent a second input mechanism.

The native Hyprland target-pointer component was built and loaded on 2026-07-17. At verification time, `session_status` reported exact background capture, targeted shortcuts, targeted Wayland and XWayland pointer input, and `native_input_currently_safe: true`. Its loaded state is compositor-session-specific. Every agent must check current status rather than relying on this historical result.

If `session_status` reports `native_plugin_loaded: false` while all native build requirements are true, run `tools/ensure-native-input-plugin.sh`, then call `session_status` again. The script locates the installed same-session plugin, asks its version-aware loader to build/load the correct Hyprland ABI artifact, and verifies `cutargetstatus`. Do not add a hardcoded cached `.so` path to Hyprland configuration; that path becomes stale when Hyprland's ABI or the Codex plugin changes.

## Safe operating sequence

1. Read the installed `same-session-computer-use` skill in full for the current turn.
2. Call `session_status`, `list_window_claims`, and every page of `list_session_windows` with a page size of at most 20.
3. For a replay probe, run `tools/prepare-reference-capture.py <experiment-id> <input.rpy> --layout padded` (or explicitly select `legacy`) and use the exact launch command it prints. The tool stages the disposable tree and non-golden capture skeleton, then publishes each with an atomic no-clobber rename; ambiguous layout is rejected rather than guessed. For a no-replay probe, create a disposable tree with `tools/create-reference-run.sh <experiment-id>`. Never experiment in the preserved source installation.
4. Find the exact IriSu window by address or exact-capture ID. Claim it with `claim_session_window`; use a lease long enough for the immediate probe and renew it during longer work. Treat the claim token as a secret.
5. Capture the exact window before sending input. Record the returned window geometry and pixel-to-window scale. Derive window-local input coordinates from that capture and bounds-check them.
6. Use `send_window_shortcut` for discrete keys. Use `targeted_pointer_click`, `targeted_pointer_drag`, or `targeted_pointer_scroll` for game coordinates. IriSu runs through Wine/XWayland, which is supported. Do not move the physical pointer or focus the window just to automate it.
7. Send one causal action or deliberately specified short sequence, then recapture and verify the result. Start a new setup with a harmless capture and single-input smoke test.
8. Renew the claim before it expires. Release it in cleanup even after an error.

The R4a live gameplay executor additionally requires a provider operation that
can target and timestamp explicit button down, button up, and release-all on the
claimed window, enforce a caller-supplied automatic-release deadline
independently of the calling thread, and neutralize buttons on claim release or
expiry. At implementation time the installed `targeted_pointer_click`
operation was atomic and did not expose that contract. It remains suitable for
a deliberately labeled harmless broker smoke, but must not be treated as a
measured gameplay press/release or used to enable live policy input. Re-check
current provider capabilities each turn; if any targeted-edge or broker release
guarantee is absent, keep gameplay input fail-closed. Do not bypass this requirement with
PyAutoGUI, raw `xdotool`, a drag disguised as a click, or global input.

Do not inspect or control unrelated windows. Do not share fencing or lease tokens in logs. Do not use the headless coordinate lease unless normal exact capture plus targeted input has failed; that fallback temporarily takes global focus and requires the tool's explicit interference acknowledgment. Never drive global physical input as a shortcut.

Native injection will refuse unsafe conditions such as a locked session, held pointer buttons, an active pointer constraint or lock, or drag-and-drop. If safety changes, stop the probe, preserve the partial artifacts, and retry only after `session_status` becomes safe. The physical pointer seat is not independent even though ordinary targeted actions preserve the physical cursor, keyboard focus, and workspace.

## Reproducible experiment bundle

Create one directory under `reference/captures/<experiment-id>/` containing:

```text
metadata.json       environment, game, window, timing, and hypothesis
actions.jsonl       monotonic timestamp, action, local coordinates, and result
frames/             exact-window captures with ordered names
video/              optional continuous puzzle-region recording
result.rpy          copied immediately after the run, before another run
measurements.json   derived observations with units and uncertainty
notes.md            anomalies and links to follow-up probes
```

At minimum, `metadata.json` should include:

- experiment ID, date, agent/tool version, hypothesis, and expected discriminating outcomes;
- game executable, data, configuration, and Box2D DLL hashes;
- game version/language patch, Wine version and prefix, Hyprland version, and whether the native input component was loaded and safe;
- monitor refresh rate, window address/capture ID, window-local dimensions, scale, puzzle-region crop, capture cadence, and timestamp clock;
- disposable run path, initial replay/save hashes, seed when known, and exact launch command;
- the action-rate limit and all intentional delays.

Hash and copy the generated `.rpy` immediately. Record all game-tree files changed by the run with before/after hashes; the game is known to mutate files outside the replay directory as part of its broader presentation.

## Measurements to prioritize

### Replay and timing

- Generate separate traces for no input, cursor motion only, one weak click, one strong click, simultaneous buttons, held buttons, rapid clicks, pause, fast-forward, reset, and game over.
- Infer frame cadence and whether a replay record is simulation-tick, render-frame, or input-poll based.
- Identify padding/layout by game version, cursor coordinate encoding and clipping, button edge-versus-level semantics, and all currently unknown high bits.
- Replay the same trace at least ten times on this machine. Compare event timing and short-horizon trajectories to quantify deterministic agreement and the onset of divergence.

### Coordinate system and physics

- Calibrate the captured puzzle crop to game coordinates, including walls, floor, ceiling, side openings, and projectile origin.
- Measure scripted descent and the activation transition separately from dynamic gravity.
- Sweep weak and strong projectile launches over cursor positions and estimate velocity, latency, lifetime, collision radius, and click-frequency limits.
- Fit each shape's dimensions, fixtures, mass/inertia, restitution, friction, damping, angular response, sleeping, tunneling/continuous collision behavior, and collision filters.
- Use isolated contacts and the shipped Box2D wrapper probe to separate engine behavior from game rules.

### Lifecycle and scoring rules

- Measure fresh, confirmed, rotten, deleted, and side-ejected transitions; confirmation and deletion delays; direct-hit accounting; projectile-projectile effects; and simultaneous-contact ordering.
- Build exact tables for chain size versus score, gauge change, level multiplier, freshness/rot penalties, bonus-orb behavior, and edge cases such as a confirmed piece receiving another direct hit.
- Test persistent anchors in upper side openings and any off-screen body lifetime or contact behavior.

### Gauge, spawn, and difficulty

- Measure passive gauge decay, floor/rotten damage, clear recovery, caps, rounding, and game-over timing.
- Record spawn intervals, colors, shapes/sizes, positions, scripted velocities, bonus frequency, and how each distribution changes with score, level, elapsed time, or board state.
- Collect enough samples per phase to attach confidence intervals rather than copying an apparent pattern from one run.

## Experimental discipline

Prefer minimal probes where competing mechanics predict visibly different outcomes. Repeat boundary and simultaneous-event probes on both sides of the threshold. Keep raw evidence immutable; derive tracks and numeric summaries into new files. A surprising result is an explicit unknown until it reproduces. Every accepted constant should enter the mechanics configuration with a provenance label (`official`, `observed`, `community`, `inferred`, or `placeholder`), units, experiment IDs, sample count, and uncertainty.
