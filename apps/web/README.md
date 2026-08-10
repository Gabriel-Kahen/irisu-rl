# Playable web app

This asset-free browser client renders the real headless simulator at 50 Hz.
Its open-bottom U-shaped well follows the measured v2.03 mode-0 geometry.

- Left click or `W`: weak shot
- Right click or `S`: strong shot
- Shift + click: both shots
- Mouse wheel down: fast-forward (wheel up stops)
- Space: pause/resume
- In replay mode, Left/Right: step backward/forward without changing play state
- Replay speed: 1x, 2x, 4x, or 8x
- `R`: restart with a new random seed
- **play replay**: open an original v2.03 normal-mode `.rpy` file
- **save replay**: download the completed run from the game-over dialog

Touch taps fire weak shots; desktop players can right-click or press `S` for a
strong shot.

## Build the static exact app

Build the existing 32-bit exact worker and provide its MSVC9 r58 host, then run:

```bash
IRISU_EXACT_WORKER=/path/to/irisu-exact-worker \
IRISU_EXACT_HOST=/path/to/libirisu_box2d_msvc_exact_multiworld.so \
  apps/web/build-static.sh
```

Runtime preparation performs a linker-only restage of that build's pinned
worker object and `libirisu_core.a` against the guest toolchain. If those files
do not share the worker's standard CMake build directory, set
`IRISU_EXACT_WORKER_OBJECT` and `IRISU_EXACT_CORE_ARCHIVE` explicitly.

This writes the GitHub Pages artifact to `apps/web/dist`. The static site boots
the exact i386 worker under v86 inside a Web Worker and requires no application
server or cross-origin isolation. Dependencies are pinned and cached in
`build-web/downloads`; generated guest binaries remain outside source control.
The client overlaps emulator and guest-engine downloads, uses v86 fast boot,
reuses parity-verified loader and C runtime libraries already in the guest
image, and shows
the current startup phase until the first exact state is available.
The workspace's standard exact-worker paths are used when the two environment
variables are omitted. The build reproducibly compiles the pinned v86
CORE-MATH patch; `IRISU_V86_WASM` may point at a prebuilt artifact with the
required hash. It also builds the pinned project-owned Linux guest; a prebuilt
copy with the required hash may be supplied as `IRISU_GUEST_BZIMAGE`. See
`EXACT_RUNTIME.md` for provenance and licensing.

Serve the completed directory with any ordinary static file host for local
testing, for example:

```bash
python3 -m http.server 8000 --directory apps/web/dist
```

Open <http://127.0.0.1:8000>. Gameplay and physics remain entirely in the
browser; this process only serves immutable files.

Saved replays use the original v2.03 52-byte header and per-20 ms input-word
format. The exact actions accepted by the simulator are recorded, including
release and idle ticks, and terminal metadata comes from the first recorded
finish. Replay playback applies the original two-frame startup edge rule.
The transport can pause, step without interrupting playback, scrub, and play at
1x, 2x, 4x, or 8x; long backward seeks are rebuilt from
the seed and immutable input stream when their exact observations have left the
bounded in-memory cache.

GitHub Pages downloads the hash-pinned runtime from the
[`web-exact-runtime-lowlatency-v2-20260809`](https://github.com/Gabriel-Kahen/irisu-rl/releases/tag/web-exact-runtime-lowlatency-v2-20260809)
release with `fetch-exact-runtime.sh`, verifies both the archive and its embedded
runtime manifest, and passes the prepared directory through
`IRISU_EXACT_RUNTIME_DIR`.

## Legacy native API server

`apps/web/server.py` and the Dockerfile are retained for native API diagnostics.
They use the portable shared library and are not the exact browser app or its
deployment path. The static app above does not call their JSON API.

```bash
docker build -f apps/web/Dockerfile -t irisu-web .
docker run --rm -p 8000:8000 irisu-web
```

`IRISU_SEED` sets the initial seed and `IRISU_CLONE_LIBRARY` selects a custom
native build. `GET /healthz` is available for health checks.

The diagnostic server owns one shared in-memory game. Before exposing its API
as a multi-user service, add per-session game instances or isolate each player
in a separate process.
