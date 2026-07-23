# Playable web app

This asset-free browser client renders the real headless simulator at 50 Hz.
Its open-bottom U-shaped well follows the measured v2.03 mode-0 geometry.

## Run locally

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
python3 apps/web/server.py
```

Open <http://127.0.0.1:8000>.

- Left click or `W`: weak shot
- Right click or `S`: strong shot
- Shift + click: both shots
- Space: pause/resume
- `R`: restart with a new random seed

Touch taps fire weak shots; desktop players can right-click or press `S` for a
strong shot.

## Build the static WebAssembly app

Install the Emscripten SDK, then run:

```bash
apps/web/build-static.sh
```

This writes the GitHub Pages artifact to `apps/web/dist`. The browser module is
single-threaded so it does not require cross-origin isolation headers. Import
the default factory from `irisu-wasm.js`, await it, and use `cwrap` with the exported
`irisu_web_create`, `irisu_web_destroy`, `irisu_web_reset`, and
`irisu_web_step` functions. Reset and step return JSON containing
`{observation, events}`.

## Host later

The app serves its static frontend and JSON API from one process, so it can be
deployed without CORS configuration. Hosting platforms should build from the
repository root and provide `HOST=0.0.0.0` plus their assigned `PORT`:

```bash
docker build -f apps/web/Dockerfile -t irisu-web .
docker run --rm -p 8000:8000 irisu-web
```

`IRISU_SEED` sets the initial seed and `IRISU_CLONE_LIBRARY` selects a custom
native build. `GET /healthz` is available for health checks.

The current server owns one shared in-memory game. Before exposing it as a
multi-user service, add per-session game instances or isolate each player in a
separate process.
