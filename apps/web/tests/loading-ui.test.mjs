import assert from "node:assert/strict";
import {readFileSync, statSync} from "node:fs";
import path from "node:path";
import test from "node:test";
import {fileURLToPath} from "node:url";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

test("exact startup has an accessible animated loading state", () => {
  const html = readFileSync(path.join(web, "static/index.html"), "utf8");
  const css = readFileSync(path.join(web, "static/styles.css"), "utf8");
  const app = readFileSync(path.join(web, "static/app.js"), "utf8");
  assert.match(html, /id="runtimeLoading"[^>]*role="status"[^>]*aria-live="polite"/);
  assert.match(html, /<strong>LOADING EMULATOR\.\.\.<\/strong>/);
  assert.doesNotMatch(html, /runtimeLoadingStatus|runtime-loading-meter/);
  assert.match(css, /@keyframes runtime-spin/);
  assert.match(css, /\.runtime-loading[\s\S]*background: #000/);
  assert.match(css, /\.runtime-loading strong[\s\S]*color: #fff/);
  assert.doesNotMatch(css, /runtime-pulse|runtime-scan/);
  assert.match(css, /prefers-reduced-motion: reduce/);
  assert.doesNotMatch(app, /runtimeLoadingStatus|updateRuntimeLoading/);
  assert.match(app, /if \(ui\.runtimeLoading\) ui\.runtimeLoading\.hidden = true/);
});

test("exact worker downloads runtime in parallel and keeps a persistent fallback cache", () => {
  const worker = readFileSync(path.join(web, "static/exact-worker.js"), "utf8");
  assert.match(worker, /Promise\.all\(guestFiles\.map/);
  assert.match(worker, /const \[bios, vgaBios, bzimage, files, wasmModule\] = await Promise\.all/);
  assert.match(worker, /compilingWasm = fetchRuntimeBuffer\("v86\.wasm"\)\.then\(WebAssembly\.compile\)/);
  assert.match(worker, /wasm_fn: imports => WebAssembly\.instantiate\(wasmModule/);
  assert.match(worker, /bios: \{buffer: bios\}/);
  assert.match(worker, /bzimage: \{buffer: bzimage\}/);
  assert.match(worker, /typeof caches === "undefined" \? Promise\.resolve\(null\)/);
  assert.match(worker, /cache\.match\(request\)/);
  assert.match(worker, /cache\.put\(request, response\.clone\(\)\)\.catch/);
  assert.match(worker, /evictRuntime\("v86\.wasm"\)/);
  assert.match(worker, /fastboot: true/);
  assert.match(worker, /rdinit=\/irisu-direct-init/);
  assert.match(worker, /waitForProtocolReady/);
  assert.doesNotMatch(worker, /waitForPrompt/);
  assert.match(worker, /memory_size: 128 \* 1024 \* 1024/);
  const guestList = worker.match(/const guestFiles = \[([\s\S]*?)\];/)?.[1] || "";
  assert.doesNotMatch(guestList, /libm\.so\.6|libgcc_s\.so\.1|ld-linux\.so\.2|libc\.so\.6/);
  assert.match(guestList, /irisu-exact-worker/);
  assert.match(guestList, /libirisu_box2d_msvc_exact_multiworld\.so/);
  assert.match(guestList, /libstdc\+\+\.so\.6/);
});

test("runtime preloads use the exact immutable worker URLs", () => {
  const html = readFileSync(path.join(web, "static/index.html"), "utf8");
  const worker = readFileSync(path.join(web, "static/exact-worker.js"), "utf8");
  const version = worker.match(/const RUNTIME_VERSION = "([^"]+)"/)?.[1];
  assert.ok(version);
  assert.match(html, new RegExp(`exact-runtime/v86\\.wasm\\?v=${version}`));
  assert.match(html, new RegExp(`exact-runtime/buildroot-bzimage68\\.bin\\?v=${version}`));
});

test("browser module cache-bust chain stays aligned", () => {
  const html = readFileSync(path.join(web, "static/index.html"), "utf8");
  const app = readFileSync(path.join(web, "static/app.js"), "utf8");
  const runtime = readFileSync(path.join(web, "static/exact-runtime.js"), "utf8");
  const version = html.match(/app\.js\?v=([0-9a-z]+)/)?.[1];
  assert.ok(version);
  assert.match(app, new RegExp(`exact-runtime\\.js\\?v=${version}`));
  assert.match(app, new RegExp(`restart-gate\\.mjs\\?v=${version}`));
  assert.match(app, new RegExp(`replay\\.mjs\\?v=${version}`));
  assert.match(runtime, new RegExp(`exact-codec\\.mjs\\?v=${version}`));
  assert.match(runtime, new RegExp(`replay\\.mjs\\?v=${version}`));
});

test("restart shows the emulator loading state and awaits the fresh worker", () => {
  const app = readFileSync(path.join(web, "static/app.js"), "utf8");
  assert.match(app, /new RestartGate\(\(pending\) => \{/);
  assert.match(app, /runtimeLoading\) ui\.runtimeLoading\.hidden = !pending/);
  assert.match(app, /ui\.restart\.disabled = pending/);
  assert.match(app, /ui\.again\.disabled = pending/);
  assert.match(app, /const restarted = await game\.restart\(seed\)/);
  assert.doesNotMatch(app, /game\.restart\(seed\);[\s\S]{0,120}syncUi\(\)/);
});

test("replay transport exposes keyboard stepping and playback speeds", () => {
  const html = readFileSync(path.join(web, "static/index.html"), "utf8");
  const app = readFileSync(path.join(web, "static/app.js"), "utf8");
  const css = readFileSync(path.join(web, "static/styles.css"), "utf8");
  assert.match(html, /id="replaySpeed"[\s\S]*value="1"[\s\S]*value="2"[\s\S]*value="4"[\s\S]*value="8"/);
  assert.match(html, /aria-label="Jump back 5 seconds">−5s/);
  assert.match(html, /aria-label="Jump forward 5 seconds">\+5s/);
  assert.match(app, /\["Space", "ArrowLeft", "ArrowRight"\]/);
  assert.match(app, /const replaySkipFrames = 5_000 \/ REPLAY_TICK_MS/);
  assert.match(app, /stepReplay\(-replaySkipFrames\)/);
  assert.match(app, /stepReplay\(replaySkipFrames\)/);
  assert.match(app, /let replayScrubbing = false/);
  assert.match(app, /let replayScrubTarget = null/);
  assert.match(app, /if \(!replayScrubbing && replayScrubTarget === null\)/);
  assert.match(app, /clampReplayScrubFrame\([\s\S]*buffered_frames/);
  assert.match(app, /seekReplay\(frame, \{preserveRunning: true\}\)/);
  assert.match(css, /\.replay-speed select/);
});

test("browser guest uses a minimal executable direct init", () => {
  const board = path.join(web, "guest", "buildroot-external", "board", "irisu");
  const initPath = path.join(board, "rootfs-overlay", "irisu-direct-init");
  const init = readFileSync(initPath, "utf8");
  const config = readFileSync(path.join(board, "linux.config"), "utf8");
  assert.equal(statSync(initPath).mode & 0o111, 0o111);
  assert.match(init, /mount -t proc proc \/proc/);
  assert.match(init, /mount -t 9p .* host9p \/mnt/);
  assert.match(init, /exec \/mnt\/irisu-exact-worker/);
  assert.doesNotMatch(init, /\/mnt\/(?:ld-linux|libc\.so)/);
  assert.doesNotMatch(init, /mount -t (?:devtmpfs|sysfs)|exec \/bin\/sh/);
  for (const option of ["ACPI", "INET", "WIRELESS", "INPUT", "VT", "DEVTMPFS", "SYSFS", "TMPFS"]) {
    assert.match(config, new RegExp(`# CONFIG_${option} is not set`));
  }
});
