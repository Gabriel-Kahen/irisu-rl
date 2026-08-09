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
  assert.match(html, /id="runtimeLoadingStatus"/);
  assert.match(css, /@keyframes runtime-spin/);
  assert.match(css, /prefers-reduced-motion: reduce/);
  assert.match(app, /if \(ui\.runtimeLoadingStatus\)/);
  assert.match(app, /ui\.runtimeLoading\?\.classList/);
  assert.match(app, /if \(ui\.runtimeLoading\) ui\.runtimeLoading\.hidden = true/);
});

test("exact worker overlaps guest downloads with emulator startup", () => {
  const worker = readFileSync(path.join(web, "static/exact-worker.js"), "utf8");
  assert.match(worker, /Promise\.all\(guestFiles\.map/);
  assert.match(worker, /Promise\.all\(\[guestDownloads, emulatorReady\]\)/);
  assert.match(worker, /fastboot: true/);
  assert.match(worker, /rdinit=\/irisu-direct-init/);
  assert.match(worker, /waitForProtocolReady/);
  assert.doesNotMatch(worker, /waitForPrompt/);
  assert.match(worker, /memory_size: 128 \* 1024 \* 1024/);
  const guestList = worker.match(/const guestFiles = \[([\s\S]*?)\];/)?.[1] || "";
  assert.doesNotMatch(guestList, /libm\.so\.6|libgcc_s\.so\.1/);
  assert.match(guestList, /ld-linux\.so\.2/);
  assert.match(guestList, /libc\.so\.6/);
});

test("browser guest uses a minimal executable direct init", () => {
  const board = path.join(web, "guest", "buildroot-external", "board", "irisu");
  const initPath = path.join(board, "rootfs-overlay", "irisu-direct-init");
  const init = readFileSync(initPath, "utf8");
  const config = readFileSync(path.join(board, "linux.config"), "utf8");
  assert.equal(statSync(initPath).mode & 0o111, 0o111);
  assert.match(init, /mount -t proc proc \/proc/);
  assert.match(init, /mount -t 9p .* host9p \/mnt/);
  assert.match(init, /exec \/mnt\/ld-linux\.so\.2/);
  assert.doesNotMatch(init, /mount -t (?:devtmpfs|sysfs)|exec \/bin\/sh/);
  for (const option of ["ACPI", "INET", "WIRELESS", "INPUT", "VT", "DEVTMPFS", "SYSFS", "TMPFS"]) {
    assert.match(config, new RegExp(`# CONFIG_${option} is not set`));
  }
});
