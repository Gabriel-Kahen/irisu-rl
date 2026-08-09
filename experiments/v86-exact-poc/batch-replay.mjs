#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import url from "node:url";
import { createHash } from "node:crypto";
import { V86 } from "./runtime/libv86.mjs";

const directory = url.fileURLToPath(new URL(".", import.meta.url));
const runtime = path.join(directory, "runtime");
const timeoutSeconds = Number(process.env.IRISU_V86_REPLAY_TIMEOUT || 1800);
const replayPaths = process.argv.slice(2).map(value => path.resolve(value));
if (!replayPaths.length) throw new Error("usage: node batch-replay.mjs replay.rpy [...]");

const guestFiles = [
  "irisu-exact-replay",
  "libirisu_box2d_msvc_exact_multiworld.so",
  "ld-linux.so.2",
  "libc.so.6",
  "libm.so.6",
  "libgcc_s.so.1",
  "libstdc++.so.6",
];
const expectedRuntimeArtifacts = Object.freeze({
  "libv86.js": "730d3e4e4bd1d9c7ede4a4995f23da87261a723b3524da9a75c90e6e0610f46b",
  "libv86.mjs": "408b0969f943dfd4d0350f6196404fc99fe676e853c36dc09a1959a0f9f751c2",
  "v86.wasm": "73d1023eba1729d6aa6a9a3d3d52122c88e8f05b775caaa0557e042f68c34403",
  "buildroot-bzimage68.bin": "389fb6e37c9f9f101232ad68b7177bced98caee9f7a531e99ea00b836833ea33",
  "seabios.bin": "73e3f359102e3a9982c35fce98eb7cd08f18303ac7f1ba6ebfbe6cdc1c244d98",
  "vgabios.bin": "a4bc0d80cc3ca028c73dafa8fee396b8d054ce87ebd8abfbd31b06b437607880",
});
const sha256 = bytes => createHash("sha256").update(bytes).digest("hex");
const replayName = index => `replay-${index.toString().padStart(2, "0")}.rpy`;
const resultName = index => `result-${index.toString().padStart(2, "0")}.json`;
const metaName = index => `meta-${index.toString().padStart(2, "0")}.json`;
const errorName = index => `error-${index.toString().padStart(2, "0")}.txt`;

let serialTail = "";
let serialLine = "";
const completedAt = new Map();
let setupReady = false;

const emulator = new V86({
  wasm_path: path.join(runtime, "v86.wasm"),
  memory_size: 128 * 1024 * 1024,
  vga_memory_size: 2 * 1024 * 1024,
  bios: { url: path.join(runtime, "seabios.bin") },
  vga_bios: { url: path.join(runtime, "vgabios.bin") },
  bzimage: { url: path.join(runtime, "buildroot-bzimage68.bin"), async: false },
  filesystem: {},
  cmdline: "console=ttyS0 tsc=reliable mitigations=off random.trust_cpu=on",
  autostart: false,
  disable_keyboard: true,
});

const parseSerialLine = line => {
  const replay = /^__IRISU_REPLAY_DONE__:(\d+):(\d+)$/.exec(line);
  if (replay) completedAt.set(Number(replay[1]), performance.now());
  if (line === "__IRISU_BATCH_READY__") setupReady = true;
};

emulator.add_listener("serial0-output-byte", byte => {
  const character = String.fromCharCode(byte);
  serialTail = (serialTail + character).slice(-16384);
  if (character === "\n" || character === "\r") {
    if (serialLine) parseSerialLine(serialLine);
    serialLine = "";
  } else {
    serialLine += character;
  }
});

const waitFor = (predicate, label, timeoutMs) => new Promise((resolve, reject) => {
  const started = Date.now();
  const timer = setInterval(() => {
    if (predicate()) {
      clearInterval(timer);
      resolve();
    } else if (Date.now() - started >= timeoutMs) {
      clearInterval(timer);
      reject(new Error(`${label} timeout; serial tail=${JSON.stringify(serialTail)}`));
    }
  }, 20);
});

const shellQuote = value => `'${value.replaceAll("'", "'\\''")}'`;

try {
  await new Promise(resolve => emulator.add_listener("emulator-ready", resolve));
  const artifacts = { runtime: {}, guest: {} };
  for (const [name, expected] of Object.entries(expectedRuntimeArtifacts)) {
    const bytes = fs.readFileSync(path.join(runtime, name));
    const actual = sha256(bytes);
    if (actual !== expected) throw new Error(`${name} hash mismatch: ${actual}`);
    artifacts.runtime[name] = { bytes: bytes.length, sha256: actual };
  }
  for (const name of guestFiles) {
    const bytes = fs.readFileSync(path.join(runtime, "guest", name));
    artifacts.guest[name] = { bytes: bytes.length, sha256: sha256(bytes) };
    await emulator.create_file(`/${name}`, new Uint8Array(bytes));
  }
  const staged = [];
  for (const [index, replayPath] of replayPaths.entries()) {
    const bytes = fs.readFileSync(replayPath);
    const name = replayName(index);
    staged.push({ index, path: replayPath, name, bytes: bytes.length, sha256: sha256(bytes) });
    await emulator.create_file(`/${name}`, new Uint8Array(bytes));
    await emulator.create_file(`/${resultName(index)}`, new Uint8Array());
    await emulator.create_file(`/${metaName(index)}`, new Uint8Array());
    await emulator.create_file(`/${errorName(index)}`, new Uint8Array());
  }

  const bootStart = performance.now();
  emulator.run();
  await waitFor(() => serialTail.includes("~% "), "boot prompt", 60000);
  const bootMs = performance.now() - bootStart;

  emulator.serial0_send(
    "chmod +x /mnt/ld-linux.so.2 /mnt/irisu-exact-replay; " +
    "cd /; printf '__IRISU_BATCH_READY__\\n'\n",
  );
  await waitFor(() => setupReady, "batch setup", 30000);
  const batchStart = performance.now();
  for (const entry of staged) {
    const id = entry.index.toString().padStart(2, "0");
    const commands = [`start=$(cut -d' ' -f1 /proc/uptime)`];
    commands.push(
      `IRISU_EXACT_CW=0x27f /mnt/ld-linux.so.2 --library-path /mnt /mnt/irisu-exact-replay ` +
      `${shellQuote(`/mnt/${entry.name}`)} >${shellQuote(`/mnt/${resultName(entry.index)}`)} ` +
      `2>${shellQuote(`/mnt/${errorName(entry.index)}`)}`,
    );
    commands.push("status=$?");
    commands.push(`finish=$(cut -d' ' -f1 /proc/uptime)`);
    commands.push(
      `printf '{"exit_code":%s,"start_uptime_seconds":"%s","end_uptime_seconds":"%s"}\\n' ` +
      `"$status" "$start" "$finish" >${shellQuote(`/mnt/${metaName(entry.index)}`)}`,
    );
    commands.push(`printf '__IRISU_REPLAY_DONE__:${id}:%s\\n' "$status"`);
    emulator.serial0_send(`${commands.join("; ")}\n`);
    const remaining = timeoutSeconds * 1000 - (performance.now() - batchStart);
    if (remaining <= 0) throw new Error(`batch timeout; serial tail=${JSON.stringify(serialTail)}`);
    await waitFor(() => completedAt.has(entry.index), `replay ${id}`, remaining);
  }
  const batchMs = performance.now() - batchStart;

  const entries = [];
  let previousCompletion = batchStart;
  for (const entry of staged) {
    const resultBytes = Buffer.from(await emulator.read_file(`/${resultName(entry.index)}`));
    const metaBytes = Buffer.from(await emulator.read_file(`/${metaName(entry.index)}`));
    const errorBytes = Buffer.from(await emulator.read_file(`/${errorName(entry.index)}`));
    const meta = JSON.parse(metaBytes.toString("utf8"));
    const completion = completedAt.get(entry.index);
    const wallMs = completion === undefined ? null : completion - previousCompletion;
    if (completion !== undefined) previousCompletion = completion;
    const guestSeconds = Number(meta.end_uptime_seconds) - Number(meta.start_uptime_seconds);
    entries.push({
      ...entry,
      exit_code: meta.exit_code,
      wall_ms: wallMs,
      guest_elapsed_seconds: guestSeconds,
      stdout_bytes: resultBytes.length,
      stdout_sha256: sha256(resultBytes),
      stderr: errorBytes.toString("utf8").trim(),
      result: meta.exit_code === 0 ? JSON.parse(resultBytes.toString("utf8")) : null,
    });
  }
  console.log(JSON.stringify({
    schema: 1,
    status: entries.every(entry => entry.exit_code === 0) ? "ok" : "runner_error",
    boot_ms: bootMs,
    batch_ms: batchMs,
    artifacts,
    entries,
  }));
} finally {
  emulator.destroy();
}
