#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import url from "node:url";
import { createHash } from "node:crypto";
import { spawn, spawnSync } from "node:child_process";
import { V86 } from "./runtime/libv86.mjs";

const here = url.fileURLToPath(new URL(".", import.meta.url));
const root = path.resolve(here, "../..");
const runtime = path.join(here, "runtime");
const defaults = {
  replay: path.join(root, "reference/replays/raw/internet/irisu_00041449_20100725_182435_7.rpy"),
  worker: path.join(root, "build-physics-integration-exact-multiworld-2/irisu-exact-worker"),
  runner: path.join(root, "build-physics-integration-exact-multiworld-2/irisu-exact-replay"),
};
const args = { ...defaults, limit: Infinity, output: null, transport: "serial" };
for (let index = 2; index < process.argv.length; index += 2) {
  const key = process.argv[index];
  const value = process.argv[index + 1];
  if (!value || !["--replay", "--worker", "--runner", "--limit", "--output", "--transport"].includes(key)) {
    throw new Error("usage: interactive-replay.mjs [--transport serial|virtio] [--replay path] [--worker path] [--runner path] [--limit ticks] [--output path]");
  }
  args[key.slice(2)] = key === "--limit" ? Number(value) : key === "--transport" ? value : path.resolve(value);
}
if (!(args.limit > 0)) throw new Error("--limit must be positive");
if (!["serial", "virtio"].includes(args.transport)) throw new Error("--transport must be serial or virtio");

const MAGIC = 0x43505249;
const VERSION = 1;
const rpcTimeoutMs = Number(process.env.IRISU_V86_RPC_TIMEOUT || 30000);
const expectedRuntime = Object.freeze({
  "libv86.js": "730d3e4e4bd1d9c7ede4a4995f23da87261a723b3524da9a75c90e6e0610f46b",
  "libv86.mjs": "408b0969f943dfd4d0350f6196404fc99fe676e853c36dc09a1959a0f9f751c2",
  "v86.wasm": "73d1023eba1729d6aa6a9a3d3d52122c88e8f05b775caaa0557e042f68c34403",
  "buildroot-bzimage68.bin": "389fb6e37c9f9f101232ad68b7177bced98caee9f7a531e99ea00b836833ea33",
  "seabios.bin": "73e3f359102e3a9982c35fce98eb7cd08f18303ac7f1ba6ebfbe6cdc1c244d98",
  "vgabios.bin": "a4bc0d80cc3ca028c73dafa8fee396b8d054ce87ebd8abfbd31b06b437607880",
});
const guestFiles = [
  "irisu-exact-worker",
  "libirisu_box2d_msvc_exact_multiworld.so",
  "ld-linux.so.2",
  "libc.so.6",
  "libm.so.6",
  "libgcc_s.so.1",
  "libstdc++.so.6",
];
const sha256 = bytes => createHash("sha256").update(bytes).digest("hex");
const frame = (opcode, requestId, payload = new Uint8Array()) => {
  const output = new Uint8Array(16 + payload.length);
  const view = new DataView(output.buffer);
  view.setUint32(0, MAGIC, true);
  view.setUint16(4, VERSION, true);
  view.setUint16(6, opcode, true);
  view.setUint32(8, requestId, true);
  view.setUint32(12, payload.length, true);
  output.set(payload, 16);
  return output;
};
const seedPayload = seed => {
  const output = new Uint8Array(8);
  new DataView(output.buffer).setBigUint64(0, BigInt(seed), true);
  return output;
};
const stepPayload = (word, index) => {
  const output = new Uint8Array(28);
  const view = new DataView(output.buffer);
  view.setUint32(0, word & 3, true);
  view.setFloat64(4, (word >>> 2) & 0x3ff, true);
  view.setFloat64(12, (word >>> 12) & 0x1ff, true);
  view.setUint32(20, 1, true);
  view.setUint32(24, index < 2 ? 1 : 0, true);
  return output;
};
const observation = bytes => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return {
    tick: Number(view.getBigUint64(0, true)),
    score: Number(view.getBigInt64(8, true)),
    gauge: Number(view.getBigInt64(16, true)),
    clears: Number(view.getBigUint64(32, true)),
    level: view.getUint32(88, true),
    highest_chain: view.getUint32(100, true),
    bodies: view.getUint32(104, true),
  };
};
const quantile = (sorted, proportion) => sorted[Math.min(sorted.length - 1, Math.floor(proportion * sorted.length))];
const summarize = values => {
  const sorted = [...values].sort((a, b) => a - b);
  const sum = values.reduce((left, right) => left + right, 0);
  return { mean: sum / values.length, p50: quantile(sorted, 0.5), p95: quantile(sorted, 0.95), p99: quantile(sorted, 0.99), max: sorted.at(-1) };
};
const firstDifference = (left, right) => {
  const size = Math.min(left.length, right.length);
  for (let index = 0; index < size; ++index) if (left[index] !== right[index]) return index;
  return left.length === right.length ? null : size;
};

class NativeRpc {
  constructor(executable) {
    this.nextId = 1;
    this.pending = new Map();
    this.buffer = Buffer.alloc(0);
    this.stderr = "";
    this.process = spawn(executable, [], { env: { ...process.env, IRISU_EXACT_CW: "0x27f" }, stdio: ["pipe", "pipe", "pipe"] });
    this.process.stderr.on("data", chunk => { this.stderr = (this.stderr + chunk).slice(-8192); });
    this.process.stdout.on("data", chunk => { this.buffer = Buffer.concat([this.buffer, chunk]); this.drain(); });
    this.process.on("error", error => { for (const pending of this.pending.values()) pending.reject(error); });
  }
  drain() {
    while (this.buffer.length >= 16) {
      const size = this.buffer.readUInt32LE(12);
      if (this.buffer.length < 16 + size) return;
      const header = this.buffer.subarray(0, 16);
      const payload = this.buffer.subarray(16, 16 + size);
      this.buffer = this.buffer.subarray(16 + size);
      const opcode = header.readUInt16LE(6);
      const requestId = header.readUInt32LE(8);
      const pending = this.pending.get(requestId);
      if (header.readUInt32LE(0) !== MAGIC || header.readUInt16LE(4) !== VERSION || !pending || pending.opcode !== opcode) {
        throw new Error(`invalid native response ${opcode}/${requestId}`);
      }
      this.pending.delete(requestId);
      const status = payload.readInt32LE(0);
      if (status) pending.reject(new Error(`native status ${status}: ${payload.subarray(4)}`));
      else pending.resolve(new Uint8Array(payload.subarray(4)));
    }
  }
  rpc(opcode, payload) {
    const requestId = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(requestId, { opcode, resolve, reject });
      this.process.stdin.write(frame(opcode, requestId, payload));
    });
  }
  destroy() {
    this.process.stdin.destroy();
    this.process.stdout.destroy();
    this.process.kill();
  }
}

class V86Rpc {
  constructor(transport) {
    this.transport = transport;
    this.nextId = 1;
    this.pending = new Map();
    this.bytes = [];
    this.bootText = "";
    this.ready = false;
    this.emulator = new V86({
      wasm_path: path.join(runtime, "v86.wasm"),
      memory_size: 128 * 1024 * 1024,
      vga_memory_size: 2 * 1024 * 1024,
      bios: { url: path.join(runtime, "seabios.bin") },
      vga_bios: { url: path.join(runtime, "vgabios.bin") },
      bzimage: { url: path.join(runtime, "buildroot-bzimage68.bin"), async: false },
      filesystem: {},
      virtio_console: transport === "virtio",
      cmdline: "console=ttyS0 tsc=reliable mitigations=off random.trust_cpu=on",
      autostart: false,
      disable_keyboard: true,
    });
    this.emulator.add_listener("serial0-output-byte", byte => {
      if (!this.ready) {
        this.bootText = (this.bootText + String.fromCharCode(byte)).slice(-8192);
        if (this.bootText.includes("__IRISU_RPC_READY__")) this.ready = true;
        return;
      }
      this.bytes.push(byte);
      this.drain();
    });
    if (transport === "virtio") this.emulator.add_listener("virtio-console0-output-bytes", bytes => {
      if (!this.ready) return;
      this.bytes.push(...bytes);
      this.drain();
    });
  }
  waitFor(predicate, label, timeout = 60000) {
    return new Promise((resolve, reject) => {
      const started = Date.now();
      const timer = setInterval(() => {
        if (predicate()) { clearInterval(timer); resolve(); }
        else if (Date.now() - started >= timeout) { clearInterval(timer); reject(new Error(`${label} timeout: ${this.bootText.slice(-1000)}`)); }
      }, 10);
    });
  }
  drain() {
    while (this.bytes.length >= 16) {
      const header = Uint8Array.from(this.bytes.slice(0, 16));
      const view = new DataView(header.buffer);
      const size = view.getUint32(12, true);
      if (this.bytes.length < 16 + size) return;
      this.bytes.splice(0, 16);
      const payload = Uint8Array.from(this.bytes.splice(0, size));
      const opcode = view.getUint16(6, true);
      const requestId = view.getUint32(8, true);
      const pending = this.pending.get(requestId);
      if (view.getUint32(0, true) !== MAGIC || view.getUint16(4, true) !== VERSION || !pending || pending.opcode !== opcode) {
        throw new Error(`invalid v86 response ${opcode}/${requestId}`);
      }
      this.pending.delete(requestId);
      const status = new DataView(payload.buffer).getInt32(0, true);
      if (status) pending.reject(new Error(`v86 status ${status}: ${new TextDecoder().decode(payload.subarray(4))}`));
      else pending.resolve(payload.subarray(4));
    }
  }
  send(bytes) {
    if (this.transport === "virtio") {
      this.emulator.bus.send("virtio-console0-input-bytes", bytes);
      return;
    }
    for (let offset = 0; offset < bytes.length; offset += 4096) {
      const chunk = bytes.subarray(offset, offset + 4096);
      this.emulator.serial0_send(String.fromCharCode(...chunk));
    }
  }
  rpc(opcode, payload) {
    const requestId = this.nextId++;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(requestId);
        reject(new Error(`${this.transport} RPC ${opcode}/${requestId} timeout`));
      }, rpcTimeoutMs);
      this.pending.set(requestId, {
        opcode,
        resolve: value => { clearTimeout(timer); resolve(value); },
        reject: error => { clearTimeout(timer); reject(error); },
      });
      this.send(frame(opcode, requestId, payload));
    });
  }
  async boot(artifacts) {
    await new Promise(resolve => this.emulator.add_listener("emulator-ready", resolve));
    for (const name of guestFiles) await this.emulator.create_file(`/${name}`, new Uint8Array(artifacts.guest[name].data));
    const started = performance.now();
    this.emulator.run();
    await this.waitFor(() => this.bootText.includes("~% "), "boot prompt");
    if (this.transport === "virtio") {
      this.emulator.serial0_send(
        "test -c /dev/hvc0 && " +
        "chmod +x /mnt/ld-linux.so.2 /mnt/irisu-exact-worker && cd / && " +
        "exec 3<>/dev/hvc0 && stty raw -echo <&3 && { " +
        "IRISU_EXACT_CW=0x27f /mnt/ld-linux.so.2 --library-path /mnt " +
        "/mnt/irisu-exact-worker <&3 >&3 2>/mnt/interactive-worker.err & " +
        "worker_pid=$!; sleep 1; " +
        "printf '\\137\\137IRISU\\137RPC\\137READY\\137\\137'; wait $worker_pid; }\n",
      );
    } else {
      this.emulator.serial0_send([
        "chmod +x /mnt/ld-linux.so.2 /mnt/irisu-exact-worker",
        "cd /",
        "stty raw -echo",
        "printf '\\137\\137IRISU\\137RPC\\137READY\\137\\137'",
        "IRISU_EXACT_CW=0x27f exec /mnt/ld-linux.so.2 --library-path /mnt /mnt/irisu-exact-worker",
      ].join(" && ") + "\n");
    }
    await this.waitFor(() => this.ready, "worker marker", 30000);
    return performance.now() - started;
  }
  destroy() { this.emulator.destroy(); }
}

const main = async () => {
  const replay = fs.readFileSync(args.replay);
  if (replay.length < 52 || (replay.length - 52) % 4) throw new Error("expected padded v2.03 replay");
  const frames = Math.min((replay.length - 52) / 4, args.limit);
  const artifacts = { runtime: {}, guest: {} };
  for (const [name, expected] of Object.entries(expectedRuntime)) {
    const data = fs.readFileSync(path.join(runtime, name));
    const actual = sha256(data);
    if (actual !== expected) throw new Error(`${name} hash mismatch: ${actual}`);
    artifacts.runtime[name] = { bytes: data.length, sha256: actual };
  }
  for (const name of guestFiles) {
    const data = fs.readFileSync(path.join(runtime, "guest", name));
    artifacts.guest[name] = { bytes: data.length, sha256: sha256(data), data };
  }
  const native = new NativeRpc(args.worker);
  const v86 = new V86Rpc(args.transport);
  try {
    const bootMs = await v86.boot(artifacts);
    await Promise.all([native.rpc(1), v86.rpc(1)]);
    const resetPayload = seedPayload(replay.readUInt32LE(0));
    const [nativeReset, v86Reset] = await Promise.all([native.rpc(2, resetPayload), v86.rpc(2, resetPayload)]);
    if (firstDifference(nativeReset, v86Reset) !== null) throw new Error("reset response mismatch");
    const responseHash = createHash("sha256");
    const latencies = [];
    const bodyCounts = [];
    let responseBytes = 0;
    let maximumResponseBytes = 0;
    let mismatch = null;
    let final = observation(v86Reset);
    const started = performance.now();
    for (let index = 0; index < frames; ++index) {
      const payload = stepPayload(replay.readUInt32LE(52 + index * 4), index);
      const tickStarted = performance.now();
      const [nativeBytes, v86Bytes] = await Promise.all([native.rpc(3, payload), v86.rpc(3, payload)]);
      latencies.push(performance.now() - tickStarted);
      const difference = firstDifference(nativeBytes, v86Bytes);
      if (difference !== null) {
        mismatch = {
          frame: index,
          byte: difference,
          native_bytes: nativeBytes.length,
          v86_bytes: v86Bytes.length,
          native_sha256: sha256(nativeBytes),
          v86_sha256: sha256(v86Bytes),
        };
        break;
      }
      const length = Buffer.allocUnsafe(4);
      length.writeUInt32LE(v86Bytes.length);
      responseHash.update(length).update(v86Bytes);
      responseBytes += v86Bytes.length;
      maximumResponseBytes = Math.max(maximumResponseBytes, v86Bytes.length);
      final = observation(v86Bytes);
      bodyCounts.push(final.bodies);
      if ((index + 1) % 5000 === 0) process.stderr.write(`interactive replay ${index + 1}/${frames}\n`);
    }
    const elapsedMs = performance.now() - started;
    const runner = spawnSync(args.runner, [args.replay], { env: { ...process.env, IRISU_EXACT_CW: "0x27f" }, encoding: "utf8", maxBuffer: 8 * 1024 * 1024 });
    if (runner.status !== 0) throw new Error(`reference runner failed: ${runner.stderr}`);
    const reference = JSON.parse(runner.stdout);
    const terminalFields = ["tick", "score", "gauge", "level", "highest_chain", "clears"];
    const terminalApplicable = frames === (replay.length - 52) / 4;
    const terminalMatches = !terminalApplicable || terminalFields.every(key => final[key] === reference[key]);
    const serializableArtifacts = {
      runtime: artifacts.runtime,
      guest: Object.fromEntries(Object.entries(artifacts.guest).map(([name, value]) => [name, { bytes: value.bytes, sha256: value.sha256 }])),
    };
    const result = {
      schema: 1,
      status: mismatch === null && terminalMatches && latencies.length === frames ? "pass" : "mismatch",
      scope: "one opcode-3 RPC and full variable-size transition response per replay tick",
      transport: args.transport === "virtio" ? "virtio-console /dev/hvc0 chunk events" : "16550 UART byte events",
      replay: { path: args.replay, bytes: replay.length, sha256: sha256(replay), available_frames: (replay.length - 52) / 4, tested_frames: frames },
      artifacts: { ...serializableArtifacts, native_worker: { path: args.worker, sha256: sha256(fs.readFileSync(args.worker)) } },
      parity: { compared_responses: latencies.length, first_mismatch: mismatch, stream_sha256: responseHash.digest("hex"), terminal_runner_comparison_applicable: terminalApplicable, terminal_matches_runner: terminalMatches, final, runner: Object.fromEntries(terminalFields.map(key => [key, reference[key]])) },
      performance: {
        boot_ms: bootMs,
        elapsed_ms: elapsedMs,
        ticks_per_second: latencies.length * 1000 / elapsedMs,
        response_bytes: responseBytes,
        response_megabytes_per_second: responseBytes / elapsedMs / 1000,
        maximum_response_bytes: maximumResponseBytes,
        response_latency_ms: summarize(latencies),
        sixty_hz_deadline_ms: 1000 / 60,
        sixty_hz_deadline_misses: latencies.filter(value => value > 1000 / 60).length,
        body_count: summarize(bodyCounts),
      },
    };
    const encoded = JSON.stringify(result, null, 2) + "\n";
    if (args.output) { fs.mkdirSync(path.dirname(args.output), { recursive: true }); fs.writeFileSync(args.output, encoded); }
    process.stdout.write(encoded);
    if (result.status !== "pass") process.exitCode = 1;
  } finally {
    native.destroy();
    v86.destroy();
  }
};

await main();
