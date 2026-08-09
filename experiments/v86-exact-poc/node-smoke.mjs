#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import url from "node:url";
import { createHash } from "node:crypto";
import { V86 } from "./runtime/libv86.mjs";

const directory = url.fileURLToPath(new URL(".", import.meta.url));
const runtime = process.env.IRISU_V86_RUNTIME || path.join(directory, "runtime");
const MAGIC = 0x43505249;
const VERSION = 1;
const READY_MARKER = "__IRISU_RPC_READY__";
const NATIVE_FINAL_SHA256 = "dcb234c9ffb3c0140ebfa98c735c5568b36603999885a733482db7e012f3f9e1";
const guestFiles = [
  "irisu-exact-worker",
  "libirisu_box2d_msvc_exact_multiworld.so",
  "ld-linux.so.2",
  "libc.so.6",
  "libm.so.6",
  "libgcc_s.so.1",
  "libstdc++.so.6",
];

let bootText = "";
let protocolReady = false;
let nextRequestId = 1;
const responseBytes = [];
const requests = new Map();
let framingSynchronized = false;

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

const sendBytes = bytes => {
  for (let offset = 0; offset < bytes.length; offset += 4096) {
    const chunk = bytes.subarray(offset, offset + 4096);
    let encoded = "";
    for (const byte of chunk) encoded += String.fromCharCode(byte);
    emulator.serial0_send(encoded);
  }
};

const drainResponses = () => {
  if (!framingSynchronized) {
    while (responseBytes.length >= 4 &&
      !(responseBytes[0] === 0x49 && responseBytes[1] === 0x52 && responseBytes[2] === 0x50 && responseBytes[3] === 0x43)) {
      responseBytes.shift();
    }
    if (responseBytes.length < 4) return;
    framingSynchronized = true;
  }
  while (responseBytes.length >= 16) {
    const header = new Uint8Array(responseBytes.slice(0, 16));
    const view = new DataView(header.buffer);
    const magic = view.getUint32(0, true);
    const version = view.getUint16(4, true);
    const opcode = view.getUint16(6, true);
    const requestId = view.getUint32(8, true);
    const size = view.getUint32(12, true);
    if (magic !== MAGIC || version !== VERSION || size < 4 || size > 4 * 1024 * 1024) {
      throw new Error(`invalid response header: ${Buffer.from(header).toString("hex")}; boot tail=${JSON.stringify(bootText.slice(-500))}`);
    }
    if (responseBytes.length < 16 + size) return;
    responseBytes.splice(0, 16);
    const response = new Uint8Array(responseBytes.splice(0, size));
    const status = new DataView(response.buffer).getInt32(0, true);
    const request = requests.get(requestId);
    if (!request || request.opcode !== opcode) throw new Error(`unexpected response ${opcode}/${requestId}`);
    requests.delete(requestId);
    const content = response.slice(4);
    if (status) request.reject(new Error(`worker status ${status}: ${new TextDecoder().decode(content)}`));
    else request.resolve(content);
  }
};

emulator.add_listener("serial0-output-byte", byte => {
  if (!protocolReady) {
    bootText = (bootText + String.fromCharCode(byte)).slice(-8192);
    if (bootText.includes(READY_MARKER)) protocolReady = true;
    return;
  }
  responseBytes.push(byte);
  try {
    drainResponses();
  } catch (error) {
    for (const request of requests.values()) request.reject(error);
    requests.clear();
  }
});

const waitFor = (predicate, label, timeoutMs = 60000) => new Promise((resolve, reject) => {
  const started = Date.now();
  const timer = setInterval(() => {
    if (predicate()) {
      clearInterval(timer);
      resolve();
    } else if (Date.now() - started >= timeoutMs) {
      clearInterval(timer);
      reject(new Error(`${label} timeout; boot tail=${JSON.stringify(bootText.slice(-1000))}`));
    }
  }, 10);
});

const rpc = (opcode, payload = new Uint8Array()) => new Promise((resolve, reject) => {
  const requestId = nextRequestId++;
  const frame = new Uint8Array(16 + payload.length);
  const view = new DataView(frame.buffer);
  view.setUint32(0, MAGIC, true);
  view.setUint16(4, VERSION, true);
  view.setUint16(6, opcode, true);
  view.setUint32(8, requestId, true);
  view.setUint32(12, payload.length, true);
  frame.set(payload, 16);
  requests.set(requestId, { opcode, resolve, reject });
  sendBytes(frame);
});

const seedPayload = seed => {
  const payload = new Uint8Array(8);
  new DataView(payload.buffer).setBigUint64(0, BigInt(seed), true);
  return payload;
};

const stepPayload = () => {
  const payload = new Uint8Array(28);
  const view = new DataView(payload.buffer);
  view.setUint32(0, 0, true);
  view.setFloat64(4, 0, true);
  view.setFloat64(12, 0, true);
  view.setUint32(20, 1, true);
  view.setUint32(24, 0, true);
  return payload;
};

const observationPrefix = bytes => {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  return {
    tick: Number(view.getBigUint64(0, true)),
    score: Number(view.getBigInt64(8, true)),
    gauge: Number(view.getBigInt64(16, true)),
    level: view.getUint32(88, true),
    bodies: view.getUint32(104, true),
  };
};

try {
  await new Promise(resolve => emulator.add_listener("emulator-ready", resolve));
  for (const name of guestFiles) {
    await emulator.create_file(`/${name}`, new Uint8Array(fs.readFileSync(path.join(runtime, "guest", name))));
  }
  const bootStart = performance.now();
  emulator.run();
  await waitFor(() => bootText.includes("~% "), "boot prompt");
  const command = [
    "chmod +x /mnt/ld-linux.so.2 /mnt/irisu-exact-worker",
    "cd /",
    "stty raw -echo",
    "printf '\\137\\137IRISU\\137RPC\\137READY\\137\\137'",
    "IRISU_EXACT_CW=0x27f exec /mnt/ld-linux.so.2 --library-path /mnt /mnt/irisu-exact-worker",
  ].join(" && ");
  emulator.serial0_send(`${command}\n`);
  await waitFor(() => protocolReady, "worker marker", 30000);
  const bootMs = performance.now() - bootStart;

  const hello = await rpc(1);
  const helloView = new DataView(hello.buffer, hello.byteOffset, hello.byteLength);
  const helloPrefix = {
    protocol: helloView.getUint32(0, true),
    pointerBits: helloView.getUint32(4, true),
    bodyCapacity: helloView.getUint32(8, true),
    controlWord: `0x${helloView.getUint32(24, true).toString(16)}`,
  };

  const resetStart = performance.now();
  const reset = observationPrefix(await rpc(2, seedPayload(0)));
  const resetMs = performance.now() - resetStart;

  const observeCount = 20;
  let observeBytes = 0;
  const observeStart = performance.now();
  for (let index = 0; index < observeCount; index++) observeBytes += (await rpc(4)).byteLength;
  const observeMs = performance.now() - observeStart;

  const stepCount = 20;
  let final;
  let finalBytes;
  const stepStart = performance.now();
  for (let index = 0; index < stepCount; index++) {
    finalBytes = await rpc(3, stepPayload());
    final = observationPrefix(finalBytes);
  }
  const stepMs = performance.now() - stepStart;
  const finalHash = createHash("sha256").update(finalBytes).digest("hex");
  if (finalHash !== NATIVE_FINAL_SHA256) throw new Error(`native reference mismatch: ${finalHash}`);


  console.log(JSON.stringify({
    status: "pass",
    boot_ms: bootMs,
    hello: helloPrefix,
    reset,
    reset_ms: resetMs,
    observe: {
      count: observeCount,
      ms_per_rpc: observeMs / observeCount,
      bytes_per_second: observeBytes * 1000 / observeMs,
    },
    step: {
      count: stepCount,
      ms_per_tick: stepMs / stepCount,
      ticks_per_second: stepCount * 1000 / stepMs,
      final_sha256: finalHash,
      final,
    },
  }, null, 2));
} finally {
  emulator.destroy();
}
