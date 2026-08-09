/* global V86 */
"use strict";

importScripts("./exact-runtime/libv86.js?v=20260809j");

const MAGIC = 0x43505249;
const VERSION = 1;
const RUNTIME_VERSION = "20260809j";
const RUNTIME_CACHE_PREFIX = "irisu-exact-runtime-";
const RUNTIME_CACHE = `${RUNTIME_CACHE_PREFIX}${RUNTIME_VERSION}`;
const READY_MARKER = "__IRISU_RPC_READY__";
const GUEST_ERROR_MARKER = "__IRISU_GUEST_ERROR__:";
const guestFiles = [
  "irisu-exact-worker",
  "libirisu_box2d_msvc_exact_multiworld.so",
  "libstdc++.so.6",
];

let emulator;
let bootText = "";
let protocolReady = false;
let framingSynchronized = false;
let nextRequestId = 1;
const responseBytes = [];
const requests = new Map();

const progress = message => postMessage({type: "progress", message});

const runtimeCache = typeof caches === "undefined" ? Promise.resolve(null) :
  caches.open(RUNTIME_CACHE).then(cache => {
    // Runtime URLs are immutable and versioned. Retain only the current set.
    caches.keys().then(names => Promise.all(names
      .filter(name => name.startsWith(RUNTIME_CACHE_PREFIX) && name !== RUNTIME_CACHE)
      .map(name => caches.delete(name)))).catch(() => {});
    return cache;
  }).catch(() => null);

async function fetchRuntimeBuffer(path) {
  const url = `./exact-runtime/${path}?v=${RUNTIME_VERSION}`;
  const request = new Request(url);
  const cache = await runtimeCache;
  if (cache) {
    try {
      const cached = await cache.match(request);
      if (cached) return await cached.arrayBuffer();
    } catch (_) {
      // A denied or unreadable CacheStorage entry must not prevent startup.
      try { await cache.delete(request); } catch (_) { /* Best effort. */ }
    }
  }
  const response = await fetch(request);
  if (!response.ok) throw new Error(`could not load exact-runtime/${path}`);
  if (cache) cache.put(request, response.clone()).catch(() => {});
  return response.arrayBuffer();
}

async function evictRuntime(path) {
  const cache = await runtimeCache;
  if (!cache) return;
  try { await cache.delete(new Request(`./exact-runtime/${path}?v=${RUNTIME_VERSION}`)); }
  catch (_) { /* Best effort. */ }
}

function fail(error) {
  const detail = error?.stack || String(error);
  for (const request of requests.values()) request.reject(new Error(detail));
  requests.clear();
  postMessage({type: "fatal", error: detail});
}

function sendBytes(bytes) {
  for (let offset = 0; offset < bytes.length; offset += 4096) {
    const chunk = bytes.subarray(offset, offset + 4096);
    let encoded = "";
    for (const byte of chunk) encoded += String.fromCharCode(byte);
    emulator.serial0_send(encoded);
  }
}

function drainResponses() {
  if (!framingSynchronized) {
    while (responseBytes.length >= 4 &&
      !(responseBytes[0] === 0x49 && responseBytes[1] === 0x52 &&
        responseBytes[2] === 0x50 && responseBytes[3] === 0x43)) {
      responseBytes.shift();
    }
    if (responseBytes.length < 4) return;
    framingSynchronized = true;
  }
  while (responseBytes.length >= 16) {
    const header = Uint8Array.from(responseBytes.slice(0, 16));
    const view = new DataView(header.buffer);
    const magic = view.getUint32(0, true);
    const version = view.getUint16(4, true);
    const opcode = view.getUint16(6, true);
    const requestId = view.getUint32(8, true);
    const size = view.getUint32(12, true);
    if (magic !== MAGIC || version !== VERSION || size < 4 || size > 4 * 1024 * 1024) {
      throw new Error(`invalid response header ${magic.toString(16)}/${version}/${size}`);
    }
    if (responseBytes.length < 16 + size) return;
    responseBytes.splice(0, 16);
    const response = Uint8Array.from(responseBytes.splice(0, size));
    const status = new DataView(response.buffer).getInt32(0, true);
    const request = requests.get(requestId);
    if (!request || request.opcode !== opcode) {
      throw new Error(`unexpected exact-worker response ${opcode}/${requestId}`);
    }
    requests.delete(requestId);
    const content = response.slice(4);
    if (status) request.reject(new Error(
      `exact-worker status ${status}: ${new TextDecoder().decode(content)}`));
    else request.resolve(content);
  }
}

function onSerialByte(byte) {
  if (!protocolReady) {
    bootText = (bootText + String.fromCharCode(byte)).slice(-8192);
    if (bootText.includes(READY_MARKER)) {
      protocolReady = true;
      progress("Exact worker is ready");
      postMessage({type: "ready"});
    }
    return;
  }
  responseBytes.push(byte);
  try { drainResponses(); }
  catch (error) { fail(error); }
}

function rpc(opcode, payload) {
  return new Promise((resolve, reject) => {
    if (!protocolReady) return reject(new Error("exact worker is not ready"));
    const requestId = nextRequestId++;
    const frame = new Uint8Array(16 + payload.length);
    const view = new DataView(frame.buffer);
    view.setUint32(0, MAGIC, true);
    view.setUint16(4, VERSION, true);
    view.setUint16(6, opcode, true);
    view.setUint32(8, requestId, true);
    view.setUint32(12, payload.length, true);
    frame.set(payload, 16);
    requests.set(requestId, {opcode, resolve, reject});
    sendBytes(frame);
  });
}

function waitForProtocolReady() {
  return new Promise((resolve, reject) => {
    const deadline = setTimeout(() => {
      clearInterval(poll);
      reject(new Error(`guest boot timeout; console tail=${JSON.stringify(bootText.slice(-500))}`));
    }, 60000);
    const poll = setInterval(() => {
      const errorAt = bootText.indexOf(GUEST_ERROR_MARKER);
      if (errorAt >= 0) {
        clearTimeout(deadline);
        clearInterval(poll);
        reject(new Error(bootText.slice(errorAt).split("\n", 1)[0]));
        return;
      }
      if (!protocolReady) return;
      clearTimeout(deadline);
      clearInterval(poll);
      resolve();
    }, 25);
  });
}

async function start() {
  progress("Downloading emulator and game engine…");
  // Begin every large transfer together. Compiling WASM while the kernel and
  // guest files download removes the serial waterfall in v86's URL loader.
  const compilingWasm = fetchRuntimeBuffer("v86.wasm").then(WebAssembly.compile)
    .catch(async error => {
      await evictRuntime("v86.wasm");
      throw error;
    });
  const [bios, vgaBios, bzimage, files, wasmModule] = await Promise.all([
    fetchRuntimeBuffer("seabios.bin"),
    fetchRuntimeBuffer("vgabios.bin"),
    fetchRuntimeBuffer("buildroot-bzimage68.bin"),
    Promise.all(guestFiles.map(async name => ({
      name, bytes: new Uint8Array(await fetchRuntimeBuffer(`guest/${name}`)),
    }))),
    compilingWasm,
  ]);
  emulator = new V86({
    wasm_fn: imports => WebAssembly.instantiate(wasmModule, imports)
      .then(instance => instance.exports).catch(async error => {
        await evictRuntime("v86.wasm");
        throw error;
      }),
    memory_size: 128 * 1024 * 1024,
    vga_memory_size: 2 * 1024 * 1024,
    bios: {buffer: bios},
    vga_bios: {buffer: vgaBios},
    bzimage: {buffer: bzimage},
    filesystem: {},
    cmdline: "rdinit=/irisu-direct-init console=ttyS0 quiet loglevel=0 tsc=reliable mitigations=off random.trust_cpu=on",
    autostart: false,
    fastboot: true,
    disable_keyboard: true,
    disable_mouse: true,
    disable_speaker: true,
  });
  emulator.add_listener("serial0-output-byte", onSerialByte);
  const emulatorReady = new Promise(resolve => emulator.add_listener("emulator-ready", resolve));
  await emulatorReady;
  progress("Preparing exact game engine…");
  for (const {name, bytes} of files) {
    await emulator.create_file(`/${name}`, bytes);
  }
  progress("Starting exact game engine…");
  emulator.run();
  await waitForProtocolReady();
}

onmessage = event => {
  const message = event.data;
  if (message.type === "close") {
    emulator?.destroy();
    close();
    return;
  }
  if (message.type !== "rpc") return;
  rpc(message.opcode, new Uint8Array(message.payload)).then(content => {
    postMessage({type: "response", messageId: message.messageId,
      payload: content.buffer}, [content.buffer]);
  }, error => {
    postMessage({type: "response", messageId: message.messageId,
      error: error?.stack || String(error)});
  });
};

start().catch(fail);
