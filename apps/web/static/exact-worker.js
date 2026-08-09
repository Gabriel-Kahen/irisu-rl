/* global V86 */
"use strict";

importScripts("./exact-runtime/libv86.js?v=20260809e");

const MAGIC = 0x43505249;
const VERSION = 1;
const READY_MARKER = "__IRISU_RPC_READY__";
const guestFiles = [
  "irisu-exact-worker",
  "libirisu_box2d_msvc_exact_multiworld.so",
  "ld-linux.so.2",
  "libc.so.6",
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

function waitForPrompt() {
  return new Promise((resolve, reject) => {
    const deadline = setTimeout(() => {
      clearInterval(poll);
      reject(new Error(`guest boot timeout; console tail=${JSON.stringify(bootText.slice(-500))}`));
    }, 60000);
    const poll = setInterval(() => {
      if (!bootText.includes("~% ")) return;
      clearTimeout(deadline);
      clearInterval(poll);
      resolve();
    }, 25);
  });
}

async function start() {
  progress("Downloading emulator and game engine…");
  const guestDownloads = Promise.all(guestFiles.map(async name => {
    const response = await fetch(`./exact-runtime/guest/${name}?v=20260809e`);
    if (!response.ok) throw new Error(`could not load exact-runtime/guest/${name}`);
    return {name, bytes: new Uint8Array(await response.arrayBuffer())};
  }));
  emulator = new V86({
    wasm_path: "./exact-runtime/v86.wasm?v=20260809e",
    memory_size: 128 * 1024 * 1024,
    vga_memory_size: 2 * 1024 * 1024,
    bios: {url: "./exact-runtime/seabios.bin?v=20260809e"},
    vga_bios: {url: "./exact-runtime/vgabios.bin?v=20260809e"},
    bzimage: {url: "./exact-runtime/buildroot-bzimage68.bin?v=20260809e", async: false},
    filesystem: {},
    cmdline: "rdinit=/irisu-init console=ttyS0 quiet loglevel=0 tsc=reliable mitigations=off random.trust_cpu=on",
    autostart: false,
    fastboot: true,
    disable_keyboard: true,
    disable_mouse: true,
    disable_speaker: true,
  });
  emulator.add_listener("serial0-output-byte", onSerialByte);
  const emulatorReady = new Promise(resolve => emulator.add_listener("emulator-ready", resolve));
  const [files] = await Promise.all([guestDownloads, emulatorReady]);
  progress("Preparing exact game engine…");
  for (const {name, bytes} of files) {
    await emulator.create_file(`/${name}`, bytes);
  }
  progress("Starting exact simulation…");
  emulator.run();
  await waitForPrompt();
  progress("Launching game engine…");
  const command = [
    "chmod +x /mnt/ld-linux.so.2 /mnt/irisu-exact-worker",
    "cd /",
    "stty raw -echo",
    "printf '\\137\\137IRISU\\137RPC\\137READY\\137\\137'",
    "IRISU_EXACT_CW=0x27f exec /mnt/ld-linux.so.2 --library-path /mnt /mnt/irisu-exact-worker",
  ].join(" && ");
  emulator.serial0_send(`${command}\n`);
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
