/* global V86 */
"use strict";

importScripts("./runtime/libv86.js");

const MAGIC = 0x43505249;
const VERSION = 1;
const READY_MARKER = "__IRISU_RPC_READY__";
const guestFiles = [
  "irisu-exact-worker",
  "libirisu_box2d_msvc_exact_multiworld.so",
  "ld-linux.so.2",
  "libc.so.6",
  "libm.so.6",
  "libgcc_s.so.1",
  "libstdc++.so.6",
];

let emulator;
let bootText = "";
let protocolReady = false;
let nextRequestId = 1;
const responseBytes = [];
const requests = new Map();
let framingSynchronized = false;

const progress = message => postMessage({ type: "progress", message });
const fatal = error => postMessage({ type: "fatal", error: error.stack || String(error) });

const sendBytes = bytes => {
  const chunkSize = 4096;
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    const chunk = bytes.subarray(offset, offset + chunkSize);
    let encoded = "";
    for (let index = 0; index < chunk.length; index++) encoded += String.fromCharCode(chunk[index]);
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
      throw new Error(`invalid response header: magic=${magic.toString(16)} version=${version} size=${size}`);
    }
    if (responseBytes.length < 16 + size) return;
    responseBytes.splice(0, 16);
    const response = new Uint8Array(responseBytes.splice(0, size));
    const status = new DataView(response.buffer).getInt32(0, true);
    const request = requests.get(requestId);
    if (!request || request.opcode !== opcode) throw new Error(`unexpected response ${opcode}/${requestId}`);
    requests.delete(requestId);
    const content = response.slice(4);
    if (status !== 0) {
      request.reject(new Error(`worker status ${status}: ${new TextDecoder().decode(content)}`));
    } else {
      request.resolve(content);
    }
  }
};

const onSerialByte = byte => {
  if (!protocolReady) {
    bootText = (bootText + String.fromCharCode(byte)).slice(-8192);
    if (bootText.includes(READY_MARKER)) {
      protocolReady = true;
      progress("Exact worker is running; starting binary RPC");
      postMessage({ type: "ready" });
    }
    return;
  }
  responseBytes.push(byte);
  try {
    drainResponses();
  } catch (error) {
    fatal(error);
  }
};

const rpc = (opcode, payload) => new Promise((resolve, reject) => {
  if (!protocolReady) {
    reject(new Error("exact worker is not ready"));
    return;
  }
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

const waitForPrompt = () => new Promise((resolve, reject) => {
  const deadline = setTimeout(() => reject(new Error(`boot prompt timeout; tail=${JSON.stringify(bootText.slice(-500))}`)), 60000);
  const poll = setInterval(() => {
    if (bootText.includes("~% ")) {
      clearTimeout(deadline);
      clearInterval(poll);
      resolve();
    }
  }, 25);
});

const start = async () => {
  progress("Loading v86 and exact-worker files");
  emulator = new V86({
    wasm_path: "./runtime/v86.wasm",
    memory_size: 128 * 1024 * 1024,
    vga_memory_size: 2 * 1024 * 1024,
    bios: { url: "./runtime/seabios.bin" },
    vga_bios: { url: "./runtime/vgabios.bin" },
    bzimage: { url: "./runtime/buildroot-bzimage68.bin", async: false },
    filesystem: {},
    cmdline: "console=ttyS0 tsc=reliable mitigations=off random.trust_cpu=on",
    autostart: false,
    disable_keyboard: true,
  });
  emulator.add_listener("serial0-output-byte", onSerialByte);
  await new Promise(resolve => emulator.add_listener("emulator-ready", resolve));

  for (const name of guestFiles) {
    const response = await fetch(`./runtime/guest/${name}`);
    if (!response.ok) throw new Error(`failed to fetch guest/${name}: ${response.status}`);
    await emulator.create_file(`/${name}`, new Uint8Array(await response.arrayBuffer()));
  }

  progress("Booting the 32-bit guest kernel");
  emulator.run();
  await waitForPrompt();
  progress("Launching exact-worker and switching serial to binary mode");
  const command = [
    "chmod +x /mnt/ld-linux.so.2 /mnt/irisu-exact-worker",
    "cd /",
    "stty raw -echo",
    "printf '\\137\\137IRISU\\137RPC\\137READY\\137\\137'",
    "IRISU_EXACT_CW=0x27f exec /mnt/ld-linux.so.2 --library-path /mnt /mnt/irisu-exact-worker",
  ].join(" && ");
  emulator.serial0_send(`${command}\n`);
};

onmessage = event => {
  if (event.data.type !== "rpc") return;
  const payload = new Uint8Array(event.data.payload);
  rpc(event.data.opcode, payload).then(content => {
    postMessage({ type: "response", messageId: event.data.messageId, payload: content.buffer }, [content.buffer]);
  }, error => {
    postMessage({ type: "response", messageId: event.data.messageId, error: error.stack || String(error) });
  });
};

start().catch(fatal);
