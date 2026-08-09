#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { V86 } from "../../experiments/v86-exact-poc/runtime/libv86.mjs";

const runtime = path.resolve("experiments/v86-exact-poc/runtime");
const wasmPath = process.env.IRISU_V86_WASM ? path.resolve(process.env.IRISU_V86_WASM) : path.join(runtime, "v86.wasm");
const [probePath, corpusPath] = process.argv.slice(2).map(value => path.resolve(value));
if (!probePath || !corpusPath) throw new Error("usage: node v86_probe.mjs PROBE CORPUS");
const sha256 = bytes => createHash("sha256").update(bytes).digest("hex");
let serial = "";
const emulator = new V86({
  wasm_path: wasmPath, memory_size: 128 << 20,
  vga_memory_size: 2 << 20, bios: { url: path.join(runtime, "seabios.bin") },
  vga_bios: { url: path.join(runtime, "vgabios.bin") },
  bzimage: { url: path.join(runtime, "buildroot-bzimage68.bin"), async: false },
  filesystem: {}, cmdline: "console=ttyS0", autostart: false, disable_keyboard: true,
});
emulator.add_listener("serial0-output-byte", byte => serial = (serial + String.fromCharCode(byte)).slice(-8192));
const waitFor = (text, timeout = 60000) => new Promise((resolve, reject) => {
  const started = Date.now();
  const timer = setInterval(() => {
    if (serial.includes(text)) { clearInterval(timer); resolve(); }
    else if (Date.now() - started > timeout) { clearInterval(timer); reject(new Error(`timeout: ${serial}`)); }
  }, 20);
});

try {
  await new Promise(resolve => emulator.add_listener("emulator-ready", resolve));
  for (const [guest, host] of [
    ["probe", probePath], ["corpus", corpusPath],
    ["ld.so", path.join(runtime, "guest/ld-linux.so.2")],
    ["libc.so.6", path.join(runtime, "guest/libc.so.6")],
  ]) await emulator.create_file(`/${guest}`, new Uint8Array(fs.readFileSync(host)));
  await emulator.create_file("/result", new Uint8Array());
  emulator.run();
  await waitFor("~% ");
  emulator.serial0_send("chmod +x /mnt/probe; /mnt/probe /mnt/corpus /mnt/result; printf '__PROBE''_DONE__:%s\\n' $?\n");
  await waitFor("__PROBE_DONE__:", 300000);
  const status = /__PROBE_DONE__:(\d+)/.exec(serial)?.[1];
  const result = Buffer.from(await emulator.read_file("/result"));
  if (status !== "0") throw new Error(`guest probe exited ${status} after ${result.length} result bytes: ${serial}`);
  console.log(JSON.stringify({ bytes: result.length, sha256: sha256(result),
    base64: result.toString("base64") }));
} finally {
  emulator.destroy();
}
