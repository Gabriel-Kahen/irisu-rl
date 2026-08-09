#!/usr/bin/env node
import fs from "node:fs";
import path from "node:path";
import url from "node:url";
import { V86 } from "./runtime/libv86.mjs";

const here = url.fileURLToPath(new URL(".", import.meta.url));
const runtime = path.join(here, "runtime");
const outputPath = process.argv.length === 4 && process.argv[2] === "--output" ? path.resolve(process.argv[3]) : null;
if (process.argv.length !== 2 && outputPath === null) throw new Error("usage: virtio-console-smoke.mjs [--output path]");
const emit = result => {
  const encoded = JSON.stringify(result) + "\n";
  if (outputPath) { fs.mkdirSync(path.dirname(outputPath), { recursive: true }); fs.writeFileSync(outputPath, encoded); }
  process.stdout.write(encoded);
};
let serial = "";
let chunks = 0;
let bytes = 0;
const output = [];
const emulator = new V86({
  wasm_path: path.join(runtime, "v86.wasm"),
  memory_size: 128 * 1024 * 1024,
  vga_memory_size: 2 * 1024 * 1024,
  bios: { url: path.join(runtime, "seabios.bin") },
  vga_bios: { url: path.join(runtime, "vgabios.bin") },
  bzimage: { url: path.join(runtime, "buildroot-bzimage68.bin"), async: false },
  filesystem: {},
  virtio_console: true,
  cmdline: "console=ttyS0 tsc=reliable mitigations=off random.trust_cpu=on",
  autostart: false,
  disable_keyboard: true,
});
emulator.add_listener("serial0-output-byte", byte => { serial = (serial + String.fromCharCode(byte)).slice(-16384); });
emulator.add_listener("virtio-console0-output-bytes", data => { ++chunks; bytes += data.length; output.push(...data); });
const waitFor = (predicate, timeout) => new Promise((resolve, reject) => {
  const started = Date.now();
  const timer = setInterval(() => {
    if (predicate()) { clearInterval(timer); resolve(); }
    else if (Date.now() - started > timeout) { clearInterval(timer); reject(new Error("timeout")); }
  }, 20);
});

try {
  await new Promise(resolve => emulator.add_listener("emulator-ready", resolve));
  const started = performance.now();
  emulator.run();
  await waitFor(() => serial.includes("~% "), 30000);
  emulator.serial0_send("if test -c /dev/hvc0; then printf '\\137\\137HVC\\137PRESENT\\137\\137\\n'; else printf '\\137\\137HVC\\137MISSING\\137\\137\\n'; fi\n");
  await waitFor(() => serial.includes("__HVC_PRESENT__") || serial.includes("__HVC_MISSING__"), 5000);
  if (!serial.includes("__HVC_PRESENT__")) throw new Error("/dev/hvc0 missing");
  emulator.serial0_send("exec 3<>/dev/hvc0; stty raw -echo <&3; dd bs=1 count=16 <&3 >&3 2>/dev/null & sleep 1; printf '\\137\\137HVC\\137ECHO\\137READY\\137\\137\\n'\n");
  await waitFor(() => serial.includes("__HVC_ECHO_READY__"), 5000);
  const sent = Uint8Array.from({ length: 16 }, (_, index) => index);
  emulator.bus.send("virtio-console0-input-bytes", sent);
  await waitFor(() => output.length >= sent.length, 5000);
  const binaryExact = Buffer.from(output).equals(Buffer.from(sent));
  emit({
    status: binaryExact ? "pass" : "mismatch",
    boot_ms: performance.now() - started,
    hvc0: serial.includes("__HVC_PRESENT__"),
    output_chunks: chunks,
    output_bytes: bytes,
    binary_exact: binaryExact,
    echoed_hex: Buffer.from(output).toString("hex"),
  });
  if (!binaryExact) process.exitCode = 1;
} catch (error) {
  emit({ status: "unavailable", error: String(error), serial_tail: serial.slice(-2000), output_chunks: chunks, output_bytes: bytes });
  process.exitCode = 1;
} finally {
  emulator.destroy();
}
