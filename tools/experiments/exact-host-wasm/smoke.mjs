import { readFile } from "node:fs/promises";

if (process.argv.length !== 3) {
  throw new Error("usage: node smoke.mjs /path/to/exact-host-smoke.wasm");
}

const bytes = await readFile(process.argv[2]);
const { instance } = await WebAssembly.instantiate(bytes, {});
const result = instance.exports.smoke_world_test() >>> 0;
const instructions = instance.exports.smoke_instruction_count() >>> 0;
const imageSize = instance.exports.host_image_size() >>> 0;

if (result !== 0x40490fdb) {
  throw new Error(`wrong angular-velocity bits: 0x${result.toString(16)}`);
}
if (instructions !== 26) {
  throw new Error(`wrong translated instruction count: ${instructions}`);
}
if (imageSize === 0) {
  throw new Error("empty packaged guest image");
}

console.log(
  JSON.stringify({
    status: "ok",
    angularVelocityBits: `0x${result.toString(16)}`,
    translatedX86Instructions: instructions,
    guestImageBytes: imageSize,
  }),
);
