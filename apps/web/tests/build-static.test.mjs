import assert from "node:assert/strict";
import {execFileSync} from "node:child_process";
import {mkdtempSync, readFileSync, writeFileSync} from "node:fs";
import {tmpdir} from "node:os";
import path from "node:path";
import test from "node:test";
import {fileURLToPath} from "node:url";

const web = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

test("static build contains only the pinned exact backend", () => {
  const build = readFileSync(path.join(web, "build-static.sh"), "utf8");
  const fetch = readFileSync(path.join(web, "fetch-exact-runtime.sh"), "utf8");
  const prepare = readFileSync(path.join(web, "prepare-exact-runtime.sh"), "utf8");
  assert.match(build, /prepare-exact-runtime\.sh/);
  assert.match(build, /IRISU_EXACT_RUNTIME_DIR/);
  assert.doesNotMatch(build, /emcmake|irisu-wasm|PHYSICS_BACKEND=portable/);
  assert.match(fetch, /web-exact-runtime-lowlatency-v2-20260809/);
  assert.match(fetch, /2761932073e3be9a8663c1aa497b2bea8f81b5381c19196c8bccb71b8ace73d3/);
  assert.match(prepare, /4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261/);
  assert.match(prepare, /812b4876d588ae9539ac164d27d2ca5efd96d423428e4367f6145d36b79e9bba/);
  assert.match(prepare, /442aefadd8b65f65ccc036e93047f7181458d384ff07eb280ca0c92ecc194c6e/);
  assert.match(prepare, /8ef81521e81a5b2a764c305ac48dad997b28476bcd2fccbd1c9aed9603322854/);
  assert.match(prepare, /ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5/);
  assert.match(prepare, /73d1023eba1729d6aa6a9a3d3d52122c88e8f05b775caaa0557e042f68c34403/);
  assert.match(prepare, /d0317109d9cec024f5d01bac9cfd7399d699bcd48816e2373195ac2ee336949c/);
  assert.match(prepare, /IRISU_GUEST_BZIMAGE/);
  assert.match(prepare, /apps\/web\/guest\/build\.sh/);
  assert.match(prepare, /relink-exact-worker\.sh/);
  assert.match(prepare, /SOURCE\.relink-exact-worker\.sh/);
  assert.doesNotMatch(prepare, /cp -L .*\/(?:ld-linux|libc|libm|libgcc_s)/);
  assert.doesNotMatch(prepare, /i\.copy\.sh/);
});

test("exact worker is relinked without recompiling application objects", () => {
  const relink = readFileSync(path.join(web, "relink-exact-worker.sh"), "utf8");
  assert.match(relink, /i686-buildroot-linux-gnu-gcc/);
  assert.match(relink, /"\$object" "\$core" "\$host" "\$libstdcpp" -lm -ldl/);
  assert.match(relink, /--remove-section \.note\.gnu\.property/);
  assert.match(relink, /--strip-all/);
  assert.match(relink, /\/lib\/ld-linux\.so\.2/);
  assert.match(relink, /Library rpath: \[\$ORIGIN\]/);
  assert.doesNotMatch(relink, /\s-c\s/);
});

test("runtime preparation rejects unapproved exact inputs before downloading", () => {
  const directory = mkdtempSync(path.join(tmpdir(), "irisu-web-hash-test-"));
  const worker = path.join(directory, "worker");
  const workerObject = path.join(directory, "worker.o");
  const coreArchive = path.join(directory, "libirisu_core.a");
  const host = path.join(directory, "host.so");
  const wasm = path.join(directory, "v86.wasm");
  writeFileSync(worker, "wrong worker");
  writeFileSync(workerObject, "wrong worker object");
  writeFileSync(coreArchive, "wrong core archive");
  writeFileSync(host, "wrong host");
  writeFileSync(wasm, "wrong wasm");
  assert.throws(() => execFileSync(path.join(web, "prepare-exact-runtime.sh"),
    [path.join(directory, "runtime")], {
      env: {...process.env, IRISU_EXACT_WORKER: worker, IRISU_EXACT_HOST: host,
        IRISU_EXACT_WORKER_OBJECT: workerObject,
        IRISU_EXACT_CORE_ARCHIVE: coreArchive, IRISU_V86_WASM: wasm},
      encoding: "utf8", stdio: "pipe",
    }), error => error.status !== 0 && /FAILED/.test(error.stdout + error.stderr));
});

test("static build refuses to overwrite its source directory", () => {
  assert.throws(() => execFileSync(path.join(web, "build-static.sh"),
    [path.join(web, "static")], {encoding: "utf8", stdio: "pipe"}),
  error => error.status !== 0 &&
      /overlapping static-build paths/.test(error.stdout + error.stderr));
});
