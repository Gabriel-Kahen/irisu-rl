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
  assert.match(fetch, /web-exact-runtime-fastboot-20260809/);
  assert.match(fetch, /ccdb8dd5a855490e54934c6985f598e4abb7ec4414c616fc355522cee257b7f2/);
  assert.match(prepare, /4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261/);
  assert.match(prepare, /ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5/);
  assert.match(prepare, /73d1023eba1729d6aa6a9a3d3d52122c88e8f05b775caaa0557e042f68c34403/);
  assert.match(prepare, /681388b6db219fbb1dc63a678cd276d73c21bbb047cd8c7a6771fc4e567591c0/);
  assert.match(prepare, /IRISU_GUEST_BZIMAGE/);
  assert.match(prepare, /apps\/web\/guest\/build\.sh/);
  assert.doesNotMatch(prepare, /i\.copy\.sh/);
});

test("runtime preparation rejects unapproved exact inputs before downloading", () => {
  const directory = mkdtempSync(path.join(tmpdir(), "irisu-web-hash-test-"));
  const worker = path.join(directory, "worker");
  const host = path.join(directory, "host.so");
  const wasm = path.join(directory, "v86.wasm");
  writeFileSync(worker, "wrong worker");
  writeFileSync(host, "wrong host");
  writeFileSync(wasm, "wrong wasm");
  assert.throws(() => execFileSync(path.join(web, "prepare-exact-runtime.sh"),
    [path.join(directory, "runtime")], {
      env: {...process.env, IRISU_EXACT_WORKER: worker, IRISU_EXACT_HOST: host,
        IRISU_V86_WASM: wasm},
      encoding: "utf8", stdio: "pipe",
    }), error => error.status !== 0 && /FAILED/.test(error.stdout + error.stderr));
});

test("static build refuses to overwrite its source directory", () => {
  assert.throws(() => execFileSync(path.join(web, "build-static.sh"),
    [path.join(web, "static")], {encoding: "utf8", stdio: "pipe"}),
  error => error.status !== 0 &&
      /overlapping static-build paths/.test(error.stdout + error.stderr));
});
