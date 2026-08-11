import assert from "node:assert/strict";
import test from "node:test";

import {RestartGate} from "../static/restart-gate.mjs";

test("restart gate exposes pending state and ignores overlapping restarts", async () => {
  const pendingChanges = [];
  const gate = new RestartGate(pending => pendingChanges.push(pending));
  let resolveRestart;
  let calls = 0;
  const first = gate.run(() => {
    calls++;
    return new Promise(resolve => { resolveRestart = resolve; });
  });

  assert.equal(gate.pending, true);
  assert.deepEqual(pendingChanges, [true]);
  assert.equal(await gate.run(() => { calls++; return true; }), false);
  assert.equal(calls, 1);

  resolveRestart(true);
  assert.equal(await first, true);
  assert.equal(gate.pending, false);
  assert.deepEqual(pendingChanges, [true, false]);
});

test("restart gate restores controls after a failed restart", async () => {
  const pendingChanges = [];
  const gate = new RestartGate(pending => pendingChanges.push(pending));

  await assert.rejects(gate.run(async () => {
    throw new Error("restart failed");
  }), /restart failed/);

  assert.equal(gate.pending, false);
  assert.deepEqual(pendingChanges, [true, false]);
});
