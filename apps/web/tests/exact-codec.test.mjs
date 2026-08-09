import assert from "node:assert/strict";
import test from "node:test";

import {
  CONTROL_WORD, EXACT_CONFIG_HASH, EXACT_LIBRARY_SHA256, decodeHello, decodeReset, decodeStep,
  encodeReset, encodeStep,
} from "../static/exact-codec.mjs";

const view = bytes => new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

function observationFixture(extra = 0) {
  const bytes = new Uint8Array(112 + 100 + extra);
  const data = view(bytes);
  data.setBigUint64(0, 7n, true);
  data.setBigInt64(8, 42n, true);
  data.setBigInt64(16, 2993n, true);
  data.setBigInt64(24, 3000n, true);
  data.setBigUint64(32, 1n, true);
  [40, 48, 56, 64, 72, 80].forEach((offset, index) =>
    data.setFloat64(offset, index + 0.5, true));
  data.setUint32(88, 2, true);
  data.setUint32(92, 5, true);
  data.setUint32(96, 99, true);
  data.setUint32(100, 3, true);
  data.setUint32(104, 1, true);
  data.setUint8(110, 1);
  const body = 112;
  data.setBigUint64(body, 12n, true);
  data.setBigInt64(body + 8, -1n, true);
  data.setBigUint64(body + 16, 4n, true);
  [24, 32, 40, 48, 56, 64, 72].forEach((offset, index) =>
    data.setFloat64(body + offset, index + 10.25, true));
  data.setUint32(body + 80, 19, true);
  data.setInt32(body + 84, 4, true);
  data.setUint32(body + 88, 8, true);
  data.setUint32(body + 92, 2, true);
  data.setUint8(body + 96, 0);
  data.setUint8(body + 97, 2);
  data.setUint8(body + 98, 2);
  return bytes;
}

test("encodes exact reset and one-tick step requests", () => {
  assert.equal(view(encodeReset(0xfedcba98)).getBigUint64(0, true), 0xfedcba98n);
  const step = encodeStep(3, 12.5, -4.25);
  assert.equal(step.byteLength, 28);
  assert.equal(view(step).getUint32(0, true), 3);
  assert.equal(view(step).getFloat64(4, true), 12.5);
  assert.equal(view(step).getFloat64(12, true), -4.25);
  assert.equal(view(step).getUint32(20, true), 1);
  assert.equal(view(step).getUint32(24, true), 0);
  assert.equal(view(encodeStep(1, 10, 20, true)).getUint32(24, true), 1);
});

test("validates and decodes the exact Hello identity", () => {
  const strings = ["exact-msvc9-r58-multiworld-forward", "gcc", EXACT_LIBRARY_SHA256];
  const encoded = strings.map(value => new TextEncoder().encode(value));
  const bytes = new Uint8Array(32 + encoded.reduce((sum, item) => sum + 2 + item.length, 0));
  const data = view(bytes);
  data.setUint32(0, 1, true);
  data.setUint32(4, 32, true);
  data.setUint32(8, 196, true);
  data.setBigUint64(16, BigInt(EXACT_CONFIG_HASH), true);
  data.setUint32(24, CONTROL_WORD, true);
  data.setUint32(28, 1, true);
  let offset = 32;
  for (const item of encoded) {
    data.setUint16(offset, item.length, true);
    bytes.set(item, offset + 2);
    offset += item.length + 2;
  }
  assert.deepEqual(decodeHello(bytes), {
    protocol_version: 1, pointer_bits: 32, body_capacity: 196, pid: 0,
    config_hash: EXACT_CONFIG_HASH, x87_control_word: CONTROL_WORD, process_model: 1,
    backend: strings[0], compiler: strings[1], exact_library_sha256: strings[2],
  });
  data.setBigUint64(16, 123n, true);
  assert.throws(() => decodeHello(bytes), /identity or ABI/);
});

test("decodes Reset observations into the browser JSON contract", () => {
  const state = decodeReset(observationFixture());
  assert.deepEqual(state.events, []);
  assert.equal(state.observation.tick, 7);
  assert.equal(state.observation.gauge_max, 3000);
  assert.deepEqual(state.observation.field, {
    x: .5, y: 1.5, width: 2.5, height: 3.5,
    side_wall_top: 4.5, side_wall_bottom: 5.5,
  });
  assert.deepEqual(state.observation.difficulty,
    {active_colors: 5, spawn_interval_ticks: 99});
  assert.deepEqual(state.observation.bodies[0], {
    id: 19, kind: "piece", shape: "triangle", lifecycle: "confirmed", color: 4,
    x: 10.25, y: 11.25, vx: 12.25, vy: 13.25, angle: 14.25,
    angular_velocity: 15.25, size: 16.25, chain_id: 8, projectile_hits: 2,
    age_ticks: 12, remaining_lifetime: -1, rot_timer: 4,
  });
});

test("decodes Step diagnostics and variable-length events", () => {
  const detail = new TextEncoder().encode("next level");
  const bytes = observationFixture(84 + 36 + detail.length);
  const data = view(bytes);
  const transition = 212;
  data.setBigInt64(transition, 9n, true);
  data.setBigUint64(transition + 8, 1n, true);
  data.setBigUint64(transition + 16, BigInt(EXACT_CONFIG_HASH), true);
  data.setBigUint64(transition + 24, 2n, true);
  data.setBigInt64(transition + 32, 40n, true);
  data.setBigUint64(transition + 40, 3n, true);
  data.setBigInt64(transition + 48, 42n, true);
  data.setBigUint64(transition + 56, 4n, true);
  data.setUint32(transition + 64, 5, true);
  data.setUint32(transition + 68, 6, true);
  data.setUint32(transition + 72, 7, true);
  data.setUint32(transition + 76, 8, true);
  data.setUint8(transition + 82, 1);
  const event = transition + 84;
  data.setBigUint64(event, 7n, true);
  data.setBigUint64(event + 8, 10n, true);
  data.setBigInt64(event + 16, 2n, true);
  data.setUint32(event + 24, 11, true);
  data.setUint32(event + 28, 12, true);
  data.setUint16(event + 32, detail.length, true);
  data.setUint8(event + 34, 13);
  bytes.set(detail, event + 36);
  const state = decodeStep(bytes);
  assert.equal(state.reward, 9);
  assert.equal(state.diagnostics.config_hash, EXACT_CONFIG_HASH);
  assert.equal(state.diagnostics.latest_final_level, 8);
  assert.deepEqual(state.events, [{
    tick: 7, sequence: 10, kind: 13, kind_name: "level_changed",
    a: 11, b: 12, value: 2, detail: "next level",
  }]);
});
