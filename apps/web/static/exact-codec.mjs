export const MAGIC = 0x43505249;
export const VERSION = 1;
export const OPCODE = Object.freeze({hello: 1, reset: 2, step: 3});
export const BODY_CAPACITY = 196;
export const CONTROL_WORD = 0x027f;
export const EXACT_LIBRARY_SHA256 = "ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5";
export const EXACT_CONFIG_HASH = "17009678407634462320";

const bodyKinds = ["piece", "projectile", "bonus"];
const shapes = ["circle", "box", "triangle"];
const lifecycles = [
  "scripted_falling", "dynamic_fresh", "confirmed", "rotten", "deleted",
];
const eventKinds = [
  "invalid_action", "spawned", "shot_fired", "activated", "contact",
  "confirmed", "chain_joined", "cleared", "rotten", "ejected",
  "destroyed", "gauge_changed", "score_changed", "level_changed",
  "game_over", "projectile_hit", "projectile_contact",
  "held_input_ignored", "level_completed",
];

const textDecoder = new TextDecoder("utf-8", {fatal: true});

function fail(message) { throw new Error(`invalid exact-worker response: ${message}`); }

function safeNumber(value, label) {
  const number = Number(value);
  if (!Number.isSafeInteger(number)) fail(`${label} exceeds JavaScript's exact integer range`);
  return number;
}

function bytesView(bytes) {
  return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
}

function enumName(values, index, label) {
  if (index >= values.length) fail(`unknown ${label} ${index}`);
  return values[index];
}

function readString(view, bytes, state) {
  if (state.offset + 2 > bytes.byteLength) fail("truncated Hello string length");
  const size = view.getUint16(state.offset, true);
  state.offset += 2;
  if (state.offset + size > bytes.byteLength) fail("truncated Hello string");
  const value = textDecoder.decode(bytes.subarray(state.offset, state.offset + size));
  state.offset += size;
  return value;
}

export function encodeReset(seed) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
    throw new RangeError("seed must be a uint32");
  }
  const bytes = new Uint8Array(8);
  bytesView(bytes).setBigUint64(0, BigInt(seed), true);
  return bytes;
}

export function encodeStep(kind, x, y, suppressFreshEdges = false) {
  if (!Number.isInteger(kind) || kind < 0 || kind > 3) {
    throw new RangeError("action kind must be in [0, 3]");
  }
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    throw new TypeError("action coordinates must be finite");
  }
  const bytes = new Uint8Array(28);
  const view = bytesView(bytes);
  view.setUint32(0, kind, true);
  view.setFloat64(4, x, true);
  view.setFloat64(12, y, true);
  view.setUint32(20, 1, true);
  view.setUint32(24, suppressFreshEdges ? 1 : 0, true);
  return bytes;
}

export function decodeHello(bytes) {
  if (bytes.byteLength < 32) fail("truncated Hello payload");
  const view = bytesView(bytes);
  const state = {offset: 32};
  const hello = {
    protocol_version: view.getUint32(0, true),
    pointer_bits: view.getUint32(4, true),
    body_capacity: view.getUint32(8, true),
    pid: view.getUint32(12, true),
    config_hash: view.getBigUint64(16, true).toString(),
    x87_control_word: view.getUint32(24, true),
    process_model: view.getUint32(28, true),
    backend: readString(view, bytes, state),
    compiler: readString(view, bytes, state),
    exact_library_sha256: readString(view, bytes, state),
  };
  if (state.offset !== bytes.byteLength) fail("Hello payload has trailing bytes");
  if (hello.protocol_version !== VERSION || hello.pointer_bits !== 32 ||
      hello.body_capacity !== BODY_CAPACITY ||
      hello.x87_control_word !== CONTROL_WORD || hello.process_model !== 1 ||
      hello.config_hash !== EXACT_CONFIG_HASH ||
      hello.backend !== "exact-msvc9-r58-multiworld-forward" ||
      hello.exact_library_sha256 !== EXACT_LIBRARY_SHA256) {
    fail("worker identity or ABI does not match the exact backend");
  }
  return hello;
}

export function decodeObservation(bytes, start = 0) {
  if (bytes.byteLength - start < 112) fail("truncated observation header");
  const view = bytesView(bytes);
  const bodyCount = view.getUint32(start + 104, true);
  if (bodyCount > BODY_CAPACITY) fail(`body count ${bodyCount} exceeds capacity`);
  const end = start + 112 + bodyCount * 100;
  if (end > bytes.byteLength) fail("truncated observation bodies");
  const bodies = [];
  for (let index = 0, offset = start + 112; index < bodyCount; index++, offset += 100) {
    const kind = view.getUint8(offset + 96);
    const shape = view.getUint8(offset + 97);
    const lifecycle = view.getUint8(offset + 98);
    if (view.getUint8(offset + 99)) fail("body reserved byte is nonzero");
    bodies.push({
      id: view.getUint32(offset + 80, true),
      kind: enumName(bodyKinds, kind, "body kind"),
      shape: enumName(shapes, shape, "shape"),
      lifecycle: enumName(lifecycles, lifecycle, "lifecycle"),
      color: view.getInt32(offset + 84, true),
      x: view.getFloat64(offset + 24, true),
      y: view.getFloat64(offset + 32, true),
      vx: view.getFloat64(offset + 40, true),
      vy: view.getFloat64(offset + 48, true),
      angle: view.getFloat64(offset + 56, true),
      angular_velocity: view.getFloat64(offset + 64, true),
      size: view.getFloat64(offset + 72, true),
      chain_id: view.getUint32(offset + 88, true),
      projectile_hits: view.getUint32(offset + 92, true),
      age_ticks: safeNumber(view.getBigUint64(offset, true), "body age"),
      remaining_lifetime: safeNumber(view.getBigInt64(offset + 8, true), "body lifetime"),
      rot_timer: safeNumber(view.getBigUint64(offset + 16, true), "body rot timer"),
    });
  }
  return {
    value: {
      tick: safeNumber(view.getBigUint64(start, true), "tick"),
      score: safeNumber(view.getBigInt64(start + 8, true), "score"),
      gauge: safeNumber(view.getBigInt64(start + 16, true), "gauge"),
      level: view.getUint32(start + 88, true),
      terminated: Boolean(view.getUint8(start + 108)),
      truncated: Boolean(view.getUint8(start + 109)),
      left_held: Boolean(view.getUint8(start + 110)),
      right_held: Boolean(view.getUint8(start + 111)),
      highest_chain: view.getUint32(start + 100, true),
      qualifying_clear_count: safeNumber(view.getBigUint64(start + 32, true), "clear count"),
      field: {
        x: view.getFloat64(start + 40, true),
        y: view.getFloat64(start + 48, true),
        width: view.getFloat64(start + 56, true),
        height: view.getFloat64(start + 64, true),
        side_wall_top: view.getFloat64(start + 72, true),
        side_wall_bottom: view.getFloat64(start + 80, true),
      },
      gauge_max: safeNumber(view.getBigInt64(start + 24, true), "gauge maximum"),
      difficulty: {
        active_colors: view.getUint32(start + 92, true),
        spawn_interval_ticks: view.getUint32(start + 96, true),
      },
      bodies,
    },
    offset: end,
  };
}

export function decodeReset(bytes) {
  const decoded = decodeObservation(bytes);
  if (decoded.offset !== bytes.byteLength) fail("Reset payload has trailing bytes");
  return {observation: decoded.value, events: []};
}

export function decodeStep(bytes, decodedObservation = null) {
  const observation = decodedObservation || decodeObservation(bytes);
  const view = bytesView(bytes);
  let offset = observation.offset;
  if (bytes.byteLength - offset < 84) fail("truncated Step diagnostics");
  const reward = safeNumber(view.getBigInt64(offset, true), "reward");
  const eventCount = safeNumber(view.getBigUint64(offset + 8, true), "event count");
  const terminated = Boolean(view.getUint8(offset + 80));
  const truncated = Boolean(view.getUint8(offset + 81));
  if (terminated !== observation.value.terminated || truncated !== observation.value.truncated) {
    fail("Step flags disagree with its observation");
  }
  const diagnostics = {
    config_hash: view.getBigUint64(offset + 16, true).toString(),
    finish_call_count: safeNumber(view.getBigUint64(offset + 24, true), "finish call count"),
    terminal_metadata_recorded: Boolean(view.getUint8(offset + 82)),
    recorded_final_score: safeNumber(view.getBigInt64(offset + 32, true), "recorded score"),
    recorded_final_highest_chain: view.getUint32(offset + 64, true),
    recorded_final_level: view.getUint32(offset + 68, true),
    recorded_final_clears: safeNumber(view.getBigUint64(offset + 40, true), "recorded clears"),
    latest_final_score: safeNumber(view.getBigInt64(offset + 48, true), "latest score"),
    latest_final_highest_chain: view.getUint32(offset + 72, true),
    latest_final_level: view.getUint32(offset + 76, true),
    latest_final_clears: safeNumber(view.getBigUint64(offset + 56, true), "latest clears"),
    invalid_action: Boolean(view.getUint8(offset + 83)),
  };
  if (diagnostics.config_hash !== EXACT_CONFIG_HASH) {
    fail("worker mechanics configuration does not match normal mode v2.03");
  }
  offset += 84;
  const events = [];
  for (let index = 0; index < eventCount; index++) {
    if (bytes.byteLength - offset < 36) fail("truncated event header");
    const detailSize = view.getUint16(offset + 32, true);
    const kind = view.getUint8(offset + 34);
    if (view.getUint8(offset + 35)) fail("event reserved byte is nonzero");
    if (offset + 36 + detailSize > bytes.byteLength) fail("truncated event detail");
    events.push({
      tick: safeNumber(view.getBigUint64(offset, true), "event tick"),
      sequence: safeNumber(view.getBigUint64(offset + 8, true), "event sequence"),
      kind,
      kind_name: enumName(eventKinds, kind, "event kind"),
      a: view.getUint32(offset + 24, true),
      b: view.getUint32(offset + 28, true),
      value: safeNumber(view.getBigInt64(offset + 16, true), "event value"),
      detail: textDecoder.decode(bytes.subarray(offset + 36, offset + 36 + detailSize)),
    });
    offset += 36 + detailSize;
  }
  if (offset !== bytes.byteLength) fail("Step payload has trailing bytes");
  return {
    observation: observation.value, reward, terminated, truncated, events, diagnostics,
  };
}
