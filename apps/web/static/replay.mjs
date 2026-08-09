export const REPLAY_HEADER_BYTES = 52;
export const REPLAY_TICK_MS = 20;
export const MAX_REPLAY_FRAMES = 5_000_000;

const INT32_MIN = -0x80000000;
const INT32_MAX = 0x7fffffff;
const X_MAX = 0x3ff;
const Y_MAX = 0x1ff;

function bytes(value) {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new TypeError("replay must be an ArrayBuffer or byte view");
}

function signed32(value, name) {
  if (!Number.isInteger(value) || value < INT32_MIN || value > INT32_MAX) {
    throw new RangeError(`${name} must fit in signed int32`);
  }
  return value;
}

function uint32(value, name) {
  if (!Number.isInteger(value) || value < 0 || value > 0xffffffff) {
    throw new RangeError(`${name} must fit in uint32`);
  }
  return value;
}

export function quantizeReplayPoint(x, y) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    throw new TypeError("replay coordinates must be finite");
  }
  return {
    x: Math.max(0, Math.min(639, Math.round(x))),
    y: Math.max(0, Math.min(479, Math.round(y))),
  };
}

export function encodeReplayWord(kind, x, y) {
  if (!Number.isInteger(kind) || kind < 0 || kind > 3) {
    throw new RangeError("replay button level must be in [0, 3]");
  }
  if (!Number.isInteger(x) || x < 0 || x > X_MAX ||
      !Number.isInteger(y) || y < 0 || y > Y_MAX) {
    throw new RangeError("replay coordinates exceed their packed fields");
  }
  return (((y << 12) | (x << 2) | kind) >>> 0);
}

export function decodeReplayWord(word) {
  uint32(word, "replay word");
  return {
    word: word >>> 0,
    kind: word & 3,
    left: Boolean(word & 1),
    right: Boolean(word & 2),
    x: (word >>> 2) & X_MAX,
    y: (word >>> 12) & Y_MAX,
    reserved: word >>> 21,
  };
}

export function parseReplay(value, {maxFrames = MAX_REPLAY_FRAMES} = {}) {
  const data = bytes(value);
  if (data.byteLength < REPLAY_HEADER_BYTES) {
    throw new Error(`replay is shorter than the ${REPLAY_HEADER_BYTES}-byte v2.03 header`);
  }
  if ((data.byteLength - REPLAY_HEADER_BYTES) % 4) {
    throw new Error("replay ends with a partial input record");
  }
  const frameCount = (data.byteLength - REPLAY_HEADER_BYTES) / 4;
  if (frameCount > maxFrames) {
    throw new Error(`replay has ${frameCount} frames; limit is ${maxFrames}`);
  }
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  const mode = view.getInt32(16, true);
  if (mode !== 0) {
    throw new Error(`replay mode ${mode} is unsupported; exact playback currently supports normal mode 0`);
  }
  const words = new Uint32Array(frameCount);
  for (let index = 0; index < frameCount; index++) {
    words[index] = view.getUint32(REPLAY_HEADER_BYTES + index * 4, true);
  }
  let zeroPadding = true;
  for (let offset = 20; offset < REPLAY_HEADER_BYTES; offset++) {
    if (view.getUint8(offset)) zeroPadding = false;
  }
  return Object.freeze({
    seed: view.getUint32(0, true),
    highestLevel: view.getInt32(4, true),
    finalScore: view.getInt32(8, true),
    highestChain: view.getInt32(12, true),
    mode,
    zeroPadding,
    words,
    frameCount,
  });
}

export function serializeReplay({seed, highestLevel, finalScore, highestChain,
  words, mode = 0}) {
  uint32(seed, "seed");
  signed32(highestLevel, "highest level");
  signed32(finalScore, "final score");
  signed32(highestChain, "highest chain");
  signed32(mode, "mode");
  if (mode !== 0) throw new Error("only normal-mode v2.03 replays can be written");
  const source = words instanceof Uint32Array ? words : Array.from(words || []);
  if (source.length > MAX_REPLAY_FRAMES) {
    throw new Error(`replay has ${source.length} frames; limit is ${MAX_REPLAY_FRAMES}`);
  }
  for (let index = 0; index < source.length; index++) uint32(source[index], `replay word ${index}`);
  const packed = words instanceof Uint32Array ? words : Uint32Array.from(source);
  const output = new Uint8Array(REPLAY_HEADER_BYTES + packed.length * 4);
  const view = new DataView(output.buffer);
  view.setUint32(0, seed, true);
  view.setInt32(4, highestLevel, true);
  view.setInt32(8, finalScore, true);
  view.setInt32(12, highestChain, true);
  view.setInt32(16, mode, true);
  for (let index = 0; index < packed.length; index++) {
    view.setUint32(REPLAY_HEADER_BYTES + index * 4, packed[index], true);
  }
  return output;
}

export class ReplayObservationCache {
  constructor({maximumBytes = 384 * 1024 * 1024} = {}) {
    this.maximumBytes = maximumBytes;
    this.entries = [];
    this.startIndex = 0;
    this.length = 0;
    this.byteLength = 0;
  }

  append(value, minimumIndexToKeep = this.startIndex) {
    const data = bytes(value);
    while (this.byteLength + data.byteLength > this.maximumBytes &&
           this.startIndex < minimumIndexToKeep) {
      const removed = this.entries[this.startIndex];
      if (removed) {
        this.byteLength -= removed.byteLength;
        this.entries[this.startIndex] = null;
      }
      this.startIndex++;
    }
    if (this.byteLength + data.byteLength > this.maximumBytes) return false;
    this.entries[this.length] = data.slice();
    this.length++;
    this.byteLength += data.byteLength;
    return true;
  }

  get(index) {
    if (!Number.isInteger(index) || index < 0 || index >= this.length) return null;
    return this.entries[index] || null;
  }
}
