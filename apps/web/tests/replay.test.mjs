import assert from "node:assert/strict";
import {createHash} from "node:crypto";
import {readFileSync} from "node:fs";
import test from "node:test";
import {fileURLToPath} from "node:url";

import {
  ReplayObservationCache, clampReplayScrubFrame, decodeReplayWord,
  encodeReplayWord, parseReplay, quantizeReplayPoint, serializeReplay,
} from "../static/replay.mjs";

const root = fileURLToPath(new URL("../../..", import.meta.url));
const scoreProbe = `${root}/reference/captures/seed41-score-parity-20260720-001/input.rpy`;

test("parses and byte-identically reserializes the original-observed v2.03 score probe", () => {
  const original = new Uint8Array(readFileSync(scoreProbe));
  const replay = parseReplay(original);
  const rebuilt = serializeReplay({...replay, words: replay.words});
  assert.deepEqual(rebuilt, original);
  assert.equal(replay.seed, 41);
  assert.equal(replay.frameCount, 520);
  assert.equal(createHash("sha256").update(rebuilt).digest("hex"),
    "1ce501febe8f3f6291e4b82736542179bd9808e412d38e0e1fb1c92d05797657");
});

test("preserves uint32 seed bits and packs every replay field at its boundary", () => {
  const words = Uint32Array.of(
    encodeReplayWord(3, 1023, 511),
    encodeReplayWord(0, 0, 0),
  );
  const data = serializeReplay({
    seed: 0xfedcba98, highestLevel: 100, finalScore: 123456,
    highestChain: 42, words,
  });
  const replay = parseReplay(data);
  assert.equal(replay.seed, 0xfedcba98);
  assert.deepEqual([...replay.words], [...words]);
  assert.deepEqual(decodeReplayWord(words[0]), {
    word: words[0], kind: 3, left: true, right: true,
    x: 1023, y: 511, reserved: 0,
  });
});

test("mirrors the v2.03 fixed header offset while rejecting unsafe inputs", () => {
  const data = serializeReplay({seed: 1, highestLevel: 1, finalScore: 0,
    highestChain: 0, words: [0]});
  data[20] = 9;
  assert.equal(parseReplay(data).zeroPadding, false);
  assert.throws(() => parseReplay(data.subarray(0, 51)), /shorter/);
  assert.throws(() => parseReplay(new Uint8Array([...data, 0])), /partial/);
  new DataView(data.buffer).setInt32(16, 1, true);
  assert.throws(() => parseReplay(data), /mode 1 is unsupported/);
});

test("quantizes browser input before simulation and bounds replay cache memory", () => {
  assert.deepEqual(quantizeReplayPoint(12.6, 479.8), {x: 13, y: 479});
  assert.deepEqual(quantizeReplayPoint(-2, 900), {x: 0, y: 479});
  const cache = new ReplayObservationCache({maximumBytes: 4});
  cache.append(Uint8Array.of(1, 2));
  cache.append(Uint8Array.of(3));
  assert.deepEqual([...cache.get(0)], [1, 2]);
  assert.deepEqual([...cache.get(1)], [3]);
  assert.equal(cache.append(Uint8Array.of(4, 5)), false);
  assert.equal(cache.append(Uint8Array.of(4, 5), 2), true);
  assert.equal(cache.get(0), null);
  assert.deepEqual([...cache.get(2)], [4, 5]);
});

test("clamps replay scrubbing to the latest buffered frame", () => {
  assert.equal(clampReplayScrubFrame(80, 35), 35);
  assert.equal(clampReplayScrubFrame(20, 35), 20);
  assert.equal(clampReplayScrubFrame(-4, 35), 0);
  assert.equal(clampReplayScrubFrame(10, 0), 0);
  assert.equal(clampReplayScrubFrame("12", "35"), 12);
  assert.equal(clampReplayScrubFrame(NaN, 35), 0);
});

test("rejects replay words instead of silently coercing them", () => {
  const base = {seed: 1, highestLevel: 1, finalScore: 0, highestChain: 0};
  assert.throws(() => serializeReplay({...base, words: [1.9]}), /word 0/);
  assert.throws(() => serializeReplay({...base, words: [NaN]}), /word 0/);
  assert.throws(() => serializeReplay({...base, words: [0x1_0000_0000]}), /word 0/);
});
