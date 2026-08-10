import assert from "node:assert/strict";
import test from "node:test";

import {BrowserGame, ExactWorkerClient} from "../static/exact-runtime.js";
import {EXACT_CONFIG_HASH} from "../static/exact-codec.mjs";
import {encodeReplayWord, parseReplay, serializeReplay} from "../static/replay.mjs";

const observation = tick => ({
  tick, score: 0, gauge: 3000 - tick, gauge_max: 3000, level: 1,
  terminated: false, truncated: false, bodies: [], field: {}, difficulty: {},
});

test("forwards worker startup progress without coupling it to RPC state", () => {
  const updates = [];
  const worker = {};
  const client = new ExactWorkerClient(worker, 100, message => updates.push(message));
  client.onMessage({type: "progress", message: "Starting exact simulation…"});
  assert.deepEqual(updates, ["Starting exact simulation…"]);
});

test("terminates a worker that never finishes startup", async () => {
  class SilentWorker {
    static latest;
    constructor() { SilentWorker.latest = this; }
    terminate() { this.terminated = true; }
  }
  await assert.rejects(ExactWorkerClient.create({WorkerClass: SilentWorker, timeoutMs: 5}),
    /exact worker startup timed out/);
  assert.equal(SilentWorker.latest.terminated, true);
});

test("serializes queued async steps and inserts a release tick after a shot", async () => {
  const calls = [];
  const resolvers = [];
  const client = {
    async reset() { return {observation: observation(0), events: []}; },
    step(kind, x, y, suppressFreshEdges) {
      calls.push({kind, x, y, suppressFreshEdges});
      return new Promise(resolve => resolvers.push(resolve));
    },
    close() {},
  };
  const snapshots = [];
  const game = new BrowserGame(client, state => snapshots.push(state), {
    seed: 7, now: () => 0, clock: {setTimeout: () => 1, clearTimeout() {}},
  });
  await game.restart(7, true);
  game.shoot("weak", 12, 34);
  game.pendingTicks = 2;
  const pumping = game.pump();
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(calls, [{kind: 1, x: 12, y: 34, suppressFreshEdges: true}]);
  resolvers.shift()({observation: observation(1), events: []});
  await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(calls, [
    {kind: 1, x: 12, y: 34, suppressFreshEdges: true},
    {kind: 0, x: 12, y: 34, suppressFreshEdges: true},
  ]);
  resolvers.shift()({observation: observation(2), events: []});
  await pumping;
  assert.equal(game.observation.tick, 2);
  assert.equal(snapshots.at(-1).observation.tick, 2);
  assert.deepEqual(game.recordedWords, [
    encodeReplayWord(1, 12, 34), encodeReplayWord(0, 12, 34),
  ]);
});

test("50 Hz scheduler queues elapsed ticks without overlapping the pump", () => {
  let now = 61;
  let delay;
  const game = new BrowserGame({close() {}}, () => {}, {
    seed: 1, now: () => now,
    clock: {setTimeout: (_callback, value) => { delay = value; return 1; }, clearTimeout() {}},
  });
  game.observation = observation(0);
  game.running = true;
  game.deadline = 20;
  let pumps = 0;
  game.pump = () => { pumps++; };
  game.schedule();
  assert.equal(game.pendingTicks, 3);
  assert.equal(pumps, 1);
  assert.equal(game.deadline, 80);
  assert.equal(delay, 19);
});

test("replay speed changes scheduler cadence", () => {
  let delay;
  const game = new BrowserGame({close() {}}, () => {}, {
    seed: 1, now: () => 11,
    clock: {setTimeout: (_callback, value) => { delay = value; return 1; }, clearTimeout() {}},
  });
  assert.throws(() => game.setReplaySpeed(3), /1x, 2x, 4x, or 8x/);
  for (const speed of [1, 2, 4, 8]) {
    game.setReplaySpeed(speed);
    assert.equal(game.replayInterval(), 20 / speed);
  }
  game.setReplaySpeed(4);
  game.mode = "replay";
  game.running = true;
  game.replayFrame = 1;
  game.replayComputed = 20;
  game.replayEffectiveTotal = 20;
  game.deadline = 5;
  game.displayReplayFrame = position => {
    game.replayFrame = position;
    return true;
  };
  game.schedule();
  assert.equal(game.replayFrame, 3);
  assert.equal(game.deadline, 15);
  assert.equal(delay, 4);
});

test("buffered replay steps accumulate and respect a later pause", () => {
  const game = new BrowserGame({close() {}}, () => {}, {
    seed: 1, now: () => 0,
    clock: {setTimeout: () => 1, clearTimeout() {}},
  });
  game.mode = "replay";
  game.observation = observation(0);
  game.running = true;
  game.replayData = {frameCount: 10, words: new Uint32Array(10)};
  game.replayEffectiveTotal = 10;
  game.stepReplay(1);
  game.stepReplay(1);
  assert.equal(game.replayRequestedFrame, 2);
  assert.equal(game.replayResumeAfterSeek, true);
  game.setRunning(false);
  assert.equal(game.replayResumeAfterSeek, false);
  game.setRunning(true);
  assert.equal(game.replayResumeAfterSeek, true);
});

test("fast-forward targets an 80-tick batch", () => {
  let delay;
  const game = new BrowserGame({close() {}}, () => {}, {
    seed: 1, now: () => 61,
    clock: {setTimeout: (_callback, value) => { delay = value; return 1; }, clearTimeout() {}},
  });
  game.observation = observation(0);
  game.running = true;
  game.fastForward = true;
  game.fastForwardRefill = true;
  game.deadline = 20;
  let pumps = 0;
  game.pump = () => { pumps++; };
  game.schedule();
  assert.equal(game.pendingTicks, 80);
  assert.equal(pumps, 1);
  assert.equal(game.deadline, 81);
  assert.equal(delay, 20);
});

test("released fast-forward drains its accelerated backlog without collapsing to five", () => {
  const game = new BrowserGame({close() {}}, () => {}, {
    seed: 1, now: () => 61,
    clock: {setTimeout: () => 1, clearTimeout() {}},
  });
  game.observation = observation(0);
  game.running = true;
  game.fastForward = true;
  game.fastForwardRefill = false;
  game.pendingTicks = 60;
  game.deadline = 20;
  game.pump = () => {};
  game.schedule();
  assert.equal(game.pendingTicks, 60);
});

test("a released wheel gesture still drains all 80 accelerated ticks", async () => {
  const calls = [];
  let tick = 2;
  const client = {
    async step(_kind, _x, _y, _suppressFreshEdges, waitTicks) {
      calls.push(waitTicks);
      tick += waitTicks;
      return {observation: observation(tick), events: []};
    },
    close() {},
  };
  const game = new BrowserGame(client, () => {}, {
    seed: 1, now: () => 0, clock: {setTimeout: () => 1, clearTimeout() {}},
  });
  game.observation = observation(2);
  game.recordedWords = [0, 0];
  game.running = true;
  game.setFastForward(true);
  game.setFastForward(false);
  while (game.processing) await new Promise(resolve => setImmediate(resolve));
  assert.deepEqual(calls, [20, 20, 20, 20]);
  assert.equal(game.observation.tick, 82);
  assert.equal(game.pendingTicks, 0);
  assert.equal(game.fastForward, false);
});

test("fast-forward renders at most every 20 accelerated ticks like v2.03", async () => {
  const calls = [];
  let tick = 2;
  const client = {
    step(kind, x, y, suppressFreshEdges, waitTicks) {
      calls.push({kind, x, y, suppressFreshEdges, waitTicks});
      tick += waitTicks;
      return Promise.resolve({observation: observation(tick), events: []});
    },
    close() {},
  };
  const game = new BrowserGame(client, () => {}, {
    seed: 1, now: () => 0, clock: {setTimeout: () => 1, clearTimeout() {}},
  });
  game.observation = observation(2);
  game.running = true;
  game.fastForward = true;
  game.pendingTicks = 47;
  game.recordedWords = [0, 0];
  await game.pump();
  assert.deepEqual(calls.map(call => call.waitTicks), [20, 20, 7]);
  assert.ok(calls.every(call => call.kind === 0 && !call.suppressFreshEdges));
  assert.equal(game.observation.tick, 49);
  assert.equal(game.recordedWords.length, 49);
  assert.equal(game.pendingTicks, 0);
});

test("fast-forward does not batch a shot or its release tick", async () => {
  const calls = [];
  let tick = 2;
  const client = {
    step(kind, _x, _y, _suppressFreshEdges, waitTicks) {
      calls.push({kind, waitTicks});
      tick += waitTicks;
      return Promise.resolve({observation: observation(tick), events: []});
    },
    close() {},
  };
  const game = new BrowserGame(client, () => {}, {
    seed: 1, now: () => 0, clock: {setTimeout: () => 1, clearTimeout() {}},
  });
  game.observation = observation(2);
  game.running = true;
  game.fastForward = true;
  game.pendingTicks = 22;
  game.recordedWords = [0, 0];
  game.queue.push({kind: 1, x: 12, y: 34});
  await game.pump();
  assert.deepEqual(calls, [
    {kind: 1, waitTicks: 1},
    {kind: 0, waitTicks: 1},
    {kind: 0, waitTicks: 20},
  ]);
});

test("restart replaces the one-reset exact worker process", async () => {
  const clients = [];
  const makeClient = async () => {
    const client = {
      resets: [], closed: false,
      async reset(seed) {
        if (this.resets.length) throw new Error("one reset per process");
        this.resets.push(seed);
        return {observation: observation(0), events: []};
      },
      close() { this.closed = true; },
    };
    clients.push(client);
    return client;
  };
  const first = await makeClient();
  const game = new BrowserGame(first, () => {}, {
    seed: 1, clientFactory: makeClient,
    now: () => 0, clock: {setTimeout: () => 1, clearTimeout() {}},
  });
  assert.equal(await game.restart(1, true), true);
  assert.equal(await game.restart(2, true), true);
  assert.equal(clients.length, 2);
  assert.equal(first.closed, true);
  assert.deepEqual(clients.map(client => client.resets), [[1], [2]]);
  assert.equal(game.client, clients[1]);
});

test("a stale replacement failure cannot stop a newer successful restart", async () => {
  const replacements = [];
  const factory = () => new Promise((resolve, reject) => replacements.push({resolve, reject}));
  const first = {
    async reset() { return {observation: observation(0), events: []}; },
    close() {},
  };
  const errors = [];
  const game = new BrowserGame(first, (_state, error) => {
    if (error) errors.push(error.message);
  }, {seed: 1, clientFactory: factory, now: () => 0,
    clock: {setTimeout: () => 1, clearTimeout() {}}});
  await game.restart(1, true);
  const older = game.restart(2, true);
  const newer = game.restart(3, true);
  const newestClient = {
    async reset() { return {observation: observation(0), events: []}; },
    close() {},
  };
  replacements[1].resolve(newestClient);
  assert.equal(await newer, true);
  replacements[0].reject(new Error("stale boot failed"));
  assert.equal(await older, false);
  assert.equal(game.running, true);
  assert.equal(game.client, newestClient);
  assert.deepEqual(errors, []);
});

test("writes the accepted terminal tick with first-finish replay metadata", async () => {
  const terminal = {...observation(1), score: 99, level: 7, highest_chain: 5,
    terminated: true};
  const client = {
    async reset() { return {observation: observation(0), events: []}; },
    async step() {
      return {observation: terminal, events: [], diagnostics: {
        terminal_metadata_recorded: true,
        recorded_final_level: 6,
        recorded_final_score: 88,
        recorded_final_highest_chain: 4,
      }};
    },
    close() {},
  };
  const game = new BrowserGame(client, () => {}, {
    seed: 0xfedcba98, now: () => 0,
    clock: {setTimeout: () => 1, clearTimeout() {}},
  });
  await game.restart(0xfedcba98, true);
  game.shoot("both", 12.7, 478.9);
  game.pendingTicks = 1;
  await game.pump();
  const replay = parseReplay(game.replayBytes());
  assert.equal(replay.seed, 0xfedcba98);
  assert.equal(replay.highestLevel, 6);
  assert.equal(replay.finalScore, 88);
  assert.equal(replay.highestChain, 4);
  assert.deepEqual([...replay.words], [encodeReplayWord(3, 13, 479)]);
});

function stepBytes(tick, {terminated = false} = {}) {
  const bytes = new Uint8Array(112 + 84);
  const view = new DataView(bytes.buffer);
  view.setBigUint64(0, BigInt(tick), true);
  view.setBigInt64(8, 0n, true);
  view.setBigInt64(16, BigInt(3000 - tick), true);
  view.setBigInt64(24, 3000n, true);
  view.setUint32(88, 1, true);
  view.setUint32(92, 3, true);
  view.setUint32(96, 50, true);
  view.setUint32(100, 0, true);
  view.setUint32(104, 0, true);
  view.setUint8(108, Number(terminated));
  const diagnostics = 112;
  view.setBigUint64(diagnostics + 8, 0n, true);
  view.setBigUint64(diagnostics + 16, BigInt(EXACT_CONFIG_HASH), true);
  view.setUint8(diagnostics + 80, Number(terminated));
  return bytes;
}

test("precomputes imported levels exactly and scrubs cached observations without new steps", async () => {
  const calls = [];
  const clients = [];
  const makeClient = async () => {
    let tick = 0;
    const client = {
      async reset() { tick = 0; return {observation: observation(0), events: []}; },
      async stepRaw(kind, x, y, suppressFreshEdges) {
        calls.push({kind, x, y, suppressFreshEdges});
        return stepBytes(++tick);
      },
      close() {},
    };
    clients.push(client);
    return client;
  };
  const first = await makeClient();
  const game = new BrowserGame(first, () => {}, {
    seed: 1, clientFactory: makeClient, now: () => 0,
    clock: {setTimeout: () => 1, clearTimeout() {}},
  });
  await game.restart(1, false);
  const replay = parseReplay(serializeReplay({
    seed: 9, highestLevel: 0, finalScore: 0, highestChain: 0,
    words: [
      encodeReplayWord(1, 10, 20), encodeReplayWord(1, 11, 21),
      encodeReplayWord(1, 12, 22), encodeReplayWord(0, 13, 23),
    ],
  }));
  assert.equal(await game.loadReplay(replay, "levels.rpy"), true);
  for (let tries = 0; !game.replayComplete && tries < 20; tries++) {
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.equal(game.replayComplete, true);
  assert.deepEqual(calls, [
    {kind: 1, x: 10, y: 20, suppressFreshEdges: true},
    {kind: 1, x: 11, y: 21, suppressFreshEdges: true},
    {kind: 1, x: 12, y: 22, suppressFreshEdges: false},
    {kind: 0, x: 13, y: 23, suppressFreshEdges: false},
  ]);
  const callCount = calls.length;
  game.seekReplay(3);
  assert.equal(game.observation.tick, 3);
  assert.equal(game.running, false);
  game.setRunning(true);
  game.stepReplay(-1);
  assert.equal(game.observation.tick, 2);
  assert.equal(game.running, true);
  game.stepReplay(1);
  assert.equal(game.observation.tick, 3);
  assert.equal(game.running, true);
  game.seekReplay(1, {preserveRunning: true});
  assert.equal(game.observation.tick, 1);
  assert.equal(game.running, true);
  game.setRunning(false);
  game.seekReplay(1);
  assert.equal(game.observation.tick, 1);
  assert.equal(game.running, false);
  assert.equal(calls.length, callCount);

  // An old position may leave the bounded cache during a very long replay.
  // Seeking there must rebuild exact state from the immutable seed/input stream.
  game.seekReplay(2);
  game.replayCache.entries[0] = null;
  game.setRunning(true);
  game.stepReplay(-1);
  for (let tries = 0; clients.length < 3 || !game.replayComplete; tries++) {
    assert.ok(tries < 20, "evicted replay position should be rebuilt");
    await new Promise(resolve => setImmediate(resolve));
  }
  assert.equal(game.observation.tick, 1);
  assert.equal(game.running, true);
  assert.equal(clients.length, 3);
  assert.deepEqual(calls.slice(-4), [
    {kind: 1, x: 10, y: 20, suppressFreshEdges: true},
    {kind: 1, x: 11, y: 21, suppressFreshEdges: true},
    {kind: 1, x: 12, y: 22, suppressFreshEdges: false},
    {kind: 0, x: 13, y: 23, suppressFreshEdges: false},
  ]);
});
