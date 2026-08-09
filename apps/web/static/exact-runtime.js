import {
  OPCODE, decodeHello, decodeObservation, decodeReset, decodeStep, encodeReset, encodeStep,
} from "./exact-codec.mjs";
import {
  ReplayObservationCache, decodeReplayWord, encodeReplayWord,
  quantizeReplayPoint, serializeReplay,
} from "./replay.mjs";

const kinds = {weak: 1, strong: 2, both: 3};
const FAST_FORWARD_TICKS = 80;

export class ExactWorkerClient {
  static async create({WorkerClass = globalThis.Worker, timeoutMs = 60000,
    onProgress = () => {}} = {}) {
    if (!WorkerClass) throw new Error("Web Workers are unavailable");
    const worker = new WorkerClass(new URL("./exact-worker.js?v=20260809j", import.meta.url));
    const client = new ExactWorkerClient(worker, timeoutMs, onProgress);
    let timer;
    try {
      await Promise.race([client.ready, new Promise((_, reject) => {
        timer = setTimeout(() => reject(new Error("exact worker startup timed out")), timeoutMs);
      })]);
    } catch (error) {
      worker.terminate?.();
      client.fail(error);
      throw error;
    } finally {
      clearTimeout(timer);
    }
    client.info = decodeHello(await client.rpc(OPCODE.hello));
    return client;
  }

  constructor(worker, timeoutMs = 60000, onProgress = () => {}) {
    this.worker = worker;
    this.timeoutMs = timeoutMs;
    this.onProgress = onProgress;
    this.nextMessageId = 1;
    this.pending = new Map();
    this.tail = Promise.resolve();
    this.ready = new Promise((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    worker.onmessage = event => this.onMessage(event.data);
    worker.onerror = event => this.fail(new Error(event.message || "exact Web Worker failed"));
  }

  onMessage(message) {
    if (message.type === "progress") {
      try { this.onProgress(message.message); }
      catch (_) { /* Loading UI failures must not interrupt the simulator. */ }
      return;
    }
    if (message.type === "ready") return this.resolveReady();
    if (message.type === "fatal") return this.fail(new Error(message.error));
    if (message.type !== "response") return;
    const request = this.pending.get(message.messageId);
    if (!request) return;
    clearTimeout(request.timer);
    this.pending.delete(message.messageId);
    if (message.error) request.reject(new Error(message.error));
    else request.resolve(new Uint8Array(message.payload));
  }

  fail(error) {
    this.rejectReady(error);
    for (const request of this.pending.values()) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    this.pending.clear();
  }

  rpc(opcode, payload = new Uint8Array()) {
    const operation = () => new Promise((resolve, reject) => {
      const messageId = this.nextMessageId++;
      const copy = payload.slice();
      const timer = setTimeout(() => {
        this.pending.delete(messageId);
        reject(new Error(`exact-worker RPC ${opcode} timed out`));
      }, this.timeoutMs);
      this.pending.set(messageId, {resolve, reject, timer});
      this.worker.postMessage({type: "rpc", messageId, opcode, payload: copy.buffer},
        [copy.buffer]);
    });
    const result = this.tail.then(operation, operation);
    this.tail = result.catch(() => {});
    return result;
  }

  resetRaw(seed) { return this.rpc(OPCODE.reset, encodeReset(seed)); }
  reset(seed) { return this.resetRaw(seed).then(decodeReset); }
  stepRaw(kind, x, y, suppressFreshEdges = false) {
    return this.rpc(OPCODE.step, encodeStep(kind, x, y, suppressFreshEdges));
  }
  step(kind, x, y, suppressFreshEdges = false) {
    return this.stepRaw(kind, x, y, suppressFreshEdges).then(decodeStep);
  }

  close() {
    this.worker.postMessage({type: "close"});
    this.worker.terminate?.();
    this.fail(new Error("exact worker closed"));
  }
}

export class BrowserGame {
  static async create(onSnapshot, options = {}) {
    const clientFactory = options.clientFactory ||
      (() => ExactWorkerClient.create(options));
    const client = options.client || await clientFactory();
    const game = new BrowserGame(client, onSnapshot, {...options,
      clientFactory: options.client && !options.clientFactory ? null : clientFactory});
    if (!await game.restart(game.seed, true)) throw game.lastError;
    game.schedule();
    return game;
  }

  constructor(client, onSnapshot, {clock = globalThis,
    now = () => performance.now(), seed, clientFactory = null} = {}) {
    this.client = client;
    this.clientFactory = clientFactory;
    this.onSnapshot = onSnapshot;
    this.clock = clock;
    this.now = now;
    this.seed = seed === undefined ? crypto.getRandomValues(new Uint32Array(1))[0] : seed >>> 0;
    this.running = false;
    this.fastForward = false;
    this.queue = [];
    this.events = [];
    this.aim = {x: 320, y: 390};
    this.mode = "live";
    this.recordedWords = [];
    this.recordedMetadata = null;
    this.replayData = null;
    this.replayName = "";
    this.replayCache = null;
    this.replayInitialObservation = null;
    this.replayFrame = 0;
    this.replayComputed = 0;
    this.replayEffectiveTotal = 0;
    this.replayComplete = false;
    this.replayRequestedFrame = null;
    this.replayBuffering = false;
    this.replayWarning = "";
    this.replayTerminalReason = null;
    this.replayLastProgressTime = 0;
    this.pendingTicks = 0;
    this.processing = false;
    this.releaseNext = false;
    this.terminalReason = null;
    this.timer = 0;
    this.epoch = 0;
    this.closed = false;
    this.hasReset = false;
  }

  emit() {
    this.onSnapshot({observation: this.observation, events: this.events,
      running: this.running, fastForward: this.fastForward,
      seed: this.seed, terminal_reason: this.terminalReason,
      mode: this.mode,
      can_save_replay: this.mode === "live" && Boolean(this.recordedMetadata),
      replay: this.mode === "replay" ? {
        name: this.replayName,
        frame: this.replayFrame,
        total_frames: this.replayEffectiveTotal,
        source_frames: this.replayData.frameCount,
        buffered_frames: this.replayComputed,
        complete: this.replayComplete,
        buffering: this.replayBuffering,
        warning: this.replayWarning,
        cursor: this.replayFrame > 0 ?
          decodeReplayWord(this.replayData.words[this.replayFrame - 1]) : null,
      } : null});
  }

  async restart(seed, running = true) {
    const epoch = ++this.epoch;
    this.seed = seed >>> 0;
    this.mode = "live";
    this.running = false;
    this.fastForward = false;
    this.pendingTicks = 0;
    this.queue.length = 0;
    this.events.length = 0;
    this.releaseNext = false;
    this.terminalReason = null;
    this.recordedWords = [];
    this.recordedMetadata = null;
    this.replayData = null;
    this.replayName = "";
    this.replayCache = null;
    this.replayInitialObservation = null;
    this.replayFrame = 0;
    this.replayComputed = 0;
    this.replayEffectiveTotal = 0;
    this.replayComplete = false;
    this.replayRequestedFrame = null;
    this.replayBuffering = false;
    this.replayWarning = "";
    this.replayTerminalReason = null;
    try {
      // Exact workers deliberately permit one successful Reset per process.
      // A new run therefore needs a fresh worker/guest, not another Reset RPC
      // to the process that owns the previous world.
      if (this.hasReset && this.clientFactory) {
        this.client?.close();
        this.client = null;
        const replacement = await this.clientFactory();
        if (this.closed || epoch !== this.epoch) {
          replacement.close();
          return;
        }
        this.client = replacement;
      }
      const state = await this.client.reset(this.seed);
      if (this.closed || epoch !== this.epoch) return;
      this.observation = state.observation;
      this.hasReset = true;
      this.running = Boolean(running && !this.observation.terminated &&
        !this.observation.truncated);
      this.deadline = this.now() + 20;
      this.emit();
      return true;
    } catch (error) {
      if (!this.closed && epoch === this.epoch) {
        this.lastError = error;
        this.report(error);
      }
      return false;
    }
  }

  shoot(kind, x, y) {
    if (!(kind in kinds)) throw new Error(`unknown shot kind: ${kind}`);
    if (this.mode !== "live") return;
    if (!this.observation || this.observation.terminated || this.observation.truncated) return;
    const point = quantizeReplayPoint(x, y);
    this.aim = point;
    if (this.queue.length < 12) this.queue.push({kind: kinds[kind], ...point});
    if (!this.running) this.deadline = this.now() + 20;
    this.running = true;
    this.emit();
  }

  setAim(x, y) {
    this.aim = quantizeReplayPoint(x, y);
  }

  setRunning(running) {
    if (this.mode === "replay") {
      const canRun = this.observation && this.replayFrame < this.replayEffectiveTotal;
      this.running = Boolean(running && canRun);
      this.replayBuffering = Boolean(this.running &&
        this.replayFrame >= this.replayComputed && !this.replayComplete);
      this.deadline = this.now() + 20;
      this.emit();
      return;
    }
    this.running = Boolean(running && this.observation &&
      !this.observation.terminated && !this.observation.truncated);
    if (!this.running) {
      this.fastForward = false;
      this.pendingTicks = 0;
    }
    this.deadline = this.now() + 20;
    this.emit();
  }

  setFastForward(active) {
    if (this.mode === "replay") return;
    const next = Boolean(active && this.running &&
      !this.observation.terminated && !this.observation.truncated);
    if (next === this.fastForward) return;
    this.fastForward = next;
    this.emit();
  }

  nextAction() {
    let action = {kind: 0, ...this.aim};
    if (this.releaseNext) this.releaseNext = false;
    else if (this.queue.length) {
      action = this.queue.shift();
      this.releaseNext = true;
    }
    return action;
  }

  acceptStep(state) {
    this.observation = state.observation;
    this.events.push(...state.events);
    if (this.events.length > 80) this.events.splice(0, this.events.length - 80);
    if (this.mode === "live" && !this.recordedMetadata &&
        this.observation.terminated && state.diagnostics?.terminal_metadata_recorded) {
      this.recordedMetadata = {
        highestLevel: state.diagnostics.recorded_final_level,
        finalScore: state.diagnostics.recorded_final_score,
        highestChain: state.diagnostics.recorded_final_highest_chain,
      };
    }
    if (this.observation.terminated || this.observation.truncated) {
      const names = new Set(state.events.map(event => event.kind_name));
      this.terminalReason = names.has("level_completed") ? "level_completed" :
        this.observation.truncated ? "time_limit" : "game_over";
      this.running = false;
      this.fastForward = false;
      this.pendingTicks = 0;
      this.queue.length = 0;
    }
  }

  replayBytes() {
    if (this.mode !== "live" || !this.recordedMetadata) {
      throw new Error("a replay is available only after a recorded game finishes");
    }
    return serializeReplay({
      seed: this.seed,
      ...this.recordedMetadata,
      words: Uint32Array.from(this.recordedWords),
    });
  }

  async loadReplay(replay, name = "replay.rpy") {
    const epoch = ++this.epoch;
    this.mode = "replay";
    this.seed = replay.seed >>> 0;
    this.running = false;
    this.fastForward = false;
    this.pendingTicks = 0;
    this.queue.length = 0;
    this.events.length = 0;
    this.releaseNext = false;
    this.terminalReason = null;
    this.recordedWords = [];
    this.recordedMetadata = null;
    this.replayData = replay;
    this.replayName = name;
    this.replayCache = new ReplayObservationCache();
    this.replayInitialObservation = null;
    this.replayFrame = 0;
    this.replayComputed = 0;
    this.replayEffectiveTotal = replay.frameCount;
    this.replayComplete = false;
    this.replayRequestedFrame = null;
    this.replayBuffering = false;
    this.replayWarning = replay.zeroPadding ? "" :
      "The v2.03 loader ignores nonzero reserved header bytes; playback uses its fixed 52-byte offset.";
    this.replayTerminalReason = null;
    try {
      if (this.hasReset) {
        if (!this.clientFactory) throw new Error("replay playback needs a fresh exact worker");
        this.client?.close();
        this.client = null;
        const replacement = await this.clientFactory();
        if (this.closed || epoch !== this.epoch) {
          replacement.close();
          return false;
        }
        this.client = replacement;
      }
      const state = await this.client.reset(this.seed);
      if (this.closed || epoch !== this.epoch) return false;
      this.hasReset = true;
      this.replayInitialObservation = state.observation;
      this.observation = state.observation;
      this.running = replay.frameCount > 0;
      this.deadline = this.now() + 20;
      this.emit();
      void this.prepareReplay(epoch);
      return true;
    } catch (error) {
      if (!this.closed && epoch === this.epoch) this.report(error);
      return false;
    }
  }

  displayReplayFrame(position, {emit = true} = {}) {
    if (this.mode !== "replay") return false;
    const maximum = Math.min(this.replayComputed, this.replayEffectiveTotal);
    if (!Number.isInteger(position) || position < 0 || position > maximum) return false;
    const cached = position === 0 ? null : this.replayCache.get(position - 1);
    if (position > 0 && !cached) return false;
    const observation = position === 0 ? this.replayInitialObservation :
      decodeObservation(cached).value;
    if (!observation) return false;
    this.replayFrame = position;
    this.observation = observation;
    this.events = [];
    this.terminalReason = this.replayComplete && position >= this.replayEffectiveTotal ?
      this.replayTerminalReason : null;
    if (emit) this.emit();
    return true;
  }

  seekReplay(position) {
    if (this.mode !== "replay") return;
    const target = Math.max(0, Math.min(this.replayEffectiveTotal,
      Math.round(Number(position))));
    this.running = false;
    this.pendingTicks = 0;
    this.deadline = this.now() + 20;
    if (target <= this.replayComputed) {
      this.replayRequestedFrame = null;
      this.replayBuffering = false;
      if (!this.displayReplayFrame(target)) void this.rebuildReplay(target);
    } else {
      this.replayRequestedFrame = target;
      this.replayBuffering = true;
      this.emit();
    }
  }

  stepReplay(delta) {
    if (this.mode === "replay") this.seekReplay(this.replayFrame + delta);
  }

  async rebuildReplay(target) {
    const epoch = ++this.epoch;
    this.running = false;
    this.pendingTicks = 0;
    this.replayCache = new ReplayObservationCache();
    this.replayComputed = 0;
    this.replayComplete = false;
    this.replayEffectiveTotal = this.replayData.frameCount;
    this.replayRequestedFrame = target;
    this.replayBuffering = true;
    this.replayWarning = this.replayData.zeroPadding ? "" :
      "The v2.03 loader ignores nonzero reserved header bytes; playback uses its fixed 52-byte offset.";
    this.replayTerminalReason = null;
    this.terminalReason = null;
    this.emit();
    try {
      if (!this.clientFactory) throw new Error("replay seeking needs a fresh exact worker");
      this.client?.close();
      this.client = null;
      const replacement = await this.clientFactory();
      if (this.closed || epoch !== this.epoch) {
        replacement.close();
        return;
      }
      this.client = replacement;
      const state = await this.client.reset(this.seed);
      if (this.closed || epoch !== this.epoch) return;
      this.hasReset = true;
      this.replayInitialObservation = state.observation;
      if (target === 0) {
        this.replayRequestedFrame = null;
        this.replayBuffering = false;
        this.displayReplayFrame(0);
      }
      void this.prepareReplay(epoch);
    } catch (error) {
      if (!this.closed && epoch === this.epoch) this.report(error);
    }
  }

  validateReplayOutcome(state) {
    const diagnostics = state?.diagnostics;
    if (!diagnostics?.terminal_metadata_recorded) {
      const warning = "Replay exhausted before a natural game ending.";
      this.replayWarning = this.replayWarning ? `${this.replayWarning} ${warning}` : warning;
      return;
    }
    const differences = [];
    const fields = [
      ["score", this.replayData.finalScore, diagnostics.recorded_final_score],
      ["level", this.replayData.highestLevel, diagnostics.recorded_final_level],
      ["highest chain", this.replayData.highestChain,
        diagnostics.recorded_final_highest_chain],
    ];
    for (const [label, expected, actual] of fields) {
      if (expected !== actual) differences.push(`${label} header ${expected}, playback ${actual}`);
    }
    if (differences.length) {
      const warning = `Replay outcome header differs: ${differences.join("; ")}.`;
      this.replayWarning = this.replayWarning ? `${this.replayWarning} ${warning}` : warning;
    }
  }

  async prepareReplay(epoch) {
    let finalState = null;
    try {
      for (let index = 0; index < this.replayData.frameCount; index++) {
        if (this.closed || epoch !== this.epoch || this.mode !== "replay") return;
        const frame = decodeReplayWord(this.replayData.words[index]);
        const raw = await this.client.stepRaw(frame.kind, frame.x, frame.y, index < 2);
        if (this.closed || epoch !== this.epoch || this.mode !== "replay") return;
        const observation = decodeObservation(raw);
        const state = decodeStep(raw, observation);
        const cached = raw.subarray(0, observation.offset);
        while (true) {
          const protectedIndex = this.replayRequestedFrame === null ?
            Math.max(0, this.replayFrame - 1) :
            Math.max(0, this.replayRequestedFrame - 1);
          if (this.replayCache.append(cached, protectedIndex)) break;
          await new Promise(resolve => globalThis.setTimeout(resolve, 20));
          if (this.closed || epoch !== this.epoch || this.mode !== "replay") return;
        }
        this.replayComputed = index + 1;
        finalState = state;

        if (this.replayRequestedFrame !== null &&
            this.replayComputed >= this.replayRequestedFrame) {
          const target = this.replayRequestedFrame;
          this.replayRequestedFrame = null;
          this.replayBuffering = false;
          this.displayReplayFrame(target);
        } else if (this.running && this.replayBuffering &&
                   this.replayComputed > this.replayFrame) {
          this.replayBuffering = false;
        }

        const now = this.now();
        if (now - this.replayLastProgressTime >= 100) {
          this.replayLastProgressTime = now;
          this.emit();
        }

        if (state.observation.terminated || state.observation.truncated) {
          this.replayEffectiveTotal = this.replayComputed;
          const names = new Set(state.events.map(event => event.kind_name));
          this.replayTerminalReason = names.has("level_completed") ? "level_completed" :
            state.observation.truncated ? "time_limit" : "game_over";
          if (this.replayComputed < this.replayData.frameCount) {
            const warning = `Playback ended with ${this.replayData.frameCount - this.replayComputed} trailing records.`;
            this.replayWarning = this.replayWarning ? `${this.replayWarning} ${warning}` : warning;
          }
          break;
        }
      }
      if (this.closed || epoch !== this.epoch || this.mode !== "replay") return;
      this.replayComplete = true;
      if (!this.replayTerminalReason) this.replayTerminalReason = "replay_exhausted";
      this.validateReplayOutcome(finalState);
      if (this.replayRequestedFrame !== null) {
        const target = Math.min(this.replayRequestedFrame, this.replayEffectiveTotal);
        this.replayRequestedFrame = null;
        this.replayBuffering = false;
        this.displayReplayFrame(target, {emit: false});
      }
      if (this.replayFrame >= this.replayEffectiveTotal) {
        this.running = false;
        this.displayReplayFrame(this.replayEffectiveTotal, {emit: false});
      }
      this.emit();
    } catch (error) {
      if (!this.closed && epoch === this.epoch) this.report(error);
    }
  }

  async pump() {
    if (this.processing || this.closed) return;
    this.processing = true;
    const pumpEpoch = this.epoch;
    try {
      while (this.pendingTicks && this.running && !this.closed) {
        const epoch = this.epoch;
        const action = this.nextAction();
        this.pendingTicks--;
        const recordIndex = this.recordedWords.length;
        const state = await this.client.step(action.kind, action.x, action.y,
          recordIndex < 2);
        if (epoch !== this.epoch) continue;
        this.recordedWords.push(encodeReplayWord(action.kind, action.x, action.y));
        this.acceptStep(state);
        this.emit();
      }
    } catch (error) {
      if (!this.closed && pumpEpoch === this.epoch) this.report(error);
    }
    finally { this.processing = false; }
  }

  schedule() {
    if (this.closed) return;
    const now = this.now();
    if (this.mode === "replay") {
      if (this.running) {
        const due = now < this.deadline ? 0 :
          Math.min(5, Math.floor((now - this.deadline) / 20) + 1);
        if (due) {
          const available = Math.min(this.replayComputed, this.replayEffectiveTotal);
          const next = Math.min(available, this.replayFrame + due);
          if (next > this.replayFrame) {
            this.replayBuffering = false;
            this.displayReplayFrame(next);
          } else if (!this.replayComplete) {
            this.replayBuffering = true;
            this.emit();
          }
          if (this.replayComplete && this.replayFrame >= this.replayEffectiveTotal) {
            this.running = false;
            this.terminalReason = this.replayTerminalReason;
            this.emit();
          }
          this.deadline += due * 20;
        }
      } else this.deadline = now + 20;
      this.timer = this.clock.setTimeout(() => this.schedule(),
        Math.max(1, this.deadline - this.now()));
      return;
    }
    if (this.running) {
      const due = now < this.deadline ? 0 :
        Math.min(5, Math.floor((now - this.deadline) / 20) + 1);
      if (due) {
        if (this.fastForward) {
          this.pendingTicks = Math.max(this.pendingTicks, FAST_FORWARD_TICKS);
        }
        else this.pendingTicks = Math.min(5, this.pendingTicks + due);
        void this.pump();
      }
      if (this.fastForward && due) this.deadline = now + 20;
      else this.deadline += due * 20;
    } else this.deadline = now + 20;
    this.timer = this.clock.setTimeout(() => this.schedule(),
      Math.max(1, this.deadline - this.now()));
  }

  report(error) {
    this.running = false;
    this.fastForward = false;
    this.pendingTicks = 0;
    this.onSnapshot(null, error);
  }

  close() {
    this.closed = true;
    this.epoch++;
    this.clock.clearTimeout(this.timer);
    this.client?.close();
    this.client = null;
  }
}
