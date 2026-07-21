// Emscripten contract: the module has an async default factory and cwrap. The
// four irisu_web_* functions exchange complete snapshots as JSON strings.
import createIrisuModule from "./irisu-wasm.js";

const kinds = {weak: 1, strong: 2, both: 3};

export class BrowserGame {
  static async create(onSnapshot) {
    const module = await createIrisuModule({
      locateFile: (file) => new URL(file, import.meta.url).href,
    });
    const api = {
      create: module.cwrap("irisu_web_create", "number", ["number"]),
      destroy: module.cwrap("irisu_web_destroy", null, ["number"]),
      reset: module.cwrap("irisu_web_reset", "string", ["number", "number"]),
      step: module.cwrap("irisu_web_step", "string", ["number", "number", "number", "number"]),
    };
    return new BrowserGame(api, onSnapshot);
  }

  constructor(api, onSnapshot) {
    this.api = api;
    this.onSnapshot = onSnapshot;
    this.seed = crypto.getRandomValues(new Uint32Array(1))[0];
    this.handle = api.create(this.seed);
    if (!this.handle) throw new Error("could not create simulator");
    this.running = false;
    this.queue = [];
    this.events = [];
    this.timer = 0;
    this.restart(this.seed, false);
    this.loop();
  }

  read(json) {
    if (!json) throw new Error("simulator error");
    return JSON.parse(json);
  }

  emit() {
    this.onSnapshot({observation: this.observation, events: this.events,
      running: this.running, seed: this.seed, terminal_reason: this.terminalReason});
  }

  restart(seed, running = true) {
    this.seed = seed >>> 0;
    const state = this.read(this.api.reset(this.handle, this.seed));
    this.observation = state.observation;
    this.queue.length = 0;
    this.events.length = 0;
    this.releaseNext = false;
    this.terminalReason = null;
    this.running = running;
    this.deadline = performance.now() + 20;
    this.emit();
  }

  shoot(kind, x, y) {
    if (!(kind in kinds)) throw new Error(`unknown shot kind: ${kind}`);
    if (this.observation.terminated || this.observation.truncated) return;
    if (this.queue.length < 12) this.queue.push({kind: kinds[kind], x, y});
    if (!this.running) this.deadline = performance.now() + 20;
    this.running = true;
    this.emit();
  }

  setRunning(running) {
    this.running = Boolean(running && !this.observation.terminated &&
      !this.observation.truncated);
    this.deadline = performance.now() + 20;
    this.emit();
  }

  step() {
    let action = {kind: 0, x: 0, y: 0};
    if (this.releaseNext) this.releaseNext = false;
    else if (this.queue.length) {
      action = this.queue.shift();
      this.releaseNext = true;
    }
    const state = this.read(this.api.step(this.handle, action.kind, action.x, action.y));
    this.observation = state.observation;
    this.events.push(...state.events);
    if (this.events.length > 80) this.events.splice(0, this.events.length - 80);
    if (this.observation.terminated || this.observation.truncated) {
      const names = new Set(state.events.map((event) => event.kind_name));
      this.terminalReason = names.has("level_completed") ? "level_completed" :
        this.observation.truncated ? "time_limit" : "game_over";
      this.running = false;
      this.queue.length = 0;
    }
  }

  loop() {
    const now = performance.now();
    if (this.running) {
      const due = now < this.deadline ? 0 :
        Math.min(5, Math.floor((now - this.deadline) / 20) + 1);
      try {
        for (let i = 0; i < due && this.running; i++) this.step();
        if (due) this.emit();
      } catch (error) {
        this.running = false;
        this.onSnapshot(null, error);
      }
      this.deadline += due * 20;
    } else this.deadline = now + 20;
    this.timer = window.setTimeout(() => this.loop(),
      Math.max(1, this.deadline - performance.now()));
  }

  close() {
    clearTimeout(this.timer);
    if (this.handle) this.api.destroy(this.handle);
    this.handle = 0;
  }
}
