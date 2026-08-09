import {BrowserGame} from "./exact-runtime.js";
import {
  activatedTrailAlphas, colorFor, hasActivatedTrail,
} from "./colors.mjs?v=20260723d";
import {parseReplay, REPLAY_TICK_MS} from "./replay.mjs";

const canvas = document.querySelector("#game");
const ctx = canvas.getContext("2d");
const $ = (selector) => document.querySelector(selector);
const ui = {
  over: $("#gameOver"), again: $("#againButton"), finalTitle: $("#finalTitle"),
  finalScore: $("#finalScore"),
  paused: $("#paused"), pause: $("#pauseButton"), restart: $("#restartButton"),
  app: $(".app"), openReplay: $("#openReplayButton"),
  replayFile: $("#replayFileInput"), saveReplay: $("#saveReplayButton"),
  replayControls: $("#replayControls"), replayName: $("#replayName"),
  replayStatus: $("#replayStatus"), replayError: $("#replayError"),
  replayAnnouncement: $("#replayAnnouncement"),
  replayPlay: $("#replayPlayButton"), replayBack: $("#replayBackButton"),
  replayForward: $("#replayForwardButton"), replayScrubber: $("#replayScrubber"),
  replayPosition: $("#replayPosition"), exitReplay: $("#exitReplayButton"),
  appError: $("#appError"),
  toast: $("#toast"),
};

let snapshot = null;
let previousObservation = null;
let snapshotTime = performance.now();
let aim = {x: 320, y: 390, visible: false};
let started = false;
let lastEvent = -1;
let toastTimer;
let fastForwardTimer;
let game = null;
let trailTick = -1;
let replayLoadEpoch = 0;
let lastReplayAnnouncement = "";
const bodyTrails = new Map();

const fastForwardIdleMs = 160;

function stopFastForward() {
  clearTimeout(fastForwardTimer);
  fastForwardTimer = 0;
  game?.setFastForward(false);
}

function continueFastForward() {
  if (!snapshot?.running) return;
  game?.setFastForward(true);
  clearTimeout(fastForwardTimer);
  fastForwardTimer = setTimeout(stopFastForward, fastForwardIdleMs);
}

function showToast(text) {
  ui.toast.textContent = text;
  ui.toast.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ui.toast.classList.remove("show"), 1200);
}

function showPersistentError(text = "") {
  ui.appError.textContent = text;
  ui.appError.hidden = !text;
}

function announceReplay(key, text) {
  if (key === lastReplayAnnouncement) return;
  lastReplayAnnouncement = key;
  ui.replayAnnouncement.textContent = text;
}

function acceptSnapshot(next, force = false) {
  force ||= Boolean(snapshot && next?.mode !== snapshot.mode);
  force ||= next?.mode === "replay" && snapshot?.mode === "replay" &&
    next.replay.frame < snapshot.replay.frame;
  if (!force && snapshot && next.seed === snapshot.seed &&
      next.observation.tick < snapshot.observation.tick) return;
  if (force || !snapshot || next.seed !== snapshot.seed ||
      next.observation.tick < trailTick) {
    bodyTrails.clear();
    trailTick = -1;
  }
  if (next.observation.tick !== trailTick) {
    const active = new Set();
    for (const body of next.observation.bodies) {
      if (!hasActivatedTrail(body)) continue;
      active.add(body.id);
      const trail = bodyTrails.get(body.id) || [];
      trail.push({x: body.x, y: body.y, angle: body.angle || 0});
      if (trail.length > activatedTrailAlphas.length + 1) trail.shift();
      bodyTrails.set(body.id, trail);
    }
    for (const id of bodyTrails.keys()) {
      if (!active.has(id)) bodyTrails.delete(id);
    }
    trailTick = next.observation.tick;
  }
  previousObservation = force ? null : snapshot?.observation || next.observation;
  snapshot = next;
  snapshotTime = performance.now();
}

function canvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: Math.max(0, Math.min(640, (event.clientX - rect.left) * 640 / rect.width)),
    y: Math.max(0, Math.min(480, (event.clientY - rect.top) * 480 / rect.height)),
  };
}

function shoot(kind = "weak") {
  if (!snapshot || snapshot.observation.terminated || snapshot.observation.truncated) return;
  started = true;
  try { game?.shoot(kind, aim.x, aim.y); }
  catch (error) { showToast(error.message); }
}

function setRunning(running) {
  if (!game) return;
  if (!running) stopFastForward();
  game.setRunning(running);
  started ||= running;
  syncUi();
}

function restart() {
  if (!game) return;
  replayLoadEpoch++;
  stopFastForward();
  ui.replayError.hidden = true;
  showPersistentError();
  const seed = crypto.getRandomValues(new Uint32Array(1))[0];
  game.restart(seed);
  started = true;
  lastEvent = -1;
  syncUi();
}

function replayFilename(state) {
  const now = new Date();
  const part = value => String(value).padStart(2, "0");
  const stamp = `${now.getFullYear()}${part(now.getMonth() + 1)}${part(now.getDate())}_` +
    `${part(now.getHours())}${part(now.getMinutes())}${part(now.getSeconds())}`;
  const score = Math.max(0, state.score).toString().padStart(8, "0").slice(-8);
  return `irisu_${score}_${stamp}_0.rpy`;
}

function saveReplay() {
  if (!game || !snapshot?.can_save_replay) return;
  try {
    const data = game.replayBytes();
    const url = URL.createObjectURL(new Blob([data], {type: "application/octet-stream"}));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = replayFilename(snapshot.observation);
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60_000);
    showToast(`saved ${anchor.download}`);
  } catch (error) {
    showToast(error.message);
  }
}

async function openReplayFile(file) {
  const epoch = ++replayLoadEpoch;
  stopFastForward();
  ui.replayError.hidden = true;
  showPersistentError();
  ui.openReplay.disabled = true;
  ui.replayStatus.textContent = "Loading exact replay…";
  announceReplay("loading", "Loading exact replay");
  try {
    const replay = parseReplay(await file.arrayBuffer());
    if (epoch !== replayLoadEpoch) return;
    const loaded = await game.loadReplay(replay, file.name || "replay.rpy");
    if (!loaded || epoch !== replayLoadEpoch) return;
    showPersistentError();
    started = true;
    lastEvent = Number.MAX_SAFE_INTEGER;
    syncUi();
  } catch (error) {
    if (epoch !== replayLoadEpoch) return;
    ui.replayError.textContent = error.message;
    ui.replayError.hidden = false;
    ui.replayControls.hidden = false;
    showPersistentError(error.message);
    showToast(error.message);
  } finally {
    if (epoch === replayLoadEpoch) ui.openReplay.disabled = !game;
  }
}

function bodyPath(body, size) {
  ctx.beginPath();
  if (body.shape === "circle") ctx.arc(0, 0, size / 2, 0, Math.PI * 2);
  else if (body.shape === "triangle") {
    ctx.moveTo(-size / 2, -size / 2);
    ctx.lineTo(-size / 2, size / 2);
    ctx.lineTo(size / 2, size / 2);
    ctx.closePath();
  } else ctx.rect(-size / 2, -size / 2, size, size);
}

function fillBody(body, size, color, pose, alpha = 1) {
  ctx.save();
  ctx.translate(pose.x, pose.y);
  ctx.rotate(pose.angle || 0);
  ctx.globalAlpha = alpha;
  bodyPath(body, size);
  ctx.fillStyle = color;
  ctx.fill();
  ctx.restore();
}

function drawBody(body, now) {
  const size = Math.max(2, body.size);
  const color = colorFor(body, now);
  if (hasActivatedTrail(body)) {
    const trail = bodyTrails.get(body.id) || [];
    trail.slice(0, -1).forEach((pose, index, echoes) => {
      const alphaOffset = activatedTrailAlphas.length - echoes.length;
      fillBody(body, size, color, pose,
        activatedTrailAlphas[alphaOffset + index]);
    });
  }
  fillBody(body, size, color, body,
    body.lifecycle === "scripted_falling" ? .62 : 1);
  // v2.03 renders rotten pieces with their normal color and shape. The small
  // gray squares in reference footage are projectiles, not dead blocks.
}

function drawBackdrop() {
  ctx.fillStyle = "#0c1517";
  ctx.fillRect(0, 0, 640, 480);
}

function outlinedText(text, x, y, size, align = "left") {
  ctx.save();
  ctx.textAlign = align;
  ctx.font = `italic 900 ${size}px "Trebuchet MS", sans-serif`;
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#55152c";
  ctx.lineWidth = 5;
  ctx.strokeText(text, x, y);
  ctx.fillStyle = "#f0e2a6";
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawHud(state) {
  outlinedText("Level", 21, 428, 28);
  outlinedText(String(state.level), 52, 458, 25, "center");

  const trackX = 151, trackY = 437, trackW = 312, trackH = 15;
  ctx.fillStyle = "#33161eaa";
  ctx.fillRect(trackX, trackY, trackW, trackH);
  const ratio = Math.max(0, Math.min(1, state.gauge / state.gauge_max));
  const gaugeGradient = ctx.createLinearGradient(trackX, 0, trackX + trackW, 0);
  gaugeGradient.addColorStop(0, "#7c1b31");
  gaugeGradient.addColorStop(1, "#b02a3f");
  ctx.fillStyle = gaugeGradient;
  ctx.fillRect(trackX, trackY, trackW * ratio, trackH);
  ctx.fillStyle = "#ffffff18";
  ctx.fillRect(trackX, trackY, trackW, 3);

  const digits = String(state.score).padStart(8, "0");
  ctx.save();
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.font = "900 26px Georgia, serif";
  ctx.strokeStyle = "#681a38";
  ctx.lineWidth = 5;
  ctx.strokeText(digits, 320, 462);
  ctx.fillStyle = "#eee0a4";
  ctx.fillText(digits, 320, 462);
  ctx.restore();
}

function drawWalls(state) {
  const f = state.field;
  const thick = 16;
  ctx.fillStyle = "#f3f3ef";
  ctx.fillRect(f.x, f.y, thick, f.height);
  ctx.fillRect(f.x + f.width + thick / 2, f.y, thick, f.height);
  ctx.fillRect(f.x, f.y + f.height + 40, f.width + thick * 2, thick);
  ctx.fillRect(f.x + thick, 0, f.width, 10);
  ctx.fillStyle = "#cad0cd55";
  ctx.fillRect(f.x, f.y, 3, f.height);
  ctx.fillRect(f.x + f.width + thick / 2, f.y, 3, f.height);
}

function interpolatedBodies(now) {
  const current = snapshot.observation;
  if (!previousObservation || previousObservation.tick === current.tick) return current.bodies;
  const tickGap = current.tick - previousObservation.tick;
  if (tickGap <= 0 || tickGap > 4) return current.bodies;
  const alpha = Math.min(1, (now - snapshotTime) / Math.min(60, Math.max(20, tickGap * 20)));
  const oldBodies = new Map(previousObservation.bodies.map((body) => [body.id, body]));
  return current.bodies.map((body) => {
    const old = oldBodies.get(body.id);
    if (!old) return body;
    let angleDelta = (body.angle - old.angle) % (Math.PI * 2);
    if (angleDelta > Math.PI) angleDelta -= Math.PI * 2;
    if (angleDelta < -Math.PI) angleDelta += Math.PI * 2;
    return {
      ...body,
      x: old.x + (body.x - old.x) * alpha,
      y: old.y + (body.y - old.y) * alpha,
      angle: old.angle + angleDelta * alpha,
    };
  });
}

function draw(now) {
  drawBackdrop();
  if (snapshot) {
    const state = snapshot.observation;
    drawWalls(state);
    [...interpolatedBodies(now)].sort((a, b) => a.id - b.id)
      .forEach((body) => drawBody(body, now));
    drawHud(state);
  }
  if (aim.visible) {
    ctx.save();
    ctx.strokeStyle = "#ece8dd";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(aim.x, aim.y, 9, 0, Math.PI * 2);
    ctx.moveTo(aim.x - 15, aim.y); ctx.lineTo(aim.x - 5, aim.y);
    ctx.moveTo(aim.x + 5, aim.y); ctx.lineTo(aim.x + 15, aim.y);
    ctx.moveTo(aim.x, aim.y - 15); ctx.lineTo(aim.x, aim.y - 5);
    ctx.moveTo(aim.x, aim.y + 5); ctx.lineTo(aim.x, aim.y + 15);
    ctx.stroke();
    ctx.restore();
  }
  requestAnimationFrame(draw);
}

function processEvents(events) {
  for (const event of events) {
    if (event.sequence <= lastEvent) continue;
    lastEvent = event.sequence;
    if (event.kind_name === "level_changed") showToast(`LEVEL ${event.value}`);
  }
}

function syncUi() {
  if (!snapshot) return;
  const state = snapshot.observation;
  const replay = snapshot.replay;
  const replayMode = snapshot.mode === "replay";
  started ||= snapshot.running || state.tick > 0;
  ui.pause.firstChild.textContent = snapshot.running ? "pause " : "resume ";
  ui.pause.disabled = replayMode && replay.frame >= replay.total_frames;
  ui.paused.hidden = !started || snapshot.running || state.terminated || state.truncated;
  ui.over.hidden = replayMode ?
    !(replay.complete && replay.frame >= replay.total_frames) :
    !(state.terminated || state.truncated);
  if (!ui.over.hidden) {
    ui.finalTitle.textContent = replayMode ?
      snapshot.terminal_reason === "replay_exhausted" ? "Replay exhausted" : "Replay complete" :
      snapshot.terminal_reason === "level_completed" ? "Level 100 complete" :
        snapshot.terminal_reason === "time_limit" ? "Time limit" : "Game over";
    ui.finalScore.textContent = `${String(state.score).padStart(8, "0")} · level ${state.level}`;
  }
  ui.saveReplay.hidden = replayMode || !snapshot.can_save_replay;
  ui.again.textContent = replayMode ? "new game" : "try again";
  ui.replayControls.hidden = !replayMode;
  ui.app.classList.toggle("replay-mode", replayMode);
  document.documentElement.dataset.mode = snapshot.mode;
  if (replayMode) {
    if (replay.cursor) aim = {x: replay.cursor.x, y: replay.cursor.y, visible: true};
    else aim.visible = false;
    ui.replayName.textContent = replay.name;
    ui.replayScrubber.max = String(replay.total_frames);
    ui.replayScrubber.value = String(replay.frame);
    ui.replayScrubber.setAttribute("aria-valuetext",
      `frame ${replay.frame} of ${replay.total_frames}`);
    ui.replayPosition.value = `frame ${replay.frame} / ${replay.total_frames}`;
    const seconds = (replay.frame * REPLAY_TICK_MS / 1000).toFixed(1);
    const totalSeconds = (replay.total_frames * REPLAY_TICK_MS / 1000).toFixed(1);
    ui.replayPosition.textContent = `${replay.frame} / ${replay.total_frames} · ${seconds}s / ${totalSeconds}s`;
    ui.replayPlay.textContent = snapshot.running ? "pause" : "play";
    ui.replayPlay.disabled = replay.frame >= replay.total_frames;
    ui.replayBack.disabled = replay.frame <= 0;
    ui.replayForward.disabled = replay.frame >= replay.total_frames;
    ui.replayStatus.textContent = replay.buffering ?
      `Buffering ${replay.buffered_frames} / ${replay.total_frames}…` :
      replay.complete ? (replay.warning || "Exact replay ready") :
        `Prepared ${replay.buffered_frames} / ${replay.total_frames}`;
    if (replay.complete) announceReplay("complete", "Replay is ready");
    else if (replay.buffering) announceReplay("buffering", "Buffering replay");
    ui.replayError.textContent = replay.warning;
    ui.replayError.hidden = !replay.warning;
    document.documentElement.dataset.replayFrame = String(replay.frame);
    document.documentElement.dataset.replayBuffered = String(replay.buffered_frames);
  } else {
    lastReplayAnnouncement = "";
    ui.replayError.hidden = true;
    delete document.documentElement.dataset.replayFrame;
    delete document.documentElement.dataset.replayBuffered;
    processEvents(snapshot.events);
  }
}

function receiveSnapshot(next, error) {
  if (error) {
    document.documentElement.dataset.ready = "false";
    document.documentElement.dataset.error = error.message;
    ui.replayError.textContent = error.message;
    ui.replayError.hidden = false;
    showPersistentError(error.message);
    showToast(error.message);
    return;
  }
  acceptSnapshot(next);
  document.documentElement.dataset.ready = "true";
  delete document.documentElement.dataset.error;
  document.documentElement.dataset.tick = String(next.observation.tick);
  document.documentElement.dataset.seed = String(next.seed);
  syncUi();
}

canvas.addEventListener("pointermove", (event) => {
  if (snapshot?.mode === "replay") return;
  aim = {...canvasPoint(event), visible: true};
  game?.setAim(aim.x, aim.y);
});
canvas.addEventListener("pointerleave", () => {
  if (snapshot?.mode !== "replay") aim.visible = false;
});
canvas.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  if (snapshot?.mode === "replay") return;
  aim = {...canvasPoint(event), visible: true};
  game?.setAim(aim.x, aim.y);
  shoot(event.shiftKey ? "both" : event.button === 2 ? "strong" : "weak");
});
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  if (snapshot?.mode === "replay") return;
  if (event.deltaY > 0) continueFastForward();
  else if (event.deltaY < 0) stopFastForward();
}, {passive: false});
ui.pause.addEventListener("click", () => setRunning(!snapshot?.running));
ui.restart.addEventListener("click", restart);
ui.again.addEventListener("click", restart);
ui.saveReplay.addEventListener("click", saveReplay);
ui.openReplay.addEventListener("click", () => {
  if (ui.replayFile.showPicker) ui.replayFile.showPicker();
  else ui.replayFile.click();
});
ui.replayFile.addEventListener("change", () => {
  const file = ui.replayFile.files?.[0];
  ui.replayFile.value = "";
  if (file) void openReplayFile(file);
});
ui.replayPlay.addEventListener("click", () => setRunning(!snapshot?.running));
ui.replayBack.addEventListener("click", () => game?.stepReplay(-1));
ui.replayForward.addEventListener("click", () => game?.stepReplay(1));
ui.replayScrubber.addEventListener("input", () => {
  const frame = Number(ui.replayScrubber.value);
  ui.replayPosition.textContent = `frame ${frame} / ${ui.replayScrubber.max}`;
});
ui.replayScrubber.addEventListener("change", () => game?.seekReplay(
  Number(ui.replayScrubber.value)));
ui.exitReplay.addEventListener("click", restart);
window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement || event.target instanceof HTMLButtonElement ||
      event.target?.isContentEditable) return;
  if (event.code === "Space") { event.preventDefault(); setRunning(!snapshot?.running); }
  if (event.key.toLowerCase() === "r") restart();
  if (snapshot?.mode !== "replay" && event.key.toLowerCase() === "w") shoot("weak");
  if (snapshot?.mode !== "replay" && event.key.toLowerCase() === "s") shoot("strong");
});
window.addEventListener("blur", stopFastForward);

draw();
BrowserGame.create(receiveSnapshot).then((instance) => {
  game = instance;
  game.setAim(aim.x, aim.y);
  document.documentElement.dataset.backend = "exact-v86";
  document.documentElement.dataset.ready = "true";
  ui.openReplay.disabled = false;
})
  .catch((error) => receiveSnapshot(null, error));
