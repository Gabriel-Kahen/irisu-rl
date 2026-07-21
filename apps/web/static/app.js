import {BrowserGame} from "./wasm-runtime.js";

const canvas = document.querySelector("#game");
const ctx = canvas.getContext("2d");
const $ = (selector) => document.querySelector(selector);
const ui = {
  connection: $("#connection"), start: $("#startCard"), startButton: $("#startButton"),
  over: $("#gameOver"), again: $("#againButton"), finalTitle: $("#finalTitle"), finalScore: $("#finalScore"),
  paused: $("#paused"), pause: $("#pauseButton"), restart: $("#restartButton"),
  toast: $("#toast"),
};

// The game's early levels use these flat, deliberately harsh primary colors.
const palette = ["#861f00", "#0005a4", "#9a9000", "#257642", "#713380", "#ae6311", "#1b747a", "#92335f"];
const activatedPalette = ["#e44717", "#2945ff", "#eee116", "#43c56d", "#b05ac2", "#ef9c27", "#35bdc4", "#e35b98"];
let snapshot = null;
let previousObservation = null;
let snapshotTime = performance.now();
let aim = {x: 320, y: 390, visible: false};
let started = false;
let lastEvent = -1;
let toastTimer;
let fastForwardTimer;
let game = null;

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

function acceptSnapshot(next, force = false) {
  if (!force && snapshot && next.seed === snapshot.seed &&
      next.observation.tick < snapshot.observation.tick) return;
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
  ui.start.hidden = true;
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
  stopFastForward();
  const seed = crypto.getRandomValues(new Uint32Array(1))[0];
  game.restart(seed);
  started = true;
  lastEvent = -1;
  syncUi();
}

function colorFor(body) {
  if (body.kind === "projectile") return "#d9dcda";
  if (body.kind === "bonus") return "#eee06b";
  const index = ((body.color % palette.length) + palette.length) % palette.length;
  const active = body.lifecycle === "dynamic_fresh" || body.lifecycle === "confirmed";
  return (active ? activatedPalette : palette)[index];
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

function drawBody(body) {
  const size = Math.max(2, body.size);
  const color = colorFor(body);
  ctx.save();
  ctx.translate(body.x, body.y);
  ctx.rotate(body.angle || 0);
  ctx.globalAlpha = body.lifecycle === "scripted_falling" ? .62 : 1;
  bodyPath(body, size);
  ctx.fillStyle = color;
  ctx.fill();

  // v2.03 renders rotten pieces with their normal color and shape. The small
  // gray squares in reference footage are projectiles, not dead blocks.
  ctx.restore();
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
    [...interpolatedBodies(now)].sort((a, b) => a.id - b.id).forEach(drawBody);
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
    if (event.kind_name === "score_changed" && event.value > 0) showToast(`+${event.value}`);
  }
}

function syncUi() {
  if (!snapshot) return;
  const state = snapshot.observation;
  started ||= snapshot.running || state.tick > 0;
  ui.start.hidden = started;
  ui.pause.firstChild.textContent = snapshot.running ? "pause " : "resume ";
  ui.paused.hidden = !started || snapshot.running || state.terminated || state.truncated;
  ui.over.hidden = !(state.terminated || state.truncated);
  if (!ui.over.hidden) {
    ui.finalTitle.textContent = snapshot.terminal_reason === "level_completed" ? "Level 100 complete" : snapshot.terminal_reason === "time_limit" ? "Time limit" : "Game over";
    ui.finalScore.textContent = `${String(state.score).padStart(8, "0")} · level ${state.level}`;
  }
  processEvents(snapshot.events);
}

function receiveSnapshot(next, error) {
  if (error) {
    ui.connection.className = "connection error";
    ui.connection.lastElementChild.textContent = "engine error";
    showToast(error.message);
    return;
  }
  acceptSnapshot(next);
  ui.connection.className = "connection live";
  ui.connection.lastElementChild.textContent = snapshot.fastForward ? "fast" :
    snapshot.running ? "live" : "ready";
  syncUi();
}

canvas.addEventListener("pointermove", (event) => { aim = {...canvasPoint(event), visible: true}; });
canvas.addEventListener("pointerleave", () => { aim.visible = false; });
canvas.addEventListener("pointerdown", (event) => {
  event.preventDefault();
  aim = {...canvasPoint(event), visible: true};
  shoot(event.shiftKey ? "both" : event.button === 2 ? "strong" : "weak");
});
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  if (event.deltaY > 0) continueFastForward();
  else if (event.deltaY < 0) stopFastForward();
}, {passive: false});
ui.startButton.addEventListener("click", () => setRunning(true));
ui.pause.addEventListener("click", () => setRunning(!snapshot?.running));
ui.restart.addEventListener("click", restart);
ui.again.addEventListener("click", restart);
window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  if (event.code === "Space") { event.preventDefault(); setRunning(!snapshot?.running); }
  if (event.key.toLowerCase() === "r") restart();
  if (event.key.toLowerCase() === "w") shoot("weak");
  if (event.key.toLowerCase() === "s") shoot("strong");
});
window.addEventListener("blur", stopFastForward);

draw();
BrowserGame.create(receiveSnapshot).then((instance) => { game = instance; })
  .catch((error) => receiveSnapshot(null, error));
