#!/usr/bin/env python3
"""Local web server for the playable headless clone."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).resolve().parent / "static"
sys.path.insert(0, str(ROOT / "python"))

from irisu_env import Action, IrisuEnv  # noqa: E402


def plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    item = getattr(value, "item", None)
    return plain(item()) if callable(item) else value


class Game:
    frame_seconds = 0.02

    def __init__(self, seed: int, library: str | None) -> None:
        self.lock = threading.RLock()
        self.wake = threading.Event()
        self.closed = False
        self.running = False
        self.seed = seed
        self.queue: deque[tuple[str, float, float]] = deque(maxlen=12)
        self.events: deque[dict[str, Any]] = deque(maxlen=80)
        self.release_next = False
        self.terminal_reason: str | None = None
        self.env = IrisuEnv(library_path=library)
        self.observation, _ = self.env.reset(seed=seed)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _step(self) -> None:
        if self.release_next:
            action = Action.wait()
            self.release_next = False
        elif self.queue:
            kind, x, y = self.queue.popleft()
            action = {"weak": Action.weak, "strong": Action.strong, "both": Action.both}[kind](x, y)
            self.release_next = True
        else:
            action = Action.wait()

        state, _, terminated, truncated, info = self.env.step(action)
        self.observation = state
        self.events.extend(info["events"])
        if terminated or truncated:
            names = {event["kind_name"] for event in info["events"]}
            self.terminal_reason = (
                "level_completed" if "level_completed" in names else
                "time_limit" if truncated else "game_over"
            )
            self.running = False
            self.queue.clear()

    def _loop(self) -> None:
        deadline = time.monotonic()
        while not self.closed:
            with self.lock:
                if self.running:
                    now = time.monotonic()
                    due = min(5, max(0, int((now - deadline) / self.frame_seconds) + 1))
                    for _ in range(due):
                        if self.running:
                            self._step()
                    deadline += due * self.frame_seconds
                    delay = max(0.001, min(self.frame_seconds, deadline - time.monotonic()))
                else:
                    deadline = time.monotonic()
                    delay = self.frame_seconds
            self.wake.wait(delay)
            self.wake.clear()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return plain({
                "observation": self.observation,
                "events": list(self.events),
                "running": self.running,
                "seed": self.seed,
                "terminal_reason": self.terminal_reason,
            })

    def shoot(self, kind: str, x: float, y: float) -> None:
        if kind not in {"weak", "strong", "both"}:
            raise ValueError("kind must be weak, strong, or both")
        if not 0 <= x <= 640 or not 0 <= y <= 480:
            raise ValueError("shot coordinates must be inside 640x480")
        with self.lock:
            if self.observation["terminated"] or self.observation["truncated"]:
                raise ValueError("restart before shooting")
            self.queue.append((kind, x, y))
            self.running = True
        self.wake.set()

    def set_running(self, running: bool) -> None:
        with self.lock:
            terminal = self.observation["terminated"] or self.observation["truncated"]
            self.running = bool(running and not terminal)
        self.wake.set()

    def restart(self, seed: int) -> None:
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("seed must fit in uint32")
        with self.lock:
            self.observation, _ = self.env.reset(seed=seed)
            self.seed = seed
            self.queue.clear()
            self.events.clear()
            self.release_next = False
            self.terminal_reason = None
            self.running = True
        self.wake.set()

    def close(self) -> None:
        self.closed = True
        self.wake.set()
        self.thread.join(timeout=1)
        self.env.close()


class Handler(BaseHTTPRequestHandler):
    game: Game
    files = {
        "/": ("index.html", "text/html; charset=utf-8"),
        "/index.html": ("index.html", "text/html; charset=utf-8"),
        "/styles.css": ("styles.css", "text/css; charset=utf-8"),
        "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    }

    def log_message(self, *_: object) -> None:
        return

    def send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length > 4096:
            raise ValueError("request too large")
        value = json.loads(self.rfile.read(length) or b"{}")
        if not isinstance(value, dict):
            raise ValueError("request body must be an object")
        return value

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/healthz":
            self.send_json({"status": "ok"})
            return
        if path == "/api/state":
            self.send_json(self.game.snapshot())
            return
        static = self.files.get(path)
        if not static:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, mime = static
        data = (STATIC / filename).read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = self.body()
            path = urlparse(self.path).path
            if path == "/api/action":
                self.game.shoot(str(body.get("kind", "")), float(body["x"]), float(body["y"]))
            elif path == "/api/control":
                if not isinstance(body.get("running"), bool):
                    raise ValueError("running must be boolean")
                self.game.set_running(body["running"])
            elif path == "/api/restart":
                self.game.restart(int(body.get("seed", self.game.seed)))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_json(self.game.snapshot())
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("IRISU_SEED", "42")))
    parser.add_argument("--library", default=os.environ.get("IRISU_CLONE_LIBRARY"))
    args = parser.parse_args()
    game = Game(args.seed, args.library)
    Handler.game = game
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"IriSu frontend: http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        game.close()


if __name__ == "__main__":
    main()
