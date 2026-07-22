#!/usr/bin/env python3
"""64-bit Python client and benchmark for the persistent exact-physics worker."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import struct
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence as AbstractSequence
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import BinaryIO, Sequence


MAGIC = 0x43505249
VERSION = 1
BODY_CAPACITY = 196
EXACT_BACKEND = "exact-msvc9-r58-multiworld-forward"
CANONICAL_X87_CONTROL_WORD = 0x027F

HELLO = 1
RESET = 2
STEP = 3
OBSERVE = 4
CLOSE = 5
CONFIGURE = 6
CONFIG_JSON = 7

HEADER = struct.Struct("<IHHII")
STATUS = struct.Struct("<i")
HELLO_FIXED = struct.Struct("<IIIIQII")
RESET_REQUEST = struct.Struct("<Q")
STEP_REQUEST = struct.Struct("<IddII")
OBSERVATION_HEADER = struct.Struct("<QqqqQddddddIIIII4B")
BODY = struct.Struct("<QqQdddddddIiII4B")
TRANSITION = struct.Struct("<qQQQqQqQIIIIBBBB")
EVENT = struct.Struct("<QQqIIHBB")
CONFIG_COUNT = struct.Struct("<I")
CONFIG_KEY_SIZE = struct.Struct("<H")
CONFIG_VALUE = struct.Struct("<d")
CONFIG_HASH = struct.Struct("<Q")


class ProtocolError(RuntimeError):
    """The worker emitted a malformed or mismatched response."""


class WorkerError(RuntimeError):
    """The worker rejected a validly framed request."""


def valid_exact_library_sha256(value: str) -> bool:
    return (
        len(value) == 64
        and value != "0" * 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


@dataclass(frozen=True, slots=True)
class WorkerInfo:
    protocol_version: int
    pointer_bits: int
    body_capacity: int
    pid: int
    config_hash: int
    x87_control_word: int
    process_model: int
    backend: str
    compiler: str
    exact_library_sha256: str


@dataclass(frozen=True, slots=True)
class BodyState:
    age_ticks: int
    remaining_lifetime: int
    rot_timer: int
    x: float
    y: float
    vx: float
    vy: float
    angle: float
    angular_velocity: float
    size: float
    id: int
    color: int
    chain_id: int
    projectile_hits: int
    kind: int
    shape: int
    lifecycle: int


@dataclass(frozen=True, slots=True)
class Observation:
    tick: int
    score: int
    gauge: int
    gauge_max: int
    qualifying_clear_count: int
    field_x: float
    field_y: float
    field_width: float
    field_height: float
    side_wall_top: float
    side_wall_bottom: float
    level: int
    active_colors: int
    spawn_interval_ticks: int
    highest_chain: int
    terminated: bool
    truncated: bool
    left_held: bool
    right_held: bool
    bodies: tuple[BodyState, ...]


@dataclass(frozen=True, slots=True)
class EventState:
    tick: int
    sequence: int
    value: int
    a: int
    b: int
    kind: int
    detail: str


@dataclass(frozen=True, slots=True)
class Transition:
    observation: Observation
    reward: int
    event_count: int
    config_hash: int
    finish_call_count: int
    recorded_final_score: int
    recorded_final_clears: int
    latest_final_score: int
    latest_final_clears: int
    recorded_final_highest_chain: int
    recorded_final_level: int
    latest_final_highest_chain: int
    latest_final_level: int
    terminated: bool
    truncated: bool
    terminal_metadata_recorded: bool
    invalid_action: bool
    events: tuple[EventState, ...]


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"worker response ended with {remaining} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(stream: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    while view:
        count = stream.write(view)
        if count is None or count <= 0:
            raise BrokenPipeError("worker request stream accepted no bytes")
        view = view[count:]
    stream.flush()


def flatten_config(config: Mapping[str, object]) -> list[tuple[str, float]]:
    """Match the native binding's flattened key/double override contract."""

    def number(value: object, label: str) -> float:
        if not isinstance(value, Real) or isinstance(value, bool):
            raise TypeError(f"configuration value {label} must be numeric")
        encoded = float(value)
        if isinstance(value, Integral) and int(encoded) != int(value):
            raise ValueError(
                f"configuration integer {label} is not exactly representable"
            )
        return encoded

    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")
    result: list[tuple[str, float]] = []
    for key, value in config.items():
        if not isinstance(key, str) or not key:
            raise TypeError("configuration keys must be non-empty strings")
        if "\0" in key:
            raise ValueError("configuration keys must not contain NUL characters")
        if isinstance(value, AbstractSequence) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            for index, item in enumerate(value):
                label = f"{key}[{index}]"
                result.append((label, number(item, label)))
        else:
            result.append((key, number(value, key)))
    result.sort(key=lambda item: item[0])
    return result


def decode_observation(payload: bytes, offset: int = 0) -> tuple[Observation, int]:
    if len(payload) - offset < OBSERVATION_HEADER.size:
        raise ProtocolError("truncated observation header")
    values = OBSERVATION_HEADER.unpack_from(payload, offset)
    offset += OBSERVATION_HEADER.size
    body_count = values[15]
    if body_count > BODY_CAPACITY:
        raise ProtocolError(f"observation body count {body_count} exceeds capacity")
    required = body_count * BODY.size
    if len(payload) - offset < required:
        raise ProtocolError("truncated observation bodies")
    bodies: list[BodyState] = []
    for _ in range(body_count):
        body = BODY.unpack_from(payload, offset)
        offset += BODY.size
        if body[14] > 2 or body[15] > 2 or body[16] > 4 or body[17] != 0:
            raise ProtocolError("observation contains an invalid body enum or flags")
        bodies.append(BodyState(*body[:17]))
    observation = Observation(
        tick=values[0],
        score=values[1],
        gauge=values[2],
        gauge_max=values[3],
        qualifying_clear_count=values[4],
        field_x=values[5],
        field_y=values[6],
        field_width=values[7],
        field_height=values[8],
        side_wall_top=values[9],
        side_wall_bottom=values[10],
        level=values[11],
        active_colors=values[12],
        spawn_interval_ticks=values[13],
        highest_chain=values[14],
        terminated=bool(values[16]),
        truncated=bool(values[17]),
        left_held=bool(values[18]),
        right_held=bool(values[19]),
        bodies=tuple(bodies),
    )
    return observation, offset


def decode_transition(payload: bytes) -> Transition:
    observation, offset = decode_observation(payload)
    if len(payload) - offset < TRANSITION.size:
        raise ProtocolError("transition has an invalid diagnostics suffix")
    values = TRANSITION.unpack_from(payload, offset)
    offset += TRANSITION.size
    if bool(values[12]) != observation.terminated:
        raise ProtocolError("transition termination flag disagrees with observation")
    if bool(values[13]) != observation.truncated:
        raise ProtocolError("transition truncation flag disagrees with observation")
    events: list[EventState] = []
    for _ in range(values[1]):
        if len(payload) - offset < EVENT.size:
            raise ProtocolError("truncated event header")
        event = EVENT.unpack_from(payload, offset)
        offset += EVENT.size
        detail_size, kind, reserved = event[5:]
        if kind > 18 or reserved:
            raise ProtocolError("event contains an invalid kind or flags")
        if len(payload) - offset < detail_size:
            raise ProtocolError("truncated event detail")
        try:
            detail = payload[offset : offset + detail_size].decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProtocolError("event detail is not UTF-8") from error
        offset += detail_size
        events.append(
            EventState(
                tick=event[0],
                sequence=event[1],
                value=event[2],
                a=event[3],
                b=event[4],
                kind=kind,
                detail=detail,
            )
        )
    if offset != len(payload):
        raise ProtocolError("transition response has trailing bytes")
    return Transition(
        observation=observation,
        reward=values[0],
        event_count=values[1],
        config_hash=values[2],
        finish_call_count=values[3],
        recorded_final_score=values[4],
        recorded_final_clears=values[5],
        latest_final_score=values[6],
        latest_final_clears=values[7],
        recorded_final_highest_chain=values[8],
        recorded_final_level=values[9],
        latest_final_highest_chain=values[10],
        latest_final_level=values[11],
        terminated=bool(values[12]),
        truncated=bool(values[13]),
        terminal_metadata_recorded=bool(values[14]),
        invalid_action=bool(values[15]),
        events=tuple(events),
    )


class ExactWorkerClient:
    """Own one exact 32-bit worker and serialize access to its single world."""

    def __init__(
        self,
        worker: str | Path,
        *,
        config: Mapping[str, object] | None = None,
    ) -> None:
        executable = Path(worker).expanduser().resolve()
        if not executable.is_file():
            raise FileNotFoundError(executable)
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("LD_") and key != "GLIBC_TUNABLES"
        }
        if trace_path := environment.get("IRISU_EXACT_TRACE"):
            environment["IRISU_EXACT_TRACE"] = os.path.abspath(trace_path)
        environment["IRISU_EXACT_CW"] = hex(CANONICAL_X87_CONTROL_WORD)
        self._process = subprocess.Popen(
            [str(executable)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            cwd=os.sep,
            env=environment,
        )
        if self._process.stdin is None or self._process.stdout is None:
            raise RuntimeError("worker pipes were not created")
        self._lock = threading.RLock()
        self._request_id = 0
        self._pending: tuple[int, int] | None = None
        self._closed = False
        self.last_response_bytes = 0
        self.info = self._hello()
        if self.info.protocol_version != VERSION:
            self.close()
            raise ProtocolError("worker reported an incompatible protocol version")
        if (
            self.info.pointer_bits != 32
            or self.info.body_capacity != BODY_CAPACITY
            or self.info.process_model != 1
        ):
            self.close()
            raise ProtocolError("worker does not expose the expected exact ABI")
        if self.info.x87_control_word != CANONICAL_X87_CONTROL_WORD:
            self.close()
            raise ProtocolError(
                "worker does not expose the canonical x87 control word"
            )
        if self.info.backend != EXACT_BACKEND:
            self.close()
            raise ProtocolError(
                "worker is not the required exact multiworld backend "
                f"(expected {EXACT_BACKEND!r}, got {self.info.backend!r})"
            )
        if not valid_exact_library_sha256(self.info.exact_library_sha256):
            self.close()
            raise ProtocolError(
                "worker did not report a valid non-placeholder exact-library SHA-256"
            )
        self.current_config_hash = self.info.config_hash
        if config is not None:
            self.configure(config)

    def _begin_request(self, opcode: int, payload: bytes = b"") -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("worker client is closed")
            if self._pending is not None:
                raise RuntimeError("worker already has an outstanding request")
            self._request_id = (self._request_id + 1) & 0xFFFFFFFF
            if self._request_id == 0:
                self._request_id = 1
            request_id = self._request_id
            header = HEADER.pack(MAGIC, VERSION, opcode, request_id, len(payload))
            assert self._process.stdin is not None
            try:
                _write_all(self._process.stdin, header + payload)
            except BrokenPipeError as error:
                code = self._process.poll()
                raise WorkerError(f"worker exited unexpectedly (status={code})") from error
            self._pending = (opcode, request_id)

    def _finish_response(self, opcode: int) -> bytes:
        with self._lock:
            if self._closed:
                raise RuntimeError("worker client is closed")
            if self._pending is None or self._pending[0] != opcode:
                raise RuntimeError("worker has no matching outstanding request")
            _, request_id = self._pending
            try:
                assert self._process.stdout is not None
                response_header = _read_exact(self._process.stdout, HEADER.size)
                magic, version, response_opcode, response_id, size = HEADER.unpack(
                    response_header
                )
                if magic != MAGIC or version != VERSION:
                    raise ProtocolError("worker returned an invalid frame header")
                if response_opcode != opcode or response_id != request_id:
                    raise ProtocolError("worker response does not match request")
                if size < STATUS.size or size > (4 << 20):
                    raise ProtocolError("worker returned an invalid payload size")
                response = _read_exact(self._process.stdout, size)
            except EOFError as error:
                code = self._process.poll()
                raise WorkerError(f"worker exited unexpectedly (status={code})") from error
            finally:
                self._pending = None
            self.last_response_bytes = HEADER.size + size
            status = STATUS.unpack_from(response)[0]
            content = response[STATUS.size :]
            if status:
                message = content.decode("utf-8", errors="replace")
                raise WorkerError(f"worker status {status}: {message}")
            return content

    def _request(self, opcode: int, payload: bytes = b"") -> bytes:
        self._begin_request(opcode, payload)
        return self._finish_response(opcode)

    def _hello(self) -> WorkerInfo:
        payload = self._request(HELLO)
        if len(payload) < HELLO_FIXED.size:
            raise ProtocolError("hello response has the wrong size")
        values = HELLO_FIXED.unpack_from(payload)
        offset = HELLO_FIXED.size
        strings: list[str] = []
        for _ in range(3):
            if len(payload) - offset < 2:
                raise ProtocolError("truncated hello string length")
            size = struct.unpack_from("<H", payload, offset)[0]
            offset += 2
            if len(payload) - offset < size:
                raise ProtocolError("truncated hello string")
            try:
                strings.append(payload[offset : offset + size].decode("utf-8"))
            except UnicodeDecodeError as error:
                raise ProtocolError("hello string is not UTF-8") from error
            offset += size
        if offset != len(payload):
            raise ProtocolError("hello response has trailing bytes")
        return WorkerInfo(*values, *strings)

    def reset(self, seed: int = 0) -> Observation:
        if not 0 <= seed <= 0xFFFFFFFF:
            raise ValueError("normal-mode seed must fit in uint32")
        payload = self._request(RESET, RESET_REQUEST.pack(seed))
        observation, offset = decode_observation(payload)
        if offset != len(payload):
            raise ProtocolError("reset response has trailing bytes")
        return observation

    def observe(self) -> Observation:
        payload = self._request(OBSERVE)
        observation, offset = decode_observation(payload)
        if offset != len(payload):
            raise ProtocolError("observation response has trailing bytes")
        return observation

    def configure(self, config: Mapping[str, object]) -> int:
        flattened = flatten_config(config)
        if len(flattened) > 1024:
            raise ValueError("configuration override count exceeds worker limit")
        payload = bytearray(CONFIG_COUNT.pack(len(flattened)))
        for key, value in flattened:
            encoded = key.encode("utf-8")
            if len(encoded) > 0xFFFF:
                raise ValueError("encoded configuration key exceeds uint16 length")
            payload.extend(CONFIG_KEY_SIZE.pack(len(encoded)))
            payload.extend(encoded)
            payload.extend(CONFIG_VALUE.pack(value))
        response = self._request(CONFIGURE, bytes(payload))
        if len(response) != CONFIG_HASH.size:
            raise ProtocolError("configure response has the wrong size")
        self.current_config_hash = CONFIG_HASH.unpack(response)[0]
        return self.current_config_hash

    def config_json(self) -> dict[str, object]:
        payload = self._request(CONFIG_JSON)
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolError("worker returned invalid config JSON") from error
        if not isinstance(value, dict):
            raise ProtocolError("worker config JSON is not an object")
        if value.get("config_hash") != self.current_config_hash:
            raise ProtocolError("worker config JSON hash disagrees with configure")
        return value

    def step(
        self,
        kind: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        wait_ticks: int = 1,
        *,
        suppress_fresh_edges: bool = False,
    ) -> Transition:
        self.send_step(
            kind,
            x,
            y,
            wait_ticks,
            suppress_fresh_edges=suppress_fresh_edges,
        )
        return self.receive_step()

    def send_step(
        self,
        kind: int = 0,
        x: float = 0.0,
        y: float = 0.0,
        wait_ticks: int = 1,
        *,
        suppress_fresh_edges: bool = False,
    ) -> None:
        """Start one step, allowing other worker processes to run concurrently."""

        if not 0 <= kind <= 3:
            raise ValueError("action kind must be in [0, 3]")
        if not 0 <= wait_ticks <= 0xFFFFFFFF:
            raise ValueError("wait_ticks must fit in uint32")
        flags = int(bool(suppress_fresh_edges))
        self._begin_request(STEP, STEP_REQUEST.pack(kind, x, y, wait_ticks, flags))

    def receive_step(self) -> Transition:
        """Finish a step started by :meth:`send_step`."""

        return decode_transition(self._finish_response(STEP))

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                if self._pending is not None:
                    self._finish_response(self._pending[0])
                if self._process.poll() is None:
                    self._request(CLOSE)
            except (OSError, RuntimeError):
                pass
            self._closed = True
            if self._process.stdin is not None:
                self._process.stdin.close()
            if self._process.stdout is not None:
                self._process.stdout.close()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            if self._process.stderr is not None:
                self._process.stderr.close()

    def __enter__(self) -> ExactWorkerClient:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _percentile(values: Sequence[int], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index] / 1_000.0


def _latency_summary(values: Sequence[int]) -> dict[str, float]:
    return {
        "mean_us": statistics.fmean(values) / 1_000.0,
        "p50_us": _percentile(values, 0.50),
        "p95_us": _percentile(values, 0.95),
        "p99_us": _percentile(values, 0.99),
    }


def _worker_identity(info: WorkerInfo) -> tuple[object, ...]:
    """Stable handshake fields; PID is expected to change between episodes."""

    values = asdict(info)
    values.pop("pid")
    return tuple(values.items())


def smoke(worker: Path, seed: int) -> dict[str, object]:
    with ExactWorkerClient(worker) as client:
        initial = client.reset(seed)
        observed = client.observe()
        transition = client.step()
        if observed != initial or transition.observation.tick != initial.tick + 1:
            raise RuntimeError("worker reset/observe/step smoke invariant failed")
        return {
            "worker": asdict(client.info),
            "reset": {
                "tick": initial.tick,
                "score": initial.score,
                "gauge": initial.gauge,
                "bodies": len(initial.bodies),
            },
            "step": {
                "tick": transition.observation.tick,
                "reward": transition.reward,
                "events": transition.event_count,
                "response_bytes": client.last_response_bytes,
            },
        }


def benchmark(
    worker: Path,
    workers: int,
    steps: int,
    warmup: int,
    observe_rounds: int,
    wait_ticks: int,
) -> dict[str, object]:
    clients = [ExactWorkerClient(worker) for _ in range(workers)]
    try:
        expected_identity = _worker_identity(clients[0].info)
        if any(_worker_identity(client.info) != expected_identity for client in clients):
            raise RuntimeError("exact worker identities disagree")

        def restart(index: int) -> None:
            candidate = ExactWorkerClient(worker)
            try:
                if _worker_identity(candidate.info) != expected_identity:
                    raise RuntimeError("replacement exact worker identity changed")
                candidate.reset(41 + index)
            except BaseException:
                candidate.close()
                raise
            previous = clients[index]
            clients[index] = candidate
            previous.close()

        for index, client in enumerate(clients):
            client.reset(41 + index)
        observe_latencies: list[int] = []
        for _ in range(observe_rounds):
            started = time.perf_counter_ns()
            clients[0].observe()
            observe_latencies.append(time.perf_counter_ns() - started)
        for _ in range(warmup):
            for index, client in enumerate(clients):
                result = client.step(wait_ticks=wait_ticks)
                if result.terminated or result.truncated:
                    restart(index)

        round_latencies: list[int] = []
        reset_count = 0
        started_all = time.perf_counter_ns()
        for _ in range(steps):
            started = time.perf_counter_ns()
            for client in clients:
                client.send_step(wait_ticks=wait_ticks)
            transitions = [client.receive_step() for client in clients]
            round_latencies.append(time.perf_counter_ns() - started)
            for index, transition in enumerate(transitions):
                if transition.terminated or transition.truncated:
                    restart(index)
                    reset_count += 1
        elapsed = (time.perf_counter_ns() - started_all) / 1_000_000_000.0
        env_steps = steps * workers
        return {
            "worker": asdict(clients[0].info),
            "workers": workers,
            "measured_vector_rounds": steps,
            "measured_env_steps": env_steps,
            "wait_ticks": wait_ticks,
            "resets_during_measurement": reset_count,
            "observation_response_bytes": clients[0].last_response_bytes,
            "observe_round_trip": _latency_summary(observe_latencies),
            "vector_round_trip": _latency_summary(round_latencies),
            "elapsed_seconds": elapsed,
            "env_steps_per_second": env_steps / elapsed,
            "physics_ticks_per_second": env_steps * wait_ticks / elapsed,
        }
    finally:
        for client in clients:
            client.close()


def replay(worker: Path, replay_path: Path) -> dict[str, object]:
    data = replay_path.read_bytes()
    if len(data) < 52 or (len(data) - 52) % 4:
        raise ValueError("expected a padded v2.03 replay")
    seed = struct.unpack_from("<I", data)[0]
    terminal_frame = len(data) // 4
    with ExactWorkerClient(worker) as client:
        observation = client.reset(seed)
        timeline_score = observation.score
        timeline_gauge = observation.gauge
        score_calls = 0
        confirmed = 0
        for frame, (word,) in enumerate(struct.iter_unpack("<I", data[52:])):
            buttons = word & 3
            kind = buttons if buttons else 0
            x = float((word >> 2) & 0x3FF)
            y = float((word >> 12) & 0x1FF)
            transition = client.step(
                kind, x, y, suppress_fresh_edges=frame < 2
            )
            observation = transition.observation
            for event in transition.events:
                if event.kind == 12:
                    score_calls += 1
                    timeline_score += event.value
                elif event.kind == 11:
                    timeline_gauge += event.value
                elif event.kind == 5:
                    confirmed += 1
            if timeline_score != observation.score:
                raise RuntimeError("score events do not reconstruct observation")
            if timeline_gauge != observation.gauge:
                raise RuntimeError("gauge events do not reconstruct observation")
            if transition.terminated and terminal_frame == len(data) // 4:
                terminal_frame = frame
        return {
            "replay": str(replay_path),
            "tick": observation.tick,
            "score": observation.score,
            "gauge": observation.gauge,
            "level": observation.level,
            "highest_chain": observation.highest_chain,
            "clears": observation.qualifying_clear_count,
            "score_calls": score_calls,
            "confirmed": confirmed,
            "terminal_frame": terminal_frame,
            "worker": asdict(client.info),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    smoke_parser = commands.add_parser("smoke")
    smoke_parser.add_argument("--seed", type=int, default=41)
    benchmark_parser = commands.add_parser("benchmark")
    benchmark_parser.add_argument("--workers", type=int, default=1)
    benchmark_parser.add_argument("--steps", type=int, default=2_000)
    benchmark_parser.add_argument("--warmup", type=int, default=100)
    benchmark_parser.add_argument("--observe-rounds", type=int, default=1_000)
    benchmark_parser.add_argument("--wait-ticks", type=int, default=1)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("path", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "smoke":
        result = smoke(args.worker, args.seed)
    elif args.command == "benchmark":
        for name in ("workers", "steps", "observe_rounds", "wait_ticks"):
            if getattr(args, name) <= 0:
                raise SystemExit(f"--{name.replace('_', '-')} must be positive")
        if args.warmup < 0:
            raise SystemExit("--warmup must be non-negative")
        result = benchmark(
            args.worker,
            args.workers,
            args.steps,
            args.warmup,
            args.observe_rounds,
            args.wait_ticks,
        )
    else:
        result = replay(args.worker, args.path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
