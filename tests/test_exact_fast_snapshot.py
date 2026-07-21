from __future__ import annotations

import os
import secrets
import socket
import struct
import subprocess
import time
import unittest
from pathlib import Path
from typing import BinaryIO


ROOT = Path(__file__).resolve().parents[1]
WORKER = os.environ.get("IRISU_EXACT_WORKER")
REPLAY = (
    ROOT
    / "reference/replays/raw/internet/irisu_00041449_20100725_182435_7.rpy"
)

MAGIC = 0x43505249
VERSION = 1
HELLO = 1
RESET = 2
STEP = 3
CLOSE = 5
FAST_CHECKPOINT = 10
FAST_RELEASE = 11
FAST_BRANCH = 12
HEADER = struct.Struct("<IHHII")
STATUS = struct.Struct("<i")
STEP_REQUEST = struct.Struct("<IddII")
CHECKPOINT_RESPONSE = struct.Struct("<16sI")
BRANCH_RESPONSE = struct.Struct("<IH16s")


class ResponseError(RuntimeError):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"worker status {status}: {detail}")
        self.status = status
        self.detail = detail


def read_exact(stream: BinaryIO, size: int) -> bytes:
    output = bytearray()
    while len(output) != size:
        chunk = stream.read(size - len(output))
        if not chunk:
            raise EOFError("worker response ended early")
        output.extend(chunk)
    return bytes(output)


def process_state(pid: int) -> str | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text()
    except FileNotFoundError:
        return None
    return value[value.rfind(")") + 2 :].split(maxsplit=1)[0]


def wait_gone(pid: int, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process_state(pid) is None:
            return
        time.sleep(0.01)
    raise AssertionError(f"process {pid} was not reaped (state={process_state(pid)})")


class RawClient:
    def __init__(
        self,
        reader: BinaryIO,
        writer: BinaryIO,
        *,
        process: subprocess.Popen[bytes] | None = None,
        connection: socket.socket | None = None,
    ) -> None:
        self.reader = reader
        self.writer = writer
        self.process = process
        self.connection = connection
        self.request_id = 0
        self.pending: tuple[int, int] | None = None
        self.closed = False

    @classmethod
    def launch(cls, path: str) -> RawClient:
        process = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )
        assert process.stdin is not None and process.stdout is not None
        client = cls(process.stdout, process.stdin, process=process)
        client.request(HELLO)
        return client

    @classmethod
    def connect(cls, address: bytes, secret: bytes) -> RawClient:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(3.0)
        connection.connect(address)
        connection.sendall(secret)
        connection.settimeout(None)
        reader = connection.makefile("rb", buffering=0)
        writer = connection.makefile("wb", buffering=0)
        client = cls(reader, writer, connection=connection)
        client.request(HELLO)
        return client

    def begin(self, opcode: int, payload: bytes = b"") -> None:
        if self.pending is not None:
            raise RuntimeError("request already pending")
        self.request_id = (self.request_id + 1) & 0xFFFFFFFF or 1
        self.writer.write(
            HEADER.pack(MAGIC, VERSION, opcode, self.request_id, len(payload))
            + payload
        )
        self.writer.flush()
        self.pending = (opcode, self.request_id)

    def finish(self) -> bytes:
        if self.pending is None:
            raise RuntimeError("no request pending")
        expected_opcode, expected_id = self.pending
        self.pending = None
        magic, version, opcode, request_id, size = HEADER.unpack(
            read_exact(self.reader, HEADER.size)
        )
        if (magic, version, opcode, request_id) != (
            MAGIC,
            VERSION,
            expected_opcode,
            expected_id,
        ):
            raise AssertionError("mismatched worker response")
        response = read_exact(self.reader, size)
        status = STATUS.unpack_from(response)[0]
        content = response[STATUS.size :]
        if status:
            raise ResponseError(status, content.decode("utf-8", errors="replace"))
        return content

    def request(self, opcode: int, payload: bytes = b"") -> bytes:
        self.begin(opcode, payload)
        return self.finish()

    def checkpoint(self) -> tuple[bytes, int]:
        response = self.request(FAST_CHECKPOINT)
        if len(response) != CHECKPOINT_RESPONSE.size:
            raise AssertionError("malformed checkpoint response")
        return CHECKPOINT_RESPONSE.unpack(response)

    def branch(self, token: bytes) -> tuple[RawClient, int]:
        response = self.request(FAST_BRANCH, token)
        if len(response) < BRANCH_RESPONSE.size:
            raise AssertionError("malformed branch response")
        pid, size, secret = BRANCH_RESPONSE.unpack_from(response)
        address = response[BRANCH_RESPONSE.size :]
        if len(address) != size or not address.startswith(b"\0"):
            raise AssertionError("malformed branch address")
        return RawClient.connect(address, secret), pid

    def close(self) -> None:
        if self.closed:
            return
        try:
            self.request(CLOSE)
        except (EOFError, OSError, ResponseError):
            pass
        self.closed = True
        self.reader.close()
        self.writer.close()
        if self.connection is not None:
            self.connection.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            if self.process.stderr is not None:
                self.process.stderr.close()


def replay_steps() -> tuple[int, list[bytes]]:
    data = REPLAY.read_bytes()
    seed = struct.unpack_from("<I", data)[0]
    actions: list[bytes] = []
    for frame, (word,) in enumerate(struct.iter_unpack("<I", data[52:])):
        buttons = word & 3
        actions.append(
            STEP_REQUEST.pack(
                buttons,
                float((word >> 2) & 0x3FF),
                float((word >> 12) & 0x1FF),
                1,
                int(frame < 2),
            )
        )
    return seed, actions


@unittest.skipUnless(
    WORKER
    and REPLAY.is_file()
    and os.name == "posix"
    and os.uname().sysname == "Linux",
    "requires the Linux exact worker and authoritative replay asset",
)
class ExactFastSnapshotTests(unittest.TestCase):
    def test_deep_checkpoint_repeated_branches_and_release_lifecycle(self) -> None:
        seed, actions = replay_steps()
        prefix = 30_000
        future = 1_000
        source = RawClient.launch(WORKER)
        branches: list[RawClient] = []
        try:
            source.request(RESET, struct.pack("<Q", seed))
            for payload in actions[:prefix]:
                source.request(STEP, payload)

            token, keeper_pid = source.checkpoint()
            self.assertEqual(len(token), 16)
            self.assertNotEqual(token, bytes(16))
            first, first_pid = source.branch(token)
            second, second_pid = source.branch(token)
            branches.extend((first, second))

            with self.assertRaisesRegex(ResponseError, "active branches"):
                source.request(FAST_RELEASE, token)

            for payload in actions[prefix : prefix + future]:
                for client in (source, first, second):
                    client.begin(STEP, payload)
                responses = [client.finish() for client in (source, first, second)]
                self.assertEqual(responses[0], responses[1])
                self.assertEqual(responses[0], responses[2])

            first.close()
            second.close()
            branches.clear()
            wait_gone(first_pid)
            wait_gone(second_pid)
            source.request(FAST_RELEASE, token)
            wait_gone(keeper_pid)

            for opcode in (FAST_BRANCH, FAST_RELEASE):
                with self.assertRaisesRegex(ResponseError, "unknown checkpoint"):
                    source.request(opcode, token)
            with self.assertRaisesRegex(ResponseError, "unknown checkpoint"):
                source.request(FAST_BRANCH, secrets.token_bytes(16))
        finally:
            for branch in branches:
                branch.close()
            source.close()

    def test_source_death_recursively_reaps_keeper_and_branch(self) -> None:
        source = RawClient.launch(WORKER)
        branch: RawClient | None = None
        try:
            source.request(RESET, struct.pack("<Q", 41))
            token, keeper_pid = source.checkpoint()
            branch, branch_pid = source.branch(token)
            assert source.process is not None
            source.process.kill()
            source.process.wait(timeout=3)
            wait_gone(keeper_pid)
            wait_gone(branch_pid)
            with self.assertRaises((EOFError, OSError, BrokenPipeError)):
                branch.request(4)
        finally:
            if branch is not None:
                branch.close()
            source.close()


if __name__ == "__main__":
    unittest.main()
