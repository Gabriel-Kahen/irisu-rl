"""Strict JSON-RPC provider for a separately audited live desktop broker.

The installed same-session bridge is intentionally not adapted here: its
atomic click cannot satisfy the explicit-edge contract.  A broker must expose
the complete protocol below and enforce fencing, release deadlines, and
claim-expiry neutralization independently of this process.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import re
import selectors
import stat
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from .harness import (
    CapturePacket,
    ClaimLease,
    ClaimToken,
    CursorSample,
    InputAcknowledgement,
    InputCapabilities,
    Rect,
    SessionSafety,
    TargetRuntimeDescriptor,
    WindowIdentity,
)

PROTOCOL = "irisu-live-broker-v1"
CLOCK_DOMAIN = "linux.clock_monotonic"
MAX_MESSAGE_BYTES = 32 << 20
MAX_FRAME_BYTES = 16 << 20
MAX_CLOCK_SKEW_NS = 50_000_000
BUTTONS = frozenset({"left", "right"})
SAFE_INPUT_BACKENDS = frozenset({"hyprland_native_targeted_edges_v1"})
SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class BrokerError(RuntimeError):
    """The live broker is unavailable, malformed, or rejected an operation."""


class BrokerTransport(Protocol):
    @property
    def executable_sha256(self) -> str: ...

    def call(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def close(self) -> None: ...


def _exact(value: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(k, str) for k in value):
        raise BrokerError(f"{label} must be an object")
    if set(value) != keys:
        raise BrokerError(
            f"{label} fields disagree: missing={sorted(keys - set(value))}, "
            f"extra={sorted(set(value) - keys)}"
        )
    return value


def _text(value: object, label: str, *, maximum: int = 256) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > maximum
        or any(character in value for character in ("\0", "\n", "\r"))
    ):
        raise BrokerError(f"{label} is invalid")
    return value


def _sha256(value: object, label: str) -> str:
    result = _text(value, label, maximum=64)
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise BrokerError(f"{label} is invalid")
    return result


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BrokerError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BrokerError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise BrokerError(f"{label} must be finite")
    return result


def _boolean(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise BrokerError(f"{label} must be a boolean")
    return value


def _identity(value: object) -> WindowIdentity:
    item = _exact(value, {"address", "capture_id"}, "window identity")
    return WindowIdentity(
        _text(item["address"], "window address"),
        _text(item["capture_id"], "capture identity"),
    )


def _identity_value(identity: WindowIdentity) -> dict[str, str]:
    return {"address": identity.address, "capture_id": identity.capture_id}


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(descriptor, 1 << 20, offset):
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BrokerError(f"duplicate broker response key: {key}")
        result[key] = value
    return result


class JsonLineBrokerTransport:
    """Bounded, sequence-checked RPC over a dedicated broker subprocess."""

    def __init__(
        self,
        executable: Path,
        arguments: Sequence[str] = (),
        *,
        expected_sha256: str,
        timeout_seconds: float = 2.0,
    ) -> None:
        if not executable.is_absolute():
            raise BrokerError("broker executable must be an absolute path")
        if isinstance(arguments, (str, bytes)) or any(
            not isinstance(arg, str) or "\0" in arg or "\n" in arg or "\r" in arg
            for arg in arguments
        ):
            raise BrokerError("broker arguments are invalid")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise BrokerError("broker timeout must be positive and finite")
        try:
            descriptor = os.open(executable, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        except OSError as exc:
            raise BrokerError(f"cannot inspect broker executable: {exc}") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise BrokerError(
                    "broker executable must be a regular non-symlink file"
                )
            if metadata.st_uid not in {0, os.getuid()}:
                raise BrokerError("broker executable has an untrusted owner")
            if metadata.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise BrokerError("broker executable must not be group/world writable")
            observed_sha256 = _sha256_descriptor(descriptor)
            if (
                len(expected_sha256) != 64
                or any(c not in "0123456789abcdef" for c in expected_sha256)
                or observed_sha256 != expected_sha256
            ):
                raise BrokerError("broker executable SHA-256 mismatch")
        except Exception:
            os.close(descriptor)
            raise
        self._timeout = float(timeout_seconds)
        self._sequence = 0
        self._buffer = bytearray()
        self._lock = threading.Lock()
        environment = {
            key: value
            for key, value in os.environ.items()
            if key
            in {
                "DISPLAY",
                "WAYLAND_DISPLAY",
                "XDG_RUNTIME_DIR",
                "HYPRLAND_INSTANCE_SIGNATURE",
                "LANG",
            }
        }
        try:
            executable_fd_path = f"/proc/self/fd/{descriptor}"
            self._process = subprocess.Popen(
                [executable_fd_path, *arguments],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd="/",
                env=environment,
                close_fds=True,
                pass_fds=(descriptor,),
            )
        except OSError as exc:
            raise BrokerError(f"cannot start broker: {exc}") from exc
        finally:
            os.close(descriptor)
        self._executable_sha256 = observed_sha256
        assert self._process.stdout is not None
        os.set_blocking(self._process.stdout.fileno(), False)

    @property
    def executable_sha256(self) -> str:
        return self._executable_sha256

    def _read_line(self) -> bytes:
        assert self._process.stdout is not None
        deadline = time.monotonic() + self._timeout
        selector = selectors.DefaultSelector()
        selector.register(self._process.stdout, selectors.EVENT_READ)
        try:
            while True:
                newline = self._buffer.find(b"\n")
                if newline >= 0:
                    line = bytes(self._buffer[: newline + 1])
                    del self._buffer[: newline + 1]
                    return line
                if len(self._buffer) > MAX_MESSAGE_BYTES:
                    raise BrokerError("broker response exceeds the size limit")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise BrokerError("broker response timed out")
                if not selector.select(remaining):
                    raise BrokerError("broker response timed out")
                chunk = os.read(self._process.stdout.fileno(), 65536)
                if not chunk:
                    raise BrokerError("broker closed its response stream")
                self._buffer.extend(chunk)
        finally:
            selector.close()

    def call(self, method: str, params: Mapping[str, Any]) -> Mapping[str, Any]:
        _text(method, "broker method", maximum=64)
        if not isinstance(params, Mapping):
            raise BrokerError("broker params must be an object")
        with self._lock:
            if self._process.poll() is not None:
                raise BrokerError("broker process is not running")
            self._sequence += 1
            request = {
                "protocol": PROTOCOL,
                "sequence": self._sequence,
                "method": method,
                "params": dict(params),
            }
            try:
                payload = (
                    json.dumps(
                        request,
                        allow_nan=False,
                        ensure_ascii=True,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("ascii")
                    + b"\n"
                )
            except (TypeError, ValueError) as exc:
                raise BrokerError("broker request is not canonical JSON") from exc
            if len(payload) > MAX_MESSAGE_BYTES:
                raise BrokerError("broker request exceeds the size limit")
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(payload)
                self._process.stdin.flush()
            except OSError as exc:
                raise BrokerError("cannot write broker request") from exc
            raw = self._read_line()
            if len(raw) > MAX_MESSAGE_BYTES:
                raise BrokerError("broker response exceeds the size limit")
            try:
                response = json.loads(raw, object_pairs_hook=_json_pairs)
            except (UnicodeDecodeError, ValueError) as exc:
                raise BrokerError("broker response is not valid JSON") from exc
            item = _exact(
                response,
                {"protocol", "sequence", "ok", "result", "error"},
                "broker response",
            )
            if item["protocol"] != PROTOCOL or item["sequence"] != self._sequence:
                raise BrokerError("broker response protocol/sequence mismatch")
            ok = _boolean(item["ok"], "broker response ok")
            if ok:
                if item["error"] is not None or not isinstance(item["result"], Mapping):
                    raise BrokerError("successful broker response is malformed")
                return item["result"]
            error = _exact(item["error"], {"code", "detail"}, "broker error")
            if item["result"] is not None:
                raise BrokerError("failed broker response contains a result")
            code = _text(error["code"], "broker error code", maximum=64)
            _text(error["detail"], "broker error detail")
            raise BrokerError(f"broker rejected {method}: {code}")

    def close(self) -> None:
        with self._lock:
            if self._process.poll() is None:
                try:
                    self._process.terminate()
                    self._process.wait(timeout=self._timeout)
                except (OSError, subprocess.TimeoutExpired):
                    self._process.kill()
                    self._process.wait(timeout=self._timeout)
            for stream in (self._process.stdin, self._process.stdout):
                if stream is not None:
                    stream.close()


class BrokerHarnessProvider:
    """Concrete :class:`HarnessProvider` backed by the strict broker protocol."""

    def __init__(
        self,
        transport: BrokerTransport,
        *,
        clock_ns=time.monotonic_ns,
    ) -> None:
        self._transport = transport
        self._clock_ns = clock_ns
        self._tokens: dict[ClaimToken, str] = {}
        self._bindings: dict[ClaimToken, ClaimLease] = {}
        self._release_deadlines: dict[tuple[ClaimToken, str], int] = {}
        self._operation_id = 0
        before = clock_ns()
        handshake = _exact(
            transport.call("handshake", {}),
            {
                "clock_domain",
                "monotonic_ns",
                "broker_instance_id",
                "broker_implementation_sha256",
                "input_backend",
                "capabilities",
            },
            "broker handshake",
        )
        after = clock_ns()
        if handshake["clock_domain"] != CLOCK_DOMAIN:
            raise BrokerError("broker uses an incompatible monotonic clock")
        broker_now = _integer(handshake["monotonic_ns"], "broker monotonic_ns")
        if (
            broker_now < before - MAX_CLOCK_SKEW_NS
            or broker_now > after + MAX_CLOCK_SKEW_NS
        ):
            raise BrokerError("broker monotonic clock is outside the handshake bound")
        self.broker_instance_id = _text(
            handshake["broker_instance_id"], "broker instance ID", maximum=128
        )
        if SAFE_IDENTIFIER.fullmatch(self.broker_instance_id) is None:
            raise BrokerError("broker instance ID is unsafe")
        implementation_sha256 = _text(
            handshake["broker_implementation_sha256"],
            "broker implementation SHA-256",
            maximum=64,
        )
        if (
            len(implementation_sha256) != 64
            or any(c not in "0123456789abcdef" for c in implementation_sha256)
            or implementation_sha256 != transport.executable_sha256
        ):
            raise BrokerError("broker handshake implementation SHA-256 mismatch")
        self.broker_implementation_sha256 = implementation_sha256
        self.input_backend = _text(
            handshake["input_backend"], "input backend", maximum=128
        )
        if SAFE_IDENTIFIER.fullmatch(self.input_backend) is None:
            raise BrokerError("input backend name is unsafe")
        self._capabilities = self._parse_capabilities(handshake["capabilities"])
        if (
            self._capabilities.supports_safe_shots
            and self.input_backend not in SAFE_INPUT_BACKENDS
        ):
            raise BrokerError("safe-shot capabilities require an audited input backend")

    @staticmethod
    def _parse_safety(value: object) -> SessionSafety:
        item = _exact(
            value,
            {
                "exact_background_capture",
                "exact_window_claims",
                "targeted_input_safe",
                "detail",
            },
            "session safety",
        )
        return SessionSafety(
            _boolean(item["exact_background_capture"], "exact background capture"),
            _boolean(item["exact_window_claims"], "exact window claims"),
            _boolean(item["targeted_input_safe"], "targeted input safety"),
            _text(item["detail"], "session safety detail") if item["detail"] else "",
        )

    @staticmethod
    def _parse_capabilities(value: object) -> InputCapabilities:
        item = _exact(
            value,
            {
                "explicit_button_down",
                "explicit_button_up",
                "release_all_buttons",
                "atomic_click_only",
                "automatic_release_deadline",
                "neutralizes_on_claim_end_or_expiry",
            },
            "input capabilities",
        )
        return InputCapabilities(
            **{key: _boolean(value, key) for key, value in item.items()}
        )

    def _token(self, token: ClaimToken) -> str:
        try:
            return self._tokens[token]
        except KeyError as exc:
            raise BrokerError("unknown or expired claim token") from exc

    @staticmethod
    def _lease(
        value: object, token: ClaimToken | None = None
    ) -> tuple[ClaimLease, str]:
        item = _exact(
            value,
            {
                "identity",
                "fencing_token",
                "expires_ns",
                "generation",
                "broker_instance",
                "target",
            },
            "claim lease",
        )
        raw_token = _text(item["fencing_token"], "fencing token", maximum=1024)
        opaque = token or ClaimToken(raw_token)
        identity = _identity(item["identity"])
        target = _exact(
            item["target"],
            {
                "identity",
                "process_id",
                "process_start_ticks",
                "launch_nonce_sha256",
                "executable_sha256",
                "runtime_sha256",
                "wine_executable_sha256",
                "wine_prefix_sha256",
            },
            "target runtime descriptor",
        )
        return (
            ClaimLease(
                identity,
                opaque,
                _integer(item["expires_ns"], "claim expiry", minimum=1),
                _integer(item["generation"], "claim generation", minimum=1),
                _text(item["broker_instance"], "claim broker instance"),
                TargetRuntimeDescriptor(
                    _identity(target["identity"]),
                    _integer(target["process_id"], "target process ID", minimum=1),
                    _integer(
                        target["process_start_ticks"],
                        "target process start ticks",
                        minimum=1,
                    ),
                    _text(
                        target["launch_nonce_sha256"],
                        "target launch nonce SHA-256",
                        maximum=64,
                    ),
                    _text(
                        target["executable_sha256"],
                        "target executable SHA-256",
                        maximum=64,
                    ),
                    _text(
                        target["runtime_sha256"],
                        "target runtime SHA-256",
                        maximum=64,
                    ),
                    _text(
                        target["wine_executable_sha256"],
                        "target Wine SHA-256",
                        maximum=64,
                    ),
                    _text(
                        target["wine_prefix_sha256"],
                        "target Wine-prefix SHA-256",
                        maximum=64,
                    ),
                ),
            ),
            raw_token,
        )

    def _ack(
        self,
        value: object,
        *,
        operation_id: int,
        identity: WindowIdentity,
        generation: int,
        button: str,
        state: str,
        deadline: int | None,
    ) -> InputAcknowledgement:
        item = _exact(
            value,
            {
                "injected_ns",
                "acknowledged_ns",
                "acknowledged",
                "detail",
                "operation_id",
                "identity",
                "claim_generation",
                "broker_instance",
                "button",
                "button_state",
                "accepted_release_deadline_ns",
                "clock_domain",
            },
            "input acknowledgment",
        )
        ack = InputAcknowledgement(
            _integer(item["injected_ns"], "injected_ns"),
            _integer(item["acknowledged_ns"], "acknowledged_ns"),
            _boolean(item["acknowledged"], "acknowledged"),
            _text(item["detail"], "ack detail") if item["detail"] else "",
            _integer(item["operation_id"], "operation ID", minimum=1),
            _identity(item["identity"]),
            _integer(item["claim_generation"], "claim generation", minimum=1),
            _text(item["broker_instance"], "ack broker instance"),
            _text(item["button"], "ack button", maximum=16),
            _text(item["button_state"], "ack button state", maximum=16),
            (
                None
                if item["accepted_release_deadline_ns"] is None
                else _integer(
                    item["accepted_release_deadline_ns"],
                    "accepted release deadline",
                    minimum=1,
                )
            ),
            _text(item["clock_domain"], "ack clock domain"),
        )
        if (
            ack.operation_id != operation_id
            or ack.identity != identity
            or ack.claim_generation != generation
            or ack.broker_instance != self.broker_instance_id
            or ack.button != button
            or ack.button_state != state
            or ack.accepted_release_deadline_ns != deadline
            or ack.clock_domain != CLOCK_DOMAIN
        ):
            raise BrokerError("input acknowledgment binding mismatch")
        return ack

    def current_session_safety(self) -> SessionSafety:
        result = _exact(
            self._transport.call("session_safety", {}), {"safety"}, "safety result"
        )
        return self._parse_safety(result["safety"])

    def discover_exact_target(
        self,
        *,
        launch_nonce: str,
        executable_sha256: str,
        runtime_sha256: str,
        wine_prefix_sha256: str,
    ) -> TargetRuntimeDescriptor:
        """Resolve exactly one broker-observed process launched with a secret nonce."""

        nonce = _text(launch_nonce, "launch nonce", maximum=128)
        if len(nonce) < 32:
            raise BrokerError("launch nonce is too short")
        expected_prefix = _sha256(
            wine_prefix_sha256,
            "Wine-prefix SHA-256",
        )
        result = _exact(
            self._transport.call(
                "discover_target",
                {
                    "launch_nonce": nonce,
                    "executable_sha256": executable_sha256,
                    "runtime_sha256": runtime_sha256,
                    "wine_prefix_sha256": expected_prefix,
                },
            ),
            {"match_count", "target", "broker_instance"},
            "target discovery",
        )
        if result["broker_instance"] != self.broker_instance_id:
            raise BrokerError("target discovery changed broker instance")
        if _integer(result["match_count"], "target match count") != 1:
            raise BrokerError("target discovery did not find exactly one process")
        target = _exact(
            result["target"],
            {
                "identity",
                "process_id",
                "process_start_ticks",
                "launch_nonce_sha256",
                "executable_sha256",
                "runtime_sha256",
                "wine_executable_sha256",
                "wine_prefix_sha256",
            },
            "discovered target",
        )
        descriptor = TargetRuntimeDescriptor(
            _identity(target["identity"]),
            _integer(target["process_id"], "target process ID", minimum=1),
            _integer(
                target["process_start_ticks"], "target process start ticks", minimum=1
            ),
            _text(
                target["launch_nonce_sha256"],
                "target launch nonce SHA-256",
                maximum=64,
            ),
            _text(target["executable_sha256"], "target executable SHA-256", maximum=64),
            _text(target["runtime_sha256"], "target runtime SHA-256", maximum=64),
            _text(target["wine_executable_sha256"], "target Wine SHA-256", maximum=64),
            _text(
                target["wine_prefix_sha256"],
                "target Wine-prefix SHA-256",
                maximum=64,
            ),
        )
        if (
            descriptor.executable_sha256 != executable_sha256
            or descriptor.runtime_sha256 != runtime_sha256
            or descriptor.wine_prefix_sha256 != expected_prefix
            or descriptor.launch_nonce_sha256
            != hashlib.sha256(nonce.encode("utf-8")).hexdigest()
        ):
            raise BrokerError("discovered target runtime identity mismatch")
        return descriptor

    def input_capabilities(self) -> InputCapabilities:
        return self._capabilities

    def claim_exact_window(
        self, identity: WindowIdentity, lease_seconds: int
    ) -> ClaimLease:
        result = self._transport.call(
            "claim",
            {"identity": _identity_value(identity), "lease_seconds": lease_seconds},
        )
        lease, raw_token = self._lease(result)
        if (
            lease.broker_instance != self.broker_instance_id
            or lease.identity != identity
            or lease.target is None
            or lease.target.identity != identity
        ):
            try:
                self._transport.call(
                    "release_claim",
                    {
                        "identity": _identity_value(lease.identity),
                        "fencing_token": raw_token,
                    },
                )
            except Exception:
                self._transport.close()
            raise BrokerError("claim target or broker binding mismatch")
        self._tokens[lease.token] = raw_token
        self._bindings[lease.token] = lease
        return lease

    def renew_exact_window_claim(
        self, identity: WindowIdentity, token: ClaimToken, lease_seconds: int
    ) -> ClaimLease:
        try:
            previous = self._bindings[token]
        except KeyError as exc:
            raise BrokerError("claim binding is unavailable") from exc
        result = self._transport.call(
            "renew",
            {
                "identity": _identity_value(identity),
                "fencing_token": self._token(token),
                "lease_seconds": lease_seconds,
            },
        )
        lease, raw_token = self._lease(result, token)
        if (
            raw_token != self._token(token)
            or identity != previous.identity
            or lease.identity != previous.identity
            or lease.token != previous.token
            or lease.generation != previous.generation
            or lease.broker_instance != previous.broker_instance
            or lease.broker_instance != self.broker_instance_id
            or lease.target != previous.target
            or lease.expires_ns <= previous.expires_ns
        ):
            raise BrokerError("claim renewal changed an immutable binding")
        self._bindings[token] = lease
        return lease

    def release_exact_window_claim(
        self, identity: WindowIdentity, token: ClaimToken
    ) -> None:
        result = self._transport.call(
            "release_claim",
            {
                "identity": _identity_value(identity),
                "fencing_token": self._token(token),
            },
        )
        _exact(result, {"released"}, "release result")
        if _boolean(result["released"], "released") is not True:
            raise BrokerError("broker did not release the claim")
        self._tokens.pop(token, None)
        self._bindings.pop(token, None)
        for key in tuple(self._release_deadlines):
            if key[0] == token:
                del self._release_deadlines[key]

    def capture_exact_window(
        self, identity: WindowIdentity, token: ClaimToken
    ) -> CapturePacket:
        result = _exact(
            self._transport.call(
                "capture",
                {
                    "identity": _identity_value(identity),
                    "fencing_token": self._token(token),
                },
            ),
            {
                "pixels_base64",
                "identity",
                "window_bounds",
                "pixel_width",
                "pixel_height",
                "request_ns",
                "start_ns",
                "completion_ns",
                "presentation_ns",
                "source_sequence",
                "color_format",
                "canonical_pixel_sha256",
            },
            "capture result",
        )
        encoded = _text(
            result["pixels_base64"],
            "capture pixels",
            maximum=(MAX_FRAME_BYTES * 4 // 3 + 8),
        )
        try:
            pixels = base64.b64decode(encoded, validate=True)
        except ValueError as exc:
            raise BrokerError("capture pixels are not canonical base64") from exc
        if not pixels or len(pixels) > MAX_FRAME_BYTES:
            raise BrokerError("capture payload size is invalid")
        pixel_width = _integer(result["pixel_width"], "pixel width", minimum=1)
        pixel_height = _integer(result["pixel_height"], "pixel height", minimum=1)
        color_format = _text(result["color_format"], "color format", maximum=32)
        if color_format != "bgra8":
            raise BrokerError("capture must use tightly packed bgra8 pixels")
        expected_size = pixel_width * pixel_height * 4
        if expected_size > MAX_FRAME_BYTES or len(pixels) != expected_size:
            raise BrokerError("capture byte length disagrees with its dimensions")
        expected_pixel_sha256 = _sha256(
            result["canonical_pixel_sha256"], "canonical pixel SHA-256"
        )
        if hashlib.sha256(pixels).hexdigest() != expected_pixel_sha256:
            raise BrokerError("capture pixel SHA-256 does not match its payload")
        bounds = _exact(
            result["window_bounds"], {"x", "y", "width", "height"}, "bounds"
        )
        presentation = result["presentation_ns"]
        sequence = result["source_sequence"]
        return CapturePacket(
            pixels,
            _identity(result["identity"]),
            Rect(
                *(
                    _number(bounds[key], f"bounds.{key}")
                    for key in ("x", "y", "width", "height")
                )
            ),
            pixel_width,
            pixel_height,
            _integer(result["request_ns"], "request_ns"),
            _integer(result["start_ns"], "start_ns"),
            _integer(result["completion_ns"], "completion_ns"),
            None if presentation is None else _integer(presentation, "presentation_ns"),
            None if sequence is None else _integer(sequence, "source_sequence"),
            color_format,
            expected_pixel_sha256,
        )

    def current_cursor(
        self, identity: WindowIdentity, token: ClaimToken
    ) -> CursorSample:
        result = _exact(
            self._transport.call(
                "cursor",
                {
                    "identity": _identity_value(identity),
                    "fencing_token": self._token(token),
                },
            ),
            {"x", "y", "observed_ns"},
            "cursor result",
        )
        return CursorSample(
            _number(result["x"], "cursor x"),
            _number(result["y"], "cursor y"),
            _integer(result["observed_ns"], "cursor observed_ns"),
        )

    def _edge(
        self,
        method: str,
        identity: WindowIdentity,
        token: ClaimToken,
        button: str,
        x: float,
        y: float,
        release_deadline_ns: int | None = None,
        latest_injection_ns: int | None = None,
    ) -> InputAcknowledgement:
        if button not in BUTTONS:
            raise BrokerError("only left/right buttons are legal")
        if method == "button_down" and (
            release_deadline_ns is None or latest_injection_ns is None
        ):
            raise BrokerError(
                "button-down requires release and latest-injection deadlines"
            )
        self._operation_id += 1
        operation_id = self._operation_id
        generation = self._claim_generation(token)
        params: dict[str, Any] = {
            "identity": _identity_value(identity),
            "fencing_token": self._token(token),
            "button": button,
            "x": _number(x, "window x"),
            "y": _number(y, "window y"),
            "operation_id": operation_id,
        }
        if release_deadline_ns is not None:
            params["release_deadline_ns"] = _integer(
                release_deadline_ns, "release deadline", minimum=1
            )
        if latest_injection_ns is not None:
            params["latest_injection_ns"] = _integer(
                latest_injection_ns, "latest injection time", minimum=1
            )
        state = "down" if method == "button_down" else "up"
        deadline = release_deadline_ns
        if method == "button_up":
            try:
                deadline = self._release_deadlines[(token, button)]
            except KeyError as exc:
                raise BrokerError(
                    "button-up has no matching acknowledged down"
                ) from exc
        ack = self._ack(
            self._transport.call(method, params),
            operation_id=operation_id,
            identity=identity,
            generation=generation,
            button=button,
            state=state,
            deadline=deadline,
        )
        if method == "button_down":
            assert release_deadline_ns is not None
            assert latest_injection_ns is not None
            if ack.injected_ns > latest_injection_ns:
                raise BrokerError(
                    "broker injected button-down after its freshness deadline"
                )
            self._release_deadlines[(token, button)] = release_deadline_ns
        else:
            del self._release_deadlines[(token, button)]
        return ack

    def _claim_generation(self, token: ClaimToken) -> int:
        self._token(token)
        try:
            lease = self._bindings[token]
        except KeyError as exc:
            raise BrokerError("claim binding is unavailable") from exc
        if lease.broker_instance != self.broker_instance_id:
            raise BrokerError("claim binding names a different broker instance")
        assert lease.generation is not None
        return lease.generation

    def targeted_button_down(
        self,
        identity: WindowIdentity,
        token: ClaimToken,
        button: str,
        x: float,
        y: float,
        release_deadline_ns: int,
        latest_injection_ns: int,
    ) -> InputAcknowledgement:
        return self._edge(
            "button_down",
            identity,
            token,
            button,
            x,
            y,
            release_deadline_ns,
            latest_injection_ns,
        )

    def targeted_button_up(
        self,
        identity: WindowIdentity,
        token: ClaimToken,
        button: str,
        x: float,
        y: float,
    ) -> InputAcknowledgement:
        return self._edge("button_up", identity, token, button, x, y)

    def release_all_buttons(
        self, identity: WindowIdentity, token: ClaimToken
    ) -> InputAcknowledgement:
        self._operation_id += 1
        operation_id = self._operation_id
        generation = self._claim_generation(token)
        ack = self._ack(
            self._transport.call(
                "release_all",
                {
                    "identity": _identity_value(identity),
                    "fencing_token": self._token(token),
                    "operation_id": operation_id,
                },
            ),
            operation_id=operation_id,
            identity=identity,
            generation=generation,
            button="all",
            state="neutral",
            deadline=None,
        )
        for key in tuple(self._release_deadlines):
            if key[0] == token:
                del self._release_deadlines[key]
        return ack

    def close(self) -> None:
        self._tokens.clear()
        self._bindings.clear()
        self._transport.close()
