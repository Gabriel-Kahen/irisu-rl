from __future__ import annotations

import base64
import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_rl.original_game.harness import WindowIdentity
from irisu_rl.original_game.live_provider import (
    CLOCK_DOMAIN,
    BrokerError,
    BrokerHarnessProvider,
    JsonLineBrokerTransport,
)

IDENTITY = WindowIdentity("0xabc", "capture-1")


class Transport:
    executable_sha256 = "a" * 64

    def __init__(self, *, safe: bool = True) -> None:
        self.safe = safe
        self.closed = False
        self.calls = []
        self.extra_handshake = False
        self.release_deadline = None
        self.wrong_claim_identity = False

    def call(self, method, params):
        self.calls.append((method, params))
        if method == "handshake":
            result = {
                "clock_domain": CLOCK_DOMAIN,
                "monotonic_ns": 1000,
                "broker_instance_id": "broker-test-1",
                "broker_implementation_sha256": self.executable_sha256,
                "input_backend": (
                    "hyprland_native_targeted_edges_v1"
                    if self.safe
                    else "same_session_atomic_targeted_click_v1"
                ),
                "capabilities": {
                    "explicit_button_down": self.safe,
                    "explicit_button_up": self.safe,
                    "release_all_buttons": self.safe,
                    "atomic_click_only": not self.safe,
                    "automatic_release_deadline": self.safe,
                    "neutralizes_on_claim_end_or_expiry": self.safe,
                },
            }
            if self.extra_handshake:
                result["unexpected"] = True
            return result
        if method == "session_safety":
            return {
                "safety": {
                    "exact_background_capture": True,
                    "exact_window_claims": True,
                    "targeted_input_safe": self.safe,
                    "detail": "",
                }
            }
        if method == "discover_target":
            return {
                "match_count": 1,
                "broker_instance": "broker-test-1",
                "target": {
                    "identity": {"address": "0xabc", "capture_id": "capture-1"},
                    "process_id": 123,
                    "process_start_ticks": 456,
                    "launch_nonce_sha256": hashlib.sha256(
                        params["launch_nonce"].encode()
                    ).hexdigest(),
                    "executable_sha256": params["executable_sha256"],
                    "runtime_sha256": params["runtime_sha256"],
                    "wine_executable_sha256": "e" * 64,
                },
            }
        if method in {"claim", "renew"}:
            identity = (
                {"address": "0xwrong", "capture_id": "capture-wrong"}
                if self.wrong_claim_identity
                else {"address": "0xabc", "capture_id": "capture-1"}
            )
            return {
                "identity": identity,
                "fencing_token": "never-log-this",
                "expires_ns": 10_000,
                "generation": 1,
                "broker_instance": "broker-test-1",
                "target": {
                    "identity": identity,
                    "process_id": 123,
                    "process_start_ticks": 456,
                    "launch_nonce_sha256": "d" * 64,
                    "executable_sha256": "b" * 64,
                    "runtime_sha256": "c" * 64,
                    "wine_executable_sha256": "e" * 64,
                },
            }
        if method == "capture":
            return {
                "pixels_base64": base64.b64encode(b"png").decode(),
                "identity": {"address": "0xabc", "capture_id": "capture-1"},
                "window_bounds": {"x": 0, "y": 0, "width": 644, "height": 484},
                "pixel_width": 644,
                "pixel_height": 484,
                "request_ns": 1100,
                "start_ns": 1110,
                "completion_ns": 1120,
                "presentation_ns": None,
                "source_sequence": 7,
                "color_format": "png",
                "canonical_pixel_sha256": hashlib.sha256(b"raw pixels").hexdigest(),
            }
        if method == "cursor":
            return {"x": 10, "y": 20, "observed_ns": 1130}
        if method in {"button_down", "button_up", "release_all"}:
            if method == "button_down":
                self.release_deadline = params["release_deadline_ns"]
            button = params.get("button", "all")
            state = {
                "button_down": "down",
                "button_up": "up",
                "release_all": "neutral",
            }[method]
            return {
                "injected_ns": 1140,
                "acknowledged_ns": 1150,
                "acknowledged": True,
                "detail": "",
                "operation_id": params["operation_id"],
                "identity": {"address": "0xabc", "capture_id": "capture-1"},
                "claim_generation": 1,
                "broker_instance": "broker-test-1",
                "button": button,
                "button_state": state,
                "accepted_release_deadline_ns": (
                    self.release_deadline if method != "release_all" else None
                ),
                "clock_domain": CLOCK_DOMAIN,
            }
        if method == "release_claim":
            return {"released": True}
        raise AssertionError(method)

    def close(self):
        self.closed = True


class R4BLiveProviderTests(unittest.TestCase):
    def provider(self, transport: Transport):
        return BrokerHarnessProvider(transport, clock_ns=lambda: 1000)

    def test_atomic_only_handshake_remains_input_ineligible(self) -> None:
        transport = Transport(safe=False)
        provider = self.provider(transport)
        self.assertFalse(provider.input_capabilities().supports_safe_shots)
        self.assertTrue(provider.input_capabilities().atomic_click_only)

    def test_claim_capture_cursor_and_release_keep_token_opaque(self) -> None:
        transport = Transport()
        provider = self.provider(transport)
        lease = provider.claim_exact_window(IDENTITY, 30)
        self.assertEqual(repr(lease.token), "ClaimToken(<redacted>)")
        self.assertNotIn("never-log-this", repr(provider))
        packet = provider.capture_exact_window(IDENTITY, lease.token)
        self.assertEqual(packet.pixels, b"png")
        self.assertEqual(packet.source_sequence, 7)
        self.assertEqual(packet.sha256, hashlib.sha256(b"raw pixels").hexdigest())
        cursor = provider.current_cursor(IDENTITY, lease.token)
        self.assertEqual((cursor.x, cursor.y), (10, 20))
        provider.release_exact_window_claim(IDENTITY, lease.token)
        with self.assertRaisesRegex(BrokerError, "unknown"):
            provider.capture_exact_window(IDENTITY, lease.token)

    def test_target_discovery_requires_one_exact_runtime_match(self) -> None:
        provider = self.provider(Transport())
        nonce = "n" * 64
        target = provider.discover_exact_target(
            launch_nonce=nonce,
            executable_sha256="b" * 64,
            runtime_sha256="c" * 64,
        )
        self.assertEqual(target.identity, IDENTITY)
        self.assertEqual(
            target.launch_nonce_sha256, hashlib.sha256(nonce.encode()).hexdigest()
        )

    def test_mismatched_claim_is_released_before_rejection(self) -> None:
        transport = Transport()
        transport.wrong_claim_identity = True
        provider = self.provider(transport)
        with self.assertRaisesRegex(BrokerError, "binding mismatch"):
            provider.claim_exact_window(IDENTITY, 30)
        self.assertEqual(transport.calls[-1][0], "release_claim")

    def test_malformed_handshake_and_clock_are_rejected(self) -> None:
        transport = Transport()
        transport.extra_handshake = True
        with self.assertRaisesRegex(BrokerError, "extra"):
            self.provider(transport)
        transport = Transport()
        with self.assertRaisesRegex(BrokerError, "clock"):
            BrokerHarnessProvider(transport, clock_ns=lambda: 1_000_000_000)

    def test_only_legal_buttons_reach_broker(self) -> None:
        transport = Transport()
        provider = self.provider(transport)
        lease = provider.claim_exact_window(IDENTITY, 30)
        with self.assertRaisesRegex(BrokerError, "left/right"):
            provider.targeted_button_down(IDENTITY, lease.token, "middle", 1, 2, 9000)
        self.assertNotIn("button_down", [method for method, _ in transport.calls])

    def test_edges_are_bound_to_operation_claim_deadline_and_broker(self) -> None:
        transport = Transport()
        provider = self.provider(transport)
        lease = provider.claim_exact_window(IDENTITY, 30)
        down = provider.targeted_button_down(
            IDENTITY, lease.token, "left", 10, 20, 9000
        )
        up = provider.targeted_button_up(IDENTITY, lease.token, "left", 10, 20)
        neutral = provider.release_all_buttons(IDENTITY, lease.token)
        self.assertEqual(
            (down.operation_id, up.operation_id, neutral.operation_id), (1, 2, 3)
        )
        self.assertEqual(down.accepted_release_deadline_ns, 9000)
        self.assertEqual(up.accepted_release_deadline_ns, 9000)
        self.assertIsNone(neutral.accepted_release_deadline_ns)

    def test_subprocess_transport_pins_file_and_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            broker = Path(temporary) / "broker"
            broker.write_text(
                f"#!{sys.executable}\n"
                "import json,sys\n"
                "for line in sys.stdin:\n"
                " request=json.loads(line)\n"
                " response={'protocol':request['protocol'],"
                "'sequence':request['sequence'],'ok':True,"
                "'result':{'echo':request['params']},'error':None}\n"
                " print(json.dumps(response,separators=(',',':')),flush=True)\n",
                encoding="utf-8",
            )
            broker.chmod(0o700)
            digest = hashlib.sha256(broker.read_bytes()).hexdigest()
            transport = JsonLineBrokerTransport(
                broker, expected_sha256=digest, timeout_seconds=1
            )
            try:
                self.assertEqual(
                    transport.call("echo", {"value": 7}), {"echo": {"value": 7}}
                )
                self.assertEqual(transport.executable_sha256, digest)
            finally:
                transport.close()
            os.chmod(broker, 0o722)
            with self.assertRaisesRegex(BrokerError, "writable"):
                JsonLineBrokerTransport(broker, expected_sha256=digest)


if __name__ == "__main__":
    unittest.main()
