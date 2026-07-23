from __future__ import annotations

import sys
import threading
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_rl.original_game.harness import (
    MONOTONIC_CLOCK_DOMAIN,
    ActionError,
    CapturePacket,
    ClaimLease,
    ClaimToken,
    CleanupError,
    EffectStatus,
    ExecutionStatus,
    FrameError,
    FrameFlag,
    GeometryAssessment,
    GeometryError,
    HarnessError,
    HarnessLimits,
    InputAcknowledgement,
    InputCapabilities,
    OriginalGameHarness,
    Rect,
    SafetyError,
    ScreenClassification,
    ScreenState,
    SessionSafety,
    ShotKind,
    TargetRuntimeDescriptor,
    UnsupportedInputError,
    WindowIdentity,
    WindowIdentityError,
)

IDENTITY = WindowIdentity("0xabc", "capture-7")
OTHER = WindowIdentity("0xdef", "capture-8")
EXECUTABLE_SHA256 = "a" * 64
RUNTIME_SHA256 = "b" * 64
NONCE_SHA256 = "c" * 64
WINE_SHA256 = "d" * 64
BROKER_INSTANCE = "fixture-broker-1"
CLAIM_GENERATION = 7


class Clock:
    def __init__(self, value: int = 1_000_000_000) -> None:
        self.value = value

    def __call__(self) -> int:
        return self.value

    def advance(self, duration: int) -> None:
        self.value += duration


class Geometry:
    def __init__(self) -> None:
        self.crop = Rect(2, 2, 640, 480)
        self.residual = 0.25
        self.confidence = 0.99
        self.age = 0
        self.drifted = False

    def assess(self, capture: CapturePacket, now_ns: int) -> GeometryAssessment:
        return GeometryAssessment(
            self.crop,
            self.age,
            self.residual,
            self.confidence,
            self.drifted,
            "fixture-v1",
        )

    def client_to_window(
        self, x: float, y: float, assessment: GeometryAssessment
    ) -> tuple[float, float]:
        return x + assessment.crop.x, y + assessment.crop.y


class Screen:
    def __init__(
        self,
        state: ScreenState = ScreenState.GAMEPLAY,
        confidence: float = 0.99,
    ) -> None:
        self.state = state
        self.confidence = confidence

    def classify(
        self, capture: CapturePacket, geometry: GeometryAssessment
    ) -> ScreenClassification:
        return ScreenClassification(self.state, self.confidence)


class FakeProvider:
    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.safety = SessionSafety(True, True, True)
        self.capabilities = InputCapabilities(
            True,
            True,
            True,
            automatic_release_deadline=True,
            neutralizes_on_claim_end_or_expiry=True,
        )
        self.claim_identity = IDENTITY
        self.capture_identity = IDENTITY
        self.source_sequence = 0
        self.frame_payload = b"frame-0"
        self.events: list[tuple[object, ...]] = []
        self.fail: set[str] = set()
        self.claimed = False
        self.token = ClaimToken("top-secret-fencing-token")
        self.cursor = (10.0, 10.0)
        self.window_bounds = Rect(0, 0, 644, 484)
        self.held_buttons: set[str] = set()
        self.release_deadlines: dict[str, int] = {}
        self.renew_headroom_ns: int | None = None
        self.claim_generation = CLAIM_GENERATION
        self.broker_instance = BROKER_INSTANCE
        self.executable_sha256 = EXECUTABLE_SHA256
        self.runtime_sha256 = RUNTIME_SHA256
        self.operation_id = 0

    def _event(
        self, name: str, identity: WindowIdentity, token: ClaimToken | None = None
    ) -> None:
        if identity != IDENTITY:
            raise AssertionError("cross-window misroute")
        if token is not None and token != self.token:
            raise AssertionError("wrong fencing token")
        self.events.append((name, identity, token))
        if name in self.fail:
            raise RuntimeError(f"forced {name} failure")

    def _descriptor(
        self, identity: WindowIdentity | None = None
    ) -> TargetRuntimeDescriptor:
        return TargetRuntimeDescriptor(
            identity or self.claim_identity,
            1234,
            5678,
            NONCE_SHA256,
            self.executable_sha256,
            self.runtime_sha256,
            WINE_SHA256,
        )

    def _lease(
        self,
        expires_ns: int,
        *,
        identity: WindowIdentity | None = None,
    ) -> ClaimLease:
        claimed_identity = identity or self.claim_identity
        return ClaimLease(
            claimed_identity,
            self.token,
            expires_ns,
            self.claim_generation,
            self.broker_instance,
            self._descriptor(claimed_identity),
        )

    def _ack(
        self,
        *,
        identity: WindowIdentity = IDENTITY,
        button: str,
        button_state: str,
        release_deadline_ns: int | None,
    ) -> InputAcknowledgement:
        injected = self.clock()
        self.clock.advance(10)
        self.operation_id += 1
        return InputAcknowledgement(
            injected,
            self.clock(),
            operation_id=self.operation_id,
            identity=identity,
            claim_generation=self.claim_generation,
            broker_instance=self.broker_instance,
            button=button,
            button_state=button_state,
            accepted_release_deadline_ns=release_deadline_ns,
            clock_domain=MONOTONIC_CLOCK_DOMAIN,
        )

    def current_session_safety(self) -> SessionSafety:
        self.events.append(("safety",))
        if "safety" in self.fail:
            raise RuntimeError("forced safety failure")
        return self.safety

    def input_capabilities(self) -> InputCapabilities:
        self.events.append(("capabilities",))
        return self.capabilities

    def claim_exact_window(
        self, identity: WindowIdentity, lease_seconds: int
    ) -> ClaimLease:
        self._event("claim", identity)
        self.claimed = True
        return self._lease(
            self.clock() + lease_seconds * 1_000_000_000,
            identity=self.claim_identity,
        )

    def renew_exact_window_claim(
        self, identity: WindowIdentity, token: ClaimToken, lease_seconds: int
    ) -> ClaimLease:
        self._event("renew", identity, token)
        return self._lease(
            self.clock()
            + (
                lease_seconds * 1_000_000_000
                if self.renew_headroom_ns is None
                else self.renew_headroom_ns
            ),
            identity=identity,
        )

    def release_exact_window_claim(
        self, identity: WindowIdentity, token: ClaimToken
    ) -> None:
        if identity not in {IDENTITY, self.claim_identity}:
            raise AssertionError("released an unrelated claim")
        if token != self.token:
            raise AssertionError("wrong fencing token")
        self.events.append(("release_claim", identity, token))
        self.held_buttons.clear()
        self.release_deadlines.clear()
        self.claimed = False

    def capture_exact_window(
        self, identity: WindowIdentity, token: ClaimToken
    ) -> CapturePacket:
        self._event("capture", identity, token)
        request = self.clock()
        self.clock.advance(10)
        start = self.clock()
        self.clock.advance(10)
        completion = self.clock()
        return CapturePacket(
            self.frame_payload,
            self.capture_identity,
            self.window_bounds,
            644,
            484,
            request,
            start,
            completion,
            source_sequence=self.source_sequence,
        )

    def current_cursor(self, identity: WindowIdentity, token: ClaimToken):
        self._event("cursor", identity, token)
        from irisu_rl.original_game.harness import CursorSample

        return CursorSample(*self.cursor, self.clock())

    def targeted_button_down(
        self,
        identity: WindowIdentity,
        token: ClaimToken,
        button: str,
        x: float,
        y: float,
        release_deadline_ns: int,
    ) -> InputAcknowledgement:
        self._event(f"down:{button}:{x}:{y}", identity, token)
        if "down" in self.fail:
            raise RuntimeError("forced down failure")
        if type(release_deadline_ns) is not int or release_deadline_ns <= self.clock():
            raise RuntimeError("invalid broker release deadline")
        self.held_buttons.add(button)
        self.release_deadlines[button] = release_deadline_ns
        return self._ack(
            button=button,
            button_state="down",
            release_deadline_ns=release_deadline_ns,
        )

    def targeted_button_up(
        self,
        identity: WindowIdentity,
        token: ClaimToken,
        button: str,
        x: float,
        y: float,
    ) -> InputAcknowledgement:
        self.enforce_release_deadlines()
        self._event(f"up:{button}:{x}:{y}", identity, token)
        if "up" in self.fail:
            raise RuntimeError("forced up failure")
        release_deadline_ns = self.release_deadlines.get(button)
        self.held_buttons.discard(button)
        self.release_deadlines.pop(button, None)
        return self._ack(
            button=button,
            button_state="up",
            release_deadline_ns=release_deadline_ns,
        )

    def release_all_buttons(
        self, identity: WindowIdentity, token: ClaimToken
    ) -> InputAcknowledgement:
        self._event("release_all", identity, token)
        self.held_buttons.clear()
        self.release_deadlines.clear()
        return self._ack(
            button="all",
            button_state="neutral",
            release_deadline_ns=None,
        )

    def enforce_release_deadlines(self) -> None:
        for button, deadline in tuple(self.release_deadlines.items()):
            if self.clock() >= deadline:
                self.held_buttons.discard(button)
                self.release_deadlines.pop(button, None)
                self.events.append((f"auto_release:{button}", IDENTITY, self.token))


def limits(**changes: object) -> HarnessLimits:
    values: dict[str, object] = {
        "press_duration_ns": 20,
        "min_click_interval_ns": 100,
        "cursor_mode": "abstract_teleport",
        "stale_after_ns": 1_000,
        "frame_buffer_capacity": 3,
    }
    values.update(changes)
    return HarnessLimits(**values)


class HarnessTests(unittest.TestCase):
    def make(
        self,
        *,
        provider: FakeProvider | None = None,
        geometry: Geometry | None = None,
        screen: Screen | None = None,
        config: HarnessLimits | None = None,
    ) -> tuple[OriginalGameHarness, FakeProvider, Geometry, Clock]:
        clock = provider.clock if provider is not None else Clock()
        provider = provider or FakeProvider(clock)
        geometry = geometry or Geometry()
        screen = screen or Screen()
        harness = OriginalGameHarness(
            provider,
            IDENTITY,
            geometry,
            screen,
            limits=config or limits(),
            clock_ns=clock,
            sleep_ns=clock.advance,
        )
        return harness, provider, geometry, clock

    def test_claim_token_repr_is_redacted(self) -> None:
        token = ClaimToken("do-not-print-me")
        self.assertNotIn("do-not-print-me", repr(token))
        self.assertNotIn("do-not-print-me", str(token))

    def test_claim_binds_an_immutable_target_and_runtime_descriptor(self) -> None:
        harness, provider, _, _ = self.make()
        harness.open()
        descriptor = harness.target_descriptor
        self.assertEqual(descriptor, provider._descriptor(IDENTITY))
        assert descriptor is not None
        with self.assertRaises(FrozenInstanceError):
            descriptor.process_id = 42  # type: ignore[misc]

        provider.runtime_sha256 = "c" * 64
        with self.assertRaisesRegex(WindowIdentityError, "target/runtime identity"):
            harness.renew()
        self.assertFalse(provider.claimed)

    def test_safety_identity_and_capability_fields_are_strictly_typed(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            WindowIdentity(1, "capture")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            SessionSafety(1, True, True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            InputCapabilities(True, "yes", True)  # type: ignore[arg-type]

    def test_open_capture_fire_and_cleanup_are_exactly_claim_bound(self) -> None:
        harness, provider, _, clock = self.make()
        harness.open()
        result = harness.fire(ShotKind.STRONG, 100, 200)
        self.assertEqual(result.status, ExecutionStatus.EXECUTED)
        self.assertEqual((result.window_x, result.window_y), (102, 202))
        self.assertEqual(result.proposed.kind, ShotKind.STRONG)
        clock.advance(100)
        harness.close()

        names = [str(event[0]).split(":")[0] for event in provider.events]
        self.assertIn("capture", names)
        self.assertEqual(names.count("down"), 1)
        self.assertEqual(names.count("up"), 1)
        self.assertEqual(names[-2:], ["release_all", "release_claim"])
        self.assertTrue(harness.watchdog.buttons_neutral)
        self.assertFalse(harness.watchdog.claim_active)
        self.assertEqual(len(harness.proposed_actions), 1)
        self.assertEqual(len(harness.executed_actions), 1)

    def test_atomic_click_only_provider_fails_closed_without_input(self) -> None:
        harness, provider, _, _ = self.make()
        provider.capabilities = InputCapabilities(False, False, False, True)
        harness.open()
        with self.assertRaisesRegex(UnsupportedInputError, "explicit down/up"):
            harness.fire("weak", 50, 50)
        self.assertFalse(
            any(str(event[0]).startswith(("down", "up")) for event in provider.events)
        )
        self.assertEqual(harness.executed_actions[-1].status, ExecutionStatus.REJECTED)
        self.assertFalse(provider.claimed)

    def test_provider_without_broker_release_guarantees_is_input_ineligible(
        self,
    ) -> None:
        harness, provider, _, _ = self.make()
        provider.capabilities = InputCapabilities(True, True, True)
        harness.open()
        with self.assertRaisesRegex(UnsupportedInputError, "broker release deadlines"):
            harness.fire("weak", 50, 50)
        self.assertFalse(
            any(str(event[0]).startswith("down:") for event in provider.events)
        )
        self.assertFalse(provider.claimed)

    def test_frame_classification_duplicate_drop_overflow(self) -> None:
        harness, provider, _, clock = self.make(
            config=limits(max_duplicate_run=5, max_source_drop_gap=5)
        )
        harness.open()
        provider.source_sequence = 1
        clock.advance(50)
        duplicate = harness.capture()
        self.assertIn(FrameFlag.DUPLICATE, duplicate.flags)
        provider.source_sequence = 4
        provider.frame_payload = b"frame-4"
        clock.advance(50)
        dropped = harness.capture()
        self.assertIn(FrameFlag.DROPPED, dropped.flags)
        self.assertEqual(dropped.dropped_count, 2)
        provider.source_sequence = 5
        provider.frame_payload = b"frame-5"
        clock.advance(50)
        overflow = harness.capture()
        self.assertIn(FrameFlag.BUFFER_OVERFLOW, overflow.flags)
        self.assertEqual(harness.watchdog.buffer_overflows, 1)
        harness.close()

    def test_stale_out_of_order_and_duplicate_limit_abort(self) -> None:
        for mode in ("stale", "out_of_order", "duplicate"):
            with self.subTest(mode=mode):
                harness, provider, _, clock = self.make(
                    config=limits(max_duplicate_run=0)
                )
                harness.open()
                if mode == "stale":
                    provider.source_sequence = 1
                    provider.fail.add("capture")
                    original = provider.capture_exact_window

                    def stale(identity, token):
                        provider.fail.remove("capture")
                        packet = original(identity, token)
                        provider.fail.add("capture")
                        clock.advance(2_000)
                        return packet

                    provider.capture_exact_window = stale  # type: ignore[method-assign]
                    expected = "stale"
                elif mode == "out_of_order":
                    provider.source_sequence = 0
                    provider.frame_payload = b"different"
                    expected = "out of order"
                else:
                    provider.source_sequence = 1
                    expected = "duplicate"
                clock.advance(50)
                with self.assertRaisesRegex(FrameError, expected):
                    harness.capture()
                self.assertFalse(provider.claimed)
                self.assertIn(
                    [event[0] for event in provider.events][-2:],
                    (
                        ["release_all", "release_claim"],
                        ["release_all", "release_claim"],
                    ),
                )

    def test_geometry_quality_and_crop_drift_abort(self) -> None:
        cases = (
            ("residual", 3.0, GeometryError),
            ("confidence", 0.5, GeometryError),
            ("age", 2_000_000_000, GeometryError),
        )
        for field, value, error in cases:
            with self.subTest(field=field):
                geometry = Geometry()
                setattr(geometry, field, value)
                harness, provider, _, _ = self.make(geometry=geometry)
                with self.assertRaises(error):
                    harness.open()
                self.assertFalse(provider.claimed)

        harness, provider, geometry, clock = self.make()
        harness.open()
        geometry.crop = Rect(4, 2, 640, 480)
        provider.source_sequence = 1
        provider.frame_payload = b"moved"
        clock.advance(50)
        with self.assertRaisesRegex(GeometryError, "drifted"):
            harness.capture()
        self.assertFalse(provider.claimed)

    def test_nonzero_absolute_origin_uses_window_local_bounds_and_drift_aborts(
        self,
    ) -> None:
        clock = Clock()
        provider = FakeProvider(clock)
        provider.window_bounds = Rect(-1_920, 100, 644, 484)
        harness, _, _, _ = self.make(provider=provider)
        harness.open()
        result = harness.fire("weak", 10, 10)
        self.assertEqual((result.window_x, result.window_y), (12.0, 12.0))
        provider.source_sequence = 1
        provider.frame_payload = b"window-moved"
        provider.window_bounds = Rect(-1_900, 100, 644, 484)
        clock.advance(100)
        with self.assertRaisesRegex(GeometryError, "drifted"):
            harness.capture()
        self.assertFalse(provider.claimed)

    def test_crop_must_stay_inside_captured_pixels(self) -> None:
        geometry = Geometry()
        geometry.crop = Rect(10, 10, 640, 480)
        harness, provider, _, _ = self.make(geometry=geometry)
        with self.assertRaisesRegex(GeometryError, "outside captured pixels"):
            harness.open()
        self.assertFalse(provider.claimed)

    def test_non_gameplay_or_low_confidence_screen_is_unusable_for_input(self) -> None:
        cases = (
            Screen(ScreenState.MENU, 0.99),
            Screen(ScreenState.GAMEPLAY, 0.5),
            Screen(ScreenState.UNKNOWN, 0.99),
        )
        for screen in cases:
            with self.subTest(state=screen.state, confidence=screen.confidence):
                harness, provider, _, _ = self.make(screen=screen)
                frame = harness.open().buffer.latest
                assert frame is not None
                self.assertIn(FrameFlag.NON_GAMEPLAY, frame.flags)
                self.assertFalse(frame.usable)
                with self.assertRaisesRegex(ActionError, "not confidently"):
                    harness.fire("weak", 20, 20)
                self.assertFalse(
                    any(
                        str(event[0]).startswith(("down:", "up:"))
                        for event in provider.events
                    )
                )
                self.assertFalse(provider.claimed)

    def test_effect_status_and_first_visible_are_bound_to_captured_frames(self) -> None:
        harness, provider, _, clock = self.make(
            config=limits(min_click_interval_ns=1, max_source_drop_gap=5)
        )
        harness.open()

        confirmed_action = harness.fire("weak", 20, 20)
        clock.advance(1)
        provider.source_sequence = 1
        provider.frame_payload = b"projectile-visible"
        visible_frame = harness.capture()
        confirmed = harness.record_action_effect(
            confirmed_action.proposed.sequence,
            EffectStatus.CONFIRMED,
            frame_sequence=visible_frame.sequence,
        )
        self.assertEqual(confirmed.effect_status, EffectStatus.CONFIRMED)
        self.assertEqual(
            confirmed.first_visible_ns, visible_frame.capture.completion_ns
        )

        clock.advance(1)
        missed_action = harness.fire("strong", 30, 30)
        provider.source_sequence = 2
        provider.frame_payload = b"no-projectile"
        missed_frame = harness.capture()
        missed = harness.record_action_effect(
            missed_action.proposed.sequence,
            EffectStatus.MISSED,
            frame_sequence=missed_frame.sequence,
            detail="no projectile in declared horizon",
        )
        self.assertEqual(missed.effect_status, EffectStatus.MISSED)
        self.assertIsNone(missed.first_visible_ns)

        clock.advance(1)
        ambiguous_action = harness.fire("weak", 40, 40)
        provider.source_sequence = 3
        provider.frame_payload = b"ambiguous-birth"
        ambiguous_frame = harness.capture()
        ambiguous = harness.record_action_effect(
            ambiguous_action.proposed.sequence,
            EffectStatus.AMBIGUOUS,
            frame_sequence=ambiguous_frame.sequence,
        )
        self.assertEqual(ambiguous.effect_status, EffectStatus.AMBIGUOUS)
        self.assertEqual(ambiguous.effect_frame_sequence, ambiguous_frame.sequence)
        self.assertEqual(
            harness.executed_actions[-1].effect_status, EffectStatus.AMBIGUOUS
        )
        harness.close()

    def test_effect_confirmation_rejects_unrecorded_timestamp(self) -> None:
        harness, provider, _, clock = self.make()
        harness.open()
        action = harness.fire("weak", 20, 20)
        clock.advance(1)
        provider.source_sequence = 1
        provider.frame_payload = b"projectile-visible"
        frame = harness.capture()
        with self.assertRaisesRegex(ActionError, "does not match"):
            harness.record_action_effect(
                action.proposed.sequence,
                "confirmed",
                frame_sequence=frame.sequence,
                first_visible_ns=frame.capture.completion_ns + 1,
            )
        self.assertFalse(provider.claimed)

    def test_terminal_effect_requires_a_later_causal_frame(self) -> None:
        harness, provider, _, _ = self.make()
        harness.open()
        action = harness.fire("weak", 20, 20)
        self.assertEqual(harness.watchdog.pending_actions, 1)
        with self.assertRaisesRegex(ActionError, "later captured frame"):
            harness.record_action_effect(action.proposed.sequence, "missed")
        self.assertFalse(provider.claimed)

    def test_identity_change_on_claim_or_capture_aborts(self) -> None:
        harness, provider, _, _ = self.make()
        provider.claim_identity = OTHER
        with self.assertRaisesRegex(WindowIdentityError, "different window"):
            harness.open()
        self.assertIn("release_claim", [event[0] for event in provider.events])

        harness, provider, _, _ = self.make()
        provider.capture_identity = OTHER
        with self.assertRaisesRegex(WindowIdentityError, "identity changed"):
            harness.open()
        self.assertFalse(provider.claimed)

    def test_session_safety_is_checked_at_open_capture_and_fire(self) -> None:
        harness, provider, _, _ = self.make()
        provider.safety = SessionSafety(True, True, False, "held buttons")
        with self.assertRaisesRegex(SafetyError, "held buttons"):
            harness.open()
        self.assertNotIn("claim", [event[0] for event in provider.events])

        harness, provider, _, _ = self.make()
        harness.open()
        provider.safety = SessionSafety(True, True, False, "pointer lock")
        with self.assertRaisesRegex(SafetyError, "pointer lock"):
            harness.fire("weak", 20, 20)
        self.assertFalse(provider.claimed)

    def test_provider_timestamps_must_be_inside_the_local_call(self) -> None:
        harness, provider, _, _ = self.make()
        original_capture = provider.capture_exact_window

        def early_capture(identity, token):
            packet = original_capture(identity, token)
            return replace(packet, request_ns=packet.request_ns - 1)

        provider.capture_exact_window = early_capture  # type: ignore[method-assign]
        with self.assertRaisesRegex(HarnessError, "provider-call interval"):
            harness.open()
        self.assertFalse(provider.claimed)

        harness, provider, _, clock = self.make()

        def early_cursor(identity, token):
            provider._event("cursor", identity, token)
            from irisu_rl.original_game.harness import CursorSample

            return CursorSample(*provider.cursor, clock() - 1)

        provider.current_cursor = early_cursor  # type: ignore[method-assign]
        with self.assertRaisesRegex(HarnessError, "provider-call interval"):
            harness.open()
        self.assertFalse(provider.claimed)

    def test_invalid_bounds_rate_and_cursor_contract_fail_closed(self) -> None:
        cases = (
            (limits(), "weak", float("nan"), 0, "finite"),
            (limits(), "weak", 640, 0, "out of bounds"),
            (
                limits(cursor_mode="unsupported"),
                "weak",
                1,
                1,
                "cursor fairness",
            ),
            (
                limits(press_duration_ns=None),
                "weak",
                1,
                1,
                "press duration",
            ),
        )
        for config, kind, x, y, message in cases:
            with self.subTest(message=message):
                harness, provider, _, _ = self.make(config=config)
                harness.open()
                with self.assertRaisesRegex(ActionError, message):
                    harness.fire(kind, x, y)
                self.assertFalse(provider.claimed)

        harness, provider, _, _ = self.make()
        harness.open()
        first = harness.fire("weak", 1, 1)
        provider.source_sequence = 1
        provider.frame_payload = b"no-effect"
        frame = harness.capture()
        harness.record_action_effect(
            first.proposed.sequence, "missed", frame_sequence=frame.sequence
        )
        with self.assertRaisesRegex(ActionError, "click-rate"):
            harness.fire("weak", 1, 1)
        with self.assertRaisesRegex(HarnessError, "not active"):
            harness.fire("weak", 1, 1)

    def test_unresolved_effect_enforces_maximum_pending_depth_one(self) -> None:
        harness, provider, _, _ = self.make(config=limits(min_click_interval_ns=1))
        harness.open()
        harness.fire("weak", 1, 1)
        self.assertEqual(harness.watchdog.pending_actions, 1)
        with self.assertRaisesRegex(ActionError, "pending-action depth"):
            harness.fire("strong", 2, 2)
        self.assertEqual(
            sum(str(event[0]).startswith("down:") for event in provider.events),
            1,
        )
        self.assertFalse(provider.claimed)

    def test_bounded_cursor_speed_is_enforced(self) -> None:
        config = limits(
            cursor_mode="bounded_speed",
            max_cursor_speed_per_second=10.0,
            stale_after_ns=2_000_000_000,
        )
        harness, provider, _, clock = self.make(config=config)
        harness.open()
        clock.advance(1_000_000_000)
        with self.assertRaisesRegex(ActionError, "cursor-speed"):
            harness.fire("weak", 100, 100)
        self.assertFalse(provider.claimed)

    def test_failure_matrix_always_attempts_release_all_then_claim_release(
        self,
    ) -> None:
        for point in ("capture", "cursor", "down", "up"):
            with self.subTest(point=point):
                harness, provider, _, clock = self.make()
                if point in {"capture", "cursor"}:
                    provider.fail.add(point)
                    with self.assertRaisesRegex(RuntimeError, "forced"):
                        harness.open()
                else:
                    harness.open()
                    provider.fail.add(point)
                    clock.advance(100)
                    with self.assertRaisesRegex(RuntimeError, "forced"):
                        harness.fire("weak", 10, 10)
                tail = [event[0] for event in provider.events][-2:]
                self.assertEqual(tail, ["release_all", "release_claim"])
                self.assertFalse(provider.claimed)
                if point in {"down", "up"}:
                    failed = harness.executed_actions[-1]
                    self.assertEqual(failed.status, ExecutionStatus.FAILED)
                    self.assertEqual((failed.window_x, failed.window_y), (12.0, 12.0))
                    self.assertEqual(failed.down is not None, point == "up")
                    self.assertIsNone(failed.up)

    def test_release_all_failure_still_releases_claim_and_is_reported(self) -> None:
        harness, provider, _, _ = self.make()
        harness.open()
        provider.fail.add("release_all")
        with self.assertRaises(CleanupError):
            harness.close()
        self.assertEqual(provider.events[-1][0], "release_claim")
        self.assertTrue(harness.watchdog.buttons_neutral)
        self.assertFalse(provider.held_buttons)

    def test_future_edge_and_cleanup_acknowledgments_fail_closed(self) -> None:
        for edge in ("down", "up", "release_all"):
            with self.subTest(edge=edge):
                harness, provider, _, clock = self.make()
                harness.open()

                def future_ack() -> InputAcknowledgement:
                    return InputAcknowledgement(clock() + 10, clock() + 20)

                if edge == "down":

                    def down(
                        identity,
                        token,
                        button,
                        x,
                        y,
                        release_deadline_ns,
                    ):
                        provider._event(f"down:{button}:{x}:{y}", identity, token)
                        provider.held_buttons.add(button)
                        return future_ack()

                    provider.targeted_button_down = down  # type: ignore[method-assign]
                    with self.assertRaisesRegex(ActionError, "future"):
                        harness.fire("weak", 10, 10)
                elif edge == "up":

                    def up(identity, token, button, x, y):
                        provider._event(f"up:{button}:{x}:{y}", identity, token)
                        return future_ack()

                    provider.targeted_button_up = up  # type: ignore[method-assign]
                    with self.assertRaisesRegex(ActionError, "future"):
                        harness.fire("weak", 10, 10)
                else:

                    def release_all(identity, token):
                        provider._event("release_all", identity, token)
                        return future_ack()

                    provider.release_all_buttons = release_all  # type: ignore[method-assign]
                    with self.assertRaises(CleanupError):
                        harness.close()

                self.assertFalse(provider.claimed)
                self.assertFalse(provider.held_buttons)
                self.assertTrue(harness.watchdog.buttons_neutral)

    def test_input_acknowledgment_is_bound_to_claim_and_operation(self) -> None:
        mutations = (
            {"identity": OTHER},
            {"claim_generation": CLAIM_GENERATION + 1},
            {"broker_instance": "other-broker"},
            {"button": "right"},
            {"button_state": "up"},
            {"accepted_release_deadline_ns": 1},
            {"clock_domain": "other-clock"},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                harness, provider, _, _ = self.make()
                harness.open()
                original_down = provider.targeted_button_down

                def mismatched_down(
                    *args,
                    _original_down=original_down,
                    _mutation=mutation,
                    **kwargs,
                ):
                    return replace(_original_down(*args, **kwargs), **_mutation)

                provider.targeted_button_down = mismatched_down  # type: ignore[method-assign]
                with self.assertRaisesRegex(ActionError, "not bound"):
                    harness.fire("weak", 10, 10)
                self.assertFalse(provider.claimed)
                self.assertFalse(provider.held_buttons)

        harness, provider, _, _ = self.make()
        harness.open()
        harness._last_input_operation_id = 5  # type: ignore[attr-defined]
        provider.operation_id = 4
        with self.assertRaisesRegex(ActionError, "not bound"):
            harness.fire("weak", 10, 10)
        self.assertFalse(provider.claimed)
        self.assertFalse(provider.held_buttons)

    def test_concurrent_fire_is_serialized_at_pending_depth_one(self) -> None:
        harness, provider, _, _ = self.make(config=limits(min_click_interval_ns=1))
        harness.open()
        entered_down = threading.Event()
        continue_down = threading.Event()
        original_down = provider.targeted_button_down

        def blocked_down(*args, **kwargs):
            entered_down.set()
            self.assertTrue(continue_down.wait(timeout=2))
            return original_down(*args, **kwargs)

        provider.targeted_button_down = blocked_down  # type: ignore[method-assign]
        results: list[object] = []
        second_attempting = threading.Event()

        def call_fire(kind: str, attempting: threading.Event | None = None) -> None:
            if attempting is not None:
                attempting.set()
            try:
                results.append(harness.fire(kind, 10, 10))
            except Exception as exc:
                results.append(exc)

        first = threading.Thread(target=call_fire, args=("weak",))
        second = threading.Thread(target=call_fire, args=("strong", second_attempting))
        first.start()
        self.assertTrue(entered_down.wait(timeout=2))
        second.start()
        self.assertTrue(second_attempting.wait(timeout=2))
        self.assertTrue(second.is_alive())
        continue_down.set()
        first.join(timeout=2)
        second.join(timeout=2)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(
            sum(not isinstance(result, Exception) for result in results),
            1,
        )
        self.assertEqual(
            sum(
                isinstance(result, ActionError)
                and "pending-action depth" in str(result)
                for result in results
            ),
            1,
        )
        self.assertEqual(
            sum(str(event[0]).startswith("down:") for event in provider.events),
            1,
        )
        self.assertLessEqual(harness.watchdog.pending_actions, 1)
        self.assertFalse(provider.held_buttons)

    def test_broker_deadline_neutralizes_oversleep_and_down_stall(self) -> None:
        harness, provider, _, clock = self.make(
            config=limits(lease_cleanup_margin_ns=100)
        )
        harness.open()

        def oversleep(duration: int) -> None:
            clock.advance(duration + 101)
            provider.enforce_release_deadlines()

        harness._sleep_ns = oversleep  # type: ignore[attr-defined]
        with self.assertRaisesRegex(ActionError, "deadline elapsed"):
            harness.fire("weak", 10, 10)
        self.assertFalse(provider.held_buttons)
        self.assertTrue(
            any(str(event[0]).startswith("auto_release:") for event in provider.events)
        )

        harness, provider, _, clock = self.make(
            config=limits(lease_cleanup_margin_ns=100)
        )
        harness.open()

        def stalled_down(
            identity,
            token,
            button,
            x,
            y,
            release_deadline_ns,
        ):
            provider._event(f"down:{button}:{x}:{y}", identity, token)
            injected = clock()
            provider.held_buttons.add(button)
            provider.release_deadlines[button] = release_deadline_ns
            clock.advance(release_deadline_ns - clock() + 1)
            provider.enforce_release_deadlines()
            provider.operation_id += 1
            return InputAcknowledgement(
                injected,
                injected + 1,
                operation_id=provider.operation_id,
                identity=IDENTITY,
                claim_generation=provider.claim_generation,
                broker_instance=provider.broker_instance,
                button=button,
                button_state="down",
                accepted_release_deadline_ns=release_deadline_ns,
                clock_domain=MONOTONIC_CLOCK_DOMAIN,
            )

        provider.targeted_button_down = stalled_down  # type: ignore[method-assign]
        with self.assertRaisesRegex(ActionError, "returned after"):
            harness.fire("strong", 10, 10)
        self.assertFalse(provider.held_buttons)
        self.assertTrue(harness.watchdog.buttons_neutral)

    def test_button_up_must_return_before_the_accepted_deadline(self) -> None:
        harness, provider, _, clock = self.make(
            config=limits(lease_cleanup_margin_ns=100)
        )
        harness.open()

        def late_up(identity, token, button, x, y):
            provider._event(f"up:{button}:{x}:{y}", identity, token)
            deadline = provider.release_deadlines[button]
            injected = clock()
            clock.advance(deadline - clock())
            provider.held_buttons.discard(button)
            provider.release_deadlines.pop(button)
            provider.operation_id += 1
            return InputAcknowledgement(
                injected,
                clock(),
                operation_id=provider.operation_id,
                identity=identity,
                claim_generation=provider.claim_generation,
                broker_instance=provider.broker_instance,
                button=button,
                button_state="up",
                accepted_release_deadline_ns=deadline,
                clock_domain=MONOTONIC_CLOCK_DOMAIN,
            )

        provider.targeted_button_up = late_up  # type: ignore[method-assign]
        with self.assertRaisesRegex(ActionError, "button-up returned after"):
            harness.fire("weak", 10, 10)
        self.assertFalse(provider.claimed)
        self.assertFalse(provider.held_buttons)

    def test_renewal_must_preserve_identity_and_token(self) -> None:
        harness, provider, _, _ = self.make()
        harness.open()
        harness.renew()
        self.assertTrue(provider.claimed)
        harness.close()

    def test_renewal_must_extend_and_still_be_live_when_it_returns(self) -> None:
        harness, provider, _, clock = self.make()
        harness.open()
        lease = harness._lease  # type: ignore[attr-defined]
        assert lease is not None

        def nonextending(identity, token, lease_seconds):
            provider._event("renew", identity, token)
            return provider._lease(lease.expires_ns, identity=identity)

        provider.renew_exact_window_claim = nonextending  # type: ignore[method-assign]
        with self.assertRaisesRegex(WindowIdentityError, "did not extend"):
            harness.renew()
        self.assertFalse(provider.claimed)

        harness, provider, _, clock = self.make()
        harness.open()

        def expired(identity, token, lease_seconds):
            provider._event("renew", identity, token)
            clock.advance(10)
            return provider._lease(clock(), identity=identity)

        provider.renew_exact_window_claim = expired  # type: ignore[method-assign]
        with self.assertRaisesRegex(
            WindowIdentityError, "expired before the provider call returned"
        ):
            harness.renew()
        self.assertFalse(provider.claimed)

    def test_claim_expiry_during_provider_capture_fails_closed(self) -> None:
        harness, provider, _, clock = self.make()
        harness.open()
        harness._lease = provider._lease(  # type: ignore[attr-defined]
            clock() + 5, identity=IDENTITY
        )
        with self.assertRaisesRegex(WindowIdentityError, "expired during capture"):
            harness.capture()
        self.assertFalse(provider.claimed)
        self.assertTrue(harness.watchdog.buttons_neutral)

    def test_near_expiry_claim_is_renewed_before_button_down(self) -> None:
        harness, provider, _, clock = self.make(
            config=limits(lease_cleanup_margin_ns=100)
        )
        harness.open()
        harness._lease = provider._lease(clock() + 120, identity=IDENTITY)  # type: ignore[attr-defined]
        result = harness.fire("weak", 10, 10)
        self.assertEqual(result.status, ExecutionStatus.EXECUTED)
        names = [str(event[0]).split(":")[0] for event in provider.events]
        self.assertLess(names.index("renew"), names.index("down"))
        harness.close()

    def test_insufficient_renewal_headroom_refuses_before_button_down(self) -> None:
        harness, provider, _, clock = self.make(
            config=limits(lease_cleanup_margin_ns=100)
        )
        harness.open()
        harness._lease = provider._lease(  # type: ignore[attr-defined]
            clock() + 119, identity=IDENTITY
        )
        provider.renew_headroom_ns = 120
        with self.assertRaisesRegex(
            WindowIdentityError, "lacks button-up cleanup headroom"
        ):
            harness.fire("weak", 10, 10)
        self.assertFalse(
            any(str(event[0]).startswith("down:") for event in provider.events)
        )
        self.assertEqual(
            [event[0] for event in provider.events][-2:],
            ["release_all", "release_claim"],
        )

    def test_renewal_failure_cleans_up_without_button_down(self) -> None:
        harness, provider, _, clock = self.make(
            config=limits(lease_cleanup_margin_ns=100)
        )
        harness.open()
        harness._lease = provider._lease(clock() + 120, identity=IDENTITY)  # type: ignore[attr-defined]
        provider.fail.add("renew")
        with self.assertRaisesRegex(RuntimeError, "forced renew failure"):
            harness.fire("strong", 10, 10)
        self.assertFalse(
            any(str(event[0]).startswith("down:") for event in provider.events)
        )
        self.assertEqual(
            [event[0] for event in provider.events][-2:],
            ["release_all", "release_claim"],
        )
        self.assertFalse(provider.claimed)

    def test_long_zero_misroute_run(self) -> None:
        harness, provider, _, clock = self.make(config=limits(min_click_interval_ns=1))
        harness.open()
        for index in range(2_000):
            if index:
                clock.advance(1)
            provider.source_sequence = index * 2 + 1
            provider.frame_payload = f"before-{index}".encode()
            harness.capture()
            result = harness.fire(
                "weak" if index % 2 == 0 else "strong",
                index % 638,
                (index * 7) % 478,
            )
            self.assertEqual(result.status, ExecutionStatus.EXECUTED)
            provider.source_sequence = index * 2 + 2
            provider.frame_payload = f"after-{index}".encode()
            effect_frame = harness.capture()
            harness.record_action_effect(
                result.proposed.sequence,
                EffectStatus.MISSED,
                frame_sequence=effect_frame.sequence,
            )
        harness.close()
        routed = [
            event
            for event in provider.events
            if event[0] not in {"safety", "capabilities"}
        ]
        self.assertTrue(all(event[1] == IDENTITY for event in routed))
        self.assertEqual(
            sum(str(event[0]).startswith("down:") for event in routed), 2_000
        )
        self.assertEqual(
            sum(str(event[0]).startswith("up:") for event in routed), 2_000
        )


if __name__ == "__main__":
    unittest.main()
