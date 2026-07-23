"""Fail-closed original-game capture and legal mouse-input harness.

The module contains no desktop automation implementation.  A live adapter must
implement :class:`HarnessProvider`; in particular, an atomic-click-only adapter
cannot execute shots because the harness requires independently acknowledged
button-down, button-up, and release-all operations.
"""

from __future__ import annotations

import hashlib
import math
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from numbers import Real
from typing import Protocol, runtime_checkable


class HarnessError(RuntimeError):
    """Base class for fail-closed harness errors."""


class SafetyError(HarnessError):
    pass


class WindowIdentityError(HarnessError):
    pass


class FrameError(HarnessError):
    pass


class GeometryError(HarnessError):
    pass


class ActionError(HarnessError):
    pass


class UnsupportedInputError(ActionError):
    pass


class CleanupError(HarnessError):
    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        super().__init__("cleanup failed: " + "; ".join(errors))


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _audit_number(value: object) -> float | str | None:
    """Represent rejected numeric input without raising or inventing a value."""

    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else repr(result)


@dataclass(frozen=True, slots=True)
class WindowIdentity:
    address: str
    capture_id: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.address, str)
            or not isinstance(self.capture_id, str)
            or not self.address
            or not self.capture_id
        ):
            raise ValueError("window address and capture identity must be non-empty")


MONOTONIC_CLOCK_DOMAIN = "linux.clock_monotonic"


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or value == "0" * 64
    ):
        raise ValueError(f"{name} must be a nonzero lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class TargetRuntimeDescriptor:
    """Immutable process/runtime identity authorized by one window claim."""

    identity: WindowIdentity
    process_id: int
    process_start_ticks: int
    launch_nonce_sha256: str
    executable_sha256: str
    runtime_sha256: str
    wine_executable_sha256: str
    wine_prefix_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.identity, WindowIdentity):
            raise TypeError("target descriptor identity is invalid")
        if type(self.process_id) is not int or self.process_id <= 0:
            raise ValueError("target descriptor process ID must be positive")
        if type(self.process_start_ticks) is not int or self.process_start_ticks <= 0:
            raise ValueError("target descriptor process start time must be positive")
        _sha256(self.launch_nonce_sha256, "target launch nonce SHA-256")
        _sha256(self.executable_sha256, "target executable SHA-256")
        _sha256(self.runtime_sha256, "target runtime SHA-256")
        _sha256(self.wine_executable_sha256, "target Wine SHA-256")
        _sha256(self.wine_prefix_sha256, "target Wine-prefix SHA-256")


class ClaimToken:
    """Opaque fencing token whose representation never discloses its value."""

    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        if not isinstance(value, str) or not value:
            raise ValueError("claim token must be a non-empty string")
        self.__value = value

    def __repr__(self) -> str:
        return "ClaimToken(<redacted>)"

    __str__ = __repr__

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ClaimToken) and self.__value == other.__value

    def __hash__(self) -> int:
        return hash(self.__value)


@dataclass(frozen=True, slots=True)
class SessionSafety:
    exact_background_capture: bool
    exact_window_claims: bool
    targeted_input_safe: bool
    detail: str = ""

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.exact_background_capture,
                self.exact_window_claims,
                self.targeted_input_safe,
            )
        ):
            raise TypeError("session safety flags must be booleans")
        if not isinstance(self.detail, str):
            raise TypeError("session safety detail must be text")

    @property
    def ready(self) -> bool:
        return (
            self.exact_background_capture
            and self.exact_window_claims
            and self.targeted_input_safe
        )


@dataclass(frozen=True, slots=True)
class InputCapabilities:
    explicit_button_down: bool
    explicit_button_up: bool
    release_all_buttons: bool
    atomic_click_only: bool = False
    automatic_release_deadline: bool = False
    neutralizes_on_claim_end_or_expiry: bool = False

    def __post_init__(self) -> None:
        if any(
            type(value) is not bool
            for value in (
                self.explicit_button_down,
                self.explicit_button_up,
                self.release_all_buttons,
                self.atomic_click_only,
                self.automatic_release_deadline,
                self.neutralizes_on_claim_end_or_expiry,
            )
        ):
            raise TypeError("input capability flags must be booleans")

    @property
    def supports_safe_shots(self) -> bool:
        return (
            self.explicit_button_down
            and self.explicit_button_up
            and self.release_all_buttons
            and not self.atomic_click_only
            and self.automatic_release_deadline
            and self.neutralizes_on_claim_end_or_expiry
        )


@dataclass(frozen=True, slots=True)
class ClaimLease:
    identity: WindowIdentity
    token: ClaimToken
    expires_ns: int
    generation: int | None = None
    broker_instance: str | None = None
    target: TargetRuntimeDescriptor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, WindowIdentity) or not isinstance(
            self.token, ClaimToken
        ):
            raise TypeError("claim lease identity/token types are invalid")
        if type(self.expires_ns) is not int or self.expires_ns <= 0:
            raise ValueError("claim expiry must be a positive monotonic timestamp")
        if self.generation is not None and (
            type(self.generation) is not int or self.generation <= 0
        ):
            raise ValueError("claim generation must be positive")
        if self.broker_instance is not None and (
            not isinstance(self.broker_instance, str)
            or not self.broker_instance
            or len(self.broker_instance) > 256
            or any(
                character in self.broker_instance for character in ("\0", "\n", "\r")
            )
        ):
            raise ValueError("broker instance is invalid")
        if self.target is not None:
            if not isinstance(self.target, TargetRuntimeDescriptor):
                raise TypeError("claim target descriptor is invalid")
            if self.target.identity != self.identity:
                raise ValueError("claim target descriptor identity differs")


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = tuple(
            _finite(value, name)
            for value, name in zip(
                (self.x, self.y, self.width, self.height),
                ("rect.x", "rect.y", "rect.width", "rect.height"),
                strict=True,
            )
        )
        if values[2] <= 0 or values[3] <= 0:
            raise ValueError("rectangle dimensions must be positive")

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def edge_drift(self, other: Rect) -> float:
        return max(
            abs(self.x - other.x),
            abs(self.y - other.y),
            abs((self.x + self.width) - (other.x + other.width)),
            abs((self.y + self.height) - (other.y + other.height)),
        )


@dataclass(frozen=True, slots=True)
class CapturePacket:
    pixels: bytes
    identity: WindowIdentity
    window_bounds: Rect
    pixel_width: int
    pixel_height: int
    request_ns: int
    start_ns: int
    completion_ns: int
    presentation_ns: int | None = None
    source_sequence: int | None = None
    color_format: str = "png"
    canonical_pixel_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.identity, WindowIdentity) or not isinstance(
            self.window_bounds, Rect
        ):
            raise TypeError("capture identity/window bounds types are invalid")
        if not isinstance(self.pixels, bytes) or not self.pixels:
            raise ValueError("capture pixels must be non-empty bytes")
        if type(self.pixel_width) is not int or type(self.pixel_height) is not int:
            raise ValueError("pixel dimensions must be integers")
        if self.pixel_width <= 0 or self.pixel_height <= 0:
            raise ValueError("pixel dimensions must be positive")
        stamps = (self.request_ns, self.start_ns, self.completion_ns)
        if any(type(value) is not int or value < 0 for value in stamps):
            raise ValueError("capture timestamps must be nonnegative integers")
        if not self.request_ns <= self.start_ns <= self.completion_ns:
            raise ValueError("capture timestamps are not ordered")
        if self.presentation_ns is not None and (
            type(self.presentation_ns) is not int
            or self.presentation_ns < 0
            or self.presentation_ns > self.completion_ns
        ):
            raise ValueError("presentation timestamp is invalid")
        if self.source_sequence is not None and (
            type(self.source_sequence) is not int or self.source_sequence < 0
        ):
            raise ValueError("source sequence must be a nonnegative integer")
        if not isinstance(self.color_format, str) or not self.color_format:
            raise ValueError("capture color format must be non-empty text")
        if self.canonical_pixel_sha256 is not None and (
            not isinstance(self.canonical_pixel_sha256, str)
            or len(self.canonical_pixel_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.canonical_pixel_sha256
            )
        ):
            raise ValueError("canonical pixel SHA-256 is invalid")

    @property
    def sha256(self) -> str:
        return self.canonical_pixel_sha256 or hashlib.sha256(self.pixels).hexdigest()


@dataclass(frozen=True, slots=True)
class GeometryAssessment:
    crop: Rect
    transform_age_ns: int
    residual_px: float
    confidence: float
    drifted: bool = False
    calibration_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.crop, Rect):
            raise TypeError("geometry crop must be a Rect")
        if type(self.transform_age_ns) is not int or self.transform_age_ns < 0:
            raise ValueError("transform age must be a nonnegative integer")
        residual = _finite(self.residual_px, "geometry residual")
        confidence = _finite(self.confidence, "geometry confidence")
        if residual < 0 or not 0 <= confidence <= 1:
            raise ValueError("invalid geometry quality")
        if type(self.drifted) is not bool or not isinstance(self.calibration_id, str):
            raise TypeError("invalid geometry drift/calibration fields")


@runtime_checkable
class GeometryAdapter(Protocol):
    """Adapter boundary for ``geometry.py`` or an equivalent calibrator."""

    def assess(self, capture: CapturePacket, now_ns: int) -> GeometryAssessment: ...

    def client_to_window(
        self,
        x: float,
        y: float,
        assessment: GeometryAssessment,
    ) -> tuple[float, float]: ...


@runtime_checkable
class TimingObserver(Protocol):
    """Optional boundary for a causal estimator in ``timing.py``."""

    def observe_frame(self, frame: FrameRecord) -> None: ...


@runtime_checkable
class RuntimeGuard(Protocol):
    """Launcher-owned runtime and exact process-generation guard."""

    def verify_runtime_unchanged(self) -> None: ...

    def verify_input_environment_unchanged(self) -> None: ...

    def verify_target_binding(
        self, process_id: int, process_start_ticks: int
    ) -> None: ...


class ScreenState(str, Enum):
    GAMEPLAY = "gameplay"
    MENU = "menu"
    GAME_OVER = "game_over"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ScreenClassification:
    state: ScreenState
    confidence: float
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.state, ScreenState):
            raise TypeError("screen state must be a ScreenState")
        confidence = _finite(self.confidence, "screen confidence")
        if not 0 <= confidence <= 1:
            raise ValueError("screen confidence must be in [0, 1]")
        if not isinstance(self.detail, str):
            raise TypeError("screen classification detail must be text")


@runtime_checkable
class ScreenClassifier(Protocol):
    def classify(
        self, capture: CapturePacket, geometry: GeometryAssessment
    ) -> ScreenClassification: ...


@dataclass(frozen=True, slots=True)
class CursorSample:
    x: float
    y: float
    observed_ns: int

    def __post_init__(self) -> None:
        _finite(self.x, "cursor x")
        _finite(self.y, "cursor y")
        if type(self.observed_ns) is not int or self.observed_ns < 0:
            raise ValueError("cursor timestamp must be nonnegative")


@dataclass(frozen=True, slots=True)
class InputAcknowledgement:
    injected_ns: int
    acknowledged_ns: int
    acknowledged: bool = True
    detail: str = ""
    operation_id: int | None = None
    identity: WindowIdentity | None = None
    claim_generation: int | None = None
    broker_instance: str | None = None
    button: str | None = None
    button_state: str | None = None
    accepted_release_deadline_ns: int | None = None
    clock_domain: str | None = None

    def __post_init__(self) -> None:
        if (
            type(self.injected_ns) is not int
            or type(self.acknowledged_ns) is not int
            or self.injected_ns < 0
            or self.acknowledged_ns < self.injected_ns
        ):
            raise ValueError("input acknowledgment timestamps are invalid")
        if type(self.acknowledged) is not bool or not isinstance(self.detail, str):
            raise TypeError("invalid input acknowledgment fields")
        if self.operation_id is not None and (
            type(self.operation_id) is not int or self.operation_id <= 0
        ):
            raise ValueError("input operation ID must be positive")
        if self.identity is not None and not isinstance(self.identity, WindowIdentity):
            raise TypeError("input acknowledgment identity is invalid")
        if self.claim_generation is not None and (
            type(self.claim_generation) is not int or self.claim_generation <= 0
        ):
            raise ValueError("input acknowledgment claim generation must be positive")
        if self.broker_instance is not None and (
            not isinstance(self.broker_instance, str)
            or not self.broker_instance
            or len(self.broker_instance) > 256
        ):
            raise ValueError("input acknowledgment broker instance is invalid")
        if self.button is not None and self.button not in {
            "left",
            "right",
            "all",
        }:
            raise ValueError("input acknowledgment button is invalid")
        if self.button_state is not None and self.button_state not in {
            "down",
            "up",
            "neutral",
        }:
            raise ValueError("input acknowledgment button state is invalid")
        if self.accepted_release_deadline_ns is not None and (
            type(self.accepted_release_deadline_ns) is not int
            or self.accepted_release_deadline_ns <= 0
        ):
            raise ValueError("accepted release deadline must be positive")
        if self.clock_domain is not None and (
            not isinstance(self.clock_domain, str) or not self.clock_domain
        ):
            raise ValueError("input acknowledgment clock domain is invalid")


@runtime_checkable
class HarnessProvider(Protocol):
    """Strict live-I/O boundary.

    Tokens must be enforced on every operation.  A provider advertising safe
    shots must neutralize a held button no later than ``release_deadline_ns``
    even if the caller stalls, and must neutralize every button when the claim
    is released or expires.
    """

    def current_session_safety(self) -> SessionSafety: ...

    def input_capabilities(self) -> InputCapabilities: ...

    def claim_exact_window(
        self, identity: WindowIdentity, lease_seconds: int
    ) -> ClaimLease: ...

    def renew_exact_window_claim(
        self, identity: WindowIdentity, token: ClaimToken, lease_seconds: int
    ) -> ClaimLease: ...

    def release_exact_window_claim(
        self, identity: WindowIdentity, token: ClaimToken
    ) -> None: ...

    def capture_exact_window(
        self, identity: WindowIdentity, token: ClaimToken
    ) -> CapturePacket: ...

    def current_cursor(
        self, identity: WindowIdentity, token: ClaimToken
    ) -> CursorSample: ...

    def targeted_button_down(
        self,
        identity: WindowIdentity,
        token: ClaimToken,
        button: str,
        x: float,
        y: float,
        release_deadline_ns: int,
        latest_injection_ns: int,
    ) -> InputAcknowledgement: ...

    def targeted_button_up(
        self,
        identity: WindowIdentity,
        token: ClaimToken,
        button: str,
        x: float,
        y: float,
    ) -> InputAcknowledgement: ...

    def release_all_buttons(
        self, identity: WindowIdentity, token: ClaimToken
    ) -> InputAcknowledgement: ...


class FrameFlag(str, Enum):
    DUPLICATE = "duplicate"
    DROPPED = "dropped"
    STALE = "stale"
    OUT_OF_ORDER = "out_of_order"
    BUFFER_OVERFLOW = "buffer_overflow"
    GEOMETRY_DRIFT = "geometry_drift"
    NON_GAMEPLAY = "non_gameplay"


@dataclass(frozen=True, slots=True)
class FrameRecord:
    sequence: int
    capture: CapturePacket
    geometry: GeometryAssessment
    screen: ScreenClassification
    flags: frozenset[FrameFlag]
    dropped_count: int = 0

    @property
    def usable(self) -> bool:
        return not self.flags.intersection(
            {
                FrameFlag.STALE,
                FrameFlag.OUT_OF_ORDER,
                FrameFlag.GEOMETRY_DRIFT,
                FrameFlag.NON_GAMEPLAY,
            }
        )


class BoundedFrameBuffer:
    def __init__(self, capacity: int) -> None:
        if type(capacity) is not int or capacity < 1:
            raise ValueError("frame-buffer capacity must be positive")
        self._frames: deque[FrameRecord] = deque(maxlen=capacity)
        self.overflow_count = 0

    def append(self, frame: FrameRecord) -> bool:
        overflow = len(self._frames) == self._frames.maxlen
        if overflow:
            self.overflow_count += 1
        self._frames.append(frame)
        return overflow

    @property
    def full(self) -> bool:
        return len(self._frames) == self._frames.maxlen

    @property
    def frames(self) -> tuple[FrameRecord, ...]:
        return tuple(self._frames)

    @property
    def latest(self) -> FrameRecord | None:
        return self._frames[-1] if self._frames else None


class ShotKind(str, Enum):
    WEAK = "weak"
    STRONG = "strong"

    @property
    def button(self) -> str:
        return "left" if self is ShotKind.WEAK else "right"


@dataclass(frozen=True, slots=True)
class ProposedAction:
    sequence: int
    kind: str
    client_x: float | str | None
    client_y: float | str | None
    requested_ns: int


class ExecutionStatus(str, Enum):
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"


class EffectStatus(str, Enum):
    PENDING = "pending"
    CONFIRMED = "confirmed"
    MISSED = "missed"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True, slots=True)
class ExecutedAction:
    proposed: ProposedAction
    status: ExecutionStatus
    window_x: float | None = None
    window_y: float | None = None
    down: InputAcknowledgement | None = None
    up: InputAcknowledgement | None = None
    completed_ns: int | None = None
    detail: str = ""
    effect_status: EffectStatus = EffectStatus.PENDING
    first_visible_ns: int | None = None
    effect_frame_sequence: int | None = None
    effect_detail: str = ""
    release_deadline_ns: int | None = None


@dataclass(frozen=True, slots=True)
class HarnessLimits:
    lease_seconds: int = 60
    frame_buffer_capacity: int = 8
    stale_after_ns: int = 100_000_000
    max_duplicate_run: int = 2
    max_source_drop_gap: int = 0
    client_width: float = 640.0
    client_height: float = 480.0
    max_transform_age_ns: int = 1_000_000_000
    max_geometry_residual_px: float = 2.0
    min_geometry_confidence: float = 0.95
    min_screen_confidence: float = 0.95
    max_crop_drift_px: float = 1.0
    press_duration_ns: int | None = None
    max_press_duration_error_ns: int = 5_000_000
    min_click_interval_ns: int | None = None
    lease_cleanup_margin_ns: int = 100_000_000
    cursor_mode: str = "unsupported"
    max_cursor_speed_per_second: float | None = None

    def __post_init__(self) -> None:
        if not 5 <= self.lease_seconds <= 300:
            raise ValueError("lease seconds must be in [5, 300]")
        for name in (
            "frame_buffer_capacity",
            "stale_after_ns",
            "max_transform_age_ns",
            "lease_cleanup_margin_ns",
            "max_press_duration_error_ns",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("max_duplicate_run", "max_source_drop_gap"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in (
            "client_width",
            "client_height",
            "max_geometry_residual_px",
            "min_geometry_confidence",
            "min_screen_confidence",
            "max_crop_drift_px",
        ):
            if _finite(getattr(self, name), name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.client_width <= 0 or self.client_height <= 0:
            raise ValueError("client dimensions must be positive")
        if not 0 <= self.min_geometry_confidence <= 1:
            raise ValueError("geometry confidence must be in [0, 1]")
        if not 0 <= self.min_screen_confidence <= 1:
            raise ValueError("screen confidence must be in [0, 1]")
        for name in ("press_duration_ns", "min_click_interval_ns"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        if self.cursor_mode not in {
            "unsupported",
            "abstract_teleport",
            "bounded_speed",
        }:
            raise ValueError("invalid cursor mode")
        if self.cursor_mode == "bounded_speed":
            if (
                self.max_cursor_speed_per_second is None
                or _finite(self.max_cursor_speed_per_second, "max cursor speed") <= 0
            ):
                raise ValueError("bounded-speed cursor mode requires a positive limit")


@dataclass(frozen=True, slots=True)
class WatchdogState:
    healthy: bool
    stopped: bool
    reasons: tuple[str, ...]
    claim_active: bool
    buttons_neutral: bool
    pending_actions: int
    frames_seen: int
    duplicate_run: int
    dropped_frames: int
    buffer_overflows: int
    cleanup_errors: tuple[str, ...]


class OriginalGameHarness:
    """Claim-bound capture and explicit-edge shot executor."""

    def __init__(
        self,
        provider: HarnessProvider,
        identity: WindowIdentity,
        geometry: GeometryAdapter,
        screen_classifier: ScreenClassifier,
        *,
        limits: HarnessLimits | None = None,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep_ns: Callable[[int], None] | None = None,
        timing_observer: TimingObserver | None = None,
        runtime_guard: RuntimeGuard | None = None,
    ) -> None:
        self.provider = provider
        self.identity = identity
        self.geometry = geometry
        self.screen_classifier = screen_classifier
        self.limits = limits or HarnessLimits()
        if runtime_guard is not None and not isinstance(runtime_guard, RuntimeGuard):
            raise TypeError("runtime guard does not implement the required protocol")
        self._clock_ns = clock_ns
        self._sleep_ns = sleep_ns or (lambda duration: time.sleep(duration / 1e9))
        self._timing_observer = timing_observer
        self._runtime_guard = runtime_guard
        self.buffer = BoundedFrameBuffer(self.limits.frame_buffer_capacity)
        self._lease: ClaimLease | None = None
        self._target: TargetRuntimeDescriptor | None = None
        self._capabilities: InputCapabilities | None = None
        self._cursor: CursorSample | None = None
        self._pending = False
        self._buttons_neutral = True
        self._stopped = False
        self._reasons: list[str] = []
        self._cleanup_errors: list[str] = []
        self._duplicate_run = 0
        self._dropped_frames = 0
        self._frame_sequence = 0
        self._action_sequence = 0
        self._last_input_operation_id = 0
        self._last_click_ns: int | None = None
        self._proposed: list[ProposedAction] = []
        self._executed: list[ExecutedAction] = []
        self._state_lock = threading.RLock()

    @property
    def proposed_actions(self) -> tuple[ProposedAction, ...]:
        return tuple(self._proposed)

    @property
    def executed_actions(self) -> tuple[ExecutedAction, ...]:
        return tuple(self._executed)

    @property
    def target_descriptor(self) -> TargetRuntimeDescriptor | None:
        return self._target

    def verify_runtime_unchanged(self) -> None:
        """Re-attest the disposable runtime or fail if no guard was installed."""

        if self._runtime_guard is None:
            raise SafetyError("live runtime re-attestation is not configured")
        self._runtime_guard.verify_runtime_unchanged()
        if self._target is not None:
            self._runtime_guard.verify_target_binding(
                self._target.process_id,
                self._target.process_start_ticks,
            )

    def _verify_runtime_if_configured(self) -> None:
        if self._runtime_guard is not None:
            self.verify_runtime_unchanged()

    def verify_input_environment_unchanged(self) -> None:
        """Re-attest Wine-prefix state used to interpret the next input."""

        if self._runtime_guard is None:
            raise SafetyError("live input-environment attestation is not configured")
        self._runtime_guard.verify_input_environment_unchanged()
        if self._target is not None:
            self._runtime_guard.verify_target_binding(
                self._target.process_id,
                self._target.process_start_ticks,
            )

    @property
    def _pending_effects(self) -> int:
        return sum(
            action.status is ExecutionStatus.EXECUTED
            and action.effect_status is EffectStatus.PENDING
            for action in self._executed
        )

    @property
    def watchdog(self) -> WatchdogState:
        return WatchdogState(
            healthy=not self._reasons
            and not self._cleanup_errors
            and not self._stopped,
            stopped=self._stopped,
            reasons=tuple(self._reasons),
            claim_active=self._lease is not None,
            buttons_neutral=self._buttons_neutral,
            pending_actions=int(self._pending) + self._pending_effects,
            frames_seen=self._frame_sequence,
            duplicate_run=self._duplicate_run,
            dropped_frames=self._dropped_frames,
            buffer_overflows=self.buffer.overflow_count,
            cleanup_errors=tuple(self._cleanup_errors),
        )

    def __enter__(self) -> OriginalGameHarness:
        return self.open()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        try:
            self.close()
        except CleanupError:
            if exc is None:
                raise
        return False

    def _require_active(self) -> ClaimLease:
        if self._stopped or self._lease is None:
            raise HarnessError("harness is not active")
        return self._lease

    def _reason(self, error: BaseException | str) -> None:
        text = str(error)
        if text not in self._reasons:
            self._reasons.append(text)

    def _validate_lease(
        self,
        lease: object,
        returned_ns: int,
        *,
        previous: ClaimLease | None = None,
    ) -> ClaimLease:
        if not isinstance(lease, ClaimLease):
            raise WindowIdentityError("provider returned an invalid claim lease")
        if lease.identity != self.identity:
            raise WindowIdentityError("claim changed window identity")
        if (
            lease.generation is None
            or lease.broker_instance is None
            or lease.target is None
        ):
            raise WindowIdentityError(
                "claim lacks generation, broker, or target/runtime binding"
            )
        if lease.target.identity != self.identity:
            raise WindowIdentityError("claim target descriptor changed window identity")
        if returned_ns >= lease.expires_ns:
            raise WindowIdentityError("claim expired before the provider call returned")
        if previous is not None:
            if lease.token != previous.token:
                raise WindowIdentityError("claim renewal changed fencing token")
            if lease.generation != previous.generation:
                raise WindowIdentityError("claim renewal changed generation")
            if lease.broker_instance != previous.broker_instance:
                raise WindowIdentityError("claim renewal changed broker instance")
            if lease.target != previous.target:
                raise WindowIdentityError(
                    "claim renewal changed target/runtime identity"
                )
            if lease.expires_ns <= previous.expires_ns:
                raise WindowIdentityError("claim renewal did not extend its expiry")
        return lease

    @staticmethod
    def _validate_provider_interval(
        start_ns: int,
        end_ns: int,
        reported_start_ns: int,
        reported_end_ns: int,
        operation: str,
    ) -> None:
        if not (start_ns <= reported_start_ns <= reported_end_ns <= end_ns):
            raise HarnessError(
                f"{operation} timestamps are outside the local provider-call interval"
            )

    @staticmethod
    def _ensure_lease_live(lease: ClaimLease, at_ns: int, operation: str) -> None:
        if at_ns >= lease.expires_ns:
            raise WindowIdentityError(f"window claim expired during {operation}")

    def open(self) -> OriginalGameHarness:
        with self._state_lock:
            return self._open()

    def _open(self) -> OriginalGameHarness:
        if self._lease is not None or self._stopped:
            raise HarnessError("harness cannot be opened in its current state")
        safety = self.provider.current_session_safety()
        if not safety.ready:
            raise SafetyError(safety.detail or "current session is unsafe")
        self._capabilities = self.provider.input_capabilities()
        if self._capabilities.supports_safe_shots and self._runtime_guard is None:
            raise SafetyError(
                "safe live input requires runtime and Wine-prefix re-attestation"
            )
        try:
            lease = self.provider.claim_exact_window(
                self.identity, self.limits.lease_seconds
            )
            claim_return_ns = self._clock_ns()
            if not isinstance(lease, ClaimLease):
                raise WindowIdentityError("provider returned an invalid claim lease")
            if lease.identity != self.identity:
                # Releasing the unexpected lease is safe; sending even a
                # release-all input to that unapproved identity is not.  A
                # release-all against the approved identity must be fenced out
                # if the provider accidentally bound the token elsewhere.
                try:
                    self.provider.release_all_buttons(self.identity, lease.token)
                except Exception as cleanup_exc:
                    self._cleanup_errors.append(f"release_all_buttons: {cleanup_exc}")
                try:
                    self.provider.release_exact_window_claim(
                        lease.identity, lease.token
                    )
                except Exception as cleanup_exc:
                    self._cleanup_errors.append(
                        f"release_unexpected_window_claim: {cleanup_exc}"
                    )
                raise WindowIdentityError("provider claimed a different window")
            self._lease = self._validate_lease(lease, claim_return_ns)
            self._target = self._lease.target
            self._last_input_operation_id = 0
            self.capture()
            lease = self._require_active()
            cursor_request_ns = self._clock_ns()
            cursor = self.provider.current_cursor(self.identity, lease.token)
            cursor_return_ns = self._clock_ns()
            self._validate_provider_interval(
                cursor_request_ns,
                cursor_return_ns,
                cursor.observed_ns,
                cursor.observed_ns,
                "cursor observation",
            )
            self._ensure_lease_live(lease, cursor_return_ns, "cursor observation")
            self._validate_window_point(cursor.x, cursor.y)
            self._cursor = cursor
            return self
        except Exception as exc:
            self._reason(exc)
            self._cleanup()
            raise

    def renew(self) -> None:
        with self._state_lock:
            self._renew()

    def _renew(self) -> None:
        lease = self._require_active()
        try:
            if self._clock_ns() >= lease.expires_ns:
                raise WindowIdentityError("window claim expired before renewal")
            renewed = self.provider.renew_exact_window_claim(
                self.identity, lease.token, self.limits.lease_seconds
            )
            returned_ns = self._clock_ns()
            self._lease = self._validate_lease(renewed, returned_ns, previous=lease)
        except Exception as exc:
            self._abort(exc)

    def _classify(
        self,
        packet: CapturePacket,
        geometry: GeometryAssessment,
        screen: ScreenClassification,
        now_ns: int,
    ) -> tuple[set[FrameFlag], int]:
        flags: set[FrameFlag] = set()
        dropped = 0
        previous = self.buffer.latest
        observed_ns = (
            packet.presentation_ns
            if packet.presentation_ns is not None
            else packet.completion_ns
        )
        if now_ns - observed_ns > self.limits.stale_after_ns:
            flags.add(FrameFlag.STALE)
        if observed_ns > now_ns:
            flags.add(FrameFlag.OUT_OF_ORDER)
        if previous is not None:
            prior = previous.capture
            prior_observed_ns = (
                prior.presentation_ns
                if prior.presentation_ns is not None
                else prior.completion_ns
            )
            if packet.sha256 == prior.sha256:
                flags.add(FrameFlag.DUPLICATE)
            if observed_ns <= prior_observed_ns:
                flags.add(FrameFlag.OUT_OF_ORDER)
            if packet.source_sequence is not None and prior.source_sequence is not None:
                if packet.source_sequence <= prior.source_sequence:
                    flags.add(FrameFlag.OUT_OF_ORDER)
                elif packet.source_sequence > prior.source_sequence + 1:
                    dropped = packet.source_sequence - prior.source_sequence - 1
                    flags.add(FrameFlag.DROPPED)
            if (
                geometry.crop.edge_drift(previous.geometry.crop)
                > self.limits.max_crop_drift_px
            ):
                flags.add(FrameFlag.GEOMETRY_DRIFT)
            if (
                packet.window_bounds.edge_drift(prior.window_bounds)
                > self.limits.max_crop_drift_px
            ):
                flags.add(FrameFlag.GEOMETRY_DRIFT)
        if geometry.drifted:
            flags.add(FrameFlag.GEOMETRY_DRIFT)
        if (
            screen.state is not ScreenState.GAMEPLAY
            or screen.confidence < self.limits.min_screen_confidence
        ):
            flags.add(FrameFlag.NON_GAMEPLAY)
        return flags, dropped

    def _validate_geometry(
        self,
        assessment: GeometryAssessment,
        capture: CapturePacket,
    ) -> None:
        if assessment.transform_age_ns > self.limits.max_transform_age_ns:
            raise GeometryError("coordinate transform is stale")
        if assessment.residual_px > self.limits.max_geometry_residual_px:
            raise GeometryError("coordinate residual exceeds watchdog limit")
        if assessment.confidence < self.limits.min_geometry_confidence:
            raise GeometryError("coordinate confidence is below watchdog limit")
        crop = assessment.crop
        if (
            crop.x < 0
            or crop.y < 0
            or crop.x + crop.width > capture.pixel_width
            or crop.y + crop.height > capture.pixel_height
        ):
            raise GeometryError("puzzle crop is outside captured pixels")

    def capture(self) -> FrameRecord:
        with self._state_lock:
            return self._capture()

    def _capture(self) -> FrameRecord:
        lease = self._require_active()
        try:
            self._verify_runtime_if_configured()
            if self._clock_ns() >= lease.expires_ns:
                raise WindowIdentityError("window claim expired before capture")
            safety = self.provider.current_session_safety()
            if not safety.ready:
                raise SafetyError(safety.detail or "current session became unsafe")
            capture_request_ns = self._clock_ns()
            self._ensure_lease_live(lease, capture_request_ns, "capture")
            packet = self.provider.capture_exact_window(self.identity, lease.token)
            self._verify_runtime_if_configured()
            if packet.identity != self.identity:
                raise WindowIdentityError("capture identity changed")
            now_ns = self._clock_ns()
            self._validate_provider_interval(
                capture_request_ns,
                now_ns,
                packet.request_ns,
                packet.completion_ns,
                "capture",
            )
            self._ensure_lease_live(lease, now_ns, "capture")
            if (
                self._capabilities is not None
                and self._capabilities.supports_safe_shots
                and (
                    packet.presentation_ns is None
                    or packet.source_sequence is None
                    or packet.canonical_pixel_sha256 is None
                )
            ):
                raise FrameError(
                    "live-qualified capture lacks presentation, source sequence, "
                    "or canonical pixel identity"
                )
            assessment = self.geometry.assess(packet, now_ns)
            self._validate_geometry(assessment, packet)
            screen = self.screen_classifier.classify(packet, assessment)
            flags, dropped = self._classify(packet, assessment, screen, now_ns)
            self._frame_sequence += 1
            if self.buffer.full:
                flags.add(FrameFlag.BUFFER_OVERFLOW)
            record = FrameRecord(
                self._frame_sequence,
                packet,
                assessment,
                screen,
                frozenset(flags),
                dropped,
            )
            self.buffer.append(record)
            self._duplicate_run = (
                self._duplicate_run + 1 if FrameFlag.DUPLICATE in flags else 0
            )
            self._dropped_frames += dropped
            if FrameFlag.OUT_OF_ORDER in flags:
                raise FrameError("capture is out of order")
            if FrameFlag.STALE in flags:
                raise FrameError("capture is stale")
            if FrameFlag.GEOMETRY_DRIFT in flags:
                raise GeometryError("capture geometry drifted")
            if self._duplicate_run > self.limits.max_duplicate_run:
                raise FrameError("duplicate-frame limit exceeded")
            if dropped > self.limits.max_source_drop_gap:
                raise FrameError("source-frame drop limit exceeded")
            if self._timing_observer is not None:
                self._timing_observer.observe_frame(record)
            return record
        except Exception as exc:
            self._abort(exc)

    def _validate_window_point(self, x: object, y: object) -> tuple[float, float]:
        x_value = _finite(x, "window x")
        y_value = _finite(y, "window y")
        latest = self.buffer.latest
        bounds = latest.capture.window_bounds if latest is not None else None
        if bounds is not None and not (
            0 <= x_value < bounds.width and 0 <= y_value < bounds.height
        ):
            raise ActionError("window-local coordinate is out of bounds")
        return x_value, y_value

    def _validate_action(
        self, kind: ShotKind | str, x: object, y: object, now_ns: int
    ) -> tuple[ShotKind, float, float]:
        try:
            parsed = ShotKind(kind)
        except (TypeError, ValueError) as exc:
            raise ActionError("only legal weak/strong shots are accepted") from exc
        try:
            client_x, client_y = _finite(x, "client x"), _finite(y, "client y")
        except (TypeError, ValueError) as exc:
            raise ActionError(str(exc)) from exc
        if not (
            0 <= client_x < self.limits.client_width
            and 0 <= client_y < self.limits.client_height
        ):
            raise ActionError("client coordinate is out of bounds")
        if self._pending or self._pending_effects:
            raise ActionError("maximum pending-action depth is one")
        if self.limits.press_duration_ns is None:
            raise UnsupportedInputError("measured press duration is not configured")
        if self.limits.min_click_interval_ns is None:
            raise UnsupportedInputError("measured click-rate limit is not configured")
        if self.limits.cursor_mode == "unsupported":
            raise UnsupportedInputError("cursor fairness contract is not configured")
        if (
            self._last_click_ns is not None
            and now_ns - self._last_click_ns < self.limits.min_click_interval_ns
        ):
            raise ActionError("click-rate limit exceeded")
        return parsed, client_x, client_y

    def _validate_ack(
        self,
        request_ns: int,
        ack: InputAcknowledgement,
        edge: str,
        lease: ClaimLease,
        *,
        button: str,
        button_state: str,
        release_deadline_ns: int | None,
        latest_injection_ns: int | None = None,
    ) -> int:
        if not isinstance(ack, InputAcknowledgement):
            raise ActionError(f"{edge} returned an invalid acknowledgment")
        if not ack.acknowledged:
            raise ActionError(f"{edge} was not acknowledged")
        if ack.injected_ns < request_ns:
            raise ActionError(f"{edge} acknowledgment predates its request")
        post_return_ns = self._clock_ns()
        if ack.acknowledged_ns > post_return_ns:
            raise ActionError(f"{edge} acknowledgment is in the future")
        expected = (
            ack.operation_id is not None
            and ack.operation_id > self._last_input_operation_id
            and ack.identity == self.identity
            and ack.claim_generation == lease.generation
            and ack.broker_instance == lease.broker_instance
            and ack.button == button
            and ack.button_state == button_state
            and ack.accepted_release_deadline_ns == release_deadline_ns
            and ack.clock_domain == MONOTONIC_CLOCK_DOMAIN
            and (
                latest_injection_ns is None
                or ack.injected_ns <= latest_injection_ns
            )
        )
        if not expected:
            raise ActionError(
                f"{edge} acknowledgment is not bound to this operation and claim"
            )
        assert ack.operation_id is not None
        self._last_input_operation_id = ack.operation_id
        return post_return_ns

    def _validate_cursor_travel(self, x: float, y: float, now_ns: int) -> None:
        if self.limits.cursor_mode != "bounded_speed":
            return
        cursor = self._cursor
        if cursor is None:
            raise ActionError("current cursor is unknown")
        elapsed = max(0, now_ns - cursor.observed_ns) / 1e9
        distance = math.hypot(x - cursor.x, y - cursor.y)
        speed = self.limits.max_cursor_speed_per_second
        assert speed is not None
        if distance > speed * elapsed:
            raise ActionError("cursor-speed limit exceeded")

    def _lease_for_button_down(self, lease: ClaimLease) -> ClaimLease:
        press_duration = self.limits.press_duration_ns
        if press_duration is None:
            raise UnsupportedInputError("measured press duration is not configured")
        required = press_duration + 2 * self.limits.lease_cleanup_margin_ns
        if lease.expires_ns - self._clock_ns() > required:
            return lease
        renewed = self.provider.renew_exact_window_claim(
            self.identity,
            lease.token,
            self.limits.lease_seconds,
        )
        returned_ns = self._clock_ns()
        renewed = self._validate_lease(renewed, returned_ns, previous=lease)
        if renewed.expires_ns - returned_ns <= required:
            raise WindowIdentityError("renewed claim lacks button-up cleanup headroom")
        self._lease = renewed
        return renewed

    def fire(self, kind: ShotKind | str, x: object, y: object) -> ExecutedAction:
        with self._state_lock:
            return self._fire(kind, x, y)

    def _fire(self, kind: ShotKind | str, x: object, y: object) -> ExecutedAction:
        lease = self._require_active()
        now_ns = self._clock_ns()
        self._action_sequence += 1
        try:
            parsed, client_x, client_y = self._validate_action(kind, x, y, now_ns)
        except Exception as exc:
            proposed = ProposedAction(
                self._action_sequence,
                kind.value
                if isinstance(kind, ShotKind)
                else kind
                if isinstance(kind, str)
                else f"<invalid {type(kind).__name__}>",
                _audit_number(x),
                _audit_number(y),
                now_ns,
            )
            self._proposed.append(proposed)
            self._executed.append(
                ExecutedAction(proposed, ExecutionStatus.REJECTED, detail=str(exc))
            )
            self._abort(exc)
        proposed = ProposedAction(
            self._action_sequence, parsed.value, client_x, client_y, now_ns
        )
        self._proposed.append(proposed)
        capabilities = self._capabilities
        if capabilities is None or not capabilities.supports_safe_shots:
            exc = UnsupportedInputError(
                "provider lacks explicit down/up/release-all, broker release "
                "deadlines, or claim-end neutralization; atomic click is unsupported"
            )
            self._executed.append(
                ExecutedAction(proposed, ExecutionStatus.REJECTED, detail=str(exc))
            )
            self._abort(exc)

        down: InputAcknowledgement | None = None
        up: InputAcknowledgement | None = None
        window_x: float | None = None
        window_y: float | None = None
        self._pending = True
        try:
            safety = self.provider.current_session_safety()
            if not safety.ready:
                raise SafetyError(safety.detail or "current session became unsafe")
            frame = self.buffer.latest
            if frame is None or not frame.usable:
                if frame is not None and FrameFlag.NON_GAMEPLAY in frame.flags:
                    raise ActionError(
                        "latest screen is not confidently classified as gameplay"
                    )
                raise FrameError("no current usable capture")
            if now_ns >= lease.expires_ns:
                raise WindowIdentityError("window claim expired")
            observed_ns = (
                frame.capture.presentation_ns
                if frame.capture.presentation_ns is not None
                else frame.capture.completion_ns
            )
            if now_ns - observed_ns > self.limits.stale_after_ns:
                raise FrameError("latest capture became stale before input")
            self._validate_geometry(frame.geometry, frame.capture)
            window_x, window_y = self.geometry.client_to_window(
                client_x, client_y, frame.geometry
            )
            window_x, window_y = self._validate_window_point(window_x, window_y)
            self._validate_cursor_travel(window_x, window_y, now_ns)
            lease = self._lease_for_button_down(lease)
            press_duration = self.limits.press_duration_ns
            assert press_duration is not None
            self._verify_runtime_if_configured()
            if self._runtime_guard is not None:
                self.verify_input_environment_unchanged()
            safety = self.provider.current_session_safety()
            if not safety.ready:
                raise SafetyError(
                    safety.detail or "current session became unsafe before button-down"
                )
            down_request = self._clock_ns()
            latest_injection_ns = min(
                observed_ns + self.limits.stale_after_ns,
                down_request + self.limits.lease_cleanup_margin_ns,
            )
            if down_request >= latest_injection_ns:
                raise FrameError("latest capture became stale before button-down")
            self._ensure_lease_live(lease, down_request, "button-down request")
            release_deadline_ns = (
                latest_injection_ns
                + press_duration
                + self.limits.lease_cleanup_margin_ns
            )
            if release_deadline_ns >= lease.expires_ns:
                raise WindowIdentityError(
                    "claim lacks broker release-deadline headroom"
                )
            down = self.provider.targeted_button_down(
                self.identity,
                lease.token,
                parsed.button,
                window_x,
                window_y,
                release_deadline_ns,
                latest_injection_ns,
            )
            down_return_ns = self._validate_ack(
                down_request,
                down,
                "button-down",
                lease,
                button=parsed.button,
                button_state="down",
                release_deadline_ns=release_deadline_ns,
                latest_injection_ns=latest_injection_ns,
            )
            self._buttons_neutral = False
            if down_return_ns >= release_deadline_ns:
                raise ActionError(
                    "button-down returned after its automatic release deadline"
                )
            self._ensure_lease_live(lease, down_return_ns, "button-down")
            if (
                lease.expires_ns - down.acknowledged_ns
                <= press_duration + self.limits.lease_cleanup_margin_ns
            ):
                raise WindowIdentityError(
                    "button-down acknowledgment consumed lease cleanup headroom"
                )
            target_up_ns = down.injected_ns + press_duration
            remaining_ns = target_up_ns - self._clock_ns()
            if remaining_ns > 0:
                self._sleep_ns(remaining_ns)
            up_request = self._clock_ns()
            if up_request >= release_deadline_ns:
                raise ActionError("automatic release deadline elapsed before button-up")
            up = self.provider.targeted_button_up(
                self.identity,
                lease.token,
                parsed.button,
                window_x,
                window_y,
            )
            up_return_ns = self._validate_ack(
                up_request,
                up,
                "button-up",
                lease,
                button=parsed.button,
                button_state="up",
                release_deadline_ns=release_deadline_ns,
            )
            self._verify_runtime_if_configured()
            if self._runtime_guard is not None:
                self.verify_input_environment_unchanged()
            if (
                up.acknowledged_ns >= release_deadline_ns
                or up_return_ns >= release_deadline_ns
            ):
                raise ActionError(
                    "button-up returned after its automatic release deadline"
                )
            self._ensure_lease_live(lease, up_return_ns, "button-up")
            actual_press_ns = up.injected_ns - down.injected_ns
            if (
                abs(actual_press_ns - press_duration)
                > self.limits.max_press_duration_error_ns
            ):
                raise ActionError(
                    "observed press duration exceeds the configured timing error"
                )
            self._buttons_neutral = True
            completed = up_return_ns
            if completed < up.acknowledged_ns:
                raise ActionError("action completion predates button-up acknowledgment")
            result = ExecutedAction(
                proposed=proposed,
                status=ExecutionStatus.EXECUTED,
                window_x=window_x,
                window_y=window_y,
                down=down,
                up=up,
                completed_ns=completed,
                release_deadline_ns=release_deadline_ns,
            )
            self._executed.append(result)
            self._last_click_ns = up.acknowledged_ns
            self._cursor = CursorSample(window_x, window_y, up.acknowledged_ns)
            return result
        except Exception as exc:
            self._executed.append(
                ExecutedAction(
                    proposed=proposed,
                    status=ExecutionStatus.FAILED,
                    window_x=window_x,
                    window_y=window_y,
                    down=down,
                    up=up,
                    completed_ns=self._clock_ns(),
                    detail=str(exc),
                    release_deadline_ns=(
                        release_deadline_ns
                        if "release_deadline_ns" in locals()
                        else None
                    ),
                )
            )
            self._abort(exc)
        finally:
            self._pending = False

    def record_action_effect(
        self,
        action_sequence: int,
        status: EffectStatus | str,
        *,
        frame_sequence: int | None = None,
        first_visible_ns: int | None = None,
        detail: str = "",
    ) -> ExecutedAction:
        """Resolve one executed shot from a later causal captured frame."""

        with self._state_lock:
            return self._record_action_effect(
                action_sequence,
                status,
                frame_sequence=frame_sequence,
                first_visible_ns=first_visible_ns,
                detail=detail,
            )

    def _record_action_effect(
        self,
        action_sequence: int,
        status: EffectStatus | str,
        *,
        frame_sequence: int | None = None,
        first_visible_ns: int | None = None,
        detail: str = "",
    ) -> ExecutedAction:
        self._require_active()
        try:
            if type(action_sequence) is not int or action_sequence < 1:
                raise ActionError("action sequence must be a positive integer")
            try:
                parsed = EffectStatus(status)
            except (TypeError, ValueError) as exc:
                raise ActionError("unknown action-effect status") from exc
            if parsed is EffectStatus.PENDING:
                raise ActionError("effect resolution cannot remain pending")
            if frame_sequence is not None and (
                type(frame_sequence) is not int or frame_sequence < 1
            ):
                raise ActionError("effect frame sequence must be a positive integer")
            if first_visible_ns is not None and (
                type(first_visible_ns) is not int or first_visible_ns < 0
            ):
                raise ActionError(
                    "first-visible timestamp must be a nonnegative integer"
                )
            if not isinstance(detail, str):
                raise ActionError("effect detail must be text")
            try:
                index = next(
                    index
                    for index, action in enumerate(self._executed)
                    if action.proposed.sequence == action_sequence
                )
            except StopIteration as exc:
                raise ActionError("unknown action sequence") from exc
            action = self._executed[index]
            if action.status is not ExecutionStatus.EXECUTED or action.up is None:
                raise ActionError("only an executed released shot can have an effect")
            if action.effect_status is not EffectStatus.PENDING:
                raise ActionError("action effect was already resolved")

            if frame_sequence is None:
                raise ActionError(
                    "terminal effect status requires a later captured frame"
                )
            frame = next(
                (
                    candidate
                    for candidate in self.buffer.frames
                    if candidate.sequence == frame_sequence
                ),
                None,
            )
            if frame is None:
                raise ActionError("effect frame is not in the bounded frame buffer")
            observed_ns = (
                frame.capture.presentation_ns
                if frame.capture.presentation_ns is not None
                else frame.capture.completion_ns
            )
            if observed_ns < action.up.acknowledged_ns:
                raise ActionError("effect observation predates button release")
            if observed_ns > self._clock_ns():
                raise ActionError("effect observation is in the future")

            visible: int | None = None
            if parsed is EffectStatus.CONFIRMED:
                if not frame.usable:
                    raise ActionError(
                        "confirmed effect requires a usable gameplay frame"
                    )
                if first_visible_ns is not None and first_visible_ns != observed_ns:
                    raise ActionError(
                        "first-visible timestamp does not match its captured frame"
                    )
                visible = observed_ns
            elif parsed is EffectStatus.MISSED:
                if first_visible_ns is not None:
                    raise ActionError(
                        "missed effect cannot have a first-visible timestamp"
                    )
            elif first_visible_ns is not None:
                if first_visible_ns != observed_ns:
                    raise ActionError(
                        "first-visible timestamp does not match its captured frame"
                    )
                visible = observed_ns

            updated = replace(
                action,
                effect_status=parsed,
                first_visible_ns=visible,
                effect_frame_sequence=frame_sequence,
                effect_detail=detail,
            )
            self._executed[index] = updated
            return updated
        except Exception as exc:
            self._abort(exc)

    def _abort(self, error: BaseException) -> None:
        self._reason(error)
        self._cleanup()
        raise error

    def _cleanup(self) -> None:
        lease = self._lease
        self._stopped = True
        if lease is None:
            return
        try:
            request_ns = self._clock_ns()
            ack = self.provider.release_all_buttons(self.identity, lease.token)
            self._validate_ack(
                request_ns,
                ack,
                "release-all",
                lease,
                button="all",
                button_state="neutral",
                release_deadline_ns=None,
            )
            self._buttons_neutral = True
        except Exception as exc:
            self._buttons_neutral = False
            self._cleanup_errors.append(f"release_all_buttons: {exc}")
        finally:
            claim_released = False
            try:
                self.provider.release_exact_window_claim(self.identity, lease.token)
                claim_released = True
            except Exception as exc:
                self._cleanup_errors.append(f"release_exact_window_claim: {exc}")
            finally:
                if (
                    claim_released
                    and self._capabilities is not None
                    and self._capabilities.neutralizes_on_claim_end_or_expiry
                ):
                    self._buttons_neutral = True
                self._lease = None

    def close(self) -> None:
        with self._state_lock:
            self._close()

    def _close(self) -> None:
        if not self._stopped:
            self._cleanup()
        if self._cleanup_errors:
            raise CleanupError(tuple(self._cleanup_errors))
