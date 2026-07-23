"""Fail-closed operational preflight for R4 live measurement."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .harness import (
    HarnessProvider,
    InputCapabilities,
    SessionSafety,
    WindowIdentity,
)
from .runtime import (
    CANONICAL_WINE_SHA256,
    DisposableRunAttestation,
    verify_attestation_unchanged,
)

PREFLIGHT_SCHEMA = "r4b-provider-preflight-v2"
CAPABILITY_NAME = "targeted_edges_broker_deadline_claim_neutralization"
MAX_PREFLIGHT_PRESENTATION_AGE_NS = 100_000_000


class OperationalError(RuntimeError):
    """A live preflight or measurement operation failed closed."""


def _capability_values(capabilities: InputCapabilities) -> dict[str, bool]:
    return {
        "explicit_button_down": capabilities.explicit_button_down,
        "explicit_button_up": capabilities.explicit_button_up,
        "release_all_buttons": capabilities.release_all_buttons,
        "atomic_click_only": capabilities.atomic_click_only,
        "automatic_release_deadline": capabilities.automatic_release_deadline,
        "neutralizes_on_claim_end_or_expiry": (
            capabilities.neutralizes_on_claim_end_or_expiry
        ),
    }


def _safety_values(safety: SessionSafety) -> dict[str, bool]:
    return {
        "exact_background_capture": safety.exact_background_capture,
        "exact_window_claims": safety.exact_window_claims,
        "targeted_input_safe": safety.targeted_input_safe,
    }


def _call_bracket(
    operation: Callable[[], Any],
    clock_ns: Callable[[], int],
) -> tuple[Any, int, int]:
    before = clock_ns()
    value = operation()
    after = clock_ns()
    if after < before:
        raise OperationalError("local monotonic clock moved backwards")
    return value, before, after


def _process_start_ticks(process_id: int) -> int:
    try:
        line = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
        return int(line[line.rfind(")") + 2 :].split()[19])
    except (OSError, ValueError, IndexError) as exc:
        raise OperationalError("claimed process generation is unavailable") from exc


def run_capture_preflight(
    provider: HarnessProvider,
    identity: WindowIdentity,
    attestation: DisposableRunAttestation,
    *,
    repo_root: Path,
    run_dir: Path,
    clock_ns: Callable[[], int] = time.monotonic_ns,
    lease_seconds: int = 30,
    launch_nonce_sha256: str,
    wine_prefix_sha256: str,
) -> dict[str, Any]:
    """Capture once under an exact claim and report input eligibility.

    This function never invokes a button operation, even when the provider is
    eligible.  Live input requires a separate explicitly armed calibration run.
    """

    if not 5 <= lease_seconds <= 300:
        raise OperationalError("preflight lease must be in [5, 300] seconds")
    safety, _, _ = _call_bracket(provider.current_session_safety, clock_ns)
    capabilities, _, _ = _call_bracket(provider.input_capabilities, clock_ns)
    if not isinstance(safety, SessionSafety) or not isinstance(
        capabilities, InputCapabilities
    ):
        raise OperationalError("provider returned malformed capability state")
    if not safety.exact_background_capture or not safety.exact_window_claims:
        raise OperationalError("provider cannot safely run an exact capture preflight")

    lease = None
    release_error: Exception | None = None
    try:
        lease, claim_before, claim_after = _call_bracket(
            lambda: provider.claim_exact_window(identity, lease_seconds),
            clock_ns,
        )
        if lease.identity != identity:
            raise OperationalError("provider claimed a different window")
        if (
            lease.generation is None
            or lease.broker_instance is None
            or lease.target is None
        ):
            raise OperationalError("claim lacks immutable broker/runtime binding")
        if (
            lease.target.executable_sha256
            != attestation.runtime_sha256["game_executable_sha256"]
            or lease.target.runtime_sha256 != attestation.runtime_identity_sha256
            or lease.target.wine_executable_sha256 != CANONICAL_WINE_SHA256
            or lease.target.wine_prefix_sha256 != wine_prefix_sha256
            or lease.target.launch_nonce_sha256 != launch_nonce_sha256
            or lease.target.process_start_ticks
            != _process_start_ticks(lease.target.process_id)
        ):
            raise OperationalError(
                "claimed process does not match the disposable runtime"
            )
        if lease.expires_ns <= claim_after:
            raise OperationalError("provider returned an expired claim")
        packet, capture_before, capture_after = _call_bracket(
            lambda: provider.capture_exact_window(identity, lease.token),
            clock_ns,
        )
        if capture_after >= lease.expires_ns:
            raise OperationalError("claim expired during capture")
        if packet.identity != identity:
            raise OperationalError("capture identity changed")
        if not (
            capture_before
            <= packet.request_ns
            <= packet.start_ns
            <= packet.completion_ns
            <= capture_after
        ):
            raise OperationalError("capture timestamps escape the local call bracket")
        cursor, cursor_before, cursor_after = _call_bracket(
            lambda: provider.current_cursor(identity, lease.token), clock_ns
        )
        if cursor_after >= lease.expires_ns:
            raise OperationalError("claim expired during cursor observation")
        if not cursor_before <= cursor.observed_ns <= cursor_after:
            raise OperationalError("cursor timestamp escapes the local call bracket")
        if not (
            math.isfinite(cursor.x)
            and math.isfinite(cursor.y)
            and 0 <= cursor.x < packet.window_bounds.width
            and 0 <= cursor.y < packet.window_bounds.height
        ):
            raise OperationalError("cursor is outside the claimed window")
        verify_attestation_unchanged(attestation, repo_root, run_dir)
        continuous_capture_ready = (
            packet.presentation_ns is not None
            and packet.source_sequence is not None
            and packet.canonical_pixel_sha256 is not None
            and capture_after - packet.presentation_ns
            <= MAX_PREFLIGHT_PRESENTATION_AGE_NS
        )
        eligible = (
            safety.ready
            and capabilities.supports_safe_shots
            and continuous_capture_ready
        )
        blockers: list[str] = []
        if not safety.targeted_input_safe:
            blockers.append("targeted_input_not_safe")
        if capabilities.atomic_click_only:
            blockers.append("atomic_click_only")
        if not capabilities.explicit_button_down:
            blockers.append("explicit_button_down_unavailable")
        if not capabilities.explicit_button_up:
            blockers.append("explicit_button_up_unavailable")
        if not capabilities.release_all_buttons:
            blockers.append("release_all_unavailable")
        if not capabilities.automatic_release_deadline:
            blockers.append("broker_release_deadline_unavailable")
        if not capabilities.neutralizes_on_claim_end_or_expiry:
            blockers.append("claim_neutralization_unavailable")
        if (
            packet.presentation_ns is None
            or packet.source_sequence is None
            or packet.canonical_pixel_sha256 is None
        ):
            blockers.append("continuous_capture_identity_unavailable")
        elif capture_after - packet.presentation_ns > MAX_PREFLIGHT_PRESENTATION_AGE_NS:
            blockers.append("capture_presentation_stale")
        return {
            "schema": PREFLIGHT_SCHEMA,
            "status": "input_eligible" if eligible else "input_ineligible",
            "input_provider_capability": CAPABILITY_NAME if eligible else None,
            "blockers": blockers,
            "safety": _safety_values(safety),
            "capabilities": _capability_values(capabilities),
            "runtime": {
                "experiment_id": attestation.experiment_id,
                "marker_sha256": attestation.marker_sha256,
                "runtime_sha256": dict(attestation.runtime_sha256),
                "wine_prefix_sha256": wine_prefix_sha256,
            },
            "capture": {
                "sha256": packet.sha256,
                "pixel_width": packet.pixel_width,
                "pixel_height": packet.pixel_height,
                "request_to_completion_ns": (packet.completion_ns - packet.request_ns),
                "call_duration_ns": capture_after - capture_before,
                "source_sequence_available": packet.source_sequence is not None,
                "presentation_timestamp_available": packet.presentation_ns is not None,
            },
            "cursor": {
                "inside_claimed_window": True,
                "call_duration_ns": cursor_after - cursor_before,
            },
            "claim": {
                "call_duration_ns": claim_after - claim_before,
                "released": True,
            },
            "input_operations_attempted": 0,
        }
    finally:
        if lease is not None:
            try:
                provider.release_exact_window_claim(lease.identity, lease.token)
            except Exception as exc:  # cleanup must override an apparent success
                release_error = exc
        if release_error is not None:
            raise OperationalError(
                "exact-window claim release failed"
            ) from release_error


def canonical_report_bytes(report: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                report,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise OperationalError("preflight report is not canonical JSON") from exc


def preflight_sha256(report: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_report_bytes(report)).hexdigest()
