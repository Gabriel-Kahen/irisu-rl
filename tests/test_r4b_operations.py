from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_rl.original_game.harness import (
    CapturePacket,
    ClaimLease,
    ClaimToken,
    CursorSample,
    InputCapabilities,
    Rect,
    SessionSafety,
    TargetRuntimeDescriptor,
    WindowIdentity,
)
from irisu_rl.original_game.operations import (
    OperationalError,
    run_capture_preflight,
)
from irisu_rl.original_game.runtime import (
    CANONICAL_WINE_SHA256,
    REQUIRED_RUNTIME_FILES,
    attest_disposable_run,
)

IDENTITY = WindowIdentity("0xabc", "capture-1")
OTHER = WindowIdentity("0xdef", "capture-2")


class Clock:
    def __init__(self) -> None:
        self.value = 1_000_000_000

    def __call__(self) -> int:
        return self.value

    def tick(self, amount: int = 10) -> int:
        self.value += amount
        return self.value


class Provider:
    def __init__(self, clock: Clock, *, safe_input: bool) -> None:
        self.clock = clock
        self.token = ClaimToken("secret")
        self.released = 0
        self.input_calls = 0
        self.capture_identity = IDENTITY
        self.fail_release = False
        self.safety = SessionSafety(True, True, safe_input)
        self.capabilities = InputCapabilities(
            safe_input,
            safe_input,
            safe_input,
            atomic_click_only=not safe_input,
            automatic_release_deadline=safe_input,
            neutralizes_on_claim_end_or_expiry=safe_input,
        )

    def current_session_safety(self):
        self.clock.tick()
        return self.safety

    def input_capabilities(self):
        self.clock.tick()
        return self.capabilities

    def claim_exact_window(self, identity, lease_seconds):
        self.clock.tick()
        stat_line = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="ascii")
        start_ticks = int(stat_line[stat_line.rfind(")") + 2 :].split()[19])
        target = TargetRuntimeDescriptor(
            identity,
            os.getpid(),
            start_ticks,
            self.launch_nonce_sha256,
            self.attestation.runtime_sha256["game_executable_sha256"],
            self.attestation.runtime_identity_sha256,
            CANONICAL_WINE_SHA256,
            "f" * 64,
        )
        return ClaimLease(
            identity,
            self.token,
            self.clock() + 10_000,
            1,
            "broker-test-1",
            target,
        )

    def renew_exact_window_claim(self, identity, token, lease_seconds):
        raise AssertionError("preflight must not renew")

    def release_exact_window_claim(self, identity, token):
        self.clock.tick()
        if self.fail_release:
            raise RuntimeError("forced")
        self.released += 1

    def capture_exact_window(self, identity, token):
        request = self.clock.tick()
        start = self.clock.tick()
        completion = self.clock.tick()
        return CapturePacket(
            b"png",
            self.capture_identity,
            Rect(0, 0, 644, 484),
            644,
            484,
            request,
            start,
            completion,
            presentation_ns=completion,
            source_sequence=1,
            canonical_pixel_sha256=hashlib.sha256(b"png").hexdigest(),
        )

    def current_cursor(self, identity, token):
        return CursorSample(20, 30, self.clock.tick())

    def targeted_button_down(self, *args, **kwargs):
        self.input_calls += 1
        raise AssertionError("preflight injected input")

    targeted_button_up = targeted_button_down
    release_all_buttons = targeted_button_down


class R4BOperationsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.run = self.root / "reference/runs/experiment-001"
        self.run.mkdir(parents=True)
        executable_hash = ""
        runtime_hashes = {}
        for index, (relative, key) in enumerate(REQUIRED_RUNTIME_FILES.items()):
            path = self.run / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = f"runtime-{index}".encode()
            path.write_bytes(payload)
            if relative == "irisu.exe":
                executable_hash = hashlib.sha256(payload).hexdigest()
            runtime_hashes[key] = hashlib.sha256(payload).hexdigest()
        (self.run / ".irisu-reference-run").write_text(
            "created_by=tools/create-reference-run.sh\n"
            "created_utc=2026-07-23T00:00:00Z\n"
            f"irisu_exe_sha256={executable_hash}\n",
            encoding="utf-8",
        )
        self.attestation = attest_disposable_run(
            self.root,
            self.run,
            expected_experiment_id="experiment-001",
            canonical_runtime_sha256=runtime_hashes,
        )
        # The fake provider is bound only after the immutable fixture exists.

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def preflight(self, provider: Provider):
        provider.attestation = self.attestation
        provider.launch_nonce_sha256 = "d" * 64
        return run_capture_preflight(
            provider,
            IDENTITY,
            self.attestation,
            repo_root=self.root,
            run_dir=self.run,
            launch_nonce_sha256=provider.launch_nonce_sha256,
            wine_prefix_sha256="f" * 64,
            clock_ns=provider.clock,
        )

    def test_atomic_provider_is_capture_capable_but_input_ineligible(self) -> None:
        provider = Provider(Clock(), safe_input=False)
        report = self.preflight(provider)
        self.assertEqual(report["status"], "input_ineligible")
        self.assertIn("atomic_click_only", report["blockers"])
        self.assertEqual(report["input_operations_attempted"], 0)
        self.assertEqual(provider.input_calls, 0)
        self.assertEqual(provider.released, 1)

    def test_complete_provider_is_only_reported_eligible_without_input(self) -> None:
        provider = Provider(Clock(), safe_input=True)
        report = self.preflight(provider)
        self.assertEqual(report["status"], "input_eligible")
        self.assertEqual(report["blockers"], [])
        self.assertEqual(provider.input_calls, 0)

    def test_identity_change_and_release_failure_fail_closed(self) -> None:
        provider = Provider(Clock(), safe_input=True)
        provider.capture_identity = OTHER
        with self.assertRaisesRegex(OperationalError, "identity"):
            self.preflight(provider)
        self.assertEqual(provider.released, 1)

        provider = Provider(Clock(), safe_input=True)
        provider.fail_release = True
        with self.assertRaisesRegex(OperationalError, "release failed"):
            self.preflight(provider)
        self.assertEqual(provider.input_calls, 0)

    def test_runtime_mutation_during_capture_is_rejected(self) -> None:
        provider = Provider(Clock(), safe_input=True)
        original = provider.capture_exact_window

        def mutate(identity, token):
            packet = original(identity, token)
            (self.run / "data/doc/irisu.ini").write_bytes(b"mutated")
            return packet

        provider.capture_exact_window = mutate
        with self.assertRaisesRegex(Exception, "canonical|changed"):
            self.preflight(provider)
        self.assertEqual(provider.released, 1)

    def test_checked_current_provider_result_is_asset_free_and_ineligible(self) -> None:
        report = json.loads(
            (ROOT / "reference/r4b-provider-preflight-20260723.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["status"], "input_ineligible")
        self.assertEqual(report["input_operations_attempted"], 0)
        self.assertTrue(report["capabilities"]["atomic_click_only"])
        self.assertFalse(report["capabilities"]["explicit_button_down"])
        payload = json.dumps(report)
        for forbidden in ("/home/", "claim_token", "pixels_base64", "Irisu syndrome"):
            self.assertNotIn(forbidden, payload)

    def test_r4b_tools_have_stdlib_only_help_paths(self) -> None:
        for name in ("run-r4b-preflight.py", "finalize-r4b-contract.py"):
            result = subprocess.run(
                [sys.executable, "-S", str(ROOT / "tools" / name), "--help"],
                text=True,
                capture_output=True,
                check=False,
            )
            with self.subTest(name=name):
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
