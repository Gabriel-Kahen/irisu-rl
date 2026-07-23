#!/usr/bin/env python3
"""Run an input-free R4b broker/runtime/window preflight."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_rl.original_game.harness import WindowIdentity
from irisu_rl.original_game.live_provider import (
    BrokerHarnessProvider,
    JsonLineBrokerTransport,
)
from irisu_rl.original_game.operations import (
    canonical_report_bytes,
    run_capture_preflight,
)
from irisu_rl.original_game.private_io import (
    publish_private_noreplace,
)
from irisu_rl.original_game.runtime import (
    attest_disposable_run,
    attest_wine_prefix,
    verify_wine_prefix_unchanged,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Claim and capture an exact disposable IriSu window without injecting "
            "input, then publish a private capability report."
        )
    )
    parser.add_argument("experiment_id")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("window_address")
    parser.add_argument("capture_id")
    parser.add_argument("launch_nonce_sha256")
    parser.add_argument("wine_prefix", type=Path)
    parser.add_argument("broker", type=Path)
    parser.add_argument("broker_sha256")
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--output-name", default="r4b-preflight.json")
    args = parser.parse_args()

    transport = JsonLineBrokerTransport(args.broker, expected_sha256=args.broker_sha256)
    provider = None
    try:
        provider = BrokerHarnessProvider(transport)
        attestation = attest_disposable_run(
            ROOT,
            args.run_dir,
            expected_experiment_id=args.experiment_id,
        )
        prefix_attestation = attest_wine_prefix(args.wine_prefix)
        report = run_capture_preflight(
            provider,
            WindowIdentity(args.window_address, args.capture_id),
            attestation,
            repo_root=ROOT,
            run_dir=args.run_dir,
            launch_nonce_sha256=args.launch_nonce_sha256,
            wine_prefix_sha256=prefix_attestation.sha256,
        )
        verify_wine_prefix_unchanged(prefix_attestation, args.wine_prefix)
        publish_private_noreplace(
            args.output_directory,
            args.output_name,
            canonical_report_bytes(report),
        )
        print(report["status"])
    except Exception as exc:
        parser.error(str(exc))
    finally:
        if provider is not None:
            provider.close()
        else:
            transport.close()


if __name__ == "__main__":
    main()
