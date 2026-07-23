#!/usr/bin/env python3
"""Build one fail-closed aggregate R4a soak report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_rl.original_game.evidence import (  # noqa: E402
    EvidenceError,
    PublicationError,
    generate_report,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a tamper-evident safe-event JSONL stream and atomically "
            "publish an aggregate threshold report."
        )
    )
    parser.add_argument("events", type=Path)
    parser.add_argument("thresholds", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args(argv)
    try:
        report = generate_report(
            arguments.events, arguments.thresholds, arguments.output
        )
    except (EvidenceError, PublicationError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")
    print(
        json.dumps(
            {
                "schema": report["schema"],
                "status": report["status"],
                "event_count": report["event_stream"]["count"],
                "chain_head_sha256": report["event_stream"]["chain_head_sha256"],
            },
            sort_keys=True,
        )
    )
    return {"pass": 0, "fail": 2, "not_evaluable": 3}[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
