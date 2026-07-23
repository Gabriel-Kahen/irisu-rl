#!/usr/bin/env python3
"""Derive R4b evidence from raw journals and publish a review-blocked contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_rl.original_game.calibration import (
    build_deployment_evidence,
    load_calibration_plan,
)
from irisu_rl.original_game.contracts import (
    finalize_deployment_contract,
    load_deployment_contract,
    render_toml,
)
from irisu_rl.original_game.evidence import load_json_document
from irisu_rl.original_game.private_io import (
    publish_private_noreplace,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild deployment evidence from a complete typed calibration journal "
            "and verified soak sources; publish evidence and a measured-pending-review "
            "contract without replacing either output."
        )
    )
    parser.add_argument("base_contract", type=Path)
    parser.add_argument("calibration_plan", type=Path)
    parser.add_argument("calibration_journal", type=Path)
    parser.add_argument("soak_report", type=Path)
    parser.add_argument("soak_events", type=Path)
    parser.add_argument("soak_thresholds", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--evidence-name", default="r4b-deployment-evidence.json")
    parser.add_argument("--contract-name", default="deployment-v1.measured.toml")
    args = parser.parse_args()
    try:
        plan = load_calibration_plan(args.calibration_plan)
        soak_report = load_json_document(args.soak_report, "soak report")
        evidence = build_deployment_evidence(
            plan,
            args.calibration_journal,
            soak_report,
            args.soak_events,
            args.soak_thresholds,
        )
        measured = finalize_deployment_contract(
            load_deployment_contract(args.base_contract),
            evidence,
            soak_report,
            args.soak_events,
            args.soak_thresholds,
        )
        evidence_payload = (
            json.dumps(
                evidence,
                allow_nan=False,
                ensure_ascii=True,
                indent=2,
                sort_keys=True,
            ).encode("ascii")
            + b"\n"
        )
        publish_private_noreplace(
            args.output_directory, args.evidence_name, evidence_payload
        )
        publish_private_noreplace(
            args.output_directory,
            args.contract_name,
            render_toml(measured).encode("utf-8"),
        )
        print(measured["measurement_status"])
    except Exception as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
