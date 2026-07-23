#!/usr/bin/env python3
"""Finalize deployment-v1 from complete measured R4a evidence."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from irisu_rl.original_game.contracts import (
    ContractError,
    finalize_deployment_contract,
    load_deployment_contract,
    render_toml,
)
from irisu_rl.original_game.evidence import EvidenceError, load_json_document


def publish_noreplace(path: Path, payload: str) -> None:
    if not path.parent.is_dir():
        raise ContractError(f"output parent directory does not exist: {path.parent}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ContractError(f"output already exists: {path}") from exc
        directory_descriptor = os.open(
            path.parent, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a measured deployment contract from complete R4a evidence."
    )
    parser.add_argument("base", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("soak_report", type=Path)
    parser.add_argument("event_stream", type=Path)
    parser.add_argument("threshold_config", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        base = load_deployment_contract(args.base)
        evidence = load_json_document(args.evidence, "measurement evidence")
        soak_report = load_json_document(args.soak_report, "soak report")
        measured = finalize_deployment_contract(
            base,
            evidence,
            soak_report,
            args.event_stream,
            args.threshold_config,
        )
        publish_noreplace(args.output, render_toml(measured))
    except (OSError, ContractError, EvidenceError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
