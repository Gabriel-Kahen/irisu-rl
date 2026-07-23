#!/usr/bin/env python3
"""Finalize deployment-v1 from complete measured R4a evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from irisu_rl.original_game.contracts import (
    ContractError,
    finalize_deployment_contract,
    load_deployment_contract,
    render_toml,
)


def publish_noreplace(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    finally:
        Path(temporary).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a measured deployment contract from complete R4a evidence."
    )
    parser.add_argument("base", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("soak_report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        base = load_deployment_contract(args.base)
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        soak_report = json.loads(args.soak_report.read_text(encoding="utf-8"))
        measured = finalize_deployment_contract(base, evidence, soak_report)
        publish_noreplace(args.output, render_toml(measured))
    except (OSError, json.JSONDecodeError, ContractError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
