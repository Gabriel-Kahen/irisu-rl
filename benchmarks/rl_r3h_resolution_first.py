#!/usr/bin/env python3
"""Train and audit the development-only R3H resolution-first student."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

import torch  # noqa: E402

from irisu_pointer.resolution_first import (  # noqa: E402
    ResolutionDataset,
    SelectiveCalibration,
    SupportCalibration,
    branch_records,
    fit_selective_calibration,
    fit_support_calibration,
    load_checkpoint,
    pilot_report,
    predict_records,
    save_checkpoint,
    select_candidates,
    train_resolution_first,
    viability_report,
)

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    if torch.get_num_interop_threads() != 1:
        raise


EXPERIMENT_ID = "r3h-resolution-first-20260730"
EXPERIMENT = ROOT / "artifacts/r3/development" / EXPERIMENT_ID
PROTOCOL = EXPERIMENT / "protocol.md"
COLLECTION = EXPERIMENT / "collection"
MODEL = EXPERIMENT / "model"
SEED_NAMESPACE = EXPERIMENT_ID
SOURCE_REVISION = "de701b36355d5ec582df30f4223aabde7bc537df"
SPLITS = {
    "train-baseline": 24,
    "support-calibration": 16,
    "margin-calibration": 24,
    "offline-screen": 16,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seeds(split: str, count: int) -> tuple[int, ...]:
    return tuple(
        int.from_bytes(
            hashlib.sha256(
                f"{SEED_NAMESPACE}|{split}|{index}".encode()
            ).digest()[:4],
            "big",
        )
        for index in range(count)
    )


def _jsonable(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def write_once(path: Path, value: object) -> str:
    encoded = (
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != encoded:
            raise FileExistsError(f"refusing to rewrite {path}")
        return sha256_file(path)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return sha256_file(path)


def source_identity() -> dict[str, object]:
    import irisu_pointer.resolution_first as learner

    collector = ROOT / "benchmarks/r3h_exact_collect.py"
    files = {
        "campaign": Path(__file__).resolve(),
        "learner": Path(learner.__file__).resolve(),
        "protocol": PROTOCOL.resolve(),
    }
    if collector.exists():
        files["collector"] = collector.resolve()
    return {
        "schema": "irisu-r3h-source-identity-v1",
        "experiment_id": EXPERIMENT_ID,
        "source_revision": SOURCE_REVISION,
        "files": {
            name: {"path": str(path), "sha256": sha256_file(path)}
            for name, path in files.items()
        },
    }


def _current_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_source_identity() -> str:
    path = EXPERIMENT / "source-identity.json"
    if not path.exists():
        raise RuntimeError("R3H source identity has not been materialized")
    if _current_revision() != SOURCE_REVISION:
        raise RuntimeError("R3H repository revision changed")
    recorded = json.loads(path.read_text())
    current = source_identity()
    if recorded != current:
        raise RuntimeError("R3H source bytes changed after preflight")
    return sha256_file(path)


def write_manifests() -> dict[str, str]:
    hashes = {}
    for split, count in SPLITS.items():
        manifest = {
            "schema": "irisu-r3h-seed-manifest-v1",
            "experiment_id": EXPERIMENT_ID,
            "derivation": (
                f'SHA256("{SEED_NAMESPACE}|<split>|<index>")[:4] big-endian'
            ),
            "split": split,
            "rows": [
                {"index": index, "seed": seed}
                for index, seed in enumerate(derive_seeds(split, count))
            ],
        }
        hashes[split] = write_once(
            EXPERIMENT / "seed-manifests" / f"{split}.json", manifest
        )
    seeds = [
        seed
        for split, count in SPLITS.items()
        for seed in derive_seeds(split, count)
    ]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("R3H seed collision")
    return hashes


def _query_files(split: str) -> list[Path]:
    if split not in SPLITS:
        raise ValueError(f"unknown R3H split {split!r}")
    expected = set(derive_seeds(split, SPLITS[split]))
    files = sorted((COLLECTION / split).glob("*.queries.jsonl"))
    actual = {int(path.name.split(".", 1)[0]) for path in files}
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"incomplete {split} collection: missing={missing}, extra={extra}"
        )
    return files


def load_queries(split: str) -> tuple[dict[str, Any], ...]:
    output: list[dict[str, Any]] = []
    for path in _query_files(split):
        for line in path.read_text().splitlines():
            if line.strip():
                value = json.loads(line)
                query = value.get("exact_query", value)
                if (
                    not isinstance(query, dict)
                    or query.get("split") != split
                    or int(query.get("seed", -1))
                    != int(path.name.split(".", 1)[0])
                ):
                    raise RuntimeError(f"foreign exact query in {path}")
                output.append(value)
    if not output:
        raise RuntimeError(f"{split} has no exact queries")
    return tuple(output)


def load_dataset(split: str) -> ResolutionDataset:
    dataset = ResolutionDataset(branch_records(load_queries(split)))
    expected = set(derive_seeds(split, SPLITS[split]))
    if set(dataset.seeds) != expected:
        raise RuntimeError(f"{split} contains a seed with no exact query")
    return dataset


def run_preflight() -> dict[str, object]:
    revision = _current_revision()
    if revision != SOURCE_REVISION:
        raise RuntimeError("R3H repository revision mismatch")
    affinity = sorted(os.sched_getaffinity(0))
    if len(affinity) != 1:
        raise RuntimeError("R3H process must be pinned to one logical CPU")
    manifests = write_manifests()
    identity = source_identity()
    identity_hash = write_once(EXPERIMENT / "source-identity.json", identity)
    return {
        "schema": "irisu-r3h-preflight-v1",
        "development_only": True,
        "sealed_test_allowed": False,
        "source_identity_sha256": identity_hash,
        "seed_manifest_sha256": manifests,
        "logical_cpu": affinity[0],
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def command_preflight(_args: argparse.Namespace) -> None:
    report = run_preflight()
    report["sha256"] = write_once(EXPERIMENT / "preflight.json", report)
    print(json.dumps(report, sort_keys=True))


def command_pilot(_args: argparse.Namespace) -> None:
    identity_sha = verify_source_identity()
    report = pilot_report(load_dataset("train-baseline").records)
    report["source_identity_sha256"] = identity_sha
    report["sha256"] = write_once(EXPERIMENT / "pilot.json", report)
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


def _require_pilot() -> dict[str, Any]:
    verify_source_identity()
    path = EXPERIMENT / "pilot.json"
    if not path.exists():
        raise RuntimeError("R3H pilot has not run")
    value = json.loads(path.read_text())
    if value.get("passed") is not True:
        raise RuntimeError("R3H pilot is terminal NO-GO")
    return value


def command_train(_args: argparse.Namespace) -> None:
    _require_pilot()
    dataset = load_dataset("train-baseline")
    model = train_resolution_first(dataset)
    checkpoint = MODEL / "resolution-first.pt"
    checkpoint_sha = save_checkpoint(
        checkpoint,
        model,
        metadata={
            "experiment_id": EXPERIMENT_ID,
            "dataset_sha256": dataset.sha256,
            "source_identity_sha256": sha256_file(
                EXPERIMENT / "source-identity.json"
            ),
        },
    )
    predictions = predict_records(model, dataset)
    summary = {
        "schema": "irisu-r3h-training-v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "dataset_sha256": dataset.sha256,
        "records": len(dataset.records),
        "seeds": len(dataset.seeds),
        "model_manifest": model.manifest(),
        "training_resolution_accuracy": sum(
            (float(row["resolution_mean"]) >= 0.5)
            == bool(row["exact_finite_pair"])
            for row in predictions
        )
        / len(predictions),
    }
    summary["sha256"] = write_once(MODEL / "training.json", summary)
    print(json.dumps(summary, sort_keys=True))


def _load_model():
    identity_sha = verify_source_identity()
    path = MODEL / "resolution-first.pt"
    model, metadata = load_checkpoint(path)
    if metadata.get("source_identity_sha256") != identity_sha:
        raise RuntimeError("R3H model source identity mismatch")
    return model, sha256_file(path)


def command_calibrate(_args: argparse.Namespace) -> None:
    model, checkpoint_sha = _load_model()
    support_dataset = load_dataset("support-calibration")
    support_predictions = predict_records(model, support_dataset)
    support = fit_support_calibration(support_predictions)
    margin_dataset = load_dataset("margin-calibration")
    margin_predictions = predict_records(model, margin_dataset)
    selective = fit_selective_calibration(
        margin_predictions,
        support,
        alpha=model.config.conformal_alpha,
        required_episodes=SPLITS["margin-calibration"],
    )
    report = {
        "schema": "irisu-r3h-selective-calibration-v1",
        "checkpoint_sha256": checkpoint_sha,
        "support_dataset_sha256": support_dataset.sha256,
        "margin_dataset_sha256": margin_dataset.sha256,
        "support": support.manifest(),
        "support_sha256": support.sha256,
        "selective": selective.manifest(),
        "selective_sha256": selective.sha256,
        "passed": math.isfinite(selective.q),
    }
    calibration_path = MODEL / "calibration.json"
    calibration_sha = write_once(calibration_path, report)
    receipt = {
        "schema": "irisu-r3h-calibration-receipt-v1",
        "artifact": str(calibration_path),
        "artifact_sha256": calibration_sha,
        "checkpoint_sha256": checkpoint_sha,
        "support_sha256": support.sha256,
        "selective_sha256": selective.sha256,
        "source_identity_sha256": verify_source_identity(),
    }
    report["receipt_sha256"] = write_once(
        MODEL / "calibration.receipt.json", receipt
    )
    print(json.dumps(_jsonable(report), sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


def _calibrations(
    checkpoint_sha256: str,
) -> tuple[SupportCalibration, SelectiveCalibration]:
    verify_source_identity()
    path = MODEL / "calibration.json"
    receipt_path = MODEL / "calibration.receipt.json"
    raw = json.loads(path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("schema") != "irisu-r3h-calibration-receipt-v1"
        or receipt.get("artifact") != str(path)
        or receipt.get("artifact_sha256") != sha256_file(path)
        or receipt.get("checkpoint_sha256") != checkpoint_sha256
        or raw.get("checkpoint_sha256") != checkpoint_sha256
        or receipt.get("source_identity_sha256") != verify_source_identity()
    ):
        raise RuntimeError("R3H calibration receipt identity mismatch")
    support_raw = dict(raw["support"])
    support_raw["grid"] = tuple(support_raw["grid"])
    support = SupportCalibration(**support_raw)
    selective_raw = dict(raw["selective"])
    if selective_raw["q"] == "Infinity":
        selective_raw["q"] = math.inf
    selective_raw["episode_residuals"] = tuple(
        (int(seed), float(value))
        for seed, value in selective_raw["episode_residuals"]
    )
    selective = SelectiveCalibration(**selective_raw)
    if (
        raw.get("support_sha256") != support.sha256
        or raw.get("selective_sha256") != selective.sha256
        or receipt.get("support_sha256") != support.sha256
        or receipt.get("selective_sha256") != selective.sha256
    ):
        raise RuntimeError("R3H calibration payload hash mismatch")
    return support, selective


def command_screen(_args: argparse.Namespace) -> None:
    model, checkpoint_sha = _load_model()
    support, selective = _calibrations(checkpoint_sha)
    dataset = load_dataset("offline-screen")
    predictions = predict_records(model, dataset)
    selections = select_candidates(predictions, support, selective)
    report = viability_report(selections, selective)
    report.update(
        {
            "checkpoint_sha256": checkpoint_sha,
            "dataset_sha256": dataset.sha256,
            "support_calibration_sha256": support.sha256,
            "selective_calibration_sha256": selective.sha256,
            "selections": [
                {
                    **asdict(selection),
                    "reasons": list(selection.reasons),
                }
                for selection in selections
            ],
        }
    )
    report["sha256"] = write_once(EXPERIMENT / "offline-screen.json", report)
    print(json.dumps(report, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(2)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    subparsers = value.add_subparsers(dest="command", required=True)
    for name, function in (
        ("preflight", command_preflight),
        ("pilot", command_pilot),
        ("train", command_train),
        ("calibrate", command_calibrate),
        ("screen", command_screen),
    ):
        subparser = subparsers.add_parser(name)
        subparser.set_defaults(function=function)
    return value


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
