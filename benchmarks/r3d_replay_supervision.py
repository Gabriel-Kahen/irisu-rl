#!/usr/bin/env python3
"""Development-only, identity-bound replay steering supervision diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import stat
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from irisu_env import IrisuEnv  # noqa: E402
from irisu_pointer.action import PointerActionSpec  # noqa: E402
from irisu_pointer.replay_supervision import (  # noqa: E402
    ReplayEvidenceIdentity,
    ReplayInputFrame,
    ReplaySteeringCollection,
    collect_replay_steering_supervision,
)
from irisu_pointer.steering_learning import (  # noqa: E402
    GoalConditionedSteeringModel,
    SteeringDataset,
    steering_examples_from_replay,
    steering_imitation_loss,
    train_goal_conditioned_steering,
)
from irisu_rl.encoding import TeacherStateEncoder  # noqa: E402


TRUSTED_REPLAY = (
    ROOT
    / "reference/replays/raw/internet/irisu_00041449_20100725_182435_7.rpy"
)
TRUSTED_EXACT_WORKER = (
    ROOT
    / "artifacts/r3/runtime/main-0c48dba-20260723/"
    "exact-runtime-backup/irisu-exact-worker"
)
INSPECT_RPY_PATH = ROOT / "tools/inspect-rpy.py"
_FORMAT = "irisu-r3d-replay-supervision-diagnostic-v1"
_FORBIDDEN_PATH = re.compile(
    r"(?:^|[/_.-])(?:sealed|test)(?:$|[/_.-])", re.IGNORECASE
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, name: str, *, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve(strict=True)
    normalized = str(resolved).replace("\\", "/")
    if _FORBIDDEN_PATH.search(normalized) or "/artifacts/r3/runs/" in normalized:
        raise ValueError(f"{name} must not reference sealed/test or canonical run data")
    metadata = resolved.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular file")
    if executable and not os.access(resolved, os.X_OK):
        raise ValueError(f"{name} is not executable")
    return resolved


def _git_revision() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_paths() -> tuple[Path, ...]:
    names = (
        "python/irisu_env/__init__.py",
        "python/irisu_env/env.py",
        "python/irisu_env/exact_ipc.py",
        "python/irisu_env/native.py",
        "python/irisu_pointer/__init__.py",
        "python/irisu_pointer/action.py",
        "python/irisu_pointer/policy.py",
        "python/irisu_pointer/replay_supervision.py",
        "python/irisu_pointer/steering.py",
        "python/irisu_pointer/steering_learning.py",
        "python/irisu_rl/__init__.py",
        "python/irisu_rl/actions.py",
        "python/irisu_rl/encoding.py",
        "python/irisu_rl/schema.py",
        "tools/inspect-rpy.py",
        "benchmarks/r3d_replay_supervision.py",
        "pyproject.toml",
        "uv.lock",
    )
    return tuple(ROOT / name for name in names)


def _source_identity(paths: Sequence[Path] | None = None) -> dict[str, object]:
    selected = _source_paths() if paths is None else tuple(paths)
    files = {
        str(path.resolve(strict=True).relative_to(ROOT)): _sha256_file(path)
        for path in selected
    }
    manifest: dict[str, object] = {
        "format": "irisu-r3d-replay-source-v1",
        "git_revision": _git_revision(),
        "files": dict(sorted(files.items())),
    }
    return {**manifest, "sha256": _sha256_bytes(_canonical_json(manifest))}


def _load_replay_parser(path: Path = INSPECT_RPY_PATH) -> ModuleType:
    parser_path = _regular_file(path, "replay parser")
    parser_sha256 = _sha256_file(parser_path)
    name = f"irisu_r3d_inspect_rpy_{parser_sha256[:16]}"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, parser_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the conservative replay parser")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if not callable(getattr(module, "parse_replay", None)):
        raise RuntimeError("conservative replay parser lacks parse_replay")
    return module


def _stable_exact_identity(env: Any, worker_sha256: str) -> dict[str, object]:
    if getattr(env, "physics_backend", None) != "exact":
        raise RuntimeError("diagnostic did not create an exact environment")
    build = dict(env.build_info)
    if build.get("worker_executable_sha256") != worker_sha256:
        raise RuntimeError("live exact worker bytes differ from the requested runtime")
    provenance = dict(env.exact_library_provenance())
    if provenance.get("status") != "captured":
        raise RuntimeError("exact library provenance was not captured")
    if build.get("exact_library_sha256") != provenance.get("sha256"):
        raise RuntimeError("worker and mapped-library identities differ")
    fields = (
        "physics_backend",
        "protocol_version",
        "pointer_bits",
        "body_capacity",
        "config_hash",
        "x87_control_word",
        "worker_backend",
        "worker_compiler",
        "exact_library_sha256",
        "worker_executable_sha256",
    )
    return {
        "build": {name: build[name] for name in fields},
        "mapped_library": provenance,
    }


def _probe_exact_identity(
    env_factory: Callable[..., Any],
    worker: Path,
    worker_sha256: str,
    seed: int,
) -> tuple[int, dict[str, object]]:
    with env_factory(
        physics_backend="exact",
        worker_path=str(worker),
        diagnostic_hashes=False,
    ) as env:
        _, reset_info = env.reset(seed=seed)
        config_hash = int(reset_info.get("config_hash", -1))
        if config_hash < 0 or int(env.config_hash()) != config_hash:
            raise RuntimeError("exact config probe returned an invalid identity")
        identity = _stable_exact_identity(env, worker_sha256)
        if int(identity["build"]["config_hash"]) != config_hash:  # type: ignore[index]
            raise RuntimeError("exact build and reset config identities differ")
        return config_hash, identity


class _ObservedEnvironment:
    """Retain the final public observation without reaching into an exact env."""

    def __init__(self, env: Any) -> None:
        self.env = env
        self.observation: dict[str, Any] | None = None

    def reset(self, *, seed: int):
        observation, info = self.env.reset(seed=seed)
        self.observation = observation
        return observation, info

    def step(self, action: Any):
        transition = self.env.step(action)
        self.observation = transition[0]
        return transition


def _temporal_split(
    examples: Sequence[Any], fraction: float
) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    if not 0.0 < fraction < 1.0:
        raise ValueError("holdout fraction must be in (0, 1)")
    ordered = tuple(
        sorted(
            examples,
            key=lambda value: (
                int(value.observation.source_tick[0]),
                value.episode_identity,
            ),
        )
    )
    if len(ordered) < 2:
        raise ValueError("temporal holdout requires at least two steering examples")
    holdout_count = max(1, min(len(ordered) - 1, round(len(ordered) * fraction)))
    return ordered[:-holdout_count], ordered[-holdout_count:]


@torch.no_grad()
def _evaluate_dataset(
    model: GoalConditionedSteeringModel, dataset: SteeringDataset
) -> dict[str, float | int | None]:
    model.eval()
    device = next(model.parameters()).device
    batch = dataset.as_tensors().to(device)
    output = model(batch.global_features, batch.body_features, batch.body_mask)
    loss = steering_imitation_loss(output, batch)
    width = output.pair_logits.shape[2]
    shot_rows = (batch.act_index == 1).nonzero(as_tuple=False).reshape(-1)
    wait_rows = (batch.act_index == 0).nonzero(as_tuple=False).reshape(-1)
    flat_pair = output.pair_logits[shot_rows].flatten(1).argmax(-1)
    selected = (
        shot_rows,
        batch.source_index[shot_rows],
        batch.destination_index[shot_rows],
    )

    def accuracy(predicted: torch.Tensor, expected: torch.Tensor) -> float:
        return float((predicted == expected).float().mean())

    return {
        "examples": batch.size,
        "loss": float(loss.total),
        "act_accuracy": accuracy(output.act_logits.argmax(-1), batch.act_index),
        "wait_accuracy": (
            accuracy(
                output.wait_logits[wait_rows].argmax(-1),
                batch.wait_index[wait_rows],
            )
            if wait_rows.numel()
            else None
        ),
        "pair_accuracy": float(
            (
                (flat_pair // width == batch.source_index[shot_rows])
                & (flat_pair % width == batch.destination_index[shot_rows])
            )
            .float()
            .mean()
        ),
        "kind_accuracy": accuracy(
            output.kind_logits[selected].argmax(-1), batch.kind_index[shot_rows]
        ),
        "template_accuracy": accuracy(
            output.template_logits[selected].argmax(-1),
            batch.template_index[shot_rows],
        ),
        "intent_accuracy": accuracy(
            output.intent_logits[selected].argmax(-1),
            batch.intent_index[shot_rows],
        ),
    }


def _model_sha256(model: GoalConditionedSteeringModel) -> str:
    digest = hashlib.sha256(_canonical_json(model.manifest()))
    for name, tensor in sorted(model.state_dict().items()):
        owned = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(owned.dtype).encode("ascii"))
        digest.update(_canonical_json(list(owned.shape)))
        digest.update(owned.numpy().tobytes())
    return digest.hexdigest()


def _training_report(
    examples: Sequence[Any],
    *,
    steps: int,
    holdout_fraction: float,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object] | None:
    if steps == 0:
        return None
    train_examples, holdout_examples = _temporal_split(examples, holdout_fraction)
    train = SteeringDataset(train_examples)
    holdout = SteeringDataset(holdout_examples)
    torch.manual_seed(seed)
    model = GoalConditionedSteeringModel(
        train.schema, pointer_spec=train.pointer_spec
    )
    before = _evaluate_dataset(model, holdout)
    fitted = train_goal_conditioned_steering(
        model,
        train,
        steps=steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    after = _evaluate_dataset(model, holdout)
    return {
        "split": {
            "order": "pre-shot-public-observation-tick",
            "train_examples": len(train),
            "holdout_examples": len(holdout),
            "train_tick_range": [
                int(train_examples[0].observation.source_tick[0]),
                int(train_examples[-1].observation.source_tick[0]),
            ],
            "holdout_tick_range": [
                int(holdout_examples[0].observation.source_tick[0]),
                int(holdout_examples[-1].observation.source_tick[0]),
            ],
            "train_dataset_sha256": train.sha256,
            "holdout_dataset_sha256": holdout.sha256,
        },
        "hyperparameters": {
            "steps": steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
        },
        "model": {
            **model.manifest(),
            "architecture_sha256": model.architecture_sha256,
            "state_sha256": _model_sha256(model),
        },
        "training": asdict(fitted),
        "holdout_before": before,
        "holdout_after": after,
    }


def run_diagnostic(
    args: argparse.Namespace,
    *,
    parser_module: Any | None = None,
    env_factory: Callable[..., Any] = IrisuEnv,
    source_paths: Sequence[Path] | None = None,
) -> dict[str, object]:
    started = time.monotonic()
    replay_path = _regular_file(Path(args.replay), "replay")
    worker_path = _regular_file(Path(args.worker), "exact worker", executable=True)
    replay_data = replay_path.read_bytes()
    replay_sha256 = _sha256_bytes(replay_data)
    worker_sha256 = _sha256_file(worker_path)
    source_before = _source_identity(source_paths)
    parser = _load_replay_parser() if parser_module is None else parser_module
    replay = parser.parse_replay(replay_data, args.layout)
    if int(replay.header.mode) != 0:
        raise ValueError("replay supervision requires normal mode")
    if not replay.frames:
        raise ValueError("replay contains no input frames")
    seed = int(replay.header.seed) & 0xFFFFFFFF
    frames = tuple(
        ReplayInputFrame.from_object(frame, index)
        for index, frame in enumerate(replay.frames)
    )
    encoder = TeacherStateEncoder()
    pointer_spec = PointerActionSpec()

    config_hash, exact_identity = _probe_exact_identity(
        env_factory, worker_path, worker_sha256, seed
    )
    evidence_identity = ReplayEvidenceIdentity(
        source_revision=str(source_before["sha256"]),
        replay_sha256=replay_sha256,
        runtime_sha256=worker_sha256,
        config_hash=config_hash,
        observation_schema_sha256=encoder.schema.sha256,
        pointer_spec_sha256=pointer_spec.sha256,
        mapped_runtime_sha256=str(
            exact_identity["mapped_library"]["sha256"]  # type: ignore[index]
        ),
    )
    with env_factory(
        physics_backend="exact",
        worker_path=str(worker_path),
        diagnostic_hashes=False,
    ) as env:
        observed_env = _ObservedEnvironment(env)
        collection = collect_replay_steering_supervision(
            observed_env,
            frames,
            seed=seed,
            identity=evidence_identity,
            pointer_spec=pointer_spec,
        )
        final_observation = observed_env.observation
        if final_observation is None:
            raise RuntimeError("exact replay produced no final observation")
        replay_exact_identity = _stable_exact_identity(env, worker_sha256)
        if int(env.config_hash()) != config_hash:
            raise RuntimeError("replay exact config identity changed")
    if replay_exact_identity != exact_identity:
        raise RuntimeError("probe and replay exact runtime identities differ")
    if collection.metrics.frames != len(frames):
        raise RuntimeError("exact replay did not consume every input record")
    if collection.metrics.final_score != int(replay.header.final_score):
        raise RuntimeError("exact replay did not reproduce the recorded final score")
    if int(final_observation.get("level", -1)) != int(replay.header.highest_level):
        raise RuntimeError("exact replay did not reproduce the recorded highest level")
    if int(final_observation.get("highest_chain", -1)) != int(
        replay.header.highest_chain
    ):
        raise RuntimeError("exact replay did not reproduce the recorded highest chain")

    safe_pairs = tuple(
        (shot, observation)
        for shot, observation in zip(
            collection.shots, collection.shot_observations, strict=True
        )
        if shot.destination_body_id is not None
        and not shot.target_grouped
        and shot.target_lifecycle not in {"confirmed", "rotten", "deleted"}
    )
    safe_collection = ReplaySteeringCollection(
        collection.identity,
        tuple(value[0] for value in safe_pairs),
        tuple(value[1] for value in safe_pairs),
        collection.metrics,
    )
    examples = steering_examples_from_replay(
        safe_collection, encoder=encoder, pointer_spec=pointer_spec
    )
    if not examples:
        raise RuntimeError("replay produced no safe ungrouped steering examples")
    grouped = sum(shot.target_grouped for shot in collection.shots)
    no_destination = sum(
        shot.destination_body_id is None for shot in collection.shots
    )
    unsafe_lifecycle = sum(
        shot.target_lifecycle in {"confirmed", "rotten", "deleted"}
        for shot in collection.shots
    )
    training = _training_report(
        examples,
        steps=args.train_steps,
        holdout_fraction=args.holdout_fraction,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.training_seed,
    )

    if _sha256_bytes(replay_path.read_bytes()) != replay_sha256:
        raise RuntimeError("replay file changed during the diagnostic")
    if _sha256_file(worker_path) != worker_sha256:
        raise RuntimeError("exact worker changed during the diagnostic")
    source_after = _source_identity(source_paths)
    if source_after != source_before:
        raise RuntimeError("bound source changed during the diagnostic")

    return {
        "format": _FORMAT,
        "ok": True,
        "scope": {
            "development_only": True,
            "canonical_evidence": False,
            "sealed_evidence": False,
        },
        "identity": {
            "source": source_before,
            "replay": {
                "path": str(replay_path),
                "sha256": replay_sha256,
                "layout": str(replay.layout),
                "frame_count": len(frames),
                "header": {
                    "seed_signed": int(replay.header.seed),
                    "seed_u32": seed,
                    "highest_level": int(replay.header.highest_level),
                    "final_score": int(replay.header.final_score),
                    "highest_chain": int(replay.header.highest_chain),
                    "mode": int(replay.header.mode),
                },
            },
            "exact_runtime": {
                "path": str(worker_path),
                "worker_sha256": worker_sha256,
                **exact_identity,
            },
            "observation_schema": {
                "version": encoder.schema.version,
                "sha256": encoder.schema.sha256,
            },
            "pointer_action": {
                **pointer_spec.manifest(),
                "sha256": pointer_spec.sha256,
            },
            "evidence": {
                **evidence_identity.manifest(),
                "sha256": evidence_identity.sha256,
            },
        },
        "conversion": {
            "collection_sha256": collection.sha256,
            "safe_collection_sha256": safe_collection.sha256,
            "safe_dataset_sha256": SteeringDataset(examples).sha256,
            "metrics": collection.metrics.manifest(),
            "first_hit_supervision": len(collection.shots),
            "safe_ungrouped_examples": len(examples),
            "excluded_grouped_hits": grouped,
            "excluded_confirmed_rotten_or_deleted_hits": unsafe_lifecycle,
            "hits_without_same_color_destination": no_destination,
            "destination_labels": "nearest-visible-same-color-peer-inference-v1",
            "reproduced_header": {
                "final_score": collection.metrics.final_score,
                "highest_level": int(final_observation["level"]),
                "highest_chain": int(final_observation["highest_chain"]),
            },
        },
        "temporal_holdout": training,
        "execution": {
            "elapsed_seconds": time.monotonic() - started,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=TRUSTED_REPLAY)
    parser.add_argument("--worker", type=Path, default=TRUSTED_EXACT_WORKER)
    parser.add_argument(
        "--layout",
        choices=("padded", "legacy"),
        default="padded",
        help=(
            "explicit replay layout; heuristic auto-detection is intentionally "
            "disabled"
        ),
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=0,
        help="fit a temporal-holdout model when positive (default: conversion only)",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--training-seed", type=int, default=2026072805)
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if (
        args.train_steps < 0
        or args.batch_size < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0.0
        or not 0.0 < args.holdout_fraction < 1.0
        or args.torch_threads < 1
        or not 0 <= args.training_seed <= 0xFFFFFFFF
    ):
        parser.error("training arguments are outside their valid ranges")
    torch.set_num_threads(args.torch_threads)
    try:
        report = run_diagnostic(args)
    except Exception as exc:
        report = {
            "format": _FORMAT,
            "ok": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        json.dump(report, sys.stdout, sort_keys=True, separators=(",", ":"))
        sys.stdout.write("\n")
        return 1
    json.dump(
        report,
        sys.stdout,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
