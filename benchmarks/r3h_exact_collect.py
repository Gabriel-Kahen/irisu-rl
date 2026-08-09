#!/usr/bin/env python3
"""Collect fresh R3H exact labels without changing frozen-v5 live behavior.

This development-only runner reuses Strategy B's identity-bound exact branch
evaluator strictly as a shadow oracle.  Alternatives are evaluated from cloned
states, but ordinal zero (the frozen-v5 incumbent) is always executed live.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"


REPOSITORY = Path("/home/gabe/Documents/irisu")
OUTPUT_ROOT = (
    REPOSITORY
    / "artifacts/r3/development/r3h-resolution-first-20260730/collection"
)
B_ROOT = Path(
    "/home/gabe/.codex/worktrees/a6b0/irisu/artifacts/r3/development/"
    "r3g-distributional-barrier-20260729"
)
B_PATHS = {
    "barrier_core": B_ROOT / "barrier_core.py",
    "campaign": B_ROOT / "campaign.py",
    "campaign_metrics": B_ROOT / "campaign_metrics.py",
}
B_SHA256 = {
    "barrier_core": (
        "b532547fa2e87afe441c8fbc7edaadfdcee48655dc1f71ba56fd279a39953e84"
    ),
    "campaign": (
        "ebb5c0e770fb3722da3c6528a9b26565ead05b16dddcb46f684aa79aad056567"
    ),
    "campaign_metrics": (
        "95df58a0345fa4f80c7fa41eea5b3fff79e70ac1647c05e6157cdc694c880e60"
    ),
}
EXPERIMENT_ID = "r3h-resolution-first-20260730"
SOURCE_REVISION = "de701b36355d5ec582df30f4223aabde7bc537df"
RUNTIME_SHA256 = "4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79"
FROZEN_V5_SHA256 = (
    "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d"
)
JOINT_V2_SHA256 = (
    "dc7009fc18a322eca5dace55b9baf982b6ced26c18517af752aab0f6365d362e"
)
PROTOCOL_SHA256 = (
    "6dfb2ffa3a76cc00447e3dcf889f6209a17ff5e2f4c3382fe0959bbabbd52991"
)
SPLIT_COUNTS = {
    "train-baseline": 24,
    "support-calibration": 16,
    "margin-calibration": 24,
    "offline-screen": 16,
}
HORIZON = 2_000
EXACT_HORIZON = 768
QUERY_STRIDE = 6
MAXIMUM_QUERIES = 4


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _pin_one_cpu() -> int | None:
    get_affinity = getattr(os, "sched_getaffinity", None)
    set_affinity = getattr(os, "sched_setaffinity", None)
    if not callable(get_affinity) or not callable(set_affinity):
        return None
    allowed = sorted(get_affinity(0))
    if not allowed:
        raise RuntimeError("collector has no available logical CPU")
    if len(allowed) != 1:
        set_affinity(0, {allowed[0]})
    pinned = sorted(get_affinity(0))
    if len(pinned) != 1:
        raise RuntimeError("collector could not pin itself to one logical CPU")
    return pinned[0]


def derive_seed(split: str, index: int) -> int:
    if split not in SPLIT_COUNTS:
        raise ValueError(f"forbidden R3H collection split: {split!r}")
    if type(index) is not int or not 0 <= index < SPLIT_COUNTS[split]:
        raise ValueError(f"R3H seed index is outside preregistered split {split!r}")
    return int.from_bytes(
        hashlib.sha256(f"{EXPERIMENT_ID}|{split}|{index}".encode()).digest()[:4],
        "big",
        signed=False,
    )


def _verify_seed_schedule() -> None:
    seen: dict[int, tuple[str, int]] = {}
    for split, count in SPLIT_COUNTS.items():
        for index in range(count):
            seed = derive_seed(split, index)
            previous = seen.get(seed)
            if previous is not None:
                raise RuntimeError(
                    f"fresh R3H seed collision: {(split, index)} and {previous}"
                )
            seen[seed] = (split, index)


def _validate_request(split: str, start: int, count: int) -> None:
    if split not in SPLIT_COUNTS:
        raise ValueError(f"forbidden R3H collection split: {split!r}")
    if type(start) is not int or type(count) is not int:
        raise TypeError("start and count must be integers")
    if start < 0 or count < 1 or start + count > SPLIT_COUNTS[split]:
        raise ValueError(
            f"requested range [{start}, {start + count}) exceeds "
            f"{split!r} capacity {SPLIT_COUNTS[split]}"
        )


def _load_module(name: str) -> ModuleType:
    path, expected = B_PATHS[name], B_SHA256[name]
    if path.resolve() != path or _sha256_file(path) != expected:
        raise RuntimeError(f"frozen Strategy B source identity changed: {path}")
    existing = sys.modules.get(name)
    if existing is not None:
        resolved = Path(getattr(existing, "__file__", "")).resolve()
        if resolved != path or _sha256_file(resolved) != expected:
            raise RuntimeError(f"refusing foreign preloaded module {name!r}")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load frozen Strategy B module {name!r}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if Path(module.__file__).resolve() != path:
        raise RuntimeError(f"frozen module {name!r} resolved to a foreign path")
    return module


def _load_frozen_b() -> tuple[ModuleType, ModuleType, dict[str, object]]:
    for name, path in B_PATHS.items():
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"missing or indirect frozen Strategy B source: {path}")
        if _sha256_file(path) != B_SHA256[name]:
            raise RuntimeError(f"frozen Strategy B source SHA-256 changed: {path}")
    _load_module("campaign_metrics")
    core = _load_module("barrier_core")
    campaign = _load_module("campaign")
    if getattr(campaign, "core", None) is not core:
        raise RuntimeError("frozen campaign resolved a foreign barrier_core")
    identities = core.verify_identities()
    expected = {
        "source_revision": SOURCE_REVISION,
        "protocol_sha256": PROTOCOL_SHA256,
        "runtime_sha256": RUNTIME_SHA256,
        "frozen_v5_sha256": FROZEN_V5_SHA256,
        "joint_v2_sha256": JOINT_V2_SHA256,
    }
    for name, value in expected.items():
        if identities.get(name) != value:
            raise RuntimeError(f"frozen evaluator identity mismatch: {name}")
    if (
        getattr(campaign.POLICY_FACTORY, "artifact_sha256", None)
        != FROZEN_V5_SHA256
    ):
        raise RuntimeError("campaign POLICY_FACTORY is not frozen-v5")
    if core.BarrierConfig().horizon_ticks != EXACT_HORIZON:
        raise RuntimeError("foreign exact branch horizon")
    affinity = (
        sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else []
    )
    if affinity and len(affinity) != 1:
        raise RuntimeError("frozen evaluator is not pinned to one logical CPU")
    if (
        core.torch.get_num_threads() != 1
        or core.torch.get_num_interop_threads() != 1
        or any(
            os.environ.get(name) != "1"
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            )
        )
    ):
        raise RuntimeError("nested numeric runtime is not single-threaded")
    return core, campaign, identities


def _collector_identity(
    *,
    split: str,
    index: int,
    seed: int,
    identities: Mapping[str, object],
) -> dict[str, object]:
    local_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if local_revision != SOURCE_REVISION:
        raise RuntimeError(
            f"local source revision is {local_revision}, expected {SOURCE_REVISION}"
        )
    return {
        "schema": "r3h-exact-collector-identity-v1",
        "development_only": True,
        "sealed_test_allowed": False,
        "experiment_id": EXPERIMENT_ID,
        "source_revision": local_revision,
        "collector_source_sha256": _sha256_file(Path(__file__).resolve()),
        "strategy_b_source_sha256": dict(B_SHA256),
        "strategy_b_identities": dict(identities),
        "split": split,
        "index": index,
        "seed": seed,
        "seed_derivation": (
            f'SHA256("{EXPERIMENT_ID}|<split>|<index>")[:4] big-endian'
        ),
        "horizon": HORIZON,
        "exact_horizon": EXACT_HORIZON,
        "query_stride": QUERY_STRIDE,
        "maximum_queries": MAXIMUM_QUERIES,
        "live_policy": "frozen-v5-incumbent-only",
        "alternatives_executed": False,
    }


def _paths(split: str, seed: int) -> tuple[Path, Path]:
    stem = OUTPUT_ROOT / split / f"{seed:010d}"
    return stem.with_suffix(".json"), stem.with_suffix(".queries.jsonl")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite R3H artifact {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _quarantine_partial(path: Path, *, identity_sha256: str) -> Path:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"cannot quarantine non-regular partial {path}")
    digest = _sha256_file(path)
    destination = (
        OUTPUT_ROOT
        / "_incomplete"
        / path.parent.name
        / f"{path.name}.{identity_sha256[:16]}.{digest[:16]}.{time.time_ns()}"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(path, destination)
    path.unlink()
    return destination


def _load_complete(
    episode_path: Path,
    query_path: Path,
    *,
    identity: Mapping[str, object],
    identity_sha256: str,
) -> dict[str, object] | None:
    if not episode_path.exists() and not query_path.exists():
        return None
    if episode_path.exists() != query_path.exists():
        partial = episode_path if episode_path.exists() else query_path
        if partial == episode_path:
            value = json.loads(partial.read_text())
            if value.get("collector_identity_sha256") != identity_sha256:
                raise RuntimeError("foreign partial R3H episode")
        else:
            rows = [
                json.loads(line)
                for line in partial.read_text().splitlines()
                if line.strip()
            ]
            if any(
                row.get("collector_identity_sha256") != identity_sha256
                for row in rows
            ):
                raise RuntimeError("foreign partial R3H query file")
        _quarantine_partial(partial, identity_sha256=identity_sha256)
        return None
    if not episode_path.is_file() or not query_path.is_file():
        raise RuntimeError("partial or non-regular R3H collection artifact")
    if episode_path.is_symlink() or query_path.is_symlink():
        raise RuntimeError("R3H collection artifacts may not be symlinks")
    value = json.loads(episode_path.read_text())
    if (
        value.get("schema") != "r3h-frozen-v5-shadow-label-episode-v1"
        or value.get("complete") is not True
        or value.get("development_only") is not True
        or value.get("sealed_test_allowed") is not False
        or value.get("collector_identity") != identity
        or value.get("collector_identity_sha256") != identity_sha256
        or value.get("query_file_sha256") != _sha256_file(query_path)
        or int(value.get("executed_alternative_count", -1)) != 0
    ):
        raise RuntimeError(f"foreign or incomplete R3H episode: {episode_path}")
    rows = [
        json.loads(line)
        for line in query_path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != int(value.get("query_rows", -1)):
        raise RuntimeError("R3H query row count differs from episode receipt")
    for query_index, row in enumerate(rows):
        if (
            row.get("schema") != "r3h-exact-shadow-query-v1"
            or row.get("collector_identity_sha256") != identity_sha256
            or int(row.get("query_index", -1)) != query_index
            or int(row.get("executed_ordinal", -1)) != 0
            or row.get("alternatives_executed") is not False
        ):
            raise RuntimeError("foreign R3H shadow-query identity")
    return value


def collect_episode(
    split: str,
    index: int,
    *,
    core: ModuleType | None = None,
    campaign: ModuleType | None = None,
    identities: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if split not in SPLIT_COUNTS:
        raise ValueError(f"forbidden R3H collection split: {split!r}")
    seed = derive_seed(split, index)
    if core is None or campaign is None or identities is None:
        core, campaign, loaded_identities = _load_frozen_b()
        identities = loaded_identities
    identity = _collector_identity(
        split=split,
        index=index,
        seed=seed,
        identities=identities,
    )
    identity_sha256 = _canonical_sha256(identity)
    episode_path, query_path = _paths(split, seed)
    existing = _load_complete(
        episode_path,
        query_path,
        identity=identity,
        identity_sha256=identity_sha256,
    )
    if existing is not None:
        return existing

    policy = campaign.POLICY_FACTORY()
    if getattr(policy, "artifact_sha256", None) != FROZEN_V5_SHA256:
        raise RuntimeError("constructed live policy is not frozen-v5")
    policy.reset(seed)
    evaluator = core.ExactBranchEvaluator(campaign.POLICY_FACTORY)
    query_rows: list[dict[str, object]] = []
    events: Counter[int] = Counter()
    started_wall, started_cpu = time.perf_counter(), time.process_time()
    env_max_ticks = HORIZON + EXACT_HORIZON

    with campaign.IrisuEnv(
        library_path=core.RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": env_max_ticks},
    ) as env:
        if (
            Path(env.library_path).resolve() != core.RUNTIME.resolve()
            or _sha256_file(Path(env.library_path)) != RUNTIME_SHA256
            or env_max_ticks < HORIZON + EXACT_HORIZON
        ):
            raise RuntimeError("foreign or undersized portable environment")
        runner = env.runner_identity_manifest()
        config_hash = int(runner["config_hash"])
        observation, reset_info = env.reset(seed=seed)
        if int(reset_info.get("config_hash", -1)) != config_hash:
            raise RuntimeError("portable reset config identity mismatch")
        start_tick = int(observation["tick"])
        initial_clears = int(observation.get("qualifying_clear_count", 0))
        terminated = bool(observation.get("terminated", False))
        truncated = bool(observation.get("truncated", False))
        seen_shots = decisions = 0

        def step_unit(action: object) -> None:
            nonlocal observation, terminated, truncated
            before_tick = int(observation["tick"])
            observation, _reward, terminated, truncated, info = env.step(action)
            if (
                int(info.get("config_hash", -1)) != config_hash
                or int(observation["tick"]) != before_tick + 1
            ):
                raise RuntimeError("portable unit-step identity failure")
            for event in info.get("events", ()):
                if isinstance(event, Mapping):
                    kind = campaign.event_kind(event)
                    if kind is not None:
                        events[int(kind)] += 1

        while (
            not terminated
            and not truncated
            and int(observation["tick"]) - start_tick < HORIZON
        ):
            decisions += 1
            if decisions > 2_000_000:
                raise RuntimeError("episode exceeded decision budget")
            incumbent = policy.predict(observation)
            if not isinstance(incumbent, campaign.SteeringDecision):
                raise TypeError("frozen-v5 returned a foreign steering decision")
            if incumbent.is_shot:
                seen_shots += 1
                scheduled = (
                    (seen_shots - 1) % QUERY_STRIDE == 0
                    and len(query_rows) < MAXIMUM_QUERIES
                )
                if scheduled:
                    snapshot = env.clone_state()
                    state_hash = env.state_hash()
                    query = evaluator.evaluate(
                        env,
                        observation,
                        incumbent,
                        seed=seed,
                        query_id=(
                            f"{EXPERIMENT_ID}:{split}:{index}:"
                            f"{int(observation['tick'])}:q{len(query_rows)}"
                        ),
                        split=split,
                        live_policy=policy,
                    )
                    if (
                        env.clone_state() != snapshot
                        or env.state_hash() != state_hash
                    ):
                        raise RuntimeError(
                            "exact shadow oracle did not restore live state"
                        )
                    if (
                        query.snapshot_sha256
                        != hashlib.sha256(snapshot).hexdigest()
                        or query.incumbent.ordinal != 0
                    ):
                        raise RuntimeError(
                            "exact query source/incumbent identity changed"
                        )
                    query_rows.append(
                        {
                            "schema": "r3h-exact-shadow-query-v1",
                            "development_only": True,
                            "sealed_test_allowed": False,
                            "collector_identity_sha256": identity_sha256,
                            "split": split,
                            "index": index,
                            "seed": seed,
                            "query_index": len(query_rows),
                            "shot_index": seen_shots,
                            "tick": int(observation["tick"]),
                            "executed_ordinal": 0,
                            "alternatives_executed": False,
                            "live_state_restored_exactly": True,
                            "exact_query": query.manifest(),
                        }
                    )

            # Shadow outcomes are labels only: execute the untouched incumbent.
            for action in incumbent.primitive_actions():
                if terminated or truncated:
                    break
                remaining = HORIZON - (int(observation["tick"]) - start_tick)
                if remaining <= 0:
                    break
                kind = campaign.ActionKind.parse(action.kind)
                duration = (
                    int(action.wait_ticks)
                    if kind is campaign.ActionKind.WAIT
                    else 1
                )
                for _ in range(min(duration, remaining)):
                    step_unit(
                        campaign.Action.wait(1)
                        if kind is campaign.ActionKind.WAIT
                        else action
                    )
                    if terminated or truncated:
                        break

        elapsed = int(observation["tick"]) - start_tick
        if truncated and elapsed < HORIZON:
            raise RuntimeError("portable environment truncated before manual censor")
        survival_ticks = min(elapsed, HORIZON)
        query_payload = b"".join(_canonical_bytes(row) + b"\n" for row in query_rows)
        query_sha256 = hashlib.sha256(query_payload).hexdigest()
        result: dict[str, object] = {
            "schema": "r3h-frozen-v5-shadow-label-episode-v1",
            "complete": True,
            "development_only": True,
            "sealed_test_allowed": False,
            "collector_identity": identity,
            "collector_identity_sha256": identity_sha256,
            "split": split,
            "index": index,
            "seed": seed,
            "horizon": HORIZON,
            "runner_config_max_episode_ticks": env_max_ticks,
            "runner": runner,
            "live_policy": "frozen-v5-incumbent-only",
            "alternatives_executed": False,
            "executed_alternative_count": 0,
            "score": int(observation.get("score", 0)),
            "survival_ticks": survival_ticks,
            "terminal": bool(terminated),
            "final_gauge": int(observation.get("gauge", 0)),
            "final_level": int(observation.get("level", 0)),
            "qualifying_clears": (
                int(observation.get("qualifying_clear_count", 0)) - initial_clears
            ),
            "decisions": decisions,
            "seen_shots": seen_shots,
            "event_counts": {
                str(kind): count for kind, count in sorted(events.items())
            },
            "final_state_sha256": hashlib.sha256(env.clone_state()).hexdigest(),
            "query_file": str(query_path),
            "query_file_sha256": query_sha256,
            "query_rows": len(query_rows),
            "wall_seconds": time.perf_counter() - started_wall,
            "cpu_seconds": time.process_time() - started_cpu,
        }

    _write_new(query_path, query_payload)
    _write_new(
        episode_path,
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False).encode()
        + b"\n",
    )
    return result


def collect(split: str, start: int, count: int) -> dict[str, object]:
    _validate_request(split, start, count)
    _verify_seed_schedule()
    pinned_cpu = _pin_one_cpu()
    core, campaign, identities = _load_frozen_b()
    episodes = [
        collect_episode(
            split,
            index,
            core=core,
            campaign=campaign,
            identities=identities,
        )
        for index in range(start, start + count)
    ]
    return {
        "schema": "r3h-exact-collection-batch-v1",
        "development_only": True,
        "sealed_test_allowed": False,
        "split": split,
        "start": start,
        "count": count,
        "pinned_cpu": pinned_cpu,
        "seeds": [int(value["seed"]) for value in episodes],
        "query_rows": sum(int(value["query_rows"]) for value in episodes),
        "complete": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect fresh R3H shadow labels from the exact B2 oracle."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    collect_parser = subparsers.add_parser(
        "collect", help="collect an immutable preregistered split range"
    )
    collect_parser.add_argument("--split", required=True, choices=tuple(SPLIT_COUNTS))
    collect_parser.add_argument("--start", required=True, type=int)
    collect_parser.add_argument("--count", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "collect":
        raise RuntimeError("unsupported command")
    print(
        json.dumps(
            collect(args.split, args.start, args.count),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
