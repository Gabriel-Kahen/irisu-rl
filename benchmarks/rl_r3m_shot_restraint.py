#!/usr/bin/env python3
"""Matched development screen for the exact wait-dominance shot gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import statistics
import struct
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_r3k_sustainable_v3 as screen
from irisu_pointer.shot_necessity import ExactWaitDominanceGate, WaitDominanceConfig


RUN_ID = "r3m-shot-restraint-screen-20260808-001"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development" / RUN_ID
HORIZON = 10_000
SEEDS = (
    3405020912, 1910994543, 387705732, 3798228772,
    2734710297, 2135580340, 778094569, 3424948582,
    3661315511, 1948608776, 3377435210, 873217288,
    3767418846, 726539816, 373307278, 3417238592,
)
ARMS = ("baseline", "wait-dominance")
GATE_CONFIG = WaitDominanceConfig(probe_ticks=128, wait_ticks=16, gauge_advantage=16)
ACTION_WORD = struct.Struct("<I")
TEST_SOURCE = ROOT / "tests/test_r3m_shot_restraint.py"
GATE_SOURCE = ROOT / "python/irisu_pointer/shot_necessity.py"
REFERENCE_IDENTITY = (
    ROOT / "artifacts/r3/development/r3k-runway-long-screen-20260808-007/source-identity.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def with_sha(value: Mapping[str, object]) -> dict[str, object]:
    result = dict(value)
    result["sha256"] = hashlib.sha256(canonical_bytes(result)).hexdigest()
    return result


def verify_self_hash(value: Mapping[str, object], label: str) -> None:
    supplied = value.get("sha256")
    unsigned = dict(value)
    unsigned.pop("sha256", None)
    if supplied != hashlib.sha256(canonical_bytes(unsigned)).hexdigest():
        raise RuntimeError(f"{label} self-hash differs")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} is not a JSON object")
    return value


def write_new(path: Path, value: Mapping[str, object]) -> None:
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_bytes_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_identity() -> dict[str, object]:
    reference = read_json(REFERENCE_IDENTITY)
    verify_self_hash(reference, "R3K-v3 reference identity")
    return with_sha(
        {
            "schema": "irisu-r3m-shot-restraint-source-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "inherited_r3k_v3_source_sha256": reference["sha256"],
            "files": {
                str(path): sha256_file(path)
                for path in (
                    Path(__file__).resolve(), TEST_SOURCE, GATE_SOURCE,
                    screen.RUNTIME, screen.BASE_CHECKPOINT, screen.CAMPAIGN_SOURCE,
                )
            },
        }
    )


def initialize(run_root: Path) -> dict[str, object]:
    if run_root.exists():
        raise FileExistsError(f"run path already exists: {run_root}")
    identity = source_identity()
    prereg = with_sha(
        {
            "schema": "irisu-r3m-shot-restraint-preregistration-v1",
            "run_id": run_root.name,
            "development_only": True,
            "sealed_test_allowed": False,
            "source_identity_sha256": identity["sha256"],
            "hypothesis": (
                "exact wait-dominance removes cooldown-locked shots without reducing "
                "mean score or survival"
            ),
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "seed_derivation": "sha256(run_id|matched-development|i) first uint32 big-endian",
            "horizon_ticks": HORIZON,
            "gate": GATE_CONFIG.manifest(),
            "model": "unchanged frozen-v5 checkpoint and learned steering geometry",
            "planner_scope": "base learned policy only; sparse R3K query macros excluded",
            "primary_metrics": ["score", "survival_ticks", "clears", "gauge_auc", "shots"],
            "promising_gate": {
                "minimum_shot_reduction_fraction": 0.50,
                "minimum_mean_score_ratio": 1.0,
                "minimum_mean_survival_ratio": 1.0,
            },
        }
    )
    run_root.mkdir(parents=True)
    write_new(run_root / "source-identity.json", identity)
    write_new(run_root / "preregistration.json", prereg)
    return prereg


def validate(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = read_json(run_root / "source-identity.json")
    prereg = read_json(run_root / "preregistration.json")
    verify_self_hash(identity, "source identity")
    verify_self_hash(prereg, "preregistration")
    if identity != source_identity() or prereg.get("source_identity_sha256") != identity["sha256"]:
        raise RuntimeError("frozen shot-restraint identity differs")
    return identity, prereg


def encode_action(action: object) -> int:
    kind = int(action.kind)
    if kind == 0:
        return 0
    x, y = int(action.cursor_x), int(action.cursor_y)
    if not (float(action.cursor_x).is_integer() and float(action.cursor_y).is_integer()):
        raise RuntimeError("shot coordinate is not an integer pixel")
    if not 0 <= x <= 1023 or not 0 <= y <= 511 or kind not in (1, 2, 3):
        raise RuntimeError("shot is not representable as a direct action word")
    return (y << 12) | (x << 2) | kind


def decode_action(core: object, word: int) -> object:
    buttons = word & 3
    x, y = (word >> 2) & 1023, (word >> 12) & 511
    if buttons == 1:
        return core.JOINT.Action.weak(x, y)
    if buttons == 2:
        return core.JOINT.Action.strong(x, y)
    if buttons == 3:
        return core.JOINT.Action.both(x, y)
    return core.JOINT.Action.wait(1)


def checkpoint(env: object, observation: Mapping[str, Any]) -> dict[str, object]:
    return {
        "tick": int(observation["tick"]),
        "score": int(observation["score"]),
        "gauge": int(observation["gauge"]),
        "level": int(observation["level"]),
        "clears": int(observation.get("qualifying_clear_count", 0)),
        "highest_chain": int(observation.get("highest_chain", 0)),
        "terminated": bool(observation.get("terminated", False)),
        "truncated": bool(observation.get("truncated", False)),
        "state_u64": f"0x{int(env.state_hash()) & 0xffffffffffffffff:016x}",
    }


def unit_path(run_root: Path, index: int, arm: str) -> Path:
    return run_root / "units" / f"{index:02d}-{arm}.json"


def run_unit(run_root: Path, index: int, arm: str) -> dict[str, object]:
    identity, prereg = validate(run_root)
    if arm not in ARMS or not 0 <= index < len(SEEDS):
        raise ValueError("invalid matched unit identity")
    result_path = unit_path(run_root, index, arm)
    if result_path.exists():
        result = read_json(result_path)
        verify_self_hash(result, "unit")
        return result
    seed = SEEDS[index]
    intent = with_sha(
        {
            "schema": "irisu-r3m-shot-restraint-intent-v1",
            "source_identity_sha256": identity["sha256"],
            "preregistration_sha256": prereg["sha256"],
            "index": index,
            "arm": arm,
            "seed": seed,
            "horizon_ticks": HORIZON,
        }
    )
    intent_path = result_path.with_suffix(".intent.json")
    if intent_path.exists():
        if read_json(intent_path) != intent:
            raise RuntimeError("unit retry intent differs")
    else:
        write_new(intent_path, intent)

    core, campaign = screen._load_external()
    policy = campaign.POLICY_FACTORY()
    policy.reset(seed)
    gate = ExactWaitDominanceGate(
        lambda decision: screen._primitive_actions(core, decision),
        config=GATE_CONFIG,
    )
    actions: list[int] = []
    checkpoints: list[dict[str, object]] = []
    gate_rows: list[dict[str, object]] = []
    attempted = kept = suppressed = 0
    reasons: Counter[str] = Counter()
    shot_ticks: list[int] = []
    minimum_gauge = 40_000
    gauge_auc = 0
    started = time.monotonic()
    terminated = truncated = False

    with campaign.IrisuEnv(
        library_path=screen.RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": HORIZON + GATE_CONFIG.probe_ticks},
    ) as env:
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("reset seed differs")
        checkpoints.append(checkpoint(env, observation))
        while int(observation["tick"]) < HORIZON and not (terminated or truncated):
            before = copy.deepcopy(policy) if arm == "wait-dominance" else None
            decision = policy.predict(observation)
            if getattr(decision, "is_shot", False):
                attempted += 1
                if arm == "wait-dominance":
                    assert before is not None
                    verdict = gate.evaluate(env, observation, before, policy, decision)
                    reasons[verdict.reason] += 1
                    row = {
                        "tick": int(observation["tick"]),
                        "source_body_id": decision.source_body_id,
                        "destination_body_id": decision.destination_body_id,
                        "intent": decision.intent.value,
                        **verdict.manifest(),
                    }
                    gate_rows.append(row)
                    if verdict.execute_shot:
                        kept += 1
                    else:
                        suppressed += 1
                        policy = before
                        decision = gate.wait_decision(verdict.reason)
                else:
                    kept += 1
            for action in screen._primitive_actions(core, decision):
                kind = int(action.kind)
                duration = int(action.wait_ticks) if kind == 0 else 1
                duration = min(duration, HORIZON - int(observation["tick"]))
                for _ in range(duration):
                    primitive = core.JOINT.Action.wait(1) if kind == 0 else action
                    word = encode_action(primitive)
                    if word & 3:
                        shot_ticks.append(int(observation["tick"]))
                    actions.append(word)
                    observation, _reward, terminated, truncated, _info = env.step(primitive)
                    gauge = int(observation["gauge"])
                    minimum_gauge = min(minimum_gauge, gauge)
                    gauge_auc += gauge
                    if int(observation["tick"]) % 500 == 0:
                        checkpoints.append(checkpoint(env, observation))
                    if terminated or truncated or int(observation["tick"]) >= HORIZON:
                        break
                if terminated or truncated or int(observation["tick"]) >= HORIZON:
                    break
        final = checkpoint(env, observation)
        if checkpoints[-1]["tick"] != final["tick"]:
            checkpoints.append(final)
        final_snapshot_sha256 = hashlib.sha256(env.clone_state()).hexdigest()

    survival = int(final["tick"])
    if survival < HORIZON:
        gauge_auc += HORIZON - survival
    trace = b"".join(ACTION_WORD.pack(word) for word in actions)
    trace_path = run_root / "traces" / f"{index:02d}-{arm}.u32le"
    write_bytes_new(trace_path, trace)
    gaps = [b - a for a, b in zip(shot_ticks, shot_ticks[1:])]
    result = with_sha(
        {
            "schema": "irisu-r3m-shot-restraint-unit-v1",
            "source_identity_sha256": identity["sha256"],
            "preregistration_sha256": prereg["sha256"],
            "intent_sha256": intent["sha256"],
            "index": index,
            "arm": arm,
            "seed": seed,
            "horizon_ticks": HORIZON,
            "survival_ticks": survival,
            "full_survivor": survival == HORIZON and not bool(final["terminated"]),
            "score": int(final["score"]),
            "clears": int(final["clears"]),
            "final_gauge": int(final["gauge"]),
            "minimum_gauge": minimum_gauge,
            "gauge_auc": gauge_auc,
            "level": int(final["level"]),
            "highest_chain": int(final["highest_chain"]),
            "attempted_shots": attempted,
            "executed_shots": len(shot_ticks),
            "kept_shots": kept,
            "suppressed_shots": suppressed,
            "exact_16_tick_gaps": sum(gap == 16 for gap in gaps),
            "shot_gap_count": len(gaps),
            "gate_reasons": dict(sorted(reasons.items())),
            "gate_receipts_sha256": hashlib.sha256(canonical_bytes(gate_rows)).hexdigest(),
            "gate_receipts": gate_rows,
            "trace_file": str(trace_path.relative_to(run_root)),
            "trace_sha256": hashlib.sha256(trace).hexdigest(),
            "action_count": len(actions),
            "checkpoints": checkpoints,
            "final": final,
            "final_snapshot_sha256": final_snapshot_sha256,
            "wall_seconds": time.monotonic() - started,
        }
    )
    write_new(result_path, result)
    print(json.dumps({"complete": str(result_path), "score": result["score"], "survival": survival, "shots": len(shot_ticks)}, sort_keys=True), flush=True)
    return result


def run_shard(run_root: Path, shard: int, shards: int) -> None:
    if not 0 <= shard < shards:
        raise ValueError("invalid shard")
    for index, seed in enumerate(SEEDS):
        if index % shards != shard:
            continue
        order = ARMS if hashlib.sha256(str(seed).encode()).digest()[0] & 1 else ARMS[::-1]
        for arm in order:
            run_unit(run_root, index, arm)


def load_units(run_root: Path) -> list[dict[str, Any]]:
    units = []
    for index in range(len(SEEDS)):
        for arm in ARMS:
            path = unit_path(run_root, index, arm)
            if not path.exists():
                raise RuntimeError(f"missing unit {index}/{arm}")
            value = read_json(path)
            verify_self_hash(value, "unit")
            units.append(value)
    return units


def arm_stats(rows: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    def values(key: str) -> list[float]:
        return [float(row[key]) for row in rows]
    return {
        "count": len(rows),
        "full_survivors": sum(bool(row["full_survivor"]) for row in rows),
        **{
            key: {"mean": statistics.fmean(values(key)), "median": statistics.median(values(key)), "sum": sum(values(key))}
            for key in ("score", "survival_ticks", "clears", "gauge_auc", "executed_shots", "minimum_gauge")
        },
        "exact_16_gap_fraction": (
            sum(int(row["exact_16_tick_gaps"]) for row in rows)
            / max(1, sum(int(row["shot_gap_count"]) for row in rows))
        ),
    }


def summarize(run_root: Path) -> dict[str, object]:
    identity, prereg = validate(run_root)
    units = load_units(run_root)
    by_arm = {arm: [row for row in units if row["arm"] == arm] for arm in ARMS}
    stats = {arm: arm_stats(rows) for arm, rows in by_arm.items()}
    pairs = []
    for index, seed in enumerate(SEEDS):
        base = next(row for row in by_arm["baseline"] if row["index"] == index)
        gated = next(row for row in by_arm["wait-dominance"] if row["index"] == index)
        pairs.append(
            {
                "index": index,
                "seed": seed,
                **{f"delta_{key}": int(gated[key]) - int(base[key]) for key in ("score", "survival_ticks", "clears", "gauge_auc", "executed_shots")},
            }
        )
    baseline = stats["baseline"]
    gated = stats["wait-dominance"]
    shot_reduction = 1.0 - float(gated["executed_shots"]["sum"]) / float(baseline["executed_shots"]["sum"])
    score_ratio = float(gated["score"]["mean"]) / max(1.0, float(baseline["score"]["mean"]))
    survival_ratio = float(gated["survival_ticks"]["mean"]) / max(1.0, float(baseline["survival_ticks"]["mean"]))
    summary = with_sha(
        {
            "schema": "irisu-r3m-shot-restraint-summary-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "source_identity_sha256": identity["sha256"],
            "preregistration_sha256": prereg["sha256"],
            "unit_sha256s": [row["sha256"] for row in units],
            "arms": stats,
            "paired": pairs,
            "paired_wins": {
                key: sum(pair[f"delta_{key}"] > 0 for pair in pairs)
                for key in ("score", "survival_ticks", "clears", "gauge_auc")
            },
            "shot_reduction_fraction": shot_reduction,
            "mean_score_ratio": score_ratio,
            "mean_survival_ratio": survival_ratio,
            "promising": bool(shot_reduction >= 0.5 and score_ratio >= 1.0 and survival_ratio >= 1.0),
        }
    )
    path = run_root / "summary.json"
    if path.exists():
        if read_json(path) != summary:
            raise RuntimeError("existing summary differs")
    else:
        write_new(path, summary)
    return summary


def verify(run_root: Path) -> dict[str, object]:
    identity, prereg = validate(run_root)
    units = load_units(run_root)
    core, campaign = screen._load_external()
    verified = []
    for row in units:
        trace_path = run_root / str(row["trace_file"])
        trace = trace_path.read_bytes()
        if hashlib.sha256(trace).hexdigest() != row["trace_sha256"]:
            raise RuntimeError("trace hash differs")
        words = [word for (word,) in struct.iter_unpack("<I", trace)]
        expected = {int(item["tick"]): item for item in row["checkpoints"]}
        with campaign.IrisuEnv(
            library_path=screen.RUNTIME,
            physics_backend="portable",
            config={"max_episode_ticks": HORIZON + GATE_CONFIG.probe_ticks},
        ) as env:
            observation, _ = env.reset(seed=int(row["seed"]))
            if checkpoint(env, observation) != expected[0]:
                raise RuntimeError("initial replay checkpoint differs")
            for word in words:
                observation, _reward, _terminated, _truncated, _info = env.step(decode_action(core, word))
                tick = int(observation["tick"])
                if tick in expected and checkpoint(env, observation) != expected[tick]:
                    raise RuntimeError(f"replay checkpoint differs at {tick}")
            final = checkpoint(env, observation)
            snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
        if final != row["final"] or snapshot_sha != row["final_snapshot_sha256"] or len(words) != row["action_count"]:
            raise RuntimeError("replay final closure differs")
        verified.append(row["sha256"])
    summary = summarize(run_root)
    result = with_sha(
        {
            "schema": "irisu-r3m-shot-restraint-verification-v1",
            "source_identity_sha256": identity["sha256"],
            "preregistration_sha256": prereg["sha256"],
            "summary_sha256": summary["sha256"],
            "verified_unit_sha256s": verified,
            "verified": True,
        }
    )
    path = run_root / "verification.json"
    if path.exists():
        if read_json(path) != result:
            raise RuntimeError("existing verification differs")
    else:
        write_new(path, result)
    return result


def status(run_root: Path) -> dict[str, object]:
    validate(run_root)
    completed = Counter()
    for arm in ARMS:
        completed[arm] = sum(unit_path(run_root, index, arm).exists() for index in range(len(SEEDS)))
    return {
        "run_id": run_root.name,
        "run_path": str(run_root),
        "completed": dict(completed),
        "total_per_arm": len(SEEDS),
        "summary_exists": (run_root / "summary.json").exists(),
        "verification_exists": (run_root / "verification.json").exists(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "run-unit", "run-shard", "summary", "verify", "status"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--index", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "init":
        value: object = initialize(args.run_root)
    elif args.command == "run-unit":
        if args.index is None or args.arm is None:
            parser.error("run-unit requires --index and --arm")
        value = run_unit(args.run_root, args.index, args.arm)
    elif args.command == "run-shard":
        run_shard(args.run_root, args.shard, args.shards)
        value = status(args.run_root)
    elif args.command == "summary":
        value = summarize(args.run_root)
    elif args.command == "verify":
        value = verify(args.run_root)
    else:
        value = status(args.run_root)
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
