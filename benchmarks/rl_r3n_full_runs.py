#!/usr/bin/env python3
"""Ten fresh full episodes with the R3M exact wait-dominance controller."""

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
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "python", ROOT / "benchmarks"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rl_r3m_shot_restraint as r3m
from irisu_pointer.shot_necessity import ExactWaitDominanceGate


RUN_ID = "r3n-wait-dominance-full-episodes-20260808-001"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development" / RUN_ID
EPISODE_COUNT = 10
MAX_TICKS = 100_000
SEEDS = tuple(
    int.from_bytes(
        hashlib.sha256(f"{RUN_ID}|full-development|{index}".encode()).digest()[:4],
        "big",
    )
    for index in range(EPISODE_COUNT)
)
TEST_SOURCE = ROOT / "tests/test_r3n_full_runs.py"
ADOPTION_SUMMARY = r3m.DEFAULT_RUN_ROOT / "summary.json"
ADOPTION_VERIFICATION = r3m.DEFAULT_RUN_ROOT / "verification.json"
ACTION_WORD = struct.Struct("<I")


def source_identity() -> dict[str, object]:
    summary = r3m.read_json(ADOPTION_SUMMARY)
    verification = r3m.read_json(ADOPTION_VERIFICATION)
    r3m.verify_self_hash(summary, "R3M adoption summary")
    r3m.verify_self_hash(verification, "R3M adoption verification")
    if summary.get("promising") is not True or verification.get("verified") is not True:
        raise RuntimeError("R3M controller lacks verified adoption evidence")
    return r3m.with_sha(
        {
            "schema": "irisu-r3n-full-run-source-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "r3m_summary_sha256": summary["sha256"],
            "r3m_verification_sha256": verification["sha256"],
            "files": {
                str(path): r3m.sha256_file(path)
                for path in (
                    Path(__file__).resolve(), TEST_SOURCE,
                    Path(r3m.__file__).resolve(), r3m.GATE_SOURCE,
                    r3m.screen.RUNTIME, r3m.screen.BASE_CHECKPOINT,
                    ADOPTION_SUMMARY, ADOPTION_VERIFICATION,
                )
            },
        }
    )


def initialize(run_root: Path) -> dict[str, object]:
    if run_root.exists():
        raise FileExistsError(f"run path already exists: {run_root}")
    identity = source_identity()
    plan = r3m.with_sha(
        {
            "schema": "irisu-r3n-full-run-plan-v1",
            "run_id": run_root.name,
            "development_only": True,
            "sealed_test_allowed": False,
            "source_identity_sha256": identity["sha256"],
            "controller": "verified R3M exact wait-dominance over unchanged frozen-v5",
            "episode_count": EPISODE_COUNT,
            "seeds": list(SEEDS),
            "seed_derivation": "sha256(run_id|full-development|i) first uint32 big-endian",
            "play_until": "GAME_OVER",
            "operational_ceiling_ticks": MAX_TICKS,
            "ceiling_rule": "alive at ceiling is censored, never called a full episode",
            "gate": r3m.GATE_CONFIG.manifest(),
            "selection": "report maximum terminal score across all ten fresh episodes",
        }
    )
    run_root.mkdir(parents=True)
    r3m.write_new(run_root / "source-identity.json", identity)
    r3m.write_new(run_root / "plan.json", plan)
    return plan


def validate(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = r3m.read_json(run_root / "source-identity.json")
    plan = r3m.read_json(run_root / "plan.json")
    r3m.verify_self_hash(identity, "R3N source identity")
    r3m.verify_self_hash(plan, "R3N plan")
    if identity != source_identity() or plan.get("source_identity_sha256") != identity["sha256"]:
        raise RuntimeError("R3N frozen identity differs")
    return identity, plan


def unit_path(run_root: Path, index: int) -> Path:
    return run_root / "units" / f"{index:02d}.json"


def run_unit(run_root: Path, index: int) -> dict[str, object]:
    identity, plan = validate(run_root)
    if not 0 <= index < EPISODE_COUNT:
        raise ValueError("invalid full-run index")
    path = unit_path(run_root, index)
    if path.exists():
        result = r3m.read_json(path)
        r3m.verify_self_hash(result, "R3N unit")
        return result
    seed = SEEDS[index]
    intent = r3m.with_sha(
        {
            "schema": "irisu-r3n-full-run-intent-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "index": index,
            "seed": seed,
            "operational_ceiling_ticks": MAX_TICKS,
        }
    )
    intent_path = path.with_suffix(".intent.json")
    if intent_path.exists():
        if r3m.read_json(intent_path) != intent:
            raise RuntimeError("R3N retry intent differs")
    else:
        r3m.write_new(intent_path, intent)

    core, campaign = r3m.screen._load_external()
    policy = campaign.POLICY_FACTORY()
    policy.reset(seed)
    gate = ExactWaitDominanceGate(
        lambda decision: r3m.screen._primitive_actions(core, decision),
        config=r3m.GATE_CONFIG,
    )
    actions: list[int] = []
    checkpoints: list[dict[str, object]] = []
    receipt_root = hashlib.sha256()
    reasons: Counter[str] = Counter()
    attempted = kept = suppressed = 0
    gauge_auc = 0
    minimum_gauge = 40_000
    started = time.monotonic()
    terminated = truncated = False

    with campaign.IrisuEnv(
        library_path=r3m.screen.RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": MAX_TICKS + r3m.GATE_CONFIG.probe_ticks},
    ) as env:
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("R3N reset seed differs")
        checkpoints.append(r3m.checkpoint(env, observation))
        while int(observation["tick"]) < MAX_TICKS and not (terminated or truncated):
            before = copy.deepcopy(policy)
            decision = policy.predict(observation)
            if getattr(decision, "is_shot", False):
                attempted += 1
                verdict = gate.evaluate(env, observation, before, policy, decision)
                receipt = {
                    "tick": int(observation["tick"]),
                    "source_body_id": decision.source_body_id,
                    "destination_body_id": decision.destination_body_id,
                    "intent": decision.intent.value,
                    **verdict.manifest(),
                }
                receipt_root.update(r3m.canonical_bytes(receipt))
                reasons[verdict.reason] += 1
                if verdict.execute_shot:
                    kept += 1
                else:
                    suppressed += 1
                    policy = before
                    decision = gate.wait_decision(verdict.reason)
            for action in r3m.screen._primitive_actions(core, decision):
                kind = int(action.kind)
                duration = int(action.wait_ticks) if kind == 0 else 1
                duration = min(duration, MAX_TICKS - int(observation["tick"]))
                for _ in range(duration):
                    primitive = core.JOINT.Action.wait(1) if kind == 0 else action
                    actions.append(r3m.encode_action(primitive))
                    observation, _reward, terminated, truncated, _info = env.step(primitive)
                    gauge = int(observation["gauge"])
                    minimum_gauge = min(minimum_gauge, gauge)
                    gauge_auc += gauge
                    tick = int(observation["tick"])
                    if tick % 1_000 == 0:
                        checkpoints.append(r3m.checkpoint(env, observation))
                    if tick % 5_000 == 0:
                        print(
                            json.dumps(
                                {
                                    "index": index, "seed": seed, "tick": tick,
                                    "score": int(observation["score"]),
                                    "gauge": gauge, "kept": kept,
                                    "suppressed": suppressed,
                                    "elapsed_seconds": time.monotonic() - started,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    if terminated or truncated or tick >= MAX_TICKS:
                        break
                if terminated or truncated or int(observation["tick"]) >= MAX_TICKS:
                    break
        final = r3m.checkpoint(env, observation)
        if checkpoints[-1]["tick"] != final["tick"]:
            checkpoints.append(final)
        snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()

    trace = b"".join(ACTION_WORD.pack(word) for word in actions)
    trace_path = run_root / "traces" / f"{index:02d}.u32le"
    r3m.write_bytes_new(trace_path, trace)
    censored = int(final["tick"]) >= MAX_TICKS and not bool(final["terminated"])
    result = r3m.with_sha(
        {
            "schema": "irisu-r3n-full-run-unit-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "intent_sha256": intent["sha256"],
            "index": index,
            "seed": seed,
            "terminal": bool(final["terminated"]),
            "censored": censored,
            "survival_ticks": int(final["tick"]),
            "score": int(final["score"]),
            "clears": int(final["clears"]),
            "final_gauge": int(final["gauge"]),
            "minimum_gauge": minimum_gauge,
            "gauge_auc": gauge_auc,
            "level": int(final["level"]),
            "highest_chain": int(final["highest_chain"]),
            "attempted_shots": attempted,
            "kept_shots": kept,
            "suppressed_shots": suppressed,
            "gate_reasons": dict(sorted(reasons.items())),
            "gate_receipt_count": attempted,
            "gate_receipt_stream_sha256": receipt_root.hexdigest(),
            "trace_file": str(trace_path.relative_to(run_root)),
            "trace_sha256": hashlib.sha256(trace).hexdigest(),
            "action_count": len(actions),
            "checkpoints": checkpoints,
            "final": final,
            "final_snapshot_sha256": snapshot_sha,
            "wall_seconds": time.monotonic() - started,
        }
    )
    r3m.write_new(path, result)
    print(json.dumps({"complete": index, "terminal": result["terminal"], "censored": censored, "score": result["score"], "ticks": result["survival_ticks"]}, sort_keys=True), flush=True)
    return result


def run_shard(run_root: Path, shard: int, shards: int) -> None:
    if not 0 <= shard < shards:
        raise ValueError("invalid shard")
    for index in range(EPISODE_COUNT):
        if index % shards == shard:
            run_unit(run_root, index)


def load_units(run_root: Path) -> list[dict[str, Any]]:
    rows = []
    for index in range(EPISODE_COUNT):
        path = unit_path(run_root, index)
        if not path.exists():
            raise RuntimeError(f"missing full-run unit {index}")
        row = r3m.read_json(path)
        r3m.verify_self_hash(row, "R3N unit")
        rows.append(row)
    return rows


def summarize(run_root: Path) -> dict[str, object]:
    identity, plan = validate(run_root)
    rows = load_units(run_root)
    terminal = [row for row in rows if row["terminal"] and not row["censored"]]
    best = max(rows, key=lambda row: (int(row["score"]), int(row["survival_ticks"])))
    result = r3m.with_sha(
        {
            "schema": "irisu-r3n-full-run-summary-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "unit_sha256s": [row["sha256"] for row in rows],
            "episode_count": len(rows),
            "terminal_count": len(terminal),
            "censored_count": sum(bool(row["censored"]) for row in rows),
            "scores": [row["score"] for row in rows],
            "mean_score": statistics.fmean(int(row["score"]) for row in rows),
            "median_score": statistics.median(int(row["score"]) for row in rows),
            "best": {
                key: best[key]
                for key in (
                    "index", "seed", "score", "survival_ticks", "clears",
                    "final_gauge", "level", "highest_chain", "terminal", "censored",
                    "kept_shots", "suppressed_shots", "sha256",
                )
            },
            "all_full": len(terminal) == EPISODE_COUNT,
        }
    )
    path = run_root / "summary.json"
    if path.exists():
        if r3m.read_json(path) != result:
            raise RuntimeError("R3N existing summary differs")
    else:
        r3m.write_new(path, result)
    return result


def verify(run_root: Path) -> dict[str, object]:
    identity, plan = validate(run_root)
    rows = load_units(run_root)
    core, campaign = r3m.screen._load_external()
    verified = []
    for row in rows:
        trace_path = run_root / str(row["trace_file"])
        trace = trace_path.read_bytes()
        if hashlib.sha256(trace).hexdigest() != row["trace_sha256"]:
            raise RuntimeError("R3N trace hash differs")
        words = [word for (word,) in struct.iter_unpack("<I", trace)]
        expected = {int(item["tick"]): item for item in row["checkpoints"]}
        with campaign.IrisuEnv(
            library_path=r3m.screen.RUNTIME,
            physics_backend="portable",
            config={"max_episode_ticks": MAX_TICKS + r3m.GATE_CONFIG.probe_ticks},
        ) as env:
            observation, _ = env.reset(seed=int(row["seed"]))
            if r3m.checkpoint(env, observation) != expected[0]:
                raise RuntimeError("R3N initial replay differs")
            for word in words:
                observation, _reward, _terminated, _truncated, _info = env.step(r3m.decode_action(core, word))
                tick = int(observation["tick"])
                if tick in expected and r3m.checkpoint(env, observation) != expected[tick]:
                    raise RuntimeError(f"R3N replay differs at tick {tick}")
            final = r3m.checkpoint(env, observation)
            snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
        if final != row["final"] or snapshot_sha != row["final_snapshot_sha256"] or len(words) != row["action_count"]:
            raise RuntimeError("R3N replay final closure differs")
        verified.append(row["sha256"])
    summary = summarize(run_root)
    result = r3m.with_sha(
        {
            "schema": "irisu-r3n-full-run-verification-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "summary_sha256": summary["sha256"],
            "verified_unit_sha256s": verified,
            "verified": True,
        }
    )
    r3m.write_new(run_root / "verification.json", result)
    return result


def status(run_root: Path) -> dict[str, object]:
    validate(run_root)
    completed = [r3m.read_json(unit_path(run_root, index)) for index in range(EPISODE_COUNT) if unit_path(run_root, index).exists()]
    return {
        "run_id": run_root.name,
        "completed": len(completed),
        "terminal": sum(bool(row["terminal"]) for row in completed),
        "censored": sum(bool(row["censored"]) for row in completed),
        "best_score": max((int(row["score"]) for row in completed), default=0),
        "active_intents": EPISODE_COUNT - len(completed),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "run-unit", "run-shard", "summary", "verify", "status"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--index", type=int)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "init":
        value: object = initialize(args.run_root)
    elif args.command == "run-unit":
        if args.index is None:
            parser.error("run-unit requires --index")
        value = run_unit(args.run_root, args.index)
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
