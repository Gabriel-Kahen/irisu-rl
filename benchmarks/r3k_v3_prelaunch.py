#!/usr/bin/env python3
"""Write-once real-simulator intervention viability gate for R3K v3."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import rl_r3k_sustainable_v3 as campaign_runner


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = (
    ROOT
    / "artifacts/r3/development/"
    "r3k-runway-long-screen-20260808-007-prelaunch"
)
REPORT = OUTPUT_ROOT / "viability-report.json"
COLLISION_PROOF = OUTPUT_ROOT / "seed-collision-proof.json"
SMOKE_COUNT = 16
QUERY_TICK = 2_500


def _derive(index: int) -> int:
    raw = hashlib.sha256(
        f"{campaign_runner.RUN_ID}|viability-smoke|{index}".encode()
    ).digest()
    return int.from_bytes(raw[:4], "big")


SEEDS = tuple(_derive(index) for index in range(SMOKE_COUNT))


def _actions(core: object, decision: object) -> tuple[object, ...]:
    return campaign_runner._primitive_actions(core, decision)


def _advance(
    core: object, env: object, policy: object, observation: dict, target: int
) -> tuple[dict, bool, bool]:
    terminated = truncated = False
    while int(observation["tick"]) < target and not (terminated or truncated):
        decision = policy.predict(observation)
        for action in _actions(core, decision):
            kind = core.JOINT.ActionKind.parse(action.kind)
            duration = int(action.wait_ticks) if kind is core.JOINT.ActionKind.WAIT else 1
            remaining = target - int(observation["tick"])
            if remaining <= 0 or terminated or truncated:
                break
            if duration > remaining:
                if kind is not core.JOINT.ActionKind.WAIT:
                    break
                action = core.JOINT.Action.wait(remaining)
                duration = remaining
            for _ in range(duration):
                primitive = (
                    core.JOINT.Action.wait(1)
                    if kind is core.JOINT.ActionKind.WAIT
                    else action
                )
                observation, _reward, terminated, truncated, _info = env.step(primitive)
                if terminated or truncated or int(observation["tick"]) >= target:
                    break
    return observation, terminated, truncated


def _execute(
    core: object, env: object, observation: dict, decision: object
) -> tuple[dict, bool, bool]:
    terminated = truncated = False
    for action in _actions(core, decision):
        kind = core.JOINT.ActionKind.parse(action.kind)
        duration = int(action.wait_ticks) if kind is core.JOINT.ActionKind.WAIT else 1
        for _ in range(duration):
            primitive = (
                core.JOINT.Action.wait(1)
                if kind is core.JOINT.ActionKind.WAIT
                else action
            )
            observation, _reward, terminated, truncated, _info = env.step(primitive)
            if terminated or truncated:
                break
    return observation, terminated, truncated


def main() -> int:
    if OUTPUT_ROOT.exists():
        raise FileExistsError(f"R3K v3 prelaunch path exists: {OUTPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True)
    development_root = ROOT / "artifacts/r3/development"
    patterns = tuple(str(value) for value in (*SEEDS, *campaign_runner.SEEDS))
    command = ["rg", "-l"]
    for pattern in patterns:
        command.extend(("-F", "-e", pattern))
    command.append(str(development_root))
    collision_run = subprocess.run(command, capture_output=True, text=True, check=False)
    if collision_run.returncode not in {0, 1}:
        raise RuntimeError(f"R3K v3 seed scan failed: {collision_run.stderr.strip()}")
    collisions = sorted(
        str(Path(value).resolve())
        for value in collision_run.stdout.splitlines()
        if value and not str(Path(value).resolve()).startswith(str(OUTPUT_ROOT.resolve()))
    )
    collision_proof = campaign_runner._with_sha(
        {
            "schema": "irisu-r3k-v3-seed-collision-proof-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "scan_root": str(development_root),
            "screen_seeds": list(campaign_runner.SEEDS),
            "viability_seeds": list(SEEDS),
            "patterns": list(patterns),
            "collisions": collisions,
        }
    )
    campaign_runner._write_new(COLLISION_PROOF, collision_proof)
    if collisions:
        raise RuntimeError(f"R3K v3 development seed collision: {collisions}")
    (OUTPUT_ROOT / "queries").mkdir()
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("R3K v3 viability process has no CPU")
    os.sched_setaffinity(0, {allowed[0]})
    core, external = campaign_runner._load_external()
    model = campaign_runner._load_model()
    started = time.monotonic()
    rows: list[dict[str, object]] = []
    unsafe_selected = rebind_failures = restore_failures = 0
    lost_selected = terminal_selected = invalid_selected = 0
    selected_rescue = selected_growth = selected_alternatives = 0
    viable_candidates = strict_survival_rescues = 0
    intervention_seeds: set[int] = set()
    nonzero_clear_targets = 0
    for index, seed in enumerate(SEEDS):
        policy = external.POLICY_FACTORY()
        policy.reset(seed)
        with external.IrisuEnv(
            library_path=campaign_runner.RUNTIME,
            physics_backend="portable",
            config={"max_episode_ticks": QUERY_TICK + campaign_runner.LONG_HORIZON + 64},
        ) as env:
            observation, info = env.reset(seed=seed)
            if int(info.get("seed", -1)) != seed:
                raise RuntimeError("R3K v3 viability reset differs")
            for query_index, target in enumerate((0, QUERY_TICK)):
                if target:
                    observation, terminated, truncated = _advance(
                        core, env, policy, observation, target
                    )
                    if terminated or truncated:
                        break
                incumbent = policy.predict(observation)
                before = env.clone_state()
                before_hash = env.state_hash()
                try:
                    decision, query = campaign_runner._query(
                        run_root=OUTPUT_ROOT,
                        core=core,
                        campaign=external,
                        model=model,
                        env=env,
                        observation=observation,
                        policy=policy,
                        incumbent=incumbent,
                        arm="r3k",
                        seed=seed,
                        index=index,
                        query_index=query_index,
                    )
                except RuntimeError as exc:
                    if "restore" in str(exc).lower():
                        restore_failures += 1
                    raise
                if env.clone_state() != before or env.state_hash() != before_hash:
                    restore_failures += 1
                    raise RuntimeError("R3K v3 viability query changed the live simulator")
                selected = int(query.get("selected_ordinal", 0)) > 0
                selected_alternatives += int(selected)
                rebind_failures += int(selected and query.get("rebind_succeeded") is not True)
                short_rows = query.get("short_outcomes", ())
                long_rows = query.get("long_outcomes", ())
                if not isinstance(short_rows, (list, tuple)) or not isinstance(
                    long_rows, (list, tuple)
                ):
                    raise RuntimeError("R3K v3 viability outcome lists are malformed")
                short_by_ordinal = {
                    int(row["identity"]["ordinal"]): row
                    for row in short_rows
                }
                long_by_ordinal = {
                    int(row["identity"]["ordinal"]): row
                    for row in long_rows
                }
                if len(short_by_ordinal) != len(short_rows) or len(long_by_ordinal) != len(long_rows):
                    raise RuntimeError("R3K v3 viability repeats an outcome identity")
                for ordinal, outcome in long_by_ordinal.items():
                    if ordinal == 0:
                        continue
                    short = short_by_ordinal[ordinal]
                    short_safe = bool(
                        short["candidate_resolved"]
                        and not short["exact_unsafe"]
                        and not short["severe_unsafe"]
                        and short["b2"] is not None
                        and float(short["b2"]) >= 0.0
                    )
                    viable_candidates += int(
                        short_safe
                        and outcome["survival_ticks"] == campaign_runner.LONG_HORIZON
                        and not outcome["terminal"]
                        and not outcome["game_over"]
                        and not outcome["cashflow_lost"]
                        and not outcome["continuation_rebind_failed"]
                        and not outcome["unresolved"]
                        and outcome["invalid_actions"] == 0
                    )
                if selected:
                    intervention_seeds.add(seed)
                    selected_rescue += int(query.get("mode") == "rescue")
                    selected_growth += int(query.get("mode") == "growth")
                    selected_short = next(
                        row
                        for row in query["short_outcomes"]
                        if row["identity"]["ordinal"] == query["selected_ordinal"]
                    )
                    unsafe_selected += int(
                        selected_short["exact_unsafe"]
                        or selected_short["severe_unsafe"]
                        or not selected_short["candidate_resolved"]
                    )
                    selected_long = long_by_ordinal[query["selected_ordinal"]]
                    lost_selected += int(
                        selected_long["cashflow_lost"] or selected_long["game_over"]
                    )
                    terminal_selected += int(selected_long["terminal"])
                    invalid_selected += int(selected_long["invalid_actions"] != 0)
                    incumbent_long = long_by_ordinal[0]
                    strict_survival_rescues += int(
                        query.get("mode") == "rescue"
                        and incumbent_long["survival_ticks"] < campaign_runner.LONG_HORIZON
                        and selected_long["survival_ticks"] == campaign_runner.LONG_HORIZON
                    )
                nonzero_clear_targets += sum(
                    any(checkpoint["clear_gain"] != 0 for checkpoint in outcome["checkpoints"])
                    for outcome in query.get("long_outcomes", ())
                )
                query_artifact = campaign_runner._with_sha(
                    {
                        "schema": "irisu-r3k-v3-viability-query-v1",
                        "development_only": True,
                        "sealed_test_allowed": False,
                        "seed": seed,
                        "index": index,
                        "query_index": query_index,
                        "query": query,
                    }
                )
                query_relative = Path("queries") / f"{index:02d}-{query_index}.json"
                campaign_runner._write_new(OUTPUT_ROOT / query_relative, query_artifact)
                rows.append(
                    {
                        "seed": seed,
                        "query_index": query_index,
                        "tick": int(observation["tick"]),
                        "mode": query.get("mode"),
                        "status": query.get("selection", {}).get("status", query.get("status")),
                        "selected_ordinal": int(query.get("selected_ordinal", 0)),
                        "rebind_succeeded": query.get("rebind_succeeded", True),
                        "short_exact_cost": int(query.get("short_exact_cost", 0)),
                        "long_exact_cost": int(query.get("long_exact_cost", 0)),
                        "query_sha256": campaign_runner._canonical_sha256(query),
                        "query_artifact": str(query_relative),
                        "query_artifact_sha256": query_artifact["sha256"],
                    }
                )
                observation, terminated, truncated = _execute(
                    core, env, observation, decision
                )
                if terminated or truncated:
                    break
    passed = bool(
        len(rows) >= 24
        and len(SEEDS) >= 16
        and viable_candidates >= 16
        and selected_alternatives >= 8
        and selected_rescue >= 2
        and selected_growth >= 4
        and len(intervention_seeds) >= 8
        and strict_survival_rescues >= 2
        and unsafe_selected == 0
        and lost_selected == 0
        and terminal_selected == 0
        and invalid_selected == 0
        and rebind_failures == 0
        and restore_failures == 0
        and nonzero_clear_targets >= 1
        and selected_rescue + selected_growth == selected_alternatives
    )
    report = campaign_runner._with_sha(
        {
            "schema": "irisu-r3k-v3-intervention-viability-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "seeds": list(SEEDS),
            "screen_seeds": list(campaign_runner.SEEDS),
            "seed_sets_disjoint": not bool(set(SEEDS) & set(campaign_runner.SEEDS)),
            "queries": len(rows),
            "complete_units": len(SEEDS),
            "query_artifacts": len(rows),
            "viable_candidates": viable_candidates,
            "selected_alternatives": selected_alternatives,
            "selected_rescue": selected_rescue,
            "selected_growth": selected_growth,
            "unsafe_selected": unsafe_selected,
            "lost_selected": lost_selected,
            "terminal_selected": terminal_selected,
            "invalid_selected": invalid_selected,
            "rebind_failures": rebind_failures,
            "restore_failures": restore_failures,
            "intervention_units": len(intervention_seeds),
            "strict_survival_rescues": strict_survival_rescues,
            "nonzero_clear_targets": nonzero_clear_targets,
            "rows": rows,
            "source_sha256": {
                str(Path(campaign_runner.__file__).resolve()): campaign_runner._sha256_file(Path(campaign_runner.__file__).resolve()),
                str(Path(campaign_runner.r3k.__file__).resolve()): campaign_runner._sha256_file(Path(campaign_runner.r3k.__file__).resolve()),
                str(Path(__file__).resolve()): campaign_runner._sha256_file(Path(__file__).resolve()),
            },
            "dependency_sha256": {
                str(path): campaign_runner._sha256_file(path)
                for path in campaign_runner.EXPECTED
            },
            "seed_collision_proof_sha256": collision_proof["sha256"],
            "wall_seconds": time.monotonic() - started,
            "passed": passed,
        }
    )
    campaign_runner._write_new(REPORT, report)
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
