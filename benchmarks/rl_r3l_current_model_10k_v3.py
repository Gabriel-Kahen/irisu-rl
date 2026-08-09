#!/usr/bin/env python3
"""Third current-model exhibition with supported late-phase conditioning.

The v2 attempt reproduced the frozen 10k prefix and then stopped before its
first new action because G4 rejects query indices above three.  This fresh run
uses the same checkpoints and R3K-v3 controller, but holds every post-prefix
query at the last supported public phase (query index 3, shot index 4).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "r3l-current-model-10k-exhibition-20260808-003"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development" / RUN_ID
V2_SOURCE = ROOT / "benchmarks/rl_r3l_current_model_10k_v2.py"
V2_RUN = (
    ROOT
    / "artifacts/r3/development/r3l-current-model-10k-exhibition-20260808-002"
)
TEST_SOURCE = ROOT / "tests/test_r3l_current_model_10k_v3.py"
EXPECTED_V2_SOURCE_ID = (
    "4a0bc077660b8d5311f2181c2e7a95d9093ce6c2739cdbd46ccf2e4451c773e7"
)
EXPECTED_V2_PLAN_ID = (
    "c0d3c4b64ec2999526a2072f5ce76e34900d32f5c34d453fc64c07f1af40dc5c"
)
EXPECTED_V2_FAILURE = "G4 public phase indices exceed frozen bounds"


def _load_v2() -> ModuleType:
    name = "irisu_r3l_exhibition_v2_frozen"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, V2_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen exhibition-v2 helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


V2 = _load_v2()
BASE = V2.BASE


def _effective_phase_index(schedule_index: int) -> int:
    if type(schedule_index) is not int or schedule_index < 0:
        raise ValueError("schedule index must be a nonnegative integer")
    return min(schedule_index, 3)


def _source_identity() -> dict[str, object]:
    current_v2 = V2._source_identity()
    recorded_v2 = BASE._read_json(V2_RUN / "source-identity.json")
    if current_v2 != recorded_v2 or recorded_v2["sha256"] != EXPECTED_V2_SOURCE_ID:
        raise RuntimeError("exhibition-v2 frozen source identity differs")
    v2_plan = BASE._read_json(V2_RUN / "plan.json")
    BASE._verify_self_hash(v2_plan, "exhibition-v2 plan")
    if v2_plan["sha256"] != EXPECTED_V2_PLAN_ID:
        raise RuntimeError("exhibition-v2 plan identity differs")
    log_path = V2_RUN / "campaign.log"
    if EXPECTED_V2_FAILURE not in log_path.read_text():
        raise RuntimeError("exhibition-v2 fail-closed record differs")
    if (V2_RUN / "episode.json").exists() or (V2_RUN / "direct-actions.u32le").exists():
        raise RuntimeError("exhibition-v2 unexpectedly contains a completed episode")
    return BASE._with_sha(
        {
            "schema": "irisu-r3l-exhibition-source-identity-v3",
            "development_only": True,
            "sealed_test_allowed": False,
            "git_head": current_v2["git_head"],
            "inherited_source_identity_sha256": current_v2["sha256"],
            "failed_v2_plan_sha256": v2_plan["sha256"],
            "failed_v2_log_sha256": BASE._sha256_file(log_path),
            "files": {
                str(path): BASE._sha256_file(path)
                for path in (Path(__file__).resolve(), TEST_SOURCE, V2_SOURCE)
            },
        }
    )


def initialize(run_root: Path) -> dict[str, object]:
    if run_root.exists():
        raise FileExistsError(f"exhibition-v3 path already exists: {run_root}")
    identity = _source_identity()
    plan = BASE._with_sha(
        {
            "schema": "irisu-r3l-current-model-10k-plan-v3",
            "run_id": run_root.name,
            "development_only": True,
            "sealed_test_allowed": False,
            "claim_scope": "outcome-selected single-episode exhibition only",
            "source_identity_sha256": identity["sha256"],
            "controller": "unchanged R3K-v3 queried every 2500 ticks",
            "seed": BASE.SEED,
            "target_score": BASE.TARGET_SCORE,
            "maximum_ticks": BASE.MAX_TICKS,
            "query_threshold_ticks": list(V2.QUERY_THRESHOLDS),
            "post_prefix_phase_condition": {
                "query_index": 3,
                "shot_index": 4,
                "reason": "last phase accepted by the frozen R3K-v3 screen",
            },
            "recording": "direct primitive action trace plus diagnostic-renderer MP4",
            "native_rpy_status": (
                "not claimed: current controller fires at frame zero while v2.03 replay "
                "suppresses fresh edges in records zero and one"
            ),
        }
    )
    run_root.mkdir(parents=True)
    BASE._write_new(run_root / "source-identity.json", identity)
    BASE._write_new(run_root / "plan.json", plan)
    BASE._progress(run_root, "initialized-v3", seed=BASE.SEED, target_score=BASE.TARGET_SCORE)
    return plan


def _validate(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    recorded = BASE._read_json(run_root / "source-identity.json")
    if recorded != _source_identity():
        raise RuntimeError("exhibition-v3 frozen source identity differs")
    plan = BASE._read_json(run_root / "plan.json")
    BASE._verify_self_hash(plan, "exhibition-v3 plan")
    if plan.get("source_identity_sha256") != recorded["sha256"]:
        raise RuntimeError("exhibition-v3 plan source binding differs")
    return recorded, plan


def _delegate(name: str, run_root: Path) -> dict[str, object]:
    original_validate = V2._validate
    original_run_id = V2.RUN_ID
    original_query = None
    screen = None
    try:
        V2._validate = _validate
        V2.RUN_ID = RUN_ID
        if name == "run_episode":
            screen = BASE._load_screen()
            original_query = screen._query

            def phase_bounded_query(**kwargs: object) -> tuple[object, dict[str, object]]:
                schedule_index = int(kwargs["query_index"])
                kwargs["query_index"] = _effective_phase_index(schedule_index)
                decision, query = original_query(**kwargs)
                if schedule_index >= 4:
                    query = dict(query)
                    query["exhibition_schedule_index"] = schedule_index
                    query["conditioned_query_index"] = 3
                    query["conditioned_shot_index"] = 4
                return decision, query

            screen._query = phase_bounded_query
        return getattr(V2, name)(run_root)
    finally:
        if screen is not None and original_query is not None:
            screen._query = original_query
        V2._validate = original_validate
        V2.RUN_ID = original_run_id


def run_episode(run_root: Path) -> dict[str, object]:
    return _delegate("run_episode", run_root)


def verify_episode(run_root: Path) -> dict[str, object]:
    return _delegate("verify_episode", run_root)


def render_video(run_root: Path) -> dict[str, object]:
    return _delegate("render_video", run_root)


def status(run_root: Path) -> dict[str, object]:
    return _delegate("status", run_root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "run", "verify", "render", "status", "all"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    args = parser.parse_args()
    try:
        if args.command == "init":
            value = initialize(args.run_root)
        elif args.command == "run":
            value = run_episode(args.run_root)
        elif args.command == "verify":
            value = verify_episode(args.run_root)
        elif args.command == "render":
            value = render_video(args.run_root)
        elif args.command == "status":
            value = status(args.run_root)
        else:
            if not args.run_root.exists():
                initialize(args.run_root)
            run_episode(args.run_root)
            verify_episode(args.run_root)
            value = render_video(args.run_root)
    except Exception as exc:
        parser.error(str(exc))
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
