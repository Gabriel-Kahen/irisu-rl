#!/usr/bin/env python3
"""Second current-model exhibition: continue R3K queries beyond tick 10k.

The first append-only attempt reproduced the frozen 10k unit but the base policy
died at score 7,919.  This fresh, explicitly outcome-selected attempt preserves
that prefix and applies the same unchanged R3K-v3 controller every 2,500 ticks.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

RUN_ID = "r3l-current-model-10k-exhibition-20260808-002"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development" / RUN_ID
BASE_SOURCE = ROOT / "benchmarks/rl_r3l_current_model_10k.py"
BASE_RUN = (
    ROOT
    / "artifacts/r3/development/r3l-current-model-10k-exhibition-20260808-001"
)
BASE_IDENTITY = BASE_RUN / "source-identity.json"
FAILED_EPISODE = BASE_RUN / "episode.json"
TEST_SOURCE = ROOT / "tests/test_r3l_current_model_10k_v2.py"

EXPECTED_BASE_SOURCE_ID = (
    "6a328c58dcfcece9ad2d4c69a0353bcd8b4cab3a2d434ff297aa985c2f9baf8e"
)
EXPECTED_FAILED_EPISODE_ID = (
    "b6de3be18ccba3147a1de5bdc8b81b77238eae52884f40a398dda5a7c0931f55"
)
QUERY_INTERVAL = 2_500
QUERY_THRESHOLDS = tuple(range(0, 60_000, QUERY_INTERVAL))


def _load_base() -> ModuleType:
    name = "irisu_r3l_exhibition_v1_frozen"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, BASE_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen exhibition-v1 helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base()


def _source_identity() -> dict[str, object]:
    inherited = BASE._read_json(BASE_IDENTITY)
    BASE._verify_self_hash(inherited, "exhibition-v1 source identity")
    if inherited["sha256"] != EXPECTED_BASE_SOURCE_ID:
        raise RuntimeError("exhibition-v1 source identity differs")
    if inherited != BASE._source_identity():
        raise RuntimeError("exhibition-v1 frozen inputs differ")
    failure = BASE._read_json(FAILED_EPISODE)
    BASE._verify_self_hash(failure, "exhibition-v1 failed episode")
    if (
        failure["sha256"] != EXPECTED_FAILED_EPISODE_ID
        or failure["success"] is not False
        or failure["final"]["score"] != 7919
        or failure["final"]["tick"] != 21994
    ):
        raise RuntimeError("exhibition-v1 failure evidence differs")
    return BASE._with_sha(
        {
            "schema": "irisu-r3l-exhibition-source-identity-v2",
            "development_only": True,
            "sealed_test_allowed": False,
            "git_head": inherited["git_head"],
            "inherited_source_identity_sha256": inherited["sha256"],
            "failed_attempt_sha256": failure["sha256"],
            "files": {
                str(path): BASE._sha256_file(path)
                for path in (
                    Path(__file__).resolve(),
                    TEST_SOURCE,
                    BASE_SOURCE,
                    BASE_IDENTITY,
                    FAILED_EPISODE,
                )
            },
        }
    )


def initialize(run_root: Path) -> dict[str, object]:
    if run_root.exists():
        raise FileExistsError(f"exhibition-v2 path already exists: {run_root}")
    identity = _source_identity()
    plan = BASE._with_sha(
        {
            "schema": "irisu-r3l-current-model-10k-plan-v2",
            "run_id": run_root.name,
            "development_only": True,
            "sealed_test_allowed": False,
            "claim_scope": "outcome-selected single-episode exhibition only",
            "source_identity_sha256": identity["sha256"],
            "controller": "unchanged R3K-v3 queried every 2500 ticks",
            "seed": BASE.SEED,
            "seed_selection": "best R3K-v3 unit; retained after 7,919-point v1 failure",
            "target_score": BASE.TARGET_SCORE,
            "maximum_ticks": BASE.MAX_TICKS,
            "query_threshold_ticks": list(QUERY_THRESHOLDS),
            "post_prefix_queries": True,
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
    BASE._progress(
        run_root,
        "initialized-v2",
        seed=BASE.SEED,
        target_score=BASE.TARGET_SCORE,
        post_prefix_queries=True,
    )
    return plan


def _validate(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    recorded = BASE._read_json(run_root / "source-identity.json")
    current = _source_identity()
    if recorded != current:
        raise RuntimeError("exhibition-v2 frozen source identity differs")
    plan = BASE._read_json(run_root / "plan.json")
    BASE._verify_self_hash(plan, "exhibition-v2 plan")
    if plan.get("source_identity_sha256") != recorded["sha256"]:
        raise RuntimeError("exhibition-v2 plan source binding differs")
    return recorded, plan


def run_episode(run_root: Path) -> dict[str, object]:
    identity, plan = _validate(run_root)
    result_path = run_root / "episode.json"
    if result_path.exists():
        result = BASE._read_json(result_path)
        BASE._verify_self_hash(result, "exhibition-v2 episode")
        return result
    intent = BASE._with_sha(
        {
            "schema": "irisu-r3l-exhibition-intent-v2",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "seed": BASE.SEED,
            "target_score": BASE.TARGET_SCORE,
            "maximum_ticks": BASE.MAX_TICKS,
            "query_threshold_ticks": list(QUERY_THRESHOLDS),
        }
    )
    intent_path = run_root / "episode-intent.json"
    if intent_path.exists():
        if BASE._read_json(intent_path) != intent:
            raise RuntimeError("exhibition-v2 retry intent differs")
    else:
        BASE._write_new(intent_path, intent)

    screen = BASE._load_screen()
    original_namespace = screen.RUN_ID
    core, campaign = screen._load_external()
    model = screen._load_model()
    policy = campaign.POLICY_FACTORY()
    policy.reset(BASE.SEED)
    words: list[int] = []
    queries: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    crossing: dict[str, object] | None = None
    prefix_sha: str | None = None
    query_cursor = 0
    started = time.monotonic()
    terminated = truncated = False

    with campaign.IrisuEnv(
        library_path=screen.RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": BASE.MAX_TICKS},
    ) as env:
        observation, info = env.reset(seed=BASE.SEED)
        if int(info.get("seed", -1)) != BASE.SEED:
            raise RuntimeError("exhibition-v2 reset seed differs")
        while int(observation["tick"]) < BASE.MAX_TICKS and not (
            terminated or truncated
        ):
            decision = policy.predict(observation)
            if (
                query_cursor < len(QUERY_THRESHOLDS)
                and int(observation["tick"]) >= QUERY_THRESHOLDS[query_cursor]
                and BASE.MAX_TICKS - int(observation["tick"]) >= screen.LONG_HORIZON
            ):
                screen.RUN_ID = (
                    original_namespace if query_cursor < 4 else RUN_ID
                )
                decision, query = screen._query(
                    run_root=run_root,
                    core=core,
                    campaign=campaign,
                    model=model,
                    env=env,
                    observation=observation,
                    policy=policy,
                    incumbent=decision,
                    arm="r3k",
                    seed=BASE.SEED,
                    index=0,
                    query_index=query_cursor,
                )
                queries.append(query)
                query_cursor += 1
            reached_boundary = False
            for action in screen._primitive_actions(core, decision):
                kind = core.JOINT.ActionKind.parse(action.kind)
                duration = (
                    int(action.wait_ticks)
                    if kind is core.JOINT.ActionKind.WAIT
                    else 1
                )
                duration = min(
                    duration, BASE.MAX_TICKS - int(observation["tick"])
                )
                for _ in range(max(0, duration)):
                    primitive = (
                        core.JOINT.Action.wait(1)
                        if kind is core.JOINT.ActionKind.WAIT
                        else action
                    )
                    words.append(BASE.encode_action_word(core, primitive))
                    observation, _reward, terminated, truncated, _info = env.step(
                        primitive
                    )
                    tick = int(observation["tick"])
                    if tick % BASE.CHECKPOINT_INTERVAL == 0:
                        checkpoints.append(BASE._public_checkpoint(env, observation))
                        BASE._progress(
                            run_root,
                            "episode-progress-v2",
                            tick=tick,
                            score=int(observation["score"]),
                            gauge=int(observation["gauge"]),
                            level=int(observation["level"]),
                            queries=len(queries),
                            elapsed_seconds=time.monotonic() - started,
                        )
                    if tick == BASE.PREFIX_TICKS:
                        prefix_sha = BASE._verify_prefix(observation, queries[:4])
                        BASE._progress(
                            run_root,
                            "prefix-reproduced-v2",
                            tick=tick,
                            score=int(observation["score"]),
                            prefix_sha256=prefix_sha,
                        )
                    if (
                        crossing is None
                        and int(observation["score"]) >= BASE.TARGET_SCORE
                    ):
                        crossing = BASE._public_checkpoint(env, observation)
                    if crossing is not None:
                        reached_boundary = True
                    if terminated or truncated or tick >= BASE.MAX_TICKS:
                        break
                if terminated or truncated or int(observation["tick"]) >= BASE.MAX_TICKS:
                    break
            if reached_boundary:
                break
        screen.RUN_ID = original_namespace
        final = BASE._public_checkpoint(env, observation)
        if not checkpoints or checkpoints[-1]["tick"] != final["tick"]:
            checkpoints.append(final)
        snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
        config_u64 = f"0x{int(env.config_hash()) & 0xFFFF_FFFF_FFFF_FFFF:016x}"

    if prefix_sha is None:
        raise RuntimeError("exhibition-v2 ended before reproducing the frozen 10k prefix")
    action_data = b"".join(BASE.ACTION_WORD.pack(word) for word in words)
    action_path = run_root / "direct-actions.u32le"
    BASE._write_bytes_new(action_path, action_data)
    success = bool(crossing is not None and int(final["score"]) >= BASE.TARGET_SCORE)
    result = BASE._with_sha(
        {
            "schema": "irisu-r3l-current-model-10k-episode-v2",
            "development_only": True,
            "sealed_test_allowed": False,
            "claim_scope": "outcome-selected single-episode exhibition only",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "intent_sha256": intent["sha256"],
            "seed": BASE.SEED,
            "target_score": BASE.TARGET_SCORE,
            "success": success,
            "first_crossing": crossing,
            "final": final,
            "query_count": len(queries),
            "queries": queries,
            "prefix_sha256": prefix_sha,
            "action_count": len(words),
            "action_file": action_path.name,
            "action_file_sha256": hashlib.sha256(action_data).hexdigest(),
            "checkpoints": checkpoints,
            "final_snapshot_sha256": snapshot_sha,
            "config_u64": config_u64,
            "wall_seconds": time.monotonic() - started,
        }
    )
    BASE._write_new(result_path, result)
    BASE._progress(
        run_root,
        "episode-complete-v2",
        success=success,
        tick=final["tick"],
        score=final["score"],
        episode_sha256=result["sha256"],
    )
    return result


def _load_actions(run_root: Path, result: Mapping[str, object]) -> tuple[int, ...]:
    path = run_root / str(result["action_file"])
    data = path.read_bytes()
    if (
        hashlib.sha256(data).hexdigest() != result["action_file_sha256"]
        or len(data) != int(result["action_count"]) * BASE.ACTION_WORD.size
    ):
        raise RuntimeError("exhibition-v2 action trace differs")
    return tuple(word for (word,) in struct.iter_unpack("<I", data))


def verify_episode(run_root: Path) -> dict[str, object]:
    identity, plan = _validate(run_root)
    result = BASE._read_json(run_root / "episode.json")
    BASE._verify_self_hash(result, "exhibition-v2 episode")
    words = _load_actions(run_root, result)
    screen = BASE._load_screen()
    core, campaign = screen._load_external()
    expected = {int(row["tick"]): row for row in result["checkpoints"]}
    reproduced = []
    with campaign.IrisuEnv(
        library_path=screen.RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": BASE.MAX_TICKS},
    ) as env:
        observation, _ = env.reset(seed=BASE.SEED)
        for word in words:
            observation, _reward, _terminated, _truncated, _info = env.step(
                BASE.decode_action_word(core, word)
            )
            tick = int(observation["tick"])
            if tick in expected:
                actual = BASE._public_checkpoint(env, observation)
                if actual != expected[tick]:
                    raise RuntimeError(f"exhibition-v2 replay diverged at tick {tick}")
                reproduced.append(tick)
        final = BASE._public_checkpoint(env, observation)
        snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
        config_u64 = f"0x{int(env.config_hash()) & 0xFFFF_FFFF_FFFF_FFFF:016x}"
    if (
        final != result["final"]
        or reproduced != sorted(expected)
        or snapshot_sha != result["final_snapshot_sha256"]
        or config_u64 != result["config_u64"]
        or result["success"] is not True
        or int(final["score"]) < BASE.TARGET_SCORE
        or len(words) != int(final["tick"])
    ):
        raise RuntimeError("exhibition-v2 replay final closure differs")
    verification = BASE._with_sha(
        {
            "schema": "irisu-r3l-current-model-10k-verification-v2",
            "development_only": True,
            "sealed_test_allowed": False,
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "episode_sha256": result["sha256"],
            "action_file_sha256": result["action_file_sha256"],
            "replayed_actions": len(words),
            "reproduced_checkpoints": reproduced,
            "final": final,
            "final_snapshot_sha256": snapshot_sha,
            "verified": True,
        }
    )
    path = run_root / "verification.json"
    if path.exists():
        if BASE._read_json(path) != verification:
            raise RuntimeError("existing exhibition-v2 verification differs")
    else:
        BASE._write_new(path, verification)
    BASE._progress(
        run_root,
        "verification-complete-v2",
        score=final["score"],
        verification_sha256=verification["sha256"],
    )
    return verification


def render_video(run_root: Path) -> dict[str, object]:
    identity, plan = _validate(run_root)
    result = BASE._read_json(run_root / "episode.json")
    verification = BASE._read_json(run_root / "verification.json")
    BASE._verify_self_hash(result, "exhibition-v2 episode")
    BASE._verify_self_hash(verification, "exhibition-v2 verification")
    if verification.get("verified") is not True:
        raise RuntimeError("exhibition-v2 is not replay-verified")
    words = _load_actions(run_root, result)
    output = run_root / "current-model-10000plus.mp4"
    manifest_path = run_root / "video-manifest.json"
    if manifest_path.exists():
        manifest = BASE._read_json(manifest_path)
        BASE._verify_self_hash(manifest, "exhibition-v2 video manifest")
        if BASE._sha256_file(output) != manifest["video_sha256"]:
            raise RuntimeError("exhibition-v2 video differs")
        return manifest
    screen = BASE._load_screen()
    core, campaign = screen._load_external()
    frame_count = 0
    with tempfile.TemporaryDirectory(prefix="irisu-r3l-v2-render-") as temporary:
        frames = Path(temporary)
        with campaign.IrisuEnv(
            library_path=screen.RUNTIME,
            physics_backend="portable",
            config={"max_episode_ticks": BASE.MAX_TICKS},
            render_mode="svg",
        ) as env:
            observation, _ = env.reset(seed=BASE.SEED)
            (frames / f"{frame_count:06d}.svg").write_text(env.render("svg"))
            frame_count += 1
            for word in words:
                observation, _reward, _terminated, _truncated, _info = env.step(
                    BASE.decode_action_word(core, word)
                )
                if int(observation["tick"]) % BASE.VIDEO_SAMPLE_TICKS == 0:
                    (frames / f"{frame_count:06d}.svg").write_text(env.render("svg"))
                    frame_count += 1
        temporary_output = run_root / f".{output.name}.{os.getpid()}.tmp.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-framerate", str(BASE.VIDEO_FPS),
                "-i", str(frames / "%06d.svg"),
                "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(temporary_output),
            ],
            check=True,
            cwd=ROOT,
        )
        os.replace(temporary_output, output)
    manifest = BASE._with_sha(
        {
            "schema": "irisu-r3l-current-model-10k-video-v2",
            "development_only": True,
            "sealed_test_allowed": False,
            "renderer": "asset-free deterministic clone SVG",
            "not_original_game_artwork": True,
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "episode_sha256": result["sha256"],
            "verification_sha256": verification["sha256"],
            "video": output.name,
            "video_sha256": BASE._sha256_file(output),
            "frame_count": frame_count,
            "fps": BASE.VIDEO_FPS,
            "sample_ticks": BASE.VIDEO_SAMPLE_TICKS,
            "duration_seconds": frame_count / BASE.VIDEO_FPS,
        }
    )
    BASE._write_new(manifest_path, manifest)
    BASE._progress(
        run_root,
        "video-complete-v2",
        video=output.name,
        video_sha256=manifest["video_sha256"],
    )
    return manifest


def status(run_root: Path) -> dict[str, object]:
    _validate(run_root)
    output: dict[str, object] = {
        "schema": "irisu-r3l-exhibition-status-v2",
        "run_id": run_root.name,
        "run_path": str(run_root),
        "episode_exists": (run_root / "episode.json").exists(),
        "verification_exists": (run_root / "verification.json").exists(),
        "video_exists": (run_root / "current-model-10000plus.mp4").exists(),
        "progress": BASE._read_json(run_root / "progress.json"),
    }
    if output["episode_exists"]:
        episode = BASE._read_json(run_root / "episode.json")
        output["success"] = episode["success"]
        output["final"] = episode["final"]
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("init", "run", "verify", "render", "status", "all")
    )
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
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
