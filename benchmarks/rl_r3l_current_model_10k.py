#!/usr/bin/env python3
"""Record one outcome-selected, development-only current-model exhibition.

This wrapper reproduces the strongest completed R3K-v3 development unit through
tick 10,000, then continues the same frozen base policy without adding queries.
Every live primitive input is retained for independent replay and video export.
It is an exhibition, not validation or evidence of generalization.
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

RUN_ID = "r3l-current-model-10k-exhibition-20260808-001"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development" / RUN_ID
SCREEN_SOURCE = ROOT / "benchmarks/rl_r3k_sustainable_v3.py"
SCREEN_IDENTITY = (
    ROOT
    / "artifacts/r3/development/r3k-runway-long-screen-20260808-007/"
    "source-identity.json"
)
REFERENCE_UNIT = (
    ROOT
    / "artifacts/r3/development/r3k-runway-long-screen-20260808-007/"
    "units/00-r3k.json"
)
TEST_SOURCE = ROOT / "tests/test_r3l_current_model_10k.py"

EXPECTED_HEAD = "de701b36355d5ec582df30f4223aabde7bc537df"
EXPECTED_SCREEN_ID = (
    "999704126ba89cbf61647799b381455267c0a7785a6c2d1ee1300470c3f4206d"
)
EXPECTED_REFERENCE_ID = (
    "e4acd0c1dd6a09f435e5be47aa6affd3a628d55a5bfdb6ec691f3b4bf01d8abc"
)
SEED = 2_448_721_699
TARGET_SCORE = 10_000
MAX_TICKS = 60_000
PREFIX_TICKS = 10_000
CHECKPOINT_INTERVAL = 500
VIDEO_SAMPLE_TICKS = 5
VIDEO_FPS = 10
ACTION_WORD = struct.Struct("<I")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise RuntimeError(f"non-object JSON artifact: {path}")
    return value


def _with_sha(value: Mapping[str, object]) -> dict[str, object]:
    payload = dict(value)
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def _verify_self_hash(value: Mapping[str, object], label: str) -> None:
    payload = dict(value)
    recorded = payload.pop("sha256", None)
    if recorded != _canonical_sha256(payload):
        raise RuntimeError(f"{label} self-hash differs")


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    _write_bytes_new(path, _canonical_bytes(value) + b"\n")


def _write_bytes_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_operational(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
    os.replace(temporary, path)


def _progress(run_root: Path, event: str, **values: object) -> None:
    row = {
        "schema": "irisu-r3l-exhibition-progress-v1",
        "time": time.time(),
        "event": event,
        **values,
    }
    _write_operational(run_root / "progress.json", row)
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)


def _load_screen() -> ModuleType:
    name = "irisu_r3k_frozen_screen_for_exhibition"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, SCREEN_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen R3K-v3 screen")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if module.RUN_ID != "r3k-runway-long-screen-20260808-007":
        raise RuntimeError("frozen R3K-v3 query namespace differs")
    return module


def _source_identity() -> dict[str, object]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    if head != EXPECTED_HEAD:
        raise RuntimeError("exhibition source revision differs")
    inherited = _read_json(SCREEN_IDENTITY)
    _verify_self_hash(inherited, "inherited R3K-v3 source identity")
    if inherited["sha256"] != EXPECTED_SCREEN_ID:
        raise RuntimeError("inherited R3K-v3 source identity differs")
    for raw_path, digest in inherited["files"].items():
        path = Path(raw_path)
        if not path.is_file() or _sha256_file(path) != digest:
            raise RuntimeError(f"inherited R3K-v3 input differs: {path}")
    reference = _read_json(REFERENCE_UNIT)
    _verify_self_hash(reference, "reference R3K-v3 unit")
    if (
        reference["sha256"] != EXPECTED_REFERENCE_ID
        or reference["seed"] != SEED
        or reference["score"] != 3346
        or reference["survival_ticks"] != PREFIX_TICKS
    ):
        raise RuntimeError("reference R3K-v3 unit differs")
    files = {
        str(path): _sha256_file(path)
        for path in (
            Path(__file__).resolve(),
            TEST_SOURCE,
            SCREEN_SOURCE,
            SCREEN_IDENTITY,
            REFERENCE_UNIT,
        )
    }
    return _with_sha(
        {
            "schema": "irisu-r3l-exhibition-source-identity-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "git_head": head,
            "inherited_source_identity_sha256": inherited["sha256"],
            "files": files,
        }
    )


def initialize(run_root: Path) -> dict[str, object]:
    if run_root.exists():
        raise FileExistsError(f"exhibition path already exists: {run_root}")
    identity = _source_identity()
    plan = _with_sha(
        {
            "schema": "irisu-r3l-current-model-10k-plan-v1",
            "run_id": run_root.name,
            "development_only": True,
            "sealed_test_allowed": False,
            "claim_scope": "outcome-selected single-episode exhibition only",
            "source_identity_sha256": identity["sha256"],
            "controller": "unchanged R3K-v3 schedule through 10k, then frozen-v5 base",
            "seed": SEED,
            "seed_selection": "highest completed R3K-v3 development score",
            "target_score": TARGET_SCORE,
            "maximum_ticks": MAX_TICKS,
            "prefix_ticks": PREFIX_TICKS,
            "query_threshold_ticks": [0, 2500, 5000, 7500],
            "post_prefix_queries": False,
            "recording": "direct primitive action trace plus diagnostic-renderer MP4",
            "native_rpy_status": (
                "not claimed: current controller fires at frame zero while v2.03 replay "
                "suppresses fresh edges in records zero and one"
            ),
        }
    )
    run_root.mkdir(parents=True)
    _write_new(run_root / "source-identity.json", identity)
    _write_new(run_root / "plan.json", plan)
    _progress(run_root, "initialized", seed=SEED, target_score=TARGET_SCORE)
    return plan


def _validate(run_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    recorded = _read_json(run_root / "source-identity.json")
    current = _source_identity()
    if recorded != current:
        raise RuntimeError("exhibition frozen source identity differs")
    plan = _read_json(run_root / "plan.json")
    _verify_self_hash(plan, "exhibition plan")
    if plan.get("source_identity_sha256") != recorded["sha256"]:
        raise RuntimeError("exhibition plan source binding differs")
    return recorded, plan


def encode_action_word(core: ModuleType, action: object) -> int:
    kind = core.JOINT.ActionKind.parse(action.kind)
    if kind is core.JOINT.ActionKind.WAIT:
        return 0
    x_float, y_float = float(action.cursor_x), float(action.cursor_y)
    if not x_float.is_integer() or not y_float.is_integer():
        raise RuntimeError("current-model shot is not representable at integer pixels")
    x, y = int(x_float), int(y_float)
    if not 0 <= x <= 1023 or not 0 <= y <= 511:
        raise RuntimeError("current-model shot exceeds replay coordinate fields")
    if kind is core.JOINT.ActionKind.WEAK_SHOT:
        buttons = 1
    elif kind is core.JOINT.ActionKind.STRONG_SHOT:
        buttons = 2
    elif kind is core.JOINT.ActionKind.BOTH_SHOTS:
        buttons = 3
    else:
        raise RuntimeError("current-model action kind is not recordable")
    return (y << 12) | (x << 2) | buttons


def decode_action_word(core: ModuleType, word: int) -> object:
    if type(word) is not int or not 0 <= word <= 0xFFFF_FFFF:
        raise ValueError("action word must fit uint32")
    buttons = word & 3
    x, y = (word >> 2) & 0x3FF, (word >> 12) & 0x1FF
    if buttons == 1:
        return core.JOINT.Action.weak(x, y)
    if buttons == 2:
        return core.JOINT.Action.strong(x, y)
    if buttons == 3:
        return core.JOINT.Action.both(x, y)
    return core.JOINT.Action.wait(1)


def _public_checkpoint(env: object, observation: Mapping[str, object]) -> dict[str, object]:
    return {
        "tick": int(observation["tick"]),
        "score": int(observation["score"]),
        "gauge": int(observation["gauge"]),
        "level": int(observation["level"]),
        "highest_chain": int(observation.get("highest_chain", 0)),
        "clears": int(observation.get("qualifying_clear_count", 0)),
        "terminated": bool(observation.get("terminated", False)),
        "truncated": bool(observation.get("truncated", False)),
        "state_u64": f"0x{int(env.state_hash()) & 0xFFFF_FFFF_FFFF_FFFF:016x}",
    }


def _verify_prefix(
    observation: Mapping[str, object], queries: list[dict[str, object]]
) -> str:
    reference = _read_json(REFERENCE_UNIT)
    expected = {
        "tick": reference["survival_ticks"],
        "score": reference["score"],
        "gauge": reference["final_gauge"],
        "clears": reference["clears"],
    }
    actual = {
        "tick": int(observation["tick"]),
        "score": int(observation["score"]),
        "gauge": int(observation["gauge"]),
        "clears": int(observation.get("qualifying_clear_count", 0)),
    }
    if actual != expected or queries != reference["queries"]:
        raise RuntimeError(
            f"extended exhibition did not reproduce frozen 10k prefix: {actual!r}"
        )
    return _canonical_sha256({"public": actual, "queries": queries})


def run_episode(run_root: Path) -> dict[str, object]:
    identity, plan = _validate(run_root)
    result_path = run_root / "episode.json"
    if result_path.exists():
        result = _read_json(result_path)
        _verify_self_hash(result, "exhibition episode")
        return result
    intent = _with_sha(
        {
            "schema": "irisu-r3l-exhibition-intent-v1",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "seed": SEED,
            "target_score": TARGET_SCORE,
            "maximum_ticks": MAX_TICKS,
        }
    )
    intent_path = run_root / "episode-intent.json"
    if intent_path.exists():
        if _read_json(intent_path) != intent:
            raise RuntimeError("exhibition retry intent differs")
    else:
        _write_new(intent_path, intent)

    screen = _load_screen()
    core, campaign = screen._load_external()
    model = screen._load_model()
    policy = campaign.POLICY_FACTORY()
    policy.reset(SEED)
    words: list[int] = []
    queries: list[dict[str, object]] = []
    checkpoints: list[dict[str, object]] = []
    crossing: dict[str, object] | None = None
    query_cursor = 0
    thresholds = tuple(screen.QUERY_THRESHOLDS)
    prefix_sha: str | None = None
    started = time.monotonic()
    terminated = truncated = False

    with campaign.IrisuEnv(
        library_path=screen.RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": MAX_TICKS},
    ) as env:
        observation, info = env.reset(seed=SEED)
        if int(info.get("seed", -1)) != SEED:
            raise RuntimeError("exhibition reset seed differs")
        while int(observation["tick"]) < MAX_TICKS and not (terminated or truncated):
            decision = policy.predict(observation)
            if (
                query_cursor < len(thresholds)
                and int(observation["tick"]) >= thresholds[query_cursor]
            ):
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
                    seed=SEED,
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
                remaining = MAX_TICKS - int(observation["tick"])
                if remaining <= 0:
                    break
                duration = min(duration, remaining)
                for _ in range(duration):
                    primitive = (
                        core.JOINT.Action.wait(1)
                        if kind is core.JOINT.ActionKind.WAIT
                        else action
                    )
                    words.append(encode_action_word(core, primitive))
                    observation, _reward, terminated, truncated, _info = env.step(
                        primitive
                    )
                    tick = int(observation["tick"])
                    if tick % CHECKPOINT_INTERVAL == 0:
                        checkpoints.append(_public_checkpoint(env, observation))
                        _progress(
                            run_root,
                            "episode-progress",
                            tick=tick,
                            score=int(observation["score"]),
                            gauge=int(observation["gauge"]),
                            level=int(observation["level"]),
                            queries=len(queries),
                            elapsed_seconds=time.monotonic() - started,
                        )
                    if tick == PREFIX_TICKS:
                        prefix_sha = _verify_prefix(observation, queries)
                        _progress(
                            run_root,
                            "prefix-reproduced",
                            tick=tick,
                            score=int(observation["score"]),
                            prefix_sha256=prefix_sha,
                        )
                    if crossing is None and int(observation["score"]) >= TARGET_SCORE:
                        crossing = _public_checkpoint(env, observation)
                    if crossing is not None:
                        reached_boundary = True
                    if terminated or truncated or tick >= MAX_TICKS:
                        break
                if terminated or truncated or int(observation["tick"]) >= MAX_TICKS:
                    break
            if reached_boundary:
                break

        final = _public_checkpoint(env, observation)
        if not checkpoints or checkpoints[-1]["tick"] != final["tick"]:
            checkpoints.append(final)
        final_snapshot_sha256 = hashlib.sha256(env.clone_state()).hexdigest()
        config_u64 = f"0x{int(env.config_hash()) & 0xFFFF_FFFF_FFFF_FFFF:016x}"

    if prefix_sha is None:
        raise RuntimeError("exhibition ended before reproducing the frozen 10k prefix")
    action_data = b"".join(ACTION_WORD.pack(word) for word in words)
    action_path = run_root / "direct-actions.u32le"
    _write_bytes_new(action_path, action_data)
    success = bool(crossing is not None and int(final["score"]) >= TARGET_SCORE)
    result = _with_sha(
        {
            "schema": "irisu-r3l-current-model-10k-episode-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "claim_scope": "outcome-selected single-episode exhibition only",
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "intent_sha256": intent["sha256"],
            "seed": SEED,
            "target_score": TARGET_SCORE,
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
            "final_snapshot_sha256": final_snapshot_sha256,
            "config_u64": config_u64,
            "wall_seconds": time.monotonic() - started,
        }
    )
    _write_new(result_path, result)
    _progress(
        run_root,
        "episode-complete",
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
        or len(data) != int(result["action_count"]) * ACTION_WORD.size
    ):
        raise RuntimeError("exhibition action trace differs")
    return tuple(word for (word,) in struct.iter_unpack("<I", data))


def verify_episode(run_root: Path) -> dict[str, object]:
    identity, plan = _validate(run_root)
    result = _read_json(run_root / "episode.json")
    _verify_self_hash(result, "exhibition episode")
    words = _load_actions(run_root, result)
    screen = _load_screen()
    core, campaign = screen._load_external()
    expected_checkpoints = {int(row["tick"]): row for row in result["checkpoints"]}
    reproduced = []
    with campaign.IrisuEnv(
        library_path=screen.RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": MAX_TICKS},
    ) as env:
        observation, _ = env.reset(seed=SEED)
        terminated = truncated = False
        for word in words:
            observation, _reward, terminated, truncated, _info = env.step(
                decode_action_word(core, word)
            )
            tick = int(observation["tick"])
            expected = expected_checkpoints.get(tick)
            if expected is not None:
                actual = _public_checkpoint(env, observation)
                if actual != expected:
                    raise RuntimeError(f"exhibition replay diverged at tick {tick}")
                reproduced.append(tick)
        final = _public_checkpoint(env, observation)
        snapshot_sha = hashlib.sha256(env.clone_state()).hexdigest()
        config_u64 = f"0x{int(env.config_hash()) & 0xFFFF_FFFF_FFFF_FFFF:016x}"
    if (
        final != result["final"]
        or reproduced != sorted(expected_checkpoints)
        or snapshot_sha != result["final_snapshot_sha256"]
        or config_u64 != result["config_u64"]
        or bool(result["success"]) is not True
        or int(final["score"]) < TARGET_SCORE
        or len(words) != int(final["tick"])
    ):
        raise RuntimeError("exhibition replay final closure differs")
    verification = _with_sha(
        {
            "schema": "irisu-r3l-current-model-10k-verification-v1",
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
        if _read_json(path) != verification:
            raise RuntimeError("existing exhibition verification differs")
    else:
        _write_new(path, verification)
    _progress(
        run_root,
        "verification-complete",
        score=final["score"],
        verification_sha256=verification["sha256"],
    )
    return verification


def render_video(run_root: Path) -> dict[str, object]:
    identity, plan = _validate(run_root)
    result = _read_json(run_root / "episode.json")
    verification = _read_json(run_root / "verification.json")
    _verify_self_hash(result, "exhibition episode")
    _verify_self_hash(verification, "exhibition verification")
    if verification.get("verified") is not True:
        raise RuntimeError("exhibition is not replay-verified")
    words = _load_actions(run_root, result)
    output = run_root / "current-model-10000plus.mp4"
    manifest_path = run_root / "video-manifest.json"
    if manifest_path.exists():
        manifest = _read_json(manifest_path)
        _verify_self_hash(manifest, "exhibition video manifest")
        if _sha256_file(output) != manifest["video_sha256"]:
            raise RuntimeError("exhibition video differs")
        return manifest

    screen = _load_screen()
    core, campaign = screen._load_external()
    frame_count = 0
    with tempfile.TemporaryDirectory(prefix="irisu-r3l-render-") as temporary:
        frames = Path(temporary)
        with campaign.IrisuEnv(
            library_path=screen.RUNTIME,
            physics_backend="portable",
            config={"max_episode_ticks": MAX_TICKS},
            render_mode="svg",
        ) as env:
            observation, _ = env.reset(seed=SEED)
            (frames / f"{frame_count:06d}.svg").write_text(env.render("svg"))
            frame_count += 1
            for word in words:
                observation, _reward, _terminated, _truncated, _info = env.step(
                    decode_action_word(core, word)
                )
                if int(observation["tick"]) % VIDEO_SAMPLE_TICKS == 0:
                    (frames / f"{frame_count:06d}.svg").write_text(
                        env.render("svg")
                    )
                    frame_count += 1
        temporary_output = run_root / f".{output.name}.{os.getpid()}.tmp.mp4"
        command = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(VIDEO_FPS),
            "-i",
            str(frames / "%06d.svg"),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(temporary_output),
        ]
        subprocess.run(command, check=True, cwd=ROOT)
        os.replace(temporary_output, output)
    manifest = _with_sha(
        {
            "schema": "irisu-r3l-current-model-10k-video-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "renderer": "asset-free deterministic clone SVG",
            "not_original_game_artwork": True,
            "source_identity_sha256": identity["sha256"],
            "plan_sha256": plan["sha256"],
            "episode_sha256": result["sha256"],
            "verification_sha256": verification["sha256"],
            "video": output.name,
            "video_sha256": _sha256_file(output),
            "frame_count": frame_count,
            "fps": VIDEO_FPS,
            "sample_ticks": VIDEO_SAMPLE_TICKS,
            "duration_seconds": frame_count / VIDEO_FPS,
        }
    )
    _write_new(manifest_path, manifest)
    _progress(
        run_root,
        "video-complete",
        video=output.name,
        video_sha256=manifest["video_sha256"],
    )
    return manifest


def status(run_root: Path) -> dict[str, object]:
    _validate(run_root)
    output: dict[str, object] = {
        "schema": "irisu-r3l-exhibition-status-v1",
        "run_id": run_root.name,
        "run_path": str(run_root),
        "episode_exists": (run_root / "episode.json").exists(),
        "verification_exists": (run_root / "verification.json").exists(),
        "video_exists": (run_root / "current-model-10000plus.mp4").exists(),
        "progress": _read_json(run_root / "progress.json"),
    }
    if output["episode_exists"]:
        episode = _read_json(run_root / "episode.json")
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
