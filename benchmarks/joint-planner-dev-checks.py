#!/usr/bin/env python3
"""Development checks for bounded joint pair/geometry planning."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from irisu_env import IrisuEnv
from irisu_pointer.action import PointerActionSpec
from irisu_pointer.joint_planner import (
    COMPACT_GEOMETRY,
    JointPairGeometrySearch,
    JointPlannerConfig,
    _commit_base_decision,
    shortlist_pairs,
)
from irisu_pointer.steering import SteeringDecision, SteeringIntent
from irisu_pointer.steering_checkpoint import load_steering_checkpoint
from irisu_pointer.steering_learning import GoalConditionedSteeringPolicy
from irisu_rl.actions import SemanticAction


RUNTIME = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/runtime/"
    "main-0c48dba-20260723/portable-build/libirisu_clone.so"
)
RUNTIME_SHA256 = (
    "4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79"
)
V5 = Path(
    "/home/gabe/Documents/irisu/artifacts/r3/development/"
    "r3d-survival-v5-20260729/long-development.pt"
)
V5_SHA256 = (
    "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _piece(
    identifier: int,
    *,
    x: float,
    y: float,
    lifecycle: str = "dynamic_fresh",
    chain_id: int = 0,
) -> dict[str, object]:
    return {
        "id": identifier,
        "kind": "piece",
        "shape": "circle",
        "lifecycle": lifecycle,
        "color": 0,
        "x": x,
        "y": y,
        "vx": 0.0,
        "vy": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "size": 32.0,
        "chain_id": chain_id,
        "projectile_hits": 0,
        "remaining_lifetime": 1000,
        "rot_timer": 0,
    }


def _synthetic_observation() -> dict[str, object]:
    return {
        "tick": 100,
        "score": 0,
        "gauge": 2500,
        "gauge_max": 6000,
        "level": 1,
        "highest_chain": 2,
        "qualifying_clear_count": 0,
        "terminated": False,
        "truncated": False,
        "field": {
            "x": 0.0,
            "y": 0.0,
            "width": 640.0,
            "height": 480.0,
        },
        "difficulty": {
            "active_colors": 3,
            "spawn_interval_ticks": 32,
        },
        "bodies": [
            _piece(1, x=100.0, y=120.0),
            _piece(2, x=180.0, y=120.0),
            _piece(3, x=140.0, y=100.0, lifecycle="rotten"),
            _piece(4, x=220.0, y=100.0, lifecycle="confirmed", chain_id=9),
            _piece(5, x=250.0, y=110.0, lifecycle="confirmed", chain_id=9),
        ],
    }


def _policy_factory(model: object) -> GoalConditionedSteeringPolicy:
    return GoalConditionedSteeringPolicy(
        model,  # type: ignore[arg-type]
        cooldown_ticks=16,
        minimum_pair_closure_sizes=0.05,
        impact_side_sizes=0.5,
        impact_below_sizes=0.75,
        source_velocity_lead_ticks=1.0,
        ticks_per_second=50.0,
        act_logit_bias=1.0,
        artifact_sha256=V5_SHA256,
    )


def main() -> None:
    if _sha256(RUNTIME) != RUNTIME_SHA256 or _sha256(V5) != V5_SHA256:
        raise RuntimeError("trusted development input identity changed")

    observation = _synthetic_observation()
    incumbent = SteeringDecision(
        SemanticAction.strong(60.0 / 640.0, 144.0 / 480.0),
        SteeringIntent.STEER_MATCH,
        source_body_id=1,
        destination_body_id=2,
        destination_chain_id=0,
    )
    shortlist = shortlist_pairs(
        observation,
        incumbent,
        config=JointPlannerConfig(pair_cap=3),
    )
    if shortlist[0].source_body_id != 1 or shortlist[0].destination_body_id != 2:
        raise RuntimeError("incumbent is not the first joint pair")
    if {value.category for value in shortlist} != {
        "fresh-match",
        "rotten-hazard",
        "viable-anchor",
    }:
        raise RuntimeError("joint shortlist lost strategic category coverage")

    pointer_templates = {
        (float(x), float(y)) for x, y in PointerActionSpec().templates
    }
    for option in COMPACT_GEOMETRY:
        if (
            (-2.0 * option.side_sizes, 2.0 * option.below_sizes)
            not in pointer_templates
            or (2.0 * option.side_sizes, 2.0 * option.below_sizes)
            not in pointer_templates
        ):
            raise RuntimeError("compact geometry is not exactly representable")

    torch.set_num_threads(1)
    checkpoint = load_steering_checkpoint(V5, expected_sha256=V5_SHA256)

    def factory() -> GoalConditionedSteeringPolicy:
        return _policy_factory(checkpoint.model)

    config = JointPlannerConfig(
        pair_cap=2,
        geometry_cap=2,
        horizons=(8, 16),
        cooldown_ticks=4,
        require_pristine_source=False,
    )
    with IrisuEnv(
        library_path=RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": 400},
    ) as env:
        current, _ = env.reset(seed=0x63D96529)
        base = factory()
        base.reset(0x63D96529)
        while True:
            decision = base.predict(current)
            if decision.is_shot:
                break
            for action in decision.primitive_actions():
                current, _, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    raise RuntimeError("development fixture ended before a shot")
        source_snapshot = env.clone_state()
        searcher = JointPairGeometrySearch(
            factory,
            config=config,
            continuation_identity_sha256=V5_SHA256,
        )
        first = searcher.search(env, current, decision)
        if env.clone_state() != source_snapshot:
            raise RuntimeError("joint search changed its source state")
        second = searcher.search(env, current, decision)
        if env.clone_state() != source_snapshot:
            raise RuntimeError("repeated joint search changed its source state")
        if (
            first.identity_manifest() != second.identity_manifest()
            or len(first.outcomes) > config.branch_cap
            or first.restore_checks != len(first.outcomes) + 1
            or any(
                len(outcome.milestones) != len(config.horizons)
                for outcome in first.outcomes
            )
        ):
            raise RuntimeError("joint search is not deterministic and bounded")
        replacement = next(
            (
                value.candidate.decision
                for value in first.outcomes
                if (
                    value.candidate.pair.source_body_id,
                    value.candidate.pair.destination_body_id,
                )
                != (
                    decision.source_body_id,
                    decision.destination_body_id,
                )
            ),
            None,
        )
        if replacement is None:
            raise RuntimeError("joint rebind check found no alternative pair")
        cooldown = base._cooldown_until
        if not _commit_base_decision(base, current, decision, replacement):
            raise RuntimeError("joint actual-pair rebind failed")
        pending = base._progress.pending_pair
        if (
            pending is None
            or (pending.source_id, pending.destination_id)
            != (
                replacement.source_body_id,
                replacement.destination_body_id,
            )
            or base._cooldown_until != cooldown
        ):
            raise RuntimeError("joint actual-pair rebind changed cadence")

    print(
        {
            "status": "passed",
            "runtime_sha256": RUNTIME_SHA256,
            "v5_sha256": V5_SHA256,
            "planner_config_sha256": config.sha256,
            "shortlist_categories": [value.category for value in shortlist],
            "branches": len(first.outcomes),
            "restore_checks": first.restore_checks,
            "search_result_sha256": first.sha256,
        }
    )


if __name__ == "__main__":
    main()
