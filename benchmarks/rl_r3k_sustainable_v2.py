#!/usr/bin/env python3
"""Fresh development-only matched G4 versus R3K long-horizon screen v2.

R3K retains G4's frozen proposal model and 768-tick exact solvency shield.
For the R3K arm only, a deterministic rescue/growth shortlist is rolled once
through 2,048 ticks from byte-identical portable snapshots.  The final choice
is survival-first, then late score, then B2 runway.  Results are append-only
per arm/seed and an interrupted unit is deterministically restarted.

This runner never reads sealed data, canonical runs, or authorization files.
V2 adds a fail-closed rule discovered by the burned v1 campaign: if the
frozen incumbent is absent from the legal joint inventory, the query abstains
and executes the unchanged incumbent.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / "python"
if str(PYTHON) not in sys.path:
    sys.path.insert(0, str(PYTHON))

from irisu_pointer import resolution_proposal_g4 as g4  # noqa: E402
from irisu_pointer import resolution_sustainable_r3k as r3k  # noqa: E402


RUN_ID = "r3k-runway-long-screen-20260808-002"
DEFAULT_RUN_ROOT = ROOT / "artifacts/r3/development" / RUN_ID
RUNTIME = (
    ROOT
    / "artifacts/r3/runtime/main-0c48dba-20260723/portable-build/"
    "libirisu_clone.so"
)
G4_CHECKPOINT = (
    ROOT
    / "artifacts/r3/development/r3i-resolution-prototypes-20260730/"
    "resolution-proposal-g4-opened56-checkpoint.json"
)
EXTERNAL_ROOT = Path(
    "/home/gabe/.codex/worktrees/a6b0/irisu/artifacts/r3/development/"
    "r3g-distributional-barrier-20260729"
)
CORE_SOURCE = EXTERNAL_ROOT / "barrier_core.py"
CAMPAIGN_SOURCE = EXTERNAL_ROOT / "campaign.py"
BASE_CHECKPOINT = (
    ROOT / "artifacts/r3/development/r3d-survival-v5-20260729/long-development.pt"
)

HORIZON = 10_000
SHORT_HORIZON = 768
LONG_HORIZON = 2_048
SEED_COUNT = 16
QUERY_THRESHOLDS = (0, 2_500, 5_000, 7_500)
BUDGET = 8
ARMS = ("g4", "r3k")
EXPECTED_HEAD = "de701b36355d5ec582df30f4223aabde7bc537df"
EXPECTED = {
    RUNTIME: "4f6928f18c83159b0db1cb895891007ac805d2542954b41d767619eedf3f7c79",
    G4_CHECKPOINT: "4a9218fbacd3d6fd4b7f24b42dc9ea260ad7f0c0c6b3142d19da63bdcd6bf61e",
    BASE_CHECKPOINT: "31c9bc5e10b0ad021eecedf0c0037de6b24bd4d74e0cfbe9b4922b77dc53da1d",
    CORE_SOURCE: "b532547fa2e87afe441c8fbc7edaadfdcee48655dc1f71ba56fd279a39953e84",
    CAMPAIGN_SOURCE: "ebb5c0e770fb3722da3c6528a9b26565ead05b16dddcb46f684aa79aad056567",
    ROOT / "python/irisu_pointer/resolution_proposal_g4.py": (
        "01b98c3e32b7044df6d05dcb187cac8583c21b982f8107228ab4e6805aaee270"
    ),
}


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


def _with_sha(value: Mapping[str, object]) -> dict[str, object]:
    payload = dict(value)
    payload["sha256"] = _canonical_sha256(payload)
    return payload


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    payload = _canonical_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_operational(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_bytes(value) + b"\n")
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if type(value) is not dict:
        raise RuntimeError(f"non-object JSON artifact: {path}")
    return value


def _progress(run_root: Path, event: str, **values: object) -> None:
    row = {
        "schema": "irisu-r3k-operational-progress-v1",
        "time": time.time(),
        "event": event,
        **values,
    }
    _write_operational(run_root / "progress.json", row)
    print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)


def _derive_seed(index: int) -> int:
    raw = hashlib.sha256(f"{RUN_ID}|matched-development|{index}".encode()).digest()
    return int.from_bytes(raw[:4], "big")


SEEDS = tuple(_derive_seed(index) for index in range(SEED_COUNT))


def _source_files() -> tuple[Path, ...]:
    return (
        Path(__file__).resolve(),
        ROOT / "python/irisu_pointer/resolution_sustainable_r3k.py",
        ROOT / "tests/test_pointer_resolution_sustainable_r3k.py",
        ROOT / "tests/test_r3k_sustainable_runner_v2.py",
        *EXPECTED,
    )


def _git_head() -> str:
    import subprocess

    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _source_identity() -> dict[str, object]:
    if _git_head() != EXPECTED_HEAD:
        raise RuntimeError("R3K source revision changed")
    for path, digest in EXPECTED.items():
        if _sha256_file(path) != digest:
            raise RuntimeError(f"R3K trusted input identity changed: {path}")
    files = {str(path): _sha256_file(path) for path in _source_files()}
    return _with_sha(
        {
            "schema": "irisu-r3k-source-identity-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "git_head": EXPECTED_HEAD,
            "files": files,
        }
    )


def _validate_source(run_root: Path) -> dict[str, Any]:
    recorded = _read_json(run_root / "source-identity.json")
    current = _source_identity()
    if current != recorded:
        raise RuntimeError("R3K frozen source identity differs")
    return recorded


def _load_external() -> tuple[ModuleType, ModuleType]:
    if str(EXTERNAL_ROOT) not in sys.path:
        sys.path.insert(0, str(EXTERNAL_ROOT))
    import barrier_core as core
    import campaign

    if (
        Path(core.__file__).resolve() != CORE_SOURCE
        or Path(campaign.__file__).resolve() != CAMPAIGN_SOURCE
    ):
        raise RuntimeError("R3K external runtime resolved to foreign source")
    core.verify_identities()
    return core, campaign


def _load_model() -> object:
    checkpoint = _read_json(G4_CHECKPOINT)
    model = g4.ResolutionProposalG4.from_manifest(checkpoint["model"])
    if (
        model.sha256 != checkpoint["model_sha256"]
        or model.sha256
        != "d1b52057e0928ffaf61369661f203a1d7658a48bd54007a28616010cd7f22ac7"
    ):
        raise RuntimeError("R3K inherited G4 proposal checkpoint differs")
    return model


def initialize(run_root: Path) -> dict[str, object]:
    if run_root.exists():
        raise FileExistsError(f"R3K run path already exists: {run_root}")
    run_root.mkdir(parents=True)
    identity = _source_identity()
    preregistration = _with_sha(
        {
            "schema": "irisu-r3k-long-screen-preregistration-v2",
            "run_id": run_root.name,
            "development_only": True,
            "sealed_test_allowed": False,
            "source_identity_sha256": identity["sha256"],
            "arms": list(ARMS),
            "seeds": list(SEEDS),
            "seed_count": SEED_COUNT,
            "horizon_ticks": HORIZON,
            "short_exact_horizon_ticks": SHORT_HORIZON,
            "runway_horizon_ticks": LONG_HORIZON,
            "query_threshold_ticks": list(QUERY_THRESHOLDS),
            "proposal_budget": BUDGET,
            "unit_order": "seed-major-g4-then-r3k",
            "selection": {
                "g4": "frozen-G4 exact selector",
                "r3k": (
                    "same 768-tick safety shield; extend deterministic rescue/"
                    "growth shortlist; survival, late-score, B2 lexicographic"
                ),
                "unrepresentable_incumbent": (
                    "fail-closed query abstention; execute unchanged incumbent"
                ),
            },
            "promising_gates": {
                "survival_ratio_min": [98, 100],
                "full_survivors_nonregression": True,
                "catastrophic_pair_loss_ticks": 2_000,
                "total_score_ratio_min": [110, 100],
                "late_score_ratio_min": [115, 100],
                "score_pair_wins_min": 10,
                "late_pair_wins_min": 10,
            },
            "sustainable_gates": {
                "late_share_min": [35, 100],
                "q4_over_q3_min": [75, 100],
                "positive_q4_pairs_min": 12,
            },
        }
    )
    _write_new(run_root / "source-identity.json", identity)
    _write_new(run_root / "preregistration.json", preregistration)
    (run_root / "units").mkdir()
    _progress(run_root, "initialized", run_id=run_root.name, pending=32)
    return preregistration


def _public(value: object) -> object:
    if value is None or type(value) in {bool, int, float, str}:
        return value
    if isinstance(value, Mapping):
        return {str(key): _public(child) for key, child in value.items()}
    item = getattr(value, "item", None)
    if callable(item):
        return _public(item())
    return [_public(child) for child in value]  # type: ignore[arg-type]


def _candidate_manifest(candidate: object) -> dict[str, object]:
    value = _public(candidate.manifest())
    if type(value) is not dict:
        raise RuntimeError("R3K candidate manifest is malformed")
    return json.loads(_canonical_bytes(value))


def _candidate_identities(
    seed: int, query_id: str, manifests: Sequence[Mapping[str, object]]
) -> tuple[g4.G4CandidateIdentity, ...]:
    output = []
    for manifest in manifests:
        stable = {
            key: value
            for key, value in manifest.items()
            if key not in {"ordinal", "pair_ordinal", "geometry_ordinal"}
        }
        output.append(
            g4.G4CandidateIdentity(
                seed,
                query_id,
                int(manifest["ordinal"]),
                _canonical_sha256(stable),
                _canonical_sha256(manifest["action"]),
            )
        )
    return tuple(output)


def _enumerate_candidates(
    evaluator: object,
    observation: Mapping[str, Any],
    incumbent: object,
) -> tuple[object, ...] | None:
    """Return ``None`` only for the planner's closed-world abstention."""

    try:
        return tuple(evaluator.candidates(observation, incumbent))
    except ValueError as exc:
        if str(exc) != "incoming incumbent is absent from the legal joint shortlist":
            raise
        return None


def _raw_manifest(raw: object) -> dict[str, object]:
    return {
        "candidate": _candidate_manifest(raw.candidate),
        "feature_vector": list(raw.feature_vector),
        "ledger": _public(raw.ledger),
        "survival_ticks": int(raw.survival_ticks),
        "score_gain": int(raw.score_gain),
        "final_gauge": int(raw.final_gauge),
        "final_level": int(raw.final_level),
        "terminal": bool(raw.terminal),
        "gauge_failure": bool(raw.gauge_failure),
        "invalid_actions": int(raw.invalid_actions),
        "continuation_rebind_failed": bool(raw.continuation_rebind_failed),
    }


def _short_outcome(
    raw: object,
    base: object,
    identity: g4.G4CandidateIdentity,
    *,
    level: int,
    observation_sha256: str,
) -> g4.G4ExactOutcome:
    base_b2 = base.ledger["b2_margin"]
    b2 = raw.ledger["b2_margin"]
    delta = None if b2 is None or base_b2 is None else int(b2) - int(base_b2)
    if identity.ordinal == 0:
        delta = 0
    survival_loss = int(base.survival_ticks) - int(raw.survival_ticks)
    catastrophic = bool(
        survival_loss >= 1_000
        and base.survival_ticks > 0
        and raw.survival_ticks / base.survival_ticks <= 0.5
    )
    unresolved = tuple(str(row) for row in raw.ledger["unresolved"])
    new_terminal = bool(raw.terminal and not base.terminal)
    new_gauge = bool(raw.gauge_failure and not base.gauge_failure)
    negative = b2 is not None and int(b2) < 0
    cashflow_lost = bool(raw.ledger["cashflow_lost"])
    exact_unsafe = bool(
        new_terminal
        or new_gauge
        or catastrophic
        or negative
        or not bool(raw.ledger["renewals_resolved"])
        or unresolved
        or raw.continuation_rebind_failed
        or raw.invalid_actions
        or cashflow_lost
    )
    severe = bool(new_terminal or new_gauge or catastrophic or negative or cashflow_lost)
    candidate_resolved = bool(
        raw.ledger["renewals_resolved"]
        and b2 is not None
        and not unresolved
        and not raw.continuation_rebind_failed
    )
    base_resolved = bool(
        base.ledger["renewals_resolved"]
        and base_b2 is not None
        and not base.ledger["unresolved"]
        and not base.continuation_rebind_failed
    )
    return g4.G4ExactOutcome(
        identity,
        True,
        level,
        candidate_resolved,
        bool(base_resolved and candidate_resolved and delta is not None),
        exact_unsafe,
        severe,
        None if b2 is None else float(b2),
        None if delta is None else float(delta),
        float(int(raw.score_gain) - int(base.score_gain)),
        _canonical_sha256(_raw_manifest(raw)),
        observation_sha256,
    )


def _query(
    *,
    run_root: Path,
    core: ModuleType,
    campaign: ModuleType,
    model: object,
    env: object,
    observation: Mapping[str, Any],
    policy: object,
    incumbent: object,
    arm: str,
    seed: int,
    index: int,
    query_index: int,
) -> tuple[object, dict[str, object]]:
    query_id = f"{RUN_ID}:{arm}:{index}:{int(observation['tick'])}:q{query_index}"
    public = json.loads(_canonical_bytes(_public(observation)))
    public_sha = _canonical_sha256(public)
    evaluator = core.ExactBranchEvaluator(campaign.POLICY_FACTORY)
    candidates = _enumerate_candidates(evaluator, observation, incumbent)
    if candidates is None:
        return incumbent, {
            "query_id": query_id,
            "tick": int(observation["tick"]),
            "status": "unrepresentable-incumbent-abstention",
            "candidate_count": 0,
            "selected_ordinal": 0,
            "rebind_succeeded": True,
            "short_exact_cost": 0,
            "long_exact_cost": 0,
        }
    manifests = tuple(_candidate_manifest(row) for row in candidates)
    identities = _candidate_identities(seed, query_id, manifests)
    if len(candidates) <= 1:
        return incumbent, {
            "query_id": query_id,
            "tick": int(observation["tick"]),
            "status": "no-alternative",
            "candidate_count": len(candidates),
            "selected_ordinal": 0,
            "short_exact_cost": 0,
            "long_exact_cost": 0,
        }
    envelope = {
        "schema": g4.INFERENCE_SCHEMA,
        "seed": seed,
        "query_id": query_id,
        "query_index": query_index,
        "shot_index": query_index + 1,
        "tick": int(public["tick"]),
        "pre_query_public_observation": public,
        "pre_query_public_observation_sha256": public_sha,
        "candidates": list(manifests),
    }
    inference = g4.g4_inference_board_from_entries([envelope])
    inventory = inference.inventory((seed, query_id))
    actual = (inventory.incumbent_identity, *inventory.identities)
    if actual != identities:
        raise RuntimeError("R3K candidate identities changed in G4 reconstruction")
    batches = model.predict_batches(inference, oof=False)
    if len(batches) != 1:
        raise RuntimeError("R3K query produced multiple G4 batches")
    prediction = batches[0]
    snapshot = env.clone_state()
    state_hash = env.state_hash()

    def restore() -> None:
        env.restore_state(snapshot)
        if env.clone_state() != snapshot or env.state_hash() != state_hash:
            raise RuntimeError("R3K exact branch restore mismatch")

    restore()
    base_raw = evaluator._evaluate_one(
        env, observation, candidates[0], candidates[0], policy
    )
    incumbent_outcome = _short_outcome(
        base_raw,
        base_raw,
        identities[0],
        level=int(public["level"]),
        observation_sha256=public_sha,
    )
    proposal = g4.propose_g4(prediction, incumbent_outcome, budget=BUDGET)
    outcome_rows = [incumbent_outcome]
    raw_rows = {0: base_raw}
    for position, identity in enumerate(proposal.identities, start=1):
        _progress(
            run_root,
            "short-exact",
            arm=arm,
            index=index,
            seed=seed,
            query_index=query_index,
            branch=position,
            branches=len(proposal.identities),
            horizon=SHORT_HORIZON,
        )
        restore()
        raw = evaluator._evaluate_one(
            env,
            observation,
            candidates[identity.ordinal],
            candidates[0],
            policy,
        )
        raw_rows[identity.ordinal] = raw
        outcome_rows.append(
            _short_outcome(
                raw,
                base_raw,
                identity,
                level=int(public["level"]),
                observation_sha256=public_sha,
            )
        )
    short_outcomes = tuple(outcome_rows)
    long_outcomes: tuple[r3k.RunwayOutcome, ...] = ()
    if arm == "g4":
        selection = g4.select_exact_g4(proposal, short_outcomes)
        selected_identity = selection.identity
        selection_manifest = selection.manifest()
    else:
        extension = r3k.extension_identities(
            proposal, short_outcomes, prediction.predictions
        )
        long_rows = []
        for position, identity in enumerate(extension):
            _progress(
                run_root,
                "long-exact",
                arm=arm,
                index=index,
                seed=seed,
                query_index=query_index,
                branch=position + 1,
                branches=len(extension),
                horizon=LONG_HORIZON,
            )
            restore()
            long_rows.append(
                r3k.rollout_candidate(
                    core,
                    env,
                    observation,
                    candidates[identity.ordinal],
                    candidates[0],
                    policy,
                    identity,
                    observation_sha256=public_sha,
                    action_spec=evaluator.action_spec,
                )
            )
        long_outcomes = tuple(long_rows)
        selection = r3k.select_sustainable(
            proposal, short_outcomes, long_outcomes
        )
        selected_identity = selection.identity
        selection_manifest = selection.manifest()
    restore()
    selected = candidates[selected_identity.ordinal]
    rebound = True
    if selected_identity.ordinal:
        rebound = core.commit_base_decision(
            policy, observation, incumbent, selected.decision
        )
        if rebound is not True:
            selected = candidates[0]
            selected_identity = identities[0]
    selected_short = next(
        row for row in short_outcomes if row.identity == selected_identity
    )
    if selected_identity.ordinal and not selected_short.absolute_safe:
        raise RuntimeError("R3K attempted to execute an unsafe alternative")
    return selected.decision, {
        "query_id": query_id,
        "tick": int(observation["tick"]),
        "mode": proposal.mode,
        "candidate_count": len(candidates),
        "proposal": proposal.manifest(),
        "prediction_batch_sha256": prediction.sha256,
        "short_outcomes": [row.manifest() for row in short_outcomes],
        "long_outcomes": [row.manifest() for row in long_outcomes],
        "selection": selection_manifest,
        "selected_ordinal": selected_identity.ordinal,
        "rebind_succeeded": rebound,
        "short_exact_cost": len(short_outcomes),
        "long_exact_cost": len(long_outcomes),
    }


def _primitive_actions(core: ModuleType, decision: object) -> tuple[object, ...]:
    spec = core.JOINT.ActionSpec()
    if hasattr(decision, "primitive_actions"):
        return tuple(decision.primitive_actions(spec))
    return core.ExactBranchEvaluator._primitive_tuple(decision, spec)


def _unit_path(run_root: Path, arm: str, index: int) -> Path:
    return run_root / "units" / f"{index:02d}-{arm}.json"


def run_unit(run_root: Path, arm: str, index: int) -> dict[str, object]:
    _validate_source(run_root)
    if arm not in ARMS or type(index) is not int or not 0 <= index < SEED_COUNT:
        raise ValueError("R3K unit identity is invalid")
    result_path = _unit_path(run_root, arm, index)
    if result_path.exists():
        return _read_json(result_path)
    seed = SEEDS[index]
    intent_path = result_path.with_suffix(".intent.json")
    intent = _with_sha(
        {
            "schema": "irisu-r3k-unit-intent-v1",
            "source_identity_sha256": _read_json(
                run_root / "source-identity.json"
            )["sha256"],
            "arm": arm,
            "index": index,
            "seed": seed,
            "horizon_ticks": HORIZON,
        }
    )
    if intent_path.exists():
        if _read_json(intent_path) != intent:
            raise RuntimeError("R3K retry intent differs")
    else:
        _write_new(intent_path, intent)
    core, campaign = _load_external()
    model = _load_model()
    policy = campaign.POLICY_FACTORY()
    policy.reset(seed)
    started = time.monotonic()
    queries: list[dict[str, object]] = []
    query_cursor = 0
    next_report = 500
    quarter_scores: list[int | None] = [None, None, None, None, None]
    quarter_ticks = (0, 2_500, 5_000, 7_500, 10_000)
    with campaign.IrisuEnv(
        library_path=RUNTIME,
        physics_backend="portable",
        config={"max_episode_ticks": HORIZON + LONG_HORIZON},
    ) as env:
        observation, info = env.reset(seed=seed)
        if int(info.get("seed", -1)) != seed:
            raise RuntimeError("R3K reset seed differs")
        initial_score = int(observation.get("score", 0))
        terminated = truncated = False
        while int(observation["tick"]) < HORIZON and not (terminated or truncated):
            decision = policy.predict(observation)
            if (
                query_cursor < len(QUERY_THRESHOLDS)
                and int(observation["tick"]) >= QUERY_THRESHOLDS[query_cursor]
                and HORIZON - int(observation["tick"]) >= LONG_HORIZON
            ):
                decision, query = _query(
                    run_root=run_root,
                    core=core,
                    campaign=campaign,
                    model=model,
                    env=env,
                    observation=observation,
                    policy=policy,
                    incumbent=decision,
                    arm=arm,
                    seed=seed,
                    index=index,
                    query_index=query_cursor,
                )
                queries.append(query)
                query_cursor += 1
            for action in _primitive_actions(core, decision):
                kind = core.JOINT.ActionKind.parse(action.kind)
                duration = int(action.wait_ticks) if kind is core.JOINT.ActionKind.WAIT else 1
                remaining = HORIZON - int(observation["tick"])
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
                    tick = int(observation["tick"])
                    for q_index, threshold in enumerate(quarter_ticks):
                        if tick >= threshold and quarter_scores[q_index] is None:
                            quarter_scores[q_index] = int(observation.get("score", 0))
                    if tick >= next_report:
                        _progress(
                            run_root,
                            "episode-progress",
                            arm=arm,
                            index=index,
                            seed=seed,
                            tick=tick,
                            score=int(observation.get("score", 0)),
                            gauge=int(observation.get("gauge", 0)),
                            queries=len(queries),
                            elapsed_seconds=time.monotonic() - started,
                        )
                        next_report += 500
                    if terminated or truncated or tick >= HORIZON:
                        break
        final_tick = min(int(observation["tick"]), HORIZON)
        final_score = int(observation.get("score", 0))
        for q_index, value in enumerate(quarter_scores):
            if q_index == 0:
                quarter_scores[q_index] = initial_score
            elif value is None:
                quarter_scores[q_index] = final_score
        quarter_gains = [
            int(quarter_scores[i + 1]) - int(quarter_scores[i])
            for i in range(4)
        ]
        result = _with_sha(
            {
                "schema": "irisu-r3k-long-screen-unit-v1",
                "development_only": True,
                "sealed_test_allowed": False,
                "source_identity_sha256": _read_json(
                    run_root / "source-identity.json"
                )["sha256"],
                "intent_sha256": intent["sha256"],
                "arm": arm,
                "index": index,
                "seed": seed,
                "horizon_ticks": HORIZON,
                "survival_ticks": final_tick,
                "survived_full_horizon": final_tick == HORIZON,
                "score": final_score,
                "score_gain": final_score - initial_score,
                "clears": int(observation.get("qualifying_clears", 0)),
                "final_gauge": int(observation.get("gauge", 0)),
                "quarter_score_gains": quarter_gains,
                "query_count": len(queries),
                "queries": queries,
                "wall_seconds": time.monotonic() - started,
            }
        )
    _write_new(result_path, result)
    _progress(
        run_root,
        "unit-complete",
        arm=arm,
        index=index,
        seed=seed,
        survival_ticks=result["survival_ticks"],
        score=result["score"],
        wall_seconds=result["wall_seconds"],
    )
    return result


def _completed(run_root: Path) -> dict[tuple[str, int], dict[str, Any]]:
    output = {}
    for index in range(SEED_COUNT):
        for arm in ARMS:
            path = _unit_path(run_root, arm, index)
            if path.exists():
                row = _read_json(path)
                if row.get("arm") != arm or row.get("index") != index:
                    raise RuntimeError("R3K unit filename and identity differ")
                output[(arm, index)] = row
    return output


def status(run_root: Path) -> dict[str, object]:
    _validate_source(run_root)
    complete = _completed(run_root)
    progress = _read_json(run_root / "progress.json")
    counts = {
        arm: sum((arm, index) in complete for index in range(SEED_COUNT))
        for arm in ARMS
    }
    return {
        "schema": "irisu-r3k-long-screen-status-v1",
        "run_id": run_root.name,
        "run_path": str(run_root),
        "completed": len(complete),
        "pending": 2 * SEED_COUNT - len(complete),
        "by_arm": counts,
        "progress": progress,
        "summary_exists": (run_root / "summary.json").exists(),
    }


def summarize(run_root: Path) -> dict[str, object]:
    _validate_source(run_root)
    complete = _completed(run_root)
    if len(complete) != 2 * SEED_COUNT:
        raise RuntimeError("R3K cannot summarize before all matched units complete")
    by_arm: dict[str, dict[str, object]] = {}
    for arm in ARMS:
        rows = [complete[(arm, index)] for index in range(SEED_COUNT)]
        quarters = [
            sum(int(row["quarter_score_gains"][q]) for row in rows)
            for q in range(4)
        ]
        by_arm[arm] = {
            "score": sum(int(row["score"]) for row in rows),
            "score_gain": sum(int(row["score_gain"]) for row in rows),
            "survival_ticks": sum(int(row["survival_ticks"]) for row in rows),
            "full_survivors": sum(bool(row["survived_full_horizon"]) for row in rows),
            "clears": sum(int(row["clears"]) for row in rows),
            "quarter_score_gains": quarters,
            "late_score_gain": quarters[2] + quarters[3],
        }
    control, treatment = by_arm["g4"], by_arm["r3k"]
    score_wins = sum(
        int(complete[("r3k", index)]["score"])
        > int(complete[("g4", index)]["score"])
        for index in range(SEED_COUNT)
    )
    late_wins = sum(
        sum(complete[("r3k", index)]["quarter_score_gains"][2:])
        > sum(complete[("g4", index)]["quarter_score_gains"][2:])
        for index in range(SEED_COUNT)
    )
    catastrophic = sum(
        bool(complete[("g4", index)]["survived_full_horizon"])
        and int(complete[("g4", index)]["survival_ticks"])
        - int(complete[("r3k", index)]["survival_ticks"])
        >= 2_000
        for index in range(SEED_COUNT)
    )
    promising_checks = {
        "survival_ratio": int(treatment["survival_ticks"]) * 100
        >= int(control["survival_ticks"]) * 98,
        "full_survivors": int(treatment["full_survivors"])
        >= int(control["full_survivors"]),
        "no_catastrophic_pair_loss": catastrophic == 0,
        "total_score_ratio": int(treatment["score"]) * 100
        >= int(control["score"]) * 110,
        "late_score_ratio": int(treatment["late_score_gain"]) * 100
        >= int(control["late_score_gain"]) * 115,
        "score_pair_wins": score_wins >= 10,
        "late_pair_wins": late_wins >= 10,
    }
    quarters = treatment["quarter_score_gains"]
    total_gain = sum(quarters)
    late_gain = quarters[2] + quarters[3]
    sustainable_checks = {
        "late_share": late_gain * 100 >= total_gain * 35,
        "q4_over_q3": quarters[3] * 100 >= quarters[2] * 75,
        "positive_q4_pairs": sum(
            int(complete[("r3k", index)]["quarter_score_gains"][3]) > 0
            for index in range(SEED_COUNT)
        )
        >= 12,
    }
    summary = _with_sha(
        {
            "schema": "irisu-r3k-long-screen-summary-v1",
            "development_only": True,
            "sealed_test_allowed": False,
            "run_id": run_root.name,
            "source_identity_sha256": _read_json(
                run_root / "source-identity.json"
            )["sha256"],
            "by_arm": by_arm,
            "paired": {
                "score_wins": score_wins,
                "late_score_wins": late_wins,
                "catastrophic_survival_losses": catastrophic,
            },
            "promising_checks": promising_checks,
            "promising": all(promising_checks.values()),
            "sustainable_checks": sustainable_checks,
            "sustainable_go": all(promising_checks.values())
            and all(sustainable_checks.values()),
        }
    )
    path = run_root / "summary.json"
    if path.exists():
        if _read_json(path) != summary:
            raise RuntimeError("R3K summary already exists with different bytes")
    else:
        _write_new(path, summary)
    _progress(
        run_root,
        "campaign-complete",
        promising=summary["promising"],
        sustainable_go=summary["sustainable_go"],
        summary_sha256=summary["sha256"],
    )
    return summary


def run_campaign(run_root: Path) -> None:
    _validate_source(run_root)
    allowed = sorted(os.sched_getaffinity(0))
    if not allowed:
        raise RuntimeError("R3K process has no CPU affinity")
    os.sched_setaffinity(0, {allowed[0]})
    lock_path = run_root / "campaign.lock"
    lock = lock_path.open("a+b")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    _write_operational(
        run_root / "process.json",
        {
            "schema": "irisu-r3k-process-v1",
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "cpu": allowed[0],
            "started_at": time.time(),
        },
    )
    for index in range(SEED_COUNT):
        for arm in ARMS:
            if not _unit_path(run_root, arm, index).exists():
                run_unit(run_root, arm, index)
    summarize(run_root)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("init", "run", "status", "summarize", "unit"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--index", type=int)
    args = parser.parse_args(argv)
    if args.command == "init":
        value = initialize(args.run_root)
    elif args.command == "run":
        run_campaign(args.run_root)
        value = status(args.run_root)
    elif args.command == "status":
        value = status(args.run_root)
    elif args.command == "summarize":
        value = summarize(args.run_root)
    else:
        if args.arm is None or args.index is None:
            parser.error("unit requires --arm and --index")
        value = run_unit(args.run_root, args.arm, args.index)
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
