from __future__ import annotations

import copy
import fcntl
import hashlib
import importlib.util
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


REPOSITORY = Path("/home/gabe/Documents/irisu")
SOURCE = (
    REPOSITORY
    / "artifacts/r3/development/r3j-g4-onpolicy-screen-20260731-001"
    / "run_g4_onpolicy_screen.py"
)
VERIFIER_SOURCE = (
    REPOSITORY
    / "artifacts/r3/development/r3j-g4-onpolicy-screen-20260731-001"
    / "verify_g4_onpolicy_screen.py"
)


@pytest.fixture(scope="module")
def runner():
    name = "_r3j_g4_onpolicy_hardening_runner"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
        assert module._ACTIVE_COMMAND_LEASE is None
        assert module._ACTIVE_EXACT_INTENT is None
    finally:
        sys.modules.pop(name, None)


@pytest.fixture(scope="module")
def verifier():
    name = "_r3j_g4_onpolicy_hardening_verifier"
    spec = importlib.util.spec_from_file_location(name, VERIFIER_SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
        assert module._ACTIVE_VERIFICATION_LEASE is None
    finally:
        sys.modules.pop(name, None)


@pytest.fixture
def isolated(runner, tmp_path: Path, monkeypatch):
    root = (tmp_path / "experiment").resolve()
    root.mkdir()
    monkeypatch.setattr(runner, "RUN_ROOT", root / "runs")
    for name in (
        "SOURCE_IDENTITY",
        "SEED_COLLISION_PROOF",
        "PREFLIGHT",
        "CHECKPOINT_ADOPTION",
        "INITIALIZE_INTENT",
        "INITIALIZE_COMPLETION",
        "SCREEN_LEDGER",
        "SCREEN_EVIDENCE",
        "SUMMARY_INTENT",
        "SUMMARY_COMPLETION",
    ):
        monkeypatch.setattr(runner, name, root / Path(getattr(runner, name)).name)

    original_entries = runner._safe_directory_entries
    original_exists = runner._safe_directory_exists

    def safe_entries(path: Path, *, root: Path = root):
        return original_entries(path, root=root)

    def safe_exists(path: Path, *, root: Path = root):
        return original_exists(path, root=root)

    monkeypatch.setattr(runner, "_safe_directory_entries", safe_entries)
    monkeypatch.setattr(runner, "_safe_directory_exists", safe_exists)
    with runner._command_lease(root, enforce_root_closure=False) as lease:
        yield SimpleNamespace(runner=runner, root=root, lease=lease)


def _artifact(runner, schema: str = "unit", **values: object) -> dict[str, object]:
    return runner._with_sha({"schema": schema, **values})


def _write_artifact(
    runner, path: Path, schema: str = "unit", **values: object
) -> dict[str, object]:
    value = _artifact(runner, schema, **values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner._canonical_bytes(value) + b"\n")
    return value


def _write_episode(runner, path: Path) -> dict[str, object]:
    body = {"schema": "unit-episode", "complete": True}
    value = {**body, "episode_sha256": runner._canonical_sha256(body)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner._canonical_bytes(value) + b"\n")
    return value


def _publish_stage_prefix(
    runner,
    root: Path,
    query_root: Path,
    count: int,
    *,
    eligible: bool,
) -> dict[str, dict[str, str]]:
    runner._mkdir_direct(query_root, root=root)
    records: dict[str, dict[str, str]] = {}
    for order, name in enumerate(runner.QUERY_STAGES[:count], 1):
        body: dict[str, object] = {"schema": f"unit-{name}"}
        if order == 1:
            body["g4_reconstruction"] = {"eligible": eligible}
        if order == 4 and not eligible:
            body["schema"] = "irisu-r3j-g4-no-alternative-incumbent-v1"
        runner._publish_query_stage(
            query_root, records, order, name, body, root=root
        )
    return records


def _write_journal_pair(runner, query_root: Path, position: int = 0) -> None:
    _write_artifact(
        runner,
        runner._journal_path(query_root, position, "intent"),
        "irisu-r3j-g4-exact-evaluation-intent-v1",
        position=position,
    )
    _write_artifact(
        runner,
        runner._journal_path(query_root, position, "result"),
        "irisu-r3j-g4-exact-evaluation-result-v1",
        position=position,
    )


def _write_exact_intent(runner, query_root: Path) -> tuple[Path, dict[str, object]]:
    anchor_path = runner._stage_path(query_root, 3, "preexact-plan")
    anchor = runner._read_canonical_json(anchor_path)
    path = runner._journal_path(query_root, 0, "intent")
    value = _artifact(
        runner,
        "irisu-r3j-g4-exact-evaluation-intent-v1",
        seed=1,
        query_id="unit-query",
        split=runner.SPLIT,
        position=0,
        role="incumbent",
        candidate={"ordinal": 0},
        candidate_identity={"ordinal": 0},
        state_snapshot_sha256="a" * 64,
        policy_state_sha256="b" * 64,
        anchor={
            "name": "preexact-plan",
            "artifact_sha256": anchor["sha256"],
            "file_sha256": runner._sha256_file(anchor_path),
        },
        previous_result=None,
    )
    path.write_bytes(runner._canonical_bytes(value) + b"\n")
    return path, value


def _publish_no_alternative_query(
    runner, root: Path, query_root: Path
) -> dict[str, dict[str, str]]:
    runner._mkdir_direct(query_root, root=root)
    records: dict[str, dict[str, str]] = {}
    policy_state = _artifact(
        runner,
        "irisu-r3j-frozen-v5-mutable-policy-state-v1",
        policy_type="unit-policy",
        configuration={},
        components={},
        mutable={},
    )
    candidates = [{"ordinal": 0}]
    public_observation = {"tick": 1}
    plan = _artifact(
        runner,
        "irisu-r3j-g4-no-alternative-plan-v1",
        candidate_receipt_sha256="6" * 64,
        requested_budget=runner.BUDGET,
        branches={"rescue": [], "growth": []},
    )
    proposal = _artifact(
        runner,
        "irisu-r3j-g4-no-alternative-proposal-v1",
        query_key=[1, "unit-query"],
        requested_budget=runner.BUDGET,
        effective_budget=0,
        identities=[],
        status="no-alternative-abstention",
    )
    selection = _artifact(
        runner,
        "irisu-r3j-g4-no-alternative-selection-v1",
        proposal_sha256=proposal["sha256"],
        status="no-alternative-abstention",
        identity={"ordinal": 0},
    )
    bodies: list[dict[str, object]] = [
        {
            "schema": "irisu-r3j-g4-candidate-receipt-v2",
            "development_only": True,
            "sealed_test_allowed": False,
            "source_identity_sha256": "1" * 64,
            "generator": {},
            "runtime": {},
            "frozen_v5_sha256": "2" * 64,
            "g4_checkpoint_sha256": "3" * 64,
            "g4_model_sha256": "4" * 64,
            "policy_state_after_incumbent_prediction": policy_state,
            "policy_state_after_incumbent_prediction_sha256": policy_state[
                "sha256"
            ],
            "seed": 1,
            "query_id": "unit-query",
            "query_index": 0,
            "shot_index": 1,
            "tick": 1,
            "level": 1,
            "public_observation": public_observation,
            "public_observation_sha256": runner._canonical_sha256(
                public_observation
            ),
            "candidate_count": 1,
            "alternative_count": 0,
            "candidates": candidates,
            "candidate_inventory_sha256": runner._canonical_sha256(
                candidates
            ),
            "candidate_identities": [{"ordinal": 0}],
            "g4_reconstruction": {
                "eligible": False,
                "reason": "no-alternative",
                "inference_envelope": None,
                "inference_envelope_sha256": None,
                "g4_inventory": None,
                "g4_inventory_sha256": None,
                "g4_board_sha256": None,
                "g4_feature_inventory_sha256": None,
            },
        },
        {
            "schema": "irisu-r3j-g4-no-alternative-prediction-v1",
            "candidate_receipt_sha256": None,
            "model_called": False,
            "prediction_batch": None,
            "prediction_batch_sha256": None,
        },
        {
            "schema": "irisu-r3j-g4-preexact-plan-receipt-v1",
            "plan": plan,
            "plan_sha256": plan["sha256"],
        },
        {
            "schema": "irisu-r3j-g4-no-alternative-incumbent-v1",
            "exact_work_spent": 0,
            "identity": {"ordinal": 0},
            "outcome": None,
        },
        {
            "schema": "irisu-r3j-g4-no-alternative-proposal-receipt-v1",
            "proposal": proposal,
            "proposal_sha256": proposal["sha256"],
        },
        {
            "schema": "irisu-r3j-g4-no-alternative-exact-outcomes-v1",
            "proposal_sha256": proposal["sha256"],
            "exact_cost": 0,
            "restore_checks": 0,
            "branches": [],
            "outcomes": [],
            "outcome_sha256": [],
        },
        {
            "schema": "irisu-r3j-g4-selection-receipt-v1",
            "proposal_sha256": proposal["sha256"],
            "selection": selection,
            "selection_sha256": selection["sha256"],
        },
        {
            "schema": "irisu-r3j-g4-live-execution-intent-v1",
            "selection_sha256": selection["sha256"],
            "identity": {"ordinal": 0},
            "candidate": {"ordinal": 0},
            "rebind_required": False,
            "required_rebind_count": 0,
            "policy_pre_state_sha256": "5" * 64,
            "exact_tau2_certificate_sha256": None,
            "start_tick": 1,
        },
        {
            "schema": "irisu-r3j-g4-live-execution-v1",
            "intent_sha256": None,
            "executed_primitive_ticks": 0,
            "end_tick": 1,
            "policy_post_state_sha256": "5" * 64,
            "rebind_count": 0,
            "complete": True,
        },
        {
            "schema": "irisu-r3j-g4-live-renewal-closure-v1",
            "execution_sha256": None,
            "required": False,
            "status": "not-required",
            "query_suppressed_until_closed": False,
            "exact_certificate": None,
            "exact_certificate_sha256": None,
            "live_certificate": None,
            "live_certificate_sha256": None,
            "exact_live_parity": None,
        },
    ]
    for order, (name, body) in enumerate(
        zip(runner.QUERY_STAGES[:10], bodies, strict=True), 1
    ):
        runner._publish_query_stage(
            query_root, records, order, name, body, root=root
        )
    prior = {
        name: records[name]
        for name in runner.QUERY_STAGES[:10]
    }
    ledger = {
        "schema": "irisu-r3j-g4-query-ledger-v1",
        "seed": 1,
        "query_id": "unit-query",
        "query_index": 0,
        "shot_index": 1,
        "candidate_receipt_sha256": records["candidate-receipt"][
            "artifact_sha256"
        ],
        "prediction_batch_sha256": None,
        "preexact_plan_sha256": plan["sha256"],
        "incumbent_outcome_sha256": records["exact-incumbent"][
            "artifact_sha256"
        ],
        "proposal_sha256": proposal["sha256"],
        "selection_sha256": selection["sha256"],
        "live_renewal_closure_sha256": records["live-renewal-closure"][
            "artifact_sha256"
        ],
        "prior_stage_records": prior,
        "mode": "no-alternative",
        "status": "complete",
        "selected_ordinal": 0,
        "reserve": None,
        "selected_b2": None,
        "selected_score_advantage": None,
        "exact_cost": 0,
        "restore_checks": 0,
        "rebind_count": 0,
        "complete": True,
    }
    runner._publish_query_stage(
        query_root, records, 11, "ledger", ledger, root=root
    )
    return records


def _verifier_policy_state(verifier) -> tuple[dict[str, object], dict[str, object]]:
    def component(kind: str, manifest: dict[str, object]) -> dict[str, object]:
        return {
            "type": kind,
            "manifest": manifest,
            "sha256": verifier.canonical_sha256(manifest),
        }

    schema = component("unit-schema", {"columns": ["unit"]})
    action = component("unit-action", {"actions": ["unit"]})
    pointer = component("unit-pointer", {"coordinates": ["x", "y"]})
    model_state = {
        "schema": "irisu-r3j-frozen-v5-model-state-v1",
        "tensors": [
            {
                "name": "unit.weight",
                "dtype": "float32",
                "device": "cpu",
                "shape": [1],
                "sha256": "a" * 64,
            }
        ],
    }
    model_state = {
        **model_state,
        "sha256": verifier.canonical_sha256(model_state),
    }
    configuration = {
        "artifact_sha256": verifier.FROZEN_V5_SHA256,
        "schema_sha256": schema["sha256"],
        "pointer_action_sha256": pointer["sha256"],
        "cooldown_ticks": 16,
        "minimum_pair_closure_sizes": 1.0,
        "impact_side_sizes": 1.0,
        "impact_below_sizes": 1.0,
        "source_velocity_lead_ticks": 1.0,
        "ticks_per_second": 60.0,
        "act_logit_bias": 0.0,
    }
    components = {
        "encoder": {"type": "unit-encoder", "schema": schema},
        "action_spec": action,
        "pointer_spec": pointer,
        "model": {
            "type": "unit-model",
            "architecture": {"width": 1},
            "architecture_sha256": verifier.canonical_sha256({"width": 1}),
            "module_modes": [
                {"name": "", "type": "unit-model", "training": False}
            ],
            "state_dict": model_state,
            "schema": schema,
            "pointer_spec": pointer,
        },
    }
    state = {
        "schema": "irisu-r3j-frozen-v5-mutable-policy-state-v1",
        "policy_type": "unit-policy",
        "configuration": configuration,
        "components": components,
        "mutable": {
            "cooldown_until": 0,
            "last_tick": None,
            "last_decision": None,
            "progress": {
                "minimum_closure_sizes": 1.0,
                "attempt": None,
                "stalled": [],
            },
        },
    }
    state = {**state, "sha256": verifier.canonical_sha256(state)}
    return state, {
        "policy_type": state["policy_type"],
        "configuration": configuration,
        "components": components,
    }


def _publish_verifier_no_alternative_query(
    runner,
    verifier,
    root: Path,
    query_root: Path,
    *,
    proposal_receipt_schema: str = (
        "irisu-r3j-g4-no-alternative-proposal-receipt-v1"
    ),
    exact_outcomes_schema: str = (
        "irisu-r3j-g4-no-alternative-exact-outcomes-v1"
    ),
) -> SimpleNamespace:
    runner._mkdir_direct(query_root, root=root)
    records: dict[str, dict[str, str]] = {}
    source_sha = "1" * 64
    seed = 17
    index = 0
    query_index = 0
    tick = 41
    level = 2
    query_id = (
        f"{verifier.EXPERIMENT_ID}:{verifier.SPLIT}:{index}:{tick}:q{query_index}"
    )
    policy_state, policy_static = _verifier_policy_state(verifier)
    candidate = {
        "ordinal": 0,
        "action": {"kind": "unit", "x_norm": 0.5, "y_norm": 0.5},
    }
    candidates = [candidate]
    identity = verifier.candidate_identity(seed, query_id, candidate)
    observation = {"tick": tick, "level": level}
    barrier_config = {"schema": "unit-barrier-config"}
    joint_config = {"schema": "unit-joint-config"}
    receipt = runner._publish_query_stage(
        query_root,
        records,
        1,
        "candidate-receipt",
        {
            "schema": "irisu-r3j-g4-candidate-receipt-v2",
            "development_only": True,
            "sealed_test_allowed": False,
            "source_identity_sha256": source_sha,
            "generator": {
                "path": str(verifier.JOINT_SOURCE),
                "sha256": verifier.JOINT_SHA256,
                "version": "unit-version",
                "barrier_core_sha256": verifier.B_SHA256["barrier_core"],
                "barrier_config": barrier_config,
                "barrier_config_sha256": verifier.canonical_sha256(
                    barrier_config
                ),
                "joint_config": joint_config,
                "joint_config_sha256": verifier.canonical_sha256(joint_config),
                "action_spec_sha256": "2" * 64,
            },
            "runtime": {
                "path": str(verifier.RUNTIME),
                "sha256": verifier.RUNTIME_SHA256,
                "runner": {"schema": "unit-runner"},
                "state_snapshot_sha256": "3" * 64,
                "state_hash": 7,
            },
            "frozen_v5_sha256": verifier.FROZEN_V5_SHA256,
            "g4_checkpoint_sha256": verifier.G4_CHECKPOINT_SHA256,
            "g4_model_sha256": verifier.G4_MODEL_SHA256,
            "policy_state_after_incumbent_prediction": policy_state,
            "policy_state_after_incumbent_prediction_sha256": policy_state[
                "sha256"
            ],
            "seed": seed,
            "query_id": query_id,
            "query_index": query_index,
            "shot_index": 1,
            "tick": tick,
            "level": level,
            "public_observation": observation,
            "public_observation_sha256": verifier.canonical_sha256(observation),
            "candidate_count": 1,
            "alternative_count": 0,
            "candidates": candidates,
            "candidate_inventory_sha256": verifier.canonical_sha256(candidates),
            "candidate_identities": [identity],
            "g4_reconstruction": {
                "eligible": False,
                "reason": "no-alternative",
                "inference_envelope": None,
                "inference_envelope_sha256": None,
                "g4_inventory": None,
                "g4_inventory_sha256": None,
                "g4_board_sha256": None,
                "g4_feature_inventory_sha256": None,
            },
        },
        root=root,
    )
    plan = _artifact(
        runner,
        "irisu-r3j-g4-no-alternative-plan-v1",
        candidate_receipt_sha256=receipt["sha256"],
        requested_budget=verifier.BUDGET,
        branches={"rescue": [], "growth": []},
    )
    runner._publish_query_stage(
        query_root,
        records,
        2,
        "prediction",
        {
            "schema": "irisu-r3j-g4-no-alternative-prediction-v1",
            "candidate_receipt_sha256": receipt["sha256"],
            "model_called": False,
            "prediction_batch": None,
            "prediction_batch_sha256": None,
        },
        root=root,
    )
    runner._publish_query_stage(
        query_root,
        records,
        3,
        "preexact-plan",
        {
            "schema": "irisu-r3j-g4-preexact-plan-receipt-v1",
            "plan": plan,
            "plan_sha256": plan["sha256"],
        },
        root=root,
    )
    runner._publish_query_stage(
        query_root,
        records,
        4,
        "exact-incumbent",
        {
            "schema": "irisu-r3j-g4-no-alternative-incumbent-v1",
            "exact_work_spent": 0,
            "identity": identity,
            "outcome": None,
        },
        root=root,
    )
    proposal = _artifact(
        runner,
        "irisu-r3j-g4-no-alternative-proposal-v1",
        query_key=[seed, query_id],
        requested_budget=verifier.BUDGET,
        effective_budget=0,
        identities=[],
        status="no-alternative-abstention",
    )
    runner._publish_query_stage(
        query_root,
        records,
        5,
        "proposal",
        {
            "schema": proposal_receipt_schema,
            "proposal": proposal,
            "proposal_sha256": proposal["sha256"],
        },
        root=root,
    )
    runner._publish_query_stage(
        query_root,
        records,
        6,
        "proposed-exact-outcomes",
        {
            "schema": exact_outcomes_schema,
            "proposal_sha256": proposal["sha256"],
            "exact_cost": 0,
            "restore_checks": 0,
            "branches": [],
            "outcomes": [],
            "outcome_sha256": [],
        },
        root=root,
    )
    selection = _artifact(
        runner,
        "irisu-r3j-g4-no-alternative-selection-v1",
        proposal_sha256=proposal["sha256"],
        status="no-alternative-abstention",
        identity=identity,
    )
    selection_stage = runner._publish_query_stage(
        query_root,
        records,
        7,
        "selection",
        {
            "schema": "irisu-r3j-g4-selection-receipt-v1",
            "proposal_sha256": proposal["sha256"],
            "selection": selection,
            "selection_sha256": selection["sha256"],
        },
        root=root,
    )
    intent = runner._publish_query_stage(
        query_root,
        records,
        8,
        "execution-intent",
        {
            "schema": "irisu-r3j-g4-live-execution-intent-v1",
            "selection_sha256": selection["sha256"],
            "identity": identity,
            "candidate": candidate,
            "rebind_required": False,
            "required_rebind_count": 0,
            "policy_pre_state_sha256": policy_state["sha256"],
            "exact_tau2_certificate_sha256": None,
            "start_tick": tick,
        },
        root=root,
    )
    execution = runner._publish_query_stage(
        query_root,
        records,
        9,
        "live-execution",
        {
            "schema": "irisu-r3j-g4-live-execution-v1",
            "intent_sha256": intent["sha256"],
            "executed_primitive_ticks": 0,
            "end_tick": tick,
            "policy_post_state_sha256": policy_state["sha256"],
            "rebind_count": 0,
            "complete": True,
        },
        root=root,
    )
    closure = runner._publish_query_stage(
        query_root,
        records,
        10,
        "live-renewal-closure",
        {
            "schema": "irisu-r3j-g4-live-renewal-closure-v1",
            "execution_sha256": execution["sha256"],
            "required": False,
            "status": "not-opened",
            "query_suppressed_until_closed": False,
            "exact_certificate": None,
            "exact_certificate_sha256": None,
            "live_certificate": None,
            "live_certificate_sha256": None,
            "exact_live_parity": False,
        },
        root=root,
    )
    reserve = 9.0
    ledger = runner._publish_query_stage(
        query_root,
        records,
        11,
        "ledger",
        {
            "schema": "irisu-r3j-g4-query-ledger-v1",
            "seed": seed,
            "query_id": query_id,
            "query_index": query_index,
            "shot_index": 1,
            "candidate_receipt_sha256": receipt["sha256"],
            "prediction_batch_sha256": None,
            "preexact_plan_sha256": plan["sha256"],
            "incumbent_outcome_sha256": None,
            "proposal_sha256": proposal["sha256"],
            "selection_sha256": selection["sha256"],
            "live_renewal_closure_sha256": closure["sha256"],
            "prior_stage_records": copy.deepcopy(records),
            "mode": "no-alternative",
            "status": "no-alternative-abstention",
            "selected_ordinal": 0,
            "reserve": reserve,
            "selected_b2": None,
            "selected_score_advantage": 0.0,
            "exact_cost": 0,
            "restore_checks": 0,
            "rebind_count": 0,
            "complete": True,
        },
        root=root,
    )
    assert selection_stage["selection_sha256"] == selection["sha256"]
    assert ledger["complete"] is True
    return SimpleNamespace(
        query_root=query_root,
        source_sha=source_sha,
        seed=seed,
        index=index,
        query_index=query_index,
        policy_static=policy_static,
        g4=SimpleNamespace(one_rot_liability=lambda _level: reserve),
    )


def test_failed_command_lease_enter_cleans_global(
    runner, tmp_path: Path, monkeypatch
) -> None:
    root = (tmp_path / "root").resolve()
    root.mkdir()

    def fail_closure() -> None:
        raise RuntimeError("synthetic closure failure")

    monkeypatch.setattr(runner, "_validate_root_allowlist", fail_closure)
    with pytest.raises(RuntimeError, match="synthetic closure"):
        with runner._command_lease(root, enforce_root_closure=True):
            pass
    assert runner._ACTIVE_COMMAND_LEASE is None
    with runner._command_lease(root, enforce_root_closure=False) as lease:
        assert runner._ACTIVE_COMMAND_LEASE is lease
    assert runner._ACTIVE_COMMAND_LEASE is None


def test_command_lease_rejects_concurrent_flock(runner, tmp_path: Path) -> None:
    root = (tmp_path / "root").resolve()
    root.mkdir()
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another R3J command"):
            with runner._command_lease(root, enforce_root_closure=False):
                pass
        assert runner._ACTIVE_COMMAND_LEASE is None
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_command_lease_detects_root_replacement(runner, tmp_path: Path) -> None:
    root = (tmp_path / "root").resolve()
    detached = tmp_path / "detached-root"
    root.mkdir()
    with runner._command_lease(root, enforce_root_closure=False) as lease:
        root.rename(detached)
        root.mkdir()
        with pytest.raises(RuntimeError, match="ancestry was replaced"):
            lease.verify(check_closure=False)


def test_durable_reopen_uses_held_descendant_and_detects_replacement(
    runner, tmp_path: Path
) -> None:
    root = (tmp_path / "root").resolve()
    root.mkdir()
    query = root / "query-00"
    stage = query / "01-candidate-receipt.json"
    with runner._command_lease(root, enforce_root_closure=False) as lease:
        runner._mkdir_direct(query, root=root)
        records: dict[str, dict[str, str]] = {}
        original = runner._publish_query_stage(
            query,
            records,
            1,
            "candidate-receipt",
            {"schema": "unit-receipt", "candidate_count": 2},
            root=root,
        )
        lease.hold_directory(("query-00",))
        query.rename(root / "detached-query")
        query.mkdir()
        _write_artifact(
            runner,
            stage,
            "replacement",
            candidate_count=999,
        )
        assert runner._read_canonical_json(stage) == original
        with pytest.raises(RuntimeError, match="descendant directory was replaced"):
            lease.verify_descendants()


def test_mutation_and_scan_require_a_command_lease(
    runner, tmp_path: Path
) -> None:
    root = (tmp_path / "root").resolve()
    root.mkdir()
    target = root / "nested" / "artifact.json"
    with pytest.raises(RuntimeError, match="lacks its command lease"):
        runner._mkdir_direct(target.parent, root=root)
    with pytest.raises(RuntimeError, match="lacks its command lease"):
        runner._write_or_match(target, _artifact(runner), root=root)
    with pytest.raises(RuntimeError, match="lacks its command lease"):
        runner._safe_directory_entries(root, root=root)
    assert not target.exists()


def test_append_only_repair_requires_exact_prefix_and_complete_anchor(
    isolated,
) -> None:
    runner, root = isolated.runner, isolated.root
    anchor = root / "anchor.json"
    anchor_value = _artifact(runner, "anchor", complete=True)
    runner._write_or_match(anchor, anchor_value, root=root)

    target = root / "nested" / "artifact.json"
    runner._mkdir_direct(target.parent, root=root)
    value = _artifact(runner, "evidence", complete=True)
    payload = runner._canonical_bytes(value) + b"\n"
    target.write_bytes(payload[:17])
    digest = runner._write_or_match(
        target, value, root=root, recovery_anchor=anchor
    )
    assert target.read_bytes() == payload
    assert digest == hashlib.sha256(payload).hexdigest()
    with pytest.raises(FileExistsError):
        runner._write_new(target, value, root=root)

    divergent = root / "nested" / "divergent.json"
    divergent.write_bytes(b"not-a-prefix")
    with pytest.raises(RuntimeError, match="not an exact byte prefix"):
        runner._write_or_match(
            divergent, value, root=root, recovery_anchor=anchor
        )

    partial_anchor = root / "partial-anchor.json"
    partial_anchor.write_bytes(b"{")
    unanchored = root / "nested" / "unanchored.json"
    unanchored.write_bytes(payload[:11])
    with pytest.raises(RuntimeError, match="complete anchor"):
        runner._write_or_match(
            unanchored,
            value,
            root=root,
            recovery_anchor=partial_anchor,
        )


def test_append_only_publisher_rejects_oversize_prefix(
    isolated, monkeypatch
) -> None:
    runner, root = isolated.runner, isolated.root
    monkeypatch.setattr(runner, "MAXIMUM_ARTIFACT_BYTES", 8)
    target = root / "oversize.json"
    target.write_bytes(b"123456789")
    with pytest.raises(RuntimeError, match="oversized"):
        runner._write_or_match(target, _artifact(runner), root=root)
    with pytest.raises(RuntimeError, match="maximum size"):
        runner._canonical_artifact_state(target)


def test_recovery_anchor_map_and_exact_prefix_classifier(isolated) -> None:
    runner, root = isolated.runner, isolated.root
    assert runner._production_recovery_anchor(runner.INITIALIZE_INTENT) is None
    for path in (
        runner.SOURCE_IDENTITY,
        runner.SEED_COLLISION_PROOF,
        runner.PREFLIGHT,
        runner.CHECKPOINT_ADOPTION,
        runner.INITIALIZE_COMPLETION,
    ):
        assert (
            runner._production_recovery_anchor(path)
            == runner.INITIALIZE_INTENT
        )

    seed = runner.derive_seed(0)
    episode_root = runner.RUN_ROOT / "g4" / f"{seed:010d}"
    assert (
        runner._production_recovery_anchor(
            episode_root / "episode-trace.json"
        )
        == episode_root / "episode-intent.json"
    )
    assert (
        runner._production_recovery_anchor(episode_root / "episode.json")
        == episode_root / "episode-trace.json"
    )
    assert (
        runner._production_recovery_anchor(
            episode_root / "episode-completion.json"
        )
        == episode_root / "episode.json"
    )

    _write_artifact(runner, runner.INITIALIZE_INTENT, "initialize-intent")
    runner.SOURCE_IDENTITY.write_bytes(b"{")
    prefix = (
        runner.INITIALIZE_INTENT,
        runner.SOURCE_IDENTITY,
        runner.SEED_COLLISION_PROOF,
    )
    assert runner._require_exact_file_prefix(prefix) == 1
    _write_artifact(runner, runner.SEED_COLLISION_PROOF, "late")
    with pytest.raises(RuntimeError, match="out of order"):
        runner._require_exact_file_prefix(prefix)


def test_partial_episode_and_completion_are_recoverable_entries(
    isolated,
) -> None:
    runner = isolated.runner
    expectations = SimpleNamespace()
    for index, partial_name in ((0, "episode-completion.json"), (1, "episode.json")):
        seed = runner.derive_seed(index)
        episode_root = runner._episode_directory("control", seed)
        _write_artifact(
            runner, episode_root / "episode-intent.json", "episode-intent"
        )
        _write_artifact(
            runner, episode_root / "episode-trace.json", "episode-trace"
        )
        if partial_name == "episode-completion.json":
            _write_episode(runner, episode_root / "episode.json")
        (episode_root / partial_name).write_bytes(b"{")
        assert runner._load_episode("control", index, expectations) is None


def test_partial_trace_reenters_episode_execution(
    isolated, monkeypatch
) -> None:
    runner, root = isolated.runner, isolated.root
    source = _write_artifact(
        runner, runner.SOURCE_IDENTITY, "source-identity"
    )
    index = 0
    seed = runner.derive_seed(index)
    episode_root = runner._episode_directory("control", seed)
    intent = runner._with_sha(
        {
            "schema": "irisu-r3j-g4-episode-intent-v2",
            "source_identity_sha256": source["sha256"],
            "arm": "control",
            "index": index,
            "seed": seed,
            "horizon": runner.HORIZON,
            "schedule": {
                "query_shots": list(runner.QUERY_SHOTS),
                "maximum_queries": runner.MAXIMUM_QUERIES,
                "exact_budget": runner.BUDGET,
                "minimum_remaining_ticks": runner.EXACT_HORIZON,
            },
        }
    )
    episode_root.mkdir(parents=True)
    (episode_root / "episode-intent.json").write_bytes(
        runner._canonical_bytes(intent) + b"\n"
    )
    (episode_root / "episode-trace.json").write_bytes(b'{"arm":')

    original_write = runner._write_or_match

    def rooted_write(path, value, **kwargs):
        kwargs["root"] = root
        return original_write(path, value, **kwargs)

    class ReenteredEpisode(RuntimeError):
        pass

    monkeypatch.setattr(runner, "_write_or_match", rooted_write)
    monkeypatch.setattr(runner, "_validate_initialized", lambda _: None)
    monkeypatch.setattr(runner, "_load_runtime", lambda _: SimpleNamespace())
    monkeypatch.setattr(
        runner, "_verify_runtime_boundary", lambda runtime, expectations: None
    )

    def stop_at_reentry(event: str, **values: object) -> None:
        assert event == "episode-start"
        raise ReenteredEpisode

    monkeypatch.setattr(runner, "_progress", stop_at_reentry)
    with pytest.raises(ReenteredEpisode):
        runner.run_episode("control", index, SimpleNamespace())


def test_terminal_first_primitive_stops_the_whole_macro(runner) -> None:
    class ActionKind:
        WAIT = 0

        @staticmethod
        def parse(value: int) -> int:
            return value

    class Action:
        @staticmethod
        def wait(ticks: int):
            return SimpleNamespace(kind=ActionKind.WAIT, wait_ticks=ticks)

    campaign = SimpleNamespace(ActionKind=ActionKind, Action=Action)
    actions = [
        SimpleNamespace(kind=1, wait_ticks=1),
        SimpleNamespace(kind=1, wait_ticks=1),
    ]
    calls: list[object] = []

    def step(action):
        calls.append(action)
        return {"tick": 1}, 0.0, True, False, {}

    final, executed = runner._execute_primitive_units(
        actions,
        campaign=campaign,
        step_unit=step,
        remaining_ticks=lambda: 10,
        stopped=lambda: False,
    )
    assert final == {"tick": 1}
    assert executed == len(calls) == 1


def test_episode_query_directories_are_capped_and_contiguous(isolated) -> None:
    runner = isolated.runner
    for index, name, message in (
        (0, "query-04", "preregistered cap"),
        (1, "query-01", "contiguous prefix"),
    ):
        seed = runner.derive_seed(index)
        episode_root = runner._episode_directory("g4", seed)
        _write_artifact(
            runner, episode_root / "episode-intent.json", "episode-intent"
        )
        (episode_root / name).mkdir()
        with pytest.raises(RuntimeError, match=message):
            runner._validate_episode_prefix("g4", index)


def test_no_alternative_query_has_all_eleven_chained_stages_and_no_journal(
    isolated,
) -> None:
    runner, root = isolated.runner, isolated.root
    query_root = root / "query-00"
    records = _publish_no_alternative_query(runner, root, query_root)
    assert len(records) == len(runner.QUERY_STAGES) == 11
    assert runner._validate_query_prefix(query_root) == 11
    assert not (query_root / runner.EXACT_JOURNAL_DIRECTORY).exists()
    second = runner._read_canonical_json(
        runner._stage_path(query_root, 2, "prediction")
    )
    first_path = runner._stage_path(query_root, 1, "candidate-receipt")
    first = runner._read_canonical_json(first_path)
    assert second["previous_stage"] == {
        "name": "candidate-receipt",
        "artifact_sha256": first["sha256"],
        "file_sha256": runner._sha256_file(first_path),
    }


def test_exact_journal_cannot_precede_stage_three(
    isolated, monkeypatch
) -> None:
    runner, root = isolated.runner, isolated.root
    monkeypatch.setattr(runner, "_validate_exact_journal", lambda _: 2)
    monkeypatch.setattr(
        runner,
        "_validate_query_stage_artifacts",
        lambda query_root, count: [
            runner._read_canonical_json(
                runner._stage_path(query_root, order, name)
            )
            for order, name in enumerate(runner.QUERY_STAGES[:count], 1)
        ],
    )
    early = root / "query-00"
    _publish_stage_prefix(runner, root, early, 2, eligible=True)
    runner._mkdir_direct(
        early / runner.EXACT_JOURNAL_DIRECTORY, root=root
    )
    with pytest.raises(RuntimeError, match="anchored after query stage 3"):
        runner._validate_query_prefix(early)

    anchored = root / "query-01"
    _publish_stage_prefix(runner, root, anchored, 3, eligible=True)
    runner._mkdir_direct(
        anchored / runner.EXACT_JOURNAL_DIRECTORY, root=root
    )
    assert runner._validate_query_prefix(anchored) == 3


def test_exact_journal_allows_only_anchored_terminal_partial_intent(
    isolated,
) -> None:
    runner, root = isolated.runner, isolated.root
    missing_anchor = root / "query-missing-anchor"
    runner._mkdir_direct(
        missing_anchor / runner.EXACT_JOURNAL_DIRECTORY, root=root
    )
    runner._journal_path(missing_anchor, 0, "intent").write_bytes(b"{")
    with pytest.raises(RuntimeError, match="complete stage anchor"):
        runner._validate_exact_journal(missing_anchor)

    partial = root / "query-partial"
    _publish_stage_prefix(runner, root, partial, 3, eligible=True)
    runner._mkdir_direct(
        partial / runner.EXACT_JOURNAL_DIRECTORY, root=root
    )
    intent_path, intent = _write_exact_intent(runner, partial)
    encoded = runner._canonical_bytes(intent) + b"\n"
    intent_path.write_bytes(encoded[: len(encoded) // 2])
    assert runner._validate_exact_journal(partial) == 1

    runner._journal_path(partial, 0, "result").write_bytes(b"{")
    with pytest.raises(RuntimeError, match="out of order"):
        runner._validate_exact_journal(partial)

    partial_later = root / "query-partial-later"
    _publish_stage_prefix(
        runner, root, partial_later, 3, eligible=True
    )
    runner._mkdir_direct(
        partial_later / runner.EXACT_JOURNAL_DIRECTORY, root=root
    )
    later_intent, later_value = _write_exact_intent(
        runner, partial_later
    )
    later_payload = runner._canonical_bytes(later_value) + b"\n"
    later_intent.write_bytes(later_payload[: len(later_payload) // 2])
    _write_artifact(
        runner,
        runner._journal_path(partial_later, 1, "intent"),
        "foreign-later-intent",
    )
    with pytest.raises(RuntimeError, match="out of order"):
        runner._validate_exact_journal(partial_later)

    partial_result = root / "query-partial-result"
    _publish_stage_prefix(
        runner, root, partial_result, 3, eligible=True
    )
    runner._mkdir_direct(
        partial_result / runner.EXACT_JOURNAL_DIRECTORY, root=root
    )
    _write_exact_intent(runner, partial_result)
    runner._journal_path(partial_result, 0, "result").write_bytes(b"{")
    with pytest.raises(RuntimeError, match="ambiguous partial"):
        runner._validate_exact_journal(partial_result)

    unmatched = root / "query-unmatched"
    _publish_stage_prefix(runner, root, unmatched, 3, eligible=True)
    runner._mkdir_direct(
        unmatched / runner.EXACT_JOURNAL_DIRECTORY, root=root
    )
    intent_path, _ = _write_exact_intent(runner, unmatched)
    with pytest.raises(RuntimeError, match="unmatched durable intent"):
        runner._validate_exact_journal(unmatched)
    runner._ACTIVE_EXACT_INTENT = intent_path
    try:
        assert runner._validate_exact_journal(unmatched) == 1
    finally:
        runner._ACTIVE_EXACT_INTENT = None


def test_terminal_partial_intent_repairs_before_exact_once(
    isolated, monkeypatch
) -> None:
    runner, root = isolated.runner, isolated.root
    candidate = SimpleNamespace(
        manifest=lambda: {
            "ordinal": 0,
            "action": {"kind": "wait", "ticks": 1},
        }
    )
    identity = SimpleNamespace(
        ordinal=0,
        manifest=lambda: {"ordinal": 0, "candidate_id": "unit"},
    )
    policy_state_sha = "b" * 64
    runtime = SimpleNamespace(
        core=SimpleNamespace(),
        g4=SimpleNamespace(),
        live_lease=SimpleNamespace(
            canonical_policy_state_hash=lambda _: policy_state_sha
        ),
    )
    branch_manifest = {"schema": "unit-branch", "ordinal": 0}
    branch = SimpleNamespace(manifest=lambda: branch_manifest)
    exact_manifest = {"schema": "unit-exact", "ordinal": 0}
    exact = SimpleNamespace(
        manifest=lambda: exact_manifest,
        sha256=runner._canonical_sha256(exact_manifest),
    )
    raw = object()
    calls: list[object] = []

    def make_lease(name: str):
        query_root = root / name
        _publish_stage_prefix(runner, root, query_root, 3, eligible=True)
        runner._mkdir_direct(
            query_root / runner.EXACT_JOURNAL_DIRECTORY, root=root
        )
        anchor_path = runner._stage_path(
            query_root, 3, "preexact-plan"
        )
        anchor = runner._read_canonical_json(anchor_path)
        lease = runner.ExactProposalLease(
            runtime=runtime,
            env=object(),
            observation={},
            incumbent=candidate,
            full_candidates=(candidate,),
            inventory=SimpleNamespace(
                level=0, observation_sha256="c" * 64
            ),
            live_policy=object(),
            seed=1,
            query_id=name,
            split=runner.SPLIT,
            expected_snapshot_sha256="d" * 64,
            expected_state_hash="state",
            source_expectations=SimpleNamespace(),
            expected_policy_state_sha256=policy_state_sha,
            query_root=query_root,
            preexact_plan_reference={
                "name": "preexact-plan",
                "artifact_sha256": anchor["sha256"],
                "file_sha256": runner._sha256_file(anchor_path),
            },
        )
        lease.evaluator = object()
        lease.journal_chain = []
        lease.exact_certificates = {}
        return lease, query_root

    original_write = runner._write_or_match

    def rooted_write(path, value, **kwargs):
        kwargs["root"] = root
        kwargs["recovery_anchor"] = runner._stage_path(
            path.parent.parent, 3, "preexact-plan"
        )
        return original_write(path, value, **kwargs)

    def evaluate_shadow(self, supplied, base):
        calls.append(supplied)
        return (
            raw,
            _artifact(
                runner,
                "irisu-r3j-live-tau2-negative-witness-v1",
                status="failed",
            ),
            "shadow branch did not reach a second renewal",
        )

    monkeypatch.setattr(runner, "_write_or_match", rooted_write)
    monkeypatch.setattr(
        runner, "_verify_runtime_boundary", lambda *_: None
    )
    monkeypatch.setattr(
        runner.ExactProposalLease, "_evaluate_shadow", evaluate_shadow
    )
    monkeypatch.setattr(
        runner.ExactProposalLease,
        "_verify_bound_exact_evidence",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(runner, "_stage_raw", lambda *_a, **_k: branch)
    monkeypatch.setattr(runner, "_branch_to_g4", lambda *_a, **_k: exact)
    monkeypatch.setattr(
        runner,
        "_verify_recovered_exact_certificate",
        lambda *_a, **_k: None,
    )
    monkeypatch.setattr(
        runner, "_raw_exact_manifest", lambda _: {"raw": "unit"}
    )

    lease, query_root = make_lease("query-repair")
    expected = lease._intent(0, "incumbent", candidate, identity, None)
    payload = runner._canonical_bytes(expected) + b"\n"
    intent_path = runner._journal_path(query_root, 0, "intent")
    intent_path.write_bytes(payload[: len(payload) // 2])
    lease._load_or_evaluate(
        position=0,
        role="incumbent",
        candidate=candidate,
        identity=identity,
        base_candidate=candidate,
        incumbent_resolved=True,
        proposal_reference=None,
    )
    assert calls == [candidate]
    assert intent_path.read_bytes() == payload
    assert runner._validate_exact_journal(query_root) == 2

    divergent, divergent_root = make_lease("query-divergent")
    runner._journal_path(divergent_root, 0, "intent").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="not an exact byte prefix"):
        divergent._load_or_evaluate(
            position=0,
            role="incumbent",
            candidate=candidate,
            identity=identity,
            base_candidate=candidate,
            incumbent_resolved=True,
            proposal_reference=None,
        )
    assert calls == [candidate]


def test_exact_journal_rejects_result_first_extra_and_linked_duplicates(
    isolated,
) -> None:
    runner, root = isolated.runner, isolated.root
    result_first = root / "query-result-first"
    runner._mkdir_direct(
        result_first / runner.EXACT_JOURNAL_DIRECTORY, root=root
    )
    _write_artifact(
        runner,
        runner._journal_path(result_first, 0, "result"),
        "result",
    )
    with pytest.raises(RuntimeError, match="out of order"):
        runner._validate_exact_journal(result_first)

    extra = root / "query-extra"
    journal = extra / runner.EXACT_JOURNAL_DIRECTORY
    runner._mkdir_direct(journal, root=root)
    _write_journal_pair(runner, extra)
    _write_artifact(runner, journal / "foreign.json", "foreign")
    with pytest.raises(RuntimeError, match="foreign entries"):
        runner._validate_exact_journal(extra)

    linked = root / "query-linked"
    journal = linked / runner.EXACT_JOURNAL_DIRECTORY
    runner._mkdir_direct(journal, root=root)
    intent = runner._journal_path(linked, 0, "intent")
    result = runner._journal_path(linked, 0, "result")
    _write_artifact(runner, intent, "intent")
    os.link(intent, result)
    with pytest.raises(RuntimeError, match="foreign entries"):
        runner._validate_exact_journal(linked)


def test_query_prefix_rejects_partial_beyond_proposal_and_after_stage6(
    isolated, monkeypatch
) -> None:
    runner, root = isolated.runner, isolated.root
    stage_sets: dict[Path, list[dict[str, object]]] = {}
    journal_counts: dict[Path, int] = {}

    def prepare(
        name: str,
        stage_count: int,
        identities: list[dict[str, object]],
        journal_count: int,
    ) -> Path:
        query_root = root / name
        _publish_stage_prefix(
            runner, root, query_root, stage_count, eligible=True
        )
        runner._mkdir_direct(
            query_root / runner.EXACT_JOURNAL_DIRECTORY, root=root
        )
        intent = _write_artifact(
            runner,
            runner._journal_path(query_root, 0, "intent"),
            "unit-intent",
        )
        result = _write_artifact(
            runner,
            runner._journal_path(query_root, 0, "result"),
            "unit-result",
            branch={"ordinal": 0},
            exact_outcome={"ordinal": 0},
            exact_outcome_sha256=runner._canonical_sha256({"ordinal": 0}),
        )
        chain0 = {
            "position": 0,
            "intent": {
                "artifact_sha256": intent["sha256"],
                "file_sha256": runner._sha256_file(
                    runner._journal_path(query_root, 0, "intent")
                ),
            },
            "result": {
                "artifact_sha256": result["sha256"],
                "file_sha256": runner._sha256_file(
                    runner._journal_path(query_root, 0, "result")
                ),
            },
        }
        stages: list[dict[str, object]] = [
            {"g4_reconstruction": {"eligible": True}},
            {},
            {},
            {
                "exact_journal": chain0,
                "branch": result["branch"],
                "exact_outcome": result["exact_outcome"],
                "exact_outcome_sha256": result["exact_outcome_sha256"],
            },
            {
                "proposal": {
                    "effective_budget": len(identities),
                    "identities": identities,
                }
            },
            {},
        ][:stage_count]
        stage_sets[query_root] = stages
        journal_counts[query_root] = journal_count
        return query_root

    monkeypatch.setattr(
        runner,
        "_validate_query_stage_artifacts",
        lambda query_root, _count: stage_sets[query_root],
    )
    monkeypatch.setattr(
        runner,
        "_validate_exact_journal",
        lambda query_root: journal_counts[query_root],
    )

    beyond = prepare("query-beyond", 5, [], 3)
    with pytest.raises(RuntimeError, match="escaped the formal proposal"):
        runner._validate_query_prefix(beyond)

    after_stage6 = prepare(
        "query-after-stage6",
        6,
        [{"ordinal": 1}],
        5,
    )
    with pytest.raises(RuntimeError):
        runner._validate_query_prefix(after_stage6)


def _synthetic_trace(runner, source_sha: str) -> dict[str, object]:
    seed = runner.derive_seed(0)
    intent_sha = "b" * 64
    initial = {
        "tick": 10,
        "score": 100,
        "gauge": 80,
        "level": 1,
        "qualifying_clear_count": 2,
    }
    genesis = runner._trace_genesis(
        source_identity_sha256=source_sha,
        episode_intent_sha256=intent_sha,
        arm="control",
        index=0,
        seed=seed,
        runner={"config_hash": 7, "runtime": "portable"},
        config_hash=7,
        reset_info={"seed": seed, "config_hash": 7},
        initial_observation=initial,
        initial_state_sha256="c" * 64,
    )
    previous = runner._canonical_sha256(genesis)
    prior_public = initial
    rows: list[dict[str, object]] = []
    for ordinal in range(2):
        current = {**prior_public, "tick": 11 + ordinal, "score": 110 + ordinal}
        info = {"config_hash": 7, "events": []}
        row = runner._with_sha(
            {
                "schema": "irisu-r3j-g4-unit-trace-row-v1",
                "ordinal": ordinal,
                "previous_sha256": previous,
                "action": {
                    "kind": 0,
                    "cursor_x": 0.0,
                    "cursor_y": 0.0,
                    "wait_ticks": 1,
                },
                "query_context": {
                    "source": "frozen-v5-incumbent",
                    "query_id": None,
                    "query_index": None,
                    "query_ledger_sha256": None,
                    "decision_sha256": "d" * 64,
                },
                "pre_observation_sha256": runner._canonical_sha256(prior_public),
                "post_observation_sha256": runner._canonical_sha256(current),
                "reward": 1.0,
                "info": info,
                "info_sha256": runner._canonical_sha256(info),
                "events": [],
                "terminated": False,
                "truncated": False,
                "config_hash": 7,
                "state_sha256": f"{ordinal + 1:x}" * 64,
            }
        )
        rows.append(row)
        previous = row["sha256"]
        prior_public = current
    return runner._with_sha(
        {
            "schema": "irisu-r3j-g4-unit-trace-v1",
            "complete": True,
            "development_only": True,
            "sealed_test_allowed": False,
            "episode_intent_sha256": intent_sha,
            "arm": "control",
            "index": 0,
            "seed": seed,
            "genesis": genesis,
            "genesis_sha256": runner._canonical_sha256(genesis),
            "rows": rows,
            "row_count": len(rows),
            "final_row_sha256": rows[-1]["sha256"],
            "final_public_observation": prior_public,
            "final_public_observation_sha256": runner._canonical_sha256(
                prior_public
            ),
            "final_state_sha256": rows[-1]["state_sha256"],
            "episode_metadata": {
                "decisions": 2,
                "seen_shots": 0,
                "queries": [],
                "skipped_query_slots": [],
            },
        }
    )


def test_closed_schema_self_hash_link_and_trace_chronology(isolated) -> None:
    runner, root = isolated.runner, isolated.root
    source = _write_artifact(
        runner, runner.SOURCE_IDENTITY, "source-identity"
    )
    trace = _synthetic_trace(runner, str(source["sha256"]))
    assert (
        runner._validate_episode_trace(
            trace,
            arm="control",
            index=0,
            intent_sha256="b" * 64,
        )
        == trace["sha256"]
    )

    bad = copy.deepcopy(trace)
    row = dict(bad["rows"][1])
    row.pop("sha256")
    row["previous_sha256"] = "f" * 64
    bad["rows"][1] = runner._with_sha(row)
    bad["final_row_sha256"] = bad["rows"][1]["sha256"]
    bad = runner._with_sha(
        {key: value for key, value in bad.items() if key != "sha256"}
    )
    with pytest.raises(RuntimeError, match="row 1 differs"):
        runner._validate_episode_trace(
            bad,
            arm="control",
            index=0,
            intent_sha256="b" * 64,
        )

    tampered = dict(source)
    tampered["schema"] = "tampered"
    with pytest.raises(RuntimeError, match="self-hash differs"):
        runner._validate_self_hash(tampered, "tampered")
    with pytest.raises(RuntimeError, match="non-closed schema"):
        runner._require_exact_keys({"only": 1, "extra": 2}, {"only"}, "row")

    linked = root / "linked.json"
    _write_artifact(runner, linked, "linked")
    os.link(linked, root / "linked-again.json")
    with pytest.raises(RuntimeError, match="single-link"):
        runner._canonical_artifact_state(linked)


def test_query_ledger_binding_is_contiguous_and_rehashes_rows(runner) -> None:
    def row(previous: str, source: str) -> dict[str, object]:
        return {
            "previous_sha256": previous,
            "query_context": {
                "source": source,
                "query_ledger_sha256": None,
            },
            "sha256": "a" * 64,
        }

    bad = [
        row("0" * 64, "g4-query-selection"),
        row("a" * 64, "frozen-v5-incumbent"),
    ]
    with pytest.raises(RuntimeError, match="context differs"):
        runner._bind_trace_query_ledger(bad, 0, "f" * 64)

    rows = [
        row("0" * 64, "g4-query-selection"),
        row("a" * 64, "g4-query-selection"),
    ]
    bound = runner._bind_trace_query_ledger(rows, 0, "f" * 64)
    assert bound["row_count"] == 2
    assert rows[1]["previous_sha256"] == rows[0]["sha256"]
    assert all(
        value["query_context"]["query_ledger_sha256"] == "f" * 64
        for value in rows
    )


def test_skip_records_are_distinct_zero_work_slots(runner) -> None:
    source = inspect.getsource(runner.run_episode)
    assert runner.QUERY_SHOTS == (1, 7, 13, 19)
    assert '"reason": "insufficient-live-tau2-horizon"' in source
    assert '"reason": "live-exclusion-suppressed"' in source
    assert source.count('"generator_calls": 0') >= 2
    assert source.count('"model_calls": 0') >= 2
    assert source.count('"exact_cost": 0') >= 2
    assert source.count('"remaining_ticks": remaining()') >= 2


def test_module_loader_rejects_same_bytes_at_a_drifted_path(
    isolated, monkeypatch
) -> None:
    runner, root = isolated.runner, isolated.root
    expected = root / "expected.py"
    drifted = root / "drifted.py"
    payload = b"VALUE = 1\n"
    expected.write_bytes(payload)
    drifted.write_bytes(payload)
    name = "_r3j_hardening_path_drift"
    monkeypatch.setitem(
        sys.modules, name, SimpleNamespace(__file__=str(drifted))
    )
    with pytest.raises(RuntimeError, match="foreign module"):
        runner._load_module(
            name,
            expected,
            hashlib.sha256(payload).hexdigest(),
        )


def test_verifier_lease_rejects_concurrent_writer(
    verifier, tmp_path: Path
) -> None:
    root = (tmp_path / "verification-root").resolve()
    root.mkdir()
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(
            verifier.VerificationError, match="held by another command"
        ):
            with verifier.VerificationLease(root):
                pass
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
    assert verifier._ACTIVE_VERIFICATION_LEASE is None


@pytest.mark.parametrize("mutation", ["root", "descendant", "file-aba"])
def test_verifier_lease_rejects_namespace_replacement_and_file_aba(
    verifier, tmp_path: Path, mutation: str
) -> None:
    root = (tmp_path / f"verification-{mutation}").resolve()
    nested = root / "nested"
    target = nested / "evidence.json"
    nested.mkdir(parents=True)
    target.write_bytes(b"same immutable bytes\n")
    detached = tmp_path / f"detached-{mutation}"

    with pytest.raises(verifier.VerificationError):
        with verifier.VerificationLease(root):
            assert verifier.file_bytes(target) == b"same immutable bytes\n"
            if mutation == "root":
                root.rename(detached)
                root.mkdir()
            elif mutation == "descendant":
                nested.rename(detached)
                nested.mkdir()
            else:
                target.rename(detached)
                target.write_bytes(b"same immutable bytes\n")
    assert verifier._ACTIVE_VERIFICATION_LEASE is None


def _synthetic_numeric_closure(verifier) -> dict[str, object]:
    site = verifier.VENV_SITE_PACKAGES.resolve()
    module_path = (site / "numpy/__init__.py").resolve()
    digest = "a" * 64
    modules = {
        name: {"path": str(module_path), "sha256": digest}
        for name in (
            "numpy",
            "numpy._core._multiarray_umath",
            "torch",
            "torch._C",
        )
    }
    distributions = {
        name: {
            "version": "unit",
            "dist_info": str(site),
            "files": {
                filename: {"path": str(module_path), "sha256": digest}
                for filename in ("metadata", "record")
            },
        }
        for name in ("numpy", "torch")
    }
    mapped_path = Path(sys.executable).resolve()
    metadata = mapped_path.stat()
    device = (
        f"{os.major(metadata.st_dev):02x}:"
        f"{os.minor(metadata.st_dev):02x}"
    )
    body = {
        "schema": "irisu-r3j-numeric-runtime-closure-v2",
        "interpreter": {
            "path": str(mapped_path),
            "sha256": digest,
            "version": sys.version,
            "cache_tag": sys.implementation.cache_tag,
        },
        "preclosure_imports": verifier.preclosure_import_manifest(),
        "modules": modules,
        "distributions": distributions,
        "loaded_file_mappings": [
            {
                "path": str(mapped_path),
                "maps_path": str(mapped_path),
                "maps_device": device,
                "maps_inode": metadata.st_ino,
                "live_device": device,
                "live_inode": metadata.st_ino,
                "map_files_target": str(mapped_path),
                "map_files_errno": None,
                "sha256": digest,
            }
        ],
    }
    return {**body, "sha256": verifier.canonical_sha256(body)}


def test_verifier_rejects_raw_and_canonical_mapping_tamper(
    verifier, monkeypatch
) -> None:
    def fixed_digest(path: Path, expected: str | None = None) -> str:
        return expected or "a" * 64

    monkeypatch.setattr(verifier, "file_sha256", fixed_digest)
    closure = _synthetic_numeric_closure(verifier)
    assert (
        verifier.validate_numeric_runtime_closure(closure, "synthetic")
        == closure
    )

    raw_tamper = copy.deepcopy(closure)
    raw_tamper["loaded_file_mappings"][0]["maps_path"] += "-alias"
    raw_body = {key: value for key, value in raw_tamper.items() if key != "sha256"}
    raw_tamper["sha256"] = verifier.canonical_sha256(raw_body)
    with pytest.raises(
        verifier.VerificationError, match="mapping 0 differs"
    ):
        verifier.validate_numeric_runtime_closure(raw_tamper, "raw tamper")

    canonical_tamper = copy.deepcopy(closure)
    canonical = canonical_tamper["loaded_file_mappings"][0]["path"]
    canonical_tamper["loaded_file_mappings"][0]["path"] = (
        str(Path(canonical).parent / ".." / Path(canonical).parent.name)
        + f"/{Path(canonical).name}"
    )
    canonical_body = {
        key: value for key, value in canonical_tamper.items() if key != "sha256"
    }
    canonical_tamper["sha256"] = verifier.canonical_sha256(canonical_body)
    with pytest.raises(
        verifier.VerificationError, match="mapping 0 differs"
    ):
        verifier.validate_numeric_runtime_closure(
            canonical_tamper, "canonical tamper"
        )

    encoding_tamper = copy.deepcopy(closure)
    encoding_row = encoding_tamper["loaded_file_mappings"][0]
    encoding_row["map_files_target"] = None
    encoding_row["map_files_errno"] = "EPERM"
    encoding_body = {
        key: value for key, value in encoding_tamper.items() if key != "sha256"
    }
    encoding_tamper["sha256"] = verifier.canonical_sha256(encoding_body)
    with pytest.raises(
        verifier.VerificationError, match="mapping 0 differs"
    ):
        verifier.validate_numeric_runtime_closure(
            encoding_tamper, "encoding tamper"
        )

    key_tamper = copy.deepcopy(closure)
    key_tamper["loaded_file_mappings"][0]["map_files_readlink"] = "verified"
    key_body = {key: value for key, value in key_tamper.items() if key != "sha256"}
    key_tamper["sha256"] = verifier.canonical_sha256(key_body)
    with pytest.raises(
        verifier.VerificationError, match="foreign schema"
    ):
        verifier.validate_numeric_runtime_closure(key_tamper, "key tamper")


def test_runner_verifier_mapping_rows_are_byte_exact(runner, verifier) -> None:
    runner_imports = runner._preclosure_import_manifest()
    verifier_imports = verifier.preclosure_import_manifest()
    assert verifier_imports == runner_imports
    assert verifier.canonical_bytes(verifier_imports) == runner._canonical_bytes(
        runner_imports
    )
    runner_rows = runner._mapped_regular_files()
    verifier_rows = verifier.mapped_regular_file_paths()
    assert verifier_rows == runner_rows
    assert verifier.canonical_bytes(verifier_rows) == runner._canonical_bytes(
        runner_rows
    )
    expected_keys = {
        "path",
        "maps_path",
        "maps_device",
        "maps_inode",
        "live_device",
        "live_inode",
        "map_files_target",
        "map_files_errno",
    }
    assert all(set(row) == expected_keys for row in verifier_rows)
    assert all(
        row["map_files_errno"] is None
        or type(row["map_files_errno"]) is int
        for row in verifier_rows
    )


def test_verifier_v2_schedule_allows_suppressed_then_later_query(
    verifier,
) -> None:
    skip = {
        "shot_index": 1,
        "tick": 100,
        "remaining_ticks": 1_900,
        "reason": "live-exclusion-suppressed",
        "generator_calls": 0,
        "model_calls": 0,
        "exact_cost": 0,
    }
    metadata = {
        "decisions": 42,
        "seen_shots": 7,
        "queries": [{"query_index": 0, "shot_index": 7}],
        "skipped_query_slots": [skip],
    }
    verifier.validate_replayed_schedule_partition(
        metadata=metadata,
        expected_skips=[skip],
        actual_query_shots=[7],
        decisions=42,
        seen_shots=7,
    )
    source = inspect.getsource(verifier.validate_replayed_schedule_partition)
    assert "6 *" not in source
    assert "1 +" not in source

    missing_query_index = copy.deepcopy(metadata)
    missing_query_index["queries"][0].pop("query_index")
    with pytest.raises(
        verifier.VerificationError, match="query chronology"
    ):
        verifier.validate_replayed_schedule_partition(
            metadata=missing_query_index,
            expected_skips=[skip],
            actual_query_shots=[7],
            decisions=42,
            seen_shots=7,
        )


def _verifier_cashflow(tick: int, *, renewal: bool) -> dict[str, object]:
    return {
        "tick": tick,
        "level": 1,
        "entry_gauge": 101 - tick,
        "gross_renewable_gain": 10 if renewal else 0,
        "net_post_clamp_gain": 10 if renewal else 0,
        "drain": 1,
        "rot_penalty": 0,
        "exit_gauge": 100 - tick,
        "public_gauge": 100 - tick,
        "renewal": renewal,
        "active_liability_ids": [],
    }


def _verifier_ledger(
    verifier,
    cashflow: list[dict[str, object]],
    *,
    unresolved: list[str] | None = None,
    checkpoints: list[int] | None = None,
) -> dict[str, object]:
    renewals = [row["tick"] for row in cashflow if row["renewal"]][:2]
    selected = range(len(cashflow)) if checkpoints is None else checkpoints
    return {
        "b2_margin": 7 if len(renewals) == 2 else None,
        "renewal_ticks": renewals,
        "renewals_resolved": len(renewals) == 2,
        "liabilities": [],
        "liability_count": 0,
        "paid_liability_count": 0,
        "cancelled_liability_count": 0,
        "active_liability_ids": [],
        "duplicate_payment_attempts": 0,
        "unresolved": list(unresolved or []),
        "minimum_public_gauge": 100 - len(cashflow),
        "cashflow_lost": False,
        "cashflow_lost_tick": None,
        "cashflow_count": len(cashflow),
        "cashflow_sha256": verifier.canonical_sha256(cashflow),
        "cashflow_checkpoints": [copy.deepcopy(cashflow[index]) for index in selected],
    }


def _rehash_verifier(verifier, value: dict[str, object]) -> dict[str, object]:
    body = copy.deepcopy(value)
    body.pop("sha256", None)
    return {**body, "sha256": verifier.canonical_sha256(body)}


def _verifier_exact_evidence(verifier, *, negative: bool):
    start = {
        "tick": 0,
        "score": 10,
        "gauge": 100,
        "level": 1,
        "terminated": False,
        "truncated": False,
    }
    publics = [
        {
            **start,
            "tick": tick,
            "score": 10 + 10 * tick,
            "gauge": 100 - tick,
        }
        for tick in (1, 2, 3)
    ]
    cashflow = [
        _verifier_cashflow(tick, renewal=tick <= 2)
        for tick in (1, 2, 3)
    ]
    rows = []
    root = "0" * 64
    previous_public = start
    for index, current_public in enumerate(publics):
        body = {
            "index": index,
            "previous_trace_sha256": root,
            "relative_tick": index + 1,
            "action": {
                "type": "irisu_env.env.Action",
                "fields": {
                    "kind": 0,
                    "cursor_x": 0.0,
                    "cursor_y": 0.0,
                    "wait_ticks": 1,
                },
            },
            "previous_public_sha256": verifier.canonical_sha256(
                previous_public
            ),
            "current_public_sha256": verifier.canonical_sha256(
                current_public
            ),
            "events": [],
            "cashflow": cashflow[index],
            "terminated": False,
            "truncated": False,
        }
        row = {**body, "sha256": verifier.canonical_sha256(body)}
        rows.append(row)
        root = row["sha256"]
        previous_public = current_public
    trace_body = {
        "schema": "irisu-r3j-live-tau2-trace-v1-unit",
        "start_tick": 0,
        "end_tick": 3,
        "rows": rows,
    }
    trace = {
        **trace_body,
        "sha256": verifier.canonical_sha256(trace_body),
    }
    tau2_ledger = _verifier_ledger(
        verifier,
        cashflow[:2],
        unresolved=["overdue-liabilities:1"] if negative else [],
        checkpoints=[0, 1],
    )
    full_ledger = _verifier_ledger(
        verifier, cashflow, unresolved=[], checkpoints=[0, 2]
    )
    raw = {
        "candidate": {"ordinal": 0},
        "feature_vector": [0.25],
        "ledger": copy.deepcopy(full_ledger),
        "survival_ticks": 3,
        "score_gain": 30,
        "final_gauge": 97,
        "final_level": 1,
        "terminal": False,
        "gauge_failure": False,
        "invalid_actions": 0,
        "continuation_rebind_failed": False,
    }
    common = {
        "start_tick": 0,
        "end_tick": 3,
        "elapsed_ticks": 3,
        "horizon_ticks": verifier.EXACT_HORIZON,
        "start_public_observation": start,
        "end_public_observation": publics[-1],
        "start_public_sha256": verifier.canonical_sha256(start),
        "end_public_sha256": verifier.canonical_sha256(publics[-1]),
        "unit_trace_root_sha256": rows[-1]["sha256"],
        "unit_trace_sha256": trace["sha256"],
        "unit_trace": trace,
        "tau2": {
            "unit_index": 1,
            "relative_tick": 2,
            "unit_trace_root_sha256": rows[1]["sha256"],
            "public_sha256": rows[1]["current_public_sha256"],
            "b2_margin": 7,
            "renewal_ticks": [1, 2],
        },
        "tau2_ledger": tau2_ledger,
        "full_ledger": full_ledger,
        "full_cashflow_sha256": full_ledger["cashflow_sha256"],
        "raw_exact_sha256": verifier.canonical_sha256(raw),
    }
    if negative:
        body = {
            "schema": "irisu-r3j-live-tau2-negative-witness-v1",
            "status": "failed",
            "certificate_mode": "exact-shadow-terminal-failure",
            "failure_reason": "shadow branch has unresolved liability evidence",
            **common,
        }
    else:
        body = {
            "schema": "irisu-r3j-live-tau2-certificate-v1",
            "status": "closed",
            "certificate_mode": "exact-shadow-full-branch",
            **common,
            "cashflow_sha256": tau2_ledger["cashflow_sha256"],
            "b2_margin": 7,
            "renewal_ticks": [1, 2],
        }
    return raw, {**body, "sha256": verifier.canonical_sha256(body)}


def test_verifier_sparse_raw_ledger_then_full_cashflow_binding(verifier) -> None:
    cashflow = [
        _verifier_cashflow(tick, renewal=tick <= 2)
        for tick in (1, 2, 3)
    ]
    sparse = _verifier_ledger(verifier, cashflow, checkpoints=[0, 2])
    assert verifier.verify_cashflow_ledger(
        sparse, "sparse recovered raw", expected_cashflow=None
    ) == sparse
    assert verifier.verify_cashflow_ledger(
        sparse, "certificate full", expected_cashflow=cashflow
    ) == sparse
    foreign = copy.deepcopy(sparse)
    foreign["cashflow_checkpoints"][0]["public_gauge"] = 999
    with pytest.raises(verifier.VerificationError, match="checkpoint"):
        verifier.verify_cashflow_ledger(
            foreign, "foreign certificate", expected_cashflow=cashflow
        )


def test_verifier_positive_certificate_binds_raw_and_rebind(verifier) -> None:
    raw, certificate = _verifier_exact_evidence(verifier, negative=False)
    verified = verifier.verify_tau2_certificate(
        certificate, "positive", mode="exact"
    )
    verifier.verify_exact_raw_terminal_binding(
        verified,
        verified["_unit_rows"],
        raw,
        "positive",
        expected_rebind_failed=False,
        game_over_kind=14,
        invalid_action_kind=0,
    )

    tampered_raw = copy.deepcopy(raw)
    tampered_raw["score_gain"] += 1
    tampered_certificate = copy.deepcopy(certificate)
    tampered_certificate["raw_exact_sha256"] = verifier.canonical_sha256(
        tampered_raw
    )
    tampered_certificate = _rehash_verifier(verifier, tampered_certificate)
    verified_tamper = verifier.verify_tau2_certificate(
        tampered_certificate, "positive tamper", mode="exact"
    )
    with pytest.raises(verifier.VerificationError, match="raw terminal"):
        verifier.verify_exact_raw_terminal_binding(
            verified_tamper,
            verified_tamper["_unit_rows"],
            tampered_raw,
            "positive tamper",
            expected_rebind_failed=False,
            game_over_kind=14,
            invalid_action_kind=0,
        )

    rebound_raw = copy.deepcopy(raw)
    rebound_raw["continuation_rebind_failed"] = True
    rebound_raw["ledger"]["b2_margin"] = None
    rebound_raw["ledger"]["unresolved"].append(
        "continuation-rebind-failed"
    )
    verifier.verify_exact_raw_terminal_binding(
        verified,
        verified["_unit_rows"],
        rebound_raw,
        "positive rebound",
        expected_rebind_failed=True,
        game_over_kind=14,
        invalid_action_kind=0,
    )
    with pytest.raises(verifier.VerificationError, match="raw terminal"):
        verifier.verify_exact_raw_terminal_binding(
            verified,
            verified["_unit_rows"],
            rebound_raw,
            "unproved rebound",
            expected_rebind_failed=False,
            game_over_kind=14,
            invalid_action_kind=0,
        )


def test_verifier_negative_witness_freezes_tau2_and_rejects_raw_flags(
    verifier,
) -> None:
    raw, witness = _verifier_exact_evidence(verifier, negative=True)
    verified = verifier.verify_negative_exact_witness(
        witness,
        "negative",
        expected_reason="shadow branch has unresolved liability evidence",
        raw=raw,
        expected_rebind_failed=False,
        game_over_kind=14,
        invalid_action_kind=0,
    )
    assert verified["tau2_ledger"]["unresolved"]
    assert verified["full_ledger"]["unresolved"] == []

    fabricated_raw = copy.deepcopy(raw)
    fabricated_raw["terminal"] = True
    fabricated = copy.deepcopy(witness)
    fabricated["failure_reason"] = (
        "shadow branch has structural failure evidence"
    )
    fabricated["raw_exact_sha256"] = verifier.canonical_sha256(
        fabricated_raw
    )
    fabricated = _rehash_verifier(verifier, fabricated)
    with pytest.raises(verifier.VerificationError, match="raw terminal"):
        verifier.verify_negative_exact_witness(
            fabricated,
            "fabricated structural",
            expected_reason="shadow branch has structural failure evidence",
            raw=fabricated_raw,
            expected_rebind_failed=False,
            game_over_kind=14,
            invalid_action_kind=0,
        )

    cleared = copy.deepcopy(witness)
    cleared["tau2_ledger"]["unresolved"] = []
    cleared = _rehash_verifier(verifier, cleared)
    with pytest.raises(
        verifier.VerificationError, match="successful exact branch"
    ):
        verifier.verify_negative_exact_witness(
            cleared,
            "cleared tau2",
            expected_reason="shadow branch has unresolved liability evidence",
            raw=raw,
            expected_rebind_failed=False,
            game_over_kind=14,
            invalid_action_kind=0,
        )


def test_runner_exact_evidence_rejects_transplanted_start_observation(
    runner, verifier
) -> None:
    raw, witness = _verifier_exact_evidence(verifier, negative=True)
    candidate = SimpleNamespace(
        decision=object(), manifest=lambda: {"ordinal": 0}
    )
    lease = runner.ExactProposalLease(
        runtime=SimpleNamespace(),
        env=object(),
        observation={
            **witness["start_public_observation"],
            "score": witness["start_public_observation"]["score"] + 1,
        },
        incumbent=candidate,
        full_candidates=(candidate,),
        inventory=object(),
        live_policy=object(),
        seed=1,
        query_id="unit",
        split=runner.SPLIT,
        expected_snapshot_sha256="a" * 64,
        expected_state_hash="state",
        source_expectations=object(),
        expected_policy_state_sha256="b" * 64,
        query_root=Path("/tmp/unit-query"),
        preexact_plan_reference={
            "name": "preexact-plan",
            "artifact_sha256": "c" * 64,
            "file_sha256": "d" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="query binding"):
        lease._verify_bound_exact_evidence(
            witness,
            raw,
            witness["failure_reason"],
            candidate,
            candidate,
        )


def test_runner_bound_verifier_requires_external_expectations(
    isolated, monkeypatch
) -> None:
    runner = isolated.runner
    monkeypatch.setattr(runner, "_BOUND_VERIFIER_CACHE", None)
    with pytest.raises(RuntimeError, match="external command expectations"):
        runner._load_bound_verifier()


def test_prewrite_semantic_replay_rejects_coherently_rehashed_ledger_poison(
    isolated, verifier, monkeypatch
) -> None:
    runner, root = isolated.runner, isolated.root
    fixture = _publish_verifier_no_alternative_query(
        runner,
        verifier,
        root,
        root / "semantic-poison" / "query-00",
    )
    monkeypatch.setattr(verifier, "derive_seed", lambda _index: fixture.seed)
    stages = verifier.verify_stage_chain(fixture.query_root)
    selection = stages[6]["selection"]
    ledger_path = verifier.stage_path(fixture.query_root, 11, "ledger")
    poisoned_ledger = copy.deepcopy(stages[10])
    poisoned_ledger["exact_cost"] = 1
    poisoned_ledger = _rehash_verifier(verifier, poisoned_ledger)
    ledger_path.write_bytes(verifier.canonical_bytes(poisoned_ledger) + b"\n")
    supplied = {
        "query_id": poisoned_ledger["query_id"],
        "query_index": 0,
        "shot_index": poisoned_ledger["shot_index"],
        "ledger_sha256": poisoned_ledger["sha256"],
        "ledger_file_sha256": verifier.file_sha256(ledger_path),
        "mode": poisoned_ledger["mode"],
        "status": poisoned_ledger["status"],
        "selected_ordinal": poisoned_ledger["selected_ordinal"],
        "candidate_count": 1,
        "exact_cost": 1,
        "restore_checks": poisoned_ledger["restore_checks"],
        "reserve": poisoned_ledger["reserve"],
        "selected_b2": poisoned_ledger["selected_b2"],
        "selected_score_advantage": poisoned_ledger[
            "selected_score_advantage"
        ],
        "tau2_lease_opened": False,
        "tau2_lease_closed": False,
        "trace_range": {
            "start_ordinal": 0,
            "end_ordinal": 0,
            "row_count": 1,
            "final_row_sha256": "e" * 64,
            "query_ledger_sha256": poisoned_ledger["sha256"],
        },
    }
    trace_rows = [
        {
            "query_context": {
                "source": "g4-query-selection",
                "query_id": poisoned_ledger["query_id"],
                "query_index": 0,
                "query_ledger_sha256": poisoned_ledger["sha256"],
                "decision_sha256": "f" * 64,
            }
        }
    ]
    runtime = SimpleNamespace(
        g4=fixture.g4,
        model=object(),
        core=object(),
        policy_static=fixture.policy_static,
    )
    with pytest.raises(verifier.VerificationError, match="ledger differs"):
        verifier.verify_episode_query_evidence(
            runtime=runtime,
            arm="g4",
            index=fixture.index,
            source_identity_sha256=fixture.source_sha,
            trace_rows=trace_rows,
            state_by_tick={},
            query_roots=[fixture.query_root],
            supplied_queries=[supplied],
            replayed_queries=[
                {
                    "query_id": poisoned_ledger["query_id"],
                    "selection": selection,
                    "selected_exact_certificate": None,
                }
            ],
        )


def test_runner_recovered_trace_extra_query_rejects_before_replay(
    isolated, monkeypatch
) -> None:
    runner = isolated.runner
    _write_artifact(runner, runner.SOURCE_IDENTITY, "source-identity")
    seed = runner.derive_seed(0)
    episode_root = runner._episode_directory("control", seed)
    runner._mkdir_direct(episode_root, root=isolated.root)
    runner._mkdir_direct(episode_root / "query-00", root=isolated.root)
    calls: list[str] = []

    fake_verifier = SimpleNamespace(
        replay_episode_trace=lambda *_a, **_k: (
            {"episode_metadata": {"queries": []}},
            {},
            {},
            [],
        ),
        verify_replayed_query_schedule=lambda **_k: calls.append("schedule"),
        verify_episode_query_evidence=lambda **_k: calls.append("semantic"),
    )
    monkeypatch.setattr(
        runner,
        "_configured_bound_episode_verifier",
        lambda _runtime: fake_verifier,
    )
    with pytest.raises(RuntimeError, match="full no-exact prewrite"):
        runner._verify_recovered_trace_before_episode_write(
            SimpleNamespace(),
            arm="control",
            index=0,
            intent={"sha256": "a" * 64},
            trace_file_sha256="b" * 64,
        )
    assert calls == []


def test_recovered_trace_failure_precedes_episode_and_completion_writes(
    isolated, monkeypatch
) -> None:
    runner = isolated.runner
    source = _write_artifact(
        runner, runner.SOURCE_IDENTITY, "source-identity"
    )
    seed = runner.derive_seed(0)
    episode_root = runner._episode_directory("control", seed)
    runner._mkdir_direct(episode_root, root=isolated.root)
    trace = _artifact(runner, "unit-complete-trace", complete=True)
    runner._episode_trace_path("control", seed).write_bytes(
        runner._canonical_bytes(trace) + b"\n"
    )
    monkeypatch.setattr(runner, "_validate_initialized", lambda _e: None)
    monkeypatch.setattr(runner, "_load_episode", lambda *_a: None)
    monkeypatch.setattr(runner, "_load_runtime", lambda _e: SimpleNamespace())
    monkeypatch.setattr(runner, "_verify_runtime_boundary", lambda *_a: None)
    monkeypatch.setattr(runner, "_validate_episode_prefix", lambda *_a: 1)
    original_write = runner._write_or_match

    def rooted_write(path, value, **kwargs):
        kwargs["root"] = isolated.root
        return original_write(path, value, **kwargs)

    monkeypatch.setattr(runner, "_write_or_match", rooted_write)

    def reject(*_a, **_k):
        raise RuntimeError("semantic prewrite poison")

    monkeypatch.setattr(
        runner, "_verify_recovered_trace_before_episode_write", reject
    )
    assert source["sha256"]
    with pytest.raises(RuntimeError, match="semantic prewrite poison"):
        runner.run_episode("control", 0, SimpleNamespace())
    assert not (episode_root / "episode.json").exists()
    assert not (episode_root / "episode-completion.json").exists()


def test_completed_episode_recovery_forbids_exact_audit_reenactment(
    isolated, monkeypatch
) -> None:
    runner = isolated.runner
    source = _write_artifact(
        runner, runner.SOURCE_IDENTITY, "source-identity"
    )
    calls: list[bool] = []

    def verify_episode(_runtime, **kwargs):
        calls.append(kwargs["audit_reenact_exact"])
        return ({"schema": "unit-completed-episode"}, [])

    fake_verifier = SimpleNamespace(verify_episode=verify_episode)
    monkeypatch.setattr(
        runner,
        "_configured_bound_episode_verifier",
        lambda _runtime: fake_verifier,
    )
    runtime = SimpleNamespace(policy_static={})
    assert source["sha256"]
    assert runner._verify_completed_episode_with_bound_verifier(
        runtime, arm="control", index=0
    ) == {"schema": "unit-completed-episode"}
    assert calls == [False]


def test_exact_audit_cannot_begin_before_artifact_only_episode_pass(
    verifier, monkeypatch
) -> None:
    original = verifier.verify_episode
    calls: list[bool] = []
    exact_calls: list[str] = []

    class InvalidCompletion(RuntimeError):
        pass

    def guarded(runtime, **kwargs):
        audit = kwargs["audit_reenact_exact"]
        calls.append(audit)
        if audit is False:
            raise InvalidCompletion("shallow completion closure is invalid")
        return original(runtime, **kwargs)

    monkeypatch.setattr(verifier, "verify_episode", guarded)
    monkeypatch.setattr(
        verifier,
        "verify_exact_journal_rerun",
        lambda **_kwargs: exact_calls.append("exact"),
    )
    with pytest.raises(InvalidCompletion, match="completion closure"):
        guarded(
            SimpleNamespace(),
            arm="control",
            index=0,
            source_identity_sha256="a" * 64,
            audit_reenact_exact=True,
        )
    assert calls == [True, False]
    assert exact_calls == []


def test_runner_no_alternative_query_replays_end_to_end(
    isolated, verifier
) -> None:
    fixture = _publish_verifier_no_alternative_query(
        isolated.runner,
        verifier,
        isolated.root,
        isolated.root / "query-no-alternative",
    )
    replay = verifier.replay_no_alternative(
        fixture.g4,
        fixture.query_root,
        source_identity_sha256=fixture.source_sha,
        seed=fixture.seed,
        index=fixture.index,
        query_index=fixture.query_index,
        policy_static=fixture.policy_static,
    )
    assert replay["mode"] == "no-alternative"
    assert replay["exact_cost"] == 0
    assert replay["selected_ordinal"] == 0


@pytest.mark.parametrize(
    ("position", "schema"),
    [
        (4, "irisu-r3j-g4-formal-proposal-receipt-v1"),
        (5, "irisu-r3j-g4-proposed-exact-outcomes-v1"),
    ],
)
def test_verifier_rejects_normal_schema_in_no_alternative_chain(
    isolated, verifier, position: int, schema: str
) -> None:
    fixture = _publish_verifier_no_alternative_query(
        isolated.runner,
        verifier,
        isolated.root,
        isolated.root / f"query-cross-schema-{position}",
    )
    stages = list(verifier.verify_stage_chain(fixture.query_root))
    stages[position] = {**stages[position], "schema": schema}
    with pytest.raises(verifier.VerificationError, match="schema differs"):
        verifier.require_query_stage_schemas(stages, no_alternative=True)


def test_verifier_no_alternative_requires_absent_exact_journal(
    isolated, verifier
) -> None:
    fixture = _publish_verifier_no_alternative_query(
        isolated.runner,
        verifier,
        isolated.root,
        isolated.root / "query-empty-journal",
    )
    isolated.runner._mkdir_direct(
        fixture.query_root / verifier.EXACT_JOURNAL_DIRECTORY,
        root=isolated.root,
    )
    with pytest.raises(
        verifier.VerificationError, match="has an exact journal"
    ):
        verifier.replay_no_alternative(
            fixture.g4,
            fixture.query_root,
            source_identity_sha256=fixture.source_sha,
            seed=fixture.seed,
            index=fixture.index,
            query_index=fixture.query_index,
            policy_static=fixture.policy_static,
        )


@pytest.mark.parametrize("present", [False, True])
def test_verifier_normal_query_requires_nonempty_exact_journal(
    verifier, tmp_path: Path, present: bool
) -> None:
    query_root = tmp_path / f"normal-journal-{int(present)}"
    query_root.mkdir()
    if present:
        (query_root / verifier.EXACT_JOURNAL_DIRECTORY).mkdir()
    with pytest.raises(verifier.VerificationError, match="journal"):
        verifier.verify_required_exact_journal(query_root)


@pytest.mark.parametrize(
    "field",
    [
        "version",
        "barrier_config",
        "joint_config",
        "action_spec",
        "runner",
        "snapshot",
        "state_hash",
        "level",
    ],
)
def test_verifier_replayed_receipt_rejects_identity_tamper(
    verifier, field: str
) -> None:
    class Manifest:
        def __init__(self, value: dict[str, object]) -> None:
            self.value = value
            self.sha256 = verifier.canonical_sha256(value)

        def manifest(self) -> dict[str, object]:
            return self.value

    joint = Manifest({"schema": "unit-joint"})

    class Config(Manifest):
        def joint_config(self):
            return joint

    config = Config({"schema": "unit-barrier"})
    action = SimpleNamespace(sha256="a" * 64)
    evaluator = SimpleNamespace(config=config, action_spec=action)
    core = SimpleNamespace(
        JOINT=SimpleNamespace(JOINT_PLANNER_VERSION="unit-version")
    )
    environment = SimpleNamespace(
        runner_identity_manifest=lambda: {"schema": "unit-runner"},
        clone_state=lambda: b"unit-state",
        state_hash=lambda: 71,
    )
    observation = {"level": 3}
    receipt = {
        "generator": {
            "path": str(verifier.JOINT_SOURCE),
            "sha256": verifier.JOINT_SHA256,
            "version": "unit-version",
            "barrier_core_sha256": verifier.B_SHA256["barrier_core"],
            "barrier_config": config.manifest(),
            "barrier_config_sha256": config.sha256,
            "joint_config": joint.manifest(),
            "joint_config_sha256": joint.sha256,
            "action_spec_sha256": action.sha256,
        },
        "runtime": {
            "path": str(verifier.RUNTIME),
            "sha256": verifier.RUNTIME_SHA256,
            "runner": {"schema": "unit-runner"},
            "state_snapshot_sha256": hashlib.sha256(b"unit-state").hexdigest(),
            "state_hash": 71,
        },
        "level": 3,
    }
    if field == "version":
        receipt["generator"]["version"] = "foreign-version"
    elif field == "barrier_config":
        replacement = {"schema": "foreign-barrier"}
        receipt["generator"]["barrier_config"] = replacement
        receipt["generator"]["barrier_config_sha256"] = (
            verifier.canonical_sha256(replacement)
        )
    elif field == "joint_config":
        replacement = {"schema": "foreign-joint"}
        receipt["generator"]["joint_config"] = replacement
        receipt["generator"]["joint_config_sha256"] = (
            verifier.canonical_sha256(replacement)
        )
    elif field == "action_spec":
        receipt["generator"]["action_spec_sha256"] = "b" * 64
    elif field == "runner":
        receipt["runtime"]["runner"] = {"schema": "foreign-runner"}
    elif field == "snapshot":
        receipt["runtime"]["state_snapshot_sha256"] = "b" * 64
    elif field == "state_hash":
        receipt["runtime"]["state_hash"] = 72
    else:
        receipt["level"] = 4
    with pytest.raises(
        verifier.VerificationError,
        match="generator/runtime/level",
    ):
        verifier.verify_replayed_candidate_source_identity(
            receipt,
            evaluator=evaluator,
            environment=environment,
            observation=observation,
            core=core,
        )


def test_verifier_incumbent_intent_rejects_exact_certificate(verifier) -> None:
    with pytest.raises(
        verifier.VerificationError, match="incumbent execution intent"
    ):
        verifier.verify_selected_exact_certificate_binding(
            selected_ordinal=0,
            intent={"exact_tau2_certificate_sha256": "a" * 64},
            closure={},
            independently_rerun_certificate=None,
        )


@pytest.mark.parametrize("tamper", ["closure", "intent"])
def test_verifier_alternative_binds_selected_journal_certificate(
    verifier, tamper: str
) -> None:
    expected_body = {"schema": "unit-exact-certificate", "value": 1}
    expected = {
        **expected_body,
        "sha256": verifier.canonical_sha256(expected_body),
    }
    foreign_body = {"schema": "unit-exact-certificate", "value": 2}
    foreign = {
        **foreign_body,
        "sha256": verifier.canonical_sha256(foreign_body),
    }
    intent = {"exact_tau2_certificate_sha256": expected["sha256"]}
    closure = {
        "exact_certificate": expected,
        "exact_certificate_sha256": expected["sha256"],
    }
    if tamper == "closure":
        closure["exact_certificate"] = foreign
        closure["exact_certificate_sha256"] = foreign["sha256"]
    else:
        intent["exact_tau2_certificate_sha256"] = foreign["sha256"]
    with pytest.raises(
        verifier.VerificationError, match="selected journal certificate"
    ):
        verifier.verify_selected_exact_certificate_binding(
            selected_ordinal=1,
            intent=intent,
            closure=closure,
            independently_rerun_certificate=expected,
        )


def test_verifier_final_epoch_drift_suppresses_result(
    verifier, tmp_path: Path, monkeypatch
) -> None:
    root = (tmp_path / "final-epoch").resolve()
    root.mkdir()
    target = root / "observed.json"
    target.write_bytes(b"stable bytes\n")
    detached = tmp_path / "detached-final-evidence"
    produced: list[dict[str, bool]] = []

    def drifting_verdict(**_kwargs):
        assert verifier.file_bytes(target) == b"stable bytes\n"
        target.rename(detached)
        target.write_bytes(b"stable bytes\n")
        value = {"passed": True}
        produced.append(value)
        return value

    monkeypatch.setattr(verifier, "ROOT", root)
    monkeypatch.setattr(verifier, "_verify_locked", drifting_verdict)
    with pytest.raises(verifier.VerificationError):
        verifier.verify(
            runner_sha256="a" * 64,
            verifier_sha256="b" * 64,
            test_sha256="c" * 64,
            hardening_test_sha256="d" * 64,
            live_tau2_sha256="e" * 64,
            live_tau2_test_sha256="f" * 64,
        )
    assert produced == [{"passed": True}]
    assert verifier._ACTIVE_VERIFICATION_LEASE is None


def test_verifier_rejects_terminal_extra_partial_journal_intent(
    verifier, tmp_path: Path
) -> None:
    root = (tmp_path / "journal-verification").resolve()
    journal = root / "exact-journal"
    journal.mkdir(parents=True)
    (journal / "00-intent.json").write_bytes(b"{}\n")
    (journal / "00-result.json").write_bytes(b"{}\n")
    (journal / "01-intent.json").write_bytes(b"{")
    with verifier.VerificationLease(root):
        with pytest.raises(
            verifier.VerificationError, match="directory closure differs"
        ):
            verifier.require_entries(
                journal,
                {
                    "00-intent.json": "file",
                    "00-result.json": "file",
                },
            )


def test_verifier_rejects_old_worktree_irisu_env_preload(
    verifier, tmp_path: Path, monkeypatch
) -> None:
    foreign = tmp_path / "__init__.py"
    foreign.write_bytes(verifier.TRANSITIVE_MODULES["irisu_env"][0].read_bytes())
    monkeypatch.setitem(
        sys.modules,
        "irisu_env",
        SimpleNamespace(__file__=str(foreign)),
    )
    with pytest.raises(
        verifier.VerificationError, match="resolved elsewhere: irisu_env"
    ):
        verifier.preload_main_irisu_env()


def test_omitted_env_preload_reproduces_old_worktree_owner(
    runner,
) -> None:
    script = f"""
import importlib, importlib.util, sys
from pathlib import Path
p = Path({str(SOURCE)!r})
s = importlib.util.spec_from_file_location("_r3j_omitted_preload", p)
m = importlib.util.module_from_spec(s)
sys.modules[s.name] = m
s.loader.exec_module(m)
m.enforce_cpu0()
m.enforce_isolated_science()
m._preclosure_import_manifest()
importlib.import_module("irisu_pointer.resolution_proposal_g4")
m._load_module(
    "campaign_metrics",
    m.B_PATHS["campaign_metrics"],
    m.B_SHA256["campaign_metrics"],
)
m._load_module("barrier_core", m.B_PATHS["barrier_core"], m.B_SHA256["barrier_core"])
print(Path(sys.modules["irisu_env"].__file__).resolve())
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-X",
            f"pycache_prefix={runner.PYTHON_CACHE_PREFIX}",
            "-c",
            script,
        ],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
        },
    )
    assert completed.stdout.strip() == str(
        runner.JOINT_SOURCE.parents[1] / "irisu_env/__init__.py"
    )


@pytest.mark.parametrize("foreign_name", ["irisu_env", "irisu_pointer", "irisu_rl"])
def test_verifier_full_closure_rejects_foreign_package_owner(
    verifier, tmp_path: Path, monkeypatch, foreign_name: str
) -> None:
    live_sha = hashlib.sha256(verifier.LIVE_TAU2_SOURCE.read_bytes()).hexdigest()
    expected = {
        **verifier.TRANSITIVE_MODULES,
        "barrier_core": (
            verifier.B_PATHS["barrier_core"],
            verifier.B_SHA256["barrier_core"],
        ),
        "campaign_metrics": (
            verifier.B_PATHS["campaign_metrics"],
            verifier.B_SHA256["campaign_metrics"],
        ),
        "campaign": (
            verifier.B_PATHS["campaign"],
            verifier.B_SHA256["campaign"],
        ),
        "r3j_live_tau2_lease": (verifier.LIVE_TAU2_SOURCE, live_sha),
    }
    for name, (path, _digest) in expected.items():
        monkeypatch.setitem(
            sys.modules,
            name,
            SimpleNamespace(__file__=str(path)),
        )
    expected_path = expected[foreign_name][0]
    foreign = tmp_path / foreign_name / expected_path.name
    foreign.parent.mkdir()
    foreign.write_bytes(expected_path.read_bytes())
    monkeypatch.setitem(
        sys.modules,
        foreign_name,
        SimpleNamespace(__file__=str(foreign)),
    )
    with pytest.raises(
        verifier.VerificationError,
        match=f"module identity differs: {foreign_name}",
    ):
        verifier.verify_module_closure(live_sha)


def test_runner_rejects_foreign_irisu_env_preload(
    runner, tmp_path: Path, monkeypatch
) -> None:
    foreign = tmp_path / "__init__.py"
    foreign.write_bytes(runner.TRANSITIVE_MODULES["irisu_env"][0].read_bytes())
    monkeypatch.setitem(
        sys.modules,
        "irisu_env",
        SimpleNamespace(__file__=str(foreign)),
    )
    with pytest.raises(RuntimeError, match="simulator module is foreign"):
        runner._preload_main_irisu_env()
