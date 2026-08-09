from __future__ import annotations

import copy
import hashlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest


REPOSITORY = Path("/home/gabe/Documents/irisu")
SOURCE = (
    REPOSITORY
    / "artifacts/r3/development/r3j-g4-onpolicy-screen-20260731-003/"
    "run_g4_onpolicy_screen.py"
)


@pytest.fixture(scope="module")
def runner():
    name = "_r3j_g4_onpolicy_test_runner"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(name, None)


def test_fresh_seed_schedule_is_exact_and_collision_free(runner) -> None:
    manifest = runner.preregistered_seed_manifest()
    proof = runner.seed_collision_proof()
    seeds = [runner.derive_seed(index) for index in range(runner.SEED_COUNT)]
    assert manifest["rows"] == [
        {"index": index, "seed": seed}
        for index, seed in enumerate(seeds)
    ]
    assert len(seeds) == len(set(seeds)) == 8
    assert proof["collision_count"] == 0
    assert proof["sha256"] == runner._canonical_sha256(
        {key: value for key, value in proof.items() if key != "sha256"}
    )


def test_seed_derivation_rejects_nonexact_or_out_of_range_indices(runner) -> None:
    for value in (-1, 8, True, 1.0):
        with pytest.raises(ValueError):
            runner.derive_seed(value)


@dataclass(frozen=True)
class _Identity:
    ordinal: int

    def manifest(self) -> dict[str, object]:
        return {"ordinal": self.ordinal, "identity": f"candidate-{self.ordinal}"}


@dataclass(frozen=True)
class _Prediction:
    identity: _Identity
    solvency_mean: float
    growth_mean: float


def _prediction_batch(count: int = 10):
    identities = tuple(_Identity(index + 1) for index in range(count))
    predictions = tuple(
        _Prediction(
            identity,
            solvency_mean=float(count - index),
            growth_mean=float(index),
        )
        for index, identity in enumerate(identities)
    )
    return SimpleNamespace(
        predictions=predictions,
        sha256="1" * 64,
        model_sha256="2" * 64,
        inventory=SimpleNamespace(sha256="3" * 64),
    )


def test_dual_plan_is_frozen_deduplicated_and_exact_k(runner) -> None:
    batch = _prediction_batch()
    first = runner.dual_plan(batch)
    second = runner.dual_plan(batch)
    assert first == second
    assert first["sha256"] == runner._canonical_sha256(
        {key: value for key, value in first.items() if key != "sha256"}
    )
    for mode in ("rescue", "growth"):
        branch = first["branches"][mode]
        ordinals = [row["ordinal"] for row in branch["identities"]]
        assert len(ordinals) == len(set(ordinals)) == runner.BUDGET
        assert len(branch["admission_sources"]) == runner.BUDGET
    assert first["branches"]["rescue"] != first["branches"]["growth"]


def test_write_once_publisher_recovers_only_an_exact_prefix(
    runner, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    target = root / "nested/evidence.json"
    anchor = root / "anchor.json"
    value = runner._with_sha({"schema": "unit-test", "complete": True})
    payload = runner._canonical_bytes(value) + b"\n"
    with runner._command_lease(root, enforce_root_closure=False):
        runner._write_or_match(
            anchor, runner._with_sha({"schema": "anchor"}), root=root
        )
        runner._mkdir_direct(target.parent, root=root)
        parent_descriptor = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            descriptor = os.open(
                target.name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o644,
                dir_fd=parent_descriptor,
            )
            try:
                os.write(descriptor, payload[:17])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            os.close(parent_descriptor)
        digest = runner._write_or_match(
            target, value, root=root, recovery_anchor=anchor
        )
        assert target.read_bytes() == payload
        assert digest == hashlib.sha256(payload).hexdigest()
        assert runner._write_or_match(target, value, root=root) == digest
        with pytest.raises(FileExistsError):
            runner._write_new(target, value, root=root)
        with pytest.raises(RuntimeError, match="prefix|oversized"):
            runner._write_or_match(
                target,
                runner._with_sha({"schema": "different"}),
                root=root,
            )


def test_output_mutation_requires_a_command_lease(
    runner, tmp_path: Path
) -> None:
    with pytest.raises(RuntimeError, match="lacks its command lease"):
        runner._write_or_match(
            tmp_path.resolve() / "artifact.json",
            runner._with_sha({"schema": "unit-test"}),
            root=tmp_path.resolve(),
        )


def test_anchored_publisher_rejects_indirect_or_linked_outputs(
    runner, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (root / "indirect").symlink_to(outside, target_is_directory=True)
    with runner._command_lease(root, enforce_root_closure=False):
        with pytest.raises(OSError):
            runner._write_or_match(
                root / "indirect/evidence.json",
                {"schema": "unit-test"},
                root=root,
            )
        direct = root / "direct.json"
        runner._write_or_match(direct, {"schema": "unit-test"}, root=root)
        os.link(direct, root / "second-link.json")
        with pytest.raises(RuntimeError, match="single-link"):
            runner._write_or_match(
                direct,
                {"schema": "unit-test"},
                root=root,
            )


def test_primitive_loop_stops_on_the_first_terminal_press(runner) -> None:
    class Kind:
        WAIT = object()

        @staticmethod
        def parse(value):
            return value

    class Action:
        def __init__(self, kind=None, wait_ticks=4):
            self.kind = Kind.WAIT if kind is None else kind
            self.wait_ticks = wait_ticks

        @staticmethod
        def wait(_ticks):
            return Action(Kind.WAIT, 1)

    campaign = SimpleNamespace(ActionKind=Kind, Action=Action)
    calls = []

    def step(action):
        calls.append(action)
        return {"tick": 1}, 0.0, True, False, {}

    final, executed = runner._execute_primitive_units(
        [Action()],
        campaign=campaign,
        step_unit=step,
        remaining_ticks=lambda: 10,
        stopped=lambda: bool(calls),
    )
    assert final == {"tick": 1}
    assert executed == len(calls) == 1


def test_exact_certificate_only_converts_explicit_negative_evidence(
    runner,
) -> None:
    class Tracer:
        def __init__(self, message):
            self.message = message

        def certificate(self):
            raise RuntimeError(self.message)

        def terminal_witness(self, message):
            return {"schema": "unit-negative-witness", "reason": message}

    allowed = next(iter(runner.EXACT_CERTIFICATE_NEGATIVE_REASONS))
    witness, reason = runner._read_exact_certificate(Tracer(allowed))
    assert witness == {"schema": "unit-negative-witness", "reason": allowed}
    assert reason == allowed
    with pytest.raises(RuntimeError, match="unexpected implementation failure"):
        runner._read_exact_certificate(
            Tracer("unexpected implementation failure")
        )


def test_query_stage_chain_binds_artifact_and_file_hashes(
    runner, tmp_path: Path
) -> None:
    root = tmp_path.resolve()
    query = root / "query-00"
    with runner._command_lease(root, enforce_root_closure=False):
        runner._mkdir_direct(query, root=root)
        records: dict[str, dict[str, str]] = {}
        first = runner._publish_query_stage(
            query,
            records,
            1,
            "candidate-receipt",
            {"schema": "receipt"},
            root=root,
        )
        second = runner._publish_query_stage(
            query,
            records,
            2,
            "prediction",
            {"schema": "prediction"},
            root=root,
        )
        assert second["previous_stage"] == {
            "name": "candidate-receipt",
            "artifact_sha256": first["sha256"],
            "file_sha256": hashlib.sha256(
                runner._canonical_bytes(first) + b"\n"
            ).hexdigest(),
        }
        with pytest.raises(RuntimeError, match="reserved"):
            runner._publish_query_stage(
                query,
                records,
                3,
                "preexact-plan",
                {"schema": "plan", "sha256": "0" * 64},
                root=root,
            )


def _synthetic_trace(runner, *, terminal: bool = False) -> tuple[dict, str, str]:
    source_sha = "a" * 64
    intent_sha = "b" * 64
    initial = {
        "tick": 10,
        "score": 100,
        "gauge": 80,
        "level": 1,
        "qualifying_clear_count": 2,
    }
    final = {
        "tick": 11,
        "score": 125,
        "gauge": 75,
        "level": 1,
        "qualifying_clear_count": 3,
    }
    genesis = runner._trace_genesis(
        source_identity_sha256=source_sha,
        episode_intent_sha256=intent_sha,
        arm="control",
        index=0,
        seed=runner.derive_seed(0),
        runner={"config_hash": 7, "runtime": "portable"},
        config_hash=7,
        reset_info={"seed": runner.derive_seed(0), "config_hash": 7},
        initial_observation=initial,
        initial_state_sha256="c" * 64,
    )
    genesis_sha = runner._canonical_sha256(genesis)
    info = {"config_hash": 7, "events": []}
    row = runner._with_sha(
        {
            "schema": "irisu-r3j-g4-unit-trace-row-v1",
            "ordinal": 0,
            "previous_sha256": genesis_sha,
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
            "pre_observation_sha256": runner._canonical_sha256(initial),
            "post_observation_sha256": runner._canonical_sha256(final),
            "reward": 25.0,
            "info": info,
            "info_sha256": runner._canonical_sha256(info),
            "events": [],
            "terminated": terminal,
            "truncated": False,
            "config_hash": 7,
            "state_sha256": "e" * 64,
        }
    )
    trace = runner._with_sha(
        {
            "schema": "irisu-r3j-g4-unit-trace-v1",
            "complete": True,
            "development_only": True,
            "sealed_test_allowed": False,
            "episode_intent_sha256": intent_sha,
            "arm": "control",
            "index": 0,
            "seed": runner.derive_seed(0),
            "genesis": genesis,
            "genesis_sha256": genesis_sha,
            "rows": [row],
            "row_count": 1,
            "final_row_sha256": row["sha256"],
            "final_public_observation": final,
            "final_public_observation_sha256": runner._canonical_sha256(final),
            "final_state_sha256": row["state_sha256"],
            "episode_metadata": {
                "decisions": 1,
                "seen_shots": 0,
                "queries": [],
                "skipped_query_slots": [],
            },
        }
    )
    return trace, source_sha, intent_sha


def test_trace_schema_is_closed_and_hash_chained(runner, monkeypatch) -> None:
    trace, source_sha, intent_sha = _synthetic_trace(runner)
    monkeypatch.setattr(
        runner,
        "_read_canonical_json",
        lambda path, expected_sha256=None: {"sha256": source_sha},
    )
    assert (
        runner._validate_episode_trace(
            trace,
            arm="control",
            index=0,
            intent_sha256=intent_sha,
        )
        == trace["sha256"]
    )
    foreign = copy.deepcopy(trace)
    foreign["rows"][0]["unreviewed"] = True
    foreign["rows"][0] = runner._with_sha(
        {
            key: value
            for key, value in foreign["rows"][0].items()
            if key != "sha256"
        }
    )
    foreign["final_row_sha256"] = foreign["rows"][0]["sha256"]
    foreign = runner._with_sha(
        {key: value for key, value in foreign.items() if key != "sha256"}
    )
    with pytest.raises(RuntimeError, match="row 0 differs"):
        runner._validate_episode_trace(
            foreign,
            arm="control",
            index=0,
            intent_sha256=intent_sha,
        )


def test_trace_ledger_binding_rehashes_the_exact_query_range(runner) -> None:
    trace, _source_sha, _intent_sha = _synthetic_trace(runner)
    rows = trace["rows"]
    rows[0]["query_context"] = {
        "source": "g4-query-selection",
        "query_ledger_sha256": None,
    }
    rows[0] = runner._with_sha(
        {key: value for key, value in rows[0].items() if key != "sha256"}
    )
    bound = runner._bind_trace_query_ledger(rows, 0, "f" * 64)
    assert bound["row_count"] == 1
    assert rows[0]["query_context"]["query_ledger_sha256"] == "f" * 64
    assert bound["final_row_sha256"] == rows[0]["sha256"]


def test_scientific_import_boundary_rejects_unisolated_pytest(runner) -> None:
    with pytest.raises(RuntimeError, match="require -I -S -B"):
        runner.enforce_isolated_science()


def test_irisu_env_module_closure_is_bound_to_current_main(runner) -> None:
    expected_root = REPOSITORY / "python/irisu_env"
    for name, (path, digest) in runner.TRANSITIVE_MODULES.items():
        if not name.startswith("irisu_env"):
            continue
        assert path == expected_root / (
            "__init__.py" if name == "irisu_env" else name.rsplit(".", 1)[1] + ".py"
        )
        assert runner._sha256_file(path) == digest


def test_model_inference_occurs_only_after_durable_g4_reconstruction() -> None:
    source = SOURCE.read_text()
    query_body = source.split("def _run_g4_query(", 1)[1].split(
        "\ndef run_episode(", 1
    )[0]
    assert query_body.index("durable_receipt = _read_canonical_json") < (
        query_body.index("runtime.model.predict_batches")
    )
    assert '"inference_envelope_sha256"' in query_body
    assert '"g4_board_sha256"' in query_body
    assert '"g4_feature_inventory_sha256"' in query_body


def test_runner_never_self_authorizes_structural_or_promising_gate() -> None:
    source = SOURCE.read_text()
    assert '"structural_go": True' not in source
    assert '"promising": True' not in source
    assert '"survival_at_least_98pct"' not in source
    assert '"score_at_least_98pct"' not in source
    assert '"score_at_least_110pct"' not in source
    assert '"material_survival_improvement"' not in source
    assert "prepare-test" not in source
