#!/usr/bin/env python3
"""Development-only synthetic correctness checks for R3G Strategy C."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _event(kind: Any, tick: int, value: int = 0, **extra: object) -> dict[str, object]:
    return {"kind": int(kind), "tick": tick, "value": value, **extra}


def _shift(events: Sequence[Mapping[str, object]], ticks: int) -> list[dict[str, object]]:
    return [{**value, "tick": int(value["tick"]) + ticks} for value in events]


def _cashflow_signature(value: Any) -> tuple[object, ...]:
    return (
        value.net_renewal_gains,
        value.b2,
        value.final_replayed_gauge,
        value.rot_events,
        value.rot_payment,
        value.passive_drain,
        value.terminal_before_second,
        tuple(
            (
                cycle.completed,
                cycle.renewal_gain,
                cycle.passive_drain,
                cycle.rot_payment,
                cycle.rot_events,
                cycle.minimum_gauge_before_renewal,
                cycle.solvency_surplus,
            )
            for cycle in value.cycles
        ),
    )


def _check_cashflow(api: Any) -> None:
    changed = api.EventKind.GAUGE_CHANGED
    standard = [
        _event(changed, 11, -1, sequence=0, detail="passive drain"),
        _event(changed, 12, 1_000, sequence=0, detail="normal burst landing"),
        _event(changed, 12, -1, sequence=1, detail="passive drain"),
        _event(changed, 13, -1, sequence=0, detail="passive drain"),
        _event(changed, 14, 700, sequence=0, detail="special color clear"),
        _event(changed, 14, -1, sequence=1, detail="passive drain"),
    ]
    split = [
        standard[0],
        _event(changed, 12, 400, sequence=0, detail="normal burst landing"),
        _event(changed, 12, 600, sequence=1, detail="normal burst landing"),
        _event(changed, 12, -1, sequence=2, detail="passive drain"),
        *standard[3:],
    ]
    replay = api.m.replay_two_renewal_cashflow(
        initial_gauge=100,
        gauge_max=40_000,
        initial_level=1,
        start_tick=10,
        final_tick=14,
        events=standard,
    )
    split_replay = api.m.replay_two_renewal_cashflow(
        initial_gauge=100,
        gauge_max=40_000,
        initial_level=1,
        start_tick=10,
        final_tick=14,
        events=split,
    )
    shifted = api.m.replay_two_renewal_cashflow(
        initial_gauge=100,
        gauge_max=40_000,
        initial_level=1,
        start_tick=110,
        final_tick=114,
        events=_shift(standard, 100),
    )
    _expect(replay.renewal_ticks == (12, 14), "renewals must use distinct ticks")
    _expect(
        split_replay.renewal_ticks == (12, 14),
        "same-tick split gains must coalesce",
    )
    _expect(
        _cashflow_signature(replay) == _cashflow_signature(split_replay),
        "splitting a same-tick gain changed cash flow",
    )
    _expect(
        _cashflow_signature(replay) == _cashflow_signature(shifted),
        "absolute tick offset changed cash flow",
    )
    _expect(
        shifted.renewal_ticks == (112, 114),
        "tick-offset replay lost renewal epochs",
    )

    rot_events = [
        _event(changed, 1, 2_000, sequence=0, detail="normal burst landing"),
        _event(changed, 1, -1, sequence=1, detail="passive drain"),
        _event(api.EventKind.ROTTEN, 1, sequence=2, a=7),
        _event(
            changed,
            1,
            -1_820,
            sequence=3,
            detail="normal rot penalty",
            a=7,
        ),
    ]
    ordered = api.m.replay_two_renewal_cashflow(
        initial_gauge=100,
        gauge_max=40_000,
        initial_level=1,
        start_tick=0,
        final_tick=1,
        events=rot_events,
    )
    _expect(ordered.final_replayed_gauge == 279, "same-tick event order is wrong")
    _expect(ordered.net_renewal_gains == (2_000,), "renewal must be net post-clamp")
    _expect(ordered.rot_payment == 1_820, "rot payment used the wrong level")

    no_rescue = api.m.replay_two_renewal_cashflow(
        initial_gauge=0,
        gauge_max=40_000,
        initial_level=1,
        start_tick=0,
        final_tick=1,
        events=[_event(changed, 1, 2_000, detail="normal burst landing")],
    )
    _expect(not no_rescue.renewal_ticks, "same-tick clear rescued an insolvent entry")
    _expect(no_rescue.terminal_before_second, "terminal entry was not recorded")


def _observation(tick: int, level: int, bodies: Sequence[Mapping[str, object]]) -> dict[str, object]:
    return {"tick": tick, "level": level, "bodies": list(bodies)}


def _check_debt_ledger(api: Any) -> None:
    body = {
        "id": 7,
        "kind": "piece",
        "lifecycle": "fresh",
        "rot_timer": 40,
    }
    ledger = api.m.CandidateDebtLedger(40)
    ledger.observe(_observation(100, 2, [body]))
    _expect(set(ledger.active) == {7}, "visible liability was not created")
    _expect(set(ledger.created) == {7}, "created liability was not retained")

    same_tick = [
        _event(api.EventKind.ROTTEN, 101, sequence=0, a=7),
        _event(
            api.EventKind.GAUGE_CHANGED,
            101,
            -1_840,
            sequence=1,
            detail="normal rot penalty",
            a=7,
        ),
        _event(api.EventKind.CLEARED, 101, sequence=2, a=7),
    ]
    ledger.apply(same_tick)
    _expect(ledger.paid == {7: 1_840}, "rot liability was not paid exactly once")
    _expect(7 not in ledger.active, "paid liability remained active")
    _expect(7 not in ledger.retired, "same-tick clear overrode a rot payment")
    _expect(not ledger.duplicate_payments, "first payment was marked duplicate")
    _expect(not ledger.unmatched_payments, "paired rot payment was unmatched")
    ledger.observe(_observation(101, 2, []))
    ledger.observe(_observation(102, 2, [body]))
    _expect(7 not in ledger.active, "paid liability was re-added")

    ledger.apply(same_tick)
    _expect(ledger.duplicate_payments > 0, "duplicate payment was not detected")

    clear_only = api.m.CandidateDebtLedger(40)
    other = {**body, "id": 8, "rot_timer": 10}
    clear_only.observe(_observation(200, 3, [other]))
    clear_only.apply([_event(api.EventKind.CLEARED, 201, sequence=0, a=8)])
    _expect(8 in clear_only.retired, "cleared unpaid liability was not retired")
    _expect(8 not in clear_only.paid, "clear-only retirement became a payment")


def _cycle(api: Any, ordinal: int, surplus: int, *, complete: bool = True) -> Any:
    return api.m.RenewableCycle(
        ordinal=ordinal,
        start_tick=(ordinal - 1) * 10,
        end_tick=ordinal * 10,
        completed=complete,
        start_gauge=1_000,
        renewal_gain=100 if complete else 0,
        passive_drain=10,
        rot_payment=0,
        rot_events=0,
        realized_rot_liability=0,
        minimum_gauge_before_renewal=900,
        solvency_surplus=surplus,
    )


def _construct(cls: type[Any], values: Mapping[str, object]) -> Any:
    parameters = inspect.signature(cls).parameters
    supplied: dict[str, object] = {}
    fallbacks: dict[str, object] = {
        "action_equivalent_to_incumbent": False,
        "issued_action_equivalent_to_incumbent": False,
        "full_issued_action_equivalent": False,
        "terminal_event": False,
        "gauge_failure": False,
    }
    for name, parameter in parameters.items():
        if name in values:
            supplied[name] = values[name]
        elif name in fallbacks:
            supplied[name] = fallbacks[name]
        elif parameter.default is inspect.Parameter.empty:
            raise AssertionError(f"synthetic fixture needs new required field {name!r}")
    return cls(**supplied)


def _outcome(
    api: Any,
    ordinal: int,
    *,
    surplus: int,
    score: int,
    complete: bool = True,
    alive: bool = True,
    action_equivalent: bool = False,
) -> Any:
    renewals = (10, 20) if complete else (10,)
    cycles = (
        _cycle(api, 1, surplus),
        _cycle(api, 2, surplus, complete=complete),
    )
    values = {
        "candidate_ordinal": ordinal,
        "candidate_name": f"candidate-{ordinal}",
        "pair_ordinal": ordinal,
        "geometry_ordinal": 0,
        "pair_category": "fresh-match",
        "start_tick": 0,
        "survival_ticks": 20,
        "alive": alive,
        "invalid_actions": 0,
        "score_gain": score,
        "final_gauge": 2_000,
        "level_delta": 0,
        "qualifying_clear_gain": 2,
        "shot_hit": True,
        "pair_joined": True,
        "renewable_ticks": renewals,
        "renewable_gains": tuple(100 for _ in renewals),
        "cycles": cycles,
        "rot_events": 0,
        "rot_payment": 0,
        "passive_drain": 20,
        "liabilities_created": 0,
        "liabilities_paid": 0,
        "liabilities_retired": 0,
        "duplicate_payments": 0,
        "unmatched_payments": 0,
        "snapshot_sha256": "0" * 64,
        "exact_state_hash": "1" * 64,
        "action_equivalent_to_incumbent": action_equivalent,
        "issued_action_equivalent_to_incumbent": action_equivalent,
        "full_issued_action_equivalent": action_equivalent,
    }
    return _construct(api.m.ExactEventOutcome, values)


def _candidate(ordinal: int) -> Any:
    return SimpleNamespace(ordinal=ordinal)


def _result(api: Any, outcomes: Sequence[Any]) -> Any:
    values = {
        "query_id": "synthetic",
        "snapshot_sha256": "0" * 64,
        "candidates": tuple(_candidate(value.candidate_ordinal) for value in outcomes),
        "outcomes": tuple(outcomes),
        "restore_checks": len(outcomes) + 1,
        "wall_seconds": 0.0,
        "cpu_seconds": 0.0,
    }
    return _construct(api.m.ExactSearchResult, values)


def _check_exact_classification(api: Any) -> None:
    base = _outcome(api, 0, surplus=20, score=0)
    safe = _outcome(api, 1, surplus=20, score=5)
    unrelated = _outcome(api, 2, surplus=-1, score=1_000)
    unresolved = _outcome(api, 3, surplus=30, score=1_000, complete=False)
    full = _result(api, (base, safe, unrelated, unresolved))
    local = _result(api, (base, safe))
    _expect(full.safe(safe) and local.safe(safe), "safe classification was nonlocal")
    _expect(not full.safe(unrelated), "insolvent candidate was classified safe")
    _expect(not full.safe(unresolved), "unresolved second renewal was classified safe")
    _expect(full.selected_ordinal == 1, "unsafe alternative changed winner")
    _expect(local.selected_ordinal == 1, "safe score winner was not selected")

    score_tie = _outcome(api, 1, surplus=20, score=0)
    _expect(
        _result(api, (base, score_tie)).selected_ordinal == 0,
        "score tie did not select frozen-v5",
    )
    outcome_fields = inspect.signature(api.m.ExactEventOutcome).parameters
    equivalence_fields = {
        "action_equivalent_to_incumbent",
        "issued_action_equivalent_to_incumbent",
        "full_issued_action_equivalent",
    } & set(outcome_fields)
    if equivalence_fields:
        equivalent = _outcome(
            api,
            1,
            surplus=30,
            score=100,
            action_equivalent=True,
        )
        _expect(
            _result(api, (base, equivalent)).selected_ordinal == 0,
            "action-equivalent candidate did not select frozen-v5",
        )


def _targets(api: Any, **updates: float) -> tuple[float, ...]:
    values = {name: 0.0 for name in api.m.TARGET_NAMES}
    values.update(
        {
            "first_renewal_reached": 1.0,
            "second_renewal_reached": 1.0,
            "b2": 100.0,
            "second_cycle_margin": 100.0,
            "final_gauge": 2_000.0,
            "alive": 1.0,
            "survival_ticks": 100.0,
            "delta_b2": 0.0,
            "delta_final_gauge": 10.0,
            "delta_score_gain": 1.0,
            "catastrophic": 0.0,
        }
    )
    values.update(updates)
    return tuple(values[name] for name in api.m.TARGET_NAMES)


def _examples(
    api: Any,
    seeds: Sequence[int],
    targets: Callable[[int, int], tuple[float, ...]],
) -> list[Any]:
    features = tuple(0.0 for _ in api.m.FEATURE_NAMES)
    return [
        api.m.ModelExample(
            "barrier-calibration",
            seed,
            f"synthetic:{seed}:{candidate}",
            0,
            candidate,
            features,
            targets(seed, candidate),
            {},
        )
        for seed in seeds
        for candidate in range(2)
    ]


def _check_conformal(api: Any) -> None:
    seeds = tuple(range(40))
    blank = _examples(api, seeds, lambda _seed, _candidate: _targets(api))
    fit_seeds, calibration_seeds = api.m.KNNEventWorldModel._seed_partition(
        blank, 0.5
    )
    ranked_calibration = sorted(calibration_seeds)
    residual = {
        seed: float(index + 1)
        for index, seed in enumerate(ranked_calibration)
    }

    def target(seed: int, candidate: int) -> tuple[float, ...]:
        if seed in fit_seeds:
            return _targets(api, delta_b2=0.0)
        value = residual[seed] - 0.25 * candidate
        return _targets(api, delta_b2=-value)

    examples = _examples(api, seeds, target)
    config = api.m.EventMPCConfig(
        neighbor_count=24,
        calibration_fraction=0.5,
        conformal_alpha=0.05,
    )
    provisional = api.m.KNNEventWorldModel(config)
    if hasattr(provisional, "fit_provisional"):
        provisional.fit_provisional(examples)
        _expect(
            not len(provisional.calibration_residuals),
            "provisional fit consumed reserved calibration targets",
        )
        _expect(
            provisional.features.shape[0] == 2 * len(fit_seeds),
            "provisional fit used the wrong whole-seed partition",
        )

    model = api.m.KNNEventWorldModel(config)
    model.fit(examples)
    expected_rank = math.ceil((len(calibration_seeds) + 1) * 0.95)
    expected_q = sorted(residual.values())[expected_rank - 1]
    _expect(set(model.fit_seeds) == fit_seeds, "fit seed partition changed")
    _expect(
        set(model.calibration_seeds) == calibration_seeds,
        "calibration seed partition changed",
    )
    _expect(math.isclose(model.conformal_q, expected_q), "wrong conformal q")

    manifest = model.manifest()
    restored = api.m.KNNEventWorldModel.from_manifest(
        json.loads(json.dumps(manifest))
    )
    _expect(
        math.isclose(restored.conformal_q, model.conformal_q),
        "serialized conformal threshold changed",
    )
    original_prediction = model.predict((0.0,) * len(api.m.FEATURE_NAMES))
    restored_prediction = restored.predict((0.0,) * len(api.m.FEATURE_NAMES))
    _expect(
        original_prediction.mean == restored_prediction.mean
        and original_prediction.lower == restored_prediction.lower,
        "serialized model prediction changed",
    )

    auditable = (
        "conformal_episode_residuals",
        "conformal_records",
        "episode_residuals",
    )
    record_key = next((key for key in auditable if key in manifest), None)
    if record_key is not None:
        records = manifest[record_key]
        _expect(
            len(records) == len(calibration_seeds),
            "manifest lost whole-seed conformal records",
        )
    for key, expected in (
        ("conformal_n", len(calibration_seeds)),
        ("conformal_rank", expected_rank),
    ):
        if key in manifest:
            _expect(int(manifest[key]) == expected, f"manifest has wrong {key}")

    tie_examples = _examples(
        api,
        seeds,
        lambda _seed, _candidate: _targets(
            api,
            b2=100.0,
            delta_b2=10.0,
            delta_final_gauge=10.0,
            delta_score_gain=0.0,
        ),
    )
    tie_model = api.m.KNNEventWorldModel(config)
    tie_model.fit(tie_examples)
    _prediction, certificate = tie_model.certify(
        (0.0,) * len(api.m.FEATURE_NAMES)
    )
    _expect(not certificate.certified, "zero score advantage was certified")
    _expect(
        "positive_score_advantage" in certificate.reasons,
        "score-tie rejection reason was lost",
    )


def _source_sha256() -> str:
    path = ROOT / "python/irisu_pointer/event_mpc.py"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_api() -> Any:
    module = importlib.import_module("irisu_pointer.event_mpc")
    environment = importlib.import_module("irisu_env")
    return SimpleNamespace(m=module, EventKind=environment.EventKind)


def main() -> int:
    checks: tuple[tuple[str, Callable[[Any], None]], ...] = (
        ("cashflow", _check_cashflow),
        ("debt_ledger", _check_debt_ledger),
        ("exact_classification", _check_exact_classification),
        ("whole_seed_conformal", _check_conformal),
    )
    rows: list[dict[str, object]] = []
    api: Any = None
    load_error: BaseException | None = None
    try:
        api = _load_api()
    except BaseException as exc:  # one machine-readable result even on import failure
        load_error = exc
    if load_error is not None:
        rows.append(
            {
                "name": "import",
                "passed": False,
                "error_type": type(load_error).__name__,
                "error": str(load_error),
            }
        )
    else:
        for name, check in checks:
            try:
                check(api)
                rows.append({"name": name, "passed": True})
            except BaseException as exc:
                rows.append(
                    {
                        "name": name,
                        "passed": False,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    passed = bool(rows) and all(bool(value["passed"]) for value in rows)
    result = {
        "schema": "irisu-r3g-event-mpc-development-check-v1",
        "development_only": True,
        "external_artifacts_used": False,
        "event_mpc_version": (
            getattr(api.m, "EVENT_MPC_VERSION", None) if api is not None else None
        ),
        "event_mpc_source_sha256": _source_sha256(),
        "logical_cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "checks": rows,
        "passed": passed,
    }
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
