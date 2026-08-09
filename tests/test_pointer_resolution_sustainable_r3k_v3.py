from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from irisu_pointer.resolution_sustainable_r3k_v3 import (
    CHECKPOINT_HORIZONS,
    RunwayCheckpoint,
    RunwayOutcome,
    _public_count,
    extension_identities,
    select_sustainable,
)


def _sha(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class Identity:
    ordinal: int

    def manifest(self) -> dict[str, int]:
        return {"ordinal": self.ordinal}


@dataclass(frozen=True)
class ShortOutcome:
    identity: Identity
    absolute_safe: bool
    b2: float | None
    exact_score_advantage: float

    @property
    def sha256(self) -> str:
        return _sha(
            [self.identity.ordinal, self.absolute_safe, self.b2, self.exact_score_advantage]
        )


def _prediction(identity: Identity, growth: float, solvency: float) -> object:
    return SimpleNamespace(
        identity=identity,
        growth_mean=growth,
        growth_std=0.1,
        solvency_mean=solvency,
        solvency_std=0.1,
    )


def _runway(
    identity: Identity,
    *,
    survival: int = 2_048,
    score: tuple[int, int, int] = (0, 10, 40),
    clears: tuple[int, int, int] = (0, 1, 4),
    gauge: int = 1_000,
    terminated: bool = False,
    truncated: bool = False,
    game_over: bool = False,
    cashflow_lost: bool = False,
    invalid: int = 0,
    rebound: bool = False,
    unresolved: tuple[str, ...] = (),
) -> RunwayOutcome:
    terminal = terminated or truncated or game_over or cashflow_lost
    checkpoints = tuple(
        RunwayCheckpoint(
            horizon,
            min(horizon, survival),
            survival >= horizon,
            score[index],
            clears[index],
            gauge,
            gauge,
            1,
            (),
        )
        for index, horizon in enumerate(CHECKPOINT_HORIZONS)
    )
    return RunwayOutcome(
        identity,
        "0" * 64,
        checkpoints,
        survival,
        terminated,
        truncated,
        game_over,
        survival == 2_048,
        terminal,
        game_over or cashflow_lost,
        invalid,
        rebound,
        cashflow_lost,
        1 if cashflow_lost else None,
        "1" * 64,
        (10, 20),
        unresolved,
    )


def _proposal(mode: str = "growth") -> SimpleNamespace:
    return SimpleNamespace(mode=mode, reserve=100)


def test_public_count_accepts_numpy_integral_without_defaulting() -> None:
    assert _public_count({"gauge": np.uint64(777)}, "gauge") == 777
    with pytest.raises(KeyError, match="absent"):
        _public_count({}, "gauge")
    for value in (True, 1.0, np.float64(1.0)):
        with pytest.raises(TypeError, match="exact integer"):
            _public_count({"gauge": value}, "gauge")


def test_checkpoint_rejects_censoring_marked_as_survival() -> None:
    with pytest.raises(ValueError, match="malformed"):
        RunwayCheckpoint(512, 100, True, 0, 0, 100, 100, 1, ())


def test_rescue_shortlist_admits_low_reserve_negative_score_safe_branch() -> None:
    incumbent = ShortOutcome(Identity(0), True, 120.0, 0.0)
    rescue = ShortOutcome(Identity(1), True, 80.0, -3.0)
    assert extension_identities(
        _proposal("rescue"),
        (incumbent, rescue),
        (_prediction(rescue.identity, 0.1, 0.9),),
    ) == (incumbent.identity, rescue.identity)
    assert extension_identities(
        _proposal("growth"),
        (incumbent, rescue),
        (_prediction(rescue.identity, 0.1, 0.9),),
    ) == (incumbent.identity,)


def test_rescue_accepts_strict_survival_gain_but_not_equal_survival_lower_b2() -> None:
    incumbent = ShortOutcome(Identity(0), True, 120.0, 0.0)
    alternative = ShortOutcome(Identity(1), True, 80.0, -3.0)
    strict = select_sustainable(
        _proposal("rescue"),
        (incumbent, alternative),
        (
            _runway(incumbent.identity, survival=1_500),
            _runway(alternative.identity),
        ),
    )
    assert strict.status == "selected-runway"
    tied = select_sustainable(
        _proposal("rescue"),
        (incumbent, alternative),
        (_runway(incumbent.identity), _runway(alternative.identity)),
    )
    assert tied.status == "incumbent-retained"


def test_extension_union_is_identity_deduped_and_bounded() -> None:
    incumbent = ShortOutcome(Identity(0), True, 120.0, 0.0)
    alternatives = tuple(
        ShortOutcome(Identity(index), True, 120.0 + index, float(index))
        for index in range(1, 8)
    )
    predictions = tuple(
        _prediction(row.identity, growth=float(8 - row.identity.ordinal), solvency=float(row.identity.ordinal))
        for row in alternatives
    )
    selected = extension_identities(
        _proposal(), (incumbent, *alternatives), predictions
    )
    assert selected[0] == incumbent.identity
    assert len(selected) == 5
    assert len(set(selected)) == len(selected)


def test_score_neutral_long_runway_improvement_is_admitted() -> None:
    incumbent = ShortOutcome(Identity(0), True, 120.0, 0.0)
    alternative = ShortOutcome(Identity(1), True, 160.0, 0.0)
    selected = select_sustainable(
        _proposal(),
        (incumbent, alternative),
        (_runway(incumbent.identity), _runway(alternative.identity)),
    )
    assert selected.status == "selected-runway"
    assert selected.identity == alternative.identity


@pytest.mark.parametrize(
    "kwargs",
    (
        {"cashflow_lost": True},
        {"game_over": True},
        {"terminated": True},
        {"truncated": True},
        {"invalid": 1},
        {"rebound": True},
        {"unresolved": ("foreign",)},
        {"survival": 2_047},
    ),
)
def test_long_safety_regressions_retain_incumbent(kwargs: dict[str, object]) -> None:
    incumbent = ShortOutcome(Identity(0), True, 120.0, 0.0)
    alternative = ShortOutcome(Identity(1), True, 200.0, 20.0)
    selected = select_sustainable(
        _proposal(),
        (incumbent, alternative),
        (
            _runway(incumbent.identity),
            _runway(alternative.identity, score=(20, 100, 500), **kwargs),
        ),
    )
    assert selected.status == "incumbent-retained"


def test_long_score_regression_and_complete_tie_retain_incumbent() -> None:
    incumbent = ShortOutcome(Identity(0), True, 120.0, 0.0)
    worse = ShortOutcome(Identity(1), True, 200.0, 5.0)
    assert select_sustainable(
        _proposal(),
        (incumbent, worse),
        (
            _runway(incumbent.identity, score=(0, 10, 50)),
            _runway(worse.identity, score=(2, 12, 49)),
        ),
    ).status == "incumbent-retained"
    tied = ShortOutcome(Identity(2), True, 120.0, 0.0)
    assert select_sustainable(
        _proposal(),
        (incumbent, tied),
        (_runway(incumbent.identity), _runway(tied.identity)),
    ).status == "incumbent-retained"
