from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from irisu_pointer.resolution_sustainable_r3k import (
    CHECKPOINT_HORIZONS,
    RunwayCheckpoint,
    RunwayOutcome,
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
            [
                self.identity.ordinal,
                self.absolute_safe,
                self.b2,
                self.exact_score_advantage,
            ]
        )


def _runway(
    identity: Identity,
    *,
    survival: int = 2_048,
    score: tuple[int, int, int] = (0, 10, 40),
    clears: tuple[int, int, int] = (0, 1, 4),
    gauge: int = 1_000,
    failed: bool = False,
) -> RunwayOutcome:
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
        )
        for index, horizon in enumerate(CHECKPOINT_HORIZONS)
    )
    return RunwayOutcome(
        identity,
        "0" * 64,
        checkpoints,
        survival,
        survival < 2_048,
        failed,
        0,
        False,
    )


def _proposal(mode: str = "growth") -> SimpleNamespace:
    return SimpleNamespace(mode=mode, reserve=100)


def test_checkpoint_rejects_censoring_marked_as_survival() -> None:
    with pytest.raises(ValueError, match="malformed"):
        RunwayCheckpoint(512, 100, True, 0, 0, 100, 100, 1)


def test_extension_union_keeps_rescue_and_growth_candidates() -> None:
    incumbent = ShortOutcome(Identity(0), True, 120.0, 0.0)
    rescue = ShortOutcome(Identity(1), True, 180.0, 0.0)
    growth = ShortOutcome(Identity(2), True, 130.0, 4.0)
    predictions = (
        SimpleNamespace(identity=rescue.identity, growth_mean=0.2, growth_std=0.1),
        SimpleNamespace(identity=growth.identity, growth_mean=0.9, growth_std=0.1),
    )
    assert extension_identities(
        _proposal(), (incumbent, rescue, growth), predictions
    ) == (incumbent.identity, rescue.identity, growth.identity)


def test_score_neutral_runway_improvement_is_admitted() -> None:
    incumbent = ShortOutcome(Identity(0), True, 120.0, 0.0)
    alternative = ShortOutcome(Identity(1), True, 160.0, 0.0)
    selected = select_sustainable(
        _proposal(),
        (incumbent, alternative),
        (_runway(incumbent.identity), _runway(alternative.identity)),
    )
    assert selected.status == "selected-runway"
    assert selected.identity == alternative.identity


def test_long_score_regression_is_rejected_even_with_better_b2() -> None:
    incumbent = ShortOutcome(Identity(0), True, 120.0, 0.0)
    alternative = ShortOutcome(Identity(1), True, 200.0, 5.0)
    selected = select_sustainable(
        _proposal(),
        (incumbent, alternative),
        (
            _runway(incumbent.identity, score=(0, 10, 50)),
            _runway(alternative.identity, score=(2, 12, 49)),
        ),
    )
    assert selected.status == "incumbent-retained"


def test_survival_regression_is_rejected_before_score() -> None:
    incumbent = ShortOutcome(Identity(0), True, 120.0, 0.0)
    alternative = ShortOutcome(Identity(1), True, 200.0, 10.0)
    selected = select_sustainable(
        _proposal(),
        (incumbent, alternative),
        (
            _runway(incumbent.identity),
            _runway(alternative.identity, survival=1_500, score=(10, 30, 100)),
        ),
    )
    assert selected.status == "incumbent-retained"


def test_unsafe_short_branch_is_rejected_before_long_score() -> None:
    incumbent = ShortOutcome(Identity(0), True, 120.0, 0.0)
    alternative = ShortOutcome(Identity(1), False, 500.0, 100.0)
    selected = select_sustainable(
        _proposal(),
        (incumbent, alternative),
        (
            _runway(incumbent.identity),
            _runway(alternative.identity, score=(30, 100, 300)),
        ),
    )
    assert selected.status == "incumbent-retained"


def test_complete_tie_retains_incumbent() -> None:
    incumbent = ShortOutcome(Identity(0), True, 120.0, 0.0)
    alternative = ShortOutcome(Identity(1), True, 120.0, 0.0)
    selected = select_sustainable(
        _proposal(),
        (incumbent, alternative),
        (_runway(incumbent.identity), _runway(alternative.identity)),
    )
    assert selected.status == "incumbent-retained"

