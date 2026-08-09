from __future__ import annotations

from types import SimpleNamespace

import pytest

from benchmarks.rl_r3k_sustainable_v2 import _enumerate_candidates


class _Evaluator:
    def __init__(self, error: str | None = None) -> None:
        self.error = error

    def candidates(self, observation: object, incumbent: object) -> tuple[object, ...]:
        if self.error is not None:
            raise ValueError(self.error)
        return (incumbent,)


def test_unrepresentable_incumbent_is_a_fail_closed_abstention() -> None:
    evaluator = _Evaluator("incoming incumbent is absent from the legal joint shortlist")
    assert _enumerate_candidates(evaluator, {}, SimpleNamespace()) is None


def test_other_inventory_failures_are_not_suppressed() -> None:
    evaluator = _Evaluator("foreign candidate identity")
    with pytest.raises(ValueError, match="foreign candidate identity"):
        _enumerate_candidates(evaluator, {}, SimpleNamespace())


def test_representable_inventory_remains_exact() -> None:
    incumbent = SimpleNamespace()
    assert _enumerate_candidates(_Evaluator(), {}, incumbent) == (incumbent,)

