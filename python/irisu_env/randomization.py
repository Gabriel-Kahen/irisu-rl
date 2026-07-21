"""Deterministic, uncertainty-aware mechanics configuration sampling."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from .policies import SplitMix64


@dataclass(frozen=True, slots=True)
class ParameterRange:
    low: float
    high: float
    integer: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.low) or not math.isfinite(self.high):
            raise ValueError("parameter range bounds must be finite")
        if self.low > self.high:
            raise ValueError("parameter range low must not exceed high")
        if self.integer and (
            not float(self.low).is_integer() or not float(self.high).is_integer()
        ):
            raise ValueError("integer parameter bounds must be integral")


# Defaults reproduce the recovered v2.03 normal-mode constants exactly.
# Robustness perturbations must be supplied explicitly by the caller.
DEFAULT_TRAINING_RANGES: dict[str, ParameterRange] = {
    "gravity_y": ParameterRange(160.0, 160.0),
    "linear_damping": ParameterRange(0.0, 0.0),
    "angular_damping": ParameterRange(0.0, 0.0),
    "scripted_fall_speed": ParameterRange(0.2, 0.2),
    "piece_density": ParameterRange(1.0, 1.0),
    "piece_friction": ParameterRange(1.0, 1.0),
    "projectile_density": ParameterRange(8.0, 8.0),
    "projectile_friction": ParameterRange(1.0, 1.0),
    "weak_projectile_vy": ParameterRange(-250.0, -250.0),
    "strong_projectile_vy": ParameterRange(-500.0, -500.0),
}


def randomized_config(
    seed: int,
    ranges: Mapping[str, ParameterRange] = DEFAULT_TRAINING_RANGES,
) -> dict[str, float | int]:
    """Sample a reproducible override mapping without exposing game RNG state."""

    rng = SplitMix64(seed)
    sampled: dict[str, float | int] = {}
    for key in sorted(ranges):
        parameter = ranges[key]
        if not isinstance(parameter, ParameterRange):
            raise TypeError("randomization ranges must contain ParameterRange values")
        if parameter.integer:
            low = int(parameter.low)
            high = int(parameter.high)
            sampled[key] = low + rng.bounded(high - low + 1)
        else:
            sampled[key] = parameter.low + (parameter.high - parameter.low) * rng.unit()
    return sampled
