"""Causal, fail-closed gameplay cadence and action-effect timing."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil, isfinite
from numbers import Real


class TimingError(ValueError):
    """Raised when timestamps or causal timing state are unsafe."""


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise TimingError(f"{name} must be finite")
    return result


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise TimingError(f"{name} must be positive")
    return value


def _median(values: tuple[float, ...]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    return (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )


@dataclass(frozen=True, slots=True)
class CadenceConfig:
    expected_period: float = 0.020
    max_period_deviation: float = 0.006
    stall_after: float = 0.250
    max_duplicate_run: int = 4
    min_samples: int = 3
    history_size: int = 64

    def __post_init__(self) -> None:
        for name in ("expected_period", "max_period_deviation", "stall_after"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        for name in ("max_duplicate_run", "min_samples", "history_size"):
            object.__setattr__(
                self, name, _positive_int(getattr(self, name), name)
            )
        if self.expected_period <= 0.0:
            raise TimingError("expected_period must be positive")
        if not 0.0 < self.max_period_deviation < self.expected_period:
            raise TimingError("max_period_deviation must be between zero and period")
        if self.stall_after <= self.expected_period:
            raise TimingError("stall_after must exceed expected_period")
        if self.history_size < self.min_samples:
            raise TimingError("history_size must be at least min_samples")


@dataclass(frozen=True, slots=True)
class FrameAssessment:
    timestamp: float
    classification: str
    safe_to_act: bool
    duplicate_run: int = 0
    dropped_frames: int = 0
    reason: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", _number(self.timestamp, "timestamp"))
        if not isinstance(self.classification, str):
            raise TypeError("classification must be a string")
        if not self.classification:
            raise TimingError("classification must not be empty")
        if not isinstance(self.safe_to_act, bool):
            raise TypeError("safe_to_act must be bool")
        for name in ("duplicate_run", "dropped_frames"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise TimingError(f"{name} must be non-negative")
        if self.reason is not None and not isinstance(self.reason, str):
            raise TypeError("reason must be a string or None")


@dataclass(frozen=True, slots=True)
class CadencePosterior:
    period: float
    period_uncertainty: float
    phase_timestamp: float
    phase_uncertainty: float
    confidence: float
    sample_count: int
    plausible_periods: tuple[float, ...]

    def __post_init__(self) -> None:
        for name in (
            "period",
            "period_uncertainty",
            "phase_timestamp",
            "phase_uncertainty",
            "confidence",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.period <= 0.0:
            raise TimingError("period must be positive")
        if self.period_uncertainty < 0.0 or self.phase_uncertainty < 0.0:
            raise TimingError("uncertainty must be non-negative")
        if not 0.0 <= self.confidence <= 1.0:
            raise TimingError("confidence must be in [0, 1]")
        _positive_int(self.sample_count, "sample_count")
        if not isinstance(self.plausible_periods, tuple) or not self.plausible_periods:
            raise TimingError("plausible_periods must be a non-empty tuple")
        plausible = tuple(
            _number(value, "plausible_period") for value in self.plausible_periods
        )
        if any(value <= 0.0 for value in plausible):
            raise TimingError("plausible periods must be positive")
        object.__setattr__(self, "plausible_periods", plausible)


@dataclass(frozen=True, slots=True)
class PollEffectEstimate:
    request_at: float
    injected_at: float
    earliest_effect_at: float
    latest_effect_at: float
    period: float
    confidence: float

    def __post_init__(self) -> None:
        for name in (
            "request_at",
            "injected_at",
            "earliest_effect_at",
            "latest_effect_at",
            "period",
            "confidence",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.injected_at < self.request_at:
            raise TimingError("injection cannot precede request")
        if self.earliest_effect_at < self.injected_at:
            raise TimingError("effect interval cannot precede injection")
        if self.latest_effect_at < self.earliest_effect_at:
            raise TimingError("effect interval is reversed")
        if self.period <= 0.0 or not 0.0 <= self.confidence <= 1.0:
            raise TimingError("invalid effect posterior")


@dataclass(frozen=True, slots=True)
class VisibleConfirmation:
    observed_at: float
    earliest_effect_to_visible: float
    latest_effect_to_visible: float

    def __post_init__(self) -> None:
        for name in (
            "observed_at",
            "earliest_effect_to_visible",
            "latest_effect_to_visible",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.earliest_effect_to_visible < 0.0:
            raise TimingError("visible confirmation cannot precede the effect interval")
        if self.latest_effect_to_visible < self.earliest_effect_to_visible:
            raise TimingError("visible latency interval is reversed")


@dataclass(frozen=True, slots=True)
class ActionTiming:
    effect: PollEffectEstimate
    first_visible: VisibleConfirmation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.effect, PollEffectEstimate):
            raise TypeError("effect must be PollEffectEstimate")
        if self.first_visible is not None and not isinstance(
            self.first_visible, VisibleConfirmation
        ):
            raise TypeError("first_visible must be VisibleConfirmation or None")

    def confirm_visible(self, observed_at: float) -> "ActionTiming":
        timestamp = _number(observed_at, "observed_at")
        if self.first_visible is not None:
            raise TimingError("first-visible confirmation is already recorded")
        if timestamp < self.effect.latest_effect_at:
            raise TimingError("visible confirmation overlaps the effect posterior")
        confirmation = VisibleConfirmation(
            timestamp,
            timestamp - self.effect.latest_effect_at,
            timestamp - self.effect.earliest_effect_at,
        )
        return ActionTiming(self.effect, confirmation)


@dataclass(frozen=True, slots=True)
class GameplayClock:
    """Immutable estimator updated only by completed, timestamped frames."""

    config: CadenceConfig = CadenceConfig()
    last_timestamp: float | None = None
    last_unique_timestamp: float | None = None
    last_content_hash: str | None = None
    period_samples: tuple[float, ...] = ()
    duplicate_run: int = 0
    generation: int = 0
    failed_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, CadenceConfig):
            raise TypeError("config must be CadenceConfig")
        for name in ("last_timestamp", "last_unique_timestamp"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _number(value, name))
        if self.last_content_hash is not None and (
            not isinstance(self.last_content_hash, str) or not self.last_content_hash
        ):
            raise TimingError("last_content_hash must be a non-empty string or None")
        if not isinstance(self.period_samples, tuple):
            raise TypeError("period_samples must be a tuple")
        checked_samples = tuple(
            _number(value, "period_sample") for value in self.period_samples
        )
        if any(value <= 0.0 for value in checked_samples):
            raise TimingError("period samples must be positive")
        object.__setattr__(self, "period_samples", checked_samples)
        for name in ("duplicate_run", "generation"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TimingError(f"{name} must be a non-negative integer")
        if self.failed_reason is not None and not isinstance(self.failed_reason, str):
            raise TypeError("failed_reason must be a string or None")

    def _fit_tolerance(self, period: float) -> float:
        return min(self.config.max_period_deviation, period * 0.1)

    def _period_candidates(self) -> tuple[float, ...]:
        if not self.period_samples:
            return (self.config.expected_period,)
        candidates = [self.config.expected_period]
        for gap in self.period_samples:
            candidates.extend(gap / divisor for divisor in range(1, 9))
        lower = self.config.expected_period / 2.0
        upper = self.config.expected_period * 2.0
        candidates = [
            candidate for candidate in candidates if lower <= candidate <= upper
        ]
        candidates.sort(reverse=True)
        unique: list[float] = []
        for candidate in candidates:
            if not unique or abs(candidate - unique[-1]) > 1e-12:
                unique.append(candidate)
        return tuple(unique)

    @staticmethod
    def _normalized_gap(gap: float, period: float) -> tuple[float, int]:
        multiple = max(1, int(gap / period + 0.5))
        return gap / multiple, multiple

    def _candidate_error(self, candidate: float) -> tuple[int, float]:
        errors = tuple(
            abs(self._normalized_gap(gap, candidate)[0] - candidate)
            for gap in self.period_samples
        )
        explained = sum(error <= self._fit_tolerance(candidate) for error in errors)
        return explained, sum(errors)

    def _period(self) -> float:
        scores = {
            candidate: self._candidate_error(candidate)
            for candidate in self._period_candidates()
        }
        best_count = max(score[0] for score in scores.values())
        best_error = min(
            score[1] for score in scores.values() if score[0] == best_count
        )
        tied = (
            candidate
            for candidate, (count, error) in scores.items()
            if count == best_count and error <= best_error + 1e-9
        )
        return max(tied)

    def _plausible_periods(self, period: float) -> tuple[float, ...]:
        required = len(self.period_samples)
        plausible = tuple(
            candidate
            for candidate in self._period_candidates()
            if self._candidate_error(candidate)[0] == required
        )
        return plausible or (period,)

    def posterior(self) -> CadencePosterior:
        if self.failed_reason is not None:
            raise TimingError(f"clock is fail-closed: {self.failed_reason}")
        if (
            len(self.period_samples) < self.config.min_samples
            or self.last_unique_timestamp is None
        ):
            raise TimingError("insufficient cadence evidence")
        period = self._period()
        deviations = tuple(
            abs(self._normalized_gap(gap, period)[0] - period)
            for gap in self.period_samples
        )
        plausible = self._plausible_periods(period)
        ambiguity = max(abs(period - candidate) for candidate in plausible)
        uncertainty = max(_median(deviations), ambiguity, 1e-6)
        confidence = min(
            1.0,
            len(self.period_samples) / self.config.history_size,
        )
        confidence *= max(
            0.0, 1.0 - _median(deviations) / self._fit_tolerance(period)
        )
        confidence /= len(plausible)
        return CadencePosterior(
            period,
            uncertainty,
            self.last_unique_timestamp,
            max(period / 2.0, uncertainty),
            confidence,
            len(self.period_samples),
            plausible,
        )

    def observe(
        self,
        timestamp: float,
        content_hash: str,
        *,
        restart: bool = False,
    ) -> tuple["GameplayClock", FrameAssessment]:
        at = _number(timestamp, "timestamp")
        if not isinstance(content_hash, str):
            raise TypeError("content_hash must be a string")
        if not content_hash:
            raise TimingError("content_hash must not be empty")
        if not isinstance(restart, bool):
            raise TypeError("restart must be bool")
        if self.last_timestamp is not None and at <= self.last_timestamp:
            failed = replace(self, failed_reason="out-of-order frame")
            return failed, FrameAssessment(
                at, "out_of_order", False, reason=failed.failed_reason
            )
        if restart:
            restarted = GameplayClock(
                config=self.config,
                last_timestamp=at,
                last_unique_timestamp=at,
                last_content_hash=content_hash,
                generation=self.generation + 1,
            )
            return restarted, FrameAssessment(
                at, "restart", False, reason="cadence evidence reset"
            )
        if self.failed_reason is not None:
            return replace(self, last_timestamp=at), FrameAssessment(
                at, "latched_failure", False, reason=self.failed_reason
            )
        if self.last_timestamp is None:
            started = replace(
                self,
                last_timestamp=at,
                last_unique_timestamp=at,
                last_content_hash=content_hash,
            )
            return started, FrameAssessment(
                at, "warmup", False, reason="insufficient cadence evidence"
            )

        frame_gap = at - self.last_timestamp
        if frame_gap > self.config.stall_after:
            failed = replace(
                self, last_timestamp=at, failed_reason="capture stall"
            )
            return failed, FrameAssessment(
                at, "stall", False, reason=failed.failed_reason
            )
        if content_hash == self.last_content_hash:
            duplicate_run = self.duplicate_run + 1
            if duplicate_run > self.config.max_duplicate_run:
                failed = replace(
                    self,
                    last_timestamp=at,
                    duplicate_run=duplicate_run,
                    failed_reason="duplicate-frame limit exceeded",
                )
                return failed, FrameAssessment(
                    at,
                    "duplicate_limit",
                    False,
                    duplicate_run=duplicate_run,
                    reason=failed.failed_reason,
                )
            updated = replace(
                self, last_timestamp=at, duplicate_run=duplicate_run
            )
            ready = len(updated.period_samples) >= self.config.min_samples
            return updated, FrameAssessment(
                at,
                "duplicate",
                ready,
                duplicate_run=duplicate_run,
                reason=None if ready else "insufficient cadence evidence",
            )

        assert self.last_unique_timestamp is not None
        unique_gap = at - self.last_unique_timestamp
        samples = (self.period_samples + (unique_gap,))[-self.config.history_size :]
        updated = replace(
            self,
            last_timestamp=at,
            last_unique_timestamp=at,
            last_content_hash=content_hash,
            period_samples=samples,
            duplicate_run=0,
        )
        period = updated._period()
        normalized, multiple = updated._normalized_gap(unique_gap, period)
        dropped = multiple - 1
        delayed = abs(normalized - period) > updated._fit_tolerance(period)
        if delayed:
            return updated, FrameAssessment(
                at,
                "delayed",
                False,
                reason="frame cadence outside calibrated tolerance",
            )
        classification = "dropped" if dropped else "unique"
        ready = len(samples) >= self.config.min_samples and dropped == 0
        return updated, FrameAssessment(
            at,
            classification,
            ready,
            dropped_frames=dropped,
            reason=None if ready else (
                "dropped frame" if dropped else "insufficient cadence evidence"
            ),
        )

    def infer_poll_effect(
        self, *, request_at: float, injected_at: float
    ) -> PollEffectEstimate:
        request = _number(request_at, "request_at")
        injected = _number(injected_at, "injected_at")
        if injected < request:
            raise TimingError("injection cannot precede request")
        if self.last_timestamp is None or request < self.last_timestamp:
            raise TimingError("action request predates the latest completed frame")
        if request - self.last_timestamp > self.config.stall_after:
            raise TimingError("latest completed frame is stale")
        posterior = self.posterior()
        steps = max(
            1,
            ceil((injected - posterior.phase_timestamp) / posterior.period),
        )
        center = posterior.phase_timestamp + steps * posterior.period
        phase_error = posterior.phase_uncertainty + steps * posterior.period_uncertainty
        earliest = max(injected, center - phase_error)
        latest = max(earliest, center + phase_error)
        latest = max(latest, earliest + posterior.period)
        return PollEffectEstimate(
            request,
            injected,
            earliest,
            latest,
            posterior.period,
            posterior.confidence,
        )
