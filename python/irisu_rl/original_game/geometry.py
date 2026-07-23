"""Fail-closed image, client, and claimed-window geometry calibration."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot, isfinite
from numbers import Real
from typing import Iterable

Point = tuple[float, float]


class GeometryError(ValueError):
    """Raised when geometry is invalid or no longer safe for input."""


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    result = float(value)
    if not isfinite(result):
        raise GeometryError(f"{name} must be finite")
    return result


def _point(value: object, name: str) -> Point:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-number point")
    return (_number(value[0], f"{name}.x"), _number(value[1], f"{name}.y"))


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "width", "height"):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.width <= 0.0 or self.height <= 0.0:
            raise GeometryError("rectangle width and height must be positive")

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    def contains(self, point: Point, *, tolerance: float = 0.0) -> bool:
        px, py = _point(point, "point")
        margin = _number(tolerance, "tolerance")
        if margin < 0.0:
            raise GeometryError("tolerance must be non-negative")
        return (
            self.x - margin <= px < self.right + margin
            and self.y - margin <= py < self.bottom + margin
        )

    def max_delta(self, other: "Rect") -> float:
        if not isinstance(other, Rect):
            raise TypeError("other must be a Rect")
        return max(
            abs(self.x - other.x),
            abs(self.y - other.y),
            abs(self.width - other.width),
            abs(self.height - other.height),
        )


@dataclass(frozen=True, slots=True)
class Affine2D:
    """Map ``(x, y)`` to ``(a*x + b*y + tx, c*x + d*y + ty)``."""

    a: float
    b: float
    c: float
    d: float
    tx: float
    ty: float

    def __post_init__(self) -> None:
        for name in ("a", "b", "c", "d", "tx", "ty"):
            object.__setattr__(self, name, _number(getattr(self, name), name))

    def apply(self, point: Point) -> Point:
        x, y = _point(point, "point")
        return (self.a * x + self.b * y + self.tx, self.c * x + self.d * y + self.ty)

    def compose(self, other: "Affine2D") -> "Affine2D":
        """Return ``self(other(point))``."""
        if not isinstance(other, Affine2D):
            raise TypeError("other must be an Affine2D")
        return Affine2D(
            self.a * other.a + self.b * other.c,
            self.a * other.b + self.b * other.d,
            self.c * other.a + self.d * other.c,
            self.c * other.b + self.d * other.d,
            self.a * other.tx + self.b * other.ty + self.tx,
            self.c * other.tx + self.d * other.ty + self.ty,
        )

    def inverse(self) -> "Affine2D":
        determinant = self.a * self.d - self.b * self.c
        scale = max(abs(self.a), abs(self.b), abs(self.c), abs(self.d), 1.0)
        if abs(determinant) <= 1e-12 * scale * scale:
            raise GeometryError("affine transform is singular")
        a = self.d / determinant
        b = -self.b / determinant
        c = -self.c / determinant
        d = self.a / determinant
        return Affine2D(
            a,
            b,
            c,
            d,
            -(a * self.tx + b * self.ty),
            -(c * self.tx + d * self.ty),
        )


@dataclass(frozen=True, slots=True)
class ResidualStats:
    count: int
    median: float
    p95: float
    worst: float

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int):
            raise TypeError("count must be an integer")
        if self.count < 1:
            raise GeometryError("count must be positive")
        for name in ("median", "p95", "worst"):
            value = _number(getattr(self, name), name)
            if value < 0.0:
                raise GeometryError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        if not self.median <= self.p95 <= self.worst:
            raise GeometryError("residual quantiles must be ordered")

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "ResidualStats":
        ordered = sorted(_number(value, "residual") for value in values)
        if not ordered:
            raise GeometryError("at least one residual is required")
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
        p95 = ordered[ceil(0.95 * len(ordered)) - 1]
        return cls(len(ordered), median, p95, ordered[-1])


def _solve_3x3(matrix: list[list[float]], values: list[float]) -> tuple[float, ...]:
    augmented = [row[:] + [value] for row, value in zip(matrix, values, strict=True)]
    scale = max(abs(value) for row in matrix for value in row)
    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12 * max(scale, 1.0):
            raise GeometryError("calibration points are collinear or ill-conditioned")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(3):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                current - factor * source
                for current, source in zip(augmented[row], augmented[column], strict=True)
            ]
    return tuple(augmented[row][3] for row in range(3))


def fit_affine(pairs: Iterable[tuple[Point, Point]]) -> tuple[Affine2D, ResidualStats]:
    """Least-squares fit an image-to-client affine transform."""
    checked: list[tuple[Point, Point]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise TypeError(f"pair {index} must contain source and target points")
        checked.append(
            (
                _point(pair[0], f"pair[{index}].source"),
                _point(pair[1], f"pair[{index}].target"),
            )
        )
    if len(checked) < 3:
        raise GeometryError("at least three point pairs are required")

    mean_x = sum(source[0] for source, _ in checked) / len(checked)
    mean_y = sum(source[1] for source, _ in checked) / len(checked)
    sxx = sum((source[0] - mean_x) ** 2 for source, _ in checked)
    syy = sum((source[1] - mean_y) ** 2 for source, _ in checked)
    sxy = sum(
        (source[0] - mean_x) * (source[1] - mean_y) for source, _ in checked
    )
    if sxx * syy - sxy * sxy <= 1e-12 * max(sxx * syy, 1.0):
        raise GeometryError("calibration source points must be non-collinear")

    sum_x = sum(source[0] for source, _ in checked)
    sum_y = sum(source[1] for source, _ in checked)
    sum_xx = sum(source[0] ** 2 for source, _ in checked)
    sum_xy = sum(source[0] * source[1] for source, _ in checked)
    sum_yy = sum(source[1] ** 2 for source, _ in checked)
    matrix = [
        [sum_xx, sum_xy, sum_x],
        [sum_xy, sum_yy, sum_y],
        [sum_x, sum_y, float(len(checked))],
    ]
    x_values = [
        sum(source[0] * target[0] for source, target in checked),
        sum(source[1] * target[0] for source, target in checked),
        sum(target[0] for _, target in checked),
    ]
    y_values = [
        sum(source[0] * target[1] for source, target in checked),
        sum(source[1] * target[1] for source, target in checked),
        sum(target[1] for _, target in checked),
    ]
    a, b, tx = _solve_3x3(matrix, x_values)
    c, d, ty = _solve_3x3(matrix, y_values)
    transform = Affine2D(a, b, c, d, tx, ty)
    residuals = ResidualStats.from_values(
        hypot(mapped[0] - target[0], mapped[1] - target[1])
        for source, target in checked
        for mapped in (transform.apply(source),)
    )
    return transform, residuals


@dataclass(frozen=True, slots=True)
class WindowGeometry:
    window_id: str
    capture_id: str
    outer: Rect
    client: Rect
    capture: Rect

    def __post_init__(self) -> None:
        for name in ("window_id", "capture_id"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            if not value:
                raise GeometryError(f"{name} must not be empty")
        for name in ("outer", "client", "capture"):
            if not isinstance(getattr(self, name), Rect):
                raise TypeError(f"{name} must be a Rect")
        if not (
            self.outer.x <= self.client.x
            and self.outer.y <= self.client.y
            and self.client.right <= self.outer.right
            and self.client.bottom <= self.outer.bottom
        ):
            raise GeometryError("client rectangle must be inside the claimed window")

    def drift_from(self, other: "WindowGeometry") -> float:
        if not isinstance(other, WindowGeometry):
            raise TypeError("other must be WindowGeometry")
        if self.window_id != other.window_id or self.capture_id != other.capture_id:
            raise GeometryError("window or capture identity changed")
        return max(
            self.outer.max_delta(other.outer),
            self.client.max_delta(other.client),
            self.capture.max_delta(other.capture),
        )


@dataclass(frozen=True, slots=True)
class Calibration:
    image_to_client: Affine2D
    residuals: ResidualStats
    geometry: WindowGeometry
    created_at: float
    max_age: float
    max_geometry_drift: float
    max_anchor_drift: float
    max_residual: float

    def __post_init__(self) -> None:
        if not isinstance(self.image_to_client, Affine2D):
            raise TypeError("image_to_client must be Affine2D")
        if not isinstance(self.residuals, ResidualStats):
            raise TypeError("residuals must be ResidualStats")
        if not isinstance(self.geometry, WindowGeometry):
            raise TypeError("geometry must be WindowGeometry")
        for name in (
            "created_at",
            "max_age",
            "max_geometry_drift",
            "max_anchor_drift",
            "max_residual",
        ):
            object.__setattr__(self, name, _number(getattr(self, name), name))
        if self.max_age <= 0.0:
            raise GeometryError("max_age must be positive")
        if (
            self.max_geometry_drift < 0.0
            or self.max_anchor_drift < 0.0
            or self.max_residual < 0.0
        ):
            raise GeometryError("geometry limits must be non-negative")
        if self.residuals.worst > self.max_residual:
            raise GeometryError("calibration residual exceeds acceptance limit")

    @classmethod
    def fit(
        cls,
        pairs: Iterable[tuple[Point, Point]],
        geometry: WindowGeometry,
        *,
        created_at: float,
        max_age: float,
        max_geometry_drift: float,
        max_anchor_drift: float,
        max_residual: float,
    ) -> "Calibration":
        transform, residuals = fit_affine(pairs)
        return cls(
            transform,
            residuals,
            geometry,
            created_at,
            max_age,
            max_geometry_drift,
            max_anchor_drift,
            max_residual,
        )

    def validate(
        self,
        *,
        now: float,
        geometry: WindowGeometry,
        anchor_drift: float,
    ) -> None:
        checked_now = _number(now, "now")
        checked_anchor = _number(anchor_drift, "anchor_drift")
        if checked_now < self.created_at or checked_now - self.created_at > self.max_age:
            raise GeometryError("calibration is stale")
        if checked_anchor < 0.0 or checked_anchor > self.max_anchor_drift:
            raise GeometryError("anchor drift exceeds calibration limit")
        if geometry.drift_from(self.geometry) > self.max_geometry_drift:
            raise GeometryError("window or capture geometry drifted")

    def to_client(
        self,
        point: Point,
        *,
        now: float,
        geometry: WindowGeometry,
        anchor_drift: float,
    ) -> Point:
        self.validate(now=now, geometry=geometry, anchor_drift=anchor_drift)
        checked_point = _point(point, "point")
        if not geometry.capture.contains(checked_point):
            raise GeometryError("image point is outside the captured surface")
        result = self.image_to_client.apply(checked_point)
        local_client = Rect(0.0, 0.0, geometry.client.width, geometry.client.height)
        if not local_client.contains(result):
            raise GeometryError("transformed point is outside the client")
        return result

    def to_window_local(
        self,
        point: Point,
        *,
        now: float,
        geometry: WindowGeometry,
        anchor_drift: float,
    ) -> Point:
        client = self.to_client(
            point, now=now, geometry=geometry, anchor_drift=anchor_drift
        )
        result = (
            client[0] + geometry.client.x - geometry.outer.x,
            client[1] + geometry.client.y - geometry.outer.y,
        )
        local_window = Rect(0.0, 0.0, geometry.outer.width, geometry.outer.height)
        if not local_window.contains(result):
            raise GeometryError("client point is outside the claimed window")
        return result
