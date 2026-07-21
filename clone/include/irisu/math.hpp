#pragma once

#include <algorithm>
#include <cmath>

namespace irisu {

struct Vec2 {
  double x{};
  double y{};

  constexpr Vec2 operator+(Vec2 rhs) const { return {x + rhs.x, y + rhs.y}; }
  constexpr Vec2 operator-(Vec2 rhs) const { return {x - rhs.x, y - rhs.y}; }
  constexpr Vec2 operator-() const { return {-x, -y}; }
  constexpr Vec2 operator*(double scalar) const { return {x * scalar, y * scalar}; }
  constexpr Vec2 operator/(double scalar) const { return {x / scalar, y / scalar}; }
  constexpr Vec2& operator+=(Vec2 rhs) { x += rhs.x; y += rhs.y; return *this; }
  constexpr Vec2& operator-=(Vec2 rhs) { x -= rhs.x; y -= rhs.y; return *this; }
  constexpr Vec2& operator*=(double scalar) { x *= scalar; y *= scalar; return *this; }
};

constexpr Vec2 operator*(double scalar, Vec2 value) { return value * scalar; }
constexpr double dot(Vec2 a, Vec2 b) { return a.x * b.x + a.y * b.y; }
constexpr double cross(Vec2 a, Vec2 b) { return a.x * b.y - a.y * b.x; }
constexpr Vec2 cross(double scalar, Vec2 value) { return {-scalar * value.y, scalar * value.x}; }
constexpr Vec2 perpendicular(Vec2 value) { return {-value.y, value.x}; }
inline double length_squared(Vec2 value) { return dot(value, value); }
inline double length(Vec2 value) { return std::sqrt(length_squared(value)); }
inline Vec2 normalized(Vec2 value) {
  const double size = length(value);
  return size > 1e-12 ? value / size : Vec2{1.0, 0.0};
}
inline Vec2 rotate(Vec2 value, double angle) {
  const double cosine = std::cos(angle);
  const double sine = std::sin(angle);
  return {cosine * value.x - sine * value.y, sine * value.x + cosine * value.y};
}
inline double clamp(double value, double low, double high) {
  return std::max(low, std::min(value, high));
}

}  // namespace irisu

