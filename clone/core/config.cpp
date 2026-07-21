#include "irisu/config.hpp"

#include "irisu/floating_point.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace irisu {
namespace {

[[noreturn]] void invalid(std::string_view detail) {
  throw std::invalid_argument("invalid mechanics config: " +
                              std::string(detail));
}

void finite(double value, std::string_view name) {
  if (!std::isfinite(value)) invalid(name);
}

void nonnegative(double value, std::string_view name) {
  finite(value, name);
  if (value < 0.0) invalid(name);
}

void positive(double value, std::string_view name) {
  finite(value, name);
  if (value <= 0.0) invalid(name);
}

float legacy_float(double value, std::string_view name) {
  finite(value, name);
  const auto narrowed = static_cast<float>(value);
  if (!std::isfinite(narrowed)) invalid(name);
  return narrowed;
}

float world_value(double value, float magnification, std::string_view name) {
  const auto narrowed = legacy_float(value, name);
  const auto converted = narrowed / magnification;
  if (!std::isfinite(converted)) invalid(name);
  return converted;
}

void positive_world_dimension(double value, float magnification,
                              std::string_view name) {
  positive(value, name);
  if (world_value(value * 0.5, magnification, name) <= 0.0F) invalid(name);
}

float finite_add(float left, float right, std::string_view name) {
  const float result = left + right;
  if (!std::isfinite(result)) invalid(name);
  return result;
}

float finite_multiply(float left, float right, std::string_view name) {
  const float result = left * right;
  if (!std::isfinite(result)) invalid(name);
  return result;
}

void safe_box_math(double size, double density, float magnification,
                   std::string_view name) {
  const float half = world_value(size * 0.5, magnification, name);
  const float square = finite_multiply(half, half, name);
  if (half <= 0.0F || square <= 0.0F ||
      density > static_cast<double>(std::numeric_limits<float>::max() / 4.0F)) {
    invalid(name);
  }
  const long double mass_bound =
      4.0L * static_cast<long double>(density) * half * half;
  const long double inertia_bound = mass_bound * 2.0L * half * half;
  if (mass_bound > std::numeric_limits<float>::max() ||
      inertia_bound > std::numeric_limits<float>::max()) {
    invalid(name);
  }
}

void static_box_in_range(double width, double height, double x, double y,
                         float magnification, float world_min_x,
                         float world_min_y, float world_max_x,
                         float world_max_y, std::string_view name) {
  const float half_x = world_value(width * 0.5, magnification, name);
  const float half_y = world_value(height * 0.5, magnification, name);
  const float center_x = world_value(x, magnification, name);
  const float center_y = world_value(y, magnification, name);
  const float min_x = finite_add(center_x, -half_x, name);
  const float max_x = finite_add(center_x, half_x, name);
  const float min_y = finite_add(center_y, -half_y, name);
  const float max_y = finite_add(center_y, half_y, name);
  if (!(min_x < max_x) || !(min_y < max_y) ||
      finite_multiply(half_x, half_x, name) <= 0.0F ||
      finite_multiply(half_y, half_y, name) <= 0.0F ||
      !(min_x < world_max_x && max_x > world_min_x &&
        min_y < world_max_y && max_y > world_min_y)) {
    invalid(name);
  }
}

}  // namespace

MechanicsConfig validated_mechanics_config(MechanicsConfig config) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  positive(config.tick_seconds, "tick_seconds");
  if (legacy_float(config.tick_seconds, "tick_seconds") <= 0.0F ||
      config.solver_iterations == 0 ||
      config.solver_iterations >
          static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
    invalid("solver timing");
  }
  positive(config.world_magnification, "world_magnification");
  const float magnification =
      legacy_float(config.world_magnification, "world_magnification");
  if (magnification <= 0.0F) invalid("world_magnification");

  const float world_min_x =
      world_value(config.world_min_x, magnification, "world_min_x");
  const float world_min_y =
      world_value(config.world_min_y, magnification, "world_min_y");
  const float world_max_x =
      world_value(config.world_max_x, magnification, "world_max_x");
  const float world_max_y =
      world_value(config.world_max_y, magnification, "world_max_y");
  if (!(world_min_x < world_max_x) || !(world_min_y < world_max_y)) {
    invalid("world bounds must be strictly ordered after float conversion");
  }
  const float world_span_x = world_max_x - world_min_x;
  const float world_span_y = world_max_y - world_min_y;
  if (!std::isfinite(world_span_x) || !std::isfinite(world_span_y) ||
      !std::isfinite(65535.0F / world_span_x) ||
      !std::isfinite(65535.0F / world_span_y)) {
    invalid("world bounds cannot be quantized safely");
  }

  finite(config.field_x, "field_x");
  finite(config.field_y, "field_y");
  positive_world_dimension(config.field_width, magnification, "field_width");
  positive_world_dimension(config.field_height, magnification, "field_height");
  nonnegative(config.field_blank, "field_blank");
  positive_world_dimension(config.field_thickness, magnification,
                           "field_thickness");
  finite(config.field_top, "field_top");
  positive_world_dimension(config.field_top_width, magnification,
                           "field_top_width");
  positive_world_dimension(config.field_top_height, magnification,
                           "field_top_height");
  positive_world_dimension(config.field_bottom_height, magnification,
                           "field_bottom_height");
  positive(config.client_width, "client_width");
  positive(config.client_height, "client_height");
  finite(config.side_wall_top, "side_wall_top");
  finite(config.side_wall_bottom, "side_wall_bottom");
  if (config.side_wall_top > config.side_wall_bottom) invalid("side wall span");
  nonnegative(config.cleanup_margin_x, "cleanup_margin_x");
  nonnegative(config.cleanup_margin_y, "cleanup_margin_y");
  nonnegative(config.floor_contact_tolerance, "floor_contact_tolerance");
  finite(config.out_of_bounds_min_x, "out_of_bounds_min_x");
  finite(config.out_of_bounds_max_x, "out_of_bounds_max_x");
  finite(config.out_of_bounds_min_y, "out_of_bounds_min_y");
  finite(config.out_of_bounds_max_y, "out_of_bounds_max_y");
  if (!(config.out_of_bounds_min_x < config.out_of_bounds_max_x) ||
      !(config.out_of_bounds_min_y < config.out_of_bounds_max_y)) {
    invalid("out-of-bounds ranges must be strictly ordered");
  }

  if (!(config.field_width > config.field_thickness)) {
    invalid("field_width must exceed field_thickness");
  }
  const double spawn_x_span =
      std::trunc(config.field_width - config.field_thickness);
  if (!std::isfinite(spawn_x_span) || spawn_x_span < 0.0 ||
      spawn_x_span >
          static_cast<double>(std::numeric_limits<std::int32_t>::max())) {
    invalid("spawn X range");
  }

  const double half_thickness = std::trunc(config.field_thickness / 2.0);
  const double half_height = std::trunc(config.field_height / 2.0);
  const double half_width = std::trunc(config.field_width / 2.0);
  const double center_x = config.field_x + half_width + config.field_thickness;
  const double left_x = config.field_x + half_thickness;
  const double right_x =
      config.field_x + config.field_width + config.field_thickness;
  const double wall_y = config.field_y + half_height;
  const double floor_width =
      config.field_width + 2.0 * config.field_thickness;
  const double floor_y = config.field_y + config.field_height +
                         config.field_blank +
                         std::trunc(config.field_bottom_height / 2.0);
  for (const auto [value, name] : {
           std::pair{center_x, std::string_view{"field center X"}},
           std::pair{left_x, std::string_view{"left wall X"}},
           std::pair{right_x, std::string_view{"right wall X"}},
           std::pair{wall_y, std::string_view{"wall Y"}},
           std::pair{floor_y, std::string_view{"floor Y"}},
           std::pair{config.field_top, std::string_view{"field_top"}},
       }) {
    (void)world_value(value, magnification, name);
  }
  positive_world_dimension(floor_width, magnification, "floor width");
  static_box_in_range(config.field_thickness, config.field_height, left_x,
                      wall_y, magnification, world_min_x, world_min_y,
                      world_max_x, world_max_y, "left wall AABB");
  static_box_in_range(config.field_thickness, config.field_height, right_x,
                      wall_y, magnification, world_min_x, world_min_y,
                      world_max_x, world_max_y, "right wall AABB");
  static_box_in_range(floor_width, config.field_bottom_height, center_x,
                      floor_y, magnification, world_min_x, world_min_y,
                      world_max_x, world_max_y, "floor AABB");
  static_box_in_range(config.field_top_width, config.field_top_height,
                      center_x, config.field_top, magnification, world_min_x,
                      world_min_y, world_max_x, world_max_y, "top AABB");

  const float gravity =
      world_value(config.gravity_y, magnification, "gravity_y");
  nonnegative(config.linear_damping, "linear_damping");
  nonnegative(config.angular_damping, "angular_damping");
  nonnegative(config.scripted_fall_speed, "scripted_fall_speed");
  (void)legacy_float(config.linear_damping, "linear_damping");
  (void)legacy_float(config.angular_damping, "angular_damping");
  const float scripted_speed =
      legacy_float(config.scripted_fall_speed, "scripted_fall_speed");
  (void)finite_add(legacy_float(config.spawn_y, "spawn_y"), scripted_speed,
                   "scripted_fall_speed update");

  for (const double size : config.piece_sizes) {
    positive_world_dimension(size, magnification, "piece_sizes");
  }
  std::uint64_t weight_total = 0;
  for (const auto weight : config.piece_size_weights) weight_total += weight;
  constexpr std::uint64_t max_rng_range =
      static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max()) + 1U;
  if (weight_total == 0 || weight_total > max_rng_range) {
    invalid("piece_size_weights total");
  }

  const auto material = [](double density, double friction, double restitution,
                           std::string_view name) {
    nonnegative(density, name);
    nonnegative(friction, name);
    nonnegative(restitution, name);
    if (restitution > 1.0) invalid(name);
    (void)legacy_float(density, name);
    (void)legacy_float(friction, name);
    (void)legacy_float(restitution, name);
  };
  material(config.piece_density, config.piece_friction,
           config.piece_restitution, "piece material");
  for (const double size : config.piece_sizes) {
    safe_box_math(size, config.piece_density, magnification,
                  "piece mass and inertia");
  }
  positive_world_dimension(config.projectile_size, magnification,
                           "projectile_size");
  material(config.projectile_density, config.projectile_friction,
           config.projectile_restitution, "projectile material");
  safe_box_math(config.projectile_size, config.projectile_density,
                magnification, "projectile mass and inertia");
  positive_world_dimension(config.bonus_size, magnification, "bonus_size");
  material(config.bonus_density, config.bonus_friction,
           config.bonus_restitution, "bonus material");
  safe_box_math(config.bonus_size, config.bonus_density, magnification,
                "bonus mass and inertia");

  const long double maximum_friction = std::max(
      {static_cast<long double>(config.piece_friction),
       static_cast<long double>(config.projectile_friction),
       static_cast<long double>(config.bonus_friction), 1.0L});
  if (maximum_friction * maximum_friction >
      std::numeric_limits<float>::max()) {
    invalid("friction pair mixing");
  }

  constexpr auto max_lifetime =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  if (config.piece_life_ticks == 0 || config.piece_life_ticks > max_lifetime ||
      config.projectile_life_ticks == 0 ||
      config.projectile_life_ticks > max_lifetime ||
      config.rot_delay_ticks > max_lifetime ||
      config.deletion_delay_ticks > max_lifetime) {
    invalid("actor lifetimes must fit positive int64 counters");
  }
  const float tick = legacy_float(config.tick_seconds, "tick_seconds");
  const float gravity_delta =
      finite_multiply(tick, gravity, "gravity integration");
  for (const auto [velocity, name] : {
           std::pair{config.weak_projectile_vy,
                     std::string_view{"weak_projectile_vy"}},
           std::pair{config.strong_projectile_vy,
                     std::string_view{"strong_projectile_vy"}},
       }) {
    const float native_velocity = world_value(velocity, magnification, name);
    const float accelerated =
        finite_add(native_velocity, gravity_delta, "projectile integration");
    (void)finite_multiply(tick, accelerated, "projectile integration");
  }

  if (config.gauge_max <= 0 || config.gauge_initial <= 0 ||
      config.gauge_initial > config.gauge_max || config.gauge_clear_unit < 0 ||
      config.rotten_penalty < 0 ||
      config.passive_gauge_decay_per_tick < 0 ||
      config.passive_gauge_decay_per_tick >
          std::numeric_limits<std::int64_t>::max() / 30) {
    invalid("gauge configuration");
  }

  constexpr auto max_rng_argument =
      static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
  if (config.spawn_interval_ticks == 0 || config.starting_colors == 0 ||
      config.maximum_colors < config.starting_colors ||
      static_cast<std::uint64_t>(config.maximum_colors) > max_rng_range ||
      config.qualifying_clears_per_level == 0 ||
      config.special_clear_random_max > max_rng_argument ||
      config.rotation_random_max > max_rng_argument ||
      config.shape_random_max > max_rng_argument ||
      config.color_level_stride == 0 || config.score_per_level == 0 ||
      config.maximum_level == 0 || config.maximum_level > 100 ||
      config.spawn_acceleration_level_stride == 0 ||
      config.max_episode_ticks == 0) {
    invalid("spawning, level, RNG, or episode range");
  }
  const std::uint64_t initial_body_count =
      static_cast<std::uint64_t>(config.initial_rotten_count) +
      config.initial_falling_count;
  if (initial_body_count >
      MechanicsConfig::actor_pool_capacity -
          MechanicsConfig::static_fixture_count) {
    invalid("initial body count exceeds the actor pool");
  }
  (void)world_value(config.initial_rotten_y, magnification,
                    "initial_rotten_y");
  (void)world_value(config.initial_falling_y, magnification,
                    "initial_falling_y");
  (void)finite_add(legacy_float(config.initial_falling_y,
                                "initial_falling_y"),
                   scripted_speed, "initial falling actor update");
  (void)world_value(config.spawn_y, magnification, "spawn_y");
  (void)world_value(config.field_x, magnification, "spawn X minimum");
  (void)world_value(config.field_x + spawn_x_span, magnification,
                    "spawn X maximum");
  nonnegative(config.chain_score_exponent, "chain_score_exponent");
  for (const auto score : config.size_score_values) {
    if (score < 0) invalid("size_score_values");
  }

  return config;
}

}  // namespace irisu
