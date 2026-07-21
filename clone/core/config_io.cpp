#include "irisu/config_io.hpp"

#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace irisu {
namespace {

template <typename T>
T integral_override(std::string_view key, double value) {
  static_assert(std::is_integral_v<T>);
  if (!std::isfinite(value) || std::trunc(value) != value || value < 0.0 ||
      static_cast<long double>(value) >
          static_cast<long double>(std::numeric_limits<T>::max())) {
    throw std::invalid_argument(
        "configuration override must be an in-range integer: " +
        std::string(key));
  }
  return static_cast<T>(value);
}

}  // namespace

void apply_config_override(MechanicsConfig& config, std::string_view key,
                           double value) {
  const auto real = [&] {
    if (!std::isfinite(value)) {
      throw std::invalid_argument("configuration override must be finite: " +
                                  std::string(key));
    }
    return value;
  };
#define IRISU_REAL(name) \
  if (key == #name) {    \
    config.name = real(); \
    return;               \
  }
#define IRISU_UINT(name)                                            \
  if (key == #name) {                                               \
    config.name = integral_override<decltype(config.name)>(key, value); \
    return;                                                         \
  }
#define IRISU_I64(name) IRISU_UINT(name)
  IRISU_REAL(tick_seconds)
  IRISU_UINT(solver_iterations)
  IRISU_REAL(world_magnification)
  IRISU_REAL(world_min_x)
  IRISU_REAL(world_min_y)
  IRISU_REAL(world_max_x)
  IRISU_REAL(world_max_y)
  IRISU_REAL(field_x)
  IRISU_REAL(field_y)
  IRISU_REAL(field_width)
  IRISU_REAL(field_height)
  IRISU_REAL(field_blank)
  IRISU_REAL(field_thickness)
  IRISU_REAL(field_top)
  IRISU_REAL(field_top_width)
  IRISU_REAL(field_top_height)
  IRISU_REAL(field_bottom_height)
  IRISU_REAL(client_width)
  IRISU_REAL(client_height)
  IRISU_REAL(side_wall_top)
  IRISU_REAL(side_wall_bottom)
  IRISU_REAL(cleanup_margin_x)
  IRISU_REAL(cleanup_margin_y)
  IRISU_REAL(floor_contact_tolerance)
  IRISU_REAL(out_of_bounds_min_x)
  IRISU_REAL(out_of_bounds_max_x)
  IRISU_REAL(out_of_bounds_min_y)
  IRISU_REAL(out_of_bounds_max_y)
  IRISU_REAL(gravity_y)
  IRISU_REAL(linear_damping)
  IRISU_REAL(angular_damping)
  IRISU_REAL(scripted_fall_speed)
  IRISU_REAL(piece_density)
  IRISU_REAL(piece_friction)
  IRISU_REAL(piece_restitution)
  IRISU_UINT(piece_life_ticks)
  IRISU_UINT(rot_delay_ticks)
  IRISU_UINT(deletion_delay_ticks)
  IRISU_REAL(projectile_size)
  IRISU_REAL(projectile_density)
  IRISU_REAL(projectile_friction)
  IRISU_REAL(projectile_restitution)
  IRISU_UINT(projectile_life_ticks)
  IRISU_REAL(weak_projectile_vy)
  IRISU_REAL(strong_projectile_vy)
  IRISU_UINT(click_cooldown_ticks)
  IRISU_REAL(bonus_size)
  IRISU_REAL(bonus_density)
  IRISU_REAL(bonus_friction)
  IRISU_REAL(bonus_restitution)
  IRISU_I64(gauge_max)
  IRISU_I64(gauge_initial)
  IRISU_I64(gauge_clear_unit)
  IRISU_I64(rotten_penalty)
  IRISU_I64(passive_gauge_decay_per_tick)
  IRISU_UINT(spawn_interval_ticks)
  IRISU_UINT(bonus_interval_spawns)
  IRISU_UINT(starting_colors)
  IRISU_UINT(maximum_colors)
  IRISU_UINT(qualifying_clears_per_level)
  IRISU_UINT(special_clear_base)
  IRISU_UINT(special_clear_random_max)
  IRISU_UINT(initial_rotten_count)
  IRISU_UINT(initial_falling_count)
  IRISU_REAL(initial_rotten_y)
  IRISU_REAL(initial_falling_y)
  IRISU_REAL(spawn_y)
  IRISU_UINT(rotation_random_max)
  IRISU_UINT(shape_random_max)
  IRISU_UINT(color_level_stride)
  IRISU_UINT(score_per_level)
  IRISU_UINT(maximum_level)
  IRISU_UINT(spawn_acceleration_level_stride)
  IRISU_REAL(chain_score_exponent)
  IRISU_UINT(max_episode_ticks)
#undef IRISU_I64
#undef IRISU_UINT
#undef IRISU_REAL
  for (std::size_t index = 0; index < config.piece_sizes.size(); ++index) {
    const std::string suffix = "[" + std::to_string(index) + "]";
    if (key == "piece_sizes" + suffix) {
      config.piece_sizes[index] = real();
      return;
    }
    if (key == "piece_size_weights" + suffix) {
      config.piece_size_weights[index] =
          integral_override<std::uint32_t>(key, value);
      return;
    }
  }
  for (std::size_t index = 0; index < config.size_score_values.size(); ++index) {
    const std::string suffix = "[" + std::to_string(index) + "]";
    if (key == "size_score_values" + suffix) {
      config.size_score_values[index] =
          integral_override<std::int64_t>(key, value);
      return;
    }
  }
  throw std::invalid_argument("unknown configuration override: " +
                              std::string(key));
}

std::string mechanics_config_json(const MechanicsConfig& c,
                                  std::uint64_t config_hash) {
  std::ostringstream output;
  output.precision(17);
  const auto array = [&](const auto& values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (index != 0) output << ',';
      output << values[index];
    }
    output << ']';
  };
  output << "{\"schema_version\":" << MechanicsConfig::schema_version
         << ",\"profile_id\":\"v2.03-normal\",\"target_game_version\":"
         << MechanicsConfig::target_game_version
         << ",\"legacy_box2d_revision\":"
         << MechanicsConfig::legacy_box2d_revision
         << ",\"actor_pool_capacity\":" << MechanicsConfig::actor_pool_capacity
         << ",\"dynamic_actor_capacity\":"
         << MechanicsConfig::actor_pool_capacity - 4U
         << ",\"config_hash\":" << config_hash
         << ",\"tick_seconds\":" << c.tick_seconds
         << ",\"solver_iterations\":" << c.solver_iterations
         << ",\"world_magnification\":" << c.world_magnification
         << ",\"world_bounds\":[" << c.world_min_x << ',' << c.world_min_y
         << ',' << c.world_max_x << ',' << c.world_max_y << ']'
         << ",\"world_min_x\":" << c.world_min_x
         << ",\"world_min_y\":" << c.world_min_y
         << ",\"world_max_x\":" << c.world_max_x
         << ",\"world_max_y\":" << c.world_max_y
         << ",\"field\":[" << c.field_x << ',' << c.field_y << ','
         << c.field_width << ',' << c.field_height << ']'
         << ",\"field_blank\":" << c.field_blank
         << ",\"field_thickness\":" << c.field_thickness
         << ",\"field_top\":" << c.field_top
         << ",\"field_top_width\":" << c.field_top_width
         << ",\"field_top_height\":" << c.field_top_height
         << ",\"field_bottom_height\":" << c.field_bottom_height
         << ",\"client\":[" << c.client_width << ',' << c.client_height << ']'
         << ",\"side_wall\":[" << c.side_wall_top << ','
         << c.side_wall_bottom << ']'
         << ",\"side_wall_top\":" << c.side_wall_top
         << ",\"side_wall_bottom\":" << c.side_wall_bottom
         << ",\"cleanup_margin\":[" << c.cleanup_margin_x << ','
         << c.cleanup_margin_y << ']'
         << ",\"cleanup_margin_x\":" << c.cleanup_margin_x
         << ",\"cleanup_margin_y\":" << c.cleanup_margin_y
         << ",\"floor_contact_tolerance\":" << c.floor_contact_tolerance
         << ",\"out_of_bounds\":[" << c.out_of_bounds_min_x << ','
         << c.out_of_bounds_max_x << ',' << c.out_of_bounds_min_y << ','
         << c.out_of_bounds_max_y << ']'
         << ",\"out_of_bounds_min_x\":" << c.out_of_bounds_min_x
         << ",\"out_of_bounds_max_x\":" << c.out_of_bounds_max_x
         << ",\"out_of_bounds_min_y\":" << c.out_of_bounds_min_y
         << ",\"out_of_bounds_max_y\":" << c.out_of_bounds_max_y
         << ",\"gravity_y\":" << c.gravity_y
         << ",\"linear_damping\":" << c.linear_damping
         << ",\"angular_damping\":" << c.angular_damping
         << ",\"scripted_fall_speed\":" << c.scripted_fall_speed
         << ",\"piece_material\":[" << c.piece_density << ','
         << c.piece_friction << ',' << c.piece_restitution << ']'
         << ",\"piece_sizes\":";
  array(c.piece_sizes);
  output << ",\"piece_size_weights\":";
  array(c.piece_size_weights);
  output << ",\"piece_life_ticks\":" << c.piece_life_ticks
         << ",\"rot_delay_ticks\":" << c.rot_delay_ticks
         << ",\"deletion_delay_ticks\":" << c.deletion_delay_ticks
         << ",\"projectile_size\":" << c.projectile_size
         << ",\"projectile_material\":[" << c.projectile_density << ','
         << c.projectile_friction << ',' << c.projectile_restitution << ']'
         << ",\"projectile_life_ticks\":" << c.projectile_life_ticks
         << ",\"projectile_velocity_y\":[" << c.weak_projectile_vy << ','
         << c.strong_projectile_vy << ']'
         << ",\"weak_projectile_vy\":" << c.weak_projectile_vy
         << ",\"strong_projectile_vy\":" << c.strong_projectile_vy
         << ",\"click_cooldown_ticks\":" << c.click_cooldown_ticks
         << ",\"bonus_size\":" << c.bonus_size
         << ",\"bonus_material\":[" << c.bonus_density << ','
         << c.bonus_friction << ',' << c.bonus_restitution << ']'
         << ",\"gauge\":[" << c.gauge_max << ',' << c.gauge_initial << ','
         << c.gauge_clear_unit << ',' << c.rotten_penalty << ','
         << c.passive_gauge_decay_per_tick << ']'
         << ",\"gauge_max\":" << c.gauge_max
         << ",\"gauge_initial\":" << c.gauge_initial
         << ",\"gauge_clear_unit\":" << c.gauge_clear_unit
         << ",\"rotten_penalty\":" << c.rotten_penalty
         << ",\"passive_gauge_decay_per_tick\":"
         << c.passive_gauge_decay_per_tick
         << ",\"spawn_interval_ticks\":" << c.spawn_interval_ticks
         << ",\"bonus_interval_spawns\":" << c.bonus_interval_spawns
         << ",\"colors\":[" << c.starting_colors << ',' << c.maximum_colors
         << ',' << c.color_level_stride << ']'
         << ",\"starting_colors\":" << c.starting_colors
         << ",\"maximum_colors\":" << c.maximum_colors
         << ",\"qualifying_clears_per_level\":"
         << c.qualifying_clears_per_level
         << ",\"special_clear_base\":" << c.special_clear_base
         << ",\"special_clear_random_max\":" << c.special_clear_random_max
         << ",\"initial_rotten_count\":" << c.initial_rotten_count
         << ",\"initial_falling_count\":" << c.initial_falling_count
         << ",\"initial_rotten_y\":" << c.initial_rotten_y
         << ",\"initial_falling_y\":" << c.initial_falling_y
         << ",\"spawn_y\":" << c.spawn_y
         << ",\"rotation_random_max\":" << c.rotation_random_max
         << ",\"shape_random_max\":" << c.shape_random_max
         << ",\"color_level_stride\":" << c.color_level_stride
         << ",\"level\":[" << c.score_per_level << ',' << c.maximum_level
         << ',' << c.spawn_acceleration_level_stride << ']'
         << ",\"score_per_level\":" << c.score_per_level
         << ",\"maximum_level\":" << c.maximum_level
         << ",\"spawn_acceleration_level_stride\":"
         << c.spawn_acceleration_level_stride
         << ",\"size_score_values\":[" << c.size_score_values[0] << ','
         << c.size_score_values[1] << ',' << c.size_score_values[2] << ']'
         << ",\"chain_score_exponent\":" << c.chain_score_exponent
         << ",\"compatibility_only_ignored\":[\"cleanup_margin_x\","
            "\"cleanup_margin_y\",\"floor_contact_tolerance\","
            "\"deletion_delay_ticks\",\"bonus_interval_spawns\","
            "\"click_cooldown_ticks\",\"color_level_stride\","
            "\"score_per_level\",\"spawn_acceleration_level_stride\","
            "\"size_score_values\",\"chain_score_exponent\"]"
         << ",\"deprecated_non_nominal\":[\"cleanup_margin_x\","
            "\"cleanup_margin_y\",\"floor_contact_tolerance\","
            "\"deletion_delay_ticks\",\"bonus_interval_spawns\","
            "\"click_cooldown_ticks\",\"color_level_stride\","
            "\"score_per_level\",\"spawn_acceleration_level_stride\","
            "\"size_score_values\",\"chain_score_exponent\"]"
         << ",\"max_episode_ticks\":" << c.max_episode_ticks << '}';
  return output.str();
}

}  // namespace irisu
