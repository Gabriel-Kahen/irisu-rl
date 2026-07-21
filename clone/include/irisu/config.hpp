#pragma once

#include <array>
#include <cstdint>

namespace irisu {

// Runtime values are centralized here. Nominal defaults reproduce the recovered
// normal path. Fields marked compatibility-only are retained for API stability,
// included in the config hash, and intentionally ignored by faithful gameplay.
struct MechanicsConfig {
  static constexpr std::uint32_t schema_version = 3;
  static constexpr std::uint32_t target_game_version = 203;
  static constexpr std::uint32_t legacy_box2d_revision = 58;
  static constexpr std::uint32_t physics_proxy_capacity = 512;
  static constexpr std::uint32_t static_fixture_count = 4;
  static constexpr std::uint32_t actor_pool_capacity = 200;

  double tick_seconds{0.020};                         // shipped-config
  std::uint32_t solver_iterations{10};               // binary-derived
  double world_magnification{10.0};                  // binary-derived normal-mode call site
  double world_min_x{0.0};                           // binary-derived b2d_init call
  double world_min_y{-200.0};                        // binary-derived b2d_init call
  double world_max_x{640.0};                         // binary-derived b2d_init call
  double world_max_y{480.0};                         // binary-derived b2d_init call

  double field_x{130.0};                             // release-default, pixels
  double field_y{120.0};                             // release-default, pixels
  double field_width{320.0};                         // release-default, pixels
  double field_height{250.0};                        // release-default, pixels
  double field_blank{40.0};                          // release-default, pixels
  double field_thickness{16.0};                      // release-default, pixels
  double field_top{-140.0};                          // release-default, fixture center Y
  double field_top_width{320.0};                     // release-default, full width
  double field_top_height{300.0};                    // release-default, full height
  double field_bottom_height{16.0};                  // release-default, full height
  double client_width{640.0};                        // binary-derived replay coordinate range
  double client_height{480.0};                       // binary-derived replay coordinate range
  double side_wall_top{120.0};                       // exact side fixture top
  double side_wall_bottom{370.0};                    // exact side fixture bottom
  double cleanup_margin_x{0.0};                      // compatibility-only; ignored
  double cleanup_margin_y{0.0};                      // compatibility-only; ignored
  double floor_contact_tolerance{1.0};               // compatibility-only; ignored
  double out_of_bounds_min_x{0.0};                   // binary-derived actor test
  double out_of_bounds_max_x{640.0};                 // binary-derived actor test
  double out_of_bounds_min_y{-30.0};                 // binary-derived actor test
  double out_of_bounds_max_y{560.0};                 // binary-derived actor test

  double gravity_y{160.0};                           // binary-derived normal-mode call site
  double linear_damping{0.0};                        // binary-derived wrapper default
  double angular_damping{0.0};                       // binary-derived wrapper default
  double scripted_fall_speed{0.2};                   // exact level-1 pixels per actor update

  double piece_density{1.0};                         // shipped-config
  double piece_friction{1.0};                        // shipped-config
  double piece_restitution{0.0};                     // shipped-config
  std::array<double, 10> piece_sizes{
      32.0, 46.0, 54.0, 60.0, 72.0, 90.0, 140.0, 5.0, 5.0, 5.0};
  std::array<std::uint32_t, 10> piece_size_weights{
      20, 28, 28, 14, 5, 3, 1, 0, 0, 0};
  std::uint64_t piece_life_ticks{100'000};           // release-default lifetime
  std::uint64_t rot_delay_ticks{40};                 // release-default strict threshold
  std::uint64_t deletion_delay_ticks{0};             // compatibility-only; ignored

  double projectile_size{24.0};                      // shipped-config, fixture meaning uncertain
  double projectile_density{8.0};                    // shipped-config
  double projectile_friction{1.0};                   // shipped-config
  double projectile_restitution{0.0};                // shipped-config
  std::uint64_t projectile_life_ticks{3'000};        // release-default lifetime
  double weak_projectile_vy{-250.0};                 // proven left-edge velocity
  double strong_projectile_vy{-500.0};               // proven right-edge velocity
  std::uint32_t click_cooldown_ticks{0};             // compatibility-only; ignored, no cooldown

  double bonus_size{24.0};                           // shipped-config
  double bonus_density{50.0};                        // shipped-config
  double bonus_friction{0.1};                        // shipped-config
  double bonus_restitution{0.6};                     // shipped-config

  std::int64_t gauge_max{40'000};                    // release-default
  std::int64_t gauge_initial{3'000};                  // release-default
  std::int64_t gauge_clear_unit{700};                // binary-derived normal reward
  std::int64_t rotten_penalty{1'800};                // binary-derived base; +20*level
  std::int64_t passive_gauge_decay_per_tick{1};      // multiplier for exact level formula

  std::uint32_t spawn_interval_ticks{100};           // exact level formula base
  std::uint32_t bonus_interval_spawns{0};            // compatibility-only; ignored
  std::uint32_t starting_colors{3};                  // exact level-1 count
  std::uint32_t maximum_colors{6};                   // exact normal maximum count
  std::uint32_t qualifying_clears_per_level{10};     // release-default
  std::uint32_t special_clear_base{40};              // exact scheduler offset base
  std::uint32_t special_clear_random_max{12};        // inclusive GetRand maximum
  std::uint32_t initial_rotten_count{10};            // binary-derived constructor fill
  std::uint32_t initial_falling_count{10};           // binary-derived constructor fill
  double initial_rotten_y{200.0};                    // exact constructor caller value
  double initial_falling_y{60.0};                    // exact constructor caller value
  double spawn_y{-50.0};                             // exact normal caller value
  std::uint32_t rotation_random_max{1000};           // inclusive GetRand maximum
  std::uint32_t shape_random_max{100};               // inclusive GetRand maximum
  std::uint32_t color_level_stride{10};              // compatibility-only; ignored
  std::uint32_t score_per_level{1'000};              // compatibility-only; ignored
  std::uint32_t maximum_level{100};                  // observed
  std::uint32_t spawn_acceleration_level_stride{4};  // compatibility-only; ignored
  std::array<std::int64_t, 3> size_score_values{20, 28, 40}; // compatibility-only; ignored
  double chain_score_exponent{2.0};                  // compatibility-only; ignored
  std::uint64_t max_episode_ticks{1'000'000};
};

// Validates every runtime field, including values that narrow to the legacy
// engine's float/int types, and returns the config for constructor chaining.
// This must run before constructing a PhysicsWorld.
[[nodiscard]] MechanicsConfig validated_mechanics_config(MechanicsConfig config);

}  // namespace irisu
