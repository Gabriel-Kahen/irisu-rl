#include "irisu/physics.hpp"
#include "irisu/simulator.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

namespace {

void rejected(const std::function<void(irisu::MechanicsConfig&)>& mutate) {
  irisu::MechanicsConfig config;
  mutate(config);
  bool simulator_rejected = false;
  try {
    irisu::Simulator simulator(config);
  } catch (const std::invalid_argument&) {
    simulator_rejected = true;
  }
  assert(simulator_rejected);

  bool physics_rejected = false;
  try {
    irisu::PhysicsWorld physics(config);
  } catch (const std::invalid_argument&) {
    physics_rejected = true;
  }
  assert(physics_rejected);
}

}  // namespace

int main() {
  irisu::Simulator valid;
  irisu::PhysicsWorld valid_physics(irisu::MechanicsConfig{});

  rejected([](auto& c) { c.world_min_x = 700.0; });
  rejected([](auto& c) { c.world_max_y = c.world_min_y; });
  rejected([](auto& c) { c.field_top_width = -1.0; });
  rejected([](auto& c) { c.field_bottom_height = 0.0; });
  rejected([](auto& c) {
    c.out_of_bounds_min_x = c.out_of_bounds_max_x;
  });
  rejected([](auto& c) {
    c.gravity_y = std::numeric_limits<double>::infinity();
  });
  rejected([](auto& c) {
    c.piece_friction = std::numeric_limits<double>::quiet_NaN();
  });
  rejected([](auto& c) { c.projectile_restitution = 1.01; });
  rejected([](auto& c) { c.piece_life_ticks = 0; });
  rejected([](auto& c) {
    c.projectile_life_ticks =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) +
        1U;
  });
  rejected([](auto& c) { c.maximum_level = 101; });
  rejected([](auto& c) { c.qualifying_clears_per_level = 0; });
  rejected([](auto& c) {
    c.initial_rotten_count =
        irisu::MechanicsConfig::actor_pool_capacity -
        irisu::MechanicsConfig::static_fixture_count + 1U;
    c.initial_falling_count = 0;
  });
  rejected([](auto& c) {
    c.initial_rotten_count =
        irisu::MechanicsConfig::actor_pool_capacity -
        irisu::MechanicsConfig::static_fixture_count;
    c.initial_falling_count = 1;
  });
  rejected([](auto& c) {
    c.initial_rotten_y = std::numeric_limits<double>::infinity();
  });
  rejected([](auto& c) {
    c.initial_falling_y = std::numeric_limits<double>::max();
  });
  rejected([](auto& c) {
    c.piece_size_weights = {
        std::numeric_limits<std::uint32_t>::max(), 1U, 0U};
  });
  rejected([](auto& c) {
    c.passive_gauge_decay_per_tick =
        std::numeric_limits<std::int64_t>::max() / 30 + 1;
  });
  rejected([](auto& c) {
    c.shape_random_max =
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) +
        1U;
  });
  rejected([](auto& c) { c.field_x = 10'000.0; });
  rejected([](auto& c) {
    c.piece_sizes[0] = std::numeric_limits<float>::max();
  });
  rejected([](auto& c) {
    c.piece_friction =
        2.0 * std::sqrt(static_cast<double>(std::numeric_limits<float>::max()));
  });
  rejected([](auto& c) {
    c.weak_projectile_vy =
        2.0 * static_cast<double>(std::numeric_limits<float>::max());
  });
  rejected([](auto& c) {
    c.spawn_y = std::numeric_limits<float>::max();
    c.scripted_fall_speed = std::numeric_limits<float>::max();
  });
  rejected([](auto& c) {
    c.initial_rotten_count =
        irisu::MechanicsConfig::actor_pool_capacity;
  });
  rejected([](auto& c) {
    c.initial_falling_y = std::numeric_limits<float>::max();
    c.scripted_fall_speed = std::numeric_limits<float>::max();
  });
}
