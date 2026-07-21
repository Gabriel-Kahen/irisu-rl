#include "irisu/physics.hpp"

#include <bit>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

irisu::MechanicsConfig config() {
  irisu::MechanicsConfig value;
  value.gravity_y = 0.0;
  value.linear_damping = 0.0;
  value.angular_damping = 0.0;
  value.world_min_x = -2000.0;
  value.world_max_x = 2000.0;
  value.world_min_y = -2000.0;
  value.world_max_y = 2000.0;
  value.field_x = -1000.0;
  value.field_width = 2000.0;
  value.field_y = 0.0;
  value.field_height = 900.0;
  value.field_blank = 0.0;
  return value;
}

irisu::Body body(irisu::BodyId id, irisu::Vec2 position,
                 irisu::Vec2 velocity = {}) {
  irisu::Body value;
  value.id = id;
  value.shape = irisu::Shape::Circle;
  value.lifecycle = irisu::Lifecycle::DynamicFresh;
  value.position = position;
  value.velocity = velocity;
  value.size = 20.0;
  value.density = 1.0;
  value.friction = 1.0;
  return value;
}

bool same_bits(double left, double right) {
  return std::bit_cast<std::uint64_t>(left) ==
         std::bit_cast<std::uint64_t>(right);
}

void require_same_bodies(const std::vector<irisu::Body>& left,
                         const std::vector<irisu::Body>& right) {
  require(left.size() == right.size(), "body count changed after restore");
  for (std::size_t index = 0; index < left.size(); ++index) {
    const auto& a = left[index];
    const auto& b = right[index];
    const auto mismatch = [&](bool matches, std::string_view field) {
      if (!matches) {
        throw std::runtime_error("body " + std::to_string(a.id) + " " +
                                 std::string(field) + " diverged after restore");
      }
    };
    mismatch(a.id == b.id, "id");
    mismatch(same_bits(a.position.x, b.position.x), "position.x");
    mismatch(same_bits(a.position.y, b.position.y), "position.y");
    mismatch(same_bits(a.velocity.x, b.velocity.x), "velocity.x");
    mismatch(same_bits(a.velocity.y, b.velocity.y), "velocity.y");
    mismatch(same_bits(a.angle, b.angle), "angle");
    mismatch(same_bits(a.angular_velocity, b.angular_velocity),
             "angular_velocity");
    mismatch(same_bits(a.native_position.x, b.native_position.x),
             "native_position.x");
    mismatch(same_bits(a.native_position.y, b.native_position.y),
             "native_position.y");
    mismatch(same_bits(a.native_center.x, b.native_center.x),
             "native_center.x");
    mismatch(same_bits(a.native_center.y, b.native_center.y),
             "native_center.y");
    mismatch(same_bits(a.native_velocity.x, b.native_velocity.x),
             "native_velocity.x");
    mismatch(same_bits(a.native_velocity.y, b.native_velocity.y),
             "native_velocity.y");
    mismatch(same_bits(a.native_angle, b.native_angle), "native_angle");
    mismatch(same_bits(a.native_angular_velocity,
                       b.native_angular_velocity),
             "native_angular_velocity");
    mismatch(a.native_center_valid == b.native_center_valid,
             "native_center_valid");
    mismatch(a.lifecycle == b.lifecycle, "lifecycle");
    mismatch(a.actor_slot == b.actor_slot, "actor_slot");
    mismatch(a.pending_delete == b.pending_delete, "pending_delete");
    mismatch(a.sleeping == b.sleeping, "sleeping");
    mismatch(same_bits(a.sleep_time, b.sleep_time), "sleep_time");
  }
}

void require_same_ordering(const irisu::PhysicsOrdering& left,
                           const irisu::PhysicsOrdering& right) {
  require(left.body_order == right.body_order &&
              left.destroy_order == right.destroy_order &&
              left.proxy_order == right.proxy_order &&
              left.proxy_ids == right.proxy_ids &&
              left.free_proxy_order == right.free_proxy_order &&
              left.static_sleep_flags == right.static_sleep_flags &&
              left.broadphase_time_stamp == right.broadphase_time_stamp &&
              left.proxy_time_stamps == right.proxy_time_stamps &&
              left.proxy_overlap_counts == right.proxy_overlap_counts &&
              left.broadphase_bounds == right.broadphase_bounds,
          "native physics ordering diverged after restore");
}

void verify_branch(irisu::PhysicsWorld& source, std::vector<irisu::Body> bodies,
                   int future_ticks) {
  const auto contacts = source.contact_impulses(bodies);
  const auto ordering = source.ordering();
  irisu::PhysicsWorld restored(config());
  restored.rebuild(bodies, contacts, ordering);
  require(restored.contact_impulses(bodies) == contacts,
          "contact manifold state changed during restore");
  require_same_ordering(restored.ordering(), ordering);

  auto restored_bodies = bodies;
  for (int tick = 0; tick < future_ticks; ++tick) {
    source.step(bodies);
    restored.step(restored_bodies);
    require_same_bodies(bodies, restored_bodies);
    require(source.contact_impulses(bodies) ==
                restored.contact_impulses(restored_bodies),
            "contact manifold future diverged after restore");
    require_same_ordering(source.ordering(), restored.ordering());
  }
}

void dense_mid_contact_branch() {
  auto mechanics = config();
  mechanics.gravity_y = 100.0;
  mechanics.field_height = 400.0;
  irisu::PhysicsWorld world(mechanics);
  std::vector<irisu::Body> bodies;
  for (std::uint32_t index = 0; index < 6; ++index) {
    auto value = body(index + 1, {0.0, 390.0 - 20.0 * index});
    value.shape = irisu::Shape::Box;
    world.initialize_mass(value);
    bodies.push_back(value);
  }
  for (int tick = 0; tick < 30; ++tick) world.step(bodies);

  const auto contacts = world.contact_impulses(bodies);
  const auto ordering = world.ordering();
  irisu::PhysicsWorld restored(mechanics);
  try {
    restored.rebuild(bodies, contacts, ordering);
  } catch (const std::exception& error) {
    throw std::runtime_error(std::string("dense rebuild: ") + error.what());
  }
  require(restored.contact_impulses(bodies) == contacts,
          "dense manifold state changed during restore");
  auto other = bodies;
  for (int tick = 0; tick < 20; ++tick) {
    world.step(bodies);
    restored.step(other);
    require_same_bodies(bodies, other);
    require(world.contact_impulses(bodies) == restored.contact_impulses(other),
            "dense manifold future diverged after restore");
  }
}

void swept_zero_manifold_branch() {
  irisu::PhysicsWorld world(config());
  std::vector bodies{
      body(1, {-100.0, 300.0}, {1000.0, 0.0}),
      body(2, {100.0, 300.0}, {-1000.0, 0.0}),
  };
  for (auto& value : bodies) world.initialize_mass(value);
  world.step(bodies);
  const auto contacts = world.contact_impulses(bodies);
  require(!contacts.empty() && contacts.front().manifold_count == 0,
          "fixture must create a swept zero-manifold contact");
  verify_branch(world, bodies, 4);
}

void deferred_contact_replacement_branch() {
  irisu::PhysicsWorld world(config());
  std::vector bodies{
      body(1, {0.0, 300.0}),
      body(2, {10.0, 300.0}),
  };
  for (auto& value : bodies) world.initialize_mass(value);
  world.synchronize(bodies);

  bodies[1].position.x = 100.0;
  world.synchronize(bodies);
  bodies[1].position.x = 10.0;
  world.synchronize(bodies);

  const auto contacts = world.contact_impulses(bodies);
  require(contacts.size() == 2 && contacts[0].a == 1 && contacts[0].b == 2 &&
              contacts[1].a == 1 && contacts[1].b == 2 &&
              contacts[0].contact_order != contacts[1].contact_order &&
              contacts[0].destroy_pending != contacts[1].destroy_pending &&
              contacts[0].manifold_count == 0 &&
              contacts[1].manifold_count == 0,
          "fixture must retain a deferred contact beside its replacement");
  verify_branch(world, bodies, 4);
}

void swept_proxy_outside_creation_range_branch() {
  irisu::MechanicsConfig mechanics;
  mechanics.gravity_y = 0.0;
  mechanics.linear_damping = 0.0;
  mechanics.angular_damping = 0.0;
  irisu::PhysicsWorld source(mechanics);
  auto value = body(1, {50.0, 460.0}, {0.0, 200.0});
  value.shape = irisu::Shape::Box;
  value.size = 24.0;
  value.density = 8.0;
  source.initialize_mass(value);
  std::vector bodies{value};
  source.step(bodies);

  const auto contacts = source.contact_impulses(bodies);
  const auto ordering = source.ordering();
  require(ordering.proxy_ids.size() == 1 &&
              ordering.proxy_ids.front() !=
                  std::numeric_limits<std::uint16_t>::max() &&
              bodies.front().position.y > mechanics.world_max_y,
          "fixture must retain a swept proxy whose current shape is out of range");

  auto restored_bodies = bodies;
  irisu::PhysicsWorld restored(mechanics);
  restored.rebuild(restored_bodies, contacts, ordering);
  require(restored.contact_impulses(restored_bodies) == contacts,
          "out-of-range swept proxy contacts changed during restore");
  require_same_ordering(restored.ordering(), ordering);
  for (int tick = 0; tick < 4; ++tick) {
    source.step(bodies);
    restored.step(restored_bodies);
    require_same_bodies(bodies, restored_bodies);
    require(source.contact_impulses(bodies) ==
                restored.contact_impulses(restored_bodies),
            "out-of-range swept proxy future diverged after restore");
    require_same_ordering(source.ordering(), restored.ordering());
  }
}

void frozen_proxy_with_deferred_contact_branch() {
  irisu::PhysicsWorld source(config());
  std::vector bodies{
      body(1, {0.0, 300.0}),
      body(2, {10.0, 300.0}),
  };
  for (auto& value : bodies) source.initialize_mass(value);
  source.step(bodies);

  bodies[1].position.x = 3000.0;
  source.synchronize(bodies);
  const auto contacts = source.contact_impulses(bodies);
  const auto ordering = source.ordering();
  const auto frozen = std::find(ordering.proxy_order.begin(),
                                ordering.proxy_order.end(), 2);
  const auto frozen_index =
      static_cast<std::size_t>(frozen - ordering.proxy_order.begin());
  require(frozen != ordering.proxy_order.end() &&
              ordering.proxy_ids[frozen_index] ==
                  std::numeric_limits<std::uint16_t>::max() &&
              std::any_of(contacts.begin(), contacts.end(), [](const auto& contact) {
                return contact.a == 1 && contact.b == 2 &&
                       contact.destroy_pending;
              }),
          "fixture must retain a deferred contact after one proxy freezes");

  auto restored_bodies = bodies;
  irisu::PhysicsWorld restored(config());
  restored.rebuild(restored_bodies, contacts, ordering);
  require(restored.contact_impulses(restored_bodies) == contacts,
          "frozen-proxy deferred contact changed during restore");
  require_same_ordering(restored.ordering(), ordering);
  for (int tick = 0; tick < 4; ++tick) {
    source.step(bodies);
    restored.step(restored_bodies);
    require_same_bodies(bodies, restored_bodies);
    require(source.contact_impulses(bodies) ==
                restored.contact_impulses(restored_bodies),
            "frozen-proxy deferred contact future diverged after restore");
    require_same_ordering(source.ordering(), restored.ordering());
  }
}

void actor_velocity_can_diverge_from_native() {
  auto mechanics = config();
  irisu::PhysicsWorld source(mechanics);
  std::vector bodies{body(1, {0.0, 250.0}, {3.25, -1.5})};
  bodies[0].actor_slot = 5;
  source.initialize_mass(bodies[0]);
  source.step(bodies);

  const auto native_before = bodies[0].native_velocity;
  bodies[0].velocity = {};
  bodies[0].remaining_lifetime = 1;
  const auto contacts = source.contact_impulses(bodies);
  const auto ordering = source.ordering();
  auto restored_bodies = bodies;
  irisu::PhysicsWorld restored(mechanics);
  restored.rebuild(restored_bodies, contacts, ordering);
  const auto restored_native = restored.raw_velocity(1);
  require(restored_bodies[0].velocity.x == 0.0 &&
              restored_bodies[0].velocity.y == 0.0 &&
              same_bits(restored_native.x, native_before.x) &&
              same_bits(restored_native.y, native_before.y),
          "actor OOB velocity zero must not overwrite native velocity");

  source.step(bodies);
  restored.step(restored_bodies);
  require_same_bodies(bodies, restored_bodies);
  require(same_bits(bodies[0].velocity.x, native_before.x) &&
              same_bits(bodies[0].velocity.y, native_before.y),
          "next physics Step must advance with the preserved native velocity");
}

void noninvertible_actor_position_preserves_native_bits() {
  auto mechanics = config();
  irisu::PhysicsWorld source(mechanics);
  const float raw_x = std::bit_cast<float>(std::uint32_t{0x3dcc'cccf});
  const float actor_x = static_cast<float>(
      raw_x * static_cast<float>(mechanics.world_magnification));
  const float reconstructed = static_cast<float>(
      actor_x / static_cast<float>(mechanics.world_magnification));
  require(std::bit_cast<std::uint32_t>(raw_x) !=
              std::bit_cast<std::uint32_t>(reconstructed),
          "adversarial position must be non-invertible through actor pixels");

  auto value = body(1, {actor_x, 300.0});
  value.actor_slot = 5;
  value.native_position = {raw_x, 30.0};
  value.native_velocity = {0.125, 0.0};
  value.native_angle = std::bit_cast<float>(std::uint32_t{0x3eaa'aaab});
  value.native_state_valid = true;
  source.initialize_mass(value);
  std::vector bodies{value};
  source.rebuild(bodies);
  verify_branch(source, bodies, 4);
}

void asymmetric_triangle_center_branch() {
  auto mechanics = config();
  irisu::PhysicsWorld source(mechanics);
  auto value = body(1, {112.50629425048828, 300.11922454833984},
                    {-3.113008737564087, -5.247770309448242});
  value.shape = irisu::Shape::Triangle;
  value.size = 48.0;
  value.angle = 2.4333908557891846;
  value.angular_velocity = 1.2486883401870728;
  value.actor_slot = 5;
  source.initialize_mass(value);
  std::vector bodies{value};
  source.synchronize(bodies);
  source.step(bodies);

  require(bodies.front().native_center_valid &&
              (!same_bits(bodies.front().native_center.x,
                          bodies.front().native_position.x) ||
               !same_bits(bodies.front().native_center.y,
                          bodies.front().native_position.y)),
          "triangle fixture must have a nonzero local center of mass");

  auto inconsistent = bodies;
  inconsistent.front().native_center.x = static_cast<double>(std::nextafter(
      static_cast<float>(inconsistent.front().native_center.x),
      std::numeric_limits<float>::infinity()));
  bool rejected = false;
  try {
    irisu::PhysicsWorld invalid(mechanics);
    invalid.rebuild(inconsistent, source.contact_impulses(bodies),
                    source.ordering());
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  require(rejected, "inconsistent triangle center must be rejected");
  verify_branch(source, bodies, 32);
}

std::uint32_t first_free_actor_slot(const std::vector<irisu::Body>& bodies) {
  for (std::uint32_t slot = 5;
       slot < irisu::MechanicsConfig::actor_pool_capacity; ++slot) {
    const bool occupied = std::any_of(
        bodies.begin(), bodies.end(), [&](const irisu::Body& candidate) {
          return candidate.lifecycle != irisu::Lifecycle::Deleted &&
                 candidate.actor_slot == slot;
        });
    if (!occupied) return slot;
  }
  throw std::runtime_error("test actor pool exhausted");
}

void compact_native_tombstones(std::vector<irisu::Body>& bodies) {
  std::erase_if(bodies, [](const irisu::Body& candidate) {
    return candidate.lifecycle == irisu::Lifecycle::Deleted &&
           !candidate.pending_delete;
  });
}

void deferred_delete_and_pair_pool_churn() {
  auto mechanics = config();
  mechanics.gravity_y = 40.0;
  irisu::PhysicsWorld source(mechanics);
  std::vector<irisu::Body> bodies;
  const std::uint32_t slots[] = {12, 5, 10, 7, 14, 6, 11, 9};
  for (std::uint32_t index = 0; index < 8; ++index) {
    auto value = body(index + 1,
                      {-24.0 + 12.0 * static_cast<double>(index % 4),
                       250.0 + 12.0 * static_cast<double>(index / 4)},
                      {0.05 * static_cast<double>(index % 3), 0.0});
    value.actor_slot = slots[index];
    source.initialize_mass(value);
    bodies.push_back(value);
  }
  source.step(bodies);

  auto* first_deleted = &bodies[1];
  auto* second_deleted = &bodies[4];
  first_deleted->lifecycle = irisu::Lifecycle::Deleted;
  first_deleted->pending_delete = true;
  first_deleted->actor_slot = 5;
  second_deleted->lifecycle = irisu::Lifecycle::Deleted;
  second_deleted->pending_delete = true;
  second_deleted->actor_slot = 14;
  bodies[0].actor_slot = 8;
  bodies[0].position.x += 3.125;

  auto inserted = body(9, {-6.0, 256.0}, {-0.125, 0.0});
  inserted.actor_slot = 5;
  source.initialize_mass(inserted);
  bodies.push_back(inserted);
  source.synchronize(bodies);
  const auto checkpoint_contacts = source.contact_impulses(bodies);
  const auto checkpoint_ordering = source.ordering();
  require(checkpoint_ordering.destroy_order ==
              std::vector<irisu::BodyId>({5, 2}),
          "actor-slot ordered deletes must determine destroy-list order");

  auto restored_bodies = bodies;
  irisu::PhysicsWorld restored(mechanics);
  restored.rebuild(restored_bodies, checkpoint_contacts, checkpoint_ordering);
  require(restored.contact_impulses(restored_bodies) == checkpoint_contacts,
          "pending-delete contacts changed during restore");
  require_same_ordering(restored.ordering(), checkpoint_ordering);

  irisu::BodyId next_id = 10;
  for (int tick = 0; tick < 100; ++tick) {
    if (tick % 7 == 0) {
      const auto victim = std::find_if(
          bodies.begin(), bodies.end(), [](const irisu::Body& candidate) {
            return candidate.lifecycle != irisu::Lifecycle::Deleted;
          });
      const auto restored_victim = std::find_if(
          restored_bodies.begin(), restored_bodies.end(),
          [&](const irisu::Body& candidate) {
            return candidate.id == victim->id;
          });
      victim->lifecycle = irisu::Lifecycle::Deleted;
      victim->pending_delete = true;
      restored_victim->lifecycle = irisu::Lifecycle::Deleted;
      restored_victim->pending_delete = true;
    }
    if (tick % 5 == 0) {
      const irisu::BodyId id = next_id++;
      auto value = body(
          id,
          {-24.0 + 12.0 * static_cast<double>(id % 5),
           250.0 + 10.0 * static_cast<double>((id / 5) % 3)},
          {0.025 * static_cast<double>(id % 5), 0.0});
      value.actor_slot = first_free_actor_slot(bodies);
      source.initialize_mass(value);
      bodies.push_back(value);
      restored_bodies.push_back(value);
    }
    if (tick % 9 == 0) {
      const auto target = std::find_if(
          bodies.rbegin(), bodies.rend(), [](const irisu::Body& candidate) {
            return candidate.lifecycle != irisu::Lifecycle::Deleted;
          });
      const auto restored_target = std::find_if(
          restored_bodies.begin(), restored_bodies.end(),
          [&](const irisu::Body& candidate) {
            return candidate.id == target->id;
          });
      target->position.x = static_cast<float>(target->position.x + 1.125f);
      restored_target->position.x = target->position.x;
    }

    source.step(bodies);
    restored.step(restored_bodies);
    compact_native_tombstones(bodies);
    compact_native_tombstones(restored_bodies);
    require_same_bodies(bodies, restored_bodies);
    require(source.contact_impulses(bodies) ==
                restored.contact_impulses(restored_bodies),
            "contact cache diverged during proxy churn");
    require_same_ordering(source.ordering(), restored.ordering());

    if (tick % 13 == 12) {
      const auto contacts = restored.contact_impulses(restored_bodies);
      const auto ordering = restored.ordering();
      irisu::PhysicsWorld rebuilt(mechanics);
      rebuilt.rebuild(restored_bodies, contacts, ordering);
      restored = std::move(rebuilt);
    }
  }
}

}  // namespace

int main() {
  try {
    dense_mid_contact_branch();
    swept_zero_manifold_branch();
    deferred_contact_replacement_branch();
    swept_proxy_outside_creation_range_branch();
    frozen_proxy_with_deferred_contact_branch();
    actor_velocity_can_diverge_from_native();
    noninvertible_actor_position_preserves_native_bits();
    asymmetric_triangle_center_branch();
    deferred_delete_and_pair_pool_churn();
    std::cout << "native snapshot physics properties passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "snapshot physics property failure: " << error.what() << '\n';
    return 1;
  }
}
