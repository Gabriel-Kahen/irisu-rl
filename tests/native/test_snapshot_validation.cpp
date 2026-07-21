#include "irisu/simulator.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

std::size_t event_count(const irisu::StepResult& result,
                        irisu::EventKind kind) {
  return static_cast<std::size_t>(std::count_if(
      result.events.begin(), result.events.end(),
      [kind](const irisu::Event& event) { return event.kind == kind; }));
}

bool same_result(const irisu::StepResult& left,
                 const irisu::StepResult& right) {
  const auto& a = left.diagnostics;
  const auto& b = right.diagnostics;
  if (left.reward != right.reward || left.terminated != right.terminated ||
      left.truncated != right.truncated ||
      a.config_hash != b.config_hash ||
      a.finish_call_count != b.finish_call_count ||
      a.terminal_metadata_recorded != b.terminal_metadata_recorded ||
      a.recorded_final_score != b.recorded_final_score ||
      a.recorded_final_highest_chain != b.recorded_final_highest_chain ||
      a.recorded_final_level != b.recorded_final_level ||
      a.recorded_final_clears != b.recorded_final_clears ||
      a.latest_final_score != b.latest_final_score ||
      a.latest_final_highest_chain != b.latest_final_highest_chain ||
      a.latest_final_level != b.latest_final_level ||
      a.latest_final_clears != b.latest_final_clears ||
      left.events.size() != right.events.size()) {
    return false;
  }
  for (std::size_t index = 0; index < left.events.size(); ++index) {
    const auto& x = left.events[index];
    const auto& y = right.events[index];
    if (x.tick != y.tick || x.kind != y.kind || x.a != y.a || x.b != y.b ||
        x.value != y.value || x.detail != y.detail ||
        x.sequence != y.sequence) {
      return false;
    }
  }
  return true;
}

template <typename Mutation>
void reject_object_atomically(irisu::Simulator& simulator,
                              Mutation mutation) {
  const auto stable = simulator.state_hash();
  auto snapshot = simulator.clone_state();
  mutation(snapshot);
  bool rejected = false;
  try {
    simulator.restore_state(snapshot);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  assert(simulator.state_hash() == stable);
}

void reject_wire_atomically(irisu::Simulator& simulator,
                            const std::vector<std::byte>& bytes) {
  const auto stable = simulator.state_hash();
  bool rejected = false;
  try {
    simulator.restore_snapshot(bytes);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  assert(simulator.state_hash() == stable);
}

template <typename Operation>
void reject_overflow_atomically(irisu::Simulator& simulator,
                                Operation operation) {
  const auto stable = simulator.state_hash();
  bool rejected = false;
  try {
    operation();
  } catch (const std::overflow_error&) {
    rejected = true;
  }
  assert(rejected);
  assert(simulator.state_hash() == stable);
}

std::uint64_t next_random(std::uint64_t& state) {
  state ^= state << 13U;
  state ^= state >> 7U;
  state ^= state << 17U;
  return state;
}

irisu::Action random_action(std::uint64_t& state) {
  const auto roll = next_random(state);
  const auto x = static_cast<double>(next_random(state) % 1024U);
  const auto y = static_cast<double>(next_random(state) % 512U);
  switch (roll % 7U) {
    case 0: return {irisu::ActionKind::Wait, 0, 0,
                    static_cast<std::uint32_t>(next_random(state) % 4U + 1U)};
    case 1: return {irisu::ActionKind::WeakShot, x, y, 1};
    case 2: return {irisu::ActionKind::StrongShot, x, y, 1};
    case 3: return {irisu::ActionKind::BothShots, x, y, 1};
    case 4: return {irisu::ActionKind::WeakShot, -1.0, y, 1};
    case 5: return {irisu::ActionKind::StrongShot, x, 512.0, 1};
    default: return {irisu::ActionKind::Wait, 0, 0, 1};
  }
}

irisu::MechanicsConfig empty_start_config() {
  irisu::MechanicsConfig config;
  config.initial_rotten_count = 0;
  config.initial_falling_count = 0;
  return config;
}

void randomized_snapshot_future_equivalence() {
  constexpr std::uint64_t seed_count = 32;
  constexpr int warmup_actions = 32;
  constexpr int future_actions = 64;
  for (std::uint64_t seed = 0; seed < seed_count; ++seed) {
    irisu::Simulator source;
    irisu::Simulator same_seed;
    const auto environment_seed = static_cast<std::uint32_t>(
        seed * 0x9e3779b97f4a7c15ULL + 17U);
    source.reset(environment_seed);
    same_seed.reset(environment_seed);
    std::uint64_t actions = seed * 0xd1342543de82ef95ULL + 1U;
    for (int index = 0; index < warmup_actions; ++index) {
      const auto action = random_action(actions);
      const auto first = source.step(action);
      const auto second = same_seed.step(action);
      assert(same_result(first, second));
      assert(source.serialize_snapshot() == same_seed.serialize_snapshot());
    }

    const auto object = source.clone_state();
    const auto wire = source.serialize_snapshot();
    irisu::Simulator object_branch;
    irisu::Simulator wire_branch;
    object_branch.restore_state(object);
    wire_branch.restore_snapshot(wire);
    assert(source.serialize_snapshot() == object_branch.serialize_snapshot());
    assert(source.serialize_snapshot() == wire_branch.serialize_snapshot());

    for (int index = 0; index < future_actions; ++index) {
      const auto action = random_action(actions);
      const auto expected = source.step(action);
      const auto from_object = object_branch.step(action);
      const auto from_wire = wire_branch.step(action);
      assert(same_result(expected, from_object));
      assert(same_result(expected, from_wire));
      const auto expected_wire = source.serialize_snapshot();
      assert(expected_wire == object_branch.serialize_snapshot());
      assert(expected_wire == wire_branch.serialize_snapshot());
    }
  }
}

void reject_pending_without_group() {
  irisu::Simulator simulator(empty_start_config());
  simulator.reset(17);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 100.0});
  reject_object_atomically(simulator, [](irisu::Snapshot& snapshot) {
    snapshot.bodies.front().successful_clear_pending = true;
  });

  auto bytes = simulator.serialize_snapshot();
  // Schema 7 keeps the fixed header at 2669 bytes; pool colors are appended.
  constexpr std::size_t body_start = 2669;
  constexpr std::size_t successful_clear_pending = 200;
  assert(bytes.size() > body_start + successful_clear_pending);
  bytes[body_start + successful_clear_pending] = std::byte{1};
  reject_wire_atomically(simulator, bytes);
}

void object_invariants_are_rejected_atomically() {
  irisu::Simulator simulator(empty_start_config());
  simulator.reset(18);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 100.0});
  const auto max = std::numeric_limits<std::uint64_t>::max();

  reject_object_atomically(simulator, [](auto& s) { ++s.schema_version; });
  reject_object_atomically(simulator, [](auto& s) { ++s.config_hash; });
  reject_object_atomically(simulator, [](auto& s) { s.rng_state.fill(0); });
  reject_object_atomically(simulator, [](auto& s) {
    s.rng_index = irisu::DxRandom::state_words + 1U;
  });
  reject_object_atomically(simulator, [&](auto& s) {
    s.tick = simulator.config().max_episode_ticks;
  });
  reject_object_atomically(simulator, [](auto& s) { s.truncated = true; });
  reject_object_atomically(simulator, [&](auto& s) { s.scene_frame = max; });
  reject_object_atomically(simulator, [](auto& s) { s.spawn_count = 1; });
  reject_object_atomically(simulator, [](auto& s) {
    s.tick = 10;
    s.spawn_count = s.next_body_id;
  });
  reject_object_atomically(simulator, [&](auto& s) {
    s.qualifying_clear_count = max;
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.next_special_clear_count = 1'000;
  });
  reject_object_atomically(simulator, [&](auto& s) {
    s.level_shape_cutoff = simulator.config().shape_random_max + 1U;
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.finish_call_count = irisu::MechanicsConfig::actor_pool_capacity;
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.resize(irisu::MechanicsConfig::physics_proxy_capacity + 1U);
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.groups.resize(1'000'001U);
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.next_body_id = 0x80000001U;
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.actor_pool_colors[5] = -3;
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.actor_pool_colors[s.bodies.front().actor_slot] = 1;
  });

  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().kind = static_cast<irisu::BodyKind>(255);
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().shape = static_cast<irisu::Shape>(255);
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().color = -1;
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().special = true;
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().density += 1.0;
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().restitution = 1.5;
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().size = std::numeric_limits<double>::max();
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().position.x = std::numeric_limits<double>::max();
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().velocity.x = std::numeric_limits<double>::max();
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().native_position.y = std::numeric_limits<double>::max();
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().native_center.x = std::numeric_limits<double>::max();
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().native_center_valid = false;
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().native_center.x = static_cast<double>(std::nextafter(
        static_cast<float>(s.bodies.front().native_center.x),
        std::numeric_limits<float>::infinity()));
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().native_velocity.y =
        std::numeric_limits<double>::max();
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().remaining_lifetime = -2;
  });
  reject_object_atomically(simulator, [&](auto& s) {
    s.bodies.front().age_ticks = max;
  });
  reject_object_atomically(simulator, [](auto& s) {
    s.bodies.front().projectile_hits = 3;
  });

  simulator.spawn_bonus({400.0, 100.0});
  reject_object_atomically(simulator, [](auto& s) {
    auto& bonus = *std::find_if(s.bodies.begin(), s.bodies.end(),
                                [](const auto& body) {
                                  return body.kind == irisu::BodyKind::Bonus;
                                });
    bonus.freshness_state = 3;
    bonus.lifecycle = irisu::Lifecycle::Rotten;
  });

  auto terminal = simulator.clone_state();
  terminal.terminated = true;
  terminal.terminal_metadata_recorded = true;
  terminal.finish_call_count = 1;
  terminal.recorded_final_score = terminal.score;
  terminal.recorded_final_highest_chain = terminal.highest_chain;
  terminal.recorded_final_level = terminal.level;
  terminal.recorded_final_clears = terminal.qualifying_clear_count;
  terminal.latest_final_score = terminal.score;
  terminal.latest_final_highest_chain = terminal.highest_chain;
  terminal.latest_final_level = terminal.level;
  terminal.latest_final_clears = terminal.qualifying_clear_count;
  simulator.restore_state(terminal);
  reject_object_atomically(simulator, [](auto& s) {
    s.latest_final_score = s.score + 1;
  });
}

void counter_boundaries_do_not_wrap() {
  const auto maximum = std::numeric_limits<std::uint64_t>::max();
  irisu::Simulator simulator(empty_start_config());
  simulator.reset(19);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 100.0});

  auto snapshot = simulator.clone_state();
  snapshot.scene_frame = maximum - 1U;
  simulator.restore_state(snapshot);
  reject_overflow_atomically(simulator, [&] {
    (void)simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  });
  reject_overflow_atomically(simulator, [&] {
    (void)simulator.step({irisu::ActionKind::Wait, 0, 0, 2});
  });

  simulator.reset(20);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 100.0});
  snapshot = simulator.clone_state();
  snapshot.bodies.front().age_ticks = maximum - 1U;
  simulator.restore_state(snapshot);
  reject_overflow_atomically(simulator, [&] {
    (void)simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  });
  reject_overflow_atomically(simulator, [&] {
    (void)simulator.step({irisu::ActionKind::Wait, 0, 0, 2});
  });

  simulator.reset(20);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 100.0});
  snapshot = simulator.clone_state();
  snapshot.bodies.front().physics_update_count = maximum - 1U;
  simulator.restore_state(snapshot);
  reject_overflow_atomically(simulator, [&] {
    (void)simulator.step({irisu::ActionKind::Wait, 0, 0, 2});
  });

  simulator.reset(20);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 100.0});
  snapshot = simulator.clone_state();
  snapshot.bodies.front().rot_timer = maximum - 1U;
  simulator.restore_state(snapshot);
  reject_overflow_atomically(simulator, [&] {
    (void)simulator.step({irisu::ActionKind::Wait, 0, 0, 2});
  });

  simulator.reset(21);
  snapshot = simulator.clone_state();
  snapshot.next_event_sequence = maximum;
  simulator.restore_state(snapshot);
  reject_overflow_atomically(simulator, [&] {
    (void)simulator.step(
        {static_cast<irisu::ActionKind>(255), 0, 0, 1});
  });

  simulator.reset(21);
  snapshot = simulator.clone_state();
  snapshot.next_event_sequence = maximum - (std::uint64_t{1} << 48U) + 1U;
  simulator.restore_state(snapshot);
  reject_overflow_atomically(simulator, [&] {
    (void)simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  });

  simulator.reset(22);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 100.0});
  snapshot = simulator.clone_state();
  constexpr irisu::BodyId exhausted_body_id = 0x80000000U;
  snapshot.next_body_id = exhausted_body_id;
  simulator.restore_state(snapshot);
  const auto id_exhausted = simulator.state_hash();
  assert(simulator.spawn_piece(irisu::Shape::Circle, 1, 30.0,
                               {200.0, 100.0}) == 0);
  assert(simulator.spawn_bonus({200.0, 100.0}) == 0);
  assert(simulator.state_hash() == id_exhausted);
  const auto no_wrapped_shot = simulator.step(
      {irisu::ActionKind::BothShots, 200.0, 100.0, 1});
  assert(event_count(no_wrapped_shot, irisu::EventKind::ShotFired) == 0);
  assert(simulator.clone_state().next_body_id == exhausted_body_id);

  auto long_episode = empty_start_config();
  long_episode.max_episode_ticks = maximum;
  irisu::Simulator final_tick(long_episode);
  final_tick.reset(22);
  snapshot = final_tick.clone_state();
  snapshot.tick = maximum - 1U;
  snapshot.scene_frame = maximum - 1U;
  final_tick.restore_state(snapshot);
  const auto completed = final_tick.step(
      {irisu::ActionKind::Wait, 0, 0, 2});
  assert(completed.truncated);
  assert(final_tick.observation().tick == maximum);
  const auto terminal_counter_hash = final_tick.state_hash();
  (void)final_tick.step({irisu::ActionKind::Wait, 0, 0, 1});
  assert(final_tick.state_hash() == terminal_counter_hash);
}

void long_numeric_horizons_are_preflighted() {
  auto config = empty_start_config();
  config.gravity_y = 0.0;
  config.spawn_interval_ticks = 100'000;
  config.max_episode_ticks = 1'000'000;
  irisu::Simulator simulator(config);
  simulator.reset(31);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 100.0});
  auto snapshot = simulator.clone_state();
  auto& body = snapshot.bodies.front();
  body.velocity.x = 1.0e34;
  body.scripted_velocity.x = 1.0e34;
  body.remaining_lifetime = std::numeric_limits<std::int64_t>::max();
  simulator.restore_state(snapshot);
  reject_overflow_atomically(simulator, [&] {
    (void)simulator.step(
        {irisu::ActionKind::Wait, 0, 0, 100'000});
  });
}

void hostile_manifold_values_are_rejected() {
  auto config = empty_start_config();
  config.gravity_y = 0.0;
  config.passive_gauge_decay_per_tick = 0;
  config.spawn_interval_ticks = 100'000;
  irisu::Simulator simulator(config);
  simulator.reset(32);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 200.0});
  simulator.spawn_piece(irisu::Shape::Box, 1, 30.0, {300.0, 200.0});
  auto snapshot = simulator.clone_state();
  snapshot.scene_frame = 1;
  for (auto& body : snapshot.bodies) {
    body.physics_owned = true;
    body.freshness_state = 2;
    body.lifecycle = irisu::Lifecycle::DynamicFresh;
    body.top_contact_enabled = true;
  }
  simulator.restore_state(snapshot);
  (void)simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  snapshot = simulator.clone_state();
  assert(std::any_of(snapshot.contact_impulses.begin(),
                     snapshot.contact_impulses.end(), [](const auto& impulse) {
                       return impulse.manifold_count != 0;
                     }));

  const auto maximum_float_bits =
      std::bit_cast<std::uint32_t>(std::numeric_limits<float>::max());
  reject_object_atomically(simulator, [&](auto& s) {
    for (auto& impulse : s.contact_impulses) {
      if (impulse.manifold_count != 0) {
        impulse.normal_x_bits = maximum_float_bits;
      }
    }
  });
  reject_object_atomically(simulator, [&](auto& s) {
    auto& impulse = *std::find_if(
        s.contact_impulses.begin(), s.contact_impulses.end(),
        [](const auto& candidate) { return candidate.manifold_count != 0; });
    impulse.point_x_bits = maximum_float_bits;
  });
}

void level_cap_terminal_snapshot_is_restorable() {
  auto config = empty_start_config();
  config.gravity_y = 0.0;
  config.passive_gauge_decay_per_tick = 0;
  config.spawn_interval_ticks = 100'000;
  irisu::Simulator simulator(config);
  simulator.reset(33);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 200.0});
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 200.0});
  auto snapshot = simulator.clone_state();
  snapshot.scene_frame = 1;
  snapshot.level = config.maximum_level;
  snapshot.qualifying_clear_count =
      static_cast<std::uint64_t>(config.maximum_level) *
          config.qualifying_clears_per_level -
      1U;
  for (auto& body : snapshot.bodies) {
    body.physics_owned = true;
    body.freshness_state = 2;
    body.lifecycle = irisu::Lifecycle::DynamicFresh;
    body.top_contact_enabled = true;
    body.rot_timer = 1;
  }
  simulator.restore_state(snapshot);
  const auto completed =
      simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  assert(completed.terminated);
  assert(simulator.clone_state().qualifying_clear_count ==
         static_cast<std::uint64_t>(config.maximum_level) *
             config.qualifying_clears_per_level);
  const auto wire = simulator.serialize_snapshot();
  irisu::Simulator restored(config);
  restored.restore_snapshot(wire);
  assert(restored.serialize_snapshot() == wire);
}

void exhausted_chain_and_contact_counters_saturate() {
  auto config = empty_start_config();
  config.gravity_y = 0.0;
  config.passive_gauge_decay_per_tick = 0;
  config.spawn_interval_ticks = 100'000;
  irisu::Simulator simulator(config);
  simulator.reset(26);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 200.0});
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 200.0});
  auto snapshot = simulator.clone_state();
  snapshot.scene_frame = 1;
  snapshot.next_chain_id = std::numeric_limits<irisu::ChainId>::max();
  for (auto& body : snapshot.bodies) {
    body.physics_owned = true;
    body.freshness_state = 2;
    body.lifecycle = irisu::Lifecycle::DynamicFresh;
    body.top_contact_enabled = true;
    body.non_wall_contacts = std::numeric_limits<std::uint32_t>::max();
  }
  simulator.restore_state(snapshot);
  (void)simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  snapshot = simulator.clone_state();
  assert(snapshot.next_chain_id ==
         std::numeric_limits<irisu::ChainId>::max());
  assert(snapshot.groups.empty());
  for (const auto& body : snapshot.bodies) {
    assert(body.non_wall_contacts ==
           std::numeric_limits<std::uint32_t>::max());
  }
}

void score_and_gauge_boundaries_saturate() {
  auto config = empty_start_config();
  config.gravity_y = 0.0;
  config.passive_gauge_decay_per_tick = 0;
  config.spawn_interval_ticks = 100'000;
  config.gauge_max = std::numeric_limits<std::int64_t>::max();
  config.gauge_initial = 1;
  config.gauge_clear_unit = std::numeric_limits<std::int64_t>::max();
  irisu::Simulator simulator(config);
  simulator.reset(28);
  const auto target = simulator.spawn_piece(
      irisu::Shape::Box, 0, 30.0, {300.0, 200.0});
  const auto source = simulator.spawn_piece(
      irisu::Shape::Box, 1, 30.0, {300.0, 200.0});
  auto snapshot = simulator.clone_state();
  snapshot.scene_frame = 1;
  snapshot.next_chain_id = 2;
  snapshot.groups.push_back(
      {1, std::numeric_limits<std::uint32_t>::max(),
       std::numeric_limits<std::uint32_t>::max(),
       std::numeric_limits<std::uint32_t>::max() - 1U});
  for (auto& body : snapshot.bodies) {
    body.physics_owned = true;
    body.freshness_state = 2;
    body.lifecycle = irisu::Lifecycle::DynamicFresh;
    body.top_contact_enabled = true;
  }
  auto& grouped = *std::find_if(
      snapshot.bodies.begin(), snapshot.bodies.end(),
      [&](const irisu::Body& body) { return body.id == target; });
  grouped.grouped = true;
  grouped.chain_id = 1;
  grouped.lifecycle = irisu::Lifecycle::Confirmed;
  auto& rot_source = *std::find_if(
      snapshot.bodies.begin(), snapshot.bodies.end(),
      [&](const irisu::Body& body) { return body.id == source; });
  rot_source.rot_timer = 1;
  simulator.restore_state(snapshot);
  (void)simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  assert(simulator.observation().score ==
         std::numeric_limits<std::int64_t>::max());
  assert(simulator.observation().gauge ==
         std::numeric_limits<std::int64_t>::max());

  config.gauge_max = 10'000;
  config.gauge_clear_unit = 700;
  config.rotten_penalty = std::numeric_limits<std::int64_t>::max();
  irisu::Simulator rotting(config);
  rotting.reset(29);
  rotting.spawn_piece(irisu::Shape::Box, 0, 30.0, {200.0, 100.0});
  rotting.spawn_piece(irisu::Shape::Box, 1, 30.0, {400.0, 100.0});
  snapshot = rotting.clone_state();
  snapshot.scene_frame = 1;
  for (auto& body : snapshot.bodies) {
    body.freshness_state = 2;
    body.rot_timer = 120;
    body.age_ticks = 121;
  }
  rotting.restore_state(snapshot);
  (void)rotting.step({irisu::ActionKind::Wait, 0, 0, 1});
  assert(rotting.observation().gauge ==
         std::numeric_limits<std::int64_t>::min());
}

void pending_tombstones_remain_restorable() {
  auto config = empty_start_config();
  config.gravity_y = 0.0;
  config.passive_gauge_decay_per_tick = 0;
  config.spawn_interval_ticks = 100'000;
  irisu::Simulator simulator(config);
  simulator.reset(27);
  constexpr std::size_t usable_slots =
      irisu::MechanicsConfig::actor_pool_capacity - 4U;
  for (std::size_t index = 0; index < usable_slots; ++index) {
    assert(simulator.spawn_piece(
               irisu::Shape::Box, static_cast<std::int32_t>(index % 3U), 2.0,
               {100.0 + 5.0 * static_cast<double>(index % 20U),
                50.0 + 5.0 * static_cast<double>(index / 20U)}) != 0);
  }
  auto snapshot = simulator.clone_state();
  snapshot.scene_frame = 1;
  for (auto& body : snapshot.bodies) {
    body.freshness_state = 2;
    body.lifecycle = irisu::Lifecycle::ScriptedFalling;
    body.delete_marked = true;
  }
  simulator.restore_state(snapshot);
  (void)simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  assert(simulator.clone_state().bodies.size() == usable_slots);
  for (std::size_t index = 0; index < usable_slots; ++index) {
    assert(simulator.spawn_piece(
               irisu::Shape::Box, static_cast<std::int32_t>(index % 3U), 2.0,
               {300.0 + 5.0 * static_cast<double>(index % 20U),
                50.0 + 5.0 * static_cast<double>(index / 20U)}) != 0);
  }
  const auto wire = simulator.serialize_snapshot();
  assert(simulator.clone_state().bodies.size() == 2U * usable_slots);
  irisu::Simulator restored(config);
  restored.restore_snapshot(wire);
  assert(restored.serialize_snapshot() == wire);
  const auto expected = simulator.step(
      {irisu::ActionKind::Wait, 0, 0, 1});
  const auto actual = restored.step(
      {irisu::ActionKind::Wait, 0, 0, 1});
  assert(same_result(expected, actual));
  assert(simulator.serialize_snapshot() == restored.serialize_snapshot());
}

void public_spawn_arguments_are_safe_and_atomic() {
  irisu::Simulator simulator;
  simulator.reset(23);
  const auto reject_piece = [&](irisu::Shape shape, std::int32_t color,
                                double size, irisu::Vec2 position) {
    const auto stable = simulator.state_hash();
    bool rejected = false;
    try {
      (void)simulator.spawn_piece(shape, color, size, position);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    assert(rejected && simulator.state_hash() == stable);
  };
  reject_piece(static_cast<irisu::Shape>(255), 0, 30.0, {100.0, 100.0});
  reject_piece(irisu::Shape::Box, -1, 30.0, {100.0, 100.0});
  reject_piece(irisu::Shape::Box,
               static_cast<std::int32_t>(simulator.config().maximum_colors),
               30.0, {100.0, 100.0});
  reject_piece(irisu::Shape::Box, 0, 0.0, {100.0, 100.0});
  reject_piece(irisu::Shape::Triangle, 0, 1.0e-12, {100.0, 100.0});
  reject_piece(irisu::Shape::Box, 0,
               std::numeric_limits<double>::infinity(), {100.0, 100.0});
  reject_piece(irisu::Shape::Box, 0, std::numeric_limits<double>::max(),
               {100.0, 100.0});
  reject_piece(irisu::Shape::Box, 0, 30.0,
               {std::numeric_limits<double>::quiet_NaN(), 100.0});
  reject_piece(irisu::Shape::Box, 0, 30.0,
               {std::numeric_limits<double>::max(), 100.0});

  const auto stable = simulator.state_hash();
  bool bonus_rejected = false;
  try {
    (void)simulator.spawn_bonus(
        {100.0, std::numeric_limits<double>::infinity()});
  } catch (const std::invalid_argument&) {
    bonus_rejected = true;
  }
  assert(bonus_rejected && simulator.state_hash() == stable);
  assert(simulator.spawn_piece(irisu::Shape::Circle, 0, 2.0,
                               {100.0, 100.0}) != 0);
  assert(simulator.spawn_piece(irisu::Shape::Triangle, 1, 2.0,
                               {150.0, 100.0}) != 0);
}

void encoded_shot_range_is_total_and_safe() {
  irisu::Simulator simulator;
  simulator.reset(24);
  for (const auto point : {irisu::Vec2{0.0, 0.0},
                           irisu::Vec2{1023.0, 0.0},
                           irisu::Vec2{0.0, 511.0},
                           irisu::Vec2{1023.0, 511.0}}) {
    const auto fired = simulator.step(
        {irisu::ActionKind::WeakShot, point.x, point.y, 1});
    assert(event_count(fired, irisu::EventKind::InvalidAction) == 0);
    assert(event_count(fired, irisu::EventKind::ShotFired) == 1);
    (void)simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  }

  const auto below_zero =
      std::nextafter(0.0, -std::numeric_limits<double>::infinity());
  const auto above_x =
      std::nextafter(1023.0, std::numeric_limits<double>::infinity());
  const auto above_y =
      std::nextafter(511.0, std::numeric_limits<double>::infinity());
  for (const auto point : {
           irisu::Vec2{below_zero, 0.0}, irisu::Vec2{0.0, below_zero},
           irisu::Vec2{above_x, 0.0}, irisu::Vec2{0.0, above_y},
           irisu::Vec2{std::numeric_limits<double>::quiet_NaN(), 0.0},
           irisu::Vec2{0.0, std::numeric_limits<double>::infinity()}}) {
    const auto tick_before = simulator.observation().tick;
    const auto rejected = simulator.step(
        {irisu::ActionKind::StrongShot, point.x, point.y, 1});
    assert(event_count(rejected, irisu::EventKind::InvalidAction) == 1);
    assert(event_count(rejected, irisu::EventKind::ShotFired) == 0);
    assert(simulator.observation().tick == tick_before + 1U);
  }
  irisu::Simulator restored;
  restored.restore_snapshot(simulator.serialize_snapshot());
  assert(restored.serialize_snapshot() == simulator.serialize_snapshot());
}

void wire_parser_is_bounded_and_mutation_safe() {
  irisu::MechanicsConfig config;
  config.field_x = 94.0;
  config.field_y = 20.0;
  config.field_width = 420.0;
  config.field_height = 370.0;
  config.field_blank = 30.0;
  config.field_top_width = 450.0;
  config.field_bottom_height = 400.0;
  config.side_wall_top = 20.0;
  config.side_wall_bottom = 390.0;
  config.piece_sizes = {
      30.0, 48.0, 64.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0};
  config.piece_size_weights = {10, 40, 20, 0, 0, 0, 0, 0, 0, 0};
  irisu::Simulator simulator(config);
  simulator.reset(25);
  for (int tick = 0; tick < 40; ++tick) {
    const auto kind = tick % 9 == 0 ? irisu::ActionKind::BothShots
                                   : irisu::ActionKind::Wait;
    (void)simulator.step({kind, 200.0 + tick, 300.0, 1});
  }
  const auto canonical = simulator.serialize_snapshot();
  assert(canonical.size() > 2669);

  for (std::size_t length = 0; length < canonical.size(); ++length) {
    std::vector<std::byte> prefix(canonical.begin(),
                                  canonical.begin() + length);
    reject_wire_atomically(simulator, prefix);
  }
  auto trailing = canonical;
  trailing.push_back(std::byte{0});
  reject_wire_atomically(simulator, trailing);

  const auto write_u32 = [](std::vector<std::byte>& bytes,
                            std::size_t offset, std::uint32_t value) {
    for (std::size_t index = 0; index < sizeof(value); ++index) {
      bytes[offset + index] =
          static_cast<std::byte>((value >> (8U * index)) & 0xffU);
    }
  };
  auto hostile = canonical;
  write_u32(hostile, 4, 0xffffffffU);
  write_u32(hostile, 2665, 0xffffffffU);
  reject_wire_atomically(simulator, hostile);
  hostile = canonical;
  hostile[8] ^= std::byte{1};
  write_u32(hostile, 2665, 0xffffffffU);
  reject_wire_atomically(simulator, hostile);
  hostile = canonical;
  write_u32(hostile, 2665,
            irisu::MechanicsConfig::physics_proxy_capacity + 1U);
  reject_wire_atomically(simulator, hostile);

  // Contact impulses precede the 200 serialized pool colors in schema 7. Find a
  // one-byte body-id substitution that still names a live native body but is
  // inconsistent with the saved broad phase/contact state. This keeps the
  // wire-safety test independent of a nominal gameplay trajectory.
  const auto snapshot = simulator.clone_state();
  constexpr std::size_t contact_impulse_bytes = 58;
  constexpr std::size_t actor_pool_color_bytes =
      irisu::MechanicsConfig::actor_pool_capacity * sizeof(std::int32_t);
  assert(!snapshot.contact_impulses.empty());
  const auto impulses_offset = canonical.size() - actor_pool_color_bytes -
      snapshot.contact_impulses.size() * contact_impulse_bytes;
  std::vector<std::byte> inconsistent_contact;
  for (std::size_t index = 0;
       index < snapshot.contact_impulses.size() && inconsistent_contact.empty();
       ++index) {
    const auto original = snapshot.contact_impulses[index].b;
    const auto replacement = original ^ 1U;
    if (replacement == snapshot.contact_impulses[index].a ||
        std::find(snapshot.physics_ordering.proxy_order.begin(),
                  snapshot.physics_ordering.proxy_order.end(), replacement) ==
            snapshot.physics_ordering.proxy_order.end()) {
      continue;
    }
    auto candidate = canonical;
    const auto body_b_offset =
        impulses_offset + index * contact_impulse_bytes + sizeof(irisu::BodyId);
    candidate[body_b_offset] ^= std::byte{1};
    irisu::Simulator probe(config);
    try {
      probe.restore_snapshot(candidate);
    } catch (const std::invalid_argument&) {
      inconsistent_contact = std::move(candidate);
    }
  }
  assert(!inconsistent_contact.empty());
  reject_wire_atomically(simulator, inconsistent_contact);

  std::uint64_t random = 0x7265706c61793135ULL;
  std::size_t accepted = 0;
  for (int mutation = 0; mutation < 768; ++mutation) {
    auto bytes = canonical;
    const auto offset = static_cast<std::size_t>(
        next_random(random) % static_cast<std::uint64_t>(bytes.size()));
    const auto bit = static_cast<unsigned>(next_random(random) % 8U);
    bytes[offset] ^= static_cast<std::byte>(1U << bit);
    const auto stable = simulator.state_hash();
    try {
      simulator.restore_snapshot(bytes);
      ++accepted;
      const auto accepted_wire = simulator.serialize_snapshot();
      irisu::Simulator branch(config);
      branch.restore_snapshot(accepted_wire);
      assert(branch.serialize_snapshot() == accepted_wire);
      const auto before_step = simulator.state_hash();
      try {
        const auto expected = simulator.step(
            {irisu::ActionKind::Wait, 0, 0, 1});
        const auto actual = branch.step(
            {irisu::ActionKind::Wait, 0, 0, 1});
        assert(same_result(expected, actual));
        assert(simulator.serialize_snapshot() == branch.serialize_snapshot());
      } catch (const std::overflow_error&) {
        assert(simulator.state_hash() == before_step);
      }
    } catch (const std::invalid_argument&) {
      assert(simulator.state_hash() == stable);
    }
    simulator.restore_snapshot(canonical);
  }
  assert(accepted != 0);
}

}  // namespace

int main() {
  randomized_snapshot_future_equivalence();
  reject_pending_without_group();
  object_invariants_are_rejected_atomically();
  counter_boundaries_do_not_wrap();
  long_numeric_horizons_are_preflighted();
  hostile_manifold_values_are_rejected();
  level_cap_terminal_snapshot_is_restorable();
  exhausted_chain_and_contact_counters_saturate();
  score_and_gauge_boundaries_saturate();
  pending_tombstones_remain_restorable();
  public_spawn_arguments_are_safe_and_atomic();
  encoded_shot_range_is_total_and_safe();
  wire_parser_is_bounded_and_mutation_safe();
}
