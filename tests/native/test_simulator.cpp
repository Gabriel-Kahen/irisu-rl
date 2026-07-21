#include "irisu/dx_random.hpp"
#include "irisu/normal_rules.hpp"
#include "irisu/simulator.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <algorithm>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

template <typename T>
concept HasNativePosition = requires(T body) { body.native_position; };

template <typename T>
concept HasPendingDelete = requires(T body) { body.pending_delete; };

template <typename T>
concept HasPhysicsOwned = requires(T body) { body.physics_owned; };

using ObservationBody =
    typename decltype(irisu::Observation{}.bodies)::value_type;
static_assert(std::same_as<ObservationBody, irisu::ObservedBody>);
static_assert(!HasNativePosition<ObservationBody>);
static_assert(!HasPendingDelete<ObservationBody>);
static_assert(!HasPhysicsOwned<ObservationBody>);

std::size_t event_count(const irisu::StepResult& result,
                        irisu::EventKind kind) {
  return static_cast<std::size_t>(std::count_if(
      result.events.begin(), result.events.end(),
      [kind](const irisu::Event& event) { return event.kind == kind; }));
}

irisu::Body& body(irisu::Snapshot& state, irisu::BodyId id) {
  const auto found = std::find_if(state.bodies.begin(), state.bodies.end(),
                                  [id](const irisu::Body& value) {
                                    return value.id == id;
                                  });
  assert(found != state.bodies.end());
  return *found;
}

const irisu::Body& body(const irisu::Snapshot& state, irisu::BodyId id) {
  const auto found = std::find_if(state.bodies.begin(), state.bodies.end(),
                                  [id](const irisu::Body& value) {
                                    return value.id == id;
                                  });
  assert(found != state.bodies.end());
  return *found;
}

irisu::MechanicsConfig controlled_config() {
  irisu::MechanicsConfig config;
  config.initial_rotten_count = 0;
  config.initial_falling_count = 0;
  config.gravity_y = 0.0;
  config.linear_damping = 0.0;
  config.angular_damping = 0.0;
  config.piece_life_ticks = 10'000;
  config.rot_delay_ticks = 120;
  config.projectile_life_ticks = 1'200;
  config.gauge_max = 10'000;
  config.gauge_initial = 1'000;
  config.passive_gauge_decay_per_tick = 0;
  config.spawn_interval_ticks = 100'000;
  config.qualifying_clears_per_level = 6;
  config.max_episode_ticks = 1'000'000;
  return config;
}

void suppress_cadence_spawn(irisu::Simulator& simulator) {
  auto state = simulator.clone_state();
  state.scene_frame = 1;
  simulator.restore_state(state);
}

void make_dynamic(irisu::Body& value, std::uint8_t freshness = 2) {
  value.physics_owned = true;
  value.freshness_state = freshness;
  value.top_contact_enabled = true;
  value.lifecycle = freshness == 3 ? irisu::Lifecycle::Rotten
                                   : value.grouped
                                         ? irisu::Lifecycle::Confirmed
                                         : irisu::Lifecycle::DynamicFresh;
}

void deterministic_trace_and_snapshot() {
  irisu::Simulator first;
  irisu::Simulator second;
  first.reset(42);
  second.reset(42);
  const std::vector<irisu::Action> actions{
      {irisu::ActionKind::Wait, 0, 0, 25},
      {irisu::ActionKind::WeakShot, 240, 350, 1},
      {irisu::ActionKind::Wait, 0, 0, 30},
      {irisu::ActionKind::StrongShot, 360, 320, 1},
  };
  for (const auto& action : actions) {
    first.step(action);
    second.step(action);
    assert(first.serialize_snapshot() == second.serialize_snapshot());
  }

  const auto checkpoint = first.serialize_snapshot();
  first.step({irisu::ActionKind::Wait, 0, 0, 40});
  const auto future = first.serialize_snapshot();
  first.restore_snapshot(checkpoint);
  first.step({irisu::ActionKind::Wait, 0, 0, 40});
  assert(first.serialize_snapshot() == future);

  irisu::Simulator other;
  other.reset(43);
  assert(other.state_hash() != second.state_hash());
}

void normal_reset_prefills_seeded_board() {
  irisu::Simulator simulator;
  const auto observation = simulator.reset(123);
  const auto state = simulator.clone_state();
  assert(observation.tick == 0 && observation.score == 0 &&
         observation.gauge == 3'000 && observation.level == 1);
  assert(observation.bodies.size() == 20);
  assert(state.scene_frame == 0 && state.spawn_count == 0);
  assert(state.level_shape_cutoff == 63 &&
         state.next_special_clear_count == 51);
  assert(state.rng_index == 102 && state.next_body_id == 21 &&
         state.actor_pool_cursor == 24);

  const auto& rotten = body(state, 1);
  assert(rotten.actor_slot == 5 && rotten.shape == irisu::Shape::Triangle &&
         rotten.color == 1 && rotten.size == 90.0 &&
         rotten.position.x == 276.0 && rotten.position.y == 200.0);
  assert(rotten.freshness_state == 3 && rotten.physics_owned &&
         rotten.top_contact_enabled &&
         rotten.lifecycle == irisu::Lifecycle::Rotten);
  assert(rotten.age_ticks == 1 && rotten.remaining_lifetime == 99'999 &&
         rotten.rot_timer == 1 && rotten.physics_update_count == 1);
  assert(rotten.velocity.x == 0.0 && rotten.velocity.y == 0.0 &&
         rotten.native_velocity.x == 0.0 && rotten.native_velocity.y == 0.0);

  const auto& falling = body(state, 11);
  assert(falling.actor_slot == 15 && falling.shape == irisu::Shape::Box &&
         falling.color == 2 && falling.size == 60.0 &&
         falling.position.x == 311.0);
  assert(std::bit_cast<std::uint32_t>(static_cast<float>(falling.position.y)) ==
         std::bit_cast<std::uint32_t>(60.2F));
  assert(falling.freshness_state == 2 && !falling.physics_owned &&
         !falling.top_contact_enabled &&
         falling.lifecycle == irisu::Lifecycle::ScriptedFalling);
  assert(falling.age_ticks == 1 && falling.remaining_lifetime == 99'999 &&
         falling.rot_timer == 0 && falling.physics_update_count == 0);
  assert(std::bit_cast<std::uint32_t>(
             static_cast<float>(falling.velocity.y)) == 0x3e4ccccdU);
  assert(falling.native_velocity.x == 0.0 &&
         falling.native_velocity.y == 0.0);

  const auto first = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  const auto after = simulator.clone_state();
  assert(event_count(first, irisu::EventKind::Spawned) == 1);
  assert(after.rng_index == 107 && after.spawn_count == 1 &&
         after.next_body_id == 22 && after.actor_pool_cursor == 25);
  const auto& cadence = body(after, 21);
  assert(cadence.actor_slot == 25 && cadence.shape == irisu::Shape::Box &&
         cadence.color == 2 && cadence.size == 90.0 &&
         cadence.position.x == 433.0);
}

void seed41_release_trace_regression() {
  irisu::Simulator simulator;
  simulator.reset(41);
  const auto first = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  std::vector<std::pair<irisu::BodyId, irisu::BodyId>> contacts;
  for (const auto& event : first.events) {
    if (event.kind == irisu::EventKind::Contact) {
      contacts.emplace_back(event.a, event.b);
    }
  }
  const std::vector<std::pair<irisu::BodyId, irisu::BodyId>> expected{
      {0, 21}, {14, 20}, {16, 20}, {19, 20}, {14, 19}, {16, 19},
      {14, 18}, {13, 18}, {14, 16}, {11, 15}, {13, 14}, {4, 10},
      {2, 10}, {1, 10}, {5, 10}, {7, 9}, {3, 9}, {1, 8}, {0, 6},
      {4, 5}, {1, 5}, {2, 4}, {1, 4},
  };
  assert(contacts == expected);
  std::vector<irisu::BodyId> destroyed;
  for (const auto& event : first.events) {
    if (event.kind == irisu::EventKind::Destroyed) {
      destroyed.push_back(event.a);
    }
  }
  assert((destroyed == std::vector<irisu::BodyId>{
                           14, 20, 16, 19, 18, 13, 11, 15}));

  irisu::StepResult score_step;
  for (std::uint32_t tick = 2; tick <= 304; ++tick) {
    const auto action = tick == 5
                            ? irisu::Action{irisu::ActionKind::StrongShot,
                                            453.0, 380.0, 1}
                        : tick == 273
                            ? irisu::Action{irisu::ActionKind::StrongShot,
                                            367.0, 380.0, 1}
                            : irisu::Action{irisu::ActionKind::Wait, 0, 0, 1};
    score_step = simulator.step(action);
    if (tick < 304) assert(simulator.observation().score == 0);
  }
  assert(score_step.reward == 16 && simulator.observation().score == 16);
  std::vector<std::int64_t> deltas;
  for (const auto& event : score_step.events) {
    if (event.kind == irisu::EventKind::ScoreChanged) {
      deltas.push_back(event.value);
    }
  }
  assert((deltas == std::vector<std::int64_t>{8, 8}));

  for (std::uint32_t tick = 305; tick <= 520; ++tick) {
    const auto action = tick == 365 || tick == 385
                            ? irisu::Action{irisu::ActionKind::StrongShot,
                                            303.0, 380.0, 1}
                            : irisu::Action{irisu::ActionKind::Wait, 0, 0, 1};
    const auto result = simulator.step(action);
    assert(event_count(result, irisu::EventKind::ScoreChanged) == 0);
  }
  assert(simulator.observation().score == 16);
}

void malformed_state_is_rejected_atomically() {
  irisu::Simulator simulator;
  simulator.reset(7);
  const auto stable = simulator.state_hash();
  auto bytes = simulator.serialize_snapshot();
  bytes.pop_back();
  bool rejected = false;
  try {
    simulator.restore_snapshot(bytes);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected && simulator.state_hash() == stable);

  auto state = simulator.clone_state();
  state.rng_state.fill(0);
  rejected = false;
  try {
    simulator.restore_state(state);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected && simulator.state_hash() == stable);

  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {300.0, 100.0});
  const auto body_stable = simulator.state_hash();
  state = simulator.clone_state();
  state.bodies.front().rule_guard_f0 = 1;
  rejected = false;
  try {
    simulator.restore_state(state);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected && simulator.state_hash() == body_stable);

  state = simulator.clone_state();
  state.next_body_id = 10;
  state.active_contact_pairs = {(std::uint64_t{8} << 32U) | 9U};
  rejected = false;
  try {
    simulator.restore_state(state);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected && simulator.state_hash() == body_stable);
}

void input_levels_are_ordered_and_held() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(5);
  const auto first =
      simulator.step({irisu::ActionKind::BothShots, 250, 350, 1});
  std::vector<irisu::EventKind> allocations;
  for (const auto& event : first.events) {
    if (event.kind == irisu::EventKind::ShotFired ||
        event.kind == irisu::EventKind::Spawned) {
      allocations.push_back(event.kind);
    }
  }
  assert((allocations == std::vector<irisu::EventKind>{
                             irisu::EventKind::ShotFired,
                             irisu::EventKind::ShotFired,
                             irisu::EventKind::Spawned}));
  assert(first.events[0].kind == irisu::EventKind::ShotFired &&
         first.events[0].value == 0);
  assert(first.events[1].kind == irisu::EventKind::ShotFired &&
         first.events[1].value == 1);
  assert(simulator.observation().left_held &&
         simulator.observation().right_held);

  const auto held =
      simulator.step({irisu::ActionKind::BothShots, 250, 350, 1});
  assert(event_count(held, irisu::EventKind::ShotFired) == 0);
  assert(event_count(held, irisu::EventKind::HeldInputIgnored) == 2);
  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  assert(!simulator.observation().left_held &&
         !simulator.observation().right_held);
  const auto pressed_again =
      simulator.step({irisu::ActionKind::BothShots, 250, 350, 1});
  assert(event_count(pressed_again, irisu::EventKind::ShotFired) == 2);
}

void replay_startup_suppresses_edges_but_retains_levels() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(5);
  suppress_cadence_spawn(simulator);

  const auto first = simulator.step(
      {irisu::ActionKind::WeakShot, 250, 350, 1, true});
  assert(event_count(first, irisu::EventKind::ShotFired) == 0);
  assert(simulator.observation().left_held &&
         !simulator.observation().right_held);

  // A different button first goes down on replay record 1. Its edge is also
  // suppressed, independently of the already-held left button.
  const auto second = simulator.step(
      {irisu::ActionKind::BothShots, 250, 350, 1, true});
  assert(event_count(second, irisu::EventKind::ShotFired) == 0);
  assert(simulator.observation().left_held &&
         simulator.observation().right_held);

  // Record 2 is no longer gated, but neither held button is a fresh edge.
  const auto third = simulator.step(
      {irisu::ActionKind::BothShots, 250, 350, 1, false});
  assert(event_count(third, irisu::EventKind::ShotFired) == 0);
  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  const auto pressed_again =
      simulator.step({irisu::ActionKind::BothShots, 250, 350, 1});
  assert(event_count(pressed_again, irisu::EventKind::ShotFired) == 2);
}

void replay_coordinate_range_preserves_edges() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(6);
  suppress_cadence_spawn(simulator);

  const auto first =
      simulator.step({irisu::ActionKind::WeakShot, 1023, 511, 1});
  assert(event_count(first, irisu::EventKind::InvalidAction) == 0);
  assert(event_count(first, irisu::EventKind::ShotFired) == 1);
  assert(simulator.observation().left_held);

  const auto held =
      simulator.step({irisu::ActionKind::WeakShot, 1023, 511, 1});
  assert(event_count(held, irisu::EventKind::ShotFired) == 0);
  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  const auto pressed_again =
      simulator.step({irisu::ActionKind::WeakShot, 1023, 511, 1});
  assert(event_count(pressed_again, irisu::EventKind::ShotFired) == 1);
  const auto invalid =
      simulator.step({irisu::ActionKind::WeakShot, 1024, 512, 1});
  assert(event_count(invalid, irisu::EventKind::InvalidAction) == 1);
  assert(event_count(invalid, irisu::EventKind::ShotFired) == 0);
  assert(!simulator.observation().left_held);
  const auto after_invalid =
      simulator.step({irisu::ActionKind::WeakShot, 1023, 511, 1});
  assert(event_count(after_invalid, irisu::EventKind::ShotFired) == 1);
}

void passive_drain_floors_at_one() {
  auto config = controlled_config();
  config.gauge_initial = 3;
  config.passive_gauge_decay_per_tick = 1;
  irisu::Simulator simulator(config);
  simulator.reset(1);
  const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 20});
  assert(!result.terminated);
  assert(simulator.observation().gauge == 1);
}

void truncation_is_persistent() {
  auto config = controlled_config();
  config.max_episode_ticks = 3;
  irisu::Simulator simulator(config);
  simulator.reset(11);
  const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 10});
  assert(!result.terminated && result.truncated);
  assert(simulator.observation().tick == 3);
  const auto hash = simulator.state_hash();
  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  assert(simulator.state_hash() == hash);
}

void fill_actor_pool(irisu::Simulator& simulator, std::size_t count) {
  for (std::size_t index = 0; index < count; ++index) {
    const irisu::Vec2 position{
        120.0 + 28.0 * static_cast<double>(index % 14),
        30.0 + 25.0 * static_cast<double>(index / 14)};
    assert(simulator.spawn_piece(irisu::Shape::Box,
                                 static_cast<std::int32_t>(index % 3), 2.0,
                                 position) != 0);
  }
}

void actor_pool_case(std::size_t existing, std::size_t expected_shots,
                     std::size_t expected_spawns,
                     std::uint32_t expected_rng_draws) {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(0x1234);
  fill_actor_pool(simulator, existing);
  const auto rng_before = simulator.clone_state().rng_index;
  const auto result =
      simulator.step({irisu::ActionKind::BothShots, 320, 200, 1});
  assert(event_count(result, irisu::EventKind::ShotFired) == expected_shots);
  assert(event_count(result, irisu::EventKind::Spawned) == expected_spawns);
  assert(simulator.clone_state().rng_index ==
         rng_before + expected_rng_draws);
}

void actor_pool_capacity_and_spawn_draws() {
  actor_pool_case(193, 2, 1, 5);
  actor_pool_case(194, 2, 0, 2);
  actor_pool_case(195, 1, 0, 2);
}

void failed_due_special_preserves_scheduler() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(0x5678);
  fill_actor_pool(simulator, 196);
  auto state = simulator.clone_state();
  state.next_special_clear_count = 0;
  simulator.restore_state(state);
  const auto rng_before = simulator.clone_state().rng_index;

  const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(event_count(result, irisu::EventKind::Spawned) == 0);
  assert(state.rng_index == rng_before + 2);
  assert(state.next_special_clear_count == 0);
}

void successful_due_special_reschedules_exactly() {
  const auto config = controlled_config();
  irisu::Simulator simulator(config);
  simulator.reset(0x9abc);
  auto state = simulator.clone_state();
  state.qualifying_clear_count = 17;
  state.next_special_clear_count = 17;
  simulator.restore_state(state);

  irisu::DxRandom expected;
  expected.restore(state.rng_state, state.rng_index);
  (void)expected.get_rand(69);
  (void)expected.get_rand(404);
  (void)expected.get_rand(1'000);
  (void)expected.get_rand(3);
  const auto next_threshold =
      state.qualifying_clear_count + config.special_clear_base +
      expected.get_rand(config.special_clear_random_max);

  const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(event_count(result, irisu::EventKind::Spawned) == 1);
  assert(state.bodies.size() == 1);
  assert(state.bodies.front().kind == irisu::BodyKind::Bonus);
  assert(state.bodies.front().shape == irisu::Shape::Circle);
  assert(state.bodies.front().special && state.bodies.front().color == -2);
  assert(state.next_special_clear_count == next_threshold);
  assert(state.rng_index == expected.index());
}

void projectile_postlude_and_oob_split() {
  auto config = controlled_config();
  config.out_of_bounds_min_y = 200.0;

  irisu::Simulator strong(config);
  strong.reset(13);
  suppress_cadence_spawn(strong);
  strong.step({irisu::ActionKind::StrongShot, 300, 100, 1});
  const auto strong_state = strong.clone_state();
  assert(strong_state.bodies.size() == 1);
  assert(strong_state.bodies.front().velocity.y == -50.0);
  assert(strong_state.bodies.front().native_velocity.y == -50.0);

  irisu::Simulator simulator(config);
  simulator.reset(13);
  suppress_cadence_spawn(simulator);

  simulator.step({irisu::ActionKind::WeakShot, 300, 100, 1});
  auto state = simulator.clone_state();
  assert(state.bodies.size() == 1);
  const auto id = state.bodies.front().id;
  assert(body(state, id).freshness_state == 2);
  assert(!body(state, id).top_contact_enabled);
  assert(body(state, id).physics_update_count == 1);
  assert(body(state, id).velocity.y == -25.0);

  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(body(state, id).top_contact_enabled);
  assert(body(state, id).physics_update_count == 2);

  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  const auto& ejected = body(state, id);
  assert(ejected.remaining_lifetime == 1);
  assert(ejected.velocity.x == 0.0 && ejected.velocity.y == 0.0);
  assert(ejected.native_velocity.y == -25.0);
  assert(ejected.scripted_velocity.y == 0.0);
  assert(ejected.physics_update_count == 3);

  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(body(state, id).pending_delete);
  assert(body(state, id).physics_update_count == 3);
}

void scripted_integrator_uses_float_stores() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(17);
  const double initial_y = static_cast<double>(100.1f);
  const auto id = simulator.spawn_piece(irisu::Shape::Box, 1, 30.0,
                                        {300.0, initial_y});
  auto state = simulator.clone_state();
  state.scene_frame = 1;
  body(state, id).angular_velocity = static_cast<double>(0.1f);
  body(state, id).angle = static_cast<double>(0.2f);
  simulator.restore_state(state);
  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  const auto& updated = body(state, id);
  const auto expected_y = static_cast<float>(
      static_cast<float>(initial_y) + static_cast<float>(0.2f));
  assert(std::bit_cast<std::uint32_t>(static_cast<float>(updated.position.y)) ==
         std::bit_cast<std::uint32_t>(expected_y));
  assert(std::bit_cast<std::uint32_t>(static_cast<float>(updated.velocity.y)) ==
         0x3e4ccccdU);
  assert(std::bit_cast<std::uint32_t>(static_cast<float>(updated.angle)) ==
         std::bit_cast<std::uint32_t>(0.3f));
  assert(updated.native_velocity.y == 0.0);
}

irisu::Snapshot burst_pair(irisu::Simulator& simulator, std::uint32_t level,
                           std::uint64_t clears, std::int64_t gauge) {
  const auto first = simulator.spawn_piece(irisu::Shape::Box, 2, 30.0,
                                           {250.0, 200.0});
  const auto second = simulator.spawn_piece(irisu::Shape::Box, 2, 30.0,
                                            {255.0, 200.0});
  auto state = simulator.clone_state();
  state.scene_frame = 1;
  state.level = level;
  state.qualifying_clear_count = clears;
  state.gauge = gauge;
  for (const auto id : {first, second}) {
    auto& value = body(state, id);
    value.physics_owned = true;
    value.freshness_state = 2;
    value.lifecycle = irisu::Lifecycle::DynamicFresh;
    value.top_contact_enabled = true;
    value.rot_timer = 1;
  }
  return state;
}

void group_burst_scores_each_member() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(19);
  simulator.restore_state(burst_pair(simulator, 1, 0, 1'000));
  const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  assert(result.reward == 16);
  const auto observation = simulator.observation();
  assert(observation.gauge == 1'700);
  assert(observation.highest_chain == 2);
  assert(observation.qualifying_clear_count == 1);
  assert(observation.bodies.empty());
  const auto state = simulator.clone_state();
  assert(state.bodies.size() == 2);
  assert(std::all_of(state.bodies.begin(), state.bodies.end(),
                     [](const irisu::Body& value) {
                       return value.pending_delete;
                     }));
}

void level_crossing_draws_cutoff_immediately() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(20);
  simulator.restore_state(burst_pair(simulator, 1, 5, 1'000));
  const auto rng_before = simulator.clone_state().rng_index;
  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  const auto state = simulator.clone_state();
  assert(state.level == 2);
  assert(state.qualifying_clear_count == 6);
  assert(state.rng_index == rng_before + 1);
  assert(state.score == 24);
}

void level_crossing_repeats_field_update() {
  auto config = controlled_config();
  config.spawn_interval_ticks = 100;
  irisu::Simulator simulator(config);
  simulator.reset(21);
  auto state = burst_pair(simulator, 1, 5, 1'000);
  state.scene_frame = 90;
  simulator.restore_state(state);
  const auto rng_before = simulator.clone_state().rng_index;

  const auto result =
      simulator.step({irisu::ActionKind::BothShots, 300, 350, 1});
  state = simulator.clone_state();
  assert(state.level == 2 && state.qualifying_clear_count == 6);
  assert(event_count(result, irisu::EventKind::LevelChanged) == 1);
  assert(event_count(result, irisu::EventKind::ShotFired) == 4);
  assert(event_count(result, irisu::EventKind::Spawned) == 2);
  assert(state.rng_index == rng_before + 11);

  std::vector<irisu::EventKind> field_allocations;
  for (const auto& event : result.events) {
    if (event.kind == irisu::EventKind::ShotFired ||
        event.kind == irisu::EventKind::Spawned) {
      field_allocations.push_back(event.kind);
    }
  }
  assert((field_allocations == std::vector<irisu::EventKind>{
                                   irisu::EventKind::ShotFired,
                                   irisu::EventKind::ShotFired,
                                   irisu::EventKind::Spawned,
                                   irisu::EventKind::ShotFired,
                                   irisu::EventKind::ShotFired,
                                   irisu::EventKind::Spawned}));
  for (const auto id : {irisu::BodyId{5}, irisu::BodyId{8}}) {
    const auto& spawned = body(state, id);
    assert(std::bit_cast<std::uint32_t>(
               static_cast<float>(spawned.scripted_velocity.y)) ==
           0x3e4ccccdU);
    assert(std::bit_cast<std::uint32_t>(
               static_cast<float>(spawned.position.y)) ==
           std::bit_cast<std::uint32_t>(-49.8F));
  }
}

irisu::Snapshot scored_actor_state(irisu::Simulator& simulator,
                                   bool delete_marked) {
  const auto id = simulator.spawn_piece(irisu::Shape::Box, 1, 30.0,
                                        {300.0, 200.0});
  auto state = simulator.clone_state();
  state.scene_frame = 1;
  state.next_chain_id = 2;
  irisu::GroupState group;
  group.id = 1;
  group.chain = 3;
  group.secondary_count = 3;
  group.num = 2;
  state.groups.push_back(group);
  auto& value = body(state, id);
  value.physics_owned = true;
  value.freshness_state = 2;
  value.grouped = true;
  value.chain_id = 1;
  value.lifecycle = irisu::Lifecycle::Confirmed;
  value.top_contact_enabled = true;
  value.successful_clear_pending = true;
  value.delete_marked = delete_marked;
  value.rot_timer = 1;
  return state;
}

void actor_postlude_order() {
  for (const bool delete_marked : {false, true}) {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(delete_marked ? 23 : 22);
    auto state = scored_actor_state(simulator, delete_marked);
    const auto id = state.bodies.front().id;
    simulator.restore_state(state);
    const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    assert(result.reward == 18);
    state = simulator.clone_state();
    const auto& updated = body(state, id);
    assert(updated.pending_delete);
    assert(state.highest_chain == 3);
    if (delete_marked) {
      assert(updated.rot_timer == 1);
      assert(updated.physics_update_count == 0);
    } else {
      assert(updated.rot_timer == 2);
      assert(updated.physics_update_count == 1);
    }
  }
}

void new_state_and_top_gate_postlude() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(29);
  const auto id = simulator.spawn_piece(irisu::Shape::Box, 1, 30.0,
                                        {300.0, 200.0});
  auto state = simulator.clone_state();
  state.scene_frame = 1;
  auto& value = body(state, id);
  value.physics_owned = true;
  value.lifecycle = irisu::Lifecycle::DynamicFresh;
  value.rot_timer = 1;
  simulator.restore_state(state);
  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(body(state, id).freshness_state == 2);
  assert(!body(state, id).top_contact_enabled);
  assert(body(state, id).rot_timer == 2);
  assert(body(state, id).physics_update_count == 1);

  auto& top_value = body(state, id);
  top_value.physics_owned = false;
  top_value.lifecycle = irisu::Lifecycle::ScriptedFalling;
  top_value.top_contact_pending = true;
  top_value.top_contact_enabled = false;
  simulator.restore_state(state);
  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(!body(state, id).top_contact_pending);
  assert(!body(state, id).top_contact_enabled);
  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  assert(simulator.clone_state().bodies.front().top_contact_enabled);
}

void activation_predicate_matrix() {
  const auto body_source = [](bool grouped, bool same_color,
                              bool expected_activation) {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(30);
    const auto target = simulator.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                              {300.0, 200.0});
    const auto source = simulator.spawn_piece(
        irisu::Shape::Box, same_color ? 0 : 1, 30.0, {300.0, 200.0});
    auto state = simulator.clone_state();
    state.scene_frame = 1;
    body(state, target).freshness_state = 2;
    make_dynamic(body(state, source));
    if (grouped) {
      state.next_chain_id = 2;
      state.groups.push_back({1, 1, 1, 0});
      body(state, source).grouped = true;
      body(state, source).chain_id = 1;
      body(state, source).lifecycle = irisu::Lifecycle::Confirmed;
    }
    simulator.restore_state(state);

    const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    state = simulator.clone_state();
    assert(body(state, target).physics_owned == expected_activation);
    assert(event_count(result, irisu::EventKind::Activated) ==
           static_cast<std::size_t>(expected_activation));
  };
  body_source(false, false, true);
  body_source(true, true, true);
  body_source(true, false, false);

  for (const auto [position, expected_activation] :
       {std::pair{irisu::Vec2{120.0, 200.0}, false},
        std::pair{irisu::Vec2{320.0, 410.0}, true}}) {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(31);
    const auto target =
        simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, position);
    auto state = simulator.clone_state();
    state.scene_frame = 1;
    body(state, target).freshness_state = 2;
    simulator.restore_state(state);
    const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    state = simulator.clone_state();
    assert(body(state, target).physics_owned == expected_activation);
    assert(event_count(result, irisu::EventKind::Activated) ==
           static_cast<std::size_t>(expected_activation));
  }

  irisu::Simulator projectile_source(controlled_config());
  projectile_source.reset(32);
  const auto target = projectile_source.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                                    {300.0, 200.0});
  auto state = projectile_source.clone_state();
  state.scene_frame = 1;
  body(state, target).freshness_state = 2;
  projectile_source.restore_state(state);
  const auto result = projectile_source.step(
      {irisu::ActionKind::WeakShot, 300.0, 200.0, 1});
  state = projectile_source.clone_state();
  const auto projectile = std::find_if(
      state.bodies.begin(), state.bodies.end(), [](const irisu::Body& value) {
        return value.kind == irisu::BodyKind::Projectile;
      });
  assert(projectile != state.bodies.end());
  assert(body(state, target).physics_owned);
  assert(projectile->delete_marked && !projectile->pending_delete);
  assert(event_count(result, irisu::EventKind::Activated) == 1);
}

void both_scripted_newborn_thresholds_and_special_exemption() {
  {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(33);
    const auto age_two = simulator.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                               {300.0, 200.0});
    const auto age_three = simulator.spawn_piece(irisu::Shape::Box, 1, 30.0,
                                                 {300.0, 200.0});
    auto state = simulator.clone_state();
    state.scene_frame = 1;
    body(state, age_two).age_ticks = 2;
    body(state, age_three).age_ticks = 3;
    simulator.restore_state(state);
    simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    state = simulator.clone_state();
    assert(body(state, age_two).pending_delete);
    assert(!body(state, age_three).pending_delete);
  }

  {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(34);
    const auto special = simulator.spawn_bonus({300.0, 200.0});
    const auto newborn = simulator.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                               {300.0, 200.0});
    auto state = simulator.clone_state();
    state.scene_frame = 1;
    body(state, special).age_ticks = 2;
    body(state, newborn).age_ticks = 2;
    simulator.restore_state(state);
    simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    state = simulator.clone_state();
    assert(!body(state, special).pending_delete);
    assert(body(state, newborn).pending_delete);
  }
}

void physics_top_contact_delays_enable_latch() {
  auto config = controlled_config();
  config.scripted_fall_speed = 40.0;
  irisu::Simulator simulator(config);
  simulator.reset(35);
  const auto id = simulator.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                        {320.0, 0.0});
  auto state = simulator.clone_state();
  state.scene_frame = 1;
  body(state, id).freshness_state = 2;
  simulator.restore_state(state);

  const auto touching = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(event_count(touching, irisu::EventKind::Contact) == 1);
  assert(!body(state, id).top_contact_pending);
  assert(!body(state, id).top_contact_enabled);
  assert(!body(state, id).physics_owned);

  const auto departed = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(event_count(departed, irisu::EventKind::Contact) == 0);
  assert(body(state, id).top_contact_enabled);
  assert(!body(state, id).physics_owned);
}

void rotten_pair_skips_contact_counters() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(31);
  auto state = burst_pair(simulator, 1, 0, 1'000);
  for (auto& value : state.bodies) {
    value.freshness_state = 3;
    value.lifecycle = irisu::Lifecycle::Rotten;
    value.rot_timer = 0;
    value.non_wall_contacts = 0;
  }
  simulator.restore_state(state);
  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(state.bodies.size() == 2);
  assert(state.bodies[0].non_wall_contacts == 0);
  assert(state.bodies[1].non_wall_contacts == 0);
}

void special_orb_rules_and_duplicate_rewards() {
  {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(31);
    const auto stale = simulator.spawn_piece(irisu::Shape::Box, 1, 30.0,
                                             {100.0, 200.0});
    auto state = simulator.clone_state();
    state.scene_frame = 1;
    make_dynamic(body(state, stale));
    body(state, stale).delete_marked = true;
    simulator.restore_state(state);
    simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    assert(simulator.clone_state().bodies.empty());

    const auto orb = simulator.spawn_bonus({250.0, 200.0});
    const auto target = simulator.spawn_piece(irisu::Shape::Box, 1, 30.0,
                                              {250.0, 200.0});
    state = simulator.clone_state();
    for (const auto id : {orb, target}) make_dynamic(body(state, id));
    simulator.restore_state(state);

    simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    state = simulator.clone_state();
    // Both the live target and the inactive slot retaining color 1 earn 700.
    assert(state.gauge == 2'400);
    assert(body(state, orb).pending_delete);
    assert(body(state, target).pending_delete);
  }

  {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(32);
    const auto first_orb = simulator.spawn_bonus({250.0, 200.0});
    const auto second_orb = simulator.spawn_bonus({250.0, 200.0});
    const auto target = simulator.spawn_piece(irisu::Shape::Box, 1, 30.0,
                                              {250.0, 200.0});
    auto state = simulator.clone_state();
    state.scene_frame = 1;
    for (const auto id : {first_orb, second_orb, target}) {
      make_dynamic(body(state, id));
    }
    simulator.restore_state(state);

    simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    state = simulator.clone_state();
    assert(state.gauge == 2'400);
    assert(body(state, first_orb).pending_delete);
    assert(body(state, second_orb).pending_delete);
    assert(body(state, target).pending_delete);
  }

  {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(33);
    const auto orb = simulator.spawn_bonus({250.0, 200.0});
    const auto target = simulator.spawn_piece(irisu::Shape::Box, 1, 30.0,
                                              {250.0, 200.0});
    auto state = simulator.clone_state();
    state.scene_frame = 1;
    make_dynamic(body(state, orb));
    make_dynamic(body(state, target), 3);
    simulator.restore_state(state);

    simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    state = simulator.clone_state();
    assert(state.gauge == 1'000);
    assert(body(state, orb).pending_delete);
    assert(!body(state, target).pending_delete);
    assert(body(state, target).freshness_state == 3);
  }

  {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(34);
    const auto orb = simulator.spawn_bonus({300.0, 200.0});
    auto state = simulator.clone_state();
    state.scene_frame = 1;
    make_dynamic(body(state, orb));
    simulator.restore_state(state);

    simulator.step({irisu::ActionKind::WeakShot, 300, 200, 1});
    state = simulator.clone_state();
    assert(!body(state, orb).delete_marked && !body(state, orb).pending_delete);
    const auto projectile = std::find_if(
        state.bodies.begin(), state.bodies.end(), [](const irisu::Body& value) {
          return value.kind == irisu::BodyKind::Projectile;
        });
    assert(projectile != state.bodies.end());
    assert(projectile->delete_marked && !projectile->pending_delete);
  }
}

void projectile_contact_dispatch_rules() {
  {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(34);
    suppress_cadence_spawn(simulator);
    simulator.step({irisu::ActionKind::BothShots, 300, 200, 1});
    const auto state = simulator.clone_state();
    assert(state.bodies.size() == 2);
    assert(std::all_of(state.bodies.begin(), state.bodies.end(),
                       [](const irisu::Body& value) {
                         return value.kind == irisu::BodyKind::Projectile &&
                                value.non_wall_contacts == 0 &&
                                !value.delete_marked && !value.pending_delete;
                       }));
  }

  {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(36);
    const auto first = simulator.spawn_piece(irisu::Shape::Box, 1, 30.0,
                                             {300.0, 200.0});
    const auto second = simulator.spawn_piece(irisu::Shape::Box, 1, 30.0,
                                              {300.0, 200.0});
    auto state = simulator.clone_state();
    state.scene_frame = 1;
    state.next_chain_id = 3;
    state.groups.push_back({1, 1, 1, 0});
    state.groups.push_back({2, 1, 1, 0});
    body(state, first).grouped = true;
    body(state, first).chain_id = 1;
    body(state, second).grouped = true;
    body(state, second).chain_id = 2;
    make_dynamic(body(state, first));
    make_dynamic(body(state, second));
    simulator.restore_state(state);

    simulator.step({irisu::ActionKind::WeakShot, 300, 200, 1});
    state = simulator.clone_state();
    assert(body(state, first).projectile_hits +
               body(state, second).projectile_hits ==
           1);
    const auto projectile = std::find_if(
        state.bodies.begin(), state.bodies.end(), [](const irisu::Body& value) {
          return value.kind == irisu::BodyKind::Projectile;
        });
    assert(projectile != state.bodies.end());
    assert(projectile->non_wall_contacts == 2);
    assert(projectile->delete_marked);
    assert(!projectile->pending_delete);
  }


  {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(37);
    const auto block_id = simulator.spawn_piece(irisu::Shape::Box, 1, 30.0,
                                                {300.0, 200.0});
    auto state = simulator.clone_state();
    state.scene_frame = 1;
    state.next_chain_id = 2;
    state.groups.push_back({1, 1, 1, 0});
    auto& block_value = body(state, block_id);
    block_value.grouped = true;
    block_value.chain_id = 1;
    block_value.projectile_hits = 1;
    make_dynamic(block_value);
    simulator.restore_state(state);

    simulator.step({irisu::ActionKind::WeakShot, 300, 200, 1});
    state = simulator.clone_state();
    assert(body(state, block_id).projectile_hits == 2);
    assert(body(state, block_id).pending_delete);
  }
}

void rotten_primary_scores_without_gauge() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(38);
  const auto id = simulator.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                        {320.0, 410.0});
  auto state = simulator.clone_state();
  state.scene_frame = 1;
  state.next_chain_id = 2;
  state.groups.push_back({1, 1, 1, 0});
  body(state, id).grouped = true;
  body(state, id).chain_id = 1;
  make_dynamic(body(state, id), 3);
  simulator.restore_state(state);

  const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  assert(result.reward == 4);
  assert(simulator.observation().gauge == 1'000);
  assert(simulator.observation().qualifying_clear_count == 1);
}

void groups_do_not_merge() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(36);
  const auto first = simulator.spawn_piece(irisu::Shape::Box, 2, 30.0,
                                           {300.0, 200.0});
  const auto second = simulator.spawn_piece(irisu::Shape::Box, 2, 30.0,
                                            {300.0, 200.0});
  auto state = simulator.clone_state();
  state.scene_frame = 1;
  state.next_chain_id = 3;
  state.groups.push_back({1, 1, 1, 0});
  state.groups.push_back({2, 1, 1, 0});
  body(state, first).grouped = true;
  body(state, first).chain_id = 1;
  body(state, second).grouped = true;
  body(state, second).chain_id = 2;
  make_dynamic(body(state, first));
  make_dynamic(body(state, second));
  simulator.restore_state(state);

  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(body(state, first).chain_id == 1);
  assert(body(state, second).chain_id == 2);
  assert(state.groups[0].chain == 1 && state.groups[1].chain == 1);
}

void rot_threshold_penalty_and_next_frame_finish() {
  {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(37);
    const auto id = simulator.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                          {320.0, 410.0});
    auto state = simulator.clone_state();
    state.scene_frame = 1;
    make_dynamic(body(state, id));
    body(state, id).age_ticks = 100;
    simulator.restore_state(state);

    simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    state = simulator.clone_state();
    assert(body(state, id).age_ticks == 101);
    assert(body(state, id).rot_timer == 0);
    simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    assert(body(simulator.clone_state(), id).rot_timer == 2);
  }

  {
    auto config = controlled_config();
    config.passive_gauge_decay_per_tick = 1;
    irisu::Simulator simulator(config);
    simulator.reset(38);
    const auto id = simulator.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                          {320.0, 200.0});
    auto state = simulator.clone_state();
    state.scene_frame = 1;
    make_dynamic(body(state, id));
    body(state, id).rot_timer = 119;
    simulator.restore_state(state);

    auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    state = simulator.clone_state();
    assert(!result.terminated && body(state, id).rot_timer == 120);
    assert(state.gauge == 999);
    result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    state = simulator.clone_state();
    assert(!result.terminated && body(state, id).freshness_state == 3);
    assert(state.gauge == -822);
    result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    assert(result.terminated);
    assert(simulator.observation().gauge == 1);
  }
}

void projectile_inherits_rot_without_rotting() {
  auto config = controlled_config();
  config.rot_delay_ticks = 1;
  irisu::Simulator simulator(config);
  simulator.reset(39);
  suppress_cadence_spawn(simulator);
  simulator.step({irisu::ActionKind::WeakShot, 300.0, 200.0, 1});
  const auto source = simulator.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                            {300.0, 200.0});
  auto state = simulator.clone_state();
  const auto projectile = std::find_if(
      state.bodies.begin(), state.bodies.end(), [](const irisu::Body& value) {
        return value.kind == irisu::BodyKind::Projectile;
      });
  assert(projectile != state.bodies.end());
  const auto projectile_id = projectile->id;
  make_dynamic(body(state, source), 3);
  body(state, source).rot_timer = 2;
  body(state, projectile_id).age_ticks = 101;
  simulator.restore_state(state);

  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert(body(state, projectile_id).rot_timer == 2);
  assert(body(state, projectile_id).freshness_state == 2);
  assert(!body(state, projectile_id).delete_marked);
  assert(!body(state, projectile_id).pending_delete);
  assert(state.gauge == config.gauge_initial);
}

void strict_oob_and_newborn_lifetime() {
  {
    irisu::Simulator boundary(controlled_config());
    boundary.reset(39);
    const auto id = boundary.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                         {320.0, 560.0});
    auto state = boundary.clone_state();
    state.scene_frame = 1;
    make_dynamic(body(state, id));
    boundary.restore_state(state);
    boundary.step({irisu::ActionKind::Wait, 0, 0, 1});
    assert(body(boundary.clone_state(), id).remaining_lifetime == 9'999);

    irisu::Simulator outside(controlled_config());
    outside.reset(40);
    const auto outside_id = outside.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                                {320.0, 561.0});
    state = outside.clone_state();
    state.scene_frame = 1;
    make_dynamic(body(state, outside_id));
    outside.restore_state(state);
    outside.step({irisu::ActionKind::Wait, 0, 0, 1});
    assert(body(outside.clone_state(), outside_id).remaining_lifetime == 1);
  }

  {
    auto config = controlled_config();
    config.projectile_life_ticks = 1;
    irisu::Simulator simulator(config);
    simulator.reset(41);
    suppress_cadence_spawn(simulator);
    simulator.step({irisu::ActionKind::WeakShot, 300, 200, 1});
    auto state = simulator.clone_state();
    const auto id = state.bodies.front().id;
    assert(body(state, id).freshness_state == 2);
    assert(body(state, id).delete_marked);
    assert(!body(state, id).pending_delete);
    simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    assert(body(simulator.clone_state(), id).pending_delete);
  }
}

void actor_slot_order_controls_destroy_queue() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(42);
  const auto first = simulator.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                           {220.0, 200.0});
  const auto second = simulator.spawn_piece(irisu::Shape::Box, 0, 30.0,
                                            {420.0, 200.0});
  auto state = simulator.clone_state();
  state.scene_frame = 1;
  state.next_chain_id = 2;
  state.groups.push_back({1, 2, 2, 1});
  for (const auto id : {first, second}) {
    body(state, id).grouped = true;
    body(state, id).chain_id = 1;
    body(state, id).successful_clear_pending = true;
    make_dynamic(body(state, id));
  }
  body(state, first).actor_slot = 6;
  body(state, second).actor_slot = 5;
  body(state, second).delete_marked = true;
  simulator.restore_state(state);

  simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  state = simulator.clone_state();
  assert((state.physics_ordering.destroy_order ==
          std::vector<irisu::BodyId>{first, second}));
  assert(body(state, first).pending_delete);
  assert(body(state, second).pending_delete);
}

void scripted_teardown_preserves_native_destroy_order() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(37);
  simulator.spawn_piece(irisu::Shape::Box, 0, 30.0, {250.0, 200.0});
  simulator.spawn_piece(irisu::Shape::Box, 1, 30.0, {255.0, 200.0});
  suppress_cadence_spawn(simulator);
  const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  const auto contact = std::find_if(
      result.events.begin(), result.events.end(), [](const irisu::Event& event) {
        return event.kind == irisu::EventKind::Contact && event.a != 0 &&
               event.b != 0;
      });
  assert(contact != result.events.end());
  const auto state = simulator.clone_state();
  assert((state.physics_ordering.destroy_order ==
          std::vector<irisu::BodyId>{contact->b, contact->a}));
  assert(std::all_of(state.bodies.begin(), state.bodies.end(),
                     [](const irisu::Body& value) {
                       return value.pending_delete;
                     }));
}

void terminal_metadata_is_first_and_latest() {
  irisu::Simulator simulator(controlled_config());
  simulator.reset(41);
  simulator.restore_state(burst_pair(simulator, 99, 593, 0));
  const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  assert(result.terminated);
  const auto observation = simulator.observation();
  const auto& diagnostics = result.diagnostics;
  assert(diagnostics.config_hash == simulator.config_hash());
  assert(diagnostics.finish_call_count == 2);
  assert(diagnostics.terminal_metadata_recorded);
  assert(diagnostics.recorded_final_score == 0);
  assert(diagnostics.recorded_final_highest_chain == 0);
  assert(diagnostics.recorded_final_level == 99);
  assert(diagnostics.recorded_final_clears == 593);
  assert(diagnostics.latest_final_score == 0);
  assert(diagnostics.latest_final_highest_chain == 0);
  assert(diagnostics.latest_final_level == 100);
  assert(diagnostics.latest_final_clears == 594);
  assert(observation.score > diagnostics.latest_final_score);
  assert(observation.highest_chain == 2);

  const auto terminal_hash = simulator.state_hash();
  const auto repeated = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
  assert(repeated.diagnostics.finish_call_count == 2);
  assert(simulator.state_hash() == terminal_hash);

  auto malformed = simulator.clone_state();
  malformed.terminated = false;
  bool rejected = false;
  try {
    simulator.restore_state(malformed);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected && simulator.state_hash() == terminal_hash);

  irisu::Simulator active(controlled_config());
  active.reset(42);
  malformed = active.clone_state();
  malformed.terminated = true;
  rejected = false;
  try {
    active.restore_state(malformed);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void level_cap_repeats_at_count_600() {
  {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(43);
    simulator.restore_state(burst_pair(simulator, 100, 594, 1'000));
    const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    assert(!result.terminated);
    assert(simulator.observation().qualifying_clear_count == 595);
    assert(result.diagnostics.finish_call_count == 0);
  }
  {
    irisu::Simulator simulator(controlled_config());
    simulator.reset(44);
    simulator.restore_state(burst_pair(simulator, 100, 599, 1'000));
    const auto rng_index = simulator.clone_state().rng_index;
    const auto result = simulator.step({irisu::ActionKind::Wait, 0, 0, 1});
    assert(result.terminated);
    const auto observation = simulator.observation();
    assert(observation.qualifying_clear_count == 600);
    assert(result.diagnostics.finish_call_count == 1);
    assert(result.diagnostics.recorded_final_clears == 600);
    assert(simulator.clone_state().rng_index == rng_index);
  }
}

void default_seed_runs_headlessly_to_game_over() {
  irisu::Simulator simulator;
  auto observation = simulator.reset(0);
  irisu::StepResult result;
  while (!observation.terminated && observation.tick < 10'000) {
    result = simulator.step({irisu::ActionKind::Wait, 0.0, 0.0, 1});
    observation = simulator.observation();
  }
  assert(result.terminated && !result.truncated);
  assert(observation.tick <= 10'000);
  assert(observation.score >= 0 && observation.level == 1);
  assert(event_count(result, irisu::EventKind::GameOver) == 1);
}

void seed_domain_is_explicit_and_rejection_is_atomic() {
  irisu::Simulator simulator;
  simulator.reset(7);
  const auto stable = simulator.serialize_snapshot();
  bool rejected = false;
  try {
    simulator.reset(std::uint64_t{1} << 32U);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
  assert(simulator.serialize_snapshot() == stable);

  const auto maximum = simulator.reset(
      std::numeric_limits<std::uint32_t>::max());
  irisu::Simulator control;
  const auto expected = control.reset(
      std::numeric_limits<std::uint32_t>::max());
  assert(maximum.tick == expected.tick && maximum.score == expected.score &&
         maximum.gauge == expected.gauge &&
         maximum.bodies.size() == expected.bodies.size());
  assert(simulator.state_hash() == control.state_hash());
}

}  // namespace

int main() {
  deterministic_trace_and_snapshot();
  normal_reset_prefills_seeded_board();
  seed41_release_trace_regression();
  malformed_state_is_rejected_atomically();
  input_levels_are_ordered_and_held();
  replay_startup_suppresses_edges_but_retains_levels();
  replay_coordinate_range_preserves_edges();
  passive_drain_floors_at_one();
  truncation_is_persistent();
  actor_pool_capacity_and_spawn_draws();
  failed_due_special_preserves_scheduler();
  successful_due_special_reschedules_exactly();
  projectile_postlude_and_oob_split();
  scripted_integrator_uses_float_stores();
  group_burst_scores_each_member();
  level_crossing_draws_cutoff_immediately();
  level_crossing_repeats_field_update();
  actor_postlude_order();
  new_state_and_top_gate_postlude();
  activation_predicate_matrix();
  both_scripted_newborn_thresholds_and_special_exemption();
  physics_top_contact_delays_enable_latch();
  rotten_pair_skips_contact_counters();
  special_orb_rules_and_duplicate_rewards();
  projectile_contact_dispatch_rules();
  rotten_primary_scores_without_gauge();
  groups_do_not_merge();
  rot_threshold_penalty_and_next_frame_finish();
  projectile_inherits_rot_without_rotting();
  strict_oob_and_newborn_lifetime();
  actor_slot_order_controls_destroy_queue();
  scripted_teardown_preserves_native_destroy_order();
  terminal_metadata_is_first_and_latest();
  level_cap_repeats_at_count_600();
  default_seed_runs_headlessly_to_game_over();
  seed_domain_is_explicit_and_rejection_is_atomic();
  std::cout << "native simulator tests passed\n";
}
