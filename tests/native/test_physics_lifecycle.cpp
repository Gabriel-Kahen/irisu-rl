#include "irisu/c_api.h"
#include "irisu/physics.hpp"
#include "irisu/simulator.hpp"

#include <Box2D.h>

#include <array>
#include <barrier>
#include <bit>
#include <cfenv>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__SSE__) || defined(_M_X64) ||                                     \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#define IRISU_TEST_HAS_MXCSR 1
#endif

namespace {

void require(bool condition, std::string_view message) {
  if (!condition)
    throw std::runtime_error(std::string(message));
}

struct NativeHandle {
  NativeHandle() : value(irisu_create()) {}
  ~NativeHandle() { irisu_destroy(value); }
  NativeHandle(const NativeHandle &) = delete;
  NativeHandle &operator=(const NativeHandle &) = delete;
  irisu_simulator *value;
};

struct FloatingPointState {
  int rounding{};
  int exceptions{};
#if defined(IRISU_GNU_X87_CONTROL_WORD_ENVIRONMENT)
  std::uint16_t x87_control{};
#endif
#if defined(IRISU_TEST_HAS_MXCSR)
  std::uint32_t mxcsr{};
#endif

  friend bool operator==(const FloatingPointState &,
                         const FloatingPointState &) = default;
};

FloatingPointState floating_point_state() {
  FloatingPointState state;
  state.rounding = std::fegetround();
  state.exceptions = std::fetestexcept(FE_ALL_EXCEPT);
#if defined(IRISU_GNU_X87_CONTROL_WORD_ENVIRONMENT)
  __asm__ __volatile__("fnstcw %0" : "=m"(state.x87_control));
#endif
#if defined(IRISU_TEST_HAS_MXCSR)
  state.mxcsr = _mm_getcsr();
#endif
  return state;
}

class HostileFloatingPointEnvironment {
public:
  HostileFloatingPointEnvironment() {
    require(std::fegetenv(&original_) == 0,
            "failed to save test floating-point environment");
    require(std::feclearexcept(FE_ALL_EXCEPT) == 0,
            "failed to clear test floating-point exceptions");
    require(std::fesetround(FE_DOWNWARD) == 0,
            "failed to set hostile rounding mode");
#if defined(IRISU_GNU_X87_CONTROL_WORD_ENVIRONMENT)
    std::uint16_t control{};
    __asm__ __volatile__("fnstcw %0" : "=m"(control));
    control = static_cast<std::uint16_t>(control & ~0x0300U);
    __asm__ __volatile__("fldcw %0" : : "m"(control));
#endif
    expected_ = floating_point_state();
  }

  ~HostileFloatingPointEnvironment() { (void)std::fesetenv(&original_); }

  const FloatingPointState &expected() const { return expected_; }

private:
  std::fenv_t original_{};
  FloatingPointState expected_{};
};

void require_caller_environment(const FloatingPointState &expected,
                                std::string_view operation) {
  require(floating_point_state() == expected,
          std::string(operation) +
              " did not restore the caller FP environment");
}

void floating_point_metadata_matches_build_capability() {
  const char *information = irisu_build_info_json();
  require(information != nullptr, "missing native build metadata");
  const std::string_view metadata(information);
#if defined(IRISU_GNU_X87_CONTROL_WORD_ENVIRONMENT)
  require(metadata.find("\"fp_environment\":\"nearest,x87-pc53\"") !=
              std::string_view::npos,
          "x87 control-word build did not advertise PC53");
#else
  require(metadata.find("\"fp_environment\":\"nearest\"") !=
              std::string_view::npos,
          "non-x87-control-word build advertised the wrong FP environment");
#endif
}

void hostile_rounding_isolated_from_native_simulation() {
  irisu::Simulator baseline;
  baseline.reset(41);
  baseline.step({irisu::ActionKind::Wait, 0.0, 0.0, 1});
  const auto expected_hash = baseline.state_hash();

  HostileFloatingPointEnvironment hostile;
  const auto caller = hostile.expected();
  irisu::Simulator guarded;
  require_caller_environment(caller, "Simulator construction");
  guarded.reset(41);
  require_caller_environment(caller, "Simulator reset");
  guarded.step({irisu::ActionKind::Wait, 0.0, 0.0, 1});
  require_caller_environment(caller, "Simulator step");
  require(guarded.state_hash() == expected_hash,
          "hostile rounding changed scalar simulator state");
  require_caller_environment(caller, "Simulator state_hash");
  try {
    guarded.restore_snapshot({});
    require(false, "empty snapshot unexpectedly restored");
  } catch (const std::invalid_argument &) {
  }
  require_caller_environment(caller, "throwing snapshot restore");
}

void hostile_rounding_isolated_from_padded_workers() {
  HostileFloatingPointEnvironment hostile;
  const auto caller = hostile.expected();
  constexpr std::size_t kLanes = 8;
  {
    std::array<NativeHandle, kLanes> batched;
    NativeHandle scalar;
    std::array<irisu_simulator *, kLanes> handles{};
    std::array<irisu_padded_observation_v1, kLanes> observations{};
    irisu_padded_observation_v1 scalar_observation{};
    require(irisu_padded_reset(scalar.value, 41, &scalar_observation) == 1,
            "hostile scalar reset failed");
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
      handles[lane] = batched[lane].value;
      require(
          irisu_padded_reset(batched[lane].value, 41, &observations[lane]) == 1,
          "hostile padded reset failed");
    }
    require_caller_environment(caller, "hostile handle creation and reset");

    for (std::uint32_t round = 0; round < 256; ++round) {
      irisu_padded_action_v1 action{};
      action.kind = round % 11U == 0 ? 2 : 0;
      action.x = 453.0;
      action.y = 380.0;
      action.wait_ticks = 1;
      std::array<irisu_padded_action_v1, kLanes> actions;
      actions.fill(action);
      std::array<irisu_padded_transition_v1, kLanes> transitions{};
      std::array<std::uint8_t, kLanes> statuses{};
      require(irisu_padded_step_batch(handles.data(), actions.data(),
                                      transitions.data(), statuses.data(),
                                      kLanes, kLanes) == 1,
              "hostile padded batch failed");
      irisu_padded_transition_v1 scalar_transition{};
      require(irisu_padded_step(scalar.value, action.kind, action.x, action.y,
                                action.wait_ticks, &scalar_transition) == 1,
              "hostile scalar padded step failed");
      for (std::size_t lane = 0; lane < kLanes; ++lane) {
        require(statuses[lane] == 1, "hostile padded lane failed");
        require(irisu_state_hash(batched[lane].value) ==
                    irisu_state_hash(scalar.value),
                "hostile padded worker diverged from scalar state");
        require(transitions[lane].observation.tick ==
                        scalar_transition.observation.tick &&
                    transitions[lane].reward == scalar_transition.reward &&
                    transitions[lane].event_count ==
                        scalar_transition.event_count,
                "hostile padded worker transition diverged from scalar");
      }
      require_caller_environment(caller,
                                 "hostile padded batch and scalar step");
    }
  }
  require_caller_environment(caller, "hostile handle destruction");
}

void dirty_pair_manager_storage_is_safe() {
  alignas(b2PairManager) std::array<std::byte, sizeof(b2PairManager)> storage;
  std::memset(storage.data(), 0xa5, storage.size());
  auto *manager = new (storage.data()) b2PairManager;
  require(manager->m_pairBufferCount == 0,
          "pair-buffer count must be initialized on dirty storage");
  require(manager->m_broadPhase == nullptr && manager->m_callback == nullptr,
          "pair-manager owner pointers must start null");
  manager->~b2PairManager();
}

void repeated_c_abi_lifecycle() {
  const irisu_config_override override_value{"gravity_y", 360.0};
  std::uint64_t expected_config = 0;
  for (std::uint64_t iteration = 0; iteration < 2'000; ++iteration) {
    std::vector<std::uint64_t> allocator_perturbation(
        static_cast<std::size_t>(iteration % 31 + 1), iteration ^ 0xa5a5a5a5U);
    NativeHandle handle;
    require(handle.value != nullptr, "irisu_create failed");
    require(irisu_configure(handle.value, &override_value, 1) == 1,
            "irisu_configure failed");
    require(irisu_reset(handle.value, 123) == 1, "irisu_reset failed");
    const std::uint64_t config = irisu_config_hash(handle.value);
    if (iteration == 0)
      expected_config = config;
    require(config == expected_config,
            "configuration hash changed across handles");
    require(irisu_step(handle.value, 0, 0.0, 0.0, 2) == 1, "irisu_step failed");
    require(irisu_state_hash(handle.value) != 0,
            "state hash unexpectedly zero");
    require(!allocator_perturbation.empty(), "allocator perturbation vanished");
  }
}

irisu::Body circle(irisu::BodyId id, double x) {
  irisu::Body body;
  body.id = id;
  body.shape = irisu::Shape::Circle;
  body.lifecycle = irisu::Lifecycle::DynamicFresh;
  body.position = {x, 200.0};
  body.size = 40.0;
  body.density = 1.0;
  body.friction = 0.7;
  body.restitution = 0.1;
  return body;
}

std::uint64_t concurrent_world_trace() {
  irisu::MechanicsConfig config;
  config.gravity_y = 100.0;
  config.spawn_interval_ticks = 100'000;
  irisu::PhysicsWorld world(config);
  std::vector bodies{circle(1, 240.0), circle(2, 268.0)};
  for (auto &body : bodies)
    world.initialize_mass(body);
  world.rebuild(bodies);
  for (int tick = 0; tick < 40; ++tick)
    world.step(bodies);

  std::uint64_t hash = 0xcbf29ce484222325ULL;
  const auto mix = [&hash](double value) {
    hash ^= std::bit_cast<std::uint64_t>(value);
    hash *= 0x100000001b3ULL;
  };
  for (const auto &body : bodies) {
    mix(body.position.x);
    mix(body.position.y);
    mix(body.velocity.x);
    mix(body.velocity.y);
    mix(body.angle);
  }
  return hash;
}

void concurrent_independent_worlds_and_handles() {
  constexpr int kThreads = 8;
  constexpr int kRounds = 100;
  std::barrier start(kThreads);
  std::array<std::uint64_t, kThreads> traces{};
  std::array<std::exception_ptr, kThreads> failures{};
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int thread_index = 0; thread_index < kThreads; ++thread_index) {
    threads.emplace_back([&, thread_index] {
      try {
        start.arrive_and_wait();
        const std::uint64_t trace = concurrent_world_trace();
        for (int round = 0; round < kRounds; ++round) {
          require(concurrent_world_trace() == trace,
                  "per-world physics trace changed under concurrency");
          NativeHandle handle;
          require(handle.value != nullptr, "concurrent irisu_create failed");
          require(irisu_reset(handle.value, 77) == 1,
                  "concurrent irisu_reset failed");
          require(irisu_step(handle.value, 0, 0.0, 0.0, 3) == 1,
                  "concurrent irisu_step failed");
        }
        traces[thread_index] = trace;
      } catch (...) {
        failures[thread_index] = std::current_exception();
      }
    });
  }
  for (auto &thread : threads)
    thread.join();
  for (const auto &failure : failures) {
    if (failure)
      std::rethrow_exception(failure);
  }
  for (int index = 1; index < kThreads; ++index) {
    require(traces[index] == traces[0],
            "identical independent worlds produced different traces");
  }
}

void padded_abi_layout_and_buffer_guards() {
  static_assert(IRISU_ACTION_KIND_WAIT == 0 &&
                IRISU_ACTION_KIND_WEAK_SHOT == 1 &&
                IRISU_ACTION_KIND_STRONG_SHOT == 2 &&
                IRISU_ACTION_KIND_BOTH_SHOTS == 3);
  static_assert(IRISU_BODY_KIND_PIECE == 0 &&
                IRISU_BODY_KIND_PROJECTILE == 1 &&
                IRISU_BODY_KIND_BONUS == 2);
  static_assert(IRISU_SHAPE_CIRCLE == 0 && IRISU_SHAPE_BOX == 1 &&
                IRISU_SHAPE_TRIANGLE == 2);
  static_assert(IRISU_LIFECYCLE_SCRIPTED_FALLING == 0 &&
                IRISU_LIFECYCLE_DELETED == 4);
  static_assert(IRISU_EVENT_KIND_INVALID_ACTION == 0 &&
                IRISU_EVENT_KIND_LEVEL_COMPLETED == 18);
  require(irisu_padded_abi_version() == 1, "unexpected padded ABI version");
  require(irisu_padded_body_capacity() == IRISU_PADDED_BODY_CAPACITY,
          "padded body capacity query drifted");
  require(irisu_padded_observation_size() ==
                  sizeof(irisu_padded_observation_v1) &&
              irisu_padded_transition_size() ==
                  sizeof(irisu_padded_transition_v1) &&
              irisu_padded_action_size() == sizeof(irisu_padded_action_v1) &&
              irisu_padded_event_size() == sizeof(irisu_padded_event_v1),
          "padded ABI size query drifted");

  require(irisu_padded_step_batch(nullptr, nullptr, nullptr, nullptr, 0, 0) ==
              1,
          "empty padded batch should be a no-op");
  require(irisu_padded_step_batch(nullptr, nullptr, nullptr, nullptr, 1, 1) ==
              0,
          "nonempty padded batch accepted null buffers");

  NativeHandle handle;
  irisu_padded_observation_v1 observation{};
  require(irisu_padded_reset(handle.value, 44, &observation) == 1,
          "padded buffer guard reset failed");
  require(irisu_padded_observation(handle.value, nullptr) == 0,
          "padded observation accepted null destination");
  require(irisu_padded_step(handle.value, IRISU_ACTION_KIND_WAIT, 0.0, 0.0,
                            1, nullptr) == 0,
          "padded step accepted null destination");
  require(irisu_padded_events(handle.value, nullptr, 0) == 1,
          "empty padded event list rejected null destination");

  irisu_padded_transition_v1 transition{};
  require(irisu_padded_step(handle.value, IRISU_ACTION_KIND_WEAK_SHOT, -1.0,
                            250.0, 1, &transition) == 1 &&
              transition.event_count != 0,
          "invalid shot did not produce padded events");
  require(irisu_padded_events(handle.value, nullptr, transition.event_count) ==
              0,
          "nonempty padded event list accepted null destination");
  require(irisu_padded_events(handle.value, nullptr,
                              transition.event_count - 1) == 0,
          "padded event list accepted undersized capacity");
  std::vector<irisu_padded_event_v1> events(transition.event_count);
  require(irisu_padded_events(handle.value, events.data(), events.size()) == 1,
          "exact padded event buffer was rejected");
}

void snapshot_restore_clears_only_successful_step_results() {
  NativeHandle handle;
  irisu_padded_observation_v1 observation{};
  require(irisu_padded_reset(handle.value, 44, &observation) == 1,
          "snapshot result reset failed");
  const auto snapshot_size = irisu_snapshot_size(handle.value);
  require(snapshot_size != 0, "snapshot result size failed");
  std::vector<std::byte> snapshot(snapshot_size);
  require(irisu_snapshot_write(handle.value, snapshot.data(), snapshot.size()) ==
              1,
          "snapshot result write failed");

  irisu_padded_transition_v1 transition{};
  require(irisu_padded_step(handle.value, IRISU_ACTION_KIND_WEAK_SHOT, -1.0,
                            250.0, 1, &transition) == 1 &&
              transition.event_count != 0,
          "snapshot result step did not produce events");
  require(irisu_snapshot_restore(handle.value, snapshot.data(), snapshot.size()) ==
              1,
          "valid snapshot restore failed");
  const char *cleared_json = irisu_step_json(handle.value);
  require(cleared_json != nullptr &&
              std::string_view(cleared_json).find("\"events\":[]") !=
                  std::string_view::npos,
          "successful restore retained stale step JSON");
  require(irisu_padded_events(handle.value, nullptr, 0) == 1,
          "successful restore retained stale padded events");

  require(irisu_padded_step(handle.value, IRISU_ACTION_KIND_WEAK_SHOT, -1.0,
                            250.0, 1, &transition) == 1 &&
              transition.event_count != 0,
          "snapshot failure setup step did not produce events");
  const auto state_before_failure = irisu_state_hash(handle.value);
  const std::string step_before_failure = irisu_step_json(handle.value);
  require(irisu_snapshot_restore(handle.value, snapshot.data(),
                                 snapshot.size() - 1) == 0,
          "truncated snapshot restore unexpectedly succeeded");
  require(irisu_state_hash(handle.value) == state_before_failure,
          "failed snapshot restore mutated simulator state");
  require(irisu_step_json(handle.value) != nullptr &&
              step_before_failure == irisu_step_json(handle.value),
          "failed snapshot restore cleared the previous step JSON");
  std::vector<irisu_padded_event_v1> events(transition.event_count);
  require(irisu_padded_events(handle.value, events.data(), events.size()) == 1,
          "failed snapshot restore cleared the previous padded events");
}

void padded_batch_matches_individual_steps() {
  constexpr std::size_t kLanes = 8;
  std::array<NativeHandle, kLanes> batched;
  std::array<NativeHandle, kLanes> controls;
  std::array<irisu_simulator *, kLanes> handles{};
  std::array<irisu_padded_observation_v1, kLanes> observations{};
  for (std::size_t lane = 0; lane < kLanes; ++lane) {
    handles[lane] = batched[lane].value;
    require(irisu_padded_reset(batched[lane].value, 900 + lane,
                               &observations[lane]) == 1,
            "padded batch reset failed");
    require(irisu_padded_reset(controls[lane].value, 900 + lane,
                               &observations[lane]) == 1,
            "padded control reset failed");
  }

  std::array<irisu_simulator *, 2> duplicate_handles{batched[0].value,
                                                     batched[0].value};
  std::array<irisu_padded_action_v1, 2> duplicate_actions{};
  std::array<irisu_padded_transition_v1, 2> duplicate_results{};
  std::array<std::uint8_t, 2> duplicate_statuses{};
  require(irisu_padded_step_batch(
              duplicate_handles.data(), duplicate_actions.data(),
              duplicate_results.data(), duplicate_statuses.data(),
              duplicate_handles.size(), duplicate_handles.size()) == 0,
          "padded batch must reject duplicate simulator handles");

  for (std::uint32_t round = 0; round < 1'000; ++round) {
    std::array<irisu_padded_action_v1, kLanes> actions{};
    std::array<irisu_padded_transition_v1, kLanes> batch_results{};
    std::array<std::uint8_t, kLanes> statuses{};
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
      actions[lane].kind = round % 7 == 0 ? 1 : 0;
      actions[lane].x = 100.0 + 40.0 * static_cast<double>(lane);
      actions[lane].y = 350.0;
      actions[lane].wait_ticks = 1U + static_cast<std::uint32_t>(lane % 5U);
    }
    require(irisu_padded_step_batch(handles.data(), actions.data(),
                                    batch_results.data(), statuses.data(),
                                    kLanes, kLanes) == 1,
            "padded batch invocation failed");
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
      require(statuses[lane] == 1, "padded batch lane failed");
      irisu_padded_transition_v1 control{};
      require(irisu_padded_step(controls[lane].value, actions[lane].kind,
                                actions[lane].x, actions[lane].y,
                                actions[lane].wait_ticks, &control) == 1,
              "padded control step failed");
      require(irisu_state_hash(batched[lane].value) ==
                  irisu_state_hash(controls[lane].value),
              "padded batch changed deterministic state");
      require(batch_results[lane].observation.tick ==
                      control.observation.tick &&
                  batch_results[lane].observation.body_count ==
                      control.observation.body_count &&
                  batch_results[lane].reward == control.reward &&
                  batch_results[lane].event_count == control.event_count,
              "padded batch transition differs from individual step");
    }
  }
}

} // namespace

int main() {
  try {
    floating_point_metadata_matches_build_capability();
    dirty_pair_manager_storage_is_safe();
    concurrent_independent_worlds_and_handles();
    padded_abi_layout_and_buffer_guards();
    snapshot_restore_clears_only_successful_step_results();
    hostile_rounding_isolated_from_native_simulation();
    hostile_rounding_isolated_from_padded_workers();
    padded_batch_matches_individual_steps();
    repeated_c_abi_lifecycle();
    std::cout << "native physics lifecycle stress passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "physics lifecycle stress failure: " << error.what() << '\n';
    return 1;
  }
}

#if defined(IRISU_TEST_HAS_MXCSR)
#undef IRISU_TEST_HAS_MXCSR
#endif
