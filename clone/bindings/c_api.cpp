#include "irisu/c_api.h"

#include "irisu/config_io.hpp"
#include "irisu/floating_point.hpp"
#include "irisu/simulator.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#ifndef IRISU_CXX_COMPILER_ID
#define IRISU_CXX_COMPILER_ID "unknown"
#endif
#ifndef IRISU_CXX_COMPILER_VERSION
#define IRISU_CXX_COMPILER_VERSION "unknown"
#endif
#ifndef IRISU_CMAKE_BUILD_TYPE
#define IRISU_CMAKE_BUILD_TYPE "unknown"
#endif
#ifndef IRISU_LEGACY_FP_MODE
#define IRISU_LEGACY_FP_MODE "unknown"
#endif
#ifndef IRISU_FP_ENVIRONMENT_MODE
#define IRISU_FP_ENVIRONMENT_MODE "unknown"
#endif
#ifndef IRISU_PHYSICS_BACKEND
#define IRISU_PHYSICS_BACKEND "unknown"
#endif
#ifndef IRISU_EXACT_LIBRARY_SHA256
#define IRISU_EXACT_LIBRARY_SHA256 "not-applicable"
#endif
#ifndef IRISU_SYSTEM_PROCESSOR
#define IRISU_SYSTEM_PROCESSOR "unknown"
#endif
#ifndef IRISU_POINTER_BITS
#define IRISU_POINTER_BITS 0
#endif

struct irisu_simulator {
  irisu_simulator() : value{} {}

  irisu::Simulator value;
  irisu::StepResult last_step;
  std::string observation_json;
  std::string step_json;
  std::string config_json;
  std::string error;
  std::vector<std::byte> snapshot;
};

namespace {

const char* kind_name(irisu::BodyKind value) {
  switch (value) {
    case irisu::BodyKind::Piece: return "piece";
    case irisu::BodyKind::Projectile: return "projectile";
    case irisu::BodyKind::Bonus: return "bonus";
  }
  return "unknown";
}

const char* shape_name(irisu::Shape value) {
  switch (value) {
    case irisu::Shape::Circle: return "circle";
    case irisu::Shape::Box: return "box";
    case irisu::Shape::Triangle: return "triangle";
  }
  return "unknown";
}

const char* lifecycle_name(irisu::Lifecycle value) {
  switch (value) {
    case irisu::Lifecycle::ScriptedFalling: return "scripted_falling";
    case irisu::Lifecycle::DynamicFresh: return "dynamic_fresh";
    case irisu::Lifecycle::Confirmed: return "confirmed";
    case irisu::Lifecycle::Rotten: return "rotten";
    case irisu::Lifecycle::Deleted: return "deleted";
  }
  return "unknown";
}

const char* event_name(irisu::EventKind value) {
  switch (value) {
    case irisu::EventKind::InvalidAction: return "invalid_action";
    case irisu::EventKind::Spawned: return "spawned";
    case irisu::EventKind::ShotFired: return "shot_fired";
    case irisu::EventKind::Activated: return "activated";
    case irisu::EventKind::Contact: return "contact";
    case irisu::EventKind::Confirmed: return "confirmed";
    case irisu::EventKind::ChainJoined: return "chain_joined";
    case irisu::EventKind::Cleared: return "cleared";
    case irisu::EventKind::Rotten: return "rotten";
    case irisu::EventKind::Ejected: return "ejected";
    case irisu::EventKind::Destroyed: return "destroyed";
    case irisu::EventKind::GaugeChanged: return "gauge_changed";
    case irisu::EventKind::ScoreChanged: return "score_changed";
    case irisu::EventKind::LevelChanged: return "level_changed";
    case irisu::EventKind::GameOver: return "game_over";
    case irisu::EventKind::ProjectileHit: return "projectile_hit";
    case irisu::EventKind::ProjectileContact: return "projectile_contact";
    case irisu::EventKind::HeldInputIgnored: return "held_input_ignored";
    case irisu::EventKind::LevelCompleted: return "level_completed";
  }
  return "unknown";
}

static_assert(IRISU_PADDED_BODY_CAPACITY ==
              irisu::MechanicsConfig::actor_pool_capacity - 4U);
static_assert(IRISU_ACTION_KIND_WAIT == static_cast<int>(irisu::ActionKind::Wait));
static_assert(IRISU_ACTION_KIND_WEAK_SHOT ==
              static_cast<int>(irisu::ActionKind::WeakShot));
static_assert(IRISU_ACTION_KIND_STRONG_SHOT ==
              static_cast<int>(irisu::ActionKind::StrongShot));
static_assert(IRISU_ACTION_KIND_BOTH_SHOTS ==
              static_cast<int>(irisu::ActionKind::BothShots));
static_assert(IRISU_BODY_KIND_PIECE == static_cast<int>(irisu::BodyKind::Piece));
static_assert(IRISU_BODY_KIND_PROJECTILE ==
              static_cast<int>(irisu::BodyKind::Projectile));
static_assert(IRISU_BODY_KIND_BONUS == static_cast<int>(irisu::BodyKind::Bonus));
static_assert(IRISU_SHAPE_CIRCLE == static_cast<int>(irisu::Shape::Circle));
static_assert(IRISU_SHAPE_BOX == static_cast<int>(irisu::Shape::Box));
static_assert(IRISU_SHAPE_TRIANGLE == static_cast<int>(irisu::Shape::Triangle));
static_assert(IRISU_LIFECYCLE_SCRIPTED_FALLING ==
              static_cast<int>(irisu::Lifecycle::ScriptedFalling));
static_assert(IRISU_LIFECYCLE_DYNAMIC_FRESH ==
              static_cast<int>(irisu::Lifecycle::DynamicFresh));
static_assert(IRISU_LIFECYCLE_CONFIRMED ==
              static_cast<int>(irisu::Lifecycle::Confirmed));
static_assert(IRISU_LIFECYCLE_ROTTEN ==
              static_cast<int>(irisu::Lifecycle::Rotten));
static_assert(IRISU_LIFECYCLE_DELETED ==
              static_cast<int>(irisu::Lifecycle::Deleted));
static_assert(IRISU_EVENT_KIND_INVALID_ACTION ==
              static_cast<int>(irisu::EventKind::InvalidAction));
static_assert(IRISU_EVENT_KIND_SPAWNED ==
              static_cast<int>(irisu::EventKind::Spawned));
static_assert(IRISU_EVENT_KIND_SHOT_FIRED ==
              static_cast<int>(irisu::EventKind::ShotFired));
static_assert(IRISU_EVENT_KIND_ACTIVATED ==
              static_cast<int>(irisu::EventKind::Activated));
static_assert(IRISU_EVENT_KIND_CONTACT ==
              static_cast<int>(irisu::EventKind::Contact));
static_assert(IRISU_EVENT_KIND_CONFIRMED ==
              static_cast<int>(irisu::EventKind::Confirmed));
static_assert(IRISU_EVENT_KIND_CHAIN_JOINED ==
              static_cast<int>(irisu::EventKind::ChainJoined));
static_assert(IRISU_EVENT_KIND_CLEARED ==
              static_cast<int>(irisu::EventKind::Cleared));
static_assert(IRISU_EVENT_KIND_ROTTEN ==
              static_cast<int>(irisu::EventKind::Rotten));
static_assert(IRISU_EVENT_KIND_EJECTED ==
              static_cast<int>(irisu::EventKind::Ejected));
static_assert(IRISU_EVENT_KIND_DESTROYED ==
              static_cast<int>(irisu::EventKind::Destroyed));
static_assert(IRISU_EVENT_KIND_GAUGE_CHANGED ==
              static_cast<int>(irisu::EventKind::GaugeChanged));
static_assert(IRISU_EVENT_KIND_SCORE_CHANGED ==
              static_cast<int>(irisu::EventKind::ScoreChanged));
static_assert(IRISU_EVENT_KIND_LEVEL_CHANGED ==
              static_cast<int>(irisu::EventKind::LevelChanged));
static_assert(IRISU_EVENT_KIND_GAME_OVER ==
              static_cast<int>(irisu::EventKind::GameOver));
static_assert(IRISU_EVENT_KIND_PROJECTILE_HIT ==
              static_cast<int>(irisu::EventKind::ProjectileHit));
static_assert(IRISU_EVENT_KIND_PROJECTILE_CONTACT ==
              static_cast<int>(irisu::EventKind::ProjectileContact));
static_assert(IRISU_EVENT_KIND_HELD_INPUT_IGNORED ==
              static_cast<int>(irisu::EventKind::HeldInputIgnored));
static_assert(IRISU_EVENT_KIND_LEVEL_COMPLETED ==
              static_cast<int>(irisu::EventKind::LevelCompleted));
static_assert(std::is_standard_layout_v<irisu_padded_body_v1> &&
              std::is_trivially_copyable_v<irisu_padded_body_v1>);
static_assert(std::is_standard_layout_v<irisu_padded_observation_v1> &&
              std::is_trivially_copyable_v<irisu_padded_observation_v1>);
static_assert(std::is_standard_layout_v<irisu_padded_transition_v1> &&
              std::is_trivially_copyable_v<irisu_padded_transition_v1>);
static_assert(std::is_standard_layout_v<irisu_padded_action_v1> &&
              std::is_trivially_copyable_v<irisu_padded_action_v1>);
static_assert(std::is_standard_layout_v<irisu_padded_event_v1> &&
              std::is_trivially_copyable_v<irisu_padded_event_v1>);
#if INTPTR_MAX == INT64_MAX
static_assert(sizeof(irisu_padded_body_v1) == 104);
static_assert(sizeof(irisu_padded_observation_v1) == 20'496);
static_assert(sizeof(irisu_padded_transition_v1) == 20'584);
static_assert(sizeof(irisu_padded_action_v1) == 24);
static_assert(sizeof(irisu_padded_event_v1) == 136);
static_assert(offsetof(irisu_padded_body_v1, age_ticks) == 0);
static_assert(offsetof(irisu_padded_body_v1, x) == 24);
static_assert(offsetof(irisu_padded_body_v1, id) == 80);
static_assert(offsetof(irisu_padded_body_v1, kind) == 96);
static_assert(offsetof(irisu_padded_observation_v1, bodies) == 112);
static_assert(offsetof(irisu_padded_transition_v1, reward) == 20'496);
static_assert(offsetof(irisu_padded_transition_v1, terminated) == 20'576);
static_assert(offsetof(irisu_padded_action_v1, kind) == 20);
static_assert(offsetof(irisu_padded_event_v1, detail) == 36);
#endif

void write_observation(const irisu::Observation& source,
                       irisu_padded_observation_v1& destination) {
  if (source.bodies.size() > IRISU_PADDED_BODY_CAPACITY) {
    throw std::overflow_error("policy observation exceeds padded body capacity");
  }
  destination.tick = source.tick;
  destination.score = source.score;
  destination.gauge = source.gauge;
  destination.gauge_max = source.gauge_max;
  destination.qualifying_clear_count = source.qualifying_clear_count;
  destination.field_x = source.field_x;
  destination.field_y = source.field_y;
  destination.field_width = source.field_width;
  destination.field_height = source.field_height;
  destination.side_wall_top = source.side_wall_top;
  destination.side_wall_bottom = source.side_wall_bottom;
  destination.level = source.level;
  destination.active_colors = source.active_colors;
  destination.spawn_interval_ticks = source.current_spawn_interval_ticks;
  destination.highest_chain = source.highest_chain;
  destination.body_count = static_cast<std::uint32_t>(source.bodies.size());
  destination.terminated = source.terminated ? 1U : 0U;
  destination.truncated = source.truncated ? 1U : 0U;
  destination.left_held = source.left_held ? 1U : 0U;
  destination.right_held = source.right_held ? 1U : 0U;
  for (std::size_t index = 0; index < source.bodies.size(); ++index) {
    const auto& body = source.bodies[index];
    auto& output = destination.bodies[index];
    output.age_ticks = body.age_ticks;
    output.remaining_lifetime = body.remaining_lifetime;
    output.rot_timer = body.rot_timer;
    output.x = body.position.x;
    output.y = body.position.y;
    output.vx = body.velocity.x;
    output.vy = body.velocity.y;
    output.angle = body.angle;
    output.angular_velocity = body.angular_velocity;
    output.size = body.size;
    output.id = body.id;
    output.color = body.color;
    output.chain_id = body.chain_id;
    output.projectile_hits = body.projectile_hits;
    output.kind = static_cast<std::uint8_t>(body.kind);
    output.shape = static_cast<std::uint8_t>(body.shape);
    output.lifecycle = static_cast<std::uint8_t>(body.lifecycle);
    output.reserved = 0;
  }
}

void write_transition(const irisu::Simulator& simulator,
                      const irisu::StepResult& source,
                      irisu_padded_transition_v1& destination) {
  write_observation(simulator.observation(), destination.observation);
  destination.reward = source.reward;
  destination.event_count = source.events.size();
  const auto& diagnostics = source.diagnostics;
  destination.config_hash = diagnostics.config_hash;
  destination.finish_call_count = diagnostics.finish_call_count;
  destination.recorded_final_score = diagnostics.recorded_final_score;
  destination.recorded_final_clears = diagnostics.recorded_final_clears;
  destination.latest_final_score = diagnostics.latest_final_score;
  destination.latest_final_clears = diagnostics.latest_final_clears;
  destination.recorded_final_highest_chain =
      diagnostics.recorded_final_highest_chain;
  destination.recorded_final_level = diagnostics.recorded_final_level;
  destination.latest_final_highest_chain =
      diagnostics.latest_final_highest_chain;
  destination.latest_final_level = diagnostics.latest_final_level;
  destination.terminated = source.terminated ? 1U : 0U;
  destination.truncated = source.truncated ? 1U : 0U;
  destination.terminal_metadata_recorded =
      diagnostics.terminal_metadata_recorded ? 1U : 0U;
  destination.invalid_action = std::any_of(
      source.events.begin(), source.events.end(), [](const irisu::Event& event) {
        return event.kind == irisu::EventKind::InvalidAction;
      }) ? 1U : 0U;
}

void quoted(std::ostream& output, const std::string& value) {
  output << '"';
  for (const char character : value) {
    const auto byte = static_cast<unsigned char>(character);
    switch (byte) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (byte < 0x20) {
          char buffer[7];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", byte);
          output << buffer;
        } else {
          output << static_cast<char>(byte);
        }
    }
  }
  output << '"';
}

std::string observation_json(const irisu::Simulator& simulator) {
  const auto observation = simulator.observation();
  std::ostringstream output;
  output.precision(17);
  output << "{\"tick\":" << observation.tick << ",\"score\":" << observation.score
         << ",\"gauge\":" << observation.gauge << ",\"level\":" << observation.level
         << ",\"terminated\":" << (observation.terminated ? "true" : "false")
         << ",\"truncated\":" << (observation.truncated ? "true" : "false")
         << ",\"left_held\":" << (observation.left_held ? "true" : "false")
         << ",\"right_held\":" << (observation.right_held ? "true" : "false")
         << ",\"highest_chain\":" << observation.highest_chain
         << ",\"qualifying_clear_count\":"
         << observation.qualifying_clear_count
         << ",\"field\":{\"x\":" << observation.field_x << ",\"y\":"
         << observation.field_y << ",\"width\":" << observation.field_width
         << ",\"height\":" << observation.field_height << ",\"side_wall_top\":"
         << observation.side_wall_top << ",\"side_wall_bottom\":"
         << observation.side_wall_bottom << "}"
         << ",\"gauge_max\":" << observation.gauge_max
         << ",\"difficulty\":{\"active_colors\":" << observation.active_colors
         << ",\"spawn_interval_ticks\":" << observation.current_spawn_interval_ticks
         << "},\"bodies\":[";
  bool first = true;
  for (const auto& body : observation.bodies) {
    if (!first) output << ',';
    first = false;
    output << "{\"id\":" << body.id << ",\"kind\":\"" << kind_name(body.kind)
           << "\",\"shape\":\"" << shape_name(body.shape) << "\",\"lifecycle\":\""
           << lifecycle_name(body.lifecycle) << "\",\"color\":" << body.color
           << ",\"x\":" << body.position.x << ",\"y\":" << body.position.y
           << ",\"vx\":" << body.velocity.x << ",\"vy\":" << body.velocity.y
           << ",\"angle\":" << body.angle << ",\"angular_velocity\":" << body.angular_velocity
           << ",\"size\":" << body.size << ",\"chain_id\":" << body.chain_id
           << ",\"projectile_hits\":" << body.projectile_hits << ",\"age_ticks\":"
           << body.age_ticks << ",\"remaining_lifetime\":"
           << body.remaining_lifetime << ",\"rot_timer\":" << body.rot_timer << '}';
  }
  output << "]}";
  return output.str();
}

std::string step_json(const irisu::StepResult& result) {
  std::ostringstream output;
  output << "{\"reward\":" << result.reward << ",\"terminated\":"
         << (result.terminated ? "true" : "false") << ",\"truncated\":"
         << (result.truncated ? "true" : "false") << ",\"events\":[";
  bool first = true;
  for (const auto& event : result.events) {
    if (!first) output << ',';
    first = false;
    output << "{\"tick\":" << event.tick << ",\"sequence\":" << event.sequence
           << ",\"kind\":" << static_cast<int>(event.kind)
           << ",\"kind_name\":\"" << event_name(event.kind) << "\""
           << ",\"a\":" << event.a << ",\"b\":" << event.b << ",\"value\":"
           << event.value << ",\"detail\":";
    quoted(output, event.detail);
    output << '}';
  }
  const auto& diagnostics = result.diagnostics;
  output << "],\"diagnostics\":{\"config_hash\":" << diagnostics.config_hash
         << ",\"finish_call_count\":" << diagnostics.finish_call_count
         << ",\"terminal_metadata_recorded\":"
         << (diagnostics.terminal_metadata_recorded ? "true" : "false")
         << ",\"recorded_final_score\":" << diagnostics.recorded_final_score
         << ",\"recorded_final_highest_chain\":"
         << diagnostics.recorded_final_highest_chain
         << ",\"recorded_final_level\":" << diagnostics.recorded_final_level
         << ",\"recorded_final_clears\":" << diagnostics.recorded_final_clears
         << ",\"latest_final_score\":" << diagnostics.latest_final_score
         << ",\"latest_final_highest_chain\":"
         << diagnostics.latest_final_highest_chain
         << ",\"latest_final_level\":" << diagnostics.latest_final_level
         << ",\"latest_final_clears\":" << diagnostics.latest_final_clears
         << "}}";
  return output.str();
}

void padded_step_impl(irisu_simulator& simulator, int action_kind, double x,
                      double y, uint32_t wait_ticks,
                      irisu_padded_transition_v1& destination) {
  if (action_kind < 0 || action_kind > 3) {
    throw std::invalid_argument("unknown action kind");
  }
  simulator.last_step = simulator.value.step(
      {static_cast<irisu::ActionKind>(action_kind), x, y, wait_ticks});
  write_transition(simulator.value, simulator.last_step, destination);
}

template <typename Function>
int protect(irisu_simulator* simulator, Function&& function) {
  if (simulator == nullptr) return 0;
  const irisu::ScopedFloatingPointEnvironment floating_point_environment;
  try {
    function();
    simulator->error.clear();
    return 1;
  } catch (const std::exception& exception) {
    simulator->error = exception.what();
    return 0;
  } catch (...) {
    simulator->error = "unknown native exception";
    return 0;
  }
}

struct PaddedBatchRequest {
  irisu_simulator* const* simulators{};
  const irisu_padded_action_v1* actions{};
  irisu_padded_transition_v1* destinations{};
  std::uint8_t* statuses{};
  std::size_t count{};
  std::size_t worker_limit{};
  std::atomic<std::size_t> next{};
  std::atomic<std::size_t> remaining{};
  std::atomic<std::size_t> consumers{};
  std::condition_variable* completion{};
  std::mutex* completion_mutex{};
};

class PaddedBatchPool {
 public:
  PaddedBatchPool() {
    try {
      const auto available = std::max(1U, std::thread::hardware_concurrency());
      // Eight single-threaded worlds match the physical cores of the supported
      // reference host; waking SMT siblings adds barrier cost without useful
      // solver throughput on this legacy float-heavy workload.
      const auto worker_count = std::min(7U, available - 1U);
      workers_.reserve(worker_count);
      for (unsigned int index = 0; index < worker_count; ++index) {
        workers_.emplace_back([this, index] { worker(index); });
      }
    } catch (...) {
      stop_and_join();
      throw;
    }
  }

  ~PaddedBatchPool() { stop_and_join(); }

  void stop_and_join() noexcept {
    {
      const std::lock_guard lock(state_mutex_);
      stopping_ = true;
    }
    start_.notify_all();
    for (auto& worker : workers_) {
      if (worker.joinable()) worker.join();
    }
  }

  void run(PaddedBatchRequest& request, std::size_t worker_count) {
    if (request.count == 0) return;
    const std::lock_guard serial(serial_mutex_);
    request.next = 0;
    request.remaining = request.count;
    request.worker_limit = std::min(
        workers_.size(), std::min(request.count - 1U, worker_count - 1U));
    request.consumers = workers_.size() + 1U;
    request.completion = &completion_;
    request.completion_mutex = &state_mutex_;
    {
      const std::lock_guard lock(state_mutex_);
      request_ = &request;
      ++generation_;
    }
    start_.notify_all();
    consume(request);
    std::unique_lock lock(state_mutex_);
    completion_.wait(lock, [&] {
      return request.remaining.load() == 0 && request.consumers.load() == 0;
    });
    request_ = nullptr;
  }

 private:
  static void consume(PaddedBatchRequest& request) {
    const irisu::ScopedFloatingPointEnvironment floating_point_environment;
    for (;;) {
      const std::size_t index = request.next.fetch_add(1);
      if (index >= request.count) break;
      irisu_simulator* simulator = request.simulators[index];
      const auto& action = request.actions[index];
      request.statuses[index] = static_cast<std::uint8_t>(protect(
          simulator, [&] {
            padded_step_impl(*simulator, action.kind, action.x, action.y,
                             action.wait_ticks, request.destinations[index]);
          }));
      request.remaining.fetch_sub(1);
    }
    std::condition_variable* completion = request.completion;
    std::mutex* completion_mutex = request.completion_mutex;
    if (request.consumers.fetch_sub(1) == 1) {
      const std::lock_guard lock(*completion_mutex);
      completion->notify_one();
    }
  }

  void worker(std::size_t worker_index) {
    const irisu::ScopedFloatingPointEnvironment floating_point_environment;
    std::uint64_t observed_generation = 0;
    for (;;) {
      PaddedBatchRequest* request = nullptr;
      {
        std::unique_lock lock(state_mutex_);
        start_.wait(lock, [&] {
          return stopping_ || generation_ != observed_generation;
        });
        if (stopping_) return;
        observed_generation = generation_;
        request = request_;
      }
      if (worker_index < request->worker_limit) {
        consume(*request);
      } else {
        std::condition_variable* completion = request->completion;
        std::mutex* completion_mutex = request->completion_mutex;
        if (request->consumers.fetch_sub(1) == 1) {
          const std::lock_guard lock(*completion_mutex);
          completion->notify_one();
        }
      }
    }
  }

  std::mutex serial_mutex_;
  std::mutex state_mutex_;
  std::condition_variable start_;
  std::condition_variable completion_;
  std::vector<std::thread> workers_;
  PaddedBatchRequest* request_{};
  std::uint64_t generation_{};
  bool stopping_{};
};

PaddedBatchPool& padded_batch_pool() {
  static PaddedBatchPool pool;
  return pool;
}

}  // namespace

extern "C" {

uint32_t irisu_abi_version(void) { return 1; }

uint32_t irisu_padded_abi_version(void) { return 1; }

size_t irisu_padded_body_capacity(void) {
  return IRISU_PADDED_BODY_CAPACITY;
}

size_t irisu_padded_observation_size(void) {
  return sizeof(irisu_padded_observation_v1);
}

size_t irisu_padded_transition_size(void) {
  return sizeof(irisu_padded_transition_v1);
}

size_t irisu_padded_action_size(void) {
  return sizeof(irisu_padded_action_v1);
}

size_t irisu_padded_event_size(void) {
  return sizeof(irisu_padded_event_v1);
}

irisu_simulator* irisu_create(void) {
  const irisu::ScopedFloatingPointEnvironment floating_point_environment;
  try { return new irisu_simulator{}; } catch (...) { return nullptr; }
}

void irisu_destroy(irisu_simulator* simulator) {
  const irisu::ScopedFloatingPointEnvironment floating_point_environment;
  delete simulator;
}

int irisu_configure(irisu_simulator* simulator, const irisu_config_override* overrides,
                    size_t override_count) {
  return protect(simulator, [&] {
    if (overrides == nullptr && override_count != 0) {
      throw std::invalid_argument("null configuration override array");
    }
    irisu::MechanicsConfig config;
    for (std::size_t index = 0; index < override_count; ++index) {
      if (overrides[index].key == nullptr) {
        throw std::invalid_argument("null configuration override key");
      }
      irisu::apply_config_override(config, overrides[index].key,
                                   overrides[index].value);
    }
    simulator->value = irisu::Simulator(config);
    simulator->last_step = {};
    simulator->snapshot.clear();
  });
}

int irisu_reset(irisu_simulator* simulator, uint64_t seed) {
  return protect(simulator, [&] { simulator->value.reset(seed); simulator->last_step = {}; });
}

int irisu_step(irisu_simulator* simulator, int action_kind, double x, double y,
               uint32_t wait_ticks) {
  return protect(simulator, [&] {
    if (action_kind < 0 || action_kind > 3) throw std::invalid_argument("unknown action kind");
    simulator->last_step = simulator->value.step(
        {static_cast<irisu::ActionKind>(action_kind), x, y, wait_ticks});
  });
}

int irisu_padded_observation(irisu_simulator* simulator,
                             irisu_padded_observation_v1* destination) {
  return protect(simulator, [&] {
    if (destination == nullptr) {
      throw std::invalid_argument("null padded observation destination");
    }
    write_observation(simulator->value.observation(), *destination);
  });
}

int irisu_padded_reset(irisu_simulator* simulator, uint64_t seed,
                       irisu_padded_observation_v1* destination) {
  return protect(simulator, [&] {
    if (destination == nullptr) {
      throw std::invalid_argument("null padded observation destination");
    }
    simulator->value.reset(seed);
    simulator->last_step = {};
    write_observation(simulator->value.observation(), *destination);
  });
}

int irisu_padded_step(irisu_simulator* simulator, int action_kind, double x,
                      double y, uint32_t wait_ticks,
                      irisu_padded_transition_v1* destination) {
  return protect(simulator, [&] {
    if (destination == nullptr) {
      throw std::invalid_argument("null padded transition destination");
    }
    padded_step_impl(*simulator, action_kind, x, y, wait_ticks, *destination);
  });
}

int irisu_padded_step_batch(irisu_simulator* const* simulators,
                            const irisu_padded_action_v1* actions,
                            irisu_padded_transition_v1* destinations,
                            uint8_t* statuses, size_t simulator_count,
                            size_t worker_count) {
  const irisu::ScopedFloatingPointEnvironment floating_point_environment;
  try {
    if (simulator_count == 0) return 1;
    if (worker_count == 0) return 0;
    if (simulators == nullptr || actions == nullptr || destinations == nullptr ||
        statuses == nullptr) {
      return 0;
    }
    for (std::size_t index = 0; index < simulator_count; ++index) {
      if (simulators[index] == nullptr) return 0;
      for (std::size_t previous = 0; previous < index; ++previous) {
        if (simulators[index] == simulators[previous]) return 0;
      }
    }
    PaddedBatchRequest request{simulators, actions, destinations, statuses,
                               simulator_count};
    padded_batch_pool().run(request, worker_count);
    return 1;
  } catch (...) {
    return 0;
  }
}

int irisu_padded_events(irisu_simulator* simulator,
                        irisu_padded_event_v1* destination,
                        size_t event_capacity) {
  return protect(simulator, [&] {
    const auto& events = simulator->last_step.events;
    if (event_capacity != events.size()) {
      throw std::invalid_argument("padded event buffer size mismatch");
    }
    if (events.empty()) return;
    if (destination == nullptr) {
      throw std::invalid_argument("null padded event destination");
    }
    for (const auto& event : events) {
      if (event.detail.size() >= IRISU_EVENT_DETAIL_CAPACITY) {
        throw std::overflow_error("event detail exceeds padded capacity");
      }
    }
    for (std::size_t index = 0; index < events.size(); ++index) {
      const auto& event = events[index];
      auto& output = destination[index];
      output.tick = event.tick;
      output.sequence = event.sequence;
      output.value = event.value;
      output.a = event.a;
      output.b = event.b;
      output.detail_size = static_cast<std::uint16_t>(event.detail.size());
      output.kind = static_cast<std::uint8_t>(event.kind);
      output.reserved = 0;
      std::memcpy(output.detail, event.detail.data(), event.detail.size());
      output.detail[event.detail.size()] = '\0';
    }
  });
}

const char* irisu_observation_json(irisu_simulator* simulator) {
  if (!protect(simulator, [&] { simulator->observation_json = observation_json(simulator->value); })) return nullptr;
  return simulator->observation_json.c_str();
}

const char* irisu_step_json(irisu_simulator* simulator) {
  if (!protect(simulator, [&] { simulator->step_json = step_json(simulator->last_step); })) return nullptr;
  return simulator->step_json.c_str();
}

uint64_t irisu_state_hash(const irisu_simulator* simulator) {
  if (simulator == nullptr) return 0;
  const irisu::ScopedFloatingPointEnvironment floating_point_environment;
  try { return simulator->value.state_hash(); } catch (...) { return 0; }
}

uint64_t irisu_config_hash(const irisu_simulator* simulator) {
  if (simulator == nullptr) return 0;
  const irisu::ScopedFloatingPointEnvironment floating_point_environment;
  try { return simulator->value.config_hash(); } catch (...) { return 0; }
}

const char* irisu_config_json(irisu_simulator* simulator) {
  if (!protect(simulator, [&] {
        simulator->config_json = irisu::mechanics_config_json(
            simulator->value.config(), simulator->value.config_hash());
      })) {
    return nullptr;
  }
  return simulator->config_json.c_str();
}

const char* irisu_build_info_json(void) {
  try {
    static const std::string information = [] {
      std::ostringstream output;
      output << "{\"abi_version\":1,\"clone_version\":\"0.1.0\","
                "\"target\":\"IriSu Syndrome v2.03 normal\","
                "\"physics\":\"Box2D 1.4.3 SourceForge SVN r58\","
                "\"snapshot_schema\":7,\"padded_abi_version\":1,"
                "\"cxx_compiler_id\":";
      quoted(output, IRISU_CXX_COMPILER_ID);
      output << ",\"cxx_compiler_version\":";
      quoted(output, IRISU_CXX_COMPILER_VERSION);
      output << ",\"cmake_build_type\":";
      quoted(output, IRISU_CMAKE_BUILD_TYPE);
      output << ",\"legacy_fp_mode\":";
      quoted(output, IRISU_LEGACY_FP_MODE);
      output << ",\"fp_environment\":";
      quoted(output, IRISU_FP_ENVIRONMENT_MODE);
      output << ",\"physics_backend\":";
      quoted(output, IRISU_PHYSICS_BACKEND);
      output << ",\"exact_library_sha256\":";
      quoted(output, IRISU_EXACT_LIBRARY_SHA256);
      output << ",\"system_processor\":";
      quoted(output, IRISU_SYSTEM_PROCESSOR);
      output << ",\"pointer_bits\":" << IRISU_POINTER_BITS;
      output << ",\"seed_bits\":32";
      output << ",\"determinism_scope\":\"same supported build\"}";
      return output.str();
    }();
    return information.c_str();
  } catch (...) {
    return nullptr;
  }
}

size_t irisu_snapshot_size(irisu_simulator* simulator) {
  if (!protect(simulator, [&] { simulator->snapshot = simulator->value.serialize_snapshot(); })) return 0;
  return simulator->snapshot.size();
}

int irisu_snapshot_write(irisu_simulator* simulator, void* destination, size_t size) {
  return protect(simulator, [&] {
    simulator->snapshot = simulator->value.serialize_snapshot();
    if (destination == nullptr || size != simulator->snapshot.size()) throw std::invalid_argument("snapshot buffer size mismatch");
    std::memcpy(destination, simulator->snapshot.data(), size);
  });
}

int irisu_snapshot_restore(irisu_simulator* simulator, const void* source, size_t size) {
  return protect(simulator, [&] {
    if (source == nullptr && size != 0) throw std::invalid_argument("null snapshot source");
    simulator->value.restore_snapshot({static_cast<const std::byte*>(source), size});
    simulator->last_step = {};
  });
}

const char* irisu_last_error(const irisu_simulator* simulator) {
  return simulator == nullptr ? "null simulator" : simulator->error.c_str();
}

}  // extern "C"
