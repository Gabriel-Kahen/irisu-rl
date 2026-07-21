#include "irisu/physics.hpp"
#include "irisu/simulator.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr std::uint64_t kExcitationPeriod = 25;

struct Options {
  std::uint64_t ticks{5000};
  std::uint64_t warmup_ticks{200};
  std::uint64_t seed{20260717};
  std::uint64_t policy_seed{0x52574157U};
  std::uint64_t simulator_decisions{};
  std::uint32_t bodies{48};
};

void mix(std::uint64_t& hash, std::uint64_t value) {
  for (int shift = 0; shift < 64; shift += 8) {
    hash ^= (value >> shift) & 0xffU;
    hash *= kFnvPrime;
  }
}

std::uint64_t splitmix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31U);
}

std::uint64_t state_hash(const std::vector<irisu::Body>& bodies) {
  std::uint64_t hash = kFnvOffset;
  for (const auto& body : bodies) {
    mix(hash, body.id);
    mix(hash, static_cast<std::uint64_t>(body.kind));
    mix(hash, static_cast<std::uint64_t>(body.shape));
    mix(hash, static_cast<std::uint64_t>(body.lifecycle));
    mix(hash, std::bit_cast<std::uint64_t>(body.size));
    mix(hash, std::bit_cast<std::uint64_t>(body.density));
    mix(hash, std::bit_cast<std::uint64_t>(body.position.x));
    mix(hash, std::bit_cast<std::uint64_t>(body.position.y));
    mix(hash, std::bit_cast<std::uint64_t>(body.velocity.x));
    mix(hash, std::bit_cast<std::uint64_t>(body.velocity.y));
    mix(hash, std::bit_cast<std::uint64_t>(body.angle));
    mix(hash, std::bit_cast<std::uint64_t>(body.angular_velocity));
    mix(hash, std::bit_cast<std::uint64_t>(body.native_position.x));
    mix(hash, std::bit_cast<std::uint64_t>(body.native_position.y));
    mix(hash, std::bit_cast<std::uint64_t>(body.native_velocity.x));
    mix(hash, std::bit_cast<std::uint64_t>(body.native_velocity.y));
    mix(hash, std::bit_cast<std::uint64_t>(body.native_angle));
    mix(hash, std::bit_cast<std::uint64_t>(body.native_angular_velocity));
    mix(hash, body.native_state_valid ? 1U : 0U);
    mix(hash, body.sleeping ? 1U : 0U);
    mix(hash, std::bit_cast<std::uint64_t>(body.sleep_time));
  }
  return hash;
}

template <typename Integer>
Integer parse_integer(std::string_view text, std::string_view label) {
  Integer value{};
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
    throw std::invalid_argument("invalid " + std::string(label));
  }
  return value;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; index += 2) {
    if (index + 1 >= argc) throw std::invalid_argument("missing option value");
    const std::string_view flag = argv[index];
    const std::string_view value = argv[index + 1];
    if (flag == "--ticks") {
      options.ticks = parse_integer<std::uint64_t>(value, "--ticks");
    } else if (flag == "--warmup") {
      options.warmup_ticks = parse_integer<std::uint64_t>(value, "--warmup");
    } else if (flag == "--seed") {
      options.seed = parse_integer<std::uint64_t>(value, "--seed");
    } else if (flag == "--bodies") {
      options.bodies = parse_integer<std::uint32_t>(value, "--bodies");
    } else if (flag == "--simulator-decisions") {
      options.simulator_decisions =
          parse_integer<std::uint64_t>(value, "--simulator-decisions");
    } else if (flag == "--policy-seed") {
      options.policy_seed = parse_integer<std::uint64_t>(value, "--policy-seed");
    } else {
      throw std::invalid_argument("unknown option: " + std::string(flag));
    }
  }
  if (options.ticks == 0 || options.warmup_ticks == 0) {
    throw std::invalid_argument("tick counts must be positive");
  }
  if (options.ticks > 100'000'000 || options.warmup_ticks > 1'000'000) {
    throw std::invalid_argument("tick count exceeds the bounded benchmark limit");
  }
  if (options.bodies == 0 || options.bodies > 96) {
    throw std::invalid_argument("--bodies must be in [1, 96]");
  }
  if (options.simulator_decisions > 100'000'000) {
    throw std::invalid_argument("--simulator-decisions exceeds bounded limit");
  }
  if (options.simulator_decisions != 0 &&
      options.seed > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("simulator benchmark seed must fit in uint32");
  }
  return options;
}

std::vector<irisu::Body> make_board(const irisu::MechanicsConfig& config,
                                    const Options& options) {
  std::vector<irisu::Body> bodies;
  bodies.reserve(options.bodies);
  for (std::uint32_t index = 0; index < options.bodies; ++index) {
    const std::uint64_t bits = splitmix(options.seed + index);
    irisu::Body body;
    body.id = index + 1;
    body.kind = index % 7 == 0 ? irisu::BodyKind::Projectile : irisu::BodyKind::Piece;
    body.shape = body.kind == irisu::BodyKind::Projectile
                     ? irisu::Shape::Box
                     : static_cast<irisu::Shape>((bits >> 8U) % 3U);
    body.lifecycle = irisu::Lifecycle::DynamicFresh;
    body.color = static_cast<std::int32_t>((bits >> 16U) % 8U);
    const double x = config.field_x + 26.0 + 52.0 * static_cast<double>(index % 8U);
    const double y = config.field_y + config.field_height - 45.0 -
                     48.0 * static_cast<double>(index / 8U);
    body.position = {x, y};
    const auto size_index = static_cast<std::size_t>(
        (bits >> 24U) % config.piece_sizes.size());
    body.size = body.kind == irisu::BodyKind::Projectile
                    ? config.projectile_size
                    : config.piece_sizes[size_index];
    body.density = body.kind == irisu::BodyKind::Projectile ? config.projectile_density
                                                            : config.piece_density;
    body.friction = body.kind == irisu::BodyKind::Projectile
                        ? config.projectile_friction
                        : config.piece_friction;
    body.restitution = body.kind == irisu::BodyKind::Projectile
                           ? config.projectile_restitution
                           : config.piece_restitution;
    body.velocity = {
        static_cast<double>(static_cast<std::int32_t>((bits >> 32U) % 81U) - 40),
        body.kind == irisu::BodyKind::Projectile ? config.weak_projectile_vy : 0.0,
    };
    body.angle = static_cast<double>(static_cast<std::int32_t>((bits >> 40U) % 21U) - 10) *
                 0.01;
    bodies.push_back(body);
  }
  return bodies;
}

struct RunResult {
  std::uint64_t contacts{};
  std::uint64_t excitations{};
  std::uint64_t excitation_hash{kFnvOffset};
};

RunResult run(irisu::PhysicsWorld& world, std::vector<irisu::Body>& bodies,
              std::uint64_t first_tick, std::uint64_t count, std::uint64_t seed) {
  RunResult result;
  for (std::uint64_t offset = 0; offset < count; ++offset) {
    const std::uint64_t tick = first_tick + offset;
    if (tick % kExcitationPeriod == 0) {
      const std::uint64_t bits = splitmix(seed ^ tick);
      auto& body = bodies[static_cast<std::size_t>(bits % bodies.size())];
      body.position.x = 110.0 + static_cast<double>((bits >> 8U) % 380U);
      body.position.y = 360.0;
      body.velocity.x = static_cast<double>(
          static_cast<std::int32_t>((bits >> 24U) % 201U) - 100);
      body.velocity.y = (bits & 1U) == 0 ? -250.0 : -500.0;
      // PhysicsWorld deliberately distinguishes actor-visible velocity from
      // the raw Box2D velocity. Set both so this benchmark actually excites
      // the solver instead of changing only an observation-side mirror.
      body.native_velocity = body.velocity;
      body.sleeping = false;
      body.sleep_time = 0.0;
      mix(result.excitation_hash, tick);
      mix(result.excitation_hash, body.id);
      mix(result.excitation_hash, std::bit_cast<std::uint64_t>(body.position.x));
      mix(result.excitation_hash, std::bit_cast<std::uint64_t>(body.position.y));
      mix(result.excitation_hash, std::bit_cast<std::uint64_t>(body.velocity.x));
      mix(result.excitation_hash, std::bit_cast<std::uint64_t>(body.velocity.y));
      ++result.excitations;
    }
    result.contacts += world.step(bodies).size();
  }
  return result;
}

std::string hex64(std::uint64_t value) {
  std::ostringstream output;
  output << "0x" << std::hex << std::setw(16) << std::setfill('0') << value;
  return output.str();
}

class RandomPolicy {
 public:
  explicit RandomPolicy(std::uint64_t seed) : state_(seed) {}

  void reset(std::uint64_t seed) { state_ = seed; }

  irisu::Action act() {
    if (unit() >= 0.25) {
      static_cast<void>(bounded(1));
      return {irisu::ActionKind::Wait, 0.0, 0.0, 1, false};
    }
    const bool strong = unit() < 0.5;
    const double x = 94.0 + unit() * 420.0;
    const double y = 260.0 + unit() * 120.0;
    return {strong ? irisu::ActionKind::StrongShot
                   : irisu::ActionKind::WeakShot,
            x, y, 1, false};
  }

 private:
  std::uint64_t next() {
    state_ += 0x9e3779b97f4a7c15ULL;
    std::uint64_t value = state_;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
  }

  double unit() {
    return static_cast<double>(next() >> 11U) *
           (1.0 / static_cast<double>(std::uint64_t{1} << 53U));
  }

  std::uint64_t bounded(std::uint64_t bound) {
    const std::uint64_t threshold = (std::uint64_t{0} - bound) % bound;
    for (;;) {
      const auto value = next();
      if (value >= threshold) return value % bound;
    }
  }

  std::uint64_t state_{};
};

void mix_event(std::uint64_t& hash, const irisu::Event& event) {
  mix(hash, event.tick);
  mix(hash, static_cast<std::uint8_t>(event.kind));
  mix(hash, event.a);
  mix(hash, event.b);
  mix(hash, static_cast<std::uint64_t>(event.value));
  mix(hash, event.sequence);
  mix(hash, event.detail.size());
  for (const unsigned char byte : event.detail) mix(hash, byte);
}

int simulator_benchmark(const Options& options) {
  irisu::Simulator simulator;
  RandomPolicy policy(options.policy_seed);
  auto observation = simulator.reset(options.seed);
  std::array<std::uint64_t, 4> action_counts{};
  std::uint64_t action_hash{kFnvOffset};
  std::uint64_t event_hash{kFnvOffset};
  std::uint64_t event_count{};
  std::uint64_t event_capacity{};
  std::uint64_t event_capacity_max{};
  std::uint64_t body_count{};
  std::uint64_t body_min = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t body_max{};
  std::uint64_t event_min = std::numeric_limits<std::uint64_t>::max();
  std::uint64_t event_max{};
  std::uint64_t resets{};
  std::chrono::nanoseconds step_elapsed{};
  std::chrono::nanoseconds observation_elapsed{};
  const auto wall_start = std::chrono::steady_clock::now();
  for (std::uint64_t decision = 0; decision < options.simulator_decisions;
       ++decision) {
    const irisu::Action action = policy.act();
    const auto kind = static_cast<std::uint8_t>(action.kind);
    ++action_counts.at(kind);
    mix(action_hash, kind);
    mix(action_hash, std::bit_cast<std::uint64_t>(action.cursor_x));
    mix(action_hash, std::bit_cast<std::uint64_t>(action.cursor_y));
    mix(action_hash, action.wait_ticks);

    const auto step_start = std::chrono::steady_clock::now();
    const irisu::StepResult transition = simulator.step(action);
    const auto step_end = std::chrono::steady_clock::now();
    observation = simulator.observation();
    const auto observation_end = std::chrono::steady_clock::now();
    step_elapsed += step_end - step_start;
    observation_elapsed += observation_end - step_end;

    const auto events = static_cast<std::uint64_t>(transition.events.size());
    const auto capacity = static_cast<std::uint64_t>(transition.events.capacity());
    event_count += events;
    event_capacity += capacity;
    event_capacity_max = std::max(event_capacity_max, capacity);
    event_min = std::min(event_min, events);
    event_max = std::max(event_max, events);
    const auto bodies = static_cast<std::uint64_t>(observation.bodies.size());
    body_count += bodies;
    body_min = std::min(body_min, bodies);
    body_max = std::max(body_max, bodies);
    for (const auto& event : transition.events) mix_event(event_hash, event);

    if ((transition.terminated || transition.truncated) &&
        decision + 1 < options.simulator_decisions) {
      ++resets;
      const auto seed = static_cast<std::uint32_t>(options.seed + resets);
      observation = simulator.reset(seed);
      policy.reset(options.policy_seed + resets);
      mix(action_hash, seed);
      mix(event_hash, seed);
    }
  }
  const auto wall_end = std::chrono::steady_clock::now();
  const auto step_ns = std::max<std::int64_t>(1, step_elapsed.count());
  const auto observation_ns =
      std::max<std::int64_t>(1, observation_elapsed.count());
  const auto core_ns = step_ns + observation_ns;
  const auto wall_ns = std::max<std::int64_t>(
      1, std::chrono::duration_cast<std::chrono::nanoseconds>(wall_end - wall_start)
             .count());
  const double decisions = static_cast<double>(options.simulator_decisions);
  std::cout << std::setprecision(12)
            << "{\"schema_version\":1"
            << ",\"workload\":\"simulator_random_policy_one_tick_v1\""
            << ",\"seed\":" << options.seed
            << ",\"policy_seed\":" << options.policy_seed
            << ",\"decision_steps\":" << options.simulator_decisions
            << ",\"episode_resets\":" << resets
            << ",\"core_step_elapsed_seconds\":"
            << static_cast<double>(step_ns) / 1'000'000'000.0
            << ",\"observation_elapsed_seconds\":"
            << static_cast<double>(observation_ns) / 1'000'000'000.0
            << ",\"step_plus_observation_per_second\":"
            << decisions * 1'000'000'000.0 / static_cast<double>(core_ns)
            << ",\"wall_decisions_per_second\":"
            << decisions * 1'000'000'000.0 / static_cast<double>(wall_ns)
            << ",\"observed_body_count_min\":" << body_min
            << ",\"observed_body_count_mean\":"
            << static_cast<double>(body_count) / decisions
            << ",\"observed_body_count_max\":" << body_max
            << ",\"event_count_min\":" << event_min
            << ",\"event_count_mean\":"
            << static_cast<double>(event_count) / decisions
            << ",\"event_count_max\":" << event_max
            << ",\"event_vector_capacity_mean\":"
            << static_cast<double>(event_capacity) / decisions
            << ",\"event_vector_capacity_max\":" << event_capacity_max
            << ",\"action_counts\":[" << action_counts[0] << ','
            << action_counts[1] << ',' << action_counts[2] << ','
            << action_counts[3] << ']'
            << ",\"action_trace_hash\":\"" << hex64(action_hash) << "\""
            << ",\"event_trace_hash\":\"" << hex64(event_hash) << "\""
            << ",\"final_tick\":" << observation.tick
            << ",\"final_score\":" << observation.score
            << ",\"final_gauge\":" << observation.gauge << "}\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.simulator_decisions != 0) {
      return simulator_benchmark(options);
    }
    const irisu::MechanicsConfig config;
    irisu::PhysicsWorld world(config);
    auto bodies = make_board(config, options);
    for (auto& body : bodies) world.initialize_mass(body);
#ifdef IRISU_EXACT_FORWARD_BENCHMARK
    // The forward exact adapter intentionally cannot restore physics
    // snapshots. This board starts in a fresh world, so ordinary fixture
    // synchronization is the equivalent initialization path.
    world.synchronize(bodies);
#else
    world.rebuild(bodies);
#endif
    const std::uint64_t initial_hash = state_hash(bodies);
    (void)run(world, bodies, 0, options.warmup_ticks, options.seed);
    const std::uint64_t timed_start_hash = state_hash(bodies);

    const auto start = std::chrono::steady_clock::now();
    const RunResult measured =
        run(world, bodies, options.warmup_ticks, options.ticks, options.seed);
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed_ns = std::max<std::int64_t>(
        1, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    const double seconds = static_cast<double>(elapsed_ns) / 1'000'000'000.0;
    const double ticks_per_second = static_cast<double>(options.ticks) / seconds;
    const std::uint64_t body_ticks = options.ticks * options.bodies;

    std::cout << std::setprecision(12)
              << "{\"schema_version\":2"
              << ",\"workload\":\"legacy_physics_typical_board_v2\""
              << ",\"seed\":" << options.seed
              << ",\"body_count\":" << options.bodies
              << ",\"warmup_ticks\":" << options.warmup_ticks
              << ",\"physics_ticks\":" << options.ticks
              << ",\"solver_iterations\":" << config.solver_iterations
              << ",\"tick_seconds\":" << config.tick_seconds
              << ",\"elapsed_seconds\":" << seconds
              << ",\"physics_ticks_per_second\":" << ticks_per_second
              << ",\"body_ticks\":" << body_ticks
              << ",\"body_ticks_per_second\":"
              << static_cast<double>(body_ticks) / seconds
              << ",\"contacts_observed\":" << measured.contacts
              << ",\"excitation_period_ticks\":" << kExcitationPeriod
              << ",\"excitation_count\":" << measured.excitations
              << ",\"excitation_trace_hash\":\""
              << hex64(measured.excitation_hash) << "\""
              << ",\"initial_state_hash\":\"" << hex64(initial_hash) << "\""
              << ",\"timed_start_state_hash\":\"" << hex64(timed_start_hash) << "\""
              << ",\"final_state_hash\":\"" << hex64(state_hash(bodies)) << "\"}"
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "physics benchmark error: " << error.what() << '\n';
    return 2;
  }
}
