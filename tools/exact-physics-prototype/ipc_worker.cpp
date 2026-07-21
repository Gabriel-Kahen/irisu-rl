#include "irisu/config_io.hpp"
#include "irisu/floating_point.hpp"
#include "irisu/simulator.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <map>
#include <poll.h>
#include <set>
#include <signal.h>
#include <stdexcept>
#include <string>
#include <sys/random.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

extern "C" void irisu_exact_physics_attestation(
    std::uint32_t* entrypoint_count, const char** device,
    std::uint64_t* inode);

namespace {

constexpr std::uint32_t kMagic = 0x43505249U;  // "IRPC" on the wire.
constexpr std::uint16_t kVersion = 1;
constexpr std::size_t kHeaderBytes = 16;
constexpr std::size_t kMaximumPayloadBytes = 1U << 20U;
constexpr std::size_t kMaximumResponsePayloadBytes = 4U << 20U;
constexpr std::size_t kMaximumResponseContentBytes =
    kMaximumResponsePayloadBytes - sizeof(std::int32_t);
constexpr std::size_t kFastTokenBytes = 16U;
constexpr std::size_t kMaximumBranchAddressBytes =
    sizeof(sockaddr_un{}.sun_path);
constexpr std::uint32_t kBodyCapacity =
    irisu::MechanicsConfig::actor_pool_capacity - 4U;
using FastToken = std::array<std::byte, kFastTokenBytes>;

enum class Opcode : std::uint16_t {
  Hello = 1,
  Reset = 2,
  Step = 3,
  Observe = 4,
  Close = 5,
  Configure = 6,
  ConfigJson = 7,
  StepPadded = 8,
  FetchEvents = 9,
  FastCheckpoint = 10,
  FastRelease = 11,
  FastBranch = 12,
  ExactAttestation = 13,
};

enum class Status : std::int32_t {
  Ok = 0,
  BadRequest = 1,
  InternalError = 2,
};

enum class KeeperCommand : std::uint8_t {
  Branch = 1,
  Release = 2,
  Shutdown = 3,
};

enum class KeeperStatus : std::uint8_t {
  Ok = 0,
  Busy = 1,
  Error = 2,
};

template <typename T>
void append(std::vector<std::byte>& destination, T value) {
  static_assert(std::is_trivially_copyable_v<T>);
  const auto bytes = std::bit_cast<std::array<std::byte, sizeof(T)>>(value);
  destination.insert(destination.end(), bytes.begin(), bytes.end());
}

template <typename T>
T read_value(const std::vector<std::byte>& source, std::size_t& offset) {
  static_assert(std::is_trivially_copyable_v<T>);
  if (offset > source.size() || source.size() - offset < sizeof(T)) {
    throw std::invalid_argument("truncated request payload");
  }
  std::array<std::byte, sizeof(T)> bytes{};
  std::copy_n(source.begin() + static_cast<std::ptrdiff_t>(offset),
              sizeof(T), bytes.begin());
  offset += sizeof(T);
  return std::bit_cast<T>(bytes);
}

bool read_exact(int descriptor, void* destination, std::size_t size,
                bool eof_is_clean) {
  auto* output = static_cast<std::byte*>(destination);
  std::size_t offset = 0;
  while (offset < size) {
    const ssize_t count = ::read(descriptor, output + offset, size - offset);
    if (count == 0) {
      if (eof_is_clean && offset == 0) return false;
      throw std::runtime_error("unexpected EOF on request stream");
    }
    if (count < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("request stream read failed");
    }
    offset += static_cast<std::size_t>(count);
  }
  return true;
}

void write_exact(int descriptor, const void* source, std::size_t size) {
  const auto* input = static_cast<const std::byte*>(source);
  std::size_t offset = 0;
  while (offset < size) {
    const ssize_t count = ::write(descriptor, input + offset, size - offset);
    if (count == 0) throw std::runtime_error("stream write made no progress");
    if (count < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("response stream write failed");
    }
    offset += static_cast<std::size_t>(count);
  }
}

void close_descriptor(int& descriptor) noexcept {
  if (descriptor < 0) return;
  while (::close(descriptor) != 0 && errno == EINTR) {
  }
  descriptor = -1;
}

FastToken random_token() {
  FastToken token{};
  std::size_t offset{};
  while (offset != token.size()) {
    const ssize_t count =
        ::getrandom(token.data() + offset, token.size() - offset, 0);
    if (count < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("getrandom failed while creating fast token");
    }
    if (count == 0) {
      throw std::runtime_error("getrandom returned no fast-token bytes");
    }
    offset += static_cast<std::size_t>(count);
  }
  return token;
}

bool tokens_equal(const FastToken& first, const FastToken& second) noexcept {
  unsigned int difference{};
  for (std::size_t index = 0; index < first.size(); ++index) {
    difference |= std::to_integer<unsigned int>(first[index] ^ second[index]);
  }
  return difference == 0;
}

std::uint16_t requested_control_word() {
  const char* text = std::getenv("IRISU_EXACT_CW");
  if (text == nullptr) return 0x027fU;
  char* end{};
  errno = 0;
  const unsigned long value = std::strtoul(text, &end, 0);
  if (errno != 0 || end == text || *end != '\0' ||
      value > std::numeric_limits<std::uint16_t>::max()) {
    throw std::invalid_argument("IRISU_EXACT_CW must fit in uint16");
  }
  return static_cast<std::uint16_t>(value);
}

#ifndef IRISU_EXACT_LIBRARY_SHA256
#define IRISU_EXACT_LIBRARY_SHA256 "unknown"
#endif
#ifndef IRISU_PHYSICS_BACKEND
#define IRISU_PHYSICS_BACKEND "unknown"
#endif

constexpr const char* kBackendName = IRISU_PHYSICS_BACKEND;
constexpr std::uint32_t kExactAttestationSchema = 1;
constexpr std::uint32_t kExactEntrypointCount = 15;

struct ExactPhysicsAttestation {
  std::uint32_t entrypoint_count{};
  std::string device;
  std::uint64_t inode{};
};

ExactPhysicsAttestation capture_exact_physics_attestation() {
  ExactPhysicsAttestation result;
  const char* device{};
  irisu_exact_physics_attestation(&result.entrypoint_count, &device,
                                  &result.inode);
  if (result.entrypoint_count != kExactEntrypointCount || device == nullptr ||
      *device == '\0' || result.inode == 0) {
    throw std::runtime_error(
        "exact physics call-target attestation is incomplete");
  }
  result.device = device;
  return result;
}

void append_string(std::vector<std::byte>& output, const std::string& value) {
  if (value.size() > std::numeric_limits<std::uint16_t>::max()) {
    throw std::overflow_error("protocol string exceeds uint16 length");
  }
  append(output, static_cast<std::uint16_t>(value.size()));
  output.insert(output.end(),
                reinterpret_cast<const std::byte*>(value.data()),
                reinterpret_cast<const std::byte*>(value.data() + value.size()));
}

std::string read_string(const std::vector<std::byte>& source,
                        std::size_t& offset) {
  const auto size = read_value<std::uint16_t>(source, offset);
  if (offset > source.size() || source.size() - offset < size) {
    throw std::invalid_argument("truncated request string");
  }
  const auto* begin = reinterpret_cast<const char*>(source.data() + offset);
  std::string value(begin, begin + size);
  offset += size;
  return value;
}

void append_event(std::vector<std::byte>& output, const irisu::Event& event) {
  if (event.detail.size() > std::numeric_limits<std::uint16_t>::max()) {
    throw std::overflow_error("event detail exceeds protocol limit");
  }
  append(output, event.tick);
  append(output, event.sequence);
  append(output, event.value);
  append(output, event.a);
  append(output, event.b);
  append(output, static_cast<std::uint16_t>(event.detail.size()));
  append(output, static_cast<std::uint8_t>(event.kind));
  append(output, std::uint8_t{});
  output.insert(output.end(),
                reinterpret_cast<const std::byte*>(event.detail.data()),
                reinterpret_cast<const std::byte*>(event.detail.data() +
                                                   event.detail.size()));
}

void append_observation(std::vector<std::byte>& output,
                        const irisu::Observation& observation) {
  if (observation.bodies.size() > kBodyCapacity) {
    throw std::overflow_error("observation exceeds actor capacity");
  }
  output.reserve(output.size() + 112U + 100U * observation.bodies.size());
  append(output, observation.tick);
  append(output, observation.score);
  append(output, observation.gauge);
  append(output, observation.gauge_max);
  append(output, observation.qualifying_clear_count);
  append(output, observation.field_x);
  append(output, observation.field_y);
  append(output, observation.field_width);
  append(output, observation.field_height);
  append(output, observation.side_wall_top);
  append(output, observation.side_wall_bottom);
  append(output, observation.level);
  append(output, observation.active_colors);
  append(output, observation.current_spawn_interval_ticks);
  append(output, observation.highest_chain);
  append(output, static_cast<std::uint32_t>(observation.bodies.size()));
  append(output, static_cast<std::uint8_t>(observation.terminated));
  append(output, static_cast<std::uint8_t>(observation.truncated));
  append(output, static_cast<std::uint8_t>(observation.left_held));
  append(output, static_cast<std::uint8_t>(observation.right_held));
  for (const auto& body : observation.bodies) {
    append(output, body.age_ticks);
    append(output, body.remaining_lifetime);
    append(output, body.rot_timer);
    append(output, body.position.x);
    append(output, body.position.y);
    append(output, body.velocity.x);
    append(output, body.velocity.y);
    append(output, body.angle);
    append(output, body.angular_velocity);
    append(output, body.size);
    append(output, body.id);
    append(output, body.color);
    append(output, body.chain_id);
    append(output, body.projectile_hits);
    append(output, static_cast<std::uint8_t>(body.kind));
    append(output, static_cast<std::uint8_t>(body.shape));
    append(output, static_cast<std::uint8_t>(body.lifecycle));
    append(output, std::uint8_t{});
  }
}

void append_transition_prefix(std::vector<std::byte>& output,
                              const irisu::Observation& observation,
                              const irisu::StepResult& transition) {
  append_observation(output, observation);
  append(output, transition.reward);
  append(output, static_cast<std::uint64_t>(transition.events.size()));
  const auto& diagnostics = transition.diagnostics;
  append(output, diagnostics.config_hash);
  append(output, diagnostics.finish_call_count);
  append(output, diagnostics.recorded_final_score);
  append(output, diagnostics.recorded_final_clears);
  append(output, diagnostics.latest_final_score);
  append(output, diagnostics.latest_final_clears);
  append(output, diagnostics.recorded_final_highest_chain);
  append(output, diagnostics.recorded_final_level);
  append(output, diagnostics.latest_final_highest_chain);
  append(output, diagnostics.latest_final_level);
  append(output, static_cast<std::uint8_t>(transition.terminated));
  append(output, static_cast<std::uint8_t>(transition.truncated));
  append(output,
         static_cast<std::uint8_t>(diagnostics.terminal_metadata_recorded));
  const bool invalid_action = std::any_of(
      transition.events.begin(), transition.events.end(),
      [](const irisu::Event& event) {
        return event.kind == irisu::EventKind::InvalidAction;
      });
  append(output, static_cast<std::uint8_t>(invalid_action));
}

void append_transition(std::vector<std::byte>& output,
                       const irisu::Simulator& simulator,
                       const irisu::StepResult& transition) {
  const irisu::Observation observation = simulator.observation();
  constexpr std::size_t kObservationFixedBytes = 112U;
  constexpr std::size_t kBodyBytes = 100U;
  constexpr std::size_t kTransitionBytes = 84U;
  constexpr std::size_t kEventFixedBytes = 36U;
  std::size_t response_bytes = kObservationFixedBytes + kTransitionBytes +
                               kBodyBytes * observation.bodies.size();
  for (const auto& event : transition.events) {
    response_bytes += kEventFixedBytes + event.detail.size();
  }
  // append_observation reserves exactly through the final body. Without this
  // whole-transition reservation, the diagnostics suffix forces a second
  // allocation and copies the complete observation on every step response.
  output.reserve(output.size() + response_bytes);
  append_transition_prefix(output, observation, transition);
  for (const auto& event : transition.events) append_event(output, event);
}

void append_padded_transition(std::vector<std::byte>& output,
                              const irisu::Simulator& simulator,
                              const irisu::StepResult& transition,
                              std::uint64_t event_generation) {
  const irisu::Observation observation = simulator.observation();
  constexpr std::size_t kObservationFixedBytes = 112U;
  constexpr std::size_t kBodyBytes = 100U;
  constexpr std::size_t kTransitionBytes = 84U;
  output.reserve(output.size() + kObservationFixedBytes + kTransitionBytes +
                 sizeof(event_generation) +
                 kBodyBytes * observation.bodies.size());
  append_transition_prefix(output, observation, transition);
  append(output, event_generation);
}

irisu::Action parse_action(const std::vector<std::byte>& payload) {
  std::size_t offset{};
  const auto raw_kind = read_value<std::uint32_t>(payload, offset);
  const auto x = read_value<double>(payload, offset);
  const auto y = read_value<double>(payload, offset);
  const auto wait_ticks = read_value<std::uint32_t>(payload, offset);
  const auto flags = read_value<std::uint32_t>(payload, offset);
  if (offset != payload.size()) {
    throw std::invalid_argument("step payload has trailing bytes");
  }
  if (raw_kind > static_cast<std::uint32_t>(irisu::ActionKind::BothShots)) {
    throw std::invalid_argument("action kind must be in [0, 3]");
  }
  if (!std::isfinite(x) || !std::isfinite(y)) {
    throw std::invalid_argument("cursor coordinates must be finite");
  }
  if ((flags & ~1U) != 0) {
    throw std::invalid_argument("step flags contain unknown bits");
  }
  return {static_cast<irisu::ActionKind>(raw_kind), x, y, wait_ticks,
          (flags & 1U) != 0};
}

void require_empty(const std::vector<std::byte>& payload) {
  if (!payload.empty()) throw std::invalid_argument("request payload must be empty");
}

class Worker;
int serve_requests(int input_descriptor, int output_descriptor, Worker& worker,
                   int error_descriptor = -1);
[[noreturn]] void run_keeper(Worker& worker, int control_descriptor);

class Worker {
 public:
  Worker(std::uint16_t control_word, ExactPhysicsAttestation attestation)
      : control_word_(control_word),
        exact_attestation_(std::move(attestation)),
        simulator_{} {}
  ~Worker() { shutdown_all_keepers(); }

  Worker(const Worker&) = delete;
  Worker& operator=(const Worker&) = delete;

  void detach_inherited_keepers() noexcept {
    for (auto& [token, keeper] : keepers_) {
      static_cast<void>(token);
      close_descriptor(keeper.control_descriptor);
    }
    keepers_.clear();
  }

  void shutdown_all_keepers() noexcept;

  void set_transport_descriptors(int input_descriptor, int output_descriptor,
                                 int error_descriptor) noexcept {
    rpc_input_descriptor_ = input_descriptor;
    rpc_output_descriptor_ = output_descriptor;
    error_descriptor_ = error_descriptor;
  }

  void close_inherited_transport() noexcept {
    if (rpc_output_descriptor_ != rpc_input_descriptor_) {
      close_descriptor(rpc_output_descriptor_);
    }
    close_descriptor(rpc_input_descriptor_);
    close_descriptor(error_descriptor_);
  }

  std::vector<std::byte> dispatch(Opcode opcode,
                                  const std::vector<std::byte>& payload,
                                  bool& keep_running) {
    std::vector<std::byte> output;
    switch (opcode) {
      case Opcode::Hello:
        require_empty(payload);
        append(output, static_cast<std::uint32_t>(kVersion));
        append(output, static_cast<std::uint32_t>(sizeof(void*) * 8U));
        append(output, kBodyCapacity);
        append(output, static_cast<std::uint32_t>(::getpid()));
        append(output, simulator_.config_hash());
        append(output, static_cast<std::uint32_t>(control_word_));
        append(output, std::uint32_t{1});  // One exact world per process.
        append_string(output, kBackendName);
        append_string(output, __VERSION__);
        append_string(output, IRISU_EXACT_LIBRARY_SHA256);
        break;
      case Opcode::Reset: {
        std::size_t offset{};
        const auto seed = read_value<std::uint64_t>(payload, offset);
        if (offset != payload.size()) {
          throw std::invalid_argument("reset payload has trailing bytes");
        }
        if (seed > std::numeric_limits<std::uint32_t>::max()) {
          throw std::invalid_argument("normal-mode seed must fit in uint32");
        }
        if (has_reset_) {
          throw std::invalid_argument(
              "exact worker permits one successful reset per process");
        }
        invalidate_padded_events();
        const auto observation = simulator_.reset(seed);
        has_reset_ = true;
        append_observation(output, observation);
        break;
      }
      case Opcode::Step: {
        const auto action = parse_action(payload);
        invalidate_padded_events();
        const auto transition = simulator_.step(action);
        append_transition(output, simulator_, transition);
        break;
      }
      case Opcode::StepPadded: {
        const auto action = parse_action(payload);
        invalidate_padded_events();
        auto transition = simulator_.step(action);
        append_padded_transition(output, simulator_, transition,
                                 event_generation_);
        padded_events_ = std::move(transition.events);
        padded_events_available_ = true;
        break;
      }
      case Opcode::FetchEvents: {
        std::size_t offset{};
        const auto generation = read_value<std::uint64_t>(payload, offset);
        if (offset != payload.size()) {
          throw std::invalid_argument("fetch-events payload has trailing bytes");
        }
        if (!padded_events_available_ || generation != event_generation_) {
          throw std::invalid_argument("lazy padded events expired");
        }
        constexpr std::size_t kEventFixedBytes = 36U;
        std::size_t response_bytes = sizeof(std::uint64_t);
        for (const auto& event : padded_events_) {
          const std::size_t record_bytes = kEventFixedBytes + event.detail.size();
          if (record_bytes > kMaximumResponseContentBytes - response_bytes) {
            throw std::invalid_argument(
                "lazy padded event batch exceeds 4 MiB response limit");
          }
          response_bytes += record_bytes;
        }
        output.reserve(response_bytes);
        append(output, static_cast<std::uint64_t>(padded_events_.size()));
        for (const auto& event : padded_events_) append_event(output, event);
        break;
      }
      case Opcode::Observe:
        require_empty(payload);
        append_observation(output, simulator_.observation());
        break;
      case Opcode::Close:
        require_empty(payload);
        keep_running = false;
        break;
      case Opcode::Configure: {
        std::size_t offset{};
        const auto count = read_value<std::uint32_t>(payload, offset);
        if (count > 1024U) {
          throw std::invalid_argument("configuration override count exceeds limit");
        }
        irisu::MechanicsConfig config;
        for (std::uint32_t index = 0; index < count; ++index) {
          const std::string key = read_string(payload, offset);
          const double value = read_value<double>(payload, offset);
          irisu::apply_config_override(config, key, value);
        }
        if (offset != payload.size()) {
          throw std::invalid_argument("configure payload has trailing bytes");
        }
        config = irisu::validated_mechanics_config(std::move(config));
        if (has_reset_) {
          throw std::invalid_argument(
              "exact worker configuration is immutable after reset");
        }
        irisu::Simulator configured(config);
        invalidate_padded_events();
        simulator_ = std::move(configured);
        append(output, simulator_.config_hash());
        break;
      }
      case Opcode::ConfigJson: {
        require_empty(payload);
        const std::string json = irisu::mechanics_config_json(
            simulator_.config(), simulator_.config_hash());
        output.insert(output.end(),
                      reinterpret_cast<const std::byte*>(json.data()),
                      reinterpret_cast<const std::byte*>(json.data() +
                                                         json.size()));
        break;
      }
      case Opcode::ExactAttestation:
        require_empty(payload);
        append(output, kExactAttestationSchema);
        append(output, exact_attestation_.entrypoint_count);
        append(output, exact_attestation_.inode);
        append_string(output, exact_attestation_.device);
        break;
      case Opcode::FastCheckpoint:
        require_empty(payload);
        create_fast_checkpoint(output);
        break;
      case Opcode::FastRelease:
        release_fast_checkpoint(payload);
        break;
      case Opcode::FastBranch:
        branch_fast_checkpoint(payload, output);
        break;
      default:
        throw std::invalid_argument("unknown request opcode");
    }
    return output;
  }

 private:
  struct Keeper {
    pid_t pid{};
    int control_descriptor{-1};
  };

  static FastToken parse_fast_token(const std::vector<std::byte>& payload) {
    if (payload.size() != kFastTokenBytes) {
      throw std::invalid_argument("fast checkpoint token must contain 16 bytes");
    }
    FastToken token{};
    std::copy(payload.begin(), payload.end(), token.begin());
    return token;
  }

  void create_fast_checkpoint(std::vector<std::byte>& output);
  void release_fast_checkpoint(const std::vector<std::byte>& payload);
  void branch_fast_checkpoint(const std::vector<std::byte>& payload,
                              std::vector<std::byte>& output);
  void forget_failed_keeper(const FastToken& token, Keeper keeper) noexcept;

  void advance_event_generation() {
    if (event_generation_ == std::numeric_limits<std::uint64_t>::max()) {
      throw std::overflow_error("padded event generation exhausted");
    }
    ++event_generation_;
  }

  void invalidate_padded_events() {
    advance_event_generation();
    padded_events_.clear();
    padded_events_available_ = false;
  }

  std::uint16_t control_word_{};
  ExactPhysicsAttestation exact_attestation_;
  irisu::Simulator simulator_;
  bool has_reset_{};
  std::uint64_t event_generation_{};
  std::vector<irisu::Event> padded_events_;
  bool padded_events_available_{};
  std::map<FastToken, Keeper> keepers_;
  int rpc_input_descriptor_{STDIN_FILENO};
  int rpc_output_descriptor_{STDOUT_FILENO};
  int error_descriptor_{STDERR_FILENO};
};

void send_keeper_status(int descriptor, KeeperStatus status) {
  const auto raw = static_cast<std::uint8_t>(status);
  write_exact(descriptor, &raw, sizeof(raw));
}

KeeperStatus receive_keeper_status(int descriptor) {
  std::uint8_t raw{};
  read_exact(descriptor, &raw, sizeof(raw), false);
  if (raw > static_cast<std::uint8_t>(KeeperStatus::Error)) {
    throw std::runtime_error("checkpoint keeper returned an invalid status");
  }
  return static_cast<KeeperStatus>(raw);
}

void send_keeper_error(int descriptor, const std::exception& error) {
  const std::string message = error.what();
  const auto bounded_size = std::min<std::size_t>(
      message.size(), std::numeric_limits<std::uint16_t>::max());
  send_keeper_status(descriptor, KeeperStatus::Error);
  const auto size = static_cast<std::uint16_t>(bounded_size);
  write_exact(descriptor, &size, sizeof(size));
  write_exact(descriptor, message.data(), bounded_size);
}

std::string receive_keeper_error(int descriptor) {
  std::uint16_t size{};
  read_exact(descriptor, &size, sizeof(size), false);
  std::string message(size, '\0');
  read_exact(descriptor, message.data(), message.size(), false);
  return message;
}

void reap_children(std::set<pid_t>& children) noexcept {
  for (auto iterator = children.begin(); iterator != children.end();) {
    int status{};
    const pid_t result = ::waitpid(*iterator, &status, WNOHANG);
    if (result == *iterator || (result < 0 && errno == ECHILD)) {
      iterator = children.erase(iterator);
    } else {
      ++iterator;
    }
  }
}

void terminate_children(std::set<pid_t>& children) noexcept {
  reap_children(children);
  for (const pid_t child : children) {
    if (::kill(child, SIGTERM) != 0 && errno != ESRCH) {
      // A direct child cannot legitimately reject our signal. Continue to the
      // bounded SIGKILL/reap phase so shutdown never leaks another child.
    }
  }
  for (int attempt = 0; attempt < 50 && !children.empty(); ++attempt) {
    ::usleep(2'000);
    reap_children(children);
  }
  for (const pid_t child : children) {
    if (::kill(child, SIGKILL) != 0 && errno != ESRCH) {
    }
  }
  for (const pid_t child : children) {
    int status{};
    while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {
    }
  }
  children.clear();
}

bool wait_for_process(pid_t process, int attempts = 100) noexcept {
  for (int attempt = 0; attempt < attempts; ++attempt) {
    int status{};
    const pid_t result = ::waitpid(process, &status, WNOHANG);
    if (result == process || (result < 0 && errno == ECHILD)) return true;
    if (result < 0 && errno != EINTR) return false;
    ::usleep(2'000);
  }
  return false;
}

void terminate_process(pid_t process) noexcept {
  if (process <= 0 || wait_for_process(process, 1)) return;
  static_cast<void>(::kill(process, SIGTERM));
  if (wait_for_process(process, 50)) return;
  static_cast<void>(::kill(process, SIGKILL));
  int status{};
  while (::waitpid(process, &status, 0) < 0 && errno == EINTR) {
  }
}

std::string token_hex(const FastToken& token) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string output;
  output.reserve(token.size() * 2U);
  for (const std::byte value : token) {
    const auto number = std::to_integer<unsigned int>(value);
    output.push_back(kHex[number >> 4U]);
    output.push_back(kHex[number & 0x0fU]);
  }
  return output;
}

struct BranchListener {
  int descriptor{-1};
  FastToken secret{};
  std::vector<std::byte> address;
};

BranchListener create_branch_listener() {
  BranchListener listener;
  listener.secret = random_token();
  listener.descriptor =
      ::socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
  if (listener.descriptor < 0) {
    throw std::runtime_error("failed to create fast-branch socket");
  }
  try {
    const std::string label =
        "irisu-exact-" + std::to_string(::getpid()) + "-" +
        token_hex(random_token());
    listener.address.reserve(label.size() + 1U);
    listener.address.push_back(std::byte{});
    listener.address.insert(
        listener.address.end(),
        reinterpret_cast<const std::byte*>(label.data()),
        reinterpret_cast<const std::byte*>(label.data() + label.size()));
    if (listener.address.size() > kMaximumBranchAddressBytes) {
      throw std::runtime_error("fast-branch address exceeds sockaddr_un");
    }
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::copy(listener.address.begin(), listener.address.end(),
              reinterpret_cast<std::byte*>(address.sun_path));
    const auto address_size = static_cast<socklen_t>(
        offsetof(sockaddr_un, sun_path) + listener.address.size());
    if (::bind(listener.descriptor,
               reinterpret_cast<const sockaddr*>(&address), address_size) != 0) {
      throw std::runtime_error("failed to bind fast-branch socket");
    }
    if (::listen(listener.descriptor, 1) != 0) {
      throw std::runtime_error("failed to listen on fast-branch socket");
    }
    return listener;
  } catch (...) {
    close_descriptor(listener.descriptor);
    throw;
  }
}

int accept_authenticated_branch(BranchListener& listener) {
  pollfd pending{listener.descriptor, POLLIN, 0};
  int ready{};
  do {
    ready = ::poll(&pending, 1, 5'000);
  } while (ready < 0 && errno == EINTR);
  if (ready <= 0 || (pending.revents & POLLIN) == 0) {
    throw std::runtime_error("fast-branch connection timed out");
  }
  int connection = ::accept4(listener.descriptor, nullptr, nullptr, SOCK_CLOEXEC);
  if (connection < 0) {
    throw std::runtime_error("failed to accept fast-branch connection");
  }
  try {
    timeval timeout{1, 0};
    if (::setsockopt(connection, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                     sizeof(timeout)) != 0) {
      throw std::runtime_error("failed to bound fast-branch authentication");
    }
    FastToken supplied{};
    read_exact(connection, supplied.data(), supplied.size(), false);
    if (!tokens_equal(supplied, listener.secret)) {
      throw std::runtime_error("fast-branch authentication failed");
    }
    timeout = timeval{};
    if (::setsockopt(connection, SOL_SOCKET, SO_RCVTIMEO, &timeout,
                     sizeof(timeout)) != 0) {
      throw std::runtime_error("failed to clear fast-branch timeout");
    }
    return connection;
  } catch (...) {
    close_descriptor(connection);
    throw;
  }
}

void send_branch_reply(int descriptor, pid_t process,
                       const BranchListener& listener) {
  if (process <= 0 ||
      static_cast<std::uint64_t>(process) >
          std::numeric_limits<std::uint32_t>::max() ||
      listener.address.empty() ||
      listener.address.size() > std::numeric_limits<std::uint16_t>::max()) {
    throw std::runtime_error("fast-branch response metadata is out of range");
  }
  send_keeper_status(descriptor, KeeperStatus::Ok);
  const auto pid = static_cast<std::uint32_t>(process);
  const auto size = static_cast<std::uint16_t>(listener.address.size());
  write_exact(descriptor, &pid, sizeof(pid));
  write_exact(descriptor, &size, sizeof(size));
  write_exact(descriptor, listener.secret.data(), listener.secret.size());
  write_exact(descriptor, listener.address.data(), listener.address.size());
}

void start_branch(Worker& worker, int control_descriptor,
                  std::set<pid_t>& children) {
  BranchListener listener = create_branch_listener();
  const pid_t branch = ::fork();
  if (branch < 0) {
    close_descriptor(listener.descriptor);
    throw std::runtime_error("failed to fork fast branch");
  }
  if (branch == 0) {
    close_descriptor(control_descriptor);
    int connection{-1};
    int result{1};
    try {
      connection = accept_authenticated_branch(listener);
      close_descriptor(listener.descriptor);
      result = serve_requests(connection, connection, worker);
    } catch (...) {
      result = 1;
    }
    worker.shutdown_all_keepers();
    close_descriptor(connection);
    close_descriptor(listener.descriptor);
    ::_exit(result);
  }

  close_descriptor(listener.descriptor);
  children.insert(branch);
  try {
    send_branch_reply(control_descriptor, branch, listener);
  } catch (...) {
    static_cast<void>(::kill(branch, SIGKILL));
    int status{};
    while (::waitpid(branch, &status, 0) < 0 && errno == EINTR) {
    }
    children.erase(branch);
    throw;
  }
}

[[noreturn]] void run_keeper(Worker& worker, int control_descriptor) {
  worker.detach_inherited_keepers();
  std::set<pid_t> children;
  bool running = true;
  while (running) {
    reap_children(children);
    pollfd request{control_descriptor, POLLIN, 0};
    int ready{};
    do {
      ready = ::poll(&request, 1, 100);
    } while (ready < 0 && errno == EINTR);
    if (ready < 0) break;
    if (ready == 0) continue;
    if ((request.revents & POLLIN) == 0) break;
    std::uint8_t raw_command{};
    if (!read_exact(control_descriptor, &raw_command, sizeof(raw_command), true)) {
      break;
    }
    const auto command = static_cast<KeeperCommand>(raw_command);
    try {
      switch (command) {
        case KeeperCommand::Branch:
          start_branch(worker, control_descriptor, children);
          break;
        case KeeperCommand::Release:
          for (int attempt = 0; attempt < 50 && !children.empty(); ++attempt) {
            reap_children(children);
            if (!children.empty()) ::usleep(2'000);
          }
          if (!children.empty()) {
            send_keeper_status(control_descriptor, KeeperStatus::Busy);
          } else {
            send_keeper_status(control_descriptor, KeeperStatus::Ok);
            running = false;
          }
          break;
        case KeeperCommand::Shutdown:
          terminate_children(children);
          send_keeper_status(control_descriptor, KeeperStatus::Ok);
          running = false;
          break;
        default:
          throw std::runtime_error("checkpoint keeper received an invalid command");
      }
    } catch (const std::exception& error) {
      try {
        send_keeper_error(control_descriptor, error);
      } catch (...) {
        running = false;
      }
    }
  }
  terminate_children(children);
  close_descriptor(control_descriptor);
  ::_exit(0);
}

void Worker::forget_failed_keeper(const FastToken& token,
                                  Keeper keeper) noexcept {
  close_descriptor(keeper.control_descriptor);
  terminate_process(keeper.pid);
  keepers_.erase(token);
}

void Worker::create_fast_checkpoint(std::vector<std::byte>& output) {
  int channel[2]{-1, -1};
  if (::socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, channel) != 0) {
    throw std::invalid_argument("fast checkpoints are unavailable: socketpair failed");
  }
  FastToken token{};
  do {
    token = random_token();
  } while (keepers_.contains(token));

  const pid_t keeper = ::fork();
  if (keeper < 0) {
    close_descriptor(channel[0]);
    close_descriptor(channel[1]);
    throw std::invalid_argument("fast checkpoint fork failed");
  }
  if (keeper == 0) {
    close_descriptor(channel[0]);
    close_inherited_transport();
    run_keeper(*this, channel[1]);
  }

  close_descriptor(channel[1]);
  try {
    keepers_.emplace(token, Keeper{keeper, channel[0]});
  } catch (...) {
    close_descriptor(channel[0]);
    terminate_process(keeper);
    throw;
  }
  output.insert(output.end(), token.begin(), token.end());
  append(output, static_cast<std::uint32_t>(keeper));
}

void Worker::release_fast_checkpoint(const std::vector<std::byte>& payload) {
  const FastToken token = parse_fast_token(payload);
  const auto iterator = keepers_.find(token);
  if (iterator == keepers_.end()) {
    throw std::invalid_argument("unknown checkpoint token");
  }
  const Keeper keeper = iterator->second;
  KeeperStatus status{};
  try {
    const auto command = static_cast<std::uint8_t>(KeeperCommand::Release);
    write_exact(keeper.control_descriptor, &command, sizeof(command));
    status = receive_keeper_status(keeper.control_descriptor);
  } catch (const std::exception&) {
    forget_failed_keeper(token, keeper);
    throw std::invalid_argument("fast checkpoint keeper is unavailable");
  }
  if (status == KeeperStatus::Busy) {
    throw std::invalid_argument("fast checkpoint has active branches");
  }
  if (status == KeeperStatus::Error) {
    std::string detail;
    try {
      detail = receive_keeper_error(keeper.control_descriptor);
    } catch (const std::exception&) {
      forget_failed_keeper(token, keeper);
      throw std::invalid_argument("fast checkpoint keeper is unavailable");
    }
    throw std::invalid_argument("fast checkpoint release failed: " + detail);
  }

  close_descriptor(iterator->second.control_descriptor);
  if (!wait_for_process(keeper.pid)) terminate_process(keeper.pid);
  keepers_.erase(iterator);
}

void Worker::branch_fast_checkpoint(const std::vector<std::byte>& payload,
                                    std::vector<std::byte>& output) {
  const FastToken token = parse_fast_token(payload);
  const auto iterator = keepers_.find(token);
  if (iterator == keepers_.end()) {
    throw std::invalid_argument("unknown checkpoint token");
  }
  const Keeper keeper = iterator->second;
  KeeperStatus status{};
  try {
    const auto command = static_cast<std::uint8_t>(KeeperCommand::Branch);
    write_exact(keeper.control_descriptor, &command, sizeof(command));
    status = receive_keeper_status(keeper.control_descriptor);
  } catch (const std::exception&) {
    forget_failed_keeper(token, keeper);
    throw std::invalid_argument("fast checkpoint keeper is unavailable");
  }
  if (status == KeeperStatus::Busy) {
    throw std::invalid_argument("fast checkpoint keeper unexpectedly reported busy");
  }
  if (status == KeeperStatus::Error) {
    std::string detail;
    try {
      detail = receive_keeper_error(keeper.control_descriptor);
    } catch (const std::exception&) {
      forget_failed_keeper(token, keeper);
      throw std::invalid_argument("fast checkpoint keeper is unavailable");
    }
    throw std::invalid_argument("fast checkpoint branch failed: " + detail);
  }

  try {
    std::uint32_t process{};
    std::uint16_t address_size{};
    FastToken secret{};
    read_exact(keeper.control_descriptor, &process, sizeof(process), false);
    read_exact(keeper.control_descriptor, &address_size, sizeof(address_size),
               false);
    read_exact(keeper.control_descriptor, secret.data(), secret.size(), false);
    if (process == 0 || address_size == 0 ||
        address_size > kMaximumBranchAddressBytes) {
      throw std::runtime_error("fast checkpoint keeper returned invalid metadata");
    }
    std::vector<std::byte> address(address_size);
    read_exact(keeper.control_descriptor, address.data(), address.size(), false);
    if (address.front() != std::byte{}) {
      throw std::runtime_error("fast checkpoint keeper returned a non-abstract socket");
    }
    append(output, process);
    append(output, address_size);
    output.insert(output.end(), secret.begin(), secret.end());
    output.insert(output.end(), address.begin(), address.end());
  } catch (const std::exception&) {
    forget_failed_keeper(token, keeper);
    throw std::invalid_argument("fast checkpoint keeper returned a malformed branch");
  }
}

void Worker::shutdown_all_keepers() noexcept {
  auto keepers = std::move(keepers_);
  keepers_.clear();
  for (auto& [token, keeper] : keepers) {
    static_cast<void>(token);
    try {
      const auto command = static_cast<std::uint8_t>(KeeperCommand::Shutdown);
      write_exact(keeper.control_descriptor, &command, sizeof(command));
      static_cast<void>(receive_keeper_status(keeper.control_descriptor));
    } catch (...) {
    }
    close_descriptor(keeper.control_descriptor);
    if (!wait_for_process(keeper.pid)) terminate_process(keeper.pid);
  }
}

void send_response(int output_descriptor, std::uint16_t opcode,
                   std::uint32_t request_id, Status status,
                   const std::vector<std::byte>& content) {
  const std::vector<std::byte>* bounded_content = &content;
  std::vector<std::byte> overflow_content;
  if (content.size() > kMaximumResponseContentBytes) {
    constexpr char kDetail[] = "response exceeds 4 MiB protocol limit";
    overflow_content.assign(
        reinterpret_cast<const std::byte*>(kDetail),
        reinterpret_cast<const std::byte*>(kDetail + sizeof(kDetail) - 1U));
    bounded_content = &overflow_content;
    status = Status::InternalError;
  }
  std::vector<std::byte> frame;
  frame.reserve(kHeaderBytes + sizeof(std::int32_t) + bounded_content->size());
  append(frame, kMagic);
  append(frame, kVersion);
  append(frame, opcode);
  append(frame, request_id);
  append(frame, static_cast<std::uint32_t>(sizeof(std::int32_t) +
                                           bounded_content->size()));
  append(frame, static_cast<std::int32_t>(status));
  frame.insert(frame.end(), bounded_content->begin(), bounded_content->end());
  write_exact(output_descriptor, frame.data(), frame.size());
}

std::vector<std::byte> error_content(const std::exception& error) {
  const std::string message = error.what();
  return {reinterpret_cast<const std::byte*>(message.data()),
          reinterpret_cast<const std::byte*>(message.data() + message.size())};
}

int serve_requests(int input_descriptor, int output_descriptor, Worker& worker,
                   int error_descriptor) {
  worker.set_transport_descriptors(input_descriptor, output_descriptor,
                                   error_descriptor);
  bool keep_running = true;
  while (keep_running) {
    std::array<std::byte, kHeaderBytes> header{};
    if (!read_exact(input_descriptor, header.data(), header.size(), true)) break;
    const std::vector<std::byte> header_vector(header.begin(), header.end());
    std::size_t offset{};
    const auto magic = read_value<std::uint32_t>(header_vector, offset);
    const auto version = read_value<std::uint16_t>(header_vector, offset);
    const auto raw_opcode = read_value<std::uint16_t>(header_vector, offset);
    const auto request_id = read_value<std::uint32_t>(header_vector, offset);
    const auto payload_size = read_value<std::uint32_t>(header_vector, offset);
    if (magic != kMagic || version != kVersion ||
        payload_size > kMaximumPayloadBytes) {
      throw std::runtime_error("invalid request frame header");
    }
    std::vector<std::byte> payload(payload_size);
    read_exact(input_descriptor, payload.data(), payload.size(), false);
    try {
      const auto content = worker.dispatch(static_cast<Opcode>(raw_opcode),
                                           payload, keep_running);
      if (content.size() > kMaximumResponseContentBytes) {
        const std::runtime_error overflow(
            "response exceeds 4 MiB protocol limit");
        send_response(output_descriptor, raw_opcode, request_id,
                      Status::InternalError, error_content(overflow));
        keep_running = false;
      } else {
        send_response(output_descriptor, raw_opcode, request_id, Status::Ok,
                      content);
      }
    } catch (const std::invalid_argument& error) {
      send_response(output_descriptor, raw_opcode, request_id,
                    Status::BadRequest,
                    error_content(error));
    } catch (const std::exception& error) {
      send_response(output_descriptor, raw_opcode, request_id,
                    Status::InternalError,
                    error_content(error));
      keep_running = false;
    }
  }
  return 0;
}

int run() {
  static_assert(std::endian::native == std::endian::little,
                "the exact worker protocol currently requires little endian");
  static_assert(sizeof(double) == 8 && sizeof(std::uint64_t) == 8);
  irisu::detail::install_canonical_floating_point_environment();
  const std::uint16_t control_word = requested_control_word();
  __asm__ __volatile__("fldcw %0" : : "m"(control_word));
  if (::signal(SIGPIPE, SIG_IGN) == SIG_ERR) {
    throw std::runtime_error("failed to ignore SIGPIPE");
  }
  auto attestation = capture_exact_physics_attestation();
  Worker worker(control_word, std::move(attestation));
  return serve_requests(STDIN_FILENO, STDOUT_FILENO, worker, STDERR_FILENO);
}

}  // namespace

int main() {
  try {
    return run();
  } catch (const std::exception& error) {
    std::cerr << "exact IPC worker: " << error.what() << '\n';
    return 1;
  }
}
