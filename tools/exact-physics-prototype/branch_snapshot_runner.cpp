#include "irisu/simulator.hpp"

#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace {

constexpr std::uint64_t kFnvOffset = 14695981039346656037ULL;
constexpr std::uint64_t kFnvPrime = 1099511628211ULL;

struct Digest {
  std::uint64_t all_hash{kFnvOffset};
  std::uint64_t contact_hash{kFnvOffset};
  std::uint64_t frames{};
  std::uint64_t events{};
  std::uint64_t contacts{};
  std::uint64_t body_samples{};
  std::uint64_t final_tick{};
  std::int64_t final_score{};
  std::int64_t final_gauge{};

  friend bool operator==(const Digest&, const Digest&) = default;
};

struct TimedDigest {
  Digest digest{};
  std::uint64_t prefix_ns{};
  std::uint64_t elapsed_ns{};
};

void hash_bytes(std::uint64_t& hash, const void* data, std::size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (std::size_t index = 0; index < size; ++index) {
    hash ^= bytes[index];
    hash *= kFnvPrime;
  }
}

template <class T>
void hash_value(std::uint64_t& hash, const T& value) {
  static_assert(std::is_trivially_copyable_v<T>);
  hash_bytes(hash, &value, sizeof(value));
}

void hash_text(std::uint64_t& hash, std::string_view value) {
  hash_value(hash, value.size());
  hash_bytes(hash, value.data(), value.size());
}

void hash_vec(std::uint64_t& hash, irisu::Vec2 value) {
  hash_value(hash, std::bit_cast<std::uint64_t>(value.x));
  hash_value(hash, std::bit_cast<std::uint64_t>(value.y));
}

void hash_body(std::uint64_t& hash, const irisu::Body& body) {
  hash_value(hash, body.id);
  hash_value(hash, body.kind);
  hash_value(hash, body.shape);
  hash_value(hash, body.lifecycle);
  hash_value(hash, body.color);
  hash_vec(hash, body.position);
  hash_vec(hash, body.velocity);
  hash_value(hash, std::bit_cast<std::uint64_t>(body.angle));
  hash_value(hash, std::bit_cast<std::uint64_t>(body.angular_velocity));
  hash_vec(hash, body.native_position);
  hash_vec(hash, body.native_center);
  hash_vec(hash, body.native_velocity);
  hash_value(hash, std::bit_cast<std::uint64_t>(body.native_angle));
  hash_value(hash,
             std::bit_cast<std::uint64_t>(body.native_angular_velocity));
  hash_value(hash, body.native_state_valid);
  hash_value(hash, body.native_center_valid);
  hash_value(hash, body.chain_id);
  hash_value(hash, body.actor_slot);
  hash_value(hash, body.projectile_hits);
  hash_value(hash, body.age_ticks);
  hash_value(hash, body.remaining_lifetime);
  hash_value(hash, body.rot_timer);
  hash_value(hash, body.physics_owned);
  hash_value(hash, body.freshness_state);
  hash_value(hash, body.grouped);
  hash_value(hash, body.successful_clear_pending);
  hash_value(hash, body.non_wall_contacts);
  hash_value(hash, body.top_contact_pending);
  hash_value(hash, body.pending_delete);
  hash_value(hash, body.sleeping);
  hash_value(hash, std::bit_cast<std::uint64_t>(body.sleep_time));
}

void hash_event(std::uint64_t& hash, const irisu::Event& event) {
  hash_value(hash, event.tick);
  hash_value(hash, event.kind);
  hash_value(hash, event.a);
  hash_value(hash, event.b);
  hash_value(hash, event.value);
  hash_text(hash, event.detail);
  hash_value(hash, event.sequence);
}

std::uint32_t word(const std::vector<unsigned char>& data, std::size_t offset) {
  std::uint32_t result{};
  std::memcpy(&result, data.data() + offset, sizeof(result));
  return result;
}

irisu::Action action(std::uint32_t value, std::size_t frame) {
  const auto buttons = value & 3U;
  const double x = static_cast<double>((value >> 2U) & 0x3ffU);
  const double y = static_cast<double>((value >> 12U) & 0x1ffU);
  const auto kind = buttons == 1U   ? irisu::ActionKind::WeakShot
                    : buttons == 2U ? irisu::ActionKind::StrongShot
                    : buttons == 3U ? irisu::ActionKind::BothShots
                                    : irisu::ActionKind::Wait;
  return {kind, x, y, 1, frame < 2};
}

void step_prefix(irisu::Simulator& simulator,
                 const std::vector<std::uint32_t>& actions,
                 std::size_t prefix) {
  for (std::size_t frame = 0; frame < prefix; ++frame) {
    simulator.step(action(actions.at(frame), frame));
  }
}

Digest step_suffix(irisu::Simulator& simulator,
                   const std::vector<std::uint32_t>& actions,
                   std::size_t prefix, std::size_t future) {
  Digest digest;
  for (std::size_t frame = prefix; frame < prefix + future; ++frame) {
    const auto result = simulator.step(action(actions.at(frame), frame));
    ++digest.frames;
    hash_value(digest.all_hash, frame);
    hash_value(digest.all_hash, result.reward);
    hash_value(digest.all_hash, result.terminated);
    hash_value(digest.all_hash, result.truncated);
    for (const auto& event : result.events) {
      ++digest.events;
      hash_event(digest.all_hash, event);
      if (event.kind == irisu::EventKind::Contact ||
          event.kind == irisu::EventKind::ProjectileContact) {
        ++digest.contacts;
        hash_event(digest.contact_hash, event);
      }
    }
    const auto& bodies = simulator.bodies();
    hash_value(digest.all_hash, bodies.size());
    for (const auto& body : bodies) {
      ++digest.body_samples;
      hash_body(digest.all_hash, body);
    }
  }
  const auto observation = simulator.observation();
  digest.final_tick = observation.tick;
  digest.final_score = observation.score;
  digest.final_gauge = observation.gauge;
  return digest;
}

TimedDigest replay_branch(const std::vector<std::uint32_t>& actions,
                          std::uint32_t seed, std::size_t prefix,
                          std::size_t future) {
  const auto started = std::chrono::steady_clock::now();
  irisu::Simulator simulator;
  simulator.reset(seed);
  step_prefix(simulator, actions, prefix);
  const auto prefix_finished = std::chrono::steady_clock::now();
  TimedDigest result;
  result.digest = step_suffix(simulator, actions, prefix, future);
  result.prefix_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(prefix_finished -
                                                           started)
          .count());
  result.elapsed_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started)
          .count());
  return result;
}

void write_full(int descriptor, const void* data, std::size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(data);
  while (size != 0) {
    const ssize_t written = ::write(descriptor, bytes, size);
    if (written <= 0) throw std::runtime_error("pipe write failed");
    bytes += written;
    size -= static_cast<std::size_t>(written);
  }
}

void read_full(int descriptor, void* data, std::size_t size) {
  auto* bytes = static_cast<unsigned char*>(data);
  while (size != 0) {
    const ssize_t received = ::read(descriptor, bytes, size);
    if (received <= 0) throw std::runtime_error("pipe read failed");
    bytes += received;
    size -= static_cast<std::size_t>(received);
  }
}

struct ForkResult {
  TimedDigest parent{};
  TimedDigest child{};
  std::uint64_t snapshot_ns{};
};

ForkResult fork_branch(const std::vector<std::uint32_t>& actions,
                       std::uint32_t seed, std::size_t prefix,
                       std::size_t future) {
  irisu::Simulator simulator;
  simulator.reset(seed);
  step_prefix(simulator, actions, prefix);

  int channel[2]{};
  if (::pipe(channel) != 0) throw std::runtime_error("pipe failed");
  const auto fork_started = std::chrono::steady_clock::now();
  const pid_t child = ::fork();
  const auto fork_finished = std::chrono::steady_clock::now();
  if (child < 0) throw std::runtime_error("fork failed");
  if (child == 0) {
    ::close(channel[0]);
    TimedDigest result;
    const auto started = std::chrono::steady_clock::now();
    result.digest = step_suffix(simulator, actions, prefix, future);
    result.elapsed_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
    write_full(channel[1], &result, sizeof(result));
    ::close(channel[1]);
    ::_exit(0);
  }

  ::close(channel[1]);
  ForkResult result;
  result.snapshot_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(fork_finished -
                                                           fork_started)
          .count());
  const auto started = std::chrono::steady_clock::now();
  result.parent.digest = step_suffix(simulator, actions, prefix, future);
  result.parent.elapsed_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - started)
          .count());
  read_full(channel[0], &result.child, sizeof(result.child));
  ::close(channel[0]);
  int status{};
  if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0) {
    throw std::runtime_error("fork child failed");
  }
  return result;
}

double milliseconds(std::uint64_t nanoseconds) {
  return static_cast<double>(nanoseconds) / 1'000'000.0;
}

void print_digest(const Digest& digest) {
  std::cout << "{\"all_hash\":\"" << std::hex << digest.all_hash
            << "\",\"contact_hash\":\"" << digest.contact_hash << std::dec
            << "\",\"frames\":" << digest.frames
            << ",\"events\":" << digest.events
            << ",\"contacts\":" << digest.contacts
            << ",\"body_samples\":" << digest.body_samples
            << ",\"final_tick\":" << digest.final_tick
            << ",\"final_score\":" << digest.final_score
            << ",\"final_gauge\":" << digest.final_gauge << '}';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument(
          "usage: branch-snapshot replay.rpy prefix_frames future_frames");
    }
    const std::size_t prefix = std::stoull(argv[2]);
    const std::size_t future = std::stoull(argv[3]);
    std::ifstream stream(argv[1], std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open replay");
    const std::vector<unsigned char> data{std::istreambuf_iterator<char>(stream),
                                          std::istreambuf_iterator<char>()};
    if (data.size() < 52 || (data.size() - 52) % 4 != 0) {
      throw std::runtime_error("expected padded v2.03 replay");
    }
    std::vector<std::uint32_t> actions;
    for (std::size_t offset = 52; offset < data.size(); offset += 4) {
      actions.push_back(word(data, offset));
    }
    if (prefix > actions.size() || future > actions.size() - prefix) {
      throw std::out_of_range("requested branch extends past replay input");
    }

    const std::uint16_t control_word = 0x027fU;
    __asm__ __volatile__("fldcw %0" : : "m"(control_word));
    const auto forked = fork_branch(actions, word(data, 0), prefix, future);
    const auto replay_a = replay_branch(actions, word(data, 0), prefix, future);
    const auto replay_b = replay_branch(actions, word(data, 0), prefix, future);
    const bool fork_equal = forked.parent.digest == forked.child.digest;
    const bool replay_equal = replay_a.digest == replay_b.digest;
    const bool cross_equal = forked.parent.digest == replay_a.digest;

    std::cout << "{\"schema\":1,\"prefix_frames\":" << prefix
              << ",\"future_frames\":" << future
              << ",\"fork_equal\":" << (fork_equal ? "true" : "false")
              << ",\"replay_equal\":"
              << (replay_equal ? "true" : "false")
              << ",\"cross_equal\":"
              << (cross_equal ? "true" : "false")
              << ",\"fork_snapshot_ms\":" << milliseconds(forked.snapshot_ns)
              << ",\"fork_parent_future_ms\":"
              << milliseconds(forked.parent.elapsed_ns)
              << ",\"fork_child_future_ms\":"
              << milliseconds(forked.child.elapsed_ns)
              << ",\"replay_a_total_ms\":"
              << milliseconds(replay_a.elapsed_ns)
              << ",\"replay_a_prefix_ms\":"
              << milliseconds(replay_a.prefix_ns)
              << ",\"replay_a_future_ms\":"
              << milliseconds(replay_a.elapsed_ns - replay_a.prefix_ns)
              << ",\"replay_b_total_ms\":"
              << milliseconds(replay_b.elapsed_ns)
              << ",\"replay_b_prefix_ms\":"
              << milliseconds(replay_b.prefix_ns)
              << ",\"replay_b_future_ms\":"
              << milliseconds(replay_b.elapsed_ns - replay_b.prefix_ns)
              << ",\"digest\":";
    print_digest(forked.parent.digest);
    std::cout << "}\n";
    return fork_equal && replay_equal && cross_equal ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
