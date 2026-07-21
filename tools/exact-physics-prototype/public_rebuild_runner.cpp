#include <bit>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

extern "C" {
int b2d_init(float, float, float, float, float, float);
void b2d_dispose();
void* b2d_create_box(float, float, float, float, float, float, float, float);
void b2d_step(float, int);
int b2d_get_contact(void**, void**);
float b2d_get_x(void*);
float b2d_get_y(void*);
float b2d_get_r(void*);
void b2d_get_v(void*, float*, float*);
void b2d_set_v(void*, float, float);
void b2d_set_user_data(void*, void*);
}

namespace {

struct PublicState {
  float x{};
  float y{};
  float angle{};
  float velocity_x{};
  float velocity_y{};
};

struct Frame {
  PublicState state{};
  std::vector<std::pair<std::uintptr_t, std::uintptr_t>> contacts{};
};

struct World {
  void* floor{};
  void* box{};

  World(float x, float y, float angle, float velocity_x,
        float velocity_y) {
    if (!b2d_init(-1000.0f, -1000.0f, 1000.0f, 1000.0f, 100.0f,
                  100.0f)) {
      throw std::runtime_error("b2d_init failed");
    }
    floor = b2d_create_box(800.0f, 40.0f, 0.0f, 400.0f, 0.0f, 0.0f,
                           1.0f, 0.0f);
    box = b2d_create_box(80.0f, 80.0f, x, y, angle, 1.0f, 0.8f, 0.0f);
    if (floor == nullptr || box == nullptr) {
      throw std::runtime_error("body creation failed");
    }
    b2d_set_user_data(floor, reinterpret_cast<void*>(1));
    b2d_set_user_data(box, reinterpret_cast<void*>(2));
    // The wrapper accepts pixel velocity and stores native world velocity.
    b2d_set_v(box, velocity_x * 100.0f, velocity_y * 100.0f);
  }

  ~World() { b2d_dispose(); }

  World(const World&) = delete;
  World& operator=(const World&) = delete;
};

PublicState public_state(void* body) {
  PublicState result;
  result.x = b2d_get_x(body);
  result.y = b2d_get_y(body);
  result.angle = b2d_get_r(body);
  b2d_get_v(body, &result.velocity_x, &result.velocity_y);
  return result;
}

Frame step(World& world) {
  b2d_step(0.02f, 10);
  Frame result;
  result.state = public_state(world.box);
  void* first{};
  void* second{};
  while (b2d_get_contact(&first, &second)) {
    result.contacts.emplace_back(reinterpret_cast<std::uintptr_t>(first),
                                 reinterpret_cast<std::uintptr_t>(second));
  }
  return result;
}

bool same_bits(float left, float right) {
  return std::bit_cast<std::uint32_t>(left) ==
         std::bit_cast<std::uint32_t>(right);
}

bool same_state(const PublicState& left, const PublicState& right) {
  return same_bits(left.x, right.x) && same_bits(left.y, right.y) &&
         same_bits(left.angle, right.angle) &&
         same_bits(left.velocity_x, right.velocity_x) &&
         same_bits(left.velocity_y, right.velocity_y);
}

std::string mismatch_field(const PublicState& left,
                           const PublicState& right) {
  if (!same_bits(left.x, right.x)) return "x";
  if (!same_bits(left.y, right.y)) return "y";
  if (!same_bits(left.angle, right.angle)) return "angle";
  if (!same_bits(left.velocity_x, right.velocity_x)) return "velocity_x";
  if (!same_bits(left.velocity_y, right.velocity_y)) return "velocity_y";
  return "contacts";
}

}  // namespace

int main() {
  try {
    const std::uint16_t control_word = 0x027fU;
    __asm__ __volatile__("fldcw %0" : : "m"(control_word));
    constexpr int future_ticks = 80;
    PublicState checkpoint;
    std::vector<Frame> expected;
    int checkpoint_tick{};
    {
      World original(-120.0f, 250.0f, 0.55f, 1.5f, 0.0f);
      for (int tick = 1; tick <= 500; ++tick) {
        const Frame frame = step(original);
        if (!frame.contacts.empty()) {
          checkpoint_tick = tick;
          checkpoint = frame.state;
          break;
        }
      }
      if (checkpoint_tick == 0) {
        throw std::runtime_error("fixture never reached the floor");
      }
      expected.reserve(future_ticks);
      for (int tick = 0; tick < future_ticks; ++tick) {
        expected.push_back(step(original));
      }
    }

    int first_state_mismatch{};
    int first_contact_mismatch{};
    std::string field;
    {
      World rebuilt(checkpoint.x, checkpoint.y, checkpoint.angle,
                    checkpoint.velocity_x, checkpoint.velocity_y);
      for (int tick = 1; tick <= future_ticks; ++tick) {
        const Frame actual = step(rebuilt);
        const Frame& wanted = expected[static_cast<std::size_t>(tick - 1)];
        if (first_state_mismatch == 0 &&
            !same_state(actual.state, wanted.state)) {
          first_state_mismatch = tick;
          field = mismatch_field(actual.state, wanted.state);
        }
        if (first_contact_mismatch == 0 &&
            actual.contacts != wanted.contacts) {
          first_contact_mismatch = tick;
        }
      }
    }

    const bool diverged = first_state_mismatch != 0 ||
                          first_contact_mismatch != 0;
    std::cout << "{\"schema\":1,\"checkpoint_tick\":"
              << checkpoint_tick << ",\"future_ticks\":" << future_ticks
              << ",\"public_rebuild_diverged\":"
              << (diverged ? "true" : "false")
              << ",\"first_state_mismatch\":" << first_state_mismatch
              << ",\"first_contact_mismatch\":"
              << first_contact_mismatch << ",\"first_state_field\":\""
              << field << "\"}\n";
    return diverged ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
