#include "irisu/dx_random.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>

namespace {

constexpr std::array<std::uint32_t, 10> kMaxima{
    100, 12, 69, 404, 1000, 3, 100, 1000, 5, 100};

void measured_vectors() {
  constexpr std::array<std::uint32_t, 4> seeds{
      0, 1, 0x12345678U, 0x3fffffffU};
  constexpr std::array<std::array<std::uint32_t, 10>, 4> expected{{
      {11, 11, 36, 264, 666, 0, 28, 813, 0, 71},
      {83, 12, 3, 55, 782, 1, 82, 879, 1, 99},
      {47, 9, 0, 302, 168, 1, 78, 895, 0, 59},
      {64, 8, 6, 353, 16, 2, 2, 684, 5, 59},
  }};
  for (std::size_t seed_index = 0; seed_index < seeds.size(); ++seed_index) {
    irisu::DxRandom random;
    random.seed(seeds[seed_index]);
    for (std::size_t draw = 0; draw < kMaxima.size(); ++draw) {
      assert(random.get_rand(kMaxima[draw]) == expected[seed_index][draw]);
    }
  }
}

void full_state_restores_future() {
  irisu::DxRandom random;
  random.seed(0xdecafbadU);
  for (int index = 0; index < 700; ++index) (void)random.raw_u32();
  const auto state = random.state();
  const auto index = random.index();
  std::array<std::uint32_t, 1000> expected{};
  for (auto& value : expected) value = random.raw_u32();

  irisu::DxRandom restored;
  restored.restore(state, index);
  for (const auto value : expected) assert(restored.raw_u32() == value);
}

}  // namespace

int main() {
  measured_vectors();
  full_state_restores_future();
  std::cout << "DxLib RNG tests passed\n";
}
