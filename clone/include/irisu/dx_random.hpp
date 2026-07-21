#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace irisu {

// Clean-room model of the RNG exported by the v2.03 DxLib.dll. The range
// argument is an inclusive maximum, matching DxLib's GetRand contract.
class DxRandom {
 public:
  static constexpr std::size_t state_words = 624;

  void seed(std::uint32_t value);
  std::uint32_t raw_u32();
  std::uint32_t get_rand(std::uint32_t maximum);

  const std::array<std::uint32_t, state_words>& state() const { return state_; }
  std::uint32_t index() const { return index_; }
  void restore(const std::array<std::uint32_t, state_words>& state,
               std::uint32_t index);

 private:
  void twist();

  std::array<std::uint32_t, state_words> state_{};
  std::uint32_t index_{state_words};
};

}  // namespace irisu
