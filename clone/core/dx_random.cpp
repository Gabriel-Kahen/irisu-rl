#include "irisu/dx_random.hpp"

#include <stdexcept>

namespace irisu {
namespace {

constexpr std::size_t kPeriodOffset = 397;
constexpr std::uint32_t kMatrixA = 0x9908b0dfU;

}  // namespace

void DxRandom::seed(std::uint32_t value) {
  for (auto& word : state_) {
    const std::uint32_t following = 69'069U * value + 1U;
    word = (value & 0xffff0000U) | (following >> 16U);
    value = 69'069U * following + 1U;
  }
  index_ = state_words;
}

void DxRandom::twist() {
  for (std::size_t index = 0; index < state_words - kPeriodOffset; ++index) {
    const std::uint32_t joined =
        (state_[index] & 0x80000000U) | (state_[index + 1] & 0x7fffffffU);
    state_[index] = state_[index + kPeriodOffset] ^ (joined >> 1U) ^
                    ((joined & 1U) != 0U ? kMatrixA : 0U);
  }
  for (std::size_t index = state_words - kPeriodOffset;
       index < state_words - 1; ++index) {
    const std::uint32_t joined =
        (state_[index] & 0x80000000U) | (state_[index + 1] & 0x7fffffffU);
    state_[index] = state_[index + kPeriodOffset - state_words] ^
                    (joined >> 1U) ^ ((joined & 1U) != 0U ? kMatrixA : 0U);
  }
  const std::uint32_t joined =
      (state_.back() & 0x80000000U) | (state_.front() & 0x7fffffffU);
  state_.back() = state_[kPeriodOffset - 1] ^ (joined >> 1U) ^
                  ((joined & 1U) != 0U ? kMatrixA : 0U);
  index_ = 0;
}

std::uint32_t DxRandom::raw_u32() {
  if (index_ >= state_words) twist();
  std::uint32_t value = state_[index_++];
  value ^= value >> 11U;
  value ^= (value << 7U) & 0x9d2c5680U;
  value ^= (value << 15U) & 0xefc60000U;
  value ^= value >> 18U;
  return value;
}

std::uint32_t DxRandom::get_rand(std::uint32_t maximum) {
  if (maximum > 0x7fffffffU) {
    throw std::invalid_argument("DxLib GetRand maximum exceeds signed int32");
  }
  return static_cast<std::uint32_t>(
      (static_cast<std::uint64_t>(raw_u32()) *
       (static_cast<std::uint64_t>(maximum) + 1U)) >>
      32U);
}

void DxRandom::restore(
    const std::array<std::uint32_t, state_words>& state,
    std::uint32_t index) {
  if (index > state_words) {
    throw std::invalid_argument("DxLib RNG index is outside [0,624]");
  }
  state_ = state;
  index_ = index;
}

}  // namespace irisu
