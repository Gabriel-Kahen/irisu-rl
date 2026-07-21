#include "irisu/floating_point.hpp"
#include "irisu/normal_rules.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>

#if defined(__SSE__) || defined(_M_X64) ||                                     \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#define IRISU_TEST_HAS_MXCSR 1
#endif

namespace {

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
    assert(std::fegetenv(&original_) == 0);
    assert(std::feclearexcept(FE_ALL_EXCEPT) == 0);
    assert(std::fesetround(FE_DOWNWARD) == 0);
#if defined(IRISU_GNU_X87_CONTROL_WORD_ENVIRONMENT)
    std::uint16_t control{};
    __asm__ __volatile__("fnstcw %0" : "=m"(control));
    control = static_cast<std::uint16_t>(control & ~0x0300U);
    __asm__ __volatile__("fldcw %0" : : "m"(control));
#endif
    assert(std::feraiseexcept(FE_INEXACT) == 0);
    expected_ = floating_point_state();
  }

  ~HostileFloatingPointEnvironment() { (void)std::fesetenv(&original_); }

  const FloatingPointState &expected() const { return expected_; }

private:
  std::fenv_t original_{};
  FloatingPointState expected_{};
};

bool same_level_parameters(const irisu::NormalLevelParameters &left,
                           const irisu::NormalLevelParameters &right) {
  return left.passive_drain_unit == right.passive_drain_unit &&
         std::bit_cast<std::uint64_t>(left.scripted_descent_per_update) ==
             std::bit_cast<std::uint64_t>(right.scripted_descent_per_update) &&
         left.rot_penalty == right.rot_penalty &&
         left.maximum_color_id == right.maximum_color_id &&
         left.spawn_interval_frames == right.spawn_interval_frames &&
         std::bit_cast<std::uint64_t>(left.score_scale) ==
             std::bit_cast<std::uint64_t>(right.score_scale) &&
         left.clear_reward_unit == right.clear_reward_unit;
}

void canonical_scope_uses_original_x87_pc53() {
  const auto caller = floating_point_state();
  {
    const irisu::ScopedFloatingPointEnvironment environment;
    assert(std::fegetround() == FE_TONEAREST);
#if defined(IRISU_GNU_X87_CONTROL_WORD_ENVIRONMENT)
    std::uint16_t control{};
    __asm__ __volatile__("fnstcw %0" : "=m"(control));
    assert(control == 0x027fU);
#endif
#if defined(IRISU_TEST_HAS_MXCSR)
    assert(_mm_getcsr() == 0x1f80U);
#endif
  }
  assert(floating_point_state() == caller);
}

std::uint64_t spawn_angle_table_hash() {
  std::uint64_t hash = 14695981039346656037ULL;
  for (std::uint32_t ticket = 0; ticket <= 1000; ++ticket) {
    const auto bits = std::bit_cast<std::uint32_t>(
        static_cast<float>(irisu::normal_spawn_angle(ticket)));
    for (unsigned shift = 0; shift < 32; shift += 8) {
      hash ^= (bits >> shift) & 0xffU;
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

void level_formulas() {
  const auto one = irisu::normal_level_parameters(1);
  assert(one.passive_drain_unit == 1);
  assert(std::bit_cast<std::uint32_t>(
             static_cast<float>(one.scripted_descent_per_update)) ==
         0x3e4ccccdU);
  assert(one.rot_penalty == 1820);
  assert(one.maximum_color_id == 2);
  assert(one.spawn_interval_frames == 90);
  assert(one.clear_reward_unit == 700);

  const auto descent_bits = [](std::uint32_t level) {
    return std::bit_cast<std::uint32_t>(static_cast<float>(
        irisu::normal_level_parameters(level).scripted_descent_per_update));
  };
  assert(descent_bits(4) == 0x3eb33334U);

  const auto six = irisu::normal_level_parameters(6);
  assert(six.scripted_descent_per_update == 1.0);
  assert(six.spawn_interval_frames == 40);
  const auto seven = irisu::normal_level_parameters(7);
  assert(std::bit_cast<std::uint32_t>(
             static_cast<float>(seven.scripted_descent_per_update)) ==
         0x3e266667U);
  assert(descent_bits(9) == 0x3e866667U);
  assert(descent_bits(14) == 0x3e333334U);
  assert(descent_bits(15) == 0x3e666667U);
  std::uint64_t descent_table_hash = 14695981039346656037ULL;
  for (std::uint32_t level = 1; level <= 99; ++level) {
    const auto bits = descent_bits(level);
    for (unsigned shift = 0; shift < 32; shift += 8) {
      descent_table_hash ^= (bits >> shift) & 0xffU;
      descent_table_hash *= 1099511628211ULL;
    }
  }
  assert(descent_table_hash == 0x3bd9453538576485ULL);
  const auto thirteen = irisu::normal_level_parameters(13);
  assert(thirteen.spawn_interval_frames == 4);
  const auto twenty = irisu::normal_level_parameters(20);
  assert(descent_bits(20) == 0x4019999aU);
  assert(twenty.spawn_interval_frames == 100);
  const auto thirty_eight = irisu::normal_level_parameters(38);
  assert(std::bit_cast<std::uint32_t>(
             static_cast<float>(thirty_eight.scripted_descent_per_update)) ==
         0x4019999aU);
  assert(thirty_eight.spawn_interval_frames == 50);
  const auto ninety_nine = irisu::normal_level_parameters(99);
  assert(std::bit_cast<std::uint32_t>(
             static_cast<float>(ninety_nine.scripted_descent_per_update)) ==
         0x3ec00000U);
  bool rejected = false;
  try {
    (void)irisu::normal_level_parameters(100);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  assert(rejected);
}

void score_and_seed_formulas() {
  assert(irisu::normal_score_delta(1, 1, 1, 0) == 4);
  assert(irisu::normal_score_delta(1, 1, 1, 5) == 4);
  assert(irisu::normal_score_delta(1, 1, 1, 6) == 8);
  const auto level_two = irisu::normal_level_parameters(2);
  assert(std::bit_cast<std::uint64_t>(level_two.score_scale) ==
         0x4019fdf8bcce533dULL);
  assert(std::bit_cast<std::uint64_t>(
             irisu::normal_level_parameters(3).score_scale) ==
         0x402142e81c889914ULL);
  assert(std::bit_cast<std::uint64_t>(
             irisu::normal_level_parameters(4).score_scale) ==
         0x40251cb453b9536cULL);
  assert(std::bit_cast<std::uint64_t>(
             irisu::normal_level_parameters(7).score_scale) ==
         0x402f3c7995604382ULL);
  assert(std::bit_cast<std::uint64_t>(
             irisu::normal_level_parameters(99).score_scale) ==
         0x4058f15933c24051ULL);
  std::uint64_t table_hash = 14695981039346656037ULL;
  for (std::uint32_t level = 1; level <= 99; ++level) {
    const auto bits = std::bit_cast<std::uint64_t>(
        irisu::normal_level_parameters(level).score_scale);
    for (unsigned shift = 0; shift < 64; shift += 8) {
      table_hash ^= (bits >> shift) & 0xffU;
      table_hash *= 1099511628211ULL;
    }
  }
  assert(table_hash == 0xbbe0275e3e261123ULL);
  assert(irisu::normal_score_delta(2, 1, 25, 0) == 162);
  assert(irisu::normal_spawn_angle(0) == 0.0);
  assert(std::bit_cast<std::uint32_t>(
             static_cast<float>(irisu::normal_spawn_angle(500))) ==
         0x40490fdbU);
  assert(std::bit_cast<std::uint32_t>(
             static_cast<float>(irisu::normal_spawn_angle(1000))) ==
         0x40c90fdbU);
  assert(spawn_angle_table_hash() == 0xdff3ca0ea6fd0e37ULL);
  assert(irisu::normal_replay_seed(123) == 123U);
  assert(irisu::normal_replay_seed(-123) == static_cast<std::uint32_t>(-123));
  assert(irisu::normal_replay_seed(std::numeric_limits<std::int32_t>::min()) == 0U);
}

void public_helpers_isolate_hostile_floating_point_state() {
  const auto expected_level = irisu::normal_level_parameters(3);
  const auto expected_score =
      irisu::normal_score_delta(2, 100'000'003U, 100'000'003U, 0);
  const auto expected_angle = irisu::normal_spawn_angle(500);
  const auto expected_angle_table_hash = spawn_angle_table_hash();
  assert(expected_score == 32'490'098'128'556'168LL);

  HostileFloatingPointEnvironment hostile;
  const auto caller = hostile.expected();
  assert(same_level_parameters(irisu::normal_level_parameters(3),
                               expected_level));
  assert(floating_point_state() == caller);
  assert(irisu::normal_score_delta(2, 100'000'003U, 100'000'003U, 0) ==
         expected_score);
  assert(floating_point_state() == caller);
  const auto actual_angle = irisu::normal_spawn_angle(500);
  const auto actual_angle_bits = std::bit_cast<std::uint64_t>(actual_angle);
  const auto expected_angle_bits = std::bit_cast<std::uint64_t>(expected_angle);
  if (actual_angle_bits != expected_angle_bits) {
    std::cerr << "hostile angle bits 0x" << std::hex << actual_angle_bits
              << " expected 0x" << expected_angle_bits << std::dec << '\n';
  }
  assert(actual_angle_bits == expected_angle_bits);
  assert(floating_point_state() == caller);
  assert(spawn_angle_table_hash() == expected_angle_table_hash);
  assert(floating_point_state() == caller);

  bool rejected = false;
  try {
    (void)irisu::normal_level_parameters(100);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  assert(rejected);
  assert(floating_point_state() == caller);
}

}  // namespace

int main() {
  canonical_scope_uses_original_x87_pc53();
  level_formulas();
  score_and_seed_formulas();
  public_helpers_isolate_hostile_floating_point_state();
  std::cout << "normal rule formula tests passed\n";
}
