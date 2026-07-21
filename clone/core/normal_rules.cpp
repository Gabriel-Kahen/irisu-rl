#include "irisu/normal_rules.hpp"

#include "irisu/floating_point.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace irisu {
namespace {

// Exact float stores recovered from the v2.03 level rule. Keeping the stores
// as bits avoids an x87 PC53 double-rounding difference in the source formula
// on 32-bit hosts. Entries include the level-%6 and levels 20/38/... overrides.
constexpr std::array<std::uint32_t, 99> kScriptedDescentBits{
    0x3e4ccccdU, 0x3e800000U, 0x3e99999aU, 0x3eb33334U, 0x3ecccccdU,
    0x3f800000U, 0x3e266667U, 0x3e59999aU, 0x3e866667U, 0x3ea00000U,
    0x3eb9999aU, 0x3f800000U, 0x3eeccccdU, 0x3e333334U, 0x3e666667U,
    0x3e8ccccdU, 0x3ea66667U, 0x3f800000U, 0x3ed9999aU, 0x4019999aU,
    0x3e400000U, 0x3e733334U, 0x3e933333U, 0x3f800000U, 0x3ec66667U,
    0x3ee00000U, 0x3ef9999aU, 0x3e4ccccdU, 0x3e800000U, 0x3f800000U,
    0x3eb33334U, 0x3ecccccdU, 0x3ee66667U, 0x3f000000U, 0x3e59999aU,
    0x3f800000U, 0x3ea00000U, 0x4019999aU, 0x3ed33334U, 0x3eeccccdU,
    0x3f033333U, 0x3f800000U, 0x3e8ccccdU, 0x3ea66667U, 0x3ec00000U,
    0x3ed9999aU, 0x3ef33334U, 0x3f800000U, 0x3e733334U, 0x3e933333U,
    0x3eaccccdU, 0x3ec66667U, 0x3ee00000U, 0x3f800000U, 0x3f09999aU,
    0x4019999aU, 0x3e99999aU, 0x3eb33334U, 0x3ecccccdU, 0x3f800000U,
    0x3f000000U, 0x3f0ccccdU, 0x3e866667U, 0x3ea00000U, 0x3eb9999aU,
    0x3f800000U, 0x3eeccccdU, 0x3f033333U, 0x3f100000U, 0x3e8ccccdU,
    0x3ea66667U, 0x3f800000U, 0x3ed9999aU, 0x4019999aU, 0x3f066667U,
    0x3f133333U, 0x3e933333U, 0x3f800000U, 0x3ec66667U, 0x3ee00000U,
    0x3ef9999aU, 0x3f09999aU, 0x3f166667U, 0x3f800000U, 0x3eb33334U,
    0x3ecccccdU, 0x3ee66667U, 0x3f000000U, 0x3f0ccccdU, 0x3f800000U,
    0x3ea00000U, 0x4019999aU, 0x3ed33334U, 0x3eeccccdU, 0x3f033333U,
    0x3f800000U, 0x3f1ccccdU, 0x3ea66667U, 0x3ec00000U,
};

// Exact QWORD stores observed immediately after v2.03 setter 0x4157be.
constexpr std::array<std::uint64_t, 99> kScoreScaleBits{
    0x4010000000000000ULL, 0x4019fdf8bcce533dULL, 0x402142e81c889914ULL,
    0x40251cb453b9536cULL, 0x4028ae6d3fc447a1ULL, 0x402c0a88fb9717f1ULL,
    0x402f3c7995604382ULL, 0x403125fbee250664ULL, 0x40329f4504b4e7e6ULL,
    0x40340c28430012e7ULL, 0x40356e3d8cca85afULL, 0x4036c6c846ab7ca7ULL,
    0x403816ce65cfe3c4ULL, 0x40395f27eaf32c5bULL, 0x403aa089a39ba854ULL,
    0x403bdb8cdadbe120ULL, 0x403d10b4fc1eec63ULL, 0x403e4073cb226e4fULL,
    0x403f6b2c9b687921ULL, 0x4040489b672a7d8dULL, 0x4040d96fe46af637ULL,
    0x40416836416e33c9ULL, 0x4041f50d649383d7ULL, 0x4042801121978c6cULL,
    0x4043095aa530dce5ULL, 0x40439100ce4040adULL, 0x404417187852199dULL,
    0x40449bb4ba4f8b2fULL, 0x40451ee71b998f87ULL, 0x4045a0bfc14cad7dULL,
    0x4046214d950e908dULL, 0x4046a09e667f3bcdULL, 0x40471ebf08304cb3ULL,
    0x40479bbb68d9ad8bULL, 0x4048179ea9613a93ULL, 0x4048927330300a96ULL,
    0x40490c42ba3aa4fcULL, 0x404985166a103ec4ULL, 0x4049fcf6d5373157ULL,
    0x404a73ec101190a5ULL, 0x404ae9fdb87b85feULL, 0x404b5f32ff4d7a6aULL,
    0x404bd392b0e5d159ULL, 0x404c47233cd8bb74ULL, 0x404cb9eabce04d70ULL,
    0x404d2beefb235f59ULL, 0x404d9d3577e6a580ULL, 0x404e0dc36eb8dad9ULL,
    0x404e7d9ddb28a02cULL, 0x404eecc97d10d2ccULL, 0x404f5b4adc868348ULL,
    0x404fc9264d72541eULL, 0x40501b2ff96eed30ULL, 0x4050517de0fe4a4cULL,
    0x4050877ec27b956aULL, 0x4050bd346ebf7347ULL, 0x4050f2a0a397535aULL,
    0x405127c50cdeffc1ULL, 0x40515ca3458559deULL, 0x4051913cd87e217cULL,
    0x4051c59341a2723eULL, 0x4051f9a7ee817706ULL, 0x40522d7c3f22ac1cULL,
    0x4052611186bae675ULL, 0x405294690c5537faULL, 0x4052c7840b70ada3ULL,
    0x4052fa63b493cc19ULL, 0x40532d092dd69a27ULL, 0x40535f759363f51eULL,
    0x405391a9f7f2da08ULL, 0x4053c3a765383f5aULL, 0x4053f56edc520cf8ULL,
    0x40542701562bb3e1ULL, 0x4054585fc3dcdbf0ULL, 0x4054898b0f0293cbULL,
    0x4054ba841a136649ULL, 0x4054eb4bc0aeb020ULL, 0x40551be2d7e7898aULL,
    0x40554c4a2e8b907eULL, 0x40557c828d65da56ULL, 0x4055ac8cb77e4ddfULL,
    0x4055dc696a55a1f6ULL, 0x40560c195e1e3824ULL, 0x40563b9d45f20673ULL,
    0x40566af5d005bfedULL, 0x40569a23a5d967b3ULL, 0x4056c9276c667751ULL,
    0x4056f801c44bbe14ULL, 0x405726b349f71c7dULL, 0x4057553c95cd3c54ULL,
    0x4057839e3c4f63b2ULL, 0x4057b1d8ce3f7f32ULL, 0x4057dfecd8c27d8fULL,
    0x40580ddae5811544ULL, 0x40583ba37ac70adeULL, 0x405869471ba10d87ULL,
    0x405896c647f93da4ULL, 0x4058c4217cb27030ULL, 0x4058f15933c24051ULL,
};

}  // namespace

NormalLevelParameters normal_level_parameters(std::uint32_t level) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  if (level == 0 || level > kScoreScaleBits.size()) {
    throw std::invalid_argument("normal level parameters exist only for 1..99");
  }

  NormalLevelParameters result;
  result.passive_drain_unit = level / 10U + 1U;
  result.scripted_descent_per_update = static_cast<double>(
      std::bit_cast<float>(kScriptedDescentBits[level - 1U]));
  result.rot_penalty = 1'800 + 20 * static_cast<std::int64_t>(level);
  result.maximum_color_id = std::min<std::uint32_t>(
      5U, (((level % 9U) + level / 15U) / 3U) + 2U);
  result.spawn_interval_frames = 100U - 10U * (level % 10U);
  if (level % 13U == 0U) result.spawn_interval_frames = 4U;
  if (result.scripted_descent_per_update > 2.0 &&
      result.spawn_interval_frames < 50U) {
    result.spawn_interval_frames = 50U;
  }
  result.score_scale = std::bit_cast<double>(kScoreScaleBits[level - 1U]);
  result.clear_reward_unit = 700;
  return result;
}

std::int64_t normal_score_delta(std::uint32_t level, std::uint32_t group_num,
                                std::uint32_t group_chain,
                                std::uint32_t size_slot) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  const auto parameters = normal_level_parameters(level);
  const auto factor = std::max<std::int64_t>(
      1, static_cast<std::int64_t>(size_slot) - 4);
  // The executable runs this x87 path at 53-bit precision. Preserve the
  // operation sequence and binary64 rounding boundaries explicitly.
  double exact = static_cast<double>(group_num);
  exact *= 0.5;
  exact += 0.5;
  exact *= static_cast<double>(group_chain);
  exact *= parameters.score_scale;
  exact *= static_cast<double>(factor);
  if (!std::isfinite(exact) || exact >= 0x1p63) {
    throw std::overflow_error("normal score delta overflow");
  }
  return static_cast<std::int64_t>(exact);
}

double normal_spawn_angle(std::uint32_t rotation_ticket) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  constexpr long double tau = 0x1.921fb54442d1846ap+2L;
  const volatile auto stored = static_cast<float>(
      static_cast<long double>(rotation_ticket) * tau / 1000.0L);
  return static_cast<double>(stored);
}

std::uint32_t normal_replay_seed(std::int32_t now_count) {
  const std::int32_t sign = now_count >> 31;
  const std::uint32_t magnitude = now_count == std::numeric_limits<std::int32_t>::min()
                                      ? 0x80000000U
                                      : static_cast<std::uint32_t>(
                                            std::abs(now_count));
  const std::uint32_t masked = magnitude & 0x3fffffffU;
  return (masked ^ static_cast<std::uint32_t>(sign)) -
         static_cast<std::uint32_t>(sign);
}

}  // namespace irisu
