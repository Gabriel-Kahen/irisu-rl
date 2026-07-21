#pragma once

#include <cstdint>

namespace irisu {

struct NormalLevelParameters {
  std::uint32_t passive_drain_unit{};
  double scripted_descent_per_update{};
  std::int64_t rot_penalty{};
  std::uint32_t maximum_color_id{};
  std::uint32_t spawn_interval_frames{};
  double score_scale{};
  std::int64_t clear_reward_unit{};
};

// The executable bypasses parameter calculation at the level-100 finish path.
NormalLevelParameters normal_level_parameters(std::uint32_t level);  // 1..99

// Exact recovered per-block scoring path.
std::int64_t normal_score_delta(std::uint32_t level, std::uint32_t group_num,
                                std::uint32_t group_chain,
                                std::uint32_t size_slot);

// Original x87 computation followed by the Block's float32 angle store.
double normal_spawn_angle(std::uint32_t rotation_ticket);

std::uint32_t normal_replay_seed(std::int32_t now_count);

}  // namespace irisu
