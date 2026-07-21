#pragma once

#include "irisu/config.hpp"

#include <cstdint>
#include <string>
#include <string_view>

namespace irisu {

// Shared by the C ABI and out-of-process exact worker so both accept exactly
// the same flattened override keys and numeric domain.
void apply_config_override(MechanicsConfig& config, std::string_view key,
                           double value);

// Canonical public readback used by all ABI surfaces.
[[nodiscard]] std::string mechanics_config_json(const MechanicsConfig& config,
                                                std::uint64_t config_hash);

}  // namespace irisu
