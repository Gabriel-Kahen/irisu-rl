#include <bit>
#include <cmath>
#include <cstdint>

extern "C" void msvc_CIcos();
extern "C" void msvc_CIsin();

namespace {

float reference_cos(float value) {
  float result{};
  asm volatile("flds %1\n\tfcos\n\tfstps %0"
               : "=m"(result)
               : "m"(value)
               : "st");
  return result;
}

float reference_sin(float value) {
  float result{};
  asm volatile("flds %1\n\tfsin\n\tfstps %0"
               : "=m"(result)
               : "m"(value)
               : "st");
  return result;
}

float hosted_cos(float value) {
  float result{};
  const auto function = &msvc_CIcos;
  asm volatile("flds %1\n\tcall *%2\n\tfstps %0"
               : "=m"(result)
               : "m"(value), "r"(function)
               : "eax", "ecx", "edx", "cc", "memory", "st");
  return result;
}

float hosted_sin(float value) {
  float result{};
  const auto function = &msvc_CIsin;
  asm volatile("flds %1\n\tcall *%2\n\tfstps %0"
               : "=m"(result)
               : "m"(value), "r"(function)
               : "eax", "ecx", "edx", "cc", "memory", "st");
  return result;
}

bool same_bits(float left, float right) {
  return std::bit_cast<std::uint32_t>(left) ==
         std::bit_cast<std::uint32_t>(right);
}

bool same_result(float left, float right) {
  return same_bits(left, right) || (std::isnan(left) && std::isnan(right));
}

bool check(float value, float other) {
  if (!same_result(hosted_sin(value), reference_sin(value))) return false;
  if (!same_result(hosted_cos(value), reference_cos(value))) return false;
  if (!same_result(hosted_sin(value), reference_sin(value))) return false;
  if (!same_result(hosted_cos(value), reference_cos(value))) return false;
  return same_result(hosted_sin(other), reference_sin(other));
}

}  // namespace

int main() {
  const std::uint16_t control_word = 0x027fU;
  asm volatile("fldcw %0" : : "m"(control_word));

  if (!check(0.0F, -0.0F) || !check(-0.0F, 1.0F) ||
      !check(3.1415927F, -3.1415927F)) {
    return 1;
  }

  constexpr std::uint32_t boundary_bits[] = {
      0x5effffffU, 0x5f000000U, 0x5f000001U, 0x7f7fffffU,
      0xdeffffffU, 0xdf000000U, 0xdf000001U, 0xff7fffffU,
      0x7f800000U, 0xff800000U, 0x7fc00001U, 0xffc00001U,
  };
  for (const auto bits : boundary_bits) {
    if (!check(std::bit_cast<float>(bits), 1.0F)) return 1;
  }

  std::uint64_t state = UINT64_C(0x9e3779b97f4a7c15);
  for (std::uint32_t index = 0; index < 100'000U; ++index) {
    state ^= state << 7U;
    state ^= state >> 9U;
    state ^= state << 8U;
    const auto bits = static_cast<std::uint32_t>(state);
    const float value = std::bit_cast<float>(bits);
    const float other = bits == 0x3f800000U ? 0.0F : 1.0F;
    if (!check(value, other)) return 1;
  }
  return 0;
}
