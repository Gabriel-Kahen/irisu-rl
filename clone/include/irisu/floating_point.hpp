#pragma once

#include <cfenv>
#include <cstdint>

#if defined(__SSE__) || defined(_M_X64) ||                                     \
    (defined(_M_IX86_FP) && _M_IX86_FP >= 1)
#include <xmmintrin.h>
#define IRISU_HAS_MXCSR 1
#endif

namespace irisu {
namespace detail {

inline thread_local std::uint32_t floating_point_scope_depth = 0;

inline void install_canonical_floating_point_environment() noexcept {
  (void)std::fesetround(FE_TONEAREST);
#if defined(IRISU_GNU_X87_CONTROL_WORD_ENVIRONMENT)
  // 53-bit precision, round-to-nearest, and all exceptions masked. This is
  // the original game's live x87 control word after DxLib initialization.
  const std::uint16_t control_word = 0x027fU;
  __asm__ __volatile__("fldcw %0" : : "m"(control_word));
#endif
#if defined(IRISU_HAS_MXCSR)
  // Round-to-nearest, gradual underflow, masked exceptions, and clear flags.
  _mm_setcsr(0x1f80U);
#endif
}

} // namespace detail

// Floating-point state is thread-local on supported hosts. Only the outermost
// nested scope touches it. A scope that cannot first capture the caller state
// leaves it unchanged; otherwise the full environment is restored even when
// the guarded operation throws.
class ScopedFloatingPointEnvironment {
public:
  ScopedFloatingPointEnvironment() noexcept
      : outermost_(detail::floating_point_scope_depth++ == 0) {
    if (!outermost_)
      return;
    saved_ = std::fegetenv(&environment_) == 0;
    if (!saved_)
      return;
    (void)std::feclearexcept(FE_ALL_EXCEPT);
    detail::install_canonical_floating_point_environment();
  }

  ~ScopedFloatingPointEnvironment() noexcept {
    --detail::floating_point_scope_depth;
    if (outermost_ && saved_)
      (void)std::fesetenv(&environment_);
  }

  ScopedFloatingPointEnvironment(const ScopedFloatingPointEnvironment &) =
      delete;
  ScopedFloatingPointEnvironment &
  operator=(const ScopedFloatingPointEnvironment &) = delete;

private:
  std::fenv_t environment_{};
  bool outermost_{};
  bool saved_{};
};

} // namespace irisu

#if defined(IRISU_HAS_MXCSR)
#undef IRISU_HAS_MXCSR
#endif
