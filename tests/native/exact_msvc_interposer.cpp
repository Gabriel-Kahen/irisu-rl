#include <unistd.h>

namespace {

constexpr char kLoadedMarker[] = "IRISU_TEST_MSVC_INTERPOSER_LOADED\n";
constexpr char kCalledMarker[] = "IRISU_TEST_MSVC_INTERPOSER_CALLED\n";

__attribute__((constructor)) void report_loaded() noexcept {
  static_cast<void>(
      ::write(STDERR_FILENO, kLoadedMarker, sizeof(kLoadedMarker) - 1));
}

}  // namespace

extern "C" void msvc_b2d_world_step(void*, float, int) {
  static_cast<void>(
      ::write(STDERR_FILENO, kCalledMarker, sizeof(kCalledMarker) - 1));
  ::_exit(87);
}
