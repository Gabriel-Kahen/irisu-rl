#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace {

constexpr char kExactSoname[] =
    "libirisu_box2d_msvc_exact_multiworld.so";
constexpr char kMappedMarker[] =
    "IRISU_TEST_GENUINE_EXACT_SONAME_MAPPED\n";

bool exact_soname_is_mapped() noexcept {
  int descriptor = ::open("/proc/self/maps", O_RDONLY | O_CLOEXEC);
  if (descriptor < 0) return false;
  std::array<char, 131072> contents{};
  std::size_t size{};
  while (size + 1 < contents.size()) {
    const ssize_t count =
        ::read(descriptor, contents.data() + size, contents.size() - size - 1);
    if (count > 0) {
      size += static_cast<std::size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR) continue;
    break;
  }
  ::close(descriptor);
  contents[size] = '\0';
  return std::strstr(contents.data(), kExactSoname) != nullptr;
}

__attribute__((constructor)) void report_exact_mapping() noexcept {
  if (exact_soname_is_mapped()) {
    static_cast<void>(
        ::write(STDERR_FILENO, kMappedMarker, sizeof(kMappedMarker) - 1));
  }
}

}  // namespace

extern "C" void b2d_world_step(void*, float, int) { ::_exit(86); }
