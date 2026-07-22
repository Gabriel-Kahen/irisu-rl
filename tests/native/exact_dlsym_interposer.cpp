#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <unistd.h>

#include <cstring>

namespace {

using DlsymFunction = void *(*)(void *, const char *);

bool armed;

DlsymFunction real_dlsym() {
  void *address = ::dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.0");
  DlsymFunction function{};
  static_assert(sizeof(function) == sizeof(address));
  std::memcpy(&function, &address, sizeof(function));
  return function;
}

__attribute__((constructor)) void report_loaded() {
  constexpr char message[] = "IRISU_TEST_DLSYM_INTERPOSER_LOADED\n";
  armed = true;
  static_cast<void>(::write(STDERR_FILENO, message, sizeof(message) - 1));
}

} // namespace

extern "C" void *dlsym(void *handle, const char *name) {
  const auto resolve = real_dlsym();
  if (resolve == nullptr)
    return nullptr;
  if (armed && std::strcmp(name, "b2d_world_get_x") == 0) {
    return resolve(handle, "b2d_world_get_y");
  }
  if (armed && std::strcmp(name, "b2d_world_get_y") == 0) {
    return resolve(handle, "b2d_world_get_x");
  }
  return resolve(handle, name);
}
