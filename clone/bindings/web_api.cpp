#include "irisu/c_api.h"

#include <string>
#include <string_view>

namespace {

thread_local std::string result;

const char* reset_result(irisu_simulator* simulator, uint32_t seed) {
  if (!irisu_reset(simulator, seed)) return nullptr;
  const char* observation = irisu_observation_json(simulator);
  if (observation == nullptr) return nullptr;
  result = "{\"observation\":";
  result += observation;
  result += ",\"events\":[]}";
  return result.c_str();
}

}  // namespace

extern "C" {

irisu_simulator* irisu_web_create(uint32_t seed) {
  irisu_simulator* simulator = irisu_create();
  if (simulator != nullptr && reset_result(simulator, seed) == nullptr) {
    irisu_destroy(simulator);
    return nullptr;
  }
  return simulator;
}

void irisu_web_destroy(irisu_simulator* simulator) {
  irisu_destroy(simulator);
}

const char* irisu_web_reset(irisu_simulator* simulator, uint32_t seed) {
  return reset_result(simulator, seed);
}

const char* irisu_web_step(irisu_simulator* simulator, int action_kind,
                           double x, double y) {
  if (!irisu_step(simulator, action_kind, x, y, 1)) return nullptr;
  const char* observation = irisu_observation_json(simulator);
  const char* transition = irisu_step_json(simulator);
  if (observation == nullptr || transition == nullptr) return nullptr;

  const std::string_view step{transition};
  const std::size_t events = step.find("\"events\":");
  const std::size_t diagnostics = step.find(",\"diagnostics\":", events);
  if (events == std::string_view::npos || diagnostics == std::string_view::npos)
    return nullptr;
  result = "{\"observation\":";
  result += observation;
  result += ',';
  result.append(step.substr(events, diagnostics - events));
  result += '}';
  return result.c_str();
}

}  // extern "C"
