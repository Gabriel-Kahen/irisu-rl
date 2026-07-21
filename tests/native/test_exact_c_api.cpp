#include "irisu/c_api.h"

#include <cassert>
#include <cstdint>
#include <cstring>

int main() {
  assert(irisu_abi_version() == 1U);
  assert(irisu_padded_abi_version() == 1U);
  assert(std::strstr(irisu_build_info_json(),
                     "\"physics_backend\":\"exact-msvc9-r58-multiworld-forward\"") !=
         nullptr);
  assert(std::strstr(irisu_build_info_json(),
                     "\"exact_library_sha256\":\"not-applicable\"") ==
         nullptr);

  irisu_simulator* simulator = irisu_create();
  irisu_simulator* second = irisu_create();
  assert(simulator != nullptr);
  assert(second != nullptr);

  irisu_padded_observation_v1 initial{};
  irisu_padded_observation_v1 second_initial{};
  assert(irisu_padded_reset(simulator, 42U, &initial));
  assert(irisu_padded_reset(second, 42U, &second_initial));
  assert(initial.tick == 0U);
  assert(initial.score == 0);
  assert(initial.gauge == 3'000);
  assert(initial.body_count == 20U);
  assert(std::memcmp(&initial, &second_initial, sizeof(initial)) == 0);

  irisu_padded_transition_v1 transition{};
  assert(irisu_padded_step(simulator, IRISU_ACTION_KIND_STRONG_SHOT, 300.0,
                           360.0, 1U, &transition));
  assert(transition.observation.tick == 1U);
  assert(!transition.terminated);
  assert(!transition.truncated);

  irisu_padded_transition_v1 second_transition{};
  assert(irisu_padded_step(second, IRISU_ACTION_KIND_STRONG_SHOT, 300.0,
                           360.0, 1U, &second_transition));
  assert(std::memcmp(&transition, &second_transition, sizeof(transition)) == 0);

  const auto nominal_hash = irisu_config_hash(second);
  const irisu_config_override gravity{"gravity_y", 200.0};
  assert(irisu_configure(second, &gravity, 1U));
  assert(irisu_config_hash(second) != nominal_hash);
  assert(irisu_padded_reset(second, 42U, &second_initial));
  assert(irisu_padded_step(second, IRISU_ACTION_KIND_WAIT, 0.0, 0.0, 1U,
                           &second_transition));

  assert(irisu_snapshot_size(simulator) == 0U);
  assert(std::strstr(irisu_last_error(simulator),
                     "cannot snapshot contacts") != nullptr);

  irisu_destroy(second);
  irisu_destroy(simulator);
}
