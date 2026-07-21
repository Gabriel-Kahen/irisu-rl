#include "irisu/simulator.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

void assert_same_body(const irisu::ObservedBody& left,
                      const irisu::ObservedBody& right) {
  assert(left.id == right.id);
  assert(left.kind == right.kind);
  assert(left.shape == right.shape);
  assert(left.lifecycle == right.lifecycle);
  assert(left.color == right.color);
  assert(left.position.x == right.position.x);
  assert(left.position.y == right.position.y);
  assert(left.velocity.x == right.velocity.x);
  assert(left.velocity.y == right.velocity.y);
  assert(left.angle == right.angle);
  assert(left.size == right.size);
}

}  // namespace

int main() {
  irisu::Simulator simulator;
  const auto first = simulator.reset(123U);
  assert(first.tick == 0);
  assert(first.score == 0);
  assert(first.gauge == 3'000);
  assert(first.bodies.size() == 20);

  const auto transition = simulator.step(
      {irisu::ActionKind::StrongShot, 300.0, 360.0, 1});
  assert(!transition.terminated);
  assert(simulator.observation().tick == 1);

  const auto repeated = simulator.reset(123U);
  assert(repeated.tick == first.tick);
  assert(repeated.score == first.score);
  assert(repeated.gauge == first.gauge);
  assert(repeated.level == first.level);
  assert(repeated.bodies.size() == first.bodies.size());
  for (std::size_t index = 0; index < first.bodies.size(); ++index) {
    assert_same_body(first.bodies[index], repeated.bodies[index]);
  }
}
