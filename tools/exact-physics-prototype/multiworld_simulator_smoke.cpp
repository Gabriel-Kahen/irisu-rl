#include "irisu/simulator.hpp"

#include <cassert>
#include <cstddef>
#include <thread>

namespace {

void advance(irisu::Simulator& simulator, int count) {
  for (int step = 0; step < count; ++step) {
    simulator.step({irisu::ActionKind::Wait, 0.0, 0.0, 1});
  }
}

void require_same(const irisu::Observation& left,
                  const irisu::Observation& right) {
  assert(left.tick == right.tick);
  assert(left.score == right.score);
  assert(left.gauge == right.gauge);
  assert(left.level == right.level);
  assert(left.bodies.size() == right.bodies.size());
  for (std::size_t index = 0; index < left.bodies.size(); ++index) {
    const auto& a = left.bodies[index];
    const auto& b = right.bodies[index];
    assert(a.id == b.id);
    assert(a.position.x == b.position.x);
    assert(a.position.y == b.position.y);
    assert(a.velocity.x == b.velocity.x);
    assert(a.velocity.y == b.velocity.y);
    assert(a.angle == b.angle);
    assert(a.lifecycle == b.lifecycle);
  }
}

}  // namespace

int main() {
  irisu::Simulator concurrent_a;
  irisu::Simulator concurrent_b;
  irisu::Simulator baseline_a;
  irisu::Simulator baseline_b;
  concurrent_a.reset(123U);
  concurrent_b.reset(456U);
  baseline_a.reset(123U);
  baseline_b.reset(456U);

  std::thread first([&] { advance(concurrent_a, 256); });
  std::thread second([&] { advance(concurrent_b, 256); });
  first.join();
  second.join();
  advance(baseline_a, 256);
  advance(baseline_b, 256);

  require_same(concurrent_a.observation(), baseline_a.observation());
  require_same(concurrent_b.observation(), baseline_b.observation());
}
