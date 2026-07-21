#include "irisu/simulator.hpp"

#include <charconv>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

void print_state(const irisu::Simulator& simulator, const irisu::StepResult* step = nullptr) {
  const auto state = simulator.observation();
  std::cout << "{\"tick\":" << state.tick << ",\"score\":" << state.score
            << ",\"gauge\":" << state.gauge << ",\"level\":" << state.level
            << ",\"terminated\":" << (state.terminated ? "true" : "false")
            << ",\"truncated\":" << (state.truncated ? "true" : "false")
            << ",\"state_hash\":";
  try {
    std::cout << simulator.state_hash();
  } catch (const std::logic_error&) {
    std::cout << "null";
  }
  std::cout << ",\"body_count\":" << state.bodies.size();
  if (step != nullptr) {
    std::cout << ",\"reward\":" << step->reward << ",\"event_count\":"
              << step->events.size();
  }
  std::cout << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::uint32_t seed = 1;
  if (argc == 3 && std::string(argv[1]) == "--seed") {
    const std::string value = argv[2];
    const auto result = std::from_chars(value.data(), value.data() + value.size(), seed);
    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
      std::cerr << "invalid seed\n";
      return 2;
    }
  } else if (argc != 1) {
    std::cerr << "usage: irisu-headless [--seed UINT32]\n"
                 "stdin actions: wait TICKS | weak X Y | strong X Y | both X Y | state | quit\n";
    return 2;
  }

  irisu::Simulator simulator;
  simulator.reset(seed);
  print_state(simulator);
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream input(line);
    std::string command;
    input >> command;
    if (command.empty()) continue;
    if (command == "quit") break;
    if (command == "state") { print_state(simulator); continue; }
    irisu::Action action;
    if (command == "wait") {
      action.kind = irisu::ActionKind::Wait;
      if (!(input >> action.wait_ticks)) { std::cerr << "invalid wait action\n"; continue; }
    } else if (command == "weak" || command == "strong" || command == "both") {
      action.kind = command == "weak" ? irisu::ActionKind::WeakShot
                    : command == "strong" ? irisu::ActionKind::StrongShot
                                           : irisu::ActionKind::BothShots;
      if (!(input >> action.cursor_x >> action.cursor_y)) { std::cerr << "invalid shot action\n"; continue; }
    } else {
      std::cerr << "unknown command\n";
      continue;
    }
    const auto result = simulator.step(action);
    print_state(simulator, &result);
  }
}
