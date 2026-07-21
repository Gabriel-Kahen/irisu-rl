#include "irisu/simulator.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

std::uint32_t word(const std::vector<unsigned char>& data, std::size_t offset) {
  std::uint32_t value{};
  std::memcpy(&value, data.data() + offset, sizeof(value));
  return value;
}

irisu::Action action(std::uint32_t value, bool suppress_fresh_edges) {
  const auto buttons = value & 3U;
  const double x = static_cast<double>((value >> 2U) & 0x3ffU);
  const double y = static_cast<double>((value >> 12U) & 0x1ffU);
  const auto kind = buttons == 1U   ? irisu::ActionKind::WeakShot
                    : buttons == 2U ? irisu::ActionKind::StrongShot
                    : buttons == 3U ? irisu::ActionKind::BothShots
                                    : irisu::ActionKind::Wait;
  return {kind, x, y, 1, suppress_fresh_edges};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("usage: exact-replay replay.rpy");
    std::ifstream stream(argv[1], std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open replay");
    const std::vector<unsigned char> data{std::istreambuf_iterator<char>(stream),
                                          std::istreambuf_iterator<char>()};
    if (data.size() < 52 || (data.size() - 52) % 4 != 0) {
      throw std::runtime_error("expected padded v2.03 replay");
    }

    const char* requested_cw = std::getenv("IRISU_EXACT_CW");
    const std::uint16_t control_word = requested_cw == nullptr
                                           ? 0x137fU
                                           : static_cast<std::uint16_t>(
                                                 std::strtoul(requested_cw,
                                                              nullptr, 0));
    __asm__ __volatile__("fldcw %0" : : "m"(control_word));
    irisu::Simulator simulator;
    auto observation = simulator.reset(word(data, 0));
    std::int64_t timeline_score = observation.score;
    std::int64_t timeline_gauge = observation.gauge;
    std::uint64_t score_calls{};
    std::uint64_t clear_calls{};
    std::vector<std::tuple<std::uint64_t, std::int64_t, std::int64_t>> scores;
    std::vector<std::tuple<std::uint64_t, std::int64_t, std::int64_t,
                           irisu::BodyId, unsigned>>
        gauge_changes;
    std::uint64_t terminal_frame = data.size() / 4;
    for (std::size_t offset = 52, frame = 0; offset < data.size();
         offset += 4, ++frame) {
      // Input.update clears fresh left/right edges while replay index < 3.
      // The index is incremented before that check, so records 0 and 1 retain
      // held history but cannot fire.
      const auto result = simulator.step(
          action(word(data, offset), frame < 2));
      for (const auto& event : result.events) {
        if (event.kind == irisu::EventKind::ScoreChanged) {
          ++score_calls;
          timeline_score += event.value;
          scores.emplace_back(event.tick, event.value, timeline_score);
        }
        if (event.kind == irisu::EventKind::GaugeChanged) {
          timeline_gauge += event.value;
          const unsigned kind = event.detail == "normal rot penalty" ? 1U
                                : event.detail ==
                                          "scene clamp and passive drain"
                                    ? 0U
                                    : 2U;
          gauge_changes.emplace_back(event.tick, event.value, timeline_gauge,
                                     event.a, kind);
        }
        if (event.kind == irisu::EventKind::Confirmed) ++clear_calls;
      }
      observation = simulator.observation();
      if (timeline_score != observation.score) {
        throw std::logic_error("score events do not reconstruct total score");
      }
      if (timeline_gauge != observation.gauge) {
        throw std::logic_error("gauge events do not reconstruct total gauge");
      }
      if (result.terminated && terminal_frame == data.size() / 4) {
        terminal_frame = frame;
      }
    }
    std::cout << "{\"tick\":" << observation.tick << ",\"score\":"
              << observation.score << ",\"gauge\":" << observation.gauge
              << ",\"level\":" << observation.level
              << ",\"highest_chain\":" << observation.highest_chain
              << ",\"clears\":" << observation.qualifying_clear_count
              << ",\"score_calls\":" << score_calls
              << ",\"confirmed\":" << clear_calls
              << ",\"terminal_frame\":" << terminal_frame
              << ",\"score_timeline\":[";
    for (std::size_t index = 0; index < scores.size(); ++index) {
      if (index != 0) std::cout << ',';
      const auto [tick, delta, total] = scores[index];
      std::cout << '[' << tick << ',' << delta << ',' << total << ']';
    }
    std::cout << "],\"gauge_timeline\":[";
    for (std::size_t index = 0; index < gauge_changes.size(); ++index) {
      if (index != 0) std::cout << ',';
      const auto [tick, delta, total, body, kind] = gauge_changes[index];
      std::cout << '[' << tick << ',' << delta << ',' << total << ',' << body
                << ',' << kind << ']';
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
