#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

extern "C" void cr_sincosf(float, float *, float *);
extern "C" std::uint32_t v86_libm_sin_f32(std::uint32_t);
extern "C" std::uint32_t v86_libm_cos_f32(std::uint32_t);

namespace {

constexpr std::uint16_t kControlWord = 0x027fU;
constexpr std::uint16_t kC2 = 0x0400U;
constexpr std::uint16_t kStatusMask = 0x047fU;
constexpr std::uint32_t kFsincosLimit = 0x5f000000U;

struct UnaryResult {
  std::uint32_t output{};
  std::uint16_t status{};
};

struct PairResult {
  std::uint32_t sine{};
  std::uint32_t cosine{};
  std::uint32_t remaining{};
  std::uint16_t status{};
  bool has_pair{};
};

struct Sample {
  std::uint32_t input{};
  std::uint32_t native{};
  std::uint32_t candidate{};
};

struct Metric {
  std::uint64_t compared{};
  std::uint64_t exact{};
  std::uint64_t nan_equivalent{};
  std::uint64_t one_ulp{};
  std::uint64_t larger_than_one_ulp{};
  std::uint64_t max_ulp{};
  std::array<std::uint64_t, 256> compared_by_exponent{};
  std::array<std::uint64_t, 256> raw_mismatches_by_exponent{};
  std::vector<Sample> samples;
};

struct Options {
  enum class Candidate { CoreMath, HostF64, V86Libm };
  std::string input_path;
  std::uint64_t random_count{};
  std::uint64_t seed{UINT64_C(0x9e3779b97f4a7c15)};
  int exponent_min{-1};
  int exponent_max{-1};
  bool include_curated{true};
  bool deduplicate{};
  std::size_t sample_limit{16};
  Candidate candidate{Candidate::CoreMath};
};

std::uint16_t read_control_word() {
  std::uint16_t value{};
  asm volatile("fnstcw %0" : "=m"(value));
  return value;
}

void write_control_word(std::uint16_t value) {
  asm volatile("fldcw %0" : : "m"(value));
}

UnaryResult native_sine(std::uint32_t input_bits) {
  const float input = std::bit_cast<float>(input_bits);
  float output{};
  std::uint16_t status{};
  asm volatile("fnclex\n\t"
               "flds %[input]\n\t"
               "fsin\n\t"
               "fnstsw %%ax\n\t"
               "movw %%ax, %[status]\n\t"
               "fstps %[output]"
               : [output] "=m"(output), [status] "=m"(status)
               : [input] "m"(input)
               : "ax", "cc", "st");
  return {std::bit_cast<std::uint32_t>(output),
          static_cast<std::uint16_t>(status & kStatusMask)};
}

UnaryResult native_cosine(std::uint32_t input_bits) {
  const float input = std::bit_cast<float>(input_bits);
  float output{};
  std::uint16_t status{};
  asm volatile("fnclex\n\t"
               "flds %[input]\n\t"
               "fcos\n\t"
               "fnstsw %%ax\n\t"
               "movw %%ax, %[status]\n\t"
               "fstps %[output]"
               : [output] "=m"(output), [status] "=m"(status)
               : [input] "m"(input)
               : "ax", "cc", "st");
  return {std::bit_cast<std::uint32_t>(output),
          static_cast<std::uint16_t>(status & kStatusMask)};
}

PairResult native_sincos(std::uint32_t input_bits) {
  const float input = std::bit_cast<float>(input_bits);
  float sine{};
  float cosine{};
  float remaining{};
  std::uint16_t status{};
  std::uint8_t has_pair{};
  asm volatile("fnclex\n\t"
               "flds %[input]\n\t"
               "fsincos\n\t"
               "fnstsw %%ax\n\t"
               "movw %%ax, %[status]\n\t"
               "testw $0x0400, %%ax\n\t"
               "jnz 1f\n\t"
               "fstps %[cosine]\n\t"
               "fstps %[sine]\n\t"
               "movb $1, %[has_pair]\n\t"
               "jmp 2f\n"
               "1:\n\t"
               "fstps %[remaining]\n\t"
               "movb $0, %[has_pair]\n"
               "2:"
               : [sine] "=m"(sine), [cosine] "=m"(cosine),
                 [remaining] "=m"(remaining), [status] "=m"(status),
                 [has_pair] "=m"(has_pair)
               : [input] "m"(input)
               : "ax", "cc", "memory", "st");
  return {std::bit_cast<std::uint32_t>(sine),
          std::bit_cast<std::uint32_t>(cosine),
          std::bit_cast<std::uint32_t>(remaining),
          static_cast<std::uint16_t>(status & kStatusMask), has_pair != 0};
}

bool is_nan(std::uint32_t bits) {
  return (bits & 0x7f800000U) == 0x7f800000U &&
         (bits & 0x007fffffU) != 0;
}

bool is_finite(std::uint32_t bits) {
  return (bits & 0x7f800000U) != 0x7f800000U;
}

std::uint16_t modeled_x87_status(std::uint32_t bits) {
  const std::uint32_t absolute = bits & 0x7fffffffU;
  const std::uint32_t exponent = absolute & 0x7f800000U;
  const std::uint32_t fraction = absolute & 0x007fffffU;
  if (absolute == 0) return 0;
  if (exponent == 0x7f800000U) {
    const bool quiet_nan = fraction != 0 && (fraction & 0x00400000U) != 0;
    return quiet_nan ? 0U : 0x0001U;
  }
  if (absolute >= kFsincosLimit) return kC2;
  if (exponent == 0) return 0x0022U;
  return 0x0020U;
}

std::uint32_t ordered_float(std::uint32_t bits) {
  return (bits & 0x80000000U) != 0 ? ~bits : bits | 0x80000000U;
}

std::uint64_t ulp_distance(std::uint32_t left, std::uint32_t right) {
  const std::uint32_t a = ordered_float(left);
  const std::uint32_t b = ordered_float(right);
  return a > b ? static_cast<std::uint64_t>(a - b)
               : static_cast<std::uint64_t>(b - a);
}

void record(Metric &metric, std::uint32_t input, std::uint32_t native,
            std::uint32_t candidate, std::size_t sample_limit) {
  const std::size_t exponent = (input >> 23U) & 0xffU;
  ++metric.compared;
  ++metric.compared_by_exponent[exponent];
  if (native == candidate) {
    ++metric.exact;
    ++metric.nan_equivalent;
    return;
  }
  ++metric.raw_mismatches_by_exponent[exponent];
  if (is_nan(native) && is_nan(candidate)) {
    ++metric.nan_equivalent;
    return;
  }
  const std::uint64_t distance = ulp_distance(native, candidate);
  metric.max_ulp = std::max(metric.max_ulp, distance);
  if (distance == 1) {
    ++metric.one_ulp;
  } else {
    ++metric.larger_than_one_ulp;
  }
  if (metric.samples.size() < sample_limit) {
    metric.samples.push_back({input, native, candidate});
  }
}

std::vector<std::uint32_t> curated_inputs() {
  return {
      0x00000000U, 0x80000000U, 0x00000001U, 0x80000001U,
      0x007fffffU, 0x807fffffU, 0x00800000U, 0x80800000U,
      0x3a800000U, 0xba800000U, 0x3f000000U, 0xbf000000U,
      0x3f800000U, 0xbf800000U, 0x40490fdbU, 0xc0490fdbU,
      0x40c90fdbU, 0xc0c90fdbU, 0x4d431ce0U, 0xcd431ce0U,
      0x5effffffU, 0xdeffffffU, 0x5f000000U, 0xdf000000U,
      0x5f000001U, 0xdf000001U, 0x7f7fffffU, 0xff7fffffU,
      0x7f800000U, 0xff800000U, 0x7fc00001U, 0xffc00001U,
      0x7f800001U, 0xff800001U,
  };
}

void append_file(std::vector<std::uint32_t> &inputs, const std::string &path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open input corpus: " + path);
  const std::streamoff signed_size = stream.tellg();
  if (signed_size < 0 || signed_size % 4 != 0) {
    throw std::runtime_error("input corpus size is not divisible by four");
  }
  const auto byte_count = static_cast<std::uint64_t>(signed_size);
  const auto words = byte_count / 4U;
  if (words > std::numeric_limits<std::size_t>::max() - inputs.size()) {
    throw std::runtime_error("input corpus is too large for this process");
  }
  const auto word_count = static_cast<std::size_t>(words);
  std::vector<std::uint32_t> loaded(word_count);
  stream.seekg(0);
  stream.read(reinterpret_cast<char *>(loaded.data()),
              static_cast<std::streamsize>(byte_count));
  if (!stream) throw std::runtime_error("cannot read complete input corpus");
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  for (auto &value : loaded) {
    value = __builtin_bswap32(value);
  }
#endif
  inputs.insert(inputs.end(), loaded.begin(), loaded.end());
}

void append_random(std::vector<std::uint32_t> &inputs, std::uint64_t count,
                   std::uint64_t state) {
  inputs.reserve(inputs.size() + static_cast<std::size_t>(count));
  for (std::uint64_t index = 0; index < count; ++index) {
    state ^= state << 7U;
    state ^= state >> 9U;
    state ^= state << 8U;
    inputs.push_back(static_cast<std::uint32_t>(state));
  }
}

void append_positive_exponents(std::vector<std::uint32_t> &inputs, int first,
                               int last) {
  if (first < 0) return;
  const std::uint64_t count =
      static_cast<std::uint64_t>(last - first + 1) << 23U;
  if (count > std::numeric_limits<std::size_t>::max() - inputs.size()) {
    throw std::runtime_error("exponent sweep is too large for this process");
  }
  inputs.reserve(inputs.size() + static_cast<std::size_t>(count));
  for (int exponent = first; exponent <= last; ++exponent) {
    const std::uint32_t prefix = static_cast<std::uint32_t>(exponent) << 23U;
    for (std::uint32_t significand = 0; significand < 0x00800000U;
         ++significand) {
      inputs.push_back(prefix | significand);
    }
  }
}

std::uint64_t parse_u64(std::string_view text) {
  std::size_t consumed{};
  const std::string copy(text);
  const auto result = std::stoull(copy, &consumed, 0);
  if (consumed != copy.size()) throw std::runtime_error("invalid integer: " + copy);
  return result;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string_view arg(argv[index]);
    if (arg == "--input" && index + 1 < argc) {
      options.input_path = argv[++index];
    } else if (arg == "--random" && index + 1 < argc) {
      options.random_count = parse_u64(argv[++index]);
    } else if (arg == "--seed" && index + 1 < argc) {
      options.seed = parse_u64(argv[++index]);
    } else if (arg == "--exponent-min" && index + 1 < argc) {
      options.exponent_min = static_cast<int>(parse_u64(argv[++index]));
    } else if (arg == "--exponent-max" && index + 1 < argc) {
      options.exponent_max = static_cast<int>(parse_u64(argv[++index]));
    } else if (arg == "--samples" && index + 1 < argc) {
      options.sample_limit = static_cast<std::size_t>(parse_u64(argv[++index]));
    } else if (arg == "--candidate" && index + 1 < argc) {
      const std::string_view candidate(argv[++index]);
      if (candidate == "core-math") options.candidate = Options::Candidate::CoreMath;
      else if (candidate == "host-f64") options.candidate = Options::Candidate::HostF64;
      else if (candidate == "v86-libm") options.candidate = Options::Candidate::V86Libm;
      else throw std::runtime_error("unknown candidate: " + std::string(candidate));
    } else if (arg == "--deduplicate") {
      options.deduplicate = true;
    } else if (arg == "--no-curated") {
      options.include_curated = false;
    } else if (arg == "--help") {
      std::cout << "usage: x87-trig-diff [--input raw-u32le.bin] [--random N] "
                   "[--seed N] [--exponent-min E --exponent-max E] "
                   "[--deduplicate] [--no-curated] [--samples N] "
                   "[--candidate core-math|host-f64|v86-libm]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + std::string(arg));
    }
  }
  if ((options.exponent_min < 0) != (options.exponent_max < 0) ||
      options.exponent_min > options.exponent_max || options.exponent_max > 254) {
    throw std::runtime_error("exponent sweep requires 0 <= min <= max <= 254");
  }
  return options;
}

std::string_view candidate_name(Options::Candidate candidate) {
  switch (candidate) {
    case Options::Candidate::CoreMath:
      return "CORE-MATH cr_sincosf with x87 range guard";
    case Options::Candidate::HostF64:
      return "host libm f64 sin/cos rounded to f32 with x87 range guard";
    case Options::Candidate::V86Libm:
      return "Rust libm f64 sin/cos rounded to f32 with x87 range guard";
  }
  __builtin_unreachable();
}

std::pair<std::uint32_t, std::uint32_t> candidate_sincos(
    Options::Candidate candidate, std::uint32_t input_bits) {
  if (candidate == Options::Candidate::HostF64) {
    const double input = static_cast<double>(std::bit_cast<float>(input_bits));
    return {std::bit_cast<std::uint32_t>(static_cast<float>(std::sin(input))),
            std::bit_cast<std::uint32_t>(static_cast<float>(std::cos(input)))};
  }
  if (candidate == Options::Candidate::V86Libm) {
    return {v86_libm_sin_f32(input_bits), v86_libm_cos_f32(input_bits)};
  }
  float sine{};
  float cosine{};
  cr_sincosf(std::bit_cast<float>(input_bits), &sine, &cosine);
  return {std::bit_cast<std::uint32_t>(sine),
          std::bit_cast<std::uint32_t>(cosine)};
}

std::string hex32(std::uint32_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setw(8) << std::setfill('0') << value;
  return out.str();
}

std::string hex16(std::uint16_t value) {
  std::ostringstream out;
  out << "0x" << std::hex << std::setw(4) << std::setfill('0') << value;
  return out.str();
}

void print_metric(const Metric &metric, std::string_view indent) {
  const double rate = metric.compared == 0
                          ? 0.0
                          : static_cast<double>(metric.exact) / metric.compared;
  std::cout << indent << "{\"compared\":" << metric.compared
            << ",\"exact\":" << metric.exact
            << ",\"exact_rate\":" << std::setprecision(12) << rate
            << ",\"nan_equivalent\":" << metric.nan_equivalent
            << ",\"one_ulp\":" << metric.one_ulp
            << ",\"larger_than_one_ulp\":" << metric.larger_than_one_ulp
            << ",\"max_ulp\":" << metric.max_ulp
            << ",\"raw_mismatches_by_exponent\":{";
  bool first_exponent = true;
  for (std::size_t exponent = 0; exponent < 256; ++exponent) {
    if (metric.raw_mismatches_by_exponent[exponent] == 0) continue;
    if (!first_exponent) std::cout << ',';
    first_exponent = false;
    std::cout << '\"' << exponent << "\":{" << "\"compared\":"
              << metric.compared_by_exponent[exponent] << ",\"mismatches\":"
              << metric.raw_mismatches_by_exponent[exponent] << '}';
  }
  std::cout << "},\"samples\":[";
  for (std::size_t i = 0; i < metric.samples.size(); ++i) {
    if (i != 0) std::cout << ',';
    const auto &sample = metric.samples[i];
    std::cout << "{\"input\":\"" << hex32(sample.input)
              << "\",\"native\":\"" << hex32(sample.native)
              << "\",\"candidate\":\"" << hex32(sample.candidate) << "\"}";
  }
  std::cout << "]}";
}

void print_status_counts(const std::map<std::uint16_t, std::uint64_t> &counts) {
  std::cout << '{';
  bool first = true;
  for (const auto &[status, count] : counts) {
    if (!first) std::cout << ',';
    first = false;
    std::cout << '\"' << hex16(status) << "\":" << count;
  }
  std::cout << '}';
}

}  // namespace

int main(int argc, char **argv) try {
#if !defined(__i386__) && !defined(__x86_64__)
  throw std::runtime_error("native x87 oracle requires an x86 host");
#endif
  const Options options = parse_options(argc, argv);
  std::vector<std::uint32_t> inputs =
      options.include_curated ? curated_inputs() : std::vector<std::uint32_t>{};
  if (!options.input_path.empty()) append_file(inputs, options.input_path);
  append_random(inputs, options.random_count, options.seed);
  append_positive_exponents(inputs, options.exponent_min, options.exponent_max);
  const std::uint64_t input_count_before_dedup = inputs.size();
  if (options.deduplicate) {
    std::sort(inputs.begin(), inputs.end());
    inputs.erase(std::unique(inputs.begin(), inputs.end()), inputs.end());
  }

  const std::uint16_t saved_control_word = read_control_word();
  write_control_word(kControlWord);
  if (read_control_word() != kControlWord) {
    throw std::runtime_error("failed to install x87 control word 0x027f");
  }

  Metric sine_metric;
  Metric cosine_metric;
  Metric pair_sine_metric;
  Metric pair_cosine_metric;
  std::uint64_t pair_available{};
  std::uint64_t pair_unavailable{};
  std::uint64_t pair_sine_disagrees_with_unary{};
  std::uint64_t pair_cosine_disagrees_with_unary{};
  std::uint64_t range_c2_mismatches_sine{};
  std::uint64_t range_c2_mismatches_cosine{};
  std::uint64_t range_c2_mismatches_pair{};
  std::uint64_t range_remaining_mismatches{};
  std::uint64_t status_model_mismatches_sine{};
  std::uint64_t status_model_mismatches_cosine{};
  std::uint64_t status_model_mismatches_pair{};
  std::map<std::uint16_t, std::uint64_t> sine_status_counts;
  std::map<std::uint16_t, std::uint64_t> cosine_status_counts;
  std::map<std::uint16_t, std::uint64_t> pair_status_counts;
  std::uint64_t finite_inputs{};
  std::uint64_t nonfinite_inputs{};
  std::uint64_t positive_zero_inputs{};
  std::uint64_t negative_zero_inputs{};
  std::uint64_t out_of_range_inputs{};
  std::uint32_t maximum_finite_absolute_bits{};

  for (const std::uint32_t input_bits : inputs) {
    const UnaryResult native_sin = native_sine(input_bits);
    const UnaryResult native_cos = native_cosine(input_bits);
    const PairResult native_pair = native_sincos(input_bits);
    ++sine_status_counts[native_sin.status];
    ++cosine_status_counts[native_cos.status];
    ++pair_status_counts[native_pair.status];
    const std::uint16_t modeled_status = modeled_x87_status(input_bits);
    if (native_sin.status != modeled_status) ++status_model_mismatches_sine;
    if (native_cos.status != modeled_status) ++status_model_mismatches_cosine;
    if (native_pair.status != modeled_status) ++status_model_mismatches_pair;

    const bool finite = is_finite(input_bits);
    const std::uint32_t absolute_bits = input_bits & 0x7fffffffU;
    if (finite) {
      ++finite_inputs;
      maximum_finite_absolute_bits =
          std::max(maximum_finite_absolute_bits, absolute_bits);
    } else {
      ++nonfinite_inputs;
    }
    if (input_bits == 0) ++positive_zero_inputs;
    if (input_bits == 0x80000000U) ++negative_zero_inputs;
    const bool out_of_range =
        finite && absolute_bits >= kFsincosLimit;
    if (out_of_range) ++out_of_range_inputs;
    if (((native_sin.status & kC2) != 0) != out_of_range) {
      ++range_c2_mismatches_sine;
    }
    if (((native_cos.status & kC2) != 0) != out_of_range) {
      ++range_c2_mismatches_cosine;
    }
    if (((native_pair.status & kC2) != 0) != out_of_range) {
      ++range_c2_mismatches_pair;
    }

    auto [core_sine_bits, core_cosine_bits] =
        candidate_sincos(options.candidate, input_bits);
    if (out_of_range) {
      core_sine_bits = input_bits;
      core_cosine_bits = input_bits;
    }
    record(sine_metric, input_bits, native_sin.output, core_sine_bits,
           options.sample_limit);
    record(cosine_metric, input_bits, native_cos.output, core_cosine_bits,
           options.sample_limit);

    if (native_pair.has_pair) {
      ++pair_available;
      record(pair_sine_metric, input_bits, native_pair.sine, core_sine_bits,
             options.sample_limit);
      record(pair_cosine_metric, input_bits, native_pair.cosine, core_cosine_bits,
             options.sample_limit);
      if (native_pair.sine != native_sin.output &&
          !(is_nan(native_pair.sine) && is_nan(native_sin.output))) {
        ++pair_sine_disagrees_with_unary;
      }
      if (native_pair.cosine != native_cos.output &&
          !(is_nan(native_pair.cosine) && is_nan(native_cos.output))) {
        ++pair_cosine_disagrees_with_unary;
      }
    } else {
      ++pair_unavailable;
      if (native_pair.remaining != input_bits) ++range_remaining_mismatches;
    }
  }

  const std::uint16_t final_control_word = read_control_word();
  write_control_word(saved_control_word);

  std::cout << "{\n  \"schema\":1,\n"
            << "  \"candidate\":\"" << candidate_name(options.candidate) << "\",\n"
            << "  \"control_word\":\"" << hex16(kControlWord) << "\",\n"
            << "  \"final_control_word\":\"" << hex16(final_control_word)
            << "\",\n"
            << "  \"input_count_before_dedup\":" << input_count_before_dedup
            << ",\n  \"input_count\":" << inputs.size() << ",\n"
            << "  \"included_curated\":"
            << (options.include_curated ? "true" : "false") << ",\n"
            << "  \"input_profile\":{" << "\"finite\":" << finite_inputs
            << ",\"nonfinite\":" << nonfinite_inputs
            << ",\"positive_zero\":" << positive_zero_inputs
            << ",\"negative_zero\":" << negative_zero_inputs
            << ",\"out_of_range\":" << out_of_range_inputs
            << ",\"maximum_finite_absolute_bits\":\""
            << hex32(maximum_finite_absolute_bits) << "\"},\n"
            << "  \"deduplicated\":" << (options.deduplicate ? "true" : "false")
            << ",\n  \"metrics\":{\n    \"fsin\":";
  print_metric(sine_metric, "");
  std::cout << ",\n    \"fcos\":";
  print_metric(cosine_metric, "");
  std::cout << ",\n    \"fsincos_sine\":";
  print_metric(pair_sine_metric, "");
  std::cout << ",\n    \"fsincos_cosine\":";
  print_metric(pair_cosine_metric, "");
  std::cout << "\n  },\n  \"native_pair_consistency\":{"
            << "\"available\":" << pair_available
            << ",\"unavailable\":" << pair_unavailable
            << ",\"sine_disagrees_with_fsin\":" << pair_sine_disagrees_with_unary
            << ",\"cosine_disagrees_with_fcos\":" << pair_cosine_disagrees_with_unary
            << "},\n  \"range_status\":{"
            << "\"fsin_c2_mismatches\":" << range_c2_mismatches_sine
            << ",\"fcos_c2_mismatches\":" << range_c2_mismatches_cosine
            << ",\"fsincos_c2_mismatches\":" << range_c2_mismatches_pair
            << ",\"fsincos_remaining_mismatches\":" << range_remaining_mismatches
            << "},\n  \"portable_status_model\":{"
            << "\"fsin_mismatches\":" << status_model_mismatches_sine
            << ",\"fcos_mismatches\":" << status_model_mismatches_cosine
            << ",\"fsincos_mismatches\":" << status_model_mismatches_pair
            << "},\n  \"native_status_counts\":{\n    \"fsin\":";
  print_status_counts(sine_status_counts);
  std::cout << ",\n    \"fcos\":";
  print_status_counts(cosine_status_counts);
  std::cout << ",\n    \"fsincos\":";
  print_status_counts(pair_status_counts);
  std::cout << "\n  }\n}\n";
  return 0;
} catch (const std::exception &error) {
  std::cerr << "x87-trig-diff: " << error.what() << '\n';
  return 2;
}
