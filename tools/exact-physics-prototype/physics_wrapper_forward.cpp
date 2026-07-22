#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "irisu/physics.hpp"

#include "irisu/floating_point.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <limits>
#include <link.h>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#ifndef IRISU_EXACT_LIBRARY_SONAME
#define IRISU_EXACT_LIBRARY_SONAME \
  "libirisu_box2d_msvc_exact_multiworld.so"
#endif

namespace irisu {
namespace {

constexpr std::uintptr_t kActorTagBase = 16;
constexpr std::size_t kExactEntrypointCount = 15;

using WorldCreate = void* (*)(float, float, float, float, float, float);
using WorldDestroy = void (*)(void*);
using WorldCreateBox = void* (*)(void*, float, float, float, float, float,
                                 float, float, float);
using WorldCreateTriangle = void* (*)(void*, float, float, float, float, float,
                                      float, float, float);
using WorldCreateCircle = void* (*)(void*, float, float, float, float, float,
                                    float);
using WorldDestroyBody = void (*)(void*, void*);
using WorldStep = void (*)(void*, float, int);
using WorldGetContact = int (*)(void*, void**, void**);
using WorldGetScalar = float (*)(void*, void*);
using WorldGetVelocity = void (*)(void*, void*, float*, float*);
using WorldSetVelocity = void (*)(void*, void*, float, float);
using WorldSetUserData = void (*)(void*, void*, void*);
using WorldSetPosition = void (*)(void*, void*, float, float, float);

struct ResolvedSymbol {
  const char* name{};
  void* address{};
};

struct ProcessMapping {
  std::uintptr_t begin{};
  std::uintptr_t end{};
  std::string permissions;
  std::string device;
  std::uint64_t inode{};
  std::string path;
};

struct ExactApi {
  void* handle{};
  link_map* link_map_identity{};
  WorldCreate world_create{};
  WorldDestroy world_destroy{};
  WorldCreateBox world_create_box{};
  WorldCreateTriangle world_create_triangle{};
  WorldCreateCircle world_create_circle{};
  WorldDestroyBody world_destroy_body{};
  WorldStep world_step{};
  WorldGetContact world_get_contact{};
  WorldGetScalar world_get_x{};
  WorldGetScalar world_get_y{};
  WorldGetScalar world_get_r{};
  WorldGetVelocity world_get_v{};
  WorldSetVelocity world_set_v{};
  WorldSetUserData world_set_user_data{};
  WorldSetPosition world_set_position{};
  std::string device;
  std::uint64_t inode{};
};

std::uintptr_t parse_address(std::string_view text) {
  std::uintptr_t value{};
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value, 16);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    throw std::runtime_error("invalid address in /proc/self/maps");
  }
  return value;
}

std::uint64_t parse_inode(std::string_view text) {
  std::uint64_t value{};
  const auto parsed =
      std::from_chars(text.data(), text.data() + text.size(), value, 10);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size()) {
    throw std::runtime_error("invalid inode in /proc/self/maps");
  }
  return value;
}

std::vector<ProcessMapping> process_mappings() {
  std::ifstream stream("/proc/self/maps");
  if (!stream) {
    throw std::runtime_error("cannot inspect exact physics process mappings");
  }
  std::vector<ProcessMapping> mappings;
  std::string line;
  while (std::getline(stream, line)) {
    std::istringstream fields(line);
    std::string range;
    std::string offset;
    std::string inode;
    ProcessMapping mapping;
    if (!(fields >> range >> mapping.permissions >> offset >> mapping.device >>
          inode)) {
      throw std::runtime_error("malformed entry in /proc/self/maps");
    }
    const auto separator = range.find('-');
    if (separator == std::string::npos) {
      throw std::runtime_error("malformed address range in /proc/self/maps");
    }
    mapping.begin = parse_address(std::string_view(range).substr(0, separator));
    mapping.end = parse_address(std::string_view(range).substr(separator + 1));
    mapping.inode = parse_inode(inode);
    std::getline(fields, mapping.path);
    const auto first = mapping.path.find_first_not_of(' ');
    mapping.path.erase(0, first == std::string::npos ? mapping.path.size()
                                                     : first);
    if (mapping.begin >= mapping.end) {
      throw std::runtime_error("empty address range in /proc/self/maps");
    }
    mappings.push_back(std::move(mapping));
  }
  if (mappings.empty()) {
    throw std::runtime_error("exact physics process has no mappings");
  }
  return mappings;
}

std::string_view basename(std::string_view path) {
  const auto separator = path.rfind('/');
  return separator == std::string_view::npos ? path
                                              : path.substr(separator + 1);
}

const ProcessMapping& mapping_for(
    void* address, const std::vector<ProcessMapping>& mappings,
    const char* symbol) {
  const auto value = reinterpret_cast<std::uintptr_t>(address);
  const ProcessMapping* result{};
  for (const auto& mapping : mappings) {
    if (value >= mapping.begin && value < mapping.end) {
      if (result != nullptr) {
        throw std::runtime_error(std::string("exact physics entrypoint ") +
                                 symbol + " has ambiguous mappings");
      }
      result = &mapping;
    }
  }
  if (result == nullptr) {
    throw std::runtime_error(std::string("exact physics entrypoint ") + symbol +
                             " has no process mapping");
  }
  return *result;
}

template <typename Function>
Function resolve_symbol(
    void* handle, const char* name,
    std::array<ResolvedSymbol, kExactEntrypointCount>& symbols,
    std::size_t& count) {
  static_assert(sizeof(Function) == sizeof(void*));
  ::dlerror();
  void* address = ::dlsym(handle, name);
  const char* error = ::dlerror();
  if (error != nullptr || address == nullptr) {
    throw std::runtime_error(std::string("cannot resolve exact physics entrypoint ") +
                             name + ": " +
                             (error == nullptr ? "null address" : error));
  }
  if (count >= symbols.size()) {
    throw std::logic_error("too many exact physics entrypoints");
  }
  symbols[count++] = {name, address};
  Function function{};
  std::memcpy(&function, &address, sizeof(function));
  return function;
}

void attest_symbols(ExactApi& api,
                    const std::array<ResolvedSymbol,
                                     kExactEntrypointCount>& symbols,
                    std::size_t count) {
  if (count != kExactEntrypointCount) {
    throw std::logic_error("exact physics entrypoint table is incomplete");
  }
  const auto mappings = process_mappings();
  std::set<std::uintptr_t> unique_addresses;
  bool have_identity = false;
  for (const auto& symbol : symbols) {
    const auto value = reinterpret_cast<std::uintptr_t>(symbol.address);
    if (!unique_addresses.insert(value).second) {
      throw std::runtime_error("exact physics entrypoint addresses are not unique");
    }
    const auto& mapping = mapping_for(symbol.address, mappings, symbol.name);
    Dl_info dynamic_info{};
    void* extra_info{};
    if (::dladdr1(symbol.address, &dynamic_info, &extra_info,
                  RTLD_DL_LINKMAP) == 0 ||
        static_cast<link_map*>(extra_info) != api.link_map_identity ||
        dynamic_info.dli_fname == nullptr || dynamic_info.dli_sname == nullptr ||
        std::strcmp(dynamic_info.dli_sname, symbol.name) != 0 ||
        dynamic_info.dli_saddr != symbol.address) {
      throw std::runtime_error(std::string("exact physics entrypoint ") +
                               symbol.name +
                               " does not have the expected symbol identity in "
                               "the loaded exact object");
    }
    if (mapping.permissions.size() < 3 || mapping.permissions[2] != 'x' ||
        mapping.inode == 0 ||
        mapping.path.ends_with(" (deleted)") ||
        basename(mapping.path) != IRISU_EXACT_LIBRARY_SONAME) {
      throw std::runtime_error(std::string("exact physics entrypoint ") +
                               symbol.name + " is not executable code from " +
                               IRISU_EXACT_LIBRARY_SONAME);
    }
    if (!have_identity) {
      api.device = mapping.device;
      api.inode = mapping.inode;
      have_identity = true;
    } else if (mapping.device != api.device || mapping.inode != api.inode) {
      throw std::runtime_error(
          "exact physics entrypoints do not share one library mapping");
    }

    ::dlerror();
    void* global = ::dlsym(RTLD_DEFAULT, symbol.name);
    const char* error = ::dlerror();
    if (error != nullptr || global != symbol.address) {
      throw std::runtime_error(std::string("exact physics global binding for ") +
                               symbol.name +
                               " does not match the attested call target");
    }
  }
  if (unique_addresses.size() != kExactEntrypointCount || !have_identity) {
    throw std::runtime_error("exact physics entrypoint attestation is incomplete");
  }
}

ExactApi load_exact_api() {
  ::dlerror();
  void* handle = ::dlopen(IRISU_EXACT_LIBRARY_SONAME,
                          RTLD_NOLOAD | RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    const char* error = ::dlerror();
    throw std::runtime_error(std::string("cannot load exact physics library ") +
                             IRISU_EXACT_LIBRARY_SONAME + ": " +
                             (error == nullptr ? "unknown loader error" : error));
  }

  ExactApi api;
  api.handle = handle;  // Intentionally retained for the process lifetime.
  if (::dlinfo(handle, RTLD_DI_LINKMAP, &api.link_map_identity) != 0 ||
      api.link_map_identity == nullptr ||
      basename(api.link_map_identity->l_name) != IRISU_EXACT_LIBRARY_SONAME) {
    throw std::runtime_error(
        "loaded exact physics handle has an unexpected link-map identity");
  }
  std::array<ResolvedSymbol, kExactEntrypointCount> symbols{};
  std::size_t count{};
  api.world_create =
      resolve_symbol<WorldCreate>(handle, "b2d_world_create", symbols, count);
  api.world_destroy =
      resolve_symbol<WorldDestroy>(handle, "b2d_world_destroy", symbols, count);
  api.world_create_box = resolve_symbol<WorldCreateBox>(
      handle, "b2d_world_create_box", symbols, count);
  api.world_create_triangle = resolve_symbol<WorldCreateTriangle>(
      handle, "b2d_world_create_triangle", symbols, count);
  api.world_create_circle = resolve_symbol<WorldCreateCircle>(
      handle, "b2d_world_create_circle", symbols, count);
  api.world_destroy_body = resolve_symbol<WorldDestroyBody>(
      handle, "b2d_world_destroy_body", symbols, count);
  api.world_step =
      resolve_symbol<WorldStep>(handle, "b2d_world_step", symbols, count);
  api.world_get_contact = resolve_symbol<WorldGetContact>(
      handle, "b2d_world_get_contact", symbols, count);
  api.world_get_x =
      resolve_symbol<WorldGetScalar>(handle, "b2d_world_get_x", symbols, count);
  api.world_get_y =
      resolve_symbol<WorldGetScalar>(handle, "b2d_world_get_y", symbols, count);
  api.world_get_r =
      resolve_symbol<WorldGetScalar>(handle, "b2d_world_get_r", symbols, count);
  api.world_get_v = resolve_symbol<WorldGetVelocity>(
      handle, "b2d_world_get_v", symbols, count);
  api.world_set_v = resolve_symbol<WorldSetVelocity>(
      handle, "b2d_world_set_v", symbols, count);
  api.world_set_user_data = resolve_symbol<WorldSetUserData>(
      handle, "b2d_world_set_user_data", symbols, count);
  api.world_set_position = resolve_symbol<WorldSetPosition>(
      handle, "b2d_world_set_position", symbols, count);
  attest_symbols(api, symbols, count);
  return api;
}

const ExactApi& exact_api() {
  static const ExactApi api = load_exact_api();
  return api;
}

struct Signature {
  Shape shape{};
  double size{};
  double density{};
  double friction{};
  double restitution{};
  friend bool operator==(const Signature&, const Signature&) = default;
};

Signature signature(const Body& body) {
  return {body.shape, body.size, body.density, body.friction,
          body.restitution};
}

struct Mirror {
  Vec2 position{};
  double angle{};
  Vec2 native_velocity{};
};

Mirror mirror(const Body& body) {
  return {body.position, body.angle, body.native_velocity};
}

void* actor_tag(BodyId id) {
  return reinterpret_cast<void*>(kActorTagBase + id);
}

void* boundary_tag(BoundaryKind boundary) {
  return reinterpret_cast<void*>(static_cast<std::uintptr_t>(boundary));
}

std::pair<BodyId, BoundaryKind> decode_tag(void* value) {
  const auto tag = reinterpret_cast<std::uintptr_t>(value);
  if (tag >= kActorTagBase) {
    return {static_cast<BodyId>(tag - kActorTagBase), BoundaryKind::None};
  }
  if (tag >= static_cast<std::uintptr_t>(BoundaryKind::Floor) &&
      tag <= static_cast<std::uintptr_t>(BoundaryKind::Top)) {
    return {0, static_cast<BoundaryKind>(tag)};
  }
  return {};
}

bool same(Vec2 left, Vec2 right) {
  return left.x == right.x && left.y == right.y;
}

std::FILE* trace_file() {
  static std::FILE* stream = [] {
    const char* path = std::getenv("IRISU_EXACT_TRACE");
    return path == nullptr ? static_cast<std::FILE*>(nullptr)
                           : std::fopen(path, "w");
  }();
  return stream;
}

void trace(const char* format, auto... values) {
  if (std::FILE* stream = trace_file()) {
    std::fprintf(stream, format, values...);
    std::fputc('\n', stream);
  }
}

std::uint32_t bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

std::uint32_t boundary_ordinal(BoundaryKind boundary) {
  switch (boundary) {
    case BoundaryKind::LeftWall: return 1;
    case BoundaryKind::RightWall: return 2;
    case BoundaryKind::Floor: return 3;
    case BoundaryKind::Top: return 4;
    case BoundaryKind::None: return 0;
  }
  return 0;
}

}  // namespace

extern "C" void irisu_exact_physics_attestation(
    std::uint32_t* entrypoint_count, const char** device,
    std::uint64_t* inode) {
  if (entrypoint_count == nullptr || device == nullptr || inode == nullptr) {
    throw std::invalid_argument("exact physics attestation output is null");
  }
  const auto& api = exact_api();
  *entrypoint_count = static_cast<std::uint32_t>(kExactEntrypointCount);
  *device = api.device.c_str();
  *inode = api.inode;
}

class PhysicsWorld::Impl {
 public:
  explicit Impl(MechanicsConfig config) : config_(std::move(config)) {
    initialize();
  }

  ~Impl() {
    if (initialized_) exact_api().world_destroy(world_);
  }

  void reset() {
    if (initialized_) exact_api().world_destroy(world_);
    world_ = nullptr;
    initialized_ = false;
    entries_.clear();
    initialize();
  }

  void synchronize(std::vector<Body>& bodies) { reconcile(bodies); }

  void queue_destroy(BodyId id) {
    const auto found = entries_.find(id);
    if (found == entries_.end()) throw std::out_of_range("unknown body id");
    if (!found->second.pending_destroy) {
      exact_api().world_destroy_body(world_, found->second.native);
      trace("D %u", id + 4);
      found->second.pending_destroy = true;
    }
  }

  std::vector<Contact> step(std::vector<Body>& bodies) {
    reconcile(bodies);
    std::vector<BodyId> pending;
    for (const auto& [id, entry] : entries_) {
      if (entry.pending_destroy) pending.push_back(id);
    }

    exact_api().world_step(world_, static_cast<float>(config_.tick_seconds),
                           static_cast<int>(config_.solver_iterations));
    ++step_;
    trace("S %u %08x %u", step_,
          bits(static_cast<float>(config_.tick_seconds)),
          config_.solver_iterations);
    for (const BodyId id : pending) {
      entries_.erase(id);
      const auto body = std::find_if(bodies.begin(), bodies.end(),
                                     [id](const Body& value) {
                                       return value.id == id;
                                     });
      if (body != bodies.end() && body->lifecycle == Lifecycle::Deleted) {
        body->pending_delete = false;
      }
    }

    std::vector<Contact> contacts;
    void* first{};
    void* second{};
    std::uint32_t call{};
    while (exact_api().world_get_contact(world_, &first, &second)) {
      ++call;
      const auto [a, boundary_a] = decode_tag(first);
      const auto [b, boundary_b] = decode_tag(second);
      trace("K %u %u %u", call,
            a == 0 ? boundary_ordinal(boundary_a) : a + 4,
            b == 0 ? boundary_ordinal(boundary_b) : b + 4);
      if (a == 0 && b == 0) continue;
      contacts.push_back(
          {a, b, a == 0 ? boundary_a : (b == 0 ? boundary_b
                                                : BoundaryKind::None)});
    }
    trace("K %u 0 0", call + 1);
    sync_bodies(bodies);
    return contacts;
  }

  Vec2 raw_velocity(BodyId id) const {
    const auto found = entries_.find(id);
    if (found == entries_.end()) throw std::out_of_range("unknown body id");
    float x{};
    float y{};
    exact_api().world_get_v(world_, found->second.native, &x, &y);
    return {x, y};
  }

 private:
  struct Entry {
    void* native{};
    Signature signature{};
    Mirror mirror{};
    bool pending_destroy{};
  };

  void initialize() {
    world_ = exact_api().world_create(
        static_cast<float>(config_.world_min_x),
        static_cast<float>(config_.world_min_y),
        static_cast<float>(config_.world_max_x),
        static_cast<float>(config_.world_max_y),
        static_cast<float>(config_.gravity_y),
        static_cast<float>(config_.world_magnification));
    if (!world_) {
      throw std::runtime_error("exact wrapper initialization failed");
    }
    trace("I %08x %08x %08x %08x %08x %08x",
          bits(static_cast<float>(config_.world_min_x)),
          bits(static_cast<float>(config_.world_min_y)),
          bits(static_cast<float>(config_.world_max_x)),
          bits(static_cast<float>(config_.world_max_y)),
          bits(static_cast<float>(config_.gravity_y)),
          bits(static_cast<float>(config_.world_magnification)));
    initialized_ = true;
    step_ = 0;
    const double half_thickness = std::trunc(config_.field_thickness / 2.0);
    const double half_height = std::trunc(config_.field_height / 2.0);
    const double half_width = std::trunc(config_.field_width / 2.0);
    const double center_x =
        config_.field_x + half_width + config_.field_thickness;
    create_boundary(BoundaryKind::LeftWall, config_.field_thickness,
                    config_.field_height,
                    config_.field_x + half_thickness,
                    config_.field_y + half_height, 1.0, 1.0);
    create_boundary(BoundaryKind::RightWall, config_.field_thickness,
                    config_.field_height,
                    config_.field_x + config_.field_width +
                        config_.field_thickness,
                    config_.field_y + half_height, 1.0, 1.0);
    create_boundary(BoundaryKind::Floor,
                    config_.field_width + 2.0 * config_.field_thickness,
                    config_.field_bottom_height, center_x,
                    config_.field_y + config_.field_height +
                        config_.field_blank +
                        std::trunc(config_.field_bottom_height / 2.0),
                    1.0, 0.0);
    create_boundary(BoundaryKind::Top, config_.field_top_width,
                    config_.field_top_height, center_x, config_.field_top, 1.0,
                    0.5);
  }

  void create_boundary(BoundaryKind boundary, double width, double height,
                       double x, double y, double friction,
                       double restitution) {
    void* body = exact_api().world_create_box(
        world_, static_cast<float>(width), static_cast<float>(height),
        static_cast<float>(x), static_cast<float>(y), 0.0f, 0.0f,
        static_cast<float>(friction), static_cast<float>(restitution));
    if (!body) throw std::runtime_error("exact wrapper boundary creation failed");
    const auto ordinal = boundary_ordinal(boundary);
    trace("B %u %08x %08x %08x %08x 00000000 00000000 %08x %08x", ordinal,
          bits(static_cast<float>(width)), bits(static_cast<float>(height)),
          bits(static_cast<float>(x)), bits(static_cast<float>(y)),
          bits(static_cast<float>(friction)),
          bits(static_cast<float>(restitution)));
    exact_api().world_set_v(world_, body, 0.0f, 0.0f);
    trace("V %u 00000000 00000000", ordinal);
    exact_api().world_set_user_data(world_, body, boundary_tag(boundary));
    trace("U %u", ordinal);
  }

  void create_body(Body& body) {
    void* native{};
    const auto x = static_cast<float>(body.position.x);
    const auto y = static_cast<float>(body.position.y);
    const auto angle = static_cast<float>(body.angle);
    const auto density = static_cast<float>(body.density);
    const auto friction = static_cast<float>(body.friction);
    const auto restitution = static_cast<float>(body.restitution);
    const auto size = static_cast<float>(body.size);
    switch (body.shape) {
      case Shape::Box:
        native = exact_api().world_create_box(
            world_, size, size, x, y, angle, density, friction, restitution);
        break;
      case Shape::Circle:
        native = exact_api().world_create_circle(
            world_, size / 2.0f, x, y, density, friction, restitution);
        break;
      case Shape::Triangle:
        native = exact_api().world_create_triangle(
            world_, size, size, x, y, angle, density, friction, restitution);
        break;
    }
    if (!native) throw std::runtime_error("exact wrapper body creation failed");
    const auto ordinal = body.id + 4;
    if (body.shape == Shape::Circle) {
      trace("C %u %08x %08x %08x %08x %08x %08x", ordinal,
            bits(size / 2.0f), bits(x), bits(y), bits(density), bits(friction),
            bits(restitution));
    } else {
      trace("%c %u %08x %08x %08x %08x %08x %08x %08x %08x",
            body.shape == Shape::Box ? 'B' : 'T', ordinal, bits(size),
            bits(size), bits(x), bits(y), bits(angle), bits(density),
            bits(friction), bits(restitution));
    }
    const auto magnification = static_cast<float>(config_.world_magnification);
    const float velocity_x =
        static_cast<float>(body.native_velocity.x) * magnification;
    const float velocity_y =
        static_cast<float>(body.native_velocity.y) * magnification;
    exact_api().world_set_v(world_, native, velocity_x, velocity_y);
    trace("V %u %08x %08x", ordinal, bits(velocity_x), bits(velocity_y));
    exact_api().world_set_user_data(world_, native, actor_tag(body.id));
    trace("U %u", ordinal);
    entries_.emplace(body.id,
                     Entry{native, signature(body), mirror(body), false});
  }

  void apply_external_changes(Body& body, Entry& entry) {
    const Mirror previous = entry.mirror;
    const Vec2 requested_velocity = body.native_velocity;
    if (!same(body.position, previous.position) || body.angle != previous.angle) {
      exact_api().world_set_position(world_, entry.native,
                                     static_cast<float>(body.position.x),
                                     static_cast<float>(body.position.y),
                                     static_cast<float>(body.angle));
      trace("P %u %08x %08x %08x", body.id + 4,
            bits(static_cast<float>(body.position.x)),
            bits(static_cast<float>(body.position.y)),
            bits(static_cast<float>(body.angle)));
      body.native_position = {
          static_cast<float>(body.position.x) /
              static_cast<float>(config_.world_magnification),
          static_cast<float>(body.position.y) /
              static_cast<float>(config_.world_magnification)};
      body.native_angle = static_cast<float>(body.angle);
      body.native_velocity = {};
    }
    if (!same(requested_velocity, previous.native_velocity)) {
      const auto scale = static_cast<float>(config_.world_magnification);
      exact_api().world_set_v(
          world_, entry.native,
          static_cast<float>(requested_velocity.x) * scale,
          static_cast<float>(requested_velocity.y) * scale);
      trace("V %u %08x %08x", body.id + 4,
            bits(static_cast<float>(requested_velocity.x) * scale),
            bits(static_cast<float>(requested_velocity.y) * scale));
      body.native_velocity = raw(entry.native);
    }
    entry.mirror = mirror(body);
  }

  void reconcile(std::vector<Body>& bodies) {
    std::vector<Body*> ordered;
    ordered.reserve(bodies.size());
    for (auto& body : bodies) ordered.push_back(&body);
    std::stable_sort(ordered.begin(), ordered.end(), [](const Body* left,
                                                        const Body* right) {
      if (left->actor_slot != right->actor_slot) {
        return left->actor_slot < right->actor_slot;
      }
      if ((left->lifecycle == Lifecycle::Deleted) !=
          (right->lifecycle == Lifecycle::Deleted)) {
        return left->lifecycle == Lifecycle::Deleted;
      }
      return left->id < right->id;
    });
    for (Body* candidate : ordered) {
      Body& body = *candidate;
      auto found = entries_.find(body.id);
      if (body.lifecycle == Lifecycle::Deleted) {
        if (found != entries_.end()) {
          if (!found->second.pending_destroy) {
            apply_external_changes(body, found->second);
          }
          queue_destroy(body.id);
        }
      } else if (found == entries_.end()) {
        create_body(body);
      } else {
        if (found->second.pending_destroy) {
          throw std::logic_error("destroyed body id revived");
        }
        if (found->second.signature != signature(body)) {
          throw std::logic_error("fixture signature changed");
        }
        apply_external_changes(body, found->second);
      }
    }
  }

  Vec2 raw(void* native) const {
    float x{};
    float y{};
    exact_api().world_get_v(world_, native, &x, &y);
    return {x, y};
  }

  void sync_bodies(std::vector<Body>& bodies) {
    const auto scale = static_cast<float>(config_.world_magnification);
    for (auto& body : bodies) {
      if (body.lifecycle == Lifecycle::Deleted) continue;
      const auto found = entries_.find(body.id);
      if (found == entries_.end()) continue;
      const float x = exact_api().world_get_x(world_, found->second.native);
      const float y = exact_api().world_get_y(world_, found->second.native);
      const float angle = exact_api().world_get_r(world_, found->second.native);
      const Vec2 velocity = raw(found->second.native);
      body.position = {x, y};
      body.velocity = velocity;
      body.angle = angle;
      body.native_position = {x / scale, y / scale};
      body.native_center = body.native_position;
      body.native_velocity = velocity;
      body.native_angle = angle;
      body.native_state_valid = true;
      body.native_center_valid = false;
      body.sleeping = false;
      body.sleep_time = 0.0;
      found->second.mirror = mirror(body);
    }
  }

  MechanicsConfig config_;
  std::map<BodyId, Entry> entries_;
  void* world_{};
  bool initialized_{};
  std::uint32_t step_{};
};

PhysicsWorld::PhysicsWorld(MechanicsConfig config)
    : config_(validated_mechanics_config(std::move(config))),
      impl_(std::make_unique<Impl>(config_)) {}

PhysicsWorld::~PhysicsWorld() = default;
PhysicsWorld::PhysicsWorld(PhysicsWorld&&) noexcept = default;
PhysicsWorld& PhysicsWorld::operator=(PhysicsWorld&&) noexcept = default;

void PhysicsWorld::initialize_mass(Body& body) const {
  const ScopedFloatingPointEnvironment floating_point_environment;
  const float scale = static_cast<float>(config_.world_magnification);
  body.native_position = {static_cast<float>(body.position.x) / scale,
                          static_cast<float>(body.position.y) / scale};
  body.native_velocity = {static_cast<float>(body.velocity.x),
                          static_cast<float>(body.velocity.y)};
  body.native_angle = static_cast<float>(body.angle);
  body.native_angular_velocity = static_cast<float>(body.angular_velocity);
  body.native_state_valid = true;
  body.native_center_valid = false;
  body.sleeping = false;
  body.sleep_time = 0.0;
  if (body.density <= 0.0 || body.lifecycle == Lifecycle::Deleted) {
    body.inverse_mass = 0.0;
    body.inverse_inertia = 0.0;
    return;
  }
  const double width = body.size / config_.world_magnification;
  double area = width * width;
  if (body.shape == Shape::Triangle) area *= 0.5;
  if (body.shape == Shape::Circle) {
    const double radius = width * 0.5;
    area = 3.14159265358979323846 * radius * radius;
  }
  const double mass = body.density * area;
  body.inverse_mass = mass > 0.0 ? 1.0 / mass : 0.0;
  body.inverse_inertia = 0.0;
}

void PhysicsWorld::reset() {
  const ScopedFloatingPointEnvironment floating_point_environment;
  impl_->reset();
}

void PhysicsWorld::synchronize(std::vector<Body>& bodies) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  impl_->synchronize(bodies);
}

void PhysicsWorld::queue_destroy(BodyId id) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  impl_->queue_destroy(id);
}

void PhysicsWorld::rebuild(std::vector<Body>&,
                           const std::vector<ContactImpulse>&,
                           const PhysicsOrdering&) {
  throw std::logic_error("forward exact wrapper cannot restore snapshots");
}

std::vector<ContactImpulse> PhysicsWorld::contact_impulses(
    const std::vector<Body>&) const {
  throw std::logic_error("forward exact wrapper cannot snapshot contacts");
}

PhysicsOrdering PhysicsWorld::ordering() const {
  throw std::logic_error("forward exact wrapper cannot snapshot ordering");
}

std::vector<Contact> PhysicsWorld::step(std::vector<Body>& bodies) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  return impl_->step(bodies);
}

Vec2 PhysicsWorld::raw_velocity(BodyId id) const {
  const ScopedFloatingPointEnvironment floating_point_environment;
  return impl_->raw_velocity(id);
}

}  // namespace irisu
