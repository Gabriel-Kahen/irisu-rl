#include "irisu/simulator.hpp"

#include "irisu/floating_point.hpp"
#include "irisu/normal_rules.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>
#include <tuple>
#include <type_traits>

namespace irisu {
namespace {

constexpr std::uint32_t kSnapshotMagic = 0x49524953U;  // IRIS
constexpr std::uint32_t kSnapshotVersion = 7;
constexpr std::size_t kSerializedBodyBytes = 227;
constexpr std::size_t kSerializedGroupBytes = 16;
constexpr std::size_t kSerializedContactImpulseBytes = 58;
constexpr std::size_t kSerializedBroadPhaseBoundBytes = 6;
constexpr std::uint8_t kMaxManifoldPoints = 2;
constexpr std::size_t kMaximumSnapshotBodies =
    MechanicsConfig::physics_proxy_capacity;
constexpr std::size_t kMaximumSnapshotGroups = 1'000'000;
constexpr BodyId kMaximumBodyId = 0x7fffffffU;
constexpr BodyId kExhaustedNextBodyId = kMaximumBodyId + 1U;
constexpr long double kNumericGuardThreshold = 1.0e5L;

bool uses_nominal_scripted_fall(double value) {
  return std::bit_cast<std::uint64_t>(value) == 0x3fc999999999999aULL;
}

float finite_legacy_float(double value, const char* detail) {
  if (!std::isfinite(value) ||
      std::abs(value) > std::numeric_limits<float>::max()) {
    throw std::invalid_argument(detail);
  }
  const auto narrowed = static_cast<float>(value);
  if (!std::isfinite(narrowed)) {
    throw std::invalid_argument(detail);
  }
  return narrowed;
}

void validate_body_numeric_safety(const MechanicsConfig& config,
                                  const Body& body,
                                  std::uint64_t update_horizon = 1) {
  const float magnification = finite_legacy_float(
      config.world_magnification, "snapshot magnification is invalid");
  const float size = finite_legacy_float(
      body.size, "snapshot body size cannot be represented by Box2D");
  const float density = finite_legacy_float(
      body.density, "snapshot body density cannot be represented by Box2D");
  (void)finite_legacy_float(
      body.friction, "snapshot body friction cannot be represented by Box2D");
  (void)finite_legacy_float(
      body.restitution,
      "snapshot body restitution cannot be represented by Box2D");
  (void)finite_legacy_float(
      body.inverse_mass,
      "snapshot inverse mass cannot be represented by Box2D");
  (void)finite_legacy_float(
      body.inverse_inertia,
      "snapshot inverse inertia cannot be represented by Box2D");
  (void)finite_legacy_float(
      body.sleep_time, "snapshot sleep time cannot be represented by Box2D");

  const float position_x = finite_legacy_float(
      body.position.x, "snapshot actor position cannot be represented");
  const float position_y = finite_legacy_float(
      body.position.y, "snapshot actor position cannot be represented");
  const float actor_velocity_x = finite_legacy_float(
      body.velocity.x, "snapshot actor velocity cannot be represented");
  const float actor_velocity_y = finite_legacy_float(
      body.velocity.y, "snapshot actor velocity cannot be represented");
  const float scripted_x = finite_legacy_float(
      body.scripted_velocity.x,
      "snapshot scripted velocity cannot be represented");
  const float scripted_y = finite_legacy_float(
      body.scripted_velocity.y,
      "snapshot scripted velocity cannot be represented");
  const float angle = finite_legacy_float(
      body.angle, "snapshot actor angle cannot be represented");
  const float angular_velocity = finite_legacy_float(
      body.angular_velocity,
      "snapshot actor angular velocity cannot be represented");
  if (!std::isfinite(position_x + scripted_x) ||
      !std::isfinite(position_y + scripted_y) ||
      !std::isfinite(angle + angular_velocity)) {
    throw std::invalid_argument("snapshot actor integration would overflow");
  }

  const float native_x = finite_legacy_float(
      body.native_position.x,
      "snapshot native position cannot be represented by Box2D");
  const float native_y = finite_legacy_float(
      body.native_position.y,
      "snapshot native position cannot be represented by Box2D");
  const float native_center_x = finite_legacy_float(
      body.native_center.x,
      "snapshot native center cannot be represented by Box2D");
  const float native_center_y = finite_legacy_float(
      body.native_center.y,
      "snapshot native center cannot be represented by Box2D");
  const float native_velocity_x = finite_legacy_float(
      body.native_velocity.x,
      "snapshot native velocity cannot be represented by Box2D");
  const float native_velocity_y = finite_legacy_float(
      body.native_velocity.y,
      "snapshot native velocity cannot be represented by Box2D");
  const float native_angle = finite_legacy_float(
      body.native_angle,
      "snapshot native angle cannot be represented by Box2D");
  const float native_angular_velocity = finite_legacy_float(
      body.native_angular_velocity,
      "snapshot native angular velocity cannot be represented by Box2D");
  if (!std::isfinite(native_x * magnification) ||
      !std::isfinite(native_y * magnification) ||
      !std::isfinite(native_center_x * magnification) ||
      !std::isfinite(native_center_y * magnification)) {
    throw std::invalid_argument("snapshot native position conversion overflows");
  }
  const long double maximum_center_offset =
      std::abs(static_cast<long double>(size) / magnification);
  if (std::abs(static_cast<long double>(native_center_x) - native_x) >
          maximum_center_offset ||
      std::abs(static_cast<long double>(native_center_y) - native_y) >
          maximum_center_offset) {
    throw std::invalid_argument("snapshot native center is inconsistent");
  }

  const float half = static_cast<float>(
      static_cast<double>(size) / (2.0 * static_cast<double>(magnification)));
  const float broad_extent = 2.0F * half;
  if (!(half > 0.0F) || !std::isfinite(half) ||
      !std::isfinite(broad_extent) ||
      !std::isfinite(native_x - broad_extent) ||
      !std::isfinite(native_x + broad_extent) ||
      !std::isfinite(native_y - broad_extent) ||
      !std::isfinite(native_y + broad_extent)) {
    throw std::invalid_argument("snapshot body geometry is unsafe for Box2D");
  }
  if (body.shape == Shape::Triangle &&
      !(2.0F * half * half > std::numeric_limits<float>::epsilon())) {
    throw std::invalid_argument("snapshot triangle area is too small for Box2D");
  }

  const long double maximum_updates = update_horizon;
  constexpr long double numeric_headroom =
      static_cast<long double>(std::numeric_limits<float>::max()) / 4.0L;
  const long double solver_headroom =
      std::sqrt(static_cast<long double>(std::numeric_limits<float>::max())) /
      16.0L;
  const long double tick = finite_legacy_float(
      config.tick_seconds, "snapshot tick size cannot be represented");
  const long double gravity = static_cast<long double>(finite_legacy_float(
                                  config.gravity_y,
                                  "snapshot gravity cannot be represented")) /
                              magnification;
  const auto within_headroom = [&](long double value) {
    return std::isfinite(value) && std::abs(value) <= numeric_headroom;
  };
  const auto within_solver_headroom = [&](long double value) {
    return std::isfinite(value) && std::abs(value) <= solver_headroom;
  };
  const long double native_x_bound =
      std::abs(static_cast<long double>(native_x)) + broad_extent +
      maximum_updates * tick *
          std::abs(static_cast<long double>(native_velocity_x));
  const long double native_y_bound =
      std::abs(static_cast<long double>(native_y)) + broad_extent +
      maximum_updates * tick *
          std::abs(static_cast<long double>(native_velocity_y)) +
      maximum_updates * maximum_updates * tick * tick *
          std::abs(gravity);
  const long double native_velocity_y_bound =
      std::abs(static_cast<long double>(native_velocity_y)) +
      maximum_updates * tick * std::abs(gravity);
  const long double native_angle_bound =
      std::abs(static_cast<long double>(native_angle)) +
      maximum_updates * tick *
          std::abs(static_cast<long double>(native_angular_velocity));
  const long double actor_x_bound =
      std::abs(static_cast<long double>(position_x)) +
      maximum_updates * std::abs(static_cast<long double>(scripted_x));
  const long double actor_y_bound =
      std::abs(static_cast<long double>(position_y)) +
      maximum_updates * std::abs(static_cast<long double>(scripted_y));
  const long double actor_angle_bound =
      std::abs(static_cast<long double>(angle)) +
      maximum_updates * std::abs(static_cast<long double>(angular_velocity));
  if (!within_headroom(native_x_bound) ||
      !within_headroom(native_y_bound) ||
      !within_solver_headroom(native_velocity_x) ||
      !within_solver_headroom(native_velocity_y) ||
      !within_solver_headroom(native_velocity_y_bound) ||
      !within_headroom(native_angle_bound) ||
      !within_solver_headroom(native_angular_velocity) ||
      !within_headroom(actor_x_bound) ||
      !within_headroom(actor_y_bound) ||
      !within_headroom(actor_velocity_x) ||
      !within_headroom(actor_velocity_y) ||
      !within_headroom(actor_angle_bound)) {
    throw std::invalid_argument(
        "snapshot body integration lacks finite numeric headroom");
  }

  if (density > 0.0F) {
    const long double h = half;
    const long double d = density;
    const long double minimum_mass = 2.0L * d * h * h;
    const long double minimum_inertia = 0.25L * d * h * h * h * h;
    const long double maximum_mass = 4.0L * d * h * h;
    const long double maximum_inertia = 16.0L * d * h * h * h * h;
    constexpr long double maximum_float =
        std::numeric_limits<float>::max();
    if (!(minimum_mass > 0.0L) || !(minimum_inertia > 0.0L) ||
        maximum_mass > maximum_float || maximum_inertia > maximum_float ||
        1.0L / minimum_mass > maximum_float ||
        1.0L / minimum_inertia > maximum_float) {
      throw std::invalid_argument(
          "snapshot body mass or inertia is unsafe for Box2D");
    }
  }
}

bool body_requires_numeric_guard(const MechanicsConfig& config,
                                 const Body& body) {
  const auto large = [](long double value) {
    return !std::isfinite(value) ||
           std::abs(value) > kNumericGuardThreshold;
  };
  return large(body.position.x) || large(body.position.y) ||
         large(body.velocity.x) || large(body.velocity.y) ||
         large(body.angle) || large(body.angular_velocity) ||
         large(body.native_position.x) || large(body.native_position.y) ||
         large(body.native_center.x) || large(body.native_center.y) ||
         large(body.native_velocity.x) || large(body.native_velocity.y) ||
         large(body.native_angle) || large(body.native_angular_velocity) ||
         large(body.scripted_velocity.x) ||
         large(body.scripted_velocity.y) || large(config.gravity_y) ||
         large(config.tick_seconds) ||
         large(static_cast<long double>(config.gravity_y) /
               config.world_magnification);
}

void validate_public_spawn_geometry(const MechanicsConfig& config,
                                    Shape shape, double size, Vec2 position,
                                    double density, double friction,
                                    double restitution,
                                    double scripted_velocity_y) {
  Body candidate;
  candidate.shape = shape;
  candidate.size = size;
  candidate.position = position;
  const auto magnification = finite_legacy_float(
      config.world_magnification, "public spawn magnification is invalid");
  candidate.native_position = {
      static_cast<double>(finite_legacy_float(
                              position.x, "public spawn position is invalid") /
                          magnification),
      static_cast<double>(finite_legacy_float(
                              position.y, "public spawn position is invalid") /
                          magnification)};
  candidate.native_center = candidate.native_position;
  candidate.native_state_valid = true;
  candidate.native_center_valid = true;
  candidate.density = density;
  candidate.friction = friction;
  candidate.restitution = restitution;
  candidate.scripted_velocity.y = scripted_velocity_y;
  candidate.inverse_mass = 1.0;
  candidate.inverse_inertia = 1.0;
  validate_body_numeric_safety(config, candidate);
}

double stored_float(double value) {
  return static_cast<double>(static_cast<float>(value));
}

double stored_float_add(double left, double right) {
  return static_cast<double>(
      static_cast<float>(static_cast<float>(left) + static_cast<float>(right)));
}

template <typename T>
void append_integer(std::vector<std::byte>& output, T value) {
  using U = std::make_unsigned_t<T>;
  U encoded = static_cast<U>(value);
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    output.push_back(static_cast<std::byte>((encoded >> (index * 8U)) & 0xffU));
  }
}

void append_double(std::vector<std::byte>& output, double value) {
  append_integer(output, std::bit_cast<std::uint64_t>(value));
}

void append_float(std::vector<std::byte>& output, double value) {
  append_integer(output,
                 std::bit_cast<std::uint32_t>(static_cast<float>(value)));
}

template <typename T>
T read_integer(std::span<const std::byte> input, std::size_t& offset) {
  if (input.size() - std::min(input.size(), offset) < sizeof(T)) {
    throw std::invalid_argument("truncated snapshot");
  }
  using U = std::make_unsigned_t<T>;
  U value{};
  for (std::size_t index = 0; index < sizeof(T); ++index) {
    const U byte = static_cast<U>(std::to_integer<unsigned char>(input[offset++]));
    value = static_cast<U>(value | static_cast<U>(byte << (index * 8U)));
  }
  return static_cast<T>(value);
}

double read_double(std::span<const std::byte> input, std::size_t& offset) {
  return std::bit_cast<double>(read_integer<std::uint64_t>(input, offset));
}

double read_float(std::span<const std::byte> input, std::size_t& offset) {
  return static_cast<double>(
      std::bit_cast<float>(read_integer<std::uint32_t>(input, offset)));
}

bool read_bool(std::span<const std::byte> input, std::size_t& offset) {
  const auto value = read_integer<std::uint8_t>(input, offset);
  if (value > 1) throw std::invalid_argument("snapshot boolean is not canonical");
  return value != 0;
}

std::uint64_t fnv1a(std::span<const std::byte> bytes) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (const auto byte : bytes) {
    hash ^= std::to_integer<std::uint8_t>(byte);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::uint64_t pair_key(BodyId a, BodyId b) {
  const auto low = std::min(a, b);
  const auto high = std::max(a, b);
  return (static_cast<std::uint64_t>(low) << 32U) | high;
}

std::uint64_t contact_key(const Contact& contact) {
  if (contact.boundary == BoundaryKind::None) return pair_key(contact.a, contact.b);
  const BodyId body = contact.a != 0 ? contact.a : contact.b;
  return (std::uint64_t{1} << 63U) |
         (static_cast<std::uint64_t>(body) << 3U) |
         static_cast<std::uint8_t>(contact.boundary);
}

bool live(const Body& body) { return body.lifecycle != Lifecycle::Deleted; }

std::uint64_t checked_u64_add(std::uint64_t left, std::uint64_t right,
                              const char* detail) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    throw std::overflow_error(detail);
  }
  return left + right;
}

std::int64_t checked_nonnegative_product(std::int64_t left,
                                         std::int64_t right,
                                         const char* detail) {
  if (left < 0 || right < 0 ||
      (left != 0 && right > std::numeric_limits<std::int64_t>::max() / left)) {
    throw std::overflow_error(detail);
  }
  return left * right;
}

std::int64_t saturated_nonnegative_product(std::int64_t left,
                                           std::int64_t right) {
  if (left <= 0 || right <= 0) return 0;
  if (right > std::numeric_limits<std::int64_t>::max() / left) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return left * right;
}

std::int64_t checked_subtract(std::int64_t left, std::int64_t right,
                              const char* detail) {
  if ((right > 0 && left < std::numeric_limits<std::int64_t>::min() + right) ||
      (right < 0 && left > std::numeric_limits<std::int64_t>::max() + right)) {
    throw std::overflow_error(detail);
  }
  return left - right;
}

std::int64_t saturated_difference(std::int64_t after, std::int64_t before) {
  if (before < 0 && after > std::numeric_limits<std::int64_t>::max() + before) {
    return std::numeric_limits<std::int64_t>::max();
  }
  if (before > 0 && after < std::numeric_limits<std::int64_t>::min() + before) {
    return std::numeric_limits<std::int64_t>::min();
  }
  return after - before;
}

void serialize_body(std::vector<std::byte>& output, const Body& body) {
  append_integer(output, body.id);
  append_integer(output, static_cast<std::uint8_t>(body.kind));
  append_integer(output, static_cast<std::uint8_t>(body.shape));
  append_integer(output, static_cast<std::uint8_t>(body.lifecycle));
  append_integer(output, body.color);
  append_double(output, body.position.x);
  append_double(output, body.position.y);
  append_double(output, body.velocity.x);
  append_double(output, body.velocity.y);
  append_double(output, body.angle);
  append_double(output, body.angular_velocity);
  append_float(output, body.native_position.x);
  append_float(output, body.native_position.y);
  append_float(output, body.native_center.x);
  append_float(output, body.native_center.y);
  append_float(output, body.native_velocity.x);
  append_float(output, body.native_velocity.y);
  append_float(output, body.native_angle);
  append_float(output, body.native_angular_velocity);
  append_integer(output, static_cast<std::uint8_t>(body.native_state_valid));
  append_double(output, body.size);
  append_double(output, body.density);
  append_double(output, body.friction);
  append_double(output, body.restitution);
  append_double(output, body.inverse_mass);
  append_double(output, body.inverse_inertia);
  append_integer(output, body.chain_id);
  append_integer(output, body.actor_slot);
  append_integer(output, body.size_slot);
  append_integer(output, body.projectile_hits);
  append_integer(output, body.age_ticks);
  append_integer(output, body.remaining_lifetime);
  append_double(output, body.scripted_velocity.x);
  append_double(output, body.scripted_velocity.y);
  append_integer(output, body.rot_timer);
  append_integer(output, static_cast<std::uint8_t>(body.physics_owned));
  append_integer(output, static_cast<std::uint8_t>(body.special));
  append_integer(output, body.freshness_state);
  append_integer(output, static_cast<std::uint8_t>(body.grouped));
  append_integer(output,
                 static_cast<std::uint8_t>(body.successful_clear_pending));
  append_integer(output, body.non_wall_contacts);
  append_integer(output, static_cast<std::uint8_t>(body.top_contact_pending));
  append_integer(output, static_cast<std::uint8_t>(body.top_contact_enabled));
  append_integer(output, body.physics_update_count);
  append_integer(output, body.rule_guard_f0);
  append_integer(output, static_cast<std::uint8_t>(body.delete_marked));
  append_integer(output, static_cast<std::uint8_t>(body.pending_delete));
  append_integer(output, static_cast<std::uint8_t>(body.sleeping));
  append_double(output, body.sleep_time);
}

void serialize_contact_impulse(std::vector<std::byte>& output,
                               const ContactImpulse& impulse) {
  append_integer(output, impulse.a);
  append_integer(output, impulse.b);
  append_integer(output, static_cast<std::uint8_t>(impulse.boundary));
  append_integer(output, static_cast<std::uint8_t>(impulse.destroy_pending));
  append_integer(output, impulse.manifold_count);
  append_integer(output, impulse.manifold_index);
  append_integer(output, impulse.point_count);
  append_integer(output, impulse.point_index);
  append_integer(output, impulse.contact_order);
  append_integer(output, impulse.feature_id);
  append_integer(output, impulse.normal_x_bits);
  append_integer(output, impulse.normal_y_bits);
  append_integer(output, impulse.point_x_bits);
  append_integer(output, impulse.point_y_bits);
  append_integer(output, impulse.separation_bits);
  append_integer(output, impulse.order_a);
  append_integer(output, impulse.order_b);
  append_integer(output, impulse.normal_impulse_bits);
  append_integer(output, impulse.tangent_impulse_bits);
}

ContactImpulse deserialize_contact_impulse(std::span<const std::byte> input,
                                           std::size_t& offset) {
  ContactImpulse impulse;
  impulse.a = read_integer<BodyId>(input, offset);
  impulse.b = read_integer<BodyId>(input, offset);
  impulse.boundary =
      static_cast<BoundaryKind>(read_integer<std::uint8_t>(input, offset));
  impulse.destroy_pending = read_bool(input, offset);
  impulse.manifold_count = read_integer<std::uint8_t>(input, offset);
  impulse.manifold_index = read_integer<std::uint8_t>(input, offset);
  impulse.point_count = read_integer<std::uint8_t>(input, offset);
  impulse.point_index = read_integer<std::uint8_t>(input, offset);
  impulse.contact_order = read_integer<std::uint32_t>(input, offset);
  impulse.feature_id = read_integer<std::uint32_t>(input, offset);
  impulse.normal_x_bits = read_integer<std::uint32_t>(input, offset);
  impulse.normal_y_bits = read_integer<std::uint32_t>(input, offset);
  impulse.point_x_bits = read_integer<std::uint32_t>(input, offset);
  impulse.point_y_bits = read_integer<std::uint32_t>(input, offset);
  impulse.separation_bits = read_integer<std::uint32_t>(input, offset);
  impulse.order_a = read_integer<std::uint32_t>(input, offset);
  impulse.order_b = read_integer<std::uint32_t>(input, offset);
  impulse.normal_impulse_bits = read_integer<std::uint32_t>(input, offset);
  impulse.tangent_impulse_bits = read_integer<std::uint32_t>(input, offset);
  return impulse;
}

Body deserialize_body(std::span<const std::byte> input, std::size_t& offset) {
  Body body;
  body.id = read_integer<BodyId>(input, offset);
  body.kind = static_cast<BodyKind>(read_integer<std::uint8_t>(input, offset));
  body.shape = static_cast<Shape>(read_integer<std::uint8_t>(input, offset));
  body.lifecycle = static_cast<Lifecycle>(read_integer<std::uint8_t>(input, offset));
  body.color = read_integer<std::int32_t>(input, offset);
  body.position.x = read_double(input, offset);
  body.position.y = read_double(input, offset);
  body.velocity.x = read_double(input, offset);
  body.velocity.y = read_double(input, offset);
  body.angle = read_double(input, offset);
  body.angular_velocity = read_double(input, offset);
  body.native_position.x = read_float(input, offset);
  body.native_position.y = read_float(input, offset);
  body.native_center.x = read_float(input, offset);
  body.native_center.y = read_float(input, offset);
  body.native_center_valid = true;
  body.native_velocity.x = read_float(input, offset);
  body.native_velocity.y = read_float(input, offset);
  body.native_angle = read_float(input, offset);
  body.native_angular_velocity = read_float(input, offset);
  body.native_state_valid = read_bool(input, offset);
  body.size = read_double(input, offset);
  body.density = read_double(input, offset);
  body.friction = read_double(input, offset);
  body.restitution = read_double(input, offset);
  body.inverse_mass = read_double(input, offset);
  body.inverse_inertia = read_double(input, offset);
  body.chain_id = read_integer<ChainId>(input, offset);
  body.actor_slot = read_integer<std::uint32_t>(input, offset);
  body.size_slot = read_integer<std::uint32_t>(input, offset);
  body.projectile_hits = read_integer<std::uint32_t>(input, offset);
  body.age_ticks = read_integer<std::uint64_t>(input, offset);
  body.remaining_lifetime = read_integer<std::int64_t>(input, offset);
  body.scripted_velocity.x = read_double(input, offset);
  body.scripted_velocity.y = read_double(input, offset);
  body.rot_timer = read_integer<std::uint64_t>(input, offset);
  body.physics_owned = read_bool(input, offset);
  body.special = read_bool(input, offset);
  body.freshness_state = read_integer<std::uint8_t>(input, offset);
  body.grouped = read_bool(input, offset);
  body.successful_clear_pending = read_bool(input, offset);
  body.non_wall_contacts = read_integer<std::uint32_t>(input, offset);
  body.top_contact_pending = read_bool(input, offset);
  body.top_contact_enabled = read_bool(input, offset);
  body.physics_update_count = read_integer<std::uint64_t>(input, offset);
  body.rule_guard_f0 = read_integer<std::uint8_t>(input, offset);
  body.delete_marked = read_bool(input, offset);
  body.pending_delete = read_bool(input, offset);
  body.sleeping = read_bool(input, offset);
  body.sleep_time = read_double(input, offset);
  return body;
}

void serialize_group(std::vector<std::byte>& output, const GroupState& group) {
  append_integer(output, group.id);
  append_integer(output, group.chain);
  append_integer(output, group.secondary_count);
  append_integer(output, group.num);
}

GroupState deserialize_group(std::span<const std::byte> input,
                             std::size_t& offset) {
  GroupState group;
  group.id = read_integer<ChainId>(input, offset);
  group.chain = read_integer<std::uint32_t>(input, offset);
  group.secondary_count = read_integer<std::uint32_t>(input, offset);
  group.num = read_integer<std::uint32_t>(input, offset);
  return group;
}

void validate_snapshot_payload(const MechanicsConfig& config, const Snapshot& snapshot,
                               std::uint64_t expected_config_hash) {
  if (snapshot.schema_version != kSnapshotVersion) {
    throw std::invalid_argument("snapshot schema mismatch");
  }
  if (snapshot.config_hash != expected_config_hash) {
    throw std::invalid_argument("snapshot mechanics profile mismatch");
  }
  if (snapshot.rng_index > DxRandom::state_words ||
      std::all_of(snapshot.rng_state.begin(), snapshot.rng_state.end(),
                  [](std::uint32_t word) { return word == 0; })) {
    throw std::invalid_argument("invalid DxLib RNG state");
  }
  const bool episode_limit_reached =
      snapshot.tick >= config.max_episode_ticks;
  const std::uint64_t active_clear_limit =
      static_cast<std::uint64_t>(config.maximum_level) *
          config.qualifying_clears_per_level -
      1U;
  const std::uint64_t terminal_clear_limit =
      active_clear_limit + (MechanicsConfig::actor_pool_capacity - 4U);
  const std::uint64_t special_offset_limit =
      static_cast<std::uint64_t>(config.special_clear_base) +
      config.special_clear_random_max;
  if (snapshot.qualifying_clear_count >
      std::numeric_limits<std::uint64_t>::max() - special_offset_limit) {
    throw std::invalid_argument("snapshot special-clear threshold overflows");
  }
  const std::uint64_t maximum_special_threshold =
      snapshot.qualifying_clear_count + special_offset_limit;
  if (snapshot.next_body_id == 0 ||
      snapshot.next_body_id > kExhaustedNextBodyId ||
      snapshot.next_chain_id == 0 || snapshot.level == 0 ||
      snapshot.level > config.maximum_level || snapshot.score < 0 ||
      snapshot.gauge > config.gauge_max ||
      (snapshot.terminated && snapshot.truncated) ||
      snapshot.tick > config.max_episode_ticks ||
      (!snapshot.terminated && snapshot.truncated != episode_limit_reached) ||
      ((!snapshot.terminated && !snapshot.truncated) &&
       snapshot.scene_frame == std::numeric_limits<std::uint64_t>::max()) ||
      snapshot.spawn_count > snapshot.tick ||
      snapshot.spawn_count >= snapshot.next_body_id ||
      snapshot.qualifying_clear_count >
          (snapshot.terminated ? terminal_clear_limit : active_clear_limit) ||
      snapshot.next_special_clear_count > maximum_special_threshold ||
      snapshot.level_shape_cutoff > config.shape_random_max ||
      snapshot.finish_call_count >
          MechanicsConfig::actor_pool_capacity - 3U ||
      snapshot.bodies.size() > kMaximumSnapshotBodies ||
      snapshot.groups.size() > kMaximumSnapshotGroups) {
    throw std::invalid_argument("snapshot global state is inconsistent");
  }
  const bool recorded_terminal_valid =
      snapshot.terminated && snapshot.terminal_metadata_recorded &&
      snapshot.finish_call_count > 0 &&
      snapshot.recorded_final_score >= 0 && snapshot.latest_final_score >= 0 &&
      snapshot.recorded_final_level >= 1 &&
      snapshot.recorded_final_level <= config.maximum_level &&
      snapshot.latest_final_level >= 1 &&
      snapshot.latest_final_level <= config.maximum_level;
  const bool empty_terminal_valid =
      !snapshot.terminated && !snapshot.terminal_metadata_recorded &&
      snapshot.finish_call_count == 0 &&
      snapshot.recorded_final_score == 0 &&
      snapshot.recorded_final_highest_chain == 0 &&
      snapshot.recorded_final_level == 0 &&
      snapshot.recorded_final_clears == 0 && snapshot.latest_final_score == 0 &&
      snapshot.latest_final_highest_chain == 0 &&
      snapshot.latest_final_level == 0 && snapshot.latest_final_clears == 0;
  if (!recorded_terminal_valid && !empty_terminal_valid) {
    throw std::invalid_argument("snapshot terminal metadata is inconsistent");
  }
  if (recorded_terminal_valid &&
      (snapshot.recorded_final_score > snapshot.latest_final_score ||
       snapshot.latest_final_score > snapshot.score ||
       snapshot.recorded_final_highest_chain >
           snapshot.latest_final_highest_chain ||
       snapshot.latest_final_highest_chain > snapshot.highest_chain ||
       snapshot.recorded_final_level > snapshot.latest_final_level ||
       snapshot.latest_final_level > snapshot.level ||
       snapshot.recorded_final_clears > snapshot.latest_final_clears ||
       snapshot.latest_final_clears > snapshot.qualifying_clear_count)) {
    throw std::invalid_argument(
        "snapshot terminal metadata does not describe monotonic game state");
  }
  BodyId previous_id = 0;
  ChainId greatest_chain_id = 0;
  std::size_t live_body_count = 0;
  for (const auto& body : snapshot.bodies) {
    const bool enum_valid = static_cast<std::uint8_t>(body.kind) <=
                                static_cast<std::uint8_t>(BodyKind::Bonus) &&
                            static_cast<std::uint8_t>(body.shape) <=
                                static_cast<std::uint8_t>(Shape::Triangle) &&
                            static_cast<std::uint8_t>(body.lifecycle) <=
                                static_cast<std::uint8_t>(Lifecycle::Deleted);
    const bool numbers_valid =
        std::isfinite(body.position.x) && std::isfinite(body.position.y) &&
        std::isfinite(body.velocity.x) && std::isfinite(body.velocity.y) &&
        std::isfinite(body.angle) && std::isfinite(body.angular_velocity) &&
        std::isfinite(body.size) && body.size > 0.0 && std::isfinite(body.density) &&
        body.density >= 0.0 && std::isfinite(body.friction) && body.friction >= 0.0 &&
        std::isfinite(body.restitution) && body.restitution >= 0.0 &&
        std::isfinite(body.inverse_mass) && body.inverse_mass >= 0.0 &&
        std::isfinite(body.inverse_inertia) && body.inverse_inertia >= 0.0 &&
        std::isfinite(body.scripted_velocity.x) &&
        std::isfinite(body.scripted_velocity.y) &&
        std::isfinite(body.native_position.x) &&
        std::isfinite(body.native_position.y) &&
        std::isfinite(body.native_center.x) &&
        std::isfinite(body.native_center.y) &&
        std::isfinite(body.native_velocity.x) &&
        std::isfinite(body.native_velocity.y) &&
        std::isfinite(body.native_angle) &&
        std::isfinite(body.native_angular_velocity) &&
        std::isfinite(body.sleep_time) && body.sleep_time >= 0.0 &&
        body.restitution <= 1.0 && body.remaining_lifetime >= -1;
    const bool piece_identity =
        body.kind != BodyKind::Piece ||
        (!body.special && body.color >= 0 &&
         static_cast<std::uint32_t>(body.color) < config.maximum_colors &&
         body.density == config.piece_density &&
         body.friction == config.piece_friction &&
         body.restitution == config.piece_restitution);
    const bool projectile_identity =
        body.kind != BodyKind::Projectile ||
        (!body.special && body.shape == Shape::Box && body.color == -1 &&
         body.size == config.projectile_size &&
         body.density == config.projectile_density &&
         body.friction == config.projectile_friction &&
         body.restitution == config.projectile_restitution &&
         body.projectile_hits == 0);
    const bool bonus_identity =
        body.kind != BodyKind::Bonus ||
        (body.special && body.shape == Shape::Circle && body.color == -2 &&
         body.size == config.bonus_size &&
         body.density == config.bonus_density &&
         body.friction == config.bonus_friction &&
         body.restitution == config.bonus_restitution &&
         body.projectile_hits == 0);
    const Lifecycle projected = body.pending_delete
                                    ? Lifecycle::Deleted
                                    : body.freshness_state == 3
                                          ? Lifecycle::Rotten
                                          : body.grouped
                                                ? Lifecycle::Confirmed
                                                : body.physics_owned
                                                      ? Lifecycle::DynamicFresh
                                                      : Lifecycle::ScriptedFalling;
    const bool lifecycle_valid =
        body.freshness_state >= 1 && body.freshness_state <= 3 &&
        body.native_state_valid && body.native_center_valid &&
        body.lifecycle == projected && body.grouped == (body.chain_id != 0) &&
        body.size_slot < config.piece_sizes.size() && body.rule_guard_f0 == 0 &&
        body.actor_slot < MechanicsConfig::actor_pool_capacity &&
        (body.actor_slot == 0 || body.actor_slot >= 5) &&
        !(body.delete_marked && body.pending_delete) &&
        (body.remaining_lifetime > 0 || body.delete_marked ||
         body.pending_delete) &&
        (!body.successful_clear_pending ||
         (body.grouped && body.kind != BodyKind::Projectile)) &&
        piece_identity && projectile_identity && bonus_identity &&
        (body.freshness_state != 3 || body.kind == BodyKind::Piece) &&
        body.projectile_hits <= 2 &&
        (body.kind != BodyKind::Projectile || body.physics_owned) &&
        ((snapshot.terminated || snapshot.truncated) ||
         (body.age_ticks != std::numeric_limits<std::uint64_t>::max() &&
          body.rot_timer != std::numeric_limits<std::uint64_t>::max() &&
          body.physics_update_count !=
              std::numeric_limits<std::uint64_t>::max())) &&
        ((body.density == 0.0 && body.inverse_mass == 0.0 &&
          body.inverse_inertia == 0.0 && !body.sleeping) ||
         (body.density > 0.0 && body.inverse_mass > 0.0 &&
          body.inverse_inertia > 0.0)) &&
        (!body.sleeping || body.inverse_mass > 0.0);
    if (!enum_valid || !numbers_valid || !lifecycle_valid || body.id == 0 ||
        body.id <= previous_id || body.id >= snapshot.next_body_id ||
        snapshot.actor_pool_colors[body.actor_slot] != body.color) {
      throw std::invalid_argument("snapshot body state is invalid or non-canonical");
    }
    validate_body_numeric_safety(config, body);
    if (body.lifecycle != Lifecycle::Deleted) ++live_body_count;
    greatest_chain_id = std::max(greatest_chain_id, body.chain_id);
    previous_id = body.id;
  }
  std::vector<std::uint32_t> occupied_actor_slots;
  for (const auto& body : snapshot.bodies) {
    if (body.lifecycle != Lifecycle::Deleted) {
      occupied_actor_slots.push_back(body.actor_slot);
    }
  }
  std::sort(occupied_actor_slots.begin(), occupied_actor_slots.end());
  if (occupied_actor_slots.size() > MechanicsConfig::actor_pool_capacity - 4U ||
      std::adjacent_find(occupied_actor_slots.begin(),
                         occupied_actor_slots.end()) !=
          occupied_actor_slots.end() ||
      snapshot.actor_pool_cursor >= MechanicsConfig::actor_pool_capacity) {
    throw std::invalid_argument("snapshot actor-pool state is invalid");
  }
  if (std::any_of(snapshot.actor_pool_colors.begin(),
                  snapshot.actor_pool_colors.end(), [&](std::int32_t color) {
                    return color < -2 ||
                           color >= static_cast<std::int32_t>(
                                        config.maximum_colors);
                  })) {
    throw std::invalid_argument("snapshot actor-pool color is invalid");
  }
  constexpr auto dynamic_proxy_capacity =
      MechanicsConfig::physics_proxy_capacity - MechanicsConfig::static_fixture_count;
  if (live_body_count > dynamic_proxy_capacity ||
      greatest_chain_id >= snapshot.next_chain_id) {
    throw std::invalid_argument("snapshot exceeds r58 body capacity or reuses a chain id");
  }
  ChainId previous_group_id = 0;
  for (const auto& group : snapshot.groups) {
    if (group.id == 0 || group.id <= previous_group_id ||
        group.id >= snapshot.next_chain_id || group.chain == 0 ||
        group.secondary_count != group.chain || group.num > group.chain) {
      throw std::invalid_argument("snapshot group state is not canonical");
    }
    previous_group_id = group.id;
  }
  std::vector<std::uint32_t> uncleared_live_members(snapshot.groups.size());
  for (const auto& body : snapshot.bodies) {
    if (body.chain_id == 0) continue;
    const auto found = std::lower_bound(
        snapshot.groups.begin(), snapshot.groups.end(), body.chain_id,
        [](const GroupState& group, ChainId id) { return group.id < id; });
    if (found == snapshot.groups.end() || found->id != body.chain_id) {
      throw std::invalid_argument("snapshot body refers to an absent group");
    }
    if (body.lifecycle != Lifecycle::Deleted &&
        !body.successful_clear_pending) {
      ++uncleared_live_members[static_cast<std::size_t>(
          std::distance(snapshot.groups.begin(), found))];
    }
  }
  for (std::size_t index = 0; index < snapshot.groups.size(); ++index) {
    const auto& group = snapshot.groups[index];
    if (uncleared_live_members[index] > group.chain - group.num) {
      throw std::invalid_argument(
          "snapshot group counters cannot describe their live members");
    }
  }
  std::vector<BodyId> live_ids;
  std::vector<BodyId> pending_ids;
  std::vector<BodyId> native_ids;
  live_ids.reserve(live_body_count);
  pending_ids.reserve(snapshot.bodies.size() - live_body_count);
  native_ids.reserve(snapshot.bodies.size());
  for (const auto& body : snapshot.bodies) {
    native_ids.push_back(body.id);
    if (body.lifecycle != Lifecycle::Deleted) {
      live_ids.push_back(body.id);
    } else {
      pending_ids.push_back(body.id);
    }
  }
  const auto valid_physics_order = [](const std::vector<BodyId>& order,
                                      const std::vector<BodyId>& expected) {
    if (order.size() != expected.size()) return false;
    auto sorted = order;
    std::sort(sorted.begin(), sorted.end());
    return sorted == expected;
  };
  if (!valid_physics_order(snapshot.physics_ordering.body_order, live_ids) ||
      !valid_physics_order(snapshot.physics_ordering.destroy_order, pending_ids) ||
      !valid_physics_order(snapshot.physics_ordering.proxy_order, native_ids)) {
    throw std::invalid_argument("snapshot physics ordering is not canonical");
  }
  constexpr std::uint32_t null_proxy =
      std::numeric_limits<std::uint16_t>::max();
  std::vector<std::uint32_t> allocated_proxy_ids;
  std::copy_if(snapshot.physics_ordering.proxy_ids.begin(),
               snapshot.physics_ordering.proxy_ids.end(),
               std::back_inserter(allocated_proxy_ids),
               [&](std::uint32_t proxy) { return proxy != null_proxy; });
  auto all_proxy_ids = allocated_proxy_ids;
  all_proxy_ids.insert(all_proxy_ids.end(),
                       snapshot.physics_ordering.free_proxy_order.begin(),
                       snapshot.physics_ordering.free_proxy_order.end());
  std::sort(all_proxy_ids.begin(), all_proxy_ids.end());
  const bool proxy_partition_valid =
      snapshot.physics_ordering.proxy_ids.size() == native_ids.size() &&
      snapshot.physics_ordering.proxy_ids.size() ==
          snapshot.physics_ordering.proxy_order.size() &&
      all_proxy_ids.size() == dynamic_proxy_capacity &&
      all_proxy_ids.front() == MechanicsConfig::static_fixture_count &&
      all_proxy_ids.back() == MechanicsConfig::physics_proxy_capacity - 1U &&
      std::adjacent_find(all_proxy_ids.begin(), all_proxy_ids.end()) ==
          all_proxy_ids.end() &&
      std::is_sorted(snapshot.physics_ordering.proxy_ids.begin(),
                     snapshot.physics_ordering.proxy_ids.end()) &&
      std::all_of(snapshot.physics_ordering.proxy_ids.begin(),
                  snapshot.physics_ordering.proxy_ids.end(),
                  [&](std::uint32_t proxy) {
                    return proxy == null_proxy ||
                           (proxy >= MechanicsConfig::static_fixture_count &&
                            proxy < MechanicsConfig::physics_proxy_capacity);
                  });
  if (!proxy_partition_valid) {
    throw std::invalid_argument(
        "snapshot broad-phase proxy partition is invalid (live=" +
        std::to_string(live_ids.size()) + ", occupied=" +
        std::to_string(snapshot.physics_ordering.proxy_ids.size()) +
        ", free=" +
        std::to_string(snapshot.physics_ordering.free_proxy_order.size()) +
        ", total=" + std::to_string(all_proxy_ids.size()) + ")");
  }
  const std::size_t native_proxy_count =
      MechanicsConfig::static_fixture_count + allocated_proxy_ids.size();
  const std::size_t axis_bound_count = 2U * native_proxy_count;
  const bool broadphase_dimensions_valid =
      (snapshot.physics_ordering.static_sleep_flags & 0xf0U) == 0 &&
      snapshot.physics_ordering.broadphase_time_stamp != 0 &&
      snapshot.physics_ordering.proxy_time_stamps.size() ==
          MechanicsConfig::physics_proxy_capacity &&
      snapshot.physics_ordering.proxy_overlap_counts.size() ==
          MechanicsConfig::physics_proxy_capacity &&
      snapshot.physics_ordering.broadphase_bounds.size() ==
          2U * axis_bound_count;
  if (!broadphase_dimensions_valid) {
    throw std::invalid_argument("snapshot broad-phase state has invalid dimensions");
  }
  std::vector<bool> occupied_proxy(MechanicsConfig::physics_proxy_capacity);
  for (std::uint32_t proxy = 0;
       proxy < MechanicsConfig::static_fixture_count; ++proxy) {
    occupied_proxy[proxy] = true;
  }
  for (const auto proxy : allocated_proxy_ids) {
    occupied_proxy[proxy] = true;
  }
  for (std::size_t proxy = 0; proxy < occupied_proxy.size(); ++proxy) {
    const bool overlap_valid =
        snapshot.physics_ordering.proxy_overlap_counts[proxy] !=
        std::numeric_limits<std::uint16_t>::max();
    if (overlap_valid != occupied_proxy[proxy]) {
      throw std::invalid_argument(
          "snapshot broad-phase proxy metadata is inconsistent");
    }
  }
  for (std::size_t axis = 0; axis < 2; ++axis) {
    std::vector<std::uint8_t> endpoint_kinds(
        MechanicsConfig::physics_proxy_capacity);
    std::uint32_t stabbing = 0;
    std::uint16_t previous_value = 0;
    for (std::size_t index = 0; index < axis_bound_count; ++index) {
      const auto& bound = snapshot.physics_ordering.broadphase_bounds[
          axis * axis_bound_count + index];
      if (bound.proxy_id >= occupied_proxy.size() ||
          !occupied_proxy[bound.proxy_id] ||
          (index != 0 && bound.value < previous_value)) {
        throw std::invalid_argument("snapshot broad-phase bound is invalid");
      }
      previous_value = bound.value;
      if ((bound.value & 1U) == 0) {
        endpoint_kinds[bound.proxy_id] |= 1U;
        ++stabbing;
      } else {
        endpoint_kinds[bound.proxy_id] |= 2U;
        if (stabbing == 0) {
          throw std::invalid_argument(
              "snapshot broad-phase stabbing count underflows");
        }
        --stabbing;
      }
      if (bound.stabbing_count != stabbing) {
        throw std::invalid_argument(
            "snapshot broad-phase stabbing metadata is invalid");
      }
    }
    bool endpoints_complete = stabbing == 0;
    for (std::size_t proxy = 0; proxy < occupied_proxy.size(); ++proxy) {
      if (occupied_proxy[proxy] && endpoint_kinds[proxy] != 3U) {
        endpoints_complete = false;
      }
    }
    if (!endpoints_complete) {
      throw std::invalid_argument(
          "snapshot broad-phase endpoints are not a complete partition");
    }
  }
  if (!std::is_sorted(snapshot.active_contact_pairs.begin(), snapshot.active_contact_pairs.end()) ||
      std::adjacent_find(snapshot.active_contact_pairs.begin(),
                         snapshot.active_contact_pairs.end()) !=
          snapshot.active_contact_pairs.end()) {
    throw std::invalid_argument("snapshot contact pairs are not canonical");
  }
  const auto body_present = [&](BodyId id) {
    const auto found = std::lower_bound(
        snapshot.bodies.begin(), snapshot.bodies.end(), id,
        [](const Body& body, BodyId wanted) { return body.id < wanted; });
    return found != snapshot.bodies.end() && found->id == id;
  };
  const std::uint64_t maximum_contact_pairs =
      static_cast<std::uint64_t>(snapshot.bodies.size()) *
          (snapshot.bodies.size() - (snapshot.bodies.empty() ? 0U : 1U)) /
          2U +
      4U * snapshot.bodies.size();
  if (snapshot.active_contact_pairs.size() > maximum_contact_pairs ||
      snapshot.contact_impulses.size() > 2U * 4096U) {
    throw std::invalid_argument("snapshot contact collections exceed capacity");
  }
  for (const auto key : snapshot.active_contact_pairs) {
    if ((key >> 63U) != 0) {
      const auto body = static_cast<BodyId>(
          (key & ~(std::uint64_t{1} << 63U)) >> 3U);
      const auto boundary = static_cast<BoundaryKind>(key & 7U);
      if (body == 0 || body >= snapshot.next_body_id || !body_present(body) ||
          boundary == BoundaryKind::None || boundary > BoundaryKind::Top ||
          key != ((std::uint64_t{1} << 63U) |
                  (static_cast<std::uint64_t>(body) << 3U) |
                  static_cast<std::uint8_t>(boundary))) {
        throw std::invalid_argument("snapshot boundary contact key is invalid");
      }
    } else {
      const auto a = static_cast<BodyId>(key >> 32U);
      const auto b = static_cast<BodyId>(key);
      if (a == 0 || a >= b || b >= snapshot.next_body_id ||
          !body_present(a) || !body_present(b)) {
        throw std::invalid_argument("snapshot body contact key is invalid");
      }
    }
  }

  const auto impulse_less = [](const ContactImpulse& left,
                               const ContactImpulse& right) {
    // Matches PhysicsWorld's pending-first duplicate-contact encoding.
    return std::tuple{left.a, left.b, left.boundary, !left.destroy_pending,
                      left.contact_order, left.manifold_index,
                      left.point_index} <
           std::tuple{right.a, right.b, right.boundary,
                      !right.destroy_pending, right.contact_order,
                      right.manifold_index, right.point_index};
  };
  if (!std::is_sorted(snapshot.contact_impulses.begin(),
                      snapshot.contact_impulses.end(), impulse_less) ||
      std::adjacent_find(snapshot.contact_impulses.begin(),
                         snapshot.contact_impulses.end(),
                         [&](const auto& left, const auto& right) {
                           return !impulse_less(left, right) &&
                                  !impulse_less(right, left);
                         }) != snapshot.contact_impulses.end()) {
    throw std::invalid_argument("snapshot contact impulses are not canonical");
  }
  const auto present_in_native_world = [&](BodyId id) {
    const auto found = std::lower_bound(
        snapshot.bodies.begin(), snapshot.bodies.end(), id,
        [](const Body& body, BodyId wanted) { return body.id < wanted; });
    return found != snapshot.bodies.end() && found->id == id;
  };
  std::vector<std::uint32_t> contact_orders;
  std::set<std::tuple<BodyId, BodyId, BoundaryKind>> live_contact_identities;
  for (std::size_t first = 0; first < snapshot.contact_impulses.size();) {
    std::size_t last = first + 1;
    while (last < snapshot.contact_impulses.size() &&
           snapshot.contact_impulses[last].a ==
               snapshot.contact_impulses[first].a &&
           snapshot.contact_impulses[last].b ==
               snapshot.contact_impulses[first].b &&
           snapshot.contact_impulses[last].boundary ==
               snapshot.contact_impulses[first].boundary &&
           snapshot.contact_impulses[last].contact_order ==
               snapshot.contact_impulses[first].contact_order) {
      ++last;
    }
    const auto& impulse = snapshot.contact_impulses[first];
    const bool identity_valid =
        impulse.a != 0 && impulse.a < snapshot.next_body_id &&
        present_in_native_world(impulse.a) &&
        ((impulse.b == 0 && impulse.boundary > BoundaryKind::None &&
          impulse.boundary <= BoundaryKind::Top) ||
         (impulse.b > impulse.a && impulse.b < snapshot.next_body_id &&
          impulse.boundary == BoundaryKind::None &&
          present_in_native_world(impulse.b)));
    const bool zero_manifold =
        impulse.manifold_count == 0 && impulse.manifold_index == 0 &&
        impulse.point_count == 0 && impulse.point_index == 0 && last == first + 1 &&
        impulse.order_a == std::numeric_limits<std::uint32_t>::max() &&
        impulse.order_b == std::numeric_limits<std::uint32_t>::max();
    const bool touching_manifold =
        impulse.manifold_count == 1 && impulse.manifold_index == 0 &&
        impulse.point_count >= 1 &&
        impulse.point_count <= kMaxManifoldPoints &&
        last - first == impulse.point_count &&
        impulse.order_a < dynamic_proxy_capacity &&
        (impulse.b == 0 ? impulse.order_b == 0
                        : impulse.order_b < dynamic_proxy_capacity);
    if (!identity_valid || impulse.contact_order >= 4096 ||
        (!zero_manifold && !touching_manifold)) {
      throw std::invalid_argument("snapshot contact impulse is invalid");
    }
    if (!impulse.destroy_pending &&
        !live_contact_identities
             .emplace(impulse.a, impulse.b, impulse.boundary)
             .second) {
      throw std::invalid_argument(
          "snapshot has multiple live contacts for one broad-phase pair");
    }
    for (std::size_t index = first; index < last; ++index) {
      const auto& point = snapshot.contact_impulses[index];
      const float normal_x = std::bit_cast<float>(point.normal_x_bits);
      const float normal_y = std::bit_cast<float>(point.normal_y_bits);
      const float point_x = std::bit_cast<float>(point.point_x_bits);
      const float point_y = std::bit_cast<float>(point.point_y_bits);
      const float separation = std::bit_cast<float>(point.separation_bits);
      const float normal_impulse =
          std::bit_cast<float>(point.normal_impulse_bits);
      const float tangent_impulse =
          std::bit_cast<float>(point.tangent_impulse_bits);
      const bool group_consistent =
          point.destroy_pending == impulse.destroy_pending &&
          point.manifold_count == impulse.manifold_count &&
          point.manifold_index == impulse.manifold_index &&
          point.point_count == impulse.point_count &&
          point.point_index == index - first &&
          point.contact_order == impulse.contact_order &&
          point.normal_x_bits == impulse.normal_x_bits &&
          point.normal_y_bits == impulse.normal_y_bits &&
          point.order_a == impulse.order_a && point.order_b == impulse.order_b;
      const long double normal_length_squared =
          static_cast<long double>(normal_x) * normal_x +
          static_cast<long double>(normal_y) * normal_y;
      const long double manifold_headroom =
          std::sqrt(static_cast<long double>(
              std::numeric_limits<float>::max())) /
          16.0L;
      const auto within_manifold_headroom = [&](float value) {
        return std::abs(static_cast<long double>(value)) <=
               manifold_headroom;
      };
      const bool normal_valid =
          zero_manifold ||
          (normal_length_squared >= 0.9L && normal_length_squared <= 1.1L);
      if (!group_consistent || !std::isfinite(normal_x) ||
          !std::isfinite(normal_y) || !std::isfinite(point_x) ||
          !std::isfinite(point_y) || !std::isfinite(separation) ||
          !std::isfinite(normal_impulse) || normal_impulse < 0.0f ||
          !std::isfinite(tangent_impulse) || !normal_valid ||
          !within_manifold_headroom(point_x) ||
          !within_manifold_headroom(point_y) ||
          !within_manifold_headroom(separation) ||
          !within_manifold_headroom(normal_impulse) ||
          !within_manifold_headroom(tangent_impulse)) {
        throw std::invalid_argument("snapshot manifold point is invalid");
      }
    }
    contact_orders.push_back(impulse.contact_order);
    first = last;
  }
  std::sort(contact_orders.begin(), contact_orders.end());
  if (std::adjacent_find(contact_orders.begin(), contact_orders.end()) !=
      contact_orders.end()) {
    throw std::invalid_argument("snapshot contact-list orders are duplicated");
  }
}

}  // namespace

Simulator::Simulator(MechanicsConfig config)
    : config_(validated_mechanics_config(std::move(config))), physics_(config_) {
  config_hash_ = calculate_config_hash();
  reset(0);
}

Observation Simulator::reset(std::uint64_t seed_value) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  if (seed_value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("normal-mode seed must fit in uint32");
  }
  rng_.seed(static_cast<std::uint32_t>(seed_value));
  bodies_.clear();
  groups_.clear();
  physics_.reset();
  active_contact_pairs_.clear();
  tick_ = 0;
  scene_frame_ = 0;
  next_body_id_ = 1;
  next_chain_id_ = 1;
  actor_pool_cursor_ = 4;
  actor_pool_colors_.fill(-1);
  next_event_sequence_ = 0;
  spawn_count_ = 0;
  score_ = 0;
  gauge_ = config_.gauge_initial;
  level_ = 1;
  qualifying_clear_count_ = 0;
  level_shape_cutoff_ = rng_.get_rand(config_.shape_random_max);
  next_special_clear_count_ =
      static_cast<std::uint64_t>(config_.special_clear_base) +
      rng_.get_rand(config_.special_clear_random_max);
  highest_chain_ = 0;
  finish_call_count_ = 0;
  terminal_metadata_recorded_ = false;
  recorded_final_score_ = 0;
  recorded_final_highest_chain_ = 0;
  recorded_final_level_ = 0;
  recorded_final_clears_ = 0;
  latest_final_score_ = 0;
  latest_final_highest_chain_ = 0;
  latest_final_level_ = 0;
  latest_final_clears_ = 0;
  previous_left_level_ = false;
  previous_right_level_ = false;
  terminated_ = false;
  truncated_ = false;
  actor_counter_guard_required_ = false;
  numeric_guard_required_ = false;
  for (std::uint32_t index = 0; index < config_.initial_rotten_count; ++index) {
    const BodyId id = spawn_random_piece(config_.initial_rotten_y, level_);
    Body* body = find_body(id);
    if (body == nullptr) {
      throw std::logic_error("validated initial rotten fill exhausted the actor pool");
    }
    body->freshness_state = 3;
    body->rot_timer = 1;
    body->physics_owned = true;
    body->top_contact_enabled = true;
    refresh_lifecycle(*body);
  }
  for (std::uint32_t index = 0; index < config_.initial_falling_count; ++index) {
    if (spawn_random_piece(config_.initial_falling_y, level_) == 0) {
      throw std::logic_error("validated initial falling fill exhausted the actor pool");
    }
  }
  // The scene transition runs one actor-pool pass after construction, before
  // replay word 0, without stepping Box2D or advancing the scene counter.
  std::vector<std::tuple<BodyId, Vec2, double>> scripted_origins;
  for (const auto& body : bodies_) {
    if (!body.physics_owned) {
      scripted_origins.emplace_back(body.id, body.position, body.angle);
    }
  }
  std::vector<Event> ignored_events;
  process_actor_updates(scripted_origins, ignored_events);
  physics_.synchronize(bodies_);
  return observation();
}

StepResult Simulator::step(const Action& action) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  StepResult result;
  if (terminated_ || truncated_) {
    result.terminated = terminated_;
    result.truncated = truncated_;
    result.diagnostics = diagnostics();
    return result;
  }
  const auto score_before = score_;
  std::uint32_t ticks = 1;
  bool left_level = false;
  bool right_level = false;
  if (action.kind == ActionKind::Wait) {
    if (action.wait_ticks == 0 || action.wait_ticks > 100'000) {
      result.events.push_back({tick_, EventKind::InvalidAction, 0, 0, action.wait_ticks,
                               "wait_ticks must be in [1, 100000]"});
      ticks = 0;
    } else {
      ticks = action.wait_ticks;
    }
  } else if (action.kind == ActionKind::WeakShot ||
             action.kind == ActionKind::StrongShot ||
             action.kind == ActionKind::BothShots) {
    const bool valid_coordinate = std::isfinite(action.cursor_x) &&
                                  std::isfinite(action.cursor_y) &&
                                  action.cursor_x >= 0.0 &&
                                  action.cursor_x <= 1023.0 &&
                                  action.cursor_y >= 0.0 &&
                                  action.cursor_y <= 511.0;
    if (!valid_coordinate) {
      result.events.push_back({tick_, EventKind::InvalidAction, 0, 0, 0,
                               "cursor outside encoded replay range"});
    } else {
      left_level = action.kind == ActionKind::WeakShot ||
                   action.kind == ActionKind::BothShots;
      right_level = action.kind == ActionKind::StrongShot ||
                    action.kind == ActionKind::BothShots;
    }
  } else {
    result.events.push_back({tick_, EventKind::InvalidAction, 0, 0,
                             static_cast<std::int64_t>(static_cast<std::uint8_t>(action.kind)),
                             "unknown action kind"});
    ticks = 0;
  }
  if (ticks != 0) {
    const auto available_ticks = config_.max_episode_ticks - tick_;
    const auto ticks_to_run =
        std::min<std::uint64_t>(ticks, available_ticks);
    const bool episode_will_end = ticks_to_run == available_ticks;
    const auto counter_has_room = [&](std::uint64_t value) {
      return value <=
             std::numeric_limits<std::uint64_t>::max() - ticks_to_run;
    };
    if (!counter_has_room(tick_) || !counter_has_room(scene_frame_) ||
        (!episode_will_end &&
         scene_frame_ ==
             std::numeric_limits<std::uint64_t>::max() - ticks_to_run)) {
      throw std::overflow_error("simulation frame counter exhausted");
    }
    if (actor_counter_guard_required_ ||
        tick_ > std::numeric_limits<std::uint64_t>::max() - 100'000U) {
      for (const auto& body : bodies_) {
        if (!live(body)) continue;
        if (!counter_has_room(body.age_ticks) ||
            !counter_has_room(body.physics_update_count) ||
            (body.rot_timer != 0 && !counter_has_room(body.rot_timer)) ||
            (!episode_will_end &&
             (body.age_ticks ==
                  std::numeric_limits<std::uint64_t>::max() - ticks_to_run ||
              body.physics_update_count ==
                  std::numeric_limits<std::uint64_t>::max() - ticks_to_run ||
              (body.rot_timer != 0 &&
               body.rot_timer ==
                   std::numeric_limits<std::uint64_t>::max() -
                       ticks_to_run)))) {
          throw std::overflow_error("actor update counter exhausted");
        }
      }
    }
    if (numeric_guard_required_) {
      for (const auto& body : bodies_) {
        if (!live(body)) continue;
        std::uint64_t body_horizon = ticks_to_run;
        if (body.delete_marked || body.pending_delete) {
          body_horizon = std::min<std::uint64_t>(body_horizon, 1U);
        } else if (body.remaining_lifetime > 0) {
          const auto lifetime_horizon =
              static_cast<std::uint64_t>(body.remaining_lifetime) + 1U;
          body_horizon = std::min(body_horizon, lifetime_horizon);
        }
        try {
          validate_body_numeric_safety(config_, body, body_horizon);
        } catch (const std::invalid_argument&) {
          throw std::overflow_error("actor numeric state would overflow");
        }
      }
    }
    // A step is capped at 100,000 frames, fewer than 4,096 native contacts
    // can exist per frame, and the actor pool has 196 usable slots. This
    // reserve is deliberately much larger than the maximum emitted batch.
    constexpr std::uint64_t event_sequence_reserve = std::uint64_t{1} << 48U;
    if (next_event_sequence_ >
        std::numeric_limits<std::uint64_t>::max() -
            event_sequence_reserve) {
      throw std::overflow_error("event sequence counter exhausted");
    }
  }
  for (std::uint32_t index = 0; index < ticks && !terminated_ && !truncated_; ++index) {
    tick_once(left_level, right_level, action.cursor_x, action.cursor_y,
              action.suppress_fresh_edges, result.events);
    if (!terminated_ && tick_ >= config_.max_episode_ticks) truncated_ = true;
  }
  result.reward = score_ - score_before;
  result.terminated = terminated_;
  result.truncated = truncated_;
  sequence_events(result.events);
  result.diagnostics = diagnostics();
  return result;
}

StepDiagnostics Simulator::diagnostics() const {
  StepDiagnostics result;
  result.config_hash = config_hash();
  result.finish_call_count = finish_call_count_;
  result.terminal_metadata_recorded = terminal_metadata_recorded_;
  result.recorded_final_score = recorded_final_score_;
  result.recorded_final_highest_chain = recorded_final_highest_chain_;
  result.recorded_final_level = recorded_final_level_;
  result.recorded_final_clears = recorded_final_clears_;
  result.latest_final_score = latest_final_score_;
  result.latest_final_highest_chain = latest_final_highest_chain_;
  result.latest_final_level = latest_final_level_;
  result.latest_final_clears = latest_final_clears_;
  return result;
}

Observation Simulator::observation() const {
  const ScopedFloatingPointEnvironment floating_point_environment;
  Observation result{tick_, score_, gauge_, level_, terminated_, truncated_, {}};
  result.bodies.reserve(bodies_.size());
  for (const auto& body : bodies_) {
    if (!live(body)) continue;
    ObservedBody observed;
    observed.id = body.id;
    observed.kind = body.kind;
    observed.shape = body.shape;
    observed.lifecycle = body.lifecycle;
    observed.color = body.color;
    observed.position = body.position;
    observed.velocity = body.velocity;
    observed.angle = body.angle;
    observed.angular_velocity = body.angular_velocity;
    observed.size = body.size;
    observed.chain_id = body.chain_id;
    observed.projectile_hits = body.projectile_hits;
    observed.age_ticks = body.age_ticks;
    observed.remaining_lifetime = body.remaining_lifetime;
    observed.rot_timer = body.rot_timer;
    result.bodies.push_back(observed);
  }
  result.field_x = config_.field_x;
  result.field_y = config_.field_y;
  result.field_width = config_.field_width;
  result.field_height = config_.field_height;
  result.side_wall_top = config_.side_wall_top;
  result.side_wall_bottom = config_.side_wall_bottom;
  result.gauge_max = config_.gauge_max;
  result.active_colors = current_color_count();
  result.current_spawn_interval_ticks = current_spawn_interval();
  result.left_held = previous_left_level_;
  result.right_held = previous_right_level_;
  result.highest_chain = highest_chain_;
  result.qualifying_clear_count = qualifying_clear_count_;
  return result;
}

std::uint32_t Simulator::current_color_count() const {
  return color_count_for_level(level_);
}

std::uint32_t Simulator::color_count_for_level(
    std::uint32_t parameter_level) const {
  const auto recovered =
      normal_level_parameters(std::min<std::uint32_t>(parameter_level, 99U))
          .maximum_color_id +
      1U;
  return std::clamp(recovered, config_.starting_colors, config_.maximum_colors);
}

std::uint32_t Simulator::current_spawn_interval() const {
  return spawn_interval_for_level(level_);
}

std::uint32_t Simulator::spawn_interval_for_level(
    std::uint32_t parameter_level) const {
  if (config_.spawn_interval_ticks != 100U) return config_.spawn_interval_ticks;
  return normal_level_parameters(std::min<std::uint32_t>(parameter_level, 99U))
      .spawn_interval_frames;
}

std::uint32_t Simulator::allocate_actor_slot() {
  std::array<bool, MechanicsConfig::actor_pool_capacity> occupied{};
  occupied[1] = true;
  occupied[2] = true;
  occupied[3] = true;
  occupied[4] = true;
  for (const auto& body : bodies_) {
    if (body.lifecycle != Lifecycle::Deleted) occupied[body.actor_slot] = true;
  }
  const auto starting_cursor = actor_pool_cursor_;
  for (std::uint32_t probe = 0; probe < MechanicsConfig::actor_pool_capacity;
       ++probe) {
    actor_pool_cursor_ =
        (actor_pool_cursor_ + 1U) % MechanicsConfig::actor_pool_capacity;
    if (!occupied[actor_pool_cursor_]) return actor_pool_cursor_;
  }
  actor_pool_cursor_ = starting_cursor;
  return std::numeric_limits<std::uint32_t>::max();
}

BodyId Simulator::spawn_piece(Shape shape, std::int32_t color, double size, Vec2 position) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  if (static_cast<std::uint8_t>(shape) >
          static_cast<std::uint8_t>(Shape::Triangle) ||
      color < 0 || static_cast<std::uint32_t>(color) >= config_.maximum_colors) {
    throw std::invalid_argument("invalid public piece identity");
  }
  const auto scripted_velocity_y =
      uses_nominal_scripted_fall(config_.scripted_fall_speed)
          ? normal_level_parameters(std::min<std::uint32_t>(level_, 99U))
                .scripted_descent_per_update
          : config_.scripted_fall_speed;
  validate_public_spawn_geometry(
      config_, shape, size, position, config_.piece_density,
      config_.piece_friction, config_.piece_restitution,
      scripted_velocity_y);
  if (next_body_id_ > kMaximumBodyId) return 0;
  const auto actor_slot = allocate_actor_slot();
  if (actor_slot == std::numeric_limits<std::uint32_t>::max()) return 0;
  Body body;
  body.id = next_body_id_;
  body.actor_slot = actor_slot;
  body.kind = BodyKind::Piece;
  body.shape = shape;
  body.lifecycle = Lifecycle::ScriptedFalling;
  body.color = color;
  body.position = position;
  body.size = size;
  body.density = config_.piece_density;
  body.friction = config_.piece_friction;
  body.restitution = config_.piece_restitution;
  body.remaining_lifetime = static_cast<std::int64_t>(config_.piece_life_ticks);
  body.scripted_velocity = {0.0, stored_float(scripted_velocity_y)};
  body.physics_owned = false;
  body.freshness_state = 1;
  actor_pool_colors_[actor_slot] = body.color;
  physics_.initialize_mass(body);
  numeric_guard_required_ =
      numeric_guard_required_ || body_requires_numeric_guard(config_, body);
  bodies_.push_back(body);
  physics_.synchronize(bodies_);
  ++next_body_id_;
  return body.id;
}

BodyId Simulator::spawn_bonus(Vec2 position) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  const auto scripted_velocity_y =
      uses_nominal_scripted_fall(config_.scripted_fall_speed)
          ? normal_level_parameters(std::min<std::uint32_t>(level_, 99U))
                .scripted_descent_per_update
          : config_.scripted_fall_speed;
  validate_public_spawn_geometry(
      config_, Shape::Circle, config_.bonus_size, position,
      config_.bonus_density, config_.bonus_friction,
      config_.bonus_restitution, scripted_velocity_y);
  if (next_body_id_ > kMaximumBodyId) return 0;
  const auto actor_slot = allocate_actor_slot();
  if (actor_slot == std::numeric_limits<std::uint32_t>::max()) return 0;
  Body body;
  body.id = next_body_id_;
  body.actor_slot = actor_slot;
  body.kind = BodyKind::Bonus;
  body.shape = Shape::Circle;
  body.lifecycle = Lifecycle::ScriptedFalling;
  body.special = true;
  body.color = -2;
  body.position = position;
  body.size = config_.bonus_size;
  body.density = config_.bonus_density;
  body.friction = config_.bonus_friction;
  body.restitution = config_.bonus_restitution;
  body.remaining_lifetime = static_cast<std::int64_t>(config_.piece_life_ticks);
  body.scripted_velocity = {0.0, stored_float(scripted_velocity_y)};
  body.physics_owned = false;
  body.freshness_state = 1;
  actor_pool_colors_[actor_slot] = body.color;
  physics_.initialize_mass(body);
  numeric_guard_required_ =
      numeric_guard_required_ || body_requires_numeric_guard(config_, body);
  bodies_.push_back(body);
  physics_.synchronize(bodies_);
  ++next_body_id_;
  return body.id;
}

void Simulator::fire(ShotStrength strength, double x, double y, std::vector<Event>& events) {
  if (next_body_id_ > kMaximumBodyId) return;
  const auto actor_slot = allocate_actor_slot();
  if (actor_slot == std::numeric_limits<std::uint32_t>::max()) return;
  Body body;
  body.id = next_body_id_;
  body.actor_slot = actor_slot;
  body.kind = BodyKind::Projectile;
  body.shape = Shape::Box;
  body.lifecycle = Lifecycle::DynamicFresh;
  body.position = {x, y};
  const auto launch_velocity = static_cast<double>(static_cast<float>(
      static_cast<float>(strength == ShotStrength::Weak
                             ? config_.weak_projectile_vy
                             : config_.strong_projectile_vy) /
      static_cast<float>(config_.world_magnification)));
  body.velocity = {0.0, launch_velocity};
  body.size = config_.projectile_size;
  body.density = config_.projectile_density;
  body.friction = config_.projectile_friction;
  body.restitution = config_.projectile_restitution;
  body.remaining_lifetime = static_cast<std::int64_t>(config_.projectile_life_ticks);
  body.physics_owned = true;
  body.top_contact_enabled = false;
  body.freshness_state = 1;
  actor_pool_colors_[actor_slot] = body.color;
  physics_.initialize_mass(body);
  numeric_guard_required_ =
      numeric_guard_required_ || body_requires_numeric_guard(config_, body);
  bodies_.push_back(body);
  physics_.synchronize(bodies_);
  ++next_body_id_;
  events.push_back({tick_, EventKind::ShotFired, body.id, 0,
                    strength == ShotStrength::Weak ? 0 : 1, "legal cursor shot"});
}

void Simulator::tick_once(bool left_level, bool right_level, double cursor_x,
                          double cursor_y, bool suppress_fresh_edges,
                          std::vector<Event>& events) {
  ++tick_;
  if (gauge_ <= 0) {
    finish_game();
    events.push_back({tick_, EventKind::GameOver, 0, 0, score_,
                      "gauge nonpositive at scene entry"});
  }
  const bool left_edge =
      left_level && !previous_left_level_ && !suppress_fresh_edges;
  const bool right_edge =
      right_level && !previous_right_level_ && !suppress_fresh_edges;
  if (left_level && !left_edge) {
    events.push_back({tick_, EventKind::HeldInputIgnored, 0, 0, 0,
                      "held left button does not repeat"});
  }
  if (right_level && !right_edge) {
    events.push_back({tick_, EventKind::HeldInputIgnored, 0, 0, 1,
                      "held right button does not repeat"});
  }
  if (left_edge) fire(ShotStrength::Weak, cursor_x, cursor_y, events);
  if (right_edge) fire(ShotStrength::Strong, cursor_x, cursor_y, events);
  previous_left_level_ = left_level;
  previous_right_level_ = right_level;
  maybe_spawn(level_, events);

  std::vector<std::tuple<BodyId, Vec2, double>> scripted_origins;
  for (const auto& body : bodies_) {
    if (live(body) && !body.physics_owned) {
      scripted_origins.emplace_back(body.id, body.position, body.angle);
    }
  }
  const auto contacts = physics_.step(bodies_);
  process_contacts(contacts, left_edge, right_edge, cursor_x, cursor_y,
                   events);
  process_scene_gauge(events);
  process_actor_updates(scripted_origins, events);
  compact_deleted();
  physics_.synchronize(bodies_);
}

BodyId Simulator::spawn_random_piece(double y,
                                     std::uint32_t parameter_level) {
  if (next_body_id_ > kMaximumBodyId) return 0;
  std::uint64_t wide_total_weight = 0;
  for (const auto weight : config_.piece_size_weights) {
    wide_total_weight += weight;
  }
  const auto total_weight = static_cast<std::uint32_t>(wide_total_weight);
  auto choice = rng_.get_rand(total_weight - 1U);
  std::size_t size_slot = 0;
  while (choice >= config_.piece_size_weights[size_slot] &&
         size_slot + 1 < config_.piece_size_weights.size()) {
    choice -= config_.piece_size_weights[size_slot++];
  }
  const double size = config_.piece_sizes[size_slot];
  const auto x_max = static_cast<std::uint32_t>(
      std::trunc(config_.field_width - config_.field_thickness));
  const Vec2 position{
      config_.field_x + static_cast<double>(rng_.get_rand(x_max)),
      y};
  const auto actor_slot = allocate_actor_slot();
  if (actor_slot == std::numeric_limits<std::uint32_t>::max()) return 0;
  const double angle = normal_spawn_angle(
      rng_.get_rand(config_.rotation_random_max));
  const auto color = static_cast<std::int32_t>(
      rng_.get_rand(color_count_for_level(parameter_level) - 1U));
  const bool special_spawn =
      qualifying_clear_count_ >= next_special_clear_count_;
  Shape shape = Shape::Circle;
  if (!special_spawn) {
    const auto shape_roll = rng_.get_rand(config_.shape_random_max);
    shape = shape_roll > level_shape_cutoff_ ? Shape::Triangle : Shape::Box;
  }
  Body body;
  body.id = next_body_id_;
  body.actor_slot = actor_slot;
  body.kind = special_spawn ? BodyKind::Bonus : BodyKind::Piece;
  body.shape = shape;
  body.lifecycle = Lifecycle::ScriptedFalling;
  body.color = special_spawn ? -2 : color;
  body.position = position;
  body.angle = angle;
  body.size = special_spawn ? config_.bonus_size : size;
  body.density = special_spawn ? config_.bonus_density : config_.piece_density;
  body.friction = special_spawn ? config_.bonus_friction : config_.piece_friction;
  body.restitution = special_spawn ? config_.bonus_restitution
                                   : config_.piece_restitution;
  body.size_slot = static_cast<std::uint32_t>(size_slot);
  body.remaining_lifetime = static_cast<std::int64_t>(config_.piece_life_ticks);
  body.scripted_velocity = {
      0.0, stored_float(uses_nominal_scripted_fall(config_.scripted_fall_speed)
                            ? normal_level_parameters(
                                  std::min<std::uint32_t>(parameter_level, 99U))
                                  .scripted_descent_per_update
                            : config_.scripted_fall_speed)};
  body.special = special_spawn;
  body.freshness_state = 1;
  actor_pool_colors_[actor_slot] = body.color;
  physics_.initialize_mass(body);
  numeric_guard_required_ =
      numeric_guard_required_ || body_requires_numeric_guard(config_, body);
  bodies_.push_back(body);
  physics_.synchronize(bodies_);
  ++next_body_id_;
  return body.id;
}

void Simulator::maybe_spawn(std::uint32_t parameter_level,
                            std::vector<Event>& events) {
  const auto interval = spawn_interval_for_level(parameter_level);
  if (interval == 0 || scene_frame_ % interval != 0) return;
  if (spawn_count_ == std::numeric_limits<std::uint64_t>::max()) return;
  const BodyId id = spawn_random_piece(config_.spawn_y, parameter_level);
  if (id == 0) return;
  const Body* body = find_body(id);
  if (body == nullptr) throw std::logic_error("spawned body is missing");
  const bool special_spawn = body->special;
  if (special_spawn) {
    const auto offset = static_cast<std::uint64_t>(config_.special_clear_base) +
                        rng_.get_rand(config_.special_clear_random_max);
    next_special_clear_count_ = checked_u64_add(
        qualifying_clear_count_, offset, "special-clear scheduler overflow");
  }
  ++spawn_count_;
  events.push_back({tick_, EventKind::Spawned, id, 0, static_cast<std::int64_t>(spawn_count_),
                    "v2.03 normal spawn"});
}

void Simulator::process_contacts(const std::vector<Contact>& contacts,
                                 bool left_edge, bool right_edge,
                                 double cursor_x, double cursor_y,
                                 std::vector<Event>& events) {
  std::vector<std::uint64_t> current;
  current.reserve(contacts.size());
  for (const auto& contact : contacts) current.push_back(contact_key(contact));
  std::sort(current.begin(), current.end());
  current.erase(std::unique(current.begin(), current.end()), current.end());

  struct Participant {
    Body* body{};
    BoundaryKind boundary{BoundaryKind::None};

    int contact_class() const {
      if (body != nullptr) return body->kind == BodyKind::Projectile ? 5 : 4;
      switch (boundary) {
        case BoundaryKind::Floor: return 2;
        case BoundaryKind::LeftWall:
        case BoundaryKind::RightWall: return 3;
        case BoundaryKind::Top: return 6;
        case BoundaryKind::None: return 0;
      }
      return 0;
    }
    bool physics_owned() const { return body == nullptr || body->physics_owned; }
    bool grouped() const { return body != nullptr && body->grouped; }
    bool rotten() const {
      return body != nullptr && body->freshness_state == 3;
    }
    std::uint8_t guard() const {
      return body == nullptr ? 0 : body->rule_guard_f0;
    }
  };

  const auto participant = [&](BodyId id, BoundaryKind boundary) {
    return Participant{id == 0 ? nullptr : find_body(id),
                       id == 0 ? boundary : BoundaryKind::None};
  };
  const auto same_color = [](const Participant& a, const Participant& b) {
    return a.body != nullptr && b.body != nullptr &&
           a.body->color == b.body->color;
  };
  const auto raw_gauge = [&](std::int64_t delta, const char* detail) {
    if (delta == 0) return;
    const auto before = gauge_;
    if (delta > 0 &&
        gauge_ > std::numeric_limits<std::int64_t>::max() - delta) {
      gauge_ = std::numeric_limits<std::int64_t>::max();
    } else if (delta < 0 &&
               gauge_ < std::numeric_limits<std::int64_t>::min() - delta) {
      gauge_ = std::numeric_limits<std::int64_t>::min();
    } else {
      gauge_ += delta;
    }
    events.push_back({tick_, EventKind::GaugeChanged, 0, 0,
                      saturated_difference(gauge_, before), detail});
  };
  const auto mark_immediate = [&](Body& body, const char* detail) {
    if (body.pending_delete) return;
    mark_deleted(body, true);
    physics_.queue_destroy(body.id);
    events.push_back({tick_, EventKind::Destroyed, body.id, 0, 0, detail});
  };
  const auto scripted_pair = [&](Participant& first, Participant& second) {
    if (first.body == nullptr || second.body == nullptr ||
        first.body->physics_owned || second.body->physics_owned ||
        first.body->rule_guard_f0 != second.body->rule_guard_f0) {
      return false;
    }
    for (Body* body : {first.body, second.body}) {
      if (body->kind == BodyKind::Piece && !body->special &&
          body->age_ticks <= 2) {
        mark_immediate(*body, "overlapping newborn scripted block");
      }
    }
    return true;
  };
  const auto top_gate = [&](Participant& target, const Participant& source) {
    if (target.body == nullptr || target.contact_class() != 4 ||
        target.body->physics_owned || source.contact_class() != 6) {
      return false;
    }
    target.body->top_contact_pending = true;
    return true;
  };
  const auto add_to_group = [&](Participant& target,
                                Participant& source) {
    if (target.body == nullptr || target.contact_class() != 4 ||
        target.body->grouped) {
      return;
    }
    GroupState* group = nullptr;
    if (source.body != nullptr && source.body->grouped) {
      group = find_group(source.body->chain_id);
      if (group == nullptr ||
          group->chain == std::numeric_limits<std::uint32_t>::max()) {
        return;
      }
    } else {
      if (next_chain_id_ == std::numeric_limits<ChainId>::max() ||
          groups_.size() >= kMaximumSnapshotGroups) {
        return;
      }
      GroupState created;
      created.id = next_chain_id_;
      groups_.push_back(created);
      ++next_chain_id_;
      group = &groups_.back();
    }
    ++group->chain;
    ++group->secondary_count;
    target.body->grouped = true;
    target.body->chain_id = group->id;
    refresh_lifecycle(*target.body);
    events.push_back({tick_, EventKind::ChainJoined, target.body->id,
                      source.body == nullptr ? 0 : source.body->id,
                      group->id, "normal group membership"});
  };
  const auto group_pair = [&](Participant& a, Participant& b) {
    if (!same_color(a, b) || (a.contact_class() != 4 && b.contact_class() != 4) ||
        (a.grouped() && b.grouped())) {
      return;
    }
    add_to_group(a, b);
    add_to_group(b, a);
  };
  const auto activate = [&](Participant& target, Participant& source) {
    if (target.body == nullptr || target.body->physics_owned ||
        !source.physics_owned() || source.contact_class() == 3) {
      return;
    }
    if (source.grouped() && !same_color(target, source) &&
        source.contact_class() != 2) {
      return;
    }
    target.body->physics_owned = true;
    refresh_lifecycle(*target.body);
    events.push_back({tick_, EventKind::Activated, target.body->id,
                      source.body == nullptr ? 0 : source.body->id, 0,
                      "normal contact activation"});
    if (source.contact_class() == 5 && source.body != nullptr) {
      mark_deleted(*source.body, false);
    }
  };
  const auto special = [&](Participant& orb, Participant& other) {
    if (orb.body == nullptr || !orb.body->special ||
        !orb.body->physics_owned || !orb.body->top_contact_enabled) {
      return;
    }
    if (other.contact_class() == 5 && other.body != nullptr) {
      mark_deleted(*other.body, false);
      events.push_back({tick_, EventKind::Destroyed, other.body->id,
                        orb.body->id, 0, "projectile armed special orb"});
      return;
    }
    if (other.body == nullptr || other.contact_class() != 4 ||
        other.body->special || other.body->grouped) {
      return;
    }
    if (!other.rotten()) {
      std::int64_t cleared = 0;
      for (auto& body : bodies_) {
        if (body.color != other.body->color ||
            body.lifecycle == Lifecycle::Deleted) {
          continue;
        }
        mark_deleted(body, false);
        ++cleared;
      }
      // clear_color (0x4032d0) walks every slot in the fixed actor pool.
      // Inactive Blocks retain their previous color and earn the same gauge
      // as live matches, even though only live actors can be torn down.
      for (const auto color : actor_pool_colors_) {
        if (color == other.body->color) {
          raw_gauge(config_.gauge_clear_unit, "special color clear");
        }
      }
      events.push_back({tick_, EventKind::Cleared, orb.body->id,
                        other.body->id, cleared, "special color clear"});
    }
    mark_deleted(*orb.body, false);
  };
  const auto burst = [&](Participant& target, Participant& source) {
    if (target.body == nullptr || target.contact_class() != 4 ||
        !target.body->grouped || target.body->successful_clear_pending ||
        (source.contact_class() != 2 &&
         (source.body == nullptr || source.body->rot_timer == 0))) {
      return;
    }
    GroupState* group = find_group(target.body->chain_id);
    if (group == nullptr) throw std::logic_error("grouped block has no group");
    target.body->successful_clear_pending = true;
    ++group->num;
    if (!target.rotten()) {
      raw_gauge(saturated_nonnegative_product(
                    static_cast<std::int64_t>(group->num),
                    config_.gauge_clear_unit),
                "normal burst landing");
    }
    ++qualifying_clear_count_;
    events.push_back({tick_, EventKind::Confirmed, target.body->id,
                      source.body == nullptr ? 0 : source.body->id,
                      group->num, "normal burst qualified"});
    update_level(left_edge, right_edge, cursor_x, cursor_y, events);
    if (source.body != nullptr && source.contact_class() == 4 &&
        same_color(target, source) && source.body->grouped &&
        !source.body->successful_clear_pending) {
      source.body->successful_clear_pending = true;
    }
  };
  const auto start_rot = [&](Participant& target, Participant& source) {
    if (target.body == nullptr ||
        (target.contact_class() != 4 && target.contact_class() != 5) ||
        target.body->special || target.body->rot_timer != 0 ||
        target.body->age_ticks <= 100 ||
        (source.contact_class() != 2 &&
         (source.body == nullptr || source.body->rot_timer == 0))) {
      return;
    }
    target.body->rot_timer = 1;
  };
  const auto direct = [&](Participant& block, Participant& projectile) {
    if (block.body == nullptr || projectile.body == nullptr ||
        block.contact_class() != 4 || projectile.contact_class() != 5) {
      return;
    }
    events.push_back({tick_, EventKind::ProjectileHit, projectile.body->id,
                      block.body->id,
                      static_cast<std::int64_t>(block.body->projectile_hits),
                      "normal direct hit"});
    if (projectile.body->non_wall_contacts == 1 && block.body->grouped) {
      if (block.body->projectile_hits < 2) ++block.body->projectile_hits;
      if (block.body->projectile_hits >= 2) {
        mark_deleted(*block.body, false);
        mark_deleted(*projectile.body, false);
      }
    }
    if (!block.rotten()) {
      mark_deleted(*projectile.body, false);
    }
  };

  for (const auto& contact : contacts) {
    const auto key = contact_key(contact);
    if (!std::binary_search(active_contact_pairs_.begin(),
                            active_contact_pairs_.end(), key)) {
      events.push_back({tick_, EventKind::Contact, contact.a, contact.b, 0,
                        "begin_contact"});
    }
    Participant a = participant(contact.a, contact.boundary);
    Participant b = participant(contact.b, contact.boundary);
    if ((contact.a != 0 && a.body == nullptr) ||
        (contact.b != 0 && b.body == nullptr)) {
      continue;
    }
    if (a.contact_class() == 5 && b.contact_class() == 5) {
      events.push_back({tick_, EventKind::ProjectileContact,
                        a.body == nullptr ? 0 : a.body->id,
                        b.body == nullptr ? 0 : b.body->id, 0,
                        "projectile-projectile contact ignored"});
      continue;
    }
    if (a.rotten() && b.rotten()) {
      continue;
    }
    if (a.body != nullptr && b.contact_class() != 3 &&
        a.body->non_wall_contacts != std::numeric_limits<std::uint32_t>::max()) {
      ++a.body->non_wall_contacts;
    }
    if (b.body != nullptr && a.contact_class() != 3 &&
        b.body->non_wall_contacts != std::numeric_limits<std::uint32_t>::max()) {
      ++b.body->non_wall_contacts;
    }
    if (scripted_pair(a, b) || scripted_pair(b, a)) continue;
    if (top_gate(b, a) || top_gate(a, b)) continue;
    group_pair(a, b);
    activate(a, b);
    activate(b, a);
    special(a, b);
    special(b, a);
    burst(a, b);
    burst(b, a);
    start_rot(a, b);
    start_rot(b, a);
    direct(b, a);
    direct(a, b);
  }
  active_contact_pairs_ = std::move(current);
}

void Simulator::process_actor_updates(
    const std::vector<std::tuple<BodyId, Vec2, double>>& scripted_origins,
    std::vector<Event>& events) {
  std::vector<Body*> actors;
  actors.reserve(bodies_.size());
  for (auto& body : bodies_) {
    if (body.lifecycle != Lifecycle::Deleted) actors.push_back(&body);
  }
  std::sort(actors.begin(), actors.end(), [](const Body* left, const Body* right) {
    return left->actor_slot < right->actor_slot;
  });

  const auto parameter_level = std::min<std::uint32_t>(level_, 99U);
  const auto parameters = normal_level_parameters(parameter_level);
  const auto rot_delay = config_.rot_delay_ticks == 120
                             ? std::uint64_t{120}
                             : config_.rot_delay_ticks;
  const auto rot_penalty = config_.rotten_penalty == 1'800
                               ? parameters.rot_penalty
                               : config_.rotten_penalty;

  for (Body* actor : actors) {
    Body& body = *actor;
    if (!body.physics_owned) {
      const auto found = std::find_if(
          scripted_origins.begin(), scripted_origins.end(),
          [&](const auto& origin) { return std::get<0>(origin) == body.id; });
      if (found != scripted_origins.end() || body.age_ticks == 0) {
        const Vec2 origin_position = found == scripted_origins.end()
                                         ? body.position
                                         : std::get<1>(*found);
        const double origin_angle = found == scripted_origins.end()
                                        ? body.angle
                                        : std::get<2>(*found);
        body.scripted_velocity = {
            stored_float(body.scripted_velocity.x),
            stored_float(body.scripted_velocity.y)};
        body.position = {
            stored_float_add(origin_position.x, body.scripted_velocity.x),
            stored_float_add(origin_position.y, body.scripted_velocity.y)};
        body.angular_velocity = stored_float(body.angular_velocity);
        body.angle = stored_float_add(origin_angle, body.angular_velocity);
        // The actor retains its scripted display-unit velocity while SetOrigin
        // independently zeros the hidden native linear velocity.
        body.velocity = body.scripted_velocity;
      }
    }

    if (body.remaining_lifetime != std::numeric_limits<std::int64_t>::min()) {
      --body.remaining_lifetime;
    }
    ++body.age_ticks;
    if (body.remaining_lifetime == 0) {
      body.delete_marked = true;
      events.push_back({tick_, EventKind::Destroyed, body.id, 0, 0,
                        "actor lifetime"});
    }

    const bool out_of_bounds =
        body.physics_owned && body.top_contact_enabled &&
        (body.kind == BodyKind::Piece || body.kind == BodyKind::Projectile ||
         body.kind == BodyKind::Bonus) &&
        (body.position.x < config_.out_of_bounds_min_x ||
         body.position.x > config_.out_of_bounds_max_x ||
         body.position.y < config_.out_of_bounds_min_y ||
         body.position.y > config_.out_of_bounds_max_y);
    if (out_of_bounds) {
      // Actor velocity is cleared, but the native body is not changed before
      // the following physics step. native_velocity is intentionally retained.
      body.velocity = {};
      body.remaining_lifetime = 1;
      events.push_back({tick_, EventKind::Ejected, body.id, 0, 0,
                        "normal strict out-of-bounds guard"});
    }

    const bool delete68 = body.delete_marked;
    if (body.freshness_state != 1) {
      if (body.successful_clear_pending &&
          body.kind != BodyKind::Projectile) {
        GroupState* group = find_group(body.chain_id);
        if (group == nullptr) {
          throw std::logic_error("burst block has no group");
        }
        std::int64_t points;
        try {
          points = normal_score_delta(parameter_level, group->num,
                                      group->chain, body.size_slot);
        } catch (const std::overflow_error&) {
          points = std::numeric_limits<std::int64_t>::max();
        }
        add_score(points, "normal burst block", events);
        mark_deleted(body, true);
        highest_chain_ = std::max(highest_chain_, group->chain);
        events.push_back({tick_, EventKind::Cleared, body.id, 0, group->num,
                          "normal burst actor teardown"});
      }

      if (delete68) {
        events.push_back({tick_, EventKind::Destroyed, body.id, 0, 0,
                          "actor deletion marker"});
        mark_deleted(body, true);
        continue;
      }

      if (!body.top_contact_pending && !body.top_contact_enabled) {
        body.top_contact_enabled = true;
      }
      body.top_contact_pending = false;
      if (body.grouped && body.top_contact_enabled) {
        body.physics_owned = true;
      }
      refresh_lifecycle(body);
    } else {
      body.freshness_state = 2;
      refresh_lifecycle(body);
    }

    if (body.rule_guard_f0 == 0 && body.freshness_state != 3 &&
        body.rot_timer != 0) {
      ++body.rot_timer;
      if (body.rot_timer > rot_delay && body.kind != BodyKind::Projectile) {
        body.freshness_state = 3;
        const auto gauge_before = gauge_;
        gauge_ = gauge_ < std::numeric_limits<std::int64_t>::min() +
                                rot_penalty
                     ? std::numeric_limits<std::int64_t>::min()
                     : gauge_ - rot_penalty;
        events.push_back({tick_, EventKind::Rotten, body.id, 0, 0,
                          "normal rot timer"});
        events.push_back({tick_, EventKind::GaugeChanged, body.id, 0,
                          saturated_difference(gauge_, gauge_before),
                          "normal rot penalty"});
      }
    }

    if (body.physics_owned) ++body.physics_update_count;
    refresh_lifecycle(body);
    if (live(body)) {
      numeric_guard_required_ =
          numeric_guard_required_ || body_requires_numeric_guard(config_, body);
    }
  }
}

void Simulator::process_scene_gauge(std::vector<Event>& events) {
  const auto before = gauge_;
  gauge_ = std::clamp<std::int64_t>(gauge_, 0, config_.gauge_max);
  ++scene_frame_;
  const auto parameter_level = std::min<std::uint32_t>(level_, 99U);
  const auto unit = checked_nonnegative_product(
      normal_level_parameters(parameter_level).passive_drain_unit,
      config_.passive_gauge_decay_per_tick,
      "passive gauge unit overflow");
  const auto drain = gauge_ > config_.gauge_max / 2
                         ? checked_nonnegative_product(
                               3, unit, "passive gauge drain overflow")
                         : unit;
  gauge_ = checked_subtract(gauge_, drain, "passive gauge subtraction overflow");
  if (gauge_ <= 0) gauge_ = 1;
  if (gauge_ != before) {
    events.push_back({tick_, EventKind::GaugeChanged, 0, 0,
                      saturated_difference(gauge_, before),
                      "scene clamp and passive drain"});
  }
}

void Simulator::finish_game() {
  if (finish_call_count_ == std::numeric_limits<std::uint64_t>::max()) {
    throw std::overflow_error("finish call counter overflow");
  }
  ++finish_call_count_;
  latest_final_score_ = score_;
  latest_final_highest_chain_ = highest_chain_;
  latest_final_level_ = level_;
  latest_final_clears_ = qualifying_clear_count_;
  if (!terminal_metadata_recorded_) {
    terminal_metadata_recorded_ = true;
    recorded_final_score_ = score_;
    recorded_final_highest_chain_ = highest_chain_;
    recorded_final_level_ = level_;
    recorded_final_clears_ = qualifying_clear_count_;
  }
  terminated_ = true;
}

void Simulator::update_level(bool left_edge, bool right_edge, double cursor_x,
                             double cursor_y, std::vector<Event>& events) {
  const auto divisor = std::max<std::uint32_t>(1, config_.qualifying_clears_per_level);
  const auto quotient = qualifying_clear_count_ / divisor;
  const auto requested = quotient == std::numeric_limits<std::uint64_t>::max()
                             ? quotient
                             : quotient + 1U;
  if (requested <= level_) return;
  if (requested >= config_.maximum_level) {
    const bool level_changed = level_ != config_.maximum_level;
    level_ = config_.maximum_level;
    if (level_changed) {
      events.push_back({tick_, EventKind::LevelChanged, 0, 0, level_,
                        "normal level cap"});
    }
    finish_game();
    events.push_back({tick_, EventKind::LevelCompleted, 0, 0, score_,
                      "normal level 100 completion"});
    return;
  }
  const auto previous_parameter_level = level_;
  level_ = static_cast<std::uint32_t>(requested);
  events.push_back({tick_, EventKind::LevelChanged, 0, 0, level_,
                    "qualifying normal clears"});

  // The executable calls Field.update from the ordinary level setter before
  // installing the new level parameters. Input edges can therefore fire a
  // second time, and cadence uses the previous level's interval and spawn
  // parameters while the public current level is already committed.
  if (left_edge) fire(ShotStrength::Weak, cursor_x, cursor_y, events);
  if (right_edge) fire(ShotStrength::Strong, cursor_x, cursor_y, events);
  maybe_spawn(previous_parameter_level, events);

  level_shape_cutoff_ = rng_.get_rand(config_.shape_random_max);
}

void Simulator::refresh_lifecycle(Body& body) {
  if (body.pending_delete) {
    body.lifecycle = Lifecycle::Deleted;
  } else if (body.freshness_state == 3) {
    body.lifecycle = Lifecycle::Rotten;
  } else if (body.grouped) {
    body.lifecycle = Lifecycle::Confirmed;
  } else if (body.physics_owned) {
    body.lifecycle = Lifecycle::DynamicFresh;
  } else {
    body.lifecycle = Lifecycle::ScriptedFalling;
  }
}

void Simulator::sequence_events(std::vector<Event>& events) {
  if (events.size() >
      std::numeric_limits<std::uint64_t>::max() - next_event_sequence_) {
    throw std::overflow_error("event sequence counter exhausted");
  }
  for (auto& event : events) event.sequence = next_event_sequence_++;
}

void Simulator::mark_deleted(Body& body, bool immediate) {
  if (!immediate) {
    if (!body.pending_delete) body.delete_marked = true;
    return;
  }
  if (body.pending_delete) return;
  body.delete_marked = false;
  body.pending_delete = true;
  body.lifecycle = Lifecycle::Deleted;
}

void Simulator::add_score(std::int64_t delta, const char* detail, std::vector<Event>& events) {
  if (delta <= 0) return;
  const auto applied = std::min(
      delta, std::numeric_limits<std::int64_t>::max() - score_);
  if (applied == 0) return;
  score_ += applied;
  events.push_back({tick_, EventKind::ScoreChanged, 0, 0, applied, detail});
}

Body* Simulator::find_body(BodyId id) {
  const auto found = std::find_if(bodies_.begin(), bodies_.end(),
                                  [id](const Body& body) { return body.id == id; });
  return found == bodies_.end() ? nullptr : &*found;
}

GroupState* Simulator::find_group(ChainId id) {
  const auto found = std::find_if(groups_.begin(), groups_.end(),
                                  [id](const GroupState& group) {
                                    return group.id == id;
                                  });
  return found == groups_.end() ? nullptr : &*found;
}

void Simulator::compact_deleted() {
  bodies_.erase(std::remove_if(bodies_.begin(), bodies_.end(), [](const Body& body) {
    return body.lifecycle == Lifecycle::Deleted && !body.pending_delete;
  }), bodies_.end());
}

Snapshot Simulator::clone_state() const {
  const ScopedFloatingPointEnvironment floating_point_environment;
  Snapshot snapshot;
  snapshot.schema_version = kSnapshotVersion;
  snapshot.config_hash = config_hash();
  snapshot.tick = tick_;
  snapshot.scene_frame = scene_frame_;
  snapshot.rng_state = rng_.state();
  snapshot.rng_index = rng_.index();
  snapshot.next_body_id = next_body_id_;
  snapshot.next_chain_id = next_chain_id_;
  snapshot.actor_pool_cursor = actor_pool_cursor_;
  snapshot.actor_pool_colors = actor_pool_colors_;
  snapshot.next_event_sequence = next_event_sequence_;
  snapshot.spawn_count = spawn_count_;
  snapshot.score = score_;
  snapshot.gauge = gauge_;
  snapshot.level = level_;
  snapshot.qualifying_clear_count = qualifying_clear_count_;
  snapshot.next_special_clear_count = next_special_clear_count_;
  snapshot.level_shape_cutoff = level_shape_cutoff_;
  snapshot.highest_chain = highest_chain_;
  snapshot.finish_call_count = finish_call_count_;
  snapshot.terminal_metadata_recorded = terminal_metadata_recorded_;
  snapshot.recorded_final_score = recorded_final_score_;
  snapshot.recorded_final_highest_chain = recorded_final_highest_chain_;
  snapshot.recorded_final_level = recorded_final_level_;
  snapshot.recorded_final_clears = recorded_final_clears_;
  snapshot.latest_final_score = latest_final_score_;
  snapshot.latest_final_highest_chain = latest_final_highest_chain_;
  snapshot.latest_final_level = latest_final_level_;
  snapshot.latest_final_clears = latest_final_clears_;
  snapshot.previous_left_level = previous_left_level_;
  snapshot.previous_right_level = previous_right_level_;
  snapshot.terminated = terminated_;
  snapshot.truncated = truncated_;
  snapshot.bodies = bodies_;
  snapshot.groups = groups_;
  snapshot.active_contact_pairs = active_contact_pairs_;
  snapshot.contact_impulses = physics_.contact_impulses(bodies_);
  snapshot.physics_ordering = physics_.ordering();
  return snapshot;
}

void Simulator::restore_state(const Snapshot& snapshot) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  validate_snapshot_payload(config_, snapshot, config_hash());
  auto restored_bodies = snapshot.bodies;
  auto restored_groups = snapshot.groups;
  auto restored_contact_pairs = snapshot.active_contact_pairs;
  PhysicsWorld restored_physics(config_);
  restored_physics.rebuild(restored_bodies, snapshot.contact_impulses,
                           snapshot.physics_ordering);
  DxRandom restored_rng = rng_;
  restored_rng.restore(snapshot.rng_state, snapshot.rng_index);
  const auto counter_guard_threshold =
      std::numeric_limits<std::uint64_t>::max() - 100'000U;
  const bool restored_actor_counter_guard = std::any_of(
      restored_bodies.begin(), restored_bodies.end(),
      [&](const Body& body) {
        return body.age_ticks >= counter_guard_threshold ||
               body.physics_update_count >= counter_guard_threshold ||
               body.rot_timer >= counter_guard_threshold;
      });
  const bool restored_numeric_guard = std::any_of(
      restored_bodies.begin(), restored_bodies.end(), [&](const Body& body) {
        return live(body) && body_requires_numeric_guard(config_, body);
      });

  tick_ = snapshot.tick;
  scene_frame_ = snapshot.scene_frame;
  rng_ = restored_rng;
  next_body_id_ = snapshot.next_body_id;
  next_chain_id_ = snapshot.next_chain_id;
  actor_pool_cursor_ = snapshot.actor_pool_cursor;
  actor_pool_colors_ = snapshot.actor_pool_colors;
  next_event_sequence_ = snapshot.next_event_sequence;
  spawn_count_ = snapshot.spawn_count;
  score_ = snapshot.score;
  gauge_ = snapshot.gauge;
  level_ = snapshot.level;
  qualifying_clear_count_ = snapshot.qualifying_clear_count;
  next_special_clear_count_ = snapshot.next_special_clear_count;
  level_shape_cutoff_ = snapshot.level_shape_cutoff;
  highest_chain_ = snapshot.highest_chain;
  finish_call_count_ = snapshot.finish_call_count;
  terminal_metadata_recorded_ = snapshot.terminal_metadata_recorded;
  recorded_final_score_ = snapshot.recorded_final_score;
  recorded_final_highest_chain_ = snapshot.recorded_final_highest_chain;
  recorded_final_level_ = snapshot.recorded_final_level;
  recorded_final_clears_ = snapshot.recorded_final_clears;
  latest_final_score_ = snapshot.latest_final_score;
  latest_final_highest_chain_ = snapshot.latest_final_highest_chain;
  latest_final_level_ = snapshot.latest_final_level;
  latest_final_clears_ = snapshot.latest_final_clears;
  previous_left_level_ = snapshot.previous_left_level;
  previous_right_level_ = snapshot.previous_right_level;
  terminated_ = snapshot.terminated;
  truncated_ = snapshot.truncated;
  actor_counter_guard_required_ = restored_actor_counter_guard;
  numeric_guard_required_ = restored_numeric_guard;
  bodies_ = std::move(restored_bodies);
  groups_ = std::move(restored_groups);
  active_contact_pairs_ = std::move(restored_contact_pairs);
  physics_ = std::move(restored_physics);
}

std::vector<std::byte> Simulator::serialize_snapshot() const {
  const ScopedFloatingPointEnvironment floating_point_environment;
  const auto snapshot = clone_state();
  if (snapshot.bodies.size() > std::numeric_limits<std::uint32_t>::max() ||
      snapshot.groups.size() > std::numeric_limits<std::uint32_t>::max() ||
      snapshot.active_contact_pairs.size() > std::numeric_limits<std::uint32_t>::max() ||
      snapshot.contact_impulses.size() > std::numeric_limits<std::uint32_t>::max() ||
      snapshot.physics_ordering.body_order.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      snapshot.physics_ordering.destroy_order.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      snapshot.physics_ordering.proxy_order.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      snapshot.physics_ordering.proxy_ids.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      snapshot.physics_ordering.free_proxy_order.size() >
          std::numeric_limits<std::uint32_t>::max() ||
      snapshot.physics_ordering.broadphase_bounds.size() >
          std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("snapshot collection exceeds wire-format limit");
  }
  std::vector<std::byte> output;
  output.reserve(4700 + snapshot.bodies.size() * kSerializedBodyBytes +
                 snapshot.groups.size() * kSerializedGroupBytes +
                 snapshot.physics_ordering.broadphase_bounds.size() *
                     kSerializedBroadPhaseBoundBytes);
  append_integer(output, kSnapshotMagic);
  append_integer(output, snapshot.schema_version);
  append_integer(output, snapshot.config_hash);
  append_integer(output, snapshot.tick);
  append_integer(output, snapshot.scene_frame);
  for (const auto word : snapshot.rng_state) append_integer(output, word);
  append_integer(output, snapshot.rng_index);
  append_integer(output, snapshot.next_body_id);
  append_integer(output, snapshot.next_chain_id);
  append_integer(output, snapshot.actor_pool_cursor);
  append_integer(output, snapshot.next_event_sequence);
  append_integer(output, snapshot.spawn_count);
  append_integer(output, snapshot.score);
  append_integer(output, snapshot.gauge);
  append_integer(output, snapshot.level);
  append_integer(output, snapshot.qualifying_clear_count);
  append_integer(output, snapshot.next_special_clear_count);
  append_integer(output, snapshot.level_shape_cutoff);
  append_integer(output, snapshot.highest_chain);
  append_integer(output, snapshot.finish_call_count);
  append_integer(output,
                 static_cast<std::uint8_t>(snapshot.terminal_metadata_recorded));
  append_integer(output, snapshot.recorded_final_score);
  append_integer(output, snapshot.recorded_final_highest_chain);
  append_integer(output, snapshot.recorded_final_level);
  append_integer(output, snapshot.recorded_final_clears);
  append_integer(output, snapshot.latest_final_score);
  append_integer(output, snapshot.latest_final_highest_chain);
  append_integer(output, snapshot.latest_final_level);
  append_integer(output, snapshot.latest_final_clears);
  append_integer(output, static_cast<std::uint8_t>(snapshot.previous_left_level));
  append_integer(output, static_cast<std::uint8_t>(snapshot.previous_right_level));
  append_integer(output, static_cast<std::uint8_t>(snapshot.terminated));
  append_integer(output, static_cast<std::uint8_t>(snapshot.truncated));
  append_integer(output, static_cast<std::uint32_t>(snapshot.bodies.size()));
  for (const auto& body : snapshot.bodies) serialize_body(output, body);
  append_integer(output, static_cast<std::uint32_t>(snapshot.groups.size()));
  for (const auto& group : snapshot.groups) serialize_group(output, group);
  append_integer(output, static_cast<std::uint32_t>(
                             snapshot.physics_ordering.body_order.size()));
  for (const auto id : snapshot.physics_ordering.body_order) {
    append_integer(output, id);
  }
  append_integer(output, static_cast<std::uint32_t>(
                             snapshot.physics_ordering.destroy_order.size()));
  for (const auto id : snapshot.physics_ordering.destroy_order) {
    append_integer(output, id);
  }
  append_integer(output, static_cast<std::uint32_t>(
                             snapshot.physics_ordering.proxy_order.size()));
  for (const auto id : snapshot.physics_ordering.proxy_order) {
    append_integer(output, id);
  }
  append_integer(output, static_cast<std::uint32_t>(
                             snapshot.physics_ordering.proxy_ids.size()));
  for (const auto proxy : snapshot.physics_ordering.proxy_ids) {
    append_integer(output, proxy);
  }
  append_integer(output, static_cast<std::uint32_t>(
                             snapshot.physics_ordering.free_proxy_order.size()));
  for (const auto proxy : snapshot.physics_ordering.free_proxy_order) {
    append_integer(output, proxy);
  }
  append_integer(output, snapshot.physics_ordering.static_sleep_flags);
  append_integer(output, snapshot.physics_ordering.broadphase_time_stamp);
  append_integer(output, static_cast<std::uint32_t>(
                             snapshot.physics_ordering.proxy_time_stamps.size()));
  for (std::size_t index = 0;
       index < snapshot.physics_ordering.proxy_time_stamps.size(); ++index) {
    append_integer(output,
                   snapshot.physics_ordering.proxy_time_stamps[index]);
    append_integer(output,
                   snapshot.physics_ordering.proxy_overlap_counts[index]);
  }
  append_integer(output, static_cast<std::uint32_t>(
                             snapshot.physics_ordering.broadphase_bounds.size()));
  for (const auto& bound : snapshot.physics_ordering.broadphase_bounds) {
    append_integer(output, bound.value);
    append_integer(output, bound.proxy_id);
    append_integer(output, bound.stabbing_count);
  }
  append_integer(output, static_cast<std::uint32_t>(snapshot.active_contact_pairs.size()));
  for (const auto pair : snapshot.active_contact_pairs) append_integer(output, pair);
  append_integer(output, static_cast<std::uint32_t>(snapshot.contact_impulses.size()));
  for (const auto& impulse : snapshot.contact_impulses) {
    serialize_contact_impulse(output, impulse);
  }
  for (const auto color : snapshot.actor_pool_colors) {
    append_integer(output, color);
  }
  return output;
}

void Simulator::restore_snapshot(std::span<const std::byte> bytes) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  std::size_t offset = 0;
  if (read_integer<std::uint32_t>(bytes, offset) != kSnapshotMagic) throw std::invalid_argument("snapshot magic mismatch");
  Snapshot snapshot;
  snapshot.schema_version = read_integer<std::uint32_t>(bytes, offset);
  snapshot.config_hash = read_integer<std::uint64_t>(bytes, offset);
  if (snapshot.schema_version != kSnapshotVersion) {
    throw std::invalid_argument("snapshot schema mismatch");
  }
  if (snapshot.config_hash != config_hash()) {
    throw std::invalid_argument("snapshot mechanics profile mismatch");
  }
  snapshot.tick = read_integer<std::uint64_t>(bytes, offset);
  snapshot.scene_frame = read_integer<std::uint64_t>(bytes, offset);
  for (auto& word : snapshot.rng_state) {
    word = read_integer<std::uint32_t>(bytes, offset);
  }
  snapshot.rng_index = read_integer<std::uint32_t>(bytes, offset);
  snapshot.next_body_id = read_integer<BodyId>(bytes, offset);
  snapshot.next_chain_id = read_integer<ChainId>(bytes, offset);
  snapshot.actor_pool_cursor = read_integer<std::uint32_t>(bytes, offset);
  snapshot.next_event_sequence = read_integer<std::uint64_t>(bytes, offset);
  snapshot.spawn_count = read_integer<std::uint64_t>(bytes, offset);
  snapshot.score = read_integer<std::int64_t>(bytes, offset);
  snapshot.gauge = read_integer<std::int64_t>(bytes, offset);
  snapshot.level = read_integer<std::uint32_t>(bytes, offset);
  snapshot.qualifying_clear_count = read_integer<std::uint64_t>(bytes, offset);
  snapshot.next_special_clear_count = read_integer<std::uint64_t>(bytes, offset);
  snapshot.level_shape_cutoff = read_integer<std::uint32_t>(bytes, offset);
  snapshot.highest_chain = read_integer<std::uint32_t>(bytes, offset);
  snapshot.finish_call_count = read_integer<std::uint64_t>(bytes, offset);
  snapshot.terminal_metadata_recorded = read_bool(bytes, offset);
  snapshot.recorded_final_score = read_integer<std::int64_t>(bytes, offset);
  snapshot.recorded_final_highest_chain =
      read_integer<std::uint32_t>(bytes, offset);
  snapshot.recorded_final_level =
      read_integer<std::uint32_t>(bytes, offset);
  snapshot.recorded_final_clears =
      read_integer<std::uint64_t>(bytes, offset);
  snapshot.latest_final_score = read_integer<std::int64_t>(bytes, offset);
  snapshot.latest_final_highest_chain =
      read_integer<std::uint32_t>(bytes, offset);
  snapshot.latest_final_level = read_integer<std::uint32_t>(bytes, offset);
  snapshot.latest_final_clears = read_integer<std::uint64_t>(bytes, offset);
  snapshot.previous_left_level = read_bool(bytes, offset);
  snapshot.previous_right_level = read_bool(bytes, offset);
  snapshot.terminated = read_bool(bytes, offset);
  snapshot.truncated = read_bool(bytes, offset);
  const auto body_count = read_integer<std::uint32_t>(bytes, offset);
  if (body_count > kMaximumSnapshotBodies) {
    throw std::invalid_argument("snapshot body count exceeds simulator capacity");
  }
  if (body_count > (bytes.size() - offset) / kSerializedBodyBytes) {
    throw std::invalid_argument("snapshot body collection is truncated");
  }
  snapshot.bodies.reserve(body_count);
  for (std::uint32_t index = 0; index < body_count; ++index) snapshot.bodies.push_back(deserialize_body(bytes, offset));
  const auto group_count = read_integer<std::uint32_t>(bytes, offset);
  if (group_count > kMaximumSnapshotGroups ||
      group_count >= snapshot.next_chain_id ||
      group_count > (bytes.size() - offset) / kSerializedGroupBytes) {
    throw std::invalid_argument("snapshot group collection is invalid or truncated");
  }
  snapshot.groups.reserve(group_count);
  for (std::uint32_t index = 0; index < group_count; ++index) {
    snapshot.groups.push_back(deserialize_group(bytes, offset));
  }
  const auto body_order_count = read_integer<std::uint32_t>(bytes, offset);
  if (body_order_count > body_count ||
      body_order_count > (bytes.size() - offset) / sizeof(BodyId)) {
    throw std::invalid_argument("snapshot physics body order is invalid or truncated");
  }
  snapshot.physics_ordering.body_order.reserve(body_order_count);
  for (std::uint32_t index = 0; index < body_order_count; ++index) {
    snapshot.physics_ordering.body_order.push_back(
        read_integer<BodyId>(bytes, offset));
  }
  const auto destroy_order_count = read_integer<std::uint32_t>(bytes, offset);
  if (destroy_order_count > body_count ||
      destroy_order_count > (bytes.size() - offset) / sizeof(BodyId)) {
    throw std::invalid_argument(
        "snapshot physics destroy order is invalid or truncated");
  }
  snapshot.physics_ordering.destroy_order.reserve(destroy_order_count);
  for (std::uint32_t index = 0; index < destroy_order_count; ++index) {
    snapshot.physics_ordering.destroy_order.push_back(
        read_integer<BodyId>(bytes, offset));
  }
  const auto proxy_order_count = read_integer<std::uint32_t>(bytes, offset);
  if (proxy_order_count > body_count ||
      proxy_order_count > (bytes.size() - offset) / sizeof(BodyId)) {
    throw std::invalid_argument("snapshot physics proxy order is invalid or truncated");
  }
  snapshot.physics_ordering.proxy_order.reserve(proxy_order_count);
  for (std::uint32_t index = 0; index < proxy_order_count; ++index) {
    snapshot.physics_ordering.proxy_order.push_back(
        read_integer<BodyId>(bytes, offset));
  }
  const auto proxy_id_count = read_integer<std::uint32_t>(bytes, offset);
  if (proxy_id_count > body_count ||
      proxy_id_count > (bytes.size() - offset) / sizeof(std::uint32_t)) {
    throw std::invalid_argument("snapshot physics proxy ids are invalid or truncated");
  }
  snapshot.physics_ordering.proxy_ids.reserve(proxy_id_count);
  for (std::uint32_t index = 0; index < proxy_id_count; ++index) {
    snapshot.physics_ordering.proxy_ids.push_back(
        read_integer<std::uint32_t>(bytes, offset));
  }
  const auto free_proxy_count = read_integer<std::uint32_t>(bytes, offset);
  if (free_proxy_count > MechanicsConfig::physics_proxy_capacity -
                             MechanicsConfig::static_fixture_count ||
      free_proxy_count > (bytes.size() - offset) / sizeof(std::uint32_t)) {
    throw std::invalid_argument(
        "snapshot free proxy order is invalid or truncated");
  }
  snapshot.physics_ordering.free_proxy_order.reserve(free_proxy_count);
  for (std::uint32_t index = 0; index < free_proxy_count; ++index) {
    snapshot.physics_ordering.free_proxy_order.push_back(
        read_integer<std::uint32_t>(bytes, offset));
  }
  snapshot.physics_ordering.static_sleep_flags =
      read_integer<std::uint8_t>(bytes, offset);
  snapshot.physics_ordering.broadphase_time_stamp =
      read_integer<std::uint16_t>(bytes, offset);
  const auto proxy_state_count = read_integer<std::uint32_t>(bytes, offset);
  if (proxy_state_count != MechanicsConfig::physics_proxy_capacity ||
      proxy_state_count > (bytes.size() - offset) /
                              (2U * sizeof(std::uint16_t))) {
    throw std::invalid_argument(
        "snapshot broad-phase proxy state is invalid or truncated");
  }
  snapshot.physics_ordering.proxy_time_stamps.reserve(proxy_state_count);
  snapshot.physics_ordering.proxy_overlap_counts.reserve(proxy_state_count);
  for (std::uint32_t index = 0; index < proxy_state_count; ++index) {
    snapshot.physics_ordering.proxy_time_stamps.push_back(
        read_integer<std::uint16_t>(bytes, offset));
    snapshot.physics_ordering.proxy_overlap_counts.push_back(
        read_integer<std::uint16_t>(bytes, offset));
  }
  const auto bound_count = read_integer<std::uint32_t>(bytes, offset);
  const auto maximum_bound_count =
      4U * (MechanicsConfig::static_fixture_count + body_count);
  if (bound_count > maximum_bound_count ||
      bound_count >
          (bytes.size() - offset) / kSerializedBroadPhaseBoundBytes) {
    throw std::invalid_argument(
        "snapshot broad-phase bounds are invalid or truncated");
  }
  snapshot.physics_ordering.broadphase_bounds.reserve(bound_count);
  for (std::uint32_t index = 0; index < bound_count; ++index) {
    BroadPhaseBound bound;
    bound.value = read_integer<std::uint16_t>(bytes, offset);
    bound.proxy_id = read_integer<std::uint16_t>(bytes, offset);
    bound.stabbing_count = read_integer<std::uint16_t>(bytes, offset);
    snapshot.physics_ordering.broadphase_bounds.push_back(bound);
  }
  const auto contact_count = read_integer<std::uint32_t>(bytes, offset);
  const auto maximum_contact_count =
      static_cast<std::uint64_t>(body_count) *
          (body_count - (body_count == 0 ? 0U : 1U)) / 2U +
      4U * body_count;
  if (contact_count > maximum_contact_count) {
    throw std::invalid_argument("snapshot contact count exceeds capacity");
  }
  if (contact_count > (bytes.size() - offset) / sizeof(std::uint64_t)) {
    throw std::invalid_argument("snapshot contact collection is truncated");
  }
  snapshot.active_contact_pairs.reserve(contact_count);
  for (std::uint32_t index = 0; index < contact_count; ++index) {
    snapshot.active_contact_pairs.push_back(read_integer<std::uint64_t>(bytes, offset));
  }
  const auto impulse_count = read_integer<std::uint32_t>(bytes, offset);
  if (impulse_count > 2U * 4096U) {
    throw std::invalid_argument("snapshot contact impulse count is unreasonable");
  }
  if (impulse_count >
      (bytes.size() - offset) / kSerializedContactImpulseBytes) {
    throw std::invalid_argument("snapshot contact impulse collection is truncated");
  }
  snapshot.contact_impulses.reserve(impulse_count);
  for (std::uint32_t index = 0; index < impulse_count; ++index) {
    snapshot.contact_impulses.push_back(
        deserialize_contact_impulse(bytes, offset));
  }
  for (auto& color : snapshot.actor_pool_colors) {
    color = read_integer<std::int32_t>(bytes, offset);
  }
  if (offset != bytes.size()) throw std::invalid_argument("snapshot has trailing bytes");
  restore_state(snapshot);
}

std::uint64_t Simulator::state_hash() const {
  const ScopedFloatingPointEnvironment floating_point_environment;
  return fnv1a(serialize_snapshot());
}

std::uint64_t Simulator::config_hash() const {
  return config_hash_;
}

std::uint64_t Simulator::calculate_config_hash() const {
  const ScopedFloatingPointEnvironment floating_point_environment;
  std::vector<std::byte> bytes;
  const auto add = [&](auto value) {
    using T = decltype(value);
    if constexpr (std::is_same_v<T, double>) append_double(bytes, value);
    else append_integer(bytes, value);
  };
  add(MechanicsConfig::schema_version);
  add(MechanicsConfig::target_game_version);
  add(MechanicsConfig::legacy_box2d_revision);
  add(MechanicsConfig::physics_proxy_capacity);
  add(MechanicsConfig::static_fixture_count);
  add(MechanicsConfig::actor_pool_capacity);
  add(config_.tick_seconds);
  add(config_.solver_iterations);
  add(config_.world_magnification);
  add(config_.world_min_x);
  add(config_.world_min_y);
  add(config_.world_max_x);
  add(config_.world_max_y);
  add(config_.field_x);
  add(config_.field_y);
  add(config_.field_width);
  add(config_.field_height);
  add(config_.field_blank);
  add(config_.field_thickness);
  add(config_.field_top);
  add(config_.field_top_width);
  add(config_.field_top_height);
  add(config_.field_bottom_height);
  add(config_.client_width);
  add(config_.client_height);
  add(config_.side_wall_top);
  add(config_.side_wall_bottom);
  add(config_.cleanup_margin_x);
  add(config_.cleanup_margin_y);
  add(config_.floor_contact_tolerance);
  add(config_.out_of_bounds_min_x);
  add(config_.out_of_bounds_max_x);
  add(config_.out_of_bounds_min_y);
  add(config_.out_of_bounds_max_y);
  add(config_.gravity_y);
  add(config_.linear_damping);
  add(config_.angular_damping);
  add(config_.scripted_fall_speed);
  add(config_.piece_density);
  add(config_.piece_friction);
  add(config_.piece_restitution);
  for (const auto value : config_.piece_sizes) add(value);
  for (const auto value : config_.piece_size_weights) add(value);
  add(config_.piece_life_ticks);
  add(config_.rot_delay_ticks);
  add(config_.deletion_delay_ticks);
  add(config_.projectile_size);
  add(config_.projectile_density);
  add(config_.projectile_friction);
  add(config_.projectile_restitution);
  add(config_.projectile_life_ticks);
  add(config_.weak_projectile_vy);
  add(config_.strong_projectile_vy);
  add(config_.click_cooldown_ticks);
  add(config_.bonus_size);
  add(config_.bonus_density);
  add(config_.bonus_friction);
  add(config_.bonus_restitution);
  add(config_.gauge_max);
  add(config_.gauge_initial);
  add(config_.gauge_clear_unit);
  add(config_.rotten_penalty);
  add(config_.passive_gauge_decay_per_tick);
  add(config_.spawn_interval_ticks);
  add(config_.bonus_interval_spawns);
  add(config_.starting_colors);
  add(config_.maximum_colors);
  add(config_.qualifying_clears_per_level);
  add(config_.special_clear_base);
  add(config_.special_clear_random_max);
  add(config_.initial_rotten_count);
  add(config_.initial_falling_count);
  add(config_.initial_rotten_y);
  add(config_.initial_falling_y);
  add(config_.spawn_y);
  add(config_.rotation_random_max);
  add(config_.shape_random_max);
  add(config_.color_level_stride);
  add(config_.score_per_level);
  add(config_.maximum_level);
  add(config_.spawn_acceleration_level_stride);
  for (const auto value : config_.size_score_values) add(value);
  add(config_.chain_score_exponent);
  add(config_.max_episode_ticks);
  return fnv1a(bytes);
}

}  // namespace irisu
