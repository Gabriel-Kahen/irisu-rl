#pragma once

#include "irisu/math.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace irisu {

using BodyId = std::uint32_t;
using ChainId = std::uint32_t;

enum class BodyKind : std::uint8_t { Piece, Projectile, Bonus };
enum class Shape : std::uint8_t { Circle, Box, Triangle };
enum class Lifecycle : std::uint8_t {
  ScriptedFalling,
  DynamicFresh,
  Confirmed,
  Rotten,
  Deleted,
};
enum class ShotStrength : std::uint8_t { Weak, Strong };
enum class ActionKind : std::uint8_t { Wait, WeakShot, StrongShot, BothShots };
enum class BoundaryKind : std::uint8_t { None, Floor, LeftWall, RightWall, Top };
enum class EventKind : std::uint8_t {
  InvalidAction,
  Spawned,
  ShotFired,
  Activated,
  Contact,
  Confirmed,
  ChainJoined,
  Cleared,
  Rotten,
  Ejected,
  Destroyed,
  GaugeChanged,
  ScoreChanged,
  LevelChanged,
  GameOver,
  ProjectileHit,
  ProjectileContact,
  HeldInputIgnored,
  LevelCompleted,
};

struct Action {
  ActionKind kind{ActionKind::Wait};
  double cursor_x{};
  double cursor_y{};
  std::uint32_t wait_ticks{1};
  // The original replay input path updates held history but clears fresh
  // button edges for its first two records. Ordinary live/RL actions leave
  // this false.
  bool suppress_fresh_edges{};
};

struct Body {
  BodyId id{};
  BodyKind kind{BodyKind::Piece};
  Shape shape{Shape::Triangle};
  Lifecycle lifecycle{Lifecycle::ScriptedFalling};
  std::int32_t color{-1};
  Vec2 position{};
  Vec2 velocity{};                  // actor-visible raw b2d_get_v result
  double angle{};
  double angular_velocity{};        // actor integrator state; not exposed by DLL
  Vec2 native_position{};           // snapshot-only Box2D world-unit origin
  Vec2 native_center{};             // exact r58 center-of-mass position
  Vec2 native_velocity{};           // snapshot-only raw Box2D linear velocity
  double native_angle{};            // snapshot-only Box2D rotation
  double native_angular_velocity{}; // snapshot-only Box2D angular velocity
  bool native_state_valid{};
  bool native_center_valid{};
  double size{30.0};
  double density{1.0};
  double friction{1.0};
  double restitution{};
  double inverse_mass{};
  double inverse_inertia{};
  ChainId chain_id{};
  std::uint32_t actor_slot{};
  std::uint32_t size_slot{};
  std::uint32_t projectile_hits{};
  std::uint64_t age_ticks{};
  std::int64_t remaining_lifetime{-1};
  Vec2 scripted_velocity{};          // actor-side velocity while c4 == 0
  std::uint64_t rot_timer{};
  bool physics_owned{};
  bool special{};
  std::uint8_t freshness_state{1};  // raw c8: 1=new, 2=fresh, 3=rotten
  bool grouped{};                   // raw d4; independent of c8
  bool successful_clear_pending{}; // raw d5
  std::uint32_t non_wall_contacts{}; // raw e0
  bool top_contact_pending{};       // raw e4
  bool top_contact_enabled{};       // raw e5
  std::uint64_t physics_update_count{}; // raw f8
  std::uint8_t rule_guard_f0{};     // raw f0; descriptive meaning unresolved
  bool delete_marked{};             // raw +0x68; consumed by a later actor update
  bool pending_delete{};
  bool sleeping{};          // snapshot-only physics state; omitted from policy observations
  double sleep_time{};      // seconds accumulated by legacy Box2D's sleep heuristic
};

// Public body state available to policies. Native Box2D state, allocator
// bookkeeping, contact caches, and pending rule flags remain snapshot-only.
struct ObservedBody {
  BodyId id{};
  BodyKind kind{BodyKind::Piece};
  Shape shape{Shape::Triangle};
  Lifecycle lifecycle{Lifecycle::ScriptedFalling};
  std::int32_t color{-1};
  Vec2 position{};
  Vec2 velocity{};
  double angle{};
  double angular_velocity{};
  double size{30.0};
  ChainId chain_id{};
  std::uint32_t projectile_hits{};
  std::uint64_t age_ticks{};
  std::int64_t remaining_lifetime{-1};
  std::uint64_t rot_timer{};
};

// The executable's group counters and the proven normal dispatcher predicates
// that mutate them are preserved explicitly.
struct GroupState {
  ChainId id{};
  std::uint32_t chain{};
  std::uint32_t secondary_count{};
  std::uint32_t num{};
};

struct Contact {
  BodyId a{};
  BodyId b{};
  BoundaryKind boundary{BoundaryKind::None};
  Vec2 point{};
  Vec2 normal{};  // points from a to b
  double penetration{};
};

// Accumulated r58 solver impulses are part of the state needed for exact
// rollout branching. Values are stored as raw float bits so snapshots preserve
// them without a widening/narrowing round trip.
struct ContactImpulse {
  BodyId a{};
  BodyId b{};
  BoundaryKind boundary{BoundaryKind::None};
  bool destroy_pending{};
  std::uint8_t manifold_count{};
  std::uint8_t manifold_index{};
  std::uint8_t point_count{};
  std::uint8_t point_index{};
  std::uint32_t contact_order{};
  std::uint32_t feature_id{};
  std::uint32_t normal_x_bits{};
  std::uint32_t normal_y_bits{};
  std::uint32_t point_x_bits{};
  std::uint32_t point_y_bits{};
  std::uint32_t separation_bits{};
  std::uint32_t order_a{};
  std::uint32_t order_b{};
  std::uint32_t normal_impulse_bits{};
  std::uint32_t tangent_impulse_bits{};

  friend bool operator==(const ContactImpulse&, const ContactImpulse&) = default;
};

struct BroadPhaseBound {
  std::uint16_t value{};
  std::uint16_t proxy_id{};
  std::uint16_t stabbing_count{};

  friend bool operator==(const BroadPhaseBound&, const BroadPhaseBound&) = default;
};

struct PhysicsOrdering {
  std::vector<BodyId> body_order{};   // native b2World list, head first
  std::vector<BodyId> destroy_order{};  // deferred native destroy list, head first
  std::vector<BodyId> proxy_order{};  // bodies sorted by broad-phase proxy id
  std::vector<std::uint32_t> proxy_ids{};
  std::vector<std::uint32_t> free_proxy_order{};
  std::uint8_t static_sleep_flags{};  // left, right, bottom, top
  std::uint16_t broadphase_time_stamp{};
  std::vector<std::uint16_t> proxy_time_stamps{};  // all 512 proxy slots
  std::vector<std::uint16_t> proxy_overlap_counts{};  // all 512 proxy slots
  std::vector<BroadPhaseBound> broadphase_bounds{};  // X then Y, exact live order
};

struct Event {
  std::uint64_t tick{};
  EventKind kind{EventKind::Contact};
  BodyId a{};
  BodyId b{};
  std::int64_t value{};
  std::string detail{};
  std::uint64_t sequence{};
};

struct StepDiagnostics {
  std::uint64_t config_hash{};
  std::uint64_t finish_call_count{};
  bool terminal_metadata_recorded{};
  std::int64_t recorded_final_score{};
  std::uint32_t recorded_final_highest_chain{};
  std::uint32_t recorded_final_level{};
  std::uint64_t recorded_final_clears{};
  std::int64_t latest_final_score{};
  std::uint32_t latest_final_highest_chain{};
  std::uint32_t latest_final_level{};
  std::uint64_t latest_final_clears{};
};

struct StepResult {
  std::int64_t reward{};
  bool terminated{};
  bool truncated{};
  std::vector<Event> events{};
  StepDiagnostics diagnostics{};
};

struct Observation {
  std::uint64_t tick{};
  std::int64_t score{};
  std::int64_t gauge{};
  std::uint32_t level{};
  bool terminated{};
  bool truncated{};
  std::vector<ObservedBody> bodies{};
  double field_x{};
  double field_y{};
  double field_width{};
  double field_height{};
  double side_wall_top{};
  double side_wall_bottom{};
  std::int64_t gauge_max{};
  std::uint32_t active_colors{};
  std::uint32_t current_spawn_interval_ticks{};
  bool left_held{};
  bool right_held{};
  std::uint32_t highest_chain{};
  std::uint64_t qualifying_clear_count{};
};

}  // namespace irisu
