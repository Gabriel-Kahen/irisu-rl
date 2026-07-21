#pragma once

#include "irisu/config.hpp"
#include "irisu/dx_random.hpp"
#include "irisu/physics.hpp"
#include "irisu/types.hpp"

#include <array>
#include <cstdint>
#include <span>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace irisu {

struct Snapshot {
  std::uint32_t schema_version{7};
  std::uint64_t config_hash{};
  std::uint64_t tick{};
  std::uint64_t scene_frame{};
  std::array<std::uint32_t, DxRandom::state_words> rng_state{};
  std::uint32_t rng_index{DxRandom::state_words};
  BodyId next_body_id{1};
  ChainId next_chain_id{1};
  std::uint32_t actor_pool_cursor{};
  std::array<std::int32_t, MechanicsConfig::actor_pool_capacity>
      actor_pool_colors{};
  std::uint64_t next_event_sequence{};
  std::uint64_t spawn_count{};
  std::int64_t score{};
  std::int64_t gauge{};
  std::uint32_t level{1};
  std::uint64_t qualifying_clear_count{};
  std::uint64_t next_special_clear_count{};
  std::uint32_t level_shape_cutoff{};
  std::uint32_t highest_chain{};
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
  bool previous_left_level{};
  bool previous_right_level{};
  bool terminated{};
  bool truncated{};
  std::vector<Body> bodies{};
  std::vector<GroupState> groups{};
  std::vector<std::uint64_t> active_contact_pairs{};
  std::vector<ContactImpulse> contact_impulses{};
  PhysicsOrdering physics_ordering{};
};

class Simulator {
 public:
  explicit Simulator(MechanicsConfig config = {});

  // Kept wide at the ABI boundary so out-of-range values can be rejected.
  Observation reset(std::uint64_t seed);  // v2.03 domain: 0..UINT32_MAX
  StepResult step(const Action& action);
  Observation observation() const;

  Snapshot clone_state() const;
  void restore_state(const Snapshot& snapshot);
  std::vector<std::byte> serialize_snapshot() const;
  void restore_snapshot(std::span<const std::byte> bytes);
  std::uint64_t state_hash() const;
  std::uint64_t config_hash() const;

  const MechanicsConfig& config() const { return config_; }
  const std::vector<Body>& bodies() const { return bodies_; }

  BodyId spawn_piece(Shape shape, std::int32_t color, double size, Vec2 position);
  BodyId spawn_bonus(Vec2 position);

 private:
  void tick_once(bool left_level, bool right_level, double cursor_x,
                 double cursor_y, bool suppress_fresh_edges,
                 std::vector<Event>& events);
  void fire(ShotStrength strength, double x, double y, std::vector<Event>& events);
  BodyId spawn_random_piece(double y, std::uint32_t parameter_level);
  void maybe_spawn(std::uint32_t parameter_level,
                   std::vector<Event>& events);
  void process_contacts(const std::vector<Contact>& contacts, bool left_edge,
                        bool right_edge, double cursor_x, double cursor_y,
                        std::vector<Event>& events);
  void process_actor_updates(
      const std::vector<std::tuple<BodyId, Vec2, double>>& scripted_origins,
                             std::vector<Event>& events);
  void process_scene_gauge(std::vector<Event>& events);
  void update_level(bool left_edge, bool right_edge, double cursor_x,
                    double cursor_y, std::vector<Event>& events);
  void finish_game();
  void refresh_lifecycle(Body& body);
  void add_score(std::int64_t delta, const char* detail, std::vector<Event>& events);
  void sequence_events(std::vector<Event>& events);
  void mark_deleted(Body& body, bool delayed);
  std::uint32_t current_color_count() const;
  std::uint32_t color_count_for_level(std::uint32_t parameter_level) const;
  std::uint32_t current_spawn_interval() const;
  std::uint32_t spawn_interval_for_level(
      std::uint32_t parameter_level) const;
  Body* find_body(BodyId id);
  GroupState* find_group(ChainId id);
  void compact_deleted();
  std::uint32_t allocate_actor_slot();
  StepDiagnostics diagnostics() const;
  std::uint64_t calculate_config_hash() const;

  MechanicsConfig config_;
  PhysicsWorld physics_;
  std::uint64_t config_hash_{};
  DxRandom rng_;
  std::vector<Body> bodies_;
  std::vector<GroupState> groups_;
  std::uint64_t tick_{};
  std::uint64_t scene_frame_{};
  BodyId next_body_id_{1};
  ChainId next_chain_id_{1};
  std::uint32_t actor_pool_cursor_{};
  std::array<std::int32_t, MechanicsConfig::actor_pool_capacity>
      actor_pool_colors_{};
  std::uint64_t next_event_sequence_{};
  std::uint64_t spawn_count_{};
  std::int64_t score_{};
  std::int64_t gauge_{};
  std::uint32_t level_{1};
  std::uint64_t qualifying_clear_count_{};
  std::uint64_t next_special_clear_count_{};
  std::uint32_t level_shape_cutoff_{};
  std::uint32_t highest_chain_{};
  std::uint64_t finish_call_count_{};
  bool terminal_metadata_recorded_{};
  std::int64_t recorded_final_score_{};
  std::uint32_t recorded_final_highest_chain_{};
  std::uint32_t recorded_final_level_{};
  std::uint64_t recorded_final_clears_{};
  std::int64_t latest_final_score_{};
  std::uint32_t latest_final_highest_chain_{};
  std::uint32_t latest_final_level_{};
  std::uint64_t latest_final_clears_{};
  bool previous_left_level_{};
  bool previous_right_level_{};
  bool terminated_{};
  bool truncated_{};
  bool actor_counter_guard_required_{};
  bool numeric_guard_required_{};
  std::vector<std::uint64_t> active_contact_pairs_{};
};

}  // namespace irisu
