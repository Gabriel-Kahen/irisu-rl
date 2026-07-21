#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(IRISU_BUILDING_DLL)
#define IRISU_API __declspec(dllexport)
#elif defined(_WIN32)
#define IRISU_API __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
#define IRISU_API __attribute__((visibility("default")))
#else
#define IRISU_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct irisu_simulator irisu_simulator;
typedef struct irisu_config_override {
  const char* key;
  double value;
} irisu_config_override;

// Different simulator handles may be used concurrently. A raw C handle has
// unique ownership: callers must externally serialize operations on the same
// handle and must not race irisu_destroy with another operation. The bundled
// Python owner provides that serialization.

// Stable symbolic values for integer action, body, and event fields. The enum
// typedef sizes are not part of the wire layout; padded structs retain their
// fixed-width integer fields below.
typedef enum irisu_action_kind {
  IRISU_ACTION_KIND_WAIT = 0,
  IRISU_ACTION_KIND_WEAK_SHOT = 1,
  IRISU_ACTION_KIND_STRONG_SHOT = 2,
  IRISU_ACTION_KIND_BOTH_SHOTS = 3,
} irisu_action_kind;

typedef enum irisu_body_kind {
  IRISU_BODY_KIND_PIECE = 0,
  IRISU_BODY_KIND_PROJECTILE = 1,
  IRISU_BODY_KIND_BONUS = 2,
} irisu_body_kind;

typedef enum irisu_shape {
  IRISU_SHAPE_CIRCLE = 0,
  IRISU_SHAPE_BOX = 1,
  IRISU_SHAPE_TRIANGLE = 2,
} irisu_shape;

typedef enum irisu_lifecycle {
  IRISU_LIFECYCLE_SCRIPTED_FALLING = 0,
  IRISU_LIFECYCLE_DYNAMIC_FRESH = 1,
  IRISU_LIFECYCLE_CONFIRMED = 2,
  IRISU_LIFECYCLE_ROTTEN = 3,
  IRISU_LIFECYCLE_DELETED = 4,
} irisu_lifecycle;

typedef enum irisu_event_kind {
  IRISU_EVENT_KIND_INVALID_ACTION = 0,
  IRISU_EVENT_KIND_SPAWNED = 1,
  IRISU_EVENT_KIND_SHOT_FIRED = 2,
  IRISU_EVENT_KIND_ACTIVATED = 3,
  IRISU_EVENT_KIND_CONTACT = 4,
  IRISU_EVENT_KIND_CONFIRMED = 5,
  IRISU_EVENT_KIND_CHAIN_JOINED = 6,
  IRISU_EVENT_KIND_CLEARED = 7,
  IRISU_EVENT_KIND_ROTTEN = 8,
  IRISU_EVENT_KIND_EJECTED = 9,
  IRISU_EVENT_KIND_DESTROYED = 10,
  IRISU_EVENT_KIND_GAUGE_CHANGED = 11,
  IRISU_EVENT_KIND_SCORE_CHANGED = 12,
  IRISU_EVENT_KIND_LEVEL_CHANGED = 13,
  IRISU_EVENT_KIND_GAME_OVER = 14,
  IRISU_EVENT_KIND_PROJECTILE_HIT = 15,
  IRISU_EVENT_KIND_PROJECTILE_CONTACT = 16,
  IRISU_EVENT_KIND_HELD_INPUT_IGNORED = 17,
  IRISU_EVENT_KIND_LEVEL_COMPLETED = 18,
} irisu_event_kind;

// Additive typed ABI for high-throughput padded observations.
#define IRISU_PADDED_BODY_CAPACITY 196u
#define IRISU_EVENT_DETAIL_CAPACITY 96u

typedef struct irisu_padded_body_v1 {
  uint64_t age_ticks;
  int64_t remaining_lifetime;
  uint64_t rot_timer;
  double x;
  double y;
  double vx;
  double vy;
  double angle;
  double angular_velocity;
  double size;
  uint32_t id;
  int32_t color;
  uint32_t chain_id;
  uint32_t projectile_hits;
  uint8_t kind;      // irisu_body_kind value
  uint8_t shape;     // irisu_shape value
  uint8_t lifecycle; // irisu_lifecycle value
  uint8_t reserved;
} irisu_padded_body_v1;

typedef struct irisu_padded_observation_v1 {
  uint64_t tick;
  int64_t score;
  int64_t gauge;
  int64_t gauge_max;
  uint64_t qualifying_clear_count;
  double field_x;
  double field_y;
  double field_width;
  double field_height;
  double side_wall_top;
  double side_wall_bottom;
  uint32_t level;
  uint32_t active_colors;
  uint32_t spawn_interval_ticks;
  uint32_t highest_chain;
  uint32_t body_count;
  uint8_t terminated;
  uint8_t truncated;
  uint8_t left_held;
  uint8_t right_held;
  irisu_padded_body_v1 bodies[IRISU_PADDED_BODY_CAPACITY];
} irisu_padded_observation_v1;

typedef struct irisu_padded_transition_v1 {
  irisu_padded_observation_v1 observation;
  int64_t reward;
  uint64_t event_count;
  uint64_t config_hash;
  uint64_t finish_call_count;
  int64_t recorded_final_score;
  uint64_t recorded_final_clears;
  int64_t latest_final_score;
  uint64_t latest_final_clears;
  uint32_t recorded_final_highest_chain;
  uint32_t recorded_final_level;
  uint32_t latest_final_highest_chain;
  uint32_t latest_final_level;
  uint8_t terminated;
  uint8_t truncated;
  uint8_t terminal_metadata_recorded;
  uint8_t invalid_action;
} irisu_padded_transition_v1;

typedef struct irisu_padded_action_v1 {
  double x;
  double y;
  uint32_t wait_ticks;
  int32_t kind; // irisu_action_kind value
} irisu_padded_action_v1;

typedef struct irisu_padded_event_v1 {
  uint64_t tick;
  uint64_t sequence;
  int64_t value;
  uint32_t a;
  uint32_t b;
  uint16_t detail_size;
  uint8_t kind; // irisu_event_kind value
  uint8_t reserved;
  char detail[IRISU_EVENT_DETAIL_CAPACITY];
} irisu_padded_event_v1;

IRISU_API uint32_t irisu_abi_version(void);
IRISU_API irisu_simulator* irisu_create(void);
IRISU_API void irisu_destroy(irisu_simulator* simulator);
IRISU_API int irisu_configure(irisu_simulator* simulator,
                              const irisu_config_override* overrides,
                              size_t override_count);
/* The v2.03 normal RNG seed is uint32; larger values return an error. */
IRISU_API int irisu_reset(irisu_simulator* simulator, uint64_t seed);
/* action_kind must be an irisu_action_kind value. */
IRISU_API int irisu_step(irisu_simulator* simulator, int action_kind, double x, double y,
                         uint32_t wait_ticks);
IRISU_API uint32_t irisu_padded_abi_version(void);
IRISU_API size_t irisu_padded_body_capacity(void);
IRISU_API size_t irisu_padded_observation_size(void);
IRISU_API size_t irisu_padded_transition_size(void);
IRISU_API size_t irisu_padded_action_size(void);
IRISU_API size_t irisu_padded_event_size(void);
IRISU_API int irisu_padded_observation(irisu_simulator* simulator,
                                      irisu_padded_observation_v1* destination);
/* The v2.03 normal RNG seed is uint32; larger values return an error. */
IRISU_API int irisu_padded_reset(irisu_simulator* simulator, uint64_t seed,
                                irisu_padded_observation_v1* destination);
/* action_kind must be an irisu_action_kind value. */
IRISU_API int irisu_padded_step(irisu_simulator* simulator, int action_kind,
                               double x, double y, uint32_t wait_ticks,
                               irisu_padded_transition_v1* destination);
IRISU_API int irisu_padded_step_batch(
    irisu_simulator* const* simulators,
    const irisu_padded_action_v1* actions,
    irisu_padded_transition_v1* destinations, uint8_t* statuses,
    size_t simulator_count, size_t worker_count);
IRISU_API int irisu_padded_events(irisu_simulator* simulator,
                                 irisu_padded_event_v1* destination,
                                 size_t event_capacity);
IRISU_API const char* irisu_observation_json(irisu_simulator* simulator);
IRISU_API const char* irisu_step_json(irisu_simulator* simulator);
IRISU_API uint64_t irisu_state_hash(const irisu_simulator* simulator);
IRISU_API uint64_t irisu_config_hash(const irisu_simulator* simulator);
IRISU_API const char* irisu_config_json(irisu_simulator* simulator);
IRISU_API const char* irisu_build_info_json(void);
IRISU_API size_t irisu_snapshot_size(irisu_simulator* simulator);
IRISU_API int irisu_snapshot_write(irisu_simulator* simulator, void* destination, size_t size);
/* A successful restore clears the last-step JSON and padded event list. */
IRISU_API int irisu_snapshot_restore(irisu_simulator* simulator, const void* source, size_t size);
IRISU_API const char* irisu_last_error(const irisu_simulator* simulator);

#ifdef __cplusplus
}
#endif
