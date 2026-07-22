#define _GNU_SOURCE

#include <dlfcn.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
  OP_INIT = 1,
  OP_CREATE_BOX,
  OP_CREATE_TRIANGLE,
  OP_CREATE_CIRCLE,
  OP_DESTROY,
  OP_CONTACT,
  OP_GET_X,
  OP_GET_Y,
  OP_GET_R,
  OP_GET_V,
  OP_SET_POSITION,
  OP_SET_USER_DATA,
  OP_SET_V,
  OP_STEP,
  OP_DISPOSE,
  OP_END = 255
};

typedef void *(*world_create_fn)(float, float, float, float, float, float);
typedef void (*world_destroy_fn)(void *);
typedef void *(*world_create_box_fn)(void *, float, float, float, float, float,
                                     float, float, float);
typedef void *(*world_create_circle_fn)(void *, float, float, float, float,
                                        float, float);
typedef void (*world_destroy_body_fn)(void *, void *);
typedef void (*world_step_fn)(void *, float, int);
typedef int (*world_get_contact_fn)(void *, void **, void **);
typedef float (*world_get_scalar_fn)(void *, void *);
typedef void (*world_get_v_fn)(void *, void *, float *, float *);
typedef void (*world_set_v_fn)(void *, void *, float, float);
typedef void (*world_set_user_data_fn)(void *, void *, void *);
typedef void (*world_set_position_fn)(void *, void *, float, float, float);

struct Api {
  world_create_fn create;
  world_destroy_fn destroy;
  world_create_box_fn create_box;
  world_create_box_fn create_triangle;
  world_create_circle_fn create_circle;
  world_destroy_body_fn destroy_body;
  world_step_fn step;
  world_get_contact_fn get_contact;
  world_get_scalar_fn get_x;
  world_get_scalar_fn get_y;
  world_get_scalar_fn get_r;
  world_get_v_fn get_v;
  world_set_v_fn set_v;
  world_set_user_data_fn set_user_data;
  world_set_position_fn set_position;
};

struct FirstMismatch {
  int present;
  int is_float;
  uint32_t sequence;
  uint32_t ordinal;
  uint32_t expected;
  uint32_t actual;
  const char *operation;
  const char *component;
};

struct State {
  struct Api api;
  void *library;
  void *world;
  void **bodies;
  size_t body_capacity;
  uint64_t commands;
  uint64_t getter_records;
  uint64_t getter_values;
  uint64_t scalar_getters;
  uint64_t velocity_getters;
  uint64_t getter_record_mismatches;
  uint64_t getter_value_mismatches;
  uint64_t contacts;
  uint64_t contact_mismatches;
  uint32_t step;
  struct FirstMismatch first;
};

static int read_bytes(void *value, size_t size) {
  return fread(value, 1, size, stdin) == size;
}

static int read_u8(uint8_t *value) { return read_bytes(value, sizeof(*value)); }
static int read_u32(uint32_t *value) {
  return read_bytes(value, sizeof(*value));
}

static float as_float(uint32_t bits) {
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static uint32_t as_bits(float value) {
  uint32_t bits;
  memcpy(&bits, &value, sizeof(bits));
  return bits;
}

static void fail(const char *message) {
  fprintf(stderr, "%s\n", message);
  exit(2);
}

static uint32_t need_u32(void) {
  uint32_t value;
  if (!read_u32(&value))
    fail("truncated getter-trace command stream");
  return value;
}

static void *need_body(struct State *state, uint32_t ordinal) {
  if (ordinal == 0 || ordinal >= state->body_capacity ||
      state->bodies[ordinal] == NULL)
    fail("getter trace references an inactive body ordinal");
  return state->bodies[ordinal];
}

static void remember_body(struct State *state, uint32_t ordinal, void *body) {
  if (ordinal == 0 || body == NULL)
    fail("exact backend rejected a traced body creation");
  if (ordinal >= state->body_capacity) {
    size_t capacity = state->body_capacity ? state->body_capacity : 64;
    while (capacity <= ordinal)
      capacity *= 2;
    void **resized = realloc(state->bodies, capacity * sizeof(*resized));
    if (resized == NULL)
      fail("cannot grow body ordinal table");
    memset(resized + state->body_capacity, 0,
           (capacity - state->body_capacity) * sizeof(*resized));
    state->bodies = resized;
    state->body_capacity = capacity;
  }
  if (state->bodies[ordinal] != NULL)
    fail("getter trace reuses a live body ordinal");
  state->bodies[ordinal] = body;
}

static void clear_bodies(struct State *state) {
  if (state->bodies != NULL)
    memset(state->bodies, 0, state->body_capacity * sizeof(*state->bodies));
}

static void mismatch(struct State *state, uint32_t sequence, uint32_t ordinal,
                     const char *operation, const char *component,
                     uint32_t expected, uint32_t actual, int is_float) {
  if (state->first.present)
    return;
  state->first = (struct FirstMismatch){
      1, is_float, sequence, ordinal, expected, actual, operation, component};
}

static void compare_getter(struct State *state, uint32_t sequence,
                           uint32_t ordinal, const char *operation,
                           const char *component, uint32_t expected,
                           uint32_t actual) {
  ++state->getter_values;
  if (expected != actual) {
    ++state->getter_value_mismatches;
    mismatch(state, sequence, ordinal, operation, component, expected, actual,
             1);
  }
}

static void load_symbol(void *library, const char *name, void *target,
                        size_t size) {
  dlerror();
  void *symbol = dlsym(library, name);
  const char *error = dlerror();
  if (error != NULL || symbol == NULL || size != sizeof(symbol))
    fail(error != NULL ? error : "cannot resolve exact backend symbol");
  memcpy(target, &symbol, size);
}

#define LOAD(api, library, field)                                              \
  load_symbol((library), "b2d_world_" #field, &(api)->field,                   \
              sizeof((api)->field))

static void load_api(struct State *state, const char *path) {
  state->library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (state->library == NULL)
    fail(dlerror());
  LOAD(&state->api, state->library, create);
  LOAD(&state->api, state->library, destroy);
  LOAD(&state->api, state->library, create_box);
  LOAD(&state->api, state->library, create_triangle);
  LOAD(&state->api, state->library, create_circle);
  LOAD(&state->api, state->library, destroy_body);
  LOAD(&state->api, state->library, step);
  LOAD(&state->api, state->library, get_contact);
  LOAD(&state->api, state->library, get_x);
  LOAD(&state->api, state->library, get_y);
  LOAD(&state->api, state->library, get_r);
  LOAD(&state->api, state->library, get_v);
  LOAD(&state->api, state->library, set_v);
  LOAD(&state->api, state->library, set_user_data);
  LOAD(&state->api, state->library, set_position);
}

static void replay_init(struct State *state) {
  uint32_t bits[6];
  uint8_t expected;
  for (size_t index = 0; index < 6; ++index)
    bits[index] = need_u32();
  if (!read_u8(&expected))
    fail("truncated init result");
  if (state->world != NULL)
    state->api.destroy(state->world);
  state->world = state->api.create(as_float(bits[0]), as_float(bits[1]),
                                   as_float(bits[2]), as_float(bits[3]),
                                   as_float(bits[4]), as_float(bits[5]));
  clear_bodies(state);
  state->step = 0;
  if ((state->world != NULL) != (expected != 0))
    fail("exact backend init result differs from trace");
}

static void replay_create(struct State *state, uint8_t opcode) {
  const uint32_t ordinal = need_u32();
  uint32_t bits[8] = {0};
  const size_t count = opcode == OP_CREATE_CIRCLE ? 6 : 8;
  for (size_t index = 0; index < count; ++index)
    bits[index] = need_u32();
  void *body;
  if (opcode == OP_CREATE_CIRCLE) {
    body = state->api.create_circle(
        state->world, as_float(bits[0]), as_float(bits[1]), as_float(bits[2]),
        as_float(bits[3]), as_float(bits[4]), as_float(bits[5]));
  } else {
    world_create_box_fn create = opcode == OP_CREATE_BOX
                                     ? state->api.create_box
                                     : state->api.create_triangle;
    body = create(state->world, as_float(bits[0]), as_float(bits[1]),
                  as_float(bits[2]), as_float(bits[3]), as_float(bits[4]),
                  as_float(bits[5]), as_float(bits[6]), as_float(bits[7]));
  }
  remember_body(state, ordinal, body);
}

static void replay_contact(struct State *state, uint32_t sequence) {
  uint8_t expected_result;
  if (!read_u8(&expected_result))
    fail("truncated contact result");
  const uint32_t expected_a = need_u32();
  const uint32_t expected_b = need_u32();
  void *a = NULL;
  void *b = NULL;
  const int actual_result = state->api.get_contact(state->world, &a, &b);
  const uint32_t actual_a = actual_result ? (uint32_t)(uintptr_t)a : 0;
  const uint32_t actual_b = actual_result ? (uint32_t)(uintptr_t)b : 0;
  int different = (actual_result != 0) != (expected_result != 0);
  if (different) {
    mismatch(state, sequence, 0, "contact", "result", expected_result,
             actual_result != 0, 0);
  } else if (expected_result && actual_a != expected_a) {
    different = 1;
    mismatch(state, sequence, 0, "contact", "a_user", expected_a, actual_a, 0);
  } else if (expected_result && actual_b != expected_b) {
    different = 1;
    mismatch(state, sequence, 0, "contact", "b_user", expected_b, actual_b, 0);
  }
  ++state->contacts;
  if (different)
    ++state->contact_mismatches;
}

static void replay_scalar_getter(struct State *state, uint8_t opcode,
                                 uint32_t sequence) {
  const uint32_t ordinal = need_u32();
  const uint32_t expected = need_u32();
  world_get_scalar_fn get = opcode == OP_GET_X   ? state->api.get_x
                            : opcode == OP_GET_Y ? state->api.get_y
                                                 : state->api.get_r;
  const char *operation = opcode == OP_GET_X   ? "get_x"
                          : opcode == OP_GET_Y ? "get_y"
                                               : "get_r";
  ++state->getter_records;
  ++state->scalar_getters;
  const uint64_t before = state->getter_value_mismatches;
  compare_getter(state, sequence, ordinal, operation, "value", expected,
                 as_bits(get(state->world, need_body(state, ordinal))));
  if (before != state->getter_value_mismatches)
    ++state->getter_record_mismatches;
}

static void replay_velocity_getter(struct State *state, uint32_t sequence) {
  const uint32_t ordinal = need_u32();
  const uint32_t expected_x = need_u32();
  const uint32_t expected_y = need_u32();
  float x = 0.0f;
  float y = 0.0f;
  state->api.get_v(state->world, need_body(state, ordinal), &x, &y);
  ++state->getter_records;
  ++state->velocity_getters;
  const uint64_t before = state->getter_value_mismatches;
  compare_getter(state, sequence, ordinal, "get_v", "x", expected_x,
                 as_bits(x));
  compare_getter(state, sequence, ordinal, "get_v", "y", expected_y,
                 as_bits(y));
  if (before != state->getter_value_mismatches)
    ++state->getter_record_mismatches;
}

static void replay_command(struct State *state, uint8_t opcode,
                           uint32_t sequence) {
  const uint32_t ordinal =
      opcode >= OP_DESTROY && opcode <= OP_SET_V && opcode != OP_CONTACT
          ? need_u32()
          : 0;
  switch (opcode) {
  case OP_INIT:
    replay_init(state);
    break;
  case OP_CREATE_BOX:
  case OP_CREATE_TRIANGLE:
  case OP_CREATE_CIRCLE:
    replay_create(state, opcode);
    break;
  case OP_DESTROY:
    state->api.destroy_body(state->world, need_body(state, ordinal));
    state->bodies[ordinal] = NULL;
    break;
  case OP_CONTACT:
    replay_contact(state, sequence);
    break;
  case OP_GET_X:
  case OP_GET_Y:
  case OP_GET_R:
    /* The common ordinal was consumed above; scalar replay reads its own. */
    fail("invalid scalar getter decoder state");
    break;
  case OP_GET_V:
    fail("invalid velocity getter decoder state");
    break;
  case OP_SET_POSITION: {
    const float x = as_float(need_u32());
    const float y = as_float(need_u32());
    const float r = as_float(need_u32());
    state->api.set_position(state->world, need_body(state, ordinal), x, y, r);
    break;
  }
  case OP_SET_USER_DATA:
    state->api.set_user_data(state->world, need_body(state, ordinal),
                             (void *)(uintptr_t)need_u32());
    break;
  case OP_SET_V: {
    const float x = as_float(need_u32());
    const float y = as_float(need_u32());
    state->api.set_v(state->world, need_body(state, ordinal), x, y);
    break;
  }
  case OP_STEP: {
    const uint32_t expected_step = need_u32();
    const float dt = as_float(need_u32());
    const int iterations = (int)need_u32();
    state->api.step(state->world, dt, iterations);
    ++state->step;
    if (state->step != expected_step)
      fail("trace step sequence differs");
    break;
  }
  case OP_DISPOSE:
    if (state->world == NULL)
      fail("dispose without an active world");
    state->api.destroy(state->world);
    state->world = NULL;
    clear_bodies(state);
    break;
  default:
    fail("unknown getter-trace command opcode");
  }
}

static void print_report(const struct State *state, uint16_t control_word) {
  const int exact = state->getter_record_mismatches == 0 &&
                    state->contact_mismatches == 0 && control_word == 0x027f;
  printf("{\"schema\":1,\"status\":\"%s\",\"x87_cw\":\"%04x\","
         "\"canonical_x87\":%s,"
         "\"commands\":%" PRIu64 ",\"final_step\":%u,"
         "\"getter_records\":%" PRIu64 ",\"getter_values\":%" PRIu64
         ",\"scalar_getters\":%" PRIu64 ",\"velocity_getters\":%" PRIu64
         ",\"getter_record_mismatches\":%" PRIu64
         ",\"getter_value_mismatches\":%" PRIu64 ",\"contacts\":%" PRIu64
         ",\"contact_mismatches\":%" PRIu64 ",\"first_mismatch\":",
         exact ? "exact" : "mismatch", control_word,
         control_word == 0x027f ? "true" : "false", state->commands,
         state->step, state->getter_records, state->getter_values,
         state->scalar_getters, state->velocity_getters,
         state->getter_record_mismatches, state->getter_value_mismatches,
         state->contacts, state->contact_mismatches);
  if (!state->first.present) {
    printf("null}\n");
    return;
  }
  printf("{\"seq\":%u,\"operation\":\"%s\",\"ordinal\":%u,"
         "\"component\":\"%s\",",
         state->first.sequence, state->first.operation, state->first.ordinal,
         state->first.component);
  if (state->first.is_float) {
    printf("\"expected_f32\":\"%08x\",\"actual_f32\":\"%08x\"}}\n",
           state->first.expected, state->first.actual);
  } else {
    printf("\"expected_u32\":%u,\"actual_u32\":%u}}\n", state->first.expected,
           state->first.actual);
  }
}

int main(int argc, char **argv) {
  if (argc != 2)
    fail("usage: getter-trace-replay exact-library.so");
  if (sizeof(void *) != 4)
    fail("getter-trace replay must be a 32-bit process");

  uint16_t control_word = 0x027f;
  __asm__ __volatile__("fldcw %0" : : "m"(control_word));
  __asm__ __volatile__("fnstcw %0" : "=m"(control_word));
  if (control_word != 0x027f)
    fail("cannot establish canonical x87 control word");

  unsigned char magic[8];
  static const unsigned char expected_magic[8] = {'I', 'R', 'G', 'T',
                                                  'R', 'C', '1', 0};
  if (!read_bytes(magic, sizeof(magic)) ||
      memcmp(magic, expected_magic, sizeof(magic)) != 0)
    fail("invalid getter-trace command stream header");

  struct State state = {0};
  load_api(&state, argv[1]);
  for (;;) {
    uint8_t opcode;
    uint32_t sequence;
    if (!read_u8(&opcode))
      fail("getter-trace command stream has no terminator");
    if (opcode == OP_END)
      break;
    if (!read_u32(&sequence))
      fail("truncated command sequence");
    ++state.commands;
    if (opcode == OP_GET_X || opcode == OP_GET_Y || opcode == OP_GET_R)
      replay_scalar_getter(&state, opcode, sequence);
    else if (opcode == OP_GET_V)
      replay_velocity_getter(&state, sequence);
    else
      replay_command(&state, opcode, sequence);
  }
  if (state.world != NULL)
    fail("getter trace ended before dispose");
  __asm__ __volatile__("fnstcw %0" : "=m"(control_word));
  print_report(&state, control_word);
  free(state.bodies);
  dlclose(state.library);
  return state.getter_record_mismatches == 0 && state.contact_mismatches == 0 &&
                 control_word == 0x027f
             ? 0
             : 1;
}
