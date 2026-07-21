#include <stdint.h>
#include <stdio.h>
#include <string.h>

int b2d_init(float, float, float, float, float, float);
void b2d_dispose(void);
void *b2d_create_box(float, float, float, float, float, float, float, float);
void b2d_step(float, int);
int b2d_get_contact(void **, void **);
float b2d_get_x(void *);
float b2d_get_y(void *);
float b2d_get_r(void *);
void b2d_get_v(void *, float *, float *);
void b2d_set_user_data(void *, void *);

void *b2d_world_create(float, float, float, float, float, float);
void b2d_world_destroy(void *);
void *b2d_world_create_box(void *, float, float, float, float, float, float,
                           float, float);
void *b2d_world_create_circle(void *, float, float, float, float, float,
                              float);
void b2d_world_step(void *, float, int);
int b2d_world_get_contact(void *, void **, void **);
float b2d_world_get_x(void *, void *);
float b2d_world_get_y(void *, void *);
float b2d_world_get_r(void *, void *);
void b2d_world_get_v(void *, void *, float *, float *);
void b2d_world_set_user_data(void *, void *, void *);

static uint32_t bits(float value) {
  uint32_t result;
  memcpy(&result, &value, sizeof(result));
  return result;
}

static int same_float(float left, float right, const char *field, int step) {
  if (bits(left) == bits(right)) return 1;
  fprintf(stderr, "%s differs at step %d: %08x != %08x\n", field, step,
          bits(left), bits(right));
  return 0;
}

static int parity(void) {
  void *world;
  void *legacy_floor;
  void *legacy_body;
  void *multi_floor;
  void *multi_body;
  int step;

  if (!b2d_init(-1000.0f, -1000.0f, 1000.0f, 1000.0f, 300.0f, 30.0f))
    return 0;
  world = b2d_world_create(-1000.0f, -1000.0f, 1000.0f, 1000.0f, 300.0f,
                           30.0f);
  legacy_floor = b2d_create_box(300.0f, 20.0f, 100.0f, 300.0f, 0.0f, 0.0f,
                                1.0f, 0.0f);
  legacy_body = b2d_create_box(20.0f, 20.0f, 100.0f, 100.0f, 0.0f, 1.0f,
                               0.2f, 0.1f);
  multi_floor = b2d_world_create_box(world, 300.0f, 20.0f, 100.0f, 300.0f,
                                     0.0f, 0.0f, 1.0f, 0.0f);
  multi_body = b2d_world_create_box(world, 20.0f, 20.0f, 100.0f, 100.0f,
                                    0.0f, 1.0f, 0.2f, 0.1f);
  if (!world || !legacy_floor || !legacy_body || !multi_floor || !multi_body)
    return 0;
  b2d_set_user_data(legacy_floor, (void *)(uintptr_t)1);
  b2d_set_user_data(legacy_body, (void *)(uintptr_t)2);
  b2d_world_set_user_data(world, multi_floor, (void *)(uintptr_t)1);
  b2d_world_set_user_data(world, multi_body, (void *)(uintptr_t)2);

  for (step = 1; step <= 180; ++step) {
    float lvx, lvy, mvx, mvy;
    void *la, *lb, *ma, *mb;
    int lc, mc;
    b2d_step(1.0f / 60.0f, 10);
    b2d_world_step(world, 1.0f / 60.0f, 10);
    if (!same_float(b2d_get_x(legacy_body), b2d_world_get_x(world, multi_body),
                    "x", step) ||
        !same_float(b2d_get_y(legacy_body), b2d_world_get_y(world, multi_body),
                    "y", step) ||
        !same_float(b2d_get_r(legacy_body), b2d_world_get_r(world, multi_body),
                    "rotation", step))
      return 0;
    b2d_get_v(legacy_body, &lvx, &lvy);
    b2d_world_get_v(world, multi_body, &mvx, &mvy);
    if (!same_float(lvx, mvx, "velocity_x", step) ||
        !same_float(lvy, mvy, "velocity_y", step))
      return 0;
    do {
      lc = b2d_get_contact(&la, &lb);
      mc = b2d_world_get_contact(world, &ma, &mb);
      if (lc != mc || (lc && (la != ma || lb != mb))) {
        fprintf(stderr, "contacts differ at step %d\n", step);
        return 0;
      }
    } while (lc);
  }
  b2d_world_destroy(world);
  b2d_dispose();
  return 1;
}

static int independent_worlds(void) {
  void *first = b2d_world_create(-1000.0f, -1000.0f, 1000.0f, 1000.0f,
                                 300.0f, 30.0f);
  void *second = b2d_world_create(-1000.0f, -1000.0f, 1000.0f, 1000.0f,
                                  -150.0f, 10.0f);
  void *first_body = b2d_world_create_circle(first, 10.0f, 100.0f, 100.0f,
                                              1.0f, 0.2f, 0.0f);
  void *second_body = b2d_world_create_circle(second, 10.0f, -50.0f, 25.0f,
                                               1.0f, 0.2f, 0.0f);
  const uint32_t second_initial = bits(b2d_world_get_y(second, second_body));
  uint32_t first_after;
  int step;
  if (!first || !second || !first_body || !second_body) return 0;

  for (step = 0; step < 60; ++step)
    b2d_world_step(first, 1.0f / 60.0f, 10);
  if (bits(b2d_world_get_y(second, second_body)) != second_initial) {
    fprintf(stderr, "stepping world A changed world B\n");
    return 0;
  }
  first_after = bits(b2d_world_get_y(first, first_body));
  for (step = 0; step < 30; ++step)
    b2d_world_step(second, 1.0f / 60.0f, 10);
  if (bits(b2d_world_get_y(first, first_body)) != first_after) {
    fprintf(stderr, "stepping world B changed world A\n");
    return 0;
  }
  b2d_world_destroy(first);
  for (step = 0; step < 30; ++step)
    b2d_world_step(second, 1.0f / 60.0f, 10);
  if (bits(b2d_world_get_y(second, second_body)) == second_initial) {
    fprintf(stderr, "world B did not advance after world A destruction\n");
    return 0;
  }
  b2d_world_destroy(second);
  return 1;
}

int main(void) {
  const uint16_t control_word = 0x027f;
  __asm__ __volatile__("fldcw %0" : : "m"(control_word));
  if (!parity() || !independent_worlds()) return 1;
  puts("180-step legacy parity and two-world isolation passed");
  return 0;
}
