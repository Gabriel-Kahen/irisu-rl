#include <stdint.h>
#include <stdlib.h>

int msvc_fltused;

void *msvc_operator_new(unsigned int size) { return malloc(size); }
void msvc_operator_delete(void *pointer) { free(pointer); }
int msvc_atexit(void (*function)(void)) {
  (void)function;
  return 0;
}
int msvc_finite(double value) {
  union {
    double value;
    uint64_t bits;
  } decoded = {value};
  return (decoded.bits & UINT64_C(0x7ff0000000000000)) !=
         UINT64_C(0x7ff0000000000000);
}
void msvc_purecall(void) { abort(); }

#define STDCALL __attribute__((stdcall))

extern void *STDCALL msvc_b2d_world_create(float, float, float, float, float,
                                            float);
extern void STDCALL msvc_b2d_world_destroy(void *);
extern void *STDCALL msvc_b2d_world_create_box(
    void *, float, float, float, float, float, float, float, float);
extern void *STDCALL msvc_b2d_world_create_triangle(
    void *, float, float, float, float, float, float, float, float);
extern void *STDCALL msvc_b2d_world_create_circle(
    void *, float, float, float, float, float, float);
extern void STDCALL msvc_b2d_world_destroy_body(void *, void *);
extern void STDCALL msvc_b2d_world_step(void *, float, int);
extern int STDCALL msvc_b2d_world_get_contact(void *, void **, void **);
extern float STDCALL msvc_b2d_world_get_x(void *, void *);
extern float STDCALL msvc_b2d_world_get_y(void *, void *);
extern float STDCALL msvc_b2d_world_get_r(void *, void *);
extern void STDCALL msvc_b2d_world_get_v(void *, void *, float *, float *);
extern void STDCALL msvc_b2d_world_set_v(void *, void *, float, float);
extern void STDCALL msvc_b2d_world_set_user_data(void *, void *, void *);
extern void STDCALL msvc_b2d_world_set_position(void *, void *, float, float,
                                                float);
extern void STDCALL msvc_b2d_world_test(void *, void *);

/* Pristine r58 has a few process-wide lazy tables/counters. Serializing calls
   makes distinct handles safe across host threads without changing MSVC code. */
static volatile int api_lock;

static void lock_api(void) {
  while (__sync_lock_test_and_set(&api_lock, 1)) {
  }
}

static void unlock_api(void) { __sync_lock_release(&api_lock); }

void *b2d_world_create(float a, float b, float c, float d, float e, float f) {
  void *result;
  lock_api();
  result = msvc_b2d_world_create(a, b, c, d, e, f);
  unlock_api();
  return result;
}

void b2d_world_destroy(void *a) {
  lock_api();
  msvc_b2d_world_destroy(a);
  unlock_api();
}

void *b2d_world_create_box(void *a, float b, float c, float d, float e,
                           float f, float g, float h, float i) {
  void *result;
  lock_api();
  result = msvc_b2d_world_create_box(a, b, c, d, e, f, g, h, i);
  unlock_api();
  return result;
}

void *b2d_world_create_triangle(void *a, float b, float c, float d, float e,
                                float f, float g, float h, float i) {
  void *result;
  lock_api();
  result = msvc_b2d_world_create_triangle(a, b, c, d, e, f, g, h, i);
  unlock_api();
  return result;
}

void *b2d_world_create_circle(void *a, float b, float c, float d, float e,
                              float f, float g) {
  void *result;
  lock_api();
  result = msvc_b2d_world_create_circle(a, b, c, d, e, f, g);
  unlock_api();
  return result;
}

void b2d_world_destroy_body(void *a, void *b) {
  lock_api();
  msvc_b2d_world_destroy_body(a, b);
  unlock_api();
}

void b2d_world_step(void *a, float b, int c) {
  lock_api();
  msvc_b2d_world_step(a, b, c);
  unlock_api();
}

int b2d_world_get_contact(void *a, void **b, void **c) {
  int result;
  lock_api();
  result = msvc_b2d_world_get_contact(a, b, c);
  unlock_api();
  return result;
}

float b2d_world_get_x(void *a, void *b) {
  float result;
  lock_api();
  result = msvc_b2d_world_get_x(a, b);
  unlock_api();
  return result;
}

float b2d_world_get_y(void *a, void *b) {
  float result;
  lock_api();
  result = msvc_b2d_world_get_y(a, b);
  unlock_api();
  return result;
}

float b2d_world_get_r(void *a, void *b) {
  float result;
  lock_api();
  result = msvc_b2d_world_get_r(a, b);
  unlock_api();
  return result;
}

void b2d_world_get_v(void *a, void *b, float *c, float *d) {
  lock_api();
  msvc_b2d_world_get_v(a, b, c, d);
  unlock_api();
}

void b2d_world_set_v(void *a, void *b, float c, float d) {
  lock_api();
  msvc_b2d_world_set_v(a, b, c, d);
  unlock_api();
}

void b2d_world_set_user_data(void *a, void *b, void *c) {
  lock_api();
  msvc_b2d_world_set_user_data(a, b, c);
  unlock_api();
}

void b2d_world_set_position(void *a, void *b, float c, float d, float e) {
  lock_api();
  msvc_b2d_world_set_position(a, b, c, d, e);
  unlock_api();
}

void b2d_world_test(void *a, void *b) {
  lock_api();
  msvc_b2d_world_test(a, b);
  unlock_api();
}
