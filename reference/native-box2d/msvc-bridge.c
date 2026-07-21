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

extern int STDCALL msvc_b2d_init(float, float, float, float, float, float);
extern void STDCALL msvc_b2d_dispose(void);
extern void *STDCALL msvc_b2d_create_box(float, float, float, float, float,
                                          float, float, float);
extern void *STDCALL msvc_b2d_create_triangle(float, float, float, float, float,
                                               float, float, float);
extern void *STDCALL msvc_b2d_create_circle(float, float, float, float, float,
                                             float);
extern void STDCALL msvc_b2d_destroy_body(void *);
extern void STDCALL msvc_b2d_step(float, int);
extern int STDCALL msvc_b2d_get_contact(void **, void **);
extern float STDCALL msvc_b2d_get_x(void *);
extern float STDCALL msvc_b2d_get_y(void *);
extern float STDCALL msvc_b2d_get_r(void *);
extern void STDCALL msvc_b2d_get_v(void *, float *, float *);
extern void STDCALL msvc_b2d_set_v(void *, float, float);
extern void STDCALL msvc_b2d_set_user_data(void *, void *);
extern void STDCALL msvc_b2d_set_position(void *, float, float, float);
extern void STDCALL msvc_b2d_test(void *);

int b2d_init(float a, float b, float c, float d, float e, float f) {
  return msvc_b2d_init(a, b, c, d, e, f);
}
void b2d_dispose(void) { msvc_b2d_dispose(); }
void *b2d_create_box(float a, float b, float c, float d, float e, float f,
                     float g, float h) {
  return msvc_b2d_create_box(a, b, c, d, e, f, g, h);
}
void *b2d_create_triangle(float a, float b, float c, float d, float e, float f,
                          float g, float h) {
  return msvc_b2d_create_triangle(a, b, c, d, e, f, g, h);
}
void *b2d_create_circle(float a, float b, float c, float d, float e, float f) {
  return msvc_b2d_create_circle(a, b, c, d, e, f);
}
void b2d_destroy_body(void *a) { msvc_b2d_destroy_body(a); }
void b2d_step(float a, int b) { msvc_b2d_step(a, b); }
int b2d_get_contact(void **a, void **b) { return msvc_b2d_get_contact(a, b); }
float b2d_get_x(void *a) { return msvc_b2d_get_x(a); }
float b2d_get_y(void *a) { return msvc_b2d_get_y(a); }
float b2d_get_r(void *a) { return msvc_b2d_get_r(a); }
void b2d_get_v(void *a, float *b, float *c) { msvc_b2d_get_v(a, b, c); }
void b2d_set_v(void *a, float b, float c) { msvc_b2d_set_v(a, b, c); }
void b2d_set_user_data(void *a, void *b) { msvc_b2d_set_user_data(a, b); }
void b2d_set_position(void *a, float b, float c, float d) {
  msvc_b2d_set_position(a, b, c, d);
}
void b2d_test(void *a) { msvc_b2d_test(a); }
