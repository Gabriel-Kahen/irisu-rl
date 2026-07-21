// Adapts the validated original one-world b2d_* ABI to the handle-taking ABI
// used by the evolving exact frontend. It is only valid when one PhysicsWorld
// exists in the process, which is the condition exercised by this prototype.

extern "C" {
int b2d_init(float, float, float, float, float, float);
void b2d_dispose();
void* b2d_create_box(float, float, float, float, float, float, float, float);
void* b2d_create_triangle(float, float, float, float, float, float, float,
                          float);
void* b2d_create_circle(float, float, float, float, float, float);
void b2d_destroy_body(void*);
void b2d_step(float, int);
int b2d_get_contact(void**, void**);
float b2d_get_x(void*);
float b2d_get_y(void*);
float b2d_get_r(void*);
void b2d_get_v(void*, float*, float*);
void b2d_set_v(void*, float, float);
void b2d_set_user_data(void*, void*);
void b2d_set_position(void*, float, float, float);

void* b2d_world_create(float min_x, float min_y, float max_x, float max_y,
                       float gravity_y, float magnification) {
  return b2d_init(min_x, min_y, max_x, max_y, gravity_y, magnification)
             ? reinterpret_cast<void*>(1)
             : nullptr;
}

void b2d_world_destroy(void*) { b2d_dispose(); }

void* b2d_world_create_box(void*, float width, float height, float x, float y,
                           float rotation, float density, float friction,
                           float restitution) {
  return b2d_create_box(width, height, x, y, rotation, density, friction,
                        restitution);
}

void* b2d_world_create_triangle(void*, float width, float height, float x,
                                float y, float rotation, float density,
                                float friction, float restitution) {
  return b2d_create_triangle(width, height, x, y, rotation, density, friction,
                             restitution);
}

void* b2d_world_create_circle(void*, float radius, float x, float y,
                              float density, float friction,
                              float restitution) {
  return b2d_create_circle(radius, x, y, density, friction, restitution);
}

void b2d_world_destroy_body(void*, void* body) { b2d_destroy_body(body); }
void b2d_world_step(void*, float dt, int iterations) {
  b2d_step(dt, iterations);
}
int b2d_world_get_contact(void*, void** first, void** second) {
  return b2d_get_contact(first, second);
}
float b2d_world_get_x(void*, void* body) { return b2d_get_x(body); }
float b2d_world_get_y(void*, void* body) { return b2d_get_y(body); }
float b2d_world_get_r(void*, void* body) { return b2d_get_r(body); }
void b2d_world_get_v(void*, void* body, float* x, float* y) {
  b2d_get_v(body, x, y);
}
void b2d_world_set_v(void*, void* body, float x, float y) {
  b2d_set_v(body, x, y);
}
void b2d_world_set_user_data(void*, void* body, void* data) {
  b2d_set_user_data(body, data);
}
void b2d_world_set_position(void*, void* body, float x, float y,
                            float rotation) {
  b2d_set_position(body, x, y, rotation);
}
}
