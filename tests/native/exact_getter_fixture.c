#include <stddef.h>
#include <stdlib.h>

struct Body {
  float x;
  float y;
  float r;
  float vx;
  float vy;
  void *user;
};

struct World {
  float scale;
  struct Body *body;
};

void *b2d_world_create(float min_x, float min_y, float max_x, float max_y,
                       float gravity_y, float scale) {
  (void)min_x;
  (void)min_y;
  (void)max_x;
  (void)max_y;
  (void)gravity_y;
  struct World *world = calloc(1, sizeof(*world));
  if (world != NULL)
    world->scale = scale;
  return world;
}

void b2d_world_destroy(void *opaque) {
  struct World *world = opaque;
  if (world != NULL)
    free(world->body);
  free(world);
}

void *b2d_world_create_box(void *opaque, float width, float height, float x,
                           float y, float r, float density, float friction,
                           float restitution) {
  (void)width;
  (void)height;
  (void)density;
  (void)friction;
  (void)restitution;
  struct World *world = opaque;
  world->body = calloc(1, sizeof(*world->body));
  if (world->body != NULL) {
    world->body->x = x;
    world->body->y = y;
    world->body->r = r;
  }
  return world->body;
}

void *b2d_world_create_triangle(void *world, float width, float height, float x,
                                float y, float r, float density, float friction,
                                float restitution) {
  return b2d_world_create_box(world, width, height, x, y, r, density, friction,
                              restitution);
}

void *b2d_world_create_circle(void *world, float radius, float x, float y,
                              float density, float friction,
                              float restitution) {
  return b2d_world_create_box(world, radius, radius, x, y, 0.0f, density,
                              friction, restitution);
}

void b2d_world_destroy_body(void *opaque, void *body) {
  struct World *world = opaque;
  if (world->body == body) {
    free(world->body);
    world->body = NULL;
  }
}

void b2d_world_step(void *opaque, float dt, int iterations) {
  (void)iterations;
  struct World *world = opaque;
  if (world->body != NULL) {
    world->body->x += world->body->vx * dt * world->scale;
    world->body->y += world->body->vy * dt * world->scale;
  }
}

int b2d_world_get_contact(void *world, void **a, void **b) {
  (void)world;
  *a = NULL;
  *b = NULL;
  return 0;
}

float b2d_world_get_x(void *world, void *body) {
  (void)world;
  return ((struct Body *)body)->x;
}

float b2d_world_get_y(void *world, void *body) {
  (void)world;
  return ((struct Body *)body)->y;
}

float b2d_world_get_r(void *world, void *body) {
  (void)world;
  return ((struct Body *)body)->r;
}

void b2d_world_get_v(void *world, void *body, float *x, float *y) {
  (void)world;
  *x = ((struct Body *)body)->vx;
  *y = ((struct Body *)body)->vy;
}

void b2d_world_set_v(void *opaque, void *body, float x, float y) {
  const struct World *world = opaque;
  ((struct Body *)body)->vx = x / world->scale;
  ((struct Body *)body)->vy = y / world->scale;
}

void b2d_world_set_user_data(void *world, void *body, void *user) {
  (void)world;
  ((struct Body *)body)->user = user;
}

void b2d_world_set_position(void *world, void *body, float x, float y,
                            float r) {
  (void)world;
  struct Body *value = body;
  value->x = x;
  value->y = y;
  value->r = r;
  value->vx = 0.0f;
  value->vy = 0.0f;
}
