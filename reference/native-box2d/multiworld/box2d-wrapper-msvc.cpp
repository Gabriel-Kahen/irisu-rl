#include <Box2D.h>
#include "Source/Collision/b2Shape.h"
#include "Source/Dynamics/b2World.h"
#include "Source/Dynamics/Contacts/b2Contact.h"

#define API extern "C" __declspec(dllexport)

struct b2dWorldHandle {
  b2World* world;
  b2Contact* contact;
  float magnification;
};

static bool owns_body(b2dWorldHandle* handle, b2Body* body) {
  return handle && handle->world && body && body->m_world == handle->world;
}

API void* __stdcall b2d_world_create(float min_x, float min_y, float max_x,
                                      float max_y, float gravity_y,
                                      float magnification) {
  b2AABB bounds;
  bounds.minVertex.Set(min_x / magnification, min_y / magnification);
  bounds.maxVertex.Set(max_x / magnification, max_y / magnification);
  b2Vec2 gravity(0.0f, gravity_y / magnification);
  b2dWorldHandle* handle = new b2dWorldHandle;
  handle->world = new b2World(bounds, gravity, true);
  handle->contact = 0;
  handle->magnification = magnification;
  return handle;
}

API void __stdcall b2d_world_destroy(void* opaque) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  if (!handle) return;
  delete handle->world;
  delete handle;
}

API void* __stdcall b2d_world_create_box(
    void* opaque, float width, float height, float x, float y, float rotation,
    float density, float friction, float restitution) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  if (!handle || !handle->world) return 0;
  b2PolyDef shape;
  const float half_width = width / 2.0f / handle->magnification;
  const float half_height = height / 2.0f / handle->magnification;
  shape.vertexCount = 4;
  shape.vertices[0].Set(-half_width, half_height);
  shape.vertices[1].Set(-half_width, -half_height);
  shape.vertices[2].Set(half_width, -half_height);
  shape.vertices[3].Set(half_width, half_height);
  shape.density = density;
  shape.friction = friction;
  shape.restitution = restitution;
  b2BodyDef body;
  body.position.Set(x / handle->magnification, y / handle->magnification);
  body.rotation = rotation;
  body.AddShape(&shape);
  return handle->world->CreateBody(&body);
}

API void* __stdcall b2d_world_create_triangle(
    void* opaque, float width, float height, float x, float y, float rotation,
    float density, float friction, float restitution) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  if (!handle || !handle->world) return 0;
  b2PolyDef shape;
  const float half_width = width / 2.0f / handle->magnification;
  const float half_height = height / 2.0f / handle->magnification;
  shape.vertexCount = 3;
  shape.vertices[0].Set(-half_width, half_height);
  shape.vertices[1].Set(-half_width, -half_height);
  shape.vertices[2].Set(half_width, half_height);
  shape.density = density;
  shape.friction = friction;
  shape.restitution = restitution;
  b2BodyDef body;
  body.position.Set(x / handle->magnification, y / handle->magnification);
  body.rotation = rotation;
  body.AddShape(&shape);
  return handle->world->CreateBody(&body);
}

API void* __stdcall b2d_world_create_circle(
    void* opaque, float radius, float x, float y, float density,
    float friction, float restitution) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  if (!handle || !handle->world) return 0;
  b2CircleDef shape;
  shape.radius = radius / handle->magnification;
  shape.density = density;
  shape.friction = friction;
  shape.restitution = restitution;
  b2BodyDef body;
  body.position.Set(x / handle->magnification, y / handle->magnification);
  body.rotation = 0.0f;
  body.AddShape(&shape);
  return handle->world->CreateBody(&body);
}

API void __stdcall b2d_world_destroy_body(void* opaque, void* body) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  b2Body* native = static_cast<b2Body*>(body);
  if (owns_body(handle, native)) handle->world->DestroyBody(native);
}

API void __stdcall b2d_world_step(void* opaque, float dt, int iterations) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  if (!handle || !handle->world) return;
  handle->world->Step(dt, iterations);
  handle->contact = handle->world->GetContactList();
}

API int __stdcall b2d_world_get_contact(void* opaque, void** first,
                                         void** second) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  if (first) *first = 0;
  if (second) *second = 0;
  if (!handle || !first || !second) return 0;
  while (handle->contact) {
    if (handle->contact->GetManifoldCount() > 0) break;
    handle->contact = handle->contact->GetNext();
  }
  if (!handle->contact) return 0;
  *first = handle->contact->GetShape1()->GetBody()->GetUserData();
  *second = handle->contact->GetShape2()->GetBody()->GetUserData();
  handle->contact = handle->contact->GetNext();
  return 1;
}

API float __stdcall b2d_world_get_x(void* opaque, void* body) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  b2Body* native = static_cast<b2Body*>(body);
  return owns_body(handle, native)
             ? native->GetOriginPosition().x * handle->magnification
             : 0.0f;
}

API float __stdcall b2d_world_get_y(void* opaque, void* body) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  b2Body* native = static_cast<b2Body*>(body);
  return owns_body(handle, native)
             ? native->GetOriginPosition().y * handle->magnification
             : 0.0f;
}

API float __stdcall b2d_world_get_r(void* opaque, void* body) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  b2Body* native = static_cast<b2Body*>(body);
  return owns_body(handle, native) ? native->GetRotation() : 0.0f;
}

API void __stdcall b2d_world_get_v(void* opaque, void* body, float* x,
                                    float* y) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  b2Body* native = static_cast<b2Body*>(body);
  if (x) *x = 0.0f;
  if (y) *y = 0.0f;
  if (!owns_body(handle, native) || !x || !y) return;
  const b2Vec2 velocity = native->GetLinearVelocity();
  *x = velocity.x;
  *y = velocity.y;
}

API void __stdcall b2d_world_set_v(void* opaque, void* body, float x,
                                    float y) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  b2Body* native = static_cast<b2Body*>(body);
  if (!owns_body(handle, native)) return;
  native->SetLinearVelocity(
      b2Vec2(x / handle->magnification, y / handle->magnification));
}

API void __stdcall b2d_world_set_user_data(void* opaque, void* body,
                                            void* user_data) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  b2Body* native = static_cast<b2Body*>(body);
  if (owns_body(handle, native)) native->m_userData = user_data;
}

API void __stdcall b2d_world_set_position(void* opaque, void* body, float x,
                                           float y, float rotation) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  b2Body* native = static_cast<b2Body*>(body);
  if (!owns_body(handle, native)) return;
  native->SetOriginPosition(
      b2Vec2(x / handle->magnification, y / handle->magnification), rotation);
  native->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
}

API void __stdcall b2d_world_test(void* opaque, void* body) {
  b2dWorldHandle* handle = static_cast<b2dWorldHandle*>(opaque);
  b2Body* native = static_cast<b2Body*>(body);
  if (owns_body(handle, native)) native->SetAngularVelocity(3.14159265f);
}
