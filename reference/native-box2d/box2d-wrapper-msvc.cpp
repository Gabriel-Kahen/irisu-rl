#include <Box2D.h>
#include "Source/Collision/b2Shape.h"
#include "Source/Dynamics/b2World.h"
#include "Source/Dynamics/Contacts/b2Contact.h"

#define API extern "C" __declspec(dllexport)

static b2World* g_world;
static b2Contact* g_contact;
static float g_magnification;

API int __stdcall b2d_init(float min_x, float min_y, float max_x,
                           float max_y, float gravity_y,
                           float magnification) {
  b2AABB bounds;
  bounds.minVertex.Set(min_x / magnification, min_y / magnification);
  bounds.maxVertex.Set(max_x / magnification, max_y / magnification);
  b2Vec2 gravity(0.0f, gravity_y / magnification);
  g_world = new b2World(bounds, gravity, true);
  g_contact = 0;
  g_magnification = magnification;
  return 1;
}

API void __stdcall b2d_dispose() {
  delete g_world;
}

API void* __stdcall b2d_create_box(float width, float height, float x,
                                    float y, float rotation, float density,
                                    float friction, float restitution) {
  b2PolyDef shape;
  float half_width = width / 2.0f / g_magnification;
  float half_height = height / 2.0f / g_magnification;
  shape.vertexCount = 4;
  shape.vertices[0].Set(-half_width, half_height);
  shape.vertices[1].Set(-half_width, -half_height);
  shape.vertices[2].Set(half_width, -half_height);
  shape.vertices[3].Set(half_width, half_height);
  shape.density = density;
  shape.friction = friction;
  shape.restitution = restitution;
  b2BodyDef body;
  body.position.Set(x / g_magnification, y / g_magnification);
  body.rotation = rotation;
  body.AddShape(&shape);
  return g_world->CreateBody(&body);
}

API void* __stdcall b2d_create_triangle(float width, float height, float x,
                                         float y, float rotation,
                                         float density, float friction,
                                         float restitution) {
  b2PolyDef shape;
  float half_width = width / 2.0f / g_magnification;
  float half_height = height / 2.0f / g_magnification;
  shape.vertexCount = 3;
  shape.vertices[0].Set(-half_width, half_height);
  shape.vertices[1].Set(-half_width, -half_height);
  shape.vertices[2].Set(half_width, half_height);
  shape.density = density;
  shape.friction = friction;
  shape.restitution = restitution;
  b2BodyDef body;
  body.position.Set(x / g_magnification, y / g_magnification);
  body.rotation = rotation;
  body.AddShape(&shape);
  return g_world->CreateBody(&body);
}

API void* __stdcall b2d_create_circle(float radius, float x, float y,
                                       float density, float friction,
                                       float restitution) {
  b2CircleDef shape;
  shape.radius = radius / g_magnification;
  shape.density = density;
  shape.friction = friction;
  shape.restitution = restitution;
  b2BodyDef body;
  body.position.Set(x / g_magnification, y / g_magnification);
  body.rotation = 0.0f;
  body.AddShape(&shape);
  return g_world->CreateBody(&body);
}

API void __stdcall b2d_destroy_body(void* body) {
  if (body) g_world->DestroyBody(static_cast<b2Body*>(body));
}

API void __stdcall b2d_step(float dt, int iterations) {
  g_world->Step(dt, iterations);
  g_contact = g_world->GetContactList();
}

API int __stdcall b2d_get_contact(void** first, void** second) {
  *first = 0;
  *second = 0;
  while (g_contact) {
    if (g_contact->GetManifoldCount() > 0) break;
    g_contact = g_contact->GetNext();
  }
  if (!g_contact) return 0;
  *first = g_contact->GetShape1()->GetBody()->GetUserData();
  *second = g_contact->GetShape2()->GetBody()->GetUserData();
  g_contact = g_contact->GetNext();
  return 1;
}

API float __stdcall b2d_get_x(void* body) {
  return static_cast<b2Body*>(body)->GetOriginPosition().x * g_magnification;
}

API float __stdcall b2d_get_y(void* body) {
  return static_cast<b2Body*>(body)->GetOriginPosition().y * g_magnification;
}

API float __stdcall b2d_get_r(void* body) {
  return static_cast<b2Body*>(body)->GetRotation();
}

API void __stdcall b2d_get_v(void* body, float* x, float* y) {
  const b2Vec2 velocity = static_cast<b2Body*>(body)->GetLinearVelocity();
  *x = velocity.x;
  *y = velocity.y;
}

API void __stdcall b2d_set_v(void* body, float x, float y) {
  static_cast<b2Body*>(body)->SetLinearVelocity(
      b2Vec2(x / g_magnification, y / g_magnification));
}

API void __stdcall b2d_set_user_data(void* body, void* user_data) {
  static_cast<b2Body*>(body)->m_userData = user_data;
}

API void __stdcall b2d_set_position(void* body, float x, float y,
                                     float rotation) {
  b2Body* native = static_cast<b2Body*>(body);
  native->SetOriginPosition(
      b2Vec2(x / g_magnification, y / g_magnification), rotation);
  native->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
}

API void __stdcall b2d_test(void* body) {
  static_cast<b2Body*>(body)->SetAngularVelocity(3.14159265f);
}
