#include <cstring>

#include <Box2D.h>

#include <bit>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace {

std::uint32_t bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

b2Body* box(b2World& world, float width, float height, float x, float y,
            float angle, float density, float friction, float restitution) {
  b2BodyDef body_definition;
  body_definition.position.Set(x / 10.0f, y / 10.0f);
  body_definition.rotation = angle;
  b2BoxDef shape_definition;
  shape_definition.extents.Set(width / 20.0f, height / 20.0f);
  shape_definition.density = density;
  shape_definition.friction = friction;
  shape_definition.restitution = restitution;
  body_definition.AddShape(&shape_definition);
  return world.CreateBody(&body_definition);
}

}  // namespace

int main() {
  const std::uint16_t control_word = 0x027fU;
  __asm__ __volatile__("fldcw %0" : : "m"(control_word));
  b2AABB bounds;
  bounds.minVertex.Set(0.0f, -20.0f);
  bounds.maxVertex.Set(64.0f, 48.0f);
  b2World world(bounds, b2Vec2(0.0f, 16.0f), true);
  box(world, 16.0f, 250.0f, 138.0f, 245.0f, 0.0f, 0.0f, 1.0f,
      1.0f);
  box(world, 16.0f, 250.0f, 466.0f, 245.0f, 0.0f, 0.0f, 1.0f,
      1.0f);
  box(world, 352.0f, 16.0f, 306.0f, 418.0f, 0.0f, 0.0f, 1.0f,
      0.0f);
  box(world, 320.0f, 300.0f, 306.0f, -140.0f, 0.0f, 0.0f, 1.0f,
      0.5f);
  b2Body* projectile = box(world, 24.0f, 24.0f, 241.0f, 237.0f,
                           0.0f, 8.0f, 1.0f, 0.0f);
  projectile->SetLinearVelocity(b2Vec2(0.0f, -50.0f));
  for (int step = 1; step <= 25; ++step) {
    world.Step(0.02f, 10);
    if (step < 23) continue;
    std::cout << std::dec << step << std::hex << std::setfill('0')
              << " px=" << std::setw(8) << bits(projectile->m_position.x)
              << " py=" << std::setw(8) << bits(projectile->m_position.y)
              << " r=" << std::setw(8) << bits(projectile->m_rotation)
              << " vx=" << std::setw(8)
              << bits(projectile->m_linearVelocity.x)
              << " vy=" << std::setw(8)
              << bits(projectile->m_linearVelocity.y)
              << " w=" << std::setw(8)
              << bits(projectile->m_angularVelocity) << '\n';
  }
}
