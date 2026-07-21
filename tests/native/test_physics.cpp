#include "irisu/physics.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

void require(bool condition, std::string_view message) {
  if (!condition) throw std::runtime_error(std::string(message));
}

void close(double actual, double expected, double tolerance, std::string_view message) {
  if (std::abs(actual - expected) > tolerance) {
    throw std::runtime_error(std::string(message) + ": expected " +
                             std::to_string(expected) + ", got " +
                             std::to_string(actual));
  }
}

irisu::MechanicsConfig isolated(double gravity = 0.0) {
  irisu::MechanicsConfig config;
  config.gravity_y = gravity;
  config.tick_seconds = 0.020;
  config.solver_iterations = 10;
  config.world_magnification = 100.0;
  config.world_min_x = -1000.0;
  config.world_max_x = 1000.0;
  config.world_min_y = -1000.0;
  config.world_max_y = 1000.0;
  config.field_x = -500.0;
  config.field_width = 1000.0;
  config.field_y = 0.0;
  config.field_height = 900.0;
  config.field_blank = 0.0;
  config.field_thickness = 20.0;
  config.side_wall_top = 0.0;
  config.side_wall_bottom = 0.0;
  config.cleanup_margin_x = 0.0;
  config.cleanup_margin_y = 0.0;
  config.scripted_fall_speed = 0.0;
  config.linear_damping = 0.0;
  config.angular_damping = 0.0;
  return config;
}

irisu::Body make_body(irisu::BodyId id, irisu::Shape shape, irisu::Vec2 position,
                      double size, double density = 1.0, double friction = 1.0,
                      double restitution = 0.0) {
  irisu::Body body;
  body.id = id;
  body.shape = shape;
  body.lifecycle = irisu::Lifecycle::DynamicFresh;
  body.position = position;
  body.size = size;
  body.density = density;
  body.friction = friction;
  body.restitution = restitution;
  return body;
}

void velocity_units_and_integration() {
  auto config = isolated();
  irisu::PhysicsWorld world(config);
  std::vector bodies{make_body(101, irisu::Shape::Box, {123.0, 234.0}, 40.0,
                               1.0, 0.7, 0.2)};
  bodies[0].angle = 0.25;
  bodies[0].velocity = {2.5, -5.0};
  world.initialize_mass(bodies[0]);
  world.rebuild(bodies);

  close(world.raw_velocity(101).x, 2.5, 1e-6, "b2d_set_v X scaling");
  close(world.raw_velocity(101).y, -5.0, 1e-6, "b2d_set_v Y scaling");
  world.step(bodies);
  close(bodies[0].position.x, 128.0, 2e-4, "0.020 step X displacement");
  close(bodies[0].position.y, 224.0, 2e-4, "0.020 step Y displacement");
  close(world.raw_velocity(101).x, 2.5, 1e-6, "b2d_get_v raw X");
  close(world.raw_velocity(101).y, -5.0, 1e-6, "b2d_get_v raw Y");

  bodies[0].position = {321.0, 222.0};
  bodies[0].angle = -0.5;
  world.step(bodies);
  close(bodies[0].position.x, 321.0, 2e-4, "set_position X");
  close(bodies[0].position.y, 222.0, 2e-4, "set_position Y");
  close(world.raw_velocity(101).x, 0.0, 1e-7, "set_position clears vx");
  close(world.raw_velocity(101).y, 0.0, 1e-7, "set_position clears vy");
}

void gravity_scaling() {
  auto config = isolated(100.0);
  irisu::PhysicsWorld world(config);
  std::vector bodies{make_body(201, irisu::Shape::Box, {0.0, 100.0}, 20.0)};
  world.initialize_mass(bodies[0]);
  world.step(bodies);
  close(world.raw_velocity(201).y, 0.02, 1e-6, "gravity is divided by magnification");
  close(bodies[0].position.y, 100.04, 2e-5, "semi-implicit gravity displacement");
}

void dimension_skin_and_contact_ticks() {
  auto config = isolated(100.0);
  config.field_y = 0.0;
  config.field_height = 290.0;
  irisu::PhysicsWorld world(config);
  std::vector<irisu::Body> bodies{
      make_body(801, irisu::Shape::Box, {-300.0, 100.0}, 20.0),
      make_body(802, irisu::Shape::Box, {-100.0, 100.0}, 40.0),
      make_body(803, irisu::Shape::Circle, {100.0, 100.0}, 20.0),
      make_body(804, irisu::Shape::Circle, {300.0, 100.0}, 40.0),
  };
  for (auto& body : bodies) world.initialize_mass(body);

  std::vector<int> first_contact(4, 0);
  for (int tick = 1; tick <= 200; ++tick) {
    const auto contacts = world.step(bodies);
    for (const auto& contact : contacts) {
      const auto id = contact.a == 0 ? contact.b : contact.a;
      if (contact.a != 0 && contact.b != 0) continue;
      if (id < 801 || id > 804) continue;
      auto& first = first_contact[id - 801];
      if (first == 0) first = tick;
    }
  }

  require(first_contact == std::vector<int>({96, 93, 96, 93}),
          "dimension contact ticks must match DLL probe (got " +
              std::to_string(first_contact[0]) + "," +
              std::to_string(first_contact[1]) + "," +
              std::to_string(first_contact[2]) + "," +
              std::to_string(first_contact[3]) + ")");
  close(bodies[0].position.y, 280.500122, 5e-4, "height-20 box rest Y");
  close(bodies[1].position.y, 270.499969, 5e-4, "height-40 box rest Y");
  close(bodies[2].position.y, 280.500061, 5e-4, "radius-10 circle rest Y");
  close(bodies[3].position.y, 270.500031, 5e-4, "radius-20 circle rest Y");
}

void restitution_uses_maximum() {
  auto config = isolated();
  config.field_x = -700.0;
  // The exact right-wall constructor adds one full thickness to field_x +
  // field_width. Choose width 980 so the width-20 wall is centered at X=300,
  // matching the DLL probe's inner face at X=290.
  config.field_width = 980.0;
  config.side_wall_top = -250.0;
  config.side_wall_bottom = 250.0;
  irisu::PhysicsWorld world(config);
  std::vector bodies{make_body(301, irisu::Shape::Circle, {100.0, 0.0}, 24.0,
                               1.0, 0.0, 0.5)};
  bodies[0].velocity = {5.0, 0.0};
  world.initialize_mass(bodies[0]);

  int first_wall_contact = 0;
  for (int tick = 1; tick <= 20; ++tick) {
    const auto contacts = world.step(bodies);
    for (const auto& contact : contacts) {
      if ((contact.a == 0 || contact.b == 0) &&
          contact.boundary == irisu::BoundaryKind::RightWall &&
          first_wall_contact == 0) {
        first_wall_contact = tick;
      }
    }
  }
  require(first_wall_contact == 19,
          "wall contact tick must match DLL probe (got " +
              std::to_string(first_wall_contact) + ")");
  close(world.raw_velocity(301).x, -5.0, 1e-5,
        "max(1.0, 0.5) restitution must rebound at full speed");
}

void friction_uses_geometric_mean() {
  auto config = isolated(100.0);
  irisu::PhysicsWorld world(config);
  std::vector<irisu::Body> bodies;
  const double centers[] = {-600.0, -200.0, 200.0, 600.0};
  const double floor_friction[] = {1.0, 1.0, 1.0, 0.25};
  const double body_friction[] = {0.0, 0.25, 1.0, 1.0};
  for (int index = 0; index < 4; ++index) {
    bodies.push_back(make_body(9101 + index, irisu::Shape::Box,
                               {centers[index], 380.0}, 180.0, 0.0,
                               floor_friction[index]));
    bodies.push_back(make_body(401 + index, irisu::Shape::Box,
                               {centers[index] - 40.0, 280.5}, 20.0, 1.0,
                               body_friction[index]));
    bodies.back().velocity = {1.0, 0.0};
  }
  for (auto& body : bodies) world.initialize_mass(body);
  for (int tick = 0; tick < 20; ++tick) world.step(bodies);

  close(world.raw_velocity(401).x, 1.0, 2e-6, "zero mixed friction");
  close(world.raw_velocity(402).x, 0.8, 2e-6, "sqrt(1*0.25) friction");
  close(world.raw_velocity(403).x, 0.6, 2e-6, "unit mixed friction");
  close(world.raw_velocity(404).x, 0.8, 2e-6, "sqrt(0.25*1) friction");
  require(world.raw_velocity(402).x == world.raw_velocity(404).x,
          "friction mixing must be symmetric and bit-identical");
}

void sleep_boundary_and_rebuild() {
  auto config = isolated(100.0);
  irisu::PhysicsWorld world(config);
  std::vector<irisu::Body> bodies{
      make_body(9201, irisu::Shape::Box, {0.0, 380.0}, 180.0, 0.0, 0.0),
      make_body(503, irisu::Shape::Box, {0.0, 280.5}, 20.0, 1.0, 0.0),
  };
  for (auto& body : bodies) world.initialize_mass(body);
  for (int tick = 1; tick <= 24; ++tick) world.step(bodies);
  require(!bodies[1].sleeping, "body must remain awake through tick 24");
  world.step(bodies);
  require(bodies[1].sleeping, "body must sleep at exactly 0.5 seconds/tick 25");
  close(bodies[1].sleep_time, 0.5, 2e-6, "sleep timer snapshot value");

  const double asleep_x = bodies[1].position.x;
  bodies[1].velocity = {1.0, 0.0};
  bodies[1].native_velocity = {1.0, 0.0};
  world.step(bodies);
  close(bodies[1].position.x, asleep_x, 1e-7, "set_v must not wake sleeping body");
  close(world.raw_velocity(503).x, 1.0, 1e-6, "sleeping body retains set_v value");
  require(bodies[1].sleeping, "set_v must preserve the sleep flag");

  irisu::PhysicsWorld restored(config);
  restored.rebuild(bodies);
  restored.step(bodies);
  close(bodies[1].position.x, asleep_x, 1e-7, "sleep state survives native rebuild");
  close(restored.raw_velocity(503).x, 1.0, 1e-6,
        "sleeping velocity survives native rebuild");
}

void triangle_geometry_and_lifecycle() {
  auto config = isolated();
  irisu::PhysicsWorld world(config);
  std::vector<irisu::Body> triangles{
      make_body(600, irisu::Shape::Triangle, {0.0, 300.0}, 100.0, 0.0),
      make_body(601, irisu::Shape::Circle, {-35.0, 300.0}, 4.0),
      make_body(602, irisu::Shape::Circle, {0.0, 325.0}, 4.0),
      make_body(603, irisu::Shape::Circle, {35.0, 300.0}, 4.0),
      make_body(604, irisu::Shape::Circle, {0.0, 275.0}, 4.0),
  };
  for (auto& body : triangles) world.initialize_mass(body);
  const auto contacts = world.step(triangles);
  std::vector<irisu::BodyId> touching;
  for (const auto& contact : contacts) {
    if (contact.a == 600 && contact.b >= 601 && contact.b <= 604) {
      touching.push_back(contact.b);
    }
  }
  std::sort(touching.begin(), touching.end());
  require(touching == std::vector<irisu::BodyId>({601, 602}),
          "triangle must use the measured lower-left right-triangle fixture (count=" +
              std::to_string(touching.size()) + ")");

  config.gravity_y = 100.0;
  config.scripted_fall_speed = 10.0;
  irisu::PhysicsWorld lifecycle(config);
  auto piece = make_body(701, irisu::Shape::Circle, {-300.0, 100.0}, 20.0);
  piece.lifecycle = irisu::Lifecycle::ScriptedFalling;
  lifecycle.initialize_mass(piece);
  std::vector pieces{piece};
  lifecycle.step(pieces);
  const auto initial_ordering = lifecycle.ordering();
  close(pieces[0].position.y, 100.04, 3e-5,
        "scripted fixture participates in gravity");
  require(pieces[0].inverse_mass > 0.0,
          "scripted fixture is dynamic from creation");
  pieces[0].lifecycle = irisu::Lifecycle::DynamicFresh;
  lifecycle.step(pieces);
  close(pieces[0].position.y, 100.12, 4e-5,
        "activation does not restart physics integration");
  const auto activated_ordering = lifecycle.ordering();
  require(initial_ordering.proxy_ids == activated_ordering.proxy_ids &&
              initial_ordering.proxy_order == activated_ordering.proxy_order,
          "activation must not recreate the fixture proxy");
  pieces[0].lifecycle = irisu::Lifecycle::Deleted;
  lifecycle.step(pieces);
  bool removed = false;
  try {
    (void)lifecycle.raw_velocity(701);
  } catch (const std::out_of_range&) {
    removed = true;
  }
  require(removed, "deleted bodies must leave the native world");
}

void simultaneous_boundary_contacts_keep_identity() {
  auto config = isolated();
  config.field_x = 0.0;
  config.field_y = 0.0;
  config.field_width = 100.0;
  config.field_height = 100.0;
  config.field_thickness = 20.0;
  config.side_wall_top = 0.0;
  config.side_wall_bottom = 100.0;
  irisu::PhysicsWorld world(config);
  std::vector bodies{make_body(777, irisu::Shape::Circle, {9.0, 91.0}, 20.0)};
  world.initialize_mass(bodies[0]);
  const auto contacts = world.step(bodies);
  std::vector<irisu::BoundaryKind> boundaries;
  for (const auto& contact : contacts) {
    if ((contact.a == 777 && contact.b == 0) ||
        (contact.a == 0 && contact.b == 777)) {
      boundaries.push_back(contact.boundary);
    }
  }
  std::sort(boundaries.begin(), boundaries.end());
  require(boundaries == std::vector<irisu::BoundaryKind>{
                            irisu::BoundaryKind::Floor,
                            irisu::BoundaryKind::LeftWall},
          "corner contacts must preserve distinct floor and wall identities");
}

void deferred_destroy_keeps_proxy_until_next_step() {
  const auto config = isolated();
  irisu::PhysicsWorld world(config);
  std::vector bodies{
      make_body(1001, irisu::Shape::Circle, {-300.0, 300.0}, 20.0),
  };
  world.initialize_mass(bodies[0]);
  world.step(bodies);

  bodies[0].lifecycle = irisu::Lifecycle::Deleted;
  bodies[0].pending_delete = true;
  world.synchronize(bodies);
  bodies.push_back(
      make_body(1002, irisu::Shape::Circle, {-200.0, 300.0}, 20.0));
  world.initialize_mass(bodies.back());
  world.synchronize(bodies);
  auto ordering = world.ordering();
  const auto new_proxy = std::find(ordering.proxy_order.begin(),
                                   ordering.proxy_order.end(), 1002);
  require(new_proxy != ordering.proxy_order.end(),
          "new body is missing from broad phase");
  require(ordering.proxy_ids[static_cast<std::size_t>(
              new_proxy - ordering.proxy_order.begin())] == 5,
          "spawn before next Step must not reuse queued proxy 4");

  world.step(bodies);
  require(!bodies[0].pending_delete,
          "native cleanup must release the actor tombstone");
  bodies.push_back(
      make_body(1003, irisu::Shape::Circle, {-100.0, 300.0}, 20.0));
  world.initialize_mass(bodies.back());
  world.synchronize(bodies);
  ordering = world.ordering();
  const auto reused = std::find(ordering.proxy_order.begin(),
                                ordering.proxy_order.end(), 1003);
  require(reused != ordering.proxy_order.end() &&
              ordering.proxy_ids[static_cast<std::size_t>(
                  reused - ordering.proxy_order.begin())] == 4,
          "proxy 4 must become reusable only after next Step cleanup");
}

void teleport_preserves_angular_and_sleep_state() {
  const auto config = isolated();
  irisu::PhysicsWorld world(config);
  std::vector bodies{
      make_body(1101, irisu::Shape::Circle, {-300.0, 300.0}, 20.0),
  };
  bodies[0].angular_velocity = 1.0;
  world.initialize_mass(bodies[0]);
  world.step(bodies);
  const double angle_before_sleep = bodies[0].angle;

  bodies[0].position = {-250.0, 300.0};
  bodies[0].sleeping = true;
  bodies[0].sleep_time = 0.5;
  world.step(bodies);
  close(bodies[0].position.x, -250.0, 1e-6,
        "teleport applies the actor transform");
  close(bodies[0].velocity.x, 0.0, 0.0,
        "teleport clears only linear velocity");
  close(bodies[0].angular_velocity, 1.0, 0.0,
        "teleport preserves angular velocity");
  close(bodies[0].angle, angle_before_sleep, 1e-7,
        "sleep flag survives teleport");
  close(bodies[0].sleep_time, 0.5, 2e-6,
        "sleep timer survives teleport");

  bodies[0].sleeping = false;
  world.step(bodies);
  close(bodies[0].angle, angle_before_sleep + config.tick_seconds, 2e-6,
        "preserved angular velocity resumes after wake");
}

}  // namespace

int main() {
  try {
    velocity_units_and_integration();
    gravity_scaling();
    dimension_skin_and_contact_ticks();
    restitution_uses_maximum();
    friction_uses_geometric_mean();
    sleep_boundary_and_rebuild();
    triangle_geometry_and_lifecycle();
    simultaneous_boundary_contacts_keep_identity();
    deferred_destroy_keeps_proxy_until_next_step();
    teleport_preserves_angular_and_sleep_state();
    std::cout << "native legacy physics differential tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "physics differential test failure: " << error.what() << '\n';
    return 1;
  }
}
