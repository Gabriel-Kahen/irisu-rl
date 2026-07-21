#pragma once

#include "irisu/config.hpp"
#include "irisu/types.hpp"

#include <memory>
#include <vector>

namespace irisu {

class PhysicsWorld {
 public:
  explicit PhysicsWorld(MechanicsConfig config);
  ~PhysicsWorld();

  PhysicsWorld(const PhysicsWorld&) = delete;
  PhysicsWorld& operator=(const PhysicsWorld&) = delete;
  PhysicsWorld(PhysicsWorld&&) noexcept;
  PhysicsWorld& operator=(PhysicsWorld&&) noexcept;

  void initialize_mass(Body& body) const;
  void reset();
  void synchronize(std::vector<Body>& bodies);
  void queue_destroy(BodyId id);

  // Restores fixtures, transforms, velocities, r58 sleep flags/timers, and the
  // accumulated manifold impulses needed for exact mid-contact branching.
  void rebuild(std::vector<Body>& bodies,
               const std::vector<ContactImpulse>& contact_impulses = {},
               const PhysicsOrdering& ordering = {});
  [[nodiscard]] std::vector<ContactImpulse> contact_impulses(
      const std::vector<Body>& bodies) const;
  [[nodiscard]] PhysicsOrdering ordering() const;
  std::vector<Contact> step(std::vector<Body>& bodies);

  // Matches b2d_get_v: raw Box2D world units, without magnification.
  [[nodiscard]] Vec2 raw_velocity(BodyId id) const;

 private:
  class Impl;

  MechanicsConfig config_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace irisu
