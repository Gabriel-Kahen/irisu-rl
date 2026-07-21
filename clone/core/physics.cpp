#include "irisu/physics.hpp"

#include "irisu/floating_point.hpp"

#include <Box2D.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace irisu {
namespace {

bool same(double a, double b) {
  return a == b;
}

bool same(Vec2 a, Vec2 b) {
  return same(a.x, b.x) && same(a.y, b.y);
}

struct BodySignature {
  Shape shape{};
  double size{};
  double density{};
  double friction{};
  double restitution{};

  friend bool operator==(const BodySignature&, const BodySignature&) = default;
};

BodySignature signature(const Body& body) {
  return {body.shape, body.size, body.density, body.friction,
          body.restitution};
}

struct BodyMirror {
  Vec2 position{};
  double angle{};
  Vec2 native_velocity{};
  double native_angular_velocity{};
  bool sleeping{};
  double sleep_time{};
};

BodyMirror mirror(const Body& body) {
  return {body.position, body.angle, body.native_velocity,
          body.native_angular_velocity,
          body.sleeping, body.sleep_time};
}

void configure_material(b2ShapeDef& shape, const Body& body) {
  shape.density = static_cast<float32>(body.density);
  shape.friction = static_cast<float32>(body.friction);
  shape.restitution = static_cast<float32>(body.restitution);
}

void configure_box(b2PolyDef& shape, double width, double height,
                   double magnification) {
  const auto scale = static_cast<float32>(magnification);
  const auto half_width = static_cast<float32>(width) / 2.0f / scale;
  const auto half_height = static_cast<float32>(height) / 2.0f / scale;
  shape.vertexCount = 4;
  shape.vertices[0].Set(-half_width, half_height);
  shape.vertices[1].Set(-half_width, -half_height);
  shape.vertices[2].Set(half_width, -half_height);
  shape.vertices[3].Set(half_width, half_height);
}

void configure_triangle(b2PolyDef& shape, double size, double magnification) {
  const auto half = static_cast<float32>(size / (2.0 * magnification));
  shape.vertexCount = 3;
  // Exact wrapper geometry. This CCW ordering occupies the measured vertex set
  // (-w/2,-h/2), (-w/2,h/2), (w/2,h/2) in pixel coordinates.
  shape.vertices[0].Set(-half, half);
  shape.vertices[1].Set(-half, -half);
  shape.vertices[2].Set(half, half);
}

}  // namespace

class PhysicsWorld::Impl {
 public:
  explicit Impl(MechanicsConfig config) : config_(std::move(config)) { reset(); }

  void reset() {
    entries_.clear();
    reverse_.clear();
    floor_ = nullptr;
    left_wall_ = nullptr;
    right_wall_ = nullptr;
    top_ = nullptr;

    b2AABB bounds;
    bounds.minVertex.Set(to_world(config_.world_min_x), to_world(config_.world_min_y));
    bounds.maxVertex.Set(to_world(config_.world_max_x), to_world(config_.world_max_y));
    world_ = std::make_unique<b2World>(
        bounds, b2Vec2(0.0f, to_world(config_.gravity_y)), true);
    create_boundaries();
  }

  void rebuild(std::vector<Body>& bodies,
               const std::vector<ContactImpulse>& contact_impulses,
               const PhysicsOrdering& requested_ordering) {
    reset();
    std::map<BodyId, Body*> body_by_id;
    for (auto& body : bodies) {
      if (body.id == 0 || !body_by_id.emplace(body.id, &body).second) {
        throw std::invalid_argument("physics bodies require unique nonzero ids");
      }
    }
    PhysicsOrdering ordering = requested_ordering;
    if (ordering.body_order.empty() && ordering.proxy_order.empty() &&
        ordering.proxy_ids.empty() && ordering.free_proxy_order.empty() &&
        ordering.destroy_order.empty()) {
      reconcile(bodies);
      restore_contact_impulses(contact_impulses);
      return;
    }
    const auto unique_known = [&](const std::vector<BodyId>& ids) {
      return std::set<BodyId>(ids.begin(), ids.end()).size() == ids.size() &&
             std::all_of(ids.begin(), ids.end(), [&](BodyId id) {
               return body_by_id.contains(id);
             });
    };
    std::set<BodyId> expected_normal;
    std::set<BodyId> expected_destroy;
    for (const BodyId id : ordering.proxy_order) {
      const auto found = body_by_id.find(id);
      if (found == body_by_id.end()) continue;
      (found->second->lifecycle == Lifecycle::Deleted ? expected_destroy
                                                       : expected_normal)
          .insert(id);
    }
    const std::set<BodyId> normal_ids(ordering.body_order.begin(),
                                      ordering.body_order.end());
    const std::set<BodyId> destroy_ids(ordering.destroy_order.begin(),
                                       ordering.destroy_order.end());
    std::vector<std::uint32_t> allocated_proxy_ids;
    std::copy_if(ordering.proxy_ids.begin(), ordering.proxy_ids.end(),
                 std::back_inserter(allocated_proxy_ids), [](std::uint32_t id) {
                   return id != b2_nullProxy;
                 });
    std::vector<std::uint32_t> all_proxy_ids = allocated_proxy_ids;
    all_proxy_ids.insert(all_proxy_ids.end(), ordering.free_proxy_order.begin(),
                         ordering.free_proxy_order.end());
    std::sort(all_proxy_ids.begin(), all_proxy_ids.end());
    const bool proxy_partition_valid =
        ordering.proxy_ids.size() == ordering.proxy_order.size() &&
        ordering.proxy_order.size() == ordering.proxy_ids.size() &&
        ordering.free_proxy_order.size() + allocated_proxy_ids.size() ==
            MechanicsConfig::physics_proxy_capacity -
                MechanicsConfig::static_fixture_count &&
        all_proxy_ids.size() == MechanicsConfig::physics_proxy_capacity -
                                    MechanicsConfig::static_fixture_count &&
        all_proxy_ids.front() == MechanicsConfig::static_fixture_count &&
        all_proxy_ids.back() == MechanicsConfig::physics_proxy_capacity - 1U &&
        std::adjacent_find(all_proxy_ids.begin(), all_proxy_ids.end()) ==
            all_proxy_ids.end();
    if (!unique_known(ordering.body_order) ||
        !unique_known(ordering.destroy_order) ||
        !unique_known(ordering.proxy_order) || normal_ids != expected_normal ||
        destroy_ids != expected_destroy ||
        normal_ids.size() != ordering.body_order.size() ||
        destroy_ids.size() != ordering.destroy_order.size() ||
        !proxy_partition_valid ||
        !std::is_sorted(ordering.proxy_ids.begin(), ordering.proxy_ids.end())) {
      throw std::invalid_argument("physics ordering does not match live bodies");
    }
    std::map<std::uint32_t, BodyId> body_by_proxy;
    std::vector<BodyId> frozen_bodies;
    for (std::size_t index = 0; index < ordering.proxy_ids.size(); ++index) {
      if (ordering.proxy_ids[index] == b2_nullProxy) {
        frozen_bodies.push_back(ordering.proxy_order[index]);
      } else {
        body_by_proxy.emplace(ordering.proxy_ids[index],
                              ordering.proxy_order[index]);
      }
    }
    std::vector<b2Body*> dummy_bodies;
    const auto maximum_proxy = allocated_proxy_ids.empty()
                                   ? MechanicsConfig::static_fixture_count - 1U
                                   : allocated_proxy_ids.back();
    for (std::uint32_t proxy = MechanicsConfig::static_fixture_count;
         proxy <= maximum_proxy; ++proxy) {
      const auto body = body_by_proxy.find(proxy);
      if (body == body_by_proxy.end()) {
        dummy_bodies.push_back(create_proxy_placeholder(proxy));
      } else {
        create_body(*body_by_id.at(body->second));
        Entry& entry = entries_.at(body->second);
        if (entry.native->m_shapeList->m_proxyId == b2_nullProxy) {
          // A swept r58 proxy can remain live even when its current fixture
          // transform is outside the range accepted by CreateProxy().
          restore_out_of_range_proxy(entry, proxy);
        }
        const auto restored_proxy = entry.native->m_shapeList->m_proxyId;
        if (restored_proxy != proxy) {
          throw std::logic_error(
              "failed to restore broad-phase proxy id for body " +
              std::to_string(body->second) + ": expected " +
              std::to_string(proxy) + ", got " +
              std::to_string(restored_proxy));
        }
      }
    }
    for (b2Body* dummy : dummy_bodies) world_->DestroyBody(dummy);
    world_->CleanBodyList();
    for (const BodyId id : frozen_bodies) {
      create_body(*body_by_id.at(id));
      Entry& entry = entries_.at(id);
      if (entry.native->m_shapeList->m_proxyId != b2_nullProxy) {
        entry.native->Freeze();
      }
      if (entry.native->m_shapeList->m_proxyId != b2_nullProxy) {
        throw std::logic_error("failed to restore frozen body without proxy");
      }
    }
    restore_free_proxy_order(ordering.free_proxy_order);
    restore_broad_phase(ordering, contact_impulses);
    for (auto id = ordering.destroy_order.rbegin();
         id != ordering.destroy_order.rend(); ++id) {
      queue_destroy(entries_.at(*id));
    }
    restore_body_order(ordering.body_order);
    restore_contact_impulses(contact_impulses);
    restore_static_sleep_flags(ordering.static_sleep_flags);
  }

  void synchronize(std::vector<Body>& bodies) {
    reconcile(bodies);
  }

  PhysicsOrdering ordering() const {
    PhysicsOrdering result;
    for (b2Body* body = world_->GetBodyList(); body != nullptr;
         body = body->GetNext()) {
      const auto found = reverse_.find(body);
      if (found != reverse_.end()) result.body_order.push_back(found->second);
    }
    for (b2Body* body = world_->m_bodyDestroyList; body != nullptr;
         body = body->m_next) {
      const auto found = reverse_.find(body);
      if (found != reverse_.end()) result.destroy_order.push_back(found->second);
    }
    std::vector<std::pair<std::uint16_t, BodyId>> proxies;
    proxies.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) {
      if (entry.native->m_shapeList == nullptr) {
        throw std::logic_error("physics body has no fixture proxy");
      }
      proxies.emplace_back(entry.native->m_shapeList->m_proxyId, id);
    }
    std::sort(proxies.begin(), proxies.end());
    for (const auto& [proxy, id] : proxies) {
      result.proxy_order.push_back(id);
      result.proxy_ids.push_back(proxy);
    }
    std::uint16_t free_proxy = world_->m_broadPhase->m_freeProxy;
    while (free_proxy != b2_nullProxy) {
      if (result.free_proxy_order.size() >= b2_maxProxies) {
        throw std::logic_error("broad-phase free proxy list contains a cycle");
      }
      result.free_proxy_order.push_back(free_proxy);
      free_proxy = world_->m_broadPhase->m_proxyPool[free_proxy].GetNext();
    }
    const b2Body* static_bodies[] = {left_wall_, right_wall_, floor_, top_};
    for (std::size_t index = 0; index < std::size(static_bodies); ++index) {
      if ((static_bodies[index]->m_flags & b2Body::e_sleepFlag) != 0) {
        result.static_sleep_flags |= static_cast<std::uint8_t>(1U << index);
      }
    }
    const b2BroadPhase* broad_phase = world_->m_broadPhase;
    if (broad_phase->m_queryResultCount != 0 ||
        broad_phase->m_pairManager.m_pairBufferCount != 0) {
      throw std::logic_error(
          "snapshot requested with uncommitted broad-phase operations");
    }
    result.broadphase_time_stamp = broad_phase->m_timeStamp;
    result.proxy_time_stamps.reserve(b2_maxProxies);
    result.proxy_overlap_counts.reserve(b2_maxProxies);
    for (std::uint32_t proxy = 0; proxy < b2_maxProxies; ++proxy) {
      result.proxy_time_stamps.push_back(
          broad_phase->m_proxyPool[proxy].timeStamp);
      result.proxy_overlap_counts.push_back(
          broad_phase->m_proxyPool[proxy].overlapCount);
    }
    const auto bound_count = static_cast<std::size_t>(
        2 * broad_phase->m_proxyCount);
    result.broadphase_bounds.reserve(2 * bound_count);
    for (int axis = 0; axis < 2; ++axis) {
      for (std::size_t index = 0; index < bound_count; ++index) {
        const b2Bound& bound = broad_phase->m_bounds[axis][index];
        result.broadphase_bounds.push_back(
            {bound.value, bound.proxyId, bound.stabbingCount});
      }
    }
    return result;
  }

  std::vector<ContactImpulse> contact_impulses(
      const std::vector<Body>& bodies) const {
    std::map<BodyId, const Body*> current;
    for (const auto& body : bodies) current.emplace(body.id, &body);
    const auto cacheable = [&](BodyId id) {
      const auto entry = entries_.find(id);
      const auto body = current.find(id);
      return entry != entries_.end() && body != current.end() &&
             entry->second.signature == signature(*body->second);
    };
    std::map<std::pair<BodyId, b2Contact*>, std::uint32_t> contact_order;
    for (const auto& [id, entry] : entries_) {
      std::uint32_t order = 0;
      for (b2ContactNode* node = entry.native->m_contactList; node != nullptr;
           node = node->next) {
        contact_order.emplace(std::pair{id, node->contact}, order++);
      }
    }

    std::vector<ContactImpulse> result;
    std::uint32_t world_order = 0;
    for (b2Contact* contact = world_->GetContactList(); contact != nullptr;
         contact = contact->GetNext()) {
      const auto this_world_order = world_order++;
      const auto identity = contact_identity(contact);
      if (!identity || !cacheable(identity->a) ||
          (identity->b != 0 && !cacheable(identity->b))) {
        continue;
      }
      if (contact->GetManifoldCount() <= 0) {
        ContactImpulse state;
        state.a = identity->a;
        state.b = identity->b;
        state.boundary = identity->boundary;
        state.destroy_pending =
            (contact->m_flags & b2Contact::e_destroyFlag) != 0;
        state.contact_order = this_world_order;
        state.order_a = std::numeric_limits<std::uint32_t>::max();
        state.order_b = std::numeric_limits<std::uint32_t>::max();
        result.push_back(state);
        continue;
      }
      for (int manifold_index = 0;
           manifold_index < contact->GetManifoldCount(); ++manifold_index) {
        const b2Manifold& manifold = contact->GetManifolds()[manifold_index];
        for (int point_index = 0; point_index < manifold.pointCount; ++point_index) {
          const b2ContactPoint& point = manifold.points[point_index];
          const auto order_a = contact_order.find({identity->a, contact});
          const auto order_b = identity->b == 0
                                   ? contact_order.end()
                                   : contact_order.find({identity->b, contact});
          if (order_a == contact_order.end() ||
              (identity->b != 0 && order_b == contact_order.end())) {
            throw std::logic_error("touching contact is absent from a body contact list");
          }
          ContactImpulse state;
          state.a = identity->a;
          state.b = identity->b;
          state.boundary = identity->boundary;
          state.destroy_pending =
              (contact->m_flags & b2Contact::e_destroyFlag) != 0;
          state.manifold_count = static_cast<std::uint8_t>(
              contact->GetManifoldCount());
          state.manifold_index = static_cast<std::uint8_t>(manifold_index);
          state.point_count = static_cast<std::uint8_t>(manifold.pointCount);
          state.point_index = static_cast<std::uint8_t>(point_index);
          state.contact_order = this_world_order;
          state.feature_id = point.id.key;
          state.normal_x_bits = std::bit_cast<std::uint32_t>(manifold.normal.x);
          state.normal_y_bits = std::bit_cast<std::uint32_t>(manifold.normal.y);
          state.point_x_bits = std::bit_cast<std::uint32_t>(point.position.x);
          state.point_y_bits = std::bit_cast<std::uint32_t>(point.position.y);
          state.separation_bits =
              std::bit_cast<std::uint32_t>(point.separation);
          state.order_a = order_a->second;
          state.order_b = identity->b == 0 ? 0U : order_b->second;
          state.normal_impulse_bits =
              std::bit_cast<std::uint32_t>(point.normalImpulse);
          state.tangent_impulse_bits =
              std::bit_cast<std::uint32_t>(point.tangentImpulse);
          result.push_back(state);
        }
      }
    }
    std::sort(result.begin(), result.end(), impulse_less);
    return result;
  }

  std::vector<Contact> step(std::vector<Body>& bodies) {
    reconcile(bodies);
    std::vector<std::pair<BodyId, b2Body*>> pending;
    for (const auto& [id, entry] : entries_) {
      if (entry.pending_destroy) pending.emplace_back(id, entry.native);
    }
    world_->Step(static_cast<float32>(config_.tick_seconds),
                 static_cast<int32>(config_.solver_iterations));
    for (const auto& [id, native] : pending) {
      reverse_.erase(native);
      entries_.erase(id);
      const auto body = std::find_if(
          bodies.begin(), bodies.end(),
          [&](const Body& candidate) { return candidate.id == id; });
      if (body != bodies.end() && body->lifecycle == Lifecycle::Deleted) {
        body->pending_delete = false;
      }
    }
    auto contacts = contacts_after_step();
    sync_bodies(bodies);
    return contacts;
  }

  Vec2 raw_velocity(BodyId id) const {
    const auto found = entries_.find(id);
    if (found == entries_.end()) throw std::out_of_range("unknown physics body id");
    const b2Vec2 value = found->second.native->GetLinearVelocity();
    return {value.x, value.y};
  }

  void queue_destroy_by_id(BodyId id) {
    const auto found = entries_.find(id);
    if (found == entries_.end()) {
      throw std::out_of_range("unknown physics body id");
    }
    queue_destroy(found->second);
  }

 private:
  struct Entry {
    b2Body* native{};
    BodySignature signature{};
    BodyMirror mirror{};
    bool pending_destroy{};
  };

  struct ContactIdentity {
    BodyId a{};
    BodyId b{};
    BoundaryKind boundary{BoundaryKind::None};
  };

  static bool impulse_less(const ContactImpulse& left,
                           const ContactImpulse& right) {
    // Deferred r58 contacts can coexist with a replacement for the same pair.
    // Keep the legacy pending-first encoding and use world order as identity.
    return std::tuple{left.a, left.b, left.boundary, !left.destroy_pending,
                      left.contact_order, left.manifold_index,
                      left.point_index} <
           std::tuple{right.a, right.b, right.boundary,
                      !right.destroy_pending, right.contact_order,
                      right.manifold_index, right.point_index};
  }

  float32 to_world(double pixels) const {
    return static_cast<float32>(pixels) /
           static_cast<float32>(config_.world_magnification);
  }

  double to_pixels(float32 world) const {
    return static_cast<double>(static_cast<float32>(
        world * static_cast<float32>(config_.world_magnification)));
  }

  b2Body* create_static_box(double width, double height, Vec2 position,
                            double friction, double restitution) {
    b2PolyDef shape;
    configure_box(shape, width, height, config_.world_magnification);
    shape.density = 0.0f;
    shape.friction = static_cast<float32>(friction);
    shape.restitution = static_cast<float32>(restitution);
    b2BodyDef definition;
    definition.position.Set(to_world(position.x), to_world(position.y));
    definition.AddShape(&shape);
    return world_->CreateBody(&definition);
  }

  void create_boundaries() {
    const double half_thickness = std::trunc(config_.field_thickness / 2.0);
    const double half_height = std::trunc(config_.field_height / 2.0);
    const double half_width = std::trunc(config_.field_width / 2.0);
    const double center_x = config_.field_x + half_width + config_.field_thickness;
    left_wall_ = create_static_box(
        config_.field_thickness, config_.field_height,
        {config_.field_x + half_thickness, config_.field_y + half_height},
        1.0, 1.0);
    right_wall_ = create_static_box(
        config_.field_thickness, config_.field_height,
        {config_.field_x + config_.field_width + config_.field_thickness,
         config_.field_y + half_height},
        1.0, 1.0);
    floor_ = create_static_box(
        config_.field_width + 2.0 * config_.field_thickness,
        config_.field_bottom_height,
        {center_x, config_.field_y + config_.field_height + config_.field_blank +
                       std::trunc(config_.field_bottom_height / 2.0)},
        1.0, 0.0);
    top_ = create_static_box(config_.field_top_width, config_.field_top_height,
                             {center_x, config_.field_top}, 1.0, 0.5);
  }

  b2Vec2 proxy_placeholder_position(std::uint32_t expected_proxy) const {
    const auto ordinal = expected_proxy - MechanicsConfig::static_fixture_count;
    constexpr std::uint32_t columns = 32;
    constexpr std::uint32_t rows = 16;
    const float32 minimum_x = to_world(config_.world_min_x);
    const float32 minimum_y = to_world(config_.world_min_y);
    const float32 span_x = to_world(config_.world_max_x - config_.world_min_x);
    const float32 span_y = to_world(config_.world_max_y - config_.world_min_y);
    return {
        minimum_x + span_x *
                        (static_cast<float32>(ordinal % columns) + 0.5f) /
                        static_cast<float32>(columns),
        minimum_y + span_y *
                        (static_cast<float32>(ordinal / columns) + 0.5f) /
                        static_cast<float32>(rows)};
  }

  b2Body* create_proxy_placeholder(std::uint32_t expected_proxy) {
    b2CircleDef shape;
    shape.radius = 0.0001f;
    shape.density = 0.0f;
    b2BodyDef definition;
    definition.position = proxy_placeholder_position(expected_proxy);
    definition.AddShape(&shape);
    b2Body* body = world_->CreateBody(&definition);
    if (body->m_shapeList == nullptr ||
        body->m_shapeList->m_proxyId != expected_proxy) {
      throw std::logic_error("failed to allocate proxy placeholder in order");
    }
    return body;
  }

  void restore_out_of_range_proxy(Entry& entry,
                                  std::uint32_t expected_proxy) {
    b2Shape* shape = entry.native->m_shapeList;
    const b2Vec2 center = proxy_placeholder_position(expected_proxy);
    constexpr float32 half_extent = 0.0001f;
    b2AABB aabb;
    aabb.minVertex.Set(center.x - half_extent, center.y - half_extent);
    aabb.maxVertex.Set(center.x + half_extent, center.y + half_extent);
    entry.native->m_flags &=
        ~static_cast<std::uint32_t>(b2Body::e_frozenFlag);
    shape->m_proxyId = world_->m_broadPhase->CreateProxy(aabb, shape);
  }

  void restore_free_proxy_order(
      const std::vector<std::uint32_t>& free_proxy_order) {
    b2BroadPhase* broad_phase = world_->m_broadPhase;
    broad_phase->m_freeProxy = free_proxy_order.empty()
                                   ? b2_nullProxy
                                   : static_cast<std::uint16_t>(
                                         free_proxy_order.front());
    for (std::size_t index = 0; index < free_proxy_order.size(); ++index) {
      const auto proxy = static_cast<std::uint16_t>(free_proxy_order[index]);
      const auto next = index + 1 == free_proxy_order.size()
                            ? b2_nullProxy
                            : static_cast<std::uint16_t>(
                                  free_proxy_order[index + 1]);
      if (broad_phase->m_proxyPool[proxy].IsValid()) {
        throw std::logic_error("live proxy appears in restored free list");
      }
      broad_phase->m_proxyPool[proxy].SetNext(next);
    }
  }

  void restore_static_sleep_flags(std::uint8_t flags) {
    b2Body* static_bodies[] = {left_wall_, right_wall_, floor_, top_};
    for (std::size_t index = 0; index < std::size(static_bodies); ++index) {
      static_bodies[index]->m_flags &=
          ~static_cast<std::uint32_t>(b2Body::e_sleepFlag);
      if ((flags & static_cast<std::uint8_t>(1U << index)) != 0) {
        static_bodies[index]->m_flags |= b2Body::e_sleepFlag;
      }
    }
  }

  std::pair<std::uint16_t, std::uint16_t> contact_proxy_ids(
      const ContactImpulse& state) const {
    const auto body_proxy = [&](BodyId id) {
      const b2Shape* shape = entries_.at(id).native->m_shapeList;
      if (shape == nullptr || shape->m_proxyId == b2_nullProxy) {
        throw std::invalid_argument("snapshot contact body has no live proxy");
      }
      return shape->m_proxyId;
    };
    const auto boundary_proxy = [&] {
      b2Body* native = nullptr;
      switch (state.boundary) {
        case BoundaryKind::Floor:
          native = floor_;
          break;
        case BoundaryKind::LeftWall:
          native = left_wall_;
          break;
        case BoundaryKind::RightWall:
          native = right_wall_;
          break;
        case BoundaryKind::Top:
          native = top_;
          break;
        case BoundaryKind::None:
          break;
      }
      if (native == nullptr || native->m_shapeList == nullptr) {
        throw std::invalid_argument("snapshot contact boundary is invalid");
      }
      return native->m_shapeList->m_proxyId;
    };
    const auto first = body_proxy(state.a);
    const auto second = state.b == 0 ? boundary_proxy() : body_proxy(state.b);
    return {std::min(first, second), std::max(first, second)};
  }

  std::pair<b2Shape*, b2Shape*> pending_contact_shapes(
      const ContactImpulse& state) const {
    b2Shape* first = entries_.at(state.a).native->m_shapeList;
    if (state.b != 0) {
      b2Shape* second = entries_.at(state.b).native->m_shapeList;
      if (first->m_proxyId != b2_nullProxy &&
          second->m_proxyId != b2_nullProxy &&
          first->m_proxyId > second->m_proxyId) {
        std::swap(first, second);
      }
      return {first, second};
    }

    b2Body* boundary_body = nullptr;
    switch (state.boundary) {
      case BoundaryKind::Floor:
        boundary_body = floor_;
        break;
      case BoundaryKind::LeftWall:
        boundary_body = left_wall_;
        break;
      case BoundaryKind::RightWall:
        boundary_body = right_wall_;
        break;
      case BoundaryKind::Top:
        boundary_body = top_;
        break;
      case BoundaryKind::None:
        break;
    }
    if (boundary_body == nullptr || boundary_body->m_shapeList == nullptr) {
      throw std::invalid_argument("snapshot contact boundary is invalid");
    }
    return {boundary_body->m_shapeList, first};
  }

  void restore_broad_phase(
      const PhysicsOrdering& ordering,
      const std::vector<ContactImpulse>& contact_impulses) {
    if (ordering.broadphase_bounds.empty() &&
        ordering.proxy_time_stamps.empty() &&
        ordering.proxy_overlap_counts.empty()) {
      return;
    }
    b2BroadPhase* broad_phase = world_->m_broadPhase;
    const auto bound_count = static_cast<std::size_t>(
        2 * broad_phase->m_proxyCount);
    if (ordering.broadphase_bounds.size() != 2 * bound_count ||
        ordering.proxy_time_stamps.size() != b2_maxProxies ||
        ordering.proxy_overlap_counts.size() != b2_maxProxies ||
        broad_phase->m_pairManager.m_pairBufferCount != 0) {
      throw std::invalid_argument("snapshot broad-phase state has invalid dimensions");
    }

    std::vector<std::pair<std::uint16_t, std::uint16_t>> dynamic_pairs;
    for (b2Pair& pair : broad_phase->m_pairManager.m_pairs) {
      if (pair.proxyId1 == b2_nullProxy || !pair.IsFinal()) continue;
      if (pair.proxyId1 >= MechanicsConfig::static_fixture_count ||
          pair.proxyId2 >= MechanicsConfig::static_fixture_count) {
        dynamic_pairs.emplace_back(pair.proxyId1, pair.proxyId2);
      }
    }
    world_->m_contactManager.m_destroyImmediate = true;
    for (const auto& [first, second] : dynamic_pairs) {
      broad_phase->m_pairManager.RemoveBufferedPair(first, second);
    }
    broad_phase->m_pairManager.Commit();
    world_->m_contactManager.m_destroyImmediate = false;

    for (int axis = 0; axis < 2; ++axis) {
      for (std::size_t index = 0; index < bound_count; ++index) {
        const auto& saved = ordering.broadphase_bounds[
            static_cast<std::size_t>(axis) * bound_count + index];
        b2Bound& bound = broad_phase->m_bounds[axis][index];
        bound.value = saved.value;
        bound.proxyId = saved.proxy_id;
        bound.stabbingCount = saved.stabbing_count;
        if (bound.proxyId >= b2_maxProxies ||
            !broad_phase->m_proxyPool[bound.proxyId].IsValid()) {
          throw std::invalid_argument("snapshot broad-phase bound has invalid proxy");
        }
        if (bound.IsLower()) {
          broad_phase->m_proxyPool[bound.proxyId].lowerBounds[axis] =
              static_cast<std::uint16_t>(index);
        } else {
          broad_phase->m_proxyPool[bound.proxyId].upperBounds[axis] =
              static_cast<std::uint16_t>(index);
        }
      }
    }
    for (std::size_t proxy = 0; proxy < b2_maxProxies; ++proxy) {
      broad_phase->m_proxyPool[proxy].timeStamp =
          ordering.proxy_time_stamps[proxy];
      broad_phase->m_proxyPool[proxy].overlapCount =
          ordering.proxy_overlap_counts[proxy];
    }
    broad_phase->m_timeStamp = ordering.broadphase_time_stamp;
    broad_phase->m_queryResultCount = 0;
    broad_phase->Validate();

    std::set<std::pair<std::uint16_t, std::uint16_t>> active_pairs;
    for (std::size_t first = 0; first < contact_impulses.size();) {
      std::size_t last = first + 1;
      while (last < contact_impulses.size() &&
             contact_impulses[last].a == contact_impulses[first].a &&
             contact_impulses[last].b == contact_impulses[first].b &&
             contact_impulses[last].boundary ==
                 contact_impulses[first].boundary &&
             contact_impulses[last].contact_order ==
                 contact_impulses[first].contact_order) {
        ++last;
      }
      const auto& state = contact_impulses[first];
      if (!state.destroy_pending) active_pairs.insert(contact_proxy_ids(state));
      first = last;
    }
    for (const auto& [first, second] : active_pairs) {
      const b2Proxy& first_proxy = broad_phase->m_proxyPool[first];
      const b2Proxy& second_proxy = broad_phase->m_proxyPool[second];
      bool overlaps = true;
      for (int axis = 0; axis < 2; ++axis) {
        const b2Bound* bounds = broad_phase->m_bounds[axis];
        if (bounds[first_proxy.lowerBounds[axis]].value >
                bounds[second_proxy.upperBounds[axis]].value ||
            bounds[first_proxy.upperBounds[axis]].value <
                bounds[second_proxy.lowerBounds[axis]].value) {
          overlaps = false;
          break;
        }
      }
      if (!overlaps) {
        throw std::invalid_argument(
            "snapshot contact is absent from restored broad phase");
      }
      broad_phase->m_pairManager.AddBufferedPair(first, second);
    }
    broad_phase->m_pairManager.Commit();

    for (std::size_t first = 0; first < contact_impulses.size();) {
      std::size_t last = first + 1;
      while (last < contact_impulses.size() &&
             contact_impulses[last].a == contact_impulses[first].a &&
             contact_impulses[last].b == contact_impulses[first].b &&
             contact_impulses[last].boundary ==
                 contact_impulses[first].boundary &&
             contact_impulses[last].contact_order ==
                 contact_impulses[first].contact_order) {
        ++last;
      }
      const auto& state = contact_impulses[first];
      if (state.destroy_pending) {
        auto [first_shape, second_shape] = pending_contact_shapes(state);
        b2Contact* contact = b2Contact::Create(
            first_shape, second_shape, &world_->m_blockAllocator);
        if (contact == nullptr) {
          throw std::invalid_argument("snapshot deferred contact cannot be rebuilt");
        }
        contact->m_flags |= b2Contact::e_destroyFlag;
        contact->m_prev = nullptr;
        contact->m_next = world_->m_contactList;
        if (contact->m_next != nullptr) contact->m_next->m_prev = contact;
        world_->m_contactList = contact;
        ++world_->m_contactCount;
      }
      first = last;
    }
  }

  void initialize_native_state(Body& body) const {
    if (body.native_state_valid) return;
    body.native_position = {to_world(body.position.x), to_world(body.position.y)};
    body.native_center_valid = false;
    body.native_velocity = {
        static_cast<float32>(body.velocity.x),
        static_cast<float32>(body.velocity.y)};
    body.native_angle = static_cast<float32>(body.angle);
    body.native_angular_velocity =
        static_cast<float32>(body.angular_velocity);
    body.native_state_valid = true;
  }

  void create_body(Body& body) {
    initialize_native_state(body);
    const bool restoring_center = body.native_center_valid;
    b2BodyDef definition;
    definition.position.Set(static_cast<float32>(body.native_position.x),
                            static_cast<float32>(body.native_position.y));
    definition.rotation = static_cast<float32>(body.native_angle);
    definition.linearDamping = static_cast<float32>(config_.linear_damping);
    definition.angularDamping = static_cast<float32>(config_.angular_damping);

    b2PolyDef box;
    b2CircleDef circle;
    b2PolyDef triangle;
    switch (body.shape) {
      case Shape::Box:
        configure_box(box, body.size, body.size,
                      config_.world_magnification);
        configure_material(box, body);
        definition.AddShape(&box);
        break;
      case Shape::Circle:
        circle.radius = to_world(body.size * 0.5);
        configure_material(circle, body);
        definition.AddShape(&circle);
        break;
      case Shape::Triangle:
        configure_triangle(triangle, body.size, config_.world_magnification);
        configure_material(triangle, body);
        definition.AddShape(&triangle);
        break;
    }

    b2Body* native = world_->CreateBody(&definition);
    if (restoring_center) {
      const b2Vec2 canonical_center = native->m_position;
      const float32 saved_center_x =
          static_cast<float32>(body.native_center.x);
      const float32 saved_center_y =
          static_cast<float32>(body.native_center.y);
      const bool canonical_center_matches =
          std::bit_cast<std::uint32_t>(canonical_center.x) ==
              std::bit_cast<std::uint32_t>(saved_center_x) &&
          std::bit_cast<std::uint32_t>(canonical_center.y) ==
              std::bit_cast<std::uint32_t>(saved_center_y);
      native->m_position.Set(saved_center_x, saved_center_y);
      native->m_position0 = native->m_position;
      native->m_rotation0 = native->m_rotation;
      native->m_R.Set(native->m_rotation);
      native->QuickSyncShapes();

      const b2Vec2 restored_origin = native->GetOriginPosition();
      const float32 expected_x = static_cast<float32>(body.native_position.x);
      const float32 expected_y = static_cast<float32>(body.native_position.y);
      if (!canonical_center_matches &&
          (std::bit_cast<std::uint32_t>(restored_origin.x) !=
               std::bit_cast<std::uint32_t>(expected_x) ||
           std::bit_cast<std::uint32_t>(restored_origin.y) !=
               std::bit_cast<std::uint32_t>(expected_y))) {
        world_->DestroyBody(native);
        world_->CleanBodyList();
        throw std::invalid_argument(
            "snapshot native center is inconsistent with its origin for body " +
            std::to_string(body.id));
      }
    }
    body.native_center = {native->m_position.x, native->m_position.y};
    body.native_center_valid = true;
    native->SetLinearVelocity(
        b2Vec2(static_cast<float32>(body.native_velocity.x),
               static_cast<float32>(body.native_velocity.y)));
    native->SetAngularVelocity(
        static_cast<float32>(body.native_angular_velocity));
    if (native->m_invMass > 0.0f) {
      native->m_sleepTime = static_cast<float32>(body.sleep_time);
      if (body.sleeping) native->m_flags |= b2Body::e_sleepFlag;
    }
    entries_.emplace(body.id,
                     Entry{native, signature(body), mirror(body), false});
    reverse_.emplace(native, body.id);
  }

  void queue_destroy(Entry& entry) {
    if (entry.pending_destroy) return;
    world_->DestroyBody(entry.native);
    entry.pending_destroy = true;
  }

  void apply_external_changes(Body& body, Entry& entry) {
    const BodyMirror previous = entry.mirror;
    const Vec2 requested_native_velocity = body.native_velocity;
    const double requested_native_angular_velocity =
        body.native_angular_velocity;
    const bool transform_changed =
        !same(body.position, previous.position) || !same(body.angle, previous.angle);
    const bool native_velocity_changed =
        !same(requested_native_velocity, previous.native_velocity);

    if (transform_changed) {
      entry.native->SetOriginPosition(
          b2Vec2(to_world(body.position.x), to_world(body.position.y)),
          static_cast<float32>(body.angle));
      entry.native->SetLinearVelocity(b2Vec2(0.0f, 0.0f));
      body.native_position = {entry.native->GetOriginPosition().x,
                              entry.native->GetOriginPosition().y};
      body.native_center = {entry.native->m_position.x,
                            entry.native->m_position.y};
      body.native_center_valid = true;
      body.native_angle = entry.native->GetRotation();
      body.native_velocity = {};
    }
    if (native_velocity_changed) {
      entry.native->SetLinearVelocity(
          b2Vec2(static_cast<float32>(requested_native_velocity.x),
                 static_cast<float32>(requested_native_velocity.y)));
      body.native_velocity = {entry.native->GetLinearVelocity().x,
                              entry.native->GetLinearVelocity().y};
    }
    if (!same(requested_native_angular_velocity,
              previous.native_angular_velocity)) {
      entry.native->SetAngularVelocity(
          static_cast<float32>(requested_native_angular_velocity));
      body.native_angular_velocity = entry.native->GetAngularVelocity();
    }

    if (body.sleeping != previous.sleeping) {
      if (body.sleeping) {
        entry.native->m_flags |= b2Body::e_sleepFlag;
      } else {
        entry.native->WakeUp();
      }
    }
    if (!same(body.sleep_time, previous.sleep_time)) {
      entry.native->m_sleepTime = static_cast<float32>(body.sleep_time);
    }
    entry.mirror = mirror(body);
  }

  void reconcile(std::vector<Body>& bodies) {
    std::vector<BodyId> described_ids;
    described_ids.reserve(bodies.size());
    std::vector<Body*> actor_order;
    actor_order.reserve(bodies.size());
    for (auto& body : bodies) {
      if (body.id == 0) {
        throw std::invalid_argument("physics bodies require unique nonzero ids");
      }
      described_ids.push_back(body.id);
      actor_order.push_back(&body);
    }
    std::sort(described_ids.begin(), described_ids.end());
    if (std::adjacent_find(described_ids.begin(), described_ids.end()) !=
        described_ids.end()) {
      throw std::invalid_argument("physics bodies require unique nonzero ids");
    }
    std::stable_sort(actor_order.begin(), actor_order.end(),
                     [](const Body* left, const Body* right) {
                       if (left->actor_slot != right->actor_slot) {
                         return left->actor_slot < right->actor_slot;
                       }
                       if ((left->lifecycle == Lifecycle::Deleted) !=
                           (right->lifecycle == Lifecycle::Deleted)) {
                         return left->lifecycle == Lifecycle::Deleted;
                       }
                       return left->id < right->id;
                     });
    for (Body* candidate : actor_order) {
      Body& body = *candidate;
      auto found = entries_.find(body.id);
      if (body.lifecycle == Lifecycle::Deleted) {
        if (found != entries_.end()) queue_destroy(found->second);
        continue;
      }
      if (found == entries_.end()) {
        create_body(body);
        continue;
      }
      if (found->second.pending_destroy) {
        throw std::invalid_argument("destroyed physics body id cannot be revived");
      }
      if (found->second.signature != signature(body)) {
        throw std::invalid_argument(
            "physics fixture signature cannot change after creation");
      }
      apply_external_changes(body, found->second);
    }
    for (auto& [id, entry] : entries_) {
      if (!std::binary_search(described_ids.begin(), described_ids.end(), id)) {
        queue_destroy(entry);
      }
    }
  }

  BoundaryKind boundary(b2Body* body) const {
    if (body == floor_) return BoundaryKind::Floor;
    if (body == left_wall_) return BoundaryKind::LeftWall;
    if (body == right_wall_) return BoundaryKind::RightWall;
    if (body == top_) return BoundaryKind::Top;
    return BoundaryKind::None;
  }

  void restore_body_order(const std::vector<BodyId>& body_order) {
    std::vector<b2Body*> ordered;
    ordered.reserve(static_cast<std::size_t>(world_->m_bodyCount));
    for (const BodyId id : body_order) ordered.push_back(entries_.at(id).native);
    for (b2Body* body = world_->GetBodyList(); body != nullptr;
         body = body->GetNext()) {
      if (!reverse_.contains(body)) ordered.push_back(body);
    }
    world_->m_bodyList = ordered.empty() ? nullptr : ordered.front();
    for (std::size_t index = 0; index < ordered.size(); ++index) {
      ordered[index]->m_prev = index == 0 ? nullptr : ordered[index - 1];
      ordered[index]->m_next =
          index + 1 == ordered.size() ? nullptr : ordered[index + 1];
    }
  }

  std::optional<ContactIdentity> contact_identity(b2Contact* contact) const {
    b2Body* body1 = contact->GetShape1()->m_body;
    b2Body* body2 = contact->GetShape2()->m_body;
    const auto id1 = reverse_.find(body1);
    const auto id2 = reverse_.find(body2);
    if (id1 != reverse_.end() && id2 != reverse_.end()) {
      return ContactIdentity{std::min(id1->second, id2->second),
                             std::max(id1->second, id2->second),
                             BoundaryKind::None};
    }
    const bool first_is_body = id1 != reverse_.end();
    const bool second_is_body = id2 != reverse_.end();
    if (first_is_body == second_is_body) return std::nullopt;
    const BoundaryKind edge = boundary(first_is_body ? body2 : body1);
    if (edge == BoundaryKind::None) return std::nullopt;
    return ContactIdentity{first_is_body ? id1->second : id2->second, 0, edge};
  }

  static void disconnect_contact(b2Contact* contact) {
    b2Body* body1 = contact->m_shape1->m_body;
    b2Body* body2 = contact->m_shape2->m_body;
    if (contact->m_node1.prev) contact->m_node1.prev->next = contact->m_node1.next;
    if (contact->m_node1.next) contact->m_node1.next->prev = contact->m_node1.prev;
    if (body1->m_contactList == &contact->m_node1) {
      body1->m_contactList = contact->m_node1.next;
    }
    if (contact->m_node2.prev) contact->m_node2.prev->next = contact->m_node2.next;
    if (contact->m_node2.next) contact->m_node2.next->prev = contact->m_node2.prev;
    if (body2->m_contactList == &contact->m_node2) {
      body2->m_contactList = contact->m_node2.next;
    }
    contact->m_node1 = {};
    contact->m_node2 = {};
  }

  static void connect_contact(b2Contact* contact) {
    b2Body* body1 = contact->m_shape1->m_body;
    b2Body* body2 = contact->m_shape2->m_body;
    contact->m_node1.contact = contact;
    contact->m_node1.other = body2;
    contact->m_node1.prev = nullptr;
    contact->m_node1.next = body1->m_contactList;
    if (contact->m_node1.next) contact->m_node1.next->prev = &contact->m_node1;
    body1->m_contactList = &contact->m_node1;
    contact->m_node2.contact = contact;
    contact->m_node2.other = body1;
    contact->m_node2.prev = nullptr;
    contact->m_node2.next = body2->m_contactList;
    if (contact->m_node2.next) contact->m_node2.next->prev = &contact->m_node2;
    body2->m_contactList = &contact->m_node2;
  }

  void restore_contact_impulses(
      const std::vector<ContactImpulse>& contact_impulses) {
    if (contact_impulses.empty()) return;

    std::vector<std::pair<b2Body*, bool>> sleep_flags;
    sleep_flags.reserve(entries_.size());
    for (const auto& [id, entry] : entries_) {
      (void)id;
      const bool sleeping = entry.mirror.sleeping;
      sleep_flags.emplace_back(entry.native, sleeping);
      entry.native->m_flags &=
          ~static_cast<std::uint32_t>(b2Body::e_sleepFlag);
    }
    world_->m_contactManager.Collide();

    using ContactKey = std::tuple<BodyId, BodyId, BoundaryKind>;
    std::map<ContactKey, std::vector<b2Contact*>> native;
    for (b2Contact* contact = world_->GetContactList(); contact != nullptr;
         contact = contact->GetNext()) {
      const auto identity = contact_identity(contact);
      if (identity) {
        native[ContactKey{identity->a, identity->b, identity->boundary}]
            .push_back(contact);
      }
    }

    std::map<b2Contact*, const ContactImpulse*> saved_contact;
    for (std::size_t first = 0; first < contact_impulses.size();) {
      std::size_t last = first + 1;
      while (last < contact_impulses.size() &&
             contact_impulses[last].a == contact_impulses[first].a &&
             contact_impulses[last].b == contact_impulses[first].b &&
             contact_impulses[last].boundary ==
                 contact_impulses[first].boundary &&
             contact_impulses[last].contact_order ==
                 contact_impulses[first].contact_order) {
        ++last;
      }
      const auto& saved = contact_impulses[first];
      const auto found = native.find(ContactKey{saved.a, saved.b, saved.boundary});
      if (found == native.end()) {
        throw std::invalid_argument("snapshot contact is absent from rebuilt broad phase");
      }
      const auto matching = std::find_if(
          found->second.begin(), found->second.end(), [&](b2Contact* contact) {
            const bool destroy_pending =
                (contact->m_flags & b2Contact::e_destroyFlag) != 0;
            return destroy_pending == saved.destroy_pending &&
                   !saved_contact.contains(contact);
          });
      if (matching == found->second.end()) {
        throw std::invalid_argument(
            "snapshot duplicate contact cannot be matched in rebuilt broad phase");
      }
      b2Contact* contact = *matching;
      saved_contact.emplace(contact, &saved);
      if (saved.destroy_pending) {
        contact->m_flags |= b2Contact::e_destroyFlag;
      } else {
        contact->m_flags &=
            ~static_cast<std::uint32_t>(b2Contact::e_destroyFlag);
      }
      const bool was_touching = contact->GetManifoldCount() > 0;
      const bool should_touch = saved.manifold_count > 0;
      if (was_touching && !should_touch) disconnect_contact(contact);

      b2Manifold& manifold = contact->GetManifolds()[0];
      if (!should_touch) {
        manifold = {};
        contact->m_manifoldCount = 0;
      } else {
        manifold = {};
        manifold.pointCount = saved.point_count;
        manifold.normal.x = std::bit_cast<float32>(saved.normal_x_bits);
        manifold.normal.y = std::bit_cast<float32>(saved.normal_y_bits);
        for (std::size_t index = first; index < last; ++index) {
          const auto& point_state = contact_impulses[index];
          b2ContactPoint& point = manifold.points[point_state.point_index];
          point.position.x = std::bit_cast<float32>(point_state.point_x_bits);
          point.position.y = std::bit_cast<float32>(point_state.point_y_bits);
          point.separation = std::bit_cast<float32>(point_state.separation_bits);
          point.id.key = point_state.feature_id;
          point.normalImpulse =
              std::bit_cast<float32>(point_state.normal_impulse_bits);
          point.tangentImpulse =
              std::bit_cast<float32>(point_state.tangent_impulse_bits);
        }
        contact->m_manifoldCount = 1;
        if (!was_touching) connect_contact(contact);
      }
      first = last;
    }

    std::vector<b2Contact*> contacts;
    for (b2Contact* contact = world_->GetContactList(); contact != nullptr;
         contact = contact->GetNext()) {
      contacts.push_back(contact);
    }
    std::stable_sort(contacts.begin(), contacts.end(), [&](const auto* left,
                                                           const auto* right) {
      const auto left_saved = saved_contact.find(const_cast<b2Contact*>(left));
      const auto right_saved = saved_contact.find(const_cast<b2Contact*>(right));
      const auto left_order = left_saved == saved_contact.end()
                                  ? std::numeric_limits<std::uint32_t>::max()
                                  : left_saved->second->contact_order;
      const auto right_order = right_saved == saved_contact.end()
                                   ? std::numeric_limits<std::uint32_t>::max()
                                   : right_saved->second->contact_order;
      return left_order < right_order;
    });
    world_->m_contactList = contacts.empty() ? nullptr : contacts.front();
    for (std::size_t index = 0; index < contacts.size(); ++index) {
      contacts[index]->m_prev = index == 0 ? nullptr : contacts[index - 1];
      contacts[index]->m_next =
          index + 1 == contacts.size() ? nullptr : contacts[index + 1];
    }

    for (const auto& [id, entry] : entries_) {
      std::vector<b2ContactNode*> nodes;
      for (b2ContactNode* node = entry.native->m_contactList; node != nullptr;
           node = node->next) {
        nodes.push_back(node);
      }
      const auto saved_order = [&](const b2ContactNode* node) {
        const auto found = saved_contact.find(node->contact);
        if (found == saved_contact.end()) {
          return std::numeric_limits<std::uint32_t>::max();
        }
        return id == found->second->a ? found->second->order_a
                                     : found->second->order_b;
      };
      std::stable_sort(nodes.begin(), nodes.end(), [&](const auto* left,
                                                       const auto* right) {
        return saved_order(left) < saved_order(right);
      });
      entry.native->m_contactList = nodes.empty() ? nullptr : nodes.front();
      for (std::size_t index = 0; index < nodes.size(); ++index) {
        nodes[index]->prev = index == 0 ? nullptr : nodes[index - 1];
        nodes[index]->next = index + 1 == nodes.size() ? nullptr : nodes[index + 1];
      }
    }
    for (const auto& [body, sleeping] : sleep_flags) {
      const auto found = reverse_.find(body);
      if (found != reverse_.end()) {
        body->m_sleepTime =
            static_cast<float32>(entries_.at(found->second).mirror.sleep_time);
      }
      if (sleeping) {
        body->m_flags |= b2Body::e_sleepFlag;
      } else {
        body->m_flags &= ~static_cast<std::uint32_t>(b2Body::e_sleepFlag);
      }
    }
  }

  std::vector<Contact> contacts_after_step() const {
    std::vector<Contact> result;
    for (b2Contact* native_contact = world_->GetContactList(); native_contact != nullptr;
         native_contact = native_contact->GetNext()) {
      if (native_contact->GetManifoldCount() <= 0) continue;
      b2Body* body1 = native_contact->GetShape1()->m_body;
      b2Body* body2 = native_contact->GetShape2()->m_body;
      const auto id1 = reverse_.find(body1);
      const auto id2 = reverse_.find(body2);
      const BoundaryKind boundary1 = boundary(body1);
      const BoundaryKind boundary2 = boundary(body2);
      if (id1 == reverse_.end() && id2 == reverse_.end()) continue;

      const b2Manifold& manifold = native_contact->GetManifolds()[0];
      if (manifold.pointCount <= 0 ||
          manifold.pointCount > b2_maxManifoldPoints) {
        continue;
      }
      int point_index = 0;
      for (int index = 1; index < manifold.pointCount; ++index) {
        if (manifold.points[index].separation <
            manifold.points[point_index].separation) {
          point_index = index;
        }
      }
      Contact contact;
      contact.point = {to_pixels(manifold.points[point_index].position.x),
                       to_pixels(manifold.points[point_index].position.y)};
      contact.penetration = std::max(
          0.0, -to_pixels(manifold.points[point_index].separation));
      contact.normal = {manifold.normal.x, manifold.normal.y};
      contact.a = id1 == reverse_.end() ? 0 : id1->second;
      contact.b = id2 == reverse_.end() ? 0 : id2->second;
      if (contact.a == 0 || contact.b == 0) {
        contact.boundary = contact.a == 0 ? boundary1 : boundary2;
        if (contact.boundary == BoundaryKind::None) continue;
      }
      result.push_back(contact);
    }
    return result;
  }

  void sync_bodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
      if (body.lifecycle == Lifecycle::Deleted) continue;
      const auto found = entries_.find(body.id);
      if (found == entries_.end()) continue;
      b2Body* native = found->second.native;
      const b2Vec2 position = native->GetOriginPosition();
      const b2Vec2 velocity = native->GetLinearVelocity();
      body.position = {to_pixels(position.x), to_pixels(position.y)};
      body.velocity = {velocity.x, velocity.y};
      body.angle = static_cast<double>(native->GetRotation());
      body.native_position = {position.x, position.y};
      body.native_center = {native->m_position.x, native->m_position.y};
      body.native_velocity = {velocity.x, velocity.y};
      body.native_angle = native->GetRotation();
      body.native_angular_velocity = native->GetAngularVelocity();
      body.native_state_valid = true;
      body.native_center_valid = true;
      body.inverse_mass = native->m_invMass;
      body.inverse_inertia = native->m_invI;
      body.sleeping = native->m_invMass > 0.0f && native->IsSleeping();
      body.sleep_time = native->m_invMass > 0.0f ? native->m_sleepTime : 0.0;
      found->second.mirror = mirror(body);
    }
  }

  MechanicsConfig config_;
  std::unique_ptr<b2World> world_;
  b2Body* floor_{};
  b2Body* left_wall_{};
  b2Body* right_wall_{};
  b2Body* top_{};
  std::map<BodyId, Entry> entries_;
  std::map<b2Body*, BodyId> reverse_;
};

PhysicsWorld::PhysicsWorld(MechanicsConfig config)
    : config_(validated_mechanics_config(std::move(config))),
      impl_([this] {
        const ScopedFloatingPointEnvironment floating_point_environment;
        return std::make_unique<Impl>(config_);
      }()) {}

PhysicsWorld::~PhysicsWorld() {
  const ScopedFloatingPointEnvironment floating_point_environment;
  impl_.reset();
}
PhysicsWorld::PhysicsWorld(PhysicsWorld&&) noexcept = default;
PhysicsWorld& PhysicsWorld::operator=(PhysicsWorld&& other) noexcept {
  const ScopedFloatingPointEnvironment floating_point_environment;
  config_ = std::move(other.config_);
  impl_ = std::move(other.impl_);
  return *this;
}

void PhysicsWorld::initialize_mass(Body& body) const {
  const ScopedFloatingPointEnvironment floating_point_environment;
  if (!body.native_state_valid) {
    const auto magnification =
        static_cast<float32>(config_.world_magnification);
    body.native_position = {
        static_cast<float32>(body.position.x) / magnification,
        static_cast<float32>(body.position.y) / magnification};
    body.native_center_valid = false;
    body.native_velocity = {
        static_cast<float32>(body.velocity.x),
        static_cast<float32>(body.velocity.y)};
    body.native_angle = static_cast<float32>(body.angle);
    body.native_angular_velocity =
        static_cast<float32>(body.angular_velocity);
    body.native_state_valid = true;
  }
  body.sleeping = false;
  body.sleep_time = 0.0;
  if (body.lifecycle == Lifecycle::Deleted || body.density <= 0.0) {
    body.inverse_mass = 0.0;
    body.inverse_inertia = 0.0;
    return;
  }

  b2MassData mass{};
  if (body.shape == Shape::Box) {
    b2PolyDef shape;
    configure_box(shape, body.size, body.size,
                  config_.world_magnification);
    shape.density = static_cast<float32>(body.density);
    shape.ComputeMass(&mass);
  } else if (body.shape == Shape::Circle) {
    b2CircleDef shape;
    shape.radius = static_cast<float32>(body.size / (2.0 * config_.world_magnification));
    shape.density = static_cast<float32>(body.density);
    shape.ComputeMass(&mass);
  } else {
    b2PolyDef shape;
    configure_triangle(shape, body.size, config_.world_magnification);
    shape.density = static_cast<float32>(body.density);
    shape.ComputeMass(&mass);
  }
  body.inverse_mass = mass.mass > 0.0f ? 1.0 / mass.mass : 0.0;
  body.inverse_inertia = mass.I > 0.0f ? 1.0 / mass.I : 0.0;
}

void PhysicsWorld::reset() {
  const ScopedFloatingPointEnvironment floating_point_environment;
  impl_->reset();
}

void PhysicsWorld::synchronize(std::vector<Body>& bodies) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  impl_->synchronize(bodies);
}

void PhysicsWorld::queue_destroy(BodyId id) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  impl_->queue_destroy_by_id(id);
}

void PhysicsWorld::rebuild(
    std::vector<Body>& bodies,
    const std::vector<ContactImpulse>& contact_impulses,
    const PhysicsOrdering& ordering) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  impl_->rebuild(bodies, contact_impulses, ordering);
}

std::vector<ContactImpulse> PhysicsWorld::contact_impulses(
    const std::vector<Body>& bodies) const {
  const ScopedFloatingPointEnvironment floating_point_environment;
  return impl_->contact_impulses(bodies);
}

PhysicsOrdering PhysicsWorld::ordering() const {
  const ScopedFloatingPointEnvironment floating_point_environment;
  return impl_->ordering();
}

std::vector<Contact> PhysicsWorld::step(std::vector<Body>& bodies) {
  const ScopedFloatingPointEnvironment floating_point_environment;
  return impl_->step(bodies);
}

Vec2 PhysicsWorld::raw_velocity(BodyId id) const {
  const ScopedFloatingPointEnvironment floating_point_environment;
  return impl_->raw_velocity(id);
}

}  // namespace irisu
