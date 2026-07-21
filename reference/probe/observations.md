# Box2D.dll probe observations

Target: shipped v2.03 `Box2D.dll`, SHA-256
`34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd`.
Runtime: bundled Wine 11.13. Probe schema: 2.
Golden trace SHA-256:
`b73f9c74db48a9695450ab04b76661c0d576030f7f428d0e53b953b4dae077b2`.

These are observed oracle results, not assumptions about a newer Box2D release.

- All 16 decorated x86 `stdcall` symbols in `binary-analysis.md` resolve.
- With magnification 100, `set_v(250, -500)` is read back as world velocity
  `(2.5, -5)`. One `dt=0.020` step moves the returned pixel position from
  `(123, 233.999985)` to `(128, 224)`.
- `set_position(321, 222, -0.5)` returns that transform and immediately resets
  both linear velocity components to zero.
- With configured gravity 100 and magnification 100, the first `0.020` step
  changes raw world Y velocity by `0.02` and pixel Y from `100` to `100.04`.
  This is semi-implicit Euler integration (`v += g*dt`, then `p += v*dt`).
- Gravity is stored in world units as `gravity_argument / magnification`.
  At magnification 100, gravity 250 gives raw Y velocities `0.05` and `0.10`
  after one and two `0.020` steps, with returned pixel Y values `100.100006`
  and `100.300003`. Gravity -250 gives `-0.05`, `-0.10` and pixel Y
  `99.900002`, `99.700005`. Changing magnification from 100 to 50 with gravity
  100 doubles raw velocity (`0.02` to `0.04`) but leaves returned pixel
  positions bit-identical (`100.039993`, then `100.119995`). Thus the
  semi-implicit free-fall formulas are `v_n = n*g/m*dt` in raw world units and
  `y_n = y_0 + g*dt^2*n*(n+1)/2` in returned pixel units.
- A density-zero box is static. Three dynamic shapes settle on its top at pixel
  centers approximately `280.5002` (20-high box), `278.5001` (radius-12
  circle), and `275.5002` (30-high triangle). This is roughly `0.5` pixel of
  allowed contact penetration at magnification 100, not an added configuration
  dimension.
- The dimension trial makes the wrapper arguments explicit. A static floor
  centered at Y 300 with height 20 has nominal top Y 290. A height-20 box and
  radius-10 circle settle at `280.500122` and `280.500061`; a height-40 box and
  radius-20 circle settle at `270.499969` and `270.500031`. Therefore box
  width/height are full dimensions converted to half-extents internally, while
  the circle argument is already a radius. All four lower extents end near
  `290.5`, directly measuring the legacy `0.005` world-unit/`0.5` pixel slop.
  The larger pair first forms floor manifolds on tick 93 and the smaller pair
  on tick 96 from the same Y=100 release.
- In the isolated fall, circle and triangle floor manifolds first appear on tick
  95; the box appears on tick 96. Stable contact iteration order is floor-box,
  floor-circle, floor-triangle once all three exist. `b2d_get_contact` returns
  the exact assigned user-data values and only manifold-bearing contacts.
- Contact enumeration is list order, newest contact first. Three simultaneous
  contacts created as bodies `701, 702, 703` enumerate `703, 702, 701`.
  Adding `704` prepends it; then adding `705` followed by `706` produces
  `706, 705, 704, 703, 702, 701`. For static/dynamic pairs the wrapper reports
  the static body's user data as `user_a` and dynamic body as `user_b`.
- A circle sent at raw wrapper velocity 500 toward a static wall is reported at
  raw world velocity 5. A wall restitution of 1 and circle restitution of 0.5
  rebound it at `-5`, evidence that this legacy engine selects the maximum
  restitution rather than multiplying the coefficients.
- Friction is symmetric `sqrt(friction_a * friction_b)`. Under world gravity
  1, `dt=0.020`, and initial raw horizontal velocity 1, coefficient pairs
  `(1,0)`, `(1,0.25)`, `(1,1)`, and `(0.25,1)` yield tick-20 velocities
  `1`, `0.800000191`, `0.600000501`, and `0.800000191`. The two mixed-0.5
  cases are bit-identical and decelerate by `0.01` per tick; mixed-1 decelerates
  by `0.02` per tick.
- Sleeping begins after exactly 25 eligible `0.020` resting-contact steps
  (`0.5` seconds). The control body's raw state changes through tick 25 and is
  bit-identical from tick 25 through 35. Setting raw X velocity to 1 after tick
  24 moves an awake body by 2 pixels on tick 25. Setting it after tick 25 leaves
  position unchanged and preserves the reported velocity at 1, proving both
  the sleep boundary and that the wrapper's `set_v` does not wake a body.
- An unrotated wrapper triangle with width 100 and height 60 occupies the right
  triangle with vertices `(-50,-30)`, `(-50,30)`, `(50,30)` relative to its
  origin. Radius-2 probes contact at the interior points `(-35,0)` and `(0,25)`
  but not `(35,0)` or `(0,-25)`. Edge probes contact at `(-51,-15)` on the
  left edge and `(-20,31)` below the bottom edge, while `(-53,15)` and
  `(20,33)` miss. This discriminates the wrapper geometry from an isosceles or
  oppositely oriented triangle.
- Calling `b2d_test` after setting linear velocity did not change the visible
  transform or linear velocity in the following step. Binary analysis shows it
  writes `0.1` to an internal body field; it remains unused game behavior.
- `destroy_body(NULL)` returns safely. Repeated init/dispose cycles complete and
  the DLL unloads cleanly. The DLL writes its own creation/destruction counters
  to CRT stdout, so the probe intentionally writes JSONL to a separate file.

Run `reference/probe/test.sh` to reproduce these values and verify exact trace
repeatability on the current host.
