# v2.03 normal-mode mechanics

This document describes the implemented headless target: IriSu Syndrome v2.03,
normal mode, one gameplay update per 0.020 seconds. The detailed clean-room
evidence and recovered addresses live in
[`reference/game-rules-analysis.md`](../reference/game-rules-analysis.md).
Presentation assets, the original executable, and original DLLs are never
runtime dependencies of the clone.

## Update order

One API tick corresponds to one replay input record and one original gameplay
update. Fast-forward only skips rendering; it does not multiply physics time.
The order is causal:

1. If gauge was nonpositive at scene entry, call the finish routine. The final
   gameplay update still runs, and a level-cap contact may call finish again.
2. Convert left and right button levels to fresh edges using the previous
   levels. Fire a weak left shot, then a strong right shot.
3. Test the current, preincrement scene counter for a scheduled falling block.
4. Step legacy Box2D r58 once at 0.020 seconds with 10 solver iterations.
5. Traverse every touching contact in native world-contact-list order and run
   the complete normal dispatcher. Persistent contacts run again every update.
6. Clamp gauge, increment the scene counter, apply passive drain, and floor the
   result to one.
7. Update the fixed actor pool in slot order 0 through 199.

The four permanent field actors occupy slots 1 through 4. Dynamic allocation
starts at slot 5, scans round-robin, and has 196 simultaneous dynamic slots.
Normal reset already occupies slots 5 through 24 with 20 seeded blocks, so the
first input-record allocation starts at slot 25. Fresh simultaneous input
therefore allocates weak shot, strong shot, then the scheduled block. A failed
scheduled allocation has already consumed only its size-ticket and X draws.

Reset also reproduces one constructor-frame actor-pool pass that happens before
replay word 0. It is part of `reset()`, not an API tick: visible tick and scene
counter remain zero and gauge remains 3000.

## Input and replay contract

Actions carry button levels, not abstract press commands. A rising left level
creates a 24-by-24 projectile at the exact cursor coordinate with display
velocity `(0,-250)`; a rising right level uses `(0,-500)`. Held levels do not
repeat. The two edges are independent and left is handled first. `left_held`
and `right_held` are observable and snapshotted because they determine whether
the next level is a fresh edge.

The live game client is 640 by 480, so ordinary interactive clicks are inside
`x=0..640`, `y=0..480`. A replay record nevertheless allocates 10 bits to X
and 9 bits to Y, encoding the wider integer ranges `x=0..1023`, `y=0..511`.
The raw headless action contract accepts that full encoded range so replay
button edges are not discarded merely because a stored coordinate is outside
the live client. Nonfinite coordinates or values outside the encoded range are
rejected.

Public body positions use display units. For physics-owned actors,
`Body.velocity` is the raw world-unit float32 returned by the shipped wrapper;
a new weak/strong projectile therefore stores `(0,-25)` or `(0,-50)` after the
wrapper's one-time division by magnification 10. A scripted actor instead keeps
its float32 display-unit `scripted_velocity` in the public actor field after
its integrator. OOB handling zeros actor-visible
`Body.velocity` but deliberately retains hidden `native_velocity`, so it does
not write the native body before the next physics step. Native x/y/angle reads
and every scripted accumulator store are rounded
to float32 at the executable's store boundaries, then widened by the clone API.

The padded v2.03 replay layout stores a 52-byte header followed by four-byte
input records. The clone's conservative parser also recognizes the older
20-byte-header layout, but an old replay is evidence, not a v2.03 golden
trajectory. Playback has one startup exception to ordinary input handling:
after loading raw held levels and updating their history, `Input.update` clears
fresh left and right edges on replay records zero and one. This prevents a
button already held as the scene begins from firing. The replay adapters mark
those two actions as edge-suppressed; normal RL reset/step semantics are
unchanged.

## Mode-0 production profile

Replay header offset `0x10` is the signed mode. The executable propagates it to
the selector at `0x4124c0`: mode 0 loads the normal table at `0x412560`, while
nonzero mode loads the separate Metsu-side table at `0x413648`. The latter
matches `data/doc/irisu.ini`; it is not the mode-0 normal configuration. Every
preserved external replay header in the current corpus explicitly says mode 0.
The clone therefore uses the mode-0 table, with config hash
`0xec0e8463feaf2670`.

## Randomness and spawning

The clone implements the exact DxLib.dll generator: a 624-word MT state,
DxLib's 69069 seed expansion, and inclusive high-product `GetRand(max)` range
mapping. A full RNG state and index are included in snapshots but not policy
observations. Public reset seeds are restricted to the target's unsigned
32-bit domain; wider values are rejected rather than truncated.

Normal initialization consumes `GetRand(100)` for the level-wide shape cutoff
and `GetRand(12)` for the first special threshold. It then creates ten rotten
physics-owned blocks at Y=200 followed by ten scripted falling blocks at Y=60.
Every one of those 20 blocks consumes the five ordinary draws below, in order:

```text
GetRand(98)       size ticket: 20x32, 28x46, 28x54, 14x60,
                  5x72, 3x90, 1x140
GetRand(304)      X offset from field_x=130
GetRand(1000)     rotation * 2*pi/1000
GetRand(C)        color ID, inclusive
GetRand(100)      triangle if roll>T, otherwise box
```

The reset blocks use their constructor Y values; cadence blocks use Y=-50. The
constructor then receives one actor-pool pass before replay word 0. Consequently
all 20 blocks have age 1 and lifetime 99999 at the public reset boundary. Rotten
blocks remain at Y=200 with rot timer 1 and native ownership. Scripted blocks
advance once at level-1 speed to float32 Y=60.2, transition from lifecycle state
1 to 2, and remain non-native. The first cadence spawn is body ID 21.

A due special consumes the first four draws, stores color `-2`, skips the shape
draw, and reschedules at
`qualifying_clears + 40 + GetRand(12)`. Its circle is size 24, density 50,
friction 0.1, restitution 0.6.

The scene counter starts at zero, so the first update spawns. Cadence uses
`counter % I == 0`; level 1 has `I=90`, not 100.

## Level parameters

For active parameter level `L`:

```text
D = floor(L/10)+1
V = .0125*floor(L/7) + .05*(L%7) + .15
if L%6==0: V=1
if L>=20 and (L-20)%18==0: V=2.4
P = 1800 + 20*L
C = min(5, floor(((L%9)+floor(L/15))/3)+2)
I = 100 - 10*(L%10)
if L%13==0: I=4
if V>2 and I<50: I=50
S = 4*pow(L,.7)
reward_unit = 700
T = GetRand(100)
```

`C` is the inclusive maximum color ID. Level 1 therefore has three colors.
The three `V` constants are float32 values widened for the x87 arithmetic, and
the result is stored back to float32. Level 1 therefore has stored
`V=0x3e4ccccd` (approximately `.20`), `P=1820`, `D=1`, `I=90`, and `S=4`.
The score-scale power thunk reverses its apparent call-site operands: level 2
stores `S=6.498019170849884` (`0x4019fdf8bcce533d`), not `4*.7^2`.

Every tenth qualifying burst raises the requested level. Requests below 100
apply the formulas immediately and draw a new `T`, so later contacts, scene
drain, and actor scoring in the same update use the new parameters. A request
of at least 100 calls finish, writes visible level 100, consumes no cutoff
draw, and retains level-99 parameters through the rest of the final update.
The comparison is against the requested candidate, not the clamped current
level: after count 990 reaches level 100, counts 991--999 do nothing, count 1000
requests 101 and calls finish again, and each later qualifying clear whose
candidate is greater than 100 calls finish again.

## Field and physics

The normal world uses display bounds `(0,-200)..(640,480)`, magnification 10,
gravity 160 display units/s², and no damping. The wrapper creates four static
fixtures in left, right, bottom, top order. The mode-0 field is
`(x=130,y=120,width=320,height=250,blank=40,thickness=16)`; the top fixture has
center Y -140, width 320, and height 300, and the bottom is 16 high. Ordinary
boxes and triangles use density 1, friction 1, restitution 0.

A new ordinary block has a dynamic fixture from creation, but its actor has
physics ownership `c4=0`. Box2D still steps it; afterward the actor overwrites
the transform using the `V` captured at spawn. That teleport zeros native
linear velocity while retaining native angular/sleep state. Once `c4=1`, the
actor reads the native transform instead. Existing scripted blocks retain the
speed captured at their spawn level.

## Contact dispatcher

Classes are bottom 2, walls 3, ordinary/special block 4, projectile 5, and top
6. For each native contact `(A,B)`:

1. Reject projectile/projectile and rotten/rotten pairs.
2. Increment `A.e0` when `B.cc!=3`, then `B.e0` when `A.cc!=3`. `e0` is
   monotonic for one actor incarnation.
3. Run the both-scripted newborn cleanup. If both have `c4=0` and equal `f0`,
   each nonspecial class-4 actor of age at most two is immediately torn down;
   the contact then short-circuits.
4. Run top gate in `B<-A`, then `A<-B` order. A scripted target touching class
   6 sets `e4` and short-circuits.
5. Run group pairing once, then activation both ways, special both ways, burst
   both ways, rot start both ways, and direct hit in `B/A` then `A/B` order.

Earlier contacts and earlier handlers mutate state seen later in the same
native traversal. Active-pair keys exist only for begin-contact diagnostics;
they never gate gameplay.

### Groups and bursts

Equal-color contacts require at least one class-4 participant and at least one
ungrouped participant. Each eligible ungrouped class-4 target reuses the
source's group or allocates a zeroed `{chain,shadow,num}` record, increments
`chain` and `shadow`, and becomes grouped. Groups never merge or explicitly
free. Exactly one rotten participant is allowed.

A grouped class-4 target not already pending a successful clear bursts when
the source is bottom or has a nonzero rot timer. It becomes `d5`, increments
group `num`, gains `num*700` gauge when not rotten, increments the global
qualifying count, and may change level immediately. A same-color eligible
source also becomes `d5` without another count or reward in that invocation.
All contacts finish before actor scoring, so every live `d5` block scores with
the final shared group counters:

```text
delta = trunc(0.5 * (group.num+1) * group.chain * S
              * max(1, size_slot-4))
```

Mode-0 normal selects slots 0 through 6. The factor is one for slots 0 through
5 and two for the rare 140-unit slot 6. Each scored block updates highest chain
and is immediately torn down. Immediate
newborn teardown can suppress this path; a normal `+68` deletion marker cannot,
because `d5` scoring runs first.

The seed-41, 520-record mode-0 replay is an exact original-game regression for
this path. Original and clone both score actor 10 then actor 12 at tick 304,
with shared `chain=2`, `num=1`, and deltas `+8,+8`; both finish the replay at
16. The formula was not tuned. Correct mode-0 field, size/weight, gauge, and
lifecycle defaults made the causal board state agree.

### Activation, special, rot, and direct hits

A scripted target activates from a physics-owned non-wall source when the
source is ungrouped, the colors match, or the source is bottom. A projectile
source is marked for deletion. Top contact instead uses the earlier gate. In a
non-new actor postlude, `e5` latches only when both `e4` and `e5` are false;
`e4` is then cleared. A current top contact therefore suppresses the latch, and
a grouped `e5` block becomes physics-owned.

An armed special requires special identity, `e5`, and physics ownership. It
consumes a projectile while persisting. Against an ungrouped nonspecial
class-4 block, it clears every actor of that nonrotten target's color and adds
700 per slot, then consumes itself. A rotten target skips the color clear but
still consumes the special. Grouped targets are a no-op.

Rot starts for a nonspecial class-4/5 target older than 100 when it touches
bottom or a participant with a running rot timer. The timer starts at one and
increments in actor updates. A nonrotten normal class-4 actor becomes rotten
when the timer is strictly greater than 40 and subtracts `P` after scene
drain; Game Over is latched at the next scene entry.

For a class-4/projectile orientation, a projectile whose `e0` is exactly one
increments direct-hit count only on a grouped block. Count two marks both for
unrewarded deletion. Independently, every nonrotten normal block consumes the
projectile; a rotten block does not.

## Actor lifecycle, gauge, and terminal states

Base actor update performs motion, decrements remaining lifetime, increments
age, and sets `+68` exactly when remaining lifetime reaches zero. Boundaries
return after this base layer. For a live class-4/5 actor, the exact remaining
order is:

```text
if c4 && e5 && strictly_out_of_bounds:
    actor_velocity = (0,0)
    remaining_lifetime = 1

if c8 != 1:
    if d5:
        score_this_block()
        tear_down_native_body_and_actor()
        update_highest_chain()
    if delete68:
        emit_delete_effect()
        tear_down_native_body_and_actor()   # null-safe after d5 teardown
        return
    if !e4 && !e5: e5 = 1
    e4 = 0
    if d4 && e5: c4 = 1
else:
    c8 = 2

if f0 == 0 && c8 != 3 && d0 != 0:
    d0 += 1
    if d0 > 40 && cc == 4:
        c8 = 3
        gauge -= P
if c4: f8 += 1
```

Creation-frame actor update counts as update one. Ordinary/special lifetime is
100000 and projectile lifetime is 3000. A new `c8=1` actor changes to `c8=2`
but still runs the rot and `f8` tail. A `d5`-only actor scores and tears down,
then still runs that tail; `+68` suppresses it. When both are set, scoring
happens first and the `+68` branch then returns.

The strict out-of-bounds test is enabled only for physics-owned class-4/5
actors with `e5` and uses `x<0`, `x>640`, `y<-30`, or `y>560`. It zeros actor
velocity and sets remaining lifetime to one; deletion occurs on the next actor
update. A new projectile changes `c8` on update one, latches `e5` on update
two when unobstructed, and can enter the OOB guard from update three.

Scene gauge handling clamps to `[0,40000]`, increments the scene counter, then
subtracts `3D` above half maximum or `D` otherwise. The apparent `5D` branch in
the executable is unreachable. Passive drain floors at one. Rot subtraction
happens later and may leave gauge nonpositive; the next update calls finish but
still completes that one final gameplay update.

Finish is not a latch-once function. Every invocation increments the finish
call count and refreshes the latest in-memory terminal metadata from the live
`score`, `highest_chain`, `level`, and qualifying-clear count. The replay write
gate persists only the first such snapshot. Later contacts and actor scoring
in the same update can therefore produce three distinct facts: first recorded
result metadata, latest finish-call metadata, and final live globals. The
policy observation exposes only the live globals. C++ `StepResult`, C step JSON,
and Python step `info["diagnostics"]` expose the finish count plus first and
latest terminal metadata. Only future gameplay steps are suppressed by the
terminal scene state.

## Snapshots and uncertainty

Schema-7 snapshots contain all causal rule state, the full Dx RNG, previous
button levels, actor slots/cursor, stale colors for all 200 actor slots, groups,
pending native tombstones, Box2D body origin and exact center-of-mass bits,
proxy ordering, broad-phase arrays, contact-list/node order, manifolds, and
warm-start impulses. The separate center
is necessary because r58's rotated-triangle transform is not bit-invertible.
Snapshot restore is supported only for the same mechanics configuration and
build. Regression/property coverage branches from dense contacts, swept
zero-manifold contacts, rotated asymmetric triangles, deferred contact
replacement, swept proxies outside their creation range, frozen proxies with
deferred contacts, divergent actor/native state, and deferred-destruction/proxy
churn; malformed object and wire snapshots are rejected without changing the
live simulator.

Remaining uncertainty is concentrated in long-horizon legacy Box2D numerical
drift, the necessarily finite coverage of rare allocator/contact combinations,
and modes or EX/Metsu states outside normal mode. Placeholder compatibility
knobs are not nominal gameplay formulas and are identified as deprecated in the
profile/configuration metadata.

The retained compatibility-only fields are `cleanup_margin_x/y`,
`floor_contact_tolerance`, `deletion_delay_ticks`, `bonus_interval_spawns`,
`click_cooldown_ticks`, `color_level_stride`, `score_per_level`,
`spawn_acceleration_level_stride`, `size_score_values`, and
`chain_score_exponent`. They are accepted and hashed for API stability but do
not alter the faithful normal rules. The shipped raw keys `a`, `b`, `c`, `d`,
and `p` likewise have no recovered normal-path use and are not runtime inputs.
