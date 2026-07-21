# Normal-mode Rules Recovered from v2.03

This is the implementation ledger for the headless normal-mode clone. It
separates facts recovered directly from the shipped machine code and local DLL
probes from rules that still need a differential trace. Presentation assets and
story behavior are outside this document.

## Evidence contract

The analyzed files are the preserved English-patched v2.03 build:

| File | SHA-256 |
|---|---|
| `irisu.exe` | `0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255` |
| `data/dll/DxLib.dll` | `d8ef638a078a8b4d24b53b174ca179623fed3690027d3f4acfe71a7d61c8b5c9` |
| `data/dll/Box2D.dll` | `34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd` |

Executable addresses below are virtual addresses with image base `0x400000`.
The main code section is VA `0x403000`, file offset `0x1600`; `.CRT$XIA` is VA
`0x432000`, file offset `0x30000`.

- **Proven/static** means the rule is a direct translation of an executable or
  DLL instruction path and its constants.
- **Proven/probe** means a local, read-only behavioral probe of the exact DLL
  confirmed the rule.
- **Strong** means the instructions are known but a class-field name or a
  higher-level state label is inferred from surrounding use.
- **Lead** is not safe to implement as a compatibility rule yet.

## World and update order

### World initialization — proven/static and proven/probe

`ScenePlay` initialization at `0x409467–0x409487` calls:

```text
b2d_init(
    min_x = 0,
    min_y = -200,
    max_x = 640,
    max_y = 480,
    gravity_y = 160,
    magnification = 10)
```

The six constants are at `0x4344c0–0x4344d0`. The last two arguments are not
solver iterations or an INI lookup. The wrapper divides bounds and vertical
gravity by `magnification`, so the legacy Box2D world receives AABB
`(0,-20)..(64,48)` and gravity `(0,16)` world units. Wrapper position getters
multiply by 10. Positive Y is downward. The shipped INI key
`world_magnification=100` is not the argument to this world-construction call;
this constant is independent of the mode-selected parameter table.

The exact triangle probe shows vertices relative to the requested center:

```text
(-width/2, -height/2)
(-width/2,  height/2)
( width/2,  height/2)
```

Box dimensions are full dimensions; the wrapper converts them to half-extents.
The circle argument is a radius.

### Fixed update — proven/static

At `0x40a9e4`, with the call at `0x40aa06–0x40aa49`:

```text
b2d_step(world_step * time_multiplier, 10)
```

`world_step` is the INI value `0.020`; `time_multiplier` is the signed integer
at `0x436a58` and begins at 1. The same multiplier controls scripted actor
integration. There is no randomized simulation cadence.

The normal-frame ordering relevant to a headless implementation is:

1. The game loop calls `Input.update` once (`0x406603`, virtual slot `+0x60`
   resolving to `0x4108a0`).
2. It calls the active scene once at `0x406659`. Within normal gameplay,
   `Field.update` (`0x40550c`) handles a fresh left shot, then a fresh right
   shot, then the cadence falling-block spawn in that exact allocation order
   (`0x405519–0x405605`).
3. `ScenePlay` calls `Field.update` at `0x409a2e`, then invokes the
   physics/contact dispatcher at `0x409a36`; that dispatcher calls
   `b2d_step` at `0x40aa49`.
4. Scene gauge/counter work completes, including the counter increment at
   `0x409d14`.
5. The block pool updates actors (`0x40666d`, dispatching through `0x4031ec`).

A body created by the field therefore participates in physics and receives its
first actor update in the same frame.

The gameplay counter at `0x4428e0` starts at 0. Field spawn modulo reads that
pre-increment value; after physics/contacts and gauge clamping, ScenePlay
increments it at `0x409d14`, performs passive drain, returns, and only then
does the outer loop update the actor pools. Thus the first update observes
counter 0, can spawn, steps that body, advances the counter to 1, drains, and
updates the new actor.

No damping setter exists in the wrapper export surface and no executable-side
damping call was found. Preserve legacy body defaults unless a future trace
proves an internal non-default value.

### Replay cadence and fast-forward -- proven/static

Live input appends exactly one replay word per `Input.update` at `0x410b50`;
replay input consumes exactly one word at `0x410ca4`. Consequently one replay
word represents one scene/physics/actor update with the normal `0.020` step.

Replay startup has a two-record fresh-edge gate. The replay path copies the
current word into `Input+0x88` and increments its index at `0x410d0a`. After
the ordinary edge helper has updated fresh/held/release/previous-held state,
`0x410aa2` tests that incremented index against 3. While it is below 3 and
playback byte `Input+0x98` is set, `0x410ac2–0x410acf` clears only the fresh
left/right bytes at `Input+8` and `Input+0xc`. Thus records 0 and 1 cannot
fire, but their held levels still become history; a button newly pressed on
either startup record remains held rather than becoming a delayed edge on
record 2. This is independent for left and right. The observed header-56
playback begins left-down on records 0 and 1: honoring this gate removes the
clone's spurious tick-1 projectile and reproduces all 11 original score calls
and the score-88 state at replay exhaustion.

The signed `time_multiplier` at `0x436a58` is initialized to 1 and has no write
xrefs in the shipped executable. Its normal-code read xrefs are `0x407826`,
`0x4078c6`, and `0x40aa3b` (plus the unused old-scene duplicate at
`0x40becf`). Fast-forward at `0x40667d–0x4066bc` skips rendering at interval
20; it does not skip input words, batch simulation ticks, or increase physics
`dt`. A replay evaluator must therefore run every recorded word, including
words traversed while the reference UI is fast-forwarding.

## Mode-selected parameter profile

The selector at `0x4124c0` reads the mode state stored at `SaveData+0x1c`:
zero selects the normal initializer at `0x412560`, while nonzero selects the
Metsu-side initializer at `0x413648`. Replay loading reads its 0x34-byte header;
header offset `+0x10` is the signed mode field passed into this state. The
selector is therefore replay-visible, not hidden save provenance. Every
preserved external replay header in the current corpus has mode 0.

The nonzero-mode values exactly match the gameplay values in
`data/doc/irisu.ini`. They are not dead data, but they are not mode-0 normal
defaults.

The supported clone default is the mode-0 normal profile observed in the
fresh-process seed-41 run:

| Parameter | Normal mode 0 | Nonzero mode / Metsu-side INI table |
|---|---:|---:|
| field `(x,y,w,h,blank,thick)` | `(130,120,320,250,40,16)` | `(94,20,420,370,30,16)` |
| top `(y,w,h)` / bottom height | `(-140,320,300)` / `16` | `(-140,450,300)` / `400` |
| sizes | `32,46,54,60,72,90,140,5,5,5` | `30,48,64,10,10,10,10,10,10,10` |
| weights | `20,28,28,14,5,3,1,0,0,0` | `10,40,20,0,0,0,0,0,0,0` |
| block life / strict rot threshold | `100000 / 40` | `10000 / 120` |
| projectile life | `3000` | `1200` |
| gauge maximum / initial | `40000 / 3000` | `10000 / 1000` |
| qualifying clears per level | `10` | `6` |

The shared score formula, level formulas, 700-unit clear reward, projectile
velocities, material constants, and level-dependent rot penalty are unchanged.
The mode-0 native config hash is `0xec0e8463feaf2670`.

This selector explains why a clone initialized from the INI alone simulated a
different board despite using correct rule code. Because every relevant replay
explicitly says mode 0, the INI table is not a valid normal-replay target.

## Static field geometry

The constructor at `0x4045d4` builds four distinct static boxes in this exact
body-creation order: left, right, bottom, top. They are stored at
`Field+0x08`, `+0x0c`, `+0x10`, and `+0x14`; their virtual `+0x68` calls are
at `0x404963`, `0x404c14`, `0x404fac`, and `0x405248`. Block vtable
`0x432390` resolves `+0x68` to `0x403bec`, which calls
`_b2d_create_box@32` exactly once. Thus these are four bodies, not four
fixtures on a shared body. Integer division below is signed truncation toward
zero:

```text
left.center = (
    field_x + trunc(field_thick / 2),
    field_y + trunc(field_height / 2))
left.size = (field_thick, field_height)

right.center = (
    field_x + field_width + field_thick,
    field_y + trunc(field_height / 2))
right.size = (field_thick, field_height)

bottom.center = (
    field_x + trunc(field_width / 2) + field_thick,
    field_y + field_height + field_blank + trunc(field_bottom_h / 2))
bottom.size = (field_width + 2 * field_thick, field_bottom_h)

top.center = (
    field_x + trunc(field_width / 2) + field_thick,
    field_top)
top.size = (field_top_w, field_top_h)
```

The right-wall addition is deliberately asymmetric: it uses the full
`field_thick`, not half of it (`0x40499a–0x404a67`). With the supported mode-0
normal profile, as independently recorded by the fresh-process Box2D proxy:

| Fixture | Center | Full size | Extents | Friction | Restitution |
|---|---:|---:|---:|---:|---:|
| left | `(138,245)` | `16x250` | x `130..146`, y `120..370` | 1.0 | 1.0 |
| right | `(466,245)` | `16x250` | x `458..474`, y `120..370` | 1.0 | 1.0 |
| bottom | `(306,418)` | `352x16` | x `130..482`, y `410..426` | 1.0 | 0.0 |
| top | `(306,-140)` | `320x300` | x `146..466`, y `-290..10` | 1.0 | 0.5 |

All four fixtures have density zero.

## Random number generator

### Exact algorithm — proven/static and proven/probe

The game wrapper at `0x40ffb0` calls `_dx_SRand@4` and resets its diagnostic
call counter at `0x442854`; the wrapper at `0x40ffc4` increments that counter
and calls `_dx_GetRand@4`.

The DLL uses 624 32-bit words and an MT19937 twist with offset 397,
`MATRIX_A=0x9908b0df`, and temper masks `0x9d2c5680` and `0xefc60000`. Its seed
expansion is DxLib-specific:

```text
value = uint32(seed)
for i in 0..623:
    following = uint32(69069 * value + 1)
    state[i] = (value & 0xffff0000) | (following >> 16)
    value = uint32(69069 * following + 1)
```

Range conversion is inclusive high-product scaling, not modulo:

```text
GetRand(maximum) = floor(raw_u32 * (maximum + 1) / 2^32)
```

DLL locations: exports `_dx_GetRand@4` RVA `0x1a60` and `_dx_SRand@4` RVA
`0x1a70`; wrappers/internal routines `0x10005c20` and `0x10005c40`; state seed
storage `0x10143590`; raw/twist/temper code `0x101439a0`, `0x10143650`, and
`0x10143840`.

The redistributable model and measured vectors are in
`reference/rng-oracle/`. The Python output exactly matches the locally held DLL
for seeds `0`, `1`, `0x12345678`, and `0x3fffffff` over the first normal-mode
maxima.

### Seed and first-call order — proven/static

Live play seeds from `GetNowCount` at `0x409845–0x409867`. For signed 32-bit
value `s`, the emitted seed is the sign-preserving reduction:

```text
sign = s >> 31
m = abs_int32(s) & 0x3fffffff
seed = (m ^ sign) - sign
```

Replay loading at `0x410d20` instead takes the first header dword
(`Input+0x4c`) and calls `SRand` at `0x410e0d–0x410e10`.

Starting immediately after either seed, the normal initialization/spawn calls
are:

```text
GetRand(100)       # level shape cutoff
GetRand(12)        # first special threshold offset
repeat 10:          # initial rotten blocks at Y=200
    GetRand(98), GetRand(304), GetRand(1000),
    GetRand(color_max), GetRand(100)
repeat 10:          # initial scripted blocks at Y=60
    GetRand(98), GetRand(304), GetRand(1000),
    GetRand(color_max), GetRand(100)
# replay word 0 cadence spawn then consumes the same five-draw tuple
```

Those maxima are for mode-0 normal. The nonzero-mode/Metsu-side table consumes
the same number of draws but uses `GetRand(69)` and `GetRand(404)` for size and
X.

Thus normal reset consumes 102 outputs before replay word 0. The first cadence
block is dynamic body 21, not body 1.

### Reset prefill and pre-word actor pass — proven/static

`ScenePlay` calls the level-setter thunk at `0x409888–0x409892`, which consumes the
shape-cutoff draw, then constructs `Field` at `0x409892–0x4098a1`. Inside the
field constructor, `0x40468a–0x40468c` calls the special-threshold scheduler at
`0x4045ac`, so the two setup draws precede every block draw.

The normal-mode branch at `0x4053a7–0x4053f7` calls Field virtual slot `+0x34`
(`0x4059dc`, the ordinary generator) ten times with Y=200. After each call it
sets `c8=3`, `d0=1`, `c4=1`, actor `+0x55=1`, and `e5=1`, producing a rotten,
physics-owned block. The following loop at `0x4053f9–0x40540c` calls the same
generator ten times with Y=60 and leaves the ordinary scripted initialization
intact. With no special due at clear count zero, each call consumes size, X,
rotation, color, and shape draws.

Construction occurs during the active-scene call at `0x406659`. The outer game
loop then updates the actor pool at `0x40666d` before its next input update at
`0x406603`. This is one actor pass without a scene or physics update. At the
replay-word-0 boundary all 20 mode-0 normal blocks therefore have age 1 and
remaining lifetime 99999. Rotten blocks remain at Y=200 with `d0=1`;
scripted blocks have moved once to float32 Y=60.2, changed `c8=1` to `c8=2`, and retain `c4=0` and
`e5=0`. Scene counter, score, and spawn count are still zero, and gauge is
still 3000. This constructor/pre-word behavior is direct binary evidence; a
fresh-process, one-record seed-123 replay capture independently corroborated
the populated starting board but is not used to infer the rule.

## Level parameters

The normal level setter is `0x415624–0x4157fd`. For positive, one-based level
`L`, it writes:

```text
D = trunc(L / 10) + 1

V = 0.0125 * trunc(L / 7) + 0.05 * (L % 7) + 0.15
if L % 6 == 0:
    V = 1.0
if L >= 20 and (L - 20) % 18 == 0:
    V = 2.4                  # this later test overrides the %6 value

P = 1800 + 20 * L
C = min(5, trunc(((L % 9) + trunc(L / 15)) / 3) + 2)
I = 100 - 10 * (L % 10)
if L % 13 == 0:
    I = 4
if V > 2.0 and I < 50:
    I = 50

reward_unit = 700
S = 4.0 * pow(L, 0.7)
T = GetRand(100)
```

| Symbol | Executable storage | Meaning |
|---|---:|---|
| `D` | `0x436a10` | passive gauge drain unit |
| `V` | `0x436a14` | scripted pre-activation descent per update |
| `P` | `0x436a1c` | rot gauge penalty |
| `C` | `0x436a20` | maximum color ID, inclusive |
| `I` | `0x436a24` | normal spawn interval in scene frames |
| `S` | `0x436a30` | score scale, double |
| reward | `0x436a38` | normal clear reward unit |
| `T` | `0x436a3c` | level-wide box/triangle cutoff |

Thus level 1 has color IDs `0..2`, three colors, rather than two. The embedded
name `level_table.txt` is not consulted by this normal path; these formulas are
hard-coded.

The operand order of the power call is easy to misread from the caller alone.
At `0x415795–0x4157b1`, the setter converts `L` to a TBYTE stack value and
pushes the TBYTE constant at `0x439160`, whose bytes are
`33 33 33 33 33 33 33 b3 fe 3f` (extended-precision `0.7`). The compiler thunk
at `0x41a5e0` reconstructs those values for the internal runtime routine at
`0x427110` with `L` as the first operand and `0.7` as the second. Its positive,
finite path reaches `0x42baa5`, which loads the second operand, then the first,
and executes `fyl2x`; the exponentiation is therefore `L^0.7`, not `0.7^L`.
That result is multiplied by the TBYTE `4.0` at `0x4157b6–0x4157bc` and rounded
once to binary64 by `fstp QWORD [0x436a30]` at `0x4157be`.

A disposable Wine-GDB run independently invoked this setter and stopped just
after that final store for every supported nonterminal level. Representative
results are level 1 `0x4010000000000000`, level 2
`0x4019fdf8bcce533d` (`6.4980191708498838`), and level 99
`0x4058f15933c24051` (`99.771069469164345`). The full authoritative dump is
[`score-scale-binary64.tsv`](score-scale-binary64.tsv). On the current host,
`4.0 * std::pow(double(L), 0.7)` differs from that dump at 83 of 99 levels,
usually by one binary64 ULP and sometimes by two; it is not replay-exact.
Use the dumped bits (or reproduce the executable's internal x87 routine), not
host `std::pow`.

The level speed expression is likewise stored through a final `fstp DWORD`.
Live setter checks give `V` bits `0x3eb33334` at level 4 and `0x3e266667` at
level 7; decimal-double evaluation followed only by a later conversion can
miss these float32 results.

### Level progression — proven/static, state labels strong

The current level at `0x4428c8` starts at 1 (`0x4094e2`). The qualifying-clear
count at `0x4428cc` starts at 0 except when resume logic supplies prior state.

At `0x40a510`, after two virtual eligibility predicates, the group count at
`group+8` increments (`0x40a553`). If `Block+0xcc == 4`, the qualifying-clear
count increments (`0x40a56b`). The executable reads the selected mode table's
deliberately misspelled `statge_level_norma` field at
`0x40a571–0x40a5b3`. Let `Q` be that value: `Q=10` in mode-0 normal and `Q=6`
in the nonzero-mode/Metsu-side table:

```text
candidate_level = floor(qualifying_clear_count / Q) + 1
if candidate_level > current_level:
    set_level(candidate_level)
```

The two predicates are decoded in the contact section below; a raw contact is
not itself a qualifying clear.

### Nonterminal level commits re-enter `Field.update` -- proven/static and live

For an ordinary request below 100, `ScenePlay.set_level` deliberately performs
more than a parameter assignment. Its exact order is:

```text
current_level = requested                          # 0x40af7c
level_display.set_level(requested)                 # 0x40af82-0x40af88
Field.update()                                     # 0x40af8b-0x40af94
install_normal_level_parameters(requested)         # 0x40af97-0x40af99
```

A live breakpoint resolved `ScenePlay+0x1c` to vtable `0x433580`; virtual
`+0x54` is `0x406dbc`. That method only stores the displayed level at object
offset `+0xbc` and rebuilds its decimal digits through `0x415538`. The object
at `ScenePlay+0x10` instead resolves to the `Field` vtable `0x432e40`;
virtual `+0x20` is the state dispatcher `0x4054c8`, whose ordinary-state
target is the complete `Field.update` at `0x40550c`.

This call occurs synchronously inside the qualifying-contact handler. It
re-reads both input buttons at `0x405519-0x4055e4` and can therefore fire the
same replay input a second time. It also repeats the normal cadence test at
`0x4055e7-0x405605`. At this point the public level global already contains
the requested level, but all level-parameter globals--including descent,
color range, spawn interval, score scale, and shape cutoff--still contain the
previous level's values. The new setter, including its one `GetRand(100)`
shape-cutoff draw, runs only after this nested field update returns.

The full replay trace demonstrates the otherwise surprising consequence at
physics step 32745. Its qualifying contact call 3 commits level 27 while the
scene counter remains 32744 and the old level-26 interval remains 4. The
nested cadence check succeeds a second time and creates ordinary triangle
ordinal 1644 (`size=60`, `x=137`, `y=-50`) between contact calls 3 and 4. The
normal field update in that scene had already created ordinal 1643 at the same
counter. This is a general re-entrant update rule, not a level-27 special case;
it produces a second spawn only when the unchanged scene counter is divisible
by the previous interval (and may also repeat other `Field.update` side
effects).

### Level 100 is terminal -- proven/static

`ScenePlay.set_level` at `0x40af48` signed-compares the request with 100. For a
request greater than or equal to 100 it:

```text
current_level = 100                         # 0x40af55
global_game_state.virtual_64(true)          # 0x40af5f–0x40af68
this.virtual_68()  # resolves to 0x40adc4    # 0x40af6b–0x40af71
```

`0x40adc4` performs the ending/result setup and synchronously stores
`ScenePlay+0x30 = 1` at `0x40ae2e`. This does not unwind the active gameplay
method: the current native contact traversal, gauge clamp, gameplay-counter
increment at `0x409d14`, passive drain, and later actor-pool pass all finish.
On the next scene update, the state switch at `0x4093bc–0x4093df` calls postgame method
`0x40a018`, which increments the result timer at `ScenePlay+0x34`; it does not
call the normal gameplay/physics method `0x40992c` again.

In the live-result branch (`ScenePlay+0x0c==0`), finish snapshots result
metadata before that remaining work. At
`0x40ae41–0x40ae63` it passes the current score, highest chain, level, and
qualifying-clear count to the global result manager. At `0x40af0f–0x40af37`
it passes the current score/highest/level and seed to Input virtual `+0x6c`;
target `0x410edc` copies them into replay fields and finalizes the replay via
`0x410f60`. (`ScenePlay+0x0c!=0` skips this persistence/re-recording block.) A
`d5` actor scored later in the same outer update therefore
changes the live score/highest globals but is excluded from the persisted
result snapshot and replay header. Some postgame selection branches later
read the live globals, so an exact headless model should retain both the
finish-time recorded values and the end-of-update live values.

`finish` has no entry guard on `ScenePlay+0x30`, and its live-result block has
no guard around the result-manager call. A rare frame can therefore enter it
more than once. For example, `0x409939` can finish for a nonpositive entry
gauge before Field/physics/contact processing, then a qualifying contact can
raise the level to 100 and call it again. Each invocation selects ending audio,
stores `ScenePlay+0x30 = 1`, and makes a fresh result-manager insertion attempt.
The manager is a ranked list, not a single terminal snapshot: `0x408170`
inserts only ahead of a strictly lower score, so two equal-score calls can
produce adjacent records when the first insertion was above the last slot.

Replay persistence has a different double-call result. Input `0x410edc`
unconditionally overwrites its in-memory header fields at `Input+0x4c` through
`+0x5c`, but gates file writing on `Input+0x48`. The first call writes through
`0x410f60` and clears that byte at `0x410f3f`; later calls overwrite the memory
fields but cannot rewrite the already-finalized replay file. Thus the first
finish wins for the replay file, while the last call wins for live scene state,
ending audio, and Input's in-memory metadata. In the gauge-then-level example,
the persisted replay header is the pre-contact gauge ending, while the live
scene finishes at level 100.

The level caller can repeat beyond the first cap crossing. `0x40a5b8` calls
`set_level` whenever unsigned `floor(qualifying_count / Q) + 1` is greater than
the current level. After the first level-100 call, the current level remains
clamped to 100. In mode-0 normal count 990 reaches level 100 and count 1000
first requests 101; in the nonzero-mode/Metsu-side table those counts are 594
and 600. Every later qualifying contact whose candidate is above 100 calls
`set_level`/`finish` again. Multiple such calls can occur in one native contact
traversal.

This terminal branch also bypasses the ordinary `<100` level-parameter path at
`0x40af7c–0x40afb9`: it does not call `0x415600` and does not consume the
per-level `GetRand(100)` shape-cutoff draw for level 100.

Therefore the headless environment should complete the step in which level
100 is reached and return terminal. It must not continue gameplay at level
100 while the reference UI's postgame timer/ending transition runs.

## Normal block generation

### Cadence and size — proven/static

The scene frame counter at `0x4428e0` starts at 0 (`0x40948c`). `Field.update`
tests `counter % I == 0` at `0x4055e7–0x405605`; the counter increments only
later at `0x409d14`. The first gameplay update therefore spawns immediately.
There is no random cadence jitter.

The weight builder `0x41207c`, called at `0x412268–0x412295`, expands each
configured slot into repeated ticket entries. Spawn code at `0x4059dc` calls
`GetRand(total_weight - 1)` and uses the resulting entry for both full size and
the original slot index. Mode-0 normal weights are:

```text
20 tickets -> size 32, slot 0
28 tickets -> size 46, slot 1
28 tickets -> size 54, slot 2
14 tickets -> size 60, slot 3
 5 tickets -> size 72, slot 4
 3 tickets -> size 90, slot 5
 1 ticket  -> size 140, slot 6
```

The mode-0 normal total is 99, so the first draw is `GetRand(98)`; slots 7
through 9 have zero weight. The nonzero-mode/Metsu-side table has the 70-ticket
`10x30,40x48,20x64` distribution and uses `GetRand(69)`.

### Position, rotation, color, and shape — proven/static

In normal mode (`game_mode == 0`) at `0x405b3e–0x405bf3`:

```text
x = field_x + GetRand(field_width - field_thick)
  = 130 + GetRand(304)             # mode-0 centers 130..434 inclusive
y = -50                            # exact caller value at 0x405c73–0x405c77
rotation = GetRand(1000) * 2*pi/1000
color = GetRand(C)                 # C is an inclusive maximum
R = GetRand(100)
shape = triangle if R > T else box
```

The rotation draw is inclusive, so both 0 and an equivalent `2*pi` encoding
are possible. `T` is one draw per level; `R` is one draw per ordinary block.
Conditional on a fixed cutoff, the nominal probabilities are
`P(triangle)=(100-T)/101` and `P(box)=(T+1)/101`.

The angle arithmetic specifically uses x87 with the live PC53 control word,
followed by a single-precision store (`0x405e82–0x405e96`): `fild` the ticket,
multiply by the 80-bit value at `0x432ce0`, divide by the 80-bit value at
`0x432cec`, then `fstp float Block+0x24`. The constant bytes and exact values
are:

```text
0x432ce0: 35 c2 68 21 a2 da 0f c9 01 40
           = 0x1.921fb54442d1846ap+2L  # 2*pi
0x432cec: 00 00 00 00 00 00 00 fa 08 40
           = 1000.0L
```

An equivalent host expression is multiply-then-divide in `long double`, cast
to `float`, and only then widen if the host model stores angles as `double`.

Normal box and triangle fixtures use density 1, friction 1, and restitution 0.
They use the selected size for both full width and full height. There are no
ordinary normal-mode circles.

### Scripted descent and physics activation — proven/static, trigger labels strong

The actor constructor `0x4075e8` initializes linear velocity and acceleration
to zero. Box/triangle creation (`0x403b08`, `0x403bec`) sends those current
values through `b2d_set_v` at `0x403b86–0x403b92` and
`0x403c68–0x403c74`, so a normal body starts at `(0,0)` velocity.

After fixture creation, normal spawn code sets `Block+0xc4 = 0` at `0x4062a4`
and actor `vy = V` at `0x4062b4–0x4062ba`. Block motion at `0x403814` has two
ownership modes:

```text
if Block.c4 == 0:
    repeat time_multiplier times:
        vx += ax
        vy += ay
        x += vx
        y += vy
        angular_v += angular_a
        rotation += angular_v
    b2d_set_position(body, x, y, rotation)
else:
    x = b2d_get_x(body)
    y = b2d_get_y(body)
    rotation = b2d_get_r(body)
    (vx, vy) = b2d_get_v(body)
```

The actor integrator is `0x40781c`; the scripted transform write is
`0x40383c–0x403850`; body reads dispatch through `0x4037cc`. Because
`b2d_set_position` zeros linear velocity, an unactivated block descends exactly
`V * time_multiplier` display pixels per normal update, with no acceleration.
The same-frame gravity step is overwritten by the later scripted transform.

### Actor/Box2D numeric boundary — proven/static

The DLL exports real single-precision values, and the executable keeps the
actor transform single precision as well. `b2d_get_x` at DLL `0x10013140` and
`b2d_get_y` at `0x10013170` obtain a two-float origin position, multiply the
selected float by the float magnification, store the product through an
`fstp DWORD`, then reload that rounded float as the x87 return value.
`b2d_get_r` at `0x100131a0` delegates to `0x10013620`, which loads the body
angle from a DWORD. The executable caller `0x4037cc–0x4037fb` stores all three
returns through `fstp DWORD` into `Block+0x0c`, `+0x10`, and `+0x24`.

Consequently a physics-owned block's actor-visible `x`, `y`, and angle are
float32 values after every actor update. A host implementation must preserve
the DLL's additional pixel-coordinate rounding:

```text
actor_x = widen_to_host(float32(float32(world_x) * float32(magnification)))
actor_y = widen_to_host(float32(float32(world_y) * float32(magnification)))
actor_r = widen_to_host(float32(world_angle))
```

Computing `double(float_world) * double(magnification)` without the float32
product store is observably different; for example, float32 `0.1` times 10
rounds to float32 `1.0` in the DLL but yields approximately
`1.0000000149011612` under the unrounded double expression. Box2D's internal
dynamic state is already float32, so the extra x/y rounding is needed for
actor rules, snapshots, and observations, not as a transform write-back on
every dynamic tick.

Script-owned transforms do feed back. Actor integrator
`0x407832–0x407865` stores each velocity, position, angular-velocity, and angle
addition through `fstp DWORD` on every time-multiplier iteration. Its result is
then passed to `b2d_set_position`. A clone using double actor fields must
therefore float32-round every scripted addition, rather than accumulating the
level descent in double precision.

`b2d_set_position` zeros the native body's linear velocity, but this scripted
branch does not call `b2d_get_v` afterward. Consequently `Actor+0x14/+0x18`
retain the scripted float32 velocity (normally `(0,V)`) at the end of the actor
update even though the native body's velocity is `(0,0)`. Actor-facing
observations must preserve that distinction.

Velocity has an intentionally different unit boundary. `b2d_get_v` at DLL
`0x100131b0–0x100131d5` copies the body's two raw world-unit float values to
the caller-provided float pointers; it does not multiply by magnification.
Executable caller `0x4037fe–0x40380c` writes those values directly to
`Actor+0x14/+0x18`. A host sync that applies the position `to_pixels`
conversion to returned velocity is incorrect by the magnification factor.

In normal mode, contact processing changes `Block+0xc4` to 1 through the
activation store at `0x40a86f` or the delayed `d4 && e5` store at `0x4036a8`.
The stores at `0x40a3d2` and `0x40a467` belong to the `f0==1` EX-mode group
branch and must not be copied into normal mode. Once `c4` becomes 1, Box2D
owns motion and gravity/contact response applies. The exact normal predicates
are decoded below.

### Special orb schedule — proven/static

The field constructor schedules:

```text
special_at = qualifying_clear_count + 40 + GetRand(12)
```

at `0x4045ac–0x40468c`. The same scheduler runs again at `0x406033` after a
special is created. Therefore the initial threshold is 40 through 52 inclusive.

The chosen spawn consumes size, X, rotation, and color draws before it is
converted to a special. Its circle uses radius `special_block_size/2 = 12`,
density 50, friction 0.1, and restitution 0.6. Spawn code marks
`Block+0xc5=1` and `Block+0xbc=-2`, reschedules with `GetRand(12)`, and skips
the ordinary `GetRand(100)` shape draw. This is at `0x405ec7–0x406046`.

Pool exhaustion splits that draw sequence. The size-ticket draw
(`0x4059f9`/`0x405a12`) and normal X draw (`0x405beb`) occur before allocator
`0x4030dc` is called at `0x405e49`. A null result branches directly from
`0x405e52` to the epilogue at `0x406321`. Rotation (`0x405e74`), color
(`0x405eb1`), the due-special comparison (`0x405ec7`), and the ordinary shape
draw (`0x406046`) all occur only after successful allocation. Therefore a
failed due-special attempt consumes exactly the size and X draws, leaves the
current special threshold unchanged, and does not consume the scheduler's
`GetRand(12)`. The due special is retried at the next spawn cadence. There is
no separate spawn-attempt or successful-spawn counter; the caller ignores the
null return, while the ordinary scene frame counter still advances later in
the update.

## Shots

The input virtuals at `0x410764` return the two four-byte button edge-state
records at `Input+8` and `Input+0xc`. Live input at `0x410c24` maps DxLib mouse
bit 1 to left and bit 2 to right. The edge helper at `0x410ba8` stores fresh
press, held, release, and previous-held in bytes 0 through 3.

`Field.update` tests byte 0 only:

| Fresh edge | Initial display velocity | Texture role |
|---|---:|---|
| left | `(0,-250)` | slow ball (`sball`) |
| right | `(0,-500)` | fast ball (`fball`) |

The two branches are independent. Simultaneous fresh presses create the left
shot first and then the right shot. Holding does not repeat, and no separate
cooldown variable or check exists on this path.

Projectile creation at `0x40632c` uses the exact mouse X/Y with no offset. It
creates a `24x24` box, not a circle, with density 8, friction 1, restitution 0,
and mode-0 normal remaining lifetime 3000. The nonzero-mode table uses 1200.
The initial velocity is supplied before fixture
creation, so the wrapper divides it by magnification 10 to Box2D velocities
`(0,-25)` or `(0,-50)`. This conversion is reflected into the actor before
creation returns: box creation sends `Actor+0x14/+0x18` through `b2d_set_v` at
`0x403c68–0x403c74`, sets `c4=1`, then immediately calls virtual `+0x58` at
`0x403c90–0x403c94`. Its `0x4037cc` target calls `b2d_get_v` and overwrites the
actor fields with the raw world values at `0x4037fe–0x40380c`.

No contact-rule handler in dispatcher range `0x40a2a4–0x40a9d7` reads the
actor velocity fields. Therefore temporarily retaining the display velocity
inside an otherwise atomic host step does not change creation-frame contact
rules, but actor observations at the end of that step must expose the raw
world velocity, not a value multiplied back by magnification.

## Normal contact and group state machine

This section is **proven/static** for normal mode (`f0==0`). The descriptive
names are chosen for the clone; the offsets and branch effects are literal.

### Block state used by contacts

The constructor at `0x403338–0x4033c3` establishes the relevant defaults:

| Offset | Default | Recovered role |
|---|---:|---|
| `bc` | `-1` | color ID; ordinary spawns overwrite it, special is `-2` |
| `c0` | 0 | size/score-factor slot |
| `c4` | 0 | scripted transform (0) versus Box2D transform (1) |
| `c5` | 0 | special-orb flag |
| `c8` | 1 | lifecycle state: new 1, fresh 2, rotten 3 |
| `cc` | 1 | collision role/type |
| `d0` | 0 | rot timer |
| `d4` | 0 | confirmed group membership |
| `d5` | 0 | successful-clear pending |
| `d8` | null | shared 12-byte `Block.CHAIN` record |
| `dc` | 0 | direct projectile-hit count |
| `e0` | 0 | per-incarnation eligible-contact traversal count |
| `e4` | 0 | top contact seen in this actor interval |
| `e5` | 0 | latched after an actor update clear of the top gate |
| `f0` | 0 | normal (0) versus EX/Metsu (1) rules |
| `f8` | 0 | actor updates spent with `c4==1` |

Normal collision-role values are bottom `cc=2`, side walls `cc=3`, falling
ordinary/special blocks `cc=4`, projectiles `cc=5`, and top `cc=6`. Their
stores are at `0x404fb7`, `0x404973`/`0x404c1a`, `0x4062c8`, `0x4064c9`, and
`0x40524e`, respectively.

### Native contact traversal -- order is observable

`ScenePlay` vtable `0x434820` maps contact slots `+0x38..+0x64` to
`0x40a2a4..0x40a9e4`. After `b2d_step`, `0x40a9e4` consumes the wrapper's
native contact list sequentially. For each returned pair `(A,B)`, the exact
gameplay order is:

```text
if A.cc == 5 and B.cc == 5:
    continue                         # 0x40aa90–0x40aaa2
if A.c8 == 3 and B.c8 == 3:
    continue                         # 0x40aaa8–0x40aabf

if B.cc != 3: A.e0 += 1             # 0x40aac5, helper 0x40ada8
if A.cc != 3: B.e0 += 1             # 0x40aad1, helper 0x40ada8

if both_scripted(B,A): continue     # virtual +0x60
if both_scripted(A,B): continue
if top_gate(target=B, source=A): continue   # virtual +0x50
if top_gate(target=A, source=B): continue

group_pair(A,B)                     # +0x38 once; internally both directions
activate(target=A, source=B)        # +0x58
activate(target=B, source=A)
special(special=A, other=B)         # +0x54
special(special=B, other=A)
burst_pair(A,B)                     # +0x48 once; internally both directions
start_rot(target=A, source=B)       # +0x4c
start_rot(target=B, source=A)
direct_hit(block=B, projectile=A)   # +0x5c
direct_hit(block=A, projectile=B)
```

Only `both_scripted` and `top_gate` return a short-circuit boolean. Later
handlers continue even after setting deletion flags. State mutations from one
native contact entry are therefore visible to every later entry in the same
step. Preserve the legacy Box2D contact-list order; sorting pairs changes
first-hit and reward behavior.

The DLL keeps a global contact cursor at `0x1002bb8c`. `b2d_get_contact`
advances it to `contact->next` before returning the current pair
(`0x1001311e–0x10013129`). DLL body destruction calls world routine
`0x10009c00`, which unlinks and destroy-queues the body but does not
immediately unlink/free its contacts. Consequently later entries in the same
step can still dispatch a Block already pool-dead from an earlier handler.
There is no ScenePlay dead-flag filter; a clone must not prune the remaining
contact traversal after `+0x1c` or `+68` is set.

The Block pool preserves native `userData` lifetime. Allocator `0x4030dc`
increments allocation cursor `0x442628` before probing, wraps at the array
length, finds a slot with `Actor+0x54==1`, flips it back to live at `0x403177`,
and returns the same D object. The cursor remains on the returned physical
slot. Starting from reset value 0, allocation order is slots
`1,2,...,199,0`; reuse continues round-robin from the last allocation. Each caller then
invokes virtual `+0x18 = 0x403338`, resetting all Block fields for the new
incarnation (ordinary spawn `0x405e49–0x405e5c`, projectile
`0x406336–0x40634e`). Pool update `0x4031ec` skips dead objects but does not
compact or free them. No Block is allocated inside contact dispatch, so a
dead object's address cannot be reused during the remaining traversal. On the
next update Field may reuse the slot before physics; native `Step` processes
the queued world destruction and the wrapper captures the new contact-list
head only after `Step` returns (`0x10013050–0x10013074`). Old contacts are
therefore gone before they could be exposed with the reused object's state.

The pool contains exactly 200 slots (`0x409426–0x40942b`), including the four
static boundary Blocks. Those are the first four allocations and permanently
occupy physical slots 1 through 4; the first dynamic Block uses slot 5.
Accordingly the gameplay ceiling is 196 simultaneous falling Blocks and
projectiles, with initial dynamic fill order `5,6,...,199,0`. Expansion control
`0x442630` remains zero, so an all-live scan sets fail latch `0x442634` and
returns null; later allocation attempts in that actor interval fail fast.
Pool update clears the latch, not the cursor. It always visits physical slots
`0..199` in backing-array order (`0x4031f8–0x403226`). There is no physical
slot index in Block (`c0` is the size ticket), so scoring and destroy-queue
order follow physical slot order rather than spawn chronology. Full pool
teardown resets the cursor to 0 at `0x4032bc`.

`e0` is not an active-contact count. Constructor zeroing at `0x403399` and the
increment at `0x40adb9` are its only writes. It is never reset or decremented
during a live Block incarnation, then returns to zero when the reused pool
object runs `0x403338` for its next incarnation. A side wall does not increment
the opposite object's counter. Projectile-projectile and rotten-rotten pairs
are skipped before either counter increment.

### Two scripted bodies -- `0x40a958`

For the normal 0/1 ownership values:

```text
if A.c4 == 0 and B.c4 == 0 and A.f0 == B.f0:
    for X in (A,B):
        if unsigned(X.age) <= 2 and not X.c5 and X.cc == 4:
            X.virtual_1c()
    return true
return false
```

Block vptr `0x432390`, slot `+0x1c`, resolves to `0x403e78`. This is real
destruction, not merely contact suppression: it destroys and clears the Box2D
body, calls actor teardown `0x407708`, and sets pool-dead `Actor+0x54=1` at
`0x407726`. The handler returns true and suppresses every later handler even
when neither participant met the young/ordinary deletion sub-predicate.

### New-to-fresh and the top gate

Physics runs before the actor pool. A normal spawn therefore enters its first
contact traversal as `c8==1`. Its first Block update calls the base actor
update, then bypasses post-contact cleanup because `c8==1`, stores `c8=2` at
`0x403479`, sets the visual-dirty flag, and refreshes its visual
(`0x403455–0x40348b`). Contacts see state 1 as nonrotten. A `d5`, `+68`, or
`e4` set during creation-frame physics remains pending until a later actor
update can run `0x403618`.

Normal `top_gate` at `0x40a6f0` is:

```text
if target.c4 == 0 and target.f0 == 0 and source.cc == 6:
    target.e4 = 1
    return true
return false
```

Because it short-circuits the pair, top contact does not directly run group or
activation handlers. In actor post-contact cleanup (`0x40366c–0x4036af`):

```text
if not e4 and not e5:
    e5 = 1
    refresh_visual()
e4 = 0
if d4 and e5:
    c4 = 1
```

Thus `e4` suppresses the `e5` latch while a scripted block overlaps the top;
`e5` becomes true on the first post-contact actor update with no top contact.
If `c8==1` skipped cleanup while `e4` was set, that stale `e4` delays the latch
by one additional actor update.

Projectile creation `0x40632c–0x40654d` starts from constructor `c8=1,e5=0`,
makes a box (which sets `c4=1`), writes `cc=5`, and never overrides `c8` or
`e4/e5`. Its first actor update only transitions `c8` from 1 to 2. On its
second actor update, cleanup sees `e4==0` and latches `e5=1`; the out-of-bounds
guard is consequently enabled starting with its third actor update.

### Same-color grouping -- `0x40a2a4` / `0x40a35c`

A pair is eligible when colors are equal, at least one participant has
`cc==4`, and they are not both already grouped. Both-rotten pairs are rejected
(and already skipped by the dispatcher); normal mode allows exactly one
rotten participant to group with one nonrotten participant.

The directional helper is:

```text
add_to_group(target, source):
    if target.d4 or target.cc != 4:
        return 0
    if source.d4:
        assert source.d8 != null
        target.d8 = source.d8
    else:
        target.d8 = gc_new_zeroed_Block_CHAIN()
    target.d8[0] += 1               # chain
    target.d8[1] += 1               # shadow/unknown
    target.d4 = 1
    return target.d8[0]
```

The pair routine calls this in both directions and returns their maximum. Two
ungrouped ordinary blocks therefore create one record and finish with chain 2.
An established group can absorb an ungrouped block, but two established
groups never merge because the pair eligibility check rejects `d4 && d4`.
Normal grouping itself does not change `c4`; `0x40a3d2` and `0x40a467` are
EX-only stores.

The record is a zeroed D GC allocation at `0x40a423–0x40a435`. Its source
type name survives as `Block.CHAIN` at `0x4393fc`; TypeInfo `0x4393d0` gives
size `0x0c`, and initializer `0x442638` is all zero. Static xref
audit finds `group[1]` incremented alongside chain at `0x40a3bf–0x40a3c5` and
`0x40a438–0x40a446`, but no normal gameplay read. It must be preserved as an
unknown shadow counter, not assigned invented semantics. Block destruction at
`0x403e78` neither decrements the record nor clears `d8`; there is no explicit
group free. The GC owns its lifetime.

The allocator is D runtime `_d_newarrayT` at `0x419080`, called through array
TypeInfo `0x439570` for one `Block.CHAIN`; its initial contents are
`{0,0,0}`. Records are not pooled and have no fixed gameplay cap.
Dead pool Blocks retain their `d8` reference until reuse runs the constructor;
the GC may reclaim/reuse a record only after every such reference is gone.
Accordingly a surviving Block's `d8` cannot alias recycled group storage.

### Physics activation -- `0x40a7d8`

For a normal scripted target:

```text
if target.c4 == 0 and source.c4 == 1 and source.cc != 3 and
   (not source.d4 or source.bc == target.bc or source.cc == 2):
    target.c4 = 1
    if shared_sound_flag == -1:
        shared_sound_flag = 0
    if source.cc == 5:
        source.delete68 = 1
```

The handler runs after grouping, so a same-color active source can first share
its group with the target and then activate it. Side walls never activate;
the top was intercepted earlier. A projectile activation consumes that
projectile through `+68`.

### Special orb -- `0x40a748`

The directional special handler requires
`special.c5 && special.e5 && special.c4`.

```text
if other.cc == 5:
    shared_sound_flag = 0
    other.delete68 = 1
elif not other.c5 and other.cc == 4 and not other.d4:
    if other.c8 != 3:
        clear_color(other.bc)        # 0x4032d0
        shared_sound_flag = 1
    special.delete68 = 1
```

Thus a projectile is consumed while the special remains. A grouped ordinary
block does nothing. A rotten ungrouped ordinary block consumes the special but
does not invoke `clear_color` and is not deleted by this handler.

`clear_color` iterates the complete 200-slot actor pool, not merely its live
members. Its only per-slot selection predicate is `Block+0xbc == color`
(`0x4032f2-0x4032fa`). A match sets `Block+0x68=1`; in normal mode it then
adds the 700-unit reward to gauge at `0x403311-0x403317`. It does not test the
pool-dead flag at `Actor+0x54`, kind `+0xcc`, special flag `+0xc5`, rotten
state `+0xc8`, or the existing deletion flag `+0x68`. Dead slots retain their
last color, so they are rewarded even though they have no native body left to
destroy. A fresh trigger also rewards matching rotten blocks, and two calls
before actor cleanup can reward an already-flagged block again.

A live Wine-GDB probe broke at function entry `0x4032d0`, each selected slot
at `0x4032fc`, and function exit `0x403324`. It produced these authoritative
counts on the full seed-41 replay:

| Tick | Color | Matches | Live | Pool-dead | Gauge before | Gauge after |
|---:|---:|---:|---:|---:|---:|---:|
| 12260 | 0 | 32 | 2 | 30 | 27140 | 49540 |
| 29665 | 3 | 19 | 6 | 13 | 15335 | 28635 |
| 34667 | 2 | 40 | 6 | 34 | 19539 | 47539 |
| 41847 | 0 | 19 | 11 | 8 | 5752 | 19052 |

Thus this path does not assign the gauge maximum. For example, tick 12260
performs 32 separate `+700` operations and temporarily reaches 49540; the
ordinary scene clamp later reduces that to 40000 before passive drain. The
reproducible breakpoint recipe and machine-readable results are in
[`README-special-gauge-probe.md`](runs/replay-41449-full-event-gdb-20260720-004/README-special-gauge-probe.md).

### Landing/burst confirmation -- `0x40a4a4`, `0x40a4d8`, `0x40a510`

The normal source and target predicates are:

```text
landing_source(source) = source.d0 > 0 or source.cc == 2
clear_target(target) = target.d4 and target.cc == 4 and not target.d5
```

The directional successful-clear routine is:

```text
if clear_target(target) and landing_source(source):
    target.set_d5(true)              # virtual +0x70 = 0x403ea4
    target.d8[2] += 1                # num
    gauge_hook(target)               # fresh: gauge += current num * 700
    qualifying_clear_count += 1
    maybe_raise_level()
    shared_burst_sound = 0

    if target.bc == source.bc and clear_target(source):
        source.set_d5(true)           # no num/gauge/level-count increment
```

The `+0x48` wrapper invokes this first as `(source=B,target=A)` and then in the
reverse direction. The secondary same-color mark occurs only inside a
successful primary branch. It is therefore incorrect to increment `num`,
gauge, or the level counter once for every member that ends with `d5==1`.

The gauge hook at `0x4098fc` reads the just-incremented `group[2]`: primary
fresh marks add 700, then 1400, then 2100, and so on. A rotten primary still
increments `num` and the global qualifying count, participates in level
progression, and later receives score, but its gauge hook returns without a
gain. Reaching level 100 here enters the terminal state described above while
the remainder of the current contact/actor step still completes.

### Rot propagation -- `0x40a66c`

After burst handling, each orientation applies:

```text
if target.cc in {4,5} and
   (source.cc == 2 or source.d0 > 0) and
   not target.c5 and target.d0 == 0 and target.age > 100:
    target.d0 = 1
```

The age comparison is strict. This can start a projectile's rot timer as well
as an ordinary block's. Block actor update increments a nonrotten normal
timer, but only `cc==4` converts to `c8=3` after the strict delay threshold.
A projectile with `d0>0` can subsequently satisfy `landing_source`.

### Direct projectile hits -- `0x40a89c`

For the block/projectile orientation:

```text
if projectile.e0 == 1 and block.d4:
    block.dc += 1
    if block.dc >= 2:
        projectile.delete68 = 1
        block.delete68 = 1
        play_second_hit_sound()
    else:
        play_first_hit_sound()

if block.c8 != 3 and block.f0 == 0:
    projectile.delete68 = 1
```

The current contact has already incremented `e0`, so only the projectile's
first lifetime-eligible non-sidewall contact can increment damage. Earlier
native contact entries in the same step count. Fresh normal blocks always
consume the projectile; a rotten block does not through the generic branch,
so its first-hit projectile can continue. The second qualifying direct hit
flags both block and projectile.

Projectile-projectile pairs are skipped before `e0` and every game handler.
Box2D still resolves the physical collision, but neither projectile receives
a counter, damage, or deletion state from the dispatcher. A direct-hit `+68`
is not a scored clear unless another earlier handler also set `d5`.

### Actor finalization, deletion, and per-member score

All contacts finish before the block pool at `0x40666d`. For `c8!=1`, Block
update calls post-contact method `0x403618`, whose order is:

```text
if d5:
    score_this_block(block)          # 0x4036b8
    block.virtual_1c()               # destroy body, mark actor dead
    update_highest_chain()
if delete68:
    emit_delete_effect()             # virtual +0x4c
    block.virtual_1c()
    return 0
process e4/e5 and delayed activation
return 1
```

`+68` and pool-dead are distinct. Store `0x403610` merely requests deferred
deletion; it leaves the body and `Actor+0x54` live. Virtual `+0x1c` at
`0x403e78` is raw teardown: it queues and clears the body and sets
`Actor+0x54=1`, so later pool passes skip the object. If `+68` is handled,
post-contact returns false and the rest of that Block update is skipped. A
`d5`-only path instead tears down raw, then returns true and the already-entered
Block update can still execute its rot/f8 tail on that dead object. If both
flags are set, scoring and raw teardown happen first, the `+68` branch performs
a null-safe second teardown, and the tail is skipped.

Every live `d5` member that reaches Block update invokes `score_this_block`
once, using its own slot factor and the shared group's final `{chain,num}`
visible at that actor update. The pool skips an actor already marked dead at
`0x403214–0x403220`; an immediate earlier `virtual_1c` can therefore suppress
the pending score, whereas `+68` alone cannot. Since the complete contact list
runs first, members cleared in the same ordinary step normally see the same
final group values. A creation-frame `c8==1` member defers this path, so
intervening contacts can mutate its shared record before it scores.

When both `d5` and `+68` are set, score executes first and any gauge reward has
already occurred in the contact handler; `+68` does not cancel either. A pure
direct-hit or special-handler deletion with no `d5` receives no normal score.
Destruction does not mutate the shared group record.

## Score, gauge, and rot

### Score — proven/static

The debug literal `burst:chain %d num %d` and its call at `0x40a5c8` confirm
`group+0 = chain` and `group+8 = num`. At `0x4036b8–0x403719`:

```text
factor = max(1, Block.slot_index - 4)
delta = trunc_toward_zero_x87(
    0.5 * (group.num + 1) * group.chain * S * factor)
score += delta
highest_chain = max(highest_chain, group.chain)
```

The score is at `0x4428c0`, highest chain at `0x4428c4`, and the slot index is
`Block+0xc0`. The nonzero-mode table selects slots 0 through 2, all with factor
one. Mode-0 normal selects slots 0 through 6: slots 0 through 5 still have
factor one, while the rare 140-unit slot 6 has factor two. With factor one and
the level formula substituted:

```text
delta = trunc(2 * (group.num + 1) * group.chain * pow(L, 0.7))
```

Operands are positive, making truncation equivalent to floor. The exact x87
operation order at `0x4036e2–0x4036f9` is:

```text
acc = fild_signed32(group.num)
acc = round_PC53(acc * binary64(0.5))
acc = round_PC53(acc + binary64(0.5))
acc = round_PC53(acc * signed32(group.chain))
acc = round_PC53(acc * binary64(S))
acc = round_PC53(acc * signed32(factor))
```

The main thread's live control word is `0x027f` both after CRT setup and after
full DxLib initialization: round-to-nearest with 53-bit precision, not the
64-bit extended-significand mode. At `0x403700–0x40370d` the routine saves that
word, loads `0x0fbf` from `0x4321ac` (truncate-toward-zero for the `fistp
DWORD`), converts, and restores `0x027f`. Loading `0x0fbf` occurs only after
the arithmetic, so it does not make the preceding products extended-precision.
A replay-exact host should preserve the listed sequential binary64 rounding;
one fused or long-double expression can disagree near an integer boundary.

For example, at level 2 with `group.num=1`, `group.chain=25`, and shipped slot
factor 1, the stored scale produces `delta=162`; levels 1, 10, and 99 with the
same group produce 100, 501, and 2494 respectively.

### Seed-41 score parity — proven/original trace and regression

The fresh-process replay with SHA-256
`1ce501febe8f3f6291e4b82736542179bd9808e412d38e0e1fb1c92d05797657`
provides the first exact original-game score transition. Its authentic-DLL
proxy trace contains 521 physics steps and records mode-0 field fixtures,
spawn dimensions, contacts, and destruction. On step 1 the original destroys
newborn actor IDs in order `[14,20,16,19,18,13,11,15]`; the clone now matches
that order.

At tick 304 the original score routine runs twice on actors 10 and 12 (size
slots 1 and 2, color 0) in one shared group with `chain=2` and `num=1`. Each
call computes `delta=8`, for the exact sequence `0 -> 8 -> 16`; native body
destruction follows in actor order 10 then 12. The clone produces the same two
`+8` events at tick 304 and remains at score 16 through all 520 replay records.
The scoring formula above was unchanged. The correction was selecting the
observed mode-0 production profile instead of assuming the nonzero-mode INI
table.

The clean proxy log is
`reference/runs/seed41-score-parity-20260720-001/data/dll/box2d-trace.jsonl`;
the decisive score-breakpoint run is under
`reference/runs/seed41-score-gdb-20260720-001/`. This is one exact replay
outcome, not broad controlled-probe or policy-transfer certification.

### Gauge gains and passive drain — proven/static

Gauge at `0x4428e4` starts from the selected mode table: 3000 with maximum
40000 for mode-0 normal, or 1000 with maximum 10000 for the nonzero-mode table.
The normal level setter hard-codes reward unit 700. On the normal
reward hook at `0x4098fc`, if `Block+0xc8 != 3` (not rotten):

```text
gauge += group.num * 700
```

The special/rainbow clear path at `0x4032d0` marks every matching-color block
for deletion (`Block+0x68=1`) and, in normal mode, adds 700 for each match.

Scene update first checks `gauge <= 0` and synchronously latches game over at
`0x409939–0x409946`, but that call does not return from the active gameplay
method. Unless a separate pause/UI branch takes its early return, the same
terminal update still runs Field spawn, physics/contacts, gauge clamping,
counter increment, passive drain, and the outer actor pass. The clamp to
`0..game_life_max` is at `0x409b9b–0x409c31`; the gameplay counter increments
at `0x409d14`; passive drain follows at `0x409d1a–0x409d70`:

```text
if gauge > 0.5 * maximum:
    drain = 3 * D
else if gauge > 0.8 * maximum:
    drain = 5 * D       # unreachable because of the preceding condition
else:
    drain = D

gauge -= drain
if gauge <= 0:
    gauge = 1
```

The game-over latch is irreversible for the scene. A contact reward later in
that last update can raise the final gauge value, but no reward or clamp clears
`ScenePlay+0x30`; the next update still takes the postgame branch. In
particular, gauge made nonpositive by the preceding frame's actor rot penalty
cannot be rescued on the next frame.

The branch order is genuine executable behavior. Normal passive drain is
therefore `3D` above 50% and `D` at or below 50%; it cannot itself trigger game
over because it floors at 1.

The INI key `game_life_plus_unit=120` is not the normal 700-point reward, and
`rotten_minus=5000` is not the normal level-dependent rot penalty described
below.

### Rot — proven/static, state labels strong

In block update at `0x4033d0`, when normal mode, `Block+0xf0 == 0`,
`Block+0xc8 != 3`, and the rot timer at `Block+0xd0` is nonzero, the timer is
incremented. It triggers on a strict comparison:

```text
rot_timer += 1
if rot_timer > block_death_delay and Block.cc == 4:
    Block.c8 = 3
    gauge -= P
```

`block_death_delay` is 40 in mode-0 normal and 120 in the nonzero-mode table.
The store making `c8=3` makes the penalty one-shot. This subtraction occurs
during the later block-pool update,
after the scene's passive floor-to-1 logic. It may make gauge nonpositive; the
next scene update observes that and enters game over.

## Lifetimes and deletion

The actor constructor at `0x4075e8` initializes remaining lifetime
`Actor+0x58=-1` and age `Actor+0x5c=0`. Base update at `0x4077c4` performs the
motion virtual, then:

```text
remaining -= 1
age += 1
if remaining == 0:
    mark_for_deletion()
```

`-1` is effectively infinite under ordinary play. Normal spawn reads the
selected `block_life_span` at `0x4062eb–0x40631e` (mode-0 100000, nonzero-mode
10000); projectile spawn reads `ball_life_span` at `0x4064e9–0x40651e`
(mode-0 3000, nonzero-mode 1200). Because a new actor receives its
first block-pool update on its creation frame, a positive lifetime `N` expires
on that frame's `N`th actor update.

The block deletion marker at `0x403610` sets `Block+0x68=1`; it is not itself
destruction. Later cleanup destroys the body and marks the pool actor dead.
The virtual destructor at `0x403e78–0x403e9c` uses the wrapper's null-safe
`b2d_destroy_body`.

The out-of-bounds path at `0x403403–0x40344e` runs only for physics-owned
blocks satisfying the surrounding enable/state predicates (`c4==1`, `e5`, and
`cc` 4 or 5). It tests:

```text
x > 640 or x < 0 or y > 560 or y < -30
```

On exit it zeros actor velocity and sets `remaining=1`. Base lifetime
decrement already occurred earlier in that update, so the deletion marker is
called on the next actor update.

Normal player projectiles acquire `e5` on their second actor update, as shown
in the top-gate section, so this guard can eject them from the third update
onward. Collision/deletion state and their profile-selected lifetime remain the
other exit paths.

The complete normal Block update ordering is:

```text
base motion/sync
remaining -= 1; age += 1; if remaining == 0: set +68
if cc not in {4,5}: return
if c4 and e5 and strictly out of bounds: velocity = 0; remaining = 1
if c8 != 1 and not post_contact(d5, +68, e4/e5): return
if c8 == 1: c8 = 2; refresh_visual()       # post_contact was skipped
if not f0 and c8 != 3 and d0 != 0:
    d0 += 1
    if d0 > rot_delay and cc == 4: rot and subtract gauge
if c4: f8 += 1
run the f4/special-visual tail; refresh_visual()
```

`f8` increments after the rot test, including after a `c8==1` bypass and after
a `d5`-only raw teardown. It does not increment after a handled `+68` return,
and boundary Blocks return before reaching it. Normal-mode rot start does not
read `f8`; the `f8>=5` guard at `0x40a6d5` belongs to the `f0==1` EX branch.

The four static boundary Blocks are live pool actors. Each update performs its
base physics sync, decrements the effectively infinite `remaining=-1`, and
increments age. Their `cc` values 3, 3, 2, and 6 fail the `{4,5}` gate, so they
never run OOB, post-contact, rot, or f8 logic.

## Configuration boundaries and remaining leads

These are safe conclusions:

- `world_magnification=100` does not override the literal magnification 10 in
  the normal world constructor.
- `game_life_plus_unit=120` does not replace the normal hard-coded reward 700.
- `rotten_minus=5000` does not replace `P=1800+20L` on the recovered normal rot
  path.
- Normal size odds are integer tickets, not percentages: 99 tickets in the
  mode-0 normal table and 70 in the nonzero-mode/Metsu-side table.
- The shipped INI values match a genuine nonzero-mode/Metsu-side table. They
  must not be substituted for mode-0 normal; replay header offset `+0x10`
  identifies the selector state.
- The color variable is an inclusive maximum ID, not a count.
- The ordinary block's initial descent is scripted kinematic motion, not a
  fixture velocity or damping effect.

The following remain **leads**, not implementation facts:

1. Legacy Box2D 1.4 contact ordering, sleeping, and numerical drift over long
   replay runs. The local wrapper probe is the numerical oracle.
2. Exact original source names for every recovered Block field. Their branch
   effects above are proven; descriptive names such as “top gate” are clone
   terminology.
3. Long-run score conversion cases where the x87 PC53 operation sequence
   differs from a reordered or fused host `double` computation by one point.
