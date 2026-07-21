# Local Binary and Format Analysis

This document records read-only findings from the authorized local v2.03 executable and DLL. Addresses and inferred semantics should be independently verified before they become clean-room implementation facts. Rules that have passed that promotion step are collected in [`game-rules-analysis.md`](game-rules-analysis.md); this file remains the lower-level format and wrapper notebook.

## Physics engine identity

The shipped `data/dll/Box2D.dll` is a PE32 x86 DLL timestamped 2008-05-11 and linked with a Visual Studio 2008-era linker. Its API and object layouts identify legacy Box2D 1.4.x, very likely the 1.4.3 / Box2DFlashAS3 1.4.3.1 lineage—not Box2D 2.0.1.

Evidence includes use of `BodyDef.AddShape`, `SetOriginPosition` and `GetOriginPosition`, a legacy body layout with `userData` at offset `+0x84`, and world/body sizes that disagree with Box2D 2.0.1. The cached `Box2D_v2.0.1.zip` is therefore a disproven version lead and must not be used as the nominal clone engine.

Either port the relevant 1.4-era semantics into a maintained core or build an explicit compatibility layer. A modern Box2D version is acceptable only after differential tests show that its contact ordering, sleeping, integration, fixture geometry, material mixing, and solver behavior match closely enough for transfer. The ignored toolchain archive includes Box2DJS 0.1.0, whose official project states it was mechanically converted from Box2DFlashAS3 1.4.3.1; use it as a readable semantics reference, not numerical ground truth.

## Recovered wrapper ABI

All exports are 32-bit `stdcall`:

```c
int b2d_init(
    float min_x, float min_y,
    float max_x, float max_y,
    float gravity_y,
    float magnification);

void b2d_dispose(void);

void *b2d_create_box(
    float width, float height,
    float x, float y, float radians,
    float density, float friction, float restitution);

void *b2d_create_triangle(
    float width, float height,
    float x, float y, float radians,
    float density, float friction, float restitution);

void *b2d_create_circle(
    float radius,
    float x, float y,
    float density, float friction, float restitution);

void b2d_destroy_body(void *body);
void b2d_step(float dt, int iterations);
int b2d_get_contact(void **user_a, void **user_b);
float b2d_get_x(void *body);
float b2d_get_y(void *body);
float b2d_get_r(void *body);
void b2d_get_v(void *body, float *vx, float *vy);
void b2d_set_v(void *body, float vx, float vy);
void b2d_set_user_data(void *body, void *value);
void b2d_set_position(void *body, float x, float y, float radians);
void b2d_test(void *body);
```

Observed wrapper behavior:

- `b2d_init` divides bounds and vertical gravity by `magnification`, creates a sleeping-enabled world, and stores the magnification globally.
- Box width/height use half-extents internally; circle's first argument is a radius. Dimensions and positions are divided by magnification while angles and material coefficients are unchanged.
- `b2d_step` calls the legacy world's two-argument `Step` and caches the contact-list head. Normal game code calls `b2d_step(world_step * time_multiplier, 10)`: 10 solver iterations, with the same integer multiplier used by D-side scripted updates.
- `b2d_get_contact` enumerates contacts with a manifold, returns the two body `userData` values, and advances an internal cursor until it returns zero.
- Positions are returned multiplied by magnification. Rotation is returned in radians.
- Velocity is asymmetric: `b2d_set_v` divides by magnification, but `b2d_get_v` returns raw world-unit velocity without multiplying back. Preserve this behavior in differential probes rather than assuming a symmetric unit API.
- `b2d_set_position` divides position by magnification, applies the origin transform, and zeros linear velocity. It does not clearly zero angular velocity.
- `b2d_destroy_body` is null-safe. `b2d_test` is unused by the game and should not be part of clone behavior.

Build a minimal Windows probe that loads this exact ignored DLL under Wine and records isolated-body results. It is a behavioral oracle for the wrapper and legacy physics, not a runtime dependency for the clean clone.

## Replay format

v2.03 writes a 52-byte header:

```text
int32 seed
int32 highest_level
int32 final_score
int32 highest_chain
int32 mode          # 0 normal, 1 Metsu
byte padding[32]
uint32 frames[]
```

One packed record is produced per game-input update:

```c
word = (y << 12) | (x << 2) | (right_down << 1) | left_down;

left  =  word        & 1;
right = (word >> 1)  & 1;
x     = (word >> 2)  & 0x3ff;  // 10 bits
y     = (word >> 12) & 0x1ff;  // 9 bits
// bits 21..31 are ignored/reserved by the decoder
```

Playback substitutes recorded state for live mouse state. Mouse validation uses the 640x480 client space. Older files used a 20-byte header; the v2.03 loader nevertheless consumes 52 bytes, swallowing the first eight legacy input records as if they were padding. Do not expect cross-version replay playback to begin at the correct input frame.

`tools/inspect-rpy.py` implements both layouts conservatively and was corrected to the executable's 9-bit Y mask. Some real files contain nonzero ignored high bits, so preserve and report them rather than demanding zero.

## DXArchive

The executable calls `SetDXArchiveKeyString` with the literal key:

```text
shine
```

The official DXArchive decoder was cached in the ignored toolchain archive and successfully decoded both the pristine Japanese and English-patched `dat.dxa` copies with that key. The decoded archive contains story/configuration resources such as scripts, music metadata, and `irisu.rng`; it did not expose the embedded `level_table.txt` name. Keep decoded files ignored and do not use copyrighted presentation resources in the clean clone.

Embedded names still provide valuable investigation leads:

```text
level_table.txt
block_color_kind
block_speed
ex_block_speed
tri_ratio
score_atozdj
statge_level_norma
yeild_interval
game_life_*
```

The misspellings are the original lookup keys. Embedded paths/modules confirm a D implementation, including `ScenePlay.d`, `ScenePlay_org.d`, `Block.Block`, `Field.Field`, `Game.Game`, `tx.Input.Input`, and `ReplayData`.

## DxLib RNG

Static analysis and a read-only probe of the exact local DLL established the
full generator, not only a few output examples. DxLib uses a 624-word MT19937
state with a two-step `69069*x+1` seed expansion and inclusive high-product
range scaling:

```text
GetRand(maximum) = floor(raw_u32 * (maximum + 1) / 2^32)
```

It is not a modulo reduction. The redistributable model, optional Windows
probe, and measured vectors are in [`rng-oracle/`](rng-oracle/). Exact game
call order and seed handling are in [`game-rules-analysis.md`](game-rules-analysis.md).

## Promoted normal-mode rules

Static correlation has now resolved the normal world-init arguments, all four
boundary fixtures, frame/spawn cadence, weighted sizes, spawn position and
rotation, level formulas, scripted pre-contact descent, shot mapping, score
operands, gauge arithmetic (including an unreachable branch), rot penalty,
and lifetime/deletion ordering. Use
[`game-rules-analysis.md`](game-rules-analysis.md) as the implementation ledger;
it records exact virtual addresses and confidence for each rule.

## Remaining reverse-engineering priorities

1. Trace the full contact/group state machine during controlled replay playback,
   especially group confirmation, activation, reward denial, and special-orb
   transitions.
2. Differential-test long replay runs against legacy Box2D 1.4 contact order,
   sleeping, and numerical drift.
3. Compare edge-case score conversions with the x87 PC53 operation sequence.
4. Turn every promoted rule into a minimal original-versus-clone differential
   test.
