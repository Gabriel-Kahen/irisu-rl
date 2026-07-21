# Mechanics Evidence Register

This is the initial evidence register for the normal puzzle clone. Values here are leads, not automatically ground truth. Copy accepted facts into versioned clone configuration only with units, provenance, an experiment reference, and uncertainty.

> **Promotion note:** executable analysis found two mode-selected parameter
> tables. The selector at `0x4124c0` reads the mode state at `SaveData+0x1c`:
> zero selects the normal table initialized at `0x412560`, while nonzero selects
> the Metsu-side table at `0x413648`. Replay header offset `0x10` supplies this
> mode, so the selector is recorded; it is not hidden save provenance. The
> nonzero-mode table below matches `data/doc/irisu.ini`, so those values are not
> dead, but they are not the mode-0 normal defaults. Every preserved external
> replay header has mode 0. Use
> [`game-rules-analysis.md`](game-rules-analysis.md) for the implemented normal
> facts and selector evidence. This document preserves the INI provenance.

## Shipped v2.03 configuration

The English-patched v2.03 game includes `data/doc/irisu.ini`, a 3,041-byte CP932 plaintext file with SHA-256 `1e29431fe8209c25784d4741f7972737561281169bbb5a56f62e3e0f0b63de35`. The values below have provenance `shipped-config`. Static initialization now proves that the listed gameplay values match the nonzero-mode/Metsu-side table. Keys outside that recovered table still require call-site evidence.

For comparison, the supported clone default is the mode-0 normal profile:

| Parameter family | Mode-0 normal value |
|---|---|
| Field | `x=130`, `y=120`, `width=320`, `height=250`, `blank=40`, `thickness=16` |
| Extra boundaries | top `(-140,320,300)`, bottom height `16`; side span `120..370` |
| Block sizes | `32,46,54,60,72,90,140,5,5,5` |
| Block weights | `20,28,28,14,5,3,1,0,0,0` |
| Block lifetime / strict rot threshold | `100000` / `40` updates |
| Projectile lifetime | `3000` updates |
| Gauge maximum / initial | `40000` / `3000` |
| Qualifying clears per level | `10` |

The executable's score formula, 700-unit clear reward, and level-dependent rot
penalty are shared rule code and were not changed to fit seed 41.

### Field and boundaries

| Key | Value |
|---|---:|
| `field_x` | 94 |
| `field_y` | 20 |
| `field_width` | 420 |
| `field_height` | 370 |
| `field_blank` | 30 |
| `field_thick` | 16 |
| `field_top` | -140 |
| `field_top_w` | 450 |
| `field_top_h` | 300 |
| `field_bottom_h` | 400 |

These are the nonzero-mode/Metsu-side fixture inputs. Their geometry and units
are resolved; mode-0 normal uses the comparison values above.

### INI block table

| Key | Value |
|---|---:|
| `block_density` | 1.0 |
| `block_friction` | 1.0 |
| `block_restitution` | 0.0 |
| `block_size0..2` | 30, 48, 64 |
| `block_odds0..2` | 10, 40, 20 |
| `block_life_span` | 10,000 |
| `block_death_delay` | 120 |

Size slots 3 through 9 are set to 10 with odds zero. These nonzero-mode
weights expand to 70 tickets and use `GetRand(69)`. Sizes are full box/triangle
dimensions. Mode-0 normal instead expands its ten weights to 99 tickets
and uses `GetRand(98)`.

### Player projectiles

| Key | Value |
|---|---:|
| `ball_size` | 24.0 |
| `ball_density` | 8.0 |
| `ball_friction` | 1.0 |
| `ball_restitution` | 0.0 |
| `ball_life_span` | 1,200 |
| `ball_init_vy1` | -250 |
| `ball_init_vy2` | -500 |

The naming strongly suggests the two vertical velocities are weak and strong shots, but their mapping, sign convention, unit scale, spawn position, and whether other impulse is applied need controlled replay/capture probes.

### Special block

| Key | Value |
|---|---:|
| `special_block_size` | 24 |
| `special_block_density` | 50 |
| `special_block_friction` | 0.1 |
| `special_block_restitution` | 0.6 |

This likely describes the heavy bonus orb. Confirm its fixture and rule behavior separately.

### Static fixtures

| Fixture | Friction | Restitution |
|---|---:|---:|
| wall | 1.0 | 1.0 |
| bottom | 1.0 | 0.0 |
| top | 1.0 | 0.5 |

Fixture geometry, collision filters, and which top surfaces use the `top` material remain unknown.

### Gauge and fixed step

| Key | Value |
|---|---:|
| `gauge_x`, `gauge_y` | 34, 100 |
| `gauge_w`, `gauge_h` | 20, 350 |
| `game_life_max` | 10,000 |
| `game_life_init` | 1,000 |
| `game_life_plus_unit` | 120 |
| `rotten_minus` | 5,000 |
| `world_magnification` | 100.0 |
| `world_step` | 0.020 |

`world_step = 0.020` is the exact 50 Hz update. The normal world constructor
passes literal magnification 10; the INI's `world_magnification=100` is not
that call argument. Gauge recovery, passive drain, rot penalty, clamping, and
game-over ordering are recovered separately from the shared rule code. The
nonzero-mode table uses the INI maximum/initial values; mode-0 normal uses
40,000/3,000.

### Other INI tuning keys

| Key | Value | Initial clue |
|---|---:|---|
| `statge_level_norma` | 6 | exact nonzero-mode/Metsu-side qualifying-clear divisor; mode-0 normal uses 10 |
| `a` | 50 | Japanese comment: normal-block frequency |
| `b` | 0.1 | comment: speed |
| `c` | 8 | comment: color |
| `d` | 1 | comment: ex-block frequency |
| `p` | 2,000 | no explanatory comment |

Do not encode guessed meanings for the one-letter keys. Correlate them with
executable reads before treating them as normal-mode inputs.

## Out-of-scope but useful comparison values

The same file contains `ex_block_*` sizes, odds, density, position adjustments, and speed. They appear related to Metsu mode, which is outside the clone goal. Preserve them as a comparison set because call sites may help identify shared selection and physics code, but do not implement Metsu mode as part of the initial clone.

## Highest-value next probes

1. Complete fresh-process golden bundles for positive match, rot, chain,
   ejection, and orb outcomes.
2. Localize the first event/trajectory divergence in the long 41,449- and
   43,791-point mode-0 traces rather than fitting their headers.
3. Run a nontrivial scripted policy in both clone and authorized original game
   to bound qualitative transfer.
4. Keep the exact DLL/RNG probes as bounded component regressions while testing
   additional rare contact/allocator states.

Any result promoted from `shipped-config` to `observed` needs an experiment bundle following `computer-use.md`.

## Contemporary behavioral documentation

The 2008 [Vector review/interview](https://www.vector.co.jp/magazine/softnews/081101/n0811014.html) is unusually precise secondary evidence and quotes the creator about contact/state-transition design. It states that:

- confirmed glowing groups award score/gauge when they later land and clear;
- a confirmed piece leaving the field or being destroyed by a second projectile hit awards neither;
- a rainbow orb contacting a fresh piece clears every on-screen piece of that color;
- the orb persists until contacting a rotten piece, in which case it clears that piece;
- level progression changes colors, fall speed, and rot gauge damage.

These are `community/documented` until reproduced on v2.03. The review dates to the early mechanics line, and the shipped changelog says clear conditions changed in 1.01; exact edge behavior may differ in v2.03.
