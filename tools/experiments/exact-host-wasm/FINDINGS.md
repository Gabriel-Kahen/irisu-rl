# Exact host inventory and translation finding

Inventory source: attested multiworld exact host SHA-256
`ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5`.

## Binary boundary

- ELF32 i386, 141,140 bytes; four `PT_LOAD` segments flatten to a 102,400-byte
  zero-based guest image.
- 16 public `b2d_world_*` C exports. The GNU-facing bridge is `cdecl`; the
  private converted MSVC wrapper is `stdcall` with 32-bit pointers and floats.
- Six strong imports: `malloc`, `free`, `memcpy`, `memmove`, `memset`, and
  `abort`. Four additional imports are weak ELF startup/profiling hooks.
- 524 relocations: 503 `R_386_RELATIVE`, 14 `R_386_PC32`, four
  `R_386_GLOB_DAT`, and three `R_386_JUMP_SLOT`. At guest base zero, the
  relative/text addends retain their linked addresses; calls crossing the six
  strong imports remain explicit runtime boundaries.

GNU objdump 2.46 decodes 26,759 instructions across 74 mnemonics. Of those,
12,727 are x87 instructions. Most are transport and basic arithmetic:

| Instruction | Count |
| --- | ---: |
| `FLD` / `FST` / `FSTP` | 8,172 |
| `FMUL` / `FMULP` | 1,683 |
| `FADD` / `FADDP` | 1,028 |
| `FSUB*` | 456 |
| comparisons and status-word reads | 454 |
| `FSIN` / `FCOS` / `FSINCOS` | 2 / 1 / 1 |
| `FSQRT` | 1 |

The complete sorted counts are generated into `manifest.json`.

## Practical route

A full PC, kernel, or Wine layer is unnecessary. The fixed image has a tiny
host boundary and a small general-purpose instruction vocabulary. The narrow
production architecture is:

1. preserve the image at guest base zero;
2. recover reachable basic blocks from the private `msvc_b2d_world_*` roots;
3. translate those blocks ahead of time to Wasm and intercept the six imports;
4. model x87 stack/control/status explicitly;
5. replace only proven PC53 arithmetic with Wasm `f64`, retaining software
   helpers for unproved edge cases and reference-matched transcendental ops.

The checked-in smoke slice proves this embedding model. It executes 26 actual
x86 instructions from `msvc_b2d_world_test` in Wasm, covers both ownership
branches and absolute `.rdata` addressing, and copies the original x87-loaded
float word `0x40490fdb` into the synthetic body. The current stripped smoke
module is deterministic at SHA-256
`2ad309fa7ecf01f338c4ee4515deb79bb3de0379dc0f3d8b50e1d0c6bfc849a5`.

This is an executable feasibility result, not physics parity. Full parity still
requires translating the reachable solver, implementing the x87 arithmetic and
trigonometric boundary, and passing the existing full wrapper/replay corpus
bit-for-bit.
