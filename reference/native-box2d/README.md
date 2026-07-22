# Native hosting of exact MSVC9 Box2D code

`tools/host-msvc9-box2d.py` converts a user-provided MSVC 9 RTM build of
pristine Box2D SVN r58 into a native ELF32 shared library. The resulting code
runs without Wine. In the exact-forward simulator it matches every active
mutation/step/contact wrapper stream through all 47,019 steps of the `0x027f`
replay, including all 573,557 contact results. The older 14,706-step `0x137f`
harness remains independent corroboration. See `validation.json` for the exact
bounded claims.

The active-stream comparison uses `tools/compare-exact-wrapper-trace.py`. It streams the
10,777,298-record original JSONL once, ignores getter-only observations, and
compares each wrapper operation kind independently because the original game
interleaves actor destruction with contact iteration. It removes two native
constructor/bootstrap generations (96 records). After all 47,019 physics steps
match, the original process alone performs a teardown suffix of 153,500
transform writes, 24 destroys, and one dispose; those calls are not gameplay
physics and are reported rather than silently counted as matches.

The independent `tools/compare-exact-getter-trace.py` closes that getter gap.
It compiles a 32-bit streaming helper, scrubs loader-control environment
variables, fixes x87 to `0x027f`, and replays all 10,777,297 recorded commands
after the proxy header in original global order against the exact host. It
directly matched all 9,810,360 getter records (7,357,770 scalar and 2,452,590
velocity), all 12,262,950 returned binary32 words, and all 573,557 contact
results through step 47,019. There were zero mismatches and no first mismatch.
The tracked
[`getter-parity-2026-07-21.json`](./getter-parity-2026-07-21.json) report has SHA-256
`815e8805ea777fdcecc265ee1a669c4664e993a74f92a05cfe2f134195703ef8`.
The proxy does not log the wrapper's `b2d_test` export, so this claim is exact
for every recorded call and returned value, not a call-count claim for that
unobserved export.

```sh
TMPDIR="$PWD/.tmp/compiler-temp" python3 tools/compare-exact-getter-trace.py \
  reference/runs/replay-41449-getters-gdb-20260720-005/data/dll/box2d-trace.jsonl \
  /path/to/libirisu_box2d_msvc_exact_multiworld.so \
  --output reference/native-box2d/getter-parity-2026-07-21.json
```

This establishes parity for the legacy physics wrapper only. It does not
validate the simulator's spawning, lifecycle, gauge, chain, or scoring rules,
and a matching physics backend must not be presented as full replay or reward
parity.

The repository intentionally contains only source and conversion tooling. It
does **not** contain Visual Studio, Microsoft headers or libraries, MSVC COFF
objects, the reconstructed PE DLL, or the generated ELF library.

## Build from existing COFF objects

The input directory must contain the 27 r58 engine objects and
`box2d-wrapper-msvc.obj`, all compiled with MSVC 9 RTM for x86 using the exact
flags recorded in `validation.json`.

```bash
tools/host-msvc9-box2d.py \
  --object-dir /path/to/msvc9-r58-objects \
  --output-dir build/native-box2d
```

## Compile and convert in one command

The compiler, its headers, and Wine remain external inputs:

```bash
tools/host-msvc9-box2d.py \
  --source-dir /path/to/pristine/box2d-code-r58 \
  --cl '/path/to/MSVC9/bin/cl.exe' \
  --vc-include '/path/to/MSVC9/include' \
  --wine /path/to/wine \
  --winepath /path/to/winepath \
  --wine-prefix /path/to/prefix \
  --output-dir build/native-box2d
```

The host needs GNU `objcopy`, GNU `ld`, and a GCC installation capable of
building and linking `-m32` code. The tool refuses an existing output path,
checks the exact object inventory, repairs COFF weak aliases, adjusts every
COFF `REL32` addend for ELF `R_386_PC32`, verifies the expected runtime-symbol
boundary and public exports, and writes `metadata.json` beside
`libirisu_box2d_msvc_exact.so`.

The checked-in runtime boundary uses one x87 `FSINCOS` for the observed MSVC
`FCOS` followed by `FSIN` with the same raw float input. It returns cosine,
caches the float-rounded sine under the raw input bits, and retains standalone
`FSIN` for a nonmatching input. Raw positive zero has an additional exact path:
it stores sine `+0` and returns cosine `1` without executing `FSINCOS`. Raw
negative zero and ordinary nonzero inputs retain the paired instruction. For
raw absolute-angle bits at or above `0x5f000000` (`|angle| >= 2**63`), the shim
uses direct `FCOS` followed by `FSIN` because `FSINCOS` cannot reduce the
operand. This changes neither the MSVC engine objects nor their solver
iteration/order semantics.

## Runtime contract

- The library is ELF32 i386 and must be loaded by a 32-bit process.
- The MSVC object code is not position independent, so the library deliberately
  contains i386 text relocations.
- Callers must install the target oracle's x87 control word: observed v2.03
  replay execution uses `0x027f`; the older component harness uses `0x137f`.
- The C API is the 16-function `b2d_*` wrapper in
  `box2d-wrapper-msvc.cpp`. It owns one global world and is forward-only.
- Calls from that public wrapper to its renamed private `msvc_b2d_*`
  implementations are linked with `-Bsymbolic-functions`. The hosting tool
  rejects every `R_386_*` relocation naming those helpers, and exact-backend
  CMake repeats the check, so the tested preload cannot redirect the internal
  bridge layer.
- Concurrent calls to this legacy single-world host are unsupported. Its
  paired-trig runtime state, world, contact cursor, lazy tables, and allocator
  state are process-global.
- Positions cross the API in magnified game units. `b2d_set_v` divides both
  components by magnification; `b2d_get_v` returns native, unmagnified velocity.
- This adapter has no snapshot/restore API and repeated world teardown in one
  process is unsafe in pristine r58. Use it only for one isolated forward
  episode; the worker-backed multiworld path is the trainable lifecycle.

## Retained exact-runtime trig optimizations

Ten dense-baseline sampling runs collected 24,020 samples at
1,267.12 ± 4.43 decisions/s. The exact MSVC host accounted for 92.678% of
samples: `SolveVelocityConstraints` 28.426%, `SolvePositionConstraints`
23.393%, `FCOS` 14.205%, and `FSIN` 11.678%. A 3,000-decision instrumented run
observed 13,096,069 raw-float-matched cosine/sine pairs.

`msvc-runtime.S` implements that measured pair, the exact positive-zero case,
and the required x87 range fallback. For ordinary inputs `FCOS` records the raw
key and computes both values with `FSINCOS`; `FSIN` reuses the saved float-
rounded sine for a matching key and otherwise executes the standalone
instruction. A pinned seven-run comparison improves the dense core from
1,246.873 to 1,448.173 decisions/s (+16.14%). No solver iteration, operation
order, or public wrapper result changed. Production validation remains exact on
all four replay oracles and every active mutation/step/contact stream through
all 47,019 steps.

The public production `IrisuEnv` exact-worker path also runs those four replays
directly and matches all 1,111 available score/rot/clear/level state
checkpoints. Its report is
[`exact-production-replay-parity-2026-07-21.json`](../../benchmarks/results/exact-production-replay-parity-2026-07-21.json),
SHA-256
`b0e5def9d05eab34f76a43c0bdc23a2ecb83e414223a5ad06bb1d06c500d1848`.

Of the 13,096,069 measured pair inputs, 833,228 (6.362%) are raw positive zero.
The retained `FCOS(+0)` fast path produces the same raw float results while
skipping `FSINCOS`; raw negative zero deliberately stays on the general path so
its signed sine is preserved. An unarchived controlled local dense-core A/B
improves 1.287% over the paired-only host; the wide pipeline does not isolate
that change.

The runtime's range guard also preserves direct MSVC intrinsic behavior for all
finite snapshot angles: raw absolute-angle bits `>= 0x5f000000` execute `FCOS`
then `FSIN` instead of the out-of-range `FSINCOS` instruction. The dedicated
runtime test covers both sides of the signed boundary, maximum finite values,
non-finite values, and 100,000 full-raw-bit randomized inputs.

The pre-trig and paired hosts also produced byte-identical 813,508-record,
13,757,907-byte forward wrapper traces on the longest replay. Both traces have
SHA-256 `cde35ca60b5511678edf128ed8f3ae09c8cf00e240696325c4acc8681f829eb0`,
covering every traced mutation, step, and contact-cursor result.

The current production multiworld
[`exact-pipeline-adaptive-wide-perf-2026-07-21.json`](../../benchmarks/results/exact-pipeline-adaptive-wide-perf-2026-07-21.json)
records 38/38 current runtime-source hashes, 88/88 true cross-path equivalence
leaves, and SHA-256
`4067fdff9360989adb696bdc5ad7d98983729f9fa424271fbd7e3e1fb9164eef`.
It records worker SHA-256
`4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261`
and host SHA-256
`ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5`.
It measures 9,685.170 decisions/s at an explicit 64 exact lanes, 48.426% of the
20,000-decision/s gate (2.065x short), plus 1,447.881 dense native wall
decisions/s (1,485.277/s for step plus observation) and 74,853.849 ticks/s on
the directly comparable 30,000-tick 48-body physics workload.

The July 20
[`exact-pipeline-paired-trig-2026-07-20.json`](../../benchmarks/results/exact-pipeline-paired-trig-2026-07-20.json)
remains the historical direct paired-versus-pre-trig comparison: its dense native
simulator improved 16.82%, 48-body physics improved 14.37%, and eight-lane
padded throughput improved 13.50% over
[`exact-pipeline-final-2026-07-20.json`](../../benchmarks/results/exact-pipeline-final-2026-07-20.json).

## Source and redistribution policy

Box2D r58 carries the zlib license, which permits modified source and binary
redistribution subject to its notice and attribution conditions. The native
bridge contains no copied Microsoft CRT object code: its remaining allocation,
memory, finite-test, pure-call, and x87 intrinsic boundaries are implemented in
the checked-in bridge sources and linked to the host C library.

That makes the generated library technically separable from the proprietary
toolchain, but this repository still does not redistribute MSVC output. The
toolchain license has not been independently adjudicated here, and project
policy treats the generated `.obj`, `.dll`, and `.so` files as local build
artifacts. Anyone distributing a generated binary should perform their own
license review and retain the Box2D notice.
