# Exact multiworld wrapper

This variant moves the legacy wrapper's `b2World`, contact cursor, and
magnification from process globals into an opaque `b2dWorldHandle`. Multiple
worlds can therefore be alive, reset, stepped, and destroyed independently in
one 32-bit process for bounded diagnostics. Pristine r58 still has process-
global allocator/lazy state that makes repeated multi-episode teardown unsafe;
production training uses a fresh worker process per episode.

The public ELF bridge serializes individual API calls. Pristine r58 has shared
lazy contact/block tables and a global allocation counter, so this preserves
thread safety without changing the exact MSVC engine objects. Worlds have
independent state, but calls do not execute in parallel inside one process.
Serialization is also an invariant of the paired-trig runtime cache, including
its positive-zero fast path. The legacy single-world host does not provide this
bridge and does not support concurrent calls.

## API

Every operation receives its owning world handle:

```c
void *b2d_world_create(float min_x, float min_y, float max_x, float max_y,
                       float gravity_y, float magnification);
void b2d_world_destroy(void *world);
void *b2d_world_create_box(void *world, /* original eight arguments */);
void *b2d_world_create_triangle(void *world, /* original eight arguments */);
void *b2d_world_create_circle(void *world, /* original six arguments */);
void b2d_world_destroy_body(void *world, void *body);
void b2d_world_step(void *world, float dt, int iterations);
int b2d_world_get_contact(void *world, void **first, void **second);
```

The scalar/vector getters and setters follow the same rule and use the
`b2d_world_get_*` / `b2d_world_set_*` names. A body from another live world is
rejected. Body pointers remain invalid after their deferred destruction is
cleaned by a step or after their world is destroyed.

## Paired and positive-zero exact-runtime trigonometry

Profile evidence identified the converted MSVC `FCOS` and `FSIN` intrinsics as
25.883% of dense-core samples. An instrumented 3,000-decision run observed
13,096,069 calls where Box2D invokes cosine immediately before sine with the
same raw float argument. The shared checked-in runtime shim computes that pair
with one x87 `FSINCOS`, returns cosine, and caches the float-rounded sine under
the raw input key. A nonmatching sine still executes standalone `FSIN`.

Of those matching inputs, 833,228 (6.362%) are raw positive zero. For this exact
case the shim stores sine `+0` and returns cosine `1` without executing
`FSINCOS`; raw negative zero and ordinary nonzero inputs retain the general
paired path. At raw absolute-angle bits `>= 0x5f000000` (`|angle| >= 2**63`),
the runtime uses direct `FCOS` then `FSIN`, preserving every finite snapshot
angle beyond `FSINCOS`'s argument-reduction range. Boundary vectors plus
100,000 full-raw-bit randomized inputs prevent recurrence. An unarchived
controlled local dense-core A/B improves 1.287% over the paired-only host; the
wide pipeline does not isolate that change.

The public bridge lock prevents another world from interleaving between the
pair. No Box2D solver iteration or operation ordering changed. A controlled
pinned seven-run A/B improves the dense core from 1,246.873 to 1,448.173
decisions/s (+16.14%). The comparable full pipeline improves the dense native
simulator by 16.82%, the 48-body physics workload by 14.37%, and the eight-lane
packed/lazy vector path by 13.50%.

## Build

Use the dedicated hosting tool with the same exact MSVC9 RTM inputs as the
single-world build:

```bash
python tools/host-msvc9-box2d-multiworld.py \
  --source-dir /path/to/pristine/box2d-code-r58 \
  --cl '/path/to/MSVC9/bin/cl.exe' \
  --vc-include '/path/to/MSVC9/include' \
  --wine /path/to/wine \
  --winepath /path/to/winepath \
  --wine-prefix /path/to/prefix \
  --output-dir /new/output/directory
```

The output is `libirisu_box2d_msvc_exact_multiworld.so`, with that stable name
embedded as its ELF `DT_SONAME`. For `--object-dir`,
the directory must contain the 27 exact engine objects plus this variant's
wrapper object under the conventional name `box2d-wrapper-msvc.obj`; the old
single-world wrapper object is not interchangeable.

Configure the current exact-forward simulator against that library with a
32-bit GNU build. No CMake changes are needed by the prototype adapter:

```bash
cmake -S . -B build-exact-multiworld \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS=-m32 \
  -DIRISU_PHYSICS_BACKEND=exact-msvc \
  -DIRISU_EXACT_BOX2D_LIBRARY=/output/libirisu_box2d_msvc_exact_multiworld.so
cmake --build build-exact-multiworld -j
```

CMake rejects an exact host without that SONAME so the linker cannot capture
its build-time absolute path. `cmake --install` places the host beside the C
ABI library and gives the installed executables origin-relative lookup paths;
the resulting install tree can therefore be moved as a unit.
The exact host remains a `DT_NEEDED` dependency and the adapter reopens only the
already-loaded object with `RTLD_NOLOAD`. A post-link verifier requires every
exact consumer to have a nonempty `RPATH`/`RUNPATH` made only of absolute or
`$ORIGIN`-relative components. Empty/current-directory and bare relative
components fail the build; install relinks rather than padding a build path
with an empty loader-search component.

The exact-forward adapter resolves 15 typed `b2d_*` pointers before simulator
construction and uses those stored pointers for every physics call. Their
targets must be unique executable addresses owned by this library's link-map
object, share one device/inode mapping, and match their process-global
bindings. Every target's `dladdr1` symbol name and exact symbol-start address
must equal the requested entrypoint. Worker opcode 13 exposes the count and
target mapping identity; Python requires it to equal an independent live-library
capture. Production launch uses working directory `/`, removes every inherited
`LD_*` variable and `GLIBC_TUNABLES`, and forces x87 control word `0x027f`.
Consequently, an ordinary preload exporting `b2d_world_step` is rejected even
though the genuine SONAME remains mapped, and a tested interposed-`dlsym`
same-host X/Y permutation fails the symbol-identity check. The library's lower
`msvc_b2d_*` bridge calls are linked with `-Bsymbolic-functions`, and both the
hosting tool and exact CMake reject every `R_386_*` relocation naming those
helpers. These are fail-closed provenance checks for the tested loader attacks,
not a general sandbox for arbitrary code already executing inside the worker.

## Validation completed 2026-07-20; revalidated 2026-07-21

- The C smoke test matched every position, rotation, raw velocity, and contact
  bit for 180 steps against the validated legacy exact library.
- Two handle worlds with different gravity/magnification advanced independently;
  destroying the first did not affect the second.
- Four simultaneous `Simulator` instances, two stepped from separate host
  threads, matched their sequential baselines after 256 ticks.
- The 47,019-tick authoritative replay produced byte-identical runner JSON to
  the legacy exact backend (SHA-256
  `bb8a81554cfc03bbfa186fdba6ee080f691eaabb451e1f562627050ed2fdf236`),
  including score 41,449, 455 score calls, and 379 clears.
- The pre-trig and paired hosts produced byte-identical 813,508-record,
  13,757,907-byte wrapper traces through that replay. Both trace SHA-256 values
  are `cde35ca60b5511678edf128ed8f3ae09c8cf00e240696325c4acc8681f829eb0`.
- The July 21 post-attestation comparison report has SHA-256
  `1d2109141426119b24c1333a9367d0e2bd84b79bd74ee4e6c040bdb3288ee29d`
  and reports no first mismatch.
- Of those records, 813,412 active mutation/step/contact records were compared,
  including all 573,557 contacts. A second global-call-order replay directly
  matched the original trace's 9,810,360 getter-only records and all 12,262,950
  returned binary32 words against the exact host through step 47,019. Its
  [`tracked report`](../getter-parity-2026-07-21.json) SHA-256 is
  `815e8805ea777fdcecc265ee1a669c4664e993a74f92a05cfe2f134195703ef8`;
  getter, contact, and first-mismatch counts are all zero/null.
- The exact replay corpus evaluator reported 4/4 full scoring, terminal-state,
  score-timeline, and rot-penalty-timeline parity.
- The production `IrisuEnv` exact-worker corpus gate independently matches all
  1,111 available original state checkpoints: 536 score, 103 rot, 431 clear,
  and 41 level events. Its attested report SHA-256 is
  `b0e5def9d05eab34f76a43c0bdc23a2ecb83e414223a5ad06bb1d06c500d1848`.
- The full current gate passes 14/14 exact Release and 14/14 exact ASAN/UBSAN
  CTest targets, including the paired-
  trig runtime test, both replay execution paths, and active
  mutation/step/contact stream through all 47,019 steps. Portable Release
  and ASAN/UBSAN builds each pass 8/8 targets. Sanitized hostile-preload tests
  put the worker-linked ELF32 `libasan` first. The GNU layers are instrumented;
  the immutable MSVC9 host is explicitly verified as uninstrumented. The
  Python suite passes 204 tests with three expected normal-build skips.
- The current production host SHA-256 is
  `ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5`;
  the worker SHA-256 is
  `4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261`,
  and the replay runner SHA-256 is
  `d06e1ff08811b21047b827bebb27ea39295445b64d3b56c772617befb1bc22f9`.
- The current
  [`exact-pipeline-adaptive-wide-perf-2026-07-21.json`](../../../benchmarks/results/exact-pipeline-adaptive-wide-perf-2026-07-21.json)
  records 38/38 current runtime-source hashes and 88/88 true cross-path equivalence
  leaves. Its SHA-256 is
  `4067fdff9360989adb696bdc5ad7d98983729f9fa424271fbd7e3e1fb9164eef`.
  Its explicit 64-lane result is 9,685.170 decisions/s, 48.426% of the
  20,000-decision/s target (2.065x short). Its directly comparable 30,000-tick
  48-body physics workload reaches 74,853.849 ticks/s.
- The July 20
  [`exact-pipeline-paired-trig-2026-07-20.json`](../../../benchmarks/results/exact-pipeline-paired-trig-2026-07-20.json)
  remains the historical directly comparable post-pair artifact; its host and worker
  identities are historical rather than the current range-safe positive-zero
  build.

The wrapper itself remains forward-only, but production `ExactSimulator`
provides durable exact snapshots by storing the reset seed plus accepted action
history and rebuilding a fresh worker through deterministic replay. Restore is
therefore O(episode age). Production Linux workers also provide local fork/COW
checkpoints for constant-time branching without pretending that public body
poses are sufficient to reconstruct hidden Box2D state.
