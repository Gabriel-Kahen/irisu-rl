# Legacy physics source

## Decision

The compatibility implementation uses the Box2D 1.4.3 engine from
[Box2D's official SourceForge SVN, revision 58](https://sourceforge.net/p/box2d/code/58/).
The immutable repository path is
[`!svn/bc/58`](https://svn.code.sf.net/p/box2d/code/!svn/bc/58/).
Revision 58 was committed on 2007-11-29; its `Readme.txt` identifies the tree as
"Box2D Version 1.4.3."

This is the strongest supported source basis for the shipped `box2d.dll`. It is
not merely a nearby version: its object layout and stepping algorithm agree with
the binary evidence, while later revisions that still call themselves 1.4.3 do
not. Source identity cannot be proved from a stripped binary, so the DLL and the
probe traces remain the behavioral oracle.

## Provenance and license

The upstream project is Erin Catto's
[Box2D SourceForge project](https://sourceforge.net/projects/box2d/). Revision 58
is distributed under the zlib license, copyright 2006-2007 Erin Catto. The
license is preserved at
[`third_party/box2d_legacy/License.txt`](../third_party/box2d_legacy/License.txt),
and every imported source file retains its upstream notice.

Only the engine's `Include/` and `Source/` trees, `License.txt`, and `Readme.txt`
were imported. Examples, generated libraries, IDE projects, documentation, and
contributed code were omitted. The joint implementations remain because the
unmodified engine core's joint factory and world code link against them, even
though the game does not create joints.

The imported baseline was normalized from CRLF to LF, then nine source/header files received
small, plainly marked portability/safety adaptations described below. No solver,
collision, integration, material-mixing, sleeping, or fixture algorithm was
changed. For auditability:

| Material | SHA-256 |
| --- | --- |
| Upstream `License.txt` bytes | `6d77641123753e054c382a0bd224f79af81a87bf69a8157ee66dff653a43b58e` |
| Upstream `Readme.txt` bytes | `71281efaf7825b823cbcd07c4a45ef9a98237a26b2c4998a320378b463855426` |
| Upstream 55-line `Include/` + `Source/` checksum manifest | `e29c662d65c4eea125f23befac8aa5e621013326fab7d25c308bf98207e0cc43` |
| Baseline LF-normalized 55-line checksum manifest | `fa06cceba7e66b7033257c6ac36e086660a62d5a0c914cc1c6933184566d3d56` |
| Current adapted 55-line checksum manifest | `5c04cb04c9c89fc696682751e180303dea7774a8a40e05443ede762a276fd393` |

Each manifest hash is the SHA-256 of the sorted output of
`sha256sum` over files named relative to the revision root (`Include/...` and
`Source/...`). The SourceForge-generated ZIP itself is not used as an integrity
identifier because regenerated directory metadata can change its archive hash.

## Marked safety adaptations

Every source change carries an `IRISU SAFETY PATCH` comment. They address
undefined behavior or process-global initialization on the supported 64-bit,
multi-environment host without changing the single-world numerical model:

- `b2PairManager` now initializes its broad-phase pointer, callback pointer,
  and pair-buffer count. Revision 58 otherwise consumes an indeterminate
  `m_pairBufferCount` while the first proxy is committed; a dirty placement-new
  regression test covers that construction context.
- `b2Alloc` uses a `max_align_t`-aligned size header. The original four-byte
  prefix can return misaligned storage on 64-bit systems. The allocation byte
  counter is a relaxed atomic because it is diagnostics only and independent
  worlds allocate concurrently.
- The immutable block-size lookup and contact-factory registry use
  `std::call_once`. Revision 58 lazily initialized both through unsynchronized
  process globals, which races when separate environments are created on
  different threads.
- The GJK iteration diagnostic is thread-local, and island position-solver
  progress is stored on each island instead of in shared function statics.
  Neither value participates in the physical result; both otherwise race when
  independent worlds step concurrently.
- World destruction runs the engine's normal deferred body teardown before
  releasing the broad phase. This ensures that live shapes, proxies, contacts,
  and joints do not outlive the storage they reference.
- Shape destruction saves the shape type before ending the polymorphic object's
  lifetime. Revision 58 read that member after the explicit destructor call,
  which is undefined behavior in modern C++ even though the old compiler happened
  to preserve the bytes.

`tests/native/test_physics_lifecycle.cpp` covers dirty construction, 2,000
create/configure/reset/destroy cycles, and concurrent independent worlds. These
changes are compatibility hardening, not evidence that the shipped 32-bit DLL
contained the same fixes.

## Why revision 58

The shipped DLL's 32-bit layouts match revision 58 exactly:

| Evidence | Shipped DLL | Revision 58 |
| --- | ---: | ---: |
| `b2Body` allocation | `0x88` bytes | `sizeof(b2Body) == 0x88` |
| body user-data offset | `0x84` | `offsetof(b2Body, m_userData) == 0x84` |
| `b2World` allocation | `0x19260` bytes | `sizeof(b2World) == 0x19260` |

The exported wrapper also exposes the revision-58-era API: `b2BodyDef::AddShape`,
origin-position accessors, an AABB world constructor with a sleep flag, and the
two-argument `World::Step`.

More decisively, disassembly of the shipped `World::Step` follows revision 58's
single-phase order:

`CleanContactList -> CleanBodyList -> Collide -> Island::Solve -> Commit`

Revision 62 and later use the materially different split pipeline
`Integrate -> Commit -> Collide -> SolvePositionConstraints`. Revision 69 also
adds a body field, moving user data to `0x88` and growing the body to `0x8c`.
Those revisions are therefore incompatible despite retaining a 1.4.3 label.

Independent probe observations agree with revision 58 semantics: semi-implicit
integration, the legacy polygon skin, retained sleeping contacts, deterministic
contact ordering, geometric-mean friction in disassembly, and maximum (rather
than multiplied) restitution.

## Build and compatibility boundary

[`third_party/box2d_legacy/CMakeLists.txt`](../third_party/box2d_legacy/CMakeLists.txt)
provides the static target `irisu::box2d_legacy`. A forced include of
`compat/cstring_compat.h` supplies declarations for `memcpy`, `memmove`, and
`memset` that the Visual C++ 2005-era code obtained transitively. The marked
source safety adaptations above are the only other changes to the imported
engine.

The shipped 32-bit MSVC DLL evaluates the solver through x87. On GNU x86 and
x86-64 builds, the compatibility target therefore uses `-mfpmath=387` instead
of GCC's x86-64 SSE default. A seed-41 crowded-board differential probe found
one-ULP SSE disagreements on the first update. The optimized GNU x87 build is
closer, but it is not bit-exact: against the full original getter oracle its
first mismatch is initial actor 10's rotation after step 1 (`403df825` shipped
versus `403df823` GNU), followed by a one-ULP X mismatch at step 2. The native
build metadata reports the arithmetic choice as `legacy_fp_mode: "x87"`; it is
a deterministic supported profile, not an MSVC-equivalence claim. Unsupported
compiler or architecture combinations report `compiler-default` and retain
their own determinism boundary.

A historical clean-room diagnostic built pristine revision 58 with 32-bit MSVC
9 RTM (`/Od /fp:precise /MT /D NDEBUG`) and an original-compatible wrapper.
After restoring the wrapper's required `b2d_set_v` division by magnification,
it matched 32,515 observed X/Y/rotation/velocity float words through step 1,340
and all 191,339 contact-cursor results through the original run's terminal
physics step 14,706 under the captured `0x137f` environment. The bounded
evidence and hashes are archived in
`reference/runs/replay-41449-msvc-r58-phase-20260720-006/`.

The current exact-forward diagnostic compiles the same pristine source with
MSVC 9 RTM (`/O2 /fp:precise /MT /D NDEBUG`), converts the COFF objects to a
native ELF32 library, and runs them under control word `0x027f`. Across the
complete 47,019-step replay it matches every active wrapper-operation stream:
2,368 creates, 2,368 velocity writes, 2,368 user-data writes, 183,387 transform
writes, 2,344 gameplay destroys, 47,019 steps, and 573,557 contact-cursor
results. This resolves compiler/code generation as the cause of the portable
GNU backend's numerical divergence. The generated host is now an opt-in Python
backend: each active episode owns an isolated 32-bit worker, production resets
replace that worker, and vector lanes run separate worker processes
concurrently. Calls through one public multiworld bridge remain serialized
because pristine r58 retains shared lazy tables and runtime state. Because the
exact legacy world cannot yet be serialized directly, durable exact snapshots
store seed plus accepted action history and restore into a fresh configured
worker by deterministic replay. This is exact but linear in episode age; Linux
fork/COW checkpoints provide the faster local, non-serializable branch path.
Its source, construction recipe, comparison method, exact bounds, and hashes
are recorded in `reference/native-box2d/README.md` and
`reference/native-box2d/validation.json`.

The forward wrapper resolves a typed table of 15 `b2d_*` entrypoints before
constructing a simulator. Every target must be a unique executable address
owned by the genuine exact-library link-map object, all targets must share one
device/inode mapping, and each global binding must equal the attested call
target. The worker publishes that count and mapping identity through protocol
opcode 13; Python requires it to match an independent live-map capture, so a
mapped genuine SONAME alone cannot conceal call-target interposition.

Exact `PaddedVectorEnv` concurrency remains capped at eight by default, but an
explicit `workers=` request may use more independent exact worker processes.
The portable backend remains capped at eight. Separately, `Simulator` mechanics
configuration is immutable after validation, so its configuration hash is
computed once at construction and reused by transitions, diagnostics, and
snapshot compatibility checks without changing its value.

### Paired and positive-zero exact-runtime trigonometry

Ten sampling runs of the dense exact baseline collected 24,020 samples at
1,267.12 ± 4.43 decisions/s. The converted MSVC host accounted for 92.678%
of samples. The largest symbols were `SolveVelocityConstraints` (28.426%),
`SolvePositionConstraints` (23.393%), the MSVC `FCOS` intrinsic (14.205%), and
the matching `FSIN` intrinsic (11.678%). Instrumentation over 3,000 decisions
then found 13,096,069 cosine/sine calls with the same raw float argument.

The checked-in [`msvc-runtime.S`](../reference/native-box2d/msvc-runtime.S)
therefore implements a narrow pair optimization. `FCOS` stores the raw float
key, executes one x87 `FSINCOS`, returns cosine on the x87 stack, and stores the
float-rounded sine. The next `FSIN` reuses that sine only when its raw input bits
match; otherwise it executes standalone x87 `FSIN`. The public multiworld bridge
continues to serialize calls, so different worlds cannot race or interleave the
pair state. Concurrent direct calls to the legacy single-world host remain
unsupported.

Instrumentation also found that 833,228 of the 13,096,069 matching inputs
(6.362%) are raw positive zero, largely from position-solver rotation updates
of static bodies. For that exact input, the retained runtime stores sine `+0`
and returns cosine `1` without executing `FSINCOS`. Raw negative zero and every
ordinary nonzero input continue through the paired instruction path. This
preserves the raw signed-zero behavior while avoiding an instruction whose
exact result is already known.

x87 `FSINCOS` sets condition bit C2 and leaves its operand unreduced when the
absolute argument is at least `2**63`. Snapshot validation permits every finite
float angle, so the runtime checks raw absolute-angle bits against `0x5f000000`
and executes direct `FCOS` followed by `FSIN` at and above that boundary. The
dedicated runtime regression covers both sides of the positive and negative
boundary, maximal finite values, non-finite values, and 100,000 full-raw-bit
randomized inputs. An unarchived controlled local dense-core A/B improved
1.287% over the paired-only host; the wide pipeline does not isolate that
change.

This is a runtime-intrinsic substitution, not a Box2D algorithm change. It does
not alter the ten solver iterations, body/contact ordering, rule calls, or
floating-point result stream. A pinned seven-run A/B improves the dense core
from 1,246.873 to 1,448.173 decisions/s (+16.14%). The paired source-manifested
pipeline records +16.82% for the dense native simulator, +14.37% for the 48-body
physics workload, and +13.50% for eight exact padded lanes. It remains exact on
all four replay oracles and the full 47,019-step stream. The retained host
passes 10/10 exact CTest targets; portable Release and ASAN/UBSAN builds each
pass 8/8, and the Python suite passes 159 tests with two optional Gym skips.

An additional old-host/new-host differential replay produced byte-identical
813,508-record wrapper traces (13,757,907 bytes) through tick 47,019, both with
SHA-256 `cde35ca60b5511678edf128ed8f3ae09c8cf00e240696325c4acc8681f829eb0`.
This covers every traced create, velocity/user-data/transform write, destroy,
step, and contact-cursor result, not only the terminal score.

The current
[`exact-pipeline-range-safe-wide-2026-07-20.json`](../benchmarks/results/exact-pipeline-range-safe-wide-2026-07-20.json)
records 37/37 current source hashes, 64/64 true cross-path equivalence leaves,
and SHA-256
`91c8db5feb9d3c8339d101940f05a42d93a4490641745964a0ca427553b8b8e9`.
It measures 7,199.041 decisions/s at an explicit 32 exact lanes, 35.995% of the
20,000-decision/s target (2.778x short), plus 1,498.136 dense native decisions/s
and 75,819.177 ticks/s on the directly comparable 30,000-tick 48-body physics
workload. The earlier
[`exact-pipeline-paired-trig-2026-07-20.json`](../benchmarks/results/exact-pipeline-paired-trig-2026-07-20.json)
remains the comparable post-pair artifact, and
[`exact-pipeline-final-2026-07-20.json`](../benchmarks/results/exact-pipeline-final-2026-07-20.json)
is its pre-trig baseline.

Every public simulator and C API operation runs inside a thread-local floating-
point boundary and restores the caller's complete environment on return or
exception. The boundary masks exceptions, selects round-to-nearest, and uses
gradual underflow for SSE operations. In the GNU x87 fidelity build it also
loads control word `0x027f` (53-bit precision and round-to-nearest), matching
the controlled full getter/event oracle. An unmodified Wine forwarding run
recorded `0x137f`, so startup precision remains an explicit runtime-provenance
boundary rather than a universal original-game claim. The supported setting
preserves deterministic GNU-profile regression behavior and narrows the
shipped-DLL error; it does not erase the documented MSVC mismatch. Padded
workers enter the same boundary independently; no process-global floating-point
state is changed. Build metadata records this as `nearest,x87-pc53` in
`fp_environment`.

The original game wrapper is a separate compatibility layer. In particular,
pixel/world magnification, the asymmetric velocity conversion, and the wrapper's
velocity reset in `set_position` are not Box2D engine behavior. They must remain
in the clone-facing adapter and should not be "corrected" inside this source.

## Snapshot state boundary

Schema-7 snapshots retain both `GetOriginPosition()` and the exact current
`m_position` center of mass. For r58 triangles, deriving one from the other is
not bit-reversible after float rotation and subtraction, and a one-bit center
change can alter the next solved frame. Restore validates their relationship by
running r58's own transform operations.

Schema 7 also preserves the stale color value in each of the original game's
200 actor-pool slots. Special clears scan those slots by color even after an
actor is inactive, so this otherwise invisible state can change gauge survival
and therefore the entire future reward trajectory.

The previous-step fields `m_position0` and `m_rotation0` are intentionally not
serialized. Simulator checkpoints occur only between completed world steps;
r58's conservative-advancement pass is compiled out in `b2World::Step`, and
`b2Island::Solve` overwrites both fields from the current transform before the
next shape synchronization. They therefore cannot affect a future step from a
supported checkpoint.
