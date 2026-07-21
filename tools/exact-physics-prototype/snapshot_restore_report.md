# Exact-physics snapshot/restore

Date: 2026-07-20

Implementation status: the Linux fork/COW keeper described below is now
available through worker opcodes 10-12 and
`ExactSimulator.fast_checkpoint()` and `IrisuEnv.fast_checkpoint()`. The
durable seed/action-log snapshot bytes remain unchanged. See
`fast_snapshots.md` for the production ownership and release contract.

## Result

A process-level snapshot is already usable for the exact MSVC9 backend. A
single-threaded 32-bit worker can retain a checkpoint process and use `fork()`
to create an exact branch. The kernel preserves the MSVC wrapper globals, the
entire r58 object graph, allocator free lists, pointer identity, contact
manifolds, warm-start impulses, broad-phase arrays, and floating-point state by
copy-on-write. No reconstruction is involved.

The checked-in prototype exercised exact-forward `Simulator` instances, not a
toy replacement. At four checkpoints in the authoritative 41,449 replay, the
forked future and a separately reset-and-replayed future had identical hashes
over every event, every contact event, and every sampled actor/native body
field. Reset plus action replay was independently repeatable as well.

The current 16-call exact wrapper is not sufficient for deterministic body
reconstruction. A controlled mid-contact experiment recreated the same world
in the same creation order from all wrapper-visible body state (`x`, `y`,
rotation, and linear velocity). Rotation diverged on the first future step and
the active contact sequence diverged three steps later.

## Measured branch-future evidence

Library: the validated ELF32 host with SHA-256
`14475fa3bf3f93e2a644abaadc12b2d7b981d7569a13db6873a32cafe642995a`.
The runner installs the replay oracle's `0x027f` x87 control word.

Replay: `irisu_00041449_20100725_182435_7.rpy`.

| Prefix | Future | Fork time | Action replay of prefix | Future contact events | Result |
|---:|---:|---:|---:|---:|---|
| 100 | 500 | 0.10 ms | included in a 25.8 ms total | 48 | exact |
| 8,000 | 4,000 | 0.14 ms | 216 ms | 1,707 | exact |
| 30,000 | 4,000 | 0.13 ms | 935 ms | 1,574 | exact |
| 45,000 | 2,000 | 0.13 ms | 1,598 ms | 441 | exact |

“Exact” means all three comparisons passed:

1. parent future equals fork-child future;
2. two fresh reset-and-action-replay runs equal one another; and
3. the fork future equals the fresh action-replay future.

The 8,000 + 4,000 experiment hashed 16,930 events and 82,719 full body
samples. Its full-future hash was `866e97c8145c1576` and its contact-event hash
was `ca0ae3b99a148055`. Additional checks passed for 4,000 + 4,000 frames of
the 43,791 replay (900 contact events) and 500 + 500 frames of the header-40
replay (14,300 contact events).

Timing is a local feasibility measurement, not a stable benchmark. The
important scaling result is that `fork()` stayed approximately constant while
action replay grew linearly with checkpoint age.

The production Python benchmark repeats the comparison through the public
worker API. At a 1,000-action history, checkpoint creation is 187.828 us,
median branch creation is 479.742 us (2,056.212/s), and durable restore is
95.933 ms (10.413/s), a 199.968x median branch advantage. Each branch matches
both the source state hash and durable snapshot bytes. The production raw
protocol regression
also compares 1,000 complete encoded responses from two frame-30,000 branches
with the authoritative replay source when that optional oracle is present; it
skips that asset-dependent case cleanly when the replay is absent.

The controlled public-rebuild witness checkpointed on the first floor contact
at physics tick 62. Rebuilding from all state exposed by the exact wrapper
diverged in `angle` at future tick 1 and in returned contacts at future tick 3.

## What the production snapshot preserves

The production GNU-r58 backend does considerably more than recreate bodies
from poses. Its `Body`, `ContactImpulse`, and `PhysicsOrdering` snapshot fields
cover:

- native origin and center-of-mass float bits, rotation, linear and angular
  velocity, sleep flag, and sleep timer;
- live world body-list order and deferred body-destroy-list order;
- fixture proxy IDs, the complete free-proxy list, static-body sleep flags,
  broad-phase timestamp, all 512 proxy timestamps and overlap counts, and the
  exact X/Y bound arrays;
- world contact-list order and each body's contact-node order;
- zero-manifold and deferred-destroy contacts as well as touching manifolds;
- manifold normal, points, separation, feature IDs, and accumulated normal and
  tangent impulses as raw float bits.

Reconstruction also allocates placeholder proxies to reproduce proxy IDs,
rebuilds and reorders broad-phase pairs and contacts, restores frozen and
out-of-range proxies, and restores pending destroys. The native tests cover
dense mid-contact stacks, swept zero-manifold contacts, deferred replacement
contacts, frozen and out-of-range proxies, asymmetric triangle centers, sleep,
and prolonged proxy/pair churn.

This is the correct model for a future structured exact snapshot. It cannot be
used through the current exact wrapper because almost none of those r58
internals cross its C ABI.

## Exact-wrapper state gap

| State needed for reconstruction | Current exact wrapper |
|---|---|
| Body origin X/Y and rotation | readable/writable |
| Linear velocity | readable/writable (with asymmetric magnification contract) |
| Angular velocity | not readable or generally writable |
| Native center of mass and conservative previous transform | unavailable |
| Sleep/frozen/destroy flags and sleep timer | unavailable |
| World body and deferred-destroy order | unavailable |
| Proxy IDs, free list, timestamps, overlap counts, and bound arrays | unavailable |
| Pair-manager table/free/buffer state | unavailable |
| Contact flags, list/node order, manifolds, feature IDs, and impulses | unavailable |
| Allocator state/pointer graph | unavailable |

The generated exact host presents independent world handles, but the bridge
serializes pristine-r58 calls and still exposes only the original narrow body/
contact operation set. It does not export solver, allocator, pair-manager, or
broad-phase internals. The exact frontend therefore throws from `rebuild()`,
`contact_impulses()`, and `ordering()`.

Pristine r58 also retains process-global allocator state across world teardown.
Direct exact C ABI/C++ callers must therefore remain one-episode diagnostics.
The worker protocol rejects a second reset, while `ExactSimulator` and
`IrisuEnv` transparently create and identity-check a fresh process for each
later episode.

## Implemented and future paths

### 1. Linux fork/COW snapshots behind the 32-bit worker (implemented)

All forking remains inside the small, single-threaded exact worker; the Python
training process is never forked. Checkpoints are accepted only at RPC request
boundaries, after simulator and wrapper calls have returned.

A snapshot handle identifies a dormant keeper process:

1. On `snapshot`, the active worker forks. The child closes the ordinary RPC
   descriptors and waits on a private keeper control socket. The active parent
   returns the keeper handle and continues.
2. On `branch`, the keeper forks again. The keeper remains frozen
   at the checkpoint; its child receives a new RPC socket and becomes an active
   rollout worker. The client authenticates the keeper's peer credentials,
   direct ancestry, and inherited executable/library file identities before
   accepting the launch-verified provenance. An explicit provenance query
   rehashes the library mapped by the live branch.
3. Releasing a handle terminates and reaps its keeper. Worker shutdown
   recursively reaps owned keepers.

This supports repeated branches without copying or interpreting r58 state.
`ExactSimulator` and `IrisuEnv` expose it through `fast_checkpoint()`. A source
refuses reset/restore while it owns a live checkpoint; a branch may reset or
restore after detaching from its parent keeper. Busy release refuses rather
than killing caller-owned branches, and source death recursively terminates and
reaps descendants.

Limitations:

- handles are Linux-local and cannot survive worker/process restart;
- each branch is a process, so vector environments need RPC batching and a
  process budget;
- copy-on-write memory grows as branches mutate heap pages;
- descriptor ownership and process reaping must be explicit;
- the worker must remain single-threaded across `fork()`; and
- sanitizer builds and platforms without `fork()` need another path.

### 2. Seed + mutation log for persistent serialized snapshots (implemented)

Production stores the seed, mechanics/config hash, exact-library/worker
fingerprints, and canonical sequence of accepted actions. Restore creates a
fresh exact worker, validates its identity/configuration, replays the log, and
swaps only after success. For ordinary `IrisuEnv`, the mutation log is the
action history. Any future test/debug API that directly spawns or mutates
bodies must log those commands too.

The durable clone operation refuses while a split exact step is pending. The
worker may already have accepted and advanced that request, so serializing
before the matching response commits its action would produce an incomplete
history.

This route is byte-serializable and process-independent, and production tests
show the same future as a live fork on the covered traces. Its cost is O(checkpoint
age): about 216 ms at tick 8,000, 935 ms at tick 30,000, and 1.60 s at tick
45,000 in this local run. Periodic fork keepers can make interactive restores
fast while the action log remains the durable representation.

### 3. Add structured r58 snapshots only if portable constant-time restore is required

Port the production reconstruction algorithm into code compiled with the exact
MSVC9 r58 headers. Add a neutral, versioned C ABI such as
`b2d_snapshot_size/write/read`; serialize integer IDs and raw float bits, never
MSVC pointers or C++ object bytes. The implementation can access the old r58
internals because they are public in its headers.

The serialization and reconstruction should live inside the MSVC-compiled
wrapper. A GNU caller must not dereference MSVC C++ objects or assume their
layout/vtables. Extend the COFF host's export rename table and stdcall bridge
for the new entry points.

Start by porting the exact fields already validated in `PhysicsOrdering` and
`ContactImpulse`, then run the same branch-future suite against MSVC execution.
Do not treat pose/velocity parity at the restore instant as success; require
future equality through dense contacts, sleep transitions, deferred destroy,
proxy churn, and all authoritative replays.

### Rejected shortcut: raw byte serialization

Copying `b2World` or allocator memory to a byte buffer is not a valid portable
snapshot. The graph contains process pointers, vtable pointers, global/static
state, malloc ownership, and free lists. Restoring it at different addresses
would require process-checkpoint machinery. `fork()` is the safe and efficient
form of raw-memory snapshot on this target.

## Historical prototype reproduction

```sh
tools/exact-physics-prototype/build_branch_snapshot.sh \
  /path/to/libirisu_box2d_msvc_exact.so \
  /tmp/irisu-exact-branch-snapshot \
  /tmp/irisu-exact-public-rebuild

/tmp/irisu-exact-branch-snapshot \
  reference/replays/raw/internet/irisu_00041449_20100725_182435_7.rpy \
  8000 4000

/tmp/irisu-exact-public-rebuild
```

Prototype files:

- `branch_snapshot_runner.cpp`: full-simulator fork and action-replay future
  equivalence;
- `public_rebuild_runner.cpp`: current-wrapper insufficiency witness;
- `single_world_compat.cpp`: adapts the validated original one-world library
  to the handle-taking frontend used by the prototype directory;
- `build_branch_snapshot.sh`: 32-bit build helper.
