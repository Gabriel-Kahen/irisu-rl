# Linux fast exact checkpoints

The exact worker supports an opt-in, process-local checkpoint path on Linux.
It complements, and does not replace, the durable seed/action-log bytes emitted
by `ExactSimulator.clone_state()`. It is exposed by both `ExactSimulator` and
the production `IrisuEnv` API.

## Process model

Opcode 10 (`FastCheckpoint`) is accepted only at an RPC request boundary. The
worker forks a dormant copy-on-write keeper after the previous simulator call
and Box2D bridge lock have returned. The keeper closes inherited standard I/O
and unrelated checkpoint control descriptors, then waits on a private inherited
socket. The response contains an unguessable 128-bit token and the keeper PID.

Opcode 12 (`FastBranch`) presents that token. The keeper forks a rollout child
from its frozen address space and returns a bounded Linux abstract-socket
address, child PID, and one-time 128-bit connection secret. The child accepts
one authenticated connection and then speaks the ordinary framed worker
protocol over that bidirectional socket. The keeper remains frozen, so a token
can create multiple identical branches.

Opcode 11 (`FastRelease`) releases a keeper. Release refuses with a deterministic
bad-request error while any handed-out branch is alive; it never silently kills
an active caller-owned branch. Close branches first, then release the token.
Closing or losing the source worker is the ownership boundary: it forcibly
terminates and reaps all keeper and rollout descendants. A keeper likewise
detects source control-socket EOF, terminates its children, and exits. Keeper
polling reaps naturally exited rollouts so they do not accumulate as zombies.

Tokens and addresses are local capabilities. They cannot survive source-worker
restart, cross machines, or be serialized as durable state. The exact Python
owner retains the reset seed and accepted action log for every fast branch, so
its regular `clone_state()` remains portable and independently restorable.
That durable clone refuses while a split step is pending, preventing a request
already accepted by the worker from preceding its action-log commit.

## Python API and ownership

After reset, `ExactSimulator.fast_checkpoint()` returns an
`ExactFastCheckpoint`; `IrisuEnv.fast_checkpoint()` returns the corresponding
`IrisuFastCheckpoint`. A checkpoint's `branch()` method may be called repeatedly
to create independent exact simulators or fully usable `IrisuEnv` instances at
the frozen state. High-level branches preserve the source configuration,
worker identity, render mode, diagnostic setting, and independent Gym spaces.
They can step, render, clone, and create nested checkpoints normally.

The source refuses reset or durable restore while it owns a live checkpoint.
A branch is different: resetting or restoring it first detaches it from the
parent keeper, then performs the ordinary fresh-worker transaction. Close all
branches before releasing their checkpoint. Source shutdown remains the final
ownership boundary and recursively terminates/reaps the complete descendant
tree.

Branch connection authenticates the keeper's peer credentials, requires the
branch to be its direct child, and verifies the inherited worker executable and
exact-library file identities before accepting launch-verified hashes. Calling
`exact_library_provenance()` on a branch performs a fresh hash of the library
mapped by that live branch and compares it with the inherited capture.

## Evidence

`tests/test_exact_fast_snapshot.py` exercises the protocol without relying on
the production Python wrapper. When the optional authoritative 41,449 replay is
available, it advances to frame 30,000, creates two branches from one keeper,
and compares the complete encoded transition response for another 1,000 replay
frames against the live source. That oracle-dependent case skips cleanly when
the replay asset is absent; all lifecycle/security checks remain asset-free.
The test also covers busy release, stale and random tokens, natural branch
reaping, keeper reaping, and recursive cleanup after abrupt source death.

`tests/test_exact_fast_snapshot_api.py` covers the production low- and high-
level APIs, reusable checkpoints after source divergence, independently usable
`IrisuEnv` branches, nested checkpoints, durable restore detachment, and fail-
closed calls before reset or from the portable backend.

At a history of 1,000 actions, the current
[source-manifested measurement](../../benchmarks/results/exact-pipeline-range-safe-wide-2026-07-20.json)
produced a 28,104-byte durable snapshot, 187.828 us checkpoint, 479.742 us
median branch creation (2,056.212/s), and 95.933 ms durable restore (10.413/s).
The local branch was therefore 199.968x faster at the median.
These timings establish the scaling difference; they are not a portable
performance guarantee.

This path intentionally forks only the single-threaded 32-bit worker. It never
forks the Python training process, and it should not be enabled in sanitizer
workers or on platforms without Linux `fork()` and abstract UNIX sockets.
