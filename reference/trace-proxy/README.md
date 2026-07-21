# Original-game Box2D trace proxy

This directory contains a redistributable, freestanding PE32 forwarding DLL.
It exposes the exact 16-function `stdcall` ABI of IriSu v2.03's shipped
`Box2D.dll`, loads the authentic DLL as `Box2D.real.dll`, and records the calls
that decide body creation, first-step contacts, and destruction. It contains no
bytes from the original DLL and is an observation tool, not a clone dependency.

## Safe workflow

Always create a fresh disposable run and pass its path explicitly:

```bash
run=$(tools/create-reference-run.sh seed41-lifecycle-002)
tools/deploy-box2d-trace-proxy.sh "$run"
IRISU_GAME_DIR="$run" tools/launch-reference-game.sh
python3 reference/trace-proxy/validate_trace.py \
  "$run/data/dll/box2d-trace.jsonl"
```

The deployment script builds and validates the proxy, then requires all of the
following before changing anything:

- the canonical target is one direct child of `reference/runs/`;
- no symlink exists anywhere in the run tree;
- no process currently has the run or a child directory as its working directory;
- `irisu.exe` and the active `Box2D.dll` have the pinned v2.03 hashes;
- neither a prior `Box2D.real.dll`, deployment marker, nor trace exists;
- no copied `launch-irisu.sh` is present.

That last check matters because the historical launcher in the preserved local
game tree points at `/home/gabe/Games/Irisu Syndrome`. New runs strip it. Use
the exact `IRISU_GAME_DIR=... tools/launch-reference-game.sh` command printed by
the deployer. The deployer refuses the preserved source tree, nested run paths,
symlink targets, already deployed runs, and non-authentic binaries.
Deployment holds an exclusive per-run lock and uses no-replace renames for the
authentic DLL, proxy, and marker. Its rollback only removes hash-verified files
that it installed, preserving unexpected concurrent paths rather than deleting
them.

On success, only the explicit disposable run changes:

```text
data/dll/Box2D.real.dll     authentic DLL, renamed without modification
data/dll/Box2D.dll          trace proxy
data/dll/box2d-trace.jsonl  trace created on the next game launch
.box2d-trace-proxy          deployment hashes and trace location
```

The marker also records source/build-script hashes, compiler/linker identities,
and the validated export/import contract, so a capture bundle can retain the
exact instrumentation provenance rather than a floating candidate binary.

The proxy opens the trace beside itself rather than relative to the process's
working directory. It uses create-new semantics: a second process cannot
overwrite a decisive trace. If the trace already exists or cannot be created,
the proxy deliberately does not load the real DLL and `b2d_init` fails. Preserve
the trace and create a fresh disposable run for another trial.

## Trace contents

JSONL schema 1 has a contiguous `seq`. The first record proves whether the
authentic sibling loaded and all exports resolved. Subsequent records cover:

- every world init and disposal;
- box, triangle, and circle creation with per-world body ordinals;
- every user-data assignment and the body-ordinal/user-pointer mapping;
- every position and velocity setter;
- every physics step;
- every contact-cursor call, including the final false result;
- every destruction, including destruction during contact enumeration.

Float inputs are `args_f32` or `dt_f32` strings containing their raw
IEEE-754 binary32 word in hexadecimal. The argument order is the recovered ABI
in [`../binary-analysis.md`](../binary-analysis.md). For example:

```python
import struct
value = struct.unpack("!f", bytes.fromhex("3ca3d70a"))[0]
```

The load and init records also capture the x87 control word before any trace
float formatting (the proxy never performs such formatting) and immediately
around authentic initialization. Body addresses and user-data pointers are
diagnostic process-local integers; body ordinals are stable within one `init`
world. Historical user mappings remain available after destruction because the
original game destroys bodies while continuing to walk the DLL's cached
contact cursor.

## Build and validation

The build uses the same installed Clang/LLD/LLVM-dlltool freestanding approach
as the standalone shipped-DLL probe:

```bash
reference/trace-proxy/build.sh
reference/trace-proxy/test.sh
```

`validate_build.py` requires a PE32/i386 DLL, the exact decorated exports and
ordinals, and only `KERNEL32.dll` static imports. `test.sh` builds in isolation,
checks preserved-tree, nested-symlink, and unsafe-launcher refusal, and—when the
ignored local v2.03 files are present—performs a disposable successful
deployment plus a redeployment refusal. It never launches Wine or the game.

An explicit, slower transparency test runs the existing standalone Box2D oracle
through the proxy and requires its 1,290-record numerical output to remain
byte-for-byte equal to the direct-DLL golden trace:

```bash
reference/trace-proxy/test_transparency.sh
```

This optional test invokes the isolated console probe under Wine; it never
launches the game or modifies a reference run.

On 2026-07-19 this test passed with the underlying 1,290-record probe trace
byte-identical to the direct-DLL golden artifact. The independent proxy trace
validated 1,994 calls: 12 init/dispose worlds, 48 creates, 49 destroys, 549
steps, and 1,263 contact-cursor calls, plus setters and the load record. This is
strong ABI/numerical transparency evidence for the standalone oracle scenarios;
it is not a substitute for the pending instrumented original-game replay.
