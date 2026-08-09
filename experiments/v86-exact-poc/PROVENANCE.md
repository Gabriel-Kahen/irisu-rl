# Static runtime provenance review

The PoC is serverless at run time, but a clean checkout cannot currently build
or package it. `runtime/` is ignored and `prepare.sh` defaults to three local
build outputs:

- exact host: `.tmp/exact-range-host-symbolic-20260721/...so`;
- exact worker and replay runner: `build-physics-integration-exact-multiworld-2/`.

The exact host used by the parity gate is
`ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5`.
The native worker and replay-runner inputs are respectively
`4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261`
and `d06e1ff08811b21047b827bebb27ea39295445b64d3b56c772617befb1bc22f9`.
Removing only their `.note.gnu.property` sections for v86 produces
`410bc7a49e345f433bfc331deacb62dc534b01ded6ba3dc6c5fc29ff0c71532f`
and `021c6bb768e4c15eb42777d86cb1b56909ccd35f066cfec9cee1d6e2502f1766`.

## Clean static-build blockers

1. The exact host is an external generated input to CMake. Its source
   compiler/toolchain inputs are not a clean-checkout dependency, and the PoC
   does not enforce its expected hash before packaging.
2. The worker is only built by a native 32-bit `exact-msvc` configuration.
   The normal Emscripten web build selects the portable backend and cannot
   produce this ELF32 guest executable.
3. `prepare.sh` records `SHA256SUMS` after staging but does not authenticate the
   caller-supplied host, worker, or runner against a committed manifest.
4. The currently tested worker and runner carry a build-tree absolute RUNPATH.
   v86 works because the explicit i386 loader uses `--library-path /mnt`, but a
   release should be built/installed with a relative RUNPATH before the note
   section is stripped and hashed.
5. v86, the BIOS files, and Debian i386 runtime libraries are downloaded and
   pinned by SHA-256. The kernel is the reproducible project-owned Buildroot
   guest under `apps/web/guest`. All staged outputs are ignored. A static release
   needs a deterministic packaging target that emits these files and a
   committed manifest. It also needs redistribution/license notices for v86,
   SeaBIOS, the Linux kernel, glibc, libgcc, and libstdc++.

## Minimum production gate

Make the exact host, installed worker, and runtime manifest explicit inputs to
one packaging command. Reject unknown hashes; strip the worker deterministically;
verify the staged manifest; run `node-smoke.mjs`; then run
`evaluate-corpus.py`. Publish only that verified directory. This produces a
static browser payload and requires no server-side execution.
