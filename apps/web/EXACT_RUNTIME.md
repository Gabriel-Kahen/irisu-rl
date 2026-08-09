# Exact browser runtime provenance

The static client runs the caller-supplied 32-bit `irisu-exact-worker` and
MSVC9 Box2D r58 host inside [v86](https://github.com/copy/v86), without a
server. Generated runtime files are intentionally excluded from the repository.

Pinned components:

- v86 npm package 0.5.432 (`f3d4472a9c934b9ad78a311f5849ba711a296d23`), BSD-2-Clause, with the bounded CORE-MATH x87 trigonometry patch using CORE-MATH commit `07cf01e12a42b82cc478341982936cad7f3f9bdc` under MIT. Its license is distributed as `LICENSE.v86`; the exact patched CORE-MATH source and MIT notice are distributed as `SOURCE.core_math_sincosf.c`.
- The project-owned Linux 6.8.12 i386/initramfs guest, built reproducibly with
  Buildroot 2024.02.13 and pinned as SHA-256
  `d0317109d9cec024f5d01bac9cfd7399d699bcd48816e2373195ac2ee336949c`.
  Its reduced kernel omits IP networking, block devices, high memory, ACPI,
  input, virtual terminals, power management, and other subsystems that the
  worker does not use. The dedicated
  `/irisu-direct-init` mounts only proc and the v86 host filesystem before
  launching the worker directly, avoiding unrelated boot services and a shell
  handoff in the browser startup path.
  The same guest supplies its pinned glibc 2.38 loader, libc, libm, and GCC
  runtime. The runtime build relinks the already-approved application object
  and core archive against this guest ABI without recompiling them, so those
  libraries do not need to cross the network a second time.
  Its complete build recipe is distributed as `SOURCE.guest-build`; the same
  recipe produces a `legal-info.tar.xz` archive containing corresponding
  sources, licenses, generated configs, and manifests.
- SeaBIOS/VGABIOS images from the pinned v86 revision. Their respective
  upstream source and license notices apply.
- Debian 13 i386 libstdc++6 14.2.0-19. Its notice is distributed as
  `COPYRIGHT.gcc-runtime`; it is covered by GPLv3 with the GCC Runtime Library
  Exception.

`SHA256SUMS` records every executable runtime artifact. The linker-only worker
stage pins and verifies its application object, core archive, linker output,
ELF interpreter, dependency set, and `$ORIGIN` lookup path. It removes the
launcher's over-declared `.note.gnu.property`, which otherwise makes glibc
reject v86's Pentium III-compatible CPU model, and strips debug metadata. This
stage reads the pinned application objects byte-for-byte without modifying or
recompiling them. The exact physics host is copied byte-for-byte.

The generated site redistributes the exact worker and physics host as binaries;
they are merely supplied as external inputs to this repository build. Anyone
distributing a generated site is responsible for ensuring they have the rights
to distribute those files. The physics host is built from Box2D r58 under the
zlib license, distributed as `LICENSE.Box2D`.

Source locations and public-distribution responsibilities are documented in
`SOURCE_OFFER.md`.
