# Runtime corresponding source

The generated static artifact contains redistributable open-source guest
components. Exact versions, binary hashes, and licenses are recorded in
`PROVENANCE.md`, `SHA256SUMS`, and the accompanying notice files.

Upstream source is available from:

- v86 and pinned BIOS inputs: https://github.com/copy/v86/tree/f3d4472a9c934b9ad78a311f5849ba711a296d23
- CORE-MATH source used by the bounded x87 trigonometry patch: https://gitlab.inria.fr/core-math/core-math/-/tree/07cf01e12a42b82cc478341982936cad7f3f9bdc
- The project-owned Linux 6.8.12 guest is built by the bundled
  `SOURCE.guest-build/build.sh` and Buildroot external tree. The pinned recipe
  downloads hash-verified Buildroot 2024.02.13 inputs and emits
  `legal-info.tar.xz`, including package sources, licenses, manifests, patches,
  and the generated Buildroot/Linux configurations.
- The linker-only worker staging recipe is bundled as
  `SOURCE.relink-exact-worker.sh`.
- Debian GCC 14.2.0-19 source package for libstdc++: https://sources.debian.org/src/gcc-14/14.2.0-19/

This local artifact is not a public-distribution approval. A distributor must
retain the exact source and build material required by every component license
(including the generated guest `legal-info.tar.xz`) and independently confirm
the rights to distribute the supplied exact worker and physics host. This file
is build provenance, not legal advice.
