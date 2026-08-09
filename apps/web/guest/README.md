# Reproducible exact-browser guest

This Buildroot external tree produces the minimal i386 Linux/initramfs image
used only to boot the exact worker under v86. The physics worker, MSVC9 host,
and their distribution policy are not inputs to this build.

```sh
apps/web/guest/build.sh build-web/guest
```

The output directory contains `bzImage`, `SHA256SUMS`, Buildroot's `legal-info`
tree, and a deterministic `legal-info.tar.xz`. The archive also carries the
Buildroot release and license, the external tree, and the generated Buildroot
and Linux configs. The build pins and hashes Buildroot 2024.02.13; Buildroot
pins the Linux, BusyBox, toolchain, and other source archives recorded by
`legal-info/manifest.csv`.

The diagnostic `/irisu-init` mounts v86's `host9p` filesystem at `/mnt`, emits
`__IRISU_GUEST_READY__` on the serial console, and exposes a raw BusyBox shell.
The browser uses `/irisu-direct-init`, which launches the exact worker directly
after mounting only proc and `host9p`.
`prepare-exact-runtime.sh` builds this image by default, or accepts an existing
copy through `IRISU_GUEST_BZIMAGE` after enforcing its pinned hash. The recipe
passed the exact 20-step v86/native response gate; the final response SHA-256
was `dcb234c9ffb3c0140ebfa98c735c5568b36603999885a733482db7e012f3f9e1`.
