#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
object=${1:?usage: relink-exact-worker.sh WORKER_OBJECT CORE_ARCHIVE HOST_SO LIBSTDCXX OUTPUT}
core=${2:?usage: relink-exact-worker.sh WORKER_OBJECT CORE_ARCHIVE HOST_SO LIBSTDCXX OUTPUT}
host=${3:?usage: relink-exact-worker.sh WORKER_OBJECT CORE_ARCHIVE HOST_SO LIBSTDCXX OUTPUT}
libstdcpp=${4:?usage: relink-exact-worker.sh WORKER_OBJECT CORE_ARCHIVE HOST_SO LIBSTDCXX OUTPUT}
output=${5:?usage: relink-exact-worker.sh WORKER_OBJECT CORE_ARCHIVE HOST_SO LIBSTDCXX OUTPUT}
toolchain=${IRISU_GUEST_TOOLCHAIN_DIR:-"$root/build-web/guest/output/host/bin"}
cc="$toolchain/i686-buildroot-linux-gnu-gcc"
objcopy="$toolchain/i686-buildroot-linux-gnu-objcopy"
strip="$toolchain/i686-buildroot-linux-gnu-strip"

for input in "$object" "$core" "$host" "$libstdcpp"; do
  [[ -f "$input" ]] || { echo "missing relink input: $input" >&2; exit 1; }
done
for tool in "$cc" "$objcopy" "$strip"; do
  [[ -x "$tool" ]] || { echo "missing guest toolchain program: $tool" >&2; exit 1; }
done
command -v readelf >/dev/null || { echo "missing required command: readelf" >&2; exit 1; }

mkdir -p "$(dirname -- "$output")"
linked=$(mktemp "${output}.linked.XXXXXX")
staged=$(mktemp "${output}.staged.XXXXXX")
cleanup() { rm -f -- "$linked" "$staged"; }
trap cleanup EXIT

# Read the already-approved application objects byte-for-byte without modifying
# or recompiling them, but select the glibc ABI from the reproducible guest
# toolchain. This makes the worker use libc, libm, libgcc, and the ELF loader
# already present in the kernel's initramfs.
"$cc" -pie -Wl,-z,now -Wl,--no-as-needed \
  '-Wl,--disable-new-dtags,-rpath,$ORIGIN' \
  -o "$linked" "$object" "$core" "$host" "$libstdcpp" -lm -ldl

# Remove the host compiler's incorrect x86-v3 policy note before stripping.
# This order is intentional: it also removes the now-empty property segment.
"$objcopy" --remove-section .note.gnu.property "$linked" "$staged"
"$strip" --strip-all "$staged"

interpreter=$(readelf -lW "$staged" | sed -n 's/.*Requesting program interpreter: \(.*\)]/\1/p')
[[ "$interpreter" == /lib/ld-linux.so.2 ]] || {
  echo "unexpected relinked-worker interpreter: $interpreter" >&2
  exit 1
}
mapfile -t needed < <(readelf -dW "$staged" |
  sed -n 's/.*Shared library: \[\(.*\)\]/\1/p' | sort)
expected=(ld-linux.so.2 libc.so.6 libgcc_s.so.1 libirisu_box2d_msvc_exact_multiworld.so libm.so.6 libstdc++.so.6)
[[ "${needed[*]}" == "${expected[*]}" ]] || {
  echo "unexpected relinked-worker dependencies: ${needed[*]}" >&2
  exit 1
}
readelf -dW "$staged" | grep -Fq 'Library rpath: [$ORIGIN]' || {
  echo 'relinked worker is missing its $ORIGIN rpath' >&2
  exit 1
}

chmod 0755 "$staged"
mv -f -- "$staged" "$output"
trap - EXIT
rm -f -- "$linked"
