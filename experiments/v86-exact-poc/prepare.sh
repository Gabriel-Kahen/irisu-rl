#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)
runtime_dir="$script_dir/runtime"
guest_dir="$runtime_dir/guest"

worker=${1:-"$repo_dir/build-physics-integration-exact-multiworld-2/irisu-exact-worker"}
host=${2:-"$repo_dir/.tmp/exact-range-host-symbolic-20260721/libirisu_box2d_msvc_exact_multiworld.so"}
runner=${3:-"$repo_dir/build-physics-integration-exact-multiworld-2/irisu-exact-replay"}
guest_override=${IRISU_GUEST_BZIMAGE:-}
guest_image=${guest_override:-"$repo_dir/build-web/guest/bzImage"}

v86_version=0.5.432
v86_archive_sha=de9379ee1ccc118903558faed9ff577a66d486c5551b9e5ef359f0d388c40ebb
kernel_sha=389fb6e37c9f9f101232ad68b7177bced98caee9f7a531e99ea00b836833ea33
seabios_sha=73e3f359102e3a9982c35fce98eb7cd08f18303ac7f1ba6ebfbe6cdc1c244d98
vgabios_sha=a4bc0d80cc3ca028c73dafa8fee396b8d054ce87ebd8abfbd31b06b437607880
v86_commit=f3d4472a9c934b9ad78a311f5849ba711a296d23
libc_deb_sha=410dae774925cb89a959a595bb9c9766f910df9ce3fdba334544fd7f0cb04b7e
libgcc_deb_sha=a4c71fd856d2a48a7505a087b4186e3cca23f94603c05e3fb7c799b27e72f761
libstdcpp_deb_sha=b6020260b92a97ac33ae58a73b16f3ab31fed7632e9b861f7cd5fc393facd6ed

for required in "$worker" "$host" "$runner"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required file: $required" >&2
    exit 1
  fi
done

mkdir -p "$runtime_dir" "$guest_dir"

fetch() {
  local url=$1 output=$2 expected=$3
  if [[ ! -f "$output" ]] || ! printf '%s  %s\n' "$expected" "$output" | sha256sum -c - >/dev/null 2>&1; then
    curl -fL --retry 3 --output "$output.tmp" "$url"
    printf '%s  %s\n' "$expected" "$output.tmp" | sha256sum -c -
    mv "$output.tmp" "$output"
  fi
}

archive="$runtime_dir/v86-$v86_version.tgz"
fetch "https://registry.npmjs.org/v86/-/v86-$v86_version.tgz" "$archive" "$v86_archive_sha"
if [[ -z "$guest_override" ]] &&
   { [[ ! -f "$guest_image" ]] ||
     ! printf '%s  %s\n' "$kernel_sha" "$guest_image" | sha256sum -c - >/dev/null 2>&1; }; then
  "$repo_dir/apps/web/guest/build.sh" "$repo_dir/build-web/guest"
fi
printf '%s  %s\n' "$kernel_sha" "$guest_image" | sha256sum -c -
cp -f "$guest_image" "$runtime_dir/buildroot-bzimage68.bin"
fetch "https://raw.githubusercontent.com/copy/v86/$v86_commit/bios/seabios.bin" "$runtime_dir/seabios.bin" "$seabios_sha"
fetch "https://raw.githubusercontent.com/copy/v86/$v86_commit/bios/vgabios.bin" "$runtime_dir/vgabios.bin" "$vgabios_sha"
fetch "https://deb.debian.org/debian/pool/main/g/glibc/libc6_2.41-12+deb13u3_i386.deb" "$runtime_dir/libc6-i386.deb" "$libc_deb_sha"
fetch "https://deb.debian.org/debian/pool/main/g/gcc-14/libgcc-s1_14.2.0-19_i386.deb" "$runtime_dir/libgcc-s1-i386.deb" "$libgcc_deb_sha"
fetch "https://deb.debian.org/debian/pool/main/g/gcc-14/libstdc++6_14.2.0-19_i386.deb" "$runtime_dir/libstdc++6-i386.deb" "$libstdcpp_deb_sha"

tar -xzf "$archive" -C "$runtime_dir" --strip-components=2 \
  package/build/libv86.js package/build/libv86.mjs package/build/v86.wasm

debian_root=$(mktemp -d "$runtime_dir/debian-root.XXXXXX")
cleanup() { rm -rf -- "$debian_root"; }
trap cleanup EXIT
for package in "$runtime_dir/libc6-i386.deb" "$runtime_dir/libgcc-s1-i386.deb" "$runtime_dir/libstdc++6-i386.deb"; do
  ar p "$package" data.tar.xz | tar -xJ -C "$debian_root"
done
debian_lib="$debian_root/usr/lib/i386-linux-gnu"

# Arch's i386 linker over-declares this launcher as x86-64-v3 even though its
# text uses the baseline i386/SSE subset. v86 intentionally exposes a Pentium
# III-like CPU, so glibc rejects the launcher before main unless this metadata
# is omitted. This does not alter its executable code or the exact MSVC9 host.
objcopy --remove-section .note.gnu.property "$worker" "$guest_dir/irisu-exact-worker"
objcopy --remove-section .note.gnu.property "$runner" "$guest_dir/irisu-exact-replay"
cp -f "$host" "$guest_dir/libirisu_box2d_msvc_exact_multiworld.so"
cp -Lf "$debian_lib/ld-linux.so.2" "$guest_dir/ld-linux.so.2"
cp -Lf "$debian_lib/libc.so.6" "$guest_dir/libc.so.6"
cp -Lf "$debian_lib/libm.so.6" "$guest_dir/libm.so.6"
cp -Lf "$debian_lib/libgcc_s.so.1" "$guest_dir/libgcc_s.so.1"
cp -Lf "$debian_lib/libstdc++.so.6" "$guest_dir/libstdc++.so.6"
chmod 0755 "$guest_dir/irisu-exact-worker" "$guest_dir/irisu-exact-replay" "$guest_dir/ld-linux.so.2"

(
  cd "$runtime_dir"
  sha256sum libv86.js libv86.mjs v86.wasm buildroot-bzimage68.bin seabios.bin vgabios.bin guest/* > SHA256SUMS
)

cleanup
trap - EXIT
echo "prepared $runtime_dir"
du -sh "$runtime_dir"
