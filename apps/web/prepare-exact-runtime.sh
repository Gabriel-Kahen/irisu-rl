#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
output=${1:?usage: prepare-exact-runtime.sh OUTPUT_DIR}
web_build=${IRISU_WEB_BUILD_DIR:-"$root/build-web"}
output=$(realpath -m -- "$output")
web_build=$(realpath -m -- "$web_build")
if [[ "$output" == / || "$output" == "$root" ||
      "$web_build" == / || "$web_build" == "$root" ]]; then
  echo "refusing unsafe exact-runtime path" >&2
  exit 1
fi
runtime_marker=.irisu-exact-runtime
cache=${IRISU_WEB_DOWNLOAD_CACHE:-"$web_build/downloads"}
worker=${IRISU_EXACT_WORKER:-"$root/build-physics-integration-exact-multiworld-2/irisu-exact-worker"}
host=${IRISU_EXACT_HOST:-"$root/.tmp/exact-range-host-symbolic-20260721/libirisu_box2d_msvc_exact_multiworld.so"}
worker_build=$(dirname -- "$worker")
worker_object=${IRISU_EXACT_WORKER_OBJECT:-"$worker_build/CMakeFiles/irisu-exact-worker.dir/tools/exact-physics-prototype/ipc_worker.cpp.o"}
core_archive=${IRISU_EXACT_CORE_ARCHIVE:-"$worker_build/libirisu_core.a"}
patched_v86=${IRISU_V86_WASM:-}
guest_override=${IRISU_GUEST_BZIMAGE:-}
guest=${guest_override:-"$web_build/guest/bzImage"}
guest_toolchain=${IRISU_GUEST_TOOLCHAIN_DIR:-"$web_build/guest/output/host/bin"}

v86_version=0.5.432
v86_commit=f3d4472a9c934b9ad78a311f5849ba711a296d23
v86_sha=de9379ee1ccc118903558faed9ff577a66d486c5551b9e5ef359f0d388c40ebb
guest_sha=d0317109d9cec024f5d01bac9cfd7399d699bcd48816e2373195ac2ee336949c
seabios_sha=73e3f359102e3a9982c35fce98eb7cd08f18303ac7f1ba6ebfbe6cdc1c244d98
vgabios_sha=a4bc0d80cc3ca028c73dafa8fee396b8d054ce87ebd8abfbd31b06b437607880
libstdcpp_sha=b6020260b92a97ac33ae58a73b16f3ab31fed7632e9b861f7cd5fc393facd6ed
gcc_base_sha=cab3cc6782d6cd3445d184ad317bbd8cc46395eb2675ccd75f4b57147469887a
exact_worker_sha=4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261
worker_object_sha=812b4876d588ae9539ac164d27d2ca5efd96d423428e4367f6145d36b79e9bba
core_archive_sha=442aefadd8b65f65ccc036e93047f7181458d384ff07eb280ca0c92ecc194c6e
exact_host_sha=ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5
patched_v86_sha=73d1023eba1729d6aa6a9a3d3d52122c88e8f05b775caaa0557e042f68c34403
staged_worker_sha=8ef81521e81a5b2a764c305ac48dad997b28476bcd2fccbd1c9aed9603322854

for command in ar cmake curl sha256sum tar; do
  command -v "$command" >/dev/null || { echo "missing required command: $command" >&2; exit 1; }
done
for input in "$worker" "$worker_object" "$core_archive" "$host"; do
  [[ -f "$input" ]] || { echo "missing exact runtime input: $input" >&2; exit 1; }
done
printf '%s  %s\n' "$exact_worker_sha" "$worker" | sha256sum -c -
printf '%s  %s\n' "$worker_object_sha" "$worker_object" | sha256sum -c -
printf '%s  %s\n' "$core_archive_sha" "$core_archive" | sha256sum -c -
printf '%s  %s\n' "$exact_host_sha" "$host" | sha256sum -c -
mkdir -p "$web_build"
if [[ -z "$patched_v86" ]]; then
  patched_v86="$web_build/v86-core-math.wasm"
  "$root/tools/x87-trig-differential/build_patched_v86.sh" \
    "$web_build/v86-core-math-src" "$patched_v86"
fi
[[ -f "$patched_v86" ]] || { echo "missing patched v86 Wasm: $patched_v86" >&2; exit 1; }
printf '%s  %s\n' "$patched_v86_sha" "$patched_v86" | sha256sum -c -
if [[ -z "$guest_override" ]] &&
   { [[ ! -f "$guest" ]] ||
     ! printf '%s  %s\n' "$guest_sha" "$guest" | sha256sum -c - >/dev/null 2>&1; }; then
  "$root/apps/web/guest/build.sh" "$web_build/guest"
fi
if [[ ! -x "$guest_toolchain/i686-buildroot-linux-gnu-gcc" ]]; then
  "$root/apps/web/guest/build.sh" "$web_build/guest"
fi
[[ -f "$guest" ]] || { echo "missing project guest image: $guest" >&2; exit 1; }
printf '%s  %s\n' "$guest_sha" "$guest" | sha256sum -c -

mkdir -p "$cache" "$(dirname -- "$output")"
fetch() {
  local url=$1 target=$2 expected=$3
  if [[ ! -f "$target" ]] || ! printf '%s  %s\n' "$expected" "$target" | sha256sum -c - >/dev/null 2>&1; then
    curl -fL --retry 3 --output "$target.tmp" "$url"
    printf '%s  %s\n' "$expected" "$target.tmp" | sha256sum -c -
    mv "$target.tmp" "$target"
  fi
}

v86="$cache/v86-$v86_version.tgz"
libstdcpp="$cache/libstdc++6_14.2.0-19_i386.deb"
gcc_base="$cache/gcc-14-base_14.2.0-19_i386.deb"
fetch "https://registry.npmjs.org/v86/-/v86-$v86_version.tgz" "$v86" "$v86_sha"
fetch "https://raw.githubusercontent.com/copy/v86/$v86_commit/bios/seabios.bin" "$cache/seabios.bin" "$seabios_sha"
fetch "https://raw.githubusercontent.com/copy/v86/$v86_commit/bios/vgabios.bin" "$cache/vgabios.bin" "$vgabios_sha"
fetch "https://deb.debian.org/debian/pool/main/g/gcc-14/libstdc++6_14.2.0-19_i386.deb" "$libstdcpp" "$libstdcpp_sha"
fetch "https://deb.debian.org/debian/pool/main/g/gcc-14/gcc-14-base_14.2.0-19_i386.deb" "$gcc_base" "$gcc_base_sha"

stage=$(mktemp -d "${output}.stage.XXXXXX")
debian=$(mktemp -d "${output}.debian.XXXXXX")
cleanup() { rm -rf -- "$stage" "$debian"; }
trap cleanup EXIT
mkdir -p "$stage/guest"

tar -xzf "$v86" -C "$stage" --strip-components=1 \
  package/LICENSE package/build/libv86.js
mv "$stage/LICENSE" "$stage/LICENSE.v86"
mv "$stage/build/libv86.js" "$stage/libv86.js"
rmdir "$stage/build"
cp "$patched_v86" "$stage/v86.wasm"
cp "$guest" "$stage/buildroot-bzimage68.bin"
cp "$cache/seabios.bin" "$cache/vgabios.bin" "$stage"
for package in "$libstdcpp" "$gcc_base"; do
  ar p "$package" data.tar.xz | tar -xJ -C "$debian"
done
debian_lib="$debian/usr/lib/i386-linux-gnu"

# Relink only the approved application objects against the guest's pinned
# Buildroot glibc. The application's compiled code and exact physics host are
# reused unchanged.
IRISU_GUEST_TOOLCHAIN_DIR="$guest_toolchain" \
  "$root/apps/web/relink-exact-worker.sh" "$worker_object" "$core_archive" \
  "$host" "$debian_lib/libstdc++.so.6" "$stage/guest/irisu-exact-worker"
printf '%s  %s\n' "$staged_worker_sha" "$stage/guest/irisu-exact-worker" | \
  sha256sum -c -
cp "$host" "$stage/guest/libirisu_box2d_msvc_exact_multiworld.so"
cp -L "$debian_lib/libstdc++.so.6" "$stage/guest/libstdc++.so.6"
cp "$debian/usr/share/doc/gcc-14-base/copyright" "$stage/COPYRIGHT.gcc-runtime"
cp "$root/tools/x87-trig-differential/core_math_sincosf.c" \
  "$stage/SOURCE.core_math_sincosf.c"
cp "$root/apps/web/relink-exact-worker.sh" \
  "$stage/SOURCE.relink-exact-worker.sh"
cp "$root/third_party/box2d_legacy/License.txt" "$stage/LICENSE.Box2D"
cp "$root/apps/web/EXACT_RUNTIME.md" "$stage/PROVENANCE.md"
cp "$root/apps/web/EXACT_RUNTIME_SOURCES.md" "$stage/SOURCE_OFFER.md"
mkdir -p "$stage/SOURCE.guest-build"
cp "$root/apps/web/guest/build.sh" "$root/apps/web/guest/README.md" \
  "$stage/SOURCE.guest-build/"
cp -a "$root/apps/web/guest/buildroot-external" \
  "$stage/SOURCE.guest-build/"
printf '%s\n' 'generated exact browser runtime; safe to replace' > \
  "$stage/$runtime_marker"
chmod 0755 "$stage/guest/irisu-exact-worker"
(
  cd "$stage"
  sha256sum libv86.js v86.wasm buildroot-bzimage68.bin seabios.bin vgabios.bin \
    SOURCE.core_math_sincosf.c SOURCE.relink-exact-worker.sh LICENSE.Box2D \
    guest/* > SHA256SUMS
)

mkdir -p "$(dirname -- "$output")"
if [[ -e "$output" ]]; then
  if [[ ! -f "$output/$runtime_marker" ]]; then
    echo "refusing to replace unmarked exact-runtime path: $output" >&2
    exit 1
  fi
  cmake -E remove_directory "$output"
fi
mv "$stage" "$output"
trap - EXIT
rm -rf -- "$debian"
echo "prepared exact browser runtime at $output"
