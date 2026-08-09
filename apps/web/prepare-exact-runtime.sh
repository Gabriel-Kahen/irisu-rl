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
patched_v86=${IRISU_V86_WASM:-}
guest_override=${IRISU_GUEST_BZIMAGE:-}
guest=${guest_override:-"$web_build/guest/bzImage"}

v86_version=0.5.432
v86_commit=f3d4472a9c934b9ad78a311f5849ba711a296d23
v86_sha=de9379ee1ccc118903558faed9ff577a66d486c5551b9e5ef359f0d388c40ebb
guest_sha=681388b6db219fbb1dc63a678cd276d73c21bbb047cd8c7a6771fc4e567591c0
seabios_sha=73e3f359102e3a9982c35fce98eb7cd08f18303ac7f1ba6ebfbe6cdc1c244d98
vgabios_sha=a4bc0d80cc3ca028c73dafa8fee396b8d054ce87ebd8abfbd31b06b437607880
libc_sha=410dae774925cb89a959a595bb9c9766f910df9ce3fdba334544fd7f0cb04b7e
libgcc_sha=a4c71fd856d2a48a7505a087b4186e3cca23f94603c05e3fb7c799b27e72f761
libstdcpp_sha=b6020260b92a97ac33ae58a73b16f3ab31fed7632e9b861f7cd5fc393facd6ed
gcc_base_sha=cab3cc6782d6cd3445d184ad317bbd8cc46395eb2675ccd75f4b57147469887a
exact_worker_sha=4faa4508a89df3e1e62b80e2871b6a35b5913f220d53fe5de43408ad6512c261
exact_host_sha=ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5
patched_v86_sha=73d1023eba1729d6aa6a9a3d3d52122c88e8f05b775caaa0557e042f68c34403
staged_worker_sha=410bc7a49e345f433bfc331deacb62dc534b01ded6ba3dc6c5fc29ff0c71532f

for command in ar cmake curl objcopy sha256sum tar; do
  command -v "$command" >/dev/null || { echo "missing required command: $command" >&2; exit 1; }
done
for input in "$worker" "$host"; do
  [[ -f "$input" ]] || { echo "missing exact runtime input: $input" >&2; exit 1; }
done
printf '%s  %s\n' "$exact_worker_sha" "$worker" | sha256sum -c -
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
libc="$cache/libc6_2.41-12+deb13u3_i386.deb"
libgcc="$cache/libgcc-s1_14.2.0-19_i386.deb"
libstdcpp="$cache/libstdc++6_14.2.0-19_i386.deb"
gcc_base="$cache/gcc-14-base_14.2.0-19_i386.deb"
fetch "https://registry.npmjs.org/v86/-/v86-$v86_version.tgz" "$v86" "$v86_sha"
fetch "https://raw.githubusercontent.com/copy/v86/$v86_commit/bios/seabios.bin" "$cache/seabios.bin" "$seabios_sha"
fetch "https://raw.githubusercontent.com/copy/v86/$v86_commit/bios/vgabios.bin" "$cache/vgabios.bin" "$vgabios_sha"
fetch "https://deb.debian.org/debian/pool/main/g/glibc/libc6_2.41-12+deb13u3_i386.deb" "$libc" "$libc_sha"
fetch "https://deb.debian.org/debian/pool/main/g/gcc-14/libgcc-s1_14.2.0-19_i386.deb" "$libgcc" "$libgcc_sha"
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
for package in "$libc" "$libgcc" "$libstdcpp" "$gcc_base"; do
  ar p "$package" data.tar.xz | tar -xJ -C "$debian"
done
debian_lib="$debian/usr/lib/i386-linux-gnu"

# Arch's i386 CRT over-declares this launcher as x86-v3. Its text is baseline
# i386/SSE, so remove only that loader policy note. The exact physics host and
# the launcher's executable sections are not rewritten.
objcopy --remove-section .note.gnu.property "$worker" "$stage/guest/irisu-exact-worker"
printf '%s  %s\n' "$staged_worker_sha" "$stage/guest/irisu-exact-worker" | \
  sha256sum -c -
cp "$host" "$stage/guest/libirisu_box2d_msvc_exact_multiworld.so"
cp -L "$debian_lib/ld-linux.so.2" "$stage/guest/ld-linux.so.2"
cp -L "$debian_lib/libc.so.6" "$stage/guest/libc.so.6"
cp -L "$debian_lib/libm.so.6" "$stage/guest/libm.so.6"
cp -L "$debian_lib/libgcc_s.so.1" "$stage/guest/libgcc_s.so.1"
cp -L "$debian_lib/libstdc++.so.6" "$stage/guest/libstdc++.so.6"
cp "$debian/usr/share/doc/libc6/copyright" "$stage/COPYRIGHT.glibc"
cp "$debian/usr/share/doc/gcc-14-base/copyright" "$stage/COPYRIGHT.gcc-runtime"
cp "$root/tools/x87-trig-differential/core_math_sincosf.c" \
  "$stage/SOURCE.core_math_sincosf.c"
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
chmod 0755 "$stage/guest/irisu-exact-worker" "$stage/guest/ld-linux.so.2"
(
  cd "$stage"
  sha256sum libv86.js v86.wasm buildroot-bzimage68.bin seabios.bin vgabios.bin \
    SOURCE.core_math_sincosf.c LICENSE.Box2D guest/* > SHA256SUMS
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
