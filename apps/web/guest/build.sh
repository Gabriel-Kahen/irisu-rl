#!/usr/bin/env bash
set -euo pipefail

here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
root=$(CDPATH= cd -- "$here/../../.." && pwd)
output=${1:-"$root/build-web/guest"}
output=$(realpath -m -- "$output")
if [[ "$output" == / || "$output" == "$root" || "$output" == "$here" ||
      "$output/" == "$here/"* ]]; then
  echo "refusing unsafe guest output path: $output" >&2
  exit 1
fi

version=2024.02.13
archive_sha=1d3e2f3c6e3d5123a734f0935f4a790650ac4f851f93539a464d4b8fb5dfa04d
source_date_epoch=1725145115
archive="$output/downloads/buildroot-$version.tar.xz"
source="$output/buildroot-$version"
build="$output/output"
host_tools="$output/host-tools"

for command in curl make realpath sha256sum tar xz; do
  command -v "$command" >/dev/null || {
    echo "missing required command: $command" >&2
    exit 1
  }
done

mkdir -p "$output/downloads"
if [[ ! -f "$archive" ]] ||
   ! printf '%s  %s\n' "$archive_sha" "$archive" | sha256sum -c - >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -fL --retry 3 \
    "https://buildroot.org/downloads/buildroot-$version.tar.xz" \
    -o "$archive.tmp"
  printf '%s  %s\n' "$archive_sha" "$archive.tmp" | sha256sum -c -
  mv "$archive.tmp" "$archive"
fi

if [[ ! -f "$source/Makefile" ]]; then
  mkdir -p "$source"
  tar -xJf "$archive" -C "$source" --strip-components=1
fi

# Buildroot deliberately requires a host bc before it can build its own host
# packages. Bootstrap the pinned GNU implementation in the output tree when a
# distribution package is unavailable; its source and license are copied into
# the final legal-info archive below.
if ! command -v bc >/dev/null; then
  bc_version=7.0.3
  bc_sha=91eb74caed0ee6655b669711a4f350c25579778694df248e28363318e03c7fc4
  bc_archive="$output/downloads/bc-$bc_version.tar.xz"
  bc_source="$output/bc-$bc_version"
  if [[ ! -f "$bc_archive" ]] ||
     ! printf '%s  %s\n' "$bc_sha" "$bc_archive" | sha256sum -c - >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -fL --retry 3 \
      "https://github.com/gavinhoward/bc/releases/download/$bc_version/bc-$bc_version.tar.xz" \
      -o "$bc_archive.tmp"
    printf '%s  %s\n' "$bc_sha" "$bc_archive.tmp" | sha256sum -c -
    mv "$bc_archive.tmp" "$bc_archive"
  fi
  if [[ ! -f "$bc_source/configure.sh" ]]; then
    mkdir -p "$bc_source"
    tar -xJf "$bc_archive" -C "$bc_source" --strip-components=1
  fi
  if [[ ! -x "$host_tools/bin/bc" ]]; then
    (
      cd "$bc_source"
      ./configure.sh --prefix="$host_tools" --bc-only --disable-history \
        --disable-generated-tests --disable-man-pages --disable-nls
      make -j2
      make install
    )
  fi
  export PATH="$host_tools/bin:$PATH"
fi

export BR2_DL_DIR="$output/downloads/sources"
export LC_ALL=C
export TZ=UTC
export SOURCE_DATE_EPOCH="$source_date_epoch"
# The extracted Buildroot release lives below the irisu worktree. Prevent its
# version probe from walking into the parent repository and embedding the
# current irisu commit/dirty state in the guest root filesystem.
export GIT_CEILING_DIRECTORIES="$output"

buildroot_make=(make -C "$source" O="$build"
  BR2_EXTERNAL="$here/buildroot-external")
# Several pinned host packages predate GCC 15's GNU C23 default. Keep their
# host-only builds on the language revisions used by this Buildroot release.
host_gcc_version=$(gcc -dumpversion)
host_gcc_major=${host_gcc_version%%.*}
if [[ "$host_gcc_major" =~ ^[0-9]+$ ]] && ((host_gcc_major >= 15)); then
  buildroot_make+=('HOST_CFLAGS=-O2 -std=gnu17')
fi
"${buildroot_make[@]}" irisu_web_guest_defconfig
"${buildroot_make[@]}" all
"${buildroot_make[@]}" legal-info

# Buildroot's legal-info target covers target and host packages, but not the
# Buildroot release itself or this external tree. Preserve both so the archive
# contains the complete recipe, generated configs, sources, and license inputs
# needed to audit or reproduce the guest.
mkdir -p "$build/legal-info/sources" \
  "$build/legal-info/licenses/host-buildroot" \
  "$build/legal-info/licenses/linux-6.8.12" \
  "$build/legal-info/licenses/linux-headers-6.8.12" \
  "$build/legal-info/provenance"
install -m 0644 "$archive" \
  "$build/legal-info/sources/host-buildroot-$version.tar.xz"
install -m 0644 "$source/COPYING" \
  "$build/legal-info/licenses/host-buildroot/COPYING"
install -m 0644 "$build/build/linux-6.8.12/COPYING" \
  "$build/legal-info/licenses/linux-6.8.12/COPYING"
install -m 0644 "$build/build/linux-headers-6.8.12/COPYING" \
  "$build/legal-info/licenses/linux-headers-6.8.12/COPYING"
buildroot_manifest_row="host-buildroot,$version,GPL-2.0-or-later,COPYING,guest build orchestrator"
grep -Fqx "$buildroot_manifest_row" "$build/legal-info/host-manifest.csv" ||
  printf '%s\n' "$buildroot_manifest_row" >>"$build/legal-info/host-manifest.csv"
install -m 0644 "$build/.config" \
  "$build/legal-info/provenance/buildroot.config"
install -m 0644 "$build/build/linux-6.8.12/.config" \
  "$build/legal-info/provenance/linux.config"
mkdir -p "$build/legal-info/provenance/buildroot-external"
cp -a "$here/buildroot-external/." \
  "$build/legal-info/provenance/buildroot-external/"

if [[ -n ${bc_archive:-} ]]; then
  mkdir -p "$build/legal-info/sources" "$build/legal-info/licenses/host-bc"
  install -m 0644 "$bc_archive" "$build/legal-info/sources/host-bc-$bc_version.tar.xz"
  install -m 0644 "$bc_source/LICENSE.md" "$build/legal-info/licenses/host-bc/LICENSE.md"
  bc_manifest_row='host-bc,7.0.3,BSD-2-Clause,LICENSE.md,bootstrap build dependency'
  grep -Fqx "$bc_manifest_row" "$build/legal-info/host-manifest.csv" ||
    printf '%s\n' "$bc_manifest_row" >>"$build/legal-info/host-manifest.csv"
fi

install -m 0644 "$build/images/bzImage" "$output/bzImage"
tar --sort=name --mtime="@$source_date_epoch" --owner=0 --group=0 \
  --numeric-owner -cJf "$output/legal-info.tar.xz" -C "$build" legal-info
(
  cd "$output"
  sha256sum bzImage legal-info.tar.xz > SHA256SUMS
)
echo "built reproducible v86 guest at $output/bzImage"
