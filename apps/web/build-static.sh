#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
build=${IRISU_WEB_BUILD_DIR:-"$root/build-web"}
output=${1:-"$root/apps/web/dist"}
runtime="$build/exact-runtime"

output=$(realpath -m -- "$output")
runtime=$(realpath -m -- "$runtime")
static=$(realpath -m -- "$root/apps/web/static")
if [[ "$output" == / || "$output" == "$root" ||
      "$runtime" == / || "$runtime" == "$root" ]]; then
  echo "refusing unsafe static-build path" >&2
  exit 1
fi
if [[ "$output" == "$static" || "$output/" == "$static/"* ||
      "$static/" == "$output/"* || "$output/" == "$runtime/"* ||
      "$runtime/" == "$output/"* ]]; then
  echo "refusing overlapping static-build paths" >&2
  exit 1
fi
site_marker=.irisu-static-site

"$root/apps/web/prepare-exact-runtime.sh" "$runtime"
stage=$(mktemp -d "${output}.stage.XXXXXX")
cleanup() { rm -rf -- "$stage"; }
trap cleanup EXIT
cmake -E copy_directory "$static" "$stage"
cmake -E copy_directory "$runtime" "$stage/exact-runtime"
printf '%s\n' 'generated exact static site; safe to replace' > "$stage/$site_marker"
if [[ -e "$output" ]]; then
  if [[ ! -f "$output/$site_marker" ]]; then
    echo "refusing to replace unmarked static-site path: $output" >&2
    exit 1
  fi
  cmake -E remove_directory "$output"
fi
mv "$stage" "$output"
trap - EXIT
echo "built serverless exact static app at $output"
