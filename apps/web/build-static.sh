#!/usr/bin/env sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
build=${IRISU_WEB_BUILD_DIR:-"$root/build-web"}
output=${1:-"$root/apps/web/dist"}

emcmake cmake -S "$root" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DIRISU_BUILD_BENCHMARKS=OFF \
  -DIRISU_BUILD_SHARED=OFF \
  -DIRISU_BUILD_TESTS=OFF \
  -DIRISU_BUILD_WEB=ON \
  -DIRISU_PHYSICS_BACKEND=portable
cmake --build "$build" --target irisu-web-module -j2
cmake -E make_directory "$output"
cmake -E copy_directory "$root/apps/web/static" "$output"
cmake -E copy "$build/irisu-wasm.js" "$build/irisu-wasm.wasm" "$output"
