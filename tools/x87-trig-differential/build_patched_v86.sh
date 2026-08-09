#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_dir=$(CDPATH= cd -- "$script_dir/../.." && pwd)
source_dir=${1:-"$repo_dir/.tmp/v86-core-math-src"}
output=${2:-"$repo_dir/experiments/v86-exact-poc/runtime/v86.wasm"}
commit=f3d4472a9c934b9ad78a311f5849ba711a296d23
core_math_sha=aa391047a7bf5813561def1c4483b1a730a0eac20018b819e3c37441eed610ea
wasm_sha=73d1023eba1729d6aa6a9a3d3d52122c88e8f05b775caaa0557e042f68c34403

printf '%s  %s\n' "$core_math_sha" "$script_dir/core_math_sincosf.c" | sha256sum -c -

if [[ ! -d "$source_dir/.git" ]]; then
  git clone --filter=blob:none https://github.com/copy/v86.git "$source_dir"
fi
git -C "$source_dir" checkout --detach "$commit"
if git -C "$source_dir" apply --check "$script_dir/v86-core-math-x87.patch" 2>/dev/null; then
  git -C "$source_dir" apply "$script_dir/v86-core-math-x87.patch"
elif ! git -C "$source_dir" apply --reverse --check "$script_dir/v86-core-math-x87.patch" 2>/dev/null; then
  echo "v86 source is neither clean nor already patched" >&2
  exit 1
fi

rustup target add wasm32-unknown-unknown
make -C "$source_dir" CORE_MATH_SINCOSF="$script_dir/core_math_sincosf.c" build/v86.wasm
mkdir -p "$(dirname -- "$output")"
cp -f "$source_dir/build/v86.wasm" "$output"
printf '%s  %s\n' "$wasm_sha" "$output" | sha256sum -c -
