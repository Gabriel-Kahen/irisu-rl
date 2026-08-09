#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /path/to/libirisu_box2d_msvc_exact_multiworld.so /new/output-dir" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
host=$1
output=$2

python3 "$script_dir/package_exact_host.py" "$host" "$output"
clang --target=wasm32 -O2 -Wall -Wextra -Werror -ffreestanding -fno-ident -nostdlib \
  -I"$output" "$script_dir/smoke.c" \
  -Wl,--no-entry \
  -Wl,--export-memory \
  -Wl,--initial-memory=262144 \
  -Wl,--strip-all \
  -o "$output/exact-host-smoke.wasm"
node "$script_dir/smoke.mjs" "$output/exact-host-smoke.wasm"
