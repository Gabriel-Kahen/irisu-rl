#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 /path/to/libirisu_box2d_msvc_exact.so branch_output [public_rebuild_output]" >&2
  exit 2
fi

repo_root=$(cd "$(dirname "$0")/../.." && pwd)
exact_library=$(realpath "$1")
output=$(realpath -m "$2")
exact_directory=$(dirname "$exact_library")
exact_name=$(basename "$exact_library")
exact_stem=${exact_name#lib}
exact_stem=${exact_stem%.so}
compat_source=()
if ! nm -D --defined-only "$exact_library" | awk '{print $3}' | \
    grep -qx b2d_world_create; then
  compat_source+=("$repo_root/tools/exact-physics-prototype/single_world_compat.cpp")
fi

g++ -m32 -std=c++20 -O2 -DNDEBUG -pthread \
  -I"$repo_root/clone/include" \
  "$repo_root/clone/core/config.cpp" \
  "$repo_root/clone/core/dx_random.cpp" \
  "$repo_root/clone/core/normal_rules.cpp" \
  "$repo_root/clone/core/simulator.cpp" \
  "$repo_root/tools/exact-physics-prototype/physics_wrapper_forward.cpp" \
  "${compat_source[@]}" \
  "$repo_root/tools/exact-physics-prototype/branch_snapshot_runner.cpp" \
  -L"$exact_directory" -Wl,-rpath,"$exact_directory" \
  -Wl,-z,notext -l"$exact_stem" -o "$output"

echo "$output"

if [[ $# -eq 3 ]]; then
  rebuild_output=$(realpath -m "$3")
  g++ -m32 -std=c++20 -O2 -DNDEBUG \
    "$repo_root/tools/exact-physics-prototype/public_rebuild_runner.cpp" \
    -L"$exact_directory" -Wl,-rpath,"$exact_directory" \
    -Wl,-z,notext -l"$exact_stem" -o "$rebuild_output"
  echo "$rebuild_output"
fi
