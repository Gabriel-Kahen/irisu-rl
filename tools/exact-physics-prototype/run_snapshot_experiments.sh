#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 /path/to/libirisu_box2d_msvc_exact.so replay.rpy" >&2
  exit 2
fi

prototype_dir=$(cd "$(dirname "$0")" && pwd)
build_dir=$(mktemp -d)
trap 'rm -rf -- "$build_dir"' EXIT

"$prototype_dir/build_branch_snapshot.sh" "$1" \
  "$build_dir/branch-snapshot" "$build_dir/public-rebuild" >/dev/null

for checkpoint in "100 500" "8000 4000" "30000 4000" "45000 2000"; do
  # Intentional word splitting: each fixed tuple supplies prefix and future.
  "$build_dir/branch-snapshot" "$2" $checkpoint
done
"$build_dir/public-rebuild"
