#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
port=${1:-8765}

if [[ ! -f "$script_dir/runtime/v86.wasm" ]]; then
  echo "runtime is missing; run ./prepare.sh first" >&2
  exit 1
fi

echo "http://127.0.0.1:$port/"
exec python3 -m http.server "$port" --bind 127.0.0.1 --directory "$script_dir"

