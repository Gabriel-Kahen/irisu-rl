#!/usr/bin/env bash
set -euo pipefail

workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
test_dir=$(mktemp -d)
cleanup() { rm -rf -- "$test_dir"; }
trap cleanup EXIT

"$workspace/tools/run-box2d-probe.sh" --output "$test_dir/first.jsonl"
"$workspace/tools/run-box2d-probe.sh" --no-build --output "$test_dir/second.jsonl"
cmp -- "$test_dir/first.jsonl" "$test_dir/second.jsonl"
cmp -- "$workspace/reference/probe/golden/box2d-v2.03-wine11.13.jsonl" \
  "$test_dir/first.jsonl"
python3 "$workspace/reference/probe/validate_trace.py" "$test_dir/first.jsonl"
echo "probe traces are byte-for-byte repeatable and match the golden trace"
