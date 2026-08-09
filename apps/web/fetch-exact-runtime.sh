#!/usr/bin/env bash
set -euo pipefail

root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
output=${1:?usage: fetch-exact-runtime.sh OUTPUT_DIR}
output=$(realpath -m -- "$output")
if [[ "$output" == / || "$output" == "$root" ]]; then
  echo "refusing unsafe exact-runtime download path" >&2
  exit 1
fi

tag=web-exact-runtime-20260809
asset=irisu-exact-runtime-20260809.tar.gz
archive_sha=9937a50853fd5f29ddf853eb6e6e3ec6e55251b496f7306ed4855f8648937928
url="https://github.com/Gabriel-Kahen/irisu-rl/releases/download/$tag/$asset"
archive=$(mktemp "${output}.archive.XXXXXX")
stage=$(mktemp -d "${output}.stage.XXXXXX")
cleanup() { rm -rf -- "$archive" "$stage"; }
trap cleanup EXIT

curl -fL --retry 3 --output "$archive" "$url"
printf '%s  %s\n' "$archive_sha" "$archive" | sha256sum -c -
tar -xzf "$archive" -C "$stage"
runtime="$stage/exact-runtime"
[[ -f "$runtime/.irisu-exact-runtime" && -f "$runtime/SHA256SUMS" ]] || {
  echo "downloaded exact runtime is malformed" >&2
  exit 1
}
(cd "$runtime" && sha256sum -c SHA256SUMS)

if [[ -e "$output" ]]; then
  [[ -f "$output/.irisu-exact-runtime" ]] || {
    echo "refusing to replace unmarked exact-runtime path: $output" >&2
    exit 1
  }
  cmake -E remove_directory "$output"
fi
mkdir -p "$(dirname -- "$output")"
mv "$runtime" "$output"
trap - EXIT
rm -f -- "$archive"
rmdir "$stage"
echo "downloaded pinned exact browser runtime to $output"
