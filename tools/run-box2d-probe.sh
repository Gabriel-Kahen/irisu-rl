#!/usr/bin/env bash
set -euo pipefail

workspace=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
probe_dir="$workspace/reference/probe"
dll=${IRISU_BOX2D_DLL:-"$workspace/reference/game/irisu-v2.03-en/data/dll/Box2D.dll"}
wine_bin=${IRISU_WINE_BIN:-/home/gabe/.local/share/irisu-syndrome/runtime/bin/wine}
wine_prefix=${IRISU_WINEPREFIX:-/home/gabe/.local/share/irisu-syndrome/prefix}
expected_dll_sha=34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd
output=-
build=1

usage() {
  echo "usage: $0 [--output PATH|-] [--no-build]" >&2
}

while (($#)); do
  case $1 in
    --output)
      (($# >= 2)) || { usage; exit 2; }
      output=$2
      shift 2
      ;;
    --no-build)
      build=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

[[ -f $dll ]] || { echo "missing shipped DLL: $dll" >&2; exit 3; }
[[ -x $wine_bin ]] || { echo "missing Wine executable: $wine_bin" >&2; exit 3; }
actual_dll_sha=$(sha256sum "$dll" | awk '{print $1}')
[[ $actual_dll_sha == "$expected_dll_sha" ]] || {
  echo "refusing non-target Box2D.dll: expected $expected_dll_sha, got $actual_dll_sha" >&2
  exit 4
}

if ((build)); then
  "$probe_dir/build.sh" >/dev/null
fi
probe_exe="$probe_dir/build/box2d-probe.exe"
[[ -f $probe_exe ]] || { echo "missing probe executable: $probe_exe" >&2; exit 5; }

run_dir=$(mktemp -d)
cleanup() { rm -rf -- "$run_dir"; }
trap cleanup EXIT
cp -- "$probe_exe" "$run_dir/box2d-probe.exe"
cp -- "$dll" "$run_dir/Box2D.dll"

set +e
(
  cd -- "$run_dir"
  WINEDEBUG=${WINEDEBUG:--all} WINEPREFIX="$wine_prefix" \
    "$wine_bin" ./box2d-probe.exe >wine.stdout 2>wine.stderr
)
status=$?
set -e
if ((status != 0)); then
  echo "probe failed with status $status" >&2
  sed -n '1,120p' "$run_dir/wine.stdout" >&2
  sed -n '1,120p' "$run_dir/wine.stderr" >&2
  exit "$status"
fi

trace="$run_dir/box2d-probe.jsonl"
[[ -s $trace ]] || { echo "probe produced no trace" >&2; exit 6; }
python3 "$probe_dir/validate_trace.py" "$trace" >&2

if [[ $output == - ]]; then
  cat -- "$trace"
else
  mkdir -p -- "$(dirname -- "$output")"
  cp -- "$trace" "$output"
  echo "wrote $output" >&2
fi

