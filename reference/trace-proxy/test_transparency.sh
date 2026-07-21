#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
proxy_dir="${repo_root}/reference/trace-proxy"
probe_dir="${repo_root}/reference/probe"
authentic_dll=${IRISU_BOX2D_DLL:-"${repo_root}/reference/game/irisu-v2.03-en/data/dll/Box2D.dll"}
wine_bin=${IRISU_WINE_BIN:-/home/gabe/.local/share/irisu-syndrome/runtime/bin/wine}
wine_prefix=${IRISU_WINEPREFIX:-/home/gabe/.local/share/irisu-syndrome/prefix}
expected_dll_sha=34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd

[[ -f ${authentic_dll} ]] || { echo "missing authentic DLL: ${authentic_dll}" >&2; exit 2; }
[[ -x ${wine_bin} ]] || { echo "missing Wine runtime: ${wine_bin}" >&2; exit 2; }
actual_dll_sha=$(sha256sum "${authentic_dll}" | awk '{print $1}')
[[ ${actual_dll_sha} == "${expected_dll_sha}" ]] || {
  echo "refusing non-authentic DLL: ${actual_dll_sha}" >&2
  exit 3
}

"${proxy_dir}/build.sh" >/dev/null
"${probe_dir}/build.sh" >/dev/null

scratch=$(mktemp -d)
cleanup() { find "${scratch}" -depth -delete; }
trap cleanup EXIT
cp -- "${proxy_dir}/build/Box2D.dll" "${scratch}/Box2D.dll"
cp -- "${authentic_dll}" "${scratch}/Box2D.real.dll"
cp -- "${probe_dir}/build/box2d-probe.exe" "${scratch}/box2d-probe.exe"

set +e
(
  cd -- "${scratch}"
  WINEDEBUG=${WINEDEBUG:--all} WINEPREFIX="${wine_prefix}" \
    "${wine_bin}" ./box2d-probe.exe >wine.stdout 2>wine.stderr
)
status=$?
set -e
if ((status != 0)); then
  echo "proxy transparency probe failed with status ${status}" >&2
  sed -n '1,120p' "${scratch}/wine.stdout" >&2
  sed -n '1,120p' "${scratch}/wine.stderr" >&2
  exit "${status}"
fi

python3 "${probe_dir}/validate_trace.py" "${scratch}/box2d-probe.jsonl" >/dev/null
cmp -- "${probe_dir}/golden/box2d-v2.03-wine11.13.jsonl" \
  "${scratch}/box2d-probe.jsonl"
python3 "${proxy_dir}/validate_trace.py" "${scratch}/box2d-trace.jsonl"
echo "proxy is byte-transparent to the shipped-DLL golden probe"

