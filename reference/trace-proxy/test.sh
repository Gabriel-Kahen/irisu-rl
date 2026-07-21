#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
proxy_dir="${repo_root}/reference/trace-proxy"
source_game="${repo_root}/reference/game/irisu-v2.03-en"
runs_root="${repo_root}/reference/runs"
expected_dll_sha=34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd

scratch=$(mktemp -d)
mkdir -p -- "${runs_root}"
run_dir=
unsafe_run=
symlink_run=
cwd_run=
cwd_pid=
cleanup() {
  if [[ -n ${cwd_pid} ]]; then
    kill "${cwd_pid}" 2>/dev/null || true
    wait "${cwd_pid}" 2>/dev/null || true
  fi
  [[ -n ${run_dir} && -e ${run_dir} ]] && find "${run_dir}" -depth -delete
  [[ -n ${unsafe_run} && -e ${unsafe_run} ]] && find "${unsafe_run}" -depth -delete
  [[ -n ${symlink_run} && -e ${symlink_run} ]] && find "${symlink_run}" -depth -delete
  [[ -n ${cwd_run} && -e ${cwd_run} ]] && find "${cwd_run}" -depth -delete
  find "${scratch}" -depth -delete
}
trap cleanup EXIT

"${proxy_dir}/build.sh" "${scratch}/build" >/dev/null
python3 "${proxy_dir}/validate_build.py" "${scratch}/build/Box2D.dll" >/dev/null

printf '%s\n' \
  '{"seq":0,"type":"proxy_loaded","schema":1,"real_loaded":true,"export_mask":"0000ffff","x87_cw":"027f","ok":true}' \
  '{"seq":1,"type":"init","world":1,"args_f32":["00000000","00000000","00000000","00000000","00000000","00000000"],"result":1,"x87_cw_before":"027f","x87_cw_after":"027f"}' \
  '{"seq":2,"type":"create","world":1,"step":0,"shape":"box","ordinal":1,"body":1,"args_f32":["00000000","00000000","00000000","00000000","00000000","00000000","00000000","00000000"]}' \
  '{"seq":3,"type":"step","world":1,"step":1,"dt_f32":"3ca3d70a","iterations":10}' \
  '{"seq":4,"type":"contact","world":1,"step":1,"call":1,"result":false,"a_user":0,"b_user":0,"a_ordinal":0,"b_ordinal":0}' \
  '{"seq":5,"type":"dispose","world":1,"step":1}' \
  >"${scratch}/valid-trace.jsonl"
python3 "${proxy_dir}/validate_trace.py" "${scratch}/valid-trace.jsonl" >/dev/null

printf '%s\n' \
  '{"seq":0,"type":"proxy_loaded","schema":1,"real_loaded":true,"export_mask":"0000ffff","x87_cw":"027f","ok":true}' \
  '{"seq":1,"type":"init","world":1,"args_f32":["00000000","00000000","00000000","00000000","00000000","00000000"],"result":1,"x87_cw_before":"027f","x87_cw_after":"027f"}' \
  '{"seq":2,"type":"mapping_overflow","world":1,"capacity":4096}' \
  >"${scratch}/overflow-trace.jsonl"
if python3 "${proxy_dir}/validate_trace.py" "${scratch}/overflow-trace.jsonl" \
    >"${scratch}/overflow.stdout" 2>"${scratch}/overflow.stderr"; then
  echo "trace validator accepted a mapping overflow" >&2
  exit 1
fi

cwd_run=$(mktemp -d "${runs_root}/trace-proxy-cwd.XXXXXX")
mkdir -p -- "${cwd_run}/data/dll"
(
  cd -- "${cwd_run}/data/dll"
  exec sleep 30
) &
cwd_pid=$!
for _ in {1..100}; do
  [[ $(readlink -- "/proc/${cwd_pid}/cwd" 2>/dev/null || true) == \
      "${cwd_run}/data/dll" ]] && break
  sleep 0.01
done
[[ $(readlink -- "/proc/${cwd_pid}/cwd") == "${cwd_run}/data/dll" ]]
if "${repo_root}/tools/deploy-box2d-trace-proxy.sh" "${cwd_run}" \
    >"${scratch}/cwd.stdout" 2>"${scratch}/cwd.stderr"; then
  echo "deployment accepted a run with an active child-directory process" >&2
  exit 1
fi
grep -q 'process working directory' "${scratch}/cwd.stderr"
kill "${cwd_pid}"
wait "${cwd_pid}" 2>/dev/null || true
cwd_pid=

symlink_run=$(mktemp -d "${runs_root}/trace-proxy-symlink.XXXXXX")
mkdir -p -- "${symlink_run}/outside"
ln -s -- "${symlink_run}/outside" "${symlink_run}/data"
if "${repo_root}/tools/deploy-box2d-trace-proxy.sh" "${symlink_run}" \
    >"${scratch}/symlink.stdout" 2>"${scratch}/symlink.stderr"; then
  echo "deployment accepted a symlinked nested parent" >&2
  exit 1
fi

printf '%s\n' \
  '{"seq":0,"type":"proxy_loaded","schema":1,"real_loaded":true,"export_mask":"0000ffff","x87_cw":"027f","ok":true}' \
  '{"seq":1,"type":"init","world":1,"args_f32":["00000000","00000000","00000000","00000000","00000000","00000000"],"result":1,"x87_cw_before":"027f","x87_cw_after":"027f"}' \
  '{"seq":2,"type":"step","world":1,"step":1,"dt_f32":"3ca3d70a","iterations":10}' \
  '{"seq":3,"type":"contact","world":1,"step":1,"call":1,"result":true,"a_user":1,"b_user":2,"a_ordinal":1,"b_ordinal":2}' \
  >"${scratch}/unterminated-trace.jsonl"
if python3 "${proxy_dir}/validate_trace.py" "${scratch}/unterminated-trace.jsonl" \
    >"${scratch}/unterminated.stdout" 2>"${scratch}/unterminated.stderr"; then
  echo "trace validator accepted an unterminated contact cursor" >&2
  exit 1
fi

if "${repo_root}/tools/deploy-box2d-trace-proxy.sh" "${source_game}" \
    >"${scratch}/preserved.stdout" 2>"${scratch}/preserved.stderr"; then
  echo "deployment accepted the preserved source tree" >&2
  exit 1
fi

unsafe_run=$(mktemp -d "${runs_root}/trace-proxy-unsafe.XXXXXX")
touch "${unsafe_run}/launch-irisu.sh"
if "${repo_root}/tools/deploy-box2d-trace-proxy.sh" "${unsafe_run}" \
    >"${scratch}/unsafe.stdout" 2>"${scratch}/unsafe.stderr"; then
  echo "deployment accepted an untrusted copied launcher" >&2
  exit 1
fi

if [[ -f ${source_game}/irisu.exe && -f ${source_game}/data/dll/Box2D.dll ]]; then
  run_dir=$(mktemp -d "${runs_root}/trace-proxy-test.XXXXXX")
  mkdir -p -- "${run_dir}/data/dll"
  cp -- "${source_game}/irisu.exe" "${run_dir}/irisu.exe"
  cp -- "${source_game}/data/dll/Box2D.dll" "${run_dir}/data/dll/Box2D.dll"

  "${repo_root}/tools/deploy-box2d-trace-proxy.sh" "${run_dir}" \
    >"${scratch}/deploy.stdout"
  [[ $(sha256sum "${run_dir}/data/dll/Box2D.real.dll" | awk '{print $1}') == \
      "${expected_dll_sha}" ]]
  cmp -- "${proxy_dir}/build/Box2D.dll" "${run_dir}/data/dll/Box2D.dll"
  [[ -s ${run_dir}/.box2d-trace-proxy ]]
  grep -q '^trace_proxy_source_sha256=' "${run_dir}/.box2d-trace-proxy"
  grep -q '^trace_proxy_build_script_sha256=' "${run_dir}/.box2d-trace-proxy"
  grep -q '^clang_identity=' "${run_dir}/.box2d-trace-proxy"
  grep -q '^validated_export_abi=' "${run_dir}/.box2d-trace-proxy"

  if "${repo_root}/tools/deploy-box2d-trace-proxy.sh" "${run_dir}" \
      >"${scratch}/redeploy.stdout" 2>"${scratch}/redeploy.stderr"; then
    echo "deployment accepted an already modified run" >&2
    exit 1
  fi
else
  echo "local original binaries absent; skipped successful-deployment fixture" >&2
fi

echo "trace proxy build and deployment guards passed"
