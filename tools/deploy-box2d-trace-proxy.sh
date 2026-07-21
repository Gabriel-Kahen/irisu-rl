#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
runs_root=$(realpath -e -- "${repo_root}/reference/runs")
proxy_dir="${repo_root}/reference/trace-proxy"
expected_dll_sha=34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd
expected_exe_sha=0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255

usage() {
  echo "usage: $0 REFERENCE_RUN_DIRECTORY" >&2
  echo "The target must be one explicit, direct reference/runs child." >&2
}

if (($# != 1)); then
  usage
  exit 2
fi

requested=${1%/}
if [[ ! -d ${requested} ]]; then
  echo "Refusing missing run directory: ${requested}" >&2
  exit 3
fi
if [[ -L ${requested} ]]; then
  echo "Refusing symlinked run directory: ${requested}" >&2
  exit 3
fi

run_dir=$(realpath -e -- "${requested}")
if [[ $(dirname -- "${run_dir}") != "${runs_root}" ]]; then
  echo "Refusing non-disposable target: ${run_dir}" >&2
  echo "Expected one direct child of ${runs_root}" >&2
  exit 4
fi
if [[ ! $(basename -- "${run_dir}") =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Refusing invalid run name: $(basename -- "${run_dir}")" >&2
  exit 4
fi

run_symlink=$(find "${run_dir}" -type l -print -quit)
if [[ -n ${run_symlink} ]]; then
  echo "Refusing run containing a symlink: ${run_symlink}" >&2
  exit 5
fi

# A copied launch-irisu.sh from the preserved tree targets the user's live game
# rather than this run.  create-reference-run.sh now strips it; reject older
# copies rather than letting a valid proxy experiment launch the wrong tree.
if [[ -e ${run_dir}/launch-irisu.sh ]]; then
  echo "Refusing run with an untrusted launch-irisu.sh: ${run_dir}" >&2
  echo "Create a fresh run with tools/create-reference-run.sh and launch it through tools/launch-reference-game.sh." >&2
  exit 5
fi

for process_cwd in /proc/[0-9]*/cwd; do
  current_cwd=$(readlink -- "${process_cwd}" 2>/dev/null || true)
  if [[ ${current_cwd} == "${run_dir}" || ${current_cwd} == "${run_dir}/"* ]]; then
    echo "Refusing a run currently used as a process working directory: ${run_dir}" >&2
    exit 5
  fi
done

exe=${run_dir}/irisu.exe
dll_dir=${run_dir}/data/dll
dll=${dll_dir}/Box2D.dll
real_dll=${dll_dir}/Box2D.real.dll
trace=${dll_dir}/box2d-trace.jsonl
marker=${run_dir}/.box2d-trace-proxy
lock_dir=${run_dir}/.box2d-trace-proxy.lock

if ! mkdir -- "${lock_dir}"; then
  echo "Refusing run with an active/stale deployment lock: ${lock_dir}" >&2
  exit 6
fi

staged=
marker_staged=
authentic_moved=0
proxy_installed=0
marker_installed=0
committed=0
lock_owned=1

move_no_replace() {
  mv --no-copy --update=none-fail -T -- "$1" "$2"
}

rollback() {
  local quarantined
  if ((!committed)); then
    if ((marker_installed)); then
      quarantined="${run_dir}/.box2d-trace-proxy.rollback.$$"
      if move_no_replace "${marker}" "${quarantined}" 2>/dev/null; then
        if [[ -f ${quarantined} && ! -L ${quarantined} &&
              $(sha256sum "${quarantined}" | awk '{print $1}') == "${marker_sha:-}" ]]; then
          rm -f -- "${quarantined}"
          marker_installed=0
        else
          move_no_replace "${quarantined}" "${marker}" 2>/dev/null || true
        fi
      fi
    fi
    if ((proxy_installed)); then
      quarantined="${dll_dir}/.Box2D.trace-proxy.rollback.$$"
      if move_no_replace "${dll}" "${quarantined}" 2>/dev/null; then
        if [[ -f ${quarantined} && ! -L ${quarantined} &&
              $(sha256sum "${quarantined}" | awk '{print $1}') == "${proxy_sha:-}" ]]; then
          rm -f -- "${quarantined}"
          proxy_installed=0
        else
          move_no_replace "${quarantined}" "${dll}" 2>/dev/null || true
        fi
      fi
    fi
    if ((authentic_moved)) && [[ ! -e ${dll} && ! -L ${dll} && -e ${real_dll} ]]; then
      if move_no_replace "${real_dll}" "${dll}" 2>/dev/null; then
        authentic_moved=0
      fi
    fi
  fi
  [[ -n ${staged:-} && -e ${staged} ]] && rm -f -- "${staged}"
  [[ -n ${marker_staged:-} && -e ${marker_staged} ]] && rm -f -- "${marker_staged}"
  if ((lock_owned)); then
    rmdir -- "${lock_dir}" 2>/dev/null || true
    lock_owned=0
  fi
}
trap rollback EXIT INT TERM

for required in "${exe}" "${dll}"; do
  if [[ ! -f ${required} || -L ${required} ]]; then
    echo "Refusing missing or symlinked authentic file: ${required}" >&2
    exit 6
  fi
done
if [[ -e ${real_dll} || -e ${marker} ]]; then
  echo "Refusing an already modified/deployed run: ${run_dir}" >&2
  exit 6
fi
if [[ -e ${trace} ]]; then
  echo "Refusing to overwrite an existing trace: ${trace}" >&2
  exit 6
fi

actual_exe_sha=$(sha256sum "${exe}" | awk '{print $1}')
actual_dll_sha=$(sha256sum "${dll}" | awk '{print $1}')
if [[ ${actual_exe_sha} != "${expected_exe_sha}" ]]; then
  echo "Refusing non-target irisu.exe: expected ${expected_exe_sha}, got ${actual_exe_sha}" >&2
  exit 7
fi
if [[ ${actual_dll_sha} != "${expected_dll_sha}" ]]; then
  echo "Refusing non-authentic Box2D.dll: expected ${expected_dll_sha}, got ${actual_dll_sha}" >&2
  exit 7
fi

"${proxy_dir}/build.sh" >/dev/null
proxy=${proxy_dir}/build/Box2D.dll
python3 "${proxy_dir}/validate_build.py" "${proxy}" >/dev/null
proxy_sha=$(sha256sum "${proxy}" | awk '{print $1}')
proxy_source_sha=$(sha256sum "${proxy_dir}/src/box2d_trace_proxy.c" | awk '{print $1}')
proxy_build_sha=$(sha256sum "${proxy_dir}/build.sh" | awk '{print $1}')
clang_identity=$(clang --version)
lld_identity=$(lld-link --version)
resolved_dlltool=$(realpath -e -- "$(command -v llvm-dlltool)")
dlltool_identity="${resolved_dlltool};sha256=$(sha256sum "${resolved_dlltool}" | awk '{print $1}')"
clang_identity=${clang_identity%%$'\n'*}
lld_identity=${lld_identity%%$'\n'*}

staged=$(mktemp "${dll_dir}/.Box2D.trace-proxy.XXXXXX")
marker_staged=$(mktemp "${run_dir}/.box2d-trace-proxy.XXXXXX")

install -m 0644 -- "${proxy}" "${staged}"
if [[ $(sha256sum "${staged}" | awk '{print $1}') != "${proxy_sha}" ]]; then
  echo "Staged proxy hash mismatch; refusing deployment." >&2
  exit 8
fi

{
  printf 'schema=1\n'
  printf 'deployed_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'run_dir=%s\n' "${run_dir}"
  printf 'irisu_exe_sha256=%s\n' "${actual_exe_sha}"
  printf 'authentic_box2d_sha256=%s\n' "${actual_dll_sha}"
  printf 'trace_proxy_sha256=%s\n' "${proxy_sha}"
  printf 'trace_proxy_source_sha256=%s\n' "${proxy_source_sha}"
  printf 'trace_proxy_build_script_sha256=%s\n' "${proxy_build_sha}"
  printf 'clang_identity=%s\n' "${clang_identity}"
  printf 'lld_link_identity=%s\n' "${lld_identity}"
  printf 'llvm_dlltool_identity=%s\n' "${dlltool_identity}"
  printf 'validated_export_abi=16 decorated stdcall exports; KERNEL32.dll-only static imports\n'
  printf 'trace_path=data/dll/box2d-trace.jsonl\n'
} >"${marker_staged}"
marker_sha=$(sha256sum "${marker_staged}" | awk '{print $1}')

run_symlink=$(find "${run_dir}" -type l -print -quit)
if [[ -n ${run_symlink} ]]; then
  echo "Refusing run that gained a symlink during deployment: ${run_symlink}" >&2
  exit 8
fi

authentic_moved=1
if ! move_no_replace "${dll}" "${real_dll}"; then
  authentic_moved=0
  echo "Authentic DLL destination appeared during deployment; refusing." >&2
  exit 8
fi
proxy_installed=1
if ! move_no_replace "${staged}" "${dll}"; then
  echo "Active DLL destination appeared during deployment; refusing." >&2
  exit 8
fi
staged=
marker_installed=1
if ! move_no_replace "${marker_staged}" "${marker}"; then
  echo "Deployment marker appeared during deployment; refusing." >&2
  exit 8
fi
marker_staged=
committed=1
rmdir -- "${lock_dir}"
lock_owned=0
trap - EXIT INT TERM

echo "Deployed trace proxy to disposable run: ${run_dir}"
echo "Authentic DLL retained at: ${real_dll}"
echo "Trace will be written at: ${trace}"
printf 'Launch only with:\n  IRISU_GAME_DIR=%q %q\n' \
  "${run_dir}" "${repo_root}/tools/launch-reference-game.sh"
