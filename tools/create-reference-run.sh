#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="${repo_root}/reference/game/irisu-v2.03-en"
run_name="${1:-run-$(date -u +%Y%m%dT%H%M%SZ)}"

if [[ ! "${run_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Run name may contain only letters, numbers, dot, underscore, and hyphen." >&2
  exit 2
fi

runs_dir="${repo_root}/reference/runs"
run_dir="${runs_dir}/${run_name}"

if [[ ! -f "${source_dir}/irisu.exe" ]]; then
  echo "Missing reference source tree: ${source_dir}" >&2
  exit 1
fi

source_symlink=$(find "${source_dir}" -type l -print -quit)
if [[ -n ${source_symlink} ]]; then
  echo "Refusing source tree containing a symlink: ${source_symlink}" >&2
  exit 1
fi

if [[ -e "${run_dir}" || -L "${run_dir}" ]]; then
  echo "Run directory already exists: ${run_dir}" >&2
  exit 1
fi

mkdir -p -- "${runs_dir}"
stage_root=$(mktemp -d "${runs_dir}/.${run_name}.XXXXXX")
staged_run="${stage_root}/run"
cleanup() {
  if [[ -e ${stage_root} ]]; then
    find "${stage_root}" -depth -delete
  fi
}
trap cleanup EXIT INT TERM

mkdir -p -- "${staged_run}"
cp -a -- "${source_dir}/." "${staged_run}/"
staged_symlink=$(find "${staged_run}" -type l -print -quit)
if [[ -n ${staged_symlink} ]]; then
  echo "Refusing copied run containing a symlink: ${staged_symlink}" >&2
  exit 1
fi

# The preserved tree contains a historical convenience launcher that targets
# /home/gabe/Games, not the copied tree.  Never carry that wrong-target hazard
# into a disposable experiment.
rm -f -- "${staged_run}/launch-irisu.sh"
{
  printf 'created_by=tools/create-reference-run.sh\n'
  printf 'created_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'irisu_exe_sha256=%s\n' \
    "$(sha256sum "${staged_run}/irisu.exe" | awk '{print $1}')"
} >"${staged_run}/.irisu-reference-run"

if ! mv --no-copy --update=none-fail -T -- "${staged_run}" "${run_dir}"; then
  echo "Run directory appeared during preparation: ${run_dir}" >&2
  exit 1
fi
trap - EXIT INT TERM
find "${stage_root}" -depth -delete
printf '%s\n' "${run_dir}"
