#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
game_dir="${IRISU_GAME_DIR:-${repo_root}/reference/game/irisu-v2.03-en}"
wine_bin="${IRISU_WINE_BIN:-/home/gabe/.local/share/irisu-syndrome/runtime/bin/wine}"
wine_prefix="${IRISU_WINEPREFIX:-/home/gabe/.local/share/irisu-syndrome/prefix}"

if [[ ! -f "${game_dir}/irisu.exe" ]]; then
  echo "Missing reference game: ${game_dir}/irisu.exe" >&2
  exit 1
fi

if [[ ! -x "${wine_bin}" ]]; then
  echo "Missing Wine runtime: ${wine_bin}" >&2
  exit 1
fi

export WINEPREFIX="${wine_prefix}"
export WINEDEBUG="-all"
export WINEDLLOVERRIDES="mscoree,mshtml="

cd -- "${game_dir}"
exec "${wine_bin}" irisu.exe
