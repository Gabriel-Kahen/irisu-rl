#!/usr/bin/env bash
set -euo pipefail

codex_root="${CODEX_HOME:-${HOME}/.codex}"
cache_root="${codex_root}/plugins/cache/codex-computer-use-linux/same-session-computer-use"
plugin_src="${SAME_SESSION_PLUGIN_SRC:-}"

if [[ -z "${plugin_src}" ]]; then
  plugin_src="$(find "${cache_root}" -mindepth 2 -maxdepth 2 -type d -name src -print 2>/dev/null | sort -V | tail -n 1)"
fi

if [[ -z "${plugin_src}" || ! -d "${plugin_src}/same_session_computer_use" ]]; then
  echo "Could not locate the installed same-session-computer-use Python package." >&2
  exit 1
fi

PYTHONPATH="${plugin_src}" python3 -c \
  'from same_session_computer_use.native_plugin import ensure_target_pointer_plugin; ensure_target_pointer_plugin()'

hyprctl plugin list | grep -F 'Plugin same-session-target-pointer'
hyprctl -j cutargetstatus
