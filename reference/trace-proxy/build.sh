#!/usr/bin/env bash
set -euo pipefail

proxy_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=${1:-"${proxy_dir}/build"}

mkdir -p -- "${build_dir}"
llvm-dlltool -m i386 -k -d "${proxy_dir}/kernel32.def" \
  -l "${build_dir}/kernel32.lib"
clang --target=i686-pc-windows-msvc -std=c11 -O2 -Wall -Wextra -Werror \
  -ffreestanding -fno-builtin -fno-stack-protector \
  -c "${proxy_dir}/src/box2d_trace_proxy.c" \
  -o "${build_dir}/box2d_trace_proxy.obj"
lld-link /machine:x86 /dll /entry:DllMainCRTStartup /nodefaultlib /timestamp:0 \
  /out:"${build_dir}/Box2D.dll" \
  "${build_dir}/box2d_trace_proxy.obj" "${build_dir}/kernel32.lib"

python3 "${proxy_dir}/validate_build.py" "${build_dir}/Box2D.dll"
