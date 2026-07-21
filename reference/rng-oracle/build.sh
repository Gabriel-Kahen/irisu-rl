#!/usr/bin/env bash
set -euo pipefail

oracle_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace=$(cd -- "$oracle_dir/../.." && pwd)
build_dir=${1:-"$oracle_dir/build"}

mkdir -p "$build_dir"
llvm-dlltool -m i386 -k -d "$workspace/reference/probe/kernel32.def" \
  -l "$build_dir/kernel32.lib"
clang --target=i686-pc-windows-msvc -O2 -ffreestanding -fno-builtin \
  -fno-stack-protector -c "$oracle_dir/src/dxlib_rng_oracle.c" \
  -o "$build_dir/dxlib_rng_oracle.obj"
lld-link /machine:x86 /subsystem:console /entry:mainCRTStartup /nodefaultlib \
  /out:"$build_dir/dxlib-rng-oracle.exe" \
  "$build_dir/dxlib_rng_oracle.obj" "$build_dir/kernel32.lib"
