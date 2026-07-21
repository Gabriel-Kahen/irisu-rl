#!/usr/bin/env bash
set -euo pipefail

probe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=${1:-"$probe_dir/build"}

mkdir -p "$build_dir"
llvm-dlltool -m i386 -k -d "$probe_dir/kernel32.def" \
  -l "$build_dir/kernel32.lib"
clang --target=i686-pc-windows-msvc -O2 -ffreestanding -fno-builtin \
  -fno-stack-protector -c "$probe_dir/src/box2d_probe.c" \
  -o "$build_dir/box2d_probe.obj"
lld-link /machine:x86 /subsystem:console /entry:mainCRTStartup /nodefaultlib \
  /out:"$build_dir/box2d-probe.exe" "$build_dir/box2d_probe.obj" \
  "$build_dir/kernel32.lib"

file "$build_dir/box2d-probe.exe"
