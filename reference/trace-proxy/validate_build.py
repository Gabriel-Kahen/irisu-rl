#!/usr/bin/env python3
"""Validate that the trace proxy is a 32-bit DLL with the exact wrapper ABI."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


EXPECTED_EXPORTS = [
    (1, "_b2d_create_box@32"),
    (2, "_b2d_create_circle@24"),
    (3, "_b2d_create_triangle@32"),
    (4, "_b2d_destroy_body@4"),
    (5, "_b2d_dispose@0"),
    (6, "_b2d_get_contact@8"),
    (7, "_b2d_get_r@4"),
    (8, "_b2d_get_v@12"),
    (9, "_b2d_get_x@4"),
    (10, "_b2d_get_y@4"),
    (11, "_b2d_init@24"),
    (12, "_b2d_set_position@16"),
    (13, "_b2d_set_user_data@8"),
    (14, "_b2d_set_v@12"),
    (15, "_b2d_step@8"),
    (16, "_b2d_test@4"),
]


def inspect(path: Path) -> str:
    command = [
        "llvm-readobj",
        "--file-headers",
        "--coff-exports",
        "--coff-imports",
        str(path),
    ]
    return subprocess.run(command, check=True, text=True, capture_output=True).stdout


def validate(path: Path) -> None:
    output = inspect(path)
    if "Format: COFF-i386" not in output or "Arch: i386" not in output:
        raise ValueError("proxy is not a PE32/i386 image")
    if "Characteristics [" not in output or "IMAGE_FILE_DLL" not in output:
        raise ValueError("proxy is not marked as a DLL")

    exports = [
        (int(ordinal), name)
        for ordinal, name in re.findall(
            r"Export \{\s+Ordinal: (\d+)\s+Name: (\S+)", output
        )
    ]
    if exports != EXPECTED_EXPORTS:
        raise ValueError(f"wrong export ABI: {exports!r}")

    import_dlls = re.findall(r"Import \{\s+Name: (\S+)", output)
    if import_dlls != ["KERNEL32.dll"]:
        raise ValueError(f"unexpected static imports: {import_dlls!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dll", type=Path)
    args = parser.parse_args()
    if not args.dll.is_file():
        parser.error(f"missing DLL: {args.dll}")
    try:
        validate(args.dll)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(f"validated trace proxy ABI: {args.dll}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

