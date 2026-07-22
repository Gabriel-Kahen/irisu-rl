#!/usr/bin/env python3
"""Turn an exact MSVC9 Box2D r58 COFF build into a native ELF32 library.

The checked-in repository contains no Microsoft binaries or compiler output.
This tool either consumes an existing object directory or invokes a user-supplied
MSVC9 compiler under Wine, then performs the clean-room COFF-to-ELF transform.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reference/native-box2d"
BRIDGE_SOURCE = ASSETS / "msvc-bridge.c"
RUNTIME_SOURCE = ASSETS / "msvc-runtime.S"
RUNTIME_DEPENDENCIES: tuple[Path, ...] = ()
EXPECTED_COMPILER_VERSION = "15.00.21022.08"
OUTPUT_LIBRARY_NAME = "libirisu_box2d_msvc_exact.so"
OUTPUT_LIBRARY_SONAME = OUTPUT_LIBRARY_NAME
COMPILE_FLAGS = (
    "/nologo",
    "/c",
    "/O2",
    "/fp:precise",
    "/MT",
    "/D",
    "NDEBUG",
    "/EHs-c-",
    "/GR-",
    "/GS-",
)
SOURCE_FILES = (
    "Source/Collision/b2BroadPhase.cpp",
    "Source/Collision/b2CollideCircle.cpp",
    "Source/Collision/b2CollidePoly.cpp",
    "Source/Collision/b2Distance.cpp",
    "Source/Collision/b2PairManager.cpp",
    "Source/Collision/b2Shape.cpp",
    "Source/Common/b2BlockAllocator.cpp",
    "Source/Common/b2Settings.cpp",
    "Source/Common/b2StackAllocator.cpp",
    "Source/Dynamics/Contacts/b2CircleContact.cpp",
    "Source/Dynamics/Contacts/b2Conservative.cpp",
    "Source/Dynamics/Contacts/b2Contact.cpp",
    "Source/Dynamics/Contacts/b2ContactSolver.cpp",
    "Source/Dynamics/Contacts/b2PolyAndCircleContact.cpp",
    "Source/Dynamics/Contacts/b2PolyContact.cpp",
    "Source/Dynamics/Joints/b2DistanceJoint.cpp",
    "Source/Dynamics/Joints/b2GearJoint.cpp",
    "Source/Dynamics/Joints/b2Joint.cpp",
    "Source/Dynamics/Joints/b2MouseJoint.cpp",
    "Source/Dynamics/Joints/b2PrismaticJoint.cpp",
    "Source/Dynamics/Joints/b2PulleyJoint.cpp",
    "Source/Dynamics/Joints/b2RevoluteJoint.cpp",
    "Source/Dynamics/b2Body.cpp",
    "Source/Dynamics/b2ContactManager.cpp",
    "Source/Dynamics/b2Island.cpp",
    "Source/Dynamics/b2World.cpp",
    "Source/Dynamics/b2WorldCallbacks.cpp",
)
OBJECT_NAMES = tuple(Path(path).with_suffix(".obj").name for path in SOURCE_FILES) + (
    "box2d-wrapper-msvc.obj",
)
RUNTIME_RENAMES = {
    "??2@YAPAXI@Z": "msvc_operator_new",
    "??3@YAXPAX@Z": "msvc_operator_delete",
    "__CIcos": "msvc_CIcos",
    "__CIsin": "msvc_CIsin",
    "__CIsqrt": "msvc_CIsqrt",
    "__finite": "msvc_finite",
    "__fltused": "msvc_fltused",
    "__purecall": "msvc_purecall",
    "_atexit": "msvc_atexit",
    "_free": "free",
    "_malloc": "malloc",
    "_memcpy": "memcpy",
    "_memmove": "memmove",
    "_memset": "memset",
}
EXPORT_RENAMES = {
    "_b2d_create_box@32": "msvc_b2d_create_box",
    "_b2d_create_circle@24": "msvc_b2d_create_circle",
    "_b2d_create_triangle@32": "msvc_b2d_create_triangle",
    "_b2d_destroy_body@4": "msvc_b2d_destroy_body",
    "_b2d_dispose@0": "msvc_b2d_dispose",
    "_b2d_get_contact@8": "msvc_b2d_get_contact",
    "_b2d_get_r@4": "msvc_b2d_get_r",
    "_b2d_get_v@12": "msvc_b2d_get_v",
    "_b2d_get_x@4": "msvc_b2d_get_x",
    "_b2d_get_y@4": "msvc_b2d_get_y",
    "_b2d_init@24": "msvc_b2d_init",
    "_b2d_set_position@16": "msvc_b2d_set_position",
    "_b2d_set_user_data@8": "msvc_b2d_set_user_data",
    "_b2d_set_v@12": "msvc_b2d_set_v",
    "_b2d_step@8": "msvc_b2d_step",
    "_b2d_test@4": "msvc_b2d_test",
}
PUBLIC_EXPORTS = tuple(name.removeprefix("msvc_") for name in EXPORT_RENAMES.values())
EXPECTED_UNDEFINED = {
    "free",
    "malloc",
    "memcpy",
    "memmove",
    "memset",
    "msvc_CIcos",
    "msvc_CIsin",
    "msvc_CIsqrt",
    "msvc_atexit",
    "msvc_finite",
    "msvc_fltused",
    "msvc_operator_delete",
    "msvc_operator_new",
    "msvc_purecall",
}
COFF_HEADER = struct.Struct("<HHIIIHH")
COFF_SYMBOL = struct.Struct("<8sIhHBB")
ELF32_SECTION = struct.Struct("<10I")
ELF32_REL = struct.Struct("<2I")
SHT_REL = 9
R_386_PC32 = 2


class HostingError(RuntimeError):
    """The input cannot be transformed without weakening the parity claim."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    command: Sequence[str | os.PathLike[str]],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stdout", None)
        suffix = f"\n{detail.rstrip()}" if detail else ""
        raise HostingError(f"command failed: {' '.join(map(str, command))}{suffix}") from exc


def tool_identity(command: str) -> str:
    output = run((command, "--version"), capture=True).stdout or ""
    return output.splitlines()[0] if output else command


def coff_symbol_name(raw: bytes, strings: bytes) -> str:
    if raw[:4] == bytes(4):
        offset = struct.unpack_from("<I", raw, 4)[0]
        if offset < 4 or offset >= len(strings):
            raise HostingError(f"invalid COFF string-table offset {offset}")
        end = strings.find(b"\0", offset)
        if end < 0:
            raise HostingError("unterminated COFF symbol name")
        raw = strings[offset:end]
    else:
        raw = raw.split(b"\0", 1)[0]
    return raw.decode("ascii")


def coff_weak_aliases(path: Path) -> list[tuple[str, str]]:
    data = path.read_bytes()
    if len(data) < COFF_HEADER.size:
        raise HostingError(f"short COFF object: {path}")
    machine, _, _, table_offset, count, _, _ = COFF_HEADER.unpack_from(data)
    if machine != 0x14C:
        raise HostingError(f"expected i386 COFF object, got machine 0x{machine:04x}: {path}")
    table_end = table_offset + count * COFF_SYMBOL.size
    if table_offset < COFF_HEADER.size or table_end + 4 > len(data):
        raise HostingError(f"invalid COFF symbol table: {path}")
    string_size = struct.unpack_from("<I", data, table_end)[0]
    if string_size < 4 or table_end + string_size > len(data):
        raise HostingError(f"invalid COFF string table: {path}")
    strings = data[table_end : table_end + string_size]

    records: dict[int, tuple[str, int, int]] = {}
    index = 0
    while index < count:
        offset = table_offset + index * COFF_SYMBOL.size
        raw_name, _, _, _, storage, aux_count = COFF_SYMBOL.unpack_from(data, offset)
        records[index] = (coff_symbol_name(raw_name, strings), storage, aux_count)
        index += 1 + aux_count

    aliases: list[tuple[str, str]] = []
    for index, (name, storage, aux_count) in records.items():
        if storage != 105:
            continue
        if not aux_count:
            raise HostingError(f"weak COFF symbol has no auxiliary record: {name}")
        aux_offset = table_offset + (index + 1) * COFF_SYMBOL.size
        fallback_index = struct.unpack_from("<I", data, aux_offset)[0]
        fallback = records.get(fallback_index)
        if fallback is None:
            raise HostingError(f"weak COFF symbol has invalid fallback: {name}")
        aliases.append((name, fallback[0]))
    return aliases


def patch_elf_rel32(path: Path) -> int:
    """Translate COFF REL32 addends after GNU objcopy emits ELF R_386_PC32."""

    data = bytearray(path.read_bytes())
    if data[:6] != b"\x7fELF\x01\x01":
        raise HostingError(f"expected little-endian ELF32 object: {path}")
    section_offset = struct.unpack_from("<I", data, 32)[0]
    section_size = struct.unpack_from("<H", data, 46)[0]
    section_count = struct.unpack_from("<H", data, 48)[0]
    if section_size != ELF32_SECTION.size:
        raise HostingError(f"unexpected ELF32 section-header size in {path}")
    sections = [
        ELF32_SECTION.unpack_from(data, section_offset + index * section_size)
        for index in range(section_count)
    ]
    patched = 0
    for section in sections:
        if section[1] != SHT_REL:
            continue
        relocation_offset, relocation_size = section[4], section[5]
        target_index, entry_size = section[7], section[9]
        if target_index >= len(sections) or entry_size not in (0, ELF32_REL.size):
            raise HostingError(f"invalid ELF32 relocation section in {path}")
        target_offset = sections[target_index][4]
        for offset in range(
            relocation_offset,
            relocation_offset + relocation_size,
            entry_size or ELF32_REL.size,
        ):
            target_relative, info = ELF32_REL.unpack_from(data, offset)
            if info & 0xFF != R_386_PC32:
                continue
            location = target_offset + target_relative
            if location + 4 > len(data):
                raise HostingError(f"ELF32 relocation points outside {path}")
            addend = struct.unpack_from("<I", data, location)[0]
            struct.pack_into("<I", data, location, (addend - 4) & 0xFFFFFFFF)
            patched += 1
    path.write_bytes(data)
    return patched


def wine_environment(prefix: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    env["WINEDEBUG"] = "-all"
    if prefix is not None:
        env["WINEPREFIX"] = str(prefix.resolve())
    return env


def windows_path(path: Path, winepath: Path, env: dict[str, str]) -> str:
    output = run((winepath, "-w", path.resolve()), env=env, capture=True).stdout or ""
    value = output.strip().splitlines()[-1] if output.strip() else ""
    if not value:
        raise HostingError(f"winepath produced no path for {path}")
    return value


def compile_coff(args: argparse.Namespace, object_dir: Path) -> tuple[str, dict[str, str]]:
    source = args.source_dir.resolve()
    wrapper = args.wrapper.resolve()
    vc_include = args.vc_include.resolve()
    required = [source / path for path in SOURCE_FILES] + [source / "Include/Box2D.h", wrapper]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise HostingError(f"missing source input(s): {', '.join(missing)}")

    env = wine_environment(args.wine_prefix)
    includes = [source / "Include", source, vc_include]
    win_includes = [windows_path(path, args.winepath, env) for path in includes]
    identity_command = (args.wine, args.cl)
    identity = run(identity_command, env=env, capture=True).stdout or ""
    identity = next(
        (line for line in identity.splitlines() if "Compiler Version" in line),
        identity.splitlines()[0] if identity.splitlines() else str(args.cl),
    )
    if EXPECTED_COMPILER_VERSION not in identity:
        raise HostingError(
            f"expected MSVC9 RTM {EXPECTED_COMPILER_VERSION}, got: {identity}"
        )

    for path in [source / relative for relative in SOURCE_FILES] + [wrapper]:
        output = object_dir / ("box2d-wrapper-msvc.obj" if path == wrapper else path.with_suffix(".obj").name)
        command = [args.wine, args.cl, *COMPILE_FLAGS]
        command.extend(f"/I{include}" for include in win_includes)
        command.extend(
            (
                f"/Fo{windows_path(output, args.winepath, env)}",
                windows_path(path, args.winepath, env),
            )
        )
        run(command, env=env)
    return identity, {str(path.relative_to(source)): sha256(path) for path in required[:-1]}


def undefined_symbols(nm: str, path: Path) -> set[str]:
    output = run((nm, "-u", path), capture=True).stdout or ""
    return {line.split()[-1] for line in output.splitlines() if line.split()}


def defined_dynamic_symbols(nm: str, path: Path) -> set[str]:
    output = run((nm, "-D", "--defined-only", path), capture=True).stdout or ""
    return {line.split()[-1] for line in output.splitlines() if line.split()}


def interposable_msvc_b2d_relocations(readelf: str, path: Path) -> set[str]:
    output = run((readelf, "-rW", path), capture=True).stdout or ""
    result: set[str] = set()
    for line in output.splitlines():
        fields = line.split()
        if (
            len(fields) >= 5
            and fields[2].startswith("R_386_")
            and fields[4].startswith("msvc_b2d_")
        ):
            result.add(fields[4].split("@", 1)[0])
    return result


def ensure_fresh_output(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise HostingError(f"output path already exists: {path}")
    path.mkdir(parents=True)


def build(args: argparse.Namespace) -> dict[str, object]:
    output = args.output_dir.resolve()
    ensure_fresh_output(output)
    elf_dir = output / "elf"
    elf_dir.mkdir()
    compiler_identity: str | None = None
    source_hashes: dict[str, str] | None = None

    if args.object_dir is not None:
        object_dir = args.object_dir.resolve()
    else:
        object_dir = output / "coff"
        object_dir.mkdir()
        compiler_identity, source_hashes = compile_coff(args, object_dir)

    missing = [name for name in OBJECT_NAMES if not (object_dir / name).is_file()]
    if missing:
        raise HostingError(f"missing expected COFF object(s): {', '.join(missing)}")
    objects = [object_dir / name for name in sorted(OBJECT_NAMES)]
    aliases = sorted({alias for path in objects for alias in coff_weak_aliases(path)})
    if not aliases:
        raise HostingError("no COFF weak aliases found; input flags/toolchain are unexpected")

    objcopy_identity = tool_identity(args.objcopy)
    if "GNU objcopy" not in objcopy_identity:
        raise HostingError("COFF-to-ELF conversion requires GNU objcopy")
    renames = dict(aliases)
    renames.update(RUNTIME_RENAMES)
    renames.update(EXPORT_RENAMES)
    rename_args = [item for old, new in renames.items() for item in ("--redefine-sym", f"{old}={new}")]
    rel32_count = 0
    elf_objects: list[Path] = []
    for path in objects:
        converted = elf_dir / path.with_suffix(".o").name
        run((args.objcopy, "-O", "elf32-i386", *rename_args, path, converted))
        rel32_count += patch_elf_rel32(converted)
        elf_objects.append(converted)

    combined = output / "box2d-msvc-combined.o"
    localized = output / "box2d-msvc-combined-local.o"
    run((args.ld, "-m", "elf_i386", "-r", "--allow-multiple-definition", *elf_objects, "-o", combined))
    unresolved = undefined_symbols(args.nm, combined)
    if unresolved != EXPECTED_UNDEFINED:
        raise HostingError(
            "unexpected unresolved symbols after conversion: "
            f"missing={sorted(EXPECTED_UNDEFINED - unresolved)}, "
            f"extra={sorted(unresolved - EXPECTED_UNDEFINED)}"
        )
    run((args.objcopy, "--wildcard", "--localize-symbol", "*@*", combined, localized))

    bridge_object = output / "msvc-bridge.o"
    runtime_object = output / "msvc-runtime.o"
    library = output / OUTPUT_LIBRARY_NAME
    run((args.cc, "-m32", "-O2", "-fPIC", "-fno-stack-protector", "-c", BRIDGE_SOURCE, "-o", bridge_object))
    run((args.cc, "-m32", "-fPIC", "-c", RUNTIME_SOURCE, "-o", runtime_object))
    run(
        (
            args.cc,
            "-m32",
            "-shared",
            "-Wl,-z,notext",
            "-Wl,--allow-multiple-definition",
            "-Wl,-Bsymbolic-functions",
            "-Wl,--build-id=none",
            f"-Wl,-soname,{OUTPUT_LIBRARY_SONAME}",
            localized,
            bridge_object,
            runtime_object,
            "-lm",
            "-o",
            library,
        )
    )
    # MSVC CodeView sections contain absolute temporary object paths. They are
    # non-runtime data and make otherwise identical builds hash differently.
    run((args.objcopy, "--strip-debug", library))
    missing_exports = set(PUBLIC_EXPORTS) - defined_dynamic_symbols(args.nm, library)
    if missing_exports:
        raise HostingError(f"native library is missing exports: {sorted(missing_exports)}")
    interposable_helpers = interposable_msvc_b2d_relocations(
        args.readelf, library
    )
    if interposable_helpers:
        raise HostingError(
            "native library retains interposable internal Box2D relocations: "
            f"{sorted(interposable_helpers)}"
        )

    metadata: dict[str, object] = {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "architecture": "ELF32 i386",
        "source_mode": "compiled_by_tool" if args.object_dir is None else "provided_coff_objects",
        "compiler_identity": compiler_identity,
        "compile_flags": list(COMPILE_FLAGS) if args.object_dir is None else None,
        "wrapper_source_sha256": sha256(args.wrapper.resolve()),
        "host_source_sha256": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in (BRIDGE_SOURCE, RUNTIME_SOURCE, *RUNTIME_DEPENDENCIES)
        },
        "source_sha256": source_hashes,
        "coff_object_sha256": {path.name: sha256(path) for path in objects},
        "weak_aliases": [{"symbol": old, "fallback": new} for old, new in aliases],
        "rel32_addends_adjusted": rel32_count,
        "tools": {
            "cc": tool_identity(args.cc),
            "ld": tool_identity(args.ld),
            "nm": tool_identity(args.nm),
            "objcopy": objcopy_identity,
            "readelf": tool_identity(args.readelf),
        },
        "output": {
            "path": str(library),
            "soname": OUTPUT_LIBRARY_SONAME,
            "sha256": sha256(library),
            "exports": sorted(PUBLIC_EXPORTS),
            "requires_32_bit_consumer": True,
            "contains_text_relocations": True,
            "internal_msvc_b2d_functions_bound_locally": True,
        },
        "validation": {
            "performed_by_this_command": False,
            "evidence": "reference/native-box2d/validation.json",
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--object-dir", type=Path, help="directory containing the 28 exact MSVC9 .obj files")
    mode.add_argument("--source-dir", type=Path, help="pristine Box2D r58 source root to compile")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--wrapper", type=Path, default=ASSETS / "box2d-wrapper-msvc.cpp")
    result.add_argument("--cl", type=Path, help="MSVC9 cl.exe (required with --source-dir)")
    result.add_argument("--vc-include", type=Path, help="MSVC9 VC include directory")
    result.add_argument("--wine", type=Path, help="Wine executable used to run cl.exe")
    result.add_argument("--winepath", type=Path, help="matching winepath executable")
    result.add_argument("--wine-prefix", type=Path)
    result.add_argument("--cc", default="gcc")
    result.add_argument("--ld", default="ld")
    result.add_argument("--nm", default="nm")
    result.add_argument("--objcopy", default="objcopy")
    result.add_argument("--readelf", default="readelf")
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.source_dir is not None:
        required = {"--cl": args.cl, "--vc-include": args.vc_include, "--wine": args.wine, "--winepath": args.winepath}
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(f"--source-dir requires {', '.join(missing)}")
    try:
        metadata = build(args)
    except HostingError as exc:
        raise SystemExit(str(exc)) from exc
    print(metadata["output"]["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
