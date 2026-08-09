#!/usr/bin/env python3
"""Package the fixed ELF32 exact-physics host as a zero-based guest image."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
from collections import Counter
from pathlib import Path


ELF_HEADER = struct.Struct("<16sHHIIIIIHHHHHH")
PROGRAM_HEADER = struct.Struct("<IIIIIIII")
SECTION_HEADER = struct.Struct("<IIIIIIIIII")
SYMBOL = struct.Struct("<IIIBBH")
RELOCATION = struct.Struct("<II")

PT_LOAD = 1
SHT_SYMTAB = 2
SHT_RELA = 4
SHT_REL = 9
SHT_DYNSYM = 11
SHN_UNDEF = 0
EM_386 = 3
ATTESTED_HOST_SHA256 = "ce14d1cab9ce4331bf494fe92bf657029487aec9f7435e7479b3c7cb579fafb5"

RELOCATION_NAMES = {
    1: "R_386_32",
    2: "R_386_PC32",
    6: "R_386_GLOB_DAT",
    7: "R_386_JUMP_SLOT",
    8: "R_386_RELATIVE",
}

PUBLIC_ABI = {
    "b2d_world_create": "void *(f32,f32,f32,f32,f32,f32)",
    "b2d_world_destroy": "void(void *)",
    "b2d_world_create_box": "void *(void *,f32,f32,f32,f32,f32,f32,f32,f32)",
    "b2d_world_create_triangle": "void *(void *,f32,f32,f32,f32,f32,f32,f32,f32)",
    "b2d_world_create_circle": "void *(void *,f32,f32,f32,f32,f32,f32)",
    "b2d_world_destroy_body": "void(void *,void *)",
    "b2d_world_step": "void(void *,f32,i32)",
    "b2d_world_get_contact": "i32(void *,void **,void **)",
    "b2d_world_get_x": "f32(void *,void *)",
    "b2d_world_get_y": "f32(void *,void *)",
    "b2d_world_get_r": "f32(void *,void *)",
    "b2d_world_get_v": "void(void *,void *,f32 *,f32 *)",
    "b2d_world_set_v": "void(void *,void *,f32,f32)",
    "b2d_world_set_user_data": "void(void *,void *,void *)",
    "b2d_world_set_position": "void(void *,void *,f32,f32,f32)",
    "b2d_world_test": "void(void *,void *)",
}

STRONG_IMPORTS = {"abort", "free", "malloc", "memcpy", "memmove", "memset"}
WEAK_IMPORTS = {
    "_ITM_deregisterTMCloneTable",
    "_ITM_registerTMCloneTable",
    "__cxa_finalize",
    "__gmon_start__",
}


def c_string(data: bytes, offset: int) -> str:
    end = data.find(b"\0", offset)
    if end < 0:
        raise ValueError("unterminated ELF string")
    return data[offset:end].decode("utf-8", "strict")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_elf(data: bytes) -> dict:
    if len(data) < ELF_HEADER.size:
        raise ValueError("input is shorter than an ELF32 header")
    fields = ELF_HEADER.unpack_from(data)
    ident = fields[0]
    if ident[:7] != b"\x7fELF\x01\x01\x01":
        raise ValueError("expected little-endian ELF32")
    machine = fields[2]
    if machine != EM_386:
        raise ValueError(f"expected EM_386, got {machine}")
    phoff, shoff = fields[5], fields[6]
    phentsize, phnum = fields[9], fields[10]
    shentsize, shnum, shstrndx = fields[11], fields[12], fields[13]
    if phentsize != PROGRAM_HEADER.size or shentsize != SECTION_HEADER.size:
        raise ValueError("unexpected ELF table entry size")

    programs = [PROGRAM_HEADER.unpack_from(data, phoff + i * phentsize) for i in range(phnum)]
    sections_raw = [SECTION_HEADER.unpack_from(data, shoff + i * shentsize) for i in range(shnum)]
    shstr = sections_raw[shstrndx]
    shstr_data = data[shstr[4] : shstr[4] + shstr[5]]
    sections = []
    for index, section in enumerate(sections_raw):
        sections.append(
            {
                "index": index,
                "name": c_string(shstr_data, section[0]),
                "type": section[1],
                "flags": section[2],
                "addr": section[3],
                "offset": section[4],
                "size": section[5],
                "link": section[6],
                "info": section[7],
                "align": section[8],
                "entsize": section[9],
            }
        )

    symbol_tables: dict[int, list[dict]] = {}
    for section in sections:
        if section["type"] not in (SHT_SYMTAB, SHT_DYNSYM):
            continue
        strings = sections[section["link"]]
        string_data = data[strings["offset"] : strings["offset"] + strings["size"]]
        symbols = []
        for offset in range(section["offset"], section["offset"] + section["size"], SYMBOL.size):
            name, value, size, info, other, shndx = SYMBOL.unpack_from(data, offset)
            symbols.append(
                {
                    "name": c_string(string_data, name),
                    "value": value,
                    "size": size,
                    "bind": info >> 4,
                    "type": info & 0xF,
                    "visibility": other & 0x3,
                    "shndx": shndx,
                }
            )
        symbol_tables[section["index"]] = symbols

    relocations = []
    for section in sections:
        if section["type"] not in (SHT_REL, SHT_RELA):
            continue
        if section["type"] == SHT_RELA:
            raise ValueError("ELF32 exact host unexpectedly uses RELA relocations")
        symbols = symbol_tables[section["link"]]
        for offset in range(section["offset"], section["offset"] + section["size"], RELOCATION.size):
            address, info = RELOCATION.unpack_from(data, offset)
            symbol_index, kind = info >> 8, info & 0xFF
            relocations.append(
                {
                    "section": section["name"],
                    "offset": address,
                    "type": RELOCATION_NAMES.get(kind, f"R_386_{kind}"),
                    "symbol": symbols[symbol_index]["name"] if symbol_index else "",
                }
            )

    dynsym_section = next(section for section in sections if section["type"] == SHT_DYNSYM)
    dynamic_symbols = symbol_tables[dynsym_section["index"]]
    imports = sorted(
        (
            {
                "name": symbol["name"],
                "binding": "weak" if symbol["bind"] == 2 else "strong",
                "type": symbol["type"],
            }
            for symbol in dynamic_symbols
            if symbol["name"] and symbol["shndx"] == SHN_UNDEF
        ),
        key=lambda item: item["name"],
    )
    exports = {
        symbol["name"]: {"address": symbol["value"], "size": symbol["size"]}
        for symbol in dynamic_symbols
        if symbol["name"] and symbol["shndx"] != SHN_UNDEF and symbol["bind"] in (1, 2)
    }

    load_segments = []
    image_size = 0
    for kind, offset, vaddr, _paddr, filesz, memsz, flags, align in programs:
        if kind != PT_LOAD:
            continue
        load_segments.append(
            {
                "file_offset": offset,
                "virtual_address": vaddr,
                "file_size": filesz,
                "memory_size": memsz,
                "flags": "".join(letter for bit, letter in ((4, "R"), (2, "W"), (1, "X")) if flags & bit),
                "alignment": align,
            }
        )
        image_size = max(image_size, vaddr + memsz)
    image_size = (image_size + 0xFFF) & ~0xFFF
    image = bytearray(image_size)
    occupied = bytearray(image_size)
    for segment in load_segments:
        source = data[segment["file_offset"] : segment["file_offset"] + segment["file_size"]]
        start = segment["virtual_address"]
        for index, value in enumerate(source, start):
            if occupied[index] and image[index] != value:
                raise ValueError(f"conflicting PT_LOAD bytes at guest address 0x{index:x}")
            image[index] = value
            occupied[index] = 1

    return {
        "image": bytes(image),
        "segments": load_segments,
        "imports": imports,
        "exports": exports,
        "relocations": relocations,
        "sections": sections,
    }


def instruction_inventory(path: Path, objdump: str) -> tuple[int, dict[str, int], str]:
    version = subprocess.run(
        [objdump, "--version"], check=True, text=True, capture_output=True
    ).stdout.splitlines()[0]
    disassembly = subprocess.run(
        [objdump, "-d", "-M", "intel", "--insn-width=16", str(path)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    counts: Counter[str] = Counter()
    pattern = re.compile(r"^\s*[0-9a-f]+:\s+(?:(?:[0-9a-f]{2})\s+)+\s*([a-z][a-z0-9]*)", re.I)
    for line in disassembly.splitlines():
        match = pattern.match(line)
        if match:
            counts[match.group(1).lower()] += 1
    return sum(counts.values()), dict(sorted(counts.items())), version


def write_image_include(path: Path, image: bytes, entry: int) -> None:
    lines = [
        "/* Generated by package_exact_host.py; do not edit. */",
        f"#define EXACT_HOST_IMAGE_SIZE {len(image)}u",
        f"#define EXACT_HOST_WORLD_TEST_ENTRY 0x{entry:08x}u",
        f"#define EXACT_HOST_IMAGE_SHA256 \"{sha256(image)}\"",
        "static unsigned char guest[EXACT_HOST_IMAGE_SIZE + 4096u] = {",
    ]
    for offset in range(0, len(image), 16):
        chunk = ", ".join(f"0x{byte:02x}" for byte in image[offset : offset + 16])
        lines.append(f"  {chunk},")
    lines.append("};")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("host", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--objdump", default="objdump")
    parser.add_argument("--expected-sha256", default=ATTESTED_HOST_SHA256)
    args = parser.parse_args()

    source = args.host.read_bytes()
    source_hash = sha256(source)
    if source_hash != args.expected_sha256:
        raise SystemExit(
            f"exact host hash changed: expected {args.expected_sha256}, got {source_hash}"
        )
    parsed = parse_elf(source)
    missing = sorted(set(PUBLIC_ABI) - set(parsed["exports"]))
    if missing:
        raise SystemExit(f"exact host is missing public ABI exports: {', '.join(missing)}")
    actual_strong = {item["name"] for item in parsed["imports"] if item["binding"] == "strong"}
    actual_weak = {item["name"] for item in parsed["imports"] if item["binding"] == "weak"}
    if actual_strong != STRONG_IMPORTS or actual_weak != WEAK_IMPORTS:
        raise SystemExit(
            "exact host import boundary changed: "
            f"strong={sorted(actual_strong)}, weak={sorted(actual_weak)}"
        )
    private_test = parsed["exports"].get("msvc_b2d_world_test")
    if not private_test:
        raise SystemExit("exact host is missing msvc_b2d_world_test")

    instruction_count, instructions, objdump_version = instruction_inventory(args.host, args.objdump)
    relocation_counts = Counter(relocation["type"] for relocation in parsed["relocations"])
    relocation_symbol_counts = Counter(
        relocation["symbol"] for relocation in parsed["relocations"] if relocation["symbol"]
    )
    x87 = {name: count for name, count in instructions.items() if name.startswith("f")}
    manifest = {
        "schema": 1,
        "source": {
            "file_name": args.host.name,
            "size": len(source),
            "sha256": source_hash,
            "format": "ELF32-i386-little-endian",
        },
        "guest_image": {
            "load_base": 0,
            "size": len(parsed["image"]),
            "sha256": sha256(parsed["image"]),
            "segments": parsed["segments"],
            "relocation_policy": "R_386_RELATIVE addends are already correct at load base zero; imported relocations remain runtime boundaries",
        },
        "abi": {
            "public_calling_convention": "cdecl",
            "private_msvc_calling_convention": "stdcall",
            "pointer_bits": 32,
            "float_bits": 32,
            "required_x87_control_word": "0x027f",
            "public_exports": PUBLIC_ABI,
        },
        "symbols": {
            "imports": parsed["imports"],
            "public_export_addresses": {
                name: parsed["exports"][name]["address"] for name in sorted(PUBLIC_ABI)
            },
            "smoke_entry": {
                "name": "msvc_b2d_world_test",
                "address": private_test["address"],
            },
        },
        "relocations": {
            "total": len(parsed["relocations"]),
            "by_type": dict(sorted(relocation_counts.items())),
            "by_import": dict(sorted(relocation_symbol_counts.items())),
        },
        "instructions": {
            "decoder": objdump_version,
            "total": instruction_count,
            "mnemonics": instructions,
            "x87_total": sum(x87.values()),
            "x87_mnemonics": x87,
        },
        "translation_route": {
            "mode": "fixed-image AOT/basic-block translation",
            "runtime_imports": sorted(STRONG_IMPORTS),
            "weak_noop_imports": sorted(WEAK_IMPORTS),
            "x87_strategy": "software x87 state; PC53 arithmetic may use f64 only after bitwise differential proof; transcendental instructions require an oracle-matched helper",
        },
    }

    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "exact-host.image.bin").write_bytes(parsed["image"])
    write_image_include(
        args.output / "exact_host_image.inc", parsed["image"], private_test["address"]
    )
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
