from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    name = "irisu_host_msvc9_box2d"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools/host-msvc9-box2d.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


class NativeBox2DHostingTests(unittest.TestCase):
    def test_internal_msvc_relocation_detection_is_fail_closed(self) -> None:
        relocations = "\n".join(
            (
                "00019004  00001707 R_386_JUMP_SLOT 00015750 msvc_b2d_world_step",
                "00018fe4  00000206 R_386_GLOB_DAT 00000000 msvc_b2d_world_get_x",
                "00019000  00000107 R_386_JUMP_SLOT 00000000 abort@GLIBC_2.0",
                "00000000  00000000 R_386_RELATIVE 00000000 msvc_b2d_ignored",
            )
        )
        with mock.patch.object(
            TOOL, "run", return_value=SimpleNamespace(stdout=relocations)
        ) as run:
            self.assertEqual(
                TOOL.interposable_msvc_b2d_relocations("readelf", Path("host.so")),
                {"msvc_b2d_world_step", "msvc_b2d_world_get_x"},
            )
        run.assert_called_once_with(
            ("readelf", "-rW", Path("host.so")), capture=True
        )

    def test_coff_weak_external_resolves_its_auxiliary_fallback(self) -> None:
        header = TOOL.COFF_HEADER.pack(0x14C, 0, 0, TOOL.COFF_HEADER.size, 3, 0, 0)
        weak = TOOL.COFF_SYMBOL.pack(b"weak\0\0\0\0", 0, 0, 0, 105, 1)
        auxiliary = struct.pack("<I", 2) + bytes(14)
        strong = TOOL.COFF_SYMBOL.pack(b"strong\0\0", 0, 1, 0, 2, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weak.obj"
            path.write_bytes(header + weak + auxiliary + strong + struct.pack("<I", 4))
            self.assertEqual(TOOL.coff_weak_aliases(path), [("weak", "strong")])

    def test_rel32_conversion_subtracts_the_elf_program_counter_bias(self) -> None:
        section_count = 3
        section_offset = 52
        data_offset = section_offset + section_count * TOOL.ELF32_SECTION.size
        ident = b"\x7fELF\x01\x01\x01" + bytes(9)
        header = ident + struct.pack(
            "<HHIIIIIHHHHHH",
            1,
            3,
            1,
            0,
            0,
            section_offset,
            0,
            52,
            0,
            0,
            TOOL.ELF32_SECTION.size,
            section_count,
            0,
        )
        null = TOOL.ELF32_SECTION.pack(*([0] * 10))
        target = TOOL.ELF32_SECTION.pack(0, 1, 0, 0, data_offset, 4, 0, 0, 4, 0)
        relocations = TOOL.ELF32_SECTION.pack(
            0, TOOL.SHT_REL, 0, 0, data_offset + 4, TOOL.ELF32_REL.size,
            0, 1, 4, TOOL.ELF32_REL.size
        )
        relocation = TOOL.ELF32_REL.pack(0, TOOL.R_386_PC32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "one.o"
            path.write_bytes(header + null + target + relocations + bytes(4) + relocation)
            self.assertEqual(TOOL.patch_elf_rel32(path), 1)
            self.assertEqual(path.read_bytes()[data_offset : data_offset + 4], b"\xfc\xff\xff\xff")

    def test_checked_in_wrapper_and_validation_are_consistent(self) -> None:
        wrapper = ROOT / "reference/native-box2d/box2d-wrapper-msvc.cpp"
        validation = json.loads(
            (ROOT / "reference/native-box2d/validation.json").read_text()
        )
        digest = hashlib.sha256(wrapper.read_bytes()).hexdigest()
        self.assertEqual(digest, validation["source"]["wrapper_sha256"])
        self.assertEqual(validation["native_host"]["full_first_bad_step"], 0)
        self.assertEqual(validation["native_host"]["rel32_addends_adjusted"], 467)
        self.assertEqual(validation["oracle"]["steps"], 47019)
        self.assertEqual(
            validation["oracle"]["status"], "exact_through_final_physics_step"
        )
        self.assertEqual(validation["oracle"]["streams"]["contact"], 573557)
        self.assertIsNone(validation["oracle"]["first_mismatch"])
        self.assertIn(
            TOOL.EXPECTED_COMPILER_VERSION, validation["compiler"]["identity"]
        )
        self.assertEqual(len(TOOL.OBJECT_NAMES), 28)
        self.assertEqual(len(TOOL.PUBLIC_EXPORTS), 16)


if __name__ == "__main__":
    unittest.main()
