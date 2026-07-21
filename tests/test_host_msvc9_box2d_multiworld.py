from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_tool():
    name = "irisu_host_msvc9_box2d_multiworld"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools/host-msvc9-box2d-multiworld.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


class NativeBox2DMultiworldHostingTests(unittest.TestCase):
    def test_generated_library_has_a_stable_name_and_soname(self) -> None:
        self.assertEqual(
            TOOL.LIBRARY_NAME, "libirisu_box2d_msvc_exact_multiworld.so"
        )
        self.assertEqual(TOOL.BASE.OUTPUT_LIBRARY_NAME, TOOL.LIBRARY_NAME)
        self.assertEqual(TOOL.BASE.OUTPUT_LIBRARY_SONAME, TOOL.LIBRARY_NAME)

    def test_exports_have_the_expected_msvc_stdcall_widths(self) -> None:
        self.assertEqual(len(TOOL.EXPORT_RENAMES), 16)
        self.assertEqual(
            TOOL.EXPORT_RENAMES["_b2d_world_create@24"],
            "msvc_b2d_world_create",
        )
        self.assertEqual(
            TOOL.EXPORT_RENAMES["_b2d_world_create_box@36"],
            "msvc_b2d_world_create_box",
        )
        self.assertEqual(
            TOOL.EXPORT_RENAMES["_b2d_world_set_position@20"],
            "msvc_b2d_world_set_position",
        )
        self.assertEqual(len(TOOL.BASE.OBJECT_NAMES), 28)

    def test_wrapper_state_is_per_handle(self) -> None:
        source = (
            ROOT
            / "reference/native-box2d/multiworld/box2d-wrapper-msvc.cpp"
        ).read_text(encoding="utf-8")
        self.assertIn("struct b2dWorldHandle", source)
        self.assertIn("b2Contact* contact;", source)
        self.assertIn("float magnification;", source)
        self.assertNotIn("static b2World* g_world", source)
        self.assertNotIn("static b2Contact* g_contact", source)

    def test_host_bridge_serializes_pristine_r58_shared_state(self) -> None:
        bridge = (
            ROOT / "reference/native-box2d/multiworld/msvc-bridge.c"
        ).read_text(encoding="utf-8")
        self.assertIn("__sync_lock_test_and_set", bridge)
        self.assertIn("__sync_lock_release", bridge)

    def test_host_metadata_covers_runtime_include_dependency(self) -> None:
        self.assertEqual(
            TOOL.BASE.BRIDGE_SOURCE,
            ROOT / "reference/native-box2d/multiworld/msvc-bridge.c",
        )
        self.assertEqual(
            TOOL.BASE.RUNTIME_SOURCE,
            ROOT / "reference/native-box2d/multiworld/msvc-runtime.S",
        )
        self.assertEqual(
            TOOL.BASE.RUNTIME_DEPENDENCIES,
            (ROOT / "reference/native-box2d/msvc-runtime.S",),
        )


if __name__ == "__main__":
    unittest.main()
