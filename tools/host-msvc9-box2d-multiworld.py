#!/usr/bin/env python3
"""Host the handle-based exact MSVC9 Box2D r58 wrapper as an ELF32 library."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "reference/native-box2d/multiworld"


def load_base():
    path = ROOT / "tools/host-msvc9-box2d.py"
    spec = importlib.util.spec_from_file_location("irisu_host_msvc9_box2d_base", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()
EXPORT_RENAMES = {
    "_b2d_world_create@24": "msvc_b2d_world_create",
    "_b2d_world_destroy@4": "msvc_b2d_world_destroy",
    "_b2d_world_create_box@36": "msvc_b2d_world_create_box",
    "_b2d_world_create_circle@28": "msvc_b2d_world_create_circle",
    "_b2d_world_create_triangle@36": "msvc_b2d_world_create_triangle",
    "_b2d_world_destroy_body@8": "msvc_b2d_world_destroy_body",
    "_b2d_world_step@12": "msvc_b2d_world_step",
    "_b2d_world_get_contact@12": "msvc_b2d_world_get_contact",
    "_b2d_world_get_x@8": "msvc_b2d_world_get_x",
    "_b2d_world_get_y@8": "msvc_b2d_world_get_y",
    "_b2d_world_get_r@8": "msvc_b2d_world_get_r",
    "_b2d_world_get_v@16": "msvc_b2d_world_get_v",
    "_b2d_world_set_v@16": "msvc_b2d_world_set_v",
    "_b2d_world_set_user_data@12": "msvc_b2d_world_set_user_data",
    "_b2d_world_set_position@20": "msvc_b2d_world_set_position",
    "_b2d_world_test@8": "msvc_b2d_world_test",
}
PUBLIC_EXPORTS = tuple(name.removeprefix("msvc_") for name in EXPORT_RENAMES.values())
LIBRARY_NAME = "libirisu_box2d_msvc_exact_multiworld.so"

BASE.ASSETS = ASSETS
BASE.BRIDGE_SOURCE = ASSETS / "msvc-bridge.c"
BASE.RUNTIME_SOURCE = ASSETS / "msvc-runtime.S"
BASE.RUNTIME_DEPENDENCIES = (ASSETS.parent / "msvc-runtime.S",)
BASE.EXPORT_RENAMES = EXPORT_RENAMES
BASE.PUBLIC_EXPORTS = PUBLIC_EXPORTS
BASE.OUTPUT_LIBRARY_NAME = LIBRARY_NAME
BASE.OUTPUT_LIBRARY_SONAME = LIBRARY_NAME


def parser():
    result = BASE.parser()
    result.description = __doc__
    return result


def build(args):
    metadata = BASE.build(args)
    metadata["schema"] = 2
    metadata["api"] = {
        "kind": "opaque_world_handles",
        "thread_safety": "public calls serialized by host bridge",
        "independent_worlds": True,
    }
    metadata["validation"] = {
        "performed_by_this_command": False,
        "evidence": "run tools/exact-physics-prototype/multiworld_smoke.c",
    }
    (args.output_dir.resolve() / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.source_dir is not None:
        required = {
            "--cl": args.cl,
            "--vc-include": args.vc_include,
            "--wine": args.wine,
            "--winepath": args.winepath,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise SystemExit(f"--source-dir requires {', '.join(missing)}")
    try:
        metadata = build(args)
    except BASE.HostingError as exc:
        raise SystemExit(str(exc)) from exc
    print(metadata["output"]["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
