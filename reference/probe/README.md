# Shipped Box2D DLL oracle

This directory contains a clean-room measurement tool for the exact
`Box2D.dll` shipped with IriSu Syndrome v2.03. It is an oracle for calibration,
not a dependency of the clone. The original DLL remains under the ignored
`reference/game/` tree and must never be copied into source distributions.

## Run

```bash
tools/run-box2d-probe.sh --output reference/probe/out/current.jsonl
```

The runner verifies the DLL SHA-256
`34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd`,
builds a PE32 probe, runs it under the bundled Wine prefix, validates the JSONL,
and writes only the probe trace. The DLL's own debug `printf` output is captured
separately in the disposable run directory so it cannot corrupt the JSONL.

Use `reference/probe/test.sh` to run the oracle twice and require byte-for-byte
identical traces. Environment overrides are `IRISU_BOX2D_DLL`,
`IRISU_WINE_BIN`, and `IRISU_WINEPREFIX`.

## Build design

No MinGW compiler is installed on this host. `build.sh` instead uses the
installed Clang/LLD/LLVM dlltool trio to produce a freestanding i686 Windows
executable. The executable imports only eight `KERNEL32.dll` operations, loads
`Box2D.dll` dynamically, and resolves the 16 decorated `stdcall` exports
recorded in `reference/binary-analysis.md`. Floating-point formatting is
resolved dynamically from Wine's `msvcrt.dll`; no C runtime is linked.

The source and import definition are redistributable project code. Generated
objects, import libraries, and executables live in ignored `build/`.

## Scenarios and units

The schema is zero-based, contiguous JSONL (`schema: 2`). Body records include
both decimal values and raw IEEE-754 bit patterns so sleeping and repeatability
can be distinguished from formatting-level convergence. It records:

- all 16 resolved export names;
- initialization bounds, gravity, magnification, and return value;
- initial box, circle, and right-triangle transforms;
- the wrapper's asymmetric velocity conversion and one `0.020`, 10-iteration
  step;
- `set_position`, including its linear-velocity reset;
- execution of the unused `b2d_test` operation;
- a density-zero static floor with falling box/circle/triangle contacts;
- a density-zero static wall and restitution response;
- controlled friction pairs (`0`, `sqrt(0.25)`, and `1`) under equal load;
- the exact resting-contact sleep transition and post-sleep `set_v` behavior;
- discriminating point/edge contacts against an unrotated `100 x 60` triangle;
- paired box heights and circle radii against the same floor;
- positive/negative gravity and magnification scaling trials;
- simultaneous and incrementally prepended contact-list ordering;
- contact cursor order and body user-data values;
- valid and null body destruction, world disposal, and DLL unloading.

Coordinates returned by the DLL are in magnified/pixel units. `vx_world` and
`vy_world` are deliberately named as such because `b2d_get_v` returns raw world
units instead of multiplying by the configured magnification.

`validate_trace.py` checks both structure and high-value numerical invariants.
Exact conclusions from the current trace are summarized in `observations.md`.
The reproducible clean numerical artifact is
`golden/box2d-v2.03-wine11.13.jsonl` (1,290 records, SHA-256
`b73f9c74db48a9695450ab04b76661c0d576030f7f428d0e53b953b4dae077b2`).
It contains measurements and identifiers only—no bytes from the original DLL.
