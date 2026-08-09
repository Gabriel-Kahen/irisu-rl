# x87 transcendental differential harness

This tool compares the native x87 `FSIN`, `FCOS`, and `FSINCOS` oracle with
the correctly rounded binary32 `cr_sincosf` implementation from CORE-MATH.
Every result is rounded to binary32 before comparison, matching the observed
Box2D call sites. The harness installs and verifies x87 control word `0x027f`,
records the architectural status flags before the final store, checks the
`|x| >= 2^63` C2/out-of-range behavior, and verifies that `FSINCOS` agrees
with the standalone instructions.

The report also checks a browser-portable status model: zero is exact, ordinary
finite nonzero inputs raise precision, subnormal inputs raise denormal plus
precision, finite values at or above the range limit set C2, infinities and
signaling NaNs raise invalid, and quiet NaNs pass without an exception. This
models the flags observed before the final binary32 store.

`core_math_sincosf.c` is copied from CORE-MATH commit
`07cf01e12a42b82cc478341982936cad7f3f9bdc` under its included MIT license.
The candidate adds an x87 range guard: finite inputs with raw absolute bits at
or above `0x5f000000` retain the input and report no `FSINCOS` pair. Status
flags are measured from the oracle and reported separately; CORE-MATH does not
claim x87 status emulation.

Build and run the smoke gate:

```sh
make -C tools/x87-trig-differential check
```

Run the preserved 3,000-decision capture (13,096,069 raw keys, or 10,080,004
unique nonzero keys plus zero after deduplication):

```sh
make -C tools/x87-trig-differential captured
```

The input format is a sequence of little-endian raw binary32 words. Other
corpora can be supplied directly:

```sh
tools/x87-trig-differential/x87-trig-diff \
  --input keys.bin --no-curated --random 1000000 --deduplicate \
  --candidate core-math > report.json
```

An exhaustive positive-significand sweep over one or more binary32 exponent
bins is also supported. Negative inputs need not be duplicated when checking
finite sine/cosine result magnitude because sine is odd and cosine is even:

```sh
tools/x87-trig-differential/x87-trig-diff --no-curated \
  --exponent-min 128 --exponent-max 131 > top-reachable-exponents.json
```

For a bounded-memory exhaustive sweep across many exponent bins, use the
chunked runner:

```sh
python3 tools/x87-trig-differential/sweep_exponents.py \
  --first 0 --last 145 --candidate v86-libm > exponent-sweep.json
```

Candidates are `core-math`, the host's f64 libm (`host-f64`), and the Rust
libm algorithm used by v86's WebAssembly build (`v86-libm`). Stock v86 is exact
on all captured keys, but not on the whole bounded domain: its first exhaustive
counterexample is FSIN input `0x46199998`, where it produces `0xbeb1fa5e`
instead of native x87 `0xbeb1fa5d`.
An actual stock-v86 guest probe also confirms that stock omits IE/DE/PE status
updates and the `2^63` C2 range behavior; it is not a total-parity candidate.

## Patched v86

`v86-core-math-x87.patch` applies a bounded CORE-MATH path to v86 commit
`f3d4472a9c934b9ad78a311f5849ba711a296d23`. Exactly binary32-reloadable
operands with magnitude below `2^19` use `cr_sincosf`; larger or non-binary32
operands retain the generic v86 fallback. The patch also implements the x87
`2^63` C2/no-push behavior and the validated IE/DE/PE status model. This bound
is deliberate: the final-binary32 CORE-MATH result is exhaustive below it,
whereas target x87 has finite-pi discrepancies for larger angles.

Build the pinned Wasm from a clean clone:

```sh
./tools/x87-trig-differential/build_patched_v86.sh \
  .tmp/v86-core-math-src /tmp/v86-core-math.wasm
```

With rustc 1.96.1 (`31fca3adb`) and clang 22.1.6 the resulting SHA-256 is
`73d1023eba1729d6aa6a9a3d3d52122c88e8f05b775caaa0557e042f68c34403`.
Run a real i386 guest instruction differential against the installed
`experiments/v86-exact-poc/runtime/v86.wasm` with:

```sh
python3 tools/x87-trig-differential/run_v86_probe.py
```

The current gate executes 100,000 stratified captured inputs plus the stock
counterexample, its sign reflection, proof-boundary vectors, and x87 range
vectors. It compares native and emulated result/status records byte for byte.

The report distinguishes exact raw-bit matches, NaN-equivalent matches,
one-ULP mismatches, larger mismatches, and a bounded set of counterexamples.
It also includes native status-word distributions and range/C2 invariant
failures. The native oracle requires an x86 host; the CORE-MATH candidate is
ordinary C and can be compiled for WebAssembly, with a correctly rounded
`fmaf` helper where the WebAssembly C runtime does not supply one.

## Current result

The 2026-08-09 Ryzen 7 3700X oracle run found exact final-binary32 agreement
for all 10,080,004 unique captured gameplay keys. An exhaustive positive-input
sweep then found exact agreement for all 1,224,736,768 binary32 encodings below
`524288`; the captured maximum was below `24`. The first discrepancies occur
in exponent 146 (`524288` through just below `1048576`): 3 sine and 1 cosine
mismatches among that exponent's 8,388,608 positive encodings. The portable
status model and `FSINCOS` range/pair behavior had no mismatches in the curated,
captured, or 9,988,273-input uniform-raw random runs. The numerical exhaustive
sweep ran before the status-model counter was added, so it is not claimed as an
exhaustive status-model validation.

The rebuilt v86 passed the 100,014-input guest differential with zero
mismatches (3,100,434 identical bytes, SHA-256
`535b054f721dbe9a72201d2d9905496143502ac56be5421c816be993a6312011`).
Its 20-tick exact-worker response matched the native SHA-256, and the
47,019-tick replay produced byte-identical full JSON with SHA-256
`bb8a81554cfc03bbfa186fdba6ee080f691eaabb451e1f562627050ed2fdf236`.
That long run used the same CORE path immediately before the `>=2^19`
fallback guard was added; the final artifact's short and direct-instruction
gates are exact, while its complete replay-corpus report is maintained by the
production integration.

See `results/validation-20260809.json`. This establishes a practical exact
browser candidate for the observed gameplay angle domain. It does not establish
arbitrary-large-angle parity; such inputs still need a correction table or an
x87-compatible finite-pi reduction implementation.
