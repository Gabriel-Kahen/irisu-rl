# DxLib RNG oracle

This directory contains a clean-room Python model of the random-number
generator used by the shipped v2.03 `DxLib.dll`, plus a tiny Windows probe for
checking the model against a locally held copy of that DLL.  No original DLL
or DxLib source is redistributed here.

The model was recovered from executable behavior and disassembly.  It uses a
624-word MT19937 state, DxLib's two-step `69069*x+1` seed expansion, the normal
MT19937 twist/temper constants, and inclusive high-product range scaling:

```text
GetRand(maximum) = floor(raw_u32 * (maximum + 1) / 2^32)
```

That last operation is not `raw_u32 % (maximum + 1)`.

## Model

```sh
python3 dxlib_rng.py --seed 0 100 12 69 404 1000 3 100 1000 5 100
python3 test_dxlib_rng.py
```

The first command prints:

```text
11
11
36
264
666
0
28
813
0
71
```

## Optional local-DLL check

The probe is intentionally built outside the source tree in this example.
It requires `clang`, `lld-link`, `llvm-dlltool`, Wine, and the preserved local
v2.03 `DxLib.dll`:

```sh
probe_build=$(mktemp -d /tmp/irisu-dxlib-rng-XXXXXX)
./build.sh "$probe_build"
cp ../game/irisu-v2.03-en/data/dll/DxLib.dll "$probe_build/"
(cd "$probe_build" && wine ./dxlib-rng-oracle.exe)
cat "$probe_build/dxlib-rng-oracle.txt"
```

The analyzed DLL has SHA-256
`d8ef638a078a8b4d24b53b174ca179623fed3690027d3f4acfe71a7d61c8b5c9`.
The checked vectors cover seeds `0`, `1`, `0x12345678`, and `0x3fffffff`
with the maxima used by the first normal-mode spawn path.  The Python model
matches every emitted result.

