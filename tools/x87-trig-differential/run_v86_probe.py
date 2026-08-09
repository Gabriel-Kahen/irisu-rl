#!/usr/bin/env python3
import argparse
import hashlib
import base64
import json
import os
import pathlib
import struct
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
CAPTURED = ROOT / ".tmp/exact-core-perf-20260721/trig-keys-3000.bin"


def digest(path: pathlib.Path) -> dict[str, object]:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


parser = argparse.ArgumentParser()
parser.add_argument("--sample-count", type=int, default=100_000)
parser.add_argument("--wasm", type=pathlib.Path)
args = parser.parse_args()
if args.sample_count < 0:
    parser.error("--sample-count must be nonnegative")

with tempfile.TemporaryDirectory(prefix="irisu-v86-trig-") as temporary:
    tmp = pathlib.Path(temporary)
    probe = tmp / "probe"
    corpus = tmp / "corpus.bin"
    native = tmp / "native.bin"
    subprocess.run([
        "gcc", "-m32", "-O2", "-nostdlib", "-static", "-fno-pie", "-no-pie",
        "-march=i686", "-mno-sse", "-fno-stack-protector",
        str(HERE / "guest_trig_probe.c"), "-o", str(probe),
    ], check=True)
    raw = CAPTURED.read_bytes()
    sample = []
    if args.sample_count:
        stride = max(1, len(raw) // 4 // args.sample_count)
        words = struct.iter_unpack("<I", raw)
        sample = [word[0] for index, word in enumerate(words) if index % stride == 0][:args.sample_count]
    sample += [0, 0x80000000, 1, 0x80000001, 0x3f800000, 0xbf800000,
               0x46199998, 0xc6199998,
               0x48ffffff, 0xc8ffffff, 0x49000000, 0xc9000000,
               0x5f000000, 0xdf000000]
    corpus.write_bytes(struct.pack(f"<{len(sample)}I", *sample))
    subprocess.run([str(probe), str(corpus), str(native)], check=True)
    environment = os.environ.copy()
    if args.wasm:
        environment["IRISU_V86_WASM"] = str(args.wasm.resolve())
    v86 = json.loads(subprocess.check_output([
        "node", str(HERE / "v86_probe.mjs"), str(probe), str(corpus)
    ], cwd=ROOT, text=True, env=environment))
    actual = base64.b64decode(v86.pop("base64"))
    expected = digest(native)
    expected_bytes = native.read_bytes()
    mismatches = []
    for index in range(len(sample)):
        left = expected_bytes[index * 31:(index + 1) * 31]
        right = actual[index * 31:(index + 1) * 31]
        if left != right:
            mismatches.append({"index": index, "input": f"0x{sample[index]:08x}",
                               "native": left.hex(), "v86": right.hex()})
    report = {"schema": 1, "inputs": len(sample), "native": expected, "v86": v86,
              "mismatch_count": len(mismatches), "mismatches": mismatches[:16],
              "exact": expected_bytes == actual}
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["exact"]:
        raise SystemExit(1)
