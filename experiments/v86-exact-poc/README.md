# v86 exact-worker browser proof of concept

This experiment runs the existing i386 `irisu-exact-worker` entirely in a
browser Web Worker. v86 boots a small Buildroot kernel, exposes the worker and
its libraries over virtio-9p, and carries the existing binary IPC protocol over
a raw emulated serial port.

It is deliberately isolated from the production web build. Stock v86 does not
implement bit-exact x87 transcendental instructions, so successful execution
here proves browser integration and gives a performance baseline; it does not
by itself prove gameplay parity.

Prepare pinned dependencies and local exact artifacts:

```sh
./prepare.sh
```

Then serve this directory and open the printed URL:

```sh
./serve.sh
```

The page performs a Hello, Reset, Observe benchmark, and a short Step
benchmark. It also checks the complete final Step response against the SHA-256
produced by the native exact worker for the same seed and 20 no-op inputs.
Generated/downloaded files live in `runtime/` and are ignored.

`prepare.sh` pins v86, the project-owned Linux guest, its BIOS images, and
Debian 13 i386 userspace packages by SHA-256. It builds the guest from
`../../apps/web/guest` by default; `IRISU_GUEST_BZIMAGE` accepts a prebuilt
copy with the required hash. Debian userspace is used because the local Arch i386
loader requires ISA levels above v86's advertised Pentium III-compatible CPU.
The staged launcher has only its over-declared `.note.gnu.property` removed;
its executable code and the MSVC9 physics host remain unchanged.

The same guest path can be checked without a browser UI, which is useful in CI:

```sh
node ./node-smoke.mjs
```

The defaults use the exact build already present in this workspace. Override
them when needed:

```sh
./prepare.sh /path/to/irisu-exact-worker /path/to/exact-host.so /path/to/irisu-exact-replay
```

## Replay-corpus parity

The corpus evaluator boots v86 once and runs each complete eligible padded
replay inside the guest. Replay bytes and result JSON cross virtio-9p; serial
is used only for completion markers, never for per-tick RPC.

```sh
./prepare.sh
python3 ./evaluate-corpus.py --output /tmp/v86-exact-corpus.json
```

For every replay the report records native and v86 output hashes, terminal
fields, compact timeline hashes, the first differing scalar or event row, the
observed v2.03 oracle comparison, and native/v86 ticks per second. The current
corpus has four eligible normal-mode padded replays, including the 47,019-tick
run. Stock v86's x87 result is evidence, not assumed parity.

The 2026-08-09 gate artifact is
`../../benchmarks/results/v86-exact-replay-corpus-2026-08-09.json`. It passed
all four native raw-output comparisons and all four observed v2.03 scoring
oracles. One boot plus 57,921 executed ticks took 131.6 seconds; the 47,019-tick
replay ran at 495.0 ticks/second.

## Interactive replay transport

`interactive-replay.mjs` sends one opcode-3 request per replay tick, compares
every complete transition response byte-for-byte with a native exact worker,
and reports latency and body-density percentiles:

```sh
node ./interactive-replay.mjs --transport serial --output /tmp/uart.json
node ./interactive-replay.mjs --transport virtio --output /tmp/virtio.json
```

On the 47,019-tick replay both transports produced the same response-stream
SHA-256, `45a452c215ceebb63788ed616891dcc2e69b0a4c97ccbc0c51a12c32d8834c52`.
UART sustained 199.0 ticks/second; `/dev/hvc0` virtio-console sustained 365.8
ticks/second. The checked result artifacts are in `../../benchmarks/results/`.
`virtio-console-smoke.mjs` is the bounded binary round-trip prerequisite for
the virtio path.

### Final production guest kernel

The final project-owned kernel
`389fb6e37c9f9f101232ad68b7177bced98caee9f7a531e99ea00b836833ea33`
passed all four corpus replays and the full 47,019-tick UART interactive gate.
UART sustained 232.9 ticks/second with seven responses above the 16.67 ms
deadline. This kernel does not expose `/dev/hvc0`, so virtio-console is recorded
as unavailable rather than tested; the deployed browser worker uses UART.
