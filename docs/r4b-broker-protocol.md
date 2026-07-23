# `irisu-live-broker-v1`

This is the external broker ABI consumed by
`irisu_rl.original_game.live_provider`. The broker is a long-lived subprocess
with one canonical JSON object per UTF-8 line on stdin/stdout. Lines and decoded
frames are bounded; duplicate keys, non-finite numbers, unknown fields,
sequence disagreement, partial lines, or extra fields are fatal.

Request envelope:

```json
{"method":"handshake","params":{},"protocol":"irisu-live-broker-v1","sequence":1}
```

Response envelope:

```json
{"error":null,"ok":true,"protocol":"irisu-live-broker-v1","result":{},"sequence":1}
```

A failed result has `result:null` and
`error:{"code":"safe_identifier","detail":"bounded text"}`. The client never
uses a shell, sends secrets in argv, or accepts a broker executable whose
already-open inode disagrees with the preregistered SHA-256.

## Handshake

`handshake {}` returns exactly:

```json
{
  "broker_instance_id": "opaque-instance-generation",
  "broker_implementation_sha256": "<broker executable sha256>",
  "capabilities": {
    "atomic_click_only": false,
    "automatic_release_deadline": true,
    "explicit_button_down": true,
    "explicit_button_up": true,
    "neutralizes_on_claim_end_or_expiry": true,
    "release_all_buttons": true
  },
  "clock_domain": "linux.clock_monotonic",
  "input_backend": "hyprland_native_targeted_edges_v1",
  "monotonic_ns": 123
}
```

The broker instance changes after every restart. Safe input is rejected unless
all six capabilities and the audited backend name agree. The handshake
timestamp must fall within the locally bracketed call plus the small configured
clock-skew allowance.

`session_safety {}` returns:

```json
{"safety":{"detail":"","exact_background_capture":true,"exact_window_claims":true,"targeted_input_safe":true}}
```

It is queried repeatedly. Session lock, pointer lock/constraint/grab, drag and
drop, held physical buttons, compositor ABI drift, ambiguous input routing, or
lost native support must set `targeted_input_safe:false`.

## Target discovery and identity

`discover_target` receives a secret launch nonce plus expected executable,
complete runtime, and Wine-prefix hashes. It must inspect process
lineage/environment and return exactly one matching disposable process:

```json
{
  "broker_instance": "opaque-instance-generation",
  "match_count": 1,
  "target": {
    "executable_sha256": "<canonical irisu.exe>",
    "identity": {"address":"0x...","capture_id":"..."},
    "launch_nonce_sha256": "<sha256 of supplied nonce>",
    "process_id": 1234,
    "process_start_ticks": 5678,
    "runtime_sha256": "<complete disposable-run identity>",
    "wine_executable_sha256": "<preregistered Wine executable>",
    "wine_prefix_sha256": "<complete attested Wine prefix>"
  }
}
```

Title/class matching is insufficient. PID start ticks prevent reuse; capture ID
prevents a recreated window at the same address; the nonce prevents selecting a
different IriSu process; hashes reject the preserved tree, trace proxy, and
mutated data or prefix state.

## Claims

`claim` receives `identity` and `lease_seconds`. `renew` additionally receives
the opaque `fencing_token`; it must strictly extend expiry while preserving
every other field. Both return:

```json
{
  "broker_instance": "opaque-instance-generation",
  "expires_ns": 123456789,
  "fencing_token": "<opaque secret>",
  "generation": 7,
  "identity": {"address":"0x...","capture_id":"..."},
  "target": {"...":"same target descriptor returned by discovery"}
}
```

Every claim-scoped operation must fence on token, generation, broker instance,
and exact target. `release_claim` returns `{"released":true}` and is idempotent
for the owning token. Claim release, expiry, and client disconnect must
neutralize all held buttons inside the authoritative injector before reporting
completion.

## Capture and cursor

`capture` receives identity/token and returns one completed exact-window frame:

```json
{
  "canonical_pixel_sha256":"<SHA-256 of decoded tightly packed client pixels>",
  "color_format":"bgra8",
  "completion_ns":130,
  "identity":{"address":"0x...","capture_id":"..."},
  "pixel_height":484,
  "pixel_width":644,
  "pixels_base64":"<base64 of tightly packed BGRA bytes>",
  "presentation_ns":120,
  "request_ns":100,
  "source_sequence":42,
  "start_ns":110,
  "window_bounds":{"height":484,"width":644,"x":0,"y":0}
}
```

All timestamps use Linux `CLOCK_MONOTONIC` and must fit inside the client's
local call bracket. Production qualification requires a real presentation
timestamp and monotonically increasing source sequence from a continuous
capture route; one-shot `grim` timestamps cannot establish gameplay cadence.
`color_format` is exactly `bgra8`; the decoded byte length must equal
`pixel_width × pixel_height × 4`. The client recomputes
`canonical_pixel_sha256` over those tightly packed bytes and rejects any
mismatch, so neither an encoder choice nor unbound broker metadata can change
duplicate-frame classification. Compressed image formats are not accepted at
this safety boundary.
`cursor` returns `{"observed_ns":123,"x":10.0,"y":20.0}` in window-local
coordinates under the same claim.

## Explicit edges

`button_down` receives identity/token, `operation_id`, left/right button,
window-local x/y, an absolute `latest_injection_ns` freshness deadline, and an
absolute `release_deadline_ns`. The authoritative injector must atomically
revalidate session safety and reject the operation if injection would occur
after the freshness deadline; it must schedule neutralization before
acknowledging. `button_up`
receives the same fields except the deadline; it remains bound to the accepted
deadline from the matching down. `release_all` receives identity/token and
operation ID.

Every edge returns:

```json
{
  "accepted_release_deadline_ns":123456,
  "acknowledged":true,
  "acknowledged_ns":120,
  "broker_instance":"opaque-instance-generation",
  "button":"left",
  "button_state":"down",
  "claim_generation":7,
  "clock_domain":"linux.clock_monotonic",
  "detail":"",
  "identity":{"address":"0x...","capture_id":"..."},
  "injected_ns":115,
  "operation_id":9
}
```

Up echoes the same accepted deadline with `button_state:"up"`. Release-all uses
`button:"all"`, `button_state:"neutral"`, and a null deadline. Operation IDs
must increase, acknowledgements must describe the actual routed target and
transition, and up/return must precede both the accepted deadline and lease
expiry.

No compatibility translation from atomic click, drag, global input, or two
caller-timed calls is permitted. A build is eligible only after native tests
prove deadline, release, expiry, disconnect, and caller-death neutralization
without the Python process continuing to run.
