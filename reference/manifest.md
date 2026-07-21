# Reference Material Manifest

Retrieval and inventory date: 2026-07-17. Last experiment update: 2026-07-20.

Raw files listed here are stored in Git-ignored directories.

## Tracked derived evidence

| Item | Workspace path | SHA-256 | Notes |
|---|---|---|---|
| Exact normal score-scale table | `score-scale-binary64.tsv` | `026462999c6ffb03708ed89d68261c9a6e95a1eb3ab3d16072094fe7ba09e677` | Wine-GDB dump of all 99 binary64 stores immediately after v2.03 executable `0x4157be`; derivation and calling-convention proof are in `game-rules-analysis.md`. |
| Box2D numerical golden | `probe/golden/box2d-v2.03-wine11.13.jsonl` | `b73f9c74db48a9695450ab04b76661c0d576030f7f428d0e53b953b4dae077b2` | 1,290 redistributable measurement records from the shipped DLL; `probe/test.sh` reproduces it byte-for-byte across two Wine 11.13 runs. |

## Local game

| Item | Workspace path | SHA-256 | Notes |
|---|---|---|---|
| IriSu 2.03 executable | `game/irisu-v2.03-en/irisu.exe` | `0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255` | Japanese 2.03 engine with English-patched data archives. |
| Box2D DLL | `game/irisu-v2.03-en/data/dll/Box2D.dll` | `34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd` | Shipped physics DLL; important for version/API investigation. |
| DxLib DLL | `game/irisu-v2.03-en/data/dll/DxLib.dll` | `d8ef638a078a8b4d24b53b174ca179623fed3690027d3f4acfe71a7d61c8b5c9` | Shipped RNG oracle; the clean-room model matches all checked outputs for seeds `0`, `1`, `0x12345678`, and `0x3fffffff`. |
| Plaintext mechanics config | `game/irisu-v2.03-en/data/doc/irisu.ini` | `1e29431fe8209c25784d4741f7972737561281169bbb5a56f62e3e0f0b63de35` | Shipped CP932 config; normalized extraction and caveats in `mechanics-evidence.md`. |
| Preserved local replay | `replays/raw/local/new-2026-07-17.rpy` | `08a58a1c5e55e383603aa886acb76fedaf41032195a5c414c855b1f139fe9023` | The only `.rpy` initially found on the machine. |
| Original 2.03 archive | `archives/irisu203.zip` | `3fcc57ff0bbc4af1d262813173956ecc1563f26a39f1e59119643cf07400f2a5` | Pristine base archive; passes `unzip -t`. |
| English 2.03 patch | `archives/IrisuSyndromeENv2.03.zip` | `ed3f0653adc5bc548381c10c266a9a9a371dbf968d125cd32e7f5c529016389c` | Contains replacement `dat.dxa`, `img.dxa`, and patch readme; passes `unzip -t`. |

The source installation remains at `/home/gabe/Games/Irisu Syndrome`. The workspace copy is separate and mutable.

## External replays

| Item | Workspace path | SHA-256 | Provenance |
|---|---|---|---|
| 214,453-point normal replay | `replays/raw/internet/irisu_00214453_20090104_005708_5.rpy` | `037b67cb889027cc5e1dabdf53586fce581614cab1e9a864fad631a661403672` | Attached to <https://w.atwiki.jp/loveinch/pages/50.html>; direct source <https://img.atwiki.jp/loveinch/attach/50/47/irisu_00214453_20090104_005708_5.rpy>. No redistribution license found. |
| 41,449-point normal replay | `replays/raw/internet/irisu_00041449_20100725_182435_7.rpy` | `73bf5b5d4a478c9bf73b62a6df98f16a01fde2cf97eb751438cd0ae857e3362d` | Original community replay uploader: <https://u7.getuploader.com/irisu/download/4>. Download-page MD5 `d262470b9198b1018e718b5af6b3f7f2` matches. No redistribution license found. |
| 43,791-point normal replay | `replays/raw/internet/irisu_00043791_20111118_222006_26.rpy` | `bcb4567326ee35e51f320c55233bf74447fc364e94bf8854117e474c11ad55ec` | Original community replay uploader: <https://u7.getuploader.com/irisu/download/9>; uploader comment says level 32. Download-page MD5 `c46659f617f8bfb806139ebee1860c5a` matches. No redistribution license found. |
| 40-point normal replay | `replays/raw/internet/irisu_00000040_20190417_184328_1.rpy` | `ba9a75ae2b802787cd0fb7c6e3186753b18013cf45fd6db827c6ea24e888bdb9` | Extracted from the public research copy at <https://archive.org/details/irisu-syndrome>, package <https://archive.org/download/irisu-syndrome/Irisu%20Syndrome%20%28decoded%29%20%21Spoilers%21.zip>. No redistribution license found. |
| 56-point normal replay | `replays/raw/internet/irisu_00000056_20190417_184425_2.rpy` | `410a8a1cec3d5bb6688c41f02934473b7b8e26e00b727aff2aad174d45332321` | Same Internet Archive package; its `replay/new.rpy` is a duplicate of this file. No redistribution license found. |

The 214,453-point file is 172,608 bytes. Its first five little-endian `int32` values are seed `4,765,293`, level `34`, score `214,453`, highest chain `42`, and mode `0` (normal). It predates v2.03 and begins 4-byte frame records immediately after the 20-byte legacy header.

The four padded files have now been replayed under the bundled v2.03 executable.
Only the 41,449 header matches that executable's outcome. The other observed
v2.03 results are 32 (header 40), 88 at replay exhaustion (header 56), and
1,794 (header 43,791). Exact-forward clone comparisons match every observed
score event on all four. The original generating builds/runtime conditions for
the mismatching headers remain unknown; a padded layout and mode field are not
outcome provenance. Full event oracles live under
`runs/replay-{41449,43791}-full-event-gdb-20260720-*` and
`runs/replay-header{40,56}-full-event-gdb-20260720-001/`.

## 2026-07-19/20 local diagnostics

| Item | Workspace path | SHA-256 | Status |
|---|---|---|---|
| Seed-123 reset replay | `captures/probe-reset1-20260719-001/probe-z-reset1.rpy` | `c42d5e6c6794f6287477af7b3a0fb3d195a33133be1c1fcb671f9ad9b6b8d04c` | One padded record. Fresh-process images visually corroborate the corrected 10-rotten/10-scripted reset layout; the bundle is missing required golden metadata and measurements. |
| Seed-41 score parity | `captures/seed41-score-parity-20260720-001/input.rpy` | `1ce501febe8f3f6291e4b82736542179bd9808e412d38e0e1fb1c92d05797657` | 520 padded mode-0 records. A fresh original process and the corrected clone both score `+8,+8` at tick 304 and finish at 16. The [adjudication](./seed-41-adjudication.md) resolves the mechanics discrepancy but keeps the incomplete capture non-golden. |
| Reproducible seed-41 input | `python3 tools/generate-controlled-rpy.py OUTPUT.rpy --preset score-seed41-parity --library build/libirisu_clone.so` | `1ce501febe8f3f6291e4b82736542179bd9808e412d38e0e1fb1c92d05797657` | Regenerates the byte-identical original-observed input and checks the two clone score events and final 16. |

The older seed-910 `probe-d-orb.rpy` capture predates the corrected reset/x87
path and is not a current orb candidate. None of the July 19 capture directories
contains the complete metadata, actions, measurements, notes, hashes, and
adjudication required by `golden/README.md`; `golden/manifest.json` remains
empty.

Online replay data therefore exists, but the open high-quality corpus remains
thin: four padded action traces targetable by v2.03, only one preserving its
claimed long outcome there, plus one pre-v2 legacy trace. Controlled
self-generated v2.03 replays and user-supplied expert traces remain essential.

The original community uploader was created specifically for replay exchange, as documented at <https://w.atwiki.jp/irisu_syndrome/pages/27.html>. It also lists a password-locked 132,117-point file at <https://u7.getuploader.com/irisu/download/30>; do not attempt to bypass the password. A 2008 archived predecessor index names a v1.01 100,000-point replay and three other replay ZIPs, but the binary attachments were not archived and have been deleted from the live service.

Replay-format research is available at <https://github.com/hoangcaominh/irisu-rpy-struct>. The repository contains a Kaitai schema but no license, so use it as a research lead and independently implement and validate the format. The original developer confirms the seed-plus-input design and describes cross-machine divergence at <https://wtetsu.hatenablog.com/entry/20080927/1222495777>. The executable-derived format is recorded in `binary-analysis.md`.

## Gameplay-video candidates

| Description | Source | Use |
|---|---|---|
| Normal mode, 100,000 points | <https://www.youtube.com/watch?v=BDhIESUV4rQ> | Downloaded as `videos/raw/BDhIESUV4rQ-10.mkv`; 640x480, 30 fps, 839.621 s; SHA-256 `7f6aae1c846dd001f9e08fe763cbd13e96efa0d603824fab05940ba87e084356`. |
| Normal mode, level 100 | <https://www.youtube.com/watch?v=CpwG9DBjhzc> | Downloaded as `videos/raw/CpwG9DBjhzc-100.mp4`; 480x360, 30 fps, 394.182 s; SHA-256 `2fa5c3af306e3b31921894ff93a304fe2d7e6cfa08774d2194da82717b4edec0`. |
| Ordinary normal-mode play | <https://www.youtube.com/watch?v=q0kdegmqDkc> | Downloaded as `videos/raw/q0kdegmqDkc-_.mkv`; 640x480, 30 fps, 659.261 s; SHA-256 `e233e2c3373d8fce9df7941f5192fec93fd5113fe314d2d986902d149de02cf7`. |
| v2.03 level 100, 330k, first half | <https://www.nicovideo.jp/watch/sm10477778> | Downloaded as `videos/raw/sm10477778-Ver2.03_Lv100_33.mkv`; 512x384, 30 fps, 900.098 s; SHA-256 `72efd846651e80377223302f57fa0a5d2cab0fba7b9d17a27faa50569d0383bd`. |
| v2.03 level 100, 330k, second half | <https://www.nicovideo.jp/watch/sm10477833> | Downloaded as `videos/raw/sm10477833-Ver2.03_Lv100_33.mkv`; 512x384, 30 fps, 947.978 s; SHA-256 `4412c11cf0b5099f8c96eea7805f0d74b1212f4378ee9d73c21070de30baf60c`. |
| 214k large-chain strategy | <https://www.nicovideo.jp/watch/sm5809036> | Downloaded as `videos/raw/sm5809036-20.mkv`; 512x384, 15 fps, 909.642 s; SHA-256 `b34a53369db00b1019cf4dc1a355f394f1f1eac2e8ba709a551fc5d193e426c7`. This corresponds to the external replay. |
| Normal-mode 40k RTA in 49 seconds | <https://youtu.be/A_TClovAoPE> | Downloaded under `videos/raw/A_TClovAoPE-*`; 640x480, 30 fps, 64.701 s; SHA-256 `672730c1276d9f1341c7d3cc6fdd4d8b266acd18530cda20d9073aefd55e2ea7`. Useful for expert early-game action cadence, not maximum-score benchmarking. |
| 100% speedrun in 58:18 | <https://www.youtube.com/watch?v=q7vfyXTwbk8> | Downloaded under `videos/raw/q7vfyXTwbk8-*`; 1280x720, 30 fps, 3,526.501 s; SHA-256 `714cdd215333e440b441eee0f3f8b2e03edcf41b711d83c4378b14b8807341d9`. Contains several normal-mode rounds but optimizes overall progression time. |
| 45,000-point normal mode plus Metsu recordings | <https://www.bilibili.com/video/BV1J4411N7uH/> | Additional footage, but the page explicitly states that redistribution is unauthorized; retain the link only unless permission is obtained. |
| Speedrun.com leaderboards | <https://www.speedrun.com/irisu_syndrome> | Current run videos and expert contacts; runs optimize time rather than maximum score. |

The original author explicitly permits gameplay-video streaming on the official manual page: <https://katatema.main.jp/irisu/manual.html>. That statement does not grant blanket permission to redistribute other creators' uploaded files. Keep downloaded research copies local and untracked, and preserve creator/source metadata.

Each downloaded video has a neighboring `.info.json` metadata file and PNG thumbnail in the ignored directory.

## Additional technical leads

- The shipped DLL is the primary physics artifact. Binary API/layout evidence identifies the Box2D 1.4.3 lineage, and the clone uses the official zlib SourceForge SVN r58 source behind a measured adapter. ABI/layout evidence and discriminating numerical regressions match the shipped wrapper; this does not prove that the PE DLL used identical compiler options or guarantee indefinite chaotic trajectory identity. The earlier 2.0.1 timing guess is disproven; see `binary-analysis.md` and `probe/observations.md`.
- `Box2D.dll` exports only 16 named stdcall functions: `b2d_init`, `b2d_dispose`, `b2d_step`, `b2d_create_box`, `b2d_create_circle`, `b2d_create_triangle`, `b2d_destroy_body`, `b2d_get_contact`, `b2d_get_x`, `b2d_get_y`, `b2d_get_r`, `b2d_get_v`, `b2d_set_position`, `b2d_set_v`, `b2d_set_user_data`, and `b2d_test`. The decorated names expose argument-byte counts. The completed Wine probe resolves all 16 and records the wrapper's numerical behavior in the tracked golden trace.
- The disproven Box2D 2.0.1 candidate archive remains cached at `archives/toolchains/Box2D_v2.0.1.zip`, passes `unzip -t`, and has SHA-256 `62857048aa089b558561074154430883cee491eedd71247f75f488cba859e21f`. Retain it only as negative evidence.
- Box2DJS 0.1.0 is cached from <https://sourceforge.net/projects/box2d-js/files/latest/download> as `archives/toolchains/box2d-js_0.1.0.zip`, passes `unzip -t`, and has SHA-256 `d8f77eb789bdc37773c2b5ef724a85559f8460e945caa04ebbb5878f22e11fd0`. Its official site states that it was mechanically converted from Box2DFlashAS3 1.4.3.1, making it a useful readable semantic reference for the legacy API. It is not proof of the exact C++ build or floating-point behavior.
- A historical `box2d` 1.4.3 Ruby gem is cached from <https://rubygems.org/gems/box2d/versions/1.4.3> as `archives/toolchains/box2d-1.4.3.gem`, SHA-256 `4b1da6bb063add46c3e99ad7bd442a73b7509759ba5facf9b92d9a74e97defc0`. It contains old compiled extension binaries but no source and is a secondary binary lead only.
- DMD 1.030 is cached from <https://ftp.digitalmars.com/dmd.1.030.zip> at `archives/toolchains/dmd.1.030.zip`, passes `unzip -t`, and has SHA-256 `2f7fe244c97690dcaf0e347aa7a1ec2c7c49a2e4ec230d4c7dd5133393549773`.
- Official DX Library 3.24f source and VC packages are cached from <https://dxlib.xsrv.jp/dxdload.html> as `archives/toolchains/DxLibMake3_24f.zip` (SHA-256 `114160dc16112600521e14b255e3ea9b3f6caabaee94188e9a4f895316ceef26`) and `DxLib_VC3_24f.zip` (SHA-256 `82fde04562e4a5128710bb95ad8a73dca708a9e080439acc14d8e1ad5a095675`). The extracted `DxaDecode.exe` has SHA-256 `63533009a62138348ca1598bac91493fbe63e084c738cd5d838e71986e7f4a59` and successfully decoded v2.03 `dat.dxa` with key `shine` under Wine. Decoded contents remain ignored.
- Official v2.03 download listing: <https://www.vector.co.jp/download/file/win95/game/fh504362.html>.
- High-score technique pages: <https://w.atwiki.jp/loveinch/pages/51.html>, <https://w.atwiki.jp/loveinch/pages/52.html>, and <https://w.atwiki.jp/loveinch/pages/53.html>.
- <https://github.com/JosephJeongs/IrisuSyndrome> was downloaded at commit `c36b98fde87b277b4a6f934ebf4194b4fadf4c1b` under `archives/source-leads/` and inspected. Despite its name, it is only the stock CMake/SFML template with an empty window loop and contains no IriSu mechanics. It is not a clone or evidence source. The ignored local copy is retained only so later agents do not repeat this lead.
- The similarly named <https://github.com/JosephJeongs/Irisu-Syndrome> was downloaded at commit `de1e0d77df8615fda8d41484a56479343a1efc7a`. It contains about 205 lines of C++ around an SFML/modern-Box2D toy: a floor, upward square bullets, and an unused contact listener. It does not implement colored pieces, spawning, gauge, chains, scoring, or the original fixed step; its constants conflict with the shipped configuration. Treat it as an incomplete visual experiment, not reference behavior. No project-specific license was found.
- The existing black-box attempt <https://github.com/Gabriel-Kahen/irisu_blackbox> was downloaded at commit `b702ddcecb86475e6e12c799fc22fe9e4e3a2de8`. It includes a Windows capture/click Gymnasium wrapper, HUD readers, action-grid code, tests, and a RecurrentPPO baseline. Its old screen coordinates and PyAutoGUI control path are machine-specific and should not replace the same-session protocol, but its perception, health/score extraction, test cases, and failure history are useful inputs when building the new reference harness. No repository license was found.
- The replay schema repository <https://github.com/hoangcaominh/irisu-rpy-struct> is cached at commit `88eebb838fe4bc409c3a6aa50588c93c862bd10a`. It has no license; independently reimplement any useful format knowledge.
