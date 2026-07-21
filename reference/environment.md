# Reference Environment Snapshot

Recorded 2026-07-17. Re-record these values inside every experiment bundle; this file is orientation, not a substitute for run metadata.

## Game and runtime

| Component | Value |
|---|---|
| Game | IriSu Syndrome v2.03, English-patched data |
| Executable SHA-256 | `0636d3e44439d88807d0c00aeb1bb072316c69fc13a21f79d67e53affad28255` |
| `data/dat.dxa` SHA-256 | `b36ef6864bf2d0e626d5087edb5b571ef548ebd5dde9fbc9b87f7b4ac3e89d4a` |
| `data/img.dxa` SHA-256 | `7ffdf24de7d9465296e14cbee086ed04927c5e8a7e442d6be597984a71e03c50` |
| `data/snd.dxa` SHA-256 | `65617d2e2692bb5481e68005745cea8146a5021e78664b336325bd7ab2d4c51d` |
| `data/doc/irisu.ini` SHA-256 | `1e29431fe8209c25784d4741f7972737561281169bbb5a56f62e3e0f0b63de35` |
| `data/dll/Box2D.dll` SHA-256 | `34f1387cbe51fb09a3e7cd868236ed3c0b73be2f70fcf5b09ae6c48187d14fcd` |
| Wine | Bundled Wine 11.13 |
| Wine executable | `/home/gabe/.local/share/irisu-syndrome/runtime/bin/wine` |
| Wine prefix | `/home/gabe/.local/share/irisu-syndrome/prefix` |
| Launch wrapper | `tools/launch-reference-game.sh` |

The generic `wine` command is not on `PATH`; use the launcher or set `IRISU_WINE_BIN` and `IRISU_WINEPREFIX` explicitly.

## Desktop and input

| Component | Value |
|---|---|
| Kernel at snapshot | Linux 7.1.2-3-cachyos x86_64 |
| Hyprland | 0.55.4, commit `a0136d8c04687bb36eb8a28eb9d1ff92aea99704` |
| Display | Dell S2425HS, 1920x1080, 100 Hz, scale 1.0, VRR off |
| Game window path | Wine/XWayland |
| Native target-pointer plugin | `same-session-target-pointer` 0.1.1, built and loaded |
| Plugin SHA-256 | `5573489bde6e80ff924d195af64787121babebec018ef4162002e9518c2f218b` |

At verification, `session_status` reported exact background capture, targeted background shortcuts, targeted Wayland and XWayland pointer actions, and native input safety all available. The plugin is tied to the compositor ABI and its loaded state does not survive every compositor restart. The same-session broker owns rebuilding/loading; always call `session_status` before use.

## Configuration evidence

`data/doc/irisu.ini` is a 3,041-byte CP932 plaintext configuration shipped with this copy. It contains high-value field, fixture, projectile, gauge, and fixed-step constants. See `mechanics-evidence.md` for a normalized extraction and the distinction between configured values and behavior verified by experiment.
