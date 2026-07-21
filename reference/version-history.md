# Version History and Benchmark Sources

## Authoritative shipped changelog

The Japanese readme inside the pristine v2.03 archive records:

| Version | Date | Shipped note |
|---|---|---|
| 1.00 | 2008-09-27 | Initial release. |
| 1.01 | 2008-10-05 | Changed clear conditions; fixed text and bugs; added the 100,000-point feature. |
| 2.00 | 2009-12-30 | C77 release of IriSu Syndrome Metsu. |
| 2.01 | 2010-01-16 | Fixed replay selection failure for filenames containing Japanese characters. |
| 2.02 | 2010-01-31 | Fixed a progress-reset bug introduced in 2.01. |
| 2.03 | 2010-02-08 | Minor fixes. |

The shipped changelog oddly omits 1.02, but contemporary Vector pages and score tables establish that it existed. Wayback snapshots of Vector identify:

| Version | Archive | Bytes | Historical Vector file page |
|---|---|---:|---|
| 1.00 | `irisu100.zip` | 22,984,232 | `fh463417.html` |
| 1.01 | `irisu101.zip` | 23,430,215 | `fh463938.html` |
| 1.02 | `irisu102.zip` | 23,430,515 | `fh469054.html` |

Known old direct URLs used the form `http://download.vector.co.jp/pack/win95/game/puzzle/drop/irisu10N.zip`. They now return 404, and no archive payload was found in the Wayback Machine. A 2017 [Pastebin directory listing](https://pastebin.com/pKtT6kL6) records an exact-size `irisu102 (Irisu Syndrome).zip`, showing that copies survived privately, but it does not expose the file. A 2012 preservation thread also records a successful temporary re-upload that has since expired.

Treat v2.03 as the sole clone target unless another version is explicitly selected. The [Peercast record table](https://wikiwiki.jp/peca/Peercast%20Record/%E3%81%84%E3%82%8A%E3%81%99%E7%97%87%E5%80%99%E7%BE%A4%21) separates v2+ normal records because mechanics changed substantially. Do not mix old and v2.03 trajectories, distributions, scores, or replay layout as if they came from one environment.

The January 2009 214,453-point replay necessarily predates 2.00 and most likely came from 1.02 based on release dates, but its exact generating binary is not proven. Use it for format research and high-level strategy, not v2.03 golden physics.

## Human performance references

The Peercast table records, among other results:

- 281,359 in v2.03 normal mode, level 100, max chain 12;
- 237,069 in v1.02 normal mode, level 102;
- 361,068 in v2.03 Metsu, level 100, max chain 15;
- 348,508 in Metsu.

The cached two-part Nico recording reaches about 330,000 in v2.03 normal mode. A 2017 community post claims that then-public video records included 510,000 without a glitch and 3,540,000 with a glitch; the post did not link or identify the recordings, so these are discovery leads rather than verified benchmarks.

Speedrun records optimize threshold time rather than maximum score, but their videos are excellent examples of efficient early-game control and provide reachable expert contacts. Current useful sources include:

- [`jako`'s 49-second normal 40k run](https://www.speedrun.com/irisu_syndrome/runs/yw122g2z), with [video](https://youtu.be/A_TClovAoPE); cached locally with metadata;
- [`Jubileus`'s 58:18 100% run](https://www.speedrun.com/irisu_syndrome/runs/y2q6x0wy), with [video](https://www.youtube.com/watch?v=q7vfyXTwbk8);
- the [IriSu Syndrome speedrun board](https://www.speedrun.com/irisu_syndrome), including moderators and additional submitted runs;
- Nico uploader [`kenshin`](https://www.nicovideo.jp/user/1632727), source of the v2.03 330k recording;
- Nico uploader and strategy author [`loveinch`](https://www.nicovideo.jp/user/3204226), source of the 214k recording/replay.

Ask human experts for original `.rpy` files, exact game version/archive hashes, whether fast-forward or glitches were used, and permission to retain/use the data. A video alone cannot recover exact action timing or seed.

## Evaluation implication

Do not define “superhuman” from a single historical maximum. Build a locked v2.03 normal-mode corpus with original replays from several strong players, fixed observation/input rules, and enough runs to estimate a distribution. Report both a reproducible expert-distribution comparison and best-score context; keep glitch-assisted results in a separate category.
