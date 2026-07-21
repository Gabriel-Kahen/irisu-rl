# Synthetic replay fixtures

These fixtures were authored from the independently documented replay bit
layout. They contain no bytes copied from the game or third-party replays and
may be redistributed with the clone.

Each `.hex` file is whitespace-separated hexadecimal. `padded.rpy.hex` covers
the v2.03 52-byte header and boundary bit values. `legacy.rpy.hex` covers the
20-byte historical header. The malformed files deliberately end in partial
records.
