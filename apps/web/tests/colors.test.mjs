import assert from "node:assert/strict";
import test from "node:test";

import {
  activatedBlockBlurFor, activatedPalette, bonusColorIntervalMs,
  bonusPalette, colorFor, palette,
} from "../static/colors.mjs";

test("green and orange slots use the requested colors in both states", () => {
  assert.equal(palette[3], "#b227b5");
  assert.equal(activatedPalette[3], "#b227b5");
  assert.equal(palette[5], "#52aba7");
  assert.equal(activatedPalette[5], "#52aba7");
});

test("the bonus ball cycles through exactly five block colors", () => {
  assert.deepEqual(bonusPalette, [
    "#e44717", "#2945ff", "#eee116", "#b227b5", "#52aba7",
  ]);
  const bonus = {kind: "bonus"};
  assert.deepEqual(
    [0, 1, 2, 3, 4, 5].map((step) => colorFor(bonus, step * bonusColorIntervalMs)),
    [...bonusPalette, bonusPalette[0]],
  );
});

test("only confirmed pieces receive the activated-block blur", () => {
  assert.equal(activatedBlockBlurFor({kind: "piece", lifecycle: "confirmed"}), 10);
  assert.equal(activatedBlockBlurFor({kind: "piece", lifecycle: "dynamic_fresh"}), 0);
  assert.equal(activatedBlockBlurFor({kind: "piece", lifecycle: "scripted_falling"}), 0);
  assert.equal(activatedBlockBlurFor({kind: "piece", lifecycle: "rotten"}), 0);
  assert.equal(activatedBlockBlurFor({kind: "bonus", lifecycle: "confirmed"}), 0);
  assert.equal(activatedBlockBlurFor({kind: "projectile", lifecycle: "confirmed"}), 0);
});
