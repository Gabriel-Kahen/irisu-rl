import assert from "node:assert/strict";
import test from "node:test";

import {
  activatedBlendAlpha, activatedPalette, bonusColorIntervalMs,
  bonusPalette, colorFor, hasActivatedBlend, palette,
} from "../static/colors.mjs";

test("green and orange slots use the requested colors in both states", () => {
  assert.equal(palette[3], "#b227b5");
  assert.equal(activatedPalette[3], "#b227b5");
  assert.equal(palette[4], "#52aba7");
  assert.equal(activatedPalette[4], "#52aba7");
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

test("activated pieces use the original half-strength additive blend", () => {
  assert.equal(activatedBlendAlpha, 128 / 255);
  assert.equal(hasActivatedBlend({kind: "piece", lifecycle: "confirmed"}), true);
  assert.equal(hasActivatedBlend({kind: "piece", lifecycle: "dynamic_fresh"}), true);
  assert.equal(hasActivatedBlend({kind: "piece", lifecycle: "scripted_falling"}), false);
  assert.equal(hasActivatedBlend({kind: "piece", lifecycle: "rotten"}), false);
  assert.equal(hasActivatedBlend({kind: "bonus", lifecycle: "confirmed"}), false);
  assert.equal(hasActivatedBlend({kind: "projectile", lifecycle: "confirmed"}), false);
});
