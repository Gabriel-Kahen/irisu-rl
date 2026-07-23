export const palette = [
  "#861f00", "#0005a4", "#9a9000", "#b227b5",
  "#52aba7", "#ae6311", "#1b747a", "#92335f",
];

export const activatedPalette = [
  "#e44717", "#2945ff", "#eee116", "#b227b5",
  "#52aba7", "#ef9c27", "#35bdc4", "#e35b98",
];

export const bonusPalette = [
  activatedPalette[0], activatedPalette[1], activatedPalette[2],
  activatedPalette[3], activatedPalette[4],
];

export const bonusColorIntervalMs = 400;
export const activatedBlendAlpha = 128 / 255;

function isActivatedBlock(body) {
  return body.kind === "piece" &&
    (body.lifecycle === "dynamic_fresh" || body.lifecycle === "confirmed");
}

export function hasActivatedBlend(body) {
  return isActivatedBlock(body);
}

export function colorFor(body, now = 0) {
  if (body.kind === "projectile") return "#d9dcda";
  if (body.kind === "bonus") {
    return bonusPalette[Math.floor(now / bonusColorIntervalMs) % bonusPalette.length];
  }
  const index = ((body.color % palette.length) + palette.length) % palette.length;
  return (isActivatedBlock(body) ? activatedPalette : palette)[index];
}
