/**
 * ONE tooth colour table for the whole application.
 *
 * THE ARCHITECTURE THIS ENFORCES:
 *
 *     segmentation :  vertex/face -> tooth id (FDI)     a provider's job
 *     display      :  tooth id    -> colour             this file, only
 *
 * WHAT WAS WRONG. The palette was fifteen hex strings indexed by POSITION IN
 * AN ARRAY of the 32 permanent teeth, `PALETTE[i % PALETTE.length]`. Three
 * separate defects fell out of that one line:
 *
 *   1. SEVENTEEN OF THE THIRTY-TWO TEETH SHARED A COLOUR WITH ANOTHER TOOTH.
 *      FDI 11 and 31 were both #e6194b, 12 and 32 both #3cb44b, and so on for
 *      every pair sixteen apart - so an upper right central incisor and a
 *      lower left central incisor were indistinguishable, which is the single
 *      most important distinction on the screen.
 *   2. The colour depended on the tooth's INDEX IN THAT LITERAL, not on its
 *      FDI number. Reorder the array, add the primary dentition, or have a
 *      provider return a set of teeth in a different order, and every colour
 *      moves.
 *   3. Nothing else could use it. The constant lived in App.jsx beside the UI,
 *      so a legend, an inspector or a second viewport would each have grown
 *      their own copy.
 *
 * SO THE COLOUR IS A PURE FUNCTION OF THE FDI NUMBER. Same tooth, same
 * colour - across a refresh, across a re-segmentation, across a change of
 * segmentation provider, and across any ordering anything returns teeth in.
 * There is no state, no array index and no module-level mutation anywhere in
 * this file.
 *
 * HOW THE 32 COLOURS ARE SEPARATED, and why it is arithmetic rather than 32
 * hand-picked hex strings. Hand-picking 32 mutually distinguishable colours
 * is a job people get wrong - the fifteen above contain #fffac8 and #ffe119,
 * which differ by very little. Instead:
 *
 *   - the 32 teeth take the 32 evenly spaced hues, 11.25 degrees apart, so
 *     the whole circle is used exactly once with no bunching;
 *   - consecutive teeth take hue slots 11 apart (11 is coprime with 32, so
 *     every slot is still hit exactly once), which puts 123.75 degrees
 *     between neighbours - FDI 11 and 12 are never similar;
 *   - lightness cycles over four levels by tooth index. Because 11 * 3 = 33
 *     = 1 (mod 32), the two teeth that land on ADJACENT hues always differ by
 *     3 in that index, so they always land in different lightness buckets.
 *     The one place hue separation is weakest is exactly where lightness
 *     separation is guaranteed.
 *
 * That is a claim with a number behind it: `verify-palette.mjs` measures the
 * minimum pairwise CIE L*a*b* distance over all 496 pairs and over every
 * tooth against the gingiva, and fails below a floor.
 */

/** The permanent dentition, FDI. Quadrants 1-4, teeth 1-8. */
export const FDI_TEETH = Object.freeze([
  11, 12, 13, 14, 15, 16, 17, 18,
  21, 22, 23, 24, 25, 26, 27, 28,
  31, 32, 33, 34, 35, 36, 37, 38,
  41, 42, 43, 44, 45, 46, 47, 48,
]);

/** Gingiva. Deliberately desaturated pink: no tooth colour is near it, and a
 *  tooth is never this unsaturated, so "unsegmented tissue" reads instantly. */
export const GINGIVA_HEX = "#d9a7a0";

/** Unsegmented / unknown tooth id. Grey, so it cannot be mistaken for a
 *  tooth that was identified. NOT_CHECKED is not CLEAR - CLAUDE.md s.14. */
export const UNKNOWN_HEX = "#8b93a0";

/** The selection accent. Applied as a TREATMENT, never as a replacement - see
 *  `selectionTreatment`. */
export const SELECTION_HEX = "#3fa9ff";

const HUE_SLOTS = 32;
const HUE_STRIDE = 11;          // coprime with 32

// CHOSEN BY SEARCH, NOT BY TASTE, and the first attempt failed its own check.
// Cycling lightness over [0.46, 0.68, 0.57, 0.77] against saturations around
// 0.6 produced a band of pale, washed-out colours: the worst pair (FDI 42 and
// 48) measured dE 11.83, and FDI 14 landed dE 4.09 from the GINGIVA - close
// enough that an unsegmented region and an upper right first premolar read as
// the same thing. That is the defect this file exists to remove, reproduced
// by a second route.
//
// These four lightnesses at full chroma are the maximum of the worst-case
// separation over a search of 15 hue strides x 15 lightness cycles x 9
// saturation cycles, scored on minimum pairwise CIE L*a*b* distance including
// the gingiva. Measured result, pinned by verify-palette.mjs:
//
//     minimum pairwise dE over all 496 pairs      15.98
//     minimum dE from any tooth to the gingiva    52.55
//
// FULL CHROMA IS DELIBERATE HERE and is not in tension with a restrained UI.
// This is the identification layer - the map that says WHICH TOOTH - and it
// has to separate 32 categories at a glance on a dark viewport. The restraint
// belongs to the interface chrome around it, which uses one accent.
const LIGHTNESS = [0.30, 0.50, 0.40, 0.62];
const SATURATION = [1.0, 1.0, 1.0, 1.0];

/** The viewport clear colour. Tooth colours are checked against it, because
 *  two colours can be far apart from each other and both invisible. */
export const VIEWPORT_BACKGROUND_HEX = "#0d0f12";

/** 0-31 for a permanent tooth, or -1. Quadrant major, tooth minor. */
export function fdiIndex(fdi) {
  const n = Number(fdi);
  if (!Number.isInteger(n)) return -1;
  const q = Math.floor(n / 10);
  const t = n % 10;
  if (q < 1 || q > 4 || t < 1 || t > 8) return -1;
  return (q - 1) * 8 + (t - 1);
}

function hslToHex(h, s, l) {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = ((h % 360) + 360) % 360 / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  const [r1, g1, b1] =
    hp < 1 ? [c, x, 0] : hp < 2 ? [x, c, 0] : hp < 3 ? [0, c, x]
    : hp < 4 ? [0, x, c] : hp < 5 ? [x, 0, c] : [c, 0, x];
  const m = l - c / 2;
  const to = (v) => Math.round(Math.min(1, Math.max(0, v + m)) * 255)
    .toString(16).padStart(2, "0");
  return `#${to(r1)}${to(g1)}${to(b1)}`;
}

/**
 * The colour for one FDI number. A PURE FUNCTION - no table lookup, no index,
 * no state. An unknown or malformed id returns `UNKNOWN_HEX` rather than
 * throwing or picking something arbitrary, because an unlabelled region must
 * never render as a confident tooth.
 */
export function colorForFDI(fdi) {
  const i = fdiIndex(fdi);
  if (i < 0) return UNKNOWN_HEX;
  const slot = (i * HUE_STRIDE) % HUE_SLOTS;
  return hslToHex(slot * (360 / HUE_SLOTS),
                  SATURATION[i % SATURATION.length],
                  LIGHTNESS[i % LIGHTNESS.length]);
}

/** `[r, g, b]` in 0..1, the form three.js and a vertex-colour buffer want. */
export function rgbForFDI(fdi) {
  const hex = colorForFDI(fdi);
  return [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  ];
}

export function rgbFromHex(hex) {
  return [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  ];
}

/**
 * Selection is a TREATMENT LAID OVER the tooth's own colour, never a
 * replacement, so the clinician can still tell WHICH tooth is selected from
 * its colour while it is selected. The base colour is returned unchanged
 * alongside the accent, so a caller restores by dropping the treatment rather
 * than by remembering what was underneath - the pattern that lost the
 * original colour when two selections overlapped.
 */
export function selectionTreatment(fdi) {
  return {
    base: colorForFDI(fdi),
    emissive: SELECTION_HEX,
    emissiveIntensity: 0.45,
    outline: SELECTION_HEX,
    /** Mix toward the accent, keeping most of the tooth's identity. */
    mix: 0.25,
  };
}

/** Hover is weaker than selection and uses the same accent, so the two read
 *  as one interaction rather than two unrelated highlights. */
export function hoverTreatment(fdi) {
  return { base: colorForFDI(fdi), emissive: SELECTION_HEX,
           emissiveIntensity: 0.18, mix: 0.10 };
}

/** Every permanent tooth and its colour. For a legend, and for the test. */
export function paletteTable() {
  return FDI_TEETH.map((fdi) => ({ fdi, hex: colorForFDI(fdi) }));
}

/** sRGB hex -> CIE L*a*b*. Used by the distinctness check, and exported so
 *  the claim "visually distinct" is measurable by anything that wants to. */
export function hexToLab(hex) {
  const lin = (v) => (v <= 0.04045 ? v / 12.92
                                   : Math.pow((v + 0.055) / 1.055, 2.4));
  const [r, g, b] = rgbFromHex(hex).map(lin);
  const X = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
  const Y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.0;
  const Z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;
  const f = (t) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);
  const [fx, fy, fz] = [f(X), f(Y), f(Z)];
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

export function labDistance(hexA, hexB) {
  const a = hexToLab(hexA);
  const b = hexToLab(hexB);
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}
