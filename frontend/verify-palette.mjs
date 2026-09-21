/**
 * The FDI colour table, checked the way the other verifiers are: plain node,
 * no browser, no bundler, numbers printed rather than asserted in prose.
 *
 * WHAT THIS CATCHES THAT A GLANCE DOES NOT. The palette it replaces was
 * fifteen colours indexed by position over 32 teeth, so seventeen teeth
 * shared a colour with another tooth - FDI 11 and 31 were both #e6194b. That
 * is invisible in a code review and obvious in a pairwise distance table.
 *
 *   node verify-palette.mjs
 */
import {
  FDI_TEETH, colorForFDI, rgbForFDI, fdiIndex, labDistance,
  GINGIVA_HEX, UNKNOWN_HEX, SELECTION_HEX, VIEWPORT_BACKGROUND_HEX,
  selectionTreatment, paletteTable,
} from "./src/fdiPalette.js";

let checks = 0;
let failures = 0;

function ok(cond, label, detail = "") {
  checks += 1;
  if (!cond) {
    failures += 1;
    console.log(`  FAIL  ${label}${detail ? "  " + detail : ""}`);
  }
  return cond;
}

console.log("[FDI palette]");

// --- 1  every permanent tooth has a colour --------------------------------
ok(FDI_TEETH.length === 32, "32 permanent teeth", `got ${FDI_TEETH.length}`);
for (const fdi of FDI_TEETH) {
  ok(/^#[0-9a-f]{6}$/.test(colorForFDI(fdi)), `FDI ${fdi} is a hex colour`,
     colorForFDI(fdi));
}

// --- 2  EVERY TOOTH IS A DIFFERENT COLOUR ---------------------------------
// The defect that motivated this file. Not "mostly different".
const seen = new Map();
let collisions = 0;
for (const fdi of FDI_TEETH) {
  const hex = colorForFDI(fdi);
  if (seen.has(hex)) {
    collisions += 1;
    console.log(`  FAIL  FDI ${fdi} shares ${hex} with FDI ${seen.get(hex)}`);
  }
  seen.set(hex, fdi);
}
ok(collisions === 0, "no two teeth share a colour", `${collisions} collisions`);
checks += 1;

// --- 3  DISTINCT IS A MEASUREMENT, NOT AN OPINION -------------------------
// Minimum pairwise CIE L*a*b* distance over all 496 pairs. Roughly, dE < 10
// is "similar shades" and > 20 is "plainly different colours" - this is a
// SCREEN LEGIBILITY FLOOR for a prototype, not a colorimetric standard.
const MIN_DE = 15.0;
let worst = Infinity;
let worstPair = null;
for (let i = 0; i < FDI_TEETH.length; i++) {
  for (let j = i + 1; j < FDI_TEETH.length; j++) {
    const d = labDistance(colorForFDI(FDI_TEETH[i]), colorForFDI(FDI_TEETH[j]));
    if (d < worst) { worst = d; worstPair = [FDI_TEETH[i], FDI_TEETH[j]]; }
  }
}
ok(worst >= MIN_DE, `minimum pairwise dE >= ${MIN_DE}`,
   `worst ${worst.toFixed(2)} between FDI ${worstPair?.[0]} and ${worstPair?.[1]}`);
console.log(`  minimum pairwise dE      ${worst.toFixed(2)}  `
            + `(FDI ${worstPair[0]} vs ${worstPair[1]})`);

// --- 4  ADJACENT TEETH ARE NOT SIMILAR ------------------------------------
// The pairs a clinician confuses are neighbours in the same quadrant and the
// mirrored pair across the midline, so those get their own floor.
let worstAdj = Infinity;
let worstAdjPair = null;
for (const q of [1, 2, 3, 4]) {
  for (let t = 1; t < 8; t++) {
    const d = labDistance(colorForFDI(q * 10 + t), colorForFDI(q * 10 + t + 1));
    if (d < worstAdj) { worstAdj = d; worstAdjPair = [q * 10 + t, q * 10 + t + 1]; }
  }
}
for (const [a, b] of [[1, 2], [3, 4]]) {
  for (let t = 1; t <= 8; t++) {
    const d = labDistance(colorForFDI(a * 10 + t), colorForFDI(b * 10 + t));
    if (d < worstAdj) { worstAdj = d; worstAdjPair = [a * 10 + t, b * 10 + t]; }
  }
}
ok(worstAdj >= MIN_DE, "neighbours and mirrored pairs are distinct",
   `worst ${worstAdj.toFixed(2)} between FDI ${worstAdjPair?.[0]} and ${worstAdjPair?.[1]}`);
console.log(`  minimum neighbour dE     ${worstAdj.toFixed(2)}  `
            + `(FDI ${worstAdjPair[0]} vs ${worstAdjPair[1]})`);

// --- 5  GINGIVA IS NOT A TOOTH --------------------------------------------
let worstGing = Infinity;
let gingPair = null;
for (const fdi of FDI_TEETH) {
  const d = labDistance(colorForFDI(fdi), GINGIVA_HEX);
  if (d < worstGing) { worstGing = d; gingPair = fdi; }
}
ok(worstGing >= MIN_DE, "gingiva is distinct from every tooth",
   `worst ${worstGing.toFixed(2)} against FDI ${gingPair}`);
console.log(`  minimum gingiva dE       ${worstGing.toFixed(2)}  `
            + `(against FDI ${gingPair})`);

// --- 5b  A COLOUR CAN BE FAR FROM EVERY OTHER AND STILL BE INVISIBLE ------
// Distance between tooth colours says nothing about whether either one can
// be seen against the viewport. Checked separately, against the clear colour.
const MIN_BG_DE = 25.0;
let worstBg = Infinity;
let bgPair = null;
for (const fdi of FDI_TEETH) {
  const d = labDistance(colorForFDI(fdi), VIEWPORT_BACKGROUND_HEX);
  if (d < worstBg) { worstBg = d; bgPair = fdi; }
}
ok(worstBg >= MIN_BG_DE, "every tooth is visible against the viewport",
   `worst ${worstBg.toFixed(2)} for FDI ${bgPair}`);
console.log(`  minimum background dE    ${worstBg.toFixed(2)}  `
            + `(FDI ${bgPair})`);

// --- 6  DETERMINISM, which is the whole contract --------------------------
// Same FDI, same colour - across a refresh, a re-segmentation and a change of
// provider. There is no state in the module, so this is checked by calling it
// again and by checking it does not depend on the ORDER teeth are asked for.
const forward = FDI_TEETH.map(colorForFDI);
const shuffled = [...FDI_TEETH].reverse().map(colorForFDI).reverse();
ok(forward.every((c, i) => c === shuffled[i]),
   "colour does not depend on the order teeth are requested in");
ok(forward.every((c, i) => c === colorForFDI(FDI_TEETH[i])),
   "repeated calls agree");
ok(colorForFDI("31") === colorForFDI(31),
   "a string FDI and a numeric FDI agree");

// --- 7  UNKNOWN IS NOT A CONFIDENT TOOTH ----------------------------------
for (const bad of [0, 19, 29, 30, 49, 99, null, undefined, NaN, "x", 11.5]) {
  ok(colorForFDI(bad) === UNKNOWN_HEX, `unknown id ${String(bad)} -> grey`,
     colorForFDI(bad));
}
ok(fdiIndex(11) === 0 && fdiIndex(48) === 31, "index spans 0..31");
ok(labDistance(UNKNOWN_HEX, GINGIVA_HEX) >= 10,
   "unknown is distinguishable from gingiva",
   labDistance(UNKNOWN_HEX, GINGIVA_HEX).toFixed(2));

// --- 8  SELECTION IS A TREATMENT, NOT A REPLACEMENT -----------------------
// A selected tooth must still be identifiable by its own colour.
for (const fdi of [11, 26, 37, 48]) {
  const t = selectionTreatment(fdi);
  ok(t.base === colorForFDI(fdi),
     `selection keeps FDI ${fdi} base colour`, `${t.base} vs ${colorForFDI(fdi)}`);
  ok(t.emissive === SELECTION_HEX, `selection accent on FDI ${fdi}`);
  ok(t.mix > 0 && t.mix <= 0.35,
     `selection mix stays a treatment on FDI ${fdi}`, String(t.mix));
}

// --- 9  the rgb form a vertex-colour buffer needs --------------------------
for (const fdi of FDI_TEETH) {
  const rgb = rgbForFDI(fdi);
  ok(rgb.length === 3 && rgb.every((x) => x >= 0 && x <= 1),
     `FDI ${fdi} rgb in 0..1`, JSON.stringify(rgb));
}

console.log("\n  FDI  colour     FDI  colour     FDI  colour     FDI  colour");
const rows = paletteTable();
for (let i = 0; i < 8; i++) {
  const line = [0, 1, 2, 3]
    .map((q) => {
      const r = rows[q * 8 + i];
      return `  ${String(r.fdi).padStart(3)}  ${r.hex}`;
    })
    .join("   ");
  console.log(line);
}

console.log(`\n  ${checks - failures}/${checks} checks passed`);
if (failures) {
  console.log("VERIFY-PALETTE FAILED");
  process.exit(1);
}
console.log("VERIFY-PALETTE PASSED");
