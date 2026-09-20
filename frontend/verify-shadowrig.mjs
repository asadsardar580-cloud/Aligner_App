/**
 * Shadow rig arithmetic check.  Run:  node verify-shadowrig.mjs
 *
 * WHY THIS EXISTS. Until commit 2d372b4 the shadow rig had never executed:
 * `three.current` was reassigned after `sun` and `catcher` were attached to it,
 * so `aimShadows` hit `if (!sun || !catcher) return;` on every call and
 * returned silently. Turning that path on for the first time is what put "Set
 * Occlusal Plane makes the viewport go black" on the table.
 *
 * There is no browser here and this file does not pretend otherwise: it cannot
 * tell you the scene is lit. What it CAN do is assert that every number the
 * rig hands to a light is finite, correctly signed and inside its own frustum
 * — so that if the viewport is still black, the arithmetic is excluded and the
 * search moves to the three.js wiring.
 */
import { computeShadowRig, FRUSTUM_MARGIN } from "./src/shadowRig.js";

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? "   " + detail : ""}`);
  if (!ok) failures++;
};

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const centreOf = (box) => box.min.map((v, i) => (v + box.max[i]) / 2);
const corners = (box) => Array.from({ length: 8 }, (_, i) => [
  i & 1 ? box.max[0] : box.min[0],
  i & 2 ? box.max[1] : box.min[1],
  i & 4 ? box.max[2] : box.min[2]]);

// Depth along the light's own view direction, the quantity near/far bracket.
const depthOf = (rig, p) => {
  const view = sub(rig.targetPosition, rig.sunPosition);
  const L = len(view);
  return dot(sub(p, rig.sunPosition), view.map((x) => x / L));
};

// A real arch's extent. case_lower.stl measures 55.6 x 52.9 x 18.3 mm.
const ARCH = { min: [-27.8, -26.5, -6.7], max: [27.8, 26.4, 11.6] };
// Both jaws, and a tilted basis, because u_occ points OUT of the mouth for
// both (arch_frame.py:50) and nothing in the rig may depend on which it is.
const JAWS = [
  { name: "mandible (u_occ = +Z)", u_occ: [0, 0, 1], u_sag: [0, 1, 0], u_tra: [1, 0, 0] },
  { name: "maxilla  (u_occ = -Z)", u_occ: [0, 0, -1], u_sag: [0, -1, 0], u_tra: [1, 0, 0] },
  { name: "tilted scanner basis  ", u_occ: [-0.176, -0.365, -0.914],
    u_sag: [0.246, 0.881, -0.399], u_tra: [0.953, -0.298, -0.064] },
];

console.log("\n1. The sun is occlusal, the catcher is tissue-side — for every basis");
for (const j of JAWS) {
  const rig = computeShadowRig({ box: ARCH, ...j });
  if (!rig) { check(j.name, false, "returned null"); continue; }
  const c = centreOf(ARCH);
  const sunSide = dot(sub(rig.sunPosition, c), j.u_occ);
  const catSide = dot(sub(rig.catcherPosition, c), j.u_occ);
  check(`${j.name}: sun on the +u_occ side`, sunSide > 0, `${sunSide.toFixed(2)} mm`);
  check(`${j.name}: catcher on the -u_occ side`, catSide < 0, `${catSide.toFixed(2)} mm`);
  check(`${j.name}: intensity 1.2 per section 3.1`, rig.intensity === 1.2);

  // Section 3.1's own words: a light exactly along the view axis gives flat,
  // relief-free shading. Assert it is genuinely off-axis.
  const dir = sub(rig.targetPosition, rig.sunPosition);
  const offAxis = Math.acos(Math.min(1, -dot(dir, j.u_occ) / len(dir))) * 180 / Math.PI;
  check(`${j.name}: sun is off-axis, not head-on`, offAxis > 10 && offAxis < 45,
        `${offAxis.toFixed(1)} deg from u_occ`);

  // The quaternion must lay the catcher's +Z normal along u_occ. Compared
  // against the NORMALISED axis: the rig normalises its inputs, and the tilted
  // basis above is 0.9998 long, which is a fair model of what arrives over
  // JSON — asserting against the raw vector would measure its length, not the
  // rotation. (It did, on the first run: dot came out 0.999798 = |u_occ|.)
  const L = len(j.u_occ), u = j.u_occ.map((v) => v / L);
  const [x, y, z, w] = rig.catcherQuaternion;
  const n = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)];
  check(`${j.name}: catcher normal lands on u_occ`, dot(n, u) > 1 - 1e-9,
        `dot = ${dot(n, u).toFixed(12)}`);
}

console.log("\n2. Everything that must cast or receive is inside the frustum");
for (const j of JAWS) {
  const rig = computeShadowRig({ box: ARCH, ...j });
  const { near, far, left, right, top, bottom } = rig.frustum;
  const depths = [...corners(ARCH), rig.catcherPosition].map((p) => depthOf(rig, p));
  check(`${j.name}: catcher inside [near, far]`,
        depthOf(rig, rig.catcherPosition) > near && depthOf(rig, rig.catcherPosition) < far,
        `catcher at ${depthOf(rig, rig.catcherPosition).toFixed(1)}, `
        + `frustum [${near.toFixed(1)}, ${far.toFixed(1)}]`);
  check(`${j.name}: all 8 box corners inside [near, far]`,
        depths.every((d) => d > near - 1e-9 && d < far + 1e-9));
  check(`${j.name}: near < far, both finite`, near < far
        && Number.isFinite(near) && Number.isFinite(far));
  check(`${j.name}: ortho box is square and centred`,
        right === -left && top === -bottom && right > 0);
  // The margin section 3.1 asks for, measured rather than assumed.
  const slack = Math.min(...depths) - near;
  check(`${j.name}: >= 1.5 x radius of near-plane slack`,
        slack >= rig.radius * FRUSTUM_MARGIN - 1e-6,
        `${slack.toFixed(1)} mm vs ${(rig.radius * FRUSTUM_MARGIN).toFixed(1)} mm`);

  // THE CATCHER MUST NOT OVERHANG THE SHADOW MAP. A fixed 400 mm plane against
  // this 151.7 mm box left 85.6% of the catcher sampling the depth texture
  // with clamped UVs, which reads as SHADOWED — a full-frame black wash, since
  // the plane is also large enough to fill the viewport by itself.
  //
  // COMPARING SIDE LENGTHS IS NOT ENOUGH, and this check passed for two
  // sections while it was wrong. Measured in a real WebGL context
  // (frontend/e2e-shadow/shadow-darkening.spec.js), with `catcherSize` set to
  // exactly `right - left`, THREE OF FOUR CORNERS were outside the map at NDC
  // 1.13 and 1.23. Two reasons a side-length test cannot see:
  //   * a square's corners reach sqrt(2) further than its edges;
  //   * the catcher is dropped along -u_occ while the light is 23.3 degrees
  //     off u_occ, so its CENTRE is already ~0.4 x radius off the light axis.
  // So the test is the real one: the furthest point of the catcher from the
  // light axis, which is its centre's offset plus its half-diagonal.
  const vraw = sub(rig.targetPosition, rig.sunPosition);
  const vlen = len(vraw);
  const vdir = vraw.map((x) => x / vlen);
  const axisOffset = (p) => {
    const d = sub(p, rig.sunPosition);
    const along = dot(d, vdir);
    const r = [d[0] - vdir[0] * along, d[1] - vdir[1] * along, d[2] - vdir[2] * along];
    return Math.hypot(r[0], r[1], r[2]);
  };
  const reach = axisOffset(rig.catcherPosition) + rig.catcherSize * Math.SQRT1_2;
  check(`${j.name}: every catcher CORNER is inside the shadow map`,
        reach <= right + 1e-9,
        `furthest corner ${reach.toFixed(1)} mm vs half-box ${right.toFixed(1)} mm`);
  const castReach = Math.max(...corners(ARCH).map(axisOffset));
  check(`${j.name}: the cast is inside the shadow map`,
        castReach <= right + 1e-9,
        `cast reach ${castReach.toFixed(1)} mm vs half-box ${right.toFixed(1)} mm`);
  check(`${j.name}: catcher still covers the cast`,
        rig.catcherSize >= 2 * rig.radius,
        `${rig.catcherSize.toFixed(1)} mm vs a ${(2 * rig.radius).toFixed(1)} mm cast`);
}

console.log("\n3. The hardcoded near/far this replaces: where they actually break");
{
  // The old shadow camera was built with near = 1, far = 400 and aimShadows
  // only ever set left/right/top/bottom. THIS IS THE HONEST VERSION OF THE
  // CLAIM: for one arch, and for two in occlusion, 400 is never exceeded. It
  // is exceeded once the combined bounding radius passes the crossover below
  // — and the reason that is reachable HERE and not in most viewers is
  // invariant 1: this app never re-centres a scan, so two arches recorded in
  // different scanner origins keep that separation in the combined box.
  const j = JAWS[0];
  const depthOfCatcher = (box) => {
    const rig = computeShadowRig({ box, ...j });
    return { d: depthOf(rig, rig.catcherPosition), far: rig.frustum.far, r: rig.radius };
  };

  const one = depthOfCatcher(ARCH);
  check("one arch: the old far=400 would in fact have held",
        one.d < 400, `catcher depth ${one.d.toFixed(1)} mm (radius ${one.r.toFixed(1)})`);

  const occluded = { min: [-28, -27, -24], max: [28, 27, 24] };   // two arches, in bite
  const two = depthOfCatcher(occluded);
  check("two arches in occlusion: 400 still holds",
        two.d < 400, `catcher depth ${two.d.toFixed(1)} mm (radius ${two.r.toFixed(1)})`);

  // Crossover, solved numerically rather than asserted from a story.
  let lo = 10, hi = 4000;
  for (let i = 0; i < 60; i++) {
    const m = (lo + hi) / 2;
    const s = m / Math.sqrt(3) / 2;
    if (depthOfCatcher({ min: [-s, -s, -s], max: [s, s, s] }).d < 400) lo = m; else hi = m;
  }
  const rCross = depthOfCatcher({ min: [-lo / Math.sqrt(3) / 2, -lo / Math.sqrt(3) / 2, -lo / Math.sqrt(3) / 2],
                                  max: [lo / Math.sqrt(3) / 2, lo / Math.sqrt(3) / 2, lo / Math.sqrt(3) / 2] }).r;
  console.log(`        crossover: the old far=400 starts clipping the catcher at a `
            + `bounding radius of ${rCross.toFixed(1)} mm`);

  // Two arches from separate scanner origins, 180 mm apart — legal, because
  // nothing in this app is allowed to move them together.
  const apart = { min: [-28, -27, -24], max: [28, 27, 204] };
  const far2 = depthOfCatcher(apart);
  check("separated scans: old far=400 WOULD have clipped the catcher",
        far2.d > 400, `catcher depth ${far2.d.toFixed(1)} mm (radius ${far2.r.toFixed(1)})`);
  check("computed far covers it", far2.far > far2.d,
        `far ${far2.far.toFixed(1)} mm > ${far2.d.toFixed(1)} mm`);
}

console.log("\n4. Bad input yields null, never a half-finite rig");
{
  const j = JAWS[0];
  const bad = {
    "null box": { box: null, ...j },
    "NaN in the box": { box: { min: [NaN, 0, 0], max: [1, 1, 1] }, ...j },
    "Infinity in the box": { box: { min: [-Infinity, 0, 0], max: [1, 1, 1] }, ...j },
    "empty box (zero size)": { box: { min: [0, 0, 0], max: [0, 0, 0] }, ...j },
    "u_occ is zero-length": { box: ARCH, u_occ: [0, 0, 0], u_sag: j.u_sag, u_tra: j.u_tra },
    "u_occ is NaN": { box: ARCH, u_occ: [NaN, 0, 1], u_sag: j.u_sag, u_tra: j.u_tra },
    "u_occ missing": { box: ARCH, u_sag: j.u_sag, u_tra: j.u_tra },
  };
  for (const [name, arg] of Object.entries(bad)) {
    let r, threw = false;
    try { r = computeShadowRig(arg); } catch { threw = true; }
    check(`${name} -> null`, !threw && r === null, threw ? "it threw instead" : "");
  }
  // A PARTIAL frame still has to work: the offsets exist to break head-on
  // lighting, and any perpendicular pair does that.
  const partial = computeShadowRig({ box: ARCH, u_occ: [0, 0, 1] });
  check("u_occ alone still produces a usable rig", partial !== null
    && Object.values(partial.frustum).every(Number.isFinite));
}

console.log(failures === 0
  ? "\nPASS  the shadow rig is finite, correctly signed and self-consistent"
  : `\nFAIL  ${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
