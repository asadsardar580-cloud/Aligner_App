/**
 * Occlusal-end rule check.  Run:  node verify-framing.mjs
 *
 * WHY THIS EXISTS. frameArch has to decide which END of the arch's flattest
 * axis points out of the mouth, because an eigenvector's sign is arbitrary and
 * getting it wrong renders the cast upside-down. The old rule asked where the
 * scan sat relative to the scanner's origin:
 *
 *     if (up.dot(camera.position - centre) < 0) up.negate();
 *
 * with camera.position still (0,0,0) on a fresh page. On case_lower.stl the
 * bounding centre projects 5.88 mm onto that axis, so the old rule got the
 * right answer — by 5.88 mm of luck. Translating the scan 6 mm flipped it, and
 * invariant 1 says a translation must change NOTHING. That is what this file
 * pins: not just that the rule is right on a fixture, but that it is right for
 * a reason that a rigid motion cannot disturb.
 *
 * THE FIXTURES ARE A MODEL, THE INVARIANCE IS A PROOF. The synthetic arches
 * below are built from arch anatomy — a broad gingival band under discrete
 * crowns of differing height at differing distance from the arch centroid —
 * and the rule is asked to recover an occlusal direction it was never told.
 * That part is only as good as the model. The translation and rotation cases
 * are not: they hold for any input at all, which is why they are the ones that
 * would have caught the real defect.
 */
import { occlusalOrientation, archAnterior } from "./src/frameArch.js";

let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`  ${ok ? "PASS" : "FAIL"}  ${name}${detail ? "   " + detail : ""}`);
  if (!ok) failures++;
};

// ---------------------------------------------------------------------------
// A synthetic arch, in a basis we choose, so the true occlusal axis is known.
//
// Anatomy that matters to the rule, and why each part is here:
//   - the GINGIVAL band is a near-continuous ribbon running right around the
//     periphery at almost constant radius: it is the LOW-variance end;
//   - the CROWNS are discrete, of unequal height, and sit at unequal distance
//     from the arch centroid (incisors well forward of it, molars close to
//     it): that is the HIGH-variance end, and also the long thin tail that
//     gives the projection its positive skew.
// Neither property is imposed on the output; both fall out of the shape.
// ---------------------------------------------------------------------------
function syntheticArch({ uOcc, uSag, uTra, teeth = 14, jitter = 0.0 }) {
  const P = [];
  const push = (t, s, r) => P.push(
    uOcc[0] * t + uSag[0] * s + uTra[0] * r,
    uOcc[1] * t + uSag[1] * s + uTra[1] * r,
    uOcc[2] * t + uSag[2] * s + uTra[2] * r);

  // Deterministic pseudo-noise: a test that flakes is worse than no test.
  let seed = 12345;
  const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  const wob = () => (rnd() - 0.5) * 2 * jitter;

  // Horseshoe centreline: theta 0 at the midline, opening posteriorly.
  const THETA = 2.0;                       // half-sweep, radians
  const R0 = 21.0;                         // arch radius, mm

  // 1. The gingival / vestibular band. Dense, wide in the axial direction,
  //    and radially TIGHT at its deepest extent.
  for (let i = 0; i < 260; i++) {
    const th = -THETA + (2 * THETA * i) / 259;
    const s = R0 * Math.cos(th), r = R0 * Math.sin(th);
    for (let j = 0; j < 26; j++) {
      const t = -7 + (7 * j) / 25;          // -7 mm up to the gingival margin
      // The flange flares slightly, then settles to a constant-radius band at
      // the deepest 2 mm — which is what makes the tissue end low-variance.
      const flare = t < -5 ? 0.2 : 1.6 * (1 + t / 7);
      for (const k of [-1, 1]) {
        const rad = R0 + k * flare + wob();
        push(t, rad * Math.cos(th), rad * Math.sin(th));
      }
    }
  }

  // 2. The crowns. Sparse, unequal heights, unequal radii.
  for (let n = 0; n < teeth; n++) {
    const th = -THETA + (2 * THETA * n) / (teeth - 1);
    const anterior = Math.abs(th) < 0.7;
    const h = anterior ? 9.5 : 6.5;        // incisal edges reach highest
    for (let j = 0; j < 16; j++) {
      const t = (h * j) / 15;
      const taper = 3.0 * (1 - 0.6 * (t / h));
      for (let a = 0; a < 10; a++) {
        const phi = (2 * Math.PI * a) / 10;
        const rad = R0 + taper * Math.cos(phi) + wob();
        const along = taper * Math.sin(phi) * 0.5;
        push(t, rad * Math.cos(th) - along * Math.sin(th),
                rad * Math.sin(th) + along * Math.cos(th));
      }
    }
  }
  return Float64Array.from(P);
}

const translate = (P, d) => {
  const Q = Float64Array.from(P);
  for (let i = 0; i < Q.length; i += 3) { Q[i] += d[0]; Q[i + 1] += d[1]; Q[i + 2] += d[2]; }
  return Q;
};
// A right-handed rotation about an arbitrary axis (Rodrigues), so the test is
// not accidentally passing on an axis-aligned special case.
const rotate = (P, axis, ang) => {
  const L = Math.hypot(...axis);
  const [kx, ky, kz] = axis.map((x) => x / L);
  const c = Math.cos(ang), s = Math.sin(ang);
  const Q = Float64Array.from(P);
  for (let i = 0; i < Q.length; i += 3) {
    const v = [Q[i], Q[i + 1], Q[i + 2]];
    const kv = kx * v[0] + ky * v[1] + kz * v[2];
    const kxv = [ky * v[2] - kz * v[1], kz * v[0] - kx * v[2], kx * v[1] - ky * v[0]];
    for (let j = 0; j < 3; j++) {
      Q[i + j] = v[j] * c + kxv[j] * s + [kx, ky, kz][j] * kv * (1 - c);
    }
  }
  return Q;
};
const rotVec = (v, axis, ang) => Array.from(rotate(Float64Array.from(v), axis, ang));
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

// ---------------------------------------------------------------------------
console.log("\n1. The rule recovers a known occlusal direction");

// Two jaws, two completely different bases. u_occ points out of the mouth for
// BOTH (arch_frame.py:50), which is exactly why no jaw branch appears here.
const CASES = [
  { name: "mandible-like  (scanner Z up)",
    uOcc: [0, 0, 1], uSag: [0, 1, 0], uTra: [1, 0, 0] },
  { name: "maxilla-like   (tilted, Y-back scanner)",
    uOcc: [-0.176, -0.365, -0.914], uSag: [0.246, 0.881, -0.399], uTra: [0.953, -0.298, -0.064] },
];

const built = [];
for (const c of CASES) {
  const P = syntheticArch({ uOcc: c.uOcc, uSag: c.uSag, uTra: c.uTra, jitter: 0.05 });
  const res = occlusalOrientation(P);
  built.push({ c, P, res });
  const d = res.axis ? dot3(res.axis, c.uOcc) : 0;
  check(c.name, d > 0.99,
        `dot=${d.toFixed(6)}  skew=${res.skew.toFixed(3)}  `
        + `spread occ ${res.spreadOcclusal.toFixed(2)} / tis ${res.spreadTissue.toFixed(2)}`);
  check(`   ${c.name.split("(")[0].trim()}: both signals agree`, res.agree);
}

// ---------------------------------------------------------------------------
console.log("\n2. Rigid motion changes nothing — the property the old rule failed");

for (const { c, P, res } of built) {
  // 50 mm, and then far enough to have flipped the old sign test many times.
  for (const d of [[50, 0, 0], [0, 50, 0], [0, 0, 50], [-180, 140, 260]]) {
    const r = occlusalOrientation(translate(P, d));
    const agree = r.axis ? dot3(r.axis, res.axis) : 0;
    check(`translate [${d}] ${c.name.split("(")[0].trim()}`,
          agree > 1 - 1e-9, `dot with baseline = ${agree.toFixed(12)}`);
  }
  const axis = [0.3, -0.8, 0.52], ang = 1.13;
  const r = occlusalOrientation(rotate(P, axis, ang));
  const expect = rotVec(res.axis, axis, ang);
  const agree = r.axis ? dot3(r.axis, expect) : 0;
  check(`rotate 1.13 rad  ${c.name.split("(")[0].trim()}`,
        agree > 1 - 1e-9, `dot with rotated baseline = ${agree.toFixed(12)}`);
}

// The old rule, run on the same fixture, to show what is being fixed rather
// than assert it from the commit message. It is reproduced here in full:
// on a fresh page camera.position is the origin, so it reduces to this.
console.log("\n3. The old rule, on the same fixture, for contrast");
{
  const { P, res } = built[0];
  const oldSign = (Q) => {
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < Q.length; i += 3) {
      const t = dot3([Q[i], Q[i + 1], Q[i + 2]], res.axis);
      if (t < lo) lo = t; if (t > hi) hi = t;
    }
    const centre = (lo + hi) / 2;           // bounding-sphere centre, projected
    return -centre >= 0 ? 1 : -1;           // up.dot(origin - centre) >= 0
  };
  // Two translations, opposite ways down the same axis. The old rule is not
  // asked whether it is RIGHT here — it is asked whether it is STABLE, and a
  // rule that answers differently for the same cast in two places is not.
  const near = oldSign(translate(P, [0, 0, 60]));
  const far = oldSign(translate(P, [0, 0, -60]));
  check("old rule gives two different answers for the same cast",
        near !== far, `+60mm -> ${near > 0 ? "+" : "-"},  -60mm -> ${far > 0 ? "+" : "-"}`);
  for (const d of [[0, 0, 60], [0, 0, -60]]) {
    const rNew = occlusalOrientation(translate(P, d));
    check(`new rule is unmoved by [${d}]`, dot3(rNew.axis, res.axis) > 1 - 1e-9);
  }
}

// ---------------------------------------------------------------------------
console.log("\n4. When the two signals disagree, skewness is the tie-break");

// A deliberate adversary, not an arch: a thin SPIKE at +Z (positive skew, but
// a tight constant radius, so LOW spread) over a ragged broad slab at -Z (HIGH
// spread). Skewness says +Z; the spread signal says -Z. The documented rule is
// that skewness wins and the disagreement is logged.
//
// The spike is kept SHORT on purpose. Make it tall enough and Z stops being
// the flattest axis at all, PCA picks something in-plane, and the fixture
// stops testing the tie-break — which is exactly what the first draft of this
// fixture did (spike to +28: sigma_z 7.4 vs sigma_xy 7.8, and the axis came
// out 0.13 off Z).
function spikeAndRaggedSlab() {
  const P = [];
  let seed = 7; const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  for (let i = 0; i < 4000; i++) {          // ragged slab at -Z: radius 2..18
    const th = 2 * Math.PI * rnd();
    const rad = 2 + 16 * rnd();
    P.push(rad * Math.cos(th), rad * Math.sin(th), -4 * rnd());
  }
  for (let i = 0; i < 500; i++) {           // thin spike, radius ~1, up to +12
    const th = 2 * Math.PI * rnd();
    const rad = 1.0 + 0.1 * rnd();
    P.push(rad * Math.cos(th), rad * Math.sin(th), 12 * rnd());
  }
  return Float64Array.from(P);
}
{
  const warnings = [];
  const realWarn = console.warn;
  console.warn = (...a) => warnings.push(a.join(" "));
  const r = occlusalOrientation(spikeAndRaggedSlab());
  console.warn = realWarn;

  check("signals genuinely disagree on this fixture", r.agree === false,
        `spread occ ${r.spreadOcclusal.toFixed(2)} / tis ${r.spreadTissue.toFixed(2)}`);
  check("skewness wins: +Z (the spike) is called occlusal",
        r.axis !== null && r.axis[2] > 0.99, `axis z = ${r.axis?.[2].toFixed(6)}`);
  check("the disagreement is logged, not swallowed",
        warnings.some((w) => w.includes("[Arch Framing]")),
        warnings[0] ? `"${warnings[0].slice(0, 72)}..."` : "no warning seen");
}

// ---------------------------------------------------------------------------
console.log("\n5. Degenerate input returns null rather than NaN");
{
  // NaN in camera.up is the black-viewport failure: every comparison against
  // NaN is false, so nothing downstream notices.
  const cases = {
    "empty": new Float64Array(0),
    "a single point": Float64Array.from([1, 2, 3]),
    "all coincident": Float64Array.from(Array(60).fill(0)),
    "carrying NaN": Float64Array.from(Array.from({ length: 300 },
      (_, i) => (i === 7 ? NaN : Math.sin(i) * 10))),
  };
  for (const [name, P] of Object.entries(cases)) {
    const r = occlusalOrientation(P);
    const clean = r.axis === null || r.axis.every(Number.isFinite);
    check(`${name}: no NaN escapes`, clean, r.axis ? `axis=[${r.axis}]` : "axis=null");
  }
}

// ---------------------------------------------------------------------------
console.log("\n6. The ANTERIOR direction, which decides the roll");

// The defect this covers, reported from a browser: camera.up was the occlusal
// axis while the camera sat 34.7 deg off that same axis, so screen-up collapsed
// to roughly the scanner's -X and a mandible could render rolled 180 deg -
// which reads exactly like a maxilla. The rule is the arch's own width: widest
// between the molars, narrowest at the incisors, for every human arch.
for (const { c, P, res } of built) {
  const r = archAnterior(P, res.axis);
  const name = c.name.split("(")[0].trim();
  if (!r) { check(`${name}: anterior is readable`, false, "returned null"); continue; }
  const d = dot3(r.anterior, c.uSag);
  check(`${name}: anterior recovered`, d > 0.99,
        `dot=${d.toFixed(6)}  width ratio ${r.ratio.toFixed(2)}`);
  // Against res.axis, the axis archAnterior was actually handed - NOT the
  // fixture's ideal uOcc. The rule recovers the axis to 0.3 deg, so a vector
  // exactly perpendicular to what it was given is 0.005 off perpendicular to
  // the ideal, and demanding 1e-9 of that measures the fixture, not the rule.
  check(`${name}: transverse is perpendicular to both`,
        Math.abs(dot3(r.transverse, res.axis)) < 1e-12
        && Math.abs(dot3(r.transverse, r.anterior)) < 1e-12,
        `vs axis ${dot3(r.transverse, res.axis).toExponential(1)}, `
        + `vs anterior ${dot3(r.transverse, r.anterior).toExponential(1)}`);
  check(`${name}: the centroid signal corroborates`, r.agree);

  // Rigid motion must not move it either - same property, same reason.
  for (const t of [[80, -40, 25]]) {
    const m = archAnterior(translate(P, t), res.axis);
    check(`${name}: anterior unmoved by [${t}]`,
          m && dot3(m.anterior, r.anterior) > 1 - 1e-9,
          m ? `dot ${dot3(m.anterior, r.anterior).toFixed(12)}` : "null");
  }
  const axis = [0.3, -0.8, 0.52], ang = 1.13;
  const rot2 = archAnterior(rotate(P, axis, ang), rotVec(res.axis, axis, ang));
  const expect = rotVec(r.anterior, axis, ang);
  check(`${name}: anterior rotates with the scan`,
        rot2 && dot3(rot2.anterior, expect) > 1 - 1e-9,
        rot2 ? `dot ${dot3(rot2.anterior, expect).toFixed(12)}` : "null");
}

// A shape with no front and no back must be REFUSED, not guessed at: returning
// a coin flip here would roll the arch 180 deg half the time, which is the
// exact symptom being fixed.
{
  const warnings = [];
  const realWarn = console.warn;
  console.warn = (...a) => warnings.push(a.join(" "));
  const disc = [];
  let seed = 3; const rnd = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed / 0x7fffffff; };
  for (let i = 0; i < 4000; i++) {
    const th = 2 * Math.PI * rnd(), rad = 18 + 2 * rnd();
    disc.push(rad * Math.cos(th), rad * Math.sin(th), 3 * (rnd() - 0.5));
  }
  const r = archAnterior(Float64Array.from(disc), [0, 0, 1]);
  console.warn = realWarn;
  check("a symmetric ring has no anterior, and is refused", r === null,
        r ? `returned ratio ${r.ratio.toFixed(2)}` : "null");
  check("and says so rather than failing silently",
        warnings.some((w) => w.includes("too alike")));
}

console.log(failures === 0
  ? "\nPASS  the occlusal-end rule is correct and invariant to rigid motion"
  : `\nFAIL  ${failures} check(s) failed`);
process.exit(failures === 0 ? 0 : 1);
