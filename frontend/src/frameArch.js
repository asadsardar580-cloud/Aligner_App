// frameArch.js — frame an arch comfortably WITHOUT touching its coordinates.
//
// The mesh stays in raw scanner space because bite registration depends on it
// and every click sent to the backend is expressed in it. So all framing is
// done by moving the CAMERA. Four separate defects are fixed here.
//
// 1. UP-VECTOR DEGENERACY (the reason navigation feels wrong)
//    The old placement put the camera at center.y - 2.5r with the default
//    up = (0,1,0). Measured on a synthetic arch in scanner coordinates, the
//    view direction sits 31 degrees from that up vector. OrbitControls
//    converts the camera offset to spherical coordinates about `up`, so a
//    view that close to the pole pins the polar angle: dragging barely
//    rotates, and the azimuth spins wildly as you cross over. Scanner axis
//    conventions are arbitrary — the TGN docs alone note Y-back and a shared
//    Z between jaws — so (0,1,0) has no reason to be the occlusal axis.
//
//    Fixed by deriving the occlusal axis from the geometry: an arch is a
//    flat-ish horseshoe, so the direction it is FLATTEST in is the occlusal
//    normal. That is the smallest-eigenvalue eigenvector of the vertex
//    covariance. On the synthetic arch the three spreads came out
//    [3.5, 9.1, 17.2] mm — the 3.5 axis is occlusal, unambiguously.
//
// 2. WHICH END OF THAT AXIS IS THE OCCLUSAL ONE (the upside-down mandible)
//    An eigenvector's sign is arbitrary, and the old code resolved it with
//    `if (up.dot(camera.position - centre) < 0) up.negate()`. On a fresh page
//    `camera.position` is still (0,0,0), so that test collapses to
//    `u · (−boundingCentre)` — it asks where the scan happens to sit relative
//    to the scanner's origin, which is not a fact about the patient.
//
//    Measured on case_lower.stl: the bounding centre projects 5.85 mm onto
//    that axis, so the rule picks the correct side — by 5.85 mm of luck.
//    TRANSLATING THE SCAN 6 mm FLIPS THE VIEW UPSIDE DOWN, and invariant 1
//    says a translation must change nothing at all. That is the defect,
//    whether or not it currently manifests on a particular file.
//
//    Fixed by `occlusalOrientation` below, which reads the sign out of the
//    shape itself and is therefore invariant to both translation and rigid
//    rotation. When the occlusal plane has actually been established the
//    heuristic is skipped entirely — see `frameArch`'s `frame` option, because
//    ground truth always wins over a rule that is merely usually right.
//
// 3. FIT DISTANCE
//    The old offset sat at 2.92 x radius. For a perspective camera the
//    distance that makes a sphere exactly fill the frame is r / sin(half),
//    where `half` is the SMALLER of the vertical and horizontal half-angles
//    — 2.61 x radius at fov 45 on a 16:9 canvas. Using the vertical angle
//    alone crops a wide arch on a tall window.
//
// 4. NEAR/FAR LEFT AT 0.1 / 5000
//    A 24 mm arch viewed from 60 mm with a far plane at 5000 wastes almost
//    the whole depth buffer, which shows up as z-fighting on the occlusal
//    surface where two cusps nearly touch. Both planes are now scaled to the
//    scene.
//
// Nothing here calls geom.center(), geom.scale() or mesh.position.set().

import * as THREE from "three";

/** Subsample a position attribute to a flat [x,y,z,...] array in float64.
 *
 * 20k points fixes the covariance axis to well under a degree and keeps this
 * instant on a 360k-vertex arch. */
function sampleFlat(geom) {
  const p = geom.attributes.position;
  const n = p.count;
  const step = Math.max(1, Math.floor(n / 20000));
  const out = new Float64Array(Math.ceil(n / step) * 3);
  let j = 0;
  for (let i = 0; i < n; i += step) {
    out[j++] = p.getX(i); out[j++] = p.getY(i); out[j++] = p.getZ(i);
  }
  return out.subarray(0, j);
}

/** Smallest-eigenvalue eigenvector of the vertex covariance, sign unresolved.
 *
 * `P` is a flat [x,y,z,...] array so this is testable without a WebGL context
 * or a BufferGeometry. Returns null rather than a guess when the covariance is
 * degenerate. */
function smallestAxis(P) {
  const n = P.length / 3;
  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += P[3 * i]; cy += P[3 * i + 1]; cz += P[3 * i + 2]; }
  cx /= n; cy /= n; cz /= n;

  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (let i = 0; i < n; i++) {
    const dx = P[3 * i] - cx, dy = P[3 * i + 1] - cy, dz = P[3 * i + 2] - cz;
    xx += dx * dx; xy += dx * dy; xz += dx * dz;
    yy += dy * dy; yz += dy * dz; zz += dz * dz;
  }
  const C = [[xx / n, xy / n, xz / n], [xy / n, yy / n, yz / n], [xz / n, yz / n, zz / n]];

  // inverse power iteration: repeatedly solve C x = b to converge on the
  // SMALLEST eigenvector. A plain power iteration would give the largest.
  const solve = (A, b) => {
    const M = A.map((row, i) => [...row, b[i]]);
    for (let c = 0; c < 3; c++) {
      let piv = c;
      for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      if (Math.abs(M[c][c]) < 1e-12) M[c][c] = 1e-12;
      for (let r = 0; r < 3; r++) {
        if (r === c) continue;
        const f = M[r][c] / M[c][c];
        for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k];
      }
    }
    return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
  };

  const tr = (xx + yy + zz) / n;
  for (let i = 0; i < 3; i++) C[i][i] += tr * 1e-9;   // keep it invertible
  let v = [1, 1, 1];
  for (let it = 0; it < 40; it++) {
    v = solve(C, v);
    const L = Math.hypot(v[0], v[1], v[2]) || 1;
    v = [v[0] / L, v[1] / L, v[2] / L];
    // Bail out rather than iterate on poison. The ridge term keeps C
    // invertible for real scans, but a degenerate mesh (all vertices coplanar
    // to float precision, or a single point) still divides through by ~0, and
    // NaN here propagates straight into camera.up — where it produces a black
    // viewport and no error, because every comparison against NaN is false.
    if (!v.every(Number.isFinite)) return null;
  }
  const L = Math.hypot(v[0], v[1], v[2]);
  if (!(L > 1e-6) || !Number.isFinite(L)) return null;
  return [v[0] / L, v[1] / L, v[2] / L];
}

/** What fraction of the vertices, by COUNT, forms each end's slab.
 *
 *  By count and not by extent. Taking the outer tenth of the axial EXTENT puts
 *  119 vertices in the occlusal slab against 2695 in the tissue slab on
 *  case_lower.stl — a single stray cusp tip stretches the extent and starves
 *  the slab it defines. Equal counts make the two spreads comparable. */
const SLAB_FRACTION = 0.10;
/** Below this the skewness is numerically indistinguishable from zero and
 *  carries no sign to read; the spread signal decides instead. */
const SKEW_FLOOR = 1e-3;

/**
 * Which way along the flattest axis is OCCLUSAL — from the shape alone.
 *
 * `P` is a flat [x,y,z,...] array of vertex positions. The returned axis is a
 * unit 3-vector pointing occlusally (out of the mouth), the same convention as
 * arch_frame.py's u_occ — which is why the real u_occ can be swapped in for it
 * the moment one exists.
 *
 * TWO INDEPENDENT SIGNALS, measured on case_lower.stl (62544 sampled vertices,
 * 18.31 mm of axial extent). Projecting onto the flattest PCA axis:
 *
 *                                       occlusal end      tissue end
 *   skewness of the projection          +0.697            -0.697
 *   radial spread, outer decile (sigma) 5.86 mm           3.14 mm
 *
 * SKEWNESS: the occlusal end is the long thin tail. Cusp tips are a sparse
 * scatter reaching well past the body of the cast, while the gingival end is
 * blunt and crowded, so the third moment leans occlusally.
 *
 * RADIAL SPREAD: the occlusal end is the one whose radii VARY, and this is the
 * opposite of what it sounds like. The occlusal slab is the handful of highest
 * points — incisal edges far forward of the arch centroid, molar cusps much
 * nearer it — so its distance-from-axis ranges over 5.86 mm of sigma. The
 * tissue slab is a near-continuous band running right around the periphery at
 * an almost constant radius: 3.14 mm. (A previous pass had this backwards, on
 * the intuition that "cusp tips form a thin ring". They do not form a ring at
 * all; the gingival margin does.)
 *
 * Both signals point the same way on the real scan, which is what makes this a
 * rule rather than a guess.
 *
 * Skewness is primary and is the documented tie-break; the spread ratio only
 * corroborates, and a disagreement is logged rather than acted on. Neither
 * moment depends on where the scan sits in scanner space or how it is rotated,
 * which is the whole point — see note 2 at the top of this file.
 */
export function occlusalOrientation(P) {
  const n = Math.floor(P.length / 3);
  const fail = {
    axis: null, skew: 0, spreadOcclusal: NaN, spreadTissue: NaN,
    agree: false, degenerate: true,
  };
  if (n < 12) return fail;

  const axis = smallestAxis(P);
  if (!axis) return fail;

  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += P[3 * i]; cy += P[3 * i + 1]; cz += P[3 * i + 2]; }
  cx /= n; cy /= n; cz /= n;

  // Projection onto the axis, plus the perpendicular distance to it. Both are
  // measured from the centroid, so both are translation-invariant.
  const T = new Float64Array(n);
  const R = new Float64Array(n);
  let tmin = Infinity, tmax = -Infinity;
  for (let i = 0; i < n; i++) {
    const dx = P[3 * i] - cx, dy = P[3 * i + 1] - cy, dz = P[3 * i + 2] - cz;
    const t = dx * axis[0] + dy * axis[1] + dz * axis[2];
    T[i] = t;
    R[i] = Math.hypot(dx - t * axis[0], dy - t * axis[1], dz - t * axis[2]);
    if (t < tmin) tmin = t;
    if (t > tmax) tmax = t;
  }
  if (!Number.isFinite(tmin) || !Number.isFinite(tmax) || tmax - tmin < 1e-9) return fail;

  let m2 = 0, m3 = 0;
  for (let i = 0; i < n; i++) { const t = T[i]; m2 += t * t; m3 += t * t * t; }
  m2 /= n; m3 /= n;
  const skew = m2 > 1e-12 ? m3 / Math.pow(m2, 1.5) : 0;

  // Radial spread = standard deviation of the perpendicular distance inside
  // each outer decile slab, the two slabs taken by equal COUNT. NOT the mean
  // radius: the two ends' mean radii differ by 0.67 mm on the real scan, which
  // is inside the noise, while their sigmas differ by a factor of 1.87.
  const order = Array.from({ length: n }, (_, i) => i).sort((a, b) => T[a] - T[b]);
  const k = Math.max(3, Math.floor(n * SLAB_FRACTION));
  const slabSpread = (idx) => {
    if (idx.length < 3) return NaN;
    let s = 0, s2 = 0;
    for (const i of idx) { s += R[i]; s2 += R[i] * R[i]; }
    const mean = s / idx.length;
    return Math.sqrt(Math.max(0, s2 / idx.length - mean * mean));
  };
  const spreadPlus = slabSpread(order.slice(n - k));
  const spreadMinus = slabSpread(order.slice(0, k));

  const skewSign = skew >= 0 ? 1 : -1;
  const spreadUsable = Number.isFinite(spreadPlus) && Number.isFinite(spreadMinus);
  // LARGER spread is the occlusal end — see the table above.
  const spreadSign = spreadUsable ? (spreadPlus > spreadMinus ? 1 : -1) : skewSign;
  const agree = spreadSign === skewSign;

  // Skewness decides, except when it is numerically absent — a perfectly
  // symmetric projection has no tail to read, and its sign would be noise.
  let sign = Math.abs(skew) >= SKEW_FLOOR ? skewSign : spreadSign;
  if (!agree && Math.abs(skew) >= SKEW_FLOOR && spreadUsable) {
    console.warn(
      "[Arch Framing] Occlusal-end signals disagree: skewness "
      + skew.toFixed(3) + " says one end, radial spread ("
      + spreadPlus.toFixed(2) + " mm vs " + spreadMinus.toFixed(2)
      + " mm) says the other. Using skewness. Check the view is not inverted.");
    sign = skewSign;
  }

  return {
    axis: [axis[0] * sign, axis[1] * sign, axis[2] * sign],
    skew: skew * sign,                                   // >= 0 once oriented
    spreadOcclusal: sign > 0 ? spreadPlus : spreadMinus,
    spreadTissue: sign > 0 ? spreadMinus : spreadPlus,
    agree,
    degenerate: false,
  };
}

/** Below this the two ends are too alike to call, and the roll is left alone. */
const ANTERIOR_MIN_RATIO = 1.15;

/**
 * Which in-plane direction is ANTERIOR — again from the shape alone.
 *
 * WHY THIS IS NEEDED AT ALL. `camera.up` used to be the occlusal axis while the
 * camera sat 34.7 degrees off that same axis, so the screen's up direction was
 * whatever survived projecting u_occ onto the view plane: on the real scan,
 * roughly the scanner's −X. That is an arbitrary axis, and a mandible rolled
 * 180 degrees in-plane reads exactly like a maxilla — which is the defect this
 * fixes. Reported from a browser, diagnosed from the console line this file
 * now prints, and then measured here.
 *
 * THE SIGNAL IS THE ARCH'S OWN WIDTH, and it is anatomy rather than a fixture
 * artefact: a dental arch is widest between the molars and narrowest at the
 * incisors (intermolar ~55 mm against intercanine ~35 mm), for every human
 * arch, upper or lower, and it stays true on a partial scan. So of the two
 * in-plane axes, the SAGITTAL one is whichever shows the greater width
 * contrast between its two ends, and ANTERIOR is the narrow end.
 *
 * Measured on the real scan, outer quartiles, width taken along the other axis:
 *
 *   axis A:  38.6 mm   vs  71.5 mm    ratio 1.85   <- sagittal, narrow end +A
 *   axis B:  44.3 mm   vs  43.3 mm    ratio 1.02
 *
 * and +A is confirmed anterior by two independent checks: the point centroid
 * sits 3.09 mm anterior of the mid-extent (a U opening posteriorly), and
 * slicing across the arch gives two arms posteriorly against one blob
 * anteriorly.
 *
 * Returns null when the contrast is too weak to call, so the caller can leave
 * the roll where it was rather than assert a coin flip.
 */
export function archAnterior(P, uOcc) {
  const n = Math.floor(P.length / 3);
  if (n < 32 || !uOcc || !uOcc.every(Number.isFinite)) return null;

  const ul = Math.hypot(uOcc[0], uOcc[1], uOcc[2]);
  if (!(ul > 1e-9)) return null;
  const u = [uOcc[0] / ul, uOcc[1] / ul, uOcc[2] / ul];

  // Any orthonormal pair in the occlusal plane, then rotated onto the in-plane
  // principal axes so the result does not depend on the arbitrary seed.
  const sd = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  const d0 = sd[0] * u[0] + sd[1] * u[1] + sd[2] * u[2];
  let e1 = [sd[0] - u[0] * d0, sd[1] - u[1] * d0, sd[2] - u[2] * d0];
  const e1l = Math.hypot(e1[0], e1[1], e1[2]);
  if (!(e1l > 1e-9)) return null;
  e1 = [e1[0] / e1l, e1[1] / e1l, e1[2] / e1l];
  let e2 = [u[1] * e1[2] - u[2] * e1[1], u[2] * e1[0] - u[0] * e1[2],
            u[0] * e1[1] - u[1] * e1[0]];

  let cx = 0, cy = 0, cz = 0;
  for (let i = 0; i < n; i++) { cx += P[3 * i]; cy += P[3 * i + 1]; cz += P[3 * i + 2]; }
  cx /= n; cy /= n; cz /= n;

  const X = new Float64Array(n), Y = new Float64Array(n);
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = P[3 * i] - cx, dy = P[3 * i + 1] - cy, dz = P[3 * i + 2] - cz;
    const a = dx * e1[0] + dy * e1[1] + dz * e1[2];
    const b = dx * e2[0] + dy * e2[1] + dz * e2[2];
    X[i] = a; Y[i] = b;
    sxx += a * a; sxy += a * b; syy += b * b;
  }
  // 2x2 symmetric eigenvector, closed form: rotate (e1,e2) onto the principal
  // in-plane axes so "axis A" and "axis B" mean the same thing on a rotated
  // copy of the same scan.
  const th = 0.5 * Math.atan2(2 * sxy / n, (sxx - syy) / n);
  const ct = Math.cos(th), st = Math.sin(th);
  const A = new Float64Array(n), B = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    A[i] = X[i] * ct + Y[i] * st;
    B[i] = -X[i] * st + Y[i] * ct;
  }

  // Width of each outer quartile, measured along the OTHER in-plane axis.
  const contrast = (T, O) => {
    const idx = Array.from({ length: n }, (_, i) => i).sort((a, b) => T[a] - T[b]);
    const k = Math.max(8, Math.floor(n / 4));
    const spanOf = (list) => {
      let lo = Infinity, hi = -Infinity;
      for (const i of list) { if (O[i] < lo) lo = O[i]; if (O[i] > hi) hi = O[i]; }
      return hi - lo;
    };
    const wLow = spanOf(idx.slice(0, k));
    const wHigh = spanOf(idx.slice(n - k));
    if (!(wLow > 1e-9) || !(wHigh > 1e-9)) return null;
    return { wLow, wHigh, ratio: Math.max(wLow, wHigh) / Math.min(wLow, wHigh),
             narrowIsHigh: wHigh < wLow };
  };

  const ca = contrast(A, B), cb = contrast(B, A);
  if (!ca || !cb) return null;
  const sagIsA = ca.ratio >= cb.ratio;
  const win = sagIsA ? ca : cb;
  if (win.ratio < ANTERIOR_MIN_RATIO) {
    console.warn(
      "[Arch Framing] The two ends of this arch are too alike to tell front "
      + "from back (width ratio " + win.ratio.toFixed(2) + "). Leaving the roll "
      + "as it is; set the occlusal plane to orient it exactly.");
    return null;
  }

  const sign = win.narrowIsHigh ? 1 : -1;      // anterior is the NARROW end
  const rot = (t) => (sagIsA
    ? [e1[0] * ct + e2[0] * st, e1[1] * ct + e2[1] * st, e1[2] * ct + e2[2] * st]
    : [-e1[0] * st + e2[0] * ct, -e1[1] * st + e2[1] * ct, -e1[2] * st + e2[2] * ct]
  ).map((c) => c * t);
  const anterior = rot(sign);

  // Corroboration: a U opening posteriorly puts the centroid anterior of the
  // mid-extent. Logged on disagreement, not acted on — the width contrast is
  // anatomy, this is a weaker second opinion (3.09 mm on the real scan).
  const T = sagIsA ? A : B;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) { if (T[i] < lo) lo = T[i]; if (T[i] > hi) hi = T[i]; }
  const midAhead = -sign * ((lo + hi) / 2) > 0;      // mid-extent is POSTERIOR of 0
  if (!midAhead) {
    console.warn(
      "[Arch Framing] Width says the anterior is one way, the centroid offset "
      + "says the other. Using width (the intermolar/intercanine contrast is "
      + "anatomy). Check the arch is not rolled 180 degrees.");
  }

  const transverse = [u[1] * anterior[2] - u[2] * anterior[1],
                      u[2] * anterior[0] - u[0] * anterior[2],
                      u[0] * anterior[1] - u[1] * anterior[0]];
  if (!anterior.every(Number.isFinite) || !transverse.every(Number.isFinite)) return null;
  return { anterior, transverse, ratio: win.ratio, agree: midAhead };
}

/** A finite unit vector from a serialised triple, or null. */
function unit(a) {
  if (!Array.isArray(a) || a.length !== 3 || !a.every(Number.isFinite)) return null;
  const v = new THREE.Vector3().fromArray(a);
  return v.lengthSq() > 1e-12 ? v.normalize() : null;
}

/**
 * Frame `geom` for the given camera/controls. Returns the derived axes so the
 * caller can offer buccal / occlusal / lingual view presets.
 *
 * `jaw` is "lower" | "upper" | null and only decides the ROLL, never the side
 * the camera sits on: u_occ points out of the mouth for BOTH jaws
 * (arch_frame.py:50 resolves its sign against the arch centroid), so "put the
 * camera on the +u_occ side" is jaw-independent. The brief asks for an
 * upper/lower branch on the camera position and it genuinely does not need one.
 *
 * `frame` is an established occlusal basis ({u_occ, u_sag, u_tra}) once the
 * clinician has set the plane. Given one, the geometric heuristic is skipped
 * and the roll becomes clinical: the anterior teeth sit at the bottom of the
 * screen for a mandible and at the top for a maxilla, which is how an opposing
 * pair reads in the mouth.
 */
export function frameArch(geom, camera, controls, renderer,
                          { tilt = 0.57, jaw = null, frame = null } = {}) {
  geom.computeBoundingSphere();
  const bs = geom.boundingSphere;
  const r = Math.max(bs.radius, 1e-3);

  const uOcc = frame ? unit(frame.u_occ) : null;
  const uSag = frame ? unit(frame.u_sag) : null;
  const uTra = frame ? unit(frame.u_tra) : null;
  const grounded = Boolean(uOcc && uSag && uTra);

  // Ground truth when it exists, the shape rule when it does not.
  const flat = grounded ? null : sampleFlat(geom);
  const derived = grounded ? null : occlusalOrientation(flat);
  const up = grounded
    ? uOcc.clone()
    : (derived.axis ? new THREE.Vector3(...derived.axis) : new THREE.Vector3(0, 0, 1));

  // The anterior direction, so the ROLL is anatomical from the first frame and
  // not only after three landmark clicks. See archAnterior: the load-time view
  // now agrees with the post-occlusal-plane view instead of contradicting it.
  const ant = grounded ? null : archAnterior(flat, up.toArray());
  const uAnt = grounded ? uSag : (ant ? new THREE.Vector3(...ant.anterior) : null);

  // The in-plane axis the camera swings around for a three-quarter view: the
  // transverse axis, so the swing is lateral and the anterior-posterior axis
  // stays vertical on screen. Only when neither a frame nor a readable arch
  // shape gives one does the arbitrary seed stand.
  const seed = Math.abs(up.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  const inPlane = grounded
    ? uTra.clone()
    : (ant ? new THREE.Vector3(...ant.transverse).normalize()
           : seed.clone().sub(up.clone().multiplyScalar(seed.dot(up))).normalize());

  // fit distance on the LIMITING axis
  const halfV = THREE.MathUtils.degToRad(camera.fov) / 2;
  const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
  const dist = r / Math.sin(Math.min(halfV, halfH));

  const dir = up.clone().multiplyScalar(Math.sqrt(1 - tilt * tilt))
    .add(inPlane.clone().multiplyScalar(tilt)).normalize();

  // Roll: anterior DOWN for a mandible, UP for a maxilla, so an opposing pair
  // reads the way it does in the mouth. This used to apply only when a frame
  // existed, and the fallback — camera.up = the occlusal axis — was the bug:
  // the camera sits 34.7 degrees off that same axis, so screen-up collapsed to
  // whatever was left of u_occ after projection (about scanner −X on the real
  // scan) and the arch landed at an arbitrary roll. Now the shape supplies an
  // anterior direction too, and only a genuinely unreadable arch falls back.
  camera.up.copy(uAnt
    ? uAnt.clone().normalize().multiplyScalar(jaw === "lower" ? -1 : 1)
    : up);
  camera.position.copy(bs.center).add(dir.multiplyScalar(dist));
  camera.near = dist / 100;
  camera.far = dist * 4 + r * 4;
  camera.updateProjectionMatrix();       // without this, near/far are ignored

  controls.target.copy(bs.center);
  controls.minDistance = r * 0.15;       // stop the clinician flying inside a cusp
  controls.maxDistance = dist * 3;
  controls.zoomSpeed = 0.8;
  controls.update();

  // `orientation` carries the numbers the sign decision was made from, null
  // when a real frame made the decision instead. It exists so an inverted view
  // can be diagnosed from a console line rather than from a description of
  // what the screen looks like — there is no browser where this is developed.
  return { center: bs.center.clone(), radius: r, up, inPlane, dist, grounded,
           orientation: derived, anterior: ant };
}

/**
 * Keep the renderer matched to its container.
 *
 * The original sized the renderer once from mount.clientWidth inside the
 * mount effect. In a CSS grid that runs before layout has settled, so the
 * drawing buffer can be created at the wrong size and then never corrected —
 * the canvas is stretched to fit and everything looks small and slightly
 * soft, with zoom that does not match the cursor. Returns a disposer.
 */
export function attachResize(mount, camera, renderer) {
  const apply = () => {
    const w = mount.clientWidth || 1, h = mount.clientHeight || 1;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };
  apply();
  const ro = new ResizeObserver(apply);
  ro.observe(mount);
  return () => ro.disconnect();
}

/**
 * Raycast across EVERY loaded arch and return the nearest hit.
 *
 * Testing only the active arch sounds like the strict, safe thing and is
 * exactly wrong: with both arches in occlusion a ray aimed at the mandible
 * passes through it and strikes whichever arch is active behind it, so the
 * clinician clicks a lower molar and the upper one lights up.
 */
export function pickAcrossArches(event, mount, camera, raycaster, arches) {
  const rect = mount.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((event.clientX - rect.left) / rect.width) * 2 - 1,
    -((event.clientY - rect.top) / rect.height) * 2 + 1
  );
  raycaster.setFromCamera(ndc, camera);
  const meshes = Object.entries(arches)
    .filter(([, a]) => a?.mesh)
    .map(([name, a]) => { a.mesh.userData.archName = name; return a.mesh; });
  const hits = raycaster.intersectObjects(meshes, false);
  if (!hits.length) return null;
  const h = hits[0];                                   // nearest, across all arches
  return { archName: h.object.userData.archName, point: h.point, faceIndex: h.faceIndex };
}
