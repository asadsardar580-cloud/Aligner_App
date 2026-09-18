// shadowRig.js — where the sun goes, how wide its frustum is, and where the
// catcher lies. All of the arithmetic, none of the three.js wiring.
//
// WHY THIS IS ITS OWN FILE. Until commit 2d372b4 the shadow rig had never
// executed once: `three.current` was reassigned after `sun` and `catcher` were
// attached to it, so `aimShadows` hit `if (!sun || !catcher) return;` on every
// call and returned silently. Fixing that did not fix a regression — it turned
// a path on for the first time. "Set Occlusal Plane makes the scene go black"
// is the behaviour of code nobody has ever watched run.
//
// So the arithmetic is pulled out here where it can be asserted in Node
// (verify-shadowrig.mjs) rather than only observed in a browser, and every
// value is checked for finiteness before it reaches a light. A viewport with
// no shadow is a cosmetic loss. A black one is a dead tool.
//
// Inputs and outputs are plain arrays, deliberately: this module imports
// nothing, so the check can run without three, without a WebGL context and
// without a DOM.

const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const mul = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1],
                         a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];
const finite3 = (a) => Array.isArray(a) && a.length === 3 && a.every(Number.isFinite);

function norm(a) {
  if (!finite3(a)) return null;
  const L = Math.hypot(a[0], a[1], a[2]);
  return L > 1e-9 ? [a[0] / L, a[1] / L, a[2] / L] : null;
}

/** Any unit vector perpendicular to `u`. Used only when the frame is partial. */
function anyPerp(u) {
  const seed = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
  return norm(sub(seed, mul(u, dot(seed, u))));
}

/** Quaternion [x,y,z,w] rotating +Z onto `u`. Three's setFromUnitVectors. */
function quatFromZTo(u) {
  const a = [0, 0, 1];
  const w = 1 + dot(a, u);
  if (w < 1e-8) {
    // u is antiparallel to +Z: any axis perpendicular to Z, half turn.
    return [1, 0, 0, 0];
  }
  const v = cross(a, u);
  const L = Math.hypot(v[0], v[1], v[2], w);
  return [v[0] / L, v[1] / L, v[2] / L, w / L];
}

/** How far past the geometry the near and far planes reach, as a multiple of
 *  the bounding radius. Section 3.1 of the brief asks for 1.5x. */
export const FRUSTUM_MARGIN = 1.5;
/** Half-width of the shadow camera's orthographic box, as a multiple of it. */
export const FRUSTUM_HALF = 1.6;
/** How far below the cast the catcher plane sits, beyond the bounding radius. */
export const CATCHER_DROP_MM = 2;

/**
 * Place the sun, aim it, size its frustum, and lay the catcher under the cast.
 *
 * `box` is {min:[x,y,z], max:[x,y,z]} in raw scanner coordinates. The axes are
 * the established occlusal basis; `u_occ` points OUT of the mouth for both
 * jaws (arch_frame.py:50), so nothing here needs to know which jaw it has.
 *
 * Returns null — never a partly-finite rig — when the inputs cannot produce
 * one. The caller is expected to treat null as "leave the lights alone".
 *
 * THREE THINGS THAT DIFFER FROM WHAT aimShadows USED TO DO:
 *
 * 1. near/far ARE COMPUTED. They were hardcoded 1 / 400 at construction and
 *    aimShadows only ever set left/right/top/bottom.
 *
 *    BEING PRECISE ABOUT WHEN 400 IS ACTUALLY WRONG, because the loose version
 *    of this claim does not survive measurement: one arch puts the catcher at
 *    156.4 mm of light depth, and two arches in occlusion at 180.9 mm. Both
 *    are comfortably inside 400. The crossover is a combined bounding radius
 *    of 101.6 mm (verify-shadowrig.mjs solves for it), which no single mouth
 *    reaches.
 *
 *    What reaches it is invariant 1. This app never re-centres a scan, so two
 *    arches captured against different scanner origins keep that separation in
 *    the combined box: 180 mm apart puts the catcher at 473.8 mm and the
 *    shadow silently stops. The hardcoded planes are not wrong for a mouth,
 *    they are unconnected to the scene — so they are now derived from the
 *    actual sun-to-catcher span instead of bracketing it by luck.
 * 2. THE SUN IS OFF-AXIS, at +u_occ + 0.35 u_sag + 0.25 u_tra. A light exactly
 *    along the view axis gives flat, relief-free shading: every surface the
 *    camera can see is lit head-on and the cusps lose their modelling.
 * 3. INTENSITY 1.2, up from 1.1, per section 3.1.
 */
export function computeShadowRig({ box, u_occ, u_sag, u_tra }) {
  if (!box || !finite3(box.min) || !finite3(box.max)) return null;
  const uOcc = norm(u_occ);
  if (!uOcc) return null;

  const centre = mul(add(box.min, box.max), 0.5);
  const size = sub(box.max, box.min);
  if (!finite3(centre) || !finite3(size)) return null;
  const radius = Math.hypot(size[0], size[1], size[2]) * 0.5;
  if (!(radius > 1e-6) || !Number.isFinite(radius)) return null;

  // A partial frame still gives a usable rig: the offsets exist to break the
  // head-on lighting, and any two perpendicular axes do that.
  const uTra = norm(u_tra) || anyPerp(uOcc);
  const uSag = norm(u_sag) || norm(cross(uOcc, uTra));
  if (!uTra || !uSag) return null;

  const sunDir = norm(add(add(uOcc, mul(uSag, 0.35)), mul(uTra, 0.25)));
  if (!sunDir) return null;

  const sunPosition = add(centre, mul(sunDir, radius * 3));
  const targetPosition = centre;
  const catcherPosition = add(centre, mul(uOcc, -(radius + CATCHER_DROP_MM)));

  // Depth along the light's own view direction, which is target - sun.
  const view = mul(sunDir, -1);
  const depth = (p) => dot(sub(p, sunPosition), view);
  let dMin = Infinity, dMax = -Infinity;
  for (let i = 0; i < 8; i++) {
    const corner = [i & 1 ? box.max[0] : box.min[0],
                    i & 2 ? box.max[1] : box.min[1],
                    i & 4 ? box.max[2] : box.min[2]];
    const d = depth(corner);
    if (d < dMin) dMin = d;
    if (d > dMax) dMax = d;
  }
  const dCatcher = depth(catcherPosition);
  dMin = Math.min(dMin, dCatcher);
  dMax = Math.max(dMax, dCatcher);

  const margin = radius * FRUSTUM_MARGIN;
  // A directional shadow camera is orthographic, so a small positive near is
  // only about depth precision, not about clipping the near geometry away.
  const near = Math.max(0.1, dMin - margin);
  const far = dMax + margin;
  const s = radius * FRUSTUM_HALF;

  const rig = {
    sunPosition, targetPosition, catcherPosition,
    catcherQuaternion: quatFromZTo(uOcc),
    intensity: 1.2,
    radius,
    frustum: { left: -s, right: s, top: s, bottom: -s, near, far },
    // THE CATCHER MUST NOT BE BIGGER THAN THE SHADOW MAP COVERS. It was a
    // fixed 400 x 400 mm plane while this box is 151.7 mm across on a real
    // arch, so 85.6% of it sampled the depth texture outside [0,1] UV — where
    // the clamped edge texel reads as SHADOWED. The plane also fills the frame
    // from the default camera (400 mm of plane against 143 mm of visible
    // height), so that misread is a full-screen 0.22-alpha black wash over
    // everything behind the cast. Sized to the frustum, every texel the
    // catcher samples is one the shadow map actually rendered.
    catcherSize: 2 * s,
  };

  const ok = finite3(rig.sunPosition) && finite3(rig.targetPosition)
    && finite3(rig.catcherPosition)
    && rig.catcherQuaternion.every(Number.isFinite)
    && Object.values(rig.frustum).every(Number.isFinite)
    && far > near;
  return ok ? rig : null;
}
