/**
 * Cross-language kinematics check.  Run:  node verify-kinematics.mjs
 *
 * WHY THIS EXISTS. The sidebar's numeric inputs are two-way bound: the gizmo
 * writes clinical values out, and typed values are turned back into a
 * transform. That only works if setClinical() and read() are exact inverses AND
 * both agree with the backend's cg.kinematic_matrix — the browser drives the
 * viewport, the server drives the exported STL, and a silent disagreement
 * between them means the printed aligner does not match what was on screen.
 *
 * The golden matrices below were produced by cg.kinematic_matrix (row-major).
 * test_kinematics_frame.py::test_matches_the_javascript_golden_matrices asserts
 * Python still produces exactly these numbers, so the two suites are pinned to
 * one shared reference. If either side drifts, one of them goes red.
 *
 * Two real bugs were caught here and are pinned by this file:
 *   - read() used Euler order ZYX while the backend composes Rx.Ry.Rz ('XYZ'),
 *     round-tripping (-12.5, 7.5, -4.0) as (-11.8, 8.5, -5.7): 1.71 deg of
 *     pure decomposition error, invisible on single-axis moves.
 *   - read() reported the delta's raw translation column, which is c + t - R.c,
 *     not t — a pure 8 deg tip showed 0.94mm of mesiodistal translation.
 */
import * as THREE from "three";

const FRAME = {
  u_md: [0.935563726474988, 0.25837028799393946, 0.2407598554289369],
  u_bl: [-0.28340921275056274, 0.9560359997469121, 0.07532851595530501],
  u_oa: [-0.2107124387223975, -0.13870818818603006, 0.9676571224859604],
};
const C_RES = [1.5, -2.0, -7.0];

const CASES = [
  { k: { tip_deg: 8.0, torque_deg: 0, rotation_deg: 0, d_md: 0, d_bl: 0, d_oa: 0 },
    M: [0.9910497450693709, -0.013120564189333068, 0.1328467297049066, 0.9171113619516237,
        0.007846842123106063, 0.99916310094437, 0.04014380126402818, 0.26756254755227804,
        -0.13326225972356157, -0.038742076694265284, 0.9903232914693998, 0.054672276482610194,
        0, 0, 0, 1] },
  { k: { tip_deg: 8.0, torque_deg: -5.0, rotation_deg: 3.0, d_md: 0.4, d_bl: -0.2, d_oa: 0.3 },
    M: [0.993606916899914, -0.04194230274913117, 0.10481477915230929, 1.0271020747414203,
        0.028206835731153637, 0.9912109636078271, 0.129248597832543, 0.7153803172391093,
        -0.10931454206409535, -0.12546580754828365, 0.9860571291915974, 0.18697547806706005,
        0, 0, 0, 1] },
  { k: { tip_deg: -12.5, torque_deg: 7.5, rotation_deg: -4.0, d_md: -0.6, d_bl: 0.35, d_oa: -0.25 },
    M: [0.9857833847971419, 0.04697988817187355, -0.16131958456817322, -1.6218057434962687,
        -0.07922406915713918, 0.9766499253545972, -0.19969594430294357, -1.111468181514139,
        0.14817106709501204, 0.20963733782944718, 0.9664872070874542, -0.3975766886659393,
        0, 0, 0, 1] },
];

// ---- the algebra attach() installs, verbatim from toothGizmo.js -----------
const axes = {
  md: new THREE.Vector3().fromArray(FRAME.u_md).normalize(),
  bl: new THREE.Vector3().fromArray(FRAME.u_bl).normalize(),
  oa: new THREE.Vector3().fromArray(FRAME.u_oa).normalize(),
};
const c = new THREE.Vector3().fromArray(C_RES);
const restQuat = new THREE.Quaternion().setFromRotationMatrix(
  new THREE.Matrix4().makeBasis(axes.md, axes.bl, axes.oa));
const rest = new THREE.Matrix4().compose(c, restQuat, new THREE.Vector3(1, 1, 1));
const restInv = rest.clone().invert();
const D2R = THREE.MathUtils.degToRad, R2D = THREE.MathUtils.radToDeg;

function deltaFromClinical(k) {
  const R = new THREE.Matrix4().makeRotationAxis(axes.md, D2R(k.torque_deg || 0))
    .multiply(new THREE.Matrix4().makeRotationAxis(axes.bl, D2R(k.tip_deg || 0)))
    .multiply(new THREE.Matrix4().makeRotationAxis(axes.oa, D2R(k.rotation_deg || 0)));
  const t = axes.md.clone().multiplyScalar(k.d_md || 0)
    .add(axes.bl.clone().multiplyScalar(k.d_bl || 0))
    .add(axes.oa.clone().multiplyScalar(k.d_oa || 0));
  return new THREE.Matrix4()
    .makeTranslation(c.x + t.x, c.y + t.y, c.z + t.z)
    .multiply(R)
    .multiply(new THREE.Matrix4().makeTranslation(-c.x, -c.y, -c.z));
}

function read(pivotMatrix) {
  const d = pivotMatrix.clone().multiply(restInv);
  const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scl = new THREE.Vector3();
  d.decompose(pos, quat, scl);
  const local = restQuat.clone().invert().multiply(quat).multiply(restQuat);
  const eul = new THREE.Euler().setFromQuaternion(local, "XYZ");
  const md = new THREE.Vector3(1, 0, 0).applyQuaternion(restQuat);
  const bl = new THREE.Vector3(0, 1, 0).applyQuaternion(restQuat);
  const oa = new THREE.Vector3(0, 0, 1).applyQuaternion(restQuat);
  const tLocal = pos.clone().sub(c).add(c.clone().applyQuaternion(quat));
  return {
    tip_deg: +R2D(eul.y).toFixed(2), torque_deg: +R2D(eul.x).toFixed(2),
    rotation_deg: +R2D(eul.z).toFixed(2),
    d_md: +tLocal.dot(md).toFixed(3), d_bl: +tLocal.dot(bl).toFixed(3),
    d_oa: +tLocal.dot(oa).toFixed(3),
    scale_check: +(scl.length() / Math.sqrt(3)).toFixed(6),
  };
}

let worstM = 0, worstAng = 0, worstMm = 0, worstScale = 0;
for (const { k, M } of CASES) {
  const delta = deltaFromClinical(k);
  const e = delta.elements;                       // three is COLUMN-major
  const rowMajor = [e[0], e[4], e[8], e[12], e[1], e[5], e[9], e[13],
                    e[2], e[6], e[10], e[14], e[3], e[7], e[11], e[15]];
  worstM = Math.max(worstM, ...rowMajor.map((v, i) => Math.abs(v - M[i])));

  const r = read(delta.clone().multiply(rest));   // setClinical, then read back
  worstAng = Math.max(worstAng, Math.abs(r.tip_deg - k.tip_deg),
                      Math.abs(r.torque_deg - k.torque_deg),
                      Math.abs(r.rotation_deg - k.rotation_deg));
  worstMm = Math.max(worstMm, Math.abs(r.d_md - k.d_md),
                     Math.abs(r.d_bl - k.d_bl), Math.abs(r.d_oa - k.d_oa));
  worstScale = Math.max(worstScale, Math.abs(r.scale_check - 1));
}

console.log(`setClinical vs cg.kinematic_matrix : max |diff| ${worstM.toExponential(2)}`);
console.log(`setClinical -> read round trip     : ${worstAng.toFixed(4)} deg, ${worstMm.toFixed(4)} mm`);
console.log(`scale drift (must stay 1.0)        : ${worstScale.toExponential(2)}`);

// ===========================================================================
// STAGING
// ---------------------------------------------------------------------------
// Stage k poses the tooth at clinical x k/N, absolute from T0 — it is NOT a
// lerp of the committed 4x4, and the difference is not stylistic. The 3x3 block
// of (1-t)I + tR is not orthonormal for any t strictly between 0 and 1, so a
// matrix lerp shears and scales the crown at every intermediate stage: a tooth
// that ends up the right shape passes through 30 stages of the wrong one, and
// the printed trays are cut from those stages. Both are measured below.
// ===========================================================================

const STAGE_CASE = { tip_deg: -12.5, torque_deg: 7.5, rotation_deg: -4.0,
                     d_md: 0.8, d_bl: -0.45, d_oa: 1.2 };
const N = 24;

const clinicalAtStage = (k, n, c) => Object.fromEntries(
  ["tip_deg", "torque_deg", "rotation_deg", "d_md", "d_bl", "d_oa"]
    .map((key) => [key, (c[key] || 0) * (n > 0 ? k / n : 0)]));

/** Largest deviation of R'R from the identity, and of det(R) from +1. */
function rigidError(M) {
  const e = M.elements;
  const R = [[e[0], e[4], e[8]], [e[1], e[5], e[9]], [e[2], e[6], e[10]]];
  let orth = 0;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      const dot = R[0][i] * R[0][j] + R[1][i] * R[1][j] + R[2][i] * R[2][j];
      orth = Math.max(orth, Math.abs(dot - (i === j ? 1 : 0)));
    }
  }
  const det = R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1])
            - R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0])
            + R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);
  return { orth, detErr: Math.abs(det - 1) };
}

const full = deltaFromClinical(STAGE_CASE);
let worstOrth = 0, worstDet = 0, worstLerpOrth = 0;

const stage0 = deltaFromClinical(clinicalAtStage(0, N, STAGE_CASE));
const stage0Err = Math.max(...stage0.elements.map((v, i) =>
  Math.abs(v - new THREE.Matrix4().elements[i])));

for (let k = 0; k <= N; k++) {
  const M = deltaFromClinical(clinicalAtStage(k, N, STAGE_CASE));
  const r = rigidError(M);
  worstOrth = Math.max(worstOrth, r.orth);
  worstDet = Math.max(worstDet, r.detErr);

  // What a naive matrix lerp would have produced at the same stage, so the
  // claim "interpolating the 4x4 is not a rotation" is measured, not asserted.
  const t = k / N;
  const L = new THREE.Matrix4();
  L.elements = full.elements.map((v, i) =>
    (1 - t) * new THREE.Matrix4().elements[i] + t * v);
  worstLerpOrth = Math.max(worstLerpOrth, rigidError(L).orth);
}

const stageNErr = Math.max(...deltaFromClinical(clinicalAtStage(N, N, STAGE_CASE))
  .elements.map((v, i) => Math.abs(v - full.elements[i])));

console.log(`stage 0 is exact identity          : ${stage0Err.toExponential(2)}`);
console.log(`stage N reproduces the commitment  : ${stageNErr.toExponential(2)}`);
console.log(`every stage stays rigid (R'R-I)    : ${worstOrth.toExponential(2)}, det-1 ${worstDet.toExponential(2)}`);
console.log(`  a 4x4 lerp would have reached    : ${worstLerpOrth.toExponential(2)} — not a rotation`);

const stagingOk = stage0Err < 1e-12 && stageNErr < 1e-12
               && worstOrth < 1e-12 && worstDet < 1e-12
               && worstLerpOrth > 1e-3;      // the bad path must measurably fail

const ok = worstM < 1e-12 && worstAng <= 0.005 && worstMm <= 0.0005 && worstScale < 1e-9
        && stagingOk;
console.log(ok ? "PASS  two-way binding and staging are exact and agree with the backend"
                : "FAIL  browser and backend disagree");
process.exit(ok ? 0 : 1);
