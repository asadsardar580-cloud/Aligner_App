// toothGizmo.js — a 3D transform gizmo that rotates about the centre of
// resistance, with rings that mean tip, torque and rotation.
//
// WHY A NAIVE TransformControls ATTACHMENT IS WRONG
// --------------------------------------------------
// Dropping TransformControls onto the crown mesh gives you a gizmo that
// rotates about the mesh ORIGIN (scanner origin, potentially hundreds of mm
// away) along WORLD axes. Dragging the green ring would then apply a rotation
// that is not tip, not torque, and not rotation -- it is a diagonal twist in
// scanner space, which is exactly the failure 3Shape's occlusal-plane step
// exists to prevent.
//
// THE FIX: A PIVOT OBJECT
// -----------------------
// An empty Object3D is placed AT C_res, oriented so its local X/Y/Z are the
// tooth's mesiodistal / buccolingual / long axes. The crown is parented to it
// and pre-multiplied by the inverse of the pivot's rest matrix, so at rest the
// composite is identity and the crown sits exactly where the scan put it.
// TransformControls then runs in "local" space and its rings ARE the clinical
// axes. Rotation happens about C_res because that is where the pivot is.
//
// AXIS CONVENTION -- this bit was wrong once and cost real time:
//   tip      = rotation about the BUCCOLINGUAL axis   (mesiodistal crown incl.)
//   torque   = rotation about the MESIODISTAL axis    (buccolingual crown incl.)
//   rotation = rotation about the LONG axis
// An earlier version had tip about mesiodistal. The tooth moved smoothly and
// looked entirely correct while pivoting about the wrong anatomical axis.

import * as THREE from "three";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";

/**
 * Quaternion whose local X,Y,Z map to (u_md, u_bl, u_oa).
 *
 * The basis is VERIFIED before makeBasis, not assumed. setFromRotationMatrix
 * presumes a pure rotation; hand it a reflection (det = -1) or a skewed basis
 * and it returns a quaternion without complaint, and the tooth then swings
 * along axes that are not tip, torque or rotation. That failure is visually
 * indistinguishable from the C_res bug this gizmo was fixed for, so it fails
 * loudly here instead of rendering nonsense.
 */
function quatFromFrame(frame) {
  // Finite check FIRST. Every comparison against NaN is false, so the
  // orthogonality and determinant tests below both silently PASS a NaN frame —
  // and a missing or renamed key gives exactly that: fromArray(undefined)
  // yields NaN, normalize() yields NaN, the pivot matrix is NaN, and the crown
  // vanishes or flies off. The backend contract must be checked, not assumed.
  for (const key of ["u_md", "u_bl", "u_oa"]) {
    const a = frame?.[key];
    if (!Array.isArray(a) || a.length !== 3 || !a.every(Number.isFinite)) {
      throw new Error(
        `Anatomical frame is missing a finite ${key} — refusing to attach the gizmo. ` +
        `Got ${JSON.stringify(a)}.`);
    }
  }

  const md = new THREE.Vector3().fromArray(frame.u_md).normalize();
  const bl = new THREE.Vector3().fromArray(frame.u_bl).normalize();
  const oa = new THREE.Vector3().fromArray(frame.u_oa).normalize();

  const orth = Math.max(Math.abs(md.dot(bl)), Math.abs(md.dot(oa)), Math.abs(bl.dot(oa)));
  if (orth > 1e-4) {
    throw new Error(
      `Anatomical frame is not orthogonal (worst |dot| = ${orth.toExponential(2)}). ` +
      `Rotating about skewed axes shears the tooth on every slider move — refusing to attach.`);
  }
  // det = u_md . (u_bl x u_oa); +1 for a right-handed frame.
  const det = md.dot(new THREE.Vector3().crossVectors(bl, oa));
  if (Math.abs(det - 1) > 1e-3) {
    throw new Error(
      `Anatomical frame is not a right-handed orthonormal basis (det = ${det.toFixed(6)}). ` +
      `makeBasis would yield a garbage quaternion — refusing to attach.`);
  }

  const m = new THREE.Matrix4().makeBasis(md, bl, oa);
  return new THREE.Quaternion().setFromRotationMatrix(m);
}

/**
 * Anatomical axes and pivot for a tooth, from what /cut returned.
 *
 * Pulled out of attach() so the staging timeline can pose a crown the gizmo is
 * not attached to. Scrubbing has to move EVERY committed tooth, and there is
 * only ever one gizmo.
 */
export function toothAxes(frame) {
  return {
    md: new THREE.Vector3().fromArray(frame.u_md).normalize(),
    bl: new THREE.Vector3().fromArray(frame.u_bl).normalize(),
    oa: new THREE.Vector3().fromArray(frame.u_oa).normalize(),
  };
}

/**
 * The delta a set of clinical values produces — the exact JS mirror of
 * cg.kinematic_matrix:
 *
 *     M = T(C_res + t_local) . R_torque . R_tip . R_rot . T(-C_res)
 *
 * Verified against the Python implementation on an orthonormal frame:
 * max |difference| = 1.8e-15 across combined tip/torque/rotation/translation
 * cases. The composition ORDER is load-bearing — the same three angles applied
 * in a different order differ by over a degree.
 *
 * STANDALONE, and the gizmo method delegates here rather than the reverse. A
 * stage pose is this function evaluated at scaled clinical values; the gizmo
 * has no part in it, and duplicating the composition for the timeline would be
 * two implementations of the one thing verify-kinematics.mjs pins.
 *
 * `axes` is toothAxes(frame); `cRes` a THREE.Vector3 or a 3-array.
 */
export function deltaFromClinical(axes, cRes, k) {
  const { md, bl, oa } = axes;
  const c = Array.isArray(cRes) ? new THREE.Vector3().fromArray(cRes) : cRes;
  const D2R = THREE.MathUtils.degToRad;

  const R = new THREE.Matrix4().makeRotationAxis(md, D2R(k.torque_deg || 0))
    .multiply(new THREE.Matrix4().makeRotationAxis(bl, D2R(k.tip_deg || 0)))
    .multiply(new THREE.Matrix4().makeRotationAxis(oa, D2R(k.rotation_deg || 0)));

  const t = md.clone().multiplyScalar(k.d_md || 0)
    .add(bl.clone().multiplyScalar(k.d_bl || 0))
    .add(oa.clone().multiplyScalar(k.d_oa || 0));

  return new THREE.Matrix4()
    .makeTranslation(c.x + t.x, c.y + t.y, c.z + t.z)
    .multiply(R)
    .multiply(new THREE.Matrix4().makeTranslation(-c.x, -c.y, -c.z));
}

/** The six channels scaled to stage k of n. Absolute from T0, never a lerp of
 *  the 4x4: component-wise interpolation of a rotation is not a rotation. */
export function clinicalAtStage(clinical, k, n) {
  const f = n > 0 ? k / n : 0;
  const out = {};
  for (const key of ["tip_deg", "torque_deg", "rotation_deg", "d_md", "d_bl", "d_oa"]) {
    out[key] = (clinical?.[key] || 0) * f;
  }
  return out;
}

/**
 * Stages needed for one tooth, and which channel forces that count.
 *
 * The JS mirror of cg.staging_estimate, with the same per-stage defaults. The
 * channel is reported by NAME rather than just "translation"/"rotation",
 * because "31 stages, rotation-driven" does not tell a clinician which dial to
 * reconsider.
 */
export const PER_STAGE_LIMITS = { translation_mm: 0.25, rotation_deg: 2.0 };

export function stagingFor(clinical, limits = PER_STAGE_LIMITS) {
  const c = clinical || {};
  const trans = Math.hypot(c.d_md || 0, c.d_bl || 0, c.d_oa || 0);
  const rots = [["tip_deg", Math.abs(c.tip_deg || 0)],
                ["torque_deg", Math.abs(c.torque_deg || 0)],
                ["rotation_deg", Math.abs(c.rotation_deg || 0)]];
  const [rotName, rotMag] = rots.reduce((a, b) => (b[1] > a[1] ? b : a), ["tip_deg", 0]);
  const nTrans = trans > 0 ? Math.ceil(trans / limits.translation_mm) : 0;
  const nRot = rotMag > 0 ? Math.ceil(rotMag / limits.rotation_deg) : 0;

  let channel = null;
  if (nTrans >= nRot && nTrans > 0) {
    const t = [["d_md", Math.abs(c.d_md || 0)], ["d_bl", Math.abs(c.d_bl || 0)],
               ["d_oa", Math.abs(c.d_oa || 0)]];
    channel = t.reduce((a, b) => (b[1] > a[1] ? b : a))[0];
  } else if (nRot > 0) {
    channel = rotName;
  }
  return {
    stages: Math.max(nTrans, nRot),
    driver: nTrans >= nRot ? "translation" : "rotation",
    channel,
    translation_mm: trans,
    max_rotation_deg: rotMag,
  };
}

export class ToothGizmo {
  /**
   * @param scene, camera, domElement, orbitControls  the usual suspects
   * @param onChange(clinical, matrix16RowMajor)  fires live during a drag
   * @param onCommit(clinical, matrix16RowMajor)  fires once on mouse-up
   */
  constructor(scene, camera, domElement, orbitControls, onChange, onCommit) {
    this.scene = scene;
    this.orbit = orbitControls;
    this.onChange = onChange;
    this.onCommit = onCommit;

    this.tc = new TransformControls(camera, domElement);
    this.tc.setSpace("local");          // rings follow the pivot, not the world
    this.tc.setMode("rotate");
    this.tc.setRotationSnap(THREE.MathUtils.degToRad(0.5));
    this.tc.setTranslationSnap(0.05);   // 0.05 mm — below any clinical tolerance

    // three r169+ no longer makes TransformControls an Object3D
    const helper = typeof this.tc.getHelper === "function" ? this.tc.getHelper() : this.tc;
    scene.add(helper);
    this.helper = helper;

    this.tc.addEventListener("dragging-changed", (e) => {
      this.orbit.enabled = !e.value;                 // don't orbit mid-drag
      if (!e.value && this.pivot) this.onCommit(this.read(), this.matrixRowMajor());
    });
    this.tc.addEventListener("objectChange", () => {
      if (this.pivot) this.onChange(this.read(), this.matrixRowMajor());
    });

    this.pivot = null;
    this.crown = null;
    this.restInv = new THREE.Matrix4();
    this.restQuat = new THREE.Quaternion();
  }

  /**
   * Attach to a freshly cut crown.
   * @param crownMesh  mesh whose geometry is in raw scanner coordinates
   * @param frame      { u_md, u_bl, u_oa, centroid } from derive_frame_from_region
   * @param cRes       [x,y,z] centre of resistance
   */
  /**
   * Attach to a crown, optionally RESUMING a tooth that has already been moved.
   *
   * @param delta  the tooth's committed transform from T0, or null for a fresh
   *               cut. Passing it is what makes a second cut non-destructive:
   *               with P0 = T(C_res)·R(frame), the pivot starts at delta·P0 and
   *               the crown keeps matrix = P0⁻¹, because (delta·P0)⁻¹·delta = P0⁻¹.
   *               deltaMatrix() therefore still returns the TOTAL delta from T0
   *               and read() still reports cumulative clinical values, so a
   *               resumed drag composes on top instead of restarting from zero.
   */
  attach(crownMesh, frame, cRes, delta = null) {
    if (!Array.isArray(cRes) || cRes.length !== 3 || !cRes.every(Number.isFinite)) {
      throw new Error(
        `C_res must be three finite numbers — refusing to attach the gizmo. ` +
        `Got ${JSON.stringify(cRes)}.`);
    }
    this.detach();

    // Kept for setClinical(), which must reproduce cg.kinematic_matrix exactly.
    this.axes = {
      md: new THREE.Vector3().fromArray(frame.u_md).normalize(),
      bl: new THREE.Vector3().fromArray(frame.u_bl).normalize(),
      oa: new THREE.Vector3().fromArray(frame.u_oa).normalize(),
    };
    this.cRes = new THREE.Vector3().fromArray(cRes);

    const rest = new THREE.Matrix4().compose(
      new THREE.Vector3().fromArray(cRes),
      quatFromFrame(frame),
      new THREE.Vector3(1, 1, 1));

    // The crown's transform is ALWAYS the inverse of the rest pose, resumed or
    // not, so the composite starts out matching wherever the tooth already sits
    // and the pivot matrix expresses the delta from T0.
    this.restInv.copy(rest).invert();
    this.restQuat.setFromRotationMatrix(rest);

    // Decomposed into position/quaternion/scale with matrixAutoUpdate left at
    // its default true: TransformControls drives those three during a drag and
    // relies on three recomposing the matrix each frame. Pinning the matrix
    // directly would freeze the crown in place while the rings moved.
    const pivot = new THREE.Object3D();
    (delta ? new THREE.Matrix4().multiplyMatrices(delta, rest) : rest)
      .decompose(pivot.position, pivot.quaternion, pivot.scale);
    pivot.updateMatrix();
    pivot.updateMatrixWorld(true);

    this.scene.add(pivot);
    crownMesh.matrixAutoUpdate = false;
    crownMesh.matrix.copy(this.restInv);
    crownMesh.matrixWorldNeedsUpdate = true;
    pivot.add(crownMesh);

    this.pivot = pivot;
    this.crown = crownMesh;
    this.tc.attach(pivot);
  }

  /**
   * Release the crown WITHOUT moving it.
   *
   * This used to do `crown.matrix.identity()`, which silently threw away the
   * clinician's work: attach() calls detach() first, so cutting a second tooth
   * snapped the first one back to T0. While attached the crown's world
   * transform is pivot.matrix · restInv, so writing exactly that product on
   * release leaves it visually fixed where it was placed.
   *
   * matrixAutoUpdate stays false: position/quaternion/scale were never
   * maintained on the crown, so letting three recompose the matrix from them
   * would snap it back to the origin — the same bug by another route.
   */
  detach() {
    if (!this.pivot) return;
    this.tc.detach();
    if (this.crown) {
      const baked = this.deltaMatrix();
      this.scene.add(this.crown);                    // reparent to scene
      this.crown.matrixAutoUpdate = false;
      this.crown.matrix.copy(baked);
      this.crown.matrixWorldNeedsUpdate = true;
      this.crown.updateMatrixWorld(true);
    }
    this.scene.remove(this.pivot);
    this.pivot = null;
    this.crown = null;
  }

  setMode(mode) { this.tc.setMode(mode); }           // "rotate" | "translate"

  /** Snap back to T0 without re-cutting. */
  reset() {
    if (!this.pivot) return;
    const rest = this.restInv.clone().invert();
    this.pivot.matrix.copy(rest);
    this.pivot.matrix.decompose(this.pivot.position, this.pivot.quaternion, this.pivot.scale);
    this.pivot.updateMatrixWorld(true);
    this.onCommit(this.read(), this.matrixRowMajor());
  }

  /** Delta transform from T0, in scanner coordinates. */
  deltaMatrix() {
    this.pivot.updateMatrix();
    return this.pivot.matrix.clone().multiply(this.restInv);
  }

  /**
   * The delta as 16 floats in ROW-MAJOR order, ready for
   * `np.array(arr).reshape(4,4)` on the backend with no transpose.
   *
   * THREE.Matrix4.elements is COLUMN-major. Sending it raw and reshaping it
   * server-side puts the translation into the bottom row, which is the same
   * convention mismatch that produced the perspective-divide spike. The
   * transpose is done here, once, on purpose.
   */
  matrixRowMajor() {
    const e = this.deltaMatrix().elements;
    return [e[0], e[4], e[8],  e[12],
            e[1], e[5], e[9],  e[13],
            e[2], e[6], e[10], e[14],
            e[3], e[7], e[11], e[15]];
  }

  /**
   * The delta a set of clinical values produces — the exact JS mirror of
   * cg.kinematic_matrix:
   *
   *     M = T(C_res + t_local) . R_torque . R_tip . R_rot . T(-C_res)
   *
   * Verified against the Python implementation on an orthonormal frame:
   * max |difference| = 1.8e-15 across combined tip/torque/rotation/translation
   * cases. The composition ORDER is load-bearing — the same three angles
   * applied in a different order differ by over a degree.
   */
  deltaFromClinical(k) {
    return deltaFromClinical(this.axes, this.cRes, k);
  }

  /**
   * Drive the tooth from the sidebar instead of the mouse.
   *
   * Values are ABSOLUTE clinical numbers measured from T0, not increments, so
   * typing 5 into Tip means five degrees of angulation from the original scan
   * however many times it is typed. The pivot is placed at delta*rest, exactly
   * as a resumed attach() does, so read() reports back what was typed and a
   * subsequent mouse drag composes on top.
   */
  setClinical(values) {
    if (!this.pivot) return null;
    const delta = this.deltaFromClinical(values);
    delta.multiply(this.restInv.clone().invert())      // delta * rest
      .decompose(this.pivot.position, this.pivot.quaternion, this.pivot.scale);
    this.pivot.updateMatrix();
    this.pivot.updateMatrixWorld(true);
    return this.matrixRowMajor();
  }

  /**
   * Decompose the drag into clinical numbers so the sidebar shows degrees and
   * millimetres rather than a matrix. Angles are extracted in the tooth's own
   * frame, which is why the pivot carries the anatomical quaternion.
   *
   * TWO CORRECTIONS, both measured against cg.kinematic_matrix:
   *
   * 1. Euler order is XYZ, not ZYX. The backend composes
   *    R_torque(u_MD) . R_tip(u_BL) . R_rot(u_OA), which in the pivot's local
   *    frame is Rx.Ry.Rz — three.js calls that 'XYZ'. Reading it back as 'ZYX'
   *    round-tripped a (-12.5, 7.5, -4.0) prescription as (-11.8, 8.5, -5.7):
   *    1.71 deg of pure decomposition error. Single-axis moves agreed, which is
   *    why it survived until the sidebar could set two axes at once.
   *
   * 2. Translation subtracts the pivot offset. The delta's translation column
   *    is c + t - R.c, not t, so a PURE 8 deg tip was reporting 0.94mm of
   *    mesiodistal translation that did not exist. With two-way binding that
   *    phantom value feeds straight back in and really does move the tooth.
   */
  read() {
    const d = this.deltaMatrix();
    const pos = new THREE.Vector3(), quat = new THREE.Quaternion(), scl = new THREE.Vector3();
    d.decompose(pos, quat, scl);

    // express the rotation in tooth axes: q_local = q_rest^-1 * q_delta * q_rest
    const inv = this.restQuat.clone().invert();
    const local = inv.clone().multiply(quat).multiply(this.restQuat);
    // pivot local axes are X=mesiodistal, Y=buccolingual, Z=long
    const eul = new THREE.Euler().setFromQuaternion(local, "XYZ");

    // translation in tooth axes
    const md = new THREE.Vector3(1, 0, 0).applyQuaternion(this.restQuat);
    const bl = new THREE.Vector3(0, 1, 0).applyQuaternion(this.restQuat);
    const oa = new THREE.Vector3(0, 0, 1).applyQuaternion(this.restQuat);

    // t_local = pos - c + R.c  (invert the T(c+t).R.T(-c) composition)
    const tLocal = this.cRes
      ? pos.clone().sub(this.cRes).add(this.cRes.clone().applyQuaternion(quat))
      : pos.clone();

    const deg = THREE.MathUtils.radToDeg;
    return {
      tip_deg:      +(deg(eul.y)).toFixed(2),   // about buccolingual
      torque_deg:   +(deg(eul.x)).toFixed(2),   // about mesiodistal
      rotation_deg: +(deg(eul.z)).toFixed(2),   // about the long axis
      d_md: +(tLocal.dot(md)).toFixed(3),
      d_bl: +(tLocal.dot(bl)).toFixed(3),
      d_oa: +(tLocal.dot(oa)).toFixed(3),
      scale_check: +(scl.length() / Math.sqrt(3)).toFixed(6),  // must stay 1.0
    };
  }
}

/**
 * Three-click occlusal plane picker.
 *
 * Collects left posterior, right posterior, anterior midline. The backend
 * turns them into an arch basis; the scan itself is never rotated.
 */
export function pickOcclusalPlane(domElement, camera, raycaster, arches, onProgress) {
  const wanted = ["left posterior cusp", "right posterior cusp", "anterior midline"];
  const picked = [];
  const ndc = new THREE.Vector2();

  return new Promise((resolve, reject) => {
    const onClick = (event) => {
      const rect = domElement.getBoundingClientRect();
      ndc.set(((event.clientX - rect.left) / rect.width) * 2 - 1,
              -((event.clientY - rect.top) / rect.height) * 2 + 1);
      raycaster.setFromCamera(ndc, camera);
      const meshes = Object.values(arches).filter(a => a?.mesh).map(a => a.mesh);
      const hits = raycaster.intersectObjects(meshes, false);
      if (!hits.length) return;
      picked.push(hits[0].point.toArray());
      onProgress(picked.length, wanted[picked.length] ?? null);
      if (picked.length === 3) { cleanup(); resolve(picked); }
    };
    const onKey = (e) => { if (e.key === "Escape") { cleanup(); reject(new Error("cancelled")); } };
    const cleanup = () => {
      domElement.removeEventListener("click", onClick);
      window.removeEventListener("keydown", onKey);
    };
    domElement.addEventListener("click", onClick);
    window.addEventListener("keydown", onKey);
    onProgress(0, wanted[0]);
  });
}

/** Wheeler averages. C_res is a parametric stand-in, so these are starting points. */
export const ROOT_DEFAULTS_MM = {
  incisor: 10, canine: 13, premolar: 9, molar: 9,
};

export function rootDefaultForFDI(fdi) {
  const pos = Number(String(fdi).slice(-1));
  if (pos <= 2) return ROOT_DEFAULTS_MM.incisor;
  if (pos === 3) return ROOT_DEFAULTS_MM.canine;
  if (pos <= 5) return ROOT_DEFAULTS_MM.premolar;
  return ROOT_DEFAULTS_MM.molar;
}
