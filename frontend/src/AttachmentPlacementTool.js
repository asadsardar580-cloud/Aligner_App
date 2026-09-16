import * as THREE from "three";
import { refreshBoundsTree } from "./bvh";

/**
 * Click-to-place composite attachments on a crown's facial surface.
 *
 * WHAT THE CLICK ACTUALLY YIELDS, and why both halves matter. A raycast against
 * the crown returns a point p AND the face normal n at that point. The point
 * decides WHERE the attachment sits; the normal decides WHICH WAY IT FACES, and
 * an attachment facing the wrong way is not a cosmetic problem — the tray bears
 * against its flat face, so the direction of n is the direction of the force
 * the appliance can deliver.
 *
 * THE NORMAL MUST BE THE FACE NORMAL IN WORLD SPACE, NOT THE INTERPOLATED
 * VERTEX NORMAL. Three.js gives `intersection.face.normal` in OBJECT space; a
 * cut crown carries a committed 4x4 from its prescription, so using it raw
 * places attachments correctly on an unmoved tooth and progressively wrongly on
 * a moved one. The normal matrix conversion below is not optional.
 *
 * RAYCASTING GOES THROUGH three-mesh-bvh. Measured on a 204,800-face mesh:
 * 12.974ms per raycast with three's default, 0.023ms with the BVH. Placement
 * previews follow the pointer, so this is the difference between a tool that
 * tracks and one that stutters.
 */

export const SHAPES = {
  vertical_rectangular: { md: 2.0, oa: 3.0, bl: 1.0,
    label: "Vertical rectangular", purpose: "resists rotation about the long axis" },
  horizontal_rectangular: { md: 3.0, oa: 2.0, bl: 1.0,
    label: "Horizontal rectangular", purpose: "resists mesiodistal tipping" },
  horizontal_bevel: { md: 3.0, oa: 2.0, bl: 1.25,
    label: "Horizontal bevel", purpose: "resists extrusion — the bevel gives the tray an occlusal face to pull against" },
  ellipsoidal: { md: 2.5, oa: 2.5, bl: 1.0,
    label: "Ellipsoidal", purpose: "general retention, least irritating to soft tissue" },
};

// Matches attachments.py. Changing one without the other lets the UI offer a
// size the server refuses, which reads to a clinician as a broken button.
export const LIMITS = {
  oa: [1.0, 5.0],   // height
  md: [1.0, 4.0],   // width
  bl: [0.5, 2.0],   // depth off the surface
  rotation_deg: [0, 360],
};

export const PREVIEW_COLOUR = 0x00d0b0;   // used nowhere else in the scene

export function clampDims(dims) {
  const out = { ...dims };
  for (const k of ["md", "oa", "bl"]) {
    const [lo, hi] = LIMITS[k];
    out[k] = Math.min(hi, Math.max(lo, Number(out[k]) || lo));
  }
  return out;
}

/** Local geometry for a shape, before it is oriented or seated. */
export function buildGeometry(shape, dims) {
  const d = clampDims(dims || SHAPES[shape] || SHAPES.vertical_rectangular);
  let g;
  if (shape === "ellipsoidal") {
    g = new THREE.SphereGeometry(0.5, 20, 14);
    g.scale(d.md, d.oa, d.bl);
  } else if (shape === "horizontal_bevel") {
    g = new THREE.BoxGeometry(d.md, d.oa, d.bl);
    // Cut the occlusal-facial edge back, leaving the ramp the tray pulls on.
    const pos = g.attributes.position;
    for (let i = 0; i < pos.count; i++) {
      if (pos.getY(i) > 0 && pos.getZ(i) > 0) pos.setZ(i, pos.getZ(i) * 0.25);
    }
    pos.needsUpdate = true;
    g.computeVertexNormals();
  } else {
    g = new THREE.BoxGeometry(d.md, d.oa, d.bl);
  }
  return g;
}

/**
 * Orient an attachment so its local +Z lies along the surface normal.
 *
 * `rotationDeg` spins it about that normal — which is the only rotation that
 * keeps it flat on the tooth, and the one a clinician actually adjusts to line
 * a rectangle up with the long axis.
 */
export function orient(mesh, point, normal, rotationDeg = 0, depth = 1.0,
                       embed = 0.3) {
  const n = normal.clone().normalize();
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
  if (rotationDeg) {
    q.multiply(new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(0, 0, 1), (rotationDeg * Math.PI) / 180));
  }
  mesh.quaternion.copy(q);
  // Seat it: half its depth out along the normal, less the embed, so the
  // boolean union has real overlap rather than a tangential face-to-face
  // contact — which manifold3d answers with coincident-but-distinct vertices
  // that binary STL cannot express.
  mesh.position.copy(point).addScaledVector(n, depth / 2 - embed);
  mesh.updateMatrix();
  mesh.updateMatrixWorld(true);
  return mesh;
}

/**
 * Attach the click-to-place tool to the viewport.
 *
 * Returns a handle with `dispose()`. The caller owns when placement is live;
 * this never grabs the pointer while another tool is active.
 */
export function installAttachmentTool({
  dom, camera, raycaster, getTargets, getSettings, onPlace, onPreview, scene,
}) {
  const preview = new THREE.Mesh(
    buildGeometry("vertical_rectangular", SHAPES.vertical_rectangular),
    new THREE.MeshBasicMaterial({
      color: PREVIEW_COLOUR, transparent: true, opacity: 0.55,
      depthTest: false, wireframe: false,
    }));
  preview.visible = false;
  preview.renderOrder = 999;      // the preview is a tool, not scene geometry
  preview.matrixAutoUpdate = false;
  scene.add(preview);

  let shape = "vertical_rectangular";
  let lastHit = null;

  const ndc = (e) => {
    const r = dom.getBoundingClientRect();
    return new THREE.Vector2(
      ((e.clientX - r.left) / r.width) * 2 - 1,
      -((e.clientY - r.top) / r.height) * 2 + 1);
  };

  /** The one place a click becomes (point, world normal). */
  function pick(e) {
    const targets = getTargets() || [];
    if (!targets.length) return null;
    raycaster.setFromCamera(ndc(e), camera);
    const hit = raycaster.intersectObjects(targets, false)[0];
    if (!hit || !hit.face) return null;

    // OBJECT space -> WORLD space. A cut crown carries its prescription as a
    // committed matrix, so skipping this places attachments correctly only on
    // teeth that have not moved yet.
    const normal = hit.face.normal.clone()
      .applyNormalMatrix(new THREE.Matrix3().getNormalMatrix(hit.object.matrixWorld))
      .normalize();
    return { point: hit.point.clone(), normal, object: hit.object,
             toothId: hit.object.userData?.toothId ?? null };
  }

  function rebuildPreview() {
    const s = getSettings() || {};
    if (s.shape && s.shape !== shape) shape = s.shape;
    preview.geometry.dispose();
    preview.geometry = buildGeometry(shape, s.dimensions);
    refreshBoundsTree(preview.geometry, { allowWeld: true });
  }

  function onMove(e) {
    const hit = pick(e);
    lastHit = hit;
    if (!hit) { preview.visible = false; onPreview?.(null); return; }
    const s = getSettings() || {};
    const d = clampDims(s.dimensions || SHAPES[shape]);
    orient(preview, hit.point, hit.normal, s.rotation_deg || 0, d.bl);
    preview.visible = true;
    onPreview?.({ toothId: hit.toothId, point: hit.point.toArray(),
                  normal: hit.normal.toArray() });
  }

  function onClick(e) {
    const hit = lastHit || pick(e);
    if (!hit) return;
    const s = getSettings() || {};
    const d = clampDims(s.dimensions || SHAPES[shape]);
    onPlace?.({
      tooth_id: hit.toothId,
      type: shape,
      position_xyz: hit.point.toArray(),
      normal_xyz: hit.normal.toArray(),
      dimensions_hwd: { height: d.oa, width: d.md, depth: d.bl },
      rotation_deg: s.rotation_deg || 0,
    });
  }

  dom.addEventListener("pointermove", onMove);
  dom.addEventListener("click", onClick);

  return {
    refresh: rebuildPreview,
    hide() { preview.visible = false; },
    dispose() {
      dom.removeEventListener("pointermove", onMove);
      dom.removeEventListener("click", onClick);
      scene.remove(preview);
      preview.geometry.dispose();
      preview.material.dispose();
    },
  };
}
