// brush.js — the pointermove drag loop that was missing from App.jsx.
//
// I gave you onPointerDown and onPointerUp and never wrote the handler that
// does the actual painting, which is why the brush dropped single dots.
//
// Three things this has to get right, none of them obvious:
//
// 1. SPEED. A raw distance test over 360k vertices on every pointermove is
//    ~360k ops per event, and pointermove fires faster than the frame rate.
//    A uniform grid hash built once per mesh reduces each stroke sample to
//    the handful of cells the brush sphere overlaps.
//
// 2. GAPS. Pointer events are sampled, not continuous. Move the mouse
//    quickly and consecutive events land several brush-widths apart, leaving
//    a dotted line. Consecutive hits are interpolated so the stroke is solid.
//
// 3. OCCLUSION. A sphere around the hit point also contains vertices on the
//    FAR side of the tooth. Painting the buccal surface would silently paint
//    the lingual too, and the clinician would never see it — until the cut
//    came out wrong. Back-facing vertices are rejected by normal.

import * as THREE from "three";

// Grid cell ~= brush radius. Exported because both call sites in App.jsx had
// hardcoded the same 1.0 literal instead, so the constant sat unused while its
// value was duplicated twice. Changing this changes selection behaviour.
export const CELL_FACTOR = 1.0;

export class BrushIndex {
  /**
   * Build once per mesh, reuse for every stroke.
   *
   * ONLY LIVE VERTICES. Extracting a tooth removes its faces from the index
   * buffer but leaves its vertices in the position buffer — the arch's vertex
   * numbering is deliberately never remapped, because every selection, wand
   * field and seed the client holds is expressed in it. Indexing the raw
   * position buffer would therefore let the brush paint a tooth that is no
   * longer on the cast, and that selection would go on to /cut as a crown built
   * from faces the arch no longer has. Walking geometry.index instead keeps the
   * brush honest, and the index shrinks as teeth come out.
   *
   * Rebuild this after any change to the index buffer.
   */
  constructor(geometry, cell) {
    const pos = geometry.attributes.position;
    this.pos = pos;
    this.normal = geometry.attributes.normal;
    this.cell = cell;
    this.map = new Map();
    const k = (x, y, z) => `${x},${y},${z}`;
    this.key = k;

    const index = geometry.getIndex();
    let live;
    if (index) {
      live = new Set();
      const arr = index.array;
      for (let i = 0; i < arr.length; i++) live.add(arr[i]);
    }

    for (let i = 0; i < pos.count; i++) {
      if (live && !live.has(i)) continue;
      const cx = Math.floor(pos.getX(i) / cell);
      const cy = Math.floor(pos.getY(i) / cell);
      const cz = Math.floor(pos.getZ(i) / cell);
      const kk = k(cx, cy, cz);
      let bucket = this.map.get(kk);
      if (!bucket) { bucket = []; this.map.set(kk, bucket); }
      bucket.push(i);
    }
    this.liveCount = live ? live.size : pos.count;
  }

  /**
   * Vertex indices within `radius` of `p`, excluding back-facing ones.
   * `viewDir` points from the camera toward the surface.
   */
  near(p, radius, viewDir, out) {
    const c = this.cell, r2 = radius * radius;
    const x0 = Math.floor((p.x - radius) / c), x1 = Math.floor((p.x + radius) / c);
    const y0 = Math.floor((p.y - radius) / c), y1 = Math.floor((p.y + radius) / c);
    const z0 = Math.floor((p.z - radius) / c), z1 = Math.floor((p.z + radius) / c);
    for (let x = x0; x <= x1; x++)
      for (let y = y0; y <= y1; y++)
        for (let z = z0; z <= z1; z++) {
          const bucket = this.map.get(this.key(x, y, z));
          if (!bucket) continue;
          for (const i of bucket) {
            const dx = this.pos.getX(i) - p.x;
            const dy = this.pos.getY(i) - p.y;
            const dz = this.pos.getZ(i) - p.z;
            if (dx * dx + dy * dy + dz * dz > r2) continue;
            if (this.normal) {
              // reject the far side of the tooth: a normal pointing away from
              // the camera is on a surface the clinician cannot see
              const nd = this.normal.getX(i) * viewDir.x
                       + this.normal.getY(i) * viewDir.y
                       + this.normal.getZ(i) * viewDir.z;
              if (nd > 0) continue;
            }
            out.add(i);
          }
        }
    return out;
  }
}

/**
 * Install the full brush. Returns a disposer.
 *
 * `getState()` supplies live values so the handlers never close over stale
 * React state:
 *   { tool, radius, arches, activeArch, selection, controls, camera, raycaster }
 * `onChange(Set)` fires during the stroke for live recolouring.
 * `onCommit(Array)` fires once on release — push to the server there, not
 * per move, or you reintroduce the round-trip the session model exists to
 * avoid.
 */
export function installBrush(domElement, getState, onChange, onCommit) {
  let stroke = null;
  const ndc = new THREE.Vector2();
  const viewDir = new THREE.Vector3();
  const lastPoint = new THREE.Vector3();
  let haveLast = false;

  const hitAt = (event) => {
    const s = getState();
    const rect = domElement.getBoundingClientRect();
    ndc.set(((event.clientX - rect.left) / rect.width) * 2 - 1,
            -((event.clientY - rect.top) / rect.height) * 2 + 1);
    s.raycaster.setFromCamera(ndc, s.camera);
    // every arch, nearest hit -- testing only the active mesh lets a ray pass
    // through the near arch and strike the active one behind it
    const meshes = Object.values(s.arches).filter(a => a?.mesh).map(a => a.mesh);
    const hits = s.raycaster.intersectObjects(meshes, false);
    if (!hits.length) return null;
    const h = hits[0];
    const active = s.arches[s.activeArch];
    if (!active || h.object !== active.mesh) return null;   // painted a different arch
    return h;
  };

  const paintAt = (point, s) => {
    const idx = s.arches[s.activeArch].brushIndex;
    viewDir.copy(point).sub(s.camera.position).normalize();
    const found = idx.near(point, s.radius, viewDir, new Set());
    if (stroke.mode === "erase") { for (const i of found) stroke.set.delete(i); }
    else                        { for (const i of found) stroke.set.add(i); }
  };

  const onDown = (event) => {
    const s = getState();
    const painting = s.tool !== "wand" || event.shiftKey || event.altKey;
    if (!painting || event.button !== 0) return;
    const hit = hitAt(event);
    if (!hit) return;

    // Shift+drag is the browser's own selection gesture. Without these two
    // lines the browser starts a native drag and pointermove never reaches
    // this handler -- add appears dead while alt-erase works, which sends you
    // hunting in entirely the wrong place.
    event.preventDefault();
    domElement.setPointerCapture(event.pointerId);

    s.controls.enabled = false;
    stroke = {
      mode: event.altKey ? "erase" : (s.tool === "erase" ? "erase" : "add"),
      set: new Set(s.selection),
    };
    haveLast = false;
    paintAt(hit.point, s);
    lastPoint.copy(hit.point); haveLast = true;
    onChange(stroke.set);
  };

  const onMove = (event) => {
    if (!stroke) return;
    const s = getState();
    const hit = hitAt(event);
    if (!hit) return;

    // Interpolate from the previous sample. Pointer events are discrete; a
    // fast drag lands consecutive samples several brush-widths apart and
    // paints a dotted line instead of a stroke.
    if (haveLast) {
      const gap = lastPoint.distanceTo(hit.point);
      const steps = Math.min(24, Math.ceil(gap / (s.radius * 0.5)));
      for (let k = 1; k < steps; k++) {
        paintAt(lastPoint.clone().lerp(hit.point, k / steps), s);
      }
    }
    paintAt(hit.point, s);
    lastPoint.copy(hit.point); haveLast = true;
    onChange(stroke.set);
  };

  const onUp = (event) => {
    if (!stroke) return;
    const s = getState();
    const done = stroke; stroke = null; haveLast = false;
    s.controls.enabled = true;
    if (domElement.hasPointerCapture(event.pointerId))
      domElement.releasePointerCapture(event.pointerId);
    onCommit([...done.set]);          // one server round trip, on release
  };

  domElement.addEventListener("pointerdown", onDown);
  domElement.addEventListener("pointermove", onMove);
  domElement.addEventListener("pointerup", onUp);
  domElement.addEventListener("pointercancel", onUp);
  domElement.addEventListener("lostpointercapture", onUp);

  return () => {
    domElement.removeEventListener("pointerdown", onDown);
    domElement.removeEventListener("pointermove", onMove);
    domElement.removeEventListener("pointerup", onUp);
    domElement.removeEventListener("pointercancel", onUp);
    domElement.removeEventListener("lostpointercapture", onUp);
  };
}

/**
 * Apply the backend's 4x4 to a mesh, with a guard against the exact failure
 * you are describing.
 *
 * cg.kinematic_matrix returns a standard homogeneous transform: rotation in
 * the upper 3x3, TRANSLATION IN THE LAST COLUMN, bottom row [0,0,0,1].
 * M.ravel() flattens that row-major, and THREE.Matrix4.set() takes row-major
 * arguments, so .set(...arr) is correct.
 *
 * fromArray() reads COLUMN-major. Feeding it this array puts the translation
 * into n41,n42,n43 -- the perspective row -- so w varies per vertex and the
 * mesh shears into a spike. That is the explosion. The check below refuses
 * the array rather than rendering it.
 */
export function applyKinematics(mesh, arr) {
  if (!arr || arr.length !== 16) throw new Error("matrix must be 16 floats");
  const bottom = [arr[12], arr[13], arr[14], arr[15]];
  const looksTransposed =
    Math.abs(bottom[0]) + Math.abs(bottom[1]) + Math.abs(bottom[2]) > 1e-6;
  if (looksTransposed) {
    throw new Error(
      `Matrix bottom row is [${bottom.map(n => n.toFixed(3))}], expected ` +
      `[0,0,0,1]. Translation is sitting in the perspective row — the array ` +
      `arrived transposed. Send M.ravel() (row-major) and use .set(), or ` +
      `send M.T.ravel() and use .fromArray(). Do not mix them.`);
  }
  mesh.matrixAutoUpdate = false;
  mesh.matrix.set(...arr);
  mesh.matrixWorldNeedsUpdate = true;   // .matrix alone does not flag this
}
