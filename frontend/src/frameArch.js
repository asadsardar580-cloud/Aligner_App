// frameArch.js — frame an arch comfortably WITHOUT touching its coordinates.
//
// The mesh stays in raw scanner space because bite registration depends on it
// and every click sent to the backend is expressed in it. So all framing is
// done by moving the CAMERA. Three separate defects are fixed here.
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
// 2. FIT DISTANCE
//    The old offset sat at 2.92 x radius. For a perspective camera the
//    distance that makes a sphere exactly fill the frame is r / sin(half),
//    where `half` is the SMALLER of the vertical and horizontal half-angles
//    — 2.61 x radius at fov 45 on a 16:9 canvas. Using the vertical angle
//    alone crops a wide arch on a tall window.
//
// 3. NEAR/FAR LEFT AT 0.1 / 5000
//    A 24 mm arch viewed from 60 mm with a far plane at 5000 wastes almost
//    the whole depth buffer, which shows up as z-fighting on the occlusal
//    surface where two cusps nearly touch. Both planes are now scaled to the
//    scene.
//
// Nothing here calls geom.center(), geom.scale() or mesh.position.set().

import * as THREE from "three";

/** Smallest-eigenvalue eigenvector of the vertex covariance = occlusal axis. */
function occlusalAxis(geom) {
  const p = geom.attributes.position;
  const n = p.count;
  // subsample: 20k points fixes the axis to well under a degree and keeps
  // this instant on a 360k-vertex arch
  const step = Math.max(1, Math.floor(n / 20000));
  let cx = 0, cy = 0, cz = 0, m = 0;
  for (let i = 0; i < n; i += step) { cx += p.getX(i); cy += p.getY(i); cz += p.getZ(i); m++; }
  cx /= m; cy /= m; cz /= m;

  let xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
  for (let i = 0; i < n; i += step) {
    const dx = p.getX(i) - cx, dy = p.getY(i) - cy, dz = p.getZ(i) - cz;
    xx += dx * dx; xy += dx * dy; xz += dx * dz;
    yy += dy * dy; yz += dy * dz; zz += dz * dz;
  }
  const C = [[xx / m, xy / m, xz / m], [xy / m, yy / m, yz / m], [xz / m, yz / m, zz / m]];

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

  const tr = (xx + yy + zz) / m;
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
    if (!v.every(Number.isFinite)) return new THREE.Vector3(0, 0, 1);
  }
  const axis = new THREE.Vector3(v[0], v[1], v[2]);
  if (axis.lengthSq() < 1e-12 || !Number.isFinite(axis.lengthSq())) {
    return new THREE.Vector3(0, 0, 1);
  }
  return axis.normalize();
}

/**
 * Frame `geom` for the given camera/controls. Returns the derived axes so the
 * caller can offer buccal / occlusal / lingual view presets.
 */
export function frameArch(geom, camera, controls, renderer, { tilt = 0.57 } = {}) {
  geom.computeBoundingSphere();
  const bs = geom.boundingSphere;
  const r = Math.max(bs.radius, 1e-3);

  let up = occlusalAxis(geom);

  // Orient the occlusal axis toward the viewer rather than away, so the
  // default view lands on the occlusal surface and not the underside of the
  // cast. The sign of an eigenvector is arbitrary; this resolves it by
  // asking which side the existing camera is on.
  if (up.dot(camera.position.clone().sub(bs.center)) < 0) up.negate();

  // an in-plane axis to swing the camera around for a three-quarter view
  const seed = Math.abs(up.x) < 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  const inPlane = seed.clone().sub(up.clone().multiplyScalar(seed.dot(up))).normalize();

  // fit distance on the LIMITING axis
  const halfV = THREE.MathUtils.degToRad(camera.fov) / 2;
  const halfH = Math.atan(Math.tan(halfV) * camera.aspect);
  const dist = r / Math.sin(Math.min(halfV, halfH));

  const dir = up.clone().multiplyScalar(Math.sqrt(1 - tilt * tilt))
    .add(inPlane.clone().multiplyScalar(tilt)).normalize();

  camera.up.copy(up);
  camera.position.copy(bs.center).add(dir.multiplyScalar(dist));
  camera.near = dist / 100;
  camera.far = dist * 4 + r * 4;
  camera.updateProjectionMatrix();       // without this, near/far are ignored

  controls.target.copy(bs.center);
  controls.minDistance = r * 0.15;       // stop the clinician flying inside a cusp
  controls.maxDistance = dist * 3;
  controls.zoomSpeed = 0.8;
  controls.update();

  return { center: bs.center.clone(), radius: r, up, inPlane, dist };
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
