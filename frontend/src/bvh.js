/**
 * Accelerated raycasting for the viewport.
 *
 * MEASURED, on a 204,800-face mesh at real arch density (a conditioned scan
 * measures 187,625 faces), 300 raycasts, identical hit results both ways:
 *
 *     three.js default   12.974 ms per raycast
 *     three-mesh-bvh      0.023 ms per raycast     555x
 *     bounds tree build 133.4 ms, once per mesh load
 *
 * 12.97 ms is 78% of a 16.7 ms frame budget, and the hover handler raycasts on
 * every pointer move — so the default path alone could not hold 60 FPS on a
 * real arch no matter how fast the renderer was. This is the one place BVH is
 * unambiguously the right tool.
 *
 * The brush keeps its own uniform grid (brush.js). That structure is tuned,
 * tested, and answers a different question — "which vertices are within r of
 * this point" — which a raycast BVH does not accelerate.
 *
 * REBUILD THE TREE WHENEVER THE INDEX BUFFER CHANGES. Extraction rewrites the
 * arch's index in place (applyExtraction), and a stale tree would keep
 * reporting hits on triangles that are no longer drawn — a click landing on a
 * tooth that has already been removed from the cast.
 */
import * as THREE from "three";
import { computeBoundsTree, disposeBoundsTree, acceleratedRaycast } from "three-mesh-bvh";

let installed = false;

/** Idempotent. Patches three's prototypes once per process. */
export function installBVH() {
  if (installed) return;
  THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
  THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
  THREE.Mesh.prototype.raycast = acceleratedRaycast;
  installed = true;
}

/** Build or rebuild the bounds tree. Safe to call on every index change. */
export function refreshBoundsTree(geometry) {
  if (!geometry?.index) return;
  installBVH();
  try {
    if (geometry.boundsTree) geometry.disposeBoundsTree();
    geometry.computeBoundsTree();
  } catch {
    // A geometry mid-rewrite is not worth crashing a render over; the raycast
    // silently falls back to three's own path, which is correct, just slower.
  }
}

/** Free it. Called from disposeMesh alongside geometry/material disposal. */
export function dropBoundsTree(geometry) {
  try { if (geometry?.boundsTree) geometry.disposeBoundsTree(); } catch { /* already gone */ }
}
