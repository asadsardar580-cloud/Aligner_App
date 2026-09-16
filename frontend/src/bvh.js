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
import { mergeVertices } from "three/examples/jsm/utils/BufferGeometryUtils.js";

let installed = false;

// ONE warning per process, not one per geometry. A pointermove handler can
// reach this path sixty times a second; a per-call console.warn turns a
// degraded-but-working viewport into a frozen devtools window, which is a
// worse failure than the one being reported.
let warnedFallback = false;

export const WARN_FALLBACK = "[BVH Warning] Falling back to proxy bounding box raycast";

function warnOnce(detail) {
  if (warnedFallback) return;
  warnedFallback = true;
  console.warn(`${WARN_FALLBACK} — ${detail}. Raycasting stays CORRECT; it drops `
    + `from ~0.02ms to ~13ms per cast, which is 78% of a 60 FPS frame budget on a `
    + `full arch, so hover may feel heavy on large meshes.`);
}

/** Test seam: the fallback is a once-per-process latch and tests need it reset. */
export function _resetFallbackWarning() { warnedFallback = false; }

/** Idempotent. Patches three's prototypes once per process. */
export function installBVH() {
  if (installed) return;
  THREE.BufferGeometry.prototype.computeBoundsTree = computeBoundsTree;
  THREE.BufferGeometry.prototype.disposeBoundsTree = disposeBoundsTree;
  THREE.Mesh.prototype.raycast = acceleratedRaycast;
  installed = true;
}

/**
 * Build or rebuild the bounds tree. Safe to call on every index change.
 *
 * THE CASCADE, and why each rung exists:
 *
 *   0. No index buffer at all. This used to return silently, which meant a
 *      non-indexed geometry raycast at 13ms per cast forever with nothing
 *      anywhere saying why.
 *   1. computeBoundsTree throws. Usually duplicated or degenerate triangles;
 *      `mergeVertices` welds them and the retry succeeds.
 *   2. Still throws, or welding is not permitted here. Fall back to three's own
 *      raycast — CORRECT, just slow — and say so once.
 *
 * `allowWeld` DEFAULTS TO FALSE, AND THAT IS THE LOAD-BEARING PART.
 * `mergeVertices` renumbers vertices. On the arch, a vertex id is the contract
 * with the backend session: `BrushIndex`, the wand selection and the cumulative
 * extraction mask are all keyed on the position attribute's indices, and the
 * server's `verts` array is never rebuilt (see CLAUDE.md §6). Welding the arch
 * would repair a raycast by silently re-pointing every selection at different
 * anatomy — a far worse failure than a 13ms cast. So the arch takes the slow
 * path and stays correct; crowns, whose ids are local and are replaced wholesale
 * from each server payload, opt in.
 *
 * The arch's realistic failure mode — NaN vertex coordinates from a scanner —
 * is caught at upload instead, where it can be repaired without renumbering.
 *
 * Returns the rung that succeeded, so a caller (and a test) can tell a fast path
 * from a degraded one instead of inferring it from frame times.
 */
export function refreshBoundsTree(geometry, { allowWeld = false } = {}) {
  if (!geometry) return "none";
  installBVH();

  try { if (geometry.boundsTree) geometry.disposeBoundsTree(); } catch { /* already gone */ }

  if (geometry.index) {
    try {
      geometry.computeBoundsTree();
      return "direct";
    } catch (err) {
      return retryViaMerge(geometry, err, allowWeld);
    }
  }
  return retryViaMerge(geometry, new Error("geometry has no index buffer"), allowWeld);
}

function retryViaMerge(geometry, firstErr, allowWeld) {
  const before = geometry.attributes?.position?.count ?? 0;

  if (!allowWeld) {
    warnOnce(`${firstErr.message}; welding is not permitted on this geometry `
      + `because its vertex ids are the session's selection keys`);
    return "fallback";
  }

  // Declared without an initialiser: the only path that reads it is after the
  // assignment has succeeded, and a `= null` here is dead.
  let merged;
  try {
    merged = mergeVertices(geometry);
  } catch (mergeErr) {
    warnOnce(`${firstErr.message}; mergeVertices also failed (${mergeErr.message})`);
    return "fallback";
  }

  try {
    // Adopt the welded attributes IN PLACE rather than returning a new object:
    // the caller holds this geometry on a live mesh and swapping it would leave
    // the old one rendering.
    for (const name of Object.keys(merged.attributes)) {
      geometry.setAttribute(name, merged.attributes[name]);
    }
    geometry.setIndex(merged.index);
    geometry.computeBoundsTree();
    merged.dispose();
    const after = geometry.attributes.position.count;
    if (after !== before) {
      console.warn(`[BVH] welded ${before - after} duplicate vertices to build a `
        + `bounds tree (${before} -> ${after}). Vertex ids on this geometry have `
        + `changed; only geometries with no external id contract reach this path.`);
    }
    return "merged";
  } catch (retryErr) {
    try { merged.dispose(); } catch { /* nothing to free */ }
    warnOnce(`${firstErr.message}; retry after mergeVertices also failed `
      + `(${retryErr.message})`);
    return "fallback";
  }
}

/**
 * Point a raycaster at the accelerated path.
 *
 * `firstHitOnly` is not a micro-optimisation here: every call site takes `[0]`
 * and throws the rest away, and without this flag three-mesh-bvh collects and
 * SORTS every intersection along the ray. On an arch that is the difference
 * between one hit and hundreds, on a handler that fires per pointermove.
 */
export function configureRaycaster(raycaster) {
  installBVH();
  raycaster.firstHitOnly = true;
  return raycaster;
}

/** Free it. Called from disposeMesh alongside geometry/material disposal. */
export function dropBoundsTree(geometry) {
  try { if (geometry?.boundsTree) geometry.disposeBoundsTree(); } catch { /* already gone */ }
}
