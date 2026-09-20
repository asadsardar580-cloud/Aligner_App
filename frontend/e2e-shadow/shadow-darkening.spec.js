import { test, expect } from "@playwright/test";

/**
 * FRONTEND / VIEWPORT BUG: "Occlusal-plane establishment causes cast darkening."
 *
 * Reported from manual testing: load an arch, click the three occlusal-plane
 * landmarks, and on the third click the whole cast goes dark.
 *
 * This file answers the question the brief asks - WHICH COMPONENT does it -
 * by rendering the real rig in a real WebGL context and reading the
 * framebuffer back, rather than by reasoning about the arithmetic. The
 * arithmetic has been right and the picture wrong twice in this subsystem
 * already (CLAUDE.md 20.2), so "the math is finite" is not evidence here.
 *
 * Every luminance is averaged over a PER-PIXEL MASK OF THE CAST ITSELF, taken
 * from an ID pass with the catcher hidden. That separates "the cast got
 * darker" from "the frame got darker", which have different causes and
 * different fixes, and which a whole-image mean silently conflates.
 */

const BOOT = async (page) => {
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await page.goto("/shadow-harness.html");
  await page.waitForFunction(() => window.__harnessReady === true, { timeout: 60_000 });
  return errors;
};

/** Baseline: the rig has never been aimed. This is the "before" the user sees. */
async function baseline(page) {
  return page.evaluate(() => {
    const h = window.__harness;
    h.catcher.visible = false;
    h.sun.intensity = 0;
    h.renderer.shadowMap.enabled = true;
    h.render();
    const { mask, count } = h.castMask();
    window.__mask = mask;
    h.render();
    return { ...h.measure(mask), maskPixels: count };
  });
}

async function condition(page, opts) {
  return page.evaluate((o) => {
    const h = window.__harness;
    h.aim(o);
    return h.measure(window.__mask);
  }, opts);
}

test("A-F: which component of the shadow rig darkens the cast", async ({ page }) => {
  const errors = await BOOT(page);

  const before = await baseline(page);
  const frame = await page.evaluate(() => {
    const h = window.__harness;
    const bs = h.arch.geometry.boundingSphere;
    return { centre: bs.center.toArray(), radius: bs.radius,
             camera: h.camera.position.toArray(),
             up: h.camera.up.toArray() };
  });
  console.log("  cast sphere", frame.centre.map(n=>n.toFixed(1)).join(","),
              "r", frame.radius.toFixed(1),
              "| camera", frame.camera.map(n=>n.toFixed(1)).join(","));
  console.log("  mask pixels", before.maskPixels, "of", 640*480);
  expect(before.maskPixels, "the synthetic cast must actually be on screen")
    .toBeGreaterThan(5000);
  expect(before.cast, "the cast must be lit before the plane is established")
    .toBeGreaterThan(60);

  // The brief's matrix, each component isolated.
  const A = before;                                                    // no aim
  const B = await condition(page, { enableSun: false, enableCatcher: false,
                                    intensity: 0 });
  const C = await condition(page, { enableCatcher: false, castShadow: false });
  const D = await condition(page, { enableCatcher: false, castShadow: true });
  const E = await condition(page, { enableCatcher: true, intensity: 0 });
  const F = await condition(page, { enableCatcher: true, castShadow: true });
  const noShadowMap = await condition(page, { enableCatcher: true,
                                              shadowMapEnabled: false });

  const rows = { A_no_aim: A, B_sun0_nocatcher: B, C_sun_noshadow_nocatcher: C,
                 D_sun_shadow_nocatcher: D, E_catcher_sun0: E, F_both: F,
                 G_both_shadowmap_off: noShadowMap };
  for (const [k, v] of Object.entries(rows)) {
    console.log(`  ${k.padEnd(26)} cast ${v.cast.toFixed(1).padStart(6)}  `
      + `bg ${v.background.toFixed(1).padStart(6)}  `
      + `castDark ${(v.castDarkFraction * 100).toFixed(1)}%`);
  }

  const geom = await page.evaluate(() => {
    const h = window.__harness;
    h.aim({});
    return { shadowCam: h.catcherInShadowCamera(), order: h.catcherOrdering(),
             rig: h.rigFor() };
  });
  console.log("  catcher corners in shadow-camera NDC:",
    geom.shadowCam.corners.map((c) => c.map((x) => x.toFixed(2)).join(",")).join("  |  "));
  console.log("  corners outside the shadow map:",
    geom.shadowCam.cornersOutsideShadowMap, "of 4");
  console.log("  catcher between camera and cast:", geom.order.catcherIsInFront);
  console.log("  catcherSize", geom.rig.catcherSize.toFixed(1),
              "frustum half", geom.rig.frustum.right.toFixed(1),
              "radius", geom.rig.radius.toFixed(1));

  expect(errors, `page errors: ${errors.join(" | ")}`).toEqual([]);

  // --- THE ASSERTIONS -----------------------------------------------------
  // The catcher must never sit between the viewer and the model.
  expect(geom.order.catcherIsInFront,
    "the shadow catcher is between the camera and the cast").toBe(false);

  // Every corner of the catcher must sample a texel the shadow map actually
  // rendered. A corner outside [-1,1] reads the clamped edge texel, which is
  // SHADOWED whatever is really there - a dark wash by construction.
  expect(geom.shadowCam.cornersOutsideShadowMap,
    "catcher corners fall outside the shadow map and will read as shadowed")
    .toBe(0);

  // The cast is opaque and receives no shadow, so ARMING THE RIG MUST NOT
  // DARKEN IT. A little movement is legitimate - the sun adds light - but it
  // may never lose luminance.
  expect(F.cast, "establishing the occlusal plane darkened the cast")
    .toBeGreaterThan(A.cast * 0.95);

  // And no large uniform dark wash over the model.
  expect(F.castDarkFraction,
    "a large fraction of the cast collapsed to near-black").toBeLessThan(0.05);
});
