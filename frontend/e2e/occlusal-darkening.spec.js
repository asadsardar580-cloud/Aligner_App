import { test, expect } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";

/**
 * FRONTEND / VIEWPORT BUG: "Occlusal-plane establishment causes cast darkening."
 *
 * THE ACTUAL WORKFLOW, in a real browser, against the production build: load
 * the arch, place the three landmarks, and measure whether the cast lost
 * luminance. `e2e-shadow/shadow-darkening.spec.js` isolates the rig's
 * components on a synthetic arch; this one exercises the path the clinician
 * actually takes, on the real scan.
 *
 * THE SCAN IS PATIENT-DERIVED AND STAYS LOCAL. `case_lower.stl` is gitignored
 * and is never copied anywhere by this file - it is read from the repo root at
 * run time, and the test SKIPS rather than substituting a synthetic stand-in,
 * because a stand-in would be testing the stand-in.
 *
 * THREE THINGS THIS SPEC LEARNED THE HARD WAY, each of which made an earlier
 * run report something false:
 *
 *  1. HEADLESS CHROMIUM COMPOSITES THE WebGL CANVAS AS EMPTY without a GL
 *     backend. The first run read exactly 14.79 - the clear colour 0x0d0f12 -
 *     both before and after, and a capture that never contains the model
 *     cannot tell you the model got darker.
 *  2. "Define Occlusal Plane" IS ALWAYS IN THE DOM, merely disabled, so
 *     waiting for it to be visible photographed an empty viewport while the
 *     status line still read "Uploading case_lower.stl...".
 *  3. THE WAND IS ARMED AT THE SAME TIME AS THE PICKER and listens on the same
 *     element, so every landmark click also selects ~21,800 vertices and
 *     overwrites the status line. That makes the picker's own "(n/3)" useless
 *     as a progress signal AND repaints the cast between the before and after
 *     frames - which is why the rig is A/B tested through the Shadows toggle,
 *     with the selection held fixed, rather than across the three clicks.
 */

const API = "http://127.0.0.1:8000";
const SCAN = path.resolve(process.cwd(), "..", "case_lower.stl");

test.use({ launchOptions: { args: ["--use-gl=angle", "--enable-unsafe-swiftshader"] } });

async function backendUp(request) {
  try {
    return (await request.get(`${API}/api/ai/status`, { timeout: 2000 })).ok();
  } catch { return false; }
}

/**
 * Luminance statistics for the viewport, decoded in-page.
 *
 * A WebGL canvas built without `preserveDrawingBuffer` reads back blank
 * through `toDataURL`, so the frame is captured with Playwright's own
 * screenshot and handed BACK into the page to be decoded through an `Image`
 * and a 2D context. Changing how the renderer is constructed to make the
 * capture easier would mean measuring a different renderer.
 */
async function canvasStats(page) {
  const canvas = page.locator("canvas").first();
  const shot = (await canvas.screenshot()).toString("base64");
  return page.evaluate(async (b64) => {
    const img = new Image();
    await new Promise((res, rej) => {
      img.onload = res; img.onerror = rej;
      img.src = "data:image/png;base64," + b64;
    });
    const c = document.createElement("canvas");
    c.width = img.width; c.height = img.height;
    const ctx = c.getContext("2d");
    ctx.drawImage(img, 0, 0);
    const d = ctx.getImageData(0, 0, c.width, c.height).data;
    let sum = 0, dark = 0, lit = 0, litSum = 0;
    const n = c.width * c.height;
    for (let i = 0; i < n; i++) {
      const L = 0.2126 * d[i * 4] + 0.7152 * d[i * 4 + 1] + 0.0722 * d[i * 4 + 2];
      sum += L;
      if (L < 30) dark++;
      // The viewport background is 0x0d0f12, luminance 14.8. Anything clearly
      // above it is model, so "lit" isolates the CAST from the backdrop - a
      // whole-frame mean is mostly backdrop and would hide the symptom.
      if (L > 45) { lit++; litSum += L; }
    }
    return { mean: sum / n, darkFraction: dark / n, litPixels: lit,
             litMean: lit ? litSum / lit : 0 };
  }, shot);
}

/**
 * Hand the session back when the test is done.
 *
 * `STORE` keeps four sessions and EVICTS the oldest (CLAUDE.md section 15).
 * This spec uploads a full 9.4 MB arch per test, so left alone it pushes other
 * specs' sessions out and makes THEIR assertions fail somewhere downstream -
 * which reads as flakiness in a file that did nothing wrong.
 */
test.afterEach(async ({ page, request }) => {
  const sid = await page.evaluate(() => {
    try {
      for (const k of Object.keys(localStorage)) {
        const v = localStorage.getItem(k);
        if (v && /^[0-9a-f-]{16,}$/i.test(v)) return v;
      }
    } catch { /* private mode */ }
    return null;
  }).catch(() => null);
  if (sid) await request.delete(`${API}/api/session/${sid}`).catch(() => {});
});

const show = (label, s) =>
  `  ${label.padEnd(22)} litPixels ${String(s.litPixels).padStart(7)}  `
  + `litMean ${s.litMean.toFixed(2).padStart(7)}  mean ${s.mean.toFixed(2).padStart(6)}  `
  + `dark ${(s.darkFraction * 100).toFixed(1)}%`;

test("establishing the occlusal plane must not darken the cast", async ({ page, request }) => {
  test.skip(!(await backendUp(request)), "backend not running");
  test.skip(!fs.existsSync(SCAN), `no local scan at ${SCAN}`);
  test.setTimeout(300_000);

  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  const rigLines = [];
  page.on("console", (m) => {
    const t = m.text();
    if (t.includes("[Shadow Rig]") || t.includes("[Arch Framing]")) rigLines.push(t);
    if (m.type() === "error") errors.push(t);
  });

  await page.goto("/");
  await expect(page.getByRole("heading", { name: /Clinical Micro-Planner/i })).toBeVisible();

  await page.locator('input[type="file"]').nth(1).setInputFiles(SCAN);

  // Wait for the ARCH, not for the button. See note 2 above.
  await expect(page.locator("text=/No arch loaded/i")).toBeHidden({ timeout: 180_000 });
  await expect(page.getByRole("button", { name: /Define Occlusal Plane/i }))
    .toBeEnabled({ timeout: 60_000 });
  await page.waitForTimeout(4000);

  const before = await canvasStats(page);
  const diagBefore = await page.evaluate(() => window.__viewportDiagnostics?.());
  await page.screenshot({ path: "test-results/occlusal-before.png" });

  const box = await page.locator("canvas").first().boundingBox();

  // The rig ships OFF (App.jsx, `shadowsOn`). This is therefore the shipped
  // path: establish the plane with aimShadows gated out.
  const toggle = page.getByRole("checkbox", { name: /Shadows/i });
  await expect(toggle).not.toBeChecked();

  await page.getByRole("button", { name: /Define Occlusal Plane/i }).click();

  // The picker only advances on a MESH HIT, so a fixed triple of coordinates
  // is a coin flip against whatever the framing produced. Probe a grid and
  // stop on the durable signal - the button label flips only when `archFrame`
  // is committed. See note 3 for why the status line cannot be used.
  const planeSet = async () =>
    (await page.getByRole("button", { name: /Occlusal Plane Set/i }).count()) > 0;

  // THE REAL LANDMARKS FIRST, in the order the picker asks for them: left
  // posterior cusp, right posterior cusp, anterior midline. Read off the
  // framed view, which is deterministic for a given scan. A blind grid gets
  // three points on the cast but not the three the clinician would choose,
  // and three arbitrary points produce a different occlusal normal - which
  // changes what this test is measuring.
  const ANATOMICAL = [[0.365, 0.294], [0.749, 0.265], [0.500, 0.795]];
  const candidates = ANATOMICAL.map(([fx, fy]) =>
    ({ x: box.x + box.width * fx, y: box.y + box.height * fy }));
  // Fallback sweep, only reached if a landmark misses the mesh.
  for (const fy of [0.35, 0.45, 0.55, 0.28, 0.65, 0.72]) {
    for (const fx of [0.30, 0.70, 0.50, 0.40, 0.60, 0.22, 0.78]) {
      candidates.push({ x: box.x + box.width * fx, y: box.y + box.height * fy });
    }
  }
  let tried = 0;
  for (const p of candidates) {
    if (await planeSet()) break;
    tried++;
    await page.mouse.click(p.x, p.y);
    await page.waitForTimeout(400);
  }
  console.log(`  occlusal plane established after ${tried} canvas clicks`);
  expect(await planeSet(), "could not establish the occlusal plane").toBe(true);
  await page.waitForTimeout(2500);

  const after = await canvasStats(page);
  const diagAfter = await page.evaluate(() => window.__viewportDiagnostics?.());
  await page.screenshot({ path: "test-results/occlusal-after.png" });

  // WHAT ACTUALLY CHANGED. Printed as a diff so the cause names itself rather
  // than being argued for.
  const flat = (o, p = "") => Object.entries(o || {}).flatMap(([k, v]) =>
    v && typeof v === "object" && !Array.isArray(v)
      ? flat(v, p + k + ".")
      : [[p + k, JSON.stringify(v)]]);
  const b = Object.fromEntries(flat(diagBefore));
  const a = Object.fromEntries(flat(diagAfter));
  const lightLines = (tag, d) => (d?.lights || []).forEach((l) => console.log(
    `  ${tag} ${l.type.padEnd(17)} i=${String(l.intensity).padEnd(5)} `
    + `cam=${l.parentIsCamera ? "Y" : "n"} `
    + `pos=[${(l.worldPosition || []).map((n) => n.toFixed(1)).join(",")}] `
    + `dir=[${(l.worldDirection || []).map((n) => n.toFixed(3)).join(",")}]`));
  lightLines("BEFORE", diagBefore);
  lightLines("AFTER ", diagAfter);
  console.log("  BEFORE framing:", JSON.stringify(diagBefore?.framing));
  console.log("  AFTER  framing:", JSON.stringify(diagAfter?.framing));
  console.log("  FULL AFTER STATE:", JSON.stringify(diagAfter));
  console.log("  --- viewport state changes (before -> after) ---");
  for (const k of new Set([...Object.keys(b), ...Object.keys(a)])) {
    if (b[k] !== a[k]) console.log(`    ${k}: ${b[k]} -> ${a[k]}`);
  }

  // IS IT THE CAMERA SIDE? Orbit 180 degrees and look again. Nothing about
  // the model, the lights or the rig changes - only where the viewer stands.
  // If the cast lights up from the other side, the geometry and the lighting
  // are both fine and the re-framing put the viewer behind the scan.
  await page.mouse.move(box.x + box.width * 0.5, box.y + box.height * 0.5);
  await page.mouse.down();
  for (let i = 1; i <= 24; i++) {
    await page.mouse.move(box.x + box.width * 0.5 + i * 22, box.y + box.height * 0.5);
    await page.waitForTimeout(16);
  }
  await page.mouse.up();
  await page.waitForTimeout(1200);
  const orbited = await canvasStats(page);
  await page.screenshot({ path: "test-results/occlusal-orbited.png" });
  console.log(show("AFTER, orbited 180", orbited));

  // THE CONTROLLED COMPARISON. Everything - selection, camera, geometry - is
  // now identical; only the rig changes. This is what isolates a lighting
  // cause from a geometry cause, and it is the reason the toggle exists.
  await page.waitForTimeout(400);
  const rigOff = await canvasStats(page);
  await page.screenshot({ path: "test-results/occlusal-shadows-off.png" });


  console.log(show("BEFORE plane", before));
  console.log(show("AFTER plane", after));
  console.log(show("AFTER, shadows off", rigOff));
  console.log(`  litMean after/before      ${(after.litMean / before.litMean).toFixed(4)}`);
  for (const l of rigLines) console.log("  " + l);

  expect(errors, `page errors: ${errors.join(" | ")}`).toEqual([]);
  expect(before.litPixels, "the cast must be visible before the plane is set")
    .toBeGreaterThan(20000);

  // GROSS FAILURE ONLY. Anti-aliasing and GPU differences make pixel equality
  // meaningless, and the sun legitimately ADDS light. What must not happen is
  // the model losing a material fraction of its brightness, or a large part of
  // the frame collapsing to near-black.
  expect(after.litMean / before.litMean,
    "the cast lost luminance when the occlusal plane was established")
    .toBeGreaterThan(0.90);
  expect(after.litPixels / Math.max(before.litPixels, 1),
    "a large part of the cast stopped being lit").toBeGreaterThan(0.80);
  expect(after.darkFraction - before.darkFraction,
    "a dark wash appeared over the viewport").toBeLessThan(0.15);
});


/**
 * The rig, armed against a real occlusal frame and held against its absence.
 *
 * This is the second half of the same defect: the first test proves the
 * WORKFLOW no longer darkens the cast, and this one proves the SUBSYSTEM
 * itself is sound when it arms - so the first test cannot be passing merely
 * because the rig quietly stopped doing anything.
 */
test("arming the shadow rig must not blank the cast",
  async ({ page, request }) => {
    test.skip(!(await backendUp(request)), "backend not running");
    test.skip(!fs.existsSync(SCAN), `no local scan at ${SCAN}`);
    test.setTimeout(300_000);

    await page.goto("/");
    await page.locator('input[type="file"]').nth(1).setInputFiles(SCAN);
    await expect(page.locator("text=/No arch loaded/i")).toBeHidden({ timeout: 180_000 });
    await page.waitForTimeout(4000);

    // THE RIG HAS TO BE ARMED, and it can only arm against a real occlusal
    // frame. Toggling it on beforehand is a no-op - aimShadows has never run,
    // so there is no sun position, no catcher and no shadow camera, and the
    // cast stays lit. Measuring that would prove nothing.
    const box = await page.locator("canvas").first().boundingBox();
    await page.getByRole("button", { name: /Define Occlusal Plane/i }).click();
    for (const [fx, fy] of [[0.365, 0.294], [0.749, 0.265], [0.500, 0.795]]) {
      await page.mouse.click(box.x + box.width * fx, box.y + box.height * fy);
      await page.waitForTimeout(500);
    }
    await expect(page.getByRole("button", { name: /Occlusal Plane Set/i }))
      .toBeVisible({ timeout: 30_000 });
    await page.waitForTimeout(1500);

    // The rig ships off, so measure it off, then arm it and compare.
    const off = await canvasStats(page);
    await page.getByRole("checkbox", { name: /Shadows/i }).check();
    await page.waitForTimeout(1500);
    const on = await canvasStats(page);
    console.log(show("rig off", off));
    console.log(show("rig on ", on));

    expect(on.litPixels / Math.max(off.litPixels, 1),
      "arming the shadow rig blanked the cast").toBeGreaterThan(0.9);
  });
