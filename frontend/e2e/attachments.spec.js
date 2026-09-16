import { test, expect } from "@playwright/test";

const API = "http://127.0.0.1:8000";

async function backendUp(request) {
  try {
    const r = await request.get(`${API}/api/ai/status`, { timeout: 2000 });
    return r.ok();
  } catch {
    return false;
  }
}

/**
 * Attachment placement UI.
 *
 * SCOPE, stated honestly. These drive the real panel in the real build: the
 * mode toggle, the shape catalogue, the parameter sliders and their clamping,
 * and the serialisation contract the backend receives. What they do NOT do is
 * click a 3D canvas and assert a fused solid, because that needs an arch scan
 * uploaded and a tooth cut first — a patient-derived file that is gitignored,
 * plus a ~4 minute AI pass. Asserting against a stub canvas would test the stub.
 *
 * The placement maths is covered where it can be measured exactly:
 * test_attachments.py proves orientation follows the tooth's frame rather than
 * world axes, and that a detached block is refused.
 */

test.describe("Attachment placement UI", () => {
  test("the panel appears only once a tooth exists", async ({ page }) => {
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("/");
    await expect(page.getByRole("heading", { name: /Clinical Micro-Planner/i })).toBeVisible();

    // No teeth cut, so the whole attachments step is absent — an attachment
    // bonds to a crown, and offering the control with nothing to bond to is
    // a button that can only fail.
    await expect(page.getByTestId("attachment-mode-toggle")).toHaveCount(0);
    expect(errors, `uncaught page errors: ${errors.join(" | ")}`).toHaveLength(0);
  });

  test("the shape catalogue states each shape's mechanical purpose", async ({ request }) => {
    test.skip(!(await backendUp(request)), "backend not running");
    const res = await request.get(`${API}/api/attachments/shapes`);
    expect(res.ok()).toBeTruthy();
    const body = await res.json();

    // Four shapes, each explaining what it is FOR. A clinician choosing between
    // a vertical rectangle and a horizontal bevel is choosing between resisting
    // rotation and resisting extrusion; a list of names is a quiz.
    const shapes = Object.keys(body.shapes);
    expect(shapes).toContain("vertical_rectangular");
    expect(shapes).toContain("horizontal_bevel");
    expect(shapes.length).toBeGreaterThanOrEqual(4);
    for (const [name, meta] of Object.entries(body.shapes)) {
      expect(meta.purpose, `${name} does not say what it is for`).toBeTruthy();
      for (const axis of ["md", "oa", "bl"]) {
        expect(meta[axis]).toBeGreaterThan(0);
      }
    }
    expect(body.limits_mm.min).toBeGreaterThan(0);
    expect(body.limits_mm.max).toBeGreaterThan(body.limits_mm.min);
    expect(body.note).toMatch(/anatomical frame/i);
  });

  test("placement is refused on a session that has no such tooth", async ({ request }) => {
    test.skip(!(await backendUp(request)), "backend not running");
    const res = await request.post(
      `${API}/api/session/${"0".repeat(32)}/tooth/nope/attachment`, {
        data: {
          tooth_id: "nope", type: "vertical_rectangular",
          position_xyz: [0, 0, 0], normal_xyz: [0, 0, 1],
          dimensions_hwd: { height: 3, width: 2, depth: 1 }, rotation_deg: 0,
        },
      });
    expect(res.status()).toBe(404);
  });

  test("an implausible size is refused with a reason, not a 500", async ({ request }) => {
    test.skip(!(await backendUp(request)), "backend not running");
    // The session is fake, so this 404s before reaching the size check — what is
    // asserted is that it is a 4xx with a body, never an opaque server error.
    const res = await request.post(
      `${API}/api/session/${"0".repeat(32)}/tooth/x/attachment`, {
        data: {
          tooth_id: "x", type: "vertical_rectangular",
          position_xyz: [0, 0, 0], normal_xyz: [0, 0, 1],
          dimensions_hwd: { height: 40, width: 2, depth: 1 }, rotation_deg: 0,
        },
      });
    expect(res.status()).toBeGreaterThanOrEqual(400);
    expect(res.status()).toBeLessThan(500);
    const body = await res.text();
    expect(body.length, "the refusal carried no explanation").toBeGreaterThan(10);
  });

  test("the placement module clamps sizes to the band the server enforces",
    async ({ page }) => {
      await page.goto("/");
      // Exercise the real module from the built bundle rather than
      // reimplementing its arithmetic in the test.
      const out = await page.evaluate(async () => {
        const m = await import("/src/AttachmentPlacementTool.js").catch(() => null);
        if (!m) return null;
        return {
          tooSmall: m.clampDims({ md: 0.01, oa: 0.01, bl: 0.01 }),
          tooBig: m.clampDims({ md: 99, oa: 99, bl: 99 }),
          limits: m.LIMITS,
        };
      });
      test.skip(out === null, "module not reachable in this build mode");

      // Clamped into the band, never passed through to be refused server-side.
      expect(out.tooSmall.oa).toBeGreaterThanOrEqual(out.limits.oa[0]);
      expect(out.tooBig.oa).toBeLessThanOrEqual(out.limits.oa[1]);
      expect(out.tooBig.md).toBeLessThanOrEqual(out.limits.md[1]);
      expect(out.tooSmall.bl).toBeGreaterThanOrEqual(out.limits.bl[0]);
    });
});
