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

test.describe("Application shell", () => {
  test("opens and renders the clinical workflow", async ({ page }) => {
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));

    await page.goto("/");
    await expect(page.getByRole("heading", { name: /Clinical Micro-Planner/i })).toBeVisible();

    // A white screen is the failure mode this app has actually shipped three
    // times, each from a dependency array referencing a const declared further
    // down the component. An uncaught pageerror is that bug.
    expect(errors, `uncaught page errors: ${errors.join(" | ")}`).toHaveLength(0);

    await expect(page.getByText(/Arches & Biometrics/i)).toBeVisible();
    await expect(page.getByText(/Load maxillary STL/i)).toBeVisible();
    await expect(page.getByText(/Load mandibular STL/i)).toBeVisible();
  });

  test("backend connection badge reports the real state", async ({ page, request }) => {
    const up = await backendUp(request);
    await page.goto("/");

    const badge = page.locator("text=/Backend (connected|not running|up,)/i").first();
    await expect(badge).toBeVisible({ timeout: 8000 });
    const text = (await badge.textContent()) || "";

    if (up) {
      expect(text, "backend is reachable but the badge says it is not")
        .toMatch(/Backend (connected|up,)/i);
    } else {
      // The state that motivated the badge: a backend that was simply never
      // started used to be indistinguishable from a broken app.
      expect(text).toMatch(/Backend not running/i);
      await expect(page.getByText(/start_backend\.bat/i)).toBeVisible();
    }
  });

  test("case validation never claims a check it did not run", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByText("CASE VALIDATION")).toBeVisible();

    // With nothing loaded, the antagonist row must be "not determined" — grey
    // and dashed — never a green tick. This is the single easiest way for this
    // software to mislead a clinician, so it is asserted directly.
    const antagonist = page.locator("text=/Antagonist check not available/i").first();
    await expect(antagonist).toBeVisible();

    const row = antagonist.locator("xpath=ancestor::div[1]");
    await expect(row).toContainText("—");
    await expect(row).not.toContainText("✓");

    await expect(page.getByText(/not a statement of\s+clinical safety/i)).toBeVisible();
  });

  test("a refresh with no stored case does not error", async ({ page }) => {
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("/");
    await page.evaluate(() => window.localStorage.clear());
    await page.reload();
    await expect(page.getByRole("heading", { name: /Clinical Micro-Planner/i })).toBeVisible();
    expect(errors, `uncaught errors on reload: ${errors.join(" | ")}`).toHaveLength(0);
  });

  test("a stale session id is discarded rather than left to fail every request",
    async ({ page, request }) => {
      test.skip(!(await backendUp(request)), "backend not running");

      await page.goto("/");
      // An id that certainly does not exist server-side.
      await page.evaluate(() => window.localStorage.setItem(
        "aligner.case.sessions", JSON.stringify({ mandibular: "0".repeat(32) })));
      await page.reload();

      await expect(page.getByText(/expired|Load an arch to begin/i)).toBeVisible({ timeout: 15000 });
      const left = await page.evaluate(
        () => window.localStorage.getItem("aligner.case.sessions"));
      expect(left, "a dead session id was left behind to fail every later request").toBeNull();
    });
});

test.describe("API contract", () => {
  test("hydration endpoints exist and 404 honestly", async ({ request }) => {
    test.skip(!(await backendUp(request)), "backend not running");
    const ghost = "0".repeat(32);
    for (const path of [`/api/session/${ghost}`,
                        `/api/session/${ghost}/frame`,
                        `/api/session/${ghost}/labels`,
                        `/api/session/${ghost}/teeth?geometry=true`]) {
      const r = await request.get(API + path);
      expect(r.status(), `${path} should 404 on an expired session`).toBe(404);
    }
  });

  test("ai status distinguishes loading from broken", async ({ request }) => {
    test.skip(!(await backendUp(request)), "backend not running");
    const s = await (await request.get(`${API}/api/ai/status`)).json();
    for (const k of ["loaded", "warming", "ready_message", "segmentation_in_progress"]) {
      expect(s, `/api/ai/status is missing ${k}`).toHaveProperty(k);
    }
    // A model still importing torch must never present as a dead pipeline.
    expect(s.loaded && s.warming, "cannot be loaded and warming at once").toBeFalsy();
  });
});
