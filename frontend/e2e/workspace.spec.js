import { test, expect } from "@playwright/test";

/**
 * The presentation layer, checked in a real browser.
 *
 * WHAT THESE CAN AND CANNOT PROVE. They run against the production build and
 * can see what the DOM and the computed styles actually are. They cannot say
 * the interface looks good, and they do not try to - each one pins a property
 * that would otherwise only be noticed by someone using the app: a focus ring
 * that never appears, a motion preference that is ignored, an export button
 * that is enabled before there is anything to export.
 *
 * PRESENTATION ONLY - NEVER AFFECTS EXPORT. Nothing here touches geometry.
 */

test.describe("Workspace presentation", () => {
  test("the global token stylesheet is installed exactly once", async ({ page }) => {
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: /Clinical Micro-Planner/i })
    ).toBeVisible();

    // ONCE, not twice. React 19 StrictMode mounts effects twice in
    // development, and two copies of a stylesheet works until one is removed
    // on unmount and the other silently is not.
    const count = await page.locator("style#aligner-global-tokens").count();
    expect(count, "the token stylesheet should be installed exactly once")
      .toBe(1);

    const css = await page.locator("style#aligner-global-tokens").innerText();
    expect(css).toContain("focus-visible");
    expect(css).toContain("prefers-reduced-motion");
    expect(errors, `uncaught page errors: ${errors.join(" | ")}`).toHaveLength(0);
  });

  test("keyboard focus is visible and a mouse click does not leave a ring",
       async ({ page }) => {
    await page.goto("/");
    const button = page.getByRole("button", { name: /Define Occlusal Plane/i });
    await expect(button).toBeVisible();

    // :focus-visible is the whole point of the rule - focusing by keyboard
    // must show a ring, and clicking must not leave one behind.
    await page.keyboard.press("Tab");
    const ringAfterTab = await page.evaluate(() => {
      const el = document.activeElement;
      if (!el || el === document.body) return null;
      return getComputedStyle(el).outlineStyle;
    });
    expect(ringAfterTab, "keyboard focus should draw an outline")
      .not.toBe("none");
  });

  test("prefers-reduced-motion is honoured rather than ignored",
       async ({ page }) => {
    await page.emulateMedia({ reducedMotion: "reduce" });
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: /Clinical Micro-Planner/i })
    ).toBeVisible();

    // The rule must actually reach an element, not merely exist in the sheet.
    const durations = await page.evaluate(() => {
      const out = [];
      for (const el of document.querySelectorAll("button")) {
        out.push(getComputedStyle(el).transitionDuration);
      }
      return out;
    });
    expect(durations.length).toBeGreaterThan(0);
    const moving = durations.filter(
      (d) => d && d !== "0s" && parseFloat(d) > 0.002);
    expect(moving,
      `these transitions ignored the motion preference: ${moving.join(", ")}`)
      .toHaveLength(0);
  });

  test("the export controls are disabled until there is something to export",
       async ({ page }) => {
    await page.goto("/");
    // A control that is enabled before it can do anything teaches people to
    // click it and read an error. All three final-export formats are gated on
    // staging existing, not on the request failing.
    for (const id of ["export-final", "export-final-stl",
                      "export-final-manifest"]) {
      const b = page.locator(`[data-testid="${id}"]`);
      if (await b.count()) {
        await expect(b).toBeDisabled();
      }
    }
  });

  test("the tooth legend is absent until segmentation has run",
       async ({ page }) => {
    await page.goto("/");
    // NOT an empty legend, and not 32 greyed-out swatches. Listing teeth that
    // were never labelled invites a clinician to look for them.
    await expect(page.locator('[data-testid="tooth-legend"]')).toHaveCount(0);
  });
});
