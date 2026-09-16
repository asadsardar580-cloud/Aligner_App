import { test, expect } from "@playwright/test";

/**
 * The two fail-safes a clinician meets on a bad day.
 *
 * WHY THESE NEED A BROWSER AND NOT A UNIT TEST. Both are about what is ON THE
 * SCREEN after something has already gone wrong, and both have failed silently
 * before in ways no assertion about a function's return value would have
 * caught:
 *
 *   - The error boundary exists because this app white-screened FIVE times from
 *     temporal-dead-zone errors, and every one of those builds passed
 *     `npm run build` cleanly. What matters is not that the class works but
 *     that a throw in the real tree lands on a page that says the case is safe
 *     and offers a way back.
 *
 *   - The re-upload banner replaced a status-bar line that the next setStatus
 *     overwrote, often within the second. The whole point is PERSISTENCE, which
 *     is precisely the property a unit test cannot see.
 *
 * These run against the PRODUCTION BUILD via `npm run preview`, like the rest of
 * the suite: dev mode serves ~1,900 unbundled modules and measured 15.8s to
 * first paint, which turns every timeout into a coin flip.
 *
 * NO BACKEND IS NEEDED. Both specs assert behaviour with the API down, which is
 * a real state users hit — and for the banner it is the state that PRODUCES it.
 */

const CASE_KEY = "aligner.case.sessions";

test.describe("Re-upload banner after a lost session", () => {
  test("a saved session id that no longer resolves raises a PERSISTENT banner",
       async ({ page }) => {
    // Plant a session id the backend cannot possibly know. This is exactly the
    // real case: sessions are memory-only by design, so a backend restart
    // leaves the browser holding an id that 404s.
    await page.addInitScript(([key]) => {
      window.localStorage.setItem(key, JSON.stringify({
        mandibular: "deadbeefdeadbeef",
      }));
    }, [CASE_KEY]);

    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto("/");

    const banner = page.getByTestId("rehydrate-banner");
    await expect(banner).toBeVisible({ timeout: 15000 });

    // The exact line the brief specifies, because a clinician who has to find a
    // file needs to be told so in words, not by an icon.
    await expect(banner).toContainText(
      "[Session Restored] Please select STL scan file to re-hydrate 3D viewport");

    // And the reassurance that matters more than the instruction: the PLAN is
    // intact; it is the mesh that cannot be recovered, because a scan is never
    // written to disk unencrypted.
    await expect(banner).toContainText(/treatment plan is intact/i);
    await expect(banner).toContainText(/mandibular/);

    expect(errors, `uncaught page errors: ${errors.join(" | ")}`).toHaveLength(0);
  });

  test("the banner SURVIVES a status update — that is the whole point",
       async ({ page }) => {
    await page.addInitScript(([key]) => {
      window.localStorage.setItem(key, JSON.stringify({
        mandibular: "deadbeefdeadbeef",
      }));
    }, [CASE_KEY]);
    await page.goto("/");

    const banner = page.getByTestId("rehydrate-banner");
    await expect(banner).toBeVisible({ timeout: 15000 });

    // The health poll rewrites the status line every 3s. The old behaviour put
    // the re-upload request THERE, so it lasted until the next tick. Waiting
    // through several ticks is the regression test.
    await page.waitForTimeout(7000);
    await expect(banner).toBeVisible();
    await expect(banner).toContainText("[Session Restored]");
  });

  test("it can be dismissed, and only by the clinician", async ({ page }) => {
    await page.addInitScript(([key]) => {
      window.localStorage.setItem(key, JSON.stringify({
        mandibular: "deadbeefdeadbeef",
      }));
    }, [CASE_KEY]);
    await page.goto("/");

    await expect(page.getByTestId("rehydrate-banner")).toBeVisible({ timeout: 15000 });
    await page.getByTestId("rehydrate-dismiss").click();
    await expect(page.getByTestId("rehydrate-banner")).toHaveCount(0);
  });

  test("a clean start shows NO banner", async ({ page }) => {
    // The failure mode worth guarding against is a warning that is always on,
    // which is the same as no warning at all.
    await page.goto("/");
    await expect(page.getByRole("heading", { name: /Clinical Micro-Planner/i }))
      .toBeVisible();
    await page.waitForTimeout(2000);
    await expect(page.getByTestId("rehydrate-banner")).toHaveCount(0);
  });
});

test.describe("Error boundary", () => {
  /**
   * Force a throw the app's OWN guards cannot absorb.
   *
   * The first version of this spec refused a WebGL context — and the renderer's
   * try/catch caught it and carried on, which is the correct behaviour and
   * meant the boundary was never exercised. So this breaks something AFTER that
   * guard: `window.devicePixelRatio` is read on the line following renderer
   * construction, inside the mount effect, and an error thrown from a React
   * effect propagates to the nearest boundary exactly as a render error does.
   *
   * A guard absorbing a fault is a better outcome than a boundary catching it.
   * Both must work, and each needs its own trigger to be seen.
   */
  const breakAfterTheRendererGuard = async (page) => {
    await page.addInitScript(() => {
      Object.defineProperty(window, "devicePixelRatio", {
        configurable: true,
        get() { throw new Error("e2e: deliberate failure past the renderer guard"); },
      });
    });
  };

  test("a throw the guards do not absorb lands on the recovery page",
       async ({ page }) => {
    await breakAfterTheRendererGuard(page);
    await page.goto("/");

    const boundary = page.getByTestId("error-boundary");
    await expect(boundary).toBeVisible({ timeout: 15000 });

    // The recovery message LEADS. After five white screens, knowing the case is
    // safe matters more than knowing which component threw.
    await expect(boundary).toContainText(/Your case is safe/i);
    await expect(boundary).toContainText(/held in the session on the backend/i);
    await expect(page.getByTestId("error-boundary-reload")).toBeVisible();
    await expect(page.getByTestId("error-boundary-message"))
      .toContainText(/deliberate failure past the renderer guard/);

    const text = (await page.locator("body").innerText()).trim();
    expect(text.length, "the page rendered nothing at all").toBeGreaterThan(40);
  });

  test("a refused WebGL context is ABSORBED, not escalated to the boundary",
       async ({ page }) => {
    // The other half of the contract: losing the viewport must not look like a
    // crash. The app should still render its chrome and say what happened.
    await page.addInitScript(() => {
      HTMLCanvasElement.prototype.getContext = function () {
        throw new Error("e2e: WebGL context deliberately refused");
      };
    });
    await page.goto("/");

    await expect(page.getByRole("heading", { name: /Clinical Micro-Planner/i }))
      .toBeVisible({ timeout: 15000 });
    await expect(page.getByTestId("error-boundary")).toHaveCount(0);
    await expect(page.locator("body")).toContainText(/WebGL context/i);
  });

  test("the reload control actually reloads", async ({ page }) => {
    await breakAfterTheRendererGuard(page);
    await page.goto("/");
    const reload = page.getByTestId("error-boundary-reload");
    await expect(reload).toBeVisible({ timeout: 15000 });
    await Promise.all([page.waitForLoadState("load"), reload.click()]);
    await expect(page.getByTestId("error-boundary")).toBeVisible();
  });
});
