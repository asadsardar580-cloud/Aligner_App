import { defineConfig, devices } from "@playwright/test";

/**
 * The shadow-rig harness runs under its OWN config, against the DEV server.
 *
 * WHY NOT THE MAIN CONFIG. That one tests the production build, deliberately
 * (playwright.config.js explains the reasoning). `shadow-harness.html` is a
 * diagnostic page and must never reach `dist/` — Vite's build input is
 * `index.html` alone, so it does not, and adding it as a second rollup input
 * to make the preview serve it would change the shipped bundle's chunking to
 * run a test. CLAUDE.md section 13 pins the smoke markup byte-for-byte; a
 * diagnostic is not allowed to move it.
 *
 * The dev server's 15.8s first paint, which is what pushed the main suite onto
 * the build, is a property of serving ~1,900 unbundled React modules. This page
 * imports three and one module of our own, so it does not pay that.
 *
 * Run: npx playwright test -c playwright.shadow.config.js
 */
export default defineConfig({
  testDir: "./e2e-shadow",
  timeout: 120_000,
  expect: { timeout: 20_000 },
  fullyParallel: false,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:5174",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{
    name: "chromium",
    use: {
      ...devices["Desktop Chrome"],
      // The rig is a SHADOW MAP. A software rasteriser that silently skips
      // depth-texture sampling would make every assertion here vacuous, which
      // is the failure mode this whole file exists to avoid.
      launchOptions: { args: ["--use-gl=angle", "--enable-unsafe-swiftshader"] },
    },
  }],
  webServer: {
    command: "npm run dev -- --port 5174 --strictPort",
    url: "http://localhost:5174/shadow-harness.html",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
});
