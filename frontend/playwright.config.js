import { defineConfig, devices } from "@playwright/test";

/**
 * Browser end-to-end tests.
 *
 * SCOPE, stated honestly. These cover the deterministic half of the workflow:
 * the app opens, the backend connection badge reflects reality, the workflow
 * steps are present, and the validation panel refuses to claim a check it did
 * not run. They do NOT cover upload -> segment -> cut -> move -> export,
 * because that needs a real arch scan (patient-derived, gitignored) and a
 * ~4-minute AI pass. Pretending otherwise by asserting against a stub would
 * test the stub.
 *
 * The backend is NOT started here. Some specs assert the app's behaviour when
 * the API is DOWN, which is a real state users hit and the one that motivated
 * the connection badge; others need it up and skip themselves when it is not.
 */
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  reporter: [["list"]],
  use: {
    baseURL: "http://localhost:4173",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],

  // Tested against the PRODUCTION BUILD, not the dev server, for two reasons.
  // Dev mode serves ~1,900 unbundled ES modules and first paint measured 15.8s,
  // which makes every timeout a coin flip rather than a signal. And the build
  // is what actually ships — an E2E suite that only ever exercises the dev
  // server can miss anything the bundler does differently.
  webServer: {
    command: "npm run build && npm run preview -- --port 4173 --strictPort",
    url: "http://localhost:4173",
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
  },
});
