/**
 * Design tokens for the clinical workspace.
 *
 * WHY A TOKEN MODULE RATHER THAN MORE HEX STRINGS. `App.jsx` carries its
 * styles in one `S` object plus inline overrides at roughly sixty call sites,
 * and the same six or seven colours are typed out at each of them - `#2c313a`
 * alone appears more than twenty times. That is survivable until something
 * needs to change everywhere at once, at which point it is a find-and-replace
 * over a 2,700-line file with no way to tell a border from a divider from a
 * disabled outline. These are the names; the values live here once.
 *
 * SCOPE, STATED HONESTLY. This module is the base for the workspace redesign,
 * not the redesign. `App.jsx` has NOT been restructured into a top bar, a
 * workflow rail and a context panel, and its existing `S` object has not been
 * migrated onto these tokens wholesale. What uses them today is the tooth
 * legend and the focus and motion rules below. See CLAUDE.md section 24.8.
 *
 * PRESENTATION ONLY - NEVER AFFECTS EXPORT. Nothing in this file is read by
 * any geometry, staging or manufacturing path. It is the same architectural
 * invariant the shadow rig carries (CLAUDE.md section 22): a viewport effect
 * may change what a clinician sees and may never change what a lab prints.
 */

/** An 8px base. Half-steps exist because 4px is the smallest gap that still
 *  reads as a gap at this density; nothing smaller is a token. */
export const space = Object.freeze({
  xs: 4, sm: 8, md: 16, lg: 24, xl: 32, xxl: 48,
});

export const radius = Object.freeze({
  sm: 4, md: 6, lg: 10, pill: 999,
});

export const color = Object.freeze({
  /** The viewport clear colour. Tooth colours are checked against this in
   *  verify-palette.mjs, because two colours can be far apart from each other
   *  and both invisible. */
  viewport: "#0d0f12",
  surface: "#1a1d22",
  surfaceRaised: "#21252b",
  surfaceSunken: "#101215",
  border: "#2c313a",
  borderStrong: "#3a414c",
  text: "#e7ebee",
  textMuted: "#8b93a0",
  textFaint: "#5a616b",

  /** ONE accent. A second one is how an interface stops having a hierarchy. */
  accent: "#3fc6d4",
  accentQuiet: "#2a8b96",
  /** Selection, shared with the 3D viewport so the two read as one idea. */
  selection: "#3fa9ff",

  ok: "#3cb44b",
  okSurface: "#12301a",
  okBorder: "#2f7d32",
  okText: "#9ae6a4",

  /** Amber is "the software needs something from you". Red is "something is
   *  wrong". The distinction is load-bearing: a red chip beside a tooth reads
   *  as a clinical finding about the patient. */
  warn: "#ffa53c",
  warnSurface: "#2a2113",
  warnBorder: "#5a4620",
  warnText: "#ffd9a0",

  danger: "#b3261e",
  dangerSurface: "#2e1416",
  dangerText: "#ffb4ab",
});

export const type = Object.freeze({
  ui: "system-ui, -apple-system, 'Segoe UI', sans-serif",
  /** Measurements only. Proportional digits jitter as a value changes, which
   *  on a live readout looks like the number is unstable when it is not. */
  numeric: "'SF Mono', 'Cascadia Mono', Consolas, monospace",
  size: { xs: 10.5, sm: 11, base: 12, md: 13, lg: 15 },
  weight: { normal: 400, medium: 600, bold: 700 },
});

/** Motion. Every duration passes through `duration()` so one media query
 *  turns the whole interface still. */
export const motion = Object.freeze({
  fast: 120, base: 180, slow: 280,
  ease: "cubic-bezier(0.2, 0.0, 0.2, 1)",
});

/**
 * Does this person want less motion?
 *
 * READ AT CALL TIME, NOT CACHED AT MODULE LOAD. The setting can change while
 * the app is open, and a value captured once at import would ignore that for
 * the life of the session. Guarded for the SSR smoke build, which has no
 * `window` and must not throw.
 */
export function prefersReducedMotion() {
  try {
    return typeof window !== "undefined" && window.matchMedia
      ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
      : false;
  } catch {
    return false;
  }
}

/** A duration in ms, or 0 when reduced motion is asked for. */
export function duration(ms) {
  return prefersReducedMotion() ? 0 : ms;
}

/** A CSS transition string honouring the same setting. */
export function transition(props = "all", ms = motion.base) {
  const d = duration(ms);
  return d === 0 ? "none" : `${props} ${d}ms ${motion.ease}`;
}

/**
 * The focus ring.
 *
 * `:focus-visible`, not `:focus`, so a mouse click does not leave a ring
 * behind while keyboard navigation still shows one. Applied through the
 * stylesheet below rather than per element, because a focus style that has to
 * be remembered at every call site is one that will be missing somewhere.
 */
export const GLOBAL_CSS = `
:root {
  --accent: ${color.accent};
  --selection: ${color.selection};
  --border: ${color.border};
  --surface: ${color.surface};
}
*:focus { outline: none; }
*:focus-visible {
  outline: 2px solid ${color.selection};
  outline-offset: 2px;
  border-radius: ${radius.sm}px;
}
button:disabled, input:disabled, [aria-disabled="true"] {
  opacity: 0.4;
  cursor: not-allowed;
}
button:not(:disabled):hover { filter: brightness(1.14); }
button:not(:disabled):active { filter: brightness(0.92); }
button { transition: filter ${motion.fast}ms ${motion.ease}; }

/* PRESENTATION ONLY. Turning motion off must never turn a control off. */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}
`;

/**
 * Install the global rules once.
 *
 * Idempotent by id: React 19 StrictMode mounts effects twice in development,
 * and two copies of a stylesheet is the kind of thing that works until one of
 * them is removed on unmount and the other silently is not.
 */
export function installGlobalStyles(doc) {
  const d = doc || (typeof document !== "undefined" ? document : null);
  if (!d) return false;                       // SSR smoke build
  const id = "aligner-global-tokens";
  if (d.getElementById(id)) return false;
  const el = d.createElement("style");
  el.id = id;
  el.textContent = GLOBAL_CSS;
  d.head.appendChild(el);
  return true;
}
